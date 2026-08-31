"""
loss.py
Loss functions for the Grounding MOMENT primitive + event detector.

Primitive Hungarian matching (HungarianMatcher):
  Np query slots vs M flat GT primitives.

  GT primitives assembled per sample by flattening:
    (k in [0, num_events)) × (i in [0, num_phrases)) → primitive_spans[k, i]
    filtering zero-width (padded) entries.

  Loss components per matched pair:
    A. Span L1 + DIoU        (matched query slots)
    B. Contrastive focal cls (all Np slots; matched → phrase_token_masks[i])
    C. Exclusion loss        (unmatched slots vs all GT primitive spans)

Event Hungarian matching (EventHungarianMatcher):
  Ne query slots vs M_e GT event spans (filtered width > 0).
  Cost = cost_l1 * L1 + cost_iou * (1 - IoU).   No cls cost.

  Loss components:
    A. Span L1 + DIoU        (matched event query slots)
    B. Optional exclusion    (unmatched event slots vs GT event spans)

Event-primitive correspondence head (EventPrimCorrHead, optional):
  Supervised when config.use_event_prim_corr_head is True.
  Positive map (Ne, Np): event query e × primitive query p is positive when
    - e was matched to some GT event k, AND
    - p was matched to a GT primitive that belongs to event occurrence k.
  Loss gated by λ₄(t): 0 while current_epoch < event_cls_warmup_epochs, then
  event_prim_corr_cls_loss_weight afterwards.

Auxiliary decoder-layer outputs are supervised with the same losses.
Encoder intermediate outputs are supervised with the primitive loss only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .config import GroundingMomentConfig


# ---------------------------------------------------------------------------
# 1D temporal span utilities
# ---------------------------------------------------------------------------

def temporal_iou(pred_spans: torch.Tensor, gt_spans: torch.Tensor) -> torch.Tensor:
    """
    Pairwise 1D IoU.
    pred_spans: (N, 2)  (start, width)
    gt_spans:   (M, 2)
    Returns:    (N, M)
    """
    p_s = pred_spans[:, 0];  p_e = pred_spans[:, 0] + pred_spans[:, 1]
    g_s = gt_spans[:, 0];    g_e = gt_spans[:, 0]   + gt_spans[:, 1]

    inter_s = torch.max(p_s.unsqueeze(1), g_s.unsqueeze(0))
    inter_e = torch.min(p_e.unsqueeze(1), g_e.unsqueeze(0))
    inter   = (inter_e - inter_s).clamp(min=0)

    p_len = (p_e - p_s).clamp(min=0).unsqueeze(1)
    g_len = (g_e - g_s).clamp(min=0).unsqueeze(0)
    union = p_len + g_len - inter
    return inter / (union + 1e-6)


def temporal_diou(pred_spans: torch.Tensor, gt_spans: torch.Tensor) -> torch.Tensor:
    """
    Pairwise 1D DIoU.
    pred_spans: (N, 2)  (start, width)
    gt_spans:   (M, 2)
    Returns:    (N, M)

    DIoU = IoU - ρ²(center_pred, center_gt) / c²
    where c is the length of the smallest enclosing interval.
    The centre-distance penalty keeps gradients flowing when predictions are
    outside or only partially overlapping the GT span, where GIoU saturates.
    """
    p_s = pred_spans[:, 0];  p_e = pred_spans[:, 0] + pred_spans[:, 1]
    g_s = gt_spans[:, 0];    g_e = gt_spans[:, 0]   + gt_spans[:, 1]

    inter_s = torch.max(p_s.unsqueeze(1), g_s.unsqueeze(0))
    inter_e = torch.min(p_e.unsqueeze(1), g_e.unsqueeze(0))
    inter   = (inter_e - inter_s).clamp(min=0)

    p_len = (p_e - p_s).clamp(min=0).unsqueeze(1)
    g_len = (g_e - g_s).clamp(min=0).unsqueeze(0)
    union = p_len + g_len - inter
    iou   = inter / (union + 1e-6)

    p_c = ((p_s + p_e) / 2).unsqueeze(1)           # (N, 1) predicted centres
    g_c = ((g_s + g_e) / 2).unsqueeze(0)           # (1, M) GT centres
    d2  = (p_c - g_c) ** 2                          # (N, M) squared centre distance

    enc_s = torch.min(p_s.unsqueeze(1), g_s.unsqueeze(0))
    enc_e = torch.max(p_e.unsqueeze(1), g_e.unsqueeze(0))
    c2    = (enc_e - enc_s).clamp(min=1e-6) ** 2   # (N, M) squared enclosing length

    return iou - d2 / c2


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def contrastive_focal_loss(
    logits: torch.Tensor,        # (N, T)
    positive_map: torch.Tensor,  # (N, T)
    alpha: float = 0.25,
    gamma: float = 2.0,
    n_pos: int | None = None,
) -> torch.Tensor:
    prob   = logits.sigmoid()
    target = positive_map.float()
    ce     = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t    = prob * target + (1 - prob) * (1 - target)
    fw     = alpha * target + (1 - alpha) * (1 - target)
    fw     = fw * (1 - p_t) ** gamma
    loss   = (fw * ce).sum()
    if n_pos is not None:
        denom = n_pos if n_pos > 0 else logits.numel()
    else:
        denom = max(int(positive_map.sum().item()), 1)
    return loss / denom


# ---------------------------------------------------------------------------
# Exclusion loss helper
# ---------------------------------------------------------------------------

def _excl_loss(
    pred_spans:  torch.Tensor,   # (Np, 2)
    gt_spans:    torch.Tensor,   # (M, 2)
    matched_idx: torch.Tensor,   # matched prediction indices (may be empty)
) -> torch.Tensor:
    """
    GT primitives are non-overlapping by construction, so any unmatched
    query slot whose IoU with any GT primitive span is > 0 is wrong.

    Returns the mean over unmatched slots of their max-IoU against the set of
    GT spans. Returns a zero tensor with grad when there are no unmatched
    slots or no GT spans.
    """
    Np  = pred_spans.shape[0]
    M   = gt_spans.shape[0]
    if M == 0 or Np == 0:
        return pred_spans.sum() * 0.0

    matched_set = set(matched_idx.tolist())
    unmatched   = [i for i in range(Np) if i not in matched_set]
    if not unmatched:
        return pred_spans.sum() * 0.0

    unmatched_idx   = torch.tensor(unmatched, dtype=torch.long, device=pred_spans.device)
    unmatched_spans = pred_spans[unmatched_idx]            # (U, 2)
    cross_iou       = temporal_iou(unmatched_spans, gt_spans)  # (U, M)
    return cross_iou.max(dim=1).values.mean()


# ---------------------------------------------------------------------------
# Flat Hungarian matchers
# ---------------------------------------------------------------------------

class EventHungarianMatcher(nn.Module):
    """
    Bipartite matching between Ne event query slot predictions and M_e GT event spans.
    Cost = cost_l1 * L1 + cost_iou * (1 - IoU).   No classification cost.
    """

    def __init__(self, cost_l1: float = 10.0, cost_iou: float = 2.0):
        super().__init__()
        self.cost_l1  = cost_l1
        self.cost_iou = cost_iou

    @torch.no_grad()
    def forward(
        self,
        pred_spans: torch.Tensor,  # (Ne, 2)
        gt_spans:   torch.Tensor,  # (M_e, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Ne  = pred_spans.shape[0]
        M_e = gt_spans.shape[0]
        if M_e == 0:
            return (torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long))

        l1_cost  = torch.cdist(pred_spans.float(), gt_spans.float(), p=1)  # (Ne, M_e)
        iou_cost = -temporal_iou(pred_spans, gt_spans)                      # (Ne, M_e)

        cost_matrix = self.cost_l1 * l1_cost + self.cost_iou * iou_cost
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        return (torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx,   dtype=torch.long))


class HungarianMatcher(nn.Module):
    """
    Bipartite matching between Np query slot predictions and M flat GT
    primitives.

    Cost = w_cls * cls_cost + w_l1 * l1_cost + w_iou * iou_cost
    """

    def __init__(self, cost_cls: float = 1.0, cost_l1: float = 5.0,
                 cost_iou: float = 2.0):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_l1  = cost_l1
        self.cost_iou = cost_iou

    @torch.no_grad()
    def forward(
        self,
        pred_spans:      torch.Tensor,  # (Np, 2)
        pred_logits:     torch.Tensor,  # (Np, T)
        gt_spans:        torch.Tensor,  # (M, 2)
        phrase_pos_maps: torch.Tensor,  # (M, T)  per-primitive phrase positive maps
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Np = pred_spans.shape[0]
        M  = gt_spans.shape[0]
        if M == 0:
            return (torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long))

        prob = pred_logits.sigmoid()

        # cls_cost[p, m] = -mean(prob[p] * phrase_pos_maps[m]) / pos_count_m
        pos_counts = phrase_pos_maps.sum(dim=-1).clamp(min=1)   # (M,)
        dot        = torch.mm(prob, phrase_pos_maps.t())         # (Np, M)
        cls_cost   = -(dot / pos_counts.unsqueeze(0))            # (Np, M)

        l1_cost  = torch.cdist(pred_spans.float(), gt_spans.float(), p=1)  # (Np, M)
        iou_cost = -temporal_iou(pred_spans, gt_spans)                      # (Np, M)

        cost_matrix = (self.cost_cls * cls_cost
                       + self.cost_l1  * l1_cost
                       + self.cost_iou * iou_cost)

        pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        return (torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx,   dtype=torch.long))


# ---------------------------------------------------------------------------
# Helper: build flat GT primitives for one sample
# ---------------------------------------------------------------------------

def _build_flat_gt(
    t: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten (K events × num_phrases) → (M, 2) gt_spans and (M, T) phrase_pos_maps.
    Filters zero-width (padded) primitives.

    Also returns gt_prim_event_idx (M,) — for each flat GT primitive, the index
    k of the event occurrence it belongs to. Used by the event-primitive
    correspondence head to build the positive (Ne, Np) map.

    Returns (gt_spans, phrase_pos_maps, gt_prim_event_idx) all on `device`.
    Empty → (0,2), (0,T), (0,) tensors.
    """
    T = t["phrase_token_masks"].shape[-1]

    if not t.get("has_events", False) or t.get("num_events", 0) == 0:
        return (torch.zeros(0, 2, device=device),
                torch.zeros(0, T, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    num_events  = int(t["num_events"])
    num_phrases = int(t.get("num_phrases", 0))
    if num_phrases == 0:
        return (torch.zeros(0, 2, device=device),
                torch.zeros(0, T, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    prim_spans   = t["primitive_spans"].to(device)    # (K, P_data, 2)
    phrase_masks = t["phrase_token_masks"].to(device)  # (P_data, T)

    spans_list      = []
    maps_list       = []
    event_idx_list  = []
    for k in range(num_events):
        for i in range(min(num_phrases, prim_spans.shape[1])):
            s, w = prim_spans[k, i, 0], prim_spans[k, i, 1]
            if w.item() > 0:
                spans_list.append(prim_spans[k, i])
                if i < phrase_masks.shape[0]:
                    maps_list.append(phrase_masks[i].float())
                else:
                    maps_list.append(torch.zeros(T, device=device))
                event_idx_list.append(k)

    if not spans_list:
        return (torch.zeros(0, 2, device=device),
                torch.zeros(0, T, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    return (
        torch.stack(spans_list, dim=0),
        torch.stack(maps_list,  dim=0),
        torch.tensor(event_idx_list, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Helper: build flat GT event spans for one sample
# ---------------------------------------------------------------------------

def _build_event_gt(t: dict, device: torch.device) -> torch.Tensor:
    """
    Return (M_e, 2) GT event spans (start, width) filtered to width > 0.

    t["event_spans"] is (K, 2) with K = num_events (may include zero-width padding).
    """
    if not t.get("has_events", False) or t.get("num_events", 0) == 0:
        return torch.zeros(0, 2, device=device)

    num_events = int(t["num_events"])
    ev_spans   = t["event_spans"].to(device)        # (K, 2)
    ev_spans   = ev_spans[:num_events]              # slice to valid events
    valid      = ev_spans[:, 1] > 0                 # width > 0
    return ev_spans[valid]


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------

class GroundingMomentLoss(nn.Module):
    def __init__(self, config: GroundingMomentConfig):
        super().__init__()
        self.config  = config
        self.matcher = HungarianMatcher(
            cost_cls=config.matcher_cost_cls,
            cost_l1 =config.matcher_cost_l1,
            cost_iou=config.matcher_cost_iou,
        )
        self.event_matcher = EventHungarianMatcher(
            cost_l1 =config.matcher_event_cost_l1,
            cost_iou=config.matcher_event_cost_iou,
        )

    # ------------------------------------------------------------------
    # Core loss for one set of decoder outputs (primitives)
    # ------------------------------------------------------------------
    def _loss_for_output(
        self,
        pred_spans:  torch.Tensor,   # (B, Np, 2)
        pred_logits: torch.Tensor,   # (B, Np, T)
        targets: list[dict],
        suffix: str = "",
        cached_prim_assignments: list[tuple] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[tuple]]:
        """
        Returns (loss_dict, prim_assignments).

        prim_assignments[b] = (pred_idx, gt_idx, gt_prim_event_idx_for_gt)
        where gt_prim_event_idx_for_gt maps flat GT primitive index → GT event k.

        When cached_prim_assignments is provided, Hungarian matching is skipped
        and the cached assignments are reused (used for aux layers and the
        correspondence head to avoid redundant solves).
        """
        B  = pred_spans.shape[0]
        Np = pred_spans.shape[1]
        T  = pred_logits.shape[2]
        device = pred_spans.device

        loss_cls  = torch.tensor(0.0, device=device)
        loss_l1   = torch.tensor(0.0, device=device)
        loss_iou  = torch.tensor(0.0, device=device)
        loss_excl = torch.tensor(0.0, device=device)

        n_samples         = 0
        n_matched_total   = 0
        n_excl_groups     = 0

        assignments: list[tuple] = []

        for b in range(B):
            t = targets[b]
            gt_spans_b, phrase_maps_b, gt_prim_ev_b = _build_flat_gt(t, device)
            M = gt_spans_b.shape[0]

            cls_targets = torch.zeros(Np, T, device=device)

            if cached_prim_assignments is not None:
                pred_idx, gt_idx = cached_prim_assignments[b][:2]
            else:
                pred_idx = gt_idx = torch.zeros(0, dtype=torch.long)

            if M > 0:
                if cached_prim_assignments is None:
                    pred_idx, gt_idx = self.matcher(
                        pred_spans[b],
                        pred_logits[b],
                        gt_spans_b,
                        phrase_maps_b,
                    )

                if len(pred_idx) > 0:
                    matched_pred = pred_spans[b][pred_idx]    # (Mp, 2)
                    matched_gt   = gt_spans_b[gt_idx]         # (Mp, 2)

                    loss_l1  = loss_l1  + F.l1_loss(matched_pred, matched_gt, reduction="sum")
                    diou     = temporal_diou(matched_pred, matched_gt)
                    loss_iou = loss_iou + (1 - diou.diag()).sum()

                    cls_targets[pred_idx] = phrase_maps_b[gt_idx]
                    n_matched_total += len(pred_idx)

                if self.config.prim_excl_loss_weight > 0:
                    pi_for_excl = pred_idx if len(pred_idx) > 0 else torch.zeros(
                        0, dtype=torch.long, device=device
                    )
                    loss_excl = loss_excl + _excl_loss(
                        pred_spans[b], gt_spans_b, pi_for_excl
                    )
                    n_excl_groups += 1

            n_pos = int(cls_targets.sum().item())
            loss_cls = loss_cls + contrastive_focal_loss(
                pred_logits[b], cls_targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                n_pos=n_pos,
            )
            n_samples += 1
            assignments.append((pred_idx, gt_idx, gt_prim_ev_b))

        if n_samples > 0:
            loss_cls = loss_cls / n_samples
        if n_matched_total > 0:
            loss_l1  = loss_l1  / (n_matched_total * 2)
            loss_iou = loss_iou / n_matched_total
        if n_excl_groups > 0:
            loss_excl = loss_excl / n_excl_groups

        w = self.config
        total = (w.cls_loss_weight  * loss_cls
                 + w.l1_loss_weight * loss_l1
                 + w.iou_loss_weight * loss_iou
                 + w.prim_excl_loss_weight * loss_excl)

        mean_diou = (1.0 - loss_iou).clamp(min=-1.0)
        loss_dict = {
            f"loss{suffix}":      total,
            f"loss_cls{suffix}":  loss_cls,
            f"loss_l1{suffix}":   loss_l1,
            f"loss_diou{suffix}": loss_iou,
            f"mean_diou{suffix}": mean_diou,
            f"loss_excl{suffix}": loss_excl,
        }
        return loss_dict, assignments

    # ------------------------------------------------------------------
    # Event loss for one set of decoder outputs
    # ------------------------------------------------------------------
    def _loss_for_event_output(
        self,
        event_pred_spans: torch.Tensor,   # (B, Ne, 2)
        targets: list[dict],
        suffix: str = "_event",
        cached_event_assignments: list[tuple] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[tuple]]:
        """
        Returns (loss_dict, event_assignments).
        event_assignments[b] = (pred_idx, gt_idx).
        """
        B  = event_pred_spans.shape[0]
        Ne = event_pred_spans.shape[1]
        device = event_pred_spans.device

        loss_l1   = torch.tensor(0.0, device=device)
        loss_iou  = torch.tensor(0.0, device=device)
        loss_excl = torch.tensor(0.0, device=device)

        n_matched_total = 0
        n_excl_groups   = 0

        assignments: list[tuple] = []

        for b in range(B):
            t = targets[b]
            gt_ev_b = _build_event_gt(t, device)   # (M_e, 2)
            M_e = gt_ev_b.shape[0]

            if cached_event_assignments is not None:
                pred_idx, gt_idx = cached_event_assignments[b]
            else:
                pred_idx = gt_idx = torch.zeros(0, dtype=torch.long)

            if M_e > 0:
                if cached_event_assignments is None:
                    pred_idx, gt_idx = self.event_matcher(
                        event_pred_spans[b], gt_ev_b
                    )

                if len(pred_idx) > 0:
                    matched_pred = event_pred_spans[b][pred_idx]  # (Mp, 2)
                    matched_gt   = gt_ev_b[gt_idx]                 # (Mp, 2)

                    loss_l1  = loss_l1  + F.l1_loss(matched_pred, matched_gt, reduction="sum")
                    diou     = temporal_diou(matched_pred, matched_gt)
                    loss_iou = loss_iou + (1 - diou.diag()).sum()
                    n_matched_total += len(pred_idx)

                if self.config.event_excl_loss_weight > 0:
                    pi_for_excl = pred_idx if len(pred_idx) > 0 else torch.zeros(
                        0, dtype=torch.long, device=device
                    )
                    loss_excl = loss_excl + _excl_loss(
                        event_pred_spans[b], gt_ev_b, pi_for_excl
                    )
                    n_excl_groups += 1

            assignments.append((pred_idx, gt_idx))

        if n_matched_total > 0:
            loss_l1  = loss_l1  / (n_matched_total * 2)
            loss_iou = loss_iou / n_matched_total
        if n_excl_groups > 0:
            loss_excl = loss_excl / n_excl_groups

        w = self.config
        total = (w.event_l1_loss_weight  * loss_l1
                 + w.event_iou_loss_weight * loss_iou
                 + w.event_excl_loss_weight * loss_excl)

        mean_diou = (1.0 - loss_iou).clamp(min=-1.0)
        loss_dict = {
            f"loss{suffix}":       total,
            f"loss{suffix}_l1":    loss_l1,
            f"loss{suffix}_diou":  loss_iou,
            f"mean_diou{suffix}":  mean_diou,
            f"loss{suffix}_excl":  loss_excl,
        }
        return loss_dict, assignments

    # ------------------------------------------------------------------
    # Event-primitive correspondence head loss
    # ------------------------------------------------------------------
    def _event_corr_loss_for_output(
        self,
        event_prim_logits:    torch.Tensor,  # (B, Ne, Np)
        prim_assignments:     list[tuple],   # per-sample (prim_pred_idx, prim_gt_idx, gt_prim_ev_idx)
        event_assignments:    list[tuple],   # per-sample (event_pred_idx, event_gt_idx)
        suffix: str = "_event_corr",
    ) -> dict[str, torch.Tensor]:
        """
        Focal loss on the (Ne, Np) positive map built from the Hungarian
        assignments of both the primitive and event heads.

        For batch item b:
          - matched event query e → GT event k_e
          - matched prim  query p → GT prim  m  →  GT event k_m = gt_prim_ev_idx[prim_gt_idx[p]]
          - positive_map[e, p] = 1  iff  k_e == k_m
        """
        B  = event_prim_logits.shape[0]
        Ne = event_prim_logits.shape[1]
        Np = event_prim_logits.shape[2]
        device = event_prim_logits.device

        loss_corr = torch.tensor(0.0, device=device)
        n_samples = 0

        for b in range(B):
            prim_pred_idx, prim_gt_idx, gt_prim_ev_idx = prim_assignments[b]
            ev_pred_idx,   ev_gt_idx                   = event_assignments[b]

            # Move all assignment tensors to the target device.
            # Hungarian matching returns CPU tensors (via scipy); gt_prim_ev_idx
            # is built on device.  All must agree before indexing.
            prim_pred_idx   = prim_pred_idx.to(device)
            prim_gt_idx     = prim_gt_idx.to(device)
            gt_prim_ev_idx  = gt_prim_ev_idx.to(device)
            ev_pred_idx     = ev_pred_idx.to(device)
            ev_gt_idx       = ev_gt_idx.to(device)

            # Build positive map (Ne, Np)
            pos_map = torch.zeros(Ne, Np, device=device)

            if len(prim_pred_idx) > 0 and len(ev_pred_idx) > 0:
                # For each matched primitive query p, determine its GT event k_m
                # prim_gt_idx[i] is the flat GT primitive index m → gt_prim_ev_idx[m] = k_m
                prim_pred_ev = gt_prim_ev_idx[prim_gt_idx]  # (Mp,) event indices for each matched prim

                # For each matched event query e → GT event k_e
                for ei, k_e in zip(ev_pred_idx.tolist(), ev_gt_idx.tolist()):
                    # Find which matched primitive queries belong to the same GT event k_e
                    same_event_mask = (prim_pred_ev == k_e)  # (Mp,) bool
                    if same_event_mask.any():
                        matching_prim_slots = prim_pred_idx[same_event_mask]
                        pos_map[ei, matching_prim_slots] = 1.0

            n_pos = int(pos_map.sum().item())
            loss_corr = loss_corr + contrastive_focal_loss(
                event_prim_logits[b],
                pos_map,
                alpha=self.config.event_prim_corr_focal_alpha,
                gamma=self.config.event_prim_corr_focal_gamma,
                n_pos=n_pos,
            )
            n_samples += 1

        if n_samples > 0:
            loss_corr = loss_corr / n_samples

        return {f"loss{suffix}": loss_corr}

    # ------------------------------------------------------------------
    # DN direct-supervision loss (flat, no Hungarian)
    # ------------------------------------------------------------------
    def _dn_loss_for_output(
        self,
        pred_spans:  torch.Tensor,   # (B, N_dn, 2)
        pred_logits: torch.Tensor,   # (B, N_dn, T)
        dn_meta: dict,
        suffix: str = "_dn",
    ) -> dict[str, torch.Tensor]:
        B      = pred_spans.shape[0]
        device = pred_spans.device

        loss_cls = torch.tensor(0.0, device=device)
        loss_l1  = torch.tensor(0.0, device=device)
        loss_iou = torch.tensor(0.0, device=device)

        n_valid_total    = 0
        n_valid_pos_pairs = 0

        valid_mask  = dn_meta["valid_mask"]    # (B, N_dn) bool
        is_positive = dn_meta["is_positive"]   # (B, N_dn) bool
        gt_spans    = dn_meta["gt_spans"]      # (B, N_dn, 2)
        gt_cls_map  = dn_meta["gt_cls_map"]    # (B, N_dn, T)

        for b in range(B):
            vm  = valid_mask[b]
            isp = is_positive[b]

            if not vm.any():
                continue

            valid_idx = vm.nonzero(as_tuple=False).view(-1)
            pos_mask  = vm & isp
            pos_idx   = pos_mask.nonzero(as_tuple=False).view(-1)

            cls_targets = gt_cls_map[b][valid_idx]
            cls_logits  = pred_logits[b][valid_idx]
            n_pos_toks  = int(cls_targets.sum().item())
            loss_cls = loss_cls + contrastive_focal_loss(
                cls_logits, cls_targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                n_pos=n_pos_toks,
            )
            n_valid_total += 1

            if len(pos_idx) > 0:
                n_pos_pairs = len(pos_idx)
                loss_l1  = loss_l1  + F.l1_loss(
                    pred_spans[b][pos_idx], gt_spans[b][pos_idx], reduction="sum"
                )
                diou = temporal_diou(pred_spans[b][pos_idx], gt_spans[b][pos_idx])
                loss_iou = loss_iou + (1 - diou.diag()).sum()
                n_valid_pos_pairs += n_pos_pairs

        if n_valid_total > 0:
            loss_cls = loss_cls / n_valid_total
        if n_valid_pos_pairs > 0:
            loss_l1  = loss_l1  / (n_valid_pos_pairs * 2)
            loss_iou = loss_iou / n_valid_pos_pairs

        w = self.config
        total = (w.cls_loss_weight  * loss_cls
                 + w.l1_loss_weight * loss_l1
                 + w.iou_loss_weight * loss_iou)

        return {
            f"loss{suffix}":      total,
            f"loss{suffix}_cls":  loss_cls,
            f"loss{suffix}_l1":   loss_l1,
            f"loss{suffix}_diou": loss_iou,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: dict,
        targets: list[dict],
        current_epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # λ₄(t) gating for the event-primitive correspondence head loss
        # ------------------------------------------------------------------
        use_corr_head = (
            self.config.use_event_prim_corr_head
            and "event_prim_pred_logits" in outputs
        )
        if use_corr_head:
            lam4 = (
                0.0
                if (current_epoch is not None
                    and current_epoch < self.config.event_cls_warmup_epochs)
                else 1.0
            )
        else:
            lam4 = 0.0

        # ------------------------------------------------------------------
        # Last decoder layer — primitive loss (also caches assignments)
        # ------------------------------------------------------------------
        prim_losses, prim_assignments = self._loss_for_output(
            outputs["prim_pred_spans"],
            outputs["prim_pred_logits"],
            targets,
        )

        # ------------------------------------------------------------------
        # Last decoder layer — event loss (also caches assignments)
        # ------------------------------------------------------------------
        ev_losses, event_assignments = self._loss_for_event_output(
            outputs["event_pred_spans"], targets,
        )

        losses = prim_losses
        losses["loss"] = losses["loss"] + ev_losses.pop("loss_event")
        losses.update(ev_losses)

        # ------------------------------------------------------------------
        # Last decoder layer — event-primitive correspondence loss
        # ------------------------------------------------------------------
        if use_corr_head:
            corr_losses = self._event_corr_loss_for_output(
                outputs["event_prim_pred_logits"],
                prim_assignments,
                event_assignments,
            )
            raw_corr_loss = corr_losses["loss_event_corr"]
            losses["loss"] = losses["loss"] + (
                lam4 * self.config.event_prim_corr_cls_loss_weight * raw_corr_loss
            )
            losses["loss_event_corr"]      = raw_corr_loss
            losses["event_cls_lambda4"]    = torch.tensor(lam4, device=losses["loss"].device)

        # ------------------------------------------------------------------
        # Decoder auxiliary losses (prim + event per layer)
        # ------------------------------------------------------------------
        if self.config.dec_aux_loss and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_prim, _ = self._loss_for_output(
                    aux["prim_pred_spans"], aux["prim_pred_logits"],
                    targets, suffix=f"_aux{i}",
                    cached_prim_assignments=prim_assignments,
                )
                losses["loss"] = losses["loss"] + aux_prim.pop(f"loss_aux{i}")
                losses.update(aux_prim)

                aux_ev, _ = self._loss_for_event_output(
                    aux["event_pred_spans"], targets, suffix=f"_event_aux{i}",
                    cached_event_assignments=event_assignments,
                )
                losses["loss"] = losses["loss"] + aux_ev.pop(f"loss_event_aux{i}")
                losses.update(aux_ev)

                if use_corr_head and "event_prim_pred_logits" in aux:
                    aux_corr = self._event_corr_loss_for_output(
                        aux["event_prim_pred_logits"],
                        prim_assignments,
                        event_assignments,
                        suffix=f"_event_corr_aux{i}",
                    )
                    raw_aux_corr = aux_corr[f"loss_event_corr_aux{i}"]
                    losses["loss"] = losses["loss"] + (
                        lam4 * self.config.event_prim_corr_cls_loss_weight * raw_aux_corr
                    )
                    losses[f"loss_event_corr_aux{i}"] = raw_aux_corr

        # ------------------------------------------------------------------
        # Encoder intermediate supervision — primitives only
        # ------------------------------------------------------------------
        if self.config.enc_interm_loss and "interm_outputs" in outputs:
            interm = outputs["interm_outputs"]
            interm_losses, _ = self._loss_for_output(
                interm["pred_spans"], interm["pred_logits"],
                targets, suffix="_interm",
            )
            losses["loss"] = losses["loss"] + interm_losses.pop("loss_interm")
            losses.update(interm_losses)

        # ------------------------------------------------------------------
        # DN direct-supervision losses (primitives only)
        # ------------------------------------------------------------------
        if "dn_outputs" in outputs and "dn_meta" in outputs:
            dn_meta = outputs["dn_meta"]
            dn_out  = outputs["dn_outputs"]
            dn_losses = self._dn_loss_for_output(
                dn_out["pred_spans"], dn_out["pred_logits"],
                dn_meta, suffix="_dn",
            )
            losses["loss"] = losses["loss"] + self.config.dn_loss_weight * dn_losses.pop("loss_dn")
            losses.update(dn_losses)

            if self.config.dec_aux_loss and "dn_aux_outputs" in outputs:
                for i, dn_aux in enumerate(outputs["dn_aux_outputs"]):
                    dn_aux_losses = self._dn_loss_for_output(
                        dn_aux["pred_spans"], dn_aux["pred_logits"],
                        dn_meta, suffix=f"_dn_aux{i}",
                    )
                    losses["loss"] = losses["loss"] + self.config.dn_loss_weight * dn_aux_losses.pop(f"loss_dn_aux{i}")
                    losses.update(dn_aux_losses)

        # ------------------------------------------------------------------
        # Scalar auxiliary MSE loss
        # ------------------------------------------------------------------
        if (self.config.scalar_aux_loss
                and "num_pred" in outputs
                and outputs["num_mask"].any()):
            pred_v = outputs["num_pred"]
            tgt_v  = outputs["num_target"]
            mask   = outputs["num_mask"]
            mse    = F.mse_loss(pred_v[mask], tgt_v[mask])
            losses["loss"]        = losses["loss"] + self.config.scalar_aux_loss_weight * mse
            losses["loss_scalar"] = mse

        # ------------------------------------------------------------------
        # Change-point BCE auxiliary loss
        # ------------------------------------------------------------------
        if (self.config.use_change_point_aux_loss
                and "cp_logits" in outputs
                and any("cp_targets" in t for t in targets)):
            cp_logits = outputs["cp_logits"]
            cp_tgts   = torch.stack([
                t["cp_targets"].to(cp_logits.device)
                for t in targets if "cp_targets" in t
            ])
            bce = F.binary_cross_entropy_with_logits(
                cp_logits[: cp_tgts.shape[0]], cp_tgts
            )
            losses["loss"]    = losses["loss"] + self.config.cp_aux_loss_weight * bce
            losses["loss_cp"] = bce

        return losses
