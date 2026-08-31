"""
dn.py
CDN-style denoising training for Grounding MOMENT — flat per-primitive DN.

Each DN slot corresponds to one GT primitive (flattened across all K event
occurrences and all num_phrases phrases for the current description).

Two sides per primitive: positive (light noise) and negative (pushed off-span).

Slot layout within the DN block (N_dn = 2 * G * M_max):
  For each denoising repeat g in [0, G):
    For each GT primitive m in [0, M_max):
      [pos_prim_m, neg_prim_m]

Keys in dn_meta:
  valid_mask    (B, N_dn) bool  — True where m < M_per_sample[b]
  is_positive   (B, N_dn) bool  — True for the positive side
  gt_spans      (B, N_dn, 2)   — GT primitive span for each slot
  gt_cls_map    (B, N_dn, T)   — positive token map (phrase mask for positives,
                                  zeros for negatives)
  n_dn          int
  M_max         int
  dn_number     int
"""

from __future__ import annotations

import torch

from .config import GroundingMomentConfig


def prepare_for_dn(
    targets: list[dict],
    text_memory: torch.Tensor,    # (B, T, D)
    attention_mask: torch.Tensor, # (B, T)
    config: GroundingMomentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict] | tuple[None, None, None, None]:
    """
    Build flat CDN denoising queries for one batch.

    Flattens all GT primitives (K events × num_phrases phrases) per sample
    into M per-sample flat primitives, then builds 2 * G * M_max DN slots.

    Returns (dn_query_content, dn_reference_points, attn_mask_dn, dn_meta)
    or (None, None, None, None) if no sample has any GT primitives.

    attn_mask_dn covers only the DN block (N_dn, N_dn).
    build_full_attn_mask() appends it before the matching queries.
    """
    B, T, D = text_memory.shape
    device  = text_memory.device
    dtype   = text_memory.dtype

    G          = config.dn_number
    span_noise = config.dn_span_noise_scale
    P_data     = config.max_phrases

    # ── 1. Flatten GT primitives per sample ─────────────────────────────────
    # A GT primitive (k, i) is valid when:
    #   k < num_events, i < num_phrases, primitive span width > 0
    gt_prim_spans_list: list[list[tuple[float, float]]] = []
    gt_prim_phrase_idx_list: list[list[int]] = []

    for b, t in enumerate(targets):
        if not t.get("has_events", False) or t.get("num_events", 0) == 0:
            gt_prim_spans_list.append([])
            gt_prim_phrase_idx_list.append([])
            continue

        num_events  = int(t["num_events"])
        num_phrases = int(t.get("num_phrases", 0))
        prim_spans  = t["primitive_spans"]  # (K, P_data, 2)

        spans_flat   = []
        phrase_flat  = []
        for k in range(num_events):
            for i in range(min(num_phrases, P_data)):
                s, w = prim_spans[k, i, 0].item(), prim_spans[k, i, 1].item()
                if w > 0:
                    spans_flat.append((s, w))
                    phrase_flat.append(i)
        gt_prim_spans_list.append(spans_flat)
        gt_prim_phrase_idx_list.append(phrase_flat)

    M_per_sample = [len(s) for s in gt_prim_spans_list]
    M_max = max(M_per_sample) if M_per_sample else 0
    if M_max == 0:
        return None, None, None, None

    N_dn = 2 * G * M_max

    # ── 2. Stack GT primitive spans into (B, M_max, 2) ──────────────────────
    gt_spans_b = torch.zeros(B, M_max, 2, device=device, dtype=dtype)
    for b, spans in enumerate(gt_prim_spans_list):
        for m, (s, w) in enumerate(spans):
            gt_spans_b[b, m, 0] = s
            gt_spans_b[b, m, 1] = w

    # ── 3. Stack phrase positive maps (B, M_max, T) ──────────────────────────
    phrase_masks_b = torch.zeros(B, M_max, T, device=device, dtype=dtype)
    for b, t in enumerate(targets):
        pm = t["phrase_token_masks"].to(device=device, dtype=dtype)  # (P_data, T)
        p_idxs = gt_prim_phrase_idx_list[b]
        for m, pi in enumerate(p_idxs):
            if pi < pm.shape[0]:
                phrase_masks_b[b, m] = pm[pi]

    # ── 4. Validity mask (B, M_max) ──────────────────────────────────────────
    M_per_t  = torch.tensor(M_per_sample, device=device, dtype=torch.long)
    m_range  = torch.arange(M_max, device=device)
    prim_valid = m_range.unsqueeze(0) < M_per_t.unsqueeze(1)   # (B, M_max)

    # ── 5. Build noised reference points ─────────────────────────────────────
    # Layout: (B, G, 2, M_max, 2)  — axis[2] = side (0=pos, 1=neg)
    starts = gt_spans_b[..., 0]   # (B, M_max)
    widths = gt_spans_b[..., 1].clamp(min=1e-4)

    pos_factor = torch.full((B, G, M_max), 0.5, device=device, dtype=dtype)
    neg_factor = 0.5 + 0.5 * torch.rand(B, G, M_max, device=device, dtype=dtype)
    factor     = torch.stack([pos_factor, neg_factor], dim=2)  # (B, G, 2, M_max)

    noise_s = torch.rand(B, G, 2, M_max, device=device, dtype=dtype) * 2 - 1
    noise_w = torch.rand(B, G, 2, M_max, device=device, dtype=dtype) * 2 - 1

    s_bcast = starts.view(B, 1, 1, M_max)
    w_bcast = widths.view(B, 1, 1, M_max)

    ns = (s_bcast + noise_s * factor * span_noise * w_bcast).clamp(0.0, 1.0 - 1e-6)
    nw = (w_bcast + noise_w * factor * span_noise * w_bcast).clamp(min=1e-6)
    nw = torch.minimum(nw, (1.0 - ns).clamp(min=1e-6))
    dn_ref = torch.stack([ns, nw], dim=-1)  # (B, G, 2, M_max, 2)
    dn_reference_points = dn_ref.reshape(B, N_dn, 2)

    # ── 6. DN query content ──────────────────────────────────────────────────
    # Per primitive: mean of text_memory over its phrase mask; fallback to global mean.
    # Shape: (B, M_max, D)
    phrase_mask_bool = phrase_masks_b.bool()  # (B, M_max, T)
    mask_f = phrase_mask_bool.float().unsqueeze(-1)  # (B, M_max, T, 1)
    # text_memory: (B, T, D) → expand to (B, 1, T, D) for broadcast
    tm_exp = text_memory.unsqueeze(1)  # (B, 1, T, D)
    pooled_prim = (tm_exp * mask_f).sum(2) / mask_f.sum(2).clamp(1)  # (B, M_max, D)

    # Fallback to global mean for primitives with empty phrase mask
    global_mean = text_memory.mean(1, keepdim=True)  # (B, 1, D)
    empty_mask  = ~phrase_mask_bool.any(dim=2)        # (B, M_max)
    pooled_prim = torch.where(
        empty_mask.unsqueeze(-1), global_mean.expand(-1, M_max, -1), pooled_prim
    )

    feat_std = text_memory.std(dim=(1, 2), keepdim=False).clamp(min=1e-4)  # (B,)

    # Expand to (B, G, 2, M_max, D) and add jitter
    pp_bcast = pooled_prim.view(B, 1, 1, M_max, D).expand(B, G, 2, M_max, D)
    jitter   = (0.1 * feat_std.view(B, 1, 1, 1, 1)
                * torch.randn(B, G, 2, M_max, D, device=device, dtype=dtype))
    dn_query_content = (pp_bcast + jitter).reshape(B, N_dn, D)

    # ── 7. Build dn_meta tensors ──────────────────────────────────────────────
    # valid_mask: expand (B, M_max) → (B, G, 2, M_max) → (B, N_dn)
    valid_exp = (prim_valid.view(B, 1, 1, M_max)
                           .expand(B, G, 2, M_max)
                           .reshape(B, N_dn)
                           .contiguous())

    # is_positive: side 0 = positive, side 1 = negative
    side_pos   = torch.tensor([True, False], device=device).view(1, 1, 2, 1)
    is_pos_exp = (side_pos.expand(B, G, 2, M_max)
                          .reshape(B, N_dn)
                          .contiguous())

    # gt_spans: GT primitive span (not noised)
    gt_s_exp = (gt_spans_b.view(B, 1, 1, M_max, 2)
                           .expand(B, G, 2, M_max, 2)
                           .reshape(B, N_dn, 2)
                           .contiguous())

    # gt_cls_map: phrase positive map for positives, zeros for negatives
    cls_exp   = (phrase_masks_b.view(B, 1, 1, M_max, T)
                               .expand(B, G, 2, M_max, T))
    side_fac  = torch.tensor([1.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 1, 1)
    gt_cls_map = (cls_exp * side_fac).reshape(B, N_dn, T).contiguous()

    dn_meta = {
        "valid_mask":  valid_exp,
        "is_positive": is_pos_exp,
        "gt_spans":    gt_s_exp,
        "gt_cls_map":  gt_cls_map,
        "n_dn":        N_dn,
        "M_max":       M_max,
        "dn_number":   G,
    }

    # ── 8. DN ↔ DN attention mask ─────────────────────────────────────────────
    # Block-diagonal per repeat g; within each repeat all 2*M_max slots can attend.
    dn_group_size = 2 * M_max
    attn_mask_dn  = torch.ones(N_dn, N_dn, dtype=torch.bool, device=device)
    diag_block    = torch.zeros(dn_group_size, dn_group_size, dtype=torch.bool, device=device)
    for g in range(G):
        lo = g * dn_group_size
        attn_mask_dn[lo:lo + dn_group_size, lo:lo + dn_group_size] = diag_block

    return dn_query_content, dn_reference_points, attn_mask_dn, dn_meta


def build_full_attn_mask(
    attn_mask_dn: torch.Tensor,                         # (N_dn, N_dn) bool
    matching_query_count: int | None = None,
    apply_prim_event_permanent_mask: bool = False,
    apply_event_prim_warmup_mask: bool = False,
    n_prim: int = 0,
    n_event: int = 0,
) -> torch.Tensor:
    """
    Compose the DN-only mask with the matching-query block into a single
    (N_dn + Np + Ne, N_dn + Np + Ne) attention mask.

    Blocking rules:
      DN   ↔ DN    : from attn_mask_dn (block-diagonal per DN repeat)
      DN   ↔ match : True (full isolation)
      match ↔ match: two independent components, each optionally applied —
                       permanent: blocks prim->event direction (rows 0..n_prim-1,
                                  cols n_prim..n_prim+n_event-1), unconditionally
                                  whenever apply_prim_event_permanent_mask=True.
                       warm-up:   blocks event->prim direction (rows n_prim..,
                                  cols 0..n_prim-1), only while
                                  apply_event_prim_warmup_mask=True.

    When matching_query_count is not given, only the DN block is returned.
    """
    N_dn   = attn_mask_dn.shape[0]
    device = attn_mask_dn.device

    if matching_query_count is None:
        return attn_mask_dn

    Nq    = matching_query_count          # = n_prim + n_event
    total = N_dn + Nq
    full  = torch.zeros(total, total, dtype=torch.bool, device=device)
    full[:N_dn, :N_dn] = attn_mask_dn
    full[:N_dn, N_dn:] = True            # DN blocks matching
    full[N_dn:, :N_dn] = True            # matching blocks DN

    if n_prim > 0 and n_event > 0:
        if apply_prim_event_permanent_mask:
            full[N_dn: N_dn + n_prim, N_dn + n_prim: N_dn + n_prim + n_event] = True
        if apply_event_prim_warmup_mask:
            full[N_dn + n_prim: N_dn + n_prim + n_event, N_dn: N_dn + n_prim] = True

    return full
