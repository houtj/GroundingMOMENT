"""
decoder.py
6-layer cross-modality decoder for Grounding MOMENT — primitive + event query layout.

Query layout (concatenated):
  [P_0, ..., P_{Np-1}, E_0, ..., E_{Ne-1}]

Per decoder layer: self-attn → text cross-attn → TS cross-attn → FFN

Span and class heads:
  span_heads       — SpanHead applied to all Np primitive slots per layer
  event_span_heads — SpanHead applied to all Ne event slots per layer
  class_heads      — ClassHead applied to primitive slots only (events have no text class head)

Primitive/event self-attention isolation masks:
  Permanent mask (apply_prim_event_permanent_mask=True):
    Blocks the prim->event direction, unconditionally, for the entire run.
    Primitive queries can never attend to event queries.
  Warm-up mask (apply_event_prim_warmup_mask=True):
    Blocks the event->prim direction, only while active (first N epochs).
    Event queries cannot attend to primitive queries until the warm-up ends,
    after which this direction is permanently open.
  Both masks are composed with the DN block mask (DN always fully isolated
  from matching queries).

DN keys (dn_outputs / dn_aux_outputs) use the original pred_spans / pred_logits keys
(primitive-only) so the existing DN loss code needs no changes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GroundingMomentConfig
from .feature_enhancer import FFN, gen_sineembed_for_position


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU hidden activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SpanHead(nn.Module):
    """
    3-layer MLP that predicts (start_delta, width_delta) in logit space.
    No sigmoid — the caller applies inverse_sigmoid + sigmoid for refinement.
    """

    def __init__(self, d_model: int, num_layers: int = 3):
        super().__init__()
        self.mlp = MLP(d_model, d_model, 2, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> (..., 2) raw logit-space deltas"""
        return self.mlp(x)


class ClassHead(nn.Module):
    """
    Contrastive classification head.

    use_extra_projections=True: learned Linear projections align independently-
    trained MOMENT and BERT feature spaces (recommended when backbones are
    frozen).

    use_extra_projections=False: parameter-free dot product (ContrastiveEmbed
    style) — recommended when both backbones are unfrozen, since end-to-end
    fine-tuning naturally aligns the feature spaces.

    In both modes, both inputs are L2-normalised before the dot product.  This
    bounds raw similarities to [-1, 1] regardless of how large the upstream
    weights grow, giving the learnable temperature full control over logit
    magnitude and preventing focal-loss saturation.

    The log-temperature is initialised at ln(5) (temperature = 5, logits in
    [-5, 5] at init) and clamped to [0, ln(100)] so the model can only tighten
    the distribution, never broaden it below temperature=1 (logits in [-1, 1]).
    """

    _LOG_SCALE_MIN: float = 0.0           # temperature >= 1
    _LOG_SCALE_MAX: float = math.log(20.0)  # temperature <= 20; logits capped at ±20

    def __init__(self, d_model: int, use_extra_projections: bool = True):
        super().__init__()
        self.use_extra_projections = use_extra_projections
        if use_extra_projections:
            self.query_proj = nn.Linear(d_model, d_model)
            self.text_proj  = nn.Linear(d_model, d_model)
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(
        self,
        query_embed: torch.Tensor,   # (B, Q, D)
        text_embed: torch.Tensor,    # (B, T, D)
    ) -> torch.Tensor:               # (B, Q, T)
        if self.use_extra_projections:
            q = F.normalize(self.query_proj(query_embed), dim=-1)
            t = F.normalize(self.text_proj(text_embed),  dim=-1)
        else:
            q = F.normalize(query_embed, dim=-1)
            t = F.normalize(text_embed,  dim=-1)
        # raw ∈ [-1, 1] by construction after L2 normalisation
        raw = torch.bmm(q, t.transpose(1, 2))
        logit_scale = self.log_logit_scale.clamp(
            min=self._LOG_SCALE_MIN, max=self._LOG_SCALE_MAX
        ).exp()
        return raw * logit_scale


class EventPrimCorrHead(nn.Module):
    """
    Event-primitive correspondence head.

    Scores which primitive queries belong to each event query via a bilinear
    dot product with L2-normalised projections and a learnable temperature.

    Output: (B, Ne, Np) logits.  Positive entries (from GT) are the primitive
    query slots Hungarian-matched to the primitives of the GT event that the
    event query was matched to.
    """

    _LOG_SCALE_MIN: float = 0.0            # temperature >= 1
    _LOG_SCALE_MAX: float = math.log(20.0) # temperature <= 20

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.event_proj = nn.Linear(d_model, hidden_dim)
        self.prim_proj  = nn.Linear(d_model, hidden_dim)
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(
        self,
        event_embed: torch.Tensor,  # (B, Ne, D)
        prim_embed:  torch.Tensor,  # (B, Np, D)
    ) -> torch.Tensor:              # (B, Ne, Np)
        e = F.normalize(self.event_proj(event_embed), dim=-1)  # (B, Ne, H)
        p = F.normalize(self.prim_proj(prim_embed),   dim=-1)  # (B, Np, H)
        raw = torch.bmm(e, p.transpose(1, 2))                  # (B, Ne, Np)
        scale = self.log_logit_scale.clamp(
            min=self._LOG_SCALE_MIN, max=self._LOG_SCALE_MAX
        ).exp()
        return raw * scale


class TemporalSpanDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)

        self.txt_cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                     dropout=dropout, batch_first=True)
        self.txt_cross_norm = nn.LayerNorm(d_model)

        self.ts_cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.ts_cross_norm = nn.LayerNorm(d_model)

        self.ffn      = FFN(d_model, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,          # (B, Q, D)
        query_pos: torch.Tensor,      # (B, Q, D)
        ts_memory: torch.Tensor,      # (B, N_ts, D)
        ts_pos: torch.Tensor,         # (B, N_ts, D)
        text_memory: torch.Tensor,    # (B, T, D)
        text_key_padding_mask: torch.Tensor,          # (B, T) True = padding
        self_attn_mask: torch.Tensor | None = None,   # (Q, Q) bool
    ) -> torch.Tensor:
        # Sub-layer 1: query self-attention
        q_normed   = self.self_norm(query)
        q_with_pos = q_normed + query_pos
        q2, _ = self.self_attn(
            q_with_pos, q_with_pos, q_normed,
            attn_mask=self_attn_mask,
        )
        query = query + q2

        # Sub-layer 2: cross-attention to text memory
        q_normed = self.txt_cross_norm(query)
        q2, _ = self.txt_cross_attn(
            q_normed + query_pos,
            text_memory,
            text_memory,
            key_padding_mask=text_key_padding_mask,
        )
        query = query + q2

        # Sub-layer 3: cross-attention to TS memory
        q_normed = self.ts_cross_norm(query)
        q2, _ = self.ts_cross_attn(
            q_normed + query_pos,
            ts_memory + ts_pos,
            ts_memory,
        )
        query = query + q2

        # Sub-layer 4: FFN
        query = query + self.ffn(self.ffn_norm(query))
        return query


class TemporalSpanDecoder(nn.Module):
    """
    Full decoder for the primitive + event query layout.

    Accepts Np + Ne query slots, reference points, and an optional attention mask.
    SpanHead (primitives), EventSpanHead (events), and ClassHead (primitives only)
    are applied per layer.

    When config.use_event_prim_corr_head is True, an EventPrimCorrHead is also
    applied per layer, producing event_prim_pred_logits (B, Ne, Np) in the output.
    """

    def __init__(
        self,
        config: GroundingMomentConfig,
        span_heads: nn.ModuleList,
        class_heads: nn.ModuleList,
        event_span_heads: nn.ModuleList,
        event_prim_corr_heads: nn.ModuleList | None = None,
    ):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        self.ref_point_head = MLP(D, D, D, 2)

        self.layers = nn.ModuleList([
            TemporalSpanDecoderLayer(D, config.decoder_heads,
                                     config.decoder_ffn_dim, config.decoder_dropout)
            for _ in range(config.decoder_layers)
        ])

        self.norm = nn.LayerNorm(D)

        self.span_heads            = span_heads
        self.class_heads           = class_heads
        self.event_span_heads      = event_span_heads
        self.event_prim_corr_heads = event_prim_corr_heads  # None when disabled

    def forward(
        self,
        ts_memory: torch.Tensor,           # (B, N_ts, D)
        ts_pos: torch.Tensor,              # (B, N_ts, D)
        text_memory: torch.Tensor,         # (B, T, D)
        text_mask: torch.Tensor,           # (B, T)  1=real, 0=padding
        query_content: torch.Tensor,       # (B, Np+Ne, D)
        reference_points: torch.Tensor,    # (B, Np+Ne, 2)
        n_prim: int,
        n_event: int,
        apply_prim_event_permanent_mask: bool = False,
        apply_event_prim_warmup_mask: bool = False,
        dn_query_content: torch.Tensor | None = None,
        dn_reference_points: torch.Tensor | None = None,
        dn_attn_mask: torch.Tensor | None = None,  # (N_dn+Np+Ne, N_dn+Np+Ne) bool
    ) -> dict:
        """
        Returns dict with keys (matching queries only, not DN):
            prim_pred_spans:           (B, Np, 2)
            prim_pred_logits:          (B, Np, T)
            event_pred_spans:          (B, Ne, 2)
            event_prim_pred_logits:    (B, Ne, Np)  — only when use_event_prim_corr_head=True
            aux_outputs:               list of {prim_pred_spans, prim_pred_logits,
                                                event_pred_spans[, event_prim_pred_logits]}
            dn_outputs:                {pred_spans, pred_logits}  — DN primitive queries, last layer
            dn_aux_outputs:            list of same
        """
        # Prepend DN queries if present (training only)
        if dn_query_content is not None:
            content = torch.cat([dn_query_content, query_content], dim=1)
            ref_pts = torch.cat([dn_reference_points, reference_points], dim=1)
        else:
            content = query_content
            ref_pts = reference_points

        n_dn   = dn_query_content.shape[1] if dn_query_content is not None else 0
        Q_match = n_prim + n_event
        text_key_padding_mask = (text_mask == 0)

        all_prim_spans       = []
        all_prim_logits      = []
        all_event_spans      = []
        all_event_prim_logits = []
        all_dn_spans         = []
        all_dn_logits        = []

        # Build the isolation self-attn block for the matching queries (Np+Ne, Np+Ne).
        # True = blocked.
        #   Permanent: primitives (rows 0..Np-1) can never attend to events (cols Np..Np+Ne-1).
        #   Warm-up:   events (rows Np..Np+Ne-1) cannot attend to primitives (cols 0..Np-1)
        #              only while apply_event_prim_warmup_mask is True.
        if apply_prim_event_permanent_mask or apply_event_prim_warmup_mask:
            match_sa_mask = torch.zeros(Q_match, Q_match, dtype=torch.bool,
                                         device=content.device)
            if apply_prim_event_permanent_mask:
                match_sa_mask[:n_prim, n_prim:] = True
            if apply_event_prim_warmup_mask:
                match_sa_mask[n_prim:, :n_prim] = True
        else:
            match_sa_mask = None

        for i, layer in enumerate(self.layers):
            # Compose DN mask with per-layer matching self-attn mask.
            # dn_attn_mask already encodes the full (N_dn+Np+Ne, N_dn+Np+Ne) block
            # (built in build_full_attn_mask with the same two flags), so it is
            # already consistent with match_sa_mask.
            layer_sa_mask = dn_attn_mask  # may be None or full combined mask

            # When no DN but we still need an isolation mask, use the matching-only mask.
            if n_dn == 0 and match_sa_mask is not None:
                layer_sa_mask = match_sa_mask

            query_pos = self.ref_point_head(
                gen_sineembed_for_position(ref_pts, content.shape[-1])
            )

            content = layer(
                content, query_pos,
                ts_memory, ts_pos,
                text_memory, text_key_padding_mask,
                layer_sa_mask,
            )

            # Split into DN and matching slices
            match_content = content[:, n_dn:]                  # (B, Np+Ne, D)
            match_ref     = ref_pts[:, n_dn:]                  # (B, Np+Ne, 2)
            prim_c        = match_content[:, :n_prim]          # (B, Np, D)
            event_c       = match_content[:, n_prim:]          # (B, Ne, D)

            # Refine reference points with separate span heads
            prim_delta   = self.span_heads[i](prim_c)          # (B, Np, 2)
            event_delta  = self.event_span_heads[i](event_c)   # (B, Ne, 2)
            match_delta  = torch.cat([prim_delta, event_delta], dim=1)
            new_match_ref = (_inverse_sigmoid(match_ref) + match_delta).sigmoid()

            # Refine DN reference points (primitive-only DN)
            if n_dn > 0:
                dn_content_i = content[:, :n_dn]
                dn_ref       = ref_pts[:, :n_dn]
                dn_delta     = self.span_heads[i](dn_content_i)
                new_dn_ref   = (_inverse_sigmoid(dn_ref) + dn_delta).sigmoid()
                ref_pts = torch.cat([new_dn_ref.detach(), new_match_ref.detach()], dim=1)
            else:
                ref_pts = new_match_ref.detach()

            # Class head — primitives only
            prim_normed   = self.norm(prim_c)
            event_normed  = self.norm(event_c)
            prim_logits_i = self.class_heads[i](prim_normed, text_memory)
            prim_spans_i  = new_match_ref[:, :n_prim]
            event_spans_i = new_match_ref[:, n_prim:]

            all_prim_spans.append(prim_spans_i)
            all_prim_logits.append(prim_logits_i)
            all_event_spans.append(event_spans_i)

            # Event-primitive correspondence head (optional)
            if self.event_prim_corr_heads is not None:
                ep_logits_i = self.event_prim_corr_heads[i](event_normed, prim_normed)
                all_event_prim_logits.append(ep_logits_i)

            if n_dn > 0:
                dn_normed    = self.norm(dn_content_i)
                dn_logits_i  = self.class_heads[i](dn_normed, text_memory)
                all_dn_spans.append(new_dn_ref)
                all_dn_logits.append(dn_logits_i)

        output = {
            "prim_pred_spans":  all_prim_spans[-1],    # (B, Np, 2)
            "prim_pred_logits": all_prim_logits[-1],   # (B, Np, T)
            "event_pred_spans": all_event_spans[-1],   # (B, Ne, 2)
        }
        if all_event_prim_logits:
            output["event_prim_pred_logits"] = all_event_prim_logits[-1]  # (B, Ne, Np)

        if self.config.dec_aux_loss:
            aux_entries = []
            for idx, (ps, pl, es) in enumerate(
                zip(all_prim_spans[:-1], all_prim_logits[:-1], all_event_spans[:-1])
            ):
                entry: dict = {
                    "prim_pred_spans":  ps,
                    "prim_pred_logits": pl,
                    "event_pred_spans": es,
                }
                if all_event_prim_logits:
                    entry["event_prim_pred_logits"] = all_event_prim_logits[idx]
                aux_entries.append(entry)
            output["aux_outputs"] = aux_entries

        if n_dn > 0:
            output["dn_outputs"] = {
                "pred_spans":  all_dn_spans[-1],
                "pred_logits": all_dn_logits[-1],
            }
            if self.config.dec_aux_loss:
                output["dn_aux_outputs"] = [
                    {"pred_spans": s, "pred_logits": l}
                    for s, l in zip(all_dn_spans[:-1], all_dn_logits[:-1])
                ]

        return output
