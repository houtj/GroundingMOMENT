"""
model.py
Grounding MOMENT model — flat single-layer primitive + event detector.

Query layout: [P_0, ..., P_{Np-1}, E_0, ..., E_{Ne-1}]
  Np = num_primitive_queries (default 64)
  Ne = num_event_queries     (default 16)

Forward pipeline:
  1-3.  Text: tokenize → BERT → project to hidden_dim
  4-6.  TS:   MOMENT backbone → multi-scale features (7 levels) → positional proposals
  7.    Feature Enhancer: bidirectional cross-modal attention (6 layers)
  8.    Encoder projection: enc_output + enc_output_norm on TS memory
  9.    Language-Guided Query Selection:
          top-(Np+Ne) patches selected; reference points initialised from those.
  10.   Query content:
          prim_content  = tgt_embed_prim[q] + type_emb_prim   (Np, D)
          event_content = tgt_embed_event[q] + type_emb_event  (Ne, D)
          concatenated to (Np+Ne, D)
  11.   Cross-Modality Decoder: 6 layers, per-layer reference refinement.
          Self-attn isolation masks (each independently toggleable via config):
            Permanent:  prim queries can never attend to event queries
                        (config.enable_prim_event_permanent_mask).
            Warm-up:    event queries cannot attend to prim queries for the
                        first config.event_stop_grad_epochs epochs
                        (config.enable_event_prim_warmup_mask), then this
                        direction is permanently open.
  12.   Output:
          prim_pred_spans   (B, Np, 2)
          prim_pred_logits  (B, Np, T)
          event_pred_spans  (B, Ne, 2)

Loss supervision:
  Primitives: single-stage Hungarian over Np slots vs flat GT primitives (see loss.py).
  Events:     separate Hungarian over Ne slots vs GT event spans (L1+IoU only).
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast

from .backbone import MOMENTBackbone
from .config import GroundingMomentConfig
from .decoder import ClassHead, EventPrimCorrHead, MLP, SpanHead, TemporalSpanDecoder
from .dn import build_full_attn_mask, prepare_for_dn
from .feature_enhancer import FeatureEnhancer, gen_sineembed_for_position
from .text_encoder import BERTTextEncoder


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def _compute_position_proposals(level_sizes: list[int],
                                  device: torch.device,
                                  base_width: float = 0.10) -> torch.Tensor:
    proposals = []
    for lvl, n in enumerate(level_sizes):
        i = torch.arange(n, device=device, dtype=torch.float32)
        starts = i / n
        width  = min(base_width * (2 ** lvl), 0.9)
        widths = torch.full((n,), width, device=device)
        proposals.append(torch.stack([starts, widths], dim=1))
    proposals_norm = torch.cat(proposals, dim=0)   # (N_total, 2)
    return _inverse_sigmoid(proposals_norm)


# ---------------------------------------------------------------------------
# Event query selector (attention-pooled text summary)
# ---------------------------------------------------------------------------

class EventQuerySelector(nn.Module):
    """
    Produces a single global text summary vector d for event query selection.

    d = Σ_j α_j M_txt(j),  where α_j = softmax(MLP(M_txt(j)))

    This allows event queries to be seeded from patches most relevant to the
    full description as a whole, rather than to any single phrase token.
    """

    def __init__(self, d_model: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.mlp = MLP(d_model, hidden_dim, 1, num_layers)

    def forward(
        self,
        text_memory: torch.Tensor,  # (B, T, D)
        text_mask:   torch.Tensor,  # (B, T)  1=real, 0=padding
    ) -> torch.Tensor:              # (B, D)
        scores = self.mlp(text_memory).squeeze(-1)                     # (B, T)
        scores = scores.masked_fill(text_mask == 0, float('-inf'))
        alpha  = scores.softmax(dim=-1)                                 # (B, T)
        return (alpha.unsqueeze(-1) * text_memory).sum(dim=1)           # (B, D)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GroundingMOMENT(nn.Module):
    def __init__(self, config: GroundingMomentConfig):
        super().__init__()
        self.config = config
        D  = config.hidden_dim
        Np = config.num_primitive_queries
        Ne = config.num_event_queries

        # ----- Backbones -----
        self.backbone     = MOMENTBackbone(config)
        self.text_encoder = BERTTextEncoder(config)

        # ----- Numeric token wiring -----
        if config.use_numeric_tokens:
            _fast_tok = BertTokenizerFast.from_pretrained(config.text_model)
            num_tok_id = _fast_tok.convert_tokens_to_ids(config.num_token_str)
            self.text_encoder.num_token_id = num_tok_id

            if config.freeze_text_encoder:
                emb_w = self.text_encoder.bert.embeddings.word_embeddings.weight
                emb_w.requires_grad_(True)
                _tok_id = num_tok_id

                def _row_grad_mask(grad: torch.Tensor) -> torch.Tensor:
                    mask = torch.zeros_like(grad)
                    mask[_tok_id] = 1.0
                    return grad * mask

                emb_w.register_hook(_row_grad_mask)

        # ----- Change-point dependency checks -----
        if config.use_change_point_aux_loss:
            assert config.use_change_point_features
        if config.use_change_point_query_bias:
            assert config.use_change_point_features

        # ----- TS-side Fourier features -----
        if config.use_ts_fourier:
            assert config.use_numeric_tokens
            assert config.use_revin
            self.backbone.attach_patch_fourier(self.text_encoder.scalar_encoder)

        # ----- Feature Enhancer -----
        self.feature_enhancer = FeatureEnhancer(config)

        # ----- Encoder projection -----
        self.enc_output      = nn.Linear(D, D)
        self.enc_output_norm = nn.LayerNorm(D)

        # ----- Primitive prediction heads -----
        _span_head  = SpanHead(D)
        _class_head = ClassHead(D, use_extra_projections=config.use_class_extra_projections)

        nn.init.constant_(_span_head.mlp.layers[-1].weight.data, 0)
        nn.init.constant_(_span_head.mlp.layers[-1].bias.data, 0)

        if config.dec_span_embed_share:
            span_heads = nn.ModuleList([_span_head] * config.decoder_layers)
        else:
            span_heads = nn.ModuleList(
                [copy.deepcopy(_span_head) for _ in range(config.decoder_layers)]
            )

        class_heads = nn.ModuleList([_class_head] * config.decoder_layers)

        if config.two_stage_span_embed_share:
            assert config.dec_span_embed_share
            self.enc_span_embed = _span_head
        else:
            self.enc_span_embed = copy.deepcopy(_span_head)

        if config.two_stage_class_embed_share:
            self.enc_out_class_embed = _class_head
        else:
            self.enc_out_class_embed = copy.deepcopy(_class_head)

        # ----- Event prediction heads (separate SpanHead, no ClassHead) -----
        _event_span_head = SpanHead(D)
        nn.init.constant_(_event_span_head.mlp.layers[-1].weight.data, 0)
        nn.init.constant_(_event_span_head.mlp.layers[-1].bias.data, 0)

        if config.dec_span_embed_share:
            event_span_heads = nn.ModuleList([_event_span_head] * config.decoder_layers)
        else:
            event_span_heads = nn.ModuleList(
                [copy.deepcopy(_event_span_head) for _ in range(config.decoder_layers)]
            )

        # ----- Event-primitive correspondence head (optional) -----
        if config.use_event_prim_corr_head:
            _corr_head = EventPrimCorrHead(D, config.event_prim_corr_hidden_dim)
            if config.event_prim_corr_share_across_layers:
                event_prim_corr_heads = nn.ModuleList(
                    [_corr_head] * config.decoder_layers
                )
            else:
                import copy as _copy
                event_prim_corr_heads = nn.ModuleList(
                    [_copy.deepcopy(_corr_head) for _ in range(config.decoder_layers)]
                )
        else:
            event_prim_corr_heads = None

        # ----- Event query selector (optional) -----
        if config.use_attn_pooled_event_selection:
            self.event_query_selector = EventQuerySelector(
                D,
                config.event_query_pool_mlp_hidden,
                config.event_query_pool_mlp_layers,
            )
        else:
            self.event_query_selector = None

        # ----- Primitive query embedding -----
        self.tgt_embed_prim  = nn.Embedding(Np, D)
        # ----- Event query embedding -----
        self.tgt_embed_event = nn.Embedding(Ne, D)

        # ----- Type residuals (small init so events start near primitives) -----
        self.type_emb_prim  = nn.Parameter(torch.zeros(D))
        self.type_emb_event = nn.Parameter(torch.zeros(D))

        # ----- Cross-modality decoder -----
        self.decoder = TemporalSpanDecoder(
            config, span_heads, class_heads, event_span_heads, event_prim_corr_heads
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tgt_embed_prim.weight,  std=0.02)
        nn.init.normal_(self.tgt_embed_event.weight, std=0.02)
        nn.init.normal_(self.type_emb_prim,  std=1e-2)
        nn.init.normal_(self.type_emb_event, std=1e-2)
        nn.init.xavier_uniform_(self.enc_output.weight)
        nn.init.constant_(self.enc_output.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        time_series: torch.Tensor,       # (B, 512, 1)
        input_ids: torch.Tensor,         # (B, T)
        attention_mask: torch.Tensor,    # (B, T)
        targets: list[dict] | None = None,
        num_values: torch.Tensor | None = None,
        num_mask: torch.Tensor | None = None,
        current_epoch: int | None = None,
    ) -> dict:
        B      = time_series.shape[0]
        device = time_series.device
        Np     = self.config.num_primitive_queries
        Ne     = self.config.num_event_queries

        # 1-3. Encode time series
        ts_features, cp_score = self.backbone(time_series)

        revin_mean = revin_stdev = revin_gamma = revin_beta = None
        if self.config.use_numeric_tokens and self.config.use_revin:
            revin = self.backbone.revin
            revin_mean  = revin.mean
            revin_stdev = revin.stdev
            if revin.affine:
                revin_gamma = revin.gamma
                revin_beta  = revin.beta

        # 4-6. Encode text
        text_feat, text_mask, v_norm, pred_v = self.text_encoder(
            input_ids, attention_mask,
            num_values=num_values, num_mask=num_mask,
            revin_mean=revin_mean, revin_stdev=revin_stdev,
            revin_gamma=revin_gamma, revin_beta=revin_beta,
        )

        # 7. Cross-modal feature enhancement
        ts_memory, text_memory, ts_pos, _ = self.feature_enhancer(
            ts_features, text_feat, text_mask
        )

        # 8. Encoder projection
        enc_out = self.enc_output_norm(self.enc_output(ts_memory))

        # 9. Language-Guided Query Selection
        enc_cls_scores = self.enc_out_class_embed(enc_out, text_memory)  # (B, N_total, T)
        text_pad_mask  = (text_mask == 0)
        enc_cls_scores_masked = enc_cls_scores.masked_fill(
            text_pad_mask.unsqueeze(1), float('-inf'),
        )

        level_sizes = [f.shape[1] for f in ts_features]

        if self.config.use_attn_pooled_event_selection:
            # Primitives: top-Np by max per-token logit (unchanged)
            prim_scores = enc_cls_scores_masked.max(dim=-1).values  # (B, N_total)
            if self.config.use_change_point_query_bias and cp_score is not None:
                prim_scores = prim_scores + self.config.cp_query_bias_weight * cp_score
            prim_topk_idx  = prim_scores.topk(Np, dim=1).indices    # (B, Np)

            # Events: top-Ne by dot product with attention-pooled text summary d
            d = self.event_query_selector(text_memory, text_mask)    # (B, D)
            event_scores = torch.bmm(enc_out, d.unsqueeze(-1)).squeeze(-1)  # (B, N_total)
            event_topk_idx = event_scores.topk(Ne, dim=1).indices    # (B, Ne)

            topk_idx = torch.cat([prim_topk_idx, event_topk_idx], dim=1)  # (B, Np+Ne)
        else:
            # Unified max-logit selection (original path)
            topk_scores = enc_cls_scores_masked.max(dim=-1).values   # (B, N_total)
            if self.config.use_change_point_query_bias and cp_score is not None:
                topk_scores = topk_scores + self.config.cp_query_bias_weight * cp_score
            topk_idx = topk_scores.topk(Np + Ne, dim=1).indices      # (B, Np+Ne)

        proposals_logit = _compute_position_proposals(
            level_sizes, device, self.config.proposal_base_width
        )
        enc_span_delta  = self.enc_span_embed(enc_out)
        enc_span_logit  = enc_span_delta + proposals_logit.unsqueeze(0)

        gather_idx      = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        ref_logit_undet = enc_span_logit.gather(1, gather_idx)
        ref_logit       = ref_logit_undet.detach()
        reference_points = ref_logit.sigmoid()              # (B, Np+Ne, 2)

        # 10. Query content (primitives first, then events)
        prim_content  = (self.tgt_embed_prim.weight + self.type_emb_prim
                         ).unsqueeze(0).expand(B, -1, -1)   # (B, Np, D)
        event_content = (self.tgt_embed_event.weight + self.type_emb_event
                         ).unsqueeze(0).expand(B, -1, -1)   # (B, Ne, D)
        query_content = torch.cat([prim_content, event_content], dim=1)  # (B, Np+Ne, D)

        # 11. Self-attn isolation masks:
        #   Permanent: prim->event is blocked for the entire run whenever enabled.
        #   Warm-up:   event->prim is blocked only for the first
        #              event_stop_grad_epochs epochs, then permanently opened.
        apply_prim_event_permanent_mask = self.config.enable_prim_event_permanent_mask
        apply_event_prim_warmup_mask = (
            self.config.enable_event_prim_warmup_mask
            and current_epoch is not None
            and current_epoch < self.config.event_stop_grad_epochs
        )

        # Build attention mask (optional DN)
        dn_query_content    = None
        dn_reference_points = None
        dn_attn_mask        = None
        dn_meta             = None

        if self.training and self.config.dn_enabled and targets is not None:
            _dn_qc, _dn_rp, _dn_mask, _dn_meta = prepare_for_dn(
                targets, text_memory, attention_mask, self.config
            )
            if _dn_qc is not None:
                dn_query_content    = _dn_qc
                dn_reference_points = _dn_rp
                dn_attn_mask        = build_full_attn_mask(
                    _dn_mask, Np + Ne,
                    apply_prim_event_permanent_mask=apply_prim_event_permanent_mask,
                    apply_event_prim_warmup_mask=apply_event_prim_warmup_mask,
                    n_prim=Np,
                    n_event=Ne,
                )
                dn_meta = _dn_meta

        # 12. Decoder
        output = self.decoder(
            ts_memory, ts_pos, text_memory, text_mask,
            query_content        = query_content,
            reference_points     = reference_points,
            n_prim               = Np,
            n_event              = Ne,
            apply_prim_event_permanent_mask = apply_prim_event_permanent_mask,
            apply_event_prim_warmup_mask    = apply_event_prim_warmup_mask,
            dn_query_content     = dn_query_content,
            dn_reference_points  = dn_reference_points,
            dn_attn_mask         = dn_attn_mask,
        )

        # 13. Package output
        result = {
            "prim_pred_spans":  output["prim_pred_spans"],   # (B, Np, 2)
            "prim_pred_logits": output["prim_pred_logits"],  # (B, Np, T)
            "event_pred_spans": output["event_pred_spans"],  # (B, Ne, 2)
        }
        if "event_prim_pred_logits" in output:
            result["event_prim_pred_logits"] = output["event_prim_pred_logits"]  # (B, Ne, Np)

        # Auxiliary decoder layer outputs
        if self.config.dec_aux_loss and "aux_outputs" in output:
            result["aux_outputs"] = output["aux_outputs"]

        # Encoder intermediate supervision (top-Np primitive spans/logits only)
        if self.config.enc_interm_loss:
            prim_topk_idx = topk_idx[:, :Np]
            interm_logits = enc_cls_scores.gather(
                1, prim_topk_idx.unsqueeze(-1).expand(-1, -1, enc_cls_scores.shape[-1])
            )
            interm_spans = ref_logit_undet[:, :Np].sigmoid()
            result["interm_outputs"] = {
                "pred_spans":  interm_spans,
                "pred_logits": interm_logits,
            }

        # DN outputs
        if dn_meta is not None and "dn_outputs" in output:
            result["dn_outputs"] = output["dn_outputs"]
            result["dn_meta"]    = dn_meta
            if self.config.dec_aux_loss and "dn_aux_outputs" in output:
                result["dn_aux_outputs"] = output["dn_aux_outputs"]

        # Numeric auxiliary outputs
        if (self.config.use_numeric_tokens
                and self.config.scalar_aux_loss
                and num_mask is not None
                and pred_v is not None
                and v_norm is not None):
            result["num_pred"]   = pred_v
            result["num_target"] = v_norm
            result["num_mask"]   = num_mask

        # Change-point auxiliary loss outputs
        if self.config.use_change_point_aux_loss and cp_score is not None:
            result["cp_logits"] = cp_score

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_values: torch.Tensor | None = None,
        num_mask: torch.Tensor | None = None,
        text_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
        seq_len: int = 512,
    ) -> list[dict]:
        """
        Post-processed inference.

        Returns list (one per batch item) of:
            prim_spans:      (Kp, 2)  integer (start, end) in [0, seq_len]
            prim_scores:     (Kp,)    confidence scores
            prim_token_idxs: list[Kp] 1D LongTensors of matched text token indices
            event_spans:     (Ke, 2)  integer (start, end) in [0, seq_len]
            event_scores:    (Ke,)    proxy scores = max IoU vs surviving prim spans
        """
        if text_threshold is None:
            text_threshold = self.config.text_threshold
        if nms_iou_threshold is None:
            nms_iou_threshold = self.config.nms_iou_threshold
        self.eval()
        out = self.forward(time_series, input_ids, attention_mask,
                           num_values=num_values, num_mask=num_mask)

        prim_spans_all  = out["prim_pred_spans"]   # (B, Np, 2)
        prim_logits_all = out["prim_pred_logits"]  # (B, Np, T)
        event_spans_all = out["event_pred_spans"]  # (B, Ne, 2)

        results = []
        for b in range(prim_spans_all.shape[0]):
            dev = prim_spans_all.device

            # ------ Primitive post-processing (unchanged logic) ------
            probs  = prim_logits_all[b].sigmoid()        # (Np, T)
            scores = probs.max(dim=-1).values             # (Np,)
            keep   = scores > text_threshold              # (Np,) bool

            spans_norm = prim_spans_all[b][keep]          # (K', 2)
            scores_k   = scores[keep]
            probs_k    = probs[keep]

            start  = spans_norm[:, 0]
            width  = spans_norm[:, 1]
            starts = (start * seq_len).clamp(0, seq_len).long()
            ends   = ((start + width) * seq_len).clamp(0, seq_len).long()

            valid    = ends > starts
            starts   = starts[valid];  ends = ends[valid]
            scores_k = scores_k[valid]; probs_k = probs_k[valid]

            order    = scores_k.argsort(descending=True)
            starts   = starts[order];  ends = ends[order]
            scores_k = scores_k[order]; probs_k = probs_k[order]

            # 1D NMS for primitives
            keep_nms   = []
            suppressed = torch.zeros(starts.shape[0], dtype=torch.bool, device=dev)
            for i in range(starts.shape[0]):
                if suppressed[i]:
                    continue
                keep_nms.append(i)
                inter_s = torch.maximum(starts[i], starts[i + 1:])
                inter_e = torch.minimum(ends[i],   ends[i + 1:])
                inter   = (inter_e - inter_s).clamp(min=0).float()
                len_i   = (ends[i] - starts[i]).float()
                len_j   = (ends[i + 1:] - starts[i + 1:]).float()
                iou     = inter / (len_i + len_j - inter + 1e-6)
                suppressed[i + 1:] |= iou > nms_iou_threshold

            if keep_nms:
                idx_t        = torch.tensor(keep_nms, device=dev)
                final_prim_s = torch.stack([starts[idx_t], ends[idx_t]], dim=1)
                final_prim_scores = scores_k[idx_t]
                surv_probs   = probs_k[idx_t]
                token_idxs   = [
                    (surv_probs[s] > text_threshold).nonzero(as_tuple=False).view(-1)
                    for s in range(final_prim_s.shape[0])
                ]
            else:
                final_prim_s      = torch.zeros(0, 2, dtype=torch.long, device=dev)
                final_prim_scores = torch.zeros(0, device=dev)
                token_idxs        = []

            # ------ Event post-processing ------
            ev_norm = event_spans_all[b]   # (Ne, 2) normalised (start, width)
            ev_s_raw = ev_norm[:, 0]
            ev_w_raw = ev_norm[:, 1]
            ev_starts = (ev_s_raw * seq_len).clamp(0, seq_len).long()
            ev_ends   = ((ev_s_raw + ev_w_raw) * seq_len).clamp(0, seq_len).long()

            ev_valid  = ev_ends > ev_starts
            ev_starts = ev_starts[ev_valid];  ev_ends = ev_ends[ev_valid]

            if ev_starts.shape[0] > 0 and final_prim_s.shape[0] > 0:
                # proxy score = max IoU of each event against surviving prim spans
                prim_s_f = final_prim_s[:, 0].float()
                prim_e_f = final_prim_s[:, 1].float()
                ev_s_f   = ev_starts.float().unsqueeze(1)  # (Ke, 1)
                ev_e_f   = ev_ends.float().unsqueeze(1)
                inter_s  = torch.maximum(ev_s_f, prim_s_f.unsqueeze(0))
                inter_e  = torch.minimum(ev_e_f, prim_e_f.unsqueeze(0))
                inter    = (inter_e - inter_s).clamp(min=0)
                ev_len   = (ev_e_f - ev_s_f).clamp(min=0)
                p_len    = (prim_e_f - prim_s_f).clamp(min=0).unsqueeze(0)
                union    = ev_len + p_len - inter
                iou_mat  = inter / (union + 1e-6)          # (Ke, Kp)
                ev_scores = iou_mat.max(dim=1).values      # (Ke,)
            else:
                ev_scores = torch.zeros(ev_starts.shape[0], device=dev)

            ev_keep = ev_scores > text_threshold
            ev_starts_k = ev_starts[ev_keep]
            ev_ends_k   = ev_ends[ev_keep]
            ev_scores_k = ev_scores[ev_keep]

            ev_order    = ev_scores_k.argsort(descending=True)
            ev_starts_k = ev_starts_k[ev_order]
            ev_ends_k   = ev_ends_k[ev_order]
            ev_scores_k = ev_scores_k[ev_order]

            # 1D NMS for events
            ev_keep_nms   = []
            ev_suppressed = torch.zeros(ev_starts_k.shape[0], dtype=torch.bool, device=dev)
            for i in range(ev_starts_k.shape[0]):
                if ev_suppressed[i]:
                    continue
                ev_keep_nms.append(i)
                inter_s = torch.maximum(ev_starts_k[i], ev_starts_k[i + 1:])
                inter_e = torch.minimum(ev_ends_k[i],   ev_ends_k[i + 1:])
                inter   = (inter_e - inter_s).clamp(min=0).float()
                len_i   = (ev_ends_k[i] - ev_starts_k[i]).float()
                len_j   = (ev_ends_k[i + 1:] - ev_starts_k[i + 1:]).float()
                iou     = inter / (len_i + len_j - inter + 1e-6)
                ev_suppressed[i + 1:] |= iou > nms_iou_threshold

            if ev_keep_nms:
                ev_idx_t     = torch.tensor(ev_keep_nms, device=dev)
                final_ev_s   = torch.stack([ev_starts_k[ev_idx_t], ev_ends_k[ev_idx_t]], dim=1)
                final_ev_sc  = ev_scores_k[ev_idx_t]
            else:
                final_ev_s   = torch.zeros(0, 2, dtype=torch.long, device=dev)
                final_ev_sc  = torch.zeros(0, device=dev)

            results.append({
                "prim_spans":      final_prim_s,
                "prim_scores":     final_prim_scores,
                "prim_token_idxs": token_idxs,
                "event_spans":     final_ev_s,
                "event_scores":    final_ev_sc,
            })

        return results
