"""
feature_enhancer.py
6-layer bidirectional cross-modal attention between time-series patches and text tokens.

This is the architectural analog of Grounding DINO's Feature Enhancer —
it fuses information between the two modalities before the decoder.

Each layer:
  1. TS self-attention
  2. Text self-attention
  3. TS → text cross-attention
  4. Text → TS cross-attention
  5. FFN on TS
  6. FFN on text
All with pre-norm + residual connections.

Also exports gen_sineembed_for_position, used by the decoder to convert
(start, width) reference points into D-dimensional sinusoidal query positions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GroundingMomentConfig


class DropPath(nn.Module):
    """Stochastic depth: drop entire samples in a batch with probability drop_prob."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (B, 1, 1, ...) to broadcast over all non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device)) / keep_prob
        return x * mask


# ---------------------------------------------------------------------------
# Positional encoding utilities
# ---------------------------------------------------------------------------

def sinusoidal_pos_encoding(seq_len: int, d_model: int,
                             device: torch.device) -> torch.Tensor:
    """Returns (1, seq_len, d_model) sinusoidal positional encoding."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


def gen_sineembed_for_position(pos_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert normalized 2-D reference points (start, width) into a D-dimensional
    sinusoidal positional embedding.  Mirrors GroundingDINO's use of sinusoidal
    embeddings so that reference points become meaningful query positional biases.

    Args:
        pos_tensor: (B, N, 2)  — (start, width) each in [0, 1]
        d_model:    int        — output embedding dimension (must be even)
    Returns:
        (B, N, d_model) sinusoidal embedding
    """
    assert d_model % 2 == 0, "d_model must be even for gen_sineembed_for_position"
    scale = 2 * math.pi
    d_half = d_model // 2

    dim_t = torch.arange(d_half, device=pos_tensor.device, dtype=torch.float32)
    # Standard sinusoidal frequency: 10000^(2i / d_half)
    dim_t = 10000 ** (2 * (dim_t // 2) / d_half)

    start = pos_tensor[..., 0:1] * scale   # (B, N, 1)
    width = pos_tensor[..., 1:2] * scale   # (B, N, 1)

    start_embed = start / dim_t            # (B, N, d_half)
    width_embed = width / dim_t            # (B, N, d_half)

    # Interleave sin(even-indexed) and cos(odd-indexed) via stack+flatten
    # (avoids in-place slice writes that can break autograd).
    start_embed = torch.stack(
        (start_embed[..., 0::2].sin(), start_embed[..., 1::2].cos()), dim=-1
    ).flatten(-2)  # (B, N, d_half)
    width_embed = torch.stack(
        (width_embed[..., 0::2].sin(), width_embed[..., 1::2].cos()), dim=-1
    ).flatten(-2)  # (B, N, d_half)

    return torch.cat([start_embed, width_embed], dim=-1)  # (B, N, d_model)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BidirectionalCrossAttention(nn.Module):
    """
    Shared-matrix bidirectional cross-attention between TS and text tokens.

    Computes ONE attention matrix A = Q_ts @ K_text^T, then derives both
    directions by softmax-ing along different axes:
      - TS output:   softmax(A,  dim=text) @ value_text  — TS attends to text
      - Text output: softmax(A^T, dim=ts)  @ value_ts    — text attends to TS

    This enforces that "ts_i finds text_j relevant" ↔ "text_j finds ts_i relevant"
    through the same underlying similarity scores, coupling gradients from both
    directions into the shared Q/K projections.
    """

    def __init__(self, d_model: int, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Shared Q/K projections into embed_dim space
        self.ts_proj   = nn.Linear(d_model, embed_dim)   # TS → queries
        self.text_proj = nn.Linear(d_model, embed_dim)   # text → keys

        # Separate value projections — each modality supplies values to the other
        self.value_ts_proj   = nn.Linear(d_model, embed_dim)  # TS values (used by text)
        self.value_text_proj = nn.Linear(d_model, embed_dim)  # text values (used by TS)

        # Output projections back to d_model
        self.out_ts_proj   = nn.Linear(embed_dim, d_model)
        self.out_text_proj = nn.Linear(embed_dim, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.ts_proj, self.text_proj,
                     self.value_ts_proj, self.value_text_proj,
                     self.out_ts_proj, self.out_text_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, embed_dim) → (B*num_heads, S, head_dim)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            B * self.num_heads, S, self.head_dim
        )

    def forward(
        self,
        ts_feat: torch.Tensor,               # (B, N_ts, d_model)
        text_feat: torch.Tensor,             # (B, N_txt, d_model)
        text_key_padding_mask: torch.Tensor, # (B, N_txt) True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N_ts, _  = ts_feat.shape
        N_txt = text_feat.shape[1]

        # --- Project into shared embed_dim space and split heads ---
        Q  = self._split_heads(self.ts_proj(ts_feat) * self.scale)  # (B*h, N_ts, head_dim)
        K  = self._split_heads(self.text_proj(text_feat))                     # (B*h, N_txt, head_dim)
        V_text = self._split_heads(self.value_text_proj(text_feat))           # (B*h, N_txt, head_dim)
        V_ts   = self._split_heads(self.value_ts_proj(ts_feat))               # (B*h, N_ts, head_dim)

        # --- ONE shared attention matrix ---
        A = torch.bmm(Q, K.transpose(1, 2))  # (B*h, N_ts, N_txt)

        # Numerical stability: per-row max shift before clamp (standard stable softmax trick)
        A = torch.clamp(
            A - A.max(dim=-1, keepdim=True)[0], min=-50000, max=50000
        )  # (B*h, N_ts, N_txt)

        A_T = A.transpose(1, 2).contiguous()  # (B*h, N_txt, N_ts)
        A_T = torch.clamp(
            A_T - A_T.max(dim=-1, keepdim=True)[0], min=-50000, max=50000
        )  # (B*h, N_txt, N_ts)

        # --- Mask padding tokens, then softmax along different axes ---
        if text_key_padding_mask is not None:
            # Expand mask for heads: (B, N_txt) → (B*h, 1, N_txt)
            mask = text_key_padding_mask[:, None, None, :].expand(
                B, self.num_heads, 1, N_txt
            ).reshape(B * self.num_heads, 1, N_txt)
            A = A.masked_fill(mask, float("-inf"))   # mask text padding for TS queries

        probs_ts   = F.dropout(A.softmax(dim=-1),   p=self.dropout, training=self.training)  # (B*h, N_ts, N_txt)
        probs_text = F.dropout(A_T.softmax(dim=-1), p=self.dropout, training=self.training)  # (B*h, N_txt, N_ts)

        # --- Aggregate the OTHER modality's values ---
        out_ts   = torch.bmm(probs_ts,   V_text)  # (B*h, N_ts, head_dim)  — TS gets text content
        out_text = torch.bmm(probs_text, V_ts)    # (B*h, N_txt, head_dim) — text gets TS content

        # --- Merge heads and project back to d_model ---
        def merge_heads(x: torch.Tensor, S: int) -> torch.Tensor:
            return x.view(B, self.num_heads, S, self.head_dim).transpose(1, 2).reshape(B, S, self.embed_dim)

        out_ts   = self.out_ts_proj(merge_heads(out_ts,   N_ts))    # (B, N_ts, d_model)
        out_text = self.out_text_proj(merge_heads(out_text, N_txt))  # (B, N_txt, d_model)

        return out_ts, out_text


class FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttentionLayer(nn.Module):
    """
    One layer of bidirectional cross-modal attention.
    Operates on flattened multi-scale TS features and text token features.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float,
                 drop_path: float = 0.0, layer_scale_init: float = 1e-4):
        super().__init__()
        # TS self-attention
        self.ts_self_attn = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.ts_self_norm = nn.LayerNorm(d_model)

        # Text self-attention
        self.txt_self_attn = nn.MultiheadAttention(d_model, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.txt_self_norm = nn.LayerNorm(d_model)

        # Bidirectional cross-attention (shared attention matrix for both directions)
        self.cross_attn = BidirectionalCrossAttention(
            d_model=d_model,
            embed_dim=ffn_dim // 2,   # matches GroundingDINO: dim_feedforward // 2
            num_heads=num_heads // 2,  # matches GroundingDINO: nhead // 2
            dropout=dropout,
        )
        self.ts_cross_norm  = nn.LayerNorm(d_model)   # pre-norm for TS before cross-attn
        self.txt_cross_norm = nn.LayerNorm(d_model)   # pre-norm for text before cross-attn

        # Layer scale: near-zero init makes cross-modal residual negligible at start
        self.gamma_ts   = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.gamma_text = nn.Parameter(layer_scale_init * torch.ones(d_model))

        # DropPath: stochastic depth on cross-modal residuals
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFNs
        self.ts_ffn = FFN(d_model, ffn_dim, dropout)
        self.ts_ffn_norm = nn.LayerNorm(d_model)

        self.txt_ffn = FFN(d_model, ffn_dim, dropout)
        self.txt_ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ts_feat: torch.Tensor,        # (B, N_ts, D)
        text_feat: torch.Tensor,      # (B, N_txt, D)
        ts_pos: torch.Tensor,         # (B, N_ts, D)  positional encoding for TS
        text_pos: torch.Tensor,       # (B, N_txt, D) positional encoding for text
        text_key_padding_mask: torch.Tensor,  # (B, N_txt) True = ignore (padding)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1. TS self-attention (pre-norm: Q, K, V all use the normed input)
        ts_normed = self.ts_self_norm(ts_feat)
        ts2, _ = self.ts_self_attn(ts_normed + ts_pos, ts_normed + ts_pos, ts_normed)
        ts_feat = ts_feat + ts2

        # 2. Text self-attention (pre-norm: Q, K, V all use the normed input)
        txt_normed = self.txt_self_norm(text_feat)
        txt2, _ = self.txt_self_attn(
            txt_normed + text_pos, txt_normed + text_pos, txt_normed,
            key_padding_mask=text_key_padding_mask,
        )
        text_feat = text_feat + txt2

        # 3 & 4. Bidirectional cross-attention (pre-norm, shared attention matrix)
        delta_ts, delta_text = self.cross_attn(
            self.ts_cross_norm(ts_feat),
            self.txt_cross_norm(text_feat),
            text_key_padding_mask,
        )
        ts_feat   = ts_feat   + self.drop_path(self.gamma_ts   * delta_ts)
        text_feat = text_feat + self.drop_path(self.gamma_text * delta_text)

        # 5. FFN on TS
        ts_feat = ts_feat + self.ts_ffn(self.ts_ffn_norm(ts_feat))

        # 6. FFN on text
        text_feat = text_feat + self.txt_ffn(self.txt_ffn_norm(text_feat))

        return ts_feat, text_feat


# ---------------------------------------------------------------------------
# Full Feature Enhancer
# ---------------------------------------------------------------------------

class FeatureEnhancer(nn.Module):
    """
    Stack of `enhancer_layers` CrossModalAttentionLayer layers.

    Takes multi-scale TS features (list of tensors) and text features,
    flattens multi-scale into a single sequence, runs bidirectional attention,
    and returns enhanced representations for both modalities.
    """

    def __init__(self, config: GroundingMomentConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                config.hidden_dim,
                config.enhancer_heads,
                config.enhancer_ffn_dim,
                config.enhancer_dropout,
                drop_path=config.enhancer_drop_path,
            )
            for _ in range(config.enhancer_layers)
        ])
        # Learned per-level embedding so identical local positions at different
        # scales produce distinct positional encodings (matches GroundingDINO).
        self.level_embed = nn.Parameter(
            torch.empty(config.num_feature_levels, config.hidden_dim)
        )
        nn.init.normal_(self.level_embed)

    def forward(
        self,
        ts_features: list[torch.Tensor],  # [(B,64,D),(B,32,D),(B,16,D),(B,8,D),(B,4,D),(B,2,D),(B,1,D)]
        text_feat: torch.Tensor,           # (B, T, D)
        text_mask: torch.Tensor,           # (B, T)  1=real, 0=padding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Returns:
            ts_feat:            (B, N_total, D)  enhanced flattened TS features
            text_feat:          (B, T, D)        enhanced text features
            ts_pos:             (B, N_total, D)  positional encoding (for decoder)
            level_start_index:  list of ints — where each scale level starts in dim 1
        """
        B = text_feat.shape[0]
        device = text_feat.device

        # -----------------------------------------------------------------------
        # Step 1: Flatten multi-scale TS features into one sequence.
        # The backbone produces features at multiple temporal resolutions
        # (e.g. 64, 32, 16 patches). Concatenate them along the sequence dim so
        # all scales are processed jointly by the attention layers.
        # A per-level learned embedding is added to positional encodings so the
        # model can distinguish tokens from different scales that share the same
        # local position index.
        # -----------------------------------------------------------------------
        ts_parts = []
        pos_parts = []
        level_start_index = []  # start index of each scale in the flattened sequence
        offset = 0

        for lvl, feat in enumerate(ts_features):
            n = feat.shape[1]                                              # number of patches at this scale
            level_start_index.append(offset)
            ts_parts.append(feat)                                          # (B, n, D)
            pos = sinusoidal_pos_encoding(n, self.config.hidden_dim, device)  # (1, n, D)
            pos = pos + self.level_embed[lvl].view(1, 1, -1)              # (1, n, D)  add scale-specific bias
            pos_parts.append(pos)
            offset += n

        ts_feat = torch.cat(ts_parts, dim=1)                              # (B, N_total, D)
        ts_pos  = torch.cat(pos_parts, dim=1).expand(B, -1, -1)          # (B, N_total, D)

        # -----------------------------------------------------------------------
        # Step 2: Build text positional encoding.
        # Standard sinusoidal encoding over token positions, generated once and
        # reused across all enhancer layers (positions don't change between layers).
        # -----------------------------------------------------------------------
        T = text_feat.shape[1]
        text_pos = sinusoidal_pos_encoding(T, self.config.hidden_dim, device).expand(B, -1, -1)  # (B, T, D)

        # -----------------------------------------------------------------------
        # Step 3: Convert padding mask convention.
        # text_mask uses 1=real token, 0=padding (BERT convention).
        # PyTorch MultiheadAttention expects True=ignore, so we flip it.
        # -----------------------------------------------------------------------
        text_key_padding_mask = (text_mask == 0)  # (B, T)  True = padding token

        # -----------------------------------------------------------------------
        # Step 4: Run stacked bidirectional cross-modal attention layers.
        # Each layer refines both ts_feat and text_feat by letting the two
        # modalities attend to each other. ts_pos and text_pos are injected as
        # additive biases to Q and K inside each layer (not updated themselves).
        # -----------------------------------------------------------------------
        for layer in self.layers:
            ts_feat, text_feat = layer(ts_feat, text_feat, ts_pos, text_pos,
                                       text_key_padding_mask)
            # ts_feat:   (B, N_total, D)
            # text_feat: (B, T, D)

        return ts_feat, text_feat, ts_pos, level_start_index
