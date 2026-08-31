"""
text_encoder.py
BERT text encoder wrapper with optional numeric-token injection (Variant 3).

Input:  input_ids (B, T), attention_mask (B, T)
        + optional: num_values (B, T), num_mask (B, T), revin statistics

Output: token features (B, T, hidden_dim), attention_mask (B, T),
        v_norm (B, T) or None, pred_v (B, T) or None

When config.use_numeric_tokens is True:
  - A ScalarEncoder maps RevIN-normalised numeric values to (B, T, hidden_dim)
    via log-spaced Fourier features + 2-layer MLP.
  - The embedding is gated by num_mask and added as a residual to the
    projected BERT features, just before they enter the Feature Enhancer.
  - A ScalarHead (optional) predicts the normalised value from the augmented
    features, enabling an MSE auxiliary warm-up loss.
"""

import torch
import torch.nn as nn
from transformers import BertModel

from .config import GroundingMomentConfig


# ---------------------------------------------------------------------------
# Scalar encoding sub-modules
# ---------------------------------------------------------------------------

class ScalarEncoder(nn.Module):
    """
    Encodes a scalar value v into a hidden_dim vector using log-spaced
    Fourier features + sign + log-magnitude, fed through a 2-layer MLP.

    Input features for each scalar v:
        [sin(2^0 * v), cos(2^0 * v), ..., sin(2^(F-1) * v), cos(2^(F-1) * v),
         sign(v), log(1 + |v|)]
    Total input dim: 2*F + 2.
    """

    def __init__(self, d_out: int, n_freq: int, mlp_hidden: int) -> None:
        super().__init__()
        # log-spaced frequencies: 1, 2, 4, …, 2^(n_freq-1)
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())  # (F,)
        in_dim = 2 * n_freq + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_out),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (B, T)  — RevIN-normalised values (0 outside num positions)
        Returns:
            (B, T, d_out)
        """
        ph   = v.unsqueeze(-1) * self.freqs                # (B, T, F)
        feat = torch.cat([
            ph.sin(),                                       # (B, T, F)
            ph.cos(),                                       # (B, T, F)
            torch.sign(v).unsqueeze(-1),                   # (B, T, 1)
            torch.log1p(v.abs()).unsqueeze(-1),             # (B, T, 1)
        ], dim=-1)                                          # (B, T, 2F+2)
        return self.mlp(feat)                               # (B, T, d_out)


class ScalarHead(nn.Module):
    """
    Lightweight regression head: projects hidden_dim → scalar prediction.
    Zero-initialised so early training does not perturb text features.
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_in)
        Returns:
            (B, T)
        """
        return self.proj(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main BERT text encoder
# ---------------------------------------------------------------------------

class BERTTextEncoder(nn.Module):
    def __init__(self, config: GroundingMomentConfig):
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.text_model)

        if config.freeze_text_encoder:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        # Project BERT 768-d to shared hidden dim
        self.proj = nn.Linear(config.text_embed_dim, config.hidden_dim)

        # --- Numeric token mechanism (Variant 3) ---
        # Instantiated only when use_numeric_tokens=True; otherwise these
        # attributes do not exist and old checkpoints load cleanly.
        if config.use_numeric_tokens:
            self.scalar_encoder = ScalarEncoder(
                d_out=config.hidden_dim,
                n_freq=config.scalar_n_freq,
                mlp_hidden=config.scalar_mlp_hidden,
            )
            if config.scalar_aux_loss:
                self.scalar_head = ScalarHead(d_in=config.hidden_dim)

        # num_token_id is set externally by GroundingMOMENT.__init__ after the
        # tokenizer is available.  Default -1 means "not set yet".
        self.num_token_id: int = -1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_values: torch.Tensor | None = None,    # (B, T) raw float values
        num_mask: torch.Tensor | None = None,      # (B, T) bool
        revin_mean: torch.Tensor | None = None,    # (B, 1, 1)
        revin_stdev: torch.Tensor | None = None,   # (B, 1, 1)
        revin_gamma: torch.Tensor | None = None,   # (1, 1, 1) learnable scale
        revin_beta: torch.Tensor | None = None,    # (1, 1, 1) learnable shift
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            input_ids:      (B, T)
            attention_mask: (B, T)  1 = real token, 0 = padding
            num_values:     (B, T)  raw numeric values at [NUM] positions; 0 elsewhere
            num_mask:       (B, T)  bool, True at [NUM] positions
            revin_mean:     (B, 1, 1)  per-sample RevIN mean
            revin_stdev:    (B, 1, 1)  per-sample RevIN std
            revin_gamma:    (1, 1, 1)  RevIN affine scale (or None)
            revin_beta:     (1, 1, 1)  RevIN affine shift (or None)

        Returns:
            token_features: (B, T, hidden_dim)
            attention_mask: (B, T)  — passed through unchanged
            v_norm:         (B, T)  RevIN-normalised values, or None
            pred_v:         (B, T)  ScalarHead prediction, or None
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_features = outputs.last_hidden_state   # (B, T, 768)
        text_feat      = self.proj(token_features)   # (B, T, hidden_dim)

        v_norm = None
        pred_v = None

        if (self.config.use_numeric_tokens
                and num_mask is not None
                and num_mask.any()
                and revin_mean is not None
                and revin_stdev is not None):

            # RevIN-normalise the raw values
            mu    = revin_mean.squeeze(-1).squeeze(-1)    # (B,)
            sigma = revin_stdev.squeeze(-1).squeeze(-1)   # (B,)
            v_norm = (num_values - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)  # (B, T)

            # Apply RevIN affine if available (matches the transformation
            # applied to the time series before MOMENT, ensuring that text
            # numeric values and TS patches live in the same normalised space)
            if revin_gamma is not None and revin_beta is not None:
                v_norm = v_norm * revin_gamma.view(1, 1) + revin_beta.view(1, 1)

            # Zero-out positions that are not NUM tokens (defensive)
            v_norm = torch.where(num_mask, v_norm, torch.zeros_like(v_norm))

            # Fourier-feature residual injection
            scalar_emb = self.scalar_encoder(v_norm)       # (B, T, hidden_dim)
            gate       = num_mask.unsqueeze(-1).float()    # (B, T, 1)
            text_feat  = text_feat + gate * scalar_emb

        # ScalarHead auxiliary prediction (for MSE warm-up loss)
        if (self.config.use_numeric_tokens
                and self.config.scalar_aux_loss
                and hasattr(self, "scalar_head")):
            pred_v = self.scalar_head(text_feat)           # (B, T)

        return text_feat, attention_mask, v_norm, pred_v
