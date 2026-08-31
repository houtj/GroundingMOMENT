"""
backbone.py
MOMENT time-series backbone wrapper.

Input:  (B, 512, 1)  — univariant time series as 2-D (time, channel)
Output: list of multi-scale features [(B, 64, D), (B, 32, D), (B, 16, D)]
        where D = config.hidden_dim

Extra feature levels (L1, L2) are produced by learnable stride-2 Conv1d layers
applied to the previous level — the same strategy GroundingDINO uses to
generate extra backbone levels from strided 2-D convolutions.

When config.use_ts_fourier=True a PatchFourierEncoder is attached (after both
backbone and text encoder are constructed in GroundingMOMENT) via
attach_patch_fourier().  It encodes per-patch summary statistics of the
RevIN-normalized signal through the SAME ScalarEncoder used by [NUM] text
tokens and adds the result as a layer-scaled residual to the level-0 patches.

When config.use_change_point_features=True a ChangePointEncoder is instantiated
per feature level and applied after level_norm + level_embed.  Each encoder
computes adjacent-patch differences and produces a layer-scaled feature residual
and a per-patch logit.  Position 0 of every level receives a structural hard
zero for both outputs (no MLP pass), distinguishing "no prior context" from a
genuinely stable interior region.  The per-level logits are concatenated to
shape (B, N_total) and returned as cp_score alongside the feature list.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline

from .config import GroundingMomentConfig
from .revin import RevIN


# ---------------------------------------------------------------------------
# TS-side Fourier encoder (cross-modal aligned)
# ---------------------------------------------------------------------------

class PatchFourierEncoder(nn.Module):
    """
    Encode per-patch summary statistics (mean, std, max, min) of the
    RevIN-normalized signal through the SHARED text-side ScalarEncoder,
    fuse them to hidden_dim, and apply a learnable layer-scale gate.

    The ScalarEncoder is held by reference inside a plain Python list so
    that PyTorch's module-registration machinery skips it — the canonical
    owner remains BERTTextEncoder, ensuring no duplicate entries in
    state_dict() or parameters().  Only the fuse Linear, LayerNorm, and
    gamma Parameter are registered as submodule/parameter of this module.
    """

    def __init__(
        self,
        scalar_encoder: nn.Module,
        d_out: int,
        patch_size: int = 8,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        # List wrapper prevents nn.Module.__setattr__ from registering this
        # as a submodule — the weights live only under text_encoder.scalar_encoder.
        self._enc_ref: list = [scalar_encoder]
        self.patch_size = patch_size

        # 4 stats × d_out → d_out
        self.fuse = nn.Linear(4 * d_out, d_out)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

        self.norm = nn.LayerNorm(d_out)

        # Per-channel learnable scale; init 0 so the residual is exactly zero
        # at step 0, preserving the pre-feature behaviour during early training.
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(d_out))

    @property
    def scalar_encoder(self) -> nn.Module:
        return self._enc_ref[0]

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, L, 1)  RevIN-normalized time series
        Returns:
            (B, n_patches, d_out) — ready to add as residual to MOMENT patches
        """
        B, L, _ = x_norm.shape
        n_patches = L // self.patch_size
        x = x_norm.squeeze(-1)                               # (B, L)
        pw = x.view(B, n_patches, self.patch_size)           # (B, P, S)

        p_mean = pw.mean(dim=-1)                             # (B, P)
        p_std  = pw.std(dim=-1, unbiased=False)              # (B, P)
        p_max  = pw.max(dim=-1).values                       # (B, P)
        p_min  = pw.min(dim=-1).values                       # (B, P)

        enc = self.scalar_encoder
        feat = torch.cat([enc(p_mean), enc(p_std),
                          enc(p_max),  enc(p_min)], dim=-1)  # (B, P, 4D)
        feat = self.fuse(feat)                               # (B, P, D)
        feat = self.norm(feat)
        return feat * self.gamma                             # (B, P, D)


class ChangePointEncoder(nn.Module):
    """
    Per-patch change-point feature encoder.

    Computes adjacent-patch differences to detect where the time-series
    distribution shifts, then exposes two outputs:

      cp_features: (B, P, D)  — layer-scaled residual to add to patch embeddings
      cp_score:    (B, P)     — per-patch logit, used by:
                                  B (BCE aux loss, supervised by cp_targets)
                                  C (additive bias on language-guided query selection)

    Difference computation
    ----------------------
    diff[i] = patch[i] - patch[i-1]  for i in 1..P-1  (P-1 real differences).
    Position 0 has no prior context and receives a structural hard zero for both
    cp_features and cp_score — the MLP is never evaluated there.  This cleanly
    separates "no prior context" (position 0) from "stable interior" (a real
    near-zero diff at some interior position i).

    Networks
    --------
    feat_mlp  : D -> cp_hidden_dim -> D   produces cp_features (pre-gate)
    score_mlp : D -> cp_hidden_dim -> 1   produces cp_score (raw logit)
    gamma      : (D,) learnable layer scale, init=cp_layer_scale (default 0)
    """

    def __init__(self, d_model: int, cp_hidden_dim: int,
                 layer_scale_init: float = 0.0) -> None:
        super().__init__()
        self.feat_mlp = nn.Sequential(
            nn.Linear(d_model, cp_hidden_dim),
            nn.ReLU(),
            nn.Linear(cp_hidden_dim, d_model),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, cp_hidden_dim),
            nn.ReLU(),
            nn.Linear(cp_hidden_dim, 1),
        )
        self.norm  = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        for mlp in (self.feat_mlp, self.score_mlp):
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (B, P, D)  — patch embeddings (post level_norm + level_embed)
        Returns:
            cp_features: (B, P, D)  layer-scaled feature residual; index 0 is exact zero
            cp_score:    (B, P)     per-patch change-point logit;  index 0 is exact zero
        """
        diff = patches[:, 1:, :] - patches[:, :-1, :]          # (B, P-1, D)

        feat  = self.norm(self.feat_mlp(diff)) * self.gamma     # (B, P-1, D)
        score = self.score_mlp(diff).squeeze(-1)                # (B, P-1)

        # Prepend structural zeros at index 0: "no prior context" ≠ "stable interior"
        cp_features = F.pad(feat,  (0, 0, 1, 0))               # (B, P, D)
        cp_score    = F.pad(score, (1, 0))                      # (B, P)
        return cp_features, cp_score


class _IdentityNorm(nn.Module):
    """
    Drop-in replacement for MOMENT's internal RevIN when we want to bypass it.

    MOMENT calls self.normalizer(x=..., mask=..., mode=...) with keyword arguments
    that nn.Identity.forward() does not accept.  This wrapper passes `x` through
    unchanged regardless of mode or mask.
    """

    def forward(self, x: torch.Tensor, mode: str = "norm", mask=None, **kwargs) -> torch.Tensor:
        return x


class MOMENTBackbone(nn.Module):
    def __init__(self, config: GroundingMomentConfig):
        super().__init__()
        self.config = config

        # Load MOMENT in embedding mode
        self.moment = MOMENTPipeline.from_pretrained(
            config.moment_model,
            model_kwargs={"task_name": "embedding"},
        )
        self.moment.init()

        # Tier A / B: external learnable RevIN applied before feeding MOMENT.
        # Must be instantiated after self.moment.init() so the Identity() swap below
        # is on the fully-initialised normalizer attribute.
        if config.use_revin:
            self.revin = RevIN(
                num_features=1,
                eps=config.revin_eps,
                affine=config.revin_affine,
                robust=config.revin_robust,
            )

        # Replace MOMENT's frozen, non-affine internal RevIN with a no-op so that
        # our external RevIN's statistics (and optional affine) are not overwritten.
        # Safe: MOMENT's normalizer has affine=False — zero parameters are removed.
        # We use _IdentityNorm instead of nn.Identity() because MOMENT calls
        # self.normalizer(x=..., mask=..., mode=...) with keyword arguments that
        # nn.Identity.forward() does not accept.
        if config.revin_bypass_moment_norm:
            self.moment.normalizer = _IdentityNorm()

        if config.freeze_backbone:
            for p in self.moment.parameters():
                p.requires_grad_(False)

        # Project from MOMENT embed dim to shared hidden dim (level 0)
        self.proj = nn.Linear(config.moment_embed_dim, config.hidden_dim)

        # Per-level layer norm
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_feature_levels)
        ])

        # Learnable level embeddings to distinguish scales
        self.level_embeds = nn.Embedding(config.num_feature_levels, config.hidden_dim)

        # Learnable stride-2 Conv1d layers for extra feature levels.
        # Each conv halves the temporal dimension, mirroring GroundingDINO's
        # strided 3×3 Conv2d used to generate extra image feature levels.
        # Input/output are in (B, D, L) layout; we transpose around the call.
        self.extra_level_convs = nn.ModuleList([
            nn.Conv1d(
                config.hidden_dim, config.hidden_dim,
                kernel_size=3, stride=2, padding=1,
            )
            for _ in range(config.num_feature_levels - 1)
        ])

        self._init_extra_convs()

        # A: change-point encoders — one per feature level, applied after level_norm +
        # level_embed so each encoder sees the same representation space that the
        # feature enhancer and decoder will attend to.
        if config.use_change_point_features:
            self.change_points = nn.ModuleList([
                ChangePointEncoder(
                    d_model=config.hidden_dim,
                    cp_hidden_dim=config.cp_hidden_dim,
                    layer_scale_init=config.cp_layer_scale,
                )
                for _ in range(config.num_feature_levels)
            ])

    def _init_extra_convs(self):
        for conv in self.extra_level_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def attach_patch_fourier(self, scalar_encoder: nn.Module) -> None:
        """Attach a PatchFourierEncoder that shares scalar_encoder with the
        text encoder.  Called by GroundingMOMENT.__init__ after both encoders
        have been constructed so the shared reference is valid."""
        self.patch_fourier = PatchFourierEncoder(
            scalar_encoder=scalar_encoder,
            d_out=self.config.hidden_dim,
            patch_size=self.config.seq_len // self.config.num_patches,
            layer_scale_init=self.config.ts_fourier_layer_scale,
        )

    def forward(
        self, time_series: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
        Args:
            time_series: (B, 512, 1)  — 2-D univariant time series
        Returns:
            features:  list of tensors [(B,64,D),(B,32,D),(B,16,D),(B,8,D),(B,4,D),(B,2,D),(B,1,D)]
            cp_score:  (B, N_total) per-patch change-point logits concatenated across
                       all feature levels (e.g. 64+32+16+8+4+2+1=127 for default config), or
                       None when use_change_point_features=False.
                       Index 0 of every level segment is a structural hard zero.
        """
        B = time_series.shape[0]

        # Apply external RevIN (Tier A / B) before handing off to MOMENT.
        # Normalizes (B, L, C) in-place per instance and per channel; if
        # revin_bypass_moment_norm=True, MOMENT's internal normalizer is Identity()
        # so RevIN's statistics are preserved through the encoder.
        x_in = time_series  # (B, 512, 1)
        if hasattr(self, "revin"):
            x_in = self.revin(x_in, mode="norm")  # (B, 512, 1)

        # MOMENT expects (B, n_channels, seq_len); transpose from (B, L, C) → (B, C, L)
        x = x_in.transpose(1, 2).float()  # (B, 1, 512)

        # All-ones input mask: no masked timesteps
        mask = torch.ones(B, self.config.seq_len,
                          dtype=torch.bool, device=time_series.device)

        output = self.moment.embed(x_enc=x, input_mask=mask, reduction="none")
        # output.embeddings: (B, n_channels, num_patches, moment_embed_dim)
        # Single channel → squeeze to (B, num_patches, moment_embed_dim)
        patches = output.embeddings.squeeze(1)  # (B, 64, 1024)

        # Project to hidden dim → level 0
        patches = self.proj(patches)  # (B, 64, D)

        # TS-side Fourier residual: per-patch summary statistics (mean, std,
        # max, min) of the RevIN-normalized signal are encoded through the SAME
        # ScalarEncoder used for [NUM] text tokens.  gamma is init-0 so the
        # residual is exactly zero at the start of training.
        if self.config.use_ts_fourier and hasattr(self, "patch_fourier"):
            patches = patches + self.patch_fourier(x_in)  # (B, 64, D)

        features = []
        cp_scores_per_level: list[torch.Tensor] = []
        prev = patches  # (B, L, D), current level in (batch, length, channel) layout

        for level in range(self.config.num_feature_levels):
            if level == 0:
                feat = prev  # (B, 64, D)
            else:
                # Apply stride-2 Conv1d on the *previous* level.
                # Conv1d expects (B, C, L) → transpose, convolve, transpose back.
                conv_in = prev.transpose(1, 2)                         # (B, D, L_prev)
                conv_out = self.extra_level_convs[level - 1](conv_in)  # (B, D, L_prev//2)
                feat = conv_out.transpose(1, 2)                        # (B, L_prev//2, D)

            feat = self.level_norms[level](feat)
            feat = feat + self.level_embeds.weight[level]  # broadcast level embedding

            # A: change-point encoder applied per level after level_norm + level_embed.
            # gamma is init-0 so the feature residual is zero at step 0.
            # Position 0 of every level receives a structural hard zero (no MLP pass).
            if self.config.use_change_point_features and hasattr(self, "change_points"):
                cp_feat_k, cp_score_k = self.change_points[level](feat)  # (B,L,D), (B,L)
                feat = feat + cp_feat_k
                cp_scores_per_level.append(cp_score_k)

            prev = feat
            features.append(feat)

        cp_score: torch.Tensor | None = None
        if cp_scores_per_level:
            cp_score = torch.cat(cp_scores_per_level, dim=1)  # (B, N_total)

        return features, cp_score  # [(B,64,D), (B,32,D), (B,16,D)], (B, N_total) or None
