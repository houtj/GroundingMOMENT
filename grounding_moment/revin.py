"""
revin.py
Reversible Instance Normalization for Grounding MOMENT.

Ported from the official RevIN implementation (ts-kim/RevIN, ICLR 2022)
and MOMENT's internal copy (momentfm/models/layers/revin.py), with the
following adaptations:

  - Layout: accepts (B, L, C) — the project-wide convention.
    Transposes to (B, C, L) internally for stat computation, returns (B, L, C).
  - Mask-aware: pass an optional bool mask of shape (B, L) where True = observed.
    Defaults to all-observed when omitted.
  - Affine: learnable per-channel gamma (init 1) and beta (init 0),
    shapes (1, num_features, 1) in (B, C, L) space.
  - Robust mode (Tier B): swaps mean/std with median/MAD (scaled by 1.4826
    to match Gaussian std).  Unaffected by extreme outliers up to 50% contamination.
  - AMP-safe: statistics are computed in float32 regardless of input dtype,
    then the normalised result is cast back to the original dtype.
  - Stats (self.mean, self.stdev) are plain Python attributes recomputed on
    every norm call — they never persist across batches (matches official impl).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers (adapted from momentfm/models/layers/revin.py)
# ---------------------------------------------------------------------------

def _nanvar(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    mean = tensor.nanmean(dim=dim, keepdim=True)
    return (tensor - mean).square().nanmean(dim=dim, keepdim=keepdim)


def _nanstd(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return _nanvar(tensor, dim=dim, keepdim=keepdim).sqrt()


def _nanmedian(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Median ignoring NaN values (torch.nanmedian is available from PyTorch 1.8)."""
    # torch.nanmedian returns a named tuple; .values gives the median tensor.
    return torch.nanmedian(tensor, dim=dim, keepdim=keepdim).values


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Args:
        num_features: number of channels C (the last dimension of the (B, L, C) input).
        eps:          numerical stability constant added to the standard deviation.
        affine:       if True, learnable per-channel gamma and beta are applied
                      after normalization (and inverted before denormalization).
        robust:       if True, use median + MAD instead of mean + std for statistics.
                      More robust to extreme outliers.  Does not affect the affine path.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        robust: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.robust = robust

        if self.affine:
            # Per-channel learnable scale and bias.
            # Shape (1, C, 1) matches the (B, C, L) layout used internally.
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_features, 1))

        # Populated by _get_statistics; plain attrs so they are never persisted.
        self.mean: torch.Tensor | None  = None
        self.stdev: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, L, C)  — input in project-wide layout.
            mode: 'norm'   — compute stats, normalize, (optionally) apply affine.
                  'denorm' — invert affine (if any), denormalize using cached stats.
            mask: (B, L) bool, True = observed timestep.  Defaults to all True.
        Returns:
            (B, L, C) tensor in the same dtype as the input.
        """
        if mode == "norm":
            return self._norm(x, mask)
        elif mode == "denorm":
            return self._denorm(x)
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_statistics(
        self, x_bcl: torch.Tensor, mask: torch.Tensor | None
    ) -> None:
        """
        Compute and cache per-instance, per-channel statistics.

        x_bcl : (B, C, L) in float32
        mask  : (B, L) bool or None
        """
        B, C, L = x_bcl.shape

        # Build (B, C, L) NaN mask: NaN where unobserved, x elsewhere.
        if mask is None:
            masked = x_bcl
        else:
            # mask: (B, L) → (B, 1, L) → (B, C, L)
            obs = mask.unsqueeze(1).expand(B, C, L)
            masked = torch.where(obs, x_bcl, torch.tensor(float("nan"), device=x_bcl.device, dtype=x_bcl.dtype))

        if self.robust:
            # Median (location) and MAD scaled to match Gaussian std (scale).
            med = _nanmedian(masked, dim=-1, keepdim=True)             # (B, C, 1)
            abs_dev = (masked - med).abs()
            # 1.4826 ≈ 1 / Φ^{-1}(3/4): makes MAD consistent with σ under normality.
            mad = 1.4826 * _nanmedian(abs_dev, dim=-1, keepdim=True)  # (B, C, 1)
            self.mean  = med.detach()
            self.stdev = (mad + self.eps).detach()
        else:
            self.mean  = x_bcl.nanmean(dim=-1, keepdim=True).detach()        # (B, C, 1)
            self.stdev = (_nanstd(masked, dim=-1, keepdim=True) + self.eps).detach()

    def _norm(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        orig_dtype = x.dtype

        # Work in float32 for numerical stability under autocast.
        x_f = x.float()                          # (B, L, C)
        x_bcl = x_f.permute(0, 2, 1)            # (B, C, L)

        self._get_statistics(x_bcl, mask)

        x_bcl = (x_bcl - self.mean) / self.stdev

        if self.affine:
            x_bcl = x_bcl * self.gamma + self.beta

        # Back to (B, L, C) and original dtype.
        return x_bcl.permute(0, 2, 1).to(orig_dtype)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.stdev is None:
            raise RuntimeError(
                "RevIN.denorm called before norm — no cached statistics available."
            )
        orig_dtype = x.dtype
        x_f = x.float()
        x_bcl = x_f.permute(0, 2, 1)            # (B, C, L)

        if self.affine:
            x_bcl = (x_bcl - self.beta) / (self.gamma + self.eps * self.eps)

        x_bcl = x_bcl * self.stdev + self.mean

        return x_bcl.permute(0, 2, 1).to(orig_dtype)
