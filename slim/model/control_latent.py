"""Optional factorized visual bottleneck; the policy only receives control factors.

The reconstruction path is training-only. DINO is frozen in the initial recipe,
so the reconstruction target cannot move to accommodate a collapsed bottleneck.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def cross_correlation_loss(control: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Feature decorrelation, not per-example vector orthogonality."""
    x = control.flatten(0, -2).float()
    y = residual.flatten(0, -2).float()
    if x.shape[0] != y.shape[0]:
        raise ValueError("Control and residual factors must describe the same samples")
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    x = x / (x.square().mean(0, keepdim=True) + 1e-4).sqrt()
    y = y / (y.square().mean(0, keepdim=True) + 1e-4).sqrt()
    return (x.T @ y / max(x.shape[0], 1)).square().mean()


def batch_variance_loss(tokens: torch.Tensor, floor: float) -> torch.Tensor:
    """Measure variation across examples at each slot, excluding fixed query diversity.

Statistics are local to each device. Use at least two examples per device for
training; gradient accumulation does not enlarge this statistics batch.
"""
    if tokens.shape[0] < 2:
        return tokens.float().sum() * 0.0
    variance = tokens.float().var(dim=0, unbiased=False)
    return F.relu(float(floor) - (variance + 1e-4).sqrt()).mean()


class FactorizedControlEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_views: int, patches_per_view: int,
                 language_dim: int, config) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_views = int(num_views)
        self.patches_per_view = int(patches_per_view)
        self.control_dim = int(config.get("control_dim", 192))
        residual_dim = int(config.get("residual_dim", 192))
        self.tokens_per_view = int(config.get("tokens_per_view", 0))
        if min(self.control_dim, residual_dim, num_views, patches_per_view) <= 0:
            raise ValueError("Factor dimensions, view count and patch count must be positive")
        if not 0 <= self.tokens_per_view <= self.patches_per_view:
            raise ValueError("tokens_per_view must be zero (dense) or <= patches_per_view")
        self.output_tokens = num_views * (self.tokens_per_view or patches_per_view)
        self.reconstruction_weight = float(config.get("reconstruction_weight", 1.0))
        self.decorrelation_weight = float(config.get("decorrelation_weight", 0.0))
        self.variance_weight = float(config.get("variance_weight", 0.1))
        self.variance_floor = float(config.get("variance_floor", 0.5))
        if min(self.reconstruction_weight, self.decorrelation_weight,
               self.variance_weight, self.variance_floor) < 0:
            raise ValueError("Regularization weights and variance floor must be nonnegative")

        self.input_norm = nn.LayerNorm(feature_dim)
        self.control_proj = nn.Linear(feature_dim, self.control_dim)
        self.residual_proj = nn.Linear(feature_dim, residual_dim)
        # Preserve the original MoT input/output width for warm-start compatibility.
        # This projection sees c only; u has no path into policy/IDM/FDM.
        self.control_readout = nn.Sequential(
            nn.Linear(self.control_dim, feature_dim), nn.LayerNorm(feature_dim)
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(self.control_dim + residual_dim, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        if self.tokens_per_view:
            heads = int(config.get("num_heads", 3))
            if heads <= 0 or self.control_dim % heads:
                raise ValueError("control_dim must be divisible by tokenizer num_heads")
            side = math.isqrt(patches_per_view)
            if side * side != patches_per_view:
                raise ValueError("The control tokenizer requires a square patch grid")
            axis = torch.linspace(-1, 1, side)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            self.register_buffer("patch_xy", torch.stack((xx, yy), -1).reshape(-1, 2))
            self.position_proj = nn.Linear(2, self.control_dim)
            self.view_embedding = nn.Embedding(num_views, self.control_dim)
            self.queries = nn.Parameter(torch.randn(
                num_views, self.tokens_per_view, self.control_dim
            ) * 0.02)
            self.language_proj = nn.Linear(language_dim, self.control_dim, bias=False)
            self.pool_attention = nn.MultiheadAttention(self.control_dim, heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(self.control_dim)
            self.pool_ffn = nn.Sequential(
                nn.Linear(self.control_dim, 2 * self.control_dim), nn.GELU(),
                nn.Linear(2 * self.control_dim, self.control_dim),
            )
            self.reconstruct_attention = nn.MultiheadAttention(
                self.control_dim, heads, batch_first=True
            )

    def forward(self, features: torch.Tensor, language=None, language_mask=None,
                *, compute_aux: bool = False):
        b, n, d = features.shape
        if n != self.num_views * self.patches_per_view or d != self.feature_dim:
            raise ValueError(f"Unexpected dense feature shape: {tuple(features.shape)}")
        normalized = self.input_norm(features)
        dense_control = self.control_proj(normalized)
        control = dense_control
        reconstructed_control = dense_control
        if self.tokens_per_view:
            v, p, k, c = self.num_views, self.patches_per_view, self.tokens_per_view, self.control_dim
            dense = dense_control.reshape(b, v, p, c)
            positions = self.position_proj(self.patch_xy.to(dense.dtype))
            view = self.view_embedding.weight
            keys = dense + positions[None, None] + view[None, :, None]
            queries = self.queries[None].expand(b, -1, -1, -1)
            queries = queries + view[None, :, None]
            if language is not None:
                language = language.to(dense.dtype)
                if language_mask is None:
                    pooled_language = language.mean(1)
                else:
                    mask = language_mask.to(language.dtype).unsqueeze(-1)
                    pooled_language = (language * mask).sum(1) / mask.sum(1).clamp_min(1)
                queries = queries + self.language_proj(pooled_language)[:, None, None]
            queries = queries.reshape(b * v, k, c)
            pooled, _ = self.pool_attention(
                queries, keys.reshape(b * v, p, c), dense.reshape(b * v, p, c),
                need_weights=False,
            )
            slots = queries + pooled
            slots = slots + self.pool_ffn(self.pool_norm(slots))
            control = slots.reshape(b, v * k, c)
            if compute_aux:
                # Dense queries reconstruct c from compressed slots, never from dense c.
                patch_queries = positions[None, None] + view[None, :, None]
                patch_queries = patch_queries.expand(b, -1, -1, -1).reshape(b * v, p, c)
                decoded, _ = self.reconstruct_attention(
                    patch_queries, slots, slots, need_weights=False
                )
                reconstructed_control = decoded.reshape(b, v * p, c)

        output = self.control_readout(control)
        if not compute_aux:
            return output, {}
        residual = self.residual_proj(normalized)
        reconstructed = self.reconstruction(torch.cat((reconstructed_control, residual), -1))
        # The target is the frozen backbone feature, not the trainable input_norm output.
        target = F.layer_norm(features.detach().float(), (self.feature_dim,))
        reconstruction = F.mse_loss(reconstructed.float(), target)
        decorrelation = cross_correlation_loss(dense_control, residual)
        variance = (batch_variance_loss(control, self.variance_floor)
                    + batch_variance_loss(residual, self.variance_floor)) * 0.5
        total = (self.reconstruction_weight * reconstruction
                 + self.decorrelation_weight * decorrelation
                 + self.variance_weight * variance)
        return output, {
            "representation_loss": total,
            "reconstruction_loss": reconstruction,
            "decorrelation_loss": decorrelation,
            "variance_loss": variance,
        }
