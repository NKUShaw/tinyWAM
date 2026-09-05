from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def _flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens is None:
        raise ValueError("tokens must not be None.")
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens shape [B, Q, D], got {tuple(tokens.shape)}.")
    bsz, num_tokens, dim = tokens.shape
    if bsz <= 0 or num_tokens <= 0 or dim <= 0:
        raise ValueError(f"Invalid token shape {tuple(tokens.shape)}.")
    return tokens.reshape(bsz * num_tokens, dim)


def compute_batch_std_metrics(tokens: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Variance-collapse signals from batch-token feature spread."""
    x = _flatten_tokens(tokens).float()
    std_per_dim = x.std(dim=0, unbiased=False)
    var_per_dim = std_per_dim.pow(2)
    low_var_ratio = (var_per_dim < eps).float().mean()
    return {
        "batch_std_mean": float(std_per_dim.mean().item()),
        "batch_std_min": float(std_per_dim.min().item()),
        "batch_std_max": float(std_per_dim.max().item()),
        "low_var_dim_ratio": float(low_var_ratio.item()),
    }


def compute_spectral_metrics(tokens: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Rank-collapse signals from covariance spectrum."""
    x = _flatten_tokens(tokens).float()
    n = x.shape[0]
    if n < 2:
        return {
            "effective_rank": 0.0,
            "svd_entropy": 0.0,
            "top1_energy_ratio": 0.0,
            "top5_energy_ratio": 0.0,
        }

    x_centered = x - x.mean(dim=0, keepdim=True)
    # Use SVD on [N, D] instead of eigvalsh on [D, D] covariance.
    # When query tokens collapse, cov is rank-deficient / ill-conditioned and
    # eigvalsh often fails even though monitoring should remain best-effort.
    s = torch.linalg.svdvals(x_centered)
    eigvals = (s * s) / max(n - 1, 1)
    eigvals = eigvals.clamp_min(0.0)
    total = eigvals.sum().clamp_min(eps)
    p = eigvals / total

    entropy = -(p * (p + eps).log()).sum()
    effective_rank = entropy.exp()

    sorted_vals, _ = torch.sort(eigvals, descending=True)
    top1 = sorted_vals[:1].sum() / total
    top5 = sorted_vals[: min(5, sorted_vals.numel())].sum() / total

    return {
        "effective_rank": float(effective_rank.item()),
        "svd_entropy": float(entropy.item()),
        "top1_energy_ratio": float(top1.item()),
        "top5_energy_ratio": float(top5.item()),
    }


def compute_collapse_metrics(tokens: torch.Tensor, include_svd: bool, eps: float = 1e-12) -> Dict[str, float]:
    metrics = compute_batch_std_metrics(tokens=tokens, eps=eps)
    if include_svd:
        metrics.update(compute_spectral_metrics(tokens=tokens, eps=eps))
    return metrics


def compute_query_token_diversity_metrics(tokens: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Within-sample diversity among Q query tokens shaped [B, Q, D].

    Batch-level ``effective_rank`` can look healthy even when all Q tokens inside
    each sample are identical.  These metrics target that failure mode directly.
    """
    if tokens is None:
        raise ValueError("tokens must not be None.")
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens shape [B, Q, D], got {tuple(tokens.shape)}.")
    x = tokens.float()
    batch_size, num_tokens, _ = x.shape
    if batch_size <= 0 or num_tokens <= 0:
        raise ValueError(f"Invalid token shape {tuple(tokens.shape)}.")
    if num_tokens < 2:
        return {
            "query_inter_token_cosine_mean": 1.0,
            "query_inter_token_cosine_max": 1.0,
            "query_effective_rank_mean": 1.0,
            "query_effective_rank_min": 1.0,
        }

    x_norm = F.normalize(x, dim=-1, eps=eps)
    sim = torch.einsum("bqd,bkd->bqk", x_norm, x_norm)
    idx = torch.triu_indices(num_tokens, num_tokens, offset=1, device=x.device)
    pairwise = sim[:, idx[0], idx[1]]
    per_sample_ranks = [
        compute_spectral_metrics(x[i : i + 1], eps=eps)["effective_rank"] for i in range(batch_size)
    ]
    return {
        "query_inter_token_cosine_mean": float(pairwise.mean().item()),
        "query_inter_token_cosine_max": float(pairwise.max().item()),
        "query_effective_rank_mean": float(sum(per_sample_ranks) / len(per_sample_ranks)),
        "query_effective_rank_min": float(min(per_sample_ranks)),
    }


def build_monitor_collapse_payload(
    loss_dict: Mapping[str, torch.Tensor | None],
    *,
    include_svd: bool = True,
) -> Dict[str, float]:
    """Flatten past/future monitor latents into wandb-friendly collapse metrics."""
    payload: Dict[str, float] = {}
    for name in ("past", "future"):
        tokens = loss_dict.get(f"monitor/{name}_vl_embs")
        if tokens is None:
            continue
        try:
            for key, value in compute_collapse_metrics(tokens, include_svd=include_svd).items():
                payload[f"collapse/{name}/{key}"] = value
            for key, value in compute_query_token_diversity_metrics(tokens).items():
                payload[f"collapse/{name}/{key}"] = value
        except Exception:
            # Monitoring must never take down training.
            payload[f"collapse/{name}/metrics_failed"] = 1.0
    return payload
