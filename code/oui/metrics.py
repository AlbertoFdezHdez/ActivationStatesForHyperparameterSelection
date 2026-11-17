from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _finite_float(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def has_nonfinite(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_nonfinite(v) for v in value)
    return not _finite_float(value)


def activation_baselines(x: torch.Tensor, prefix: str = "") -> dict[str, float]:
    z = x.detach().float()
    if z.ndim > 2:
        z = z.flatten(1)
    eps = 1e-8
    active = z > eps
    p = active.float().mean(dim=0)
    entropy = -(p * torch.log2(p.clamp_min(eps)) + (1.0 - p) * torch.log2((1.0 - p).clamp_min(eps)))
    abs_z = z.abs()
    return {
        f"{prefix}activation_sparsity": float((abs_z <= eps).float().mean().item()),
        f"{prefix}dead_units_fraction": float((p <= eps).float().mean().item()),
        f"{prefix}activation_entropy": float(entropy.mean().item()),
        f"{prefix}saturated_units_fraction": float((p >= 1.0 - eps).float().mean().item()),
    }


def representation_stats(features: torch.Tensor, prefix: str = "", top_k: int = 8) -> dict[str, Any]:
    z = features.detach().float()
    if z.ndim > 2:
        z = z.flatten(1)
    if z.numel() == 0:
        return {}
    n = z.shape[0]
    std = z.std(dim=0, unbiased=False)
    norms = z.norm(dim=1)
    centered = z - z.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    weights = singular / (singular.sum() + 1e-12)
    effective_rank = torch.exp(-(weights * torch.log(weights + 1e-12)).sum())
    eig = singular.pow(2) / max(1, n - 1)
    eig_sorted = torch.sort(eig, descending=True).values
    positive = eig_sorted[eig_sorted > 1e-12]
    condition = eig_sorted[0] / positive[-1] if positive.numel() else torch.tensor(float("nan"), device=z.device)
    normalized = torch.nn.functional.normalize(z, dim=1)
    cosine = normalized @ normalized.T
    if n > 1:
        tri = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
        pairwise_mean = tri.mean()
        pairwise_std = tri.std(unbiased=False)
    else:
        pairwise_mean = torch.tensor(float("nan"), device=z.device)
        pairwise_std = torch.tensor(float("nan"), device=z.device)
    out: dict[str, Any] = {
        f"{prefix}representation_source": None,
        f"{prefix}embedding_norm_mean": float(norms.mean().item()),
        f"{prefix}embedding_norm_std": float(norms.std(unbiased=False).item()),
        f"{prefix}feature_std_mean": float(std.mean().item()),
        f"{prefix}feature_std_min": float(std.min().item()),
        f"{prefix}feature_variance_mean": float(z.var(dim=0, unbiased=False).mean().item()),
        f"{prefix}feature_std_lt_0_01_fraction": float((std < 0.01).float().mean().item()),
        f"{prefix}feature_std_lt_0_05_fraction": float((std < 0.05).float().mean().item()),
        f"{prefix}feature_std_lt_0_1_fraction": float((std < 0.1).float().mean().item()),
        f"{prefix}rankme": float(effective_rank.item()),
        f"{prefix}effective_rank": float(effective_rank.item()),
        f"{prefix}covariance_top_eigenvalues": [float(v.item()) for v in eig_sorted[:top_k]],
        f"{prefix}covariance_condition_proxy": float(condition.item()),
        f"{prefix}pairwise_cosine_mean": float(pairwise_mean.item()),
        f"{prefix}pairwise_cosine_std": float(pairwise_std.item()),
    }
    out.update(activation_baselines(z, prefix=prefix))
    return out


def timing_summary_aliases(timing_seconds: dict[str, float], total_time: float) -> dict[str, float]:
    return {
        "wall_time_total": total_time,
        "wall_time_train": float(timing_seconds.get("train_core_seconds", timing_seconds.get("learn_seconds", 0.0))),
        "wall_time_eval": float(
            timing_seconds.get(
                "full_validation_seconds",
                timing_seconds.get("ssl_probe_seconds", timing_seconds.get("eval_seconds", 0.0)),
            )
        ),
        "wall_time_oui_train": float(timing_seconds.get("oui_train_seconds", 0.0)),
        "wall_time_oui_probe": float(timing_seconds.get("oui_probe_seconds", 0.0)),
        "wall_time_metrics_extra": float(
            timing_seconds.get("scalar_metrics_seconds", 0.0)
            + timing_seconds.get("validation_extra_metrics_seconds", 0.0)
            + timing_seconds.get("knn_eval_seconds", 0.0)
            + timing_seconds.get("linear_probe_seconds", 0.0)
            + timing_seconds.get("other_overhead_seconds", 0.0)
        ),
    }


class PenultimateActivationTracker:
    def __init__(self, model: nn.Module) -> None:
        self.activation: torch.Tensor | None = None
        self.handle = None
        target = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                target = module
        if target is not None:
            self.handle = target.register_forward_pre_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            self.activation = inputs[0].detach()

    def stats(self, source: str = "penultimate_pre_classifier", prefix: str = "repr_") -> dict[str, Any]:
        if self.activation is None:
            return {f"{prefix}representation_source": source, f"{prefix}representation_available": False}
        stats = representation_stats(self.activation, prefix=prefix)
        stats[f"{prefix}representation_source"] = source
        stats[f"{prefix}representation_available"] = True
        return stats

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
