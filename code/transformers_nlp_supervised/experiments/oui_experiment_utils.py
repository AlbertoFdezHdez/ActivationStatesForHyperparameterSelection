from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
SHARED_CODE_ROOT = CODE_ROOT.parent
REPO_ROOT = SHARED_CODE_ROOT.parent
FAMILY_NAME = CODE_ROOT.name
DEFAULT_RESULTS_ROOT = REPO_ROOT / "descargas" / FAMILY_NAME / "results"
DEFAULT_LOGS_ROOT = REPO_ROOT / "descargas" / FAMILY_NAME / "logs"

for import_root in (CODE_ROOT, SHARED_CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from oui.core import compute_new_oui_from_activations
from oui.metrics import has_nonfinite, timing_summary_aliases


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_jsonable(row), sort_keys=False) + "\n")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_torch() -> torch.device:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def bf16_available(device: torch.device) -> bool:
    return bool(device.type == "cuda" and torch.cuda.is_bf16_supported())


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "bf16" and bf16_available(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "bf16" if bf16_available(device) else ("fp16" if device.type == "cuda" else "fp32")
    return requested


def build_run_dir(
    cfg: dict[str, Any],
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    output_dir: str | None,
    smoke: bool,
) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    results_root = Path(cfg.get("output", {}).get("results_root") or DEFAULT_RESULTS_ROOT)
    if not results_root.is_absolute():
        results_root = (CODE_ROOT / results_root).resolve()
    experiment_id = str(cfg["experiment_id"])
    root = results_root / ("smoke" if smoke else "") / experiment_id
    return root / f"lr_{lr:.0e}__wd_{weight_decay:.0e}__seed_{seed}"


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets

        versions["datasets"] = datasets.__version__
    except Exception:
        pass
    return versions


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        is_embedding = any(token in lname for token in ("embed", "embedding", "wte", "wpe", "shared"))
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in lname or "ln_" in lname or "layernorm" in lname or is_embedding:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).item())


def global_weight_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [param.detach().float().norm(2) for param in parameters if param.requires_grad]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).item())


def extract_layer_idx(name: str) -> int:
    matches = re.findall(r"(?:layers|layer|h|block|encoder\.layers|decoder\.layers)\.(\d+)", name)
    if matches:
        return int(matches[-1])
    digits = re.findall(r"\.(\d+)\.", name)
    return int(digits[-1]) if digits else -1


class FFNOUIProbe:
    def __init__(
        self,
        model: torch.nn.Module,
        patterns: list[str],
        eps: float = 1e-8,
        saturation: float = 6.0,
        sequence_mode: str | None = "last_token",
        max_token_samples: int | None = None,
    ) -> None:
        self.patterns = tuple(patterns)
        self.eps = float(eps)
        self.saturation = float(saturation)
        self.sequence_mode = sequence_mode
        self.max_token_samples = None if max_token_samples is None or int(max_token_samples) <= 0 else int(max_token_samples)
        self.enabled = False
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.previous_signs: dict[str, torch.Tensor] = {}
        self.block_meta: dict[str, dict[str, Any]] = {}
        self.reset_capture_stats()
        self._register(model)

    def _matches(self, module_name: str) -> bool:
        return any(module_name.endswith(pattern) or pattern in module_name for pattern in self.patterns)

    def _register(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if not name or not self._matches(name):
                continue
            self.block_meta[name] = {
                "block_name": name,
                "layer_idx": extract_layer_idx(name),
                "module_type": module.__class__.__name__,
            }

            def hook(_module, _inputs, output, *, block_name=name):
                if not self.enabled:
                    return
                start = time.perf_counter()
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(tensor):
                    self.activations[block_name] = tensor.detach()
                    self.capture_count += 1
                    self.capture_elements += int(tensor.numel())
                    self.capture_bytes += int(tensor.numel() * tensor.element_size())
                self.capture_seconds += time.perf_counter() - start

            self.handles.append(module.register_forward_hook(hook))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def reset_capture_stats(self) -> None:
        self.capture_seconds = 0.0
        self.capture_count = 0
        self.capture_elements = 0
        self.capture_bytes = 0

    def capture_stats(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_hook_capture_seconds": self.capture_seconds,
            f"{prefix}_hook_capture_count": self.capture_count,
            f"{prefix}_activation_elements": self.capture_elements,
            f"{prefix}_activation_bytes": self.capture_bytes,
        }

    def clear(self) -> None:
        self.activations = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def rows(self, *, step: int, split: str, attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None) -> list[dict[str, Any]]:
        rows = []
        for name, activation in sorted(self.activations.items(), key=lambda item: (extract_layer_idx(item[0]), item[0])):
            rows.append(self._row(name, activation, step=step, split=split, attention_mask=self._mask_for_block(name, attention_mask)))
        return rows

    def _mask_for_block(self, name: str, attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None) -> torch.Tensor | None:
        if not isinstance(attention_mask, dict):
            return attention_mask
        lname = name.lower()
        if "decoder" in lname:
            keys = ("decoder_attention_mask", "decoder_mask", "labels_mask")
        elif "encoder" in lname:
            keys = ("attention_mask", "encoder_attention_mask", "encoder_mask")
        else:
            keys = ("attention_mask", "decoder_attention_mask")
        for key in keys:
            mask = attention_mask.get(key)
            if mask is not None:
                return mask
        return None

    def _row(
        self,
        name: str,
        activation: torch.Tensor,
        *,
        step: int,
        split: str,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        x = activation.detach().float().cpu()
        token_matrix = self._token_matrix(x, attention_mask)
        sequence_matrix = self._sequence_matrix(x, attention_mask)
        stats_matrix = token_matrix if token_matrix.numel() else x.reshape(max(1, x.shape[0]), -1)
        positives = stats_matrix > 0
        positive_fraction = float(positives.float().mean().item()) if stats_matrix.numel() else 0.0
        sparsity = float((stats_matrix.abs() <= self.eps).float().mean().item()) if stats_matrix.numel() else 0.0
        entropy = self._binary_entropy(positive_fraction)
        unit_positive_rates = positives.float().mean(dim=0) if positives.ndim == 2 and positives.numel() else torch.zeros(1)
        minority = torch.minimum(unit_positive_rates, 1.0 - unit_positive_rates)
        majority_sign = unit_positive_rates >= 0.5
        previous = self.previous_signs.get(name)
        flip_rate = None
        if previous is not None and previous.shape == majority_sign.shape:
            flip_rate = float((previous != majority_sign).float().mean().item())
        self.previous_signs[name] = majority_sign

        meta = self.block_meta.get(name, {"block_name": name, "layer_idx": extract_layer_idx(name), "module_type": "unknown"})
        sequence_oui = None
        if sequence_matrix is not None and sequence_matrix.shape[0] >= 2:
            sequence_oui = compute_new_oui_from_activations(sequence_matrix)

        return {
            "step": int(step),
            "split": split,
            **meta,
            "oui_token": compute_new_oui_from_activations(token_matrix) if token_matrix.shape[0] >= 2 else 0.0,
            "oui_sequence": sequence_oui,
            "activation_sparsity": sparsity,
            "positive_fraction": positive_fraction,
            "activation_entropy": entropy,
            "dead_unit_fraction": float((unit_positive_rates <= 0.0).float().mean().item()),
            "saturated_unit_fraction": float((stats_matrix.abs() >= self.saturation).float().mean().item()) if stats_matrix.numel() else 0.0,
            "skewed_fraction_alpha_0_1": float((minority < 0.1).float().mean().item()),
            "skewed_fraction_alpha_0_3": float((minority < 0.3).float().mean().item()),
            "flip_rate_vs_previous_measurement": flip_rate,
            "activation_shape": list(x.shape),
            "samples_token": int(token_matrix.shape[0]) if token_matrix.ndim == 2 else 0,
            "features": int(token_matrix.shape[1]) if token_matrix.ndim == 2 else 0,
        }

    def _token_matrix(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if x.ndim == 3:
            if attention_mask is not None and tuple(attention_mask.shape[:2]) == tuple(x.shape[:2]):
                mask = attention_mask.detach().bool().cpu()
                selected = x[mask]
                if selected.ndim == 2 and selected.shape[0] >= 1:
                    return self._sample_rows(selected)
            return self._sample_rows(x.reshape(-1, x.shape[-1]))
        return x.reshape(x.shape[0], -1)

    def _sample_rows(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.max_token_samples is None or matrix.ndim != 2 or matrix.shape[0] <= self.max_token_samples:
            return matrix
        idx = torch.linspace(0, matrix.shape[0] - 1, steps=self.max_token_samples).round().long()
        return matrix.index_select(0, idx)

    def _sequence_matrix(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if self.sequence_mode in (None, "none"):
            return None
        if x.ndim == 3:
            if self.sequence_mode == "cls":
                return x[:, 0, :]
            if attention_mask is not None and tuple(attention_mask.shape[:2]) == tuple(x.shape[:2]):
                mask = attention_mask.detach().long().cpu()
                idx = torch.clamp(mask.sum(dim=1) - 1, min=0)
                return x[torch.arange(x.shape[0]), idx]
            return x[:, -1, :]
        if x.ndim >= 2:
            return x.reshape(x.shape[0], -1)
        return None

    @staticmethod
    def _binary_entropy(p: float) -> float:
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def aggregate_oui(rows: list[dict[str, Any]], *, step: int, split: str) -> dict[str, Any]:
    values = np.array([row["oui_token"] for row in rows if row.get("oui_token") is not None], dtype=np.float64)
    if values.size == 0:
        return {
            "step": int(step),
            "split": split,
            "oui_mean": None,
            "oui_min": None,
            "oui_max": None,
            "oui_std": None,
            "oui_first_third_mean": None,
            "oui_middle_third_mean": None,
            "oui_last_third_mean": None,
        }
    thirds = np.array_split(values, 3)
    return {
        "step": int(step),
        "split": split,
        "oui_mean": float(values.mean()),
        "oui_min": float(values.min()),
        "oui_max": float(values.max()),
        "oui_std": float(values.std()),
        "oui_first_third_mean": float(thirds[0].mean()) if len(thirds[0]) else None,
        "oui_middle_third_mean": float(thirds[1].mean()) if len(thirds[1]) else None,
        "oui_last_third_mean": float(thirds[2].mean()) if len(thirds[2]) else None,
    }


def oui_partition(block_name: str) -> str | None:
    lname = block_name.lower()
    if "decoder" in lname:
        return "decoder"
    if "encoder" in lname:
        return "encoder"
    return None


def aggregate_oui_partitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in ("encoder", "decoder"):
        part_rows = [row for row in rows if oui_partition(str(row.get("block_name", ""))) == part]
        values = [row.get("oui_token") for row in part_rows if row.get("oui_token") is not None]
        out[f"oui_{part}_mean"] = float(np.mean(values)) if values else None
        out[f"oui_{part}_layers"] = [
            {
                "block_name": row.get("block_name"),
                "layer_idx": row.get("layer_idx"),
                "oui_token": row.get("oui_token"),
                "samples_token": row.get("samples_token"),
            }
            for row in part_rows
        ] if part_rows else None
    return out


def aggregate_block_rows(rows: list[dict[str, Any]], *, step: int, split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        block_name = row.get("block_name")
        if block_name:
            grouped.setdefault(str(block_name), []).append(row)

    numeric_keys = [
        "oui_token",
        "oui_sequence",
        "activation_sparsity",
        "positive_fraction",
        "activation_entropy",
        "dead_unit_fraction",
        "saturated_unit_fraction",
        "skewed_fraction_alpha_0_1",
        "skewed_fraction_alpha_0_3",
        "flip_rate_vs_previous_measurement",
        "samples_token",
        "features",
    ]
    out = []
    for block_name, block_rows in sorted(grouped.items(), key=lambda item: (item[1][0].get("layer_idx", -1), item[0])):
        first = block_rows[0]
        row = {
            "record_type": "block_aggregate",
            "step": int(step),
            "split": split,
            "block_name": block_name,
            "layer_idx": first.get("layer_idx"),
            "module_type": first.get("module_type"),
            "probe_batches": len(block_rows),
        }
        for key in numeric_keys:
            vals = [r.get(key) for r in block_rows if r.get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


@torch.no_grad()
def snapshot_trainable_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


@torch.no_grad()
def update_norms(model: torch.nn.Module, before: list[torch.Tensor]) -> tuple[float, float]:
    delta_sq = 0.0
    weight_sq = 0.0
    idx = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        current = param.detach().float()
        previous = before[idx].float().to(device=current.device)
        delta_sq += float((current - previous).pow(2).sum().item())
        weight_sq += float(current.pow(2).sum().item())
        idx += 1
    update_norm = math.sqrt(delta_sq)
    weight_norm = math.sqrt(weight_sq)
    return update_norm, update_norm / max(weight_norm, 1e-12)
