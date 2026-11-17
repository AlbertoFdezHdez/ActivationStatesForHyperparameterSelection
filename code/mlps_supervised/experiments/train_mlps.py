from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import urllib.request
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from oui.core import compute_new_oui_with_stats
from oui.metrics import PenultimateActivationTracker, has_nonfinite, timing_summary_aliases


RESULTS_ROOT = REPO_ROOT / "descargas" / "mlps_supervised" / "results"
LOGS_ROOT = REPO_ROOT / "descargas" / "mlps_supervised" / "logs"
MILESTONE_PCTS = (1, 2, 5, 10, 15, 20, 30, 50, 75, 100)
RUN_STATE: dict[str, Any] = {"step": None, "last_train_loss": None, "last_val_loss": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised MLP OUI early-triage training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


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
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(payload), handle, indent=2, sort_keys=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_jsonable(payload), sort_keys=False) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def output_dir_for(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    exp_id = cfg["experiment_id"]
    if args.smoke:
        return RESULTS_ROOT / "smoke" / exp_id / timestamp()
    return RESULTS_ROOT / "full" / exp_id / f"lr_{args.lr:.0e}__wd_{args.weight_decay:.0e}__seed_{args.seed}"


def stratified_split_indices(labels: np.ndarray, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return tuple(((arr - mean) / std).astype(np.float32) for arr in (train_x, *others))


class FlattenDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x.reshape(-1), int(y)


def build_image_datasets(cfg: dict[str, Any]):
    from torchvision import datasets, transforms

    name = cfg["data"]["name"]
    root = Path(cfg["data"].get("data_dir") or (REPO_ROOT / "data" / name)).expanduser()
    download = bool(cfg["data"].get("download", False))
    if name == "fashion_mnist":
        mean, std = (0.2860,), (0.3530,)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_base = datasets.FashionMNIST(str(root), train=True, transform=tf, download=download)
        test_ds = FlattenDataset(datasets.FashionMNIST(str(root), train=False, transform=tf, download=download))
    elif name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_base = datasets.CIFAR10(str(root), train=True, transform=train_tf, download=download)
        test_ds = FlattenDataset(datasets.CIFAR10(str(root), train=False, transform=val_tf, download=download))
    else:
        raise ValueError(f"Unsupported image MLP dataset: {name}")

    labels = np.asarray(getattr(train_base, "targets"))
    train_idx, val_idx = stratified_split_indices(labels, float(cfg["data"].get("val_fraction", 0.1)), seed=0)
    train_ds = FlattenDataset(Subset(train_base, train_idx.tolist()))

    if name == "cifar10":
        val_base = datasets.CIFAR10(str(root), train=True, transform=val_tf, download=download)
    else:
        val_base = train_base
    val_ds = FlattenDataset(Subset(val_base, val_idx.tolist()))
    return train_ds, val_ds, test_ds, {str(i): i for i in range(int(cfg["data"]["num_classes"]))}, {
        "validation_indices": val_idx.tolist(),
        "split_source": "stratified validation from official train split, seed 0",
    }


def load_uci_har(cfg: dict[str, Any]):
    root = Path(cfg["data"].get("data_dir") or os.environ.get("UCI_HAR_ROOT", "")).expanduser()
    if not root:
        root = REPO_ROOT / "data" / "uci_har"
    if not root.exists() or not ((root / "train" / "X_train.txt").exists() or (root / "UCI HAR Dataset").exists()):
        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / "UCI_HAR_Dataset.zip"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    if (root / "UCI HAR Dataset").exists():
        root = root / "UCI HAR Dataset"
    train_x = np.loadtxt(root / "train" / "X_train.txt", dtype=np.float32)
    train_y = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    test_x = np.loadtxt(root / "test" / "X_test.txt", dtype=np.float32)
    test_y = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    train_idx, val_idx = stratified_split_indices(train_y, float(cfg["data"].get("val_fraction", 0.15)), seed=0)
    x_train_raw, y_train = train_x[train_idx], train_y[train_idx]
    x_val_raw, y_val = train_x[val_idx], train_y[val_idx]
    x_train, x_val, x_test = standardize(x_train_raw, x_val_raw, test_x)
    return (
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(test_y)),
        {str(i): i for i in range(int(cfg["data"]["num_classes"]))},
        {"validation_indices": val_idx.tolist(), "split_source": "stratified validation from official train split, seed 0"},
    )


def load_covertype(cfg: dict[str, Any]):
    try:
        from sklearn.datasets import fetch_covtype
    except ImportError as exc:
        raise ImportError("Covertype requires scikit-learn on the server.") from exc

    root = Path(cfg["data"].get("data_dir") or (REPO_ROOT / "data" / "covertype")).expanduser()
    bunch = fetch_covtype(data_home=str(root), download_if_missing=bool(cfg["data"].get("download", False)))
    x = bunch.data.astype(np.float32)
    y = bunch.target.astype(np.int64) - 1
    train_idx, tmp_idx = stratified_split_indices(y, float(cfg["data"].get("holdout_fraction", 0.3)), seed=0)
    tmp_y = y[tmp_idx]
    rel_val_idx, rel_test_idx = stratified_split_indices(tmp_y, 0.5, seed=0)
    val_idx = tmp_idx[rel_val_idx]
    test_idx = tmp_idx[rel_test_idx]
    x_train, x_val, x_test = standardize(x[train_idx], x[val_idx], x[test_idx])
    return (
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y[train_idx])),
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y[val_idx])),
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y[test_idx])),
        {str(i): i for i in range(int(cfg["data"]["num_classes"]))},
        {
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "split_source": "stratified train/val/test split from full Covertype, seed 0",
        },
    )


def maybe_subset(ds: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return ds
    return Subset(ds, range(min(int(limit), len(ds))))


def build_datasets(cfg: dict[str, Any]):
    name = cfg["data"]["name"]
    if name in {"fashion_mnist", "cifar10"}:
        train_ds, val_ds, test_ds, mapping, split_info = build_image_datasets(cfg)
    elif name == "uci_har":
        train_ds, val_ds, test_ds, mapping, split_info = load_uci_har(cfg)
    elif name == "covertype":
        train_ds, val_ds, test_ds, mapping, split_info = load_covertype(cfg)
    else:
        raise ValueError(f"Unsupported MLP dataset: {name}")
    train_ds = maybe_subset(train_ds, cfg["data"].get("smoke_train_examples"))
    val_ds = maybe_subset(val_ds, cfg["data"].get("smoke_val_examples"))
    test_ds = maybe_subset(test_ds, cfg["data"].get("smoke_test_examples"))
    return train_ds, val_ds, test_ds, mapping, split_info


class InstrumentedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropouts: list[float]) -> None:
        super().__init__()
        self.hidden = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            self.hidden.append(nn.Linear(prev, hidden_dim))
            self.dropouts.append(nn.Dropout(dropouts[idx] if idx < len(dropouts) else 0.0))
            prev = hidden_dim
        self.output = nn.Linear(prev, num_classes)
        self.capture_activations = False
        self.last_preactivations: list[torch.Tensor] = []
        self.last_postactivations: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_preactivations = []
        self.last_postactivations = []
        x = x.reshape(x.shape[0], -1)
        for layer, dropout in zip(self.hidden, self.dropouts):
            pre = layer(x)
            post = torch.relu(pre)
            if self.capture_activations:
                self.last_preactivations.append(pre.detach())
                self.last_postactivations.append(post.detach())
            x = dropout(post)
        return self.output(x)


def build_model(cfg: dict[str, Any]) -> InstrumentedMLP:
    return InstrumentedMLP(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dims=[int(v) for v in cfg["model"]["hidden_dims"]],
        num_classes=int(cfg["data"]["num_classes"]),
        dropouts=[float(v) for v in cfg["model"].get("dropouts", [])],
    )


def aggregate_oui(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "oui_mean": math.nan,
            "oui_std": math.nan,
            "oui_min": math.nan,
            "oui_max": math.nan,
            "oui_first": math.nan,
            "oui_last": math.nan,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "oui_mean": float(arr.mean()),
        "oui_std": float(arr.std(ddof=0)),
        "oui_min": float(arr.min()),
        "oui_max": float(arr.max()),
        "oui_first": float(arr[0]),
        "oui_last": float(arr[-1]),
    }


def activation_stats(post: torch.Tensor) -> dict[str, float]:
    flat = post.reshape(post.shape[0], -1)
    positive = flat > 0
    p = positive.float().mean(dim=0)
    entropy = -(p * torch.log2(p.clamp_min(1e-8)) + (1 - p) * torch.log2((1 - p).clamp_min(1e-8)))
    return {
        "activation_sparsity": float((flat == 0).float().mean().item()),
        "activation_entropy": float(entropy.mean().item()),
        "dead_unit_fraction": float((p == 0).float().mean().item()),
        "saturated_unit_fraction": float((p == 1).float().mean().item()),
    }


@torch.no_grad()
def compute_oui_rows(model: InstrumentedMLP, step: int, phase: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    values = []
    compute_seconds = 0.0
    for idx, (pre, post) in enumerate(zip(model.last_preactivations, model.last_postactivations), start=1):
        start = time.perf_counter()
        oui, oui_stats = compute_new_oui_with_stats(pre)
        elapsed = time.perf_counter() - start
        compute_seconds += elapsed
        values.append(oui)
        rows.append(
            {
                "record_type": "oui_layer",
                "step": step,
                "phase": phase,
                "name": f"hidden_{idx}",
                "depth": idx - 1,
                "module_type": "mlp_hidden_linear_preactivation",
                "oui": oui,
                "activation_shape": tuple(pre.shape),
                "feature_dim": oui_stats.feature_dim,
                "batch_size": oui_stats.batch_size,
                "working_set_bytes": oui_stats.estimated_working_set_bytes,
                "compute_seconds": elapsed,
                **activation_stats(post),
            }
        )
    agg = aggregate_oui(values)
    agg["oui_compute_seconds"] = compute_seconds
    return rows, agg


def write_oui(path: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], step: int, phase: str) -> None:
    for row in rows:
        append_jsonl(path, row)
    append_jsonl(path, {"record_type": "aggregate", "step": step, "phase": phase, **aggregate})


def build_optimizer(model: nn.Module, wd: float, lr: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def precision_context(device: torch.device):
    if device.type != "cuda":
        return "fp32", None
    if torch.cuda.is_bf16_supported():
        return "bf16", None
    scaler = torch.cuda.amp.GradScaler()
    return "fp16", scaler


def amp_factory(precision: str):
    if precision == "fp32":
        return nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def split_batch(batch, device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True).float(), y.to(device, non_blocking=True).long()


def param_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            total += param.detach().float().pow(2).sum()
    return float(total.sqrt().item())


def snapshot_params(model: nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def update_norm(model: nn.Module, before: list[torch.Tensor]) -> float:
    total = torch.zeros((), device=before[0].device if before else next(model.parameters()).device)
    idx = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        total += (param.detach() - before[idx]).float().pow(2).sum()
        idx += 1
    return float(total.sqrt().item())


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum()
    return float(total.sqrt().item())


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_ctx_factory, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = split_batch(batch, device)
        with amp_ctx_factory():
            logits = model(x)
            loss = criterion(logits, y)
        batch_size = y.numel()
        loss_sum += float(loss.item()) * batch_size
        total += batch_size
        top1 += int((logits.argmax(dim=1) == y).sum().item())
    return {
        "val_loss": loss_sum / max(1, total),
        "val_accuracy": top1 / max(1, total),
        "val_examples": total,
    }


def metric_steps(total_steps: int, interval: int) -> set[int]:
    steps = {1, total_steps}
    steps.update(range(interval, total_steps + 1, interval))
    for pct in MILESTONE_PCTS:
        steps.add(max(1, round(total_steps * pct / 100)))
    return steps


def main_impl() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.smoke:
        cfg = deep_update(cfg, cfg.get("smoke", {}))
    set_seed(args.seed)
    out_dir = output_dir_for(cfg, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", {"config": cfg, "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed, "smoke": args.smoke})

    train_ds, val_ds, test_ds, mapping, split_info = build_datasets(cfg)
    train_cfg = cfg["training"]
    per_device = int(train_cfg["per_device_batch_size"])
    workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=per_device, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=per_device, shuffle=False, num_workers=workers, pin_memory=True)
    effective_bs = int(train_cfg["effective_batch_size"])
    grad_accum = max(1, math.ceil(effective_bs / per_device))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = int(train_cfg["epochs"]) * steps_per_epoch
    if train_cfg.get("max_steps"):
        max_steps = int(train_cfg["max_steps"])
        total_steps = max_steps if args.smoke else min(total_steps, max_steps)
    metric_interval = max(20, round(total_steps / 1000))
    val_interval = max(1, round(total_steps / 20))
    metric_step_set = metric_steps(total_steps, metric_interval)
    val_step_set = metric_steps(total_steps, val_interval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    repr_tracker = PenultimateActivationTracker(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.weight_decay, args.lr)
    warmup_steps = int(round(total_steps * 0.05))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    precision, scaler = precision_context(device)
    amp_ctx_factory = amp_factory(precision)
    probe_batch = split_batch(next(iter(val_loader)), device)

    write_json(
        out_dir / "run_metadata.json",
        {
            "experiment_id": cfg["experiment_id"],
            "dataset": cfg["data"]["name"],
            "dataset_split": split_info,
            "class_mapping": mapping,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "test_examples": len(test_ds),
            "effective_batch_size": effective_bs,
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "total_steps": total_steps,
            "metric_interval": metric_interval,
            "val_interval": val_interval,
            "milestone_percents": MILESTONE_PCTS,
            "precision": precision,
            "oui_convention": "OUI is measured on hidden Linear preactivations before ReLU; post-ReLU sparsity/dead/saturated stats are also logged.",
        },
    )

    best_val_loss = math.inf
    best_val_acc = -math.inf
    best_val_step = -1
    best_val_acc_step = -1
    last_val_loss = None
    last_val_acc = None
    timing_totals = {
        "train_core_seconds": 0.0,
        "oui_train_seconds": 0.0,
        "oui_probe_seconds": 0.0,
        "full_validation_seconds": 0.0,
        "other_overhead_seconds": 0.0,
    }
    timing_counts = {"steps": 0, "oui_train": 0, "oui_probe": 0, "full_validation": 0}
    last_train_loss = None
    global_step = 0
    micro_step = 0
    start_wall = time.time()
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
            micro_step += 1
            will_step = micro_step % grad_accum == 0
            next_optimizer_step = global_step + 1
            is_metric_step = will_step and next_optimizer_step in metric_step_set
            before = snapshot_params(model) if is_metric_step else []
            x, y = split_batch(batch, device)
            phase_times: dict[str, float] = {}

            model.train()
            model.capture_activations = is_metric_step
            train_start = time.perf_counter()
            with amp_ctx_factory():
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            phase_times["train_forward_backward_seconds"] = time.perf_counter() - train_start
            timing_totals["train_core_seconds"] += phase_times["train_forward_backward_seconds"]
            model.capture_activations = False
            accum_loss += float(loss.item()) * grad_accum

            if not will_step:
                continue

            opt_start = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
            pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            post_clip = grad_norm(model)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            phase_times["optimizer_seconds"] = time.perf_counter() - opt_start
            timing_totals["train_core_seconds"] += phase_times["optimizer_seconds"]
            timing_counts["steps"] += 1
            global_step = next_optimizer_step

            last_train_loss = accum_loss / grad_accum
            RUN_STATE["step"] = global_step
            RUN_STATE["last_train_loss"] = last_train_loss
            accum_loss = 0.0

            if is_metric_step:
                train_rows, train_agg = compute_oui_rows(model, global_step, "train")
                write_oui(out_dir / "oui_train.jsonl", train_rows, train_agg, global_step, "train")
                timing_totals["oui_train_seconds"] += float(train_agg.get("oui_compute_seconds", 0.0) or 0.0)
                timing_counts["oui_train"] += 1
                upd = update_norm(model, before) if before else math.nan
                pnorm = param_norm(model)

                model.eval()
                probe_start = time.perf_counter()
                with torch.no_grad():
                    model.capture_activations = True
                    with amp_ctx_factory():
                        _ = model(probe_batch[0])
                    model.capture_activations = False
                probe_rows, probe_agg = compute_oui_rows(model, global_step, "probe")
                probe_agg["oui_probe_total_seconds"] = time.perf_counter() - probe_start
                timing_totals["oui_probe_seconds"] += float(probe_agg.get("oui_probe_total_seconds", 0.0) or 0.0)
                timing_counts["oui_probe"] += 1
                write_oui(out_dir / "oui_probe.jsonl", probe_rows, probe_agg, global_step, "probe")

                append_jsonl(
                    out_dir / "metrics.jsonl",
                    {
                        "record_type": "train_metrics",
                        "step": global_step,
                        "train_loss": last_train_loss,
                        "current_lr": optimizer.param_groups[0]["lr"],
                        "configured_lr": args.lr,
                        "configured_wd": args.weight_decay,
                        "grad_norm_pre_clip": float(pre_clip.item() if torch.is_tensor(pre_clip) else pre_clip),
                        "grad_norm_post_clip": post_clip,
                        "clipping_applied": float(pre_clip) > 1.0,
                        "param_norm": pnorm,
                        "update_norm": upd,
                        "update_to_weight_ratio": upd / max(pnorm, 1e-12),
                        "wall_time": time.time() - start_wall,
                        **phase_times,
                        **repr_tracker.stats(),
                        **train_agg,
                        **{f"probe_{k}": v for k, v in probe_agg.items()},
                    },
                )

            if global_step in val_step_set:
                val_start = time.perf_counter()
                val_max_batches = cfg.get("tracking", {}).get("validation_max_batches")
                val = evaluate(model, val_loader, criterion, device, amp_ctx_factory, val_max_batches)
                val_seconds = time.perf_counter() - val_start
                timing_totals["full_validation_seconds"] += val_seconds
                timing_counts["full_validation"] += 1
                last_val_loss = val["val_loss"]
                last_val_acc = val["val_accuracy"]
                RUN_STATE["last_val_loss"] = last_val_loss
                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    best_val_step = global_step
                if last_val_acc > best_val_acc:
                    best_val_acc = last_val_acc
                    best_val_acc_step = global_step
                append_jsonl(
                    out_dir / "metrics.jsonl",
                    {
                        "record_type": "validation_metrics",
                        "step": global_step,
                        **val,
                        "validation_seconds": val_seconds,
                        "current_lr": optimizer.param_groups[0]["lr"],
                        "configured_lr": args.lr,
                        "configured_wd": args.weight_decay,
                        "wall_time": time.time() - start_wall,
                    },
                )

    total_time = time.time() - start_wall
    measured = sum(v for k, v in timing_totals.items() if k != "other_overhead_seconds")
    timing_totals["other_overhead_seconds"] = max(0.0, total_time - measured)
    timing_percentages = {k.replace("_seconds", "_pct"): (v / total_time * 100.0 if total_time > 0 else 0.0) for k, v in timing_totals.items()}
    timing_averages = {
        "train_core_avg_seconds": timing_totals["train_core_seconds"] / max(1, timing_counts["steps"]),
        "oui_train_avg_seconds": timing_totals["oui_train_seconds"] / max(1, timing_counts["oui_train"]),
        "oui_probe_avg_seconds": timing_totals["oui_probe_seconds"] / max(1, timing_counts["oui_probe"]),
        "full_validation_avg_seconds": timing_totals["full_validation_seconds"] / max(1, timing_counts["full_validation"]),
    }
    summary = {
            "experiment_id": cfg["experiment_id"],
            "failed_run": False,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "last_val_loss": last_val_loss,
            "last_val_acc": last_val_acc,
            "best_val_step": best_val_step,
            "best_val_acc_step": best_val_acc_step,
            "final_step": global_step,
            "total_steps": total_steps,
            "seed": args.seed,
            "lr": args.lr,
            "wd": args.weight_decay,
            "last_train_loss": last_train_loss,
            "precision": precision,
            "wall_time": total_time,
            **timing_summary_aliases(timing_totals, total_time),
            "timing_seconds": timing_totals,
            "timing_percentages": timing_percentages,
            "timing_counts": timing_counts,
            "timing_averages": timing_averages,
            "integrity": {
                "run_completed": global_step >= total_steps,
                "nonfinite_in_summary": False,
                "num_train_checkpoints": timing_counts["steps"],
                "num_metric_checkpoints": timing_counts["oui_train"],
                "num_evaluations": timing_counts["full_validation"],
                "num_oui_train_measurements": timing_counts["oui_train"],
                "num_oui_probe_measurements": timing_counts["oui_probe"],
            },
        }
    summary["integrity"]["nonfinite_in_summary"] = has_nonfinite(summary)
    write_json(out_dir / "summary.json", summary)
    repr_tracker.close()


def main() -> None:
    try:
        main_impl()
    except Exception as exc:
        args = parse_args()
        cfg = read_yaml(args.config)
        if args.smoke:
            cfg = deep_update(cfg, cfg.get("smoke", {}))
        out_dir = output_dir_for(cfg, args)
        write_json(
            out_dir / "failed_run.json",
            {
                "failed_run": True,
                "experiment_id": cfg.get("experiment_id"),
                "lr": args.lr,
                "wd": args.weight_decay,
                "seed": args.seed,
                "step": RUN_STATE.get("step"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "last_train_loss": RUN_STATE.get("last_train_loss"),
                "last_val_loss": RUN_STATE.get("last_val_loss"),
            },
        )
        raise


if __name__ == "__main__":
    main()
