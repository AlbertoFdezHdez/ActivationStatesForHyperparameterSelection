from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

CODE_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from oui.core import compute_new_oui_with_stats
from oui.metrics import PenultimateActivationTracker, has_nonfinite, timing_summary_aliases


RESULTS_ROOT = REPO_ROOT / "descargas" / "vision_transformers_supervised" / "results"
LOGS_ROOT = REPO_ROOT / "descargas" / "vision_transformers_supervised" / "logs"
MILESTONE_PCTS = (1, 2, 5, 10, 15, 20, 30, 50, 75, 100)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RUN_STATE: dict[str, Any] = {"step": None, "last_train_loss": None, "last_val_loss": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised vision-transformer OUI early-triage training.")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(payload), handle, indent=2, sort_keys=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_jsonable(payload), sort_keys=False) + "\n")


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_dir_for(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    exp_id = cfg["experiment_id"]
    if args.smoke:
        return RESULTS_ROOT / "smoke" / exp_id / timestamp()
    return RESULTS_ROOT / "full" / exp_id / f"lr_{args.lr:.0e}__wd_{args.weight_decay:.0e}__seed_{args.seed}"


def build_transforms(cfg: dict[str, Any]):
    from torchvision import transforms

    image_size = int(cfg["data"]["image_size"])
    mean = tuple(cfg["data"].get("mean", IMAGENET_MEAN))
    std = tuple(cfg["data"].get("std", IMAGENET_STD))
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if image_size == 64:
        val_tf = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        val_tf = transforms.Compose(
            [
                transforms.Resize(int(round(image_size * 256 / 224))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return train_tf, val_tf


def resolve_data_dir(cfg: dict[str, Any]) -> Path | None:
    data = cfg["data"]
    if data.get("data_dir"):
        return Path(data["data_dir"]).expanduser()
    env_var = data.get("env_var")
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var]).expanduser()
    if env_var:
        raise FileNotFoundError(f"{env_var} is not set and no data.data_dir was configured.")
    return None


def dataset_root_for_metadata(cfg: dict[str, Any]) -> Path:
    data = cfg["data"]
    if data.get("data_dir"):
        return Path(data["data_dir"]).expanduser()
    env_var = data.get("env_var")
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var]).expanduser()
    return REPO_ROOT / "data" / data["name"]


def imagefolder_splits(root: Path, train_tf, val_tf):
    from torchvision.datasets import ImageFolder

    candidates = [
        (root / "train", root / "val"),
        (root / "train", root / "validation"),
        (root / "training", root / "validation"),
    ]
    for train_root, val_root in candidates:
        if train_root.exists() and val_root.exists():
            return ImageFolder(train_root, transform=train_tf), ImageFolder(val_root, transform=val_tf)
    raise FileNotFoundError(
        f"Expected ImageFolder split directories under {root}: train/val or train/validation."
    )


def build_datasets(cfg: dict[str, Any]):
    from torchvision import datasets

    train_tf, val_tf = build_transforms(cfg)
    data = cfg["data"]
    name = data["name"]
    root = resolve_data_dir(cfg) or (REPO_ROOT / "data" / name)
    download = bool(data.get("download", False))

    if name == "cifar100":
        train_ds = datasets.CIFAR100(root=str(root), train=True, transform=train_tf, download=download)
        val_ds = datasets.CIFAR100(root=str(root), train=False, transform=val_tf, download=download)
    elif name == "flowers102":
        train_ds = datasets.Flowers102(root=str(root), split="train", transform=train_tf, download=download)
        val_ds = datasets.Flowers102(root=str(root), split="val", transform=val_tf, download=download)
    elif name == "oxford_iiit_pet":
        train_ds = datasets.OxfordIIITPet(root=str(root), split="trainval", target_types="category", transform=train_tf, download=download)
        val_ds = datasets.OxfordIIITPet(root=str(root), split="test", target_types="category", transform=val_tf, download=download)
    elif data.get("loader") == "imagefolder":
        train_ds, val_ds = imagefolder_splits(root, train_tf, val_tf)
    else:
        raise ValueError(f"Unsupported ViT dataset loader/name: {data.get('loader')} / {name}")

    if "smoke_train_examples" in data:
        train_ds = Subset(train_ds, range(min(int(data["smoke_train_examples"]), len(train_ds))))
    if "smoke_val_examples" in data:
        val_ds = Subset(val_ds, range(min(int(data["smoke_val_examples"]), len(val_ds))))
    return train_ds, val_ds


def class_mapping(dataset) -> dict[str, int]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, "class_to_idx"):
        return dict(base.class_to_idx)
    if hasattr(base, "classes"):
        return {str(name): idx for idx, name in enumerate(base.classes)}
    return {str(idx): idx for idx in range(int(getattr(base, "num_classes", 0)))}


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    import timm

    model_name = cfg["model"]["name"]
    pretrained = bool(cfg["model"].get("pretrained", True))
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


class HookedOUI:
    def __init__(self, model: nn.Module, hook_specs: list[tuple[str, nn.Module, str]]) -> None:
        self.records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.enabled = False
        self.handles = []
        for depth, (name, module, module_type) in enumerate(hook_specs):
            self.records[name] = {"name": name, "depth": depth, "module_type": module_type}
            self.handles.append(module.register_forward_hook(self._hook(name, depth, module_type)))

    def _hook(self, name: str, depth: int, module_type: str):
        def fn(_module, _inputs, output):
            if not self.enabled:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                return
            start = time.perf_counter()
            oui, stats = compute_new_oui_with_stats(tensor.detach())
            elapsed = time.perf_counter() - start
            self.records[name] = {
                "name": name,
                "depth": depth,
                "module_type": module_type,
                "oui": oui,
                "activation_shape": tuple(tensor.shape),
                "feature_dim": stats.feature_dim,
                "batch_size": stats.batch_size,
                "working_set_bytes": stats.estimated_working_set_bytes,
                "compute_seconds": elapsed,
            }
        return fn

    def capture(self):
        class CaptureContext:
            def __init__(self, owner: HookedOUI) -> None:
                self.owner = owner

            def __enter__(self):
                self.owner.enabled = True
                for record in self.owner.records.values():
                    for key in ("oui", "activation_shape", "feature_dim", "batch_size", "working_set_bytes", "compute_seconds"):
                        record.pop(key, None)
                return self.owner

            def __exit__(self, exc_type, exc, tb):
                self.owner.enabled = False
                return False

        return CaptureContext(self)

    def rows_and_aggregate(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        rows = [dict(row) for row in self.records.values() if "oui" in row]
        values = [float(row["oui"]) for row in rows]
        aggregate = aggregate_oui(values)
        aggregate["oui_compute_seconds"] = sum(float(row.get("compute_seconds", 0.0)) for row in rows)
        return rows, aggregate

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def get_module(model: nn.Module, dotted: str) -> nn.Module:
    module: nn.Module = model
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def vit_hook_specs(model: nn.Module, model_name: str) -> list[tuple[str, nn.Module, str]]:
    specs: list[tuple[str, nn.Module, str]] = []
    raw_names: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        is_mlp_fc1 = lname.endswith("mlp.fc1") or lname.endswith("mlp.fc1.linear")
        is_ffn_first = ("ffn" in lname or "mlp" in lname) and (lname.endswith(".0") or lname.endswith(".fc1"))
        if is_mlp_fc1 or is_ffn_first:
            raw_names.append((name, module))
    seen = set()
    for name, module in raw_names:
        if id(module) in seen:
            continue
        seen.add(id(module))
        specs.append((f"block_{len(specs)}", module, f"vit_mlp_preactivation:{name}"))
    if not specs:
        raise ValueError(
            f"No ViT MLP/FFN hook points found for {model_name}. "
            "Check timm model internals before changing the model."
        )
    return specs


def aggregate_oui(values: list[float]) -> dict[str, float]:
    if not values:
        return {"oui_mean": math.nan, "oui_std": math.nan, "oui_min": math.nan, "oui_max": math.nan}
    arr = np.asarray(values, dtype=np.float64)
    thirds = np.array_split(arr, 3)
    return {
        "oui_mean": float(arr.mean()),
        "oui_std": float(arr.std(ddof=0)),
        "oui_min": float(arr.min()),
        "oui_max": float(arr.max()),
        "oui_first_third": float(thirds[0].mean()) if len(thirds[0]) else math.nan,
        "oui_middle_third": float(thirds[1].mean()) if len(thirds[1]) else math.nan,
        "oui_last_third": float(thirds[2].mean()) if len(thirds[2]) else math.nan,
    }


def dataloaders(cfg: dict[str, Any], train_ds, val_ds):
    train_cfg = cfg["training"]
    per_device = int(train_cfg["per_device_batch_size"])
    workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=per_device, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=per_device, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def split_batch(batch, device: torch.device):
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def build_optimizer(model: nn.Module, wd: float, lr: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LocalResponseNorm,
    )
    module_by_param = {id(param): module for module in model.modules() for param in module.parameters(recurse=False)}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module = module_by_param.get(id(param))
        lname = name.lower()
        if param.ndim < 2 or name.endswith(".bias") or isinstance(module, norm_types) or "norm" in lname or "bn" in lname or "embed" in lname:
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
    top5 = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = split_batch(batch, device)
        with amp_ctx_factory():
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = labels.numel()
        loss_sum += float(loss.item()) * batch_size
        total += batch_size
        pred1 = logits.argmax(dim=1)
        top1 += int((pred1 == labels).sum().item())
        if logits.shape[1] >= 5:
            top5_idx = logits.topk(5, dim=1).indices
            top5 += int((top5_idx == labels.view(-1, 1)).any(dim=1).sum().item())
    return {
        "val_loss": loss_sum / max(1, total),
        "val_accuracy_top1": top1 / max(1, total),
        "val_accuracy_top5": top5 / max(1, total) if total > 0 else math.nan,
        "val_examples": total,
    }


def amp_factory(precision: str):
    if precision == "fp32":
        return nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def metric_steps(total_steps: int, interval: int) -> set[int]:
    steps = {1, total_steps}
    steps.update(range(interval, total_steps + 1, interval))
    for pct in MILESTONE_PCTS:
        steps.add(max(1, round(total_steps * pct / 100)))
    return steps


def write_oui(path: Path, step: int, phase: str, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    for row in rows:
        append_jsonl(path, {"record_type": "oui_block", "step": step, "phase": phase, **row})
    append_jsonl(path, {"record_type": "aggregate", "step": step, "phase": phase, **aggregate})


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = build_datasets(cfg)
    dataset_root = dataset_root_for_metadata(cfg)
    dataset_report = dataset_root / "dataset_report.json"
    mapping = class_mapping(train_ds)
    num_classes = int(cfg["data"].get("num_classes") or max(mapping.values()) + 1)
    train_loader, val_loader = dataloaders(cfg, train_ds, val_ds)
    effective_bs = int(cfg["training"]["effective_batch_size"])
    per_device_bs = int(cfg["training"]["per_device_batch_size"])
    grad_accum = max(1, math.ceil(effective_bs / per_device_bs))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = int(cfg["training"]["epochs"]) * steps_per_epoch
    if cfg["training"].get("max_steps"):
        max_steps = int(cfg["training"]["max_steps"])
        total_steps = max_steps if args.smoke else min(total_steps, max_steps)
    metric_interval = max(20, round(total_steps / 1000))
    val_interval = max(1, round(total_steps / 20))
    metric_step_set = metric_steps(total_steps, metric_interval)
    val_step_set = metric_steps(total_steps, val_interval)

    model = build_model(cfg, num_classes).to(device)
    repr_tracker = PenultimateActivationTracker(model)
    oui = HookedOUI(model, vit_hook_specs(model, cfg["model"]["name"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, args.weight_decay, args.lr)
    warmup_steps = int(cfg["training"].get("warmup_steps") or round(total_steps * float(cfg["training"].get("warmup_fraction", 0.05))))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    precision, scaler = precision_context(device)
    amp_ctx_factory = amp_factory(precision)
    probe_batch = split_batch(next(iter(val_loader)), device)

    write_json(
        out_dir / "run_metadata.json",
        {
            "experiment_id": cfg["experiment_id"],
            "dataset": cfg["data"]["name"],
            "dataset_root": str(dataset_root),
            "dataset_report_path": str(dataset_report) if dataset_report.exists() else None,
            "dataset_split": cfg["data"].get("split_description", "train/validation"),
            "class_mapping": mapping,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "effective_batch_size": effective_bs,
            "per_device_batch_size": per_device_bs,
            "gradient_accumulation_steps": grad_accum,
            "total_steps": total_steps,
            "metric_interval": metric_interval,
            "val_interval": val_interval,
            "milestone_percents": MILESTONE_PCTS,
            "precision": precision,
            "oui_convention": "ViT/Swin/MobileViT OUI is measured on MLP/FFN first Linear outputs before activation, one row per transformer block.",
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
            images, labels = split_batch(batch, device)
            is_metric_step = will_step and next_optimizer_step in metric_step_set
            phase_times: dict[str, float] = {}
            before = snapshot_params(model) if is_metric_step else []

            model.train()
            train_start = time.perf_counter()
            capture_ctx = oui.capture() if is_metric_step else nullcontext()
            with capture_ctx:
                with amp_ctx_factory():
                    logits = model(images)
                    loss = criterion(logits, labels) / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            phase_times["train_forward_backward_seconds"] = time.perf_counter() - train_start
            timing_totals["train_core_seconds"] += phase_times["train_forward_backward_seconds"]
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
                train_rows, train_agg = oui.rows_and_aggregate()
                write_oui(out_dir / "oui_train.jsonl", global_step, "train", train_rows, train_agg)
                timing_totals["oui_train_seconds"] += float(train_agg.get("oui_compute_seconds", 0.0) or 0.0)
                timing_counts["oui_train"] += 1
                upd = update_norm(model, before) if before else math.nan
                pnorm = param_norm(model)

                probe_every = int(cfg["tracking"].get("oui_probe_every_metric_steps", 1))
                do_probe = ((global_step // max(1, metric_interval)) % probe_every == 0) or global_step in {1, total_steps}
                probe_agg: dict[str, Any] = {}
                if do_probe:
                    model.eval()
                    probe_start = time.perf_counter()
                    with torch.no_grad(), oui.capture():
                        with amp_ctx_factory():
                            _ = model(probe_batch[0])
                    probe_rows, probe_agg = oui.rows_and_aggregate()
                    probe_agg["oui_probe_total_seconds"] = time.perf_counter() - probe_start
                    timing_totals["oui_probe_seconds"] += float(probe_agg.get("oui_probe_total_seconds", 0.0) or 0.0)
                    timing_counts["oui_probe"] += 1
                    write_oui(out_dir / "oui_probe.jsonl", global_step, "probe", probe_rows, probe_agg)

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
                val_max_batches = cfg["tracking"].get("validation_max_batches")
                val = evaluate(model, val_loader, criterion, device, amp_ctx_factory, val_max_batches)
                val_seconds = time.perf_counter() - val_start
                timing_totals["full_validation_seconds"] += val_seconds
                timing_counts["full_validation"] += 1
                last_val_loss = val["val_loss"]
                last_val_acc = val["val_accuracy_top1"]
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
    oui.close()


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
