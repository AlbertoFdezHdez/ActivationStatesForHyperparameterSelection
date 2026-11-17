from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from oui_experiment_utils import (
    FFNOUIProbe,
    aggregate_block_rows,
    aggregate_oui,
    aggregate_oui_partitions,
    append_jsonl,
    autocast_context,
    build_run_dir,
    build_scheduler,
    dependency_versions,
    global_grad_norm,
    global_weight_norm,
    has_nonfinite,
    log,
    parameter_groups,
    read_yaml,
    resolve_precision,
    set_seed,
    setup_torch,
    snapshot_trainable_params,
    timing_summary_aliases,
    update_norms,
    write_json,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OUI-Triage fine-tuning runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", "--wd", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_split(cfg: dict[str, Any], split: str):
    data_cfg = cfg["dataset"]
    local_dir = data_cfg.get("local_dir")
    if local_dir and Path(local_dir).exists():
        ds = datasets.load_from_disk(local_dir)
        return ds[split] if split in ds else ds
    kwargs = {"path": data_cfg["name"], "split": split}
    if data_cfg.get("config"):
        kwargs["name"] = data_cfg["config"]
    if data_cfg.get("cache_dir"):
        kwargs["cache_dir"] = data_cfg["cache_dir"]
    return datasets.load_dataset(**kwargs)


def sample_examples(cfg: dict[str, Any], split: str, count: int) -> list[dict[str, Any]]:
    ds = load_split(cfg, split)
    count = min(count, len(ds))
    rows = []
    for idx in range(count):
        ex = dict(ds[idx])
        if cfg["task_type"] == "sequence_classification":
            joined = " ".join(str(ex.get(col, "")) for col in cfg["dataset"].get("input_columns", []))
        else:
            joined = " ".join(str(ex.get(col, "")) for col in [cfg["dataset"]["source_column"], cfg["dataset"]["target_column"]])
        rows.append({"index": idx, "example": ex, "text_hash": text_hash(joined)})
    if not rows:
        raise RuntimeError(f"No examples collected for split={split}")
    return rows


def iter_example_batches(cfg: dict[str, Any], split: str, batch_size: int):
    ds = load_split(cfg, split)
    batch = []
    for idx in range(len(ds)):
        batch.append(dict(ds[idx]))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def shuffled_batches(cfg: dict[str, Any], split: str, batch_size: int, seed: int):
    ds = load_split(cfg, split)
    while True:
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
        batch = []
        for idx in order:
            batch.append(dict(ds[idx]))
            if len(batch) == batch_size:
                yield batch
                batch = []
        seed += 1


def build_tokenizer(cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"].get("cache_dir"),
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def build_model(cfg: dict[str, Any]):
    if cfg["task_type"] == "sequence_classification":
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=int(cfg["model"]["num_labels"]),
            cache_dir=cfg["model"].get("cache_dir"),
        )
    return AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["name"], cache_dir=cfg["model"].get("cache_dir"))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rouge_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_f1(overlap: int, pred_count: int, ref_count: int) -> float:
    if overlap <= 0 or pred_count <= 0 or ref_count <= 0:
        return 0.0
    precision = overlap / pred_count
    recall = overlap / ref_count
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def rouge_lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    rouge1, rouge_l = [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = rouge_tokens(pred)
        ref_tokens = rouge_tokens(ref)
        ref_counts: dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        rouge1.append(rouge_f1(overlap, len(pred_tokens), len(ref_tokens)))
        rouge_l.append(rouge_f1(rouge_lcs_len(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens)))
    denom = max(1, len(rouge1))
    return {"rouge1": sum(rouge1) / denom, "rougeL": sum(rouge_l) / denom}


def examples_from_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row["example"]) for row in rows]


def build_batch(cfg: dict[str, Any], tokenizer, examples: list[dict[str, Any]], *, device: torch.device) -> dict[str, torch.Tensor]:
    if cfg["task_type"] == "sequence_classification":
        columns = cfg["dataset"]["input_columns"]
        first = [str(ex[columns[0]]) for ex in examples]
        second = [str(ex[columns[1]]) for ex in examples] if len(columns) > 1 else None
        encoded = tokenizer(
            first,
            second,
            max_length=int(cfg["training"]["max_sequence_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(ex[cfg["dataset"]["label_column"]]) for ex in examples], dtype=torch.long)
    else:
        encoded = tokenizer(
            [str(ex[cfg["dataset"]["source_column"]]) for ex in examples],
            max_length=int(cfg["training"]["max_source_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tokenizer(
            [str(ex[cfg["dataset"]["target_column"]]) for ex in examples],
            max_length=int(cfg["training"]["max_target_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        encoded["decoder_attention_mask"] = labels.ne(-100).long()
    return {key: value.to(device) for key, value in encoded.items()}


def label_stats(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    labels = batch.get("labels")
    if labels is None or labels.ndim != 2:
        return {}
    valid = labels.ne(-100)
    return {
        "label_ignore_fraction": float((~valid).float().mean().item()),
        "target_nonignored_mean": float(valid.sum(dim=1).float().mean().item()),
    }


def oui_attention_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    if batch.get("attention_mask") is not None:
        masks["attention_mask"] = batch["attention_mask"]
    if batch.get("decoder_attention_mask") is not None:
        masks["decoder_attention_mask"] = batch["decoder_attention_mask"]
    elif batch.get("labels") is not None and batch["labels"].ndim == 2:
        masks["decoder_attention_mask"] = batch["labels"].ne(-100)
    return masks


@torch.no_grad()
def evaluate_full_validation(model, cfg, tokenizer, device, precision) -> dict[str, float]:
    model.eval()
    batch_size = int(cfg["validation"]["batch_size"])
    max_batches = cfg["validation"].get("max_batches")
    max_batches = None if max_batches is None else int(max_batches)
    total_loss = 0.0
    total_items = 0
    correct = 0
    seen_for_acc = 0
    loss_seconds = 0.0
    extra_seconds = 0.0
    batches = 0
    for examples in iter_example_batches(cfg, cfg["dataset"]["validation_split"], batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        batch = build_batch(cfg, tokenizer, examples, device=device)
        sync_if_cuda(device)
        loss_start = time.perf_counter()
        with autocast_context(device, precision):
            outputs = model(**batch)
        sync_if_cuda(device)
        loss_seconds += time.perf_counter() - loss_start
        total_loss += float(outputs.loss.item()) * len(examples)
        total_items += len(examples)
        if cfg["task_type"] == "sequence_classification":
            sync_if_cuda(device)
            extra_start = time.perf_counter()
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((preds == labels).sum().item())
            seen_for_acc += int(labels.numel())
            sync_if_cuda(device)
            extra_seconds += time.perf_counter() - extra_start
        batches += 1
    model.train()
    validation_loss = total_loss / max(1, total_items)
    metrics = {
        "validation_loss": validation_loss,
        "validation_items": float(total_items),
        "validation_loss_seconds": loss_seconds,
        "validation_extra_metrics_seconds": extra_seconds,
    }
    if seen_for_acc:
        metrics["accuracy"] = correct / seen_for_acc
    if cfg["task_type"] == "seq2seq_summarization":
        metrics["validation_perplexity"] = math.exp(min(validation_loss, 50.0))
    return metrics


@torch.no_grad()
def measure_oui(model, probe, cfg, tokenizer, samples, device, precision, *, step: int, split: str, output_path: Path) -> dict[str, float]:
    model.eval()
    batch_size = int(cfg["oui"]["probe_batch_size"])
    all_rows = []
    total_loss = 0.0
    total_items = 0
    correct = 0
    seen_for_acc = 0
    forward_seconds = 0.0
    compute_seconds = 0.0
    probe.reset_capture_stats()
    for batch_idx, start in enumerate(range(0, len(samples), batch_size)):
        examples = examples_from_probe(samples[start : start + batch_size])
        batch = build_batch(cfg, tokenizer, examples, device=device)
        probe.clear()
        probe.set_enabled(True)
        sync_if_cuda(device)
        forward_start = time.perf_counter()
        with autocast_context(device, precision):
            outputs = model(**batch)
        sync_if_cuda(device)
        probe.set_enabled(False)
        forward_seconds += time.perf_counter() - forward_start
        total_loss += float(outputs.loss.item()) * len(examples)
        total_items += len(examples)
        if cfg["task_type"] == "sequence_classification":
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((preds == labels).sum().item())
            seen_for_acc += int(labels.numel())
        compute_start = time.perf_counter()
        rows = probe.rows(step=step, split=split, attention_mask=oui_attention_masks(batch))
        if bool(cfg["oui"].get("save_probe_batches", False)):
            for row in rows:
                row["record_type"] = "block_probe_batch"
                row["probe_batch_idx"] = batch_idx
                append_jsonl(output_path, row)
        all_rows.extend(rows)
        compute_seconds += time.perf_counter() - compute_start
    compute_start = time.perf_counter()
    block_aggregates = aggregate_block_rows(all_rows, step=step, split=split)
    for row in block_aggregates:
        append_jsonl(output_path, row)
    oui_aggregate = aggregate_oui(block_aggregates, step=step, split=split)
    oui_aggregate.update(aggregate_oui_partitions(block_aggregates))
    append_jsonl(output_path, {"record_type": "aggregate", **oui_aggregate})
    compute_seconds += time.perf_counter() - compute_start
    model.train()
    metrics = {"probe_loss": total_loss / max(1, total_items), "probe_items": float(total_items)}
    if seen_for_acc:
        metrics["probe_accuracy"] = correct / seen_for_acc
    if cfg["task_type"] == "seq2seq_summarization":
        metrics["probe_perplexity"] = math.exp(min(metrics["probe_loss"], 50.0))
    metrics.update({"oui_probe_mean": oui_aggregate.get("oui_mean"), "oui_probe_min": oui_aggregate.get("oui_min"), "oui_probe_max": oui_aggregate.get("oui_max"), "oui_probe_std": oui_aggregate.get("oui_std"), "oui_probe_encoder_mean": oui_aggregate.get("oui_encoder_mean"), "oui_probe_decoder_mean": oui_aggregate.get("oui_decoder_mean"), "oui_probe_forward_seconds": forward_seconds, "oui_probe_compute_seconds": compute_seconds, **probe.capture_stats("oui_probe"), **label_stats(batch)})
    return metrics


def write_oui_rows(rows, *, step: int, split: str, output_path: Path) -> dict[str, Any]:
    block_aggregates = aggregate_block_rows(rows, step=step, split=split)
    for row in block_aggregates:
        append_jsonl(output_path, row)
    oui_aggregate = aggregate_oui(block_aggregates, step=step, split=split)
    oui_aggregate.update(aggregate_oui_partitions(block_aggregates))
    append_jsonl(output_path, {"record_type": "aggregate", **oui_aggregate})
    return oui_aggregate


def estimate_full_time(cfg: dict[str, Any], elapsed: float, completed_steps: int) -> dict[str, float]:
    full_steps = int(cfg["training"].get("full_max_steps", cfg["training"].get("max_steps", completed_steps)))
    sec_per_step = elapsed / max(1, completed_steps)
    return {
        "seconds_per_step": sec_per_step,
        "estimated_full_steps": full_steps,
        "estimated_full_time_sec": sec_per_step * full_steps,
        "estimated_full_time_hours": sec_per_step * full_steps / 3600.0,
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.setdefault("training", {})["full_max_steps"] = int(cfg["training"].get("max_steps", 0))
    if args.smoke:
        cfg = {**cfg, "training": {**cfg["training"], **cfg.get("smoke", {}).get("training", {})}}
        cfg = {**cfg, "validation": {**cfg["validation"], **cfg.get("smoke", {}).get("validation", {})}}
        cfg = {**cfg, "oui": {**cfg["oui"], **cfg.get("smoke", {}).get("oui", {})}}
        cfg = {**cfg, "output": {**cfg.get("output", {}), **cfg.get("smoke", {}).get("output", {})}}
    cfg.setdefault("output", {})["save_final_checkpoint"] = False

    lr = float(args.lr if args.lr is not None else cfg["training"]["learning_rates"][0])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg["training"]["weight_decays"][0])
    seed = int(args.seed if args.seed is not None else cfg["training"].get("seeds", [0])[0])
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = int(args.max_steps)
    cfg["training"]["selected_lr"] = lr
    cfg["training"]["selected_weight_decay"] = weight_decay
    cfg["training"]["selected_seed"] = seed

    set_seed(seed)
    device = setup_torch()
    precision = resolve_precision(args.precision or cfg["training"].get("precision", "auto"), device)
    run_dir = build_run_dir(cfg, lr=lr, weight_decay=weight_decay, seed=seed, output_dir=args.output_dir, smoke=args.smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    for stale_name in (
        "metrics.jsonl",
        "oui_train.jsonl",
        "oui_probe.jsonl",
        "oui_validation.jsonl",
        "oui_val.jsonl",
    ):
        stale_path = run_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    write_yaml(run_dir / "config.yaml", cfg)

    summary = {"status": "running", "experiment_id": cfg["experiment_id"], "versions": dependency_versions(), "checkpoint_saved": False}
    write_json(run_dir / "summary.json", summary)
    start_time = time.time()
    try:
        log(f"Run start | experiment={cfg['experiment_id']} smoke={args.smoke} device={device} precision={precision}")
        tokenizer = build_tokenizer(cfg)
        model = build_model(cfg).to(device)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        optimizer = torch.optim.AdamW(parameter_groups(model, weight_decay), lr=lr, betas=tuple(cfg["training"].get("betas", [0.95, 0.95])))
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
        max_steps = int(cfg["training"]["max_steps"])
        warmup_steps = max(1, int(max_steps * float(cfg["training"]["warmup_ratio"])))
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=str(cfg["training"].get("scheduler", "linear")),
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr_ratio=float(cfg["training"].get("min_lr_ratio", 0.0)),
        )
        probe = FFNOUIProbe(model, cfg["oui"]["hook_patterns"], sequence_mode=cfg["oui"].get("sequence_mode", "last_token"), max_token_samples=cfg["oui"].get("max_token_samples"))
        if not probe.block_meta:
            raise RuntimeError(f"No FFN modules matched OUI patterns: {cfg['oui']['hook_patterns']}")

        val_probe = sample_examples(cfg, cfg["dataset"]["validation_split"], int(cfg["oui"]["probe_size"]))
        write_json(run_dir / "probe_indices.json", {"val_probe": val_probe})
        train_batches = shuffled_batches(cfg, cfg["dataset"]["train_split"], int(cfg["training"]["batch_size"]), seed)

        grad_accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
        eval_every = int(cfg["validation"]["eval_every"])
        oui_every = int(cfg["oui"]["measure_every"])
        best_val_loss = float("inf")
        best_val_step = None
        last_val_loss = None
        final_step_val_loss = None
        best_task_metric = None
        last_task_metric = None
        last_train_loss = None
        examples_seen = 0
        tokens_seen = 0
        last_train_oui_masks = None
        last_label_stats: dict[str, float] = {}
        timing_totals = {
            "train_core_seconds": 0.0,
            "scalar_metrics_seconds": 0.0,
            "oui_train_seconds": 0.0,
            "oui_probe_seconds": 0.0,
            "full_validation_seconds": 0.0,
            "validation_loss_seconds": 0.0,
            "validation_extra_metrics_seconds": 0.0,
            "step_wall_seconds": 0.0,
        }
        timing_counts = {
            "steps": 0,
            "scalar_metrics": 0,
            "oui_train": 0,
            "oui_probe": 0,
            "full_validation": 0,
        }
        model.train()

        for step in range(1, max_steps + 1):
            step_wall_start = time.perf_counter()
            is_scalar_metric_step = step == 1 or step % int(cfg["training"].get("log_every", 25)) == 0 or step == max_steps
            is_oui_step = step == 1 or step % oui_every == 0 or step == max_steps
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            sync_if_cuda(device)
            train_core_start = time.perf_counter()
            for accum_idx in range(grad_accum):
                examples = next(train_batches)
                batch = build_batch(cfg, tokenizer, examples, device=device)
                examples_seen += len(examples)
                if "attention_mask" in batch:
                    tokens_seen += int(batch["attention_mask"].sum().item())
                last_train_oui_masks = oui_attention_masks(batch)
                last_label_stats = label_stats(batch)
                capture_train_oui = is_oui_step and accum_idx == grad_accum - 1
                if capture_train_oui:
                    probe.clear()
                    probe.reset_capture_stats()
                probe.set_enabled(capture_train_oui)
                with autocast_context(device, precision):
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum
                probe.set_enabled(False)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += float(loss.item()) * grad_accum
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            clip_value = float(cfg["training"].get("gradient_clip", 1.0))
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else global_grad_norm(model.parameters())
            sync_if_cuda(device)
            timing_totals["train_core_seconds"] += time.perf_counter() - train_core_start
            before_update = None
            if is_scalar_metric_step:
                metric_start = time.perf_counter()
                before_update = snapshot_trainable_params(model)
                timing_totals["scalar_metrics_seconds"] += time.perf_counter() - metric_start
            sync_if_cuda(device)
            train_core_start = time.perf_counter()
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            sync_if_cuda(device)
            timing_totals["train_core_seconds"] += time.perf_counter() - train_core_start
            update_norm, relative_update_norm, weight_norm = None, None, None
            if is_scalar_metric_step and before_update is not None:
                metric_start = time.perf_counter()
                update_norm, relative_update_norm = update_norms(model, before_update)
                weight_norm = global_weight_norm(model.parameters())
                timing_totals["scalar_metrics_seconds"] += time.perf_counter() - metric_start
                timing_counts["scalar_metrics"] += 1
            last_train_loss = loss_accum / grad_accum
            current_lr = float(scheduler.get_last_lr()[0])

            if is_scalar_metric_step:
                append_jsonl(
                    run_dir / "metrics.jsonl",
                    {
                        "record_type": "train_metrics",
                        "step": step,
                        "train_loss": last_train_loss,
                        "validation_loss": None,
                        **last_label_stats,
                        "current_lr": current_lr,
                        "configured_lr": lr,
                        "current_weight_decay": weight_decay,
                        "configured_wd": weight_decay,
                        "tokens_seen": tokens_seen,
                        "examples_seen": examples_seen,
                        "global_gradient_norm": float(clipped_norm),
                        "grad_norm_pre_clip": float(clipped_norm),
                        "grad_norm_post_clip": min(float(clipped_norm), clip_value) if clip_value > 0 else float(clipped_norm),
                        "global_weight_norm": weight_norm,
                        "param_norm": weight_norm,
                        "update_norm": update_norm,
                        "relative_update_norm": relative_update_norm,
                        "update_to_weight_ratio": relative_update_norm,
                        "clipping_applied": float(clipped_norm) > clip_value if clip_value > 0 else False,
                        "wall_clock_time": time.time() - start_time,
                        "wall_time": time.time() - start_time,
                    },
                )
                log(f"step={step}/{max_steps} train_loss={last_train_loss:.4f} lr={current_lr:.3e}")

            if is_oui_step:
                train_capture_stats = probe.capture_stats("oui_train")
                oui_train_start = time.perf_counter()
                train_rows = probe.rows(step=step, split="train_batch", attention_mask=last_train_oui_masks)
                train_oui = write_oui_rows(train_rows, step=step, split="train_batch", output_path=run_dir / "oui_train.jsonl")
                oui_train_seconds = time.perf_counter() - oui_train_start
                timing_totals["oui_train_seconds"] += oui_train_seconds
                timing_counts["oui_train"] += 1
                sync_if_cuda(device)
                oui_probe_start = time.perf_counter()
                probe_metrics = measure_oui(model, probe, cfg, tokenizer, val_probe, device, precision, step=step, split="probe", output_path=run_dir / "oui_probe.jsonl")
                sync_if_cuda(device)
                oui_probe_seconds = time.perf_counter() - oui_probe_start
                timing_totals["oui_probe_seconds"] += oui_probe_seconds
                timing_counts["oui_probe"] += 1
                append_jsonl(
                    run_dir / "metrics.jsonl",
                    {
                        "record_type": "probe_metrics",
                        "step": step,
                        **probe_metrics,
                        "oui_train_mean": train_oui.get("oui_mean"),
                        "oui_train_min": train_oui.get("oui_min"),
                        "oui_train_max": train_oui.get("oui_max"),
                        "oui_train_std": train_oui.get("oui_std"),
                        "oui_train_encoder_mean": train_oui.get("oui_encoder_mean"),
                        "oui_train_decoder_mean": train_oui.get("oui_decoder_mean"),
                        "oui_train_seconds": oui_train_seconds,
                        "oui_train_compute_seconds": oui_train_seconds,
                        **train_capture_stats,
                        "oui_probe_seconds": oui_probe_seconds,
                        "current_lr": current_lr,
                        "configured_lr": lr,
                        "current_weight_decay": weight_decay,
                        "configured_wd": weight_decay,
                        "tokens_seen": tokens_seen,
                        "examples_seen": examples_seen,
                        "wall_clock_time": time.time() - start_time,
                        "wall_time": time.time() - start_time,
                    },
                )

            if step % eval_every == 0 or step == max_steps:
                sync_if_cuda(device)
                validation_start = time.perf_counter()
                metrics = evaluate_full_validation(model, cfg, tokenizer, device, precision)
                sync_if_cuda(device)
                validation_seconds = time.perf_counter() - validation_start
                timing_totals["full_validation_seconds"] += validation_seconds
                timing_totals["validation_loss_seconds"] += float(metrics.get("validation_loss_seconds", 0.0))
                validation_overhead_seconds = max(0.0, validation_seconds - float(metrics.get("validation_loss_seconds", 0.0)) - float(metrics.get("validation_extra_metrics_seconds", 0.0)))
                timing_totals["validation_extra_metrics_seconds"] += float(metrics.get("validation_extra_metrics_seconds", 0.0))
                timing_counts["full_validation"] += 1
                last_val_loss = metrics["validation_loss"]
                if metrics["validation_loss"] < best_val_loss:
                    best_val_loss = metrics["validation_loss"]
                    best_val_step = step
                if step == max_steps:
                    final_step_val_loss = metrics["validation_loss"]
                if "accuracy" in metrics:
                    last_task_metric = metrics["accuracy"]
                    best_task_metric = max(best_task_metric or 0.0, metrics["accuracy"])
                append_jsonl(
                    run_dir / "metrics.jsonl",
                    {
                        "record_type": "full_validation",
                        "step": step,
                        "train_loss": last_train_loss,
                        **metrics,
                        "validation_seconds": validation_seconds,
                        "validation_overhead_seconds": validation_overhead_seconds,
                        "current_lr": current_lr,
                        "configured_lr": lr,
                        "current_weight_decay": weight_decay,
                        "configured_wd": weight_decay,
                        "tokens_seen": tokens_seen,
                        "examples_seen": examples_seen,
                        "wall_clock_time": time.time() - start_time,
                        "wall_time": time.time() - start_time,
                    },
                )
                log(f"eval step={step}/{max_steps} val_loss={metrics['validation_loss']:.4f}")
            timing_totals["step_wall_seconds"] += time.perf_counter() - step_wall_start
            timing_counts["steps"] += 1

        saved_checkpoint = False
        total_time = time.time() - start_time
        measured_total = sum(timing_totals[key] for key in ("train_core_seconds", "scalar_metrics_seconds", "oui_train_seconds", "oui_probe_seconds", "full_validation_seconds"))
        timing_totals["other_overhead_seconds"] = max(0.0, total_time - measured_total)
        timing_percentages = {key.replace("_seconds", "_pct"): (value / total_time * 100.0 if total_time > 0 else 0.0) for key, value in timing_totals.items() if key.endswith("_seconds")}
        timing_averages = {key.replace("_seconds", "_avg_seconds"): timing_totals[key] / max(1, timing_counts.get(key.replace("_seconds", ""), timing_counts.get("steps", 1))) for key in timing_totals if key.endswith("_seconds")}
        summary.update(
            {
                "status": "smoke" if args.smoke else "completed",
                "last_train_loss": last_train_loss,
                "last_val_loss": last_val_loss,
                "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
                "best_val_step": best_val_step,
                "final_step": max_steps,
                "final_step_val_loss": final_step_val_loss,
                "best_task_metric": best_task_metric,
                "final_task_metric": last_task_metric,
                "total_steps": max_steps,
                "total_time": total_time,
                "wall_time": total_time,
                **timing_summary_aliases(timing_totals, total_time),
                **estimate_full_time(cfg, total_time, max_steps),
                "timing_seconds": timing_totals,
                "timing_percentages": timing_percentages,
                "timing_counts": timing_counts,
                "timing_averages": timing_averages,
                "oui_blocks": len(probe.block_meta),
                "checkpoint_saved": False,
                "integrity": {
                    "run_completed": True,
                    "nonfinite_in_summary": False,
                    "num_train_checkpoints": timing_counts.get("scalar_metrics", 0),
                    "num_metric_checkpoints": timing_counts.get("scalar_metrics", 0),
                    "num_evaluations": timing_counts.get("full_validation", 0),
                    "num_oui_train_measurements": timing_counts.get("oui_train", 0),
                    "num_oui_probe_measurements": timing_counts.get("oui_probe", 0),
                },
            }
        )
        summary["integrity"]["nonfinite_in_summary"] = has_nonfinite(summary)
        write_json(run_dir / "summary.json", summary)
        probe.close()
        log(f"Run finished | output_dir={run_dir}")
    except Exception as exc:
        summary.update({"status": "failed", "error_message": f"{type(exc).__name__}: {exc}", "total_time": time.time() - start_time, "checkpoint_saved": False})
        write_json(run_dir / "summary.json", summary)
        log(f"Run failed | {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
