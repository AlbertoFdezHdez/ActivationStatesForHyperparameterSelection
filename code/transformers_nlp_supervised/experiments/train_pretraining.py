from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import shutil
import time
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, T5Config, T5ForConditionalGeneration

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
    update_norms,
    timing_summary_aliases,
    write_json,
    write_yaml,
)

CONFIG_CLASSES = {"gpt2": GPT2Config, "llama": LlamaConfig, "t5": T5Config}
MODEL_CLASSES = {"gpt2": GPT2LMHeadModel, "llama": LlamaForCausalLM, "t5": T5ForConditionalGeneration}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OUI-Triage pretraining/continued-pretraining runner.")
    p.add_argument("--config", required=True)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", "--wd", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def get_text(example: dict[str, Any], columns: list[str]) -> str:
    parts = [str(example.get(column, "") or "") for column in columns]
    text = " ".join(part.strip() for part in parts if part and part.strip())
    if not text:
        for value in example.values():
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    return text if text else "empty example"


def load_split(cfg: dict[str, Any], split: str, *, streaming: bool):
    data_cfg = cfg["dataset"]
    local_dir = data_cfg.get("local_dir")
    if local_dir and Path(local_dir).exists():
        ds = datasets.load_from_disk(local_dir)
        return ds[split] if split in ds else ds
    kwargs = {"path": data_cfg["name"], "split": split, "streaming": streaming}
    if data_cfg.get("config"):
        kwargs["name"] = data_cfg["config"]
    if data_cfg.get("cache_dir"):
        kwargs["cache_dir"] = data_cfg["cache_dir"]
    return datasets.load_dataset(**kwargs)


def uses_train_holdout_validation(cfg: dict[str, Any]) -> bool:
    data_cfg = cfg["dataset"]
    return (
        int(data_cfg.get("validation_holdout_size", 0)) > 0
        and data_cfg.get("validation_split") == data_cfg.get("train_split")
    )


def split_for_role(cfg: dict[str, Any], split: str, *, role: str, streaming: bool):
    ds = load_split(cfg, split, streaming=streaming)
    if not uses_train_holdout_validation(cfg):
        if role in {"validation", "probe"}:
            subset_size = int(cfg["dataset"].get("validation_subset_size", 0))
            if subset_size > 0:
                return ds.take(subset_size) if streaming else ds.select(range(min(subset_size, len(ds))))
        return ds
    holdout_size = int(cfg["dataset"]["validation_holdout_size"])
    if role in {"validation", "probe"}:
        return ds.take(holdout_size) if streaming else ds.select(range(min(holdout_size, len(ds))))
    if role == "train":
        return ds.skip(holdout_size) if streaming else ds.select(range(min(holdout_size, len(ds)), len(ds)))
    return ds


def sample_texts(cfg: dict[str, Any], split: str, count: int, *, seed: int, streaming: bool) -> list[dict[str, Any]]:
    ds = split_for_role(cfg, split, role="probe", streaming=streaming)
    columns = cfg["dataset"].get("text_columns", ["text"])
    if streaming:
        if uses_train_holdout_validation(cfg):
            iterator = itertools.islice(ds, count)
        else:
            iterator = itertools.islice(ds.shuffle(seed=seed, buffer_size=max(1000, count * 20)), count)
    else:
        count = min(count, len(ds))
        iterator = (ds[i] for i in range(count))
    rows = []
    for idx, example in enumerate(iterator):
        text = get_text(example, columns)
        rows.append({"index": idx, "text": text, "text_hash": text_hash(text)})
    if not rows:
        raise RuntimeError(f"No examples collected for split={split}")
    return rows


def iter_text_batches(cfg: dict[str, Any], split: str, batch_size: int, *, streaming: bool):
    ds = split_for_role(cfg, split, role="validation", streaming=streaming)
    columns = cfg["dataset"].get("text_columns", ["text"])
    batch = []
    for example in ds:
        batch.append(get_text(example, columns))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def infinite_text_batches(cfg: dict[str, Any], split: str, batch_size: int, seed: int):
    columns = cfg["dataset"].get("text_columns", ["text"])
    streaming = bool(cfg["dataset"].get("streaming", False))
    while True:
        ds = split_for_role(cfg, split, role="train", streaming=streaming)
        if streaming:
            iterator = iter(ds.shuffle(seed=seed, buffer_size=int(cfg["dataset"].get("shuffle_buffer", 10000))))
        else:
            order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
            iterator = (ds[i] for i in order)
        batch = []
        for example in iterator:
            batch.append(get_text(example, columns))
            if len(batch) == batch_size:
                yield batch
                batch = []
        seed += 1


def span_corrupt_texts(
    texts: list[str],
    tokenizer,
    seed: int,
    step: int,
    *,
    noise_density: float,
    mean_span_length: float,
) -> tuple[list[str], list[str]]:
    rng = torch.Generator().manual_seed(seed + step * 1000003)
    sources = []
    targets = []
    for text in texts:
        words = text.split()
        if len(words) < 4:
            sources.append(text)
            targets.append(f"<extra_id_0> {text} <extra_id_1>")
            continue
        target_noise = max(1, min(len(words) - 1, int(round(len(words) * noise_density))))
        mask = [False] * len(words)
        masked_count = 0
        attempts = 0
        max_span = max(1, int(round(mean_span_length * 2)))
        while masked_count < target_noise and attempts < len(words) * 8:
            attempts += 1
            start = int(torch.randint(0, len(words), (1,), generator=rng).item())
            if mask[start]:
                continue
            span_len = int(torch.randint(1, max_span + 1, (1,), generator=rng).item())
            for idx in range(start, min(len(words), start + span_len)):
                if masked_count >= target_noise:
                    break
                if not mask[idx]:
                    mask[idx] = True
                    masked_count += 1

        spans_seen = 0
        in_span = False
        for idx, is_masked in enumerate(mask):
            if is_masked and not in_span:
                spans_seen += 1
                in_span = True
            elif not is_masked:
                in_span = False
            if spans_seen > 99:
                mask[idx] = False

        source_tokens: list[str] = []
        target_tokens: list[str] = []
        sentinel_idx = 0
        idx = 0
        while idx < len(words):
            if not mask[idx]:
                source_tokens.append(words[idx])
                idx += 1
                continue
            sentinel = f"<extra_id_{sentinel_idx}>"
            source_tokens.append(sentinel)
            target_tokens.append(sentinel)
            while idx < len(words) and mask[idx]:
                target_tokens.append(words[idx])
                idx += 1
            sentinel_idx += 1
        target_tokens.append(f"<extra_id_{sentinel_idx}>")
        sources.append(" ".join(source_tokens) or tokenizer.unk_token or "empty")
        targets.append(" ".join(target_tokens))
    return sources, targets


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


def build_batch(cfg: dict[str, Any], tokenizer, texts: list[str], *, device: torch.device, seed: int, step: int) -> dict[str, torch.Tensor]:
    train_cfg = cfg["training"]
    if cfg["task_type"] == "seq2seq_denoising":
        sources, targets = span_corrupt_texts(
            texts,
            tokenizer,
            seed,
            step,
            noise_density=float(train_cfg.get("noise_density", 0.15)),
            mean_span_length=float(train_cfg.get("mean_noise_span_length", 3.0)),
        )
        batch = tokenizer(sources, max_length=int(train_cfg["max_source_length"]), truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(targets, max_length=int(train_cfg["max_target_length"]), truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["decoder_attention_mask"] = labels.ne(-100).long()
    else:
        batch = tokenizer(texts, max_length=int(train_cfg["sequence_length"]), truncation=True, padding="max_length", return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
    return {k: v.to(device) for k, v in batch.items()}


def build_tokenizer(cfg: dict[str, Any]):
    tok_cfg = cfg["model"].get("tokenizer", {})
    tok = AutoTokenizer.from_pretrained(tok_cfg["name"], cache_dir=tok_cfg.get("cache_dir"), use_fast=True, clean_up_tokenization_spaces=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token or tok.sep_token
    return tok


def build_model(cfg: dict[str, Any], tokenizer):
    model_cfg = cfg["model"]
    if model_cfg.get("init", "scratch") == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], cache_dir=model_cfg.get("cache_dir"), trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model
    family = model_cfg["family"]
    params = dict(model_cfg.get("config", {}))
    params["vocab_size"] = len(tokenizer)
    params.setdefault("pad_token_id", tokenizer.pad_token_id)
    if family == "t5":
        params.setdefault("decoder_start_token_id", tokenizer.pad_token_id)
        params.setdefault("eos_token_id", tokenizer.eos_token_id)
    config = CONFIG_CLASSES[family](**params)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    return MODEL_CLASSES[family](config)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_probe(model, cfg, tokenizer, samples, device, precision) -> tuple[float, int]:
    model.eval()
    bs = int(cfg["oui"]["probe_batch_size"])
    total_loss = 0.0
    total_items = 0
    for batch_idx, start in enumerate(range(0, len(samples), bs)):
        texts = [row["text"] for row in samples[start:start + bs]]
        batch = build_batch(cfg, tokenizer, texts, device=device, seed=12345, step=batch_idx)
        with autocast_context(device, precision):
            outputs = model(**batch)
        total_loss += float(outputs.loss.item()) * len(texts)
        total_items += len(texts)
    model.train()
    if total_items == 0:
        raise RuntimeError("Probe split produced no batches")
    return total_loss / total_items, total_items


@torch.no_grad()
def evaluate_full_validation(model, cfg, tokenizer, device, precision, *, streaming: bool) -> tuple[float, int, float]:
    model.eval()
    bs = int(cfg["validation"]["batch_size"])
    max_batches = cfg["validation"].get("max_batches")
    max_batches = None if max_batches is None else int(max_batches)
    total_loss = 0.0
    total_items = 0
    loss_seconds = 0.0
    split = cfg["dataset"]["validation_split"]
    for batch_idx, texts in enumerate(iter_text_batches(cfg, split, bs, streaming=streaming)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = build_batch(cfg, tokenizer, texts, device=device, seed=54321, step=batch_idx)
        sync_if_cuda(device)
        loss_start = time.perf_counter()
        with autocast_context(device, precision):
            outputs = model(**batch)
        sync_if_cuda(device)
        loss_seconds += time.perf_counter() - loss_start
        total_loss += float(outputs.loss.item()) * len(texts)
        total_items += len(texts)
    model.train()
    if total_items == 0:
        raise RuntimeError(f"Validation split {split} produced no batches")
    return total_loss / total_items, total_items, loss_seconds


@torch.no_grad()
def measure_oui(model, probe, cfg, tokenizer, samples, device, precision, *, step: int, split: str, output_path: Path) -> tuple[float, int, dict[str, Any], dict[str, float]]:
    model.eval()
    bs = int(cfg["oui"]["probe_batch_size"])
    all_rows = []
    total_loss = 0.0
    total_items = 0
    forward_seconds = 0.0
    compute_seconds = 0.0
    probe.reset_capture_stats()
    for batch_idx, start in enumerate(range(0, len(samples), bs)):
        texts = [row["text"] for row in samples[start:start + bs]]
        batch = build_batch(cfg, tokenizer, texts, device=device, seed=777, step=step + batch_idx)
        probe.clear()
        probe.set_enabled(True)
        sync_if_cuda(device)
        forward_start = time.perf_counter()
        with autocast_context(device, precision):
            outputs = model(**batch)
        sync_if_cuda(device)
        probe.set_enabled(False)
        forward_seconds += time.perf_counter() - forward_start
        total_loss += float(outputs.loss.item()) * len(texts)
        total_items += len(texts)
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
    if total_items == 0:
        raise RuntimeError("OUI probe produced no batches")
    return total_loss / total_items, total_items, oui_aggregate, {"oui_probe_forward_seconds": forward_seconds, "oui_probe_compute_seconds": compute_seconds, **probe.capture_stats("oui_probe")}


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
    wd = float(args.weight_decay if args.weight_decay is not None else cfg["training"]["weight_decays"][0])
    seed = int(args.seed if args.seed is not None else cfg["training"].get("seeds", [0])[0])
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = int(args.max_steps)
    cfg["training"]["selected_lr"] = lr
    cfg["training"]["selected_weight_decay"] = wd
    cfg["training"]["selected_seed"] = seed

    set_seed(seed)
    device = setup_torch()
    precision = resolve_precision(args.precision or cfg["training"].get("precision", "auto"), device)
    run_dir = build_run_dir(cfg, lr=lr, weight_decay=wd, seed=seed, output_dir=args.output_dir, smoke=args.smoke)
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
        model = build_model(cfg, tokenizer).to(device)
        optimizer = torch.optim.AdamW(parameter_groups(model, wd), lr=lr, betas=tuple(cfg["training"].get("betas", [0.95, 0.95])))
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
        max_steps = int(cfg["training"]["max_steps"])
        scheduler = build_scheduler(optimizer, scheduler_type=str(cfg["training"].get("scheduler", "cosine")), warmup_steps=max(1, int(max_steps * float(cfg["training"]["warmup_ratio"]))), total_steps=max_steps, min_lr_ratio=float(cfg["training"].get("min_lr_ratio", 0.1)))
        probe = FFNOUIProbe(model, cfg["oui"]["hook_patterns"], sequence_mode=cfg["oui"].get("sequence_mode", "last_token"), max_token_samples=cfg["oui"].get("max_token_samples"))
        if not probe.block_meta:
            raise RuntimeError(f"No FFN modules matched OUI patterns: {cfg['oui']['hook_patterns']}")

        streaming = bool(cfg["dataset"].get("streaming", False))
        val_probe = sample_texts(cfg, cfg["dataset"]["validation_split"], int(cfg["oui"]["probe_size"]), seed=seed + 1, streaming=streaming)
        write_json(run_dir / "probe_indices.json", {"val_probe": val_probe})
        train_batches = infinite_text_batches(cfg, cfg["dataset"]["train_split"], int(cfg["training"]["batch_size"]), seed)

        grad_accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
        eval_every = int(cfg["validation"]["eval_every"])
        oui_every = int(cfg["oui"]["measure_every"])
        best_val_loss = float("inf")
        best_val_step = None
        last_val_loss = None
        final_step_val_loss = None
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
                texts = next(train_batches)
                batch = build_batch(cfg, tokenizer, texts, device=device, seed=seed, step=step * grad_accum + accum_idx)
                examples_seen += len(texts)
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
                append_jsonl(run_dir / "metrics.jsonl", {"record_type": "train_metrics", "step": step, "train_loss": last_train_loss, "validation_loss": None, "validation_perplexity": None, **last_label_stats, "current_lr": current_lr, "configured_lr": lr, "current_weight_decay": wd, "configured_wd": wd, "tokens_seen": tokens_seen, "examples_seen": examples_seen, "global_gradient_norm": float(clipped_norm), "grad_norm_pre_clip": float(clipped_norm), "grad_norm_post_clip": min(float(clipped_norm), clip_value) if clip_value > 0 else float(clipped_norm), "global_weight_norm": weight_norm, "param_norm": weight_norm, "update_norm": update_norm, "relative_update_norm": relative_update_norm, "update_to_weight_ratio": relative_update_norm, "clipping_applied": float(clipped_norm) > clip_value if clip_value > 0 else False, "wall_clock_time": time.time() - start_time, "wall_time": time.time() - start_time})
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
                probe_loss, probe_items, probe_oui, probe_timing = measure_oui(model, probe, cfg, tokenizer, val_probe, device, precision, step=step, split="probe", output_path=run_dir / "oui_probe.jsonl")
                sync_if_cuda(device)
                oui_probe_seconds = time.perf_counter() - oui_probe_start
                timing_totals["oui_probe_seconds"] += oui_probe_seconds
                timing_counts["oui_probe"] += 1
                append_jsonl(run_dir / "metrics.jsonl", {"record_type": "probe_metrics", "step": step, "probe_loss": probe_loss, "probe_perplexity": math.exp(min(probe_loss, 50.0)) if cfg["task_type"] != "seq2seq_denoising" else None, "probe_items": probe_items, "oui_train_mean": train_oui.get("oui_mean"), "oui_train_min": train_oui.get("oui_min"), "oui_train_max": train_oui.get("oui_max"), "oui_train_std": train_oui.get("oui_std"), "oui_train_encoder_mean": train_oui.get("oui_encoder_mean"), "oui_train_decoder_mean": train_oui.get("oui_decoder_mean"), "oui_probe_mean": probe_oui.get("oui_mean"), "oui_probe_min": probe_oui.get("oui_min"), "oui_probe_max": probe_oui.get("oui_max"), "oui_probe_std": probe_oui.get("oui_std"), "oui_probe_encoder_mean": probe_oui.get("oui_encoder_mean"), "oui_probe_decoder_mean": probe_oui.get("oui_decoder_mean"), "oui_train_seconds": oui_train_seconds, "oui_train_compute_seconds": oui_train_seconds, **train_capture_stats, "oui_probe_seconds": oui_probe_seconds, **probe_timing, "current_lr": current_lr, "configured_lr": lr, "current_weight_decay": wd, "configured_wd": wd, "tokens_seen": tokens_seen, "examples_seen": examples_seen, "wall_clock_time": time.time() - start_time, "wall_time": time.time() - start_time})

            if step % eval_every == 0 or step == max_steps:
                sync_if_cuda(device)
                validation_start = time.perf_counter()
                val_loss, val_items, validation_loss_seconds = evaluate_full_validation(model, cfg, tokenizer, device, precision, streaming=streaming)
                sync_if_cuda(device)
                validation_seconds = time.perf_counter() - validation_start
                timing_totals["full_validation_seconds"] += validation_seconds
                timing_totals["validation_loss_seconds"] += validation_loss_seconds
                validation_overhead_seconds = max(0.0, validation_seconds - validation_loss_seconds)
                timing_totals["validation_extra_metrics_seconds"] += 0.0
                timing_counts["full_validation"] += 1
                last_val_loss = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                if step == max_steps:
                    final_step_val_loss = val_loss
                append_jsonl(run_dir / "metrics.jsonl", {"record_type": "full_validation", "step": step, "train_loss": last_train_loss, "validation_loss": val_loss, "validation_perplexity": math.exp(min(val_loss, 50.0)) if cfg["task_type"] != "seq2seq_denoising" else None, "validation_items": val_items, "validation_seconds": validation_seconds, "validation_loss_seconds": validation_loss_seconds, "validation_extra_metrics_seconds": 0.0, "validation_overhead_seconds": validation_overhead_seconds, "current_lr": current_lr, "configured_lr": lr, "current_weight_decay": wd, "configured_wd": wd, "tokens_seen": tokens_seen, "examples_seen": examples_seen, "wall_clock_time": time.time() - start_time, "wall_time": time.time() - start_time})
                log(f"eval step={step}/{max_steps} val_loss={val_loss:.4f}")
            timing_totals["step_wall_seconds"] += time.perf_counter() - step_wall_start
            timing_counts["steps"] += 1

        total_time = time.time() - start_time
        measured_total = sum(timing_totals[key] for key in ("train_core_seconds", "scalar_metrics_seconds", "oui_train_seconds", "oui_probe_seconds", "full_validation_seconds"))
        timing_totals["other_overhead_seconds"] = max(0.0, total_time - measured_total)
        timing_percentages = {key.replace("_seconds", "_pct"): (value / total_time * 100.0 if total_time > 0 else 0.0) for key, value in timing_totals.items() if key.endswith("_seconds")}
        timing_averages = {key.replace("_seconds", "_avg_seconds"): timing_totals[key] / max(1, timing_counts.get(key.replace("_seconds", ""), timing_counts.get("steps", 1))) for key in timing_totals if key.endswith("_seconds")}
        summary.update({"status": "smoke" if args.smoke else "completed", "last_train_loss": last_train_loss, "last_val_loss": last_val_loss, "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss, "best_val_step": best_val_step, "final_step": max_steps, "final_step_val_loss": final_step_val_loss, "total_steps": max_steps, "total_time": total_time, "wall_time": total_time, **timing_summary_aliases(timing_totals, total_time), **estimate_full_time(cfg, total_time, max_steps), "timing_seconds": timing_totals, "timing_percentages": timing_percentages, "timing_counts": timing_counts, "timing_averages": timing_averages, "oui_blocks": len(probe.block_meta), "checkpoint_saved": False, "integrity": {"run_completed": True, "nonfinite_in_summary": False, "num_train_checkpoints": timing_counts.get("scalar_metrics", 0), "num_metric_checkpoints": timing_counts.get("scalar_metrics", 0), "num_evaluations": timing_counts.get("full_validation", 0), "num_oui_train_measurements": timing_counts.get("oui_train", 0), "num_oui_probe_measurements": timing_counts.get("oui_probe", 0)}})
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
