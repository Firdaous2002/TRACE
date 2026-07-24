from __future__ import annotations

import json
import os
import random
import re
import gc
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from data.datasets import load_category_map, load_dataset, load_popular_items
from models.trainer import load_trained_model


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str, overrides: list[str] | None = None):
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        return OmegaConf.merge(base_cfg, cli_cfg)
    return base_cfg


def clone_config(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def clear_device_cache(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_experiment_selection(cfg, model_name: str | None = None, dataset: str | None = None, target_mode: str | None = None):
    if model_name is not None:
        cfg.model.name = model_name
    if dataset is not None:
        cfg.data.dataset = dataset
    cfg.training.device = resolve_device(cfg.training.device)
    cfg.targeted.enabled = target_mode is not None
    if target_mode is not None:
        cfg.targeted.mode = target_mode
    return cfg


def get_rank_constraint_spec(*, cfg=None, target_mode: str | None = None) -> dict[str, Any] | None:
    target_cfg = getattr(cfg, "targeted", None) if cfg is not None else None
    enabled = bool(getattr(target_cfg, "enabled", False)) if target_cfg is not None else (target_mode is not None)
    if not enabled:
        return None

    raw_kind = str(getattr(target_cfg, "constraint", "bring_in")) if target_cfg is not None else "bring_in"
    kind = raw_kind.strip().lower().replace("-", "_")
    if kind not in {"bring_in", "push_out"}:
        raise ValueError(f"Unknown targeted.constraint: {raw_kind}")

    top_k = int(getattr(target_cfg, "top_k", 1)) if target_cfg is not None else 1
    top_k = max(1, top_k)
    rank_min = getattr(target_cfg, "rank_min", None) if target_cfg is not None else None
    rank_max = getattr(target_cfg, "rank_max", None) if target_cfg is not None else None

    if kind == "push_out":
        return {
            "kind": "push_out",
            "top_k": top_k,
        }

    resolved_mode = target_mode
    if resolved_mode is None and target_cfg is not None:
        resolved_mode = str(getattr(target_cfg, "mode", "easy"))
    spec = {
        "kind": "bring_in",
        "top_k": top_k,
    }
    if rank_min is not None or rank_max is not None:
        if rank_min is not None:
            spec["rank_min"] = int(rank_min)
        if rank_max is not None:
            spec["rank_max"] = int(rank_max)
        return spec
    spec["mode"] = resolved_mode or "easy"
    return spec


def target_suffix(target_mode: str | None = None, cfg=None) -> str:
    rank_constraint = get_rank_constraint_spec(cfg=cfg, target_mode=target_mode)
    if rank_constraint is None:
        return ""
    if rank_constraint["kind"] == "push_out":
        return f"_pushout_top{rank_constraint['top_k']}"
    rank_min = rank_constraint.get("rank_min")
    rank_max = rank_constraint.get("rank_max")
    if rank_min is not None or rank_max is not None:
        lo = int(rank_min if rank_min is not None else 1)
        hi = int(rank_max if rank_max is not None else lo)
        return f"_targeted_r{lo}_{hi}"
    resolved_target_mode = str(rank_constraint.get("mode", target_mode or "easy"))
    return f"_targeted_{resolved_target_mode}" if resolved_target_mode else ""


def sanitize_tag(tag: str | None) -> str:
    if not tag:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(tag).strip())
    cleaned = cleaned.strip("-_.")
    return cleaned


def run_tag_suffix(tag: str | None = None, cfg=None) -> str:
    if tag is None and cfg is not None:
        evaluation_cfg = getattr(cfg, "evaluation", None)
        if evaluation_cfg is not None:
            tag = getattr(evaluation_cfg, "run_tag", None)
    cleaned = sanitize_tag(tag)
    return f"_{cleaned}" if cleaned else ""


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_path(path: str, base_dir: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    anchor = base_dir or os.getcwd()
    return os.path.abspath(os.path.join(anchor, path))


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def build_checkpoint_path(cfg, model_name: str | None = None, dataset: str | None = None) -> str:
    selected_model = model_name or cfg.model.name
    selected_dataset = dataset or cfg.data.dataset
    return os.path.join(cfg.training.checkpoint_dir, f"{selected_model}_{selected_dataset}.pt")


def load_experiment_model(cfg, model_name: str | None = None, dataset: str | None = None):
    checkpoint_path = build_checkpoint_path(cfg, model_name=model_name, dataset=dataset)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_trained_model(checkpoint_path, device=cfg.training.device)
    model.eval()

    return model, checkpoint_path


def load_experiment_data(cfg, dataset: str | None = None) -> dict[str, Any]:
    selected_dataset = dataset or cfg.data.dataset
    user_seqs, info = load_dataset(cfg.data.data_dir, selected_dataset)
    return {
        "user_seqs": user_seqs,
        "info": info,
        "popular_items": load_popular_items(cfg.data.data_dir, selected_dataset),
        "category_map": load_category_map(cfg.data.data_dir, selected_dataset),
    }


def validate_data_model_compatibility(user_seqs, num_items: int) -> None:
    max_item = max((max(seq) for seq in user_seqs.values() if seq), default=0)
    assert max_item <= num_items, f"Data/model mismatch: max_item={max_item}, num_items={num_items}"


def numeric_stats(seed_metrics: list[dict]) -> dict[str, float]:
    agg: dict[str, float] = {}
    keys = sorted({key for metrics in seed_metrics for key in metrics.keys()})
    for key in keys:
        values = []
        for metrics in seed_metrics:
            value = metrics.get(key)
            if not isinstance(value, (int, float)):
                continue
            if not np.isfinite(value):
                continue
            values.append(value)
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg
