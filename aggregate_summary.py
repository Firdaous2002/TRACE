from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple, TypeGuard

import numpy as np


MODELS: List[str] = ["sasrec", "bert4rec", "gru4rec"]
DATASETS: List[str] = ["ml-100k", "ml-1m", "steam", "beauty", "sports"]
METHODS: List[str] = ["random", "greedy", "ga", "beam", "gradient"]

TASKS: List[str] = ["bring_in", "push_out"]

METHOD_LABEL = {
    "random": "Random",
    "greedy": "Greedy",
    "ga": "GECE",
    "beam": "TRACE",
    "gradient": "Gradient-CF",
}


def _is_number(value: object) -> TypeGuard[int | float]:
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def parse_dotlist(overrides: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _task_regex(task: Optional[str]) -> str:
    if task is None:
        return ""
    if task == "bring_in":
        return r"_targeted_r\d+_\d+"
    if task == "push_out":
        return r"_pushout_top\d+"
    raise ValueError(f"Unknown task: {task!r} (expected one of {TASKS} or None)")


def find_method_seed_files(
    results_dir: str,
    model: str,
    dataset: str,
    method: str,
    *,
    task: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """Return [(seed, path)] sorted by seed for files matching exactly the given task/run_tag.

    When task and run_tag are both None, only matches files with no suffix between method
    and _seed<N>.json (the pre-task legacy naming).
    """
    # Broad glob then regex-filter for correctness (glob is ambiguous across dashes/underscores).
    broad = os.path.join(results_dir, f"explain_{model}_{dataset}_{method}*seed*.json")
    tag_regex = f"_{re.escape(run_tag)}" if run_tag else ""
    task_regex = _task_regex(task)
    pattern = re.compile(
        rf"^explain_{re.escape(model)}_{re.escape(dataset)}_{re.escape(method)}{task_regex}{tag_regex}_seed(\d+)\.json$"
    )

    seed_files: List[Tuple[int, str]] = []
    for path in glob(broad):
        basename = os.path.basename(path)
        m = pattern.match(basename)
        if m is None:
            continue
        seed_files.append((int(m.group(1)), path))

    seed_files.sort(key=lambda x: x[0])
    return seed_files


def _collect_numeric_metric_keys(metrics_by_seed: List[Tuple[int, Dict[str, object]]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for _, metrics in metrics_by_seed:
        for k, v in metrics.items():
            if k not in seen and _is_number(v):
                keys.append(k)
                seen.add(k)
    return keys


def aggregate_method_from_seed_metrics(metrics_by_seed: List[Tuple[int, Dict[str, object]]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not metrics_by_seed:
        return agg

    metric_keys = _collect_numeric_metric_keys(metrics_by_seed)
    for key in metric_keys:
        values: List[float] = []
        for _, metrics in metrics_by_seed:
            value = metrics.get(key)
            if _is_number(value):
                values.append(float(value))
        if not values:
            continue
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    return agg


def load_method_seed_metrics(
    results_dir: str,
    model: str,
    dataset: str,
    method: str,
    *,
    task: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Tuple[List[int], List[Tuple[int, Dict[str, object]]]]:
    seed_files = find_method_seed_files(results_dir, model, dataset, method, task=task, run_tag=run_tag)
    seeds: List[int] = []
    metrics_by_seed: List[Tuple[int, Dict[str, object]]] = []

    for seed, file_path in seed_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}")
            continue

        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            print(f"[WARN] Missing/invalid metrics in {file_path}; skipping")
            continue

        seeds.append(seed)
        metrics_by_seed.append((seed, metrics))

    return seeds, metrics_by_seed


def print_seed_table(seeds_by_method: Dict[str, List[int]], model: str, dataset: str, task: Optional[str], run_tag: Optional[str]) -> None:
    scope = f" [task={task}]" if task else ""
    scope += f" [tag={run_tag}]" if run_tag else ""
    print(f"\n=== Seed availability: {model} / {dataset}{scope} ===")
    print(f"{'Method':<10} {'Seeds':<20} {'Count':>5}")
    print("-" * 40)
    for method in METHODS:
        seeds = seeds_by_method.get(method, [])
        seeds_str = ", ".join(str(s) for s in seeds) if seeds else "-"
        print(f"{method:<10} {seeds_str:<20} {len(seeds):>5}")


def _summary_filename(model: str, dataset: str, task: Optional[str], run_tag: Optional[str]) -> str:
    parts = [f"summary_{model}_{dataset}"]
    if task:
        parts.append({"bring_in": "bringin", "push_out": "pushout"}[task])
    if run_tag:
        parts.append(run_tag)
    return "_".join(parts) + ".json"


def aggregate_model_dataset(
    model: str,
    dataset: str,
    results_dir: str,
    *,
    task: Optional[str] = None,
    run_tag: Optional[str] = None,
    methods: Optional[List[str]] = None,
    write_summary: bool = True,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    seeds_by_method: Dict[str, List[int]] = {}
    methods_to_run = methods if methods is not None else METHODS

    for method in methods_to_run:
        seeds, metrics_by_seed = load_method_seed_metrics(
            results_dir, model, dataset, method, task=task, run_tag=run_tag
        )
        seeds_by_method[method] = seeds

        if not metrics_by_seed:
            print(f"[WARN] No valid seed files for {model}/{dataset}/{method}"
                  f"{f' (task={task})' if task else ''}"
                  f"{f' (tag={run_tag})' if run_tag else ''}; skipping")
            continue

        summary[method] = aggregate_method_from_seed_metrics(metrics_by_seed)

    print_seed_table(seeds_by_method, model, dataset, task, run_tag)

    if write_summary:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, _summary_filename(model, dataset, task, run_tag))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Wrote {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed explanation outputs")
    parser.add_argument("--all", action="store_true", help="Regenerate all (model, dataset) summaries")
    parser.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Task suffix filter: bring_in -> *_targeted_r*_*; push_out -> *_pushout_top*; None -> legacy no-suffix files")
    parser.add_argument("--run-tag", default=None,
                        help="Additional run_tag suffix to match (e.g. 'ablation-full')")
    parser.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (model.name=..., data.dataset=...)")
    args = parser.parse_args()

    results_dir = args.results_dir

    if args.all:
        for dataset in DATASETS:
            for model in MODELS:
                aggregate_model_dataset(
                    model=model, dataset=dataset, results_dir=results_dir,
                    task=args.task, run_tag=args.run_tag, write_summary=True,
                )
        return

    cli = parse_dotlist(args.overrides)
    model = cli.get("model.name")
    dataset = cli.get("data.dataset")

    if not model or not dataset:
        raise SystemExit(
            "Please provide model.name and data.dataset overrides, or use --all. "
            "Example: python aggregate_summary.py --task bring_in model.name=bert4rec data.dataset=ml-100k"
        )

    aggregate_model_dataset(
        model=model, dataset=dataset, results_dir=results_dir,
        task=args.task, run_tag=args.run_tag, write_summary=True,
    )


if __name__ == "__main__":
    main()
