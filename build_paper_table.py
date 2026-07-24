from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from aggregate_summary import (
    DATASETS,
    METHODS,
    METHOD_LABEL,
    MODELS,
    aggregate_model_dataset,
    find_method_seed_files,
)
from data.datasets import load_category_map, load_dataset, load_popular_items
from evaluation.evaluator import build_explainer, run_explanation_for_users
from evaluation.metrics import compute_sequence_nll_batch
from models.trainer import load_trained_model


MODEL_LABEL = {
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "gru4rec": "GRU4Rec",
}

DATASET_LABEL = {
    "ml-1m": "ML-1M",
    "ml-100k": "ML-100K",
    "steam": "Steam",
    "beauty": "Beauty",
    "sports": "Sports",
}

METRIC_SPEC = [
    ("validity", "validity_rate_mean", "max"),
    ("hamming", "hamming_mean_mean", "min"),
    ("queries", "queries_mean", "min"),
    ("delta_nll", "delta_nll", "min"),
]


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_validity(m: Dict[str, float | None]) -> str:
    val = m.get("validity_rate_mean")
    std = m.get("validity_rate_std")
    if val is None:
        return "N/A"
    return f"{val:.3f}$\\pm${std:.3f}" if std is not None else f"{val:.3f}"


def _format_hamming(m: Dict[str, float | None]) -> str:
    val = m.get("hamming_mean_mean")
    std = m.get("hamming_mean_std")
    if val is None or not math.isfinite(val):
        return "N/A"
    return f"{val:.2f}$\\pm${std:.2f}" if std is not None and math.isfinite(std) else f"{val:.2f}"


def _format_queries(m: Dict[str, float | None]) -> str:
    val = m.get("queries_mean")
    if val is None:
        return "N/A"
    return f"{val:.0f}"


def _format_delta_nll(m: Dict[str, float | None]) -> str:
    value = m.get("delta_nll")
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def _latex_wrap(text: str, bold: bool) -> str:
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _best_methods_for_model(method_records: Dict[str, Dict[str, float | None]]) -> Dict[str, List[str]]:
    best: Dict[str, List[str]] = {}
    eps = 1e-12
    for metric_name, key, direction in METRIC_SPEC:
        vals: List[Tuple[str, float]] = []
        for method, rec in method_records.items():
            if key == "queries_mean" and method == "random":
                continue
            value = rec.get(key)
            if value is None or not math.isfinite(value):
                continue
            vals.append((method, value))
        if not vals:
            best[metric_name] = []
            continue
        target = max(v for _, v in vals) if direction == "max" else min(v for _, v in vals)
        best[metric_name] = [method for method, val in vals if abs(val - target) <= eps]
    return best


def _summary_path(results_dir: str, model: str, dataset: str, task: Optional[str]) -> str:
    task_suffix = ""
    if task:
        task_suffix = "_" + {"bring_in": "bringin", "push_out": "pushout"}[task]
    return os.path.join(results_dir, f"summary_{model}_{dataset}{task_suffix}.json")


def load_or_aggregate_summary(
    model: str, dataset: str, results_dir: str, *, task: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    summary_path = _summary_path(results_dir, model, dataset, task)
    if os.path.exists(summary_path):
        try:
            summary = _read_json(summary_path)
            if isinstance(summary, dict):
                return summary
        except Exception as exc:
            print(f"[WARN] Failed reading {summary_path}: {exc}; falling back to per-seed aggregation")
    else:
        print(f"[INFO] {summary_path} missing; aggregating per-seed inline")

    return aggregate_model_dataset(
        model=model, dataset=dataset, results_dir=results_dir,
        task=task, write_summary=False,
    )


def _load_saved_per_user(path: str) -> Dict[int, Dict] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    per_user = payload.get("per_user", [])
    return {int(row["user_id"]): row for row in per_user if "user_id" in row}


def _select_input_sequences(user_seqs: Dict[int, List[int]], max_len: int, seed: int, n_users: int) -> Dict[int, List[int]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_uids = sorted(user_seqs.keys())
    test_uids = random.sample(all_uids, n_users) if len(all_uids) > n_users else all_uids
    out: Dict[int, List[int]] = {}
    for uid in test_uids:
        seq = user_seqs[uid]
        if len(seq) < 3:
            continue
        input_seq = seq[:-1][-max_len:]
        pad_len = max_len - len(input_seq)
        out[uid] = [0] * pad_len + list(input_seq)
    return out


def _extract_saved_cf_sequences(method: str, per_user_by_uid: Dict[int, Dict]) -> Dict[int, List[int] | None]:
    seq_by_uid: Dict[int, List[int] | None] = {}
    for uid, row in per_user_by_uid.items():
        if not bool(row.get("valid", False)):
            seq_by_uid[uid] = None
            continue
        # Methods that save top-k CFs: ga (GECE), gradient, beam (TRACE)
        if method in ("ga", "gradient", "beam"):
            top_k = row.get("top_k_cfs", [])
            cf_entry = None
            if isinstance(top_k, list) and top_k:
                for entry in top_k:
                    if isinstance(entry, dict) and entry.get("valid", False) and "seq" in entry:
                        cf_entry = entry
                        break
                if cf_entry is None and isinstance(top_k[0], dict) and "seq" in top_k[0]:
                    cf_entry = top_k[0]
            seq_by_uid[uid] = cf_entry["seq"] if cf_entry is not None else None
        else:
            seq_by_uid[uid] = None
    return seq_by_uid


def _rerun_sequences(cfg, model, user_seqs, popular_items, category_map, method: str, seed: int, device: str, max_len: int):
    exp_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    exp_cfg.explainer.name = method
    explainer = build_explainer(exp_cfg, model, model.num_items)
    run = run_explanation_for_users(
        model=model,
        explainer=explainer,
        user_seqs=user_seqs,
        max_len=max_len,
        popular_items=popular_items,
        category_map=category_map,
        k_pos=cfg.explainer.k_pos,
        k_item=cfg.explainer.k_item,
        n_users=cfg.evaluation.n_users,
        device=device,
        seed=seed,
    )
    seq_by_uid = {}
    for row, cf_seq in zip(run.get("per_user_results", []), run.get("cf_seqs", [])):
        uid = int(row["user_id"])
        seq_by_uid[uid] = cf_seq if row.get("valid", False) else None
    return seq_by_uid


def _compute_delta_nll_stats(
    cfg, model_name: str, dataset: str, method: str, results_dir: str, *, task: Optional[str] = None,
) -> Dict[str, float] | None:
    ckpt = os.path.join(cfg.training.checkpoint_dir, f"{model_name}_{dataset}.pt")
    if not os.path.exists(ckpt):
        return None

    user_seqs, _ = load_dataset(cfg.data.data_dir, dataset)
    popular_items = load_popular_items(cfg.data.data_dir, dataset)
    category_map = load_category_map(cfg.data.data_dir, dataset)
    device = cfg.training.device
    model = load_trained_model(ckpt, device=device)
    model.eval()
    model_max_len = int(getattr(model, "max_len", cfg.data.max_len))

    seed_means: List[float] = []
    for seed, path in find_method_seed_files(results_dir, model_name, dataset, method, task=task):
        per_user_by_uid = _load_saved_per_user(path)
        if not per_user_by_uid:
            continue
        original_seq_by_uid = _select_input_sequences(user_seqs, model_max_len, seed, cfg.evaluation.n_users)
        if method in ("random", "greedy"):
            cf_seq_by_uid = _rerun_sequences(cfg, model, user_seqs, popular_items, category_map, method, seed, device, model_max_len)
        else:
            cf_seq_by_uid = _extract_saved_cf_sequences(method, per_user_by_uid)

        orig_batch: List[List[int]] = []
        cf_batch: List[List[int]] = []
        for uid, row in per_user_by_uid.items():
            if not bool(row.get("valid", False)) or uid not in original_seq_by_uid:
                continue
            cf_seq = cf_seq_by_uid.get(uid)
            if cf_seq is None:
                continue
            orig_batch.append(original_seq_by_uid[uid])
            cf_batch.append(list(cf_seq))

        if not orig_batch:
            continue

        with torch.no_grad():
            orig_nlls = np.array(compute_sequence_nll_batch(model, orig_batch, max_len=model_max_len, device=device), dtype=np.float64)
            cf_nlls = np.array(compute_sequence_nll_batch(model, cf_batch, max_len=model_max_len, device=device), dtype=np.float64)

        finite_mask = np.isfinite(orig_nlls) & np.isfinite(cf_nlls)
        if not np.any(finite_mask):
            continue
        deltas = cf_nlls[finite_mask] - orig_nlls[finite_mask]
        if deltas.size:
            seed_means.append(float(np.mean(deltas)))

    if not seed_means:
        return None
    return {
        "delta_nll": float(np.mean(seed_means)),
        "delta_nll_std": float(np.std(seed_means)),
    }


def build_records(
    results_dir: str, *, task: Optional[str] = None, recompute_delta_nll: bool = True,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]]:
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config", "default.yaml"))
    records: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]] = {}
    for dataset in DATASETS:
        records[dataset] = {}
        for model in MODELS:
            summary = load_or_aggregate_summary(model=model, dataset=dataset, results_dir=results_dir, task=task)
            model_records: Dict[str, Dict[str, float | None]] = {}

            for method in METHODS:
                seed_count = len(find_method_seed_files(results_dir, model, dataset, method, task=task))
                if seed_count == 0:
                    print(f"[WARN] {model}/{dataset}/{method} (task={task}) has 0 seeds; skipping")
                    continue

                summary_metrics = summary.get(method)
                if not isinstance(summary_metrics, dict):
                    print(f"[WARN] Missing summary metrics for {model}/{dataset}/{method}; skipping")
                    continue
                if "validity_rate_mean" not in summary_metrics:
                    print(f"[WARN] Missing validity_rate_mean for {model}/{dataset}/{method}; skipping")
                    continue

                metric_map: Mapping[str, object] = summary_metrics
                validity_rate_mean = metric_map.get("validity_rate_mean")
                validity_rate_std = metric_map.get("validity_rate_std", 0.0)
                if not isinstance(validity_rate_mean, (int, float)) or not isinstance(validity_rate_std, (int, float)):
                    print(f"[WARN] Invalid validity stats for {model}/{dataset}/{method}; skipping")
                    continue

                def _safe_float(key: str) -> float | None:
                    value = metric_map.get(key)
                    if not isinstance(value, (int, float)):
                        return None
                    value = float(value)
                    return value if math.isfinite(value) else None

                # delta_nll is written either by _compute_delta_nll_stats (as `delta_nll`)
                # or by seed aggregation (as `delta_nll_mean`). Prefer the recomputed value
                # when present, fall back to the aggregated mean across seeds.
                delta_nll = _safe_float("delta_nll")
                if delta_nll is None:
                    delta_nll = _safe_float("delta_nll_mean")

                model_records[method] = {
                    "seed_count": float(seed_count),
                    "validity_rate_mean": float(validity_rate_mean),
                    "validity_rate_std": float(validity_rate_std),
                    "hamming_mean_mean": _safe_float("hamming_mean_mean"),
                    "hamming_mean_std": _safe_float("hamming_mean_std"),
                    "queries_mean": _safe_float("queries_mean"),
                    "queries_std": _safe_float("queries_std"),
                    "delta_nll": delta_nll,
                    "delta_nll_std": _safe_float("delta_nll_std"),
                }

                if recompute_delta_nll:
                    recomputed_delta = _compute_delta_nll_stats(cfg, model, dataset, method, results_dir, task=task)
                    if recomputed_delta is not None:
                        model_records[method]["delta_nll"] = recomputed_delta["delta_nll"]
                        model_records[method]["delta_nll_std"] = recomputed_delta["delta_nll_std"]

            records[dataset][model] = model_records
    return records


def print_console_tables(records: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]]) -> None:
    for dataset in DATASETS:
        print(f"\n=== {DATASET_LABEL[dataset]} ===")
        print(f"{'Model':<10} {'Method':<12} {'Validity':<12} {'Hamming':<11} {'Queries':>8} {'ΔNLL':>8}")
        print("-" * 69)
        for model in MODELS:
            model_records = records[dataset][model]
            if not model_records:
                print(f"{MODEL_LABEL[model]:<10} {'-':<12} {'-':<12} {'-':<11} {'-':>8} {'-':>8}")
                continue

            best_per_metric = _best_methods_for_model(model_records)
            wins = {method: 0 for method in model_records}
            for metric_name, _, _ in METRIC_SPEC:
                for method in best_per_metric.get(metric_name, []):
                    wins[method] += 1
            best_row_method = max(wins.items(), key=lambda item: item[1])[0] if wins else None

            for idx, method in enumerate(METHODS):
                if method not in model_records:
                    continue
                record = model_records[method]
                marker = " ← BEST" if method == best_row_method else ""
                model_name = MODEL_LABEL[model] if idx == 0 else ""
                print(
                    f"{model_name:<10} "
                    f"{METHOD_LABEL[method]:<12} "
                    f"{_format_validity(record):<12} "
                    f"{_format_hamming(record):<11} "
                    f"{_format_queries(record):>8} "
                    f"{_format_delta_nll(record):>8}"
                    f"{marker}"
                )


def build_latex(records: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]]) -> str:
    lines: List[str] = []
    newline = r"\\"
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    col_spec = "l|" + "|".join(["cccc" for _ in MODELS])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    for dataset in DATASETS:
        dataset_title = DATASET_LABEL[dataset]
        total_cols = 1 + 4 * len(MODELS)
        lines.append(f"\\multicolumn{{{total_cols}}}{{c}}{{\\textbf{{{dataset_title}}}}} {newline}")
        lines.append("\\hline")

        header_group = ["Method"]
        for model in MODELS:
            header_group.append(f"\\multicolumn{{4}}{{c}}{{{MODEL_LABEL[model]}}}")
        lines.append(" & ".join(header_group) + f" {newline}")

        sub = [""]
        for _ in MODELS:
            sub.extend([
                "Validity$\\uparrow$",
                "Hamming$\\downarrow$",
                "Queries$\\downarrow$",
                "$\\Delta$NLL$\\downarrow$",
            ])
        lines.append(" & ".join(sub) + f" {newline}")
        lines.append("\\hline")

        best_map_by_model = {model: _best_methods_for_model(records[dataset][model]) for model in MODELS}
        for method in METHODS:
            row = [METHOD_LABEL[method]]
            for model in MODELS:
                model_records = records[dataset][model]
                record = model_records.get(method)
                if record is None:
                    row.extend(["-"] * 4)
                    continue
                best_map = best_map_by_model[model]
                row.extend([
                    _latex_wrap(_format_validity(record).replace("±", "$\\pm$"), method in best_map.get("validity", [])),
                    _latex_wrap(_format_hamming(record).replace("±", "$\\pm$"), method in best_map.get("hamming", [])),
                    _latex_wrap(_format_queries(record), method in best_map.get("queries", [])),
                    _latex_wrap(_format_delta_nll(record), method in best_map.get("delta_nll", [])),
                ])
            lines.append(" & ".join(row) + f" {newline}")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\caption{Counterfactual explanation metrics across datasets and models. Bold indicates the best method per metric within each model.}")
    lines.append("\\label{tab:cf-main-results}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def write_csv(records: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "model",
            "method",
            "method_label",
            "seed_count",
            "validity_mean",
            "validity_std",
            "hamming_mean",
            "hamming_std",
            "queries_mean",
            "queries_std",
            "delta_nll",
            "delta_nll_std",
        ])

        for dataset in DATASETS:
            for model in MODELS:
                for method in METHODS:
                    record = records[dataset][model].get(method)
                    if record is None:
                        continue
                    writer.writerow([
                        dataset,
                        model,
                        method,
                        METHOD_LABEL[method],
                        int(record.get("seed_count") or 0),
                        f"{record['validity_rate_mean']:.6f}",
                        f"{record['validity_rate_std']:.6f}",
                        "" if record.get("hamming_mean_mean") is None else f"{record['hamming_mean_mean']:.6f}",
                        "" if record.get("hamming_mean_std") is None else f"{record['hamming_mean_std']:.6f}",
                        "" if record.get("queries_mean") is None else f"{record['queries_mean']:.6f}",
                        "" if record.get("queries_std") is None else f"{record['queries_std']:.6f}",
                        "" if record.get("delta_nll") is None else f"{record['delta_nll']:.6f}",
                        "" if record.get("delta_nll_std") is None else f"{record['delta_nll_std']:.6f}",
                    ])


def write_json(records: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables from aggregated explanation summaries")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--task", choices=["bring_in", "push_out"], default=None,
                        help="Aggregate the bring_in (*_targeted_r*_*) or push_out (*_pushout_top*) task; "
                             "None = legacy no-suffix files.")
    parser.add_argument("--no-recompute-delta-nll", action="store_true",
                        help="Skip the forward-pass recomputation of delta_nll (faster, uses summary values as-is)")
    args = parser.parse_args()

    records = build_records(
        results_dir=args.results_dir,
        task=args.task,
        recompute_delta_nll=not args.no_recompute_delta_nll,
    )
    print_console_tables(records)

    if args.task:
        suffix = "_" + {"bring_in": "bringin", "push_out": "pushout"}[args.task]
    else:
        suffix = ""
    tex_path = os.path.join(args.results_dir, f"paper_table{suffix}.tex")
    os.makedirs(args.results_dir, exist_ok=True)
    tex = build_latex(records)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    csv_path = os.path.join(args.results_dir, f"paper_table{suffix}.csv")
    write_csv(records, csv_path)

    json_path = os.path.join(args.results_dir, f"paper_table{suffix}.json")
    write_json(records, json_path)

    print(f"\n[OK] Wrote {tex_path}")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {json_path}")

    print("\nNotes:")
    print("- Validity: higher is better")
    print("- Hamming: lower is better")
    print("- Queries: lower is better")
    print("- ΔNLL: lower is better")


if __name__ == "__main__":
    main()
