import argparse
import csv
import json
import math
import os
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from aggregate_summary import METHOD_LABEL, find_method_seed_files
from data.datasets import load_dataset
from evaluation.metrics import compute_sequence_nll_batch
from models.trainer import load_trained_model


MODELS = ["sasrec", "bert4rec", "gru4rec"]
DATASETS = ["ml-100k", "ml-1m", "steam", "beauty", "sports"]
METHODS = ["random", "greedy", "ga", "beam", "gradient"]
SEEDS = [42, 43, 44]

MODEL_LABEL = {
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "gru4rec": "GRU4Rec",
}

DATASET_LABEL = {
    "ml-100k": "ML-100K",
    "ml-1m": "ML-1M",
    "steam": "Steam",
    "beauty": "Beauty",
    "sports": "Sports",
}


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def format_p(p: float) -> str:
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.4f}"


def format_signed(x: float, dec: int = 3) -> str:
    fmt = f"{{:+.{dec}f}}"
    return fmt.format(x)


def chi2_sf_1df(x: float) -> float:
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_test(a_valid: np.ndarray, b_valid: np.ndarray) -> Dict:
    a = int(np.sum((a_valid == 1) & (b_valid == 1)))
    b = int(np.sum((a_valid == 1) & (b_valid == 0)))
    c = int(np.sum((a_valid == 0) & (b_valid == 1)))
    d = int(np.sum((a_valid == 0) & (b_valid == 0)))

    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
        test_variant = "mcnemar_exact"
        stat = 0.0
    elif n_disc < 25:
        k = min(b, c)
        prob = 0.0
        for i in range(k + 1):
            prob += math.comb(n_disc, i) * (0.5 ** n_disc)
        p_value = min(1.0, 2.0 * prob)
        test_variant = "mcnemar_exact"
        stat = float("nan")
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / n_disc
        p_value = chi2_sf_1df(stat)
        test_variant = "mcnemar_chi2_cc"

    odds_ratio = (b + 0.5) / (c + 0.5)
    return {
        "test": "mcnemar",
        "variant": test_variant,
        "n_pairs": int(len(a_valid)),
        "both_valid": a,
        "a_only": b,
        "b_only": c,
        "both_invalid": d,
        "statistic": float(stat) if not np.isnan(stat) else None,
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
    }


def paired_bootstrap_mean_diff(diffs: np.ndarray, B: int = 10000, seed: int = 0) -> Dict:
    n = len(diffs)
    if n == 0:
        return {
            "test": "paired_bootstrap",
            "n_pairs": 0,
            "skipped": True,
            "reason": "no paired valid observations",
        }

    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=np.float64)

    chunk = 1000
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        boot[s:e] = diffs[idx].mean(axis=1)

    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    if obs > 0:
        p = 2.0 * float(np.mean(boot <= 0))
    elif obs < 0:
        p = 2.0 * float(np.mean(boot >= 0))
    else:
        p = 1.0
    p = min(1.0, p)

    return {
        "test": "paired_bootstrap",
        "n_pairs": n,
        "mean_diff": obs,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p,
        "bootstrap_B": B,
    }


def load_saved_per_user(path: str) -> Optional[Dict[int, Dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    per_user = payload.get("per_user", [])
    return {int(row["user_id"]): row for row in per_user if "user_id" in row}


def select_input_sequences(user_seqs: Dict[int, List[int]], max_len: int, seed: int, n_users: int) -> Dict[int, List[int]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_uids = sorted(user_seqs.keys())
    if len(all_uids) > n_users:
        test_uids = random.sample(all_uids, n_users)
    else:
        test_uids = all_uids

    out = {}
    for uid in test_uids:
        seq = user_seqs[uid]
        if len(seq) < 3:
            continue
        input_seq = seq[:-1][-max_len:]
        pad_len = max_len - len(input_seq)
        out[uid] = [0] * pad_len + list(input_seq)
    return out


def extract_saved_cf_sequences(method: str, per_user_by_uid: Dict[int, Dict]) -> Dict[int, Optional[List[int]]]:
    # random/greedy don't persist cf_seqs, and their ΔNLL is not reported in
    # the paper — return None for every uid so compute_user_metrics leaves
    # delta_nll unset for those methods.
    seq_by_uid: Dict[int, Optional[List[int]]] = {}
    for uid, row in per_user_by_uid.items():
        valid = bool(row.get("valid", False))
        if not valid or method not in ("ga", "gradient", "beam"):
            seq_by_uid[uid] = None
            continue

        top_k = row.get("top_k_cfs", [])
        cf_entry = None
        if top_k and isinstance(top_k, list):
            for entry in top_k:
                if isinstance(entry, dict) and entry.get("valid", False) and "seq" in entry:
                    cf_entry = entry
                    break
            if cf_entry is None and "seq" in top_k[0]:
                cf_entry = top_k[0]
        if cf_entry is not None:
            seq_by_uid[uid] = cf_entry["seq"]
        else:
            seq_by_uid[uid] = None
            warn(f"{method}: missing top_k_cfs seq for uid={uid}; treating as invalid for continuous metrics")
    return seq_by_uid


def compute_user_metrics(
    per_user_by_uid: Dict[int, Dict],
    original_seq_by_uid: Dict[int, List[int]],
    cf_seq_by_uid: Dict[int, Optional[List[int]]],
    model,
    device: str,
    max_len: int,
) -> Dict[int, Dict[str, Optional[float]]]:
    out = {}
    valid_uids = []
    orig_batch = []
    cf_batch = []

    for uid, row in per_user_by_uid.items():
        valid = bool(row.get("valid", False))
        out_row = {
            "validity": 1 if valid else 0,
            "hamming": None,
            "queries": None,
            "delta_nll": None,
        }

        if uid not in original_seq_by_uid:
            out[uid] = out_row
            continue

        if valid:
            out_row["hamming"] = float(row.get("edit_distance", np.nan))
            out_row["queries"] = float(row.get("queries_used", np.nan))

            cf_seq = cf_seq_by_uid.get(uid)
            if cf_seq is not None:
                valid_uids.append(uid)
                orig_batch.append(original_seq_by_uid[uid])
                cf_batch.append(list(cf_seq))

        out[uid] = out_row

    if not valid_uids:
        return out

    model.eval()
    with torch.no_grad():
        orig_nlls = np.array(compute_sequence_nll_batch(model, orig_batch, max_len=max_len, device=device), dtype=np.float64)
        cf_nlls = np.array(compute_sequence_nll_batch(model, cf_batch, max_len=max_len, device=device), dtype=np.float64)

    deltas = cf_nlls - orig_nlls
    for i, uid in enumerate(valid_uids):
        delta = float(deltas[i])
        if np.isfinite(delta):
            out[uid]["delta_nll"] = delta

    return out


def collect_combo_data(
    cfg, model_name: str, dataset: str, *, task: Optional[str] = None,
) -> Optional[Dict[str, Dict[Tuple[int, int], Dict]]]:
    device = cfg.training.device
    ckpt = os.path.join(cfg.training.checkpoint_dir, f"{model_name}_{dataset}.pt")
    if not os.path.exists(ckpt):
        warn(f"Missing checkpoint: {ckpt}. Skipping {model_name}/{dataset}.")
        return None

    user_seqs, _ = load_dataset(cfg.data.data_dir, dataset)
    model = load_trained_model(ckpt, device=device)
    model.eval()
    model_max_len = int(getattr(model, "max_len", cfg.data.max_len))

    pooled = {m: {} for m in METHODS}

    seeds_used: List[int] = []
    for seed in SEEDS:
        seen_any = False
        for method in METHODS:
            files = find_method_seed_files(
                cfg.evaluation.results_dir, model_name, dataset, method, task=task,
            )
            if any(s == seed for s, _ in files):
                seen_any = True
                break
        if seen_any:
            seeds_used.append(seed)

    if not seeds_used:
        warn(f"No seed files found for {model_name}/{dataset} task={task}")
        return None

    for seed in seeds_used:
        print(f"  Processing seed={seed}")
        original_seq_by_uid = select_input_sequences(user_seqs, model_max_len, seed, cfg.evaluation.n_users)
        selected_uids = set(original_seq_by_uid.keys())

        per_user_saved: Dict[str, Dict[int, Dict]] = {}
        for method in METHODS:
            files = find_method_seed_files(
                cfg.evaluation.results_dir, model_name, dataset, method, task=task,
            )
            path = next((p for s, p in files if s == seed), None)
            if path is None:
                warn(f"Missing results file: {model_name}/{dataset}/{method}/seed{seed} (task={task})")
                continue
            data = load_saved_per_user(path)
            if data is None:
                warn(f"Failed to parse: {path}")
                continue
            per_user_saved[method] = data

            saved_uids = set(data.keys())
            if saved_uids != selected_uids:
                warn(
                    f"UID mismatch ({model_name}/{dataset}/{method}/seed{seed}): "
                    f"saved={len(saved_uids)} selected={len(selected_uids)}"
                )

        for method, saved_map in per_user_saved.items():
            cf_seq_by_uid = extract_saved_cf_sequences(method, saved_map)

            per_user_metrics = compute_user_metrics(
                per_user_by_uid=saved_map,
                original_seq_by_uid=original_seq_by_uid,
                cf_seq_by_uid=cf_seq_by_uid,
                model=model,
                device=device,
                max_len=model_max_len,
            )

            for uid, metrics in per_user_metrics.items():
                pooled[method][(seed, int(uid))] = metrics

    return pooled


def run_pairwise_tests(pooled: Dict[str, Dict[Tuple[int, int], Dict]]) -> Dict[str, Dict]:
    out = {}
    for a, b in combinations(METHODS, 2):
        comp_key = f"{a}_vs_{b}"
        a_map = pooled.get(a, {})
        b_map = pooled.get(b, {})
        matched = sorted(set(a_map.keys()) & set(b_map.keys()))

        if not matched:
            continue  # one side missing; caller can see via seed availability

        a_valid = np.array([a_map[k]["validity"] for k in matched], dtype=np.int32)
        b_valid = np.array([b_map[k]["validity"] for k in matched], dtype=np.int32)

        comp = {
            "validity": mcnemar_test(a_valid, b_valid),
        }

        for metric in ("hamming", "queries", "delta_nll"):
            diffs = []
            for k in matched:
                va = a_map[k].get(metric)
                vb = b_map[k].get(metric)
                if va is None or vb is None:
                    continue
                if not np.isfinite(va) or not np.isfinite(vb):
                    continue
                diffs.append(float(va - vb))

            comp[metric] = paired_bootstrap_mean_diff(np.array(diffs, dtype=np.float64), B=10000, seed=0)

        out[comp_key] = comp

    return out


def print_combo_table(model_name: str, dataset: str, stats: Dict[str, Dict]) -> None:
    print(f"\n=== {MODEL_LABEL.get(model_name, model_name)} / {DATASET_LABEL.get(dataset, dataset)} ===")
    for comp_key, res in stats.items():
        a, b = comp_key.split("_vs_")
        print(f"Comparison: {METHOD_LABEL.get(a, a)} vs {METHOD_LABEL.get(b, b)}")

        v = res["validity"]
        print(f"  Validity:  McNemar {format_p(v['p_value'])}, OR={v['odds_ratio']:.3f}")

        for metric, label, dec in [
            ("hamming", "Hamming", 3),
            ("queries", "Queries", 1),
            ("delta_nll", "ΔNLL", 4),
        ]:
            m = res[metric]
            if m.get("skipped", False):
                print(f"  {label:<9}: skipped (no paired valid users)")
                continue
            print(
                f"  {label:<9}: diff={format_signed(m['mean_diff'], dec)} "
                f"95%CI[{format_signed(m['ci_lower'], dec)},{format_signed(m['ci_upper'], dec)}] "
                f"{format_p(m['p_value'])}"
            )


def flatten_for_csv(all_stats: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = []
    for dataset, by_model in all_stats.items():
        for model_name, by_comp in by_model.items():
            for comp_key, metrics in by_comp.items():
                a, b = comp_key.split("_vs_")

                v = metrics["validity"]
                rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "comparison": comp_key,
                    "method_a": a,
                    "method_b": b,
                    "metric": "validity",
                    "test": v.get("test"),
                    "n_pairs": v.get("n_pairs"),
                    "mean_diff": "",
                    "ci_lower": "",
                    "ci_upper": "",
                    "p_value": v.get("p_value"),
                    "odds_ratio": v.get("odds_ratio"),
                    "both_valid": v.get("both_valid"),
                    "a_only": v.get("a_only"),
                    "b_only": v.get("b_only"),
                    "both_invalid": v.get("both_invalid"),
                })

                for metric in ("hamming", "queries", "delta_nll"):
                    m = metrics[metric]
                    rows.append({
                        "dataset": dataset,
                        "model": model_name,
                        "comparison": comp_key,
                        "method_a": a,
                        "method_b": b,
                        "metric": metric,
                        "test": m.get("test"),
                        "n_pairs": m.get("n_pairs"),
                        "mean_diff": m.get("mean_diff", ""),
                        "ci_lower": m.get("ci_lower", ""),
                        "ci_upper": m.get("ci_upper", ""),
                        "p_value": m.get("p_value", ""),
                        "odds_ratio": "",
                        "both_valid": "",
                        "a_only": "",
                        "b_only": "",
                        "both_invalid": "",
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical tests for CF explanation methods")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--task", choices=["bring_in", "push_out"], default=None,
                        help="Task suffix to filter seed files (bring_in / push_out). "
                             "None = legacy no-suffix files.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    override_keys = set(x.split("=")[0] for x in args.overrides)
    models = [cfg.model.name] if "model.name" in override_keys else MODELS
    datasets = [cfg.data.dataset] if "data.dataset" in override_keys else DATASETS

    results_json: Dict[str, Dict[str, Dict]] = {}

    for dataset in datasets:
        if dataset not in results_json:
            results_json[dataset] = {}

        for model_name in models:
            print(f"\nProcessing combo: model={model_name}, dataset={dataset}, task={args.task}")
            combo_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            combo_cfg.model.name = model_name
            combo_cfg.data.dataset = dataset

            pooled = collect_combo_data(combo_cfg, model_name=model_name, dataset=dataset, task=args.task)
            if pooled is None:
                continue

            stats = run_pairwise_tests(pooled)
            results_json[dataset][model_name] = stats
            print_combo_table(model_name, dataset, stats)

    os.makedirs(cfg.evaluation.results_dir, exist_ok=True)
    if args.task:
        suffix = "_" + {"bring_in": "bringin", "push_out": "pushout"}[args.task]
    else:
        suffix = ""
    json_path = os.path.join(cfg.evaluation.results_dir, f"stat_tests{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    csv_path = os.path.join(cfg.evaluation.results_dir, f"stat_tests{suffix}.csv")
    rows = flatten_for_csv(results_json)
    fieldnames = [
        "dataset", "model", "comparison", "method_a", "method_b", "metric", "test", "n_pairs",
        "mean_diff", "ci_lower", "ci_upper", "p_value", "odds_ratio",
        "both_valid", "a_only", "b_only", "both_invalid",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
