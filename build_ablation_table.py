"""Aggregate ablation results produced by scripts/reproduce_ablation.sh.

reproduce_ablation.sh runs `run_explain.py` with `evaluation.run_tag=ablation-<variant>`
for each of the four cumulative variants {naive, beam, influence, full} on three
combos (SASRec/ML-100K, BERT4Rec/ML-1M, GRU4Rec/Steam), bring-in and push-out each.

Seed files therefore carry the suffix `_<target_suffix>_ablation-<variant>_seed<N>.json`.
This script collects them and produces `results/ablation_<task>.json` matching the
layout consumed by `build_paper_numbers.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

from aggregate_summary import aggregate_model_dataset


ABLATION_COMBOS: List[Tuple[str, str]] = [
    ("sasrec", "ml-100k"),
    ("bert4rec", "ml-1m"),
    ("gru4rec", "steam"),
]

VARIANTS: List[str] = ["naive", "beam", "influence", "full"]

VARIANT_LABEL: Dict[str, str] = {
    "naive": "Naive",
    "beam": "+Beam",
    "influence": "+Influence",
    "full": "Full TRACE",
}

COMBO_LABEL: Dict[Tuple[str, str], str] = {
    ("sasrec", "ml-100k"): "SASRec / ML-100K",
    ("bert4rec", "ml-1m"): "BERT4Rec / ML-1M",
    ("gru4rec", "steam"): "GRU4Rec / Steam",
}


def _pair(metric_map: Dict[str, float] | None, mean_key: str, std_key: str, scale: float = 1.0) -> List[float]:
    if not metric_map:
        return [float("nan"), 0.0]
    m = metric_map.get(mean_key)
    s = metric_map.get(std_key)
    m = float(m) * scale if isinstance(m, (int, float)) and math.isfinite(float(m)) else float("nan")
    s = float(s) * scale if isinstance(s, (int, float)) and math.isfinite(float(s)) else 0.0
    return [m, s]


def aggregate_variant(
    results_dir: str, model: str, dataset: str, variant: str, task: str,
) -> Dict[str, float] | None:
    run_tag = f"ablation-{variant}"
    summary = aggregate_model_dataset(
        model=model, dataset=dataset, results_dir=results_dir,
        task=task, run_tag=run_tag,
        methods=["beam"], write_summary=False,
    )
    beam = summary.get("beam")
    if not isinstance(beam, dict) or "validity_rate_mean" not in beam:
        return None
    return beam


def build_bringin_validity(results_dir: str) -> Dict:
    combos_labels = [COMBO_LABEL[c] for c in ABLATION_COMBOS]
    rows: List[Dict] = []
    for variant in VARIANTS:
        values: List[List[float]] = []
        for combo in ABLATION_COMBOS:
            model, dataset = combo
            metrics = aggregate_variant(results_dir, model, dataset, variant, "bring_in")
            values.append(_pair(metrics, "validity_rate_mean", "validity_rate_std", scale=100.0))
        rows.append({"variant": VARIANT_LABEL[variant], "values": values})
    return {
        "description": "Table 4: cumulative ablation of TRACE components on bring-in validity (%). Higher is better.",
        "combos": combos_labels,
        "rows": rows,
    }


def build_pushout_efficiency(results_dir: str) -> Dict:
    combos_labels = [COMBO_LABEL[c] for c in ABLATION_COMBOS]
    rows: List[Dict] = []
    for variant in VARIANTS:
        q_values: List[List[float]] = []
        dnll_values: List[List[float]] = []
        for combo in ABLATION_COMBOS:
            model, dataset = combo
            metrics = aggregate_variant(results_dir, model, dataset, variant, "push_out")
            q_values.append(_pair(metrics, "queries_mean", "queries_std"))
            dnll_values.append(_pair(metrics, "delta_nll_mean", "delta_nll_std"))
        rows.append({"variant": VARIANT_LABEL[variant], "Q": q_values, "delta_nll": dnll_values})
    return {
        "description": "Table 5: push-out ablation on Queries (Q) and Delta-NLL. Lower is better.",
        "combos": combos_labels,
        "rows": rows,
        "notes": "Per-variant runs use run_tag=ablation-<variant>. Variants are cumulative: naive -> +beam -> +influence -> full.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    bringin = build_bringin_validity(args.results_dir)
    pushout = build_pushout_efficiency(args.results_dir)

    os.makedirs(args.results_dir, exist_ok=True)
    for name, payload in (("ablation_bringin.json", bringin), ("ablation_pushout.json", pushout)):
        path = os.path.join(args.results_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Wrote {path}")


if __name__ == "__main__":
    main()
