"""Regenerate paper figures from per-combo summary JSONs.

Reads `results/summary_{model}_{dataset}_{method}_{task}_*.json` (one file
per method x combo x task) and produces the single figure the paper uses:

    figures/fig_trace_gece_greedy_triplet_combined.{pdf,png}

It is a 2x3 dumbbell / connected-scatter panel: the upper row is bring-in,
the lower row is push-out. In each row, the three panels show Validity,
Queries (log), and Hamming distance. Every row of 15 is one
(dataset, model) combination; the three dots per row are the three methods
(TRACE = beam, GECE = ga, Greedy).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator


MODELS = ["sasrec", "bert4rec", "gru4rec"]
MODEL_LABELS = {"sasrec": "SASRec", "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec"}
DATASETS = ["ml-100k", "ml-1m", "steam", "beauty", "sports"]
DATASET_LABELS = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "steam": "Steam",
                  "beauty": "Beauty", "sports": "Sports"}

METHODS = ["TRACE", "GECE", "Greedy"]
METHOD_KEY = {"TRACE": "beam", "GECE": "ga", "Greedy": "greedy"}
METHOD_COLOR = {"TRACE": "#d55e00", "GECE": "#0072b2", "Greedy": "#009e73"}

TASKS = {
    "bringin": {
        "title": "Bring-in",
        "query_xlim": (300, 6000),
        "query_ticks": [300, 1000, 3000, 6000],
        "hamming_xlim": (1.5, 4.0),
    },
    "pushout": {
        "title": "Push-out",
        "query_xlim": (60, 3000),
        "query_ticks": [100, 300, 1000, 3000],
        "validity_xlim": (0.9, 1.01),
        "hamming_xlim": (0.9, 3.5),
    },
}


def _summary_path(results_dir: Path, task_key: str, method: str, model: str, dataset: str) -> Path:
    task_suffix = "targeted_r2_50" if task_key == "bringin" else "pushout_top50"
    return results_dir / f"summary_{model}_{dataset}_{METHOD_KEY[method]}_{task_suffix}.json"


def _metric(rec: Dict[str, Any], mean_key: str, plain_key: str) -> float:
    if mean_key in rec:
        return float(rec[mean_key])
    return float(rec[plain_key])


def _load_point(results_dir: Path, task_key: str, method: str, model: str, dataset: str) -> Dict[str, float]:
    path = _summary_path(results_dir, task_key, method, model, dataset)
    with open(path) as f:
        rec = json.load(f)[METHOD_KEY[method]]
    return {
        "validity": _metric(rec, "validity_mean", "validity"),
        "queries": _metric(rec, "queries_mean", "queries"),
        "hamming": _metric(rec, "hamming_mean_mean", "hamming_mean"),
    }


def _collect_rows(results_dir: Path, task_key: str) -> List[Dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            pts = {m: _load_point(results_dir, task_key, m, model, dataset) for m in METHODS}
            rows.append({
                "label": f"{MODEL_LABELS[model]} / {DATASET_LABELS[dataset]}",
                "validity": {m: pts[m]["validity"] for m in METHODS},
                "queries": {m: pts[m]["queries"] for m in METHODS},
                "hamming": {m: pts[m]["hamming"] for m in METHODS},
            })
    return rows


def _draw_panel(ax_v, ax_q, ax_h, rows: List[Dict[str, Any]], task: Dict[str, Any]) -> None:
    y_positions = list(range(len(rows)))
    labels = [r["label"] for r in rows]

    for y, row in zip(y_positions, rows):
        vv = row["validity"]; qq = row["queries"]; hh = row["hamming"]
        ordered_v = sorted(vv.items(), key=lambda x: x[1])
        ordered_q = sorted(qq.items(), key=lambda x: x[1])
        ordered_h = sorted(hh.items(), key=lambda x: x[1])
        ax_v.plot([v for _, v in ordered_v], [y] * len(ordered_v), color="0.8", linewidth=1.1, zorder=1)
        ax_q.plot([v for _, v in ordered_q], [y] * len(ordered_q), color="0.8", linewidth=1.1, zorder=1)
        ax_h.plot([v for _, v in ordered_h], [y] * len(ordered_h), color="0.8", linewidth=1.1, zorder=1)
        for method in METHODS:
            size = 54 if method == "TRACE" else 44
            ax_v.scatter(vv[method], y, s=size, color=METHOD_COLOR[method], edgecolor="black", linewidth=0.5, zorder=2)
            ax_q.scatter(qq[method], y, s=size, color=METHOD_COLOR[method], edgecolor="black", linewidth=0.5, zorder=2)
            ax_h.scatter(hh[method], y, s=size, color=METHOD_COLOR[method], edgecolor="black", linewidth=0.5, zorder=2)

    for ax in (ax_v, ax_q, ax_h):
        ax.set_yticks(y_positions)
        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    ax_v.set_yticklabels(labels)
    ax_q.set_yticklabels([])
    ax_h.set_yticklabels([])

    ax_v.set_xlim(*task.get("validity_xlim", (0.0, 1.05)))
    ax_q.set_xscale("log")
    ax_q.set_xlim(*task["query_xlim"])
    ax_q.xaxis.set_major_locator(FixedLocator(task["query_ticks"]))
    ax_q.xaxis.set_major_formatter(FixedFormatter([str(v) for v in task["query_ticks"]]))
    ax_h.set_xlim(*task["hamming_xlim"])

    ax_v.set_title(f"{task['title']} / Validity")
    ax_q.set_title(f"{task['title']} / Queries")
    ax_h.set_title(f"{task['title']} / Hamming")


def fig_triplet_combined(results_dir: Path, outdir: Path) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 9

    fig, axes = plt.subplots(2, 3, figsize=(18.2, 10.2),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})

    for row_idx, task_key in enumerate(("bringin", "pushout")):
        rows = _collect_rows(results_dir, task_key)
        _draw_panel(axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2], rows, TASKS[task_key])

    axes[1, 0].set_xlabel("Validity")
    axes[1, 1].set_xlabel("Queries (log scale)")
    axes[1, 2].set_xlabel("Hamming Distance")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLOR[m],
                   markeredgecolor="black", markersize=7, label=m)
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "fig_trace_gece_greedy_triplet_combined.pdf"
    png = outdir / "fig_trace_gece_greedy_triplet_combined.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote {pdf} and {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper figures from pipeline artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    fig_triplet_combined(results_dir, figures_dir)
    print("Done.")


if __name__ == "__main__":
    main()
