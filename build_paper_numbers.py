"""Assemble `results/paper_numbers.json` from pipeline artifacts.

Inputs (all under `results/`):
  - backbone_eval.json           (eval_backbones.py output)
  - paper_table_bringin.json     (build_paper_table.py --task bring_in)
  - paper_table_pushout.json     (build_paper_table.py --task push_out)
  - stat_tests_bringin.json      (run_stat_tests.py --task bring_in)
  - stat_tests_pushout.json      (run_stat_tests.py --task push_out)
  - ablation_bringin.json        (build_ablation_table.py)
  - ablation_pushout.json        (build_ablation_table.py)

Outputs:
  - results/paper_numbers.json   (consumed by reproduce.ipynb)

Static sections (hyperparameters, datasets, case_studies) are read from an
existing `paper_numbers.json` template when present; otherwise sensible
defaults are written and the user is warned. Case studies are hand-picked
qualitative examples and cannot be regenerated programmatically.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from aggregate_summary import DATASETS as CANONICAL_DATASETS
from aggregate_summary import MODELS as CANONICAL_MODELS


MODEL_LABEL = {"sasrec": "SASRec", "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec"}
DATASET_LABEL = {
    "ml-100k": "ML-100K",
    "ml-1m": "ML-1M",
    "steam": "Steam",
    "beauty": "Beauty",
    "sports": "Sports",
}

METHODS_FOR_DELTA_TABLE = ["ga", "beam"]
METHOD_DISPLAY = {"ga": "GECE", "beam": "TRACE"}


STATIC_DEFAULTS: Dict[str, Any] = {
    "hyperparameters": {
        "query_budget_Q": 10000,
        "edit_budget_K_max": 5,
        "n_eval_users": 500,
        "seeds": [42, 43, 44],
        "beam_width_B": 5,
        "k_context": 6,
        "k_target_nn": 3,
        "k_source_nn": 3,
        "shortlist_size": 12,
        "rank_range_bring_in": [2, 50],
        "rank_threshold_push_out": 50,
    },
    "datasets": {
        "ml-100k": {"users": 943, "items": 1349},
        "ml-1m":   {"users": 6040, "items": 3416},
        "steam":   {"users": 334536, "items": 13046},
        "beauty":  {"users": 22363, "items": 12101},
        "sports":  {"users": 35598, "items": 18357},
    },
    "case_studies": [],
}


def _read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_backbone(path: str) -> Optional[Dict]:
    payload = _read_json(path)
    if not payload:
        return None
    models = payload.get("models", {})
    hr: Dict[str, List[Optional[float]]] = {}
    ndcg: Dict[str, List[Optional[float]]] = {}
    for m_key in CANONICAL_MODELS:
        row_hr, row_ndcg = [], []
        m_rec = models.get(m_key, {})
        for d in CANONICAL_DATASETS:
            row_hr.append(m_rec.get(d, {}).get("hr10"))
            row_ndcg.append(m_rec.get(d, {}).get("ndcg10"))
        hr[MODEL_LABEL[m_key]] = row_hr
        ndcg[MODEL_LABEL[m_key]] = row_ndcg
    return {
        "description": "SRS backbone recommendation performance (HR@10, NDCG@10) over five datasets.",
        "datasets": [DATASET_LABEL[d] for d in CANONICAL_DATASETS],
        "HR@10": hr,
        "NDCG@10": ndcg,
    }


def _extract_delta_pair(record: Dict[str, Any] | None) -> List[float]:
    if record is None:
        return [float("nan"), 0.0]
    mean = record.get("delta_nll")
    std = record.get("delta_nll_std")
    mean = float(mean) if isinstance(mean, (int, float)) else float("nan")
    std = float(std) if isinstance(std, (int, float)) else 0.0
    return [mean, std]


def build_delta_nll_main(paper_bringin_path: str, paper_pushout_path: str) -> Optional[Dict]:
    bringin = _read_json(paper_bringin_path)
    pushout = _read_json(paper_pushout_path)
    if bringin is None or pushout is None:
        return None

    datasets_display = [DATASET_LABEL[d] for d in CANONICAL_DATASETS]
    rows: List[Dict[str, Any]] = []
    for model in CANONICAL_MODELS:
        for method in METHODS_FOR_DELTA_TABLE:
            row: Dict[str, Any] = {
                "model": MODEL_LABEL[model],
                "method": METHOD_DISPLAY[method],
            }
            for d in CANONICAL_DATASETS:
                bringin_rec = bringin.get(d, {}).get(model, {}).get(method)
                pushout_rec = pushout.get(d, {}).get(model, {}).get(method)
                row[DATASET_LABEL[d]] = [
                    _extract_delta_pair(bringin_rec),
                    _extract_delta_pair(pushout_rec),
                ]
            rows.append(row)

    return {
        "description": ("Table 3: Delta-NLL (mean +/- std) over valid users. "
                        "In = bring-in, Out = push-out. Lower is better; "
                        "negative = counterfactual more natural than original."),
        "datasets": datasets_display,
        "rows": rows,
    }


def _iter_model_dataset(stat_json: Dict[str, Any]) -> List[tuple[str, str, Dict[str, Any]]]:
    cells = []
    for dataset, by_model in stat_json.items():
        if not isinstance(by_model, dict):
            continue
        for model, by_comp in by_model.items():
            if not isinstance(by_comp, dict):
                continue
            cells.append((dataset, model, by_comp))
    return cells


def _verdict(n_sig: int, n_total: int, direction: Optional[str] = None) -> str:
    if n_total == 0:
        return "0/0"
    tag = f"{n_sig}/{n_total} sig"
    if direction:
        tag += f" {direction}"
    return tag


def _count_pair(
    stat_json: Dict[str, Any], comp_key: str, metric: str,
    direction: Optional[str] = None, alpha: float = 0.05,
) -> tuple[int, int]:
    n_sig = 0
    n_total = 0
    for _, _, by_comp in _iter_model_dataset(stat_json):
        entry = by_comp.get(comp_key)
        if entry is None:
            continue
        metric_rec = entry.get(metric)
        if not isinstance(metric_rec, dict) or metric_rec.get("skipped"):
            continue
        p = metric_rec.get("p_value")
        if not isinstance(p, (int, float)):
            continue
        n_total += 1
        if p >= alpha:
            continue
        if direction is None:
            n_sig += 1
            continue
        mean_diff = metric_rec.get("mean_diff")
        if not isinstance(mean_diff, (int, float)):
            continue
        if direction == "lower" and mean_diff < 0:
            n_sig += 1
        elif direction == "higher" and mean_diff > 0:
            n_sig += 1
    return n_sig, n_total


def build_significance_summary(stat_bringin_path: str, stat_pushout_path: str) -> Optional[Dict]:
    bringin = _read_json(stat_bringin_path)
    pushout = _read_json(stat_pushout_path)
    if bringin is None and pushout is None:
        return None

    summary: Dict[str, Any] = {}

    if bringin is not None:
        bi_trace_vs_gece: Dict[str, str] = {}
        # McNemar is directionless; report sig count only.
        s, t = _count_pair(bringin, "ga_vs_beam", "validity")
        bi_trace_vs_gece["validity"] = _verdict(s, t)
        s, t = _count_pair(bringin, "ga_vs_beam", "queries")
        bi_trace_vs_gece["queries"] = _verdict(s, t)
        s, t = _count_pair(bringin, "ga_vs_beam", "hamming")
        bi_trace_vs_gece["hamming"] = _verdict(s, t)
        summary["bring_in_trace_vs_gece"] = bi_trace_vs_gece

        bi_trace_vs_greedy: Dict[str, str] = {}
        s, t = _count_pair(bringin, "greedy_vs_beam", "validity")
        bi_trace_vs_greedy["validity"] = _verdict(s, t)
        s, t = _count_pair(bringin, "greedy_vs_beam", "queries")
        bi_trace_vs_greedy["queries"] = _verdict(s, t)
        s, t = _count_pair(bringin, "greedy_vs_beam", "hamming")
        bi_trace_vs_greedy["hamming"] = _verdict(s, t)
        summary["bring_in_trace_vs_greedy"] = bi_trace_vs_greedy

    if pushout is not None:
        po_trace_vs_gece: Dict[str, str] = {}
        s, t = _count_pair(pushout, "ga_vs_beam", "validity")
        po_trace_vs_gece["validity"] = _verdict(s, t) if t else "ceiling"
        s, t = _count_pair(pushout, "ga_vs_beam", "queries", direction="higher")  # ga > beam i.e. beam lower
        po_trace_vs_gece["queries"] = _verdict(s, t, direction="lower")
        s, t = _count_pair(pushout, "ga_vs_beam", "hamming", direction="higher")
        po_trace_vs_gece["hamming"] = _verdict(s, t, direction="lower")
        summary["push_out_trace_vs_gece"] = po_trace_vs_gece

        po_trace_vs_greedy: Dict[str, str] = {}
        s, t = _count_pair(pushout, "greedy_vs_beam", "queries", direction="higher")
        po_trace_vs_greedy["queries"] = _verdict(s, t, direction="lower")
        s, t = _count_pair(pushout, "greedy_vs_beam", "hamming", direction="higher")
        po_trace_vs_greedy["hamming"] = _verdict(s, t, direction="lower")
        summary["push_out_trace_vs_greedy"] = po_trace_vs_greedy

    # Combined ΔNLL sig count across bring-in + push-out
    sig_total = 0
    n_total = 0
    for src in (bringin, pushout):
        if src is None:
            continue
        s, t = _count_pair(src, "ga_vs_beam", "delta_nll", direction="higher")  # ga > beam → beam lower
        sig_total += s
        n_total += t
    if n_total:
        summary["delta_nll_trace_vs_gece"] = (
            f"{sig_total}/{n_total} sig (matched valid users, paired bootstrap, p<0.05)"
        )

    return summary


def assemble(
    results_dir: str,
    template_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Template holds the previously-assembled paper_numbers.json; sections
    # whose upstream artifacts are missing fall back to the template so a
    # partial pipeline run (e.g. `make paper_numbers` alone) never drops
    # keys that `reproduce.ipynb` depends on.
    template: Dict[str, Any] = {}
    if template_path and os.path.exists(template_path):
        template_raw = _read_json(template_path)
        if template_raw is not None:
            template = template_raw

    paper_numbers: Dict[str, Any] = {}

    def _assign(key: str, fresh: Any, artifact_label: str) -> None:
        if fresh is not None:
            paper_numbers[key] = fresh
        elif key in template:
            paper_numbers[key] = template[key]
            print(f"[INFO] '{key}': {artifact_label} missing; kept template value")
        else:
            print(f"[WARN] '{key}': {artifact_label} missing and not in template; section omitted")

    _assign(
        "backbone",
        build_backbone(os.path.join(results_dir, "backbone_eval.json")),
        "backbone_eval.json",
    )
    _assign(
        "delta_nll_main",
        build_delta_nll_main(
            os.path.join(results_dir, "paper_table_bringin.json"),
            os.path.join(results_dir, "paper_table_pushout.json"),
        ),
        "paper_table_bringin.json/paper_table_pushout.json",
    )
    _assign(
        "ablation_bringin_validity",
        _read_json(os.path.join(results_dir, "ablation_bringin.json")),
        "ablation_bringin.json",
    )
    _assign(
        "ablation_pushout_efficiency",
        _read_json(os.path.join(results_dir, "ablation_pushout.json")),
        "ablation_pushout.json",
    )
    _assign(
        "significance_summary",
        build_significance_summary(
            os.path.join(results_dir, "stat_tests_bringin.json"),
            os.path.join(results_dir, "stat_tests_pushout.json"),
        ),
        "stat_tests_bringin.json/stat_tests_pushout.json",
    )

    for key in ("hyperparameters", "datasets", "case_studies"):
        if key in template:
            paper_numbers[key] = template[key]
        else:
            paper_numbers[key] = STATIC_DEFAULTS[key]
            if key == "case_studies" and not STATIC_DEFAULTS[key]:
                print(f"[INFO] 'case_studies' is empty (hand-picked; provide via --template to preserve)")

    return paper_numbers


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble paper_numbers.json from pipeline outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--template",
                        help="Path to existing paper_numbers.json to source static sections "
                             "(hyperparameters, datasets, case_studies). Defaults to "
                             "results/paper_numbers.json if present.")
    parser.add_argument("--output", default=None, help="Output path (default: <results-dir>/paper_numbers.json)")
    args = parser.parse_args()

    template_path = args.template
    if template_path is None:
        default_template = os.path.join(args.results_dir, "paper_numbers.json")
        template_path = default_template if os.path.exists(default_template) else None

    payload = assemble(args.results_dir, template_path=template_path)

    output_path = args.output or os.path.join(args.results_dir, "paper_numbers.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote {output_path}")
    print(f"Sections: {list(payload.keys())}")


if __name__ == "__main__":
    main()
