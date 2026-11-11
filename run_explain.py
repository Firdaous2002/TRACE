"""
Generate counterfactual explanations.

Usage:
    python run_explain.py                                    # default config (Grad-Beam on SASRec/ML-1M)
    python run_explain.py explainer.name=ga                  # use GA explainer
    python run_explain.py model.name=bert4rec data.dataset=beauty  # different model/dataset
    python run_explain.py --all-explainers                   # run all paper explainers
"""

import argparse
import os
from typing import Dict, Tuple

from evaluation.evaluator import prepare_explanation_inputs
from evaluation.pipeline import run_and_score_explainer
from utils.experiment_utils import (
    clear_device_cache,
    clone_config,
    ensure_dir,
    get_rank_constraint_spec,
    load_config,
    load_experiment_data,
    load_experiment_model,
    numeric_stats,
    resolve_path,
    run_tag_suffix,
    save_json,
    target_suffix,
)
from explainers.factory import build_explainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config', 'default.yaml')


def run_single_explainer(
    cfg, model, user_seqs, info, popular_items, category_map, seed=42,
    prepared_data=None,
):
    """Run a single explainer and return results."""
    device = cfg.training.device
    max_len = cfg.data.max_len
    num_items = info['num_items']

    explainer = build_explainer(cfg, model, num_items)
    explainer.forbid_constraint_items = bool(cfg.explainer.get('forbid_constraint_items', False))
    rank_constraint = get_rank_constraint_spec(cfg=cfg)

    model_max_len = getattr(model, 'max_len', max_len)
    return run_and_score_explainer(
        model=model,
        explainer=explainer,
        user_seqs=user_seqs,
        max_len=model_max_len,
        popular_items=popular_items,
        category_map=category_map,
        k_pos=cfg.explainer.k_pos,
        k_item=cfg.explainer.k_item,
        n_users=cfg.evaluation.n_users,
        device=device,
        seed=seed,
        fidelity_k=cfg.evaluation.fidelity_k,
        target_mode=rank_constraint.get('mode') if rank_constraint is not None else None,
        rank_constraint=rank_constraint,
        prepared_data=prepared_data,
    )


def _get_seed_local_shared_state(
    cache: Dict[int, dict],
    seed: int,
    *,
    model,
    user_seqs,
    max_len: int,
    n_users: int,
    device: str,
    rank_constraint,
) -> dict:
    cached = cache.get(seed)
    if cached is not None:
        return cached

    prepared_data = prepare_explanation_inputs(
        model=model,
        user_seqs=user_seqs,
        max_len=max_len,
        n_users=n_users,
        device=device,
        seed=seed,
        target_mode=rank_constraint.get('mode') if rank_constraint is not None else None,
        rank_constraint=rank_constraint,
    )
    cache[seed] = prepared_data
    return cache[seed]


def main():
    parser = argparse.ArgumentParser(description='Generate CF explanations')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--all-explainers', action='store_true',
                        help='Run all paper explainers')
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional run tag appended to result filenames')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, base_dir=SCRIPT_DIR), args.overrides)
    if args.tag is not None:
        cfg.evaluation.run_tag = args.tag

    device = cfg.training.device
    dataset = cfg.data.dataset
    model_name = cfg.model.name

    # Load data
    data_bundle = load_experiment_data(cfg, dataset)
    user_seqs = data_bundle['user_seqs']
    info = data_bundle['info']
    popular_items = data_bundle['popular_items']
    category_map = data_bundle['category_map']

    # Load trained model
    try:
        model, checkpoint_path = load_experiment_model(cfg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Please run run_train.py first.")
        return

    print(f"Loaded {model_name} trained on {dataset}")

    explainer_names = ['random', 'greedy', 'ga', 'beam', 'gradient', 'heuristic'] if args.all_explainers else [cfg.explainer.name]

    results_dir = ensure_dir(cfg.evaluation.results_dir)

    all_metrics = {}

    run_rank_constraint = get_rank_constraint_spec(cfg=cfg)
    run_max_len = getattr(model, 'max_len', cfg.data.max_len)
    seed_shared_state: Dict[int, dict] = {}

    for exp_name in explainer_names:
        print(f"\n{'='*60}")
        print(f"Running {exp_name} explainer on {model_name}/{dataset}")
        print(f"{'='*60}")

        exp_cfg = clone_config(cfg)
        exp_cfg.explainer.name = exp_name
        if exp_name == 'gradient':
            print("  Grad-Beam settings:")
            print(
                "   "
                f"alpha={exp_cfg.gradient.alpha}, "
                f"top_k_positions={exp_cfg.gradient.top_k_positions}, "
                f"n_candidates_per_position={exp_cfg.gradient.n_candidates_per_position}, "
                f"beam_width={getattr(exp_cfg.gradient, 'beam_width', 3)}, "
                f"use_hotflip={getattr(exp_cfg.gradient, 'use_hotflip', True)}, "
                f"use_blocker_loss={getattr(exp_cfg.gradient, 'use_blocker_loss', True)}"
            )

        n_seeds = cfg.evaluation.n_seeds
        seed_metrics = []

        for seed_idx in range(n_seeds):
            seed = cfg.seed + seed_idx
            print(f"\n  Seed {seed_idx + 1}/{n_seeds} (seed={seed})")

            prepared_data = _get_seed_local_shared_state(
                seed_shared_state, seed,
                model=model,
                user_seqs=user_seqs,
                max_len=run_max_len,
                n_users=cfg.evaluation.n_users,
                device=device,
                rank_constraint=run_rank_constraint,
            )

            run_results, metrics = run_single_explainer(
                exp_cfg, model, user_seqs, info, popular_items, category_map, seed,
                prepared_data=prepared_data,
            )
            seed_metrics.append(metrics)

            print(f"  Results (seed={seed}):")
            print(f"    Hamming={metrics.get('hamming_mean', 0):.2f}")
            print(f"    Queries={metrics['queries']:.0f}")
            delta_nll = metrics.get('delta_nll')
            if delta_nll is not None:
                print(f"    ΔNLL={delta_nll:.4f}")
            print(f"    Time={metrics.get('time_seconds', 0):.1f}s")

            # Save per-seed results
            file_suffix = target_suffix(cfg=cfg) + run_tag_suffix(cfg=cfg)
            save_path = os.path.join(
                results_dir,
                f"explain_{model_name}_{dataset}_{exp_name}{file_suffix}_seed{seed}.json"
            )
            save_data = {
                'metrics': metrics,
                'per_user': [
                    {
                        **{k: v for k, v in r.items() if k != 'cf_seq'},
                        'target_rec': r.get('target_rec'),
                    }
                    for r in run_results['per_user_results']
                ],
            }
            save_json(save_path, save_data)
            clear_device_cache(device)

        # Aggregate
        all_metrics[exp_name] = numeric_stats(seed_metrics)

    # Print comparison table
    if len(explainer_names) > 1:
        display_names = {
            'random': 'random',
            'greedy': 'greedy',
            'ga': 'ga',
            'beam': 'no-grad beam',
            'gradient': 'grad-beam',
            'heuristic': 'baseline_heuristic',
        }
        print(f"\n{'='*60}")
        print(f"Comparison: {model_name} on {dataset}")
        print(f"{'='*60}")
        print(f"{'Method':<14} {'Validity':>10} {'Hamming':>10} {'Queries':>10} {'ΔNLL':>8}")
        print("-" * 58)
        for exp_name in explainer_names:
            m = all_metrics[exp_name]
            vr  = f"{m.get('validity_rate_mean', 0):.3f}±{m.get('validity_rate_std', 0):.3f}"
            ham = f"{m.get('hamming_mean_mean', 0):.2f}±{m.get('hamming_mean_std', 0):.2f}"
            qc  = f"{m.get('queries_mean', 0):.0f}"
            delta_nll = m.get('delta_nll_mean')
            delta_nll_text = f"{delta_nll:.4f}" if delta_nll is not None else "N/A"
            print(f"{display_names.get(exp_name, exp_name):<14} {vr:>10} {ham:>10} {qc:>10} {delta_nll_text:>8}")

    # Save all metrics
    _tsuf = target_suffix(cfg=cfg) + run_tag_suffix(cfg=cfg)
    if len(explainer_names) == 1:
        summary_filename = f"summary_{model_name}_{dataset}_{explainer_names[0]}{_tsuf}.json"
    else:
        summary_filename = f"summary_{model_name}_{dataset}{_tsuf}.json"
    summary_path = os.path.join(results_dir, summary_filename)
    save_json(summary_path, all_metrics)
    print(f"\nSaved summary to {summary_path}")


if __name__ == '__main__':
    main()
