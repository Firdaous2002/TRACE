"""
Evaluation pipeline for running all explainers and comparing results.
"""

import importlib
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from explainers.factory import build_explainer
from explainers.base import RankConstraint


def _target_topk_size(target_mode: Optional[str], num_items: int) -> int:
    if target_mode is None:
        return 1
    if target_mode == 'easy':
        return min(10, num_items)
    if target_mode == 'hard':
        return min(50, num_items)
    raise ValueError(f'Unknown target_mode: {target_mode}')


def _target_rank_window(rank_constraint: Optional[Dict[str, Any]], num_items: int) -> Optional[tuple[int, int]]:
    if rank_constraint is None:
        return None

    rank_min = rank_constraint.get('rank_min')
    rank_max = rank_constraint.get('rank_max')
    if rank_min is None and rank_max is None:
        return None

    lo = max(1, int(rank_min if rank_min is not None else 1))
    hi = max(lo, int(rank_max if rank_max is not None else num_items))
    hi = min(hi, num_items)
    return (lo, hi)


def _target_candidates_from_preds(preds: List[int], target_mode: str) -> List[int]:
    if target_mode == 'easy':
        return preds[1:min(10, len(preds))]
    if target_mode == 'hard':
        return preds[10:min(50, len(preds))]
    raise ValueError(f'Unknown target_mode: {target_mode}')


def _legacy_rank_constraint(target_mode: Optional[str]) -> Optional[Dict[str, Any]]:
    if target_mode is None:
        return None
    return {'kind': 'bring_in', 'mode': target_mode, 'top_k': 1}


def _constraint_topk_size(rank_constraint: Optional[Dict[str, Any]], num_items: int) -> int:
    if rank_constraint is None:
        return 1
    kind = rank_constraint.get('kind', 'bring_in')
    if kind == 'bring_in':
        rank_window = _target_rank_window(rank_constraint, num_items)
        if rank_window is not None:
            return rank_window[1]
        return max(
            1,
            min(
                num_items,
                _target_topk_size(rank_constraint.get('mode'), num_items),
            ),
        )
    if kind == 'push_out':
        return max(1, min(num_items, int(rank_constraint.get('top_k', 1))))
    raise ValueError(f"Unknown rank constraint kind: {kind}")


def _build_user_rank_constraint(preds: List[int], rank_constraint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if rank_constraint is None:
        return None

    kind = rank_constraint.get('kind', 'bring_in')
    if kind == 'bring_in':
        rank_window = _target_rank_window(rank_constraint, len(preds))
        lo_rank = hi_rank = None
        if rank_window is not None:
            lo_rank, hi_rank = rank_window
            start_idx = max(0, lo_rank - 1)
            end_idx = min(len(preds), hi_rank)
            candidates = preds[start_idx:end_idx]
            target_mode = None
        else:
            target_mode = str(rank_constraint.get('mode', 'easy'))
            candidates = _target_candidates_from_preds(preds, target_mode)
        target_item = int(random.choice(candidates)) if candidates else None
        spec = {
            'kind': 'bring_in',
            'top_k': max(1, int(rank_constraint.get('top_k', 1))),
            'target_item': target_item,
        }
        if target_mode is not None:
            spec['mode'] = target_mode
        if rank_window is not None:
            spec['rank_min'] = lo_rank
            spec['rank_max'] = hi_rank
        return spec

    if kind == 'push_out':
        return {
            'kind': 'push_out',
            'top_k': max(1, int(rank_constraint.get('top_k', 1))),
            'target_item': int(preds[0]) if preds else None,
        }

    raise ValueError(f"Unknown rank constraint kind: {kind}")


def _rank_constraint_is_satisfied(model, cf_seq: Optional[List[int]], rank_constraint: Optional[Dict[str, Any]], device: str) -> bool:
    if cf_seq is None or rank_constraint is None:
        return False

    target_item = rank_constraint.get('target_item')
    if target_item is None:
        return False

    top_k = max(1, int(rank_constraint.get('top_k', 1)))
    seq_tensor = torch.tensor([cf_seq], dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model.top_k(seq_tensor, k=top_k)[0].tolist()

    if rank_constraint.get('kind') == 'push_out':
        return int(target_item) not in preds
    if rank_constraint.get('kind') == 'bring_in':
        return int(target_item) in preds
    raise ValueError(f"Unknown rank constraint kind: {rank_constraint.get('kind')}")


def _to_explainer_rank_constraint(rank_constraint: Optional[Dict[str, Any]]) -> Optional[RankConstraint]:
    if rank_constraint is None:
        return None
    target_item = rank_constraint.get('target_item')
    if target_item is None:
        return None
    top_k = max(1, int(rank_constraint.get('top_k', 1)))
    kind = rank_constraint.get('kind', 'bring_in')
    if kind == 'bring_in':
        return RankConstraint.targeted(int(target_item), top_k=top_k)
    if kind == 'push_out':
        return RankConstraint.push_out(int(target_item), top_k=top_k)
    raise ValueError(f"Unknown rank constraint kind: {kind}")


def _build_available_items(all_items: List[int], user_items: set) -> List[int]:
    contains = user_items.__contains__
    return [item for item in all_items if not contains(item)]


def prepare_explanation_inputs(
    model,
    user_seqs: Dict[int, List[int]],
    max_len: int,
    n_users: int = 1000,
    device: str = 'cuda',
    seed: int = 42,
    target_mode: Optional[str] = None,
    rank_constraint: Optional[Dict[str, Any]] = None,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_uids = sorted(user_seqs.keys())
    if len(all_uids) > n_users:
        test_uids = random.sample(all_uids, n_users)
    else:
        test_uids = all_uids

    all_input_seqs = {}
    for uid in test_uids:
        seq = user_seqs[uid]
        if len(seq) < 3:
            continue
        input_seq = seq[:-1][-max_len:]
        pad_len = max_len - len(input_seq)
        input_seq = [0] * pad_len + list(input_seq)
        all_input_seqs[uid] = input_seq

    uid_list = list(all_input_seqs.keys())
    original_recs = {}
    target_recs: Dict[int, Optional[int]] = {}
    rank_constraints: Dict[int, Optional[Dict[str, Any]]] = {}
    chunk_size = 512
    active_rank_constraint = rank_constraint or _legacy_rank_constraint(target_mode)
    k_top = _constraint_topk_size(active_rank_constraint, model.num_items)

    for start in range(0, len(uid_list), chunk_size):
        chunk_uids = uid_list[start:start + chunk_size]
        chunk_seqs = [all_input_seqs[u] for u in chunk_uids]
        seq_tensor = torch.tensor(chunk_seqs, dtype=torch.long, device=device)
        with torch.no_grad():
            topk = model.top_k(seq_tensor, k=k_top)
        # Single CUDA->CPU transfer for the whole chunk (was per-row .tolist()).
        topk_rows = topk.tolist()
        for j, uid in enumerate(chunk_uids):
            preds = topk_rows[j]
            original_recs[uid] = preds[0]
            user_rank_constraint = _build_user_rank_constraint(preds, active_rank_constraint)
            rank_constraints[uid] = user_rank_constraint
            if user_rank_constraint is None:
                continue
            target_recs[uid] = user_rank_constraint.get('target_item') if user_rank_constraint.get('kind') == 'bring_in' else None

    return {
        'test_uids': uid_list,
        'all_input_seqs': all_input_seqs,
        'original_recs': original_recs,
        'target_recs': target_recs,
        'rank_constraints': rank_constraints,
    }


def run_explanation_for_users(
    model,
    explainer,
    user_seqs: Dict[int, List[int]],
    max_len: int,
    popular_items: List[int],
    category_map: Optional[Dict[int, List[str]]],
    k_pos: Optional[int],
    k_item: Optional[int] = None,
    n_users: int = 1000,
    device: str = 'cuda',
    seed: int = 42,
    position_selection: str = 'all',
    target_mode: Optional[str] = None,
    rank_constraint: Optional[Dict[str, Any]] = None,
    prepared_data: Optional[Dict] = None,
) -> Dict:
    """
    Run CF explanation for a set of test users.

    Returns dict with per-user results and aggregate metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()

    if prepared_data is None:
        prepared_data = prepare_explanation_inputs(
            model=model,
            user_seqs=user_seqs,
            max_len=max_len,
            n_users=n_users,
            device=device,
            seed=seed,
            target_mode=target_mode,
            rank_constraint=rank_constraint,
        )

    all_input_seqs = prepared_data['all_input_seqs']
    test_uids = [uid for uid in prepared_data['test_uids'] if uid in all_input_seqs]
    original_recs = prepared_data['original_recs']
    target_recs = prepared_data['target_recs']
    rank_constraints = prepared_data.get('rank_constraints', {})
    all_items = list(range(1, model.num_items + 1))
    score_positions_fn = None
    explainer_name = explainer.__class__.__name__
    uses_model_candidate_mode = explainer_name in ('GAExplainer', 'GradientExplainer')
    if position_selection == 'score_drop':
        score_positions_fn = importlib.import_module('utils.item_candidates').score_positions

    results = []
    original_seqs = []
    cf_seqs = []
    top_k_cfs_all = []  # for paper Fidelity@k (multi-explanation)
    valid_so_far = 0

    disable_tqdm = (
        os.environ.get('TQDM_DISABLE', '').lower() in {'1', 'true', 'yes'}
        or not sys.stderr.isatty()
    )
    pbar = tqdm(
        test_uids,
        desc=f"Explaining ({explainer.__class__.__name__})",
        disable=disable_tqdm,
        dynamic_ncols=not disable_tqdm,
        mininterval=0.5,
    )
    for i, uid in enumerate(pbar):
        seq = user_seqs[uid]
        input_seq = all_input_seqs[uid]
        original_rec = original_recs[uid]
        user_rank_constraint = rank_constraints.get(uid)
        explainer_rank_constraint = _to_explainer_rank_constraint(user_rank_constraint)
        target_rec = target_recs.get(uid) if user_rank_constraint is not None and user_rank_constraint.get('kind') == 'bring_in' else None
        user_items = set(seq)

        if uses_model_candidate_mode:
            cf_seq, info = explainer.explain(
                input_seq,
                original_rec,
                model.num_items,
                user_items,
                target_rec=target_rec if explainer_rank_constraint is None else None,
                rank_constraint=explainer_rank_constraint,
            )
        else:
            non_pad_positions = [p for p in range(len(input_seq)) if input_seq[p] != 0]
            available_items = _build_available_items(all_items, user_items)

            if position_selection == 'score_drop' and len(non_pad_positions) > 0:
                assert score_positions_fn is not None
                k_pos_for_scoring = k_pos if k_pos is not None else len(non_pad_positions)
                candidate_positions = score_positions_fn(
                    model, input_seq, original_rec, model.num_items,
                    k_pos=k_pos_for_scoring, device=device,
                )
                if not candidate_positions:
                    candidate_positions = non_pad_positions
            elif position_selection == 'random_k' and k_pos is not None and len(non_pad_positions) > k_pos:
                import random as _random
                candidate_positions = _random.sample(non_pad_positions, k_pos)
            else:
                candidate_positions = non_pad_positions

            if k_item is not None and len(available_items) > k_item:
                import random as _random
                items_subset = _random.sample(available_items, k_item)
            else:
                items_subset = available_items

            candidate_items = dict.fromkeys(candidate_positions, items_subset)
            cf_seq, info = explainer.explain(
                input_seq,
                original_rec,
                candidate_positions,
                candidate_items,
                target_rec=target_rec if explainer_rank_constraint is None else None,
                rank_constraint=explainer_rank_constraint,
            )

        info['user_id'] = uid
        info['original_rec'] = original_rec
        info['target_rec'] = target_rec
        info['rank_constraint'] = user_rank_constraint
        if user_rank_constraint is not None:
            info['constraint_kind'] = user_rank_constraint.get('kind')
            info['constraint_top_k'] = user_rank_constraint.get('top_k')
            info['constraint_item'] = user_rank_constraint.get('target_item')
            if user_rank_constraint.get('kind') == 'push_out' or int(user_rank_constraint.get('top_k', 1)) != 1:
                info['valid'] = _rank_constraint_is_satisfied(model, cf_seq, user_rank_constraint, device)
        results.append(info)
        if info.get('valid', False):
            valid_so_far += 1
        original_seqs.append(input_seq)
        cf_seqs.append(cf_seq)
        top_k_cfs_all.append(info.get('top_k_cfs', []))

        pbar.set_postfix(
            valid=f"{valid_so_far}/{len(results)}",
            vr=f"{valid_so_far/len(results):.2%}",
        )

    return {
        'per_user_results': results,
        'original_seqs': original_seqs,
        'cf_seqs': cf_seqs,
        'top_k_cfs': top_k_cfs_all,
    }


def evaluate_all_explainers(
    model,
    user_seqs: Dict[int, List[int]],
    cfg,
    popular_items: List[int],
    category_map: Optional[Dict[int, List[str]]] = None,
) -> Dict:
    """
    Evaluate all explainer methods and return comparison results.

    Runs each explainer over n_seeds and computes mean/std of metrics.
    """
    device = cfg.training.device
    compute_all_metrics = importlib.import_module('evaluation.metrics').compute_all_metrics
    max_len = cfg.data.max_len
    k_pos = cfg.explainer.k_pos
    k_item = cfg.explainer.k_item
    n_users = cfg.evaluation.n_users
    n_seeds = cfg.evaluation.n_seeds
    fidelity_k = list(cfg.evaluation.fidelity_k)
    num_items = model.num_items

    explainer_names = ['random', 'greedy', 'ga', 'gradient']
    all_results = {}

    for exp_name in explainer_names:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        from omegaconf import OmegaConf
        exp_cfg = OmegaConf.create(dict(cfg))
        exp_cfg.explainer.name = exp_name

        seed_metrics = []

        # For deterministic methods (greedy), only 1 seed needed
        actual_seeds = 1 if exp_name == 'greedy' else n_seeds

        for seed_idx in range(actual_seeds):
            seed = 42 + seed_idx
            print(f"\n  Seed {seed_idx + 1}/{actual_seeds} (seed={seed})")
            prepared_data = prepare_explanation_inputs(
                model=model,
                user_seqs=user_seqs,
                max_len=max_len,
                n_users=n_users,
                device=device,
                seed=seed,
            )

            explainer = build_explainer(exp_cfg, model, num_items)

            run_results = run_explanation_for_users(
                model=model,
                explainer=explainer,
                user_seqs=user_seqs,
                max_len=max_len,
                popular_items=popular_items,
                category_map=category_map,
                k_pos=k_pos,
                k_item=k_item,
                n_users=n_users,
                device=device,
                seed=seed,
                prepared_data=prepared_data,
            )

            # Compute metrics
            metrics = compute_all_metrics(
                results=run_results['per_user_results'],
                original_seqs=run_results['original_seqs'],
                cf_seqs=run_results['cf_seqs'],
                top_k_cfs=run_results.get('top_k_cfs', []),
                fidelity_k=fidelity_k,
                model=model,
                max_len=max_len,
                device=device,
            )

            seed_metrics.append(metrics)
            ham = metrics.get('hamming_mean', 0)
            delta_nll = metrics.get('delta_nll')
            delta_nll_text = f", ΔNLL={delta_nll:.4f}" if delta_nll is not None else ""
            print(f"  Hamming={ham:.2f}, Queries={metrics['queries']:.0f}{delta_nll_text}")

        # Aggregate across seeds
        agg_metrics = {}
        metric_keys = seed_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in seed_metrics]
            agg_metrics[f"{key}_mean"] = float(np.mean(values))
            agg_metrics[f"{key}_std"] = float(np.std(values))

        all_results[exp_name] = agg_metrics

        print(f"\n  Aggregated ({actual_seeds} seeds):")
        print(f"    Hamming: {agg_metrics.get('hamming_mean_mean', 0):.2f} ± {agg_metrics.get('hamming_mean_std', 0):.2f}")
        print(f"    Queries: {agg_metrics.get('queries_mean', 0):.0f} ± {agg_metrics.get('queries_std', 0):.0f}")
        delta_nll_mean = agg_metrics.get('delta_nll_mean')
        delta_nll_std = agg_metrics.get('delta_nll_std')
        if delta_nll_mean is not None and delta_nll_std is not None:
            print(f"    ΔNLL: {delta_nll_mean:.4f} ± {delta_nll_std:.4f}")

    return all_results
