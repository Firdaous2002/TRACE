from __future__ import annotations

import time
from typing import Optional

import numpy as np

from evaluation.evaluator import run_explanation_for_users
from evaluation.metrics import compute_all_metrics


def run_and_score_explainer(
    *,
    model,
    explainer,
    user_seqs,
    max_len: int,
    popular_items,
    category_map,
    k_pos,
    k_item,
    n_users: int,
    device: str,
    seed: int,
    fidelity_k,
    target_mode: Optional[str] = None,
    rank_constraint=None,
    position_selection: str = 'all',
    prepared_data=None,
):
    start_time = time.time()
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
        position_selection=position_selection,
        target_mode=target_mode,
        rank_constraint=rank_constraint,
        prepared_data=prepared_data,
    )

    metrics = compute_all_metrics(
        results=run_results['per_user_results'],
        original_seqs=run_results['original_seqs'],
        cf_seqs=run_results['cf_seqs'],
        top_k_cfs=run_results.get('top_k_cfs', []),
        fidelity_k=list(fidelity_k),
        model=model,
        max_len=max_len,
        device=device,
    )

    per_user = run_results['per_user_results']
    validity_rate = float(np.mean([1.0 if r.get('valid', False) else 0.0 for r in per_user])) if per_user else 0.0
    metrics['validity_rate'] = validity_rate
    metrics['validity'] = validity_rate
    metrics['time_seconds'] = time.time() - start_time
    return run_results, metrics
