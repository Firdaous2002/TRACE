from __future__ import annotations

import importlib
from typing import Any


def _cfg_value(cfg_section: Any, key: str, default: Any) -> Any:
    return getattr(cfg_section, key, default)


def build_explainer(cfg, model, num_items: int):
    max_edits = int(cfg.explainer.max_edits)
    query_budget = int(cfg.explainer.query_budget)
    device = str(cfg.training.device)

    name = cfg.explainer.name
    if name == 'random':
        cls = getattr(importlib.import_module('explainers.random_explainer'), 'RandomExplainer')
        return cls(
            model=model,
            num_items=num_items,
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
        )
    if name == 'greedy':
        cls = getattr(importlib.import_module('explainers.greedy_explainer'), 'GreedyExplainer')
        return cls(
            model=model,
            num_items=num_items,
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
        )
    if name == 'ga':
        cls = getattr(importlib.import_module('explainers.ga_explainer'), 'GAExplainer')
        return cls(
            population_size=cfg.ga.population_size,
            generations=cfg.ga.generations,
            mutation_rate=cfg.ga.mutation_rate,
            crossover_rate=cfg.ga.crossover_rate,
            w_edit=cfg.ga.w_edit,
            w_logit=cfg.ga.w_logit,
            tournament_size=cfg.ga.tournament_size,
            elitism_ratio=cfg.ga.elitism_ratio,
            model=model,
            num_items=num_items,
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
        )
    if name == 'gradient':
        cls = getattr(importlib.import_module('explainers.gradient_explainer'), 'GradientExplainer')
        return cls(
            model=model,
            num_items=num_items,
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
            alpha=cfg.gradient.alpha,
            alpha_search_steps=cfg.gradient.alpha_search_steps,
            top_k_positions=cfg.gradient.top_k_positions,
            n_candidates_per_position=cfg.gradient.n_candidates_per_position,
            position_selection=_cfg_value(cfg.gradient, 'position_selection', 'gradient'),
            item_selection=_cfg_value(cfg.gradient, 'item_selection', 'cosine'),
            alpha_search=_cfg_value(cfg.gradient, 'alpha_search', 'multi'),
            objective_variant=str(_cfg_value(cfg.gradient, 'objective_variant', 'legacy')),
            rank_loss_tau=float(_cfg_value(cfg.gradient, 'rank_loss_tau', 1.0)),
            rank_loss_softplus_tau=float(_cfg_value(cfg.gradient, 'rank_loss_softplus_tau', 1.0)),
            rank_loss_margin=float(_cfg_value(cfg.gradient, 'rank_loss_margin', 0.0)),
            rank_boundary_band=int(_cfg_value(cfg.gradient, 'rank_boundary_band', 2)),
            rank_anchor_weight=float(_cfg_value(cfg.gradient, 'rank_anchor_weight', 0.1)),
            use_blocker_loss=bool(_cfg_value(cfg.gradient, 'use_blocker_loss', True)),
            blocker_variant=str(_cfg_value(cfg.gradient, 'blocker_variant', 'hard')),
            blocker_top_m=int(_cfg_value(cfg.gradient, 'blocker_top_m', 10)),
            blocker_tau=float(_cfg_value(cfg.gradient, 'blocker_tau', 1.0)),
            blocker_weight_tau=float(_cfg_value(cfg.gradient, 'blocker_weight_tau', 1.0)),
            blocker_weight=float(_cfg_value(cfg.gradient, 'blocker_weight', 1.0)),
            use_hotflip=bool(_cfg_value(cfg.gradient, 'use_hotflip', True)),
            beam_width=int(_cfg_value(cfg.gradient, 'beam_width', 3)),
            beam_rerank_factor=int(_cfg_value(cfg.gradient, 'beam_rerank_factor', 1)),
            context_candidate_top_m=int(_cfg_value(cfg.gradient, 'context_candidate_top_m', 0)),
            hotflip_distance_weight=float(_cfg_value(cfg.gradient, 'hotflip_distance_weight', 0.01)),
            hotflip_rss_weight=float(_cfg_value(cfg.gradient, 'hotflip_rss_weight', 0.0)),
            hotflip_context_weight=float(_cfg_value(cfg.gradient, 'hotflip_context_weight', 0.0)),
            local_cosine_top_m=int(_cfg_value(cfg.gradient, 'local_cosine_top_m', 0)),
            eval_batch_size=int(_cfg_value(cfg.gradient, 'eval_batch_size', 256)),
        )
    if name == 'beam':
        cls = getattr(importlib.import_module('explainers.beam_explainer'), 'BeamSearchExplainer')
        return cls(
            model=model,
            num_items=num_items,
            beam_width=int(_cfg_value(cfg.beam, 'beam_width', 3)),
            k_context=int(_cfg_value(cfg.beam, 'k_context', 2)),
            k_target_nn=int(_cfg_value(cfg.beam, 'k_target_nn', 2)),
            k_local_nn=int(_cfg_value(cfg.beam, 'k_local_nn', 3)),
            candidate_strategy=str(_cfg_value(cfg.beam, 'candidate_strategy', 'pool')),
            position_strategy=str(_cfg_value(cfg.beam, 'position_strategy', 'all')),
            position_top_m=int(_cfg_value(cfg.beam, 'position_top_m', 0)),
            influence_gamma=float(_cfg_value(cfg.beam, 'influence_gamma', 1.0)),
            pacer_shortlist_size=int(_cfg_value(cfg.beam, 'pacer_shortlist_size', 0)),
            pacer_weight_context=float(_cfg_value(cfg.beam, 'pacer_weight_context', 1.0)),
            pacer_weight_target=float(_cfg_value(cfg.beam, 'pacer_weight_target', 1.0)),
            pacer_weight_local=float(_cfg_value(cfg.beam, 'pacer_weight_local', 0.5)),
            eval_batch_size=int(_cfg_value(cfg.beam, 'eval_batch_size', 256)),
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
        )
    if name =='heuristic':
        cls = getattr(importlib.import_module('explainers.heuristic_explainer'), 'HeuristicExplainer')
        return cls(
            model=model,
            num_items=num_items,
            max_edits=max_edits,
            query_budget=query_budget,
            device=device,
        )

    raise ValueError(f"Unknown explainer: {name}")
