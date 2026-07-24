"""
GECE: Genetic Counterfactual Explanations (Algorithm 2 from paper).

Paper: "Demystifying Sequential Recommendations: Counterfactual Explanations
via Genetic Algorithms" (Scarcelli et al., arXiv:2508.03606, 2025)

Algorithm 2:
  1. Initialize population with N identical copies of source sequence
  2. Compute source logits
  3. For each generation:
     a. Apply mutation (Replace/Add/Delete) at random position with random item
     b. Apply crossover
     c. Compute fitness = w_edit * edit_distance + w_logit * logit_distance
     d. Select best individuals
  4. Return final population

Fixed-length sequences: Add/Delete mutations maintain sequence length.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseExplainer, RankConstraint


class GAExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        num_items: int,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        w_edit: float = 1.0,
        w_logit: float = 1.0,
        tournament_size: int = 5,
        elitism_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(model, num_items, **kwargs)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.w_edit = w_edit
        self.w_logit = w_logit
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio

    def _initialize_population(self, original_seq: List[int]) -> np.ndarray:
        return np.tile(np.array(original_seq, dtype=np.int32), (self.population_size, 1))

    def _compute_source_logits(self, original_seq: List[int]) -> torch.Tensor:
        seq_tensor = torch.tensor([original_seq], dtype=torch.long, device=self.device)
        amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                   if self.device == 'cuda' else contextlib.nullcontext())
        with amp_ctx, torch.no_grad():
            scores = self.model.predict(seq_tensor)
        self.query_count += 1
        return scores[0].float()

    def _compute_fitness_batch(
        self,
        population: np.ndarray,
        original_seq: List[int],
        source_logits: Optional[torch.Tensor],
        original_rec: int,
        target_rec: Optional[int] = None,
        rank_constraint: Optional[RankConstraint] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(population)
        if N == 0:
            return np.array([]), np.array([]), np.array([], dtype=bool)

        seq_tensor = torch.from_numpy(np.ascontiguousarray(population)).long().to(self.device)
        amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                   if self.device == 'cuda' else contextlib.nullcontext())
        with amp_ctx, torch.no_grad():
            all_scores = self.model.predict(seq_tensor)
        all_scores = all_scores.float()
        self.query_count += N

        orig_arr = np.array(original_seq, dtype=np.int32)
        edit_dists = np.sum(
            (population != orig_arr) & (orig_arr != 0),
            axis=1,
        ).astype(np.float32)
        within_budget = edit_dists <= float(self.max_edits)

        masked_scores = all_scores.clone()
        self.mask_scores_batch_inplace(masked_scores, seq_tensor)
        if target_rec is not None and rank_constraint is None:
            validities = masked_scores.argmax(dim=1).eq(target_rec).detach().cpu().numpy()
            validities = np.logical_and(validities, within_budget)
            target_score_gap = (all_scores[:, original_rec] - all_scores[:, target_rec]).detach().cpu().numpy()
            fitnesses = self.w_edit * edit_dists + self.w_logit * target_score_gap
            return fitnesses, target_score_gap, validities

        if rank_constraint is not None:
            constraint = self.resolve_rank_constraint(target_rec=target_rec, rank_constraint=rank_constraint)
            if constraint is None:
                raise ValueError('rank_constraint branch requires a resolved constraint')
            summaries = [constraint.summarize(masked_scores[i]) for i in range(N)]
            validities = np.asarray([summary.satisfied for summary in summaries], dtype=bool)
            validities = np.logical_and(validities, within_budget)
            violations = np.asarray([summary.beam_sort_key[0] for summary in summaries], dtype=np.float32)
            score_terms = np.asarray([summary.beam_sort_key[2] for summary in summaries], dtype=np.float32)
            fitnesses = self.w_edit * edit_dists + self.w_logit * violations + 0.01 * score_terms
            return fitnesses, violations, validities

        if source_logits is None:
            raise ValueError('source_logits must be provided for untargeted GA fitness')

        sl = source_logits.unsqueeze(0).expand(N, -1)
        logit_dists = torch.mean((sl - all_scores) ** 2, dim=1).detach().cpu().numpy()
        validities = masked_scores.argmax(dim=1).ne(original_rec).detach().cpu().numpy()
        validities = np.logical_and(validities, within_budget)

        fitnesses = self.w_edit * edit_dists + self.w_logit * logit_dists
        return fitnesses, logit_dists, validities

    def _tournament_batch(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        N = len(population)
        k = min(self.tournament_size, N)
        idx = np.random.randint(0, N, size=(n_select, k))
        row_fits = fitnesses[idx]
        best_local = np.argmin(row_fits, axis=1)
        parent_idx = idx[np.arange(n_select), best_local]
        return population[parent_idx].copy()

    def _crossover_batch(self, parents1: np.ndarray, parents2: np.ndarray) -> np.ndarray:
        N, L = parents1.shape
        non_pad = parents1 != 0
        cx_mask = (np.random.random((N, L)) < self.crossover_rate) & non_pad
        return np.where(cx_mask, parents2, parents1)

    def _mutate_batch(
        self,
        children: np.ndarray,
        mutate_rows: np.ndarray,
        available_arr: np.ndarray,
    ) -> None:
        if mutate_rows.size == 0 or available_arr.size == 0:
            return

        subset = children[mutate_rows].copy()
        active_mask = subset != 0
        active_counts = active_mask.sum(axis=1)
        valid_rows_mask = active_counts > 0
        if not np.any(valid_rows_mask):
            return

        valid_rows = mutate_rows[valid_rows_mask]
        subset = subset[valid_rows_mask]
        active_mask = active_mask[valid_rows_mask]
        row_count, seq_len = subset.shape

        pos_scores = np.random.random((row_count, seq_len))
        pos_scores[~active_mask] = -1.0
        positions = pos_scores.argmax(axis=1)
        current_items = subset[np.arange(row_count), positions]

        n_available = available_arr.size
        insert_idx = np.searchsorted(available_arr, current_items)
        current_present = (
            (insert_idx < n_available) &
            (available_arr[np.clip(insert_idx, 0, n_available - 1)] == current_items)
        )
        choice_sizes = n_available - current_present.astype(np.int32)
        selectable_mask = choice_sizes > 0
        if not np.any(selectable_mask):
            return

        valid_rows = valid_rows[selectable_mask]
        subset = subset[selectable_mask]
        positions = positions[selectable_mask]
        current_items = current_items[selectable_mask]
        insert_idx = insert_idx[selectable_mask]
        current_present = current_present[selectable_mask]
        choice_sizes = choice_sizes[selectable_mask]

        row_count = subset.shape[0]
        base_choice = np.floor(np.random.random(row_count) * choice_sizes).astype(np.int32)
        chosen_idx = base_choice.copy()
        shift_mask = current_present & (base_choice >= insert_idx)
        chosen_idx[shift_mask] += 1
        new_items = available_arr[chosen_idx]

        ops = np.random.randint(1, 4, size=row_count)
        row_idx = np.arange(row_count)

        replace_mask = ops == 1
        if np.any(replace_mask):
            replace_rows = row_idx[replace_mask]
            subset[replace_rows, positions[replace_mask]] = new_items[replace_mask]

        base_idx = np.arange(seq_len, dtype=np.int32)

        add_mask = ops == 2
        if np.any(add_mask):
            add_rows = row_idx[add_mask]
            add_pos = positions[add_mask]
            add_subset = subset[add_mask]
            add_index = np.broadcast_to(base_idx, add_subset.shape).copy()
            add_index[base_idx[None, :] > add_pos[:, None]] -= 1
            add_index[base_idx[None, :] == add_pos[:, None]] = 0
            add_result = add_subset[np.arange(add_subset.shape[0])[:, None], add_index]
            add_result[np.arange(add_subset.shape[0]), add_pos] = new_items[add_mask]
            subset[add_rows] = add_result

        delete_mask = ops == 3
        if np.any(delete_mask):
            delete_rows = row_idx[delete_mask]
            delete_pos = positions[delete_mask]
            delete_subset = subset[delete_mask]
            delete_index = np.broadcast_to(base_idx, delete_subset.shape).copy()
            tail_mask = base_idx[None, :] >= delete_pos[:, None]
            delete_index[tail_mask] += 1
            delete_index[:, -1] = seq_len - 1
            delete_result = delete_subset[np.arange(delete_subset.shape[0])[:, None], delete_index]
            delete_result[:, -1] = 0
            subset[delete_rows] = delete_result

        children[valid_rows] = subset

    def explain(
        self,
        original_seq: List[int],
        original_rec: int,
        candidate_positions: Any,
        candidate_items: Any,
        target_rec: Optional[int] = None,
        rank_constraint: Optional[RankConstraint] = None,
    ) -> Tuple[Optional[List[int]], Dict]:
        self.reset_query_count()
        legacy_targeted = target_rec is not None and rank_constraint is None
        constraint = None if legacy_targeted else self.resolve_rank_constraint(target_rec=target_rec, rank_constraint=rank_constraint)

        if isinstance(candidate_positions, int):
            num_items = candidate_positions
        else:
            num_items = self.num_items

        if isinstance(candidate_items, set):
            user_items = candidate_items
        else:
            user_items = set(candidate_items)

        population = self._initialize_population(original_seq)
        source_logits = None if (constraint is not None or legacy_targeted) else self._compute_source_logits(original_seq)
        available_items = [i for i in range(1, num_items + 1) if i not in user_items]
        available_arr = np.array(available_items, dtype=np.int32)
        last_fitnesses = None
        last_validities = None

        for gen in range(self.generations):
            if self.budget_remaining() <= 0:
                break

            fitnesses, _, validities_gen = self._compute_fitness_batch(
                population,
                original_seq,
                source_logits,
                original_rec,
                target_rec=target_rec,
                rank_constraint=constraint,
            )
            last_fitnesses = fitnesses
            last_validities = validities_gen

            # Early stopping: halt as soon as a valid CF is found (matches
            # other methods' termination behaviour for fair query comparison).
            if np.any(validities_gen):
                break

            n_elite = max(1, int(self.population_size * self.elitism_ratio))
            elite_idx = np.argsort(fitnesses)[:n_elite]
            elite = population[elite_idx]

            n_new = self.population_size - n_elite
            parents1 = self._tournament_batch(population, fitnesses, n_new)
            parents2 = self._tournament_batch(population, fitnesses, n_new)
            children = self._crossover_batch(parents1, parents2)

            mutate_flags = np.random.random(n_new) < self.mutation_rate
            self._mutate_batch(children, np.where(mutate_flags)[0], available_arr)

            population = np.vstack([elite, children])

        if last_fitnesses is not None and last_validities is not None and len(last_fitnesses) == len(population):
            fitnesses = last_fitnesses
            validities = last_validities
        else:
            fitnesses, _, validities = self._compute_fitness_batch(
                population,
                original_seq,
                source_logits,
                original_rec,
                target_rec=target_rec,
                rank_constraint=constraint,
            )

        orig_arr = np.array(original_seq, dtype=np.int32)
        edit_dists = np.sum((population != orig_arr) & (orig_arr != 0), axis=1)
        order = np.lexsort((edit_dists, fitnesses))

        seen = set()
        top_k_cfs = []
        for i in order:
            row = np.ascontiguousarray(population[i])
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                seq = row.tolist()
                top_k_cfs.append({
                    'seq': seq,
                    'edit_distance': int(edit_dists[i]),
                    'valid': bool(validities[i]),
                })
            if len(top_k_cfs) >= self.population_size:
                break

        best_cf = None
        best_edits = self.max_edits + 1
        for cf in top_k_cfs:
            if cf['valid'] and cf['edit_distance'] < best_edits:
                best_cf = cf['seq']
                best_edits = cf['edit_distance']
        # No fallback to original sequence — return None if no valid CF found
        if best_cf is None:
            best_edits = 0

        info = {
            'method': 'ga',
            'valid': best_cf is not None,
            'edit_distance': best_edits if best_cf else 0,
            'queries_used': self.query_count,
            'generations': self.generations,
            'population_size': self.population_size,
            'top_k_cfs': top_k_cfs,
            'target_rec': constraint.item if constraint is not None and constraint.kind == 'bring_in' else target_rec,
        }
        if constraint is not None:
            info['rank_constraint'] = {
                'kind': constraint.kind,
                'top_k': constraint.top_k,
                'target_item': constraint.item,
            }

        return best_cf, info
