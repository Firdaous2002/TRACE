"""
Greedy baseline explainer.

At each step, tries all (position, item) combinations and picks the one
that reduces the original item's score the most. Repeats up to max_edits.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseExplainer, RankConstraint, hamming_distance


class GreedyExplainer(BaseExplainer):
    """
    Greedy counterfactual explainer.

    Strategy: At each edit step, exhaustively evaluate all (position, item)
    candidates and pick the one that causes the largest score drop for the
    originally recommended item.
    """

    def explain(
        self,
        original_seq: List[int],
        original_rec: int,
        candidate_positions: List[int],
        candidate_items: Dict[int, List[int]],
        target_rec: Optional[int] = None,
        rank_constraint: Optional[RankConstraint] = None,
    ) -> Tuple[Optional[List[int]], Dict]:
        self.reset_query_count()
        constraint = self.resolve_rank_constraint(target_rec=target_rec, rank_constraint=rank_constraint)

        s = list(original_seq)
        edited_positions = set()

        # Get original score for the recommended item
        original_scores = self.get_scores(s)
        original_score = original_scores[original_rec].item()
        original_target_score = original_scores[target_rec].item() if target_rec is not None else None
        final_top1 = original_rec
        current_summary = None
        if constraint is not None:
            self.mask_scores_inplace(original_scores, s)
            current_summary = constraint.summarize(original_scores)

        for step in range(self.max_edits):
            if self.budget_remaining() <= 0:
                break

            best_action: Optional[Tuple[int, int]] = None
            best_score_drop = -float('inf')
            best_constraint_key = None
            budget = self.budget_remaining()
            batch_size = 2048

            s_arr = np.array(s, dtype=np.int64)
            pending_chunks: List[np.ndarray] = []
            batch_actions: List[Tuple[int, int]] = []
            pending_count = 0
            total_candidates = 0

            def evaluate_pending() -> None:
                nonlocal best_action, best_score_drop, best_constraint_key, pending_chunks, batch_actions, pending_count
                if not pending_chunks:
                    return

                chunk_arr = np.vstack(pending_chunks)
                chunk_tensor = torch.from_numpy(np.ascontiguousarray(chunk_arr)).to(self.device)
                scores = self.get_scores_batch(chunk_tensor).float()

                if constraint is not None:
                    self.mask_scores_batch_inplace(scores, chunk_tensor)
                    for idx, action in enumerate(batch_actions):
                        summary = constraint.summarize(scores[idx])
                        key = summary.beam_sort_key
                        if best_constraint_key is None or key < best_constraint_key:
                            best_constraint_key = key
                            best_action = action
                    pending_chunks = []
                    batch_actions = []
                    pending_count = 0
                    return

                if target_rec is not None:
                    assert original_target_score is not None
                    chunk_gains = scores[:, target_rec] - original_target_score
                else:
                    chunk_gains = original_score - scores[:, original_rec]

                chunk_best_idx = int(chunk_gains.argmax().item())
                chunk_best_gain = float(chunk_gains[chunk_best_idx].item())
                if chunk_best_gain > best_score_drop:
                    best_score_drop = chunk_best_gain
                    best_action = batch_actions[chunk_best_idx]

                pending_chunks = []
                batch_actions = []
                pending_count = 0

            for p in candidate_positions:
                if total_candidates >= budget:
                    break
                if p in edited_positions:
                    continue
                items = candidate_items.get(p, [])
                if not items:
                    continue
                valid_items = [j for j in items if j != s[p]]
                if not valid_items:
                    continue
                start = 0
                while start < len(valid_items) and total_candidates < budget:
                    room = min(batch_size - pending_count, budget - total_candidates)
                    end = min(start + room, len(valid_items))
                    item_slice = valid_items[start:end]
                    if not item_slice:
                        break

                    chunk = np.repeat(s_arr[None, :], len(item_slice), axis=0)
                    chunk[:, p] = item_slice
                    pending_chunks.append(chunk)
                    batch_actions.extend((p, j) for j in item_slice)
                    pending_count += len(item_slice)
                    total_candidates += len(item_slice)
                    start = end

                    if pending_count == batch_size:
                        evaluate_pending()

            evaluate_pending()

            if best_action is None:
                break

            # Apply best action
            p, j = best_action
            s[p] = j
            edited_positions.add(p)

            # Check if recommendation changed
            if self.budget_remaining() > 0:
                if constraint is not None:
                    scores = self.get_scores(s)
                    self.mask_scores_inplace(scores, s)
                    current_summary = constraint.summarize(scores)
                    if current_summary.satisfied:
                        break
                else:
                    final_top1 = self.get_top1(s)
                    if (target_rec is not None and final_top1 == target_rec) or (target_rec is None and final_top1 != original_rec):
                        break

        is_valid = False
        if self.budget_remaining() > 0 or self.query_count <= self.query_budget:
            if constraint is not None:
                if current_summary is None:
                    scores = self.get_scores(s)
                    self.mask_scores_inplace(scores, s)
                    current_summary = constraint.summarize(scores)
                is_valid = current_summary.satisfied
            else:
                is_valid = (final_top1 == target_rec) if target_rec is not None else (final_top1 != original_rec)

        cf_seq = s if is_valid else None
        info = {
            'method': 'greedy',
            'valid': is_valid,
            'edit_distance': hamming_distance(original_seq, s) if is_valid else hamming_distance(original_seq, s),
            'queries_used': self.query_count,
            'target_rec': constraint.item if constraint is not None and constraint.kind == 'bring_in' else target_rec,
        }
        if constraint is not None:
            info['rank_constraint'] = {
                'kind': constraint.kind,
                'top_k': constraint.top_k,
                'target_item': constraint.item,
            }

        return cf_seq, info
