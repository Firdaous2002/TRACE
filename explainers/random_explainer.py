"""
Random baseline explainer.

Makes a single random (position, item) substitution and checks if the
recommendation changes. No iterative search — one attempt, one query.
This represents a true lower-bound baseline: what happens with zero guidance.
"""

import random
from typing import Dict, List, Optional, Tuple

from .base import BaseExplainer, RankConstraint, hamming_distance


class RandomExplainer(BaseExplainer):
    """
    Random counterfactual explainer.

    Strategy: Pick one random position and one random replacement item,
    check if the recommendation changes. Single attempt, single query.
    Serves as a lower-bound baseline showing performance with zero guidance.
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

        if not candidate_positions:
            info = {
                'method': 'random',
                'valid': False,
                'edit_distance': 0,
                'queries_used': 0,
                'target_rec': target_rec,
            }
            return None, info

        # Single random position
        pos = random.choice(candidate_positions)
        items = candidate_items.get(pos, [])

        if not items:
            info = {
                'method': 'random',
                'valid': False,
                'edit_distance': 0,
                'queries_used': 0,
                'target_rec': target_rec,
            }
            return None, info

        # Single random item
        item = random.choice(items)

        # Make replacement and check validity (1 query)
        cf_seq = list(original_seq)
        cf_seq[pos] = item
        if constraint is not None:
            scores = self.get_scores(cf_seq)
            self.mask_scores_inplace(scores, cf_seq)
            is_valid = constraint.is_satisfied(scores)
        else:
            top1 = self.get_top1(cf_seq)
            is_valid = (top1 == target_rec) if target_rec is not None else (top1 != original_rec)

        info = {
            'method': 'random',
            'valid': is_valid,
            'edit_distance': hamming_distance(original_seq, cf_seq),
            'queries_used': self.query_count,
            'target_rec': constraint.item if constraint is not None and constraint.kind == 'bring_in' else target_rec,
        }
        if constraint is not None:
            info['rank_constraint'] = {
                'kind': constraint.kind,
                'top_k': constraint.top_k,
                'target_item': constraint.item,
            }

        return cf_seq if is_valid else None, info
