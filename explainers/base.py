"""
Abstract base class for counterfactual explainers.
"""

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass
import os
from typing import Dict, List, Literal, Optional, Tuple

import torch


@dataclass(frozen=True, slots=True)
class RankConstraintSummary:
    item: int
    kind: Literal['bring_in', 'push_out']
    top_k: int
    rank: int
    score: float

    @property
    def satisfied(self) -> bool:
        if self.kind == 'bring_in':
            return self.rank <= self.top_k
        return self.rank > self.top_k

    @property
    def beam_sort_key(self) -> Tuple[int, int, float]:
        if self.kind == 'bring_in':
            violation = max(0, self.rank - self.top_k)
            return (violation, self.rank, -self.score)

        violation = max(0, self.top_k - self.rank + 1)
        return (violation, -self.rank, self.score)


@dataclass(frozen=True, slots=True)
class RankConstraint:
    item: int
    top_k: int = 1
    kind: Literal['bring_in', 'push_out'] = 'bring_in'

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError('RankConstraint.top_k must be >= 1')
        if self.kind not in {'bring_in', 'push_out'}:
            raise ValueError(f'Unsupported RankConstraint.kind: {self.kind}')

    @classmethod
    def targeted(cls, item: int, top_k: int = 1) -> 'RankConstraint':
        return cls(item=item, top_k=top_k, kind='bring_in')

    @classmethod
    def push_out(cls, item: int, top_k: int = 1) -> 'RankConstraint':
        return cls(item=item, top_k=top_k, kind='push_out')

    def summarize(self, scores: torch.Tensor) -> RankConstraintSummary:
        item_score = float(scores[self.item].item())
        item_rank = int((scores > scores[self.item]).sum().item()) + 1
        return RankConstraintSummary(
            item=self.item,
            kind=self.kind,
            top_k=self.top_k,
            rank=item_rank,
            score=item_score,
        )

    def is_satisfied(self, scores: torch.Tensor) -> bool:
        return self.summarize(scores).satisfied


class BaseExplainer(ABC):
    """
    Abstract interface for counterfactual explanation methods.

    All explainers take a trained model (black-box), an original sequence,
    and produce a counterfactual sequence that changes the top-1 recommendation.
    """

    def __init__(
        self,
        model,
        num_items: int,
        max_edits: int = 5,
        query_budget: int = 5000,
        device: str = 'cuda',
    ):
        self.model = model
        self.num_items = num_items
        self.max_edits = max_edits
        self.query_budget = query_budget
        self.device = device
        self.query_count = 0  # track black-box queries per instance

    def reset_query_count(self):
        """Reset the query counter for a new instance."""
        self.query_count = 0

    def _model_name(self) -> str:
        return self.model.__class__.__name__.lower()

    def _predict_chunk_size(self) -> int:
        override = os.environ.get('CF_PREDICT_BATCH_SIZE', '').strip()
        if override:
            try:
                return max(0, int(override))
            except ValueError:
                pass
        if self._model_name() == 'bert4rec' and str(self.device).startswith('cuda'):
            return 256
        return 0

    def _amp_ctx(self):
        disable_amp = os.environ.get('CF_DISABLE_AMP', '').lower() in {'1', 'true', 'yes'}
        auto_disable_amp = self._model_name() == 'bert4rec' and str(self.device).startswith('cuda')
        if str(self.device).startswith('cuda') and not disable_amp and not auto_disable_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    def _as_long_tensor(self, seqs: List[int] | List[List[int]] | torch.Tensor) -> torch.Tensor:
        target_device = torch.device(self.device)
        if isinstance(seqs, torch.Tensor):
            if seqs.dtype == torch.long and seqs.device == target_device:
                return seqs
            return seqs.to(device=target_device, dtype=torch.long)
        return torch.as_tensor(seqs, dtype=torch.long, device=target_device)

    def mask_scores_inplace(self, scores: torch.Tensor, seq: List[int] | Tuple[int, ...]) -> torch.Tensor:
        scores[0] = -float('inf')
        if not seq:
            return scores
        seen_items = [item for item in seq if item != 0]
        if seen_items:
            scores[seen_items] = -float('inf')
        return scores

    def mask_scores_batch_inplace(self, scores: torch.Tensor, seq_tensor: torch.Tensor) -> torch.Tensor:
        scores[:, 0] = -float('inf')
        if seq_tensor.numel() > 0:
            scores.scatter_(1, seq_tensor, -float('inf'))
            scores[:, 0] = -float('inf')
        return scores

    @torch.no_grad()
    def predict_scores_no_count(self, seq: List[int] | torch.Tensor) -> torch.Tensor:
        seq_tensor = self._as_long_tensor(seq)
        if seq_tensor.ndim == 1:
            seq_tensor = seq_tensor.unsqueeze(0)
        with self._amp_ctx():
            scores = self.model.predict(seq_tensor)
        return scores[0].float()

    @torch.no_grad()
    def predict_scores_batch_no_count(self, seqs: List[List[int]] | torch.Tensor) -> torch.Tensor:
        seq_tensor = self._as_long_tensor(seqs)
        chunk_size = self._predict_chunk_size()
        if chunk_size and seq_tensor.size(0) > chunk_size:
            chunks = []
            for start in range(0, seq_tensor.size(0), chunk_size):
                chunk_tensor = seq_tensor[start:start + chunk_size]
                with self._amp_ctx():
                    chunk_scores = self.model.predict(chunk_tensor)
                chunks.append(chunk_scores.float())
            return torch.cat(chunks, dim=0)
        with self._amp_ctx():
            scores = self.model.predict(seq_tensor)
        return scores.float()

    @torch.no_grad()
    def get_top1(self, seq: List[int]) -> int:
        """Get the top-1 recommended item for a sequence. Counts as 1 query."""
        self.query_count += 1
        scores = self.predict_scores_no_count(seq)
        self.mask_scores_inplace(scores, seq)
        return int(scores.argmax().item())

    @torch.no_grad()
    def get_scores(self, seq: List[int]) -> torch.Tensor:
        """Get full score vector for a sequence. Counts as 1 query."""
        self.query_count += 1
        return self.predict_scores_no_count(seq)

    @torch.no_grad()
    def get_scores_batch(self, seqs: List[List[int]] | torch.Tensor) -> torch.Tensor:
        """Get scores for a batch of sequences. Counts as len(seqs) queries."""
        self.query_count += len(seqs)
        return self.predict_scores_batch_no_count(seqs)

    def budget_remaining(self) -> int:
        """Number of queries remaining in budget."""
        return max(0, self.query_budget - self.query_count)

    def resolve_rank_constraint(
        self,
        target_rec: Optional[int] = None,
        rank_constraint: Optional[RankConstraint] = None,
    ) -> Optional[RankConstraint]:
        if target_rec is not None:
            targeted_constraint = RankConstraint.targeted(target_rec, top_k=1)
            if rank_constraint is None:
                return targeted_constraint
            if rank_constraint != targeted_constraint:
                raise ValueError('target_rec and rank_constraint specify different constraints')
        return rank_constraint

    @abstractmethod
    def explain(
        self,
        original_seq: List[int],
        original_rec: int,
        candidate_positions: List[int],
        candidate_items: Dict[int, List[int]],
        target_rec: Optional[int] = None,
        rank_constraint: Optional[RankConstraint] = None,
    ) -> Tuple[Optional[List[int]], Dict]:
        """
        Generate a counterfactual explanation.

        Args:
            original_seq: original interaction sequence (with padding)
            original_rec: the item originally recommended (top-1)
            candidate_positions: positions eligible for editing
            candidate_items: {position: [candidate_item_ids]}
            target_rec: legacy alias for targeted top-1 CFs.
            rank_constraint: rank validity constraint. Supports at least bringing
                            an item into top-k or pushing an item beyond top-k.
                            If both target_rec and rank_constraint are omitted,
                            untargeted recommendation change is used.

        Returns:
            cf_seq: counterfactual sequence, or None if no valid CF found
            info: dict with metadata (edit_distance, queries_used, etc.)
        """
        raise NotImplementedError


def hamming_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Compute edit distance (Hamming) between two sequences.
    Only counts non-padding positions that differ.
    """
    assert len(seq1) == len(seq2)
    dist = 0
    for a, b in zip(seq1, seq2):
        if a != 0 and b != 0 and a != b:
            dist += 1
        elif (a == 0) != (b == 0):
            dist += 1
    return dist


def get_edited_positions(seq1: List[int], seq2: List[int]) -> List[int]:
    """Get list of positions that differ between two sequences."""
    positions = []
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            positions.append(i)
    return positions
