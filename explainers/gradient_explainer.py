"""
Gradient-guided Counterfactual Explanation for differentiable SRS.

Uses gradients of the target item score w.r.t. item embeddings to find
which position to edit and which direction to move in embedding space,
then projects to the nearest real item.
"""

import random
import warnings
import heapq
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as F

from .base import BaseExplainer, RankConstraint, RankConstraintSummary, hamming_distance


@dataclass(slots=True)
class BeamNode:
    seq_key: Tuple[int, ...]
    seq: Tuple[int, ...]
    edited_bits: int
    summary: RankConstraintSummary
    embed_state: Dict[str, Any]


@dataclass(slots=True)
class CandidateNode:
    seq: Tuple[int, ...]
    parent_state: Dict[str, Any]
    edited_bits: int
    pos: int
    cand: int
    surrogate_score: float


class GradientExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        num_items: int,
        alpha: float = 1.0,
        alpha_search_steps: int = 5,
        top_k_positions: int = 3,
        n_candidates_per_position: int = 5,
        position_selection: str = 'gradient',
        item_selection: str = 'cosine',
        alpha_search: str = 'multi',
        objective_variant: str = 'legacy',
        rank_loss_tau: float = 1.0,
        rank_loss_softplus_tau: float = 1.0,
        rank_loss_margin: float = 0.0,
        rank_boundary_band: int = 2,
        rank_anchor_weight: float = 0.1,
        use_blocker_loss: bool = True,
        blocker_variant: str = 'hard',
        blocker_top_m: int = 10,
        blocker_tau: float = 1.0,
        blocker_weight_tau: float = 1.0,
        blocker_weight: float = 1.0,
        use_hotflip: bool = True,
        beam_width: int = 1,
        beam_rerank_factor: int = 1,
        context_candidate_top_m: int = 0,
        hotflip_distance_weight: float = 0.01,
        hotflip_rss_weight: float = 0.0,
        hotflip_context_weight: float = 0.0,
        local_cosine_top_m: int = 0,
        eval_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(model, num_items, **kwargs)
        self.alpha = alpha
        self.top_k_positions = top_k_positions
        self.n_candidates_per_position = n_candidates_per_position
        self.position_selection = position_selection
        self.item_selection = item_selection
        self.alpha_search = alpha_search
        self.objective_variant = str(objective_variant)
        self.rank_loss_tau = float(rank_loss_tau)
        self.rank_loss_softplus_tau = float(rank_loss_softplus_tau)
        self.rank_loss_margin = float(rank_loss_margin)
        self.rank_boundary_band = max(1, int(rank_boundary_band))
        self.rank_anchor_weight = float(rank_anchor_weight)
        self.use_blocker_loss = use_blocker_loss
        self.blocker_variant = str(blocker_variant)
        self.blocker_top_m = max(1, int(blocker_top_m))
        self.blocker_tau = float(blocker_tau)
        self.blocker_weight_tau = float(blocker_weight_tau)
        self.blocker_weight = float(blocker_weight)
        self.use_hotflip = use_hotflip
        self.beam_width = beam_width
        self.beam_rerank_factor = int(beam_rerank_factor)
        self.context_candidate_top_m = max(0, int(context_candidate_top_m))
        self.hotflip_distance_weight = hotflip_distance_weight
        self.hotflip_rss_weight = hotflip_rss_weight
        self.hotflip_context_weight = hotflip_context_weight
        self.local_cosine_top_m = max(0, int(local_cosine_top_m))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.alpha_search_steps = alpha_search_steps
        if alpha_search_steps != 5:
            warnings.warn(
                "GradientExplainer.alpha_search_steps is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def _is_bert(self) -> bool:
        return hasattr(self.model, 'output_proj')

    @property
    def _is_gru(self) -> bool:
        return hasattr(self.model, 'gru')

    def _scale_emb(self, emb: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'hidden_dim') and hasattr(self.model, 'pos_emb') and not self._is_bert:
            return emb * (self.model.hidden_dim ** 0.5)
        return emb

    def _build_runtime_cache(self, original_seq: List[int]) -> Dict[str, Any]:
        seq_len = len(original_seq)
        item_emb_table = self.model.item_emb.weight[1:self.num_items + 1]
        scaled_item_emb_table = self._scale_emb(item_emb_table.unsqueeze(0))[0]
        scaled_item_emb_norm_sq = scaled_item_emb_table.pow(2).sum(dim=-1)
        normalized_scaled_item_emb_table = F.normalize(scaled_item_emb_table, dim=-1)
        position_emb_table = None
        if hasattr(self.model, 'pos_emb'):
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
            position_emb_table = self.model.pos_emb(positions)

        editable_positions_mask = torch.tensor(
            [item != 0 for item in original_seq],
            dtype=torch.bool,
            device=self.device,
        )

        cache = {
            'seq_len': seq_len,
            'item_emb_table': item_emb_table,
            'scaled_item_emb_table': scaled_item_emb_table,
            'scaled_item_emb_norm_sq': scaled_item_emb_norm_sq,
            'normalized_scaled_item_emb_table': normalized_scaled_item_emb_table,
            'position_emb_table': position_emb_table,
            'editable_positions_mask': editable_positions_mask,
            'edited_mask_cache': {0: torch.zeros(seq_len, dtype=torch.bool, device=self.device)},
        }
        cache['base_state'] = self._build_embed_state(original_seq, cache)
        return cache

    def _edited_bits_with_pos(self, edited_bits: int, pos: int) -> int:
        return edited_bits | (1 << pos)

    def _edited_mask_from_bits(self, edited_bits: int, runtime_cache: Dict[str, Any]) -> torch.Tensor:
        edited_mask_cache = runtime_cache['edited_mask_cache']
        cached_mask = edited_mask_cache.get(edited_bits)
        if cached_mask is not None:
            return cached_mask

        mask = torch.zeros(runtime_cache['seq_len'], dtype=torch.bool, device=self.device)
        bitset = edited_bits
        while bitset:
            lsb = bitset & -bitset
            pos = lsb.bit_length() - 1
            mask[pos] = True
            bitset ^= lsb
        edited_mask_cache[edited_bits] = mask
        return mask

    def _mask_importance_with_bits(
        self,
        importance: torch.Tensor,
        edited_bits: int,
        runtime_cache: Dict[str, Any],
    ) -> torch.Tensor:
        if edited_bits == 0:
            return importance
        return importance.masked_fill(self._edited_mask_from_bits(edited_bits, runtime_cache), 0.0)

    def _build_model_input_tensor(self, seq: Sequence[int]) -> torch.Tensor:
        seq_tensor = torch.as_tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
        if not self._is_bert:
            return seq_tensor
        masked_seq = torch.zeros_like(seq_tensor)
        masked_seq[:, :-1] = seq_tensor[:, 1:]
        masked_seq[:, -1] = self.model.mask_token
        return masked_seq

    def _compute_seq_emb_from_input(self, input_seq_tensor: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        seq_emb = self.model.item_emb(input_seq_tensor)
        seq_emb = self._scale_emb(seq_emb)
        if cache['position_emb_table'] is not None:
            seq_emb = seq_emb + cache['position_emb_table']
        return seq_emb

    def _seq_key(self, seq: Sequence[int]) -> Tuple[int, ...]:
        if isinstance(seq, tuple):
            return seq
        return tuple(seq)

    def _replace_seq_item(
        self,
        seq: Tuple[int, ...],
        edit_pos: int,
        new_item: int,
    ) -> Tuple[int, ...]:
        return seq[:edit_pos] + (new_item,) + seq[edit_pos + 1:]

    def _build_embed_state(self, seq: Sequence[int], cache: Dict[str, Any]) -> Dict[str, Any]:
        seq_key = self._seq_key(seq)
        input_seq_tensor = self._build_model_input_tensor(seq_key)
        seq_emb = self._compute_seq_emb_from_input(input_seq_tensor, cache)
        return {
            'seq': seq_key,
            'seq_key': seq_key,
            'original_seq_tensor': torch.as_tensor(seq_key, dtype=torch.long, device=self.device).unsqueeze(0),
            'input_seq_tensor': input_seq_tensor,
            'seq_emb': seq_emb,
        }

    def _normalize_context_scores(self, current_scores: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.hotflip_context_weight <= 0.0 or current_scores is None:
            return None
        context_scores = current_scores[1:self.num_items + 1]
        return (context_scores - context_scores.mean()) / (
            context_scores.std(unbiased=False) + 1e-8
        )

    def _context_candidate_items(
        self,
        current_scores: torch.Tensor,
        exclude_items: set,
        current_item: int,
        *,
        limit: int,
        preferred_item: Optional[int] = None,
    ) -> List[int]:
        if limit <= 0:
            return []
        masked = current_scores.detach().clone().float()
        masked[0] = -float('inf')
        for item in exclude_items:
            if 0 < item <= self.num_items:
                masked[item] = -float('inf')
        if 0 < current_item <= self.num_items:
            masked[current_item] = -float('inf')

        candidates: List[int] = []
        seen = set()
        if (
            preferred_item is not None
            and 0 < preferred_item <= self.num_items
            and preferred_item not in exclude_items
            and preferred_item != current_item
        ):
            candidates.append(int(preferred_item))
            seen.add(int(preferred_item))

        top_k = min(self.num_items, limit + len(exclude_items) + 1)
        _, top_idx = masked.topk(top_k)
        for item in top_idx.tolist():
            if item <= 0 or item == current_item or item in exclude_items or item in seen:
                continue
            seen.add(int(item))
            candidates.append(int(item))
            if len(candidates) >= limit:
                break
        return candidates

    def _embed_sequence_from_parent(
        self,
        parent_state: Dict[str, Any],
        edit_pos: int,
        new_item: int,
        cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        parent_seq = parent_state['seq']
        old_item = parent_seq[edit_pos]
        if old_item == new_item:
            return parent_state

        new_seq = self._replace_seq_item(parent_seq, edit_pos, new_item)
        original_seq_tensor = parent_state['original_seq_tensor'].clone()
        original_seq_tensor[0, edit_pos] = new_item
        input_seq_tensor = parent_state['input_seq_tensor'].clone()
        seq_emb = parent_state['seq_emb'].clone()

        if self._is_bert:
            if edit_pos > 0:
                input_idx = edit_pos - 1
                input_seq_tensor[0, input_idx] = new_item
                updated = self.model.item_emb(input_seq_tensor[:, input_idx:input_idx + 1])
                updated = self._scale_emb(updated)
                if cache['position_emb_table'] is not None:
                    updated = updated + cache['position_emb_table'][:, input_idx:input_idx + 1, :]
                seq_emb[:, input_idx:input_idx + 1, :] = updated
        else:
            input_seq_tensor[0, edit_pos] = new_item
            updated = self.model.item_emb(input_seq_tensor[:, edit_pos:edit_pos + 1])
            updated = self._scale_emb(updated)
            if cache['position_emb_table'] is not None:
                updated = updated + cache['position_emb_table'][:, edit_pos:edit_pos + 1, :]
            seq_emb[:, edit_pos:edit_pos + 1, :] = updated

        return {
            'seq': new_seq,
            'seq_key': new_seq,
            'original_seq_tensor': original_seq_tensor,
            'input_seq_tensor': input_seq_tensor,
            'seq_emb': seq_emb,
        }

    def _embed_sequences_from_parents_batch(
        self,
        parent_states: List[Dict[str, Any]],
        seq_keys: List[Tuple[int, ...]],
        edit_positions: List[int],
        new_items: List[int],
        cache: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not parent_states:
            return []

        original_seq_tensor_batch = torch.cat([state['original_seq_tensor'] for state in parent_states], dim=0)
        input_seq_tensor_batch = torch.cat([state['input_seq_tensor'] for state in parent_states], dim=0)
        seq_emb_batch = torch.cat([state['seq_emb'] for state in parent_states], dim=0)

        row_idx = torch.arange(len(parent_states), device=self.device)
        edit_pos_tensor = torch.tensor(edit_positions, dtype=torch.long, device=self.device)
        new_item_tensor = torch.tensor(new_items, dtype=torch.long, device=self.device)

        original_seq_tensor_batch[row_idx, edit_pos_tensor] = new_item_tensor

        if self._is_bert:
            affected_mask = edit_pos_tensor > 0
            if affected_mask.any():
                affected_rows = row_idx[affected_mask]
                affected_idx = edit_pos_tensor[affected_mask] - 1
                input_seq_tensor_batch[affected_rows, affected_idx] = new_item_tensor[affected_mask]
                replacement_items = input_seq_tensor_batch[affected_rows, affected_idx].unsqueeze(1)
                updated = self.model.item_emb(replacement_items)
                updated = self._scale_emb(updated)
                if cache['position_emb_table'] is not None:
                    updated = updated + cache['position_emb_table'][0, affected_idx, :].unsqueeze(1)
                seq_emb_batch[affected_rows, affected_idx, :] = updated[:, 0, :]
        else:
            input_seq_tensor_batch[row_idx, edit_pos_tensor] = new_item_tensor
            replacement_items = input_seq_tensor_batch[row_idx, edit_pos_tensor].unsqueeze(1)
            updated = self.model.item_emb(replacement_items)
            updated = self._scale_emb(updated)
            if cache['position_emb_table'] is not None:
                updated = updated + cache['position_emb_table'][0, edit_pos_tensor, :].unsqueeze(1)
            seq_emb_batch[row_idx, edit_pos_tensor, :] = updated[:, 0, :]

        return [
            {
                'seq': seq_key,
                'seq_key': seq_key,
                'original_seq_tensor': original_seq_tensor_batch[i:i + 1],
                'input_seq_tensor': input_seq_tensor_batch[i:i + 1],
                'seq_emb': seq_emb_batch[i:i + 1],
            }
            for i, seq_key in enumerate(seq_keys)
        ]

    @contextmanager
    def _gradient_model_mode(self):
        was_training = self.model.training
        if self._is_gru:
            self.model.train()
        try:
            yield
        finally:
            if self._is_gru and not was_training:
                self.model.eval()

    def _zero_objective(self, scores: torch.Tensor, fallback_item: int) -> torch.Tensor:
        return scores[fallback_item] * 0.0

    def _soft_blocker_objective(
        self,
        diffs: torch.Tensor,
    ) -> torch.Tensor:
        if diffs.numel() == 0:
            return diffs.sum() * 0.0
        top_m = min(self.blocker_top_m, diffs.numel())
        top_diffs = diffs.topk(top_m).values
        tau = max(1.0e-6, self.blocker_tau)
        weight_tau = max(1.0e-6, self.blocker_weight_tau)
        weights = torch.sigmoid(top_diffs / weight_tau)
        weighted_diffs = weights * top_diffs
        return -self.blocker_weight * tau * torch.logsumexp(weighted_diffs / tau, dim=0)

    def _soft_rank_blocker_count(
        self,
        scores: torch.Tensor,
        item: int,
        input_seq: Sequence[int],
    ) -> torch.Tensor:
        item_score = scores[item]
        masked_scores = scores.detach().clone()
        self.mask_scores_inplace(masked_scores, cast(List[int] | Tuple[int, ...], input_seq))
        valid_mask = torch.isfinite(masked_scores)
        valid_mask[item] = False
        if not bool(valid_mask.any().item()):
            return item_score * 0.0
        tau = max(1.0e-6, self.rank_loss_tau)
        diffs = (scores[valid_mask] - item_score) / tau
        return torch.sigmoid(diffs).sum()

    def _soft_rank_objective(
        self,
        scores: torch.Tensor,
        input_seq: Sequence[int],
        constraint: RankConstraint,
    ) -> torch.Tensor:
        soft_blockers = self._soft_rank_blocker_count(scores, constraint.item, input_seq)
        softplus_tau = max(1.0e-6, self.rank_loss_softplus_tau)
        margin = self.rank_loss_margin
        if constraint.kind == 'bring_in':
            target_blockers = float(max(0, constraint.top_k - 1))
            violation = soft_blockers - target_blockers - margin
            return -F.softplus(violation / softplus_tau)
        required_blockers = float(constraint.top_k)
        violation = required_blockers - soft_blockers - margin
        return -F.softplus(violation / softplus_tau)

    def _boundary_rank_threshold(
        self,
        scores: torch.Tensor,
        input_seq: Sequence[int],
        top_k: int,
    ) -> torch.Tensor:
        masked_scores = scores.detach().clone()
        self.mask_scores_inplace(masked_scores, cast(List[int] | Tuple[int, ...], input_seq))
        finite_scores = scores[torch.isfinite(masked_scores)]
        num_finite = finite_scores.numel()
        if num_finite == 0:
            return scores.new_zeros(())

        boundary_idx = min(max(0, top_k - 1), num_finite - 1)
        band = self.rank_boundary_band
        end = min(num_finite, boundary_idx + band + 1)
        sorted_scores = finite_scores.topk(end).values
        start = max(0, boundary_idx - band)
        boundary_scores = sorted_scores[start:end]
        if boundary_scores.numel() == 1:
            return boundary_scores[0]

        tau = max(1.0e-6, self.rank_loss_tau)
        center = boundary_scores[boundary_idx - start]
        weights = torch.softmax(-(boundary_scores - center).abs() / tau, dim=0)
        return (weights * boundary_scores).sum()

    def _boundary_rank_objective(
        self,
        scores: torch.Tensor,
        input_seq: Sequence[int],
        constraint: RankConstraint,
    ) -> torch.Tensor:
        item_score = scores[constraint.item]
        threshold = self._boundary_rank_threshold(scores, input_seq, constraint.top_k)
        tau = max(1.0e-6, self.rank_loss_softplus_tau)
        margin = self.rank_loss_margin
        anchor = self.rank_anchor_weight * item_score
        if constraint.kind == 'bring_in':
            violation = threshold - item_score + margin
            return -(F.softplus(violation / tau) + anchor)
        violation = item_score - threshold + margin
        return -(F.softplus(violation / tau) - anchor)

    def _constraint_objective(
        self,
        scores: torch.Tensor,
        input_seq: Sequence[int],
        original_rec: int,
        constraint: Optional[RankConstraint],
    ) -> torch.Tensor:
        if constraint is None:
            return -scores[original_rec]

        if self.objective_variant == 'soft_rank':
            return self._soft_rank_objective(scores, input_seq, constraint)
        if self.objective_variant == 'boundary_rank':
            return self._boundary_rank_objective(scores, input_seq, constraint)

        item = constraint.item
        item_score = scores[item]
        masked_scores = scores.detach().clone()
        self.mask_scores_inplace(masked_scores, cast(List[int] | Tuple[int, ...], input_seq))

        if constraint.kind == 'bring_in':
            if not self.use_blocker_loss:
                return item_score

            competitor_scores = scores[masked_scores > item_score.detach()]
            allowed_above = max(0, constraint.top_k - 1)
            active_count = competitor_scores.numel() - allowed_above
            if active_count <= 0:
                return self._zero_objective(scores, item)

            sorted_competitors = competitor_scores.topk(competitor_scores.numel()).values
            active_blockers = sorted_competitors[allowed_above:]
            blocker_diffs = active_blockers - item_score
            if self.blocker_variant == 'soft':
                return self._soft_blocker_objective(blocker_diffs)
            tau = 1.0
            return -tau * torch.logsumexp(blocker_diffs / tau, dim=0)

        if not self.use_blocker_loss:
            return -item_score

        higher_count = int((masked_scores > item_score.detach()).sum().item())
        challengers_needed = constraint.top_k - higher_count
        if challengers_needed <= 0:
            return self._zero_objective(scores, item)

        challenger_mask = masked_scores <= item_score.detach()
        challenger_mask[item] = False
        challenger_scores = scores[challenger_mask]
        if challenger_scores.numel() == 0:
            return self._zero_objective(scores, item)

        active_challengers = challenger_scores.topk(min(challengers_needed, challenger_scores.numel())).values
        challenger_diffs = active_challengers - item_score
        if self.blocker_variant == 'soft':
            return -self._soft_blocker_objective(challenger_diffs)
        tau = 1.0
        return tau * torch.logsumexp(challenger_diffs / tau, dim=0)

    def _compute_gradient(
        self,
        embed_state: Dict[str, Any],
        original_rec: int,
        runtime_cache: Dict[str, Any],
        constraint: Optional[RankConstraint] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_emb = embed_state['seq_emb'].detach().requires_grad_(True)
        input_seq = embed_state['seq']
        input_seq_tensor = embed_state['input_seq_tensor']

        with torch.enable_grad():
            if input_seq_tensor is not None:
                scores = self.model.forward_from_emb(seq_emb, input_seq_tensor)
            else:
                scores = self.model.forward_from_emb(seq_emb)

            objective = self._constraint_objective(scores[0], input_seq, original_rec, constraint)
            objective.backward()

        grad = seq_emb.grad
        if grad is None:
            raise RuntimeError('Gradient is None in GradientExplainer._compute_gradient')

        importance = grad[0].norm(dim=-1)
        importance = importance.masked_fill(~runtime_cache['editable_positions_mask'], 0.0)
        self.query_count += 1
        return importance, grad, scores[0].detach()

    def _compute_gradient_batch(
        self,
        embed_states: List[Dict[str, Any]],
        original_rec: int,
        runtime_cache: Dict[str, Any],
        constraint: Optional[RankConstraint] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not embed_states:
            raise ValueError('embed_states must be non-empty')

        seq_emb_batch = torch.cat([state['seq_emb'] for state in embed_states], dim=0).detach().requires_grad_(True)
        input_seq_tensor_batch = torch.cat([state['input_seq_tensor'] for state in embed_states], dim=0)

        with torch.enable_grad():
            scores = self.model.forward_from_emb(seq_emb_batch, input_seq_tensor_batch)

            objectives = [
                self._constraint_objective(scores[i], embed_states[i]['seq'], original_rec, constraint)
                for i in range(len(embed_states))
            ]
            torch.stack(objectives).sum().backward()

        grad = seq_emb_batch.grad
        if grad is None:
            raise RuntimeError('Gradient is None in GradientExplainer._compute_gradient_batch')

        importance = grad.norm(dim=-1)
        importance = importance.masked_fill(~runtime_cache['editable_positions_mask'].unsqueeze(0), 0.0)
        self.query_count += len(embed_states)
        return importance, grad, scores.detach()

    def _perturb_and_project(
        self,
        input_seq: Sequence[int],
        position: int,
        grad: torch.Tensor,
        alpha: float,
        exclude_items: set,
        runtime_cache: Dict[str, Any],
        item_selection: str = 'cosine',
        constraint: Optional[RankConstraint] = None,
        n_candidates: Optional[int] = None,
        context_scores: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> List[int] | List[Tuple[int, float]]:
        n_top = n_candidates or self.n_candidates_per_position
        if item_selection == 'random':
            available = [i for i in range(1, self.num_items + 1) if i not in exclude_items]
            random.shuffle(available)
            random_items = [int(i) for i in available[:n_top]]
            if return_scores:
                return [(item, float(n_top - idx)) for idx, item in enumerate(random_items)]
            return random_items

        all_embs_scaled = runtime_cache['scaled_item_emb_table']
        all_embs_norm_sq = runtime_cache['scaled_item_emb_norm_sq']
        normalized_all_embs_scaled = runtime_cache['normalized_scaled_item_emb_table']
        original_emb_scaled = all_embs_scaled[input_seq[position] - 1]
        use_hotflip = self.use_hotflip and constraint is not None
        local_mask = None
        if self.local_cosine_top_m > 0:
            normalized_original_emb = F.normalize(original_emb_scaled.unsqueeze(0), dim=-1)[0]
            local_k = min(self.num_items, max(1, self.local_cosine_top_m + 1))
            local_indices = (normalized_all_embs_scaled @ normalized_original_emb).topk(local_k).indices
            local_mask = torch.ones(self.num_items, dtype=torch.bool, device=self.device)
            local_mask[local_indices] = False

        if use_hotflip:
            g_p = grad[0, position]
            emb_dot_grad = all_embs_scaled @ g_p
            orig_dot_grad = torch.dot(original_emb_scaled, g_p)
            emb_dot_orig = all_embs_scaled @ original_emb_scaled
            orig_norm_sq = original_emb_scaled.dot(original_emb_scaled)
            taylor_gain = emb_dot_grad - orig_dot_grad
            scaled_taylor_gain = alpha * taylor_gain
            distance_penalty = self.hotflip_distance_weight * (
                all_embs_norm_sq + orig_norm_sq - (2.0 * emb_dot_orig)
            )
            score = scaled_taylor_gain - distance_penalty

            if self.hotflip_rss_weight > 0.0:
                normalized_original_emb = F.normalize(original_emb_scaled.unsqueeze(0), dim=-1)[0]
                rss_penalty = self.hotflip_rss_weight * (
                    1.0 - (normalized_all_embs_scaled @ normalized_original_emb)
                )
                score = score - rss_penalty

            if context_scores is not None:
                score = score + (self.hotflip_context_weight * context_scores)
        else:
            perturb_dir = grad[0, position]
            perturb_dir = perturb_dir / (perturb_dir.norm() + 1e-8)
            perturbed_emb = original_emb_scaled + alpha * perturb_dir
            normalized_perturbed_emb = F.normalize(perturbed_emb.unsqueeze(0), dim=-1)[0]
            score = normalized_all_embs_scaled @ normalized_perturbed_emb

        if local_mask is not None:
            score = score.masked_fill(local_mask, -float('inf'))

        default_exclude_items = runtime_cache.get('default_exclude_items')
        default_exclude_indices = runtime_cache.get('default_exclude_item_indices')
        if exclude_items is default_exclude_items and default_exclude_indices is not None and default_exclude_indices.numel() > 0:
            score[default_exclude_indices] = -float('inf')
        else:
            for item in exclude_items:
                if 0 < item <= self.num_items:
                    score[item - 1] = -float('inf')

        k = min(n_top, self.num_items)
        top_scores, top_indices = score.topk(k)
        top_indices_list = top_indices.tolist()
        if return_scores:
            top_scores_list = top_scores.tolist()
            return [
                (idx + 1, float(candidate_score))
                for idx, candidate_score in zip(top_indices_list, top_scores_list)
            ]
        return [idx + 1 for idx in top_indices_list]

    def _perturb_and_project_positions(
        self,
        input_seq: Tuple[int, ...],
        positions: torch.Tensor,
        grad: torch.Tensor,
        alpha: float,
        exclude_items: set,
        runtime_cache: Dict[str, Any],
        item_selection: str = 'cosine',
        constraint: Optional[RankConstraint] = None,
        n_candidates: Optional[int] = None,
        context_scores: Optional[torch.Tensor] = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        position_list = positions.tolist()
        if not position_list:
            return {}

        if item_selection == 'random':
            return {
                pos: cast(
                    List[Tuple[int, float]],
                    self._perturb_and_project(
                        input_seq,
                        pos,
                        grad,
                        alpha,
                        exclude_items=exclude_items,
                        runtime_cache=runtime_cache,
                        item_selection=item_selection,
                        constraint=constraint,
                        n_candidates=n_candidates,
                        context_scores=context_scores,
                        return_scores=True,
                    ),
                )
                for pos in position_list
            }

        n_top = n_candidates or self.n_candidates_per_position
        all_embs_scaled = runtime_cache['scaled_item_emb_table']
        all_embs_norm_sq = runtime_cache['scaled_item_emb_norm_sq']
        normalized_all_embs_scaled = runtime_cache['normalized_scaled_item_emb_table']
        pos_tensor = positions.to(device=self.device, dtype=torch.long)
        original_items = torch.tensor([input_seq[pos] for pos in position_list], device=self.device, dtype=torch.long)
        original_embs_scaled = all_embs_scaled[original_items - 1]
        use_hotflip = self.use_hotflip and constraint is not None
        if self.local_cosine_top_m > 0:
            normalized_original_embs = F.normalize(original_embs_scaled, dim=-1)
            local_k = min(self.num_items, max(1, self.local_cosine_top_m + 1))
            local_indices = (normalized_original_embs @ normalized_all_embs_scaled.T).topk(local_k, dim=1).indices
            local_mask = torch.ones((len(position_list), self.num_items), dtype=torch.bool, device=self.device)
            local_mask.scatter_(1, local_indices, False)
        else:
            normalized_original_embs = None
            local_mask = None

        if use_hotflip:
            grad_pos = grad[0, pos_tensor]
            emb_dot_grad = grad_pos @ all_embs_scaled.T
            orig_dot_grad = (original_embs_scaled * grad_pos).sum(dim=1, keepdim=True)
            emb_dot_orig = original_embs_scaled @ all_embs_scaled.T
            orig_norm_sq = original_embs_scaled.pow(2).sum(dim=1, keepdim=True)
            taylor_gain = emb_dot_grad - orig_dot_grad
            score = alpha * taylor_gain
            score = score - self.hotflip_distance_weight * (
                all_embs_norm_sq.unsqueeze(0) + orig_norm_sq - (2.0 * emb_dot_orig)
            )

            if self.hotflip_rss_weight > 0.0:
                if normalized_original_embs is None:
                    normalized_original_embs = F.normalize(original_embs_scaled, dim=-1)
                rss_penalty = self.hotflip_rss_weight * (
                    1.0 - (normalized_original_embs @ normalized_all_embs_scaled.T)
                )
                score = score - rss_penalty

            if context_scores is not None:
                score = score + (self.hotflip_context_weight * context_scores.unsqueeze(0))
        else:
            grad_pos = grad[0, pos_tensor]
            perturb_dir = grad_pos
            perturb_dir = perturb_dir / (perturb_dir.norm(dim=1, keepdim=True) + 1e-8)
            perturbed_embs = original_embs_scaled + alpha * perturb_dir
            normalized_perturbed_embs = F.normalize(perturbed_embs, dim=-1)
            score = normalized_perturbed_embs @ normalized_all_embs_scaled.T

        if local_mask is not None:
            score = score.masked_fill(local_mask, -float('inf'))

        default_exclude_items = runtime_cache.get('default_exclude_items')
        default_exclude_indices = runtime_cache.get('default_exclude_item_indices')
        if exclude_items is default_exclude_items and default_exclude_indices is not None and default_exclude_indices.numel() > 0:
            score[:, default_exclude_indices] = -float('inf')
        else:
            exclude_indices = [item - 1 for item in exclude_items if 0 < item <= self.num_items]
            if exclude_indices:
                exclude_tensor = torch.tensor(exclude_indices, device=self.device, dtype=torch.long)
                score[:, exclude_tensor] = -float('inf')

        k = min(n_top, self.num_items)
        top_scores, top_indices = score.topk(k, dim=1)
        top_indices_rows = top_indices.tolist()
        top_scores_rows = top_scores.tolist()
        return {
            pos: [
                (item_idx + 1, float(item_score))
                for item_idx, item_score in zip(row_indices, row_scores)
            ]
            for pos, row_indices, row_scores in zip(position_list, top_indices_rows, top_scores_rows)
        }

    def _target_rank(self, scores: torch.Tensor, target: int) -> int:
        return int((scores > scores[target]).sum().item()) + 1

    def _score_sequence_chunks(
        self,
        seqs: List[Tuple[int, ...]] | List[List[int]],
    ) -> Tuple[List[Tuple[int, ...] | List[int]], Optional[torch.Tensor]]:
        if not seqs:
            return [], None

        budget_now = self.budget_remaining()
        if budget_now <= 0:
            return [], None

        eval_seqs = list(seqs[:budget_now])
        batch_size = min(self.eval_batch_size, len(eval_seqs))
        score_chunks: List[torch.Tensor] = []

        for start in range(0, len(eval_seqs), batch_size):
            chunk_seqs = eval_seqs[start:start + batch_size]
            chunk_tensor = torch.as_tensor(chunk_seqs, dtype=torch.long, device=self.device)
            with self._amp_ctx():
                chunk_scores = self.model.predict(chunk_tensor)
            chunk_scores = chunk_scores.float()
            self.query_count += len(chunk_seqs)
            self.mask_scores_batch_inplace(chunk_scores, chunk_tensor)
            score_chunks.append(chunk_scores)

        if not score_chunks:
            return [], None
        return eval_seqs, torch.cat(score_chunks, dim=0)

    def _populate_cache_batch(self, uncached_keys: List[Tuple[int, ...]], cache: dict) -> None:
        eval_seqs, batch_scores = self._score_sequence_chunks(uncached_keys)
        if batch_scores is None:
            return
        for i, key in enumerate(uncached_keys[:len(eval_seqs)]):
            scores = batch_scores[i]
            cache[key] = scores

    def _score_candidates_batch(
        self,
        candidate_seqs: List[Tuple[int, ...]] | List[List[int]],
    ) -> Tuple[List[Tuple[int, ...] | List[int]], Optional[torch.Tensor]]:
        return self._score_sequence_chunks(candidate_seqs)

    def _get_scores_cached(self, seq: List[int], cache: dict) -> torch.Tensor:
        key = self._seq_key(seq)
        if key not in cache:
            scores = self.predict_scores_no_count(seq)
            self.query_count += 1
            self.mask_scores_inplace(scores, seq)
            cache[key] = scores
        return cache[key]

    def _constraint_summary_from_scores(
        self,
        scores: torch.Tensor,
        constraint: RankConstraint,
    ) -> RankConstraintSummary:
        return constraint.summarize(scores)

    def _best_surrogate_candidate(
        self,
        seq: Tuple[int, ...],
        importance: torch.Tensor,
        grad: torch.Tensor,
        user_items: set,
        runtime_cache: Dict[str, Any],
        constraint: RankConstraint,
        normalized_context_scores: Optional[torch.Tensor],
        edited_mask: torch.Tensor,
    ) -> Optional[Tuple[Tuple[int, ...], int]]:
        positive = int((importance > 0).sum().item())
        if positive <= 0:
            return None

        k = max(1, min(self.top_k_positions, positive))
        if self.position_selection == 'random':
            non_pad = torch.nonzero(importance > 0, as_tuple=False).view(-1).tolist()
            sampled = random.sample(non_pad, min(k, len(non_pad)))
            top_positions = torch.tensor(sampled, device=self.device)
        else:
            _, top_positions = importance.topk(k)

        best: Optional[Tuple[float, Tuple[int, ...], int]] = None
        n_cand = max(self.n_candidates_per_position, 10)
        position_candidates = self._perturb_and_project_positions(
            seq,
            top_positions,
            grad,
            self.alpha,
            exclude_items=user_items,
            runtime_cache=runtime_cache,
            item_selection=self.item_selection,
            constraint=constraint,
            n_candidates=n_cand,
            context_scores=normalized_context_scores,
        )
        for pos in top_positions.tolist():
            if edited_mask[pos]:
                continue
            candidates = position_candidates.get(pos, [])
            for cand, surrogate_score in candidates:
                if cand == seq[pos]:
                    continue
                next_seq = self._replace_seq_item(seq, pos, cand)
                if best is None or surrogate_score > best[0]:
                    best = (surrogate_score, next_seq, pos)
        if best is None:
            return None
        return best[1], best[2]

    def _populate_constraint_cache_batch(
        self,
        uncached_keys: List[Tuple[int, ...]],
        cache: dict,
        constraint: RankConstraint,
    ) -> None:
        eval_seqs, batch_scores = self._score_sequence_chunks(uncached_keys)
        if batch_scores is None:
            return
        for i, key in enumerate(uncached_keys[:len(eval_seqs)]):
            scores = batch_scores[i]
            cache[key] = self._constraint_summary_from_scores(scores, constraint)

    def _get_constraint_summary_cached(
        self,
        seq: List[int],
        cache: dict,
        constraint: RankConstraint,
    ) -> RankConstraintSummary:
        key = self._seq_key(seq)
        if key not in cache:
            scores = self.predict_scores_no_count(seq)
            self.query_count += 1
            self.mask_scores_inplace(scores, seq)
            cache[key] = self._constraint_summary_from_scores(scores, constraint)
        return cache[key]

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
        constraint = self.resolve_rank_constraint(target_rec=target_rec, rank_constraint=rank_constraint)
        runtime_cache = self._build_runtime_cache(original_seq)
        base_state = runtime_cache['base_state']
        user_items = candidate_items if isinstance(candidate_items, set) else set(original_seq)
        runtime_cache['default_exclude_items'] = user_items
        runtime_cache['default_exclude_item_indices'] = torch.tensor(
            [item - 1 for item in sorted(user_items) if 0 < item <= self.num_items],
            device=self.device,
            dtype=torch.long,
        )

        if constraint is not None and self.beam_width > 1:
            return self._explain_beam(original_seq, original_rec, user_items, constraint, runtime_cache)

        current_state = base_state
        best_cf = None
        best_edits = self.max_edits + 1
        edited_mask = torch.zeros(runtime_cache['seq_len'], dtype=torch.bool, device=self.device)

        with self._gradient_model_mode():
            for step in range(self.max_edits):
                importance, grad, current_scores = self._compute_gradient(
                    current_state, original_rec, runtime_cache, constraint=constraint
                )
                if constraint is not None and constraint.is_satisfied(current_scores):
                    ed = hamming_distance(original_seq, list(current_state['seq']))
                    return list(current_state['seq']), {
                        'method': 'gradient_hotflip_only',
                        'valid': True,
                        'edit_distance': ed,
                        'queries_used': self.query_count,
                        'edits': int(edited_mask.sum().item()),
                        'target_rec': target_rec,
                        'rank_constraint': constraint,
                        'top_k_cfs': [{
                            'seq': list(current_state['seq']),
                            'edit_distance': ed,
                            'valid': True,
                        }],
                    }
                importance = importance.masked_fill(edited_mask, 0.0)
                normalized_context_scores = self._normalize_context_scores(current_scores)

                positive = int((importance > 0).sum().item())
                if positive <= 0:
                    break

                if constraint is not None:
                    best_candidate = self._best_surrogate_candidate(
                        current_state['seq'],
                        importance,
                        grad,
                        user_items,
                        runtime_cache,
                        constraint,
                        normalized_context_scores,
                        edited_mask,
                    )
                    if best_candidate is None:
                        break
                    next_seq, pos = best_candidate
                    current_state = self._embed_sequence_from_parent(
                        current_state,
                        pos,
                        next_seq[pos],
                        runtime_cache,
                    )
                    edited_mask[pos] = True
                    continue

                k = max(1, min(self.top_k_positions, positive))
                if self.position_selection == 'random':
                    non_pad = torch.nonzero(importance > 0, as_tuple=False).view(-1).tolist()
                    sampled = random.sample(non_pad, min(k, len(non_pad)))
                    top_positions = torch.tensor(sampled, device=self.device)
                    position_candidates = None
                else:
                    _, top_positions = importance.topk(k)
                    position_candidates = None

                found_cf = False
                for pos in top_positions.tolist():
                    if edited_mask[pos]:
                        continue

                    if constraint is not None:
                        alpha_scales = [1.0]
                        n_cand = max(self.n_candidates_per_position, 10)
                    else:
                        alpha_scales = [1.0] if self.alpha_search == 'fixed' else [0.5, 1.0, 2.0, 5.0]
                        n_cand = self.n_candidates_per_position

                    candidate_seq_by_key: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
                    for alpha_scale in alpha_scales:
                        alpha = self.alpha * alpha_scale
                        candidates = cast(
                            List[int],
                            self._perturb_and_project(
                                current_state['seq'],
                                pos,
                                grad,
                                alpha,
                                exclude_items=user_items,
                                runtime_cache=runtime_cache,
                                item_selection=self.item_selection,
                                constraint=constraint,
                                n_candidates=n_cand,
                                context_scores=normalized_context_scores,
                            ),
                        )
                        for candidate in candidates:
                            if candidate == current_state['seq'][pos]:
                                continue
                            next_seq = self._replace_seq_item(current_state['seq'], pos, candidate)
                            candidate_seq_by_key.setdefault(next_seq, next_seq)

                    eval_seqs, batch_scores = self._score_candidates_batch(
                        list(candidate_seq_by_key.values())
                    )
                    if batch_scores is None:
                        continue

                    for i, cf_seq in enumerate(eval_seqs):
                        is_valid = (
                            constraint.is_satisfied(batch_scores[i])
                            if constraint is not None
                            else batch_scores[i].argmax().item() != original_rec
                        )
                        if is_valid:
                            ed = hamming_distance(original_seq, list(cf_seq))
                            if ed < best_edits:
                                best_cf = list(cf_seq)
                                best_edits = ed
                            edited_mask[pos] = True
                            found_cf = True
                            break

                    if found_cf:
                        break

                if found_cf:
                    info = {
                        'method': 'gradient',
                        'valid': True,
                        'edit_distance': best_edits,
                        'queries_used': self.query_count,
                        'edits': step + 1,
                        'target_rec': target_rec,
                        'rank_constraint': constraint,
                        'top_k_cfs': [{
                            'seq': best_cf,
                            'edit_distance': best_edits,
                            'valid': True,
                        }],
                    }
                    return best_cf, info

                if self.budget_remaining() <= 0:
                    break

        return None, {
            'method': 'gradient',
            'valid': False,
            'edit_distance': 0,
            'queries_used': self.query_count,
            'edits': 0,
            'top_k_cfs': [],
            'target_rec': target_rec,
            'rank_constraint': constraint,
        }

    def _explain_beam(
        self,
        original_seq: List[int],
        original_rec: int,
        user_items: set,
        constraint: RankConstraint,
        runtime_cache: Dict[str, Any],
    ) -> Tuple[Optional[List[int]], Dict]:
        return self._explain_beam_hotflip_only(original_seq, original_rec, user_items, constraint, runtime_cache)

    def _explain_beam_hotflip_only(
        self,
        original_seq: List[int],
        original_rec: int,
        user_items: set,
        constraint: RankConstraint,
        runtime_cache: Dict[str, Any],
    ) -> Tuple[Optional[List[int]], Dict]:
        base_state = runtime_cache['base_state']
        beam = [
            BeamNode(
                seq_key=base_state['seq_key'],
                seq=base_state['seq'],
                edited_bits=0,
                summary=RankConstraintSummary(
                    item=constraint.item,
                    kind=constraint.kind,
                    top_k=constraint.top_k,
                    rank=constraint.top_k,
                    score=0.0,
                ),
                embed_state=base_state,
            )
        ]
        seen_seq_keys = {base_state['seq_key']}

        with self._gradient_model_mode():
            for _depth in range(self.max_edits + 1):
                beam_states = [node.embed_state for node in beam]
                beam_importance, beam_grad, beam_scores = self._compute_gradient_batch(
                    beam_states, original_rec, runtime_cache, constraint=constraint
                )
                summaries = [
                    self._constraint_summary_from_scores(beam_scores[i], constraint)
                    for i in range(len(beam))
                ]
                for node, summary in zip(beam, summaries):
                    if summary.satisfied:
                        ed = hamming_distance(original_seq, list(node.seq))
                        return list(node.seq), {
                            'method': 'gradient_beam_hotflip_only',
                            'valid': True,
                            'edit_distance': ed,
                            'queries_used': self.query_count,
                            'edits': node.edited_bits.bit_count(),
                            'target_rec': constraint.item if constraint.kind == 'bring_in' and constraint.top_k == 1 else None,
                            'rank_constraint': constraint,
                            'top_k_cfs': [{'seq': list(node.seq), 'edit_distance': ed, 'valid': True}],
                        }

                if _depth == self.max_edits or self.budget_remaining() <= 0:
                    break

                candidate_nodes: Dict[Tuple[int, ...], CandidateNode] = {}
                beam_masked_importance = [
                    self._mask_importance_with_bits(beam_importance[i], beam[i].edited_bits, runtime_cache)
                    for i in range(len(beam))
                ]
                beam_positive_counts = torch.stack(
                    [m.gt(0).sum() for m in beam_masked_importance]
                ).tolist()
                for beam_idx, node in enumerate(beam):
                    importance = beam_masked_importance[beam_idx]
                    positive = beam_positive_counts[beam_idx]
                    if positive <= 0:
                        continue

                    n_pos = max(1, min(self.top_k_positions, positive))
                    _, top_pos = importance.topk(n_pos)
                    normalized_context_scores = self._normalize_context_scores(beam_scores[beam_idx])
                    position_candidates = self._perturb_and_project_positions(
                        node.seq,
                        top_pos,
                        beam_grad[beam_idx:beam_idx + 1],
                        self.alpha,
                        exclude_items=user_items,
                        runtime_cache=runtime_cache,
                        item_selection=self.item_selection,
                        constraint=constraint,
                        n_candidates=max(self.n_candidates_per_position, 8),
                        context_scores=normalized_context_scores,
                    )

                    for pos in top_pos.tolist():
                        if node.edited_bits & (1 << pos):
                            continue
                        merged_candidates = list(position_candidates.get(pos, []))
                        if self.context_candidate_top_m > 0:
                            preferred_item = constraint.item if constraint.kind == 'bring_in' else None
                            context_items = self._context_candidate_items(
                                beam_scores[beam_idx],
                                user_items,
                                node.seq[pos],
                                limit=self.context_candidate_top_m,
                                preferred_item=preferred_item,
                            )
                            max_score = max((score for _item, score in merged_candidates), default=0.0)
                            seen_items = {item for item, _score in merged_candidates}
                            for ctx_rank, item in enumerate(context_items):
                                if item in seen_items:
                                    continue
                                merged_candidates.append((item, max_score - (1.0e-4 * (ctx_rank + 1))))
                                seen_items.add(item)
                        for cand, surrogate_score in merged_candidates:
                            if cand == node.seq[pos]:
                                continue
                            key = self._replace_seq_item(node.seq, pos, cand)
                            if key in seen_seq_keys:
                                continue
                            candidate = CandidateNode(
                                seq=key,
                                parent_state=node.embed_state,
                                edited_bits=self._edited_bits_with_pos(node.edited_bits, pos),
                                pos=pos,
                                cand=cand,
                                surrogate_score=surrogate_score,
                            )
                            prev_candidate = candidate_nodes.get(key)
                            if prev_candidate is None or candidate.surrogate_score > prev_candidate.surrogate_score:
                                candidate_nodes[key] = candidate

                if not candidate_nodes:
                    break

                rerank_factor = self.beam_rerank_factor
                if rerank_factor <= 0:
                    rerank_size = len(candidate_nodes)
                else:
                    rerank_size = min(
                        len(candidate_nodes),
                        max(self.beam_width, self.beam_width * rerank_factor),
                    )
                rerank_pool = heapq.nlargest(
                    rerank_size,
                    candidate_nodes.values(),
                    key=lambda candidate: candidate.surrogate_score,
                )
                top_children = rerank_pool
                if rerank_pool:
                    eval_seqs, batch_scores = self._score_sequence_chunks([child.seq for child in rerank_pool])
                    if batch_scores is not None:
                        summary_by_key = {
                            tuple(seq): self._constraint_summary_from_scores(batch_scores[i], constraint)
                            for i, seq in enumerate(eval_seqs)
                        }
                        scored_children = [
                            (child, summary_by_key[child.seq])
                            for child in rerank_pool
                            if child.seq in summary_by_key
                        ]
                        if scored_children:
                            top_children = [
                                child for child, _summary in heapq.nsmallest(
                                    self.beam_width,
                                    scored_children,
                                    key=lambda item: item[1].beam_sort_key,
                                )
                            ]
                next_states = self._embed_sequences_from_parents_batch(
                    parent_states=[child.parent_state for child in top_children],
                    seq_keys=[child.seq for child in top_children],
                    edit_positions=[child.pos for child in top_children],
                    new_items=[child.cand for child in top_children],
                    cache=runtime_cache,
                )
                beam = [
                    BeamNode(
                        seq_key=child.seq,
                        seq=child.seq,
                        edited_bits=child.edited_bits,
                        summary=RankConstraintSummary(
                            item=constraint.item,
                            kind=constraint.kind,
                            top_k=constraint.top_k,
                            rank=constraint.top_k,
                            score=child.surrogate_score,
                        ),
                        embed_state=next_state,
                    )
                    for child, next_state in zip(top_children, next_states)
                ]
                seen_seq_keys.update(child.seq for child in top_children)

        return None, {
            'method': 'gradient_beam_hotflip_only',
            'valid': False,
            'edit_distance': 0,
            'queries_used': self.query_count,
            'edits': 0,
            'top_k_cfs': [],
            'target_rec': constraint.item if constraint.kind == 'bring_in' and constraint.top_k == 1 else None,
            'rank_constraint': constraint,
        }
