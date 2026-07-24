import heapq
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from .base import BaseExplainer, RankConstraint, hamming_distance


class BeamSearchExplainer(BaseExplainer):

    def __init__(
        self,
        model,
        num_items: int,
        beam_width: int = 3,
        k_context: int = 2,
        k_target_nn: int = 2,
        k_local_nn: int = 3,
        candidate_strategy: str = 'pool',
        position_strategy: str = 'all',
        position_top_m: int = 0,
        influence_gamma: float = 1.0,
        pacer_shortlist_size: int = 0,
        pacer_weight_context: float = 1.0,
        pacer_weight_target: float = 1.0,
        pacer_weight_local: float = 0.5,
        eval_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(model, num_items, **kwargs)
        self.beam_width = beam_width
        self.k_context = k_context
        self.k_target_nn = k_target_nn
        self.k_local_nn = k_local_nn
        self.candidate_strategy = candidate_strategy
        self.position_strategy = position_strategy
        self.position_top_m = position_top_m
        self.influence_gamma = influence_gamma
        self.pacer_shortlist_size = pacer_shortlist_size
        self.pacer_weight_context = pacer_weight_context
        self.pacer_weight_target = pacer_weight_target
        self.pacer_weight_local = pacer_weight_local
        self.eval_batch_size = eval_batch_size
        self._emb_norm: Optional[torch.Tensor] = None
        self._emb_norm_device: Optional[str] = None
        self._nn_cache: Dict[Tuple[int, int], List[int]] = {}
        self._nn_filtered_cache: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}

    def _get_scores(self, seq: Sequence[int], cache: dict) -> torch.Tensor:
        seq_list = list(seq)
        key = self._seq_key(seq_list)
        if key not in cache:
            scores = self.predict_scores_no_count(seq_list)
            self.query_count += 1
            self.mask_scores_inplace(scores, seq_list)
            cache[key] = scores.detach().float().clone()
        return cache[key]

    def _seq_key(self, seq: List[int] | Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(seq, tuple):
            return seq
        return tuple(seq)

    def _replace_seq_item(
        self,
        seq: Tuple[int, ...],
        pos: int,
        item: int,
    ) -> Tuple[int, ...]:
        return seq[:pos] + (item,) + seq[pos + 1:]

    def _target_rank(self, scores: torch.Tensor, target: int) -> int:
        return int((scores > scores[target]).sum().item()) + 1

    def _constraint_summary(self, scores: torch.Tensor, constraint: RankConstraint):
        return constraint.summarize(scores)

    def _constraint_margin(self, scores: torch.Tensor, constraint: RankConstraint) -> float:
        item = constraint.item
        item_score = float(scores[item].item())
        others = scores.clone()
        others[item] = -float('inf')
        boundary = float(others.topk(constraint.top_k).values[-1].item())
        if constraint.kind == 'bring_in':
            return item_score - boundary
        return boundary - item_score

    def _populate_cache_batch(self, uncached_keys: List[Tuple[int, ...]], cache: dict) -> None:
        if not uncached_keys:
            return
        batch_size = min(self.eval_batch_size, len(uncached_keys))
        for start in range(0, len(uncached_keys), batch_size):
            chunk_keys = uncached_keys[start:start + batch_size]
            chunk_seqs = [list(key) for key in chunk_keys]
            chunk_tensor = torch.tensor(chunk_seqs, dtype=torch.long, device=self.device)
            batch_scores = self.get_scores_batch(chunk_tensor)
            self.mask_scores_batch_inplace(batch_scores, chunk_tensor)
            for i, key in enumerate(chunk_keys):
                cache[key] = batch_scores[i].detach().float().clone()
            del batch_scores
            del chunk_tensor

    def _ensure_similarity_index(self) -> None:
        device_key = str(self.device)
        if self._emb_norm is not None and self._emb_norm_device == device_key:
            return
        with torch.no_grad():
            emb = self.model.item_emb.weight[1:self.num_items + 1]
            self._emb_norm = F.normalize(emb, dim=-1)
            self._emb_norm_device = device_key
            self._nn_cache = {}
            self._nn_filtered_cache = {}

    def _nn_items(
        self,
        item_id: int,
        k: int,
        exclude: Set[int],
    ) -> List[int]:
        if k <= 0:
            return []
        if self._emb_norm is None:
            raise RuntimeError('Similarity index not initialized in BeamSearchExplainer.explain')

        exclude_key = tuple(sorted(int(x) for x in exclude))
        filtered_key = (int(item_id), exclude_key)
        filtered_cached = self._nn_filtered_cache.get(filtered_key)
        if filtered_cached is not None and len(filtered_cached) >= k:
            return filtered_cached[:k]

        exclude_set = set(exclude_key)
        cache_key = (int(item_id), max(k + len(exclude_set), 32))
        cached = self._nn_cache.get(cache_key)
        if cached is not None:
            filtered = [item for item in cached if item not in exclude_set]
            if len(filtered) >= k:
                self._nn_filtered_cache[filtered_key] = filtered
                return filtered[:k]

        sim = self._emb_norm @ self._emb_norm[item_id - 1]
        exclude_indices = [ex - 1 for ex in exclude_set if 1 <= ex <= self.num_items]
        if exclude_indices:
            sim[torch.tensor(exclude_indices, dtype=torch.long, device=sim.device)] = -float('inf')

        top_k = min(max(cache_key[1], k), self.num_items)
        _, idx = sim.topk(top_k)
        items = [int(i.item()) + 1 for i in idx]
        self._nn_cache[cache_key] = items
        filtered = [item for item in items if item not in exclude_set]
        self._nn_filtered_cache[filtered_key] = filtered
        return filtered[:k]

    def _minmax(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        if float((v_max - v_min).abs().item()) < 1e-12:
            return torch.zeros_like(values)
        return (values - v_min) / (v_max - v_min)

    def _context_candidates(
        self,
        scores: torch.Tensor,
        exclude_items: Sequence[int],
        extra_exclude: int = 0,
    ) -> List[int]:
        slack = len(exclude_items) + extra_exclude
        top_k = min(self.k_context + slack, self.num_items)
        _, ctx_idx = scores.topk(top_k)
        excluded = set(exclude_items)
        merged: List[int] = []
        for item in ctx_idx.tolist():
            if item > 0 and item not in excluded:
                merged.append(int(item))
                if len(merged) >= self.k_context:
                    break
        return merged

    def _pool_candidates(
        self,
        pool: Sequence[int],
        seq_item_set: Set[int],
        current_item: int,
        limit: int,
        preferred_item: Optional[int] = None,
    ) -> List[int]:
        merged: List[int] = []
        seen = set()
        if preferred_item is not None and preferred_item > 0 and preferred_item not in seq_item_set and preferred_item != current_item:
            seen.add(preferred_item)
            merged.append(preferred_item)
        for item in pool:
            item = int(item)
            if item <= 0 or item in seq_item_set or item == current_item or item in seen:
                continue
            seen.add(item)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _merge_shortlist(
        self,
        ctx_candidates: Sequence[int],
        pool_candidates: Sequence[int],
        tgt_nn: Sequence[int],
        local_nn: Sequence[int],
        seq_item_set: Set[int],
        current_item: int,
    ) -> List[int]:
        merged = []
        seen = set()
        for item in ctx_candidates:
            if item > 0 and item not in seq_item_set and item != current_item and item not in seen:
                seen.add(item)
                merged.append(item)
        for item in tgt_nn:
            if item not in seq_item_set and item != current_item and item != 0 and item not in seen:
                seen.add(item)
                merged.append(item)
        for item in local_nn:
            if item not in seq_item_set and item != current_item and item != 0 and item not in seen:
                seen.add(item)
                merged.append(item)
        for item in pool_candidates:
            if item not in seq_item_set and item != current_item and item != 0 and item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

    def _build_shortlist(
        self,
        seq: Tuple[int, ...],
        pos: int,
        scores: torch.Tensor,
        seq_items: Set[int],
        ctx_candidates: Sequence[int],
        candidate_pool: Sequence[int],
        constraint: RankConstraint,
        tgt_nn: Sequence[int],
    ) -> List[int]:
        preferred_item = constraint.item if constraint.kind == 'bring_in' else None
        use_cosine = self.candidate_strategy in {'cosine_mixed', 'hybrid', 'pacer_lite'}
        use_pool = self.candidate_strategy in {'pool', 'hybrid', 'pacer_lite'}

        local_nn: List[int] = []
        if use_cosine and self.k_local_nn > 0:
            local_nn = self._nn_items(seq[pos], self.k_local_nn, seq_items | {0})

        pool_candidates: List[int] = []
        if use_pool:
            pool_candidates = self._pool_candidates(
                candidate_pool,
                seq_items,
                seq[pos],
                limit=max(0, self.k_target_nn + self.k_local_nn),
                preferred_item=preferred_item,
            )
        merged = self._merge_shortlist(
            ctx_candidates,
            pool_candidates,
            tgt_nn if use_cosine else [],
            local_nn,
            seq_items,
            seq[pos],
        )
        if self.candidate_strategy != 'pacer_lite' or not merged:
            return merged

        self._ensure_similarity_index()
        merged_tensor = torch.tensor(merged, dtype=torch.long, device=scores.device)
        item_tensor = merged_tensor - 1
        cand_scores = scores[merged_tensor].detach().float()
        ctx_component = self._minmax(cand_scores)
        if constraint.kind == 'bring_in':
            target_vec = self._emb_norm[constraint.item - 1]
            tgt_component = self._minmax(self._emb_norm[item_tensor] @ target_vec)
        else:
            tgt_component = torch.zeros_like(ctx_component)
        local_vec = self._emb_norm[seq[pos] - 1]
        local_component = self._minmax(self._emb_norm[item_tensor] @ local_vec)
        combined = (
            self.pacer_weight_context * ctx_component
            + self.pacer_weight_target * tgt_component
            + self.pacer_weight_local * local_component
        )
        shortlist_size = self.pacer_shortlist_size if self.pacer_shortlist_size > 0 else len(merged)
        top_idx = combined.topk(min(shortlist_size, combined.numel())).indices.tolist()
        return [merged[i] for i in top_idx]

    def _select_positions(
        self,
        seq: Tuple[int, ...],
        scores: torch.Tensor,
        editable: Sequence[int],
        cache: Dict[Tuple[int, ...], torch.Tensor],
        constraint: RankConstraint,
    ) -> List[int]:
        editable = list(editable)
        if self.position_strategy != 'influence' or len(editable) <= 1:
            return editable

        base_margin = self._constraint_margin(scores, constraint)
        loo_keys: List[Tuple[int, ...]] = []
        pos_to_key: Dict[int, Tuple[int, ...]] = {}
        for pos in editable:
            key = self._replace_seq_item(seq, pos, 0)
            pos_to_key[pos] = key
            if key not in cache:
                loo_keys.append(key)
        if loo_keys:
            self._populate_cache_batch(loo_keys[:self.budget_remaining()], cache)

        influences = []
        seq_len = len(seq)
        for pos in editable:
            key = pos_to_key[pos]
            if key not in cache:
                continue
            loo_scores = cache[key]
            influence = abs(base_margin - self._constraint_margin(loo_scores, constraint))
            if self.influence_gamma != 1.0:
                influence *= self.influence_gamma ** max(0, seq_len - 1 - pos)
            influences.append((float(influence), pos))

        if not influences:
            return editable
        influences.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_m = self.position_top_m if self.position_top_m > 0 else len(influences)
        selected = [pos for _score, pos in influences[:top_m]]
        return selected if selected else editable

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

        if constraint is None:
            info = {'method': 'beam', 'valid': False, 'edit_distance': 0,
                    'queries_used': 0, 'edits': 0, 'top_k_cfs': [], 'target_rec': None}
            return None, info

        cache: dict = {}
        if isinstance(candidate_positions, list) and candidate_positions:
            editable = [int(p) for p in candidate_positions if 0 <= int(p) < len(original_seq) and original_seq[int(p)] != 0]
        else:
            editable = [i for i, item in enumerate(original_seq) if item != 0]
        if isinstance(candidate_items, dict):
            candidate_pool_by_pos = {
                int(pos): [int(item) for item in items]
                for pos, items in candidate_items.items()
            }
        else:
            shared_pool = [int(item) for item in candidate_items] if candidate_items is not None else []
            candidate_pool_by_pos = {pos: shared_pool for pos in editable}

        if self.candidate_strategy in {'cosine_mixed', 'hybrid', 'pacer_lite'}:
            self._ensure_similarity_index()

        scores0 = self._get_scores(original_seq, cache)
        seq_items0 = {item for item in original_seq if item != 0}
        tgt_nn = []
        if self.candidate_strategy in {'cosine_mixed', 'hybrid', 'pacer_lite'} and self.k_target_nn > 0:
            tgt_nn = self._nn_items(constraint.item, self.k_target_nn, seq_items0 | {0})

        beam = [(self._seq_key(original_seq), tuple())]
        best_seq = None
        best_summary = self._constraint_summary(scores0, constraint)

        seen_seqs = {self._seq_key(original_seq)}
        for depth in range(self.max_edits):
            candidates = []
            uncached_keys: List[Tuple[int, ...]] = []
            uncached_set = set()
            budget_left = self.budget_remaining()

            for seq, edits in beam:
                scores = cache[seq]
                seq_items = {item for item in seq if item != 0}
                exclude_items = sorted(seq_items)
                ctx_candidates = self._context_candidates(scores, exclude_items)
                edited_pos = {e[0] for e in edits}
                candidate_positions = [pos for pos in editable if pos not in edited_pos]
                candidate_positions = self._select_positions(
                    seq,
                    scores,
                    candidate_positions,
                    cache,
                    constraint,
                )
                for pos in candidate_positions:
                    candidate_pool = candidate_pool_by_pos.get(pos, [])
                    shortlist = self._build_shortlist(
                        seq,
                        pos,
                        scores,
                        seq_items,
                        ctx_candidates,
                        candidate_pool,
                        constraint,
                        tgt_nn,
                    )
                    for item in shortlist:
                        key = self._replace_seq_item(seq, pos, item)
                        if key in seen_seqs:
                            continue

                        if key not in cache and key not in uncached_set:
                            uncached_set.add(key)
                            uncached_keys.append(key)

                        candidates.append((key, edits + ((pos, item),)))

            if uncached_keys:
                top_uncached = uncached_keys[:self.budget_remaining()]
                self._populate_cache_batch(top_uncached, cache)

            scored_candidates = []
            for new_seq, new_edits in candidates:
                if new_seq not in cache:
                    continue
                new_scores = cache[new_seq]
                summary = self._constraint_summary(new_scores, constraint)
                scored_candidates.append((new_seq, new_edits, summary))

            candidates = scored_candidates

            if not candidates:
                break

            top_candidates = heapq.nsmallest(
                self.beam_width,
                candidates,
                key=lambda x: x[2].beam_sort_key,
            )
            depth_seen = set()
            new_beam = []
            for seq, edits, _summary in top_candidates:
                if seq not in depth_seen:
                    depth_seen.add(seq)
                    new_beam.append((seq, edits))

            beam = new_beam
            seen_seqs.update(seq for seq, _edits in beam)

            top_seq, top_edits = beam[0]
            top_scores = cache[top_seq]
            top_summary = self._constraint_summary(top_scores, constraint)
            if top_summary.beam_sort_key < best_summary.beam_sort_key:
                best_summary = top_summary
                best_seq = list(top_seq)

            if top_summary.satisfied:
                ed = hamming_distance(original_seq, top_seq)
                info = {
                    'method': 'beam', 'valid': True, 'edit_distance': ed,
                    'queries_used': self.query_count, 'edits': len(top_edits),
                    'target_rec': constraint.item if constraint.kind == 'bring_in' else None,
                    'rank_constraint': {
                        'kind': constraint.kind,
                        'top_k': constraint.top_k,
                        'target_item': constraint.item,
                    },
                    'top_k_cfs': [{'seq': list(top_seq), 'edit_distance': ed, 'valid': True}],
                }
                return list(top_seq), info

            if self.budget_remaining() <= 0:
                break

        info = {
            'method': 'beam', 'valid': False, 'edit_distance': 0,
            'queries_used': self.query_count, 'edits': 0,
            'top_k_cfs': [], 'target_rec': constraint.item if constraint.kind == 'bring_in' else None,
            'rank_constraint': {
                'kind': constraint.kind,
                'top_k': constraint.top_k,
                'target_item': constraint.item,
            },
        }
        return None, info
