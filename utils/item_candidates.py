"""
Candidate item and position selection for counterfactual explanations.

Position selection: attention-agnostic scoring via single-removal score drop.
Item selection: same-category items + popular items.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np


def score_positions(
    model,
    original_seq: List[int],
    original_rec: int,
    num_items: int,
    k_pos: int = 10,
    device: str = 'cuda',
) -> List[int]:
    """
    Score positions by single-removal importance.

    For each non-padding position p, replace S[p] with a random item
    (not in sequence) and measure score drop on the originally recommended item.
    Return top-K_pos positions with largest score drop.

    Args:
        model: trained SRS model
        original_seq: original item sequence (list of item IDs, may include 0 padding)
        original_rec: item ID of the original top-1 recommendation
        num_items: total number of items in the dataset
        k_pos: number of top positions to return
        device: torch device

    Returns:
        List of position indices (into original_seq) sorted by importance (descending)
    """
    seq_len = len(original_seq)

    # Get original score for the recommended item
    seq_tensor = torch.tensor([original_seq], dtype=torch.long, device=device)
    with torch.no_grad():
        original_scores = model.predict(seq_tensor)  # (1, num_items+1)

    # Find non-padding positions
    non_pad_positions = [p for p in range(seq_len) if original_seq[p] != 0]

    if len(non_pad_positions) == 0:
        return []

    # Pick a random replacement item (not in sequence)
    seq_set = set(original_seq)
    candidates = [i for i in range(1, num_items + 1) if i not in seq_set]
    replacement = random.choice(candidates) if candidates else 1

    # Batch evaluate all positions
    batch_seqs = []
    for p in non_pad_positions:
        modified = list(original_seq)
        modified[p] = replacement
        batch_seqs.append(modified)

    batch_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
    with torch.no_grad():
        batch_scores = model.predict(batch_tensor)  # (N, num_items+1)

    # Single host transfer for all target-item scores (was N+1 CUDA syncs via
    # per-position `.item()` calls inside the original score-drop loop).
    target_scores = torch.cat(
        (original_scores[0:1, original_rec], batch_scores[:, original_rec])
    ).cpu().tolist()
    original_score = target_scores[0]
    drops = [original_score - s for s in target_scores[1:]]

    # Sort by score drop (descending) and take top k_pos
    score_drops = sorted(
        zip(non_pad_positions, drops), key=lambda x: x[1], reverse=True,
    )
    top_positions = [p for p, _ in score_drops[:k_pos]]

    return top_positions


def select_candidate_items(
    position: int,
    original_seq: List[int],
    num_items: int,
    user_items: Optional[Set[int]] = None,
) -> List[int]:
    """
    Select candidate replacement items for a given position.

    Paper mutation: randomly select an item not already present in the sequence.
    We sample from the full item range [1, num_items], excluding items in the
    user's history and padding (0).

    Args:
        position: position index in the sequence
        original_seq: original item sequence
        num_items: total number of items in the dataset
        user_items: set of items to exclude (items in user's history)

    Returns:
        List of candidate item IDs (all items not in sequence)
    """
    current_item = original_seq[position]
    exclude = user_items or set(original_seq)
    exclude = exclude | {0}  # also exclude padding
    exclude = exclude | {current_item}

    # All valid items not in exclude set
    candidates = [i for i in range(1, num_items + 1) if i not in exclude]
    random.shuffle(candidates)
    return candidates


def build_candidate_map(
    original_seq: List[int],
    candidate_positions: List[int],
    num_items: int,
    user_items: Optional[Set[int]] = None,
) -> Dict[int, List[int]]:
    """
    Build a map of candidate items for each candidate position.

    Paper: mutation selects random items not already present.
    Each position gets the full set of available items.

    Args:
        original_seq: original sequence
        candidate_positions: list of position indices to consider
        num_items: total number of items in the dataset
        user_items: items to exclude

    Returns:
        {position: [candidate_item_1, ..., candidate_item_n]}
    """
    candidate_map = {}
    for p in candidate_positions:
        candidate_map[p] = select_candidate_items(
            p, original_seq, num_items, user_items,
        )
    return candidate_map
