import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict, List, Optional



def average_query_count(results: List[Dict]) -> float:
    """Mean number of black-box queries used per instance."""
    if not results:
        return 0.0
    return float(np.mean([r.get('queries_used', 0) for r in results]))


def fidelity_at_k(
    top_k_cfs: List[List[Dict]],
    k_values: List[int],
) -> Dict[int, float]:
    """
    Fidelity@k (paper §4.3).

    For each user, we compute the mean proportion of valid CFs among
    the top-k candidates (Precision@k).

    Args:
        top_k_cfs: per-user list of CF candidates.
                   Each candidate is {'seq': List[int], 'edit_distance': int, 'valid': bool}
        k_values: list of k values to compute Fidelity@k for

    Returns:
        {k: mean_fidelity}
    """
    if not top_k_cfs:
        return {k: 0.0 for k in k_values}

    results = {}
    for k in k_values:
        user_has_valid = []
        for user_cfs in top_k_cfs:
            if not user_cfs:
                user_has_valid.append(0.0)
                continue
            top_k = user_cfs[:k]
            valid_count = sum(1 for cf in top_k if cf.get('valid', False))
            user_has_valid.append(valid_count / len(top_k))
        results[k] = float(np.mean(user_has_valid))
    return results


def hamming_distance_per_instance(
    original_seqs: List[List[int]],
    cf_seqs: List[Optional[List[int]]],
) -> Dict[str, float]:
    pairs = [(o, c) for o, c in zip(original_seqs, cf_seqs) if c is not None]
    if not pairs:
        return {'hamming_mean': float('nan'), 'hamming_std': float('nan')}
    orig_arr = np.array([o for o, _ in pairs], dtype=np.int32)
    cf_arr = np.array([c for _, c in pairs], dtype=np.int32)
    dists = np.sum(orig_arr != cf_arr, axis=1)
    dists = dists[dists > 0]
    if len(dists) == 0:
        return {'hamming_mean': float('nan'), 'hamming_std': float('nan')}
    return {
        'hamming_mean': float(np.mean(dists)),
        'hamming_std': float(np.std(dists)),
    }


def _autoregressive_sequence_nll(model, seq_tensor: torch.Tensor) -> List[float]:
    if seq_tensor.size(1) < 2:
        return [float('nan')] * seq_tensor.size(0)

    input_seq = seq_tensor[:, :-1]
    target_seq = seq_tensor[:, 1:]
    mask = target_seq != 0
    if mask.sum() == 0:
        return [float('nan')] * seq_tensor.size(0)

    hidden = model.forward(input_seq)
    if hasattr(model, 'dense'):
        hidden = model.dense(hidden)
    logits = hidden @ model.item_emb.weight.T
    token_loss = F.cross_entropy(
        logits[mask],
        target_seq[mask],
        reduction='none',
    )

    per_seq = torch.full((seq_tensor.size(0),), float('nan'), device=seq_tensor.device)
    counts = mask.sum(dim=1)
    if token_loss.numel() > 0:
        seq_ids = torch.arange(seq_tensor.size(0), device=seq_tensor.device).unsqueeze(1).expand_as(mask)[mask]
        sums = torch.zeros(seq_tensor.size(0), dtype=token_loss.dtype, device=seq_tensor.device)
        sums.scatter_add_(0, seq_ids, token_loss)
        valid = counts > 0
        per_seq[valid] = sums[valid] / counts[valid]
    return [float(x) for x in per_seq.detach().cpu().tolist()]


def _bert_pseudo_sequence_nll(model, seq_tensor: torch.Tensor) -> List[float]:
    batch_size, seq_len = seq_tensor.shape
    results = torch.full((batch_size,), float('nan'), device=seq_tensor.device)
    owner_ids: List[int] = []
    masked_inputs: List[torch.Tensor] = []
    labels: List[int] = []

    for row_idx in range(batch_size):
        positions = torch.nonzero(seq_tensor[row_idx] != 0, as_tuple=False).flatten()
        for pos in positions.tolist():
            masked = seq_tensor[row_idx].clone()
            labels.append(int(masked[pos].item()))
            masked[pos] = model.mask_token
            masked_inputs.append(masked)
            owner_ids.append(row_idx)

    if not masked_inputs:
        return [float('nan')] * batch_size

    masked_batch = torch.stack(masked_inputs, dim=0)
    hidden = model.forward(masked_batch)
    mask_positions = masked_batch == model.mask_token
    logits = model.output_proj(hidden[mask_positions])
    targets = torch.tensor(labels, dtype=torch.long, device=seq_tensor.device)
    losses = F.cross_entropy(logits, targets, reduction='none')

    owner_tensor = torch.tensor(owner_ids, dtype=torch.long, device=seq_tensor.device)
    sums = torch.zeros(batch_size, dtype=losses.dtype, device=seq_tensor.device)
    counts = torch.zeros(batch_size, dtype=losses.dtype, device=seq_tensor.device)
    sums.scatter_add_(0, owner_tensor, losses)
    counts.scatter_add_(0, owner_tensor, torch.ones_like(losses))

    valid = counts > 0
    results[valid] = sums[valid] / counts[valid]
    return [float(x) for x in results.detach().cpu().tolist()]


def compute_sequence_nll_batch(
    model,
    sequences: List[List[int]],
    max_len: int = 100,
    device: str = 'cuda',
    batch_size: int = 256,
) -> List[float]:
    del max_len
    if not sequences:
        return []

    was_training = model.training
    model.eval()
    outputs: List[float] = []
    model_name = model.__class__.__name__.lower()

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            seq_tensor = torch.tensor(batch, dtype=torch.long, device=device)
            if 'bert4rec' in model_name:
                outputs.extend(_bert_pseudo_sequence_nll(model, seq_tensor))
            else:
                outputs.extend(_autoregressive_sequence_nll(model, seq_tensor))

    if was_training:
        model.train()
    return outputs


def compute_delta_nll(
    original_seqs: List[List[int]],
    cf_seqs: List[Optional[List[int]]],
    model,
    max_len: int = 100,
    device: str = 'cuda',
    original_nll_values: Optional[List[float]] = None,
) -> Dict[str, float]:
    pairs = [(orig, cf) for orig, cf in zip(original_seqs, cf_seqs) if cf is not None]
    if not pairs:
        return {'delta_nll': float('nan')}

    orig_batch = [orig for orig, _ in pairs]
    cf_batch = [cf for _, cf in pairs]
    if original_nll_values is not None and len(original_nll_values) == len(orig_batch):
        orig_nll = np.asarray(original_nll_values, dtype=np.float64)
    else:
        orig_nll = np.asarray(compute_sequence_nll_batch(model, orig_batch, max_len=max_len, device=device), dtype=np.float64)
    cf_nll = np.asarray(compute_sequence_nll_batch(model, cf_batch, max_len=max_len, device=device), dtype=np.float64)

    finite_mask = np.isfinite(orig_nll) & np.isfinite(cf_nll)
    if not np.any(finite_mask):
        return {'delta_nll': float('nan')}
    delta = cf_nll[finite_mask] - orig_nll[finite_mask]
    return {'delta_nll': float(np.mean(delta))}


def compute_all_metrics(
    results: List[Dict],
    original_seqs: Optional[List[List[int]]] = None,
    cf_seqs: Optional[List[Optional[List[int]]]] = None,
    top_k_cfs: Optional[List[List[Dict]]] = None,
    fidelity_k: Optional[List[int]] = None,
    model=None,
    max_len: int = 100,
    device: str = 'cuda',
    original_nll_values: Optional[List[float]] = None,
) -> Dict[str, Optional[float]]:
    del top_k_cfs, fidelity_k

    metrics: Dict[str, Optional[float]] = {
        'queries': average_query_count(results),
    }

    if original_seqs and cf_seqs:
        metrics.update(hamming_distance_per_instance(original_seqs, cf_seqs))

    if model is not None and original_seqs and cf_seqs:
        metrics.update(compute_delta_nll(original_seqs, cf_seqs, model, max_len=max_len, device=device, original_nll_values=original_nll_values))
    return metrics
