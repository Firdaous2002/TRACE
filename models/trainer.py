"""
Unified training loop for SASRec, BERT4Rec, GRU4Rec.

Supports:
- Early stopping on validation HR@10
- Checkpoint saving/loading
- Training progress logging
"""

import os
import time
import contextlib
import importlib
import heapq
import random
from typing import Any, Dict, Optional, cast

import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from adam_mini import Adam_mini

_datasets = importlib.import_module('data.datasets')
SASRecTrainDataset = _datasets.SASRecTrainDataset
BERT4RecTrainDataset = _datasets.BERT4RecTrainDataset
BERT4RecEvalDataset = _datasets.BERT4RecEvalDataset
GRU4RecEvalDataset = _datasets.GRU4RecEvalDataset
SequentialDataset = _datasets.SequentialDataset
load_gru4rec_session_data = _datasets.load_gru4rec_session_data
load_popular_items = _datasets.load_popular_items

_sasrec = importlib.import_module('models.sasrec')
_bert4rec = importlib.import_module('models.bert4rec')
_gru4rec = importlib.import_module('models.gru4rec')
SASRec = _sasrec.SASRec
BERT4Rec = _bert4rec.BERT4Rec
GRU4Rec = _gru4rec.GRU4Rec




def _resolve(cfg, dataset: str, model_name: str, key: str, fallback):
    """Resolve key: dataset.model > dataset > model > fallback."""
    ds_cfg = cfg.get('datasets', {}).get(dataset, {})
    # Try datasets.{dataset}.{model}.{key}
    if model_name in ds_cfg and key in ds_cfg[model_name]:
        return ds_cfg[model_name][key]
    # Try datasets.{dataset}.{key}
    if key in ds_cfg:
        return ds_cfg[key]
    # Try {model}.{key}
    model_cfg = cfg.get(model_name, {})
    if key in model_cfg:
        return model_cfg[key]
    return fallback


def build_model(model_name: str, num_items: int, cfg) -> nn.Module:
    dataset = cfg.data.dataset
    hidden_dim = _resolve(cfg, dataset, model_name, 'hidden_dim', cfg.model.hidden_dim)
    max_len = _resolve(cfg, dataset, model_name, 'max_len', cfg.data.max_len)
    dropout = _resolve(cfg, dataset, model_name, 'dropout', cfg.model.dropout)

    if model_name == 'sasrec':
        num_heads = _resolve(cfg, dataset, model_name, 'num_heads', cfg.model.num_heads)
        num_layers = _resolve(cfg, dataset, model_name, 'num_layers', cfg.model.num_layers)
        return SASRec(
            num_items=num_items, hidden_dim=hidden_dim, max_len=max_len,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )
    elif model_name == 'bert4rec':
        num_heads = _resolve(cfg, dataset, model_name, 'num_heads', cfg.model.num_heads)
        num_layers = _resolve(cfg, dataset, model_name, 'num_layers', cfg.model.num_layers)
        return BERT4Rec(
            num_items=num_items, hidden_dim=hidden_dim, max_len=max_len,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )
    elif model_name == 'gru4rec':
        num_gru_layers = _resolve(cfg, dataset, model_name, 'num_gru_layers', 1)
        emb_dim = _resolve(cfg, dataset, model_name, 'emb_dim', 64)
        return GRU4Rec(
            num_items=num_items, hidden_dim=hidden_dim, max_len=max_len,
            dropout=dropout, num_gru_layers=num_gru_layers, emb_dim=emb_dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_optimizer(model: Any, cfg, dataset: str, model_name: str, lr: float, betas: tuple, weight_decay: float):
    optimizer_kwargs = {
        'named_parameters': model.named_parameters(),
        'lr': lr,
        'betas': betas,
        'weight_decay': weight_decay,
        'verbose': False,
    }

    if model_name in {'sasrec', 'bert4rec'}:
        optimizer_kwargs['dim'] = int(_resolve(cfg, dataset, model_name, 'hidden_dim', cfg.model.hidden_dim))
        optimizer_kwargs['n_heads'] = int(_resolve(cfg, dataset, model_name, 'num_heads', cfg.model.num_heads))

    optimizer = Adam_mini(**optimizer_kwargs)
    optimizer.embd_names.update({'item_emb', 'pos_emb'})
    optimizer.output_names.update({'output_proj'})
    optimizer.attn_proj_names.update({'out_proj', 'output_linear'})
    optimizer.mlp_names.update({'ffn'})
    return optimizer


def _unwrap_model(model: Any) -> Any:
    return getattr(model, '_orig_mod', model)


def build_train_dataset(
    model_name: str,
    user_seqs: dict,
    max_len: int,
    num_items: int,
    mask_prob: float = 0.15,
):
    if model_name == 'sasrec':
        return SASRecTrainDataset(user_seqs, max_len, num_items)
    elif model_name == 'bert4rec':
        return BERT4RecTrainDataset(user_seqs, max_len, num_items, mask_prob=mask_prob)
    elif model_name == 'gru4rec':
        return SASRecTrainDataset(user_seqs, max_len, num_items)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_eval_dataset(
    model: nn.Module,
    user_seqs: dict,
    max_len: int,
    num_items: int,
    mode: str,
    session_data: Optional[Dict] = None,
):
    if isinstance(_unwrap_model(model), BERT4Rec):
        return BERT4RecEvalDataset(user_seqs, max_len, num_items, mode=mode)
    return SequentialDataset(user_seqs, max_len, num_items, mode=mode)


@torch.no_grad()
def evaluate_metrics(
    model: Any,
    user_seqs: dict,
    max_len: int,
    num_items: int,
    mode: str = 'valid',
    k: int = 10,
    device: str = 'cuda',
    batch_size: int = 128,
    session_data: Optional[Dict] = None,
) -> dict:
    model.eval()
    dataset = build_eval_dataset(model, user_seqs, max_len, num_items, mode=mode, session_data=session_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    hits = 0
    ndcg_sum = 0.0
    total = 0

    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)
        B = seq.size(0)

        scores = model.predict(seq).float()
        scores[:, 0] = -float('inf')
        seq_mask = seq.masked_fill(seq > num_items, 0)
        scores.scatter_(1, seq_mask, -float('inf'))
        scores[:, 0] = -float('inf')

        _, topk_ids = scores.topk(k, dim=-1)

        # HR@k vectorized
        target_exp = target.unsqueeze(1).expand_as(topk_ids)
        hit_mask = (topk_ids == target_exp).any(dim=1)
        hits += hit_mask.sum().item()

        # NDCG@k vectorized
        match = (topk_ids == target_exp).nonzero(as_tuple=True)
        if len(match[0]) > 0:
            ranks = match[1] + 1
            ndcg_sum += (1.0 / torch.log2(ranks.float() + 1)).sum().item()

        total += B

    hr = hits / total if total > 0 else 0.0
    ndcg = ndcg_sum / total if total > 0 else 0.0
    return {'hr': hr, 'ndcg': ndcg}


def evaluate_hr_at_k(model, user_seqs, max_len, num_items, mode='valid',
                     k=10, device='cuda', batch_size=128, session_data: Optional[Dict] = None) -> float:
    return evaluate_metrics(
        model, user_seqs, max_len, num_items, mode, k, device, batch_size, session_data=session_data,
    )['hr']


def evaluate_ndcg_at_k(model, user_seqs, max_len, num_items, mode='valid',
                       k=10, device='cuda', batch_size=128, session_data: Optional[Dict] = None) -> float:
    return evaluate_metrics(
        model, user_seqs, max_len, num_items, mode, k, device, batch_size, session_data=session_data,
    )['ndcg']


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_path: str,
    epoch: int,
    hr10: float,
    ndcg10: float,
    num_items: int,
    model_name: str,
    cfg,
):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hr10': hr10,
        'ndcg10': ndcg10,
        'num_items': num_items,
        'model_name': model_name,
        'config': dict(cfg),
    }, checkpoint_path)


def _load_best_checkpoint(model: Any, checkpoint_path: str, device: str) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def _build_gru4rec_sampling_state(
    data_dir: str,
    dataset: str,
    train_sessions: list,
    num_items: int,
    sample_alpha: float,
    device: str,
) -> Dict[str, torch.Tensor]:
    popular_items = [int(item_id) for item_id in load_popular_items(data_dir, dataset)]
    popular_item_set = set(popular_items)
    missing_items = [item_id for item_id in range(1, num_items + 1) if item_id not in popular_item_set]
    ordered_items = popular_items + missing_items

    if train_sessions:
        flat_items = np.concatenate([np.asarray(session, dtype=np.int64) for session in train_sessions])
        counts = np.bincount(flat_items, minlength=num_items + 1)
    else:
        counts = np.ones(num_items + 1, dtype=np.int64)

    supports = torch.tensor(
        np.maximum(counts[np.asarray(ordered_items, dtype=np.int64)], 1),
        dtype=torch.float32,
        device=device,
    )
    probs = supports.pow(float(sample_alpha))
    probs = probs / probs.sum()

    return {
        'items': torch.tensor(ordered_items, dtype=torch.long, device=device),
        'probs': probs,
    }


def _sample_gru4rec_negatives(
    target_items: torch.Tensor,
    sampling_state: Dict[str, torch.Tensor],
    n_sample: int,
) -> torch.Tensor:
    sample_indices = torch.multinomial(
        sampling_state['probs'],
        target_items.size(0) * n_sample,
        replacement=True,
    ).view(target_items.size(0), n_sample)
    negatives = sampling_state['items'][sample_indices]
    matches = negatives.eq(target_items.unsqueeze(1))

    while matches.any():
        resample_indices = torch.multinomial(
            sampling_state['probs'],
            int(matches.sum().item()),
            replacement=True,
        )
        negatives[matches] = sampling_state['items'][resample_indices]
        matches = negatives.eq(target_items.unsqueeze(1))

    return negatives


def _compute_gru4rec_bpr_max_loss(
    model: Any,
    step_hidden: torch.Tensor,
    target_items: torch.Tensor,
    negative_items: torch.Tensor,
    bpreg: float,
) -> torch.Tensor:
    pos_scores = model.score_items(step_hidden, target_items).unsqueeze(1)
    neg_scores = model.score_items(step_hidden, negative_items)
    neg_weights = torch.softmax(neg_scores, dim=1)
    pairwise = torch.sigmoid(pos_scores - neg_scores)
    loss = -torch.log((pairwise * neg_weights).sum(dim=1).clamp_min(1e-24))
    if bpreg > 0:
        loss = loss + float(bpreg) * (neg_scores.pow(2) * neg_weights).sum(dim=1)
    return loss.mean()


def _count_gru4rec_epoch_iterations(train_sessions: list, session_batch_size: int) -> int:
    session_steps = [len(session) - 1 for session in train_sessions if len(session) >= 2]
    if not session_steps:
        return 0

    active_count = min(session_batch_size, len(session_steps))
    heap = session_steps[:active_count]
    heapq.heapify(heap)

    for steps in session_steps[active_count:]:
        earliest_finish = heapq.heappop(heap)
        heapq.heappush(heap, earliest_finish + steps)

    return max(heap)


def _count_gru4rec_epoch_iterations_from_lengths(session_lengths: torch.Tensor, session_batch_size: int) -> int:
    session_steps = (session_lengths - 1).clamp_min(0).tolist()
    if not session_steps:
        return 0

    active_count = min(session_batch_size, len(session_steps))
    heap = session_steps[:active_count]
    heapq.heapify(heap)

    for steps in session_steps[active_count:]:
        earliest_finish = heapq.heappop(heap)
        heapq.heappush(heap, earliest_finish + steps)

    return max(heap)


def _pack_gru4rec_sessions(train_sessions: list, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_sessions = [session for session in train_sessions if len(session) >= 2]
    if not ordered_sessions:
        empty = torch.empty(0, 0, dtype=torch.long, device=device)
        lengths = torch.empty(0, dtype=torch.long, device=device)
        return empty, lengths

    lengths_list = [len(session) for session in ordered_sessions]
    max_len = max(lengths_list)
    packed = torch.zeros(len(ordered_sessions), max_len, dtype=torch.long, device=device)
    for idx, session in enumerate(ordered_sessions):
        packed[idx, :len(session)] = torch.as_tensor(session, dtype=torch.long, device=device)
    return packed, torch.as_tensor(lengths_list, dtype=torch.long, device=device)


def _train_gru4rec_epoch(
    model: Any,
    train_sessions: list,
    optimizer: Any,
    scaler: torch.amp.GradScaler,
    session_batch_size: int,
    grad_clip: float,
    device: str,
    use_amp: bool,
    epoch: int,
    epochs: int,
    sampling_state: Dict[str, torch.Tensor],
    n_sample: int,
    loss_type: str,
    bpreg: float,
) -> float:
    packed_sessions, session_lengths = _pack_gru4rec_sessions(train_sessions, device)
    num_sessions = int(session_lengths.numel())
    if num_sessions == 0:
        return 0.0

    perm = torch.randperm(num_sessions, device=device)
    packed_sessions = packed_sessions.index_select(0, perm)
    session_lengths = session_lengths.index_select(0, perm)

    active_count = min(session_batch_size, num_sessions)
    active_session_idx = torch.arange(active_count, device=device, dtype=torch.long)
    active_positions = torch.zeros(active_count, device=device, dtype=torch.long)
    next_session_idx = active_count
    hidden = _unwrap_model(model).init_hidden(active_count, device)

    total_events = int(torch.clamp(session_lengths - 1, min=0).sum().item())
    total_iterations = _count_gru4rec_epoch_iterations_from_lengths(session_lengths, session_batch_size)
    epoch_loss_sum = torch.zeros((), device=device)
    processed_events = 0
    processed_iterations = 0

    pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{epochs}", leave=False)
    while active_session_idx.numel() > 0:
        batch_size = int(active_session_idx.numel())
        input_items = packed_sessions[active_session_idx, active_positions]
        target_items = packed_sessions[active_session_idx, active_positions + 1]
        negative_items = _sample_gru4rec_negatives(target_items, sampling_state, n_sample)

        amp_ctx = torch.autocast(device_type='cuda', enabled=use_amp) if use_amp else contextlib.nullcontext()
        with amp_ctx:
            step_hidden, hidden = model.encode_step(input_items, hidden)
            if loss_type == 'bpr-max':
                loss = _compute_gru4rec_bpr_max_loss(model, step_hidden, target_items, negative_items, bpreg)
            else:
                raise ValueError(f'Unsupported GRU4Rec loss_type: {loss_type}')

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        hidden = hidden.detach()
        epoch_loss_sum += loss.detach() * batch_size
        processed_events += batch_size
        processed_iterations += 1
        pbar.update(1)
        if processed_iterations % 20 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", events=processed_events, active=batch_size)

        active_positions = active_positions + 1
        finished_mask = active_positions >= (session_lengths[active_session_idx] - 1)

        if finished_mask.any():
            finished_slots = finished_mask.nonzero(as_tuple=False).squeeze(1)
            num_finished = int(finished_slots.numel())
            num_available = max(0, num_sessions - next_session_idx)
            num_replacements = min(num_finished, num_available)

            if num_replacements > 0:
                replace_slots = finished_slots[:num_replacements]
                replacement_indices = torch.arange(
                    next_session_idx,
                    next_session_idx + num_replacements,
                    device=device,
                    dtype=torch.long,
                )
                active_session_idx[replace_slots] = replacement_indices
                active_positions[replace_slots] = 0
                hidden[:, replace_slots, :] = 0.0
                next_session_idx += num_replacements

            if num_replacements < num_finished:
                drop_slots = finished_slots[num_replacements:]
                keep_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                keep_mask[drop_slots] = False
                active_session_idx = active_session_idx[keep_mask]
                active_positions = active_positions[keep_mask]
                hidden = hidden[:, keep_mask, :].contiguous()

    pbar.close()
    return (epoch_loss_sum / max(processed_events, 1)).item()


def _train_gru4rec_session_parallel(
    model: Any,
    user_seqs: dict,
    session_data: Dict,
    cfg,
    device: str,
    max_len: int,
    num_items: int,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_epochs: int,
    weight_decay: float,
    betas: tuple,
    checkpoint_path: str,
):
    dataset = cfg.data.dataset
    model_name = cfg.model.name
    session_batch_size = int(_resolve(cfg, dataset, model_name, 'session_batch_size', batch_size))
    grad_clip = float(_resolve(cfg, dataset, model_name, 'grad_clip', 5.0))
    loss_type = str(_resolve(cfg, dataset, model_name, 'loss_type', 'bpr-max')).lower().replace('_', '-')
    n_sample = int(_resolve(cfg, dataset, model_name, 'n_sample', 2048))
    sample_alpha = float(_resolve(cfg, dataset, model_name, 'sample_alpha', 0.5))
    bpreg = float(_resolve(cfg, dataset, model_name, 'bpreg', 1.0))

    if loss_type != 'bpr-max':
        raise ValueError(f'Unsupported GRU4Rec loss_type: {loss_type}')

    optimizer = _build_optimizer(model, cfg, dataset, model_name, lr, betas, weight_decay)
    sampling_state = _build_gru4rec_sampling_state(
        cfg.data.data_dir,
        dataset,
        session_data['train_sessions'],
        num_items,
        sample_alpha,
        device,
    )

    best_hr = 0.0
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, Any] = {'train_loss': [], 'valid_hr10': [], 'valid_ndcg10': []}

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nTraining {model_name} on {cfg.data.dataset}")
    print(f"  num_items={num_items}, max_len={max_len}, session_batch_size={session_batch_size}")
    print(f"  lr={lr}, betas={betas}, weight_decay={weight_decay}, grad_clip={grad_clip}")
    print(f"  loss_type={loss_type}, n_sample={n_sample}, sample_alpha={sample_alpha}, bpreg={bpreg}")
    print(f"  epochs={epochs}, patience={patience}, min_epochs={min_epochs}, amp={use_amp}")
    print(f"  train_sessions={len(session_data['train_sessions'])}, valid_examples={len(session_data['valid_examples'])}, test_examples={len(session_data['test_examples'])}")
    print(f"  device={device}")

    ndcg10 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        avg_loss = _train_gru4rec_epoch(
            model=model,
            train_sessions=session_data['train_sessions'],
            optimizer=optimizer,
            scaler=scaler,
            session_batch_size=session_batch_size,
            grad_clip=grad_clip,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            epochs=epochs,
            sampling_state=sampling_state,
            n_sample=n_sample,
            loss_type=loss_type,
            bpreg=bpreg,
        )

        eval_every = 10
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate_metrics(
                model, user_seqs, max_len, num_items, 'valid', k=10, device=device,
                batch_size=session_batch_size, session_data=session_data,
            )
            hr10 = val_metrics['hr']
            ndcg10 = val_metrics['ndcg']
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, HR@10={hr10:.4f}, NDCG@10={ndcg10:.4f}")

            if epoch == 1 or hr10 > best_hr:
                best_hr = hr10
                best_epoch = epoch
                patience_counter = 0
                _save_checkpoint(model, optimizer, checkpoint_path, epoch, hr10, ndcg10, num_items, model_name, cfg)
                print(f"  -> New best! Saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= min_epochs:
                    print(f"Early stopping at epoch {epoch}. Best HR@10={best_hr:.4f} at epoch {best_epoch}")
                    break
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        history['train_loss'].append(avg_loss)
        history['valid_hr10'].append(best_hr)
        history['valid_ndcg10'].append(
            ndcg10 if epoch % eval_every == 0 or epoch == 1 or epoch == epochs
            else history['valid_ndcg10'][-1] if history['valid_ndcg10'] else 0.0
        )

    checkpoint = _load_best_checkpoint(model, checkpoint_path, device)
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} (HR@10={checkpoint['hr10']:.4f})")

    test_hr10 = evaluate_hr_at_k(
        model, user_seqs, max_len, num_items, 'test', k=10, device=device,
        batch_size=session_batch_size, session_data=session_data,
    )
    test_ndcg10 = evaluate_ndcg_at_k(
        model, user_seqs, max_len, num_items, 'test', k=10, device=device,
        batch_size=session_batch_size, session_data=session_data,
    )
    print(f"Test HR@10={test_hr10:.4f}, Test NDCG@10={test_ndcg10:.4f}")

    history['test_hr10'] = test_hr10
    history['test_ndcg10'] = test_ndcg10
    history['best_epoch'] = best_epoch

    del optimizer, scaler
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    return history


def train_model(
    model: Any,
    user_seqs: dict,
    cfg,
    device: str = 'cuda',
) -> Dict:
    """
    Train a sequential recommendation model.

    Returns dict with training history.
    """
    model_name = cfg.model.name
    dataset = cfg.data.dataset
    max_len = _resolve(cfg, dataset, model_name, 'max_len', cfg.data.max_len)
    num_items = int(model.num_items)
    batch_size = _resolve(cfg, dataset, model_name, 'batch_size', cfg.training.batch_size)
    lr = _resolve(cfg, dataset, model_name, 'lr', cfg.training.lr)
    epochs = _resolve(cfg, dataset, model_name, 'epochs', cfg.training.epochs)
    patience = _resolve(cfg, dataset, model_name, 'patience', cfg.training.patience)
    min_epochs = _resolve(cfg, dataset, model_name, 'min_epochs', getattr(cfg.training, 'min_epochs', 1))
    mask_prob = _resolve(cfg, dataset, model_name, 'mask_prob', 0.15)
    grad_clip = float(_resolve(cfg, dataset, model_name, 'grad_clip', 5.0))

    model = model.to(device)
    if device == 'cuda':
        if model_name != 'gru4rec':
            try:
                model = cast(Any, torch.compile(model))
            except Exception:
                pass
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    weight_decay = _resolve(cfg, dataset, model_name, 'weight_decay', 0.0)
    betas_cfg = _resolve(cfg, dataset, model_name, 'betas', [0.9, 0.999])
    betas = (float(betas_cfg[0]), float(betas_cfg[1])) if hasattr(betas_cfg, '__iter__') else (0.9, 0.999)

    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_{cfg.data.dataset}.pt"
    )

    optimizer = _build_optimizer(model, cfg, dataset, model_name, lr, betas, weight_decay)

    train_dataset = build_train_dataset(
        model_name,
        user_seqs,
        max_len,
        num_items,
        mask_prob=mask_prob,
    )
    if device == 'cuda':
        if batch_size >= 4096:
            num_workers = 0
        elif batch_size >= 2048:
            num_workers = 2
        else:
            num_workers = 8
    else:
        num_workers = 0

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': device == 'cuda',
    }
    if device == 'cuda' and num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    best_hr = 0.0
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, Any] = {'train_loss': [], 'valid_hr10': [], 'valid_ndcg10': []}

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nTraining {model_name} on {cfg.data.dataset}")
    print(f"  num_items={num_items}, max_len={max_len}, batch_size={batch_size}")
    print(f"  lr={lr}, betas={betas}, weight_decay={weight_decay}")
    print(f"  epochs={epochs}, patience={patience}, min_epochs={min_epochs}, amp={use_amp}")
    print(f"  device={device}")

    ndcg10 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            amp_ctx = torch.autocast(device_type='cuda', enabled=use_amp) if use_amp else contextlib.nullcontext()
            if model_name == 'sasrec':
                input_seq, target_pos, target_neg = [b.to(device, non_blocking=True) for b in batch]
                with amp_ctx:
                    loss = model.compute_loss(input_seq, target_pos, target_neg)
            elif model_name == 'bert4rec':
                masked_input, labels, mask_positions = [b.to(device, non_blocking=True) for b in batch]
                with amp_ctx:
                    loss = model.compute_loss(masked_input, labels, mask_positions)
            elif model_name == 'gru4rec':
                input_seq, target_pos, target_neg = [b.to(device, non_blocking=True) for b in batch]
                with amp_ctx:
                    loss = model.compute_loss(input_seq, target_pos, target_neg)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.detach()
            n_batches += 1
            if n_batches % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = (epoch_loss_sum / max(n_batches, 1)).item()

        eval_every = 5
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate_metrics(model, user_seqs, max_len, num_items, 'valid', k=10, device=device)
            hr10 = val_metrics['hr']
            ndcg10 = val_metrics['ndcg']
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, HR@10={hr10:.4f}, NDCG@10={ndcg10:.4f}")

            if epoch == 1 or hr10 > best_hr:
                best_hr = hr10
                best_epoch = epoch
                patience_counter = 0
                _save_checkpoint(model, optimizer, checkpoint_path, epoch, hr10, ndcg10, num_items, model_name, cfg)
                print(f"  -> New best! Saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= min_epochs:
                    print(f"Early stopping at epoch {epoch}. Best HR@10={best_hr:.4f} at epoch {best_epoch}")
                    break
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        history['train_loss'].append(avg_loss)
        history['valid_hr10'].append(best_hr)
        history['valid_ndcg10'].append(ndcg10 if epoch % eval_every == 0 or epoch == 1 or epoch == epochs else history['valid_ndcg10'][-1] if history['valid_ndcg10'] else 0.0)

    checkpoint = _load_best_checkpoint(model, checkpoint_path, device)
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} (HR@10={checkpoint['hr10']:.4f})")

    test_hr10 = evaluate_hr_at_k(model, user_seqs, max_len, num_items, 'test', k=10, device=device)
    test_ndcg10 = evaluate_ndcg_at_k(model, user_seqs, max_len, num_items, 'test', k=10, device=device)
    print(f"Test HR@10={test_hr10:.4f}, Test NDCG@10={test_ndcg10:.4f}")

    history['test_hr10'] = test_hr10
    history['test_ndcg10'] = test_ndcg10
    history['best_epoch'] = best_epoch

    del train_loader, train_dataset, optimizer, scaler
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    return history


def load_trained_model(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    num_items = checkpoint['num_items']

    from omegaconf import OmegaConf
    cfg = OmegaConf.create(checkpoint['config'])

    model = build_model(model_name, num_items, cfg)
    # torch.compile saves keys with '_orig_mod.' prefix; strip it for loading
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
