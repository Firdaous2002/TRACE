"""
Evaluate trained backbone checkpoints on the held-out test split and write
HR@10 / NDCG@10 for every (model, dataset) combination to
``results/backbone_eval.json``.

This script is what populates Table 1 of the paper. It does not train anything
and does not touch the explainers. Run it after
``scripts/train_backbones.sh`` (or after dropping the 15 ``.pt`` files into
``./checkpoints/``).

Usage:
    python eval_backbones.py
    python eval_backbones.py --models sasrec --datasets ml-100k ml-1m
    python eval_backbones.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf

from data.datasets import load_dataset, load_gru4rec_session_data
from models.trainer import _resolve, evaluate_metrics, load_trained_model
from utils.experiment_utils import load_config

DEFAULT_MODELS = ["sasrec", "bert4rec", "gru4rec"]
DEFAULT_DATASETS = ["ml-100k", "ml-1m", "steam", "beauty", "sports"]

MODEL_LABEL = {"sasrec": "SASRec", "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec"}
DATASET_LABEL = {
    "ml-100k": "ML-100K", "ml-1m": "ML-1M", "steam": "Steam",
    "beauty": "Beauty", "sports": "Sports",
}


def _evaluate_one(
    model_name: str,
    dataset: str,
    checkpoint_dir: str,
    data_dir: str,
    config_path: str,
    device: str,
) -> Dict[str, float] | None:
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset}.pt")
    ds_dir = os.path.join(data_dir, dataset)
    if not os.path.isfile(ckpt_path):
        print(f"  [skip] checkpoint missing: {ckpt_path}")
        return None
    if not os.path.isdir(ds_dir):
        print(f"  [skip] processed data missing: {ds_dir}")
        return None

    cfg = load_config(config_path, overrides=[f"model.name={model_name}", f"data.dataset={dataset}"])
    max_len = int(_resolve(cfg, dataset, model_name, 'max_len', cfg.data.max_len))

    user_seqs, info = load_dataset(data_dir, dataset)
    num_items = info['num_items']

    session_data = None
    if model_name == 'gru4rec':
        session_data = load_gru4rec_session_data(data_dir, dataset)

    model = load_trained_model(ckpt_path, device=device)

    t0 = time.time()
    metrics = evaluate_metrics(
        model, user_seqs, max_len=max_len, num_items=num_items,
        mode='test', k=10, device=device, batch_size=128,
        session_data=session_data,
    )
    elapsed = time.time() - t0
    print(f"  HR@10={metrics['hr']:.4f}  NDCG@10={metrics['ndcg']:.4f}  ({elapsed:.1f}s)")

    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return {
        'hr10': float(metrics['hr']),
        'ndcg10': float(metrics['ndcg']),
        'num_users': len(user_seqs),
        'num_items': int(num_items),
        'eval_seconds': round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--data-dir', default='./data/processed')
    parser.add_argument('--config', default='./config/default.yaml')
    parser.add_argument('--output', default='./results/backbone_eval.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"[eval] device={args.device} checkpoints={args.checkpoint_dir} data={args.data_dir}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name in args.models:
        results.setdefault(model_name, {})
        for dataset in args.datasets:
            print(f"[{model_name} / {dataset}]")
            rec = _evaluate_one(model_name, dataset, args.checkpoint_dir, args.data_dir, args.config, args.device)
            if rec is not None:
                results[model_name][dataset] = rec

    payload = {
        'schema': 'backbone_eval.v1',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'device': args.device,
        'models': results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote {args.output}")


if __name__ == '__main__':
    main()
