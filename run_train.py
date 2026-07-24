"""
Train SRS models (SASRec, BERT4Rec, GRU4Rec).

Usage:
    python run_train.py                          # default config
    python run_train.py model.name=bert4rec      # override model
    python run_train.py data.dataset=beauty      # override dataset
    python run_train.py --train-all              # train all 9 combinations
"""

import argparse
import json
import os
import gc

import torch
import numpy as np
import random

from omegaconf import OmegaConf

from data.datasets import load_dataset
from models.trainer import build_model, train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_single(cfg):
    """Train a single model on a single dataset."""
    set_seed(cfg.seed)

    dataset = cfg.data.dataset
    model_name = cfg.model.name
    device = cfg.training.device

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset}")
    print(f"{'='*60}")

    # Load data
    data_dir = cfg.data.data_dir
    user_seqs, info = load_dataset(data_dir, dataset)
    num_items = info['num_items']

    print(f"  Users: {info['num_users']}, Items: {num_items}, "
          f"Interactions: {info['num_interactions']}")

    # Build model
    model = build_model(model_name, num_items, cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Train
    history = train_model(model, user_seqs, cfg, device=device)

    # Save training history
    results_dir = cfg.evaluation.results_dir
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, f"train_{model_name}_{dataset}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train SRS models')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all 9 model-dataset combinations')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides (e.g., model.name=bert4rec)')
    args = parser.parse_args()

    # Load config
    base_cfg = OmegaConf.load(args.config)
    # Apply CLI overrides
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    if args.train_all:
        datasets = ['ml-100k', 'ml-1m', 'steam']
        models = ['sasrec', 'bert4rec', 'gru4rec']
        all_results = {}

        for dataset in datasets:
            for model_name in models:
                cfg_copy = OmegaConf.create(dict(cfg))
                cfg_copy.data.dataset = dataset
                cfg_copy.model.name = model_name

                history = train_single(cfg_copy)
                key = f"{model_name}_{dataset}"
                all_results[key] = {
                    'test_hr10': history.get('test_hr10', 0),
                    'test_ndcg10': history.get('test_ndcg10', 0),
                    'best_epoch': history.get('best_epoch', 0),
                }

                # Free memory between runs to prevent segfaults on large datasets
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Print summary
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Dataset':<12} {'HR@10':>8} {'NDCG@10':>8} {'Epoch':>6}")
        print("-" * 50)
        for key, r in all_results.items():
            model_name, dataset = key.split('_', 1)
            print(f"{model_name:<12} {dataset:<12} {r['test_hr10']:>8.4f} "
                  f"{r['test_ndcg10']:>8.4f} {r['best_epoch']:>6}")
    else:
        train_single(cfg)


if __name__ == '__main__':
    main()
