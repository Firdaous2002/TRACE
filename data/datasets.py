"""
PyTorch Dataset classes for sequential recommendation.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    """
    Sequential recommendation dataset.

    For each user with history [i1, i2, ..., iN]:
      - train: input=[i1,...,i(N-2)], target=i(N-1)
      - valid: input=[i1,...,i(N-1)], target=iN
      - test:  input=[i1,...,i(N-1)], target=iN

    Sequences are truncated to max_len (most recent) and left-padded with 0.
    """

    def __init__(
        self,
        user_seqs: Dict[int, List[int]],
        max_len: int,
        num_items: int,
        mode: str = 'train',
    ):
        assert mode in ('train', 'valid', 'test')
        self.max_len = max_len
        self.num_items = num_items
        self.mode = mode
        self.data = []

        for uid, seq in user_seqs.items():
            if len(seq) < 3:
                continue  # need at least 3 items for train/valid/test split

            if mode == 'train':
                input_seq = seq[:-2]   # exclude valid & test
                target = seq[-2]       # valid item is target during training
            elif mode == 'valid':
                input_seq = seq[:-2]   # exclude valid & test
                target = seq[-2]       # valid item
            else:  # test
                input_seq = seq[:-1]
                target = seq[-1]

            if len(input_seq) == 0:
                continue

            # Truncate to max_len (take most recent)
            input_seq = input_seq[-max_len:]
            # Left-pad with 0
            pad_len = max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq

            self.data.append((
                torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return seq, target


class SASRecTrainDataset(Dataset):
    """
    Training dataset for SASRec with BCE loss.

    For each user sequence, creates (input_seq, positive_target, negative_target) tuples.
    At each position t, the model predicts item at t+1.
    One negative is sampled uniformly from items not in user history.
    """

    def __init__(
        self,
        user_seqs: Dict[int, List[int]],
        max_len: int,
        num_items: int,
    ):
        self.max_len = max_len
        self.num_items = num_items
        self.data = []

        for uid, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            train_seq = seq[:-2]  # exclude valid & test
            if len(train_seq) < 2:
                continue

            input_ids = train_seq[:-1]
            target_pos = train_seq[1:]

            input_ids = input_ids[-self.max_len:]
            target_pos = target_pos[-self.max_len:]

            pad_len = self.max_len - len(input_ids)
            padded_input = [0] * pad_len + input_ids
            padded_target = [0] * pad_len + target_pos
            padded_neg = [0] * self.max_len

            self.data.append((
                torch.tensor(padded_input, dtype=torch.long),
                torch.tensor(padded_target, dtype=torch.long),
                torch.tensor(padded_neg, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BERT4RecTrainDataset(Dataset):

    def __init__(
        self,
        user_seqs: Dict[int, List[int]],
        max_len: int,
        num_items: int,
        mask_prob: float = 0.15,
    ):
        self.max_len = max_len
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1
        self.data = []

        for uid, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            train_seq = seq[:-2]
            if len(train_seq) == 0:
                continue
            self.data.append((int(uid), train_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, seq = self.data[idx]
        tokens = seq[-self.max_len:]
        masked_tokens = list(tokens)
        labels = [0] * len(tokens)
        mask_positions = [0] * len(tokens)

        if len(tokens) > 0:
            n_to_mask = max(1, int(round(len(tokens) * self.mask_prob)))
            n_to_mask = min(len(tokens), min(30, n_to_mask))
            selected_positions = random.sample(range(len(tokens)), k=n_to_mask)

            for pos in selected_positions:
                labels[pos] = tokens[pos]
                mask_positions[pos] = 1
                masked_tokens[pos] = self.mask_token

        pad_len = self.max_len - len(masked_tokens)
        masked_tokens = [0] * pad_len + masked_tokens
        labels = [0] * pad_len + labels
        mask_positions = [0] * pad_len + mask_positions

        return (
            torch.tensor(masked_tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(mask_positions, dtype=torch.bool),
        )


class BERT4RecEvalDataset(Dataset):

    def __init__(
        self,
        user_seqs: Dict[int, List[int]],
        max_len: int,
        num_items: int,
        mode: str = 'valid',
    ):
        assert mode in ('valid', 'test')
        self.max_len = max_len
        self.num_items = num_items
        self.mode = mode
        self.mask_token = num_items + 1
        self.data = []

        for uid, seq in user_seqs.items():
            if len(seq) < 3:
                continue

            if mode == 'valid':
                input_seq = seq[:-2]
                target = seq[-2]
            else:
                input_seq = seq[:-1]
                target = seq[-1]

            input_seq = input_seq[-(max_len - 1):]
            pad_len = max_len - len(input_seq) - 1
            masked_input = [0] * pad_len + input_seq + [self.mask_token]
            self.data.append((
                torch.tensor(masked_input, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return seq, target


class GRU4RecEvalDataset(Dataset):

    def __init__(
        self,
        eval_examples: List[Dict],
        max_len: int,
    ):
        self.max_len = max_len
        self.data = []

        for example in eval_examples:
            input_seq = example['input_seq'][-max_len:]
            if len(input_seq) == 0:
                continue
            pad_len = max_len - len(input_seq)
            padded_seq = [0] * pad_len + input_seq
            self.data.append((
                torch.tensor(padded_seq, dtype=torch.long),
                torch.tensor(int(example['target']), dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GRU4RecTrainDataset(Dataset):
    """
    Training dataset for GRU4Rec with BPR loss.

    Each sample: (input_seq, positive_target, negative_target).
    """

    def __init__(
        self,
        user_seqs: Dict[int, List[int]],
        max_len: int,
        num_items: int,
    ):
        self.max_len = max_len
        self.num_items = num_items
        self.data = []

        for uid, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            train_seq = seq[:-2]
            if len(train_seq) < 2:
                continue
            user_items = set(seq)
            self.data.append((int(uid), train_seq, user_items))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, seq, user_items = self.data[idx]

        input_ids = seq[:-1]
        target_pos = seq[-1]

        # Truncate
        input_ids = input_ids[-self.max_len:]

        # Negative sample
        neg = random.randint(1, self.num_items)
        while neg in user_items:
            neg = random.randint(1, self.num_items)

        # Pad
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def load_dataset(data_dir: str, dataset: str):
    """Load preprocessed dataset. Returns user_seqs dict and info dict."""
    ds_dir = os.path.join(data_dir, dataset)

    with open(os.path.join(ds_dir, 'user_seqs.json'), 'r') as f:
        user_seqs = json.load(f)
    # Convert string keys to int
    user_seqs = {int(k): v for k, v in user_seqs.items()}

    with open(os.path.join(ds_dir, 'info.json'), 'r') as f:
        info = json.load(f)

    return user_seqs, info


def load_gru4rec_session_data(data_dir: str, dataset: str) -> Dict:
    ds_dir = os.path.join(data_dir, dataset)
    path = os.path.join(ds_dir, 'gru4rec_sessions.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Re-run data/preprocess.py for dataset '{dataset}' to create GRU4Rec session data."
        )

    with open(path, 'r') as f:
        session_data = json.load(f)

    session_data['train_sessions'] = [
        [int(item_id) for item_id in session]
        for session in session_data.get('train_sessions', [])
    ]

    for key in ('valid_examples', 'test_examples'):
        session_data[key] = [
            {
                'user_id': int(example['user_id']),
                'input_seq': [int(item_id) for item_id in example['input_seq']],
                'target': int(example['target']),
            }
            for example in session_data.get(key, [])
        ]

    return session_data


def load_popular_items(data_dir: str, dataset: str) -> List[int]:
    """Load item popularity ranking."""
    ds_dir = os.path.join(data_dir, dataset)
    with open(os.path.join(ds_dir, 'popular_items.json'), 'r') as f:
        return json.load(f)


def load_category_map(data_dir: str, dataset: str) -> Dict[int, List[str]]:
    """Load item category map."""
    ds_dir = os.path.join(data_dir, dataset)
    path = os.path.join(ds_dir, 'category_map.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        cat_map = json.load(f)
    return {int(k): v for k, v in cat_map.items()}
