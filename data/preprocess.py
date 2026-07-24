"""
Download and preprocess datasets: MovieLens-100K, MovieLens-1M, Steam.

Preprocessing steps (in order):
1. Load raw interaction data
2. 5-core iterative filtering (min 5 interactions per user AND item)
3. Sort by timestamp per user
4. Assign contiguous integer IDs (items 1..|I|, users 1..|U|, 0=padding)
5. Split: last=test, second-to-last=valid, rest=train
6. Save as JSON: {user_id: [item_id_1, ...], ...}
"""

import os
import json
import zipfile
import gzip
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_URLS = {
    'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'steam': 'https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz',
    'beauty': 'https://jmcauley.ucsd.edu/data/amazon/links/reviews_Beauty_5.json.gz',
    'sports': 'https://jmcauley.ucsd.edu/data/amazon/links/reviews_Sports_and_Outdoors_5.json.gz',
}

AMAZON_5CORE_FALLBACK_URLS = {
    'beauty': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
    'sports': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
}

# Category metadata URLs (for candidate item selection)
METADATA_URLS = {
    'ml-100k': None,   # genres are in the u.item file
    'ml-1m': None,     # genres are in the movies.dat file
    'steam': 'https://cseweb.ucsd.edu/~wckang/steam_games.json.gz',
}

# ML-100K genre list (from u.genre), order matches the binary columns in u.item
ML100K_GENRES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]

SESSION_GAP_SECONDS = 3600


def download_file(url: str, dest: str):
    """Download a file from URL to destination."""
    import urllib.request
    print(f"Downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest}")


# ── MovieLens-100K ───────────────────────────────────────────────────────────

def load_ml100k(raw_dir: str) -> pd.DataFrame:
    """Load MovieLens-100K dataset."""
    zip_path = os.path.join(raw_dir, 'ml-100k.zip')
    if not os.path.exists(zip_path):
        download_file(DATASET_URLS['ml-100k'], zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_dir)

    ratings_path = os.path.join(raw_dir, 'ml-100k', 'u.data')
    df = pd.read_csv(
        ratings_path,
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1',
    )
    return pd.DataFrame(df[['user_id', 'item_id', 'timestamp']])


def load_ml100k_genres(raw_dir: str) -> dict:
    """Load ML-100K genre metadata from u.item. Returns {item_id: [genre1, ...]}."""
    item_path = os.path.join(raw_dir, 'ml-100k', 'u.item')
    genres = {}
    with open(item_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            # Last 19 columns are binary genre indicators
            genre_flags = parts[5:]  # 19 genre columns
            genre_list = [
                ML100K_GENRES[i]
                for i in range(min(len(genre_flags), len(ML100K_GENRES)))
                if genre_flags[i] == '1'
            ]
            if genre_list:
                genres[item_id] = genre_list
    return genres


# ── MovieLens-1M ─────────────────────────────────────────────────────────────

def load_ml1m(raw_dir: str) -> pd.DataFrame:
    """Load MovieLens-1M dataset."""
    zip_path = os.path.join(raw_dir, 'ml-1m.zip')
    if not os.path.exists(zip_path):
        download_file(DATASET_URLS['ml-1m'], zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_dir)

    ratings_path = os.path.join(raw_dir, 'ml-1m', 'ratings.dat')
    df = pd.read_csv(
        ratings_path,
        sep='::',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1',
    )
    return pd.DataFrame(df[['user_id', 'item_id', 'timestamp']])


def load_ml1m_genres(raw_dir: str) -> dict:
    """Load MovieLens-1M genre metadata. Returns {item_id: [genre1, genre2, ...]}."""
    movies_path = os.path.join(raw_dir, 'ml-1m', 'movies.dat')
    genres = {}
    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            item_id = int(parts[0])
            genre_list = parts[2].split('|')
            genres[item_id] = genre_list
    return genres


# ── Steam ────────────────────────────────────────────────────────────────────

def load_steam(raw_dir: str) -> pd.DataFrame:
    """
    Load Steam reviews dataset (Kang & McAuley, ICDM 2018).

    Format: one JSON object per line (gzipped).
    Fields: username, product_id, date (YYYY-MM-DD), page_order, etc.
    """
    gz_path = os.path.join(raw_dir, 'steam_reviews.json.gz')
    if not os.path.exists(gz_path):
        download_file(DATASET_URLS['steam'], gz_path)

    records = []
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading steam"):
            try:
                d = eval(line.strip())  # Steam data uses Python dict format
                username = d.get('username', '')
                product_id = d.get('product_id', '')
                date_str = d.get('date', '')
                if username and product_id and date_str:
                    # Convert date string to unix timestamp for sorting
                    try:
                        from datetime import datetime
                        ts = int(datetime.strptime(date_str, '%Y-%m-%d').timestamp())
                    except (ValueError, TypeError):
                        ts = 0
                    records.append({
                        'user_id': username,
                        'item_id': product_id,
                        'timestamp': ts,
                    })
            except (SyntaxError, ValueError):
                continue

    df = pd.DataFrame(records)
    # Deduplicate: keep first review per (user, item) pair
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    return pd.DataFrame(df[['user_id', 'item_id', 'timestamp']])


def _load_amazon_5core(raw_dir: str, dataset_key: str, filename: str) -> pd.DataFrame:
    gz_path = os.path.join(raw_dir, filename)
    if not os.path.exists(gz_path):
        try:
            download_file(DATASET_URLS[dataset_key], gz_path)
        except Exception:
            download_file(AMAZON_5CORE_FALLBACK_URLS[dataset_key], gz_path)

    records = []
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {dataset_key}"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = d.get('reviewerID')
            item_id = d.get('asin')
            timestamp = d.get('unixReviewTime')

            if user_id and item_id and timestamp is not None:
                try:
                    ts = int(timestamp)
                except (TypeError, ValueError):
                    continue
                records.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': ts,
                })

    df = pd.DataFrame(records)
    return pd.DataFrame(df[['user_id', 'item_id', 'timestamp']])


def load_amazon_beauty(raw_dir: str) -> pd.DataFrame:
    return _load_amazon_5core(raw_dir, 'beauty', 'reviews_Beauty_5.json.gz')


def load_amazon_sports(raw_dir: str) -> pd.DataFrame:
    return _load_amazon_5core(raw_dir, 'sports', 'reviews_Sports_and_Outdoors_5.json.gz')


def load_steam_metadata(raw_dir: str) -> dict:
    """Load Steam game metadata for genre/tag info. Returns {product_id: [tag1, ...]}."""
    gz_path = os.path.join(raw_dir, 'steam_games.json.gz')
    if not os.path.exists(gz_path):
        url = METADATA_URLS['steam']
        if url:
            download_file(url, gz_path)
        else:
            return {}

    categories = {}
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                d = eval(line.strip())
                app_id = str(d.get('id', ''))
                tags = d.get('tags', [])
                genres = d.get('genres', [])
                cats = tags if tags else genres
                if isinstance(cats, str):
                    cats = [cats]
                if app_id and cats:
                    categories[app_id] = cats[:5]  # keep top 5 tags
            except (SyntaxError, ValueError):
                continue
    return categories


# ── Common utilities ─────────────────────────────────────────────────────────

def five_core_filter(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Iterative 5-core filtering until convergence."""
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        # Filter users
        user_counts = Counter(df['user_id'].tolist())
        valid_users = [u for u, c in user_counts.items() if c >= min_count]
        df = pd.DataFrame(df[df['user_id'].isin(valid_users)])
        # Filter items
        item_counts = Counter(df['item_id'].tolist())
        valid_items = [i for i, c in item_counts.items() if c >= min_count]
        df = pd.DataFrame(df[df['item_id'].isin(valid_items)])
        iteration += 1
    print(f"5-core filtering converged after {iteration} iterations")
    print(f"Remaining: {df['user_id'].nunique()} users, {df['item_id'].nunique()} items, {len(df)} interactions")
    return df


def assign_ids(df: pd.DataFrame):
    """Assign contiguous integer IDs. Items: 1..|I|, Users: 1..|U|. 0=padding."""
    unique_items = sorted(df['item_id'].unique())
    item2id = {item: idx + 1 for idx, item in enumerate(unique_items)}

    unique_users = sorted(df['user_id'].unique())
    user2id = {user: idx + 1 for idx, user in enumerate(unique_users)}

    df = df.copy()
    df['item_id'] = df['item_id'].apply(lambda x: item2id[x])
    df['user_id'] = df['user_id'].apply(lambda x: user2id[x])

    return df, user2id, item2id


def build_user_sequences(df: pd.DataFrame) -> dict:
    """Build chronologically sorted user sequences."""
    df = df.sort_values(['user_id', 'timestamp'])
    user_seqs = {}
    for uid, group in df.groupby('user_id'):
        user_seqs[int(str(uid))] = group['item_id'].tolist()
    return user_seqs


def sessionize_items(items: list, timestamps: list, gap_seconds: int = SESSION_GAP_SECONDS) -> list:
    sessions = []
    current_session = []
    prev_ts = None

    for item_id, timestamp in zip(items, timestamps):
        ts = int(timestamp)
        if current_session and prev_ts is not None and (ts - prev_ts) >= gap_seconds:
            sessions.append(current_session)
            current_session = []

        current_session.append(int(item_id))
        prev_ts = ts

    if current_session:
        sessions.append(current_session)
    return sessions


def build_gru4rec_eval_example(items: list, timestamps: list, gap_seconds: int = SESSION_GAP_SECONDS):
    sessions = sessionize_items(items, timestamps, gap_seconds=gap_seconds)
    if not sessions:
        return None

    last_session = sessions[-1]
    if len(last_session) < 2:
        return None

    return {
        'input_seq': last_session[:-1],
        'target': int(last_session[-1]),
    }


def build_gru4rec_session_artifact(df: pd.DataFrame, gap_seconds: int = SESSION_GAP_SECONDS) -> dict:
    df = df.sort_values(['user_id', 'timestamp'])

    train_sessions = []
    valid_examples = []
    test_examples = []
    total_sessions = 0
    total_session_interactions = 0

    for uid, group in df.groupby('user_id'):
        uid_int = int(str(uid))
        items = [int(item_id) for item_id in group['item_id'].tolist()]
        timestamps = [int(ts) for ts in group['timestamp'].tolist()]
        full_sessions = sessionize_items(items, timestamps, gap_seconds=gap_seconds)
        total_sessions += len(full_sessions)
        total_session_interactions += sum(len(session) for session in full_sessions)

        if len(items) < 3:
            continue

        train_prefix_items = items[:-2]
        train_prefix_timestamps = timestamps[:-2]
        for session in sessionize_items(train_prefix_items, train_prefix_timestamps, gap_seconds=gap_seconds):
            if len(session) >= 2:
                train_sessions.append(session)

        valid_example = build_gru4rec_eval_example(items[:-1], timestamps[:-1], gap_seconds=gap_seconds)
        if valid_example is not None:
            valid_examples.append({'user_id': uid_int, **valid_example})

        test_example = build_gru4rec_eval_example(items, timestamps, gap_seconds=gap_seconds)
        if test_example is not None:
            test_examples.append({'user_id': uid_int, **test_example})

    avg_session_length = (total_session_interactions / total_sessions) if total_sessions > 0 else 0.0
    return {
        'session_gap_seconds': gap_seconds,
        'session_boundary_rule': f'new_session_if_gap_at_least_{gap_seconds}_seconds',
        'native_sessions': False,
        'train_sessions': train_sessions,
        'valid_examples': valid_examples,
        'test_examples': test_examples,
        'stats': {
            'num_total_sessions': total_sessions,
            'avg_session_length': avg_session_length,
            'num_train_sessions': len(train_sessions),
            'num_train_steps': sum(max(0, len(session) - 1) for session in train_sessions),
            'num_valid_examples': len(valid_examples),
            'num_test_examples': len(test_examples),
        },
    }


def build_category_map(raw_categories: dict, item2id: dict) -> dict:
    """Convert raw category map to use integer item IDs. Returns {int_item_id: [cat1, ...]}."""
    cat_map = {}
    for orig_id, cats in raw_categories.items():
        if orig_id in item2id:
            cat_map[item2id[orig_id]] = cats
    return cat_map


def compute_item_popularity(user_seqs: dict) -> list:
    """Compute item popularity ranking. Returns list of item IDs sorted by frequency (descending)."""
    counter = Counter()
    for seq in user_seqs.values():
        counter.update(seq)
    return [item_id for item_id, _ in counter.most_common()]


# ── Main pipeline ────────────────────────────────────────────────────────────

def preprocess_dataset(dataset: str, raw_dir: str, output_dir: str, min_interactions: int = 5):
    """Full preprocessing pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"Preprocessing: {dataset}")
    print(f"{'='*60}")

    # 1. Load raw data
    if dataset == 'ml-100k':
        df = load_ml100k(raw_dir)
    elif dataset == 'ml-1m':
        df = load_ml1m(raw_dir)
    elif dataset == 'steam':
        df = load_steam(raw_dir)
    elif dataset == 'beauty':
        df = load_amazon_beauty(raw_dir)
    elif dataset == 'sports':
        df = load_amazon_sports(raw_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Raw: {df['user_id'].nunique()} users, {df['item_id'].nunique()} items, {len(df)} interactions")

    # 2. 5-core filtering
    df = five_core_filter(df, min_count=min_interactions)

    # 3. Assign integer IDs
    df, user2id, item2id = assign_ids(df)

    num_users = len(user2id)
    num_items = len(item2id)
    print(f"Final: {num_users} users, {num_items} items")

    # 4. Build user sequences (sorted by timestamp)
    user_seqs = build_user_sequences(df)

    gru4rec_sessions = build_gru4rec_session_artifact(df, gap_seconds=SESSION_GAP_SECONDS)

    # 5. Item popularity
    popular_items = compute_item_popularity(user_seqs)

    # 6. Category metadata
    if dataset == 'ml-100k':
        raw_cats = load_ml100k_genres(raw_dir)
        cat_map = {}
        for orig_id, cats in raw_cats.items():
            if orig_id in item2id:
                cat_map[item2id[orig_id]] = cats
    elif dataset == 'ml-1m':
        raw_cats = load_ml1m_genres(raw_dir)
        cat_map = {}
        for orig_id, cats in raw_cats.items():
            if orig_id in item2id:
                cat_map[item2id[orig_id]] = cats
    elif dataset == 'steam':
        raw_cats = load_steam_metadata(raw_dir)
        cat_map = build_category_map(raw_cats, item2id)
    else:
        cat_map = {}

    # 7. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    ds_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    # User sequences
    with open(os.path.join(ds_dir, 'user_seqs.json'), 'w') as f:
        json.dump(user_seqs, f)

    with open(os.path.join(ds_dir, 'gru4rec_sessions.json'), 'w') as f:
        json.dump(gru4rec_sessions, f)

    # Dataset info
    info = {
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': len(df),
        'dataset': dataset,
        'gru4rec_sessions': {
            'artifact': 'gru4rec_sessions.json',
            'native_sessions': False,
            'session_gap_seconds': SESSION_GAP_SECONDS,
            'session_boundary_rule': gru4rec_sessions['session_boundary_rule'],
            **gru4rec_sessions['stats'],
        },
    }
    with open(os.path.join(ds_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # Item popularity
    with open(os.path.join(ds_dir, 'popular_items.json'), 'w') as f:
        json.dump(popular_items, f)

    # Category map
    cat_map_str = {str(k): v for k, v in cat_map.items()}
    with open(os.path.join(ds_dir, 'category_map.json'), 'w') as f:
        json.dump(cat_map_str, f)

    # ID mappings (for debugging / case studies)
    with open(os.path.join(ds_dir, 'item2id.json'), 'w') as f:
        json.dump({str(k): v for k, v in item2id.items()}, f)

    print(f"Saved to {ds_dir}")
    return user_seqs, info


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess datasets for targeted CF experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['ml-100k', 'ml-1m', 'steam', 'beauty', 'sports', 'all'],
                        help='Dataset to preprocess')
    parser.add_argument('--raw_dir', type=str, default='./data/raw',
                        help='Directory for raw downloaded data')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Directory for processed data')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum interactions for k-core filtering')
    args = parser.parse_args()

    datasets = ['ml-100k', 'ml-1m', 'steam', 'beauty', 'sports'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        preprocess_dataset(ds, args.raw_dir, args.output_dir, args.min_interactions)


if __name__ == '__main__':
    main()
