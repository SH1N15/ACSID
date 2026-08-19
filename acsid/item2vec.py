"""Item2Vec (SGNS) collaborative embedding training.

Trains a word2vec-style skip-gram model over user interaction sequences using
ONLY the train split (PROJECT_PLAN.md §12.1 — collaborative signal must not
be contaminated by valid/test interactions). Outputs an [N, dim] numpy array
indexed by item_id == row index, matching the text .npy layout.

Run on CPU (plan §4.4); gensim 4.x API. Designed to be idempotent: given the
same train CSV and seed it produces deterministic weights (Word2Vec seeded).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def build_user_sequences(train_csv: str) -> list[list[int]]:
    """One sequence per user = history + target, in chronological order.

    The CSV stores list cells as pandas repr (history_item_id is the repr of
    a Python list of ints). We eval to recover lists, exactly as data.py
    does. Each user_longest-train record is one sequence.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    sequences: list[list[int]] = []
    for hist, tgt in zip(df["history_item_id"], df["item_id"]):
        try:
            history_list = eval(hist) if isinstance(hist, str) else hist
        except Exception:
            history_list = []
        if history_list is None:
            history_list = []
        history_list = [int(x) for x in history_list]
        tgt = int(tgt)
        seq = history_list + [tgt]
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def train_item2vec(
    train_csv: str,
    output_npy: str,
    dim: int = 256,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 20,
    seed: int = 42,
    n_items: Optional[int] = None,
    sg: int = 1,
    workers: int = 4,
) -> np.ndarray:
    """Train Item2Vec via gensim Word2Vec (sg=1 => skip-gram / SGNS).

    Writes cf.npy of shape [n_items, dim], row i == item i. Items never seen
    in train get a zero vector (alpha_i will be 0 for them, so their CF row
    is unused by the fusion — see adaptive_fusion.compute_alpha).
    """
    from gensim.models import Word2Vec

    sequences = build_user_sequences(train_csv)
    # gensim expects lists of str tokens
    sentences = [[str(x) for x in seq] for seq in sequences]
    print(f"[item2vec] {len(sentences)} train sequences, max_len={max(map(len,sentences)) if sentences else 0}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    # Determine N: prefer caller-supplied n_items, else infer from observed max
    seen_ids: set[str] = set()
    for s in sentences:
        seen_ids.update(s)
    observed_max = max((int(t) for t in seen_ids), default=-1)
    n = n_items if n_items is not None else (observed_max + 1)
    if n <= 0:
        raise RuntimeError("no interactions found in train CSV")

    arr = np.zeros((n, dim), dtype=np.float32)
    key_index = model.wv
    got = 0
    for i in range(n):
        key = str(i)
        if key in key_index:
            arr[i] = key_index[key]
            got += 1
    print(f"[item2vec] wrote cf.npy shape={arr.shape}; items_with_vector={got}/{n}")

    os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
    np.save(output_npy, arr)
    return arr


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Item2Vec (SGNS) collaborative embedding trainer")
    ap.add_argument("--train_csv", required=True, help="path to the train split CSV")
    ap.add_argument("--output_npy", required=True, help="path to write cf.npy")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_items", type=int, default=None, help="optional item count (rows); else inferred")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    train_item2vec(
        train_csv=args.train_csv,
        output_npy=args.output_npy,
        dim=args.dim,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        seed=args.seed,
        n_items=args.n_items,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
