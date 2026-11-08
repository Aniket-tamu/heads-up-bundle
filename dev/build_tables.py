"""Build plo5bot/data/pair_triple_rank.npy and dense_scores.npy from evaluator.py.

Run from the repo root: python dev/build_tables.py
"""

import os
import sys
import time
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluator  # noqa: E402

OUT = os.environ.get("PLO5_DATA_OUT") or os.path.join(ROOT, "plo5bot", "data")  # override to rebuild elsewhere

# COMB[n, k] = C(n, k) for n < 37, k < 6.
COMB = np.zeros((37, 6), dtype=np.int64)
for n in range(37):
    for k in range(6):
        COMB[n, k] = 1 if k == 0 else (0 if n < k else COMB[n - 1, k - 1] + COMB[n - 1, k])


def colex5(sorted5):
    """Colex index of sorted 5-card rows (last axis has 5 ascending cards)."""
    return sum(COMB[sorted5[..., k], k + 1] for k in range(5))


def main():
    t0 = time.perf_counter()
    table = evaluator.get_table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    masks = (np.int64(1) << combos).sum(axis=1)
    scores = np.fromiter((table[int(m)] for m in masks), dtype=np.int64, count=len(masks))
    uniq = np.unique(scores)
    assert len(uniq) == 1404, len(uniq)
    rank5 = np.empty(len(combos), dtype=np.uint16)
    rank5[colex5(combos)] = np.searchsorted(uniq, scores)

    pairs = np.array([(a, b) for b in range(36) for a in range(b)], dtype=np.int16)
    triples = np.array([(a, b, c) for c in range(36) for b in range(c) for a in range(b)], dtype=np.int16)
    five = np.concatenate([
        np.broadcast_to(pairs[:, None, :], (630, 7140, 2)),
        np.broadcast_to(triples[None, :, :], (630, 7140, 3)),
    ], axis=2).copy()
    five.sort(axis=2)
    overlap = (five[..., 1:] == five[..., :-1]).any(axis=2)
    five[overlap] = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    pt = (rank5[colex5(five.astype(np.int64))].astype(np.int32) + 1)
    pt[overlap] = 0
    pt = pt.astype(np.uint16)

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "pair_triple_rank.npy"), pt, allow_pickle=False)
    np.save(os.path.join(OUT, "dense_scores.npy"), uniq.astype(np.int32), allow_pickle=False)
    print(f"built in {time.perf_counter() - t0:.2f}s: PT {pt.shape} {pt.dtype}, "
          f"{int((pt > 0).sum())} valid entries, 1404 ranks")


if __name__ == "__main__":
    main()
