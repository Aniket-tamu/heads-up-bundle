"""Offline preflop tables for all 376,992 hole hands (indexed by colex of the sorted 5 cards).

Writes plo5bot/data/:
  preflop_class.npy       uint16[376992] suit-isomorphism class id (0..19997)
  preflop_eq.npy          float32[19998] all-in equity vs one uniformly random hand
  preflop_eq_bucket.npy   float32[19998, 20] equity against each 5% band
  preflop_band_edges.npy  float32[21] equity cut points of the 20 bands

Run from repo root: python dev/build_preflop.py [--samples 20000] [--workers 30]
"""
import argparse
import os
import sys
import time
from itertools import combinations, permutations
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from plo5bot import cards  # noqa: E402
OUT = os.environ.get("PLO5_DATA_OUT") or cards.DATA_DIR
NBANDS = 20
COMB = np.zeros((37, 6), dtype=np.int64)
for _n in range(37):
    for _k in range(6):
        COMB[_n, _k] = 1 if _k == 0 else (0 if _n < _k else COMB[_n - 1, _k - 1] + COMB[_n - 1, _k])
PID = np.full((36, 36), -1, dtype=np.int32)
for _a in range(36):
    for _b in range(36):
        if _a != _b:
            PID[_a, _b] = cards.pair_id(_a, _b)
TID = np.full((36, 36, 36), -1, dtype=np.int32)
for _a, _b, _c in permutations(range(36), 3):
    TID[_a, _b, _c] = cards.triple_id(_a, _b, _c)
P10 = np.array(list(combinations(range(5), 2)))
T10 = np.array(list(combinations(range(5), 3)))

def colex5(sorted5):
    return sum(COMB[sorted5[..., k], k + 1] for k in range(5))

def iso_classes():
    """Class id per colex index, plus one representative hand per class."""
    hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    idx = colex5(hands)
    best = None
    for perm in permutations(range(4)):
        p = np.array(perm)
        mapped = (hands >> 2) * 4 + p[hands & 3]
        mapped.sort(axis=1)
        c = colex5(mapped)
        best = c if best is None else np.minimum(best, c)
    canon, cls_of_hand = np.unique(best, return_inverse=True)
    cls = np.empty(len(hands), dtype=np.int16)
    cls[idx] = cls_of_hand
    reps = np.empty((len(canon), 5), dtype=np.int64)
    reps[cls_of_hand] = hands
    counts = np.bincount(cls_of_hand, minlength=len(canon))
    return cls, reps, counts

def _values(hole_pairs, opp, board):
    """Dense values (1..1404) of hero (fixed pairs) and opponents (S x 5) on boards (S x 5)."""
    tids = TID[board[:, T10[:, 0]], board[:, T10[:, 1]], board[:, T10[:, 2]]]
    opp_pairs = PID[opp[:, P10[:, 0]], opp[:, P10[:, 1]]]
    hv = cards.PT[hole_pairs[None, :, None], tids[:, None, :]].max(axis=(1, 2))
    ov = cards.PT[opp_pairs[:, :, None], tids[:, None, :]].max(axis=(1, 2))
    return hv, ov, opp

def _job(args):
    cid, hand, samples, band_of_class, cls_lookup = args
    rng = np.random.default_rng(1_000_003 + cid)
    rest = np.array([c for c in range(36) if c not in set(hand)], dtype=np.int64)
    hole_pairs = PID[hand[P10[:, 0]], hand[P10[:, 1]]]
    wins = np.zeros(NBANDS)
    games = np.zeros(NBANDS)
    total = 0.0
    done = 0
    while done < samples:
        s = min(4000, samples - done)
        pick = np.argsort(rng.random((s, 31)), axis=1)[:, :10]
        dealt = rest[pick]
        opp, board = np.sort(dealt[:, :5], axis=1), dealt[:, 5:]
        hv, ov, _ = _values(hole_pairs, opp, board)
        res = (hv > ov) + 0.5 * (hv == ov)
        total += res.sum()
        if band_of_class is not None:
            band = band_of_class[cls_lookup[colex5(opp)]]
            wins += np.bincount(band, weights=res, minlength=NBANDS)
            games += np.bincount(band, minlength=NBANDS)
        done += s
    return cid, total / samples, wins, games

def run(jobs, workers):
    import multiprocessing as mp
    out = []
    with mp.get_context("fork").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            out.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    return sorted(out, key=lambda r: r[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--band-samples", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cls, reps, counts = iso_classes()
    print(f"{len(reps)} classes ({time.perf_counter() - t0:.1f}s)", flush=True)
    
    res = run([(c, reps[c], args.samples, None, None) for c in range(len(reps))], args.workers)
    eq = np.array([r[1] for r in res], dtype=np.float64)
    print(f"pass 1 done ({time.perf_counter() - t0:.1f}s)", flush=True)
    
    order = np.argsort(eq, kind="stable")
    cum = np.cumsum(counts[order]) / counts.sum()
    band_sorted = np.minimum((cum * NBANDS - 1e-9).astype(np.int64), NBANDS - 1)
    band_of_class = np.empty(len(eq), dtype=np.int64)
    band_of_class[order] = band_sorted
    edges = [float(eq[order][0])] + [float(eq[order][band_sorted == b].max()) for b in range(NBANDS)]
    
    cls_lookup = cls.astype(np.int64)
    res2 = run([(c, reps[c], args.band_samples, band_of_class, cls_lookup) for c in range(len(reps))], args.workers)
    bucket = np.array([r[2] / np.maximum(r[3], 1) for r in res2], dtype=np.float32)
    print(f"pass 2 done ({time.perf_counter() - t0:.1f}s)", flush=True)
    
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "preflop_class.npy"), cls, allow_pickle=False)
    np.save(os.path.join(OUT, "preflop_eq.npy"), eq.astype(np.float32), allow_pickle=False)
    np.save(os.path.join(OUT, "preflop_eq_bucket.npy"), bucket, allow_pickle=False)
    np.save(os.path.join(OUT, "preflop_band_edges.npy"), np.array(edges, dtype=np.float32), allow_pickle=False)
    np.save(os.path.join(OUT, "preflop_class_count.npy"), counts.astype(np.int32), allow_pickle=False)
    w = counts / counts.sum()
    print(f"saved, weighted mean equity {float((eq * w).sum()):.4f} (should be ~0.5) "
          f"min {eq.min():.3f} max {eq.max():.3f}; total {time.perf_counter() - t0:.1f}s")

if __name__ == "__main__":
    main()
