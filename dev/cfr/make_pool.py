"""Generate a pre-bucketed deal pool for the CFR trainer (see dev/cfr/ABSTRACTION.md).

Writes to dev/cfr/out/ (plain .npy, no pickles, nothing imports plo5bot to read them):
  pool_cards.npy     uint8  (N, 15)   seat 0 hole (sorted) | seat 1 hole (sorted) | board (flop 3, turn, river)
  pool_buckets.npy   uint16 (N, 2, 4)  [deal, seat, street] bucket id (street 0 preflop .. 3 river)
  pool_winner.npy    int8   (N,)       showdown on the full board: +1 seat 0 wins, -1 seat 1 wins, 0 tie
  pool_features.npy  int32  (N, 2, 9)  raw integer features (re-bucket without recomputing); --no-features skips
  pool_meta.json     sizes, seed, bucket counts, abstraction meta

python dev/cfr/make_pool.py --n 5000000 [--seed 1] [--workers 16] [--chunk 250000] [--out DIR]
python dev/cfr/make_pool.py --rebucket [--out DIR]  # new cuts, same deals: rewrite pool_buckets.npy
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cfeat  # noqa: E402
from plo5bot import abstraction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--rebucket", action="store_true", help="re-bucket an existing pool from pool_features.npy")
    args = ap.parse_args()
    if args.rebucket:
        return rebucket(args.out)
    os.makedirs(args.out, exist_ok=True)
    n = args.n
    mm = np.lib.format.open_memmap
    cards_f = mm(os.path.join(args.out, "pool_cards.npy"), mode="w+", dtype=np.uint8, shape=(n, 15))
    buck_f = mm(os.path.join(args.out, "pool_buckets.npy"), mode="w+", dtype=np.uint16, shape=(n, 2, 4))
    win_f = mm(os.path.join(args.out, "pool_winner.npy"), mode="w+", dtype=np.int8, shape=(n,))
    feat_f = None if args.no_features else mm(os.path.join(args.out, "pool_features.npy"), mode="w+",
                                              dtype=np.int32, shape=(n, 2, abstraction.NFEAT))

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for s in range(0, n, args.chunk):
        m = min(args.chunk, n - s)
        deals = cfeat.random_deals(m, rng)
        feats, winner = cfeat.features(deals, args.workers)
        cards_f[s:s + m] = deals
        buck_f[s:s + m] = abstraction.buckets_from_features(feats)
        win_f[s:s + m] = winner
        if feat_f is not None:
            feat_f[s:s + m] = feats
        done = s + m
        el = time.perf_counter() - t0
        print(f"  {done}/{n} deals, {el:.0f}s, {done / el:.0f} deals/s", flush=True)
    for f in (cards_f, buck_f, win_f, feat_f):
        if f is not None:
            f.flush()
    b = np.asarray(buck_f)
    counts = [abstraction.num_buckets(k) for k in range(4)]
    usage = {name: int((np.bincount(b[:, :, k].ravel(), minlength=counts[k]) > 0).sum())
             for k, name in enumerate(("preflop", "flop", "turn", "river"))}
    w = np.asarray(win_f)
    meta = {
        "n": n, "seed": args.seed, "generator": "numpy default_rng(seed): argsort of uniform(36) per deal",
        "files": files_doc(feat_f is not None),
        "bucket_counts": dict(zip(("preflop", "flop", "turn", "river"), counts)),
        "buckets_used": usage,
        "winner_share": {"seat0": float((w == 1).mean()), "tie": float((w == 0).mean()),
                         "seat1": float((w == -1).mean())},
        "abstraction_meta": abstraction._tables()["meta"],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(args.out, "pool_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: meta[k] for k in ("n", "bucket_counts", "buckets_used", "winner_share", "seconds")}))


def rebucket(out, chunk=1_000_000):
    t0 = time.perf_counter()
    feats = np.load(os.path.join(out, "pool_features.npy"), mmap_mode="r")
    buck = np.lib.format.open_memmap(os.path.join(out, "pool_buckets.npy"), mode="r+")
    for s in range(0, len(feats), chunk):
        buck[s:s + chunk] = abstraction.buckets_from_features(feats[s:s + chunk])
    buck.flush()
    with open(os.path.join(out, "pool_meta.json")) as fh:
        meta = json.load(fh)
    counts = [abstraction.num_buckets(k) for k in range(4)]
    b = np.asarray(buck)
    meta["bucket_counts"] = dict(zip(("preflop", "flop", "turn", "river"), counts))
    meta["buckets_used"] = {name: int((np.bincount(b[:, :, k].ravel(), minlength=counts[k]) > 0).sum())
                            for k, name in enumerate(("preflop", "flop", "turn", "river"))}
    meta["abstraction_meta"] = abstraction._tables()["meta"]
    meta["files"] = files_doc(True)
    with open(os.path.join(out, "pool_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"re-bucketed {len(feats)} deals in {time.perf_counter() - t0:.0f}s: {meta['buckets_used']}")


def files_doc(with_features):
    return {
        "pool_cards.npy": "uint8 (N, 15): [0:5] seat 0 hole sorted, [5:10] seat 1 hole sorted, "
                          "[10:13] flop, [13] turn, [14] river; card = 4 * rank + suit",
        "pool_buckets.npy": "uint16 (N, 2, 4): bucket of (deal, seat, street), street 0..3 = preflop..river",
        "pool_winner.npy": "int8 (N,): +1 seat 0 has the better showdown hand, -1 seat 1, 0 tie",
        "pool_features.npy": None if not with_features else
        "int32 (N, 2, {abstraction.NFEAT}): pf_class, flop_eq2, flop_now2, turn_eq2, turn_now2, turn_sq, "
        "river_eq2, flop_sqt, flop_sqr (see dev/cfr/ABSTRACTION.md)",
    }


if __name__ == "__main__":
    main()
