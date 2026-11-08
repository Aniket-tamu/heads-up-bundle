# Original lines 1-12 are outside the supplied photos.











import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "plo5bot", "data")
SEED = 20261107

# Sample sizes (the runtime cost of the flop and turn features grows with these).
M_FLOP = 2          # opponents per flop runout (all 378 runouts are enumerated)
M_TURN = 96         # opponents per river card (all 27 river cards are enumerated)
RIVER_OPP = 65780   # the river feature is exact: all C(26, 5) opponent holdings

# Bucket grid.
PF_BUCKETS = 256
FLOP_EQ, FLOP_BUCKETS = 16, 128     # equity bins, total buckets (HS_now bins spread over the rows)
TURN_EQ, TURN_BUCKETS = 16, 128
RIVER_BUCKETS = 128
GAMMA_EQ = 1.5       # quantile levels 1 - (1 - i/n)^gamma: finer bins for strong hands
GAMMA_RIVER = 1.25


def _pick(rng, allowed, rows):
    """`rows` sorted 5-subsets of the positions in `allowed`."""
    allowed = np.asarray(allowed, dtype=np.uint8)
    pick = np.argsort(rng.random((rows, len(allowed))), axis=1)[:, :5]
    return np.sort(allowed[pick], axis=1)


def build_tables():
    rng = np.random.default_rng(SEED)
    runs = list(combinations(range(28), 2))
    flop = np.stack([_pick(rng, [p for p in range(28) if p not in (i, j)], M_FLOP) for i, j in runs])
    turn = np.stack([_pick(rng, [p for p in range(27) if p != j], M_TURN) for j in range(27)])
    for name, arr in (("abs_flop_opp", flop), ("abs_turn_opp", turn)):
        np.save(os.path.join(DATA, name + ".npy"), arr.astype(np.uint8), allow_pickle=False)
        print(f"{name}: {arr.shape}")


def weighted_kmeans(x, w, k, iters, seed):
    """Lloyd's algorithm with k-means++ seeding; returns labels."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.choice(n, p=w / w.sum())]]
    d2 = ((x - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = w * d2
        c = x[rng.choice(n, p=p / p.sum())]
        centers.append(c)
        d2 = np.minimum(d2, ((x - c) ** 2).sum(axis=1))
    centers = np.array(centers)
    xx = (x ** 2).sum(axis=1)
    labels = None
    for it in range(iters):
        dist = xx[:, None] - 2 * x @ centers.T + (centers ** 2).sum(axis=1)[None, :]
        new = dist.argmin(axis=1)
        if labels is not None and (new == labels).all():
            break
        labels = new
        sw = np.bincount(labels, weights=w, minlength=k)
        for j in range(x.shape[1]):
            centers[:, j] = np.bincount(labels, weights=w * x[:, j], minlength=k) / np.maximum(sw, 1e-12)
        empty = sw == 0
        if empty.any():                             # re-seed empty clusters on the worst-fit points
            far = np.argsort(-dist[np.arange(n), labels])[:int(empty.sum())]
            centers[empty] = x[far]
    inertia = float((w * dist[np.arange(n), labels]).sum() / w.sum())
    return labels, inertia, it + 1


def build_preflop():
    band = np.load(os.path.join(DATA, "preflop_eq_bucket.npy"), allow_pickle=False).astype(np.float64)
    eq = np.load(os.path.join(DATA, "preflop_eq.npy"), allow_pickle=False).astype(np.float64)
    cnt = np.load(os.path.join(DATA, "preflop_class_count.npy"), allow_pickle=False).astype(np.float64)
    t0 = time.perf_counter()
    labels, inertia, its = weighted_kmeans(band, cnt, PF_BUCKETS, 100, SEED)
    sw = np.bincount(labels, weights=cnt, minlength=PF_BUCKETS)
    mean_eq = np.bincount(labels, weights=cnt * eq, minlength=PF_BUCKETS) / np.maximum(sw, 1e-12)
    order = np.argsort(mean_eq, kind="stable")
    rename = np.empty(PF_BUCKETS, dtype=np.int64)
    rename[order] = np.arange(PF_BUCKETS)
    pf = rename[labels].astype(np.uint16)
    np.save(os.path.join(DATA, "abs_pf_bucket.npy"), pf, allow_pickle=False)
    share = np.bincount(pf, weights=cnt, minlength=PF_BUCKETS) / cnt.sum()
    within = np.sqrt(inertia)
    print(f"preflop: {PF_BUCKETS} buckets, {its} iterations, rms distance to center {within:.4f}, "
          f"hand share min {share.min():.5f} max {share.max():.5f} ({time.perf_counter() - t0:.1f}s)")


def _greedy_starts(mass, n, gamma):
    """Start indices (into the distinct values) of bins 1..n-1, filled greedily from the bottom.

    Target masses follow the levels 1 - (1 - i/n)^gamma; each bin takes whole distinct
    values until it holds its share of the mass still unassigned (a bin under half its
    share also takes the next value, even a heavy tie); every bin keeps one distinct value.
    """
    levels = np.concatenate([[0.0], 1.0 - (1.0 - np.arange(1, n) / n) ** gamma, [1.0]])
    w = np.diff(levels)
    starts, k, rem_mass, rem_w = [], 0, float(mass.sum()), 1.0
    for b in range(n - 1):
        target = w[b] / rem_w * rem_mass
        limit = len(mass) - (n - 1 - b)
        acc, start = 0.0, k
        while k < limit and (k == start or acc + mass[k] / 2 <= target or acc < 0.5 * target):
            acc += mass[k]
            k += 1
        starts.append(k)
        rem_mass -= acc
        rem_w -= w[b]
    return starts


def _rebalance(mass, starts, n, gamma, min_frac):
    """Merge sliver bins away and split the fullest bins instead, keeping the bin count.

    A bin's fill is its mass over the target mass at its position, (gamma / n) * (1 - x)^(1 - 1/gamma)
    with x its central cumulative mass. While the emptiest bin is below min_frac, it is merged
    into its lighter neighbour and the fullest bin with 2+ distinct values is split at its mass
    median (only if both halves end up fuller than the sliver was).
    """
    cum = np.concatenate([[0.0], np.cumsum(mass)])

    def fill(lo, hi):
        x = min(0.5 * (cum[lo] + cum[hi]), 1.0 - 1e-9)
        return (cum[hi] - cum[lo]) / ((gamma / n) * (1.0 - x) ** (1.0 - 1.0 / gamma))

    starts = list(starts)
    for _ in range(4 * (len(starts) + 1)):
        bounds = [0] + starts + [len(mass)]
        fills = [fill(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        bm = np.diff(cum[bounds])
        small = int(np.argmin(fills))
        if fills[small] >= min_frac:
            break
        if small == 0:
            drop = 0
        elif small == len(bm) - 1:
            drop = small - 1
        else:
            drop = small - 1 if bm[small - 1] <= bm[small + 1] else small
        trial = starts[:drop] + starts[drop + 1:]
        bounds = [0] + trial + [len(mass)]
        best = None
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < 2:
                continue
            cut = min(max(int(np.searchsorted(cum, 0.5 * (cum[lo] + cum[hi]))), lo + 1), hi - 1)
            worst = min(fill(lo, cut), fill(cut, hi))
            if best is None or worst > best[0]:
                best = (worst, cut)
        if best is None or best[0] <= fills[small]:    # no split beats keeping the sliver
            break
        starts = sorted(trial + [best[1]])
    return starts


def quantile_cuts(values, n, gamma, min_frac=0.3):
    """n-1 strictly ascending integer cuts; bin b = number of cuts <= value (searchsorted right).

    Bin b holds the values in [cut[b-1], cut[b]); masses follow 1 - (1 - i/n)^gamma
    (gamma > 1: finer bins for strong hands). Tied values are never split: a top value
    heavier than 1.5 top bins ("beats every opponent") gets a bin of its own; then bins
    below min_frac / n of the mass are merged away and the heaviest bins split instead.
    """
    u, c = np.unique(np.asarray(values), return_counts=True)
    if len(u) <= n:                                 # one bin per value; extra top bins stay empty
        return np.concatenate([u[1:], u[-1] + 1 + np.arange(n - len(u))]).astype(np.int32)
    mass = c / c.sum()
    top_w = 1.0 - (1.0 - (n - 1) / n) ** gamma
    top_w = 1.0 - top_w
    if n > 2 and mass[-1] > 1.5 * top_w:
        starts = _greedy_starts(mass[:-1] / mass[:-1].sum(), n - 1, gamma) + [len(u) - 1]
    else:
        starts = _greedy_starts(mass, n, gamma)
    starts = _rebalance(mass, starts, n, gamma, min_frac)
    return u[starts].astype(np.int32)


SENTINEL = 2 ** 30         # pads the per-row now-cuts (no feature reaches it)


def grid_cuts(eq, now, n_eq, n_total, gamma):
    """Equity bins, then n_b HS_now bins inside equity bin b with sum(n_b) = n_total.

    n_b grows like (mass_b * var_b)^(1/3), the MSE-optimal split for quantising HS_now
    (rows near the nuts have almost no HS_now spread and get 1-2 bins). Returns
    (eq_cuts (n_eq-1,), now_cuts (n_eq, max n_b - 1) padded with SENTINEL, offsets (n_eq,)).
    """
    eq_cuts = quantile_cuts(eq, n_eq, gamma)
    eb = np.searchsorted(eq_cuts, eq, side="right")
    mass = np.bincount(eb, minlength=n_eq) / len(eq)
    var = np.array([now[eb == b].astype(np.float64).var() if (eb == b).any() else 0.0 for b in range(n_eq)])
    distinct = np.array([len(np.unique(now[eb == b])) for b in range(n_eq)])
    score = np.cbrt(mass * var) + 1e-12
    alloc = np.ones(n_eq, dtype=np.int64)
    while alloc.sum() < n_total:                    # greedy: the row with most score per bucket
        gain = np.where(alloc < distinct, score / alloc, -1.0)
        if gain.max() < 0:
            break
        alloc[int(np.argmax(gain))] += 1
    width = max(1, int(alloc.max()) - 1)
    now_cuts = np.full((n_eq, width), SENTINEL, dtype=np.int32)
    for b in range(n_eq):
        if alloc[b] > 1:
            now_cuts[b, :alloc[b] - 1] = quantile_cuts(now[eb == b], int(alloc[b]), 1.0)
    offsets = np.concatenate([[0], np.cumsum(alloc)[:-1]]).astype(np.int32)
    return eq_cuts, now_cuts, offsets, int(alloc.sum())


def calibrate(n_deals, workers):
    sys.path.insert(0, os.path.join(ROOT, "dev", "cfr"))
    import cfeat
    from plo5bot import abstraction
    t0 = time.perf_counter()
    cache = os.path.join(ROOT, "dev", "cfr", "out", f"calib_features_{n_deals}.npy")
    if os.path.exists(cache):
        feats = np.load(cache, allow_pickle=False)
    else:
        rng = np.random.default_rng(SEED + 1)
        cards = cfeat.random_deals(n_deals, rng)
        feats, _ = cfeat.features(cards, workers)
        np.save(cache, feats, allow_pickle=False)
    print(f"calibration features for {n_deals} deals in {time.perf_counter() - t0:.1f}s")
    f = feats.reshape(-1, feats.shape[-1])            # both seats: (2N, NFEAT)
    counts = {"preflop": PF_BUCKETS}
    alloc = {}
    # second feature inside each equity row: E[HS^2] on both streets (dev/cfr/eval_vs_strong.py)
    for street, ne, nt, ie, inow in (("flop", FLOP_EQ, FLOP_BUCKETS, cfeat.F_FLOP_EQ, cfeat.F_FLOP_SQR),
                                    ("turn", TURN_EQ, TURN_BUCKETS, cfeat.F_TURN_EQ, cfeat.F_TURN_SQ)):
        eq_cuts, now_cuts, offsets, total = grid_cuts(f[:, ie], f[:, inow], ne, nt, GAMMA_EQ)
        np.save(os.path.join(DATA, f"abs_{street}_eq_cuts.npy"), eq_cuts, allow_pickle=False)
        np.save(os.path.join(DATA, f"abs_{street}_now_cuts.npy"), now_cuts, allow_pickle=False)
        np.save(os.path.join(DATA, f"abs_{street}_offsets.npy"), offsets, allow_pickle=False)
        counts[street] = total
        alloc[street] = np.diff(np.append(offsets, total)).tolist()
    river_cuts = quantile_cuts(f[:, cfeat.F_RIVER_EQ], RIVER_BUCKETS, GAMMA_RIVER)
    np.save(os.path.join(DATA, "abs_river_cuts.npy"), river_cuts, allow_pickle=False)
    counts["river"] = RIVER_BUCKETS
    meta = {
        "version": 1,
        "counts": counts,
        "eq_bins": {"flop": FLOP_EQ, "turn": TURN_EQ},
        "second_feature": {"flop": "flop_sqr (E[HS_river^2])", "turn": "turn_sq (E[HS^2])"},
        "features": ["pf_class", "flop_eq2", "flop_now2", "turn_eq2", "turn_now2", "turn_sq", "river_eq2",
                     "flop_sqt", "flop_sqr"],
        "second_bins_per_eq_bin": alloc,
        "samples": {"flop_runouts": 378, "flop_opp_per_runout": M_FLOP, "turn_rivers": 27,
                    "turn_opp_per_river": M_TURN, "river_opp": "all 65780 (exact)"},
        "totals": {"flop": 378 * M_FLOP, "turn": 27 * M_TURN, "river": RIVER_OPP},
        "gamma": {"eq": GAMMA_EQ, "river": GAMMA_RIVER},
        "calibration": {"deals": n_deals, "views_per_street": 2 * n_deals, "seed": SEED + 1},
        "seed": SEED,
        "files": {
            "preflop": "abs_pf_bucket.npy (uint16[19998], by preflop_class.npy class id)",
            "flop": "abs_flop_opp.npy, abs_flop_eq_cuts.npy, abs_flop_now_cuts.npy, abs_flop_offsets.npy",
            "turn": "abs_turn_opp.npy, abs_turn_eq_cuts.npy, abs_turn_now_cuts.npy, abs_turn_offsets.npy",
            "river": "abs_river_cuts.npy",
        },
    }
    with open(os.path.join(DATA, "abstraction_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    abstraction.reload()
    b = abstraction.buckets_from_features(feats)
    for s, name in enumerate(("preflop", "flop", "turn", "river")):
        cnt = np.bincount(b[:, :, s].ravel(), minlength=counts[name])
        sh = np.sort(cnt / cnt.sum())
        print(f"{name}: {counts[name]} buckets, used {int((cnt > 0).sum())}, "
              f"share min {sh[0]:.5f} p10 {sh[len(sh) // 10]:.5f} median {sh[len(sh) // 2]:.5f} max {sh[-1]:.5f}")
    for street in alloc:
        print(f"{street} second-feature bins per equity row: {alloc[street]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "tables", "preflop", "calibrate"])
    ap.add_argument("--calib", type=int, default=400_000, help="calibration deals (2 views each)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.stage in ("all", "tables"):
        build_tables()
    if args.stage in ("all", "preflop"):
        build_preflop()
    if args.stage in ("all", "calibrate"):
        calibrate(args.calib, args.workers)


if __name__ == "__main__":
    main()
