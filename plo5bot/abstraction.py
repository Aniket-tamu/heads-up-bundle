"""Imperfect-recall card abstraction for CFR: one bucket id per (street, hole, board).

Every feature is an integer count (wins 2, ties 1: "2 x equity" units). The flop and
turn use FIXED opponent-sample tables and the river is exact, so the bucket is a pure
function of the card sets (no RNG at runtime) and dev/cfr/features.c reproduces it
bit for bit.

preflop  abs_pf_bucket[preflop class]: 256 k-means clusters of the 20-hand equity vectors
flop     E[HS] and E[HS^2] (unbiased product of the 2 opponent results per runout)
         over all 378 (turn, river) runouts x 2 sampled opponents each; 16 equity rows,
         each split into 1..n E[HS^2] bins, 128 buckets in all
turn     E[HS] and E[HS^2] over all 27 river cards x 96 sampled opponents each (HS_now on
         the turn board is computed too, unused); 16 equity rows split by E[HS^2], 128 buckets
river    exact HS against all 65,780 opponent holdings; 128 buckets

"Live" cards are the cards not in our hand and not on the current board, sorted
ascending; sample tables hold positions into that list. Bucket ids grow with
strength: preflop by equity; flop/turn id = offset[eq_bin] + bin of E[HS^2] inside that
equity row; river by HS. (HS_now and a turn-card potential are extra, unused features.)
See dev/cfr/ABSTRACTION.md for the exact definitions and file formats.
"""

import json
import os
from itertools import combinations
from math import comb

import numpy as np

from . import cards
from .equity import P10, PID, T10, TID

DATA = cards.DATA_DIR
PT = cards.PT
PT_FLAT = PT.ravel()
PTT = np.ascontiguousarray(PT.T)
META_FILE = os.path.join(DATA, "abstraction_meta.json")

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
# Feature vector layout (also the columns of dev/cfr/out/pool_features.npy).
F_PF, F_FLOP_EQ, F_FLOP_NOW, F_TURN_EQ, F_TURN_NOW, F_TURN_SQ, F_RIVER_EQ, F_FLOP_SQT, F_FLOP_SQR = range(9)
NFEAT = 9

PF_CLASS = np.load(os.path.join(DATA, "preflop_class.npy"), allow_pickle=False)
COMB = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)


def _local_pairs(n):
    """Colex-ordered position pairs (i < j) of an n-card live list."""
    lp = [(i, j) for j in range(n) for i in range(j)]
    return np.array([i for i, _ in lp], dtype=np.int64), np.array([j for _, j in lp], dtype=np.int64)


def _local_pair_ids(pos):
    """Local pair ids (colex) of the 10 pairs of each sorted 5-position row. Shape (..., 10)."""
    a, b = pos[..., P10[:, 0]].astype(np.int64), pos[..., P10[:, 1]].astype(np.int64)
    return b * (b - 1) // 2 + a


LPI28, LPJ28 = _local_pairs(28)
LPI27, LPJ27 = _local_pairs(27)
LPI26, LPJ26 = _local_pairs(26)
RUN28 = np.array(list(combinations(range(28), 2)), dtype=np.int64)
T4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)
P4 = np.array(list(combinations(range(4), 2)), dtype=np.int64)

FLOP_OPP = np.load(os.path.join(DATA, "abs_flop_opp.npy"), allow_pickle=False)
TURN_OPP = np.load(os.path.join(DATA, "abs_turn_opp.npy"), allow_pickle=False)
FLOP_LP = _local_pair_ids(FLOP_OPP)
TURN_LP = _local_pair_ids(TURN_OPP)
TURN_FLAT = np.arange(27)[:, None, None] * 351 + TURN_LP
TOTALS = {FLOP: FLOP_OPP.shape[0] * FLOP_OPP.shape[1], TURN: 27 * TURN_OPP.shape[1],
          RIVER: comb(26, 5)}

_TABLES = {}


def reload():
    """(Re)load the bucket cut points and meta written by dev/cfr/build_abstraction.py."""
    _TABLES.clear()
    with open(META_FILE) as fh:
        meta = json.load(fh)
    t = {"meta": meta, "counts": [meta["counts"][s] for s in ("preflop", "flop", "turn", "river")],
         "pf": np.load(os.path.join(DATA, "abs_pf_bucket.npy"), allow_pickle=False),
         "river": np.load(os.path.join(DATA, "abs_river_cuts.npy"), allow_pickle=False).astype(np.int64)}
    for s in ("flop", "turn"):
        t[s + "_eq"] = np.load(os.path.join(DATA, f"abs_{s}_eq_cuts.npy"), allow_pickle=False).astype(np.int64)
        t[s + "_now"] = np.load(os.path.join(DATA, f"abs_{s}_now_cuts.npy"), allow_pickle=False).astype(np.int64)
        t[s + "_off"] = np.load(os.path.join(DATA, f"abs_{s}_offsets.npy"), allow_pickle=False).astype(np.int64)
    _TABLES.update(t)
    return t


def _tables():
    return _TABLES if _TABLES else reload()


def num_buckets(street):
    return _tables()["counts"][int(street)]


# ----------------------------- features

def _live(hole, board):
    mask = np.ones(36, dtype=bool)
    mask[hole] = False
    mask[board] = False
    return np.flatnonzero(mask)


def _pairs(hole):
    return PID[hole[P10[:, 0]], hole[P10[:, 1]]]


def _score(hv, ov):
    """2 x wins + ties of hero value(s) hv against opponent values ov."""
    return 2 * int(np.count_nonzero(hv > ov)) + int(np.count_nonzero(hv == ov))


def preflop_class(hole):
    """Suit-isomorphism class (0..19997) of a 5-card hand; vectorised over (..., 5) arrays."""
    h = np.sort(np.asarray(hole, dtype=np.int64), axis=-1)
    idx = sum(COMB[h[..., k], k + 1] for k in range(5))
    return PF_CLASS[idx].astype(np.int64)


def flop_features(hole, board, step=1, extras=True):
    """(eq2, now2, sqt, sqr) on a 3-card board. step > 1 uses every step-th runout (approximate).

    With w = 2 x win + tie per (runout, opponent) and s_u the runout sums:
    sqt = sum over turn cards of (S_p^2 - sum of s_u^2 over runouts u holding p), S_p = sum of
    those s_u: an unbiased multiple of E_turn[HS_turn^2]; sqr = sum over runouts of the products
    w_a * w_b of its opponent pairs: an unbiased multiple of E[HS_river^2]. extras=False skips
    now2 and sqt (returned as -1), which the buckets do not use.
    """
    hole = np.asarray(hole, dtype=np.int64)
    b = np.asarray(board[:3], dtype=np.int64)
    live = _live(hole, b)
    hp = _pairs(hole)
    lp = PID[live[LPI28], live[LPJ28]]
    rows = slice(None, None, step)
    run = RUN28[rows]
    b5 = np.empty((len(run), 5), dtype=np.int64)
    b5[:, :3] = b
    b5[:, 3:] = live[run]
    tids = TID[b5[:, T10[:, 0]], b5[:, T10[:, 1]], b5[:, T10[:, 2]]]
    m = FLOP_LP.shape[1]
    pairs = np.empty((len(run), 10 * (1 + m)), dtype=np.int64)
    pairs[:, :10] = hp
    pairs[:, 10:] = lp[FLOP_LP[rows]].reshape(len(run), -1)
    v = np.take(PT_FLAT, pairs[:, :, None] * cards.NUM_TRIPLES + tids[:, None, :])
    v = v.reshape(len(run), 1 + m, 100).max(axis=2)
    w = 2 * (v[:, :1] > v[:, 1:]) + (v[:, :1] == v[:, 1:])
    su = w.sum(axis=1)
    eq2 = int(su.sum())
    sqr = int(((su * su - (w * w).sum(axis=1)) // 2).sum())
    now2 = sqt = -1
    if extras:
        S = np.bincount(run[:, 0], su, 28) + np.bincount(run[:, 1], su, 28)
        T = np.bincount(run[:, 0], su * su, 28) + np.bincount(run[:, 1], su * su, 28)
        sqt = int(round(float((S * S - T).sum())))
        pf = PTT[TID[b[0], b[1], b[2]]]
        now2 = _score(pf[hp].max(), pf[lp[FLOP_LP[rows]]].max(axis=2))
    if step > 1:
        scale = TOTALS[FLOP] / (len(run) * m)
        eq2, sqr = int(round(eq2 * scale)), int(round(sqr * scale))
        if extras:
            now2, sqt = int(round(now2 * scale)), int(round(sqt * scale * scale))
    return eq2, now2, sqt, sqr


def turn_features(hole, board, extras=True):
    """(eq2, now2, sq) on a 4-card board; sq = sum over river cards of (that river's eq2)^2.

    extras=False skips now2 (returned as -1), which the buckets do not use.
    """
    hole = np.asarray(hole, dtype=np.int64)
    b = np.asarray(board[:4], dtype=np.int64)
    live = _live(hole, b)
    hp = _pairs(hole)
    lp = PID[live[LPI27], live[LPJ27]]
    base4 = PTT[TID[b[T4[:, 0]], b[T4[:, 1]], b[T4[:, 2]]]].max(axis=0)
    tid6 = TID[b[P4[:, 0]][None, :], b[P4[:, 1]][None, :], live[:, None]]
    pbv = np.maximum(PTT[tid6.ravel()].reshape(27, 6, 630).max(axis=1), base4)
    hv = pbv[:, hp].max(axis=1)
    ov = np.take(pbv[:, lp], TURN_FLAT).max(axis=2)
    w2 = 2 * (hv[:, None] > ov).sum(axis=1) + (hv[:, None] == ov).sum(axis=1)
    now2 = _score(base4[hp].max(), base4[lp[TURN_LP]].max(axis=2)) if extras else -1
    return int(w2.sum()), now2, int((w2 * w2).sum())


# River: exact counts over all C(26, 5) = 65,780 opponent holdings. An opponent holding
# is below value v iff none of its 10 pairs reaches v, i.e. it is an independent 5-set of
# the graph on the 26 live cards whose edges are the pairs worth >= v. Each independent
# 5-set is counted once: its 4 lowest cards (an independent 4-set) plus one higher card
# adjacent to none of them.
_Q4 = np.array(list(combinations(range(26), 4)), dtype=np.int64)
Q4I, Q4J, Q4K, Q4L = (np.ascontiguousarray(_Q4[:, c]) for c in range(4))
BITS_26 = np.int64(1) << np.arange(26, dtype=np.int64)
NEED_I = BITS_26[Q4J] | BITS_26[Q4K] | BITS_26[Q4L]
NEED_J = BITS_26[Q4K] | BITS_26[Q4L]
NEED_K = BITS_26[Q4L]
ABOVE_L = ((np.int64(1) << 26) - 1) ^ ((BITS_26[Q4L] << 1) - 1)
POP13 = np.array([bin(i).count("1") for i in range(1 << 13)], dtype=np.int64)


def _popcount(m):
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(m).sum())
    return int((POP13[m & 8191].sum() + POP13[m >> 13].sum()))


def _independent5(edge):
    """Number of 5-subsets of the 26 live cards containing no pair with edge[local pair id] set."""
    adj = np.zeros((26, 26), dtype=bool)
    adj[LPI26, LPJ26] = edge
    adj |= adj.T
    nonadj = ((~adj).astype(np.int64) @ BITS_26) - BITS_26
    ni, nj, nk = nonadj[Q4I], nonadj[Q4J], nonadj[Q4K]
    ok = ((ni & NEED_I) == NEED_I) & ((nj & NEED_J) == NEED_J) & ((nk & NEED_K) == NEED_K)
    m = ni & nj & nk & nonadj[Q4L] & ABOVE_L
    return _popcount(m[ok])


def river_features(hole, board):
    """eq2 = 2 x wins + ties against every opponent holding (exact, out of 2 x 65,780)."""
    hole = np.asarray(hole, dtype=np.int64)
    b = np.asarray(board[:5], dtype=np.int64)
    live = _live(hole, b)
    pb = PTT[TID[b[T10[:, 0]], b[T10[:, 1]], b[T10[:, 2]]]].max(axis=0)
    lv = pb[PID[live[LPI26], live[LPJ26]]]
    hv = pb[_pairs(hole)].max()
    below = _independent5(lv >= hv)
    if not (lv == hv).any():
        return 2 * below
    return below + _independent5(lv > hv)


# ----------------------------- buckets

def _grid(t, street, eq2, second):
    """offset[eq_bin] + number of that row's second-feature cuts <= second (rows padded with 2**30)."""
    eq_bin = np.searchsorted(t[street + "_eq"], eq2, side="right")
    sub = (np.asarray(second)[..., None] >= t[street + "_now"][eq_bin]).sum(axis=-1)
    return t[street + "_off"][eq_bin] + sub


def buckets_from_features(f):
    """Integer features (..., NFEAT) -> buckets (..., 4) as uint16 (preflop, flop, turn, river)."""
    t = _tables()
    f = np.asarray(f, dtype=np.int64)
    out = np.empty(f.shape[:-1] + (4,), dtype=np.uint16)
    out[..., 0] = t["pf"][f[..., F_PF]]
    out[..., 1] = _grid(t, "flop", f[..., F_FLOP_EQ], f[..., F_FLOP_SQR])
    out[..., 2] = _grid(t, "turn", f[..., F_TURN_EQ], f[..., F_TURN_SQ])
    out[..., 3] = np.searchsorted(t["river"], f[..., F_RIVER_EQ], side="right")
    return out


FAST_BELOW_S = 0.0006  # below this budget the flop uses every 4th runout (approximate bucket)


def bucket(street, hole, board, rng=None, budget_s=None):
    """Bucket id of our hand on this street. Deterministic in the card sets; rng is unused."""
    t = _tables()
    street = int(street)
    if street == PREFLOP:
        return int(t["pf"][preflop_class(hole)])
    if street == FLOP:
        step = 4 if (budget_s is not None and budget_s < FAST_BELOW_S) else 1
        eq2, _, _, sqr = flop_features(hole, board, step, extras=False)
        return int(_grid(t, "flop", eq2, sqr))
    if street == TURN:
        eq2, _, sq = turn_features(hole, board, extras=False)
        return int(_grid(t, "turn", eq2, sq))
    return int(np.searchsorted(t["river"], river_features(hole, board), side="right"))


def features(hole, board):
    """All features available with this board (later-street entries stay -1)."""
    f = np.full(NFEAT, -1, dtype=np.int64)
    f[F_PF] = preflop_class(hole)
    if len(board) >= 3:
        f[F_FLOP_EQ], f[F_FLOP_NOW], f[F_FLOP_SQT], f[F_FLOP_SQR] = flop_features(hole, board)
    if len(board) >= 4:
        f[F_TURN_EQ], f[F_TURN_NOW], f[F_TURN_SQ] = turn_features(hole, board)
    if len(board) >= 5:
        f[F_RIVER_EQ] = river_features(hole, board)
    return f
