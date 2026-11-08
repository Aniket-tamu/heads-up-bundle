"""Equity (P(win) + P(tie)/2) of our hand against one opponent holding.

Opponent holdings come from the live cards (not ours, not on the board),
either uniformly or as a weighted sample. All costs are a few hundred
microseconds to about 2 ms in numpy on one core.
"""

from itertools import combinations, permutations

import numpy as np

from . import cards

PT = cards.PT
PID = np.zeros((36, 36), dtype=np.int32)
for _a, _b in permutations(range(36), 2):
    PID[_a, _b] = cards.pair_id(_a, _b)
TID = np.zeros((36, 36, 36), dtype=np.int32)
for _a, _b, _c in permutations(range(36), 3):
    TID[_a, _b, _c] = cards.triple_id(_a, _b, _c)
P10 = np.array(list(combinations(range(5), 2)), dtype=np.int64)
T10 = np.array(list(combinations(range(5), 3)), dtype=np.int64)

_LP = [(i, j) for j in range(26) for i in range(j)]
_LPI = np.array([i for i, _ in _LP], dtype=np.int64)
_LPJ = np.array([j for _, j in _LP], dtype=np.int64)
_LOCAL_PAIR = {p: k for k, p in enumerate(_LP)}
_C26 = np.array(list(combinations(range(26), 5)), dtype=np.int64)
RIVER_OPP_PAIRS = np.ascontiguousarray(np.stack(
    [np.array([_LOCAL_PAIR[(int(a), int(b))] for a, b in zip(_C26[:, i], _C26[:, j])], dtype=np.int16)
     for i, j in combinations(range(5), 2)]))
RIVER_OPP_HANDS = _C26


def live_cards(hole, board):
    used = set(hole) | set(board)
    return np.array([c for c in range(36) if c not in used], dtype=np.int64)


def hero_pairs(hole):
    hole = np.asarray(hole, dtype=np.int64)
    return PID[hole[P10[:, 0]], hole[P10[:, 1]]]


def river_opp_values(hole, board):
    """(our value, values of all 65,780 opponent holdings, live cards) on a 5-card board."""
    pb = cards.board_pair_best(board)
    hv = int(pb[hero_pairs(hole)].max())
    live = live_cards(hole, board)
    local = pb[PID[live[_LPI], live[_LPJ]]]
    ov = local[RIVER_OPP_PAIRS[0]]
    for k in range(1, 10):
        np.maximum(ov, local[RIVER_OPP_PAIRS[k]], out=ov)
    return hv, ov, live


def river_equity(hole, board, weights=None):
    """Exact river equity against every opponent holding (optionally weighted, length 65,780)."""
    hv, ov, _ = river_opp_values(hole, board)
    res = (ov < hv) + 0.5 * (ov == hv)
    if weights is None:
        return float(res.mean())
    total = float(weights.sum())
    return float((res * weights).sum() / total) if total > 0 else float(res.mean())


def sample_opponents(live, n, rng):
    """n uniformly random 5-card holdings from the live cards (n x 5 cards)."""
    pick = np.argsort(rng.random((n, len(live))), axis=1)[:, :5]
    return live[pick]


def turn_equity(hole, board, n_opp=400, rng=None, opp=None):
    """Turn equity: every river card, against n_opp sampled (or given) opponent holdings."""
    rng = rng or np.random.default_rng(0)
    live = live_cards(hole, board)
    if opp is None:
        opp = sample_opponents(live, n_opp, rng)
    b4 = np.asarray(board, dtype=np.int64)
    boards = np.concatenate([np.broadcast_to(b4, (len(live), 4)), live[:, None]], axis=1)
    tids = TID[boards[:, T10[:, 0]], boards[:, T10[:, 1]], boards[:, T10[:, 2]]]
    pbr = PT[:, tids].max(axis=2).T
    hv = pbr[:, hero_pairs(hole)].max(axis=1)
    opp_pairs = PID[opp[:, P10[:, 0]], opp[:, P10[:, 1]]]
    ov = pbr[:, opp_pairs].max(axis=2)
    valid = ~(live[:, None, None] == opp[None, :, :]).any(axis=2)
    res = (ov < hv[:, None]) + 0.5 * (ov == hv[:, None])
    return float((res * valid).sum() / valid.sum())


def flop_equity(hole, board, n=800, rng=None, opp=None):
    """Flop equity by joint sampling of (opponent holding, turn, river)."""
    rng = rng or np.random.default_rng(0)
    live = live_cards(hole, board)
    if opp is None:
        pick = np.argsort(rng.random((n, len(live))), axis=1)[:, :7]
        dealt = live[pick]
        opp, runout = dealt[:, :5], dealt[:, 5:]
    else:
        n = len(opp)
        free = np.where((live[None, :, None] == opp[:, None, :]).any(axis=2), 2.0, rng.random((n, len(live))))
        runout = live[np.argsort(free, axis=1)[:, :2]]
    b = np.concatenate([np.broadcast_to(np.asarray(board, dtype=np.int64), (n, 3)), runout], axis=1)
    tids = TID[b[:, T10[:, 0]], b[:, T10[:, 1]], b[:, T10[:, 2]]]
    hv = PT[hero_pairs(hole)[None, :, None], tids[:, None, :]].max(axis=(1, 2))
    opp_pairs = PID[opp[:, P10[:, 0]], opp[:, P10[:, 1]]]
    ov = PT[opp_pairs[:, :, None], tids[:, None, :]].max(axis=(1, 2))
    return float(((ov < hv) + 0.5 * (ov == hv)).mean())
