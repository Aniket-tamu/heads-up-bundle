"""Equity (P(win) + P(tie)/2) of our hand against one opponent holding.

Opponent holdings come from the live cards (not ours, not on the board),
either uniformly or as a weighted sample. All costs are a few hundred
microseconds to about 2 ms in numpy on one core.
"""

from itertools import combinations

import numpy as np

from . import cards

PT = cards.PT
# PID[a, b] = colex id of the 2-card set (a, b) (0 when a == b); TID[a, b, c] likewise for 3 distinct cards.
# Built vectorized (same values as cards.pair_id / cards.triple_id) to keep startup short.
_CARDS = np.arange(36, dtype=np.int64)
_HI = np.maximum(_CARDS[:, None], _CARDS[None, :])
_LO = np.minimum(_CARDS[:, None], _CARDS[None, :])
PID = np.where(_CARDS[:, None] != _CARDS[None, :], _HI * (_HI - 1) // 2 + _LO, 0).astype(np.int32)
_S = np.sort(np.stack(np.meshgrid(_CARDS, _CARDS, _CARDS, indexing="ij"), axis=-1), axis=-1)
TID = np.where((_S[..., 0] != _S[..., 1]) & (_S[..., 1] != _S[..., 2]),
               _S[..., 2] * (_S[..., 2] - 1) * (_S[..., 2] - 2) // 6 + _S[..., 1] * (_S[..., 1] - 1) // 2 + _S[..., 0],
               0).astype(np.int32)
del _HI, _LO, _S
P10 = np.array(list(combinations(range(5), 2)), dtype=np.int64)  # 10 x 2
T10 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # 10 x 3

# River enumeration: the 65,780 five-card subsets of 26 live cards as local pair ids (10 x 65780).
_LP = [(i, j) for j in range(26) for i in range(j)]  # local pair id order (colex)
_LPI = np.array([i for i, _ in _LP], dtype=np.int64)
_LPJ = np.array([j for _, j in _LP], dtype=np.int64)
_C26 = np.array(list(combinations(range(26), 5)), dtype=np.int64)
# local pair id of positions (a < b) = C(b, 2) + a, the index of (a, b) in _LP
RIVER_OPP_PAIRS = np.ascontiguousarray(np.stack([
    (_C26[:, j] * (_C26[:, j] - 1) // 2 + _C26[:, i]).astype(np.int16)
    for i, j in combinations(range(5), 2)]))  # 65780 x 5 local positions
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
    live = live_cards(hole, board)  # 27 cards
    if opp is None:
        opp = sample_opponents(live, n_opp, rng)
    b4 = np.asarray(board, dtype=np.int64)
    boards = np.concatenate([np.broadcast_to(b4, (len(live), 4)), live[:, None]], axis=1)  # R x 5
    tids = TID[boards[:, T10[:, 0]], boards[:, T10[:, 1]], boards[:, T10[:, 2]]]  # R x 10
    pbr = PT[:, tids].max(axis=2).T  # R x 630
    hv = pbr[:, hero_pairs(hole)].max(axis=1)  # R
    opp_pairs = PID[opp[:, P10[:, 0]], opp[:, P10[:, 1]]]  # M x 10
    ov = pbr[:, opp_pairs].max(axis=2)  # R x M
    valid = ~(live[:, None, None] == opp[None, :, :]).any(axis=2)  # river not in opp hand
    res = (ov < hv[:, None]) + 0.5 * (ov == hv[:, None])
    return float((res * valid).sum() / valid.sum())


def flop_equity(hole, board, n=800, rng=None, opp=None):
    """Flop equity by joint sampling of (opponent holding, turn, river)."""
    rng = rng or np.random.default_rng(0)
    live = live_cards(hole, board)  # 28 cards
    if opp is None:
        pick = np.argsort(rng.random((n, len(live))), axis=1)[:, :7]
        dealt = live[pick]
        opp, runout = dealt[:, :5], dealt[:, 5:]
    else:
        # runouts drawn from cards not in each given opponent hand
        n = len(opp)
        free = np.where((live[None, :, None] == opp[:, None, :]).any(axis=2), 2.0, rng.random((n, len(live))))
        runout = live[np.argsort(free, axis=1)[:, :2]]
    b = np.concatenate([np.broadcast_to(np.asarray(board, dtype=np.int64), (n, 3)), runout], axis=1)
    tids = TID[b[:, T10[:, 0]], b[:, T10[:, 1]], b[:, T10[:, 2]]]  # n x 10
    hv = PT[hero_pairs(hole)[None, :, None], tids[:, None, :]].max(axis=(1, 2))
    opp_pairs = PID[opp[:, P10[:, 0]], opp[:, P10[:, 1]]]
    ov = PT[opp_pairs[:, :, None], tids[:, None, :]].max(axis=(1, 2))
    return float(((ov < hv) + 0.5 * (ov == hv)).mean())
