"""Decision policy (v2: equity-driven tight-aggressive play).

act(obs, memory, budget_s) returns an Action from obs.legal_actions.
observe(result, memory) learns from finished hands. Both may raise: the
caller (bot.py) catches everything and falls back to a legal action.
"""

import os
from math import comb

import numpy as np

from game import Action, Street

from . import cards, equity, opponent

DATA = cards.DATA_DIR
PF_CLASS = np.load(os.path.join(DATA, "preflop_class.npy"), allow_pickle=False)
PF_EQ = np.load(os.path.join(DATA, "preflop_eq.npy"), allow_pickle=False)
PF_BAND = np.load(os.path.join(DATA, "preflop_eq_bucket.npy"), allow_pickle=False)
COMB = [[comb(n, k) for k in range(6)] for n in range(37)]


def colex(hole):
    h = sorted(hole)
    return COMB[h[0]][1] + COMB[h[1]][2] + COMB[h[2]][3] + COMB[h[3]][4] + COMB[h[4]][5]


def preflop_equities(hole):
    """(equity vs random, vs top 40%, vs top 20%) for a 5-card hand."""
    c = PF_CLASS[colex(hole)]
    band = PF_BAND[c]
    return float(PF_EQ[c]), float(band[12:].mean()), float(band[16:].mean())


def new_opponent_model():
    return opponent.new_model()


def _rng(obs):
    seed = (obs.hand_index * 1_000_003 + sum(obs.hole_cards) * 7919 + len(obs.action_history)) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def _top_fraction(values, frac):
    """Values of the strongest `frac` of opponent holdings (by current strength)."""
    k = max(1, int(len(values) * frac))
    return np.partition(values, len(values) - k)[len(values) - k:]


def range_equity(obs, frac, budget_s):
    """Equity against the strongest `frac` of opponent holdings, ranked by current hand value."""
    hole, board = list(obs.hole_cards), list(obs.board)
    rng = _rng(obs)
    if len(board) == 5:
        hv, ov, _ = equity.river_opp_values(hole, board)
        top = _top_fraction(ov, frac) if frac < 1 else ov
        return float(((top < hv) + 0.5 * (top == hv)).mean())
    live = equity.live_cards(hole, board)
    n = 300 if budget_s > 0.003 else 150
    pool = equity.sample_opponents(live, int(n / max(frac, 0.15)), rng)
    if frac < 1:
        tids = np.array([cards.triple_id(*t) for t in _triples(board)], dtype=np.int64)
        opp_pairs = equity.PID[pool[:, equity.P10[:, 0]], pool[:, equity.P10[:, 1]]]
        cur = cards.PT[opp_pairs[:, :, None], tids[None, None, :]].max(axis=(1, 2))
        keep = np.argsort(-cur, kind="stable")[:max(20, int(len(pool) * frac))]
        pool = pool[keep]
    if len(board) == 4:
        return equity.turn_equity(hole, board, opp=pool[:n], rng=rng)
    return equity.flop_equity(hole, board, opp=pool[:n], rng=rng)


def _triples(board):
    b = list(board)
    return [(b[i], b[j], b[k]) for i in range(len(b)) for j in range(i + 1, len(b)) for k in range(j + 1, len(b))]


def _raise(obs, prefer_pot=True):
    if prefer_pot and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    if Action.RAISE_HALF in obs.legal_actions:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def _fold_or_check(obs):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def act(obs, memory, budget_s):
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal or Action.RAISE_HALF in legal
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0

    if obs.street == Street.PREFLOP:
        e, e40, e20 = preflop_equities(obs.hole_cards)
        opp_raises = sum(1 for a in obs.action_history if not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT))
        if obs.to_call == 0:                       # BB after a limp
            return _raise(obs) if (can_raise and e >= 0.56) else Action.CHECK_CALL
        if opp_raises == 0:                       # SB first decision
            if can_raise and e >= 0.52:
                return _raise(obs)
            return Action.CHECK_CALL if e >= 0.42 else Action.FOLD
        eq = e40 if opp_raises == 1 else e20
        if can_raise and e >= 0.55 and opp_raises == 1 and e >= 0.56:
            return _raise(obs)
        return Action.CHECK_CALL if eq >= price + 0.03 else Action.FOLD

    facing = obs.to_call > 0
    opp_aggr = sum(1 for a in obs.action_history if not a.is_me and a.street == obs.street
                   and a.action in (Action.RAISE_HALF, Action.RAISE_POT))
    frac = 1.0 if not facing else (0.5 if opp_aggr <= 1 else 0.25)
    eq = range_equity(obs, frac, budget_s)
    if facing:
        if can_raise and eq >= 0.70:
            return _raise(obs)
        margin = 0.04 if obs.street == Street.RIVER else 0.02
        return Action.CHECK_CALL if eq >= price + margin else Action.FOLD
    if can_raise and eq >= 0.62:
        return _raise(obs)
    if can_raise and eq >= 0.55:
        return _raise(obs, prefer_pot=False)
    return Action.CHECK_CALL


def observe(result, memory):
    seq, me = memory["seqs"].pop(result.hand_index, ((), 1))  # never acted => we were the big blind
    opponent.update(memory["opp"], seq, me, result)
