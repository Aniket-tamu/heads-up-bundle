# Original lines 1–27 are outside the supplied photos.


























import hashlib
import json
import math
import os

import numpy as np

from game import Action

from . import abstraction, cards, history, tree
from . import strategy as v0

DATA = cards.DATA_DIR
POLICY_FILE = os.path.join(DATA, "cfr_policy.npy")
META_FILE = os.path.join(DATA, "cfr_meta.json")
NEAR_RADIUS = 8           # never-visited row: look this many bucket ids up and down for a visited one
DEFAULT_MODE = "sample"
DEFAULT_T = 0.1
DEFAULT_WIDEN = False
_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
_MASK = (1 << 64) - 1


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _load():
    """(policy, row of each tree node or -1, problem string); policy is None when unusable."""
    rows = np.full(len(tree.NODES), -1, dtype=np.int64)
    try:
        with open(META_FILE) as fh:
            meta = json.load(fh)
        ids = [int(i) for i in meta["decision_node_ids"]]
        if ids != [n.id for n in tree.DECISIONS]:
            return None, rows, "decision ids differ from plo5bot.tree"
        if [meta["actions"][str(a)] for a in range(4)] != [a.name for a in _ACTIONS]:
            return None, rows, "action columns differ from game.Action"
        for name, digest in meta["abstraction_sha256"].items():
            if _sha256(os.path.join(DATA, name)) != digest:
                return None, rows, f"{name} differs from the trained abstraction"
        pol = np.load(POLICY_FILE, allow_pickle=False)
        if pol.dtype != np.uint8 or pol.shape != (len(ids), 256, 4):
            return None, rows, f"bad policy array {pol.dtype} {pol.shape}"
        rows[ids] = np.arange(len(ids))
        return pol, rows, ""
    except Exception as exc:    # missing or broken files: v0 plays everything
        return None, rows, f"{type(exc).__name__}: {exc}"


POLICY, ROW, PROBLEM = _load()
NUM_BUCKETS = [256, 128, 128, 128]


# ---------------------------------------------------------------- preflop range widening (opt-in)
# The blueprint answers a preflop raise as if the raiser used the blueprint's own raising range.
# Real opponents raise far more often (pot-opens: BRT 89%, Barreler 63% vs the blueprint's 17%),
# so their range is weaker and the blueprint over-folds. For our first answer to the opponent's
# first voluntary raise, we model its range as the top-q hands by preflop equity (q = how often it
# took that action at that node) and compute T = the equity against the top-q_bp range at which
# the blueprint itself stops folding (q_bp = the blueprint's own frequency; T is the weighted
# quantile of our reaching hands at the blueprint's fold share). With q_obs > q_bp (lower bound),
# a blueprint fold becomes a call when our equity against the top-q_obs range reaches T.
# params "widen": True turns it on; "tighten": True also turns calls into folds when q_obs < q_bp.
WIDEN_MIN_N = 20          # opponent decisions at the parent node before the estimate is used
WIDEN_PRIOR = 10          # prior weight (decisions) of the blueprint's frequency
WIDEN_Z = 1.0             # confidence-bound width (standard errors) on the observed frequency


def _top_equity(cls, q):
    """Equity of preflop class(es) cls against the strongest fraction q of hands (20 bands of 5%)."""
    k = min(max(q, 0.05), 1.0) * 20.0
    kf = int(k)
    tot = _BAND_TOP[cls, kf]
    if kf < 20:
        tot = tot + (k - kf) * _BAND[cls, 19 - kf]
    return tot / k


def _widen_tables():
    """{our node id: (parent node id, opponent action, q_bp, T)} for the nodes the widening covers."""
    out = {}
    if POLICY is None:
        return out
    pf = abstraction._tables()["pf"]
    w_cls = np.load(os.path.join(DATA, "preflop_class_count.npy"), allow_pickle=False).astype(np.float64)
    w_bkt = np.bincount(pf, w_cls, 256) / w_cls.sum()
    for node in tree.DECISIONS:
        if node.street != 0 or not node.seq or node.seq[-1] not in (2, 3) or len(node.seq) > 2:
            continue
        parent = tree.INDEX[node.seq[:-1]]
        a = node.seq[-1]
        pol_p = POLICY[ROW[parent]].astype(np.float64) / 255.0
        q_bp = float((w_bkt * pol_p[:, a]).sum())
        reach = np.ones(len(w_cls))
        if len(node.seq) == 2:                             # we acted first at the root
            reach = POLICY[ROW[0]][pf, node.seq[0]].astype(np.float64) / 255.0
        fold = POLICY[ROW[node.id]][pf, 0].astype(np.float64) / 255.0
        wt = w_cls * reach
        share = float((wt * fold).sum() / wt.sum())
        e = _top_equity(np.arange(len(w_cls)), q_bp)
        order = np.argsort(e)
        cum = np.cumsum(wt[order]) / wt.sum()
        thr = float(e[order][min(int(np.searchsorted(cum, share)), len(order) - 1)])
        out[node.id] = (parent, a, q_bp, thr)
    return out


try:
    _BAND = v0.PF_BAND.astype(np.float64)                                    # (classes, 20), band 19 strongest
    _BAND_TOP = np.concatenate([np.zeros((len(_BAND), 1)), np.cumsum(_BAND[:, ::-1], axis=1)], axis=1)
    WIDEN = _widen_tables()
except Exception:            # widening is optional: without its tables the blueprint plays alone
    WIDEN = {}


def widen(obs, node_id, action, memory, tighten=False):
    """The widened (or tightened) answer instead of the blueprint's `action`, or None to keep it."""
    spot = WIDEN.get(node_id)
    if spot is None:
        return None
    parent, a, q_bp, thr = spot
    counts = ((memory.get("opp") or {}).get("node") or {}).get(parent)
    if not counts or sum(counts) < WIDEN_MIN_N:
        return None
    n = sum(counts) + WIDEN_PRIOR
    q = (counts[a] + WIDEN_PRIOR * q_bp) / n
    se = WIDEN_Z * (q * (1.0 - q) / n) ** 0.5
    cls = int(abstraction.preflop_class(list(obs.hole_cards)))
    if action == Action.FOLD and q - se > q_bp and _top_equity(cls, q - se) >= thr:
        return Action.CHECK_CALL
    if tighten and action == Action.CHECK_CALL and q + se < q_bp and _top_equity(cls, q + se) < thr:
        return Action.FOLD
    return None


def new_opponent_model():
    return v0.new_opponent_model()


def _uniform(obs):
    """A number in [0, 1) that is a fixed function of the observation (splitmix64 over its fields)."""
    h = (obs.hand_index + 0x9E3779B97F4A7C15) & _MASK
    vals = list(obs.hole_cards) + [64] + list(obs.board) + [65]
    vals += [int(a.action) * 2 + int(a.is_me) for a in obs.action_history]
    for v in vals:
        h = (h ^ (int(v) + 0x9E3779B97F4A7C15)) & _MASK
        h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK
        h ^= h >> 31
    return (h >> 11) / float(1 << 53)


def _state(memory):
    st = memory.get("cfr")
    if st is None:
        st = memory["cfr"] = {"hand": -1, "bucket": {}, "n": 0, "v0": 0, "near": 0, "why": {}}
    return st


def _fallback(obs, memory, budget_s, st, why):
    st["v0"] += 1
    st["why"][why] = st["why"].get(why, 0) + 1
    return v0.act(obs, memory, budget_s)


def _bucket(obs, st, budget_s):
    if st["hand"] != obs.hand_index:
        st["hand"], st["bucket"] = obs.hand_index, {}
    street = int(obs.street)
    b = st["bucket"].get(street)
    if b is None:
        b = st["bucket"][street] = int(abstraction.bucket(street, list(obs.hole_cards), list(obs.board),
                                                           budget_s=budget_s))
    return b


def weights(obs, bucket, st=None):
    """(uint8 policy weights over obs.legal_actions, None) at this node and bucket, or (None, reason)."""
    node = tree.NODES[tree.node_for(obs.action_history)]
    d = int(ROW[node.id])
    legal = [int(a) for a in obs.legal_actions]
    if (d < 0 or node.street != int(obs.street) or node.actor != (0 if obs.is_button else 1)
            or node.pot != obs.pot or node.to_call != obs.to_call or sorted(legal) != sorted(node.legal)):
        return None, "mismatch"
    row = POLICY[d, bucket]
    if not row.any():
        nb = NUM_BUCKETS[node.street]
        for r in range(1, NEAR_RADIUS + 1):
            hit = [b for b in (bucket - r, bucket + r) if 0 <= b < nb and POLICY[d, b].any()]
            if hit:
                row = POLICY[d, hit[0]]
                if st is not None:
                    st["near"] += 1
                break
        else:
            return None, "unvisited"
    w = [int(row[a]) for a in legal]
    return (w, None) if sum(w) > 0 else (None, "unvisited")


def choose(w, u, mode=DEFAULT_MODE, t=DEFAULT_T):
    """Index of the chosen action in weights w and a uniform u, after the purification `mode`."""
    if mode == "argmax":
        return max(range(len(w)), key=lambda i: (w[i], -i))
    if mode == "threshold":
        top = max(w)
        w = [x if (x >= t * 255 or x == top) else 0 for x in w]
    total = sum(w)
    x = u * total
    acc = 0
    for i, wi in enumerate(w):
        acc += wi
        if x < acc:
            return i
    return max(i for i, wi in enumerate(w) if wi > 0)


def act(obs, memory, budget_s):
    st = _state(memory)
    st["n"] += 1
    if POLICY is None:
        return _fallback(obs, memory, budget_s, st, "policy")
    try:
        w, why = weights(obs, _bucket(obs, st, budget_s), st)
    except KeyError:                 # history outside the tree
        w, why = None, "tree"
    except Exception:                # anything else in the lookup: v0 plays this decision
        w, why = None, "error"
    if w is None:
        return _fallback(obs, memory, budget_s, st, why)
    prm = memory.get("params") or {}
    i = choose(w, _uniform(obs), prm.get("mode", DEFAULT_MODE), float(prm.get("t", DEFAULT_T)))
    action = obs.legal_actions[i]
    if prm.get("widen", DEFAULT_WIDEN) and obs.street == 0:
        try:
            alt = widen(obs, tree.node_for(obs.action_history), action, memory, bool(prm.get("tighten", False)))
        except Exception:
            alt = None
        if alt is not None and alt in obs.legal_actions:
            st["widen"] = st.get("widen", 0) + 1
            return alt
    return action


# ----------------------------------------------------------- balanced-opponent guard (opt-in)
# For use under plo5bot.exploit with params "card" / "sd": those card-aware adjustments win big against
# unbalanced bots but lose about 0.18 bb/deal to this blueprint itself (dev/cfr/RUNTIME.md). With params
# "guard": True, after every hand we compare the opponent's preflop first actions (as small blind at the
# root: fold-or-limp / half / pot; as big blind after our limp / half open / pot open: fold / call /
# raise) with the blueprint's own frequencies. While all of them agree (within GUARD_TOL plus 2 standard
# errors, at least GUARD_MIN_N root samples), the opponent plays like the blueprint and "card" and "sd"
# are added to params["off"] (the exploit layer reads it on every decision); otherwise params["off"] is
# what it was configured to be.
GUARD_MIN_N = 30
GUARD_TOL = 0.08


def _guard_spots():
    """[(node id, action groups, blueprint frequency of each group)] for the opponent's first decisions."""
    if POLICY is None:
        return []
    pf = abstraction._tables()["pf"]
    w_cls = np.load(os.path.join(DATA, "preflop_class_count.npy"), allow_pickle=False).astype(np.float64)
    w_bkt = np.bincount(pf, w_cls, 256) / w_cls.sum()
    out = []
    for seq, groups in (((), ((0, 1), (2,), (3,))), ((1,), ((1,), (2, 3))),
                        ((2,), ((0,), (1,), (2, 3))), ((3,), ((0,), (1,), (2, 3)))):
        nid = tree.INDEX[seq]
        mix = (POLICY[ROW[nid]].astype(np.float64) / 255.0 * w_bkt[:, None]).sum(axis=0)
        out.append((nid, groups, tuple(float(sum(mix[a] for a in g)) for g in groups)))
    return out


try:
    GUARD = _guard_spots()
except Exception:
    GUARD = []


def balanced(model):
    """True when the opponent's observed preflop first-action frequencies match the blueprint's."""
    if not GUARD:
        return False
    nodes = model.get("node") or {}
    for nid, groups, f_bp in GUARD:
        c = nodes.get(nid)
        n = sum(c) if c else 0
        if n < GUARD_MIN_N:
            if nid == 0:
                return False
            continue
        for g, fb in zip(groups, f_bp):
            fo = sum(c[a] for a in g) / n
            if abs(fo - fb) > GUARD_TOL + 2.0 * (max(fb * (1.0 - fb), 0.01) / n) ** 0.5:
                return False
    return True


# --------------------------------------------------- postflop balanced-opponent guard (opt-in)
# The preflop guard only knows the blueprint's own preflop frequencies, so a bot that plays like the blueprint
# after the flop but opens differently keeps "card"/"sd" on, and balanced postflop play beats them
# (dev/adv2/NOTES.md). With params "pguard": True, after every hand the opponent's postflop decisions are
# compared, per (street, facing a bet) class, with the blueprint's own frequencies at the same nodes (PG_EXP):
# its bet/raise rate and, facing a bet, its fold rate. While every class with PG_MIN decisions agrees (within
# PG_TOL plus PG_Z standard errors) and at least PG_CLASSES classes are tested, "card" and "sd" go off as with
# the preflop guard (the two guards are OR-ed), except against a nit (its shown postflop bets are rarely weak:
# a one-sided binomial test against the blueprint's share PG_SD_P, which needs PG_SD_MIN showdowns first) or,
# with params "pg_overfold", an over-folder (it folds to our postflop bets clearly more than the blueprint).
# Params "pg_tol", "pg_classes" override PG_TOL, PG_CLASSES; "pg_dry": True only counts (dev diagnostics);
# "pg_sprt": True replaces the binomial nit test by a sequential likelihood-ratio test (PG_SD_NIT, PG_SD_LLR) that
# also keeps the exploits on while the showdown evidence is still ambiguous; "pg_lifted": True (with exploit's
# "bounded") leaves blueprint-like opponents to the bound and switches off only once the bound is lifted.
PG_MIN = 40
PG_CLASSES = 3
PG_TOL = 0.06
PG_Z = 2.0
PG_SD_MIN = 12           # showdowns after its postflop bets/raises before the guard can switch anything off
PG_SD_CAP = 200          # showdowns evaluated per half (0.3 ms each)
PG_SD_HS = 0.5           # weak: river strength (share of holdings beaten) below this
PG_SD_P = 0.33           # blueprint-like bots show weak hands in about 0.3-0.4 of these showdowns
PG_SD_TOL = 0.05         # nit: weak share below PG_SD_P - PG_SD_TOL by more than PG_SD_Z standard errors
PG_SD_Z = 2.0
PG_SD_NIT = 0.06         # params "pg_sprt": sequential test of that share, blueprint-like PG_SD_P vs nit PG_SD_NIT;
PG_SD_LLR = 3.0          # switch off only once the log-likelihood ratio for blueprint-like reaches this
PG_FOLD_TOL = 0.03       # over-folder: fold rate facing our postflop bets above the blueprint's by this + PG_FOLD_Z se
PG_FOLD_Z = 2.0
CONF_FILE = os.path.join(DATA, "cfr_conf.npy")
_HS_SCALE = 2.0 * 65780  # abstraction.river_features counts 2 per win and 1 per tie against every holding


def _expected_freqs():
    """(decisions, 4) float32 action frequencies of the blueprint at each node; NaN rows were never visited.

    Each bucket's policy row is weighted by its training visit count (cfr_conf.npy = 4 log2(1 + w)), which is
    proportional to how often the actor reaches the node with that bucket (dev/guard/conf_freq_check.py: within
    0.01 of self-play counts per class)."""
    if POLICY is None:
        return None
    conf = np.load(CONF_FILE, allow_pickle=False)
    if conf.shape != POLICY.shape[:2]:
        return None
    w = np.exp2(conf.astype(np.float32) * 0.25) - 1.0
    out = np.empty((len(conf), 4), dtype=np.float32)
    for a in range(4):
        out[:, a] = (w * POLICY[:, :, a]).sum(axis=1)
    tot = out.sum(axis=1, keepdims=True)
    out[tot[:, 0] <= 0] = np.nan
    return out / np.maximum(tot, 1e-9)


try:
    PG_EXP = _expected_freqs()
except Exception:
    PG_EXP = None


def pg_record(result, memory, st):
    """Add the finished hand's postflop opponent decisions (and its shown strength after aggression) to st."""
    seq, me = (memory.get("seqs") or {}).get(result.hand_index, ((), 1))
    full = history.complete(seq, me, result)
    if full is None:
        return
    pg = st.setdefault("pg", {})
    aggr = False
    for nid, a in history.opponent_decisions(full, me):
        node = tree.NODES[nid]
        d = int(ROW[nid])
        if node.street == 0 or d < 0:
            continue
        aggr = aggr or a >= 2
        e = PG_EXP[d]
        if not e[0] >= 0.0:                   # NaN: never visited in training
            continue
        facing = node.to_call > 0
        s = pg.get(node.street * 2 + int(facing))
        if s is None:
            s = pg[node.street * 2 + int(facing)] = [0, 0, 0.0, 0.0, 0, 0.0, 0.0]
        pa = float(e[2] + e[3])
        s[0] += 1
        s[1] += int(a >= 2)
        s[2] += pa
        s[3] += pa * (1.0 - pa)
        if facing:
            pf = float(e[0])
            s[4] += int(a == 0)
            s[5] += pf
            s[6] += pf * (1.0 - pf)
    sd = st.setdefault("pg_sd", [0, 0])
    if aggr and result.opp_hole and len(result.board) == 5 and sd[0] < PG_SD_CAP:
        hs = float(abstraction.river_features(list(result.opp_hole), list(result.board))) / _HS_SCALE
        sd[0] += 1
        sd[1] += int(hs < PG_SD_HS)


def balanced_post(st, prm=None):
    """True when the opponent's postflop rates match the blueprint's at the nodes it played (see pg_record)."""
    prm = prm or {}
    tol, need = float(prm.get("pg_tol", PG_TOL)), int(prm.get("pg_classes", PG_CLASSES))
    tested = 0
    for k, s in (st.get("pg") or {}).items():
        n = s[0]
        if n < PG_MIN:
            continue
        tested += 1
        if abs(s[1] - s[2]) > tol * n + PG_Z * max(s[3], 1.0) ** 0.5:
            return False
        if k & 1 and abs(s[4] - s[5]) > tol * n + PG_Z * max(s[6], 1.0) ** 0.5:
            return False
    return tested >= need


def pg_exempt(st, prm=None):
    """Why the postflop guard must not switch off ('wait', 'nit', 'overfold'), or '' when it may."""
    n, weak = st.get("pg_sd") or (0, 0)
    if (prm or {}).get("pg_sprt"):
        llr = weak * math.log(PG_SD_P / PG_SD_NIT) + (n - weak) * math.log((1.0 - PG_SD_P) / (1.0 - PG_SD_NIT))
        if llr < PG_SD_LLR:
            return "nit" if llr <= -PG_SD_LLR else "wait"
    elif n < PG_SD_MIN:
        return "wait"
    elif weak < n * (PG_SD_P - PG_SD_TOL) - PG_SD_Z * (n * PG_SD_P * (1.0 - PG_SD_P)) ** 0.5:
        return "nit"
    if (prm or {}).get("pg_overfold"):
        n = of = ef = vf = 0.0
        for k, s in (st.get("pg") or {}).items():
            if k & 1:
                n, of, ef, vf = n + s[0], of + s[4], ef + s[5], vf + s[6]
        if n >= PG_MIN and of - ef > PG_FOLD_TOL * n + PG_FOLD_Z * max(vf, 1.0) ** 0.5:
            return "overfold"
    return ""


def _bound_lifted(memory, prm):
    """Whether the exploit layer has evidence to lift its blueprint support bound."""
    try:
        from . import exploit
        return exploit.off_policy(memory.get("xp") or {}, exploit.OFFP_CLASSES,
                                  float(prm.get("offp_p0", exploit.OFFP_P0)))
    except Exception:
        return True


def _guard(memory):
    prm = memory.get("params")
    if not prm or not (prm.get("guard") or prm.get("pguard")):
        return
    st = _state(memory)
    if "off0" not in st:
        st["off0"] = list(prm.get("off") or [])
    on = bool(prm.get("guard")) and balanced(memory.get("opp") or {})
    if on:
        st["guarded"] = st.get("guarded", 0) + 1
    elif prm.get("pguard") and balanced_post(st, prm):
        why = pg_exempt(st, prm)
        if not why and prm.get("pg_lifted") and prm.get("bounded") and not _bound_lifted(memory, prm):
            why = "bounded"  # the exploit layer's bound still keeps its changes inside the blueprint's support
        on = not why and not prm.get("pg_dry")
        key = "pg_" + (why or "on")
        st[key] = st.get(key, 0) + 1
    prm["off"] = st["off0"] + (["card", "sd"] if on else [])


def observe(result, memory):
    try:
        pg_record(result, memory, _state(memory))
    except Exception:
        pass
    if "seqs" in memory and "opp" in memory:
        v0.observe(result, memory)
    try:
        _guard(memory)
    except Exception:
        pass


# The supplied photos end here. Any further original lines are unavailable.
