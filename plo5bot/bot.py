"""Entry points called from agent.py: new_half, choose_action, on_hand_end.

This layer never raises and always returns an Action from obs.legal_actions.
It keeps a time ledger that estimates what the judge charges us (each request
costs its wall time minus 1 ms; HTTP and JSON overhead was measured at about
0.42 ms per request in the tournament image) and gives the strategy a compute
budget for each decision so the 8 s bank can never run out.

A strategy is any module with:
    new_opponent_model() -> dict
    act(obs, memory, budget_s) -> Action    (may raise; we fall back)
    observe(result, memory)                (may raise; we swallow)
Optional per-instance parameters are stored in memory["params"].
"""

import time

from game import Action, Observation, Street

from . import strategy as default_strategy

OVERHEAD_S = 0.0007      # HTTP + JSON + judge work per request (measured 0.42 ms, plus margin)
GRACE_S = 0.001         # the judge does not charge the first 1 ms of a request
RESERVE_S = 1.5         # bank we never plan to spend
MAX_BUDGET_S = 0.020    # never think longer than this on one decision
MIN_BUDGET_S = 0.0002
EMERGENCY_LEFT_S = 3.0  # below this much estimated bank left, play the cheap policy only


def cheap_policy(obs):
    """(0.05 ms) emergency play: preflop equity table, made-hand category postflop."""
    from evaluator import FLUSH, FULL_HOUSE, STRAIGHT, THREE_OF_A_KIND, category_of

    from . import cards, strategy as tables
    legal = obs.legal_actions
    raise_ok = Action.RAISE_POT in legal
    if obs.street == Street.PREFLOP:
        e = tables.preflop_equities(obs.hole_cards)[0]
        if raise_ok and e >= 0.58:
            return Action.RAISE_POT
        if obs.to_call == 0 or e >= 0.47:
            return Action.CHECK_CALL
        return Action.FOLD
    cat = category_of(int(cards.DENSE_SCORES[cards.omaha_rank(obs.hole_cards, obs.board)]))
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if cat >= FLUSH or (cat >= FULL_HOUSE and obs.street != Street.RIVER):
        return Action.RAISE_POT if raise_ok else Action.CHECK_CALL
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if cat >= FULL_HOUSE or (cat >= THREE_OF_A_KIND and price <= 0.34) or (cat >= STRAIGHT and price <= 0.25):
        return Action.CHECK_CALL
    return Action.FOLD


def bank_left(memory):
    """Estimated seconds of time bank left this half."""
    cfg = memory.get("cfg") or {"bank": 8.0}
    clock = memory.get("clock") or {"charged": 0.0}
    return cfg["bank"] - clock["charged"]


def safe_default(obs):
    """CHECK_CALL when free, else FOLD; both are always legal in those spots."""
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def _num(config, name, default, cast):
    try:
        value = cast(getattr(config, name, default))
        return value if value > 0 else default
    except Exception:
        return default


def _charge(memory, elapsed):
    """Charge one /act request: our choose_action time plus any batched on_hand_end time."""
    clock = memory.get("clock")
    if clock is None:
        return
    elapsed += clock.pop("pending", 0.0)
    clock["requests"] += 1
    clock["charged"] += max(0.0, elapsed + OVERHEAD_S - GRACE_S)
    if elapsed > clock["max"]:
        clock["max"] = elapsed


def budget(obs, memory):
    """Seconds of compute we may spend on this decision without endangering the bank."""
    cfg = memory.get("cfg") or {"hands": 1000, "bank": 8.0}
    clock = memory.get("clock") or {"charged": 0.0, "requests": 0}
    hands_left = max(1, cfg["hands"] - obs.hand_index)
    done = max(1, obs.hand_index)
    per_hand = max(2.0, clock["requests"] / done) if obs.hand_index > 20 else 4.0
    remaining = cfg["bank"] - RESERVE_S - clock["charged"]
    if remaining <= 0:
        return MIN_BUDGET_S
    free = GRACE_S - OVERHEAD_S
    return max(MIN_BUDGET_S, min(MAX_BUDGET_S, remaining / (hands_left * per_hand) + free))


def _remember(obs, action, memory):
    """Keep our view of this hand's sequence for rebuilding it when the result arrives."""
    seqs = memory.get("seqs")
    if seqs is None:
        return
    seqs[obs.hand_index] = (tuple(int(a.action) for a in obs.action_history) + (int(action),),
                            0 if obs.is_button else 1)
    if len(seqs) > 32:
        for k in [k for k in seqs if k < obs.hand_index - 8]:
            del seqs[k]


class Bot:
    """One bot = one strategy module (+ optional params), with the safety net and time ledger."""

    def __init__(self, strategy_module=None, params=None):
        self.strategy = strategy_module or default_strategy
        self.params = dict(params or {})

    def new_half(self, config, memory):
        """Must never raise: a failing /new-half forfeits the half at once."""
        memory["cfg"] = {
            "stack": _num(config, "starting_stack", 80, int),
            "bb": _num(config, "big_blind", 2, int),
            "hands": _num(config, "num_hands", 1000, int),
            "bank": _num(config, "time_bank_seconds", 8.0, float),
            "cap": _num(config, "raise_cap", 4, int),
        }
        memory["clock"] = {"charged": 0.0, "requests": 0, "max": 0.0, "fallbacks": 0, "errors": 0}
        memory["last_error"] = ""
        memory["seqs"] = {}      # hand_index -> (our view of the sequence incl. our action, role)
        memory["params"] = dict(self.params)
        try:
            memory["opp"] = self.strategy.new_opponent_model()
        except BaseException as exc:
            memory["opp"] = {}
            memory["last_error"] = f"{type(exc).__name__}: {exc}"

    def choose_action(self, obs, memory):
        t0 = time.perf_counter()
        action = None
        try:
            if bank_left(memory) < EMERGENCY_LEFT_S:
                clock = memory.get("clock")
                if clock is not None:
                    clock["emergency"] = clock.get("emergency", 0) + 1
                action = cheap_policy(obs)
            else:
                action = self.strategy.act(obs, memory, budget(obs, memory))
        except BaseException as exc:  # never let anything escape to the judge
            clock = memory.get("clock")
            if clock is not None:
                clock["errors"] += 1
            memory["last_error"] = f"{type(exc).__name__}: {exc}"
        if action == Action.FOLD and obs.street == Street.PREFLOP and obs.is_button and not obs.action_history:
            action = Action.CHECK_CALL  # never fold the small blind on our first decision: limp instead
        if type(action) is not Action or action not in obs.legal_actions:
            clock = memory.get("clock")
            if clock is not None:
                clock["fallbacks"] += 1
            action = safe_default(obs)
        try:
            _remember(obs, action, memory)
        except BaseException:
            pass
        _charge(memory, time.perf_counter() - t0)
        return action

    def on_hand_end(self, result, memory):
        t0 = time.perf_counter()
        try:
            self.strategy.observe(result, memory)
        except BaseException as exc:
            memory["last_error"] = f"{type(exc).__name__}: {exc}"
        clock = memory.get("clock")
        if clock is not None:  # charged with the next /act, which carries this result
            clock["pending"] = clock.get("pending", 0.0) + (time.perf_counter() - t0)

    def warmup(self):
        """Run a few decisions at import so the first real request is not slow."""
        hole = (0, 5, 18, 27, 35)
        samples = [
            (Street.PREFLOP, (), 3, 1, 1, 2, 79, 78, True),
            (Street.FLOP, (8, 13, 30), 12, 0, 0, 0, 74, 74, False),
            (Street.TURN, (8, 13, 30, 2), 24, 6, 0, 6, 68, 62, False),
            (Street.RIVER, (8, 13, 30, 2, 21), 36, 0, 0, 0, 62, 62, True),
        ]
        for street, board, pot, to_call, my_bet, opp_bet, my_stack, opp_stack, button in samples:
            legal = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT) if to_call else \
                    (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
            obs = Observation(hand_index=0, street=street, hole_cards=hole, board=board, pot=pot,
                              to_call=to_call, my_bet=my_bet, opp_bet=opp_bet, my_stack=my_stack,
                              opp_stack=opp_stack, is_button=button, legal_actions=legal,
                              raise_to={Action.RAISE_HALF: opp_bet + (pot + to_call) // 2,
                                        Action.RAISE_POT: opp_bet + pot + to_call},
                              min_raise_to=opp_bet + 2, raises_this_street=1 if to_call else 0,
                              action_history=(), big_blind=2)
            memory = {}
            try:
                self.new_half(None, memory)
                for _ in range(3):
                    self.strategy.act(obs, memory, 0.004)
            except BaseException:
                pass


from . import exploit, strat_cfr  # noqa: E402,F401  (M2 shipped: exploitation layer over the CFR blueprint)

DEFAULT = Bot(exploit, {"base": "plo5bot.strat_cfr", "card": True, "sd": True, "guard": True,
                        "value": True, "trap": True, "bounded": True,
                        "pguard": True, "pg_overfold": True, "pg_sprt": True, "pg_lifted": True})


def new_half(config, memory):
    DEFAULT.new_half(config, memory)


def choose_action(obs, memory):
    return DEFAULT.choose_action(obs, memory)


def on_hand_end(result, memory):
    DEFAULT.on_hand_end(result, memory)


def warmup():
    DEFAULT.warmup()
