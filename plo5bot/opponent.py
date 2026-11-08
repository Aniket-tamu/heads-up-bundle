"""What we learn about the opponent within one half (nothing carries over).

For every finished hand we rebuild the full action sequence, then count the
opponent's action at each betting-tree node and at a coarser node class, and
keep showdown records (their cards and the line they took) for range work.
"""

from . import history, tree

ACTIONS = 4  # FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT


def node_class(node):
    """Coarse situation key: (street, facing a bet, raises this street capped at 2, big bet)."""
    facing = node.to_call > 0
    big = facing and node.to_call * 2 > (node.pot - node.to_call)  # bet bigger than half the pot before it
    return (node.street, int(facing), min(node.raises, 2), int(big))


def new_model():
    return {"hands": 0, "unmatched": 0, "node": {}, "cls": {}, "showdowns": [], "done": set()}


def update(model, seq, me, result):
    """Learn from one finished hand. seq: our view up to and incl. our last action; me: 0 button, 1 BB."""
    if result.hand_index in model["done"]:
        return
    model["done"].add(result.hand_index)
    full = history.complete(seq, me, result)
    if full is None:
        model["unmatched"] += 1
        return
    model["hands"] += 1
    for nid, a in history.opponent_decisions(full, me):
        node = tree.NODES[nid]
        model["node"].setdefault(nid, [0] * ACTIONS)[a] += 1
        model["cls"].setdefault(node_class(node), [0] * ACTIONS)[a] += 1
    if result.opp_hole:
        model["showdowns"].append((tuple(result.opp_hole), tuple(result.board), full, me))


def frequencies(counts, prior, strength):
    """Dirichlet-smoothed action probabilities: (counts + strength * prior) / (n + strength)."""
    n = sum(counts)
    return [(c + strength * p) / (n + strength) for c, p in zip(counts, prior)], n


def class_counts(model, node):
    return model["cls"].get(node_class(node), [0] * ACTIONS)


def node_counts(model, nid):
    return model["node"].get(nid, [0] * ACTIONS)
