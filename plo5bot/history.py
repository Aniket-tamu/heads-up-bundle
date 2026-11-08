"""Rebuild a finished hand's full action sequence from what our bot saw.

Observations stop at our last decision, so the opponent's closing action
(a fold, a call or a check) is never observed. HandResult tells us whether
it went to showdown, who won and our net, and the betting tree has every
node's contributions, so the missing tail can be recovered exactly.
"""

from game import Action

from . import tree

_FOLD = int(Action.FOLD)


def _consistent(node, me, result):
    """Does terminal `node` match the result for relative player `me` (0 button, 1 big blind)?"""
    if not node.terminal:
        return False
    mine, theirs = node.contrib[me], node.contrib[1 - me]
    if node.fold_by >= 0:
        if result.went_to_showdown:
            return False
        if node.fold_by == me:
            return result.my_net == -mine
        return result.my_net == theirs
    if not result.went_to_showdown:
        return False
    if result.winner == "split":
        return result.my_net == 0
    return result.my_net == (theirs if result.winner == "me" else -mine)


def complete(seq, me, result, max_extra=3):
    """Full action sequence of a finished hand.

    seq: Action values up to and including our last action (() if we never acted).
    me: 0 if we were the button, 1 if big blind.
    Returns the full sequence (a tuple) or None if nothing fits.
    Only opponent actions can follow our last action without a new decision
    for us, so we search opponent-only continuations.
    """
    start = tree.INDEX.get(tuple(seq))
    if start is None:
        return None
    frontier = [start]
    for _ in range(max_extra + 1):
        nxt = []
        for nid in frontier:
            node = tree.NODES[nid]
            if node.terminal:
                if _consistent(node, me, result):
                    return node.seq
                continue
            if node.actor == me:
                continue
            nxt.extend(node.children.values())
        frontier = nxt
        if not frontier:
            break
    return None


def opponent_decisions(full_seq, me):
    """[(node id, action value)] for every opponent decision in a full sequence."""
    out = []
    nid = tree.INDEX[()]
    for a in full_seq:
        node = tree.NODES[nid]
        if not node.terminal and node.actor != me:
            out.append((nid, a))
        nid = node.children[a]
    return out
