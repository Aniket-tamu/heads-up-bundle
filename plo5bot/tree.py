"""The exact betting tree of one hand, built with the referee's own game.py rules.

Every node is identified by the sequence of Action values played so far in
the hand (blinds excluded). The button always acts first, so that sequence
fixes the whole betting state. Players are relative: 0 = button (small blind),
1 = big blind. Loaded at import from data/tree_cache.npz when that cache matches
the current game.py (sha256), otherwise built from the rules (about a second).
"""

import hashlib
import os
from dataclasses import dataclass

import numpy as np

from game import Action, Deal, Game, Street

_DUMMY = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))


@dataclass
class Node:
    id: int
    seq: tuple           # Action values played so far
    terminal: bool
    actor: int           # 0 button, 1 big blind (decision nodes)
    street: int
    legal: tuple         # Action values (decision nodes)
    children: dict       # Action value -> node id
    pot: int
    to_call: int
    contrib: tuple       # (button, big blind) chips in the hand
    fold_by: int         # -1, or the relative player who folded (terminal)
    allin: bool          # terminal reached with a player all-in before the river ended
    raises: int = 0      # voluntary bets/raises so far on this street
    stacks: tuple = ()   # (button, big blind) chips behind


def build():
    nodes = []
    index = {}

    def visit(game, seq):
        nid = len(nodes)
        index[seq] = nid
        b = game.button_seat
        contrib = (game.contributed[b], game.contributed[1 - b])
        if game.is_terminal:
            folded = [r for r in game.history if r.action == Action.FOLD]
            fold_by = -1
            if folded:
                fold_by = 0 if folded[-1].seat == b else 1
            allin = not folded and (game.stacks[0] == 0 or game.stacks[1] == 0)
            nodes.append(Node(nid, seq, True, -1, int(game.street), (), {}, game.pot, 0, contrib, fold_by, allin,
                              game.raises_this_street, (game.stacks[b], game.stacks[1 - b])))
            return nid
        seat = game.to_act
        legal = tuple(int(a) for a in game.legal_actions())
        to_call = max(0, game.bets[1 - seat] - game.bets[seat])
        node = Node(nid, seq, False, 0 if seat == b else 1, int(game.street), legal, {}, game.pot, to_call,
                    contrib, -1, False, game.raises_this_street, (game.stacks[b], game.stacks[1 - b]))
        nodes.append(node)
        for a in legal:
            child = game.clone()
            child.apply(Action(a))
            node.children[a] = visit(child, seq + (a,))
        return nid

    visit(Game(_DUMMY, 0), ())
    return nodes, index


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tree_cache.npz")


def _game_sha():
    """sha256 of the referee rules file the tree was built from (a stale cache is never used)."""
    import game as _game
    with open(_game.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_cache(path=CACHE_FILE):
    """Flatten the tree into arrays (no pickles) so startup can skip rebuilding it."""
    nodes, _ = build()
    n, maxlen = len(nodes), max(len(x.seq) for x in nodes)
    seq = np.full((n, maxlen), -1, np.int8)
    lens = np.zeros(n, np.int8)
    ints = np.zeros((n, 12), np.int32)  # terminal actor street pot to_call c0 c1 fold_by allin raises s0 s1
    kids = np.full((n, 4), -1, np.int32)
    legal = np.zeros((n, 4), np.int8)
    for x in nodes:
        seq[x.id, :len(x.seq)] = x.seq
        lens[x.id] = len(x.seq)
        ints[x.id] = (x.terminal, x.actor, x.street, x.pot, x.to_call, x.contrib[0], x.contrib[1], x.fold_by,
                      x.allin, x.raises, x.stacks[0], x.stacks[1])
        for a, c in x.children.items():
            kids[x.id, a] = c
        for a in x.legal:
            legal[x.id, a] = 1
    sha = np.frombuffer(bytes.fromhex(_game_sha()), dtype=np.uint8)
    np.savez(path, seq=seq, lens=lens, ints=ints, kids=kids, legal=legal, game_sha=sha)


def _load_cache(path=CACHE_FILE):
    with np.load(path, allow_pickle=False) as z:
        if bytes(z["game_sha"].tobytes()).hex() != _game_sha():
            raise ValueError("tree cache was built from a different game.py")
        seq, lens, ints = z["seq"].tolist(), z["lens"].tolist(), z["ints"].tolist()
        kids, legal = z["kids"].tolist(), z["legal"].tolist()
    nodes, index = [], {}
    for i in range(len(lens)):
        s = tuple(seq[i][:lens[i]])
        v = ints[i]
        nodes.append(Node(i, s, bool(v[0]), v[1], v[2], tuple(a for a in range(4) if legal[i][a]),
                          {a: c for a, c in enumerate(kids[i]) if c >= 0}, v[3], v[4], (v[5], v[6]), v[7],
                          bool(v[8]), v[9], (v[10], v[11])))
        index[s] = i
    return nodes, index


def load():
    """The tree from the cache when it matches this game.py, else rebuilt from the rules."""
    try:
        return _load_cache()
    except Exception:
        return build()


NODES, INDEX = load()
DECISIONS = [n for n in NODES if not n.terminal]


def node_for(action_history):
    """Node id for an observation's action_history (tuple of ObservedAction)."""
    return INDEX[tuple(int(a.action) for a in action_history)]


def street_counts():
    counts = [0, 0, 0, 0]
    for n in DECISIONS:
        counts[n.street] += 1
    return counts


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    nodes, _ = build()
    print(f"{len(nodes)} nodes, {sum(not n.terminal for n in nodes)} decisions {street_counts()}, "
          f"{sum(n.terminal for n in nodes)} terminals in {time.perf_counter() - t0:.2f}s")
