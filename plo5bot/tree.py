"""The exact betting tree of one hand, built with the referee's own game.py rules.

Every node is identified by the sequence of Action values played so far in
the hand (blinds excluded); the button always acts first, so that sequence
fixes the whole betting state. Players are relative: 0 = button (small blind),
1 = big blind. Built once at import (well under a second).
"""

from dataclasses import dataclass

from game import Action, Deal, Game, Street

_DUMMY = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))


@dataclass
class Node:
    id: int
    seq: tuple
    terminal: bool
    actor: int
    street: int
    legal: tuple
    children: dict
    pot: int
    to_call: int
    contrib: tuple
    fold_by: int
    allin: bool
    raises: int = 0
    stacks: tuple = ()


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
            nodes.append(Node(nid, seq, True, -1, int(game.street), (), {}, game.pot, 0,
                              contrib, fold_by, allin, game.raises_this_street,
                              (game.stacks[b], game.stacks[1 - b])))
            return nid
        seat = game.to_act
        legal = tuple(int(a) for a in game.legal_actions())
        to_call = max(0, game.bets[1 - seat] - game.bets[seat])
        node = Node(nid, seq, False, 0 if seat == b else 1, int(game.street), legal, {},
                    game.pot, to_call, contrib, -1, False, game.raises_this_street,
                    (game.stacks[b], game.stacks[1 - b]))
        nodes.append(node)
        for a in legal:
            child = game.clone()
            child.apply(Action(a))
            node.children[a] = visit(child, seq + (a,))
        return nid

    visit(Game(_DUMMY, 0), ())
    return nodes, index


NODES, INDEX = build()
DECISIONS = [n for n in NODES if not n.terminal]


def node_for(action_history):
    """Node id for an observation's action history (tuple of ObservedAction)."""
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
