# Original lines 1-22 are outside the supplied photos.





















import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from plo5bot import abstraction as ab  # noqa: E402
from plo5bot import tree  # noqa: E402

OUT = os.path.join(ROOT, "dev", "cfr", "out")
TERM_FOLD_BUTTON, TERM_FOLD_BB, TERM_SHOWDOWN = 1, 2, 3


def export(out_dir=OUT):
    nodes = tree.NODES
    n = len(nodes)
    nb = [ab.num_buckets(s) for s in range(4)]
    actor = np.full(n, -1, np.int8)
    street = np.zeros(n, np.int8)
    nact = np.zeros(n, np.int8)
    child = np.full((n, 4), -1, np.int32)
    action = np.full((n, 4), -1, np.int8)
    term = np.zeros(n, np.int8)
    contrib = np.zeros((n, 2), np.int16)
    value = np.zeros(n, np.int16)
    dec = np.full(n, -1, np.int32)
    iset = np.full(n, -1, np.int32)
    nd = 0
    ni = 0
    for node in nodes:
        i = node.id
        assert nodes[i] is node
        street[i] = node.street
        contrib[i] = node.contrib
        if node.terminal:
            if node.fold_by == 0:
                term[i], value[i] = TERM_FOLD_BUTTON, -node.contrib[0]
            elif node.fold_by == 1:
                term[i], value[i] = TERM_FOLD_BB, node.contrib[1]
            else:
                assert node.contrib[0] == node.contrib[1], node
                term[i], value[i] = TERM_SHOWDOWN, node.contrib[0]
            continue
        actor[i] = node.actor
        nact[i] = len(node.legal)
        for j, a in enumerate(node.legal):
            child[i, j] = node.children[a]
            action[i, j] = a
        assert list(node.legal) == sorted(node.legal)
        dec[i] = nd
        iset[i] = ni
        nd += 1
        ni += nb[node.street]
    meta = {
        "nodes": n,
        "decisions": nd,
        "infosets": ni,
        "buckets": nb,
        "decisions_per_street": tree.street_counts(),
        "terminals": int((term > 0).sum()),
        "fold_terminals": int(((term == 1) | (term == 2)).sum()),
        "showdown_terminals": int((term == 3).sum()),
        "slot_order": "legal order of the node: FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT (when legal)",
        "infoset": "tree_iset[node] + bucket of the actor on the node's street",
    }
    os.makedirs(out_dir, exist_ok=True)
    arrays = dict(actor=actor, street=street, nact=nact, child=child, action=action, term=term,
                  contrib=contrib, value=value, dec=dec, iset=iset)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"tree_{name}.npy"), arr, allow_pickle=False)
    with open(os.path.join(out_dir, "tree_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return arrays, meta


def es_cost(arrays):
    """Expected nodes touched by one external-sampling traversal (uniform opponent)."""
    child, actor, nact = arrays["child"], arrays["actor"], arrays["nact"]
    n = len(actor)
    cost = np.zeros((2, n))
    for p in range(2):
        for i in range(n - 1, -1, -1):            # children have larger ids (DFS preorder)
            if actor[i] < 0:
                cost[p, i] = 1
                continue
            c = [cost[p, child[i, j]] for j in range(nact[i])]
            cost[p, i] = 1 + (sum(c) if actor[i] == p else sum(c) / len(c))
    return cost[:, 0]


if __name__ == "__main__":
    arrays, meta = export()
    print(json.dumps(meta, indent=1))
    print("expected ES traversal size (uniform opponent), traverser button / bb:", es_cost(arrays))
