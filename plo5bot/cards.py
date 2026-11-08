"""Fast short-deck Omaha hand evaluation from a precomputed table.

Cards are ints 0..35 exactly as in evaluator.py (rank = c >> 2, suit = c & 3).
PT[pair_id, triple_id] holds 1 + the dense rank (0..1403, higher is better) of
the 5-card hand made of a 2-card set and a 3-card set, or 0 where they share a
card. The best Omaha hand (exactly 2 hole + exactly 3 board cards) is a max
over the hand's 10 pairs and the board's triples. dev/build_tables.py builds
the table from evaluator.py, and tests/test_tables.py checks it against
evaluator.omaha_score.
"""

import os
from itertools import combinations

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PT_FILE = os.path.join(DATA_DIR, "pair_triple_rank.npy")
SCORES_FILE = os.path.join(DATA_DIR, "dense_scores.npy")

NUM_CARDS = 36
NUM_PAIRS = 630       # C(36, 2)
NUM_TRIPLES = 7140    # C(36, 3)
NUM_RANKS = 1404      # distinct short-deck hand values


def pair_id(a, b):
    """Colex index of the 2-card set {a, b}."""
    if a > b:
        a, b = b, a
    return b * (b - 1) // 2 + a


def triple_id(a, b, c):
    """Colex index of the 3-card set {a, b, c}."""
    a, b, c = sorted((a, b, c))
    return c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a


# PAIR_ID[a][b] for every ordered pair of distinct cards.
PAIR_ID = [[pair_id(a, b) if a != b else -1 for b in range(NUM_CARDS)] for a in range(NUM_CARDS)]
# PAIR_CARDS[pid] = (a, b) with a < b; TRIPLE_CARDS[tid] = (a, b, c) ascending.
PAIR_CARDS = np.array([(a, b) for b in range(NUM_CARDS) for a in range(b)], dtype=np.int8)
TRIPLE_CARDS = np.array(
    [(a, b, c) for c in range(NUM_CARDS) for b in range(c) for a in range(b)], dtype=np.int8
)

_P5 = tuple(combinations(range(5), 2))
_T = {n: tuple(combinations(range(n), 3)) for n in (3, 4, 5)}


def hand_pair_ids(hole):
    """The 10 pair ids of a 5-card hole hand."""
    return [PAIR_ID[hole[i]][hole[j]] for i, j in _P5]


def board_triple_ids(board):
    """Triple ids of every 3-card subset of a 3-, 4- or 5-card board."""
    return [triple_id(board[i], board[j], board[k]) for i, j, k in _T[len(board)]]


def load_table():
    """Load PT (uint16, shape (630, 7140)) and the dense-rank -> score map."""
    pt = np.load(PT_FILE, allow_pickle=False)
    scores = np.load(SCORES_FILE, allow_pickle=False)
    if pt.shape != (NUM_PAIRS, NUM_TRIPLES) or pt.dtype != np.uint16:
        raise ValueError(f"bad table {PT_FILE}: {pt.shape} {pt.dtype}")
    if scores.shape != (NUM_RANKS,):
        raise ValueError(f"bad table {SCORES_FILE}: {scores.shape}")
    return pt, scores


PT, DENSE_SCORES = load_table()


def omaha_rank(hole, board):
    """Dense rank (0..1403) of the best exactly-2-plus-3 hand; board has 3-5 cards."""
    return int(PT[np.ix_(hand_pair_ids(hole), board_triple_ids(board))].max()) - 1


def omaha_score(hole, board):
    """Same value as evaluator.omaha_score for a 5-card board."""
    return int(DENSE_SCORES[omaha_rank(hole, board)])


def board_pair_best(board):
    """For a 3-5 card board: best (1 + dense rank) of every pair id with that board.

    Returns a uint16 vector of length 630; pairs that share a card with every
    board triple get 0. A hand's value is the max over its 10 pair ids.
    """
    return PT[:, board_triple_ids(board)].max(axis=1)
