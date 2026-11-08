"""ctypes driver for dev/cfr/features.c (offline only).

features(cards, workers) -> (feats int32 (N, 2, 9), winner int8 (N,)) with the same
integer features as plo5bot.abstraction (column 0 = preflop class, filled here).
The library is (re)built with dev/cfr/build.sh when features.c is newer than it.
"""

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB = os.path.join(HERE, "out", "libfeatures.so")
SRC = os.path.join(HERE, "features.c")
PT_FILE = os.path.join(ROOT, "plo5bot", "data", "pair_triple_rank.npy")

F_PF, F_FLOP_EQ, F_FLOP_NOW, F_TURN_EQ, F_TURN_NOW, F_TURN_SQ, F_RIVER_EQ, F_FLOP_SQT, F_FLOP_SQR = range(9)
NFEAT = 9

_lib = None


def build(force=False):
    if force or not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.run(["sh", os.path.join(HERE, "build.sh")], check=True)


def _u8(a):
    return np.ascontiguousarray(a, dtype=np.uint8)


def load():
    """Load the library and hand it PT plus the sample tables of plo5bot.abstraction."""
    global _lib, _keep
    if _lib is not None:
        return _lib
    build()
    from plo5bot import abstraction as ab
    lib = ctypes.CDLL(LIB)
    p = ctypes.c_void_p
    lib.feat_init.argtypes = [ctypes.c_char_p, p, ctypes.c_int, p, ctypes.c_int]
    lib.feat_init.restype = ctypes.c_int
    lib.feat_batch.argtypes = [p, ctypes.c_int64, p, p, ctypes.c_int]
    lib.feat_batch.restype = None
    lib.showdown_values.argtypes = [p, ctypes.c_int64, p, ctypes.c_int]
    lib.showdown_values.restype = None
    flop, turn = _u8(ab.FLOP_OPP), _u8(ab.TURN_OPP)
    rc = lib.feat_init(PT_FILE.encode(), flop.ctypes.data, flop.shape[1], turn.ctypes.data, turn.shape[1])
    if rc != 0:
        raise RuntimeError(f"feat_init failed with {rc}")
    _keep = (flop, turn)
    _lib = lib
    return lib


def random_deals(n, rng, chunk=500_000):
    """n random deals as uint8 (n, 15): seat 0 hole (sorted), seat 1 hole (sorted), board (deal order)."""
    out = np.empty((n, 15), dtype=np.uint8)
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        d = np.argsort(rng.random((m, 36)), axis=1)[:, :15].astype(np.uint8)
        d[:, 0:5].sort(axis=1)
        d[:, 5:10].sort(axis=1)
        out[s:s + m] = d
    return out


def features(cards, workers=16):
    from plo5bot import abstraction as ab
    lib = load()
    cards = np.ascontiguousarray(cards, dtype=np.uint8)
    n = len(cards)
    feats = np.empty((n, 2, NFEAT), dtype=np.int32)
    winner = np.empty(n, dtype=np.int8)
    lib.feat_batch(cards.ctypes.data, n, feats.ctypes.data, winner.ctypes.data, workers)
    feats[:, 0, F_PF] = ab.preflop_class(cards[:, 0:5])
    feats[:, 1, F_PF] = ab.preflop_class(cards[:, 5:10])
    return feats, winner


def showdown_values(cards, workers=16):
    lib = load()
    cards = np.ascontiguousarray(cards, dtype=np.uint8)
    out = np.empty((len(cards), 2), dtype=np.uint16)
    lib.showdown_values(cards.ctypes.data, len(cards), out.ctypes.data, workers)
    return out
