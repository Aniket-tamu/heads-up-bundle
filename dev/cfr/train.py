"""Driver for dev/cfr/mccfr.c: train, checkpoint, measure exploitability, export the policy.

PY=/home/aniavrh/.claude/jobs/2010e17e/tmp/bin/py
$PY dev/cfr/train.py --minutes 120 --threads 16          # train from scratch (log: dev/cfr/out/train_log.jsonl)
$PY dev/cfr/train.py --resume --minutes 0                # continue from dev/cfr/out/ckpt_*.npy
$PY dev/cfr/train.py --resume --minutes 0 --eval-n 100000  # evaluate the checkpoint only
$PY dev/cfr/train.py --resume --minutes 0 --export       # write plo5bot/data/cfr_policy.npy + cfr_meta.json

Algorithm: external-sampling MCCFR, Hogwild on OpenMP threads, with linear CFR: after
epoch k (a fixed number of seconds of traversals) regrets and average-strategy sums are
multiplied by k / (k + 1). Optional regret-based pruning after a warm-up (see mccfr.c).

Exploitability estimate (abstract game, same imperfect-recall info sets): for each
player p a best response to the average strategy is computed on views of held-out deals
(dev/cfr/out/test1m, rows [0, n)), then evaluated on other held-out deals (rows [n, 2n)),
both seat assignments of every deal. Reported in mbb/hand: (BR as button + BR as big
blind) / 2, which is what a best-responding opponent wins per hand in a duplicate match.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(ROOT, "plo5bot", "data")
LIB = os.environ.get("MCCFR_LIB", os.path.join(OUT, "libmccfr.so"))  # another name while a trainer runs
SRC = os.path.join(HERE, "mccfr.c")
LOG = os.path.join(OUT, "train_log.jsonl")
CKPT = os.path.join(OUT, "ckpt")
NB = (256, 128, 128, 128)

sys.path.insert(0, ROOT)


c_dp = ctypes.POINTER(ctypes.c_double)
c_fp = ctypes.POINTER(ctypes.c_float)


def ptr(a, t=ctypes.c_void_p):
    assert a.flags["C_CONTIGUOUS"]
    return a.ctypes.data_as(t)


def build():
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return
    subprocess.check_call([
        "gcc", "-std=c11", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared",
        "-Wall", "-Wextra", "-o", LIB, SRC, "-lm"
    ])


def load_lib():
    build()
    lib = ctypes.CDLL(LIB)
    vp, i64, i32, dbl = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_double
    lib.cfr_setup.argtypes = [i32, vp, vp, vp, vp, vp, vp, vp, vp, i64, vp, vp, i64, vp, vp]
    lib.cfr_setup.restype = i32
    lib.cfr_run.argtypes = [dbl, i64, i32, ctypes.c_uint64, i32, dbl, dbl, dbl, dbl, ctypes.POINTER(i64)]
    lib.cfr_run.restype = i64
    lib.cfr_discount.argtypes = [dbl, dbl, dbl, i32]
    lib.cfr_strategy.argtypes = [i32, vp, vp, i32]
    lib.cfr_br.argtypes = [i32, i32, vp, i32, vp, vp, vp, vp, i32, dbl, dbl, vp]
    lib.cfr_br.restype = i32
    return lib


def load_tree():
    names = ["actor", "street", "nact", "child", "action", "term", "contrib", "value", "dec", "iset"]
    if not all(os.path.exists(os.path.join(OUT, f"tree_{n}.npy")) for n in names):
        subprocess.check_call([sys.executable, os.path.join(HERE, "export_tree.py")])
    t = {n: np.ascontiguousarray(np.load(os.path.join(OUT, f"tree_{n}.npy"), allow_pickle=False)) for n in names}
    with open(os.path.join(OUT, "tree_meta.json")) as f:
        t["meta"] = json.load(f)
    assert tuple(t["meta"]["buckets"]) == NB
    return t


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def abstraction_hashes():
    return {
        f: file_sha(os.path.join(DATA, f))
        for f in sorted(os.listdir(DATA))
        if f.startswith("abs_") or f in ("abstraction_meta.json", "preflop_class.npy")
    }


class Trainer:
    def __init__(self, threads, pool_dir=OUT):
        self.lib = load_lib()
        self.t = load_tree()
        self.threads = threads
        self.ni = int(self.t["meta"]["infosets"])
        self.reg = np.zeros(self.ni * 4, np.float64)
        self.ssum = np.zeros(self.ni * 4, np.float64)
        self.bk = np.ascontiguousarray(np.load(os.path.join(pool_dir, "pool_buckets.npy"), allow_pickle=False))
        self.win = np.ascontiguousarray(np.load(os.path.join(pool_dir, "pool_winner.npy"), allow_pickle=False))
        assert self.bk.dtype == np.uint16 and self.bk.shape[1:] == (2, 4) and self.win.dtype == np.int8
        self.state = {
            "epoch": 0,
            "traversals": 0,
            "nodes": 0,
            "train_seconds": 0.0,
            "pool": os.path.relpath(pool_dir, ROOT),
            "pool_deals": int(len(self.win)),
        }
        self._setup()

    def _setup(self):
        t = self.t
        nb = np.array(NB, np.int32)
        self._nb = nb
        r = self.lib.cfr_setup(
            len(t["actor"]), ptr(t["child"]), ptr(t["actor"]), ptr(t["street"]), ptr(t["nact"]),
            ptr(t["term"]), ptr(t["value"]), ptr(t["iset"]), ptr(nb), self.ni,
            ptr(self.reg), ptr(self.ssum), len(self.win), ptr(self.bk), ptr(self.win),
        )
        assert r == 0, r

    def save(self, path=CKPT):
        for name, arr in (("reg", self.reg), ("ssum", self.ssum)):
            np.save(path + f"_{name}.tmp.npy", arr)
            os.replace(path + f"_{name}.tmp.npy", path + f"_{name}.npy")
        with open(path + "_state.tmp.json", "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(path + "_state.tmp.json", path + "_state.json")

    def load(self, path=CKPT):
        self.reg[:] = np.load(path + "_reg.npy")
        self.ssum[:] = np.load(path + "_ssum.npy")
        with open(path + "_state.json") as f:
            self.state.update(json.load(f))

    def run_epoch(self, seconds, seed, prune, prune_prob, prune_thr, floor, max_trav=0, eps=0.0):
        nodes = ctypes.c_int64(0)
        t0 = time.time()
        n = self.lib.cfr_run(seconds, max_trav, self.threads, seed, int(prune), prune_prob, prune_thr, floor, eps,
                             ctypes.byref(nodes))
        dt = time.time() - t0
        self.state["traversals"] += int(n)
        self.state["nodes"] += int(nodes.value)
        self.state["train_seconds"] += dt
        return int(n), int(nodes.value), dt

    def discount(self, k):
        f = k / (k + 1.0)
        self.lib.cfr_discount(f, f, f, self.threads)

    def strategy(self, which=0):
        out = np.empty(self.ni * 4, np.float32)
        mass = np.empty(self.ni, np.float64)
        self.lib.cfr_strategy(which, ptr(out), ptr(mass), self.threads)
        return out, mass


def views(pool_dir, start, count):
    """Both seat assignments of pool rows [start, start + count): bk (2, 4, 2n) (player, street, view), w [2n]."""
    bk = np.load(os.path.join(pool_dir, "pool_buckets.npy"), mmap_mode="r")[start:start + count]
    win = np.load(os.path.join(pool_dir, "pool_winner.npy"), mmap_mode="r")[start:start + count]
    n = len(win)
    out = np.empty((2, 4, 2 * n), np.uint16)
    out[0, :, :n] = bk[:, 0, :].T
    out[1, :, :n] = bk[:, 1, :].T
    out[0, :, n:] = bk[:, 1, :].T
    out[1, :, n:] = bk[:, 0, :].T
    w = np.concatenate((win, -win)).astype(np.float32)
    return np.ascontiguousarray(out), w


def evaluate(tr, n, n_held=None, which=0, selfplay=True, tau=1e-6, margin=40.0, sig=None):
    """Exploitability estimate of the average (which=0) or current (1) strategy, in chips per hand.

    The best response is chosen on both seatings of training-pool rows [0, n) and evaluated
    on both seatings of held-out rows [0, n_held) of dev/cfr/out/test1m. It deviates from the
    strategy only where the gain is significant (margin, see mccfr.c), so the held-out number
    is a lower bound of the abstract-game (local best response) exploitability.
    sig: evaluate this strategy array (info sets x 4) instead of the trainer's.
    """
    mass = None
    if sig is None:
        sig, mass = tr.strategy(which)
    bkA, wA = views(OUT, 0, n)
    bkB, wB = views(os.path.join(OUT, "test1m"), 0, n_held or n)
    K = len(wA)
    res = {"views_br": K, "views_held_out": len(wB), "tau": tau, "margin": margin,
           "which": ["average", "current"][which]}
    vals = {}
    t0 = time.time()
    for p in (0, 1):
        bra = np.empty(tr.ni, np.int8)
        outA = np.empty(K, np.float32)
        r = tr.lib.cfr_br(0, p, ptr(sig), K, ptr(bkA), ptr(wA), ptr(bra), ptr(outA), tr.threads, tau, margin, None)
        assert r == 0
        outB = np.empty(len(wB), np.float32)
        r = tr.lib.cfr_br(1, p, ptr(sig), len(wB), ptr(bkB), ptr(wB), ptr(bra), ptr(outB), tr.threads, tau, margin,
                          None)
        assert r == 0
        vals[p] = outB
        res[f"br{p}_in_sample"] = float(outA.mean())
        res[f"br{p}_held_out"] = float(outB.mean())
    if selfplay:
        out2 = np.empty(len(wB), np.float32)
        tr.lib.cfr_br(2, 0, ptr(sig), len(wB), ptr(bkB), ptr(wB), None, ptr(out2), tr.threads, tau, margin, None)
        res["selfplay_button"] = float(out2.mean())
    # per deal: both seat views, both best responses -> what a best responder wins per hand
    half = len(wB) // 2
    e = (vals[0] + vals[1]) / 2.0
    e_deal = (e[:half] + e[half:]) / 2.0
    res["expl_chips"] = float(e_deal.mean())
    res["expl_se_chips"] = float(e_deal.std(ddof=1) / np.sqrt(half))
    res["expl_mbb"] = 1000.0 * res["expl_chips"] / 2.0
    res["expl_se_mbb"] = 1000.0 * res["expl_se_chips"] / 2.0
    res["expl_in_sample_mbb"] = 1000.0 * (res["br0_in_sample"] + res["br1_in_sample"]) / 4.0
    res["eval_seconds"] = time.time() - t0
    res["unvisited_infosets"] = int((mass <= 0).sum()) if mass is not None else None
    return res


def head_to_head(tr, sig_x, sig_y, n, tau=1e-6):
    """Chips per hand won by strategy sig_x against sig_y, both seats of both seatings of
    held-out rows [0, n) (duplicate style, exact expectation over both strategies)."""
    bk, w = views(os.path.join(OUT, "test1m"), 0, n)
    K = len(w)
    u = np.empty((2, K), np.float32)
    for p in (0, 1):
        r = tr.lib.cfr_br(2, p, ptr(sig_y), K, ptr(bk), ptr(w), None, ptr(u[p]), tr.threads, tau, 0.0, ptr(sig_x))
        assert r == 0
    e = u.mean(axis=0)
    e_deal = (e[:K // 2] + e[K // 2:]) / 2.0
    return float(e_deal.mean()), float(e_deal.std(ddof=1) / np.sqrt(len(e_deal)))


def to_policy(t, sig, mass):
    """Slot-order strategy (info sets x 4) -> uint8 (decisions, 256, 4) by Action value, rows sum to 255.

    Largest-remainder rounding; rows with mass <= 0 (never visited) are all zero.
    """
    sig = sig.reshape(-1, 4)
    dec_nodes = np.flatnonzero(t["actor"] >= 0)
    D = len(dec_nodes)
    assert np.array_equal(t["dec"][dec_nodes], np.arange(D))
    pol = np.zeros((D, 256, 4), np.uint8)
    mass_tab = np.zeros((D, 256), np.float64)
    for n in dec_nodes:
        d, nb, k, base = t["dec"][n], NB[t["street"][n]], t["nact"][n], t["iset"][n]
        p = sig[base:base + nb, :k].astype(np.float64)
        q = np.floor(p * 255.0).astype(np.int64)
        rem = 255 - q.sum(axis=1)
        order = np.argsort(-(p * 255.0 - q), axis=1, kind="stable")
        for r in range(k):
            q[np.arange(nb), order[:, r]] += (rem > r)
        q[mass[base:base + nb] <= 0] = 0
        pol[d, :nb, t["action"][n, :k]] = q.T.astype(np.uint8)
        mass_tab[d, :nb] = mass[base:base + nb]
    return pol, mass_tab, dec_nodes


def export(tr, policy_path=os.path.join(DATA, "cfr_policy.npy"), meta_path=os.path.join(DATA, "cfr_meta.json"),
           extra=None, conf_path=None):
    """Average strategy -> policy file + meta; the current strategy and visit mass go to dev/cfr/out."""
    t = tr.t
    sig, mass = tr.strategy(0)
    pol, mass_tab, dec_nodes = to_policy(t, sig, mass)
    np.save(policy_path, pol, allow_pickle=False)
    stem = os.path.splitext(os.path.basename(policy_path))[0]
    np.save(os.path.join(OUT, f"{stem}_mass.npy"), mass_tab.astype(np.float32), allow_pickle=False)
    # confidence: 4 * log2(1 + LCFR-weighted number of average-strategy updates), uint8
    conf = np.clip(np.round(4.0 * np.log2(1.0 + mass_tab)), 0, 255).astype(np.uint8)
    if conf_path:
        np.save(conf_path, conf, allow_pickle=False)
    cur, _ = tr.strategy(1)
    cur_pol, _, _ = to_policy(t, cur, np.where(mass > 0, 1.0, 0.0))
    np.save(os.path.join(OUT, f"{stem}_current.npy"), cur_pol, allow_pickle=False)
    meta = {
        "version": 1,
        "file": os.path.basename(policy_path),
        "shape": list(pol.shape),
        "layout": "policy[d, bucket, action] = round(255 * P(action)); d = index of the node in "
                  "plo5bot.tree.DECISIONS (= position among non-terminal NODES); bucket = "
                  "plo5bot.abstraction.bucket(street, hole, board) of the player to act; action = game.Action value",
        "rows": "visited info sets sum to exactly 255 (largest remainder rounding); all-zero row = never "
                "visited in training (use a fallback); postflop buckets == 128 are unused",
        "confidence": (os.path.basename(conf_path) + ": uint8 (decisions, 256) = round(4 * log2(1 + w)), w = "
                       "LCFR-weighted number of average-strategy updates of the info set; 0 = never visited")
                      if conf_path else None,
        "strategy": "Linear-CFR-weighted average strategy of external-sampling MCCFR (dev/cfr/TRAINING.md)",
        "buckets": list(NB),
        "decisions": int(len(dec_nodes)),
        "decision_node_ids_sha256": hashlib.sha256(np.asarray(dec_nodes, np.int32).tobytes()).hexdigest(),
        "decision_node_ids": [int(x) for x in dec_nodes],  # row d -> plo5bot.tree node id
        "actions": {str(i): name for i, name in enumerate(("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT"))},
        "tree_meta": t["meta"],
        "abstraction_hashes": abstraction_hashes(),
        "abstraction_sha256": abstraction_hashes(),
        "pool": tr.state["pool"],
        "pool_deals": tr.state["pool_deals"],
        "unvisited_infosets": int((mass <= 0).sum()),
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return pol, meta


def log(rec):
    rec = dict(rec, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(rec), flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0, help="training wall time for this run")
    ap.add_argument("--epoch-sec", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prune-after", type=int, default=3, help="first epoch with pruning (0 = never)")
    ap.add_argument("--prune-prob", type=float, default=0.95)
    ap.add_argument("--prune-c", type=float, default=2000.0, help="prune threshold = -prune_c * epoch")
    ap.add_argument("--floor-mult", type=float, default=3.0, help="regret floor = floor_mult * threshold")
    ap.add_argument("--explore", type=float, default=0.03, help="opponent exploration epsilon in traversals")
    ap.add_argument("--eval-every", type=int, default=15, help="epochs between evaluations (0 = never)")
    ap.add_argument("--eval-n", type=int, default=50000, help="held-out deals per BR sample")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--ckpt", default=CKPT, help="checkpoint path prefix for --resume")
    ap.add_argument("--margin", type=float, default=40.0, help="BR noise margin (chips), see mccfr.c")
    ap.add_argument("--which", type=int, nargs="+", default=[0], help="evaluate 0 = average, 1 = current")
    ap.add_argument("--export", action="store_true", help="export the policy at the end")
    ap.add_argument("--policy-out", default=os.path.join(DATA, "cfr_policy.npy"))
    ap.add_argument("--meta-out", default=os.path.join(DATA, "cfr_meta.json"))
    ap.add_argument("--conf-out", default=os.path.join(DATA, "cfr_conf.npy"))
    ap.add_argument("--max-trav", type=int, default=0, help="traversals per epoch instead of seconds (tests)")
    a = ap.parse_args()

    tr = Trainer(a.threads)
    if a.resume:
        tr.load(a.ckpt)
    log({"event": "start", "args": vars(a), "infosets": tr.ni, "state": tr.state})
    t_end = time.time() + 60.0 * a.minutes
    stop_file = os.path.join(OUT, "STOP")
    while time.time() < t_end - 1.0 and not os.path.exists(stop_file):
        k = tr.state["epoch"] + 1
        prune = a.prune_after > 0 and k >= a.prune_after
        thr = -a.prune_c * k
        sec = min(a.epoch_sec, max(1.0, t_end - time.time()))
        n, nodes, dt = tr.run_epoch(
            0 if a.max_trav else sec, a.seed * 1000003 + k, prune, a.prune_prob, thr,
            a.floor_mult * thr if prune else -1e300, a.max_trav, a.explore,
        )
        tr.discount(k)
        tr.state["epoch"] = k
        log({
            "event": "epoch", "epoch": k, "traversals": n, "trav_per_s": n / max(dt, 1e-9),
            "nodes_per_trav": nodes / max(n, 1), "prune": prune, "total_traversals": tr.state["traversals"],
            "train_seconds": round(tr.state["train_seconds"], 1),
        })
        if a.ckpt_every and k % a.ckpt_every == 0:
            tr.save()
        if a.eval_every and k % a.eval_every == 0:
            log(dict(event="eval", epoch=k, **evaluate(tr, a.eval_n)))
        if a.max_trav and k >= a.minutes:  # test mode: --minutes counts epochs
            break
    if a.minutes > 0:
        tr.save()
    if a.eval_every and a.minutes == 0:
        for which in a.which:
            log(dict(event="eval", epoch=tr.state["epoch"], **evaluate(tr, a.eval_n, which=which, margin=a.margin)))
    if a.export:
        _, meta = export(tr, a.policy_out, a.meta_out, conf_path=a.conf_out)
        log({"event": "export", "unvisited_infosets": meta["unvisited_infosets"]})


if __name__ == "__main__":
    main()
