#!/bin/sh
# Fastest tested way to create plo5bot/data from the text files only, on an 8-vCPU machine. Run from the repo root:
#
#   PY=python3 sh dev/make_data_8core.sh       (THREADS=8 and MINUTES=30 by default; needs numpy and gcc)
#
# Dry run 2026-11-08 on a text-only copy pinned to 8 vCPUs (4 cores + hyperthreads): 52.5 min in all (preflop 6.6 min,
# card buckets 40 s, deal pool 15.0 min, CFR 30 min). 19 of the 22 files came out byte-identical to the shipped ones;
# the 3 CFR files (5.0e9 traversals) differ but play like the shipped policy: 24 of 24 chops head to head with the
# shipped bot (-0.04 +/- 0.03 bb/deal), the same results against the zoo, adversaries and near-equilibrium bots.

set -e
PY=${PY:-python}
T=${THREADS:-8}

if [ -e plo5bot/data/cfr_policy.npy ] && [ "${FORCE:-0}" != 1 ]; then
    echo "plo5bot/data/cfr_policy.npy exists: refusing to overwrite the data (FORCE=1 to rebuild anyway)"; exit 2
fi

"$PY" dev/build_tables.py                          # pair_triple_rank, dense_scores
"$PY" -c "import sys; sys.path.insert(0, '.'); from plo5bot import tree; tree.save_cache()"  # tree_cache.npz
"$PY" dev/build_preflop.py --samples 20000 --band-samples 40000 --workers "$T"  # preflop_* tables
sh dev/cfr/build.sh
"$PY" dev/cfr/build_abstraction.py --workers "$T"    # abs_*, abstraction_meta.json
"$PY" dev/cfr/make_pool.py --n 10000000 --seed 1 --workers "$T"  # training deals in dev/cfr/out
"$PY" dev/cfr/export_tree.py                       # train tree
"$PY" dev/cfr/train.py --minutes "${MINUTES:-30}" --threads "$T" --explore 0.02 --prune-after 4 --floor-mult 1.05 \
    --eval-every 0
MCCFR_LIB=dev/cfr/out/libmccfr_eval.so "$PY" dev/cfr/train.py --resume --minutes 0 --export --eval-every 0
"$PY" -c "from plo5bot import strat_cfr; print(strat_cfr.PROBLEM or 'CFR policy OK')"
