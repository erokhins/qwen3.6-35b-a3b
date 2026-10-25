"""Diagnostic for the four residual finding groups after the weight-read polarity fix.

Each section answers one question with a count, not a theory:
  A real-row weights: is the pinned topk_weights vector already bf16-exact? If yes,
    MG_MODE_CAST_AFTER structurally cannot move a real-row combine case, and calling that case
    MG_EXP_ALT is a mislabelled must-not-catch case, not a kernel defect.
  B the synthetic row's combine injections: which raw slice does the kernel read, and which one did
    the oracle derive its expectation from? A PINNED-source injection beside a KERNEL_RAW honest
    case judges two different inputs against each other.
  C MG_MODE_ROW_OFF_1: how many elements of the kernel's row-rotate fall outside the unit, i.e.
    become refused reads, under (row+1) % MG_TWO_I versus (row+1) % u->rows.
  D the ROTATE/HALVES alternatives: which units does each side actually read, and are they the same?
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LANE))
sys.path.insert(0, os.path.join(LANE))
import oracle_arm as oa                                    # noqa: E402

# ---- A: the pinned weights and their bf16 rounding --------------------------------------------
import json                                               # noqa: E402
_man = json.load(open(os.path.join(REPO, "checkpoints/dump-manifest-2026-10-26.json")))
oa.DUMP = {r["name"]: r for r in _man["dumps"]}
for tag in ("layer0", "layer39"):
    w, _ = oa.dump_arrays(tag + ".topk_weights")
    cast = oa.bf16_bits_to_f32(oa.f32_to_bf16_bits(w))
    same = int(np.count_nonzero(cast.view(np.uint32) != w.view(np.uint32)))
    print("A %s topk_weights shape=%s words=%d bf16-rounding-changes=%d -> cast-before == cast-after"
          % (tag, w.shape, w.size, same))

# ---- B: the synthetic row's combine raw source -------------------------------------------------
plan = oa.Plan("work", 1)
print("B combine_src column of cases.tsv (0=PINNED slice, 1=KERNEL_RAW):")
import collections
cnt = collections.Counter()
for line in open(os.path.join(LANE, "work", "cases.tsv")):
    if line.startswith("#") or not line.strip():
        continue
    f = line.rstrip("\n").split("\t")
    if int(f[1]) == 1:                        # combine cases
        cnt[(f[3], f[9], int(f[15]))] += 1
for k in sorted(cnt):
    print("B   mode=%-28s class=%-16s combine_src=%d  n=%d" % (k[0], k[1], k[2], cnt[k]))

# ---- C: the row-rotate modulus -----------------------------------------------------------------
H, I = oa.HC["MG_H"], oa.HC["MG_I"]
for role, rows in (("gate/up", I), ("down", H)):
    for mod, name in ((2 * I, "MG_TWO_I (what the kernel did)"), (rows, "u->rows (what the oracle does)")):
        bad = sum(1 for r in range(rows) if (r + 1) % mod >= rows)
        print("C %-8s rows=%d modulus=%-5d %-28s refused_rows=%d" % (role, rows, mod, name, bad))

# ---- D: which units each side reads for the rotate/halves cases --------------------------------
lines = [l.rstrip("\n") for l in open(os.path.join(LANE, "work", "plan.tsv")) if l.strip()]
U = [l.split("\t") for l in lines if l.startswith("U")]
C = [l.split("\t") for l in lines if l.startswith("C")]
Cs = [l for l in open(os.path.join(LANE, "work", "cases.tsv")) if not l.startswith("#") and l.strip()]
for row in Cs:
    f = row.split("\t")
    if f[3] in ("MG_MODE_EXPERT_ROTATE", "MG_MODE_HALVES_SWAPPED"):
        ci = int(f[0])
        c = C[ci]
        print("D case=%d %-26s block=%s u_gate=%s u_up=%s u_down=%s -> units (%s,%s,%s) experts (%s,%s,%s)"
              % (ci, f[3], c[3], c[6], c[7], c[8],
                 U[int(c[6])][4], U[int(c[7])][4], U[int(c[8])][4],
                 U[int(c[6])][5], U[int(c[7])][5], U[int(c[8])][5]))
print("D note: the kernel resolves ROTATE as units[block.unit_base + (set_pos+1)%main_n*3 + role] and")
print("D halves-swapped as the case's own u_up for gate and u_gate for up.")
