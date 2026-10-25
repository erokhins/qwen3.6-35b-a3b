"""Apply a named witness mutation to moe_gather_kernels.h.

`python3 work/mutate.py A|B` — used by work/negctl.sh. Heredoc-into-an-interpreter is a blocked form
in this repo (pitfalls/50 §"CC Safety Net blocked forms"), so the mutations live in a file.

  A  read mg_unit_elt's success as a refusal again — the exact inversion that made all 70 expert
     cases produce no value, and the thing the new hard-failure checks must refuse.
  B  make MG_MODE_EXPERT_ROTATE compare the array index against the MoE block number again, so the
     resolve returns NULL and mg_verify_plan must refuse the plan before a case runs.
"""
import sys

A_OLD = """            if (mg_unit_elt(p, u, role, row, j, MG_H, mode, &w,
                            (mode == MG_MODE_CORRECT) ? &w32 : 0)) {"""
A_NEW = """            if (!mg_unit_elt(p, u, role, row, j, MG_H, mode, &w,
                             (mode == MG_MODE_CORRECT) ? &w32 : 0)) {"""
B_OLD = "            if (r->block != c->block || r->role != role) return 0;"
B_NEW = "            if (r->block != b->index || r->role != role) return 0;"

PAIRS = {"A": (A_OLD, A_NEW), "B": (B_OLD, B_NEW)}

which = sys.argv[1] if len(sys.argv) > 1 else "?"
if which not in PAIRS:
    raise SystemExit("usage: python3 work/mutate.py A|B")
old, new = PAIRS[which]
path = "moe_gather_kernels.h"
text = open(path).read()
if text.count(old) != 1:
    raise SystemExit("mutation %s: anchor found %d times, refusing" % (which, text.count(old)))
open(path, "w").write(text.replace(old, new, 1))
print("mutation %s applied to %s" % (which, path))
