#!/usr/bin/env bash
# negctl.sh — watch the hard-failure checks fire, then restore. WITNESS ONLY: it mutates the source,
# rebuilds, runs the host arm, and always restores before it exits (the trap runs the restore even on
# a kill). Nothing here is a fix; the point is that each new check is seen going red on a mutation a
# reader can name, which is what makes the green run mean something (pitfalls/40 §"gates must fail for
# something").
#
# Mutations (work/mutate.py holds both, because heredoc-into-an-interpreter is a blocked form here):
#   A  the shipped defect, re-introduced: read mg_unit_elt's success as a refusal, so every expert
#      case produces no value again.
#   B  make MG_MODE_EXPERT_ROTATE compare the array index against the MoE block number again, so a
#      resolve returns NULL and mg_verify_plan must refuse the plan before any case runs.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LANE=$(cd "$HERE/.." && pwd)
cd "$LANE" || exit 1
BK=$LANE/work/negctl-backup
mkdir -p "$BK"
cp moe_gather_kernels.h "$BK/moe_gather_kernels.h.orig"

restore() {
  cp "$BK/moe_gather_kernels.h.orig" "$LANE/moe_gather_kernels.h"
  if ./build.sh >/dev/null 2>&1; then echo "RESTORED honest build ok"; else echo "RESTORED build FAILED"; fi
}
trap restore EXIT INT TERM

run_mutation() {
  name="$1"
  if ! ./build.sh >/dev/null 2>&1; then echo "$name: BUILD FAILED (mutation does not compile)"; return; fi
  ./gate --check-only > "work/negctl-$name.log" 2>&1
  echo "--- $name RESULT: $(grep '^RESULT' "work/negctl-$name.log")"
  echo "--- $name GATE/plan refusals:"
  grep -E '^(GATE FAIL|FAIL plan|FAIL argv)' "work/negctl-$name.log" | head -4
  echo "--- $name new-check names that went red (unique):"
  grep '^CHECK.*ok=0' "work/negctl-$name.log" | sed 's/^CHECK name=//; s/ case=[0-9]*/ case=N/; s/ ok=0.*//' | sort -u | head -14
  echo "--- $name aggregate lines:"
  grep -E '^(INTEGRITY|INJECTIONS|ALTREACH)' "work/negctl-$name.log" | head -6
  echo "--- $name count of red checks: $(grep -c '^CHECK.*ok=0' "work/negctl-$name.log")"
}

echo "=== MUTATION A: invert mg_unit_elt's success test back (the defect that shipped) ==="
python3 work/mutate.py A || { restore; exit 1; }
run_mutation A

restore
cp moe_gather_kernels.h "$BK/moe_gather_kernels.h.orig"

echo "=== MUTATION B: EXPERT_ROTATE resolves NULL on a block whose MoE number is not its index ==="
python3 work/mutate.py B || { restore; exit 1; }
run_mutation B

restore
echo "=== final state: honest build ==="
./gate --check-only > work/negctl-final.log 2>&1
grep -E '^(RESULT|INJECTIONS|INTEGRITY)' work/negctl-final.log
