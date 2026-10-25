#!/usr/bin/env python3
"""One row per device-gate run: how many device words diverge, and does the ownership probe fire.

The defect being characterised is that the divergence COUNT MOVED between identical device runs
(73 -> 56 -> 48). A number that moves has to be measured, not averaged, so this reads one finished
gate run's artifacts and appends one row to evidence/device-tail-determinism-2026-10-26.tsv:

  total/raw/out   device-vs-host word differences at the two H-length fields (the same arithmetic as
                  work/dbg_device_word_diff.py, recomputed here so a run is characterised from the
                  buffers it wrote, not from a gate line that only counts case-field pairs)
  gate_got        the gate's own `every declared-exact field is bit-identical to the oracle` got= in
                  the DEVICE section — an independent count of the same defect (two rulers, one fact)
  own_violations  the write-ownership probe's count (words before `out` that stage 3 changed)
  reads_sg_case0  the device's own sg-read counter for case 0: 17*2048 = 34816 if every stage-3 block
                  recomputed the probability, 2048 if only the sentinel block did
  poison_case10   the down_half case's poison-hole count: 2047 means a real value landed in a hole

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/tail_determinism.py <state> <run-log>
No device, no lock, no timing. Prints counts only; no duration appears anywhere.
"""
import os
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
OFF = {"raw": 1536, "out": 3584}
LEN = {"raw": 2048, "out": 2048}
TSV = os.path.join(LANE, "evidence", "device-tail-determinism-2026-10-26.tsv")
HDR = ["run", "state", "total_diff", "raw_diff", "out_diff", "gate_got", "own_violations",
       "reads_sg_case0", "poison_case10", "result"]


def words(name):
    a = array("I")
    with open(os.path.join(HERE, name), "rb") as fh:
        a.frombytes(fh.read())
    return a


def field_counts():
    host, dev = words("host_out.f32"), words("dev_out.f32")
    dsts = []
    for line in open(os.path.join(HERE, "cases.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        dsts.append(int(line.rstrip("\n").split("\t")[7]))
    n = {"raw": 0, "out": 0}
    for dst in dsts:
        for f in ("raw", "out"):
            for i in range(LEN[f]):
                if host[dst + OFF[f] + i] != dev[dst + OFF[f] + i]:
                    n[f] += 1
    return n, len(dsts)


def main():
    state, runlog = sys.argv[1], sys.argv[2]
    gate = open(runlog).read()
    devrun = open(os.path.join(HERE, "device-run.log")).read()

    n, n_cases = field_counts()
    got = None
    for line in gate.splitlines():
        if line.startswith("CHECK every declared-exact field is bit-identical"):
            got = line.split("got=")[1].split()[0]      # last one is the device section
    own = [l for l in devrun.splitlines() if l.startswith("OWNERSHIPSUMMARY")]
    if len(own) != 1:
        raise SystemExit("expected exactly 1 OWNERSHIPSUMMARY line, found %d — run is uncharacterised"
                         % len(own))
    ok = dict(tok.split("=", 1) for tok in own[0].split()[1:])
    if ok["lines"] != ok["violations"]:
        raise SystemExit("ownership probe printed %s lines for %s violations" % (ok["lines"],
                                                                                ok["violations"]))
    reads_sg = open(os.path.join(HERE, "dev_report.tsv")).read().splitlines()[1].split("\t")[9]
    poison = "?"
    for line in gate.splitlines():
        if "down_half: report=" in line:
            poison = line.split("report=")[1].split()[0]
            break
    else:
        for line in gate.splitlines():
            if line.startswith("CHECK poison holes are counted") and "got=43" in line:
                poison = "2048"                        # all cases matched their expectation
    result = [l for l in gate.splitlines() if l.startswith("RESULT")]
    row = [str(len([l for l in open(TSV)]) - 1) if os.path.exists(TSV) else "0", state,
           str(n["raw"] + n["out"]), str(n["raw"]), str(n["out"]), str(got), ok["violations"],
           reads_sg, poison, (result[0][:78] if result else "NO RESULT LINE")]
    new = not os.path.exists(TSV)
    with open(TSV, "a") as fh:
        if new:
            fh.write("# " + "\t".join(HDR) + "\n")
        fh.write("\t".join(row) + "\n")
    print("ROW " + " ".join("%s=%s" % (h, v) for h, v in zip(HDR, row)))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
