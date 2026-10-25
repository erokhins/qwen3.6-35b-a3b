#!/usr/bin/env python3
"""three_arm.py — the three-arm agreement, host-side, with relL2 as the primary metric.

    <venv>/bin/python three_arm.py [--work work]

CPU only: no iGPU, no GPU lock, no timer, and no ms / tok/s / GB/s figure on any line this prints.

THE THREE ARMS
  host_twin        host_probe            — lm_head_kernels.h compiled by g++ -O2 (no ROCm at all)
  hip_hostpath     lm_head --host-path   — the SAME headers inside the device translation unit,
                                           compiled by hipcc for gfx1151 and executed on the CPU.
                                           This is the device compiler's codegen, not the iGPU; the
                                           arm says so in its own RUN line (hip_calls=0).
  oracle           expect.f32            — oracle_arm.py's numpy CPU reference: bf16 bits widened by
                                           <<16, Q8_0 through tools/gguf_reader's audited dequant, and
                                           math.fsum over the exact products.

WHAT THE METRICS ARE, AND WHICH ONE DECIDES
  relL2            ||a-b||2 / ||b||2                      PRIMARY (scale-free)
  relL2_bandnorm   sqrt(mean((|a-b|/band)^2))              PRIMARY against the oracle: the band this
                                                           lane derives (floor_probe.py) is the only
                                                           tolerance with a derivation behind it
  maxabs           max|a-b|                                REPORTED, NEVER LOAD-BEARING
  cos              <a,b>/(||a|| ||b||)                     PRINTED ALONGSIDE, never load-bearing
  ndiff            words whose BIT IMAGE differs          load-bearing for the compiled-pair claim
  bit_identical    ndiff == 0

Why maxabs is not allowed to decide anything, in this repo's own words: in
prototype/shared_expert/work/derive.log (run 2026-10-26, same box) the honest row map scored
maxabs 0.96875 while a WRONG map scored 0.961426 — the wrong candidate was NEARER on the absolute
metric, and only the relative one separated them (cos 0.785072923 vs 0.707297027). An absolute
metric that cannot see what you are claiming is not conservative, it is blind. This file therefore
prints maxabs for every group and takes its verdict from relL2 / relL2_bandnorm / ndiff, and it
counts the words a maxabs-thresholded judge would have waved through (maxabs_blind_words).

The device arm proper (the iGPU) is NOT here: it belongs to gate.py's --device run in the main
agent's exclusive, halogen-down, gpu_lock-held window.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate as G                                              # noqa: E402  the lane's own readers

MAX_ROWS, DEEP_ROWS = 4, 32
NUM_FIELDS = G.COUNTER_FIELDS + ("oxor", "osum")


class Refuse(Exception):
    pass


def metrics(a, b, bands=None):
    """a, b: equal-length lists of float. Returns the field's agreement metrics. `bands` (same length,
    >0 where defined) turns on the band-normalised primary used against the oracle."""
    n = len(a)
    if n == 0:
        raise Refuse("metrics asked over an empty group: a comparison of nothing always passes")
    sa = np.asarray(a, dtype=np.float64)
    sb = np.asarray(b, dtype=np.float64)
    dev = sa - sb
    ndiff = sum(1 for i in range(n)
                if struct.pack("<f", a[i]) != struct.pack("<f", b[i]))
    maxabs = float(np.max(np.abs(dev)))
    nb = float(np.linalg.norm(sb))
    relL2 = float(np.linalg.norm(dev) / nb) if nb > 0 else (0.0 if maxabs == 0 else float("inf"))
    na = float(np.linalg.norm(sa))
    cos = float(np.dot(sa, sb) / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)
    bn = float("nan")
    worst_word = float("nan")
    if bands is not None:
        r = [abs(dev[i]) / bands[i] for i in range(n) if bands[i] > 0.0]
        if not r:
            raise Refuse("no positive band in a %d-word group: relL2_bandnorm would be vacuous" % n)
        bn = math.sqrt(sum(x * x for x in r) / len(r))
        worst_word = max(r)
    gmax = maxabs if maxabs > 0 else 1.0
    blind = sum(1 for i in range(n)
                if abs(dev[i]) > 1e-3 * max(abs(sb[i]), 1e-300) and abs(dev[i]) < 0.01 * gmax)
    return dict(n=n, ndiff=ndiff, bit_identical=(ndiff == 0), maxabs=maxabs, relL2=relL2,
                cos=cos, relL2_bandnorm=bn, worst_word=worst_word, maxabs_blind_words=blind)


def fmt(m):
    return ("n=%d bit_identical=%d ndiff=%d maxabs=%.6g relL2=%.6g relL2_bandnorm=%.4g "
            "worst_word=%.4g cos=%.9g maxabs_blind_words=%d"
            % (m["n"], 1 if m["bit_identical"] else 0, m["ndiff"], m["maxabs"], m["relL2"],
               m["relL2_bandnorm"], m["worst_word"], m["cos"], m["maxabs_blind_words"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    a = ap.parse_args()
    work = a.work

    fc, en, fields, modes, fmts = G.contract()
    cfg = G.read_config()
    p = G.build(cfg, fc, en, MAX_ROWS, DEEP_ROWS)
    mine = G.write_plan(os.path.join(work, "threearm_plan.tsv"), p,
                        dict((c["idx"], "BAND" if c["mode"] == en["LMH_MODE_CORRECT"] else "FIRES")
                             for c in p.cases))
    disk = [l for l in open(os.path.join(work, "plan.tsv")).read().splitlines()
            if not l.startswith("#")]
    if len(mine) != len(disk) or any(x.split("\t")[:17] != y.split("\t")[:17]
                                      or x.split("\t")[18:] != y.split("\t")[18:]
                                      for x, y in zip(mine, disk)):
        raise Refuse("work/plan.tsv is not the population this file is reasoning about")

    # ---- run the two compiled arms over the SAME bytes -------------------------------------------
    rc, out_h = G.run_arm(G.PROBE, work, p, cfg, "ta_host_out.f32", "ta_host.tsv",
                          "three-arm-host.log")
    rc2, out_d = G.run_arm(G.DRIVER, work, p, cfg, "ta_hip_out.f32", "ta_hip.tsv",
                           "three-arm-hiphost.log", extra=["--host-path"])
    for l in (out_h + "\n" + out_d).splitlines():
        if l.startswith("RUN ") or l.startswith("RESULT ") or l.startswith("FAIL "):
            print("ARM " + l)
    if rc != 0 or rc2 != 0:
        raise Refuse("an arm failed: host_probe rc=%d, lm_head --host-path rc=%d" % (rc, rc2))
    if "arm=device-hostpath" not in out_d or "hip_calls=0" not in out_d:
        raise Refuse("the third arm did not announce itself as the device-compiled host path with "
                     "hip_calls=0; a mislabelled arm is worse than a missing one")
    host_rep = G.read_report(os.path.join(work, "ta_host.tsv"), p, fields, fc, fmts)
    hip_rep = G.read_report(os.path.join(work, "ta_hip.tsv"), p, fields, fc, fmts)
    hostw = G.words_of(os.path.join(work, "ta_host_out.f32"), p.out_floats)
    hipw = G.words_of(os.path.join(work, "ta_hip_out.f32"), p.out_floats)
    expw = G.words_of(os.path.join(work, "expect.f32"), p.out_floats)
    bandw = G.doubles_of(os.path.join(work, "band.f64"), p.out_floats)

    def f32s(ws, idxs):
        return [struct.unpack("<f", struct.pack("<I", ws[i]))[0] for i in idxs]

    # ---- the word groups: honest words, then sliced by the geometry that could plausibly differ ---
    honest = [c for c in p.cases if c["mode"] == en["LMH_MODE_CORRECT"]]
    if not honest:
        raise Refuse("the plan has no honest cases: the triangle has nothing to agree about")

    def words_of(c):
        slot = p.slot(c)
        return [c["out_off"] + b * slot + j for b in range(c["bcount"]) for j in range(c["n_out"])]

    groups = [("all_honest", honest)]
    for ti in sorted(set(c["table"] for c in honest)):
        groups.append(("table%d_enc%d" % (ti, p.tables[ti]["enc"]),
                       [c for c in honest if c["table"] == ti]))
    for an, av in (("assoc_seq", en["LMH_ASSOC_SEQ"]), ("assoc_rev", en["LMH_ASSOC_REV"]),
                   ("assoc_split8", en["LMH_ASSOC_SPLIT8"])):
        groups.append((an, [c for c in honest if c["assoc"] == av]))
    for pn, pv in (("path_dense", en["LMH_PATH_DENSE"]), ("path_gather", en["LMH_PATH_GATHER"])):
        groups.append((pn, [c for c in honest if c["path"] == pv]))
    for bn, bv in (("batch_batched", en["LMH_BATCH_BATCHED"]),
                   ("batch_perrow", en["LMH_BATCH_PERROW"])):
        groups.append((bn, [c for c in honest if c["batching"] == bv]))

    printed, findings = [], []
    for pair, (A, B, use_band) in (("host_twin_vs_hip_hostpath", (hostw, hipw, False)),
                                   ("host_twin_vs_oracle", (hostw, expw, True)),
                                   ("hip_hostpath_vs_oracle", (hipw, expw, True))):
        for gname, cs in groups:
            idxs = []
            for c in cs:
                idxs += words_of(c)
            if not idxs:
                print("THREEARM pair=%s group=%s EMPTY=1 note=not-a-claim-not-counted" % (pair, gname))
                continue
            m = metrics(f32s(A, idxs), f32s(B, idxs),
                        [bandw[i] for i in idxs] if use_band else None)
            line = "THREEARM pair=%s group=%s %s" % (pair, gname, fmt(m))
            printed.append(line)
            print(line)
            if not use_band and not m["bit_identical"]:
                findings.append("%s / %s: %d of %d words differ between the two compiled arms "
                                "(relL2=%.6g) — one source, two compilers, so this is a codegen "
                                "finding and NOT something to band away"
                                % (pair, gname, m["ndiff"], m["n"], m["relL2"]))
            if use_band and not (m["relL2_bandnorm"] <= 1.0 and m["worst_word"] <= 1.0):
                findings.append("%s / %s: relL2_bandnorm=%.4g worst_word=%.4g — outside the derived "
                                "band (primary metric; maxabs %.6g was not consulted)"
                                % (pair, gname, m["relL2_bandnorm"], m["worst_word"], m["maxabs"]))
        # the report's own counters, arm to arm
        if not use_band:
            for fld in NUM_FIELDS:
                va = [float(host_rep[c["idx"]][fld]) for c in p.cases]
                vb = [float(hip_rep[c["idx"]][fld]) for c in p.cases]
                m = metrics(va, vb)
                line = "THREEARMF pair=%s field=%s %s" % (pair, fld, fmt(m))
                printed.append(line)
                print(line)
                if not m["bit_identical"]:
                    bad = [c["idx"] for c in p.cases
                           if host_rep[c["idx"]][fld] != hip_rep[c["idx"]][fld]][:4]
                    findings.append("report field %s: arms differ on %d of %d cases, first cases %s"
                                    % (fld, m["ndiff"], m["n"], bad))

    # would a maxabs-thresholded judge have disagreed with the primary? printed, not assumed
    for pair in ("host_twin_vs_oracle", "hip_hostpath_vs_oracle"):
        rows = [l for l in printed if ("pair=%s " % pair) in l and l.startswith("THREEARM ")]
        if len(rows) < 2:
            continue
        def val(l, key):
            return float([t for t in l.split() if t.startswith(key + "=")][0].split("=")[1])
        worst_rel = max(rows, key=lambda l: val(l, "relL2_bandnorm"))
        worst_abs = max(rows, key=lambda l: val(l, "maxabs"))
        agree = worst_rel.split("group=")[1].split()[0] == worst_abs.split("group=")[1].split()[0]
        print("THREEARMRANK pair=%s relL2_worst_group=%s maxabs_worst_group=%s ranking_agrees=%d "
              "note=maxabs_is_reported_only"
              % (pair, worst_rel.split("group=")[1].split()[0], worst_abs.split("group=")[1].split()[0],
                 1 if agree else 0))

    n_words = sum(1 for l in printed if l.startswith("THREEARM "))
    n_fields = sum(1 for l in printed if l.startswith("THREEARMF "))
    print("THREEARMAGG records_printed=%d word_groups=%d field_rows=%d arms=3 pairs=%d "
          "honest_cases=%d words_compared=%d load_bearing=relL2_bandnorm+ndiff "
          "maxabs_load_bearing=0 cos_load_bearing=0"
          % (len(printed), n_words, n_fields, 3, len(honest),
             sum(int([t for t in l.split() if t.startswith("n=")][0].split("=")[1])
                 for l in printed if l.startswith("THREEARM "))))
    for b in findings:
        print("FINDING %s" % b)
    if findings:
        print("RESULT FAIL lm_head three-arm host agreement findings=%d" % len(findings))
        return 1
    print("RESULT PASS lm_head three-arm host agreement groups=%d field_rows=%d "
          "compiled_pair_bit_identical=1 oracle_inside_band=1" % (n_words, n_fields))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as e:
        print("REFUSE three-arm: %s" % e)
        sys.exit(1)
