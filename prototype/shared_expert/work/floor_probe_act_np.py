#!/usr/bin/env python3
"""floor_probe_act_np.py — the numpy half of the `act` floor probe, and the half that licenses a band.

Why: the field `act` of the f32_act case carries `se_swiglu(gate, up, f32_act=1)` = `silu(gate) * up`,
where `gate` is a raw f32 DOT PRODUCT. The band it was judged against came from
`work/floor_probe_exp*`, which enumerates the bf16 input grid — the domain of the *bf16* route, not of
this field. Judging one by the other was honest only as a refusal (`UNVERIFIED by probe domain`). This
probe closes the gap by measuring over the fixture's own f32 dot values.

The domain, and why it is exhaustive rather than sampled:
    every (gate_j, up_j) pair the fixture's f32 dot fields carry, read out of the oracle's expect dump
    (the arms' own gate/up words are asserted bit-identical to those before a single word is measured,
    so the population is what BOTH sides carry). The alternative — the whole f32 continuum inside the
    value hull — is ~1e9 patterns and no honest band can rest on striding through it.

The derivation, printed not typed (same shape the gdn lane's floor probe uses, copied as a method from
`speccheck/gdn_ref.py` §BAND_FACTORS on 2026-10-26, where `load_band()` re-derives `probe x factor` and
refuses a disagreeing file):
    leg_c  = max steps(route_the_kernel_runs,  anchor)   anchor = the same route in double, rounded
    leg_np = max steps(route_the_oracle_runs, anchor)     to f32 once (error ~2e-9 of one f32 ULP)
    band   = leg_c + leg_np                              the triangle inequality on the rank metric
The ruler's distance is |order_key(a) - order_key(b)|, an absolute difference of a monotone key, so
the triangle holds for any triple and is asserted per row here as an arithmetic self-check: a mis-parsed
table or a NaN in the output makes it fail and the probe refuses to license anything. A band is a
statement about the probe, so a probe that measured nothing exits non-zero instead of defaulting.

    work/floor_probe_act --pairs work/act_pairs.tsv --emit work/act_c_table.tsv   (done for you)
    <venv python> work/floor_probe_act_np.py --cases work/cases.tsv --expect work/inputs.bin \
                  --work work [--arm-out work/host_out.f32]
"""
import argparse
import os
import subprocess
import sys
from array import array

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LANE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, LANE)
import ulp_policy as UP                                        # noqa: E402  the one ruler (D-17)
import oracle_arm as OA                                        # noqa: E402  the numpy leg is the
                                                                #            oracle's OWN silu_f32
OFF = {"gate": 0, "up": 512, "act": 1024, "raw": 1536, "out": 3584, "prob": 5632}
LEN = {"gate": 512, "up": 512}


class Refuse(Exception):
    pass


def read_cases(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 16:
                raise Refuse("%s: row has %d columns, the oracle writes 16" % (path, len(f)))
            rows.append(dict(index=int(f[0]), label=f[1], mode=f[2], dst_off=int(f[7]),
                             exp_off=int(f[8]), widths=dict(gate=f[10], up=f[11], act=f[12])))
    if not rows:
        raise Refuse("%s holds no cases: a probe over an empty case list measures nothing" % path)
    return rows


def read_words(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) % 4:
        raise Refuse("%s holds %d bytes, not a whole number of words" % (path, len(buf)))
    a = array("I")
    a.frombytes(buf)
    return a


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--expect", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm-out", default=None,
                    help="an arm's output file; when given, the population is asserted bit-identical "
                         "in the gate/up fields before anything is measured")
    a = ap.parse_args(argv[1:])

    cases = read_cases(a.cases)
    expect = read_words(a.expect)
    arm = read_words(a.arm_out) if a.arm_out else None

    # ---- the population: every f32 (gate, up) pair the fixture carries -------------------------
    pairs, judged_rows, all_rows = [], [], []
    identity_checked = 0
    for c in cases:
        if c["widths"]["gate"] != "f32" or c["widths"]["up"] != "f32":
            continue                                  # a case that carries no f32 dot adds no pair
        for j in range(LEN["gate"]):
            eo, do = c["exp_off"], c["dst_off"]
            gb = expect[eo + OFF["gate"] + j]
            ub = expect[eo + OFF["up"] + j]
            if arm is not None:
                a_g, a_u = arm[do + OFF["gate"] + j], arm[do + OFF["up"] + j]
                if a_g != gb or a_u != ub:
                    raise Refuse("case %d j=%d: the arm carries gate=%08X up=%08X where the oracle "
                                 "carries %08X/%08X; the probe population would not be the population "
                                 "the field sees" % (c["index"], j, a_g, a_u, gb, ub))
                identity_checked += 1
            pairs.append((int(gb), int(ub)))
            all_rows.append(len(pairs) - 1)
            if c["widths"]["act"] == "f32":
                judged_rows.append(len(pairs) - 1)
    if not pairs:
        raise Refuse("no f32 dot pair found in %s: nothing to probe, so nothing may be licensed"
                     % a.cases)
    if not judged_rows:
        raise Refuse("no case in %s declares an f32 `act` field: the population contains no instance "
                     "of the field this band is for" % a.cases)
    pairs_path = os.path.join(a.work, "act_pairs.tsv")
    with open(pairs_path, "w") as fh:
        fh.write("# gate_bits up_bits — one line per f32 dot pair the fixture carries\n")
        for gb, ub in pairs:
            fh.write("%08X %08X\n" % (gb, ub))

    # ---- the C leg and the anchor ---------------------------------------------------------------
    exe = os.path.join(HERE, "floor_probe_act")
    if not os.path.exists(exe):
        raise Refuse("%s is missing; build it (build.sh does): g++ -O2 -std=c++11 -ffp-contract=off "
                     "-I%s work/floor_probe_act.c -o work/floor_probe_act -lm"
                     % (exe, LANE))
    table = os.path.join(a.work, "act_c_table.tsv")
    p = subprocess.run([exe, "--pairs", pairs_path, "--emit", table],
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        raise Refuse("floor_probe_act rc=%d: %s" % (p.returncode, p.stderr.strip().splitlines()[:1]))
    c_rows = []
    with open(table) as fh:
        head = fh.readline()
        if not head.startswith("idx"):
            raise Refuse("%s has no header line" % table)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != 5:
                raise Refuse("%s: malformed row %r" % (table, line))
            c_rows.append((int(f[0]), int(f[1], 16), int(f[2], 16), int(f[3], 16), int(f[4], 16)))
    if len(c_rows) != len(pairs):
        raise Refuse("parsed %d rows from %s, the population holds %d pairs (parsed != printed)"
                     % (len(c_rows), table, len(pairs)))
    for i, (idx, gb, ub, _ac, _an) in enumerate(c_rows):
        if idx != i or (gb, ub) != pairs[i]:
            raise Refuse("%s row %d is (%d,%08X,%08X), not the pair that was asked for: the two sides "
                         "would be measuring different inputs" % (table, i, idx, gb, ub))

    # ---- the numpy leg, in the oracle's own function --------------------------------------------
    gate = (np.array([r[1] for r in c_rows], dtype=np.uint32)).view(np.float32)
    up = (np.array([r[2] for r in c_rows], dtype=np.uint32)).view(np.float32)
    c_act = np.array([r[3] for r in c_rows], dtype=np.uint32)
    anchor = np.array([r[4] for r in c_rows], dtype=np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        np_act = (OA.silu_f32(gate) * up).astype(np.float32).view(np.uint32)
    n_nan = 0
    for arr in (c_act, np_act, anchor):
        v = np.asarray(arr, dtype=np.uint32)
        exp_all = (((v >> np.uint32(23)) & np.uint32(0xFF)) == np.uint32(0xFF))
        mant_nonzero = ((v & np.uint32(0x007FFFFF)) != np.uint32(0))
        n_nan += int(np.count_nonzero(exp_all & mant_nonzero))
    if n_nan:
        raise Refuse("%d of the measured words are NaN: a distance to NaN is not a rounding step and "
                     "no band may be folded from it" % n_nan)

    r = UP.Ruler("f32")

    def cen(rows, a_arr, b_arr):
        return r.census([(i, int(a_arr[i]), int(b_arr[i])) for i in rows])

    leg_c_all = cen(all_rows, c_act, anchor)
    leg_np_all = cen(all_rows, np_act, anchor)
    direct_all = cen(all_rows, c_act, np_act)
    leg_c_judged = cen(judged_rows, c_act, anchor)
    leg_np_judged = cen(judged_rows, np_act, anchor)
    direct_judged = cen(judged_rows, c_act, np_act)

    # the triangle, asserted per row on every population (a self-check on the folding, not a hope)
    tri_bad = 0
    for i in all_rows:
        s_cn = r.classify(int(c_rows[i][3]), int(np_act[i]))[1]
        s_ca = r.classify(int(c_rows[i][3]), int(anchor[i]))[1]
        s_an = r.classify(int(anchor[i]), int(np_act[i]))[1]
        if s_cn is None or s_ca is None or s_an is None or s_cn > s_ca + s_an:
            tri_bad += 1
    for name, c in (("c-vs-anchor", leg_c_all), ("numpy-vs-anchor", leg_np_all),
                    ("c-vs-numpy", direct_all)):
        print(c.line(name, extra="probe=act-over-the-fixture-f32-dot-domain")
              .replace("CENSUS", "CENSUSACT"))
    for name, c in (("c-vs-anchor", leg_c_judged), ("numpy-vs-anchor", leg_np_judged),
                    ("c-vs-numpy", direct_judged)):
        print(c.line(name + "[judged_subset]", extra="probe=act-over-the-f32_act-field-pairs")
              .replace("CENSUS", "CENSUSACT"))
    band_all = leg_c_all.max_steps + leg_np_all.max_steps
    band_judged = leg_c_judged.max_steps + leg_np_judged.max_steps
    print("FLOOREXP_ACT_POP cases_probed=%d pairs=%d judged_pairs=%d arm_words_identical_to_oracle=%d "
          "distinct_gate=%d distinct_up=%d"
          % (len(cases), len(pairs), len(judged_rows), identity_checked,
             len({r_[1] for r_ in c_rows}), len({r_[2] for r_ in c_rows})))
    print("FLOOREXP_ACT_DERIVE derivation=triangle-inequality-on-floor-probe "
          "leg_c_all=%d leg_np_all=%d band_all=%d leg_c_field=%d leg_np_field=%d band_field=%d "
          "licensed=band_field"
          % (leg_c_all.max_steps, leg_np_all.max_steps, band_all,
             leg_c_judged.max_steps, leg_np_judged.max_steps, band_judged))
    print("FLOOREXP_ACT_TRI rows=%d per_row_violations=%d direct_max=%d band_all=%d direct_field=%d "
          "band_field=%d" % (len(all_rows), tri_bad, direct_all.max_steps, band_all,
                             direct_judged.max_steps, band_judged))
    if tri_bad:
        raise Refuse("%d rows break the triangle the band is derived from: the fold or the table is "
                     "wrong, refuse to license anything" % tri_bad)
    if band_judged > band_all:
        raise Refuse("the field's own legs (%d+%d=%d) exceed the whole-fixture legs (%d+%d=%d): the "
                     "judged subset is not a subset of the population, so the table was mis-parsed"
                     % (leg_c_judged.max_steps, leg_np_judged.max_steps, band_judged,
                        leg_c_all.max_steps, leg_np_all.max_steps, band_all))
    if direct_all.max_steps > band_all or direct_judged.max_steps > band_judged:
        raise Refuse("the measured c-vs-numpy divergence (%d all / %d field) exceeds the sum of the "
                     "legs (%d / %d): the anchor cannot be inside both legs at once, so this probe is "
                     "inconsistent" % (direct_all.max_steps, direct_judged.max_steps, band_all,
                                       band_judged))
    print("FLOOREXP_ACT max_ulp_f32=%d legs=%d+%d domain=every f32 (gate,up) dot pair the fixture "
          "carries (%d pairs, %d of them the f32_act field itself); the band on an f32-carrying act "
          "field is this number, derived here and not chosen"
          % (band_judged, leg_c_judged.max_steps, leg_np_judged.max_steps, len(pairs),
             len(judged_rows)))
    print("FLOOREXP_ACT verdict=" + (
        "domain-covered: the probe population is every value the judged field carries, on both sides"
        if identity_checked and band_judged >= direct_judged.max_steps else
        "NOT covered: band_field=%d direct_field=%d arm_identity=%d" % (band_judged,
                                                                        direct_judged.max_steps,
                                                                        identity_checked)))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Refuse as e:
        print("FLOOREXP_ACT verdict=REFUSED reason=%s" % e)
        print("no timing taken")
        sys.exit(1)
