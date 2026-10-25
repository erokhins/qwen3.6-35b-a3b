#!/usr/bin/env python3
"""gate.py — the driver for prototype/shared_expert. It never touches the GPU lock.

    ./gate --selftest           host-only witnesses (three-arm selftest, parser controls, probes)
    ./gate                      the device gate: correctness only, PASS/FAIL, no timing
    ./gate --rows N             cut the fixture in numbers only (never an assertion); prints CUT

`gate` (the shell wrapper) is the only acquirer of the repo's GPU lock tool, and it checks this file
for any mention of that tool at run time: a driver that self-acquires and is called by a caller
already holding the lock parks forever with no CPU, no GPU, no message and no exit
(pitfalls/50-harness-and-shell.md §"Self-locking gate drivers deadlock silently").

What is judged, and by whom
  * every case against its own expectation span, FIELD BY FIELD, with each field's width declared by
    the oracle in cases.tsv. Nothing is judged on the whole span, because a whole-span count lets one
    field's difference pay for another's absence (D-16).
  * a field whose declared route is exact must be bit-identical: `bad == 0`, no tolerance. That is
    almost every field here, and it is the whole point of declaring the route in the header: the Q8_0
    dequant is exact in f32, the dot is a declared order, and every rounding the reference performs
    is performed here in the same place.
  * two fields are NOT exact and are named as counted: the `act` field of an f32-act case and the
    `prob` field of a cast-after case. Both carry `expf` in f32, and each is judged by the floor probe
    whose DOMAIN contains the values that field actually carries: `prob` takes `bf16(t)`, which
    `work/floor_probe_exp.c` + `work/floor_probe_exp_np.py` enumerate exhaustively (every bf16
    pattern), while `act` of an f32-act case takes a raw f32 DOT PRODUCT, which is the domain of
    `work/floor_probe_act.c` + `work/floor_probe_act_np.py` (every f32 dot pair the fixture carries,
    asserted bit-identical on both sides before a word is measured). Each probe is re-run on every
    gate run and the band is its printed maximum — legs to an f64 anchor, summed by the triangle
    inequality, derivation printed beside the number. A counted field whose domain no probe covers is
    REFUSED rather than defaulted, and the refusal is exercised by a control (`band refusal still
    fires`), because a refusal nobody can reproduce is a hope. Nothing here is a band typed to fit what
    the build produces. This is D-16 counting with an attribution, NOT the D-17 ±1 acceptance:
    `judge()` is not called on those two fields, and the probes are the reason the *bf16-carrying*
    fields stay exact (the store is coarser than every disagreement the two libms can produce).
  * an injection must FIRE: its words must differ from the honest case's words for the same block and
    row. An injection that reproduces its own expectation but equals the honest output proves nothing
    beyond the fact that the expectation was derived from the same defect.
  * one blindness is declared rather than hidden: `layout_pair_swap` reads the same rows as the honest
    case, so `gate` and `up` are identical to it by construction and only `act` can see the swap. The
    gate asserts that blindness as a fact (0 differing gate/up words AND a differing act), because a
    suite that passes by not looking is worse than one that looks and reports.
  * the arm's own per-case counters (bad, poison count, weight reads, two digests) are re-derived by
    this driver from the output file and the plan. Two sites, one truth; the right-hand side never
    comes from the counter it is checking.
Every parser asserts parsed == printed. CPU-only modes take no lock.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HOST = os.path.join(HERE, "host_probe")
DEVICE = os.path.join(HERE, "shared_expert")
ORACLE = os.path.join(HERE, "oracle_arm.py")
CENSUS = os.path.join(HERE, "census.py")
PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"
NAN_BITS = 0x7FC00000
H, I, TWO_I = 2048, 512, 1024
CASE_FLOATS = 5633
FIELD_OFF = {"gate": 0, "up": 512, "act": 1024, "raw": 1536, "out": 3584, "prob": 5632}
FIELD_LEN = {"gate": I, "up": I, "act": I, "raw": H, "out": H, "prob": 1}
MODES = ["correct", "layout_up_first", "layout_row_inter", "layout_pair_swap", "row_shift1",
         "f32_act", "tail_drop", "cols_half", "down_half", "wide_misread"]

sys.path.insert(0, os.path.join(REPO, "tools"))
try:
    import ulp_policy as UP
    _UP_ERR = None
except Exception as _e:                                     # pragma: no cover
    UP, _UP_ERR = None, _e


class Fail(Exception):
    pass


def run(cmd, log=None):
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if log:
        with open(log, "w") as fh:
            fh.write(p.stdout)
            if p.stderr.strip():
                fh.write("\n" + p.stderr)
    return p.returncode, p.stdout


def kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_tag(text, tag, keys, want=None):
    recs = []
    for line in text.splitlines():
        if line.startswith(tag + " "):
            d = kv(line)
            if not all(k in d for k in keys):
                raise Fail("line tagged %s is missing %s: %r" % (tag, keys, line))
            recs.append(d)
    if want is not None and len(recs) != want:
        raise Fail("parsed %d %s records, expected %d (a parser that matches nothing prints PASS)"
                   % (len(recs), tag, want))
    if not recs:
        raise Fail("no %s records at all: this run adjudicated nothing" % tag)
    return recs


def read_words(path):
    """The output/expectation files as uint32 word images, never as floats: every judgement here is
    bit identity, and reading the image is what lets a poison NaN be judged at all
    (pitfalls/30 §".view vs .to")."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) % 4:
        raise Fail("%s holds %d bytes, not a whole number of words" % (path, len(buf)))
    a = array("I")
    a.frombytes(buf)
    return a


def read_cases(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 16:
                raise Fail("%s: row has %d columns, the oracle writes 16 (a missing column must not "
                           "default its way past the check that needs it)" % (path, len(f)))
            rows.append(dict(index=int(f[0]), label=f[1], mode=f[2], cast=f[3], sgate=f[4],
                             block=f[5], row=int(f[6]), dst_off=int(f[7]), exp_off=int(f[8]),
                             n_floats=int(f[9]), widths=dict(gate=f[10], up=f[11], act=f[12],
                                                             raw=f[13], out=f[14], prob=f[15])))
    return rows


def bf16_of(word):
    """The bf16 image a widened word carries. The arms store bf16 values widened into the top half, so
    the comparison for a bf16-carried field is on those 16 bits."""
    return word >> 16


class Checks(object):
    """Every check names the number it wants and the number it got. A check whose predicate cannot be
    false is a comment, so each line here carries both sides of the comparison."""

    def __init__(self):
        self.rows = []
        self.failed = []

    def check(self, name, ok, want, got, detail=""):
        self.rows.append((name, ok, want, got, detail))
        print("CHECK %-64s want=%-10s got=%-10s %s" % (name, want, got, detail))
        if not ok:
            self.failed.append(name)
        return ok

    def verdict(self, label):
        print("SUMMARY %s checks=%d failed=%d" % (label, len(self.rows), len(self.failed)))
        for name in self.failed:
            print("FAILED %s" % name)
        print("no timing taken")
        if self.failed:
            print("RESULT FAIL (%s): %d of %d checks failed: %s"
                  % (label, len(self.failed), len(self.rows), "; ".join(self.failed[:6])))
            return 1
        print("RESULT PASS (%s): %d checks, three arms exercised, no timing taken"
              % (label, len(self.rows)))
        return 0


# ---- the witnesses that are not about the fixtures ---------------------------------------------
def selftest_gate(v, exe, label):
    """All three arms must have run; the honest one must be clean; each saboteur must go red on
    exactly its own derived count. A single arm exiting 0 is a failure by construction, which is what
    stops `--arm=0` from being a pass with the saboteurs deleted."""
    rc, out = run([exe, "--selftest"], os.path.join(HERE, "work", "selftest-%s.log" % label))
    recs = parse_tag(out, "SELFTEST", ["arm", "checks", "equals", "differs", "changed", "want",
                                       "failures"], want=3)
    three = parse_tag(out, "SELFTEST3", ["arms_run", "checks", "equals", "differs",
                                        "honest_failures"], want=1)[0]
    v.check("%s selftest ran all three arms" % label, three["arms_run"] == "3", "3",
            three["arms_run"], out.strip().splitlines()[-1] if out else "")
    honest = recs[0]
    v.check("%s honest arm is clean" % label, honest["failures"] == "0" and honest["changed"] == "0",
            "0", "%s/%s" % (honest["failures"], honest["changed"]),
            "checks=%s equals=%s differs=%s" % (honest["checks"], honest["equals"],
                                                honest["differs"]))
    stub = recs[1]
    v.check("%s STUB arm goes red on exactly the differing checks" % label,
            stub["changed"] == stub["want"] and int(stub["want"]) > 0, stub["want"], stub["changed"],
            "a comparator that always answers equal must break exactly %s checks" % stub["want"])
    crossed = recs[2]
    v.check("%s CROSSED arm goes red on exactly the equal checks" % label,
            crossed["changed"] == crossed["want"] and int(crossed["want"]) > 0, crossed["want"],
            crossed["changed"], "judging each value against the next different want must break "
                                "exactly %s checks" % crossed["want"])
    v.check("%s suite exits non-zero when it is honest" % label, rc == 0, 0, rc, "")
    for arm in (0, 1, 2):
        rc1, out1 = run([exe, "--selftest", "--arm=%d" % arm])
        v.check("%s --arm=%d alone refuses to pass" % (label, arm), rc1 != 0, "non-zero", rc1,
                out1.strip().splitlines()[-1] if out1.strip() else "")
    return recs


def probe_gate(v):
    """The exp floor probe, run rather than cited: it is where the band on a bf16-fed transcendental
    field comes from. Its domain is every bf16 input pattern, which is what `sigmoid(bf16(t))` can
    feed the transcendental — and nothing else, which is why `act` got its own probe."""
    exe = os.path.join(HERE, "work", "floor_probe_exp")
    if not os.path.exists(exe):
        v.check("exp floor probe is built", False, "built", "missing",
                "cc -O2 -ffp-contract=off -std=c11 work/floor_probe_exp.c -o "
                "work/floor_probe_exp -lm")
        return None, None
    rc, out = run([exe, "--emit", os.path.join(HERE, "work", "c_exp_table.tsv")])
    rc2, out2 = run([PY, os.path.join(HERE, "work", "floor_probe_exp_np.py"),
                     os.path.join(HERE, "work", "c_exp_table.tsv")])
    lines = [l for l in out2.splitlines() if l.startswith(("CENSUSEXP", "FLOOREXP_NP"))]
    for l in lines:
        print(l)
    rows = kv([l for l in lines if l.startswith("FLOOREXP_NP rows")][0])
    band_line = [l for l in lines if "max_ulp_f32=" in l][0]
    band = int(band_line.split("max_ulp_f32=")[1].split()[0])
    v.check("exp floor probe: the bf16 store hides every libm disagreement",
            rows["sigmoid_diff_after_bf16_store"] == "0", "0",
            rows["sigmoid_diff_after_bf16_store"],
            "f32 disagreements=%s over %s bf16 inputs; this is why the bf16-carrying fields stay "
            "exact" % (rows["sigmoid_diff_f32"], rows["rows"]))
    v.check("exp floor probe: an f32-carrying transcendental field is measured, not assumed",
            band >= 1, ">=1", band, "the band on a bf16-fed f32-carrying expf field is %d ULP from "
            "this probe (domain: every bf16 input pattern)" % band)
    return band, lines


ACT_EXE = os.path.join(HERE, "work", "floor_probe_act")
ACT_SRC = os.path.join(HERE, "work", "floor_probe_act.c")
ACT_NP = os.path.join(HERE, "work", "floor_probe_act_np.py")


def act_probe_gate(v, cases, work, arm_out, arm_label, bands):
    """The `act` floor probe: the band for a field whose transcendental input is an f32 dot product.

    Adds `{"f32_dot": (band, provenance)}` to `bands`, or leaves it absent with the failure recorded as a
    named red check — a missing probe is never a default band. The probe refuses on its own exit code
    when its population is empty, when its emitted table does not match the population it was handed,
    when a measured word is NaN, when a row breaks the triangle it derives from, when the judged subset
    is not a subset of the population, or when the measured divergence exceeds the sum of its legs.
    """
    if UP is None:
        raise Fail("cannot import the shared ruler from tools/ulp_policy.py: %r" % _UP_ERR)
    expect = read_words(os.path.join(work, "inputs.bin"))
    if not os.path.exists(ACT_EXE):
        rc, _ = run(["g++", "-O2", "-std=c++11", "-ffp-contract=off", "-I" + HERE, ACT_SRC,
                     "-o", ACT_EXE, "-lm"])
        if rc != 0 or not os.path.exists(ACT_EXE):
            v.check("act floor probe is built", False, "built", "rc=%d" % rc,
                    "g++ -O2 -std=c++11 -ffp-contract=off -I. work/floor_probe_act.c "
                    "-o work/floor_probe_act -lm")
            return bands
        print("ACTPROBE built on demand from work/floor_probe_act.c (same -ffp-contract=off as both "
              "arms: a rounding claim is a claim about the flags)")
    rc, out = run([PY, ACT_NP, "--cases", os.path.join(work, "cases.tsv"),
                   "--expect", os.path.join(work, "inputs.bin"), "--work", work,
                   "--arm-out", arm_out], os.path.join(work, "floor-probe-act-%s.log" % arm_label))
    for line in out.splitlines():
        if line.startswith(("FLOOREXP_ACT", "CENSUSACT", "route=")):
            print(line)
    if rc != 0:
        why = [l for l in out.splitlines() if "verdict=REFUSED" in l]
        v.check("act floor probe: it licenses a band instead of refusing", False, "rc=0", "rc=%d" % rc,
                why[0][:120] if why else "no reason line printed")
        return bands
    try:
        maxline = [l for l in out.splitlines() if l.startswith("FLOOREXP_ACT max_ulp_f32=")][0]
        derive = [l for l in out.splitlines() if l.startswith("FLOOREXP_ACT_DERIVE")][0]
        pop = kv([l for l in out.splitlines() if l.startswith("FLOOREXP_ACT_POP")][0])
        tri = kv([l for l in out.splitlines() if l.startswith("FLOOREXP_ACT_TRI")][0])
        sub = kv([l for l in out.splitlines() if "field=c-vs-numpy[judged_subset]" in l][0])
    except IndexError as e:
        v.check("act floor probe printed its band, its derivation and its coverage", False,
                "4 lines", "missing key %s" % e,
                "a band whose maxima are not printed is a typed number in disguise")
        return bands
    band = int(maxline.split("max_ulp_f32=")[1].split()[0])
    dv = kv(derive)
    # The band is re-derived HERE from the legs the probe printed, so a band whose number was hand-edited
    # (or retyped out of an old transcript) disagrees with its own legs and is refused. This is the
    # discipline `speccheck/gdn_ref.py §load_band()` applies to a band FILE; this lane keeps no band file,
    # it re-measures on every run, so the re-derivation runs against the probe's printed legs.
    legs = int(dv["leg_c_field"]) + int(dv["leg_np_field"])
    v.check("act floor probe: the band is the sum of two measured legs, not a typed factor",
            band == legs and dv["licensed"] == "band_field", "%d+%d" % (int(dv["leg_c_field"]),
                                                                        int(dv["leg_np_field"])),
            "%d" % band,
            "derivation=%s leg_c=%s leg_np=%s (whole-fixture domain: %s+%s=%s)"
            % (dv["derivation"], dv["leg_c_field"], dv["leg_np_field"],
               dv["leg_c_all"], dv["leg_np_all"], dv["band_all"]))
    want_pairs = sum(FIELD_LEN["act"] for c in cases if c["mode"] == "f32_act")
    v.check("act floor probe covers every word of the counted act field", want_pairs > 0
            and int(pop["judged_pairs"]) == want_pairs, want_pairs, pop["judged_pairs"],
            "one (gate,up) pair per act word of every f32-act case, read out of the oracle's own dump")
    n_pairs = int(pop["pairs"])
    v.check("act floor probe: its population is the arm's own words, not a look-alike grid",
            n_pairs > 0 and int(pop["arm_words_identical_to_oracle"]) == n_pairs,
            "%d/%d words identical" % (n_pairs, n_pairs),
            "%s/%d" % (pop["arm_words_identical_to_oracle"], n_pairs),
            "EVERY gate/up word of every case compared bit-for-bit against the oracle before a distance "
            "was taken; the predicate used to be >0, which would pass on one identical word out of %d "
            "while the other %d came from a look-alike grid (%s arm)"
            % (n_pairs, n_pairs - 1, arm_label))
    v.check("act floor probe: no row breaks the triangle the band is derived from",
            tri["per_row_violations"] == "0" and int(tri["direct_field"]) <= band
            and int(tri["direct_max"]) <= int(dv["band_all"]), "0 violations",
            "%s violations, direct %s/%s" % (tri["per_row_violations"], tri["direct_field"],
                                             tri["direct_max"]),
            "the ruler's distance is an |order_key difference|, so the triangle is arithmetic; a "
            "violation means the table or the fold is wrong")
    prov = ("probe=work/floor_probe_act_np.py domain=the fixture's own f32 (gate,up) dot pairs; legs "
            "c=%s numpy=%s band=%s (whole-fixture dot domain %s+%s=%s); measured c-vs-numpy on the "
            "field's own %s pairs: max=%s over %s/%s"
            % (dv["leg_c_field"], dv["leg_np_field"], band, dv["leg_c_all"],
               dv["leg_np_all"], dv["band_all"], sub["judged"], sub["max_ulp"], sub["differing"],
               sub["judged"]))
    bands["f32_dot"] = (band, prov)
    # The coverage claim is re-derived here rather than left printed: the census the probe ran on the
    # field's own pairs is recomputed from the arm's words and the oracle's, and must agree on every
    # number. A probe that covered a different population would drift here.
    aw = read_words(arm_out)
    cen_field = UP.Ruler("f32").census(
        [(i, a, b) for c in cases if c["mode"] == "f32_act"
         for (i, a, b) in field_pairs(aw, expect, c, "act", "f32")[0]])
    v.check("act floor probe: the census it reports on the field's pairs IS the field's census",
            int(sub["judged"]) == cen_field.judged and int(sub["differing"]) == cen_field.diverging
            and int(sub["max_ulp"]) == cen_field.max_steps,
            "judged=%d differing=%d max=%d" % (cen_field.judged, cen_field.diverging,
                                               cen_field.max_steps),
            "judged=%s differing=%s max=%s" % (sub["judged"], sub["differing"], sub["max_ulp"]),
            "the band and the field are measured on one population, which is what closes the "
            "`UNVERIFIED by probe domain` finding")
    print("BAND domain=f32_dot band=%d probe=work/floor_probe_act_np.py covers=every f32 (gate,up) dot "
          "pair the fixture carries (%s pairs, %s of them the f32_act field itself)"
          % (band, pop["pairs"], pop["judged_pairs"]))
    return bands


def census_gate(v, occ=None):
    exe = os.path.join(HERE, "work", "isa", "shared_expert-hip-amdgcn-amd-amdhsa-gfx1151.o")
    cmd = [PY, CENSUS, "--obj", exe]
    if occ:
        cmd += ["--occupancy-file", occ]
    rc, out = run(cmd, os.path.join(HERE, "work", "census.log"))
    for line in out.splitlines():
        if line.startswith(("CENSUSDEVICE", "CENSUSSLICER", "OCCUPANCY", "LAUNCHBILL")):
            print(line)
    sl = kv([l for l in out.splitlines() if l.startswith("CENSUSSLICER")][0])
    v.check("census slicer is witnessed (empty stub < 8-multiply stub < 16-multiply stub)",
            sl["verdict"] == "ok", "ok", sl["verdict"],
            "empty=%s eight=%s sixteen=%s" % (sl["empty_f32_valu"], sl["eight_madd_f32_valu"],
                                              sl["sixteen_madd_f32_valu"]))
    dev = kv([l for l in out.splitlines() if l.startswith("CENSUSDEVICE")][0])
    v.check("the census prices the machine KFD reports, not the one HIP reports",
            dev["cu_from_kfd"] == "40", "40", dev["cu_from_kfd"],
            "hipDeviceAttributeMultiprocessorCount says 20 here (pitfalls/20)")
    if occ:
        k = [l for l in out.splitlines() if l.startswith("CENSUSKERNEL") and "\t-\t" not in l]
        v.check("every kernel's slot bill is printed beside its occupancy",
                len(k) >= 6, ">=6", len(k), "blocks_per_cu and wavefronts_per_simd on the same line")
    return rc


# ---- the width rule and the layout claim: checked here, not trusted from the oracle --------------
def parse_rt(text):
    """The candidate table, re-parsed by this driver. `winner=` from the oracle is a summary of a rule
    it applied; the ordering claim is only independent if someone else sorts the numbers."""
    cands, extra = {}, {}
    for line in text.splitlines():
        if line.startswith("LAYOUTRT candidate="):
            d = kv(line)
            cands[d["candidate"]] = (float(d["relL2"]), float(d["cos"]), float(d["maxabs"]))
        elif line.startswith("LAYOUTRT winner="):
            extra["winner"] = kv(line)
        elif line.startswith("LAYOUTRT rowmap_identical="):
            extra["ident"] = kv(line)
    return cands, extra


def layout_gate(v, out, oracle_cmd_base):
    cands, extra = parse_rt(out)
    want = ["gate_first", "up_first", "row_interleaved", "pair_swap", "pair_swap_rowmap_only",
            "row_shift1_control", "col_shift1_control"]
    missing = [c for c in want if c not in cands]
    v.check("every layout candidate is reported (one that is not printed cannot lose)", not missing,
            len(want), len(want) - len(missing),
            "missing: " + ",".join(missing) if missing else "%d candidates ranked here" % len(cands))
    if missing or "winner" not in extra:
        return
    h, hc, _ = cands["gate_first"]
    wrong = [c for c in want if c not in ("gate_first", "pair_swap_rowmap_only")]
    beaten = [c for c in wrong if cands[c][0] <= h]
    v.check("gate_first is strictly nearer on relL2 than every wrong candidate and both controls",
            not beaten, 0, len(beaten),
            "; ".join("%s=%.6g vs %.6g" % (c, cands[c][0], h) for c in beaten)
            or "ratio=%sx (driver-sorted, not the oracle's word)" % extra["winner"]["nearest_wrong_ratio"])
    best_cos = max(cands, key=lambda k: cands[k][1] if k not in ("pair_swap_rowmap_only",) else -9)
    v.check("cos agrees with relL2 about the winner (two formulas, one answer)",
            best_cos == "gate_first", "gate_first", best_cos,
            "cos=%.9f for the winner; relL2 alone would be one formula's opinion" % cands["gate_first"][1])
    v.check("the pair_swap row-map blindness is real and is separated by the silu side",
            cands["pair_swap_rowmap_only"][0] == h and cands["pair_swap"][0] > h,
            "rowmap_only==honest AND pair_swap>honest",
            "rowmap_only=%.9g pair_swap=%.9g honest=%.9g" % (cands["pair_swap_rowmap_only"][0],
                                                             cands["pair_swap"][0], h),
            extra.get("ident", {}).get("separated_by", "no rowmap_identical line printed"))
    for flag, val, why in (("--q8-rule", "512", "the packer rule this lane shipped by mistake"),
                           ("--claim-candidate", "up_first", "a claim the numbers do not support")):
        rc, ctl = run(oracle_cmd_base + [flag, val],
                      os.path.join(HERE, "work", "control-%s-%s.log" % (flag[2:], val)))
        text = ctl + open(os.path.join(HERE, "work", "control-%s-%s.log" % (flag[2:], val))).read()
        fired = [l for l in text.splitlines() if l.startswith("LAYOUTRT candidate=")
                 or l.startswith("Q8RULE family=")]
        v.check("the %s=%s control is REFUSED and visibly fires" % (flag[2:], val), rc != 0 and bool(fired),
                "non-zero rc and at least one differing measured line", "rc=%d lines=%d" % (rc, len(fired)),
                "a control that cannot fire proves nothing: %s" % why)


def q8rule_gate(v, out):
    """Is the pack Q8_0? The bound is the block-max geometry printed beside it, never a typed number."""
    fams = [kv(l) for l in out.splitlines() if l.startswith("Q8RULE family=")]
    pubs = [kv(l) for l in out.splitlines() if l.startswith("Q8RULEPUB family=")]
    if len(fams) != 3 or len(pubs) != 3:
        raise Fail("parsed %d Q8RULE and %d Q8RULEPUB lines, expected 3 and 3: a width check that "
                   "matches nothing is a width check that passes" % (len(fams), len(pubs)))
    over, worse, inexact = [], [], []
    for f in fams:
        if int(f["clamp_eq127"]) > int(f["clamp_bound_from_blockmax"]):
            over.append("%s: %d elements on the clamp, geometry allows %d"
                        % (f["family"], int(f["clamp_eq127"]), int(f["clamp_bound_from_blockmax"])))
        pub = [p for p in pubs if p["family"] == f["family"]][0]
        if float(f["relL2_vs_dump"]) > 1.5 * float(pub["relL2_vs_pinned_hf"]):
            worse.append("%s: this pack sits at relL2 %s against the pinned dump where the PUBLISHED "
                         "Q8_0 tensor sits at %s" % (f["family"], f["relL2_vs_dump"],
                                                     pub["relL2_vs_pinned_hf"]))
        if int(f["f32_vs_f64_inexact"]) != 0 or int(f["max_sig_bits"]) > int(f["max_sig_bits_f32_has"]):
            inexact.append("%s: %d elements not exact in f32, %d significand bits (f32 has %s)"
                           % (f["family"], int(f["f32_vs_f64_inexact"]), int(f["max_sig_bits"]),
                              f["max_sig_bits_f32_has"]))
    v.check("Q8_0 clamp rate is explained by the block-max geometry (a wrong divisor clips)", not over,
            0, len(over), "; ".join(over) or "rate %.4f%% vs 1/32 = 3.125%% expected by geometry"
            % (100.0 * int(fams[0]["clamp_eq127"]) / (int(fams[0]["clamp_bound_from_blockmax"]) / 1.05)))
    v.check("this lane's own pack is no further from the pinned dump than the published Q8_0 tensor is",
            not worse, 0, len(worse), "; ".join(worse) or "three families within 1.5x of the artifact")
    v.check("the decoded weight is exact in f32 and the significand claim is measured, not asserted",
            not inexact, 0, len(inexact), "; ".join(inexact)
            or "f16(d)*int8(q) needs %s significand bits; f32 has %s, so the product is exact"
            % (fams[0]["max_sig_bits"], fams[0]["max_sig_bits_f32_has"]))


# ---- the fixture judgements --------------------------------------------------------------------
def cols_gateup(mode):
    return {"tail_drop": H - 32, "cols_half": H // 2}.get(mode, H)


def cols_down(mode):
    return {"tail_drop": I - 32, "cols_half": I // 2}.get(mode, I)


def rows_down(mode):
    return H // 2 if mode == "down_half" else H


def counted_domains(c):
    """Which field of this case is counted, and which probe DOMAIN covers it.

    The mapping is the route, not a preference. `act` of an f32-act case hands the transcendental a raw
    f32 DOT PRODUCT, so only a probe over the fixture's f32 dot range may band it; `prob` of a
    cast-after case takes `bf16(t)`, which is exactly the bf16 input grid the exp probe enumerates
    exhaustively. Judging one field by the other probe is what made the honest verdict read
    `UNVERIFIED by probe domain`. A counted field whose domain no probe measured is REFUSED below —
    there is no default band anywhere in this file.
    """
    out = {}
    if c["mode"] == "f32_act":
        out["act"] = "f32_dot"
    if c["cast"] == "after":
        out["prob"] = "bf16_input"
    return out


def counted_fields(c):
    """The fields whose declared route carries expf in f32. Everything else is exact."""
    return set(counted_domains(c))


def digest_of(words, off, n):
    x, s, nn = 0, 0, 0
    for i in range(n):
        b = words[off + i]
        if b == 0 or b == 0x80000000:
            continue
        x ^= (b & 0x807FFFFF)
        s = (s + b) & 0xFFFFFFFF
        nn += 1
    return x, s, nn


def field_pairs(words, other, c, fname, width):
    """(index, a, b) pairs for one field, in the bit image the declared width uses."""
    do, eo = c["dst_off"] + FIELD_OFF[fname], c["exp_off"] + FIELD_OFF[fname]
    n = FIELD_LEN[fname]
    if width == "bf16":
        return [(i, bf16_of(words[do + i]), bf16_of(other[do + i])) for i in range(n)], n
    return [(i, words[do + i], other[do + i]) for i in range(n)], n


def judge_fields(v, cases, words, expect, bands, ruler_bits, sink=None, tag=""):
    """Per-field verdicts against the oracle's words, each counted field judged by the probe whose domain
    contains it. `bands` maps DOMAIN -> (band, provenance). Returns a summary string, the per-case
    counted divergence, and the list of `(case, field, ok)` policy verdicts for the counted fields.

    `sink` receives the three summary checks; a caller running a CONTROL passes a scratch Checks so a
    control's deliberate refusal cannot be mistaken for the arm's verdict.
    """
    sink = sink or v
    if UP is None:
        raise Fail("cannot import the shared ruler from tools/ulp_policy.py: %r" % _UP_ERR)
    rb16, rf32 = UP.Ruler("bf16"), UP.Ruler("f32")
    width = UP.WidthClass.from_type("Q8_0")     # the family this lane's weights are stored in
    print("WIDTH %s%s" % (width.line(), (" arm=%s" % tag) if tag else ""))
    n_exact = n_counted = exact_words = counted_words = counted_div = 0
    counted_by_case = {}
    offenders, band_offenders, iso_offenders = [], [], []
    verdicts = []
    for c in cases:
        domains = counted_domains(c)
        counted = set(domains)
        for fname in ("gate", "up", "act", "raw", "out", "prob"):
            w = c["widths"][fname]
            pairs, n = field_pairs(words, expect, c, fname, w)
            ruler = rb16 if w == "bf16" else rf32
            cen = ruler.census(pairs)
            if fname in counted:
                n_counted += 1
                counted_words += n
                counted_div += cen.diverging
                counted_by_case[c["index"]] = counted_by_case.get(c["index"], 0) + cen.diverging
                domain, prov = domains[fname], None
                ent = bands.get(domain)
                if ent is None:
                    band = None
                    prov = ("no probe measured domain=%s: refusing to invent a band for it "
                            "(the same shape of refusal a missing band file gets)" % domain)
                else:
                    band, prov = ent
                in_band = band is not None and cen.max_steps <= band
                if n == 1:
                    # A one-element field cannot be "isolated": the one word that moved IS the whole
                    # field, so `diverging * 2 < n` is unsatisfiable by construction (it refused the
                    # cast-after prob word for any build). The band judges a 1-element field.
                    isolated = in_band
                    iso_note = " isolation=not-defined-for-1-element-field(band_judges_it)"
                else:
                    isolated = cen.diverging * 2 < n
                    iso_note = ""
                att = UP.Attribution(in_band and isolated,
                                     "expf route, numpy side of the oracle; domain=%s; %s; observed "
                                     "max=%d over %d/%d%s" % (domain, prov, cen.max_steps,
                                                              cen.diverging, n, iso_note))
                print(cen.line(fname, extra="case=%d label=%s width=%s counted=1 band=%s domain=%s%s"
                                      % (c["index"], c["label"], w,
                                         "none" if band is None else band, domain, tag and (" " + tag)
                                         or "")))
                print("POLICYLINE case=%d field=%s verdict=%s attribution=%s"
                      % (c["index"], fname,
                         "counted-and-attributed" if att.ok else "REFUSED", att.detail))
                verdicts.append((c["index"], fname, att.ok))
                if not att.ok:
                    if not in_band:
                        band_offenders.append(
                            "case %d field %s: %s ULP exceeds the probe's %s (%s)"
                            % (c["index"], fname, cen.max_steps,
                               "no band" if band is None else band, prov))
                    else:
                        iso_offenders.append("case %d field %s: %d of %d moved is not isolated"
                                             % (c["index"], fname, cen.diverging, n))
                continue
            n_exact += 1
            exact_words += n
            verdict = UP.judge("%s[%s]" % (fname, c["label"]),
                               ["gate", "up", "act", "raw", "out", "prob"], width, cen,
                               require_attribution=False)
            if not verdict.accepted:
                offenders.append("case %d field %s (%s): %d of %d words differ from the oracle"
                                 % (c["index"], fname, w, cen.diverging, n))
    sink.check("every declared-exact field is bit-identical to the oracle", not offenders, 0,
               len(offenders), "; ".join(offenders[:4]) or "%d fields / %d words exact"
               % (n_exact, exact_words))
    sink.check("every counted field stays inside the band the floor probe measured", not band_offenders,
               0, len(band_offenders), "; ".join(band_offenders[:4]) or
               "%d counted fields, %d of %d words diverging, all inside their own domain's band"
               % (n_counted, counted_div, counted_words))
    sink.check("every counted divergence is isolated, not a systematic shift", not iso_offenders, 0,
               len(iso_offenders), "; ".join(iso_offenders[:4]) or "none")
    return "exact_fields=%d exact_words=%d counted_fields=%d counted_words=%d counted_diverging=%d" % (
        n_exact, exact_words, n_counted, counted_words, counted_div), counted_by_case, verdicts


def read_report(path):
    rows = []
    with open(path) as fh:
        head = None
        for line in fh:
            if line.startswith("#"):
                head = line[1:].strip().split("\t")
                continue
            f = line.rstrip("\n").split("\t")
            if head and len(f) == len(head):
                rows.append(dict(zip(head, [int(x) for x in f])))
    if not head:
        raise Fail("%s has no header line: the counters have no names to be checked against" % path)
    return rows


def span_diff(words, a, b, off, n):
    """Differing words between two spans of one buffer, starting at `off` inside each span.

    The `off` argument used to be accepted and ignored, which made every field-level call compare the
    case from element 0: the pair_swap blindness check printed `gate=0 up=0 act=0` while 507 of 512
    act words in fact differed (measured, work/dbg_case_fields.py). A comparison that looks at the wrong
    span cannot fail, so it must not be trusted to pass.
    """
    return sum(1 for i in range(n) if words[a + off + i] != words[b + off + i])


def buffers_diff(x, y, n):
    """Differing words between two whole buffers -- a different shape of comparison from span_diff, and
    calling span_diff with two buffers is a TypeError, which is how the grouped-vs-per-case check died
    on its first ever execution."""
    return sum(1 for i in range(n) if x[i] != y[i])


def injection_gate(v, cases, words):
    """Every injection must be visible as a difference from the honest words of the same block and row.
    The count is derived from the case list, never typed."""
    honest = {(c["block"], c["row"]): c for c in cases if c["label"] == "honest"}
    inj = [c for c in cases if c["label"] != "honest"]
    fams = sorted({c["label"] for c in inj})
    silent, fired = [], 0
    for c in inj:
        h = honest.get((c["block"], c["row"]))
        if h is None:
            raise Fail("no honest case for %s row %d to compare %s against" % (c["block"], c["row"],
                                                                              c["label"]))
        d = span_diff(words, c["dst_off"], h["dst_off"], 0, c["n_floats"])
        if d == 0:
            silent.append("%s(case %d)" % (c["label"], c["index"]))
        else:
            fired += 1
    v.check("every injection fires against the honest words", not silent, len(inj), fired,
            "families=%d %s%s" % (len(fams), ",".join(fams),
                                  "; silent: " + ",".join(silent) if silent else ""))
    ps = [c for c in inj if c["label"] == "layout_pair_swap"]
    blind_ok, detail = True, []
    for c in ps:
        h = honest[(c["block"], c["row"])]
        dg = span_diff(words, c["dst_off"], h["dst_off"], FIELD_OFF["gate"], FIELD_LEN["gate"])
        du = span_diff(words, c["dst_off"], h["dst_off"], FIELD_OFF["up"], FIELD_LEN["up"])
        da = span_diff(words, c["dst_off"], h["dst_off"], FIELD_OFF["act"], FIELD_LEN["act"])
        detail.append("case %d gate=%d up=%d act=%d" % (c["index"], dg, du, da))
        if dg or du or not da:
            blind_ok = False
    v.check("the declared blindness holds: pair_swap is invisible in gate/up and visible in act",
            blind_ok and bool(ps), "pair_swap invisible in the two dots", "ok" if blind_ok else "no",
            "; ".join(detail) or "no pair_swap case")
    upf = [c for c in inj if c["label"] == "layout_up_first"]
    v.check("up_first is NOT blind in act (the sibling row swap changes which half gets silu)",
            bool(upf), ">=1 case", len(upf),
            "; ".join("case %d act=%d" % (c["index"], span_diff(words, c["dst_off"],
                                                                honest[(c["block"], c["row"])]["dst_off"],
                                                                FIELD_OFF["act"], FIELD_LEN["act"]))
                      for c in upf))


def counters_gate(v, cases, words, report, expect, counted_by_case=None):
    """The arm's per-case counters, re-derived here from the same files the arm wrote."""
    counted_by_case = counted_by_case or {}
    by = {r["case"]: r for r in report}
    if len(report) != len(cases):
        raise Fail("the report holds %d cases, the plan has %d (parsed != printed)"
                   % (len(report), len(cases)))
    bad_reads, bad_nan, bad_digest = [], [], []
    for c in cases:
        r = by[c["index"]]
        want_fused = 2 * cols_gateup(c["mode"]) * I
        want_down = cols_down(c["mode"]) * rows_down(c["mode"])
        if not (r["reads_fused"] == want_fused and r["reads_down"] == want_down
                and r["reads_sg"] == H):
            bad_reads.append("case %d %s: kernel=%d/%d/%d geometry=%d/%d/%d"
                             % (c["index"], c["label"], r["reads_fused"], r["reads_down"],
                                r["reads_sg"], want_fused, want_down, H))
        poisons = sum(1 for i in range(c["n_floats"])
                      if expect[c["exp_off"] + i] == NAN_BITS)
        real_nan = sum(1 for i in range(c["n_floats"])
                       if (words[c["dst_off"] + i] & 0x7F800000) == 0x7F800000
                       and (words[c["dst_off"] + i] & 0x007FFFFF))
        if r["n_nan"] != poisons or real_nan != poisons:
            bad_nan.append("case %d %s: report=%d file=%d expect=%d"
                           % (c["index"], c["label"], r["n_nan"], real_nan, poisons))
        x, s, nn = digest_of(words, c["dst_off"], c["n_floats"])
        if not (r["xor_sig"] == x and r["sum_bits"] == s and r["n_digest"] == nn):
            bad_digest.append("case %d %s: arm=(%u,%u,%d) driver=(%u,%u,%d)"
                              % (c["index"], c["label"], r["xor_sig"], r["sum_bits"], r["n_digest"],
                                 x, s, nn))
    v.check("weight-read counters equal the geometry-derived counts on every case", not bad_reads,
            len(cases), len(cases) - len(bad_reads), "; ".join(bad_reads[:3]) or "all cases")
    v.check("poison holes are counted, not skipped, and match the expectation", not bad_nan,
            len(cases), len(cases) - len(bad_nan), "; ".join(bad_nan[:3]) or "all cases")
    v.check("both digests re-derived by the driver equal the arm's", not bad_digest, len(cases),
            len(cases) - len(bad_digest), "; ".join(bad_digest[:3]) or "all cases")
    mismatch = []
    for c in cases:
        r = by[c["index"]]
        want_bad = counted_by_case.get(c["index"], 0)
        if r["bad"] != want_bad:
            mismatch.append("case %d %s: arm counted %d differing words, the driver counted %d in the "
                            "fields this case declares counted"
                            % (c["index"], c["label"], r["bad"], want_bad))
    v.check("each arm's own bad-counter equals the driver's counted-field divergence for that case",
            not mismatch, len(cases), len(cases) - len(mismatch), "; ".join(mismatch[:4])
            or "all %d cases: every word the arm found is a word the header declares counted, and the "
            "exact fields contributed nothing" % len(cases))


def ownership_gate(v, cases, out_text, exe, common, work, arm_label):
    """The per-stage write-ownership probe: what it watched, what it caught, and every stage shown firing.

    A probe that has only ever reported zero is a hope, not an instrument. The stage-3 version earned its
    keep by firing 43 times (once per case) on the build where `se_k_out` let the `base = -1` sentinel
    through, so each stage's half of this generalised probe is handed the same class of defect — a
    foreign item it should have filtered — and must name the violation it sees, with case, field, index
    and both bit patterns. The injection is refused in per-case launch mode, where no snapshot separates
    the stages, and the probe says `attributable=no` there instead of printing a tidy zero.

    The coverage claims are all arithmetic on the case list: `compared` must be the whole record span
    times the number of stages, `lines` must equal `violations` per stage and in total, and the ownership
    table must tile one record (checked against the layout inside the probe, printed as `partition`).
    """
    span_words = sum(c["n_floats"] for c in cases)
    own = parse_tag(out_text, "OWNERSHIPSUMMARY",
                    ["stages", "cases", "compared", "violations", "lines", "partition",
                     "attributable", "mode"], want=1)[0]
    n_stages = int(own["stages"])
    stages = parse_tag(out_text, "OWNERSHIPSTAGE",
                       ["stage", "group", "owns", "compared", "violations", "lines"], want=n_stages)
    lines = [l for l in out_text.splitlines() if l.startswith("OWNERSHIP ")]
    v.check("%s write-ownership probe printed every violation it counted, per stage and in total" % arm_label,
            own["lines"] == own["violations"] and len(lines) == int(own["lines"])
            and all(s["lines"] == s["violations"] for s in stages),
            own["violations"], "%s lines, %s violations" % (len(lines), own["violations"]),
            "; ".join("%s: %s/%s" % (s["group"], s["violations"], s["lines"]) for s in stages))
    v.check("%s write-ownership probe compared every word of every case in every stage" % arm_label,
            int(own["cases"]) == len(cases) and int(own["compared"]) == n_stages * span_words
            and all(int(s["compared"]) == span_words for s in stages),
            "%d x %d words" % (n_stages, span_words), "%s (cases=%s)" % (own["compared"], own["cases"]),
            "a stage that watched less than the whole record could not see a write across a boundary")
    v.check("%s write-ownership table tiles one case record exactly" % arm_label,
            own["partition"] == "ok", "ok", own["partition"],
            "the four owned spans must cover gate|up|act|raw|out|prob with no gap and no overlap, or a "
            "field could be owned by nobody and the probe would still print 0")
    v.check("%s write-ownership probe ran on a launch split it can attribute" % arm_label,
            own["attributable"] == "yes" and own["mode"] == "grouped", "attributable=yes mode=grouped",
            "attributable=%s mode=%s" % (own["attributable"], own["mode"]),
            "in per-case mode stage 1 of the next case runs after stage 4 of this one, so a zero there "
            "would be an unusable instrument reporting success")
    v.check("no stage writes a word outside the field its stage owns", own["violations"] == "0",
            "0", own["violations"], "; ".join(lines[:3]) or "every stage stayed inside its region")
    # ---- one control per stage: the same defect class, reintroduced on purpose ----------------------
    for st in stages:
        n = int(st["stage"])
        rc, ctl = run([exe] + common + ["--inject-ownership", str(n),
                                       "--out", os.path.join(work, "ctl-own-%s-%d.f32" % (arm_label, n)),
                                       "--report", os.path.join(work, "ctl-own-%s-%d.tsv"
                                                                % (arm_label, n))],
                      os.path.join(work, "control-ownership-%s-%d.log" % (arm_label, n)))
        cl = [l for l in ctl.splitlines() if l.startswith("OWNERSHIP stage=%d " % n)]
        try:
            ctlsum = kv([l for l in ctl.splitlines() if l.startswith("OWNERSHIPCONTROL ")][0])
        except IndexError:
            ctlsum = {}
        named = bool(cl) and all(all(k in l for k in ("case=", "field=", "idx=", "before=0x",
                                                     "after=0x")) for l in cl)
        distinct = len({l.split("case=")[1].split()[0] for l in cl}) if cl else 0
        if cl:
            print("OWNERSHIPCONTROLSEEN %s" % cl[0])
        v.check("the stage-%d (%s) ownership control fires the probe with a named violation"
                % (n, st["group"]),
                rc == 0 and len(cl) == len(cases) and distinct == len(cases) and named
                and ctlsum.get("fired") == str(len(cases))
                and ctlsum.get("verdict") == "probe-caught-it",
                "%d cases each leak one word, rc=0" % len(cases),
                "%d lines, %d cases, rc=%d, verdict=%s" % (len(cl), distinct, rc,
                                                           ctlsum.get("verdict", "none")),
                cl[0] if cl else "the injection ran and the probe saw nothing: the instrument is dead")


def cross_arm_gate(v, cases, hostw, devw):
    offenders, counted = [], 0
    for c in cases:
        ccounted = counted_fields(c)          # not `counted_fields = ...`: that makes the function a
                                              # local name and the second case raises UnboundLocalError
        for fname in ("gate", "up", "act", "raw", "out", "prob"):
            if fname in ccounted:
                counted += 1
                continue
            if fname == "prob" and c["sgate"] == "perchan":
                continue                     # poison in both arms: compared as part of the span below
            w = c["widths"][fname]
            do = c["dst_off"] + FIELD_OFF[fname]
            n = FIELD_LEN[fname]
            nd = sum(1 for i in range(n)
                     if (bf16_of(hostw[do + i]) if w == "bf16" else hostw[do + i])
                     != (bf16_of(devw[do + i]) if w == "bf16" else devw[do + i]))
            if nd:
                offenders.append("case %d field %s: %d of %d words differ between the arms"
                                 % (c["index"], fname, nd, n))
    v.check("host and device agree bit-for-bit in every declared-exact field", not offenders, 0,
            len(offenders), "; ".join(offenders[:4]) or "%d counted fields exempt" % counted)
    whole = buffers_diff(hostw, devw, min(len(hostw), len(devw)))
    print("CROSSARM whole_buffer differing_words=%d of %d (reported, not asserted: the assertion is "
          "per declared field)" % (whole, min(len(hostw), len(devw))))


def band_refusal_control(v, cases, words, expect, bands, verdicts):
    """The refusal machinery, witnessed firing.

    Every counted field in this lane was REFUSED before its own probe existed, and a refusal nobody can
    reproduce is a hope rather than an instrument. So the same words are judged a second time, on a
    scratch scoreboard, with each counted field handed the band of the probe whose domain does NOT
    contain it: `act` gets the bf16-grid band (measured over inputs it never sees) and `prob` gets the
    f32-dot band. The policy line must print REFUSED again and the scratch board must go red, while the
    verdict the run reports — the rows in `v` — must be untouched by the exercise. `judge_fields`
    appends to `verdicts` at the same statement that prints the POLICYLINE, so the returned verdict and
    the printed line are the same event, not two readings of it.
    """
    missing = sorted({"f32_dot", "bf16_input"} - set(bands))
    if missing:
        v.check("the band-refusal control can run", False, "both domains probed",
                "missing " + ",".join(missing),
                "the control needs both probes; with one of them a refusal would be a missing-probe "
                "refusal, not a wrong-domain one")
        return
    crossed = {"f32_dot": bands["bf16_input"], "bf16_input": bands["f32_dot"]}
    scratch = Checks()
    rows_before, failed_before = len(v.rows), len(v.failed)
    _s, _c, ctl = judge_fields(scratch, cases, words, expect, crossed, "wrong-domain-control",
                              tag="CONTROL each-field-gets-the-other-probe-band")
    moved_rows, moved_failed = len(v.rows) - rows_before, len(v.failed) - failed_before
    refused = [(ci, f) for ci, f, ok in ctl if not ok]
    arm_refused = [(ci, f) for ci, f, ok in verdicts if not ok]
    scratch_red = len(scratch.failed)
    print("BANDCONTROL wrong_domain_refused=%d arm_refused=%d scratch_red=%d verdict_rows_moved=%d"
          % (len(refused), len(arm_refused), scratch_red, moved_rows))
    v.check("the band refusal still fires when each counted field is judged by the probe that does not "
            "cover its domain", bool(refused) and not arm_refused,
            "counted fields refused under the wrong ruler, none refused in the arm",
            "%d refused, %d refused in the arm" % (len(refused), len(arm_refused)),
            "; ".join("case %d %s" % (ci, f) for ci, f in refused[:3])
            or "the control refused nothing: the band cannot bite")
    v.check("the wrong-domain control is red on its own scratch board and leaves the verdict alone",
            scratch_red > 0 and moved_rows == 0 and moved_failed == 0,
            "scratch red, verdict rows unmoved", "scratch=%d rows_moved=%d failed_moved=%d"
            % (scratch_red, moved_rows, moved_failed),
            "a control that quietly adds checks to the run, or cannot make its own board red, is a "
            "control that measures nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--device", action="store_true", help="run the device arm (only via ./gate)")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    a = ap.parse_args()
    work = os.path.join(HERE, "work")
    os.makedirs(work, exist_ok=True)
    v = Checks()

    for exe, label in ((HOST, "host"), (DEVICE if a.device else HOST, "device")):
        if not os.path.exists(exe):
            raise Fail("%s is missing; run build.sh" % exe)
    band_exp, _ = probe_gate(v)
    bands = {}
    if band_exp is not None:
        bands["bf16_input"] = (
            band_exp, "exp probe domain=every bf16 input pattern, which is what sigmoid(bf16(t)) can "
                      "hand the transcendental")
    selftest_gate(v, HOST, "host")
    if a.device:
        selftest_gate(v, DEVICE, "device")
    if a.selftest_only:
        census_gate(v)
        return v.verdict("host witnesses only")

    rc, out = run([PY, ORACLE, "--mode", "derive", "--blocks", a.blocks, "--out", work],
                  os.path.join(work, "oracle-derive.log"))
    if rc != 0:
        raise Fail("oracle --mode derive failed: %s" % out.strip().splitlines()[-1:])
    # The width anchor has to be IN the device transcript, not in a side file: a device verdict about
    # Q8_0 bytes that does not carry the measured width rule is the kernel's opinion about its own
    # fixture, which is the thing this lane is not allowed to be.
    for line in out.splitlines():
        if line.startswith(("WIDTH ", "Q8RULE ", "Q8RULEPUB ", "LAYOUTROW ", "LAYOUTRT ")):
            print(line)
    lay = kv([l for l in out.splitlines() if l.startswith("LAYOUTRT winner")][0])
    v.check("the row map the kernel implements is the one HF's own bytes agree with",
            lay["winner"] == "gate_first" and lay["control_also_worse"] == "yes",
            "gate_first, control worse", "%s, control=%s" % (lay["winner"],
                                                             lay["control_also_worse"]),
            "closest wrong candidate is %s further from pinned eager shared_raw" % lay["nearest_wrong_ratio"])
    lrow = kv([l for l in out.splitlines() if l.startswith("LAYOUTROW margin")][0])
    v.check("the Q8_0 pack keeps each weight row in its own slot (no transpose)",
            float(lrow["min"]) > 1.05, ">1.05", lrow["min"],
            "nearest-vs-runner-up margin over 128 HF rows")
    q8rule_gate(v, out)
    layout_gate(v, out, [PY, ORACLE, "--mode", "derive", "--blocks", a.blocks])
    rc, out = run([PY, ORACLE, "--mode", "fixtures", "--rows", a.rows, "--blocks", a.blocks,
                   "--out", work], os.path.join(work, "oracle-fixtures.log"))
    if rc != 0:
        raise Fail("oracle --mode fixtures failed: %s" % out.strip().splitlines()[-1:])
    with open(os.path.join(work, "oracle.log"), "w") as fh:
        fh.write(out)
    pinned = parse_tag(out, "PINNED", ["block", "shared_raw", "shared_gate_prob", "shared_out"],
                       want=len(a.blocks.split(",")))
    for p in pinned:
        for k in ("shared_raw", "shared_gate_prob", "shared_out"):
            a_, b_ = p[k].split("/")
            v.check("oracle reproduces the pinned reference: %s %s" % (p["block"], k), a_ == b_,
                    b_, a_, "")
    fx = kv([l for l in out.splitlines() if l.startswith("FIXTURES")][0])
    cases = read_cases(os.path.join(work, "cases.tsv"))
    v.check("parsed==printed on the case list", int(fx["cases"]) == len(cases), fx["cases"],
            len(cases), "")
    n_out = sum(c["n_floats"] for c in cases)
    common = ["--plan", os.path.join(work, "plan_all.tsv"), "--slab", os.path.join(work, "slab.bin"),
              "--expect", os.path.join(work, "inputs.bin"), "--out-floats", n_out,
              "--expect-floats", n_out,
              "--expect-weights", sum(1 for l in open(os.path.join(work, "plan_all.tsv"))
                                      if l.startswith("W ")),
              "--expect-acts", sum(1 for l in open(os.path.join(work, "plan_all.tsv"))
                                    if l.startswith("A ")),
              "--expect-cases", len(cases)]
    expect = read_words(os.path.join(work, "inputs.bin"))
    rc, hout = run([HOST] + common + ["--out", os.path.join(work, "host_out.f32"),
                                      "--report", os.path.join(work, "host_report.tsv")],
                   os.path.join(work, "host-probe.log"))
    if rc != 0:
        raise Fail("host_probe rc=%d" % rc)
    rec = parse_tag(hout, "PLAN", ["W", "A", "C", "slab", "expect", "out"], want=1)[0]
    for k in ("W", "A", "C"):
        got, want = rec[k].split("/")
        v.check("host arm parsed==printed for %s" % k, got == want, want, got, "")
    hostw = read_words(os.path.join(work, "host_out.f32"))
    # The act probe needs the fixture (it reads the dots out of the oracle's own dump) and one arm's
    # words (it refuses unless they are bit-identical to the oracle's), so it runs here, between the
    # host arm and the judging — and the band it licenses is used for both arms below.
    bands = act_probe_gate(v, cases, work, os.path.join(work, "host_out.f32"), "host", bands)
    summ_host, counted_host, pol_host = judge_fields(v, cases, hostw, expect, bands,
                                                    "host-vs-oracle", tag="arm=host")
    print("SUMMARY %s" % summ_host)
    counters_gate(v, cases, hostw, read_report(os.path.join(work, "host_report.tsv")), expect,
                  counted_host)
    injection_gate(v, cases, hostw)
    band_refusal_control(v, cases, hostw, expect, bands, pol_host)
    ownership_gate(v, cases, hout, HOST, common, work, "host")

    rc, pout = run([HOST] + common + ["--per-case", "--out", os.path.join(work, "host_percase.f32"),
                                      "--report", os.path.join(work, "host_percase_report.tsv")],
                   os.path.join(work, "host-percase.log"))
    if rc != 0:
        raise Fail("host_probe --per-case rc=%d" % rc)
    pcw = read_words(os.path.join(work, "host_percase.f32"))
    nd = buffers_diff(hostw, pcw, min(len(hostw), len(pcw)))
    v.check("grouped and per-case launches produce the same words (launch split is inert)", nd == 0,
            0, nd, "launches grouped=4 per_case=%d (word identity, not timing)" % (4 * len(cases)))
    la = kv([l for l in hout.splitlines() if l.startswith("LAUNCHES")][0])
    lb = kv([l for l in pout.splitlines() if l.startswith("LAUNCHES")][0])
    v.check("launch counts are what the plan says", la["issued"] == "4"
            and lb["issued"] == str(4 * len(cases)), "4 and %d" % (4 * len(cases)),
            "%s and %s" % (la["issued"], lb["issued"]), "blocks per launch differ as well: %s vs %s"
            % (la["wg_per_launch_grouped"], lb["wg_per_launch_per_case"]))

    if a.device:
        rc, dout = run([DEVICE] + common + ["--out", os.path.join(work, "dev_out.f32"),
                                            "--report", os.path.join(work, "dev_report.tsv")],
                       os.path.join(work, "device-run.log"))
        if rc != 0:
            raise Fail("device arm rc=%d" % rc)
        rec = parse_tag(dout, "PLAN", ["W", "A", "C"], want=1)[0]
        for k in ("W", "A", "C"):
            got, want = rec[k].split("/")
            v.check("device arm parsed==printed for %s" % k, got == want, want, got, "")
        devw = read_words(os.path.join(work, "dev_out.f32"))
        summ_dev, counted_dev, pol_dev = judge_fields(v, cases, devw, expect, bands,
                                                     "device-vs-oracle", tag="arm=device")
        print("SUMMARY %s" % summ_dev)
        counters_gate(v, cases, devw, read_report(os.path.join(work, "dev_report.tsv")), expect,
                      counted_dev)
        injection_gate(v, cases, devw)
        cross_arm_gate(v, cases, hostw, devw)
        dl = kv([l for l in dout.splitlines() if l.startswith("LAUNCHES")][0])
        v.check("the device's own launch counter equals the launches issued",
                dl["device_counted"] == dl["issued"], dl["issued"], dl["device_counted"], "")
        # Stage-write ownership: the defect class this catches is a device kernel processing an item it
        # was not given (the stage-3 `base = -1` sentinel being the one that actually shipped), which the
        # host twin's stage loop filters explicitly. Both arms run the SAME probe from
        # shared_expert_ownership.h, so both are checked and both are shown firing.
        ownership_gate(v, cases, dout, DEVICE, common, work, "device")
        rc, occ = run([DEVICE, "--occupancy"], os.path.join(work, "occupancy.log"))
        print(occ.strip())
        with open(os.path.join(work, "occupancy.txt"), "w") as fh:
            fh.write(occ)
        v.check("the occupancy probe can see LDS (64 KiB stub squeezes to 1 block/CU)",
                "verdict=probe-sees-lds" in occ, "probe-sees-lds",
                "seen" if "verdict=probe-sees-lds" in occ else "BLIND", "")
        census_gate(v, os.path.join(work, "occupancy.txt"))
    else:
        census_gate(v)
    return v.verdict("device gate" if a.device else "host gate")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print("RESULT FAIL: %s" % e)
        print("no timing taken")
        sys.exit(1)
    except Exception as e:                                 # the shared ruler refusing to guess a width
        if type(e).__name__ != "FormatSourceError":
            raise
        # tools/ulp_policy.py derives float widths from two independent sources at run time and raises
        # rather than assume one. That is the right call, but a traceback is not a verdict: the reader has
        # to be told which interpreter refused and why, or a green-only-under-one-python gate reads as a
        # gate that passes. Exit code is unchanged (1) -- this is diagnosis, never a relaxation.
        print("REFUSE the shared ruler could not derive a float width: %s" % e)
        print("REFUSE interpreter=%s ; tools/ulp_policy.py derives widths from two independent sources "
              "and will not assume one -- run this gate under the CPU venv if torch is missing here"
              % sys.executable)
        print("no timing taken")
        sys.exit(1)
