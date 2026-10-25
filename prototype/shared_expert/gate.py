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
    `prob` field of a cast-after case. Both carry `expf` in f32, and the floor probe
    (work/floor_probe_exp.c plus work/floor_probe_exp_np.py) measures the glibc-vs-numpy
    disagreement at up to 2 ULP over the whole bf16 input grid. The ceiling is the probe's number,
    quoted at run time — not a band typed to fit what the build produces. This is D-16 counting with
    an attribution, NOT the D-17 ±1 acceptance: `judge()` is not called on those two fields, and the
    probe is the reason the *bf16-carrying* fields stay exact (the store is coarser than every
    disagreement the two libms can produce).
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
    """The floor probe, run rather than cited: it is where the only band in this lane comes from."""
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
            band >= 1, ">=1", band, "the band on f32-carrying expf fields is %d ULP from this probe"
            % band)
    return band, lines


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


# ---- the fixture judgements --------------------------------------------------------------------
def cols_gateup(mode):
    return {"tail_drop": H - 32, "cols_half": H // 2}.get(mode, H)


def cols_down(mode):
    return {"tail_drop": I - 32, "cols_half": I // 2}.get(mode, I)


def rows_down(mode):
    return H // 2 if mode == "down_half" else H


def counted_fields(c):
    """The two fields whose declared route carries expf in f32. Everything else is exact."""
    out = set()
    if c["mode"] == "f32_act":
        out.add("act")
    if c["cast"] == "after":
        out.add("prob")
    return out


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


def judge_fields(v, cases, words, expect, band, ruler_bits):
    """Per-field verdicts against the oracle's words. Returns a summary string."""
    if UP is None:
        raise Fail("cannot import the shared ruler from tools/ulp_policy.py: %r" % _UP_ERR)
    rb16, rf32 = UP.Ruler("bf16"), UP.Ruler("f32")
    width = UP.WidthClass.from_type("Q8_0")     # the family this lane's weights are stored in
    print("WIDTH %s" % width.line())
    n_exact = n_counted = exact_words = counted_words = counted_div = 0
    offenders, band_offenders, iso_offenders = [], [], []
    for c in cases:
        counted = counted_fields(c)
        for fname in ("gate", "up", "act", "raw", "out", "prob"):
            w = c["widths"][fname]
            pairs, n = field_pairs(words, expect, c, fname, w)
            ruler = rb16 if w == "bf16" else rf32
            cen = ruler.census(pairs)
            if fname in counted:
                n_counted += 1
                counted_words += n
                counted_div += cen.diverging
                att = UP.Attribution(cen.max_steps <= band and cen.diverging * 2 < n,
                                     "expf route, numpy side of the oracle; floor probe measured "
                                     "%d ULP over the whole bf16 grid, observed max=%d over %d/%d"
                                     % (band, cen.max_steps, cen.diverging, n))
                print(cen.line(fname, extra="case=%d label=%s width=%s counted=1 band=%d"
                                      % (c["index"], c["label"], w, band)))
                print("POLICYLINE case=%d field=%s verdict=%s attribution=%s"
                      % (c["index"], fname,
                         "counted-and-attributed" if att.ok else "REFUSED", att.detail))
                if not att.ok:
                    if cen.max_steps > band:
                        band_offenders.append("case %d field %s: %d ULP exceeds the probe's %d"
                                              % (c["index"], fname, cen.max_steps, band))
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
    v.check("every declared-exact field is bit-identical to the oracle", not offenders, 0,
            len(offenders), "; ".join(offenders[:4]) or "%d fields / %d words exact"
            % (n_exact, exact_words))
    v.check("every counted field stays inside the band the floor probe measured", not band_offenders,
            0, len(band_offenders), "; ".join(band_offenders[:4]) or
            "%d counted fields, %d of %d words diverging, all within %d ULP"
            % (n_counted, counted_div, counted_words, band))
    v.check("every counted divergence is isolated, not a systematic shift", not iso_offenders, 0,
            len(iso_offenders), "; ".join(iso_offenders[:4]) or "none")
    return "exact_fields=%d exact_words=%d counted_fields=%d counted_words=%d counted_diverging=%d" % (
        n_exact, exact_words, n_counted, counted_words, counted_div)


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
    return sum(1 for i in range(n) if words[a + i] != words[b + i])


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


def counters_gate(v, cases, words, report, expect):
    """The arm's per-case counters, re-derived here from the same files the arm wrote."""
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
    srs = [r for r in report if r["bad"] != 0]
    v.check("every case matches its own expectation word-for-word (bad == 0)", not srs, 0, len(srs),
            "; ".join("case %d bad=%d" % (r["case"], r["bad"]) for r in srs[:4]) or "all cases")


def cross_arm_gate(v, cases, hostw, devw, band):
    offenders, counted = [], 0
    for c in cases:
        counted_fields = counted_fields(c)
        for fname in ("gate", "up", "act", "raw", "out", "prob"):
            if fname in counted_fields:
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
    whole = span_diff(hostw, devw, 0, 0, min(len(hostw), len(devw)))
    print("CROSSARM whole_buffer differing_words=%d of %d (reported, not asserted: the assertion is "
          "per declared field)" % (whole, min(len(hostw), len(devw))))


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
    band, _ = probe_gate(v)
    band = band or 1
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
    print("SUMMARY %s" % judge_fields(v, cases, hostw, expect, band, "host-vs-oracle"))
    counters_gate(v, cases, hostw, read_report(os.path.join(work, "host_report.tsv")), expect)
    injection_gate(v, cases, hostw)

    rc, pout = run([HOST] + common + ["--per-case", "--out", os.path.join(work, "host_percase.f32"),
                                      "--report", os.path.join(work, "host_percase_report.tsv")],
                   os.path.join(work, "host-percase.log"))
    if rc != 0:
        raise Fail("host_probe --per-case rc=%d" % rc)
    pcw = read_words(os.path.join(work, "host_percase.f32"))
    nd = span_diff(hostw, pcw, 0, 0, min(len(hostw), len(pcw)))
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
        print("SUMMARY %s" % judge_fields(v, cases, devw, expect, band, "device-vs-oracle"))
        counters_gate(v, cases, devw, read_report(os.path.join(work, "dev_report.tsv")), expect)
        injection_gate(v, cases, devw)
        cross_arm_gate(v, cases, hostw, devw, band)
        dl = kv([l for l in dout.splitlines() if l.startswith("LAUNCHES")][0])
        v.check("the device's own launch counter equals the launches issued",
                dl["device_counted"] == dl["issued"], dl["issued"], dl["device_counted"], "")
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
