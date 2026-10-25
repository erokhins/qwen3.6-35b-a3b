#!/usr/bin/env python3
"""gate.py — the driver for prototype/moe_gather. It never touches the GPU lock.

    ./gate --selftest           host-only witnesses (three-arm selftest, parser controls)
    ./gate --check-only         build nothing, run the structural plan checks on both arms
    ./gate                      the device gate: correctness only, PASS/FAIL, no timing
    ./gate --max-rows N         cut the SAMPLE in numbers only (never an assertion); prints CUT

`gate` (the shell wrapper) is the only acquirer of the repo's GPU lock tool, and it checks this file
for any mention of that tool at run time: a driver that self-acquires and is called by a caller already
holding the lock parks forever with no CPU, no GPU, no message and no exit
(pitfalls/50-harness-and-shell.md §"Self-locking gate drivers deadlock silently").

What is judged, and by whom:
  * every case's own span against `expect.f32` (the oracle's words), classified by its plan class:
      MATCH / ALT / IMMUNE  -> bad == 0, bit-exact, no tolerance
      MISMATCH              -> bad  > 0, i.e. the injection FIRED
      REPORT                -> counted per field, never banded (the transcendental route)
  * the ALT/IMMUNE class is also checked against the honest case's OUTPUT in the same run, so an
    injection cannot pass by agreeing with an expectation nobody derived: an IMMUNE case must equal the
    honest case's words, and an ALT case must differ from them.
  * host arm vs device arm, per DECLARED FIELD (D-16 option A): bit-identical in every field the case
    declares exact, census-and-attribution in the fields the lane counts and never bands. Same plan,
    same slab, same expectation. See cross_arm().
  * the kernel's own per-case counters against gate.py's recount from the output file (two sites, one
    truth: the right-hand side comes from the file, not from the counter).
  * launch counts, grouped vs per-slot, and the device's own launch counter.
Every parser asserts parsed == printed. CPU-only modes take no lock.
"""

import argparse
import os
import re
import subprocess
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRIVER = os.path.join(HERE, "moe_gather")
PROBE = os.path.join(HERE, "host_probe")
ORACLE = os.path.join(HERE, "oracle_arm.py")
PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"

# The repo's one ULP ruler (tools/ulp_policy.py, DECISIONS.md D-17). Imported, not re-derived: five
# files used to carry their own ULP logic and two of the rulers disagreed about what a step is. The import
# is deferred-and-loud rather than optional: a census nobody can compute is a census nobody prints,
# and D-16 permits a divergence only while it is printed.
sys.path.insert(0, os.path.join(REPO, "tools"))
try:
    import ulp_policy as UP
    _UP_ERR = None
except Exception as _e:                                   # pragma: no cover
    UP, _UP_ERR = None, _e


class Fail(Exception):
    pass


def run(cmd, log=None):
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if log:
        open(log, "w").write(p.stdout + (("\n" + p.stderr) if p.stderr.strip() else ""))
    return p.returncode, p.stdout


def kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_tag(text, tag, keys, want=None):
    """Pull key=value records for a line tag. The value pattern takes any non-space token, so a
    word-valued field is never silently dropped (pitfalls/40 §"a key=value parser whose value pattern
    is numeric")."""
    recs = []
    for line in text.splitlines():
        if line.startswith(tag + " ") or line == tag:
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


def read_cases(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 17:
                raise Fail("%s: case row has %d columns, the oracle writes 17 (a missing column must "
                           "not default its way past the check that needs it)" % (path, len(f)))
            rows.append(dict(case_index=int(f[0]), kind=int(f[1]), mode=int(f[2]), mode_name=f[3],
                             block=int(f[4]), block_index=int(f[5]), row=int(f[6]), slot=int(f[7]),
                             expect=int(f[8]), expect_name=f[9], src=f[10], exact=f[11],
                             dst_off=int(f[12]), exp_off=int(f[13]), n_floats=int(f[14]),
                             combine_src=int(f[15]), field_exact=f[16]))
    return rows


def read_words(path, n):
    """The output/expectation files as uint32 word images.

    Read as words, not floats: every judgement here is bit identity, and comparing the image is what
    lets a NaN case be judged at all (pitfalls/30 §".view vs .to" — a `.to(float)` would launder a
    payload change into equality). Native byte order, same machine that wrote it."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) != 4 * n:
        raise Fail("%s holds %d bytes, the plan wants %d floats" % (path, len(buf), n))
    a = array("I")
    a.frombytes(buf)
    return a


def _same_word(x, y):
    if x == y:
        return True
    # two NaNs of any payload are "the same poison": the bit pattern asked for is NaN, and which
    # quiet-NaN payload the compiler emitted is not a fact the gate claims to control
    return ((x & 0x7F800000) == 0x7F800000 and (x & 0x007FFFFF) != 0 and
            (y & 0x7F800000) == 0x7F800000 and (y & 0x007FFFFF) != 0)


def is_nan_word(w):
    """NaN-ness, any payload — the same NaN rule `_same_word` and `mg_word_eq` use."""
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


def span_census(w, off, n):
    """(nan, zero, value) words in one span.

    `value` = finite and non-zero, the only kind of word that can carry a computed result. A span of
    exact zeros, of -0.0s, or of the driver's MG_NAN_BITS pre-fill all read value == 0, which is what
    makes "nothing was computed" a count rather than a reading of the log. Judged against the
    expectation's own census, so this never declares zero illegal in general — only absent where the
    oracle says a value belongs (results.md §6: silence was the bug, not the number).
    """
    n_nan = n_zero = n_val = 0
    for i in range(off, off + n):
        x = w[i]
        if is_nan_word(x):
            n_nan += 1
        elif x == 0 or x == 0x80000000:
            n_zero += 1
        else:
            n_val += 1
    return n_nan, n_zero, n_val



def n_diff2(a, off_a, b, off_b, n):
    """Words [off_a, off_a+n) of a against words [off_b, off_b+n) of b."""
    c = 0
    for i in range(n):
        if not _same_word(a[off_a + i], b[off_b + i]):
            c += 1
    return c


def n_diff(a, b, off=0, n=None):
    if n is None:
        n = len(a)
    return n_diff2(a, off, b, off, n)


def fields(kind, I, H):
    """(name, offset, length) per field, from the header's own layout."""
    if kind == 0:
        return [("gate", 0, I), ("up", I, I), ("s", 2 * I, I), ("raw", 3 * I, H)]
    return [("routed", 0, H)]


def selftest(v):
    """Both arms, three arms each, plus the control that makes the three-arm rule falsifiable."""
    for name, binp in (("host_probe", PROBE), ("moe_gather", DRIVER)):
        if not os.path.exists(binp):
            raise Fail("missing %s; run ./build.sh" % binp)
        rc, out = run([binp, "--selftest"], os.path.join(HERE, "work", "selftest-%s.log" % name))
        st = parse_tag(out, "SELFTEST", ["arm", "changed", "failures"], want=3)
        fin = parse_tag(out, "SELFTEST3", ["arms_run"], want=1)[0]
        honest = [s for s in st if s["arm"] == "0"][0]
        stub = [s for s in st if s["arm"] == "1"][0]
        crossed = [s for s in st if s["arm"] == "2"][0]
        ok = (rc == 0 and int(fin["arms_run"]) == 3 and int(honest["failures"]) == 0
              and int(honest["changed"]) == 0 and int(stub["changed"]) > 0
              and int(crossed["changed"]) > 0)
        v.check("selftest-%s arms=3 honest_changed=0 stub_changed>0 crossed_changed>0" % name, ok,
                3, "arms_run=%s honest_changed=%s stub_changed=%s crossed_changed=%s rc=%d"
                % (fin["arms_run"], honest["changed"], stub["changed"], crossed["changed"], rc))
        # THE CONTROL: a single clean arm must NOT pass. This is the defect HANDOFF §7.2 recorded, and
        # it is witnessed firing rather than asserted in prose.
        for arm in ("0", "1", "2"):
            rc2, out2 = run([binp, "--selftest", "--arm=%s" % arm])
            bad = parse_tag(out2, "SELFTEST3", ["arms_run"], want=1)[0]
            v.check("selftest-%s --arm=%s alone must FAIL" % (name, arm),
                    rc2 != 0 and int(bad["arms_run"]) == 1, 1, "arms_run=%s rc=%d"
                    % (bad["arms_run"], rc2))
        for badarg in ("--arm=9", "--arm=x", "--nonsense", "--arm"):
            rc3, _ = run([binp, "--selftest", badarg])
            v.check("argv-%s refused" % badarg, rc3 == 2, 1, "rc=%d" % rc3)
        # and the '=' form must be the same arm as the space form (HANDOFF §7.1)
        rc_eq, out_eq = run([binp, "--selftest", "--arm=1"])
        rc_sp, out_sp = run([binp, "--selftest", "--arm", "1"])
        v.check("argv-`--arm=1` == `--arm 1`", out_eq == out_sp and rc_eq == rc_sp, 1,
                "identical stdout and rc=%d" % rc_eq)
        # the crossed arm must print the expectation it ACTUALLY used (HANDOFF §7.3). Before the fix
        # the line printed the honest `want` while the verdict ran on the neighbour's, so a re-reader
        # saw `got == want` on a line that had just failed. The witness is therefore that every crossed
        # FAIL line names a want_src that is NOT its own check: the printed want is the borrowed one.
        # (got == want on such a line is honest — the borrowed answer can coincide.)
        rcx, outx = run([binp, "--selftest", "--arm", "2"])
        xs = [kv(l) for l in outx.splitlines() if l.startswith("FAIL selftest arm=2 ")]
        borrowed = [r for r in xs if r["want_src"] != r["check"]]
        v.check("crossed-arm FAIL lines print the borrowed want", len(xs) > 0
                and len(borrowed) == len(xs), len(xs),
                "%d crossed FAIL lines, %d of them name a want_src other than their own check"
                % (len(xs), len(borrowed)))


def declare(cases, geo, max_rows, cut):
    kinds = {}
    classes = {}
    modes = set()
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        classes[c["expect_name"]] = classes.get(c["expect_name"], 0) + 1
        modes.add(c["mode_name"])
    print("SAMPLE cases=%d expert=%d combine=%d modes=%d classes=%s"
          % (len(cases), kinds.get(0, 0), kinds.get(1, 0), len(modes),
             " ".join("%s=%d" % (k, classes.get(k, 0)) for k in
                      ("MG_EXP_MATCH", "MG_EXP_MISMATCH", "MG_EXP_ALT", "MG_EXP_IMMUNE",
                       "MG_EXP_REPORT"))))
    print("ROWS real=%d synthetic=%d (real rows are pinned tokens; synthetic rows are the "
          "integer-exact fixtures)" % (geo["real"], geo["synth"]))
    print("ELEMENTS judged=%d words (sum of case spans)" % sum(c["n_floats"] for c in cases))
    if cut:
        print("CUT max_rows=%d: pinned tokens per real block cut from %d; no case class, no mode "
              "and no assertion was removed by the cut" % (max_rows, geo["max_rows_full"]))
    return modes


class V(object):
    def __init__(self):
        self.n = 0
        self.bad = []

    def check(self, name, ok, n, note=""):
        self.n += 1
        if not ok:
            self.bad.append(name)
        print("CHECK name=%s ok=%d n=%d %s" % (name, 1 if ok else 0, n, note))

    def verdict(self, label):
        print("CHECKS ran=%d failed=%d %s" % (self.n, len(self.bad),
                                              ",".join(self.bad[:8]) if self.bad else ""))
        print("RESULT %s %s checks=%d findings=%d" % ("PASS" if not self.bad else "FAIL", label,
                                                      self.n, len(self.bad)))
        return 0 if not self.bad else 1


def adjudicate(v, cases, outw, expw, report, tag):
    """One arm's words against the oracle's words, plus the kernel's counters against the recount."""
    I, H = int(os.environ.get("MG_I", 512)), int(os.environ.get("MG_H", 2048))
    # What one expert case must decode if it reads every weight it is supposed to: gate|up are 2I rows
    # of H, down is H rows of I. Derived from the geometry, so it is a witness and not a band.
    want_reads = 2 * I * H + H * I
    rep = {}
    with open(report) as fh:
        lines = [l for l in fh if l.strip() and not l.startswith("#")]
    if len(lines) != len(cases):
        raise Fail("%s: report has %d records, the plan has %d cases" % (tag, len(lines), len(cases)))
    for line in lines:
        f = line.split("\t")
        rep[int(f[0])] = dict(bad=int(f[6]), nan=int(f[7]), route_bad=int(f[8]), route_n=int(f[9]))
    if len(rep) != len(cases):
        raise Fail("%s: parsed %d distinct report records for %d cases" % (tag, len(rep), len(cases)))
    honest_out = {}
    for c in cases:
        if c["mode_name"] == "MG_MODE_CORRECT":
            honest_out[(c["kind"], c["block"], c["row"], c["slot"])] = c
    counts = {"MATCH": 0, "MISMATCH": 0, "ALT": 0, "IMMUNE": 0, "REPORT": 0}
    fired = []            # (case_kind, name, fired?) - one entry per negative control that ran
    silent = no_value = nanmove = 0
    gate_cases = gate_ok = 0
    reach = {}            # mode_name -> [cases with a derived alternative, cases where it differs]
    # The injection denominator, recounted from the plan rather than accumulated by the loop below:
    # every MISMATCH case plus every ALT case that has a distinct honest counterpart to differ from.
    # If a case silently lost its honest counterpart its control would vanish from the denominator and
    # a smaller `checked` would read as a smaller problem (pitfalls/40 §"the unnamed denominator").
    want_inj = 0
    for c in cases:
        if c["expect_name"] == "MG_EXP_MISMATCH":
            want_inj += 1
        elif c["expect_name"] == "MG_EXP_ALT":
            hc = honest_out.get((c["kind"], c["block"], c["row"], c["slot"]))
            if hc is not None and hc["case_index"] != c["case_index"]:
                want_inj += 1
    for c in cases:
        cls = c["expect_name"].replace("MG_EXP_", "")
        counts[cls] = counts.get(cls, 0) + 1
        s, n = c["dst_off"], c["n_floats"]
        diff = n_diff(outw, expw, s, n)
        rec = rep[c["case_index"]]
        v.check("%s case=%d %s counter==recount" % (tag, c["case_index"], c["mode_name"]),
                rec["bad"] == diff, n, "kernel bad=%d recount=%d" % (rec["bad"], diff))
        # ---- the hard failure results.md §6 asked for, and it comes before any verdict on a number -
        # A record that carries no value is a DEFECT, not a row: never written (the pre-fill survives),
        # written from refused weight reads (all MG_NAN_BITS), or written from no weights at all (all
        # zero, which is also what a zero-filled device buffer looked like). Four data-derived clauses
        # on one attributed line:
        #   1 a word is NaN in the output exactly when it is NaN in the oracle
        #   2 where the oracle has a value, the record has at least one
        #   3 if the words show a NaN where the oracle has none, the kernel must have counted it
        #   4 if the kernel counted a NaN, the words must show one
        e_nan, e_zero, e_val = span_census(expw, c["exp_off"], n)
        o_nan, o_zero, o_val = span_census(outw, s, n)
        nanmove_here = sum(1 for i in range(n)
                           if is_nan_word(outw[s + i]) != is_nan_word(expw[c["exp_off"] + i]))
        said_nan = rec["nan"] > 0
        leak = o_nan > 0 and e_nan == 0
        ok_silent = (nanmove_here == 0
                     and (e_val == 0 or o_val > 0)
                     and (not leak or said_nan)
                     and (not said_nan or o_nan > 0))
        v.check("%s case=%d %s record carries a value" % (tag, c["case_index"], c["mode_name"]),
                ok_silent, n,
                "out nan=%d zero=%d value=%d | expect nan=%d zero=%d value=%d | nan-moved=%d "
                "kernel_nan=%d"
                % (o_nan, o_zero, o_val, e_nan, e_zero, e_val, nanmove_here, 1 if said_nan else 0))
        if not ok_silent:
            silent += 1
            no_value += 1 if (e_val > 0 and o_val == 0) else 0
            nanmove += 1 if nanmove_here > 0 else 0
        if c["kind"] == 0 and c["mode_name"] == "MG_MODE_CORRECT":
            # The expectation-independent witness that the expert path really read weights: the kernel
            # counted `route_n` successful decodes and geometry says how many there must be. The state
            # this lane shipped in had route_n=0 on every case and nothing asserted on it.
            v.check("%s case=%d weight reads == geometry" % (tag, c["case_index"]),
                    rec["route_n"] == want_reads, want_reads,
                    "kernel route_n=%d geometry=%d (gate|up 2I*H + down H*I)"
                    % (rec["route_n"], want_reads))
            # and the witness §6 named: a fixture whose expected gate field is entirely non-zero must
            # produce a non-zero gate, not a record that is merely "different"
            if span_census(expw, c["exp_off"], I)[2] == I:
                gate_cases += 1
                if span_census(outw, s, I)[2] == I:
                    gate_ok += 1
        if cls in ("MATCH", "ALT", "IMMUNE"):
            if cls in ("ALT", "IMMUNE") and c["kind"] == 0 and c["field_exact"] == "gate_up":
                # The transcendental field rule, applied to an alternative the same way it is applied
                # to a REPORT case: gate and up are fp32 dots over published bytes and must match the
                # alternative bit for bit; s and raw cross expf, so they are counted and never banded.
                # Which cases this covers is declared per case from the cls of the units the case
                # actually reads (MG_MODE_EXPERT_ROTATE on a synthetic row reads real experts), not
                # assumed per class, and gate.py refuses it for any MG_EXP_MATCH case.
                for fname, fo, fn in fields(0, I, H):
                    nd = n_diff(outw, expw, c["dst_off"] + fo, fn)
                    if fname in ("gate", "up"):
                        v.check("%s case=%d %s(%s) field=%s exact vs alt"
                                % (tag, c["case_index"], c["mode_name"], cls, fname),
                                nd == 0, fn, "differing=%d" % nd)
                    else:
                        print("COUNT case=%d %s field=%s differing=%d/%d (transcendental route, "
                              "counted not banded)"
                              % (c["case_index"], c["mode_name"], fname, nd, fn))
            else:
                v.check("%s case=%d %s(%s) bit-exact vs oracle"
                        % (tag, c["case_index"], c["mode_name"], cls), diff == 0, n,
                        "differing=%d" % diff)
        elif cls == "MISMATCH":
            v.check("%s case=%d %s FIRED (must differ)" % (tag, c["case_index"], c["mode_name"]),
                    diff > 0, n, "differing=%d" % diff)
            fired.append((c["kind"], c["mode_name"], diff > 0))
        if cls in ("ALT", "IMMUNE"):
            hc = honest_out.get((c["kind"], c["block"], c["row"], c["slot"]))
            if hc is not None and hc["case_index"] != c["case_index"]:
                nd = n_diff2(outw, hc["dst_off"], outw, s, n)
                # The class has to agree with the oracle's OWN pair of expectations, and that is
                # cross-checked here on the expectation buffer rather than trusted from cases.tsv:
                # an ALT whose derived alternative is bit-identical to the honest expectation cannot
                # differ from anything (asserting it must is asserting the impossible), and an IMMUNE
                # whose alternative does differ is a mislabelled control.
                e_nd = n_diff2(expw, hc["exp_off"], expw, c["exp_off"], n)
                v.check("%s case=%d %s class matches its derived expectation"
                        % (tag, c["case_index"], c["mode_name"]),
                        (e_nd > 0) if cls == "ALT" else (e_nd == 0), n,
                        "oracle's alternative vs oracle's honest differs in %d words; class=%s"
                        % (e_nd, cls))
                r = reach.setdefault(c["mode_name"], [0, 0])
                r[0] += 1
                r[1] += 1 if e_nd > 0 else 0
                if cls == "IMMUNE":
                    v.check("%s case=%d %s equals the honest output (immunity)"
                            % (tag, c["case_index"], c["mode_name"]), nd == 0, n,
                            "differs from honest in %d words" % nd)
                else:
                    v.check("%s case=%d %s differs from the honest output"
                            % (tag, c["case_index"], c["mode_name"]), nd > 0, n,
                            "differs from honest in %d words" % nd)
                    fired.append((c["kind"], "%s(alt-vs-honest)" % c["mode_name"], nd > 0))
        if cls == "REPORT":
            # the transcendental route is counted per field, never banded: gate and up are pure fp32
            # dots and must be bit-exact; s and raw depend on expf and are reported.
            for fname, fo, fn in fields(c["kind"], I, H):
                nd = n_diff(outw, expw, c["dst_off"] + fo, fn)
                if c["kind"] == 0 and fname in ("gate", "up") or c["kind"] == 1:
                    v.check("%s case=%d field=%s exact" % (tag, c["case_index"], fname),
                            nd == 0, fn, "differing=%d" % nd)
                else:
                    print("COUNT case=%d %s field=%s differing=%d/%d (transcendental route, "
                          "counted not banded)" % (c["case_index"], c["mode_name"], fname, nd, fn))
    print("POPULATION %s %s" % (tag, " ".join("%s=%d" % (k, counts.get(k, 0))
                                              for k in ("MATCH", "MISMATCH", "ALT", "IMMUNE",
                                                        "REPORT"))))
    print("INTEGRITY %s cases=%d silent_records=%d of which no_value=%d nan-moved=%d"
          % (tag, len(cases), silent, no_value, nanmove))
    v.check("%s no record is silent" % tag, silent == 0, len(cases),
            "silent=%d (no_value=%d nan-moved=%d)" % (silent, no_value, nanmove))
    # A mode may be immune on SOME rows — that is arithmetic, and the cases say so — but a mode that is
    # immune on EVERY row is untested, and an untested route has no gate (pitfalls/40 §"a route no
    # fixture reaches has no tests"). Derived from the plan's own alternatives, per mode.
    unreachable = sorted(m for m, (n_tot, n_dif) in reach.items() if n_dif == 0)
    print("ALTREACH %s modes=%d reachable=%d immune_everywhere=%s"
          % (tag, len(reach), len(reach) - len(unreachable), ",".join(unreachable) or "none"))
    v.check("%s every injection mode is reachable somewhere in the sample" % tag,
            len(reach) > 0 and not unreachable, len(reach),
            "immune on every case of their kind: %s" % (",".join(unreachable) or "none"))
    v.check("%s expert gate field nonzero wherever the oracle's is" % tag,
            gate_cases > 0 and gate_ok == gate_cases, gate_cases,
            "%d expert cases have an all-nonzero expected gate field, %d produced an all-nonzero gate"
            % (gate_ok, gate_cases))
    # The injections, split by which half of the kernel they test. The expert half is the one that
    # could not fire while the expert path produced nothing, and a total of 30/49 printed beside a
    # combine arm that was fully green read as "most controls work". Each half now has to carry its
    # own non-zero denominator, and the denominator itself is cross-checked against the plan.
    exp_inj = [x for x in fired if x[0] == 0]
    com_inj = [x for x in fired if x[0] == 1]
    print("INJECTIONS %s expert checked=%d fired=%d"
          % (tag, len(exp_inj), sum(1 for _, _, ok in exp_inj if ok)))
    print("INJECTIONS %s combine checked=%d fired=%d"
          % (tag, len(com_inj), sum(1 for _, _, ok in com_inj if ok)))
    print("INJECTIONS %s checked=%d fired=%d"
          % (tag, len(fired), sum(1 for _, _, ok in fired if ok)))
    for kind, nm, ok in fired:
        print("INJECTION arm=%s kind=%s fired=%d"
              % (nm, "expert" if kind == 0 else "combine", 1 if ok else 0))
    v.check("%s injection denominator matches the plan" % tag,
            want_inj > 0 and len(fired) == want_inj, want_inj,
            "plan-derived=%d counted=%d (expert=%d combine=%d)"
            % (want_inj, len(fired), len(exp_inj), len(com_inj)))
    v.check("%s expert injections all fired" % tag,
            len(exp_inj) > 0 and all(ok for _, _, ok in exp_inj), len(exp_inj),
            "did not fire: %s" % (",".join(nm for _, nm, ok in exp_inj if not ok) or "none"))
    v.check("%s combine injections all fired" % tag,
            len(com_inj) > 0 and all(ok for _, _, ok in com_inj), len(com_inj),
            "did not fire: %s" % (",".join(nm for _, nm, ok in com_inj if not ok) or "none"))
    return counts


def cross_arm(v, cases, hostw, devw, expw):
    """Host arm vs device arm, judged per DECLARED FIELD (DECISIONS.md D-16, the user's option A).

    The check used to compare the whole output buffer, which spanned the two fields this lane declares
    it does not band: `s` and `raw` carry an `expf`, glibc's and ROCm's libdevice disagree, and the
    lane counts that route and never bands it (`field_exact=gate_up`, 18 cases). Option A makes the
    cross-arm check inherit that declaration, with the three teeth D-16 attaches to it:

      1. every field the case declares exact -- `gate`/`up` for those 18 cases, everything elsewhere --
         must be bit-identical between the arms. That is *new* teeth: whole-buffer equality could never
         say so, and "gate/up differ 0 times" was a fact about the data rather than an assertion;
      2. the counted fields stay PRINTED, per case and per field, as a census in the shared ruler's
         format: element counts, the ULP histogram, max ULP, where the divergences sit;
      3. the accepted divergence is ATTRIBUTED by data, not waved through: both arms must differ from
         the pinned oracle in the same field (one arm alone diverging from the oracle is a device
         defect, not a route), and the divergence must be ISOLATED -- a strict majority of the field
         bit-identical between the arms. A wholesale change of rounding mode moves every element and
         is refused here. That is the tripwire that makes acceptance safe: it is what distinguishes
         "two compilers round `expf` differently on a few elements" from "the device arm computes a
         different function".

    What this deliberately is NOT: it is not the D-17 ±1-ULP acceptance, and `ulp_policy.judge()` is
    not called on `s`/`raw`. The measured cross-arm divergence runs to 3840 f32 ULP on `raw`
    (`work/probe_ulp_hist.py`), so ±1 could not forgive it and pretending otherwise would be a band
    typed to fit what the build produces. D-16 A is the authority here; what is shared with D-17 is
    the *ruler* and the census format, not a tolerance.
    """
    if UP is None:
        raise Fail("cannot import the shared ULP ruler from tools/ulp_policy.py: %r" % _UP_ERR)
    I, H = int(os.environ.get("MG_I", 512)), int(os.environ.get("MG_H", 2048))
    ruler = UP.Ruler("f32")
    print("RULER %s" % ruler.format.line())
    exact_words = counted_words = counted_div = 0
    exact_offenders = []
    att_offenders = []
    iso_offenders = []
    nan_payload = 0
    n_counted_fields = n_exact_fields = 0
    worst = (0.0, "")
    for c in cases:
        for fname, fo, fn in fields(c["kind"], I, H):
            declared_exact = c["field_exact"] == "all" or fname in ("gate", "up")
            do, eo = c["dst_off"] + fo, c["exp_off"] + fo
            nd = n_diff2(hostw, do, devw, do, fn)
            if declared_exact:
                n_exact_fields += 1
                exact_words += fn
                if nd:
                    exact_offenders.append("case %d field %s differs in %d of %d words"
                                           % (c["case_index"], fname, nd, fn))
                continue
            n_counted_fields += 1
            counted_words += fn
            counted_div += nd
            cen = ruler.census((i, hostw[do + i], devw[do + i]) for i in range(fn))
            nan_payload += cen.count(UP.NAN_PAIR)
            ho = n_diff2(hostw, do, expw, eo, fn)
            de = n_diff2(devw, do, expw, eo, fn)
            share = (cen.diverging / fn) if fn else 0.0
            if share > worst[0]:
                worst = (share, "case %d field %s" % (c["case_index"], fname))
            # D-16 condition 3, as data: the divergence appears in a field BOTH arms disagree with the
            # pinned oracle in. The oracle is an independent third side -- if only one arm moved, the
            # arms would not be taking the same transcendental route, one of them would be wrong.
            att = UP.Attribution(ho > 0 and de > 0,
                                 "expf route: glibc vs ROCm libdevice; host_vs_oracle=%d "
                                 "device_vs_oracle=%d arms_apart=%d" % (ho, de, nd))
            isolated = cen.diverging * 2 < fn
            print(cen.line(fname, extra="case=%d mode=%s differing_words_lane_rule=%d share=%.4f"
                                  % (c["case_index"], c["mode_name"], nd, share)))
            print("ARMDIFF case=%d mode=%s field=%s verdict=%s attribution=%s isolated=%d "
                  "strict_majority_identical=%d/%d"
                  % (c["case_index"], c["mode_name"], fname,
                     "counted-and-attributed" if att.ok and isolated else "REFUSED",
                     att.detail, 1 if isolated else 0, fn - cen.diverging, fn))
            if not att.ok:
                att_offenders.append("case %d field %s: %s" % (c["case_index"], fname, att.detail))
            if not isolated:
                iso_offenders.append("case %d field %s: %d of %d elements differ between the arms"
                                     % (c["case_index"], fname, cen.diverging, fn))
    outside = 0
    outside_cases = 0
    for c in cases:
        if c["field_exact"] == "gate_up":
            continue
        outside_cases += 1
        outside += n_diff(hostw, devw, c["dst_off"], c["n_floats"])
    nd_all = n_diff(hostw, devw)
    print("ARMDIFFSUM exact_fields=%d exact_words=%d counted_fields=%d counted_words=%d "
          "counted_differing=%d exact_differing=%d nan_payload_pairs=%d worst_diverging_share=%.4f (%s)"
          % (n_exact_fields, exact_words, n_counted_fields, counted_words, counted_div,
             len(exact_offenders), nan_payload, worst[0], worst[1] or "none"))
    print("ARMDIFFALL whole_buffer differing_words=%d of %d (the number D-16 was decided on; it is "
          "reported, not asserted -- the assertion is now per declared field)" % (nd_all, len(hostw)))
    v.check("cross-arm: every declared-exact field is bit-identical between the arms",
            not exact_offenders, exact_words,
            "; ".join(exact_offenders[:5]) if exact_offenders
            else "0 differing words in %d words of declared-exact fields" % exact_words)
    v.check("cross-arm: nothing outside the declared counted cases moves at all",
            outside == 0, outside, "%d differing words across %d cases outside the 18 "
            "field_exact=gate_up cases" % (outside, outside_cases))
    v.check("cross-arm: every accepted divergence is attributed by an independent third side",
            not att_offenders, n_counted_fields,
            "fields where only one arm differs from the pinned oracle: %s"
            % ("; ".join(att_offenders[:4]) if att_offenders else "none"))
    v.check("cross-arm: every accepted divergence is isolated, not a systematic shift",
            not iso_offenders, n_counted_fields,
            "fields where a majority of elements moved (that is a different function, not rounding): %s"
            % ("; ".join(iso_offenders[:4]) if iso_offenders else "none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--max-rows", type=int, default=1)
    ap.add_argument("--device", action="store_true", help="run the device arm (only via ./gate)")
    a = ap.parse_args()
    work = a.work if os.path.isabs(a.work) else os.path.join(HERE, a.work)
    v = V()
    geo = {"real": 0, "synth": 0, "max_rows_full": 2}

    if a.selftest:
        selftest(v)
        return v.verdict("host selftest")

    rc, out = run([PY, ORACLE, "--work", work, "--max-rows", a.max_rows],
                  os.path.join(work, "oracle.log"))
    if rc != 0:
        raise Fail("oracle_arm.py failed rc=%d; see work/oracle.log" % rc)
    for line in out.splitlines():
        print(line)
    cases = read_cases(os.path.join(work, "cases.tsv"))
    declared = parse_tag(out, "SAMPLE", ["cases", "expert_cases", "combine_cases"], want=1)[0]
    if int(declared["cases"]) != len(cases):
        raise Fail("oracle printed %s cases, cases.tsv holds %d" % (declared["cases"], len(cases)))
    rl = [l for l in out.splitlines() if l.startswith("BLOCK ")]
    geo["real"] = sum(int(kv(l)["real_rows"]) for l in rl)
    geo["synth"] = sum(int(kv(l)["main_n"]) - int(kv(l)["real_rows"]) for l in rl)
    modes = declare(cases, geo, a.max_rows, a.max_rows < geo["max_rows_full"])
    # Every mode in the header's enum must be exercised. The enum is anonymous (`enum { MG_MODE_... }`),
    # so the scrape takes the union of identifiers inside every `enum { ... }` block in the header
    # rather than scraping prose, which mentions the names in comments and made the check noise.
    src = open(os.path.join(HERE, "moe_gather_kernels.h")).read()
    modes_h = set()
    for body in re.findall(r"enum\s*[A-Za-z_]*\s*\{([^}]*)\}", src, re.S):
        modes_h |= set(re.findall(r"\bMG_MODE_[A-Z0-9_]+\b", body))
    modes_h.discard("MG_MODE_COUNT")
    if len(modes_h) < 5:
        raise Fail("the mode-enum scrape found %d names; a scrape that finds nothing cannot fail" % len(modes_h))
    v.check("every mode is exercised", modes_h <= modes, len(modes_h),
            "header=%d plan=%d missing=%s" % (len(modes_h), len(modes),
                                              sorted(modes_h - modes) or "none"))
    # the plan's own class counts, from two places: the oracle's line and cases.tsv
    for cls, n in (("MG_EXP_MATCH", 0), ("MG_EXP_MISMATCH", 0), ("MG_EXP_ALT", 0),
                   ("MG_EXP_IMMUNE", 0), ("MG_EXP_REPORT", 0)):
        n2 = sum(1 for c in cases if c["expect_name"] == cls)
        line = [l for l in out.splitlines() if l.startswith("CLASSES ")][0]
        n1 = int(kv(line)[cls])
        v.check("parsed==printed for %s" % cls, n1 == n2, n2, "oracle=%d cases.tsv=%d" % (n1, n2))
    # ...and the same for the declared exactness rule. A MATCH case may never be judged on less than
    # the whole span, or "exact" would stop meaning anything.
    fxline = [l for l in out.splitlines() if l.startswith("FIELDEXACT ")]
    if not fxline:
        raise Fail("the oracle printed no FIELDEXACT line: which fields are exact is undeclared")
    n_declared = int(kv(fxline[0])["gate_up"])
    n_declared_all = int(kv(fxline[0])["all"])
    n_gate_up = sum(1 for c in cases if c["field_exact"] == "gate_up")
    v.check("parsed==printed for field_exact", n_declared == n_gate_up
            and n_declared_all + n_gate_up == len(cases), len(cases),
            "oracle gate_up=%d all=%d; cases.tsv gate_up=%d of %d"
            % (n_declared, n_declared_all, n_gate_up, len(cases)))
    match_not_all = [c["case_index"] for c in cases
                     if c["expect_name"] == "MG_EXP_MATCH" and c["field_exact"] != "all"]
    v.check("every MG_EXP_MATCH case is declared fully exact", not match_not_all,
            sum(1 for c in cases if c["expect_name"] == "MG_EXP_MATCH"),
            "MATCH cases declaring gate_up: %s" % (match_not_all or "none"))

    nf = sum(c["n_floats"] for c in cases)
    env = os.environ
    hdr = open(os.path.join(HERE, "moe_gather_kernels.h")).read()
    env["MG_I"] = hdr.split("#define MG_I ")[1].split()[0]
    env["MG_H"] = hdr.split("#define MG_H ")[1].split()[0]
    argv_common = ["--plan", os.path.join(work, "plan.tsv"), "--slab", os.path.join(work, "slab.bin"),
                   "--expect", os.path.join(work, "expect.f32"),
                   "--out-floats", nf, "--expect-floats", nf,
                   "--expect-units", sum(1 for l in open(os.path.join(work, "plan.tsv"))
                                         if l.startswith("U")),
                   "--expect-rows", sum(1 for l in open(os.path.join(work, "plan.tsv"))
                                        if l.startswith("R")),
                   "--expect-cases", len(cases),
                   "--expect-blocks", sum(1 for l in open(os.path.join(work, "plan.tsv"))
                                          if l.startswith("B"))]
    rc, out = run([PROBE] + argv_common + ["--out", os.path.join(work, "host_out.f32"),
                                           "--report", os.path.join(work, "host_report.tsv")],
                  os.path.join(work, "host-probe.log"))
    if rc != 0:
        raise Fail("host_probe rc=%d: %s" % (rc, out.strip().splitlines()[-1] if out else ""))
    planrec = parse_tag(out, "PLAN", ["units", "rows", "cases", "blocks"], want=1)[0]
    v.check("host arm parsed==printed on the plan", all(int(v2.split("/")[0]) == int(v2.split("/")[1])
          for v2 in (planrec["units"], planrec["rows"], planrec["cases"], planrec["blocks"])),
          4, "units=%s rows=%s cases=%s blocks=%s" % (planrec["units"], planrec["rows"],
                                                     planrec["cases"], planrec["blocks"]))
    expw = read_words(os.path.join(work, "expect.f32"), nf)
    hostw = read_words(os.path.join(work, "host_out.f32"), nf)
    adjudicate(v, cases, hostw, expw, os.path.join(work, "host_report.tsv"), "host")

    if a.check_only:
        return v.verdict("plan check (no device)")

    rc, out = run([DRIVER] + argv_common + ["--out", os.path.join(work, "out.f32"),
                                            "--report", os.path.join(work, "report.tsv")],
                  os.path.join(work, "driver.log"))
    if rc != 0:
        raise Fail("device driver rc=%d: %s" % (rc, out.strip().splitlines()[-1] if out else ""))
    devw = read_words(os.path.join(work, "out.f32"), nf)
    adjudicate(v, cases, devw, expw, os.path.join(work, "report.tsv"), "device")
    # The cross-arm check, scoped by the declared field rule (D-16 option A). It replaced a whole-buffer
    # `nd == 0` check that spanned the fields this lane declares it does not band; see cross_arm().
    cross_arm(v, cases, hostw, devw, expw)
    lc = parse_tag(out, "LAUNCHES", ["mode", "issued", "expected", "device_counted"], want=1)[0]
    v.check("device launch counter == launches issued",
            int(lc["issued"]) == int(lc["device_counted"]), int(lc["issued"]),
            "mode=%s issued=%s device_counted=%s" % (lc["mode"], lc["issued"], lc["device_counted"]))
    hs = parse_tag(out, "HOST_SYNC", ["d2h_total", "d2h_during_launches"], want=1)[0]
    v.check("no device->host read during the launches", int(hs["d2h_during_launches"]) == 0, 1,
            "d2h_total=%s d2h_during=%s" % (hs["d2h_total"], hs["d2h_during_launches"]))
    rc2, out2 = run([DRIVER] + argv_common + ["--per-slot",
                                              "--out", os.path.join(work, "out_perslot.f32"),
                                              "--report", os.path.join(work, "report_perslot.tsv")],
                    os.path.join(work, "driver-perslot.log"))
    if rc2 != 0:
        raise Fail("device driver --per-slot rc=%d" % rc2)
    lc2 = parse_tag(out2, "LAUNCHES", ["mode", "issued", "device_counted"], want=1)[0]
    v.check("per-slot and grouped differ in launch count (prediction 2)",
            lc2["mode"] == "per-slot" and int(lc2["issued"]) != int(lc["issued"])
            and int(lc2["issued"]) == int(lc2["device_counted"]), 2,
            "grouped=%s per-slot=%s" % (lc["issued"], lc2["issued"]))
    pw = read_words(os.path.join(work, "out_perslot.f32"), nf)
    nd2 = n_diff(pw, devw)
    v.check("per-slot shape produces the same words (so the count is the only difference)",
            nd2 == 0, nf, "differing=%d" % nd2)
    return v.verdict("moe_gather device gate")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print("GATE FAIL %s" % e)
        sys.exit(1)
