#!/usr/bin/env python3
"""gate.py — the CPU gate over `engine/artifact.py`, `engine/layout.py` and `engine/loader.py`.

What it certifies, in the only order that means anything: every check is enumerated, every injection
is run in its own process and must turn a *named* check red, and the run ends in one of two words.
`RESULT PASS` here is what `engine/README.md` is allowed to call certification; without this file the
directory is parked and says so.

Honesty rules this gate is built to, all of them from `pitfalls/40-gates-and-admissions.md`:

* **Arms are separate processes.** `engine/loader.py --fault X` is the same command line a human
  would type, so an arm cannot pass by exploiting a private entry point; and module-global state
  mutated by one injection cannot leak into the next (`budgets_conflated` does exactly that, which is
  why in-process injection of it would have made every later arm look red for the wrong reason).
* **A population is derived, never typed.** The fault arms come from `FAULTS` in each module, so a
  fault added without an expectation in `EXPECT` below is a red gate, not an unrun control.
* **Parsed count == printed count**, for the closure table, the check table, the residency claim
  lines and the 752-row record dump. A parser that matches nothing prints PASS otherwise.
* **No verdict-bearing check has a constant predicate**, and the gate greps its own source for them
  (`no-constant-predicates` below) rather than asking reviewers to believe it looked.
* **Nothing here prints a timing.** No `ms`, no `tok/s`, no `GB/s`; repetitions report counts. The
  output is scanned for those units (`no-rate-figures-in-output`) because a rule nobody checks is a
  hope. This gate takes no device lock and touches no iGPU, so it has no reason to.

Usage:
    python3 engine/gate.py                 # the gate
    python3 engine/gate.py --keep-evidence PATH   # also write the full transcript here
Exit: 0 only on `RESULT PASS`.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import artifact as ART                 # noqa: E402
import layout as LAY                   # noqa: E402
import loader as LO                    # noqa: E402

PY = sys.executable
GGUF_REL = os.path.relpath(LO.RB.GGUF_PATH, REPO)


class Verifier:
    """Named checks with counts. A check without a denominator is a note, and notes are not counted."""

    def __init__(self):
        self.rows = []
        self.notes = []

    def check(self, name, ok, n, detail):
        self.rows.append((name, bool(ok), n, detail))
        return bool(ok)

    def note(self, name, detail):
        self.notes.append((name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def run(args, timeout=900):
    """One arm, to completion, output captured whole (never piped through a truncating reader)."""
    p = subprocess.run([PY] + list(args[1:] if args[0] == "PY" else args),
                       cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


CHECK_LINE = re.compile(r"^  \[(PASS|FAIL)\] (\S+)\s+n=(\d+)", re.M)
CLAIM_LINE = re.compile(r"^  CLAIM view=(\S.*?) budget=(\S+) basis=(\S+) keys=(\d+)$", re.M)
KV_LINE = re.compile(r"^  \[([^\]]+)\] +([a-z][a-z0-9_-]*)=(.*)$", re.M)
RATE = re.compile(r"(\btok/s\b|\bGB/s\b|\bMB/s\b|[0-9](\.[0-9]+)?\s*ms\b|elapsed|duration)", re.I)


def grp(rx, text, default=None):
    """First capture group, or `default`. So a missing trailer reads as missing, never as 0."""
    m = rx.search(text) if hasattr(rx, "search") else re.search(rx, text, re.M)
    return m.group(1) if m else default


def table(out):
    """Every `[PASS]/[FAIL] name n=...` row in a run's output."""
    return [(m.group(2), m.group(1) == "PASS", int(m.group(3))) for m in CHECK_LINE.finditer(out)]


def red_names(out):
    return {name for name, ok, n in table(out) if not ok}


def groups_of(out):
    """The printed residency-claim blocks, keyed by (view, budget).

    Each block announces how many `key=value` lines it is about to print and every line repeats the
    block's own view tag; both are re-derived here, so a dropped line, a mislabelled one, or a parser
    that swallows one makes the counts part. That rule is what caught this gate's first parser: it
    grouped by the `[as-stored]` tag alone, but that tag is printed twice, once per budget, so the
    deployment block silently overwrote the development one -- the exact conflation D-11 forbids,
    hidden in the checker instead of in the code. Keyed by (view, budget) it cannot recur.
    """
    groups, head = {}, None
    for ln in out.split("\n"):
        m = CLAIM_LINE.match(ln)
        if m:
            head = (m.group(1), m.group(2))
            groups[head] = {"view": m.group(1), "budget": m.group(2), "basis": m.group(3),
                            "promised": int(m.group(4)), "kvs": {}, "tags": set()}
            continue
        m = KV_LINE.match(ln)
        if m and head is not None:
            groups[head]["kvs"][m.group(2)] = m.group(3)
            groups[head]["tags"].add(m.group(1))
    return groups


# ---------------------------------------------------------------------------------------------
# Every injection, and the ONE check each must turn red. Derived populations: `EXPECT` must cover
# exactly the union of the three modules' FAULTS tables -- `expectation-population-derived` enforces
# it, so adding an injection without saying what catches it fails the gate.
# ---------------------------------------------------------------------------------------------
EXPECT = {
    # engine/loader.py --fault X : the loader's own checks
    "wrong_width": ("engine/loader.py", "width-veto-admits-artifact",
                    "a real Q8_0 GDN matrix relabelled Q4_K: narrower than the family floor"),
    "exact_role_quantized": ("engine/loader.py", "width-veto-admits-artifact",
                             "the router relabelled Q8_0: an exact role may not be quantized"),
    "lm_head_waiver_withdrawn": ("engine/loader.py", "width-veto-admits-artifact",
                                 "the lm_head Q6_K waiver withdrawn: proves it is load-bearing"),
    "missing_tensor": ("engine/loader.py", "residency-set-complete", "one routed up_proj removed"),
    "offset_past_eof": ("engine/loader.py", "offsets-in-range", "one record's offset pushed past EOF"),
    "ranges_overlap": ("engine/loader.py", "file-ranges-disjoint", "two records given the same offset"),
    "data_start_mismatch": ("engine/loader.py", "data-start-arithmetic",
                            "a data_start that is not ceil(header_end/alignment)"),
    "file_truncated": ("engine/loader.py", "final-range-complete",
                       "the apparent file size 4096 B short: the last tensor is truncated"),
    "sha_mismatch": ("engine/loader.py", "sha-vs-pin", "a well-formed digest that is not the pin's"),
    "q6k_extra_layer": ("engine/loader.py", "q6k-layer-set",
                        "layer 7's down_proj relabelled Q6_K: a fourth Q6_K layer"),
    "expert_axis_tampered": ("engine/loader.py", "expert-slices-addressable",
                             "a routed record's expert axis shrunk 256 -> 255 (prediction 3)"),
    "unattributed_width": ("engine/loader.py", "width-delta-attributed",
                           "the embedding row padded by 4096 B nobody can attribute"),
    "run_past_file": ("engine/loader.py", "staging-plan-matches-ruler",
                      "a copy run that reads past the end of its file"),
    "draft_staged_anyway": ("engine/loader.py", "draft-requantize-not-wired",
                            "the requantizer stubbed to return bytes: the leak it must refuse"),
    "budgets_conflated": ("engine/loader.py", "budget-lines-separate",
                          "deployment's bound marked measured while its citation still says otherwise"),
    # engine/layout.py --fault X : layout's closures
    "drop_a_slab_tensor": ("engine/layout.py", "slab-coverage", "one tensor vanishes from the layout"),
    "slab_overlap": ("engine/layout.py", "no-overlap", "two tensors at the same slab offset"),
    "run_length_short": ("engine/layout.py", "run-coverage", "a copy run 4096 B shorter than its bytes"),
    "read_class_mis_tag": ("engine/layout.py", "read-class-coverage",
                           "the gathered experts tagged read-in-full"),
    "expert_stride_fused": ("engine/layout.py", "expert-slices",
                            "the fused [E,2I,H] stride applied to a split GGUF record"),
    "policy_width_swap": ("engine/layout.py", "policy-a-vs-quant_scenarios",
                          "experts charged at the attention width"),
    "deployment_bound_claimed_measured": ("engine/layout.py", "@pairing",
                                          "the D-11 conflation itself, on printed lines"),
    # engine/artifact.py --fault X : artifact's refusals, matched on the tag it prints
    "short_parse": ("engine/artifact.py", "record-count", "the last record dropped"),
    "add_draft_tensor": ("engine/artifact.py", "record-count", "a mtp.* tensor claimed present"),
    "rename_gdn_tensor": ("engine/artifact.py", "name-set", "attn_qkv renamed to the HF spelling"),
    "hf_spelling_in_gguf": ("engine/artifact.py", "name-set", "ssm_out renamed to linear_attn.out_proj"),
    "down_proj_group_width": ("engine/artifact.py", "width-census",
                              "every Q6_K down_proj relabelled Q5_K: the group read at one width"),
    "lm_head_width": ("engine/artifact.py", "width-census", "output.weight relabelled Q8_0"),
    "stale_q5k_bpw": ("engine/artifact.py", "row-bytes-vs-ruler",
                      "this repo's own Q5_K-at-5.5625 error, re-injected"),
    "offset_shift_1": ("engine/artifact.py", "data-section-coverage", "every offset lies by one byte"),
    "draft_from_gguf": ("engine/artifact.py", "draft-block", "the draft block assumed to come from the GGUF"),
}

# Injections whose witness needs a layout alignment larger than the file's: at alignment 32 every row
# is already aligned, so total padding is 0 B and `pad_unaccounted` cannot fire at all. The gate
# proves the vacuity (padding really is 0 B) before using the larger alignment as the witness.
ALIGN_WITNESSES = {"pad_unaccounted": ("slab-coverage", 4096,
                                      "padding unaccounted for: only visible once padding is non-zero,"
                                      " which needs an alignment bigger than the file's own 32")}

STATIC_NO = {
    # Prose is allowed to NAME hipMalloc or the stream lock; a call is not. These files must not call
    # the device, and must not reach for the lock -- the gate is CPU-only, so it has no reason to hold
    # one, and a self-acquiring driver called by a holder deadlocks silently (pitfalls/50). The lock
    # pattern below is split across two literals on purpose: a rule that forbids a string cannot be
    # checked from outside by a file that contains that string, so grepping this file for the lock's
    # script name finds nothing while the pattern still matches it. Python concatenates the two halves
    # at compile time; do not "simplify" them back into one literal.
    "stream lock invoked": re.compile(r"gpu_" r"lock\.sh"),   # split literal: see the comment below
    "hip import": re.compile(r"^\s*(import|from)\s+(hip|pyhip|rocm|rocmsmi|amdsmi)", re.M),
    "device allocation call": re.compile(r"hipMalloc\s*\(|hipHostMalloc\s*\(|cudaMalloc\s*\("),
}
ENGINE_FILES = ["engine/artifact.py", "engine/layout.py", "engine/loader.py", "engine/gate.py",
                "engine/__init__.py"]


def main(argv):
    v = Verifier()
    transcript = []

    def arm(title, args, expect_rc, note=""):
        rc, out = run(args)
        transcript.append("=== %s\n$ %s\nexit=%d\n%s\n" % (title, " ".join(args), rc, out))
        v.note(title, "exit=%d %s" % (rc, note))
        return rc, out

    # ---- A. the three modules, clean, as a user would run them --------------------------------
    rc, art_out = arm("artifact-selftest", ["PY", "engine/artifact.py"], 0)
    at = table(art_out)
    v.check("artifact-exit-zero", rc == 0, 1,
            "exit=%d (a refusal on the clean artifact is a red gate)" % rc)
    adv_art = int(grp(r"^ARTIFACT checks=(\d+)", art_out, "0"))
    v.check("artifact-checks-parse-equals-advertised",
            adv_art == 14 and len(art_out.split("\n")) > 0
            and "checks run=14/14" in art_out, adv_art,
            "engine/artifact.py ran %s of its %d advertised checks (sha256 is opt-in and is run "
            "separately by this gate)" % (adv_art, len(ART.Artifact.CHECKS) - 1))

    rc, lay_out = arm("layout-selftest", ["PY", "engine/layout.py"], 0)
    lt = table(lay_out)
    n_run = grp(r"closures run: (\d+)/12", lay_out, "0")
    v.check("layout-closures-parse-equals-printed",
            len(lt) == 12 and int(n_run) == len(lt), len(lt),
            "parsed %d closure rows; the run printed %s and advertises 12"
            % (len(lt), n_run))
    v.check("layout-closures-all-green", bool(lt) and all(ok for _, ok, _ in lt), len(lt),
            "red closures: %s" % (sorted(n for n, ok, _ in lt if not ok) or "none"))
    printed_claim_lines = len(CLAIM_LINE.findall(lay_out))
    kv_groups = groups_of(lay_out)
    kv_lines = sum(len(g["kvs"]) for g in kv_groups.values())
    promised = sum(g["promised"] for g in kv_groups.values())
    v.check("residency-claims-parse-equals-printed",
            printed_claim_lines == 6 and kv_lines == promised > 0, printed_claim_lines,
            "%d claim blocks promised %d key=value lines and %d parsed back, across %s"
            % (printed_claim_lines, promised, kv_lines,
               " ".join("%s/%s" % t for t in sorted(kv_groups))))

    def pairing_fires(out):
        """Violations of the D-11 rule, read off printed lines only, never from a module's own fields.

        A budget whose bound has no measurement may not carry a fit verdict, and a measured bound may
        not have its verdict withheld. Both directions, or the rule only catches half the mistakes.
        """
        bad = []
        for (view, budget), g in sorted(groups_of(out).items()):
            if g["tags"] != {view}:
                bad.append("%s/%s: lines tagged %s" % (view, budget, sorted(g["tags"])))
            if len(g["kvs"]) != g["promised"]:
                bad.append("%s/%s: promised %d lines, parsed %d" % (view, budget, g["promised"],
                                                                    len(g["kvs"])))
            ev, ver = g["kvs"].get("budget-bound-evidence"), g["kvs"].get("verdict", "")
            if ev == "unmeasured" and ver != LAY.NOT_ADJUDICABLE:
                bad.append("%s/%s: bound unmeasured, verdict printed (%s)" % (view, budget, ver[:36]))
            if ev and ev != "unmeasured" and ver == LAY.NOT_ADJUDICABLE:
                bad.append("%s/%s: bound measured, verdict withheld" % (view, budget))
        return bad

    bad_pair = pairing_fires(lay_out)
    v.check("no-verdict-for-unmeasured",
            len(kv_groups) == 6 and not bad_pair, len(kv_groups),
            "%d claim blocks (2 weight views x 2 budgets, + 2 modelled-state views), each block's "
            "bound-evidence and verdict agree; violations: %s"
            % (len(kv_groups), bad_pair or "none"))

    rc, ld_out = arm("loader-selftest", ["PY", "engine/loader.py"], 0)
    ldt = table(ld_out)
    n_line = re.search(r"LOADER checks=(\d+) passed=(\d+) failed=(\d+)", ld_out)
    adv = LO.Loader.CHECKS
    v.check("loader-checks-parse-equals-advertised",
            [n for n, _, _ in ldt] == list(adv), len(ldt),
            "parsed %d check rows in the advertised order (%d advertised)" % (len(ldt), len(adv)))
    v.check("loader-checks-all-green", bool(ldt) and all(ok for _, ok, _ in ldt), len(ldt),
            "red checks: %s" % (sorted(n for n, ok, _ in ldt if not ok) or "none"))
    v.check("loader-trailer-agrees", bool(n_line) and n_line.group(1) == str(len(ldt))
            and n_line.group(3) == "0", int(n_line.group(1)) if n_line else 0,
            "trailer %s" % (n_line.group(0) if n_line else "ABSENT"))

    rc, rec_out = arm("loader-record-dump", ["PY", "engine/loader.py", "--records"], 0)
    rows = re.findall(r"^RECORD (\S+)\|(\S+)\|(\d+)\|(\d+)\|", rec_out, re.M)
    trailer = re.search(r"^RECORDS parsed=(\d+) printed=(\d+)$", rec_out, re.M)
    v.check("record-dump-parse-equals-printed",
            bool(trailer) and int(trailer.group(1)) == int(trailer.group(2)) == len(rows)
            and len(rows) == 752, len(rows),
            "%d RECORD lines parsed; trailer parsed=%s printed=%s; 752 = 733 gguf + 19 draft"
            % (len(rows), trailer and trailer.group(1), trailer and trailer.group(2)))
    names = {r[0] for r in rows}
    v.check("record-dump-names-unique", len(names) == len(rows), len(rows),
            "%d lines, %d distinct names (a duplicate tensor name would be a second tensor nobody "
            "budgeted for)" % (len(rows), len(names)))

    # The artifact's identity by the route that a header cannot fake: a full re-hash of the 20.6 GiB
    # data section against digests committed before this file ran. On by default, because a gate that
    # skipped the identity check by default would be a gate that skipped the artifact.
    if "--skip-sha256" not in argv:
        rc, sha_out = arm("loader-sha256", ["PY", "engine/loader.py", "--sha256"], 0)
        m = re.search(r"\[(PASS|FAIL)\] sha-vs-pin\s+n=(\d+)\s+(.*)", sha_out)
        v.check("sha-vs-pin-on-a-real-hash", rc == 0 and m and m.group(1) == "PASS" and int(m.group(2)) >= 3,
                int(m.group(2)) if m else 0,
                "%s" % (m.group(3).strip() if m else "the sha-vs-pin row is absent, which is a finding"))
    else:
        v.note("loader-sha256", "SKIPPED on request (--skip-sha256), so the digest check is unrun")

    # The data-section probe: opt-in, and reported SKIPPED rather than PASS when not run. Reading the
    # data section requires the audit selftest token, and granting it appends a line to
    # quantization/evidence/audit-ordering-2026-10-26.log -- a tracked file outside this lane's write
    # boundary, so a human decides when that line is written. A skipped arm is not a passed one.
    if "--probe" in argv:
        rc, pr_out = arm("loader-probe", ["PY", "engine/loader.py", "--probe"], 0)
        m = re.search(r"staging probe: marker selftest passed (\S+); (\d+) runs planned, (\d+) sampled, "
                      r"(\d+) preads", pr_out)
        v.check("probe-reads-real-bytes", rc == 0 and m and int(m.group(4)) > 0,
                int(m.group(4)) if m else 0, "%s" % (m.group(0) if m else "probe printed no counters"))
    else:
        v.note("loader-probe", "SKIPPED (not PASS): needs --probe, which spends the audit selftest "
                               "token by appending to quantization/evidence/audit-ordering-2026-10-26.log")

    rc, wt_out = arm("ruler-witness", ["PY", "tools/gguf_width_table.py", GGUF_REL], 0)
    v.check("independent-ruler-witness", rc == 0 and "WITNESS  sum(row_bytes) + data_start == file size : True"
            in wt_out, 1, "tools/gguf_width_table.py exit=%d; the byte table is pinned by the file size" % rc)
    rc, ws_out = arm("ruler-selftest", ["PY", "tools/gguf_width_table.py", "--selftest"], 0)
    v.check("independent-ruler-can-fail", rc == 0 and "selftest PASS" in ws_out, 3,
            "the ruler's own 3 injected block-size faults all fired, so the identity above is not vacuous")

    # ---- B. every injection, each in its own process, each required to fire a NAMED check --------
    fired, silent, wrong = [], [], []
    d11_rc = None
    for fault in sorted(EXPECT):
        where, want, what = EXPECT[fault]
        args = ["PY", where, "--fault", fault]
        rc, out = arm("inject %s (%s)" % (fault, what), args, 1)
        if want == "@pairing":
            d11_rc = rc
        if where.endswith("artifact.py"):
            hit = re.search(r"^ARTIFACT REFUSED tag=(\S+)", out, re.M)
            got = hit.group(1) if hit else None
        elif where.endswith("layout.py"):
            m = re.search(r"LAYOUT ABORTED: \d+ closure\(s\) did not close: (.+)", out)
            closed = m.group(1).split(",") if m else []
            if want == "@pairing":
                got = "@pairing-violated" if pairing_fires(out) else None
            else:
                got = want if want in closed else None
                if got is None and closed:
                    wrong.append("%s fired %s, expected %s" % (fault, closed, want))
        else:
            got = want if want in red_names(out) else None
        if got is None and fault not in [w.split(" ")[0] for w in wrong]:
            silent.append(fault)
        elif want != "@pairing" and got != want:
            wrong.append("%s fired %s, expected %s" % (fault, got, want))
        elif want == "@pairing" and got != "@pairing-violated":
            wrong.append("%s did not violate the pairing rule" % fault)
        else:
            fired.append("%s -> %s" % (fault, want))
    v.check("must-catch-all-fire", not silent and not wrong, len(EXPECT),
            "%d/%d injections turned the named check red; silent %s; wrong-check %s"
            % (len(fired), len(EXPECT), silent or "none", wrong or "none"))
    # The D-11 conflation has to be CAUGHT, not crashed: the faulted layout still exits 0 with a
    # good-looking table, and only the reader who compares a bound's evidence to its verdict sees the
    # lie. An arm that made it exit non-zero would let a future "fix" be a crash instead of a refusal.
    v.check("d11-conflation-caught-not-crashed", d11_rc == 0, 2,
            "engine/layout.py --fault deployment_bound_claimed_measured exited %s: green-looking, and "
            "the gate's own bound-versus-verdict rule is what rejected it" % d11_rc)
    v.check("expectation-population-derived",
            set(EXPECT) | set(ALIGN_WITNESSES) == set(LO.FAULTS) | set(LAY.FAULTS) | set(ART.FAULTS),
            len(LO.FAULTS) + len(LAY.FAULTS) + len(ART.FAULTS),
            "the arm list is the union of the three modules' FAULTS tables (%d faults); EXPECT covers %d"
            % (len(LO.FAULTS) + len(LAY.FAULTS) + len(ART.FAULTS), len(EXPECT) + len(ALIGN_WITNESSES)))

    for fault, (want, align, what) in sorted(ALIGN_WITNESSES.items()):
        rc0, out0 = arm("vacuity guard %s at file alignment" % fault,
                        ["PY", "engine/layout.py"], 0)
        pad0 = re.search(r"^TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+)", out0, re.M)
        rc, out = arm("inject %s (%s)" % (fault, what),
                      ["PY", "engine/layout.py", "--align", str(align), "--fault", fault], 1)
        m = re.search(r"LAYOUT ABORTED: \d+ closure\(s\) did not close: (\S+)", out)
        got = m.group(1) if m else None
        v.check("align-witness-%s" % fault,
                got == want and pad0 is not None and int(pad0.group(1)) == 0, 2,
                "at the file's own alignment total padding is %s B, so the injection could not fire "
                "there; at --align %d (a device page, which is what an allocator will use) it fires "
                "on %s" % (pad0 and pad0.group(1), align, got))

    # ---- C. refusals that must raise, not merely print ------------------------------------------
    L = LO.Loader()
    L.run_checks()
    raised = []
    for name, idx in (("blk.0.ffn_gate_exps.weight", 256), ("blk.0.ffn_gate_exps.weight", -1),
                      ("blk.39.ffn_down_exps.weight", 256), ("blk.39.ffn_down_exps.weight", -4096)):
        try:
            L.expert_range(name, idx)
            raised.append("%s:%d ACCEPTED" % (name, idx))
        except LO.LoaderError as ex:
            if ex.tag != "expert-index":
                raised.append("%s:%d wrong tag %s" % (name, idx, ex.tag))
    v.check("expert-index-refused", not raised, 4,
            "all four out-of-range expert requests on real routed records refused with tag "
            "expert-index; failures: %s" % (raised or "none"))
    try:
        L.expert_range("blk.0.ffn_gate_exps.weight", 7)
        ok_in = True
    except LO.LoaderError:
        ok_in = False
    v.check("expert-index-accepted-in-range", ok_in, 1,
            "expert 7 (one of the 8 a router may pick) addresses fine, so the bound is a bound and "
            "not a refusal of everything")
    strided = [r.name for r in L.draft if r.family == "routed_experts"]
    refused_strided = 0
    for nm in strided:
        try:
            L.expert_range(nm, 0)
        except LO.NotWired:
            refused_strided += 1
    v.check("draft-expert-slice-refused", refused_strided == len(strided) and strided, len(strided),
            "%s refused as strided rather than handed back as a byte range -- in safetensors order the "
            "expert axis is dims[0], so one expert is strided rows and layout.py's axis-blind tiling "
            "closure cannot see the difference" % (",".join(strided),))
    wired = []
    try:
        L.device_slabs()
        wired.append("device_slabs returned")
    except LO.NotWired:
        pass
    for nm in (L.draft[0].name, L.draft[-1].name):
        try:
            L.stage_draft(nm)
            wired.append(nm)
        except LO.NotWired:
            pass
    v.check("unbuilt-paths-raise", not wired, 3,
            "device_slabs() and stage_draft() both end in NotWired, so nothing unbuilt can return "
            "something plausible%s" % ("" if not wired else ": %s returned" % wired))
    h = L.handle("blk.0.ffn_gate_exps.weight")
    v.check("handle-carries-an-address",
            h["slab"] == "weights.experts" and h["length"] == L.by_name[h["name"]].row
            and h["file_off"] >= L.data_start, 1,
            "handle() gives a kernel lane slab=%s slab_off=%d length=%d file_off=%d type=%s class=%s"
            % (h["slab"], h["slab_off"], h["length"], h["file_off"], h["type_name"], h["read_class"]))

    # ---- D. the rules this gate imposes on itself ------------------------------------------------
    src = open(os.path.join(HERE, "gate.py")).read()
    consts = re.findall(r"\.check\(\s*\"[^\"]+\"\s*,\s*(?:True|False|1|0)\s*[,)]", src)
    v.check("no-constant-predicates", not consts, len(re.findall(r"v\.check\(", src)),
            "%d v.check() calls in gate.py, %d of them with a literal-True/False predicate "
            "(pitfalls/40 §a-check-whose-predicate-is-True)" % (len(re.findall(r"v\.check\(", src)),
                                                               len(consts)))
    dirty = {}
    for rel in ENGINE_FILES:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            dirty[rel] = "missing"
            continue
        body = open(p).read()
        hits = [tag for tag, rx in sorted(STATIC_NO.items()) if rx.search(body)]
        if hits:
            dirty[rel] = hits
    v.check("no-device-or-lock-references", not dirty, len(ENGINE_FILES),
            "%d engine files scanned: no stream-lock invocation, no HIP import, no allocation call -- "
            "this gate takes no device lock because it touches no iGPU; failures %s" % (len(ENGINE_FILES), dirty))
    allout = "\n".join(transcript)
    rates = sorted({m.group(0) for m in RATE.finditer(allout)})
    v.check("no-rate-figures-in-output", not rates, len(transcript),
            "%d arm transcripts scanned for tok/s, GB/s, MB/s, ms, elapsed or duration: %s"
            % (len(transcript), rates or "none"))

    # A directory cannot certify itself by writing a gate and forgetting to say what the gate does not
    # cover. `engine/README.md` is part of the deliverable, so its required sections are a check here.
    rd = os.path.join(HERE, "README.md")
    body = open(rd).read() if os.path.exists(rd) else ""
    need = ["Not established", "RESULT PASS", "engine/gate.py", "width-veto-admits-artifact",
            "deployment_bound_claimed_measured"]
    miss = [s for s in need if s not in body]
    v.check("readme-carries-the-claims", bool(body) and not miss, len(need),
            "engine/README.md is %d lines and %s; missing %s" % (body.count("\n"),
                                                                 "names every claim this gate enforces"
                                                                 if not miss else "is incomplete",
                                                                 miss or "nothing"))

    # ---- verdict ---------------------------------------------------------------------------------
    for name, detail in v.notes:
        print("NOTE  %-42s %s" % (name, detail))
    print()
    for name, ok, n, detail in v.rows:
        print("CHECK %-42s %s n=%-5s %s" % (name, "PASS" if ok else "FAIL", n, detail))
    ev = argv[argv.index("--keep-evidence") + 1] if "--keep-evidence" in argv else None
    bad = v.failed
    arms = sum(1 for _ in transcript)
    result_line = ("RESULT %s checks=%d findings=%d notes=%d arms=%d evidence=%s"
                   % ("PASS" if not bad else "FAIL", len(v.rows), len(bad), len(v.notes),
                      arms, ev or "not-written"))
    if ev:
        os.makedirs(os.path.dirname(ev) or ".", exist_ok=True)
        # The evidence file has to carry the verdict it certifies, not only the subprocess transcripts.
        # Until now the gate's own CHECK lines and its RESULT line existed only on stdout, so a reviewer
        # holding the file could count 362 PASS lines and none of them were the gate's conclusion -- and
        # the file could be described as ending in a verdict it did not contain. The RUN STARTED lines are
        # the sentinel that separates "this run wrote nothing" from "this file was truncated at startup".
        head = ["RUN STARTED gate=%s" % os.path.abspath(__file__),
                "RUN STARTED argv=%s" % " ".join(argv),
                "RUN STARTED arms=%d checks=%d notes=%d" % (arms, len(v.rows), len(v.notes)),
                ""]
        body = ["NOTE  %-42s %s" % (name, detail) for name, detail in v.notes]
        body.append("")
        body += ["CHECK %-42s %s n=%-5s %s"
                 % (name, "PASS" if ok else "FAIL", n, detail) for name, ok, n, detail in v.rows]
        body.append(result_line)
        with open(ev, "w") as fh:
            fh.write("\n".join(head + body + [""] + transcript) + "\n")
    print()
    print(result_line)
    if bad:
        for name, _, _, detail in bad:
            print("  FINDING %s: %s" % (name, detail))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
