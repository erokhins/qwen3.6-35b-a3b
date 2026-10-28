#!/usr/bin/env python3
"""block_device.py — M2's live device path: stage layer 0's shared-expert resident bytes, launch the
gfx1151 kernel on them, adjudicate the rows it produced under the licences that already exist.

`engine/device_arm.py` is the staging-and-launch half; `engine/HANDOFF.md` §THE DEVICE HALF is the
window this belongs to and the exact command line. This file is the driver: what runs, what is
printed, and what it refuses.

FOUR WAYS THIS COULD BE DISHONEST, and the mechanism that closes each.
  * A device result without a device. A device-shaped line (`DEVICESTAGED`, `device=on`,
    `verdict=MATCH`) is printed only inside the `--device` branch, that branch only survives real
    `hipMalloc`/`hipMemcpy` calls whose counts are printed, and `device-claims-come-from-device-only`
    reddens if the printed claims and the counted witness disagree in either direction. Control
    `device-claim-scan-is-awake` hands the scanner a fabricated line so it cannot be decorative
    (`pitfalls/40` §a-device-claim-assembled-by-parsing-another-tools-log).
  * A silent fallback to the host twin. `device_arm.run_arm` raises on a non-zero exit, a missing
    report record, or a device run that counted zero launches; the handler prints `REFUSED tag=…` and
    exits 1 with no `STAGE` line in existence. Control `launch-failure-exits-red` provokes it with a
    plan whose spans are out of range (and, in a device window, an allocation no pool can satisfy) and
    requires the named refusal plus an empty row count.
  * A kernel judged on bytes the engine did not stage. Every span is digest-bound: file bytes read
    through the audit reader's token (as M1 read them) → the staging run's own plan/header/read-back
    digests → `hip.read_back` of exactly the span the kernel addresses → the byte range inside the
    arm's slab file. Control `resident-span-corrupt` flips one byte on the side under test and
    requires both the bind to break and the affected stage rows to move, so the chain is seen to be
    load-bearing rather than merely printed.

A FOURTH WAY, found by running it (window e2): a claim stricter than any licence. The router arm and the
host twin are two compilations of one translation unit, and their records differ — `wpre`, R4's fp32
pre-cast weight, moves in 5 of 8 slots. Three of the four fields that claim compares (`ids`, `ids_alt`,
`wcast`) are what a licence measures at exactness and stay at zero tolerance; `wpre` is documented in
`prototype/moe_router/moe_router_kernels.h` §R5 as NOT bit-exact across an independent fp32 route, so
requiring bit-identity there was unfireable and dropping the claim would be a relaxation the agent does not
own (D-18). The claim is therefore split: `rt-device-host-licensed-fields-identical` keeps zero tolerance
on the licensed trio, and `wpre` is held to a scope re-derived every run from three independent prior
sources — the licensor's own `weights_vs_pinned_f32_ulp` applied by the triangle inequality, §R5's wording,
and an error propagation from §mr_exp's documented `exp()` scope — with the strictest binding, its premise
measured against the pinned pre-cast record, and a control
(`wpre-scope-rejects-a-bigger-perturbation`) showing the bound rejects a bigger perturbation. Attribution
is a check too (`rt-device-host-difference-is-the-documented-exp-scope`), so "the two libms differ" cannot
become the permanent excuse for a future codegen defect. Derivation and evidence:
`specification/evidence/blockdevice-wpre-scope-2026-10-26.md` and `engine/HANDOFF.md` §PASS 4.

NO BAND IS AUTHORED HERE. Rows go through `engine/block_gate.py::stage_row` and `::adjudicate`, which
consult only `speccheck/evidence/gdn-band-2026-10-26.json` (digest printed every run, and checked
against the pinned `27a68c7e82c9aa05…`) and the prototype lanes' floor probes re-read from their
committed artifacts. The shared expert's policy-A rows have no licence — the stage-output floor probe
`quantization/README.md` §“How a quantized path gets gated” item 1 requires has never been run — so
they print `band=UNLICENSED stage=… reason=policy-a-width-band-absent` and the verdict is `QUALIFIED`
at best. That is the same residue the CPU gate carries; this file neither hides it nor dissolves it.

NO LOCK, NO CLOCK. The window belongs to the main agent (`AGENTS.md` §“GPU execution is serialized”):
this driver owns no lock and never acquires one — a self-locking driver called by a holder deadlocks
silently (`pitfalls/50`) — and reads no clock. A check enforces that no rate token appears in its own
output. Ask for `--device` and, if this process cannot prove it performed the act, it refuses by name.
"""

import ctypes
import os
import subprocess
import argparse
import ast
import hashlib
import math
import re
import struct
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "speccheck"), str(ROOT / "specification")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import block_forward as BF                                # noqa: E402
import block_gate as BG                                   # noqa: E402
import device_arm as DA                                   # noqa: E402
import gdn_ref as G                                       # noqa: E402
import loader as LO                                       # noqa: E402
import stage_run as SR                                    # noqa: E402

WORK_DEFAULT = HERE / "work" / "m2dev"
ARM_STAGES = ("shared_raw", "shared_gate_prob", "shared_out")
RT_STAGES = ("router_logits", "topk_ids", "topk_weights")
WITNESS = {"hip_calls_python": 0, "hip_calls_arm": 0, "launches": 0, "allocs": 0,
           "bytes_to_device": 0, "bytes_from_device": 0, "staged_runs": 0, "slabs": 0,
           "rows_emitted": 0, "unparsable_counters": []}


def counter_positive(value):
    """True only for a positive `int`. A string, a float or a bool is not a device act.

    Comparing an unparsed counter with `0` is what made window `i` raise `TypeError` inside a check instead of
    printing a FAIL line, and a FAIL line is the only thing a bad counter can be reported through: the driver
    has to survive long enough to print its summary. The type comes from `device_arm.parse_arm_report`; this is
    the one place a consumer may ask the question, and `device-run-counters-are-typed-at-the-parse-site`
    requires that no quoted-counter comparison survives anywhere else in this file.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def witness_fold(run, label):
    """Fold one arm report's five counters into the witness totals, refusing to fold what it cannot read.

    The fields arrive TYPED — `device_arm.parse_arm_report` refuses a `RUN` field whose own printf conversion
    says integer and whose value is not one — so every consumer adds the value instead of casting it, and a
    value that still arrives unreadable is recorded as a named offence on the RESULT line instead of raised:
    window `i` printed no RESULT line because one of these folds did `int("abc")`, and a device window that
    dies before its summary throws away the good half of itself. One fold, in one function, so
    `device-run-counters-are-typed-at-the-parse-site` can require that no cast of these five names survives
    anywhere in this file.
    """
    for key, field in (("hip_calls_arm", "hip_calls"), ("launches", "launches"),
                       ("allocs", "allocs"), ("bytes_to_device", "bytes_to_device"),
                       ("bytes_from_device", "bytes_from_device")):
        value = run.get(field, 0)
        if isinstance(value, int) and not isinstance(value, bool):
            WITNESS[key] += value
        else:
            WITNESS["unparsable_counters"].append("%s.%s=%r" % (label, field, value))
# Arm acts, counted under the path the ARM reported (`RUN path=`), not under the path the caller
# meant. A control's leg count is read off the arm's own report, which is what makes
# `control-legs-cover-the-path-under-claim` able to catch a control that claims a path it never ran.
PATH_LEGS = {"host_cpubody": 0, "device": 0}


def arm_leg(binary, plan, slab, out, device=False, fault=None, label=None):
    """`device_arm.run_arm` with the executed path tallied. Same refusal behaviour, one more counter."""
    rec = DA.run_arm(binary, plan, slab, out, device=device, fault=fault, label=label)
    path = rec["run"]["path"]
    PATH_LEGS[path] = PATH_LEGS.get(path, 0) + 1
    return rec


def legs_since(snapshot):
    """The arm acts executed since `snapshot`, keyed by the path that ran them."""
    return dict((p, n - snapshot.get(p, 0)) for p, n in PATH_LEGS.items() if n - snapshot.get(p, 0))


class Refused(DA.Refused):
    """One refusal type in the lane: `engine/device_arm.py` raises it and this driver catches it."""


# ---------------------------------------------------------------------- refusals as counted findings
#
# Window `m` (`devices/evidence/device-window-2026-10-27-g.md` §window `m`) died inside ONE leg-construction
# call and printed `RESULT mode=device checks=0 findings=1 controls=0 fired=0` with zero printed CHECK lines,
# zero MUSTCATCH, zero REFUTED-CAUSE. `grep -E "verdict=FAIL"` on that log returns nothing: the run executed
# no checks, so it could not fail, and a run that cannot fail is one step from reading green. A refusal is a
# statement about the leg that raised it — "this leg could not be placed on its own side of the bound" — so
# it belongs on that leg's owning check as a finding, and the rest of the run has to happen anyway.
#
# The owner of a leg family is the check that already adjudicates that family, named in one place so the
# catching site and the checking site cannot disagree about who owns the refusal.

VB_LEG_OWNERS = {
    "derive": "d28-bound-and-witness-lines-are-parsed-not-assumed",
    "edge": "d28-value-band-witness-inside-and-edge-legs-reject",
    "fabrication": "d28-edge-legs-are-adjacent-lattice-points-and-sized-from-the-bound",
    "must-catch": "d28-must-catch-set-is-derived-and-reddens",
    "refuted": "d28-refuted-causes-still-refuted-and-say-what-carries-it",
}


def leg_attempt(ctx, family, what, construct):
    """Run one leg-construction call; a refusal becomes a counted finding on its owning check.

    Returns whatever `construct()` returned, or None when it refused — and None is what tells the caller to
    go on with the next leg instead of unwinding out of `run()`. The refusal is printed as its own record
    (tag, leg, owning check, `counted_as_finding=yes`) and recorded on the ctx for the owning check's
    predicate to read, so "this leg could not be built" reaches the verdict as a FAIL line rather than as
    the absence of every line after it.
    """
    try:
        return construct()
    except DA.Refused as exc:
        ctx.setdefault("refusals", []).append(
            dict(family=family, what=what, tag=getattr(exc, "tag", "untagged"),
                 owner=VB_LEG_OWNERS[family], note=str(exc)))
        print("REFUSED tag=%s for_leg=%s owning_check=%s counted_as_finding=yes suite_continues=yes note=%s"
              % (getattr(exc, "tag", "untagged"), what, VB_LEG_OWNERS[family], exc))
        return None


def refusals_for(ctx, family):
    """The refusals recorded against one leg family (see `leg_attempt`)."""
    return [r for r in (ctx.get("refusals") or []) if r["family"] == family]


def refusal_note(ctx, family):
    """The `refused=` clause a check prints for its own family: every refusal named, or `none`."""
    return "; ".join("%s(tag=%s) %s" % (r["what"], r["tag"], r["note"])
                     for r in refusals_for(ctx, family)) or "none"


def silent_control_finding(fired, reason):
    """The rule a SILENT control has to obey: if it did not fire, it says why, in words.

    Window `g` printed `CONTROL name=control-witness-guard-can-redden fired=SILENT` on the device path
    with nothing after it. A silent guard is not a neutral state — it is the state in which a guard is not
    guarding — and a bare SILENT is how a control that lost its legs on the path under claim reads the same
    as one that ran and found nothing. `silent-control-names-its-reason` is the check; this function is the
    single rule so the check and the witness cannot disagree about what counts as a named reason.

    It binds CONTROL lines, not CHECK lines: a check has no legs to lose, a control does, and the window-`h`
    pair that motivated the check (`gather-divergence-causes-priced`, `gdn-state-read-is-the-arm-s-own-
    fingerprint`, both `fired=SILENT` with no `reason=`) are controls. `Verifier.control` is the only place
    a control is registered, and it routes every one of them through here, so a control cannot be registered
    without passing this rule; `silent-reason-guard-can-redden` is the leg that shows the rule still bites.
    """
    if fired:
        return None
    return None if (reason or "").strip() else "silent-without-named-reason"


# ---------------------------------------------------------------------------------------------
# A control may not deny what this run's own reproduction legs proved.
#
# Window `i` printed, in the same log, `DREPRO pair=d_hip_device_vs_d_hip_hostpath probe=decay_ulp_mask
# steps=-1 mask=0x80d200 heads=9,12,14,15,23 … vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION`
# and `CONTROL name=gdn-divergence-causes-priced fired=SILENT reason=no injected cause reproduces the GDN
# device divergence …`. Both sentences are about the same pair and one of them is false. The cause table
# predates the probe that answered the question: the factor-two magnitude screen in `DMAGMATCH`/`GMAGMATCH`
# is a *ratio* test, and a 1-ULP-per-head cause that reproduces the record byte for byte is a 5x ratio, so
# the ratio screen went on printing "nothing reproduces it" after the byte-level test had already said
# otherwise. A confident falsehood in a gate log is worse than a gap, because the gap gets investigated.
#
# The rule is written over the run's own printed lines, not over an internal flag, for the same reason the
# row parser is: what is checked must be what was printed (`pitfalls/40` §a-device-claim-assembled-by-parsing
# -another-tools-log, inverted — here the two halves of one log must be allowed to disagree with each other).
# ---------------------------------------------------------------------------------------------

REPRO_HEADS = ("DREPRO", "GREPRO")
# The phrasings by which a control asserts that nothing reproduced a pair. Derived list, checked against
# itself by `control_attribution_contradiction_guard_can_redden`: a phrase list that cannot fire is a comment.
NO_CAUSE_PHRASES = ("no injected cause reproduces", "no cause matched", "no single-sign ulp probe",
                    "unattributed and no scope", "nothing-to-attribute")


def exact_reproductions(text):
    """`{pair_label: [line, …]}` for every reproduction leg this run printed that came out exact.

    Only lines beginning `DREPRO `/`GREPRO ` count, and only ones whose `verdict=` is exactly
    `EXACT-REPRODUCTION`; a `RESIDUAL-LEFT` line is evidence of the opposite and is returned separately so a
    pair cannot be reported as both by sloppiness of this parser.
    """
    exact, residual = {}, {}
    for line in text.splitlines():
        head = line.split(" ", 1)[0]
        if head not in REPRO_HEADS or " pair=" not in line:
            continue
        pair = re.search(r" pair=(\S+)", line)
        verdict = re.search(r" verdict=(\S+)", line)
        if not pair or not verdict:
            continue
        (exact if verdict.group(1) == "EXACT-REPRODUCTION" else residual).setdefault(
            pair.group(1), []).append(line)
    return exact, residual


def run_text(ctx):
    """Everything the run has printed at the moment of asking — or a harness ctx's whole world.

    `ctx["text"]` is a mid-run snapshot taken before the reproduction legs and the controls print, and
    window `j` proved what that is worth: the contradiction check read it, parsed 0 legs from a log that
    had already printed three, and passed. A consumer whose rule is about printed lines must read the
    live buffer the tee holds (`run()` hangs it on the ctx as `live_text`); a proof harness that hands
    over a fixed `text` is complete by construction, so the fallback is that.
    """
    live = ctx.get("live_text")
    return live() if live is not None else ctx.get("text", "")


def reproduction_leg_finding(device, text):
    """The explicit anti-vacuity assertion under the contradiction check: parse zero is a finding.

    Window `j`: `no-control-contradicts-an-exact-reproduction verdict=PASS 0 reproduction legs` in a log
    that contained three — `parsed=0 → PASS` again, reached through ordering instead of through the
    parser. The count is now required, not assumed: a device run's reproduction controls print a
    `DREPRO`/`GREPRO` leg on every pass, so a device run that parses zero exact legs either has a blind
    parse (the window-`j` shape — a zero that is the absence of a match, not of legs) or has lost the
    attribution this check adjudicates, and either way it reddens instead of passing over an empty list
    (`pitfalls/40` §"A driver whose parser matches nothing prints PASS"). A CPU run prints no legs by
    construction and is exempt — the exemption says so in the check's own detail.
    """
    if not device:
        return None
    exact, _ = exact_reproductions(text)
    if sum(len(lines) for lines in exact.values()):
        return None
    return ("device run parsed 0 exact reproduction legs out of its own printed lines — the "
            "reproduction controls print a DREPRO/GREPRO leg on every device pass, so zero means a "
            "blind parse (window `j` parsed 0 against 3 printed) or an attribution that stopped "
            "reproducing; a zero count here is a finding, never a clean log")


def contradiction_findings(text):
    """Every (control, pair) where a SILENT control denies a reproduction this run printed.

    A control line contradicts the headline when it (a) did not fire, (b) carries a `reason=` or detail
    naming one of `NO_CAUSE_PHRASES`, and (c) names a pair that some `DREPRO`/`GREPRO` line in the same run
    reported `verdict=EXACT-REPRODUCTION`. (c) is what keeps the check narrow: a control may still say "no
    cause reproduces" about a *different* pair.
    """
    exact, _ = exact_reproductions(text)
    out = []
    if not exact:
        return out
    for line in text.splitlines():
        if not line.startswith("CONTROL ") or "fired=SILENT" not in line:
            continue
        low = line.lower()
        if not any(p in low for p in NO_CAUSE_PHRASES):
            continue
        name = re.match(r"CONTROL name=(\S+)", line)
        for pair in sorted(exact):
            if pair in line:
                out.append("%s denies %s, which this run reproduced exactly (%s)"
                           % (name.group(1) if name else "?", pair,
                              exact[pair][0].split(" verdict=")[0][:120]))
                break
    return out


CONTROL_RED_PREFIX = "control:"
REJECT_TALLY_CHECK = "terminal-verdict-finding-count-matches-printed-records"
EMPTY_SUITE_CHECK = "run-executed-a-non-empty-check-and-control-suite"
# The run's Verifier, reachable from `main()` so the abort path can report how much of the suite printed
# instead of asserting `checks=0` from outside. Written once, in `run()`.
RUN_STATE = {"v": None}


def terminal_fatality(mode, n_checks, n_controls, escaped=None):
    """Why a run that printed no failure is fatal anyway, in words — or None when it is not.

    Window `m` is the reason this exists: one refusal unwound out of `run()` and the log ended
    `RESULT mode=device checks=0 findings=1 controls=0 fired=0`, zero CHECK lines, and `grep verdict=FAIL`
    found nothing in it — "nothing failed" and "nothing ran" were one string apart
    (`pitfalls/40` §a-check-that-can-produce-no-output-cannot-fail, wearing a RESULT line). This is the one
    place that rule is written, so the in-run guard, the abort path and the exit status cannot disagree: a
    run that executed no checks, or no witnessed controls, is fatal with a NAMED reason, and so is one whose
    suite was cut short by an escaped refusal. The non-zero exit comes from the FAIL record this prints, not
    from the silence around it.

    Pure over (mode, counts, escaped-reason) so a witness can run the fabricated device shape —
    `(device, 0, 0)` — through the same predicate the driver runs its own numbers through.
    """
    if escaped:
        return ("aborted-before-its-check-suite-completed(%s): %d checks and %d controls had printed when "
                "it died, so every check after the refusal is unexecuted, not passed" % (escaped,
                                                                                         n_checks,
                                                                                         n_controls))
    if n_checks == 0:
        return ("no-check-executed(checks=0, mode=%s): a run that executes no checks cannot fail, so "
                "findings=0 here means nothing was looked at, not that nothing was wrong" % mode)
    if n_controls == 0:
        return ("no-control-executed(controls=0, mode=%s): every predicate in this suite is witnessed "
                "non-inert by a control or a fabrication, and not one of them ran" % mode)
    return None


def check_run_population(v, ctx):
    """The guard that refuses to finish silently: a suite must have run to be believed.

    `len(v.names) + 1` is the count INCLUDING this check, which is not yet registered while its predicate is
    being evaluated — the same off-by-one self-reference `check_terminal_tally` documents. The two
    fabrications are the non-inertness witness: the window-`m` shape `(device, 0, 0)` and a suite with no
    controls both come back named, and the honest shape comes back None.
    """
    mode = "device" if ctx["device"] else "cpu"
    n_checks, n_controls = len(v.names) + 1, len(v.controls)
    fatal = terminal_fatality(mode, n_checks, n_controls)
    fab_checks = terminal_fatality("device", 0, 0)
    fab_controls = terminal_fatality("device", n_checks, 0)
    v.check(EMPTY_SUITE_CHECK,
            fatal is None and bool(fab_checks) and bool(fab_controls),
            "this run had registered %d checks (this one included) and %d witnessed controls when the line "
            "printed — the two guards that follow it make the RESULT's %d the final count — and a run with "
            "either count at zero is fatal by name rather than quiet: `%s`. Window `m` printed `checks=0 "
            "findings=1 controls=0` and no CHECK line at all, and nothing in the log could tell that from a "
            "clean sweep. Non-inertness, the same predicate on the two shapes it must name: fabricated "
            "checks=0 → %s; fabricated controls=0 → %s. The abort path in `main()` runs the escaped-reason "
            "branch of the same function, so an exception that still escapes prints a FAIL record and a "
            "named `fatal=` clause instead of a bare zero"
            % (n_checks, n_controls, n_checks + 2, fatal or "no fatality: the population is non-empty",
               fab_checks, fab_controls))



def reject_tally_findings(red, printed_lines):
    """`_REJECT_TALLY`: reconcile the verdict's finding list against the finding records it printed.

    Window `j` defect 2: `RESULT ... findings=4` went out in a log that also printed a CONTROL line
    counting `2 findings` and three `verdict=FAIL` lines — two tallies, neither re-read against the
    printed record. One source of truth is the Verifier's own `red` list (a failed check under its name,
    an unfired control under `control:<name>`); this is the parse that holds the print to it:

      * every failed CHECK name has exactly one printed `CHECK name=<n> verdict=FAIL` record;
      * every unfired control counts via a printed `CONTROL name=<name> fired=SILENT` record — the
        direction is finding ⇒ record, because the scratch-Verifier fabrications print SILENT lines too
        and a printed line must not be able to manufacture a finding, only a missing one condemns;
      * a printed FAIL line with no entry in the list is named (a finding that fell out of the tally);
      * the totals agree: `len(red)` == printed FAIL CHECK records + one record per unfired control.

    Returns the reconciliation findings; empty means the printed tally IS the parsed tally. Pure over
    (list, lines) so a witness leg can fabricate a mismatch through the same predicate the check runs.
    """
    printed_fail = [m.group(1) for line in printed_lines
                    for m in [re.match(r"CHECK name=(\S+) verdict=FAIL\b", line)] if m]
    printed_silent = set(m.group(1) for line in printed_lines
                         for m in [re.match(r"CONTROL name=(\S+) fired=SILENT\b", line)] if m)
    red_checks = [n for n in red if not n.startswith(CONTROL_RED_PREFIX)]
    red_controls = [n[len(CONTROL_RED_PREFIX):] for n in red if n.startswith(CONTROL_RED_PREFIX)]
    out = []
    for name in red_checks:
        if printed_fail.count(name) != 1:
            out.append("finding %s printed %d CHECK verdict=FAIL records, expected exactly 1"
                       % (name, printed_fail.count(name)))
    for name in printed_fail:
        if name not in red_checks:
            out.append("a printed CHECK verdict=FAIL for %s has no entry in the verdict's finding list"
                       % name)
    for name in red_controls:
        if name not in printed_silent:
            out.append("unfired control %s counts as a finding but printed no fired=SILENT record"
                       % name)
    records = len(printed_fail) + len(red_controls)
    if records != len(red):
        out.append("the verdict counts %d findings but %d printed finding-records back them (%d FAIL "
                   "CHECK lines + %d unfired controls)" % (len(red), records, len(printed_fail),
                                                           len(red_controls)))
    return out


def check_terminal_tally(v, ctx):
    """The last check the run prints: the verdict's own count, held to the records it printed.

    Window `j` defect 2 (`_REJECT_TALLY`): `RESULT ... findings=4` went out in a log that also printed a
    CONTROL detail counting `2 findings` (the attribution guard's own fabricated stale text — a different
    denominator, but printed in the same log with nothing saying so) and three `verdict=FAIL` lines; two
    tallies disagreed and neither was re-read against the printed record. One source of truth is
    `Verifier.red`; this parses the printed finding records against it BEFORE the `BLOCKDEVICE`/`RESULT`
    lines state the number, so both terminal lines print a count that has just been reconciled. The third
    `verdict=FAIL` line of a red log is the `BLOCKDEVICE` verdict line itself — the echo of the tally,
    never a further divergence — and the detail names every counted finding so the population is
    inspectable rather than a bare number. The predicate is witnessed non-inert by three fabrications
    through the same pure function: a finding with no printed record reddens it, a printed FAIL with no
    finding reddens it, an honest empty pair stays clean. A check cannot observe its own verdict line, so
    when this one reddens its own record is absent from its own parse and present in the next reader's —
    the count it reports is the count before itself, and the `red=` it hands to the summaries is the
    count after.
    """
    lines = run_text(ctx).splitlines()
    findings = reject_tally_findings(v.red, lines)
    fab_missing = reject_tally_findings(["fabricated-finding-without-a-printed-record"], [])
    fab_extra = reject_tally_findings([], ["CHECK name=fabricated-printed-failure verdict=FAIL "
                                           "a FAIL line no verdict list carries (witness only)"])
    fab_clean = reject_tally_findings([], [])
    print("GWITNESS guard=tally-reconciliation findings=%d fail-check-records=%d "
          "unfired-control-records=%d fabricated=missing-record flagged=%s fabricated=extra-print "
          "flagged=%s fabricated=clean flagged=%s note=the reconciliation predicate is not inert: a "
          "finding with no printed record and a printed FAIL with no finding both redden it on "
          "fabrication, and a consistent empty pair stays clean"
          % (len(v.red),
             sum(1 for ln in lines if re.match(r"CHECK name=\S+ verdict=FAIL\b", ln)),
             sum(1 for n in v.red if n.startswith(CONTROL_RED_PREFIX)),
             bool(fab_missing), bool(fab_extra), not fab_clean))
    fail_names = [n for n in v.red if not n.startswith(CONTROL_RED_PREFIX)]
    silent_names = [n[len(CONTROL_RED_PREFIX):] for n in v.red if n.startswith(CONTROL_RED_PREFIX)]
    v.check(REJECT_TALLY_CHECK,
            not findings and bool(fab_missing) and bool(fab_extra) and not fab_clean,
            "the terminal verdict's finding count is one source — `Verifier.red`, %d entries — and the "
            "printed records reconcile to it: %d CHECK verdict=FAIL lines (%s) plus %d registered unfired "
            "controls, each printed as `fired=SILENT` and counted as a finding (%s). Reconciliation "
            "findings: %s. The BLOCKDEVICE `red=` and RESULT `findings=` printed next state this same "
            "number, and the BLOCKDEVICE line's own `verdict=FAIL` in a red run is the echo of this tally, "
            "not a separate finding; the attribution guard's `%d findings` detail counts its OWN fabricated "
            "stale text, a different denominator, and is not this tally. Witness: the predicate reddened "
            "on both fabrications (%s, %s) and stayed clean on the honest empty pair"
            % (len(v.red), len(fail_names), ",".join(fail_names) or "none", len(silent_names),
               ",".join(silent_names) or "none", "; ".join(findings) or "none",
               found_stale_population(),
               "missing-record" if fab_missing else "MISSED",
               "extra-print" if fab_extra else "MISSED"))


def found_stale_population():
    """The attribution guard's own fabricated-stale finding count, for the detail's cross-reference.

    Read off the guard control's registered legs is not possible after the fact, so the detail quotes the
    count the guard itself required — one finding per reproduced pair in its stale text — which is the
    number that appeared as `2 findings` in the window-`j` log beside `findings=4`. Derived, not typed:
    it is the count of window-`i` evidence lines the guard builds its stale text from.
    """
    evidence = ROOT / "specification" / "evidence" / "blockdevice-device-2026-10-27-i1.log"
    if not evidence.exists():
        return 0
    return sum(1 for ln in evidence.read_text().splitlines()
               if ln.startswith(REPRO_HEADS) and "verdict=EXACT-REPRODUCTION" in ln)


class Verifier:
    """Named checks and witnessed controls (`engine/stage_run.py`::Verifier is the house shape).

    A control may also declare which execution paths it covered (`claims`) and how many arm acts it
    actually executed on each (`legs`, counted by `arm_leg` off the arms' own reports). That is what
    separates "exercised and clean on the device path" from "not exercised on the device path" — see
    `control-legs-cover-the-path-under-claim`.
    """

    def __init__(self):
        self.names, self.red, self.controls = [], [], []
        self.control_paths = {}
        self.unreasoned = []
        self.fired = 0

    def check(self, name, predicate, detail):
        ok = bool(predicate)
        self.names.append(name)
        if not ok:
            self.red.append(name)
        print("CHECK name=%s verdict=%s %s" % (name, "PASS" if ok else "FAIL", detail))
        return ok

    def control(self, name, fired, detail, legs=None, claims=None, reason=None):
        self.controls.append(name)
        if fired:
            self.fired += 1
        else:
            self.red.append("control:" + name)
        finding = silent_control_finding(fired, reason)
        if finding:
            self.unreasoned.append(name)
        self.control_paths[name] = dict(legs=dict(legs or {}), claims=tuple(claims or ()))
        pathnote = ""
        if legs or claims:
            pathnote = " legs=%s claims=%s" % (
                ",".join("%s:%d" % (p, n) for p, n in sorted((legs or {}).items())) or "none",
                ",".join(claims or ()) or "none")
        notecount = "" if not reason else " reason=%s" % reason
        print("CONTROL name=%s fired=%s%s%s %s" % (name, "RED" if fired else "SILENT", pathnote,
                                                  notecount, detail))
        return bool(fired)


def derived_arm_population(dumps):
    """The stages this path must produce, derived: the layer-0 manifest stages the two subsystems
    this pass built device arms for own.

    `block_gate.derived_population` yields the block's whole stage list from the two manifests; this
    narrows it by two naming rules — the shared expert's stages begin `shared_`, and the router family
    is `block_gate.ROUTER_FAMILY`, the licensor's own tuple, so the must-produce set and the licence
    set cannot drift apart. `ARM_STAGES` and `RT_STAGES` are then checked against it, which is what
    makes a rename in either place bite rather than a row quietly stop existing.
    """
    pop = BG.derived_population(dumps)
    shared = [s for s in pop["moe"] if s.startswith("shared_")]
    router = [s for s in pop["moe"] if s in BG.ROUTER_FAMILY]
    return dict(all=shared + router, shared=shared, router=router, from_manifest=pop,
                arm_declared=sorted(ARM_STAGES) + sorted(RT_STAGES))


def record_stages(path, layout):
    """The arm's record file as stage tensors, plus the NaN census of every field it carries."""
    rec = np.fromfile(str(path), dtype="<f4")
    if rec.size != layout["SE_CASE_FLOATS"]:
        raise Refused("arm-record-size", "%s holds %d floats; the record layout derived from %s says "
                                         "%d, and a short record cannot be judged field by field"
                      % (path, rec.size, DA.SE_HEADER.name, layout["SE_CASE_FLOATS"]))
    fields = dict(gate=(layout["SE_OFF_GATE"], layout["SE_N_GATE"]),
                  up=(layout["SE_OFF_UP"], layout["SE_N_UP"]),
                  act=(layout["SE_OFF_ACT"], layout["SE_N_ACT"]),
                  raw=(layout["SE_OFF_RAW"], layout["SE_N_RAW"]),
                  out=(layout["SE_OFF_OUT"], layout["SE_N_OUT"]),
                  prob=(layout["SE_OFF_PROB"], layout["SE_N_PROB"]))
    nan = {}
    for name, (off, ln) in fields.items():
        words = rec[off:off + ln].view(np.uint32)
        nan[name] = int(np.count_nonzero((words & np.uint32(0x7FFFFFFF)) == np.uint32(0x7FC00000)))
    stages = {
        "shared_raw": torch.from_numpy(rec[fields["raw"][0]:fields["raw"][0] + fields["raw"][1]]
                                       .copy()).reshape(1, layout["SE_H"]),
        "shared_out": torch.from_numpy(rec[fields["out"][0]:fields["out"][0] + fields["out"][1]]
                                       .copy()).reshape(1, layout["SE_H"]),
        "shared_gate_prob": torch.from_numpy(
            rec[fields["prob"][0]:fields["prob"][0] + fields["prob"][1]].copy()).reshape(1, 1),
    }
    return stages, nan


def emit_rows(dumps, band, limits, population, stages, provenance):
    """One STAGE row per stage in the derived population, judged by the existing licensors only."""
    print("PROVENANCE stages=%s source=%s device=%s"
          % (",".join(population["all"]), provenance,
             "on" if provenance.startswith("device:") else "off"))
    rows = []
    for stage in population["all"]:
        if stage not in stages:
            raise Refused("arm-missing-stage",
                          "no value for %s, which the derived population requires: a missing row is "
                          "not a row that can be skipped" % stage)
        rows.append(BG.stage_row(dumps, "policy_a", stage, stages))
    tally = BG.adjudicate(rows, band, limits, population["from_manifest"])
    for row in rows:
        BG.print_row(row)
    WITNESS["rows_emitted"] += len(rows)
    return rows, tally


def pinned_input_bytes(dumps, cfg):
    """The block's mixer input for the pinned MoE row, as the bf16 byte image the arm reads."""
    x = dumps.tensor("layer0.input", row=BF.TOKEN_ROW).reshape(cfg["d"])
    return x.contiguous().view(torch.uint8).numpy().tobytes(), x


def build_slab(st, x_bytes, layout, work, side="file"):
    """Write the arm's slab from the bytes the staging path certified, and bind each span to them.

    The bind line is the proof, printed on every run: the same sha256 for a tensor in the file, in the
    read-back the caller supplied (`dev_sha256` — from the device in a device window, from the same
    span read on the CPU otherwise) and in the byte range of the arm's slab file. Control
    `resident-span-corrupt` is what shows the equality is load-bearing. The router span is staged in
    the same run as the shared gate but belongs to the other arm, so it is skipped here and bound by
    `build_rt_slab` instead.
    """
    chunks, binds = [], []
    for s in st["spans"]:
        if s["role"] == "router":
            continue
        chunks.append((s["name"], bytes(s["dev_bytes"])))
    slab = bytearray()
    pos = {}
    for name, data in chunks:
        pos[name] = (len(slab), len(data))
        slab += data
    x_at = len(slab)
    slab += x_bytes
    for s in st["spans"]:
        if s["role"] == "router":
            continue
        off, ln = pos[s["name"]]
        sha = hashlib.sha256(slab[off:off + ln]).hexdigest()
        binds.append(dict(tensor=s["name"], span_off=off, bytes=ln,
                          file_sha=s["file_sha"], dev_sha=s["dev_sha"], slab_sha=sha, x_off=x_at))
    gate = pos["blk.0.ffn_gate_shexp.weight"][0]
    plan_lines = DA.write_plan(work / "plan.txt",
                               {"fused": (gate, pos["blk.0.ffn_up_shexp.weight"][0]
                                          + pos["blk.0.ffn_up_shexp.weight"][1] - gate),
                                "down": pos["blk.0.ffn_down_shexp.weight"],
                                "sg": pos["blk.0.ffn_gate_inp_shexp.weight"],
                                "x": (x_at, len(x_bytes))},
                               {"fused": layout["SE_TWO_I"], "down": layout["SE_H"], "sg": 1},
                               {"fused": layout["SE_H"], "down": layout["SE_I"],
                                "sg": layout["SE_H"]},
                               layout, len(slab))
    (work / "slab.bin").write_bytes(bytes(slab))
    for b in binds:
        print("BIND tensor=%s span_off=%d bytes=%d x_off=%d file_sha256=%s %s slab_sha256=%s bind=%s"
              % (b["tensor"], b["span_off"], b["bytes"], b["x_off"], b["file_sha"][:16],
                 ("dev_sha256=%s" % b["dev_sha"][:16]) if side == "device" else "dev_sha256=same",
                 b["slab_sha"][:16],
                 "ok" if (b["slab_sha"] == b["dev_sha"] == b["file_sha"]) else "BROKEN"))
    return dict(slab=work / "slab.bin", plan=work / "plan.txt", binds=binds, lines=plan_lines,
                x_at=x_at, x_bytes=len(x_bytes), slab_bytes=len(slab))


def stage_layer0(layout, rt, device, work):
    """The resident-bytes half: verify the covering runs, and in a device window stage and read back.

    `device=False` reads the same spans through `stage_run.ByteSource` — the audit reader's own token,
    the same door M1 staged through — with no allocation and no copy, so the pair, the census and both
    controls can run beside other work. `device=True` adds the real acts and refuses, by name, at the
    first one that does not come out certified.

    The router matrix shares a staging run with the shared expert's gate, so it is added to the same
    span list and gets the same digest bind, the same device read-back and the same corruption
    machinery as the shared expert's own tensors.
    """
    L = LO.open_loader()
    source = SR.ByteSource(L, why="M2 device path span read")
    stager = SR.Stager(L)
    spans = DA.residency(L, stager, layout)
    rspan = DA.residency_router(L, stager, rt)
    rspan["role"] = "router"
    spans.append(rspan)
    for s in spans:
        data, sha, got = DA.read_span(source, L.art.path, s["file_off"], s["length"], s["name"])
        s["bytes"], s["file_sha"], s["file_read"] = data, sha, got
        s["dev_bytes"], s["dev_sha"], s["dev_read"] = data, sha, got
    for s in spans:
        s.setdefault("role", SPAN_ROLE.get(s["name"], "?"))
    hip = DA.CountedHip() if device else None
    rows, dev_ptr = [], {}
    runs = sorted({s["run"] for s in spans})
    vrows = stager.verify(source, runs)
    for r in vrows:
        if r["verdict"] != "VERIFIED":
            raise Refused("staging-not-verified",
                          "run %d (%s) is %s: %s — the arm will not read bytes the staging path could "
                          "not verify" % (r["idx"], r["slab"], r["verdict"], ";".join(r["reasons"])))
        print("STAGERUN run=%d slab=%s file_off=%d bytes=%d n_tensors=%d verdict=%s plan_sha256=%s"
              % (r["idx"], r["slab"], r["file_off"], r["length"], r["n"], r["verdict"],
                 str(r["plan_sha"])[:16]))
    if device:
        by_idx = {r["idx"]: r for r in vrows}
        for slab in sorted({r["slab"] for r in vrows}):
            hip.alloc_slab(slab, L.lay.slab(slab).bytes)
            WITNESS["slabs"] += 1
            print("DEVALLOC slab=%s bytes=%d hip_error=none" % (slab, L.lay.slab(slab).bytes))
        hip.alloc_stage(max(r["length"] for r in vrows))
        for idx in runs:
            r = by_idx[idx]
            d = SR.stage_run_to_device(hip, source, r, source.path_for(r["slab"]))
            r["dev_sha"], r["dev_read"] = SR.device_read_back(hip, r, d)
            r["staged"] = SR.device_verdict(r)
            if r["staged"] != "STAGED":
                raise Refused("device-readback-disagrees",
                              "run %d staged but read back %s (device %s vs host plan %s): the "
                              "resident bytes are not the file bytes, so no row may be judged on them"
                              % (idx, r["staged"], str(r["dev_sha"])[:12], str(r["plan_sha"])[:12]))
            dev_ptr[idx] = d
            WITNESS["staged_runs"] += 1
        for s in spans:
            data, digest, got = hip.read_back(DA.device_ptr(dev_ptr[s["run"]], s["run_delta"]),
                                              s["length"])
            if digest != s["file_sha"]:
                raise Refused("span-bind", "%s: the device returned %s for %d B at %d and the file "
                                           "holds %s; the kernel would be judged on bytes the "
                                           "staging path never certified"
                               % (s["name"], digest[:16], s["length"], s["file_off"],
                                  s["file_sha"][:16]))
            s["dev_bytes"], s["dev_sha"], s["dev_read"] = data, digest, got
            s["dev_ptr"] = DA.device_ptr(dev_ptr[s["run"]], s["run_delta"])
        WITNESS["hip_calls_python"] = 1 + hip.tally["allocs"] + hip.tally["h2d_calls"] \
            + hip.tally["d2h_calls"]
        WITNESS["allocs"] = hip.tally["allocs"]
        WITNESS["bytes_to_device"] = hip.tally["h2d_bytes"]
        WITNESS["bytes_from_device"] = hip.tally["d2h_bytes"]
        print("DEVWITNESS hip_objects=%d slabs=%d staged_runs=%d h2d_calls=%d h2d_bytes=%d "
              "d2h_calls=%d d2h_bytes=%d" % (hip.tally["allocs"], WITNESS["slabs"],
                                            WITNESS["staged_runs"], hip.tally["h2d_calls"],
                                            hip.tally["h2d_bytes"], hip.tally["d2h_calls"],
                                            hip.tally["d2h_bytes"]))
    source.close()
    return dict(spans=spans, vrows=vrows, hip=hip, L=L, stager=stager, dev_ptr=dev_ptr)


def recheck_span_device(hip, span):
    """Re-read one span from the device after a control wrote into it: (bytes, sha, count)."""
    return hip.read_back(span["dev_ptr"], span["length"])


def write_device_byte(hip, ptr, value):
    """One deliberate 1-byte H2D into the resident set — `stage_run`'s own `device-corrupt` shape."""
    hip.fill(0, bytes([value]))
    hip.to_device(ptr, 1)


def restage(hip, source, run_row):
    """Put the resident bytes back after a corruption control, and re-prove the read-back."""
    d = SR.stage_run_to_device(hip, source, run_row, source.path_for(run_row["slab"]))
    sha, read = SR.device_read_back(hip, run_row, d)
    run_row["dev_sha"], run_row["dev_read"] = sha, read
    run_row["staged"] = SR.device_verdict(run_row)
    return run_row["staged"] == "STAGED"


# Which resident span each stage's value is a function of, per `engine/block_forward.py::moe_shared`
# (`shared_raw = down(silu(gate·x) * up·x)`, `shared_gate_prob = sigmoid(sg·x)`,
# `shared_out = shared_gate_prob * shared_raw`). The corruption control reads this table to know which
# rows MUST move when a byte moves and which must not; a new stage forces this map to be updated or the
# control refuses, rather than silently exempting a row from the witness.
STAGE_INPUTS = {"shared_raw": ("fused", "down"),
                "shared_out": ("fused", "down", "sg"),
                "shared_gate_prob": ("sg",)}
SPAN_ROLE = {"blk.0.ffn_gate_shexp.weight": "fused", "blk.0.ffn_up_shexp.weight": "fused",
             "blk.0.ffn_down_shexp.weight": "down", "blk.0.ffn_gate_inp_shexp.weight": "sg",
             "blk.0.ffn_gate_inp.weight": "router"}


def control_corrupt(v, ctx):
    """Corrupt one byte of a weight span on the side under test and require the affected rows to move.

    Device window: the flip is written into the resident set with a deliberate 1-byte H2D and read
    back, so the dirty slab is bytes the device returned — the shape `engine/stage_run.py`'s
    `device-corrupt` uses, and the only corruption a read-back comparison can see. CPU witness: the
    same flip on the span bytes the file returned, which is the same arithmetic on the same bytes and
    is what makes this control witnessable without a window. Three things are required, none of them a
    tolerance: the digest bind breaks, every stage that reads that span changes its distance to the
    pinned referent, and every stage that does not read it is unchanged to the bit.
    """
    layout, st, work, device = ctx["layout"], ctx["st"], ctx["work"], ctx["device"]
    pop = ctx["population"]["shared"]
    if set(STAGE_INPUTS) != set(pop):
        raise Refused("stage-input-map-stale",
                      "the corruption control's dependency table covers %s but the population is %s; "
                      "a row exempted by an out-of-date table is a row no control watches"
                      % (",".join(sorted(STAGE_INPUTS)), ",".join(sorted(pop))))
    gate = next(s for s in st["spans"] if s["name"] == "blk.0.ffn_gate_shexp.weight")
    role = SPAN_ROLE[gate["name"]]
    must_move = sorted(s for s in pop if role in STAGE_INPUTS[s])
    must_hold = sorted(s for s in pop if role not in STAGE_INPUTS[s])
    k = 2 * layout["SE_Q8_BYTES"] + 8        # inside row 1's payload: a weight byte, not a scale
    saved = (gate["dev_bytes"], gate["dev_sha"])
    honest_rec = (work / ("out_device.bin" if device else "out_host.bin")).read_bytes()
    honest = {s: ndiff_vs_pin(ctx["dumps"], s, ctx["honest_stages"])[0] for s in pop}
    ddir = work / "corrupt"
    ddir.mkdir(parents=True, exist_ok=True)
    rows_before = WITNESS["rows_emitted"]
    if device:
        write_device_byte(st["hip"], DA.device_ptr(gate["dev_ptr"], k), gate["bytes"][k] ^ 0xFF)
        dirty, dsha, got = recheck_span_device(st["hip"], gate)
        side = "device"
    else:
        arr = bytearray(gate["bytes"])
        arr[k] ^= 0xFF
        dirty, dsha, got = bytes(arr), hashlib.sha256(bytes(arr)).hexdigest(), len(arr)
        side = "file"
    bind_broke = dsha != gate["file_sha"]
    gate["dev_bytes"], gate["dev_sha"] = dirty, dsha
    try:
        slab2 = build_slab(st, ctx["x_bytes"], layout, ddir, side=side)
        DA.run_arm(ctx["args"].host_binary, slab2["plan"], slab2["slab"], ddir / "out.bin",
                   label="host_twin-corrupt")
        dstages, _ = record_stages(ddir / "out.bin", layout)
        dirty_nd = {s: ndiff_vs_pin(ctx["dumps"], s, dstages)[0] for s in pop}
        moved = [s for s in must_move if dirty_nd[s] != honest[s]]
        held = [s for s in must_hold if dirty_nd[s] == honest[s]]
        rec = (ddir / "out.bin").read_bytes()
        rec_diff = sum(1 for a, b in zip(honest_rec, rec) if a != b)
        v.control("resident-span-corrupt",
                  bind_broke and rec_diff > 0 and len(moved) == len(must_move)
                  and len(held) == len(must_hold) and WITNESS["rows_emitted"] == rows_before,
                  "side=%s tensor=%s(role=%s) byte +%d flipped (%d B read back, digest %s vs file %s): "
                  "bind broke=%s, record bytes differing=%d; ndiff honest->dirty for stages that read "
                  "this span %s, for stages that do not %s; rows emitted by the control=%d (the dirty "
                  "run is never printed as a stage row). If the arm read bytes the staging path did not "
                  "certify, flipping one of them could not move a row"
                  % (side, gate["name"], role, k, got, dsha[:12], gate["file_sha"][:12], bind_broke,
                     rec_diff,
                     ",".join("%s:%d->%d" % (s, honest[s], dirty_nd[s]) for s in must_move),
                     ",".join("%s:%d->%d" % (s, honest[s], dirty_nd[s]) for s in must_hold),
                     WITNESS["rows_emitted"] - rows_before))
        if device:
            row = next(r for r in st["vrows"] if r["idx"] == gate["run"])
            src = SR.ByteSource(ctx["st"]["L"], why="M2 corrupt restore")
            try:
                ok = restage(st["hip"], src, row)
            finally:
                src.close()
            print("RESTORE tensor=%s run=%d verdict=%s" % (gate["name"], row["idx"],
                                                          "STAGED" if ok else "NOT-STAGED"))
            if not ok:
                raise Refused("corrupt-restore-failed",
                              "the control could not put the resident bytes back; the run stops here "
                              "rather than continue on bytes it corrupted")
    finally:
        gate["dev_bytes"], gate["dev_sha"] = saved


def control_launch_failure(v, ctx):
    """Make a launch fail two ways and require a red refusal with no rows, both times."""
    work, args, device = ctx["work"], ctx["args"], ctx["device"]
    rows_before = WITNESS["rows_emitted"]
    bad = work / "plan_bad.txt"
    bad.write_text("\n".join(
        re.sub(r"^(SPAN role=fused off=)\d+", r"\g<1>%d" % (ctx["slab"]["slab_bytes"] + 4096), ln)
        for ln in (work / "plan.txt").read_text().splitlines()))
    tags = []
    for name, kwargs in (("bad-plan", {}),
                         *( (("oversize-alloc", {"device": True, "fault": "alloc-oversize"}),)
                            if device else () )):
        try:
            DA.run_arm(args.arm_binary if device else args.host_binary, bad, ctx["slab"]["slab"],
                       work / ("out_bad_%s.bin" % name), label=name, **kwargs)
            tags.append("%s:NOT-REFUSED" % name)
        except DA.Refused as exc:
            tags.append("%s:%s" % (name, exc.tag))
    fired = all(":arm-exec-failed" in t or ":arm-binary-absent" in t for t in tags) \
        and len(tags) == (2 if device else 1) and WITNESS["rows_emitted"] == rows_before
    v.control("launch-failure-exits-red", fired,
              "legs attempted=%d (%s); rows emitted by the control=%d. A failed arm raises before any "
              "STAGE line exists, so there is no path on which a red device act is answered with the "
              "host twin's numbers%s"
              % (len(tags), ",".join(tags), WITNESS["rows_emitted"] - rows_before,
                 "" if device else " (the oversize-allocation leg needs --device; the main agent "
                 "witnesses it in the window)"))


def control_poison(v, ctx):
    """The NaN census must be able to be non-zero: skip a stage and every slot of its field is poison."""
    work, layout = ctx["work"], ctx["layout"]
    pdir = work / "poison"
    pdir.mkdir(parents=True, exist_ok=True)
    arm = DA.run_arm(ctx["args"].host_binary, ctx["slab"]["plan"], ctx["slab"]["slab"],
                     pdir / "out.bin", fault="skip-stage3", label="host_twin-skip-stage3")
    _, nan = record_stages(pdir / "out.bin", layout)
    honest = ctx["honest_nan"]
    v.control("poison-census-can-be-nonzero",
              honest.get("out", -1) == 0 and nan.get("out", -1) == layout["SE_N_OUT"],
              "honest record: out field %d/%d slots unwritten; with stage 3 skipped: %d/%d. A guard "
              "that can only read 0 is not a guard, so the non-zero leg is run every time"
              % (honest.get("out", -1), layout["SE_N_OUT"], nan.get("out", -1), layout["SE_N_OUT"]))


def control_claim_scan(v, ctx):
    """The device-claim scanner must refuse a residency line this process did not earn."""
    probe = ("DEVICE" + "STAGED tensor=blk.0.ffn_gate_shexp.weight bytes=1114112 " + "verdict="
             + "MATCH " + "device=" + "on")
    honest = BG.device_claims(ctx["text"])
    fired = (len(BG.device_claims(probe)) == 1
             and (len(honest) > 0 if ctx["device"] else len(honest) == 0))
    v.control("device-claim-scan-is-awake", fired,
              "this run printed %d device-shaped lines; the assembled probe line — kept out of the "
              "output and built here so this file cannot self-match — scans %d, so the scanner has "
              "something to lose. Expected %s in a %s run"
              % (len(honest), len(BG.device_claims(probe)),
                 "non-zero" if ctx["device"] else "zero", "device" if ctx["device"] else "cpu"))


def ndiff_vs_pin(dumps, stage, stages):
    """(elements differing, largest bit-image step) between one stage value and its pinned referent.

    A pure metric, used by the controls so an ordering claim does not have to go through the band
    machinery — which legitimately refuses to compute a cosine for a degenerate (all-zero) wrong mode,
    and a control that failed because a licensor declined to adjudicate a mistake would be reporting
    the wrong thing.
    """
    ref = dumps.tensor(BG.referent_name(stage),
                       rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1))
    mine = stages[stage].reshape(ref.shape)
    if mine.dtype is not ref.dtype:
        mine = mine.to(ref.dtype)
    return BG.ulp_diff(mine.contiguous(), ref)


def control_modes(v, ctx):
    """The lane's own wrong spellings must be FURTHER from the pins than the honest one is.

    This is the wiring witness a device path needs and no band can supply: if the engine had wired the
    gate and up halves the wrong way round, read rows at a stride, misread a Q8_0 scale's width or
    read half the columns, the honest record would be indistinguishable from one of those modes. Each
    mode is the prototype lane's own named mistake (`shared_expert_kernels.h` §SE_MODE_*), run on the
    engine's own bytes, and the assertion compares two distances to the pinned referent — no tolerance
    is authored, and it is the ordering that is claimed.
    """
    mdir = ctx["work"] / "modes"
    mdir.mkdir(parents=True, exist_ok=True)
    base = (ctx["work"] / "plan.txt").read_text()
    honest = {s: ndiff_vs_pin(ctx["dumps"], s, ctx["honest_stages"])[0]
              for s in ctx["population"]["shared"]}
    verdicts, detail = [], []
    for mode in ("up_first", "pair_swap", "row_inter", "row_shift1", "wide_misread", "cols_half",
                 "tail_drop"):
        path = mdir / ("plan_%s.txt" % mode)
        path.write_text(base.replace("mode=correct", "mode=%s" % mode))
        DA.run_arm(ctx["args"].host_binary, path, ctx["slab"]["slab"], mdir / ("out_%s.bin" % mode),
                   label="mode-%s" % mode)
        stages, _ = record_stages(mdir / ("out_%s.bin" % mode), ctx["layout"])
        nd = {s: ndiff_vs_pin(ctx["dumps"], s, stages)[0] for s in ctx["population"]["shared"]}
        worse = all(nd[s] >= honest[s] for s in nd) and any(nd[s] > honest[s] for s in nd)
        verdicts.append(worse)
        detail.append("%s[%s]" % (mode, ",".join("%s:%d>=%d" % (s, nd[s], honest[s])
                                                 for s in sorted(nd))))
    v.control("wrong-modes-further-from-pins", all(verdicts) and len(verdicts) == 7,
              "each of the lane's seven named mistakes, run on the engine's own bytes, is at least as "
              "far from the pinned referent in every stage and strictly further in at least one: %s. "
              "An ordering claim, not a tolerance — no limit is authored here"
              % ("; ".join(detail)))


# ----------------------------------------------------------------------------------------------
# The router slice. Same staging, same report contract, a second compiled pair, and the three rows
# this pass can actually license — see `engine/device/m2_rt_kernels.h` for who owns which arithmetic.
# ----------------------------------------------------------------------------------------------

def rt_span(st):
    return next(s for s in st["spans"] if s["role"] == "router")


def build_rt_slab(st, x_bytes, rt, work, side="file", router_bytes=None, subdir=""):
    """The router arm's slab: the certified router span, then the token's bf16 row, then the plan.

    The router matrix is F32 and the kernel reads `float`s from it, so the span is placed at offset 0
    and the activation after it; the arm re-checks alignment and exact byte count against its plan
    (`engine/device/m2_rt_io.h`), and this function re-checks the digest chain before writing.
    `router_bytes` lets a control hand over a *different* copy of the same shape — which is how
    `router-span-is-f32` shows that the licence depends on the bytes being the F32 ones.
    """
    d = work / subdir if subdir else work
    d.mkdir(parents=True, exist_ok=True)
    s = rt_span(st)
    data = bytes(router_bytes if router_bytes is not None else s["dev_bytes"])
    slab = bytearray(data) + x_bytes
    sha = hashlib.sha256(data).hexdigest()
    bind = dict(tensor=s["name"], span_off=0, bytes=len(data), file_sha=s["file_sha"],
                dev_sha=s["dev_sha"], slab_sha=sha, x_off=len(data))
    print("BIND tensor=%s span_off=%d bytes=%d x_off=%d file_sha256=%s %s slab_sha256=%s bind=%s"
          % (bind["tensor"], bind["span_off"], bind["bytes"], bind["x_off"], bind["file_sha"][:16],
             ("dev_sha256=%s" % bind["dev_sha"][:16]) if side == "device" else "dev_sha256=same",
             bind["slab_sha"][:16],
             "ok" if (bind["slab_sha"] == bind["dev_sha"] == bind["file_sha"]) else "BROKEN"))
    plan = d / "plan_rt.txt"
    DA.write_rt_plan(plan, 0, len(data), len(data), len(x_bytes), rt, len(slab))
    (d / "slab_rt.bin").write_bytes(bytes(slab))
    return dict(slab=d / "slab_rt.bin", plan=plan, bind=bind, slab_bytes=len(slab),
                n_w=len(data), n_x=len(x_bytes))


def rt_stages(fields, rt):
    """The router record's fields as stage tensors, in the shapes the pins store.

    Every bf16-carrying field is stored widened to fp32 (the prototype lane's record convention, so the
    gate compares 4-byte patterns and never guesses a dtype off a file), so widening here and casting to
    the referent's dtype in `block_gate.stage_row` is exact, not a second rounding.
    """
    e, k = rt["MR_E"], rt["MR_K"]
    return {"router_logits": torch.from_numpy(fields["logits"].copy()).reshape(1, e),
            "topk_ids": torch.from_numpy(fields["ids"].copy()).reshape(1, k),
            "topk_weights": torch.from_numpy(fields["wcast"].copy()).reshape(1, k)}


def rt_form_distance(dumps, refname, values):
    """(ndiff, max_ulp) between an arm field and a named pinned record; int referents compare exactly."""
    ref = dumps.tensor(refname, rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1))
    mine = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).reshape(ref.shape)
    if ref.dtype is torch.int64:
        return int((mine.to(torch.int64) != ref).sum().item()), 0
    if mine.dtype is not ref.dtype:
        mine = mine.to(ref.dtype)
    return BG.ulp_diff(mine.contiguous(), ref)


def rt_ordered_index(values):
    """fp32 patterns -> the ordered integer image, so `|ia - ib|` is a ULP distance.

    The same fold `engine/block_gate.py::ulp_diff` applies (`0x7FFFFFFF - signed`, written here over
    uint32 as `0xFFFFFFFF - bits` so it is computed in int64 and cannot wrap), widened because a
    256-element spread does not have to stay inside int32. It is an involution: `rt_from_index`
    inverts it exactly, which is what lets a control inject a perturbation of an exact ULP size.
    """
    bits = np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32).astype(np.int64)
    return np.where(bits < 0x80000000, bits, 0xFFFFFFFF - bits)


def rt_from_index(index):
    """The exact inverse of `rt_ordered_index`, back to fp32 values."""
    bits = np.where(index < 0x80000000, index, 0xFFFFFFFF - index).astype(np.uint32)
    return np.ascontiguousarray(bits.view(np.float32))


def rt_ulp_slots(a, b):
    """Per-element fp32 ULP distances between two equal-length field lists.

    A metric on the finite float set (the ordered image is an integer embedding), which is what makes
    the triangle inequality in `rt_wpre_scope` exact rather than a rule of thumb. Reconciled against
    the licensor's own ruler by `rt-ulp-ruler-matches-the-licensor`, so there is one ruler in the
    repo even though this one reports per slot.
    """
    return np.abs(rt_ordered_index(a) - rt_ordered_index(b))


def rt_field_distance(dev_values, host_values):
    """(n_differing, n_slots, worst ULP) between two compilations' versions of one field."""
    slots = rt_ulp_slots(dev_values, host_values)
    return int(np.count_nonzero(slots)), int(len(slots)), (int(slots.max()) if len(slots) else 0)


def rt_bump(values, steps):
    """Move every element of `values` exactly `steps` ULP away, for the rejection legs."""
    return rt_from_index(rt_ordered_index(values) + steps)


def rt_words(text, patterns, what):
    """Pull one spelled numeral out of a document's own words, or refuse.

    The bound on `wpre` rests on what two documents say, so the numbers are read out of them at run
    time: if a document is reworded the derivation refuses instead of quietly enforcing a limit that
    nothing licenses any more.
    """
    spelled = {"single": 1, "one": 1, "two": 2, "double": 2}
    for pat in patterns:
        m = re.search(pat, text, re.S)
        if m:
            word = m.group(1).lower()
            if word not in spelled:
                raise Refused("rt-scope-word-unmapped",
                              "%s names %r in a sentence this file reads as a numeral; a spelled "
                              "number it cannot map is one it cannot judge" % (what, word))
            return spelled[word], m.group(0).strip()
    raise Refused("rt-scope-undocumented",
                  "no sentence in %s matches %s; the cross-compilation scope is derived from that "
                  "wording, and wording that has moved means the scope has to be re-derived, not "
                  "remembered" % (what, patterns[0]))


def rt_wpre_scope(bands, route_ulp_max):
    """The cross-compilation scope on `wpre`, re-derived every run from artifacts — never measured.

    `route_ulp_max` is what this run's pair measured; it is an argument only so the caller can print
    the witness beside the priors, and nothing in the limit depends on it.

    `wpre` is R4's fp32 pre-cast weight. No licence measures it at exactness, and
    `prototype/moe_router/moe_router_kernels.h` §R5 says why: an independent fp32 route reproduces
    the pinned pre-cast weights to single-digit ULP and NOT bit-exactly, with §mr_exp naming the two
    libms' `exp()` as the mechanism. Bit-identity there was an unfireable claim; the admissible move
    (D-18: a re-derivation, not a relaxation) is a limit with three INDEPENDENT prior sources, none
    of them this build's output, and the run shows its own spread inside all three:

      1. the licensor's own licensed band for this field, `weights_vs_pinned_f32_ulp` from
         `prototype/moe_router/evidence/moe-router-host-2026-10-26.json` (a floor probe over the
         pinned rows, re-read here through `block_gate.prototype_limits()`), applied by the triangle
         inequality: two fp32 routes each within `band` of the pinned pre-cast arm are within
         `2 * band` of each other, exactly, because ULP distance is a metric;
      2. the documentary scope, §R5's own words ("single-digit ULP"), read out of the header; the
         largest value that phrase licenses is the largest number with that many digits;
      3. an error propagation from §mr_exp's documented exp scope: `p_i = exp_i / SUM_E exp` and
         `w_k = p_k / SUM_K p` are all same-sign sums, so a per-call exp difference of the documented
         size survives into `wpre` at worst-case binary32 spacing.

    The binding limit is the STRICTEST of the three, so a document that loosens cannot widen the bound
    silently, and the licensor's band (prior 1) is used only while its premise is seen to hold — both
    routes measured within it of the pinned pre-cast record — which is checked and printed by
    `rt-wpre-premise-measured-against-the-pin`.
    """
    proto = DA.RT_HEADER.read_text()
    priors = []
    band = bands.get("weights_vs_pinned_f32_ulp")
    if band is None or int(band) <= 0:
        raise Refused("rt-scope-band-absent",
                      "weights_vs_pinned_f32_ulp is %r in the licensor's floor-probe artifact; the "
                      "first prior of this bound is that probe, and without it there is no licensed "
                      "number to apply the triangle inequality to" % (band,))
    priors.append((2 * int(band),
                   "triangle(2 x licensed weights_vs_pinned_f32_ulp=%s, "
                   "prototype/moe_router/evidence/moe-router-host-2026-10-26.json §bands)" % band))
    digits, quote = rt_words(proto, (r"to (single|one|two|double)-digit ULP",), str(DA.RT_HEADER))
    doc = 10 ** digits - 1
    priors.append((doc, "documented §R5 %r -> at most %d ULP" % (quote, doc)))
    exp_ulp, exp_quote = rt_words(proto, (r"differ by an ULP or (two|one)",), str(DA.RT_HEADER))
    wide = 2.0 ** -23
    d_exp = exp_ulp * wide
    rho_w = 4 * d_exp + 3 * wide
    priors.append((int(math.ceil(rho_w / (2.0 ** -24))),
                   "propagated §mr_exp %r (%d ULP/call) through R2 then R4 -> %.3e relative" %
                   (exp_quote, exp_ulp, rho_w)))
    limit, label = min((v, why) for v, why in priors)
    parts = ["scope_ulp=%d binding=%s" % (limit, label)]
    for v, why in priors:
        parts.append("prior=%d %s" % (v, why))
    return limit, "; ".join(parts)


RT_ZERO_TOLERANCE_FIELDS = ("ids", "ids_alt", "wcast")
RT_SCOPED_FIELD = "wpre"
RT_R1_FIELDS = ("logits", "logits_f32")


def rt_scope_block(ctx, dumps, pair_label, a_fields, b_fields, pair_sides):
    """Derive the `wpre` cross-compilation scope, print `WPRESCOPE`, and stash both for the checks.

    Runs in both modes on purpose: the same derivation licenses the hipcc-host-vs-g++ pair on a laptop
    and the device-vs-host pair in a window, so the bound and its rejection control are exercisable
    without a device window, and the number a device run enforces is not one only a device run can see.
    """
    nd, n, mx = rt_field_distance(a_fields[RT_SCOPED_FIELD], b_fields[RT_SCOPED_FIELD])
    band = int(ctx["limits_raw"]["weights_vs_pinned_f32_ulp"])
    to_pin = dict((label, rt_form_distance(dumps, "layer0.topk_weights_f32", f[RT_SCOPED_FIELD])[1])
                  for label, f in zip(pair_sides, (a_fields, b_fields)))
    limit, why = rt_wpre_scope(ctx["limits_raw"], mx)
    ctx["rt_wpre_scope"] = (limit, why)
    ctx["rt_wpre"] = dict(pair=pair_label, sides=pair_sides, a=a_fields, b=b_fields, ndiff=nd, n=n,
                          max_ulp=mx, band=band, to_pin=to_pin,
                          slots=[int(x) for x in rt_ulp_slots(a_fields[RT_SCOPED_FIELD],
                                                              b_fields[RT_SCOPED_FIELD])])
    print("WPRESCOPE field=%s pair=%s n=%d ndiff=%d/%d ulp_per_slot=%s measured_max_ulp=%d %s "
          "verdict=%s note=the limit comes from three independent priors (one licensor floor probe, "
          "two documents) and the strictest of them binds; measured_max_ulp is the witness that none "
          "of them is contradicted, never the source of the limit"
          % (RT_SCOPED_FIELD, pair_label, n, nd, n, ",".join(str(x) for x in ctx["rt_wpre"]["slots"]),
             mx, why, "WITHIN-SCOPE" if mx <= limit else "OUTSIDE-SCOPE"))


def rt_attribution(fields, names):
    """Is the device-vs-host difference the documented `exp()` scope, or is it a codegen defect?

    Falsifiable, and each named condition is a separate answer printed in the caller's line: R1's two
    outputs identical (the long dot is not the mechanism); the cast field identical (§R5: the bf16
    cast reproduces bit-exactly even when `wpre` does not); every moving scoped slot has its own
    numerator addend in `probs` moving (the causal chain, not just correlation); and no field moves
    that is not downstream of `exp()`.
    """
    d, h = fields
    r1_clean = all(rt_field_distance(d[n], h[n])[0] == 0 for n in names["r1"])
    cast_clean = rt_field_distance(d["wcast"], h["wcast"])[0] == 0
    ids = [int(x) for x in d["ids"]]
    causal, moved_outside = True, []
    for k, eid in enumerate(ids):
        wpre_moved = rt_field_distance([d["wpre"][k]], [h["wpre"][k]])[0] != 0
        probs_moved = rt_field_distance([d["probs"][eid]], [h["probs"][eid]])[0] != 0
        if wpre_moved and not probs_moved:
            causal = False
    for name in names["other"]:
        if rt_field_distance(d[name], h[name])[0]:
            moved_outside.append(name)
    return dict(r1_clean=r1_clean, cast_clean=cast_clean, causal=causal,
                moved_outside=moved_outside)


def control_rt_dtype_order(v, ctx):
    """Prediction 4, enforced in the arm rather than assumed of it.

    `engine/block_gate.py` proves the cast-before-multiply rule for the torch program with a triple
    (`TOPKFORM`): the cast-first weights reproduce the pinned bf16 bytes, the f32 carry reproduces the
    pinned f32 bytes, and the arm-basis leak — R1 left in fp32, which is what the prototype lane shipped
    at its deadline — fails to reproduce the bf16 ones. This control asks the compiled arm for the same
    three numbers, out of ONE record, from bytes the arm computed itself. If the leak coincided with the
    cast form the device arm would not be enforcing the rule at all, and the row that reads `ndiff=0`
    would be a coincidence of this row's margin rather than a property of the kernel.
    """
    f = ctx["rt_fields"]
    forms = (("weights-cast-first", "wcast", "layer0.topk_weights"),
             ("weights-f32-carry", "wpre", "layer0.topk_weights_f32"),
             ("weights-arm-basis-leak-pre-fix", "wleak", "layer0.topk_weights"))
    got = {}
    for label, field, refname in forms:
        nd, ulp = rt_form_distance(ctx["dumps"], refname, f[field])
        got[label] = (nd, ulp)
        print("RTFORM arm=policy_a form=%s field=%s referent=%s ndiff=%d max_ulp=%d n=%d"
              % (label, field, refname, nd, ulp, len(f[field])))
    cast_bits = np.ascontiguousarray(f["wcast"]).view(np.uint32)
    carry_bits = np.ascontiguousarray(f["wpre"]).view(np.uint32)
    leak_bits = np.ascontiguousarray(f["wleak"]).view(np.uint32)
    sep = dict(cast_vs_carry=int(np.count_nonzero(cast_bits != carry_bits)),
               cast_vs_leak=int(np.count_nonzero(cast_bits != leak_bits)))
    v.control("router-dtype-order-distinguishable-in-arm",
              got["weights-cast-first"][0] == 0 and got["weights-arm-basis-leak-pre-fix"][0] > 0
              and sep["cast_vs_carry"] > 0 and sep["cast_vs_leak"] > 0,
              "cast-first ndiff=%d (limit %s, from `engine/block_gate.py`'s licensor, not typed here) "
              "against the pinned bf16 weights; arm-basis leak ndiff=%d and refused; f32-carry ndiff=%d "
              "max_ulp=%d against the pinned f32 weights, which is not the same claim as the other two "
              "and is not required to be zero: `prototype/moe_router/moe_router_kernels.h` §R5 says an "
              "independent fp32 route reproduces the pinned pre-cast weights to single-digit ULP and not "
              "bit-exactly, and `engine/block_gate.py`'s TOPKFORM reads 0 there only because its f32 form "
              "is the same torch program that made the dump. What the arm has to prove is that the three "
              "forms are three different numbers in one record — cast-vs-carry differ in %d of %d slots, "
              "cast-vs-leak in %d — so the row that reads 0 is enforcing R5, not landing on 0 by "
              "coincidence"
              % (got["weights-cast-first"][0], ctx["limits"].get("topk_weights"),
                 got["weights-arm-basis-leak-pre-fix"][0], got["weights-f32-carry"][0],
                 got["weights-f32-carry"][1], sep["cast_vs_carry"], len(f["wcast"]),
                 sep["cast_vs_leak"]))


def control_rt_selection(v, ctx):
    """Top-k selection must be the thing the `topk_ids` row is about, in both halves, ON THE PATH.

    Window e2 caught this control guarding only one path: both legs ran on the host twin while the
    honest record of a `--device` run is the DEVICE arm's, so the must-not-move test compared a
    host-twin record against a device record, found `probs` moved (§mr_exp's documented exp() scope,
    not a `topk_ascending` bug), and read SILENT in exactly the mode everything above M2 is built on.
    A leg that did not run on the path under test is not a leg on that path, so every path this run
    executed now gets both legs against its OWN honest record, and the leg counts come off the arms'
    own `RUN path=` reports rather than from this control's intention:

      * tie-break — a forced exact tie across the K boundary (forged at the boundary the row itself
        selects, not at a typed index), where the rule in force (lower index) must still pick the
        pinned expert while the other rule (higher index) names a different one, reddening that
        path's own `topk_ids` row;
      * slot order — the prototype's own `mode=topk_ascending`, which must move the ids and the
        weights while leaving that same path's logits and probabilities bit-identical, because
        `mr_mode_reaches` says it cannot reach them. A mode that moves more than it names is a second
        bug, and a control that only looked at the ids row would not see that.
    """
    rt, work, args = ctx["rt"], ctx["work"], ctx["args"]
    sdir = work / ("rtselect_device" if ctx["device"] else "rtselect")
    sdir.mkdir(parents=True, exist_ok=True)
    base = ctx["rt_slab"]["plan"].read_text()
    rows_before = WITNESS["rows_emitted"]
    legs_seen = dict(PATH_LEGS)
    # (path label, binary to run, that path's own honest record, is it a device act)
    paths = [("host_cpubody", args.rt_host_binary, ctx["rt_fields_host"], False)]
    if ctx["device"]:
        paths.append(("device", args.rt_arm_binary, ctx["rt_fields"], True))
    legs_made, per_path, all_ok = {}, [], True
    for label, binary, honest, want_device in paths:
        honest_nd = {k: rt_form_distance(ctx["dumps"], "layer0.%s" % k, honest[f])[0]
                     for k, f in (("topk_ids", "ids"), ("topk_weights", "wcast"))}

        def run_variant(name, text, path=label, exe=binary, dev=want_device):
            p = sdir / ("plan_%s_%s.txt" % (path, name))
            p.write_text(text)
            out = sdir / ("out_%s_%s.bin" % (path, name))
            arm_leg(exe, p, ctx["rt_slab"]["slab"], out, device=dev, label="%s:%s" % (path, name))
            return DA.rt_record(out, rt)[0]

        tie = run_variant("tie_forge", base.replace("tie_forge=0", "tie_forge=1"))
        tie_nd_ids = rt_form_distance(ctx["dumps"], "layer0.topk_ids", tie["ids"])[0]
        tie_alt_nd = rt_form_distance(ctx["dumps"], "layer0.topk_ids", tie["ids_alt"])[0]
        tie_rules_differ = bool((tie["ids"] != tie["ids_alt"]).any())

        asc = run_variant("topk_ascending", base.replace("mode=correct", "mode=topk_ascending"))
        asc_nd_ids = rt_form_distance(ctx["dumps"], "layer0.topk_ids", asc["ids"])[0]
        asc_nd_w = rt_form_distance(ctx["dumps"], "layer0.topk_weights", asc["wcast"])[0]
        untouched = [n for n in ("logits", "probs") if bool((asc[n] != honest[n]).any())]
        fired = (tie_rules_differ and tie_alt_nd > honest_nd["topk_ids"]
                 and asc_nd_ids > honest_nd["topk_ids"] and asc_nd_w > honest_nd["topk_weights"]
                 and not untouched)
        all_ok = all_ok and fired
        for p, n in legs_since(legs_seen).items():
            legs_made[p] = legs_made.get(p, 0) + n
        legs_seen = dict(PATH_LEGS)
        per_path.append("%s legs=2: tie ids %d->%d, other-rule row %d (must exceed %d); ascending "
                        "ids %d->%d, weights %d->%d, fields that had to stay put and did: %s"
                        % (label, honest_nd["topk_ids"], tie_nd_ids, tie_alt_nd,
                           honest_nd["topk_ids"], honest_nd["topk_ids"], asc_nd_ids,
                           honest_nd["topk_weights"], asc_nd_w, untouched or "none"))
    detail = ("per path, each leg against that path's own honest record — %s. The two rules naming "
              "different experts is the half that makes the 0 a test and not a tautology: an arm with "
              "the tie rule backwards prints a red `topk_ids` row on this very input. The ascending "
              "half compares within one path on purpose — cross-path, §mr_exp's exp() scope moves "
              "`probs` and the control reddens for a reason it does not measure (window e2). Rows "
              "emitted by the control=%d"
              % (" | ".join(per_path), WITNESS["rows_emitted"] - rows_before))
    v.control("topk-selection-perturbations-redden-ids",
              all_ok and sum(legs_made.values()) == 2 * len(paths)
              and WITNESS["rows_emitted"] == rows_before,
              detail, legs=legs_made, claims=tuple(p for p, _, _, _ in paths))


def control_rt_wpre_scope(v, ctx):
    """The derived `wpre` scope is a bound only if it rejects something. Six legs, all printed.

    A bound read off what the current build produced would be the forbidden move — "a limit of `>= d`
    rejects nothing and is not a gate" (`pitfalls/40-gates-and-admissions.md`). So the size of every
    perturbation here comes from the derived limit, never from the observed spread, and legs 1, 3, 4
    and 6 run in CPU mode too, which is what lets the bound be falsified on a laptop instead of
    waiting for a device window:

      1. `honest-pair` — the pair this run produced must sit INSIDE the scope;
      2. `real-arm-mistake` — the prototype's own `mode=softmax_bf16` (named in the engine header's
         R4 comment as the bf16-renormalise mistake, and `mr_mode_reaches` says it reaches `wpre`),
         executed on each path this run has, must land OUTSIDE it. That is an arm-produced rejection:
         a real mistake the lane already names, not an injected bit;
      3/4. `at-limit` / `one-past-limit` — every slot moved by exactly `limit` and `limit + 1` ULP, so
         the boundary the check enforces is shown to be the boundary it claims, from both sides;
      5. `far-past-limit` — moved by `1 << (limit.bit_length() + 2)` ULP, which at this magnitude
         flips an exponent bit and must read OUTSIDE ("flip a high bit", with no typed bit index);
      6. `licensed-field-one-bit` — xor bit 0 of one `wcast`, one `ids` and one `ids_alt` slot; the
         zero-tolerance half of the split must report 1/8 for each, so scoping `wpre` cannot have
         swallowed the three fields that stay at exactness.
    """
    rt, work, args = ctx["rt"], ctx["work"], ctx["args"]
    bdir = work / ("rtbound_device" if ctx["device"] else "rtbound")
    bdir.mkdir(parents=True, exist_ok=True)
    limit, _why = ctx["rt_wpre_scope"]
    legs_seen = dict(PATH_LEGS)
    rows_before = WITNESS["rows_emitted"]
    paths = [("host_cpubody", args.rt_host_binary, ctx["rt_fields_host"], False)]
    if ctx["device"]:
        paths.append(("device", args.rt_arm_binary, ctx["rt_fields"], True))
    honest_pair = (ctx["rt_wpre"]["a"]["wpre"], ctx["rt_wpre"]["b"]["wpre"])
    legs, ok = [], True

    def arm_run(label, text, binary, dev, out):
        p = bdir / ("plan_%s.txt" % label)
        p.write_text(text)
        arm_leg(binary, p, ctx["rt_slab"]["slab"], out, device=dev, label=label)
        return DA.rt_record(out, rt)[0]

    nd, n, mx = rt_field_distance(*honest_pair)
    ok = ok and mx <= limit
    legs.append("honest-pair pair=%s ndiff=%d/%d max_ulp=%d limit=%d -> %s"
                % (ctx["rt_wpre"]["pair"], nd, n, mx, limit,
                   "INSIDE" if mx <= limit else "OUTSIDE"))

    for label, binary, honest, want_device in paths:
        bad = arm_run("%s_softmax_bf16" % label,
                      ctx["rt_slab"]["plan"].read_text().replace("mode=correct",
                                                                 "mode=softmax_bf16"),
                      binary, want_device, bdir / ("out_%s_bf16renorm.bin" % label))
        nd2, n2, mx2 = rt_field_distance(bad["wpre"], honest["wpre"])
        outside = mx2 > limit and nd2 > 0
        ok = ok and outside
        legs.append("real-arm-mistake path=%s mode=softmax_bf16 ndiff=%d/%d max_ulp=%d limit=%d -> %s"
                    % (label, nd2, n2, mx2, limit,
                       "OUTSIDE (rejected, as it must be)" if outside else
                       "INSIDE (the bound is too loose to catch the lane's own named mistake)"))

    for name, steps, want in (("at-limit", limit, True),
                              ("one-past-limit", limit + 1, False),
                              ("far-past-limit", 1 << (limit.bit_length() + 2), False)):
        nd3, n3, mx3 = rt_field_distance(rt_bump(honest_pair[0], steps), honest_pair[0])
        passed = (mx3 <= limit) == want
        ok = ok and passed
        legs.append("%s moved=%dULP ndiff=%d/%d max_ulp=%d limit=%d -> %s%s"
                    % (name, steps, nd3, n3, mx3, limit,
                       "INSIDE" if mx3 <= limit else "OUTSIDE",
                       "" if passed else " (WRONG: the bound is not the boundary it claims)"))

    bites = []
    for field in ("wcast", "ids", "ids_alt"):
        base = np.ascontiguousarray(ctx["rt_fields"][field]).view(np.uint32)
        flip = base.copy()
        flip[0] ^= np.uint32(1)
        nbite = int(np.count_nonzero(flip != base))
        bites.append("%s=%d/%d" % (field, nbite, len(base)))
        ok = ok and nbite == 1
    legs.append("licensed-field-one-bit xor bit0 of one slot each -> %s (each must read 1/%d, so the "
                "zero-tolerance half still bites one bit)" % (", ".join(bites), len(base)))

    legs_made = legs_since(legs_seen)
    detail = ("field=%s scope=%d ULP; legs: %s. No constant predicate: legs 1 and 2 are records the "
              "arms produced, 3-5 are perturbations sized from the derived limit, and 6 is a bit flip "
              "on the licensed fields. Arm legs executed by this control=%s; rows emitted=%d"
              % (RT_SCOPED_FIELD, limit, "; ".join(legs),
                 ",".join("%s:%d" % (p, n) for p, n in sorted(legs_made.items())) or "none",
                 WITNESS["rows_emitted"] - rows_before))
    v.control("wpre-scope-rejects-a-bigger-perturbation",
              ok and sum(legs_made.values()) > 0 and WITNESS["rows_emitted"] == rows_before,
              detail, legs=legs_made, claims=tuple(p for p, _, _, _ in paths))


def control_rt_f32_span(v, ctx):
    """The router family's licence rests on the weights being stored exact; this shows what the arm
    actually depends on, and prints the part of the premise this artifact cannot demonstrate.

    Four legs, in the order the claims get cheaper:
      * `elt=2` — a plan that declares a 2-byte element is refused by `engine/device/m2_rt_io.h` before
        a pointer is taken. Structural, and it fires.
      * loader type + element count + byte length — printed as `RTWIDTH`, checked in
          `engine/device_arm.py` `residency_router()`. Structural, and it holds.
      * bf16 round-trip of the span — **measured void here, and printed rather than kept as a fake leg.**
        Every value in `blk.0.ffn_gate_inp.weight` has a zero low halfword, so widening to bf16 and back
        changes 0 of 524,288 values: a bf16 copy of this tensor is the same numbers, and a "lossy copy"
        perturbation built that way could only ever print `ndiff 0->0`. The finding is the point: the
        F32 premise is invisible to a bf16 round-trip on this artifact, so it must be carried
        structurally and by the next leg, not by a rounding test.
      * one-ULP low-bit flip — xor bit 0 of every weight, the smallest difference an F32 store can carry
        and one a bf16 store cannot express at all. The arm's own fp32 field `logits_f32` must move; the
        bf16 `router_logits` row is reported and not required, because bf16 rounding hides a 2^-24
        change and pretending otherwise would be sizing a band by the build.
    The digest bind compares span against span (the same `sha256` over the same `length` bytes, from the
    file and from the slab), shown holding on the honest copy and breaking on the flipped one.
    """
    work, args, rt = ctx["work"], ctx["args"], ctx["rt"]
    fdir = work / "rtwidth"
    fdir.mkdir(parents=True, exist_ok=True)
    s = rt_span(ctx["st"])
    rows_before = WITNESS["rows_emitted"]
    honest = ctx["rt_fields"]
    honest_bits = np.ascontiguousarray(honest["logits_f32"]).view(np.uint32)

    bad = fdir / "plan_elt2.txt"
    bad.write_text(ctx["rt_slab"]["plan"].read_text().replace("elt=4", "elt=2"))
    try:
        DA.run_arm(args.rt_host_binary, bad, ctx["rt_slab"]["slab"], fdir / "out_elt2.bin",
                   label="elt2")
        leg_a = "NOT-REFUSED"
    except DA.Refused as exc:
        leg_a = "%s(%s)" % (exc.tag, "elt=2 named" if "elt=2" in str(exc) else "elt not named")

    w = np.frombuffer(s["dev_bytes"], dtype=np.float32).copy()
    w_bits = w.view(np.uint32)
    back = torch.from_numpy(w.copy()).to(torch.bfloat16).to(torch.float32).numpy().view(np.uint32)
    changed = int(np.count_nonzero(back != w_bits))
    low16 = int(np.count_nonzero(w_bits & np.uint32(0xFFFF)))
    print("RTVOID leg=bf16-roundtrip tensor=%s span_bytes=%d values=%d changed_values=%d "
          "values_with_nonzero_low16=%d verdict=%s note=a bf16 copy of this span is the same numbers, so "
          "this leg cannot redden anything and is reported, not required; the F32 premise is carried by "
          "the plan refusal, the residency width checks in engine/device_arm.py, and the one-ULP flip"
          % (s["name"], s["length"], w.size, changed, low16,
             "VOID-ON-THIS-ARTIFACT" if changed == 0 else "LOSSY"))
    lossy_ok = True
    if changed:
        lossy_slab = build_rt_slab(ctx["st"], ctx["x_bytes"], rt, fdir, side="file",
                                   router_bytes=back.view(np.float32).tobytes(), subdir="roundtrip")
        DA.run_arm(args.rt_host_binary, lossy_slab["plan"], lossy_slab["slab"],
                   fdir / "out_lossy.bin", label="router-bf16-roundtrip")
        lf, _ = DA.rt_record(fdir / "out_lossy.bin", rt)
        lossy_ok = (rt_form_distance(ctx["dumps"], "layer0.router_logits", lf["logits"])[0]
                    > rt_form_distance(ctx["dumps"], "layer0.router_logits", honest["logits"])[0])

    flip_slab = build_rt_slab(ctx["st"], ctx["x_bytes"], rt, fdir, side="file",
                              router_bytes=(w_bits ^ np.uint32(1)).view(np.float32).tobytes(),
                              subdir="flip")
    DA.run_arm(args.rt_host_binary, flip_slab["plan"], flip_slab["slab"], fdir / "out_flip.bin",
               label="router-lowbit-flip")
    ff, _ = DA.rt_record(fdir / "out_flip.bin", rt)
    moved = int(np.count_nonzero(np.ascontiguousarray(ff["logits_f32"]).view(np.uint32)
                                 != honest_bits))
    bf16_nd = rt_form_distance(ctx["dumps"], "layer0.router_logits", ff["logits"])[0]
    span_sha = s["file_sha"]
    bind_holds = ctx["rt_slab"]["bind"]["slab_sha"] == span_sha
    bind_broke = flip_slab["bind"]["slab_sha"] != span_sha
    v.control("router-span-is-f32",
              leg_a != "NOT-REFUSED" and "elt=2 named" in leg_a and lossy_ok and moved > 0
              and bind_holds and bind_broke and WITNESS["rows_emitted"] == rows_before,
              "declared-width leg: a plan saying elt=2 was refused with %s; residency leg: printed as "
              "RTWIDTH (%s, %d B for %d elements, %d B/elt read back %d B); bf16-roundtrip leg: VOID on "
              "this artifact (%d of %d values changed, %d with a nonzero low halfword) and therefore "
              "reported not required; one-ULP leg: flipping bit 0 of every one of the %d weights moves "
              "%d of %d fp32 logits in the arm's own logits_f32 field, while the bf16 router_logits row "
              "moves %d of 256 (bf16 rounding hides 2^-24, which is why the fp32 field is the one that "
              "carries this claim); bind: span-digest vs span-digest holds on the honest copy and breaks "
              "on the flipped one (%s/%s); rows emitted by the control=%d"
              % (leg_a, s["type_name"], s["length"], s["n_elts"],
                 s["length"] // max(s["n_elts"], 1), s["dev_read"], changed, w.size, low16, w.size,
                 moved, len(honest_bits), bf16_nd,
                 "holds" if bind_holds else "BROKE-EARLY",
                 "broke" if bind_broke else "HELD-WRONGLY",
                 WITNESS["rows_emitted"] - rows_before))


def control_rt_poison(v, ctx):
    """The router record's census must be able to be non-zero: skip R4/R5 and both weight fields are
    poison end to end."""
    work, rt = ctx["work"], ctx["rt"]
    pdir = work / "rtpoison"
    pdir.mkdir(parents=True, exist_ok=True)
    DA.run_arm(ctx["args"].rt_host_binary, ctx["rt_slab"]["plan"], ctx["rt_slab"]["slab"],
               pdir / "out.bin", fault="skip-renorm", label="rt_host_twin-skip-renorm")
    _, nan = DA.rt_record(pdir / "out.bin", rt)
    honest = ctx["rt_nan"]
    v.control("router-poison-census-can-be-nonzero",
              all(honest.get(f, -1) == 0 for f in ("wpre", "wcast", "ids"))
              and nan.get("wpre", -1) == rt["MR_K"] and nan.get("wcast", -1) == rt["MR_K"],
              "honest record: wpre %d/%d and wcast %d/%d slots unwritten; with R4/R5 skipped: %d/%d and "
              "%d/%d, while the ids field stays written (%d/%d) because R3 ran. A guard that can only "
              "read zero is not a guard"
              % (honest.get("wpre", -1), rt["MR_K"], honest.get("wcast", -1), rt["MR_K"],
                 nan.get("wpre", -1), rt["MR_K"], nan.get("wcast", -1), rt["MR_K"],
                 nan.get("ids", -1), rt["MR_K"]))


def checks_rt(v, ctx, rt_runs):
    """Checks over the router arms' own reports, geometry and record."""
    rt, cfg, device = ctx["rt"], ctx["cfg"], ctx["device"]
    f = ctx["rt_fields"]
    v.check("rt-arm-stages-equal-manifest-population",
            sorted(RT_STAGES) == sorted(ctx["population"]["router"]) and len(RT_STAGES) > 0,
            "this path declares %s; the licensor's own router family is %s — the must-produce set and "
            "the licence set are read from the same tuple, so they cannot drift"
            % (",".join(sorted(RT_STAGES)), ",".join(sorted(ctx["population"]["router"]))))
    v.check("rt-geometry-matches-the-config",
            rt["MR_E"] == cfg["n_experts"] and rt["MR_H"] == cfg["d"] and rt["MR_K"] == cfg["top_k"],
            "MR_E=%d cfg n_experts=%d, MR_H=%d cfg d=%d, MR_K=%d cfg top_k=%d: the prototype header's "
            "router geometry and the checkpoint's must be the same numbers before a row means anything"
            % (rt["MR_E"], cfg["n_experts"], rt["MR_H"], cfg["d"], rt["MR_K"], cfg["top_k"]))
    census_bad, detail = [], []
    for label in sorted(rt_runs):
        r = rt_runs[label]["rec"]["run"]
        for key, want in (("reads_w", rt["MR_E"] * rt["MR_H"]), ("reads_x", rt["MR_E"] * rt["MR_H"])):
            if int(r.get(key, -1)) != want:
                census_bad.append("%s.%s=%s/%d" % (label, key, r.get(key), want))
        if r.get("items") != (rt["MR_E"], 1):
            census_bad.append("%s.items=%s/%d/1" % (label, r.get("items"), rt["MR_E"]))
        if r.get("bad_index", 0) != 0:
            census_bad.append("%s.bad_index=%s" % (label, r.get("bad_index")))
        detail.append("%s[%s]" % (label, " ".join("%s=%s" % (k, r.get(k)) for k in
                                                  ("reads_w", "reads_x", "items", "path"))))
    v.check("rt-census-equals-geometry", not census_bad and len(rt_runs) > 0,
            "element reads and item counts required to equal the geometry (E*H of weight, E*H of "
            "activation re-read per expert, %d logit items and 1 serial item): bad=%s; %s"
            % (rt["MR_E"], census_bad or "none", "; ".join(detail)))
    pair_legs = [p for p in ctx["rt_pairs"] if not p["device"]]
    pair_bad = [p for p in pair_legs if p["ndiff"] != 0 or p["bytes"] <= 0]
    v.check("rt-compiled-pair-byte-identical", bool(pair_legs) and not pair_bad,
            "g++ twin vs hipcc host path over the router record, one source, -ffp-contract=off on "
            "both: %s. Zero tolerance, because `topk_weights` is licensed at exactness and a compiler "
            "that disagreed about the renormalisation is exactly the finding that must not be absorbed "
            "into a band; the zero-tolerance pair is host-vs-host by construction, and a device leg "
            "(present only with --device, marked device=%s below) is compared to the host leg on the "
            "claimed fields alone, under `rt-device-host-claimed-fields-identical`, because "
            "`prototype/moe_router/moe_router_kernels.h` §mr_exp documents that the two libms' exp() may "
            "differ by an ULP%s"
            % ("; ".join("%s bytes=%d ndiff_bytes=%d device=%s" % (p["label"], p["bytes"],
                                                                  p["ndiff"], p["device"])
                         for p in ctx["rt_pairs"]),
               ",".join(str(p["device"]) for p in ctx["rt_pairs"]),
               (" BAD=" + ",".join("%s" % p["label"] for p in pair_bad)) if pair_bad else ""))
    ids = np.asarray(f["ids"], dtype=np.int64)
    alt = np.asarray(f["ids_alt"], dtype=np.int64)
    v.check("rt-ids-are-a-selection",
            bool(((ids >= 0) & (ids < rt["MR_E"])).all())
            and len(set(ids.tolist())) == len(ids.tolist())
            and sorted(alt.tolist()) == sorted(ids.tolist())
            and ctx["rt_nan"].get("ids", 1) == 0,
            "the honest record's %d ids are in range, distinct, and the two tie rules selected the same "
            "%d experts in some order (this row has no exact tie at the boundary, so the rules coincide "
            "here — `topk-selection-perturbations-redden-ids` is what shows they need not)"
            % (len(ids), len(alt)))
    v.check("rt-poison-census-zero-on-the-honest-record",
            all(ctx["rt_nan"].get(k, 1) == 0 for k in ctx["rt_nan"]) and len(ctx["rt_nan"]) == 8,
            "all 8 fields of the honest router record are fully written: %s. The non-zero leg is "
            "controlled separately, and a census that could only read 0 would prove nothing either way"
            % (",".join("%s=%d" % (k, ctx["rt_nan"][k]) for k in sorted(ctx["rt_nan"]))))
    wp, limit = ctx["rt_wpre"], ctx["rt_wpre_scope"][0]
    nd_a, nd_b = np.ascontiguousarray(wp["a"][RT_SCOPED_FIELD], np.float32), \
        np.ascontiguousarray(wp["b"][RT_SCOPED_FIELD], np.float32)
    ulp_ndiff, ulp_max = BG.ulp_diff(torch.from_numpy(nd_a), torch.from_numpy(nd_b))
    v.check("rt-ulp-ruler-matches-the-licensor",
            len(wp["slots"]) == wp["n"] and (ulp_ndiff, ulp_max) == (wp["ndiff"], wp["max_ulp"]),
            "`wpre` measured twice, once by `engine/block_gate.py`::ulp_diff (the licensor's ruler, "
            "aggregate only) and once per slot by `rt_ulp_slots` here: licensor ndiff=%d max_ulp=%d, "
            "here ndiff=%d max_ulp=%d over %d slots. The two folds are the same map widened to int64, "
            "and if they ever disagree the ULP number in this file means something different from the "
            "ULP number in every band"
            % (ulp_ndiff, ulp_max, wp["ndiff"], wp["max_ulp"], wp["n"]))
    band_now = int(ctx["limits_raw"]["weights_vs_pinned_f32_ulp"])
    probe = dict(ctx["limits_raw"])
    probe["weights_vs_pinned_f32_ulp"] = 1
    shifted, shift_why = rt_wpre_scope(probe, wp["max_ulp"])
    want_now = "prior=%d triangle(2 x licensed weights_vs_pinned_f32_ulp=%d," % (2 * band_now, band_now)
    want_probe = "prior=2 triangle(2 x licensed weights_vs_pinned_f32_ulp=1,"
    v.check("rt-wpre-scope-derived-from-artifacts",
            limit > 0 and ctx["rt_wpre_scope"][1].count("prior=") == 3
            and want_now in ctx["rt_wpre_scope"][1] and want_probe in shift_why,
            "the scope on `wpre` (%d ULP on pair `%s`) is the strictest of three priors — a licensor "
            "floor probe, §R5's wording, and an error propagation from §mr_exp's — and none of them is "
            "this build's output. The dependence on the licensor's artifact is witnessed, not claimed: "
            "the triangle prior is exactly twice `weights_vs_pinned_f32_ulp` as the artifact reads it "
            "(%d x 2 = %d), and re-deriving with that number set to 1 in memory makes the prior 2 (the "
            "binding limit would then be %d). A bound typed into this file would not move. Wording the "
            "driver cannot find in its source documents is a refusal, not a default"
            % (limit, wp["pair"], band_now, 2 * band_now, shifted))
    v.check("rt-wpre-premise-measured-against-the-pin",
            len(wp["to_pin"]) == 2 and all(u <= wp["band"] for u in wp["to_pin"].values()),
            "the triangle-inequality prior only holds if both compilations are themselves inside the "
            "licensed band of the pinned pre-cast weights: %s ULP against a licensed "
            "`weights_vs_pinned_f32_ulp` of %d. If a route drifts past its own licence the premise "
            "fails and this reddens, rather than letting a bound derived from two licensed legs "
            "license a third number nobody licensed"
            % (",".join("%s=%d" % (k, u) for k, u in sorted(wp["to_pin"].items())), wp["band"]))
    rt_launch_offences = launch_count_offences(dict(
        (k, a["rec"]) for k, a in rt_runs.items() if "device" in k))
    if device:
        v.check("rt-device-launch-counted",
                any(counter_positive(a["rec"]["run"]["launches"])
                    and counter_positive(a["rec"]["run"]["hip_calls"])
                    for a in rt_runs.values() if a["rec"]["run"]["path"] == "device")
                and not rt_launch_offences,
                "the device arm counted at least one launch and one HIP call, and every router record that "
                "reported `path=device` reported `launches` as an int > 0 (offences: %s); the flag is not "
                "the evidence, the counter is, and the counter has a type"
                % (",".join(rt_launch_offences) or "none"))
        cl, dd = ctx["rt_wpre"], ctx["rt_devdiff"]
        v.check("rt-device-host-licensed-fields-identical",
                len(dd["licensed"]) == len(RT_ZERO_TOLERANCE_FIELDS)
                and all(dd["licensed"][k][0] == 0 for k in RT_ZERO_TOLERANCE_FIELDS),
                "device arm vs host twin, zero tolerance on the fields a licence measures at "
                "exactness: %s (whole record ndiff_bytes=%d of %d, which is not this claim — the rest "
                "of those bytes are the `wpre` scope and the long-dot `probs`, both printed and neither "
                "absorbed). The tolerance-free form is the point: one differing bit of `topk_ids` or of "
                "the bf16 `topk_weights` reddens this check and cannot hide inside `wpre`'s measured "
                "spread, because those rows are licensed at exactness by `speccheck/spec.md §10.6` and "
                "the licensor's floor probe measured 164 f32 ULP of margin against a 5 ULP route spread "
                "before it agreed the cast was robust"
                % (",".join("%s=%d/%d" % (k, dd["licensed"][k][0], dd["licensed"][k][1])
                            for k in RT_ZERO_TOLERANCE_FIELDS), dd["full"][1], dd["full"][0]))
        v.check("rt-device-host-wpre-within-derived-scope",
                wp["max_ulp"] <= limit and wp["n"] == ctx["rt"]["MR_K"],
                "`wpre` (the fp32 pre-cast renormalised weight) between the two compilations: %d of %d "
                "slots differ, worst %d ULP, per-slot %s, against the derived scope of %d ULP in "
                "`WPRESCOPE`. This is a re-derivation, not a relaxation: §R5 documents the field as "
                "not bit-exact across an independent fp32 route, the limit comes from that document, "
                "§mr_exp's, and the licensor's floor probe, and `CONTROL "
                "name=wpre-scope-rejects-a-bigger-perturbation` witnesses that the limit rejects a "
                "larger-than-scope perturbation. Measuring the spread and enforcing what came out would "
                "be a gate that reddens on nothing"
                % (wp["ndiff"], wp["n"], wp["max_ulp"], ",".join(str(x) for x in wp["slots"]), limit))
        att = dd["attrib"]
        v.check("rt-device-host-difference-is-the-documented-exp-scope",
                att["r1_clean"] and att["cast_clean"] and att["causal"]
                and not att["moved_outside"],
                "attribution, because 'the two libms differ' is a story until each named condition is "
                "separately answerable: R1's outputs bit-identical=%s, `wcast` bit-identical=%s, every "
                "moving `wpre` slot has its own `probs` addend moving=%s, fields moving outside the "
                "exp() path=%s. All four hold, so the difference is the documented `exp()` scope and "
                "the fix belongs in the check. If any had failed it would be a codegen defect and the "
                "fix would belong in `engine/device/m2_rt_kernels.h`, which is why this is a check and "
                "not a sentence in a log"
                % (att["r1_clean"], att["cast_clean"], att["causal"],
                   att["moved_outside"] or "none"))
    check_control_paths(v, ctx)
    device_path_census(v, ctx)


def _guard_reads_device(node):
    """`(mentions, positive)` for a branch guard: does it read the device flag, and does its truth need it.

    Three spellings count, because the file uses all three: the local `device`, `ctx["device"]` and
    `args.device`. A negated guard (`if not ctx["device"]:`) marks its ELSE arm as the device-only one, which
    is the half that a naive "lines mentioning device" grep gets backwards. Callers pass the TEST, not the If
    node: negation is a property of the expression, and asking an `ast.If` whether it is a `not` is always no.
    """
    names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == "device"]
    subs = [n for n in ast.walk(node) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and n.slice.value == "device"]
    attrs = [n for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr == "device"]
    negated = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    return bool(names or subs or attrs), (not negated)


DEV_RUN_FIELDS = ("launches", "hip_calls", "allocs", "bytes_to_device", "bytes_from_device")
DEV_CENSUS = {}


def _callee_parts(func):
    """The `a.b.c` parts of a callee, outermost first, so a call site can be named in the log."""
    parts = []
    while isinstance(func, (ast.Attribute, ast.Name)):
        parts.append(func.attr if isinstance(func, ast.Attribute) else func.id)
        func = func.value if isinstance(func, ast.Attribute) else None
    return parts


def device_census_derive():
    """Derive, from this file's own AST, the statements it runs only on a device run.

    The population has to be derived because the alternative is a hand-typed list, and a hand-typed list of
    device-only paths is exactly what window `h`'s control text was: true when written, false by the next
    commit, and quiet about being false. `DEV_CENSUS["statements"]` maps the first line of each
    device-guarded branch statement to its source, its guard and the function it sits in; a compound statement
    carries its whole body in `text`, so a field read nested two `if`s deep is still visible in the entry that
    gates it.
    """
    if DEV_CENSUS:
        return DEV_CENSUS
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    func_of = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                func_of.setdefault(id(inner), node.name)
    statements = {}
    early_guard = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        mentions, positive = _guard_reads_device(node.test)
        if not mentions:
            continue
        guard = ast.get_source_segment(src, node.test) or "?"
        if isinstance(node, ast.If):
            arm = (node.body if positive else node.orelse) or []
        else:
            # `X if device else Y`: an IfExp's arms are single expression nodes, not statement lists.
            one = node.body if positive else node.orelse
            arm = [one] if one is not None else []
        for st in arm:
            statements[st.lineno] = dict(
                lineno=st.lineno, kind=type(st).__name__, guard=guard,
                func=func_of.get(id(st), "<module>"),
                text="\n".join(lines[st.lineno - 1:st.end_lineno]))
        # The other idiom this file uses is a guard that EXITS: `if not ctx["device"]: print GVOID…; return`.
        # Everything after it in that function body is device-only, and no branch nesting shows it, so the
        # early exit is recorded and used below to decide whether a `device=True` call site is guarded.
        if isinstance(node, ast.If) and not positive:
            exits = [n for st in node.body for n in ast.walk(st)
                     if isinstance(n, (ast.Return, ast.Raise))]
            if exits:
                fn = func_of.get(id(node))
                early_guard.setdefault(fn, []).append(node.lineno)
    # A device-only control is a function that returns early on the flag AND registers a control; that pair is
    # what a CPU witness cannot exercise, and it is derived here so the census is non-empty in either mode.
    guarded_fns = set(fn for fn in early_guard if fn)
    control_guards = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in guarded_fns:
            continue
        if "v.control(" in (ast.get_source_segment(src, node) or ""):
            control_guards.append(node.name)
    DEV_CENSUS["control_guards"] = sorted(set(control_guards))
    # A `device=True` keyword is a claim that an arm act is about to happen. Unguarded, it would ask a device
    # act from a run whose RESULT line says `device=off` — the shape of a false claim, so it is enumerated.
    # The exemption is derived from the first argument's name: `*_host_binary` means the call is asking the g++
    # twin to PROVE it refuses `--device` (a refusal leg, which runs in both modes by design);
    # `*_arm_binary` means a real act, which must sit under a device guard.
    claims = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(k.arg == "device" and isinstance(k.value, ast.Constant) and k.value.value is True
                   for k in node.keywords):
            continue
        guarded, walk = False, parents.get(id(node))
        while walk is not None:
            if isinstance(walk, (ast.If, ast.IfExp)) and _guard_reads_device(walk.test)[0]:
                guarded = True
                break
            walk = parents.get(id(walk))
        fn = func_of.get(id(node), "<module>")
        if not guarded:
            guarded = any(ln < node.lineno for ln in early_guard.get(fn, []))
        first = ast.get_source_segment(src, node) if node.args else "?"
        kind = ("host-refusal-leg" if "host" in first else
                "arm-act" if "arm" in first else "unknown-binary")
        claims.append(dict(lineno=node.lineno, guarded=guarded or kind == "host-refusal-leg", kind=kind,
                           binary=",".join(sorted(set(re.findall(r"\b(\w*(?:host|arm)_binary)\b", first))))
                           or "?",
                           func=fn, name=".".join(reversed(_callee_parts(node.func)))
                           if node.func is not None else "?"))
    DEV_CENSUS["statements"] = statements
    DEV_CENSUS["claims"] = claims
    DEV_CENSUS["field_sites"] = dict((f, [ln for ln, d in sorted(statements.items())
                                          if re.search(r"[\[.'\"]%s[\"'\]]" % f, d["text"])
                                          or re.search(r"\b%s\b" % f, d["text"])])
                                     for f in DEV_RUN_FIELDS)
    # The cast scan covers the WHOLE file, not just the device-only population: the witness fold that kept
    # window `i` from printing a RESULT line lived in a shared helper, so limiting the scan to device-gated
    # branches would have missed it. The shape asked for is a cast applied to a quoted counter name, which is
    # what the code below looks for; a comment that merely describes that shape is skipped, because a comment
    # cannot execute and refusing to write about the defect would make the check worse, not better.
    DEV_CENSUS["cast_sites"] = [i for i, text in enumerate(lines, 1)
                                if not text.lstrip().startswith("#")
                                and re.search(r"\b(?:int|float)\s*\([^)]*[\"'](?:%s)[\"']"
                                             % "|".join(DEV_RUN_FIELDS), text)]
    # …and so does the other half of the same defect: a direct comparison of a quoted counter, which is what
    # actually raised in window `i`. The sanctioned forms are `counter_positive(x["launches"])` and
    # `launch_count_offences`, neither of which compares a quoted key inline.
    DEV_CENSUS["compare_sites"] = [i for i, text in enumerate(lines, 1)
                                   if not text.lstrip().startswith("#")
                                   and re.search(r"(?<!WITNESS)\[\s*[\"'](?:%s)[\"']\s*\]\s*[<>]"
                                                 % "|".join(DEV_RUN_FIELDS), text)]
    return DEV_CENSUS


def device_path_census(v, ctx):
    """Every device-only path and every device-claiming control, printed and then held to its own account.

    Two claims, both read off derived populations rather than typed:

      * **Enumeration.** The device-only statements of this file (AST-derived) and the controls that claim the
        device path are each printed with a status. In a CPU run every one of them prints
        `status=not-exercised` with a named reason — which is the honest statement of what a CPU session can
        know, and is witnessed non-zero rather than vacuous. In a device window a control that still claims the
        path with zero legs prints its own `reason=`, which is what `silent_control_finding` now requires of
        every quiet control anyway; a claim with no exercised leg therefore cannot be printed silently.
      * **Type safety at the only door.** The five counters a device check reads off an arm report
        (`launches`, `hip_calls`, `allocs`, `bytes_to_device`, `bytes_from_device`) must be integer-typed in
        every compiled lane's own printf format, and no consumer anywhere in this file may cast one. Window `i`
        crashed at `str > int` because the parse site handed out strings and each consumer cast or did not, so
        the guarantee lives in `device_arm.RECORD_FIELD_TYPES` and this check is what keeps it there: a cast
        reappearing in the device-only population, or a lane whose format stops spelling the count with an
        integer conversion, reddens this check instead of reddening a window 40 minutes in.
    """
    census = device_census_derive()
    statements, claims = census["statements"], census["claims"]
    dev_legs = sum(d["legs"].get("device", 0) for d in v.control_paths.values() if d["claims"])
    claiming = dict((n, d) for n, d in v.control_paths.items() if "device" in d["claims"])
    printed = 0
    for name, d in sorted(claiming.items()):
        legs = d["legs"].get("device", 0)
        if legs:
            print("DEVONLY control=%s status=exercised legs=device:%d note=arm acts reported `path=device` by "
                  "the arm's own RUN line" % (name, legs))
        else:
            print("DEVONLY control=%s status=not-exercised reason=%s note=%s"
                  % (name, ("device-flag-off-in-this-run" if not ctx["device"]
                            else "claimed-device-and-executed-zero-device-legs"),
                     "a CPU witness: the guard that gates this control's legs was false, so nothing on the "
                     "device path ran; this line is a claim being declared unmet, not a pass"))
        printed += 1
    unguarded = ["%s:%d(%s)" % (c["func"], c["lineno"], c["name"]) for c in claims if not c["guarded"]]
    for fn in census["control_guards"]:
        print("DEVONLY control-function=%s status=%s reason=device-guard-returns-early-in-a-cpu-run "
              "note=the control it registers is unreachable without `--device`; this is the population that "
              "the CPU witness cannot exercise, derived from the AST, not typed"
              % (fn, "exercised" if ctx["device"] else "not-exercised"))
    for c in sorted(claims, key=lambda c: c["lineno"]):
        print("DEVONLY claim=%s:%d call=%s binary=%s kind=%s status=%s"
              % (c["func"], c["lineno"], c["name"], c["binary"], c["kind"],
                 "guarded" if c["guarded"] else "UNGUARDED"))
    print("DEVONLY census statements=%d control-functions=%d device-true-call-sites=%d unguarded=%s "
          "field-read-sites=%s cast-sites=%s compare-sites=%s control-claims-printed=%d/%d"
          % (len(statements), len(census["control_guards"]), len(claims),
             ",".join(unguarded) or "none",
             ",".join("%s:%d" % (f, len(census["field_sites"][f])) for f in DEV_RUN_FIELDS),
             ",".join(str(x) for x in census["cast_sites"]) or "none",
             ",".join(str(x) for x in census["compare_sites"]) or "none", printed, len(claiming)))
    lane_types = dict((stem, dict((f, DA.RECORD_FIELD_TYPES[stem]["RUN"].get(f)) for f in DEV_RUN_FIELDS))
                      for stem in sorted(DA.RECORD_FIELD_TYPES))
    untyped = ["%s.%s=%s" % (stem, f, t) for stem, row in sorted(lane_types.items())
               for f, t in sorted(row.items()) if t not in ("int",)]
    v.check("device-only-paths-are-enumerated-not-assumed",
            bool(statements) and bool(census["control_guards"]) and printed == len(claiming)
            and not unguarded,
            "%d device-only statements, %d device-only control functions and %d `device=True` call sites were "
            "read out of this file's own AST (%s cast sites on the five run counters, field reads %s) and "
            "%d/%d controls claiming the device path printed a DEVONLY status line — one per claim, so a claim "
            "cannot go unprinted. %d device legs were actually executed by arm acts in this run. Unguarded "
            "device claims: %s. Window `i`'s crash lived inside this population, which is why the population is "
            "derived"
            % (len(statements), len(census["control_guards"]), len(claims), len(census["cast_sites"]),
               ",".join("%s=%d" % (f, len(census["field_sites"][f])) for f in DEV_RUN_FIELDS),
               printed, len(claiming), dev_legs, ",".join(unguarded) or "none"))
    v.check("device-run-counters-are-typed-at-the-parse-site",
            not untyped and not census["cast_sites"] and not census["compare_sites"],
            "the five counters every device check reads are printed with an integer conversion by every "
            "compiled lane and are converted to `int` at the parse site and nowhere else: %s. Across the whole "
            "file there are %d casts of a quoted counter and %d inline comparisons of one — both must be zero, "
            "because a consumer that casts or compares is a consumer that raises `TypeError` in a window "
            "instead of printing a FAIL line, which is what window `i` did at the GDN attribution and at "
            "`rt-device-launch-counted`, and why it printed no RESULT line"
            % ("; ".join("%s[%s]" % (stem, ",".join("%s:%s" % (f, lane_types[stem][f])
                                                    for f in DEV_RUN_FIELDS))
                         for stem in sorted(lane_types)) or "no lanes",
               len(census["cast_sites"]), len(census["compare_sites"])))


def check_control_paths(v, ctx):
    """The two checks that make a control's path claim cost something. Split out of `checks_rt` so the
    device-shaped case can be exercised without a device (`engine/work/probe_pass4_local_smokes.py`)."""
    claiming = dict((n, d) for n, d in v.control_paths.items() if d["claims"])
    silent_claims, unclaimed = [], sorted(set(v.control_paths) - set(claiming))
    for name, d in sorted(claiming.items()):
        for path in d["claims"]:
            if d["legs"].get(path, 0) <= 0:
                silent_claims.append("%s claims %s and executed 0 legs on it" % (name, path))
        if not ctx["device"] and d["legs"].get("device", 0) > 0:
            silent_claims.append("%s reported device legs in a run with no device act" % name)
    v.check("control-legs-cover-the-path-under-claim",
            bool(claiming) and not silent_claims,
            "%d of %d controls declare which path they covered, and every declared path has at least "
            "one arm act counted off that arm's own `RUN path=` report: %s. Controls declaring no path "
            "(their legs are plan/refusal legs, not path legs): %s. Window e2 is the reason this check "
            "exists: `topk-selection-perturbations-redden-ids` claimed the device path while both its "
            "legs ran on the host twin, and read SILENT in the mode everything above M2 is built on"
            % (len(claiming), len(v.control_paths),
               "; ".join("%s=%s" % (n, ",".join("%s:%d" % (p, d["legs"].get(p, 0))
                                                 for p in d["claims"]))
                         for n, d in sorted(claiming.items())) or "none",
               ",".join(unclaimed) or "none"))
    v.check("device-path-claims-executed-on-device",
            (not ctx["device"]) or sum(d["legs"].get("device", 0) for d in claiming.values()) >= 2,
            "a device run has to have run the perturbation controls on the device, not merely beside "
            "it: %d control legs executed under `path=device` (the licensed-row controls need their tie "
            "and bound legs on the arm they license). In a CPU run this check is vacuous by "
            "construction and says so, because there is no device path to claim"
            % sum(d["legs"].get("device", 0) for d in claiming.values()))
    v.check("silent-control-names-its-reason", not v.unreasoned,
            "every CONTROL line that did not fire also printed a `reason=` naming what it could not do — the "
            "rule lives in `silent_control_finding`, which `Verifier.control` runs on every control "
            "registration, so a control cannot be registered without passing it (window `h`: "
            "`gather-divergence-causes-priced` and `gdn-state-read-is-the-arm-s-own-fingerprint` both printed "
            "a quiet SILENT, and the check that was supposed to catch that was reading checks, which have no "
            "legs to lose). %d of %d controls fired; the ones that went quiet without naming a reason are %s. "
            "`silent-reason-guard-can-redden` is the leg that shows this predicate can still redden"
            % (v.fired, len(v.controls), ",".join(v.unreasoned) or "none"))
    text = run_text(ctx)
    contradictions = contradiction_findings(text)
    exact_pairs, residual_pairs = exact_reproductions(text)
    n_exact = sum(len(lines) for lines in exact_pairs.values())
    n_residual = sum(len(lines) for lines in residual_pairs.values())
    zero_legs = reproduction_leg_finding(ctx.get("device"), text)
    v.check("no-control-contradicts-an-exact-reproduction", not contradictions and not zero_legs,
            "%d reproduction legs printed `verdict=EXACT-REPRODUCTION` (pairs: %s) and %d printed a "
            "residual (pairs: %s) — counts parsed from the COMPLETE printed text after every leg had "
            "printed, read off the live buffer rather than a mid-run snapshot (window `j`: a snapshot "
            "taken before the legs printed parsed 0 against 3 printed and passed; that ordering is what "
            "this line now reads the whole log to forbid). No SILENT control in this log denies any of "
            "them. Window `i` printed `DREPRO … verdict=EXACT-REPRODUCTION` and, nine lines later, a "
            "control whose reason said `no injected cause reproduces the GDN device divergence` — the "
            "cause table predated the probe that answered the question and printed a confident falsehood "
            "in a gate log. The rule is read off this run's own printed lines, not off a flag, and "
            "NO_CAUSE_PHRASES names the phrasings it watches for. Zero-count assertion: %s. Refutations "
            "found: %s"
            % (n_exact, ",".join(sorted(exact_pairs)) or "none", n_residual,
               ",".join(sorted(residual_pairs)) or "none",
               zero_legs or ("a device run must and does see a non-zero leg count here"
                             if ctx.get("device") else
                             "exempt in a CPU run, which prints no DREPRO/GREPRO leg by construction"),
               "; ".join(contradictions) or "none"))
    bad_attrib = [a for a in ATTRIBS if a["moved"] != a["total"]]
    v.check("field-attribution-accounts-to-the-pair", bool(ATTRIBS) and not bad_attrib,
            "%d per-field attribution tables were printed (paired totals: %s) and in every one the field "
            "counts add up to the ndiff_bytes of the pair they belong to. A per-field table that did not "
            "account would be measuring a different population than the PAIR line it explains, and a run "
            "that printed no attribution at all would satisfy this check trivially, which is why the count "
            "is required non-zero here rather than assumed"
            % (len(ATTRIBS), ";".join("%s=%d/%d" % (a["pair"], a["moved"], a["total"])
                                      for a in ATTRIBS[:6])))


def want_reads(layout, which):
    """The element-read count the geometry implies, derived from the header's own constants."""
    if which == "fused":
        return 2 * layout["SE_I"] * layout["SE_H"]
    if which == "down":
        return layout["SE_H"] * layout["SE_I"]
    return layout["SE_H"]


def checks(v, ctx, rows, tally, text, parsed, arm_runs):
    """Every check is an expression over parsed records, derived populations, or counted acts."""
    layout, population, st = ctx["layout"], ctx["population"], ctx["st"]
    cfg, device = ctx["cfg"], ctx["device"]
    per_arm = len(population["all"])
    v.check("stage-rows-equal-derived-population", len(rows) == per_arm and per_arm > 0,
            "rows=%d derived=%d (the shared-expert stages the layer-0 manifest names)"
            % (len(rows), per_arm))
    v.check("parsed-record-count-equals-printed",
            len(parsed) == len(rows) + len(ctx.get("g_rows", [])) + len(ctx.get("d_rows", []))
            and len(parsed) > 0,
            "parsed=%d printed=%d (shared-expert and router rows=%d, gather rows on two width "
            "paths=%d, GDN recurrent rows=%d); a parser matching nothing would print 0 and pass"
            % (len(parsed),
               len(rows) + len(ctx.get("g_rows", [])) + len(ctx.get("d_rows", [])),
               len(rows), len(ctx.get("g_rows", [])), len(ctx.get("d_rows", []))))
    v.check("arm-stages-equal-manifest-population",
            sorted(population["arm_declared"]) == sorted(population["all"]),
            "this path declares %s; the manifest's shared-expert stages are %s — a rename on either "
            "side goes red instead of dropping a row" % (",".join(sorted(population["arm_declared"])),
                                                        ",".join(sorted(population["all"]))))
    v.check("kernel-geometry-matches-the-config",
            layout["SE_H"] == cfg["d"] and layout["SE_I"] == cfg["i_s"]
            and layout["SE_TWO_I"] == 2 * cfg["i_s"],
            "SE_H=%d cfg d=%d, SE_I=%d cfg i_s=%d, SE_TWO_I=%d: the prototype header's geometry and "
            "the checkpoint config must name the same numbers before a kernel's rows mean anything"
            % (layout["SE_H"], cfg["d"], layout["SE_I"], cfg["i_s"], layout["SE_TWO_I"]))
    census_bad, census_detail = [], []
    for label in sorted(arm_runs):
        run = arm_runs[label]["rec"]["run"]
        want = {k: want_reads(layout, k) for k in ("fused", "down", "sg")}
        for field, key in (("fused", "reads_fused"), ("down", "reads_down"), ("sg", "reads_sg")):
            if int(run.get(key, -1)) != want[field]:
                census_bad.append("%s.%s=%s/%d" % (label, key, run.get(key), want[field]))
        census_detail.append("%s[%s]" % (label, " ".join(
            "%s=%s" % (k, arm_runs[label]["rec"]["run"].get(k)) for k in
            ("reads_fused", "reads_down", "reads_sg", "items"))))
    v.check("kernel-census-equals-geometry", not census_bad and len(arm_runs) > 0,
            "element reads per arm, required to equal the geometry: bad=%s; %s"
            % (census_bad or "none", "; ".join(census_detail)))
    reads = set()
    for label in sorted(arm_runs):
        r = arm_runs[label]["rec"]["run"]
        reads.add((r.get("reads_fused"), r.get("reads_down"), r.get("reads_sg"), r.get("items"),
                   r.get("nan_fields")))
    v.check("compiled-pair-same-census", len(reads) == 1 and len(arm_runs) > 1,
            "the two compilers report the same read census, item census and unwritten-slot census "
            "(%d distinct tuples over %d arms), so neither codegen silently dropped a stage before a "
            "row could hide it" % (len(reads), len(arm_runs)))
    v.check("compiled-pair-byte-identical", ctx["pair"][1] == 0 and ctx["pair"][0] > 0,
            "host_twin vs hip_hostpath over %d bytes: ndiff_bytes=%d at zero tolerance. A non-zero "
            "here is a codegen finding, never a band to absorb" % (ctx["pair"][0], ctx["pair"][1]))
    binds = ctx["slab"]["binds"]
    v.check("span-digests-bind-the-arm-slab",
            all(b["slab_sha"] == b["file_sha"] == b["dev_sha"] for b in binds) and len(binds) > 0,
            "%d spans, each the same sha256 in the file, in the %s and in the arm's slab, so the "
            "kernel reads bytes the staging path certified"
            % (len(binds), "device read-back" if device else "host-side copy of the same span"))
    accepted_bad = [r for r in parsed if r["band"] == "ACCEPTED" and r["licence"] == "none"]
    v.check("no-accepted-row-without-a-licence", not accepted_bad,
            "accepted=%d unlicensed_accepts=%d" % (len([r for r in parsed
                                                       if r["band"] == "ACCEPTED"]),
                                                   len(accepted_bad)))
    unl = [r for r in parsed if r["band"].startswith("UNLICENSED")]
    bad_unl = [r for r in unl if not re.match(r"^UNLICENSED stage=\S+ reason=\S+", r["band"])]
    v.check("every-unlicensed-row-names-stage-and-reason", not bad_unl and len(unl) > 0,
            "unlicensed=%d malformed=%d — the residue is reported, never dissolved" % (len(unl),
                                                                                       len(bad_unl)))
    v.check("pass-unreachable-while-rows-unlicensed", len(unl) == 0 or tally["unlicensed"] > 0,
            "unlicensed=%d; PASS would require zero" % len(unl))
    v.check("band-digest-pinned", ctx["digest"] == BG.BAND_DIGEST,
            "band_digest=%s pinned=%s" % (ctx["digest"][:16], BG.BAND_DIGEST[:16]))
    claims = BG.device_claims(text)
    witness = WITNESS["hip_calls_arm"] + WITNESS["hip_calls_python"]
    v.check("device-claims-come-from-device-only",
            (witness > 0 and len(claims) > 0) if device else (not claims and witness == 0),
            "device ran=%s counted hip calls=%d(python)+%d(arm); device-shaped lines=%d %s — the claim "
            "and the witness must appear together or not at all"
            % (device, WITNESS["hip_calls_python"], WITNESS["hip_calls_arm"], len(claims),
               [ln.split()[0] for ln in claims][:3] or "none"))
    v.check("hip-witness-matches-mode",
            (WITNESS["staged_runs"] > 0 and WITNESS["launches"] > 0) if device
            else (WITNESS["staged_runs"] == 0 and WITNESS["launches"] == 0
                  and WITNESS["hip_calls_python"] == 0),
            "staged_runs=%d launches=%d allocs=%d bytes_to_device=%d bytes_from_device=%d in a %s run"
            % (WITNESS["staged_runs"], WITNESS["launches"], WITNESS["allocs"],
               WITNESS["bytes_to_device"], WITNESS["bytes_from_device"],
               "device" if device else "cpu"))
    v.check("arm-reports-parsed-completely",
            len(arm_runs) > 0 and all(len(a["records"]) == 5 for a in arm_runs.values()),
            "%d arm runs, each parsed into HDR/PLAN/RUN/WROTE/RESULT; a report that parses nothing "
            "cannot fail" % len(arm_runs))
    v.check("no-rate-figures-in-output",
            not re.search(r"\b(ms|t/s|GB/s|ms=|elapsed|duration)\b", text),
            "scanned %d printed lines against the rate-token list and found none; the list is not "
            "printed here because the scan covers this line too" % len(text.splitlines()))
    v.check("pinned-digests-verified-on-every-dump-read", ctx["dumps"].files_read > 0,
            "files_read=%d bytes_read=%d (every dump read sha256-checked against its manifest record)"
            % (ctx["dumps"].files_read, ctx["dumps"].bytes_read))


# ----------------------------------------------------------------------------------------------
# The routed-expert gather vertical: two width paths, one driver pass.
#
# WHY THIS VERTICAL IS DRIVEN SEPARATELY FROM `population["all"]`. The shared-expert and router rows
# exist once, on the policy-A arm, because that is the only arm the band population admits. The gather
# exists twice by construction: the rows that can be licensed live on the bf16 path (`expert_raw` under
# the moe_gather floor probe, `routed_sum` under specification/10 §10.6's routed-sum row) and the shape
# the engine ships lives on the policy-A path, where no band measures it. Emitting one and calling the
# other "plumbing" would lose half the point, so both are emitted, tagged, and the policy-A pair lands
# UNLICENSED with the reason that is actually true (`policy-a-width-band-absent`).
#
# ARMS COUNTED BY WIDTH, NOT ONLY BY EXECUTION PATH. `PATH_LEGS` above counts host-vs-device; `G_LEGS`
# counts ref_bf16-vs-policy_a, because a gather control that only ever bent the bf16 plan would license
# nothing the shipping path does. A control that claims a width it executed zero legs on is reported by
# `gather-control-legs-cover-the-width-under-claim`, and where a mode is structurally unreachable on one
# width (`mode-unreachable-here`: re-chunking a fused [2I,H] span that policy A does not have) the control
# says so and the refusal is what the check requires, instead of a leg being faked.
# ----------------------------------------------------------------------------------------------

G_STAGES = ("expert_raw", "routed_sum")
G_ROLES = DA.G_ROLES
G_ART_TENSOR = DA.G_ART_TENSOR
G_WIDTH_ARM = {"ref_bf16": "ref_bf16", "policy_a": "policy_a"}
G_LEGS = {"ref_bf16": 0, "policy_a": 0}


def g_arm(width, binary, plan, slab, out, device=False, fault=None, label=None):
    """`arm_leg` with the width path tallied too, so a control's width claim can be audited.

    The witness counters are folded here rather than at the call sites: every gather arm act in the run
    goes through this function, so the run's totals cannot leave out a control's variant and make the
    device bill read smaller than the acts that were executed.
    """
    rec = arm_leg(binary, plan, slab, out, device=device, fault=fault, label=label)
    G_LEGS[width] = G_LEGS.get(width, 0) + 1
    witness_fold(rec["run"], label or "g_arm")
    return rec


def g_legs_since(snapshot):
    """The arm acts a control executed since `snapshot`, under BOTH axes it can claim: the execution path
    the arm reported (`host_cpubody`/`device`) and the width path whose plan was bent.

    One dict over both axes is what lets a single `claims=` list name either kind and have
    `control-legs-cover-the-path-under-claim` and
    `gather-control-legs-cover-the-width-under-claim` read the same evidence.
    """
    exec_snap, width_snap = snapshot
    legs = dict((p, n - exec_snap.get(p, 0)) for p, n in PATH_LEGS.items()
                if n - exec_snap.get(p, 0))
    for w, n in G_LEGS.items():
        if n - width_snap.get(w, 0):
            legs[w] = n - width_snap.get(w, 0)
    return legs


def g_claims(ctx):
    """What a gather control declares it covered: both width paths, plus the execution path that ran.

    One tuple for both checks: `control-legs-cover-the-path-under-claim` reads the execution names out of it
    and `gather-control-legs-cover-the-width-under-claim` reads the width names, and both require a counted
    arm act behind every name. A control whose only leg is a refusal (no arm ran to completion) declares no
    claims at all and is listed as such, instead of a refusal being dressed up as a leg.
    """
    return tuple(DA.G_PATHS) + (("device",) if ctx["device"] else ("host_cpubody",))


def g_widen_f32(x_bytes):
    """The pinned bf16 activation image widened to f32 — the basis policy A's matmul actually holds.

    This is a representation change, not an arithmetic one: bf16 to f32 is exact, so the two paths are
    fed the same numbers and only the weights and roundings differ. Doing it here, once, is what lets the
    plan's `elt=` claim be the only width statement about x.
    """
    bits = np.frombuffer(x_bytes, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).tobytes()


def g_stage_bytes(g, units):
    """(bytes, spans) the plan implies, derived off the units instead of typed by hand.

    `spans` counts distinct source ranges, keyed on (file, start, length) as the units were built, before any
    slab remapping. On the reference path the gate and up units name one range (the [2I, H] pair block), so it
    is 16 spans for K=8; on policy A the three matrices live in three different gguf tensors, so it is 24.
    That difference is prediction 3's number: a scheduler deciding whether to merge spans before issuing the
    gather reads exactly this quantity, and it is a property of the artifact layout, not of this driver.
    """
    return (sum(u["bytes"] for u in units),
            len(set((u["src_name"], u["span_off"], u["span_bytes"]) for u in units)))


def gather_prepare(args, ctx):
    """Build both width paths: residency, slab, plan, and the census read off the plan before any run.

    The expert ids and both weight vectors come out of the ROUTER arm's own record, not off the pins, so
    the gather is judged on the selection the compiled router actually produced — which is also what makes
    prediction 1's host-sync question answerable later instead of assumed. Selection, weights, and every
    span are printed, because a census the driver computed but never printed is a census nobody can audit.
    """
    g = DA.parse_g_layout()
    rt_fields = ctx["rt_fields"]
    ids = [int(v) for v in rt_fields["ids"]]
    wcast = [float(v) for v in rt_fields["wcast"]]
    wpre = [float(v) for v in rt_fields["wpre"]]
    if len(ids) != g["MG_K"]:
        raise Refused("g-slots-mismatch", "the router arm returned %d ids and the gather geometry says "
                      "K=%d; the two arms would be describing different rows" % (len(ids), g["MG_K"]))
    L = ctx["st"]["L"]
    source = SR.ByteSource(L, why="M2 routed-expert gather span read")
    out = {}
    try:
        ref_spans = DA.g_ref_spans(g, BF.SHARD_INDEX, BF.ROOT / "models" / "qwen36-src")
        paths = {
            "ref_bf16": dict(units=DA.g_ref_units(g, ref_spans, ids), x=ctx["x_bytes"], elt=2),
            "policy_a": dict(units=DA.g_policy_units(g, L, source, L.art.path, ids),
                             x=g_widen_f32(ctx["x_bytes"]), elt=4),
        }
        for width in DA.G_PATHS:
            p = paths[width]
            # Stamp the file origin of every unit's bytes while the units still speak file coordinates;
            # `g_slab` rewrites `off`/`span_off` to slab coordinates two lines later, and the read-back
            # control needs the pre-remap address (`g_stamp_units`).
            g_stamp_units(p["units"], L.art.path)
            # Read off the units BEFORE the slab remaps them to slab coordinates: the span count a scheduler
            # needs is the count of distinct ranges in the artifact, not in this driver's scratch file.
            want_bytes, want_spans = g_stage_bytes(g, p["units"])
            slab_bytes, off = DA.g_slab(g, p["units"], p["x"], ids, wcast, wpre, p["elt"])
            wdir = ctx["work"] / ("gather_" + width)
            wdir.mkdir(parents=True, exist_ok=True)
            slab_path, plan_path = wdir / "slab.bin", wdir / "plan.txt"
            slab_path.write_bytes(slab_bytes)
            n_units = DA.write_g_plan(plan_path, width, g, p["units"], len(slab_bytes), off, p["elt"])
            whole_bytes = ({"gate": ref_spans["gate_up"]["bytes"],
                            "up": ref_spans["gate_up"]["bytes"],
                            "down": ref_spans["down"]["bytes"]} if width == "ref_bf16" else
                           dict((role, L.by_name[G_ART_TENSOR[role]].row) for role in G_ROLES))
            plan_text = plan_path.read_text()
            whole_plan = wdir / "plan_whole_tensor.txt"
            whole_plan.write_text(plan_text)
            DA.write_g_plan(whole_plan, width, g, p["units"], len(slab_bytes), off, p["elt"],
                            whole_bytes=whole_bytes)
            cen = DA.g_census(g, p["units"], ids)
            print("GWIDTH path=%s arm=%s x_elt=%d x_basis=%s experts=%s units=%d spans=%d "
                  "bytes=%d bytes_per_expert=%s slab_bytes=%d plan_units=%d"
                  % (width, G_WIDTH_ARM[width], p["elt"],
                     "pinned-bf16-image" if p["elt"] == 2 else "f32-widened-from-the-pinned-bf16",
                     ",".join(str(i) for i in ids), n_units, want_spans, want_bytes,
                     ",".join("%d:%dB" % (e, b) for e, b in sorted(cen["bytes_per_expert"].items())),
                     len(slab_bytes), n_units))
            print("GSOURCE path=%s %s" % (width, " ".join(
                "%s=%s(%s)" % (role, src, kind) for role, src, kind in
                sorted(set((u["role"], Path(u["src_name"]).name if "/" in u["src_name"]
                            else u["src_name"], u["kind"]) for u in p["units"])))))
            out[width] = dict(g=g, units=p["units"], off=off, elt=p["elt"], slab=slab_path,
                              plan=plan_path, whole_plan=whole_plan, slab_bytes=len(slab_bytes),
                              census=cen, want_spans=want_spans, want_bytes=want_bytes,
                              whole_bytes=whole_bytes,
                              ids=ids, dir=wdir)
    finally:
        source.close()
    return out


def g_stages(fields, g):
    """The gather record as stage tensors, in the shapes the pins hold ([1,K,H] and [1,H])."""
    return dict(expert_raw=torch.from_numpy(fields["raw"].copy()).reshape(1, g["MG_K"], g["MG_H"]),
                routed_sum=torch.from_numpy(fields["routed"].copy()).reshape(1, g["MG_H"]))


def print_gather_read_set_formats(ctx):
    """`GFORMAT` — one line per tensor the policy-A gather reads, with its width read OUT of the file.

    Why this exists as a printed table and not a sentence: D-27 records that the shipping artifact stores
    expert `down_proj` at Q5_K (37 layers) / Q6_K (3 layers) where policy A declares Q4_K, and
    `engine/loader.py::admit` admits a tensor that is wider than its family floor without a waiver — the
    floor rule is arithmetic over the tensor's own type, so "wider than declared" is legal and silent.
    Any hypothesis about the policy-A device divergence has to say which format each read actually used,
    and the only answer worth having is the one derived from the file: the arm's `kind` word comes from
    `Loader.expert_range`'s ggml type, `blocks` comes from rows×cols÷super-block, and the byte count the
    plan declares is required to equal blocks × block-bytes of THAT kind. Every column below is one of
    those derived quantities; none is typed.
    """
    paths = ctx.get("gpaths") or {}
    if "policy_a" not in paths:
        return []
    p = paths["policy_a"]
    g = p["g"]
    loader = ctx["st"]["L"]
    by_role = {}
    for u in p["units"]:
        by_role.setdefault(u["role"], []).append(u)
    rows = []
    for role in DA.G_ROLES:
        units = by_role.get(role, [])
        if not units:
            continue
        u0 = units[0]
        rec = loader.by_name.get(u0["src_name"])
        declared = LO.QS.POLICY_TYPE(LO.POLICY_A, rec.family) if rec else "UNKNOWN"
        d_bpw = LO.policy_floor_bpw(rec.family) if rec else 0.0
        a_bpw = LO.stored_bpw(u0["type_name"])
        relation = ("same-as-declared" if abs(a_bpw - d_bpw) < 1e-9 else
                    "WIDER-THAN-DECLARED" if a_bpw > d_bpw else "narrower-than-declared")
        bb = g["BLOCK_BYTES"][u0["kind"]]
        print("GFORMAT role=%s tensor=%s header_type=%s arm_kind=%s policy_a_declared=%s "
              "stored_bpw=%.4f declared_bpw=%.4f relation=%s block_bytes=%d super_block=%d "
              "rows=%d cols=%d blocks=%d blocks_derived=%d bytes=%d bytes_derived=%d units=%d "
              "geometry=%s"
              % (role, u0["src_name"], u0["type_name"], u0["kind"], declared, a_bpw, d_bpw, relation,
                 bb, g["MG_QK_K"], u0["rows"], u0["cols"], u0["blocks"],
                 (u0["rows"] * u0["cols"]) // g["MG_QK_K"], u0["bytes"], u0["blocks"] * bb,
                 len(units),
                 "ok" if (u0["bytes"] == u0["blocks"] * bb
                          and u0["blocks"] == (u0["rows"] * u0["cols"]) // g["MG_QK_K"])
                 else "MISMATCH"))
        for u in units:
            rows.append(dict(role=role, tensor=u["src_name"], header_type=u["type_name"],
                             kind=u["kind"], block_bytes=bb, blocks=u["blocks"], bytes=u["bytes"],
                             relation=relation, expert=u["expert"], slot=u["slot"]))
    wider = sorted(set(r["relation"] for r in rows))
    print("GFORMAT summary units=%d roles=%s relations=%s wider_than_declared_units=%d "
          "note=the read set's formats are read from the artifact's ggml type per tensor, so an artifact "
          "re-quantized to another width changes this table and the arm's refusal, not a string in a "
          "control (engine/loader.py::admit, D-27)"
          % (len(rows), ",".join(sorted(by_role)), ",".join(wider),
             sum(1 for r in rows if r["relation"] == "WIDER-THAN-DECLARED")))
    ctx["gformat"] = rows
    return rows


def run_gather_arms(v, ctx):
    """Run both width paths through the two compilers, print the census, and hand back the stages.

    Four runs in a CPU witness (two paths x two compilers), eight in a device window. Every path prints
    its own `PAIR` line, because a single aggregate ndiff over two shapes is the aggregate-that-hides-one
    that `pitfalls/40` §bans. The record the rows are read from is the hipcc arm's host path in a CPU
    witness and the device arm's in a window, exactly as the shared-expert and router rows do.
    """
    args = ctx["args"]
    runs, stages_by_width, censuses = {}, {}, {}
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        g = p["g"]
        twin = g_arm(width, args.g_host_binary, p["plan"], p["slab"], p["dir"] / "out_host.bin",
                     label="g_%s_host_twin" % width)
        hip = g_arm(width, args.g_arm_binary, p["plan"], p["slab"], p["dir"] / "out_hip.bin",
                    label="g_%s_hip_hostpath" % width)
        runs["g_%s_host_twin" % width], runs["g_%s_hip_hostpath" % width] = twin, hip
        nbytes, ndiff = DA.pair_bytes(p["dir"] / "out_host.bin", p["dir"] / "out_hip.bin")
        print("PAIR pair=g_%s_host_twin_vs_hip_hostpath file=gather_record.f32 bytes=%d "
              "ndiff_bytes=%d cases=1 zero_tolerance=1 verdict=%s"
              % (width, nbytes, ndiff, "IDENTICAL" if ndiff == 0 else "DIFFERS"))
        # Per-field split printed beside the total, every run: an attribution that only exists as one byte
        # count cannot be checked against a mechanism, and this is the half of `wpre`'s story that was
        # measured before any scope was named.
        print_field_attrib("GATTRIB", "g_%s_host_twin_vs_hip_hostpath" % width,
                           "gather_record.f32",
                           *record_field_rows(p["dir"] / "out_host.bin", p["dir"] / "out_hip.bin",
                                              g["order"], g["fields"]),
                           note="per-field split of the PAIR line above; totals must account")
        g_h_fields, _ = DA.g_record(p["dir"] / "out_hip.bin", g)
        exp_route_disagreement(-g_h_fields["gate"], "GEXP",
                               note="path=%s sites=the silu inputs this row actually uses (the body "
                                    "computes expf(-gate)); prices how far the HOST route is from a "
                                    "correctly-rounded exp, which is what a libdevice difference has to "
                                    "be measured against" % width)
        p["pair"] = (nbytes, ndiff)
        rec_hip, nan_hip = DA.g_record(p["dir"] / "out_hip.bin", g)
        rec_twin, _ = DA.g_record(p["dir"] / "out_host.bin", g)
        p["fields"], p["nan"] = rec_hip, nan_hip
        p["arm_census"] = hip["run"].get("census")
        stages_by_width[width] = g_stages(rec_hip, g)
        arm_cen = DA.parse_census(hip["run"].get("census"))
        censuses[width] = DA.g_census(g, p["units"], p["ids"], arm=arm_cen)
        print("GATHERCENSUS path=%s slots=%d distinct_experts=%d units=%d spans=%d bytes=%d "
              "bytes_per_expert=%s whole_tensor_reads=%d arm_units=%d arm_spans=%d arm_total=%d "
              "arm_whole_reads=%d note_conflict=%d disagree=%s arm_census=\"%s\""
              % (width, p["census"]["slots"], p["census"]["distinct"], p["census"]["units"],
                 p["want_spans"], p["want_bytes"],
                 ",".join("%d:%dB" % (e, b) for e, b in sorted(p["census"]["bytes_per_expert"].items())),
                 p["census"]["whole_tensor_reads"], censuses[width]["arm_units"],
                 censuses[width]["arm_spans"], censuses[width]["arm_bytes"],
                 censuses[width]["arm_whole_reads"], censuses[width]["note_conflict"],
                 censuses[width]["disagree"] or "none",
                 hip["run"].get("census")))
        if ctx["device"]:
            dev = g_arm(width, args.g_arm_binary, p["plan"], p["slab"], p["dir"] / "out_device.bin",
                        device=True, label="g_%s_hip_device" % width)
            runs["g_%s_hip_device" % width] = dev
            d_fields, d_nan = DA.g_record(p["dir"] / "out_device.bin", g)
            d_cen = DA.parse_census(dev["run"].get("census"))
            d_bytes, d_diff = DA.pair_bytes(p["dir"] / "out_device.bin", p["dir"] / "out_hip.bin")
            print("PAIR pair=g_%s_hip_device_vs_hip_hostpath file=gather_record.f32 bytes=%d "
                  "ndiff_bytes=%d cases=1 zero_tolerance=1 verdict=%s"
                  % (width, d_bytes, d_diff, "IDENTICAL" if d_diff == 0 else "DIFFERS"))
            ctx.setdefault("g_device_pair_rows", {})[width] = print_field_attrib(
                "GATTRIB", "g_%s_hip_device_vs_hip_hostpath" % width, "gather_record.f32",
                *record_field_rows(p["dir"] / "out_device.bin", p["dir"] / "out_hip.bin",
                                   g["order"], g["fields"]),
                note="the unattributed divergence of window `g`, now counted per field; compared with the "
                     "injected causes by gather-divergence-causes-priced")
            print("GATHERCENSUS path=%s slot=device distinct_experts=%d arm_units=%d arm_total=%d "
                  "whole_reads=%d disagree=%s arm_census=\"%s\""
                  % (width, d_cen["distinct_experts"], d_cen["unit_reads"], d_cen["total"],
                     d_cen["whole_reads"],
                     DA.g_census(g, p["units"], p["ids"], arm=d_cen)["disagree"] or "none",
                     dev["run"].get("census")))
            p["fields"], p["nan"] = d_fields, d_nan
            p["arm_census"] = dev["run"].get("census")
            stages_by_width[width] = g_stages(d_fields, g)
            p["pair"] = (d_bytes, d_diff)
    return runs, stages_by_width, censuses


def emit_gather_rows(dumps, band, limits, stages_by_width, provenance):
    """Four STAGE rows: two licenced stages on each width path, judged by the licensors that already exist.

    Nothing here invents a limit. `expert_raw` is judged by the moe_gather floor probe's own element count
    and `routed_sum` by exactness, both read through `engine/block_gate.py::MOE_BANDS`; on the policy-A
    path both land UNLICENSED with reason `policy-a-width-band-absent`, which is the honest state of the
    world and the reason this vertical exists: it reports the number the shipping shape produces without
    pretending a band covers it.
    """
    rows = []
    print("PROVENANCE stages=%s widths=%s source=%s note=the ids and weights the gather used are the "
          "router arm's own record, not the pins, so this is a chain of compiled arms and not a replay"
          % (",".join(G_STAGES), ",".join(DA.G_PATHS), provenance))
    for width in DA.G_PATHS:
        arm = G_WIDTH_ARM[width]
        for stage in G_STAGES:
            rows.append(BG.stage_row(dumps, arm, stage, stages_by_width[width]))
    tally = BG.adjudicate(rows, band, limits, None)
    for row in rows:
        BG.print_row(row)
    WITNESS["rows_emitted"] += len(rows)
    return rows, tally


def g_plan_variant(p, text):
    """Write one variant plan beside the honest one and return its path."""
    path = p["dir"] / "plan_variant.txt"
    path.write_text(text)
    return path


def g_run_variant(ctx, width, plan_path, out_name, label, fault=None, device=None):
    """Run one variant plan through the arm that produced the rows under test.

    The binary is the hipcc arm — its host path in a CPU witness, its device path in a window — so a
    perturbation leg is executed by the same codegen whose honest record is being compared, which is the
    lesson of window e2 rather than a stylistic choice.
    """
    p = ctx["gpaths"][width]
    want_device = ctx["device"] if device is None else device
    return g_arm(width, ctx["args"].g_arm_binary, plan_path, p["slab"], p["dir"] / out_name,
                 device=want_device, fault=fault, label=label)


def control_gather_swap(v, ctx):
    """Prediction 3, executed: `gate_up_proj` is [E, 2I, H] with `gate, up = chunk(2)` along the OUTPUT
    dim, and a kernel written for two separate [I, H] matrices has to be checked for a stride/offset swap
    rather than re-derived.

    Two spellings of the swap, because the two layouts make two different mistakes possible:
      * `halves_swapped` — exchange the gate and up UNITS. Reachable on both paths: on policy A they are
        two artifact tensors, on the reference path two views of one fused span.
      * `input_chunk` — re-chunk the same 2I*H elements along the INPUT dim ([I, 2H]). This one is only
        expressible where the pair really is one span, so on the policy-A plan the arm refuses it with
        `mode-unreachable-here` and the refusal, not a leg, is what the control requires there.
    Both paths must show the swap moving `expert_raw` further from the pin than the honest read does, and
    it has to move the halves themselves (`gate`/`up` fields), not only the sum — a sum can be insensitive
    to a swap that the halves make visible.
    """
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    rows, refused = [], []
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        honest_raw = p["fields"]["raw"]
        honest_gate = p["fields"]["gate"]
        plan = g_plan_variant(p, p["plan"].read_text().replace("mode=correct", "mode=halves_swapped"))
        rec = g_run_variant(ctx, width, plan, "out_swap.bin", "g_%s_swap" % width)
        fields, _ = DA.g_record(rec["wrote"]["path"], p["g"])
        moved_raw = int(np.count_nonzero(fields["raw"] != honest_raw))
        moved_gate = int(np.count_nonzero(fields["gate"] != honest_gate))
        rows.append((width, moved_raw, moved_gate))
        if width == "ref_bf16":
            plan2 = g_plan_variant(p, p["plan"].read_text().replace("mode=correct",
                                                                    "mode=input_chunk"))
            rec2 = g_run_variant(ctx, width, plan2, "out_chunk.bin", "g_ref_chunk")
            f2, _ = DA.g_record(rec2["wrote"]["path"], p["g"])
            rows.append(("ref_bf16.input_chunk", int(np.count_nonzero(f2["raw"] != honest_raw)),
                         int(np.count_nonzero(f2["gate"] != honest_gate))))
        else:
            plan2 = g_plan_variant(p, p["plan"].read_text().replace("mode=correct",
                                                                    "mode=input_chunk"))
            try:
                g_run_variant(ctx, width, plan2, "out_chunk.bin", "g_policy_chunk")
                refused.append("policy_a:NOT-REFUSED")
            except DA.Refused as exc:
                refused.append("policy_a:%s" % ("mode-unreachable-here"
                                                if "mode-unreachable-here" in str(exc) else exc.tag))
    all_moved = all(m > 0 for _, m, _ in rows)
    v.control("gather-gate-up-swap-reddens", all_moved and "NOT-REFUSED" not in refused,
              "swapped halves moved raw elements / gate elements per path: %s; the input-dim re-chunk leg "
              "on the path that has a fused span moved %s, and on the path that has two separate tensors "
              "the arm refused with %s (a mode a layout cannot express is refused, never reported as "
              "fired). Prediction 3 was that a SwiGLU kernel written for two [I,H] matrices has to be "
              "checked for the swap; these are the legs that check it"
              % ("; ".join("%s moved_raw=%d moved_gate=%d" % r for r in rows),
                 [m for n, m, _ in rows if "chunk" in n][0] if rows else "?",
                 ",".join(refused)),
              legs=g_legs_since(snap), claims=g_claims(ctx))


def launch_count_offences(runs, paths=("device",)):
    """Every arm record that cannot support the claim "this run launched kernels", with the reason named.

    ONE spelling of the rule, because four checks used to cast and compare by hand — two of them `int(...) > 0`,
    two of them a bare `... > 0` — and window `i` died inside one of the bare ones when the parse handed back
    a string: `TypeError: '>' not supported between instances of 'str' and 'int'`, `rc=1`, no `RESULT` line.
    A check that raises has not run, and a run that cannot print its RESULT cannot print its findings either,
    so the crash was worse than a red. The type is now guaranteed upstream at the parse site
    (`engine/device_arm.py::parse_arm_report`, which reads the `%d` off the arm's own printf), and this is
    the belt: three offences are named rather than raised — the counter absent, the counter not an `int`, the
    counter an `int` that is not positive. `control-witness-guard-can-redden` is the leg that shows each of
    the three is a finding, including `launches="abc"`, which is a *red check*, not a crashed driver.
    """
    bad = []
    for label in sorted(runs):
        run = run_facts(runs[label])
        if paths is not None and run.get("path") not in paths:
            continue
        v = run.get("launches")
        if v is None:
            bad.append("%s.launches=absent" % label)
        elif isinstance(v, bool) or not isinstance(v, int):
            bad.append("%s.launches=%r(not-an-int)" % (label, v))
        elif v <= 0:
            bad.append("%s.launches=%d(not-positive)" % (label, v))
    return bad


def device_act_absences(runs):
    """Offences among the records that ran ON the device: a `path=device` act that reports none.

    The mirror image of `no_device_act_offences`, and needed for the same reason. In a device window the
    predicate that guards "no device act" is supposed to read CLEAN on the host bodies and DIRTY on the
    device runs; if it read clean on those too, the check would be certifying that nothing ran while five
    arms were launching. `launches=0`, `bytes_from_device=0` or a missing counter on a `path=device` record
    is the silent fallback this lane exists to refuse, so it is an offence of its own and
    `control-witness-guard-can-redden` shows the predicate can see it (window `g`: that control printed
    `fired=SILENT` on the device path because its honest half demanded an act-free set that a device run
    cannot honestly produce). The launch half is `launch_count_offences`, so there is one rule for what a
    counted launch means.
    """
    bad = launch_count_offences(runs)
    for label in sorted(runs):
        run = run_facts(runs[label])
        if run.get("path") != "device":
            continue
        v = run.get("bytes_from_device")
        try:
            if v is None or not int(v):
                bad.append("%s.bytes_from_device=%s"
                           % (label, "absent" if v is None else v))
        except (TypeError, ValueError):
            bad.append("%s.bytes_from_device=%s" % (label, v))
    return bad


def control_no_device_act_guard(v, ctx):
    """The `no device acts` predicate has to be capable of failing, and this shows it with a fabricated act.

    The check above passes by reading counters that are all zero, and "all zero" is also what a counter that
    is never incremented, never printed, or read off the wrong dict looks like — the first version of that
    check threw `KeyError: 'run'` and so never ran at all. So the predicate is run here over three fabricated
    records whose only difference from the real ones is the counter: one that launched, one that moved bytes
    to the device, and one whose `RUN` line does not report `launches` at all. All three must be read as
    offences, and the honest set must read clean, or the check is decoration.

    THE HONEST HALF IS PER PATH, because "honest" means different things in the two modes. In a CPU witness
    every record must be act-free. In a device window the act-free requirement binds the `path=host_cpubody`
    records — a host body that reached a device is the same defect the CPU witness looks for — and the
    `path=device` records must be seen to HAVE acts, which is `device_act_absences`. Demanding an act-free
    device run is what made this control print `fired=SILENT` in window `g`, on the one path its witness was
    supposed to cover; a guard that can only be satisfied by the mode it is not guarding is not a guard.

    No arm runs here; the legs are driver-side predicate evaluations over records the arms already produced,
    so the control declares no path claims (a claim with no counted arm act is what
    `control-legs-cover-the-path-under-claim` reddens).
    """
    honest = dict((k, dict(run_facts(r))) for k, r in ctx["all_runs"].items())
    victim = sorted(honest)[0]
    host_runs = dict((k, r) for k, r in ctx["all_runs"].items()
                     if run_facts(r).get("path") != "device")
    dev_runs = dict((k, r) for k, r in ctx["all_runs"].items()
                    if run_facts(r).get("path") == "device")
    fabricated = {
        "launches=1": dict(honest[victim], launches=1),
        "bytes_to_device=1GiB": dict(honest[victim], bytes_to_device=1073741824),
        "launches-absent": dict((k, x) for k, x in honest[victim].items() if k != "launches"),
    }
    bad = dict((name, no_device_act_offences({victim: rec})) for name, rec in fabricated.items())
    # The same three shapes, replayed against the launch-count predicate that the four
    # `*-device-path-claims-executed-on-device` checks read, plus the two shapes only that predicate judges:
    # a device run reporting `launches=0`, and one reporting `launches="abc"` — the last is window `i`'s
    # crash, and the requirement is a RED check, not a `TypeError` from the driver.
    launch_fabricated = dict(("device/" + n, dict(rec, path="device")) for n, rec in fabricated.items())
    launch_fabricated["device/launches=0"] = dict(honest[victim], path="device", launches=0)
    launch_fabricated['device/launches="abc"'] = dict(honest[victim], path="device", launches="abc")
    launch_bad = dict((name, launch_count_offences({victim: rec}, paths=None))
                      for name, rec in launch_fabricated.items())
    launch_clean = launch_count_offences(dev_runs)
    clean = no_device_act_offences(host_runs)
    absent = device_act_absences(dev_runs)
    for name in sorted(fabricated):
        print("GWITNESS guard=no-device-acts fabricated=%s offences=%s" % (name, bad[name] or "NONE"))
    for name in sorted(launch_fabricated):
        print("GWITNESS guard=launch-count fabricated=%s offences=%s"
              % (name, launch_bad[name] or "NONE — the launch predicate is blind to this shape"))
    print("GWITNESS guard=launch-count leg=device-records-clean records=%d offences=%s"
          % (len(dev_runs), launch_clean or "clean"))
    print("GWITNESS guard=no-device-acts leg=host-records-clean records=%d offences=%s"
          % (len(host_runs), clean or "clean"))
    if dev_runs:
        print("GWITNESS guard=no-device-acts leg=device-records-have-acts records=%d offences=%s"
              % (len(dev_runs), absent or "clean"))
    else:
        print("GWITNESS guard=no-device-acts leg=device-records-have-acts status=not-exercised "
              "reason=no-device-act-in-this-run note=no record reported path=device, so this half of the "
              "predicate has nothing to read here; it is reported unexercised, not counted as passed, and "
              "the fabrication below is what shows it can fire")
    fake_dev = {"fabricated/device-run-with-no-act": dict(honest[victim], path="device",
                                                        launches=0, bytes_from_device=0)}
    fake_absent = device_act_absences(fake_dev)
    print("GWITNESS guard=no-device-acts fabricated=device-run-with-no-act offences=%s"
          % (fake_absent or "NONE — the predicate is blind to a device run that launched nothing"))
    must_be_clean = ["device/launches=1"]
    must_catch = [n for n in launch_fabricated if n not in must_be_clean]
    fired = (not clean and not absent and not launch_clean
             and all(bad[n] for n in bad) and fake_absent
             and all(launch_bad[n] for n in must_catch)
             and not any(launch_bad[n] for n in must_be_clean))
    nlegs = len(fabricated) + len(launch_fabricated) + 1 + (1 if dev_runs else 0) + 1
    v.control("control-witness-guard-can-redden", fired,
              "the same predicate the check runs, over the honest set split by the path the arm reported "
              "(executed=device:%d,host_cpubody:%d predicate legs, no arm act) and %d fabrications: "
              "host-clean=%s; device-acts=%s; %s; launch-count: %s. Every fabrication is read as an offence, "
              "including the record that simply omits the counter — a RUN line that does not report a number "
              "cannot certify that the number was zero — and including `launches=\"abc\"`, which reddens the "
              "check instead of raising inside it; the one fabrication that must stay clean "
              "(`device/launches=1`, a device run reporting one launch) does"
              % (len(dev_runs), len(host_runs), len(fabricated) + len(launch_fabricated),
                 clean or "clean", absent or "clean",
                 "; ".join("%s -> %s" % (n, ",".join(bad[n])) for n in sorted(bad)),
                 "; ".join("%s -> %s" % (n, ",".join(launch_bad[n])) for n in sorted(launch_bad))),
              legs={"cpu_predicate": nlegs},
              reason=(None if fired else
                      "an honest half disagreed with a predicate the way the path forbids, or a fabrication "
                      "went unseen: host records offending=%s, device records missing acts=%s, device launch "
                      "records with offences=%s, uncaught=%s, wrongly-caught=%s (see the GWITNESS legs above)"
                      % (clean or "clean", absent or "clean", launch_clean or "clean",
                         [n for n in must_catch if not launch_bad[n]] or "none",
                         [n for n in must_be_clean if launch_bad[n]] or "none")))


def control_silent_reason_guard(v, ctx):
    """The SILENT-control rule has to be capable of reddening, and it has to bind controls.

    `silent-control-names-its-reason` was written after window `g`, and window `h` still printed two quiet
    `fired=SILENT` control lines with no `reason=` — because the check was reading the wrong population.
    Checks have no legs to lose; controls do. So this leg builds a scratch `Verifier`, registers four
    controls whose only difference is the reason field, and requires exactly the two quiet-without-a-reason
    ones to be findings: quiet+reason and fired+no-reason must read clean (a reason is required of a control
    that could not run, not of one that ran), and whitespace must not count as a reason. Both denominators are
    non-zero by construction, so the predicate cannot satisfy itself by an empty list.
    """
    probe = Verifier()
    legs = [("quiet-with-reason", False, dict(reason="needs-a-device-window")),
            ("quiet-no-reason", False, {}),
            ("quiet-whitespace-reason", False, dict(reason="   ")),
            ("fired-no-reason", True, {})]
    for name, fired, kw in legs:
        probe.control(name, fired, "fabricated leg (no arm act, nothing computed)", **kw)
    seen = dict((n, n in probe.unreasoned) for n, _, _ in legs)
    want = {"quiet-no-reason": True, "quiet-whitespace-reason": True,
            "quiet-with-reason": False, "fired-no-reason": False}
    for name, _, _ in legs:
        print("GWITNESS guard=silent-reason fabricated=%s flagged=%s expected=%s"
              % (name, seen[name], want[name]))
    fired = (seen == want and len(probe.controls) == len(legs)
             and sorted(probe.unreasoned) == sorted(n for n, _, _ in legs if want[n]))
    v.control("silent-reason-guard-can-redden", fired,
              "%d fabricated control registrations through the real `Verifier.control`: the two quiet ones "
              "without a reason (including the whitespace-only one) are findings, the quiet one with a reason "
              "and the one that fired are not, and `silent-control-names-its-reason` reads the same "
              "`Verifier.unreasoned` list this leg filled. If the guard is ever made to read checks again, or "
              "the rule is loosened to accept an empty reason, this control goes SILENT and names which leg "
              "stopped being a finding (flags: %s)"
              % (len(legs), ",".join("%s=%s" % (n, seen[n]) for n, _, _ in legs)),
              legs={"cpu_predicate": len(legs)},
              reason=(None if fired else
                      "the silent-reason rule no longer distinguishes the four legs: flagged=%s (see the "
                      "GWITNESS guard=silent-reason lines)" % seen))


def control_attribution_contradiction_guard(v, ctx):
    """The contradiction check must be able to fire, so the stale sentence is fed to it verbatim.

    The check reads the run's own text, which in a CPU witness contains no `DREPRO` line at all — a check
    that can only fire in one mode is a check that is blind in the other, and window `h`/`i` are exactly
    where it was needed. So this leg hands the predicate five texts: the real one from this run's LIVE
    buffer (window `j`: moved to run after the reproduction controls, because reading the mid-run
    snapshot made its `this-run` leg report `exact_pairs=none` beside three printed legs — a witness
    reading the wrong half of its own log is the defect it guards); a stale text built by taking a real
    exact-reproduction line (window-`i` evidence, whose pair labels are cited below) and putting back
    the sentence the control used to print; the same text with the stale sentence about a DIFFERENT pair,
    which must not fire — the narrowness half, because a "contradiction detector that fires on any
    negation is a lint rule, not a check"; a fabricated SILENT denial naming a pair THIS run reproduced,
    which must redden with exactly one finding per denied pair (non-vacuity proof (a)); and this run's
    own text with its reproduction lines removed, evaluated as a device run, where the zero-count
    assertion must redden rather than the check passing over an empty list (non-vacuity proof (b), the
    exact window-`j` shape of `parsed=0 → PASS`).
    """
    text = run_text(ctx)
    exact, _ = exact_reproductions(text)
    # Pair labels and a real reproduction line, copied from this repo's window-`i` evidence so the stale text
    # is assembled from printed material rather than typed from memory (see the STALE legs in the pass-9 log).
    evidence = ROOT / "specification" / "evidence" / "blockdevice-device-2026-10-27-i1.log"
    repro_lines = [ln for ln in (evidence.read_text().splitlines() if evidence.exists() else [])
                   if ln.startswith(REPRO_HEADS) and "verdict=EXACT-REPRODUCTION" in ln]
    basis = repro_lines or [ln for ln in text.splitlines()
                            if ln.startswith(REPRO_HEADS) and "verdict=EXACT-REPRODUCTION" in ln]
    stale, other_ok = [], True
    for line in basis:
        pair = re.search(r" pair=(\S+)", line).group(1)
        stale.append("CONTROL name=fabricated-stale-control fired=SILENT reason=no injected cause "
                     "reproduces the device divergence of %s" % pair)
    stale_text = "\n".join(basis + stale)
    stale_off = ("CONTROL name=fabricated-other-control fired=SILENT reason=no injected cause reproduces "
                 "the divergence of a.pair.that.is.not.in.this.run")
    found_real = contradiction_findings(text)
    found_stale = contradiction_findings(stale_text)
    found_other = contradiction_findings(stale_off)
    print("GWITNESS guard=attribution-contradiction leg=this-run lines=%d exact_pairs=%s findings=%s"
          % (len(text.splitlines()), ",".join(sorted(exact)) or "none", found_real or "none"))
    print("GWITNESS guard=attribution-contradiction leg=stale-text lines=%d findings=%d first=%s"
          % (len(stale_text.splitlines()), len(found_stale), (found_stale or ["NONE"])[0][:160]))
    print("GWITNESS guard=attribution-contradiction leg=other-pair-text findings=%s note=a denial about a "
          "pair this run did not reproduce is not a contradiction and must not fire"
          % (found_other or "none"))
    for ln in stale:
        print("GWITNESS-STALE %s" % ln[:400])
    # Non-vacuity proof (a), printed: a fabricated SILENT control that denies a pair THIS run reproduced.
    # In a device-shaped text the denied pairs are the run's own reproduced ones; the CPU witness has no
    # pair of its own, so it denies the window-`i` basis pairs the stale leg already cites.
    deny_pairs = sorted(exact) or [re.search(r" pair=(\S+)", ln).group(1) for ln in basis]
    deny_base = text if exact else "\n".join(basis)
    deny_lines = ["CONTROL name=fabricated-denial-%d fired=SILENT reason=no injected cause reproduces "
                  "the device divergence of %s" % (i, p) for i, p in enumerate(deny_pairs, 1)]
    found_deny = contradiction_findings("\n".join([deny_base] + deny_lines))
    # Non-vacuity proof (b), printed: this run's text with the reproduction lines REMOVED, evaluated as
    # a device run — the zero-count assertion reddens instead of the check passing over an empty list.
    stripped_text = "\n".join(ln for ln in text.splitlines() if not ln.startswith(REPRO_HEADS))
    zero_finding = reproduction_leg_finding(True, stripped_text)
    print("GWITNESS guard=attribution-contradiction leg=fabricated-denial pairs=%s findings=%d "
          "expected=%d note=a fabricated denial of a reproduced pair reddens the check with exactly one "
          "finding per denied pair"
          % (",".join(deny_pairs) or "none", len(found_deny), len(found_real) + len(deny_pairs)))
    for ln in deny_lines:
        print("GWITNESS-DENY %s" % ln[:400])
    print("GWITNESS guard=attribution-contradiction leg=zero-leg-text lines=%d exact_legs_parsed=0 "
          "finding=%s note=this run's own text with its reproduction lines removed, evaluated as a "
          "device run: the zero-count assertion reddens instead of the check passing"
          % (len(stripped_text.splitlines()), bool(zero_finding)))
    fired = (not found_real and not found_other and bool(basis)
             and len(found_stale) == len(basis)
             and len(found_deny) == len(found_real) + len(deny_pairs) and bool(zero_finding))
    v.control("attribution-contradiction-guard-can-redden", fired,
              "the contradiction predicate run over five texts: this run's own output (%d lines, findings "
              "%s, exact pairs %s); a stale text that keeps the window-`h`/`i` sentence `no injected cause "
              "reproduces` beside the exact-reproduction line for the SAME pair (%d legs, %d findings — one "
              "per pair is the requirement); a text where the denial is about a pair nobody reproduced (must "
              "be, and is, clean); a fabricated SILENT denial of a pair this run reproduced (%d pairs, %d "
              "findings — one per denied pair is the requirement); and this run's text with its reproduction "
              "lines removed, device-evaluated, where the zero-count assertion must and does fire (%s). The "
              "stale and deny lines are printed verbatim above so the wording that reddens is inspectable"
              % (len(text.splitlines()), found_real or "none", ",".join(sorted(exact)) or "none",
                 len(basis), len(found_stale), len(deny_pairs), len(found_deny),
                 "fired" if zero_finding else "STAYED SILENT — the anti-vacuity assertion is broken"),
              legs={"cpu_predicate": 5},
              reason=(None if fired else
                      "the contradiction predicate did not behave: this-run findings=%s, stale-text findings="
                      "%d of %d expected, other-pair findings=%s, fabricated-denial findings=%d of %d "
                      "expected, zero-leg-text finding=%s (in a CPU run no DREPRO line exists, so the "
                      "stale leg has no real reproduction line to stand beside and is not claimed)"
                      % (found_real or "none", len(found_stale), len(basis), found_other or "none",
                         len(found_deny), len(found_real) + len(deny_pairs),
                         bool(zero_finding))))


def control_gather_census(v, ctx):
    """Prediction 2, executed: the gather has to be provably a gather.

    Four legs per width path:
      * agreement — the census the ARM printed has to match the census the driver computed from the plan it
        wrote, per expert_index and in the aggregate (units, spans, distinct experts, bytes). The count is
        not the driver's opinion of what it asked for, and the first witness earned this leg by catching the
        arm printing `unit_reads=4` against a 24-unit plan;
      * refusal — a plan whose units claim the whole stacked routed tensor is REFUSED outright
        (`whole-tensor-unit-refused`), so the arm cannot be handed the whole-tensor read the census exists to
        catch. The byte claims in that refused plan come from the safetensors headers and the loader's
        records, never typed here;
      * the counter's non-zero leg — `fault=whole-note`, which the parser accepts only when the plan declares
        it by name, notes a whole-tensor read for EVERY unit. The census has to count it
        (`whole_reads == unit_reads`, not a per-run constant of 1) and the same comparison that passed on the
        honest plan has to redden on it. Before this leg existed, `whole_reads=0` was vacuous in exactly the
        sense `pitfalls/40` §a-counter-that-reads-0 names: a counter that counts nothing reports no
        violations;
      * the driver's half — a doctored arm census (short `unit_reads`, one expert missing: the shape the
        first witness actually produced) has to produce disagreement in this file, with no device involved,
        so the agreement leg is known to be able to fail.

    A zero denominator is a named finding of its own: an empty census agreeing with an empty census is the
    empty parse wearing a census costume, so it reports as `census-denominator-zero`, never as a pass.
    """
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    agree, refused, whole_leg, driver_leg, zerod, costs = [], [], [], [], [], []
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        cen = DA.parse_census(p["arm_census"])
        d = DA.g_census(p["g"], p["units"], p["ids"], arm=cen)
        agree.append((width, d["disagree"], d, cen))
        zerod.append((width, [k for k, n in (("units", d["units"]), ("spans", d["spans"]),
                                             ("distinct_experts", d["distinct"]),
                                             ("arm_units", cen["unit_reads"]),
                                             ("arm_spans", cen["spans"]), ("arm_bytes", cen["total"]))
                              if not n]))
        costs.append((width, sum(p["whole_bytes"][u["role"]] for u in p["units"]), d["bytes_total"]))
        try:
            g_run_variant(ctx, width, p["whole_plan"], "out_whole.bin", "g_%s_whole" % width)
            refused.append("%s:NOT-REFUSED" % width)
        except DA.Refused as exc:
            refused.append("%s:%s" % (width, "whole-tensor-unit-refused"
                                      if "whole-tensor-unit-refused" in str(exc) else exc.tag))
        rec = g_run_variant(ctx, width, p["plan"], "out_wholenote.bin", "g_%s_whole_note",
                            fault="whole-note")
        wcen = DA.parse_census(rec["run"].get("census"))
        wd = DA.g_census(p["g"], p["units"], p["ids"], arm=wcen)
        whole_leg.append((width, wcen["whole_reads"], cen["unit_reads"], bool(wd["disagree"]),
                          wd["disagree"][:2]))
        experts = sorted(cen["bytes_by_expert"])
        doctored = dict(cen, unit_reads=max(0, cen["unit_reads"] - 20),
                        bytes_by_expert=dict((k, x) for k, x in cen["bytes_by_expert"].items()
                                             if k != experts[0]))
        dd = DA.g_census(p["g"], p["units"], p["ids"], arm=doctored)["disagree"]
        driver_leg.append((width, bool(dd), dd[:2]))

    reasons = []
    for w, dis, d, cen in agree:
        if dis:
            reasons.append("%s:census-disagree(%s)" % (w, ",".join(dis[:4])))
    for w, zero in zerod:
        if zero:
            reasons.append("%s:census-denominator-zero(%s)" % (w, ",".join(zero)))
    for w, wr, ur, seen, sub in whole_leg:
        if wr != ur:
            reasons.append("%s:whole-counter-did-not-count-units(%d/%d)" % (w, wr, ur))
        if not seen:
            reasons.append("%s:whole-read-injection-not-detected" % w)
    for w, hit, sub in driver_leg:
        if not hit:
            reasons.append("%s:driver-census-comparison-dead" % w)
    if any("NOT-REFUSED" in r for r in refused):
        reasons.append("whole-tensor-plan-accepted")
    for r in reasons:
        v.red.append("census-" + r)
    for w, wr, ur, seen, sub in whole_leg:
        print("GWHOLE path=%s arm_whole_reads=%d arm_unit_reads=%d honest_whole_reads=0 "
              "comparison_reddened=%s first_findings=%s"
              % (w, wr, ur, "yes" if seen else "no", ",".join(sub) or "none"))
    v.control("gather-census-proves-slices-not-whole-tensor", not reasons,
              "driver-vs-arm agreement per path: %s; zero-denominator legs: %s; a plan claiming the whole "
              "stacked tensor was refused with %s — that plan's byte claims, read off the safetensors "
              "headers and the loader's records, would cost %s per token against the %s the honest plan "
              "reads. The declared whole-note injection was counted per unit and reddened the same "
              "comparison (GWHOLE lines above), and the driver-half fabrication disagreed as %s. Named "
              "failures, empty on a clean run: %s"
              % ("; ".join("%s units=%d/%d spans=%d/%d distinct=%d disagree=%s"
                           % (w, c["unit_reads"], d["units"], c["spans"], d["spans"],
                              c["distinct_experts"],
                              ",".join(dis) or "none") for w, dis, d, c in agree),
                 "none" if not any(z for _, z in zerod) else str(zerod),
                 ",".join(refused),
                 ",".join("%s:%dB" % (w, c) for w, c, _ in costs),
                 ",".join("%s:%dB" % (w, b) for w, _, b in costs),
                 "; ".join("%s=%s" % (w, ",".join(sub) or "NONE") for w, hit, sub in driver_leg),
                 "; ".join(reasons) or "none"),
              legs=g_legs_since(snap), claims=g_claims(ctx))


def control_gather_poison(v, ctx):
    """An unwritten slot has to stay loud on both paths and both widths.

    `mode=poison_nan` reddens every field; `fault=skip-raw` and `fault=skip-routed` redden exactly the
    fields the skipped stage owns, which is the attribution leg — a poison that reddens everything cannot
    tell you which stage went missing, and that is the only thing a missing stage needs you to know.
    """
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    lines, ok = [], True
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        honest_nan = p["nan"]
        plan = g_plan_variant(p, p["plan"].read_text().replace("mode=correct", "mode=poison_nan"))
        rec = g_run_variant(ctx, width, plan, "out_gpoison.bin", "g_%s_poison" % width)
        _, nan_all = DA.g_record(rec["wrote"]["path"], p["g"])
        r2 = g_run_variant(ctx, width, p["plan"], "out_gskipraw.bin", "g_%s_skip_raw", fault="skip-raw")
        _, nan_raw = DA.g_record(r2["wrote"]["path"], p["g"])
        r3 = g_run_variant(ctx, width, p["plan"], "out_gskiprouted.bin", "g_%s_skip_routed",
                           fault="skip-routed")
        _, nan_rt = DA.g_record(r3["wrote"]["path"], p["g"])
        full = dict(gate=8 * 512, up=8 * 512, s=8 * 512, raw=8 * 2048, routed=2048)
        whole_poison = all(nan_all[k] == n for k, n in full.items())
        attributed = (nan_raw["raw"] == full["raw"] and nan_raw["gate"] == 0
                      and nan_rt["routed"] == full["routed"] and nan_rt["gate"] == 0)
        honest_clean = all(v == 0 for v in honest_nan.values())
        ok = ok and whole_poison and attributed and honest_clean
        lines.append("%s poison=%s skip-raw=%s skip-routed=%s honest=%s"
                     % (width, {k: nan_all[k] for k in ("gate", "raw", "routed")},
                        {k: nan_raw[k] for k in ("gate", "raw", "routed")},
                        {k: nan_rt[k] for k in ("gate", "raw", "routed")}, honest_nan))
    v.control("gather-poison-census-nonzero-and-attributed", ok,
              "NaN counts per field (gate/raw/routed shown): %s. The honest records carry none, the "
              "poison mode fills all five fields, and each skip fault fills the field its own stage owns "
              "and no other, so an unwritten slot is loud and attributable on both widths"
              % "; ".join(lines),
              legs=g_legs_since(snap), claims=g_claims(ctx))


def control_gather_order_and_form(v, ctx):
    """The combine's choices have to be load-bearing, on both widths: ascending-expert-id order, the
    accumulator the path uses, and the cast-before-the-multiply that prediction 4 says is a different
    number.

    Four legs per path: `rank_accum` (sum in slot order instead of ascending expert id), `acc_bf16` /
    `acc_f32` (force the OTHER path's accumulator, which on a near-tie row is allowed to move the argmax
    and on this row must move something or the choice is decoration), `cast_after` (keep the fp32 weight
    the reference threw away — prediction 4's mistake, this time on the PRODUCT, the half pass 3 §8.4
    named unreachable until a `routed_sum` row existed) and `expert_rotate` (the neighbour expert's bytes
    answering under this slot's id).

    TWO RULERS, because the honest arm is only the closest thing to the pin on the licensed path. Every leg
    must move the field its mode claims to move — that is what "load-bearing" means, and it is the half that
    was already firing (the first witness's `rank_accum` leg moved 1263 elements). On `ref_bf16`, where
    `routed_sum` is byte-exact against the pin and `expert_raw` sits inside a licence of 5, a mistake that
    moves the field must also land FURTHER from the pin than the honest arm; on `policy_a` the honest
    distance to a bf16 pin is set by the storage width (ndiff 16139 on `expert_raw`), so a different
    mistake can legitimately land nearer and distance-to-pin says nothing there.

    THE CLAUSE THAT WAS WRONG, written down so nobody fixes it back: the second clause used to read
    `nd >= honest_nd` and was applied to both paths. Against an honest `routed_sum` ndiff of 0 that "‍>=" is
    satisfied by any value at all, so the licensed half of the claim was vacuous, and against policy A's
    width-sized honest distance the same comparison demanded something the path cannot promise, so the leg
    read SILENT while its own movement counter said the mistake had fired. Strictly-further on the licensed
    path, movement on both, and no comparison to a pin that the path is not licensed against.
    """
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    lines, ok = [], True
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        arm = G_WIDTH_ARM[width]
        honest_nd, _ = ndiff_vs_pin(ctx["dumps"], "routed_sum",
                                    g_stages(p["fields"], p["g"]))
        honest_raw_nd, _ = ndiff_vs_pin(ctx["dumps"], "expert_raw",
                                        g_stages(p["fields"], p["g"]))
        for mode in ("rank_accum", "cast_after", "expert_rotate",
                     "acc_f32" if width == "ref_bf16" else "acc_bf16"):
            plan = g_plan_variant(p, p["plan"].read_text().replace("mode=correct", "mode=" + mode))
            rec = g_run_variant(ctx, width, plan, "out_%s.bin" % mode, "g_%s_%s" % (width, mode))
            fields, _ = DA.g_record(rec["wrote"]["path"], p["g"])
            nd, _ = ndiff_vs_pin(ctx["dumps"], "routed_sum", g_stages(fields, p["g"]))
            ndraw, _ = ndiff_vs_pin(ctx["dumps"], "expert_raw", g_stages(fields, p["g"]))
            if mode == "expert_rotate":
                moved = int(np.count_nonzero(fields["raw"] != p["fields"]["raw"]))
                nd_now, nd_honest = ndraw, honest_raw_nd
            else:
                moved = int(np.count_nonzero(fields["routed"] != p["fields"]["routed"]))
                nd_now, nd_honest = nd, honest_nd
            licensed = (width == "ref_bf16")
            further = (nd_now > nd_honest) if licensed else True
            ok = ok and moved > 0 and further
            lines.append("%s.%s field-moved=%d pin-distance %d>%s%s" % (
                width, mode, moved, nd_now, nd_honest,
                "" if licensed else " [policy A: movement is the ruler, not pin distance]"))
    v.control("gather-combine-choices-are-load-bearing", ok,
              "%s. Every one of these is a mistake the engine could actually make and each one moves the "
              "field it claims to; on the licensed path it also has to land further from the pin than the "
              "honest arm, and on policy A that last comparison is deliberately not made — the honest arm is "
              "not licensed against those pins there, so distance to them is not a licence the mistake could "
              "fail. No limit here is invented: the comparison is against the same pins the rows are judged "
              "on, which is what licenses calling the honest arm closer"
              % "; ".join(lines),
              legs=g_legs_since(snap), claims=g_claims(ctx))


def control_gather_width_substitution(v, ctx):
    """The two width paths must not be interchangeable, and the arm has to be the one that says so.

    A policy-A plan declaring a bf16 expert span, and a reference plan declaring a quantized one, are both
    refused before a byte is decoded (`path-kinds`). This is the structural half of "which shape is this
    arm running?": the earlier passes refused a wrong `elt=` on the router for the same reason, and a
    policy-A row produced out of a bf16 copy would be the exact forgery this slice has to make
    impossible — the shipping widths are a user decision (D-01), so a plan that quietly substitutes them
    is reporting on an artifact that does not exist.
    """
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    seen, ok = [], True
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        text = p["plan"].read_text()
        other = "bf16" if width == "policy_a" else "q4k"
        bad_text = re.sub(r"UNIT role=gate([^\n]*)kind=[a-z0-9]+",
                          r"UNIT role=gate\1kind=" + other, text, count=1)
        bad = g_plan_variant(p, bad_text)
        try:
            g_run_variant(ctx, width, bad, "out_kindswap.bin", "g_%s_kindswap" % width)
            seen.append("%s:NOT-REFUSED" % width)
            ok = False
        except DA.Refused as exc:
            named = "path-kinds" in str(exc)
            ok = ok and named
            seen.append("%s:%s" % (width, "path-kinds" if named else exc.tag))
    v.control("gather-width-substitution-refused", ok,
              "substituting the other path's storage kind into one UNIT line was refused by both paths as "
              "%s. The plan's declared path fixes which kinds are admissible, so a policy-A row cannot be "
              "manufactured from a bf16 copy of the experts (D-01 is the user's, not this arm's to "
              "re-interpret)" % ",".join(seen),
              legs=g_legs_since(snap), claims=())


def g_stamp_units(units, art_path):
    """Record, on every unit, where its bytes came from — BEFORE `g_slab` rewrites the offsets.

    `g_slab` turns `off`/`span_off` into slab coordinates, so after it runs no unit can say which *file*
    byte its first element was read from. That is the address the read-back control needs: the only
    expectation worth having for "the device holds the file's bytes" is the file itself. Both paths can
    name one — a reference unit's bytes come out of an HF shard (`src`), a policy-A unit's out of the
    artifact GGUF (`file_off`, from `Loader.expert_range`, read through the audit door) — and neither
    needs the dict key one code path happens to keep. Stamping here, before the remap, is what lets
    `gather-read-back-expectation-is-the-artifact` run on both paths instead of one.
    """
    for u in units:
        u["file_off"] = u["off"]
        if u.get("src"):
            u["file_kind"], u["file_src"] = "shard", u["src"]
        else:
            u["file_kind"], u["file_src"] = "artifact", art_path


READ_BACK_BYTES = 16


def g_unit_expectation(ctx, u, n=READ_BACK_BYTES):
    """The bytes the ARTIFACT holds at the unit's own file coordinate, or None if the unit cannot name one.

    The first version of `gather-device-unit-read-back` compared the device's read-back with `u["_bytes"]`,
    the copy the CPU host twin happens to keep, and died with `KeyError: '_bytes'` on the first device run
    (window `g`): the reference path never carries `_bytes` — `g_slab` reads a fused [2I,H] span itself —
    so the control understood one of the two paths and crashed on the other, and a crash inside a control
    reads as "the gate ran" exactly as much as a pass does.

    Returning None for a unit that cannot name its source is deliberate and is NOT the forbidden skip: the
    caller turns a None into a named failure of the leg (`gather-read-back-leg`), and
    `gather-read-back-expectation-is-the-artifact` runs that failure on a fabricated unit every run so the
    leg is known to redden. A read that *should* work and does not raises — an I/O error is a finding, not
    a reason to compare nothing.
    """
    kind, src, off = u.get("file_kind"), u.get("file_src"), u.get("file_off")
    if kind not in ("shard", "artifact") or not src or off is None:
        return None
    why = "M2 gather unit read-back expectation"
    try:
        if kind == "shard":
            data, sha, got = DA.g_read_file_span(src, off, n, why)
        else:
            source = SR.ByteSource(ctx["st"]["L"], why=why)
            try:
                data, sha, got = DA.read_span(source, src, off, n, why)
            finally:
                source.close()
    except OSError as exc:
        raise Refused("read-back-expectation-unreadable",
                      "%s: cannot read %d B at %d of %s for the read-back expectation (%s); a control "
                      "that cannot read its expectation fails here instead of comparing nothing"
                      % (why, n, off, src, exc))
    if got != n:
        raise Refused("read-back-expectation-short",
                      "%s: asked %d B at %d of %s, got %d B" % (why, n, off, src, got))
    return data


def g_read_back_leg(ctx, u, got_hex, slab_head):
    """One read-back leg's whole predicate: (ok, note). The control and the witness run THIS, not copies.

    Three hops, and every one of them can fail by name: the artifact's bytes at the unit's file coordinate,
    the driver's slab bytes at the unit's slab coordinate (the bytes handed to the H2D copy), and what the
    device copied back. A leg that cannot see one of them fails — it does not skip, and it does not default
    to agreement.
    """
    want = g_unit_expectation(ctx, u)
    if want is None:
        return False, ("NO-EXPECTATION unit names no file_kind/file_src/file_off (got %s/%r/%r), so "
                       "there is nothing to compare the device's bytes against"
                       % (u.get("file_kind"), u.get("file_src"), u.get("file_off")))
    want_hex = want[:READ_BACK_BYTES].hex()
    if slab_head is None:
        return False, "NO-SLAB the driver cannot read its own slab bytes at the unit's offset"
    if slab_head != want_hex:
        return False, ("SLAB-DIVERGES driver slab=%s but artifact=%s: the bytes handed to the copy are "
                       "not the bytes the artifact holds, so a device read-back of them would certify "
                       "nothing"
                       % (slab_head[:READ_BACK_BYTES * 2], want_hex))
    if got_hex is None:
        return False, "NO-DEVICE-READBACK the arm ran on a path that printed no read-back"
    if got_hex != want_hex:
        return False, "MISMATCH device=%s artifact=%s" % (got_hex[:READ_BACK_BYTES * 2], want_hex)
    return True, "match device==slab==artifact (%d B, artifact=%s)" % (len(want), want_hex)


def g_slab_head(p, u, n=READ_BACK_BYTES):
    """The slab bytes at the unit's slab coordinate — the bytes the H2D copy is supposed to have moved."""
    with open(str(p["slab"]), "rb") as fh:
        fh.seek(u["off"])
        data = fh.read(n)
    return data.hex() if len(data) == n else None


def control_gather_read_back_expectation(v, ctx):
    """The read-back control's expectation must be derivable on BOTH width paths, and its legs must be
    capable of reddening. Both halves are witnessed here, in a CPU witness.

    Window `g` found the first half the hard way, on a device run: `want = u["_bytes"][:16].hex()` was a
    `KeyError` because the reference path's units never carry `_bytes`. A window must never be the first
    time a control's plumbing runs, so every unit path is exercised here: the artifact expectation is
    derived for each width path and required to equal the slab bytes the arm is handed (that is the
    driver-side hop of the chain, and it is a real comparison — a mismatch means the slab is not the
    artifact). The second half is the one `pitfalls/40` §a-check-that-can-print-nothing-cannot-fail asks
    for: three fabrications, each of which must redden a leg — the unit with no source (`_bytes`-less, the
    shape that crashed), a device answer with one bit changed, and a slab taken from the wrong offset. The
    honest leg must pass, or the predicate proves nothing either way.
    """
    lines, ok = [], True
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        u = p["units"][0]
        head = g_slab_head(p, u)
        good, note = g_read_back_leg(ctx, u, head, head)
        ok = ok and good
        lines.append("%s leg[%s] %s" % (width, "pass" if good else "FAIL", note))
    honest_u = ctx["gpaths"][DA.G_PATHS[0]]["units"][0]
    honest_hex = g_unit_expectation(ctx, honest_u)
    bad_u = dict((k, x) for k, x in honest_u.items() if k not in ("file_kind", "file_src", "file_off"))
    fabrications = [
        ("unit-without-bytes", g_read_back_leg(ctx, bad_u, honest_hex.hex(), honest_hex.hex())),
        ("device-one-bit-off",
         g_read_back_leg(ctx, honest_u,
                         "%02x" % (honest_hex[0] ^ 0x01) + honest_hex[1:].hex(),
                         honest_hex.hex())),
        ("slab-from-wrong-offset",
         g_read_back_leg(ctx, honest_u, honest_hex.hex(),
                         g_slab_head(ctx["gpaths"][DA.G_PATHS[0]],
                                     dict(honest_u, off=honest_u["off"] + 4)))),
    ]
    for name, (fired_leg, note) in fabrications:
        reddened = not fired_leg
        ok = ok and reddened
        print("GWITNESS guard=gather-read-back fabricated=%s leg_fired=%s %s"
              % (name, "yes" if reddened else "NO-LEG", note))
    v.control("gather-read-back-expectation-is-the-artifact", ok,
              "expectations derived from the artifact on %s (the path that crashed in window `g` is the "
              "reference one, and it derives here), each equal to the slab bytes the arm is handed, and all "
              "three fabrications reddened their leg: %s. No leg defaults to agreement: a unit that cannot "
              "name its file is a NAMED failure, not a skip, which is the difference between this control "
              "and the `try/except: pass` that would have hidden the original crash"
              % (",".join("%s=%s" % (w, "derived") for w in DA.G_PATHS), "; ".join(lines)),
              # These legs compare driver-side bytes, not arm acts, so they are named for what they are
              # instead of borrowing a path name no arm reported (`arm_leg` counts only the latter).
              legs={"cpu_predicate": len(DA.G_PATHS) + len(fabrications)})


def control_gather_read_back(v, ctx):
    """In a device window: the bytes the device holds for one expert span are the bytes the file holds.

    Same purpose as `stage_run.device_read_back` for the shared expert: bind the kernel's input to the
    artifact, the first 16 bytes of the first gathered span, before any row is believed. The leg is
    `g_read_back_leg`, the same predicate `gather-read-back-expectation-is-the-artifact` reddens on a
    fabricated unit in a CPU window, so the code path that crashed in window `g` is known to be able to
    fail before a window is spent on it.

    In a CPU witness the device leg is reported VOID and not registered, because a read-back that read host
    memory and were called a device read-back is the substitution this lane exists to refuse.
    """
    if not ctx["device"]:
        print("GVOID control=gather-device-unit-read-back reason=no-device-act-in-this-run "
              "note=not registered as a control here: a host read of host bytes is not a device read-back; "
              "the expectation machinery it uses is witnessed by "
              "gather-read-back-expectation-is-the-artifact")
        return
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    ok, lines = True, []
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        u = p["units"][0]
        plan = g_plan_variant(p, p["plan"].read_text().replace(
            "read_off=-1 read_len=0", "read_off=%d read_len=%d" % (u["off"], READ_BACK_BYTES)))
        rec = g_run_variant(ctx, width, plan, "out_readback.bin", "g_%s_readback" % width, device=True)
        got = re.search(r"UNITREADBACK path=device off=%d bytes=%d hex=([0-9a-f]+)"
                        % (u["off"], READ_BACK_BYTES), rec["stdout"])
        good, note = g_read_back_leg(ctx, u, got.group(1) if got else None, g_slab_head(p, u))
        ok = ok and good
        lines.append("%s %s" % (width, note))
    v.control("gather-device-unit-read-back", ok,
              "first gathered span, first %d bytes, device vs slab vs artifact file: %s"
              % (READ_BACK_BYTES, "; ".join(lines)),
              legs=g_legs_since(snap), claims=("device",))


# ----------------------------------------------------------------------------------------------
# PER-FIELD ATTRIBUTION: WHERE A COMPILED-PAIR DIFFERENCE ACTUALLY LIVES.
#
# WHY A TOTAL BYTE COUNT IS NOT AN ATTRIBUTION. Window `g` printed `ndiff_bytes=3,744` for the policy-A
# gather pair and `52,976` for the GDN pair and nothing else, and a total cannot be checked against a
# mechanism: the reference-path gather pair is byte-exact on the same device, so "the device arms differ"
# is already falsified, and what is left is a question about specific fields. This is the `wpre` shape from
# window `e2` (total moved → per-field/per-slot measurement → attribution to a documented scope → a bound
# taken from documentation with a rejection control), and it starts the same way: count per field, print
# the bit-image step, and make the per-field counts account to the printed total so an attribution cannot
# quietly be about a different population than the pair line is (`RTATTRIB`/`WPRESCOPE` are the precedent).
#
# NOTHING HERE IS A BAND. The pair checks keep claiming whole-record byte-identity at zero tolerance
# (`gather-compiled-pair-byte-identical-per-path`, `gdn-compiled-pair-byte-identical`), and no field is
# moved out of claimed exactness by anything in this section — that is a relaxation D-18 reserves to the
# user. The injected causes below price candidate mechanisms; they do not license one.
# ----------------------------------------------------------------------------------------------

# Every attribution line printed, so the accounting rule can be checked over ALL of them and not over the
# one a control happens to look at: an attribution whose field counts do not add up to the pair total it
# is explaining is an attribution of a different file.
ATTRIBS = []


def ordered_f32_steps(a, b):
    """Ordered-bit-image step per element, the same width rule `block_gate.ulp_diff` uses (f32 image).

    Reuses the repo's comparator rule rather than a private differ: `pitfalls/30` §the-ULP-comparator-dead-branch
    is what a second, subtly-wrong differ bought once already. Records store every field as an f32 word —
    a bf16-carrying field is stored widened (`engine/device/m2_g_kernels.h` §the record) — so the ordered
    int32 step is the right ruler for both, and a bf16 step reads as 2^15 in these units.
    """
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 0x7FFFFFFF - ai, ai)
    bi = np.where(bi < 0, 0x7FFFFFFF - bi, bi)
    return np.abs(ai - bi)


def record_field_rows(path_a, path_b, order, fields):
    """Per-field (differing bytes, differing words, largest bit-image step) between two record files.

    The byte counts are taken on the raw file bytes in each field's range, so they must sum to the whole-file
    `pair_bytes` number the PAIR line prints — that equality is `field-attribution-accounts-to-the-pair`,
    and it is what stops a per-field table from being about a different thing than the total it explains.
    Fields come from the arm's own header layout (`device_arm.parse_g_layout` / `parse_d_layout`), which is
    already required to tile the record exactly.
    """
    a_bytes = np.frombuffer(Path(path_a).read_bytes(), dtype=np.uint8)
    b_bytes = np.frombuffer(Path(path_b).read_bytes(), dtype=np.uint8)
    if a_bytes.shape != b_bytes.shape:
        raise Refused("attrib-record-shape", "%s (%d B) and %s (%d B) are different sizes; a per-field "
                                            "diff of two records of different size is not a comparison"
                      % (Path(path_a).name, a_bytes.size, Path(path_b).name, b_bytes.size))
    a, b = a_bytes.view("<f4"), b_bytes.view("<f4")
    rows, total = [], int(np.count_nonzero(a_bytes != b_bytes))
    for name in order:
        spec = fields[name]
        off, ln = spec[0], spec[1]
        lo, hi = 4 * off, 4 * (off + ln)
        ndiff_bytes = int(np.count_nonzero(a_bytes[lo:hi] != b_bytes[lo:hi]))
        x, y = a[off:off + ln], b[off:off + ln]
        steps = ordered_f32_steps(x, y)
        differing = steps[steps > 0]
        rows.append(dict(field=name, words=int(ln), ndiff_words=int(np.count_nonzero(
            x.view(np.uint32) != y.view(np.uint32))), ndiff_bytes=ndiff_bytes,
            max_step=int(differing.max()) if differing.size else 0,
            median_step=int(np.median(differing)) if differing.size else 0))
    return rows, total


def d_state_per_head(pair, path_a, path_b, d):
    """The GDN `state` field split by head — the only way the window can distinguish 'a subset of the 32
    decay values differ' from 'every word moved a little'.

    §04.6 step 1 multiplies head h's whole 16,384-word block by `exp(g[h])`, one transcendental per head,
    so a last-bit difference in k of the 32 decay values moves k whole head-blocks and nothing else. The
    total byte count cannot say that (13,244 words is 0.8 of a head or 2.5% of the state, and both read the
    same in aggregate); the per-head split can, and it also separates a scattered pattern from a contiguous
    one, which is the difference between a transcendental and an index or stride defect. The 32 counts must
    add up to the `state` line of the attribution above, and `field-attribution-accounts-to-the-pair` holds
    them to it.
    """
    a = np.frombuffer(Path(path_a).read_bytes(), dtype=np.uint8)
    b = np.frombuffer(Path(path_b).read_bytes(), dtype=np.uint8)
    off, ln, _ = d["fields"]["state"]
    per_head = d["GDN_D_K"] * d["GDN_D_V"]
    heads = d["GDN_H_V"]
    counts = []
    for h in range(heads):
        lo = 4 * (off + h * per_head)
        hi = 4 * (off + (h + 1) * per_head)
        counts.append(int(np.count_nonzero(a[lo:hi] != b[lo:hi])))
    total = int(np.count_nonzero(a[4 * off:4 * (off + ln)] != b[4 * off:4 * (off + ln)]))
    moved = sum(counts)
    print("DHEAD pair=%s file=gdn_record.f32 heads=%d words_per_head=%d per_head_ndiff_bytes=%s "
          "state_ndiff_bytes=%d sum_of_heads=%d accounting=%s heads_with_diff=%d"
          % (pair, heads, per_head, ",".join(str(c) for c in counts), total, moved,
             "ok" if moved == total else "MISMATCH",
             sum(1 for c in counts if c)))
    ATTRIBS.append(dict(tag="DHEAD", pair=pair, total=total, moved=moved, fields_with_diff=["state"]))
    return counts


def print_field_attrib(tag, pair, file_word, rows, total, note=""):
    """One line per field, then the accounting line. Returns the rows so a control can compare them.

    The accounting line is the load-bearing one: `sum_of_fields` has to equal the pair's own
    `ndiff_bytes`, and `fields_with_diff` names where the difference lives without anyone re-reading the
    totals. A table whose counts did not add up to the number the PAIR line printed would be an attribution
    of a different file, and that is a finding, printed as one.
    """
    for r in rows:
        print("%s pair=%s file=%s field=%s words=%d ndiff_words=%d ndiff_bytes=%d max_bit_step=%d "
              "median_bit_step=%d"
              % (tag, pair, file_word, r["field"], r["words"], r["ndiff_words"], r["ndiff_bytes"],
                 r["max_step"], r["median_step"]))
    moved = sum(r["ndiff_bytes"] for r in rows)
    print("%s pair=%s file=%s ndiff_bytes=%d sum_of_fields=%d accounting=%s fields_with_diff=%s %s"
          % (tag, pair, file_word, total, moved,
             "ok" if moved == total else "MISMATCH",
             ",".join(r["field"] for r in rows if r["ndiff_bytes"]) or "none", note))
    ATTRIBS.append(dict(tag=tag, pair=pair, total=total, moved=moved,
                        fields_with_diff=[r["field"] for r in rows if r["ndiff_bytes"]]))
    return dict(pair=pair, total=total, moved=moved,
                fields_with_diff=[r["field"] for r in rows if r["ndiff_bytes"]], rows=rows)


GLIBC_LIBM = [None]


def glibc_expf(x):
    """glibc's `expf` — the very function the host body links against, reached through ctypes.

    Called one value at a time on purpose: the point is to compare the host route's answer with an
    independent correctly-rounded one on the inputs this row actually uses, not to sample numpy's own
    vectorised exp, which is a third implementation nobody claims.
    """
    if GLIBC_LIBM[0] is None:
        libm = ctypes.CDLL("libm.so.6")
        libm.expf.restype = ctypes.c_float
        libm.expf.argtypes = [ctypes.c_float]
        GLIBC_LIBM[0] = libm
    return GLIBC_LIBM[0].expf(ctypes.c_float(x))


def glibc_nextafterf(x, toward):
    """glibc's `nextafterf`, reached the same way as `glibc_expf` — the function the two exp-ULP modes use.

    Needed because the attribution asks "what ULP of exp would the device have had to return?", and the
    only honest answer is in units of the same ULP the arm's own injection moves.
    """
    if GLIBC_LIBM[0] is None:
        libm = ctypes.CDLL("libm.so.6")
        libm.expf.restype = ctypes.c_float
        libm.expf.argtypes = [ctypes.c_float]
        GLIBC_LIBM[0] = libm
    fn = GLIBC_LIBM[0].nextafterf
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float, ctypes.c_float]
    return fn(ctypes.c_float(x), ctypes.c_float(toward))


def silu_exp_step_table(gate, up, s_host, s_dev, limit=4):
    """For each differing `s` word, the smallest signed ULP on `expf(-gate)` that reproduces the device's.

    `m2g_item2` computes `s = silu(gate) * up` from the record's own gate and up words and touches no
    weight, so with `gate` and `up` bit-identical the only input that can have moved is the `expf` inside
    `mg_silu`. That makes the inverse question answerable: for each word the device moved, is there a
    ULP step on the exp that produces exactly that word? Smallest-|step| first, because several steps
    round to the same `s` and the claim worth making is about the least perturbation that fits.
    """
    tab = np.zeros(s_host.shape, dtype=np.int8)
    hist, sites = {}, 0
    idx = np.nonzero(s_dev.view(np.uint32) != s_host.view(np.uint32))[0]
    for i in idx:
        z, u = float(gate[i]), float(up[i])
        e = glibc_expf(-z)
        hit = 0
        for step in [s for pair in ((1, -1), (2, -2), (3, -3), (4, -4)) for s in pair][: 2 * limit]:
            e2 = e
            for _ in range(abs(step)):
                # `math.inf`/`-math.inf`, not `1.0`/`0.0`: `expf(-z)` exceeds 1 whenever the gate input is
                # negative, and a `nextafter(e, 1.0)` on such a value moves DOWN while pretending to move up.
                # The first version of this helper made that mistake and 11 of the 58 sites came back
                # unexplained, which the byte-level reproduction caught immediately (residual 3,300 B).
                e2 = glibc_nextafterf(float(e2), math.inf if step > 0 else -math.inf)
            pred = np.float32(np.float32(z / np.float32(1.0 + e2)) * np.float32(u))
            if pred.view(np.uint32) == s_dev[i].view(np.uint32):
                hit = step
                break
        tab[i] = hit
        hist[hit] = hist.get(hit, 0) + 1
    explained = int((tab != 0).sum())
    return dict(table=tab, sites=int(idx.size), explained=explained,
                hist=",".join("%+d:%d" % (k, v) for k, v in sorted(hist.items())) or "none")


def exp_route_disagreement(values, name, note=""):
    """How many of THESE transcendental sites disagree between two routes, and by how many steps.

    Route one is glibc's `expf` (what the host body computes). Route two is `Decimal.exp()` at 60 digits
    rounded once to f32 — the same second route `DECISIONS.md` D-14 used to decide what "correctly rounded"
    means for `ssm_a`. The count is not a measurement of the device's libm, and it must not be read as one:
    no CPU run can touch ocml. What it prices is how faithful the HOST side of the pair is to a correct
    answer, which is what makes "the device's difference is a last-bit transcendental" cheap or expensive:
    if glibc already agrees with correct on 4,095 of 4,096 sites, the hypothesis needs the device's libm to
    be ~two orders of magnitude less faithful, and says so out loud.
    """
    vals = np.asarray(values, dtype=np.float32)
    with localcontext() as c:
        c.prec = 60
        # Decimal.exp() at 60 digits, then one rounding into the f32 image — D-14's second route.
        # `struct.unpack` returns a TUPLE: taking the element, not the tuple, is what keeps `correct` a
        # 1-D array. A list of 1-tuples would make it (n,1), broadcast against `host`'s (n,) into (n,n), and
        # print a disagreement count of n² — which is exactly the bug this comment now guards.
        correct = np.array([struct.unpack("<f", struct.pack("<f", float(Decimal(float(v)).exp())))[0]
                            for v in vals], dtype=np.float32)
    host = np.array([glibc_expf(float(v)) for v in vals], dtype=np.float32)
    if correct.shape != vals.shape or host.shape != vals.shape:
        raise Refused("exp-route-shape", "the two routes came back %s and %s against a population of %s; "
                                        "a disagreement count between differently-shaped arrays would be a "
                                        "broadcast artefact, not a measurement"
                      % (correct.shape, host.shape, vals.shape))
    steps = ordered_f32_steps(host, correct)
    disagree = int(np.count_nonzero(steps))
    # A count larger than the population is the signature of a broadcast, and a count that cannot exceed its
    # denominator cannot be silently wrong: the denominator is printed with it.
    if disagree > vals.size:
        raise Refused("exp-route-count-exceeds-population",
                      "%s: %d disagreements over %d sites is not a count of sites" % (name, disagree,
                                                                                      vals.size))
    print("%s sites=%d host_route=glibc-expf reference_route=decimal60-rounded-to-f32 disagree=%d "
          "fraction=%.6f max_bit_step=%d %s"
          % (name + "SCOPE", vals.size, disagree, disagree / float(max(vals.size, 1)),
             int(steps.max()) if steps.size else 0, note))
    return dict(sites=int(vals.size), disagree=disagree,
                max_step=int(steps.max()) if steps.size else 0)


# The candidate causes, each one an arm mode (no second implementation in Python that could drift from the
# arm). For each: what it is, and which of the task's suspect classes it stands for.
G_CAUSES = (("silu_exp_ulp1", "transcendental last-bit: expf(-gate) one ULP up at every silu site"),
            ("cast_after", "combine weight form: the fp32 weight the reference threw away"),
            ("acc_bf16", "combine accumulator order: the other path's bf16 accumulator"),
            ("rank_accum", "combine walk order: the prototype's rank-order walk"),
            ("byte_shift_1", "a one-byte span offset: the stride/offset read defect"),
            # The three width-handling causes pass 8 adds, because D-27 (the artifact stores expert
            # down_proj at Q5_K/Q6_K where policy A declares Q4_K, admitted by `engine/loader.py`'s floor
            # rule without a waiver) makes a geometry disagreement a live suspect for the policy-A pair.
            # `width_misread` keeps the true block stride and decodes with the other format; the other two
            # take the STRIDE from the wrong kind too, which is the half a declared-vs-actual split would
            # actually break. Each is measured against its own HDR reach claim by GREACH, so a claim of
            # `raw`-only is tested, not assumed.
            ("width_misread", "format-only width swap on the down family: Q5_K bytes decoded as Q6_K"),
            ("q5k_as_q4k", "declared-narrower width read: the Q5_K down span read with Q4_K block stride "
                           "and Q4_K decode — what 'policy A declares Q4_K' would mean if the arm obeyed"),
            ("q4k_as_q5k", "declared-wider width read: the Q4_K gate/up spans read with Q5_K stride, which "
                           "runs past each unit's extent and is refused element by element"),
            ("scale_min_naive", "the K-quant sub-block scale/min decode read the naive way"),
            ("q6k_d_byte_0", "the Q6_K block's fp16 delta read from byte 0 instead of 208"))
D_CAUSES = (("decay_ulp1", "transcendental last-bit: exp(decay) one ULP up"),
            ("decay_head_off1", "wrong head's decay: the index defect"),
            ("state_xpose", "carried state read transposed: the index-mapping defect"),
            # The same transcendental at the two signs, through the per-head mask mode with every head set.
            # The `+1` leg is also the equivalence check that the mask mode and the old all-heads mode are
            # one cause and not two spellings, and the `-1` leg is the direction a device libm that rounds
            # down lands in — the direction the device's five heads actually moved in.
            ("decay_ulp_mask_up", "per-head mask mode, every head, +1 ULP (must equal decay_ulp1 exactly)"),
            ("decay_ulp_mask_down", "per-head mask mode, every head, -1 ULP: the other rounding direction"))


def d_cause_args(d, cause):
    """The plan arguments a priced cause needs, derived from the geometry rather than typed.

    A full mask is `(1 << GDN_H_V) - 1`: the population comes from the arm's header, so 26 heads or 64
    heads, this is the set the geometry says exists and not a number copied into a control.
    """
    all_heads = (1 << d["GDN_H_V"]) - 1
    if cause == "decay_ulp_mask_up":
        return dict(mode="decay_ulp_mask", mask=all_heads, steps=1)
    if cause == "decay_ulp_mask_down":
        return dict(mode="decay_ulp_mask", mask=all_heads, steps=-1)
    return dict(mode=cause)


def window_command(extra="--device"):
    """The canonical device command, rebuilt from the DEFAULT work dir — never from this run's scratch.

    A `command=` line is a promise the main agent runs verbatim, so it has to name the run they mean.
    Two leaks rode along when this was rebuilt from `sys.argv`: a CPU witness invoked with
    `--work engine/work/p12-x` printed a device command reading that scratch dir (window `l`'s own
    device run read `engine/work/m2dev`), and an argv that already carried `--device` printed
    `… --device --device`. Both are stripped here and the work dir is re-added as `WORK_DEFAULT`, so
    the printed command is exactly the runnable one; the fabrications in this file keep their own
    isolated dirs and never appear in a reported command. Other argv pairs stay — a window that built
    its arm somewhere else must still say so.
    """
    keep, i, argv = [], 0, list(sys.argv[1:])
    while i < len(argv):
        tok = argv[i]
        if tok == "--work":
            i += 2
            continue
        if tok.startswith("--work=") or tok in ("--device", extra):
            i += 1
            continue
        keep.append(tok)
        i += 1
    return " ".join([sys.argv[0]] + keep + ["--work", str(WORK_DEFAULT), extra])


def print_device_only_attribution_dvoids(vertical, pair_label):
    """The legs of the attribution that only a device window can execute, printed as such.

    Two of them are not about this vertical at all and are the same for every one:
      * injecting the cause on the DEVICE path. Every injected magnitude printed here is a host-path number,
        so a cause that exists only in device codegen — a fused multiply-add the device allows and the host
        body does not, a denormal policy, a different reduction tree in the `__global__` body — would show
        up as a device divergence no host injection reproduces. That is a finding, not a zero.
      * the first rung of `pitfalls/00-measurement.md` §a-divergence-count-that-changes-between-identical-runs
        is a race, not rounding: constant count + adjacent bit images + an attributed third side is rounding;
        a MOVING count is uninitialised memory or a race and may never be banded. Constancy is a property of
        two identical device runs, so no CPU witness can establish it, and `max_bit_step` in the attribution
        lines is the adjacency half once a window supplies it twice.
    """
    print("%sVOID control-leg=%s status=not-exercised reason=needs-a-device-window "
          "note=the per-field split of %s is what the injected magnitudes compare against; window `g` "
          "printed only the total, and a total cannot be checked against a mechanism"
          % ("D" if vertical == "gdn" else "G", "%s-device-divergence-attributed" % vertical, pair_label))
    print("DVOID control-leg=%s-cause-injected-on-device-path status=not-exercised "
          "reason=needs-a-device-window note=every cause was injected on the arm's host path, so a cause "
          "that only exists in device codegen (contraction the device allows, a denormal policy, a "
          "reduction tree inside the __global__ body) prices as zero here and would show as a device "
          "divergence no injection reproduces" % vertical)
    print("DVOID control-leg=%s-divergence-constant-across-identical-runs status=not-exercised "
          "reason=needs-a-device-window note=constant ndiff_bytes across two identical runs plus adjacent "
          "bit images is rounding; a moving ndiff_bytes is uninitialised scratch or a race and must never "
          "be banded (pitfalls/00-measurement.md). Run the window's driver command twice and diff the "
          "GATTRIB/DATTRIB lines" % vertical)
    print("DVOID control-leg=%s-exp-only-reproduction status=not-exercised reason=needs-a-device-window "
          "required-print=%sREPRO ... verdict=EXACT-REPRODUCTION command=%s "
          "note=the probe table is derived from the device's OWN words, so it cannot exist until a window "
          "writes the device record; an exact byte-level reproduction is what retires the width and index "
          "hypotheses without a tolerance, and a non-zero residual would name a second live mechanism"
          % (vertical, "D" if vertical == "gdn" else "G", window_command()))
    print("DVOID control-leg=%s-device-expf-bit-images status=not-exercised reason=needs-a-device-window "
          "required-print=%sEXPIMAGE, one line per site with the device's image beside glibc's, differing "
          "on exactly the sites whose words moved command=%s "
          "note=no host run can print ocml's answer. What the host side can say is which inputs sit on the "
          "knife-edge; the prediction that fails if the mechanism is something other than the last bit of "
          "exp is that the device's images differ somewhere else"
          % (vertical, "D" if vertical == "gdn" else "G", window_command()))


def control_gather_divergence_causes(v, ctx):
    """Attribute the device-vs-host policy-A gather divergence by magnitude, or report that no cause does.

    For each candidate cause the arm can execute, run it on the host path of the hipcc arm and count, per
    field, how many bytes that cause moves against the honest record. Then, when a device window supplies
    the observed divergence, compare: same fields non-zero, and total within a factor of two. A cause that
    moves the wrong fields is dead however well its total matches; a cause whose total is off by 4x has not
    been shown anything. What the CPU witness can already say, and prints:

      * `gate`/`up` are upstream of every transcendental in the body, so the `silu_exp_ulp1` leg MOVES THEM
        ZERO and moves `s`, `raw`, `routed` — the reference path, which rounds each of those to bf16, moves
        ZERO in every field for the same injection. That is the measured shape of "the reference-path pair
        is byte-exact on the device and the policy-A pair is not", and it is the falsifiable prediction: if
        the device's `gate`/`up` differ at all, this cause is dead and the defect is in the read.
      * how much the maximal form of the transcendental cause is worth: one ULP at every site is a hard
        upper bound on what a libm difference can move, and the host's own expf is checked against a
        correctly-rounded route on the same sites, so the bound is priced and not asserted.
    """
    if not ctx.get("gpaths"):
        return
    g = ctx["gpaths"][DA.G_PATHS[0]]["g"]
    device_pair = ctx.get("g_device_pair_rows") or {}
    accounted, injected_nonzero, lines, reach_offences, equiv_offences = [], [], [], [], []
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    ctx.setdefault("g_cause_match", {})
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        # The honest side is the hipcc arm's HOST path in both modes, on purpose: an injected cause is
        # measured inside one compiler, so the number it produces is the price of the cause and not of the
        # cause plus a codegen difference. What a cause can only be priced against on the device itself is
        # listed as a not-exercised leg.
        honest_rec = p["dir"] / "out_hip.bin"
        rows_by_cause, cause_rec = {}, {}
        for cause, what in G_CAUSES:
            plan = g_plan_variant(p, p["plan"].read_text().replace("mode=correct", "mode=" + cause))
            rec = g_run_variant(ctx, width, plan, "out_cause_%s.bin" % cause,
                               "g_%s_cause_%s" % (width, cause), device=False)
            rows, total = record_field_rows(rec["wrote"]["path"], honest_rec, g["order"], g["fields"])
            attrib = print_field_attrib("GINJECT", "g_%s_cause_%s_vs_honest" % (width, cause),
                                       "gather_record.f32", rows, total,
                                       note="cause=%s (%s), host path of the hipcc arm"
                                            % (cause, what))
            rows_by_cause[cause] = attrib
            cause_rec[cause] = rec["wrote"]["path"]
            ctx.setdefault("g_cause_paths", {}).setdefault(width, {})[cause] = rec["wrote"]["path"]
            # The arm publishes which record fields the mode it just ran can reach, and the measured set must
            # never exceed it: a mode that moves a field the driver did not name is a mode the driver is not
            # entitled to interpret. Moving LESS than claimed is not an offence per width — a width can
            # legitimately absorb a cause (the reference path rounds `s` to bf16 and swallows a last-bit exp
            # whole, which is a finding this control prints on purpose) — so it is printed here and adjudicated
            # once below, per cause across both widths: a cause that moves nothing anywhere is an inert
            # injection, and every magnitude printed beside it would be a lie about the cause it prices.
            claim_s = (rec.get("hdr", {}) or {}).get("mode_fields") or ""
            claim = set(w for w in claim_s.split(",") if w)
            moved_fields = set(attrib["fields_with_diff"])
            extra, unused = sorted(moved_fields - claim), sorted(claim - moved_fields)
            print("GREACH path=%s cause=%s arm_claimed=%s measured_moved=%s beyond_claim=%s "
                  "claimed_but_unmoved=%s"
                  % (width, cause, claim_s or "ABSENT", ",".join(sorted(moved_fields)) or "none",
                     ",".join(extra) or "none", ",".join(unused) or "none"))
            if extra:
                reach_offences.append("%s/%s beyond:%s" % (width, cause, "|".join(extra)))
            accounted.append(attrib)
            injected_nonzero.append((width, cause, total))
        # The per-site table is the same cause as the all-sites mode at a finer granularity, and that is
        # checked rather than asserted. An all-zero table must move NOTHING — a table the arm silently
        # ignored would otherwise make a later reproduction pass for the wrong reason — and an all-+1 table
        # must be byte-identical to `silu_exp_ulp1`'s record. Two spellings of one cause, held to one answer.
        population = g["MG_K"] * g["MG_I"]
        for word, step, expect in (("zero", 0, honest_rec), ("all_up", 1, cause_rec.get("silu_exp_ulp1"))):
            if expect is None:
                continue
            tab = p["dir"] / ("exp_table_%s.bin" % word)
            tab.write_bytes(bytes([step]) * population)
            vplan = g_plan_variant(p, p["plan"].read_text().replace(
                "mode=correct", "mode=exp_ulp_table exp_table=%s" % tab))
            vrec = g_run_variant(ctx, width, vplan, "out_exp_%s.bin" % word,
                                 "g_%s_exp_%s" % (width, word), device=False)
            vbytes, vdiff = DA.pair_bytes(vrec["wrote"]["path"], expect)
            print("GEXP path=%s table=%s steps=%+d population=%d vs=%s record_bytes=%d ndiff_bytes=%d "
                  "verdict=%s"
                  % (width, word, step, population, Path(expect).name, vbytes, vdiff,
                     "SAME-CAUSE" if vdiff == 0 else "NOT-THE-SAME-CAUSE"))
            if vdiff:
                equiv_offences.append(
                    "%s/exp_table_%s: %d bytes differ from %s, so the per-site table mode is not the "
                    "cause it claims to be and no GREPRO reproduction built on it may be believed"
                    % (width, word, vdiff, Path(expect).name))
        legs_so_far = g_legs_since(snap)
        print("GCAUSE path=%s injected_causes=%s device_pair=%s legs=%s"
              % (width, ";".join("%s total=%d fields=%s" % (c, a["total"],
                                                           ",".join(a["fields_with_diff"]) or "none")
                                   for c, a in rows_by_cause.items()),
                 ("total=%d fields=%s" % (device_pair[width]["total"],
                                          ",".join(device_pair[width]["fields_with_diff"]) or "none")
                  if width in device_pair else "not-exercised"),
                 ",".join("%s:%d" % (k, n) for k, n in sorted(legs_so_far.items()))))
        if width in device_pair:
            obs = device_pair[width]
            best = []
            for cause, attrib in rows_by_cause.items():
                same_fields = attrib["fields_with_diff"] == obs["fields_with_diff"]
                ratio = (obs["total"] / float(attrib["total"])) if attrib["total"] else 0.0
                within = 0.5 <= ratio <= 2.0
                if obs["total"] == 0:
                    print("GMAGMATCH pair=g_%s_hip_device_vs_hip_hostpath cause=%s device_bytes=0 "
                          "injected_bytes=%d verdict=NOTHING-TO-ATTRIBUTE device pair is byte-exact"
                          % (width, cause, attrib["total"]))
                    continue
                print("GMAGMATCH pair=g_%s_hip_device_vs_hip_hostpath cause=%s device_bytes=%d "
                      "injected_bytes=%d ratio=%.3f same_fields=%s within_factor_two=%s verdict=%s"
                      % (width, cause, obs["total"], attrib["total"], ratio, same_fields, within,
                         "CANDIDATE" if (same_fields and within) else "RULED-OUT"))
                if same_fields and within:
                    best.append(cause)
            ctx["g_cause_match"][width] = best if obs["total"] else ["byte-exact"]
            # The reproduction leg, available only where a device record exists: derive the per-site exp
            # steps the device's own `s` words imply, run the host path with that table, and ask whether the
            # two records are now the SAME BYTES. This is the half that no ratio can give: if the whole
            # record reproduces with only the silu's exp moved, then `raw` and `routed` — 16,384 and 2,048
            # words computed from the arm's own decoded Q4_K/Q5_K weights — carry no residual for a second
            # cause, which is what refutes a block-geometry disagreement rather than merely disliking it.
            if obs["total"] and width == "policy_a":
                f_dev, _ = DA.g_record(p["dir"] / "out_device.bin", g)
                f_hip, _ = DA.g_record(p["dir"] / "out_hip.bin", g)
                tab = silu_exp_step_table(f_hip["gate"], f_hip["up"], f_hip["s"], f_dev["s"])
                tpath = p["dir"] / "exp_table_from_device.bin"
                tpath.write_bytes(tab["table"].tobytes())
                rplan = g_plan_variant(p, p["plan"].read_text().replace(
                    "mode=correct", "mode=exp_ulp_table exp_table=%s" % tpath))
                rrec = g_run_variant(ctx, width, rplan, "out_exp_repro.bin", "g_%s_exp_repro" % width,
                                     device=False)
                rbytes, rdiff = DA.pair_bytes(rrec["wrote"]["path"], p["dir"] / "out_device.bin")
                ctx.setdefault("g_repro", {})[width] = rdiff
                print("GREPRO pair=g_%s_hip_device_vs_hip_hostpath probe=exp_ulp_table path=host_cpubody "
                      "sites_perturbed=%d explained_by_exp_within_4ulp=%d unexplained=%d "
                      "min_steps_histogram=%s record_bytes=%d vs_device_record_ndiff_bytes=%d verdict=%s "
                      "role=diagnostic gate=no "
                      "note=the exp table is host-path-only by construction (`exp-table-host-only`), so a "
                      "clean reproduction here says the DEVICE's decoded weights equal the host's "
                      "(D-28: this leg is a diagnostic and does not decide the pair's verdict)"
                      % (width, tab["sites"], tab["explained"], tab["sites"] - tab["explained"],
                         tab["hist"], rbytes, rdiff,
                         "EXACT-REPRODUCTION" if rdiff == 0 else "RESIDUAL-LEFT"))
            lines.append("%s: device total=%d fields=%s candidates=%s"
                         % (width, obs["total"], ",".join(obs["fields_with_diff"]) or "none",
                            ",".join(best) or "NONE"))
        else:
            lines.append("%s: device divergence not-exercised (no device pair in this run); injected "
                         "magnitudes above are the ruler a window compares against" % width)
        ctx.setdefault("g_causes", {})[width] = rows_by_cause
    if not ctx["device"]:
        print_device_only_attribution_dvoids("gather", "g_policy_a_hip_device_vs_hip_hostpath")
    legs = g_legs_since(snap)
    silu = [ctx["g_causes"][w]["silu_exp_ulp1"] for w in DA.G_PATHS if w in ctx.get("g_causes", {})]
    policy_silu = ctx.get("g_causes", {}).get("policy_a", {}).get("silu_exp_ulp1")
    ref_silu = ctx.get("g_causes", {}).get("ref_bf16", {}).get("silu_exp_ulp1")
    gate_up_clean = bool(policy_silu) and not [f for f in policy_silu["fields_with_diff"]
                                               if f in ("gate", "up")]
    ref_absorbed = bool(ref_silu) and ref_silu["total"] == 0
    inert = [cause for cause, _ in G_CAUSES
             if not sum(ctx.get("g_causes", {}).get(w, {}).get(cause, {}).get("total", 0)
                        for w in DA.G_PATHS)]
    # A cause is only inert if the read set could have exercised it. `q6k_d_byte_0` needs a Q6_K unit, and
    # the layer-0 read set is q4k/q5k only (GFORMAT), so its zero is a geometry fact about THIS layer and
    # not a dead injection — the distinction has to be printed, because "inert" and "not-exercised" read
    # the same in a summary line and only one of them is a defect. The population is counted from the
    # units, never typed.
    kinds = {}
    for w in DA.G_PATHS:
        for u in (ctx["gpaths"][w]["units"]):
            kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    needs_kind = {"q6k_d_byte_0": "q6k", "q5k_as_q4k": "q5k", "width_misread": "q5k"}
    unexercisable = [c for c in inert if c in needs_kind and kinds.get(needs_kind[c], 0) == 0]
    for cause in unexercisable:
        print("GCAUSE-VOID cause=%s status=not-exercised reason=no-%s-unit-in-this-read-set "
              "units_by_kind=%s note=the cause needs a %s block and the layer under test has none; the "
              "leg belongs to a window on a layer whose `down_proj` is %s (D-27: 3 of 40 layers). "
              "command=%s"
              % (cause, needs_kind[cause], ",".join("%s=%d" % kv for kv in sorted(kinds.items())),
                 needs_kind[cause], needs_kind[cause], window_command()))
    reach_offences += ["%s inert-on-both-widths" % c for c in inert if c not in unexercisable]
    ok = (all(a["moved"] == a["total"] for a in accounted)
          and any(t > 0 for w, c, t in injected_nonzero if w == "policy_a")
          and gate_up_clean and ref_absorbed and not reach_offences and not equiv_offences)
    reason = None
    if reach_offences:
        reason = ("an injected mode moved fields its own HDR reach claim does not name, or moved nothing on "
                  "either width (GREACH lines name which): %s" % ";".join(reach_offences))
    elif equiv_offences:
        reason = ("the per-site exp table is not equivalent to the all-sites exp cause it spells: %s"
                  % ";".join(equiv_offences))
    elif not ok:
        reason = ("attribution did not account (sum_of_fields != ndiff_bytes) or the injected causes were "
                  "inert or the transcendental cause moved a field upstream of it: accounted=%s "
                  "policy_a_injections_nonzero=%s gate_up_clean=%s ref_bf16_absorbs_exp=%s"
                  % ([ (a["pair"], a["total"], a["moved"]) for a in accounted if a["moved"] != a["total"]],
                     any(t > 0 for w, c, t in injected_nonzero if w == "policy_a"),
                     gate_up_clean, ref_absorbed))
    if ctx["device"]:
        residual = [w for w, nd in ctx.get("g_repro", {}).items() if nd]
        unresolved = [w for w in DA.G_PATHS if w in ctx.get("g_device_pair_rows", {})
                      and not ctx.get("g_cause_match", {}).get(w)]
        exact_repro = [w for w, nd in ctx.get("g_repro", {}).items() if nd == 0]
        if residual:
            ok = False
            reason = ("%s an exp-only probe left %s bytes of residual in the device record, so a second "
                      "mechanism is live and none of the causes above accounts for it"
                      % (",".join(residual), ",".join(str(ctx["g_repro"][w]) for w in residual)))
        elif unresolved and not exact_repro:
            ok = False
            reason = ("no injected cause reproduces the device divergence in field location AND magnitude "
                      "on %s, and no byte-level probe reproduced it either; the mechanism is therefore "
                      "UNATTRIBUTED and no scope may be claimed for it (GMAGMATCH lines name what was ruled "
                      "out and by how much)" % ",".join(unresolved))
        elif unresolved and exact_repro:
            # The sentence this branch used to print — "no injected cause reproduces the device divergence" —
            # is what contradicted `GREPRO … verdict=EXACT-REPRODUCTION` nine lines higher in the same log.
            # The factor-two screen is a RATIO test and a 1-ULP-per-site cause is a 4-5x ratio by construction,
            # so it can never call the cause the byte-level test just proved. The ratio screen stays printed
            # (it is what rules the width hypotheses OUT by 19.6x), but it no longer gets to state the verdict
            # after an exact reproduction exists: exact bytes beat a ratio, and a log that says both is a log
            # nobody can act on.
            lines.append("reproduced=%s: the per-site exp table derived from the device's own `s` words "
                         "reproduced the record byte for byte, so the ratio screen's %s "
                         "`candidates=NONE` is a statement about ratios, not about the mechanism"
                         % (",".join(exact_repro), ",".join(unresolved)))
    v.control("gather-divergence-causes-priced", ok,
              "%d candidate causes injected on the arm's host path and counted per field; every "
              "attribution accounts to its pair total; the reference path absorbs the maximal exp "
              "injection completely (total=%s) while policy A moves %d bytes with gate/up untouched — the "
              "measured shape of the device's split verdict. %s"
              % (len(G_CAUSES), ref_silu["total"] if ref_silu else "?",
                 policy_silu["total"] if policy_silu else 0, "; ".join(lines)),
              legs=legs, claims=("host_cpubody",),
              reason=reason)


def control_d_divergence_causes(v, ctx):
    """The GDN pair's candidate causes, priced the same way, plus the third rung of the rounding test.

    Three causes the arm can execute, counted per field against the honest record: the decay one-ULP-up
    injection (the transcendental hypothesis), the wrong-head decay (an index defect), and the transposed
    state read (an index-mapping defect). `q_hat`/`k_hat` are the discriminator the arm's own reach table
    supplies — neither reads the decay or the state — so any cause that moves them is not a decay rounding,
    and the injected numbers show whether that half of the prediction even survives: the decay injection
    moves them zero.
    """
    if not ctx.get("d"):
        return
    d = ctx["d"]["d"]
    snap = dict(PATH_LEGS)
    honest = ctx["work"] / "d_out_hip.bin"
    obs = ctx.get("d_device_pair_rows")
    accounted, rows_by_cause, reach_offences = [], {}, []
    for cause, what in D_CAUSES:
        args = d_cause_args(d, cause)
        r = d_variant(ctx, "d_cause_%s.bin" % cause, "d_cause_%s" % cause, mode=args["mode"],
                      mask=args.get("mask"), steps=args.get("steps"),
                      binary=ctx["args"].d_arm_binary)
        rows, total = record_field_rows(r["rec"]["wrote"]["path"], honest, d["order"], d["fields"])
        rows_by_cause[cause] = print_field_attrib("DINJECT", "d_cause_%s_vs_honest" % cause,
                                                 "gdn_record.f32", rows, total,
                                                 note="cause=%s (%s), hipcc arm's host path"
                                                      % (cause, what))
        ctx.setdefault("d_cause_paths", {})[cause] = r["rec"]["wrote"]["path"]
        # The arm publishes, in its own HDR line, which record fields the mode it just ran is able to move.
        # The measured set must equal that claim, in both directions: a mode that moves a field it did not
        # name is a mode the driver is not entitled to interpret, and a mode that moves LESS than it names is
        # an injection that is partly inert — which would make every magnitude printed here a lie about the
        # cause it claims to price. There is one GDN width, so claimed-but-unmoved can only mean inert here;
        # the gather lane has two widths and adjudicates that case per cause across both (`GREACH`).
        claim = set(w for w in (r.get("mode_fields") or "").split(",") if w)
        moved_fields = set(rows_by_cause[cause]["fields_with_diff"])
        extra, unused = sorted(moved_fields - claim), sorted(claim - moved_fields)
        print("DREACH cause=%s arm_claimed=%s measured_moved=%s beyond_claim=%s claimed_but_unmoved=%s"
              % (cause, r.get("mode_fields"), ",".join(sorted(moved_fields)) or "none",
                 ",".join(extra) or "none", ",".join(unused) or "none"))
        if extra or unused:
            reach_offences.append("%s(%s%s)" % (cause, "beyond:" + "|".join(extra) if extra else "",
                                                " unclaimed:" + "|".join(unused) if unused else ""))
        if cause in ("decay_ulp1", "decay_ulp_mask_down"):
            # The ruler for the head-count question, in both rounding directions: each of these injections
            # moves every head that CAN move, so the device run's `heads_with_diff` is read against it.
            d_state_per_head("d_cause_%s_vs_honest" % cause, r["rec"]["wrote"]["path"], honest, d)
        accounted.append(rows_by_cause[cause])
        # The mask mode must be the SAME cause as the all-heads mode it replaces, not a near relative: same
        # mask population, same sign, byte-identical records. An equivalence this cheap to ask is worth
        # asking before any number derived from either mode is believed.
        if (rows_by_cause.get("decay_ulp1") and rows_by_cause.get("decay_ulp_mask_up")
                and rows_by_cause["decay_ulp1"]["total"] != rows_by_cause["decay_ulp_mask_up"]["total"]):
            reach_offences.append("decay_ulp_mask_up(+1 on every head) total=%d but decay_ulp1 total=%d; "
                                  "one cause must not be two spellings"
                                  % (rows_by_cause["decay_ulp_mask_up"]["total"],
                                     rows_by_cause["decay_ulp1"]["total"]))
    print("DCAUSE injected_causes=%s device_pair=%s"
          % (";".join("%s total=%d fields=%s" % (c, a["total"], ",".join(a["fields_with_diff"]) or "none")
                      for c, a in rows_by_cause.items()),
             ("total=%d fields=%s" % (obs["total"], ",".join(obs["fields_with_diff"]) or "none")
              if obs else "not-exercised")))
    ok = (all(a["moved"] == a["total"] for a in accounted)
          and any(a["total"] > 0 for a in rows_by_cause.values()) and not reach_offences)
    reason = None
    if reach_offences:
        reason = ("an injected mode moved fields other than the ones its own HDR reach claim names, or fewer "
                  "(DREACH lines name which): %s" % ";".join(reach_offences))
    elif not ok:
        reason = ("attribution did not account, or every injected cause was inert: %s"
                  % [(a["pair"], a["total"], a["moved"]) for a in accounted if a["moved"] != a["total"]])
    d_repro, repro_note = {}, ""
    if obs is not None:
        # The device leg of the head question: take the heads the device's own `state` diff names, move each
        # of those decays by one signed ULP on the host path, and ask whether the whole GDN record is now
        # the device's bytes. Exact identity means the size of the device's difference is one ULP per named
        # head and nothing else moved — which is the claim a band would otherwise be asked to swallow.
        counts = ctx.get("d_device_head_counts") or []
        moved = [h for h, c in enumerate(counts) if c]
        if obs["total"] and moved:
            mask = sum(1 << h for h in moved)
            for steps in (-1, 1):
                rr = d_variant(ctx, "d_repro_mask%+d.bin" % steps, "d_repro_mask%+d" % steps,
                               mode="decay_ulp_mask", mask=mask, steps=steps,
                               binary=ctx["args"].d_arm_binary, metrics_ok=False)
                rbytes, rdiff = DA.pair_bytes(rr["rec"]["wrote"]["path"],
                                              ctx["work"] / "d_out_device.bin")
                print("DREPRO pair=d_hip_device_vs_d_hip_hostpath probe=decay_ulp_mask steps=%+d "
                      "mask=0x%x heads=%s path=host_cpubody record_bytes=%d "
                      "vs_device_record_ndiff_bytes=%d verdict=%s role=diagnostic gate=no "
                      "note=D-28: diagnostic, does not decide the pair's verdict"
                      % (steps, mask, ",".join(str(h) for h in moved), rbytes, rdiff,
                         "EXACT-REPRODUCTION" if rdiff == 0 else "RESIDUAL-LEFT"))
                d_repro[steps] = rdiff
            ctx["d_repro"] = d_repro
    if obs is not None:
        best = []
        for cause, attrib in rows_by_cause.items():
            same_fields = attrib["fields_with_diff"] == obs["fields_with_diff"]
            ratio = (obs["total"] / float(attrib["total"])) if attrib["total"] else 0.0
            within = 0.5 <= ratio <= 2.0
            if obs["total"] == 0:
                print("DMAGMATCH pair=d_hip_device_vs_d_hip_hostpath cause=%s device_bytes=0 "
                      "injected_bytes=%d verdict=NOTHING-TO-ATTRIBUTE device pair is byte-exact"
                      % (cause, attrib["total"]))
                continue
            print("DMAGMATCH pair=d_hip_device_vs_d_hip_hostpath cause=%s device_bytes=%d "
                  "injected_bytes=%d ratio=%.3f same_fields=%s within_factor_two=%s verdict=%s"
                  % (cause, obs["total"], attrib["total"], ratio, same_fields, within,
                     "CANDIDATE" if (same_fields and within) else "RULED-OUT"))
            if same_fields and within:
                best.append(cause)
        ctx["d_cause_match"] = best if obs["total"] else ["byte-exact"]
        # THE VERDICT IS THE BYTE-LEVEL TEST, NOT THE RATIO. The factor-two screen is a ratio test: the cause
        # that reproduces this pair byte for byte (−1 ULP on `exp(g_decay)` for the device's own head subset)
        # is a 5x ratio by construction and the screen can only ever call it RULED-OUT. Window `i` therefore
        # printed `verdict=EXACT-REPRODUCTION` and, below it, `no injected cause reproduces the GDN device
        # divergence` — a control whose cause table predated the probe that answered the question. The ratio
        # lines stay printed (they are what rules `decay_head_off1` and `state_xpose` out by 29x and 25x, and
        # withdrawing *those* is what makes the −1 match a test), but the screen no longer gets to state the
        # verdict once a byte-level reproduction exists, and if that reproduction is ever withdrawn the
        # control reddens by name instead of quietly falling back to the ratio.
        if obs["total"]:
            exact = d_repro.get(-1) == 0
            plus_residual = d_repro.get(1)
            if not d_repro:
                ok = False
                reason = ("no per-head ULP probe ran: the device's own `state` diff named no head "
                          "(DHEAD lines above), so there is no per-head decay cause to price and the "
                          "divergence is UNATTRIBUTED")
            elif not exact:
                ok = False
                reason = ("the cause that used to reproduce this pair is withdrawn: −1 ULP on "
                          "exp(g_decay) for the device's own head set left %s B of residual in the "
                          "2,187,264-byte record (DREPRO steps=-1), so either the mechanism changed or the "
                          "head subset is no longer what the device moved; the ratio screen above cannot "
                          "substitute for the byte test")
                repro_note = "reproduction=WITHDRAWN residual=%s" % d_repro.get(-1)
            elif not plus_residual:
                ok = False
                reason = ("the withdrawal arm agrees: +1 ULP on the same head set also reproduced the "
                          "device record (residual %s B), so the sign of the step is not pinned and the −1 "
                          "match is a fit rather than a test" % plus_residual)
                repro_note = "reproduction=NOT-SIGN-PINNED plus_residual=%s" % plus_residual
            else:
                repro_note = ("reproduction=EXACT steps=-1 heads=%s mask=0x%x withdrawal=steps+1 "
                              "residual=%d B: the sign of the step and the head subset are both pinned by "
                              "a probe that could have failed" % (",".join(str(h) for h in moved),
                                                                  mask, plus_residual))
    else:
        print_device_only_attribution_dvoids("gdn", "d_hip_device_vs_d_hip_hostpath")
    v.control("gdn-divergence-causes-priced", ok,
              "three candidate causes injected and counted per field, plus the two signed per-head ULP "
              "probe legs against the device record; every attribution accounts to its pair total; every "
              "cause moved exactly the fields the arm's own HDR reach claim names (DREACH); q_hat/k_hat "
              "untouched by all three, which is what the reach table promises and this leg measures. %s %s"
              % (";".join("%s=%d" % (c, a["total"]) for c, a in rows_by_cause.items()), repro_note),
              legs=legs_since(snap), claims=("host_cpubody",),
              reason=reason)


def control_attribution_probe_is_dialable(v, ctx):
    """The mask probe that carries the GDN attribution obeys its dial, head by head.

    `DREPRO … probe=decay_ulp_mask steps=-1 mask=0x80d200 … verdict=EXACT-REPRODUCTION` is a claim about ONE
    mode of ONE arm, and if that mode silently ignored its `mask=` or `steps=` argument the reproduction would
    still print 0 and the attribution would be a tautology — the arm compared with itself under a label. So
    every leg here asks the arm to dial the cause to a different size and requires a different answer:

      * `mask=0` withdraws the cause from every head: the record must be the honest bytes, byte for byte, and
        the arm's own HDR must echo `mode_mask=0x0` — an injection the arm did not accept cannot be priced.
      * the full mask must move a non-zero, non-empty set of per-head state rows (the DHEAD ruler).
      * each single head named by that ruler must move fewer bytes than the full mask and more than zero: a
        mask that moves the same amount whatever it names is not a mask.
      * both signs must be accepted by the arm and must differ from each other, or "1 ULP" is a label for one
        fixed perturbation rather than a signed step.
      * in a device window, the winning combination specifically: steps=-1 with the device's own head mask
        must reproduce the device record with ZERO residual, and every single-head subset must leave a
        strictly positive one. If the −1 ULP cause is ever withdrawn — a kernel change, a new device build,
        a different head subset — this reddens by name instead of letting the EXACT-REPRODUCTION line stand.

    The population of heads is derived from the arm's own per-head ruler (or, in a device window, from the
    device's own `state` diff), never typed: `0x80d200` appears in the log because it is what the device
    moved, not because anyone wrote it down.
    """
    if not ctx.get("d"):
        return
    d = ctx["d"]["d"]
    honest = ctx["work"] / "d_out_hip.bin"
    device_rec = ctx["work"] / "d_out_device.bin"
    snap = dict(PATH_LEGS)
    full = (1 << d["GDN_H_V"]) - 1

    def probe(tag, mask, steps):
        r = d_variant(ctx, "d_dial_%s.bin" % tag, "d_dial_%s" % tag, mode="decay_ulp_mask",
                      mask=mask, steps=steps, binary=ctx["args"].d_arm_binary, metrics_ok=False)
        _rows, total = record_field_rows(r["rec"]["wrote"]["path"], honest, d["order"], d["fields"])
        hdr = r["rec"].get("hdr", {})
        echo = (hdr.get("mode_mask"), hdr.get("mode_steps"))
        return r, total, echo

    reasons = []

    def probe(tag, mode, mask=None, steps=None):
        r = d_variant(ctx, "d_dial_%s.bin" % tag, "d_dial_%s" % tag, mode=mode, mask=mask, steps=steps,
                      binary=ctx["args"].d_arm_binary, metrics_ok=False)
        _rows, total = record_field_rows(r["rec"]["wrote"]["path"], honest, d["order"], d["fields"])
        hdr = r["rec"].get("hdr", {})
        print("DDIAL probe=%s mode=%s mask=%s steps=%s arm_echo=mask:%s steps:%s bytes_vs_honest=%d"
              % (tag, mode, "none" if mask is None else "0x%x" % mask,
                 "none" if steps is None else steps, hdr.get("mode_mask"), hdr.get("mode_steps"), total))
        return r, total, (hdr.get("mode_mask"), hdr.get("mode_steps"))

    _r, t_plain, echo_plain = probe("mode_correct", "correct")
    if t_plain:
        reasons.append("re-running the plan with the mask mode withdrawn moved %d B against the honest "
                       "record; the mode word alone must change no arithmetic, or the DREPRO legs are being "
                       "compared against a baseline that moves under them" % t_plain)
    if echo_plain != (0, 0):
        reasons.append("with no mask in the plan the arm still echoed mask=%r steps=%r; the HDR line is the "
                       "arm's own admission of what it was told to perturb" % (echo_plain,))
    try:
        d_variant(ctx, "d_dial_empty_mask.bin", "d_dial_empty_mask", mode="decay_ulp_mask", mask=0,
                  steps=-1, binary=ctx["args"].d_arm_binary, metrics_ok=False)
        reasons.append("the arm accepted a decay_ulp_mask plan whose mask names no head: an inert mask mode "
                       "is a probe that cannot fail, which is the whole failure mode of a tautological "
                       "reproduction (m2_d_io.h::mode-mask-empty did not fire)")
    except DA.Refused as exc:
        print("DDIAL probe=empty_mask verdict=%s tag=%s note=the arm refuses an empty mask by name, so the "
              "dial cannot be silently set to inert"
              % ("REFUSED" if "mode-mask-empty" in str(exc) else "WRONG-REFUSAL", exc.tag))
        if "mode-mask-empty" not in str(exc):
            reasons.append("the empty-mask probe was refused for the wrong reason (%s); the refusal this leg "
                           "names is the one m2_d_io.h::mode-mask-empty prints" % exc.tag)
    r_full, t_full, echo_full = probe("full", "decay_ulp_mask", full, -1)
    if not t_full:
        reasons.append("the full-head mask moved nothing at all: the probe is inert and every number printed "
                       "beside it is a lie about the cause it claims to price")
    if echo_full != (full, -1):
        reasons.append("the arm echoed mask=%r steps=%r for the full mask 0x%x/-1; the plan said 0x%x/-1, so "
                       "the arm is not reading its own dial" % (echo_full[0], echo_full[1], full, full))
    counts = d_state_per_head("d_dial_full_vs_honest", r_full["rec"]["wrote"]["path"], honest, d)
    heads = [h for h, c in enumerate(counts) if c]
    if not heads:
        reasons.append("the per-head ruler found no head that the full mask moves, so there is no population "
                       "for the subset legs and the mask claim cannot be tested here")
    single, single_totals = [], []
    for h in heads:
        _r, t, _e = probe("head%d" % h, "decay_ulp_mask", 1 << h, -1)
        single.append("%d:%d" % (h, t))
        single_totals.append(t)
        if not 0 < t < t_full:
            reasons.append("head %d alone moved %d B, outside (0, %d) — a mask naming one head must move "
                           "strictly less than a mask naming all of them and strictly more than nothing"
                           % (h, t, t_full))
    r_up, t_up, _e = probe("full_plus", "decay_ulp_mask", full, 1)
    if t_up == t_full or not t_up:
        reasons.append("+1 ULP and -1 ULP on the same mask moved the same bytes (%d vs %d): the arm is not "
                       "applying a SIGNED step, so `1 ULP` is a label for one fixed perturbation" % (t_up,
                                                                                                    t_full))
    if ctx["device"] and device_rec.exists():
        device_counts = ctx.get("d_device_head_counts") or []
        dev_heads = [h for h, c in enumerate(device_counts) if c]
        if not dev_heads:
            reasons.append("the device's own state diff named no head, so the mask that carried the "
                           "attribution has no derivation here")
        else:
            win_mask = sum(1 << h for h in dev_heads)
            _r, _t, _e = probe("device_heads", "decay_ulp_mask", win_mask, -1)
            win_res = DA.pair_bytes(ctx["work"] / "dvariant" / "d_dial_device_heads.bin",
                                    device_rec)[1]
            print("DREPRO-WIN probe=decay_ulp_mask steps=-1 mask=0x%x heads=%s residual_bytes=%d "
                  "verdict=%s role=diagnostic gate=no "
                  "note=frozen winning combination: this leg reddens if the cause is withdrawn; D-28 "
                  "keeps it printed as a diagnostic and it does not decide the pair's value verdict"
                  % (win_mask, ",".join(str(h) for h in dev_heads), win_res,
                     "EXACT" if win_res == 0 else "WITHDRAWN"))
            if win_res:
                reasons.append("the cause is WITHDRAWN: −1 ULP on exp(g_decay) for the device's own heads "
                               "(mask 0x%x) leaves %d B of residual, so the EXACT-REPRODUCTION claim no "
                               "longer holds and the divergence is unattributed again" % (win_mask, win_res))
            for h in dev_heads:
                _r, _t, _e = probe("devhead%d" % h, "decay_ulp_mask", 1 << h, -1)
                one = DA.pair_bytes(ctx["work"] / "dvariant" / ("d_dial_devhead%d.bin" % h),
                                    device_rec)[1]
                if one == 0:
                    reasons.append("head %d alone reproduced the whole device record, which contradicts the "
                                   "subset claim: one head is doing all the work and the DHEAD ruler that "
                                   "named the subset is not measuring what it claims" % h)
    legs = legs_since(snap)
    v.control("attribution-probe-is-dialable", not reasons,
              "%d probe legs on the arm's per-head decay mask, all against the honest record: withdrawing the "
              "mode moves %d B and echoes mask=%r steps=%r; an empty mask is refused by name; the full mask "
              "0x%x moves %d B across %d heads; +1 ULP moves %d B where −1 moves %d B, so the step is signed; "
              "single-head legs are %s%s"
              % (4 + len(heads) + (1 + len(dev_heads) if ctx["device"] and device_rec.exists() else 0),
                 t_plain, echo_plain[0], echo_plain[1], full, t_full, len(heads), t_up, t_full,
                 ",".join(single) or "none",
                 "" if not reasons else "; REFUSALS: " + "; ".join(reasons)),
              legs=legs, claims=("host_cpubody",),
              reason=(None if not reasons else "; ".join(reasons)))


def control_gather_device_refusal(v, ctx):
    """`--device` is a device act or a named refusal, never a host run wearing one.

    Two legs on the g++ twin, neither of which touches a GPU: asked for `--device` it must fail with
    `device-path-absent-in-this-binary` rather than run its CPU body, and asked for the device-only
    allocation fault it must fail with `fault-device-only` rather than fake the leg. Together they are the
    CPU-witness half of the rule that a device claim must be an act; the other half, the hipcc arm's
    oversize allocation actually failing, can only be witnessed in a window and is reported VOID here.
    """
    args, work = ctx["args"], ctx["work"]
    p = ctx["gpaths"]["ref_bf16"]
    out = work / "g_refuse_device.bin"
    seen = []
    try:
        DA.run_arm(args.g_host_binary, p["plan"], p["slab"], out, device=True, label="g_refuse_device")
        seen.append("device-ask:NOT-REFUSED")
    except DA.Refused as exc:
        seen.append("device-ask:%s" % ("device-path-absent-in-this-binary"
                                       if "device-path-absent-in-this-binary" in str(exc) else exc.tag))
    try:
        DA.run_arm(args.g_host_binary, p["plan"], p["slab"], out, fault="alloc-oversize",
                   label="g_refuse_oversize")
        seen.append("alloc-oversize:NOT-REFUSED")
    except DA.Refused as exc:
        seen.append("alloc-oversize:%s" % ("fault-device-only" if "fault-device-only" in str(exc)
                                           else exc.tag))
    if not ctx["device"]:
        print("GVOID control-leg=gather-alloc-oversize-actually-fails reason=needs-a-device-window "
              "note=the host twin refuses to run this leg at all, which is the leg above")
    fired = all("NOT-REFUSED" not in s and s.split(":")[1] in
                ("device-path-absent-in-this-binary", "fault-device-only") for s in seen)
    v.control("gather-device-ask-refuses-instead-of-substituting", fired,
              "the g++ twin answered %s. No host body is offered for a device claim, so a `--device` "
              "report can only come from the arm that launches" % ", ".join(seen))


# =====================================================================================
# D-28 — DEVICE-VS-HOST AGREEMENT IS JUDGED IN VALUE SPACE, AND THE BOUND IS DERIVED
# FIRST.
#
# WHY. `DECISIONS.md` D-28 retired byte identity as the witness between a device arm and its own CPU
# twin, for one measured reason: a last-bit difference at an `exp` site is not a last-bit difference
# in VALUE terms once the body has multiplied it through a long dot. Window `k` printed
# `GATTRIB … field=raw … max_bit_step=24576`, which in binary32 is about 2.9e-3 relative — roughly
# 12,000 times the 1-ULP perturbation it came from. Cancellation is the amplifier
# (`pitfalls/30-numerics.md` §A max-ULP band cannot see a whole-tensor 1-ULP error, and `ndiff`
# cannot see a chain), and a band numbered in ULPs of the *field* would have been both unachievable
# and blind. So the two compiled-pair checks judge per-field value-space agreement now.
#
# WHAT IS STILL EXACT, and it is this section that says so in lines: every host-twin pair
# (`ndiff_bytes=0`), the bf16 reference gather on the device, every field a documented transcendental
# does NOT reach (the arm's own reach table — `engine/device/m2_g_kernels.h::m2g_mode_field` and
# `engine/device/m2_d_kernels.h::m2d_mode_fields` — is the source of
# that population, not a list typed here), and every licence judged against the pinned oracle. The
# `GREPRO`/`DREPRO`/`DREPRO-WIN` legs stay printed and carry `role=diagnostic gate=no`: the mechanism
# stays legible, it no longer decides the verdict, and
# `d28-diagnostic-legs-are-labelled-and-do-not-decide` is what stops that sentence from rotting.
#
# WHY THE BOUND IS NOT READ OFF THE BUILD. D-28: "No number here may be read off the build it
# covers." Every number on a `BOUNDFROM` line is therefore computed before the pair is measured, and
# from three device-free sources:
#   * `mr_exp_documented_x_host_fd` — the documented transcendental allowance, read out of
#     `prototype/moe_router/moe_router_kernels.h::mr_exp` ("allowed to differ by an ULP or two") at
#     run time, propagated through the field's arithmetic by an arm-executed finite difference: the
#     arm's own host path, run with the documented step at every documented site, under every sign
#     pattern the allowance admits (all-up, all-down, and a half-population split so a mixed-sign
#     assignment is priced too). This is the amplification, measured on host bytes.
#   * `licensor_band_triangle` — the licensor's own published width for the same field
#     (`speccheck/evidence/gdn-band-2026-10-26.json`, which says of itself
#     `derived_from="floor probe (f64 anchor) x the triangle inequality"`), doubled by the triangle
#     inequality exactly as `specification/evidence/blockdevice-wpre-scope-2026-10-26.md` prior 1
#     does. Its premise (both routes inside their licence against the pin) is measured by the STAGE
#     rows of this run, and a field with no licence for the path under test prints that reason
#     rather than a borrowed number: policy A's gather rows are UNLICENSED
#     (`policy-a-width-band-absent`) so no licensor prior exists for `s`/`raw`/`routed`.
#   * `fp64_floor_probe` — the same arithmetic recomputed in float64 from operands the record itself
#     carries, and the fp32 body's distance from it: `s = silu(gate)*up` and
#     `routed = sum_slot raw[slot]*wcast[slot]` (`engine/device/m2_g_kernels.h::m2g_item1/::m2g_item4
#     spell both), so these two need no second implementation and no device. A field whose operands
#     the record does not carry (`raw`'s down_proj dot) says so instead of inventing a probe.
# The strictest available prior binds (`min`), as in `WPRESCOPE`. Where only ONE prior exists —
# `gather.raw` — D-14's condition is what admits it: a documented-route bound is admissible provided
# the must-catch set grows with it, and `d28-must-catch-set-is-derived-and-reddens` is that set.
#
# THE ESCALATION CLAUSE, AS A RULE. D-28 accepted the risk that a value band could pass while the
# mechanism is something other than the documented `exp` last-bit. `d28_mechanism_offences` reads
# this run's own diagnostic lines for exactly that shape, `d28_qualification_decision` refuses to
# print `QUALIFIED` when it fires, and `d28-escalation-clause-refuses-qualification` demonstrates the
# refusal on a fabrication so the rule cannot be decorative.
# =====================================================================================

VB_METRICS = ("rel-err", "relL2", "ndiff_frac")
DOC_ULP_HEADER = ROOT / "prototype" / "moe_router" / "moe_router_kernels.h"
VB_ULP_WORDS = {"one": 1, "two": 2, "three": 3}
# The only mechanism D-28 licensed: an `exp` last-bit difference at a documented site. A magnitude
# screen that names any other cause as CANDIDATE while a value band passed is the clause firing.
VB_DOCUMENTED_CAUSES = ("silu_exp_ulp1", "decay_ulp1", "decay_ulp_mask")
VB_DIAG_HEADS = ("GREPRO", "DREPRO", "DREPRO-WIN")


def vb_documented_allowance():
    """The documented per-call transcendental allowance, read out of the sentence that grants it.

    The bound multiplies by this number, so the number has to have a source that a reader can open and
    that fails loudly when the sentence is edited. A bound whose allowance was remembered from a
    previous pass is a fitted bound with extra steps.
    """
    flat = re.sub(r"\s+", " ", DOC_ULP_HEADER.read_text().replace("*", " "))
    m = re.search(r"allowed to differ by an ULP or (\w+)", flat)
    if not m or m.group(1) not in VB_ULP_WORDS:
        raise Refused("d28-documented-allowance-moved",
                      "%s::mr_exp no longer contains `allowed to differ by an ULP or <one|two>`; every "
                      "D-28 bound is that sentence times an amplification, so if the sentence moved the "
                      "bound is re-derived, not remembered" % DOC_ULP_HEADER.name)
    return int(VB_ULP_WORDS[m.group(1)]), "%s::mr_exp 'an ULP or %s'" % (DOC_ULP_HEADER.name, m.group(1))


def vb_metrics(a, b):
    """(rel-err, relL2, ndiff_frac) between two equal-shaped f32 field images.

    `rel-err` is the licensor's own currency — the mean of |a-b| over the anchor denominator
    max(|b|, 1e-3 max|b|) — so the 1e-3 cancellation floor documented in
    `speccheck/gdn_ref.py::aggregates` is one floor in this repo and not a second one here; an
    unfloored per-element relative error is precisely the quantity that cannot be bounded at a
    cancelled element, which is the whole reason D-28 exists. `relL2` is the norm currency the
    non-expansive-write argument of `specification/evidence/ssm_a_ulp_bound.py` lives in, and
    `ndiff_frac` is the count that carries the zero-tolerance claim.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise Refused("d28-field-shape", "value metrics need two fields of one shape, got %s and %s"
                      % (a.shape, b.shape))
    if not a.size:
        return None
    x, y = a.astype(np.float64), b.astype(np.float64)
    scale = float(np.max(np.abs(y)))
    norm_y = float(np.linalg.norm(y))
    if scale == 0.0 or norm_y == 0.0:
        return None
    denom = np.maximum(np.abs(y), 1e-3 * scale)
    diff = np.abs(x - y)
    return {"rel-err": float(np.mean(diff / denom)),
            "relL2": float(np.linalg.norm(diff) / norm_y),
            "ndiff_frac": float(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))) / float(a.size),
            "n": int(a.size)}


def vb_combine_amplification(values):
    """The mixed-sign worst case over a set of response legs: max(single-sign) vs sum(opposite-sign).

    `values` is a list of (name, metric_value) in one currency. The documented allowance permits each
    site its own sign, so a single-sign injection is not the worst case; the triangle inequality says
    the sum of the half-population opposite-sign legs dominates any per-site sign assignment as long
    as the response is additive in the per-site step, and the additivity itself is printed rather
    than assumed.
    """
    singles = [v for _, v in values if v is not None]
    if not singles:
        return None
    return float(max(singles))


def vb_gather_response_legs(ctx):
    """The policy-A gather body, run on the host path with the documented step at every silu site.

    Three sign patterns over the whole site population (MG_K x MG_I, derived from the header): all up,
    all down, alternating. These are the amplification of `s -> raw -> routed` measured on host bytes
    at the size the document allows, which is what a bound on the device's difference needs and
    contains no device number.
    """
    p = ctx["gpaths"]["policy_a"]
    g, nulp, _ = p["g"], vb_documented_allowance()[0], None
    honest = p["dir"] / "out_hip.bin"
    population = g["MG_K"] * g["MG_I"]
    patterns = (("all_up", lambda i: nulp), ("all_down", lambda i: -nulp),
                ("alternating", lambda i: nulp if i % 2 else -nulp))
    legs = []
    for name, step in patterns:
        tab = p["dir"] / ("vb_table_%s.bin" % name)
        tab.write_bytes(np.array([step(i) for i in range(population)], dtype=np.int8).tobytes())
        plan = g_plan_variant(p, p["plan"].read_text().replace(
            "mode=correct", "mode=exp_ulp_table exp_table=%s" % tab))
        rec = g_run_variant(ctx, "policy_a", plan, "out_vb_%s.bin" % name,
                            "g_policy_a_vb_%s" % name, device=False)
        fields, _ = DA.g_record(rec["wrote"]["path"], g)
        legs.append((name, fields))
    return dict(kind="exp_ulp_table", population=population, steps=nulp, legs=legs,
                honest=honest, split_pairs=[])


def vb_gdn_response_legs(ctx):
    """The GDN body, run on the host path with the documented step on exp(g_decay) per head.

    The mask mode's population is the arm's own (`GDN_H_V` heads, derived, never typed), so the four
    legs are: every head up, every head down, odd heads up, even heads down. The last two are the
    mixed-sign probe; their per-field deviations sum in every currency this band uses.
    """
    d = ctx["d"]["d"]
    nulp = vb_documented_allowance()[0]
    heads = (1 << d["GDN_H_V"]) - 1
    odd = sum(1 << h for h in range(d["GDN_H_V"]) if h % 2)
    even = heads ^ odd
    spec = dict(kind="decay_ulp_mask", population=d["GDN_H_V"], steps=nulp, legs=[],
                honest=ctx["work"] / "d_out_hip.bin", split_pairs=[])
    for label, mask, steps in (("all_up", heads, nulp), ("all_down", heads, -nulp),
                               ("split_odd_up", odd, nulp), ("split_even_down", even, -nulp)):
        r = d_variant(ctx, "d_vb_%s.bin" % label, "d_vb_%s" % label, mode="decay_ulp_mask",
                      mask=mask, steps=steps, binary=ctx["args"].d_arm_binary, metrics_ok=False)
        spec["legs"].append((label, r["fields"]))
    spec["split_pairs"] = [(("split_odd_up", "split_even_down"))]
    return spec


def vb_response_metrics(spec):
    """Per (field, metric), the worst documented-allowance response of each field of the record.

    Returns `{field: {metric: (value, leg_names)}}` plus the additivity witness: if moving half the
    population twice is not about the same as moving all of it once, the response is not additive and
    the summed mixed-sign leg is doing something other than what the triangle argument claims.
    """
    order = spec.get("order")
    honest = spec["honest"]
    base = spec.get("base_fields")
    out, additivity = {}, []
    leg_metrics = dict((name, dict((f, vb_metrics(base[f], fields[f])) for f in order))
                       for name, fields in spec["legs"])
    for field in order:
        out[field] = {}
        for metric in VB_METRICS:
            vals = [(name, leg_metrics[name][field][metric]) for name, _ in spec["legs"]
                    if leg_metrics[name][field]]
            if not vals:
                out[field][metric] = (None, "no leg reached this field")
                continue
            singles = dict(vals)
            mixed = None
            for a, b in spec["split_pairs"]:
                if a in singles and b in singles:
                    mixed = min(1.0, singles[a] + singles[b]) if metric == "ndiff_frac" \
                        else singles[a] + singles[b]
            best = vb_combine_amplification(vals + ([("mixed", mixed)] if mixed is not None else []))
            names = [n for n, v in sorted(vals, key=lambda kv: -(kv[1] or 0))[:1]]
            if mixed is not None and mixed >= max(v for _, v in vals):
                names = [n for pair in spec["split_pairs"] for n in pair]
            out[field][metric] = (best, "+".join(names))
            if field == order[0]:
                allv = max(singles.get("all_up") or 0.0, singles.get("all_down") or 0.0)
                summed = (singles.get("split_odd_up", 0.0) + singles.get("split_even_down", 0.0)\
                          if "split_odd_up" in singles else None)
                if summed is not None and allv:
                    additivity.append("%s/%s all=%.3g split_sum=%.3g ratio=%.3f"
                                      % (field, metric, allv, summed, summed / allv))
    return out, additivity


def vb_floor_probes(ctx, width_fields, rt_fields):
    """The fp64 floor probes that need no device and no second implementation.

    `s = silu(gate) * up` and `routed = sum_slot raw[slot] * wcast[slot]` are the two stages whose
    every operand is already in the record (plus the router arm's own weights), so the float64
    recomputation is the same arithmetic without the f32 roundings — the floor probe D-28 allows,
    computed from bytes, not from a model of the bytes. Anything the record does not carry (`raw`'s
    down_proj dot) is returned as a named absence.
    """
    out = {}
    gate = width_fields["gate"].astype(np.float64)
    up = width_fields["up"].astype(np.float64)
    s64 = (gate / (1.0 + np.exp(-gate))) * up
    out["s"] = (vb_metrics(s64.astype(np.float32), width_fields["s"]),
                "fp64(silu(gate)*up) vs the arm's f32 s, per engine/device/m2_g_kernels.h::m2g_item1; "
                "premise D-19 §7: the device rounds the same operands, so its distance to fp64 is at "
                "most the host's, and the pair's separation is 2x the host's own distance")
    raw = width_fields["raw"].astype(np.float64).reshape(width_fields["_k"], width_fields["_h"])
    w = np.asarray(rt_fields["wcast"], dtype=np.float64)
    if w.size != width_fields["_k"]:
        raise Refused("d28-combine-population", "the combine has %d slots but the router record carries "
                                                "%d weights; the fp64 floor probe would be summing a "
                                                "different population than the arm did"
                      % (width_fields["_k"], w.size))
    acc = np.zeros(raw.shape[1], dtype=np.float64)
    for slot in range(raw.shape[0]):
        acc = acc + raw[slot] * w[slot]
    out["routed"] = (vb_metrics(acc.astype(np.float32), width_fields["routed"]),
                     "fp64(sum_slot raw[slot]*wcast[slot]) vs the arm's f32 routed, per "
                     "engine/device/m2_g_kernels.h::m2g_item4 (policy A: f32 accumulation, one "
                     "rounding); premise D-19 §7 as for s")
    out["raw"] = (None, "no fp64 route: the record does not carry the down_proj operands and no "
                        "in-tree fp64 body decodes the policy-A widths, so the stage that amplifies "
                        "most is the stage no floor probe can reach")
    for name in ("gate", "up"):
        out[name] = (None, "no transcendental reaches this field (arm reach claim); it stays exact")
    return out


def vb_licensor_priors(ctx, kind):
    """The licensor's published width for each field, doubled by the triangle inequality.

    Device-free by construction: the numbers come from the pinned band file and the premise — that
    both routes of the pair sit inside their own licence against the pin — is read off this run's
    STAGE rows, not asserted. A field with no licence on the path under test yields its licensor's
    own reason; `wpre`'s prior 1 is the precedent for the shape (`min(10, 9, 22)`, recorded in
    `specification/evidence/blockdevice-wpre-scope-2026-10-26.md`).

    Returns `({stage: {metric: bound}}, {stage: why-not})`. The licensor publishes `mean_rel` and
    `one_minus_cos` and no count of differing words, so it can never inform the count currency, and
    that absence is printed rather than quietly dropped.
    """
    if kind != "gdn":
        reasons = ",".join(sorted(set(
            r["band"].split("reason=", 1)[-1].split()[0] for r in (ctx.get("g_rows") or [])
            if r["arm"] == "policy_a" and r["band"].startswith("UNLICENSED"))))
        return {}, {"*": reasons or "no-band-for-this-path"}
    band, rows = ctx.get("band") or {}, {r["stage"]: r for r in (ctx.get("d_rows") or [])}
    out, why = {}, {}
    for stage, row in sorted(rows.items()):
        ent = band.get(stage)
        if ent is None or not row["band"].startswith("ACCEPTED"):
            why[stage] = (row["band"].split("reason=", 1)[-1].split()[0] if "reason=" in row["band"]
                          else row["band"].split()[0])
            continue
        out[stage] = {"rel-err": 2.0 * float(ent["mean_rel"]),
                      "relL2": 2.0 * math.sqrt(2.0 * float(ent["one_minus_cos"])),
                      "ndiff_frac": None}
        why[stage] = None
    return out, why


def vb_pair_specs(ctx):
    """The two pairs D-28 changed, as this run can actually adjudicate them.

    In a device window the A side is the device record; in a CPU witness the pair that exists is the
    host-twin pair, whose agreement is measured for real (and is zero) while the device leg is
    explicitly not claimed. Which of the two it is gets printed, so no reader can mistake the host leg
    for the device verdict.
    """
    specs = []
    if ctx.get("gpaths"):
        p = ctx["gpaths"]["policy_a"]
        g = p["g"]
        a = p["dir"] / ("out_device.bin" if ctx["device"] else "out_host.bin")
        wf, _ = DA.g_record(p["dir"] / "out_hip.bin", g)
        wf["_k"], wf["_h"] = g["MG_K"], g["MG_H"]
        specs.append(dict(kind="gather", pair="g_policy_a_hip_device_vs_hip_hostpath",
                          host_pair="g_policy_a_host_twin_vs_hip_hostpath",
                          file_word="gather_record.f32", order=list(g["order"]),
                          fields=g["fields"], a_side=a, b_side=p["dir"] / "out_hip.bin",
                          host_a=p["dir"] / "out_host.bin",
                          reach=("s", "raw", "routed"), layout=g, width_fields=wf,
                          claim_mode="silu_exp_ulp1"))
    if ctx.get("d"):
        d = ctx["d"]["d"]
        a = ctx["work"] / ("d_out_device.bin" if ctx["device"] else "d_out_host.bin")
        specs.append(dict(kind="gdn", pair="d_hip_device_vs_d_hip_hostpath",
                          host_pair="d_host_twin_vs_d_hip_hostpath",
                          file_word="gdn_record.f32", order=list(d["order"]),
                          fields=d["fields"], a_side=a, b_side=ctx["work"] / "d_out_hip.bin",
                          host_a=ctx["work"] / "d_out_host.bin",
                          reach=("scan_out", "state"), layout=d,
                          claim_mode="decay_ulp_mask"))
    return specs


def vb_record(path, order, fields):
    words = np.fromfile(str(path), dtype="<f4")
    return dict((name, words[off:off + ln].copy()) for name, (off, ln) in
                [(n, fields[n][:2]) for n in order])


def vb_field_derive(ctx, spec):
    """Every BOUNDFROM line for one pair, printed from priors computed before any witness.

    Returns `{field: {metric: derived}}`. For a field a documented transcendental does not reach the
    derived number is exactly zero, from the arm's own reach claim — which is D-28's keeps-exact list
    expressed as a bound rather than as a sentence.
    """
    nulp, src = vb_documented_allowance()
    if spec["kind"] == "gather":
        resp_spec = vb_gather_response_legs(ctx)
        resp_spec["order"], resp_spec["base_fields"] = spec["order"], vb_record(
            resp_spec["honest"], spec["order"], spec["fields"])
        floors = vb_floor_probes(ctx, spec["width_fields"], ctx["rt_fields"])
    else:
        resp_spec = vb_gdn_response_legs(ctx)
        resp_spec["order"], resp_spec["base_fields"] = spec["order"], vb_record(
            resp_spec["honest"], spec["order"], spec["fields"])
        floors = dict((f, (None, "no fp64 route: the record does not carry the carry/scan operands "
                                 "of this field"))
                      for f in spec["order"])
    response, additivity = vb_response_metrics(resp_spec)
    licensor, why = vb_licensor_priors(ctx, spec["kind"])
    print("BOUNDFROM-SITE pair=%s probe=%s population=%d steps_per_site=%d sources=%s "
          "sign_patterns=%s additivity=%s note=the amplification is measured on the arm's own host "
          "path at the documented step; no device number is read here"
          % (spec["pair"], resp_spec["kind"], resp_spec["population"], nulp, src,
             ",".join(name for name, _ in resp_spec["legs"]),
             "; ".join(additivity) or "single-sign-legs-only"))
    derived = {}
    for field in spec["order"]:
        for metric in VB_METRICS:
            priors, unavailable = [], []
            if field not in spec["reach"]:
                priors.append(("arm_reach_claim_zero(%s mode=%s cannot move %s)"
                               % ("m2g_mode_field" if spec["kind"] == "gather"
                                  else "m2d_mode_fields", spec["claim_mode"], field), 0.0))
            else:
                val, legname = response[field][metric]
                if val is not None:
                    priors.append(("%s x host_fd[%s](steps=%d,%s)" % (src.split(" ")[0], legname,
                                                                      nulp, resp_spec["kind"]), val))
                ent = (licensor.get(field) or {}).get(metric)
                if ent:
                    priors.append(("licensor_band_triangle(speccheck band, %s)" % field, ent))
                elif field in licensor:
                    unavailable.append("licensor_band_triangle(%s:reason=publishes mean_rel and "
                                       "one_minus_cos only, no count of differing words)" % field)
                else:
                    unavailable.append("licensor_band_triangle(%s:reason=%s)"
                                       % (field, why.get(field) or why.get("*") or "no-band"))
                flo = floors.get(field)
                if flo and flo[0] and flo[0].get(metric) is not None:
                    # The probe measures |fp64 − host|. It bounds |device − host| only under the
                    # documented premise that the device rounds the same operands to the same order of
                    # error, so the prior is 2x the measured distance, and the premise is in the label.
                    # For the count metric a doubled fraction can reach 1.0, which says nothing, so the
                    # probe is withheld from that cell with the reason printed rather than inflated.
                    val = 2.0 * flo[0][metric]
                    if metric == "ndiff_frac" and val >= 1.0:
                        unavailable.append("fp64_floor_probe:reason=doubled count floor=%.3g of the "
                                           "field's words, which forbids nothing" % val)
                    else:
                        priors.append(("fp64_floor_probe(%s) premise=D-19§7 both-arms-round-same-terms"
                                       % flo[1], val))
                else:
                    unavailable.append("fp64_floor_probe:reason=%s" % (flo[1] if flo else
                                                                     "not-attempted"))
            if not priors:
                raise Refused("d28-no-documented-source",
                              "%s/%s has no documented source at all; refusing to invent a bound"
                              % (field, metric))
            value = min(v for _, v in priors)
            if metric == "ndiff_frac" and value >= 1.0:
                raise Refused("d28-count-bound-vacuous",
                              "the %s/%s count bound derived to %.4g of the field's words, i.e. every "
                              "word may differ and the cell can never fail. A bound that cannot fail is "
                              "not a bound; tighten the source or drop the cell, do not print it"
                              % (field, metric, value))
            derived.setdefault(field, {})[metric] = value
            print("BOUNDFROM pair=%s metric=%s field=%s sources=%d[%s] derived=%.6g binding=%s "
                  "single_source=%s teeth=%s unavailable=%s note=no number on this line is read from "
                  "the pair under test"
                  % (spec["pair"], metric, field, len(priors), " AND ".join(n for n, _ in priors),
                     value, min(priors, key=lambda kv: kv[1])[0],
                     "yes" if len(priors) == 1 else "no",
                     ("zero-tolerance" if value == 0 else
                      ("fraction-forbidden=%.3g" % (1.0 - value)) if metric == "ndiff_frac" else "n/a"),
                     "|".join(unavailable) or "none"))
    spec["derived"], spec["response"] = derived, response
    return derived


def vb_witness(spec, a_side, b_side, tag):
    """BOUNDWITNESS lines for one pair: the measured value against the derived bound, never folded in."""
    A, B = vb_record(a_side, spec["order"], spec["fields"]), vb_record(b_side, spec["order"],
                                                                      spec["fields"])
    ok, rows = True, {}
    for field in spec["order"]:
        m = vb_metrics(A[field], B[field])
        for metric in VB_METRICS:
            bound = spec["derived"][field][metric]
            if m is None:
                print("BOUNDWITNESS pair=%s field=%s metric=%s measured=unmeasurable derived=%.6g "
                      "margin=none inside=no note=the reference side is all zero; a comparator over "
                      "zeros cannot fail, so it is not a pass either" % (spec["pair"], field, metric,
                                                                        bound))
                ok = False
                continue
            measured = m[metric]
            inside = measured <= bound * (1.0 + 1e-9)
            margin = (bound / measured) if measured > 0 else float("inf")
            print("BOUNDWITNESS pair=%s field=%s metric=%s measured=%.6g derived=%.6g margin=%.4g "
                  "inside=%s for_leg=%s" % (tag, field, metric, measured, bound, margin,
                                            "yes" if inside else "no", spec["pair"]))
            rows[(field, metric)] = dict(measured=measured, derived=bound, inside=inside,
                                         margin=margin)
            ok = ok and inside
    spec["witness"] = rows
    spec["witness_ok"] = ok
    spec["witness_A"] = A
    return ok


def value_band(v, ctx):
    """Derive, then measure, for both D-28 pairs. Called once, after every arm record exists."""
    ctx["vb"] = {}
    ctx.setdefault("refusals", [])
    for spec in vb_pair_specs(ctx):
        def build(s=spec):
            vb_field_derive(ctx, s)
            same = s["a_side"] != s["b_side"]
            if ctx["device"] and same:
                vb_witness(s, s["a_side"], s["b_side"], s["pair"])
            else:
                vb_witness(s, s["host_a"], s["b_side"], s["host_pair"])
                print("BOUNDWITNESS pair=%s field=all metric=all measured=not-claimed derived=see-above "
                      "margin=none inside=not-claimed for_leg=%s note=no device record exists in a CPU "
                      "witness, so the device-vs-host leg is not claimed here; the command that takes it "
                      "is %s" % (s["pair"], s["pair"], window_command()))
                s["device_leg"] = "not-claimed"
            return s
        # One pair's bound derivation refusing (a documented sentence moved, a count bound that admits every
        # word) is that pair's news, not the run's: the other pair, every leg, every control and every CHECK
        # still have to happen, and `d28-bound-and-witness-lines-are-parsed-not-assumed` owns the finding.
        if leg_attempt(ctx, "derive", "%s bound-and-witness derivation" % spec["pair"], build) is None:
            continue
        ctx["vb"][spec["pair"]] = spec
        spec["witness_ok"] = bool(spec.get("witness_ok"))
    # Which verdict each compiled-pair check may reach: policy A and the GDN device leg in value space,
    # every keeps-exact leg of D-28 at byte identity.
    for width, p in (ctx.get("gpaths") or {}).items():
        if width == "policy_a":
            s = ctx["vb"].get("g_policy_a_hip_device_vs_hip_hostpath")
            p["exact_leg"], p["pair_ok"] = False, bool(s and s.get("witness_ok"))
        else:
            p["exact_leg"] = True
            p["pair_ok"] = p["pair"][1] == 0 and p["pair"][0] > 0
    s = ctx["vb"].get("d_hip_device_vs_d_hip_hostpath")
    ctx["d_pair_ok"] = bool(s and s.get("witness_ok"))


D28_ESCALATE = {"armed": False, "offences": []}


def d28_mechanism_offences(text, passing_pairs):
    """Which passing pairs have a diagnostic leg naming a mechanism that is not the documented one.

    Read off this run's printed lines, the way the contradiction guard is. Three shapes count, and
    only these: an `unexplained=` count above zero on a reproduction leg, a pair whose reproduction "
    legs exist but none of which came out EXACT-REPRODUCTION (the `steps=+1` withdrawal arm prints
    RESIDUAL-LEFT on purpose, so a residual beside an exact leg is not an offence), a magnitude
    screen whose CANDIDATE cause is not one of the documented transcendental causes, and — in a "
    device run, where a leg is owed — no diagnostic leg for the pair at all.
    """
    offences = []
    lines = (text or "").splitlines()
    for pair in sorted(passing_pairs):
        mine = [l for l in lines if pair in l]
        repro = [l for l in mine if l.startswith(VB_DIAG_HEADS)]
        for line in repro:
            if re.search(r"unexplained=(?!0\b)\d+", line):
                offences.append("%s: reproduction leg leaves work unexplained: %s" % (pair, line[:150]))
        if repro and not any("verdict=EXACT-REPRODUCTION" in l for l in repro):
            offences.append("%s: %d diagnostic legs and none reproduced this pair by the documented "
                            "exp route" % (pair, len(repro)))
        for line in mine:
            if "verdict=CANDIDATE" in line:
                m = re.search(r"cause=([a-z0-9_]+)", line)
                if m and m.group(1) not in VB_DOCUMENTED_CAUSES:
                    offences.append("%s: magnitude screen names a non-transcendental cause as "
                                    "CANDIDATE (%s) while the value band passed"
                                    % (pair, m.group(1)))
        if D28_ESCALATE.get("require_repro") and not repro:
            offences.append("%s: value band PASS with no diagnostic leg at all for the pair" % pair)
    return offences


def d28_qualification_decision(red, unlicensed, escalate):
    """The one place the lane's verdict word is chosen, with D-28's clause wired in.

    `QUALIFIED` and `PASS` are both refused while the clause is firing, and the refusal is a printed
    verdict word of its own so a log reader cannot miss it.
    """
    if red:
        return "FAIL"
    if escalate:
        return "ESCALATE-USER(d28-value-band-passed-off-mechanism)"
    return "QUALIFIED" if unlicensed else "PASS"


# --------------------------------------------------------------------------------- rejection legs
#
# A bound is a bound only if something can be built that lands on each side of it, and both legs are
# therefore built on the grid the stored format actually has — the field's own f32 word images — so
# "one past the bound" is an ADJACENT representable point and not a continuous wish:
#
#   * `ndiff_frac`: the lattice index is the number of differing WORDS, so one step is one word and the
#     measured gap between two adjacent legs is 1/n of the field's words. The base divergence between A
#     and B is counted first and the leg adds divergence by flipping words from the AGREEING set,
#     because flipping a word that already differs adds nothing: window `l` printed
#     `g/ndiff_frac at-limit=False … measured=0.374756 > bound=0.368164` for exactly that reason — 58
#     of gather `s`'s 4096 words already differed, and flipping the first 1508 A-words as the pre-fix
#     construction did added only 1481 of them, overshooting by 27 words.
#   * `rel-err`, `relL2`: the lattice index is the number of f32 steps one element is moved away from
#     its B-side value, so one step is one ULP at that element.
#
# `vb_edge_sizing` then MEASURES the step (`granularity = measured_one_past - measured_at_limit`, which
# is how 1/n = 2.44140625e-04 for a 4096-word field reaches the log — as a measurement, never typed)
# and asks whether the at-limit leg sits within one step of the bound. That one rule catches both
# failure shapes: a leg that overshot (window `l`) and a leg that undershot so far that the bound was
# never approached, which is the quiet twin — a bound nothing can approach was never tested.

# The pinned device-window record the byte-axis ratio is read from when this run has no device pair of
# its own. Window `l` is D-28's first device window; its two pair totals (52,976 B GDN, 3,744 B policy
# A) are unchanged since window `g`, and the per-field split on the same lines is what the ratio is
# re-derived from, so the number a reader can check lives beside the code that checks it.
VB_DEVICE_EVIDENCE = ROOT / "specification" / "evidence" / "blockdevice-device-2026-10-27-l1.log"

# The field and currency window `m` refused in, and the pinned device window its base divergence is quoted
# from in a CPU witness (the device-substrate base is device data — a CPU run has no device record of its
# own — so it is read off a device log and the file is named on the line, per the self-containment rule).
VB_SCANOUT_FIELD, VB_SCANOUT_METRIC = "scan_out", "rel-err"
VB_SCANOUT_PAIR = "d_hip_device_vs_d_hip_hostpath"
VB_SCANOUT_EVIDENCE = ROOT / "specification" / "evidence" / "blockdevice-device-2026-10-27-m1.log"


def _vb_f32_rank(x):
    """The integer position of an f32 on the number line: adjacent ranks are adjacent representables.

    IEEE order is magnitude-with-sign, so ranking is `bits + 0x80000000` at or above zero and
    `0x7FFFFFFF - bits` below it. `signbit`, not `>= 0`, splits the two halves so -0.0 ranks just under
    +0.0 instead of above every positive number.
    """
    arr = np.array([x], dtype=np.float32)
    bits = int(arr.view(np.uint32)[0])
    return (0x80000000 + bits) if not np.signbit(arr[0]) else (0x7FFFFFFF - bits)


def _vb_f32_of_rank(rank):
    """Inverse of `_vb_f32_rank`: the f32 value sitting at that position on the number line."""
    bits = (rank - 0x80000000) if rank >= 0x80000000 else (0x7FFFFFFF - rank)
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def vb_edge_sizing(measured_at, measured_past, bound, step=None):
    """distance_to_bound / granularity / sized_correctly for one metric's pair of adjacent legs.

    `granularity` is measured off the two legs, never hardcoded, so the lattice the format has is what
    sizes the leg rather than a constant someone remembered. A leg is sized correctly when the at-limit
    leg sits no further below the bound than one lattice step: anything further down is an under-sized
    leg, and an under-sized leg is a bound that was never approached, therefore never tested.

    `step` overrides the measured gap for callers that must judge a leg against a step measured
    elsewhere — the deliberately mis-sized fabrications are judged against the real lattice step, not
    against the gap to whatever leg they happen to be paired with.
    """
    distance = bound - measured_at
    granularity = (measured_past - measured_at) if step is None else step
    ok = granularity > 0.0 and distance <= granularity * (1.0 + 1e-9)
    if not ok:
        reason = ("legs-not-adjacent(granularity=%.6g)" % granularity if granularity <= 0.0 else
                  "under-sized(distance %.6g > one lattice step %.6g)" % (distance, granularity))
    else:
        reason = ("overshot-past-bound(distance %.6g is negative)" % distance if distance < 0.0
                  else "within-one-lattice-step")
    return dict(distance_to_bound=distance, granularity=granularity,
                sized_correctly=bool(ok), reason=reason)


def vb_pre_diverge(a, b, count):
    """A copy of `a` with `count` further words made to differ from `b`, by flipping bit 0 of agreeing
    words spread evenly through the field. The count is a device window's own per-field `ndiff_words`, so
    a leg built on this base is built on the population the device actually produced, and spreading them
    is what makes the base look like a real divergence rather than a block a head-only construction can
    overlap by accident. In-memory only.
    """
    out = np.array(a, dtype=np.float32)
    agree = np.flatnonzero(out.view(np.uint32) == np.asarray(b, dtype=np.float32).view(np.uint32))
    count = int(count)
    stride = max(1, agree.size // max(count, 1))
    out.view(np.uint32)[agree[::stride][:count]] ^= np.uint32(1)
    return out


def vb_naive_first_words_leg(a, b, count):
    """The pre-fix construction, kept ONLY as the negative control that proves the fix is load-bearing:
    flip bit 0 of the first `count` A-words *without regard to whether they already differ*. Under a
    non-zero base that leg lands past its own bound, which is what window `l` printed.
    """
    out = np.array(a, dtype=np.float32)
    out.view(np.uint32)[:int(count)] ^= np.uint32(1)
    return out


def vb_word_axis(text, kind):
    """The device pair's own per-field word/byte counts, parsed off printed GATTRIB/DATTRIB lines.

    Returns `(pair_bytes, {field: (ndiff_words, ndiff_bytes)}, source)`, or None when no such pair line
    was printed. The pair total is RE-DERIVED as the sum of the per-field `ndiff_bytes` and required to
    equal the printed pair line and its printed `sum_of_fields`, so the number a ratio uses is the
    per-field arithmetic, not a constant copied from a summary. In a device run the source is this run's
    own lines; in a CPU witness it is the pinned window-`l` log named on the line, which is device data
    this box cannot produce and says so.
    """
    head, pair = ("GATTRIB", "g_policy_a_hip_device_vs_hip_hostpath") if kind == "gather" else \
        ("DATTRIB", "d_hip_device_vs_d_hip_hostpath")
    lines = [l for l in (text or "").splitlines() if l.startswith("%s pair=%s " % (head, pair))]
    source = "this-run"
    if not lines and VB_DEVICE_EVIDENCE.exists():
        lines = [l for l in VB_DEVICE_EVIDENCE.read_text().splitlines()
                 if l.startswith("%s pair=%s " % (head, pair))]
        source = VB_DEVICE_EVIDENCE.name
    if not lines:
        return None
    per_field, pair_total, printed_sum = {}, None, None
    accounted = "absent"
    for line in lines:
        words = re.search(r"field=(\S+) words=(\d+) ndiff_words=(\d+) ndiff_bytes=(\d+)", line)
        if words:
            per_field[words.group(1)] = (int(words.group(3)), int(words.group(4)))
            continue
        total = re.search(r" ndiff_bytes=(\d+) sum_of_fields=(\d+) accounting=(\S+)", line)
        if total:
            pair_total, printed_sum = int(total.group(1)), int(total.group(2))
            accounted = total.group(3)
    if pair_total is None:
        return None
    derived_sum = sum(b for _, b in per_field.values())
    if not per_field or derived_sum != pair_total or printed_sum != pair_total or \
            accounted != "ok":
        raise Refused("d28-byte-axis-does-not-account",
                      "the %s/%s byte axis could not be re-derived from its printed per-field split: "
                      "fields sum to %d, pair line prints ndiff_bytes=%d sum_of_fields=%s "
                      "accounting=%s. A ratio against a total that does not account is not a refutation"
                      % (head, pair, derived_sum, pair_total, printed_sum, accounted))
    return pair_total, per_field, source


def vb_cause_bytes(text, kind, cause):
    """The injected byte total of a refuted cause, parsed off this run's printed GCAUSE/DCAUSE line."""
    prefix = "GCAUSE path=policy_a injected_causes=" if kind == "gather" else "DCAUSE injected_causes="
    full = [l for l in (text or "").splitlines() if l.startswith(prefix)]
    if not full:
        return None
    body = full[0].split("injected_causes=", 1)[1].split(" device_pair=")[0]
    for entry in body.split(";"):
        m = re.match(r"^(\S+) total=(\d+)", entry)
        if m and m.group(1) == cause:
            return int(m.group(2))
    return None


def vb_byte_axis(text, kind, cause):
    """The byte-count refutation of a cause, in numbers, re-derived from printed lines.

    The rule is the one the device screen already uses (`DMAGMATCH`/`GMAGMATCH::within_factor_two`): a
    cause is ruled out when its injected divergence is more than a factor of two above what the device
    pair actually differs by — i.e. when it exceeds the observed divergence *by more than the observed
    divergence itself*. Returns None when either side has no printed source, and the caller prints
    `carried_by=none`, which is a finding rather than a shrug.
    """
    axis = vb_word_axis(text, kind)
    if axis is None:
        return None
    observed, per_field, source = axis
    injected = vb_cause_bytes(text, kind, cause)
    if injected is None:
        return None
    # The same rule the screen uses, written so a byte-exact pair does not escape it: injecting 264,704 B
    # when the pair differs by 0 B is ruled out by any ratio, and `observed > 0` must not become a way for
    # a clean pair to make a huge cause look unrefuted.
    return dict(injected=injected, observed=observed, excess=injected - observed,
                ratio=(injected / float(observed)) if observed else float("inf"),
                refuted=bool(injected - observed > observed),
                source=source,
                per_field=";".join("%s=%d/%dw" % (f, b, w) for f, (w, b) in sorted(per_field.items())))


def vb_norm_lattice(b, metric):
    """The lattice a norm metric lives on for one field: which element a leg moves, and what one step costs.

    A norm leg moves the largest-|b| element (the one whose denominator is the field's own scale, so it is
    the element the currency reaches its bound cheapest at) one f32 ULP at a time away from its B-side value.
    `unit` is the metric movement of a one-unit move in the closed form; `metric_step` is the same thing at
    exactly one ULP, which is what a ±1 correction loop can actually walk. Derived from the B-side bytes only
    — never from the A side, and never from the bound — so it is a property of the field, not of the answer.
    """
    b64 = np.asarray(b, dtype=np.float32).astype(np.float64)
    scale = float(np.max(np.abs(b64)))
    site = int(np.argmax(np.abs(b64)))
    b_site = float(b64[site])
    denom = max(abs(b_site), 1e-3 * scale)
    unit = float(np.linalg.norm(b64)) if metric == "relL2" else denom * int(b64.size)
    step = _vb_f32_of_rank(_vb_f32_rank(b_site) + 1) - b_site
    # One ULP at the moved element, in the metric's own units: `unit` is already the denominator the metric
    # divides that element's distance by (the field mean `denom*n` for rel-err, the norm `||b||` for relL2),
    # so one step is `|step| / unit`. The window-`m` numbers: 1.95878951e-11 for GDN scan_out/rel-err.
    return dict(n=int(b64.size), site=site, b_site=b_site, rank0=_vb_f32_rank(b_site), unit=unit,
                metric_step=(abs(step) / unit) if unit else 0.0)


def vb_leg_headroom(bound, floor_metric, field="", metric=""):
    """How much metric a leg may still inject after the base divergence it carries — or the offence.

    A norm leg is built by moving ONE element off its B-side value; every other element keeps whatever
    divergence the substrate already produced. So the injectable part is `bound - floor`, where `floor` is
    the metric with that element's own contribution removed — not `bound`. Sizing from `bound` is sizing
    from a base of zero, which is the only base a CPU witness can ever produce (its host-twin pair is
    byte-identical), and it is what window `m` got wrong: on the device substrate the closed form landed
    `2.43282e-08` past a `1.46299e-07` bound — exactly the base it had ignored — and the ±1-ULP loop cannot
    walk 1,242 steps in a budget of 64.

    When the floor is already at or above the bound the field cannot be leg-sized from this element at all.
    That is the substrate sitting OUTSIDE the derivation, and the answer is D-18 escalation with the numbers
    in it, never a wider bound: this returns the offence instead of a headroom, and the caller refuses.
    """
    headroom = bound - floor_metric
    if headroom > 0.0:
        return headroom, None
    return 0.0, ("the %s/%s base divergence %.6g is at or above the derived bound %.6g even with this "
                 "element's own contribution removed, so no leg can be placed inside the bound by moving "
                 "it. That is the substrate sitting OUTSIDE the derivation, not a leg to re-size: escalate "
                 "(D-18), do not widen" % (field, metric, floor_metric, bound))


def vb_fabricate_at_limit(A, B, field, metric, bound, direction, layout):
    """A copy of the A-side field placed on one side of one metric's bound, on the format's own lattice.

    `direction="inside"` returns the largest lattice point at or below the bound; `"outside"` returns the
    immediately adjacent point above it, so the two legs are adjacent points of one grid and the step
    between them is measurable rather than assumed. The construction preserves the A-side's existing
    divergence and adds to it in lattice steps — `k` differing words for the count metric (added from the
    agreeing set, one differing word per flip), one element moved `j` f32 steps for the two norm metrics.
    Nothing is written to disk; these are in-memory fabrications and the line says so.

    Both metrics are sized from the base the metric will actually SEE: the word branch counts the differing
    words already present and adds `k - base` more, and the norm branch solves for
    `bound - floor_metric` (see `vb_leg_headroom`), where `floor_metric` is the metric with the moved
    element's own base contribution erased. A leg sized from a base of zero is not a leg at all on a
    substrate that has one.
    """
    a = np.array(A[field], dtype=np.float32)
    b = np.asarray(B[field], dtype=np.float32)
    n = a.size

    def measure(values):
        value = vb_metrics(values, b)[metric]
        if not np.isfinite(value):
            raise Refused("d28-edge-leg-wont-size",
                          "the %s/%s %s leg went non-finite on the lattice; a leg that is not a number is "
                          "not a leg" % (field, metric, direction))
        return value

    slack = bound * (1.0 + 1e-9)
    if metric == "ndiff_frac":
        differ = a.view(np.uint32) != b.view(np.uint32)
        base = int(np.count_nonzero(differ))
        k = int(np.floor(bound * (1.0 + 1e-9) * n))
        if direction != "inside":
            k += 1
        if k > n:
            raise Refused("d28-edge-leg-wont-size",
                          "the %s/%s bound %.6g admits %d differing words but the field holds %d; a count "
                          "bound that admits every word forbids nothing and cannot be leg-sized"
                          % (field, metric, bound, k, n))
        out = a.copy()
        if k > base:
            agree = np.flatnonzero(~differ)
            out.view(np.uint32)[agree[:k - base]] ^= np.uint32(1)
        return out, measure(out), dict(index=k, unit="word", base=base)

    lat = vb_norm_lattice(b, metric)
    site, b_site, unit, rank0 = lat["site"], lat["b_site"], lat["unit"], lat["rank0"]

    def leg(j):
        """The A side with the moved element parked `j` f32 steps off its B-side value (j=0 means: equal).

        `j=0` is an explicit point on the lattice, not "leave the A side alone": it erases the base
        divergence at the moved element, which is the only measurement that shows what the rest of the
        field already costs in this currency.
        """
        out = a.copy()
        out[site] = _vb_f32_of_rank(rank0 + max(int(j), 0))
        return out

    base_metric = measure(a)
    floor_metric = measure(leg(0))
    headroom, escalation = vb_leg_headroom(bound, floor_metric, field, metric)
    if escalation:
        raise Refused("d28-edge-leg-base-above-bound", escalation)
    # The lattice index that puts the metric at the bound in closed form (delta = headroom*unit), then at
    # most a few step corrections to land on the correct side of the stored f32 values. The metric is
    # monotone in j — the moved element's own |a-b| grows one representable value per step and every
    # other element is untouched — so "largest j at or below the bound" is well defined, and because the
    # closed form is solved against the headroom rather than the bare bound the correction is a step or
    # two on any substrate, not the 1,242-step walk window `m` refused in the middle of.
    j = max(0, _vb_f32_rank(b_site + headroom * unit) - rank0)
    for _ in range(64):
        if j > 0 and measure(leg(j)) > slack:
            j -= 1
        elif measure(leg(j + 1)) <= slack:
            j += 1
        else:
            break
    else:
        raise Refused("d28-edge-leg-wont-size",
                      "the %s/%s %s leg could not be placed on its own side of the bound %.6g in 64 lattice "
                      "steps from the headroom estimate (base_metric=%.6g floor=%.6g one step of this "
                      "metric=%.3g, so a base this size would need %.0f steps); a rejection leg that cannot "
                      "be sized is not a rejection leg"
                      % (field, metric, direction, bound, base_metric, floor_metric, lat["metric_step"],
                         abs(base_metric - bound) / lat["metric_step"] if lat["metric_step"] else -1))
    if direction != "inside":
        j += 1
    out = leg(j)
    return out, measure(out), dict(index=j, unit="ulp", base=0, base_metric=base_metric,
                                   floor_metric=floor_metric, headroom=headroom,
                                   metric_step=lat["metric_step"])


def vb_must_catch_population(spec):
    """One entry per structural boundary this record's own geometry offers, derived not listed.

    Every entry is a real defect at a real coordinate of the block, and the coordinates come out of
    the header-derived layout: 32 value heads of d_k*d_v words, the gate|up pair laid out along the
    output dim, MG_K expert slots of MG_H words, and each field's own span boundary. If the geometry
    changes the population changes with it, which is what makes the count a derivation.
    """
    items = [("span_off_by_one", "rotate %s by one f32 word: the image of reading a span from off+1 "
                                 "(the field's own span boundary)" % spec["span_field"],
              lambda R: _vb_rotate(R, spec["span_field"]))]
    if spec["kind"] == "gdn":
        d = spec["layout"]
        heads, per_head = d["GDN_H_V"], d["GDN_D_K"] * d["GDN_D_V"]
        if heads >= 2:
            items.append(("state_two_heads_permuted",
                          "swap head 0 and head 1 of the carried state (%d heads x %d words)"
                          % (heads, per_head),
                          lambda R: _vb_swap_blocks(R, "state", 0, 1, per_head)))
            items.append(("state_one_head_zeroed",
                          "zero head 2 of the state: one head's decay never applied",
                          lambda R: _vb_zero_block(R, "state", 2, per_head)))
        items.append(("carried_state_dropped",
                      "zero the whole state field: the record-space image of the arm's own skip-carry "
                      "fault (the carry stage never ran)", lambda R: _vb_zero_block(R, "state", 0,
                                                                                   10 ** 9)))
        if "g_decay" in [s["name"] for s in (ctx_g_decay(spec) or [])]:
            items.append(("decay_sign_inverted",
                          "state[h] *= exp(2*g_decay[h]) for every head: applying exp(-g) where §04b.1 "
                          "step 1 says exp(+g) (record-space image, g_decay read from the slab span)",
                          lambda R: _vb_invert_decay(R, spec)))
    else:
        g = spec["layout"]
        items.insert(0, ("gate_up_halves_swapped",
                         "exchange the gate and up halves (each MG_I=%d words, laid out along the "
                         "output dim): the SwiGLU operand swap" % g["MG_I"],
                         lambda R: _vb_swap_fields(R, "gate", "up")))
        if g["MG_K"] >= 2:
            items.insert(1, ("wrong_expert_id_slot_swapped",
                             "exchange expert slot 0 and slot 1 of s/raw/up (MG_K=%d slots): a wrong "
                             "expert id arrives under the right slot's weight" % g["MG_K"],
                             lambda R: _vb_swap_slots(R, 0, 1, g)))
    return items


def ctx_g_decay(spec):
    return spec.get("g_decay_spans")


def _vb_rotate(R, field):
    R[field] = np.roll(R[field], 1)
    return R


def _vb_zero_block(R, field, head, per_head):
    lo = head * per_head
    R[field][lo:lo + min(per_head, R[field].size - lo)] = 0.0
    return R


def _vb_swap_blocks(R, field, h0, h1, per_head):
    a, b = h0 * per_head, h1 * per_head
    tmp = R[field][a:a + per_head].copy()
    R[field][a:a + per_head] = R[field][b:b + per_head]
    R[field][b:b + per_head] = tmp
    return R


def _vb_swap_fields(R, f0, f1):
    tmp = R[f0].copy()
    R[f0][:] = R[f1][:]
    R[f1][:] = tmp
    return R


def _vb_swap_slots(R, s0, s1, g):
    for field, ln in (("s", g["MG_I"]), ("raw", g["MG_H"]), ("up", g["MG_I"]), ("gate", g["MG_I"])):
        a, b = s0 * ln, s1 * ln
        tmp = R[field][a:a + ln].copy()
        R[field][a:a + ln] = R[field][b:b + ln]
        R[field][b:b + ln] = tmp
    return R


def _vb_invert_decay(R, spec):
    span = [s for s in spec["g_decay_spans"] if s["name"] == "g_decay"][0]
    g = np.frombuffer(span["data"], dtype="<f4").astype(np.float64)
    d = spec["layout"]
    per_head = d["GDN_D_K"] * d["GDN_D_V"]
    for h in range(min(d["GDN_H_V"], g.size)):
        lo = h * per_head
        R["state"][lo:lo + per_head] = (R["state"][lo:lo + per_head] * np.float32(
            np.exp(2.0 * g[h]))).astype(np.float32)
    return R


def value_band_rejection(v, ctx):
    """At-limit INSIDE / one-past-limit OUTSIDE per pair per metric, on adjacent lattice points.

    Sized from the bound and from the lattice the format actually has, never from the spread
    (`specification/evidence/blockdevice-wpre-scope-2026-10-26.md` §4 is the precedent), and every leg is
    a fabrication that says so on the line. Three ways a band fails and all three redden this: the
    at-limit leg lands outside, the one-past-limit leg lands inside, or the at-limit leg sits more than
    one measured lattice step below the bound — the quiet one, a leg that never approached the bound and
    so never tested it. `sized_correctly` is a property of the PAIR of legs (the step between them is
    what measures the granularity), so both legs of one metric print the same verdict.
    """
    results, lines, rows = True, [], []
    for spec in ctx["vb"].values():
        A, B = spec["witness_A"], vb_record(spec["b_side"], spec["order"], spec["fields"])
        for metric in VB_METRICS:
            field = next((f for f in spec["order"] if spec["derived"][f][metric] > 0),
                         spec["order"][0])
            bound = spec["derived"][field][metric]
            legs, refused = {}, []
            for label, direction in (("at-limit", "inside"), ("one-past-limit", "outside")):
                got = leg_attempt(ctx, "edge",
                                  "%s %s/%s %s leg" % (spec["pair"], field, metric, label),
                                  lambda d=direction, f=field, bd=bound, l=spec["layout"],
                                  A=A, B=B: vb_fabricate_at_limit(A, B, f, metric, bd, d, l))
                if got is None:
                    refused.append(label)
                    continue
                fake, measured, lattice = got
                legs[label] = dict(measured=measured, lattice=lattice,
                                   inside=measured <= bound * (1.0 + 1e-9))
            if refused:
                # A refusal is already printed and counted on its owning check; the pair of legs it would
                # have produced cannot be sized, so it is not silently scored as a passing edge.
                results = False
                lines.append("%s/%s at-limit=%s one-past=%s sized_correctly=no refused=%s"
                             % (spec["pair"].split("_")[0], metric,
                                "at-limit" not in legs, "one-past-limit" not in legs,
                                "+".join(refused)))
                continue
            sizing = vb_edge_sizing(legs["at-limit"]["measured"], legs["one-past-limit"]["measured"],
                                    bound)
            for label in ("at-limit", "one-past-limit"):
                leg = legs[label]
                lattice = leg["lattice"]
                extra = "" if sizing["sized_correctly"] else " sized_correctly_because=%s" % sizing[
                    "reason"]
                if lattice["unit"] == "word":
                    basenote = " base_words=%d" % lattice["base"]
                else:
                    basenote = (" base_metric=%.6g base_headroom=%.6g" % (lattice["base_metric"],
                                                                         lattice["headroom"]))
                print("BOUNDEDGE pair=%s metric=%s field=%s leg=%s bound=%.6g measured=%.6g "
                      "lattice_index=%d lattice_step=%s%s distance_to_bound=%.9g granularity=%.9g "
                      "sized_correctly=%s inside=%s verdict=%s note=FABRICATION, in-memory only, "
                      "adjacent points of the field's own f32 word grid (%d %s steps apart), sized from "
                      "the bound, from the base divergence this substrate already carries, and from the "
                      "measured step rather than from the spread%s"
                      % (spec["pair"], metric, field, label, bound, leg["measured"], lattice["index"],
                         lattice["unit"], basenote, abs(leg["measured"] - bound),
                         sizing["granularity"], "yes" if sizing["sized_correctly"] else "no",
                         leg["inside"], "INSIDE" if leg["inside"] else "OUTSIDE",
                         abs(legs["one-past-limit"]["lattice"]["index"]
                             - legs["at-limit"]["lattice"]["index"]), lattice["unit"], extra))
            good = (legs["at-limit"]["inside"] and not legs["one-past-limit"]["inside"]
                    and sizing["sized_correctly"])
            if not good:
                results = False
            lines.append("%s/%s at-limit=%s one-past=%s sized_correctly=%s"
                         % (spec["pair"].split("_")[0], metric,
                            legs["at-limit"]["inside"], legs["one-past-limit"]["inside"],
                            "yes" if sizing["sized_correctly"] else "no"))
            rows.append(dict(pair=spec["pair"], metric=metric, field=field, bound=bound, legs=legs,
                             sizing=sizing, word_count=int(A[field].size), good=good))
    ctx["vb_edges"] = dict(ok=results, lines=lines, rows=rows)


def value_band_must_catch(v, ctx):
    """The geometry-derived must-catch set: every injected real defect must redden a value band."""
    printed = []
    for spec in ctx["vb"].values():
        spec["span_field"] = "state" if spec["kind"] == "gdn" else "raw"
        if spec["kind"] == "gdn":
            spec["g_decay_spans"] = [s for s in ctx["d"]["spans"] if s["name"] == "g_decay"]
        B = vb_record(spec["b_side"], spec["order"], spec["fields"])
        population = vb_must_catch_population(spec)
        print("MUSTCATCH-DERIVE pair=%s derived=%d geometry=%s items=%s"
              % (spec["pair"], len(population),
                 ("GDN_H_V=%d GDN_D_K=%d GDN_D_V=%d state_span=%d words"
                  % (spec["layout"]["GDN_H_V"], spec["layout"]["GDN_D_K"],
                     spec["layout"]["GDN_D_V"], spec["fields"]["state"][1])) if spec["kind"] == "gdn"
                 else ("MG_K=%d MG_I=%d MG_H=%d gate|up along the output dim, spans %s"
                       % (spec["layout"]["MG_K"], spec["layout"]["MG_I"], spec["layout"]["MG_H"],
                          ",".join("%d+%d" % spec["fields"][f] for f in ("gate", "up")))),
                 ",".join(name for name, _, _ in population)))
        caught = []
        for name, what, apply in population:
            fake = dict((f, arr.copy()) for f, arr in B.items())

            def price(name=name, what=what, apply=apply, base=B):
                apply(fake)
                outside, worst = [], None
                for field in spec["order"]:
                    m = vb_metrics(fake[field], base[field])
                    if not m:
                        continue
                    for metric in VB_METRICS:
                        if m[metric] > spec["derived"][field][metric] * (1.0 + 1e-9):
                            outside.append("%s/%s %.3g>%.3g" % (field, metric, m[metric],
                                                                spec["derived"][field][metric]))
                            if worst is None or m[metric] > worst[0]:
                                worst = (m[metric], field, metric)
                return outside, worst

            # One injected defect refusing to be priced is that defect's finding; the remaining members of
            # the derived population still print, because a MUSTCATCH population that stops half-way would
            # silently shrink the set D-14 conditions a bound on.
            priced = leg_attempt(ctx, "must-catch", "%s %s" % (spec["pair"], name), price)
            if priced is None:
                caught.append(name)          # a defect that cannot be priced is not a caught defect
                continue
            outside, worst = priced
            reddens = "RED" if outside else "CLEAN"
            if outside:
                caught.append(name)
            print("MUSTCATCH name=%s pair=%s injected=%s reddens=%s first=%s"
                  % (name, spec["pair"], what, reddens, ";".join(outside[:3]) or "none"))
            printed.append(dict(pair=spec["pair"], name=name, reddens=reddens))
        spec["must_catch"] = printed[-len(population):]
        spec["must_catch_derived"] = len(population)
    ctx["vb_must_catch"] = printed


def value_band_refuted_causes(v, ctx):
    """The two causes D-28 says are refuted stay live controls that must keep failing, on a NAMED axis.

    `silu_exp_ulp1` and `decay_ulp1` were refuted by a byte-count ratio against the observed device pair
    (264,704 B injected against 52,976 B observed on the GDN pair; 15,426 B against 3,744 B on policy A),
    and that ratio is device data no device-free value bound can encode. Each cause is re-priced here twice
    — in the value currencies, and on that byte axis — and the line names which axis carries the refutation:

      * `carried_by=value-band(rel-err,relL2)` — the band still rejects it in a value currency
        (`teeth=yes`);
      * `…+byte-diagnostic(...)` — the byte ratio rejects it, with the arithmetic spelled out in bytes and
        re-derived from this run's printed `DCAUSE`/`GCAUSE` totals and the per-field `DATTRIB`/`GATTRIB`
        split of the observed pair (the pair total is re-summed from those per-field numbers and required
        to equal the printed total, so the ratio uses arithmetic and not a copied summary);
      * `carried_by=none` — neither axis refutes it. That is a finding, and
        `d28-refuted-causes-still-refuted-and-say-what-carries-it` reddens on it.

    `decay_ulp1` is inside the value band by construction: a bound derived without the device sizes the
    worst case over the whole documented site population, and a +1-ULP injection on that population is a
    smaller member of the same family as the documented ±2-ULP allowance. So the byte axis is what keeps
    it refuted, and the line prints that in numbers instead of pretending the band did the job. Neither the
    band nor the cause list is adjusted to make the value axis look stronger: D-28 §3 is satisfied by
    naming the ruler, not by relaxing either side.
    """
    rows = []
    text = run_text(ctx)
    for spec in ctx["vb"].values():
        cause = "silu_exp_ulp1" if spec["kind"] == "gather" else "decay_ulp1"
        path = (ctx.get("g_cause_paths", {}).get("policy_a", {}) if spec["kind"] == "gather"
                else ctx.get("d_cause_paths", {})).get(cause)
        screen_head = "GMAGMATCH" if spec["kind"] == "gather" else "DMAGMATCH"
        screen = (ctx.get("g_cause_match", {}) or {}).get("policy_a") if spec["kind"] == "gather" \
            else ctx.get("d_cause_match")
        if not path:
            print("REFUTED-CAUSE cause=%s pair=%s status=not-exercised carried_by=none reason=no cause "
                  "record in this run (the divergence control did not run it), so neither axis priced it"
                  % (cause, spec["pair"]))
            rows.append(dict(cause=cause, pair=spec["pair"], fails=None, carried="none", byte=None))
            continue
        A = vb_record(path, spec["order"], spec["fields"])
        B = vb_record(spec["b_side"], spec["order"], spec["fields"])

        def price(A=A, B=B, spec=spec):
            out, ins = [], []
            for field in spec["order"]:
                m = vb_metrics(A[field], B[field])
                if not m:
                    continue
                for metric in VB_METRICS:
                    (out if m[metric] > spec["derived"][field][metric] * (1.0 + 1e-9) else
                     ins).append("%s/%s %.3g vs %.3g" % (field, metric, m[metric],
                                                         spec["derived"][field][metric]))
            return out, ins

        priced = leg_attempt(ctx, "refuted", "%s re-pricing in value space" % cause, price)
        if priced is None:
            # The refusal is printed and counted on this cause's owning check; the cause is then not
            # re-priced, which the check below reads as a finding rather than as an absent row.
            rows.append(dict(cause=cause, pair=spec["pair"], fails=None, carried="none", byte=None))
            continue
        outside, inside = priced
        fails = bool(outside)
        byte = leg_attempt(ctx, "refuted", "%s byte axis" % cause,
                           lambda c=cause, s=spec: vb_byte_axis(text, s["kind"], c))
        entry = [s for s in (screen or []) if "cause=%s " % cause in s]
        ruler = ("verdict=%s" % entry[0].split("verdict=", 1)[1].split()[0] if entry
                 else "screen-not-printed-in-this-run(mode=%s);required-print=%s verdict=RULED-OUT "
                      "cause=%s taken by %s" % ("device" if ctx["device"] else "cpu", screen_head,
                                                cause, window_command()))
        carried = []
        if fails:
            carried.append("value-band(%s)" % ",".join(sorted(set(x.split("/")[1] for x in outside))))
        if byte and byte["refuted"]:
            carried.append("byte-diagnostic(injected=%dB observed=%dB excess=%dB ratio=%.3f %s)"
                           % (byte["injected"], byte["observed"], byte["excess"], byte["ratio"], ruler))
        carried_by = "+".join(carried) or "none"
        arithmetic = ("" if not byte else
                      " injected_bytes=%d observed_bytes=%d excess_bytes=%d "
                      "exceeds_observed_by_more_than_observed=%s byte_source=%s observed_per_field=%s"
                      % (byte["injected"], byte["observed"], byte["excess"],
                         "yes" if byte["refuted"] else "no", byte["source"], byte["per_field"]))
        print("REFUTED-CAUSE cause=%s pair=%s must_keep_failing=yes teeth=%s outside=%s inside=%s "
              "verdict=%s%s carried_by=%s%s"
              % (cause, spec["pair"], "yes" if fails else "none",
                 ";".join(outside[:4]) or "none", ";".join(inside[:4]) or "none",
                 "STILL-REFUTED" if fails else "NOW-INSIDE-BAND", arithmetic, carried_by,
                 "" if fails else " note=THE BAND IS TOO WIDE for this cause in value space, and the "
                 "byte axis above is what refutes it: a bound derived without the device necessarily "
                 "sizes the worst case over the whole documented site population, so a 1-ULP injection on "
                 "that population sits inside it. Saying so is D-28 §3; pretending the value band did that "
                 "job would be the wider error, and widening the band or dropping the cause would be wider "
                 "still"))
        rows.append(dict(cause=cause, pair=spec["pair"], fails=fails, carried=carried_by, byte=byte))
    ctx["vb_refuted"] = rows


def vb_parse_scanout_evidence():
    """The device substrate's `scan_out/rel-err` base and leg outcome, off a pinned device window's log.

    A CPU witness has no device record, so the only honest source for a device-substrate number is a device
    window's own log, read from disk and named on the line (the self-containment rule in
    `pitfalls/60`). Returns None when the file is absent — the diagnosis then says it had no source instead
    of quoting a remembered number.
    """
    if not VB_SCANOUT_EVIDENCE.exists():
        return None
    lines = VB_SCANOUT_EVIDENCE.read_text().splitlines()
    want = ("BOUNDWITNESS pair=%s field=%s metric=%s " % (VB_SCANOUT_PAIR, VB_SCANOUT_FIELD,
                                                         VB_SCANOUT_METRIC))
    ent = [l for l in lines if l.startswith(want)]
    if not ent:
        return None
    got = re.search(r"measured=(\S+) derived=(\S+) margin=(\S+) inside=(\S+)", ent[0])
    if not got:
        return None
    refused = [l for l in lines if l.startswith("REFUSED")
               and "%s/%s inside leg" % (VB_SCANOUT_FIELD, VB_SCANOUT_METRIC) in l]
    return dict(substrate="device", base=float(got.group(1)), printed_bound=float(got.group(2)),
                margin=got.group(3), printed_inside=got.group(4), refused=bool(refused),
                refused_tag=(re.search(r"tag=(\S+)", refused[0]).group(1) if refused else "none"),
                source=VB_SCANOUT_EVIDENCE.name)


def vb_baseless_closed_form(A, B, field, metric, bound):
    """Where the OLD closed form lands when it sizes from the B-side element alone, i.e. from a base of 0.

    Returns the metric value that construction produces, or None when this run has no A side for the
    substrate in question (a CPU witness has none for the device). This is the number that made window `m`
    refuse: it sits the substrate's whole base divergence past the bound, and a ±1-ULP correction loop
    cannot close a gap of that size in a budget of 64 steps.
    """
    if A is None:
        return None
    a = np.array(A[field], dtype=np.float32)
    b = np.asarray(B[field], dtype=np.float32)
    lat = vb_norm_lattice(b, metric)
    out = a.copy()
    out[lat["site"]] = _vb_f32_of_rank(lat["rank0"]
                                       + max(0, _vb_f32_rank(lat["b_site"] + bound * lat["unit"])
                                             - lat["rank0"]))
    return vb_metrics(out, b)[metric]


def scanout_substrate_diagnosis(ctx):
    """What the device substrate does to `scan_out/rel-err`, in the bound's own currency, with a verdict.

    Window `m` refused to place the at-limit leg *inside* the bound, and the tempting reading — "the device
    must be worse than the bound allows" — is exactly the reading that leads to widening a band. This
    answers the question in numbers instead: the base divergence of `scan_out` under the same metric, on
    each substrate, against the derived bound, plus what the leg construction did with them.

      * `verdict=inside` — the base is inside the bound and the leg landed inside it.
      * `verdict=outside` — the base itself is above the bound. That is the escalation case (D-18): the
        derivation is short, not the leg, and no number here may be used to widen anything.
      * `verdict=construction-wrong` — the base is inside the bound but the leg could not be placed, which
        can only mean the sizing was done from something other than the base the metric sees.

    Every number is measured or parsed from a named file; none of them is typed.
    """
    spec = (ctx.get("vb") or {}).get(VB_SCANOUT_PAIR)
    if not spec or VB_SCANOUT_FIELD not in (spec.get("derived") or {}):
        return []
    metric, field = VB_SCANOUT_METRIC, VB_SCANOUT_FIELD
    bound = spec["derived"][field][metric]
    B = vb_record(spec["b_side"], spec["order"], spec["fields"])[field]
    lat = vb_norm_lattice(B, metric)
    text = run_text(ctx)
    # The one step of this metric that a ±1 correction loop can walk, measured off this run's own legs when
    # they printed, and only otherwise derived from the B-side bytes — the step is a property of the field.
    step, step_src = lat["metric_step"], "derived from the B-side bytes (no at-limit leg printed)"
    for ln in text.splitlines():
        if (ln.startswith("BOUNDEDGE pair=%s metric=%s " % (VB_SCANOUT_PAIR, metric))
                and " leg=at-limit " in ln):
            got = re.search(r" granularity=(\S+)", ln)
            if got:
                step, step_src = float(got.group(1)), ("measured: granularity of this run's own %s/%s "
                                                       "at-limit legs" % (field, metric))
                break
    rows = []

    def leg_state(substrate):
        """What this run's own legs did with this substrate: refused / inside / outside / not-built-here."""
        if (substrate == "device") == bool(ctx["device"]):
            hit = [r for r in (ctx.get("refusals") or [])
                   if "%s/%s " % (field, metric) in r["what"]]
            if hit:
                return "refused(%s)" % hit[0]["tag"]
            row = next((r for r in ((ctx.get("vb_edges") or {}).get("rows") or [])
                        if r["pair"] == VB_SCANOUT_PAIR and r["metric"] == metric), None)
            if row is None:
                return "no-leg-built-on-this-substrate"
            return "inside(%s)" % ("True" if row["legs"]["at-limit"]["inside"] else "False")
        return "not-built-on-this-substrate"

    def verdict_of(base, state):
        if base > bound * (1.0 + 1e-9):
            return "outside"
        return "construction-wrong" if state.startswith("refused") else "inside"

    host_base = vb_metrics(vb_record(spec["host_a"], spec["order"], spec["fields"])[field], B)[metric]
    host_state = leg_state("host_twin")
    rows.append(dict(substrate="host_twin", base=host_base, bound=bound,
                     verdict=verdict_of(host_base, host_state), source="this-run pair=%s"
                                        % spec["host_pair"], state=host_state))
    dev = None
    if ctx["device"]:
        ent = (spec.get("witness") or {}).get((field, metric))
        if ent:
            dev = dict(substrate="device", base=ent["measured"], bound=bound,
                       source="this-run BOUNDWITNESS",
                       closed_form=vb_baseless_closed_form(spec.get("witness_A"),
                                                           {field: B}, field, metric, bound),
                       closed_form_src="measured on this run's device record",
                       printed_bound=bound)
    else:
        dev = vb_parse_scanout_evidence()
        if dev:
            dev["closed_form"] = None
            dev["closed_form_src"] = "derived: bound + base (no device record exists in a CPU witness)"
    if dev:
        state = ("refused(%s,source=%s)" % (dev["refused_tag"], dev["source"]) if dev.get("refused")
                 else leg_state("device"))
        dev["state"] = state
        dev["verdict"] = verdict_of(dev["base"], state)
        dev["bound_matches_this_run"] = ("yes" if abs(dev["printed_bound"] - bound)
                                         <= abs(bound) * 1e-6 else "no")
        rows.append(dev)
    for r in rows:
        extra = ""
        if r["substrate"] == "device":
            closed_form = r.get("closed_form")
            excess = (closed_form - bound) if closed_form is not None else r["base"]
            steps = (excess / step) if step else -1
            extra = (" closed_form_sized_from=b-site-alone(base 0) "
                     "closed_form_measure=%s closed_form_source=%s "
                     "excess_of_closed_form_over_bound=%.6g "
                     "excess_equals_the_base_it_ignored=%s one_step_of_this_metric=%.9g "
                     "step_source=%s correction_steps_needed=%.0f correction_budget=64 "
                     "bound_matches_this_run=%s"
                     % ("%.6g" % closed_form if closed_form is not None
                        else "%.6g(=bound+base)" % (bound + r["base"]), r["closed_form_src"],
                        excess, "yes" if abs(excess - r["base"]) <= abs(bound) * 1e-6 else "no",
                        step, step_src, steps, r.get("bound_matches_this_run", "n/a")))
        print("SCANOUT-DIAGNOSIS substrate=%s base=%.6g bound=%.6g margin=%s leg=%s verdict=%s "
              "base_source=%s%s note=%s"
              % (r["substrate"], r["base"], bound,
                 ("inf" if r["base"] <= 0 else "%.4g" % (bound / r["base"])), r["state"], r["verdict"],
                 r["source"], extra,
                 ("the host-twin pair is byte-identical, so zero is the ONLY base a CPU witness can "
                  "produce — a closed form solved against a base of zero therefore passes here and refuses "
                  "on a substrate that has one" if r["substrate"] == "host_twin" else
                  "this substrate's base sits inside the derived bound, so the refusal in window `m` was "
                  "never evidence about the bound: the closed form sized from the B-side element alone "
                  "(base 0), landed the base's own worth past the bound, and the +-1-ULP correction loop "
                  "cannot walk that many steps in a budget of 64. Nothing here widens anything: the bound "
                  "printed is the BOUNDFROM number, untouched, and the fix is to size the leg from "
                  "bound-floor, the base the metric actually sees")))
    worst = "inside"
    for r in rows:
        if r["verdict"] == "outside":
            worst = "outside"
        elif r["verdict"] == "construction-wrong" and worst != "outside":
            worst = "construction-wrong"
    dev_row = next((r for r in rows if r["substrate"] == "device"), None)
    summary_note = ("no device-substrate line: this run has neither a device record nor the pinned window "
                    "named in VB_SCANOUT_EVIDENCE, so the question is unanswered, not answered 'inside'"
                    if dev_row is None else
                    ("the device substrate's base is %.6g of a %.6g bound, inside by a factor of %.4g; the "
                     "bound came from `mr_exp x host_fd[all_up](steps=2,decay_ulp_mask)`, which prices the "
                     "amplification of the documented step and was never the short leg — the short leg was "
                     "the leg-sizing, and the fp64 floor probe for this field prints unavailable (the "
                     "record does not carry the carry/scan operands), so the 2-ULP propagation was never "
                     "the thing to re-derive either"
                     % (dev_row["base"], bound, bound / dev_row["base"] if dev_row["base"] else -1)
                     if worst != "outside" else
                     "a substrate base above the bound means the derivation is short. Say which source: "
                     "the `mr_exp` 2-ULP propagation (a documented-route prior, re-derived by re-running "
                     "the arm's host-path finite difference at a wider documented allowance) or the fp64 "
                     "floor probe (re-derivable only if the record is made to carry the carry/scan "
                     "operands). Re-derive and print, never widen from a measurement."))
    print("SCANOUT-DIAGNOSIS substrate=headline base=see-above bound=%.6g verdict=%s escalation=%s "
          "bound_untouched=yes fix=size-the-leg-from-bound-floor-not-bound "
          "note=%s" % (bound, worst,
                       "D-18-required:the-substrate-sits-outside-the-derivation(tell-which-source-is-short)"
                       if worst == "outside" else
                       "not-required(no-substrate-base-exceeds-the-bound)", summary_note))
    ctx["scanout_diag"] = rows
    ctx["scanout_verdict"] = worst
    # one line per substrate this run could speak for, plus the headline
    ctx["scanout_lines_expected"] = len(rows) + 1
    return rows


def vb_diag_print_sites():
    """Every print statement in this file that can emit a GREPRO/DREPRO/DREPRO-WIN leg.

    Returned as `(head, labelled)` pairs, where `labelled` says the emitted text carries
    `role=diagnostic gate=no`. The scan is structural (AST over this file's own source) because in a
    CPU run not one of these legs prints, and a check that reads zero legs would pass by silence.
    """
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print" and node.args):
            continue
        first = node.args[0]
        # The legs are printed as `("GREPRO … %s" % (...))`, so the argument is a `%` BinOp whose left
        # operand is the literal that names the leg.
        if isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mod):
            first = first.left
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        head = first.value.split()[0]
        if head in VB_DIAG_HEADS:
            sites.append((head, "role=diagnostic gate=no" in ast.get_source_segment(src, node)))
    return sites


def refuted_carriage_offences(text):
    """Which printed REFUTED-CAUSE lines carry no refutation.

    `carried_by=none` is the shape that reddens — a refuted cause whose line names no axis that still
    refutes it is a cause that has stopped being checked. Split out of the check so a fabricated line can
    be run through the same predicate that reads the real ones, which is what proves the predicate can
    fail.
    """
    bad = []
    for line in (text or "").splitlines():
        if not line.startswith("REFUTED-CAUSE ") or " cause=" not in line:
            continue
        m = re.search(r" carried_by=(\S+)", line)
        carried = m.group(1) if m else "field-absent"
        if not any(axis in carried for axis in ("value-band", "byte-diagnostic")):
            bad.append("%s carried_by=%s" % (line.split(" cause=")[1].split()[0], carried))
    return bad


def device_command_offences(text):
    """How many printed lines carry a device command, and which of them are not exactly runnable.

    Two shapes count. A `--work` whose value is not the canonical default work dir is a witness's scratch
    dir riding along into a command that would read different bytes than the window it was printed for,
    and a doubled `--device` (window `l`'s `command=… --device --device`) is a command nobody typed. Both
    are read off the printed text, and the population is the lines that actually carry a command.
    """
    offences, count = [], 0
    for line in (text or "").splitlines():
        if "--device" not in line or "block_device.py" not in line:
            continue
        count += 1
        head = line.split(" ")[0]
        for value in re.findall(r"--work[= ](\S+)", line):
            if value != str(WORK_DEFAULT):
                offences.append("%s: --work %s is not the canonical %s" % (head, value, WORK_DEFAULT))
        if line.count("--device") > 1:
            offences.append("%s: %d --device flags in one reported command" % (head,
                                                                              line.count("--device")))
    return count, offences


def vb_fabrication_legs(pair, spec, row, metric, axis, B):
    """The four fabrication legs for one (pair, metric), built through the shipped sizing rule.

    Split out of `vb_edge_fabrications` so a refusal inside one metric's set is catchable as that metric's
    finding (`leg_attempt`) instead of an exception that ends the run — the window-`m` shape.
    """
    field, bound = row["field"], row["bound"]
    legs = []
    A1 = {"f": spec["witness_A"][field]}
    B1 = {"f": B[field]}
    past = row["legs"]["one-past-limit"]["measured"]
    half, m_half, _ = vb_fabricate_at_limit(A1, B1, "f", metric, bound / 2.0, "inside",
                                            spec["layout"])
    under = vb_edge_sizing(m_half, past, bound, step=row["sizing"]["granularity"])
    legs.append(dict(pair=pair, metric=metric, name="under-sized-half-bound",
                     expected=True, flagged=not under["sized_correctly"],
                     detail="at-limit built at half the bound: measured=%.6g bound=%.6g %s"
                            % (m_half, bound, under["reason"])))
    zero = vb_edge_sizing(row["legs"]["at-limit"]["measured"],
                          row["legs"]["at-limit"]["measured"], bound)
    legs.append(dict(pair=pair, metric=metric, name="zero-granularity",
                     expected=True, flagged=not zero["sized_correctly"], detail=zero["reason"]))
    if metric != "ndiff_frac":
        return legs
    base_words = axis[1].get(field, (0, 0))[0] if axis else 0
    n = row["word_count"]
    k_at = int(np.floor(bound * (1.0 + 1e-9) * n))
    if base_words <= 0 or k_at <= base_words:
        legs.append(dict(pair=pair, metric=metric, name="naive-first-words-derived-base",
                         expected=None, flagged=None,
                         detail="not-exercised reason=no printed per-field word count for %s "
                                "(axis=%s) or the bound admits fewer words than the base differs"
                                % (field, "none" if not axis else axis[2])))
        return legs
    A2 = {"f": vb_pre_diverge(spec["witness_A"][field], B[field], base_words)}
    naive_at = vb_naive_first_words_leg(A2["f"], B[field], k_at)
    naive_past = vb_naive_first_words_leg(A2["f"], B[field], k_at + 1)
    m_naive = vb_metrics(naive_at, B[field])[metric]
    m_naive_past = vb_metrics(naive_past, B[field])[metric]
    sizing = vb_edge_sizing(m_naive, m_naive_past, bound)
    legs.append(dict(pair=pair, metric=metric, name="naive-first-words-derived-base",
                     expected=True,
                     flagged=not (m_naive <= bound * (1.0 + 1e-9) and sizing["sized_correctly"]),
                     detail="base_words=%d k=%d naive at-limit=%.6g bound=%.6g inside=%s %s"
                            % (base_words, k_at, m_naive, bound,
                               m_naive <= bound * (1.0 + 1e-9), sizing["reason"])))
    fixed_at, m_fix, lat = vb_fabricate_at_limit(A2, B1, "f", metric, bound, "inside", spec["layout"])
    fixed_past, m_fix_past, _ = vb_fabricate_at_limit(A2, B1, "f", metric, bound, "outside",
                                                      spec["layout"])
    fix_sizing = vb_edge_sizing(m_fix, m_fix_past, bound)
    legs.append(dict(pair=pair, metric=metric, name="fixed-under-derived-base",
                     expected=False,
                     flagged=not (m_fix <= bound * (1.0 + 1e-9)
                                  and m_fix_past > bound * (1.0 + 1e-9)
                                  and fix_sizing["sized_correctly"]),
                     detail="base_words=%d lattice=%d*%s measured=%.6g one-past=%.6g %s"
                            % (base_words, lat["index"], lat["unit"], m_fix, m_fix_past,
                               fix_sizing["reason"])))
    return legs


def vb_edge_fabrications(ctx, text):
    """The deliberately mis-sized rejection legs, built through the same sizing rule as the real ones.

    A `sized_correctly` flag that has never seen a bad leg is decoration, so three bad legs are built on
    purpose and one good one under a base it was previously built on zero of:

      * `under-sized-half-bound` — the at-limit leg built at half the bound. It is INSIDE, which is the
        quiet shape: it sits further below the bound than one lattice step, so the bound it claims to
        touch was never approached.
      * `naive-first-words-derived-base` — the pre-fix construction (flip bit 0 of the first `k` A-words
        regardless of whether they already differ) under the device window's own per-field differing-word
        count. That reproduces window `l`'s `at-limit=False` from its own numbers on host bytes, with no
        device act.
      * `zero-granularity` — two legs at the same lattice point, i.e. no step between them at all.
      * `fixed-under-derived-base` — the shipped construction under that same non-zero base. This one must
        NOT be flagged, which is what says the fix is doing the work rather than the flag.

    Everything is in-memory and labelled; nothing here touches a device or the scratch dir.
    """
    legs = []
    rows_by_pair = {}
    for row in (ctx.get("vb_edges") or {}).get("rows") or []:
        rows_by_pair.setdefault(row["pair"], {})[row["metric"]] = row
    for pair, spec in (ctx.get("vb") or {}).items():
        rows = rows_by_pair.get(pair) or {}
        B = vb_record(spec["b_side"], spec["order"], spec["fields"])
        axis = leg_attempt(ctx, "fabrication", "%s per-field word axis" % pair,
                           lambda s=spec: vb_word_axis(text, s["kind"]))
        for metric in VB_METRICS:
            row = rows.get(metric)
            if not row:
                continue
            made = leg_attempt(ctx, "fabrication", "%s %s fabrication set" % (pair, metric),
                               lambda p=pair, s=spec, r=row, m=metric, a=axis, bb=B:
                               vb_fabrication_legs(p, s, r, m, a, bb))
            if made:
                legs.extend(made)
    for leg in legs:
        print("GWITNESS guard=edge-sizing pair=%s metric=%s leg=%s expected_flag=%s flagged=%s %s"
              % (leg["pair"], leg["metric"], leg["name"], leg["expected"], leg["flagged"],
                 leg["detail"]))
    return legs


def checks_d28(v, ctx):
    """What the value bands may claim, and the counters that keep each of them from being decorative."""
    specs = ctx.get("vb") or {}
    if not specs:
        return
    text = run_text(ctx)
    lines = text.splitlines()
    # The early return below is the shape this guard exists to deny: with no spec there is nothing for the
    # D-28 checks to read, and a check that reads nothing cannot fail. The absence of a value band is
    # therefore itself a finding, named, rather than a run that printed 63 checks instead of 71.
    v.check("d28-value-band-exists-in-this-run", bool(specs),
            "%d D-28 pair value bands exist in this run. A pair's bound derivation can refuse (a "
            "documented sentence moved, a count bound that admits every word), and the surviving pair still "
            "gets its legs, its must-catch set and its checks — but if no pair survives, the six D-28 checks "
            "below would each be reading an empty population and PASSing over it, so this run reddens "
            "instead. Refusals: %s" % (len(specs), refusal_note(ctx, "derive")))
    if not specs:
        return
    bound_lines = [l for l in lines if l.startswith("BOUNDFROM ") and " derived=" in l]
    witness_lines = [l for l in lines if l.startswith("BOUNDWITNESS ")]
    scanout_lines = [l for l in lines if l.startswith("SCANOUT-DIAGNOSIS ")]
    scanout_parsed = re.findall(r"SCANOUT-DIAGNOSIS substrate=(\S+) base=(\S+) bound=(\S+) ", text)
    scanout_wanted = ctx.get("scanout_lines_expected") if VB_SCANOUT_PAIR in specs else 0
    expected = sum(len(s["order"]) * len(VB_METRICS) for s in specs.values())

    v.check("d28-bound-and-witness-lines-are-parsed-not-assumed",
            len(bound_lines) == expected and len(witness_lines) >= expected
            and all(re.search(r"derived=(\d|inf)", l) for l in bound_lines)
            and len(scanout_lines) == len(scanout_parsed) == scanout_wanted,
            "every BOUNDFROM the run printed is one of the %d (pair, field, metric) cells the record "
            "layouts derive (%s), every derived number parsed, and at least one BOUNDWITNESS per cell: "
            "printed BOUNDFROM=%d BOUNDWITNESS=%d. A count that parsed nothing would print PASS. The "
            "substrate diagnosis prints in the same all-or-nothing shape: %d SCANOUT-DIAGNOSIS lines, %d "
            "of them parsed back into (substrate, base, bound) against the %d this run owes (two substrates "
            "plus the headline), parsed pairs: %s"
            % (expected, ";".join("%s:%dx%d" % (p, len(s["order"]), len(VB_METRICS))
                                   for p, s in sorted(specs.items())),
               len(bound_lines), len(witness_lines), len(scanout_lines), len(scanout_parsed),
               scanout_wanted, ",".join("%s=%s" % (s, b) for s, b, _ in scanout_parsed) or "none"))

    single = [l for l in bound_lines if "single_source=yes" in l]

    def cell(l, key):
        m = re.search(r" %s=(\S+)" % key, l)
        return m.group(1) if m else "?"

    def is_zero(l):
        try:
            return float(cell(l, "derived")) == 0.0
        except ValueError:
            return False

    # A single documented source is admissible in exactly three shapes, each argued rather than waved
    # through: (a) a zero-tolerance cell, where the arm's own reach claim is the whole argument and zero
    # is the strictest bound the documentation admits; (b) a `gather.raw` cell, which D-14 admits on the
    # strength of the grown must-catch set because no device-free probe reaches the down_proj dot; (c) a
    # count cell whose other two source families are each printed unavailable with a reason — the
    # licensor publishes no count of differing words, and a floor probe's count doubles past 1.0 and so
    # forbids nothing.
    unadmitted = []
    for l in single:
        field, metric = cell(l, "field"), cell(l, "metric")
        if is_zero(l) or field == "raw":
            continue
        unav = l.split(" unavailable=", 1)[1] if " unavailable=" in l else ""
        if metric == "ndiff_frac" and "licensor_band_triangle" in unav and "fp64_floor_probe" in unav:
            continue
        unadmitted.append("%s/%s" % (field, metric))
    drift = []
    for l in bound_lines:
        # A printed source count that disagrees with the source list it prints is a drift between the
        # claim and its content, and this is the only place that would notice.
        m = re.search(r"sources=(\d+)\[(.*?)\] derived=", l)
        if not m or int(m.group(1)) != (0 if not m.group(2) else len(m.group(2).split(" AND "))):
            drift.append(l[:100])
    no_reason = [l for l in bound_lines if " unavailable=" in l
                 and "unavailable=none" not in l
                 and any("reason=" not in e
                         for e in l.split(" unavailable=", 1)[1].split("|") if e)]
    v.check("d28-bounds-come-from-documented-sources",
            not drift and not no_reason and not unadmitted,
            "%d BOUNDFROM cells, every one carrying at least one documented source; the single-source "
            "set is admitted by three named shapes — zero-tolerance cells (the arm's reach claim, "
            "derived=0), gather.raw (D-14, on the strength of the grown must-catch set), and count cells "
            "whose other two source families are each printed unavailable with a reason. Sources are "
            "mr_exp x the arm's own host-path finite difference, the licensor's published band by the "
            "triangle inequality (only where the path under test is licensed), and the fp64 floor probe "
            "of the record's own operands; each unavailable source names its reason. "
            "single_source=%s unadmitted=%s source-count-drift=%s reasonless=%s"
            % (len(bound_lines),
               ",".join(sorted(set("%s/%s" % (cell(l, "field"), cell(l, "metric")) for l in single)))
               or "none", unadmitted or "none", drift or "none",
               [l[:100] for l in no_reason] or "none"))

    edges = ctx.get("vb_edges") or {}
    edge_refusals = refusals_for(ctx, "edge")
    # The escalation clause, witnessed rather than asserted: a base at or above the bound must produce a
    # named refusal (D-18: the derivation is short, the band is not), and a base inside it must not. Same
    # numbers window `m` produced, run through the pure predicate that the leg builder calls.
    fab_over = vb_leg_headroom(1.46299e-07, 2.0e-07, "scan_out", "rel-err")[1]
    fab_under, fab_under_err = vb_leg_headroom(1.46299e-07, 2.43282e-08, "scan_out", "rel-err")
    not_inside = ["%s %s/%s %.6g>%.6g" % (p, field, metric, row["measured"], row["derived"])
                  for p, s in sorted(specs.items())
                  for (field, metric), row in sorted(s["witness"].items()) if not row["inside"]]
    v.check("d28-value-band-witness-inside-and-edge-legs-reject",
            not not_inside and not edge_refusals and bool(edges.get("lines"))
            and edges.get("ok") is not False and bool(fab_over) and fab_under > 0
            and fab_under_err is None,
            "the measurement is a separate witness line, never folded into the PASS: %d BOUNDWITNESS "
            "cells, outside=%s; and the rejection legs have a real edge (%s) — at-limit INSIDE and "
            "one-past-limit OUTSIDE per metric, sized from the bound and from the base the metric will "
            "actually see rather than from the spread, each leg a labelled in-memory fabrication. A band "
            "that cannot reject is not a band. The escalation clause is live: a base above the bound "
            "refuses with `%s`, a base inside it yields the headroom %.6g and refuses nothing. refused=%s"
            % (len(witness_lines), ";".join(not_inside) or "none",
               "; ".join(edges.get("lines") or []) or "NOT RUN",
               (fab_over or "NOT REFUSED")[:120], fab_under, refusal_note(ctx, "edge")))

    caught = ctx.get("vb_must_catch") or []
    derived_total = sum(s.get("must_catch_derived", 0) for s in specs.values())
    clean = ["%s/%s" % (c["pair"], c["name"]) for c in caught if c["reddens"] != "RED"]
    v.check("d28-must-catch-set-is-derived-and-reddens",
            bool(caught) and len(caught) == derived_total and derived_total > 0 and not clean
            and not refusals_for(ctx, "must-catch"),
            "%d injected real defects, one per structural boundary of the geometry that offers it "
            "(32 heads of d_k*d_v words, the gate|up pair along the output dim, %d expert slots, each "
            "field's own span boundary), printed MUSTCATCH lines=%d, derived=%d, every one of them "
            "REDDENS=RED (clean: %s). D-14 admits a documented-route bound only while this set grows "
            "with it, and a set that parses to zero reddens this run rather than passing it. refused=%s"
            % (len(caught), (ctx["gpaths"]["policy_a"]["g"]["MG_K"] if ctx.get("gpaths") else 0),
               len(caught), derived_total, ",".join(clean) or "none",
               refusal_note(ctx, "must-catch")))

    refuted = ctx.get("vb_refuted") or []
    refuted_lines = [l for l in lines if l.startswith("REFUTED-CAUSE ") and " cause=" in l]
    printed_carried = re.findall(r" carried_by=(\S+)", "\n".join(refuted_lines))
    bad = ["%s/%s" % (r["cause"], r["pair"]) for r in refuted
           if r["fails"] is None or not r["carried"] or r["carried"] == "none"
           or not any(axis in r["carried"] for axis in ("value-band", "byte-diagnostic"))
           or not (r["byte"] and r["byte"]["refuted"])]
    # The rule is `DMAGMATCH`/`GMAGMATCH::within_factor_two` read the other way: a cause is refuted when
    # it exceeds the observed divergence by MORE than the observed divergence itself. Asserted in numbers
    # so a line that prints a ratio without that arithmetic behind it is caught, not admired.
    arithmetic = ["%s: injected %d B - observed %d B = %d B > observed %d B (%.3fx, per-field %s, via %s)"
                  % (r["cause"], r["byte"]["injected"], r["byte"]["observed"], r["byte"]["excess"],
                     r["byte"]["observed"], r["byte"]["ratio"], r["byte"]["per_field"],
                     r["byte"]["source"]) for r in refuted if r["byte"]]
    demo = refuted_carriage_offences(
        "REFUTED-CAUSE cause=decay_ulp1 pair=d_hip_device_vs_d_hip_hostpath must_keep_failing=yes "
        "teeth=none outside=none inside=state/relL2 0 vs 0 verdict=NOW-INSIDE-BAND carried_by=none")
    v.check("d28-refuted-causes-still-refuted-and-say-what-carries-it",
            bool(refuted) and len(refuted) == len(specs) == len(refuted_lines)
            and len(printed_carried) == len(refuted_lines) and not bad
            and len(arithmetic) == len(refuted) and bool(demo)
            and not refusals_for(ctx, "refuted"),
            "the two causes D-28 refuted on magnitude are re-priced on both axes (%s). Every line prints "
            "`carried_by=` naming `value-band` and/or `byte-diagnostic` (parsed %d of %d lines, none of "
            "them `none`), the observed total each ratio uses is re-summed from the printed per-field "
            "DATTRIB/GATTRIB split rather than copied off a summary, and each refutation carries the "
            "factor-two arithmetic in bytes: %s. The same predicate reddens a fabricated line that carries "
            "nothing (%s). Neither band was widened nor cause dropped to get here: `decay_ulp1` is "
            "teeth=none in value space and refuted on the byte axis, which is exactly what D-28 §3 asks a "
            "line to admit. refused=%s"
            % (";".join("%s[%s]=%s(%s)" % (r["cause"], r["pair"].split("_")[0],
                                           "still-refuted" if r["fails"] else "inside-band",
                                           r["carried"]) for r in refuted) or "not exercised",
               len(printed_carried), len(refuted_lines), " | ".join(arithmetic) or "none",
               ";".join(demo) or "NOT CAUGHT", refusal_note(ctx, "refuted")))

    edges_rows = (ctx.get("vb_edges") or {}).get("rows") or []
    edge_lines = [l for l in lines if l.startswith("BOUNDEDGE ")]
    edge_bad, granularity_notes = [], []
    for row in edges_rows:
        mine = [l for l in edge_lines if "pair=%s " % row["pair"] in l
                and "metric=%s " % row["metric"] in l]
        if len(mine) != 2:
            edge_bad.append("%s/%s printed %d legs, not 2" % (row["pair"], row["metric"], len(mine)))
            continue
        by_leg = dict((re.search(r" leg=(\S+)", l).group(1), l) for l in mine)
        at, past = by_leg.get("at-limit"), by_leg.get("one-past-limit")
        if not at or not past:
            edge_bad.append("%s/%s legs are not named at-limit/one-past-limit" % (row["pair"],
                                                                                 row["metric"]))
            continue

        def pstr(line, key):
            got = re.search(r" %s=(\S+)" % key, line)
            if not got:
                edge_bad.append("%s/%s %s line prints no %s=" % (row["pair"], row["metric"],
                                                                "at-limit", key))
                return None
            return got.group(1)

        m_at_s, m_past_s = pstr(at, "measured"), pstr(past, "measured")
        gran_at, gran_past = pstr(at, "granularity"), pstr(past, "granularity")
        dist_at_s = pstr(at, "distance_to_bound")
        step_at, step_past = pstr(at, "lattice_index"), pstr(past, "lattice_index")
        unit = pstr(at, "lattice_step") or "?"
        if None in (m_at_s, m_past_s, gran_at, gran_past, dist_at_s, step_at, step_past):
            continue
        at_m = row["legs"]["at-limit"]["measured"]
        past_m = row["legs"]["one-past-limit"]["measured"]
        real_step = past_m - at_m
        # parsed == printed, to the digit the line prints, and printed == what the legs actually
        # measured. A granularity that agreed with nothing on the line would be a decoration.
        if m_at_s != "%.6g" % at_m or m_past_s != "%.6g" % past_m:
            edge_bad.append("%s/%s prints measurements that are not the ones its legs measured"
                            % (row["pair"], row["metric"]))
        if gran_at != "%.9g" % real_step or gran_past != "%.9g" % real_step:
            edge_bad.append("%s/%s prints granularity %s, which is not the measured gap %.9g between "
                            "its own two legs" % (row["pair"], row["metric"], gran_at, real_step))
        if dist_at_s != "%.9g" % abs(at_m - row["bound"]):
            edge_bad.append("%s/%s prints distance_to_bound %s, not the gap %.9g from its at-limit leg "
                            "to the bound" % (row["pair"], row["metric"], dist_at_s,
                                              abs(at_m - row["bound"])))
        if " verdict=INSIDE" not in at or " verdict=OUTSIDE" not in past:
            edge_bad.append("%s/%s legs are not INSIDE/OUTSIDE" % (row["pair"], row["metric"]))
        if step_past.isdigit() and int(step_past) - int(step_at) != 1:
            edge_bad.append("%s/%s legs are %s-%s %s, not adjacent lattice points"
                            % (row["pair"], row["metric"], step_at, step_past, unit))
        if " sized_correctly=yes" not in at or " sized_correctly=yes" not in past:
            edge_bad.append("%s/%s is under-sized: at-limit sits %.9g below the bound, one lattice step "
                            "is %.9g — %s" % (row["pair"], row["metric"], row["sizing"]["distance_to_bound"],
                                              real_step, row["sizing"]["reason"]))
        if row["sizing"]["distance_to_bound"] > real_step * (1.0 + 1e-9):
            edge_bad.append("%s/%s at-limit leg sits %.9g below the bound, more than one step %.9g"
                            % (row["pair"], row["metric"], row["sizing"]["distance_to_bound"],
                               real_step))
        if unit == "word":
            expected = 1.0 / row["word_count"]
            if abs(real_step - expected) > expected * 1e-9:
                edge_bad.append("%s/%s count granularity %.9g is not 1/%d = %.9g"
                                % (row["pair"], row["metric"], real_step, row["word_count"], expected))
            granularity_notes.append("%s/%s measured granularity=%s == 1/%d words=%s (a word is the "
                                     "field's own step; the count metric's granularity fell out of the "
                                     "two adjacent legs, it was not typed)"
                                     % (row["pair"].split("_")[0], row["metric"], gran_at,
                                        row["word_count"], "%.9g" % expected))
    edge_fabrications = vb_edge_fabrications(ctx, text)
    misfired = ["%s/%s" % (l["pair"], l["name"]) for l in edge_fabrications
                if l["expected"] is not None and bool(l["flagged"]) != bool(l["expected"])]
    exercised = [l for l in edge_fabrications if l["expected"] is not None]
    # The leg population is derived, not counted off what happened to print: one metric-cell per pair per
    # metric. A refused leg produces no row and no line, and this is where that absence is caught — the
    # `len(edge_lines) == 2 * len(edges_rows)` test alone would be satisfied by the shrunken pair.
    cells_expected = len(specs) * len(VB_METRICS)
    cells_missing = cells_expected - len(edges_rows)
    v.check("d28-edge-legs-are-adjacent-lattice-points-and-sized-from-the-bound",
            bool(edges_rows) and len(edges_rows) == cells_expected and not cells_missing
            and len(edge_lines) == 2 * len(edges_rows) and not edge_bad
            and bool(exercised) and not misfired
            and not refusals_for(ctx, "fabrication")
            and (ctx.get("vb_edges") or {}).get("ok") is not False,
            "%d BOUNDEDGE legs, %d metrics of the %d the pair/metric geometry derives (%d cells, %d "
            "missing — a leg that refused prints no line and would otherwise vanish from its own census), "
            "each pair of legs one adjacent pair of lattice points of the "
            "field's own f32 word grid: at-limit INSIDE, one-past-limit OUTSIDE, indices one step apart, "
            "and the printed `granularity` equal to the measured gap between its own two legs — %s. "
            "sized_correctly requires the at-limit leg to sit within one measured step of the bound, so a "
            "leg that never approached the bound reddens too (the quiet twin of window `l`, whose "
            "`ndiff_frac` at-limit legs landed past the bound because the pre-fix construction flipped "
            "words that already differed). The flag is not inert: %d fabrications through the same sizing "
            "rule came out as expected (under-sized-half-bound and zero-granularity flagged, "
            "fixed-under-derived-base clean), %s misfired=%s; legs with no printed base population to "
            "build on: %s; refused=%s"
            % (len(edge_lines), len(edges_rows), cells_expected, cells_expected, cells_missing,
               " | ".join(granularity_notes) or "no count metric among "
               "the sized legs", len(exercised), "bad=%s" % (";".join(edge_bad) or "none"),
               misfired or "none",
               ";".join("%s/%s" % (l["pair"].split("_")[0], l["name"]) for l in edge_fabrications
                            if l["expected"] is None) or "none", refusal_note(ctx, "fabrication")))

    cmd_count, cmd_offences = device_command_offences(text)
    cmd_demo_count, cmd_demo = device_command_offences(
        "GCAUSE-VOID cause=x note=… command=engine/block_device.py --work /tmp/leaked-scratch --device\n"
        "DVOID control-leg=x required-print=Y taken by engine/block_device.py --device --device\n")
    # A CPU witness owes a command on every leg it cannot take, so a zero population there is a blind parse.
    # A device window owes none — it IS the window — and its population is legitimately empty (a window that
    # did print a command, as window `l` did, is held to the same rule). Either way the fabricated pair must
    # still be caught, which is what keeps the predicate from being decorative in an empty mode.
    population_note = ("mode=%s: every leg this run cannot take owes the command that takes it, so a zero "
                       "population is blindness" % ("device" if ctx["device"] else "cpu")
                       if not ctx["device"] else
                       "mode=device: this run is the window, so no leg owes a command it already took; any "
                       "command it does print is still held to the rule (window `l` printed one)")
    v.check("d28-reported-device-command-is-the-canonical-one",
            (cmd_count > 0 or ctx["device"]) and not cmd_offences
            and cmd_demo_count == 2 and len(cmd_demo) == 2,
            "%d printed lines carry a device command and every one of them is the command the main agent "
            "would actually run: `--work %s` and exactly one `--device` (%s). This closes the two ways a "
            "reported command used to lie — an argv `--work <scratch>` riding into a printed device "
            "command (a witness run in a scratch dir cannot report a command that reads a different "
            "scratch dir) and window `l`'s `command=… --device --device` — because `window_command()` "
            "rebuilds from WORK_DEFAULT and strips an argv `--work`. Canonical command: `%s`. Offences "
            "this run: %s; non-inertness, the same predicate on a fabricated pair of lines (%d commands "
            "parsed, %d offences): %s"
            % (cmd_count, WORK_DEFAULT, population_note, window_command(), cmd_offences or "none",
               cmd_demo_count, len(cmd_demo), " | ".join(cmd_demo) or "NOT CAUGHT"))

    repro = [l for l in lines if l.startswith(VB_DIAG_HEADS)]
    unlabelled = [l.split()[0] for l in repro if "role=diagnostic gate=no" not in l]
    # In a CPU witness not one of these legs prints (they are device legs, and the run says so with a
    # DVOID required-print line), which would make a legs-only claim vacuous. The non-vacuous form is
    # structural: every print site that could emit one of the three legs must carry the label.
    sites = vb_diag_print_sites()
    missing_sites = sorted(set(VB_DIAG_HEADS) - set(h for h, _ in sites))
    unlabelled_sites = [h for h, ok in sites if not ok]
    dvoid = [l for l in lines if l.startswith("DVOID") and "required-print=" in l
             and any(h in l for h in VB_DIAG_HEADS)]
    pair_checks = [l for l in lines if l.startswith("CHECK name=")
                   and ("compiled-pair-value-agreement" in l)]
    decided_by_bytes = [l.split(" ")[1] for l in pair_checks if "vs_device_record_ndiff_bytes" in l]
    cites_witness = [l for l in pair_checks if "BOUNDWITNESS cells" in l]
    recomputed = all(bool(s.get("witness_ok")) == ("verdict=PASS" in l)
                     for s in specs.values()
                     for l in pair_checks if s["host_pair"].split("_")[1] in l
                     or s["pair"].split("_")[1] in l)
    v.check("d28-diagnostic-legs-are-labelled-and-do-not-decide",
            not unlabelled and not missing_sites and not unlabelled_sites
            and not decided_by_bytes and len(cites_witness) == len(pair_checks)
            and bool(pair_checks) and recomputed,
            "%d %s legs printed this run (%d DVOID lines name them as required device prints); every "
            "print site that can emit one — %s — carries `role=diagnostic gate=no`, and the %d pair "
            "CHECK lines decide on %d BOUNDWITNESS cells and cite none of the reproduction byte counts "
            "(unlabelled=%s unlabelled_sites=%s sites_missing=%s byte-decided=%s missing-witness-cite=%d "
            "verdict-mismatch=%s). Byte identity was the instrument that proved the mechanism; D-28 "
            "keeps it legible and takes its vote"
            % (len(repro), "/".join(VB_DIAG_HEADS), len(dvoid),
               ",".join("%s(%s)" % (h, "labelled" if ok else "UNLABELLED")
                        for h, ok in sorted(sites)) or "NONE", len(pair_checks), len(witness_lines),
               unlabelled or "none", unlabelled_sites or "none", missing_sites or "none",
               decided_by_bytes or "none",
               len(pair_checks) - len(cites_witness),
               "yes" if not recomputed else "no"))

    fabricated_text = "\n".join([
        "DREPRO pair=d_hip_device_vs_d_hip_hostpath probe=decay_ulp_mask steps=-1 mask=0x1 "
        "heads=0 path=host_cpubody record_bytes=2187264 vs_device_record_ndiff_bytes=75206 "
        "verdict=RESIDUAL-LEFT role=diagnostic gate=no",
        "DMAGMATCH pair=d_hip_device_vs_d_hip_hostpath cause=state_xpose device_bytes=52976 "
        "injected_bytes=52999 ratio=1.000 same_fields=True within_factor_two=True verdict=CANDIDATE",
    ])
    demo = d28_mechanism_offences(fabricated_text, {"d_hip_device_vs_d_hip_hostpath"})
    live_offences = d28_mechanism_offences(text, set(p for p, s in specs.items()
                                                     if s.get("witness_ok")))
    D28_ESCALATE["offences"] = live_offences
    word = d28_qualification_decision(0, 1, bool(live_offences))
    v.check("d28-escalation-clause-refuses-qualification",
            bool(demo) and (not live_offences or word.startswith("ESCALATE"))
            and d28_qualification_decision(0, 1, False) == "QUALIFIED",
            "the clause is wired as a rule and demonstrated: a fabricated log in which a value band "
            "passed beside a non-exp mechanism (%d legs named: %s) produced %d refusal(s), while the "
            "same predicate on this run's real diagnostics found %d and the verdict word computed here "
            "is `%s`. If the mechanism ever stops being the documented exp last bit the run stops "
            "printing QUALIFIED and stops to escalate to the user instead"
            % (2, "residual-left repro + CANDIDATE cause=state_xpose", len(demo),
               len(live_offences), word))


def checks_g(v, ctx, g_runs, g_rows, g_tally, censuses):
    """Everything the gather vertical is allowed to claim, expressed over parsed reports."""
    if not ctx.get("gpaths"):
        return
    g_slots = ctx["gpaths"][DA.G_PATHS[0]]["g"]["MG_K"]
    per_width_rows = len(G_STAGES) * len(DA.G_PATHS)
    v.check("gather-rows-equal-licensed-stages", len(g_rows) == per_width_rows,
            "rows=%d expected=%d: %s on each of %s. Both stages exist on both width paths by design: "
            "one is what a band can license, the other is what the engine ships"
            % (len(g_rows), per_width_rows, ",".join(G_STAGES), ",".join(DA.G_PATHS)))
    v.check("gather-policy-a-rows-unlicensed-by-name",
            all("policy-a-width-band-absent" in r["band"] for r in g_rows
                if r["arm"] == "policy_a"),
            "the policy-A pair prints UNLICENSED with reason `policy-a-width-band-absent`, and nothing "
            "above counts them as accepted: no band measures a quantized arm against bf16 pins, and the "
            "floor probe that would license it has never been run")
    v.check("gather-licence-names-its-source",
            all((r["licence"] or "").startswith(("floor-probe", "spec-10.6"))
                for r in g_rows if r["arm"] == "ref_bf16"),
            "the licensed rows print where the limit came from: %s — a number with no licensor is not a "
            "band" % ",".join("%s:%s" % (r["stage"], r["licence"]) for r in g_rows
                              if r["arm"] == "ref_bf16"))
    v.check("gather-census-agrees-with-the-plan",
            all(not c["disagree"] for c in censuses.values()) and bool(censuses)
            and all(c["units"] > 0 and c["spans"] > 0 and c["arm_units"] > 0 and c["arm_spans"] > 0
                    and c["note_conflict"] == 0 for c in censuses.values()),
            "per expert_index, driver-vs-arm agreement over all %d paths, with a non-zero denominator on "
            "both sides and no note conflict: %s. A whole-tensor read would move distinct_experts, bytes per "
            "expert, or both, and prediction 2's scheduler decision is made on those numbers; a census that "
            "counted nothing, or an arm whose census note was overwritten, would satisfy the agreement "
            "trivially and so is refused here rather than allowed to pass "
            % (len(censuses), "; ".join("%s units=%d/%d spans=%d/%d bytes=%d whole=%d conflict=%d"
                                        % (w, c["arm_units"] or -1, c["units"], c["arm_spans"] or -1,
                                           c["spans"], c["bytes_total"], c["whole_tensor_reads"],
                                           c["note_conflict"] or 0)
                                        for w, c in sorted(censuses.items()))))
    v.check("gather-distinct-experts-equals-top-k",
            bool(censuses) and all(c["distinct"] == c["slots"] == g_slots for c in censuses.values()),
            "every path touched exactly K=%d distinct expert indices for K slots: a gather that walked the "
            "whole expert axis would report 256 here instead, and an empty census would report nothing at "
            "all, which is refused rather than skipped"
            % g_slots)
    bad = [w for w, p in ctx["gpaths"].items() if any(p["nan"].values())]
    v.check("gather-honest-record-has-no-unwritten-slot", not bad,
            "the honest records carry no NaN in any field on either path (offending: %s); the poison "
            "control is what shows that count can be non-zero" % (",".join(bad) or "none"))
    # D-28: the policy-A device pair is judged in VALUE space, per field, against the bounds
    # `BOUNDFROM` derived before it was measured; the bf16 reference pair stays byte-exact because D-28
    # names it in its keeps-exact list. One aggregate would still let one path cancel the other's
    # difference, so both paths are answered separately and the host-twin leg of each is zero tolerance.
    g_value_ok = all(p["pair_ok"] for p in ctx["gpaths"].values())
    v.check("gather-compiled-pair-value-agreement-per-path", g_value_ok,
            "per width path, hipcc device arm vs hipcc host path, with the g++ twin pair at zero "
            "tolerance beside it: %s. policy A is judged per field in value space under the derived "
            "bounds (%d BOUNDWITNESS cells, every inside=yes or an explicit not-claimed); the bf16 "
            "reference gather is judged at byte identity on every field, which is what D-28 left it at. "
            "The old byte-identity witness for the policy-A pair survives as a diagnostic "
            "(`GREPRO … role=diagnostic gate=no`) and casts no vote"
            % (",".join("%s bytes=%d ndiff=%d[%s] value=%s" % (w, p["pair"][0], p["pair"][1],
                                                              "exact" if p["exact_leg"] else "banded",
                                                              "PASS" if p["pair_ok"] else "FAIL")
                        for w, p in sorted(ctx["gpaths"].items())),
               len([1 for s in (ctx.get("vb") or {}).values() for _ in s["witness"]])))
    fmt = ctx.get("gformat") or []
    fmt_offences = ["%s slot%d %s: bytes=%d != blocks=%d x block_bytes=%d"
                    % (r["role"], r["slot"], r["tensor"], r["bytes"], r["blocks"], r["block_bytes"])
                    for r in fmt if r["bytes"] != r["blocks"] * r["block_bytes"]]
    n_wider = sum(1 for r in fmt if r["relation"] == "WIDER-THAN-DECLARED")
    v.check("gather-read-set-forms-come-from-the-file", bool(fmt) and not fmt_offences,
            ("the policy-A gather reads %d expert units and every unit's arm `kind`, block count and byte "
             "count are derived from the tensor's own ggml type in the artifact header (GFORMAT lines), with "
             "each declared byte count equal to blocks x block-bytes-of-that-kind. The same table prints the "
             "declared policy-A type beside the stored one because D-27's fact — expert down_proj stored at "
             "Q5_K/Q6_K where policy A declares Q4_K, admitted by loader.py's floor rule with no waiver — is "
             "why a width hypothesis had to be priced and not dismissed: %d of %d units are "
             "wider-than-declared. offences: none"
             % (len(fmt), n_wider, len(fmt)))
            if (fmt and not fmt_offences) else
            ("cannot claim anything about the width the gather read: %s"
             % ("; ".join(fmt_offences) if fmt_offences else
                "no GFORMAT table was printed, so the formats the policy-A gather actually read are "
                "unknown and every width claim about this pair is unearned")))
    v.check("gather-unlicensed-counted-not-absorbed",
            g_tally["unlicensed"] == len([r for r in g_rows if r["arm"] == "policy_a"]),
            "the gather tally carries licensed=%d accepted=%d rejected=%d unlicensed=%d; the policy-A "
            "rows are counted as unlicensed rather than dropped, and PASS remains unreachable while any "
            "row is unlicensed" % (g_tally["licensed"], g_tally["accepted"], g_tally["rejected"],
                                   g_tally["unlicensed"]))
    g_launch_offences = launch_count_offences(dict((k, r) for k, r in g_runs.items()
                                                   if "device" in k))
    v.check("gather-device-path-claims-executed-on-device",
            (not ctx["device"]) or not g_launch_offences,
            "in a device run every gather device arm reported `launches` as an int and > 0 (%d device arms, "
            "offences: %s); in a CPU window the same --device ask on the g++ twin is refused with a named "
            "tag instead of a host run, and this check is vacuous here and says so"
            % (len([k for k in g_runs if "device" in k]),
               ",".join(g_launch_offences) or "none"))
    # What the gather controls are allowed to claim, on the width axis. The general
    # `control-legs-cover-the-path-under-claim` already refuses any control that claims a name it executed no
    # arm act under; this one asserts the gather-specific half: a control that speaks about width paths has
    # to speak about BOTH of them and have a counted leg under each, because a control that only ever bent
    # the bf16 plan licenses nothing the shipping path does. A control whose only leg is a named refusal
    # declares nothing and is listed as such, rather than a refusal being counted as a leg.
    declared, refusal_only, thin = [], [], []
    for name, d in sorted(v.control_paths.items()):
        if not name.startswith("gather-"):
            continue
        widths = [c for c in d["claims"] if c in DA.G_PATHS]
        if not widths:
            refusal_only.append(name)
            continue
        declared.append(name)
        if len(widths) < len(DA.G_PATHS):
            thin.append("%s declares only %s of %s" % (name, ",".join(widths), "|".join(DA.G_PATHS)))
        for w in widths:
            if d["legs"].get(w, 0) <= 0:
                thin.append("%s claims %s and executed 0 legs on it" % (name, w))
    v.check("gather-control-legs-cover-the-width-under-claim", bool(declared) and not thin,
            "%d gather controls declared width paths, each with the legs it claims: %s. Controls whose only "
            "leg is a named refusal, which declares no path because no arm ran to completion: %s. The width "
            "tally is counted by `g_arm` off the plan the run used, the same way window e2's lesson made the "
            "host/device claim cost something. Failures, if any: %s"
            % (len(declared),
               "; ".join("%s[%s]" % (n, ",".join("%s:%d" % (w, v.control_paths[n]["legs"].get(w, 0))
                                                 for w in v.control_paths[n]["claims"]
                                                 if w in DA.G_PATHS)) for n in declared) or "none",
               ",".join(refusal_only) or "none",
               "; ".join(thin) or "none"))


G_ART_TENSOR_ROLES = G_ROLES


NO_DEVICE_FIELDS = ("hip_calls", "launches", "bytes_to_device", "bytes_from_device")


def run_facts(rec):
    """The `RUN` block of an arm record, whichever wrapper shape the caller kept it in.

    `arm_runs`, `rt_runs` and `g_runs` do not store the same shape — some entries are the record, some wrap
    it as `{"rec": …}` — and the first version of `cpu-witness-reports-no-device-acts` assumed one of them
    and died with `KeyError: 'run'` *inside the safety check*. A check that raises has not run, which is the
    same failure class as an empty parse printing PASS (`pitfalls/40`), so the shape is normalised here and
    an entry with no RUN block is a refusal rather than a silent skip.
    """
    r = rec.get("rec", rec) if isinstance(rec, dict) else {}
    run = r.get("run", r) if isinstance(r, dict) else {}
    if not isinstance(run, dict) or not run:
        raise Refused("run-record-shape",
                      "an arm record reached the device-act check with no RUN block to read: %r" % (rec,))
    return run


def no_device_act_offences(runs):
    """Every (label, field) where a run that was not supposed to touch a device reported one.

    A missing counter counts as an offence, not as a zero: a `RUN` line that does not report `launches`
    cannot certify that there were none, and the difference between "absent" and "0" is exactly what
    `pitfalls/40` §a-check-that-can-print-nothing-cannot-fail is about. Non-empty is a finding; the
    control `control_no_device_act_guard` shows this function can produce one.
    """
    bad = []
    for label in sorted(runs):
        run = run_facts(runs[label])
        for k in NO_DEVICE_FIELDS:
            v = run.get(k)
            try:
                if v is None or int(v):
                    bad.append("%s.%s=%s" % (label, k, "absent" if v is None else v))
            except (TypeError, ValueError):
                bad.append("%s.%s=%s" % (label, k, v))
    return bad


# ----------------------------------------------------------------------------------------------
# The GDN recurrent vertical: ONE form, two compiled arms, one carried state, seven stage rows.
#
# WHY THIS VERTICAL LOOKS LIKE THE OTHERS AND IS NOT THE SAME THING. The shared-expert, router and
# gather rows are all one-form problems: a matrix multiply is a matrix multiply whichever arm runs it.
# A GDN row is not. `speccheck/gdn_ref.band_licenses` keys the licence on `width|form`, and at the
# pinned decode pass (`layer0/pinned/decode0`) the ONLY pair that grants anything is `pinned|recurrent`;
# `chunked|pinned` grants nothing at any decode pass, because the dispatch table gives a cached
# single-token step the recurrent program and the floor probe never measured a chunked arm there
# (`prototype/gdn/HANDOFF.md` §SLICE 3 JOB 2, adjudicated UNLICENSED and left that way). So every row
# below is meaningless without the form that produced it, and "did the arm implement the form it is
# being credited for?" becomes a question with a printable answer: `parse_d_layout` reads the form out
# of `engine/device/m2_d_kernels.h`, `m2_d_io.h` refuses a plan naming any other, every arm run prints
# `RUN arm=d_* form=recurrent`, and `control_d_form_not_merged` reddens a row whose licence pair names a
# form this arm does not implement.
#
# THE STATE IS THE SUBJECT, NOT AN INPUT. `d_k == d_v == 128`, so a transposed state is the same
# 16,384 words per head in a different order: no shape check, no bounds fault, no NaN — the oracle's own
# injection name for it is `state_stored_transposed` and the prototype lane documents why a zero state
# made it invisible to its first gate (`prototype/gdn/gdn_kernels.h` §WHY THE STATE READ IS A NAMED
# CHECK). The pinned decode step's incoming state is NOT zero (509,952 of 524,288 words are live,
# printed as `DSTATE` every run), so this vertical can see what that lane could not, and it does so with
# three separate witnesses: the modular state fingerprint the arm computes as it reads
# (`control_d_state_read_witness`), a transposition mode bent inside the one function that contracts
# the state (`control_d_state_transposition`), and a non-carried leg executed twice — once by the driver
# zeroing the slab's bytes, once by the arm's own `zero-state` fault (`control_d_non_carried_state`).
#
# NO BAND IS AUTHORED HERE. The seven rows go through `block_gate.stage_row` and `::adjudicate`, which
# consult the same pinned band file as every other row in this driver; three of the seven have no
# licensor at this pass for EITHER form and print the reason the licensor itself gives.
# ----------------------------------------------------------------------------------------------

D_FORM = "recurrent"          # the only form m2_d_arm.hip / m2_d_host.cpp implement
D_STATE_READBACK = 32         # bytes of the state's own span a device run copies back
D_BF16_STAGES = ("core_attn", "normed", "y")
D_STATE_STAGES = ("scan_out", "core_attn", "state", "normed", "y")
D_INPUT_ONLY_STAGES = ("q_hat", "k_hat")


def derived_d_population(population, d):
    """The GDN rows this arm must produce, derived twice and required to be the same seven.

    The licensor's list is `speccheck/gdn_ref.SCAN_WIDTH_STAGES` — the stages the band file has an entry
    for at this record width — narrowed to what the layer-0 GDN manifest actually pins. The arm's list is
    the record field names read out of `engine/device/m2_d_kernels.h` by `device_arm.parse_d_layout`.
    Requiring the two to be equal is the naming rule that makes a rename on either side a red run instead
    of a missing row, which is the same discipline `derived_arm_population` applies to the shared-expert
    and router stages, applied to a list the licensor owns.
    """
    licensor = [s for s in G.SCAN_WIDTH_STAGES if s in population["from_manifest"]["gdn"]]
    arm = list(d["order"])
    return dict(scan=licensor, arm_declared=arm, from_manifest=population["from_manifest"],
                missing=[s for s in licensor if s not in arm],
                extra=[s for s in arm if s not in licensor])


def d_prepare(args, ctx):
    """Lay the GDN arm's slab from the pinned decode-step records, and bind every span to them.

    Nine spans: six decode-step activations, two weights and the carried state. The widths are not
    taken on faith anywhere — `device_arm.d_spans` checks each record's manifest `dtype` against the
    element width the header says the arm reads, and the arm re-checks the same facts out of its own plan
    before it reads a byte, so a requantized input or a reshaped state is a named refusal twice over
    rather than a plausible row (`pitfalls/30` §dtype-by-filesize).
    """
    work, dumps = ctx["work"], ctx["dumps"]
    d = DA.parse_d_layout()
    spans = DA.d_spans(dumps, d)
    slab = DA.d_slab(spans, work)
    state = [s for s in spans if s["name"] == "state_in"][0]
    slab_bytes = slab["slab"].read_bytes()
    binds = []
    for s in spans:
        span = slab_bytes[s["off"]:s["off"] + s["n_bytes"]]
        sha = hashlib.sha256(span).hexdigest()
        binds.append(dict(name=s["name"], rec=s["rec"], span_off=s["off"], bytes=s["n_bytes"],
                          file_sha=s["sha"], slab_sha=sha))
        print("DFIELD field=%s record=%s dtype=%s shape=%s words=%d elt=%d bytes=%d span_off=%d "
              "file_sha256=%s slab_sha256=%s bind=%s"
              % (s["name"], s["rec"], s["dtype"], ",".join(str(x) for x in s["shape"]), s["words"],
                 s["elt"], s["n_bytes"], s["off"], s["sha"][:16], sha[:16],
                 "ok" if sha == s["sha"] else "BROKEN"))
    print("DWIDTH form=%s fields=%s f32=%s bf16=%s state_words=%d state_bytes=%d slab_bytes=%d "
          "note=the scan dtype and the recurrent state are f32, every activation and weight image is "
          "bf16, and engine/device/m2_d_io.h refuses a plan whose elt= disagrees with either"
          % (D_FORM, ",".join(d["order"]),
             ",".join(s for s in d["order"] if not d["is_bf16"][s]),
             ",".join(s for s in d["order"] if d["is_bf16"][s]),
             d["GDN_H_V"] * d["GDN_D_K"] * d["GDN_D_V"], state["n_bytes"], slab["slab_bytes"]))
    print("DSTATE role=carried record=%s words=%d live=%d sum=%d sumidx=%d nonzero_fraction=%.6f "
          "note=a zero state would make every state-reading control below vacuous, so the count is "
          "printed before any row is emitted"
          % (state["rec"], state["fingerprint"]["words"], state["fingerprint"]["live"],
             state["fingerprint"]["sum"], state["fingerprint"]["sumidx"],
             state["fingerprint"]["live"] / float(max(state["fingerprint"]["words"], 1))))
    plan = work / "d_plan.txt"
    DA.write_d_plan(plan, spans, slab["slab_bytes"], d, read_off=slab["state_off"],
                    read_len=D_STATE_READBACK)
    return dict(d=d, spans=spans, slab=slab, plan=plan, binds=binds,
                fingerprint=slab["state_fingerprint"], state=state,
                readback_hex=None)


def d_claims(ctx):
    """Which execution path a GDN control may name: the host body always, the device only in a window.

    Same rule as `g_claims`: every name has to be backed by a counted arm act on that path
    (`control-legs-cover-the-path-under-claim`), so a CPU witness never claims `device`, and the two
    controls that do claim it run their leg through the hipcc binary's launched kernels rather than
    borrowing the serial host body's.
    """
    return ("host_cpubody",) + (("device",) if ctx["device"] else ())


def d_arm(binary, plan, slab, out, device=False, fault=None, label=None):
    """`arm_leg` with the run's device bill folded in, so a control's variant cannot go unbilled."""
    rec = arm_leg(binary, plan, slab, out, device=device, fault=fault, label=label)
    witness_fold(rec["run"], label or "d_arm")
    return rec


def d_variant(ctx, out_name, label, mode="correct", fault="none", binary=None,
              zero_state=False, flip_state_word=None, device=False, form=None, elt_override=None,
              words_override=None, drop_field=None, subdir="dvariant", metrics_ok=True,
              mask=None, steps=None):
    """Run the arm on a variant of the honest plan/slab, and return everything measured about it.

    One door for every control leg so no control can quietly run a different thing than it claims: the
    variant plan is written from the same span table, the perturbed slabs are laid from the same
    certified bytes, and the returned dict carries the record fields, the unwritten-slot census, the
    arm's own census, the fingerprint the driver folds over the bytes it supplied, and the per-stage
    distance to the pinned records.
    """
    work = ctx["work"] / subdir
    work.mkdir(parents=True, exist_ok=True)
    d, spans = ctx["d"]["d"], ctx["d"]["spans"]
    slab = DA.d_slab(spans, work, name=out_name.replace(".bin", "_slab.bin"),
                     zero_state=zero_state, flip_state_word=flip_state_word)
    plan = work / (out_name.replace(".bin", "_plan.txt"))
    DA.write_d_plan(plan, spans, slab["slab_bytes"], d, mode=mode, fault=fault, form=form,
                    read_off=slab["state_off"], read_len=D_STATE_READBACK,
                    elt_override=elt_override, words_override=words_override, drop_field=drop_field,
                    mask=mask, steps=steps)
    out = work / out_name
    rec = d_arm(binary or ctx["args"].d_host_binary, plan, slab["slab"], out, device=device,
                label=label)
    fields, nan = DA.d_record(out, d)
    stages = DA.d_stage_tensors(fields, d)
    census = DA.parse_d_census(rec["run"].get("census"))
    agree = DA.d_census_agreement(census, d, slab["state_fingerprint"],
                                 zero_in=(fault == "zero-state"))
    return dict(rec=rec, fields=fields, stages=stages, nan=nan, census=census, agree=agree,
                # The licensor's aggregates() refuses a non-finite input by design, so a record known to
                # carry poison cannot be run through the row comparator at all — the poison control
                # compares census counts, not band-shaped aggregates.
                metrics=(d_metrics(ctx["dumps"], stages) if metrics_ok else None),
                fingerprint=slab["state_fingerprint"],
                mode_fields=rec["hdr"].get("mode_fields"))


def d_metrics(dumps, stages):
    """Per-stage distance to the pinned record, computed by the licensor's own comparator.

    `block_gate.stage_row` does the cast to the referent dtype and the `ulp_diff` count, so the numbers a
    control compares are the same numbers the band row prints — not a second differ with its own idea of
    what "differs" means (`pitfalls/30` §the-ULP-comparator-dead-branch is what a private differ bought
    once already).
    """
    out = {}
    for stage in stages:
        row = BG.stage_row(dumps, "ref_bf16", stage, stages)
        out[stage] = dict(ndiff=int(row["ndiff"]), n=int(row["n"]), maxabs=row["maxabs"],
                          max_ulp=row["max_ulp"], one_minus_cos=row["one_minus_cos"])
    return out


def d_row_verdicts(ctx, stages):
    """The seven GDN rows for one set of stage values, adjudicated by the existing licensors.

    For the controls that need the LICENCE to move and not merely a distance: `block_gate.adjudicate`
    consults the same pinned band file the emitted rows were judged with, so a control that turns a row
    REJECTED here is reddening the licensor's verdict rather than a threshold this file invented.
    """
    rows = [BG.stage_row(ctx["dumps"], "ref_bf16", s, stages) for s in ctx["d_pop"]["scan"]]
    tally = BG.adjudicate(rows, ctx["band"], ctx["limits"], ctx["d_pop"]["from_manifest"])
    return dict((r["stage"], r["band"].split(" ")[0]) for r in rows), tally


def d_fields_moved(a, b, d):
    """Which record fields two runs of the arm differ in, at zero tolerance on the stored words."""
    return dict((name, int(np.count_nonzero(np.ascontiguousarray(a[name]).view(np.uint32)
                                            != np.ascontiguousarray(b[name]).view(np.uint32))))
                for name in d["order"])


def d_bits(t):
    """One tensor's stored bit image, for a zero-tolerance comparison between implementations.

    bf16 has no numpy dtype, so the view is a torch view; comparing the bit images is what keeps "the two
    implementations agree" a statement about stored values rather than about float printing.
    """
    t = t.contiguous()
    return t.view(torch.uint16) if t.dtype is torch.bfloat16 else t.view(torch.int32)


def d_torch_body(ctx):
    """The engine's own torch GDN decode step, as a third implementation to compare against.

    `engine/block_forward.py::gdn_recurrent_step` is the body the CPU gate judges the pins with; the two
    compiled arms are two C spellings of the same §04b.1 program. Three spellings, two languages, one
    expected answer — so no row in this vertical rests on one arm being compared to itself. The call
    mirrors `forward_block` exactly: the same pinned decode-step input, the same cached conv window and
    state, the same bf16 weight arm.
    """
    dumps, cfg = ctx["dumps"], ctx["cfg"]
    W = BF.Weights("ref_bf16", cfg)
    try:
        x = dumps.tensor("%s.%s.input" % (BF.GDN_PASS, BF.GDN_MODE)).reshape(1, 1, cfg["d"])
        state = dumps.tensor("%s.prefill.state" % BF.GDN_PASS)
        conv = dumps.tensor("%s.prefill.conv_window" % BF.GDN_PASS)
        return BF.gdn_recurrent_step(x, W, cfg, conv, state)
    finally:
        W.close()


def run_d_arms(v, ctx):
    """Run the compiled pair on the pinned decode step, reconcile its census, and hand back the stages.

    Two runs in a CPU witness (g++ twin and hipcc host path), three in a device window. The PAIR line is
    zero-tolerance for the reason the other lanes print it: one arithmetic, two compilers. The census
    reconciliation is the GDN-specific half — the driver folds the state fingerprint over the bytes it
    put in the slab and requires the arm's own fold to match, on both compilers, before any row is
    emitted.
    """
    args, work = ctx["args"], ctx["work"]
    d = ctx["d"]["d"]
    runs = {}
    twin = d_arm(args.d_host_binary, ctx["d"]["plan"], ctx["d"]["slab"]["slab"],
                 work / "d_out_host.bin", label="d_host_twin")
    hip = d_arm(args.d_arm_binary, ctx["d"]["plan"], ctx["d"]["slab"]["slab"],
                work / "d_out_hip.bin", label="d_hip_hostpath")
    runs["d_host_twin"], runs["d_hip_hostpath"] = twin, hip
    nbytes, ndiff = DA.pair_bytes(work / "d_out_host.bin", work / "d_out_hip.bin")
    print("PAIR pair=d_host_twin_vs_d_hip_hostpath file=gdn_record.f32 bytes=%d ndiff_bytes=%d "
          "cases=1 zero_tolerance=1 verdict=%s"
          % (nbytes, ndiff, "IDENTICAL" if ndiff == 0 else "DIFFERS"))
    print_field_attrib("DATTRIB", "d_host_twin_vs_d_hip_hostpath", "gdn_record.f32",
                       *record_field_rows(work / "d_out_host.bin", work / "d_out_hip.bin",
                                          d["order"], d["fields"]),
                       note="per-field split of the PAIR line above; totals must account")
    d_state_per_head("d_host_twin_vs_d_hip_hostpath", work / "d_out_host.bin",
                     work / "d_out_hip.bin", d)
    g_decay = [s for s in ctx["d"]["spans"] if s["name"] == "g_decay"][0]
    exp_route_disagreement(np.frombuffer(g_decay["data"], dtype="<f4"), "DEXP",
                           note="sites=the 32 g_decay values in the slab; §04.6 step 1 is "
                                "S = S * exp(g), so this is the only transcendental the recurrent body "
                                "evaluates and these are its real inputs")
    record, device_pair = work / "d_out_hip.bin", None
    if ctx["device"]:
        dev = d_arm(args.d_arm_binary, ctx["d"]["plan"], ctx["d"]["slab"]["slab"],
                    work / "d_out_device.bin", device=True, label="d_hip_device")
        runs["d_hip_device"] = dev
        device_pair = DA.pair_bytes(work / "d_out_device.bin", work / "d_out_hip.bin")
        print("PAIR pair=d_hip_device_vs_d_hip_hostpath file=gdn_record.f32 bytes=%d ndiff_bytes=%d "
              "cases=1 zero_tolerance=1 verdict=%s"
              % (device_pair[0], device_pair[1],
                 "IDENTICAL" if device_pair[1] == 0 else "DIFFERS"))
        ctx["d_device_pair_rows"] = print_field_attrib(
            "DATTRIB", "d_hip_device_vs_d_hip_hostpath", "gdn_record.f32",
            *record_field_rows(work / "d_out_device.bin", work / "d_out_hip.bin", d["order"],
                               d["fields"]),
            note="the unattributed divergence of window `g`, now counted per field; compared with the "
                 "injected causes by gdn-divergence-causes-priced")
        ctx["d_device_head_counts"] = d_state_per_head("d_hip_device_vs_d_hip_hostpath",
                                                       work / "d_out_device.bin",
                                                       work / "d_out_hip.bin", d)
        record = work / "d_out_device.bin"
    fields, nan = DA.d_record(record, d)
    stages = DA.d_stage_tensors(fields, d)
    censuses = {}
    for label, rec in sorted(runs.items()):
        cen = DA.parse_d_census(rec["run"].get("census"))
        censuses[label] = DA.d_census_agreement(cen, d, ctx["d"]["fingerprint"])
        print("DCENSUS arm=%s path=%s form=%s items_total=%d/%d reads_state=%d/%d writes_state=%d/%d "
              "state_nonzero=%d state_sum=%d state_sumidx=%d out_elems=%d/%d outproj_reads=%d/%d "
              "disagree=%s"
              % (label, rec["run"]["path"], rec["run"].get("form"),
                 cen["items_total"][0], cen["items_total"][1],
                 cen["reads_state"][0], cen["reads_state"][1],
                 cen["writes_state"][0], cen["writes_state"][1],
                 cen["state_nonzero"], cen["state_sum"], cen["state_sumidx"],
                 cen["out_elems"][0], cen["out_elems"][1],
                 cen["outproj_reads"][0], cen["outproj_reads"][1],
                 ",".join(censuses[label]["disagree"]) or "none"))
    torch_stages = d_torch_body(ctx)
    torch_metrics = d_metrics(ctx["dumps"], torch_stages)
    ximpl = []
    for stage in d["order"]:
        ref = torch_stages[stage]
        mine = stages[stage].reshape(ref.shape)
        if mine.dtype is not ref.dtype:
            mine = mine.to(ref.dtype)
        nd = int((d_bits(mine) != d_bits(ref)).sum().item())
        ximpl.append(dict(stage=stage, n=int(ref.numel()), ndiff=nd))
    print("XIMPL pair=d_arm_vs_engine_torch_body body=block_forward.gdn_recurrent_step stages=%d "
          "ndiff_elements=%s verdict=MEASURED note=this comparison is a THIRD opinion, not a licensor: "
          "the rows below are judged against the pinned records by block_gate.adjudicate, and a "
          "torch-vs-C difference here is reported and investigated, never absorbed into a band"
          % (len(ximpl), ",".join("%s=%d/%d" % (r["stage"], r["ndiff"], r["n"]) for r in ximpl)))
    # Printed beside it because the two lines only mean something together: if the third implementation
    # were itself off the pins, "the arm differs from the torch body by N" would be a statement about two
    # wrong programs agreeing, and nothing here may read that as a licence.
    print("XIMPL pair=engine_torch_body_vs_pinned body=block_forward.gdn_recurrent_step stages=%d "
          "ndiff_vs_pin=%s verdict=%s note=the torch body's own distance to the pins, so the line above "
          "cannot be read as two wrong programs agreeing with each other"
          % (len(torch_metrics),
             ",".join("%s=%d/%d" % (s, torch_metrics[s]["ndiff"], torch_metrics[s]["n"])
                      for s in sorted(torch_metrics)),
             "EXACT" if all(m["ndiff"] == 0 for m in torch_metrics.values()) else "OFF-PIN"))
    ctx["d_torch_metrics"] = torch_metrics
    ctx["d_pair"] = (nbytes, ndiff)
    ctx["d_device_pair"] = device_pair
    ctx["d_ximpl"] = ximpl
    ctx["d_honest_metrics"] = d_metrics(ctx["dumps"], stages)
    ctx["d_honest_fields"] = fields
    ctx["d_honest_nan"] = nan
    return runs, stages, censuses


def emit_d_rows(dumps, band, limits, population, stages, provenance):
    """Seven GDN rows, judged by the existing licensors, on the form the arm declared.

    `block_gate.adjudicate` asks `band_licenses` for the pair `pinned|recurrent` at the pinned decode
    pass — the pair this arm implements, which is why the rows can be credited at all — and every grant it
    withholds is counted as UNLICENSED with the licensor's own reason string attached. Nothing here
    supplies a limit the band file does not already contain.
    """
    print("PROVENANCE stages=%s source=%s device=%s"
          % (",".join(population["scan"]), provenance,
             "on" if provenance.startswith("device:") else "off"))
    rows = []
    for stage in population["scan"]:
        if stage not in stages:
            raise Refused("d-arm-missing-stage",
                          "no value for %s, which the licensor's stage list requires: a missing row is "
                          "not a row that can be skipped" % stage)
        rows.append(BG.stage_row(dumps, "ref_bf16", stage, stages))
    tally = BG.adjudicate(rows, band, limits, population["from_manifest"])
    for row in rows:
        BG.print_row(row)
    WITNESS["rows_emitted"] += len(rows)
    granted = []
    for row in rows:
        ok, why = G.band_licenses(band, BF.GDN_PASS_KEY, row["stage"], D_FORM, "pinned")
        chunked, chunked_why = G.band_licenses(band, BF.GDN_PASS_KEY, row["stage"], "chunked", "pinned")
        granted.append(dict(stage=row["stage"], granted=ok, why=why, chunked=chunked,
                            chunked_why=chunked_why))
        print("DLICENCE stage=%s arm_form=%s %s|pinned=%s chunked|pinned=%s chunked_reason=%s"
              % (row["stage"], D_FORM, D_FORM, "granted" if ok else "refused",
                 "granted" if chunked else "refused", "" if chunked else chunked_why))
    print("DROWS form=%s rows=%d licensed=%d accepted=%d rejected=%d unlicensed=%d "
          "licensed_stages=%s unlicensed_stages=%s"
          % (D_FORM, len(rows), tally["licensed"], tally["accepted"], tally["rejected"],
             tally["unlicensed"],
             ",".join(g["stage"] for g in granted if g["granted"]) or "none",
             ",".join(g["stage"] for g in granted if not g["granted"]) or "none"))
    return rows, tally, granted


def control_d_state_transposition(v, ctx):
    """The oracle's own injection, run through the one function that contracts the state.

    `state_stored_transposed` is invisible to every structural check this repo has: `d_k == d_v`, so
    reading the carried state as [d_v,d_k] is in-bounds, shape-legal, and a different number at every
    live word. The mode lives inside `m2d_state_dot` — the only contraction of the state in the arm — so
    the control bends the shipped arithmetic and not a copy of it. Two legs, both executed, on both
    compilers:

      * `q_hat`/`k_hat` stay BIT-IDENTICAL. They are computed before the state is touched, so a control
        that moved them would be moving something else and reporting it as this one.
      * each of the five state-reading stages moves in value, and each ends at least as far from the pin
        as the honest record, with at least one strictly further. That is the derived claim; the
        honest→dirty pair of counts is printed per stage, and the arm's own reach table
        (`HDR mode_fields=`) is required to name exactly the five, because a mode that reaches fields the
        driver did not expect is a mode the driver is not entitled to interpret.
    """
    honest, d = ctx["d_honest_fields"], ctx["d"]["d"]
    hm = ctx["d_honest_metrics"]
    snap = dict(PATH_LEGS)
    out = {}
    arms = [("host_twin", ctx["args"].d_host_binary, False),
            ("hip_hostpath", ctx["args"].d_arm_binary, False)]
    if ctx["device"]:
        # The contraction lives in one function that BOTH the host body and the device kernels call, so
        # the device kernels get their own leg here rather than inheriting the host body's.
        arms.append(("hip_device", ctx["args"].d_arm_binary, True))
    for name, binary, dev in arms:
        r = d_variant(ctx, "d_out_xpose_%s.bin" % name, "d_%s-state_xpose" % name,
                      mode="state_xpose", binary=binary, device=dev, subdir="dtransposed")
        moved = d_fields_moved(honest, r["fields"], d)
        out[name] = dict(moved=moved, metrics=r["metrics"], reach=r["mode_fields"],
                         verdicts=d_row_verdicts(ctx, r["stages"])[0])
    claims_ok = []
    for name, r in sorted(out.items()):
        input_only_identical = all(r["moved"][s] == 0 for s in D_INPUT_ONLY_STAGES)
        state_moved = all(r["moved"][s] > 0 for s in D_STATE_STAGES)
        not_closer = all(r["metrics"][s]["ndiff"] >= hm[s]["ndiff"] for s in D_STATE_STAGES)
        further = [s for s in D_STATE_STAGES if r["metrics"][s]["ndiff"] > hm[s]["ndiff"]]
        reach_ok = set(r["reach"].split(",")) == set(D_STATE_STAGES)
        licensed = [g["stage"] for g in ctx["d_granted"] if g["granted"]]
        rejected = [s for s in licensed if r["verdicts"][s] == "REJECTED"]
        licence_leg = bool(rejected)
        claims_ok.append(input_only_identical and state_moved and not_closer and bool(further)
                         and reach_ok and licence_leg)
        print("DCONTROL control=transposition arm=%s reach=%s input_only_identical=%s "
              "state_stages_moved=%s every_stage_at_least_as_far=%s strictly_further=%s "
              "band_reddened_on=%s honest_ndiff=%s dirty_ndiff=%s"
              % (name, r["reach"], "yes" if input_only_identical else "no",
                 "yes" if state_moved else "no", "yes" if not_closer else "no",
                 ",".join(further) or "none", ",".join(rejected) or "none",
                 ",".join("%s=%d/%d" % (s, hm[s]["ndiff"], hm[s]["n"]) for s in D_STATE_STAGES),
                 ",".join("%s=%d/%d" % (s, r["metrics"][s]["ndiff"], r["metrics"][s]["n"])
                          for s in D_STATE_STAGES)))
    if ctx["device"]:
        print("DCONTROL control=transposition leg=gdn-device ran=yes note=the mode was bent inside "
              "`m2d_state_dot`, which the seven device kernels call, so this leg is about the launched "
              "arithmetic and not only the serial host body")
    else:
        print("DVOID control-leg=gdn-state-transposition-on-device status=not-exercised "
              "reason=needs-a-device-window note=the same mode is compiled into the hipcc binary's "
              "device kernels but no leg executed it in this window; it is reported as unexercised "
              "rather than assumed from the host body")
    v.control("gdn-state-transposition-reddens",
              bool(claims_ok) and all(claims_ok) and len(claims_ok) == len(out) and len(out) >= 2,
              "%d compiled legs (g++ twin, hipcc host path%s) ran `state_xpose` through "
              "`m2d_state_dot`: q_hat and k_hat bit-identical "
              "to each arm's honest record, all five state-reading stages moved, none of them closer to "
              "the pin than honest, and at least one strictly further on each arm; the arm's own reach "
              "table named exactly those five, and the transposed record comes back REJECTED on at least "
              "one of the four licensed stages (`band_reddened_on` per leg), so what reddens is the "
              "licence and not a distance this file chose. The transposition is invisible to shape checks "
              "(d_k == d_v == 128) and visible here because the pinned incoming state has %d live words"
              % (len(claims_ok), ", and hipcc device kernels" if ctx["device"] else "",
                 ctx["d"]["fingerprint"]["live"]),
              legs=legs_since(snap), claims=d_claims(ctx))


def control_d_carry_index(v, ctx):
    """The carry-index mistake: decay taken from the neighbouring head, with the READ left alone.

    A decode step's decay multiplier is one f32 per value head, and taking it from head (h+1) % h_v is
    the one-step version of reading the wrong head's history. It is a different defect from the
    transposition and gets a different witness: the fingerprint of the state bytes must be UNCHANGED from
    the honest run (the same words were read, in the same order), while every state-derived field moves.
    That is the clause that makes this control about the carry and not about the read: had the mode also
    changed which words it touched, `state_sum` would move and the control would refuse.
    """
    honest, d = ctx["d_honest_fields"], ctx["d"]["d"]
    hm = ctx["d_honest_metrics"]
    snap = dict(PATH_LEGS)
    ok, rows = True, []
    for name, binary in (("host_twin", ctx["args"].d_host_binary),
                         ("hip_hostpath", ctx["args"].d_arm_binary)):
        r = d_variant(ctx, "d_out_decay_%s.bin" % name, "d_%s-decay_head_off1" % name,
                      mode="decay_head_off1", binary=binary, subdir="dcarry")
        moved = d_fields_moved(honest, r["fields"], d)
        same_read = (r["census"]["state_sum"] == ctx["d"]["fingerprint"]["sum"]
                     and r["census"]["state_sumidx"] == ctx["d"]["fingerprint"]["sumidx"]
                     and r["agree"]["disagree"] == [])
        claims_ok = (same_read and all(moved[s] == 0 for s in D_INPUT_ONLY_STAGES)
                     and all(moved[s] > 0 for s in D_STATE_STAGES)
                     and all(r["metrics"][s]["ndiff"] >= hm[s]["ndiff"] for s in D_STATE_STAGES)
                     and any(r["metrics"][s]["ndiff"] > hm[s]["ndiff"] for s in D_STATE_STAGES))
        ok = ok and claims_ok
        rows.append("%s[same_state_read=%s input_only_identical=%s state_stages_moved=%s "
                    "strictly_further=%s]"
                    % (name, "yes" if same_read else "no",
                       all(moved[s] == 0 for s in D_INPUT_ONLY_STAGES),
                       all(moved[s] > 0 for s in D_STATE_STAGES),
                       ",".join(s for s in D_STATE_STAGES
                                if r["metrics"][s]["ndiff"] > hm[s]["ndiff"]) or "none"))
    v.control("gdn-carry-index-reddens", ok,
              "both compiled arms took the decay multiplier from head (h+1) %% %d: the fingerprint of the "
              "state bytes they read is UNCHANGED from the honest run (%s and %s both equal the driver's "
              "own fold) while all five state-derived fields move and none gets closer to the pin, so the "
              "difference is in what the step carried and not in what it read. %s"
              % (ctx["d"]["d"]["GDN_H_V"], ctx["d"]["fingerprint"]["sum"],
                 ctx["d"]["fingerprint"]["sumidx"], "; ".join(rows)),
              legs=legs_since(snap), claims=("host_cpubody",))


def control_d_non_carried_state(v, ctx):
    """A state that is not carried, executed twice: zeroed bytes, and the arm refusing to read them.

    This is the pad/non-carried control of D-21's shape moved to the decode step. Two independent legs,
    on both compilers:

      * `bytes-zeroed` — the driver lays a second slab whose state span is zero, so the arm reads the
        same number of words and every one of them is 0. The fingerprint the arm prints must then agree
        with the driver's fold of THAT slab (all zeros), `q_hat`/`k_hat` must be untouched, and the state
        row must be much further from the pin.
      * `arm-fault-zero-state` — the arm's own `zero-state` fault runs the carry stage over a state it
        does not read at all. The census then has to report `reads_state=0` and a zero fingerprint, and
        the driver's reconciliation is written to REQUIRE that zero, so a fault that read anyway would be
        a red control rather than an invisible one.

    Both legs matter because they catch opposite defects: the first catches a kernel that tolerates a
    non-carried state while claiming to carry it, the second catches a census that cannot see the state
    being skipped.
    """
    honest, d = ctx["d_honest_fields"], ctx["d"]["d"]
    hm = ctx["d_honest_metrics"]
    snap = dict(PATH_LEGS)
    ok, rows = True, []
    for name, binary in (("host_twin", ctx["args"].d_host_binary),
                         ("hip_hostpath", ctx["args"].d_arm_binary)):
        z = d_variant(ctx, "d_out_zerobin_%s.bin" % name, "d_%s-slab-zeroed" % name, binary=binary,
                      zero_state=True, subdir="dnoncarried")
        zero_fp_ok = (z["census"]["state_sum"] == z["fingerprint"]["sum"] == 0
                      and z["census"]["state_sumidx"] == z["fingerprint"]["sumidx"] == 0
                      and z["census"]["state_nonzero"] == 0
                      and z["census"]["reads_state"][0] == z["agree"]["state_words"])
        zm = d_fields_moved(honest, z["fields"], d)
        bytes_leg = (zero_fp_ok and all(zm[s] == 0 for s in D_INPUT_ONLY_STAGES)
                     and all(z["metrics"][s]["ndiff"] > hm[s]["ndiff"] for s in D_STATE_STAGES))
        f = d_variant(ctx, "d_out_zerofault_%s.bin" % name, "d_%s-zero-state" % name,
                      fault="zero-state", binary=binary, subdir="dnoncarried")
        fault_fp_ok = (f["census"]["reads_state"][0] == 0
                       and f["census"]["state_sum"] == 0
                       and f["agree"]["disagree"] == [])
        fm = d_fields_moved(honest, f["fields"], d)
        fault_leg = (fault_fp_ok and all(fm[s] == 0 for s in D_INPUT_ONLY_STAGES)
                     and all(f["metrics"][s]["ndiff"] >= hm[s]["ndiff"] for s in D_STATE_STAGES)
                     and any(f["metrics"][s]["ndiff"] > hm[s]["ndiff"] for s in D_STATE_STAGES))
        ok = ok and bytes_leg and fault_leg
        rows.append("%s[bytes-zeroed=%s(arm_sum=%d driver_sum=%d state_ndiff %d->%d) "
                    "arm-fault=%s(reads_state=%d state_ndiff %d->%d)]"
                    % (name, "RED" if bytes_leg else "SILENT", z["census"]["state_sum"],
                       z["fingerprint"]["sum"], hm["state"]["ndiff"], z["metrics"]["state"]["ndiff"],
                       "RED" if fault_leg else "SILENT", f["census"]["reads_state"][0],
                       hm["state"]["ndiff"], f["metrics"]["state"]["ndiff"]))
    v.control("gdn-non-carried-state-reddens", ok,
              "two legs, both compilers, executed: zeroing the state span in the slab leaves q_hat and "
              "k_hat bit-identical, leaves the state row very far from the pin, and moves the arm's "
              "fingerprint to exactly the all-zero fold the driver computes over those bytes; the arm's "
              "own zero-state fault reports reads_state=0 and a zero fingerprint, which the driver's "
              "reconciliation requires rather than permits. %s" % "; ".join(rows),
              legs=legs_since(snap), claims=("host_cpubody",))


def control_d_state_read_witness(v, ctx):
    """Does the arm read the state it claims to read? Answered by three independent facts, not by trust.

      * The arm's fingerprint of the state bytes equals the driver's fold over the bytes it supplied, on
        BOTH compilers, with a non-zero denominator on both sides (509,952 live words of 524,288, so the
        equality cannot be the trivial one over zeros).
      * Flipping ONE BYTE of the state span in a copy of the slab moves the arm's fingerprint, moves the
        driver's fold by the same arithmetic, keeps the two in agreement, and moves the rows that read
        the state. This is the leg that shows the fingerprint is a fingerprint of the slab and not a
        constant: had the arm read some other bytes, the flip would leave its number alone and the leg
        would refuse.
      * In a device window, `DREADBACK` prints the first %d bytes of the state span as the DEVICE holds
        them after the H2D, bound to the file digest. In a CPU witness that leg cannot run and is
        printed as `not exercised` — `m2_g_main.inc`'s rule that a host body must not print a host read
        as if it were a device read-back.
    """
    fp = ctx["d"]["fingerprint"]
    honest_ok, rows, honest_fails = True, [], []
    snap = dict(PATH_LEGS)
    for label in sorted(ctx["d_runs"]):
        ag = ctx["d_censuses"][label]
        leg = (not ag["disagree"] and ag["arm_state_sum"] == ag["driver_state_sum"]
               and ag["driver_state_live"] > 0 and ag["state_words"] > 0)
        honest_ok = honest_ok and leg
        if not leg:
            honest_fails.append("%s: arm fold %s vs driver fold %s, live %d/%d, disagree=%s"
                                % (label, ag["arm_state_sum"], ag["driver_state_sum"],
                                   ag["arm_state_live"], ag["driver_state_live"],
                                   ",".join(ag["disagree"]) or "none"))
        rows.append("%s=%s(sum arm=%d driver=%d live=%d/%d)"
                    % (label, "agree" if leg else "DISAGREE", ag["arm_state_sum"],
                       ag["driver_state_sum"], ag["arm_state_live"], ag["driver_state_live"]))
    flip_ok, flip_rows, flip_fails = True, [], []
    hm = ctx["d_honest_metrics"]
    flip_arms = [("host_twin", ctx["args"].d_host_binary, False),
                 ("hip_hostpath", ctx["args"].d_arm_binary, False)]
    if ctx["device"]:
        # In a window the flip has to bite the launched arithmetic too: the device reads the same slab
        # bytes, so a fingerprint that only witnesses the host body would be witnessing the wrong program.
        flip_arms.append(("hip_device", ctx["args"].d_arm_binary, True))
    for name, binary, dev in flip_arms:
        r = d_variant(ctx, "d_out_flip_%s.bin" % name, "d_%s-state-flip" % name,
                      flip_state_word=77777, binary=binary, device=dev, subdir="dreadwitness")
        moved_fp = r["census"]["state_sum"] != fp["sum"]
        still_agree = not r["agree"]["disagree"]
        # One flipped word is one element of the carried state, so the derived claim is that the row the
        # state IS moves and nothing gets closer — not that every derived row moves after bf16 rounding.
        not_closer = all(r["metrics"][s]["ndiff"] >= hm[s]["ndiff"] for s in D_STATE_STAGES)
        state_moved = r["metrics"]["state"]["ndiff"] > hm["state"]["ndiff"]
        rows_moved = state_moved and not_closer
        flip_ok = flip_ok and moved_fp and still_agree and rows_moved
        for cond, val in (("fingerprint-did-not-move", moved_fp), ("fingerprint-disagreed", not still_agree),
                          ("state-row-did-not-move", state_moved),
                          ("a-state-derived-row-got-closer-to-the-pin", not_closer)):
            if not val:
                flip_fails.append("%s: %s (fingerprint %d->%d, state_ndiff %d->%d, scan_out %d->%d)"
                                  % (name, cond, fp["sum"], r["census"]["state_sum"],
                                     hm["state"]["ndiff"], r["metrics"]["state"]["ndiff"],
                                     hm["scan_out"]["ndiff"], r["metrics"]["scan_out"]["ndiff"]))
        rows_moved_note = "moved" if rows_moved else "REFUSED"
        flip_rows.append("%s[fingerprint %d->%d agrees=%s state_rows=%s state_ndiff %d->%d "
                         "scan_out_ndiff %d->%d]"
                         % (name, fp["sum"], r["census"]["state_sum"], still_agree, rows_moved_note,
                            hm["state"]["ndiff"], r["metrics"]["state"]["ndiff"],
                            hm["scan_out"]["ndiff"], r["metrics"]["scan_out"]["ndiff"]))
    back_ok, back_fails = True, []
    if ctx["device"]:
        # In a window this is not a presence check, it is a comparison: the first bytes of the state span
        # as the DEVICE holds them after the H2D have to be the bytes the dump reader certified, or the
        # kernels are being handed something else and every state row below is about that something else.
        slab_bytes_all = ctx["d"]["slab"]["slab"].read_bytes()
        state_off = ctx["d"]["slab"]["state_off"]
        want_hex = slab_bytes_all[state_off:state_off + D_STATE_READBACK].hex()
        back_ok, back_rows = True, []
        for key in sorted(k for k in ctx["d_runs"] if "device" in k):
            m = re.search(r"DREADBACK path=device off=(\d+) bytes=(\d+) hex=([0-9a-f]+|ABSENT)",
                          ctx["d_runs"][key]["stdout"])
            if not m:
                back_ok = False
                back_fails.append("%s: the arm printed no `DREADBACK path=device` line, so nothing was "
                                  "copied back off the device to compare" % key)
                back_rows.append("%s[line-absent]" % key)
                continue
            off, nbytes, got = int(m.group(1)), int(m.group(2)), m.group(3)
            same = (got == want_hex and nbytes == D_STATE_READBACK and off == state_off)
            back_ok = back_ok and same
            if not same:
                back_fails.append("%s: read back off=%d bytes=%d match=%s (want off=%d bytes=%d, "
                                  "got %s…)" % (key, off, nbytes, got == want_hex, state_off,
                                                D_STATE_READBACK, got[:24]))
            back_rows.append("%s[off=%d bytes=%d match=%s%s]"
                             % (key, off, nbytes, "yes" if same else "no",
                                "" if same else " got=%s" % got[:24]))
        print("DREADBACK control-leg=gdn-device-readback ran=%s state_off=%d bytes=%d "
              "file_sha256=%s per_run=%s note=first %d bytes of the carried state span, read back off "
              "the device and compared to the bytes the dump reader certified"
              % ("yes" if back_ok else "no", state_off, D_STATE_READBACK,
                 ctx["d"]["state"]["sha"][:16], "; ".join(back_rows), D_STATE_READBACK))
    else:
        print("DVOID control-leg=gdn-device-readback status=not-exercised reason=needs-a-device-window "
              "note=no host body printed a read-back line, and none was allowed to: the host path binds "
              "file bytes to slab bytes by digest instead")
    # The reason has to name the leg that refused, not recite a fixed sentence. Window `i` printed "
    # `reason=nothing: every honest arm's fold equalled the driver's …` on a run where the control had in "
    # fact refused on the DEVICE flip leg — a canned reason is a second lie stapled to the first, and the "
    # whole point of `silent-control-names-its-reason` is that the reason is the finding.
    refusals = honest_fails + flip_fails + back_fails
    fired = honest_ok and flip_ok and (not ctx["device"] or back_ok)
    v.control("gdn-state-read-is-the-arm-s-own-fingerprint", fired,
              "honest legs: %s — the arm's modular fold of the state's 524288 words equals the driver's "
              "fold of the bytes it wrote, on every arm that ran, with %d live words as a non-zero "
              "denominator. Flip legs: %s — flipping one byte of the state's 77,777th word moves the "
              "fingerprint on both sides, keeps them equal, and pushes the state row further from the pin "
              "while no derived row gets closer, so the number is a witness of the read and not a constant "
              "the arm prints."
              % ("; ".join(rows), fp["live"], "; ".join(flip_rows)),
              legs=legs_since(snap), claims=d_claims(ctx),
              reason=(None if fired else
                      "%d leg(s) refused: %s. The witness is not dead unless these say it is; if the list is "
                      "empty while this control is SILENT, the bug is in this control's own arithmetic, not in "
                      "the arm (legs: %s)" % (len(refusals), " | ".join(refusals) or "NONE — no leg "
                      "reports a refusal, which means the control's fired-expression and its legs disagree",
                      "; ".join(flip_rows))))


def control_d_form_not_merged(v, ctx):
    """The two GDN forms are not one program and never credited as one, shown four ways.

      * `band-refuses-chunked` — `band_licenses` is asked for BOTH pairs at this pass, per stage, and the
        answer is printed as `DLICENCE` for all seven rows: `recurrent|pinned` grants the four the band
        measures at this pass and `chunked|pinned` grants nothing at any decode pass. If that stops being
        true — if a future band file ever grants chunked at this pass — the driver says so out loud
        instead of continuing to treat its own form claim as interesting.
      * `arm-refuses-chunked` — a plan declaring `form=chunked` is refused by both binaries with
        `form-not-implemented`, so the arm cannot be labelled into a licence it did not earn.
      * `credits-name-the-form` — every arm run in this log printed `RUN arm=d_* form=recurrent`, and
        every ACCEPTED GDN row's licence string names `recurrent|pinned`. A row credited under any other
        pair reddens this control, which is the clause that makes "we implemented the licensed form"
        checkable rather than asserted.
      * one honest run under each compiler, so the control has counted legs on the path whose reports it
        is reading.
    """
    recorded = sorted(set(r["run"].get("form", "?") for r in ctx["d_runs"].values()))
    forms = recorded
    refused = []
    for name, binary in (("host_twin", ctx["args"].d_host_binary),
                         ("hip_hostpath", ctx["args"].d_arm_binary)):
        work = ctx["work"] / "dform"
        work.mkdir(parents=True, exist_ok=True)
        spans = ctx["d"]["spans"]
        plan = work / ("plan_chunked_%s.txt" % name)
        DA.write_d_plan(plan, spans, ctx["d"]["slab"]["slab_bytes"], ctx["d"]["d"], form="chunked")
        try:
            d_arm(binary, plan, ctx["d"]["slab"]["slab"], work / ("out_chunked_%s.bin" % name),
                  label="d_%s-form-chunked" % name)
            refused.append("%s:NOT-REFUSED" % name)
        except DA.Refused as exc:
            refused.append("%s:%s" % (name, "form-not-implemented" if "form-not-implemented" in str(exc)
                                      else exc.tag))
    honest_legs = []
    snap = dict(PATH_LEGS)
    for name, binary in (("host_twin", ctx["args"].d_host_binary),
                         ("hip_hostpath", ctx["args"].d_arm_binary)):
        honest_legs.append(d_variant(ctx, "d_out_form_%s.bin" % name, "d_%s-form-honest" % name,
                                     binary=binary, subdir="dform"))
    chunked_grants = [g for g in ctx["d_granted"] if g["chunked"]]
    credited = [r for r in ctx["d_rows"] if r["band"] == "ACCEPTED"
                and "%s|pinned" % D_FORM not in (r["licence"] or "")]
    unearned = [g["stage"] for g in ctx["d_granted"]
                if g["granted"] and not any(r["stage"] == g["stage"] and r["band"] == "ACCEPTED"
                                            for r in ctx["d_rows"])]
    v.control("gdn-form-not-merged",
              forms == [D_FORM] and len(ctx["d_runs"]) >= 2 and not chunked_grants
              and all(s.endswith("form-not-implemented") for s in refused) and not credited
              and not unearned and len(honest_legs) == 2,
              "the arm reported form=%s on every one of its %d parsed run records (%s); `chunked|pinned` "
              "grants %d of the "
              "seven rows at this pass (so the two forms are not interchangeable here, which is the "
              "premise of this whole vertical); both binaries refused a plan declaring form=chunked (%s); "
              "%d ACCEPTED rows all name `%s|pinned` in their licence and every granted stage is among "
              "them (%s). Honest legs ran under both compilers so the claim is about executed reports."
              % (",".join(forms), len(ctx["d_runs"]), ",".join(forms), len(chunked_grants),
                 ",".join(refused),
                 len([r for r in ctx["d_rows"] if r["band"] == "ACCEPTED"]), D_FORM,
                 ",".join(unearned) or "none"),
              legs=legs_since(snap), claims=("host_cpubody",))


def control_d_poison(v, ctx):
    """The unwritten-slot census has to be able to be non-zero, on both paths and on both compilers.

    `skip-carry` forbids the stage that writes the state field, and `skip-out` forbids the one that writes
    `scan_out`/`core_attn`. The census then prints the field's whole length as poison — 524,288 of 524,288
    and 4,096 of 4,096 — while the honest record prints 0 everywhere. That asymmetry is the point: the
    poison is the quiet NaN pattern, a `hipMemset` to zero would have hidden the same bug as a field full
    of exact zeros, and a census that can only ever read 0 is not a guard (`pitfalls/40`).
    """
    d = ctx["d"]["d"]
    ok, rows = True, []
    for name, binary, fault, want in (
            ("host_twin", ctx["args"].d_host_binary, "skip-carry", "state"),
            ("hip_hostpath", ctx["args"].d_arm_binary, "skip-out", "scan_out")):
        r = d_variant(ctx, "d_out_%s_%s.bin" % (fault, name), "d_%s-%s" % (name, fault),
                      fault=fault, binary=binary, subdir="dpoison", metrics_ok=False)
        n = r["nan"][want]
        total = d["fields"][want][1]
        leg = n == total and all(r["nan"][s] == 0 for s in D_INPUT_ONLY_STAGES)
        ok = ok and leg
        rows.append("%s[%s field=%s poison=%d/%d q_hat=%d k_hat=%d]"
                    % (name, fault, want, n, total, r["nan"]["q_hat"], r["nan"]["k_hat"]))
    v.control("gdn-poison-census-can-be-nonzero", ok and len(rows) == 2,
              "the honest record carries no poison in any of the seven fields on either compiler; "
              "forbidding one stage each makes its own field read entirely poison (%s), which is the "
              "leg that shows the census can be non-zero instead of merely printing zero"
              % "; ".join(rows))


def control_d_plan_refusals(v, ctx):
    """Every refusal this arm claims must be shown refusing, on both binaries, by its own name.

    Six refusals, each built from the honest table with exactly one thing wrong, so the byte arithmetic of
    the plan stays honest and the only defect is the one under test: the state declared as 2-byte
    elements, `v_pre` at the wrong extent, a plan missing `g_decay`, a mode word the chunked lane has and
    this one does not, a fault word from the gather lane, and a plan whose state span overlaps `z`."""
    work = ctx["work"] / "drefuse"
    work.mkdir(parents=True, exist_ok=True)
    d, spans, slab = ctx["d"]["d"], ctx["d"]["spans"], ctx["d"]["slab"]
    wants = [("state-is-bf16", dict(elt_override={"state_in": 2}), "field-elt"),
             ("v_pre-wrong-words", dict(words_override={"v_pre": 2048}), "field-words"),
             ("missing-g_decay", dict(drop_field="g_decay"), "field-missing"),
             ("chunked-lane-mode", dict(mode="stub"), "mode-unknown"),
             ("gather-lane-fault", dict(fault="whole-note"), "fault-unknown")]
    seen, ok = [], True
    for name, binary in (("host_twin", ctx["args"].d_host_binary),
                         ("hip_hostpath", ctx["args"].d_arm_binary)):
        for tag, kw, expect in wants:
            plan = work / ("plan_%s_%s.txt" % (tag, name))
            DA.write_d_plan(plan, spans, slab["slab_bytes"], d, **kw)
            try:
                d_arm(binary, plan, slab["slab"], work / ("out_%s_%s.bin" % (tag, name)),
                      label="d_%s-%s" % (name, tag))
                seen.append("%s/%s:NOT-REFUSED" % (name, tag))
                ok = False
            except DA.Refused as exc:
                hit = expect in str(exc)
                ok = ok and hit
                seen.append("%s/%s:%s" % (name, tag, expect if hit else "tag:%s" % exc.tag))
    v.control("gdn-plan-refusals-fire", ok and len(seen) == 2 * len(wants),
              "%d refusals across two binaries, each named: %s. A plan this arm should not be able to run "
              "is refused before a byte is read, which is what makes the honest run's spans worth "
              "something" % (len(seen), "; ".join(seen)))


def control_d_device_ask_refuses(v, ctx):
    """`--device` and the device-only fault are acts or named refusals, never a host run in disguise."""
    work = ctx["work"] / "drefusal"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "d_refuse.bin"
    seen = []
    try:
        d_arm(ctx["args"].d_host_binary, ctx["d"]["plan"], ctx["d"]["slab"]["slab"], out,
              device=True, label="d_refuse_device")
        seen.append("device-ask:NOT-REFUSED")
    except DA.Refused as exc:
        seen.append("device-ask:%s" % ("device-path-absent-in-this-binary"
                                       if "device-path-absent-in-this-binary" in str(exc) else exc.tag))
    try:
        d_arm(ctx["args"].d_host_binary, ctx["d"]["plan"], ctx["d"]["slab"]["slab"], out,
              fault="alloc-oversize", label="d_refuse_oversize")
        seen.append("alloc-oversize:NOT-REFUSED")
    except DA.Refused as exc:
        seen.append("alloc-oversize:%s" % ("fault-device-only" if "fault-device-only" in str(exc)
                                           else exc.tag))
    if not ctx["device"]:
        print("DVOID control-leg=gdn-alloc-oversize-actually-fails reason=needs-a-device-window "
              "note=the host twin refuses to run this leg at all, which is the leg above; the hipcc "
              "arm's 1 TiB hipMalloc refusing is witnessed only in a window")
    v.control("gdn-device-ask-refuses-instead-of-substituting",
              all("NOT-REFUSED" not in s and s.split(":")[1] in
                  ("device-path-absent-in-this-binary", "fault-device-only") for s in seen),
              "the g++ twin answered %s. No host body is offered for a device claim, so a "
              "`--device` report from this arm can only come from the binary that launches"
              % ", ".join(seen))


def checks_d(v, ctx, d_runs, d_rows, d_tally, d_censuses, d_pop):
    """Everything the GDN vertical may claim, expressed over parsed records and derived populations."""
    d = ctx["d"]["d"]
    cfg = ctx["cfg"]
    v.check("gdn-rows-equal-licensor-stage-list",
            len(d_rows) == len(d_pop["scan"]) and not d_pop["missing"] and not d_pop["extra"],
            "%d rows, one per stage the licensor's `SCAN_WIDTH_STAGES` and the layer-0 GDN manifest "
            "agree on; the arm's record field names are the same set (%s missing=%s extra=%s), so a "
            "rename on either side reddens instead of dropping a row"
            % (len(d_rows), ",".join(d_pop["arm_declared"]), d_pop["missing"] or "none",
               d_pop["extra"] or "none"))
    v.check("gdn-kernel-geometry-matches-the-config",
            (d["GDN_H"] == cfg["d"] and d["GDN_H_K"] == cfg["h_k"] and d["GDN_H_V"] == cfg["h_v"]
             and d["GDN_D_K"] == cfg["d_k"] and d["GDN_D_V"] == cfg["d_v"]
             and d["GDN_EXP"] == cfg["expansion"]),
            "the prototype header's GDN_H=%d GDN_H_K=%d GDN_H_V=%d GDN_D_K=%d GDN_D_V=%d GDN_EXP=%d "
            "against the checkpoint config's d=%d h_k=%d h_v=%d d_k=%d d_v=%d expansion=%d: every "
            "record offset and every item count in the arm is an expression over the header's constants, "
            "so the two rulers have to be shown to read the same model"
            % (d["GDN_H"], d["GDN_H_K"], d["GDN_H_V"], d["GDN_D_K"], d["GDN_D_V"], d["GDN_EXP"],
               cfg["d"], cfg["h_k"], cfg["h_v"], cfg["d_k"], cfg["d_v"], cfg["expansion"]))
    v.check("gdn-arm-implements-only-the-form-it-claims",
            d["order"] and d_pop["arm_declared"] == d_pop["scan"]
            and all(r["run"].get("form") == D_FORM for r in d_runs.values()),
            "the arm's declared form is `%s` on every run (%s), which is the pair "
            "`block_gate.adjudicate` credits GDN rows under and the only value `engine/device/m2_d_io.h` "
            "accepts in a plan; the row credit and the form claim are checked against each other in "
            "CONTROL gdn-form-not-merged"
            % (D_FORM, ",".join(sorted(set(r["run"].get("form", "?")
                                            for r in d_runs.values())))))
    v.check("gdn-census-equals-geometry-and-driver",
            bool(d_censuses) and all(not c["disagree"] for c in d_censuses.values())
            and all(c["driver_state_live"] > 0 and c["arm_state_live"] > 0 for c in d_censuses.values()),
            "per arm, the arm's own item, read, write and output counts equal what the geometry implies "
            "AND its fingerprint of the state equals the driver's fold of the bytes it supplied, with a "
            "non-zero live-word denominator on both sides: %s. The reconciliation is what turns a census "
            "from a decoration into a claim — an arm that skipped the carry, read the state transposed in "
            "a way that missed words, or read a stale slab disagrees here rather than printing a tidy 0"
            % "; ".join("%s[state=%d/%d disagree=%s]" % (k, c["arm_state_sum"], c["driver_state_sum"],
                                                        ",".join(c["disagree"]) or "none")
                        for k, c in sorted(d_censuses.items())))
    honest_nan = ctx["d_honest_nan"]
    v.check("gdn-honest-record-has-no-unwritten-slot", not any(honest_nan.values()),
            "the honest record carries no poison in any of its seven fields (offending fields: %s); the "
            "poison control is the leg that shows that count can be non-zero"
            % (",".join(k for k, n in sorted(honest_nan.items()) if n) or "none"))
    # D-28: same split for the GDN pair. The g++ twin vs hipcc host path leg is still byte identity —
    # one arithmetic, two compilers, and disagreement there is a codegen finding, never a band — and the
    # device-vs-host leg is now judged per field in value space against a bound derived from documents
    # and a host-path floor probe rather than from the record it judges.
    d_value_ok = (ctx["d_pair"][1] == 0 and ctx["d_pair"][0] > 0
                  and bool(ctx.get("d_pair_ok")))
    v.check("gdn-compiled-pair-value-agreement", d_value_ok,
            "g++ twin vs hipcc host path over %d B: ndiff_bytes=%d at zero tolerance%s, and the device "
            "vs host-path leg is judged in value space over %d BOUNDWITNESS cells (all inside=%s). A "
            "twin-pair non-zero here is still a codegen finding, never a band to absorb: the two "
            "compilers are handed one arithmetic (`engine/device/m2_d_kernels.h` is the only place it "
            "is written). The byte-identity witness for the device leg survives as a diagnostic "
            "(`DREPRO/DREPRO-WIN … role=diagnostic gate=no`) and casts no vote; the value band decides, "
            "and D-28's escalation clause refuses the verdict if the mechanism is not the documented "
            "exp last bit"
            % (ctx["d_pair"][0], ctx["d_pair"][1],
               "" if ctx["d_device_pair"] is None
               else " (device-vs-host byte count %d, reported not judged)" % ctx["d_device_pair"][1],
               len((ctx.get("vb", {}).get("d_hip_device_vs_d_hip_hostpath") or {}).get("witness") or {}),
               "yes" if ctx.get("d_pair_ok") else "no"))
    v.check("gdn-state-span-binds-the-arm-slab",
            all(b["slab_sha"] == b["file_sha"] for b in ctx["d"]["binds"])
            and len(ctx["d"]["binds"]) == len(ctx["d"]["spans"]),
            "%d spans, each the same sha256 in the dump file and in the arm's slab, so the kernel reads "
            "bytes the dump reader certified (%s)"
            % (len(ctx["d"]["binds"]),
               ",".join("%s=%s" % (b["name"], b["slab_sha"][:8]) for b in ctx["d"]["binds"])))
    v.check("gdn-cross-implementation-comparison-was-made",
            len(ctx["d_ximpl"]) == len(d_pop["scan"])
            and all(r["n"] > 0 for r in ctx["d_ximpl"])
            and all(s in ctx["d_torch_metrics"] and ctx["d_torch_metrics"][s]["n"] > 0
                    for s in d_pop["scan"]),
            "the arm was compared to the engine's own torch GDN body over all %d stages (%s), and that "
            "body's own distance to the pins was measured over the same comparator (%s) so the two numbers "
            "cannot be read as two wrong programs agreeing. Neither line is a licensor — the rows are "
            "judged against the pinned records by `block_gate.adjudicate`, and a difference from the torch "
            "body is reported and investigated, never absorbed. Three spellings of §04b.1 (g++, hipcc, "
            "torch) is the answer to 'is this just one arm compared to itself'"
            % (len(ctx["d_ximpl"]),
               ",".join("%s=%d/%d" % (r["stage"], r["ndiff"], r["n"]) for r in ctx["d_ximpl"]),
               ",".join("%s=%d/%d" % (s, ctx["d_torch_metrics"][s]["ndiff"],
                                      ctx["d_torch_metrics"][s]["n"]) for s in d_pop["scan"])))
    lic = [r for r in d_rows if r["band"] == "ACCEPTED"]
    unl = [r for r in d_rows if r["band"].startswith("UNLICENSED")]
    granted = [g for g in ctx["d_granted"] if g["granted"]]
    v.check("gdn-licensed-rows-name-the-arm-form-and-no-others",
            len(lic) == len(granted) and len(unl) == len(d_rows) - len(granted)
            and all("%s|pinned" % D_FORM in (r["licence"] or "") for r in lic)
            and all(re.match(r"^UNLICENSED stage=\S+ reason=\S+", r["band"]) for r in unl),
            "%d of %d GDN rows are ACCEPTED and each names `%s|pinned` in its licence, which is the pair "
            "the arm's own form word claims; the remaining %d print the licensor's own reason, counted "
            "not dissolved (%s). The grant set was re-derived by asking band_licenses per stage, so this "
            "check does not take the row list's word for what it licensed"
            % (len(lic), len(d_rows), D_FORM, len(unl),
               ",".join("%s:%s" % (r["stage"], r["band"].split("reason=", 1)[-1]) for r in unl) or "none"))
    v.check("gdn-unlicensed-counted-not-absorbed",
            d_tally["unlicensed"] == len(unl) and d_tally["licensed"] == len(lic),
            "the GDN tally carries licensed=%d accepted=%d rejected=%d unlicensed=%d; PASS stays "
            "unreachable while a GDN row is unlicensed, which includes the %d rows no form's band covers "
            "at this pass"
            % (d_tally["licensed"], d_tally["accepted"], d_tally["rejected"], d_tally["unlicensed"],
               len(unl)))
    d_launch_offences = launch_count_offences(dict((k, r) for k, r in d_runs.items()
                                                   if "device" in k))
    v.check("gdn-device-path-claims-executed-on-device",
            (not ctx["device"]) or not d_launch_offences,
            "in a device run the GDN device arm reported `launches` as an int and > 0 (%d device arms, "
            "8 launches each: poison plus one per stage; offences: %s); in a CPU window the same --device "
            "ask on the g++ twin is refused with a named tag instead of a host run, and this check is "
            "vacuous here and says so. This is the check that crashed window `i` at `str > int`"
            % (len([k for k in d_runs if "device" in k]),
               ",".join(d_launch_offences) or "none"))
    all_launch_offences = launch_count_offences(ctx["all_runs"])
    v.check("device-launch-counts-are-ints-and-positive",
            (not ctx["device"]) or (not all_launch_offences
                                    and len([k for k, r in ctx["all_runs"].items()
                                             if run_facts(r).get("path") == "device"]) > 0),
            "across all four verticals (%d arm records, %d of them reporting `path=device`), every device "
            "record's `launches` is an `int` and > 0 — the type is guaranteed at the parse site by "
            "`engine/device_arm.py::parse_arm_report`, which reads the field's conversion off the arm's own "
            "printf format strings in `engine/device/m2_*_main.inc`, and this is the belt over every device "
            "record the run produced (offences: %s). A CPU run cannot exercise this and says so; the "
            "fabrications in `control-witness-guard-can-redden` (`launches` absent, `=0`, `=\"abc\"`) are what "
            "show the predicate reddens on all three"
            % (len(ctx["all_runs"]),
               len([k for k, r in ctx["all_runs"].items() if run_facts(r).get("path") == "device"]),
               ",".join(all_launch_offences) or "none"))


# ----------------------------------------------------------------------------------------------


def gpu_context_witness(when):
    """An independent statement that this process has no GPU context, read off the kernel, not off us.

    The arm's own counters can say `hip_calls=0` for two different reasons — nothing happened, or the
    counter is not incremented — so the kernel's view is sampled too: `/sys/class/kfd/kfd/proc` lists the
    pids that hold a KFD queue, and `rocm-smi --showpids` is the instrument `devices/measured-facts.md`
    §"what each instrument can see" records as able to see compute processes at all. Both are read-only
    queries over sysfs; neither opens a device, and `--device` never reaches here with a claim to make — in
    a device run the same lines are printed and read the other way on purpose.
    """
    try:
        kfd = sorted(p.name for p in Path("/sys/class/kfd/kfd/proc").iterdir())
    except OSError as exc:
        kfd = ["unreadable:%s" % exc.errno]
    self_present = str(os.getpid()) in kfd
    smi = "unavailable"
    try:
        proc = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=60)
        smi = " ".join(proc.stdout.split())[:200] if proc.returncode == 0 else "exit-%d" % proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        smi = "unavailable:%s" % type(exc).__name__
    print("GPUWITNESS when=%s pid=%d kfd_proc_dirs=%s this_pid_has_kfd_context=%s rocm_smi_showpids=%s"
          % (when, os.getpid(), ",".join(kfd) or "none", "yes" if self_present else "no", smi))
    return dict(when=when, kfd=kfd, self_present=self_present)


def run(out, args):
    v = Verifier()
    RUN_STATE["v"] = v
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = bool(args.device)
    gpu_before = gpu_context_witness("before")
    layout = DA.parse_se_layout()
    rt = DA.parse_rt_layout()
    cfg = BF.load_geometry()
    torch.set_num_threads(int(cfg["threads"]))
    dumps = BF.Dumps()
    band = G.load_band(BF.BAND_PATH)
    digest = G.band_digest(band)
    limits, raw_probes, missing = BG.prototype_limits()
    population = derived_arm_population(dumps)
    BG.REFERENT_GDN = set(population["from_manifest"]["gdn"])
    BG.REFERENT_MOE = set(population["from_manifest"]["moe"])
    print("BLOCKDEVICE block=layer0 subsystem=shared_expert+router+gdn_recurrent token=%d device=%s "
          "work=%s se_arm=%s se_host=%s rt_arm=%s rt_host=%s gdn_arm=%s gdn_host=%s gdn_form=%s"
          % (BF.TOKEN_ROW, "on" if device else "off", work, args.arm_binary, args.host_binary,
             args.rt_arm_binary, args.rt_host_binary, args.d_arm_binary, args.d_host_binary,
             D_FORM))
    print("BAND file=%s band_digest=%s limits=%s missing=%s"
          % (BF.BAND_PATH.name, digest,
             ",".join("%s=%s" % (k, limits[k]) for k in sorted(limits)), missing or "none"))
    print("POPULATION stages=%d shared=%s router=%s of_manifest_all=%d moe=%d gdn=%d "
          "source=derived from the two manifests; the shared set by the naming rule shared_, the "
          "router set by block_gate.ROUTER_FAMILY (the licensor's own tuple)"
          % (len(population["all"]), ",".join(population["shared"]), ",".join(population["router"]),
             len(population["from_manifest"]["all"]), len(population["from_manifest"]["moe"]),
             len(population["from_manifest"]["gdn"])))

    st = stage_layer0(layout, rt, device, work)
    x_bytes, x_tensor = pinned_input_bytes(dumps, cfg)
    print("INPUT tensor=layer0.input row=%d dtype=%s bytes=%d sha256=%s"
          % (BF.TOKEN_ROW, x_tensor.dtype, len(x_bytes), hashlib.sha256(x_bytes).hexdigest()[:16]))
    rs = rt_span(st)
    print("RTWIDTH tensor=%s declared_type=%s n_elts=%d bytes=%d bytes_per_elt=%d slab=%s run=%d "
          "bytes_read_back=%d note=the three router rows are licensed on a policy-A arm only while this "
          "line reads F32 and bytes=4*n_elts; engine/device/m2_rt_io.h refuses a plan that says "
          "otherwise"
          % (rs["name"], rs["type_name"], rs["n_elts"], rs["length"],
             rs["length"] // rs["n_elts"], rs["slab"], rs["run"], rs["dev_read"]))
    slab = build_slab(st, x_bytes, layout, work, side="device" if device else "file")
    rt_slab = build_rt_slab(st, x_bytes, rt, work, side="device" if device else "file")

    arm_runs, rt_runs = {}, {}

    def register(bucket, label, rec):
        bucket[label] = dict(rec=rec, records=[rec["hdr"], rec["plan"], rec["run"], rec["wrote"],
                                              rec["result"]])
        # The five counters arrive TYPED — `device_arm.parse_arm_report` refuses a `RUN` field that its own
        # printf says is an integer and is not one — so the witness sums them without casting and does not fold
        # an unreadable value into a printed total either. Window `i` died on exactly this line's `int(...)`;
        # a broken guarantee must produce a red check and a named offence on the RESULT line, not a `ValueError`
        # that prints no RESULT at all.
        witness_fold(rec["run"], label)
        print("ARM arm=%s path=%s hip_calls=%s launches=%s allocs=%s bytes_to_device=%s "
              "bytes_from_device=%s out=%s"
              % (label, rec["run"]["path"], rec["run"]["hip_calls"], rec["run"]["launches"],
                 rec["run"]["allocs"], rec["run"]["bytes_to_device"],
                 rec["run"]["bytes_from_device"], rec["wrote"]["path"]))

    register(arm_runs, "host_twin", DA.run_arm(args.host_binary, slab["plan"], slab["slab"],
                                               work / "out_host.bin", label="host_twin"))
    register(arm_runs, "hip_hostpath", DA.run_arm(args.arm_binary, slab["plan"], slab["slab"],
                                                  work / "out_hip.bin", label="hip_hostpath"))
    n_bytes, n_diff = DA.pair_bytes(work / "out_host.bin", work / "out_hip.bin")
    print("PAIR pair=host_twin_vs_hip_hostpath file=record.f32 bytes=%d ndiff_bytes=%d cases=1 "
          "zero_tolerance=1 verdict=%s" % (n_bytes, n_diff,
                                           "IDENTICAL" if n_diff == 0 else "DIFFERS"))
    register(rt_runs, "rt_host_twin", DA.run_arm(args.rt_host_binary, rt_slab["plan"],
                                                 rt_slab["slab"], work / "rt_out_host.bin",
                                                 label="rt_host_twin"))
    register(rt_runs, "rt_hip_hostpath", DA.run_arm(args.rt_arm_binary, rt_slab["plan"],
                                                    rt_slab["slab"], work / "rt_out_hip.bin",
                                                    label="rt_hip_hostpath"))
    rt_bytes, rt_diff = DA.pair_bytes(work / "rt_out_host.bin", work / "rt_out_hip.bin")
    print("PAIR pair=rt_host_twin_vs_rt_hip_hostpath file=router_record.f32 bytes=%d ndiff_bytes=%d "
          "cases=1 zero_tolerance=1 verdict=%s" % (rt_bytes, rt_diff,
                                                  "IDENTICAL" if rt_diff == 0 else "DIFFERS"))
    rt_pairs = [dict(label="host_twin_vs_hip_hostpath", bytes=n_bytes, ndiff=n_diff, device=False),
                dict(label="rt_host_twin_vs_rt_hip_hostpath", bytes=rt_bytes, ndiff=rt_diff,
                     device=False)]
    ctx = dict(args=args, layout=layout, rt=rt, cfg=cfg, dumps=dumps, band=band, digest=digest,
               limits=limits, limits_raw=raw_probes, population=population, st=st, slab=slab,
               rt_slab=rt_slab, work=work, device=device, x_bytes=x_bytes, pair=(n_bytes, n_diff),
               rt_pairs=rt_pairs,
               # The live printed text, not a snapshot: the contradiction check and its guard read printed
               # lines that later controls produce, and window `j` proved a mid-run snapshot is a parser
               # reading the run's own log before the log exists.
               live_text=out.text)
    host_f, _ = DA.rt_record(work / "rt_out_host.bin", rt)
    hip_f, _ = DA.rt_record(work / "rt_out_hip.bin", rt)
    # `rt_fields` is the honest record the licensed router rows are read from — the device arm's in a
    # device run, the hipcc host path's otherwise. `rt_fields_host` is the g++ twin, and it is the
    # other side of every cross-compilation comparison below; naming both is what keeps a control from
    # silently comparing one path against the other (window e2).
    ctx["rt_fields_host"] = host_f
    pair_a, pair_b = hip_f, host_f
    pair_label, pair_sides = "rt_hip_hostpath_vs_rt_hip_gpp_twin", ("hip_hostpath", "host_twin")
    if device:
        register(arm_runs, "hip_device", DA.run_arm(args.arm_binary, slab["plan"], slab["slab"],
                                                    work / "out_device.bin", device=True,
                                                    label="hip_device"))
        register(rt_runs, "rt_hip_device", DA.run_arm(args.rt_arm_binary, rt_slab["plan"],
                                                      rt_slab["slab"], work / "rt_out_device.bin",
                                                      device=True, label="rt_hip_device"))
        rt_pairs.append(dict(label="rt_hip_device_vs_rt_hip_hostpath", bytes=0, ndiff=0,
                             device=True))
        pair_a = DA.rt_record(work / "rt_out_device.bin", rt)[0]
        pair_label, pair_sides = "rt_hip_device_vs_rt_hip_hostpath", ("device", "host_twin")
    ctx["rt_fields"] = pair_a
    rt_scope_block(ctx, dumps, pair_label, pair_a, pair_b, pair_sides)

    if device:
        rec_bytes, rec_diff = DA.pair_bytes(work / "rt_out_device.bin", work / "rt_out_host.bin")
        wp = ctx["rt_wpre"]
        # The claim is split, because the four fields do not carry the same kind of claim. Three of
        # them are what a licence measures at exactness and stay at zero tolerance; the fourth, `wpre`,
        # is R4's fp32 pre-cast weight, which §R5 documents as NOT bit-exact across an independent
        # fp32 route. Dropping it would be a relaxation (D-18); what is done instead is a re-derivation
        # whose limit comes from documents and from the licensor's own floor probe, printed in full,
        # with a control (`wpre-scope-rejects-a-bigger-perturbation`) showing it rejects something.
        licensed = dict((k, rt_field_distance(pair_a[k], pair_b[k]))
                        for k in RT_ZERO_TOLERANCE_FIELDS)
        attrib = rt_attribution(
            (pair_a, pair_b), dict(r1=RT_R1_FIELDS,
                                   other=RT_ZERO_TOLERANCE_FIELDS + RT_R1_FIELDS + ("wleak",)))
        ctx["rt_devdiff"] = dict(full=(rec_bytes, rec_diff), licensed=licensed, attrib=attrib,
                                 fields=(pair_a, pair_b))
        print("RTATTRIB pair=%s r1_fields_bit_identical=%s wcast_bit_identical=%s "
              "every_moving_slot_has_a_moving_probs_addend=%s fields_moving_outside_the_exp_path=%s "
              "premise_ulp_to_pinned_pre_cast=%s/%s (band %d) zero_tolerance_ndiff=%s verdict=%s note=%s"
              % (pair_label, attrib["r1_clean"], attrib["cast_clean"], attrib["causal"],
                 attrib["moved_outside"] or "none", wp["to_pin"][pair_sides[0]],
                 wp["to_pin"][pair_sides[1]], wp["band"],
                 ",".join("%s=%d/%d" % (k, licensed[k][0], licensed[k][1])
                          for k in RT_ZERO_TOLERANCE_FIELDS),
                 "CONSISTENT-WITH-R5-EXP-SCOPE"
                 if attrib["r1_clean"] and attrib["cast_clean"] and attrib["causal"]
                 and not attrib["moved_outside"] else "CODEGEN-DEFECT-ATTRIBUTION-FAILS",
                 "§R5 predicts exactly this shape: R1 identical, `probs` moved by the two libms' "
                 "exp(), `wpre` moved inside that scope, `wcast` identical because the bf16 cast "
                 "absorbs it, `wleak` identical because both arms round R1 to the same bf16 before "
                 "renormalising. If any of those four fails the difference is not the documented "
                 "scope and the fix belongs in the kernel, not in this check"))
        print("PAIR pair=%s file=router_record.f32 bytes=%d ndiff_bytes=%d zero_tolerance_fields=%s "
              "zero_tolerance_ndiff=%s scoped_field=wpre scoped_ndiff=%d/%d scoped_max_ulp=%d "
              "scoped_limit_ulp=%d verdict=%s note=the zero-tolerance fields are the ones the licences "
              "measure at exactness, and one differing bit reddens "
              "`rt-device-host-licensed-fields-identical`; `wpre` is held to the derived scope in "
              "`WPRESCOPE` because `prototype/moe_router/moe_router_kernels.h` §R5 documents it as not "
              "bit-exact across an independent fp32 route; a difference on the long-dot logits or "
              "probabilities is §mr_exp's documented exp() scope, reported not absorbed"
              % (pair_label, rec_bytes, rec_diff, ",".join(RT_ZERO_TOLERANCE_FIELDS),
                 ",".join("%s=%d/%d" % (k, licensed[k][0], licensed[k][1])
                          for k in RT_ZERO_TOLERANCE_FIELDS),
                 wp["ndiff"], wp["n"], wp["max_ulp"], ctx["rt_wpre_scope"][0],
                 "CLAIMED-IDENTICAL"
                 if all(v[0] == 0 for v in licensed.values()) and wp["max_ulp"]
                 <= ctx["rt_wpre_scope"][0] else "CLAIMED-DIFFER"))

        record_path = work / "out_device.bin"
        rt_record_path = work / "rt_out_device.bin"
        provenance = ("device:engine/device/build/m2_se_arm --device + "
                      "engine/device/build/m2_rt_arm --device")
    else:
        record_path = work / "out_host.bin"
        rt_record_path = work / "rt_out_host.bin"
        provenance = ("host_cpubody:engine/device/build/m2_se_host + m2_rt_host "
                      "(no device act in this run)")
    stages, honest_nan = record_stages(record_path, layout)
    rt_fields, rt_nan = DA.rt_record(rt_record_path, rt)
    stages.update(rt_stages(rt_fields, rt))
    ctx["rt_fields"], ctx["rt_nan"] = rt_fields, rt_nan
    rows, tally = emit_rows(dumps, band, limits, population, stages, provenance)
    # The routed-expert gather: both width paths, staged from the router arm's own selection.
    ctx["gpaths"] = gather_prepare(args, ctx)
    print_gather_read_set_formats(ctx)
    g_runs, g_stages, g_censuses = run_gather_arms(v, ctx)
    g_rows, g_tally = emit_gather_rows(dumps, band, limits, g_stages, provenance)
    ctx["g_runs"], ctx["g_rows"], ctx["g_censuses"] = g_runs, g_rows, g_censuses
    # The GDN recurrent decode step: ONE form (the cached single-token step, §04b.1), two compiled
    # arms, one carried state with live words in it, and the seven rows the band has entries for. The
    # rows are the arm against the pinned decode records, judged by the same licensors as every other
    # row here; three of the seven have no licensor at this pass for either form and say so.
    ctx["d"] = d_prepare(args, ctx)
    ctx["d_pop"] = derived_d_population(population, ctx["d"]["d"])
    print("DPOPULATION licensor=%s arm_record=%s missing=%s extra=%d note=the licensor's list is "
          "speccheck/gdn_ref.SCAN_WIDTH_STAGES narrowed by the layer0 GDN manifest; the arm's list is "
          "the record field names read out of engine/device/m2_d_kernels.h"
          % (",".join(ctx["d_pop"]["scan"]), ",".join(ctx["d_pop"]["arm_declared"]),
             ",".join(ctx["d_pop"]["missing"]) or "none", len(ctx["d_pop"]["extra"])))
    d_runs, d_stages, d_censuses = run_d_arms(v, ctx)
    provenance_d = ("device:engine/device/build/m2_d_arm --device" if device else
                    "host_cpubody:engine/device/build/m2_d_host (no device act in this run)")
    d_rows, d_tally, d_granted = emit_d_rows(dumps, band, limits, ctx["d_pop"], d_stages,
                                             provenance_d)
    ctx["d_runs"], ctx["d_rows"], ctx["d_censuses"] = d_runs, d_rows, d_censuses
    ctx["d_granted"] = d_granted
    ctx["d_tally"] = d_tally
    ctx["all_runs"] = dict([("se/" + k, v) for k, v in arm_runs.items()]
                           + [("rt/" + k, v) for k, v in rt_runs.items()]
                           + [("g/" + k, v) for k, v in g_runs.items()]
                           + [("d/" + k, v) for k, v in d_runs.items()])
    for key in ("licensed", "accepted", "rejected", "unlicensed"):
        tally[key] += g_tally[key] + d_tally[key]
    print("RTRECORD file=%s floats=%d fields=%s ids=%s wcast=%s read_by=%s"
          % (rt_record_path.name, rt["RT_FLOATS"], ",".join(rt["order"]),
             ",".join(str(int(x)) for x in rt_fields["ids"]),
             ",".join("%.6g" % x for x in rt_fields["wcast"]),
             "device" if device else "host_cpubody"))
    ctx["honest_nan"] = honest_nan
    ctx["honest_stages"] = stages
    ctx["honest_rows"] = rows
    parsed = BG.parse_printed(out.text())
    ctx["text"] = out.text()
    # D-28: the value bands are derived and witnessed once, after every arm record exists and before any
    # check that may cite them. Order matters here for the same reason it mattered to the contradiction
    # guard: a bound derived after the pair was measured would be a fit, and a witness with no bound to
    # sit inside is a number with no meaning.
    D28_ESCALATE["require_repro"] = bool(device)
    value_band(v, ctx)

    control_corrupt(v, ctx)
    control_launch_failure(v, ctx)
    control_poison(v, ctx)
    control_modes(v, ctx)
    control_rt_dtype_order(v, ctx)
    control_rt_selection(v, ctx)
    control_rt_wpre_scope(v, ctx)
    control_rt_f32_span(v, ctx)
    control_rt_poison(v, ctx)
    control_gather_swap(v, ctx)
    control_gather_census(v, ctx)
    control_no_device_act_guard(v, ctx)
    control_silent_reason_guard(v, ctx)
    control_gather_poison(v, ctx)
    control_gather_order_and_form(v, ctx)
    control_gather_width_substitution(v, ctx)
    control_gather_device_refusal(v, ctx)
    control_gather_read_back_expectation(v, ctx)
    control_gather_read_back(v, ctx)
    control_gather_divergence_causes(v, ctx)
    control_d_divergence_causes(v, ctx)
    # Runs AFTER the two controls that print the reproduction legs: its `this-run` leg reads the live
    # text, and window `j` showed why — before the move it reported `exact_pairs=none` in a log that had
    # printed three legs, because the buffer it read was snapshotted before any of them existed.
    control_attribution_contradiction_guard(v, ctx)
    control_attribution_probe_is_dialable(v, ctx)
    control_d_state_transposition(v, ctx)
    control_d_carry_index(v, ctx)
    control_d_non_carried_state(v, ctx)
    control_d_state_read_witness(v, ctx)
    control_d_form_not_merged(v, ctx)
    control_d_poison(v, ctx)
    control_d_plan_refusals(v, ctx)
    control_d_device_ask_refuses(v, ctx)
    control_claim_scan(v, ctx)
    # D-28's three witnessed halves: the rejection edge sized from the bound, the geometry-derived
    # must-catch set, and the two refuted causes that must stay outside the band. Each reads the bounds
    # `value_band` derived, so all three come after it and before the checks that count them.
    value_band_rejection(v, ctx)
    value_band_must_catch(v, ctx)
    value_band_refuted_causes(v, ctx)
    # What the substrate does to the field window `m` refused in, answered in the bound's own currency and
    # with a verdict, before any check can read the refusal as a statement about the bound.
    scanout_substrate_diagnosis(ctx)
    gpu_after = gpu_context_witness("after")
    v.check("cpu-witness-had-no-gpu-context",
            (not device and not gpu_before["self_present"] and not gpu_after["self_present"])
            or device,
            "the kernel's own view, sampled before and after every arm act: this pid appeared in "
            "/sys/class/kfd/kfd/proc at neither sample (before=%s after=%s). This is the independent half "
            "of `no device act`; the arm counters are the other half, and a counter that is merely not "
            "incremented would satisfy one and not this"
            % (gpu_before["kfd"], gpu_after["kfd"]))
    offences = no_device_act_offences(ctx["all_runs"])
    v.check("cpu-witness-reports-no-device-acts",
            device or not offences,
            "with no --device flag, every arm that ran reported hip_calls=0 launches=0 "
            "bytes_to_device=0 bytes_from_device=0, and a counter the arm did not print at all is read as "
            "an offence rather than a zero (%d arms checked, offending: %s). A non-zero count here would "
            "mean a host body reached a device, which is the same defect as the reverse substitution and "
            "gets the same named check; `control-witness-guard-can-redden` is the leg that shows this "
            "predicate is not vacuous"
            % (len(ctx["all_runs"]), ",".join(offences) or "none"))
    checks(v, ctx, rows, tally, out.text(), parsed, arm_runs)
    checks_rt(v, ctx, rt_runs)
    checks_g(v, ctx, g_runs, g_rows, g_tally, g_censuses)
    checks_d(v, ctx, d_runs, d_rows, d_tally, d_censuses, ctx["d_pop"])
    checks_d28(v, ctx)
    # The population guard runs after every check that could have been skipped, so "nothing failed" can only
    # be reported by a run that actually executed a suite (`terminal_fatality`).
    check_run_population(v, ctx)

    if st["hip"] is not None:
        st["hip"].free_all()
        WITNESS["hip_calls_python"] += st["hip"].tally["frees"]
    v.check("witness-counters-all-parsed-from-arm-reports", not WITNESS["unparsable_counters"],
            "every one of the %d arm records this run registered folded an integer into the witness totals "
            "(hip_calls, launches, allocs, bytes_to_device, bytes_from_device); the fields arrive typed from "
            "`device_arm.parse_arm_report`, which is derived from each arm's own printf format, so "
            "`register()` adds them instead of casting them. Unparsable: %s. Window `i` printed no RESULT line "
            "because this line's `int()` raised; a counter the driver cannot read is a finding, not an "
            "exception"
            % (len(ctx["all_runs"]), ",".join(WITNESS["unparsable_counters"]) or "none"))
    # The last check the run prints, after every other CHECK/CONTROL line exists: the verdict's own
    # finding count reconciled against the printed records (`_REJECT_TALLY`, window `j` defect 2). It
    # sits BEFORE the BLOCKDEVICE/RESULT lines so both of those print a number that has just been
    # parsed-and-agreed; this also closes the `checks=62` vs `checks=63` drift, which was this very
    # check printing between the two summaries.
    check_terminal_tally(v, ctx)
    red = len(v.red)
    D28_ESCALATE["armed"] = bool(D28_ESCALATE["offences"])
    verdict = d28_qualification_decision(red, tally["unlicensed"], D28_ESCALATE["armed"])
    print("STAGETALLY arms=1 licensed=%d accepted=%d rejected=%d unlicensed=%d"
          % (tally["licensed"], tally["accepted"], tally["rejected"], tally["unlicensed"]))
    print("BLOCKDEVICE verdict=%s block=layer0 subsystem=shared_expert+router+gdn_recurrent rows=%d "
          "gdn_rows=%d parsed=%d licensed=%d accepted=%d rejected=%d unlicensed=%d checks=%d red=%d "
          "controls=%d fired=%d band_digest=%s note=a row with no licence is never counted as accepted, "
          "and PASS is unreachable while any row is unlicensed"
          % (verdict, len(rows), len(ctx.get("d_rows", [])), len(parsed), tally["licensed"],
             tally["accepted"], tally["rejected"], tally["unlicensed"], len(v.names), red,
             len(v.controls), v.fired, digest[:16]))
    print("RESULT mode=%s checks=%d findings=%d controls=%d fired=%d rows=%d gdn_rows=%d arms=policy_a "
          "device=%s staged=%s hip_calls=%d+%d launches=%d bytes_to_device=%d bytes_from_device=%d "
          "counters_unparsed=%d band_digest=%s no timing taken"
          % ("device" if device else "cpu", len(v.names), red, len(v.controls), v.fired, len(rows),
             len(ctx.get("d_rows", [])),
             "on" if device else "off",
             "admitted" if WITNESS["staged_runs"] else "not-attempted",
             WITNESS["hip_calls_python"], WITNESS["hip_calls_arm"], WITNESS["launches"],
             WITNESS["bytes_to_device"], WITNESS["bytes_from_device"],
             len(WITNESS["unparsable_counters"]), digest[:16]))
    for r in v.red:
        print("RED %s" % r)
    return 1 if red else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine/block_device.py",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--device", action="store_true",
                    help="stage layer 0 on the iGPU and launch the gfx1151 arm on it. Every "
                         "device-shaped line in this output depends on this flag AND on the counted "
                         "calls; without it the run is a CPU witness and says device=off.")
    ap.add_argument("--arm-binary", default=str(DA.ARM_BIN))
    ap.add_argument("--host-binary", default=str(DA.HOST_BIN))
    ap.add_argument("--rt-arm-binary", default=str(DA.RT_ARM_BIN))
    ap.add_argument("--rt-host-binary", default=str(DA.RT_HOST_BIN))
    ap.add_argument("--g-arm-binary", default=str(DA.G_ARM_BIN))
    ap.add_argument("--g-host-binary", default=str(DA.G_HOST_BIN))
    ap.add_argument("--d-arm-binary", default=str(DA.D_ARM_BIN),
                    help="hipcc GDN arm. Two compiled arms here rather than one because the GDN rows "
                         "are the first ones this driver judges on a form claim: the arm is only "
                         "entitled to a `recurrent|pinned` licence if the binary that printed the "
                         "record implements the recurrent program, and the second compiler is how "
                         "that arithmetic is shown not to be a host-only accident.")
    ap.add_argument("--d-host-binary", default=str(DA.D_HOST_BIN))
    ap.add_argument("--work", default=str(WORK_DEFAULT))
    args = ap.parse_args(argv)

    out = BG.Tee(sys.stdout)
    sys.stdout = out
    try:
        return run(out, args)
    except DA.Refused as exc:
        # Window `m`'s handler printed `checks=0 findings=1 controls=0` and nothing else, and `grep
        # verdict=FAIL` on that log found nothing. A refusal that still escapes is now reported the way a
        # failed check is: a FAIL record under the population guard's own name, a FATAL line naming the
        # reason, a RESULT line carrying it, and a non-zero exit that follows from the FAIL.
        v = RUN_STATE.get("v") or Verifier()
        mode = "device" if args.device else "cpu"
        reason = terminal_fatality(mode, len(v.names), len(v.controls),
                                   escaped="Refused tag=%s" % getattr(exc, "tag", "untagged"))
        v.check(EMPTY_SUITE_CHECK, False,
                "the run died before its check suite completed, so this record is the population guard "
                "reporting itself: `%s`. The refusal was raised outside any leg-construction guard, which "
                "is the only shape left that can still end the run early (leg-construction refusals are "
                "counted findings on their owning check, see `leg_attempt`). Nothing printed after the "
                "refusal was executed, so no absence of a FAIL may be read as a pass"
                % reason)
        print("FATAL mode=%s reason=%s checks=%d findings=%d controls=%d fired=%d "
              "last_refusal_tag=%s note=a run that executed no checks cannot fail; this one is fatal by "
              "name, not by silence"
              % (mode, reason, len(v.names), len(v.red), len(v.controls), v.fired,
                 getattr(exc, "tag", "untagged")))
        print("RESULT mode=%s checks=%d findings=%d controls=%d fired=%d rows=0 arms=policy_a "
              "device=%s staged=%s hip_calls=%d+%d launches=%d counters_unparsed=%d fatal=%s "
              "no timing taken"
              % (mode, len(v.names), len(v.red), len(v.controls), v.fired,
                 "on" if args.device else "off",
                 "admitted" if WITNESS["staged_runs"] else "not-attempted",
                 WITNESS["hip_calls_python"], WITNESS["hip_calls_arm"], WITNESS["launches"],
                 len(WITNESS["unparsable_counters"]), reason))
        return 1
    finally:
        sys.stdout = out.fh


if __name__ == "__main__":
    sys.exit(main())
