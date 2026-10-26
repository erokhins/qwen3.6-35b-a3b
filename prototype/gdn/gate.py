#!/usr/bin/env python3
"""gate.py — the GDN slice-2 gate: three arms, the derived band, the exact read checks, the live
state carry, the state-read sabotage witness, the determinism check and the honesty controls, in
one CPU-only driver.

    <venv>/bin/python prototype/gdn/gate.py [--only three-arm|band|chunkcarry|carry|exact|controls|witness|oracle|determinism]
    <venv>/bin/python prototype/gdn/gate.py --keep     (keep work/*.bin for inspection)

WHAT THE THREE ARMS ARE
  dump        the pinned bytes of `data/dumps/2026-10-26-gdn/`, read ONLY through
              `tools/dump_reference_gdn.py::read_dump_tensor`, which re-hashes every file against
              `checkpoints/dump-manifest-gdn-2026-10-26.json` before handing back a tensor. These
              are the oracle's own bytes (`produced_by: gdn_block/gdn_sequence under NO_FAULTS`),
              witnessed bit-exact against the installed `transformers` module at write time.
  oracle_cli  `speccheck/gdn_ref.py` run through its existing CLI (`--cross-form`), so the oracle
              that derived the band also re-verifies it in this window. speccheck/ is not edited
              and not imported-and-called-as-a-library-for-numbers; `gdn_ref` is imported here only
              for `load_band`, `band_check_row`, `aggregates`, `downstream_closure` and the geometry
              — the comparators and the band are the oracle's, the numbers being compared are the
              arms'.
  kernel      this lane's C, compiled twice: `host_probe` (g++) and `gdn` (hipcc, executed on the
              CPU, `hip_calls=0`). The pair is compared at ZERO tolerance to itself; a disagreement
              between two compilers on one source is a finding, never something to band.

WHAT SLICE 2 ADDS, AND WHY IT IS NOT OPTIONAL
  Slice 1 ran every case from a ZERO recurrent state, so it could never have caught a transposed
  read of that state — at a zero state the honest read and the `S^T` read return the same number.
  The pinned dump contains a real state-carry chain: prefill (32 tokens, one chunk) then decode0,
  decode1, decode2, each a cached step whose incoming state is the state the previous pass wrote
  (`checkpoints/dump-manifest-gdn-2026-10-26.json` §reading_rules.decode_rows). Every one of those
  incoming states is non-zero and pinned, and every one has a pinned `state` reference, so the live
  state needs no invented fixture. `carry` runs the three decode steps per block twice: once with
  the pinned predecessor state in the input block, once with `carry=` so each case reads the state
  the PREVIOUS CASE emitted (the arm's own per-head carry). `sabotage` (inside `carry`) then runs
  `build/host_probe_xpose` — the same source with the state read flipped back to `S^T` — and requires
  it to be INVISIBLE on the zero-state plan and RED on the live-state plans. `determinism` re-runs
  the numerics sections twice in this process's name and refuses if a single STAGE line moves.

METRICS, AND WHICH ONES DECIDE
  relL2            PRIMARY for arm-to-arm agreement (scale-free); load-bearing as an ORDERING
                   against the state-read sabotage on the same case (see CARRYORDER)
  ndiff            PRIMARY; load-bearing for the compiled pair, for every exactness check, and as
                   an ordering against the sabotage on the live-state stages
  band             PRIMARY against the dump, and ONLY through `gdn_ref.band_check_row()` over the
                   four keys the probe derived (max_abs, per_head_max, mean_rel, one_minus_cos)
  maxabs           REPORTED ONLY, never load-bearing
  cos              REPORTED ONLY, never load-bearing
  Why the demotion is not decoration: `speccheck/results-gdn.md` and `prototype/shared_expert/work/
  derive.log` (same box, 2026-10-26) both record a WRONG candidate that was NEARER on the absolute
  metric than the honest one (maxabs 0.961426 vs 0.96875), and `cos` prints 1 to nine digits while
  1263 of 1884 `lm_head` words differ in their bit image. An absolute metric that cannot see what is
  being claimed is not conservative, it is blind.

WHAT SLICE 3 ADDS — LICENCE PROVENANCE, AND THE MULTI-CHUNK QUESTION ANSWERED WITH A FIELD NAME
  Every banded row in this lane compares one of this kernel's forms against the pinned bytes, which
  is the cross-form pair `chunked|pinned`. A cross-form row may only be CREDITED if the probe
  licensed that pair at that pass (`speccheck/gdn_ref.py band_licenses`), and slice 2 credited 84
  decode rows without ever asking. `band` now asks, per row, and prints one `LICENCE` line per
  banded row with two independent fields: `licensed=` (does the licence table grant
  `chunked|pinned` at THIS pass and stage) and `form_probed=` (is the form that produced the row
  one the probe ran at that pass, by scan family and chunk width, both read out of `gdn_ref.Form`).
  The answer for the decode rows is `UNLICENSED`: `probe_arms_for("decode","pinned")` returns the
  recurrent family only, because §04b.1 gives a cached decode step the recurrent program, so no
  probe record for the pair exists at a decode pass and no licence can. Those rows are still
  measured, still printed, still ACCEPTED by the band — and NOT credited, and `GATE PASS` cannot
  print while any banded row is uncredited. Nothing was loosened, widened or deleted.
  `chunkcarry` answers the multi-chunk-prefill question twice. (1) Against pinned bytes it is NOT
  adjudicable, and the proof is a field name: every pinned pass's `dumps[].shape[1]` is at most 32
  against `geometry.chunk_width=64`, so `chunks_per_pass=1` for all eight pinned passes and there
  is no pinned multi-chunk pass to carry across. The census prints that per pass, cross-checks the
  manifest's T against the T the bytes read back, and refuses to invent a fixture. (2) The carry
  MECHANISM is then tested at the only width the material allows: the pinned 32-token prefill is
  split into two equal chunks of `T/2`, chunk 1 reads the state chunk 0 EMITTED IN THE SAME RUN
  (`carry=`), and the final state is judged against the pinned `prefill.state` through
  `band_check_row`. Those rows are labelled `LICENSED_FORM_UNPROBED` — the pair is licensed at a
  prefill pass but chunk width 16 is not a width any probe arm ran — so they are also not credited.

THE q_hat ROWS: CLOSED BY A KERNEL FORM, UNDER AN UNCHANGED BAND (D-23)
  `q_hat`'s `max_abs` used to be 2.980e-08 against a licensed 2.900e-08 — one f32 ULP over — and the
  band turned out to be exactly right. It is a triangle inequality: licensed `max_abs` = 2 x (max over
  the probe population of |arm − f64 anchor|), so the test of an implementation is not only its
  distance to `pinned` but its OWN distance to the anchor against that population maximum
  (1.4500060693811712e-08). The old ascending-f32 sum of squares measured 2.626e-08 against that
  anchor — outside every record that licensed the band — so licensing it would have needed ~5.25e-08,
  a relaxation dressed as a re-derivation, which is reserved to the user and was refused. The kernel
  was wrong and is fixed: `gdn_sumsq` now combines its 128 squares as a declared pairwise f32 tree
  (`gdn_kernels.h`), the row sits at 1.490e-08, and the band, its probe block and its digest are
  byte-unchanged. `qhat_checks` at the end of the `band` section is what makes that a checked fact:
  the numpy model of the reduction is bit-identical to the compiled arm, the arm's own distance to
  the anchor is inside the population, the licensed number is re-folded out of the probe log rather
  than trusted, a perturbation at a rung derived from the band's own `mean_rel` is still rejected,
  and every one of those is witnessed RED by `build/host_probe_sumsq` (the same source with the old
  ascending accumulator put back) or by a tampered copy of the band.

DEVICE POLICY: this file touches no device, reads no timer, and holds no device lock of any kind —
the wrapper is the sole acquirer and there is nothing here for it to wrap. Every printed counter is
a count; no line this prints carries a duration, a token rate or a bandwidth.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "speccheck"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))

import gdn_ref as G                                    # noqa: E402  (read-only: the oracle)
import dump_reference_gdn as DG                        # noqa: E402  (read_dump_tensor, manifest)

PY = sys.executable
WORK = HERE / "work"
EVID = HERE / "evidence"
HOST = HERE / "host_probe"
DEVI = HERE / "gdn"
BROKEN = HERE / "build" / "host_probe_broken"
XPOSE = HERE / "build" / "host_probe_xpose"
# D-21's second control build: the same source with `-DGDN_BROKEN_PAD`, which makes one pad row of
# every head carry data instead of §04b.2's zeros. Arm-internal by necessity — the pads are
# synthesised inside `gdn_item_enter`, so no driver-side mutation can reach one.
PAD = HERE / "build" / "host_probe_pad"
# Slice 4's control build: the same source with `-DGDN_BROKEN_SUMSQ_ASC32`, which puts the row of
# squares back into ONE ascending f32 accumulator instead of the declared pairwise tree it ships
# with. The `q_hat` band was re-derived rather than widened (D-23), so the ACCEPTED verdict on that
# row is only a check if some build can be RED under the unchanged band; this is that build, and
# `qhat_checks` is the lane that says it fired.
SUMSQ = HERE / "build" / "host_probe_sumsq"

# The authoritative band digest: `band_digest(load_band(BAND_PATH))` as measured 2026-10-26. It is
# pinned here so a silently regenerated band cannot be judged against itself. If `load_band()`
# refuses a file, the file is wrong; if the digest differs, THIS constant is stale and the human
# who regenerated the band updates it — the gate does not bend to the file.
BAND_DIGEST = "27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee"

STAGES = ["q_hat", "k_hat", "scan_out", "core_attn", "state", "normed", "y"]
# The stages a change to the STATE READ can reach, derived from the oracle's own dataflow table and
# not typed: the read enters at `scan_out` (both §04b.4's delta rule and §04b.5's `q_decayed S` term
# feed it) and at `state` itself (the carry term), and everything downstream of those is reached by
# dataflow. `q_hat`/`k_hat` are upstream of the scan and must be untouched by the sabotage; that is
# asserted, not assumed. If the oracle's table ever changes, this assertion changes with it.
STATE_AFFECTED = list(G.downstream_closure(("scan_out", "state")))
# The pinned cached-decode chain: each pass's incoming state is the previous pass's OUTPUT state,
# and the manifest names that dependency (§reading_rules.decode_rows). PRED[p] is the pass whose
# pinned `state` is pass p's incoming state.
DEC_PASSES = ["decode0", "decode1", "decode2"]
# The two plans the carry section judges, in the order it judges them: the 42 cached-decode cases
# with their pinned incoming state, and the same cases chained so each case reads the state the
# previous one emitted. These strings are the `plan=` field on every LICENCE and CARRYVERDICT line, so
# a derived denominator and a printed row cannot drift apart.
CARRY_PLAN_LABELS = ("pinned_state", "arm_carried")
PRED = {"decode0": "prefill", "decode1": "decode0", "decode2": "decode1"}
STATE_WORDS = 32 * 128 * 128
STAGE_IS_BF16 = {"q_hat": False, "k_hat": False, "scan_out": False, "core_attn": True,
                 "state": False, "normed": True, "y": True}
# the eight loaded fields, in the arm's echo order, then the field it fingerprints instead
FIELDS = ["q_pre", "k_pre", "v_pre", "beta", "g_decay", "z", "norm_weight", "out_weight"]
ECHO_FIELDS = FIELDS[:7]
FP_FIELD = "out_weight"
SECTIONS = ["three-arm", "band", "chunkcarry", "carry", "exact", "controls", "witness", "oracle",
            "determinism"]

# The arm pair every banded row in this lane measures: this kernel is a chunked scan at the
# reference width, the reference side is the pinned dump bytes. Keyed the way `gdn_ref` keys it.
LICENCE_PAIR = ("chunked", "pinned")
# D-21: the licence a DECODE row may be judged under. `chunked|pinned` has no probe record at a
# decode pass (§04b.1 gives a cached step the recurrent program), and forcing one there would
# license a program the model never dispatches. The pair is NOT typed either: the left arm is the
# recurrent-family arm the dispatch table itself names at a decode pass, the right arm is
# `gdn_ref.ARM_PINNED`, and `bridge_pair()` refuses rather than guessing if that name is ambiguous.
VERDICTS = ("LICENSED", "LICENSED_BRIDGED", "LICENSED_FORM_UNPROBED",
            "UNLICENSED", "UNLICENSED_ARM_NOT_DISPATCHED", "CANNOT_BE_ESTABLISHED")


def bridge_pair():
    """The licence pair for a decode row, read out of the oracle's dispatch table.

    `probe_arms_for("decode", "pinned")` names the arms that exist at a cached step; exactly one of
    them has `Form.scan == "recurrent"`, and that arm plus `ARM_PINNED` is the pair the licence is
    looked up under. Two such arms, or none, is a refusal: the whole point of D-21 is that the row
    is judged under the licence for the program the dispatch RUNS, so choosing between candidates
    would be this lane picking a ruler instead of reading one."""
    arms = G.probe_arms_for("decode", "pinned")
    cand = [a for a in arms if a in G.PROBE_ARMS and G.Form.named(a).scan == "recurrent"]
    if len(cand) != 1:
        raise Refuse("the §04b.1 dispatch names %d recurrent-family arms at a decode pass (%s); "
                     "D-21's licence is the one that exists for the dispatched program and it "
                     "cannot be named while the choice is ambiguous"
                     % (len(cand), ",".join(arms)))
    return (cand[0], G.ARM_PINNED)


def licensed_rows_from_file():
    """The licence population, read by a SECOND and independent reader.

    `Licence.cite` asks `gdn_ref.band_licenses`, which answers from the dict `load_band` built. The
    reconciliation checks need an expected value that does not come out of that object, or a
    mutation of it would move the row list and its own expected value together. This reads the band
    FILE again through `json.loads` and takes `licensed_rows` per stage, so the two sides share the
    bytes on disk and nothing else."""
    doc = json.loads(G.BAND_PATH.read_text())
    return {st: set(ent.get("licensed_rows", []))
            for st, ent in doc["band"].items()}


class PadWitness:
    """D-21's bridging witness: the state of the `-DGDN_BROKEN_PAD` control in THIS run.

    The claim being bridged is "a C=64 chunked kernel with 63 zero pads implements the recurrent
    single-token step". The pads are synthesised inside the arm (`gdn_item_enter` writes the zeros,
    `gdn_item_cum` adds 0.0 for a pad row), so the only mutation that can reach them is the arm's
    own `#ifdef`; a witness written in the driver could never fire, which is what D-21 forbids.
    This object records what that mutation did — per plan, at which T, how many bytes of EMITTED
    output it moved — and answers the one question the crediting rule asks: did it turn the
    comparison RED on the plans that actually produce a decode row?

    Three states, all named, because D-21's fallback is a named third state and not a silence:
      FIRED                 its self-test failed, and every plan that produces a banded decode row
                            — and every case that has pad columns — moved bytes;
      INVISIBLE             the control ran and moved nothing somewhere it must move: the witness
                            would be decorative, so it is not counted;
      CANNOT_BE_ESTABLISHED the control build is absent or its self-test cannot fail, so no
                            mutation reached the pads and no row may be credited under the bridge.
    """

    def __init__(self):
        self.plans = []          # one dict per (plan, T) the control was run over
        self.selftest = None     # None until the control build has been run
        self.reason = "the pad control has not run in this process"

    def add_plan(self, label, kind, T, t_real, chunk_width, bytes_total, ndiff_bytes,
                 produces_rows):
        self.plans.append(dict(label=label, kind=kind, T=T, t_real=t_real,
                               pad_cols=chunk_width - t_real, bytes=bytes_total,
                               ndiff=ndiff_bytes, produces_rows=produces_rows))

    @property
    def state(self):
        if self.selftest is None:
            self.reason = ("the pad control build was never run: %s is not built "
                           "(`./build.sh --broken`)" % PAD)
            return "CANNOT_BE_ESTABLISHED"
        if not self.selftest:
            self.reason = ("the pad control build passed its own self-test, so the pad checks in "
                           "the self-test cannot fail and the build is not a control")
            return "CANNOT_BE_ESTABLISHED"
        live = [p for p in self.plans if p["produces_rows"]]
        if not live:
            self.reason = ("no plan that produces a banded decode row was run under the control, "
                           "so any red it prints says nothing about the rows being credited")
            return "CANNOT_BE_ESTABLISHED"
        silent = [p for p in live if p["ndiff"] == 0]
        if silent:
            self.reason = ("the control moved 0 bytes of emitted output on %s: a pad row that "
                           "carries data did not change a decode row, so the equivalence claim is "
                           "not witnessed by anything in this run"
                           % ",".join(p["label"] for p in silent))
            return "INVISIBLE"
        inert = [p for p in self.plans if p["pad_cols"] > 0 and p["ndiff"] == 0]
        if inert:
            self.reason = ("the control moved 0 bytes on %s although those cases have %d pad "
                           "columns" % (",".join(p["label"] for p in inert), inert[0]["pad_cols"]))
            return "INVISIBLE"
        return "FIRED"

    def fires_on_decode_rows(self):
        return self.state == "FIRED"

    def describe(self):
        return ("state=%s selftest=%s plans=[%s] reason=%s"
                % (self.state, "FIRE" if self.selftest is True else
                   ("SILENT" if self.selftest is False else "NOT-RUN"),
                   " ".join("%s:T%d/pad_cols=%d/bytes=%d/ndiff=%d%s" %
                            (p["label"], p["T"], p["pad_cols"], p["bytes"], p["ndiff"],
                             "/produces_decode_rows" if p["produces_rows"] else "")
                            for p in self.plans),
                   self.reason))


class Checks:
    """Named checks with a denominator in the detail and a predicate that is an expression over
    data. `pitfalls/40` §A check whose predicate is `True`: a counted check that no mutation can
    turn red is a comment wearing a counter, so every predicate here is computed from a record set,
    a counter the arm printed, or a band verdict, and every one of the checks added in slice 3 is
    witnessed FIRING by a sabotage run in the same section that adds it.

    This is deliberately a different object from `fails`: a check FAILS the gate; a `note` printed
    by a section is a fact reported and kept out of the count (also `pitfalls/40`)."""

    def __init__(self, fails):
        self.fails = fails
        self.names = []
        self.red = 0

    def check(self, name, predicate, detail):
        ok = bool(predicate)
        self.names.append(name)
        if not ok:
            self.red += 1
            self.fails.append("check %s: %s" % (name, detail))
        print("CHECK name=%s verdict=%s %s" % (name, "PASS" if ok else "FAIL", detail))
        return ok

    def witness(self, name, fired, detail):
        """A control that is SUPPOSED to fail: the check fired, or the lane cannot claim it can."""
        self.names.append(name)
        if not fired:
            self.red += 1
            self.fails.append("control %s did not fire: %s" % (name, detail))
        print("CONTROL check=%s fired=%s %s" % (name, "RED" if fired else "SILENT", detail))
        return fired


class Licence:
    """Cites the oracle's own licence table once per banded row, and counts what it found.
    Two independent questions, because they are two different holes:
      `licensed`     — does `band_licenses` grant THE PAIR this row is judged under, at THIS pass
                       and stage? It answers from the band file's own `licensed_rows`, which
                       `load_band` already refused to let anyone hand-edit.
      `form_probed`  — is the FORM that produced this row one the probe actually ran at that pass?
                       Read out of `gdn_ref.probe_arms_for` (the §04b.1 dispatch table) and
                       `gdn_ref.Form.named(a).chunk_width`, so neither the arm list nor the chunk
                       widths are typed here.
    A `chunked|pinned` row is CREDITED only when both are true. A DECODE row is judged under the
    bridge pair instead (`recurrent|pinned`, D-21) because that is the only licence the dispatch can
    produce for the program it actually runs, and it is credited only when the licence grants it, the
    licence arm is one §04b.1 runs at that pass, AND this run's pad-equivalence witness FIRED. The
    bridge claim is "a C=64 chunked kernel with 63 zero pads implements the recurrent single-token
    step", and a row may not be credited on a claim nobody in this run tested — which is why the
    witness is an argument to `classify`, not a comment beside it. A row that is measured but not
    credited stays printed with its band verdict: nobody can read `ACCEPTED` as `credited` without
    the licence line beside it."""

    def __init__(self, band, g, verbose=True, pad=None):
        self.band = band
        self.pad = pad
        self.rows = []
        self.mark = 0
        self.chunked_arms = [a for a in G.PROBE_ARMS if G.Form.named(a).scan == "chunked"]
        self.probed_widths = sorted({(G.Form.named(a).chunk_width or g.chunk_width)
                                     for a in self.chunked_arms})
        if not verbose:
            return
        print("LICENCEH pair=%s probed_chunked_arms=%s probed_chunk_widths=%s source=gdn_ref."
              "probe_arms_for,gdn_ref.Form"
              % ("|".join(LICENCE_PAIR), ",".join(self.chunked_arms),
                 ",".join(str(w) for w in self.probed_widths)))

    def classify(self, pass_key, stage, pair, form_width=None):
        """One row's licence state, computed and not printed, so a witness can re-run THIS predicate
        over a mutated table, a shortened row set, or a withdrawn witness and compare."""
        pair = tuple(pair)
        licensed, why = G.band_licenses(self.band, pass_key, stage, *pair)
        kind = "decode" if pass_key.split("/")[-1].startswith("decode") else "prefill"
        arms = G.probe_arms_for(kind, "pinned")
        arm_dispatched = int(pair[0] in arms or pair[0] == G.ARM_PINNED)
        scan_ok = LICENCE_PAIR[0] in arms
        width_ok = form_width is None or form_width in self.probed_widths
        form_probed = int(scan_ok and width_ok)
        bridged = pair != LICENCE_PAIR
        witnessed = (not bridged) or (self.pad is not None and self.pad.fires_on_decode_rows())
        if not licensed:
            verdict = "UNLICENSED"
        elif not arm_dispatched:
            verdict = "UNLICENSED_ARM_NOT_DISPATCHED"
            why = ("the licence names %s, which §04b.1 does not run at a %s pass "
                   "(arms_at_pass=%s)" % (pair[0], kind, ",".join(arms)))
        elif bridged and not witnessed:
            verdict = "CANNOT_BE_ESTABLISHED"
            why = ("the licence for %s exists at this pass, but the pad-equivalence witness "
                   "(-DGDN_BROKEN_PAD) did not fire in this run, so the claim that lets a chunked "
                   "kernel's row stand for a recurrent one is untested here" % "|".join(pair))
        elif bridged:
            verdict = "LICENSED_BRIDGED"
        elif form_probed:
            verdict = "LICENSED"
        else:
            verdict = "LICENSED_FORM_UNPROBED"
        return dict(licensed=int(licensed), arms=arms, arm_dispatched=arm_dispatched,
                    form_probed=form_probed, verdict=verdict, why=why,
                    credited=int(verdict in ("LICENSED", "LICENSED_BRIDGED")))

    def cite(self, block, pass_, stage, plan, form_width=None, pair=None):
        pair = LICENCE_PAIR if pair is None else tuple(pair)
        pass_key = "%s/pinned/%s" % (block, pass_)
        c = self.classify(pass_key, stage, pair, form_width=form_width)
        print("LICENCE row=%s plan=%s pass=%s stage=%s pair=%s licensed=%d form_probed=%d "
              "licence_arm_at_pass=%d form_chunk_width=%s arms_at_pass=%s pad_witness=%s "
              "verdict=%s credited=%d why=%s"
              % (G.cross_form_key(pass_key, *pair), plan, pass_key, stage, "|".join(pair),
                 c["licensed"], c["form_probed"], c["arm_dispatched"],
                 "reference" if form_width is None else form_width, ",".join(c["arms"]),
                 "n/a" if pair == LICENCE_PAIR else (self.pad.state if self.pad else "ABSENT"),
                 c["verdict"], c["credited"], c["why"]))
        self.rows.append(dict(plan=plan, pass_key=pass_key, stage=stage, pair=pair,
                              verdict=c["verdict"], licensed=c["licensed"],
                              form_probed=c["form_probed"], arm_dispatched=c["arm_dispatched"],
                              credited=c["credited"]))
        return c["verdict"]

    def tally(self, label, derived):
        """Count the rows cited since the last tally and refuse an empty or short population: a
        licence table that was never consulted must not look like a run that consulted it
        (`pitfalls/40` §a driver whose parser matches nothing prints PASS)."""
        rows = self.rows[self.mark:]
        self.mark = len(self.rows)
        n = len(rows)
        by = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
        cred = sum(r["credited"] for r in rows)
        print("LICENCETALLY scope=%s rows=%d derived=%d credited=%d %s"
              % (label, n, derived, cred,
                 " ".join("%s=%d" % (k.lower(), v) for k, v in sorted(by.items()) if v)))
        return n, by


class Refuse(Exception):
    pass


# ------------------------------------------------------------------ geometry and the dump
def geom():
    g = G.load_gdn_geometry()
    want = {"hidden_size": 2048, "h_k": 16, "h_v": 32, "d_k": 128, "d_v": 128, "k_conv": 4}
    got = {"hidden_size": g.hidden_size, "h_k": g.h_k, "h_v": g.h_v, "d_k": g.d_k,
           "d_v": g.d_v, "k_conv": g.k_conv}
    if want != got:
        raise Refuse(f"the config geometry {got} is not what gdn_kernels.h hardcodes {want}; "
                     "the header constants and the committed config disagree. Fix the header.")
    if g.chunk_width != 64:
        raise Refuse(f"chunk_width {g.chunk_width} != 64: slice 1 is ONE chunk of the reference "
                     "width (A-16: changing C is a numerics change, not a knob)")
    print("GEOM %s chunk_width=%d c_arm=%d linear_layers=%d config=%s"
          % (json.dumps(got), g.chunk_width, 64, len(g.linear_layers), g.config_path))
    return g


def load_dump():
    m = DG.load_manifest()
    recs = {}
    for r in m["dumps"]:
        recs[(r["block"], r["mode"], r["tensor_key"])] = r
        recs[("name", r["name"])] = r
    blocks = [b for b in ("layer0", "layer1")
              if (b, "prefill", "q_pre") in recs]
    if not blocks:
        raise Refuse("no block in the manifest has a pinned prefill pass; nothing to compare")
    T = recs[(blocks[0], "prefill", "q_pre")]["shape"][1]
    if T > 64:
        raise Refuse(f"pinned prefill T={T} exceeds one chunk of 64; this slice is one chunk "
                     "per case")
    for b in blocks:
        for p in DEC_PASSES:
            for key in ("q_pre", "state"):
                if (b, p, key) not in recs:
                    raise Refuse("%s/%s/%s is not in the dump manifest: the state-carry chain is "
                                 "incomplete and slice 2 has no material" % (b, p, key))
        if recs[(b, "decode0", "q_pre")]["shape"][1] != 1:
            raise Refuse("%s/decode0/q_pre is not a single-token step; the arm's decode geometry "
                         "assumes T=1" % b)
    print("DUMP %s T=%d records=%d blocks=%s n_chunks=1 pad=%d dec_passes=%s dec_T=1"
          % (m["lane"], T, len(m["dumps"]), ",".join(blocks), 64 - T, ",".join(DEC_PASSES)))
    return m, recs, blocks, T


def dump_tensor(recs, block, mode, key):
    return DG.read_dump_tensor(recs[(block, mode, key)])


def check_state_affected():
    """The derived must-catch set must be 'everything the state read can reach', which in this
    pipeline is every stage except the two the scan entry produces. Typed sets go stale; this one is
    recomputed from `gdn_ref.STAGE_CONSUMERS` at every import and refused here."""
    if set(STATE_AFFECTED) != set(STAGES) - {"q_hat", "k_hat"}:
        raise Refuse("the state-affected stage set derived from gdn_ref.STAGE_CONSUMERS is %s, "
                     "which is not 'every stage except q_hat and k_hat'; the sabotage's "
                     "must-catch set would be a guess and not a derivation"
                     % ",".join(STATE_AFFECTED))
    print("STAGESET state_read_reaches=%s untouched=%s derived_from=gdn_ref.downstream_closure"
          % (",".join(STATE_AFFECTED), ",".join(s for s in STAGES if s not in STATE_AFFECTED)))


def weight_tensor(recs, block, role):
    """A stored weight, looked up by its manifest NAME (`layer0.weights.norm`). The weight records
    carry `tensor_key` = the checkpoint's own suffix (`linear_attn.norm.weight`), so keying them the
    way the stage records are keyed is a KeyError that a lazier driver would report as "the dump has
    no norm weight"."""
    name = "%s.weights.%s" % (block, role)
    if ("name", name) not in recs:
        raise Refuse("%s is not a record in the dump manifest" % name)
    return DG.read_dump_tensor(recs[("name", name)])


def words_of(t: torch.Tensor) -> np.ndarray:
    """One u32 word per element: bf16 by BIT IMAGE in the low 16 bits, f32 by bit image whole.
    `view`, never `cast` — `.to()` of a bf16 tensor to an integer type converts the value and
    produces a plausible, wrong word (`pitfalls/30` §Tensor.to casts, .view bitcasts)."""
    if t.dtype is torch.bfloat16:
        return t.contiguous().view(torch.uint16).to(torch.int64).numpy().astype(np.uint32)
    if t.dtype is torch.float32:
        return t.contiguous().view(torch.int32).numpy().astype(np.uint32)
    raise Refuse(f"field dtype {t.dtype} is neither bf16 nor f32; the arm's word rule is undefined")


def field_words(recs, block, T, mode="prefill"):
    """The six scan-input fields for one block, in the arm's field order."""
    keys = ["q_pre", "k_pre", "v_pre", "beta", "g_decay", "z"]
    out = []
    for k in keys:
        t = dump_tensor(recs, block, mode, k)
        if t.numel() == 0:
            raise Refuse(f"{block}.{mode}.{k}: empty tensor cannot be a field")
        if t.numel() != field_n(k, T):
            raise Refuse(f"{block}.{mode}.{k}: {t.numel()} elements, the arm's layout says "
                         f"{field_n(k, T)} for T={T}; the field would be read at the wrong offset")
        out.append(words_of(t.reshape(-1)))
    return out


def weight_words(recs, block):
    nw, ow = weight_tensor(recs, block, "norm"), weight_tensor(recs, block, "out")
    if nw.numel() != field_n("norm_weight", 1) or ow.numel() != field_n("out_weight", 1):
        raise Refuse("%s weight shapes %s/%s are not the [128] and [2048,4096] the arm lays out"
                     % (block, tuple(nw.shape), tuple(ow.shape)))
    return [words_of(nw.reshape(-1)), words_of(ow.reshape(-1))]


# ------------------------------------------------------------------ the work files
def field_n(field: str, T: int) -> int:
    return {"q_pre": T * 16 * 128, "k_pre": T * 16 * 128, "v_pre": T * 32 * 128,
            "beta": T * 32, "g_decay": T * 32, "z": T * 32 * 128, "norm_weight": 128,
            "out_weight": 2048 * 4096}[field]


def write_plan(path: Path, cases) -> int:
    """cases: list of dicts(block, name, T, mode, wslot, carry). Returns the declared case count.
    `carry` is the index of the case whose EMITTED state this case reads as its incoming state, or
    -1 to take the state block out of inputs.bin. One plan declares one T, because one stride
    addresses every case; the arm refuses a plan that breaks that rule."""
    lines = ["#GDNPLAN v2 n_cases=%d T=%d" % (len(cases), cases[0]["T"])]
    for i, c in enumerate(cases):
        lines.append("CASE idx=%d block=%s name=%s T=%d mode=%s wslot=%d carry=%d"
                     % (i, c["block"], c["name"], c["T"], c["mode"], c["wslot"],
                        c.get("carry", -1)))
    path.write_text("\n".join(lines) + "\n")
    return len(cases)


def write_inputs(path: Path, per_case_fields, T: int, states=None):
    """inputs.bin = per case: the six fields (one u32 word per element), then the h_v*d_k*d_v
    incoming state.

    `states[i)` is the incoming state for case i, as f32 bit-image words. Slice 1 passed nothing and
    got zeros, which is what made 'the S term contributed exactly zero' a checked fact of the case;
    slice 2 passes the PINNED state the previous pass produced, so the live state is a dumped tensor
    and not a fixture this file invented. A case whose slot is not filled is still zero, so the
    zero-state plan needs no change here."""
    buf = []
    for i, fields in enumerate(per_case_fields):
        for w in fields:
            buf.append(w.astype(np.uint32))
        if states is None or states[i] is None:
            buf.append(np.zeros(STATE_WORDS, dtype=np.uint32))
        else:
            s = states[i].astype(np.uint32)
            if s.size != STATE_WORDS:
                raise Refuse("case %d's incoming state is %d words, the arm lays out %d"
                             % (i, s.size, STATE_WORDS))
            buf.append(s)
    arr = np.concatenate(buf)
    path.write_bytes(arr.tobytes())
    return int(arr.size), T


def write_weights(path: Path, slots):
    """weights.bin = one slot per weight block: norm (128 words) then out_proj (8,388,608 words)."""
    buf = []
    for norm, outw in slots:
        buf.append(norm.astype(np.uint32))
        buf.append(outw.astype(np.uint32))
    arr = np.concatenate(buf)
    path.write_bytes(arr.tobytes())
    return int(arr.size)


def stage_words(T: int) -> dict:
    return {"q_hat": T * 32 * 128, "k_hat": T * 32 * 128, "scan_out": T * 32 * 128,
            "core_attn": T * 32 * 128, "state": 32 * 128 * 128, "normed": T * 32 * 128,
            "y": T * 2048}


def stage_total(T: int) -> int:
    return sum(stage_words(T).values())


def stage_offset(T: int, stage: str) -> int:
    return sum(stage_words(T)[s] for s in STAGES[:STAGES.index(stage)])


def echo_total(T: int) -> int:
    return sum(field_n(f, T) for f in ECHO_FIELDS)


def echo_offset(T: int, field: str) -> int:
    return sum(field_n(f, T) for f in ECHO_FIELDS[:ECHO_FIELDS.index(field)])


def stage_token_axis(stage: str, ref, T: int):
    """Which axis of one pinned stage is the token axis — DERIVED, and ambiguous is a refusal.

    `checkpoints/dump-manifest-gdn-2026-10-26.json` §reading_rules.orientations gives
    q_hat/k_hat/scan_out as `[B,h_v,T,d]` and core_attn/normed as `[B,T,h_v,d_v]`, so one slice axis
    is wrong for half the stages, and `q_hat` is `[1,32,32,128]` with `h_v == T == 32`: a guessed
    axis compares heads against tokens and still prints numbers. The oracle's own head-axis table
    (`gdn_ref.HEAD_AXIS`) removes the head axis from the candidate set; the pass's T removes the
    feature axes. What is left must be exactly one axis — `state` is `[B,h_v,d_k,d_v]` and so has no
    token axis at all, which the caller uses to mean 'this stage has no per-chunk reference'."""
    head = G.HEAD_AXIS.get(stage)
    cands = [d for d in range(1, ref.ndim) if d != head and int(ref.shape[d]) == T]
    if len(cands) != 1:
        if not cands and stage == "state":
            return None
        raise Refuse("the token axis of stage %s in shape %s is not derivable (head axis %s, "
                     "candidates %s for T=%d): slicing it would be a guess"
                     % (stage, tuple(ref.shape), head, cands, T))
    return cands[0]


def stage_token_slice(stage: str, ref, T: int, lo: int, hi: int):
    """Rows [lo,hi) of one stage, in the reference's OWN orientation, so `aggregates`' head fold
    still folds what it was derived to fold."""
    ax = stage_token_axis(stage, ref, T)
    idx = [slice(None)] * ref.ndim
    idx[ax] = slice(lo, hi)
    return ref[tuple(idx)]


# ------------------------------------------------------------------ running an arm
def run_arm(binary: Path, arm_name: str, plan: Path, inputs: Path, weights: Path, tag: str):
    out = WORK / ("%s.%s.out.bin" % (tag, arm_name))
    ech = WORK / ("%s.%s.echo.bin" % (tag, arm_name))
    rep = WORK / ("%s.%s.tsv" % (tag, arm_name))
    cmd = [str(binary), "--plan", str(plan), "--inputs", str(inputs), "--weights", str(weights),
           "--out", str(out), "--echo", str(ech), "--report", str(rep)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = (p.stdout + p.stderr).splitlines()
    return dict(rc=p.returncode, lines=lines, out=out, echo=ech, rep=rep, cmd=" ".join(cmd))


def parse_arm(res, n_declared: int, T: int):
    """CASE, FP and RUN lines, with the parsed count required to equal the declared count. A run that
    printed nothing can only ever have had nothing to say, so an empty parse refuses (`pitfalls/40`).
    Every CASE line must also carry `state_nonzero` and `carry`: the first is the counter that makes
    a live state a measured fact, the second is what makes a carry plan checkable case by case, and a
    driver that silently read a missing field would read a zero instead."""
    cases, fps, run = [], {}, None
    for ln in res["lines"]:
        if ln.startswith("CASE "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            for must in ("state_nonzero", "carry", "pad_nonzero", "out_hash"):
                if must not in d:
                    raise Refuse("arm CASE line is missing %s: %s" % (must, ln))
            cases.append(d)
        elif ln.startswith("FP "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            fps[(int(d["idx"]), d["field"])] = d
        elif ln.startswith("RUN "):
            run = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
    if not cases:
        raise Refuse("arm printed no CASE lines: %s\n  %s" % (res["cmd"], "\n  ".join(res["lines"][:6])))
    if len(cases) != n_declared:
        raise Refuse("arm parsed %d CASE lines, plan declared %d — parsed != printed is a refusal, "
                     "not a partial run" % (len(cases), n_declared))
    if run is None:
        raise Refuse("arm printed no RUN line")
    want_fp = n_declared * len(FIELDS)
    if len(fps) != want_fp:
        raise Refuse("arm printed %d FP lines, derived denominator is %d (%d cases x %d fields)"
                     % (len(fps), want_fp, n_declared, len(FIELDS)))
    res["case_words"] = int(run["case_words"])
    res["echo_words"] = int(run["echo_words"])
    res["n_cases"] = len(cases)
    if res["case_words"] < stage_total(T):
        raise Refuse("the arm's per-case stride %d cannot hold the %d stage words this lane emits"
                     % (res["case_words"], stage_total(T)))
    if res["echo_words"] != echo_total(T):
        raise Refuse("the arm echoes %d words per case, the derived echo is %d"
                     % (res["echo_words"], echo_total(T)))
    return cases, fps, run


def stage_tensor(res, idx: int, T: int, stage: str) -> torch.Tensor:
    """One stage of one case out of the arm's out.bin.

    The per-case stride is the STRIDE THE ARM PRINTED (`case_words=` on the RUN line), not a number
    this file recomputes: the arm's per-case block also carries its scratch, and two independent
    spellings of that size is how a driver ends up reading a plausible-looking neighbour region.
    The stage offset inside the block is derived here and cross-checked against the arm's total."""
    n, off = stage_words(T)[stage], stage_offset(T, stage)
    base = idx * res["case_words"]
    if base + stage_total(T) > res["case_words"] * res["n_cases"]:
        raise Refuse("stage window runs past out.bin")
    words = np.fromfile(res["out"], dtype=np.uint32, count=n, offset=(base + off) * 4)
    if words.size != n:
        raise Refuse("out.bin is short at stage %s case %d" % (stage, idx))
    if STAGE_IS_BF16[stage]:
        return torch.from_numpy((words & 0xFFFF).astype(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(words.copy()).view(torch.float32)


def echo_tensor(res, idx: int, T: int, field: str) -> np.ndarray:
    n = field_n(field, T)
    base = idx * res["echo_words"]
    w = np.fromfile(res["echo"], dtype=np.uint32, count=n,
                    offset=(base + echo_offset(T, field)) * 4)
    if w.size != n:
        raise Refuse("echo.bin is short at field %s case %d" % (field, idx))
    return w


def fp_of_words(w: np.ndarray):
    """64-bit positional fingerprint of the words the arm would have loaded, computed with numpy's
    wrapping unsigned arithmetic — the same three quantities the arm sums inside its load loop."""
    u = w.astype(np.uint64)
    n = u.size
    s = int(u.sum(dtype=np.uint64))
    # positional sum in chunks so the index vector never materialises 67 MB at once
    si = np.uint64(0)
    step = 1 << 20
    for a in range(0, n, step):
        seg = u[a:a + step]
        idx = (np.arange(a, a + seg.size, dtype=np.uint64) + np.uint64(1))
        si = si + (seg * idx).sum(dtype=np.uint64)
    return s, int(si), n


# ------------------------------------------------------------------ metrics
def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    x = a.reshape(-1).to(torch.float64)
    y = b.reshape(-1).to(torch.float64)
    d = x - y
    nb = float(torch.linalg.vector_norm(y))
    na = float(torch.linalg.vector_norm(x))
    den = na * nb
    bits_a = (a.reshape(-1).view(torch.uint16) if a.dtype is torch.bfloat16
              else a.reshape(-1).view(torch.int32).to(torch.int32))
    bits_b = (b.reshape(-1).view(torch.uint16) if b.dtype is torch.bfloat16
              else b.reshape(-1).view(torch.int32).to(torch.int32))
    ndiff = int((bits_a != bits_b).sum())
    return dict(n=int(a.numel()), ndiff=ndiff,
                relL2=(float(torch.linalg.vector_norm(d)) / nb if nb else 0.0),
                maxabs=float(d.abs().max()),
                cos=(float(torch.dot(x, y)) / den if den else 1.0))


def fmt(m) -> str:
    return ("n=%d ndiff=%d relL2=%.6g maxabs=%.6g[reported_only] cos=%.9g[reported_only]"
            % (m["n"], m["ndiff"], m["relL2"], m["maxabs"], m["cos"]))


# Why this exists: the FAIL branch used to be the only verdict line that printed a finding count,
# so on the QUALIFIED/PASS paths a reader could not tell "zero findings, measured" apart from
# "findings never counted on this path", and a REJECTED row whose fail_msg is None never becomes a
# finding at all. Every terminal verdict line now carries the tally so the machinery is seen alive.
_REJECT_TALLY = {}


def band_row(band, tag, block, pass_, stage, mine, ref, stage_lines, fails, fail_msg=None,
             name_key=None):
    """One arm's stage against one pinned stage: the metrics, the oracle's own verdict, one line.

    Every banded comparison in the lane goes through here so that `band_check_row` is called the same
    way for the prefill chunk, for the live-state decode steps, for the re-chunked prefill and for
    the sabotage build, and so the determinism check compares a grammar and not three hand-written
    formats. `maxabs`/`cos` are labelled by `fmt`; the four band keys are the oracle's. Nothing here
    judges a number: the verdict is `band_check_row`'s. An empty row set cannot pass, because the
    caller declares the derived denominator and asserts it."""
    if mine.numel() != ref.numel():
        raise Refuse("%s/%s/%s: the arm emitted %d words, the dump holds %d elements"
                     % (block, pass_, stage, mine.numel(), ref.numel()))
    mine = mine.reshape(ref.shape)
    m = metrics(mine, ref)
    ag = G.aggregates(mine, ref,
                      name=name_key or "%s/pinned/%s|%s" % (block, pass_, stage), stage=stage)
    ok, why = G.band_check_row(band, stage, ag)
    line = ("STAGE arm=%s block=%s pass=%s stage=%s %s band=%s meanrel=%.3e 1-cos=%.3e "
            "perhead=%.3e(h%d/%d) verdict=%s why=%s"
            % (tag, block, pass_, stage, fmt(m), "ACCEPTED" if ok else "REJECTED", ag["mean_rel"],
               ag["one_minus_cos"], ag["per_head_max"], ag["worst_head"], ag["n_heads"],
               "PASS" if ok else "FAIL", why))
    print(line)
    stage_lines.append(line)
    if not ok:
        _REJECT_TALLY["rejected"] = _REJECT_TALLY.get("rejected", 0) + 1
        # A REJECTED row with fail_msg=None is an EXPECTED rejection (a sabotage or control row);
        # one with a fail_msg is a finding. The two must never be conflated, and the counts below
        # are what prove the classifier ran in this very run rather than in some earlier one.
        key = "untaxed" if fail_msg is None else "taxed"
        _REJECT_TALLY[key] = _REJECT_TALLY.get(key, 0) + 1
    if not ok and fail_msg is not None:
        fails.append(fail_msg(block, pass_, stage, why) if callable(fail_msg) else fail_msg)
    return ok, m


# ------------------------------------------------------- the q_hat reduction rulers (§04.4, D-23)
#
# The band on `q_hat` is a statement about the PROBE, not about this kernel: each licensed key is
# `BAND_FACTORS[key]` x the worst record in the probe population, and each record is one arm's
# distance to the f64 anchor. So the ruler that decides whether a new spelling of §04.4's l2norm is
# inside the family the band was sized from is the arm's OWN distance to that anchor, and it has to
# be computed here in plain numbers: `torch` only READS pinned tensors in this file, so nothing in an
# arithmetic path shares a compiler with the arm being judged.
#
# Three spellings are transcribed once, here, and every `q_hat` number this gate prints uses them:
#   `qhat_sumsq`   a row of f32 squares combined in one of the three shapes `gdn_sumsq` has — the
#                  shipped pairwise tree, the wide-f64 alternative, the pre-fix ascending f32.
#   `qhat_l2`      §04.4's l2norm exactly as `gdn_item_enter` spells it: f32 products, eps added IN
#                  f32, the reciprocal root taken in double and rounded once, and the scale as TWO
#                  multiplies `(q * rq) * qs` with `qs = (float)(1/sqrt(128))`. Folding the two
#                  constants into one product is a different f32 and the model stops being the arm.
#   `qhat_anchor`  the f64 exact arm: same program, f64 accumulation, a DIVIDE by sqrt(d_k), result
#                  left in f64. Rounding it to f32 here would move every distance by up to half an
#                  ULP and break the agreement with the probe log it is reconciled against.

QH_PROBE_LOG = REPO / "speccheck" / "evidence" / "gdn-floor-probe-2026-10-26.txt"
# The floor probe's two table shapes: one record line per (pass, arm, stage) and one band summary
# line per stage inside the `### band` section. Both parsers must account for every line they claim
# (`pitfalls/40` §a check that can produce no output cannot fail).
QH_STAGES = r"q_hat|k_hat|scan_out|core_attn|state|normed|y"
QH_REC_RE = re.compile(r"^\s+(\S+)\s+(\S+)\s+(" + QH_STAGES + r")\s+(float32|float64)\s+"
                       r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)/(\d+)"
                       r"(?:\s+(\S*))?\s*$")
QH_REC_SEEN = re.compile(r"\s(" + QH_STAGES + r")\s+(?:float32|float64)\s")
QH_SUM_RE = re.compile(r"^\s+(" + QH_STAGES + r")\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)"
                       r"\s+([\d.eE+-]+)")


def qhat_sumsq(p: np.ndarray, form: str) -> np.ndarray:
    """One row of f32 squares, combined the way `gdn_sumsq` combines it. `form` names the spelling
    in the header: "tree" (shipped), "wide64" (`GDN_SUMSQ_WIDE_F64`, llama.cpp's shape), "asc32"
    (`GDN_BROKEN_SUMSQ_ASC32`, the pre-fix defect and the control build). Last axis is reduced."""
    n = int(p.shape[-1])
    if n & (n - 1):
        raise Refuse("q_hat sum-of-squares over %d terms: the declared tree halves to 1, so the "
                     "term count has to be a power of two (as gdn_kernels.h's #error says)" % n)
    if form == "tree":
        t = p.copy()
        w = n
        while w > 1:
            h = w >> 1
            t[..., :h] = (t[..., :h] + t[..., h:w]).astype(np.float32)
            w = h
        return t[..., 0]
    if form == "wide64":
        acc = np.zeros(p.shape[:-1], dtype=np.float64)
        for i in range(n):
            acc = acc + p[..., i].astype(np.float64)
        return acc.astype(np.float32)
    if form == "asc32":
        acc = np.zeros(p.shape[:-1], dtype=np.float32)
        for i in range(n):
            acc = (acc + p[..., i]).astype(np.float32)
        return acc
    raise Refuse("unknown gdn_sumsq spelling %r" % (form,))


def qhat_l2(x: np.ndarray, form: str) -> np.ndarray:
    """§04.4's l2norm as `gdn_item_enter` spells it, with the sum-of-squares spelling selectable."""
    sq = qhat_sumsq((x * x).astype(np.float32), form)
    root = 1.0 / np.sqrt((sq + np.float32(G.L2NORM_EPS)).astype(np.float32).astype(np.float64))
    rq = np.asarray(root, dtype=np.float32)
    qs = np.float32(1.0 / np.sqrt(128.0))
    return ((x * rq[..., None]).astype(np.float32) * qs).astype(np.float32)


def qhat_anchor(x: np.ndarray) -> np.ndarray:
    """The f64 exact arm (`gdn_ref.Form("exact")`'s shape): f64 accumulation, DIVIDE by sqrt(d_k),
    and the result stays f64 — that is what the probe log's records are distances to."""
    xf = x.astype(np.float64)
    r = 1.0 / np.sqrt((xf * xf).sum(axis=-1) + np.float64(G.L2NORM_EPS))
    return xf * (r[..., None] / np.sqrt(float(x.shape[-1])))


def qhat_f32_words(a) -> np.ndarray:
    """The u32 bit image of f32 values, for zero-tolerance word comparisons."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)


def qhat_f64_of_words(w) -> np.ndarray:
    """f64 values from u32 bit images — the widening used for every distance, never `.view`."""
    return np.ascontiguousarray(np.asarray(w, dtype=np.uint32)).view(np.float32).astype(np.float64)


def qhat_probe_log() -> dict:
    """Parse the pinned floor-probe log: every per-record line, and the band summary block.

    Refuses rather than reports a short parse: printed count must equal parsed count, or a drifted
    log format would silently shrink the population this whole argument is sized from."""
    if not QH_PROBE_LOG.exists():
        raise Refuse("%s is missing: the q_hat population check reads the probe log the band was "
                     "derived from, and cannot be run without it" % QH_PROBE_LOG)
    recs, summary, printed, unparsed = [], {}, 0, 0
    in_band = False
    for ln in QH_PROBE_LOG.read_text().splitlines():
        if ln.startswith("###"):
            in_band = ln.startswith("### band")
            continue
        if QH_REC_SEEN.search(ln):
            printed += 1
            m = QH_REC_RE.match(ln)
            if not m:
                unparsed += 1
                print("QHATUNPARSED %s" % ln.strip())
                continue
            recs.append(dict(pass_=m.group(1), arm=m.group(2), stage=m.group(3), dtype=m.group(4),
                             maxabs=float(m.group(5)), oneminus=float(m.group(6)),
                             meanrel=float(m.group(7)), perhead=float(m.group(8)),
                             ndiff=int(m.group(9)), n=int(m.group(10)), fold=m.group(11) or ""))
        if in_band:
            m = QH_SUM_RE.match(ln)
            if m:
                summary[m.group(1)] = dict(arms=int(m.group(2)), passes=int(m.group(3)),
                                           records=int(m.group(4)), probe_maxabs=float(m.group(5)),
                                           band_maxabs=float(m.group(6)))
    if printed != len(recs) or printed == 0:
        raise Refuse("probe log parse: printed=%d parsed=%d unparsed=%d — parsed != printed is a "
                     "refusal, not a partial read (pitfalls/40)" % (printed, len(recs), unparsed))
    if len(summary) != len(STAGES):
        raise Refuse("probe log band summary parsed %d stage rows, this lane bands %d stages"
                     % (len(summary), len(STAGES)))
    print("PROBELOG file=%s printed_lines=%d parsed=%d equal=1 band_summary_rows=%d "
          "derived_rows=%d note=the log is read-only here; it is the population the band folds and "
          "the ruler these checks reconcile against, so both parsers are required to account for "
          "every line they claim"
          % (QH_PROBE_LOG.name, printed, len(recs), len(summary), len(STAGES)))
    return dict(recs=recs, summary=summary, printed=printed)


def qhat_checks(band, digest, recs, blocks, T, plan, ins, wts, armres, fails, counts, chk):
    """The `q_hat` reduction, made permanent (D-23).

    Called at the end of the `band` section, where the honest arm's own `q_hat` words are already in
    scope (`armres`). Five checks, and every one of them is witnessed RED in this same run — by
    `build/host_probe_sumsq`, the same source with the pre-fix ascending accumulator put back, or by
    a tampered copy of the band written into `work/`. Findings are recorded through `chk`, which
    appends to `fails` itself; `fails` is carried so the call site reads like the section's own.

    The band's shape is why these are the right checks. Each licensed key is `BAND_FACTORS[key]` x the
    worst record in the probe population, and every record is one arm's distance to the f64 anchor,
    so `licensed.max_abs = 2 x population_max` is the triangle inequality on
    `|arm - pinned| <= |arm - anchor| + |pinned - anchor|`. An implementation is inside the family the
    band sizes only if its OWN distance to the anchor is inside that population maximum — the leg a
    plain band verdict cannot see, and the leg the old kernel failed: it measured 2.980e-08 against
    `pinned` (licence 2.900e-08) and 2.626e-08 against the anchor (population 1.450e-08). Licensing
    it would have needed ~5.25e-08 — a relaxation, reserved to the user, refused. No band, probe or
    digest is written by anything here."""
    n_before = len(chk.names)
    qprobe = band["q_hat"]["probe"]["max_abs"]
    qlic = band["q_hat"]["max_abs"]
    qmrel = band["q_hat"]["mean_rel"]
    plog = qhat_probe_log()

    # ---- one ruler per block, and the three spellings of §04.4's reduction measured against it
    per = {}
    for i, block in enumerate(blocks):
        q_pre = dump_tensor(recs, block, "prefill", "q_pre")
        ref = dump_tensor(recs, block, "prefill", "q_hat")
        h_k, h_v, d = int(q_pre.shape[2]), int(ref.shape[1]), int(q_pre.shape[3])
        if (q_pre.ndim != 4 or int(q_pre.shape[1]) != T or int(ref.shape[2]) != T
                or int(ref.shape[3]) != d or h_v % h_k or d & (d - 1)):
            raise Refuse("%s/prefill: q_pre %s and q_hat %s are not the [1,T,h_k,d] -> [1,h_v,T,d] "
                         "expansion with h_v a multiple of h_k and a power-of-two d_k; the head "
                         "expansion below would be a guess about the dump's orientation"
                         % (block, tuple(q_pre.shape), tuple(ref.shape)))
        # `q_pre` is [1,T,h_k,d]; every STAGE tensor is the oracle's scan orientation [1,h_v,T,d].
        # The key-to-value-head expansion is a broadcast and the transpose a permute: neither is a
        # reduction, so neither can move a number the ruler is computed from.
        x = np.transpose(np.repeat(q_pre.to(torch.float32).numpy()[0], h_v // h_k, axis=1),
                         (1, 0, 2))
        pin = ref.numpy()[0]
        arm = stage_tensor(armres, i, T, "q_hat")
        armw = words_of(arm).reshape(-1)
        if int(arm.numel()) != pin.size:
            raise Refuse("%s: the arm emitted %d q_hat words, the dump holds %d elements"
                         % (block, int(arm.numel()), pin.size))
        anch = qhat_anchor(x)
        pin64 = pin.astype(np.float64)
        forms = {}
        for form in ("tree", "wide64", "asc32"):
            m = qhat_l2(x, form)
            m64 = m.astype(np.float64)
            mw = qhat_f32_words(m).reshape(-1)
            pw = qhat_f32_words(pin).reshape(-1)
            forms[form] = dict(model=m, words=mw,
                               vs_anchor=float(np.abs(m64 - anch).max()),
                               vs_pinned=float(np.abs(m64 - pin64).max()),
                               ndiff_vs_pinned=int((mw != pw).sum()),
                               vs_arm_ndiff=int((mw != armw).sum()))
        plog_pin = {(r["pass_"], r["arm"]): r for r in plog["recs"] if r["stage"] == "q_hat"}[
            ("%s/pinned/prefill" % block, "pinned")]
        pinned_vs_anchor = float(np.abs(pin64 - anch).max())
        per[block] = dict(i=i, ref=ref, pin=pin, anch=anch, armw=armw, forms=forms, n=int(pin.size),
                          pinned_vs_anchor=pinned_vs_anchor,
                          ruler_matches_log=float("%.3e" % pinned_vs_anchor) == plog_pin["maxabs"],
                          log_pinned_record=plog_pin["maxabs"],
                          arm_vs_anchor=forms["tree"]["vs_anchor"],
                          arm_vs_pinned=forms["tree"]["vs_pinned"],
                          ndiff_vs_pinned=forms["tree"]["ndiff_vs_pinned"],
                          model_vs_arm_ndiff=forms["tree"]["vs_arm_ndiff"])
        print("QHATRULER block=%s stage=q_hat n=%d d_k=%d h_k=%d h_v=%d shipped=pairwise_f32_tree "
              "model_vs_compiled_arm_ndiff=%d arm_vs_anchor=%.6e arm_vs_pinned=%.6e "
              "ndiff_vs_pinned=%d pinned_vs_anchor=%.6e probe_log_pinned_record=%.3e "
              "probe_population_max=%.17g licensed_max_abs=%.17g inside_population=%s "
              "inside_licence=%s note=the reference's own distance to this anchor is the probe log's "
              "own `pinned` record for this row, which is what makes the anchor the oracle's and not "
              "one this gate invented"
              % (block, per[block]["n"], d, h_k, h_v, per[block]["model_vs_arm_ndiff"],
                 per[block]["arm_vs_anchor"], per[block]["arm_vs_pinned"],
                 per[block]["ndiff_vs_pinned"], pinned_vs_anchor, per[block]["log_pinned_record"],
                 qprobe, qlic, per[block]["arm_vs_anchor"] <= qprobe,
                 per[block]["arm_vs_pinned"] <= qlic))
        for form, label, shipped in (("tree", "pairwise_f32_tree", 1),
                                     ("wide64", "wide_f64_accumulator_llama_cpp_shape", 0),
                                     ("asc32", "ascending_f32_pre_fix_defect", 0)):
            f = forms[form]
            print("QHATARM block=%s arm=%s shipped=%d source=numpy_model_of_gdn_sumsq "
                  "arm_vs_anchor=%.6e inside_population=%s arm_vs_pinned=%.6e ndiff_vs_pinned=%d "
                  "inside_licensed_max_abs=%s"
                  % (block, label, shipped, f["vs_anchor"], f["vs_anchor"] <= qprobe,
                     f["vs_pinned"], f["ndiff_vs_pinned"], f["vs_pinned"] <= qlic))

    # ---- 1: the numpy spelling IS the compiled arm, at zero tolerance
    derived_words = len(blocks) * stage_words(T)["q_hat"]
    seen_words = sum(p["n"] for p in per.values())
    model_ndiff = sum(p["model_vs_arm_ndiff"] for p in per.values())
    chk.check("qhat_numpy_model_bit_identical",
              derived_words > 0 and seen_words == derived_words and model_ndiff == 0,
              "words=%d derived=%d differing_words_vs_compiled_arm=%d blocks=%s "
              "source=stage_tensor(armres) vs qhat_l2(x,'tree') "
              "note=every q_hat distance printed in this run is measured from this model, so the "
              "model has to be the arm and not a look-alike; a spelling that differs from the "
              "compiled words by one word would make the population check below about numpy "
              "(witness: sumsq_control_breaks_the_model_bit_identity)"
              % (seen_words, derived_words, model_ndiff, ",".join(blocks)))

    # ---- 2: the leg the band cannot see on its own — the arm's own distance to the anchor
    inside = [p["arm_vs_anchor"] <= qprobe for p in per.values()]
    ruler_ok = all(p["ruler_matches_log"] for p in per.values())
    chk.check("qhat_arm_inside_probe_population",
              len(per) == len(blocks) and all(inside) and ruler_ok,
              "arm_vs_anchor=%s inside=%d/%d probe_population_max=%.17g licensed_max_abs=%.17g "
              "ruler_agrees_with_probe_log_pinned_records=%s "
              "note=THIS is the verdict: licensed max_abs is 2 x this population maximum because the "
              "band is a triangle inequality on the probe, so an arm inside the licence by being "
              "OUTSIDE the population is outside the family the band sizes even when its row passes. "
              "The reference's own f32 arm sits in the same population, which is the second direction. "
              "(witness: sumsq_control_is_outside_the_probe_population)"
              % (",".join("%s=%.6e" % (b, per[b]["arm_vs_anchor"]) for b in per),
                 sum(inside), len(per), qprobe, qlic, ruler_ok))

    # ---- 3: the rows are ACCEPTED and the licence grants them, at the pinned digest
    row_ok, lic_ok, rows = [], [], []
    for block in sorted(per):
        p = per[block]
        ag = G.aggregates(stage_tensor(armres, p["i"], T, "q_hat").reshape(p["ref"].shape),
                          p["ref"], name="%s/pinned/prefill|%s" % (block, "q_hat"), stage="q_hat")
        ok, why = G.band_check_row(band, "q_hat", ag)
        pass_key = "%s/pinned/%s" % (block, "prefill")
        licensed, lwhy = G.band_licenses(band, pass_key, "q_hat", *LICENCE_PAIR)
        row_ok.append(ok)
        lic_ok.append(licensed)
        rows.append(G.cross_form_key(pass_key, *LICENCE_PAIR))
        p["ag"] = ag
        p["band_ok"] = ok
        print("QHATROW block=%s row=%s max_abs=%.6e licensed=%.6e per_head=%.6e mean_rel=%.6e "
              "licensed_mean_rel=%.6e band=%s licence_granted=%d digest_pinned=%d why=%s licence=%s"
              % (block, rows[-1], ag["max_abs"], band["q_hat"]["max_abs"], ag["per_head_max"],
                 ag["mean_rel"], band["q_hat"]["mean_rel"], "ACCEPTED" if ok else "REJECTED",
                 int(licensed), int(digest == BAND_DIGEST), why, lwhy))
    chk.check("qhat_rows_accepted_and_licensed",
              len(rows) == len(blocks) and all(row_ok) and all(lic_ok) and digest == BAND_DIGEST,
              "rows=%s accepted=%d/%d licensed=%d/%d band_digest=%s note=the verdict is "
              "band_check_row's and the licence is band_licenses', both over the pair this row "
              "actually measures (%s); the digest pin is what stops a regenerated band from being "
              "judged against itself (witnesses: sumsq_control_is_red_at_the_unchanged_band for the "
              "band leg, qhat_licence_fires_on_a_revoked_prefill_row for the licence leg)"
              % (",".join(rows), sum(row_ok), len(rows), sum(lic_ok), len(rows), digest[:16],
                 "|".join(LICENCE_PAIR)))

    # ---- 4: the licence is a fold of its own probe, recomputed rather than trusted
    folds = [(st, key, band[st]["probe"][key] * G.BAND_FACTORS[key], band[st][key])
             for st in sorted(band) for key in sorted(G.BAND_FACTORS)]
    folds_ok = [f[2] == f[3] for f in folds]
    qsum = plog["summary"]["q_hat"]
    qrecs = [r for r in plog["recs"] if r["stage"] == "q_hat" and r["fold"] == "folded"]
    log_max = max(r["maxabs"] for r in qrecs)
    log_probe_keys = dict(maxabs=log_max,
                          mean_rel=max(r["meanrel"] for r in qrecs))
    print("QHATFOLD stages=%d keys=%d folds_checked=%d every_licensed_equals_probe_times_factor=%s | "
          "q_hat parsed_folded_records=%d summary_records_column=%d equal=%s | "
          "log_worst_probe_maxabs=%.3e band_probe_maxabs_at_log_precision=%.3e "
          "log_band_maxabs=%.3e licensed_at_log_precision=%.3e | source=%s"
          % (len(band), len(G.BAND_FACTORS), len(folds), all(folds_ok), len(qrecs),
             qsum["records"], len(qrecs) == qsum["records"], log_probe_keys["maxabs"],
             float("%.3e" % qprobe), qsum["band_maxabs"], float("%.3e" % qlic),
             QH_PROBE_LOG.name))
    chk.check("qhat_bound_is_a_fold_of_its_probe",
              len(folds) == len(band) * len(G.BAND_FACTORS) and all(folds_ok)
              and len(qrecs) == qsum["records"] and len(qrecs) > 0
              and float("%.3e" % qprobe) == log_max
              and float("%.3e" % qlic) == qsum["band_maxabs"],
              "folds_ok=%d/%d q_hat records parsed=%d summary=%d worst_log_record=%.3e "
              "band_probe=%.3e summary_probe_column=%.3e summary_band_column=%.3e licensed=%.17g "
              "note=the licensed number is re-derived from the log's own per-record table instead of "
              "read off the band file, at the log's own printed precision, so a hand-edited bound "
              "fails here as well as in load_band (witnesses: band_tamper_wider_is_refused_by_load_band, "
              "band_tamper_narrower_is_refused_by_load_band, "
              "band_probe_consistent_inflation_is_caught_by_the_digest)"
              % (sum(folds_ok), len(folds), len(qrecs), qsum["records"], log_probe_keys["maxabs"],
                 qprobe, qsum["probe_maxabs"], qsum["band_maxabs"], qlic))

    # ---- 5: the band still refuses a perturbation at a rung derived from its own number
    rung = int(np.floor(-np.log2(8.0 * qmrel)))
    if rung < 1:
        raise Refuse("the derived perturbation rung is %d: licensed mean_rel %.3e is too wide for "
                     "8x to be expressible, so this check has no rung and must not be run" % (rung,
                                                                                              qmrel))
    rel = float(2.0 ** (-rung))
    factor_ratio = rel / qmrel
    pert_ok, ratios = [], []
    for block in sorted(per):
        p = per[block]
        pert = (p["forms"]["tree"]["model"] * np.float32(1.0 + rel)).astype(np.float32)
        agp = G.aggregates(torch.from_numpy(pert).reshape(p["ref"].shape), p["ref"],
                           name="qhat_rung/%s/%s" % (block, "q_hat"), stage="q_hat")
        ok_p, why_p = G.band_check_row(band, "q_hat", agp)
        ratios.append(agp["mean_rel"] / qmrel)
        pert_ok.append(not ok_p)
        p["perturbed"] = (pert, agp, ok_p, why_p)
        print("QHATRUNG block=%s rung=%d derived_from=floor(-log2(8 x licensed_mean_rel)) "
              "licensed_mean_rel=%.6e factor=1+2^-%d relative=%.10g factor_ratio_to_licensed=%.4g "
              "perturbed_mean_rel=%.6e measured_ratio_to_licensed=%.4g band=%s unperturbed_band=%s "
              "why=%s"
              % (block, rung, qmrel, rung, 1.0 + rel, factor_ratio, agp["mean_rel"],
                 agp["mean_rel"] / qmrel, "REJECTED" if not ok_p else "ACCEPTED",
                 "ACCEPTED" if p["band_ok"] else "REJECTED", why_p))
    chk.check("qhat_band_rejects_a_perturbation_at_a_derived_rung",
              len(pert_ok) == len(blocks) and all(pert_ok)
              and all(p["band_ok"] for p in per.values())
              and factor_ratio >= 8.0 and min(ratios) > 1.0,
              "rung=%d factor=1+2^-%d relative_factor_ratio_to_licensed=%.4g measured_mean_rel_ratio="
              "%s rejected=%d/%d unperturbed_accepted=%d/%d "
              "note=the rung comes from the band's own licensed mean_rel and nothing else, and the "
              "UNPERTURBED row must still pass the same comparator, so a predicate that rejects "
              "everything cannot pass this. The measured ratio sits a little under 8 because the "
              "perturbation partly cancels the error that is already there — a real property of a "
              "signed difference, printed rather than hidden "
              "(witness: qhat_perturbation_fires_on_a_probe_consistent_inflation)"
              % (rung, rung, factor_ratio, ",".join("%.4g" % r for r in ratios), sum(pert_ok),
                 len(pert_ok), sum(1 for p in per.values() if p["band_ok"]), len(per)))

    # ---- the two licensed alternatives, measured side by side, plus the pre-fix row
    alt_ok, alt_words_differ = {}, None
    for form in ("tree", "wide64", "asc32"):
        alt_ok[form] = all(per[b]["forms"][form]["vs_anchor"] <= qprobe
                           and per[b]["forms"][form]["vs_pinned"] <= qlic for b in per)
    alt_words_differ = int((per[sorted(per)[0]]["forms"]["tree"]["words"]
                            != per[sorted(per)[0]]["forms"]["wide64"]["words"]).sum())
    chk.check("qhat_reduction_alternatives_both_licensed",
              alt_ok["tree"] and alt_ok["wide64"] and (not alt_ok["asc32"]) and alt_words_differ > 0,
              "pairwise_f32_tree_inside=%s wide_f64_accumulator_inside=%s "
              "ascending_f32_pre_fix_inside=%s tree_vs_wide64_differing_words=%d "
              "note=the shipped form and llama.cpp's wide-accumulator shape are both inside the "
              "population AND the licence at the SAME max_abs, which is what lets the device "
              "budget (fp64 is derated on gfx1151; AGENTS.md §GOAL's >= 48 t/s bar) pick the tree "
              "without touching the licence; the pre-fix ascending row is inside NEITHER, which is "
              "the second direction this predicate needs to have teeth "
              "(witness: sumsq_control_is_outside_the_probe_population)"
              % (alt_ok["tree"], alt_ok["wide64"], alt_ok["asc32"], alt_words_differ))

    qhat_witness_control(band, blocks, T, plan, ins, wts, per, qprobe, chk)
    qhat_witness_band(band, digest, per, blocks, chk)
    counts["qhat_words"] = seen_words
    counts["qhat_probe_records"] = len(qrecs)
    counts["qhat_probe_log_records"] = plog["printed"]
    counts["qhat_derived_rung"] = rung
    counts["qhat_checks_added"] = len(chk.names) - n_before


def qhat_witness_control(band, blocks, T, plan, ins, wts, per, qprobe, chk):
    """`build/host_probe_sumsq` — the same source with `-DGDN_BROKEN_SUMSQ_ASC32`, one ascending f32
    accumulator — witnessed doing three separate damage jobs at the UNCHANGED band.

    This is the proof the fix is load-bearing and not cosmetic. Under the control:
      1. the `q_hat` row is REJECTED at the licence as it stands (the old red: 2.980e-08 against a
         licensed 2.900e-08) and its words differ from the shipped arm's;
      2. its own distance to the anchor is OUTSIDE the population maximum, so the population check
         rejects it and not only the band — the leg that would have been silently relaxed had anyone
         widened the bound to ~5.25e-08 to accommodate it;
      3. the numpy model of the SHIPPED spelling no longer reproduces its words, while the model of
         the ascending spelling reproduces them at zero tolerance. That is what licenses using the
         model as the ruler at all: it tracks the compiled arm across a change of spelling, so the
         thing check 1 verified is the arm and not a look-alike that happens to agree today.
    A missing control build is a named refusal in all three witnesses (the same third state D-21 uses
    for the pad witness), not a silence: a witness that cannot run has witnessed nothing."""
    names = ("sumsq_control_is_red_at_the_unchanged_band",
             "sumsq_control_is_outside_the_probe_population",
             "sumsq_control_breaks_the_model_bit_identity")
    if not SUMSQ.exists():
        for nm in names:
            chk.witness(nm, False,
                        "build=%s exists=0 note=run ./build.sh --broken. With no build that is RED at "
                        "the unchanged band, the q_hat ACCEPTED verdict restates the number it prints"
                        % SUMSQ)
        return
    ctl = run_arm(SUMSQ, "host_twin", plan, ins, wts, "sumsq")
    parse_arm(ctl, len(blocks), T)
    red, moved, outside, model_breaks, model_matches = [], [], [], [], []
    for block in sorted(per):
        p = per[block]
        ref = p["ref"]
        mine = stage_tensor(ctl, p["i"], T, "q_hat")
        w = words_of(mine).reshape(-1)
        w64 = qhat_f64_of_words(w)
        ag = G.aggregates(mine.reshape(ref.shape), ref, name="sumsq_control/%s/q_hat" % block,
                          stage="q_hat")
        ok, why = G.band_check_row(band, "q_hat", ag)
        vs_anchor = float(np.abs(w64 - p["anch"].reshape(-1)).max())
        vs_pin = float(np.abs(w64 - p["pin"].astype(np.float64).reshape(-1)).max())
        nd_honest = int((w != p["armw"]).sum())
        nd_tree = int((w != p["forms"]["tree"]["words"]).sum())
        nd_asc = int((w != p["forms"]["asc32"]["words"]).sum())
        red.append(not ok)
        moved.append(nd_honest > 0)
        outside.append(vs_anchor > qprobe)
        model_breaks.append(nd_tree > 0)
        model_matches.append(nd_asc == 0)
        print("QHATCONTROL block=%s build=host_probe_sumsq define=GDN_BROKEN_SUMSQ_ASC32 "
              "ndiff_vs_shipped_arm=%d arm_vs_anchor=%.6e outside_population=%s arm_vs_pinned=%.6e "
              "ndiff_vs_pinned=%d licensed_max_abs=%.17g population_max=%.17g band=%s "
              "model_tree_words_differ=%d model_asc32_words_differ=%d why=%s"
              % (block, nd_honest, vs_anchor, vs_anchor > qprobe, vs_pin, ag["ndiff"],
                 band["q_hat"]["max_abs"], qprobe, "REJECTED" if not ok else "ACCEPTED",
                 nd_tree, nd_asc, why))
    print("QHATCONTROLSUM build=host_probe_sumsq blocks=%d rejected=%d/%d outside_population=%d/%d "
          "model_shipped_broken=%d/%d model_pre_fix_matches=%d/%d note=the pre-fix spelling, "
          "compiled; these are the two red rows of HANDOFF §SLICE 4, and the point of printing them "
          "from a build is that they are red at a band nothing in this run touched"
          % (len(per), sum(red), len(per), sum(outside), len(per), sum(model_breaks), len(per),
             sum(model_matches), len(per)))
    chk.witness("sumsq_control_is_red_at_the_unchanged_band",
                len(red) == len(blocks) and all(red) and all(moved),
                "rejected=%d/%d every_row_moved_vs_shipped_arm=%s licensed_max_abs=%.17g "
                "note=the band, its probe block and its digest are the ones the honest run judged; a "
                "build that stayed inside them would mean the shipped change did nothing"
                % (sum(red), len(red), all(moved), band["q_hat"]["max_abs"]))
    chk.witness("sumsq_control_is_outside_the_probe_population",
                len(outside) == len(blocks) and all(outside),
                "outside=%d/%d population_max=%.17g note=the same predicate check 2 applies to the "
                "shipped arm, applied to the pre-fix arm, is false — that is the measured difference "
                "between `the band was one rounding tight` and `the kernel was outside the family the "
                "band sizes`, and the reason ~5.25e-08 was never on the table"
                % (sum(outside), len(outside), qprobe))
    chk.witness("sumsq_control_breaks_the_model_bit_identity",
                len(model_breaks) == len(blocks) and all(model_breaks) and all(model_matches),
                "model_shipped_differs_from_control=%d/%d model_pre_fix_identical_to_control=%d/%d "
                "note=both directions: the shipped model must NOT describe the control's words and "
                "the pre-fix model must describe them exactly, or the ruler check 1 verified belongs "
                "to a spelling nobody compiled"
                % (sum(model_breaks), len(model_breaks), sum(model_matches), len(model_matches)))


def qhat_tampered_band_file(tag, mutate):
    """Write a tampered COPY of the band into work/ and return its path.

    Only the file's own bytes are read and nothing at `G.BAND_PATH` is ever written, so the honest
    object the witnesses are compared against cannot be moved by the tampering."""
    doc = json.loads(G.BAND_PATH.read_text())
    mutate(doc["band"])
    path = WORK / ("band_tamper_qhat_%s.json" % tag)
    path.write_text(json.dumps(doc))
    return path


def qhat_witness_band(band, digest, per, blocks, chk):
    """The band itself, witnessed as a live constraint and not a transcription.

    Three tamperings, each printed beside the honest leg so no witness can fire on a broken tool:
      wider / narrower   one licensed key moved on its own. `load_band` enforces
                         `licensed == probe x BAND_FACTORS` per key, so either direction is an
                         `AssertionError`. Both directions are needed: a reader that refused every
                         file would pass a one-sided test.
      probe-consistent   probe and licensed moved TOGETHER, which is arithmetically self-consistent
                         and therefore exactly the edit `load_band` cannot refuse. The pinned digest
                         is what catches it, so this is the witness that the digest pin is load-
                         bearing and not decoration — the (a) that was rejected in this slice.
      revoked row        the licence grant for one prefill row withdrawn from an in-memory copy, so
                         the ACCEPTED-plus-licensed check is shown to answer to the table.
      inflated + rung    the derived-rung perturbation, judged under the inflated band: it passes """
    honest_digest = G.band_digest(G.load_band(G.BAND_PATH))

    def tamper_one(factor):
        def mutate(b):
            ent = b["q_hat"]
            ent["max_abs"] = ent["probe"]["max_abs"] * G.BAND_FACTORS["max_abs"] * factor
        return mutate

    for label, factor, name in (("wider", 2.0, "band_tamper_wider_is_refused_by_load_band"),
                                ("narrower", 0.5, "band_tamper_narrower_is_refused_by_load_band")):
        path = qhat_tampered_band_file(label, tamper_one(factor))
        refused, reason = False, ""
        try:
            G.load_band(path)
        except AssertionError as exc:
            refused, reason = True, str(exc)
        chk.witness(name, refused and honest_digest == BAND_DIGEST,
                    "tamper=q_hat.max_abs_x%s file=%s refused=%s "
                    "honest_still_loads_at_pinned_digest=%s reason=%s note=a load_band that refused "
                    "every file would satisfy the refused leg alone, so the honest digest beside it is "
                    "the second direction"
                    % (factor, path.name, refused, honest_digest == BAND_DIGEST, reason[:150]))

    def inflate(b, scale=16.0):
        for key in G.BAND_FACTORS:
            b["q_hat"]["probe"][key] = b["q_hat"]["probe"][key] * scale
            b["q_hat"][key] = b["q_hat"][key] * scale

    infl_path = qhat_tampered_band_file("inflated", inflate)
    infl_band, infl_ok, infl_reason = None, False, ""
    try:
        infl_band = G.load_band(infl_path)
        infl_ok = True
    except AssertionError as exc:
        infl_reason = str(exc)
    infl_digest = G.band_digest(infl_band) if infl_band is not None else ""
    chk.witness("band_probe_consistent_inflation_is_caught_by_the_digest",
                infl_ok and infl_digest != BAND_DIGEST and honest_digest == BAND_DIGEST,
                "tamper=q_hat_probe_and_licensed_x16 file=%s load_band_accepted=%s reason=%s "
                "inflated_digest=%s honest_digest_pinned=%d note=this is the inflation load_band "
                "CANNOT refuse, because every key is still its own probe times its own factor; the "
                "digest pin is the guard, which is the measured answer to `could the band have been "
                "quietly widened to license the old kernel` "
                % (infl_path.name, infl_ok, infl_reason[:120], infl_digest[:16],
                   honest_digest == BAND_DIGEST))

    pass_key = "%s/pinned/prefill" % sorted(per)[0]
    row_key = G.cross_form_key(pass_key, *LICENCE_PAIR)
    granted_honest, _ = G.band_licenses(band, pass_key, "q_hat", *LICENCE_PAIR)
    revoked = json.loads(json.dumps(band))
    revoked["q_hat"]["licensed_rows"] = [k for k in revoked["q_hat"].get("licensed_rows", [])
                                         if k != row_key]
    granted_revoked, rwhy = G.band_licenses(revoked, pass_key, "q_hat", *LICENCE_PAIR)
    chk.witness("qhat_licence_fires_on_a_revoked_prefill_row",
                granted_honest and not granted_revoked,
                "row=%s stage=q_hat honest_licence=%d revoked_licence=%d revoked_reason=%s "
                "note=the licence leg of check 3 is a lookup, so it has to be shown to answer to the "
                "table: revoke the grant and the same call must refuse "
                % (row_key, int(granted_honest), int(granted_revoked), rwhy[:80]))

    infl_oks = []
    if infl_band is None:
        chk.witness("qhat_perturbation_fires_on_a_probe_consistent_inflation", False,
                    "inflated_band=%s could_not_be_loaded=%s note=with no inflated band to judge "
                    "against, the rung rejection has no second direction and the check stays "
                    "unwitnessed: reason=%s" % (infl_path.name, True, infl_reason[:120]))
        return
    for block in sorted(per):
        _, agp, ok_honest, why_p = per[block]["perturbed"]
        ok_infl, why_infl = G.band_check_row(infl_band, "q_hat", agp)
        infl_oks.append((bool(ok_infl), not ok_honest))
        print("QHATINFLATE block=%s rung_perturbation_under_honest_band=%s "
              "same_perturbation_under_inflated_band=%s licensed_mean_rel_honest=%.6e "
              "inflated_mean_rel=%.6e perturbed_mean_rel=%.6e why_honest=%s why_inflated=%s"
              % (block, "REJECTED" if not ok_honest else "ACCEPTED",
                 "ACCEPTED" if ok_infl else "REJECTED", band["q_hat"]["mean_rel"],
                 infl_band["q_hat"]["mean_rel"], agp["mean_rel"], why_p[:60], why_infl[:60]))
    chk.witness("qhat_perturbation_fires_on_a_probe_consistent_inflation",
                len(infl_oks) == len(blocks) and all(o and r for o, r in infl_oks),
                "rejected_under_honest_band=%d/%d accepted_under_inflated_band=%d/%d "
                "note=the rung rejection is a fact about the licensed number and not about the code: "
                "inflate the band 16x and the same perturbation passes. That is the witness that "
                "check 5 measures the band and not its own arithmetic "
                % (sum(1 for _, r in infl_oks if r), len(infl_oks),
                   sum(1 for o, _ in infl_oks if o), len(infl_oks)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--oracle-seconds", type=int, default=1800,
                    help="wall limit for the oracle CLI arm; it is CPU work and it is not a timing")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    def want(name):
        return only is None or name in only

    WORK.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    fails = []
    counts = {}
    ran = []
    stage_lines = []          # every STAGE line this run prints; `determinism` compares it
    g = geom()
    check_state_affected()
    man, recs, blocks, T = load_dump()

    band = G.load_band(G.BAND_PATH)
    digest = G.band_digest(band)
    print("BAND file=%s digest=%s stages=%d authoritative=%d keys=%s"
          % (G.BAND_PATH.name, digest, len(band), 1 if digest == BAND_DIGEST else 0,
             ",".join(sorted(band))))
    if digest != BAND_DIGEST:
        raise Refuse("band digest %s is not the pinned %s: refusing to judge anything against a "
                     "band this lane did not derive" % (digest[:16] + "…", BAND_DIGEST[:16] + "…"))
    for s in STAGES:
        if s not in band:
            raise Refuse("stage %s has no band entry" % s)
    chk = Checks(fails)
    pad = PadWitness()
    lic = Licence(band, g, pad=pad)

    # ---------------- the honest plan: one case per block, one chunk, zero initial state -------
    fields = {b: field_words(recs, b, T) for b in blocks}
    wslots = {b: weight_words(recs, b) for b in blocks}
    plan = WORK / "plan.tsv"
    write_plan(plan, [dict(block=b, name="prefill", T=T, mode="honest", wslot=i)
                      for i, b in enumerate(blocks)])
    ins = WORK / "inputs.bin"
    wts = WORK / "weights.bin"
    in_words, _ = write_inputs(ins, [fields[b] for b in blocks], T)
    w_words = write_weights(wts, [wslots[b] for b in blocks])
    print("PLAN cases=%d T=%d in_words=%d weight_words=%d zero_state=1 pad_cols=%d"
          % (len(blocks), T, in_words, w_words, 64 - T))

    # ============================================================ arm 3: the two compiled arms
    if want("three-arm"):
        ran.append("three-arm")
        for b in (HOST, DEVI):
            if not b.exists():
                raise Refuse("%s is not built; run ./build.sh" % b)
        arms = {}
        for path, name in ((HOST, "host_twin"), (DEVI, "hip_hostpath")):
            st = subprocess.run([str(path), "--selftest"], capture_output=True, text=True)
            print("SELFTEST %s" % "\n".join((st.stdout + st.stderr).splitlines()))
            if st.returncode != 0:
                fails.append("selftest of %s failed (rc=%d)" % (name, st.returncode))
            arms[name] = run_arm(path, name, plan, ins, wts, "honest")
            r = arms[name]
            if r["rc"] != 0:
                fails.append("arm %s rc=%d: %s" % (name, r["rc"], r["lines"][-1] if r["lines"] else ""))
            cs, fps, run = parse_arm(r, len(blocks), T)
            print("ARM arm=%s cases=%d compiler=%s hip_calls=%s carried=%s stage_words=%s"
                  % (name, len(cs), run.get("compiler"), run.get("hip_calls"),
                     run.get("carried"), run.get("case_words")))
            if name == "hip_hostpath":
                if run.get("arm") != "hip_hostpath":
                    fails.append("the device TU did not announce arm=hip_hostpath (said %r)"
                                 % run.get("arm"))
                if run.get("hip_calls") != "0":
                    fails.append("the device-compiled arm issued hip_calls=%s; the CPU arm must "
                                 "issue none, or it is not the arm this comparison claims"
                                 % run.get("hip_calls"))
            if int(run.get("carried", "1")) != 0:
                fails.append("a zero-state plan reported carried=%s; no case in it may read a "
                             "carried state, or the plan the band is judged on is not the plan "
                             "this section says it ran" % run.get("carried"))
            arms[name + "_cases"], arms[name + "_fps"] = cs, fps
        if len(arms) < 4:
            raise Refuse("fewer than two compiled arms ran; a single-arm run cannot show anything")

        # the compiled pair, at ZERO tolerance
        for key, label in (("out", "out.bin"), ("echo", "echo.bin")):
            per = arms["host_twin"]["case_words"] if key == "out" else arms["host_twin"]["echo_words"]
            ba = np.fromfile(arms["host_twin"][key], dtype=np.uint8)
            bb = np.fromfile(arms["hip_hostpath"][key], dtype=np.uint8)
            if ba.size != bb.size:
                fails.append("compiled pair: %s sizes differ (%d vs %d)" % (label, ba.size, bb.size))
                continue
            nd = int((ba != bb).sum())
            print("PAIR pair=host_twin_vs_hip_hostpath file=%s bytes=%d ndiff_bytes=%d "
                  "cases=%d per_case_words=%d zero_tolerance=1 verdict=%s"
                  % (label, ba.size, nd, len(blocks), per, "IDENTICAL" if nd == 0 else "DIFFERS"))
            if nd:
                fails.append("compiled pair differs on %d bytes of %s: one source, two compilers, "
                             "so this is a codegen finding and not something a band may absorb"
                             % (nd, label))
        counts["pair_files"] = 2
        counts["pair_words"] = 2 * len(blocks)

    # ============================================================ arms 1 and 3, per stage, banded
    if want("band"):
        ran.append("band")
        armres = run_arm(HOST, "host_twin", plan, ins, wts, "honest")
        cs, fps, run = parse_arm(armres, len(blocks), T)
        n_rows = 0
        for i, block in enumerate(blocks):
            for st in STAGES:
                mine = stage_tensor(armres, i, T, st)
                ref = dump_tensor(recs, block, "prefill", st)
                ok, m = band_row(band, "kernel_vs_dump", block, "prefill", st, mine, ref,
                                 stage_lines, fails,
                                 lambda b, p, s, w: "%s/%s outside the derived band: %s"
                                                    % (b, s, w))
                lic.cite(block, "prefill", st, plan="single_chunk")
                n_rows += 1
                if st == "q_hat":
                    print("KNOWN stage=q_hat block=%s pass=prefill measured_maxabs=%.3e "
                          "licensed_maxabs=%.3e band=%s resolution=kernel_reduction_form_d23 "
                          "note=band NOT touched NOT widened: digest %s unchanged; the row closed "
                          "when gdn_sumsq became a declared pairwise f32 tree, numbers in QHATRULER "
                          "below"
                          % (block, m["maxabs"], band["q_hat"]["max_abs"],
                             "ACCEPTED" if ok else "REJECTED", digest[:16]))
        counts["stage_rows"] = n_rows
        if n_rows != len(blocks) * len(STAGES):
            fails.append("stage rows %d != derived %d" % (n_rows, len(blocks) * len(STAGES)))
        if n_rows == 0:
            raise Refuse("no stage rows: a banded verdict over nothing is a constant predicate")
        licence_tally(chk, lic, "band:single_chunk", len(blocks) * len(STAGES), counts)
        qhat_checks(band, digest, recs, blocks, T, plan, ins, wts, armres, fails, counts, chk)

    # ============================================ the multi-chunk question (slice 3)
    if want("chunkcarry"):
        ran.append("chunkcarry")
        chunk_carry_section(band, man, recs, blocks, T, g, wslots, plan, ins, wts, lic, chk,
                            fails, counts, stage_lines)

    # ============================================================ the live-state carry (slice 2)
    if want("carry"):
        ran.append("carry")
        carry_section(band, g, recs, blocks, T, wslots, plan, ins, wts, stage_lines,
                      fails, counts, lic, chk, pad)

    # ============================================================ the exact read/decode checks
    if want("exact"):
        ran.append("exact")
        armres = run_arm(HOST, "host_twin", plan, ins, wts, "honest")
        cs, fps, run = parse_arm(armres, len(blocks), T)
        n_ok, n_bad = 0, 0
        for i, block in enumerate(blocks):
            for f in ECHO_FIELDS:
                seen = echo_tensor(armres, i, T, f)
                # the same seven fields the arm echoes, in the arm's order: the six scan inputs and
                # norm_weight (which lives in weights.bin but is echoed in the same order)
                truth = np.concatenate(list(fields[block]) + [wslots[block][0]])[
                    echo_offset(T, f):echo_offset(T, f) + field_n(f, T)]
                nd = int((seen != truth).sum())
                n_ok += nd == 0
                n_bad += nd != 0
                print("EXACT arm=kernel block=%s field=%s n=%d ndiff=%d maxword=0x%x verdict=%s"
                      % (block, f, seen.size, nd,
                         int((seen ^ truth).max()) if seen.size else 0,
                         "BIT_IDENTICAL" if nd == 0 else "RED"))
                if nd:
                    fails.append("%s: the kernel loaded %d word(s) of %s that are not the dumped "
                                 "bytes — a read/decode fact has no tolerance" % (block, nd, f))
            # out_weight is 8,388,608 words per case: fingerprinted, not echoed
            w = wslots[block][1]
            s, si, n = fp_of_words(w)
            got = fps[(i, FP_FIELD)]
            okn = int(got["n"]) == n
            oks = int(got["sum"], 16) == s
            oki = int(got["sumidx"], 16) == si
            n_ok += okn and oks and oki
            n_bad += not (okn and oks and oki)
            print("EXACT arm=kernel block=%s field=%s n=%d n_agree=%d sum_agree=%d sumidx_agree=%d "
                  "verdict=%s" % (block, FP_FIELD, n, okn, oks, oki,
                                  "BIT_IDENTICAL" if (okn and oks and oki) else "RED"))
            if not (okn and oks and oki):
                fails.append("%s: the kernel's out_weight fingerprint is not the dumped tensor's"
                             % block)
        counts["exact_field_checks"] = n_ok + n_bad
        counts["exact_field_red"] = n_bad
        if counts["exact_field_checks"] != len(blocks) * len(FIELDS):
            fails.append("exact-field checks ran %d times, derived denominator is %d"
                         % (counts["exact_field_checks"], len(blocks) * len(FIELDS)))

    # ============================================================ honesty controls
    if want("controls"):
        ran.append("controls")
        # (a) STUB: no arithmetic at all. Every stage must move, on every block, or the comparator
        #     is reading a constant and not the differ.
        stub_plan = WORK / "plan_stub.tsv"
        write_plan(stub_plan, [dict(block=b, name="prefill", T=T, mode="stub", wslot=i)
                               for i, b in enumerate(blocks)])
        res = run_arm(HOST, "host_twin", stub_plan, ins, wts, "stub")
        parse_arm(res, len(blocks), T)
        caught, rows = 0, 0
        for i, block in enumerate(blocks):
            for st in STAGES:
                mine = stage_tensor(res, i, T, st).reshape(
                    dump_tensor(recs, block, "prefill", st).shape)
                ref = dump_tensor(recs, block, "prefill", st)
                m = metrics(mine, ref)
                ag = G.aggregates(mine, ref, name="stub/%s/%s" % (block, st), stage=st)
                ok, why = G.band_check_row(band, st, ag)
                rows += 1
                caught += 0 if ok else 1
                if ok or m["ndiff"] == 0:
                    fails.append("STUB control: %s/%s survived the band with ndiff=%d — the "
                                 "comparator is not reading the differ" % (block, st, m["ndiff"]))
        print("CONTROL arm=stub rows=%d caught=%d survivors=%d derived=%d verdict=%s"
              % (rows, caught, rows - caught, len(blocks) * len(STAGES),
                 "RED_ON_DERIVED_COUNT" if rows == caught == len(blocks) * len(STAGES) else "FAIL"))
        if rows != len(blocks) * len(STAGES):
            fails.append("stub control ran %d rows, derived denominator is %d"
                         % (rows, len(blocks) * len(STAGES)))

        # (b) CROSSED: case i is fed block i^1's inputs but judged against block i's pinned bytes.
        #     Every row must go red; a row that stays inside the band means the pipeline is not
        #     actually reading the case it claims to.
        cross_plan = WORK / "plan_cross.tsv"
        order = list(reversed(range(len(blocks))))
        write_plan(cross_plan, [dict(block=blocks[i], name="prefill_crossed", T=T, mode="honest",
                                     wslot=order[i]) for i in range(len(blocks))])
        cins = WORK / "inputs_cross.bin"
        write_inputs(cins, [fields[blocks[j]] for j in order], T)
        res = run_arm(HOST, "host_twin", cross_plan, cins, wts, "cross")
        parse_arm(res, len(blocks), T)
        caught, rows = 0, 0
        for i, block in enumerate(blocks):
            for st in STAGES:
                mine = stage_tensor(res, i, T, st).reshape(
                    dump_tensor(recs, block, "prefill", st).shape)
                ref = dump_tensor(recs, block, "prefill", st)
                m = metrics(mine, ref)
                ag = G.aggregates(mine, ref, name="crossed/%s/%s" % (block, st), stage=st)
                ok, why = G.band_check_row(band, st, ag)
                rows += 1
                caught += 0 if ok else 1
                if ok:
                    fails.append("CROSSED control: block %s judged against the wrong block's "
                                 "inputs is still inside the band at %s (%s)" % (block, st, why))
        print("CONTROL arm=crossed rows=%d caught=%d survivors=%d derived=%d verdict=%s"
              % (rows, caught, rows - caught, len(blocks) * len(STAGES),
                 "RED_ON_DERIVED_COUNT" if rows == caught == len(blocks) * len(STAGES) else "FAIL"))
        if rows != len(blocks) * len(STAGES):
            fails.append("crossed control ran %d rows, derived denominator is %d"
                         % (rows, len(blocks) * len(STAGES)))

        # (c) single-arm refusal: a three-arm claim that only found one arm must refuse.
        try:
            _require_three_arms(["dump", "kernel"])
            fails.append("single-arm refusal did not fire")
            print("CONTROL arm=single_arm verdict=NOT_REFUSED")
        except Refuse as e:
            print("CONTROL arm=single_arm arms=2 verdict=REFUSED reason=%s" % e)

        # (d) the broken control build: the same source with a bf16 rule that truncates. Every
        #     check that is supposed to be able to fail must fail here, or it cannot fail.
        if BROKEN.exists():
            st = subprocess.run([str(BROKEN), "--selftest"], capture_output=True, text=True)
            tail = (st.stdout + st.stderr).strip().splitlines()[-1]
            selftest_fired = st.returncode != 0
            res = run_arm(BROKEN, "host_twin", plan, ins, wts, "broken")
            parse_arm(res, len(blocks), T)
            red_stages = 0
            for i, block in enumerate(blocks):
                for s2 in STAGES:
                    mine = stage_tensor(res, i, T, s2).reshape(
                        dump_tensor(recs, block, "prefill", s2).shape)
                    ref = dump_tensor(recs, block, "prefill", s2)
                    ag = G.aggregates(mine, ref, name="broken/%s/%s" % (block, s2), stage=s2)
                    ok, _ = G.band_check_row(band, s2, ag)
                    red_stages += 0 if ok else 1
            print("CONTROL arm=broken_bf16_trunc selftest=%s stage_rows=%d red=%d verdict=%s"
                  % ("FIRE" if selftest_fired else "SILENT", len(blocks) * len(STAGES), red_stages,
                     "WITNESSED_FAILING" if (selftest_fired and red_stages) else "FAIL"))
            if not selftest_fired:
                fails.append("the broken control build passed the self-test: the self-test cannot fail")
            if not red_stages:
                fails.append("the broken control build stayed inside every band: the band "
                             "comparison cannot fail")
        else:
            print("CONTROL arm=broken_bf16_trunc verdict=NOT-BUILT run ./build.sh --broken")
            counts["broken_control"] = 0

        # (e) the two honesty controls again on the LIVE-STATE cases. A stub that survives only when
        #     the state is zero would be a comparator that reads the state term and nothing else, and
        #     a crossed case fed another block's live state is the sharpest version of "is the
        #     pipeline reading the case it claims" — the state has to be crossed with the fields.
        cse, sts, _ = carry_material(recs, blocks)
        ins_state = WORK / "inputs_state.bin"
        write_inputs(ins_state, [c["fields"] for c in cse], 1, states=sts)
        n_c = len(cse)
        stub2 = WORK / "plan_stub_state.tsv"
        for c in cse:
            c["mode"] = "stub"
        write_plan(stub2, cse)
        res = run_arm(HOST, "host_twin", stub2, ins_state, wts, "stub_state")
        parse_arm(res, n_c, 1)
        caught, rows = 0, 0
        for i, c in enumerate(cse):
            for st in STAGES:
                ref = dump_tensor(recs, c["block"], c["name"], st)
                mine = stage_tensor(res, i, 1, st).reshape(ref.shape)
                m = metrics(mine, ref)
                ag = G.aggregates(mine, ref, name="stub_state/%s/%s" % (c["block"], c["name"]),
                                  stage=st)
                ok, _ = G.band_check_row(band, st, ag)
                rows += 1
                caught += 0 if ok else 1
                if ok or m["ndiff"] == 0:
                    fails.append("STUB(live-state) %s/%s survived the band with ndiff=%d"
                                 % (c["block"], c["name"], m["ndiff"]))
        print("CONTROL arm=stub_live_state rows=%d caught=%d survivors=%d derived=%d verdict=%s"
              % (rows, caught, rows - caught, n_c * len(STAGES),
                 "RED_ON_DERIVED_COUNT" if rows == caught == n_c * len(STAGES) else "FAIL"))
        if rows != n_c * len(STAGES):
            fails.append("stub-live-state control ran %d rows, derived denominator is %d"
                         % (rows, n_c * len(STAGES)))

        order = list(reversed(range(len(blocks))))
        cross_cases = []
        cross_states = []
        for bi, b in enumerate(blocks):
            for p in DEC_PASSES:
                src = blocks[order[bi]]
                cross_cases.append(dict(block=b, name=p, T=1, mode="honest", wslot=order[bi],
                                        carry=-1, fields=field_words(recs, src, 1, p)))
                cross_states.append(words_of(
                    dump_tensor(recs, src, PRED[p], "state").reshape(-1)))
        cross2 = WORK / "plan_cross_state.tsv"
        write_plan(cross2, cross_cases)
        cins2 = WORK / "inputs_cross_state.bin"
        write_inputs(cins2, [c["fields"] for c in cross_cases], 1, states=cross_states)
        res = run_arm(HOST, "host_twin", cross2, cins2, wts, "cross_state")
        parse_arm(res, n_c, 1)
        caught, rows = 0, 0
        for i, c in enumerate(cse):
            for st in STAGES:
                ref = dump_tensor(recs, c["block"], c["name"], st)
                mine = stage_tensor(res, i, 1, st).reshape(ref.shape)
                ag = G.aggregates(mine, ref, name="cross_state/%s/%s" % (c["block"], c["name"]),
                                  stage=st)
                ok, _ = G.band_check_row(band, st, ag)
                rows += 1
                caught += 0 if ok else 1
                if ok:
                    fails.append("CROSSED(live-state): %s/%s fed the other block's fields AND its "
                                 "live state is still inside the band at %s" % (c["block"],
                                                                               c["name"], st))
        print("CONTROL arm=crossed_live_state rows=%d caught=%d survivors=%d derived=%d verdict=%s"
              % (rows, caught, rows - caught, n_c * len(STAGES),
                 "RED_ON_DERIVED_COUNT" if rows == caught == n_c * len(STAGES) else "FAIL"))
        if rows != n_c * len(STAGES):
            fails.append("crossed-live-state control ran %d rows, derived denominator is %d"
                         % (rows, n_c * len(STAGES)))

        # (f) the plan rules the carry depends on, each witnessed REFUSING. A self-carry or a
        #     forward carry would read a state buffer that has not been written yet — calloc zeros,
        #     which is precisely the zero a state test cannot see — and a cross-block carry reads
        #     another block's head history. A case T that differs from the plan's T would bind the
        #     case at a stride its own field sizes disagree with.
        bads = [
            ("self_carry", ["#GDNPLAN v2 n_cases=2 T=1",
                            "CASE idx=0 block=layer0 name=decode0 T=1 mode=honest wslot=0 carry=-1",
                            "CASE idx=1 block=layer0 name=decode1 T=1 mode=honest wslot=0 carry=1"]),
            ("cross_block_carry", ["#GDNPLAN v2 n_cases=2 T=1",
                                   "CASE idx=0 block=layer0 name=decode0 T=1 mode=honest wslot=0 carry=-1",
                                   "CASE idx=1 block=layer1 name=decode0 T=1 mode=honest wslot=1 carry=0"]),
            ("case_T_differs", ["#GDNPLAN v2 n_cases=1 T=1",
                                "CASE idx=0 block=layer0 name=decode0 T=2 mode=honest wslot=0 carry=-1"]),
        ]
        refused, bad_rows = 0, 0
        for label, lines in bads:
            badp = WORK / ("plan_bad_%s.tsv" % label)
            badp.write_text("\n".join(lines) + "\n")
            pr = subprocess.run([str(HOST), "--plan", str(badp), "--inputs", str(ins_state),
                                 "--weights", str(wts), "--out", str(WORK / "bad.out.bin"),
                                 "--echo", str(WORK / "bad.echo.bin")],
                                capture_output=True, text=True)
            msg = ((pr.stdout + pr.stderr).strip().splitlines() or ["<silent>"])[-1]
            bad_rows += 1
            refused += 1 if pr.returncode != 0 else 0
            if pr.returncode == 0:
                fails.append("the arm ACCEPTED a %s plan, which the carry rules forbid: %s"
                             % (label, msg))
            print("CONTROL arm=bad_plan rule=%s rc=%d verdict=%s reason=%s"
                  % (label, pr.returncode, "REFUSED" if pr.returncode else "ACCEPTED", msg[:100]))
        if refused != len(bads):
            fails.append("bad-plan refusals fired %d of %d" % (refused, len(bads)))
        counts["controls"] = (4 if BROKEN.exists() else 3) + 2 + len(bads)

    # ============================================================ the norm_weight witness
    if want("witness"):
        ran.append("witness")
        witness(band, recs, blocks, T, fields, wslots, plan, ins, wts, counts)

    # ============================================================ the oracle's own CLI arm
    if want("oracle"):
        ran.append("oracle")
        cmd = [PY, str(REPO / "speccheck" / "gdn_ref.py"), "--cross-form", "--threads", "8"]
        print("ORACLECMD %s" % " ".join(cmd))
        log = EVID / "gdn-cross-form-in-lane-2026-10-26.txt"
        with log.open("w") as fh:
            try:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                                   timeout=a.oracle_seconds, cwd=str(REPO))
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = -1
        txt = log.read_text()
        dig = [l for l in txt.splitlines() if "digest" in l][:1]
        tail = [l for l in txt.splitlines() if l.strip()][-1] if txt.strip() else "<empty>"
        print("ORACLE rc=%d log=%s lines=%d digest_line=%s last=%s"
              % (rc, log.name, len(txt.splitlines()), (dig[0].strip()[:80] if dig else "NONE"),
                 tail[:80]))
        if rc != 0:
            fails.append("the oracle CLI arm (--cross-form) exited %d; the band it derived is not "
                         "re-verifiable in this window" % rc)
        counts["oracle_rows"] = len(txt.splitlines())

    # ============================================================ determinism, as a named check
    if want("determinism"):
        ran.append("determinism")
        determinism_section(stage_lines, fails, counts, ran)

    # ------------------------------------------------------------------ verdict
    print("METRICS primary=relL2+ndiff+band_check_row reported_only=maxabs,cos "
          "maxabs_load_bearing=0 cos_load_bearing=0 band_source=band_check_row_only")
    lic_cred = sum(r["credited"] for r in lic.rows)
    lic_unlicensed = sum(1 for r in lic.rows if r["verdict"] == "UNLICENSED")
    lic_unprobed = sum(1 for r in lic.rows if r["verdict"] == "LICENSED_FORM_UNPROBED")
    lic_unarm = sum(1 for r in lic.rows if r["verdict"] == "UNLICENSED_ARM_NOT_DISPATCHED")
    lic_unest = sum(1 for r in lic.rows if r["verdict"] == "CANNOT_BE_ESTABLISHED")
    lic_uncredited = len(lic.rows) - lic_cred
    counts["pad_witness_fired"] = 1 if pad.fires_on_decode_rows() else 0
    print("LICENCEVERDICT scope=whole_run banded_rows=%d credited=%d unlicensed=%d "
          "licensed_form_unprobed=%d unlicensed_arm_not_dispatched=%d cannot_be_established=%d "
          "pad_witness=%s checks=%d checks_red=%d verdict=%s "
          "note=an uncredited row keeps its band verdict and its measurement; what it does not get "
          "is the word licensed. The prefill rows need a probe record for `chunked|pinned` at that "
          "pass, which only `speccheck/gdn_ref.py --floor-probe` may write; the decode rows are "
          "judged under the licence §04b.1 can actually produce for them (%s) and are credited only "
          "while the arm-internal pad witness fires (D-21)"
          % (len(lic.rows), lic_cred, lic_unlicensed, lic_unprobed, lic_unarm, lic_unest,
             pad.state, len(chk.names), chk.red,
             "CLEAN" if not lic_uncredited else "QUALIFIED",
             "|".join(bridge_pair())))
    counts["banded_rows_licensed"] = lic_cred
    counts["banded_rows_uncredited"] = lic_uncredited
    counts["named_checks"] = len(chk.names)
    counts["named_checks_red"] = chk.red
    print("COUNTS %s" % " ".join("%s=%d" % kv for kv in sorted(counts.items())))
    if fails:
        print("GATE FAIL findings=%d uncredited_banded_rows=%d" % (len(fails), lic_uncredited))
        for f in fails:
            print("  - " + f)
        return 1
    if not counts:
        raise Refuse("no section ran; a gate that checked nothing cannot pass")
    missing = [s for s in SECTIONS if s not in ran]
    if missing:
        print("GATE PARTIAL ran=%s missing=%s uncredited_banded_rows=%d findings=%d "
              "note=a partial run does not claim PASS: the three-arm "
              "claim needs the dump arm, the oracle CLI arm and the kernel arm, and one of them is "
              "not in this run" % (",".join(ran), ",".join(missing), lic_uncredited, len(fails)))
        return 2
    tally = ("findings=%d band_rows_rejected=%d rejected_untaxed=%d rejected_became_findings=%d"
             % (len(fails), _REJECT_TALLY.get("rejected", 0),
                _REJECT_TALLY.get("untaxed", 0), _REJECT_TALLY.get("taxed", 0)))
    if lic_uncredited:
        print("GATE QUALIFIED arms=3 stages=%d blocks=%d T=%d band_digest=%s device=none "
              "timing=taken=no banded_rows=%d credited=%d uncredited=%d %s "
              "note=GATE PASS is unreachable while any banded row lacks a licence; this is the "
              "licence state, not a band failure and not a finding"
              % (len(STAGES), len(blocks), T, digest[:16], len(lic.rows), lic_cred,
                 lic_uncredited, tally))
        return 1
    print("GATE PASS arms=3 stages=%d blocks=%d T=%d band_digest=%s device=none timing=taken=no "
          "banded_rows=%d credited=%d %s"
          % (len(STAGES), len(blocks), T, digest[:16], len(lic.rows), lic_cred, tally))
    return 0


def licence_tally(chk, lic, scope, derived, counts):
    """Count the licence citations this scope printed and refuse a short or empty population.

    The predicate is `cited == derived`, where `derived` is geometry arithmetic the caller declares
    (blocks x passes x stages x plans) before the rows are walked — never a count of what happened to
    be printed, which is the same thing and could not fail (`pitfalls/40` §an unnamed denominator)."""
    n, by = lic.tally(scope, derived)
    return chk.check("licence_cited_%s" % scope.replace(":", "_"),
                     n == derived and n > 0,
                     "rows=%d derived=%d licensed=%d form_unprobed=%d unlicensed=%d "
                     "note=every banded row must carry its own LICENCE line"
                     % (n, derived, by["LICENSED"], by["LICENSED_FORM_UNPROBED"],
                        by["UNLICENSED"]))


def licence_independent_expect(lic, pass_keys, pair=LICENCE_PAIR):
    """A SECOND derivation of how many of a set of rows cannot be licensed, from the dispatch table
    rather than from the licence file: `probe_arms_for` names the arms that exist at a pass, and a
    pair can only be licensed where both arms were run there. The licence table and this must agree
    without either being read back off the other, which is what makes the count a check."""
    exp = 0
    for pk in pass_keys:
        kind = "decode" if pk.split("/")[-1].startswith("decode") else "prefill"
        if pair[0] not in G.probe_arms_for(kind, "pinned"):
            exp += 1
    return exp


def _require_three_arms(arms):
    if len(arms) < 3:
        raise Refuse("three-arm agreement needs three arms; found %d (%s)"
                     % (len(arms), ",".join(arms)))


def pair_bytes(a_res, b_res, key):
    """Byte-difference count between two arm outputs, at zero tolerance."""
    ba = np.fromfile(a_res[key], dtype=np.uint8)
    bb = np.fromfile(b_res[key], dtype=np.uint8)
    if ba.size != bb.size:
        raise Refuse("%s sizes differ (%d vs %d)" % (key, ba.size, bb.size))
    return int((ba != bb).sum()), int(ba.size)


def carry_material(recs, blocks):
    """The pinned state-carry chain, block-major: for each block, decode0, decode1, decode2 — each a
    single cached step whose incoming state is the state the previous pass wrote. Both the step's
    inputs and its predecessor state are pinned bytes, so the live state is a dumped tensor and not a
    fixture this driver invented.

    The premise is CHECKED, not assumed: a case whose predecessor state is all zero would be a
    slice-1 case wearing a decode name, and the whole point of the section is a state that is live."""
    cases, states, nz = [], [], []
    for bi, b in enumerate(blocks):
        for p in DEC_PASSES:
            cases.append(dict(block=b, name=p, T=1, mode="honest", wslot=bi, carry=-1,
                              fields=field_words(recs, b, 1, p)))
            t = dump_tensor(recs, b, PRED[p], "state")
            states.append(words_of(t.reshape(-1)))
            nz.append(int((t.reshape(-1) != 0).sum()))
    return cases, states, nz


def pad_witness_run(chk, pad, runs, wts, chunk_width, counts):
    """D-21's precondition, run: the arm-internal pad control over every plan whose rows are judged.

    `runs` is a list of dicts with label, kind ("prefill" | "decode"), plan, inputs, res (the honest
    arm's result over the SAME bytes), cs (the honest arm's parsed CASE lines), T and produces_rows.
    For each, `build/host_probe_pad` — the same source with `-DGDN_BROKEN_PAD`, one pad row of every
    head carrying data instead of the zeros §04b.2 mandates — is run over those bytes and the EMITTED
    output is compared word for word.

    Four checks, each with a derived denominator:
      `pad_witness_fires`                   the control's self-test failed AND every plan that
                                            produces a banded decode row moved emitted bytes;
      `pad_witness_confined`                per (plan, case, stage): bytes moved exactly on the set the
                                            oracle's own dataflow says a pad row can reach and NOWHERE
                                            else — `downstream_closure(("state",))` inside a case that
                                            reads its own incoming state from the file, and the whole
                                            `STATE_AFFECTED` set for a case whose incoming state is the
                                            previous case's poisoned `state`. Both directions are
                                            asserted, so a control that perturbs everything fails it as
                                            hard as one that perturbs nothing;
      `pad_witness_visible_at_zero_state`   the polarity, stated and tested: unlike the state-read
                                            control this one must be VISIBLE with a zero incoming
                                            state, because §04b.2's padding is a property of the chunk
                                            and not of the state;
      `pad_census_moves_under_the_control`  the arm's own `pad_nonzero` counter — printed as a fact
                                            since slice 3 and never counted, because no mutation could
                                            reach it — is 0 on every honest case and nonzero on every
                                            case under the control. That is what turns it from a note
                                            into a counter.
    Nothing here reads a timer: every number printed counts bytes or words.
    """
    if not PAD.exists():
        pad.selftest = None
        chk.check("pad_witness_present", False,
                  "build=%s exists=0 note=with no arm-internal pad switch no mutation can reach a "
                  "pad word, so the bridging claim is untestable and no decode row may be credited "
                  "under the bridge pair; that is D-21's named fallback, CANNOT_BE_ESTABLISHED, and "
                  "the fix is ./build.sh --broken rather than a row credited on an untested claim"
                  % PAD)
        return
    st = subprocess.run([str(PAD), "--selftest"], capture_output=True, text=True)
    pad.selftest = st.returncode != 0
    tail = ((st.stdout + st.stderr).strip().splitlines() or ["<silent>"])[-1]
    print("SABOTAGE build=host_probe_pad define=GDN_BROKEN_PAD selftest=%s tail=%s "
          "note=one pad row of every head carries data instead of §04b.2's zeros; the honest build "
          "never defines it, and its STAGE lines are the same 178 bytes for bytes"
          % ("FIRE" if pad.selftest else "SILENT", tail[:96]))
    # must-catch sets, DERIVED from the oracle's dataflow table and not typed: a pad row enters the
    # emitted stages through the state write's accumulation over all C rows (unmasked) and the carry
    # multiplier exp(cum[C-1]), both of which read a pad slot as data; the intra-chunk output is
    # protected because §04b.3 masks `intra_attn` above the diagonal BEFORE the exp, so a pad column
    # can never reach a real row's output. A carried case additionally reads the poisoned state, and
    # from there the whole scan closure is reachable.
    pad_reaches_own = tuple(G.downstream_closure(("state",)))
    pad_reaches_carried = STATE_AFFECTED
    if not pad_reaches_own or not set(pad_reaches_own) <= set(pad_reaches_carried):
        raise Refuse("the oracle's dataflow says a pad row reaches %s inside its own case and %s for "
                     "a carried one, which is not a subset: the pad control's must-catch set cannot "
                     "be derived from STAGE_CONSUMERS as written"
                     % (",".join(pad_reaches_own), ",".join(pad_reaches_carried)))
    moved_rows, should_move_did_not, should_not_move_did = 0, 0, 0
    honest_pad_cases, control_pad_cases, cases_with_pads = 0, 0, 0
    for r in runs:
        res = run_arm(PAD, "host_twin", r["plan"], r["inputs"], wts, "pad_%s" % r["label"])
        cs, _, run = parse_arm(res, r["res"]["n_cases"], r["T"])
        if [c["out_hash"] for c in cs] == [c["out_hash"] for c in r["cs"]]:
            pass   # identical hashes are what `ndiff_bytes` below measures; nothing is assumed here
        nd, nb = pair_bytes(r["res"], res, "out")
        pad.add_plan(r["label"], r["kind"], r["T"], r["T"], chunk_width, nb, nd, r["produces_rows"])
        for i in range(len(cs)):
            # `carry[i] >= 0` is the established meaning of "carried" in this gate: plan B builds
            # carry_idx with -1 for a first-of-block case (state read from file) and the previous
            # case index otherwise, and its own guard counts `sum(1 for v in carry_idx if v >= 0)`.
            # bool(-1) is True, so a truthiness test here marked the two first-of-block cases as
            # carried and judged them against the WIDER carried must-catch set -- which is where
            # should_move_but_did_not came from. Note the trap in the obvious fix: plan A below
            # passes a constant per case, so a bare `>= 0` test would silently mark every
            # non-carried case as carried. Both sides are corrected together.
            carried = (int(r["carry"][i]) >= 0) if r["carry"] else False
            must = pad_reaches_carried if carried else pad_reaches_own
            moved = {}
            for st_ in STAGES:
                moved[st_] = int((stage_tensor(res, i, r["T"], st_).reshape(-1)
                                  != stage_tensor(r["res"], i, r["T"], st_).reshape(-1)).sum())
                if st_ in must:
                    moved_rows += 1
                    if moved[st_] == 0:
                        should_move_did_not += 1
                elif moved[st_] != 0:
                    should_not_move_did += 1
            if int(cs[i]["pad_nonzero"]) > 0:
                control_pad_cases += 1
            if int(r["cs"][i]["pad_nonzero"]) != 0:
                honest_pad_cases += 1
            if r["T"] < chunk_width:
                cases_with_pads += 1
            print("SABOTAGE_PAD plan=%s case=%d T=%d pad_cols=%d carried=%d must_move=%s %s "
                  "pad_nonzero_honest=%s pad_nonzero_control=%s plan_ndiff_bytes=%d verdict=%s"
                  % (r["label"], i, r["T"], chunk_width - r["T"], int(carried),
                     ",".join(must),
                     " ".join("moved[%s]=%d" % (s, moved[s]) for s in STAGES),
                     r["cs"][i]["pad_nonzero"], cs[i]["pad_nonzero"], nd,
                     "VISIBLE" if nd else "INVISIBLE"))
    chk.check("pad_witness_fires",
              bool(pad.selftest) and any(p["produces_rows"] for p in pad.plans)
              and all(p["ndiff"] > 0 for p in pad.plans if p["produces_rows"]),
              "selftest=%s %s" % ("FIRE" if pad.selftest else "SILENT", pad.describe()))
    chk.check("pad_witness_confined",
              moved_rows > 0 and should_move_did_not == 0 and should_not_move_did == 0,
              "must_catch_rows=%d should_move_but_did_not=%d moved_though_upstream=%d "
              "derived_from=gdn_ref.downstream_closure+STAGE_CONSUMERS note=must_move is (%s) inside "
              "a case that reads its own incoming state and (%s) for a case whose incoming state is "
              "the previous case's emitted state; every other stage must be byte-identical, which is "
              "what makes this a pad control and not a blanket perturbation"
              % (moved_rows, should_move_did_not, should_not_move_did,
                 ",".join(pad_reaches_own), ",".join(pad_reaches_carried)))
    zero = [p for p in pad.plans if p["kind"] == "prefill"]
    chk.check("pad_witness_visible_at_zero_state",
              bool(zero) and all(p["ndiff"] > 0 for p in zero),
              "zero_state_plans=%s ndiff_bytes=%s verdict=%s note=the state-read control reads "
              "INVISIBLE_AT_ZERO_STATE and this one must NOT: §04b.2's padding is a property of the "
              "chunk, so a pad row that carries data moves the emitted state with no state to carry"
              % (",".join(p["label"] for p in zero) or "none",
                 ",".join(str(p["ndiff"]) for p in zero) or "none",
                 "VISIBLE_AT_ZERO_STATE" if zero and all(p["ndiff"] > 0 for p in zero)
                 else "INVISIBLE_AT_ZERO_STATE"))
    chk.check("pad_census_moves_under_the_control",
              honest_pad_cases == 0 and control_pad_cases == cases_with_pads and cases_with_pads > 0,
              "honest_cases_with_nonzero_pads=%d control_cases_with_nonzero_pads=%d "
              "cases_with_pad_cols=%d note=the arm's own GDN_ST_PAD_NONZERO counter is the thing "
              "slice 3 could print and not count because no driver-side mutation could move it; under "
              "the arm-internal define it leaves zero on every case that has pad columns, which is "
              "what makes a pad census a measurement and not a constant"
              % (honest_pad_cases, control_pad_cases, cases_with_pads))
    counts["pad_control_plans"] = len(pad.plans)
    counts["pad_control_must_catch"] = moved_rows
    counts["pad_control_cases"] = cases_with_pads


# The state a pad row reaches inside the case that synthesised it, and the state a pad row reaches
# once that case's emitted `state` has been carried into the next one. Both are read out of the
# oracle's dataflow table (`speccheck/gdn_ref.py STAGE_CONSUMERS`) by `pad_witness_run`; the names
# live here so the crediting rule and the witness cannot drift apart.
PAD_REACHES_OWN = tuple(G.downstream_closure(("state",)))
PAD_REACHES_CARRIED = STATE_AFFECTED


def carry_section(band, g, recs, blocks, T, wslots, plan, ins, wts, stage_lines, fails, counts,
                  lic, chk, pad):
    """Slice 2's own question: does a LIVE state survive the boundary, and can anything here tell?

    TWO plans over the same six cases, so the only difference between them is where a case's incoming
    state comes from:
      pinned_state — each case reads the PINNED state its predecessor pass produced;
      arm_carried  — cases 1 and 2 of each block read the state the PREVIOUS CASE of the same block
                     emitted in this same arm run (`carry=` in the plan), which is the recurrence
                     rather than a re-read of the reference.
    Then the sabotage: `build/host_probe_xpose` is the same source with `gdn_state_dot` reading the
    state as `S^T` — the oracle's `state_stored_transposed`. It must be INVISIBLE on the zero-state
    prefill plan (that is the proof slice 1 could not have caught it) and RED on both live plans.
    """
    if not XPOSE.exists():
        raise Refuse("%s is not built; run ./build.sh --broken — the state-read sabotage is not a "
                     "check this lane may claim without its witness" % XPOSE)
    cases, states, nz = carry_material(recs, blocks)
    n = len(cases)
    DEC_T = 1
    # ---- the premise: every case here has a non-zero incoming state, and the count is per case
    for i, c in enumerate(cases):
        if nz[i] == 0:
            raise Refuse("%s/%s: the pinned incoming state (%s.state) is all zero, so this case "
                         "would not test a live state" % (c["block"], c["name"], PRED[c["name"]]))
    print("CARRY plan=pinned_state cases=%d blocks=%s dec_T=%d state_words=%d nonzero_per_case=%s "
          "source=state_of_previous_pinned_pass"
          % (n, ",".join(blocks), DEC_T, STATE_WORDS, ",".join(str(v) for v in nz)))

    planA = WORK / "plan_state.tsv"
    write_plan(planA, cases)
    insA = WORK / "inputs_state.bin"
    write_inputs(insA, [c["fields"] for c in cases], DEC_T, states=states)

    # ---- plan B: the same cases, but the state comes from the previous case's own output
    carry_idx = []
    for bi in range(len(blocks)):
        for j in range(len(DEC_PASSES)):
            carry_idx.append(-1 if j == 0 else bi * len(DEC_PASSES) + j - 1)
    casesB = [dict(c) for c in cases]
    for i, c in enumerate(casesB):
        c["carry"] = carry_idx[i]
    planB = WORK / "plan_carry.tsv"
    write_plan(planB, casesB)

    # ---- the honest arms on both plans, and the compiled pair on the carried one
    resA = run_arm(HOST, "host_twin", planA, insA, wts, "state")
    csA, _, runA = parse_arm(resA, n, DEC_T)
    resB = run_arm(HOST, "host_twin", planB, insA, wts, "carry")
    csB, _, runB = parse_arm(resB, n, DEC_T)
    resBh = run_arm(DEVI, "hip_hostpath", planB, insA, wts, "carry")
    csh, _, runh = parse_arm(resBh, n, DEC_T)
    if runh.get("arm") != "hip_hostpath" or runh.get("hip_calls") != "0":
        fails.append("the carried-plan device arm announced arm=%s hip_calls=%s"
                     % (runh.get("arm"), runh.get("hip_calls")))
    if int(runB.get("carried", "0")) != sum(1 for v in carry_idx if v >= 0):
        fails.append("the carried plan reported carried=%s, the plan declares %d carry cases"
                     % (runB.get("carried"), sum(1 for v in carry_idx if v >= 0)))
    for key, label in (("out", "out.bin"), ("echo", "echo.bin")):
        nd, nb = pair_bytes(resB, resBh, key)
        print("PAIR pair=host_twin_vs_hip_hostpath plan=arm_carried file=%s bytes=%d ndiff_bytes=%d "
              "cases=%d zero_tolerance=1 verdict=%s"
              % (label, nb, nd, n, "IDENTICAL" if nd == 0 else "DIFFERS"))
        if nd:
            fails.append("compiled pair differs on %d bytes of %s on the carried plan: one source, "
                         "two compilers, so this is a codegen finding" % (nd, label))

    # ---- the arm's own count of the live state it read, against the count the dump implies
    bad = 0
    for i, d in enumerate(csA):
        if int(d["state_nonzero"]) != nz[i] or int(d["carry"]) != -1:
            bad += 1
    print("EXACT plan=pinned_state field=init_state checks=%d agree=%d derived=nonzero_count_of_%s"
          % (n, n - bad, "pinned_predecessor_state"))
    if bad:
        fails.append("%d case(s) of the pinned_state plan read a state whose non-zero count is not "
                     "the pinned tensor's: the state block was read at the wrong offset, or not read"
                     % bad)
    bad = 0
    carried_nz = []
    for i, d in enumerate(csB):
        src = carry_idx[i]
        want = nz[i] if src < 0 else int((stage_tensor(resB, src, DEC_T, "state") != 0).sum())
        carried_nz.append(want)
        if int(d["state_nonzero"]) != want or int(d["carry"]) != src:
            bad += 1
    if min(carried_nz) == 0:
        fails.append("a carried case received an all-zero state: the carry produced nothing, so the "
                     "section would be reporting zero-state numbers under a live-state name")
    print("EXACT plan=arm_carried field=init_state checks=%d agree=%d carried=%d "
          "carried_nonzero_min=%d note=carried counts are read back out of the arm's own out.bin"
          % (n, n - bad, sum(1 for v in carry_idx if v >= 0), min(carried_nz)))
    if bad:
        fails.append("%d case(s) of the carried plan disagree about the state they received "
                     "(count or carry index)" % bad)

    # ---- the banded verdicts, both plans, all seven stages. A decode row is judged under the bridge
    # pair (`recurrent|pinned`) — the licence that exists for the program §04b.1 dispatches at a
    # cached step — and every other row under `chunked|pinned`.
    BRIDGE = bridge_pair()

    def judge(res, tag, plan_label):
        rows, ok_rows, rejected = 0, 0, set()
        for i, c in enumerate(cases):
            for st in STAGES:
                mine = stage_tensor(res, i, DEC_T, st)
                ref = dump_tensor(recs, c["block"], c["name"], st)
                ok, _ = band_row(band, tag, c["block"], c["name"], st, mine, ref, stage_lines,
                                 fails, lambda b, p, s, w: "%s/%s/%s outside the derived band: %s"
                                                        % (tag, b, p, w))
                lic.cite(c["block"], c["name"], st, plan=plan_label, pair=BRIDGE)
                rows += 1
                if ok:
                    ok_rows += 1
                else:
                    rejected.add(st)
        return rows, ok_rows, rejected

    # The claim under test is "a C=64 chunked kernel with 63 zero pads implements the recurrent
    # single-token step", and the only mutation that can reach a pad word is the arm's own
    # `-DGDN_BROKEN_PAD` (`gdn_kernels.h::gdn_item_enter`, `gdn_kernels.h::gdn_item_cum`), because
    # the pads are synthesised inside the item entry and never pass through anything a driver writes.
    # The three plans below are the three ways this lane produces emitted bytes: the zero-state
    # prefill (where the state-read control is provably blind), the non-carried decode, and the
    # carried decode. THIS CALL MUST PRECEDE THE JUDGING LOOP: Licence.classify asks the witness
    # whether the pad control fired, and a witness that fires after `judge` can credit nothing.
    # The honest prefill baseline is produced here for the same reason: the zero-state pad plan and
    # the xpose comparison below must both be measured against THIS run, not two separate copies of
    # what is meant to be one honest reference.
    h0 = run_arm(HOST, "host_twin", plan, ins, wts, "xzero")
    cs0, _, _ = parse_arm(h0, len(blocks), T)
    pad_witness_run(chk, pad, [
        dict(label="zero_state_prefill", kind="prefill", plan=plan, inputs=ins, res=h0, cs=cs0,
             T=T, produces_rows=False, carry=None),
        dict(label=CARRY_PLAN_LABELS[0], kind="decode", plan=planA, inputs=insA, res=resA, cs=csA,
             T=DEC_T, produces_rows=True, carry=[-1] * n),
        dict(label=CARRY_PLAN_LABELS[1], kind="decode", plan=planB, inputs=insA, res=resB, cs=csB,
             T=DEC_T, produces_rows=True, carry=carry_idx),
    ], wts, g.chunk_width, counts)

    mark0 = len(lic.rows)
    sl0 = len(stage_lines)
    verdicts = []
    for res, tag, plan_label in ((resA, "kernel_vs_dump", CARRY_PLAN_LABELS[0]),
                                 (resB, "kernel_carried_vs_dump", CARRY_PLAN_LABELS[1])):
        rows, ok_rows, rejected = judge(res, tag, plan_label)
        if rows != n * len(STAGES):
            fails.append("%s plan produced %d stage rows, derived denominator is %d"
                         % (plan_label, rows, n * len(STAGES)))
        n_cite, by = lic.tally("carry:" + plan_label, n * len(STAGES))
        cred = by["LICENSED"] + by["LICENSED_BRIDGED"]
        chk.check("licence_cited_carry_%s" % plan_label,
                  n_cite == rows and n_cite > 0,
                  "rows=%d derived=%d credited=%d unlicensed=%d note=the band accepted %d of "
                  "them; crediting is the licence's business, not the band's"
                  % (n_cite, rows, cred, by["UNLICENSED"], ok_rows))
        print("CARRYVERDICT plan=%s rows=%d accepted=%d rejected=%d derived=%d "
              "rejected_stages=%s credited=%d unlicensed=%d verdict=%s"
              % (plan_label, rows, ok_rows, rows - ok_rows, n * len(STAGES),
                 ",".join(sorted(rejected)) or "none", cred, by["UNLICENSED"],
                 "CREDITED" if cred == rows else
                 ("UNLICENSED" if cred == 0 else "PARTLY_LICENSED")))
        verdicts.append((rows, ok_rows, rejected))
    rowsA, okA, rejA = verdicts[0]
    rowsB, okB, rejB = verdicts[1]
    # ---- D-21: the CREDITED count, derived a second and independent way, with the pad witness as a
    # precondition on it. Reading (a) is the count the citations produced, through `band_licenses`;
    # reading (b) rebuilds the population from the run's own geometry (blocks x decode passes x stages
    # x 2 plans), asks the §04b.1 dispatch table whether the licence arm runs at each pass and the band
    # FILE — through a second JSON reader, not through the dict `load_band` built — whether that row is
    # licensed. Neither reading can move without the other noticing, and the predicate that decided
    # each row is `Licence.classify`, the same one the witnesses below mutate.
    raw_licensed = licensed_rows_from_file()
    population = [("%s/pinned/%s" % (b, p), st)
                  for b in blocks for p in DEC_PASSES for st in STAGES]

    def bridge_expect(pop, licensed_map=None):
        lmap = raw_licensed if licensed_map is None else licensed_map
        return sum(1 for pk, st in pop
                   if BRIDGE[0] in G.probe_arms_for(
                          "decode" if pk.split("/")[-1].startswith("decode") else "prefill",
                          "pinned")
                   and G.cross_form_key(pk, *BRIDGE) in lmap.get(st, ()))

    lic_carry = lic.rows[mark0:]
    expected_cred = bridge_expect(population) * len(CARRY_PLAN_LABELS)
    got_cred = sum(r["credited"] for r in lic_carry)
    got_chunked = sum(1 for r in lic_carry if r["verdict"] == "LICENSED")
    lic_unlic = sum(1 for r in lic_carry if r["verdict"] == "UNLICENSED")
    lic_unarm = sum(1 for r in lic_carry if r["verdict"] == "UNLICENSED_ARM_NOT_DISPATCHED")
    # The unlicensed population, derived a SECOND way: `probe_arms_for` names the arms that exist at a
    # pass (the §04b.1 dispatch table), and the band file names the licensed rows. A row is unlicensed
    # exactly when one of the two refuses it, and the two terms are disjoint — a row whose licence arm
    # does not run at its pass is never looked up in the file — so the sum is the whole population.
    lic_expected_unarm = licence_independent_expect(lic, [r["pass_key"] for r in lic_carry], BRIDGE)
    lic_expected_unfile = len(lic_carry) - expected_cred
    chk.check("carry_rows_unlicensed_eq_dispatch",
              len(lic_carry) == 2 * n * len(STAGES)
              and lic_unarm == lic_expected_unarm
              and lic_unlic == lic_expected_unarm + lic_expected_unfile,
              "unlicensed=%d derived=%d+%d unlicensed_arm_not_dispatched=%d "
              "derived_from_dispatch_table=%d rows=%d derived_rows=%d pair=%s "
              "derived_from=gdn_ref.probe_arms_for+band_file_json note=the dispatch table gives a "
              "cached decode step the recurrent program, so the bridge arm runs at every decode pass "
              "and every unlicensed row here comes from the file, not from the dispatch"
              % (lic_unlic, lic_expected_unarm, lic_expected_unfile, lic_unarm,
                 lic_expected_unarm, len(lic_carry), 2 * n * len(STAGES), "|".join(BRIDGE)))
    chk.check("carry_rows_credited_eq_licence_file",
              len(lic_carry) == len(population) * len(CARRY_PLAN_LABELS)
              and got_cred == expected_cred and expected_cred > 0,
              "credited=%d derived_from=geometry+gdn_ref.probe_arms_for+band_file_json=%d "
              "rows=%d derived_rows=%d pair=%s pad_witness=%s note=the second reading never calls "
              "band_licenses, so a licence census that agreed with itself and not with the file "
              "would be caught here"
              % (got_cred, expected_cred, len(lic_carry),
                 len(population) * len(CARRY_PLAN_LABELS), "|".join(BRIDGE), pad.state))
    chk.check("carry_credit_requires_firing_pad_witness",
              got_cred == 0 or (pad.fires_on_decode_rows() and pad.selftest is True),
              "credited=%d pad_witness=%s pad_selftest=%s note=a row is credited under the bridge "
              "pair ONLY while the arm-internal pad control is present and firing; with the "
              "witness withdrawn the rows go back to uncredited, which is the check the task "
              "requires and the witness below proves it is load-bearing"
              % (got_cred, pad.state, pad.selftest))
    sl1 = len(stage_lines)
    chk.check("carry_rows_are_printed_not_deleted",
              sum(1 for ln in stage_lines[sl0:sl1] if ln.startswith("STAGE "))
              == 2 * n * len(STAGES) and got_chunked == 0,
              "stage_lines=%d derived=%d credited_under_chunked_pinned=%d uncredited=%d "
              "note=the 84 rows are still measured and still printed with their band verdicts; the "
              "only thing D-21 changed is who may be credited, and `chunked|pinned` still licenses "
              "no decode row"
              % (sl1 - sl0, 2 * n * len(STAGES), got_chunked, len(lic_carry) - got_cred))
    if lic_unlic or len(lic_carry) != got_cred:
        print("LICENCEVERDICT scope=carry rows=%d credited=%d uncredited=%d unlicensed=%d "
              "licensed_form_unprobed=%d cannot_be_established=%d pad_witness=%s verdict=QUALIFIED "
              "note=band verdicts stand, acceptances are NOT deleted, and no row is credited without "
              "a licence row and a firing pad witness behind it"
              % (len(lic_carry), got_cred, len(lic_carry) - got_cred, lic_unlic,
                 by["LICENSED_FORM_UNPROBED"], by["CANNOT_BE_ESTABLISHED"], pad.state))

    # ---- witnesses that the licence census is a census and not a decoration.
    #  (a) a licence table that silently grants the pair at a decode pass must be NOTICEABLE: the
    #      same classification over the same rows, run against an in-memory copy of the band with
    #      one such row added, has to disagree with the dispatch-table derivation. The band FILE is
    #      never touched, and `load_band` would refuse such an edit on disk anyway.
    poison_band = json.loads(json.dumps(band))
    pstage = STAGES[0]
    pkey = "%s/pinned/%s" % (blocks[0], DEC_PASSES[0])
    poison_band[pstage]["licensed_rows"] = sorted(
        set(poison_band[pstage].get("licensed_rows", []))
        | {G.cross_form_key(pkey, *LICENCE_PAIR)})
    plic = Licence(poison_band, g, verbose=False)
    for c0 in cases:
        for st in STAGES:
            plic.cite(c0["block"], c0["name"], st, plan="licence_sabotage_poisoned_band")
    p_now_licensed = sum(1 for r in plic.rows if r["verdict"] != "UNLICENSED")
    p_expected_unlicensed = licence_independent_expect(lic, [r["pass_key"] for r in plic.rows])
    chk.witness("licence_census_fires_on_a_poisoned_table",
                p_now_licensed > 0 and (len(plic.rows) - p_now_licensed) != p_expected_unlicensed,
                "sabotage=in_memory_band_row row=%s|%s|%s stage=%s now_licensed=%d "
                "unlicensed_vs_dispatch_derived=%d/%d fired=%d note=a band that grew a licence for a "
                "pair no probe record supports would flip these rows and this census is what would "
                "see it"
                % (pkey, LICENCE_PAIR[0], LICENCE_PAIR[1], pstage, p_now_licensed,
                   len(plic.rows) - p_now_licensed, p_expected_unlicensed,
                   int(p_now_licensed > 0)))
    #  (b) the same predicate over a row set with one row dropped must refuse: an acceptance list
    #      that silently loses a row is the empty-parse failure mode in a smaller hat.
    short = lic_carry[:-1]
    short_expected = licence_independent_expect(lic, [r["pass_key"] for r in short])
    pred_short = (len(short) == len(lic_carry)
                  and sum(1 for r in short if r["verdict"] == "UNLICENSED") == short_expected)
    chk.witness("licence_census_fires_on_a_dropped_row", not pred_short,
                "sabotage=one_of_%d_rows_dropped rows=%d derived=%d predicate=%s fired=1 "
                "note=the predicate the honest run above passes must be the one that fails here"
                % (len(lic_carry), len(short), len(lic_carry), pred_short))

    # ---- D-21's witnesses: the crediting rule has to be able to refuse, and to be able to grant.
    #  (c) the required one — a row credited while the pad-equivalence witness is absent must be
    #      impossible, so the SAME classification rows, run through a `Licence` with no witness, has
    #      to credit nothing. This is not a re-reading of the honest tally: `classify` is called
    #      again, on the same pass keys, stages and pair, against a witness that is not there.
    nowitness = Licence(band, g, verbose=False, pad=None)
    cred_without = sum(nowitness.classify(r["pass_key"], r["stage"], BRIDGE)["credited"]
                       for r in lic_carry)
    chk.check("credit_refused_without_pad_witness",
              expected_cred > 0 and cred_without == 0,
              "credited_with_witness=%d credited_without_witness=%d derived_from=the_same_classify_"
              "over_the_same_rows note=if this can only ever see 0==0 it is decoration; the witness "
              "directly below is what keeps it honest"
              % (got_cred, cred_without))
    chk.witness("credit_fires_when_the_witness_is_present", got_cred > 0 and cred_without == 0,
                "sabotage=withdraw_the_pad_witness credited=%d without=%d fired=%d note=a refusal "
                "that refuses in both worlds is not a refusal, it is a constant"
                % (got_cred, cred_without, int(got_cred > 0 and cred_without == 0)))
    #  (d) the crediting path must actually read the licence table: revoke the first credited row in
    #      an in-memory copy of the band and the SAME predicate must stop crediting it.
    first_cred = next((r for r in lic_carry if r["credited"]), None)
    if first_cred is None:
        chk.witness("licence_credit_fires_on_a_revoked_row", False,
                    "sabotage=revoke_a_licensed_row skipped=1 note=no row was credited, so there is "
                    "no crediting path to mutate — which is the same as the licence table never "
                    "having been read")
    else:
        rkey = G.cross_form_key(first_cred["pass_key"], *BRIDGE)
        revoke_band = json.loads(json.dumps(band))
        revoke_band[first_cred["stage"]]["licensed_rows"] = [
            k for k in revoke_band[first_cred["stage"]].get("licensed_rows", []) if k != rkey]
        rc = Licence(revoke_band, g, verbose=False, pad=pad).classify(
            first_cred["pass_key"], first_cred["stage"], BRIDGE)
        chk.witness("licence_credit_fires_on_a_revoked_row",
                    rc["credited"] == 0 and rc["verdict"] == "UNLICENSED",
                    "sabotage=revoke_a_licensed_row row=%s stage=%s pair=%s honest_verdict="
                    "LICENSED_BRIDGED revoked_verdict=%s fired=%d note=the licence is read through "
                    "gdn_ref.band_licenses per row, so a table that stopped granting must change the "
                    "verdict and not merely the printout" % (rkey, first_cred["stage"],
                                                              "|".join(BRIDGE), rc["verdict"],
                                                              int(rc["credited"] == 0)))
        #  (e) and the SECOND reader must be a reader: the same revocation, applied to the raw JSON
        #      map, has to move the independently derived expected count by exactly one plan's worth.
        raw_revoked = {st_: (ks - {rkey} if st_ == first_cred["stage"] else set(ks))
                       for st_, ks in raw_licensed.items()}
        exp_revoked = bridge_expect(population, raw_revoked) * len(CARRY_PLAN_LABELS)
        chk.witness("licence_population_fires_on_a_revoked_row",
                    exp_revoked == expected_cred - len(CARRY_PLAN_LABELS),
                    "sabotage=revoke_the_same_row_in_the_raw_json expected=%d revoked=%d "
                    "delta=%d fired=%d note=the reconciliation has two readers precisely so that one "
                    "of them can be shown to be listening; this is that demonstration"
                    % (expected_cred, exp_revoked, expected_cred - exp_revoked,
                       int(exp_revoked == expected_cred - len(CARRY_PLAN_LABELS))))

    # ---- the sabotage, witnessed on BOTH kinds of plan
    st = subprocess.run([str(XPOSE), "--selftest"], capture_output=True, text=True)
    xpose_selftest = st.returncode != 0
    rx0 = run_arm(XPOSE, "host_twin", plan, ins, wts, "xpose_zero")
    parse_arm(rx0, len(blocks), T)
    # h0 and cs0 come from above, before the pad witness, so that the zero-state pad plan and this
    # xpose comparison share one honest reference run.
    nd0, nb0 = pair_bytes(h0, rx0, "out")
    ndA, nbA = (None, 0)
    rxA = run_arm(XPOSE, "host_twin", planA, insA, wts, "xpose_state")
    parse_arm(rxA, n, DEC_T)
    ndA, nbA = pair_bytes(resA, rxA, "out")
    rxB = run_arm(XPOSE, "host_twin", planB, insA, wts, "xpose_carry")
    parse_arm(rxB, n, DEC_T)
    ndB, nbB = pair_bytes(resB, rxB, "out")
    print("SABOTAGE build=host_probe_xpose define=GDN_BROKEN_STATE_TRANSPOSE selftest=%s"
          % ("FIRE" if xpose_selftest else "SILENT"))
    print("SABOTAGE plan=zero_state_prefill file=out.bin bytes=%d ndiff_bytes=%d "
          "verdict=%s note=at a zero state the honest read and the S^T read are the SAME NUMBER, so "
          "slice 1 had no way to see this defect"
          % (nb0, nd0, "INVISIBLE_AT_ZERO_STATE" if nd0 == 0 else "UNEXPECTED"))
    if nd0 != 0:
        fails.append("the state-read sabotage moves %d bytes of a ZERO-state run: the control is "
                     "not confined to the state read, so its reds would prove nothing" % nd0)
    if not xpose_selftest:
        fails.append("the state-read sabotage build passed the self-test: the state-read check in "
                     "the self-test cannot fail")
    for nd, nb, label in ((ndA, nbA, "pinned_state"), (ndB, nbB, "arm_carried")):
        print("SABOTAGE plan=%s file=out.bin bytes=%d ndiff_bytes=%d verdict=%s"
              % (label, nb, nd, "VISIBLE" if nd else "INVISIBLE"))
        if nd == 0:
            fails.append("the state-read sabotage is indistinguishable from the honest arm on the "
                         "%s plan: the live-state case does not actually exercise the state read"
                         % label)
    counts["xpose_selftest_fired"] = 1 if xpose_selftest else 0

    # ---- what the sabotage must move and must not move, per stage, on both live plans
    must_red, x_rows, x_rejected = 0, 0, 0
    for res_x, res_h, label in ((rxA, resA, "pinned_state"), (rxB, resB, "arm_carried")):
        for i, c in enumerate(cases):
            for st_ in STAGES:
                mx = stage_tensor(res_x, i, DEC_T, st_)
                mh = stage_tensor(res_h, i, DEC_T, st_)
                moved = int((mx.reshape(-1) != mh.reshape(-1)).sum())
                should = st_ in STATE_AFFECTED
                if should and moved == 0:
                    fails.append("%s/%s/%s did not move under the state-read sabotage, but the "
                                 "oracle's dataflow closure says the state read reaches it"
                                 % (label, c["block"], c["name"]))
                if not should and moved != 0:
                    fails.append("%s/%s/%s moved under the state-read sabotage although it is "
                                 "upstream of the state read: the control is not confined"
                                 % (label, c["block"], c["name"]))
                must_red += 1 if should else 0
                if should:
                    # the polarity here is the OPPOSITE of an honest row: a row the sabotage leaves
                    # INSIDE the band is the finding, not a row it throws out.
                    ok_x, _ = band_row(band, "xpose_vs_dump", c["block"], c["name"], st_, mx,
                                       dump_tensor(recs, c["block"], c["name"], st_),
                                       stage_lines, fails)
                    x_rows += 1
                    x_rejected += 0 if ok_x else 1
                    if ok_x:
                        fails.append("state-read sabotage stayed INSIDE the band at %s/%s/%s: the "
                                     "band does not see the state read at this stage"
                                     % (label, c["block"], c["name"]))
    print("SABOTAGE_BAND rows=%d rejected=%d accepted=%d derived=%d verdict=%s"
          % (x_rows, x_rejected, x_rows - x_rejected, 2 * n * len(STATE_AFFECTED),
             "RED_ON_DERIVED_COUNT" if x_rows == x_rejected == 2 * n * len(STATE_AFFECTED)
             else "FAIL"))
    print("SABOTAGE_SET moved_stage_rows=%d derived=%d untouched_rows=%d "
          "note=q_hat and k_hat are byte-identical under the sabotage, by assertion"
          % (must_red, 2 * n * len(STATE_AFFECTED),
             2 * n * (len(STAGES) - len(STATE_AFFECTED))))
    if must_red != 2 * n * len(STATE_AFFECTED):
        fails.append("sabotage must-catch rows %d != derived %d" % (must_red,
                                                                    2 * n * len(STATE_AFFECTED)))

    # ---- relL2 and ndiff made load-bearing as an ORDERING against the sabotage, per live stage
    order_fail, order_rows = 0, 0
    for res_x, res_h, label in ((rxA, resA, "pinned_state"), (rxB, resB, "arm_carried")):
        for i, c in enumerate(cases):
            ref_st = {s: dump_tensor(recs, c["block"], c["name"], s) for s in STATE_AFFECTED}
            for st_ in STATE_AFFECTED:
                mh = metrics(stage_tensor(res_h, i, DEC_T, st_).reshape(ref_st[st_].shape),
                             ref_st[st_])
                mx = metrics(stage_tensor(res_x, i, DEC_T, st_).reshape(ref_st[st_].shape),
                             ref_st[st_])
                good = mh["relL2"] < mx["relL2"] and mh["ndiff"] < mx["ndiff"]
                order_fail += 0 if good else 1
                order_rows += 1
                if not good:
                    fails.append("%s/%s/%s: the honest arm is NOT closer to the pinned bytes than "
                                 "the transposed-state arm (relL2 %.3e vs %.3e, ndiff %d vs %d) — "
                                 "relL2 and ndiff are not seeing the state read"
                                 % (label, c["block"], c["name"], mh["relL2"], mx["relL2"],
                                    mh["ndiff"], mx["ndiff"]))
                if st_ == "state":
                    print("CARRYORDER plan=%s block=%s pass=%s stage=%s relL2_honest=%.6g "
                          "relL2_xpose=%.6g ratio=%.4g ndiff_honest=%d ndiff_xpose=%d n=%d "
                          "load_bearing=relL2,ndiff verdict=%s"
                          % (label, c["block"], c["name"], st_, mh["relL2"], mx["relL2"],
                             (mx["relL2"] / mh["relL2"]) if mh["relL2"] else float("inf"),
                             mh["ndiff"], mx["ndiff"], mh["n"], "OK" if good else "FAIL"))
    print("CARRYORDER rows=%d failures=%d derived=%d note=ordering, not a typed threshold; the "
          "absolute verdict is band_check_row's"
          % (order_rows, order_fail, 2 * n * len(STATE_AFFECTED)))
    counts["carry_rows"] = rowsA + rowsB
    counts["carry_xpose_rows"] = must_red

    # ---- D-21 step 1: the pad-equivalence witness, BEFORE any decode row can be credited.
    # The pad witness runs BEFORE the judging loop, not after it: Licence.classify consults the
    # witness, so a witness that fires after the verdict is a witness that credited nothing. See
    # just above `mark0`.
    counts["carry_order_rows"] = order_rows


def pinned_passes(recs, key="q_pre"):
    """The (block, pass) pairs the dump pins for one tensor key. `recs` also carries `('name', ...)`
    keys for the stored weights, so the record key is unpacked by length and not by hope."""
    return sorted({k[:2] for k in recs if len(k) == 3 and k[2] == key})


def chunk_census(recs, chunk_width, label, verify_T=None):
    """One CHUNKSRC line per pinned pass, and the chunk count DERIVED from the manifest's own field.

    The field is `dumps[].shape[1]` — the token axis of the pass's `q_pre` record — against
    `geometry.chunk_width`. When `verify_T` is given the manifest's T is also checked against the T
    of the tensor the dump reader actually handed back, so the census is anchored to bytes and not
    only to the manifest's claim about them."""
    pop = pinned_passes(recs)
    if not pop:
        raise Refuse("no pinned pass has a q_pre record: there is no pass to count chunks over")
    chunks = {}
    for (b, m) in pop:
        rec = recs[(b, m, "q_pre")]
        shape = [int(x) for x in rec["shape"]]
        t_man = shape[1]
        tb = ""
        if verify_T is not None:
            if verify_T[(b, m)] != t_man:
                raise Refuse("%s/%s: the manifest declares T=%d but the bytes read back T=%d"
                             % (b, m, t_man, verify_T[(b, m)]))
            tb = " T_bytes=%d" % verify_T[(b, m)]
        ch = -(-t_man // chunk_width)
        chunks[(b, m)] = ch
        print("CHUNKSRC label=%s block=%s pass=%s manifest_record=%s manifest_field=shape[1] "
              "shape=%s T_manifest=%d%s chunk_width=%d chunks=%d pad_cols=%d"
              % (label, b, m, rec.get("name"), shape, t_man, tb, chunk_width, ch,
                 ch * chunk_width - t_man))
    return pop, chunks


def multichunk_verdict(label, chunks, man, chunk_width):
    multi = sorted(k for k, v in chunks.items() if v > 1)
    inp = man.get("input", {})
    print("MULTICHUNK label=%s passes=%d max_chunks=%d multichunk_passes=%s "
          "manifest_fields=dumps[].shape[1],input.n_tokens=%s,input.n_decode_steps=%s,"
          "counts.passes_per_block=%s geometry.chunk_width=%d dispatch.decode=%r verdict=%s "
          "note=%s"
          % (label, len(chunks), max(chunks.values()),
             ",".join("%s/%s" % k for k in multi) or "none",
             inp.get("n_tokens"), inp.get("n_decode_steps"),
             man.get("counts", {}).get("passes_per_block"), chunk_width,
             man.get("dispatch", {}).get("decode"),
             "ADJUDICATION_REQUIRED" if multi else "NO_PINNED_MULTICHUNK_PASS",
             ("a pinned pass with T>chunk_width exists; a multi-chunk carry against pinned bytes "
              "must now be adjudicated and the arm cannot express it (one case = one chunk), so "
              "this lane would have to grow a chunk loop before crediting one"
              if multi else
              "no pinned pass exceeds one chunk, so a multi-chunk PREFILL carry has no pinned "
              "reference at any interior chunk boundary and no fixture is invented to make one; "
              "the carry mechanism itself is exercised at T/n_chunks below and across passes in "
              "`carry`")))
    return multi


def chunk_carry_section(band, man, recs, blocks, T, g, wslots, plan, ins, wts, lic, chk, fails,
                        counts, stage_lines):
    """Slice 3's two questions: is there pinned material for a multi-chunk prefill (no, and the
    manifest names the field that says so), and does a chunk N reading chunk N-1's EMITTED state
    reproduce pinned bytes (yes, at the only width the material allows).

    The split is `n_chunks = 2` because two is the smallest number of chunks that forces a carry,
    and the width is `T // n_chunks` because the split is equal and `T` is the manifest's own field
    — neither is typed. A chunk of `T/2` rows is a shape the engine only ever takes as the LAST
    chunk of a prefill, so these rows are labelled `LICENSED_FORM_UNPROBED`: the pair
    `chunked|pinned` IS licensed at a prefill pass, but no probe arm ran chunk width `T/2`, so
    `band_licenses` must not be read as crediting them. Measured, printed, banded, not credited.

    The pad census is printed as a FACT here and deliberately not counted as a check: the chunk-pad
    rows are synthesised inside the kernel (`gdn_item_enter` writes 0 for `i >= t`) and are not bytes
    the driver can reach, so no driver-side mutation can make it leave zero and a check on it would
    be a check that cannot fail (`pitfalls/40` §A check that can produce no output cannot fail).
    """
    # The split rows go into the run's shared STAGE list, which is what `determinism` re-runs and
    # compares, so these 20 rows are held to the same reproducibility claim as the other 158.
    split_from = len(stage_lines)
    split_lines = stage_lines
    C = g.chunk_width
    verify = {p: int(dump_tensor(recs, p[0], p[1], "q_pre").shape[1]) for p in pinned_passes(recs)}
    pop, chunks = chunk_census(recs, C, "manifest", verify_T=verify)
    derived_passes = len(blocks) * int(man.get("counts", {}).get("passes_per_block", 0))
    chk.check("chunk_census_covers_pinned_passes",
              len(pop) == derived_passes and derived_passes > 0,
              "passes_counted=%d derived_from_manifest_counts=%d note=one CHUNKSRC line per "
              "(block,pass) that has a q_pre record" % (len(pop), derived_passes))
    multi = multichunk_verdict("manifest", chunks, man, C)
    chk.check("chunk_no_pinned_multichunk_pass", len(multi) == 0,
              "multichunk_passes=%s note=a pinned multi-chunk prefill needs a pass with "
              "T>chunk_width; if one ever appears this refusal has to become an adjudication"
              % (",".join("%s/%s" % k for k in multi) or "none"))

    # --- the witness that the census is not decoration: the SAME census over a manifest field that
    #     claims two chunks must report ADJUDICATION_REQUIRED. Only the in-memory copy of the record
    #     is mutated; no file is written and no bytes are read for the fake pass.
    fake_key = (blocks[0], "prefill", "q_pre")
    fake_rec = dict(recs[fake_key])
    fake_rec["shape"] = [1, 2 * C] + [int(x) for x in recs[fake_key]["shape"][2:]]
    fake_recs = dict(recs)
    fake_recs[fake_key] = fake_rec
    fake_pop, fake_chunks = chunk_census(fake_recs, C, "sabotage_manifest_field")
    fake_multi = multichunk_verdict("sabotage_manifest_field", fake_chunks, man, C)
    chk.witness("chunk_census_fires_on_a_multichunk_pass",
                fake_multi == [(blocks[0], "prefill")] and fake_pop == pop,
                "sabotage=shape[1] %d->%d on %s derived_chunks=%d verdict=ADJUDICATION_REQUIRED "
                "fired=%d note=an absent-material verdict that cannot become a present-material "
                "verdict is a constant wearing a name"
                % (recs[fake_key]["shape"][1], 2 * C, fake_key[0],
                   fake_chunks[(blocks[0], "prefill")], int(bool(fake_multi))))

    # --- the arm's own rule: one case is one chunk, witnessed in BOTH directions so neither a
    #     parser that refuses everything nor one that accepts everything can pass it (the same
    #     two-direction argument `speccheck/gdn_ref.py` uses for its accept/reject witnesses).
    refused = 0
    for Tbad in (C + 1, 2 * C):
        badp = WORK / ("plan_bad_T%d.tsv" % Tbad)
        badp.write_text("#GDNPLAN v2 n_cases=1 T=%d\nCASE idx=0 block=%s name=two_chunk_case T=%d "
                        "mode=honest wslot=0 carry=-1\n" % (Tbad, blocks[0], Tbad))
        pr = subprocess.run([str(HOST), "--plan", str(badp), "--inputs", str(ins),
                             "--weights", str(wts), "--out", str(WORK / "bad_chunk.out.bin"),
                             "--echo", str(WORK / "bad_chunk.echo.bin")],
                            capture_output=True, text=True)
        msg = ((pr.stdout + pr.stderr).strip().splitlines() or ["<silent>"])[-1]
        refused += 1 if pr.returncode != 0 else 0
        print("CONTROL arm=bad_plan rule=case_T_over_chunk T=%d chunk_width=%d rc=%d verdict=%s "
              "reason=%s" % (Tbad, C, pr.returncode, "REFUSED" if pr.returncode else "ACCEPTED",
                             msg[:110]))
    ok_plan = subprocess.run([str(HOST), "--plan", str(plan), "--inputs", str(ins),
                              "--weights", str(wts), "--out", str(WORK / "ok_chunk.out.bin"),
                              "--echo", str(WORK / "ok_chunk.echo.bin")],
                             capture_output=True, text=True)
    print("CONTROL arm=plan_rule rule=one_chunk_case_accepted T=%d rc=%d verdict=%s "
          "note=the accept leg is what makes the refusal above a chunk rule and not a parser that "
          "refuses every plan; no number from this run is judged"
          % (T, ok_plan.returncode, "ACCEPTED" if ok_plan.returncode == 0 else "REFUSED"))
    chk.check("chunk_case_refused_by_arm", refused == 2 and ok_plan.returncode == 0,
              "refused_T_over_chunk=%d of=2 accept_leg_rc=%d accept_T=%d note=a plan claiming more "
              "than one chunk per case must be refused: the scratch is sized for one chunk, so a "
              "silently-accepted one reads past it" % (refused, ok_plan.returncode, T))

    # ---------------- the split: chunk N reads chunk N-1's EMITTED state, in this same run -------
    n_chunks = 2
    if T % n_chunks or not 1 <= T // n_chunks < C:
        raise Refuse("T=%d cannot be split into %d equal chunks below the reference width %d; no "
                     "split carry is possible and none is faked" % (T, n_chunks, C))
    C_SPLIT = T // n_chunks
    FIELD_KEYS = ["q_pre", "k_pre", "v_pre", "beta", "g_decay", "z"]
    cases = []
    for bi, b in enumerate(blocks):
        base = len(cases)
        for j in range(n_chunks):
            fl = []
            for k in FIELD_KEYS:
                full = dump_tensor(recs, b, "prefill", k)
                fl.append(words_of(stage_token_slice(k, full, T,
                                                     j * C_SPLIT, (j + 1) * C_SPLIT)
                                   .contiguous().reshape(-1)))
            cases.append(dict(block=b, name="chunk%d" % j, j=j, T=C_SPLIT, mode="honest",
                              wslot=bi, carry=(-1 if j == 0 else base), fields=fl))
    derived_rows = len(blocks) * (len(STATE_AFFECTED) * n_chunks - 1)
    derived_bit = len(blocks) * n_chunks * (len(STAGES) - len(STATE_AFFECTED))
    print("SPLIT plan=split_%dchunk blocks=%s T_pass=%d n_chunks=%d chunk_widths_probed=%s "
          "chunk_width_measured=%d carry=last<-previous-per-block rows_banded_derived=%d "
          "rows_bitexact_derived=%d note=T/2 is a chunk shape the engine takes only as a final "
          "chunk, which is why these rows are cited and not credited"
          % (n_chunks, ",".join(blocks), T, n_chunks,
             ",".join(str(w) for w in lic.probed_widths), C_SPLIT, derived_rows, derived_bit))
    sp = WORK / "plan_split.tsv"
    write_plan(sp, cases)
    sp_in = WORK / "inputs_split.bin"
    write_inputs(sp_in, [c["fields"] for c in cases], C_SPLIT, states=[None] * len(cases))
    res = run_arm(HOST, "host_twin", sp, sp_in, wts, "split")
    cs, _, run = parse_arm(res, len(cases), C_SPLIT)
    if int(run["carried"]) != len(blocks) * (n_chunks - 1):
        fails.append("the split plan reported carried=%s, the plan declares %d carries: without a "
                     "carry every chunk reads zeros and nothing is being carried"
                     % (run["carried"], len(blocks) * (n_chunks - 1)))
    premise_bad, carried_nz = 0, []
    for i, c in enumerate(cases):
        src = c["carry"]
        if src < 0:
            continue
        want = int((stage_tensor(res, src, C_SPLIT, "state") != 0).sum())
        carried_nz.append(want)
        if want == 0 or int(cs[i]["state_nonzero"]) != want or int(cs[i]["carry"]) != src:
            premise_bad += 1
    chk.check("split_carry_premise", premise_bad == 0 and bool(carried_nz) and min(carried_nz) > 0,
              "carried_cases=%d carried_nonzero_min=%s mismatched=%d note=each carried chunk's "
              "state_nonzero is checked against the non-zero count of the state its predecessor "
              "emitted, read back out of the arm's own out.bin"
              % (len(carried_nz), min(carried_nz) if carried_nz else "none", premise_bad))
    print("SPLITFACT cases=%d pad_nonzero_per_case=%s state_nonzero_per_case=%s "
          "note=reported, not counted: the pad rows are synthesised inside the arm and no driver "
          "mutation can move them"
          % (len(cases), ",".join(d["pad_nonzero"] for d in cs),
             ",".join(d["state_nonzero"] for d in cs)))

    # ---- the banded rows: the stages the chunk boundary can reach, against the pinned slice
    n_rows, skipped = 0, 0
    state_rows = {}
    for i, c in enumerate(cases):
        for st in STATE_AFFECTED:
            full = dump_tensor(recs, c["block"], "prefill", st)
            if stage_token_axis(st, full, T) is None:
                if c["j"] != n_chunks - 1:
                    skipped += 1
                    print("SPLITSKIP block=%s chunk=%s stage=%s verdict=NO_PINNED_REFERENCE "
                          "note=the dump pins the state at the END of a pass, so an interior chunk "
                          "boundary has nothing to be compared against and none is invented"
                          % (c["block"], c["name"], st))
                    continue
                ref = full
            else:
                ref = stage_token_slice(st, full, T, c["j"] * C_SPLIT, (c["j"] + 1) * C_SPLIT)
            mine = stage_tensor(res, i, C_SPLIT, st)
            ok, m = band_row(band, "kernel_split_vs_dump", c["block"], c["name"], st, mine, ref,
                             split_lines, fails,
                             lambda b, p, s, w: "re-chunked %s/%s outside the derived band: %s"
                                                % (b, s, w),
                             name_key="%s/split_%dchunk/%s|%s" % (c["block"], n_chunks,
                                                                  c["name"], st))
            lic.cite(c["block"], "prefill", st, plan="split_%dchunk" % n_chunks,
                     form_width=C_SPLIT)
            n_rows += 1
            if st == "state":
                state_rows[c["block"]] = (ok, m)
    chk.check("split_banded_rows_eq_derived", n_rows == derived_rows and n_rows > 0,
              "rows=%d derived=%d skipped_no_reference=%d (=blocks %d x closure %d x chunks %d "
              "minus interior state %d) note=the closure is gdn_ref.downstream_closure("
              "scan_out,state), not a typed list"
              % (n_rows, derived_rows, skipped, len(blocks), len(STATE_AFFECTED), n_chunks,
                 len(blocks)))

    # ---- the stages the chunk boundary CANNOT reach: zero tolerance against the honest arm's own
    #      rows. Stronger than a band, and it is what certifies the slicing the rows above depend on.
    n_bit, bit_bad = 0, 0
    honest = run_arm(HOST, "host_twin", plan, ins, wts, "split_ref")
    parse_arm(honest, len(blocks), T)
    for i, c in enumerate(cases):
        ni = blocks.index(c["block"])
        for st in STAGES:
            if st in STATE_AFFECTED:
                continue
            full = dump_tensor(recs, c["block"], "prefill", st)
            ref = stage_token_slice(st, full, T, c["j"] * C_SPLIT, (c["j"] + 1) * C_SPLIT)
            mine = stage_tensor(res, i, C_SPLIT, st).reshape(ref.shape)
            hrow = stage_token_slice(st, stage_tensor(honest, ni, T, st).reshape(full.shape),
                                     T, c["j"] * C_SPLIT, (c["j"] + 1) * C_SPLIT)
            view = torch.int16 if mine.dtype is torch.bfloat16 else torch.int32
            nd = int((hrow.reshape(-1).view(view) != mine.reshape(-1).view(view)).sum())
            bit_bad += nd != 0
            n_bit += 1
            print("CHUNKLOCAL arm=split_vs_single_chunk block=%s chunk=%s stage=%s n=%d "
                  "ndiff=%d verdict=%s note=q_hat/k_hat sit outside the derived state-read closure, "
                  "so re-chunking must not move them; their pinned-bytes band verdict is the prefill "
                  "row of this same run and is not re-judged here"
                  % (c["block"], c["name"], st, ref.numel(), nd,
                     "BIT_IDENTICAL" if nd == 0 else "DIFFERS"))
    chk.check("split_row_local_stages_bit_identical", n_bit == derived_bit and bit_bad == 0,
              "rows=%d derived=%d differing=%d note=a wrong slice axis moves these by tens of "
              "thousands of words, so the zero they print certifies the slicing the banded rows "
              "above lean on" % (n_bit, derived_bit, bit_bad))

    for b in blocks:
        if b not in state_rows:
            continue
        ok, m = state_rows[b]
        print("CHUNKCARRY block=%s pass=prefill chunks=%d chunk_width=%d carried=last<-previous "
              "pinned_field=%s.prefill.state relL2=%.6g ndiff=%d/%d band=%s verdict=%s "
              "note=the state the last chunk emitted IN THIS RUN, from the state the first chunk "
              "emitted IN THIS RUN, against the pinned bytes"
              % (b, n_chunks, C_SPLIT, b, m["relL2"], m["ndiff"], m["n"],
                 "ACCEPTED" if ok else "REJECTED", "RECONSTITUTED" if ok else "NOT_RECONSTITUTED"))
    licence_tally(chk, lic, "chunkcarry:split_%dchunk" % n_chunks, n_rows, counts)

    # ---- sabotage: substitute a zero state for the carry. If the carry is load-bearing the last
    #      chunk's state row must go RED; if it prints ACCEPTED then that row was never being
    #      carried, and the section's headline claim would be about a number nobody read.
    zp = WORK / "plan_split_zerostate.tsv"
    write_plan(zp, [dict(c, carry=-1) for c in cases])
    resz = run_arm(HOST, "host_twin", zp, sp_in, wts, "split_zerostate")
    parse_arm(resz, len(cases), C_SPLIT)
    zs_red, zs_rows = 0, 0
    for bi, b in enumerate(blocks):
        i = bi * n_chunks + n_chunks - 1
        ref = dump_tensor(recs, b, "prefill", "state")
        ok, mz = band_row(band, "zerostate_sabotage_vs_dump", b, "chunk%d" % (n_chunks - 1),
                          "state", stage_tensor(resz, i, C_SPLIT, "state"), ref, split_lines,
                          fails, name_key="%s/sabotage_zero_state|state" % b)
        zs_rows += 1
        zs_red += 0 if ok else 1
    zs_detail = " | ".join(l.split(" stage=")[1][:56] for l in split_lines
                           if l.startswith("STAGE arm=zerostate"))
    chk.witness("split_carry_is_load_bearing", zs_rows == len(blocks) and zs_red == len(blocks),
                "sabotage=carry_replaced_by_zero_state rows=%d rejected=%d derived=%d "
                "note=a state row that survives having its incoming state removed was not being "
                "carried: %s" % (zs_rows, zs_red, len(blocks), zs_detail[:220]))
    counts["split_rows"] = n_rows
    counts["split_bit_rows"] = n_bit
    counts["chunk_census_passes"] = len(pop)
    counts["split_stage_lines"] = len(stage_lines) - split_from


def determinism_section(stage_lines, fails, counts, ran):
    """Determinism as a check, not a claim.

    The lane's premise is that this pipeline is deterministic, so the same command must print the
    same STAGE lines. This re-runs every section that printed a STAGE line as two further gate
    processes and compares their STAGE lines against each other AND against the ones this run already
    printed. The child `--only` list is DERIVED from what this run actually ran, so a partial run
    compares against a child that does the same work rather than a typed list that would silently
    compare different populations. A single differing line is a finding: on f32 data with no device,
    no timer and no thread count knob, a moving digit is a race or an uninitialised read, not
    rounding."""
    stage_secs = [s for s in ("band", "chunkcarry", "carry") if s in ran]
    if not stage_secs:
        raise Refuse("this run printed no STAGE lines; determinism over nothing is a constant "
                     "predicate — run with --only including band, chunkcarry or carry")
    cmd = [PY, str(HERE / "gate.py"), "--only", ",".join(stage_secs)]
    runs = []
    for i in (1, 2):
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        runs.append([l for l in (p.stdout + p.stderr).splitlines() if l.startswith("STAGE ")])
        # the two runs' own lines go to disk so the comparison is an artefact and not a printed claim
        (EVID / ("gdn-determinism-run%d-2026-10-26.log" % i)).write_text("\n".join(runs[-1]) + "\n")
        if not runs[-1]:
            fails.append("determinism run %d printed no STAGE lines (rc=%d): an empty parse cannot "
                         "prove agreement" % (i, p.returncode))
    here_set = list(stage_lines)
    if not here_set:
        raise Refuse("this run printed no STAGE lines; determinism over nothing is a constant "
                     "predicate — run with --only including band, chunkcarry or carry")
    same12 = runs[0] == runs[1]
    same_h = all(r == here_set for r in runs)
    diff_at = next((k for k in range(min(len(runs[0]), len(runs[1])))
                    if runs[0][k] != runs[1][k]), min(len(runs[0]), len(runs[1])))
    print("DETERMINISM here=%d run1=%d run2=%d run1_eq_run2=%d both_eq_this_run=%d "
          "first_differing_line=%s verdict=%s"
          % (len(here_set), len(runs[0]), len(runs[1]), int(same12), int(same_h),
             ("none" if (same12 and same_h) else str(diff_at)),
             "IDENTICAL" if (same12 and same_h) else "DIFFERS"))
    if not same12:
        fails.append("two runs of `%s` printed different STAGE lines at line %d:\n    run1: %s\n"
                     "    run2: %s" % (" ".join(cmd), diff_at, runs[0][diff_at][:200],
                                       runs[1][diff_at][:200]))
    if same12 and not same_h:
        fails.append("the two re-runs agree with each other but not with this run's STAGE lines "
                     "(%d vs %d lines): the driver, not the arm, is where the digit moved" %
                     (len(here_set), len(runs[0])))
    counts["determinism_lines"] = len(here_set)


def ladder_depth(dtype) -> int:
    """The same depth rule the oracle's own margin ladder uses: halvings the storage width can
    still express, derived from the dtype's epsilon and not typed."""
    k = int(round(-np.log2(torch.finfo(dtype).eps))) + 1
    if k < 2:
        raise Refuse("ladder depth %d from eps: the width cannot express a scaled defect" % k)
    return k


def witness(band, recs, blocks, T, fields, wslots, plan, ins, wts, counts):
    """The named `norm_weight` check, its accepted case reproduced in this lane, and both verdicts
    printed side by side.

    The band on `normed` is MEASURED to be able to hide a defect: the oracle's own margin ladder
    records `norm_weight_f32` at 1/2**2 as ACCEPTED with 190 of 131,072 stored elements still
    changed (speccheck/evidence/gdn-cross-form-2026-10-26.txt). The band is NOT tightened here — a
    smaller `normed` limit would be a number nobody measured. What closes the hole is an EXACT
    check on the weight the kernel loaded, because how the kernel obtained 128 words is a
    read/decode fact and not a reduction."""
    block = blocks[0]
    # --- the accepted case, reproduced: blend the honest normed toward the norm_weight_f32 normed
    honest = run_arm(HOST, "host_twin", plan, ins, wts, "honest")
    parse_arm(honest, len(blocks), T)
    ncase = blocks.index(block)
    nw_plan = WORK / "plan_nw.tsv"
    write_plan(nw_plan, [dict(block=b, name="prefill", T=T, mode="norm_weight_f32", wslot=i)
                         for i, b in enumerate(blocks)])
    dirty = run_arm(HOST, "host_twin", nw_plan, ins, wts, "nwf32")
    parse_arm(dirty, len(blocks), T)
    ref = dump_tensor(recs, block, "prefill", "normed")
    a0 = stage_tensor(honest, ncase, T, "normed").reshape(ref.shape)
    d0 = stage_tensor(dirty, ncase, T, "normed").reshape(ref.shape)
    a64, d64 = a0.to(torch.float64), d0.to(torch.float64)
    print("WITNESS case=norm_weight_f32_ladder block=%s depth=%d ref_n=%d"
          % (block, ladder_depth(torch.bfloat16), ref.numel()))
    accepted = []
    for k in range(0, ladder_depth(torch.bfloat16) + 1):
        t = (a64 + (2.0 ** (-k)) * (d64 - a64)).to(torch.bfloat16)
        moved = int((t != a0).sum())
        ag = G.aggregates(t, ref, name="margin/%s/norm_weight_f32/k%d" % (block, k), stage="normed")
        ok, why = G.band_check_row(band, "normed", ag)
        print("LADDER block=%s fault=norm_weight_f32 k=%d moved=%d/%d meanrel=%.3e 1-cos=%.3e "
              "maxabs=%.3e[reported_only] band=%s why=%s"
              % (block, k, moved, ref.numel(), ag["mean_rel"], ag["one_minus_cos"], ag["max_abs"],
                 "ACCEPTED" if ok else "REJECTED", why))
        if ok and moved > 0:
            accepted.append((k, moved, ag))
    if not accepted:
        fails_note = ("the ladder never accepted: this lane could not reproduce the accepted case, "
                      "so the witness below is not attached to a case the band accepted")
        print("WITNESS note=%s" % fails_note)
    else:
        kmin = min(accepted, key=lambda r: r[0])
        print("WITNESS least_scaled_accepted_k=%d moved=%d/%d meanrel=%.3e licensed_meanrel=%.3e"
              % (kmin[0], kmin[1], ref.numel(), kmin[2]["mean_rel"], band["normed"]["mean_rel"]))

    # --- now perturb norm_weight IN MEMORY and run again: exact check vs band, same case
    pristine = wslots[block][0].copy()
    print("SIDEBY header1=band_check_row(normed) header2=norm_weight_bit_exact")
    any_pair = None
    for rung in range(7, -1, -1):
        w = pristine.copy()
        idx = np.arange(w.size, dtype=np.uint32)
        sel = (idx & np.uint32((1 << rung) - 1)) == 0
        w[sel] = w[sel] ^ np.uint32(1)               # flip the bf16 LSB of 1 word in 2**rung
        slots = [wslots[b] for b in blocks]
        slots[ncase] = (w, wslots[block][1])
        pins = WORK / ("weights_perturb%d.bin" % rung)
        write_weights(pins, slots)
        res = run_arm(HOST, "host_twin", plan, ins, pins, "perturb%d" % rung)
        cs, fps, run = parse_arm(res, len(blocks), T)
        seen = echo_tensor(res, ncase, T, "norm_weight")
        nd = int((seen != pristine).sum())
        moved_expect = int(sel.sum())
        exact_red = nd > 0
        mine = stage_tensor(res, ncase, T, "normed").reshape(ref.shape)
        ag = G.aggregates(mine, ref, name="perturb%d/%s/normed" % (rung, block), stage="normed")
        ok, why = G.band_check_row(band, "normed", ag)
        m = metrics(mine, ref)
        print("SIDEBY rung=%d perturb=bf16_lsb_flip words_perturbed=%d words_moved=%d "
              "derived_moved=%d exact=%s band=%s %s why=%s"
              % (rung, moved_expect, nd, moved_expect, "RED" if exact_red else "GREEN",
                 "ACCEPTED" if ok else "REJECTED", fmt(m), why))
        if nd != moved_expect:
            fails_note = ("exact check moved %d words when the perturbation was derived to move %d"
                          % (nd, moved_expect))
            print("WITNESS note=%s" % fails_note)
        if ok and exact_red:
            any_pair = rung
    print("WITNESS side_by_side_pair_found=%s rung=%s note=a band-ACCEPTED norm_weight byte defect "
          "and a RED byte-exactness check on the same case"
          % ("YES" if any_pair is not None else "NO", any_pair))
    counts["witness_rungs"] = 8
    counts["witness_ladder_rows"] = ladder_depth(torch.bfloat16) + 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as exc:
        print("REFUSE gate: %s" % exc)
        sys.exit(1)
