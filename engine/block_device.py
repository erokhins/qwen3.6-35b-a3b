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
           "rows_emitted": 0}
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


def silent_control_finding(fired, reason):
    """The rule a SILENT control has to obey: if it did not fire, it says why, in words.

    Window `g` printed `CONTROL name=control-witness-guard-can-redden fired=SILENT` on the device path
    with nothing after it. A silent guard is not a neutral state — it is the state in which a guard is not
    guarding — and a bare SILENT is how a control that lost its legs on the path under claim reads the same
    as one that ran and found nothing. `silent-control-names-its-reason` is the check; this function is the
    single rule so the check and the witness cannot disagree about what counts as a named reason.
    """
    if fired:
        return None
    return None if (reason or "").strip() else "silent-without-named-reason"


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
        if r.get("items") != "%d/1" % rt["MR_E"]:
            census_bad.append("%s.items=%s/%d/1" % (label, r.get("items"), rt["MR_E"]))
        if r.get("bad_index", "0") != "0":
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
    if device:
        v.check("rt-device-launch-counted",
                any(int(a["rec"]["run"]["launches"]) > 0 and int(a["rec"]["run"]["hip_calls"]) > 0
                    for a in rt_runs.values() if a["rec"]["run"]["path"] == "device"),
                "the device arm counted at least one launch and one HIP call; the flag is not the "
                "evidence, the counter is")
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
            "every control that did not fire also printed a `reason=` naming what it could not do: %d of %d "
            "controls fired, and the ones that went quiet are %s. Window `g` is why this is a check and not a "
            "style rule: `control-witness-guard-can-redden` printed a quiet line with no reason on the device "
            "path, which reads exactly like a guard that ran and found nothing. A control that cannot run a "
            "leg prints that leg as not-exercised instead; a control that goes quiet names the finding"
            % (v.fired, len(v.controls), ",".join(v.unreasoned) or "none"))
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
    run = rec["run"]
    WITNESS["hip_calls_arm"] += int(run["hip_calls"])
    WITNESS["launches"] += int(run["launches"])
    WITNESS["allocs"] += int(run["allocs"])
    WITNESS["bytes_to_device"] += int(run["bytes_to_device"])
    WITNESS["bytes_from_device"] += int(run["bytes_from_device"])
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


def device_act_absences(runs):
    """Offences among the records that ran ON the device: a `path=device` act that reports none.

    The mirror image of `no_device_act_offences`, and needed for the same reason. In a device window the
    predicate that guards "no device act" is supposed to read CLEAN on the host bodies and DIRTY on the
    device runs; if it read clean on those too, the check would be certifying that nothing ran while five
    arms were launching. `launches=0`, `bytes_from_device=0` or a missing counter on a `path=device` record
    is the silent fallback this lane exists to refuse, so it is an offence of its own and
    `control-witness-guard-can-redden` shows the predicate can see it (window `g`: that control printed
    `fired=SILENT` on the device path because its honest half demanded an act-free set that a device run
    cannot honestly produce).
    """
    bad = []
    for label in sorted(runs):
        run = run_facts(runs[label])
        if run.get("path") != "device":
            continue
        for k in ("launches", "bytes_from_device"):
            v = run.get(k)
            try:
                if v is None or not int(v):
                    bad.append("%s.%s=%s" % (label, k, "absent" if v is None else v))
            except (TypeError, ValueError):
                bad.append("%s.%s=%s" % (label, k, v))
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
    clean = no_device_act_offences(host_runs)
    absent = device_act_absences(dev_runs)
    for name in sorted(fabricated):
        print("GWITNESS guard=no-device-acts fabricated=%s offences=%s" % (name, bad[name] or "NONE"))
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
    fired = bool(not clean and not absent and all(bad[n] for n in bad) and fake_absent)
    nlegs = len(fabricated) + 1 + (1 if dev_runs else 0) + 1
    v.control("control-witness-guard-can-redden", fired,
              "the same predicate the check runs, over the honest set split by the path the arm reported "
              "(executed=device:%d,host_cpubody:%d predicate legs, no arm act) and three fabrications: "
              "host-clean=%s; device-acts=%s; %s. Every fabrication is read as an offence, including the "
              "record that simply omits the counter — a RUN line that does not report a number cannot "
              "certify that the number was zero"
              % (len(dev_runs), len(host_runs), clean or "clean", absent or "clean",
                 "; ".join("%s -> %s" % (n, ",".join(bad[n])) for n in sorted(bad))),
              legs={"cpu_predicate": nlegs},
              reason=(None if fired else
                      "honest set disagreed with the predicate the way the path forbids: host records "
                      "offending=%s, device records missing acts=%s (see the GWITNESS legs above)" %
                      (clean or "clean", absent or "clean")))


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
              claims=("device",))


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
            ("byte_shift_1", "a one-byte span offset: the stride/offset read defect"))
D_CAUSES = (("decay_ulp1", "transcendental last-bit: exp(decay) one ULP up"),
            ("decay_head_off1", "wrong head's decay: the index defect"),
            ("state_xpose", "carried state read transposed: the index-mapping defect"))


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
    accounted, injected_nonzero, lines, reach_offences = [], [], [], []
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    ctx.setdefault("g_cause_match", {})
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        # The honest side is the hipcc arm's HOST path in both modes, on purpose: an injected cause is
        # measured inside one compiler, so the number it produces is the price of the cause and not of the
        # cause plus a codegen difference. What a cause can only be priced against on the device itself is
        # listed as a not-exercised leg.
        honest_rec = p["dir"] / "out_hip.bin"
        rows_by_cause = {}
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
    reach_offences += ["%s inert-on-both-widths" % c for c in inert]
    ok = (all(a["moved"] == a["total"] for a in accounted)
          and any(t > 0 for w, c, t in injected_nonzero if w == "policy_a")
          and gate_up_clean and ref_absorbed and not reach_offences)
    reason = None
    if reach_offences:
        reason = ("an injected mode moved fields its own HDR reach claim does not name, or moved nothing on "
                  "either width (GREACH lines name which): %s" % ";".join(reach_offences))
    elif not ok:
        reason = ("attribution did not account (sum_of_fields != ndiff_bytes) or the injected causes were "
                  "inert or the transcendental cause moved a field upstream of it: accounted=%s "
                  "policy_a_injections_nonzero=%s gate_up_clean=%s ref_bf16_absorbs_exp=%s"
                  % ([ (a["pair"], a["total"], a["moved"]) for a in accounted if a["moved"] != a["total"]],
                     any(t > 0 for w, c, t in injected_nonzero if w == "policy_a"),
                     gate_up_clean, ref_absorbed))
    if ctx["device"]:
        unresolved = [w for w in DA.G_PATHS if w in ctx.get("g_device_pair_rows", {})
                      and not ctx.get("g_cause_match", {}).get(w)]
        if unresolved:
            ok = False
            reason = ("no injected cause reproduces the device divergence in field location AND magnitude "
                      "on %s; the mechanism is therefore UNATTRIBUTED and no scope may be claimed for it "
                      "(GMAGMATCH lines name what was ruled out and by how much)" % ",".join(unresolved))
    v.control("gather-divergence-causes-priced", ok,
              "five candidate causes injected on the arm's host path and counted per field; every "
              "attribution accounts to its pair total; the reference path absorbs the maximal exp "
              "injection completely (total=%s) while policy A moves %d bytes with gate/up untouched — the "
              "measured shape of the device's split verdict. %s"
              % (ref_silu["total"] if ref_silu else "?",
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
    honest = ctx["work"] / "d_out_hip.bin"
    obs = ctx.get("d_device_pair_rows")
    accounted, rows_by_cause, reach_offences = [], {}, []
    for cause, what in D_CAUSES:
        r = d_variant(ctx, "d_cause_%s.bin" % cause, "d_cause_%s" % cause, mode=cause,
                      binary=ctx["args"].d_arm_binary)
        rows, total = record_field_rows(r["rec"]["wrote"]["path"], honest, d["order"], d["fields"])
        rows_by_cause[cause] = print_field_attrib("DINJECT", "d_cause_%s_vs_honest" % cause,
                                                 "gdn_record.f32", rows, total,
                                                 note="cause=%s (%s), hipcc arm's host path"
                                                      % (cause, what))
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
        if cause == "decay_ulp1":
            # The ruler for the head-count question: this injection moves EVERY head's block, so the device
            # run's `heads_with_diff` says directly how many of the 32 decays differ there.
            d_state_per_head("d_cause_%s_vs_honest" % cause, r["rec"]["wrote"]["path"], honest, d)
        accounted.append(rows_by_cause[cause])
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
        if not best and obs["total"]:
            ok = False
            reason = ("no injected cause reproduces the GDN device divergence in field location AND "
                      "magnitude; the mechanism is UNATTRIBUTED and no scope may be claimed for it "
                      "(DMAGMATCH names what was ruled out and by how much)")
    else:
        print_device_only_attribution_dvoids("gdn", "d_hip_device_vs_d_hip_hostpath")
    v.control("gdn-divergence-causes-priced", ok,
              "three candidate causes injected and counted per field; every attribution accounts to its "
              "pair total; every cause moved exactly the fields the arm's own HDR reach claim names "
              "(DREACH); q_hat/k_hat untouched by all three, which is what the reach table promises and "
              "this leg measures. %s" % (
                  ";".join("%s=%d" % (c, a["total"]) for c, a in rows_by_cause.items())),
              reason=reason)


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
    v.check("gather-compiled-pair-byte-identical-per-path",
            all(p["pair"][1] == 0 and p["pair"][0] > 0 for p in ctx["gpaths"].values()),
            "per width path, g++ twin vs hipcc: %s at zero tolerance. One aggregate would let one path "
            "cancel the other's difference" % ",".join("%s bytes=%d ndiff=%d"
                                                      % (w, p["pair"][0], p["pair"][1])
                                                      for w, p in sorted(ctx["gpaths"].items())))
    v.check("gather-unlicensed-counted-not-absorbed",
            g_tally["unlicensed"] == len([r for r in g_rows if r["arm"] == "policy_a"]),
            "the gather tally carries licensed=%d accepted=%d rejected=%d unlicensed=%d; the policy-A "
            "rows are counted as unlicensed rather than dropped, and PASS remains unreachable while any "
            "row is unlicensed" % (g_tally["licensed"], g_tally["accepted"], g_tally["rejected"],
                                   g_tally["unlicensed"]))
    v.check("gather-device-path-claims-executed-on-device",
            (not ctx["device"]) or all(r["run"]["launches"] > 0 for k, r in g_runs.items()
                                       if "device" in k),
            "in a device run every gather device arm reported launches>0 (%d device arms); in a CPU window "
            "the same --device ask on the g++ twin is refused with a named tag instead of a host run, and "
            "this check is vacuous here and says so" % len([k for k in g_runs if "device" in k]))
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
    run = rec["run"]
    WITNESS["hip_calls_arm"] += int(run["hip_calls"])
    WITNESS["launches"] += int(run["launches"])
    WITNESS["allocs"] += int(run["allocs"])
    WITNESS["bytes_to_device"] += int(run["bytes_to_device"])
    WITNESS["bytes_from_device"] += int(run["bytes_from_device"])
    return rec


def d_variant(ctx, out_name, label, mode="correct", fault="none", binary=None,
              zero_state=False, flip_state_word=None, device=False, form=None, elt_override=None,
              words_override=None, drop_field=None, subdir="dvariant", metrics_ok=True):
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
                    elt_override=elt_override, words_override=words_override, drop_field=drop_field)
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
        d_state_per_head("d_hip_device_vs_d_hip_hostpath", work / "d_out_device.bin",
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
    honest_ok, rows = True, []
    snap = dict(PATH_LEGS)
    for label in sorted(ctx["d_runs"]):
        ag = ctx["d_censuses"][label]
        leg = (not ag["disagree"] and ag["arm_state_sum"] == ag["driver_state_sum"]
               and ag["driver_state_live"] > 0 and ag["state_words"] > 0)
        honest_ok = honest_ok and leg
        rows.append("%s=%s(sum arm=%d driver=%d live=%d/%d)"
                    % (label, "agree" if leg else "DISAGREE", ag["arm_state_sum"],
                       ag["driver_state_sum"], ag["arm_state_live"], ag["driver_state_live"]))
    flip_ok, flip_rows = True, []
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
        rows_moved = (r["metrics"]["state"]["ndiff"] > hm["state"]["ndiff"]
                      and all(r["metrics"][s]["ndiff"] >= hm[s]["ndiff"] for s in D_STATE_STAGES))
        flip_ok = flip_ok and moved_fp and still_agree and rows_moved
        flip_rows.append("%s[fingerprint %d->%d agrees=%s state_ndiff %d->%d scan_out_ndiff %d->%d]"
                         % (name, fp["sum"], r["census"]["state_sum"], still_agree,
                            hm["state"]["ndiff"], r["metrics"]["state"]["ndiff"],
                            hm["scan_out"]["ndiff"], r["metrics"]["scan_out"]["ndiff"]))
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
                back_ok, back_rows = False, back_rows + ["%s=line-absent" % key]
                continue
            off, nbytes, got = int(m.group(1)), int(m.group(2)), m.group(3)
            same = (got == want_hex and nbytes == D_STATE_READBACK and off == state_off)
            back_ok = back_ok and same
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
    v.control("gdn-state-read-is-the-arm-s-own-fingerprint",
              honest_ok and flip_ok and (not ctx["device"] or back_ok),
              "honest legs: %s — the arm's modular fold of the state's 524288 words equals the driver's "
              "fold of the bytes it wrote, on every arm that ran, with %d live words as a non-zero "
              "denominator. Flip legs: %s — flipping one byte of the state's 77,777th word moves the "
              "fingerprint on both sides, keeps them equal, and pushes the state row further from the pin "
              "while no derived row gets closer, so the number is a witness of the read and not a constant "
              "the arm prints."
              % ("; ".join(rows), fp["live"], "; ".join(flip_rows)),
              legs=legs_since(snap), claims=d_claims(ctx))


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
    v.check("gdn-compiled-pair-byte-identical",
            ctx["d_pair"][1] == 0 and ctx["d_pair"][0] > 0
            and (ctx["d_device_pair"] is None or ctx["d_device_pair"][1] == 0),
            "g++ twin vs hipcc host path over %d B: ndiff_bytes=%d at zero tolerance%s. A non-zero here "
            "is a codegen finding, never a band to absorb: the two compilers are handed one arithmetic "
            "(`engine/device/m2_d_kernels.h` is the only place it is written), so agreement is what the "
            "shared body promises and disagreement is a compiler question, not a licence question"
            % (ctx["d_pair"][0], ctx["d_pair"][1],
               "" if ctx["d_device_pair"] is None
               else ", and device vs host path ndiff_bytes=%d" % ctx["d_device_pair"][1]))
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
    v.check("gdn-device-path-claims-executed-on-device",
            (not ctx["device"]) or all(r["run"]["launches"] > 0 for k, r in d_runs.items()
                                       if "device" in k),
            "in a device run the GDN device arm reported launches>0 (%d device arms, 8 launches each: "
            "poison plus one per stage); in a CPU window the same --device ask on the g++ twin is refused "
            "with a named tag instead of a host run, and this check is vacuous here and says so"
            % len([k for k in d_runs if "device" in k]))


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
        WITNESS["hip_calls_arm"] += int(rec["run"]["hip_calls"])
        WITNESS["launches"] += int(rec["run"]["launches"])
        WITNESS["allocs"] += int(rec["run"]["allocs"])
        WITNESS["bytes_to_device"] += int(rec["run"]["bytes_to_device"])
        WITNESS["bytes_from_device"] += int(rec["run"]["bytes_from_device"])
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
               rt_pairs=rt_pairs)
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
    control_gather_poison(v, ctx)
    control_gather_order_and_form(v, ctx)
    control_gather_width_substitution(v, ctx)
    control_gather_device_refusal(v, ctx)
    control_gather_read_back_expectation(v, ctx)
    control_gather_read_back(v, ctx)
    control_gather_divergence_causes(v, ctx)
    control_d_divergence_causes(v, ctx)
    control_d_state_transposition(v, ctx)
    control_d_carry_index(v, ctx)
    control_d_non_carried_state(v, ctx)
    control_d_state_read_witness(v, ctx)
    control_d_form_not_merged(v, ctx)
    control_d_poison(v, ctx)
    control_d_plan_refusals(v, ctx)
    control_d_device_ask_refuses(v, ctx)
    control_claim_scan(v, ctx)
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

    if st["hip"] is not None:
        st["hip"].free_all()
        WITNESS["hip_calls_python"] += st["hip"].tally["frees"]
    red = len(v.red)
    verdict = "FAIL" if red else ("QUALIFIED" if tally["unlicensed"] else "PASS")
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
          "band_digest=%s no timing taken"
          % ("device" if device else "cpu", len(v.names), red, len(v.controls), v.fired, len(rows),
             len(ctx.get("d_rows", [])),
             "on" if device else "off",
             "admitted" if WITNESS["staged_runs"] else "not-attempted",
             WITNESS["hip_calls_python"], WITNESS["hip_calls_arm"], WITNESS["launches"],
             WITNESS["bytes_to_device"], WITNESS["bytes_from_device"], digest[:16]))
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
        print("REFUSED tag=%s note=%s" % (exc.tag, exc))
        print("RESULT mode=%s checks=0 findings=1 controls=0 fired=0 rows=0 arms=policy_a "
              "device=%s staged=%s hip_calls=%d+%d launches=%d no timing taken"
              % ("device" if args.device else "cpu", "on" if args.device else "off",
                 "admitted" if WITNESS["staged_runs"] else "not-attempted",
                 WITNESS["hip_calls_python"], WITNESS["hip_calls_arm"], WITNESS["launches"]))
        return 1
    finally:
        sys.stdout = out.fh


if __name__ == "__main__":
    sys.exit(main())
