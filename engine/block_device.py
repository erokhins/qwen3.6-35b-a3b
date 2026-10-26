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

import os
import subprocess
import argparse
import hashlib
import math
import re
import sys
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
        self.fired = 0

    def check(self, name, predicate, detail):
        ok = bool(predicate)
        self.names.append(name)
        if not ok:
            self.red.append(name)
        print("CHECK name=%s verdict=%s %s" % (name, "PASS" if ok else "FAIL", detail))
        return ok

    def control(self, name, fired, detail, legs=None, claims=None):
        self.controls.append(name)
        if fired:
            self.fired += 1
        else:
            self.red.append("control:" + name)
        self.control_paths[name] = dict(legs=dict(legs or {}), claims=tuple(claims or ()))
        pathnote = ""
        if legs or claims:
            pathnote = " legs=%s claims=%s" % (
                ",".join("%s:%d" % (p, n) for p, n in sorted((legs or {}).items())) or "none",
                ",".join(claims or ()) or "none")
        print("CONTROL name=%s fired=%s%s %s" % (name, "RED" if fired else "SILENT", pathnote,
                                                 detail))
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
            len(parsed) == len(rows) + len(ctx.get("g_rows", [])) and len(parsed) > 0,
            "parsed=%d printed=%d (shared-expert and router rows=%d, gather rows on two width paths=%d); "
            "a parser matching nothing would print 0 and pass"
            % (len(parsed), len(rows) + len(ctx.get("g_rows", [])), len(rows),
               len(ctx.get("g_rows", []))))
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


def control_no_device_act_guard(v, ctx):
    """The `no device acts` predicate has to be capable of failing, and this shows it with a fabricated act.

    The check above passes by reading counters that are all zero, and "all zero" is also what a counter that
    is never incremented, never printed, or read off the wrong dict looks like — the first version of that
    check threw `KeyError: 'run'` and so never ran at all. So the predicate is run here over three fabricated
    records whose only difference from the real ones is the counter: one that launched, one that moved bytes
    to the device, and one whose `RUN` line does not report `launches` at all. All three must be read as
    offences, and the honest set must read clean, or the check is decoration.

    No arm runs here; the legs are driver-side, so the control declares no path claims.
    """
    honest = dict((k, dict(run_facts(r))) for k, r in ctx["all_runs"].items())
    victim = sorted(honest)[0]
    fabricated = {
        "launches=1": dict(honest[victim], launches=1),
        "bytes_to_device=1GiB": dict(honest[victim], bytes_to_device=1073741824),
        "launches-absent": dict((k, x) for k, x in honest[victim].items() if k != "launches"),
    }
    bad = dict((name, no_device_act_offences({victim: rec})) for name, rec in fabricated.items())
    clean = no_device_act_offences(ctx["all_runs"])
    for name in sorted(fabricated):
        print("GWITNESS guard=no-device-acts fabricated=%s offences=%s" % (name, bad[name] or "NONE"))
    fired = (not clean and all(bad[n] for n in bad))
    v.control("control-witness-guard-can-redden", fired,
              "the same predicate the check runs, over the honest set and three fabrications: honest=%s; "
              "%s. Every fabrication is read as an offence, including the record that simply omits the "
              "counter — a RUN line that does not report a number cannot certify that the number was zero"
              % (clean or "clean",
                 "; ".join("%s -> %s" % (n, ",".join(bad[n])) for n in sorted(bad))))


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


def control_gather_read_back(v, ctx):
    """In a device window: the bytes the device holds for one expert span are the bytes the file holds.

    Same purpose as `stage_run.device_read_back` for the shared expert: bind the kernel's input to the
    artifact, 16 bytes at the first gathered span, before any row is believed.

    In a CPU witness this control is reported VOID and not registered, because a read-back that read host
    memory and were called a device read-back is the substitution this lane exists to refuse.
    """
    if not ctx["device"]:
        print("GVOID control=gather-device-unit-read-back reason=no-device-act-in-this-run "
              "note=not registered as a control here: a host read of host bytes is not a device read-back")
        return
    snap = (dict(PATH_LEGS), dict(G_LEGS))
    ok, lines = True, []
    for width in DA.G_PATHS:
        p = ctx["gpaths"][width]
        u = p["units"][0]
        plan = g_plan_variant(p, p["plan"].read_text().replace(
            "read_off=-1 read_len=0", "read_off=%d read_len=16" % u["off"]))
        rec = g_run_variant(ctx, width, plan, "out_readback.bin", "g_%s_readback" % width, device=True)
        want = u["_bytes"][:16].hex()
        got = re.search(r"UNITREADBACK path=device off=%d bytes=16 hex=([0-9a-f]+)" % u["off"],
                        rec["stdout"])
        match = bool(got) and got.group(1) == want
        ok = ok and match
        lines.append("%s %s=%s" % (width, "match" if match else "MISMATCH",
                                    (got.group(1) if got else "ABSENT")[:16]))
    v.control("gather-device-unit-read-back", ok,
              "first gathered span, first 16 bytes, device vs file: %s" % "; ".join(lines),
              legs=g_legs_since(snap), claims=g_claims(ctx))


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
    print("BLOCKDEVICE block=layer0 subsystem=shared_expert+router token=%d device=%s work=%s "
          "se_arm=%s se_host=%s rt_arm=%s rt_host=%s"
          % (BF.TOKEN_ROW, "on" if device else "off", work, args.arm_binary, args.host_binary,
             args.rt_arm_binary, args.rt_host_binary))
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
    ctx["all_runs"] = dict([("se/" + k, v) for k, v in arm_runs.items()]
                           + [("rt/" + k, v) for k, v in rt_runs.items()]
                           + [("g/" + k, v) for k, v in g_runs.items()])
    for key in ("licensed", "accepted", "rejected", "unlicensed"):
        tally[key] += g_tally[key]
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
    control_gather_read_back(v, ctx)
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

    if st["hip"] is not None:
        st["hip"].free_all()
        WITNESS["hip_calls_python"] += st["hip"].tally["frees"]
    red = len(v.red)
    verdict = "FAIL" if red else ("QUALIFIED" if tally["unlicensed"] else "PASS")
    print("STAGETALLY arms=1 licensed=%d accepted=%d rejected=%d unlicensed=%d"
          % (tally["licensed"], tally["accepted"], tally["rejected"], tally["unlicensed"]))
    print("BLOCKDEVICE verdict=%s block=layer0 subsystem=shared_expert+router rows=%d parsed=%d "
          "licensed=%d accepted=%d rejected=%d unlicensed=%d checks=%d red=%d controls=%d fired=%d "
          "band_digest=%s note=a row with no licence is never counted as accepted, and PASS is "
          "unreachable while any row is unlicensed"
          % (verdict, len(rows), len(parsed), tally["licensed"], tally["accepted"],
             tally["rejected"], tally["unlicensed"], len(v.names), red, len(v.controls), v.fired,
             digest[:16]))
    print("RESULT mode=%s checks=%d findings=%d controls=%d fired=%d rows=%d arms=policy_a "
          "device=%s staged=%s hip_calls=%d+%d launches=%d bytes_to_device=%d bytes_from_device=%d "
          "band_digest=%s no timing taken"
          % ("device" if device else "cpu", len(v.names), red, len(v.controls), v.fired, len(rows),
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
