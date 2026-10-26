#!/usr/bin/env python3
"""block_device.py — M2's live device path: stage layer 0's shared-expert resident bytes, launch the
gfx1151 kernel on them, adjudicate the rows it produced under the licences that already exist.

`engine/device_arm.py` is the staging-and-launch half; `engine/HANDOFF.md` §THE DEVICE HALF is the
window this belongs to and the exact command line. This file is the driver: what runs, what is
printed, and what it refuses.

THREE WAYS THIS COULD BE DISHONEST, and the mechanism that closes each.
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

import argparse
import hashlib
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


class Refused(DA.Refused):
    """One refusal type in the lane: `engine/device_arm.py` raises it and this driver catches it."""


class Verifier:
    """Named checks and witnessed controls (`engine/stage_run.py`::Verifier is the house shape)."""

    def __init__(self):
        self.names, self.red, self.controls = [], [], []
        self.fired = 0

    def check(self, name, predicate, detail):
        ok = bool(predicate)
        self.names.append(name)
        if not ok:
            self.red.append(name)
        print("CHECK name=%s verdict=%s %s" % (name, "PASS" if ok else "FAIL", detail))
        return ok

    def control(self, name, fired, detail):
        self.controls.append(name)
        if fired:
            self.fired += 1
        else:
            self.red.append("control:" + name)
        print("CONTROL name=%s fired=%s %s" % (name, "RED" if fired else "SILENT", detail))
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
    """Top-k selection must be the thing the `topk_ids` row is about, in both of its halves.

    Two legs, both run on the engine's own bytes:
      * tie-break — a forced exact tie across the K boundary (forged at the boundary the row itself
        selects, not at a typed index), where the lower-index rule and the higher-index rule must name
        different experts, and the row must move;
      * slot order — the prototype's own `topk_ascending` mode, which must move the ids and the weights
        while leaving the logits and the probabilities bit-identical, because `mr_mode_reaches` says it
        cannot reach them. A mode that moves more than it names is a second bug, and a control that
        only looked at the ids row would not see that.
    """
    rt, work, args = ctx["rt"], ctx["work"], ctx["args"]
    sdir = work / "rtselect"
    sdir.mkdir(parents=True, exist_ok=True)
    honest = ctx["rt_fields"]
    base = ctx["rt_slab"]["plan"].read_text()
    rows_before = WITNESS["rows_emitted"]
    honest_nd = {k: rt_form_distance(ctx["dumps"], "layer0.%s" % k, honest[f])[0]
                 for k, f in (("topk_ids", "ids"), ("topk_weights", "wcast"))}

    def run_variant(name, text, fault=None):
        p = sdir / ("plan_%s.txt" % name)
        p.write_text(text)
        out = sdir / ("out_%s.bin" % name)
        DA.run_arm(args.rt_host_binary, p, ctx["rt_slab"]["slab"], out, fault=fault, label=name)
        fields, _ = DA.rt_record(out, rt)
        return fields

    tie = run_variant("tie_forge", base.replace("tie_forge=0", "tie_forge=1"))
    tie_nd_ids = rt_form_distance(ctx["dumps"], "layer0.topk_ids", tie["ids"])[0]
    tie_alt_nd = rt_form_distance(ctx["dumps"], "layer0.topk_ids", tie["ids_alt"])[0]
    tie_rules_differ = bool((tie["ids"] != tie["ids_alt"]).any())

    asc = run_variant("topk_ascending", base.replace("mode=correct", "mode=topk_ascending"))
    asc_nd_ids = rt_form_distance(ctx["dumps"], "layer0.topk_ids", asc["ids"])[0]
    asc_nd_w = rt_form_distance(ctx["dumps"], "layer0.topk_weights", asc["wcast"])[0]
    untouched = [n for n in ("logits", "probs") if bool((asc[n] != honest[n]).any())]
    detail = ("tie-forge: with an exact tie forged at the boundary, the rule in force (lower index) still "
              "picked the pinned expert — ids row ndiff %d->%d — but the two tie rules named %s, and the "
              "other rule's ids row is ndiff %d, so an arm that had the rule backwards would print a red "
              "`topk_ids` row on this very input; topk_ascending: ids row ndiff %d->%d, weights row ndiff "
              "%d->%d, fields that must not move and did: %s; rows emitted by the control=%d"
              % (honest_nd["topk_ids"], tie_nd_ids,
                 "different experts" if tie_rules_differ else "THE SAME (forgery failed)", tie_alt_nd,
                 honest_nd["topk_ids"], asc_nd_ids, honest_nd["topk_weights"], asc_nd_w,
                 untouched or "none", WITNESS["rows_emitted"] - rows_before))
    v.control("topk-selection-perturbations-redden-ids",
              tie_rules_differ and tie_alt_nd > honest_nd["topk_ids"]
              and asc_nd_ids > honest_nd["topk_ids"] and asc_nd_w > honest_nd["topk_weights"]
              and not untouched and WITNESS["rows_emitted"] == rows_before, detail)


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
    if device:
        v.check("rt-device-launch-counted",
                any(int(a["rec"]["run"]["launches"]) > 0 and int(a["rec"]["run"]["hip_calls"]) > 0
                    for a in rt_runs.values() if a["rec"]["run"]["path"] == "device"),
                "the device arm counted at least one launch and one HIP call; the flag is not the "
                "evidence, the counter is")
        cl = ctx["rt_devdiff"]["claimed"]
        v.check("rt-device-host-claimed-fields-identical",
                all(v[0] == 0 for v in cl.values()) and len(cl) == 4,
                "device arm vs host twin on the four fields the licences measure at exactness: %s "
                "(whole record ndiff_bytes=%d of %d). The long-dot fields are outside this claim on "
                "purpose: `prototype/moe_router/moe_router_kernels.h` §mr_exp documents that the two "
                "libms' exp() may differ by an ULP, and that is a printed fact, not a band"
                % (",".join("%s=%d/%d" % (k, v[0], v[1]) for k, v in sorted(cl.items())),
                   ctx["rt_devdiff"]["full"][1], ctx["rt_devdiff"]["full"][0]))


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
    v.check("parsed-record-count-equals-printed", len(parsed) == len(rows) and len(parsed) > 0,
            "parsed=%d printed=%d (a parser matching nothing would print 0 and pass)"
            % (len(parsed), len(rows)))
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


def run(out, args):
    v = Verifier()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = bool(args.device)
    layout = DA.parse_se_layout()
    rt = DA.parse_rt_layout()
    cfg = BF.load_geometry()
    torch.set_num_threads(int(cfg["threads"]))
    dumps = BF.Dumps()
    band = G.load_band(BF.BAND_PATH)
    digest = G.band_digest(band)
    limits, _raw, missing = BG.prototype_limits()
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
               limits=limits, population=population, st=st, slab=slab, rt_slab=rt_slab, work=work,
               device=device, x_bytes=x_bytes, pair=(n_bytes, n_diff), rt_pairs=rt_pairs)

    if device:
        register(arm_runs, "hip_device", DA.run_arm(args.arm_binary, slab["plan"], slab["slab"],
                                                    work / "out_device.bin", device=True,
                                                    label="hip_device"))
        register(rt_runs, "rt_hip_device", DA.run_arm(args.rt_arm_binary, rt_slab["plan"],
                                                      rt_slab["slab"], work / "rt_out_device.bin",
                                                      device=True, label="rt_hip_device"))
        rt_pairs.append(dict(label="rt_hip_device_vs_rt_hip_hostpath", bytes=0, ndiff=0,
                             device=True))
        rec_bytes, rec_diff = DA.pair_bytes(work / "rt_out_device.bin", work / "rt_out_host.bin")
        host_f, _ = DA.rt_record(work / "rt_out_host.bin", rt)
        dev_f, _ = DA.rt_record(work / "rt_out_device.bin", rt)
        claimed = ("ids", "ids_alt", "wpre", "wcast")
        ctx["rt_devdiff"] = dict(
            full=(rec_bytes, rec_diff),
            claimed=dict((k, (int((np.ascontiguousarray(dev_f[k]).view(np.uint32)
                                    != np.ascontiguousarray(host_f[k]).view(np.uint32)).sum()),
                               int(len(dev_f[k])))) for k in claimed))
        print("PAIR pair=rt_hip_device_vs_rt_hip_hostpath file=router_record.f32 bytes=%d "
              "ndiff_bytes=%d claimed_fields=%s claimed_ndiff=%s verdict=%s note=reported, not banded: "
              "the claimed fields are the ones the licences measure at exactness, and a difference on "
              "the long-dot logits or probabilities is the prototype header's documented exp() scope, "
              "not a tolerance to absorb"
              % (rec_bytes, rec_diff, ",".join(claimed),
                 ",".join("%s=%d/%d" % (k, v[0], v[1]) for k, v in
                          sorted(ctx["rt_devdiff"]["claimed"].items())),
                 "CLAIMED-IDENTICAL" if all(v[0] == 0 for v in ctx["rt_devdiff"]["claimed"].values())
                 else "CLAIMED-DIFFER"))
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
    control_rt_f32_span(v, ctx)
    control_rt_poison(v, ctx)
    control_claim_scan(v, ctx)
    checks(v, ctx, rows, tally, out.text(), parsed, arm_runs)
    checks_rt(v, ctx, rt_runs)

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
