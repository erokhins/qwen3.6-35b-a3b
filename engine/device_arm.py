#!/usr/bin/env python3
"""device_arm.py — the staging-and-launch half of M2's device path, split out so the driver that
adjudicates stage rows (`engine/block_device.py`) is not also the code that calls HIP.

Two jobs, both narrow:

  *stage*   allocate the layout's slabs and bulk-copy the staging runs that carry layer 0's
            shared-expert tensors, using `engine/stage_run.py`'s own `HipStaging`, `ByteSource`,
            `Stager`, `verify_run`, `stage_run_to_device` and `device_read_back`. This file adds no
            allocator, no second digest and no read-back spelling of its own: `CountedHip` subclasses
            `SR.HipStaging` and only counts, and `StagedSet.extract` reads a device span with
            `HipStaging.from_device` + `.view`, the two calls M1's own read-back uses. A span that
            does not hash the same on the device as it did in the file is refused
            (`device-readback-disagrees`), because then the kernel would be judged on bytes the
            staging path never certified.
  *launch*  run one of the two compiled arms (`engine/device/build/m2_se_arm`, `m2_se_host`) on a
            plan and slab written from `engine/loader.py`'s addresses, and turn its report into
            records. A missing report line, a non-zero exit, or a launch count of zero in a device run
            raises `Refused` with a named tag; the caller prints `REFUSED tag=…` and exits 1 without
            producing any stage row. That is what makes a silent fallback to the host twin impossible
            rather than merely unlikely.

Nothing here runs unless a caller asks, nothing here takes a lock (a self-locking driver deadlocks
silently — `pitfalls/50`), and no clock is read anywhere in this file.
"""

import ast
import ctypes
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

import stage_run as SR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SE_HEADER = ROOT / "prototype" / "shared_expert" / "shared_expert_kernels.h"
RT_HEADER = ROOT / "prototype" / "moe_router" / "moe_router_kernels.h"
RT_KERNEL_HEADER = HERE / "device" / "m2_rt_kernels.h"
BUILD = HERE / "device" / "build"
ARM_BIN = BUILD / "m2_se_arm"
HOST_BIN = BUILD / "m2_se_host"


class Refused(Exception):
    """A claim this process is not entitled to make. `tag` names it so a control can require it."""

    def __init__(self, tag, msg):
        super().__init__("[%s] %s" % (tag, msg))
        self.tag = tag


class CountedHip(SR.HipStaging):
    """`engine/stage_run.py`'s staging device with the calls counted, nothing else changed.

    Subclassing rather than re-implementing is the point: every allocation, bulk copy, read-back and
    free is the proven call, and the only addition is the tally the driver must print before it is
    allowed to say anything about residency. `pitfalls/10` §Allocation-recipe is the recipe
    `HipStaging` already follows (one `hipMalloc` per slab, never `Managed`; one bulk copy per run).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tally = {"allocs": 0, "slab_bytes": 0, "h2d_calls": 0, "h2d_bytes": 0,
                      "d2h_calls": 0, "d2h_bytes": 0, "frees": 0}

    def alloc_slab(self, name, nbytes):
        p = super().alloc_slab(name, nbytes)
        self.tally["allocs"] += 1
        self.tally["slab_bytes"] += int(nbytes)
        return p

    def to_device(self, dev_ptr, nbytes):
        super().to_device(dev_ptr, nbytes)
        self.tally["h2d_calls"] += 1
        self.tally["h2d_bytes"] += int(nbytes)

    def from_device(self, dev_ptr, nbytes):
        super().from_device(dev_ptr, nbytes)
        self.tally["d2h_calls"] += 1
        self.tally["d2h_bytes"] += int(nbytes)

    def free_all(self):
        self.tally["frees"] += len(self.ptrs) + (1 if self.stage is not None else 0)
        return super().free_all()

    def read_back(self, dev_ptr, nbytes):
        """D2H exactly `nbytes` at `dev_ptr`; returns (bytes, sha256, count read).

        `staging buffer -> view(0, n)` is `stage_run.device_read_back`'s own two steps, so the repo
        has one read-back spelling. The count is returned and compared by the caller: a check that
        cannot see a short read cannot fail.
        """
        self.from_device(dev_ptr, nbytes)
        data = self.view(0, nbytes)
        return data, hashlib.sha256(data).hexdigest(), len(data)


def parse_se_layout():
    """The prototype lane's record layout and geometry, read out of its header, never typed here.

    Cross-checks the record against itself: field offsets must ascend with no hole, each field's
    length must reach the next field's offset, and the last must end exactly at `SE_CASE_FLOATS`. A
    header edit that makes the record lie to itself therefore goes red here instead of producing a
    plausible-looking stage row.
    """
    text = SE_HEADER.read_text()
    defines = {n: int(v) for n, v in
               re.findall(r"^#define\s+(SE_[A-Z0-9_]+)\s+(\d+)\b", text, re.M)}
    want = ["SE_H", "SE_I", "SE_TWO_I", "SE_Q8_ELTS", "SE_Q8_BYTES", "SE_FIELD_COUNT",
            "SE_CASE_FLOATS", "SE_OFF_GATE", "SE_OFF_UP", "SE_OFF_ACT", "SE_OFF_RAW",
            "SE_OFF_OUT", "SE_OFF_PROB", "SE_N_GATE", "SE_N_UP", "SE_N_ACT", "SE_N_RAW",
            "SE_N_OUT", "SE_N_PROB"]
    # Two of the geometry constants are written as expressions in the header, not as digits. They are
    # derived here from the operands AND text-matched against the expression, so a header that changed
    # the division is caught instead of silently re-evaluated.
    derived = {"SE_GATE_BLOCKS": ("SE_H", "SE_Q8_ELTS"), "SE_DOWN_BLOCKS": ("SE_I", "SE_Q8_ELTS")}
    for name, (lhs, rhs) in derived.items():
        pat = r"^#define\s+%s\s+\(\s*%s\s*/\s*%s\s*\)" % (name, lhs, rhs)
        if not re.search(pat, text, re.M):
            raise Refused("se-layout-unparsed",
                          "%s does not spell %s as `(%s / %s)`; a geometry constant this file cannot "
                          "read is one it cannot judge" % (SE_HEADER, name, lhs, rhs))
        defines[name] = defines[lhs] // defines[rhs]
    missing = [w for w in want if w not in defines]
    if missing:
        raise Refused("se-layout-unparsed",
                      "%s defines none of %s; a layout this file cannot parse is a layout it cannot "
                      "judge" % (SE_HEADER, ",".join(missing)))
    order = [("gate", "SE_OFF_GATE", "SE_N_GATE"), ("up", "SE_OFF_UP", "SE_N_UP"),
             ("act", "SE_OFF_ACT", "SE_N_ACT"), ("raw", "SE_OFF_RAW", "SE_N_RAW"),
             ("out", "SE_OFF_OUT", "SE_N_OUT"), ("prob", "SE_OFF_PROB", "SE_N_PROB")]
    cursor, holes = 0, []
    for name, off, ln in order:
        if defines[off] != cursor:
            holes.append("%s@%d!=cursor%d" % (name, defines[off], cursor))
        cursor = defines[off] + defines[ln]
    if holes or cursor != defines["SE_CASE_FLOATS"] or len(order) != defines["SE_FIELD_COUNT"]:
        raise Refused("se-layout-inconsistent",
                      "the record's fields do not tile it: %s; last_offset=%d SE_CASE_FLOATS=%d "
                      "fields=%d declared=%d" % (holes or "no holes", cursor,
                                                 defines["SE_CASE_FLOATS"], len(order),
                                                 defines["SE_FIELD_COUNT"]))
    defines["fields"] = order
    return defines


def read_span(source, art_path, file_off, length, why):
    """One artifact span read through the audit-gated source, with its digest and byte count."""
    got, pos = b"", 0
    while pos < length:
        want = min(SR.CHUNK, length - pos)
        chunk = source.raw_read(art_path, file_off + pos, want)
        if len(chunk) != want:
            raise Refused("span-short", "%s: asked %d B at +%d of %s, got %d B"
                          % (why, want, pos, art_path, len(chunk)))
        got += chunk
        pos += len(chunk)
    return got, hashlib.sha256(got).hexdigest(), len(got)


def residency(loader, stager, layout):
    """Where layer 0's shared-expert tensors live, and which staging run carries each one.

    Addresses come from `Loader.handle`; the covering run is found by asking the plan's coordinates
    which run's slab range contains the tensor's, never by re-deriving the merge that produced the
    runs (`stage_run.Stager._resolve_members` is the same discipline). Exactly one covering run per
    tensor is required. The per-role element counts are the geometry the arm will claim, so a tensor
    that is not the size the record layout implies is refused here and not in a stage row.
    """
    want_elts = {"fused": layout["SE_I"] * layout["SE_H"], "down": layout["SE_H"] * layout["SE_I"],
                 "sg": layout["SE_H"]}
    out = []
    for role, names in (("fused", ("blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight")),
                        ("down", ("blk.0.ffn_down_shexp.weight",)),
                        ("sg", ("blk.0.ffn_gate_inp_shexp.weight",))):
        for name in names:
            h = loader.handle(name)
            hits = [(i, r) for i, r in enumerate(stager.plan)
                    if r["slab"] == h["slab"] and r["slab_off"] <= h["slab_off"]
                    and h["slab_off"] + h["length"] <= r["slab_end"]]
            if len(hits) != 1:
                raise Refused("residency-cover",
                              "%s lies inside %d staging runs of slab %s; exactly one is required, "
                              "or the digest chain has no single owner"
                              % (name, len(hits), h["slab"]))
            idx, run = hits[0]
            delta = h["slab_off"] - run["slab_off"]
            if run["file_off"] + delta != h["file_off"]:
                raise Refused("residency-address",
                              "%s: slab delta %d lands at file offset %d but the loader names %d; "
                              "those are two different files" % (name, delta,
                                                                run["file_off"] + delta,
                                                                h["file_off"]))
            need = want_elts[role]
            if int(h["n_elts"]) != need:
                raise Refused("residency-geometry",
                              "%s stores %s elements; the record layout the arm will claim needs %d "
                              "for role %s" % (name, h["n_elts"], need, role))
            out.append(dict(role=role, name=name, slab=h["slab"], slab_off=h["slab_off"],
                            file_off=h["file_off"], length=h["length"], type_name=h["type_name"],
                            n_elts=int(h["n_elts"]), run=idx, run_delta=delta,
                            run_off=run["file_off"], run_len=run["length"]))
    return out


def arm_slab(spans, x_bytes, layout):
    """Lay the arm's slab: fused (gate then up), down, sg, x. Returns (bytes, offsets).

    The concatenation is an address arrangement and nothing more — the two `fused` halves are the
    bytes of the two policy-A tensors, in that order, which is exactly the row map the prototype
    declares (§11.4, gate first). Every byte is digest-bound to its source span by the caller.
    """
    buf = bytearray()
    off = {}
    by_role = {}
    for s in spans:
        by_role.setdefault(s["role"], []).append(s)
    for role in ("fused", "down", "sg"):
        start = len(buf)
        for s in by_role[role]:
            buf += s["bytes"]
        off[role] = (start, len(buf) - start)
    off["x"] = (len(buf), len(x_bytes))
    buf += x_bytes
    rows = {"fused": layout["SE_TWO_I"], "down": layout["SE_H"], "sg": 1}
    cols = {"fused": layout["SE_H"], "down": layout["SE_I"], "sg": layout["SE_H"]}
    return bytes(buf), off, rows, cols


def write_plan(path, off, rows, cols, layout, slab_bytes, fault="none"):
    lines = ["M2SEPLAN version=1", "SLAB bytes=%d" % slab_bytes]
    for role in ("fused", "down", "sg"):
        o, n = off[role]
        lines.append("SPAN role=%s off=%d bytes=%d rows=%d cols=%d"
                     % (role, o, n, rows[role], cols[role]))
    o, n = off["x"]
    lines.append("SPAN role=x off=%d bytes=%d rows=1 cols=%d" % (o, n, layout["SE_H"]))
    lines.append("FORM mode=correct sgate=scalar cast=before")
    lines.append("FAULT name=%s" % fault)
    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines) - 2


def run_arm(binary, plan, slab, out, device=False, fault=None, label=None):
    """Run one compiled arm and parse its report. Any gap in the report is a Refused.

    The arm prints `HDR`, `PLAN`, `RUN`, `WROTE`, `RESULT`. Each is parsed back off the text; the run
    is accepted only when all five appeared, the arm exited 0, the `RESULT` verdict is `OK`, and —
    for a device run — the launch and call counts are non-zero. A device run whose counts are zero is
    the exact shape of "answered --device with a host run", which is why the count, not the flag, is
    the evidence.
    """
    if not Path(binary).exists():
        raise Refused("arm-binary-absent",
                      "%s does not exist; build it with engine/device/build.sh and rerun" % binary)
    argv = [str(binary), "--plan", str(plan), "--slab", str(slab), "--out", str(out)]
    if device:
        argv.append("--device")
    if fault:
        argv += ["--fault", fault]
    proc = subprocess.run(argv, capture_output=True, text=True)
    recs = {}
    for line in proc.stdout.splitlines():
        head = line.split(" ", 1)[0]
        if head in ("HDR", "PLAN", "RUN", "WROTE", "RESULT"):
            recs.setdefault(head, []).append(dict(f.split("=", 1) for f in line.split()[1:]
                                                 if "=" in f))
    missing = [k for k in ("HDR", "PLAN", "RUN", "WROTE", "RESULT") if not recs.get(k)]
    if proc.returncode != 0 or missing:
        tag = "arm-fault-%s" % (missing[0].lower() if missing else "exit-%d" % proc.returncode)
        raise Refused("arm-exec-failed",
                      "%s %s: exit=%d missing=%s stderr=%s — a failed arm is a red run, never a host "
                      "substitution" % (Path(binary).name, " ".join(argv[4:]), proc.returncode,
                                        missing or "none", proc.stderr.strip()[:200] or "clean"))
    run = recs["RUN"][-1]
    verdict = recs["RESULT"][-1].get("verdict")
    if verdict != "OK":
        raise Refused("arm-verdict-%s" % (verdict or "missing"),
                      "%s exited 0 but its RESULT line says verdict=%s; an arm that reports failure "
                      "while exiting clean is a report bug and gets treated as one"
                      % (Path(binary).name, verdict or "missing"))
    if device and (int(run.get("launches", 0)) == 0 or int(run.get("hip_calls", 0)) == 0):
        raise Refused("arm-device-claim-uncounted",
                      "%s answered --device with hip_calls=%s launches=%s; a device run that cannot "
                      "count a launch did not launch"
                      % (Path(binary).name, run.get("hip_calls"), run.get("launches")))
    return dict(hdr=recs["HDR"][-1], plan=recs["PLAN"][-1], run=run, wrote=recs["WROTE"][-1],
                result=recs["RESULT"][-1], label=label or Path(binary).name,
                stdout=proc.stdout, argv=argv)


def pair_bytes(out_a, out_b):
    """(n_bytes, n_differing) between two arms' records — the compiled pair at zero tolerance."""
    a = Path(out_a).read_bytes()
    b = Path(out_b).read_bytes()
    if len(a) != len(b):
        return min(len(a), len(b)), abs(len(a) - len(b)) + sum(1 for x, y in zip(a, b) if x != y)
    return len(a), sum(1 for x, y in zip(a, b) if x != y)


def device_ptr(base, delta):
    """`base + delta` on a raw address, never on a typed pointer (`stage_run._shift`'s reason)."""
    return ctypes.c_void_p(base.value + int(delta))


# ----------------------------------------------------------------------------------------------
# The router slice: the same staging machinery, a second compiled pair, a second record layout.
# ----------------------------------------------------------------------------------------------

RT_ARM_BIN = BUILD / "m2_rt_arm"
RT_HOST_BIN = BUILD / "m2_rt_host"


def _c_defines(text, prefix):
    """`#define NAME <expr>` for names starting with `prefix`, expressions left as text.

    Block comments are stripped first (the prototype writes its geometry comments on the same line as
    the constants), and only integer-literal-and-`+ - * / ( )` expressions over already-read names are
    accepted: an expression the evaluator below cannot represent is refused, not guessed at.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return dict(re.findall(r"^#define\s+(%s[A-Z0-9_]+)\s+(.+?)\s*$" % prefix, text, re.M))


def _c_eval(expr, env, where):
    """Evaluate an integer C-ish constant expression, or refuse. `ast`, not `eval`: the grammar that
    is allowed is right here (literals, names, `+ - * /`), and anything else — a call, a comparison,
    a ternary — is a layout this file will not guess."""
    node = ast.parse(expr.strip(), mode="eval").body   # SyntaxError escapes; the caller catches it
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b}

    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise Refused("rt-layout-unparsed",
                              "%s uses `%s`, which is not a constant this file read" % (where, n.id))
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](walk(n.left), walk(n.right))
        raise Refused("rt-layout-unparsed",
                      "%s: `%s` is outside the integer-expression grammar (literals, names, + - * /)"
                      % (where, expr))

    return walk(node)


def _c_resolve(defines, env, where):
    """Resolve as many `NAME -> expression` pairs as possible, in dependency order.

    A define that cannot be evaluated (a function-like macro, a cast, a string) is skipped, not an
    error: what matters is that every name a *record offset* depends on resolves, and the caller asks
    for those by name and refuses if one is missing.
    """
    todo = dict(defines)
    while todo:
        progress = False
        for name in sorted(todo):
            try:
                env[name] = _c_eval(todo[name], env, where)
                progress = True
            except (Refused, SyntaxError, ValueError, TypeError, ZeroDivisionError):
                continue
            del todo[name]
        if not progress:
            break
    return env


def parse_rt_layout():
    """The router's geometry from the prototype lane's header, and the record layout from the
    engine's composition header — both read out of the files that define them, never typed here.

    `MR_E`, `MR_K`, `MR_H` are the prototype's own constants (`prototype/moe_router/
    moe_router_kernels.h` §geometry) and `M2_RT_OFF_*`/`M2_RT_FLOATS` are the engine's, written as
    expressions over the prototype's names so the two cannot silently drift. The check that ties them
    together is arithmetic: the field offsets must tile the record with no hole and no overlap, each
    field's length must be what the geometry implies (E for the logit fields, K for the weight and id
    fields), and the last field must end exactly at `M2_RT_FLOATS`. A header edit that lets the record
    lie about itself goes red here instead of producing a plausible-looking `topk_weights` row.
    """
    proto = _c_defines(RT_HEADER.read_text(), "MR_")
    eng = _c_defines(RT_KERNEL_HEADER.read_text(), "M2_RT_")
    env = _c_resolve(proto, {}, RT_HEADER.name)
    for name in ("MR_E", "MR_K", "MR_H"):
        if name not in env:
            raise Refused("rt-layout-unparsed",
                          "%s does not define %s as an integer constant this file can read; a "
                          "geometry it cannot read is one it cannot judge" % (RT_HEADER, name))
    _c_resolve(eng, env, RT_KERNEL_HEADER.name)
    e, k = env["MR_E"], env["MR_K"]
    want = [("M2_RT_OFF_LOGITS", e), ("M2_RT_OFF_PROBS", e), ("M2_RT_OFF_WPRE", k),
            ("M2_RT_OFF_WCAST", k), ("M2_RT_OFF_IDS", k), ("M2_RT_OFF_LOGITS_F32", e),
            ("M2_RT_OFF_WLEAK", k), ("M2_RT_OFF_IDS_ALT", k)]
    fields = {}
    for name, length in want:
        if name not in eng:
            raise Refused("rt-layout-unparsed",
                          "%s does not define %s; the record layout is read from that file, not "
                          "re-typed here" % (RT_KERNEL_HEADER, name))
        fields[name] = (_c_eval(eng[name], env, RT_KERNEL_HEADER.name), length)
    if "M2_RT_FLOATS" not in eng:
        raise Refused("rt-layout-unparsed", "%s defines no M2_RT_FLOATS; the record has no end"
                      % RT_KERNEL_HEADER.name)
    total = _c_eval(eng["M2_RT_FLOATS"], env, RT_KERNEL_HEADER.name)
    seq = sorted(fields.items(), key=lambda kv: kv[1][0])
    bad, cursor = [], 0
    for name, (off, ln) in seq:
        if off != cursor:
            bad.append("%s@%d(expected %d)" % (name, off, cursor))
        cursor = off + ln
    if bad or total != cursor:
        raise Refused("rt-layout-inconsistent",
                      "the router record's fields do not tile it: bad=%s, fields end at %d but "
                      "M2_RT_FLOATS=%d" % (bad or "none", cursor, total))
    short = {"M2_RT_OFF_LOGITS": "logits", "M2_RT_OFF_PROBS": "probs", "M2_RT_OFF_WPRE": "wpre",
             "M2_RT_OFF_WCAST": "wcast", "M2_RT_OFF_IDS": "ids",
             "M2_RT_OFF_LOGITS_F32": "logits_f32", "M2_RT_OFF_WLEAK": "wleak",
             "M2_RT_OFF_IDS_ALT": "ids_alt"}
    return dict(MR_E=e, MR_K=k, MR_H=env["MR_H"], RT_FLOATS=total,
                fields=dict((short[n], fields[n]) for n, _ in seq),
                order=[short[n] for n, _ in seq])


def residency_router(loader, stager, rt):
    """Where layer 0's router matrix lives, and a width check that is not taken on faith.

    The three router rows are the only rows in this pass that a *licensed* band covers on a policy-A
    arm, and they are covered for one reason only: the router weights are stored F32-exact (D-01a), so
    the arm's arithmetic and the pinned arm's arithmetic are over the same numbers. If that premise
    fails, the licence is unearned, so the premise is checked against the artifact here: declared
    `F32`, `n_elts == E*H`, and a byte length of exactly `4 * n_elts`. A Q8_0 or bf16 copy of the same
    tensor is refused by name before any kernel is offered a pointer to it.
    """
    name = "blk.0.ffn_gate_inp.weight"
    h = loader.handle(name)
    need = rt["MR_E"] * rt["MR_H"]
    if h["type_name"] != "F32":
        raise Refused("router-width-not-exact",
                      "%s is stored %s, not F32: the router family's licence rests on the weights "
                      "being exact, so a requantized artifact must not be judged under it"
                      % (name, h["type_name"]))
    if int(h["n_elts"]) != need:
        raise Refused("router-geometry", "%s stores %s elements; the geometry read from %s says %d"
                      % (name, h["n_elts"], RT_HEADER.name, need))
    if int(h["length"]) != 4 * need:
        raise Refused("router-span-width",
                      "%s spans %s bytes for %s elements; an F32 matrix of that shape spans %d, and "
                      "a span that is narrower is a requantized copy wearing the same name"
                      % (name, h["length"], h["n_elts"], 4 * need))
    hits = [(i, r) for i, r in enumerate(stager.plan)
            if r["slab"] == h["slab"] and r["slab_off"] <= h["slab_off"]
            and h["slab_off"] + h["length"] <= r["slab_end"]]
    if len(hits) != 1:
        raise Refused("residency-cover",
                      "%s lies inside %d staging runs of slab %s; exactly one is required"
                      % (name, len(hits), h["slab"]))
    idx, run = hits[0]
    delta = h["slab_off"] - run["slab_off"]
    if run["file_off"] + delta != h["file_off"]:
        raise Refused("residency-address",
                      "%s: slab delta %d lands at file offset %d but the loader names %d"
                      % (name, delta, run["file_off"] + delta, h["file_off"]))
    return dict(name=name, slab=h["slab"], slab_off=h["slab_off"], file_off=h["file_off"],
                length=int(h["length"]), type_name=h["type_name"], n_elts=int(h["n_elts"]),
                run=idx, run_delta=delta)


def write_rt_plan(path, off_w, n_w, off_x, n_x, rt, slab_bytes, mode="correct", cast="act_dtype",
                  skip="none", tie_forge=0, fault="none"):
    """The router arm's plan file. `elt=4` is written explicitly because the width claim is the one
    thing this slice must not assume; `control_rt_f32_span` asks the arm to refuse a plan that says
    otherwise."""
    lines = ["M2RTPLAN version=1 slab_bytes=%d floats=%d" % (slab_bytes, rt["RT_FLOATS"]),
             "ROUTER span_off=%d bytes=%d rows=%d cols=%d elt=4 top_k=%d"
             % (off_w, n_w, rt["MR_E"], rt["MR_H"], rt["MR_K"]),
             "X span_off=%d bytes=%d cols=%d" % (off_x, n_x, rt["MR_H"]),
             "FORM mode=%s cast=%s skip=%s tie_forge=%d fault=%s"
             % (mode, cast, skip, tie_forge, fault)]
    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines)


def rt_record(path, rt):
    """The arm's router record: per-field float lists and the poison census, size-checked first."""
    words = np.fromfile(str(path), dtype="<f4")
    if words.size != rt["RT_FLOATS"]:
        raise Refused("rt-record-size", "%s holds %d floats; the layout read from %s says %d"
                      % (path, words.size, RT_KERNEL_HEADER.name, rt["RT_FLOATS"]))
    out, nan = {}, {}
    for name, (off, ln) in rt["fields"].items():
        field = words[off:off + ln]
        out[name] = field.copy()
        bits = field.view(np.uint32)
        nan[name] = int(np.count_nonzero((bits & np.uint32(0x7FFFFFFF)) == np.uint32(0x7FC00000)))
    return out, nan


# ----------------------------------------------------------------------------------------------
# The routed-expert gather slice: two width paths, one plan format, one record.
# ----------------------------------------------------------------------------------------------

G_ARM_BIN = BUILD / "m2_g_arm"
G_HOST_BIN = BUILD / "m2_g_host"
G_HEADER = HERE / "device" / "m2_g_kernels.h"
MG_HEADER = ROOT / "prototype" / "moe_gather" / "moe_gather_kernels.h"
G_PATHS = ("ref_bf16", "policy_a")

# The three routed families, named once. The artifact spelling is policy A's; the HF spelling is the
# reference path's, and the two are not interchangeable: one expert is a Q4_K span in the first and a
# bf16 matrix image in the second, which is why `m2_g_io.h` refuses a plan that mixes them.
G_ART_TENSOR = {"gate": "blk.0.ffn_gate_exps.weight", "up": "blk.0.ffn_up_exps.weight",
                "down": "blk.0.ffn_down_exps.weight"}
G_HF_TENSOR = {"gate_up": "model.language_model.layers.0.mlp.experts.gate_up_proj",
               "down": "model.language_model.layers.0.mlp.experts.down_proj"}
G_KIND_BY_ART_TYPE = {"Q4_K": "q4k", "Q5_K": "q5k", "Q6_K": "q6k"}
G_ROLES = ("gate", "up", "down")


def parse_g_layout():
    """The gather's geometry from the prototype lane's header and its record layout from the engine's.

    Same discipline as `parse_rt_layout`: `MG_H/MG_I/MG_K/MG_E/MG_QK_K` and the three block byte counts
    come from `prototype/moe_gather/moe_gather_kernels.h`, the field offsets from
    `engine/device/m2_g_kernels.h`, and neither is typed in here. The tiling check is the point — the
    five fields must cover the record with no hole and no overlap, and each field's length must be what
    the geometry implies — because a record that lies about its own shape produces a plausible-looking
    stage row out of garbage bytes.
    """
    proto = _c_defines(MG_HEADER.read_text(), "MG_")
    eng = _c_defines(G_HEADER.read_text(), "M2G_")
    env = _c_resolve(proto, {}, MG_HEADER.name)
    for name in ("MG_H", "MG_I", "MG_K", "MG_E", "MG_TWO_I", "MG_QK_K", "MG_Q4K_BYTES",
                 "MG_Q5K_BYTES", "MG_Q6K_BYTES"):
        if name not in env:
            raise Refused("g-layout-unparsed",
                          "%s does not define %s as an integer constant this file can read" %
                          (MG_HEADER, name))
    _c_resolve(eng, env, G_HEADER.name)
    h, i, k = env["MG_H"], env["MG_I"], env["MG_K"]
    want = [("M2G_OFF_GATE", k * i), ("M2G_OFF_UP", k * i), ("M2G_OFF_S", k * i),
            ("M2G_OFF_RAW", k * h), ("M2G_OFF_ROUTED", h)]
    fields = {}
    for name, length in want:
        if name not in eng:
            raise Refused("g-layout-unparsed", "%s does not define %s; the record layout is read from "
                                               "that file, not re-typed here" % (G_HEADER, name))
        fields[name] = (_c_eval(eng[name], env, G_HEADER.name), length)
    if "M2G_FLOATS" not in eng:
        raise Refused("g-layout-unparsed", "%s defines no M2G_FLOATS; the record has no end" % G_HEADER)
    total = _c_eval(eng["M2G_FLOATS"], env, G_HEADER.name)
    seq = sorted(fields.items(), key=lambda kv: kv[1][0])
    short = {"M2G_OFF_GATE": "gate", "M2G_OFF_UP": "up", "M2G_OFF_S": "s", "M2G_OFF_RAW": "raw",
             "M2G_OFF_ROUTED": "routed"}
    bad, cursor = [], 0
    for name, (off, ln) in seq:
        if off != cursor:
            bad.append("%s@%d(expected %d)" % (name, off, cursor))
        cursor = off + ln
    if bad or total != cursor or len(seq) != 5:
        raise Refused("g-layout-inconsistent",
                      "the gather record's fields do not tile it: bad=%s, fields end at %d but "
                      "M2G_FLOATS=%d (fields=%d)" % (bad or "none", cursor, total, len(seq)))
    block_bytes = {"q4k": env["MG_Q4K_BYTES"], "q5k": env["MG_Q5K_BYTES"], "q6k": env["MG_Q6K_BYTES"]}
    return dict(MG_H=h, MG_I=i, MG_K=k, MG_E=env["MG_E"], MG_TWO_I=env["MG_TWO_I"],
                MG_QK_K=env["MG_QK_K"], BLOCK_BYTES=block_bytes, G_FLOATS=total,
                fields=dict((short[n], fields[n]) for n, _ in seq),
                order=[short[n] for n, _ in seq])


def _g_pad(buf, align=8):
    """Pad a slab builder to `align`, so an f32 or i32 field is never handed a misaligned offset."""
    while len(buf) % align:
        buf += b"\0"
    return buf


def g_span_digest(buf):
    """The digest of a slab's bytes — the same hash the staging path uses, so a span bound printed here is
    comparable with the one `stage_run` printed for the same bytes."""
    return hashlib.sha256(bytes(buf)).hexdigest()


def g_record(path, g):
    """The arm's gather record: per-field float lists and the unwritten-slot census, size-checked first.

    The NaN census is returned alongside the fields because it is the load-bearing half: a slot the arm
    never wrote reads the poison pattern on both paths, and a bulk memset that silently cleared it is the
    defect class this repo has already been bitten by (`engine/HANDOFF.md` on the `lm_head` sentinel).
    """
    words = np.fromfile(str(path), dtype="<f4")
    if words.size != g["G_FLOATS"]:
        raise Refused("g-record-size", "%s holds %d floats; the layout read from %s says %d"
                      % (Path(path).name, words.size, G_HEADER.name, g["G_FLOATS"]))
    out, nan = {}, {}
    for name, (off, ln) in g["fields"].items():
        field = words[off:off + ln]
        out[name] = field.copy()
        bits = field.view(np.uint32)
        nan[name] = int(np.count_nonzero((bits & np.uint32(0x7FFFFFFF)) == np.uint32(0x7FC00000)))
    return out, nan


def g_ref_spans(g, shard_index, shard_dir, layer=0):
    """Where each routed expert's bf16 matrix bytes live in the HF shards, from the shard headers only.

    No tensor is loaded: the safetensors header's `data_offsets` give the byte range, and one expert is
    one contiguous slice of it (the expert axis is the outer one, so the slice is `e * expert_bytes`
    into the tensor). The dtype is checked to be BF16 here rather than assumed, because the reference
    path's licence rests on these being the bytes the pinned dumps were made from — the same argument
    `residency_router` makes for the router's F32 matrix. The shard index and directory come in as
    arguments: `engine/block_forward.py` owns those paths and this file does not re-state them.
    """
    index = json.loads(Path(shard_index).read_text())["weight_map"]
    out = {}
    for which, hf in G_HF_TENSOR.items():
        name = hf.replace(".layers.0.", ".layers.%d." % layer)
        shard = Path(shard_dir) / index[name]
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        meta = head.get(name)
        if meta is None:
            raise Refused("g-ref-tensor-absent", "%s is not in %s" % (name, shard.name))
        if meta["dtype"] != "BF16":
            raise Refused("g-ref-width-not-bf16",
                          "%s is %s in %s; the reference path is licensed on the bf16 image the pins "
                          "were made from, so a different storage width is a different artifact"
                          % (name, meta["dtype"], shard.name))
        begin, end = meta["data_offsets"]
        out[which] = dict(path=str(shard), name=name, dtype=meta["dtype"], shape=list(meta["shape"]),
                          data_start=8 + n + begin, bytes=end - begin)
    e, two_i, h, i = g["MG_E"], g["MG_TWO_I"], g["MG_H"], g["MG_I"]
    if out["gate_up"]["shape"] != [e, two_i, h]:
        raise Refused("g-ref-geometry", "gate_up_proj is %s; prediction 3 says [E,2I,H]=[%d,%d,%d], and "
                                        "a different axis order is a different gather"
                      % (out["gate_up"]["shape"], e, two_i, h))
    if out["down"]["shape"] != [e, h, i]:
        raise Refused("g-ref-geometry", "down_proj is %s; expected [E,H,I]=[%d,%d,%d]"
                      % (out["down"]["shape"], e, h, i))
    for which, rows, cols in (("gate_up", two_i, h), ("down", h, i)):
        per = rows * cols * 2
        if out[which]["bytes"] != e * per:
            raise Refused("g-ref-span-width", "%s spans %d B for %d experts; %d B per expert implies %d"
                          % (which, out[which]["bytes"], e, per, e * per))
        out[which]["expert_bytes"] = per
        out[which]["rows"], out[which]["cols"] = rows, cols
    return out


def g_read_file_span(path, off, length, why):
    """One byte span out of a shard, with its digest — the same shape as `read_span` for the artifact."""
    with open(path, "rb") as fh:
        fh.seek(off)
        got = fh.read(length)
    if len(got) != length:
        raise Refused("span-short", "%s: asked %d B at %d of %s, got %d B" % (why, length, off,
                                                                             Path(path).name, len(got)))
    return got, hashlib.sha256(got).hexdigest(), len(got)


def g_ref_units(g, spans, ids):
    """The reference path's unit list: one fused gate|up span and one down span per selected expert.

    The gate and up units name the SAME span (fused=1) at different offsets, which is what the artifact
    actually is ([2I,H], gate first, §11.4 and AGENTS.md prediction 3), and `m2_g_io.h` refuses the pair
    unless both views agree on `span_off`/`span_bytes` and disagree on `off`.
    """
    units = []
    i, h = g["MG_I"], g["MG_H"]
    gu, dn = spans["gate_up"], spans["down"]
    half = gu["expert_bytes"] // 2
    for slot, e in enumerate(ids):
        for role in ("gate", "up"):
            units.append(dict(role=role, slot=slot, expert=int(e), kind="bf16", elt=2, rows=i, cols=h,
                              blocks=0, fused=1, whole=0,
                              off=gu["data_start"] + int(e) * gu["expert_bytes"]
                                 + (0 if role == "gate" else half),
                              bytes=half,
                              span_off=gu["data_start"] + int(e) * gu["expert_bytes"],
                              span_bytes=gu["expert_bytes"],
                              src=gu["path"], src_name=gu["name"]))
        units.append(dict(role="down", slot=slot, expert=int(e), kind="bf16", elt=2, rows=h, cols=i,
                          blocks=0, fused=0, whole=0,
                          off=dn["data_start"] + int(e) * dn["expert_bytes"],
                          bytes=dn["expert_bytes"],
                          span_off=dn["data_start"] + int(e) * dn["expert_bytes"],
                          span_bytes=dn["expert_bytes"],
                          src=dn["path"], src_name=dn["name"]))
    return units


def g_policy_units(g, loader, source, art_path, ids):
    """Policy A's unit list: one artifact expert span per slot per family, from `Loader.expert_range`.

    The addresses are the loader's, so the gather and the loader cannot disagree about where an expert
    begins (`Loader.expert_range` cross-checks itself against `layout.expert_slice` for exactly that
    reason). The kind word comes from the record's own ggml type, never from a table of what policy A is
    expected to hold, so an artifact re-quantized to a different width shows up as a `path-kinds` refusal
    in the arm instead of as a row that silently means something else. Bytes come through
    `read_span`/`ByteSource`, the same audit-gated door M1 staged through.

    `span_off`/`span_bytes` are the EXPERT's own range, not the family's: the census counts distinct spans
    and a per-family span made 24 expert reads look like 3 spans, which is the quantity prediction 3's
    scheduler decision is priced on. The reference path gets this from `g_ref_units` (one span per fused
    [2I,H] pair, one per expert down); policy A has three tensors, so it is three spans per slot.
    """
    units = []
    i, h = g["MG_I"], g["MG_H"]
    geom = {"gate": (i, h), "up": (i, h), "down": (h, i)}
    for role in G_ROLES:
        name = G_ART_TENSOR[role]
        rec = loader.by_name.get(name)
        if rec is None:
            raise Refused("g-tensor-absent", "%s is not in the resident set, so there is nothing to "
                                            "gather and no licence to read" % name)
        rows, cols = geom[role]
        blocks = (rows * cols) // g["MG_QK_K"]
        for slot, e in enumerate(ids):
            r = loader.expert_range(name, int(e))
            kind = G_KIND_BY_ART_TYPE.get(r["type_name"])
            if kind is None:
                raise Refused("g-kind-unknown",
                              "%s expert %d is stored %s, which this arm has no decoder for; the arm "
                              "would refuse it too, later and less clearly" % (name, e, r["type_name"]))
            if r["length"] != blocks * g["BLOCK_BYTES"][kind]:
                raise Refused("g-span-width", "%s expert %d spans %d B; %d blocks of %d at %s is %d B"
                              % (name, e, r["length"], blocks, g["MG_QK_K"], kind,
                                 blocks * g["BLOCK_BYTES"][kind]))
            data, sha, got = read_span(source, art_path, r["file_off"], r["length"],
                                       "%s expert %d" % (name, e))
            units.append(dict(role=role, slot=slot, expert=int(e), kind=kind, elt=1, rows=rows,
                              cols=cols, blocks=blocks, fused=0, whole=0, off=r["file_off"],
                              bytes=r["length"], span_off=r["file_off"], span_bytes=r["length"],
                              file_off=r["file_off"],
                              file_sha=sha, file_read=got, type_name=r["type_name"], src_name=name,
                              _bytes=data))
    return units


def g_slab(g, units, x_bytes, ids, wcast, wpre, elt_x):
    """Concatenate the gathered spans and the router arm's selection into one slab; assign offsets.

    The slab is an address arrangement, not a tensor: each unit's bytes go in at its own offset, and a
    fused pair shares one span (written once, from the gate unit's bytes). `off`/`span_off` are then
    rewritten to slab coordinates, so the plan describes exactly the bytes the arm will read.
    """
    if len(x_bytes) != g["MG_H"] * elt_x:
        raise Refused("g-x-width", "the activation image is %d B; path %s stores %d B/elt for H=%d, so "
                      "it must be %d B — the activation is not a field to guess at"
                      % (len(x_bytes), elt_x, elt_x, g["MG_H"], g["MG_H"] * elt_x))
    buf = bytearray()
    off, seen_span, span_sha = {}, {}, {}
    for u in units:
        key = (u["src_name"], u.get("span_off"), u.get("span_bytes"))
        if u["fused"]:
            # One span, two views. It is read once, whole, from its own file coordinates, and both the
            # gate and the up unit are then offsets inside it — which is the only honest way to address a
            # fused [2I,H] block, and the arrangement `m2_g_io.h` makes the plan prove.
            if key not in seen_span:
                _g_pad(buf)
                seen_span[key] = len(buf)
                data, sha, got = g_read_file_span(u["src"], u["span_off"], u["span_bytes"],
                                                  "%s expert %d fused pair" % (u["src_name"],
                                                                               u["expert"]))
                buf += data
                span_sha[key] = (sha, got)
            span = seen_span[key]
            u["off"] = span + (u["bytes"] if u["role"] == "up" else 0)
            u["span_off"] = span
            u["file_sha"], u["file_read"] = span_sha[key]
        else:
            data = u.get("_bytes")
            if data is None:
                data, sha, got = g_read_file_span(u["src"], u["off"], u["bytes"],
                                                  "%s expert %d" % (u["src_name"], u["expert"]))
                u["file_sha"], u["file_read"] = sha, got
            else:
                u.setdefault("file_sha", hashlib.sha256(data).hexdigest())
                u.setdefault("file_read", len(data))
            _g_pad(buf)
            u["off"] = u["span_off"] = len(buf)
            u["span_bytes"] = u["bytes"]
            buf += data
        off[(u["slot"], u["role"])] = u["off"]
    _g_pad(buf)
    o_x = len(buf)
    buf += x_bytes
    _g_pad(buf)
    o_ids = len(buf)
    buf += struct.pack("<%di" % len(ids), *ids)
    _g_pad(buf)
    o_wcast = len(buf)
    buf += struct.pack("<%df" % len(wcast), *wcast)
    _g_pad(buf)
    o_wpre = len(buf)
    buf += struct.pack("<%df" % len(wpre), *wpre)
    return bytes(buf), dict(x=(o_x, len(x_bytes)), ids=o_ids, wcast=o_wcast, wpre=o_wpre,
                            spans=seen_span)


def write_g_plan(path, path_word, g, units, slab_bytes, off, elt_x, mode="correct", fault="none",
                 corrupt_off=-1, read_off=-1, read_len=0, whole_bytes=None):
    """The gather plan: one M2GPLAN line, the activation and the router arm's selection, one UNIT line
    per (slot, role), and the FORM line. Nothing else, because nothing else is the arm's to know.

    `whole_bytes` is the control that has to be refused: given a role→bytes map holding the real stacked
    tensor sizes, every unit is rewritten to claim the whole routed tensor instead of one expert, and
    `m2_g_io.h` refuses such a plan by name (`whole-tensor-unit-refused`). The sizes come from the
    metadata, never typed, because a refusal that quotes an invented byte count is a refusal of a fiction.
    A gather that *could* read a whole tensor is the failure AGENTS.md prediction 2 is about, so the arm is
    made structurally unable to, and the census's whole-tensor counter is witnessed non-zero separately by
    the `whole-note` fault leg.
    """
    h, i = g["MG_H"], g["MG_I"]
    lines = ["M2GPLAN version=1 path=%s slots=%d slab_bytes=%d" % (path_word, g["MG_K"], slab_bytes),
             "X span_off=%d bytes=%d cols=%d elt=%d" % (off["x"][0], off["x"][1], h, elt_x),
             "IDS span_off=%d slots=%d" % (off["ids"], g["MG_K"]),
             "WCAST span_off=%d slots=%d" % (off["wcast"], g["MG_K"]),
             "WPRE span_off=%d slots=%d" % (off["wpre"], g["MG_K"])]
    for u in units:
        claim = u["bytes"] if whole_bytes is None else whole_bytes[u["role"]]
        # A fused gate|up pair names ONE span (the [2I,H] expert block) and two unit views inside it, so
        # `span_bytes` is not the unit's own byte count — collapsing the two is what made the honest plan
        # look like a half-span fused unit and got it refused.
        span_bytes = u["span_bytes"] if whole_bytes is None else claim
        lines.append("UNIT role=%s slot=%d expert=%d off=%d bytes=%d rows=%d cols=%d kind=%s elt=%d "
                     "blocks=%d fused=%d whole=%d span_off=%d span_bytes=%d"
                     % (u["role"], u["slot"], u["expert"], u["off"], claim,
                        h if u["role"] == "down" else i, i if u["role"] == "down" else h,
                        u["kind"], u["elt"], u["blocks"],
                        0 if whole_bytes else u["fused"], 1 if whole_bytes else 0,
                        u["span_off"], span_bytes))
    lines.append("FORM mode=%s fault=%s corrupt_off=%d read_off=%d read_len=%d"
                 % (mode, fault, corrupt_off, read_off, read_len))
    Path(path).write_text("\n".join(lines) + "\n")
    return len(units)


def parse_census(text):
    """The arm's census token back into a dict, refusing to pretend a missing field is a zero.

    Every key the arm prints is required here: `pitfalls/40` §a-check-that-can-print-nothing-cannot-fail
    is the reason a census parsed with `.get(k, 0)` would be worse than no census at all — a truncated
    report would read as `whole_reads=0` and the whole-tensor check would pass on an arm that printed
    nothing. `bytes_by_expert` comes back as {expert: (units, bytes)} and must be non-empty.
    """
    want = ("slots", "unit_reads", "want_units", "distinct_experts", "want_experts", "whole_reads",
            "elt_reads", "route_n", "route_bad", "note_conflict", "total", "want_bytes", "spans")
    got, per = {}, {}
    for chunk in (text or "").strip().strip('"').split(";"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        if k == "bytes_gathered":
            for item in v.split(","):
                if ":" in item and "/" in item:
                    e, rest = item.split(":", 1)
                    n, b = rest.split("/")
                    per[int(e)] = (int(n), int(b.rstrip("B")))
            continue
        got[k] = v.rstrip("B")
    missing = [k for k in want if k not in got]
    if missing:
        raise Refused("census-unparsed",
                      "the arm's census carried none of %s; a census this file cannot parse cannot "
                      "refuse anything either (census=%r)" % (",".join(missing), text))
    if not per:
        raise Refused("census-empty",
                      "the arm reported a census with no per-expert rows at all: %r. A counter that "
                      "printed nothing is not a counter reading zero" % (text,))
    out = dict((k, int(got[k])) for k in want)
    out["bytes_by_expert"] = per
    return out


def g_census(g, units, ids, arm=None):
    """The driver-side read census, keyed on expert_index, from the plan the driver itself wrote.

    Three claims, all of which a whole-tensor read would break and none of which a summary count would
    show: the number of distinct expert indices gathered equals the number of slots (a gather that read
    every expert would show 256 here); every unit's span is one expert's matrix, never the stacked tensor;
    and the bytes per expert are the quantized or bf16 size the artifact says, not a rounded guess.
    `arm` is the arm's own census dict when a run produced one, and the two are required to agree — the
    count must not be merely the driver's opinion of what it asked for.
    """
    per_expert, spans = {}, {}
    for u in units:
        pe = per_expert.setdefault(u["expert"], dict(units=0, bytes=0, roles=[]))
        pe["units"] += 1
        pe["bytes"] += u["bytes"]
        pe["roles"].append(u["role"])
        spans.setdefault((u["src_name"], u["span_off"], u["span_bytes"]), 0)
    whole = [u for u in units if u["bytes"] > 8 * 1024 * 1024]
    rows, disagree = [], []
    for e in sorted(per_expert):
        pe = per_expert[e]
        rows.append(dict(expert=e, units=pe["units"], bytes=pe["bytes"], roles=",".join(pe["roles"])))
        if arm is not None:
            a = arm["bytes_by_expert"].get(e)
            if a is None:
                disagree.append("%d:absent-in-arm" % e)
            elif a != (pe["units"], pe["bytes"]):
                disagree.append("%d:driver=%d/%d arm=%d/%d" % (e, pe["units"], pe["bytes"], a[0], a[1]))
    if arm is not None:
        if arm["distinct_experts"] != len(per_expert):
            disagree.append("distinct=%d/%d" % (arm["distinct_experts"], len(per_expert)))
        if arm["unit_reads"] != len(units):
            disagree.append("units=%d/%d" % (arm["unit_reads"], len(units)))
        if arm["spans"] != len(spans):
            disagree.append("spans=%d/%d" % (arm["spans"], len(spans)))
        if arm["whole_reads"] != len(whole):
            disagree.append("whole=%d/%d" % (arm["whole_reads"], len(whole)))
        if arm["total"] != sum(u["bytes"] for u in units):
            disagree.append("bytes=%d/%d" % (arm["total"], sum(u["bytes"] for u in units)))
        if arm["note_conflict"]:
            disagree.append("note_conflict=%d" % arm["note_conflict"])
    return dict(distinct=len(per_expert), slots=len(ids), units=len(units), spans=len(spans),
                bytes_total=sum(u["bytes"] for u in units),
                bytes_per_expert={e: per_expert[e]["bytes"] for e in per_expert},
                whole_tensor_reads=len(whole), rows=rows, disagree=disagree,
                arm_bytes=(arm["total"] if arm else None),
                arm_units=(arm["unit_reads"] if arm else None),
                arm_spans=(arm["spans"] if arm else None),
                arm_whole_reads=(arm["whole_reads"] if arm else None),
                note_conflict=(arm["note_conflict"] if arm else None))
