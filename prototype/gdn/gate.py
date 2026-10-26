#!/usr/bin/env python3
"""gate.py — the GDN slice-1 gate: three arms, the derived band, the exact read checks, and the
honesty controls, in one CPU-only driver.

    <venv>/bin/python prototype/gdn/gate.py [--only three-arm|band|exact|controls|witness|oracle]
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
              for `load_band`, `band_check_row`, `aggregates` and the geometry — the comparators and
              the band are the oracle's, the numbers being compared are the arms'.
  kernel      this lane's C, compiled twice: `host_probe` (g++) and `gdn` (hipcc, executed on the
              CPU, `hip_calls=0`). The pair is compared at ZERO tolerance to itself; a disagreement
              between two compilers on one source is a finding, never something to band.

METRICS, AND WHICH ONES DECIDE
  relL2            PRIMARY for arm-to-arm agreement (scale-free)
  ndiff            PRIMARY; load-bearing for the compiled pair and for every exactness check
  band             PRIMARY against the dump, and ONLY through `gdn_ref.band_check_row()` over the
                   four keys the probe derived (max_abs, per_head_max, mean_rel, one_minus_cos)
  maxabs           REPORTED ONLY, never load-bearing
  cos              REPORTED ONLY, never load-bearing
  Why the demotion is not decoration: `speccheck/results-gdn.md` and `prototype/shared_expert/work/
  derive.log` (same box, 2026-10-26) both record a WRONG candidate that was NEARER on the absolute
  metric than the honest one (maxabs 0.961426 vs 0.96875), and `cos` prints 1 to nine digits while
  1263 of 1884 `lm_head` words differ in their bit image. An absolute metric that cannot see what is
  being claimed is not conservative, it is blind.

DEVICE POLICY: this file touches no device, reads no timer, and holds no device lock of any kind —
the wrapper is the sole acquirer and there is nothing here for it to wrap. Every printed counter is
a count; no line this prints carries a duration, a token rate or a bandwidth.
"""
from __future__ import annotations

import argparse
import json
import os
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

# The authoritative band digest: `band_digest(load_band(BAND_PATH))` as measured 2026-10-26. It is
# pinned here so a silently regenerated band cannot be judged against itself. If `load_band()`
# refuses a file, the file is wrong; if the digest differs, THIS constant is stale and the human
# who regenerated the band updates it — the gate does not bend to the file.
BAND_DIGEST = "27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee"

STAGES = ["q_hat", "k_hat", "scan_out", "core_attn", "state", "normed", "y"]
STAGE_IS_BF16 = {"q_hat": False, "k_hat": False, "scan_out": False, "core_attn": True,
                 "state": False, "normed": True, "y": True}
# the eight loaded fields, in the arm's echo order, then the field it fingerprints instead
FIELDS = ["q_pre", "k_pre", "v_pre", "beta", "g_decay", "z", "norm_weight", "out_weight"]
ECHO_FIELDS = FIELDS[:7]
FP_FIELD = "out_weight"
SECTIONS = ["three-arm", "band", "exact", "controls", "witness", "oracle"]


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
        raise Refuse(f"pinned prefill T={T} exceeds one chunk of 64; slice 1 cannot express it")
    print("DUMP %s T=%d records=%d blocks=%s n_chunks=1 pad=%d"
          % (m["lane"], T, len(m["dumps"]), ",".join(blocks), 64 - T))
    return m, recs, blocks, T


def dump_tensor(recs, block, mode, key):
    return DG.read_dump_tensor(recs[(block, mode, key)])


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
    """cases: list of dicts(block, name, T, mode, wslot). Returns the declared case count."""
    lines = ["#GDNPLAN v1 n_cases=%d T=%d" % (len(cases), cases[0]["T"])]
    for i, c in enumerate(cases):
        lines.append("CASE idx=%d block=%s name=%s T=%d mode=%s wslot=%d"
                     % (i, c["block"], c["name"], c["T"], c["mode"], c["wslot"]))
    path.write_text("\n".join(lines) + "\n")
    return len(cases)


def write_inputs(path: Path, per_case_fields, T: int):
    """inputs.bin = per case: the six fields (one u32 word per element), then the h_v*d_k*d_v
    zero initial state. Slice 1 REFUSES a nonzero state, so the zeros are what makes 'the S term
    contributed exactly zero' a checked fact of the case rather than an assumption."""
    buf = []
    for fields in per_case_fields:
        for w in fields:
            buf.append(w.astype(np.uint32))
        buf.append(np.zeros(32 * 128 * 128, dtype=np.uint32))
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
    """CASE and FP lines, with the parsed count required to equal the declared count. A run that
    printed nothing can only ever have had nothing to say, so an empty parse refuses (`pitfalls/40`)."""
    cases, fps, run = [], {}, None
    for ln in res["lines"]:
        if ln.startswith("CASE "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
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
    g = geom()
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
            print("ARM arm=%s cases=%d compiler=%s hip_calls=%s refused=%s stage_words=%s"
                  % (name, len(cs), run.get("compiler"), run.get("hip_calls"),
                     run.get("refused"), run.get("case_words")))
            if name == "hip_hostpath":
                if run.get("arm") != "hip_hostpath":
                    fails.append("the device TU did not announce arm=hip_hostpath (said %r)"
                                 % run.get("arm"))
                if run.get("hip_calls") != "0":
                    fails.append("the device-compiled arm issued hip_calls=%s; the CPU arm must "
                                 "issue none, or it is not the arm this comparison claims"
                                 % run.get("hip_calls"))
            if int(run.get("refused", "1")) != 0:
                fails.append("a case refused to run (nonzero initial state)")
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
                ref = dump_tensor(recs, block, "prefill", st).reshape(-1)
                if mine.numel() != ref.numel():
                    raise Refuse("%s/%s: kernel emitted %d words, the dump holds %d elements"
                                 % (block, st, mine.numel(), ref.numel()))
                mine = mine.reshape(ref.shape)
                m = metrics(mine, ref)
                ag = G.aggregates(mine, ref, name="%s/pinned/prefill|%s" % (block, st), stage=st)
                ok, why = G.band_check_row(band, st, ag)
                n_rows += 1
                line = ("STAGE arm=kernel_vs_dump block=%s stage=%s %s band=%s meanrel=%.3e "
                        "1-cos=%.3e perhead=%.3e(h%d/%d) verdict=%s why=%s"
                        % (block, st, fmt(m), "ACCEPTED" if ok else "REJECTED", ag["mean_rel"],
                           ag["one_minus_cos"], ag["per_head_max"], ag["worst_head"],
                           ag["n_heads"], "PASS" if ok else "FAIL", why))
                print(line)
                if not ok:
                    fails.append("%s/%s outside the derived band: %s" % (block, st, why))
        counts["stage_rows"] = n_rows
        if n_rows != len(blocks) * len(STAGES):
            fails.append("stage rows %d != derived %d" % (n_rows, len(blocks) * len(STAGES)))
        if n_rows == 0:
            raise Refuse("no stage rows: a banded verdict over nothing is a constant predicate")

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
        counts["controls"] = 4 if BROKEN.exists() else 3

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

    # ------------------------------------------------------------------ verdict
    print("COUNTS %s" % " ".join("%s=%d" % kv for kv in sorted(counts.items())))
    print("METRICS primary=relL2+ndiff+band_check_row reported_only=maxabs,cos "
          "maxabs_load_bearing=0 cos_load_bearing=0 band_source=band_check_row_only")
    if fails:
        print("GATE FAIL findings=%d" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    if not counts:
        raise Refuse("no section ran; a gate that checked nothing cannot pass")
    missing = [s for s in SECTIONS if s not in ran]
    if missing:
        print("GATE PARTIAL ran=%s missing=%s note=a partial run does not claim PASS: the three-arm "
              "claim needs the dump arm, the oracle CLI arm and the kernel arm, and one of them is "
              "not in this run" % (",".join(ran), ",".join(missing)))
        return 2
    print("GATE PASS arms=3 stages=%d blocks=%d T=%d band_digest=%s device=none timing=taken=no"
          % (len(STAGES), len(blocks), T, digest[:16]))
    return 0


def _require_three_arms(arms):
    if len(arms) < 3:
        raise Refuse("three-arm agreement needs three arms; found %d (%s)"
                     % (len(arms), ",".join(arms)))


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
