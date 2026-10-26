"""M2 block forward: layer 0 of Qwen3.6-35B-A3B, one token, two weight arms.

WHAT THIS IS. `engine/block_forward.py` is the forward half of M2 in the milestone ladder
(`AGENTS.md` §GOAL): one decoder block's three named subsystems — the GDN recurrence, the routed
MoE (top-8 of 256), and the shared expert — computed for ONE token on real bytes, ending at that
block's mixer outputs. It is a library plus a CPU driver; the adjudication (band verdicts, checks,
controls, the `BLOCKGATE` line) lives in `engine/block_gate.py`, which imports this file.

THE DISPATCH IS NOT A CHOICE. §04b.1 (line 104) gives one row per shape, and one token with a
cached state is a DECODE step: the GDN side is a single recurrent step in the §04.6 order, not a
1-column chunk. The MoE side is one row of the pinned MoE input. `gdn_ref.probe_arms_for("decode",
"pinned")` is what the gate cites for the arm the licence names; this file implements the program
that dispatch runs.

TWO WEIGHT ARMS, ONE PROGRAM. Both arms run the same arithmetic; only the bytes differ:

  arm=policy_a   the engine's own artifact read through `engine/loader.py` + `engine/layout.py`
                 addresses and dequantized by `tools/gguf_reader.py`: experts Q4_K (gate/up) and
                 Q5_K (down) sliced per expert with `Loader.expert_range`, attention/GDN/shared
                 Q8_0, router / shared gate / norms F32 exact (D-01, D-01a, D-08). This is the
                 shipping shape and the only shape the engine may ship.
  arm=ref_bf16   the same program on the bf16 weights the pinned dumps were made from (the weight
                 records of the two manifests, plus the two expert tensors sliced out of the local
                 safetensors shards). The reason it exists is that every band in this repo
                 measures PROGRAM error on one width: the licensed row `recurrent|pinned` in
                 `speccheck/evidence/gdn-band-2026-10-26.json` was probed with both arms reading
                 the bf16 weights. A policy-A row therefore cannot be judged by it at all, and
                 pretending otherwise would spend a band on an error budget it never measured
                 (`pitfalls/30` §sizes bands against an exact reference). Running both arms is what
                 lets the gate say "our program reproduces pinned inside the licensed band, and our
                 width is unbanded" instead of collapsing the two.

WHAT IS NOT COMPUTED, AND WHY. §03.2's steps 3 and 6 add the mixer outputs to the residual stream.
Neither M0 manifest pins the layer-0 residual stream: each pins a mixer INPUT (a synthetic RMS-1
tensor, `specification/10` §10.1 [A-01]) and that mixer's output. There is no pinned `x` to add,
and inverting a RMSNorm to manufacture one changes the quantity being compared, which is exactly
what `specification/03` §03.3 forbids. So the two residual adds are emitted as stages with no
referent and the gate cannot credit them; that is a named gap in the fixture material, not a bug
in this file.

DEVICE CLAIMS. Nothing here touches a device, and nothing here is allowed to pretend otherwise.
`--device` is refused with the tag `live-device-path-absent`: this file allocates nothing, launches
nothing and imports no HIP. What it *can* do is admit M1's already-proven staging evidence
(`engine/stage_run.py --device --verify`, whose read-back log is the artifact) through
`staged_precondition()`, and that is labelled in the output as a PRECONDITION on staging integrity —
it re-hashes the artifact regions the log claims were read back and refuses, by a named tag, if the
log is not about the file this process is reading. A precondition is not a device result and the gate
prints no device-shaped verdict for it (`engine/block_gate.py::device-claims-come-from-device-only`).
This file contains no HIP call, no `hip` import and no device lock; `grep -c` for the lock script
name is 0 here by design (a driver that self-acquires a lock and is called by a caller already
holding it deadlocks silently — `pitfalls/50`).

NO TIMINGS. Nothing in this file reads a clock, and the gate that drives it prints no `ms`, no
`t/s` and no `GB/s` (`pitfalls/00` §"A sub-millisecond timer is not a ruler").
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "speccheck"), str(ROOT / "specification")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gdn_ref as G                                  # noqa: E402  (the band machinery, reused)
import gguf_reader as GR                             # noqa: E402
import engine.loader as LD                           # noqa: E402

CONFIG_PATH = ROOT / "checkpoints" / "Qwen3.6-35B-A3B.config.json"
MOE_MANIFEST = ROOT / "checkpoints" / "dump-manifest-2026-10-26.json"
GDN_MANIFEST = ROOT / "checkpoints" / "dump-manifest-gdn-2026-10-26.json"
SHARD_INDEX = ROOT / "checkpoints" / "Qwen3.6-35B-A3B.safetensors.index.json"
BAND_PATH = G.BAND_PATH
M1_EVIDENCE = ROOT / "specification" / "evidence" / "stage-run-device-readback-2026-10-26.log"
GGUF_PIN = ROOT / "checkpoints" / "Qwen3.6-35B-A3B.gguf-pin.json"

BLOCK = 0                                             # layer 0: layer_types[0] == linear_attention
TOKEN_ROW = 0                                         # the single token of the MoE pass
GDN_PASS = "layer0"                                   # the GDN manifest's block name
GDN_MODE = "decode0"                                  # 1 token, cached conv window + state
GDN_PASS_KEY = "layer0/pinned/decode0"                # the band file's pass key shape
MOE_PASS_KEY = "layer0/pinned/row0"                   # this gate's own key for the MoE row

_DTYPE = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32,
          "torch.int64": torch.int64}
_ITEMSIZE = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int64": 8}

# The GGUF permutes every V axis from HF's grouped order to tiled order
# (specification/04 §04.9, ASSUMPTIONS A-12): GGUF V head j holds HF V head (j % h_k) * exp +
# (j // h_k). Reading a policy-A V axis without this map pairs half the V heads with the wrong K
# head and raises nothing, which is the trap P-03 records.
def v_order_gguf_to_hf(h_v, h_k, expansion, head_dim):
    """Index vector such that `hf = gguf[v_order]` puts a GGUF V axis back in HF's order."""
    idx = torch.arange(h_v)
    hf_of_gg = (idx % h_k) * expansion + (idx // h_k)
    order = torch.empty(h_v, dtype=torch.long)
    order[hf_of_gg] = idx
    return order.repeat_interleave(head_dim)


class Refused(Exception):
    """A claim this file is not allowed to make. Carries a tag so a control can name it."""

    def __init__(self, tag, msg):
        super().__init__("%s: %s" % (tag, msg))
        self.tag = tag


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        left = limit
        while left is None or left > 0:
            chunk = fh.read(1 << 20 if left is None else min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
    return h.hexdigest()


class Dumps:
    """The two pinned dump sets, read by manifest record and sha-verified on every read.

    `reading_rules` in each manifest is the contract this implements: raw little-endian element
    bytes, and a bf16 payload is a bit image — read the uint16 and bitcast, never cast a uint16 to
    a float (`pitfalls/30` §Read a dump's dtype). A record whose file cannot hold its own declared
    shape is refused rather than truncated.
    """

    def __init__(self):
        self.sets = {}
        for lane, path, base in (("moe", MOE_MANIFEST, ROOT), ("gdn", GDN_MANIFEST, ROOT)):
            doc = json.loads(path.read_text())
            idx = {}
            for rec in doc["dumps"]:
                idx[rec["name"]] = dict(rec, _base=(base / rec["file"]).resolve(), _lane=lane)
            self.sets[lane] = idx
        self.files_read = 0
        self.bytes_read = 0

    def rec(self, name):
        for idx in self.sets.values():
            if name in idx:
                return idx[name]
        raise Refused("no-dump-record", "no pinned record named %r" % name)

    def names(self, lane):
        return sorted(self.sets[lane])

    def stage_names(self, lane, block, modes=None, arms=("ref", "spec")):
        """Stage records for one block, derived from the manifest, never listed by hand.

        `arm` is how a manifest distinguishes the reference dump from the exact-f64 and spec-form
        dumps of the same tensor_key; the default pair is what the engine's stages can be compared
        against. Records whose name has no stage part (weights, inputs) are not stages.
        """
        out = []
        for name, rec in self.sets[lane].items():
            if rec["block"] != block or rec.get("arm") not in arms:
                continue
            stage = rec["tensor_key"]
            if stage in ("input",) or stage.endswith(".weight"):
                continue
            if modes is not None and rec.get("mode") not in modes:
                continue
            out.append(stage)
        return sorted(set(out))

    def tensor(self, name, row=None, rows=None):
        rec = self.rec(name)
        path = rec["_base"]
        n = 1
        for d in rec["shape"]:
            n *= int(d)
        want = n * _ITEMSIZE[rec["dtype"]]
        if path.stat().st_size != want:
            raise Refused("dump-size-mismatch",
                          "%s is %d B on disk, manifest says %d B for %s %s"
                          % (name, path.stat().st_size, want, rec["shape"], rec["dtype"]))
        digest = sha256_file(path)
        if digest != rec["sha256"]:
            raise Refused("dump-digest-mismatch",
                          "%s sha256 %s, manifest pins %s" % (name, digest[:16], rec["sha256"][:16]))
        self.files_read += 1
        self.bytes_read += want
        dt = _DTYPE[rec["dtype"]]
        if dt is torch.bfloat16:
            t = torch.from_numpy(np.fromfile(str(path), dtype="<u2")).view(torch.bfloat16)
        elif dt is torch.float32:
            t = torch.from_numpy(np.fromfile(str(path), dtype="<f4")).clone()
        else:
            t = torch.from_numpy(np.fromfile(str(path), dtype="<i8")).clone()
        t = t.reshape(rec["shape"])
        if rows is not None:
            return t[rows[0]:rows[1]].contiguous()
        if row is not None:
            return t[row].reshape((1,) + tuple(t.shape[1:])).contiguous()
        return t.contiguous()


class PolicyA:
    """Policy-A weights, addressed by the engine's own loader and read with the audit token.

    Addresses come from `Loader.handle` (whole tensors) and `Loader.expert_range` (one routed
    expert), so the block reads exactly the byte ranges M1's staging plan stages, and the gathered
    expert slices stay in the `per_token_gathered` read class `engine/layout.py` claims for them.
    Dequantization is `tools/gguf_reader.dequant`, the audited reader's own decoder; the widths the
    artifact actually stores are printed per read and checked against policy A.
    """

    WIDTH = {                                        # policy A, as D-01/D-01a/D-08 fixed it
        "blk.0.attn_qkv.weight": "Q8_0", "blk.0.attn_gate.weight": "Q8_0",
        "blk.0.ssm_out.weight": "Q8_0", "blk.0.ssm_conv1d.weight": "F32",
        "blk.0.ssm_alpha.weight": "F32", "blk.0.ssm_beta.weight": "F32",
        "blk.0.ssm_dt.bias": "F32", "blk.0.ssm_norm.weight": "F32", "blk.0.ssm_a": "F32",
        "blk.0.ffn_gate_inp.weight": "F32", "blk.0.ffn_gate_inp_shexp.weight": "F32",
        "blk.0.ffn_gate_shexp.weight": "Q8_0", "blk.0.ffn_up_shexp.weight": "Q8_0",
        "blk.0.ffn_down_shexp.weight": "Q8_0",
        "blk.0.ffn_gate_exps.weight": "Q4_K", "blk.0.ffn_up_exps.weight": "Q4_K",
        "blk.0.ffn_down_exps.weight": "Q5_K",
    }

    def __init__(self, faults=()):
        self.faults = set(faults or ())
        self.L = LD.open_loader()
        self.reads = []                              # provenance: one dict per weight read
        self._tok = None
        self._hdr = None
        self.last_type = None                        # ggml type of the last read, as the loader named it
        self.basis = torch.float32                   # dequantized values are read as f32

    def close(self):
        if self._hdr is not None:
            self._hdr.close()
            self._hdr = None

    def region_sha(self, file_off, length, chunk=1 << 22):
        """sha256 over exactly `length` artifact bytes at `file_off`, and how many bytes that was.

        Reads through the same gated reader (and the same audit token) every weight read uses, in
        chunks, so an evidence check can prove a staging log describes *this* file without holding a
        whole 184 MB expert tensor in memory.
        """
        h = hashlib.sha256()
        left, done = int(length), 0
        while left > 0:
            n = min(chunk, left)
            buf = self._bytes(file_off + done, n, "<artifact-region>",
                              why="staging-evidence digest re-derivation")
            h.update(buf)
            done += n
            left -= n
        return h.hexdigest(), done

    def _earn_token(self):
        if self._tok is not None:
            return self._tok
        import audit_published_quant as APQ

        class _Section:
            section = "engine/block_forward.py policy-A weight reads"

        mark = APQ.require_selftest(_Section())
        if mark is None:
            raise Refused("data-read-not-granted",
                          "%s refused the selftest token, so not one tensor-data byte is read. This "
                          "file reuses the audit reader's gate rather than building a second one "
                          "(engine/README.md §Not established)" % APQ.SELFTEST_MARK)
        self._tok = APQ.SELFTEST_TOKEN
        self._hdr = self.L.art._hdr
        return self._tok

    def _bytes(self, file_off, length, name, why):
        tok = self._earn_token()
        if "wrong_offset_weight_read" in self.faults:
            file_off += 1                # the CONTROL: one byte of shift, every block misaligned
        buf = self._hdr.read(tok, file_off, length, why=why)
        if len(buf) != length:
            raise Refused("short-weight-read", "%s: asked %d B at %d, got %d"
                          % (name, length, file_off, len(buf)))
        return buf

    def tensor(self, name, why="M2 block forward"):
        h = self.L.handle(name)
        want = self.WIDTH.get(name)
        if want is not None and h["type_name"] != want:
            raise Refused("policy-a-width", "%s: artifact stores %s, policy A declares %s"
                          % (name, h["type_name"], want))
        vals = GR.dequant(self._bytes(h["file_off"], h["length"], name, why),
                          h["type_name"], h["n_elts"])
        arr = np.array(vals, dtype=np.float32).reshape(tuple(reversed(h["dims"])))
        self.last_type = h["type_name"]
        self.reads.append(dict(name=name, type=h["type_name"], kind="tensor", bytes=h["length"],
                               n_elts=h["n_elts"], read_class=h["read_class"]))
        return torch.from_numpy(arr)

    def expert(self, name, idx, shape_hf, why="M2 block forward"):
        """One expert's matrix, dequantized, in HF `[N, K]` row-major order.

        `Loader.expert_range` answers with a contiguous range because the GGUF's expert axis is its
        slowest axis; the elements it returns are the expert's `[N, K]` slab in row-major order, so
        `shape_hf` must be that slab (no transpose, no strided guess)."""
        r = self.L.expert_range(name, idx)
        vals = GR.dequant(self._bytes(r["file_off"], r["length"], name, why),
                          r["type_name"], r["elts"])
        if int(np.prod(shape_hf)) != r["elts"]:
            raise Refused("expert-shape", "%s expert %d: range holds %d elements, %s wants %d"
                          % (name, idx, r["elts"], tuple(shape_hf), int(np.prod(shape_hf))))
        arr = np.array(vals, dtype=np.float32).reshape(tuple(shape_hf))
        self.last_type = r["type_name"]
        self.reads.append(dict(name=name, type=r["type_name"], kind="expert", bytes=r["length"],
                               n_elts=r["elts"], expert=idx, read_class="per_token_gathered"))
        return torch.from_numpy(arr)

    def widths(self):
        return sorted(self.WIDTH)


# Every V-axis of the published file is in tiled order and every V-axis of the dumps is in HF
# grouped order (specification/04 §04.9, ASSUMPTIONS A-12). The engine could work in tiled order and
# pair K head `j % h_k` with V head `j`; this arm instead un-permutes on read so one order is in
# force inside the block, which is what §04.9 warns about: pairing and order must come from the same
# assumption. `axis` is which tensor axis the V axis is, `start` where the V part begins on it, and
# `head` the elements per V head.
V_ORDERED = {
    "blk.0.attn_qkv.weight": ("row", 4096, 128),
    "blk.0.attn_gate.weight": ("row", 0, 128),
    "blk.0.ssm_alpha.weight": ("row", 0, 1),
    "blk.0.ssm_beta.weight": ("row", 0, 1),
    "blk.0.ssm_dt.bias": ("flat", 0, 1),
    "blk.0.ssm_a": ("flat", 0, 1),
    "blk.0.ssm_conv1d.weight": ("row", 4096, 128),
    "blk.0.ssm_out.weight": ("col", 0, 128),
}


class Weights:
    """The block's weights under one width policy, addressed by engine names.

    `basis` is the dtype the block's matmuls run in for this arm. A policy-A tensor is read as the
    dequantized f32 values (`f16 scale × int` is not bf16-representable), so its matmuls run in F32;
    a bf16 arm runs the activation-dtype matmul the reference writes. That difference is the point
    of two arms, and it is also why no policy-A stage row may be judged by a band that measured
    program error on one width.
    """

    def __init__(self, arm, cfg, faults=(), ctx="forward"):
        self.arm = arm
        self.cfg = cfg
        self.faults = set(faults or ())
        # 'forward' for the run that produces rows; 'control:<name>' for every control.
        self.ctx = ctx
        self.reads = []
        self._cache = {}
        if arm == "policy_a":
            self.basis = torch.float32
            self.pa = PolicyA(faults=self.faults)
        elif arm == "ref_bf16":
            self.basis = torch.bfloat16
            self.pa = None
            self.dumps = Dumps()
            self._shards = {}
        else:
            raise Refused("unknown-arm", "arm must be policy_a or ref_bf16, got %r" % arm)

    # ---- name maps ---------------------------------------------------------------------------------
    GDN_NAME = {                       # engine artifact name -> pinned dump weight record
        "blk.0.attn_qkv.weight": "layer0.weights.qkv",
        "blk.0.attn_gate.weight": "layer0.weights.z",
        "blk.0.ssm_alpha.weight": "layer0.weights.a",
        "blk.0.ssm_beta.weight": "layer0.weights.b",
        "blk.0.ssm_dt.bias": "layer0.weights.dt_bias",
        "blk.0.ssm_norm.weight": "layer0.weights.norm",
        "blk.0.ssm_conv1d.weight": "layer0.weights.conv",
        "blk.0.ssm_out.weight": "layer0.weights.out",
        "blk.0.ffn_gate_inp.weight": "layer0.mlp.gate.weight",
        "blk.0.ffn_gate_inp_shexp.weight": "layer0.mlp.shared_expert_gate.weight",
        "blk.0.ffn_gate_shexp.weight": "layer0.mlp.shared_expert.gate_proj.weight",
        "blk.0.ffn_up_shexp.weight": "layer0.mlp.shared_expert.up_proj.weight",
        "blk.0.ffn_down_shexp.weight": "layer0.mlp.shared_expert.down_proj.weight",
    }
    # The routed experts are referenced_not_dumped in the MoE manifest; the reference arm reads the
    # two tensors it names out of the local shards, sliced to the experts this token selected.
    EXPERT_HF = {"gate": "model.language_model.layers.0.mlp.experts.gate_up_proj",
                 "up": "model.language_model.layers.0.mlp.experts.gate_up_proj",
                 "down": "model.language_model.layers.0.mlp.experts.down_proj"}

    def close(self):
        if self.pa is not None:
            self.pa.close()

    def _shard_tensor(self, hf_name):
        from safetensors import safe_open

        if hf_name not in self._shards:
            idx = json.loads(SHARD_INDEX.read_text())["weight_map"]
            shard = ROOT / "models" / "qwen36-src" / idx[hf_name]
            self._shards[hf_name] = (safe_open(str(shard), framework="pt"), hf_name)
        return self._shards[hf_name]

    # ---- the one read path -------------------------------------------------------------------------
    def get(self, name):
        """One whole tensor by engine name, in HF axis order, in this arm's basis dtype."""
        if name in self._cache:
            return self._cache[name]
        if self.pa is not None:
            t = self.pa.tensor(name)
            self.reads.append(dict(name=name, kind="tensor", ctx=self.ctx, dtype=str(t.dtype),
                                   ggml_type=self.pa.last_type))
            if name in V_ORDERED:
                t = _unpermute_v(t, V_ORDERED[name], self.cfg)
        else:
            rec = self.dumps.rec(self.GDN_NAME[name])
            t = self.dumps.tensor(self.GDN_NAME[name])
            self.reads.append(dict(name=name, kind="tensor", ctx=self.ctx, dtype=rec["dtype"],
                                   ggml_type="", source=rec["file"]))
        self._cache[name] = t
        return t

    def expert_slice(self, which, e):
        """One selected expert: `gate`/`up` are `[I_e, d]`, `down` is `[d, I_e]`.

        Policy A reads one gathered expert range (`Loader.expert_range`), which is the read class
        the engine's per-token gather is priced in. The reference arm slices the same expert out of
        the fused HF `gate_up_proj`, taking `chunk(2)` gate-first (§11.4, AGENTS.md prediction 3).
        """
        key = ("expert", which, e)
        if key in self._cache:
            return self._cache[key]
        cfg = self.cfg
        if self.pa is not None:
            name = {"gate": "blk.0.ffn_gate_exps.weight", "up": "blk.0.ffn_up_exps.weight",
                    "down": "blk.0.ffn_down_exps.weight"}[which]
            i = e
            if "swapped_expert_slice" in self.faults and which == "gate":
                i = (e + 1) % cfg["n_experts"]     # the CONTROL: a neighbouring expert's slice
            h = self.pa.L.handle(name)
            shape = {("gate",): (cfg["i_e"], cfg["d"]), ("up",): (cfg["i_e"], cfg["d"]),
                     ("down",): (cfg["d"], cfg["i_e"])}[(which,)]
            t = self.pa.expert(name, i, shape)
            self.reads.append(dict(name=name, kind="expert", ctx=self.ctx, ggml_type=h["type_name"],
                                   dtype=str(t.dtype), expert=e, expert_read=i))
        else:
            hf = self.EXPERT_HF[which]
            fh, name = self._shard_tensor(hf)
            i = e
            if "swapped_expert_slice" in self.faults and which == "gate":
                i = (e + 1) % cfg["n_experts"]     # same fault, same meaning, reference-arm reader
            if which in ("gate", "up"):
                full = fh.get_tensor(name)[i]                 # [2*I_e, d]
                half = full.shape[0] // 2
                t = full[:half] if which == "gate" else full[half:]
            else:
                t = fh.get_tensor(name)[i]
            self.reads.append(dict(name=hf, kind="expert", ctx=self.ctx, ggml_type="",
                                   dtype="torch.bfloat16", expert=e, expert_read=i))
        self._cache[key] = t
        return t

    def a_eff(self):
        """The per-V-head decay coefficient `-exp(A_log)`, the one tensor both arms compute differently.

        The published file stores it precomputed (`blk.0.ssm_a`, §01.6 item 6), and its 1-ulp history
        is governed by D-14 / A-13; the checkpoint stores `A_log` and the reference computes
        `-exp(A_log.float())` (§04.5). Both arms therefore hand back the same *quantity* by two
        routes, and the `g_decay` stage row is where that difference shows up.
        """
        if self.pa is not None:
            return self.get("blk.0.ssm_a")
        return -torch.exp(self.dumps.tensor("layer0.weights.A_log").to(torch.float32))

    def n_reads(self):
        return len(self.reads)


def _unpermute_v(t, spec, cfg):
    """Move one tensor's V axis from GGUF tiled order back to HF grouped order (§04.9)."""
    axis_name, start, head = spec
    order = v_order_gguf_to_hf(cfg["h_v"], cfg["h_k"], cfg["expansion"], head)
    if axis_name == "flat":
        return t.index_select(0, order + start)
    idx = torch.arange(t.shape[1 if axis_name == "col" else 0], dtype=torch.long)
    if axis_name == "row":
        keep = torch.cat([torch.arange(start), order + start,
                          torch.arange(start + cfg["h_v"] * head, t.shape[0])])
        return t.index_select(0, keep)
    keep = torch.cat([torch.arange(start), order + start,
                      torch.arange(start + cfg["h_v"] * head, t.shape[1])])
    return t.index_select(1, keep)



# --------------------------------------------------------------------------- geometry
def load_geometry():
    """Every number the block needs, read from the committed config and the two manifests.

    Nothing here is typed by hand: the GDN widths come from `gdn_ref.load_gdn_geometry` (the config),
    the MoE widths from the MoE manifest's own `geometry` block, and the thread count from
    `interpreter.threads`, because `prototype/moe_router/results.md` §band records that the pinned
    logits reproduce at `ndiff 0` only at the manifest's recorded thread count.

    `act_dtype` is the model's own activation dtype, read from the config's `text_config.dtype` — which
    is also the dtype the pinned `layer0.router_logits` record stores (the gate checks that the two
    derivations agree). It is the dtype R1's logits and R5's cast are defined against: §11.2 states R5 as
    "cast back to the logits' dtype", and on the policy-A arm the arm's compute basis is F32 only
    because dequantized values arrive as F32 — letting that basis stand in for the logits dtype silently
    skips the cast.
    """
    g = G.load_gdn_geometry(CONFIG_PATH)
    act_dtype = getattr(torch, json.loads(CONFIG_PATH.read_text())["text_config"]["dtype"])
    mo = json.loads(MOE_MANIFEST.read_text())
    mg = mo["geometry"]
    cfg = dict(d=g.hidden_size, h_k=g.h_k, h_v=g.h_v, d_k=g.d_k, d_v=g.d_v, k_conv=g.k_conv,
               expansion=g.expansion, key_dim=g.key_dim, value_dim=g.value_dim,
               conv_dim=g.conv_dim, eps_rms=g.rms_norm_eps, l2norm_eps=g.l2norm_eps,
               scan_dtype=getattr(torch, g.mamba_ssm_dtype), act_dtype=act_dtype,
               n_experts=mg["num_experts"],
               top_k=mg["top_k"], i_e=mg["moe_intermediate_size"],
               i_s=mg["shared_expert_intermediate_size"],
               threads=int(mo["interpreter"]["threads"]))
    return cfg


def _linear(x, w):
    """`F.linear` with the activation moved to the weight's dtype — one place, so an arm's dtype
    choice is visible as one expression rather than a per-callsite cast."""
    return F.linear(x.to(w.dtype), w)


def gdn_recurrent_step(x, W, cfg, conv_window, state):
    """One cached decode step of the GDN block, in the order §04.6 pins.

    `x` is the mixer input `[1, 1, d]` (a normed tensor, §03.3), `conv_window` the stored
    `[1, conv_dim, k_conv]` window of mixed_qkv INPUT columns (§04.3 consequence 3, A-09) and
    `state` the `[1, h_v, d_k, d_v]` F32 recurrent state. Stage names and shapes are the dumps'.
    """
    st = {}
    d, h_k, h_v, d_k, d_v = (cfg["d"], cfg["h_k"], cfg["h_v"], cfg["d_k"], cfg["d_v"])
    k_conv = cfg["k_conv"]
    wq = W.get("blk.0.attn_qkv.weight")
    x1 = x.reshape(1, 1, d)
    st["mixed_qkv"] = _linear(x1, wq)                                    # [1,1,conv_dim]
    st["z"] = _linear(x1, W.get("blk.0.attn_gate.weight")).view(1, 1, h_v, d_v)
    st["b_raw"] = _linear(x1, W.get("blk.0.ssm_beta.weight"))             # [1,1,h_v]
    st["a_raw"] = _linear(x1, W.get("blk.0.ssm_alpha.weight"))
    # §04.3: the window is a window of INPUT columns; append this token's column, keep k_conv.
    new = st["mixed_qkv"].transpose(1, 2).to(conv_window.dtype)           # [1,conv_dim,1]
    win = torch.cat([conv_window[:, :, 1:], new], dim=2).contiguous()     # [1,conv_dim,k_conv]
    st["conv_window"] = win
    wc = W.get("blk.0.ssm_conv1d.weight").reshape(cfg["conv_dim"], k_conv)
    conv_pre = F.conv1d(win.to(wc.dtype), wc.reshape(cfg["conv_dim"], 1, k_conv),
                        groups=cfg["conv_dim"])                           # [1,conv_dim,1]
    st["conv_pre"] = conv_pre
    conv_out = F.silu(conv_pre)                                           # silu on all channels
    st["conv_out"] = conv_out
    c = conv_out.transpose(1, 2).reshape(1, 1, cfg["conv_dim"])
    st["q_pre"] = c[..., :cfg["key_dim"]].reshape(1, 1, h_k, d_k).contiguous()
    st["k_pre"] = c[..., cfg["key_dim"]:2 * cfg["key_dim"]].reshape(1, 1, h_k, d_k).contiguous()
    st["v_pre"] = c[..., 2 * cfg["key_dim"]:].reshape(1, 1, h_v, d_v).contiguous()
    st["beta"] = torch.sigmoid(st["b_raw"])
    # §04.5: both casts are explicit, and softplus runs on the SUM a + dt_bias.
    a_eff = W.a_eff().to(torch.float32)
    u = st["a_raw"].to(torch.float32) + W.get("blk.0.ssm_dt.bias").to(torch.float32)
    st["g_decay"] = a_eff.reshape(1, 1, h_v) * F.softplus(u)
    # §04.4: grouped K->V expansion in the activation dtype, then the scan dtype, l2norm, q scale.
    q, k, v = st["q_pre"], st["k_pre"], st["v_pre"]
    if cfg["expansion"] > 1:
        q = q.repeat_interleave(cfg["expansion"], dim=2)
        k = k.repeat_interleave(cfg["expansion"], dim=2)
    sd = cfg["scan_dtype"]
    q = q.transpose(1, 2).to(sd)
    k = k.transpose(1, 2).to(sd)
    v = v.transpose(1, 2).to(sd)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + cfg["l2norm_eps"])
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + cfg["l2norm_eps"])
    q = q / (cfg["d_k"] ** 0.5)
    st["q_hat"], st["k_hat"] = q, k
    kk = k[..., 0, :]                                                     # [1,h_v,d_k], T == 1
    # [1,1,h_v] -> [1,h_v]: a reshape, not `[..., 0, :]`, which would index the head axis by mistake
    # and broadcast ONE head's beta and decay over all h_v (witnessed: every head but head 0 moved).
    beta = st["beta"].to(sd).reshape(1, h_v)
    decay = st["g_decay"].to(sd).reshape(1, h_v)
    S = state.to(sd)
    S = S * torch.exp(decay)[:, :, None, None]                            # step 1: decay first
    kv_mem = (S * kk[:, :, :, None]).sum(dim=2)                           # step 2
    delta = (v[..., 0, :] - kv_mem) * beta[:, :, None]                    # step 3
    S = S + kk[:, :, :, None] * delta[:, :, None, :]                      # step 4
    st["scan_out"] = (S * q[..., 0, :, None]).sum(dim=2).unsqueeze(2)      # [1,h_v,1,d_v]
    st["state"] = S
    core = st["scan_out"].transpose(1, 2)                                 # [1,1,h_v,d_v]
    st["core_attn"] = core.to(x.dtype)
    # §04.7 Form B: variance over d_v, weight applied to the value already cast back, gate in F32.
    f = st["core_attn"].float()
    var = f.pow(2).mean(-1, keepdim=True)
    t = f * torch.rsqrt(var + cfg["eps_rms"])
    wnorm = W.get("blk.0.ssm_norm.weight").reshape(1, 1, 1, d_v)
    gated = wnorm * t.to(st["core_attn"].dtype)
    st["normed"] = (gated * F.silu(st["z"].float())).to(st["core_attn"].dtype)
    st["y"] = _linear(st["normed"].reshape(1, 1, cfg["value_dim"]),
                      W.get("blk.0.ssm_out.weight"))
    return st


def moe_router(g_row, W, cfg):
    """R1-R5 on one token: logits, the F32 softmax, the top-K set, the renormalised weights, and
    R5's cast to the logits' dtype. Split out so a control can recompute just the router.

    TWO dtype points, both of them AGENTS.md prediction 4 and neither of them a tolerance question:

    * R1's result is put INTO the model's activation dtype (`cfg["act_dtype"]`, the config's own
      `text_config.dtype`, which is also the dtype the pinned `layer0.router_logits` record stores).
      On the policy-A arm the dequantized router weight arrives as F32, so leaving R1 in the arm's
      basis makes the softmax see logits that were never rounded and makes R5's cast a no-op.
    * R5 casts the renormalised weights to that dtype BEFORE `moe_routed` multiplies, which is the
      form the pinned `topk_weights` bytes were made by. `prototype/moe_router/results.md`
      §Prediction 4 measured the cast-after alternative (`wpre` carried into the multiply) as a
      different number on 128 of 128 real rows; `topk_weights_f32` is returned alongside it so
      `engine/block_gate.py` prints both forms against the pinned row instead of asserting it.
    """
    x = g_row.reshape(1, cfg["d"])
    logits = wto(_linear(x, W.get("blk.0.ffn_gate_inp.weight")), cfg["act_dtype"])   # R1, model dtype
    probs = torch.softmax(logits.float(), dim=-1)                          # R2, F32
    tw, ids = torch.topk(probs, cfg["top_k"], dim=-1)                      # R3
    wpre = tw / tw.sum(-1, keepdim=True)                                   # R4, F32
    wcast = wto(wpre, logits.dtype)                                        # R5, the cast first
    return dict(router_logits=logits, topk_ids=ids, topk_weights=wcast, topk_weights_f32=wpre)


def moe_shared(g_row, W, cfg):
    """The always-on dense SwiGLU and its scalar sigmoid gate (§11.6): plain add, no scaling
    constant, one scalar per token broadcast over `d`."""
    x = g_row.reshape(1, cfg["d"])
    wg, wu, wd = (W.get("blk.0.ffn_gate_shexp.weight"), W.get("blk.0.ffn_up_shexp.weight"),
                  W.get("blk.0.ffn_down_shexp.weight"))
    st = {}
    st["shared_raw"] = _linear(F.silu(_linear(x, wg)) * _linear(x, wu), wd)
    st["shared_gate_prob"] = torch.sigmoid(_linear(x, W.get("blk.0.ffn_gate_inp_shexp.weight")))
    st["shared_out"] = st["shared_gate_prob"] * st["shared_raw"]
    return st


def moe_routed(g_row, W, cfg, route):
    """The K=8 gathered experts and their sum: `expert_raw` in top-k slot order, `routed_sum`
    accumulated in ASCENDING expert-id order (§11.5)."""
    x = g_row.reshape(1, cfg["d"])
    ids, wcast = route["topk_ids"], route["topk_weights"]
    raws = []
    for slot in range(cfg["top_k"]):                                      # top-k slot order
        e = int(ids[0, slot].item())
        gate = _linear(x, W.expert_slice("gate", e))
        up = _linear(x, W.expert_slice("up", e))
        raws.append(_linear(F.silu(gate) * up, W.expert_slice("down", e)))
    expert_raw = torch.stack(raws, dim=1)                                 # [1, top_k, d]
    acc = torch.zeros(1, cfg["d"], dtype=wcast.dtype)
    for slot in torch.argsort(ids[0]).tolist():                           # ascending expert id
        acc = acc + expert_raw[0, slot] * wcast[0, slot]
    return dict(expert_raw=expert_raw, routed_sum=acc.reshape(1, cfg["d"]))


def moe_block(g_row, W, cfg):
    """The MoE block for one token, in §11.1's order: shared expert, router, routed experts, sum.

    R5's cast to the logits' dtype happens BEFORE the multiply (§11.2, AGENTS.md prediction 4), and
    the routed sum accumulates in ascending expert-id order, reaching each weight through the slot
    the expert holds in the descending top-k order (§11.5).
    """
    st = moe_shared(g_row, W, cfg)
    st.update(moe_router(g_row, W, cfg))
    st.update(moe_routed(g_row, W, cfg, st))
    st["combined_out"] = st["routed_sum"] + st["shared_out"]
    return st


def wto(t, dtype):
    return t.to(dtype) if dtype is not t.dtype else t


def forward_block(dumps, W, cfg, mode=GDN_MODE, row=TOKEN_ROW):
    """Layer 0, one token, both mixer sub-chains, ending at the block's mixer outputs.

    The GDN chain is fed its own pinned decode-step input and cache; the MoE chain is fed one row of
    the pinned MoE input. They are different tokens (the GDN decode input is a later row of the same
    generator draw, per the GDN manifest's `source_tensor`), and the manifests pin no residual stream
    at layer 0, so §03.2's two residual adds have no referent and are not computed here. Feeding each
    mixer its own pinned input is what keeps every stage row attributable: a chain built by inventing
    the missing anchor would reproduce nothing and still print a plausible number.
    """
    stages, meta = {}, {}
    x_gdn = dumps.tensor("%s.%s.input" % (GDN_PASS, mode)).reshape(1, 1, cfg["d"])
    cache_state = dumps.tensor("%s.prefill.state" % GDN_PASS)
    cache_conv = dumps.tensor("%s.prefill.conv_window" % GDN_PASS)
    gdn = gdn_recurrent_step(x_gdn, W, cfg, cache_conv, cache_state)
    x_moe = dumps.tensor("layer0.input", row=row).reshape(1, cfg["d"])
    moe = moe_block(x_moe, W, cfg)
    for st in (gdn, moe):
        stages.update(st)
    meta["gdn_pass"] = "%s.%s" % (GDN_PASS, mode)
    meta["moe_pass"] = "layer0.row%d" % row
    meta["gdn_stages"] = len(gdn)
    meta["moe_stages"] = len(moe)
    return stages, meta


def _kv(line):
    """`key=value` record parser whose value domain is 'not a space' — a numeric-only value pattern
    silently drops every word-valued field (`pitfalls/40` §a-driver-whose-parser-matches-nothing)."""
    return dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)


def staged_precondition(evidence):
    """Admit M1's staging read-back log as a PRECONDITION on staging integrity — not as a device result.

    Nothing here allocates, copies or computes on a device. What this proves is narrower and fully
    checkable on CPU: that the log `engine/stage_run.py --device --verify` produced describes the
    artifact THIS process is reading. Each half is a refusal with its own tag:

      * the log carries a `RESULT PASS ... readback=<n>/<n>` line with n > 0, so the run it quotes
        completed and counted its own read-backs (`staged-evidence-result-line`);
      * the log names the pinned artifact file and the file on disk is the pinned size
        (`staged-evidence-artifact-name`, `staged-evidence-artifact-size`);
      * for every policy-A tensor this block reads, the log names a run, that run's read-back verdict
        is MATCH with all three digests equal, and its byte range and stored width agree with
        `engine/loader.py`'s handle for the same name (`staged-evidence-incomplete`,
        `staged-evidence-readback`, `staged-evidence-layout`, `staged-evidence-width`);
      * sha256 over that byte range of the artifact, recomputed here through the gated reader, equals
        the digest the log says came back (`staged-evidence-region-digest`).

    The brief asks for "the GGUF sha the log names" to be compared against the artifact's. The log names
    no whole-file sha — one digest per staging run and none for the file, `grep -c` over the log for a
    64-hex field attached to a file name returns nothing — and hashing 22 GB to compare a string would
    be a slow way to learn less than the region digests say. So the identity test is per region, over
    exactly the bytes this block reads, and the pinned whole-file sha is printed as the artifact's
    identity from `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json` (the file the loader was opened against).
    """
    path = Path(evidence)
    if not path.exists():
        raise Refused("staged-evidence-absent",
                      "%s does not exist; run engine/stage_run.py --device --verify first" % path)
    text = path.read_text()
    tensors, runs, reads = {}, {}, {}
    for line in text.splitlines():
        if line.startswith("TENSOR "):
            f = _kv(line)
            tensors[f["name"]] = f
        elif line.startswith("RUN "):
            f = _kv(line)
            runs[int(f["idx"])] = f
        elif line.startswith("READBACK "):
            f = _kv(line)
            reads[int(f["idx"])] = f
    if not tensors or not runs or not reads:
        raise Refused("staged-evidence-unparsed",
                      "%s yielded %d TENSOR, %d RUN and %d READBACK records; a check that parses "
                      "nothing cannot fail" % (path, len(tensors), len(runs), len(reads)))
    halves = [m.groups() for line in text.splitlines() if line.startswith("RESULT PASS")
              for m in [re.search(r"\breadback=(\d+)/(\d+)\b", line)] if m]
    if not [h for h in halves if int(h[0]) == int(h[1]) > 0]:
        raise Refused("staged-evidence-result-line",
                      "%s has no `RESULT PASS ... readback=<n>/<n>` line with n>0 (found %s); the log "
                      "of a run that did not finish cannot certify staging" % (path, halves or "none"))
    readback = "%s/%s" % halves[0]
    pin = json.loads(GGUF_PIN.read_text())
    if pin["artifact"] not in text:
        raise Refused("staged-evidence-artifact-name",
                      "the log never names %s, so it is not a log about the pinned artifact" %
                      pin["artifact"])
    W = PolicyA()
    try:
        art_path = Path(W.L.art.path)
        if art_path.name != pin["artifact"]:
            raise Refused("staged-evidence-artifact-name",
                          "this process opened %s but the pin names %s" % (art_path.name, pin["artifact"]))
        size = art_path.stat().st_size
        if size != int(pin["bytes_on_disk"]):
            raise Refused("staged-evidence-artifact-size",
                          "%s is %d B on disk, the pin records %d B" %
                          (art_path.name, size, int(pin["bytes_on_disk"])))
        rows, hashed, seen = [], 0, {}
        for name in POLICY_A_TENSORS:
            t = tensors.get(name)
            if t is None:
                raise Refused("staged-evidence-incomplete",
                              "%s is a policy-A tensor this block reads but the log names no run for it"
                              % name)
            run = int(t["run"])
            rb, rr = reads.get(run), runs.get(run)
            if rb is None or rr is None:
                raise Refused("staged-evidence-incomplete",
                              "%s: log run %d has no %s record" % (name, run,
                                                                   "READBACK" if rb is None else "RUN"))
            three_equal = (rb["readback_sha256"] == rb["expected_plan_sha256"]
                           == rb["expected_header_sha256"])
            if rb.get("verdict") != "MATCH" or not three_equal:
                raise Refused("staged-evidence-readback",
                              "%s: log run %d read-back verdict=%s, three digests not equal"
                              % (name, run, rb.get("verdict")))
            h = W.L.handle(name)
            off, length = int(rr["file_off"]), int(rr["length"])
            if int(rr.get("read_off", off)) != off:
                raise Refused("staged-evidence-layout",
                              "%s: log run %d reads at %s but names file_off %s; the two must agree for a "
                              "digest of the file to be a digest of what was staged" % (name, run,
                                                                                        rr.get("read_off"),
                                                                                        off))
            # The plan coalesces small tensors into one run, so a run is a SPAN that contains this
            # tensor's own range rather than being equal to it; containment is the honest requirement,
            # and the digest below is over the span the read-back actually covered.
            t_off, t_len = int(h["file_off"]), int(h["length"])
            if not (off <= t_off and t_off + t_len <= off + length):
                raise Refused("staged-evidence-layout",
                              "%s: engine/loader.py addresses it as %d B at %d, which is not inside the "
                              "log's run %d (%d B at %d) — two different files"
                              % (name, t_len, t_off, run, length, off))
            if t.get("type") != h["type_name"]:
                raise Refused("staged-evidence-width",
                              "%s: log says %s, the artifact stores %s" % (name, t.get("type"),
                                                                          h["type_name"]))
            if run not in seen:
                got, n = W.region_sha(off, length)
                hashed += n
                seen[run] = got
                if got != rb["readback_sha256"]:
                    raise Refused("staged-evidence-region-digest",
                                  "%s: sha256 of %s B at %s of the file this process is reading is %s, "
                                  "the log's read-back for run %d says %s"
                                  % (name, n, off, got[:16], run, rb["readback_sha256"][:16]))
            rows.append(dict(tensor=name, run=str(run), slab=t["slab"], type=h["type_name"],
                             digest=rb["readback_sha256"], bytes_hashed=length))
    finally:
        W.close()
    return dict(rows=rows, log=path.name, artifact=pin["artifact"], pin_sha=pin["sha256"],
                readback=readback, bytes_hashed=hashed, tensors=len(rows), runs=len(seen))


def device_refusal():
    """Raise the refusal every device claim from this process deserves. There is no device path here."""
    raise Refused("live-device-path-absent",
                  "this file allocates nothing, launches nothing and imports no HIP, so --device cannot "
                  "be honoured here; engine/loader.py::device_slabs still raises device-allocator-absent. "
                  "The residency claim belongs to engine/stage_run.py --device --verify, whose read-back "
                  "log --staged-evidence admits as a precondition — and a precondition is not a result.")


POLICY_A_TENSORS = tuple(sorted(
    [n for n in PolicyA.WIDTH if not n.endswith("_exps.weight")]) +
    ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"])
