#!/usr/bin/env python3
"""oracle_arm.py — the independent CPU side of prototype/moe_gather's gate.

Run with the venv interpreter (system python3 has no torch):
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python oracle_arm.py [--work work]

It is the only thing in this lane that produces expectations, and no expectation is read out of the C++
under test. Three independent sources:

  * `tools/gguf_reader.py` — the audited CPU dequantizer (quantization/evidence/
    audit-selftest-2026-10-26.json), used at its own double-route precision and rounded ONCE to f32,
    which is the route the header declares as the judged one.
  * the pinned dumps (checkpoints/dump-manifest-2026-10-26.json) for real rows: activations, top-k
    ids, both weight dtypes, and the pinned `expert_raw` / `routed_sum` bit images.
  * `speccheck/moe_ref.py` for prediction 3's candidate readings of the fused `gate_up_proj`.

Emits under --work:
  plan.tsv      the tagged plan both arms parse (U/R/B/C; one parser, mg_read_plan).
  slab.bin      every byte and vector the plan indexes.
  expect.f32    one fp32 word per judged element, at each case's exp_off.
  cases.tsv     one record per case, with the expectation's SOURCE and its exactness bound.
  units.tsv, rows.tsv, blocks.tsv   plan inventory for gate.py's parsed==printed assertions.
  oracle_p3.tsv, oracle_p3_matrix.tsv  prediction 3 and its distinctness matrix.
  oracle_floor.tsv  the derived figures every band here comes from. There is no typed tolerance.

Every section prints the number of records it wrote. CPU only: prints `no timing taken`, never a
duration of the device or of a kernel.
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # prototype/moe_gather -> repo root
for p in (os.path.join(REPO, "tools"), REPO, os.path.join(REPO, "speccheck"),
          os.path.join(REPO, "prototype", "moe_router")):
    if p not in sys.path:
        sys.path.insert(0, p)

import gguf_reader as gr              # noqa: E402
import audit_published_quant as apq   # noqa: E402
import moe_ref                        # noqa: E402
import oracle_arm as router_oracle    # noqa: E402  (load_manifest/read_arr only)

KERNELS_H = os.path.join(HERE, "moe_gather_kernels.h")
GGUF_PATH = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
ELEM = 256                              # MG_QK_K, asserted against the header below
ROLE_OF = {"gate": 0, "up": 1, "down": 2}
WIDTH_OF = {"Q4_K": 0, "Q5_K": 1, "Q6_K": 2}
GGUF_TENSOR = {"gate": "blk.%d.ffn_gate_exps.weight",
               "up": "blk.%d.ffn_up_exps.weight",
               "down": "blk.%d.ffn_down_exps.weight"}
# What each sampled block buys. layer2 is absent on purpose: shard 3 is not on disk (HANDOFF §5.4),
# so a plan that assumed it would be a plan that silently skips a case.
BLOCKS = [("layer0", 0, 0), ("layer39", 39, 0), ("mtp0", 40, 1)]


class GateError(Exception):
    pass


# -----------------------------------------------------------------------------------------------
# the header is the source of the enums; a Python mirror of them would be a second spelling
# -----------------------------------------------------------------------------------------------
def read_header_constants():
    text = open(KERNELS_H).read()
    c = {}
    for name in ("MG_H", "MG_I", "MG_TWO_I", "MG_K", "MG_E", "MG_QK_K",
                 "MG_Q4K_BYTES", "MG_Q5K_BYTES", "MG_Q6K_BYTES", "MG_NAN_BITS",
                 "MG_PLAN_FIELDS_U", "MG_PLAN_FIELDS_R", "MG_PLAN_FIELDS_B", "MG_PLAN_FIELDS_C",
                 "MG_FIELD_COUNT", "MG_CASE_KIND_COUNT"):
        m = re.search(r"#define\s+%s\s+(\S+)" % name, text)
        if not m:
            raise GateError("no #define %s in %s" % (name, KERNELS_H))
        c[name] = int(m.group(1).rstrip("uUlL"), 0)
    modes = dict((n, int(v)) for n, v in re.findall(r"\b(MG_MODE_[A-Z0-9_]+)\s*=\s*(\d+)", text))
    c["MG_MODE_COUNT"] = modes.pop("MG_MODE_COUNT")   # the enum terminator is not itself a mode
    c["MODES"] = modes
    c["EXPS"] = dict((n, int(v)) for n, v in re.findall(r"\b(MG_EXP_[A-Z]+)\s*=\s*(\d+)", text))
    if len(c["MODES"]) != c["MG_MODE_COUNT"]:
        raise GateError("header declares MG_MODE_COUNT=%d but defines %d modes"
                        % (c["MG_MODE_COUNT"], len(c["MODES"])))
    if "MG_EXPERT_FLOATS" not in text:
        raise GateError("no MG_EXPERT_FLOATS in %s" % KERNELS_H)
    # MG_OFF_* are arithmetic #defines, so compute them the way the header does rather than parse them.
    c["OFF_GATE"], c["OFF_UP"] = 0, c["MG_I"]
    c["OFF_S"], c["OFF_RAW"] = 2 * c["MG_I"], 3 * c["MG_I"]
    c["EXPERT_FLOATS"] = 3 * c["MG_I"] + c["MG_H"]
    return c


HC = read_header_constants()
MODES, EXPS = HC["MODES"], HC["EXPS"]
EXP_NAME = [k for k, _ in sorted(EXPS.items(), key=lambda kv: kv[1])]
MODE_NAME = {v: k for k, v in MODES.items()}
NAN_WORD = np.uint32(HC["MG_NAN_BITS"]).view(np.float32)


# -----------------------------------------------------------------------------------------------
# geometry: the config is the census, the published file has to match it
# -----------------------------------------------------------------------------------------------
def reconcile_geometry():
    cfg = json.load(open(os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")))
    t = cfg["text_config"]
    got = (t["hidden_size"], t["moe_intermediate_size"], t["num_experts_per_tok"], t["num_experts"])
    want = (HC["MG_H"], HC["MG_I"], HC["MG_K"], HC["MG_E"])
    if got != want:
        raise GateError("config says H,I,K,E=%s but %s says %s" % (got, KERNELS_H, want))
    if 2 * t["moe_intermediate_size"] != HC["MG_TWO_I"]:
        raise GateError("MG_TWO_I %d != 2*moe_intermediate_size" % HC["MG_TWO_I"])
    for w, hdr in (("Q4_K", "MG_Q4K_BYTES"), ("Q5_K", "MG_Q5K_BYTES"), ("Q6_K", "MG_Q6K_BYTES")):
        if gr.BLOCK[w][0] != ELEM or gr.BLOCK[w][1] != HC[hdr]:
            raise GateError("reader and header disagree on %s: reader=%s header=%d"
                            % (w, gr.BLOCK[w], HC[hdr]))
    return t


def require_audit_marker():
    """Mirrors tools/audit_published_quant.py §require_selftest WITHOUT appending to its ordering log
    (that log belongs to the step-1 audit; HANDOFF §11 warns not to attribute other lanes' lines).
    The property kept: no tensor-data byte is read unless the audit's synthetic-block selftest passed
    against the CURRENT bytes of that gate and this reader."""
    if not os.path.exists(apq.SELFTEST_MARK):
        raise GateError("no selftest marker at %s — run tools/audit_published_quant.py --selftest"
                        % apq.SELFTEST_MARK)
    mark = json.load(open(apq.SELFTEST_MARK))

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    if mark.get("gate_sha256") != sha(os.path.join(REPO, "tools/audit_published_quant.py")) \
            or mark.get("reader_sha256") != sha(os.path.join(REPO, "tools/gguf_reader.py")):
        raise GateError("the audit selftest marker is STALE against the current tools/")
    gr.grant_selftest_token(apq.SELFTEST_TOKEN)
    return mark


# -----------------------------------------------------------------------------------------------
# pinned dumps
# -----------------------------------------------------------------------------------------------
DUMP = {}


def dump_arrays(name):
    """(f32 view, uint16 bit image) of one dumped tensor, whatever dtype the record declares.

    bf16 is expanded by BIT IMAGE (<<16), never by a float cast, so a comparison downstream is a
    bit-image comparison (pitfalls/30 §".view vs .to")."""
    rec = DUMP[name]                       # a typo is a KeyError, which is the right loud failure
    raw = open(os.path.join(REPO, rec["file"]), "rb").read()
    if len(raw) != rec["bytes"]:
        raise GateError("%s: %d bytes, manifest says %d" % (name, len(raw), rec["bytes"]))
    if rec["dtype"] == "torch.bfloat16":
        u = np.frombuffer(raw, dtype=np.uint16).reshape(rec["shape"])
        return (u.astype(np.uint32) << 16).view(np.float32), u
    if rec["dtype"] == "torch.float32":
        f = np.frombuffer(raw, dtype=np.float32).reshape(rec["shape"])
        return f, None
    if rec["dtype"] == "torch.int64":
        return np.frombuffer(raw, dtype=np.int64).reshape(rec["shape"]), None
    raise GateError("unhandled dump dtype %s for %s" % (rec["dtype"], name))


# -----------------------------------------------------------------------------------------------
# the declared arithmetic form, spelled once, in numpy
# -----------------------------------------------------------------------------------------------
def dot_ascending(W, x):
    """W (rows, n) against x (n,), accumulator walked j = 0..n-1 ascending, in fp32.

    The header's declared order is one thread per output element walking j ascending; this is the same
    sequence of fp32 multiplies and adds with the rows vectorised instead of looped. No contraction:
    numpy does not fuse, and both C++ arms build with -ffp-contract=off.

    Zero columns are NOT skipped, even though 0.0*w is an exact zero for finite w: a misreading mode
    can put an infinity or a NaN in a column the activation zeroes, and `0 * inf` is NaN — skipping it
    would make this mirror disagree with the kernel on exactly the cases the injections produce.
    """
    W = np.asarray(W, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    acc = np.zeros(W.shape[0], dtype=np.float32)
    for j in range(W.shape[1]):
        acc = (acc + x[j] * W[:, j]).astype(np.float32)
    return acc


def silu_f32(z):
    """z / (1 + exp(-z)), fp32 result. Python's exp is double, so OFF the exact fixtures this is the
    transcendental the header refuses to band; ON them |z| >= MG_EXACT_EXP_MIN and expf(-z) flushes to
    zero, so both routes return z exactly (or a signed zero for z <= -MG_EXACT_EXP_MIN)."""
    z64 = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        s = z64 / (1.0 + np.exp(-z64))
    return np.asarray(s, dtype=np.float32)


def f32_to_bf16_bits(a):
    """Round-to-nearest-even over the top 16 bits of f32 words, as mg_bf16_of does."""
    a = np.asarray(a, dtype=np.float32)
    b = np.frombuffer(a.tobytes(), dtype=np.uint32).reshape(a.shape).copy()
    lb = (b & 0xFFFF).astype(np.uint64)
    r = (b >> 16).astype(np.uint32)
    inc = ((lb > 0x8000) | ((lb == 0x8000) & ((r & 1).astype(bool)))).astype(np.uint32)
    return (r + inc).astype(np.uint16)


def bf16_bits_to_f32(u):
    u = np.asarray(u, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


def order_expert_asc(ids):
    return sorted(range(len(ids)), key=lambda k: (int(ids[k]), k))


def combine_form(raw_f32, w_f32, ids, order="expert_asc", cast="before", acc="bf16",
                 w_by="rank"):
    """The reference combine, plus each of the four injections, in one spelling.

    Correct form (probe6: 0 of 65536 elements differ per block against the pinned `routed_sum`):
        prod = bf16( f32(raw[t,k]) * f32(w_bf16[t,k]) )    weight cast BEFORE the multiply
        acc  = bf16( f32(acc) + f32(prod) )                bf16 accumulator
    with addends in ASCENDING EXPERT-ID order and the weight reached BY RANK.
      order="rank"      -> MG_MODE_RANK_ACCUM
      cast="after"      -> MG_MODE_CAST_AFTER (README prediction 4's different number)
      acc="f32"         -> MG_MODE_F32_COMBINE
      w_by="expert_id"  -> MG_MODE_W_BY_EXPERT_ID
    Returns bf16 bit images (uint16 array of length H).
    """
    K = raw_f32.shape[0]
    seq = order_expert_asc(ids) if order == "expert_asc" else list(range(K))
    if acc == "f32":
        tot = np.zeros(raw_f32.shape[1], dtype=np.float32)
        for k in seq:
            wk = w_f32[ids[k] % K] if w_by == "expert_id" else w_f32[k]
            tot = (tot + raw_f32[k] * wk).astype(np.float32)
        return f32_to_bf16_bits(tot)
    out = np.zeros(raw_f32.shape[1], dtype=np.uint16)
    for k in seq:
        wk = w_f32[ids[k] % K] if w_by == "expert_id" else w_f32[k]
        if cast == "before":
            wk = bf16_bits_to_f32(f32_to_bf16_bits(np.asarray([wk], dtype=np.float32)))[0]
        prod = f32_to_bf16_bits((raw_f32[k] * wk).astype(np.float32))
        out = f32_to_bf16_bits(bf16_bits_to_f32(out) + bf16_bits_to_f32(prod))
    return out


# -----------------------------------------------------------------------------------------------
# slab
# -----------------------------------------------------------------------------------------------
class Slab(object):
    def __init__(self):
        self.buf = bytearray()

    def add_bytes(self, raw):
        off = len(self.buf)
        self.buf.extend(raw)
        return off

    def _align(self):
        while len(self.buf) % 4:
            self.buf.append(0xFF)          # a misaligned float slice is a plan bug; pad loudly
        return len(self.buf)

    def add_f32(self, arr):
        off = self._align()
        self.buf.extend(np.asarray(arr, dtype=np.float32).reshape(-1).tobytes())
        return off

    def add_i32(self, arr):
        off = self._align()
        self.buf.extend(np.asarray(arr, dtype=np.int32).reshape(-1).tobytes())
        return off

    def guard(self, n=512):
        """Trailing bytes so MG_MODE_BYTE_SHIFT_1 on a final block stays inside the slab and reads
        real (zero) bytes instead of falling off the end into `undefined`."""
        self.buf.extend(bytes(n))


# -----------------------------------------------------------------------------------------------
# synthetic exact fixtures
# -----------------------------------------------------------------------------------------------
F16_ONE = 0x3C00
LIVE_COLS = (7, 263)          # two live columns, 256 apart, so BLOCK_OFF_1 lands on the other one
X0 = 512.0                    # the activation value at every live column


def synth_block_q4k(vals):
    """256 integers in 0..15 -> one Q4_K block decoding to them (d=1.0, dmin=0, sc=1, mg=0)."""
    b = bytearray(144)
    b[0], b[1] = F16_ONE & 0xFF, F16_ONE >> 8
    b[4:8] = bytes([1, 1, 1, 1])
    b[12:16] = bytes([1, 1, 1, 1])
    for e in range(ELEM):
        v = int(vals[e])
        if not 0 <= v <= 15:
            raise GateError("Q4_K synthetic value %d is off the grid" % v)
        g, half, l = e >> 6, (e >> 5) & 1, e & 31
        i = 16 + 32 * g + l
        b[i] = (b[i] & 0x0F) | (v << 4) if half else (b[i] & 0xF0) | v
    return bytes(b)


def synth_block_q6k(vals):
    """256 integers in -32..31 -> one Q6_K block (d=1.0 at byte 208, every scale +1)."""
    b = bytearray(210)
    b[208], b[209] = F16_ONE & 0xFF, F16_ONE >> 8
    for i in range(16):
        b[192 + i] = 1
    for e in range(ELEM):
        q = int(vals[e]) + 32
        if not 0 <= q <= 63:
            raise GateError("Q6_K synthetic value %d is off the grid" % (q - 32))
        g, k, l = e >> 7, (e >> 5) & 3, e & 31
        iql = 64 * g + l + 32 * (k & 1)
        b[iql] = (b[iql] & 0xF0) | (q & 15) if k < 2 else (b[iql] & 0x0F) | ((q & 15) << 4)
        ihi = 128 + 32 * g + l
        b[ihi] = (b[ihi] & ~(3 << (2 * k))) | (((q >> 4) & 3) << (2 * k))
    return bytes(b)


SYNTH_BUILD = {"Q4_K": synth_block_q4k, "Q6_K": synth_block_q6k}


def synth_matrix(role, rows, cols):
    """The chosen value matrix of a synthetic unit. Integers only, and small:

    gate[i]  = X0 * Vg[i, 7]  (+ X0 * Vg[i, 263] = 0 on the honest read)  -> 512 or 1536
    up[i]    = X0 * Vu[i, 7]                                             -> 1024
    s[i]     = silu(gate)*up, and |gate| >= 128 makes silu the identity   -> 524288 / 1572864
    raw[t]   = s[3]*Vd[t, 3] + s[259]*Vd[t, 259]                          -> <= 4 718 592 < 2**24
    Every intermediate is an integer below 2**24, so f32 addition is exact on it and no association
    question exists to disagree about — which is what licenses MG_EXP_MATCH on these cases. The
    second live column (263 = 7+256) is what makes the block-shift injection produce numbers rather
    than zeros.
    """
    m = np.zeros((rows, cols), dtype=np.float32)
    if role == "gate":
        m[:, LIVE_COLS[0]] = [3.0 if (i % 2) else 1.0 for i in range(rows)]
        m[:, LIVE_COLS[1]] = 2.0
    elif role == "up":
        m[:, LIVE_COLS[0]] = 2.0
    else:
        m[:, 3] = [0.0, 1.0, 2.0, 1.0][0]      # replaced below; keep the shape explicit
        for i in range(rows):
            m[i, 3] = [0.0, 1.0, 2.0, 1.0][i % 4]
            m[i, 259] = [1.0, 0.0, 1.0, 0.0][i % 4]
    return m


def build_synth_unit_bytes(matrix, kind):
    """Bytes for a whole unit, confirmed by the audited reader to decode to exactly `matrix`."""
    build = SYNTH_BUILD[kind]
    flat = matrix.reshape(-1)
    if flat.size % ELEM:
        raise GateError("synthetic unit has %d elements, not a whole number of blocks" % flat.size)
    out = bytearray()
    for bi in range(flat.size // ELEM):
        out.extend(build(flat[bi * ELEM:(bi + 1) * ELEM]))
    got = dequant_f32(bytes(out), kind, flat.size).reshape(matrix.shape)
    if not np.array_equal(got.view(np.uint32), matrix.view(np.uint32)):
        nd = int(np.count_nonzero(got.view(np.uint32) != matrix.view(np.uint32)))
        raise GateError("synthetic %s fixture does not decode to its declared matrix (%d of %d "
                        "elements differ): the FIXTURE BUILDER is wrong; the sample is refused "
                        "rather than judged" % (kind, nd, flat.size))
    return bytes(out)


# -----------------------------------------------------------------------------------------------
# published bytes, through the audited reader
# -----------------------------------------------------------------------------------------------
def dequant_f32(raw, tname, n_elts):
    """The audited reader's double route, rounded once to f32 — the declared judged route."""
    return np.asarray(gr.dequant(raw, tname, n_elts), dtype=np.float64).astype(np.float32)


def read_expert_role(gg, block_index, role, expert):
    """(raw run, tname, rows, cols, tensor_index) for one expert of one role, HF order."""
    tname_tensor = GGUF_TENSOR[role] % block_index
    t = gg.by_name.get(tname_tensor)
    if t is None:
        raise GateError("published gguf has no %s" % tname_tensor)
    if len(t.dims) != 3:
        raise GateError("%s has dims %s, expected three" % (tname_tensor, t.dims))
    cols, rows, e_axis = (int(x) for x in t.dims)      # GGUF order: dims[0] is fastest
    if e_axis != HC["MG_E"] or not 0 <= expert < e_axis:
        raise GateError("%s expert axis %d cannot hold %d" % (tname_tensor, e_axis, expert))
    if rows * cols % ELEM:
        raise GateError("%s: %d elements per expert is not whole blocks"
                        % (tname_tensor, rows * cols))
    bb = gr.BLOCK[t.type_name][1]
    start = t.offset + (rows * cols // ELEM) * bb * expert
    raw = gg.read(apq.SELFTEST_TOKEN, gg.data_start + start, (rows * cols // ELEM) * bb,
                  "moe_gather oracle sample")
    return raw, t.type_name, rows, cols, t.index


# -----------------------------------------------------------------------------------------------
# the misreading addressings, as functions over a unit's honest matrix
# -----------------------------------------------------------------------------------------------
def remap_input_chunk(M):
    """MG_MODE_INPUT_CHUNK: the [rows, cols] run read as [2*rows, cols]... the kernel computes
    flat = (2*row + half)*n_cols + c with c < n_cols, so half is always 0 and the read is row 2i of
    the same run; rows at or past the run's own block count are `undefined` and read MG_NAN_BITS."""
    rows, cols = M.shape
    out = np.full((rows, cols), NAN_WORD, dtype=np.float32)
    n_flat = M.size
    for i in range(rows):
        base = 2 * i * cols
        if base + cols - 1 >= n_flat:
            continue                       # blk >= n_blocks in the kernel -> undefined -> NaN
        out[i] = M.reshape(-1)[base:base + cols]
    return out


def remap_row_off_1(M):
    return np.roll(M, -1, axis=0)          # row r read as (r+1) % rows


def remap_block_off_1(M):
    """flat -> flat + 256, wrapping inside the run (the kernel does blk = (blk+1) % n_blocks)."""
    flat = M.reshape(-1)
    shifted = np.empty_like(flat)
    nb = flat.size // ELEM
    for blk in range(nb):
        nxt = (blk + 1) % nb
        shifted[blk * ELEM:(blk + 1) * ELEM] = flat[nxt * ELEM:(nxt + 1) * ELEM]
    return shifted.reshape(M.shape)


def remap_byte_shift_1(raw, tname, rows, cols):
    """The same bytes read one byte late, decoded by the audited reader — which is what a converter
    that dropped an 8-byte header length actually sees (the very bug that made the router lane's
    prediction-3 probe read 523 841 of 524 288 elements wrongly)."""
    bb = gr.BLOCK[tname][1]
    take = (rows * cols // ELEM) * bb + 1        # one extra byte, because the read is late by one
    return dequant_f32(raw[1:take], tname, rows * cols).reshape(rows, cols)


def shift_mat(plan, u):
    """The unit's own bytes read one byte late, from the FINAL slab.

    This has to run after every unit and the trailing guard are in place, because the byte that follows
    the last block of the last unit is the guard's zero, not something inside the unit. Called from
    build_sample's second pass, which is why the byte-shift case carries a placeholder expectation
    until then."""
    start = u["src_off"]
    raw = bytes(plan.slab.buf[start:start + u.n_bytes + 1])
    if len(raw) < u.n_bytes + 1:
        raise GateError("unit %d has no following byte: the slab guard is missing" % u.n)
    return dequant_f32(raw[1:], u.tname, u["rows"] * u.cols).reshape(u["rows"], u.cols)


def q6k_d_at_byte0(matrix, raw):
    """MG_MODE_Q6K_D_BYTE_0 on a synthetic Q6_K unit: only `d` moves, from byte 208 to byte 0. The
    value is d*sc*(q-32) with sc=1 here, so the whole matrix scales by f16(raw[0:2])/f16(0x3C00)."""
    d0 = gr.f16(raw, 0)
    d1 = gr.f16(raw, 208)
    if d1 == 0.0:
        raise GateError("synthetic Q6_K fixture has d=0 at byte 208")
    return np.asarray(np.float64(matrix) * (np.float64(d0) / np.float64(d1)), dtype=np.float32)


# -----------------------------------------------------------------------------------------------
# case bookkeeping
# -----------------------------------------------------------------------------------------------
class Unit(object):
    def __init__(self, n, **kw):
        self.n = n
        self.d = dict(kind=0, declared_kind=0, block=0, role=0, expert=-1, src_off=0, slot_bytes=0,
                      n_blocks=0, rows=0, set_pos=0, main_n=1, cls=0, tensor_index=-1)
        self.d.update(kw)
        self.mat = None
        self.raw = b""
        self.tname = "?"

    def plan_row(self):
        d = self.d
        # 16 columns, mg_read_plan's MG_PLAN_FIELDS_U; the last two are padding the field count wants
        return ["U", d["kind"], d["declared_kind"], d["block"], d["role"], d["expert"], d["src_off"],
                d["slot_bytes"], d["n_blocks"], d["rows"], d["set_pos"], d["main_n"], d["cls"],
                d["tensor_index"], "rsv", "rsv"]

    def __getitem__(self, k):
        return self.d[k]


class Plan(object):
    def __init__(self, work, max_rows):
        self.work = work
        self.slab = Slab()
        self.units, self.rows, self.blocks, self.cases = [], [], [], []
        self.expect = []
        self.expect_len = 0
        self.out_len = 0
        self.max_rows = max_rows
        self.n_units = 0
        self.modes_used = set()
        self.bshift_jobs = []

    # ---- inventory ---------------------------------------------------------------------
    def new_unit(self, **kw):
        u = Unit(self.n_units, **kw)
        self.n_units += 1
        self.units.append(u)
        return u

    def add_real_units(self, block_tag, block_index, gg, experts, main_n):
        base = len(self.units)
        for pos, e in enumerate(experts):
            for role in ("gate", "up", "down"):
                raw, tname, rows, cols, tidx = read_expert_role(gg, block_index, role, e)
                off = self.slab.add_bytes(raw)
                u = self.new_unit(kind=WIDTH_OF[tname], declared_kind=WIDTH_OF[tname],
                                  block=block_tag, role=ROLE_OF[role], expert=e, src_off=off,
                                  slot_bytes=gr.BLOCK[tname][1], n_blocks=len(raw) // gr.BLOCK[tname][1],
                                  rows=rows, set_pos=pos, main_n=main_n, cls=0, tensor_index=tidx)
                u.mat = dequant_f32(raw, tname, rows * cols).reshape(rows, cols)
                u.raw, u.tname, u.cols, u.n_bytes = raw, tname, cols, len(raw)
        return base

    def add_synth_units(self, block_tag, set_pos0, main_n, kind_down="Q6_K"):
        base = len(self.units)
        for pos1 in range(1):
            pos = set_pos0 + pos1
            for role, kind, rows, cols in (("gate", "Q4_K", HC["MG_I"], HC["MG_H"]),
                                           ("up", "Q4_K", HC["MG_I"], HC["MG_H"]),
                                           ("down", kind_down, HC["MG_H"], HC["MG_I"])):
                m = synth_matrix(role, rows, cols)
                raw = build_synth_unit_bytes(m, kind)
                off = self.slab.add_bytes(raw)
                u = self.new_unit(kind=WIDTH_OF[kind], declared_kind=WIDTH_OF[kind], block=block_tag,
                                  role=ROLE_OF[role], expert=-1, src_off=off,
                                  slot_bytes=gr.BLOCK[kind][1],
                                  n_blocks=len(raw) // gr.BLOCK[kind][1], rows=rows, set_pos=pos,
                                  main_n=main_n, cls=1, tensor_index=-1)
                u.mat, u.raw, u.tname, u.cols = m, raw, kind, cols
                u.n_bytes = len(raw)
        return base

    def add_row(self, block_tag, token, fixture, ids, wpre, x, raw=None, rawk_base=0, rawk_stride=0):
        ids = [int(v) for v in ids]
        wpre = [float(v) for v in wpre]
        wcast = [float(bf16_bits_to_f32(f32_to_bf16_bits(np.asarray([v], dtype=np.float32)))[0])
                 for v in wpre]
        if raw is None:
            raw = np.zeros((HC["MG_K"], HC["MG_H"]), dtype=np.float32)
        r = dict(index=len(self.rows), block=block_tag, row=token, fixture=fixture,
                 ids_off=self.slab.add_i32(ids),
                 wcast_off=self.slab.add_f32(wcast), wpre_off=self.slab.add_f32(wpre),
                 x_off=self.slab.add_f32(x),
                 raw_off=(self.slab.add_f32(np.asarray(raw, dtype=np.float32).reshape(-1))
                          if raw is not None else 0),
                 id_asc=1 if all(ids[k] <= ids[k + 1] for k in range(len(ids) - 1)) else 0,
                 w_by_eid_fixed=1 if all(ids[k] % HC["MG_K"] == k for k in range(len(ids))) else 0,
                 rawk_base=rawk_base, rawk_stride=rawk_stride,
                 ids=ids, wpre=wpre, wcast=wcast,
                 x=np.asarray(x, dtype=np.float32),
                 raw=np.asarray(raw, dtype=np.float32).reshape(HC["MG_K"], HC["MG_H"])
                     if raw is not None else None)
        self.rows.append(r)
        return r

    def plan_row(self, r):
        return ["R", r["block"], r["row"], r["fixture"], r["ids_off"], r["wcast_off"], r["wpre_off"],
                r["x_off"], r["raw_off"], r["id_asc"], r["w_by_eid_fixed"], r["rawk_base"],
                r["rawk_stride"]]

    # ---- cases -------------------------------------------------------------------------
    def add_case(self, kind, mode, block_tag, row, slot, units, cls, src, exact, expect,
                 combine_src=0):
        nf = HC["EXPERT_FLOATS"] if kind == 0 else HC["MG_H"]
        exp = np.asarray(expect, dtype=np.float32).reshape(-1)
        if exp.size != nf:
            raise GateError("case mode=%s expects %d floats, got %d" % (MODE_NAME.get(mode, mode),
                                                                       nf, exp.size))
        dst = self.out_len
        self.out_len += nf
        off = self.expect_len
        self.expect_len += nf
        self.expect.append(exp)
        self.modes_used.add(mode)
        self.cases.append(dict(kind=kind, mode=mode, block=block_tag, row=row["row"], slot=slot,
                               u=(units[0].n, units[1].n, units[2].n), row_idx=row["index"],
                               combine_src=combine_src, dst_off=dst, exp_off=off, cmp_off=-1,
                               expect=cls, n_floats=nf, group=block_tag, src=src, exact=exact,
                               case_index=len(self.cases)))
        return self.cases[-1]

    def plan_row_c(self, c):
        return ["C", c["kind"], c["mode"], c["block"], c["row"], c["slot"], c["u"][0], c["u"][1],
                c["u"][2], c["row_idx"], c["combine_src"], c["dst_off"], c["exp_off"], c["cmp_off"],
                c["expect"], c["n_floats"], c["group"]]


def honest_expert(plan, units, row, mode=MODES["MG_MODE_CORRECT"]):
    """Compute one expert case's output span under an addressing mode, using the Python matrices.

    Returns the MG_EXPERT_FLOATS word span: gate[I] | up[I] | s[I] | raw[H]. This is the oracle's own
    arithmetic, never the kernel's: it reads `mat` (audited reader output) and applies the declared
    form. Modes that the kernel resolves by choosing a DIFFERENT unit (HALVES_SWAPPED, EXPERT_ROTATE)
    are handled by the caller passing different units; the address-remapping modes are handled here.
    """
    g, u_, d = units
    I, H, K = HC["MG_I"], HC["MG_H"], HC["MG_K"]
    if mode == MODES["MG_MODE_POISON_NAN"]:
        return np.full(HC["EXPERT_FLOATS"], NAN_WORD, dtype=np.float32)
    Mg, Mu, Md = g.mat, u_.mat, d.mat
    if mode == MODES["MG_MODE_INPUT_CHUNK"]:
        Mg, Mu = remap_input_chunk(Mg), remap_input_chunk(Mu)
    elif mode == MODES["MG_MODE_ROW_OFF_1"]:
        Mg, Mu, Md = remap_row_off_1(Mg), remap_row_off_1(Mu), remap_row_off_1(Md)
    elif mode == MODES["MG_MODE_BLOCK_OFF_1"]:
        Mg, Mu, Md = remap_block_off_1(Mg), remap_block_off_1(Mu), remap_block_off_1(Md)
    elif mode == MODES["MG_MODE_BYTE_SHIFT_1"]:
        Mg, Mu, Md = shift_mat(plan, g), shift_mat(plan, u_), shift_mat(plan, d)
    elif mode == MODES["MG_MODE_Q6K_D_BYTE_0"]:
        if d.tname != "Q6_K":
            raise GateError("Q6K_D_BYTE_0 asked for a %s down unit; the mode cannot move it"
                            % d.tname)
        Md = q6k_d_at_byte0(Md, d.raw)
    elif mode not in (MODES["MG_MODE_CORRECT"], MODES["MG_MODE_EXPERT_ROTATE"],
                      MODES["MG_MODE_HALVES_SWAPPED"], MODES["MG_MODE_SCALE_MIN_NAIVE"],
                      MODES["MG_MODE_WIDTH_MISREAD"]):
        raise GateError("honest_expert cannot model mode %s" % MODE_NAME.get(mode, mode))
    x = row["x"]
    gate = dot_ascending(Mg, x[:H])
    up = dot_ascending(Mu, x[:H])
    s = (silu_f32(gate) * up).astype(np.float32)
    raw = dot_ascending(Md, s[:I])
    out = np.empty(HC["EXPERT_FLOATS"], dtype=np.float32)
    out[HC["OFF_GATE"]:HC["OFF_GATE"] + I] = gate
    out[HC["OFF_UP"]:HC["OFF_UP"] + I] = up
    out[HC["OFF_S"]:HC["OFF_S"] + I] = s
    out[HC["OFF_RAW"]:HC["OFF_RAW"] + H] = raw
    return out


def honest_combine(row, cast="before", order="expert_asc", acc="bf16", w_by="rank", raw=None):
    """One combine span as f32 words, under the reference form unless a kwarg says otherwise.

    `cast="before"` uses the row's cast-BEFORE weights (the bf16 value R5 leaves behind); `cast=
    "after"` uses the fp32 pre-cast weights, which is prediction 4's different number.
    """
    w = np.asarray(row["wpre"] if cast == "after" else row["wcast"], dtype=np.float32)
    src = row["raw"] if raw is None else raw
    bits = combine_form(src, w, row["ids"], order=order, cast=cast, acc=acc, w_by=w_by)
    return bf16_bits_to_f32(bits)




# -----------------------------------------------------------------------------------------------
# the sample — declared before any device work; --max-rows cuts NUMBERS, never an assertion
# -----------------------------------------------------------------------------------------------
# One id set per block, chosen for what it makes catchable. These are geometry choices, and the
# plan's own id_asc / w_by_eid_fixed flags (which the kernel's immunity rules read) are derived from
# them, not typed alongside them.
SYNTH_IDS = {
    "layer0": [2, 5, 9, 20, 33, 61, 100, 150],     # ascending  -> RANK_ACCUM immune; ids%K != k
    "layer39": [7, 3, 5, 1, 6, 2, 4, 0],           # scrambled  -> both combine orders must fire
    "mtp0": [1, 9, 17, 25, 33, 41, 49, 57],        # ascending AND ids[k]%K == k -> both immune
}
FORM_CHECK = []          # combine-form vs pinned routed_sum, recorded for the floor table


def units_of(plan, base, pos):
    """The (gate, up, at set_pos `pos` of this block's main set."""
    return (plan.units[base + pos * 3], plan.units[base + pos * 3 + 1], plan.units[base + pos * 3 + 2])


def synth_x():
    x = np.zeros(HC["MG_H"], dtype=np.float32)
    for j in LIVE_COLS:
        x[j] = X0
    return x


def synth_wpre(K):
    """Weights that are not bf16-representable, so cast-before and cast-after are different numbers
    on the synthetic rows as well as on the real ones."""
    return [float(1.0 + (k + 1) * 2.0 ** -9) for k in range(K)]


def combine_injections(plan, row, ut, src_note):
    """The four combine modes on one row, with the class derived from THAT ROW'S GEOMETRY.

    RANK_ACCUM cannot move a row whose ids are already ascending, and W_BY_EXPERT_ID cannot move a row
    where ids[k] % K == k for every k — both are facts about the id vector, both are read back out of
    the plan's own id_asc / w_by_eid_fixed fields, and a case whose mode cannot reach it must be judged
    IMMUNE (must equal the honest output exactly). That is the must-not-catch side of
    pitfalls/40-gates-and-admissions.md §"A negative control with an empty must-catch set".
    """
    MM, EXP = MODES, EXPS
    spec = [(MM["MG_MODE_CAST_AFTER"], dict(cast="after"),
             "multiply by the fp32 weight and round the product afterwards (prediction 4)", None),
            (MM["MG_MODE_F32_COMBINE"], dict(acc="f32"),
             "fp32 accumulator rounded once instead of a bf16 accumulator", None),
            (MM["MG_MODE_RANK_ACCUM"], dict(order="rank"),
             "addends in rank order instead of ascending expert id", "id_asc"),
            (MM["MG_MODE_W_BY_EXPERT_ID"], dict(w_by="expert_id"),
             "weight array indexed by expert id instead of by rank", "w_by_eid_fixed")]
    for mode, kw, why, immune_field in spec:
        immune = immune_field is not None and row[immune_field] == 1
        cls = EXP["MG_EXP_IMMUNE"] if immune else EXP["MG_EXP_ALT"]
        exact = ("must equal the honest output exactly: this row's %s makes the mode a no-op"
                 % immune_field if immune else
                 "exact both ways: the alt must match AND differ from the honest span")
        plan.add_case(1, mode, row["block"], row, 0, ut, cls, src_note + ": " + why, exact,
                      honest_combine(row, **kw))


def build_sample(plan, gg):
    MM, EXP, K = MODES, EXPS, HC["MG_K"]
    meta = []
    for block_tag, (tag, idx, is_mtp) in enumerate(BLOCKS):
        n_real = 0 if tag == "mtp0" else plan.max_rows
        experts, ids_pin, w_pin, x_pin, raw_pin, rout_u = [], None, None, None, None, None
        if n_real:
            x_pin, _ = dump_arrays(tag + ".input")
            ids_pin, _ = dump_arrays(tag + ".topk_ids")
            w_pin, _ = dump_arrays(tag + ".topk_weights")
            raw_pin, raw_u = dump_arrays(tag + ".expert_raw")
            _, rout_u = dump_arrays(tag + ".routed_sum")
            ids_pin = ids_pin[:n_real].astype(np.int64)
            for t in range(n_real):
                for k in range(K):
                    e = int(ids_pin[t][k])
                    if e not in experts:
                        experts.append(e)
            experts.sort()
        main_n = len(experts) + 1
        base_real = plan.add_real_units(block_tag, idx, gg, experts, main_n)
        base_synth = plan.add_synth_units(block_tag, len(experts), main_n)
        first_case = len(plan.cases)
        m = dict(tag=tag, index=idx, is_mtp=is_mtp, block_tag=block_tag, main_n=main_n,
                 n_real=n_real, n_experts=len(experts), base_real=base_real,
                 base_synth=base_synth)

        # ---------------- real rows ---------------------------------------------------------
        for t in range(n_real):
            row = plan.add_row(block_tag, t, 0, ids_pin[t], w_pin[t], x_pin[t], raw=raw_pin[t])
            ut0 = units_of(plan, base_real, experts.index(int(ids_pin[t][0])))
            for k in range(K):
                ut = units_of(plan, base_real, experts.index(int(ids_pin[t][k])))
                plan.add_case(0, MM["MG_MODE_CORRECT"], block_tag, row, k, ut, EXP["MG_EXP_REPORT"],
                              "audited gguf_reader dequant of published bytes + declared fp32 form",
                              "gate/up exact; s/raw counted (silu is a transcendental off the "
                              "exact fixtures)", honest_expert(plan, ut, row))
            exp = honest_combine(row)
            # The expectation this case is judged by is itself checked against the pinned
            # routed_sum, in the same run, so it is not a self-derived reference.
            bits = f32_to_bf16_bits(exp)
            nd = int(np.count_nonzero(bits != rout_u[t]))
            FORM_CHECK.append((tag, t, nd, int(rout_u[t].size)))
            plan.add_case(1, MM["MG_MODE_CORRECT"], block_tag, row, 0, ut0, EXP["MG_EXP_MATCH"],
                          "bf16 combine over the PINNED expert_raw (form fixed by work/probe6.py),"
                          " cross-checked against pinned routed_sum (%d of %d differ)"
                          % (nd, rout_u[t].size),
                          "exact: every addend is a bf16 value and the accumulator is bf16", exp)
            combine_injections(plan, row, ut0, "real row, pinned raw")
            plan.add_case(1, MM["MG_MODE_POISON_NAN"], block_tag, row, 0, ut0, EXP["MG_EXP_ALT"],
                          "leak guard: a poisoned combine record must read the loud pattern",
                          "exact: all MG_NAN_BITS", np.full(HC["MG_H"], NAN_WORD, dtype=np.float32))

        # ---------------- the synthetic exact row, and the expert-body injections ---------------
        srow = plan.add_row(block_tag, 1000 + block_tag, 1, SYNTH_IDS[tag], synth_wpre(K), synth_x())
        ut = units_of(plan, base_synth, 0)
        first_dst = None
        for k in range(K):
            c = plan.add_case(0, MM["MG_MODE_CORRECT"], block_tag, srow, k, ut, EXP["MG_EXP_MATCH"],
                              "hand-built integer fixture, confirmed element-for-element by the "
                              "audited reader before the plan was written",
                              "exact: every intermediate is an integer below 2**24", 
                              honest_expert(plan, ut, srow))
            if first_dst is None:
                first_dst = c["dst_off"]
        srow["rawk_base"] = first_dst
        srow["rawk_stride"] = HC["EXPERT_FLOATS"]
        raw_synth = honest_expert(plan, ut, srow)[HC["OFF_RAW"]:HC["OFF_RAW"] + HC["MG_H"]]
        tiled = np.tile(raw_synth, (K, 1))
        plan.add_case(1, MM["MG_MODE_CORRECT"], block_tag, srow, 0, ut, EXP["MG_EXP_MATCH"],
                      "this kernel's OWN mode-0 raw outputs (the end-to-end arm)",
                      "exact only because the fixture is integer-exact",
                      honest_combine(srow, raw=tiled), combine_src=1)
        combine_injections(plan, srow, ut, "synthetic raw (this arm's own output)")
        for mode, why in (
                (MM["MG_MODE_HALVES_SWAPPED"], "gate and up exchanged: the [E,2I,H] half swap"),
                (MM["MG_MODE_INPUT_CHUNK"], "the fused run read as [I,2H]: output row i reads run "
                                            "row 2i; rows past the run read MG_NAN_BITS"),
                (MM["MG_MODE_ROW_OFF_1"], "output row i read as (i+1) % rows"),
                (MM["MG_MODE_BLOCK_OFF_1"], "the 256-element block index is one late, wrapping "
                                            "inside the run"),
                (MM["MG_MODE_BYTE_SHIFT_1"], "the tensor data start is one byte late"),
                (MM["MG_MODE_Q6K_D_BYTE_0"], "Q6_K's d read at byte 0 instead of 208")):
            if mode == MM["MG_MODE_Q6K_D_BYTE_0"] and plan.units[ut[2].n].tname != "Q6_K":
                continue                      # the mode cannot move a unit of another width
            if mode == MM["MG_MODE_BYTE_SHIFT_1"]:
                # Deferred: this misread reaches one byte past its own unit, so its alternative has
                # to come from the FINAL slab. Placeholder now, real alt in the second pass below.
                c = plan.add_case(0, mode, block_tag, srow, 0, ut, EXP["MG_EXP_ALT"],
                                  "alt derived from the slab bytes themselves, read one byte late",
                                  "exact both ways: the alt must match AND differ from the honest span",
                                  honest_expert(plan, ut, srow))
                plan.bshift_jobs.append((c["case_index"], ut, srow))
                continue
            plan.add_case(0, mode, block_tag, srow, 0, ut, EXP["MG_EXP_ALT"],
                          "alt derived from the fixture matrix: " + why,
                          "exact both ways: the alt must match AND differ from the honest span",
                          honest_expert(plan, ut, srow, mode=mode))
        if main_n > 1:
            nxt = units_of(plan, base_real, 0) if len(experts) else ut
            plan.add_case(0, MM["MG_MODE_EXPERT_ROTATE"], block_tag, srow, 0, ut, EXP["MG_EXP_ALT"],
                          "alt = the next sampled expert of this block's main set",
                          "exact both ways", honest_expert(plan, nxt, srow))
        else:
            plan.add_case(0, MM["MG_MODE_EXPERT_ROTATE"], block_tag, srow, 0, ut, EXP["MG_EXP_IMMUNE"],
                          "this block's main set holds ONE expert, so (pos+1) %% main_n == pos and "
                          "the mode cannot reach the record",
                          "must equal the honest output exactly", honest_expert(plan, ut, srow))
        for mode, why in ((MM["MG_MODE_SCALE_MIN_NAIVE"],
                           "the 96 scale/min bits read as 16 contiguous 6-bit fields"),
                          (MM["MG_MODE_WIDTH_MISREAD"],
                           "a Q5_K block decoded with the Q6_K decoder and the reverse")):
            if mode == MM["MG_MODE_WIDTH_MISREAD"] and plan.units[ut[2].n].tname not in ("Q5_K",
                                                                                         "Q6_K"):
                continue
            plan.add_case(0, mode, block_tag, srow, 0, ut, EXP["MG_EXP_MISMATCH"],
                          "must differ from the honest expectation: " + why,
                          "no independent alt is derived for this mode in this lane; the assertion "
                          "is that it differs somewhere", honest_expert(plan, ut, srow))
        plan.add_case(0, MM["MG_MODE_POISON_NAN"], block_tag, srow, 0, ut, EXP["MG_EXP_ALT"],
                      "leak guard on an expert record", "exact: all MG_NAN_BITS",
                      np.full(HC["EXPERT_FLOATS"], NAN_WORD, dtype=np.float32))
        m["case_lo"], m["case_hi"] = first_case, len(plan.cases) - 1
        meta.append(m)
    # Second pass: every unit is placed and the guard is appended, so the one mode whose misread
    # reaches past its own unit can now have its alternative derived from the bytes the kernel will
    # actually read rather than from a copy of them.
    plan.slab.guard()
    for cidx, ut, srow in plan.bshift_jobs:
        plan.expect[cidx] = honest_expert(plan, ut, srow, mode=MM["MG_MODE_BYTE_SHIFT_1"])
    for m in meta:
        ne = sum(1 for i in range(m["case_lo"], m["case_hi"] + 1) if plan.cases[i]["kind"] == 0)
        plan.blocks.append(dict(index=m["index"], is_mtp=m["is_mtp"], unit_base=m["base_real"],
                                main_n=m["main_n"], sup_lo=0, sup_hi=-1, case_lo=m["case_lo"],
                                case_hi=m["case_hi"], n_expert_cases=ne, tag=m["tag"],
                                n_real=m["n_real"], n_experts=m["n_experts"]))
    return meta


def write_plan_files(plan, work):
    """plan.tsv / slab.bin / expect.f32, with every field count asserted against the header."""
    rows = []
    for u in plan.units:
        rows.append(u.plan_row())
    for r in plan.rows:
        rows.append(plan.plan_row(r))
    for i, b in enumerate(plan.blocks):
        base = b["unit_base"]
        rows.append(["B", b["index"], b["is_mtp"], base, b["main_n"], b["sup_lo"], b["sup_hi"],
                     b["case_lo"], b["case_hi"], b["n_expert_cases"]])
    for c in plan.cases:
        rows.append(plan.plan_row_c(c))
    want = {"U": HC["MG_PLAN_FIELDS_U"], "R": HC["MG_PLAN_FIELDS_R"],
            "B": HC["MG_PLAN_FIELDS_B"], "C": HC["MG_PLAN_FIELDS_C"]}
    for r in rows:
        if len(r) != want[r[0]]:
            raise GateError("%s row has %d fields, the parser wants %d" % (r[0], len(r), want[r[0]]))
    path = os.path.join(work, "plan.tsv")
    with open(path, "w") as fh:
        fh.write("# moe_gather plan: U unit, R row, B block, C case; tab separated; field counts "
                 "from moe_gather_kernels.h MG_PLAN_FIELDS_*\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(os.path.join(work, "slab.bin"), "wb") as fh:
        fh.write(bytes(plan.slab.buf))
    exp = np.concatenate(plan.expect) if plan.expect else np.zeros(1, dtype=np.float32)
    if exp.size != plan.expect_len:
        raise GateError("expect buffer is %d words, bookkeeping said %d" % (exp.size, plan.expect_len))
    exp.astype(np.float32).tofile(os.path.join(work, "expect.f32"))
    return len(rows), exp.size


def write_inventory(plan, work, meta):
    """cases.tsv / units.tsv / rows.tsv — the record counts gate.py must reproduce."""
    with open(os.path.join(work, "cases.tsv"), "w") as fh:
        fh.write("# case_index kind mode mode_name block_tag block_index row slot expect "
                 "expect_name source exactness dst_off exp_off n_floats combine_src\n")
        for c in plan.cases:
            bt = BLOCKS[c["block"]][0]
            fh.write("\t".join(str(x) for x in (
                c["case_index"], c["kind"], c["mode"], MODE_NAME[c["mode"]], c["block"],
                BLOCKS[c["block"]][1], c["row"], c["slot"], c["expect"], EXP_NAME[c["expect"]],
                c["src"], c["exact"], c["dst_off"], c["exp_off"], c["n_floats"],
                c["combine_src"])) + "\n")
    with open(os.path.join(work, "units.tsv"), "w") as fh:
        fh.write("# unit kind declared block_tag role expert src_off slot_bytes n_blocks rows "
                 "set_pos main_n cls tensor_index width\n")
        for u in plan.units:
            fh.write("\t".join(str(x) for x in (
                u.n, u["kind"], u["declared_kind"], u["block"], u["role"], u["expert"],
                u["src_off"], u["slot_bytes"], u["n_blocks"], u["rows"], u["set_pos"], u["main_n"],
                u["cls"], u["tensor_index"], u.tname)) + "\n")
    with open(os.path.join(work, "rows.tsv"), "w") as fh:
        fh.write("# row_idx block_tag token fixture id_asc w_by_eid_fixed ids\n")
        for r in plan.rows:
            fh.write("\t".join(str(x) for x in (r["index"], r["block"], r["row"], r["fixture"],
                                                r["id_asc"], r["w_by_eid_fixed"],
                                                ",".join(str(v) for v in r["ids"]))) + "\n")
    return len(plan.cases)


# -----------------------------------------------------------------------------------------------
# PREDICTION 3 — the five candidate readings of the fused gate_up_proj, and whether the data can
# tell them apart at all.
#
# README prediction 3 claims `gate_up_proj` is [E, 2I, H] with `gate, up = chunk(2)` — gate FIRST.
# The router lane recorded this as UNRESOLVED on a probe that memmapped the shard eight bytes early
# (tmp/verify_p3_offset.py proved that: the shifted read corrupts 523 841 of 524 288 elements and
# every one of the 256 rows, so NO reading could have matched). With the offset corrected, the pinned
# `expert_raw` reproduces bit-exactly on the sanctioned path. What was still missing — and what this
# section supplies — is the DISTINCTION: a candidate that merely reproduces has not been shown to be
# a different function from its neighbours.
#
# So two things are required per block, and both are printed:
#   reproduction   every candidate's expert output, from ONE shared apply function so the arithmetic
#                  is held fixed and only the reading varies, against the pinned `expert_raw` as a
#                  bf16 bit image; "reproduces" means within the cost of a legitimate re-association
#                  of the same arithmetic (measured here, never typed).
#   distinctness   the candidates compared as ADDRESS MAPS, element by element. This is an exact
#                  statement about the functions, with no numerical confounding, and it is the only
#                  thing that can catch two spellings that are secretly the same function — in which
#                  case the data cannot distinguish them and saying so is the honest verdict.
# -----------------------------------------------------------------------------------------------
P3_BLOCKS = ["layer0", "layer1", "layer39", "mtp0"]
P3_CANDIDATES = ["output_chunk_gate_first", "output_chunk_up_first", "input_chunk_gate_even",
                 "input_chunk_gate_odd", "stack_transposed"]


def p3_row_offsets(gu_total, E, two_i, I, H):
    """Per candidate: {expert: (gate_row_starts, up_row_starts)} as element offsets into the flat
    gate_up run. Offsets are GLOBAL, which is what lets `stack_transposed` — a reading that crosses
    expert boundaries — be compared on the same footing as the ones that do not."""
    out = {}
    for name in P3_CANDIDATES:
        per = {}
        for e in range(E):
            base = e * two_i * H
            i = np.arange(I, dtype=np.int64)
            if name == "output_chunk_gate_first":
                per[e] = (base + i * H, base + (I + i) * H)
            elif name == "output_chunk_up_first":
                per[e] = (base + (I + i) * H, base + i * H)
            elif name == "input_chunk_gate_even":      # [2I,H] read as [I,2H], chunked on the input
                per[e] = (base + (2 * i) * H, base + (2 * i + 1) * H)
            elif name == "input_chunk_gate_odd":       # the other assignment of the same interleave
                per[e] = (base + (2 * i + 1) * H, base + (2 * i) * H)
            elif name == "stack_transposed":          # the buffer is really [2, E, I, H]
                per[e] = (e * I * H + i * H, E * I * H + e * I * H + i * H)
            else:
                raise GateError("no offset map for candidate %s" % name)
        out[name] = per
    return out


def p3_block(tag, tokens, work):
    import torch
    torch.set_grad_enabled(False)
    g = moe_ref.load_geometry()
    idx = moe_ref.load_index()
    bo = moe_ref.Block.parse(tag)
    E, I, H, two_i = g.num_experts, g.moe_intermediate_size, g.hidden_size, g.two_i

    def flat(role):
        name = bo.prefix + moe_ref.SUFFIXES[role]
        shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(name, idx))
        head, _hlen = moe_ref.read_header(shard)
        # The data region begins AFTER the 8-byte little-endian header length. Missing those eight
        # bytes is the bug that made the earlier probe unreadable (HANDOFF §9).
        start = 8 + _hlen + head[name]["data_offsets"][0]
        n = 1
        for d in head[name]["shape"]:
            n *= int(d)
        return np.memmap(shard, dtype=np.uint16, mode="r", offset=start, shape=(n,))

    gu, dn = flat("gate_up"), flat("down")
    # dump_arrays returns (f32 view, uint16 bit image); every comparison in this section is a bit-image
    # comparison, so the SECOND element is the one wanted (pitfalls/30 §".view vs .to").
    _, x_u = dump_arrays(tag + ".input")
    _, raw_u = dump_arrays(tag + ".expert_raw")
    ids, _ = dump_arrays(tag + ".topk_ids")
    n_tok = min(tokens, raw_u.shape[0])
    pairs = [(t, k) for t in range(n_tok) for k in range(g.top_k)]
    experts = sorted({int(ids[t][k]) for t, k in pairs})

    # --- the bound: what a LEGITIMATE re-association of the reference's own arithmetic costs.
    # Two canonical batchings are measured here; the hand-rolled single-row path the candidates share
    # is added to the floor after the candidate pass, because their fair noise floor is the largest
    # deviation any legal re-association of THIS arithmetic produces on THIS data. Measured, not typed.
    xt = router_oracle.u16_to_torch_bf16(np.asarray(x_u[:n_tok], dtype=np.uint16))
    w = moe_ref.load_block_weights(bo, g)
    leg = {}
    for label, form in (("canonical_grouped", "ref"),
                        ("canonical_per_slot", moe_ref.Form(batching="per_slot"))):
        got = moe_ref.moe_block(xt, w, g, form)["expert_raw"].view(torch.uint16).numpy()
        per_row = np.count_nonzero(got != raw_u[:n_tok], axis=-1)
        leg[label] = int(per_row.sum())
    leg["handrolled_reference"] = 0        # filled after the candidate pass

    offs = p3_row_offsets(gu.size, E, two_i, I, H)
    xrow = {t: torch.from_numpy(np.ascontiguousarray(x_u[t], dtype=np.uint16)).view(torch.bfloat16)
            for t in range(n_tok)}
    dmat = {}
    for e in experts:                        # the down projection is the same for every candidate
        blk = np.ascontiguousarray(dn[e * H * I:(e + 1) * H * I], dtype=np.uint16)
        dmat[e] = torch.from_numpy(blk).view(torch.bfloat16).reshape(H, I)

    def apply(e, gsel, usel):
        gm = torch.from_numpy(np.ascontiguousarray(gu[gsel[:, None] + np.arange(H)],
                                                   dtype=np.uint16)).view(torch.bfloat16)
        um = torch.from_numpy(np.ascontiguousarray(gu[usel[:, None] + np.arange(H)],
                                                   dtype=np.uint16)).view(torch.bfloat16)
        res = {}
        for t, k in pairs:
            if int(ids[t][k]) != e:
                continue
            x = xrow[t][None, :]
            gg = x @ gm.t()
            uu = x @ um.t()
            s = torch.nn.functional.silu(gg) * uu
            res[(t, k)] = (s @ dmat[e].t()).view(torch.uint16).reshape(-1).numpy()
        return res

    per_cand, elem_total = {}, n_tok * g.top_k * H
    for name in P3_CANDIDATES:
        diff = 0
        rows_bad = 0
        rows = 0
        for e in experts:
            gs, us = offs[name][e]
            got = apply(e, gs, us)
            for (t, k), v in got.items():
                nd = int(np.count_nonzero(v != raw_u[t][k]))
                diff += nd
                rows += 1
                rows_bad += 1 if nd else 0
        per_cand[name] = dict(diff=diff, rows=rows, rows_bad=rows_bad, elements=elem_total)
    # --- distinctness as address maps: exact, and independent of any arithmetic -------------------
    matrix = []
    for a in range(len(P3_CANDIDATES)):
        for b in range(a + 1, len(P3_CANDIDATES)):
            na, nb = P3_CANDIDATES[a], P3_CANDIDATES[b]
            same = all(np.array_equal(offs[na][e][0], offs[nb][e][0])
                       and np.array_equal(offs[na][e][1], offs[nb][e][1]) for e in experts)
            nd = 0
            for e in experts:
                ga, ua = offs[na][e]
                gb, ub = offs[nb][e]
                nd += int(np.count_nonzero(ga != gb)) + int(np.count_nonzero(ua != ub))
            matrix.append((na, nb, 1 if same else 0, nd))
    # --- the verdict ------------------------------------------------------------------------------
    leg["handrolled_reference"] = per_cand["output_chunk_gate_first"]["diff"]
    bound = max(leg.values())
    ok = [n for n, r in per_cand.items() if r["diff"] <= bound]
    dupes = ["%s==%s" % (a, b) for a, b, same, nd in matrix if same]
    if len(ok) == 1 and ok[0] == "output_chunk_gate_first":
        verdict = "PREDICTED_AND_CONFIRMED"
    elif len(ok) > 1:
        verdict = "UNRESOLVED"
    else:
        verdict = "UNRESOLVED"
    return dict(tag=tag, bound=bound, leg=leg, per=per_cand, matrix=matrix, ok=ok, dupes=dupes,
                verdict=verdict, experts=len(experts), pairs=len(pairs),
                note="layer2 is not on disk (shard 3 absent) and is therefore unprobed, not skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--max-rows", type=int, default=1,
                    help="pinned tokens per sampled real block; a CUT in numbers only")
    ap.add_argument("--p3-blocks", default=",".join(P3_BLOCKS))
    ap.add_argument("--p3-tokens", type=int, default=8)
    ap.add_argument("--skip-p3", action="store_true")
    a = ap.parse_args()
    work = a.work if os.path.isabs(a.work) else os.path.join(HERE, a.work)
    os.makedirs(work, exist_ok=True)

    geo = reconcile_geometry()
    print("GEOMETRY H=%d I=%d K=%d E=%d (checkpoints/Qwen3.6-35B-A3B.config.json, reconciled "
          "against %s)" % (geo["hidden_size"], geo["moe_intermediate_size"],
                           geo["num_experts_per_tok"], geo["num_experts"],
                           os.path.basename(KERNELS_H)))
    man, by = router_oracle.load_manifest()
    global DUMP
    DUMP = by
    print("MANIFEST file=%s records=%d" % (os.path.basename(apq.SELFTEST_MARK), len(by)))
    require_audit_marker()
    gg = gr.Gguf(GGUF_PATH)
    print("GGUF tensors=%d data_start=%d" % (gg.n_tensors, gg.data_start))

    plan = Plan(work, a.max_rows)
    meta = build_sample(plan, gg)
    nrows, exp_floats = write_plan_files(plan, work)
    ncases = write_inventory(plan, work, meta)

    # ---- sample declaration (printed here so the device run inherits exactly this) ---------------
    real_rows = sum(1 for r in plan.rows if r["fixture"] == 0)
    synth_rows = sum(1 for r in plan.rows if r["fixture"] == 1)
    classes = {}
    for c in plan.cases:
        classes[EXP_NAME[c["expect"]]] = classes.get(EXP_NAME[c["expect"]], 0) + 1
    print("SAMPLE blocks=%d real_rows=%d synthetic_rows=%d units=%d cases=%d "
          "expert_cases=%d combine_cases=%d slab_bytes=%d out_floats=%d expect_floats=%d"
          % (len(plan.blocks), real_rows, synth_rows, len(plan.units), len(plan.cases),
             sum(1 for c in plan.cases if c["kind"] == 0),
             sum(1 for c in plan.cases if c["kind"] == 1),
             len(plan.slab.buf), plan.out_len, plan.expect_len))
    print("CLASSES " + " ".join("%s=%d" % (k, classes.get(k, 0))
                                for k in ("MG_EXP_MATCH", "MG_EXP_MISMATCH", "MG_EXP_ALT",
                                          "MG_EXP_IMMUNE", "MG_EXP_REPORT")))
    missing = sorted(MODE_NAME[m] for m in set(MODES.values()) - plan.modes_used)
    for b in plan.blocks:
        ne = sum(1 for c in plan.cases if c["block"] == plan.blocks.index(b) and c["kind"] == 0)
        print("BLOCK tag=%s index=%d is_mtp=%d main_n=%d real_rows=%d experts=%d cases=[%d,%d] "
              "expert_cases=%d" % (b["tag"], b["index"], b["is_mtp"], b["main_n"], b["n_real"],
                                   b["n_experts"], b["case_lo"], b["case_hi"], ne))
    print("PLANROWS rows=%d parsed_field_counts=U%d/R%d/B%d/C%d"
          % (nrows, HC["MG_PLAN_FIELDS_U"], HC["MG_PLAN_FIELDS_R"], HC["MG_PLAN_FIELDS_B"],
             HC["MG_PLAN_FIELDS_C"]))
    print("EXPECT file=%s floats=%d" % (os.path.join(work, "expect.f32"), exp_floats))
    if missing:
        raise GateError("the plan has no case for %d of the %d modes: %s — an unexercised mode is "
                        "an unexercised gate (pitfalls/40)"
                        % (len(missing), HC["MG_MODE_COUNT"], ", ".join(missing)))
    print("MODECOVERAGE modes=%d covered=%d (the enum came from the header, not from a list here)"
          % (HC["MG_MODE_COUNT"], len(plan.modes_used)))

    # the oracle's own combine form, judged against the pinned routed_sum
    fd = sum(x[2] for x in FORM_CHECK)
    fe = sum(x[3] for x in FORM_CHECK)
    print("COMBINEFORM rows=%d elements=%d differing=%d -> the expectation the combine cases are "
          "judged by reproduces the pinned routed_sum%s"
          % (len(FORM_CHECK), fe, fd, "" if fd == 0 else " NOT; the oracle is wrong"))
    if fd:
        raise GateError("the oracle's combine form differs from pinned routed_sum in %d of %d "
                        "elements; refusing to judge the device against a broken ruler" % (fd, fe))

    with open(os.path.join(work, "cases.tsv"), "a") as fh:
        pass

    # ---- prediction 3 -----------------------------------------------------------------------------
    if not a.skip_p3:
        tags = [t for t in a.p3_blocks.split(",") if t]
        lines_c, lines_m = [], []
        verdicts = {}
        for tag in tags:
            r = p3_block(tag, a.p3_tokens, work)
            verdicts[tag] = r
            print("P3 block=%s pairs=%d experts=%d elements_per_candidate=%d "
                  "legitimate_reassociation=%d (grouped=%d per_slot=%d hand_rolled=%d)"
                  % (tag, r["pairs"], r["experts"], list(r["per"].values())[0]["elements"],
                     r["bound"], r["leg"]["canonical_grouped"], r["leg"]["canonical_per_slot"],
                     r["leg"]["handrolled_reference"]))
            for name in P3_CANDIDATES:
                p = r["per"][name]
                keeps = p["diff"] <= r["bound"]
                print("P3CAND block=%s cand=%-24s differing=%7d/%d rows_bad=%d/%d "
                      "ratio_to_bound=%s reproduces=%d"
                      % (tag, name, p["diff"], p["elements"], p["rows_bad"], p["rows"],
                         ("inf" if r["bound"] == 0 else "%.1f" % (p["diff"] / float(max(r["bound"], 1)))),
                         1 if keeps else 0))
                lines_c.append("%s\t%s\t%d\t%d\t%d\t%d\t%d\n" % (
                    tag, name, p["diff"], p["elements"], p["rows_bad"], p["rows"],
                    1 if keeps else 0))
            for na, nb, same, nd in r["matrix"]:
                lines_m.append("%s\t%s\t%s\t%d\t%d\n" % (tag, na, nb, same, nd))
            ndist = sum(1 for _, _, s, _ in r["matrix"] if not s)
            print("P3DISTINCT block=%s pairs_of_candidates=%d distinguished=%d identical=%d [%s]"
                  % (tag, len(r["matrix"]), ndist, len(r["matrix"]) - ndist, ",".join(r["dupes"])))
            print("P3VERDICT block=%s consistent_candidates=[%s] verdict=%s"
                  % (tag, ",".join(r["ok"]), r["verdict"]))
        open(os.path.join(work, "oracle_p3.tsv"), "w").write(
            "# block candidate differing elements rows_bad rows reproduces\n" + "".join(lines_c))
        open(os.path.join(work, "oracle_p3_matrix.tsv"), "w").write(
            "# block candidate_a candidate_b same_address_map differing_row_starts\n"
            + "".join(lines_m))
        print("P3RECORDS candidates=%d distinct_pairs=%d files=oracle_p3.tsv,oracle_p3_matrix.tsv"
              % (len(lines_c), len(lines_m)))
        allv = set(v["verdict"] for v in verdicts.values())
        print("P3OVERALL verdicts=%s blocks=%d unprobed_blocks=layer2(not on disk)"
              % (",".join(sorted(allv)), len(verdicts)))

    # ---- the floor table: every band in this lane comes from here, never from a typed number ------
    with open(os.path.join(work, "oracle_floor.tsv"), "w") as fh:
        fh.write("# figure value source\n")
        fh.write("combine_form_vs_pinned_routed_sum\t%d\tthis run, %d elements over %d real rows\n"
                 % (fd, fe, len(FORM_CHECK)))
        for tag, r in sorted((verdicts or {}).items()) if not a.skip_p3 else []:
            fh.write("legitimate_reassociation_%s\t%d\tspeccheck Form batching grouped vs per_slot, "
                     "this run\n" % (tag, r["bound"]))
            fh.write("f32_then_round_%s\tsee tmp/verify_p3_offset.py\t308076 of 524288 on layer0; "
                     "an f32 accumulate does not reproduce the dump\n" % tag)
        fh.write("exactness_bound_synthetic\t16777216\t2**24: every intermediate of the synthetic "
                 "fixture is an integer below it, so f32 addition is exact and MG_EXP_MATCH is "
                 "licensed (re-derived per case by gate.py)\n")
        fh.write("silu_exact_threshold\t%s\tMG_EXACT_EXP_MIN in %s: at or above it expf(-z) flushes "
                 "to zero and silu is the identity on both routes\n"
                 % (open(KERNELS_H).read().split("MG_EXACT_EXP_MIN ")[1].split()[0],
                    os.path.basename(KERNELS_H)))
    print("FLOOR file=oracle_floor.tsv rows=%d"
          % (sum(1 for _ in open(os.path.join(work, "oracle_floor.tsv"))) - 1))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as e:
        print("ORACLE FAIL %s" % e)
        sys.exit(1)
