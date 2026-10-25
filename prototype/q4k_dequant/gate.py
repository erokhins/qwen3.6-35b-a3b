#!/usr/bin/env python3
"""gate.py — the device gate for `prototype/q4k_dequant`: Q4_K and Q8_0 blocks dequantized on the
iGPU must come out **bit-identical** to this repo's audited CPU reader, and every injected
misreading must fire only where it belongs.

PLAN.md §"Step 2 and beyond" names this as the engine's first device piece, unblocked because the
oracle already exists: `tools/gguf_reader.py` + `tools/audit_published_quant.py` passed the 1.0
selftest (40/40 synthetic blocks bit-exact, 6/6 must-catch, 0 wrongly-fired) and the published-file
injections 150/150 with 0 leaks (`quantization/evidence/audit-selftest-2026-10-26.json`,
`quantization/evidence/audit-injections-2026-10-26.json`, 2026-10-26). This gate witnesses a device
implementation against an implementation that has already been witnessed — not against itself.

What is compared
  the device kernel (hipcc, gfx1151) against `tools/gguf_reader.py`'s dequant, on the SAME bytes,
  judged on the 4 bytes of `struct.pack('<f', v)` — one round-to-nearest of the reader's double
  result. The kernel's f32 arithmetic is exact for both formats (`q4k_dequant_kernels.h`
  §"Why the arithmetic below is plain f32"), so there is no tolerance anywhere in this file.

Where the sample comes from
  every tensor and every block index is read out of the GGUF's own tensor-info array and reconciled
  against a census *derived* from `checkpoints/Qwen3.6-35B-A3B.config.json` (256 experts,
  moe_intermediate_size 512, hidden_size 2048, 40 layers with full_attention_interval 4, vocab
  248320). Nothing here is a hand-typed block list (`pitfalls/40-gates-and-admissions.md`
  §"An admission over a fixture list typed from memory is not an admission").

Discipline
  correctness only: no timer, no warm loop, no burst, no best-of-N, and no ms / GB/s / tok/s figure
  in any line this file prints — this module imports no timer at all, and the lock record's wall
  clock is printed by the `gate` wrapper, not here. `gate` (the shell wrapper) alone owns the
  machine-wide GPU lock; **this file never acquires it** (`pitfalls/50-harness-and-shell.md`
  §"Self-locking gate drivers deadlock silently"), and the lock tool's filename is deliberately not
  spelled anywhere in this file so that `grep -c` for it returns 0. It also never pipes the verdict:
  the driver's stdout goes to a file that is then parsed, and the subprocess return code is checked
  directly (`pitfalls/50` §"Piping a gate through `tail` swallows its exit code").
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

import gguf_reader as gr                                    # noqa: E402  the reference dequantizer
import audit_published_quant as apq                         # noqa: E402  marker + sampler + geometry

BIN = os.path.join(HERE, "q4k_dequant")
PROBE_BIN = os.path.join(HERE, "host_probe")
WORK = os.path.join(HERE, "work")
EV = os.path.join(HERE, "evidence")
DAY = "2026-10-26"
EVIDENCE_JSON = os.path.join(EV, "q4k-dequant-device-%s.json" % DAY)
SAMPLE_MD = os.path.join(EV, "q4k-dequant-sample-%s.md" % DAY)

# ---- the device enum, mirrored from q4k_dequant_kernels.h ---------------------------------------
# One spelling per concept: these integers are the header's, asserted against it by
# check_header_enum(), so a renumbered mode cannot quietly make the gate judge something else.
KIND_Q4K, KIND_Q8_0 = 0, 1
EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE = 0, 1, 2, 3
EXP_NAME = {EXP_MATCH: "MATCH", EXP_MISMATCH: "MISMATCH", EXP_ALT: "ALT", EXP_IMMUNE: "IMMUNE"}
MODES = [
    # (name, kind-or-None, class)   class: ref | arith | addr
    ("correct", None, "ref"),
    ("q4k_as_q4nl", "Q4_K", "arith"),
    ("q4k_drop_min", "Q4_K", "arith"),
    ("q4k_naive_sc_min", "Q4_K", "arith"),
    ("q4k_subblock_rot3", "Q4_K", "arith"),
    ("q4k_block_off_1", "Q4_K", "addr"),
    ("q4k_byte_shift_1", "Q4_K", "addr"),
    ("q4k_xpose_expert", "Q4_K", "addr"),
    ("q8_0_block_off_1", "Q8_0", "addr"),
    ("q8_0_byte_shift_1", "Q8_0", "addr"),
]
MODE_ID = {m[0]: i for i, m in enumerate(MODES)}
MODE_CLASS = {m[0]: m[2] for m in MODES}
MODE_KIND = {m[0]: m[1] for m in MODES}
BLOCK_BYTES = {"Q4_K": 144, "Q8_0": 34}          # asserted against gr.BLOCK and against the header
BLOCK_ELTS = {"Q4_K": 256, "Q8_0": 32}
# The header spells the block-size macros after ggml's type names, which squash the underscore
# (Q4_K -> Q4K) except in Q8_0. Deriving the macro name from the GGUF type string is how this
# check first went red: a #define it could not see is a #define that cannot disagree.
TYPE_MACRO = {"Q4_K": "Q4K", "Q8_0": "Q8_0", "Q4_NL": "Q4NL"}
CLS_REAL, CLS_SYNTH, CLS_SUPPORT = 0, 1, 2

# The tensors the brief names, as membership requirements ON the derived sample — not as the sample.
NAMED_REQUIRED = ["blk.0.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight",
                  "blk.0.ssm_out.weight", "token_embd.weight"]

WANT_Q4K_ELTS = 6144        # 24 blocks of 256, stratified over the expert axis by apq.sampled_blocks
WANT_Q8_0_ELTS = 768        # 24 blocks of 32
TENSORS_PER_FAMILY = 3


# -----------------------------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------------------------
class GateError(Exception):
    pass


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def floats_to_hex(vals):
    return ",".join("%08x" % f32bits(v) for v in vals)


def check_header_enum():
    """Gate and kernel must agree on the integers, or the gate judges another binary. The header is
    the source of truth: this parses its enum bodies instead of trusting the copy above."""
    text = open(os.path.join(HERE, "q4k_dequant_kernels.h")).read()

    def enum_body(first):
        # The capture includes the first enumerator: a body that starts after it would report the
        # first name as missing, which is exactly how this check first went red.
        m = re.search(r"enum \{\s*(" + first + r"\s*=\s*0,.*?)\};", text, re.S)
        if not m:
            raise GateError("cannot find the enum starting at %s in q4k_dequant_kernels.h" % first)
        body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)     # comments are prose, not facts
        return body

    def vals(body):
        return {k: int(v, 0) for k, v in
                re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)", body)}

    kinds = vals(enum_body("Q4KD_KIND_Q4K"))
    exps = vals(enum_body("Q4KD_EXP_MATCH"))
    modes = vals(enum_body("Q4KD_MODE_CORRECT"))
    for tbl, want, label in ((kinds, {"Q4KD_KIND_Q4K": KIND_Q4K, "Q4KD_KIND_Q8_0": KIND_Q8_0},
                              "kind"),
                             (exps, {"Q4KD_EXP_MATCH": EXP_MATCH,
                                     "Q4KD_EXP_MISMATCH": EXP_MISMATCH,
                                     "Q4KD_EXP_ALT": EXP_ALT,
                                     "Q4KD_EXP_IMMUNE": EXP_IMMUNE}, "expect"),
                             (modes, {"Q4KD_MODE_" + n.upper(): i for n, i in MODE_ID.items()},
                              "mode")):
        for k, v in want.items():
            if tbl.get(k) != v:
                raise GateError("%s enum disagrees: header %s=%s, gate says %s"
                                % (label, k, tbl.get(k), v))
    if modes.get("Q4KD_MODE_COUNT") != len(MODES):
        raise GateError("Q4KD_MODE_COUNT=%s but the gate has %d modes"
                        % (modes.get("Q4KD_MODE_COUNT"), len(MODES)))
    for t, bb in BLOCK_BYTES.items():
        if gr.BLOCK[t][1] != bb or gr.BLOCK[t][0] != BLOCK_ELTS[t]:
            raise GateError("block geometry for %s: reader says %s, gate says %s"
                            % (t, gr.BLOCK[t], (BLOCK_ELTS[t], bb)))
        m = re.search(r"#define Q4KD_%s_BYTES\s+(\d+)" % TYPE_MACRO[t], text)
        if not m or int(m.group(1)) != bb:
            raise GateError("Q4KD_%s_BYTES disagrees with the reader's derived geometry" % t)
    return len(MODES)


def require_audit_marker():
    """The audit's own gate, mirrored (tools/audit_published_quant.py §require_selftest) — but
    without appending to its ordering log, which belongs to the step-1 audit. The property is kept:
    not one tensor-data byte is read unless the synthetic-block selftest passed against the CURRENT
    bytes of that gate and this reader."""
    if not os.path.exists(apq.SELFTEST_MARK):
        raise GateError("no selftest marker at %s — run: python3 tools/audit_published_quant.py "
                        "--selftest" % apq.SELFTEST_MARK)
    mark = json.load(open(apq.SELFTEST_MARK))
    cur_gate = sha256_file(os.path.join(TOOLS, "audit_published_quant.py"))
    cur_reader = sha256_file(os.path.join(TOOLS, "gguf_reader.py"))
    if mark.get("gate_sha256") != cur_gate or mark.get("reader_sha256") != cur_reader:
        raise GateError("the selftest marker is STALE (recorded %s/%s, current %s/%s). Re-run: "
                        "python3 tools/audit_published_quant.py --selftest"
                        % (mark.get("gate_sha256"), mark.get("reader_sha256"), cur_gate, cur_reader))
    gr.grant_selftest_token(apq.SELFTEST_TOKEN)
    return mark


# -----------------------------------------------------------------------------------------------
# geometry, derived from the committed config and then reconciled with the file's own records
# -----------------------------------------------------------------------------------------------
def derived_census():
    cfg = json.load(open(os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")))
    t = cfg["text_config"]
    H, E, I, Is = t["hidden_size"], t["num_experts"], t["moe_intermediate_size"], \
        t["shared_expert_intermediate_size"]
    L = t["num_hidden_layers"]
    nq, nkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    nkh, khd, nv, hdv = (t["linear_num_key_heads"], t["linear_key_head_dim"],
                         t["linear_num_value_heads"], t["linear_value_head_dim"])
    vocab = t["vocab_size"]
    layers = t["layer_types"]
    full = [i for i, x in enumerate(layers) if x == "full_attention"]
    lin = [i for i, x in enumerate(layers) if x == "linear_attention"]
    # attn_output_gate: true (text_config) makes the Q projection emit the gate alongside the
    # queries, so attn_q carries 2*n_heads*head_dim rows while attn_output keeps n_heads*head_dim.
    # Derived from the flag, not from the file's dims: the file is what this census has to reconcile
    # against, so a shape copied out of it would reconcile with itself.
    qrows = nq * hd * (2 if t["attn_output_gate"] else 1)
    spec = [
        ("routed gate", "blk.%d.ffn_gate_exps.weight", "Q4_K", list(range(L)), [H, I, E]),
        ("routed up", "blk.%d.ffn_up_exps.weight", "Q4_K", list(range(L)), [H, I, E]),
        ("shared gate", "blk.%d.ffn_gate_shexp.weight", "Q8_0", list(range(L)), [H, Is]),
        ("shared up", "blk.%d.ffn_up_shexp.weight", "Q8_0", list(range(L)), [H, Is]),
        ("shared down", "blk.%d.ffn_down_shexp.weight", "Q8_0", list(range(L)), [Is, H]),
        ("attn q", "blk.%d.attn_q.weight", "Q8_0", full, [H, qrows]),
        ("attn k", "blk.%d.attn_k.weight", "Q8_0", full, [H, nkv * hd]),
        ("attn v", "blk.%d.attn_v.weight", "Q8_0", full, [H, nkv * hd]),
        ("attn output", "blk.%d.attn_output.weight", "Q8_0", full, [nq * hd, H]),
        ("GDN qkv", "blk.%d.attn_qkv.weight", "Q8_0", lin, [H, 2 * nkh * khd + nv * hdv]),
        ("GDN z-gate", "blk.%d.attn_gate.weight", "Q8_0", lin, [H, nv * hdv]),
        ("GDN out_proj", "blk.%d.ssm_out.weight", "Q8_0", lin, [nv * hdv, H]),
    ]
    whole = [("embeddings", "token_embd.weight", "Q8_0", [H, vocab])]
    keys = ("num_experts", "moe_intermediate_size", "hidden_size", "num_hidden_layers",
            "vocab_size", "full_attention_interval", "attn_output_gate", "num_attention_heads",
            "head_dim")
    return {"spec": spec, "whole": whole, "L": L, "E": E, "I": I, "H": H, "vocab": vocab,
            "full": full, "lin": lin, "config_keys": {k: t[k] for k in keys}}


def expected_tensors(geo):
    """{(name): (type, dims, family)} derived from the config — the census the file must match."""
    out = {}
    for fam, tmpl, tname, layers, dims in geo["spec"]:
        for i in layers:
            out[tmpl % i] = (tname, tuple(dims), fam)
    for fam, name, tname, dims in geo["whole"]:
        out[name] = (tname, tuple(dims), fam)
    return out


def reconcile_census(gg, geo):
    """Every expected tensor must be in the file with the expected type and dims, and every Q4_K or
    Q8_0 tensor in the file must be claimed by exactly one expected name. A census that reconciles is
    what makes the sample *derived*; without it the sample would be a guess about the file. Prints
    the census it checked, and aborts before any value byte is read."""
    if gg.n_tensors != apq.EXPECTED_TENSOR_RECORDS or len(gg.tensors) != gg.n_tensors:
        raise GateError("tensor record count %d/%d != expected %d"
                        % (gg.n_tensors, len(gg.tensors), apq.EXPECTED_TENSOR_RECORDS))
    exp = expected_tensors(geo)
    for n in sorted(exp):
        if n not in gg.by_name:
            raise GateError("expected tensor %s is not in the file" % n)
        t = gg.by_name[n]
        tname, dims, fam = exp[n]
        if t.type_name != tname:
            raise GateError("tensor %s is stored %s, derived %s" % (n, t.type_name, tname))
        if t.dims != dims:
            raise GateError("tensor %s has GGUF dims %s, derived %s" % (n, t.dims, dims))
        if t.n_elts % BLOCK_ELTS[tname]:
            raise GateError("tensor %s holds %d elements, not a whole number of %s blocks"
                            % (n, t.n_elts, tname))
    extra = [(t.name, t.type_name) for t in gg.tensors
             if t.type_name in ("Q4_K", "Q8_0") and t.name not in exp]
    if extra:
        raise GateError("%d Q4_K/Q8_0 tensors are in the file but not in the derived census; "
                        "first: %s" % (len(extra), extra[:5]))
    fam_count = {}
    for n, (tname, dims, fam) in exp.items():
        fam_count[(fam, tname)] = fam_count.get((fam, tname), 0) + 1
    rows = sorted((fam, tname, c) for (fam, tname), c in fam_count.items())
    print("CENSUS families=%d tensors=%d records=%d" % (len(rows), len(exp), gg.n_tensors))
    for fam, tname, c in rows:
        print("CENSUS family=%-14s type=%-5s tensors=%d" % (fam, tname, c))
    if len(rows) == 0 or len(exp) == 0:
        raise GateError("the derived census is empty — nothing was reconciled")
    return rows


def select_samples(gg, geo):
    """(family, tensor, type) triples to sample, derived from the reconciled census: even-spaced over
    each family's own tensor list. The brief's named tensors are then *required* to be in the
    derived set — asserted, never silently appended."""
    reconcile_census(gg, geo)
    by_fam = {}
    for name, (tname, dims, fam) in expected_tensors(geo).items():
        by_fam.setdefault((fam, tname), []).append(name)
    picked, seen = [], set()
    for key in sorted(by_fam):
        names = sorted(by_fam[key])
        sel = set(names[i] for i in apq.even_spaced(len(names),
                                                   min(TENSORS_PER_FAMILY, len(names))))
        sel |= (set(NAMED_REQUIRED) & set(names))   # the brief names these, so the RULE includes
        for nm in sorted(sel):                      # them; it is not a special case for them
            if nm not in seen:
                picked.append((key[0], nm, key[1]))
                seen.add(nm)
    for nm in NAMED_REQUIRED:
        if nm not in seen:
            raise GateError("the brief names %s as a sampled tensor and the derived selection did "
                            "not include it — fix the selection rule, do not special-case it" % nm)
    return picked


def xpose_index(block_index, nb_per_row, n_rows, n_outer):
    """The map the device computes in q4kd_resolve_src, written here independently:
    correct b = (e*n_rows + n)*nb_per_row + j  ->  transposed b' = (j*n_rows + n)*n_outer + e."""
    j = block_index % nb_per_row
    t = block_index // nb_per_row
    n = t % n_rows
    e = t // n_rows
    return (j * n_rows + n) * n_outer + e


# -----------------------------------------------------------------------------------------------
# structural immunity predicates — derived from the format's arithmetic, with no kernel formula
# mirrored here (mirroring the kernel would make the classification self-confirming).
# -----------------------------------------------------------------------------------------------
def pack_scale_min(sc, mg):
    """The inverse of get_scale_min_k4, so a synthetic block can be designed by (sc, mg) values
    instead of by guessing byte patterns. Round-tripped against tools/gguf_reader.py below."""
    q = [0] * 12
    for j in range(4):
        q[j] |= sc[j] & 63
        q[j + 4] |= mg[j] & 63
    for j in range(4, 8):
        q[j] |= (mg[j] >> 4) << 6                     # top 2 bits of mg[j]
        q[j - 4] |= ((sc[j] >> 4) & 3) << 6           # top 2 bits of sc[j]
        # byte j+4 carries two low nibbles: sc[j]'s low 4 bits low, mg[j]'s low 4 bits high. The
        # top 2 bits of each live elsewhere (the two lines above), which is what makes this layout
        # worth a round-trip assert instead of a eyeball.
        q[j + 4] |= (sc[j] & 15) | ((mg[j] & 15) << 4)
    return bytes(q)


def build_q4k(d_bits, dm_bits, sc, mg, qs):
    raw = bytearray(144)
    raw[0:2] = struct.pack("<H", d_bits)
    raw[2:4] = struct.pack("<H", dm_bits)
    raw[4:16] = pack_scale_min(sc, mg)
    raw[16:144] = bytes(qs)
    assert gr.qk_scale_min(raw[4:16]) == (list(sc), list(mg)), \
        "pack_scale_min does not round-trip through the reader's unpack"
    assert len(raw) == 144
    return bytes(raw)


def immune_scale_mode(raw, mode):
    """True when the mode provably cannot change any value of this Q4_K block.

    drop_min      value = scale*q - mins becomes scale*q, so the block is unchanged iff every
                  mins_j is +0.0. mins_j = dmin*mg_j is an exact product, so it is zero exactly when
                  dmin == 0 or mg_j == 0 — and it must additionally carry no sign bit, because
                  x - (-0.0) turns a -0.0 result into +0.0, which a bit comparison sees.
    naive, rot3   They replace (sc_j, mg_j). scale = d*sc is exact, so changing sc can only be
                  harmless when d is 0 (every scale is then a signed zero whose sign comes from d);
                  otherwise sc must be unchanged elementwise. Same argument for (dmin, mg).
    as_q4nl       No structural predicate: it reads a different format out of the same bytes, so it
                  is always expected to fire. If it ever does not, this gate goes red — the safe
                  direction for a control."""
    d = gr.f16(raw, 0)
    dmin = gr.f16(raw, 2)
    sc, mg = gr.qk_scale_min(raw[4:16])
    if mode == "q4k_drop_min":
        for j in range(8):
            mins = dmin * mg[j]
            if mins != 0.0 or (struct.pack("<d", mins)[7] & 0x80):
                return False
        return True
    if mode in ("q4k_naive_sc_min", "q4k_subblock_rot3"):
        if mode == "q4k_naive_sc_min":
            def field(bit):
                b0 = raw[4 + (bit >> 3)]
                nb = (bit >> 3) + 1
                b1 = 0 if nb >= 12 else raw[4 + nb]
                return ((b0 | (b1 << 8)) >> (bit & 7)) & 63
            sc2 = [field(6 * j) for j in range(8)]
            mg2 = [field(6 * (j + 8)) for j in range(8)]
        else:
            sc2 = [sc[(j + 3) & 7] for j in range(8)]
            mg2 = [mg[(j + 3) & 7] for j in range(8)]
        scale_ok = (d == 0.0) or all(sc[j] == sc2[j] for j in range(8))
        min_ok = (dmin == 0.0) or all(mg[j] == mg2[j] for j in range(8))
        return scale_ok and min_ok
    return False


# -----------------------------------------------------------------------------------------------
# the synthetic population: constructed so every counter the gate relies on has a case that must
# fire it AND a case that must not. Byte patterns are deliberate, and each one says why.
# -----------------------------------------------------------------------------------------------
SC_DISTINCT = [0, 9, 17, 26, 35, 44, 53, 63]
MG_DISTINCT = [63, 53, 44, 35, 26, 17, 9, 0]
SC_FLAT = [5] * 8
MG_FLAT = [3] * 8
F16_ONE = 0x3C00
F16_HALF = 0x3800
F16_TWO = 0x4000


def synthetic_blocks():
    """-> list of (label, tname, dims_gguf, raw_bytes, why). Every entry is a *fixture*, not a
    measurement; the reader judges them exactly as it judges the real bytes."""
    zs = [0] * 8
    qs_grid = bytes([(i * 37 + 11) & 0xFF for i in range(128)])
    qs_zero = bytes(128)
    out = []
    out.append(("q4k_zero", "Q4_K", (256, 2, 1), build_q4k(0x0000, 0x0000, zs, zs, qs_zero),
                "all-zero block: correct decode is +0.0 everywhere, so drop_min/naive/rot3 are"
                " structurally immune while as_q4nl still fires (0.0*(0-8) is -0.0, a different"
                " bit pattern from +0.0)"))
    out.append(("q4k_zero_b", "Q4_K", (256, 2, 1), build_q4k(0x0000, 0x0000, zs, zs, qs_zero),
                "the second half of the byte-identical pair: an off-by-one read of this block's"
                " neighbour cannot change anything (must-not-catch, derived from byte equality)"))
    out.append(("q4k_grid", "Q4_K", (256, 2, 1),
                build_q4k(F16_ONE, F16_HALF, SC_DISTINCT, MG_DISTINCT, qs_grid),
                "all 8 scales and all 8 mins distinct, nibbles covering every value: fires under"
                " every arithmetic and addressing mode"))
    out.append(("q4k_flat", "Q4_K", (256, 2, 1),
                build_q4k(F16_ONE, F16_HALF, SC_FLAT, MG_FLAT, qs_grid),
                "every scale equal and every min equal: a sub-block rotation cannot change this"
                " block, so rot3 must not fire here while it fires on q4k_grid"))
    out.append(("q4k_minonly", "Q4_K", (256, 2, 1),
                build_q4k(0x0000, F16_ONE, zs, [63] * 8, qs_zero),
                "d=0, dmin=1, every min 63, every q 0: the correct value is exactly -63, so the"
                " dropped min term cannot hide"))
    out.append(("q8_0_zero", "Q8_0", (32, 2), bytes(34),
                "all-zero Q8_0 block with a zero guard byte after it: neither a one-byte shift nor a"
                " one-block shift can change the answer (must-not-catch)"))
    out.append(("q8_0_zero_b", "Q8_0", (32, 2), bytes(34),
                "the second half of the all-zero Q8_0 pair"))
    out.append(("q8_0_d0", "Q8_0", (32, 2),
                bytes(2) + bytes([(i * 23 - 100) & 0xFF for i in range(32)]),
                "d=0 with a non-zero qs: the answer is zero whatever qs says, but a one-byte shift"
                " moves the delta field and must then fire (attribution, not immunity)"))
    out.append(("q8_0_grid", "Q8_0", (32, 2),
                struct.pack("<H", F16_TWO) + bytes([(i * 41 - 77) & 0xFF for i in range(32)]),
                "d=2.0 and a spread of int8 quants: fires under both addressing modes"))
    return out


def synth_groups():
    """Synthetic blocks live in groups of two so the neighbour (off-by-one) rule has a partner to
    wrap against: (q4k_zero, q4k_zero_b), (q4k_grid, q4k_flat), (q8_0_zero, q8_0_zero_b),
    (q8_0_d0, q8_0_grid)."""
    return [["q4k_zero", "q4k_zero_b"], ["q4k_grid", "q4k_flat"],
            ["q8_0_zero", "q8_0_zero_b"], ["q8_0_d0", "q8_0_grid"]]


# -----------------------------------------------------------------------------------------------
# the plan
# -----------------------------------------------------------------------------------------------
class Plan:
    def __init__(self):
        self.units = []
        self.cases = []
        self.slab = bytearray()
        self.probe = []
        self.sample_rows = []
        self.out_floats = 0
        self.launches = 0

    def add_slot(self, block_bytes, raw, guard):
        assert len(raw) == block_bytes, "slab slot must be exactly one block"
        off = len(self.slab)
        self.slab.extend(raw)
        self.slab.append(guard & 0xFF)
        return off

    def new_unit(self, **kw):
        kw["unit"] = len(self.units)
        kw.setdefault("n_blocks", 1)
        kw.setdefault("grp_u0", kw["unit"])
        kw.setdefault("grp_u1", kw["unit"])
        self.units.append(kw)
        return kw

    def new_case(self, **kw):
        kw["case_id"] = len(self.cases)
        self.cases.append(kw)
        return kw


def reader_decode(raw, tname, inj=None):
    """The reference. Exactly tools/gguf_reader.py's dequant, on exactly these bytes."""
    return gr.dequant(raw, tname, n_elts=BLOCK_ELTS[tname], inj=inj)


def next_in_group(plan, unit):
    """Python's independent spelling of the kernel's neighbour rule: the next unit of the group in
    unit-array order, wrapping at the *front* of the group (q4kd_resolve_src: `me >= grp_u1 ? grp_u0
    : me + 1`). Both spellings must agree or the ALT expectations are wrong and the gate says so.

    Wrapping to the front rather than to the unit itself is the point: an off-by-one that wrapped to
    itself would make the last sampled block of every tensor immune to the block-off-by-one injection,
    which is exactly the boundary where a converter spills into the next tensor. The first version of
    this function wrapped to grp_u1 and 40 IMMUNE cases went red on the device for it.
    """
    return plan.units[unit["grp_u0"] if unit["unit"] >= unit["grp_u1"] else unit["unit"] + 1]


def group_unit_at(plan, unit, block_index):
    for i in range(unit["grp_u0"], unit["grp_u1"] + 1):
        if plan.units[i]["block_index"] == block_index:
            return plan.units[i]
    raise GateError("block %d is not in the group of unit %d (the device's binary search would miss"
                    " and write NaN)" % (block_index, unit["unit"]))


def addressed_bytes(plan, unit, mname):
    """The byte string the kernel is expected to read under this addressing mode."""
    if mname.endswith("block_off_1"):
        return next_in_group(plan, unit)["raw"]
    if mname.endswith("byte_shift_1"):
        off = unit["src_off"] + 1
        return bytes(plan.slab[off:off + unit["block_bytes"]])
    if mname == "q4k_xpose_expert":
        bp = xpose_index(unit["block_index"], unit["nb_per_row"], unit["n_rows"], unit["n_outer"])
        return group_unit_at(plan, unit, bp)["raw"]
    raise GateError("unknown addressing mode %s" % mname)


def modes_for(unit):
    if unit["cls"] == CLS_SUPPORT:
        return ["correct"]
    if unit["tname"] == "Q4_K":
        names = ["correct", "q4k_as_q4nl", "q4k_drop_min", "q4k_naive_sc_min", "q4k_subblock_rot3",
                 "q4k_block_off_1", "q4k_byte_shift_1"]
        if unit["xpose_ok"]:
            names.append("q4k_xpose_expert")
        return names
    return ["correct", "q8_0_block_off_1", "q8_0_byte_shift_1"]


def emit_cases(plan, unit):
    own = unit["raw"]
    ref = reader_decode(own, unit["tname"])
    ref_bits = [f32bits(v) for v in ref]
    for mname in modes_for(unit):
        mode = MODE_ID[mname]
        klass = MODE_CLASS[mname]
        if mode == 0:
            expect, why = EXP_MATCH, "the reference case"
        elif klass == "arith":
            if immune_scale_mode(own, mname):
                expect, why = EXP_IMMUNE, "structural immunity: %s cannot change this block" % mname
            else:
                expect, why = EXP_MISMATCH, "arithmetic misreading must change at least one value"
        else:
            src = addressed_bytes(plan, unit, mname)
            if src == own:
                expect, why = EXP_IMMUNE, "the mode addresses the same bytes"
            else:
                alt_bits = [f32bits(v) for v in reader_decode(src, unit["tname"])]
                if alt_bits == ref_bits:
                    expect, why = EXP_IMMUNE, "different bytes, identical decode (structurally mute)"
                else:
                    expect, why = EXP_ALT, ("attribution: equals the reader's decode of the bytes "
                                            "this mode reads")
        plan.new_case(unit=unit["unit"], mode=mode, mname=mname, expect=expect, why=why,
                      n_elts=unit["n_elts"], kind=unit["kind"], cls=unit["cls"],
                      tensor=unit["tensor"], tensor_index=unit["tensor_index"],
                      block_index=unit["block_index"], family=unit["family"],
                      dst_off=-1, cmp_off=-1, ref_bits=ref_bits,
                      alt_bits=([f32bits(v) for v in reader_decode(
                          addressed_bytes(plan, unit, mname), unit["tname"])]
                          if klass == "addr" else None))


def build_plan(gg, geo):
    plan = Plan()
    tok = apq.SELFTEST_TOKEN
    picked = select_samples(gg, geo)
    print("SAMPLE tensors=%d (derived from %d census families)" % (len(picked), len(geo["spec"])))
    for fam, name, tname in picked:
        t = gg.by_name[name]
        be, bb = gr.BLOCK[tname]
        rank = len(t.dims)
        nb_total = t.n_elts // be
        nb_per_row = t.dims[0] // be
        n_rows = t.dims[1] if rank >= 2 else 1
        n_outer = t.dims[2] if rank >= 3 else 1
        blocks, _ = apq.sampled_blocks(t, WANT_Q4K_ELTS if tname == "Q4_K" else WANT_Q8_0_ELTS)
        if not blocks:
            raise GateError("the derived sample produced 0 blocks for %s" % name)
        xpose_eligible = (tname == "Q4_K" and rank >= 3 and nb_per_row > 0 and n_outer > 1)
        member = set(blocks)
        if xpose_eligible:                       # one round of closure is enough: support blocks
            for bi in list(blocks):              # are never transpose-eligible
                member.add(xpose_index(bi, nb_per_row, n_rows, n_outer))
        raws = {}
        for bi in sorted(member):
            raws[bi] = gg.read(tok, gg.data_start + t.offset + bi * bb, bb,
                               why="q4k_dequant device gate, %s block %d" % (name, bi))
            if len(raws[bi]) != bb:
                raise GateError("short read on %s block %d" % (name, bi))
        grp0 = len(plan.units)
        us = {}
        for bi in sorted(member):
            us[bi] = plan.new_unit(
                kind=KIND_Q4K if tname == "Q4_K" else KIND_Q8_0, tname=tname,
                tensor_index=t.index, tensor=name, family=fam,
                cls=CLS_SUPPORT if bi not in blocks else CLS_REAL, block_index=bi,
                n_elts=be, block_bytes=bb, nb_total=nb_total, nb_per_row=nb_per_row,
                n_rows=n_rows, n_outer=n_outer,
                xpose_ok=1 if (xpose_eligible and bi in blocks) else 0, raw=raws[bi],
                src_off=-1, dst_off=-1)
        grp1 = len(plan.units) - 1
        for bi in sorted(member):
            us[bi]["grp_u0"], us[bi]["grp_u1"] = grp0, grp1
        for bi in sorted(member):
            guard = 0
            if bi + 1 < nb_total:                # the byte that really follows this block in the file
                guard = gg.read(tok, gg.data_start + t.offset + (bi + 1) * bb, 1,
                                why="guard byte for the byte-shift mode")[0]
            us[bi]["src_off"] = plan.add_slot(bb, raws[bi], guard)
        plan.sample_rows.append({"family": fam, "tensor": name, "type": tname, "index": t.index,
                                 "dims_gguf": list(t.dims), "dims_hf": list(reversed(t.dims)),
                                 "n_elts": t.n_elts, "nb_total": nb_total,
                                 "blocks_sampled": sorted(blocks), "n_sampled": len(blocks),
                                 "support_blocks": sorted(set(member) - set(blocks)),
                                 "xpose_eligible": int(xpose_eligible)})
    # ---------- synthetic groups ----------
    syn = {label: (tname, dims, raw, why) for label, tname, dims, raw, why in synthetic_blocks()}
    for gi, group in enumerate(synth_groups()):
        tname = syn[group[0]][0]
        be, bb = gr.BLOCK[tname]
        dims = syn[group[0]][1]
        if any(syn[x][0] != tname for x in group):
            raise GateError("synthetic group %s mixes kinds" % group)
        nb_total = 1
        for d in dims:
            nb_total *= d
        nb_total //= be
        grp0 = len(plan.units)
        us = []
        for pos, label in enumerate(group):
            us.append(plan.new_unit(
                kind=KIND_Q4K if tname == "Q4_K" else KIND_Q8_0, tname=tname,
                tensor_index=-(1 + gi), tensor="synth:%s" % label, family="synthetic",
                cls=CLS_SYNTH, block_index=pos, n_elts=be, block_bytes=bb, nb_total=nb_total,
                nb_per_row=dims[0] // be, n_rows=dims[1] if len(dims) >= 2 else 1,
                n_outer=dims[2] if len(dims) >= 3 else 1, xpose_ok=0,
                raw=syn[label][2], src_off=-1, dst_off=-1))
        grp1 = len(plan.units) - 1
        for u in us:
            u["grp_u0"], u["grp_u1"] = grp0, grp1
        for i, u in enumerate(us):
            guard = us[i + 1]["raw"][0] if i + 1 < len(us) else 0
            u["src_off"] = plan.add_slot(bb, u["raw"], guard)
        n_elts_total = 1
        for d in dims:
            n_elts_total *= d
        plan.sample_rows.append({"family": "synthetic", "tensor": "synth:" + "+".join(group),
                                 "type": tname, "index": -(1 + gi), "dims_gguf": list(dims),
                                 "dims_hf": list(reversed(dims)), "n_elts": n_elts_total,
                                 "nb_total": nb_total,
                                 "blocks_sampled": list(range(len(group))),
                                 "n_sampled": len(group), "support_blocks": [],
                                 "xpose_eligible": 0})
    # ---------- cases, then the output regions they write into ----------
    # Ordering matters: the regions are laid out over plan.cases, so they must be assigned AFTER the
    # cases exist. Assigning them first silently leaves every injection writing over mode 0.
    for u in plan.units:
        emit_cases(plan, u)
    total = 0
    for u in plan.units:
        u["dst_off"] = total
        total += u["n_elts"]
    for c in plan.cases:
        if c["mode"] == 0:
            c["dst_off"] = plan.units[c["unit"]]["dst_off"]
            c["cmp_off"] = -1
        else:
            c["dst_off"] = total
            total += c["n_elts"]
            c["cmp_off"] = plan.units[c["unit"]]["dst_off"]
    plan.out_floats = total
    plan.launches = len(set(c["mode"] for c in plan.cases))
    # ---------- probe rows: the decoders alone, host and device both judged by them ----------
    for label, tname, dims, raw, why in synthetic_blocks():
        plan.probe.append((label, tname, "correct", None, raw))
        if tname == "Q4_K":
            plan.probe.append((label, tname, "q4k_as_q4nl", "q4k_as_q4nl", raw))
            plan.probe.append((label, tname, "q4k_drop_min", "q4k_drop_min", raw))
            for m in ("q4k_naive_sc_min", "q4k_subblock_rot3"):
                if not immune_scale_mode(raw, m):
                    plan.probe.append((label, tname, m, None, raw))
    return plan


# -----------------------------------------------------------------------------------------------
# writing the plan out (by file, never by argv: pitfalls/50 §"One argv slot caps at 131,072 bytes")
# -----------------------------------------------------------------------------------------------
UNIT_HEADER = ("# U\tkind\tsrc_off\tslot_bytes\tdst_off\tn_elts\tn_blocks\tblock_bytes\tnb_total"
               "\tnb_per_row\tn_rows\tn_outer\tgrp_u0\tgrp_u1\txpose_ok\tblock_index\ttensor_index"
               "\tcls")
CASE_HEADER = "# C\tcase_id\tunit\tmode\tdst_off\tcmp_off"


def write_plan_files(plan, work):
    if not os.path.isdir(work):
        os.makedirs(work)
    up = os.path.join(work, "units.tsv")
    cp = os.path.join(work, "cases.tsv")
    sp = os.path.join(work, "slab.raw")
    with open(up, "w") as fh:
        fh.write(UNIT_HEADER + "\n")
        for u in plan.units:
            fh.write("U\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n"
                     % (u["kind"], u["src_off"], u["block_bytes"] + 1, u["dst_off"], u["n_elts"],
                        u["n_blocks"], u["block_bytes"], u["nb_total"], u["nb_per_row"],
                        u["n_rows"], u["n_outer"], u["grp_u0"], u["grp_u1"], u["xpose_ok"],
                        u["block_index"], u["tensor_index"], u["cls"]))
    with open(cp, "w") as fh:
        fh.write(CASE_HEADER + "\n")
        for c in plan.cases:
            fh.write("C\t%d\t%d\t%d\t%d\t%d\n"
                     % (c["case_id"], c["unit"], c["mode"], c["dst_off"], c["cmp_off"]))
    with open(sp, "wb") as fh:
        fh.write(bytes(plan.slab))
    return up, cp, sp


def write_probe_files(plan, work):
    """Two files. `probe.tsv` is the honest one: every row's expected words are the reader's own
    output, and MISMATCH rows are the ones the format says must differ. `probe_bad.tsv` is the
    control on the control — one MATCH row carries a perturbed word, and host_probe.cpp must fail on
    it. A comparator that has never rejected anything is not evidence (pitfalls/40)."""
    gp = os.path.join(work, "probe.tsv")
    bp = os.path.join(work, "probe_bad.tsv")
    rows = []
    for i, (label, tname, mname, inj, raw) in enumerate(plan.probe):
        kind = KIND_Q4K if tname == "Q4_K" else KIND_Q8_0
        if inj is None and MODE_CLASS[mname] == "arith":
            expect = EXP_MISMATCH
            vals = reader_decode(raw, tname)
        elif inj is not None:
            expect = EXP_MATCH
            vals = reader_decode(raw, tname, inj=inj)
        else:
            expect = EXP_MATCH
            vals = reader_decode(raw, tname)
        rows.append((i, kind, MODE_ID[mname], expect, BLOCK_ELTS[tname], raw.hex(),
                     floats_to_hex(vals), label, mname))
    with open(gp, "w") as fh:
        fh.write("# P\tcase_id\tkind\tmode\texpect\tn_elts\thexbytes\thexbits\n")
        for r in rows:
            fh.write("P\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n" % r[:7])
    bad = list(rows)
    victims = [i for i, r in enumerate(bad) if r[3] == EXP_MATCH and r[2] == 0]
    if not victims:
        raise GateError("no MATCH probe row to corrupt: the corrupted-probe witness is impossible")
    i = victims[0]
    _id, kind, mode, _e, n_elts, hx, hbits, label, mname = bad[i]
    words = hbits.split(",")
    words[0] = "%08x" % (int(words[0], 16) ^ 1)          # flip the low bit of the first value
    bad[i] = (_id, kind, mode, EXP_MATCH, n_elts, hx, ",".join(words), label, mname)
    with open(bp, "w") as fh:
        fh.write("# P\tcase_id\tkind\tmode\texpect\tn_elts\thexbytes\thexbits (corrupted control)\n")
        for r in bad:
            fh.write("P\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n" % r[:7])
    return gp, bp, len(rows), label


# -----------------------------------------------------------------------------------------------
# the driver, and the parsers that must not be able to read an empty run as a pass
# -----------------------------------------------------------------------------------------------
def run_bin(cmd, log_path=None):
    """subprocess.run with stdout written to a FILE, never through a pipe: a filter that can hide a
    section cannot gate on it (pitfalls/50). The return code is the binary's own."""
    if log_path:
        with open(log_path, "w") as fh:
            rc = subprocess.call(list(cmd), stdout=fh, stderr=subprocess.STDOUT)
        with open(log_path) as fh:
            text = fh.read()
    else:
        sys.stderr.write("run_bin without a log path is not allowed for a verdict\n")
        return 2, ""
    return rc, text


def parse_kv(line):
    # value pattern is [^\s]+, not a number: the driver prints kind=Q4_K, and a numeric pattern
    # silently drops word-valued fields (pitfalls/40 §"A driver whose parser matches nothing").
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))


KIND_FROM_WORD = {"Q4_K": KIND_Q4K, "Q8_0": KIND_Q8_0}


def parse_deqcases(text):
    printed = [l for l in text.splitlines() if l.startswith("DEQCASE ")]
    recs = {}
    for l in printed:
        kv = parse_kv(l)
        for k in ("mode", "kind", "cls", "units", "out_sum"):
            if k not in kv:
                raise GateError("DEQCASE line is missing %s: %r" % (k, l))
        try:
            mode, cls, units = int(kv["mode"]), int(kv["cls"]), int(kv["units"])
            out_sum = float(kv["out_sum"])
        except ValueError:
            raise GateError("DEQCASE line carries a non-numeric field: %r" % l)
        if kv["kind"] not in KIND_FROM_WORD:
            raise GateError("DEQCASE kind=%r is not a kind I know: %r" % (kv["kind"], l))
        if units <= 0:
            raise GateError("DEQCASE cell printed 0 units, which cannot be a cell: %r" % l)
        key = (mode, KIND_FROM_WORD[kv["kind"]], cls)
        if key in recs:
            raise GateError("driver printed cell %s twice" % (key,))
        recs[key] = units, out_sum
    if len(recs) != len(printed):
        raise GateError("parsed %d DEQCASE records from %d printed lines"
                        % (len(recs), len(printed)))
    return recs, len(printed)


def parse_report(path, want_rows):
    rows = {}
    printed = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9:
                raise GateError("report row %d has %d fields, expected 9: %r" % (printed, len(f), line))
            try:
                cid = int(f[0])
                rec = dict(unit=int(f[1]), mode=int(f[2]), kind=int(f[3]), cls=int(f[4]),
                           n_elts=int(f[5]), out_sum=float(f[6]), bad=int(f[7]), nan=int(f[8]))
            except ValueError:
                raise GateError("report row %d has a non-numeric field: %r" % (printed, line))
            if cid in rows:
                raise GateError("report row %d printed twice" % cid)
            rows[cid] = rec
            printed += 1
    if printed != want_rows:
        raise GateError("report printed %d rows, the plan has %d cases" % (printed, want_rows))
    if printed == 0:
        raise GateError("the report has zero rows: a section with no cases cannot pass")
    return rows, printed


def parse_single(text, tag, keys):
    hits = [l for l in text.splitlines() if l.startswith(tag + " ")]
    if len(hits) != 1:
        raise GateError("expected exactly one %r line, found %d" % (tag, len(hits)))
    kv = parse_kv(hits[0])
    for k in keys:
        if k not in kv:
            raise GateError("%s line is missing %s: %r" % (tag, k, hits[0]))
    return kv, hits[0]


def plan_cells(plan):
    cells = {}
    for c in plan.cases:
        k = (c["mode"], c["kind"], c["cls"])
        cells[k] = cells.get(k, 0) + 1
    return cells


def plan_leaks(plan):
    """Mode discipline, checked on the PLAN before the device is touched: a Q4_K-only mode must never
    be applied to a Q8_0 unit and vice versa. The device has a backstop (it writes NaN, counted as
    cnan), but a plan that asks for the wrong pairing is a plan bug and must be red here."""
    bad = []
    for c in plan.cases:
        kind_name = "Q4_K" if c["kind"] == KIND_Q4K else "Q8_0"
        mk = MODE_KIND[c["mname"]]
        if mk is not None and mk != kind_name:
            bad.append((c["case_id"], c["mname"], kind_name))
    return bad


# -----------------------------------------------------------------------------------------------
# the comparison: device bits against the reader's bits, zero tolerance, per expectation class
# -----------------------------------------------------------------------------------------------
def read_out(path, n_floats):
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) != 4 * n_floats:
        raise GateError("device output is %d bytes, expected %d" % (len(blob), 4 * n_floats))
    return struct.unpack("<%dI" % n_floats, blob), struct.unpack("<%df" % n_floats, blob)


def is_nan_bits(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def word_eq(a, b):
    """Python mirror of `q4kd_word_eq` in q4k_dequant_kernels.h — the same rule the device counters and
    the host probe use, so a row cannot be clean on one arm and dirty on another.

    Bit-exact on finite values including the sign of a zero; two NaNs are equal whatever their sign or
    payload, because neither route defines those bits for a NaN product (the reader multiplies in
    Python doubles, CPython folds `nan * -1.0` into `-nan`, an f32 multiply keeps the first operand).
    """
    if a == b:
        return True
    return is_nan_bits(a) and is_nan_bits(b)


def nan_word_positions(words):
    """Indices of the NaN words in an expected-value list — the ones the sign-flip control touches."""
    return [i for i, w in enumerate(words) if is_nan_bits(w)]


class Verdict:
    """Every check the gate runs appends here, and the run is only allowed to end with exactly
    EXPECTED_CHECKS of them, each non-vacuous. A check that never ran is a red result, not a shorter
    list (pitfalls/40 §"A requirement written after the object it constrains is a note")."""

    def __init__(self):
        self.checks = []
        self.findings = []

    def check(self, name, ok, n, detail=""):
        if n <= 0:
            ok = False
            detail = (detail + " | " if detail else "") + "n=%d is vacuous" % n
        self.checks.append((name, bool(ok), int(n), detail))
        print("CHECK name=%s ok=%d n=%d%s" % (name, 1 if ok else 0, n,
                                              (" detail=%s" % detail) if detail else ""))
        if not ok:
            self.findings.append(name)
        return bool(ok)

    def report(self):
        print("CHECKS ran=%d failed=%d%s" % (len(self.checks), len(self.findings),
                                             (" failed_list=" + ",".join(self.findings))
                                             if self.findings else ""))


def compare(plan, ibits, fvals, rows, v):
    """Returns per-case detail and fills the verdict. The comparison is on 32-bit patterns, so
    -0.0 vs +0.0 is a mismatch and must be: the reader produces the same signed zeros by the same
    expression, and a kernel that flushed them would be wrong."""
    tally = {EXP_MATCH: [0, 0], EXP_MISMATCH: [0, 0], EXP_ALT: [0, 0], EXP_IMMUNE: [0, 0]}
    fired_by_mode = {}
    bad_detail = []
    cross_bad = 0
    cross_sum_bad = 0
    nan_cases = 0
    unexpected_nan = 0
    control_elements = 0
    total_elements = 0
    cell_sums = {}
    for c in plan.cases:
        n, off = c["n_elts"], c["dst_off"]
        dev = ibits[off:off + n]
        ref = c["ref_bits"]
        nbad = 0
        for i in range(n):
            if not word_eq(dev[i], ref[i]):
                nbad += 1
        alt = c["alt_bits"]
        nbad_alt = None if alt is None else sum(1 for i in range(n) if not word_eq(dev[i], alt[i]))
        anynan = 1 if any(is_nan_bits(b) for b in dev) else 0
        nan_cases += anynan
        # NaN is only *unexpected* on a row that has to reproduce the reader's words: the control rows
        # (mode 0) and the structurally-immune rows. On a MISMATCH row a NaN is just one way of
        # differing — an injected misreading can walk a scale field into an f16 NaN pattern — and
        # expect-MISMATCH already requires that row to differ, so scanning it here would only let the
        # check fire on the thing it is not about.
        if c["expect"] in (EXP_MATCH, EXP_IMMUNE):
            control_elements += n
            for i in range(n):
                if is_nan_bits(dev[i]) and not is_nan_bits(ref[i]):
                    unexpected_nan += 1
                    if len(bad_detail) < 8:
                        bad_detail.append("case %d (%s on %s blk %d): element %d is NaN on the device "
                                          "but finite in the reader's decode"
                                          % (c["case_id"], c["mname"], c["tensor"], c["block_index"], i))
        total_elements += n
        m0 = plan.units[c["unit"]]["dst_off"]
        mine_vs_mode0 = 0 if c["mode"] == 0 else sum(
            1 for i in range(n) if not word_eq(dev[i], ibits[m0 + i]))
        r = rows[c["case_id"]]
        if r["bad"] != mine_vs_mode0:
            cross_bad += 1
            if len(bad_detail) < 8:
                bad_detail.append("case %d (%s on %s blk %d): driver counted %d bad vs mode 0, "
                                  "I count %d" % (c["case_id"], c["mname"], c["tensor"],
                                                  c["block_index"], r["bad"], mine_vs_mode0))
        if r["nan"] != anynan:
            cross_bad += 1
            if len(bad_detail) < 8:
                bad_detail.append("case %d: driver nan=%d, I see nan=%d" % (c["case_id"], r["nan"],
                                                                            anynan))
        cell_sums.setdefault((c["mode"], c["kind"], c["cls"]), 0.0)
        cell_sums[(c["mode"], c["kind"], c["cls"])] += sum(fvals[off:off + n])
        t = tally[c["expect"]]
        t[0] += 1
        ok = True
        if c["expect"] in (EXP_MATCH, EXP_IMMUNE):
            if nbad != 0:
                ok = False
                bad_detail.append("case %d (%s on %s blk %d, %s): %d/%d elements differ from the "
                                  "reader's decode of the SAME bytes"
                                  % (c["case_id"], c["mname"], c["tensor"], c["block_index"],
                                     EXP_NAME[c["expect"]], nbad, n))
        elif c["expect"] == EXP_MISMATCH:
            if nbad == 0:
                ok = False
                bad_detail.append("case %d (%s on %s blk %d): MISMATCH did not fire — the injected "
                                  "misreading produced the correct answer"
                                  % (c["case_id"], c["mname"], c["tensor"], c["block_index"]))
            else:
                fired_by_mode[c["mname"]] = fired_by_mode.get(c["mname"], 0) + 1
        else:  # EXP_ALT: must equal the ALTERNATE reference and differ from the primary
            if nbad_alt != 0:
                ok = False
                bad_detail.append("case %d (%s on %s blk %d): ALT case does not equal the reader's "
                                  "decode of the bytes this mode reads (%d/%d differ) — the kernel "
                                  "made a different error than the one being injected"
                                  % (c["case_id"], c["mname"], c["tensor"], c["block_index"],
                                     nbad_alt, n))
            elif nbad == 0:
                ok = False
                bad_detail.append("case %d (%s): ALT case equals the primary reference too, so it "
                                  "proves nothing" % (c["case_id"], c["mname"]))
            else:
                fired_by_mode[c["mname"]] = fired_by_mode.get(c["mname"], 0) + 1
        if ok:
            t[1] += 1
    for exp in (EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE):
        total, ok = tally[exp]
        v.check("expect-%s" % EXP_NAME[exp], ok == total, total,
                "ok=%d/%d" % (ok, total))
    v.check("driver-bad-column-agrees", cross_bad == 0, len(plan.cases),
            "mismatches=%d%s" % (cross_bad, ("; " + " | ".join(bad_detail)) if bad_detail else ""))
    v.check("no-unexpected-nan", unexpected_nan == 0 and control_elements > 0, control_elements,
            "control/immune elements scanned=%d, device NaN where the reader is finite=%d; cases "
            "carrying NaN anywhere=%d of %d (the reader produces NaN for some real bytes, so that "
            "number is a fact about the file, not a verdict)"
            % (control_elements, unexpected_nan, nan_cases, len(plan.cases)))
    v.check("expect-classes-populated",
            all(tally[e][0] > 0 for e in (EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE)),
            sum(tally[e][0] for e in tally),
            " ".join("%s=%d" % (EXP_NAME[e], tally[e][0]) for e in
                     (EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE)))
    used = set(c["mname"] for c in plan.cases if c["mode"] != 0)
    silent = sorted(m for m in used if fired_by_mode.get(m, 0) == 0)
    v.check("every-injection-mode-fires", not silent, len(used),
            "modes_used=%d silent=%s" % (len(used), silent or "none"))
    return tally, cell_sums, fired_by_mode


def witness_flip(plan, ibits, fvals):
    """Non-vacuity for the comparator itself: flip one bit of one element and require the comparison
    to see exactly one bad element. If this returns False the whole gate is decoration."""
    m0 = [c for c in plan.cases if c["mode"] == 0 and c["n_elts"] > 0]
    if not m0:
        return False, 0
    c = m0[0]
    off, n = c["dst_off"], c["n_elts"]
    probe = list(ibits[off:off + n])
    want = c["ref_bits"]
    before = sum(1 for i in range(n) if probe[i] != want[i])
    probe[0] ^= 1
    after = sum(1 for i in range(n) if probe[i] != want[i])
    return (after == before + 1), after


def witness_decoder(probe_bad_bin_log):
    """host_probe must REJECT the corrupted fixture. Returns True when it did."""
    return "DEVIATE" in probe_bad_bin_log and "PROBE RESULT FAIL" in probe_bad_bin_log


def witness_parser():
    """Each parser must produce a record when fed one, and must refuse when fed nothing or garbage.
    Returns (ok, n_cases_exercised)."""
    good = ("DEQCASE mode=0 kind=Q4_K cls=0 units=7 out_sum=1.5\n"
            "DEQCASE mode=5 kind=Q8_0 cls=1 units=2 out_sum=-0.25\n")
    recs, printed = parse_deqcases(good)
    if len(recs) != 2 or printed != 2 or recs[(5, KIND_Q8_0, 1)][0] != 2:
        return False, 0
    try:
        parse_deqcases("DEQCASE mode=0 kind=Q4_K cls=0 units=7\n")   # out_sum missing
        return False, 0
    except GateError:
        pass
    try:
        parse_deqcases("DEQCASE mode=x kind=Q4_K cls=0 units=7 out_sum=1\n")
        return False, 0
    except GateError:
        pass
    try:
        parse_deqcases("DEQCASE mode=0 kind=Q4x cls=0 units=7 out_sum=1\n")
        return False, 0
    except GateError:
        pass
    try:
        parse_deqcases("DEQCASE mode=0 kind=Q4_K cls=0 units=0 out_sum=1\n")
        return False, 0
    except GateError:
        pass
    if parse_deqcases("nothing at all\n")[1] != 0:
        return False, 0
    try:
        parse_single("RESULT PASS device\n", "DEQCELLS", ["cells"])
        return False, 0
    except GateError:
        pass
    return True, 2


# -----------------------------------------------------------------------------------------------
# the checks the two entry points are required to run, by name. A check that did not run is a red
# result, not a shorter list.
# -----------------------------------------------------------------------------------------------
DEVICE_CHECKS = [
    "geometry-keys", "plan-mode-kind-pairing", "driver-plan-parse", "driver-planchannel",
    "driver-result", "driver-out-file", "deqcells-cells", "deqcells-out_sum",
    "host-probe-clean", "host-probe-corrupt-rejects", "host-probe-f16-exhaustive",
    "host-probe-f16-corrupt-rejects", "comparator-nan-rule", "comparator-rules-witnessed",
    "driver-selftest",
    "expect-MATCH", "expect-IMMUNE", "expect-MISMATCH", "expect-ALT",
    "driver-bad-column-agrees", "no-unexpected-nan", "expect-classes-populated",
    "every-injection-mode-fires", "census-families-covered", "named-tensors-sampled",
    "sampled-blocks-in-range", "real-and-synthetic-units", "cell-kind-purity",
    "witness-comparator-flip", "witness-parsers",
]
SELFTEST_CHECKS = [
    "header-enum", "driver-selftest", "host-probe-clean", "host-probe-corrupt-rejects",
    "host-probe-f16-exhaustive", "host-probe-f16-corrupt-rejects", "comparator-nan-rule",
    "comparator-rules-witnessed",
    "witness-parsers", "empty-cell-parse-is-refused", "census-derived",
    "pack-scale-min-roundtrip", "fixture-immunity-classes", "fixture-signed-zero-discriminates",
    "transpose-map-is-a-permutation",
]


def witness_comparator_rules(v):
    """The equality rule itself, on the Python side, in the four directions that matter. Each is a
    claim about the comparator, not about the data, and the C++ side of the same rule is witnessed by
    the comparator-nan-rule differential above and the corrupted-probe control."""
    cases = [
        ("+0.0 vs -0.0 are different words", word_eq(0x00000000, 0x80000000), False),
        ("two NaNs of different sign are the same answer", word_eq(0x7FC00000, 0xFFC00000), True),
        ("a NaN is not a finite number", word_eq(0x7FC00000, 0x3FC00000), False),
        ("one bit off in a finite value is caught", word_eq(0x3F800000, 0x3F800001), False),
        ("an infinity is not its own negative", word_eq(0x7F800000, 0xFF800000), False),
    ]
    bad = [what for what, got, want in cases if bool(got) != want]
    v.check("comparator-rules-witnessed", not bad, len(cases),
            "5 claims about the comparator; %s" % ("all hold" if not bad else "FAILED: %s" % bad))


def run_driver_selftest(v, work):
    rc, text = run_bin([BIN, "--selftest"], os.path.join(work, "driver-selftest.log"))
    lines = [l for l in text.splitlines() if l.startswith("SELFTEST ")]
    parsed = 0
    ok_all = True
    for l in lines:
        kv = parse_kv(l)
        for k in ("name", "ok", "n"):
            if k not in kv:
                raise GateError("driver SELFTEST line lost %s: %r" % (k, l))
        parsed += 1
        if int(kv["ok"]) != 1 or int(kv["n"]) <= 0:
            ok_all = False
    res = [l for l in text.splitlines() if l.startswith("DRIVER SELFTEST RESULT")]
    if len(res) != 1:
        raise GateError("driver selftest printed %d RESULT lines, expected 1\n%s"
                        % (len(res), text[-2000:]))
    kv = parse_kv(res[0])
    checks_printed = int(kv.get("checks", "-1"))
    failures = int(kv.get("failures", "-1"))
    v.check("driver-selftest", rc == 0 and ok_all and parsed >= 6 and checks_printed == parsed
            and failures == 0, parsed,
            "lines=%d checks_printed=%d failures=%d rc=%d" % (parsed, checks_printed, failures, rc))
    return rc


def write_f16_table(work):
    """All 65536 half-precision patterns with the CPU reader's own f16->f32 answer, and a corrupted
    copy. The expectation side is `gr.f16` (struct.unpack('<e')), never a formula from the decoder
    under test, so this arm can disagree with it. The corruption is the defect that actually happened
    here on the first device build: a bit-rebias that keeps an sNaN payload where the oracle collapses
    to the canonical quiet NaN, which moves a whole 32-element sub-block of the as_q4nl control row.
    -> (clean_path, corrupt_path, rows_written, coverage)"""
    clean = os.path.join(work, "f16.tsv")
    bad = os.path.join(work, "f16_bad.tsv")
    rows = []
    cov = {"nan": 0, "denormal": 0, "inf": 0, "zero": 0}
    victim = -1
    for h in range(1 << 16):
        v = gr.f16(struct.pack("<H", h))
        b = struct.unpack("<I", struct.pack("<f", v))[0]
        rows.append((h, b))
        if ((b >> 23) & 0xFF) == 0xFF and (b & 0x007FFFFF):
            cov["nan"] += 1
            if victim < 0 and (h & 0x3FF):
                victim = h
        elif ((b >> 23) & 0xFF) == 0xFF:
            cov["inf"] += 1
        if (h & 0x7C00) == 0 and (h & 0x03FF):
            cov["denormal"] += 1
        if (h & 0x7FFF) == 0:
            cov["zero"] += 1
    if victim < 0:
        raise GateError("no payload-carrying NaN in the f16 table: the corrupt control is impossible")

    def emit(path, perturb):
        with open(path, "w") as fh:
            fh.write("# F\thex4\thex8   f16 pattern -> the reader's f32 bits\n")
            for h, b in rows:
                if perturb and h == victim:
                    b = ((b & 0x80000000) | 0x7F800000 | ((h & 0x3FF) << 13))
                fh.write("F\t%04x\t%08x\n" % (h, b))

    emit(clean, False)
    emit(bad, True)
    return clean, bad, len(rows), cov, victim


def write_nanrule_probe(work, clean_probe):
    """The control on the one exception q4kd_word_eq grants: every row of the clean probe file, with the
    sign bit of every expected NaN word flipped. The rows, their modes and their expectation classes
    are otherwise untouched, so the run must reproduce the clean run's counts exactly. Flip a sign the
    rule does not grant (a finite word) and the same file goes red — that direction is witnessed by
    probe_bad.tsv.

    -> (path, nan_words_flipped, rows_touched)"""
    out_path = os.path.join(work, "probe_nanrule.tsv")
    flipped = 0
    touched = 0
    with open(clean_probe) as fh, open(out_path, "w") as oh:
        oh.write("# P\tcase_id\tkind\tmode\texpect\tn_elts\thexbytes\thexbits (NaN signs flipped)\n")
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 8:
                raise GateError("clean probe row has %d fields, expected 8" % len(f))
            words = f[7].split(",")
            ib = [int(w, 16) for w in words]
            pos = nan_word_positions(ib)
            if pos:
                touched += 1
                flipped += len(pos)
                for i in pos:
                    ib[i] ^= 0x80000000
                f[7] = ",".join("%08x" % w for w in ib)
            oh.write("\t".join(f) + "\n")
    return out_path, flipped, touched


def run_host_probes(plan, work, v):
    gp, bp, n_rows, victim = write_probe_files(plan, work)
    ft, ft_bad, f_rows_written, f_cov, f_victim = write_f16_table(work)
    rc1, t1 = run_bin([PROBE_BIN, gp], os.path.join(work, "probe-clean.log"))
    lines1 = [l for l in t1.splitlines() if l.startswith("PROBE id=")]
    res1 = [l for l in t1.splitlines() if l.startswith("PROBE RESULT rows=")]
    if len(res1) != 1:
        raise GateError("host_probe printed %d RESULT rows= lines on the clean fixture:\n%s"
                        % (len(res1), t1[-2000:]))
    kv = parse_kv(res1[0])
    for k in ("rows", "bad_rows", "unexpected", "match_rows", "mismatch_fired"):
        if k not in kv:
            raise GateError("host_probe RESULT lost %s: %r" % (k, res1[0]))
    # counted from the row records themselves, not from a prefix that also matches the two RESULT
    # lines; an off-by-one here is the empty-parse shape (pitfalls/40 §"A parser that can print PASS
    # on an empty parse").
    v.check("host-probe-clean",
            rc1 == 0 and int(kv["rows"]) == n_rows and int(kv["bad_rows"]) == 0
            and int(kv["unexpected"]) == 0 and int(kv["rows"]) == len(lines1)
            and int(kv["match_rows"]) > 0 and int(kv["mismatch_fired"]) > 0,
            int(kv["rows"]),
            "rows=%s match=%s fired=%s rc=%d (row records in the log: %d, probe file rows: %d)"
            % (kv["rows"], kv["match_rows"], kv["mismatch_fired"], rc1, len(lines1), n_rows))
    rc2, t2 = run_bin([PROBE_BIN, bp], os.path.join(work, "probe-corrupt.log"))
    v.check("host-probe-corrupt-rejects", rc2 != 0 and witness_decoder(t2), 1,
            "corrupted row is %s; probe rc=%d and printed DEVIATE=%s"
            % (victim, rc2, "DEVIATE" in t2))

    # the exhaustive f16 arm, and its own control on the control
    rc3, t3 = run_bin([PROBE_BIN, gp, ft], os.path.join(work, "probe-f16.log"))
    fline = [l for l in t3.splitlines() if l.startswith("F16 ")]
    if len(fline) != 1:
        raise GateError("host_probe printed %d F16 lines with a table:\n%s" % (len(fline), t3[-2000:]))
    fk = parse_kv(fline[0])
    for k in ("given", "rows", "bad"):
        if k not in fk:
            raise GateError("host_probe F16 line lost %s: %r" % (k, fline[0]))
    v.check("host-probe-f16-exhaustive",
            rc3 == 0 and int(fk["given"]) == 1 and int(fk["rows"]) == f_rows_written == 1 << 16
            and int(fk["bad"]) == 0 and all(f_cov[k] > 0 for k in f_cov),
            int(fk["rows"]),
            "rows=%s bad=%s coverage=%s (lines written: %d)" % (fk["rows"], fk["bad"], f_cov,
                                                                f_rows_written))
    rc4, t4 = run_bin([PROBE_BIN, gp, ft_bad], os.path.join(work, "probe-f16-corrupt.log"))
    fline4 = [l for l in t4.splitlines() if l.startswith("F16 ")]
    fk4 = parse_kv(fline4[0]) if len(fline4) == 1 else {}
    v.check("host-probe-f16-corrupt-rejects",
            rc4 != 0 and int(fk4.get("bad", 0)) >= 1 and "FAIL f16" in t4, 1,
            "corrupted f16 input is 0x%04x; probe rc=%d printed bad=%s"
            % (f_victim, rc4, fk4.get("bad")))

    np_path, n_flipped, n_touched = write_nanrule_probe(work, gp)
    rc5, t5 = run_bin([PROBE_BIN, np_path], os.path.join(work, "probe-nanrule.log"))
    res5 = [l for l in t5.splitlines() if l.startswith("PROBE RESULT rows=")]
    if len(res5) != 1:
        raise GateError("host_probe printed %d RESULT lines on the NaN-sign file:\n%s"
                        % (len(res5), t5[-2000:]))
    kv5 = parse_kv(res5[0])
    v.check("comparator-nan-rule",
            n_flipped > 0 and n_touched > 0 and rc5 == 0
            and all(kv5.get(k) == kv.get(k) for k in
                    ("rows", "bad_rows", "unexpected", "match_rows", "mismatch_fired")),
            n_touched,
            "%d expected NaN words in %d rows had their sign bit flipped; probe still reports "
            "rows=%s unexpected=%s match=%s fired=%s, identical to the clean run"
            % (n_flipped, n_touched, kv5.get("rows"), kv5.get("unexpected"),
               kv5.get("match_rows"), kv5.get("mismatch_fired")))
    return rc1, rc2


def check_cells(plan, recs, printed, v, cell_sums):
    want = plan_cells(plan)
    same_keys = set(recs) == set(want)
    units_ok = same_keys and all(recs[k][0] == want[k] for k in want)
    v.check("deqcells-cells", same_keys and units_ok and printed == len(want) and len(want) > 0,
            len(want),
            "driver printed %d cells, the plan implies %d; keys_match=%s units_match=%s"
            % (printed, len(want), same_keys, units_ok))
    if same_keys:
        worst = None
        bad = 0
        for k in want:
            a, b = recs[k][1], cell_sums.get(k, 0.0)
            tol = 1e-3 * max(1.0, abs(b))
            if abs(a - b) > tol:
                bad += 1
                if worst is None:
                    worst = (k, a, b)
        # The sum is a copy-and-cell-identity witness (an atomic float accumulator against a double
        # sum), not a numerics band: the values themselves are judged bit-for-bit by compare().
        v.check("deqcells-out_sum", bad == 0, len(want),
                "cells_outside_accumulator_allowance=%d%s"
                % (bad, "" if worst is None else " first=%r" % (worst,)))
        # cell purity: no Q4_K-only mode may have a Q8_0 cell and vice versa
        impure = []
        for (mode, kind, cls) in recs:
            mname = MODES[mode][0]
            mk = MODE_KIND[mname]
            kind_name = "Q4_K" if kind == KIND_Q4K else "Q8_0"
            if mk is not None and mk != kind_name:
                impure.append((mname, kind_name))
        v.check("cell-kind-purity", not impure, len(recs), "leaks=%s" % (impure or "none"))
    else:
        v.check("deqcells-out_sum", False, 0, "cells did not match, so sums cannot be paired")
        v.check("cell-kind-purity", False, 0, "cells did not match")


# -----------------------------------------------------------------------------------------------
# the two entry points
# -----------------------------------------------------------------------------------------------
def fixture_class_checks(v):
    """The synthetic fixtures must have the expectation classes the gate claims for them, computed
    here rather than assumed: the all-zero Q4_K block is immune to the three scale/min misreadings
    and still fires under as_q4nl; the flat block is immune to the rotation and fires under naive;
    the grid block fires under all four."""
    syn = {label: raw for label, tname, dims, raw, why in synthetic_blocks()
           if tname == "Q4_K"}
    want = [("q4k_zero", "q4k_drop_min", True), ("q4k_zero", "q4k_naive_sc_min", True),
            ("q4k_zero", "q4k_subblock_rot3", True), ("q4k_zero", "q4k_as_q4nl", False),
            ("q4k_flat", "q4k_subblock_rot3", True), ("q4k_flat", "q4k_naive_sc_min", False),
            ("q4k_grid", "q4k_subblock_rot3", False), ("q4k_grid", "q4k_naive_sc_min", False),
            ("q4k_grid", "q4k_drop_min", False), ("q4k_grid", "q4k_as_q4nl", False),
            ("q4k_minonly", "q4k_drop_min", False)]
    bad = []
    for label, mode, expect_immune in want:
        got = immune_scale_mode(syn[label], mode)
        if got != expect_immune:
            bad.append("%s/%s: predicate said immune=%s, expected %s"
                       % (label, mode, got, expect_immune))
    v.check("fixture-immunity-classes", not bad, len(want), "; ".join(bad))
    zero = reader_decode(syn["q4k_zero"], "Q4_K")
    allpos = all(f32bits(x) == 0 for x in zero)
    nl = reader_decode(syn["q4k_zero"], "Q4_K", inj="q4k_as_q4nl")
    negzero = any(f32bits(x) == 0x80000000 for x in nl)
    v.check("fixture-signed-zero-discriminates", allpos and negzero, 1,
            "the all-zero block decodes to +0.0 everywhere=%s while as_q4nl yields a -0.0=%s, so "
            "the immunity cases are not just zeros agreeing with zeros" % (allpos, negzero))


def selftest(args, v):
    n_modes = check_header_enum()
    v.check("header-enum", n_modes == len(MODES), n_modes,
            "%d modes, and the kind/expect/mode enums plus block sizes match the header" % n_modes)
    if not os.path.exists(BIN) or not os.path.exists(PROBE_BIN):
        raise GateError("missing binaries: %s / %s — run ./build.sh first" % (BIN, PROBE_BIN))
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    run_driver_selftest(v, WORK)
    plan = Plan()
    for label, tname, dims, raw, why in synthetic_blocks():
        plan.probe.append((label, tname, "correct", None, raw))
        if tname == "Q4_K":
            plan.probe.append((label, tname, "q4k_as_q4nl", "q4k_as_q4nl", raw))
            plan.probe.append((label, tname, "q4k_drop_min", "q4k_drop_min", raw))
            for m in ("q4k_naive_sc_min", "q4k_subblock_rot3"):
                if not immune_scale_mode(raw, m):
                    plan.probe.append((label, tname, m, None, raw))
    for u in synthetic_units(plan):
        emit_cases(plan, u)
    total = 0
    for u in plan.units:
        u["dst_off"] = total
        total += u["n_elts"]
    for c in plan.cases:
        if c["mode"] == 0:
            c["dst_off"] = plan.units[c["unit"]]["dst_off"]
            c["cmp_off"] = -1
        else:
            c["dst_off"] = total
            total += c["n_elts"]
            c["cmp_off"] = plan.units[c["unit"]]["dst_off"]
    run_host_probes(plan, WORK, v)
    witness_comparator_rules(v)
    ok, n = witness_parser()
    v.check("witness-parsers", ok, n, "the parsers must yield records, refuse garbage and refuse "
                                      "an empty cell table")
    empty_v = Verdict()
    check_cells(Plan(), {}, 0, empty_v, {})
    cell_names = ("deqcells-cells", "deqcells-out_sum", "cell-kind-purity")
    v.check("empty-cell-parse-is-refused",
            all(nm in empty_v.findings for nm in cell_names)
            and len(empty_v.checks) == len(cell_names), len(cell_names),
            "an empty cell table must turn every cell-derived check red, not print an empty PASS "
            "(failed: %s of %s)" % (sorted(set(empty_v.findings) & set(cell_names)),
                                    len(cell_names)))
    geo = derived_census()
    v.check("census-derived", len(geo["spec"]) > 0 and geo["E"] == 256 and geo["I"] == 512
            and geo["H"] == 2048 and geo["L"] == 40 and geo["vocab"] == 248320
            and len(geo["full"]) + len(geo["lin"]) == geo["L"], len(geo["spec"]),
            "E=%d I=%d H=%d L=%d vocab=%d full=%d linear=%d"
            % (geo["E"], geo["I"], geo["H"], geo["L"], geo["vocab"], len(geo["full"]),
               len(geo["lin"])))
    rt = True
    for sc, mg in ((SC_DISTINCT, MG_DISTINCT), (SC_FLAT, MG_FLAT), ([0] * 8, [0] * 8),
                   ([63] * 8, [0] * 8), ([0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0])):
        if gr.qk_scale_min(pack_scale_min(sc, mg)) != (list(sc), list(mg)):
            rt = False
    v.check("pack-scale-min-roundtrip", rt, 5,
            "the synthetic packer must round-trip through the reader's get_scale_min_k4")
    fixture_class_checks(v)
    H, I, E = 2048, 512, 256
    npr, nr, no = H // 256, I, E
    # The expert-transpose misread is a permutation of the tensor's blocks, and the two claims that
    # make it a usable injection are (a) it is a bijection — read through the swapped extents
    # (n_outer, n_rows, nb_per_row) it undoes itself, so an ALT expectation is well defined — and
    # (b) it actually moves blocks while leaving at least one in place, so the same mode has a
    # must-catch case and a must-not-catch case. "Own inverse under the SAME extents" was the wrong
    # claim and went red: nb_per_row (8) and n_outer (256) are not equal, so the map is not an
    # involution of the index space onto itself, it is an involution between two layouts.
    dom = npr * nr * no
    moved, fixed = 0, 0
    seen = set()
    for b in range(dom):
        bp = xpose_index(b, npr, nr, no)
        seen.add(bp)
        if bp != b:
            moved += 1
        else:
            fixed += 1
        if xpose_index(bp, no, nr, npr) != b:
            raise GateError("the transpose map is not invertible at block %d of an expert tensor "
                            "(b' -> b'' gave %d)" % (b, xpose_index(bp, no, nr, npr)))
    for e in (0, 1, 7, 63, 128, 255):
        for n in (0, 1, 300, 511):
            for j in (0, 1, 7):
                b = (e * nr + n) * npr + j
                assert xpose_index(xpose_index(b, npr, nr, no), no, nr, npr) == b, b
    v.check("transpose-map-is-a-permutation", len(seen) == dom and moved > 0 and fixed > 0, dom,
            "%d expert blocks, a bijection on itself (%d distinct images), moves %d and fixes %d "
            "(the fixed block is the ALT-immunity case; a map that moved nothing could not be caught)"
            % (dom, len(seen), moved, fixed))
    return finish(v, "selftest", SELFTEST_CHECKS)


def synthetic_units(plan):
    """The synthetic groups, built without touching the GGUF — the same construction build_plan uses,
    so the offline selftest judges the same fixtures."""
    syn = {label: (tname, dims, raw, why) for label, tname, dims, raw, why in synthetic_blocks()}
    units = []
    for gi, group in enumerate(synth_groups()):
        tname = syn[group[0]][0]
        be, bb = gr.BLOCK[tname]
        dims = syn[group[0]][1]
        nb_total = 1
        for d in dims:
            nb_total *= d
        nb_total //= be
        grp0 = len(plan.units)
        for pos, label in enumerate(group):
            units.append(plan.new_unit(
                kind=KIND_Q4K if tname == "Q4_K" else KIND_Q8_0, tname=tname,
                tensor_index=-(1 + gi), tensor="synth:%s" % label, family="synthetic",
                cls=CLS_SYNTH, block_index=pos, n_elts=be, block_bytes=bb, nb_total=nb_total,
                nb_per_row=dims[0] // be, n_rows=dims[1] if len(dims) >= 2 else 1,
                n_outer=dims[2] if len(dims) >= 3 else 1, xpose_ok=0,
                raw=syn[label][2], src_off=len(plan.slab), dst_off=-1))
            plan.slab.extend(syn[label][2])
            nxt = syn[group[pos + 1]][2][0] if pos + 1 < len(group) else 0
            plan.slab.append(nxt)
        for u in plan.units[grp0:]:
            u["grp_u0"], u["grp_u1"] = grp0, len(plan.units) - 1
    return units


CHECK_ONLY_CHECKS = ["geometry-keys", "plan-mode-kind-pairing", "driver-plan-parse"]


def finish(v, label, roster):
    v.report()
    names = [c[0] for c in v.checks]
    missing = sorted(set(roster) - set(names))
    extra = sorted(set(names) - set(roster))
    dupes = sorted(set(n for n in names if names.count(n) > 1))
    if missing or extra or dupes:
        print("FAIL check roster: missing=%s extra=%s duplicated=%s" % (missing, extra, dupes))
        v.findings.append("check-roster")
    ok = not v.findings
    print("RESULT %s %s checks=%d findings=%d"
          % ("PASS" if ok else "FAIL", label, len(v.checks), len(v.findings)))
    return 0 if ok else 1


def device(args, v):
    if not os.path.exists(BIN) or not os.path.exists(PROBE_BIN):
        raise GateError("missing binaries: %s / %s — run ./build.sh first" % (BIN, PROBE_BIN))
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    marker = require_audit_marker()
    print("MARKER file=%s first_pass=%s latest_pass=%s"
          % (os.path.basename(apq.SELFTEST_MARK), marker.get("utc_first_selftest_pass"),
             marker.get("utc_latest_selftest_pass")))
    gg = gr.Gguf(apq.GGUF_PATH)
    print("GGUF file=%s bytes=%d n_tensors=%d header_bytes_read=%d data_start=%d alignment=%d"
          % (os.path.basename(gg.path), gg.size, gg.n_tensors, gg.bytes_read, gg.data_start,
             gg.alignment))
    geo = derived_census()
    print("GEOMETRY " + " ".join("%s=%s" % (k, geo["config_keys"][k])
                                 for k in sorted(geo["config_keys"])))
    good_keys = (geo["E"] == 256 and geo["I"] == 512 and geo["H"] == 2048 and geo["L"] == 40
                 and geo["vocab"] == 248320)
    v.check("geometry-keys", good_keys, len(geo["config_keys"]),
            " ".join("%s=%s" % (k, geo["config_keys"][k]) for k in sorted(geo["config_keys"])))
    global TENSORS_PER_FAMILY
    TENSORS_PER_FAMILY = max(1, args.tensors_per_family)
    plan = build_plan(gg, geo)
    while len(plan.units) > args.max_units and TENSORS_PER_FAMILY > 1:
        old = TENSORS_PER_FAMILY
        TENSORS_PER_FAMILY -= 1
        print("CUT reason=max_units=%d units=%d action=tensors_per_family %d->%d "
              "(the sample is cut, no assertion is)" % (args.max_units, len(plan.units), old,
                                                        TENSORS_PER_FAMILY))
        plan = build_plan(gg, geo)
    if len(plan.units) > args.max_units:
        raise GateError("even one tensor per family gives %d units, above --max-units=%d; lower "
                        "--want-elts instead of dropping assertions"
                        % (len(plan.units), args.max_units))
    leaks = plan_leaks(plan)
    v.check("plan-mode-kind-pairing", not leaks, len(plan.cases),
            "Q4_K-only modes applied to Q8_0 units (or the reverse): %d" % len(leaks))
    up, cp, sp = write_plan_files(plan, WORK)
    by_kind_cls = {}
    for u in plan.units:
        by_kind_cls[(u["kind"], u["cls"])] = by_kind_cls.get((u["kind"], u["cls"]), 0) + 1
    by_expect = {}
    for c in plan.cases:
        by_expect[EXP_NAME[c["expect"]]] = by_expect.get(EXP_NAME[c["expect"]], 0) + 1
    print("PLAN units=%d cases=%d launches=%d out_floats=%d slab_bytes=%d modes=%s"
          % (len(plan.units), len(plan.cases), plan.launches, plan.out_floats, len(plan.slab),
             ",".join(MODES[m][0] for m in sorted(set(c["mode"] for c in plan.cases)))))
    print("PLAN units_by_kind_cls " + " ".join(
        "%s/%s=%d" % ("Q4_K" if k[0] == KIND_Q4K else "Q8_0",
                      {CLS_REAL: "real", CLS_SYNTH: "synth", CLS_SUPPORT: "support"}[k[1]], n)
        for k, n in sorted(by_kind_cls.items())))
    print("PLAN cases_by_expect " + " ".join("%s=%d" % kv for kv in sorted(by_expect.items())))
    print("PLAN tensors_sampled=%d blocks_sampled=%d support_blocks=%d"
          % (len(plan.sample_rows), sum(r["n_sampled"] for r in plan.sample_rows),
             sum(len(r["support_blocks"]) for r in plan.sample_rows)))
    print("PLAN files units=%s cases=%s slab=%s" % (os.path.basename(up), os.path.basename(cp),
                                                    os.path.basename(sp)))
    if args.check_only:
        rc, text = run_bin([BIN, "--plan", up, "--cases", cp, "--slab", sp,
                            "--out", os.path.join(WORK, "out.f32"),
                            "--report", os.path.join(WORK, "report.tsv"),
                            "--expect-units", str(len(plan.units)),
                            "--expect-cases", str(len(plan.cases)), "--check"],
                           os.path.join(WORK, "driver-check.log"))
        print("DRIVER-CHECK rc=%d" % rc)
        for line in text.splitlines():
            print("DRIVER " + line)
        v.check("driver-plan-parse", rc == 0 and "RESULT PASS check" in text, 1,
                "check-only pass: no device was touched")
        return finish(v, "check-only", CHECK_ONLY_CHECKS)
    # ---- host arm first: the same decoders, the host compiler, the same expectations ----------
    run_host_probes(plan, WORK, v)
    run_driver_selftest(v, WORK)
    # ---- device arm ---------------------------------------------------------------------------
    cmd = [BIN, "--plan", up, "--cases", cp, "--slab", sp, "--out", os.path.join(WORK, "out.f32"),
           "--report", os.path.join(WORK, "report.tsv"),
           "--expect-units", str(len(plan.units)), "--expect-cases", str(len(plan.cases))]
    print("CMD " + " ".join(cmd))
    rc, text = run_bin(cmd, os.path.join(WORK, "driver.log"))
    for line in text.splitlines():
        if line.startswith(("FAIL", "PLAN ", "PLANCHECK ", "REPORT ", "DEQCELLS ", "RESULT ")):
            print("DRIVER " + line)
    kv, _ = None, None
    results = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if len(results) != 1:
        raise GateError("the driver printed %d RESULT lines, expected exactly 1" % len(results))
    v.check("driver-result", rc == 0 and results[0].split()[1] == "PASS", len(results),
            "%r rc=%d" % (results[0], rc))
    pkv, _ = parse_single(text, "PLANCHECK", ["units", "cases", "checks_run"])
    v.check("driver-planchannel", int(pkv["checks_run"]) > 0 and int(pkv["units"]) == len(plan.units)
            and int(pkv["cases"]) == len(plan.cases), int(pkv["checks_run"]),
            "structural checks_run=%s units=%s cases=%s"
            % (pkv["checks_run"], pkv["units"], pkv["cases"]))
    ppkv, _ = parse_single(text, "PLAN", ["units_file", "cases_file", "units_parsed",
                                          "cases_parsed"])
    v.check("driver-plan-parse",
            int(ppkv["units_parsed"]) == len(plan.units)
            and int(ppkv["cases_parsed"]) == len(plan.cases)
            and ppkv["units_file"] == up and ppkv["cases_file"] == cp, 2,
            "driver parsed %s/%s from %s + %s, the plan has %d/%d"
            % (ppkv["units_parsed"], ppkv["cases_parsed"], os.path.basename(ppkv["units_file"]),
               os.path.basename(ppkv["cases_file"]), len(plan.units), len(plan.cases)))
    recs, printed = parse_deqcases(text)
    dcv, _ = parse_single(text, "DEQCELLS", ["cells", "cases"])
    if int(dcv["cells"]) != printed or int(dcv["cases"]) != len(plan.cases):
        raise GateError("the driver's own tally says cells=%s cases=%s; I parsed cells=%d cases=%d"
                        % (dcv["cells"], dcv["cases"], printed, len(plan.cases)))
    rows, nrows = parse_report(os.path.join(WORK, "report.tsv"), len(plan.cases))
    okv, _ = parse_single(text, "OUT", ["file", "floats", "bytes"])
    v.check("driver-out-file",
            okv["file"] == os.path.join(WORK, "out.f32")
            and int(okv["floats"]) == plan.out_floats
            and int(okv["bytes"]) == 4 * plan.out_floats, 1,
            "driver dumped %s floats / %s bytes to %s; the plan wants %d floats"
            % (okv["floats"], okv["bytes"], os.path.basename(okv["file"]), plan.out_floats))
    ibits, fvals = read_out(os.path.join(WORK, "out.f32"), plan.out_floats)
    tally, cell_sums, fired = compare(plan, ibits, fvals, rows, v)
    check_cells(plan, recs, printed, v, cell_sums)
    fams = set(r["family"] for r in plan.sample_rows)
    want_fams = set(row[0] for row in geo["spec"]) | set(row[0] for row in geo["whole"])
    v.check("census-families-covered", want_fams <= fams, len(want_fams),
            "families sampled=%d of %d derived; missing=%s"
            % (len(fams & want_fams), len(want_fams), sorted(want_fams - fams) or "none"))
    sampled_names = set(r["tensor"] for r in plan.sample_rows)
    missing_named = [n for n in NAMED_REQUIRED if n not in sampled_names]
    v.check("named-tensors-sampled", not missing_named, len(NAMED_REQUIRED),
            "missing=%s" % (missing_named or "none"))
    inrange = all(0 <= b < r["nb_total"] for r in plan.sample_rows for b in r["blocks_sampled"])
    total_blocks = sum(r["n_sampled"] for r in plan.sample_rows)
    v.check("sampled-blocks-in-range", inrange and total_blocks > 0, total_blocks,
            "every sampled block index is inside its own tensor's block count")
    nreal = sum(1 for u in plan.units if u["cls"] == CLS_REAL)
    nsyn = sum(1 for u in plan.units if u["cls"] == CLS_SYNTH)
    v.check("real-and-synthetic-units", nreal > 0 and nsyn > 0, nreal + nsyn,
            "real=%d synthetic=%d support=%d"
            % (nreal, nsyn, sum(1 for u in plan.units if u["cls"] == CLS_SUPPORT)))
    ok, n = witness_flip(plan, ibits, fvals)
    v.check("witness-comparator-flip", ok, n,
            "flipping one bit of one element must make exactly one element disagree")
    witness_comparator_rules(v)
    ok, n = witness_parser()
    v.check("witness-parsers", ok, n, "parsers must yield records, refuse garbage and refuse empty")
    write_evidence(plan, geo, gg, marker, v, tally, fired, args)
    return finish(v, "device", DEVICE_CHECKS)


def write_evidence(plan, geo, gg, marker, v, tally, fired, args):
    if not os.path.isdir(EV):
        os.makedirs(EV)
    doc = {
        "utc": apq.now(),
        "artifact": os.path.relpath(gg.path, REPO),
        "artifact_bytes": gg.size,
        "tensor_records": gg.n_tensors,
        "reader_sha256": marker.get("reader_sha256"),
        "audit_gate_sha256": marker.get("gate_sha256"),
        "audit_selftest_first_pass": marker.get("utc_first_selftest_pass"),
        "binary_sha256": sha256_file(BIN),
        "host_probe_sha256": sha256_file(PROBE_BIN),
        "gate_sha256": sha256_file(os.path.join(HERE, "gate.py")),
        "kernels_header_sha256": sha256_file(os.path.join(HERE, "q4k_dequant_kernels.h")),
        "hip_sha256": sha256_file(os.path.join(HERE, "q4k_dequant.hip")),
        "geometry": geo["config_keys"],
        "units": len(plan.units),
        "cases": len(plan.cases),
        "launches": plan.launches,
        "out_floats": plan.out_floats,
        "slab_bytes": len(plan.slab),
        "cases_by_expect": {EXP_NAME[k]: sum(1 for c in plan.cases if c["expect"] == k)
                            for k in (EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE)},
        "cases_fired_by_mode": fired,
        "checks": [{"name": n, "ok": o, "n": c, "detail": d} for n, o, c, d in v.checks],
        "findings": v.findings,
        "sample": plan.sample_rows,
        "tensors_per_family": TENSORS_PER_FAMILY,
    }
    with open(EVIDENCE_JSON, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    with open(SAMPLE_MD, "w") as fh:
        fh.write("# Q4_K / Q8_0 device gate — what was sampled (%s)\n\n" % DAY)
        fh.write("Every row below is a tensor-info record of `%s`, selected by rule\n"
                 "(`gate.py §select_samples`) and reconciled against the census derived from\n"
                 "`checkpoints/Qwen3.6-35B-A3B.config.json` (%s). Block indices are the GGUF's own\n"
                 "block numbering for that tensor; `support` blocks exist only so a transposed\n"
                 "expert read has a real block to land on.\n\n"
                 % (os.path.basename(gg.path),
                    ", ".join("%s %s" % (k, geo["config_keys"][k])
                              for k in sorted(geo["config_keys"]))))
        fh.write("| family | tensor | GGUF type | GGUF dims [K,N,E] | HF dims | elts | blocks total"
                 " | blocks sampled | support | xpose |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in plan.sample_rows:
            fh.write("| %s | `%s` | %s | %s | %s | %d | %d | %d (%s) | %d | %d |\n"
                     % (r["family"], r["tensor"], r["type"], ",".join(map(str, r["dims_gguf"])),
                        ",".join(map(str, r["dims_hf"])), r["n_elts"], r["nb_total"],
                        r["n_sampled"],
                        ",".join(str(b) for b in r["blocks_sampled"][:8])
                        + ("…" if r["n_sampled"] > 8 else ""),
                        len(r["support_blocks"]), r["xpose_eligible"]))
        fh.write("\nUnits %d, cases %d, launches %d, slab %d B, f32 written %d.\n"
                 % (len(plan.units), len(plan.cases), plan.launches, len(plan.slab),
                    plan.out_floats))
        fh.write("\nCounts by expectation class: %s\n"
                 % ", ".join("%s=%d" % (EXP_NAME[k], sum(1 for c in plan.cases
                                                         if c["expect"] == k))
                             for k in (EXP_MATCH, EXP_MISMATCH, EXP_ALT, EXP_IMMUNE)))
        fh.write("\nVerdicts: %d checks, findings: %s\n"
                 % (len(v.checks), ", ".join(v.findings) or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="host-only witnesses: driver selftest, host probe, parsers, predicates")
    ap.add_argument("--check-only", action="store_true",
                    help="build the plan and let the driver verify it; no device work")
    ap.add_argument("--max-units", type=int, default=1400,
                    help="cap the sampled unit count by cutting tensors per family, never assertions")
    ap.add_argument("--tensors-per-family", type=int, default=TENSORS_PER_FAMILY,
                    help="how many tensors of each census family to sample (must cover every "
                         "family; the named tensors are always included)")
    a = ap.parse_args()
    try:
        check_header_enum()
        if a.selftest:
            return selftest(a, Verdict())
        return device(a, Verdict())
    except GateError as e:
        print("FAIL gate: %s" % e)
        print("RESULT FAIL gate aborted: %s" % e.__class__.__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
