#!/usr/bin/env python3
"""gguf_reader.py — read a GGUF and dequant its blocks, written from the FORMAT, not from llama.cpp.

Why this file is separate from the gate it serves (`tools/audit_published_quant.py`): the audit's
whole argument is that a check whose reference side is derived from the thing under test cannot
adjudicate it (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from
the thing under test"). So this module is the reader under test, the gate holds the oracle, and the
bf16 checkpoint is the authority. llama.cpp reads nothing here — it is not even built.

Every layout fact below is copied in with its source (rule: `pitfalls/60-docs-and-citations.md`
§"The self-containment rule"). Source of record: the pinned checkout
`engines/llama.cpp` @ d327bf1f0780f87b8c01b367b61853b0c291d7c4 (b11869), files
`ggml/src/ggml-common.h` (block structs + static_asserts), `ggml/src/ggml-quants.c`
(`dequantize_row_q4_K`, `_q5_K`, `_q6_K`, `_q8_0`, `_q4_0`, `get_scale_min_k4`, and the
`quantize_row_*_ref` packers), `ggml/include/ggml.h` (the `ggml_type` enum). Fetched 2026-10-26.

Block geometry, from the struct definitions and their `static_assert`s (QK_K = 256,
K_SCALE_SIZE = 12, QK8_0 = 32):

    type   bytes / block    elems / block    bpw
    F32      4                  1            32
    F16      2                  1            16
    BF16     2                  1            16
    Q8_0    34 (= 2 + 32)        32           8.5
    Q4_K   144 (= 4 + 12 + 128)  256          4.5
    Q5_K   176 (= 4 + 12 + 32 + 128) 256      5.5      <- NOT 5.5625; see quant_scenarios.py note
    Q6_K   210 (= 128 + 64 + 16 + 2) 256      6.5625
    Q4_0    18 (= 2 + 16)        32           4.5

Element order inside a super-block (this is the part a converter gets wrong; each line is a
transcription of the C dequant loop it comes from):

  Q4_K  8 sub-blocks of 32. Sub-block j covers elements [32j, 32j+32). Byte index of element e is
        16 + 32*(e>>6) + (e & 31), nibble = (e>>5) & 1 (so elements 0..31 are the LOW nibbles of
        bytes 0..31 of the qs array, elements 32..63 the HIGH nibbles of the same 32 bytes).
        value(e) = d*sc[j]*q - dmin*m[j], with (sc, m) the 6-bit pairs unpacked by
        `get_scale_min_k4`, d = f16(dm[0]), dmin = f16(dm[1]). NOTE the stored `m` is the
        MAGNITUDE of the sub-block minimum (dequant subtracts it), not a signed min.
  Q5_K  same 8x32 sub-block layout and the same nibble rule for the LOW 4 bits; the 5th bit of
        element e is bit (e>>5) of qh[e & 31]  (qh is 32 bytes, one per nibble position).
        q in [0,32): value(e) = d*sc[j]*q - dmin*m[j].
  Q6_K  16 sub-blocks of 16. Element e (0..255): within the 128-group g = e>>7, position
        p = e & 127, chunk c = p>>5, l = p & 31: low 4 bits from ql[c?]: byte (c & 1)*32 + l of
        the group's 64-byte ql span, nibble (c>>1) — as written by quantize_row_q6_K_ref:
        ql[l] = q1 | (q3 << 4), ql[l+32] = q2 | (q4 << 4), qh[l] = (L[l]>>4) | (L[l+32]>>4)<<2 |
        (L[l+64]>>4)<<4 | (L[l+96]>>4)<<6. q = (six-bit value) - 32 in [-32, 31], and
        value(e) = d * scales[ib] * q with ib = the 16-element sub-block index of e and
        scales int8 (so d carries the sign: quantize_row_q6_K_ref sets d = -max_scale/128).
  Q8_0  32-element blocks: value = f16(d) * int8(qs[j]).
  F32   little-endian float32, element order = array order.
  BF16  the upper 16 bits of a float32 (so bf16 -> f32 is `bits << 16`, exact).

Honesty notes:
  * This reader is NOT a llama.cpp build and is not linked to one. Its selftest
    (`tools/audit_published_quant.py --selftest`) proves it agrees with an independently derived
    packer + expected-value oracle over synthetic blocks; the witness that the FORMAT reading is
    right is a different one — the dequant comparison against the bf16 authority lands at the
    format's own rounding floor, and every injected misreading lands far outside it.
  * Tensor data is only ever read through `Gguf.data_window()`, which refuses until the caller
    hands it a selftest token (`Gguf.SELFTEST_TOKEN_REQUIRED`), so "the gate was written before
    the bytes" is enforced by the tool and not just claimed by it.
"""
import os
import struct
import sys

# ---------------------------------------------------------------------------------------------
# ggml_type ids — copied from ggml/include/ggml.h (b11869). An id not in this table is reported
# as `type_<id>` and never guessed (`pitfalls/40` — a value pattern must cover its value domain).
# ---------------------------------------------------------------------------------------------
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}
# (elems per block, bytes per block) — block_q8_0 / block_q4_K / block_q5_K / block_q6_K /
# block_q4_0 sizes, derived at import from the pinned ggml-common.h structs (tools/ggml_block_geom.py)
# rather than typed in here. This table used to be hand-written from the static_asserts quoted in the
# docstring above; the same class of hand-typed width put Q5_K at 5.5625 in two other tools, and the
# only reason this copy was right was luck. The derivation is now covered by the audit selftest, which
# decodes a synthetic block of every one of these types and checks it against an independent oracle.
BLOCK = None            # filled by _load_block_geom() at import; None means "not derived yet"


def _load_block_geom():
    global BLOCK
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ggml_block_geom as GBG
    need = ("F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K")   # == DEQUANTIZABLE, spelled here
    try:                                                        # so import order cannot break it
        g = GBG.geom(require=need)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit("gguf_reader cannot price a width without the pinned ggml header: %s" % e)
    BLOCK = {t: (g[t]["elts"], g[t]["bytes"]) for t in need}
    for t in ("Q4_0", "IQ4_NL"):            # readable-but-not-dequantized, kept for row sizing
        if t in g:
            BLOCK[t] = (g[t]["elts"], g[t]["bytes"])
    return BLOCK


_load_block_geom()

# Types this reader can actually dequantize. Anything else must fail, not be approximated.
DEQUANTIZABLE = ("F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K")

GIB = 2.0 ** 30
S_HALF = struct.Struct("<e")       # IEEE binary16, same as ggml_half (GGML_FP16_TO_FP32 is exact)
S_HALF2 = struct.Struct("<ee")
_LO = bytes([i & 15 for i in range(256)])
_HI = bytes([i >> 4 for i in range(256)])


def f16(buf, at=0):
    return S_HALF.unpack_from(buf, at)[0]


def row_bytes(tname, n_elts):
    """ggml_row_size() exactly: `(type_size*n + blck_size - 1) / blck_size` (integer ceil).

    The sum of every tensor's row_bytes plus the data-section start must equal the file size; the
    structure pass asserts that, which is what witnesses this table rather than trusting it."""
    if tname not in BLOCK:
        raise KeyError("no block geometry for type %r" % (tname,))
    be, bb = BLOCK[tname]
    return (bb * n_elts + be - 1) // be


def n_blocks(tname, n_elts):
    be, _ = BLOCK[tname]
    return -(-n_elts // be)


def bf16_to_f32_bytes(b):
    """bf16 -> f32 bit widening, at C speed: f32 = bf16 << 16, so LE bytes are [0,0,lo,hi]."""
    out = bytearray(len(b) * 2)
    out[2::4] = b[0::2]
    out[3::4] = b[1::2]
    return bytes(out)


def bf16_to_floats(b):
    """b is bf16 storage (2 bytes/elt); the result is one f32 per input element."""
    return struct.unpack("<%df" % (len(b) // 2), bf16_to_f32_bytes(b))


def f16_to_floats(b):
    return struct.unpack("<%de" % (len(b) // 2), b)


def f32_to_floats(b):
    return struct.unpack("<%df" % (len(b) // 4), b)


# ---------------------------------------------------------------------------------------------
# scale/min packing for the K-quants.
#
# The 12 `scales` bytes of Q4_K/Q5_K hold 8 6-bit scales and 8 6-bit mins (96 bits). Route used
# here is the decoder's (`get_scale_min_k4` in ggml-quants.c), transcribed:
#     j < 4:   d = q[j] & 63            m = q[j+4] & 63
#     j >= 4:  d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
#              m = (q[j+4] >>  4) | ((q[j]   >> 6) << 4)
# ---------------------------------------------------------------------------------------------
def qk_scale_min(scales):
    """-> ([sc0..sc7], [m0..m7]) as ints, decoder route."""
    sc = [0] * 8
    mg = [0] * 8
    for j in range(4):
        sc[j] = scales[j] & 63
        mg[j] = scales[j + 4] & 63
    for j in range(4, 8):
        sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        mg[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mg


def dequant(buf, tname, n_elts=None, inj=None):
    """Dequantize `buf` (exactly the bytes of a run of whole blocks) -> list[float].

    `inj` selects an INJECTED misreading (used only to prove the must-catch set fires); None is the
    correct reading. Injections are documented at INJECTIONS below and each one is a real converter
    mistake, not a random perturbation.
    """
    if tname not in DEQUANTIZABLE:
        raise NotImplementedError("refusing to guess type %s; DEQUANTIZABLE=%s"
                                  % (tname, sorted(DEQUANTIZABLE)))
    if tname == "F32":
        return list(f32_to_floats(buf[:4 * n_elts] if n_elts else buf))
    if tname == "F16":
        return list(f16_to_floats(buf[:2 * n_elts] if n_elts else buf))
    if tname == "BF16":
        return list(bf16_to_floats(buf[:2 * n_elts] if n_elts else buf))
    if tname == "Q4_K" and inj == "q4k_as_q4nl":
        # "read-Q4_K-as-Q4_NL": a K-quant read as a plain 4-bit block — 18-byte blocks of 32
        # elements, one f16 delta per block, value = d*(q-8). It drops the 6-bit sub-block scales
        # AND the min term (a plain 4-bit block has no min), and it walks the byte stream with the
        # wrong stride, which is exactly the failure a converter that mixed the two layouts makes.
        return _deq_q4_0(buf, n_elts)
    if tname == "Q8_0":
        return _deq_q8_0(buf, n_elts)
    if tname == "Q4_K":
        return _deq_q4_k(buf, n_elts, drop_min=(inj == "q4k_drop_min"))
    if tname == "Q5_K":
        return _deq_q5_k(buf, n_elts)
    if tname == "Q6_K":
        return _deq_q6_k(buf, n_elts)
    raise NotImplementedError(tname)


INJECTIONS = {
    "q4k_as_q4nl": "Q4_K bytes read as a plain (non-K) 4-bit block: 18-byte blocks, d*(q-8). "
                   "Drops the 6-bit sub-block scales and the min term.",
    "q4k_drop_min": "Q4_K decoded correctly except that the `min` term is omitted "
                    "(value = d*sc*q, never - dmin*m): a systematic positive offset.",
    "gate_up_swap": "the two halves of the fused HF `gate_up_proj` [E,2I,H] are taken in the wrong "
                    "order, so GGUF ffn_gate_exps gets HF rows [I,2I) and ffn_up_exps gets [0,I).",
    "expert_stack_transpose": "the [E,N,K] expert stack is indexed as if the expert axis were the "
                              "trailing axis (GGUF dims [K,N,E] read as [E,N,K]).",
    "block_shift_1": "the tensor data start is shifted by one byte, so every block is read from a "
                     "misaligned offset.",
}


def _deq_q4_0(buf, n_elts=None):
    out = []
    n = n_elts or (len(buf) // 18 * 32)
    for i in range(0, n // 32):
        d = f16(buf, i * 18)
        qs = buf[i * 18 + 2: i * 18 + 18]
        lo = qs.translate(_LO)
        hi = qs.translate(_HI)
        out.extend([d * (q - 8) for q in lo])
        out.extend([d * (q - 8) for q in hi])
    return out[:n]


def _deq_q8_0(buf, n_elts=None):
    out = []
    nb = len(buf) // 34 if n_elts is None else -(-n_elts // 32)
    for i in range(nb):
        d = f16(buf, i * 34)
        base = i * 34 + 2
        out.extend([d * q for q in struct.unpack_from("<32b", buf, base)])
    return out[:n_elts] if n_elts else out


def _deq_q4_k(buf, n_elts=None, drop_min=False):
    out = []
    nb = len(buf) // 144 if n_elts is None else -(-n_elts // 256)
    for i in range(nb):
        b = i * 144
        d = f16(buf, b)
        dm = f16(buf, b + 2)
        sc, mg = qk_scale_min(buf[b + 4: b + 16])
        qs = buf[b + 16: b + 144]
        lo = qs.translate(_LO)
        hi = qs.translate(_HI)
        for g in range(4):
            k0 = 32 * g
            s1, s2 = sc[2 * g], sc[2 * g + 1]
            m1 = 0.0 if drop_min else dm * mg[2 * g]
            m2 = 0.0 if drop_min else dm * mg[2 * g + 1]
            t1 = [d * s1 * q - m1 for q in range(16)]
            t2 = [d * s2 * q - m2 for q in range(16)]
            out.extend([t1[v] for v in lo[k0:k0 + 32]])
            out.extend([t2[v] for v in hi[k0:k0 + 32]])
    return out[:n_elts] if n_elts else out


def _deq_q5_k(buf, n_elts=None):
    out = []
    nb = len(buf) // 176 if n_elts is None else -(-n_elts // 256)
    for i in range(nb):
        b = i * 176
        d = f16(buf, b)
        dm = f16(buf, b + 2)
        sc, mg = qk_scale_min(buf[b + 4: b + 16])
        qh = buf[b + 16: b + 48]
        qs = buf[b + 48: b + 176]
        lo = qs.translate(_LO)
        hi = qs.translate(_HI)
        for g in range(4):
            k0 = 32 * g
            s1, s2 = sc[2 * g], sc[2 * g + 1]
            m1, m2 = dm * mg[2 * g], dm * mg[2 * g + 1]
            # 5th bit of element e is bit (e>>5) of qh[e & 31]: in this loop the sub-block index is
            # 2g (low nibbles) and 2g+1 (high nibbles), so bit 2g / bit 2g+1 at nibble position l.
            b1, b2 = 1 << (2 * g), 1 << (2 * g + 1)
            t1 = [d * s1 * ((q & 15) + (16 if qh[l] & b1 else 0)) - m1
                  for l, q in enumerate(lo[k0:k0 + 32])]
            t2 = [d * s2 * ((q & 15) + (16 if qh[l] & b2 else 0)) - m2
                  for l, q in enumerate(hi[k0:k0 + 32])]
            out.extend(t1)
            out.extend(t2)
    return out[:n_elts] if n_elts else out


def _deq_q6_k(buf, n_elts=None):
    out = []
    nb = len(buf) // 210 if n_elts is None else -(-n_elts // 256)
    for i in range(nb):
        b = i * 210
        d = f16(buf, b + 208)
        ql = buf[b: b + 128]
        qh = buf[b + 128: b + 192]
        sc = struct.unpack_from("<16b", buf, b + 192)
        for g in range(2):          # two 128-element groups; ql advances 64, qh advances 32
            qo, ho = 64 * g, 32 * g
            a = [ql[qo + l] & 15 | (((qh[ho + l] >> 0) & 3) << 4) for l in range(32)]
            bb = [ql[qo + l + 32] & 15 | (((qh[ho + l] >> 2) & 3) << 4) for l in range(32)]
            c = [ql[qo + l] >> 4 | (((qh[ho + l] >> 4) & 3) << 4) for l in range(32)]
            dd = [ql[qo + l + 32] >> 4 | (((qh[ho + l] >> 6) & 3) << 4) for l in range(32)]
            for k, tbl in enumerate((a, bb, c, dd)):
                # The scale index is the 16-element sub-block, not the 32-element chunk: in
                # dequantize_row_q6_K the four writes of a 128-group take sc[is+0], sc[is+2],
                # sc[is+4], sc[is+6] with is = l/16, and `sc += 8` after each 128-group, so the
                # index is 8g + 2k + (l>>4) = element//16.
                s0 = d * sc[8 * g + 2 * k]
                s1 = d * sc[8 * g + 2 * k + 1]
                out.extend([(s0 if l < 16 else s1) * (q - 32) for l, q in enumerate(tbl)])
    return out[:n_elts] if n_elts else out


# ---------------------------------------------------------------------------------------------
# Reference quantizers for the FLOOR PROBE.
#
# Transcribed from quantize_row_q8_0_ref / q4_K_ref / q5_K_ref / q6_K_ref in ggml-quants.c with
# one deliberate simplification: where the C code fits a sub-block scale iteratively
# (`make_qkx2_quants` / `make_qx_quants`), these use the plain min/max fit
# (scale = (max-min)/n_bins, or amax/n_bins where the C code uses a symmetric fit). That makes
# these quantizers *legitimate but unweighted*: they define what error is achievable on this data
# at this width with this grid, which is what a floor probe is for. They are NOT the reference the
# audit compares against - the bf16 authority is - and they only ever size a band.
# ---------------------------------------------------------------------------------------------
def _fp16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def _round_half_up(x):
    """C `nearest_int()` is round-half-away-from-zero; python's round() is half-to-even."""
    import math
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def quant_ref_deq(tname, xs, steps=None):
    """Quantize xs (list[float], len a whole number of blocks) and return the DEQUANTIZED values.

    If `steps` is a list, the realized step (the grid spacing actually used, per element) is
    appended to it. The floor probe's honesty check needs the grid the quantizer ACTUALLY
    produced, not the ideal one it started from: a 6-bit shared scale can coarsen a small-range
    sub-block well past its own min/max fit.
    """
    if tname == "F32":
        if steps is not None:
            steps.extend([0.0] * len(xs))
        return list(xs)
    if tname in ("F16", "BF16"):
        if steps is not None:
            steps.extend([0.0] * len(xs))
        return [_fp16(x) for x in xs]
    fn = {"Q8_0": _qref_q8_0, "Q4_K": _qref_q4_k, "Q5_K": _qref_q5_k, "Q6_K": _qref_q6_K}
    if tname not in fn:
        raise NotImplementedError(tname)
    return fn[tname](xs, steps)


def _qref_q8_0(xs, steps=None):
    out = []
    for i in range(len(xs) // 32):
        blk = xs[i * 32: i * 32 + 32]
        amax = max(abs(v) for v in blk)
        d = amax / 127.0
        dh = _fp16(d)
        if steps is not None:
            steps.extend([abs(dh)] * 32)
        if dh == 0.0:
            out.extend([0.0] * 32)
            continue
        out.extend([dh * max(-127, min(127, _round_half_up(v / dh))) for v in blk])
    if steps is not None:
        steps.extend([0.0] * (len(xs) - len(steps)))
    return out + list(xs[len(out):])


def _qref_k(xs, n_bins, steps=None):
    """Q4_K (n_bins=15) / Q5_K (n_bins=31): 8 sub-blocks of 32, shared 6-bit scale + min."""
    out = []
    for i in range(len(xs) // 256):
        blk = xs[i * 256: i * 256 + 256]
        scales, mins = [], []
        for j in range(8):
            sub = blk[j * 32: j * 32 + 32]
            lo, hi = min(sub), max(sub)
            scales.append((hi - lo) / n_bins)
            mins.append(max(0.0, -lo))
        max_scale = max(scales)
        max_min = max(mins)
        inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
        inv_min = 63.0 / max_min if max_min > 0 else 0.0
        d = _fp16(max_scale / 63.0)
        dmin = _fp16(max_min / 63.0)
        ls = [min(63, _round_half_up(inv_scale * scales[j])) for j in range(8)]
        lm = [min(63, _round_half_up(inv_min * mins[j])) for j in range(8)]
        for j in range(8):
            step = d * ls[j]
            off = dmin * lm[j]
            sub = blk[j * 32: j * 32 + 32]
            if steps is not None:
                steps.extend([abs(step)] * 32)
            if step == 0.0:
                out.extend([0.0 - off] * 32)
                continue
            out.extend([step * max(0, min(n_bins, _round_half_up((v + off) / step))) - off
                        for v in sub])
    return out + list(xs[len(out):])


def _qref_q4_k(xs, steps=None):
    return _qref_k(xs, 15, steps)


def _qref_q5_k(xs, steps=None):
    return _qref_k(xs, 31, steps)


def _qref_q6_K(xs, steps=None):
    out = []
    for i in range(len(xs) // 256):
        blk = xs[i * 256: i * 256 + 256]
        raw = []
        for j in range(16):
            sub = blk[j * 16: j * 16 + 16]
            amax = max(abs(v) for v in sub)
            raw.append(amax / 32.0)
        max_scale = max(raw)
        if max_scale == 0.0:
            out.extend([0.0] * 256)
            continue
        iscale = -128.0 / max_scale
        d = _fp16(1.0 / iscale)
        sc = [max(-128, min(127, _round_half_up(iscale * r))) for r in raw]
        for j in range(16):
            step = d * sc[j]
            sub = blk[j * 16: j * 16 + 16]
            if steps is not None:
                steps.extend([abs(step)] * 16)
            if step == 0.0:
                out.extend([0.0] * 16)
                continue
            out.extend([step * max(-32, min(31, _round_half_up(v / step))) for v in sub])
    return out + list(xs[len(out):])


# ---------------------------------------------------------------------------------------------
# GGUF container
# ---------------------------------------------------------------------------------------------
GGUF_VALUE_KINDS = {0: "U8", 1: "I8", 2: "U16", 3: "I16", 4: "U32", 5: "I32", 6: "F32",
                    7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "U64", 11: "I64", 12: "F64"}
_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
           6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
# GGUF BOOL is ONE byte, not eight. Reading it as 8 desyncs the metadata walk and the next string
# length comes back absurd; that bug cost the first run of tools/gguf_header_types.py (evidence:
# quantization/evidence/q4km-published-width-map-2026-10-26.md §"What else the file carries").


class Tensor:
    __slots__ = ("name", "dims", "type_id", "type_name", "offset", "n_elts", "row", "index")

    def __init__(self, index, name, dims, type_id, offset):
        self.index = index
        self.name = name
        self.dims = tuple(dims)                       # GGUF order: dims[0] is the fastest axis
        self.type_id = type_id
        self.type_name = GGML_TYPES.get(type_id, "type_%d" % type_id)
        self.offset = offset                          # bytes from the start of the data section
        self.n_elts = 1
        for x in dims:
            self.n_elts *= int(x)
        self.row = row_bytes(self.type_name, self.n_elts)

    @property
    def shape_hf(self):
        """HF (row-major, slowest-first) shape: the reverse of the GGUF order."""
        return tuple(reversed(self.dims))

    def block_bytes(self, block_index):
        be, bb = BLOCK[self.type_name]
        return self.offset + block_index * bb, bb


class Gguf:
    """Header parse + data windows. Nothing in __init__ touches the tensor-data section.

    `SELFTEST_TOKEN_REQUIRED` is enforced: data windows only open when the caller passes a token
    issued by `tools/audit_published_quant.py` after its synthetic-block selftest passed.
    """

    SELFTEST_TOKEN_REQUIRED = "run tools/audit_published_quant.py --selftest first"

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        # The header is parsed by pulling exactly the bytes each field asks for, so not one byte
        # past header_end — the first byte of the tensor-data section — is ever read. That is what
        # makes "this pass touched structure only" a witnessed claim instead of an assertion.
        with open(path, "rb") as fh:
            self._parse(fh)
            self.bytes_read = fh.tell()
        self.bytes_read_of_data_region_uninterpreted = 0
        self._fh = None
        self._mm = None

    class _Short(Exception):
        pass

    def _parse(self, fh):
        def take(k):
            v = fh.read(k)
            if len(v) != k:
                raise Gguf._Short("header truncated: asked for %d bytes at offset %d, got %d"
                                  % (k, fh.tell(), len(v)))
            return v

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def s():
            k = u64()
            return take(k).decode("utf-8", "replace")

        def value(vt):
            if vt == 8:
                return s()
            if vt in _SCALAR:
                f, k = _SCALAR[vt]
                v = struct.unpack(f, take(k))[0]
                return bool(v) if vt == 7 else v
            raise ValueError("unknown gguf value type %d at byte %d" % (vt, fh.tell()))

        if take(4) != b"GGUF":
            raise ValueError("not a GGUF (magic=%r)" % (self.path,))
        self.version = u32()
        self.n_tensors = u64()
        self.n_kv = u64()
        self.meta = {}
        for _ in range(self.n_kv):
            k = s()
            vt = u32()
            if vt == 9:
                et = u32()
                cnt = u64()
                if et in _SCALAR and et != 8:
                    take(_SCALAR[et][1] * cnt)
                    self.meta[k] = "%s x %d" % (GGUF_VALUE_KINDS.get(et, et), cnt)
                else:
                    sample = []
                    for i in range(cnt):
                        v = value(et)
                        if i < 3:
                            sample.append(str(v))
                    self.meta[k] = "%s x %d [%s...]" % (GGUF_VALUE_KINDS.get(et, et), cnt,
                                                        ", ".join(sample))
            else:
                self.meta[k] = value(vt)
        self.tensors = []
        for i in range(self.n_tensors):
            name = s()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            dt = u32()
            off = u64()
            self.tensors.append(Tensor(i, name, dims, dt, off))
        self.header_end = fh.tell()
        self.alignment = int(self.meta.get("general.alignment", 32)) or 32
        self.data_start = -(-self.header_end // self.alignment) * self.alignment
        self.by_name = {t.name: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise ValueError("duplicate tensor names in the header")

    # ---- data windows ---------------------------------------------------------------
    def open_data(self, token, why=""):
        """Open the tensor-data section. Refuses without a selftest token."""
        if token != SELFTEST_TOKEN:
            raise PermissionError(
                "tensor data is gated: this reader only opens the data section for a caller that "
                "holds the selftest token (%s). Refusing: %s" % (self.SELTEST_TOKEN_REQUIRED, why))
        if self._mm is None:
            import mmap
            self._fh = open(self.path, "rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def close(self):
        for x in (self._mm, self._fh):
            try:
                if x is not None:
                    x.close()
            except Exception:
                pass
        self._mm = None
        self._fh = None

    def read(self, token, start, length, why=""):
        mm = self.open_data(token, why)
        mm.seek(start)
        return mm.read(length)

    def blocks(self, token, tensor, block_indices, why=""):
        """-> {block_index: bytes} for the requested whole blocks of `tensor`."""
        mm = self.open_data(token, why)
        out = {}
        for bi in block_indices:
            rel, bb = tensor.block_bytes(bi)
            mm.seek(self.data_start + rel)
            out[bi] = mm.read(bb)
        return out


# The token is a fixed string known only to the audit tool's own module... which it is not: it is
# imported from there, so that a caller cannot hand in a literal without going through the gate.
SELFTEST_TOKEN = None  # set by tools/audit_published_quant.py via grant_selftest_token()


def grant_selftest_token(tok):
    """Called by the audit tool when (and only when) its synthetic-block selftest has passed."""
    global SELFTEST_TOKEN
    SELFTEST_TOKEN = tok
