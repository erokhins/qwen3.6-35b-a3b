/* moe_gather_kernels.h — the routed-expert gather: quantized expert read, SwiGLU, and the
 * rank-weighted K=8 combine, in ONE source compiled twice.
 *
 *   - `moe_gather.hip` -> hipcc, gfx1151: the device implementation under test.
 *   - `host_probe.cpp` -> g++: a host build of the SAME bodies, so a device-vs-host disagreement
 *     is attributable to the device compiler/ISA and not to two spellings of the arithmetic
 *     (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the
 *     thing under test"). The plan parser lives here too, for the same reason: both arms must see
 *     one plan, not two parsers.
 *
 * The reference this is gated against is NOT here. It is
 *   `speccheck/moe_ref.py`   — the committed independent implementation (the oracle), and
 *   `tools/gguf_reader.py`   — the audited CPU dequantizer (`quantization/evidence/
 *                             audit-selftest-2026-10-26.json`),
 * both driven from `gate.py` / `oracle_arm.py`.
 *
 * ---------------------------------------------------------------------------------------------
 * The declared arithmetic form (each line is what this file does, and what `speccheck/moe_ref.py`
 * is asked to reproduce under `Form(accumulate="f32", order="expert_asc", batching="grouped",
 * weight_cast="before")` or against the pinned dump):
 *
 *   dequant   computed in DOUBLE and rounded to f32 once, mirroring the reader's Python-double
 *             route. Deliberate: Q6_K's `d * sc * (q-32)` needs 11+8+6 = 25 significand bits and is
 *             therefore NOT exact in f32, which is exactly the case
 *             `prototype/q4k_dequant/q4k_dequant_kernels.h` §"Why the arithmetic below is plain f32"
 *             names as "breaks the exactness argument and must be re-gated". The f32 route is still
 *             computed, on the device, and its divergence from this one is counted per width as a
 *             census — a measurement with a count, not a band.
 *   gate/up   fp32 accumulator over H=2048 products, one declared order: each output element is
 *             owned by one thread walking j = 0..H-1 ascending. Long dots are NOT claimed bit-exact
 *             against torch's internal order (`specification/10-reference-dumps-and-gates.md`
 *             §"Bands are derived, not invented"); the bit-exact expert verdicts in this lane are
 *             carried by the exact fixtures, where every intermediate is an integer below 2^23 and
 *             therefore order-independent (`gate.py` §exactness bounds asserts it, per case).
 *   silu      z / (1 + exp(-z)), fp32, one expf per element — the model's `hidden_act: silu`
 *             (`specification/11-moe-block.md` §11.4). On the exact fixtures every gate satisfies
 *             |gate| >= MG_EXACT_EXP_MIN, so expf(-z) flushes to +0 or +inf and silu degenerates to
 *             the identity or to a signed zero: exact, and identical on both compilers. Off those
 *             fixtures the transcendental is compared and reported, never banded.
 *   combine   the reference form, verified against the pinned dumps before this file was written
 *             (`prototype/moe_gather/work/probe6.py`: 0 of 65536 elements differ per block for this
 *             form, while the fp32-accumulator variant differs in ~38.5k of 65536 — the form is
 *             load-bearing, not stylistic):
 *                 prod = bf16( f32(expert_raw[t,k]) * f32(w_bf16[t,k]) )   one rounding
 *                 acc  = bf16( f32(acc) + f32(prod) )                      one rounding
 *             with the addends taken in ASCENDING EXPERT-ID order (§11.3 row 3) and the weight
 *             reached BY RANK, not by expert id. R5's cast back to the logits dtype happens BEFORE
 *             the multiply (§11.2, prediction 4); `MG_MODE_CAST_AFTER` is the wrong order, and this
 *             gate's job is to be able to tell the two apart on real rows.
 *   FMA       contraction is compiled out (`-ffp-contract=off` in build.sh) for both arms, so the two
 *             compilers round the same sequence of operations. Where a product is provably exact this
 *             is irrelevant; where it is not, this file does not claim bit-exactness.
 * ---------------------------------------------------------------------------------------------
 */
#ifndef MOE_GATHER_KERNELS_H
#define MOE_GATHER_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define MG_DEV __host__ __device__
#define MG_GLOBAL __global__
#else
#define MG_DEV
#define MG_GLOBAL
#endif

/* ---- geometry, from checkpoints/Qwen3.6-35B-A3B.config.json --------------------------------- *
 * gate.py re-derives every one of these from the config and refuses to run if header and config
 * disagree (`check_header_enum`), so they are pinned twice rather than typed once.               */
#define MG_H 2048                 /* hidden_size                                                   */
#define MG_I 512                  /* moe_intermediate_size                                         */
#define MG_TWO_I 1024             /* 2*I: the fused gate|up row block, gate first (§11.4)          */
#define MG_K 8                    /* num_experts_per_tok                                           */
#define MG_E 256                  /* num_experts                                                   */
#define MG_QK_K 256               /* ggml super-block size for Q4_K / Q5_K / Q6_K                  */
#define MG_EXACT_EXP_MIN 128.0f   /* |gate| at or above this makes expf(-z) flush to +0 or +inf    */

/* ---- block geometry, from the pinned ggml-common.h structs via tools/ggml_block_geom.py ------- *
 * Q4_K = 144 B / 256 elts, Q5_K = 176 B / 256, Q6_K = 210 B / 256
 * (`quantization/evidence/audit-block-geometry-2026-10-26.md`). gate.py asserts these against the
 * reader's derived BLOCK table, so a hand-typed width cannot survive (`pitfalls/40`).            */
#define MG_Q4K_BYTES 144
#define MG_Q5K_BYTES 176
#define MG_Q6K_BYTES 210
#define MG_MAX_BLOCK_BYTES 210

#define MG_NAN_BITS 0x7FC00000u   /* what an unwritten slot reads, so a leak is loud               */

enum { MG_KIND_Q4K = 0, MG_KIND_Q5K = 1, MG_KIND_Q6K = 2, MG_KIND_COUNT = 3 };

/* Width per weight family, as measured on the published file
 * (`quantization/evidence/audit-published-quant-2026-10-26.md` §"Stored widths"): ffn_gate_exps and
 * ffn_up_exps are Q4_K in all 40 layers; ffn_down_exps is Q5_K in 37 layers and Q6_K in layers 34,
 * 38 and 39. Width is a per-family-per-tensor fact, not a per-file fact — `general.file_type` names
 * the headline of 80 tensors out of 733.                                                         */
enum { MG_ROLE_GATE = 0, MG_ROLE_UP = 1, MG_ROLE_DOWN = 2, MG_ROLE_COUNT = 3 };

/* ---- the modes: each is one converter or combine mistake an engine can actually make ---------- */
enum {
    MG_MODE_CORRECT = 0,
    /* --- expert-body modes (MG_CASE_EXPERT) --- */
    MG_MODE_HALVES_SWAPPED = 1,   /* gate and up exchanged: prediction 3's stride/offset swap, the
                                     `chunk(2)` boundary read from the wrong half of the 2I rows     */
    MG_MODE_INPUT_CHUNK = 2,      /* the fused pair chunked along the INPUT dim: the 2I*H elements
                                     read as [I, 2H] with gate = [:, 0:H], up = [:, H:2H]           */
    MG_MODE_EXPERT_ROTATE = 3,    /* the gather reads the next sampled expert of this block's set  */
    MG_MODE_ROW_OFF_1 = 4,        /* output row i read as row (i+1) % rows of the expert matrix    */
    MG_MODE_BLOCK_OFF_1 = 5,      /* the 256-element block index is off by one inside its matrix   */
    MG_MODE_BYTE_SHIFT_1 = 6,     /* the tensor data start is one byte late                        */
    MG_MODE_WIDTH_MISREAD = 7,    /* a Q5_K block decoded with the Q6_K decoder, and the reverse   */
    MG_MODE_SCALE_MIN_NAIVE = 8,  /* the 96 scale/min bits as 16 contiguous 6-bit fields           */
    MG_MODE_Q6K_D_BYTE_0 = 9,     /* Q6_K's d read at byte 0 instead of 208                        */
    /* --- combine modes (MG_CASE_COMBINE) --- */
    MG_MODE_W_BY_EXPERT_ID = 10,  /* w indexed by expert id instead of by rank (§11.3 row 3)       */
    MG_MODE_RANK_ACCUM = 11,      /* the K addends summed in rank order, not ascending expert id   */
    MG_MODE_CAST_AFTER = 12,      /* multiply by the fp32 weight, round the product (prediction 4) */
    MG_MODE_F32_COMBINE = 13,     /* fp32 accumulator rounded once, instead of a bf16 accumulator  */
    /* --- both --- */
    MG_MODE_POISON_NAN = 14,      /* leak guard: NaN every slot of the record                      */
    MG_MODE_COUNT = 15
};

/* What a case's output is judged against. Every case carries exactly one, and gate.py asserts each
 * class is populated (pitfalls/40: a counter that reads 0 is vacuous until witnessed non-zero).   */
enum {
    MG_EXP_MATCH = 0,     /* must equal the oracle/pinned expectation, bit for bit                  */
    MG_EXP_MISMATCH = 1,  /* must differ from the correct expectation somewhere                      */
    MG_EXP_ALT = 2,       /* must equal the independently derived expectation for THIS misreading    */
    MG_EXP_IMMUNE = 3,    /* must equal the correct expectation: this mode cannot move this case     */
    MG_EXP_REPORT = 4     /* compared and counted, not asserted: this arm crosses the quantization   */
                          /* (bf16 dump vs the published Q4_K bytes) or the transcendental boundary, */
                          /* and sizing a band for either is a floor probe plus a user decision,     */
                          /* not this gate's to invent (specification/11 §11.8).                    */
};

/* ---- what a case writes: one contiguous span of fp32 words ------------------------------------ *
 * bf16-carrying values are stored widened to fp32 (the repo's convention) so the gate compares
 * 4-byte patterns for every field and never guesses a dtype off a file (pitfalls/30 §"Read a dump's
 * dtype off its file size").                                                                     */
#define MG_OFF_GATE 0
#define MG_OFF_UP (MG_OFF_GATE + MG_I)
#define MG_OFF_S (MG_OFF_UP + MG_I)
#define MG_OFF_RAW (MG_OFF_S + MG_I)
#define MG_EXPERT_FLOATS (MG_OFF_RAW + MG_H)
#define MG_COMBINE_FLOATS MG_H
#define MG_MAX_CASE_FLOATS MG_EXPERT_FLOATS

#define MG_FIELD_COUNT 5
#define MG_F_GATE 0
#define MG_F_UP 1
#define MG_F_S 2
#define MG_F_RAW 3
#define MG_F_ROUTED 4

#define MG_CASE_EXPERT 0
#define MG_CASE_COMBINE 1
#define MG_CASE_KIND_COUNT 2

/* Where a combine case's K pre-weight vectors come from: the pinned dump slice, or this kernel's own
 * mode-0 raw outputs (the end-to-end arm on the exact fixtures). */
enum { MG_SRC_PINNED = 0, MG_SRC_KERNEL_RAW = 1 };


/* ---- the plan --------------------------------------------------------------------------------- */
typedef struct {
    int32_t kind;                 /* MG_KIND_* the bytes in the slab actually are                  */
    int32_t declared_kind;        /* MG_KIND_* the plan says the tensor IS; a disagreement makes the
                                     unit read NaN (loud) instead of silently re-decoding           */
    int32_t block;                /* index into MgBlock                                            */
    int32_t role;                 /* MG_ROLE_*                                                     */
    int32_t expert;               /* the expert id whose bytes these are                           */
    int32_t src_off;              /* byte offset of block 0 inside the slab                        */
    int32_t slot_bytes;           /* stride between consecutive blocks of THIS unit, in bytes. It is
                                     the stride the reader uses, and mg_verify_plan requires it to
                                     equal the kind's block size: a sampled unit is a CONTIGUOUS run
                                     of a real tensor's blocks, exactly as they sit in the file, and
                                     on a contiguous run the byte that really follows block k is
                                     block k+1's first byte — which is what MG_MODE_BYTE_SHIFT_1 is
                                     supposed to see. The earlier text claimed the slots were
                                     block_bytes+1 wide while the reader walked at block_bytes, i.e.
                                     two spellings of one address (pitfalls/40 §"Two self-consistent
                                     layouts joined by a branch are invisible to both sides"); the
                                     reader now uses this field and the plan must agree with it.    */
    int32_t n_blocks;             /* blocks in this unit = (rows * n_cols) / 256                   */
    int32_t rows;                 /* output rows this unit covers: I for gate/up, H for down       */
    int32_t set_pos;              /* position in the block's expert set, for the rotate mode       */
    int32_t main_n;               /* experts in the block's main (non-support) set                 */
    int32_t cls;                  /* 0 real bytes, 1 synthetic exact, 2 support                    */
    int32_t tensor_index;         /* the GGUF tensor-info record these bytes came from             */
} MgUnit;

typedef struct {
    int32_t block;
    int32_t row;                  /* token index within the block                                  */
    int32_t fixture;              /* 0 = real pinned row, 1 = exact synthetic row                  */
    int32_t ids_off;              /* byte offset of the K int32 expert ids                         */
    int32_t wcast_off;            /* byte offset of the K bf16-as-f32 cast-BEFORE weights (R5)     */
    int32_t wpre_off;             /* byte offset of the K fp32 pre-cast weights                    */
    int32_t x_off;                /* byte offset of the H fp32 activation values                   */
    int32_t raw_off;              /* byte offset of the K*H pre-weight vectors                     */
    int32_t id_asc;               /* 1 when the K ids are already ascending -> RANK_ACCUM immune   */
    int32_t w_by_eid_fixed;       /* 1 when ids[k] % K == k for all k -> W_BY_EXPERT_ID immune     */
    int32_t rawk_base;            /* float offset of slot 0's mode-0 raw output, for KERNEL_RAW arms */
    int32_t rawk_stride;          /* stride between consecutive slots' raw outputs                 */
} MgRow;

typedef struct {
    int32_t kind;                 /* MG_CASE_EXPERT | MG_CASE_COMBINE                              */
    int32_t mode;
    int32_t block, row, slot;
    int32_t u_gate, u_up, u_down; /* the units the CORRECT read uses                               */
    int32_t row_idx;              /* index into MgRow                                              */
    int32_t combine_src;          /* MG_SRC_* (COMBINE cases only)                                 */
    int32_t dst_off;              /* float offset of this case's record in out                     */
    int32_t exp_off;              /* float offset of its expectation in expect                     */
    int32_t cmp_off;              /* float offset of the same case at mode 0, or -1                */
    int32_t expect;               /* MG_EXP_*                                                      */
    int32_t n_floats;
    int32_t group;                /* the launch group = the block: one grouped launch per block    */
    int32_t bad;                  /* written by the kernel: elements differing from expect         */
    int32_t nan;                  /* written by the kernel: 1 if any slot came out NaN             */
} MgCase;

typedef struct {
    int32_t index;                /* the MoE block index (0, 1, 39, ...)                           */
    int32_t is_mtp;
    int32_t unit_base;            /* first unit of the main set: unit_base + pos*3 + role          */
    int32_t main_n;
    int32_t sup_lo, sup_hi;       /* inclusive support-unit range                                  */
    int32_t case_lo, case_hi;     /* inclusive case range of this block's grouped launch           */
    int32_t n_expert_cases, n_combine_cases;
} MgBlock;

typedef struct {
    MgUnit *units;   int n_units;
    MgCase *cases;   int n_cases;
    MgRow  *rows;    int n_rows;
    MgBlock *blocks; int n_blocks;
    const char *slab; size_t slab_bytes;
    const uint32_t *expect;     /* the uploaded expectation words, indexed by case exp_off       */
} MgPlan;

/* ---- shared primitives ------------------------------------------------------------------------ */
MG_DEV static inline uint32_t mg_bits_of(float f) {
    uint32_t b; memcpy(&b, &f, sizeof(b)); return b;
}
MG_DEV static inline float mg_float_from_bits(uint32_t b) {
    float f; memcpy(&f, &b, sizeof(f)); return f;
}
MG_DEV static inline uint16_t mg_bf16_of(float f) {
    /* round-to-nearest-even over the top 16 bits; a NaN stays a quiet NaN */
    const uint32_t b = mg_bits_of(f);
    if (((b >> 23) & 0xFFu) == 0xFFu && (b & 0x7FFFFFu)) return (uint16_t)((b >> 16) | 0x0040u);
    const uint32_t lb = b & 0xFFFFu;
    uint32_t r = b >> 16;
    if (lb > 0x8000u || (lb == 0x8000u && (r & 1u))) r += 1u;
    return (uint16_t)r;
}
MG_DEV static inline float mg_bf16_to_f32(uint16_t u) { return mg_float_from_bits((uint32_t)u << 16); }
MG_DEV static inline uint16_t mg_bf16_add(uint16_t a, uint16_t b) {
    return mg_bf16_of(mg_bf16_to_f32(a) + mg_bf16_to_f32(b));
}
MG_DEV static inline uint32_t mg_u16le(const uint8_t *p) {
    /* byte-wise on purpose: the byte-shift mode makes these reads unaligned, and a reinterpreted
     * uint16_t* there is UB the compiler is free to exploit. */
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}
MG_DEV static inline float mg_f16(uint32_t bits) {
    /* f16 -> f32 by bit pattern, so the denormal path is ours rather than the C library's and the
     * two compilers cannot disagree. NaN is canonicalised, matching the reader's struct.unpack('<e')
     * route — see prototype/q4k_dequant/q4k_dequant_kernels.h §q4kd_f16 for why. */
    const uint32_t sign = (bits >> 15) & 1u, rest = bits & 0x7FFFu;
    uint32_t fb;
    if ((rest & 0x7C00u) == 0u) {
        if (rest == 0u) fb = sign << 31;
        else { const float m = (float)rest;
               return sign ? -(m * 5.9604644775390625e-8f) : (m * 5.9604644775390625e-8f); }
    } else if ((rest & 0x7C00u) == 0x7C00u) {
        fb = (rest & 0x3FFu) ? ((sign << 31) | 0x7FC00000u) : ((sign << 31) | 0x7F800000u);
    } else {
        fb = (sign << 31) | ((((rest >> 10) & 0x1Fu) + 112u) << 23) | ((rest & 0x3FFu) << 13);
    }
    return mg_float_from_bits(fb);
}
/* The one equality rule this gate uses, mirrored in gate.py §word_eq: strict bit equality on finite
 * values including the sign of a zero, with the single NaN exception the q4k lane established
 * (neither route defines the sign or payload of a NaN product). */
MG_DEV static inline int mg_word_eq(uint32_t a, uint32_t b) {
    if (a == b) return 1;
    const uint32_t ma = a & 0x7FFFFFFFu, mb = b & 0x7FFFFFFFu;
    return ma > 0x7F800000u && mb > 0x7F800000u;
}
MG_DEV static inline float mg_nan(void) { return mg_float_from_bits(MG_NAN_BITS); }
MG_DEV static inline int mg_is_nan(float f) {
    return (mg_bits_of(f) & 0x7F800000u) == 0x7F800000u && (mg_bits_of(f) & 0x007FFFFFu) != 0u;
}

/* ---- the decoders ------------------------------------------------------------------------------ *
 * Element order inside a 256-element super-block, transcribed from tools/gguf_reader.py's _deq_q4_k /
 * _deq_q5_k / _deq_q6_k (the audited reference) and uniform across the three widths in the
 * (g, half, l) coordinates the q4k lane uses:
 *   Q4_K / Q5_K: e = 64g + 32*half + l; the nibble pair lives at qs[32g + l], the sub-block is
 *                j = 2g + half; Q5_K's fifth bit is bit j of qh[l]. qs is at byte 48 for Q5_K
 *                (qh occupies 16..47) and at byte 16 for Q4_K.
 *   Q6_K:        e = 128g + 32k + l, k in 0..3; the low nibble is ql[64g + l + 32*(k&1)] (low half
 *                for k<2, high half otherwise) and the two high bits are qh[32g + l] >> 2k;
 *                the scale index is 8g + 2k + (l>>4), sc is 16 SIGNED bytes at 192, and d is the f16
 *                at byte 208 — not byte 0, which is what MG_MODE_Q6K_D_BYTE_0 gets wrong.
 * Both routes evaluate every product exactly and round once; the double route is the judged one.
 */
MG_DEV static inline void mg_scale_min(int j, const uint8_t *q, int32_t *sc, int32_t *mgv) {
    /* get_scale_min_k4 (ggml-quants.c, the decoder route), as transcribed in tools/gguf_reader.py
     * §qk_scale_min. */
    if (j < 4) { *sc = (int32_t)q[j] & 63; *mgv = (int32_t)q[j + 4] & 63; }
    else {
        *sc = ((int32_t)q[j + 4] & 0xF) | (((int32_t)q[j - 4] >> 6) << 4);
        *mgv = ((int32_t)q[j + 4] >> 4) | (((int32_t)q[j] >> 6) << 4);
    }
}
MG_DEV static inline void mg_scale_min_naive(int j, const uint8_t *q, int32_t *sc, int32_t *mgv) {
    /* the WRONG 6-bit unpack: 16 contiguous 6-bit fields over the same 12 bytes. Plausible from the
     * header comment "scales and mins, quantized with 6 bits", structurally different from the route
     * above — which is the point: a gate that cannot tell these apart cannot tell any scale-order bug
     * apart. */
    const int b1 = 6 * j, b2 = 6 * (j + 8);
    const uint32_t w1 = (uint32_t)q[b1 >> 3] | ((uint32_t)q[(b1 >> 3) + 1] << 8);
    const uint32_t w2 = (uint32_t)q[b2 >> 3] | ((uint32_t)q[(b2 >> 3) + 1] << 8);
    *sc = (int32_t)((w1 >> (b1 & 7)) & 63u);
    *mgv = (int32_t)((w2 >> (b2 & 7)) & 63u);
}

MG_DEV static inline int mg_kind_ok(int kind) { return kind >= 0 && kind < MG_KIND_COUNT; }
MG_DEV static inline int mg_block_bytes(int kind) {
    return kind == MG_KIND_Q4K ? MG_Q4K_BYTES : (kind == MG_KIND_Q5K ? MG_Q5K_BYTES : MG_Q6K_BYTES);
}

MG_DEV static inline float mg_deq_dbl(const uint8_t *blk, int kind, int e, int mode) {
    if (kind == MG_KIND_Q6K) {
        const int d_off = (mode == MG_MODE_Q6K_D_BYTE_0) ? 0 : 208;
        const double d = (double)mg_f16(mg_u16le(blk + d_off));
        const int g = e >> 7, k = (e >> 5) & 3, l = e & 31;
        const uint8_t lo = blk[64 * g + l + 32 * (k & 1)];
        const uint8_t hi = blk[128 + 32 * g + l];
        const int nib = (k < 2) ? (lo & 15) : (lo >> 4);
        const int q = nib | ((((hi >> (2 * k)) & 3) << 4));
        const int8_t s = (int8_t)blk[192 + 8 * g + 2 * k + (l >> 4)];
        return (float)(d * (double)s * (double)(q - 32));
    }
    const int is5 = (kind == MG_KIND_Q5K);
    const double d = (double)mg_f16(mg_u16le(blk + 0));
    const double dm = (double)mg_f16(mg_u16le(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + (is5 ? 48 : 16);
    const int g = e >> 6, half = (e >> 5) & 1, l = e & 31, j = 2 * g + half;
    int32_t sc, mgv;
    if (mode == MG_MODE_SCALE_MIN_NAIVE) mg_scale_min_naive(j, scales, &sc, &mgv);
    else mg_scale_min(j, scales, &sc, &mgv);
    const uint8_t byte = qs[32 * g + l];
    int q = half ? (byte >> 4) : (byte & 15);
    if (is5) q |= (((blk[16 + l] >> j) & 1) << 4);
    const double scale = d * (double)sc;
    const double mins = dm * (double)mgv;
    return (float)(scale * (double)q - mins);
}

/* The fp32 route, computed only to be COUNTED: how far the engine's cheap path sits from the double
 * route this gate judges on. Q4_K (11+6+4 = 21 significand bits) and Q5_K (11+6+5 = 22) are exact in
 * f32, so their divergence count must be 0; Q6_K (11+8+6 = 25) is not, and a non-zero count there is
 * the finding that prototype/q4k_dequant's header predicts. */
MG_DEV static inline float mg_deq_f32(const uint8_t *blk, int kind, int e, int mode) {
    if (kind == MG_KIND_Q6K) {
        const int d_off = (mode == MG_MODE_Q6K_D_BYTE_0) ? 0 : 208;
        const float d = mg_f16(mg_u16le(blk + d_off));
        const int g = e >> 7, k = (e >> 5) & 3, l = e & 31;
        const uint8_t lo = blk[64 * g + l + 32 * (k & 1)];
        const uint8_t hi = blk[128 + 32 * g + l];
        const int nib = (k < 2) ? (lo & 15) : (lo >> 4);
        const int q = nib | ((((hi >> (2 * k)) & 3) << 4));
        const int8_t s = (int8_t)blk[192 + 8 * g + 2 * k + (l >> 4)];
        return d * (float)s * (float)(q - 32);
    }
    const int is5 = (kind == MG_KIND_Q5K);
    const float d = mg_f16(mg_u16le(blk + 0)), dm = mg_f16(mg_u16le(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + (is5 ? 48 : 16);
    const int g = e >> 6, half = (e >> 5) & 1, l = e & 31, j = 2 * g + half;
    int32_t sc, mgv;
    if (mode == MG_MODE_SCALE_MIN_NAIVE) mg_scale_min_naive(j, scales, &sc, &mgv);
    else mg_scale_min(j, scales, &sc, &mgv);
    const uint8_t byte = qs[32 * g + l];
    int q = half ? (byte >> 4) : (byte & 15);
    if (is5) q |= (((blk[16 + l] >> j) & 1) << 4);
    return d * (float)sc * (float)q - dm * (float)mgv;
}

/* ---- one element read of one unit, under a mode ------------------------------------------------ *
 * All addressing is char-offset arithmetic scaled by the block stride, never a typed-pointer offset
 * (pitfalls/40: `ptr + slot * bytes` on a typed pointer is a scaled offset and only bites at slot
 * >= 1). Returns 0 and writes *v on a defined read; returns 1 means "not defined on this plan", for
 * which the caller writes MG_NAN_BITS so a leak stays loud instead of reading as a match.
 */
MG_DEV static inline int mg_unit_elt(const MgPlan *p, const MgUnit *u, int role, int row, int col,
                                     int n_cols, int mode, float *v, float *v32) {
    if (u->kind != u->declared_kind || !mg_kind_ok(u->kind)) return 1;   /* plan bug: refuse       */
    int kind = u->kind;
    if (mode == MG_MODE_WIDTH_MISREAD) {
        if (kind == MG_KIND_Q5K) kind = MG_KIND_Q6K;
        else if (kind == MG_KIND_Q6K) kind = MG_KIND_Q5K;
        else return 1;                               /* Q4_K has no defined width swap             */
    }
    const int bb = mg_block_bytes(kind);
    long long flat;
    if (mode == MG_MODE_INPUT_CHUNK && role != MG_ROLE_DOWN) {
        /* the fused [2I, H] buffer read as [I, 2H]: output row i, output column c takes element
         * (2i + (c >= H)) * H + (c % H) of the SAME expert buffer. Within-buffer, so it is judgeable
         * on a bounded slab, unlike the transposed-stack reading. */
        const int half = (col >= n_cols) ? 1 : 0;
        const int c = col - half * n_cols;
        flat = ((long long)row * 2 + half) * n_cols + c;
    } else {
        int r = row;
        if (mode == MG_MODE_ROW_OFF_1) r = (role == MG_ROLE_DOWN) ? (row + 1) % MG_H
                                                                  : (row + 1) % MG_TWO_I;
        if (r < 0 || r >= u->rows || col < 0 || col >= n_cols) return 1;
        flat = (long long)r * n_cols + col;
    }
    long long blk = flat / MG_QK_K;
    const int e = (int)(flat % MG_QK_K);
    if (blk < 0 || blk >= u->n_blocks) return 1;
    if (mode == MG_MODE_BLOCK_OFF_1) blk = (blk + 1) % u->n_blocks;   /* wraps inside this unit     */
    long long off = (long long)u->src_off + blk * u->slot_bytes;   /* the ONE stride expression     */
    if (mode == MG_MODE_BYTE_SHIFT_1) off += 1;      /* the byte that really precedes the read      */
    if (off < 0 || off + bb > (long long)p->slab_bytes) return 1;
    const uint8_t *bp = (const uint8_t *)p->slab + off;
    *v = mg_deq_dbl(bp, kind, e, mode);
    if (v32) *v32 = mg_deq_f32(bp, kind, e, mode);
    return 0;
}

/* ---- which unit a mode reads ------------------------------------------------------------------- */
MG_DEV static inline const MgUnit *mg_resolve_unit(const MgPlan *p, const MgCase *c, int role,
                                                   int mode) {
    const MgBlock *b = &p->blocks[c->block];
    const MgUnit *u = (role == MG_ROLE_GATE) ? &p->units[c->u_gate]
                    : (role == MG_ROLE_UP)   ? &p->units[c->u_up]
                                             : &p->units[c->u_down];
    switch (mode) {
        case MG_MODE_CORRECT:
        case MG_MODE_W_BY_EXPERT_ID:
        case MG_MODE_RANK_ACCUM:
        case MG_MODE_CAST_AFTER:
        case MG_MODE_F32_COMBINE:
        case MG_MODE_BLOCK_OFF_1:
        case MG_MODE_BYTE_SHIFT_1:
        case MG_MODE_WIDTH_MISREAD:
        case MG_MODE_SCALE_MIN_NAIVE:
        case MG_MODE_Q6K_D_BYTE_0:
        case MG_MODE_ROW_OFF_1:
        case MG_MODE_INPUT_CHUNK:
            return u;
        case MG_MODE_EXPERT_ROTATE: {
            if (b->main_n <= 0 || u->cls == 2) return 0;
            const int pos = (u->set_pos + 1) % b->main_n;      /* next sampled expert of THIS block  */
            const MgUnit *r = &p->units[b->unit_base + pos * MG_ROLE_COUNT + role];
            /* The unit's `block` field is an INDEX INTO p->blocks (that is what mg_verify_plan
             * compares case.block against), while b->index is the MoE block NUMBER (0, 39, 40).
             * Comparing the two made this mode resolve to NULL on every block whose MoE number is not
             * its array index — i.e. every block but layer0 — and the case would have read NaN and
             * looked like a mismatch instead of a defect. One address expression, one identity. */
            if (r->block != c->block || r->role != role) return 0;
            return r;
        }
        case MG_MODE_HALVES_SWAPPED:
            if (role == MG_ROLE_DOWN) return u;
            /* gate and up exchanged: the fused 2I-row block read from the other half — the
             * stride/offset swap prediction 3 says must be checked, not re-derived. */
            return (role == MG_ROLE_GATE) ? &p->units[c->u_up] : &p->units[c->u_gate];
        default:
            return u;
    }
}

/* ---- per-case counters ------------------------------------------------------------------------- *
 * The kernel counts its own disagreements against the uploaded expectation, and gate.py recomputes
 * the same numbers from the words it read back out of out.f32. Two sites, one truth: a device counter
 * and a host comparator that disagree is a finding, not a rounding (`pitfalls/40` §"A counter
 * incremented at two sites double-counts" — the right-hand side here comes from an independent source,
 * the output file). */
typedef struct { int32_t bad, nan, route_bad, route_n; } MgCounters;

#if defined(__HIPCC__) || defined(__CUDACC__)
#define MG_BARRIER() __syncthreads()
#define MG_ACC(dst, v) atomicAdd((dst), (v))
#else
#define MG_BARRIER() ((void)0)      /* the host build runs one case with tid = 0, nthreads = 1     */
#define MG_ACC(dst, v) (*(dst) += (v))
#endif

/* silu(z) = z / (1 + exp(-z)) — specification/11-moe-block.md §11.4, hidden_act: silu. */
MG_DEV static inline float mg_silu(float z) { return z / (1.0f + expf(-z)); }

/* ---- the expert body --------------------------------------------------------------------------- *
 * One case = one (block, row, slot): one selected expert of one token. `out` receives
 * gate[I] | up[I] | s[I] | raw[H]. One thread per output element, the dot walked j ascending. This is
 * the shape a correctness gate takes, NOT the shape the engine's throughput kernel will take (no
 * tiling, no shared-memory staging, one dequant per MAC), and no timing is taken anywhere in this
 * lane, so nothing here prices anything.
 */
MG_DEV static inline void mg_expert_case(const MgPlan *p, const MgCase *c, float *out,
                                         int tid, int nthreads, MgCounters *ct) {
    const MgRow *r = &p->rows[c->row_idx];
    const int mode = c->mode;
    const float *x = (const float *)(p->slab + r->x_off);
    int bad = 0, nanseen = 0, rbad = 0, rn = 0;
    if (mode == MG_MODE_POISON_NAN) {
        for (int i = tid; i < MG_EXPERT_FLOATS; i += nthreads) out[i] = mg_nan();
        MG_ACC(&ct->nan, 1);
        return;
    }
    /* --- phase 1: gate and up. Row t of the fused [2I, H] read; gate first (§11.4). ------------ */
    for (int t = tid; t < MG_TWO_I; t += nthreads) {
        const int role = (t < MG_I) ? MG_ROLE_GATE : MG_ROLE_UP;
        const int row = (t < MG_I) ? t : t - MG_I;
        const MgUnit *u = mg_resolve_unit(p, c, role, mode);
        float acc = 0.0f;
        if (!u) { out[t] = mg_nan(); nanseen = 1; continue; }
        for (int j = 0; j < MG_H; ++j) {
            float w, w32 = 0.0f;
            if (!mg_unit_elt(p, u, role, row, j, MG_H, mode, &w, (mode == MG_MODE_CORRECT) ? &w32 : 0)) {
                acc = mg_nan(); nanseen = 1; break;
            }
            if (mode == MG_MODE_CORRECT) {
                /* the width census: same bytes, two routes, counted per case */
                rn += 1;
                if (!mg_word_eq(mg_bits_of(w), mg_bits_of(w32))) rbad += 1;
            }
            acc += x[j] * w;
        }
        out[t] = acc;
    }
    MG_BARRIER();
    /* --- phase 2: s = silu(gate) * up ---------------------------------------------------------- */
    for (int t = tid; t < MG_I; t += nthreads) {
        const float g = out[MG_OFF_GATE + t], up = out[MG_OFF_UP + t];
        out[MG_OFF_S + t] = mg_silu(g) * up;
    }
    MG_BARRIER();
    /* --- phase 3: raw = down @ s ---------------------------------------------------------------- */
    for (int t = tid; t < MG_H; t += nthreads) {
        const MgUnit *u = mg_resolve_unit(p, c, MG_ROLE_DOWN, mode);
        float acc = 0.0f;
        if (!u) { out[MG_OFF_RAW + t] = mg_nan(); nanseen = 1; continue; }
        for (int j = 0; j < MG_I; ++j) {
            float w, w32 = 0.0f;
            if (!mg_unit_elt(p, u, MG_ROLE_DOWN, t, j, MG_I, mode, &w,
                             (mode == MG_MODE_CORRECT) ? &w32 : 0)) {
                acc = mg_nan(); nanseen = 1; break;
            }
            if (mode == MG_MODE_CORRECT) {
                rn += 1;
                if (!mg_word_eq(mg_bits_of(w), mg_bits_of(w32))) rbad += 1;
            }
            acc += out[MG_OFF_S + j] * w;
        }
        out[MG_OFF_RAW + t] = acc;
    }
    /* --- the case's own verdict against the uploaded expectation -------------------------------- */
    const uint32_t *e = p->expect + c->exp_off;
    for (int i = tid; i < MG_EXPERT_FLOATS; i += nthreads) {
        const uint32_t got = mg_bits_of(out[i]);
        if (got == MG_NAN_BITS && e[i] != MG_NAN_BITS) nanseen = 1;
        if (!mg_word_eq(got, e[i])) bad += 1;
    }
    MG_ACC(&ct->bad, bad);
    if (nanseen) MG_ACC(&ct->nan, 1);
    MG_ACC(&ct->route_bad, rbad);
    MG_ACC(&ct->route_n, rn);
}

/* ---- the combine ------------------------------------------------------------------------------- *
 * routed[t] = sum over the token's K experts, in ASCENDING EXPERT-ID order, of
 * bf16(bf16(expert_raw[t,k] * w[t,rank]) added into a bf16 accumulator) — the form that reproduces
 * the pinned `routed_sum` bit for bit (header §combine). The weight is reached BY RANK.
 */
MG_DEV static inline void mg_combine_order(const int *ids, int mode, int *order) {
    for (int k = 0; k < MG_K; ++k) order[k] = k;
    if (mode == MG_MODE_RANK_ACCUM) return;                 /* rank order is the injection          */
    for (int a = 0; a < MG_K - 1; ++a)                      /* stable insertion sort by expert id    */
        for (int b = 0; b < MG_K - 1 - a; ++b)
            if (ids[order[b]] > ids[order[b + 1]]) {
                const int t = order[b]; order[b] = order[b + 1]; order[b + 1] = t;
            }
}

MG_DEV static inline void mg_combine_case(const MgPlan *p, const MgCase *c, const float *rawbase,
                                          int raw_stride, int raw_off,
                                          float *out, int tid, int nthreads, MgCounters *ct) {
    const uint32_t *e = p->expect + c->exp_off;
    const MgRow *r = &p->rows[c->row_idx];
    const int mode = c->mode;
    const int *ids = (const int *)(p->slab + r->ids_off);
    const float *wcast = (const float *)(p->slab + r->wcast_off);
    const float *wpre = (const float *)(p->slab + r->wpre_off);
    int bad = 0, nanseen = 0;
    if (mode == MG_MODE_POISON_NAN) {
        for (int i = tid; i < MG_COMBINE_FLOATS; i += nthreads) out[i] = mg_nan();
        MG_ACC(&ct->nan, 1);
        return;
    }
    int order[MG_K];
    mg_combine_order(ids, mode, order);
    for (int h = tid; h < MG_H; h += nthreads) {
        uint16_t acc = 0u;                                   /* +0.0 in the model dtype             */
        if (mode == MG_MODE_F32_COMBINE) {
            float facc = 0.0f;
            for (int i = 0; i < MG_K; ++i) {
                const int k = order[i];
                const float w = (mode == MG_MODE_CAST_AFTER) ? wpre[k] : wcast[k];
                facc += rawbase[(long long)k * raw_stride + raw_off + h] * w;
            }
            out[h] = mg_bf16_to_f32(mg_bf16_of(facc));
            continue;
        }
        for (int i = 0; i < MG_K; ++i) {
            const int k = order[i];
            /* cast-before (the reference, R5): the weight is the bf16 value. cast-after keeps the
             * 8 mantissa bits the reference threw away — prediction 4's different number. */
            float w = (mode == MG_MODE_CAST_AFTER) ? wpre[k] : wcast[k];
            if (mode == MG_MODE_W_BY_EXPERT_ID) {
                /* §11.3 row 3: the weight array indexed by expert id instead of by rank. Both arrays
                 * carry K entries at the site, so nothing but the numbers can catch it. */
                const int e = ids[k] % MG_K;
                w = (mode == MG_MODE_CAST_AFTER) ? wpre[e] : wcast[e];
            }
            const uint16_t prod = mg_bf16_of(rawbase[(long long)k * raw_stride + raw_off + h] * w);
            acc = mg_bf16_add(acc, prod);
        }
        out[h] = mg_bf16_to_f32(acc);
    }
    for (int i = tid; i < MG_COMBINE_FLOATS; i += nthreads) {
        const uint32_t got = mg_bits_of(out[i]);
        if (got == MG_NAN_BITS && e[i] != MG_NAN_BITS) nanseen = 1;
        if (!mg_word_eq(got, e[i])) bad += 1;
    }
    MG_ACC(&ct->bad, bad);
    if (nanseen) MG_ACC(&ct->nan, 1);
}

/* ---- the reach table ---------------------------------------------------------------------------- *
 * The kernel's own claim about which fields a mode can move, compiled rather than mirrored in Python
 * (the same trick prototype/moe_router uses: the gate asks the binary, so a renumbered enum cannot
 * quietly change what is judged). NO = a move there would be a second bug; YES = it must move
 * somewhere; MAY = reachable but data-dependent. */
MG_DEV static inline int mg_reaches(int mode, int case_kind, int field) {
    const int is_exp = (case_kind == MG_CASE_EXPERT);
    if (mode == MG_MODE_POISON_NAN) return 2;                       /* MAY: every field, by design  */
    if (!is_exp && field != MG_F_ROUTED) return 0;
    if (is_exp && field == MG_F_ROUTED) return 0;
    switch (mode) {
        case MG_MODE_CORRECT: return 0;
        /* expert body */
        case MG_MODE_HALVES_SWAPPED:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_INPUT_CHUNK:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_EXPERT_ROTATE:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_ROW_OFF_1:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_BLOCK_OFF_1:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_BYTE_SHIFT_1:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_WIDTH_MISREAD:
            return (field == MG_F_RAW) ? 1 : 0;                     /* a down-only misread           */
        case MG_MODE_SCALE_MIN_NAIVE:
            return (field == MG_F_GATE || field == MG_F_UP) ? 1
                 : (field == MG_F_S || field == MG_F_RAW) ? 1 : 0;
        case MG_MODE_Q6K_D_BYTE_0:
            return (field == MG_F_RAW) ? 1 : 0;                     /* a down-only misread           */
        /* combine */
        case MG_MODE_W_BY_EXPERT_ID:
        case MG_MODE_RANK_ACCUM:
        case MG_MODE_CAST_AFTER:
        case MG_MODE_F32_COMBINE:
            return (field == MG_F_ROUTED) ? 1 : 0;
        default: return 0;
    }
}

/* ---- where a combine case's K pre-weight vectors live ------------------------------------------ *
 * One spelling shared by both arms, because "the writer's address and the reader's address" is
 * exactly the pair that a two-self-consistent-layout bug hides in (pitfalls/40 §"Two self-consistent
 * layouts joined by a branch"). Returns 0 on success; 1 means the source is not defined and the
 * caller must NaN the record.
 */
MG_DEV static inline int mg_combine_raw_src(const MgPlan *p, const MgCase *c, const float *outbuf,
                                            const float **rawbase, int *stride, int *off) {
    const MgRow *r = &p->rows[c->row_idx];
    if (c->combine_src == MG_SRC_PINNED) {
        *rawbase = (const float *)(p->slab + r->raw_off);
        *stride = MG_H; *off = 0;
        return 0;
    }
    if (c->combine_src == MG_SRC_KERNEL_RAW) {
        if (r->rawk_stride <= 0 || c->slot < 0) return 1;
        *rawbase = outbuf + r->rawk_base;
        *stride = r->rawk_stride; *off = MG_OFF_RAW;
        return 0;
    }
    return 1;
}

/* ---- the field spans --------------------------------------------------------------------------- */
MG_DEV static inline int mg_field_span(int case_kind, int field, int *off, int *n) {
    if (case_kind == MG_CASE_EXPERT) {
        if (field == MG_F_GATE)  { *off = MG_OFF_GATE; *n = MG_I; return 1; }
        if (field == MG_F_UP)    { *off = MG_OFF_UP;   *n = MG_I; return 1; }
        if (field == MG_F_S)     { *off = MG_OFF_S;    *n = MG_I; return 1; }
        if (field == MG_F_RAW)   { *off = MG_OFF_RAW;  *n = MG_H; return 1; }
    } else if (case_kind == MG_CASE_COMBINE && field == MG_F_ROUTED) {
        *off = 0; *n = MG_H; return 1;
    }
    return 0;
}

/* The dispatcher both arms call, so "what a case does" has exactly one spelling. `out` is the case's
 * own record span; `rawbase`/`raw_stride`/`raw_off` locate the K pre-weight vectors for a COMBINE
 * case and are ignored for an EXPERT case. */
MG_DEV static inline void mg_run_case(const MgPlan *p, const MgCase *c, const float *rawbase,
                                      int raw_stride, int raw_off, float *out,
                                      int tid, int nthreads, MgCounters *ct) {
    if (c->kind == MG_CASE_EXPERT) mg_expert_case(p, c, out, tid, nthreads, ct);
    else mg_combine_case(p, c, rawbase, raw_stride, raw_off, out, tid, nthreads, ct);
}

/* ---- the plan file, parsed once per arm by identical code -------------------------------------- *
 * Tagged rows, tab separated: U / R / B / C. Host-only (it opens files), which is why it is plain
 * `static inline` rather than MG_DEV: a __host__ __device__ function may not call fopen when the
 * device half is compiled. Field counts are asserted, because a driver that parsed zero rows would
 * otherwise launch nothing and print PASS (pitfalls/40 §"A driver whose parser matches nothing
 * prints PASS"). The caller asserts parsed == counted.
 */
#define MG_PLAN_FIELDS_U 16
#define MG_PLAN_FIELDS_R 13
#define MG_PLAN_FIELDS_B 10
#define MG_PLAN_FIELDS_C 17

static inline int mg_split(char *line, char **out, int max) {
    int n = 0;
    char *s = line;
    while (n < max) {
        out[n++] = s;
        char *t = strchr(s, '\t');
        if (!t) break;
        *t = '\0';
        s = t + 1;
    }
    while (n > 0 && (line[strlen(line) - 1] == '\n' || line[strlen(line) - 1] == '\r')) {
        line[strlen(line) - 1] = '\0';
    }
    return n;
}

static inline int mg_read_plan(const char *path, MgPlan *p, int want_units, int want_rows,
                               int want_cases, int want_blocks,
                               int *parsed_u, int *parsed_r, int *parsed_c, int *parsed_b) {
    FILE *fh = fopen(path, "rb");
    if (!fh) return -1;
    char line[4096];
    char *f[24];
    int nu = 0, nr = 0, nc = 0, nb = 0, bad = 0;
    *parsed_u = *parsed_r = *parsed_c = *parsed_b = 0;
    while (fgets(line, sizeof(line), fh)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        const int nf = mg_split(line, f, 24);
        if (f[0][0] == 'U') {
            if (nf != MG_PLAN_FIELDS_U) { bad++; continue; }
            MgUnit *u = &p->units[nu];
            u->kind = atoi(f[1]); u->declared_kind = atoi(f[2]); u->block = atoi(f[3]);
            u->role = atoi(f[4]); u->expert = atoi(f[5]); u->src_off = atoi(f[6]);
            u->slot_bytes = atoi(f[7]); u->n_blocks = atoi(f[8]); u->rows = atoi(f[9]);
            u->set_pos = atoi(f[10]); u->main_n = atoi(f[11]); u->cls = atoi(f[12]);
            u->tensor_index = atoi(f[13]);
            nu++;
        } else if (f[0][0] == 'R') {
            if (nf != MG_PLAN_FIELDS_R) { bad++; continue; }
            MgRow *r = &p->rows[nr];
            r->block = atoi(f[1]); r->row = atoi(f[2]); r->fixture = atoi(f[3]);
            r->ids_off = atoi(f[4]); r->wcast_off = atoi(f[5]); r->wpre_off = atoi(f[6]);
            r->x_off = atoi(f[7]); r->raw_off = atoi(f[8]);
            r->id_asc = atoi(f[9]); r->w_by_eid_fixed = atoi(f[10]);
            r->rawk_base = atoi(f[11]); r->rawk_stride = atoi(f[12]);
            nr++;
        } else if (f[0][0] == 'B') {
            if (nf != MG_PLAN_FIELDS_B) { bad++; continue; }
            MgBlock *b = &p->blocks[nb];
            b->index = atoi(f[1]); b->is_mtp = atoi(f[2]); b->unit_base = atoi(f[3]);
            b->main_n = atoi(f[4]); b->sup_lo = atoi(f[5]); b->sup_hi = atoi(f[6]);
            b->case_lo = atoi(f[7]); b->case_hi = atoi(f[8]);
            b->n_expert_cases = atoi(f[9]);
            b->n_combine_cases = 0;   /* filled by verify from the case table */
            nb++;
        } else if (f[0][0] == 'C') {
            if (nf != MG_PLAN_FIELDS_C) { bad++; continue; }
            MgCase *c = &p->cases[nc];
            c->kind = atoi(f[1]); c->mode = atoi(f[2]); c->block = atoi(f[3]);
            c->row = atoi(f[4]); c->slot = atoi(f[5]);
            c->u_gate = atoi(f[6]); c->u_up = atoi(f[7]); c->u_down = atoi(f[8]);
            c->row_idx = atoi(f[9]); c->combine_src = atoi(f[10]);
            c->dst_off = atoi(f[11]); c->exp_off = atoi(f[12]); c->cmp_off = atoi(f[13]);
            c->expect = atoi(f[14]); c->n_floats = atoi(f[15]); c->group = atoi(f[16]);
            c->bad = 0; c->nan = 0;
            nc++;
        } else {
            bad++;
        }
    }
    fclose(fh);
    if (bad) return 1000 + bad;
    if (nu != want_units || nr != want_rows || nc != want_cases || nb != want_blocks) return 2000;
    *parsed_u = nu; *parsed_r = nr; *parsed_c = nc; *parsed_b = nb;
    for (int i = 0; i < nb; ++i) {
        int ne = 0, ncm = 0;
        for (int j = p->blocks[i].case_lo; j <= p->blocks[i].case_hi; ++j)
            { if (p->cases[j].kind == MG_CASE_EXPERT) ne++; else ncm++; }
        p->blocks[i].n_expert_cases = ne;
        p->blocks[i].n_combine_cases = ncm;
    }
    return 0;
}



/* ---- plan verification, host-only, run by BOTH arms before anything launches -------------------- *
 * A plan that cannot be honoured is refused before it can print a verdict. The checks are the ones a
 * silently-wrong plan would otherwise survive: an out-of-range slab read, a case writing past its
 * record, a mode paired with the wrong case kind, an empty case table (pitfalls/40 §"a check that can
 * produce no output cannot fail"). */
static inline int mg_verify_plan(const MgPlan *p, long long out_floats, long long expect_floats,
                                 int verbose) {
    int f = 0;
    if (p->n_units <= 0) { printf("FAIL plan: no units\n"); f++; }
    if (p->n_rows <= 0) { printf("FAIL plan: no rows\n"); f++; }
    if (p->n_cases <= 0) { printf("FAIL plan: no cases: a run that launches nothing cannot fail\n"); f++; }
    if (p->n_blocks <= 0) { printf("FAIL plan: no blocks\n"); f++; }
    int ncls[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < p->n_units; ++i) {
        const MgUnit *u = &p->units[i];
        if (!mg_kind_ok(u->kind)) { printf("FAIL plan: unit %d kind %d\n", i, u->kind); f++; }
        if (!mg_kind_ok(u->declared_kind)) { printf("FAIL plan: unit %d declared_kind %d\n", i, u->declared_kind); f++; }
        if (u->role < 0 || u->role >= MG_ROLE_COUNT) { printf("FAIL plan: unit %d role %d\n", i, u->role); f++; }
        if (u->block < 0 || u->block >= p->n_blocks) { printf("FAIL plan: unit %d block %d\n", i, u->block); f++; }
        if (u->n_blocks <= 0 || u->rows <= 0) { printf("FAIL plan: unit %d geometry %dx%d\n", i, u->n_blocks, u->rows); f++; }
        if (u->kind != u->declared_kind) { printf("FAIL plan: unit %d kind %d != declared %d\n", i, u->kind, u->declared_kind); f++; }
        const long long bb = mg_block_bytes(u->kind);
        if (u->slot_bytes != (int32_t)bb) {
            printf("FAIL plan: unit %d slot_bytes %d != block_bytes %lld (a sampled unit is a "
                   "contiguous run; the reader walks at slot_bytes)\n", i, u->slot_bytes, bb);
            f++;
        }
        if ((long long)u->src_off < 0 ||
            (long long)u->src_off + (long long)(u->n_blocks - 1) * u->slot_bytes + bb > (long long)p->slab_bytes) {
            printf("FAIL plan: unit %d slab read from %d runs past slab_bytes=%zu\n", i, u->src_off, p->slab_bytes); f++;
        }
        if (u->cls != 2 && (u->set_pos < 0 || u->set_pos >= u->main_n)) { printf("FAIL plan: unit %d set_pos %d outside main_n %d\n", i, u->set_pos, u->main_n); f++; }
    }
    for (int i = 0; i < p->n_rows; ++i) {
        const MgRow *r = &p->rows[i];
        if (r->fixture != 0 && r->fixture != 1) { printf("FAIL plan: row %d fixture %d\n", i, r->fixture); f++; }
        if (r->block < 0 || r->block >= p->n_blocks) { printf("FAIL plan: row %d block %d\n", i, r->block); f++; }
        const int offs[5] = { r->ids_off, r->wcast_off, r->wpre_off, r->x_off, r->raw_off };
        const int sizes[5] = { MG_K * 4, MG_K * 4, MG_K * 4, MG_H * 4, MG_K * MG_H * 4 };
        for (int k = 0; k < 5; ++k) {
            if (offs[k] < 0 || (offs[k] & 3) || (long long)offs[k] + sizes[k] > (long long)p->slab_bytes) {
                printf("FAIL plan: row %d slice %d at %d size %d out of the slab\n", i, k, offs[k], sizes[k]); f++;
            }
        }
    }
    for (int i = 0; i < p->n_blocks; ++i) {
        const MgBlock *b = &p->blocks[i];
        if (b->main_n < 0 || b->unit_base < 0 ||
            b->unit_base + (b->main_n - 1) * MG_ROLE_COUNT + (MG_ROLE_COUNT - 1) >= p->n_units) {
            printf("FAIL plan: block %d unit range [%d, +3*%d) does not fit %d units\n", i, b->unit_base, b->main_n, p->n_units); f++;
        }
        if (b->case_lo < 0 || b->case_hi >= p->n_cases || b->case_lo > b->case_hi) {
            printf("FAIL plan: block %d case range [%d,%d] outside %d cases\n", i, b->case_lo, b->case_hi, p->n_cases); f++;
        }
        if (b->sup_lo > b->sup_hi && !(b->sup_lo == 0 && b->sup_hi == -1)) { printf("FAIL plan: block %d support range [%d,%d]\n", i, b->sup_lo, b->sup_hi); f++; }
        for (int j = b->case_lo; j <= b->case_hi; ++j)
            if (p->cases[j].block != i) { printf("FAIL plan: case %d names block %d, not %d\n", j, p->cases[j].block, i); f++; }
    }
    for (int i = 0; i < p->n_cases; ++i) {
        const MgCase *c = &p->cases[i];
        if (c->kind != MG_CASE_EXPERT && c->kind != MG_CASE_COMBINE) { printf("FAIL plan: case %d kind %d\n", i, c->kind); f++; }
        if (c->mode < 0 || c->mode >= MG_MODE_COUNT) { printf("FAIL plan: case %d mode %d\n", i, c->mode); f++; }
        if (c->expect < 0 || c->expect >= 5) { printf("FAIL plan: case %d expect %d\n", i, c->expect); f++; }
        if (c->block < 0 || c->block >= p->n_blocks) { printf("FAIL plan: case %d block %d\n", i, c->block); f++; }
        if (c->row_idx < 0 || c->row_idx >= p->n_rows) { printf("FAIL plan: case %d row_idx %d\n", i, c->row_idx); f++; }
        const int want = (c->kind == MG_CASE_EXPERT) ? MG_EXPERT_FLOATS : MG_COMBINE_FLOATS;
        if (c->n_floats != want) { printf("FAIL plan: case %d n_floats %d != %d for kind %d\n", i, c->n_floats, want, c->kind); f++; }
        if ((long long)c->dst_off + c->n_floats > out_floats) { printf("FAIL plan: case %d writes [%d,+%d) past out_floats %lld\n", i, c->dst_off, c->n_floats, out_floats); f++; }
        if ((long long)c->exp_off + c->n_floats > expect_floats) { printf("FAIL plan: case %d expects [%d,+%d) past expect_floats %lld\n", i, c->exp_off, c->n_floats, expect_floats); f++; }
        if (c->cmp_off >= 0 && (long long)c->cmp_off + c->n_floats > out_floats) { printf("FAIL plan: case %d compares [%d,+%d) past out_floats\n", i, c->cmp_off, c->n_floats); f++; }
        if (c->group != c->block) { printf("FAIL plan: case %d group %d != block %d: the grouped launch is per block (prediction 2)\n", i, c->group, c->block); f++; }
        const int is_exp_mode = (c->mode >= MG_MODE_HALVES_SWAPPED && c->mode <= MG_MODE_Q6K_D_BYTE_0);
        const int is_com_mode = (c->mode >= MG_MODE_W_BY_EXPERT_ID && c->mode <= MG_MODE_F32_COMBINE);
        if ((is_exp_mode && c->kind != MG_CASE_EXPERT) || (is_com_mode && c->kind != MG_CASE_COMBINE)) {
            printf("FAIL plan: case %d pairs mode %d with case kind %d\n", i, c->mode, c->kind); f++;
        }
        for (int rl = 0; rl < MG_ROLE_COUNT; ++rl) {
            const int ui = (rl == MG_ROLE_GATE) ? c->u_gate : (rl == MG_ROLE_UP) ? c->u_up : c->u_down;
            if (ui < 0 || ui >= p->n_units) { printf("FAIL plan: case %d unit %d for role %d\n", i, ui, rl); f++; continue; }
            if (p->units[ui].role != rl) { printf("FAIL plan: case %d unit %d is role %d, not %d\n", i, ui, p->units[ui].role, rl); f++; }
            if (p->units[ui].block != c->block) { printf("FAIL plan: case %d unit %d is block %d, not %d\n", i, ui, p->units[ui].block, c->block); f++; }
            const int want_rows = (rl == MG_ROLE_DOWN) ? MG_H : MG_I;
            if (p->units[ui].rows != want_rows) { printf("FAIL plan: case %d unit %d covers %d rows, not %d\n", i, ui, p->units[ui].rows, want_rows); f++; }
        }
        if (c->mode == MG_MODE_CORRECT) ncls[c->expect]++;
    }
    if (verbose) {
        printf("PLANCHECK units=%d rows=%d cases=%d blocks=%d out_floats=%lld expect_floats=%lld "
               "match=%d mismatch=%d alt=%d immune=%d report=%d\n",
               p->n_units, p->n_rows, p->n_cases, p->n_blocks, out_floats, expect_floats,
               ncls[0], ncls[1], ncls[2], ncls[3], ncls[4]);
    }
    return f;
}

#endif /* MOE_GATHER_KERNELS_H */
