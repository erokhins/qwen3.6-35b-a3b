/* q4k_dequant_kernels.h — device-side block decoders for Q4_K and Q8_0.
 *
 * Compiled twice from ONE source, which is the whole point of keeping the decoders in a header:
 *   - `q4k_dequant.hip`  -> hipcc, gfx1151, the device implementation under test;
 *   - `host_probe.cpp`   -> g++, a host build of the SAME functions, so a device-vs-host
 *                           disagreement is attributable to the device compiler/ISA and not to two
 *                           spellings of the decoder (`pitfalls/40-gates-and-admissions.md`
 *                           §"A check whose reference side is derived from the thing under test").
 * The reference this is gated against is NOT here: it is `tools/gguf_reader.py`'s dequant, which
 * passed the audit selftest (40/40 synthetic blocks bit-exact, 6/6 must-catch, 0 false fires) and
 * the published-file injections 150/150 with 0 leaks
 * (`quantization/evidence/audit-selftest-2026-10-26.json`,
 * `quantization/evidence/audit-injections-2026-10-26.json`, 2026-10-26).
 *
 * ---------------------------------------------------------------------------------------------
 * Block layout, transcribed verbatim from `engines/llama.cpp/ggml/src/ggml-common.h` at the pinned
 * commit d327bf1f0780f87b8c01b367b61853b0c291d7c4 (tag b11869), fetched 2026-10-26. Field names,
 * field order and the `static_assert`s are the header's; only the GGML_EXTENSION /
 * GGML_COMMON_AGGR_S / GGML_COMMON_AGGR_U attribute macros are dropped (they exist for MSVC's
 * anonymous-struct-in-union and change no layout).
 *
 *   #define QK_K 256            // super-block size
 *   #define K_SCALE_SIZE 12     // scales+mins bytes, Q4_K/Q5_K
 *   #define QK8_0 32
 *   #define QK4_NL 32
 *
 *   block_q4_K   { union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
 *                  uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; }
 *      static_assert(sizeof == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2)   = 144 B / 256 elts
 *   block_q8_0   { ggml_half d; int8_t qs[QK8_0]; }
 *      static_assert(sizeof == sizeof(ggml_half) + QK8_0)                      = 34 B /  32 elts
 *   block_iq4_nl { ggml_half d; uint8_t qs[QK4_NL/2]; }
 *      static_assert(sizeof == sizeof(ggml_half) + QK4_NL/2)                   = 18 B /  32 elts
 *
 * So Q4_K is 144 B per 256 weights = 4.5 bpw and Q8_0 is 34 B per 32 = 8.5 bpw — the two widths
 * policy A ships (experts Q4_K, attention/GDN/shared/embeddings Q8_0: `quantization/README.md`,
 * decisions D-01/D-01a in `DECISIONS.md`). Q6_K is deliberately NOT implemented here: it is not
 * shaped like the K-quants (int8 scales, no min, a single `d` at byte 208 —
 * `tools/gguf_reader.py` §"Element order inside a super-block"), so a K-quant reader must not be
 * generalised across widths.
 *
 * ---------------------------------------------------------------------------------------------
 * Why the arithmetic below is plain f32 and still bit-identical to the Python reader
 *
 * `tools/gguf_reader.py` evaluates `d * sc * q - dmin * mg` in Python doubles. Every factor here is
 * exactly representable and every product is exact in f32, so the two routes cannot disagree:
 *   d, dmin   : f16 -> f32 widening is exact (11-bit significand);
 *   d * sc    : <= 11 + 6 = 17 significand bits -> exact in f32 (24 available);
 *   * q       : <= 17 + 4 = 21 bits              -> exact in f32;
 *   dmin * mg : <= 11 + 6 = 17 bits               -> exact in f32;
 *   Q8_0 d*q  : <= 11 + 8 = 19 bits               -> exact in f32.
 * The only rounding in either route is the final subtraction (Q8_0 has none at all), so each side
 * rounds once, identically. FMA contraction cannot change that: the product is already exact, so
 * `fma(scale, q, -mins)` and `RN(scale*q - mins)` are the same value. This was measured and not
 * assumed: over 4000 real Q4_K blocks (1,024,000 elements) sampled from blk.0.ffn_gate_exps.weight
 * and blk.1.ffn_up_exps.weight, an f32 evaluation of this expression differed from the reader's
 * double route rounded once to f32 in 0 elements (W6, 2026-10-26, `results.md`
 * §"Why f32 is exact"). A change that widens any factor — a Q6_K `d * sc` with 8-bit scales, a
 * fused bias, a bf16 staging type — breaks the exactness argument and must be re-gated, not trusted.
 */
#ifndef Q4KD_KERNELS_H
#define Q4KD_KERNELS_H

#include <stdint.h>
#include <string.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
/* __host__ __device__ so ONE source serves both arms: the kernel calls these from the device and
 * this file's own self-tests (and host_probe.cpp, built by g++) call the same bodies on the host. */
#define Q4KD_DEV __host__ __device__
#define Q4KD_GLOBAL __global__
#else
#define Q4KD_DEV
#define Q4KD_GLOBAL
#endif

/* ---- block geometry, from the pinned header's macros and static_asserts ---------------------- */
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32
#define QK4_NL 32

#define Q4KD_Q4K_BYTES 144          /* 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 */
#define Q4KD_Q8_0_BYTES 34          /* sizeof(ggml_half) + QK8_0                   */
#define Q4KD_Q4NL_BYTES 18          /* sizeof(ggml_half) + QK4_NL/2                */
#define Q4KD_MAX_BLOCK_BYTES 144    /* the widest block this file decodes          */
#define Q4KD_MAX_BLOCK_ELTS QK_K
#define Q4KD_NAN_BITS 0x7FC00000u   /* what an inapplicable (kind, mode) pair writes, so a leak is
                                       loud (NaN != anything) instead of silently "correct" */

typedef uint16_t ggml_half;         /* storage only: decoded by bit pattern in q4kd_f16() so the
                                       denormal path is ours, not the C library's, on either compiler */

typedef struct {
    union {
        struct { ggml_half d, dmin; };
        uint8_t dm_bytes[4];
    };
    uint8_t scales[K_SCALE_SIZE];   /* scales and mins, quantized with 6 bits */
    uint8_t qs[QK_K / 2];           /* 4-bit quants */
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2,
              "wrong q4_K block size/padding");

typedef struct {
    ggml_half d;                    /* delta */
    int8_t qs[QK8_0];               /* quants */
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL / 2,
              "wrong iq4_nl block size/padding");

enum { Q4KD_KIND_Q4K = 0, Q4KD_KIND_Q8_0 = 1, Q4KD_KIND_COUNT = 2 };

/* Each mode is a converter mistake, not a random perturbation. Modes 1-4 reproduce, on the device,
 * the misreadings the CPU audit gate holds in its INJECTIONS table
 * (`tools/audit_published_quant.py` §INJECTIONS); modes 5-9 are index arithmetic an engine can
 * actually get wrong. Only Q4_K units take modes 1-7 and only Q8_0 units take 8-9; anything else is
 * a plan bug and writes NaN. */
enum {
    Q4KD_MODE_CORRECT = 0,
    Q4KD_MODE_Q4K_AS_Q4NL = 1,        /* Q4_K read as 8 x 18-byte Q4_NL blocks: the 6-bit sub-block
                                         scales AND the min term are gone. 8*18 == 144, so it walks
                                         the same bytes with the wrong stride — silent at the
                                         container level. */
    Q4KD_MODE_Q4K_DROP_MIN = 2,       /* value = d*sc*q, the - dmin*mg term omitted */
    Q4KD_MODE_Q4K_NAIVE_SC_MIN = 3,   /* the 96 bits of scales[] read as 16 contiguous 6-bit fields
                                         instead of by get_scale_min_k4 */
    Q4KD_MODE_Q4K_SUBBLOCK_ROT3 = 4,  /* sub-block j is given sub-block (j+3)&7's scale AND min */
    Q4KD_MODE_Q4K_BLOCK_OFF_1 = 5,    /* the block index is off by one inside its tensor group */
    Q4KD_MODE_Q4K_BYTE_SHIFT_1 = 6,   /* the tensor data start is one byte late */
    Q4KD_MODE_Q4K_XPOSE_EXPERT = 7,   /* the [E,N,K] expert stack is indexed as if the expert axis
                                         were trailing (GGUF dims [K,N,E] read as [E,N,K]) */
    Q4KD_MODE_Q8_0_BLOCK_OFF_1 = 8,
    Q4KD_MODE_Q8_0_BYTE_SHIFT_1 = 9,
    Q4KD_MODE_COUNT = 10
};

/* What the predicate compares a case's output against. Every case carries exactly one of these and
 * the gate asserts the population of each class is non-zero (pitfalls/40: a counter that reads 0 is
 * vacuous until witnessed non-zero). */
enum {
    Q4KD_EXP_MATCH = 0,               /* must equal the reader's decode of these bytes, bit-for-bit */
    Q4KD_EXP_MISMATCH = 1,            /* must differ from the reader's correct decode somewhere */
    Q4KD_EXP_ALT = 2,                 /* must equal the reader's decode of the bytes this mode should
                                         have read — the attribution half: it proves the kernel made
                                         exactly this addressing error and not noise */
    Q4KD_EXP_IMMUNE = 3               /* must equal the correct decode: this mode cannot change this
                                         unit's values, by the format's own arithmetic (derived from
                                         the block bytes, never hand-listed) */
};

/* ---- the plan, as the device sees it ------------------------------------------------------------ */
/* One unit == one block of one tensor. Every position is derived from the GGUF's own tensor-info
 * record (dims, block index) by gate.py; nothing here is typed in by hand. */
typedef struct {
    int32_t kind;
    int32_t src_off;        /* byte offset of this block's first byte inside the slab */
    int32_t slot_bytes;     /* stride between slab slots = block_bytes + 1; the extra byte is a
                               DEFINED guard byte, so the byte-shift mode can read bb bytes past a
                               block's start without running off the end of the slab. gate.py fills
                               it with the byte that really follows this block in the file (the first
                               byte of block_index+1), so a misaligned read sees what a misaligned
                               read would see. */
    int32_t dst_off;        /* float offset of this unit's mode-0 output in `out` */
    int32_t n_elts;         /* elements this unit writes */
    int32_t n_blocks;       /* blocks in this unit (1 today; kept so a multi-block launch cannot
                               silently degrade to a 1-block launch — pitfalls/40) */
    int32_t block_bytes;    /* 144 or 34, from the derived block geometry */
    int32_t nb_total;       /* blocks in the whole tensor */
    int32_t nb_per_row;     /* dims[0] / block elts */
    int32_t n_rows;         /* dims[1] */
    int32_t n_outer;        /* dims[2] (the expert axis), or 1 for rank <= 2 */
    int32_t grp_u0, grp_u1; /* inclusive unit-array range of this unit's group, sorted by block_index */
    int32_t xpose_ok;       /* 1 for units the transposed-expert mode may be applied to */
    int32_t block_index;    /* block index within the tensor */
    int32_t tensor_index;   /* index in the GGUF tensor-info array */
    int32_t cls;            /* 0 = real bytes from the GGUF, 1 = synthetic, 2 = support block
                               (present in the slab so a transposed read has somewhere real to land;
                               gets a mode-0 case, never an injection case) */
} Q4kUnit;

typedef struct {
    int32_t unit;           /* index into units */
    int32_t mode;
    int32_t dst_off;        /* float offset this case writes */
    int32_t cmp_off;        /* float offset of the same unit's mode-0 output, or -1 */
} Q4kCase;

/* ---- element order, transcribed from the dequant loop ------------------------------------------ *
 * Q4_K: 8 sub-blocks of 32. Element e lives at block byte 16 + 32*(e>>6) + (e & 31), nibble
 * (e>>5)&1, and its scale/min pair is sub-block index 2*(e>>6) + ((e>>5)&1). "The 32-element chunk"
 * and "e>>5" are the same function here: chunk g's low nibbles are elements 64g..64g+31 (e>>5 == 2g)
 * and its high nibbles are elements 64g+32..64g+63 (e>>5 == 2g+1). That was verified against
 * tools/gguf_reader.py rather than argued from the docstring: with d=1.0, dmin=0, scales[1]=63,
 * every other scale/min 0, and nibbles at qs[32g] and qs[32g+1], the reader emits exactly one
 * non-zero element — index 33, value 63*15 = 945.0 — so the high-nibble half of chunk 0 is the one
 * carrying sc[1] (W6 probe 2026-10-26, `results.md` §"Which sub-block index the reader uses").
 * Hence one lane handles, for g = 0..3, the byte qs[32g + lane] and writes BOTH nibbles: the low
 * nibble to element 64g + lane (sub-block 2g), the high nibble to element 64g + 32 + lane
 * (sub-block 2g+1). Same 8 writes, same addresses, for every lane.
 * ----------------------------------------------------------------------------------------------- */

Q4KD_DEV static inline uint32_t q4kd_u16le(const uint8_t *p) {
    /* byte-wise on purpose: the byte-shift mode makes the read unaligned, and a reinterpreted
     * uint16_t* there is UB the compiler is free to exploit. */
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

Q4KD_DEV static inline float q4kd_bits_to_float(uint32_t fb) {
    /* A bit cast, spelled portably: the same 4-byte memcpy compiles to a register move on the GPU
     * and avoids leaning on whether __uint_as_float is marked host-callable in a given ROCm. */
    float f;
    memcpy(&f, &fb, sizeof(f));
    return f;
}

Q4KD_DEV static inline float q4kd_f16(uint32_t bits) {
    /* f16 -> f32 by bit pattern: rebias the exponent 15 -> 127, shift the mantissa 13. Denormals go
     * the long way (mantissa * 2^-24, exact) because a bit rebias is wrong for them.
     *
     * NaN is canonicalized to +/-0x7FC00000, payload dropped. That is what the oracle does —
     * tools/gguf_reader.py's f16() is struct.unpack('<e') and CPython collapses every NaN payload to
     * the canonical quiet NaN — and hardware conversion (what ggml's GGML_FP16_TO_FP32 ends up as)
     * quiets an sNaN too, so it only differs in bits nobody is supposed to look at. Keeping the
     * payload here would put a whole 32-element sub-block of the as_q4nl control row out of
     * agreement with the reader, so the rule is stated instead of discovered: a corrupt scale still
     * arrives as NaN, and the comparator stays a pure bit-pattern compare with no NaN exception. */
    const uint32_t sign = (bits >> 15) & 1u;
    const uint32_t rest = bits & 0x7FFFu;
    uint32_t fb;
    if ((rest & 0x7C00u) == 0u) {
        if (rest == 0u) {
            fb = sign << 31;
        } else {
            const float m = (float)rest;                       /* exact: rest < 1024 */
            const float scaled = m * 5.9604644775390625e-8f;   /* 2^-24, exact */
            return sign ? -scaled : scaled;
        }
    } else if ((rest & 0x7C00u) == 0x7C00u) {
        fb = (rest & 0x3FFu) ? ((sign << 31) | 0x7FC00000u) : ((sign << 31) | 0x7F800000u);
    } else {
        fb = (sign << 31) | ((((rest >> 10) & 0x1Fu) + 112u) << 23) | ((rest & 0x3FFu) << 13);
    }
    return q4kd_bits_to_float(fb);
}

/* get_scale_min_k4 (ggml-quants.c, the decoder route), transcribed:
 *   j < 4:   sc = q[j] & 63            m = q[j+4] & 63
 *   j >= 4:  sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
 *            m  = (q[j+4] >>  4) | ((q[j]   >> 6) << 4)
 */
Q4KD_DEV static inline void q4kd_scale_min(int j, const uint8_t *q, uint32_t *sc, uint32_t *mg) {
    if (j < 4) {
        *sc = (uint32_t)q[j] & 63u;
        *mg = (uint32_t)q[j + 4] & 63u;
    } else {
        *sc = ((uint32_t)q[j + 4] & 0xFu) | (((uint32_t)q[j - 4] >> 6) << 4);
        *mg = ((uint32_t)q[j + 4] >> 4) | (((uint32_t)q[j] >> 6) << 4);
    }
}

/* The WRONG 6-bit unpack: 16 contiguous 6-bit fields across the same 12 bytes. Plausible from the
 * header comment "scales and mins, quantized with 6 bits", and structurally different from the route
 * above — which is the point: a gate that cannot tell these two apart cannot tell any scale-order
 * bug apart. */
Q4KD_DEV static inline uint32_t q4kd_bits6(const uint8_t *q, int bit) {
    const uint32_t w = (uint32_t)q[bit >> 3] | ((uint32_t)q[(bit >> 3) + 1] << 8);
    return (w >> (bit & 7)) & 63u;
}

Q4KD_DEV static inline void q4kd_scale_min_naive(int j, const uint8_t *q,
                                                 uint32_t *sc, uint32_t *mg) {
    *sc = q4kd_bits6(q, 6 * j);
    *mg = q4kd_bits6(q, 6 * (j + 8));
}

Q4KD_DEV static inline uint32_t q4kd_as_float_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

/* The one equality rule this gate uses, in one place, shared by the device kernel's counters, the
 * host probe and (mirrored in Python) the gate's comparator.
 *
 * Bit-exact on everything finite — including the sign of a zero, which stays significant: +0.0 and
 * -0.0 are different words and the reader produces both, so folding them would hide a kernel that
 * flushed negative zeros. The single exception is NaN: the reader's route is Python double arithmetic
 * and CPython folds `nan * -1.0` into `-nan`, while an f32 multiply keeps the first operand's NaN.
 * Neither route *defines* the sign or payload of a NaN product, so the claim that can be gated is
 * "a NaN in, a NaN out", not "the same NaN bits out". That is a rule about what is comparable, not
 * a numeric band: no finite value is ever called equal to a different finite value.
 */
Q4KD_DEV static inline int q4kd_word_eq(uint32_t a, uint32_t b) {
    if (a == b) return 1;
    const uint32_t ma = a & 0x7FFFFFFFu, mb = b & 0x7FFFFFFFu;
    return ma > 0x7F800000u && mb > 0x7F800000u;
}

Q4KD_DEV static inline void q4kd_nan_fill(float *dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = q4kd_bits_to_float(Q4KD_NAN_BITS);
}

/* ---- the decoders ------------------------------------------------------------------------------- *
 * Each takes the block bytes and one lane index and writes that lane's share of the block. The
 * device kernel calls them from warp 0 into a shared-memory staging array, so the emitted element
 * order is unambiguous on both compilers; host_probe.cpp calls them for lane 0..31 in a loop.
 */

Q4KD_DEV static inline void q4k_decode_lane(const uint8_t *blk, int mode, int lane, float *dst) {
    if (mode >= Q4KD_MODE_Q8_0_BLOCK_OFF_1) {           /* Q8_0-only mode on a Q4_K unit */
        /* NaN the slots THIS lane owns, not the first 8 of the block: a lane's slots are scattered
         * at 64g + 32h + lane, and every element of every case must be written by every mode, or a
         * stale zero could read as a match against an all-zero reference. */
        for (int g = 0; g < QK_K / 64; ++g)
            for (int half = 0; half < 2; ++half)
                dst[64 * g + 32 * half + lane] = q4kd_bits_to_float(Q4KD_NAN_BITS);
        return;
    }
    if (mode == Q4KD_MODE_Q4K_AS_Q4NL) {                /* different format, same 144 bytes */
        const int l = lane & 15;
        const int h = lane >> 4;
        for (int i = 0; i < QK_K / QK4_NL; ++i) {
            const uint8_t *b = blk + Q4KD_Q4NL_BYTES * i;
            const float d = q4kd_f16(q4kd_u16le(b));
            const uint32_t q = ((uint32_t)b[2 + l] >> (4 * h)) & 15u;
            dst[QK4_NL * i + lane] = d * ((float)q - 8.0f);
        }
        return;
    }
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 4 + K_SCALE_SIZE;
    const float d = q4kd_f16(q4kd_u16le(blk + 0));
    const float dmin = q4kd_f16(q4kd_u16le(blk + 2));
    for (int g = 0; g < QK_K / 64; ++g) {
        const uint8_t byte = qs[32 * g + lane];
        for (int half = 0; half < 2; ++half) {
            const int j = 2 * g + half;
            const int js = (mode == Q4KD_MODE_Q4K_SUBBLOCK_ROT3) ? ((j + 3) & 7) : j;
            uint32_t sc, mg;
            if (mode == Q4KD_MODE_Q4K_NAIVE_SC_MIN) q4kd_scale_min_naive(js, scales, &sc, &mg);
            else q4kd_scale_min(js, scales, &sc, &mg);
            const uint32_t q = half ? ((uint32_t)byte >> 4) : ((uint32_t)byte & 15u);
            const float scale = d * (float)sc;
            const float mins = (mode == Q4KD_MODE_Q4K_DROP_MIN) ? 0.0f : dmin * (float)mg;
            dst[64 * g + 32 * half + lane] = scale * (float)q - mins;
        }
    }
}

Q4KD_DEV static inline void q8_0_decode_lane(const uint8_t *blk, int mode, int lane, float *dst) {
    /* Mode 0 is the correct path for BOTH kinds, so the leak guard is "a Q4_K-only injection", not
     * "any mode below the first Q8_0 one". Written as `mode <= Q4K_XPOSE_EXPERT` it NaN-filled the
     * control row too, which the driver's q8_0 hand-check caught on the first build. */
    if (mode > Q4KD_MODE_CORRECT && mode < Q4KD_MODE_Q8_0_BLOCK_OFF_1) {
        dst[lane] = q4kd_bits_to_float(Q4KD_NAN_BITS);
        return;
    }
    const float d = q4kd_f16(q4kd_u16le(blk));
    dst[lane] = d * (float)((int8_t)blk[2 + lane]);
}

/* Decode a whole block with all 32 "lanes". The device kernel calls the lane functions from warp 0
 * into shared memory; the host build and the driver's self-tests call this. Same writes, same
 * addresses either way — which is the point of putting the decoders in a header. */
Q4KD_DEV static inline void q4kd_decode_block(const uint8_t *blk, int kind, int mode, float *dst) {
    for (int lane = 0; lane < 32; ++lane) {
        if (kind == Q4KD_KIND_Q4K) q4k_decode_lane(blk, mode, lane, dst);
        else q8_0_decode_lane(blk, mode, lane, dst);
    }
}

/* ---- the addressing the injection modes apply --------------------------------------------------- *
 * All of it is char-offset arithmetic scaled by block_bytes, never a typed-pointer offset
 * (pitfalls/40: `ptr + slot * bytes` on a typed pointer is a scaled offset and only bites at slot
 * >= 1). The neighbour rule for the off-by-one modes wraps inside the tensor's own sampled group, so
 * a wrong index stays inside the tensor it came from — and gate.py implements the same rule
 * independently to build the alternative expectation, so the two have to agree.
 *
 * Returns 0 and writes *src_off when the mode applies to this unit; returns 1 when it does not, in
 * which case the caller writes NaN instead of a number (a leak must be loud). */
Q4KD_DEV static inline int q4kd_resolve_src(const Q4kUnit *units, int me, const Q4kUnit *u,
                                            int mode, int32_t *src_off) {
    const int q4 = (u->kind == Q4KD_KIND_Q4K);
    *src_off = u->src_off;
    if (mode == Q4KD_MODE_CORRECT) return 0;
    if (q4 ? (mode >= Q4KD_MODE_Q8_0_BLOCK_OFF_1) : (mode < Q4KD_MODE_Q8_0_BLOCK_OFF_1)) {
        return 1;                                              /* wrong kind for this mode */
    }
    if (mode == Q4KD_MODE_Q4K_BLOCK_OFF_1 || mode == Q4KD_MODE_Q8_0_BLOCK_OFF_1) {
        /* the next sampled block of the same tensor, wrapping inside the group */
        const int32_t idx = (me >= u->grp_u1) ? u->grp_u0 : (me + 1);
        *src_off = units[idx].src_off;
    } else if (mode == Q4KD_MODE_Q4K_BYTE_SHIFT_1 || mode == Q4KD_MODE_Q8_0_BYTE_SHIFT_1) {
        *src_off = u->src_off + 1;                             /* slab slots are bb+1 bytes wide */
    } else if (mode == Q4KD_MODE_Q4K_XPOSE_EXPERT) {
        if (!u->xpose_ok) return 1;
        const int32_t b = u->block_index;
        const int32_t j = b % u->nb_per_row;
        const int32_t t = b / u->nb_per_row;
        const int32_t n = t % u->n_rows;
        const int32_t e = t / u->n_rows;
        /* correct: b = (e*n_rows + n)*nb_per_row + j, i.e. GGUF dims [K,N,E] with K fastest.
         * transposed: the same (e,n,j) addressed as if the expert axis were trailing,
         * b' = (j*n_rows + n)*n_outer + e. Everything below is the device's own arithmetic; the
         * binary search resolves the permuted index to a slab slot, and a miss is NaN (loud), which
         * is why gate.py closes the sample set under this map before it ever emits the mode. */
        if (u->n_outer <= 1 || u->nb_per_row <= 0 || u->n_rows <= 0 ||
            (int64_t)u->nb_per_row * u->n_rows * u->n_outer != (int64_t)u->nb_total) return 1;
        const int32_t bp = (j * u->n_rows + n) * u->n_outer + e;
        if (bp < 0 || bp >= u->nb_total) return 1;
        int lo = u->grp_u0, hi = u->grp_u1, found = -1;
        while (lo <= hi) {
            const int mid = (lo + hi) >> 1;
            if (units[mid].block_index == bp) { found = mid; break; }
            if (units[mid].block_index < bp) lo = mid + 1; else hi = mid - 1;
        }
        if (found < 0) return 1;
        *src_off = units[found].src_off;
    }
    return 0;
}

#endif /* Q4KD_KERNELS_H */
