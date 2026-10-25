/* lm_head_kernels.h — the un-tied output projection, one source compiled twice.
 *
 * README.md prediction 5: `lm_head` is **untied** (`tie_word_embeddings: false`, `vocab_size`
 * 248320, `hidden_size` 2048 — read out of `checkpoints/Qwen3.6-35B-A3B.config.json` by gate.py,
 * never typed here) so the projection is a separate 1,017,118,720 B = 0.9473 GiB bf16 read every
 * token, a fifth of the per-token traffic (`README.md` §the weight census). The question this lane
 * answers is the one that prediction names: the row-batching idea prototyped elsewhere for a *tied*
 * table must be re-validated against an untied one, "where the row-gather and the full GEMV read
 * different memory".
 *
 * COMPILED TWICE FROM ONE SOURCE, which is the point of keeping the arithmetic in a header:
 *   - `lm_head.hip`     -> hipcc, gfx1151, the device implementation under test;
 *   - `host_probe.cpp`  -> g++, a host build of the SAME functions.
 * A device/host disagreement is then attributable to the device compiler and not to two spellings of
 * the dot. The reference is NOT here: it is torch on CPU (`oracle_arm.py`, no shared code), plus
 * `tools/gguf_reader.py` for the Q8_0 bytes (`pitfalls/40-gates-and-admissions.md` §"A check whose
 * reference side is derived from the thing under test").
 *
 * ---------------------------------------------------------------------------------------------
 * What is bit-exact here, and what is not — said before the code, because it decides the judge
 * ---------------------------------------------------------------------------------------------
 * Products are exact; sums are not.
 *   bf16 x bf16 -> f32 : 8 + 8 = 16 significand bits, exact in f32's 24.
 *   Q8_0 d * q -> f32  : f16 d has 11, |q| <= 127 has 8, so 11 + 8 = 19 bits, exact.
 *   (The same argument `prototype/q4k_dequant/q4k_dequant_kernels.h` §"Why the arithmetic below is
 *    plain f32" makes for the dequant, and the same reason `tools/gguf_reader.py`'s Python-double
 *    route can be compared after ONE rounding to f32.)
 * FMA contraction is therefore neutral: `fma(x,y,acc)` and `RN(RN(x*y)+acc)` are the same value when
 * `x*y` is exact, so hipcc and g++ may contract or not without moving a bit.
 *
 * What is NOT exact is the K=2048-long accumulation, and no amount of shared source makes two
 * association orders agree: `pitfalls/30-numerics.md` §"A max-ULP band cannot see a whole-tensor
 * 1-ULP error" states that "max 1 ULP against the dump" is not achievable for a long dot. So:
 *
 *   bit-exact, zero tolerance, because they are the SAME association (LMH_ASSOC_SEQ):
 *       device vs host twin;  batched vs per-row;  gather vs dense
 *   band-judged, because they are legal RE-ASSOCIATIONS of the same terms:
 *       LMH_ASSOC_REV, LMH_ASSOC_SPLIT8, and every arm against the torch oracle.
 *
 * The band is computed at run time from the data as gamma_K * sum_k |h_k|*|w_k| with K from the
 * config and gamma_K = K*u/(1-K*u), u = 2^-24; it is witnessed non-vacuous by a floor probe that IS
 * this file's own re-associations, and by requiring every must-fire injection's predicted delta to
 * exceed it. There is no typed tolerance anywhere in this lane (`AGENTS.md` §"Never loosen a gate to
 * make it pass"); see `results.md` §"Where every number comes from".
 */
#ifndef LM_HEAD_KERNELS_H
#define LM_HEAD_KERNELS_H

#include <stdint.h>
#include <string.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
/* One source, both arms. The atomics sit behind the LMH_ACC_* macros so the g++ build of these
 * bodies is the ordinary non-atomic code a single-threaded host driver wants and the hipcc build
 * gets real atomics — one spelling of the ARITHMETIC either way.
 *
 * The guard is __HIP_DEVICE_COMPILE__, not __HIPCC__: hipcc compiles this translation unit TWICE
 * (a host pass and a device pass) and a __host__ __device__ function that names atomicAdd in the
 * host pass is a hard error ("reference to __device__ function 'atomicAdd' in __host__ __device__
 * function"), which is exactly what the host runner in lm_head_plan.h needs to compile inside the
 * device binary. Which branch each pass took is witnessed, not assumed —
 * tmp/worker-lm-head/probe6_guard.hip prints BRANCH_HIP_DEVICE for the gfx1151 pass and BRANCH_PLAIN
 * for the host pass (2026-10-26). */
#define LMH_DEV __host__ __device__
#define LMH_GLOBAL __global__
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define LMH_ACC_ADD64(p, v) atomicAdd((unsigned long long *)(p), (unsigned long long)(v))
#define LMH_ACC_ADD32(p, v) atomicAdd((unsigned int *)(p), (unsigned int)(v))
#define LMH_ACC_XOR32(p, v) atomicXor((unsigned int *)(p), (unsigned int)(v))
#define LMH_ACC_MIN(p, v) atomicMin((unsigned int *)(p), (unsigned int)(v))
#define LMH_ACC_MAX(p, v) atomicMax((unsigned int *)(p), (unsigned int)(v))
#else
#define LMH_ACC_ADD64(p, v) (*(unsigned long long *)(p) += (unsigned long long)(v))
#define LMH_ACC_ADD32(p, v) (*(unsigned int *)(p) += (unsigned int)(v))
#define LMH_ACC_XOR32(p, v) (*(unsigned int *)(p) ^= (unsigned int)(v))
LMH_DEV static inline void lmh_host_min(unsigned int *p, unsigned int v) { if (v < *p) *p = v; }
LMH_DEV static inline void lmh_host_max(unsigned int *p, unsigned int v) { if (v > *p) *p = v; }
#define LMH_ACC_MIN(p, v) lmh_host_min((unsigned int *)(p), (unsigned int)(v))
#define LMH_ACC_MAX(p, v) lmh_host_max((unsigned int *)(p), (unsigned int)(v))
#endif
#else
#define LMH_DEV
#define LMH_GLOBAL
#define LMH_ACC_ADD64(p, v) (*(unsigned long long *)(p) += (unsigned long long)(v))
#define LMH_ACC_ADD32(p, v) (*(unsigned int *)(p) += (unsigned int)(v))
#define LMH_ACC_XOR32(p, v) (*(unsigned int *)(p) ^= (unsigned int)(v))
LMH_DEV static inline void lmh_host_min(unsigned int *p, unsigned int v) { if (v < *p) *p = v; }
LMH_DEV static inline void lmh_host_max(unsigned int *p, unsigned int v) { if (v > *p) *p = v; }
#define LMH_ACC_MIN(p, v) lmh_host_min((unsigned int *)(p), (unsigned int)(v))
#define LMH_ACC_MAX(p, v) lmh_host_max((unsigned int *)(p), (unsigned int)(v))
#endif

/* ---- format geometry, from the pinned ggml-common.h struct ------------------------------------ *
 * Q8_0 is 34 B per 32 weights. The bpw that implies (8.5) is DERIVED by gate.py from
 * `tools/ggml_block_geom.py` and cross-checked by `tools/gguf_width_table.py`'s file-size witness;
 * no bpw is typed in this lane, and `gate.py` greps its own sources for one. Q8_0 is the width
 * policy A / `DECISIONS.md` D-10 assigns to `lm_head` and to the embeddings — the published
 * artifact's own width for `output.weight` is Q6_K, which gate.py reads, reports and cites rather
 * than assuming away (`quantization/README.md` §the D-10 caveat, `DECISIONS.md` D-10 note). */
#define LMH_Q8_0_ELTS 32
#define LMH_Q8_0_BYTES 34
typedef struct {
    uint16_t d;                   /* ggml_half delta, read by bit pattern, not by the C library */
    int8_t qs[LMH_Q8_0_ELTS];
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + LMH_Q8_0_ELTS,
              "wrong q8_0 block size/padding");
#define LMH_BF16_BYTES 2

#define LMH_NAN_BITS 0x7FC00000u  /* what an inapplicable (mode, encoding, path) pair writes, so a
                                     leak is loud (NaN != anything) instead of a stale zero matching
                                     a zero reference */
#define LMH_PAD_BYTE 0xCDu        /* fills the gaps between tables in the slab. Not a poison test:
                                     real bf16 bytes can legitimately be 0xCD, so the instrument that
                                     adjudicates "did this walk leave its table" is the address-extent
                                     counter `cross`, not a byte value. The pad exists so a stray
                                     read cannot silently land in another table's data. */

/* ---- the enums. gate.py parses these bodies and refuses to run if its own copy disagrees, so a
   renumbered mode cannot quietly make the gate judge a different binary. ----------------------- */
enum { LMH_ENC_BF16 = 0, LMH_ENC_Q8_0 = 1, LMH_ENC_COUNT = 2 };

/* Which tensor a table's bytes came from. ROLE_TIED_* is the 0.8B-shaped CONTROL: both roles point
 * at the SAME bytes, which is precisely the assumption the batching precedent was validated on. */
enum {
    LMH_ROLE_LMHEAD = 0,          /* real bf16 rows of lm_head.weight          (shard 00026) */
    LMH_ROLE_EMBD = 1,            /* real bf16 rows of embed_tokens.weight     (shard 00001) */
    LMH_ROLE_LMHEAD_Q8 = 2,       /* real Q8_0 bytes at [vocab,hidden] geometry (token_embd.weight) */
    LMH_ROLE_TIED_LMHEAD = 3,     /* control: the same bytes as LMH_ROLE_TIED_EMBD */
    LMH_ROLE_TIED_EMBD = 4,       /* control: the bytes those two roles share */
    LMH_ROLE_SYNTH_DUP = 5,       /* synthetic, contains an adjacent byte-identical row pair */
    LMH_ROLE_COUNT = 6
};

/* Row selection. DENSE walks a contiguous row range (the full-GEMV shape); GATHER reads the rows
 * named by an index list (the MTP / speculative-scoring shape). Batching is orthogonal: a BATCHED
 * case covers bcount hidden vectors in ONE walk of the rows, a PERROW case walks them alone, and
 * the B PERROW cases sharing a grp are the un-batched control for one BATCHED case. */
enum { LMH_PATH_DENSE = 0, LMH_PATH_GATHER = 1, LMH_PATH_COUNT = 2 };
enum { LMH_BATCH_BATCHED = 0, LMH_BATCH_PERROW = 1, LMH_BATCH_COUNT = 2 };

/* Association order of the K-long sum. SEQ is the reference association inside which every
 * bit-exact claim is made; REV and SPLIT8 are legal re-associations, judged by the derived band. */
enum { LMH_ASSOC_SEQ = 0, LMH_ASSOC_REV = 1, LMH_ASSOC_SPLIT8 = 2, LMH_ASSOC_COUNT = 3 };
#define LMH_SPLIT 8
#define LMH_MAX_BATCH 8          /* bcount is bounded by this; both drivers assert it */

/* Each mode is a mistake an engine can actually make on THIS tensor, not a random perturbation.
 * `class` is none | addr | value | bytes; gate.py holds the same table with the encoding/path
 * applicability and the gate asserts the two agree, so a mode cannot be quietly applied to a case
 * it does not belong to (an inapplicable pair writes LMH_NAN_BITS and goes red loudly). */
enum {
    LMH_MODE_CORRECT = 0,
    LMH_MODE_GATHER_ROW_OFF_1 = 1,       /* addr  | gather reads ids[i]+1. Immunity is derived from
                                            the BYTES: it cannot fire on a row whose successor is
                                            byte-identical, and LMH_ROLE_SYNTH_DUP supplies one. */
    LMH_MODE_BATCH_SLOT_STRIDE = 2,      /* addr  | batch b is written at out + b*h instead of
                                            out + b*n_out. Fires only when bcount > 1 and h != n_out;
                                            structurally immune at bcount == 1. This is the
                                            "geometry that grows from 1 to N needs a case reaching
                                            slot 1" trap, adjudicated on both sides. */
    LMH_MODE_TIED_ALIAS = 3,             /* addr  | lm_head rows read out of the embedding table:
                                            the tied-table assumption applied to an untied model.
                                            Must fire on every untied case and be structurally
                                            immune on the tied control, which IS one buffer. */
    LMH_MODE_ROW_STRIDE_HEAD_DIM = 4,    /* addr  | row stride taken as head_dim (256) instead of
                                            hidden_size (2048). Immune for a hidden vector that is
                                            zero on k in [head_dim, hidden_size). */
    LMH_MODE_HALF_ACCUM = 5,             /* value | every partial is rounded to bf16. Immune where
                                            each product is already a bf16 value — a one-hot hidden
                                            row: the product IS the weight. */
    LMH_MODE_DROP_LAST_K = 6,            /* value | the last term of the slice is omitted. Immune
                                            where that term's hidden element is 0. */
    LMH_MODE_Q8_DROP_SCALE = 7,          /* value | Q8_0 q multiplied without its d. Q8_0 only. */
    LMH_MODE_CROSS_TABLE_READ = 8,       /* bytes | read the PEER table at the same row. On the tied
                                            control the output AND both byte counters are unchanged,
                                            because the peer is the same bytes — that pair is what
                                            makes the disjointness instrument demonstrable. */
    LMH_MODE_BATCH_RELOOPS = 9,          /* bytes | the batched walk re-reads the whole row set once
                                            per batch row. Outputs bit-identical, bytes_addressed
                                            bcount x. This is the mode a byte trace alone can see —
                                            which is why the lane traces addresses and not merely
                                            arithmetic: a "batched" kernel that keeps the per-row
                                            loop inside it passes every numeric gate. */
    LMH_MODE_COUNT = 10
};

/* What a case is judged against. Every case carries exactly one, and gate.py requires every class
 * to have a non-zero population (`pitfalls/00-measurement.md` §"A counter that reads exactly 0 is
 * vacuous"). */
enum {
    LMH_EXP_EXACT = 0,            /* bit-identical to the words at cmp_off (same association) */
    LMH_EXP_BAND = 1,             /* inside the derived band of the torch oracle */
    LMH_EXP_FIRES = 2,            /* an injection: must move >=1 element beyond the band */
    LMH_EXP_IMMUNE = 3,           /* an injection: must leave every element bit-identical, by
                                     geometry or by the bytes — derived by gate.py, never typed */
    LMH_EXP_BYTES = 4,            /* judged on the read-set counters, not on the outputs */
    LMH_EXP_COUNT = 5
};

/* ---- the plan, as both drivers see it ---------------------------------------------------------- *
 * Every position here is written by gate.py out of the pinned config, the safetensors headers and
 * the GGUF's own tensor-info records. Nothing in this lane hand-lists a row, a block or a width. */
typedef struct {
    int32_t idx;
    int32_t enc;                  /* LMH_ENC_* */
    int32_t role;                 /* LMH_ROLE_* */
    long long base;                 /* byte offset of this table's first byte inside the slab */
    long long nbytes;               /* extent: a load outside [base, base+nbytes) is a CROSS read */
    long long v_rows;               /* rows in this table (vocab_size for the real ones) */
    int32_t h;                    /* elements per row (hidden_size) */
    int32_t peer;                 /* table index the untied/tied peer read goes to, or -1 */
    int32_t real;                 /* 1 = bytes came from a file, 0 = synthetic */
} LmhTable;

typedef struct {
    int32_t idx;
    int32_t table;                /* index into tables */
    int32_t path;                 /* LMH_PATH_* */
    int32_t batching;             /* LMH_BATCH_* */
    int32_t b0, bcount;           /* slice of the hidden-vector slab this case consumes */
    int32_t row_lo, nrows;        /* DENSE: contiguous rows [row_lo, row_lo+nrows) */
    int32_t ids_off, n_ids;       /* GATHER: index list in the ids slab */
    int32_t assoc;                /* LMH_ASSOC_* */
    int32_t mode;                 /* LMH_MODE_* */
    int32_t out_off;              /* float offset this case writes: out[out_off + b*slot + j] */
    int32_t n_out;                /* rows this case emits: nrows (DENSE) or n_ids (GATHER) */
    int32_t cmp_off;              /* float offset of the words this case must equal, or -1 */
    int32_t grp;                  /* BATCHED case <-> the PERROW cases that are its control */
    int32_t expect;               /* LMH_EXP_* */
    int32_t on_device;            /* 0 = host-twin-only case (the full-vocab amortisation pair) */
    int32_t trace;                /* 1 = the read-set counters are judged for this case */
} LmhCase;

/* The per-case read-set record. Everything in it is ORDER-FREE (sums, an XOR, a min, a max), so the
 * device value cannot depend on which lane got which term or on the block schedule. That is what
 * lets the host twin and the device be compared bit-for-bit on BYTES TOUCHED and not just on
 * arithmetic (`pitfalls/20-rates-and-occupancy.md` §"A device selftest must verify bytes touched,
 * not arithmetic"). */
typedef struct {
    unsigned long long bytes;               /* weight bytes addressed, summed over lanes */
    unsigned long long loads;               /* ranges the tracer was called with */
    unsigned int xorb;            /* XOR of the f32 bit pattern of every weight element loaded */
    unsigned long long vsum;      /* INTEGER SUM of the same bit patterns: a second, non-XOR value
                                     digest, because a uniform power-of-two rescale cancels in an XOR
                                     over an even-count set (see lmh_trace_value) */
    unsigned long long bsum;                /* sum of the raw storage byte values loaded */
    unsigned int min_off;             /* lowest byte offset relative to the table base (init 0xFFFFFFFF) */
    unsigned int max_off;             /* highest byte offset + width relative to base (init 0) */
    unsigned int cross;               /* ranges that fell outside [base, base+nbytes) */
    unsigned int peer_hits;           /* ranges that fell inside the PEER table's extent */
    unsigned int lanes;               /* contributing (case, item, lane) triples */
    unsigned int bover;               /* writes that would have left `out` (must be 0; see results.md) */
} LmhTrace;

/* ---- tiny portable bits ------------------------------------------------------------------------ */
LMH_DEV static inline uint32_t lmh_u16le(const uint8_t *p) {
    /* byte-wise on purpose: an injected stride makes some of these reads unaligned, and a
     * reinterpreted uint16_t* there is UB the compiler may exploit. */
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

LMH_DEV static inline float lmh_bits_to_float(uint32_t fb) {
    float f;
    memcpy(&f, &fb, sizeof(f));
    return f;
}

LMH_DEV static inline uint32_t lmh_float_to_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

LMH_DEV static inline float lmh_bf16_to_float(uint32_t bits) {
    /* bf16 -> f32 is a left shift by 16 with NO rebias: same 8-bit exponent field, same NaN-ness.
     * Byte-for-byte the route `tools/gguf_reader.py` §bf16_to_f32_bytes takes (LE bytes [0,0,lo,hi]),
     * so the weights this file widens and the hidden vectors gate.py widened with the reader cannot
     * disagree. */
    return lmh_bits_to_float(bits << 16);
}

LMH_DEV static inline uint16_t lmh_float_to_bf16_bits(float f) {
    /* round-to-nearest-even over the discarded low 16 bits, which is what a bf16 store does. Used
     * ONLY by the LMH_MODE_HALF_ACCUM saboteur, whose whole point is that it rounds; never on a path
     * whose claim is bit-exactness. */
    const uint32_t b = lmh_float_to_bits(f);
    const uint32_t half = b & 0xFFFFu;
    const uint32_t lsb = (b >> 16) & 1u;
    const uint32_t rnd = (half > 0x8000u || (half == 0x8000u && lsb != 0u)) ? 1u : 0u;
    return (uint16_t)((b >> 16) + rnd);
}

LMH_DEV static inline float lmh_f16_to_float(uint32_t bits) {
    /* f16 -> f32 by bit pattern (rebias 15 -> 127, mantissa << 13), denormals the long exact way,
     * NaN canonicalised to +/-0x7FC00000 with the payload dropped. The canonicalisation is not local
     * colour: `tools/gguf_reader.py`'s f16() is struct.unpack('<e') and CPython collapses every
     * payload to the canonical quiet NaN, so keeping the payload here would put this file out of
     * agreement with the audited reader. Same rule, same reason as
     * `prototype/q4k_dequant/q4k_dequant_kernels.h` §q4kd_f16. */
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
    return lmh_bits_to_float(fb);
}

/* The one equality rule the lane uses, in one place, mirrored by gate.py's comparator.
 * Bit-exact on every finite value INCLUDING the sign of a zero; the single exception is NaN, whose
 * sign and payload neither route defines (the reader multiplies in Python doubles, an f32 multiply
 * keeps the first operand's sign). A rule about what is comparable, not a band: no finite value is
 * ever called equal to a different finite value. */
LMH_DEV static inline int lmh_word_eq(uint32_t a, uint32_t b) {
    if (a == b) return 1;
    const uint32_t ma = a & 0x7FFFFFFFu, mb = b & 0x7FFFFFFFu;
    return ma > 0x7F800000u && mb > 0x7F800000u;
}

/* ---- weight addressing, one spelling shared by the tracer and the dots ------------------------- *
 * Row offsets are computed on long long, never on a typed pointer: `w + row * h` on a uint16_t* is a
 * SCALED offset and is exactly how a one-row gate passes while the second slot SIGSEGVs
 * (`pitfalls/40-gates-and-admissions.md` §"A geometry that grows from 1 to N slots needs a case that
 * reaches slot >= 1"). `h_used` is the stride the caller believes the row has: normally the table's
 * own h, and LMH_MODE_ROW_STRIDE_HEAD_DIM makes the kernel pass head_dim instead. */
LMH_DEV static inline const uint8_t *lmh_row_ptr(const LmhTable *t, const uint8_t *slab,
                                                 long long row, int h_used, long long *nbytes) {
    /* The SAME stride is used for the base and for the extent, because that is what the bug does:
     * a kernel that took the row length from the wrong config key believes the table is
     * [v_rows, head_dim] and lays its rows out that way. It stays inside the table for
     * row < v_rows*h_used/h, which is why this injection cannot fault the context. */
    if (t->enc == LMH_ENC_BF16) {
        *nbytes = (long long)h_used * LMH_BF16_BYTES;
        return slab + t->base + row * ((long long)h_used * LMH_BF16_BYTES);
    }
    *nbytes = ((long long)h_used / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES;
    return slab + t->base + (row * ((long long)h_used / LMH_Q8_0_ELTS)) * LMH_Q8_0_BYTES;
}

/* The byte range one lane really reads, so `bytes` is the addressed weight traffic and NOT a count
 * inflated by how many lanes the work was cut into. Q8_0 lanes must break on block boundaries
 * (32 elements), which both drivers assert. */
LMH_DEV static inline const uint8_t *lmh_lane_ptr(const LmhTable *t, const uint8_t *w, int k0,
                                                  int kn, long long *nbytes) {
    if (t->enc == LMH_ENC_BF16) {
        *nbytes = (long long)kn * LMH_BF16_BYTES;
        return w + (long long)k0 * LMH_BF16_BYTES;
    }
    *nbytes = ((long long)kn / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES;
    return w + ((long long)k0 / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES;
}

/* ---- the read-set tracer ----------------------------------------------------------------------- *
 * Called once per (case, item, lane) with the address range that lane actually walked, and fed a
 * per-element digest from inside the dot loop. Every combine here is commutative and associative, so
 * the recorded value is a property of the BYTES TOUCHED and not of the schedule — which is what makes
 * a host-vs-device comparison of these fields a check instead of a coin flip. */
LMH_DEV static inline void lmh_trace_range(LmhTrace *tr, const uint8_t *p, long long n,
                                           const uint8_t *tbase, long long tnbytes,
                                           const uint8_t *pbase, long long pnbytes) {
    if (n <= 0) return;
    LMH_ACC_ADD64(&tr->bytes, (unsigned long long)n);
    LMH_ACC_ADD64(&tr->loads, 1u);
    /* `lanes` is NOT incremented here: it counts (case, item, lane) triples and is the caller's
     * bookkeeping, so LMH_MODE_BATCH_RELOOPS — which calls this function bcount times for one item —
     * must move bytes and loads while leaving lanes alone. That asymmetry is the injection's whole
     * signature, and a counter incremented in both places would erase it. */
    const long long off = (const char *)p - (const char *)tbase;
    if (off < 0 || off + n > tnbytes) {
        LMH_ACC_ADD32(&tr->cross, 1u);
    } else {
        LMH_ACC_MIN(&tr->min_off, (uint32_t)off);
        LMH_ACC_MAX(&tr->max_off, (uint32_t)(off + n));
    }
    if (pbase && pnbytes > 0) {
        const long long po = (const char *)p - (const char *)pbase;
        if (po >= 0 && po + n <= pnbytes) LMH_ACC_ADD32(&tr->peer_hits, 1u);
    }
    for (long long i = 0; i < n; ++i) LMH_ACC_ADD64(&tr->bsum, (uint64_t)p[i]);
}

LMH_DEV static inline void lmh_trace_value(LmhTrace *tr, uint32_t value_bits) {
    LMH_ACC_XOR32(&tr->xorb, value_bits);
    /* TWO order-free value digests, not one, because XOR alone provably misses a class of value
     * bugs: scaling every decoded weight by a power of two flips the same exponent bit in every
     * element, so the XOR over an even-count set is UNCHANGED. Measured on this file's own fixture
     * (tmp/worker-lm-head/probe9_literals.cpp, 2026-10-26): d = 0.5 and d = 1.0 over four nonzero
     * quants both give xorb = 0xFF400000. A plain integer sum has no such blind spot, so a value
     * injection must move vsum even when xorb is allowed to coincide. `pitfalls/40`: a counter that
     * cannot move is a decoration, and a digest whose blind spot is undocumented is a lie. */
    LMH_ACC_ADD64(&tr->vsum, (unsigned long long)value_bits);
}

/* ---- the dots ---------------------------------------------------------------------------------- *
 * One output element = one dot = one lane. The accumulation order is WRITTEN OUT, not left to the
 * compiler: a plain forward loop for SEQ, a plain backward loop for REV, and LMH_SPLIT sequential
 * slices joined by lmh_combine's fixed order for SPLIT8. Neither compiler may reassociate a float
 * chain without -ffast-math (and neither build passes it), so both builds keep what the source says.
 */
LMH_DEV static inline float lmh_bf16_term(const uint8_t *w, int k) {
    return lmh_bf16_to_float(lmh_u16le(w + 2 * (long long)k));
}

LMH_DEV static inline float lmh_q8_term(const uint8_t *w, int k, int mode) {
    /* w points at the row's first block. The block index inside the row is k/32 and the element
     * index inside the block is k%32, on long long for the same reason lmh_row_ptr is. */
    const long long e = (long long)k;
    const uint8_t *b = w + (e / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES;
    const float d = (mode == LMH_MODE_Q8_DROP_SCALE) ? 1.0f
                                                     : lmh_f16_to_float(lmh_u16le(b));
    const int q = (int)(signed char)b[2 + (e % LMH_Q8_0_ELTS)];
    return d * (float)q;
}

LMH_DEV static inline float lmh_term(const LmhTable *t, const uint8_t *w, int k, int mode) {
    return t->enc == LMH_ENC_BF16 ? lmh_bf16_term(w, k) : lmh_q8_term(w, k, mode);
}

/* One lane's slice of one vocab row, for ALL bcount batch rows at once; k in [k0, k1).
 *
 * The weight element is loaded ONCE and multiplied by every hidden vector — that single line is what
 * batching means, and it is why `bytes` in the trace does not grow with bcount. With nb == 1 this is
 * character-for-character the single-row dot, which is why "batched output == per-row output" is a
 * structural consequence here rather than a hope.
 *
 * Under LMH_MODE_DROP_LAST_K the term added LAST is undone: k1-1 for the forward walk, k0 for the
 * backward one. Both branches are written out so "undid the wrong end" is visible in the source
 * instead of being argued away. */
LMH_DEV static inline void lmh_dot_batch(const LmhTable *t, const uint8_t *w, const float *hv,
                                         int hstride, int k0, int k1, int rev, int mode, int nb,
                                         LmhTrace *tr, float *acc) {
    if (!rev) {
        for (int k = k0; k < k1; ++k) {
            const float v = lmh_term(t, w, k, mode);
            if (tr) lmh_trace_value(tr, lmh_float_to_bits(v));
            for (int b = 0; b < nb; ++b) {
                float p = hv[(long long)b * hstride + k] * v;
                if (mode == LMH_MODE_HALF_ACCUM)
                    p = lmh_bf16_to_float(lmh_float_to_bf16_bits(p));
                acc[b] += p;
            }
        }
    } else {
        for (int k = k1 - 1; k >= k0; --k) {
            const float v = lmh_term(t, w, k, mode);
            if (tr) lmh_trace_value(tr, lmh_float_to_bits(v));
            for (int b = 0; b < nb; ++b) {
                float p = hv[(long long)b * hstride + k] * v;
                if (mode == LMH_MODE_HALF_ACCUM)
                    p = lmh_bf16_to_float(lmh_float_to_bf16_bits(p));
                acc[b] += p;
            }
        }
    }
    if (mode == LMH_MODE_DROP_LAST_K && k1 > k0) {
        const int k = rev ? k0 : k1 - 1;
        const float v = lmh_term(t, w, k, mode);
        for (int b = 0; b < nb; ++b) acc[b] -= hv[(long long)b * hstride + k] * v;
    }
}

/* The fixed combine for LMH_ASSOC_SPLIT8: the LMH_SPLIT per-lane partials in index order, written
 * as one loop and called by BOTH arms with the same array contents, so the tree cannot be one order
 * on the device and another on the host. Starting from p[0] rather than from 0.0f is load-bearing:
 * 0.0f + (-0.0f) is +0.0f, so a leading zero would silently destroy the sign of a zero result for
 * the single-lane (LMH_ASSOC_SEQ) case, which is the case every bit-exact claim is made inside. */
LMH_DEV static inline float lmh_combine(const float *p, int n) {
    float acc = p[0];
    for (int i = 1; i < n; ++i) acc += p[i];
    return acc;
}

/* ---- the item: one vocab row, one lane's slice of it, ALL batch rows --------------------------- *
 *
 * This is where the batching question is actually decided. One (jj, lane) item loads each weight
 * element ONCE and multiplies it by all bcount hidden values, holding bcount accumulators; that is
 * what makes the addressed weight bytes independent of bcount, which is the only sense in which
 * batching "amortises" that a read-set can prove without a timer. PERROW is the same code at
 * bcount == 1, B times over, so the un-batched control addresses B times the bytes for bit-identical
 * outputs. The policy — which row, which table, which stride, which k-slice, what gets traced, where
 * the results are written — lives in these functions so the device kernel and the host twin cannot
 * hold two versions of it. The WRITE has one site (lmh_store) for the reason `pitfalls/20` gives for
 * the per-thread read-back bug: with LMH_SPLIT lanes per item, a store inside the dot would have
 * every lane of an item writing the same float.
 */
typedef struct {
    const LmhTable *tabs;
    int n_tables;
    const uint8_t *slab;
    const float *hidden;
    const int32_t *ids;
    int head_dim;                 /* from the config; LMH_MODE_ROW_STRIDE_HEAD_DIM uses it AS the
                                     row length, which is the mistake this lane is gating */
    float *out;
    long long out_floats;
} LmhCtx;

/* Which (mode, encoding, path) pairs mean something. gate.py mirrors this table, asserts the plan
 * contains no inapplicable case, and the drivers NaN-fill one if they ever see it — a loud wrong
 * answer instead of a stale zero that matches a zero reference. */
LMH_DEV static inline int lmh_mode_applicable(const LmhCase *c, const LmhTable *t) {
    switch (c->mode) {
        case LMH_MODE_CORRECT:            return 1;
        case LMH_MODE_GATHER_ROW_OFF_1:   return c->path == LMH_PATH_GATHER;
        case LMH_MODE_BATCH_SLOT_STRIDE:  return 1;         /* immune at bcount == 1, by geometry */
        case LMH_MODE_TIED_ALIAS:         return t->peer >= 0;
        case LMH_MODE_ROW_STRIDE_HEAD_DIM:return 1;
        case LMH_MODE_HALF_ACCUM:         return 1;
        case LMH_MODE_DROP_LAST_K:        return 1;
        case LMH_MODE_Q8_DROP_SCALE:      return t->enc == LMH_ENC_Q8_0;
        case LMH_MODE_CROSS_TABLE_READ:   return t->peer >= 0;
        case LMH_MODE_BATCH_RELOOPS:      return c->batching == LMH_BATCH_BATCHED;
        default:                          return 0;
    }
}

LMH_DEV static inline void lmh_nan_fill(float *dst, long long n) {
    for (long long i = 0; i < n; ++i) dst[i] = lmh_bits_to_float(LMH_NAN_BITS);
}

/* The row this (case, item) reads. Written once, so the gather index arithmetic — the arithmetic
 * this lane exists to gate — has exactly one spelling on both arms AND in the driver self-test.
 * The two selectors are separate functions rather than one with a flag, so the self-test can drive
 * each and a mode cannot quietly change the wrong one. */
LMH_DEV static inline long long lmh_case_row_dense(const LmhCase *c, int jj) {
    return (long long)c->row_lo + jj;
}

LMH_DEV static inline long long lmh_case_row_direct(const int32_t *ids, int jj, long long v_rows,
                                                  int mode) {
    long long row = (long long)ids[jj];
    if (mode == LMH_MODE_GATHER_ROW_OFF_1) row += 1;
    (void)v_rows;                     /* deliberately NOT clamped: reading the successor of the last
                                         row is a real thing a gather bug does, and the trace's cross
                                         counter is what reports it. Clamping here would hide it. */
    return row;
}

LMH_DEV static inline long long lmh_case_row(const LmhCtx *ctx, const LmhCase *c,
                                             const LmhTable *t, int jj, int mode) {
    if (c->path == LMH_PATH_DENSE) return lmh_case_row_dense(c, jj);
    return lmh_case_row_direct(ctx->ids + c->ids_off, jj, t->v_rows, mode);
}

/* The one item: (vocab row jj, k-slice lane), every batch row. See the block comment above LmhCtx.
 * `acc` receives bcount partials; the caller combines them across lanes with lmh_combine and writes
 * them with lmh_store, so no two lanes of an item ever touch the same float. */
LMH_DEV static inline void lmh_item(const LmhCtx *ctx, const LmhCase *c, int jj, int lane,
                                    int nlanes, LmhTrace *tr, float *acc) {
    const LmhTable *own = &ctx->tabs[c->table];
    const int mode = c->mode;
    /* Which tensor the BYTES come from, and which extent the tracer checks against. Under
     * LMH_MODE_TIED_ALIAS they are different tensors — for an untied model that is the entire
     * question, and on the tied control they are the same bytes, which is what makes the counter
     * pair demonstrable instead of assumed. */
    const LmhTable *src = own;
    if (mode == LMH_MODE_TIED_ALIAS && own->peer >= 0) src = &ctx->tabs[own->peer];

    const int h_used = (mode == LMH_MODE_ROW_STRIDE_HEAD_DIM) ? ctx->head_dim : own->h;
    const long long row = lmh_case_row(ctx, c, own, jj, mode);

    long long row_nb = 0;
    const uint8_t *wrow = lmh_row_ptr(src, ctx->slab, row, h_used, &row_nb);
    const int per = h_used / nlanes;      /* both drivers assert h_used % nlanes == 0 and, for
                                             Q8_0, per % LMH_Q8_0_ELTS == 0 */
    const int k0 = lane * per;
    const int k1 = (lane == nlanes - 1) ? h_used : k0 + per;
    long long nb = 0;
    const uint8_t *w = lmh_lane_ptr(own, wrow, k0, k1 - k0, &nb);

    const LmhTable *peer = (own->peer >= 0) ? &ctx->tabs[own->peer] : (const LmhTable *)0;
    const uint8_t *pbase = peer ? ctx->slab + peer->base : (const uint8_t *)0;
    const long long pn = peer ? peer->nbytes : 0;

    if (tr) {
        /* The re-looping "batched" kernel: reads (and traces) the row set once per batch row while
         * still producing one correct output per batch row. No arithmetic gate can see it. */
        int reps = 1;
        if (mode == LMH_MODE_BATCH_RELOOPS && c->batching == LMH_BATCH_BATCHED && c->bcount > 1)
            reps = c->bcount;
        for (int r = 0; r < reps; ++r)
            lmh_trace_range(tr, w, nb, ctx->slab + own->base, own->nbytes, pbase, pn);
        if (mode == LMH_MODE_CROSS_TABLE_READ && peer) {
            /* the witness mode: touch the PEER's bytes for the same row, compute from own. Untied
             * this must move cross and peer_hits; tied it cannot move cross, because the peer range
             * is the own range. */
            long long pnb = 0;
            const uint8_t *pw = lmh_row_ptr(peer, ctx->slab, row, h_used, &pnb);
            long long pnb2 = 0;
            const uint8_t *pl = lmh_lane_ptr(peer, pw, k0, k1 - k0, &pnb2);
            lmh_trace_range(tr, pl, pnb2, ctx->slab + own->base, own->nbytes, pbase, pn);
        }
        LMH_ACC_ADD32(&tr->lanes, 1u);
    }

    for (int b = 0; b < c->bcount; ++b) acc[b] = 0.0f;
    lmh_dot_batch(own, wrow, ctx->hidden + (long long)c->b0 * (long long)own->h, own->h,
                  k0, k1, c->assoc == LMH_ASSOC_REV ? 1 : 0, mode, c->bcount, tr, acc);
}

LMH_DEV static inline void lmh_store(const LmhCtx *ctx, const LmhCase *c, int b, int jj, float v,
                                     LmhTrace *tr) {
    /* The slot stride is the second half of the batching question: a batched GEMV must give batch
     * row b its own n_out slots. LMH_MODE_BATCH_SLOT_STRIDE is the typed-pointer-shaped mistake of
     * reusing the row length for the output stride. */
    const LmhTable *t = &ctx->tabs[c->table];
    const int slot = (c->mode == LMH_MODE_BATCH_SLOT_STRIDE) ? t->h : c->n_out;
    const long long at = (long long)c->out_off + (long long)b * slot + jj;
    if (at < 0 || at >= ctx->out_floats) {
        if (tr) LMH_ACC_ADD32(&tr->bover, 1u);
        return;
    }
    ctx->out[at] = v;
}

#endif /* LM_HEAD_KERNELS_H */
