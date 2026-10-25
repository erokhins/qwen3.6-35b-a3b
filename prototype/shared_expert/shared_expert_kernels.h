/* shared_expert_kernels.h — the shared expert of `Qwen/Qwen3.6-35B-A3B`: fused gate|up, SwiGLU,
 * `down`, and the scalar `shared_expert_gate`, in ONE source compiled twice.
 *
 *   - `shared_expert.hip` -> hipcc, gfx1151: the device implementation under test.
 *   - `host_probe.cpp`    -> g++: a host build of the SAME bodies, so a device-vs-host disagreement
 *     is attributable to the device compiler/ISA and not to two spellings of the arithmetic
 *     (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the
 *     thing under test"). The plan parser lives here for the same reason: both arms must read one
 *     plan, not two parsers.
 *
 * The reference is NOT in this file. It is
 *   `oracle_arm.py`   — which reproduces the installed reference's own form (verified bit-exact
 *                       against the pinned dumps, `results.md` §1) and independently emulates the
 *                       arithmetic declared below, and
 *   `tools/gguf_reader.py` — the audited CPU dequantizer of the published Q8_0 bytes.
 *
 * ---------------------------------------------------------------------------------------------
 * GEOMETRY — every number here is derived, and `derive_geometry.py` re-derives it from the pinned
 * artifacts and refuses the run if this header disagrees:
 *
 *   H      = 2048  hidden_size                          (checkpoints/Qwen3.6-35B-A3B.config.json)
 *   I      = 512   shared_expert_intermediate_size      (same file; equals moe_intermediate_size)
 *   TWO_I  = 1024  the fused gate|up row count, 2*I     (derived here: 2*512)
 *   Q8_0   = 34 bytes per 32 elements, so a gate/up row (2048 elts) is 64 whole blocks and a
 *            down row (512 elts) is 16 whole blocks. No row is ragged: 2048/32 = 64 and 512/32 = 16
 *            are exact (derived here, widths re-read from the file by tools/ggml_block_geom.py).
 *   Types, read from the pinned file (models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf):
 *            blk.N.ffn_gate_shexp.weight  Q8_0  dims (2048,512)  = HF [512, 2048]
 *            blk.N.ffn_up_shexp.weight    Q8_0  dims (2048,512)  = HF [512, 2048]
 *            blk.N.ffn_down_shexp.weight  Q8_0  dims (512,2048)  = HF [2048, 512]
 *            blk.N.ffn_gate_inp_shexp.weight F32 dims (2048,)    = HF [1, 2048] (rank 1 in file)
 *          i.e. the FILE does not fuse gate and up: the fused [TWO_I, H] view this kernel reads is
 *          an engine assembly, and which assembly is the right one is what §layout below settles.
 *
 * THE LAYOUT QUESTION (this lane's form of AGENTS.md prediction 3). The routed experts ship fused
 * ([E, 2I, H], gate,up = chunk(2)); the shared expert ships as two tensors. The kernel still wants
 * one launch covering both dots, so it needs an index rule for "fused row j". Four candidate rules
 * are compiled in and selectable per case, and the gate decides between them by WHICH candidates
 * match rather than by whether one does (`results.md` §3) — a shifted read makes every candidate
 * look equally wrong, which is how prediction 3 survived two investigations in the router lane.
 *
 * THE SHARED-EXPERT GATE. `shared_expert_gate` is `nn.Linear(hidden_size, 1, bias=False)` in the
 * installed reference (`transformers` 5.19.0, `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
 * line 900, applied at line 909 as `F.sigmoid(gate(x)) * shared_expert_output`), and the pinned dump
 * records it as [1, 2048] with `shared_gate_prob` [T, 1]. So it is ONE scalar per token, not
 * per-channel. This file nevertheless implements BOTH forms (`SE_SGATE_SCALAR`, `SE_SGATE_PERCHAN`)
 * because "assumed" and "tested" are different things: the gate runs both and requires the scalar
 * form to match the pinned probability vector and the per-channel form to disagree with it.
 *
 * THE DECLARED ARITHMETIC (each line is what this file does and what `oracle_arm.py` reproduces):
 *   dequant   Q8_0 value = f16(d) * int8(q). Exact in f32: f16->f32 is exact, int8->f32 is exact,
 *             and the product needs at most 11+7 = 18 significand bits, under f32's 24. So the
 *             device's f32 multiply, the host's, and the oracle's Python-double-then-round-once all
 *             agree EXACTLY. This is the q4k_dequant exactness argument, and it holds here and does
 *             not hold for Q4_K/Q5_K/Q6_K.
 *   dot       one output element per thread, accumulator f32, k = 0..cols-1 ASCENDING, one product
 *             and one add per step, contraction compiled out on both arms (`-ffp-contract=off`).
 *             `oracle_arm.py` §dot_ascending is the same order column by column, so the dots are
 *             bit-comparable between arms and against the oracle — unlike a torch matmul, whose
 *             blocked order nobody outside torch can reproduce.
 *   swiglu    the sibling policy, transferred as arithmetic (planning/port-inventory-2026-10-26.md
 *             §1 "What transfers" item 2):
 *                 g_b = bf16(gate_f32)
 *                 s_b = bf16(silu_f32(g_b))        silu_f32(z) = z * (1/(1+expf(-z)))
 *                 u_b = bf16(up_f32)                   (HF's own order, NOT z/(1+expf(-z)))
 *                 act = bf16(s_b * u_b)
 *             Computing this entirely in f32 — the "prettier" reading — is SE_MODE_F32_ACT here and
 *             must be caught, because the sibling measured it moving 27 % of act by 1 ULP.
 *   raw       bf16(dot_f32(act, down_row)): `down` is a bf16 Linear in the reference, so its f32
 *             accumulation is rounded once on the way out.
 *   gate dot  t = dot_f32(x, sg) with sg read EXACT (the file stores it F32); t_b = bf16(t) mirrors
 *             the reference's bf16 Linear output; p = bf16(sigmoid_f32(t_b)).
 *   cast form BEFORE (SE_CAST_BEFORE, the reference's form, `F.sigmoid` returning the model dtype):
 *                 out = bf16(f32(p) * f32(raw_b))     the probability is cast BEFORE the multiply
 *             AFTER (SE_CAST_AFTER, the wrong order, prediction 4's cousin):
 *                 out = bf16(sigmoid_f32(t_b) * f32(raw_b))   keeps the f32 probability
 *             and `pitfalls/30` §"Free-run parity" is why this lane cares: a cast moved across a
 *             multiply changes the winner of a near-tie.
 *   FMA       contraction compiled out on BOTH arms so the two compilers round the same sequence.
 * ---------------------------------------------------------------------------------------------
 */
#ifndef SHARED_EXPERT_KERNELS_H
#define SHARED_EXPERT_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define SE_DEV __host__ __device__
#else
#define SE_DEV
#endif

/* ---- geometry (config-derived; derive_geometry.py refuses a disagreement) ------------------ */
#define SE_H 2048                 /* hidden_size                                                   */
#define SE_I 512                  /* shared_expert_intermediate_size                               */
#define SE_TWO_I 1024             /* 2*I: the fused gate|up row count                              */
#define SE_Q8_ELTS 32             /* QK8_0, from the pinned ggml-common.h block_q8_0               */
#define SE_Q8_BYTES 34            /* 2*sizeof(ggml_half) + QK8_0, same struct                      */
#define SE_GATE_BLOCKS (SE_H / SE_Q8_ELTS)     /* 64 whole blocks per gate/up row    (derived)    */
#define SE_DOWN_BLOCKS (SE_I / SE_Q8_ELTS)     /* 16 whole blocks per down row       (derived)    */
#define SE_THREADS 128            /* threads per block; each thread owns one output row            */
#define SE_NAN_BITS 0x7FC00000u   /* what an unwritten slot reads, so a short grid is loud         */

/* ---- output-field layout of one case's record (floats) --------------------------------------- *
 * `gate`/`up` are f32 dots; `act`/`raw`/`out`/`prob` carry bf16 values widened to f32 (a lossless
 * widening, so gate.py can narrow them back with >>16 and judge them with the bf16 ruler).        */
#define SE_N_GATE 512
#define SE_N_UP 512
#define SE_N_ACT 512
#define SE_N_RAW 2048
#define SE_N_OUT 2048
#define SE_N_PROB 1
#define SE_OFF_GATE 0
#define SE_OFF_UP 512
#define SE_OFF_ACT 1024
#define SE_OFF_RAW 1536
#define SE_OFF_OUT 3584
#define SE_OFF_PROB 5632
#define SE_CASE_FLOATS 5633
#define SE_FIELD_COUNT 6

enum { SE_F_GATE = 0, SE_F_UP = 1, SE_F_ACT = 2, SE_F_RAW = 3, SE_F_OUT = 4, SE_F_PROB = 5 };

/* ---- the fused-row layout candidates (see the header comment) -------------------------------- */
enum {
    SE_LAYOUT_GATE_FIRST = 0,   /* gate row j = fused j,      up row j = fused I + j               */
    SE_LAYOUT_UP_FIRST = 1,     /* gate row j = fused I + j,  up row j = fused j                   */
    SE_LAYOUT_ROW_INTERLEAVED = 2, /* gate row j = fused 2j,  up row j = fused 2j+1                */
    SE_LAYOUT_PAIR_SWAP = 3,    /* same rows as GATE_FIRST, but the epilogue pairs silu(up)*gate   */
    SE_LAYOUT_COUNT = 4
};

/* ---- the gate form and the cast position, independent of the layout -------------------------- */
enum { SE_SGATE_SCALAR = 0, SE_SGATE_PERCHAN = 1, SE_SGATE_COUNT = 2 };
enum { SE_CAST_BEFORE = 0, SE_CAST_AFTER = 1, SE_CAST_COUNT = 2 };

/* ---- the modes: each is one mistake an engine or a converter can actually make ---------------- */
enum {
    SE_MODE_CORRECT = 0,
    SE_MODE_LAYOUT_UP_FIRST = 1,        /* halves swapped: chunk(2) read from the wrong half       */
    SE_MODE_LAYOUT_ROW_INTER = 2,       /* row-interleaved fused view                             */
    SE_MODE_LAYOUT_PAIR_SWAP = 3,       /* silu applied to the up half instead of the gate half    */
    SE_MODE_ROW_SHIFT1 = 4,             /* every fused row index read one row late: the shifted    */
                                        /* read that made prediction 3 look ambiguous              */
    SE_MODE_F32_ACT = 5,                /* SwiGLU epilogue entirely in f32, no bf16 roundings      */
    SE_MODE_TAIL_DROP = 6,              /* last Q8_0 block of every weight row not read (short K)  */
    SE_MODE_COLS_HALF = 7,              /* dot covers half the columns: a stride/limit bug         */
    SE_MODE_DOWN_HALF = 8,              /* down launch covers H/2 rows: leaves poison holes        */
    SE_MODE_WIDE_MISREAD = 9,           /* Q8_0 scale read as bf16 instead of f16: a width misread  */
    SE_MODE_COUNT = 10
};

/* weight roles in the slab */
enum { SE_W_FUSED = 0, SE_W_DOWN = 1, SE_W_SG = 2, SE_W_COUNT = 3 };

/* ---- shared primitives ---------------------------------------------------------------------- */
SE_DEV static inline uint32_t se_bits_of(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}
SE_DEV static inline float se_float_of(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}
/* bf16 bit image -> f32: the widening is `bits << 16`, exact by construction (pitfalls/30 names the
 * `.view`/`.to` confusion this avoids). */
SE_DEV static inline float se_f32_of_bf16(uint16_t u) { return se_float_of((uint32_t)u << 16); }
SE_DEV static inline float se_f32_of_bf16_bytes(const uint8_t *p) {
    return se_f32_of_bf16((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}
/* f32 -> bf16, round-to-nearest-even over the top 16 bits; a NaN stays a quiet NaN. */
SE_DEV static inline uint16_t se_bf16_bits_of(float f) {
    const uint32_t b = se_bits_of(f);
    if (((b >> 23) & 0xFFu) == 0xFFu && (b & 0x007FFFFFu)) return (uint16_t)((b >> 16) | 0x0040u);
    const uint32_t lsb = (b >> 16) & 1u;
    return (uint16_t)((b + 0x7FFFu + lsb) >> 16);
}
SE_DEV static inline float se_bf16_rt(float f) { return se_f32_of_bf16(se_bf16_bits_of(f)); }
/* ggml_half (IEEE binary16) -> f32, EXACT for every encoding including subnormals: normal and
 * subnormal f16 significands fit f32 with room to spare, and the subnormal path multiplies by the
 * power of two 2^-24 so it cannot round. Q8_0 stores its scale as ggml_half, so this is the only
 * width conversion on the weight path. */
SE_DEV static inline float se_f16_to_f32(uint16_t h) {
    const uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t e = ((uint32_t)h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) return se_float_of(s);
        return se_float_of(se_bits_of((float)m * 5.9604644775390625e-08f) | s);
    }
    if (e == 0x1Fu) return se_float_of(s | 0x7F800000u | (m << 13));
    return se_float_of(s | ((e + 112u) << 23) | (m << 13));
}
/* bf16 bit image reinterpreted as f16 bit image: the WRONG width reading (SE_MODE_WIDE_MISREAD).
 * It is a legal mis-transcription of `block_q8_0::d` — a reader that typed the scale
 * `__nv_bfloat16` instead of `__half` produces exactly this, and it is not far-off enough to be
 * caught by a loose band, which is why this lane reads widths from the file instead. */
SE_DEV static inline float se_f16_bits_as_bf16(uint16_t h) {
    return se_f32_of_bf16(h);
}

/* One Q8_0 block, 34 bytes: { ggml_half d; int8_t qs[32]; }. */
SE_DEV static inline int se_q8_block_bytes(void) { return SE_Q8_BYTES; }
SE_DEV static inline float se_q8_value(const uint8_t *blk, int e, int wide_misread) {
    const uint16_t dh = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    const float d = wide_misread ? se_f16_bits_as_bf16(dh) : se_f16_to_f32(dh);
    return d * (float)(int8_t)blk[2 + e];
}

/* The declared dot: ONE output element, f32 accumulator, k ASCENDING, one mul then one add per
 * step (contraction compiled out). `cols` is the number of elements the mode actually reads, and
 * *reads counts them, so the report can be reconciled against geometry. */
SE_DEV static inline float se_dot_q8_row(const uint8_t *row, int cols, const float *x,
                                         int wide_misread, int *reads) {
    float acc = 0.0f;
    for (int k = 0; k < cols; ++k) {
        const int b = k >> 5;
        const float w = se_q8_value(row + (size_t)b * SE_Q8_BYTES, k & 31, wide_misread);
        acc += x[k] * w;
        *reads += 1;
    }
    return acc;
}
/* The `shared_expert_gate` dot: sg is stored F32 (exact, policy A keeps it exact) so this is a
 * plain f32 row dot with the same ascending order. */
SE_DEV static inline float se_dot_f32_row(const uint8_t *row, int cols, const float *x, int *reads) {
    float acc = 0.0f;
    for (int k = 0; k < cols; ++k) {
        float w;
        memcpy(&w, row + (size_t)k * 4u, 4u);
        acc += x[k] * w;
        *reads += 1;
    }
    return acc;
}

SE_DEV static inline float se_silu_f32(float z) { return z * (1.0f / (1.0f + expf(-z))); }
SE_DEV static inline float se_sigmoid_f32(float z) { return 1.0f / (1.0f + expf(-z)); }

/* SwiGLU, three roundings, the sibling policy transferred as arithmetic. F32_ACT is the wrong
 * spelling and must be caught, not tolerated. */
SE_DEV static inline float se_swiglu(float gate_f32, float up_f32, int f32_act) {
    if (f32_act) return se_silu_f32(gate_f32) * up_f32;
    const float g_b = se_bf16_rt(gate_f32);
    const float s_b = se_bf16_rt(se_silu_f32(g_b));
    const float u_b = se_bf16_rt(up_f32);
    return se_bf16_rt(s_b * u_b);
}

/* The gate-weight multiply, both cast positions (prediction 4's cousin). */
SE_DEV static inline float se_gate_mul(float t_b, float raw_b, int cast_after) {
    if (cast_after) return se_bf16_rt(se_sigmoid_f32(t_b) * raw_b);
    const float p_b = se_bf16_rt(se_sigmoid_f32(t_b));
    return se_bf16_rt(p_b * raw_b);
}
SE_DEV static inline float se_gate_prob(float t_b, int cast_after) {
    /* the field records the probability the multiply is allowed to see: rounded to the model dtype
       in the reference's form, unrounded in the wrong one (which is how the field itself tells the
       two arms apart) */
    return cast_after ? se_sigmoid_f32(t_b) : se_bf16_rt(se_sigmoid_f32(t_b));
}

/* ---- the two digests (each blind to what the line at its print says it is blind to) ---------- *
 * XOR of the SIGNIFICAND+SIGN bits with the exponent field dropped: provably blind to a uniform
 *   power-of-two rescale, because every element's exponent moves and no mantissa bit does.
 * ORDER-FREE SUM of the full bit images mod 2^32: sees that rescale, and is blind only where the
 *   integer sums coincide (5,7 -> 6,6), which the XOR does see. Neither alone is a witness.
 */
typedef struct { uint32_t xor_sig; uint32_t sum_bits; uint32_t n_nonzero; } SeDigest;
SE_DEV static inline void se_digest_init(SeDigest *d) { d->xor_sig = 0u; d->sum_bits = 0u; d->n_nonzero = 0u; }
SE_DEV static inline void se_digest_add(SeDigest *d, float v) {
    const uint32_t b = se_bits_of(v);
    if (b == 0u || b == 0x80000000u) return;      /* zeros are absent, not neutral-and-counted */
    d->xor_sig ^= (b & 0x807FFFFFu);
    d->sum_bits += b;
    d->n_nonzero += 1u;
}
SE_DEV static inline int se_is_nan(float f) {
    const uint32_t b = se_bits_of(f);
    return ((b & 0x7F800000u) == 0x7F800000u) && ((b & 0x007FFFFFu) != 0u);
}

/* ---- the plan: one parser, both arms -------------------------------------------------------- */
#define SE_PLAN_FIELDS_W 8
#define SE_PLAN_FIELDS_A 4
#define SE_PLAN_FIELDS_C 17

typedef struct {
    int32_t id, role, n_rows, n_cols, kind;   /* role: SE_W_*; kind: 0 Q8_0, 1 F32                */
    int64_t byte_off, n_bytes;
} SeWeight;

typedef struct { int32_t id, n_elems; int64_t byte_off; } SeAct;   /* bf16 image, 2 B per element */

typedef struct {
    int32_t index;
    int32_t w_fused, w_down, w_sg, a;
    int32_t mode, sgate_form, cast_form, expect;
    int32_t dst_off, exp_off, n_floats, cmp_case;
    int32_t block, row;
    char field_exact[16];
    /* arm-written, and re-derived independently by gate.py from the output file */
    int32_t bad, nan, reads_fused, reads_down, reads_sg, n_nan, n_zero, n_value;
    uint32_t xor_sig, sum_bits;
    int32_t n_digest;
} SeCase;

typedef struct {
    SeWeight *w; int n_w;
    SeAct *a; int n_a;
    SeCase *c; int n_c;
    const uint8_t *slab; size_t slab_bytes;
    const float *expect; int n_expect;
} SePlan;

static inline int se_words(const char *line, char fields[][64], int max) {
    int n = 0;
    const char *p = line;
    while (n < max) {
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p || *p == '\n') break;
        int k = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') {
            if (k < 63) fields[n][k++] = *p;
            ++p;
        }
        fields[n][k] = 0;
        ++n;
    }
    return n;
}

/* Reads plan.tsv. Refuses a record whose field count is not the declared one, refuses an unknown
 * tag, and reports the per-tag counts so the caller can assert parsed == printed
 * (pitfalls/40 §"A driver whose parser matches nothing prints PASS"). */
static inline int se_read_plan(const char *path, SePlan *p, int want_w, int want_a, int want_c,
                               char *err, size_t errn) {
    FILE *fh = fopen(path, "r");
    if (!fh) { snprintf(err, errn, "cannot open plan %s", path); return 1; }
    char line[2048];
    char f[24][64];
    int nw = 0, na = 0, nc = 0, bad = 0;
    while (fgets(line, sizeof(line), fh)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int nf = se_words(line, f, 24);
        if (f[0][0] == 'W') {
            if (nf != SE_PLAN_FIELDS_W) { bad++; continue; }
            if (p->w) {
                SeWeight *u = &p->w[nw];
                u->id = atoi(f[1]); u->role = atoi(f[2]); u->n_rows = atoi(f[3]);
                u->n_cols = atoi(f[4]); u->kind = atoi(f[5]);
                u->byte_off = (int64_t)strtoll(f[6], 0, 10);
                u->n_bytes = (int64_t)strtoll(f[7], 0, 10);
            }
            ++nw;
        } else if (f[0][0] == 'A') {
            if (nf != SE_PLAN_FIELDS_A) { bad++; continue; }
            if (p->a) { p->a[na].id = atoi(f[1]); p->a[na].n_elems = atoi(f[2]);
                        p->a[na].byte_off = (int64_t)strtoll(f[3], 0, 10); }
            ++na;
        } else if (f[0][0] == 'C') {
            if (nf != SE_PLAN_FIELDS_C) { bad++; continue; }
            if (p->c) {
                SeCase *c = &p->c[nc];
                c->index = atoi(f[1]);
                c->w_fused = atoi(f[2]); c->w_down = atoi(f[3]); c->w_sg = atoi(f[4]);
                c->a = atoi(f[5]);
                c->mode = atoi(f[6]); c->sgate_form = atoi(f[7]); c->cast_form = atoi(f[8]);
                c->expect = atoi(f[9]);
                c->dst_off = atoi(f[10]); c->exp_off = atoi(f[11]); c->n_floats = atoi(f[12]);
                { int k = 0; while (k < 15 && f[13][k]) { c->field_exact[k] = f[13][k]; ++k; }
                  c->field_exact[k] = 0; }
                c->cmp_case = atoi(f[14]); c->block = atoi(f[15]); c->row = atoi(f[16]);
                c->bad = c->nan = c->reads_fused = c->reads_down = c->reads_sg = 0;
                c->n_nan = c->n_zero = c->n_value = 0;
                c->xor_sig = c->sum_bits = 0u; c->n_digest = 0;
            }
            ++nc;
        } else {
            bad++;     /* an unknown tag is a defect, not a line to skip silently */
        }
    }
    fclose(fh);
    if (bad) { snprintf(err, errn, "%s: %d records were malformed or carried an unknown tag", path, bad); return 1; }
    if (nw != want_w || na != want_a || nc != want_c) {
        snprintf(err, errn, "%s: parsed W=%d A=%d C=%d, plan expected W=%d A=%d C=%d",
                 path, nw, na, nc, want_w, want_a, want_c);
        return 1;
    }
    p->n_w = nw; p->n_a = na; p->n_c = nc;
    return 0;
}

SE_DEV static inline const SeWeight *se_weight(const SePlan *p, int id) {
    for (int i = 0; i < p->n_w; ++i) if (p->w[i].id == id) return &p->w[i];
    return 0;
}
SE_DEV static inline const SeAct *se_act(const SePlan *p, int id) {
    for (int i = 0; i < p->n_a; ++i) if (p->a[i].id == id) return &p->a[i];
    return 0;
}

/* ---- the stage bodies: one source, both arms ------------------------------------------------- */

/* How many columns a mode actually reads, and the row count a mode launches. Every one of these is
 * a GEOMETRY statement the report is reconciled against: `se_want_reads()` below is the same
 * arithmetic, and gate.py requires kernel count == geometry count on an honest case. */
SE_DEV static inline int se_cols_of(int cols_full, int mode) {
    if (mode == SE_MODE_TAIL_DROP) return cols_full - SE_Q8_ELTS;
    if (mode == SE_MODE_COLS_HALF) return cols_full / 2;
    return cols_full;
}
SE_DEV static inline int se_layout_of(int mode) {
    switch (mode) {
        case SE_MODE_LAYOUT_UP_FIRST: return SE_LAYOUT_UP_FIRST;
        case SE_MODE_LAYOUT_ROW_INTER: return SE_LAYOUT_ROW_INTERLEAVED;
        case SE_MODE_LAYOUT_PAIR_SWAP: return SE_LAYOUT_PAIR_SWAP;
        default: return SE_LAYOUT_GATE_FIRST;
    }
}
SE_DEV static inline int se_shift_of(int mode) { return mode == SE_MODE_ROW_SHIFT1 ? 1 : 0; }

/* fused row indices for the candidate layout, with the optional one-row shift. The shifted read is
 * what makes every candidate look equally wrong, and this lane can now tell the two apart. */
SE_DEV static inline void se_layout_rows(int layout, int shift, int j, int *gate_row, int *up_row) {
    int g, u;
    switch (layout) {
        case SE_LAYOUT_UP_FIRST:  g = SE_I + j; u = j; break;
        case SE_LAYOUT_ROW_INTERLEAVED: g = 2 * j; u = 2 * j + 1; break;
        default:                  g = j; u = SE_I + j; break;
    }
    if (shift) { g = (g + shift) % SE_TWO_I; u = (u + shift) % SE_TWO_I; }
    *gate_row = g; *up_row = u;
}

/* Stage 1: one fused work item j -> gate dot, up dot, act. */
SE_DEV static inline void se_stage_gateup(const SePlan *p, const SeCase *c, float *out, int j,
                                          int *reads) {
    const SeWeight *wf = se_weight(p, c->w_fused);
    const SeAct *ax = se_act(p, c->a);
    const uint8_t *x = p->slab + ax->byte_off;
    const int stride = SE_GATE_BLOCKS * SE_Q8_BYTES;
    const int cols = se_cols_of(SE_H, c->mode);
    int gr, ur;
    se_layout_rows(se_layout_of(c->mode), se_shift_of(c->mode), j, &gr, &ur);
    if (c->mode == SE_MODE_ROW_SHIFT1) { /* keep both halves inside the fused tensor, no wrap */
        if (gr >= SE_TWO_I) gr -= SE_TWO_I;
        if (ur >= SE_TWO_I) ur -= SE_TWO_I;
    }
    const uint8_t *grow = p->slab + wf->byte_off + (size_t)gr * stride;
    const uint8_t *uprow = p->slab + wf->byte_off + (size_t)ur * stride;
    const int wide = (c->mode == SE_MODE_WIDE_MISREAD);
    float gate = 0.0f, up = 0.0f;
    for (int k = 0; k < cols; ++k) {
        gate += se_f32_of_bf16_bytes(x + 2 * (size_t)k)
              * se_q8_value(grow + (size_t)(k >> 5) * SE_Q8_BYTES, k & 31, wide);
        up += se_f32_of_bf16_bytes(x + 2 * (size_t)k)
            * se_q8_value(uprow + (size_t)(k >> 5) * SE_Q8_BYTES, k & 31, wide);
    }
    *reads += 2 * cols;
    out[c->dst_off + SE_OFF_GATE + j] = gate;
    out[c->dst_off + SE_OFF_UP + j] = up;
    /* PAIR_SWAP reads the SAME rows as the honest case and swaps which one gets silu, so gate and up
     * come out identical to the honest case and only `act` can see it (measured: 0 of 512 gate words,
     * 0 of 512 up words, 505 of 512 act words differ from the honest case on layer0 row0). UP_FIRST
     * reads different rows, so it moves gate and up AND act (505 of 512 act words, 7 rows coinciding
     * because silu(a)*b and silu(b)*a agree for those). The blindness is the pair-swap one, and it is
     * the reason the gate judges `act` and never only the two dots. */
    const int pair_swapped = (se_layout_of(c->mode) == SE_LAYOUT_PAIR_SWAP);
    const float a = pair_swapped ? se_swiglu(up, gate, c->mode == SE_MODE_F32_ACT)
                                 : se_swiglu(gate, up, c->mode == SE_MODE_F32_ACT);
    out[c->dst_off + SE_OFF_ACT + j] = a;
}

/* Stage 2: one output row h of `down`, reading stage 1's act out of the case record. */SE_DEV static inline void se_stage_down(const SePlan *p, const SeCase *c, float *out, int h,
                                        int *reads) {
    const SeWeight *wd = se_weight(p, c->w_down);
    const int stride = SE_DOWN_BLOCKS * SE_Q8_BYTES;
    const int cols = se_cols_of(SE_I, c->mode);
    const uint8_t *row = p->slab + wd->byte_off + (size_t)h * stride;
    const float *act = out + c->dst_off + SE_OFF_ACT;
    const int wide = (c->mode == SE_MODE_WIDE_MISREAD);
    float acc = 0.0f;
    for (int k = 0; k < cols; ++k) acc += act[k] * se_q8_value(row + (size_t)(k >> 5) * SE_Q8_BYTES,
                                                               k & 31, wide);
    *reads += cols;
    out[c->dst_off + SE_OFF_RAW + h] = se_bf16_rt(acc);
}

/* Stage 3a: the shared-expert gate's scalar probability (the reference's [T,1]). */
SE_DEV static inline float se_stage_prob(const SePlan *p, const SeCase *c, int *reads) {
    const SeWeight *ws = se_weight(p, c->w_sg);
    const SeAct *ax = se_act(p, c->a);
    const uint8_t *x = p->slab + ax->byte_off;
    const uint8_t *sg = p->slab + ws->byte_off;
    float acc = 0.0f;
    for (int k = 0; k < SE_H; ++k) {
        float w;
        memcpy(&w, sg + 4 * (size_t)k, 4u);
        acc += se_f32_of_bf16_bytes(x + 2 * (size_t)k) * w;
    }
    *reads += SE_H;
    return se_gate_prob(se_bf16_rt(acc), c->cast_form);
}

/* Stage 3b: one output channel, both gate forms and both cast positions. */
SE_DEV static inline float se_stage_out_channel(const SePlan *p, const SeCase *c, const float *out,
                                                float prob, int h, int *reads) {
    const float raw = out[c->dst_off + SE_OFF_RAW + h];
    if (c->sgate_form == SE_SGATE_PERCHAN) {
        const SeWeight *ws = se_weight(p, c->w_sg);
        const SeAct *ax = se_act(p, c->a);
        const uint8_t *x = p->slab + ax->byte_off;
        const uint8_t *sg = p->slab + ws->byte_off;
        float w;
        memcpy(&w, sg + 4 * (size_t)h, 4u);
        const float t_b = se_bf16_rt(se_f32_of_bf16_bytes(x + 2 * (size_t)h) * w);
        *reads += 1;
        return se_gate_mul(t_b, raw, c->cast_form);
    }
    (void)reads;
    return se_bf16_rt(prob * raw);
}

#endif  /* SHARED_EXPERT_KERNELS_H */
