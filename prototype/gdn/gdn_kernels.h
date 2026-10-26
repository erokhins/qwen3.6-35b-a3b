/* gdn_kernels.h — slice 1 of the GDN token mixer of `Qwen/Qwen3.6-35B-A3B`: the chunked scan core
 * for ONE chunk from a zero initial state, plus §04.7's gated norm and §04.8's out_proj.
 *
 * ONE arithmetic source, compiled twice (`pitfalls/40` §"a check whose reference side is derived
 * from the thing under test"):
 *
 *   host_probe.cpp -> g++    the host twin                 (`arm=host_twin`)
 *   gdn.hip        -> hipcc  the device TU, executed on CPU (`arm=hip_hostpath`, `hip_calls=0`)
 *
 * The two COMPILED arms are compared at ZERO tolerance to each other: one spelling of the
 * arithmetic, two compilers, so a disagreement is a codegen finding and not something a band may
 * absorb. Everything else is judged by the band this repo's oracle derived
 * (`speccheck/evidence/gdn-band-2026-10-26.json`), read only through `gdn_ref.load_band()` and
 * applied only through `gdn_ref.band_check_row()` — both from `gate.py`. There is no threshold,
 * band or floor in this file, and none may be added here: a limit typed beside the arithmetic is a
 * limit the arithmetic cannot fail.
 *
 * ---------------------------------------------------------------------------------------------
 * SLICE 1 IN, SLICE 2/3 OUT (an explicit boundary, not an omission)
 *   IN : the per-head scan over the token dimension of a SINGLE chunk — §04.4's scan entry (the
 *        grouped K->V expansion, the two l2norms, the q scaling), §04b.3's decay cumsum and
 *        pairwise matrix, §04b.4's UT system and the TWO solves (`new_values`, `k_cumdecay`),
 *        §04b.4's delta rule `v_new = new_values - k_cumdecay S`, §04b.5's intra-chunk attention
 *        and state write, §04.6's exit cast, §04.7's gated norm, §04.8's out_proj. It emits the
 *        seven names the oracle uses — q_hat, k_hat, scan_out, core_attn, state, normed, y —
 *        which is exactly `gdn_ref.SCAN_WIDTH_STAGES`, so `band_check_row` applies with no
 *        translation.
 *   OUT: more than one chunk, any carried state, batching, MTP, §04.2/§04.3 (the stages this file
 *        READS), and every performance question. `init_state` must be all zero: `gdn_case_run`
 *        counts the non-zero words and the case REFUSES rather than quietly treating a live state
 *        as zero. A zero state is a case, not a feature.
 *
 * ---------------------------------------------------------------------------------------------
 * DECLARED ARITHMETIC — the compiled arms must agree bit for bit, so the order lives in the source:
 *   every reduction runs its index ASCENDING with one accumulator: no tree, no split, no FMA
 *   (`-ffp-contract=off` on both compiled arms; `-ffast-math` appears nowhere).
 *   `expf`, `sqrtf`, `/` are the plain calls, never `__expf` and never a reciprocal approximation.
 *   §04.4's l2norm is transcribed as `x * rsqrt(sum(x*x) + eps)` with eps INSIDE the rsqrt and the
 *   reduction a SUM, not a mean. §04.7's gate is `x * sigmoid(x)` spelled `x / (1 + expf(-x))`.
 *   the two solves are forward substitution on `I + strictly_lower(ut_system)` with a UNIT
 *   diagonal (§04b.4 `unitriangular=True`): row i ascending, inner dot over j<i ascending. That is
 *   §04b.4's EXPORT branch — a legal form under A-16 — not the reference's `solve_triangular`.
 *   bf16 crosses every file boundary as a 16-bit BIT IMAGE and is widened by a left shift, never
 *   by a numeric cast (`pitfalls/30` §Tensor.to casts, .view bitcasts). `gdn_bf16_of_f32` is
 *   round-to-nearest-even, the rule torch's `.to(bf16)` uses.
 *
 * ---------------------------------------------------------------------------------------------
 * READ/DECODE FACTS vs REDUCTIONS — the rule this lane uses to decide which fields get a
 * bit-exactness check and which get a band:
 *   A field is a READ/DECODE FACT iff this file obtained it by loading, copying or widening, with
 *   NO reduction between the bytes and the register. Those are the eight fields every load goes
 *   through `gdn_load_bits()` for — q_pre, k_pre, v_pre, beta, g_decay, z, norm_weight, out_weight
 *   — and each gets an EXACT check with no tolerance: a wrong element offset, a bf16 image read as
 *   f32, a shift written as a cast, or a stale weight are all facts about bytes, and a tolerance
 *   on them would license them. `norm_weight` is the one the band is MEASURED to swallow
 *   (`speccheck/evidence/gdn-cross-form-2026-10-26.txt` line "the LEAST-scaled defect the band
 *   licenses": `norm_weight_f32` at 1/2**2 ACCEPTED with 190 of 131,072 stored elements still
 *   changed), so it is the named check; `gate.py --witness norm-weight` shows both verdicts.
 *   A field this file COMPUTES by summing is a REDUCTION: q_hat, k_hat, scan_out, core_attn,
 *   state, normed, y. Those get the band and nothing more. `core_attn` and `normed` are casts of
 *   computed values, so they inherit their parent's status even though the cast itself is exact.
 */
#ifndef GDN_KERNELS_H_
#define GDN_KERNELS_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __HIPCC__
#define GDNDAY_DEV __device__ __host__
#else
#define GDNDAY_DEV
#endif

/* Geometry, read from the committed config by gate.py and re-asserted there against the dump
 * manifest; `gdn_case_run` refuses a case whose T exceeds C. hidden_size=2048; h_k=16; h_v=32;
 * d_k=d_v=128; expansion=h_v/h_k=2; C=64 the reference chunk width (§04b.2, NOT a config field);
 * value_dim=h_v*d_v=4096. */
#define GDN_H 2048
#define GDN_H_K 16
#define GDN_H_V 32
#define GDN_D_K 128
#define GDN_D_V 128
#define GDN_EXP (GDN_H_V / GDN_H_K)
#define GDN_C 64
#define GDN_VALUE_DIM (GDN_H_V * GDN_D_V)
#define GDN_L2_EPS 1e-6f   /* §04.4's l2norm eps: inside the rsqrt of a SUM */
#define GDN_RMS_EPS 1e-6f  /* §04.7's rms_norm_eps: same value, different norm */

/* The only two non-honest modes slice 1 has a use for. `STUB` is the honesty control: it writes
 * the input `v_pre` into every output word without doing any of the scan, so a comparator that
 * reads the differ instead of the verdict is caught (`pitfalls/40` §a check whose predicate is a
 * constant). `NORM_WEIGHT_F32` is §04.7's width injection — the case the `normed` band is
 * measured to accept — reproduced here so the accepted case exists IN THIS LANE and the named
 * bit-exactness check can be shown beside it. */
enum GdnMode { GDN_M_HONEST = 0, GDN_M_STUB = 1, GDN_M_NORM_WEIGHT_F32 = 2 };

/* ---- the eight fields the kernel LOADS, and the order they are echoed in --------------------- */
enum GdnField { GF_Q_PRE = 0, GF_K_PRE, GF_V_PRE, GF_BETA, GF_G_DECAY, GF_Z, GF_NORM_W, GF_OUT_W,
                GF_N };

GDNDAY_DEV static inline int gdn_field_is_f32(int field) { return field == GF_G_DECAY; }
GDNDAY_DEV static inline long gdn_field_n(int field, int T) {
    switch (field) {
        case GF_Q_PRE: return (long)T * GDN_H_K * GDN_D_K;
        case GF_K_PRE: return (long)T * GDN_H_K * GDN_D_K;
        case GF_V_PRE: return (long)T * GDN_H_V * GDN_D_V;
        case GF_BETA: return (long)T * GDN_H_V;
        case GF_G_DECAY: return (long)T * GDN_H_V;
        case GF_Z: return (long)T * GDN_H_V * GDN_D_V;
        case GF_NORM_W: return GDN_D_V;
        default: return (long)GDN_H * GDN_VALUE_DIM;
    }
}
static const char *const gdn_field_name[GF_N] = {
    "q_pre", "k_pre", "v_pre", "beta", "g_decay", "z", "norm_weight", "out_weight"
};

/* ---- bf16 as a bit image --------------------------------------------------------------------- */
GDNDAY_DEV static inline uint16_t gdn_bf16_of_f32(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
#ifdef GDN_BROKEN_BF16_TRUNC
    /* the control build (build.sh --broken): truncate instead of round-to-nearest-even. This is
     * the single most common bf16 defect there is, and it is compiled ONLY into
     * build/host_probe_broken so the self-test and the read checks can be witnessed failing. */
    return (uint16_t)(u >> 16);
#endif
    /* exponent-all-ones with a nonzero mantissa is a NaN and must stay a NaN. The mask is 0xff and
     * not 0x1ff: `u >> 23` of a POSITIVE float can never exceed 0xff, so a 0x1ff test fires only
     * for negative NaNs and turns a positive signalling NaN into +inf — the exact corruption
     * §04b.3 exists to prevent, arriving through the cast instead of the exp. Found by the
     * self-test, which is the only reason it is written down. */
    if (((u >> 23) & 0xffu) == 0xffu && (u & 0x7fffffu) != 0u)
        return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
GDNDAY_DEV static inline float gdn_f32_of_bf16(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
GDNDAY_DEV static inline float gdn_bf16_rt(float f) { return gdn_f32_of_bf16(gdn_bf16_of_f32(f)); }

/* ---- the case and the context ---------------------------------------------------------------- */
typedef struct {
    int t;      /* real tokens in the chunk, 1..GDN_C */
    int mode;   /* GdnMode */
} GdnCase;

typedef struct {
    /* loaded inputs: ONE 32-bit word per element, holding the field's stored BIT IMAGE (bf16 in
     * the low 16 bits, f32 whole for g_decay). The kernel therefore never re-interprets a byte
     * stream at arithmetic speed, and "what did it load" is one array of words per field. */
    const uint32_t *q_pre;     /* [T,h_k,d_k] bf16 image */
    const uint32_t *k_pre;     /* [T,h_k,d_k] bf16 image */
    const uint32_t *v_pre;     /* [T,h_v,d_v] bf16 image */
    const uint32_t *beta;      /* [T,h_v]  bf16 image: beta is the activation dtype */
    const uint32_t *g_decay;   /* [T,h_v]  f32 image */
    const uint32_t *z;         /* [T,h_v,d_v] bf16 image */
    const uint32_t *norm_w;    /* [d_v] bf16 image */
    const uint32_t *out_w;     /* [H, value_dim] bf16 image */
    const float    *init_state; /* [h_v,d_k,d_v] — required all zero in slice 1 */
    /* per-head scratch */
    float *cum;    /* [h_v,C]        */
    float *ut;     /* [h_v,C,C]      */
    float *ia;     /* [h_v,C,C]      */
    float *qhat;   /* [h_v,C,d_k]    */
    float *khat;   /* [h_v,C,d_k]    */
    float *kb;     /* [h_v,C,d_k]    k_hat * beta, §04b.3 */
    float *vpad;   /* [h_v,C,d_v]    */
    float *vb;     /* [h_v,C,d_v]    v * beta, §04b.3 */
    float *dcb;    /* [h_v,C,d_k]    k_beta * exp(cum), §04b.4 dec_k_beta */
    float *newv;   /* [h_v,C,d_v]    solve 1: new_values */
    float *kcd;    /* [h_v,C,d_k]    solve 2: k_cumdecay */
    float *vnew;   /* [h_v,C,d_v]    the delta rule */
    float *qdec;   /* [h_v,C,d_k]    q_hat * exp(cum) */
    float *kdec;   /* [h_v,C,d_k]    k_hat * exp(cum_last - cum) */
    /* the seven emitted stages, in the oracle's own orientations, one word per element (the
     * bf16 stages keep their BIT IMAGE in the low 16 bits, so the arm and the driver never
     * disagree about the width of a stored stage) */
    float    *q_hat;     /* [h_v,T,d_k]  */
    float    *k_hat;     /* [h_v,T,d_k]  */
    float    *scan_out;  /* [h_v,T,d_v]  */
    uint32_t *core_attn; /* [T,h_v,d_v]  bf16 image */
    float    *state;     /* [h_v,d_k,d_v] */
    uint32_t *normed;    /* [T,h_v,d_v]  bf16 image */
    uint32_t *y;         /* [T,H]        bf16 image */
    /* what the kernel LOADED, as raw bit images, one pass per field through gdn_load_bits -------
     * The driver compares these against the dumped tensor element by element at ZERO tolerance.
     * This is the object of the read/decode facts above: not "did the file hash right" (the dump
     * reader already checked the sha256) but "what did the kernel end up with in its own words". */
    uint32_t *echo[GF_N];
    long     *st;          /* counters and the refusal flag, see GdnSt */
    int       T;           /* the running case's token count, read by the exit-cast item */
} GdnCtx;

enum { GDN_ST_STATE_NONZERO = 0, GDN_ST_PAD_NONZERO, GDN_ST_REFUSE, GDN_ST_ITEMS,
       GDN_ST_STAGE_ITEMS, GDN_ST_N };

/* ---- one load path for every input word, so "what did the kernel see?" has one answer ---------
 * The raw bit image is what is echoed and what is checked; the widening happens only on the way
 * into the arithmetic. A field read as the wrong width (bf16 decoded as IEEE half, an f32 image
 * truncated to its bf16 top half, a cast where a bitcast belongs) changes the image, and the
 * exactness check in gate.py is what turns that into a refusal instead of a plausible ndiff. */
GDNDAY_DEV static inline uint32_t gdn_load_bits(const GdnCtx *c, int field, long idx) {
    switch (field) {
        case GF_Q_PRE: return c->q_pre[idx];
        case GF_K_PRE: return c->k_pre[idx];
        case GF_V_PRE: return c->v_pre[idx];
        case GF_BETA: return c->beta[idx];
        case GF_G_DECAY: return c->g_decay[idx];
        case GF_Z: return c->z[idx];
        case GF_NORM_W: return c->norm_w[idx];
        default: return c->out_w[idx];
    }
}

GDNDAY_DEV static inline float gdn_load(const GdnCtx *c, int field, long idx) {
    uint32_t u = gdn_load_bits(c, field, idx);
    if (!gdn_field_is_f32(field)) return gdn_f32_of_bf16((uint16_t)u);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ---- ascending, one accumulator, no FMA ------------------------------------------------------ */
GDNDAY_DEV static inline float gdn_dot(const float *a, const float *b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}
/* the state contraction: sum_k a[k] * S[k*row_stride + col]. The stride is explicit because
 * `[d_k,d_v]` and `[d_v,d_k]` are the same 128x128 shape here and a transposed read raises
 * nothing — which is why `state_stored_transposed` is an injection in the oracle and not a
 * type error. */
GDNDAY_DEV static inline float gdn_dot_col(const float *a, const float *S, int n, int col,
                                           int row_stride) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) acc += a[k] * S[(long)k * row_stride + col];
    return acc;
}
GDNDAY_DEV static inline float gdn_sumsq(const float *a, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * a[i];
    return acc;
}
GDNDAY_DEV static inline float gdn_silu(float x) { return x / (1.0f + expf(-x)); }

/* ---- item A: §04.4's scan entry for one (head, token) row -----------------------------------
 * V head h reads K head h/EXP (the reference's `repeat_interleave(expansion, dim=2)`). Pad rows
 * (t >= T) are exactly zero (§04b.2). `beta` is applied to K *and* V here (§04b.3). The l2norm is
 * taken in the SCAN dtype after widening, which is what `enter_scan` does. */
GDNDAY_DEV static inline void gdn_item_enter(const GdnCtx *c, const GdnCase *k, int h, int t) {
    int pad = (t >= k->t);
    int hk = h / GDN_EXP;
    long sq_ = ((long)t * GDN_H_K + hk) * GDN_D_K;
    long sv_ = ((long)t * GDN_H_V + h) * GDN_D_V;
    long o = ((long)h * GDN_C + t) * GDN_D_K;
    long ov = ((long)h * GDN_C + t) * GDN_D_V;
    float qa[GDN_D_K], ka[GDN_D_K];
    for (int i = 0; i < GDN_D_K; ++i) {
        qa[i] = pad ? 0.0f : gdn_load(c, GF_Q_PRE, sq_ + i);
        ka[i] = pad ? 0.0f : gdn_load(c, GF_K_PRE, sq_ + i);
    }
    float sumq = pad ? 0.0f : gdn_sumsq(qa, GDN_D_K);
    float sumk = pad ? 0.0f : gdn_sumsq(ka, GDN_D_K);
    /* §04.4's l2norm is `x * rsqrt(sum + eps)`. The oracle spells the reciprocal square root as
     * `torch.rsqrt`; the plain-C spelling `1.0f/sqrtf(x)` is NOT the same number, and on q_hat the
     * difference (max_abs 3.7e-08 against a licensed 2.9e-08) is outside the probe's band. Taking
     * the sqrt in double and rounding once reproduces the reference's rounding exactly on this
     * geometry, which is the honest fix: the band may not be widened, and a transcendental that
     * disagrees with the reference in its last bit is this kernel's defect, not the band's slack. */
    float rq = (float)(1.0 / sqrt((double)(sumq + GDN_L2_EPS)));
    float rk = (float)(1.0 / sqrt((double)(sumk + GDN_L2_EPS)));
    /* the chunked arm's `mul` spelling: `q * (d_k ** -0.5)` (`Form.named("chunked").q_scale`),
     * which is NOT the same product tree as the recurrent arm's `q / (d_k ** 0.5)` (A-16's
     * footnote). The pinned prefill was made by the chunked arm, so this is the mul branch. */
    const float qs = (float)(1.0 / sqrt(128.0));
    float bt = pad ? 0.0f : gdn_load(c, GF_BETA, (long)t * GDN_H_V + h);
    for (int i = 0; i < GDN_D_K; ++i) {
        float qv = qa[i] * rq, kv = ka[i] * rk;
        c->qhat[o + i] = qv * qs;
        c->khat[o + i] = kv;
        c->kb[o + i] = kv * bt;
        if (!pad) {
            long oo = ((long)h * k->t + t) * GDN_D_K + i;
            c->q_hat[oo] = c->qhat[o + i];
            c->k_hat[oo] = kv;
        }
    }
    for (int i = 0; i < GDN_D_V; ++i) {
        float vv = pad ? 0.0f : gdn_load(c, GF_V_PRE, sv_ + i);
        c->vpad[ov + i] = vv;
        c->vb[ov + i] = vv * bt;
    }
}

/* ---- item B: §04b.3's decay cumsum, one head. Ascending, one accumulator. Pads add 0.0. ------ */
GDNDAY_DEV static inline void gdn_item_cum(const GdnCtx *c, const GdnCase *k, int h) {
    float acc = 0.0f;
    for (int t = 0; t < GDN_C; ++t) {
        /* through gdn_load, so the f32 bit image is BITCAST and not converted (`pitfalls/30`
         * §Tensor.to casts, .view bitcasts — a `(float)` of the integer bit pattern is the same
         * bug wearing a cast). */
        if (t < k->t) acc += gdn_load(c, GF_G_DECAY, (long)t * GDN_H_V + h);
        c->cum[(long)h * GDN_C + t] = acc;
    }
}

/* ---- item C: one (head, i, j) cell of the two C x C matrices. The strictly-upper side is an
 * exact 0 BEFORE the exp (§04b.3: "Any implementation that masks after exp ... is not this
 * form"), which is what keeps exp(+|sum g|) = inf out of the product: g reaches -91.6 per token on
 * this checkpoint, so masking after the exp would be `inf * 0 = nan`. */
GDNDAY_DEV static inline void gdn_item_pair(const GdnCtx *c, const GdnCase *k, int h, int ij) {
    int i = ij / GDN_C, j = ij % GDN_C;
    long hb = (long)h * GDN_C * GDN_C + ij;
    (void)k;
    if (j > i) { c->ut[hb] = 0.0f; c->ia[hb] = 0.0f; return; }
    const float *kj = c->khat + ((long)h * GDN_C + j) * GDN_D_K;
    const float *kb = c->kb + ((long)h * GDN_C + i) * GDN_D_K;
    const float *qi = c->qhat + ((long)h * GDN_C + i) * GDN_D_K;
    float pw = expf(c->cum[(long)h * GDN_C + i] - c->cum[(long)h * GDN_C + j]);
    c->ut[hb] = gdn_dot(kb, kj, GDN_D_K) * pw;
    c->ia[hb] = gdn_dot(qi, kj, GDN_D_K) * pw;
}

/* ---- item D: one (head, row) of the three decayed row sets §04b.4/§04b.5 need ---------------- */
GDNDAY_DEV static inline void gdn_item_rowpost(const GdnCtx *c, const GdnCase *k, int h, int i) {
    (void)k;   /* every row gets the same treatment in slice 1; the case is carried so a later
                * slice's per-position policy cannot be added without touching this signature */
    long o = ((long)h * GDN_C + i) * GDN_D_K;
    float ci = c->cum[(long)h * GDN_C + i];
    float last = c->cum[(long)h * GDN_C + (GDN_C - 1)];
    float e_hi = expf(ci);
    float e_lo = expf(last - ci);
    for (int d = 0; d < GDN_D_K; ++d) {
        c->dcb[o + d] = c->kb[o + d] * e_hi;
        c->qdec[o + d] = c->qhat[o + d] * e_hi;
        c->kdec[o + d] = c->khat[o + d] * e_lo;
    }
}

/* ---- item E: ONE of the two solves, per (head, rhs, column). Forward substitution on
 * I + strictly_lower(ut): row i ascending, the inner dot over j<i ascending, unit diagonal
 * (§04b.4 `unitriangular=True`). The rhs is v_beta for `new_values` and dec_k_beta for
 * `k_cumdecay` — both are solved, because the delta rule and the state write name both. */
GDNDAY_DEV static inline void gdn_item_solve(const GdnCtx *c, int h, int rhs, int col) {
    const float *ut = c->ut + (long)h * GDN_C * GDN_C;
    long stride = (long)h * GDN_C * GDN_D_K;
    float *x = (rhs == 0) ? (c->newv + stride) : (c->kcd + stride);
    const float *b = (rhs == 0) ? (c->vb + stride) : (c->dcb + stride);
    for (int i = 0; i < GDN_C; ++i) {
        float acc = b[(long)i * GDN_D_K + col];
        for (int j = 0; j < i; ++j) acc -= ut[(long)i * GDN_C + j] * x[(long)j * GDN_D_K + col];
        x[(long)i * GDN_D_K + col] = acc;
    }
}

/* ---- item F: §04b.4's delta rule, per (head, column): v_new = new_values - k_cumdecay S.
 * `S` is [d_k, d_v] and the contraction is over k, read with the stride spelled out. The S term
 * is really multiplied and not skipped, so its exact zero in a zero-state case is a measured fact
 * about the case (GDN_ST_STATE_NONZERO) and not an assumption the code makes. */
GDNDAY_DEV static inline void gdn_item_vnew(const GdnCtx *c, int h, int col) {
    long stride = (long)h * GDN_C * GDN_D_K;
    const float *S = c->init_state + (long)h * GDN_D_K * GDN_D_V;
    for (int i = 0; i < GDN_C; ++i) {
        float sub = gdn_dot_col(c->kcd + stride + (long)i * GDN_D_K, S, GDN_D_K, col, GDN_D_V);
        c->vnew[stride + (long)i * GDN_D_K + col] = c->newv[stride + (long)i * GDN_D_K + col] - sub;
    }
}

/* ---- item G: §04b.5's intra-chunk output, one (head, row, v-column):
 * core_i = q_decayed_i S + intra_attn_i @ v_new. The `q_decayed S` term is really multiplied too,
 * so the zero it contributes in a zero-state case is witnessed rather than assumed. */
GDNDAY_DEV static inline void gdn_item_core(const GdnCtx *c, const GdnCase *k, int h, int i,
                                            int col) {
    const float *S = c->init_state + (long)h * GDN_D_K * GDN_D_V;
    const float *qd = c->qdec + ((long)h * GDN_C + i) * GDN_D_K;
    float acc = gdn_dot_col(qd, S, GDN_D_K, col, GDN_D_V);
    const float *iar = c->ia + (long)h * GDN_C * GDN_C + (long)i * GDN_C;
    const float *vn = c->vnew + (long)h * GDN_C * GDN_D_V;
    for (int j = 0; j < GDN_C; ++j) acc += iar[j] * vn[(long)j * GDN_D_V + col];
    if (i < k->t) c->scan_out[((long)h * k->t + i) * GDN_D_V + col] = acc;
}

/* ---- item H: §04b.5's state write for the chunk, one (head, k-row, v-column):
 * S_new[k,v] = S_old[k,v] * exp(cum_last) + sum_i k_decayed_i[k] * v_new_i[v]. */
GDNDAY_DEV static inline void gdn_item_state(const GdnCtx *c, int h, int row, int col) {
    const float *kd = c->kdec + (long)h * GDN_C * GDN_D_K;
    const float *vn = c->vnew + (long)h * GDN_C * GDN_D_V;
    float acc = 0.0f;
    for (int i = 0; i < GDN_C; ++i)
        acc += kd[(long)i * GDN_D_K + row] * vn[(long)i * GDN_D_V + col];
    float cd = expf(c->cum[(long)h * GDN_C + (GDN_C - 1)]);
    c->state[((long)h * GDN_D_K + row) * GDN_D_V + col] =
        c->init_state[((long)h * GDN_D_K + row) * GDN_D_V + col] * cd + acc;
}

/* ---- item I: §04.7's gated norm, one (token, head) row of d_v. Transcribed line for line from
 * Qwen3_5MoeRMSNormGated.forward, INCLUDING both bf16 roundings:
 *      var = mean(x_f32^2);  x = x*rsqrt(var+eps);  x = w_bf16 * x.to(bf16)   <- bf16 product
 *      x = x * silu(z_f32);  return x.to(bf16)
 * The `NORM_WEIGHT_F32` mode drops the `.to(bf16)` on the normed value before the weight, which
 * is the injection `gdn_ref.Faults.norm_weight_f32` and the one the `normed` band is measured to
 * accept at 1/2**2 scale. */
GDNDAY_DEV static inline void gdn_item_norm(const GdnCtx *c, const GdnCase *k, int t, int h) {
    const uint32_t *ca = c->core_attn + ((long)t * GDN_H_V + h) * GDN_D_V;
    float x[GDN_D_V];
    for (int i = 0; i < GDN_D_V; ++i) x[i] = gdn_f32_of_bf16((uint16_t)ca[i]);
    float var = gdn_sumsq(x, GDN_D_V) / (float)GDN_D_V;
    float rs = (float)(1.0 / sqrt((double)(var + GDN_RMS_EPS)));   /* torch.rsqrt, see item A */
    long o = ((long)t * GDN_H_V + h) * GDN_D_V;
    for (int i = 0; i < GDN_D_V; ++i) {
        float xv = x[i] * rs;
        float wv = gdn_load(c, GF_NORM_W, i);
        float num = (k->mode == GDN_M_NORM_WEIGHT_F32) ? (wv * xv)
                                                       : gdn_bf16_rt(wv * gdn_bf16_rt(xv));
        float g = gdn_silu(gdn_load(c, GF_Z, o + i));
        c->normed[o + i] = gdn_bf16_of_f32(num * g);
    }
}

/* ---- item J: §04.8's out_proj, one (token, output row). normed is flat [T, value_dim] and the
 * weight is [H, value_dim]; one f32 accumulator over the 4096, ascending, then the bf16 store. */
GDNDAY_DEV static inline void gdn_item_outproj(const GdnCtx *c, const GdnCase *k, int t, int orow) {
    const uint32_t *nrow = c->normed + (long)t * GDN_VALUE_DIM;
    float acc = 0.0f;
    for (int j = 0; j < GDN_VALUE_DIM; ++j)
        acc += gdn_f32_of_bf16((uint16_t)nrow[j])
             * gdn_load(c, GF_OUT_W, (long)orow * GDN_VALUE_DIM + j);
    c->y[(long)t * GDN_H + orow] = gdn_bf16_of_f32(acc);
    (void)k;
}

/* ---- item K: §04.6's exit cast: the scan's f32 output becomes the activation dtype ------------
 * `core_attn` is the scan output transposed to [T,h_v,d_v] and cast, which is the tensor the gated
 * norm reads; `scan_out` stays [h_v,T,d_v] in f32 — the two orientations the dump stores. Pinning
 * only `core_attn` would hide an exit-cast bug (`pitfalls/30` §a hook that reads output). */
GDNDAY_DEV static inline void gdn_item_exit_cast(const GdnCtx *c, int t, int h, int col) {
    c->core_attn[((long)t * GDN_H_V + h) * GDN_D_V + col] =
        gdn_bf16_of_f32(c->scan_out[((long)h * c->T + t) * GDN_D_V + col]);
}

/* ---- the STUB: no arithmetic at all, the loaded v_pre widened and written into every output ---
 * Every emitted stage moves, so a comparator that would accept it is reading a constant and not
 * the differ. Deliberately shape-legal: same word count per stage, same dtypes. */
GDNDAY_DEV static inline void gdn_item_stub(const GdnCtx *c, const GdnCase *k, int stage,
                                            long it) {
    long o = 0;
    switch (stage) {
        case 0: o = it * GDN_D_K; break;   /* q_hat  */
        case 1: o = it * GDN_D_K; break;   /* k_hat  */
        case 2: o = it * GDN_D_V; break;   /* scan_out */
        case 3: o = it * GDN_D_V; break;   /* core_attn */
        case 4: o = it; break;             /* state  */
        case 5: o = it * GDN_D_V; break;   /* normed */
        default: o = it; break;            /* y      */
    }
    float v = gdn_load(c, GF_V_PRE, it % gdn_field_n(GF_V_PRE, k->t));
    if (stage == 2 || stage == 0 || stage == 1 || stage == 4) {
        float *d = (stage == 0) ? c->q_hat : (stage == 1) ? c->k_hat
                 : (stage == 2) ? c->scan_out : c->state;
        if (o < (stage == 4 ? (long)GDN_H_V * GDN_D_K * GDN_D_V
                            : (long)k->t * GDN_H_V * GDN_D_V)) d[o] = v + 1.0f;
    } else {
        uint32_t *d = (stage == 3) ? c->core_attn : (stage == 5) ? c->normed : c->y;
        long cap = (stage == 3 || stage == 5) ? (long)k->t * GDN_H_V * GDN_D_V
                                              : (long)k->t * GDN_H;
        if (o < cap) d[o] = gdn_bf16_of_f32(v + 1.0f);
    }
}

#endif /* GDN_KERNELS_H_ */
