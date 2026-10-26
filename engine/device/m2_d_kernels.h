/* engine/device/m2_d_kernels.h — the GDN recurrent decode step (§04b.1) as ONE item body, compiled
 * twice: `m2_d_host.cpp` by g++ (the engine's own host-side GDN body) and `m2_d_arm.hip` by hipcc.
 *
 * THE FORM THIS IMPLEMENTS, stated once because the licences are per-form.
 *   This is the CACHED, SINGLE-TOKEN recurrent step — `specification/04` §04b.1: decay the state,
 *   read `k_hat . S`, form the delta, apply the rank-one update, read `q_hat . S`. The oracle's name
 *   for it is the arm `recurrent`, and the row key that licenses it at the pinned decode pass is
 *   `layer0/pinned/decode0|pinned|recurrent`.
 *   It is NOT the chunked scan of `prototype/gdn/gdn_kernels.h`, and the two are not interchangeable
 *   pieces of evidence: `speccheck/gdn_ref.band_licenses` grants `chunked|pinned` at NO decode pass
 *   (the dispatch table gives a cached decode step the recurrent program, so the floor probe never
 *   measured a chunked arm there — `prototype/gdn/HANDOFF.md` §SLICE 3 JOB 2, adjudicated UNLICENSED
 *   and left that way). `m2_d_io.h` therefore REFUSES a plan whose `form=` word is anything but
 *   `recurrent` (`form-not-implemented`), and `engine/block_device.py` prints the form on every run
 *   and reddens a row credited under a pair whose form this arm does not implement
 *   (control `gdn-form-not-merged`). Implementing this form earns nothing for the other.
 *
 * WHO OWNS WHICH ARITHMETIC — read this before trusting a row.
 *   Included VERBATIM from `prototype/gdn/gdn_kernels.h` (via -I prototype/gdn, never copied):
 *   `gdn_bf16_of_f32` / `gdn_f32_of_bf16` / `gdn_bf16_rt` (round-to-nearest-even, checked against
 *   torch over 200k f32 patterns by that lane's self-test), `gdn_sumsq` (the declared pairwise f32
 *   tree of D-23 — the reduction shape that put `q_hat` inside its band), `gdn_silu`, and the geometry
 *   constants `GDN_H/GDN_H_K/GDN_H_V/GDN_D_K/GDN_D_V/GDN_EXP`, `GDN_L2_EPS`, `GDN_RMS_EPS`.
 *   ENGINE-NEW, i.e. the arithmetic no prototype gate vouches for: the recurrent step itself. The
 *   prototype lane implements the chunked scan (UT system, two solves, intra-chunk attention); a
 *   cached decode step has none of those, and its five steps are written here as `m2d_item_*`. The
 *   state contraction is `m2d_state_dot`, the one place the incoming state is contracted, so the
 *   `state_stored_transposed` question has exactly one answer in this file — and exactly one place
 *   for the control to bend it.
 *
 * WHY THIS ARM READS PINNED ACTIVATIONS AND NOT RESIDENT POLICY-A BYTES. The licensed GDN rows are
 * cross-form program error on the BF16 width: `band_licenses` grants the decode rows to the bf16 arm
 * only, and `engine/block_gate.py::adjudicate` answers a policy-A GDN row with
 * `policy-a-arm-outside-band-population`. So the arm's inputs are the pinned decode-step field
 * records (q_pre, k_pre, v_pre, beta, g_decay, z) plus the pinned prefill state, read through
 * `engine/block_forward.py::Dumps` (sha256-verified per read), and the rows are this arm against the
 * pinned decode records of the same pass. Reading Q8_0 projections instead would produce rows no
 * licence can judge. No width, band, floor or tolerance is authored here.
 *
 * NO ATOMICS AND NO BARRIERS IN THIS FILE — the same boundary `m2_se_kernels.h` and
 * `m2_g_kernels.h` draw: every function here is `__device__ __host__`, so a `__global__` body folds
 * tallies with `atomicAdd` and a host path folds with plain `+=`. The stages are separate launches;
 * a stage's inputs are the previous launch's outputs, so no barrier is needed.
 *
 * DECLARED ARITHMETIC (the two compiled arms must agree bit for bit, so the order lives in the
 * source): every reduction runs its index ASCENDING with one f32 accumulator — except `gdn_sumsq`,
 * which is the declared pairwise tree it inherits — and nothing splits, regroups or fuses
 * (`-ffp-contract=off -fno-fast-math` on both arms). `expf`/`sqrt`/`/` are the plain calls. The q
 * scale uses the RECURRENT arm's divide spelling (`q / (d_k**0.5)`), not the chunked arm's multiply:
 * `speccheck/gdn_ref.py::Form.named("recurrent")` is `q_scale="div"`, and D-23's note says the two
 * spellings are not the same product tree. The l2norm and the gated-norm reciprocal roots are spelled
 * `(float)(1.0/sqrt((double)(x+eps)))`, the one spelling measured to reproduce `torch.rsqrt` on
 * this geometry (`prototype/gdn/gdn_kernels.h` §item A).
 *
 * NO TIMER, NO RATE: nothing here reads a clock (`pitfalls/00`).
 */
#ifndef M2_D_KERNELS_H
#define M2_D_KERNELS_H

#include <stdint.h>
#include <string.h>

#include "gdn_kernels.h"                   /* via -I prototype/gdn; included, never copied */

/* ---- the form this translation unit implements -------------------------------------------------- */
#define M2D_FORM_RECURRENT 0
#define M2D_FORM_NAME(f) ((f) == M2D_FORM_RECURRENT ? "recurrent" : "unknown")

/* ---- the mistake words the controls bend, and the faults that make a census non-zero ----------- *
 * `state_xpose` is the oracle's own injection `state_stored_transposed`: `d_k == d_v == 128`, so
 * [d_k,d_v] and [d_v,d_k] are the same 16,384 words per head, no shape changes and nothing raises —
 * which is exactly why the read has to be a named control and not a type error. `decay_head_off1` is
 * the carry-index mistake: the state's decay multiplier is a per-value-head scalar, and taking it
 * from the neighbouring head is the one-step version of reading the wrong head's history. */
#define M2D_MODE_CORRECT         0
#define M2D_MODE_STATE_XPOSE     1
#define M2D_MODE_DECAY_HEAD_OFF1 2

#define M2D_FAULT_NONE        0
#define M2D_FAULT_SKIP_CARRY  1     /* the carry stage forbidden: the state field stays poison    */
#define M2D_FAULT_SKIP_OUT    2     /* the output stage forbidden: scan_out/core_attn stay poison */
#define M2D_FAULT_ZERO_STATE  3     /* non-carried: the state bytes are never read                 */
#define M2D_FAULT_ALLOC_OVER  4     /* device-only: an allocation no pool can satisfy              */

/* ---- the record: fp32 words throughout; a bf16-carrying field is stored WIDENED, the repo's
 * convention, so the comparator compares 4-byte patterns and never guesses a dtype off a file.
 * Field order is the oracle's stage order (`speccheck/gdn_ref.SCAN_WIDTH_STAGES`), which is what lets
 * `block_gate.stage_row` be called with the oracle's own stage string and no translation layer. -- */
#define M2D_OFF_Q_HAT   0
#define M2D_OFF_K_HAT   (M2D_OFF_Q_HAT + GDN_H_V * GDN_D_K)
#define M2D_OFF_SCAN    (M2D_OFF_K_HAT + GDN_H_V * GDN_D_K)
#define M2D_OFF_CORE    (M2D_OFF_SCAN + GDN_H_V * GDN_D_V)
#define M2D_OFF_STATE   (M2D_OFF_CORE + GDN_H_V * GDN_D_V)
#define M2D_OFF_NORMED  (M2D_OFF_STATE + GDN_H_V * GDN_D_K * GDN_D_V)
#define M2D_OFF_Y       (M2D_OFF_NORMED + GDN_H_V * GDN_D_V)
#define M2D_FLOATS      (M2D_OFF_Y + GDN_H)

#define M2D_FIELD_COUNT 7
static const char *const M2D_FIELD_NAME[M2D_FIELD_COUNT] = {
    "q_hat", "k_hat", "scan_out", "core_attn", "state", "normed", "y"
};
static const int M2D_FIELD_OFF[M2D_FIELD_COUNT] = {
    M2D_OFF_Q_HAT, M2D_OFF_K_HAT, M2D_OFF_SCAN, M2D_OFF_CORE, M2D_OFF_STATE, M2D_OFF_NORMED,
    M2D_OFF_Y
};
static const int M2D_FIELD_LEN[M2D_FIELD_COUNT] = {
    GDN_H_V * GDN_D_K, GDN_H_V * GDN_D_K, GDN_H_V * GDN_D_V, GDN_H_V * GDN_D_V,
    GDN_H_V * GDN_D_K * GDN_D_V, GDN_H_V * GDN_D_V, GDN_H
};
/* 1 when the field's stored word carries a bf16 image (widened), 0 when it is a true f32 field.
 * The driver asserts this against the dump manifest instead of inferring a dtype from a file size
 * (`pitfalls/30` §dtype-by-filesize), which is the same distinction `prototype/gdn/gdn_case.h` keeps. */
static const int M2D_FIELD_IS_BF16[M2D_FIELD_COUNT] = { 0, 0, 0, 1, 0, 1, 1 };

/* ---- the input fields: six decode-step activations, two weights, one carried state ------------- *
 * Words and element widths are derived from the geometry in ONE place; the plan has to agree with
 * this table or `m2_d_io.h` refuses it by name, so a field whose width or shape moved is a refusal
 * and not a plausible row. `elt` is the STORED width: bf16 fields are 2-byte bit images, `g_decay`
 * and the recurrent state are f32. */
#define M2D_F_Q_PRE   0
#define M2D_F_K_PRE   1
#define M2D_F_V_PRE   2
#define M2D_F_BETA    3
#define M2D_F_G_DECAY 4
#define M2D_F_Z       5
#define M2D_F_NORM_W  6
#define M2D_F_OUT_W   7
#define M2D_F_STATE   8
#define M2D_F_COUNT   9
static const char *const M2D_FIELD_IN_NAME[M2D_F_COUNT] = {
    "q_pre", "k_pre", "v_pre", "beta", "g_decay", "z", "norm_w", "out_w", "state_in"
};
static const long M2D_FIELD_IN_WORDS[M2D_F_COUNT] = {
    (long)GDN_H_K * GDN_D_K, (long)GDN_H_K * GDN_D_K, (long)GDN_H_V * GDN_D_V, (long)GDN_H_V,
    (long)GDN_H_V, (long)GDN_H_V * GDN_D_V, (long)GDN_D_V,
    (long)GDN_H * GDN_H_V * GDN_D_V, (long)GDN_H_V * GDN_D_K * GDN_D_V
};
static const int M2D_FIELD_IN_ELT[M2D_F_COUNT] = { 2, 2, 2, 2, 4, 2, 2, 2, 4 };

/* ---- the work decomposition: items, and their counts derived from geometry --------------------- *
 * `m2d_item_count` is the denominator the driver declares BEFORE a run and reconciles against the
 * arm's own tally (`pitfalls/40` §an unnamed denominator). One item = one output row of one stage,
 * which is also the device's thread shape, so neither path can tally two units into one item. */
enum M2DItem { M2D_IT_ENTER = 0, M2D_IT_CARRY, M2D_IT_KVMEM, M2D_IT_WRITE, M2D_IT_OUT,
               M2D_IT_NORM, M2D_IT_OUTPROJ, M2D_IT_N };
static const char *const M2D_ITEM_NAME[M2D_IT_N] = {
    "enter", "carry", "kvmem", "write", "out", "norm", "outproj"
};
static inline long m2d_item_count(int item) {
    switch (item) {
        case M2D_IT_ENTER:  return GDN_H_V;                       /* one value head     */
        case M2D_IT_CARRY:  return (long)GDN_H_V * GDN_D_K;        /* one (head, k row)  */
        case M2D_IT_KVMEM:  return (long)GDN_H_V * GDN_D_V;        /* one (head, v column) */
        case M2D_IT_WRITE:  return (long)GDN_H_V * GDN_D_K;
        case M2D_IT_OUT:    return (long)GDN_H_V * GDN_D_V;
        case M2D_IT_NORM:   return GDN_H_V;
        default:            return GDN_H;                          /* one out_proj row   */
    }
}
static inline long m2d_items_total(void) {
    long s = 0;
    int i;
    for (i = 0; i < M2D_IT_N; ++i) s += m2d_item_count(i);
    return s;
}

/* ---- the plan ---------------------------------------------------------------------------------- */
struct M2DPlan {
    int form;                     /* M2D_FORM_RECURRENT, and the only value the checker admits */
    int mode;                     /* M2D_MODE_* */
    int fault;                    /* M2D_FAULT_* */
    long off_in[M2D_F_COUNT];     /* byte offset of each input field inside the slab */
    long bytes_in[M2D_F_COUNT];
    long read_off; int read_len;  /* the device read-back window (the state's first words) */
    long slab_bytes;
};

/* ---- the counters: caller-owned, folded by the caller ----------------------------------------- *
 * `state_*` is the fingerprint of the bytes the carry stage actually read, and it is the object of
 * the control that asks "did the arm read the state it claims to read?". A modular sum and a
 * modular index-weighted sum, because modular addition is associative and commutative: the device
 * can fold them with `atomicAdd` in any order and still agree bit for bit with the host's `+=`,
 * which is what lets the fingerprint be a cross-compilation fact rather than an accident of order.
 * `state_nonzero` counts the live words of the incoming state — the counter the prototype lane made
 * load-bearing for the same reason (`prototype/gdn/gdn_arm.h::main`, `st_state_nonzero`): a case
 * cannot claim a live state it did not receive, and a counter that can only ever read 0 is not a
 * guard (`pitfalls/40` §a check that can produce no output cannot fail). */
struct M2DCounters {
    uint32_t items[M2D_IT_N], items_total;
    uint32_t reads_q, reads_k, reads_v, reads_state, writes_state;
    uint32_t state_nonzero, state_sum, state_sumidx;
    uint32_t out_elems, outproj_reads;
};

static inline void m2d_counters_fold(struct M2DCounters *ct, const struct M2DCounters *ad) {
    int i;
    if (!ct || !ad) return;
    for (i = 0; i < M2D_IT_N; ++i) ct->items[i] += ad->items[i];
    ct->items_total += ad->items_total;
    ct->reads_q += ad->reads_q;      ct->reads_k += ad->reads_k;
    ct->reads_v += ad->reads_v;      ct->reads_state += ad->reads_state;
    ct->writes_state += ad->writes_state;
    ct->state_nonzero += ad->state_nonzero;
    ct->state_sum += ad->state_sum;              /* modular: order-independent */
    ct->state_sumidx += ad->state_sumidx;
    ct->out_elems += ad->out_elems;  ct->outproj_reads += ad->outproj_reads;
}

/* ---- the poison, and the unwritten-slot census -------------------------------------------------- *
 * The prototype lane's NaN pattern (quiet NaN, 0x7FC00000), so an unwritten slot means the same
 * thing in both lanes and a `hipMemset` that cleared it is visible as a change rather than as a 0. */
#define M2D_NAN_BITS 0x7FC00000u

GDNDAY_DEV static inline float m2d_nan(void) {
    const uint32_t u = M2D_NAN_BITS;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
GDNDAY_DEV static inline uint32_t m2d_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
GDNDAY_DEV static inline void m2d_poison(float *out) {
    const float nan = m2d_nan();
    int i;
    for (i = 0; i < M2D_FLOATS; ++i) out[i] = nan;
}
GDNDAY_DEV static inline int m2d_field_nan(const float *out, int field) {
    const int off = M2D_FIELD_OFF[field], len = M2D_FIELD_LEN[field];
    int i, n = 0;
    for (i = 0; i < len; ++i)
        if ((m2d_bits(out[off + i]) & 0x7FFFFFFFu) == M2D_NAN_BITS) n++;
    return n;
}

/* ---- slab element access: one place per stored width ------------------------------------------ *
 * A bf16 field is a 16-bit BIT IMAGE read little-endian and widened by a left shift, never a numeric
 * cast of an integer (`pitfalls/30` §Tensor.to casts, .view bitcasts — the bug class that turned a
 * positive signalling NaN into +inf in the prototype's first cast). `g_decay` and the state are f32
 * and are bitcast. All addressing is byte-offset arithmetic. */
GDNDAY_DEV static inline float m2d_in_bf16(const uint8_t *slab, long off, long idx) {
    const uint8_t *q = slab + off + 2 * idx;
    return gdn_f32_of_bf16((uint16_t)((uint32_t)q[0] | ((uint32_t)q[1] << 8)));
}
GDNDAY_DEV static inline float m2d_in_f32(const uint8_t *slab, long off, long idx) {
    const uint8_t *q = slab + off + 4 * idx;
    const uint32_t u = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16)
                     | ((uint32_t)q[3] << 24);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
GDNDAY_DEV static inline uint32_t m2d_in_f32_bits(const uint8_t *slab, long off, long idx) {
    const uint8_t *q = slab + off + 4 * idx;
    return (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
}
GDNDAY_DEV static inline void m2d_put_bf16(float *slot, uint16_t bits) { *slot = gdn_f32_of_bf16(bits); }
/* A bf16-carrying record slot is READ as a float, because the record stores bf16 fields widened (see
 * M2D_FIELD_IS_BF16): the value in the word is already the bf16 number, so re-rounding it to bf16 and
 * re-widening is a no-op, and reading the raw word as if it were the bit pattern is not — it hands back
 * a denormal made from the exponent bits, which is the mistake this comment exists to prevent. */

/* THE state contraction — the one place the incoming state is contracted, and the one place the
 * `state_stored_transposed` injection is spelled. Honest: S is [d_k, d_v] and `sum_k a[k]*S[k,col]`.
 * Injected: S read as [d_v, d_k] (`sum_k a[k]*S[col,k]`), which is in-bounds, shape-legal and a
 * different number at every live word — the two-self-consistent-layouts trap at tensor level. */
GDNDAY_DEV static inline float m2d_state_dot(const float *a, const float *S, long sbase, int col,
                                             int mode) {
    float acc = 0.0f;
    int k;
    if (mode == M2D_MODE_STATE_XPOSE) {
        for (k = 0; k < GDN_D_K; ++k) acc += a[k] * S[sbase + (long)col * GDN_D_K + k];
        return acc;
    }
    for (k = 0; k < GDN_D_K; ++k) acc += a[k] * S[sbase + (long)k * GDN_D_V + col];
    return acc;
}

/* ---- scratch: what one run needs between stages -------------------------------------------------- *
 * `v_w` is the bf16 `v_pre` widened once at entry so the delta stage does not re-decode it; `beta`
 * is the per-head gate scalar, staged once instead of re-read per column; `kv` and `delta` are the
 * two per-(head, v-column) temporaries §04b.1 names. q_hat/k_hat live in the record, because they
 * ARE emitted stages. */
struct M2DScratch {
    float *v_w;      /* [h_v, d_v] */
    float *kv;       /* [h_v, d_v] */
    float *delta;    /* [h_v, d_v] */
    float *beta;     /* [h_v]      */
};

/* ============================================================================================== *
 * STAGE 1 — enter: §04.4's l2norms and the recurrent arm's q scale, one value head per item.
 * V head h reads K head h/EXP (the reference's `repeat_interleave(expansion, dim=2)`). The scan
 * dtype is f32, the stored images are bf16, and the widening is a shift.
 * ============================================================================================== */
GDNDAY_DEV static inline void m2d_item_enter(const struct M2DPlan *p, const uint8_t *slab,
                                             float *rec, struct M2DScratch *sc, int h,
                                             struct M2DCounters *ct) {
    const int hk = h / GDN_EXP;
    const long krow = (long)hk * GDN_D_K, vrow = (long)h * GDN_D_V;
    const long oq = (long)h * GDN_D_K, ov = (long)h * GDN_D_V;
    float qa[GDN_D_K], ka[GDN_D_K];
    const float qs = (float)sqrt((double)GDN_D_K);     /* the recurrent arm DIVIDES by sqrt(d_k) */
    float sumq, sumk, rq, rk;
    int i;
    for (i = 0; i < GDN_D_K; ++i) {
        qa[i] = m2d_in_bf16(slab, p->off_in[M2D_F_Q_PRE], krow + i);
        ka[i] = m2d_in_bf16(slab, p->off_in[M2D_F_K_PRE], krow + i);
    }
    sumq = gdn_sumsq(qa, GDN_D_K);
    sumk = gdn_sumsq(ka, GDN_D_K);
    rq = (float)(1.0 / sqrt((double)(sumq + GDN_L2_EPS)));   /* torch.rsqrt, see gdn_kernels.h */
    rk = (float)(1.0 / sqrt((double)(sumk + GDN_L2_EPS)));
    sc->beta[h] = m2d_in_bf16(slab, p->off_in[M2D_F_BETA], h);
    for (i = 0; i < GDN_D_K; ++i) {
        rec[M2D_OFF_Q_HAT + oq + i] = (qa[i] * rq) / qs;
        rec[M2D_OFF_K_HAT + oq + i] = ka[i] * rk;
    }
    for (i = 0; i < GDN_D_V; ++i)
        sc->v_w[ov + i] = m2d_in_bf16(slab, p->off_in[M2D_F_V_PRE], vrow + i);
    if (ct) {
        ct->items[M2D_IT_ENTER] = 1u;
        ct->items_total = 1u;
        ct->reads_q += (uint32_t)GDN_D_K;
        ct->reads_k += (uint32_t)GDN_D_K;
        ct->reads_v += (uint32_t)GDN_D_V;
    }
}

/* ============================================================================================== *
 * STAGE 2 — carry: §04b.1 step 1, `S = S_old * exp(decay_head)`, one (head, k-row) per item.
 * This is also the ONLY stage that reads every word of the incoming state, so it is where the state
 * fingerprint is taken: `state_sum`/`state_sumidx`/`state_nonzero`/`reads_state` cover exactly the
 * words of the input state, once each, and the driver reconciles them against its own count over
 * the bytes it put in the slab. `zero_in` is the non-carried leg executed by the arm itself: the
 * state bytes are then never read, the fingerprint stays 0, and the driver's reconciliation is what
 * turns that into a finding instead of a silently cheap step.
 * ============================================================================================== */
GDNDAY_DEV static inline void m2d_item_carry(const struct M2DPlan *p, const uint8_t *slab,
                                             float *rec, int h, int row, int zero_in,
                                             struct M2DCounters *ct) {
    const long sbase = (long)h * GDN_D_K * GDN_D_V + (long)row * GDN_D_V;
    const int dh = (p->mode == M2D_MODE_DECAY_HEAD_OFF1) ? (h + 1) % GDN_H_V : h;
    const float e = expf(m2d_in_f32(slab, p->off_in[M2D_F_G_DECAY], dh));
    int v;
    for (v = 0; v < GDN_D_V; ++v) {
        const long g = sbase + v;
        float f = 0.0f;
        if (!zero_in) {
            const uint32_t bits = m2d_in_f32_bits(slab, p->off_in[M2D_F_STATE], g);
            memcpy(&f, &bits, 4);
            if (ct) {
                if (f != 0.0f) ct->state_nonzero += 1u;
                ct->state_sum += bits;
                ct->state_sumidx += bits * (uint32_t)(g + 1);
                ct->reads_state += 1u;
            }
        }
        rec[M2D_OFF_STATE + g] = f * e;
        if (ct) ct->writes_state += 1u;
    }
    if (ct) { ct->items[M2D_IT_CARRY] = 1u; ct->items_total = 1u; }
}

/* ============================================================================================== *
 * STAGES 3-5 — §04b.1 steps 2, 4 and 5: the read that forms the delta, the rank-one write, and the
 * output read. One (head, v-column) or (head, k-row) per item; every contraction ascending.
 * ============================================================================================== */
GDNDAY_DEV static inline void m2d_item_kvmem(const struct M2DPlan *p, const float *rec,
                                             struct M2DScratch *sc, int h, int col,
                                             struct M2DCounters *ct) {
    const long sbase = (long)h * GDN_D_K * GDN_D_V;
    const long o = (long)h * GDN_D_V + col;
    const float kv = m2d_state_dot(rec + M2D_OFF_K_HAT + (long)h * GDN_D_K, rec,
                                   M2D_OFF_STATE + sbase, col, p->mode);
    sc->kv[o] = kv;
    sc->delta[o] = (sc->v_w[o] - kv) * sc->beta[h];
    if (ct) { ct->items[M2D_IT_KVMEM] = 1u; ct->items_total = 1u; }
}

GDNDAY_DEV static inline void m2d_item_write(float *rec, const struct M2DScratch *sc,
                                             int h, int row, struct M2DCounters *ct) {
    const long sbase = (long)h * GDN_D_K * GDN_D_V + (long)row * GDN_D_V;
    const float k = rec[M2D_OFF_K_HAT + (long)h * GDN_D_K + row];
    const float *dl = sc->delta + (long)h * GDN_D_V;
    int v;
    for (v = 0; v < GDN_D_V; ++v) rec[M2D_OFF_STATE + sbase + v] += k * dl[v];
    if (ct) { ct->items[M2D_IT_WRITE] = 1u; ct->items_total = 1u; }
}

GDNDAY_DEV static inline void m2d_item_out(const struct M2DPlan *p, const float *rec, float *out,
                                           int h, int col, struct M2DCounters *ct) {
    const long sbase = (long)h * GDN_D_K * GDN_D_V;
    const long o = (long)h * GDN_D_V + col;
    const float acc = m2d_state_dot(rec + M2D_OFF_Q_HAT + (long)h * GDN_D_K, rec,
                                    M2D_OFF_STATE + sbase, col, p->mode);
    out[M2D_OFF_SCAN + o] = acc;
    m2d_put_bf16(out + M2D_OFF_CORE + o, gdn_bf16_of_f32(acc));   /* §04.6's exit cast */
    if (ct) { ct->items[M2D_IT_OUT] = 1u; ct->items_total = 1u; ct->out_elems += 1u; }
}

/* ============================================================================================== *
 * STAGES 6-7 — §04.7's gated norm (both bf16 roundings, transcribed as the prototype transcribes it)
 * and §04.8's out_proj. Same spellings as `gdn_item_norm`/`gdn_item_outproj`, over the decode step's
 * single token. Those two stages are unlicensed at this pass for EITHER arm pair, so they are printed
 * as measured residue and nothing here leans on them.
 * ============================================================================================== */
GDNDAY_DEV static inline void m2d_item_norm(const struct M2DPlan *p, const uint8_t *slab,
                                            const float *rec, float *out, int h,
                                            struct M2DCounters *ct) {
    const long base = (long)h * GDN_D_V;
    float x[GDN_D_V];
    float var, rs;
    int i;
    for (i = 0; i < GDN_D_V; ++i) x[i] = rec[M2D_OFF_CORE + base + i];
    var = gdn_sumsq(x, GDN_D_V) / (float)GDN_D_V;
    rs = (float)(1.0 / sqrt((double)(var + GDN_RMS_EPS)));
    for (i = 0; i < GDN_D_V; ++i) {
        const float w = m2d_in_bf16(slab, p->off_in[M2D_F_NORM_W], i);
        const float num = gdn_bf16_rt(w * gdn_bf16_rt(x[i] * rs));
        const float g = gdn_silu(m2d_in_bf16(slab, p->off_in[M2D_F_Z], base + i));
        m2d_put_bf16(out + M2D_OFF_NORMED + base + i, gdn_bf16_of_f32(num * g));
    }
    if (ct) { ct->items[M2D_IT_NORM] = 1u; ct->items_total = 1u; }
}

GDNDAY_DEV static inline void m2d_item_outproj(const struct M2DPlan *p, const uint8_t *slab,
                                               const float *rec, float *out, int orow,
                                               struct M2DCounters *ct) {
    float acc = 0.0f;
    int j;
    for (j = 0; j < GDN_H_V * GDN_D_V; ++j)
        acc += rec[M2D_OFF_NORMED + j]
             * m2d_in_bf16(slab, p->off_in[M2D_F_OUT_W], (long)orow * GDN_H_V * GDN_D_V + j);
    m2d_put_bf16(out + M2D_OFF_Y + orow, gdn_bf16_of_f32(acc));
    if (ct) {
        ct->items[M2D_IT_OUTPROJ] = 1u;
        ct->items_total = 1u;
        ct->outproj_reads += (uint32_t)(GDN_H_V * GDN_D_V);
    }
}

/* ---- the scratch size, spelled once so no two callers disagree ---------------------------------- */
static inline long m2d_scratch_floats(void) {
    return 3L * GDN_H_V * GDN_D_V + GDN_H_V;
}

/* ---- the whole step, serial: the order of acts on a host path ------------------------------------ *
 * The device reaches the same items from seven launches instead (`m2_d_arm.hip`), which is why the
 * ITEMS and not this loop are the arithmetic: this function is what `m2_d_host.cpp` runs, and
 * `m2_d_arm.hip` never calls it. Every item gets a zeroed counter struct and is folded, so the
 * per-stage tallies mean "items executed", one unit at a time, identically on both paths.
 *
 * The faults are executed HERE, so both paths can run them: `skip-carry` forbids the carry stage
 * (the state field stays poison and the state fingerprint stays zero), `skip-out` forbids the output
 * stage (scan_out and core_attn stay poison), and `zero-state` runs the carry stage over a state the
 * arm does not read — the non-carried leg, which is what a silently-dropped carry looks like. */
static inline void m2d_run(const struct M2DPlan *p, const uint8_t *slab, float *out,
                           struct M2DScratch *sc, struct M2DCounters *ct) {
    struct M2DCounters one;
    const int zero_in = (p->fault == M2D_FAULT_ZERO_STATE);
    int h, row, col;
    memset(ct, 0, sizeof(*ct));
    m2d_poison(out);
    for (h = 0; h < GDN_H_V; ++h) {
        memset(&one, 0, sizeof(one));
        m2d_item_enter(p, slab, out, sc, h, &one);
        m2d_counters_fold(ct, &one);
    }
    if (p->fault == M2D_FAULT_SKIP_CARRY) {
        /* The carry stage is forbidden: no state word is read, the field stays poison, and the
         * fingerprint stays zero. The stages downstream then run over poison, which is the point — a
         * step that reads a state it never loaded must be visible as one, not absorbed. */
    } else {
        for (h = 0; h < GDN_H_V; ++h)
            for (row = 0; row < GDN_D_K; ++row) {
                memset(&one, 0, sizeof(one));
                m2d_item_carry(p, slab, out, h, row, zero_in, &one);
                m2d_counters_fold(ct, &one);
            }
    }
    for (h = 0; h < GDN_H_V; ++h)
        for (col = 0; col < GDN_D_V; ++col) {
            memset(&one, 0, sizeof(one));
            m2d_item_kvmem(p, out, sc, h, col, &one);
            m2d_counters_fold(ct, &one);
        }
    for (h = 0; h < GDN_H_V; ++h)
        for (row = 0; row < GDN_D_K; ++row) {
            memset(&one, 0, sizeof(one));
            m2d_item_write(out, sc, h, row, &one);
            m2d_counters_fold(ct, &one);
        }
    if (p->fault != M2D_FAULT_SKIP_OUT)
        for (h = 0; h < GDN_H_V; ++h)
            for (col = 0; col < GDN_D_V; ++col) {
                memset(&one, 0, sizeof(one));
                m2d_item_out(p, out, out, h, col, &one);
                m2d_counters_fold(ct, &one);
            }
    for (h = 0; h < GDN_H_V; ++h) {
        memset(&one, 0, sizeof(one));
        m2d_item_norm(p, slab, out, out, h, &one);
        m2d_counters_fold(ct, &one);
    }
    for (row = 0; row < GDN_H; ++row) {
        memset(&one, 0, sizeof(one));
        m2d_item_outproj(p, slab, out, out, row, &one);
        m2d_counters_fold(ct, &one);
    }
}

/* What the census has to be able to refuse, derived rather than typed: the state words one decode
 * step must read once, the element outputs the scan must produce, and the total item count. The
 * driver requires the arm's own numbers to equal these on an honest run, so a stage that ran over
 * the wrong extent cannot report a clean census. */
static inline long m2d_want_state_words(void) { return (long)GDN_H_V * GDN_D_K * GDN_D_V; }
static inline long m2d_want_out_elems(void)   { return (long)GDN_H_V * GDN_D_V; }

GDNDAY_DEV static inline int m2d_mode_of(const char *w) {
    if (!strcmp(w, "correct")) return M2D_MODE_CORRECT;
    if (!strcmp(w, "state_xpose")) return M2D_MODE_STATE_XPOSE;
    if (!strcmp(w, "decay_head_off1")) return M2D_MODE_DECAY_HEAD_OFF1;
    return -1;
}
GDNDAY_DEV static inline int m2d_fault_of(const char *w) {
    if (!strcmp(w, "none")) return M2D_FAULT_NONE;
    if (!strcmp(w, "skip-carry")) return M2D_FAULT_SKIP_CARRY;
    if (!strcmp(w, "skip-out")) return M2D_FAULT_SKIP_OUT;
    if (!strcmp(w, "zero-state")) return M2D_FAULT_ZERO_STATE;
    if (!strcmp(w, "alloc-oversize")) return M2D_FAULT_ALLOC_OVER;
    return -1;
}
/* Which record fields a mode or fault can move, as a reach table the driver checks BEFORE running:
 * a control that asks for a leg a mode cannot produce is refused, rather than an inert mode being
 * reported as a fired control. `q_hat`/`k_hat` read no state, so a state mode CANNOT move them, and
 * that is what makes "q_hat stayed bit-identical" a structural fact rather than a lucky result. */
GDNDAY_DEV static inline const char *m2d_mode_fields(int mode) {
    return mode == M2D_MODE_CORRECT ? "none" : "scan_out,core_attn,state,normed,y";
}
GDNDAY_DEV static inline const char *m2d_fault_fields(int fault) {
    if (fault == M2D_FAULT_SKIP_CARRY) return "state,scan_out,core_attn,normed,y";
    if (fault == M2D_FAULT_SKIP_OUT) return "scan_out,core_attn,normed,y";
    if (fault == M2D_FAULT_ZERO_STATE) return "scan_out,core_attn,state,normed,y";
    return "none";
}

#endif /* M2_D_KERNELS_H */
