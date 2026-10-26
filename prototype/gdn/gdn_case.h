/* gdn_case.h — the plan, the item runner, the load census and the counters, in ONE file read by
 * every arm (g++ host twin, hipcc device TU run on CPU) so a green cross-arm comparison compares
 * compilers and not two drivers. Nothing here reads the oracle or the dumps: the arms read
 * `work/inputs.bin` + `work/weights.bin` and write `work/<arm>.out.bin` + `work/<arm>.echo.bin`,
 * and print one `CASE` line plus a `RUN` and a `RESULT` line. `gate.py` parses those and asserts
 * parsed == printed == derived.
 *
 * THE PLAN CARRIES STATE (slice 2). A case may name `carry=<idx>`: its incoming state is then the
 * `state` stage that case <idx> produced IN THIS RUN, not the state block in inputs.bin. That is
 * what a cached decode step is — chunk n reads the buffer chunk n-1 wrote — and it makes the carry
 * chain the arm's own behaviour rather than something the driver arranges behind the arm's back.
 * Rules the arm ENFORCES, because a wrong carry is a silently plausible number: the source case
 * must exist, must precede this one, and must be the same block (a state is per-block, per-head;
 * carrying one across blocks is a different head's history). Every case in a plan must also name
 * the SAME `T` as the plan header, because one stride addresses every case.
 *
 * TWO NAME SPACES, deliberately separated:
 *   ITEMS  are this file's work decomposition (enter, cum, pair, rowpost, solve, vnew, core,
 *          state, cast, norm, outproj). They exist so an item count can be derived and checked.
 *   STAGES are the seven names `speccheck/gdn_ref.py` STAGES uses for the tensors this slice emits
 *          — q_hat, k_hat, scan_out, core_attn, state, normed, y. Every emitted word belongs to
 *          exactly one of them, in the order below, which is what lets `band_check_row(band,
 *          stage, ag)` be called with the oracle's own stage string and no translation layer.
 *
 * NO TIMER IS READ ANYWHERE IN THIS FILE. Every counter is a count (`AGENTS.md` §GPU execution;
 * `pitfalls/00-measurement.md`).
 */
#ifndef GDN_CASE_H_
#define GDN_CASE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gdn_kernels.h"

#define GDN_MAX_CASES 16

/* ---- the seven emitted stages, in output order ---------------------------------------------- */
enum GdnStage { GS_Q_HAT = 0, GS_K_HAT, GS_SCAN_OUT, GS_CORE_ATTN, GS_STATE, GS_NORMED, GS_Y,
                GS_N };
static const char *const gdn_stage_name[GS_N] = {
    "q_hat", "k_hat", "scan_out", "core_attn", "state", "normed", "y"
};
/* 1 when the stage's emitted word holds a bf16 BIT IMAGE in its low half, 0 when it is f32.
 * Read by gate.py as the dtype of each stage, and asserted there against the dump manifest — the
 * dtype is never inferred from the file size on this side of the seam. */
static const int gdn_stage_is_bf16[GS_N] = { 0, 0, 0, 1, 0, 1, 1 };

GDNDAY_DEV static inline long gdn_stage_n(int stage, int T) {
    long t = (long)T;
    switch (stage) {
        case GS_Q_HAT: case GS_K_HAT: return t * GDN_H_V * GDN_D_K;
        case GS_SCAN_OUT: case GS_CORE_ATTN: case GS_NORMED: return t * GDN_H_V * GDN_D_V;
        case GS_STATE: return (long)GDN_H_V * GDN_D_K * GDN_D_V;
        default: return t * GDN_H;
    }
}
GDNDAY_DEV static inline long gdn_stage_words(int T) {
    long s = 0;
    for (int st = 0; st < GS_N; ++st) s += gdn_stage_n(st, T);
    return s;
}
/* word offset of stage `stage` inside one case's output block — derived in ONE place so no two
 * arms, and no arm and no driver, can disagree about where a stage lives */
GDNDAY_DEV static inline long gdn_stage_off(int T, int stage) {
    long off = 0;
    for (int s = 0; s < stage; ++s) off += gdn_stage_n(s, T);
    return off;
}

/* ---- the input layout: the eight loaded fields, then the zero initial state ------------------ */
GDNDAY_DEV static inline long gdn_field_off(int T, int field) {
    long off = 0;
    for (int f = 0; f < field; ++f) off += gdn_field_n(f, T);
    return off;
}
GDNDAY_DEV static inline long gdn_in_words(int T) {
    /* the SIX scan-input fields live in inputs.bin; norm_weight and out_weight live in
     * weights.bin, because two blocks share nothing and one file per kind is what makes a
     * crossed weight a refusal instead of an off-by-one field offset. */
    long s = 0;
    for (int f = 0; f <= GF_Z; ++f) s += gdn_field_n(f, T);
    return s + (long)GDN_H_V * GDN_D_K * GDN_D_V;
}
/* the fields echoed element by element, and the one that is fingerprinted instead: out_weight is
 * 8,388,608 words per case and echoing it per case would cost more than the arithmetic does. Its
 * exactness check is a per-element 64-bit sum plus a positional sum plus the count, which a wrong
 * offset, stride or dtype changes, and gate.py witnesses it by perturbing one word in memory. */
#define GDN_ECHO_N (GF_OUT_W)

/* the scratch the items need, in u32 words, spelled with the SAME expressions `gdn_bind` walks so
 * the per-case block size and the pointer walk cannot disagree */
GDNDAY_DEV static inline long gdn_case_scratch_words(int T) {
    (void)T;
    return (long)GDN_H_V * GDN_C                       /* cum   */
         + (long)GDN_H_V * GDN_C * GDN_C * 2           /* ut, ia */
         + (long)GDN_H_V * GDN_C * GDN_D_K * 7         /* qhat, khat, kb, dcb, kcd, qdec, kdec */
         + (long)GDN_H_V * GDN_C * GDN_D_V * 4;        /* vpad, vb, newv, vnew */
}

/* ---- the compute items ----------------------------------------------------------------------- */
enum GdnItem { IT_ENTER = 0, IT_CUM, IT_PAIR, IT_ROWPOST, IT_SOLVE, IT_VNEW, IT_CORE, IT_STATE,
               IT_CAST, IT_NORM, IT_OUTPROJ, IT_N };
static const char *const gdn_item_name[IT_N] = {
    "enter", "cum", "pair", "rowpost", "solve", "vnew", "core", "state", "cast", "norm", "outproj"
};

/* Items per item, derived from the geometry — the denominator the driver declares BEFORE the run
 * and compares against the arm's own tally (`pitfalls/40` §an unnamed denominator). */
GDNDAY_DEV static inline long gdn_item_count(int item, int T) {
    long t = (long)T;
    switch (item) {
        case IT_ENTER: return (long)GDN_H_V * GDN_C;
        case IT_CUM: return GDN_H_V;
        case IT_PAIR: return (long)GDN_H_V * GDN_C * GDN_C;
        case IT_ROWPOST: return (long)GDN_H_V * GDN_C;
        case IT_SOLVE: return (long)GDN_H_V * 2 * GDN_D_V;
        case IT_VNEW: return (long)GDN_H_V * GDN_D_V;
        case IT_CORE: return (long)GDN_H_V * t * GDN_D_V;
        case IT_STATE: return (long)GDN_H_V * GDN_D_K * GDN_D_V;
        case IT_CAST: return t * GDN_H_V * GDN_D_V;
        case IT_NORM: return t * GDN_H_V;
        default: return t * GDN_H;
    }
}
GDNDAY_DEV static inline long gdn_case_items(int T) {
    long s = 0;
    for (int it = 0; it < IT_N; ++it) s += gdn_item_count(it, T);
    return s;
}

/* ---- the per-field load fingerprint (used for out_weight; computed for every field) ---------- */
typedef struct {
    uint64_t sum;      /* sum of the raw bit images, mod 2^64 */
    uint64_t sumidx;   /* sum of (bit image) * (index + 1), mod 2^64 */
    long n;
} GdnFp;

/* ---- the plan -------------------------------------------------------------------------------- */
typedef struct {
    int idx;
    int T;
    int mode;                     /* GdnMode */
    int wslot;                    /* which weight block of weights.bin this case binds */
    int carry;                    /* case whose emitted state is this case's incoming state; -1 = file */
    long in_off, out_off;         /* word offsets in the work files */
    char block[32];
    char name[64];
} GdnPlanCase;

typedef struct {
    GdnPlanCase c[GDN_MAX_CASES];
    int n;
    long in_words, out_words;
} GdnPlan;

/* echo layout: the first GDN_ECHO_N fields, in field order, one 32-bit word per element, sized by
 * THIS case's T (a case with T=32 reserves 32 rows of q_pre, not 64) */
GDNDAY_DEV static inline long gdn_echo_n(int field, int T) {
    return (field < GDN_ECHO_N) ? gdn_field_n(field, T) : 0;
}
GDNDAY_DEV static inline long gdn_echo_words(int T) {
    long s = 0;
    for (int f = 0; f < GDN_ECHO_N; ++f) s += gdn_field_n(f, T);
    return s;
}

/* ---- bind a case's pointers into the context; ONE address expression, shared by every item --- */
static inline void gdn_bind(GdnCtx *c, const GdnCase *k, uint32_t *in, uint32_t *w, uint32_t *out,
                            uint32_t *echo, int T) {
    (void)k;
    long o = 0;
    c->q_pre = in + o; o += gdn_field_n(GF_Q_PRE, T);
    c->k_pre = in + o; o += gdn_field_n(GF_K_PRE, T);
    c->v_pre = in + o; o += gdn_field_n(GF_V_PRE, T);
    c->beta = in + o; o += gdn_field_n(GF_BETA, T);
    c->g_decay = in + o; o += gdn_field_n(GF_G_DECAY, T);
    c->z = in + o; o += gdn_field_n(GF_Z, T);
    c->norm_w = w;
    c->out_w = w + GDN_D_V;
    c->init_state = (const float *)(in + o);
    c->T = T;
    o += (long)GDN_H_V * GDN_D_K * GDN_D_V;
    (void)o;
    /* scratch lives after the emitted stages in the same per-case block */
    long so = gdn_stage_words(T);
    float *base = (float *)out;
    c->q_hat = base + gdn_stage_off(T, GS_Q_HAT);
    c->k_hat = base + gdn_stage_off(T, GS_K_HAT);
    c->scan_out = base + gdn_stage_off(T, GS_SCAN_OUT);
    c->core_attn = (uint32_t *)(base + gdn_stage_off(T, GS_CORE_ATTN));
    c->state = base + gdn_stage_off(T, GS_STATE);
    c->normed = (uint32_t *)(base + gdn_stage_off(T, GS_NORMED));
    c->y = (uint32_t *)(base + gdn_stage_off(T, GS_Y));
    c->cum = base + so;
    c->ut = c->cum + (long)GDN_H_V * GDN_C;
    c->ia = c->ut + (long)GDN_H_V * GDN_C * GDN_C;
    c->qhat = c->ia + (long)GDN_H_V * GDN_C * GDN_C;
    c->khat = c->qhat + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->kb = c->khat + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->vpad = c->kb + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->vb = c->vpad + (long)GDN_H_V * GDN_C * GDN_D_V;
    c->dcb = c->vb + (long)GDN_H_V * GDN_C * GDN_D_V;
    c->newv = c->dcb + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->kcd = c->newv + (long)GDN_H_V * GDN_C * GDN_D_V;
    c->vnew = c->kcd + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->qdec = c->vnew + (long)GDN_H_V * GDN_C * GDN_D_V;
    c->kdec = c->qdec + (long)GDN_H_V * GDN_C * GDN_D_K;
    c->echo[0] = echo;
    for (int f = 1; f < GF_N; ++f) c->echo[f] = c->echo[f - 1] + gdn_echo_n(f - 1, T);
}

/* ---- the load census: one pass per field through the SAME gdn_load_bits the arithmetic uses ---
 * Object of the check: not "did the file hash right on disk" (read_dump_tensor already sha256'd
 * it) but "what did the kernel end up with". A bf16 image decoded as half, a cast where a bitcast
 * belongs, or an off-by-one field offset all show up here as a mismatching word. */
static inline void gdn_census(const GdnCtx *c, GdnFp *fp) {
    for (int f = 0; f < GF_N; ++f) {
        long n = gdn_field_n(f, c->T);
        uint64_t s = 0, si = 0;
        for (long i = 0; i < n; ++i) {
            uint32_t b = gdn_load_bits(c, f, i);
            s += b;
            si += (uint64_t)b * (uint64_t)(i + 1);
            if (f < GDN_ECHO_N) c->echo[f][i] = b;
        }
        fp[f].sum = s; fp[f].sumidx = si; fp[f].n = n;
    }
}

/* ---- one ordered pass over every item of one case.
 * `GDN_ST_STATE_NONZERO` counts the live words in the incoming state. In slice 1 that count was a
 * REFUSAL condition (a nonzero state was outside the slice); in slice 2 it is a MEASURED FACT about
 * the case and `gate.py` requires it to equal the non-zero count of the pinned tensor the state was
 * read from, so a case cannot claim a live state it did not receive, and cannot silently receive a
 * zero one it was not meant to. */
static inline void gdn_case_run(const GdnCtx *c, const GdnCase *k, long *st) {
    long nz = 0;
    long n_state = (long)GDN_H_V * GDN_D_K * GDN_D_V;
    for (long i = 0; i < n_state; ++i) if (c->init_state[i] != 0.0f) ++nz;
    st[GDN_ST_STATE_NONZERO] = nz;

    if (k->mode == GDN_M_STUB) {
        for (int s = 0; s < GS_N; ++s) {
            long n = gdn_stage_n(s, k->t);
            for (long i = 0; i < n; ++i) gdn_item_stub(c, k, s, i);
            st[GDN_ST_ITEMS] += n;
        }
        return;
    }
    for (int it = 0; it < IT_N; ++it) {
        long n = gdn_item_count(it, k->t);
        for (long i = 0; i < n; ++i) {
            switch (it) {
                case IT_ENTER: gdn_item_enter(c, k, (int)(i / GDN_C), (int)(i % GDN_C)); break;
                case IT_CUM: gdn_item_cum(c, k, (int)i); break;
                case IT_PAIR: gdn_item_pair(c, k, (int)(i / (GDN_C * GDN_C)),
                                            (int)(i % (GDN_C * GDN_C))); break;
                case IT_ROWPOST: gdn_item_rowpost(c, k, (int)(i / GDN_C), (int)(i % GDN_C)); break;
                case IT_SOLVE: {
                    int per = 2 * GDN_D_V;
                    int h = (int)(i / per), r = (int)(i % per);
                    gdn_item_solve(c, h, r / GDN_D_V, r % GDN_D_V);
                    break;
                }
                case IT_VNEW: gdn_item_vnew(c, (int)(i / GDN_D_V), (int)(i % GDN_D_V)); break;
                case IT_CORE: {
                    int per = k->t * GDN_D_V;
                    int h = (int)(i / per), rem = (int)(i % per);
                    gdn_item_core(c, k, h, rem / GDN_D_V, rem % GDN_D_V);
                    break;
                }
                case IT_STATE: {
                    int per = GDN_D_K * GDN_D_V;
                    int h = (int)(i / per), rem = (int)(i % per);
                    gdn_item_state(c, h, rem / GDN_D_V, rem % GDN_D_V);
                    break;
                }
                case IT_CAST: {
                    int per = GDN_H_V * GDN_D_V;
                    int t = (int)(i / per), rem = (int)(i % per);
                    gdn_item_exit_cast(c, t, rem / GDN_D_V, rem % GDN_D_V);
                    break;
                }
                case IT_NORM: gdn_item_norm(c, k, (int)(i / GDN_H_V), (int)(i % GDN_H_V)); break;
                default: gdn_item_outproj(c, k, (int)(i / GDN_H), (int)(i % GDN_H)); break;
            }
            st[GDN_ST_STAGE_ITEMS]++;
        }
        st[GDN_ST_ITEMS] += n;
    }
    /* the chunk-pad columns of the delta rule must be exactly zero: §04b.2 says pads are ZERO and
     * this counts the ones that are not. A nonzero pad is the padding policy doing work the spec
     * says it does not — printed as a fact about this case, never assumed. */
    long pad = 0;
    for (int h = 0; h < GDN_H_V; ++h)
        for (int i = k->t; i < GDN_C; ++i)
            for (int v = 0; v < GDN_D_V; ++v)
                if (c->vnew[((long)h * GDN_C + i) * GDN_D_V + v] != 0.0f) ++pad;
    st[GDN_ST_PAD_NONZERO] = pad;
}

#endif /* GDN_CASE_H_ */
