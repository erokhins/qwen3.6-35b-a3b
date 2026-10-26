/* engine/device/m2_rt_kernels.h — the MoE router (R1..R5) on the engine's device arm.
 *
 * WHO OWNS WHICH ARITHMETIC — read this before trusting a row.
 *
 *   R2 softmax, R3 top-K with both tie rules, R4 renormalisation and R5 the cast to the logits'
 *   dtype are called VERBATIM from the gated prototype lane: `mr_softmax_row`, `mr_topk_row`,
 *   `mr_renorm_row` and the dtype edges `mr_bf16_to_f32`/`mr_f32_to_bf16`, all in
 *   `prototype/moe_router/moe_router_kernels.h`. They are not re-derived here, which is the point:
 *   the prototype's own gate already bends those functions over a plan of injected modes, and the
 *   mode words below are that lane's own named mistakes (`MR_MODE_*`), so the engine cannot quietly
 *   define a softer router than the lane that was gated.
 *
 *   R1 is NEW here, in `m2_rt_logit_acc_f32`. It has to be: policy A stores the router matrix
 *   F32-exact (DECISIONS.md D-01a), while the prototype's `mr_logit_row` reads a bf16 matrix and
 *   would read four-byte weights as two-byte ones. That is also why control `router-span-is-f32`
 *   exists — a width claim that is parsed off a plan is a width claim, not a width check.
 *
 *   NOTHING in this file, or anywhere in `engine/device/`, calls, includes or transliterates
 *   `engine/block_forward.py::moe_router`. The torch program is the oracle side of the comparison;
 *   this arm answers from the C functions above plus `m2_rt_logit_acc_f32`, so a device arm and the
 *   host program agreeing is two implementations agreeing, not one compiled twice. The f32 softmax
 *   used for the arm-basis leak field (`m2_rt_softmax_f32_row`) is engine code too, and is named as
 *   such in the driver's disclosure because the leak row is judged against a pinned referent.
 *
 * WHAT THIS SLICE DOES NOT COMPUTE: the routed experts. There is no `MR_F_ROUTED` field, so the four
 * prototype modes that reach only the routed sum (`MR_MODE_CAST_AFTER`, `MR_MODE_RANK_ACCUM`,
 * `MR_MODE_W_BY_EXPERT_ID`, `MR_MODE_F32_COMBINE`) are refused by the plan checker with
 * `mode-unreachable-here` rather than accepted and silently inert. The dtype-order question (AGENTS.md
 * prediction 4) is provable without them, on the weights themselves: see `m2_rt_item2`'s three
 * emitted weight forms.
 *
 * NO TIMER, NO RATE: nothing here reads a clock (`pitfalls/00`).
 */
#ifndef M2_RT_KERNELS_H
#define M2_RT_KERNELS_H

#include <stdint.h>
#include <string.h>

#include "moe_router_kernels.h"          /* via -I prototype/moe_router */

/* The record: the prototype's five fields at the prototype's offsets, then three fields this slice
 * adds. Appending rather than renumbering means a field's offset in the engine's record is the same
 * number the prototype gate prints, so a record can be read by either lane's tooling.
 *   logits      R1 rounded to the activation dtype (the shipping R1)        E
 *   probs       R2 over `logits`                                          E
 *   wpre        R4, fp32 (the f32-carry form)                              K
 *   wcast       R5, cast before anything multiplies (the shipping form)     K
 *   ids         R3, lower-index tie rule, descending probability           K
 *   logits_f32  R1 unrounded: the arm's own basis                          E
 *   wleak       R2..R5 recomputed from `logits_f32` (the pre-fix leak form) K
 *   ids_alt     R3 with the higher-index tie rule                          K
 */
#define M2_RT_OFF_LOGITS     MR_OFF_LOGITS
#define M2_RT_OFF_PROBS      MR_OFF_PROBS
#define M2_RT_OFF_WPRE       MR_OFF_WPRE
#define M2_RT_OFF_WCAST      MR_OFF_WCAST
#define M2_RT_OFF_IDS        MR_OFF_IDS
#define M2_RT_OFF_LOGITS_F32 (MR_OFF_IDS + MR_K)
#define M2_RT_OFF_WLEAK      (M2_RT_OFF_LOGITS_F32 + MR_E)
#define M2_RT_OFF_IDS_ALT    (M2_RT_OFF_WLEAK + MR_K)
#define M2_RT_FLOATS         (M2_RT_OFF_IDS_ALT + MR_K)

#define M2_RT_FIELD_COUNT 8
#define M2_RT_F_LOGITS      0
#define M2_RT_F_PROBS       1
#define M2_RT_F_WPRE        2
#define M2_RT_F_WCAST       3
#define M2_RT_F_IDS         4
#define M2_RT_F_LOGITS_F32  5
#define M2_RT_F_WLEAK       6
#define M2_RT_F_IDS_ALT     7

static const char *const M2_RT_FIELD_NAME[M2_RT_FIELD_COUNT] = {
    "logits", "probs", "wpre", "wcast", "ids", "logits_f32", "wleak", "ids_alt"
};
static const int M2_RT_FIELD_OFF[M2_RT_FIELD_COUNT] = {
    M2_RT_OFF_LOGITS, M2_RT_OFF_PROBS, M2_RT_OFF_WPRE, M2_RT_OFF_WCAST, M2_RT_OFF_IDS,
    M2_RT_OFF_LOGITS_F32, M2_RT_OFF_WLEAK, M2_RT_OFF_IDS_ALT
};
static const int M2_RT_FIELD_LEN[M2_RT_FIELD_COUNT] = {
    MR_E, MR_E, MR_K, MR_K, MR_K, MR_E, MR_K, MR_K
};

/* cast_form: the two answers to "what dtype does R1 land in". ACT_DTYPE is §11.2 and the pins;
 * ARM_BASIS is the shape that shipped at the prototype lane's deadline and was later attributed. */
#define M2_RT_CAST_ACT_DTYPE 0
#define M2_RT_CAST_ARM_BASIS 1

/* skip_stage: which stage is forbidden to write, so the NaN census has a non-zero leg. */
#define M2_RT_SKIP_NONE      0
#define M2_RT_SKIP_RENORM    1
#define M2_RT_SKIP_TOPK      2

struct M2RtPlan {
    long off_w;                 /* router matrix, F32, E rows x H cols, row-major */
    long off_x;                 /* the token's bf16 activation row, H elements */
    long n_w_bytes;
    int rows, cols, top_k, elt;
    int mode;                   /* MR_MODE_* */
    int cast_form;              /* M2_RT_CAST_* */
    int skip_stage;             /* M2_RT_SKIP_* */
    int tie_forge;              /* 1 = force an exact tie across the K boundary before R3 */
};

struct M2RtCounters {
    uint32_t reads_w, reads_x, items1, items2;
};

/* The prototype's poison pattern, same bits the prototype gate counts (§MR_NAN_BITS). A counter that
 * can only read zero is not a guard. */
MR_DEV inline void m2_rt_poison(float *out) {
    int i;
    const float nan = mr_bits_to_f32(MR_NAN_BITS);
    for (i = 0; i < M2_RT_FLOATS; ++i) out[i] = nan;
}

MR_DEV inline int m2_rt_field_nan(const float *out, int field) {
    const int off = M2_RT_FIELD_OFF[field], len = M2_RT_FIELD_LEN[field];
    int i, n = 0;
    for (i = 0; i < len; ++i) {
        const uint32_t b = mr_f32_bits(out[off + i]);
        if ((b & 0x7FFFFFFFu) == MR_NAN_BITS) n++;
    }
    return n;
}

/* --------------------------------------------------------------------------------------------- */
/* R1 at the policy-A width. NEW engine code, and the reason it is new is in the header comment.
 * Documented order: ascending column index, one fp32 accumulator, no contraction (`-ffp-contract=off`
 * in build.sh, so the written order is the machine order). The result is returned UNROUNDED; whether
 * it is rounded to the activation dtype is the `cast_form` question, decided by the caller.          */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline float m2_rt_logit_acc_f32(const uint16_t *x, const float *w, int rows, int cols,
                                        int expert, int mode, int *bad) {
    long row = expert;
    float acc = 0.0f;
    int j;
    if (mode == MR_MODE_ROW_OFF_1) row = (expert + 1) % rows;
    for (j = 0; j < cols; ++j) {
        const long off = (mode == MR_MODE_WG_TRANSPOSED) ? ((long)j * rows + row)
                                                        : (row * cols + j);
        if (off < 0 || off >= (long)rows * cols) {   /* the transposed read stays inside the span:
                                                       the injected bug is the stride, not a wild
                                                       address, so it must not fault */
            if (bad) *bad = 1;
            acc += mr_bf16_to_f32(x[j]) * w[(long)rows * cols - 1];
            continue;
        }
        acc += mr_bf16_to_f32(x[j]) * w[off];
    }
    return acc;
}

/* The leak form needs a softmax over logits that were never rounded, and the prototype's
 * `mr_softmax_row` takes bf16 bit patterns by design. Same documented order as the prototype's:
 * max first, exp ascending, denominator ascending, then divide. Engine code, named as such. */
MR_DEV inline void m2_rt_softmax_f32_row(const float *v, int n, float *out) {
    float maxv = v[0], s = 0.0f;
    int i;
    for (i = 1; i < n; ++i) if (v[i] > maxv) maxv = v[i];
    for (i = 0; i < n; ++i) out[i] = mr_exp(v[i] - maxv);
    for (i = 0; i < n; ++i) s += out[i];
    for (i = 0; i < n; ++i) out[i] = out[i] / s;
}

/* One E-row stage: the logits in both forms. `bad` is the out-of-range counter for the transposed
 * injection; it is counted, never silently clamped away. */
MR_DEV inline void m2_rt_item1(const struct M2RtPlan *p, const uint8_t *slab, float *out,
                               int expert, struct M2RtCounters *t, int *bad) {
    const uint16_t *x = (const uint16_t *)(slab + p->off_x);
    const float *w = (const float *)(slab + p->off_w);
    const float acc = m2_rt_logit_acc_f32(x, w, p->rows, p->cols, expert, p->mode, bad);
    out[M2_RT_OFF_LOGITS_F32 + expert] = acc;
    out[M2_RT_OFF_LOGITS + expert] = (p->cast_form == M2_RT_CAST_ARM_BASIS)
                                       ? acc
                                       : mr_bf16_to_f32(mr_f32_to_bf16(acc));
    if (t) { t->reads_w += (uint32_t)p->cols; t->reads_x += (uint32_t)p->cols; t->items1 += 1; }
}

/* Where the K boundary is, for the tie forgery: the slot pair (K-1, K) in the honest selection
 * order. Derived from the row's own probabilities, not typed: the forgery has to straddle the
 * boundary or it proves nothing, and hand-picking an index would be the typed constant this repo
 * forbids. Returns 0 when there is nothing to forge. */
MR_DEV inline int m2_rt_tie_boundary(const float *probs, int n, int k, int *hi) {
    int used[MR_E], i, slot;
    for (i = 0; i < n; ++i) used[i] = 0;
    for (slot = 0; slot < k; ++slot) {
        int best = -1;
        float bv = 0.0f;
        for (i = 0; i < n; ++i) {
            if (used[i]) continue;
            if (best < 0 || probs[i] > bv) { best = i; bv = probs[i]; }
        }
        if (best < 0) return 0;
        used[best] = 1;
        if (slot == k - 1) {
            int next = -1;
            float nv = 0.0f;
            for (i = 0; i < n; ++i) {
                if (used[i]) continue;
                if (next < 0 || probs[i] > nv) { next = i; nv = probs[i]; }
            }
            if (next < 0) return 0;
            *hi = next;
            return best;
        }
    }
    return 0;
}

/* R2..R5, both tie rules, three weight forms. One item, because the router is the block's serial
 * point: this is the shape the engine has to make fast later, and it is printed as such by the item
 * census rather than hidden behind a fake parallel decomposition. */
MR_DEV inline void m2_rt_item2(const struct M2RtPlan *p, const float *logits_field,
                               const float *logits_f32, float *out, struct M2RtCounters *t) {
    uint16_t bits[MR_E];
    uint16_t wbits[MR_K];
    int ids[MR_K], ids_alt[MR_K], i;
    if (p->skip_stage == M2_RT_SKIP_TOPK || p->skip_stage == M2_RT_SKIP_RENORM) {
        if (t) t->items2 += 1;
        return;                                  /* probs/ids/weights stay poison: the census leg */
    }
    for (i = 0; i < p->rows; ++i) bits[i] = mr_f32_to_bf16(logits_field[i]);
    mr_softmax_row(bits, p->mode, out + M2_RT_OFF_PROBS);

    if (p->tie_forge) {
        int hi = 0;
        const int lo = m2_rt_tie_boundary(out + M2_RT_OFF_PROBS, p->rows, p->top_k, &hi);
        if (lo > 0) out[M2_RT_OFF_PROBS + hi] = out[M2_RT_OFF_PROBS + lo];
    }

    mr_topk_row(out + M2_RT_OFF_PROBS, p->mode, ids);
    mr_topk_row(out + M2_RT_OFF_PROBS, MR_MODE_TIE_HIGH, ids_alt);
    for (i = 0; i < p->top_k; ++i) {
        out[M2_RT_OFF_IDS + i] = (float)ids[i];
        out[M2_RT_OFF_IDS_ALT + i] = (float)ids_alt[i];
    }
    if (p->skip_stage == M2_RT_SKIP_RENORM) { if (t) t->items2 += 1; return; }

    mr_renorm_row(out + M2_RT_OFF_PROBS, ids, p->mode, out + M2_RT_OFF_WPRE, wbits);
    for (i = 0; i < p->top_k; ++i) out[M2_RT_OFF_WCAST + i] = mr_bf16_to_f32(wbits[i]);

    /* The arm-basis leak: R2..R5 again, from logits that were never rounded. Judgment material, so
     * it is produced in the same run as the honest fields and read out of the same record. */
    {
        float leak_probs[MR_E];
        m2_rt_softmax_f32_row(logits_f32, p->rows, leak_probs);
        if (p->tie_forge) {
            int hi = 0;
            const int lo = m2_rt_tie_boundary(leak_probs, p->rows, p->top_k, &hi);
            if (lo > 0) leak_probs[hi] = leak_probs[lo];
        }
        mr_renorm_row(leak_probs, ids, p->mode, out + M2_RT_OFF_WLEAK, wbits);
    }
    if (t) t->items2 += 1;
}

/* The two-stage program, in the order the specification writes it: R1 for every expert, then R2..R5.
 * `bad` is reported by the caller; a transposed read that stayed in bounds is still a stride bug and
 * the census is where it shows. */
MR_DEV inline void m2_rt_run(const struct M2RtPlan *p, const uint8_t *slab, float *out,
                             struct M2RtCounters *ctr, int *bad) {
    int e;
    memset(ctr, 0, sizeof(*ctr));
    m2_rt_poison(out);
    if (p->skip_stage != M2_RT_SKIP_TOPK && p->skip_stage != M2_RT_SKIP_RENORM) {
        for (e = 0; e < p->rows; ++e) m2_rt_item1(p, slab, out, e, ctr, bad);
    } else {
        ctr->items1 = 0;                          /* stage 1 forbidden: reads must read 0 too */
    }
    m2_rt_item2(p, out + M2_RT_OFF_LOGITS, out + M2_RT_OFF_LOGITS_F32, out, ctr);
}

/* Can `mode` move anything this slice emits? Derived from the prototype's own reach table, so the
 * engine does not get to decide which injections are interesting. `MR_MODE_CORRECT` is the honest run:
 * it "reaches" nothing by the table's definition because it injects nothing, and it is obviously
 * admissible. Everything else has to name at least one field this record carries, which is what turns
 * the four routed-sum-only modes into a refusal instead of a silent no-op. */
MR_DEV inline int m2_rt_mode_reaches_anything(int mode) {
    int f;
    if (mode == MR_MODE_CORRECT) return 1;
    for (f = 0; f < M2_RT_FIELD_COUNT; ++f) {
        const int field = (f <= M2_RT_F_IDS) ? f : (f == M2_RT_F_LOGITS_F32 ? MR_F_LOGITS
                                : (f == M2_RT_F_IDS_ALT ? MR_F_IDS : MR_F_WCAST));
        if (mr_mode_reaches(mode, field, MR_ARM_KERNEL) != MR_REACH_NO) return 1;
    }
    return 0;
}

MR_DEV inline int m2_rt_mode_of(const char *w) {
    if (!strcmp(w, "correct")) return MR_MODE_CORRECT;
    if (!strcmp(w, "softmax_bf16")) return MR_MODE_SOFTMAX_BF16;
    if (!strcmp(w, "no_renorm")) return MR_MODE_NO_RENORM;
    if (!strcmp(w, "row_off1")) return MR_MODE_ROW_OFF_1;
    if (!strcmp(w, "wg_transposed")) return MR_MODE_WG_TRANSPOSED;
    if (!strcmp(w, "tie_high")) return MR_MODE_TIE_HIGH;
    if (!strcmp(w, "topk_ascending")) return MR_MODE_TOPK_ASC;
    if (!strcmp(w, "no_max_sub")) return MR_MODE_NO_MAX_SUB;
    if (!strcmp(w, "poison_nan")) return MR_MODE_POISON_NAN;
    if (!strcmp(w, "cast_after")) return MR_MODE_CAST_AFTER;
    if (!strcmp(w, "rank_accum")) return MR_MODE_RANK_ACCUM;
    if (!strcmp(w, "w_by_expert_id")) return MR_MODE_W_BY_EXPERT_ID;
    if (!strcmp(w, "f32_combine")) return MR_MODE_F32_COMBINE;
    return -1;
}

#endif /* M2_RT_KERNELS_H */
