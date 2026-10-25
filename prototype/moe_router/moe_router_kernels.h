/* moe_router_kernels.h — the MoE router's arithmetic, in one __host__ __device__ path.
 *
 * The router is the block's only per-token serial point: R1 the 256-way logit dot, R2 the fp32
 * softmax over all E, R3 the top-K of those probabilities, R4 the renormalisation over the K, R5
 * the cast of the renormalised weights back to the LOGITS dtype (`specification/11-moe-block.md`
 * §Stage-by-stage). Everything the expert block reads is downstream of R3, so a router that is
 * wrong by one slot is an engine that reads eight wrong 6 MiB experts.
 *
 * WHY ONE SHARED HEADER
 *   The device kernel and the host probe compile this file together, so "the host arm and the
 *   device arm agreed" cannot mean "two implementations were written to agree". The parallel
 *   structure (which thread does which element) lives in moe_router.hip; the ARITHMETIC lives only
 *   here, and every function below is a fixed-order scalar loop.
 *
 * WHERE EXACTNESS IS CLAIMED, AND WHERE IT IS NOT
 *   Claimed bit-exact (the gate compares 4-byte patterns, zero tolerance):
 *     - the bf16 cast weights, R5's output — the manifest's authoritative arm
 *       (`checkpoints/dump-manifest-2026-10-26.json`, dump `topk_weights`);
 *     - the top-K ids, on rows the tie-break does not make ambiguous (A-03);
 *     - the routed sum built from the pinned `expert_raw` with the reference's combine form.
 *   Not claimed, and named here rather than discovered later:
 *     - R1's 2048-term dot and R2's 256-term denominator are long dots: their last bits follow the
 *       summation order, and the reference's order is torch's internal one. `specification/
 *       10-reference-dumps-and-gates.md` §"Bands are derived, not invented" puts long dots under a
 *       floor probe and a band, so this file takes a documented order, prints it, and the gate sizes
 *       the band from a re-association probe of the reference's own math. The measured consequence,
 *       which is why the claim sits on R5 and not on R1: across the four pinned blocks an
 *       independent fp32 route reproduces the pinned f32 pre-cast weights to single-digit ULP and
 *       NOT bit-exactly, while their bf16 cast is reproduced bit-exactly on every element.
 *   FMA cannot break the exactness claims: every product R1 forms multiplies two bf16 values, whose
 *   8+8 = 16 significand bits are exact in fp32, so `fma(a,b,acc)` and `add(mul(a,b),acc)` round the
 *   same value; contraction is therefore invisible in R1. R4's quotients are single divisions with
 *   no product to contract, and the combine's product is bf16-rounded before the add, where the add
 *   is fp32-with-one-rounding (mr_bf16_add) — the shape torch uses for `bf16 += bf16`.
 */
#ifndef MOE_ROUTER_KERNELS_H
#define MOE_ROUTER_KERNELS_H

#if defined(__HIPCC__) || defined(__CUDACC__)
/* One source, two arms: the device kernel and host_probe.cpp (g++) call the same bodies, so a
 * device-vs-host disagreement is attributable to the device compiler and not to two spellings of
 * the arithmetic (pitfalls/40 §"A check whose reference side is derived from the thing under
 * test"). */
#define MR_DEV __host__ __device__
#define MR_GLOBAL __global__
#else
#define MR_DEV
#define MR_GLOBAL
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ---- geometry, from checkpoints/Qwen3.6-35B-A3B.config.json --------------------------------- *
 * Asserted against the config by gate.py (check_header_enum) and against the manifest's dump
 * shapes, so a re-typed constant that drifts goes red instead of silently shrinking the sample.  */
#define MR_E 256        /* num_experts         */
#define MR_K 8          /* num_experts_per_tok */
#define MR_H 2048       /* hidden_size         */

/* The output record one case writes: one contiguous span of fp32 words. The bf16-carrying fields
 * are stored widened to fp32 (prototype/q4k_dequant's convention) so the gate compares 4-byte
 * patterns for every field and never has to guess a dtype off a file (pitfalls/30). */
#define MR_OFF_LOGITS 0
#define MR_OFF_PROBS (MR_OFF_LOGITS + MR_E)
#define MR_OFF_WPRE (MR_OFF_PROBS + MR_E)
#define MR_OFF_WCAST (MR_OFF_WPRE + MR_K)
#define MR_OFF_IDS (MR_OFF_WCAST + MR_K)
#define MR_OFF_ROUTED (MR_OFF_IDS + MR_K)
#define MR_CASE_FLOATS (MR_OFF_ROUTED + MR_H)

#define MR_FIELD_COUNT 6
#define MR_F_LOGITS 0
#define MR_F_PROBS 1
#define MR_F_WPRE 2
#define MR_F_WCAST 3
#define MR_F_IDS 4
#define MR_F_ROUTED 5

/* What an unwritten slot reads. mr_row fills the whole record with this pattern before any stage
 * runs, so a stage that forgets to write its field is caught by the gate's per-field NaN count
 * instead of passing on a 0.0 left behind by the buffer's initialisation. A counter that can only
 * read zero is not a guard (pitfalls/40): the driver's --selftest poisons one row and requires this
 * pattern to appear, and gate.py requires the poison row to be present in the plan. */
#define MR_NAN_BITS 0x7FC00000u

enum {
    MR_MODE_CORRECT = 0,        /* the reference form: R1..R5 then the reference combine */
    MR_MODE_SOFTMAX_BF16 = 1,   /* R2 in bf16: the tempting rewrite §11.2 names as out of scope */
    MR_MODE_NO_RENORM = 2,      /* R4 dropped: the weights are not divided by their own sum */
    MR_MODE_CAST_AFTER = 3,     /* R5 moved after the multiply — prediction 4's wrong order */
    MR_MODE_RANK_ACCUM = 4,     /* the K addends summed in rank order, not ascending expert id */
    MR_MODE_W_BY_EXPERT_ID = 5, /* w indexed by expert id instead of by rank (§11.3, row 3) */
    MR_MODE_ROW_OFF_1 = 6,      /* R1 reads router weight row (l+1) % E */
    MR_MODE_WG_TRANSPOSED = 7,  /* R1 indexes the router weight as [j*E+l] instead of [l*H+j] */
    MR_MODE_TIE_HIGH = 8,       /* R3 sends an exact tie to the higher index (A-03's corner) */
    MR_MODE_F32_COMBINE = 9,    /* fp32 accumulator, rounded to bf16 once at the end */
    MR_MODE_TOPK_ASC = 10,      /* R3 returns the K ascending by expert id */
    MR_MODE_NO_MAX_SUB = 11,    /* R2 without the max subtraction */
    MR_MODE_POISON_NAN = 12,    /* leak guard: writes MR_NAN_BITS into every slot of the record */
    MR_MODE_COUNT = 13
};

enum { MR_ARM_KERNEL = 0, MR_ARM_LOGITS = 1 };

/* Which fields a mode can reach, derived from the algebra and written once:
 *   1  softmax in bf16      probs, and everything downstream of probs
 *   2  no renorm            the weights and the routed sum; NOT the ids — dividing a row by one
 *                           positive constant is strictly monotone, so no comparison changes sign
 *   3  cast after multiply  the routed sum only: the weight the reference would have used is still
 *                           computed and still written, so WCAST is identical and only the product
 *                           differs — which is the attribution half of prediction 4
 *   4  rank accumulation    the routed sum only, and only when the ids are not already ascending
 *   5  w by expert id       the routed sum only (the (id,weight) pairing is broken)
 *   6  row off by one       R1 and everything downstream
 *   7  weight transposed    R1 and everything downstream
 *   8  tie goes high        R3 only: the ids, the weights if the SET changed, the sum if it did
 *   9  fp32 combine         the routed sum only
 *  10  top-k ascending      the ids and the weights, together; NOT the routed sum, because the
 *                           reference's addend order is ascending expert id, which is invariant to
 *                           how the K are permuted — the pair case to mode 5
 *  11  no max subtraction   probs and downstream; NOT the logits
 *  12  poison               everything, by construction
 * A mode listed as unable to move a field is a MUST NOT MOVE assertion in the gate: an injection
 * that moves more than it names is a different bug than the one that was injected. */
/* Can `mode` change `field` at all, for this arm? 0 = no, and a move there is a second bug; 1 = yes
 * by construction, and a field that never moves anywhere is a dead control; 2 = yes in principle but
 * only for a population that exercises it (a re-associated sum whose ulp the bf16 cast may absorb, or
 * a reordering that only matters on a near-tie), which the gate must count as MAY rather than demand
 * per case. The arm is part of the question because arm LOGITS never reads the activation or the
 * router weight, so R1's two addressing injections cannot move anything at all on that arm — an
 * immunity the gate counts, not a control that silently stops being a control.
 *
 * The table is the kernel's own claim about itself; gate.py asks the compiled predicate via
 * `moe_router --reach-table` and counts the records it parsed against the records printed. */
#define MR_REACH_NO   0
#define MR_REACH_YES  1
#define MR_REACH_MAY  2

MR_DEV inline int mr_mode_reaches(int mode, int field, int arm) {
    if (arm == MR_ARM_LOGITS && (mode == MR_MODE_ROW_OFF_1 || mode == MR_MODE_WG_TRANSPOSED))
        return MR_REACH_NO;                    /* R1 is not computed on that arm at all */
    switch (mode) {
        case MR_MODE_CORRECT: return 0;
        case MR_MODE_POISON_NAN: return 1;
        /* bf16 probabilities always move the fp32 weights; whether a near-tie changes which expert
         * wins, or whether the bf16 cast of the weight absorbs the move, is population-dependent */
        case MR_MODE_SOFTMAX_BF16:
            if (field == MR_F_LOGITS) return MR_REACH_NO;
            if (field == MR_F_PROBS || field == MR_F_WPRE) return MR_REACH_YES;
            return MR_REACH_MAY;
        case MR_MODE_NO_RENORM:
            if (field == MR_F_IDS || field == MR_F_PROBS || field == MR_F_LOGITS)
                return MR_REACH_NO;
            return (field == MR_F_WPRE) ? MR_REACH_YES : MR_REACH_MAY;
        case MR_MODE_CAST_AFTER: return field == MR_F_ROUTED;
        case MR_MODE_RANK_ACCUM: return field == MR_F_ROUTED;
        case MR_MODE_W_BY_EXPERT_ID: return field == MR_F_ROUTED;
        case MR_MODE_F32_COMBINE: return field == MR_F_ROUTED;
        case MR_MODE_ROW_OFF_1: return 1;
        case MR_MODE_WG_TRANSPOSED: return 1;
        /* the weights of a tied pair are equal by definition, so only the ids can be obliged to
         * move, and only when the tied group straddles the K boundary (the gate's tie classes) */
        case MR_MODE_TIE_HIGH:
            if (field == MR_F_LOGITS || field == MR_F_PROBS) return MR_REACH_NO;
            return (field == MR_F_IDS) ? MR_REACH_YES : MR_REACH_MAY;
        /* sorting the top-k re-associates the renorm denominator, so the weights move by
         * construction and the routed sum may follow if the ulp survives the bf16 cast */
        case MR_MODE_TOPK_ASC:
            if (field == MR_F_IDS || field == MR_F_WPRE || field == MR_F_WCAST)
                return MR_REACH_YES;
            return (field == MR_F_ROUTED) ? MR_REACH_MAY : MR_REACH_NO;
        /* the max subtraction is exact in real arithmetic: it perturbs the fp32 denominator, and on
         * a fixture whose logits span 17 orders of magnitude in exp() the normalised probabilities
         * come out bit-identical. Everything downstream of that, including the probabilities
         * themselves, is absorbed or not depending on the row */
        case MR_MODE_NO_MAX_SUB:
            if (field == MR_F_LOGITS) return MR_REACH_NO;
            return MR_REACH_MAY;
        default: return 1;                                /* unknown mode: assume it moves things */
    }
}

/* --------------------------------------------------------------------------------------------- */
/* dtype edges. The dumps store the bf16 BIT IMAGE (uint16), so the kernel reads bits and converts
 * by shifting — never by re-running torch's cast — and rounds with one round-to-nearest-even.     */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline float mr_bf16_to_f32(uint16_t u) {
    const uint32_t b = (uint32_t)u << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

MR_DEV inline uint16_t mr_f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    /* A NaN arriving here is a bug upstream, not a value to preserve: it is quieted to one pattern
     * so the gate's NaN==NaN rule has exactly one thing to compare. +-Inf passes through as the
     * bf16 infinities; the gate's non-finite check is the failure, not this function. */
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) return 0x7FC0u;
    const uint32_t lsb = (b >> 16) & 1u;
    return (uint16_t)((b + 0x7FFFu + lsb) >> 16);
}

MR_DEV inline uint32_t mr_f32_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

MR_DEV inline float mr_bits_to_f32(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* `bf16 += bf16`: torch computes this in fp32 (opmath) with one rounding back, which is why the
 * combine below accumulates through here instead of adding two bf16 values in bf16. */
MR_DEV inline uint16_t mr_bf16_add(uint16_t a, uint16_t b) {
    return mr_f32_to_bf16(mr_bf16_to_f32(a) + mr_bf16_to_f32(b));
}

MR_DEV inline float mr_exp(float x) {
    /* expf on both arms: glibc's on the host, ROCm's libdevice's on the device. They are allowed to
     * differ by an ULP or two, and that is a printed finding rather than a failure, because R2 is
     * band-governed (header headnote) and R5's cast absorbs it. The host/device attribution check in
     * gate.py is therefore field-scoped: bit-identity on the claimed fields, band agreement on the
     * long-dot fields. */
    return expf(x);
}

/* --------------------------------------------------------------------------------------------- */
/* R1 the logit dot.  logits[l] = bf16( sum_j f32(x[j]) * f32(W[l*H + j]) ), accumulated fp32.     */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline uint16_t mr_logit_row(const uint16_t *x, const uint16_t *wg, int expert,
                                                 int row_shift, int transposed, int *bad_index) {
    const int l = row_shift ? (expert + 1) % MR_E : expert;
    float acc = 0.0f;
    for (int j = 0; j < MR_H; ++j) {
        long off = transposed ? (long)j * MR_E + l : (long)l * MR_H + j;
        if (off < 0 || off >= (long)MR_E * MR_H) {         /* the transposed read stays inside the
                                                            * slab: the bug is the stride, not a
                                                            * wild read, so it must not fault */
            if (bad_index) *bad_index = 1;
            off = (long)MR_E * MR_H - 1;
        }
        acc += mr_bf16_to_f32(x[j]) * mr_bf16_to_f32(wg[off]);
    }
    return mr_f32_to_bf16(acc);
}

/* --------------------------------------------------------------------------------------------- */
/* R2 the softmax. All E experts: a partial softmax is a different function, not an optimisation.  */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline void mr_softmax_row(const uint16_t *logits_bits, int mode, float *probs) {
    float maxv = mr_bf16_to_f32(logits_bits[0]);
    for (int l = 1; l < MR_E; ++l) {                       /* max is exact: order cannot matter */
        const float v = mr_bf16_to_f32(logits_bits[l]);
        if (v > maxv) maxv = v;
    }
    const int nosub = (mode == MR_MODE_NO_MAX_SUB);
    const int bf16out = (mode == MR_MODE_SOFTMAX_BF16);
    for (int l = 0; l < MR_E; ++l) {
        const float v = mr_bf16_to_f32(logits_bits[l]);
        probs[l] = mr_exp(nosub ? v : (v - maxv));
    }
    float s = 0.0f;
    for (int l = 0; l < MR_E; ++l) s += probs[l];          /* documented order: ascending index */
    for (int l = 0; l < MR_E; ++l) {
        float p = probs[l] / s;
        if (bf16out) p = mr_bf16_to_f32(mr_f32_to_bf16(p));
        probs[l] = p;
    }
}

/* --------------------------------------------------------------------------------------------- */
/* R3 top-K. The documented engine rule, and the only one this kernel implements:
 *     probabilities strictly descending; an exact tie goes to the LOWER expert index; the K slots
 *     are returned descending by probability, ties in ascending expert index.
 * A-03 (specification/ASSUMPTIONS.md) reserves the tie-break itself. The pinned arm's tie order is
 * torch's partial-sort artefact and is neither index rule: a lower-index rule reproduces the pinned
 * ORDER on 112 of 128 pinned rows and the pinned SET on 121, and no index rule reproduces layer1
 * row 17, where a three-way boundary tie takes the first and the last of the group. So the rule
 * here exists so the tie injections have something to be wrong against; the rows where it cannot
 * reproduce the pinned arm are adjudicated by the gate with their margins printed, not smoothed.  */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline void mr_topk_row(const float *probs, int mode, int *ids) {
    int used[MR_E];
    for (int l = 0; l < MR_E; ++l) used[l] = 0;
    for (int k = 0; k < MR_K; ++k) {
        int best = -1;
        float bv = 0.0f;
        for (int l = 0; l < MR_E; ++l) {
            if (used[l]) continue;
            const float v = probs[l];
            /* strict > keeps the lowest index on a tie; >= keeps the highest. Those two spellings
             * are the whole of the tie-break, which is why the injection is one character. */
            const int better = (mode == MR_MODE_TIE_HIGH) ? (v >= bv) : (v > bv);
            if (best < 0 || better) { best = l; bv = v; }
        }
        used[best] = 1;
        ids[k] = best;
    }
    if (mode == MR_MODE_TOPK_ASC) {
        /* selection sort by expert id: the (id, weight) pair survives, only its slot moves
         * (§11.3's failure table, row 3 — the ordering half of the pairing question). */
        for (int a = 0; a < MR_K - 1; ++a) {
            int m = a;
            for (int b = a + 1; b < MR_K; ++b)
                if (ids[b] < ids[m]) m = b;
            if (m != a) { const int t = ids[a]; ids[a] = ids[m]; ids[m] = t; }
        }
    }
}

/* --------------------------------------------------------------------------------------------- */
/* R4 renormalise over the K in fp32, in rank order — then R5 the cast to the logits dtype.
 * The reference's `vals / vals.sum(-1)` is one fp32 reduction over K=8; the order is fixed at
 * ascending rank because that is the order whose answer the pinned f32 arm reproduces.           */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline void mr_renorm_row(const float *probs, const int *ids, int mode,
                                              float *wpre, uint16_t *wcast) {
    float s = 0.0f;
    for (int k = 0; k < MR_K; ++k) s += probs[ids[k]];     /* ascending rank; 0.0 + v keeps -0 */
    if (mode == MR_MODE_SOFTMAX_BF16) {
        /* Form(softmax_dtype="bf16") divides bf16 by a bf16 sum: the sum is rounded to bf16 before
         * the quotient and the quotient is bf16 as well, so a bf16 weight is its own R5 cast. */
        s = mr_bf16_to_f32(mr_f32_to_bf16(s));
    }
    for (int k = 0; k < MR_K; ++k) {
        float w = probs[ids[k]];
        if (mode != MR_MODE_NO_RENORM) w = (s == 0.0f) ? 0.0f : w / s;
        if (mode == MR_MODE_SOFTMAX_BF16) w = mr_bf16_to_f32(mr_f32_to_bf16(w));
        wpre[k] = w;
        /* [REF] R5: cast to the LOGITS dtype here, before anything multiplies by it. */
        wcast[k] = mr_f32_to_bf16(w);
    }
}

/* --------------------------------------------------------------------------------------------- */
/* The R5 consumer: the weighted sum of the K expert outputs. The expert outputs are INPUTS (the
 * pinned `expert_raw`, or a deterministic synthetic stand-in) — this lane does not implement the
 * gather; it implements the one multiply that makes the cast position observable. Without this
 * stage the two cast orders are the same number, because bf16(w_f32) IS the bf16 arm: a comparator
 * that stops at the weights cannot tell the orders apart and is not a comparator.
 *
 * The reference form (§11.2 R5 and §11.5; reproduces the pinned `routed_sum` bit-exactly, 0 of
 * 65536 elements differing in each of the four blocks):
 *   prod = bf16( f32(expert_raw[t,k]) * f32(w_bf16[t,k]) )   one rounding
 *   acc  = bf16( f32(acc) + f32(prod) )                       one rounding, addends ascending id
 * --------------------------------------------------------------------------------------------- */
MR_DEV inline void mr_combine_row(const uint16_t *raw /* [K][H] */, const float *wpre,
                                               const uint16_t *wcast, const int *ids, int mode,
                                               uint16_t *out /* [H] */) {
    int order[MR_K];
    for (int k = 0; k < MR_K; ++k) order[k] = k;
    if (mode != MR_MODE_RANK_ACCUM) {
        /* ascending expert id, stable in rank for the (topk-impossible) duplicate-id case */
        for (int a = 0; a < MR_K - 1; ++a)
            for (int b = 0; b < MR_K - 1 - a; ++b)
                if (ids[order[b]] > ids[order[b + 1]]) {
                    const int t = order[b]; order[b] = order[b + 1]; order[b + 1] = t;
                }
    }
    for (int h = 0; h < MR_H; ++h) {
        if (mode == MR_MODE_F32_COMBINE) {
            float facc = 0.0f;
            for (int i = 0; i < MR_K; ++i) {
                const int k = order[i];
                const float w = (mode == MR_MODE_CAST_AFTER) ? wpre[k] : mr_bf16_to_f32(wcast[k]);
                facc += mr_bf16_to_f32(raw[(long)k * MR_H + h]) * w;
            }
            out[h] = mr_f32_to_bf16(facc);
            continue;
        }
        uint16_t acc = 0u;                                 /* +0.0 in the model dtype */
        for (int i = 0; i < MR_K; ++i) {
            const int k = order[i];
            /* cast-before: the weight is the bf16 value, widened. cast-after: the fp32 weight is
             * used as-is, which keeps the 8 mantissa bits the reference threw away. */
            float w = (mode == MR_MODE_CAST_AFTER) ? wpre[k] : mr_bf16_to_f32(wcast[k]);
            if (mode == MR_MODE_W_BY_EXPERT_ID) {
                /* the pairing bug §11.3 row 3: the weight array indexed by the expert id instead of
                 * by the rank. Both arrays carry K entries at the site, so nothing catches it. */
                const int e = ids[k] % MR_K;
                w = (mode == MR_MODE_CAST_AFTER) ? wpre[e] : mr_bf16_to_f32(wcast[e]);
            }
            const uint16_t prod = mr_f32_to_bf16(mr_bf16_to_f32(raw[(long)k * MR_H + h]) * w);
            acc = mr_bf16_add(acc, prod);
        }
        out[h] = acc;
    }
}

/* --------------------------------------------------------------------------------------------- */
/* one router row, end to end                                                                     */
/* --------------------------------------------------------------------------------------------- */
struct MrCase {
    int id;             /* index into the case array; also the row of out.f32's record table */
    int arm;            /* MR_ARM_KERNEL | MR_ARM_LOGITS */
    int mode;           /* MR_MODE_* */
    int tie_class;      /* derived by gate.py from the reference probabilities; see gate.py §ties */
    int id_asc;         /* 1 when the reference's K ids are already ascending -> modes 4 and 5 are
                           structurally immune on this row; derived, never typed */
    int raw_src;        /* 0 pinned expert_raw, 1 synthetic (a deterministic stand-in) */
    int src_key;        /* stable row identity for the report: block*32+token, or 10000+synth index */
    int x_off, wg_off, lg_off, raw_off;      /* element offsets into the bf16 slab */
    int dst_off, cmp_off;                    /* element offsets into out.f32; cmp_off < 0 for mode 0 */
    /* filled by the kernel, read back by the driver: how many elements of each field this injection
     * moved relative to the same row's mode-0 record. The gate recomputes the same numbers from the
     * raw words, so a device counter and a host comparator that disagree is a finding. */
    int bad[MR_FIELD_COUNT];
    int bad_index;
};

struct MrRowState {
    uint16_t logits[MR_E];
    float probs[MR_E];
    float wpre[MR_K];
    uint16_t wcast[MR_K];
    int ids[MR_K];
    uint16_t routed[MR_H];
};

MR_DEV inline void mr_nan_fill(float *dst) {
    for (int i = 0; i < MR_CASE_FLOATS; ++i) dst[i] = mr_bits_to_f32(MR_NAN_BITS);
}

/* The field table the driver and the host probe both walk: one spelling of offset and length.
 * Declared above mr_row because mr_row's self-comparison walks it. */
MR_DEV inline int mr_field_len(int f) {
    switch (f) {
        case MR_F_LOGITS: case MR_F_PROBS: return MR_E;
        case MR_F_WPRE: case MR_F_WCAST: case MR_F_IDS: return MR_K;
        default: return MR_H;
    }
}
MR_DEV inline int mr_field_off(int f) {
    switch (f) {
        case MR_F_LOGITS: return MR_OFF_LOGITS;
        case MR_F_PROBS: return MR_OFF_PROBS;
        case MR_F_WPRE: return MR_OFF_WPRE;
        case MR_F_WCAST: return MR_OFF_WCAST;
        case MR_F_IDS: return MR_OFF_IDS;
        default: return MR_OFF_ROUTED;
    }
}

MR_DEV inline void mr_row(const uint16_t *slab, struct MrCase *c,
                                       struct MrRowState *st, float *buf) {
    /* `buf` is the WHOLE output buffer and this case's slot is reached through c->dst_off, because
     * c->cmp_off is an absolute index into the same buffer. Handing mr_row `buf + dst_off` (as an
     * earlier revision did) made the self-comparison read `buf[dst_off + cmp_off + ...]`: with the
     * full 3224-case plan that walks off the end (AddressSanitizer: SEGV at this comparison), and
     * with a small plan it silently compared some other case's words and printed bad[f]=0. The gate's
     * own verdicts come from the words it reads out of the file, so they never depended on this. */
    float *out = buf + c->dst_off;
    mr_nan_fill(out);
    if (c->mode == MR_MODE_POISON_NAN) return;             /* the leak guard's own arm */
    if (c->arm == MR_ARM_KERNEL) {
        const uint16_t *x = slab + c->x_off;
        const uint16_t *wg = slab + c->wg_off;
        const int shift = (c->mode == MR_MODE_ROW_OFF_1);
        const int xpose = (c->mode == MR_MODE_WG_TRANSPOSED);
        for (int l = 0; l < MR_E; ++l) st->logits[l] = mr_logit_row(x, wg, l, shift, xpose,
                                                                    &c->bad_index);
    } else {
        for (int l = 0; l < MR_E; ++l) st->logits[l] = slab[c->lg_off + l];
    }
    mr_softmax_row(st->logits, c->mode, st->probs);
    mr_topk_row(st->probs, c->mode, st->ids);
    mr_renorm_row(st->probs, st->ids, c->mode, st->wpre, st->wcast);
    mr_combine_row(slab + c->raw_off, st->wpre, st->wcast, st->ids, c->mode, st->routed);

    /* The three bf16 fields are widened into the output buffer by SHIFTING, not by reinterpreting:
     * mr_bits_to_f32 takes a 32-bit pattern, and feeding it a uint16_t silently produces a denormal
     * instead of the value. The gate compares these words bit-for-bit against the dump's bit image
     * shifted the same way, so the shift is the output contract (the driver selftest's
     * `output_widening` check is the witness that it holds). */
    for (int l = 0; l < MR_E; ++l) out[MR_OFF_LOGITS + l] = mr_bf16_to_f32(st->logits[l]);
    for (int l = 0; l < MR_E; ++l) out[MR_OFF_PROBS + l] = st->probs[l];
    for (int k = 0; k < MR_K; ++k) out[MR_OFF_WPRE + k] = st->wpre[k];
    /* R5's output, widened. The gate separately checks bf16_bits(WPRE) == bf16_bits(WCAST) from
     * these words, which is the double-rounding check; the kernel does not police itself here. */
    for (int k = 0; k < MR_K; ++k) out[MR_OFF_WCAST + k] = mr_bf16_to_f32(st->wcast[k]);
    for (int k = 0; k < MR_K; ++k) out[MR_OFF_IDS + k] = (float)st->ids[k];
    for (int h = 0; h < MR_H; ++h) out[MR_OFF_ROUTED + h] = mr_bf16_to_f32(st->routed[h]);

    for (int f = 0; f < MR_FIELD_COUNT; ++f) c->bad[f] = 0;
    if (c->cmp_off >= 0) {
        for (int f = 0; f < MR_FIELD_COUNT; ++f) {
            const int off = mr_field_off(f), len = mr_field_len(f);
            for (int i = 0; i < len; ++i)
                if (mr_f32_bits(out[off + i]) != mr_f32_bits(buf[c->cmp_off + off + i])) ++c->bad[f];
        }
    }
}

/* (the field table is above mr_row, one spelling of offset and length) */

/* --------------------------------------------------------------------------------------------- */
/* plan verification, shared so the host probe and the device driver refuse a malformed plan for
 * the same reason. Every increment of `bad` is a check gate.py counts against its roster.        */
/* --------------------------------------------------------------------------------------------- */
MR_DEV inline int mr_verify_cases(const struct MrCase *cs, int n,
                                               long long slab_elems, long long out_floats,
                                               int *checks_run) {
    int bad = 0, nm0 = 0;
    for (int i = 0; i < n; ++i) {
        const struct MrCase c = cs[i];
        if (c.id != i) ++bad;
        if (c.arm != MR_ARM_KERNEL && c.arm != MR_ARM_LOGITS) ++bad;
        if (c.mode < 0 || c.mode >= MR_MODE_COUNT) ++bad;
        if (c.tie_class < 0 || c.tie_class > 7) ++bad;
        if (c.raw_src != 0 && c.raw_src != 1) ++bad;
        if ((long long)c.dst_off + MR_CASE_FLOATS > out_floats) ++bad;
        if ((long long)c.raw_off + (long long)MR_K * MR_H > slab_elems) ++bad;
        if ((long long)c.lg_off + MR_E > slab_elems) ++bad;
        if ((long long)c.x_off + MR_H > slab_elems) ++bad;
        if ((long long)c.wg_off + (long long)MR_E * MR_H > slab_elems) ++bad;
        if (c.mode == 0) {
            ++nm0;
            if (c.cmp_off != -1) ++bad;                     /* the ref case compares to nothing */
        } else {
            if (c.cmp_off < 0) ++bad;
            else if (((long long)c.cmp_off + MR_CASE_FLOATS > out_floats) ||
                     ((c.cmp_off % MR_CASE_FLOATS) != 0)) ++bad;  /* must name a whole record */
        }
        ++*checks_run;
    }
    if (n > 0 && nm0 == 0) ++bad;                            /* injections compare against nothing */
    return bad;
}

#endif /* MOE_ROUTER_KERNELS_H */
