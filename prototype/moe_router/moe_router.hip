/* moe_router.hip — the device arm of prototype/moe_router: R1..R5 of the MoE router plus the one
 * multiply that makes R5's cast position observable, on the iGPU, judged against the pinned dumps.
 *
 * Compiled once by hipcc for gfx1151; the arithmetic is in moe_router_kernels.h and is compiled a
 * second time by host_probe.cpp under g++ so a host/device disagreement names the compiler.
 *
 * NO TIMER IN THIS FILE. Correctness only: no clock, no warm loop, no burst, no best-of-N. The
 * lock record's wall clock is printed by the `gate` wrapper, not here.
 *
 * SHAPE, AND WHY IT IS SHAPED LIKE THIS
 *   One workgroup per case, the row's serial spine executed by lane 0 by calling the same mr_row()
 *   the host probe calls. That is deliberately not the fastest shape: it is the shape that makes the
 *   host and device words identical bit-for-bit, which is what the attribution claim needs
 *   (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the thing
 *   under test"). The parallelisable half — R1's 256 independent 2048-term dots, R2's 256
 *   independent exps — is written up in results.md as the finding prediction 1 asks for, not
 *   optimised here.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "moe_router_kernels.h"

static int g_failures = 0;

static void fail(const char *fmt) {
    printf("FAIL %s\n", fmt);
    ++g_failures;
}

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t _e = (x);                                               \
        if (_e != hipSuccess) {                                            \
            printf("FAIL hip %s at %s:%d: %s\n", #x, __FILE__, __LINE__,   \
                   hipGetErrorString(_e));                                 \
            return 1;                                                      \
        }                                                                  \
    } while (0)

/* --------------------------------------------------------------------------------------------- */
/* the kernel                                                                                      */
/* --------------------------------------------------------------------------------------------- */
MR_GLOBAL void mr_router_kernel(struct MrCase *cases, const uint16_t *slab, float *out, int n) {
    __shared__ struct MrRowState st;
    __shared__ struct MrCase csh;
    const int i = blockIdx.x;
    if (i >= n) return;
    if (threadIdx.x == 0) csh = cases[i];
    __syncthreads();
    if (threadIdx.x == 0) mr_row(slab, &csh, &st, out);   /* buf base: mr_row indexes by dst_off */
    __syncthreads();
    if (threadIdx.x == 0) cases[i] = csh;                  /* carry the per-field counters back out */
}

/* --------------------------------------------------------------------------------------------- */
/* plan parsing: one TSV, a GEOM line and one CASE line per case                                   */
/* --------------------------------------------------------------------------------------------- */
static int read_plan(const char *path, std::vector<struct MrCase> *cases, long long *slab_elems,
                     long long *out_floats, int *geom_e, int *geom_k, int *geom_h, int *nprinted) {
    FILE *fh = fopen(path, "r");
    if (!fh) {
        printf("FAIL cannot open plan %s: %s\n", path, strerror(errno));
        return 1;
    }
    char line[1024];
    int bad = 0;
    *nprinted = 0;
    while (fgets(line, sizeof(line), fh)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') continue;
        ++*nprinted;
        if (strncmp(line, "GEOM", 4) == 0) {
            int e = 0, k = 0, h = 0, se = 0, of = 0;
            if (sscanf(line + 4, " E=%d K=%d H=%d slab_elems=%d out_floats=%d", &e, &k, &h, &se,
                       &of) != 5) {
                printf("FAIL plan GEOM line malformed: %s", line);
                ++bad;
                continue;
            }
            *geom_e = e; *geom_k = k; *geom_h = h; *slab_elems = se; *out_floats = of;
            continue;
        }
        if (strncmp(line, "CASE", 4) != 0) {
            printf("FAIL plan line is neither GEOM nor CASE: %s", line);
            ++bad;
            continue;
        }
        struct MrCase c;
        memset(&c, 0, sizeof(c));
        int got = sscanf(line + 4,
                         " %d %d %d %d %d %d %d %d %d %d %d %d %d",
                         &c.id, &c.arm, &c.mode, &c.tie_class, &c.id_asc, &c.raw_src, &c.src_key,
                         &c.x_off, &c.wg_off, &c.lg_off, &c.raw_off, &c.dst_off, &c.cmp_off);
        if (got != 13) {
            printf("FAIL plan CASE line has %d of 13 fields: %s", got, line);
            ++bad;
            continue;
        }
        cases->push_back(c);
    }
    fclose(fh);
    if (bad) {
        printf("FAIL plan %s: %d malformed lines\n", path, bad);
        return 1;
    }
    return 0;
}

/* --------------------------------------------------------------------------------------------- */
/* driver selftest: no device, no dumps, and every check can fail                                  */
/* --------------------------------------------------------------------------------------------- */
static uint32_t fbits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

static void expect_bits(const char *what, float got, uint32_t want) {
    if (fbits(got) != want) {
        printf("FAIL selftest %s: got %08x want %08x\n", what, fbits(got), want);
        ++g_failures;
    }
}

/* An independent bf16 rounding: the correctly-rounded double route, which shares no line of
 * arithmetic with mr_f32_to_bf16. A selftest whose second ruler was written from the first is
 * decoration. */
static uint16_t bf16_via_double(double d) {
    uint64_t b;
    memcpy(&b, &d, 8);
    const int neg = (int)((b >> 63) & 1u);
    uint64_t mant = b & 0xFFFFFFFFFFFFFULL;
    int exp = (int)((b >> 52) & 0x7FFu) - 1023;
    if ((b & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) {
        if (mant != 0) return 0x7FC0u;                       /* NaN, sign dropped as documented */
        return (uint16_t)((neg << 15) | 0x7F80);
    }
    if (exp < -133) {                                        /* bf16 subnormal territory */
        const double scale = 1.0 / 18446744073709551616.0;   /* 2^-64 */
        double ad = d < 0 ? -d : d;
        uint64_t q = (uint64_t)(ad / scale + 0.5);           /* double has enough bits here */
        return (uint16_t)((neg << 15) | (q & 0x7FFFu));
    }
    /* 52 stored mantissa bits -> 7 kept: round half to even on bit 8 of the discarded field */
    const uint64_t keep = mant >> 45;
    const uint64_t rest = mant & ((1ULL << 45) - 1);
    const uint64_t half = 1ULL << 44;
    uint64_t r = keep;
    if (rest > half || (rest == half && (keep & 1))) ++r;
    if (r == 0x100) { r = 0; ++exp; }
    if (exp > 128) return (uint16_t)((neg << 15) | 0x7F80);
    if (exp < -126) {                                        /* subnormal bf16: renormalise */
        const int shift = -126 - exp + 8;
        const uint64_t full = (1ULL << 8) | keep;            /* implicit leading one */
        uint64_t q = full >> shift;
        const uint64_t rembits = full & ((1ULL << shift) - 1);
        const uint64_t h2 = 1ULL << (shift - 1);
        if (rembits > h2 || (rembits == h2 && (q & 1))) ++q;
        return (uint16_t)((neg << 15) | (q & 0x7FFFu));
    }
    return (uint16_t)((neg << 15) | ((uint32_t)(exp + 127) << 7) | (uint32_t)r);
}

static int driver_selftest() {
    int n = 0;
    /* --- 1. the bf16 cast, hand-worked. bf16 is 1+8+7 bits: 7 STORED mantissa bits, so one ULP at
     *     1.0 is 2^-7 and the f32->bf16 cast keeps the top 16 of the 32 bits (23-7 = 16). The
     *     exponents below are therefore 127+e, and 1/256 = 2^-8 is 0x3B80, 1/8 = 2^-3 is 0x3E00.  */
    expect_bits("bf16 +1.0", mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(1.0f)) << 16), 0x3F800000u);
    expect_bits("bf16 -1.0", mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(-1.0f)) << 16), 0xBF800000u);
    expect_bits("bf16 +0.0", mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(0.0f)) << 16), 0x00000000u);
    expect_bits("bf16 -0.0 keeps its sign",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(-0.0f)) << 16), 0x80000000u);
    expect_bits("bf16 1.0078125 (1+2^-7, exact)",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(1.0078125f)) << 16), 0x3F810000u);
    expect_bits("bf16 tie 1.00390625 -> even 1.0",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(1.00390625f)) << 16), 0x3F800000u);
    expect_bits("bf16 tie 1.01171875 -> even 1.015625",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(1.01171875f)) << 16), 0x3F820000u);
    expect_bits("bf16 tie -1.00390625 -> even -1.0",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(-1.00390625f)) << 16), 0xBF800000u);
    expect_bits("bf16 2^-133 (min subnormal)",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(mr_bits_to_f32(0x00010000u))) << 16),
                0x00010000u);
    expect_bits("bf16 +inf", mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(
                                mr_bits_to_f32(0x7F800000u))) << 16), 0x7F800000u);
    expect_bits("bf16 -inf", mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(
                                mr_bits_to_f32(0xFF800000u))) << 16), 0xFF800000u);
    expect_bits("bf16 qNaN canonicalises",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(mr_bits_to_f32(0x7FC00001u))) << 16),
                0x7FC00000u);
    expect_bits("bf16 sNaN canonicalises, sign dropped",
                mr_bits_to_f32(((uint32_t)mr_f32_to_bf16(mr_bits_to_f32(0xFFC00001u))) << 16),
                0x7FC00000u);
    printf("SELFTEST name=bf16_cast ok=%d n=12\n", g_failures == 0 ? 1 : 0);
    ++n;
    /* --- 2. the same cast against a second, independently written ruler, swept ---------------- */
    {
        int diff = 0, swept = 0;
        for (uint32_t e = 100; e <= 140; ++e) {              /* exponents around the router's range */
            for (uint32_t m = 0; m < 1024; ++m) {
                const uint32_t b = (e << 23) | (m << 12);
                const float f = mr_bits_to_f32(b);
                if (mr_f32_to_bf16(f) != bf16_via_double(f)) ++diff;
                ++swept;
            }
        }
        if (swept < 40000) { printf("FAIL selftest bf16 sweep covered only %d values\n", swept);
                             ++g_failures; }
        if (diff) { printf("FAIL selftest bf16 sweep: %d of %d disagree with the double route\n",
                           diff, swept); ++g_failures; }
        printf("SELFTEST name=bf16_double_ruler ok=%d n=%d\n", diff == 0 ? 1 : 0, swept);
    }
    ++n;
    /* --- 3. bf16 += bf16, hand-worked --------------------------------------------------------- */
    {
        const uint16_t one = mr_f32_to_bf16(1.0f);
        if (mr_bf16_add(one, mr_f32_to_bf16(1.0f)) != 0x4000u) {   /* 2.0 = 2^1 -> 0x4000 */
            printf("FAIL selftest bf16_add 1+1 = %04x, want 4000 (2.0)\n",
                   mr_bf16_add(one, mr_f32_to_bf16(1.0f)));
            ++g_failures;
        }
        /* 1.0 + 2^-8 is an exact tie in bf16 (half of the 2^-7 spacing) and must round to the even
         * significand 1.0 */
        if (mr_bf16_add(one, mr_f32_to_bf16(0.00390625f)) != 0x3F80u) {
            printf("FAIL selftest bf16_add tie did not round to even\n");
            ++g_failures;
        }
        printf("SELFTEST name=bf16_add ok=%d n=2\n", g_failures == 0 ? 1 : 0);
    }
    ++n;
    /* --- 4. R2/R3/R4 on an all-equal logit row: everything is a power of two and exact --------- */
    {
        static uint16_t lg[MR_E];
        float probs[MR_E], wpre[MR_K];
        uint16_t wcast[MR_K];
        int ids[MR_K];
        for (int l = 0; l < MR_E; ++l) lg[l] = mr_f32_to_bf16(0.0f);
        mr_softmax_row(lg, MR_MODE_CORRECT, probs);
        int ok = (fbits(probs[0]) == 0x3B800000u);           /* 1/256 = 2^-8 exactly */
        for (int l = 1; l < MR_E; ++l) ok &= (fbits(probs[l]) == 0x3B800000u);
        if (!ok) { printf("FAIL selftest softmax of equal logits is not 1/256 (%08x)\n",
                          fbits(probs[0])); ++g_failures; }
        mr_topk_row(probs, MR_MODE_CORRECT, ids);
        for (int k = 0; k < MR_K; ++k)
            if (ids[k] != k) { printf("FAIL selftest tie-break did not take the lowest ids\n");
                               ++g_failures; break; }
        mr_topk_row(probs, MR_MODE_TIE_HIGH, ids);
        for (int k = 0; k < MR_K; ++k)
            if (ids[k] != MR_E - 1 - k) {
                printf("FAIL selftest tie-high did not take the highest ids (slot %d got %d)\n", k,
                       ids[k]);
                ++g_failures; break;
            }
        mr_topk_row(probs, MR_MODE_CORRECT, ids);
        mr_renorm_row(probs, ids, MR_MODE_CORRECT, wpre, wcast);
        /* sum = 8/256 = 2^-5, so every weight is exactly 1/8 = 2^-3 = 0x3E000000 */
        for (int k = 0; k < MR_K; ++k) expect_bits("equal-row weight 1/8", wpre[k], 0x3E000000u);
        for (int k = 0; k < MR_K; ++k)
            if (wcast[k] != 0x3E00u) { printf("FAIL selftest equal-row cast weight %04x\n",
                                               wcast[k]); ++g_failures; break; }
        /* and the ids ascending means the combine-order injections are structurally immune here —
         * the same predicate the plan carries, computed here so the two cannot silently differ */
        int asc = 1;
        for (int k = 1; k < MR_K; ++k) asc &= (ids[k] > ids[k - 1]);
        if (!asc) fail("selftest equal-row ids are not ascending, so the immunity fixture is wrong");
        printf("SELFTEST name=equal_row ok=%d n=%d\n", g_failures == 0 ? 1 : 0, 2 * MR_E + 3 * MR_K);
    }
    ++n;
    /* --- 5. prediction 4 in six numbers: the two cast orders, hand-worked to differ ----------- *
     * One live addend: raw = 1.00390625 (bf16 0x3F81), w_f32 = 1.00390625, whose bf16 cast is 1.0
     * (a tie, rounded to the even significand).
     *   cast-before: 1.00390625 * 1.0        = 1.00390625 -> bf16 0x3F81
     *   cast-after : 1.00390625 * 1.00390625 = 1.00783...  -> bf16 0x3F82   (nearest is 0x3F82:
     *     1.0078125 lies 1.8e-5 away, 1.01171875 lies 3.9e-3 away)
     * If these two ever come out the same word, the gate's cast comparator is comparing nothing. */
    {
        static uint16_t raw[MR_K * MR_H];
        float wpre[MR_K];
        uint16_t wcast[MR_K];
        int ids[MR_K];
        static uint16_t r_before[MR_H], r_after[MR_H];
        for (int i = 0; i < MR_K * MR_H; ++i) raw[i] = 0x3F81u;
        for (int k = 0; k < MR_K; ++k) {
            wpre[k] = (k == 0) ? 1.00390625f : 0.0f;
            wcast[k] = mr_f32_to_bf16(wpre[k]);
            ids[k] = 8 * k + 7;                              /* not ascending: the order injections
                                                              * would have work to do on this row */
        }
        if (wcast[0] != 0x3F80u) { printf("FAIL selftest cast weight %04x want 3F80\n", wcast[0]);
                                   ++g_failures; }
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, r_before);
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CAST_AFTER, r_after);
        expect_bits("cast-before single addend",
                    mr_bits_to_f32(((uint32_t)r_before[0]) << 16), 0x3F810000u);
        expect_bits("cast-after single addend",
                    mr_bits_to_f32(((uint32_t)r_after[0]) << 16), 0x3F820000u);
        if (r_before[0] == r_after[0])
            fail("cast-before and cast-after produced the SAME product: the comparator cannot tell "
                 "the two orders apart and is not a comparator");
        /* and all eight addends live, against the same algorithm evaluated in doubles: an
         * independent implementation of mr_combine_row, not a copy of it */
        for (int k = 0; k < MR_K; ++k) { wpre[k] = 0.125f; wcast[k] = 0x3E00u; }
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, r_before);
        int order[MR_K];
        for (int k = 0; k < MR_K; ++k) order[k] = k;
        for (int a = 0; a < MR_K - 1; ++a)
            for (int b = 0; b < MR_K - 1 - a; ++b)
                if (ids[order[b]] > ids[order[b + 1]]) { const int t = order[b]; order[b] =
                    order[b + 1]; order[b + 1] = t; }
        int diff = 0;
        for (int h = 0; h < MR_H; ++h) {
            double accd = 0.0;
            for (int i = 0; i < MR_K; ++i) {
                /* prod = bf16(raw * w), then acc = bf16(acc + prod): the same algorithm as
                 * mr_combine_row, evaluated on doubles with an independently written rounding. */
                const double prod = (double)mr_bits_to_f32(((uint32_t)bf16_via_double(
                    (double)mr_bf16_to_f32(raw[(long)order[i] * MR_H + h]) * 0.125)) << 16);
                accd = (double)mr_bits_to_f32(((uint32_t)bf16_via_double(accd + prod)) << 16);
            }
            if (r_before[h] != bf16_via_double(accd)) ++diff;
        }
        if (diff) { printf("FAIL selftest combine vs the double route: %d of %d differ\n", diff,
                           MR_H); ++g_failures; }
        printf("SELFTEST name=cast_order_differs ok=%d n=%d\n", g_failures == 0 ? 1 : 0,
               4 * MR_H);
    }
    ++n;
    /* --- 6. the pairing injection is not the ordering injection (mode 5 vs mode 10) ------------ */
    {
        static uint16_t raw[MR_K * MR_H];
        float wpre[MR_K];
        uint16_t wcast[MR_K];
        int ids[MR_K];
        static uint16_t a[MR_H], b[MR_H], c[MR_H];
        for (int i = 0; i < MR_K * MR_H; ++i) raw[i] = 0x3F80u;
        /* DISTINCT weights: with all eight equal, indexing w by the expert id instead of by the rank
         * selects an identical value and the pairing injection cannot fire. A must-catch whose
         * fixture makes it immune is decoration (pitfalls/40). */
        for (int k = 0; k < MR_K; ++k) { wpre[k] = 0.0625f + 0.015625f * (float)k;
                                         wcast[k] = mr_f32_to_bf16(wpre[k]);
                                         ids[k] = 200 - 24 * k; }
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, a);
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_W_BY_EXPERT_ID, b);
        int asc = 1;
        for (int k = 1; k < MR_K; ++k) asc &= (ids[k] > ids[k - 1]);
        if (asc) fail("selftest pairing fixture is already ascending; mode 5 would be immune");
        int paired = 1;
        for (int k = 0; k < MR_K; ++k) paired &= (ids[k] % MR_K == k);
        if (paired) fail("selftest pairing fixture satisfies ids%%K==k; mode 5 would be immune");
        int d5 = 0, d4 = 0;
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_RANK_ACCUM, c);
        for (int h = 0; h < MR_H; ++h) {
            if (a[h] != b[h]) ++d5;
            if (a[h] != c[h]) ++d4;
        }
        if (d5 == 0) fail("the weight-by-expert-id injection did not fire on a non-ascending row");
        printf("SELFTEST name=pairing_vs_order ok=%d n=%d d5=%d d4=%d\n", d5 > 0 ? 1 : 0,
               3 * MR_H, d5, d4);
    }
    ++n;
    /* --- 7. the leak guard: mode 12 writes the canonical NaN into EVERY slot ------------------ */
    {
        /* big enough for one combine's raw (K*H), one logit row (E) and one activation (H): the
         * offsets below are the only reason the kernel is allowed to read this array. */
        static uint16_t slab[MR_K * MR_H + MR_E + MR_H];
        std::vector<float> out((size_t)MR_CASE_FLOATS, 0.0f);
        struct MrCase c;
        struct MrRowState st;
        memset(&c, 0, sizeof(c));
        memset(slab, 0, sizeof(slab));
        c.id = 0; c.arm = MR_ARM_LOGITS; c.mode = MR_MODE_POISON_NAN; c.cmp_off = -1;
        c.lg_off = MR_K * MR_H;
        c.x_off = MR_K * MR_H + MR_E;
        c.wg_off = 0;                                        /* arm LOGITS never reads it */
        c.raw_off = 0;
        mr_row(slab, &c, &st, out.data());
        int nan = 1;
        for (int i = 0; i < MR_CASE_FLOATS; ++i) nan &= (fbits(out[i]) == MR_NAN_BITS);
        if (!nan) fail("poison mode left a slot that is not the canonical NaN");
        /* and a real mode must leave NONE: a stage that forgets its field would show up here */
        c.mode = MR_MODE_CORRECT;
        mr_row(slab, &c, &st, out.data());
        int left = 0;
        for (int i = 0; i < MR_CASE_FLOATS; ++i) left += (fbits(out[i]) == MR_NAN_BITS);
        if (left) { printf("FAIL selftest correct mode left %d NaN slots\n", left); ++g_failures; }
        printf("SELFTEST name=leak_guard ok=%d n=2\n", g_failures == 0 ? 1 : 0);
        /* --- 7b. the output-widening contract: every bf16 field lands as bits << 16 -------------
         * The gate compares these words bit-for-bit against the dump's bit image shifted the same
         * way. Writing them with mr_bits_to_f32(uint16_t) — which reinterprets 32 bits and so
         * silently yields a denormal — passed every other selftest, so this one is the witness for
         * the output path specifically. */
        {
            for (int l = 0; l < MR_E; ++l) slab[c.lg_off + l] = (uint16_t)(0x3c00u + 3 * l);
            for (int i = 0; i < MR_K * MR_H; ++i) slab[i] = (uint16_t)(0x3800u + (i % 251u));
            c.mode = MR_MODE_CORRECT;
            mr_row(slab, &c, &st, out.data());
            int widened = 1;
            for (int l = 0; l < MR_E; ++l)
                widened &= (fbits(out[MR_OFF_LOGITS + l]) == ((uint32_t)st.logits[l] << 16));
            for (int k = 0; k < MR_K; ++k) {
                widened &= (fbits(out[MR_OFF_WCAST + k]) == ((uint32_t)st.wcast[k] << 16));
                widened &= ((fbits(out[MR_OFF_WCAST + k]) & 0xFFFFu) == 0u);
            }
            for (int h = 0; h < MR_H; ++h) {
                widened &= (fbits(out[MR_OFF_ROUTED + h]) == ((uint32_t)st.routed[h] << 16));
                widened &= ((fbits(out[MR_OFF_ROUTED + h]) & 0xFFFFu) == 0u);
            }
            /* and the supplied logits really travelled, which is what arm LOGITS is for */
            for (int l = 0; l < MR_E; ++l)
                widened &= (fbits(out[MR_OFF_LOGITS + l]) == ((uint32_t)slab[c.lg_off + l] << 16));
            if (!widened) fail("the bf16 fields are not widened into the output as bits << 16");
            printf("SELFTEST name=output_widening ok=%d n=%d\n", g_failures == 0 ? 1 : 0,
                   MR_E + 2 * MR_K + 2 * MR_H);
        }
        ++n;
    }
    ++n;
    /* --- 8. plan verification actually refuses a bad plan ------------------------------------- */
    {
        std::vector<struct MrCase> cs(3);
        int checks = 0;
        for (int i = 0; i < 3; ++i) {
            memset(&cs[i], 0, sizeof(struct MrCase));
            cs[i].id = i; cs[i].arm = MR_ARM_KERNEL; cs[i].mode = i;   /* 0,1,2 */
            cs[i].dst_off = i * MR_CASE_FLOATS;
            cs[i].cmp_off = (i == 0) ? -1 : 0;
        }
        if (mr_verify_cases(&cs[0], 3, 1LL << 30, 3LL * MR_CASE_FLOATS, &checks) != 0)
            fail("a well-formed plan was refused");
        if (checks != 3) { printf("FAIL verify_cases ran %d checks for 3 cases\n", checks);
                           ++g_failures; }
        cs[2].mode = MR_MODE_COUNT;                            /* out of range */
        if (mr_verify_cases(&cs[0], 3, 1LL << 30, 3LL * MR_CASE_FLOATS, &checks) == 0)
            fail("verify_cases accepted a mode out of range");
        cs[2].mode = 2; cs[2].cmp_off = -1;
        if (mr_verify_cases(&cs[0], 3, 1LL << 30, 3LL * MR_CASE_FLOATS, &checks) == 0)
            fail("verify_cases accepted an injection with nothing to compare against");
        cs[2].cmp_off = 0; cs[2].raw_off = 1 << 30;          /* at or past the slab end */
        if (mr_verify_cases(&cs[0], 3, 1LL << 30, 3LL * MR_CASE_FLOATS, &checks) == 0)
            fail("verify_cases accepted a slab read out of range");
        printf("SELFTEST name=verify_cases ok=%d n=%d\n", g_failures == 0 ? 1 : 0, checks);
    }
    ++n;
    printf("DRIVER SELFTEST RESULT checks=%d failures=%d\n", n, g_failures);
    if (g_failures) { printf("RESULT FAIL driver selftest failures=%d\n", g_failures); return 1; }
    printf("RESULT PASS driver selftest checks=%d\n", n);
    return 0;
}

/* --------------------------------------------------------------------------------------------- */
/* main                                                                                           */
/* --------------------------------------------------------------------------------------------- */
static const char *arg_value(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}
static int have_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    /* Unbuffered: a driver that dies mid-run must still have printed the records it produced
     * (pitfalls/50 §"a check that can produce no output cannot fail" — a lost buffer is the same
     * failure wearing a different hat). */
    setvbuf(stdout, NULL, _IONBF, 0);
    if (have_flag(argc, argv, "--selftest")) return driver_selftest();
    if (have_flag(argc, argv, "--reach-table")) {
        /* The reachability table printed by the compiled predicate rather than re-implemented in the
         * gate: one spelling of the claim, and the gate counts the records it parsed against the
         * count printed here. Host-only — no HIP call is made on this path, so it takes no lock. */
        int rec = 0;
        for (int a = 0; a < 2; ++a)
            for (int m = 0; m < MR_MODE_COUNT; ++m)
                for (int f = 0; f < MR_FIELD_COUNT; ++f) {
                    printf("REACH arm=%d mode=%d field=%d v=%d\n", a, m, f,
                           mr_mode_reaches(m, f, a));
                    ++rec;
                }
        printf("REACHCOUNT arms=2 modes=%d fields=%d records=%d\n", MR_MODE_COUNT, MR_FIELD_COUNT,
               rec);
        printf("RESULT PASS reach table records=%d\n", rec);
        return 0;
    }

    const char *plan_path = arg_value(argc, argv, "--plan", NULL);
    const char *slab_path = arg_value(argc, argv, "--slab", NULL);
    const char *out_path = arg_value(argc, argv, "--out", NULL);
    const char *rep_path = arg_value(argc, argv, "--report", NULL);
    const int expect_cases = (int)strtol(arg_value(argc, argv, "--expect-cases", "-1"), NULL, 10);
    if (!plan_path || !slab_path || !out_path || !rep_path) {
        printf("usage: moe_router --plan P --slab S --out O --report R [--expect-cases N]\n"
               "       moe_router --selftest | --check\n");
        return 2;
    }
    std::vector<struct MrCase> cases;
    long long slab_elems = 0, out_floats = 0;
    int E = 0, K = 0, H = 0, nprinted = 0;
    if (read_plan(plan_path, &cases, &slab_elems, &out_floats, &E, &K, &H, &nprinted)) return 1;
    printf("PLAN plan_file=%s cases_parsed=%d cases_lines=%d E=%d K=%d H=%d slab_elems=%lld "
           "out_floats=%lld\n", plan_path, (int)cases.size(), nprinted, E, K, H, slab_elems,
           out_floats);
    if (nprinted != (int)cases.size() + 1) {     /* +1: the GEOM line is the other printed record */
        printf("FAIL the plan printed %d records and parsed %d cases (expect cases+1)\n", nprinted,
               (int)cases.size());
        return 1;
    }
    if (E != MR_E || K != MR_K || H != MR_H) {
        printf("FAIL the plan's geometry E=%d K=%d H=%d disagrees with the kernel's %d/%d/%d\n", E,
               K, H, MR_E, MR_K, MR_H);
        return 1;
    }
    if (expect_cases >= 0 && (int)cases.size() != expect_cases) {
        printf("FAIL plan holds %zu cases, the gate announced %d\n", cases.size(), expect_cases);
        return 1;
    }
    std::vector<uint16_t> slab;
    {
        FILE *fh = fopen(slab_path, "rb");
        if (!fh) { printf("FAIL cannot open slab %s: %s\n", slab_path, strerror(errno)); return 1; }
        fseek(fh, 0, SEEK_END);
        long long nb = ftello(fh);
        fseek(fh, 0, SEEK_SET);
        if (nb % 2) { printf("FAIL slab size %lld is not a whole number of bf16 elements\n", nb);
                      fclose(fh); return 1; }
        slab.resize((size_t)(nb / 2));
        if (nb && fread(slab.data(), 1, (size_t)nb, fh) != (size_t)nb) {
            printf("FAIL short read on slab\n"); fclose(fh); return 1;
        }
        fclose(fh);
    }
    if ((long long)slab.size() != slab_elems) {
        printf("FAIL slab holds %zu elements, the plan's GEOM says %lld\n", slab.size(), slab_elems);
        return 1;
    }
    int checks = 0;
    const int bad = mr_verify_cases(cases.data(), (int)cases.size(), (long long)slab.size(),
                                    out_floats, &checks);
    printf("PLANCHECK cases=%zu checks_run=%d malformed=%d\n", cases.size(), checks, bad);
    if (bad) { printf("RESULT FAIL plan verification malformed=%d\n", bad); return 1; }

    if (have_flag(argc, argv, "--check")) {
        printf("RESULT PASS check plan cases=%zu slab_elems=%lld (no device touched)\n",
               cases.size(), slab_elems);
        return 0;
    }

    /* ---- device work ------------------------------------------------------------------------- */
    struct MrCase *d_cases = nullptr;
    uint16_t *d_slab = nullptr;
    float *d_out = nullptr;
    CK(hipMalloc(&d_cases, sizeof(struct MrCase) * cases.size()));
    CK(hipMalloc(&d_slab, sizeof(uint16_t) * slab.size()));
    CK(hipMalloc(&d_out, sizeof(float) * (size_t)out_floats));
    CK(hipMemcpy(d_cases, cases.data(), sizeof(struct MrCase) * cases.size(),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_slab, slab.data(), sizeof(uint16_t) * slab.size(), hipMemcpyHostToDevice));
    /* Unwritten slots must read the canonical NaN, not 0.0: fill the buffer with the NaN pattern
     * before anything runs, so the gate's per-field NaN count is a real guard. */
    {
        std::vector<float> seed((size_t)out_floats, mr_bits_to_f32(MR_NAN_BITS));
        CK(hipMemcpy(d_out, seed.data(), sizeof(float) * (size_t)out_floats,
                     hipMemcpyHostToDevice));
    }
    /* One launch per mode, mode 0 first, so an injection case can compare against the same row's
     * reference record while it is still in device memory. */
    for (int mode = 0; mode < MR_MODE_COUNT; ++mode) {
        std::vector<struct MrCase> sub;
        for (size_t i = 0; i < cases.size(); ++i)
            if (cases[i].mode == mode) sub.push_back(cases[i]);
        if (sub.empty()) continue;
        struct MrCase *d_sub = nullptr;
        CK(hipMalloc(&d_sub, sizeof(struct MrCase) * sub.size()));
        CK(hipMemcpy(d_sub, sub.data(), sizeof(struct MrCase) * sub.size(), hipMemcpyHostToDevice));
        mr_router_kernel<<<(int)sub.size(), 256>>>(d_sub, d_slab, d_out, (int)sub.size());
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(sub.data(), d_sub, sizeof(struct MrCase) * sub.size(), hipMemcpyDeviceToHost));
        /* write the kernel's own counters back into the master list by id */
        for (size_t j = 0; j < sub.size(); ++j) cases[sub[j].id] = sub[j];
        CK(hipFree(d_sub));
    }
    std::vector<float> h_out((size_t)out_floats, 0.0f);
    CK(hipMemcpy(h_out.data(), d_out, sizeof(float) * (size_t)out_floats, hipMemcpyDeviceToHost));
    CK(hipFree(d_cases));
    CK(hipFree(d_slab));
    CK(hipFree(d_out));

    /* The device's own words, dumped raw. This file — not any number the driver prints — is what
     * gate.py compares, element by element, against the pinned dumps. */
    FILE *of = fopen(out_path, "wb");
    if (!of) { printf("FAIL cannot write out %s: %s\n", out_path, strerror(errno)); return 1; }
    if (out_floats && fwrite(h_out.data(), sizeof(float), (size_t)out_floats, of) !=
                          (size_t)out_floats) {
        printf("FAIL short write on out\n"); fclose(of); return 1;
    }
    if (fflush(of) || ferror(of)) { printf("FAIL flush of out\n"); fclose(of); return 1; }
    fclose(of);
    printf("OUT file=%s floats=%lld bytes=%lld\n", out_path, out_floats, 4LL * out_floats);

    FILE *rp = fopen(rep_path, "wb");
    if (!rp) { printf("FAIL cannot write report %s: %s\n", rep_path, strerror(errno)); return 1; }
    fprintf(rp, "# id\tarm\tmode\ttie\tid_asc\traw\tsrc_key\tbad_logits\tbad_probs\tbad_wpre\t"
                "bad_wcast\tbad_ids\tbad_routed\tnan\tbad_index\twcast_hex\tids\n");
    int nrows = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const struct MrCase &c = cases[i];
        int nan = 0;
        for (int f = 0; f < MR_FIELD_COUNT; ++f) {
            const int off = mr_field_off(f), len = mr_field_len(f);
            for (int j = 0; j < len; ++j)
                if (fbits(h_out[c.dst_off + off + j]) == MR_NAN_BITS) ++nan;
        }
        char whex[8 * MR_K + 1];
        for (int k = 0; k < MR_K; ++k)
            sprintf(whex + 8 * k, "%08x ", fbits(h_out[c.dst_off + MR_OFF_WCAST + k]));
        char idstr[8 * MR_K + 1];
        for (int k = 0; k < MR_K; ++k)
            sprintf(idstr + 8 * k, "%d ", (int)h_out[c.dst_off + MR_OFF_IDS + k]);
        fprintf(rp, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",
                c.id, c.arm, c.mode, c.tie_class, c.id_asc, c.raw_src, c.src_key,
                c.bad[0], c.bad[1], c.bad[2], c.bad[3], c.bad[4], c.bad[5], nan, c.bad_index,
                whex, idstr);
        ++nrows;
    }
    fclose(rp);
    printf("REPORT file=%s rows=%d\n", rep_path, nrows);
    if (nrows != (int)cases.size()) {
        printf("FAIL the report holds %d rows for %zu cases\n", nrows, cases.size());
        return 1;
    }
    printf("RESULT PASS device plan cases=%zu out_floats=%lld\n", cases.size(), out_floats);
    return 0;
}
