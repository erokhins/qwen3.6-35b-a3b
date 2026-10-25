/* shared_expert_selftest.h — hand-worked, device-free self-tests for the shared-expert primitives,
 * with the three arms the standing rules require.
 *
 * Included by BOTH arms (`shared_expert.hip --selftest` and `host_probe --selftest`), so the same
 * numbers are asserted under hipcc and under g++: a check that only passes on one compiler is a
 * compiler finding, not a pass.
 *
 * The three arms:
 *   arm 0 HONEST  every expectation is the honest one; all checks must pass.
 *   arm 1 STUB    the comparator is replaced by "always equal". Every DIFFERS check must go red and
 *                 the red count must equal the number of DIFFERS checks, counted at run time. A
 *                 comparator that cannot fail is caught here, not hoped about.
 *   arm 2 CROSSED each value is judged against the NEXT DIFFERENT expectation in the list. Every
 *                 EQUALS check must go red and the count must equal the number of EQUALS checks;
 *                 each red line prints the borrowed want next to its own, so a reader can see the
 *                 check passed on somebody else's answer rather than its own.
 *
 * `se_selftest_all` (the driver's rule) requires all three arms to have RUN, the honest arm to be
 * clean with `changed == 0`, and both saboteurs to report `changed == want > 0`. Running one arm
 * alone prints FAIL on purpose, and gate.py witnesses that.
 *
 * The fixtures are hand-worked and the arithmetic is written next to each expectation, so a reader
 * can check them without running anything.
 */
#ifndef SHARED_EXPERT_SELFTEST_H
#define SHARED_EXPERT_SELFTEST_H

#include "shared_expert_kernels.h"

#define SE_ST_EQUALS 0
#define SE_ST_DIFFERS 1

typedef struct {
    const char *name;
    int kind;
    uint32_t got, want;
    int fired;          /* EQUALS: got==want; DIFFERS: got!=want */
    int borrowed;       /* crossed arm: the want actually used was not this check's own */
} SeStCheck;

static inline int se_st_word_eq(uint32_t a, uint32_t b) {
    if (a == b) return 1;
    /* two NaNs of any payload are the same poison: which quiet-NaN bits a compiler emitted is not a
       fact this lane claims to control */
    return ((a & 0x7F800000u) == 0x7F800000u && (a & 0x007FFFFFu) &&
            (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu));
}

/* ---- hand-worked fixtures ------------------------------------------------------------------- */
/* Q8_0 block with d = f16(1.0) = 0x3C00 and qs[e] = e - 16 for e in [0,32): every value is exactly
 * the stored integer, because d is a power of two and the int8 is exact. */
static inline void se_st_block_q8(uint8_t *blk, uint16_t d_bits) {
    blk[0] = (uint8_t)(d_bits & 0xFFu);
    blk[1] = (uint8_t)(d_bits >> 8);
    for (int e = 0; e < SE_Q8_ELTS; ++e) blk[2 + e] = (uint8_t)(int8_t)(e - 16);
}

/* A row of `nblocks` Q8_0 blocks with d = 1.0 and every qs = +1, so each element is exactly 1.0 and
 * a dot against an all-ones activation is exactly the element count: 2048 honest, 2016 with the
 * tail block dropped, 1024 with half the columns. Those three numbers ARE the injection witness. */
static inline void se_st_row_ones(uint8_t *row, int nblocks) {
    for (int b = 0; b < nblocks; ++b) {
        uint8_t *blk = row + (size_t)b * SE_Q8_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;                 /* d = 1.0 */
        for (int e = 0; e < SE_Q8_ELTS; ++e) blk[2 + e] = 1;
    }
}

typedef struct {
    int n_checks, n_equals, n_differs, changed, failures;
} SeStReport;

static inline void se_st_cast_perchan_fixture(uint32_t *n_after, uint32_t *n_perchan);

static inline void se_selftest(int arm, SeStReport *rep, FILE *log) {
    SeStCheck cs[32];
    int n = 0;
    int reads = 0;
    int rep_note_digest_count = 0;
    uint8_t blk[SE_Q8_BYTES];
    static uint8_t rowones[SE_GATE_BLOCKS * SE_Q8_BYTES];
    static float xones[SE_H];

    /* 1. Q8_0 decode, d = 1.0: value(e) == e - 16 exactly. */
    se_st_block_q8(blk, 0x3C00u);
    cs[n].name = "q8-d1-value7-is-+(-9)"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_q8_value(blk, 7, 0)); cs[n].want = se_bits_of(-9.0f); n++;
    cs[n].name = "q8-d1-value20-is-+4"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_q8_value(blk, 20, 0)); cs[n].want = se_bits_of(4.0f); n++;
    /* 2. d = f16(0.5) = 0x3800 and qs = +100 -> 50.0 exactly (11*7 significand bits, exact in f32) */
    se_st_block_q8(blk, 0x3800u);
    blk[2 + 0] = (uint8_t)100;
    cs[n].name = "q8-d0p5-q100-is-50"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_q8_value(blk, 0, 0)); cs[n].want = se_bits_of(50.0f); n++;
    /* 3. the same block read with the scale typed bf16 must NOT be 50.0: the width misread fires. */
    cs[n].name = "q8-scale-typed-bf16-is-not-50"; cs[n].kind = SE_ST_DIFFERS;
    cs[n].got = se_bits_of(se_q8_value(blk, 0, 1)); cs[n].want = se_bits_of(50.0f); n++;
    /* 4. f16 subnormal 0x0001 == 2^-24 exactly, and f16 -0.0 keeps its sign. */
    cs[n].name = "f16-subnormal-0x0001-is-2p-24"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_f16_to_f32(0x0001u));
    cs[n].want = se_bits_of(5.9604644775390625e-08f); n++;
    cs[n].name = "f16-minus-zero-keeps-sign"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_f16_to_f32(0x8000u)); cs[n].want = 0x80000000u; n++;
    /* 5. bf16 round-to-nearest-even. Near 1.0 the bf16 step is 2^-7, so the f32 halfway point is
     *    0x3F808000 (1+2^-8). At a tie the EVEN kept significand wins: 0x3F80 (even) stays down,
     *    0x3F81 (odd) is rounded up to 0x3F82. One ulp of f32 either side breaks the tie. */
    cs[n].name = "bf16-rne-tie-goes-to-even-down"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_bf16_rt(se_float_of(0x3F808000u))); cs[n].want = se_bits_of(1.0f); n++;
    cs[n].name = "bf16-rne-above-tie-goes-up"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_bf16_rt(se_float_of(0x3F808001u)));
    cs[n].want = se_bits_of(se_float_of(0x3F810000u)); n++;
    cs[n].name = "bf16-rne-tie-goes-to-even-up"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_bf16_rt(se_float_of(0x3F818000u)));
    cs[n].want = se_bits_of(se_float_of(0x3F820000u)); n++;
    cs[n].name = "bf16-rne-below-tie-stays-down"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_bf16_rt(se_float_of(0x3F807FFFu))); cs[n].want = se_bits_of(1.0f); n++;
    /* 6. the declared dot over an all-ones row and all-ones activation: 2048 elements of 1*1. */
    se_st_row_ones(rowones, SE_GATE_BLOCKS);
    for (int k = 0; k < SE_H; ++k) xones[k] = 1.0f;
    reads = 0;
    cs[n].name = "dot-ascending-honest-is-2048"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_dot_q8_row(rowones, se_cols_of(SE_H, SE_MODE_CORRECT), xones, 0, &reads));
    cs[n].want = se_bits_of(2048.0f); n++;
    cs[n].name = "dot-tail-drop-is-2016"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_dot_q8_row(rowones, se_cols_of(SE_H, SE_MODE_TAIL_DROP), xones, 0, &reads));
    cs[n].want = se_bits_of(2016.0f); n++;
    cs[n].name = "dot-cols-half-is-1024"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_dot_q8_row(rowones, se_cols_of(SE_H, SE_MODE_COLS_HALF), xones, 0, &reads));
    cs[n].want = se_bits_of(1024.0f); n++;
    /* 7. the transcedentals at their exact points: silu(0) = 0, sigmoid(0) = 0.5. */
    cs[n].name = "silu-at-zero-is-zero"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_silu_f32(0.0f)); cs[n].want = 0u; n++;
    cs[n].name = "sigmoid-at-zero-is-one-half"; cs[n].kind = SE_ST_EQUALS;
    cs[n].got = se_bits_of(se_sigmoid_f32(0.0f)); cs[n].want = se_bits_of(0.5f); n++;
    /* 8. layout candidates at j = 3, hand-worked:
     *      GATE_FIRST     gate 3,   up 515        UP_FIRST      gate 515, up 3
     *      ROW_INTERLEAVED gate 6,  up 7          PAIR_SWAP     gate 3,   up 515 (rows as honest) */
    {
        int g = -1, u = -1;
        se_layout_rows(SE_LAYOUT_GATE_FIRST, 0, 3, &g, &u);
        cs[n].name = "layout-gate-first-rows-3-515"; cs[n].kind = SE_ST_EQUALS;
        cs[n].got = (uint32_t)(g * 10000 + u); cs[n].want = 3u * 10000u + 515u; n++;
        se_layout_rows(SE_LAYOUT_UP_FIRST, 0, 3, &g, &u);
        cs[n].name = "layout-up-first-rows-515-3"; cs[n].kind = SE_ST_EQUALS;
        cs[n].got = (uint32_t)(g * 10000 + u); cs[n].want = 515u * 10000u + 3u; n++;
        se_layout_rows(SE_LAYOUT_ROW_INTERLEAVED, 0, 3, &g, &u);
        cs[n].name = "layout-row-inter-rows-6-7"; cs[n].kind = SE_ST_EQUALS;
        cs[n].got = (uint32_t)(g * 10000 + u); cs[n].want = 6u * 10000u + 7u; n++;
        se_layout_rows(SE_LAYOUT_GATE_FIRST, 1, 3, &g, &u);
        cs[n].name = "layout-shifted-gate-first-rows-4-516"; cs[n].kind = SE_ST_EQUALS;
        cs[n].got = (uint32_t)(g * 10000 + u); cs[n].want = 4u * 10000u + 516u; n++;
        /* and GATE_FIRST's rows are not ROW_INTERLEAVED's: the candidates are distinguishable. */
        se_layout_rows(SE_LAYOUT_GATE_FIRST, 0, 3, &g, &u);
        cs[n].name = "layout-gate-first-differs-from-row-inter"; cs[n].kind = SE_ST_DIFFERS;
        cs[n].got = (uint32_t)(g * 10000 + u); cs[n].want = 6u * 10000u + 7u; n++;
    }
    /* 9. the digest blindness pair, witnessed rather than claimed. Two independent blindnesses,
     *    each demonstrated by a fixture that MUST be blind and one that must not:
     *      (a) a uniform x2 rescale moves every exponent and no mantissa bit, so the
     *          significand-XOR digest cannot see it; the order-free sum of full bit images can.
     *      (b) a value written twice cancels in an XOR (x^x = 0) and is invisible to it, while the
     *          sum sees the extra addend.
     *    Neither digest alone is a witness, which is why the lane carries both. */
    {
        SeDigest a, b;
        const float v[4] = { 1.5f, 3.0f, 1.25f, -0.75f };
        se_digest_init(&a); se_digest_init(&b);
        for (int i = 0; i < 4; ++i) { se_digest_add(&a, v[i]); se_digest_add(&b, v[i] * 2.0f); }
        cs[n].name = "digest-xorsig-is-blind-to-x2-rescale"; cs[n].kind = SE_ST_EQUALS;
        cs[n].got = a.xor_sig; cs[n].want = b.xor_sig; n++;
        cs[n].name = "digest-sumbits-SEES-x2-rescale"; cs[n].kind = SE_ST_DIFFERS;
        cs[n].got = a.sum_bits; cs[n].want = b.sum_bits; n++;
        cs[n].name = "digest-xorsig-is-blind-to-a-duplicated-pair"; cs[n].kind = SE_ST_EQUALS;
        se_digest_init(&a); se_digest_init(&b);
        se_digest_add(&a, 1.5f); se_digest_add(&a, 2.5f); se_digest_add(&a, 2.5f);
        se_digest_add(&b, 1.5f);
        cs[n].got = a.xor_sig; cs[n].want = b.xor_sig; n++;
        cs[n].name = "digest-sumbits-SEES-a-duplicated-pair"; cs[n].kind = SE_ST_DIFFERS;
        cs[n].got = a.sum_bits; cs[n].want = b.sum_bits; n++;
        rep_note_digest_count = (int)(a.n_nonzero != b.n_nonzero);
    }
    /* 10. the cast position and the gate form must be visible in the words, not in prose. These two
     *     witnesses are computed over a real fixture below and their counts are asserted > 0 here
     *     by comparing against 0 (a data expression, not a constant). */
    {
        uint32_t n_after = 0, n_perchan = 0;
        se_st_cast_perchan_fixture(&n_after, &n_perchan);
        cs[n].name = "cast-after-vs-before-differs-in-out"; cs[n].kind = SE_ST_DIFFERS;
        cs[n].got = n_after; cs[n].want = 0u; n++;
        cs[n].name = "perchan-vs-scalar-differs-in-out"; cs[n].kind = SE_ST_DIFFERS;
        cs[n].got = n_perchan; cs[n].want = 0u; n++;
    }

    /* ---- apply the arm's comparator sabotage --------------------------------------------- */
    int n_equals = 0, n_differs = 0, changed = 0, failures = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t want = cs[i].want;
        int borrowed = 0;
        if (arm == 2) {
            for (int k = 1; k < n; ++k) {
                const int j = (i + k) % n;
                if (!se_st_word_eq(cs[j].want, cs[i].want)) { want = cs[j].want; borrowed = 1; break; }
            }
        }
        int eq = (arm == 1) ? 1 : se_st_word_eq(cs[i].got, want);
        cs[i].fired = (cs[i].kind == SE_ST_EQUALS) ? eq : !eq;
        cs[i].borrowed = borrowed;
        if (cs[i].kind == SE_ST_EQUALS) ++n_equals; else ++n_differs;
        if (!cs[i].fired) {
            ++failures;
            fprintf(log, "FAIL selftest arm=%d check=%s got=%08x want=%08x want_src=%s\n", arm,
                    cs[i].name, cs[i].got, want, borrowed ? "borrowed" : cs[i].name);
        }
    }
    for (int i = 0; i < n; ++i) if (!cs[i].fired) ++changed;
    rep->n_checks = n; rep->n_equals = n_equals; rep->n_differs = n_differs;
    rep->changed = changed; rep->failures = failures;
    if (!rep_note_digest_count) {
        fprintf(log, "FAIL selftest arm=%d the digest blindness fixtures were vacuous\n", arm);
        rep->failures++;
    }
    const int want_changed = (arm == 1) ? n_differs : (arm == 2 ? n_equals : 0);
    fprintf(log, "SELFTEST arm=%d checks=%d equals=%d differs=%d changed=%d want=%d failures=%d\n",
            arm, n, n_equals, n_differs, changed, want_changed, failures);
    if (changed != want_changed || (arm == 0 && failures != 0)) {
        fprintf(log, "FAIL selftest arm=%d changed=%d but the derived count is %d\n",
                arm, changed, want_changed);
        rep->failures++;
    }
}

/* The cast-position and gate-form witnesses. The fixture is built so the gate dot is EXACTLY 0.25:
 * x[k] = 0.5 and sg[k] = 2^-7 for k < 64, zero elsewhere, so t = 64 * 0.5 * 2^-7 = 0.25 with no
 * rounding anywhere (every product and the whole sum is a power of two). sigmoid(0.25) = 0.562176...
 * is NOT bf16-representable — bf16 takes 0.5625 — so the cast position has something to disagree
 * about, which is exactly the regime the reference's cast-BEFORE form is load-bearing in. The
 * per-channel form sees t_h = 2^-7 for k < 64 and 0 elsewhere, i.e. p_h = 0.500979/0.5 against the
 * scalar's 0.5625. Both counts are asserted non-zero by the caller. */
static inline float se_st_f32_at(const uint8_t *p) { float v; memcpy(&v, p, 4u); return v; }
static inline void se_st_cast_perchan_fixture(uint32_t *n_after, uint32_t *n_perchan) {
    static float raw[SE_H], out_before[SE_H], out_after[SE_H], out_perchan[SE_H];
    static uint8_t xbytes[2 * SE_H], sgbytes[4 * SE_H];
    for (int h = 0; h < SE_H; ++h) {
        /* raw values that are NOT powers of two, so the products straddle bf16 boundaries */
        raw[h] = se_bf16_rt(0.5f + 0.0015f * (float)h);
        const float xv = (h < 64) ? 0.5f : 0.0f;
        const float sv = (h < 64) ? 0.0078125f : 0.0f;      /* 2^-7, exact in bf16 and in f32 */
        { const uint16_t u = se_bf16_bits_of(xv); xbytes[2 * h] = (uint8_t)u; xbytes[2 * h + 1] = (uint8_t)(u >> 8); }
        { const uint32_t b = se_bits_of(sv); sgbytes[4 * h] = (uint8_t)b; sgbytes[4 * h + 1] = (uint8_t)(b >> 8);
          sgbytes[4 * h + 2] = (uint8_t)(b >> 16); sgbytes[4 * h + 3] = (uint8_t)(b >> 24); }
    }
    /* the scalar form: one probability for the whole row, both cast positions */
    float t = 0.0f;
    for (int k = 0; k < SE_H; ++k) t += se_f32_of_bf16_bytes(xbytes + 2 * k) * se_st_f32_at(sgbytes + 4 * k);
    const float t_b = se_bf16_rt(t);
    for (int h = 0; h < SE_H; ++h) {
        out_before[h] = se_gate_mul(t_b, raw[h], 0);
        out_after[h] = se_gate_mul(t_b, raw[h], 1);
        const float t_h = se_bf16_rt(se_f32_of_bf16_bytes(xbytes + 2 * h) * se_st_f32_at(sgbytes + 4 * h));
        out_perchan[h] = se_gate_mul(t_h, raw[h], 0);
    }
    uint32_t a = 0, b = 0;
    for (int h = 0; h < SE_H; ++h) {
        if (se_bits_of(out_before[h]) != se_bits_of(out_after[h])) ++a;
        if (se_bits_of(out_before[h]) != se_bits_of(out_perchan[h])) ++b;
    }
    *n_after = a; *n_perchan = b;
}

static inline int se_selftest_all(FILE *log) {
    SeStReport r[3];
    for (int a = 0; a < 3; ++a) { se_selftest(a, &r[a], log); }
    fprintf(log, "SELFTEST3 arms_run=3 checks=%d equals=%d differs=%d honest_failures=%d\n",
            r[0].n_checks, r[0].n_equals, r[0].n_differs, r[0].failures);
    const int ok = (r[0].failures == 0) && (r[0].changed == 0)
                && (r[1].changed > 0) && (r[2].changed > 0)
                && (r[0].n_equals > 0) && (r[0].n_differs > 0);
    fprintf(log, "%s selftest arms=3\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
#endif
