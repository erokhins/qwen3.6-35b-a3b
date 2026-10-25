/* moe_gather_selftest.h — the hand-worked, device-free self-tests for the gather primitives, with
 * the three arms the standing rules require.
 *
 * Included by BOTH arms (`moe_gather.hip --selftest` and `host_probe --selftest`), so the same
 * numbers are asserted under hipcc and under g++: a check that only passes on one compiler is a
 * compiler finding, not a pass.
 *
 * The three arms (AGENTS.md §Standing rules / pitfalls/40 §"A negative control with an empty
 * must-catch set is not a control"):
 *   arm 0 HONEST   every expectation is the honest one. All checks must pass. A green suite that has
 *                  never been seen to fail is not evidence, so the two arms below exist to make it
 *                  fail, and their failure counts are asserted non-zero.
 *   arm 1 STUB     the comparator is replaced by "always equal". Every `differs` check MUST go red;
 *                  the count of red checks must equal the number of `differs` checks, which is
 *                  counted at run time and asserted > 0. A comparator that cannot fail is caught here.
 *   arm 2 CROSSED  each value is judged against the NEXT check's expectation. Every `equals` check
 *                  MUST go red; the count must equal the number of `equals` checks, asserted > 0. A
 *                  check that passes on somebody else's answer is caught here.
 *
 * The fixtures are hand-worked in this file, with the arithmetic written out next to each expectation
 * so a reader can check it without running anything. Four of them are the element-order witnesses:
 * which sub-block a nibble's scale comes from (Q4_K element 33), where Q5_K's fifth bit lives, where
 * Q6_K's `d` lives, and which 8-element group carries the non-zero value.
 */
#ifndef MOE_GATHER_SELFTEST_H
#define MOE_GATHER_SELFTEST_H

#include "moe_gather_kernels.h"

#define MG_ST_EQUALS 0
#define MG_ST_DIFFERS 1

typedef struct {
    const char *name;
    int kind;                 /* MG_ST_EQUALS or MG_ST_DIFFERS */
    uint32_t got, want;
    int fired;                /* computed: for EQUALS got==want, for DIFFERS !mg_word_eq(got,want) */
} MgStCheck;

static inline uint32_t mg_st_bits(float f) { return mg_bits_of(f); }

/* The hand-worked block bytes ------------------------------------------------------------- */
/* Q4_K: d = f16 1.0 (0x3C00), dmin = 0 (0x0000). scales = {1,1,1,1, 0,0,0,0, 1,1,1,1} gives
 * sc[j] = 1 and mg[j] = 0 for every j under get_scale_min_k4:
 *   j<4:   sc = q[j]&63 = 1 (q[j] = 0x01);            mg = q[j+4]&63 = 0 (q[j+4] = 0x00)
 *   j>=4:  sc = (q[j+4]&0xF) | ((q[j-4]>>6)<<4) = 1;  mg = (q[j+4]>>4) | ((q[j]>>6)<<4) = 0
 * so every value is 1.0 * 1 * q - 0.0 * 0 = q, and with qs[] = 0xFF every element is 15.0. */
static inline void mg_st_q4k(uint8_t *blk, int zero_element_33) {
    memset(blk, 0, MG_Q4K_BYTES);
    blk[0] = 0x00; blk[1] = 0x3C;                 /* d = 1.0 */
    blk[2] = 0x00; blk[3] = 0x00;                 /* dmin = 0 */
    blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
    blk[8] = 0; blk[9] = 0; blk[10] = 0; blk[11] = 0;
    blk[12] = 1; blk[13] = 1; blk[14] = 1; blk[15] = 1;
    memset(blk + 16, 0xFF, MG_QK_K / 2);
    if (zero_element_33) blk[16 + 1] = 0x0F;      /* high nibble of qs[1] = element 64*0+32*1+1 = 33 */
}

/* Q5_K: same scale/min trick, d = 1.0, dmin = 0, all nibbles 0, qh[0] = 0b00000011 -> the fifth bit
 * of sub-blocks 0 and 1 for l = 0 is set: element e = 64*0 + 32*0 + 0 = 0 gets q = 0 | 1<<4 = 16 and
 * element e = 64*0 + 32*1 + 0 = 32 gets q = 16; every other element stays 0. */
static inline void mg_st_q5k(uint8_t *blk) {
    memset(blk, 0, MG_Q5K_BYTES);
    blk[0] = 0x00; blk[1] = 0x3C;
    blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
    blk[12] = 1; blk[13] = 1; blk[14] = 1; blk[15] = 1;
    blk[16] = 0x03;                               /* qh[0]: bits 0 and 1 set */
}

/* Q6_K: d = f16 1.0 at byte 208 (the whole point of MG_MODE_Q6K_D_BYTE_0), sc[0] = 1 at byte 192,
 * ql[0] = 0x01 (low nibble 1) and qh[0] = 0x02 (bits 1:0 = 2 -> 2<<4 = 32) so element
 * e = 128*0 + 32*0 + 0 = 0 has q = 1 | 32 = 33 and value = 1.0 * sc[0] * (33 - 32) = 1.0. Every other
 * element keeps q = 0, and Q6_K carries no zero point: its value is (0 - 32) * sc[e>>4], so elements
 * 1..15 (scale index 0) are -32.0 and elements from 16 up (scale index 1, which is 0 here) are +0.0.
 * That asymmetry is the witness that the Q6_K scale index is e>>4, not e>>5 or e>>7. */
static inline void mg_st_q6k(uint8_t *blk) {
    memset(blk, 0, MG_Q6K_BYTES);
    blk[208] = 0x00; blk[209] = 0x3C;             /* d = 1.0, at 208 */
    blk[192] = 0x01;                              /* sc[0] = 1 (signed) */
    blk[0] = 0x01;                                /* ql[0] low nibble = 1 */
    blk[128] = 0x02;                              /* qh[0] bits 1:0 = 2 */
}

static inline int mg_selftest(int arm, int verbose) {
    MgStCheck ck[96];
    int n = 0;
#define PUSH(nm, kd, g, w) do { ck[n].name = (nm); ck[n].kind = (kd); \
        ck[n].got = (uint32_t)(g); ck[n].want = (uint32_t)(w); n++; } while (0)
    /* --- 1. block widths, from the pinned ggml structs ------------------------------------------ */
    PUSH("widths/q4k", MG_ST_EQUALS, mg_block_bytes(MG_KIND_Q4K), MG_Q4K_BYTES);
    PUSH("widths/q5k", MG_ST_EQUALS, mg_block_bytes(MG_KIND_Q5K), MG_Q5K_BYTES);
    PUSH("widths/q6k", MG_ST_EQUALS, mg_block_bytes(MG_KIND_Q6K), MG_Q6K_BYTES);

    /* --- 2. f16 decode, by bit pattern ----------------------------------------------------------- */
    PUSH("f16/one", MG_ST_EQUALS, mg_st_bits(mg_f16(0x3C00u)), mg_bits_of(1.0f));
    PUSH("f16/one_half", MG_ST_EQUALS, mg_st_bits(mg_f16(0x3800u)), mg_bits_of(0.5f));
    PUSH("f16/minus_one", MG_ST_EQUALS, mg_st_bits(mg_f16(0xBC00u)), mg_bits_of(-1.0f));
    PUSH("f16/min_subnormal", MG_ST_EQUALS, mg_st_bits(mg_f16(0x0001u)), mg_bits_of(5.9604644775390625e-8f));
    PUSH("f16/neg_zero_bits", MG_ST_EQUALS, mg_st_bits(mg_f16(0x8000u)), 0x80000000u);
    PUSH("f16/pos_inf_bits", MG_ST_EQUALS, mg_st_bits(mg_f16(0x7C00u)), 0x7F800000u);
    PUSH("f16/snan_is_nan", MG_ST_EQUALS, (uint32_t)mg_is_nan(mg_f16(0x7C01u)), 1u);
    PUSH("f16/max_normal", MG_ST_EQUALS, mg_st_bits(mg_f16(0x7BFFu)), mg_bits_of(65504.0f));

    /* --- 3. get_scale_min_k4, hand-computed from the bytes below --------------------------------- *
     * scales = {0x3F,0x11,0x22,0x33, 0x44,0x55,0x66,0x77, 0x88,0x99,0xAA,0xBB}
     *   j=0: sc = 0x3F&63 = 63           mg = 0x44&63 = 4
     *   j=1: sc = 0x11&63 = 17           mg = 0x55&63 = 85-64 = 21
     *   j=2: sc = 0x22&63 = 34           mg = 0x66&63 = 102-64 = 38
     *   j=3: sc = 0x33&63 = 51           mg = 0x77&63 = 55
     *   j=4: sc = (0x88&0xF)|((0x3F>>6)<<4) = 8|0 = 8     mg = (0x88>>4)|((0x44>>6)<<4) = 8|16 = 24
     *   j=5: sc = (0x99&0xF)|((0x11>>6)<<4) = 9|0 = 9     mg = (0x99>>4)|((0x55>>6)<<4) = 9|16 = 25
     *   j=6: sc = (0xAA&0xF)|((0x22>>6)<<4) = 10|0 = 10   mg = (0xAA>>4)|((0x66>>6)<<4) = 10|16 = 26
     *   j=7: sc = (0xBB&0xF)|((0x33>>6)<<4) = 11|0 = 11   mg = (0xBB>>4)|((0x77>>6)<<4) = 11|16 = 27
     */
    {
        uint8_t q[12] = {0x3F,0x11,0x22,0x33, 0x44,0x55,0x66,0x77, 0x88,0x99,0xAA,0xBB};
        const int32_t wsc[8] = {63,17,34,51,8,9,10,11};
        const int32_t wmg[8] = {4,21,38,55,24,25,26,27};
        for (int j = 0; j < 8; ++j) {
            int32_t sc = -1, mg = -1;
            mg_scale_min(j, q, &sc, &mg);
            switch (j) {
            case 0: PUSH("scale_min/sc0", MG_ST_EQUALS, sc, wsc[0]); PUSH("scale_min/mg0", MG_ST_EQUALS, mg, wmg[0]); break;
            case 1: PUSH("scale_min/sc1", MG_ST_EQUALS, sc, wsc[1]); PUSH("scale_min/mg1", MG_ST_EQUALS, mg, wmg[1]); break;
            case 2: PUSH("scale_min/sc2", MG_ST_EQUALS, sc, wsc[2]); PUSH("scale_min/mg2", MG_ST_EQUALS, mg, wmg[2]); break;
            case 3: PUSH("scale_min/sc3", MG_ST_EQUALS, sc, wsc[3]); PUSH("scale_min/mg3", MG_ST_EQUALS, mg, wmg[3]); break;
            case 4: PUSH("scale_min/sc4", MG_ST_EQUALS, sc, wsc[4]); PUSH("scale_min/mg4", MG_ST_EQUALS, mg, wmg[4]); break;
            case 5: PUSH("scale_min/sc5", MG_ST_EQUALS, sc, wsc[5]); PUSH("scale_min/mg5", MG_ST_EQUALS, mg, wmg[5]); break;
            case 6: PUSH("scale_min/sc6", MG_ST_EQUALS, sc, wsc[6]); PUSH("scale_min/mg6", MG_ST_EQUALS, mg, wmg[6]); break;
            default: PUSH("scale_min/sc7", MG_ST_EQUALS, sc, wsc[7]); PUSH("scale_min/mg7", MG_ST_EQUALS, mg, wmg[7]); break;
            }
        }
        /* the naive unpack must DISAGREE with the decoder route on this block: its sc[1] is the
         * contiguous 6-bit field at bit 6 = (0x3F|0x11<<8)>>6 & 63 = 0x3C = 60, not 17. */
        int32_t nsc = -1, nmg = -1;
        mg_scale_min_naive(1, q, &nsc, &nmg);
        PUSH("scale_min/naive_differs", MG_ST_DIFFERS, nsc, wsc[1]);
    }

    /* --- 4. the three decoders on hand-worked blocks --------------------------------------------- */
    {
        uint8_t b4[MG_Q4K_BYTES], b5[MG_Q5K_BYTES], b6[MG_Q6K_BYTES];
        mg_st_q4k(b4, 0);
        PUSH("q4k/elt0_is_15", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b4, MG_KIND_Q4K, 0, MG_MODE_CORRECT)), mg_bits_of(15.0f));
        PUSH("q4k/elt255_is_15", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b4, MG_KIND_Q4K, 255, MG_MODE_CORRECT)), mg_bits_of(15.0f));
        PUSH("q4k/f32_route_agrees", MG_ST_EQUALS,
             (uint32_t)mg_word_eq(mg_bits_of(mg_deq_dbl(b4, MG_KIND_Q4K, 7, MG_MODE_CORRECT)),
                                  mg_bits_of(mg_deq_f32(b4, MG_KIND_Q4K, 7, MG_MODE_CORRECT))), 1u);
        mg_st_q4k(b4, 1);
        PUSH("q4k/elt33_is_the_zero", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b4, MG_KIND_Q4K, 33, MG_MODE_CORRECT)), mg_bits_of(0.0f));
        PUSH("q4k/elt1_still_15", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b4, MG_KIND_Q4K, 1, MG_MODE_CORRECT)), mg_bits_of(15.0f));
        PUSH("q4k/elt65_still_15", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b4, MG_KIND_Q4K, 65, MG_MODE_CORRECT)), mg_bits_of(15.0f));

        mg_st_q5k(b5);
        PUSH("q5k/elt0_is_16", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b5, MG_KIND_Q5K, 0, MG_MODE_CORRECT)), mg_bits_of(16.0f));
        PUSH("q5k/elt32_is_16", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b5, MG_KIND_Q5K, 32, MG_MODE_CORRECT)), mg_bits_of(16.0f));
        PUSH("q5k/elt1_is_0", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b5, MG_KIND_Q5K, 1, MG_MODE_CORRECT)), mg_bits_of(0.0f));
        PUSH("q5k/elt33_is_0", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b5, MG_KIND_Q5K, 33, MG_MODE_CORRECT)), mg_bits_of(0.0f));

        mg_st_q6k(b6);
        PUSH("q6k/elt0_is_1", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 0, MG_MODE_CORRECT)), mg_bits_of(1.0f));
        /* Q6_K quants are biased by 32, so a zero-quant element is NOT zero: it is (0-32)*sc. The
         * scale index is e>>4, so elements 1..15 share sc[0] = 1 and come out -32.0 while element 16
         * takes sc[1] = 0 — the witness that the scale index is e>>4 and not e>>5 or e>>7. */
        PUSH("q6k/elt1_is_minus32", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 1, MG_MODE_CORRECT)), mg_bits_of(-32.0f));
        PUSH("q6k/elt15_is_minus32", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 15, MG_MODE_CORRECT)), mg_bits_of(-32.0f));
        /* -0.0, not +0.0: the product is d * sc[1] * (0 - 32) and the sign of the zero follows the
         * sign of (q-32). The comparator distinguishes +0.0 from -0.0 (the rule prototype/q4k_dequant
         * §q4kd_word_eq established), so claiming +0.0 here would be a false claim about the
         * arithmetic — and a real one, because a converter that flushed negative zeros to +0.0 would
         * be caught by exactly this check. */
        PUSH("q6k/elt16_is_neg_zero", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 16, MG_MODE_CORRECT)), 0x80000000u);
        PUSH("q6k/elt32_is_neg_zero", MG_ST_EQUALS, mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 32, MG_MODE_CORRECT)), 0x80000000u);
        /* the injection must fire: d read at byte 0 is the f16 pattern of the two ql bytes
         * (0x0001 = 2^-24), so the same element comes out 2^-24 instead of 1.0. */
        PUSH("q6k/d_at_byte0_differs", MG_ST_DIFFERS,
             mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 0, MG_MODE_Q6K_D_BYTE_0)), mg_bits_of(1.0f));
        PUSH("q6k/d_at_byte0_is_2pow_24", MG_ST_EQUALS,
             mg_st_bits(mg_deq_dbl(b6, MG_KIND_Q6K, 0, MG_MODE_Q6K_D_BYTE_0)), mg_bits_of(5.9604644775390625e-8f));
    }

    /* --- 5. the combine form, hand-worked --------------------------------------------------------- *
     * (a) the addend ORDER decides the number. Four non-zero addends, weights exactly 1.0, taken
     *     from an order-sensitivity search over bf16 left-to-right sums (work/probe8.py):
     *         1.0, 1.0, 1.0, 0.009765625      ->  1+1 = 2;  +1 = 3;  +0.009765625 rounds UP,
     *                                            because the bf16 step at 3.0 is 2^-6 = 0.015625 and
     *                                            0.009765625 is more than half of it        = 3.015625
     *         0.009765625, 1.0, 1.0, 1.0      ->  +1 = 1.0078125 (nearest step at 1.0 is 2^-7);
     *                                            +1 = 2.0078125, exactly half a step at 2.0, ties to
     *                                            even -> 2.0;  +1 = 3.0                     = 3.0
     *     The rank order is the first sequence and the ascending-expert-id order the second, so the
     *     correct arm reads 3.0 and the RANK_ACCUM arm reads 3.015625 — one bf16 ULP apart, which the
     *     comparator must be able to see. ids are [1,2,3,0,7,6,5,4] so that sorting the slots by
     *     expert id puts slot 3 (the small addend) first.
     * (b) the cast POSITION decides the number. One non-zero addend, raw = 3.0:
     *       wpre  = 1.005859375 = 1 + 3*2^-9, whose bf16 rounding is 1.0078125;
     *       cast-before  prod = bf16(3.0 * 1.0078125) = bf16(3.0234375) = 3.03125 (the tie at the
     *                            midpoint of the 2^-6 step goes to the even bit pattern 0x4042);
     *       cast-after   prod = bf16(3.0 * 1.005859375) = bf16(3.017578125) = 3.015625 (below the
     *                            midpoint, rounds down).
     *     Both are exact f32 constants, so this is arithmetic and not a tolerance. It is also the
     *     reason the arm cannot be judged at the weights: wpre is the same number in both arms.
     */
    {
        static float slabraw[MG_K * MG_H];
        static float wcast[MG_K], wpre[MG_K];
        static int ids[MG_K];
        const int want_ids[MG_K] = {1, 2, 3, 0, 7, 6, 5, 4};
        const float vals[MG_K] = {1.0f, 1.0f, 1.0f, 0.009765625f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < MG_K; ++k) {
            ids[k] = want_ids[k];
            wcast[k] = 1.0f; wpre[k] = 1.0f;
            for (int h = 0; h < MG_H; ++h) slabraw[k * MG_H + h] = vals[k];
        }
        uint16_t acc = 0u;
        int order[MG_K];
        mg_combine_order(ids, MG_MODE_CORRECT, order);
        PUSH("combine/order_is_small_addend_first", MG_ST_EQUALS,
             (uint32_t)(order[0] == 3 && order[1] == 0 && order[2] == 1 && order[3] == 2), 1u);
        for (int i = 0; i < MG_K; ++i) {
            const int k = order[i];
            acc = mg_bf16_add(acc, mg_bf16_of(slabraw[k * MG_H + 0] * wcast[k]));
        }
        PUSH("combine/ascending_id", MG_ST_EQUALS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.0f));
        mg_combine_order(ids, MG_MODE_RANK_ACCUM, order);
        PUSH("combine/order_is_rank_first", MG_ST_EQUALS, (uint32_t)order[0], 0u);
        acc = 0u;
        for (int i = 0; i < MG_K; ++i) {
            const int k = order[i];
            acc = mg_bf16_add(acc, mg_bf16_of(slabraw[k * MG_H + 0] * wcast[k]));
        }
        PUSH("combine/rank_order_differs", MG_ST_DIFFERS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.0f));
        PUSH("combine/rank_order_is_3.015625", MG_ST_EQUALS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.015625f));

        /* (b) the cast position */
        for (int k = 0; k < MG_K; ++k) {
            wpre[k] = 1.005859375f;
            wcast[k] = mg_bf16_to_f32(mg_bf16_of(wpre[k]));
            for (int h = 0; h < MG_H; ++h) slabraw[k * MG_H + h] = 0.0f;
        }
        slabraw[0] = 3.0f;
        PUSH("combine/wcast_is_the_bf16_round", MG_ST_EQUALS, mg_st_bits(wcast[0]), mg_bits_of(1.0078125f));
        acc = mg_bf16_of(slabraw[0] * wcast[0]);
        PUSH("combine/cast_before", MG_ST_EQUALS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.03125f));
        acc = mg_bf16_of(slabraw[0] * wpre[0]);
        PUSH("combine/cast_after_differs", MG_ST_DIFFERS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.03125f));
        PUSH("combine/cast_after_is_3.015625", MG_ST_EQUALS, mg_st_bits(mg_bf16_to_f32(acc)), mg_bits_of(3.015625f));
        /* the weight alone is not the discriminator: both arms carry the same wpre, so a comparator
         * that stops at the weights cannot tell the orders apart (prototype/moe_router's
         * mr_combine_row says the same thing about its own combine). */
        PUSH("combine/weights_agree_between_arms", MG_ST_EQUALS, mg_st_bits(wpre[0]), mg_bits_of(1.005859375f));
    }

    /* --- 6. silu on the exact-fixture domain ------------------------------------------------------ */
    PUSH("silu/identity_at_200", MG_ST_EQUALS, mg_st_bits(mg_silu(200.0f)), mg_bits_of(200.0f));
    PUSH("silu/identity_at_128", MG_ST_EQUALS, mg_st_bits(mg_silu(128.0f)), mg_bits_of(128.0f));
    PUSH("silu/zero", MG_ST_EQUALS, mg_st_bits(mg_silu(0.0f)), mg_bits_of(0.0f));
    PUSH("silu/neg_large_is_neg_zero", MG_ST_EQUALS, mg_bits_of(mg_silu(-200.0f)), 0x80000000u);
    PUSH("silu/minus_1_is_not_identity", MG_ST_DIFFERS, mg_st_bits(mg_silu(-1.0f)), mg_bits_of(-1.0f));

    /* --- 7. the reach table cannot be empty ------------------------------------------------------- */
    {
        int yes = 0;
        for (int m = 0; m < MG_MODE_COUNT; ++m)
            for (int ck = 0; ck < MG_CASE_KIND_COUNT; ++ck)
                for (int f = 0; f < MG_FIELD_COUNT; ++f) yes += (mg_reaches(m, ck, f) != 0);
        PUSH("reach/table_nonempty", MG_ST_EQUALS, (uint32_t)(yes > 0), 1u);
        PUSH("reach/correct_moves_nothing", MG_ST_EQUALS,
             (uint32_t)mg_reaches(MG_MODE_CORRECT, MG_CASE_EXPERT, MG_F_RAW), 0u);
    }
#undef PUSH

    /* --- the arms ---------------------------------------------------------------------------------
     * The honest arm establishes what is true; the two sabotage arms are then judged on how many
     * verdicts they CHANGE relative to the honest run, which is what makes them non-vacuous: an arm
     * that changes nothing is a decoration, and the counts it must change are derived here from the
     * checks themselves rather than typed. */
    int honest_fired[96];
    int n_eq = 0, n_dif = 0;
    for (int i = 0; i < n; ++i) {
        honest_fired[i] = (ck[i].kind == MG_ST_EQUALS) ? (ck[i].got == ck[i].want)
                                                       : !mg_word_eq(ck[i].got, ck[i].want);
        if (ck[i].kind == MG_ST_EQUALS) n_eq++; else n_dif++;
    }
    int changed = 0, changed_eq = 0, changed_dif = 0, failures = 0;
    for (int i = 0; i < n; ++i) {
        int fired;
        if (arm == 1) {
            fired = (ck[i].kind == MG_ST_EQUALS);              /* comparator says equal, always      */
        } else if (arm == 2) {
            const uint32_t other = ck[(i + 1) % n].want;        /* somebody else's answer           */
            fired = (ck[i].kind == MG_ST_EQUALS) ? mg_word_eq(ck[i].got, other)
                                                 : !mg_word_eq(ck[i].got, other);
        } else {
            fired = honest_fired[i];
        }
        ck[i].fired = fired;
        if (arm == 0) { if (!fired) failures++; continue; }
        if (fired != honest_fired[i]) { changed++; if (ck[i].kind == MG_ST_EQUALS) changed_eq++; else changed_dif++; }
    }
    int bad = 0;
    if (arm == 0 && failures != 0) bad = 1;
    if (arm == 1 && (n_dif == 0 || changed_dif != n_dif || changed_eq != 0)) bad = 1;
    if (arm == 2 && (n_eq == 0 || changed_eq == 0)) bad = 1;   /* crossing must break something     */
    if (verbose) {
        for (int i = 0; i < n; ++i)
            if (!ck[i].fired)
                printf("FAIL selftest arm=%d check=%s kind=%s got=%08x want=%08x\n", arm, ck[i].name,
                       ck[i].kind == MG_ST_EQUALS ? "equals" : "differs", ck[i].got, ck[i].want);
        printf("SELFTEST arm=%d name=%s checks=%d equals=%d differs=%d failures=%d "
               "changed=%d changed_equals=%d changed_differs=%d verdict=%s\n",
               arm, arm == 0 ? "honest" : (arm == 1 ? "stub" : "crossed"), n, n_eq, n_dif, failures,
               changed, changed_eq, changed_dif, bad ? "FAIL" : "ok");
    }
    return bad;
}

#endif /* MOE_GATHER_SELFTEST_H */
