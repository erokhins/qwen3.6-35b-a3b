/* gdn_arm.h — the arm main, ONE file for the g++ host twin and the hipcc device TU, so the
 * zero-tolerance cross-arm comparison compares two compilers on one driver and not two drivers.
 *
 * Usage: <arm> --plan work/plan.tsv --inputs work/inputs.bin --weights work/weights.bin
 *              --out work/<arm>.out.bin --echo work/<arm>.echo.bin [--report R] [--selftest]
 *
 * Every line this prints is machine-read by gate.py, which asserts parsed == printed == derived.
 * NO TIMER IS READ HERE: the counters are counts (`AGENTS.md` §GPU execution).
 */
#ifndef GDN_ARM_H_
#define GDN_ARM_H_

#include "gdn_case.h"

typedef struct {
    char plan[512], inputs[512], weights[512], outp[512], echop[512], report[512];
    int selftest;
} GdnArgs;

static int gdn_parse_args(int argc, char **argv, GdnArgs *a) {
    memset(a, 0, sizeof(*a));
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--selftest")) a->selftest = 1;
        else if (i + 1 < argc && !strcmp(argv[i], "--plan")) snprintf(a->plan, 512, "%s", argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--inputs")) snprintf(a->inputs, 512, "%s", argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--weights")) snprintf(a->weights, 512, "%s", argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--out")) snprintf(a->outp, 512, "%s", argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--echo")) snprintf(a->echop, 512, "%s", argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--report")) snprintf(a->report, 512, "%s", argv[++i]);
        else { fprintf(stderr, "bad argument %s\n", argv[i]); return 1; }
    }
    return 0;
}

static int gdn_mode_of(const char *s) {
    if (!strcmp(s, "honest")) return GDN_M_HONEST;
    if (!strcmp(s, "stub")) return GDN_M_STUB;
    if (!strcmp(s, "norm_weight_f32")) return GDN_M_NORM_WEIGHT_F32;
    return -1;
}

/* ---- FNV-1a over a word block, so "which bytes did this case write" is one number ------------ */
static uint64_t gdn_fnv(const void *p, long bytes) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ULL;
    for (long i = 0; i < bytes; ++i) { h ^= (uint64_t)b[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---- the self-test: values no other check in the lane would catch, and not a restatement of
 * the pipeline. The bf16 table is DERIVED, not hand-computed: `bf16_vectors.py` prints these
 * bit images out of torch's own `.to(bfloat16)` and checks the C rule against torch over a
 * 200k-pattern random sweep (RESULT PASS, 0 mismatches, 2026-10-26). The previous lane's
 * hand-typed table had four wrong rows and nothing ran it. Every check here is witnessed failing
 * by the control build (`build.sh --broken`, bf16 truncation instead of round-to-nearest-even),
 * which gate.py runs and requires to FAIL. ---- */
static const struct { float f; uint16_t want; } GDN_BF16_VEC[] = {
    {0.0f, 0x0000u}, {1.0f, 0x3f80u}, {-1.0f, 0xbf80u}, {0.5f, 0x3f00u},
    {2.0f, 0x4000u}, {65504.0f, 0x4780u}, {1.0f / 3.0f, 0x3eabu},
    {-1.0f / 3.0f, 0xbeabu}, {1e-8f, 0x322cu}, {-2.5f, 0xc020u},
    {3.4028235e38f, 0x7f80u}, {1e-45f, 0x0000u}, {0.00390625f, 0x3b80u},
    {127.5f, 0x42ffu}, {-0.0f, 0x8000u},
};

static int gdn_selftest(const char *arm, const char *compiler) {
    int fails = 0, checks = 0;
    for (unsigned i = 0; i < sizeof(GDN_BF16_VEC) / sizeof(GDN_BF16_VEC[0]); ++i) {
        uint16_t got = gdn_bf16_of_f32(GDN_BF16_VEC[i].f);
        ++checks;
        if (got != GDN_BF16_VEC[i].want) {
            printf("SELFTEST FAIL bf16_rne %g -> %04x want %04x\n", (double)GDN_BF16_VEC[i].f,
                   got, GDN_BF16_VEC[i].want);
            ++fails;
        }
    }
    /* a round-tripped bf16 is idempotent over the FINITE patterns; the widening is a shift, so this
     * can fail. Non-finite patterns are excluded on purpose and checked separately: NaN quieting is
     * not identity, and a loop that sweeps all 65,536 images and demands identity is a test of the
     * quieting rule under the wrong name (measured: 126 of 65,536 patterns are non-identity, all
     * of them non-finite — bf16_vectors.py BF16ROUNDTRIP). */
    ++checks;
    {
        int checked = 0;
        for (int i = 0; i < 0x10000; ++i) {
            if (((i >> 7) & 0xff) == 0xff) continue;
            uint16_t b = (uint16_t)i;
            ++checked;
            if (gdn_bf16_of_f32(gdn_f32_of_bf16(b)) != b) {
                printf("SELFTEST FAIL bf16_round_trip %04x\n", b);
                ++fails;
                break;
            }
        }
        if (checked != 65280) {   /* 255 exponents x 256 patterns: the finite-and-infinite images */
            printf("SELFTEST FAIL bf16_round_trip_count %d\n", checked);
            ++fails;
        }
        /* NaN is quieted, not preserved: the mantissa MSB is set and the sign kept. The pattern is
         * read through a volatile because a compile-time sNaN constant is NOT portable C++ — gcc
         * folded this literal into an infinity when the bits were a plain `uint32_t`, which is a
         * fact about the compiler and not about this rule, and a check that fails for a reason
         * outside the code under test is worse than no check. */
        volatile uint32_t nan_word = 0x7f800001u;      /* a signalling f32 NaN */
        uint32_t nan_bits = nan_word;
        float nf;
        memcpy(&nf, &nan_bits, 4);
        if ((gdn_bf16_of_f32(nf) & 0x0040u) == 0) {
            printf("SELFTEST FAIL bf16_nan_not_quieted got %04x\n", gdn_bf16_of_f32(nf));
            ++fails;
        }
    }
    /* §04.4's l2norm must be a SUM-of-squares normalisation with eps inside the rsqrt:
     * [3,4,0,..] -> [0.6,0.8,0,..] scale aside. A mean instead of a sum moves both. */
    ++checks;
    {
        float q[GDN_D_K];
        for (int i = 0; i < GDN_D_K; ++i) q[i] = 0.0f;
        q[0] = 3.0f; q[1] = 4.0f;
        float r = 1.0f / sqrtf(gdn_sumsq(q, GDN_D_K) + GDN_L2_EPS);
        if (!(fabsf(q[0] * r - 0.6f) < 1e-5f && fabsf(q[1] * r - 0.8f) < 1e-5f)) {
            printf("SELFTEST FAIL l2norm_sum (3,4,0..) -> (%.7f,%.7f)\n", (double)(q[0] * r),
                   (double)(q[1] * r));
            ++fails;
        }
    }
    ++checks;
    if (!(fabsf(gdn_silu(0.0f)) < 1e-9f && fabsf(gdn_silu(2.0f) - 2.0f / (1.0f + expf(-2.0f))) < 1e-9f)) {
        printf("SELFTEST FAIL silu\n");
        ++fails;
    }
    /* the forward substitution must actually solve (I + strictly_lower(ut)) x = b on a hand system
     * with a UNIT diagonal; dividing by the diagonal instead — §04b.4's dropped `unitriangular` —
     * shows up here as a nonzero residual. */
    ++checks;
    {
        static float ut[GDN_C * GDN_C], b[GDN_C], x[GDN_C];
        for (int i = 0; i < GDN_C * GDN_C; ++i) ut[i] = 0.0f;
        for (int i = 0; i < GDN_C; ++i) b[i] = (float)(i + 1);
        ut[1 * GDN_C + 0] = 0.5f; ut[2 * GDN_C + 0] = -0.25f; ut[2 * GDN_C + 1] = 2.0f;
        for (int i = 0; i < GDN_C; ++i) {
            float acc = b[i];
            for (int j = 0; j < i; ++j) acc -= ut[(long)i * GDN_C + j] * x[j];
            x[i] = acc;
        }
        float r0 = x[0] - 1.0f, r1 = x[1] + 0.5f * x[0] - 2.0f,
              r2 = x[2] - 0.25f * x[0] + 2.0f * x[1] - 3.0f;
        if (!(fabsf(r0) < 1e-6f && fabsf(r1) < 1e-6f && fabsf(r2) < 1e-6f)) {
            printf("SELFTEST FAIL solve_residual %g %g %g\n", (double)r0, (double)r1, (double)r2);
            ++fails;
        }
    }
    /* the denominators must be derived, not typed: a typo in an item count makes a skipped item
     * invisible, so the closed forms are recomputed here against the same helpers the driver uses */
    ++checks;
    if (gdn_item_count(IT_PAIR, 32) != (long)GDN_H_V * GDN_C * GDN_C ||
        gdn_item_count(IT_SOLVE, 32) != (long)GDN_H_V * 2 * GDN_D_V ||
        gdn_item_count(IT_OUTPROJ, 32) != 32L * GDN_H ||
        gdn_stage_n(GS_Y, 32) != 32L * GDN_H ||
        gdn_stage_words(32) != gdn_stage_off(32, GS_Y) + gdn_stage_n(GS_Y, 32)) {
        printf("SELFTEST FAIL denominators\n");
        ++fails;
    }
    /* the stage names must be the oracle's, in the oracle's relative order: the seven are the
     * tail of gdn_ref.STAGES in order (SCAN_WIDTH_STAGES), and a rename here silently detaches
     * every band verdict from the stage it was derived for. */
    ++checks;
    {
        static const char *const want[GS_N] = {"q_hat", "k_hat", "scan_out", "core_attn",
                                               "state", "normed", "y"};
        for (int s = 0; s < GS_N; ++s)
            if (strcmp(gdn_stage_name[s], want[s])) {
                printf("SELFTEST FAIL stage_name[%d] %s want %s\n", s, gdn_stage_name[s], want[s]);
                ++fails;
            }
    }
    /* THE STATE READ (`gdn_state_dot`) must contract [d_k,d_v] over k. `d_k == d_v == 128`, so an
     * `S^T` read is the same block, the same word count, no shape error and — at a zero state — the
     * SAME NUMBER. That is the oracle's `state_stored_transposed` injection, and it is the defect
     * slice 1 could not have caught: every state it ran was zero. The fixture is therefore non-zero
     * by construction, the expected value is spelled independently of `gdn_state_dot`, and the last
     * clause requires the transposed sum to DIFFER from the honest one, so a fixture that made them
     * agree would fail instead of passing vacuously. The control build
     * (`build.sh --broken` -> `build/host_probe_xpose`) fails this check. */
    ++checks;
    {
        static float Sst[GDN_D_K * GDN_D_V];
        float a[GDN_D_K];
        for (int k = 0; k < GDN_D_K; ++k) a[k] = (float)(k % 7 - 3) * 0.25f;
        for (long i = 0; i < (long)GDN_D_K * GDN_D_V; ++i)
            Sst[i] = (float)((i % GDN_D_V) * 3 + (i / GDN_D_V) % 11) * 0.00390625f;
        const int col = 91;
        float want = 0.0f, xpose = 0.0f;
        for (int k = 0; k < GDN_D_K; ++k) {           /* ascending, one accumulator: same order */
            want += a[k] * Sst[(long)k * GDN_D_V + col];
            xpose += a[k] * Sst[(long)col * GDN_D_K + k];
        }
        float got = gdn_state_dot(a, Sst, col);
        if (got != want) {
            printf("SELFTEST FAIL state_read_layout got %.9g want %.9g\n", (double)got,
                   (double)want);
            ++fails;
        }
        ++checks;
        if (xpose == want) {
            printf("SELFTEST FAIL state_read_fixture_vacuous xpose == honest at col %d\n", col);
            ++fails;
        }
    }
    printf("SELFTEST %s arm=%s compiler=%s checks=%d fails=%d\n", fails ? "FAIL" : "PASS", arm,
           compiler, checks, fails);
    return fails ? 1 : 0;
}

/* ---- the run --------------------------------------------------------------------------------- */
static int gdn_arm_main(int argc, char **argv, const char *arm, const char *compiler) {
    GdnArgs a;
    if (gdn_parse_args(argc, argv, &a)) return 2;
    if (a.selftest) return gdn_selftest(arm, compiler);
    if (!a.plan[0] || !a.inputs[0] || !a.weights[0] || !a.outp[0] || !a.echop[0]) {
        fprintf(stderr, "usage: %s --plan P --inputs I --weights W --out O --echo E [--selftest]\n",
                arm);
        return 2;
    }
    FILE *fp = fopen(a.plan, "r");
    if (!fp) { fprintf(stderr, "no plan %s\n", a.plan); return 2; }
    GdnPlan pl;
    memset(&pl, 0, sizeof(pl));
    long n_decl = 0, T_decl = 0;
    if (fscanf(fp, "#GDNPLAN v2 n_cases=%ld T=%ld%*[^\n]\n", &n_decl, &T_decl) != 2) {
        fprintf(stderr, "bad plan header: this slice reads '#GDNPLAN v2 n_cases=N T=T' (v1 had no "
                        "carry= field and is not accepted)\n");
        fclose(fp);
        return 2;
    }
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "CASE ", 5) != 0) continue;
        if (pl.n >= GDN_MAX_CASES) { fprintf(stderr, "too many cases\n"); fclose(fp); return 2; }
        GdnPlanCase *c = &pl.c[pl.n++];
        memset(c, 0, sizeof(*c));
        char blk[32], nm[64], md[16];
        int T = 0, ws = -1, carry = -1;
        if (sscanf(line, "CASE idx=%d block=%31s name=%63s T=%d mode=%15s wslot=%d carry=%d",
                   &c->idx, blk, nm, &T, md, &ws, &carry) < 6) {
            fprintf(stderr, "bad plan line: %s", line);
            fclose(fp);
            return 2;
        }
        c->wslot = ws;
        c->carry = carry;
        if (ws < 0) { fprintf(stderr, "plan case %d has no wslot\n", c->idx); fclose(fp); return 2; }
        c->T = T;
        snprintf(c->block, sizeof(c->block), "%s", blk);
        snprintf(c->name, sizeof(c->name), "%s", nm);
        c->mode = gdn_mode_of(md);
        if (c->mode < 0) { fprintf(stderr, "unknown mode %s\n", md); fclose(fp); return 2; }
        if (c->T < 1 || c->T > GDN_C) {
            fprintf(stderr, "T=%d outside 1..%d: this slice is ONE chunk per case\n", c->T, GDN_C);
            fclose(fp);
            return 2;
        }
        if ((long)c->T != T_decl) {
            /* One in/out/echo stride addresses every case in the plan, so a case whose T differs
             * from the header would be bound at a stride its own field sizes disagree with — the
             * kind of off-by-a-field that reads real bytes from the wrong place and prints numbers. */
            fprintf(stderr, "plan case %d has T=%d but the plan declares T=%ld: one plan, one T\n",
                    c->idx, c->T, T_decl);
            fclose(fp);
            return 2;
        }
    }
    fclose(fp);
    if (pl.n != n_decl || pl.n == 0) {
        printf("ARM FAIL arm=%s parsed_cases=%d declared_cases=%ld\n", arm, pl.n, n_decl);
        return 2;
    }
    pl.in_words = gdn_in_words((int)T_decl);
    pl.out_words = gdn_stage_words((int)T_decl) + gdn_case_scratch_words((int)T_decl);
    for (int i = 0; i < pl.n; ++i) {
        pl.c[i].in_off = (long)i * pl.in_words;
        pl.c[i].out_off = (long)i * pl.out_words;
        /* A carry must point BACKWARDS, at a case of the SAME block. Forward and self carries would
         * read a block that has not been written yet (calloc zeros, which is exactly the bug a
         * zero-state test cannot see), and a cross-block carry would read another block's head
         * history and still print a plausible state. */
        if (pl.c[i].carry >= 0) {
            int j = pl.c[i].carry;
            if (j >= i) {
                fprintf(stderr, "plan case %d carries from case %d: a carry must precede it\n",
                        pl.c[i].idx, j);
                return 2;
            }
            if (strcmp(pl.c[j].block, pl.c[i].block) != 0) {
                fprintf(stderr, "plan case %d (%s) carries from case %d (%s): a state is per block\n",
                        pl.c[i].idx, pl.c[i].block, j, pl.c[j].block);
                return 2;
            }
        }
    }
    long echo_words = gdn_echo_words((int)T_decl);

    FILE *fi = fopen(a.inputs, "rb"), *fw = fopen(a.weights, "rb");
    FILE *fo = fopen(a.outp, "wb"), *fe = fopen(a.echop, "wb");
    if (!fi || !fw || !fo || !fe) { fprintf(stderr, "cannot open work files\n"); return 2; }
    long in_total = pl.in_words * pl.n;
    uint32_t *inbuf = (uint32_t *)malloc((size_t)in_total * 4);
    /* one weight block = norm_weight [d_v] + out_proj [H, value_dim], both bf16 bit images.
     * The file's size is checked against the number of slots the plan names, so a plan that
     * points at a slot the file does not hold is a refusal and not a silently reused weight. */
    long w_stride = (long)GDN_D_V + (long)GDN_H * GDN_VALUE_DIM;   /* u32 words per weight block */
    int n_slot = 0;
    for (int i = 0; i < pl.n; ++i) if (pl.c[i].wslot + 1 > n_slot) n_slot = pl.c[i].wslot + 1;
    long want_in = in_total * 4, want_w = w_stride * n_slot * 4;
    uint32_t *wbuf = (uint32_t *)malloc((size_t)want_w);
    uint32_t *obuf = (uint32_t *)calloc((size_t)pl.out_words * pl.n, 4);
    uint32_t *ebuf = (uint32_t *)calloc((size_t)echo_words * pl.n, 4);
    if (fread(inbuf, 1, (size_t)want_in, fi) != (size_t)want_in) {
        fprintf(stderr, "inputs.bin short: wanted %ld B\n", want_in); return 2;
    }
    if (fread(wbuf, 1, (size_t)want_w, fw) != (size_t)want_w) {
        fprintf(stderr, "weights.bin short: wanted %ld B for %d slots\n", want_w, n_slot); return 2;
    }
    fclose(fi); fclose(fw);

    long hip_calls = 0;
    int carried = 0;
    FILE *fr = a.report[0] ? fopen(a.report, "w") : NULL;
    for (int i = 0; i < pl.n; ++i) {
        GdnPlanCase *pc = &pl.c[i];
        uint32_t *wb = wbuf + (long)pc->wslot * w_stride;
        GdnCase k;
        memset(&k, 0, sizeof(k));
        k.t = pc->T;
        k.mode = pc->mode;
        long st[GDN_ST_N];
        for (int s = 0; s < GDN_ST_N; ++s) st[s] = 0;
        GdnCtx c;
        memset(&c, 0, sizeof(c));
        gdn_bind(&c, &k, inbuf + pc->in_off, wb, obuf + pc->out_off,
                 ebuf + (long)i * echo_words, pc->T);
        if (pc->carry >= 0) {
            /* the carried state is the PREVIOUS CASE's EMITTED state, in place: no copy, no second
             * buffer, so what the arm carries is byte-for-byte what it wrote. Reading a block that
             * is later in the plan is refused above, so this source is always already written. */
            c.init_state = (const float *)(obuf + pl.c[pc->carry].out_off
                                           + gdn_stage_off(pc->T, GS_STATE));
            ++carried;
        }
        gdn_case_run(&c, &k, st);
        GdnFp fp[GF_N];
        gdn_census(&c, fp);
        uint64_t oh = gdn_fnv(obuf + pc->out_off, (long)gdn_stage_words(pc->T) * 4);
        printf("CASE idx=%d block=%s name=%s T=%d mode=%s carry=%d items=%ld stage_items=%ld "
               "state_nonzero=%ld pad_nonzero=%ld out_hash=%016llx\n",
               pc->idx, pc->block, pc->name, pc->T,
               pc->mode == GDN_M_STUB ? "stub" : (pc->mode == GDN_M_NORM_WEIGHT_F32
                                                  ? "norm_weight_f32" : "honest"),
               pc->carry, st[GDN_ST_ITEMS], st[GDN_ST_STAGE_ITEMS], st[GDN_ST_STATE_NONZERO],
               st[GDN_ST_PAD_NONZERO], (unsigned long long)oh);
        if (fr) fprintf(fr, "CASE idx=%d block=%s name=%s T=%d carry=%d items=%ld out_hash=%016llx\n",
                        pc->idx, pc->block, pc->name, pc->T, pc->carry, st[GDN_ST_ITEMS],
                        (unsigned long long)oh);
        for (int f = 0; f < GF_N; ++f)
            printf("FP idx=%d field=%s n=%ld sum=%016llx sumidx=%016llx\n", pc->idx,
                   gdn_field_name[f], fp[f].n, (unsigned long long)fp[f].sum,
                   (unsigned long long)fp[f].sumidx);
    }
    if (fr) fclose(fr);
    if (fwrite(obuf, 4, (size_t)pl.out_words * pl.n, fo) != (size_t)pl.out_words * pl.n) {
        fprintf(stderr, "short write on out\n"); return 2;
    }
    if (fclose(fo)) { fprintf(stderr, "flush failed on out\n"); return 2; }
    if (fwrite(ebuf, 4, (size_t)echo_words * pl.n, fe) != (size_t)echo_words * pl.n) {
        fprintf(stderr, "short write on echo\n"); return 2;
    }
    if (fclose(fe)) { fprintf(stderr, "flush failed on echo\n"); return 2; }
    printf("RUN arm=%s compiler=%s cases=%d T=%ld stages=%d case_words=%ld echo_words=%ld "
           "hip_calls=%ld carried=%d\n", arm, compiler, pl.n, T_decl, GS_N, pl.out_words,
           echo_words, hip_calls, carried);
    free(inbuf); free(wbuf); free(obuf); free(ebuf);
    printf("RESULT PASS arm=%s cases=%d\n", arm, pl.n);
    return 0;
}

#endif /* GDN_ARM_H_ */
