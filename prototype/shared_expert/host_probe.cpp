/* host_probe.cpp — the CPU twin of the device arm: the SAME `shared_expert_kernels.h` bodies, built
 * by g++ with the same `-ffp-contract=off`, walked over the SAME work decomposition the device uses.
 *
 * It exists so that a host-vs-device disagreement is attributable to the device compiler or ISA and
 * not to two spellings of the arithmetic. It is not a model of the device: it runs the identical
 * stage functions, in the identical block/row order, and writes the identical report.
 *
 *   ./host_probe --selftest [--arm=N]         the three-arm hand-worked suite (no files, no device)
 *   ./host_probe --plan P --slab S --expect E --out O --report R \
 *                --expect-weights N --expect-acts N --expect-cases N --out-floats N
 *
 * No timing of any kind is taken here, and nothing here touches the device or the GPU lock.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_expert_kernels.h"
#include "shared_expert_selftest.h"

typedef struct {
    const char *plan_path, *slab_path, *expect_path, *out_path, *report_path;
    int want_w, want_a, want_c, out_floats, expect_floats;
    int per_case;
} Args;

static int arg_int(const char *s, const char *why) {
    char *end = 0;
    long v = strtol(s, &end, 10);
    if (!end || *end) { fprintf(stderr, "bad integer %s for %s\n", s, why); exit(2); }
    return (int)v;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END); *n = (size_t)ftell(fh); fseek(fh, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc(*n ? *n : 1);
    if (fread(b, 1, *n, fh) != *n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(fh);
    return b;
}

static void parse(int argc, char **argv, Args *a) {
    a->plan_path = a->slab_path = a->expect_path = a->out_path = a->report_path = 0;
    a->want_w = a->want_a = a->want_c = a->out_floats = a->expect_floats = 0;
    a->per_case = 0;
    for (int i = 1; i < argc; ++i) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : 0;
        if (!strcmp(k, "--per-case")) { a->per_case = 1; continue; }
        if (!v) { fprintf(stderr, "%s needs a value\n", k); exit(2); }
        if (!strcmp(k, "--plan")) a->plan_path = v;
        else if (!strcmp(k, "--slab")) a->slab_path = v;
        else if (!strcmp(k, "--expect")) a->expect_path = v;
        else if (!strcmp(k, "--out")) a->out_path = v;
        else if (!strcmp(k, "--report")) a->report_path = v;
        else if (!strcmp(k, "--expect-weights")) a->want_w = arg_int(v, k);
        else if (!strcmp(k, "--expect-acts")) a->want_a = arg_int(v, k);
        else if (!strcmp(k, "--expect-cases")) a->want_c = arg_int(v, k);
        else if (!strcmp(k, "--out-floats")) a->out_floats = arg_int(v, k);
        else if (!strcmp(k, "--expect-floats")) a->expect_floats = arg_int(v, k);
        else { fprintf(stderr, "unknown argument %s\n", k); exit(2); }
        ++i;
    }
    if (!a->plan_path || !a->slab_path || !a->expect_path || !a->out_path || !a->report_path
        || a->out_floats <= 0 || a->want_c <= 0) {
        fprintf(stderr, "--plan --slab --expect --out --report --out-floats --expect-cases "
                        "are all required\n");
        exit(2);
    }
}

/* The work decomposition, built ONCE and used by the host loops and (as a device buffer) by the
 * kernels, so both arms split the same way. Every count is a multiple of SE_THREADS by geometry:
 * SE_I/128 = 4, SE_H/128 = 16, (SE_H/2)/128 = 8. */
typedef struct { int32_t case_idx, base; } SeItem;
typedef struct { SeItem *s1, *s2, *s3; int n1, n2, n3; } SeWork;

static int rows_of_mode(int mode) { return mode == SE_MODE_DOWN_HALF ? SE_H / 2 : SE_H; }

static void work_build(const SePlan *p, SeWork *w) {
    int n1 = 0, n2 = 0, n3 = 0;
    for (int i = 0; i < p->n_c; ++i) {
        n1 += SE_I / SE_THREADS;
        n2 += rows_of_mode(p->c[i].mode) / SE_THREADS;
        n3 += 1 + SE_H / SE_THREADS;      /* one prob block, then the channel blocks */
    }
    w->s1 = (SeItem *)malloc(sizeof(SeItem) * n1);
    w->s2 = (SeItem *)malloc(sizeof(SeItem) * n2);
    w->s3 = (SeItem *)malloc(sizeof(SeItem) * n3);
    w->n1 = w->n2 = w->n3 = 0;
    int b1 = 0, b2 = 0, b3 = 0;
    for (int i = 0; i < p->n_c; ++i) {
        for (int b = 0; b < SE_I / SE_THREADS; ++b) {
            w->s1[b1].case_idx = i; w->s1[b1].base = b * SE_THREADS; ++b1;
        }
        const int rows = rows_of_mode(p->c[i].mode);
        for (int b = 0; b < rows / SE_THREADS; ++b) {
            w->s2[b2].case_idx = i; w->s2[b2].base = b * SE_THREADS; ++b2;
        }
        w->s3[b3].case_idx = i; w->s3[b3].base = -1; ++b3;      /* the prob block */
        for (int b = 0; b < SE_H / SE_THREADS; ++b) {
            w->s3[b3].case_idx = i; w->s3[b3].base = b * SE_THREADS; ++b3;
        }
    }
    /* The counts are the filled ones. Setting them to zero before the fill and never back was the
     * first version's bug, and it was invisible except that every output word stayed poison. */
    w->n1 = b1; w->n2 = b2; w->n3 = b3;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--selftest")) {
        int only = -1;
        for (int i = 2; i < argc; ++i) {
            if (!strncmp(argv[i], "--arm=", 6)) only = atoi(argv[i] + 6);
            else if (!strcmp(argv[i], "--arm") && i + 1 < argc) only = atoi(argv[++i]);
            else { fprintf(stderr, "selftest: unknown argument %s\n", argv[i]); return 2; }
        }
        if (only == -1) return se_selftest_all(stdout);
        if (only < 0 || only > 2) { fprintf(stderr, "selftest: --arm must be 0, 1 or 2\n"); return 2; }
        SeStReport rep;
        se_selftest(only, &rep, stdout);
        fprintf(stdout, "SELFTEST3 arms_run=1 arm=%d checks=%d changed=%d\n", only, rep.n_checks,
                rep.changed);
        fprintf(stdout, "FAIL selftest single-arm: one arm alone is not a pass (arm=%d)\n", only);
        return 1;        /* by design; gate.py witnesses it */
    }

    Args a;
    parse(argc, argv, &a);
    size_t slab_n = 0;
    uint8_t *slab = read_file(a.slab_path, &slab_n);
    size_t exp_n = 0;
    uint8_t *expb = read_file(a.expect_path, &exp_n);
    if ((int)(exp_n / 4) != a.expect_floats) {
        fprintf(stderr, "expect holds %d floats, the plan wants %d\n", (int)(exp_n / 4),
                a.expect_floats);
        return 1;
    }
    SePlan p;
    memset(&p, 0, sizeof(p));
    p.w = (SeWeight *)calloc(a.want_w ? a.want_w : 1, sizeof(SeWeight));
    p.a = (SeAct *)calloc(a.want_a ? a.want_a : 1, sizeof(SeAct));
    p.c = (SeCase *)calloc(a.want_c, sizeof(SeCase));
    char err[512];
    if (se_read_plan(a.plan_path, &p, a.want_w, a.want_a, a.want_c, err, sizeof(err))) {
        fprintf(stderr, "plan: %s\n", err);
        return 1;
    }
    p.slab = slab; p.slab_bytes = slab_n;
    p.expect = (const float *)expb; p.n_expect = (int)(exp_n / 4);

    float *out = (float *)malloc(sizeof(float) * a.out_floats);
    for (int i = 0; i < a.out_floats; ++i) out[i] = se_float_of(SE_NAN_BITS);

    SeWork w;
    work_build(&p, &w);
    /* Launch bookkeeping. The grouped arm issues four launches for the whole plan; the per-case arm
     * issues four per case. The two arms differ in workgroups-per-launch as well as in launch count,
     * which `pitfalls/20` §"read workgroups per launch on both sides of an A/B" names as a confound -
     * and it is inert here only because this lane takes NO timing: the per-case arm exists to show the
     * WORDS are the same at a different split, which is a correctness statement. */
    const int launches = a.per_case ? 4 * p.n_c : 4;
    int counted = launches;

    for (int stage = 1; stage <= 3; ++stage) {
        const SeItem *items = stage == 1 ? w.s1 : (stage == 2 ? w.s2 : w.s3);
        const int n = stage == 1 ? w.n1 : (stage == 2 ? w.n2 : w.n3);
        for (int blk = 0; blk < n; ++blk) {
            const SeCase *c = &p.c[items[blk].case_idx];
            const int base = items[blk].base;
            if (stage == 1) {
                for (int t = 0; t < SE_THREADS; ++t) {
                    const int j = base + t;
                    int reads = 0;
                    se_stage_gateup(&p, c, out, j, &reads);
                    p.c[items[blk].case_idx].reads_fused += reads;
                }
            } else if (stage == 2) {
                for (int t = 0; t < SE_THREADS; ++t) {
                    int reads = 0;
                    se_stage_down(&p, c, out, base + t, &reads);
                    p.c[items[blk].case_idx].reads_down += reads;
                }
            } else if (base < 0) {
                if (c->sgate_form == SE_SGATE_SCALAR) {
                    int reads = 0;
                    const float prob = se_stage_prob(&p, c, &reads);
                    out[c->dst_off + SE_OFF_PROB] = prob;
                    p.c[items[blk].case_idx].reads_sg += reads;
                }
            } else {
                const float prob = out[c->dst_off + SE_OFF_PROB];
                for (int t = 0; t < SE_THREADS; ++t) {
                    int reads = 0;
                    const int h = base + t;
                    out[c->dst_off + SE_OFF_OUT + h] =
                        se_stage_out_channel(&p, c, out, prob, h, &reads);
                    /* the per-channel form reads one gate weight per channel; the scalar form reads
                     * none here, so both arms arrive at the same geometry-derived total SE_H */
                    p.c[items[blk].case_idx].reads_sg += reads;
                }
            }
        }
    }

    /* per-case counters: bad/nan/census/digest, all over the case's own span */
    for (int i = 0; i < p.n_c; ++i) {
        SeCase *c = &p.c[i];
        if (c->dst_off + c->n_floats > a.out_floats || c->exp_off + c->n_floats > p.n_expect) {
            fprintf(stderr, "case %d span %d+%d does not fit out=%d expect=%d\n", c->index,
                    c->dst_off, c->n_floats, a.out_floats, p.n_expect);
            return 1;
        }
        int nb = 0, nn = 0, nz = 0, nv = 0;
        SeDigest dg;
        se_digest_init(&dg);
        for (int k = 0; k < c->n_floats; ++k) {
            const float v = out[c->dst_off + k];
            if (!se_st_word_eq(se_bits_of(v), se_bits_of(p.expect[c->exp_off + k]))) ++nb;
            if (se_is_nan(v)) ++nn; else if (v == 0.0f) ++nz; else ++nv;
            se_digest_add(&dg, v);
        }
        c->bad = nb; c->nan = nn > 0 ? 1 : 0; c->n_nan = nn; c->n_zero = nz; c->n_value = nv;
        c->xor_sig = dg.xor_sig; c->sum_bits = dg.sum_bits; c->n_digest = (int)dg.n_nonzero;
    }

    FILE *fh = fopen(a.out_path, "wb");
    if (!fh || fwrite(out, 4, a.out_floats, fh) != (size_t)a.out_floats) {
        fprintf(stderr, "cannot write %s\n", a.out_path); return 1;
    }
    fclose(fh);
    fh = fopen(a.report_path, "w");
    fprintf(fh, "# case\tmode\tsgate\tcast\texpect\tbad\tnan\treads_fused\treads_down\treads_sg"
                "\tn_nan\tn_zero\tn_value\txor_sig\tsum_bits\tn_digest\n");
    for (int i = 0; i < p.n_c; ++i) {
        const SeCase *c = &p.c[i];
        fprintf(fh, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%u\t%d\n",
                c->index, c->mode, c->sgate_form, c->cast_form, c->expect, c->bad, c->nan,
                c->reads_fused, c->reads_down, c->reads_sg, c->n_nan, c->n_zero, c->n_value,
                c->xor_sig, c->sum_bits, c->n_digest);
    }
    fclose(fh);
    printf("PLAN W=%d/%d A=%d/%d C=%d/%d slab=%zu expect=%d out=%d\n", p.n_w, a.want_w, p.n_a,
           a.want_a, p.n_c, a.want_c, slab_n, p.n_expect, a.out_floats);
    printf("LAUNCHES mode=%s issued=%d device_counted=%d blocks_stage1=%d stage2=%d stage3=%d "
           "threads=%d wg_per_launch_grouped=%d,%d,%d wg_per_launch_per_case=%d,%d,%d\n",
           a.per_case ? "per-case" : "grouped", launches, counted, w.n1, w.n2, w.n3, SE_THREADS,
           w.n1, w.n2, w.n3, SE_I / SE_THREADS,
           SE_H / SE_THREADS, 1 + SE_H / SE_THREADS);
    printf("ARM host compiled=%s\n",
#if defined(__clang__)
           "clang-host"
#else
           "gpp"
#endif
           );
    printf("no timing taken\n");
    return 0;
}
