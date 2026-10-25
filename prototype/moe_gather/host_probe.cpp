/* host_probe.cpp — the g++ build of the gather bodies, run on the same plan, slab and expectation
 * buffer as the device kernel.
 *
 * Two arms, one source: `moe_gather` (hipcc, gfx1151) and `host_probe` (g++) both call
 * `mg_run_case()` from `moe_gather_kernels.h`, so a disagreement between the two files is a fact
 * about the device compiler or the ISA and not about two spellings of the arithmetic
 * (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the thing
 * under test"). gate.py requires the two output files to be bit-identical.
 *
 * Run order matches the device's launch order exactly: per block, every EXPERT case of that block,
 * then every COMBINE case of that block, mode 0 first — a COMBINE case reading MG_SRC_KERNEL_RAW
 * consumes the mode-0 EXPERT cases' raw outputs, so the order is load-bearing on both arms.
 *
 * The host build runs each case with tid = 0, nthreads = 1: it is the arithmetic that is being
 * compared across compilers, not the work partition, and the device runs the same case with 256
 * threads, which is what makes the cross-arm comparison a parallelism test as well.
 *
 * No timer, no warm loop, no timing of any kind (AGENTS.md §"GPU execution is serialized"): this
 * program exists to produce words, not durations.
 */
#include "moe_gather_kernels.h"
#include "moe_gather_selftest.h"

#include <errno.h>

static const char *arg_value(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return dflt;
}
static int have_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

static char *read_file(const char *path, size_t *n_out) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { printf("FAIL cannot open %s: %s\n", path, strerror(errno)); return 0; }
    fseek(fh, 0, SEEK_END);
    const long n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n ? (size_t)n : 1);
    if (!b) { fclose(fh); printf("FAIL oom reading %s\n", path); return 0; }
    if (fread(b, 1, (size_t)n, fh) != (size_t)n) { fclose(fh); free(b); printf("FAIL short read %s\n", path); return 0; }
    fclose(fh);
    *n_out = (size_t)n;
    return b;
}

int main(int argc, char **argv) {
    if (have_flag(argc, argv, "--selftest")) {
        const int want_arm = atoi(arg_value(argc, argv, "--arm", "0"));
        int bad = 0, n = 0;
        for (int arm = 0; arm < 3; ++arm) {
            if (want_arm >= 0 && arm != want_arm) continue;
            bad += mg_selftest(arm, 1);
            n++;
        }
        printf("HOSTPROBE SELFTEST arms=%d failures=%d\n", n, bad);
        if (bad) printf("RESULT FAIL host_probe selftest failures=%d\n", bad);
        else printf("RESULT PASS host_probe selftest arms=%d\n", n);
        return bad ? 1 : 0;
    }

    const char *plan_path = arg_value(argc, argv, "--plan", 0);
    const char *slab_path = arg_value(argc, argv, "--slab", 0);
    const char *expect_path = arg_value(argc, argv, "--expect", 0);
    const char *out_path = arg_value(argc, argv, "--out", 0);
    const char *report_path = arg_value(argc, argv, "--report", 0);
    const long long out_floats = atoll(arg_value(argc, argv, "--out-floats", "0"));
    const long long expect_floats = atoll(arg_value(argc, argv, "--expect-floats", "0"));
    const int exp_units = atoi(arg_value(argc, argv, "--expect-units", "-1"));
    const int exp_rows = atoi(arg_value(argc, argv, "--expect-rows", "-1"));
    const int exp_cases = atoi(arg_value(argc, argv, "--expect-cases", "-1"));
    const int exp_blocks = atoi(arg_value(argc, argv, "--expect-blocks", "-1"));
    if (!plan_path || !slab_path || !expect_path || !out_path || !report_path) {
        printf("usage: host_probe --plan P --slab S --expect E --out O --report R "
               "--out-floats N --expect-floats N --expect-units N --expect-rows N "
               "--expect-cases N --expect-blocks N | --selftest [--arm A]\n");
        return 2;
    }

    /* count the plan's rows first, so "parsed" and "counted" are two numbers and not one */
    int cu = 0, cr = 0, cc = 0, cb = 0;
    {
        size_t n = 0;
        char *t = read_file(plan_path, &n);
        if (!t) return 1;
        for (size_t i = 0; i < n; ++i) {
            if (i == 0 || t[i - 1] == '\n') {
                if (t[i] == 'U') cu++;
                else if (t[i] == 'R') cr++;
                else if (t[i] == 'B') cb++;
                else if (t[i] == 'C') cc++;
            }
        }
        free(t);
    }
    MgPlan p;
    memset(&p, 0, sizeof(p));
    p.units = (MgUnit *)calloc((size_t)(cu ? cu : 1), sizeof(MgUnit));
    p.rows = (MgRow *)calloc((size_t)(cr ? cr : 1), sizeof(MgRow));
    p.cases = (MgCase *)calloc((size_t)(cc ? cc : 1), sizeof(MgCase));
    p.blocks = (MgBlock *)calloc((size_t)(cb ? cb : 1), sizeof(MgBlock));
    int pu = 0, pr = 0, pc = 0, pb = 0;
    const int rc = mg_read_plan(plan_path, &p, cu, cr, cc, cb, &pu, &pr, &pc, &pb);
    printf("PLAN plan_file=%s units=%d/%d rows=%d/%d cases=%d/%d blocks=%d/%d rc=%d\n",
           plan_path, pu, cu, pr, cr, pc, cc, pb, cb, rc);
    if (rc != 0 || pu != cu || pr != cr || pc != cc || pb != cb) {
        printf("FAIL plan counted %d/%d/%d/%d rows but holds %d/%d/%d/%d records (rc=%d)\n",
               cu, cr, cc, cb, pu, pr, pc, pb, rc);
        return 1;
    }
    if (pu != exp_units || pr != exp_rows || pc != exp_cases || pb != exp_blocks) {
        printf("FAIL plan holds %d/%d/%d/%d records, gate expected %d/%d/%d/%d\n",
               pu, pr, pc, pb, exp_units, exp_rows, exp_cases, exp_blocks);
        return 1;
    }
    p.n_units = pu; p.n_rows = pr; p.n_cases = pc; p.n_blocks = pb;

    size_t sb = 0;
    p.slab = read_file(slab_path, &sb);
    p.slab_bytes = sb;
    if (!p.slab) return 1;
    size_t eb = 0;
    char *expect = read_file(expect_path, &eb);
    if (!expect) return 1;
    if ((long long)(eb / 4) != expect_floats) {
        printf("FAIL expect file holds %lld floats, gate said %lld\n", (long long)(eb / 4), expect_floats);
        return 1;
    }
    p.expect = (const uint32_t *)(void *)expect;

    if (mg_verify_plan(&p, out_floats, expect_floats, 1)) {
        printf("FAIL plan verification found defects; refusing to run\n");
        return 1;
    }

    float *out = (float *)calloc((size_t)(out_floats ? out_floats : 1), sizeof(float));
    MgCounters *cts = (MgCounters *)calloc((size_t)(pc ? pc : 1), sizeof(MgCounters));
    if (!out || !cts) { printf("FAIL oom\n"); return 1; }
    for (long long i = 0; i < out_floats; ++i) out[i] = mg_nan();

    /* the run order the device reproduces: per block, one grouped pass over every EXPERT case of that
     * block (all selected experts of the block in one launch - prediction 2), then one pass over its
     * COMBINE cases. Mode 0 goes first inside each phase because a COMBINE case with
     * MG_SRC_KERNEL_RAW consumes the mode-0 EXPERT records' raw outputs. Cases are otherwise
     * independent: each writes only its own record and its own counters. */
    for (int b = 0; b < p.n_blocks; ++b) {
        for (int round = 0; round < 2; ++round) {
            for (int phase = 0; phase < 2; ++phase) {
                for (int i = p.blocks[b].case_lo; i <= p.blocks[b].case_hi; ++i) {
                    MgCase *c = &p.cases[i];
                    if (c->kind != phase) continue;
                    if ((c->mode == MG_MODE_CORRECT) != (round == 0)) continue;
                    const float *rawbase; int stride = 0, roff = 0;
                    if (c->kind == MG_CASE_COMBINE) {
                        if (mg_combine_raw_src(&p, c, out, &rawbase, &stride, &roff)) {
                            printf("FAIL case %d has no defined combine source\n", i);
                            return 1;
                        }
                    } else {
                        rawbase = 0;
                    }
                    mg_run_case(&p, c, rawbase, stride, roff, out + c->dst_off, 0, 1, &cts[i]);
                }
            }
        }
    }

    FILE *of = fopen(out_path, "wb");
    if (!of) { printf("FAIL cannot write %s\n", out_path); return 1; }
    if (fwrite(out, 4, (size_t)out_floats, of) != (size_t)out_floats) {
        printf("FAIL short write on %s\n", out_path); fclose(of); return 1;
    }
    fclose(of);
    printf("OUT file=%s floats=%lld bytes=%lld\n", out_path, out_floats, out_floats * 4);

    FILE *rf = fopen(report_path, "wb");
    if (!rf) { printf("FAIL cannot write %s\n", report_path); return 1; }
    long long sum_bad = 0, sum_nan = 0;
    for (int i = 0; i < pc; ++i) {
        fprintf(rf, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", i, p.cases[i].kind, p.cases[i].mode,
                p.cases[i].block, p.cases[i].row, p.cases[i].slot, cts[i].bad, cts[i].nan,
                cts[i].route_bad);
        sum_bad += cts[i].bad;
        sum_nan += cts[i].nan;
    }
    fclose(rf);
    printf("REPORT file=%s cases=%d bad=%lld nan=%lld\n", report_path, pc, sum_bad, sum_nan);
    printf("RESULT PASS host_probe cases=%d out_floats=%lld\n", pc, out_floats);
    return 0;
}
