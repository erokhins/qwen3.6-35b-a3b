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
#include "moe_gather_args.h"

#include <errno.h>

/* The one flag table this binary accepts. Anything not listed here is an ERROR at the gate of main(),
 * not a silent default (moe_gather_args.h) — that is the fix for HANDOFF §7.1. */
static const MgArgSpec MG_ARGS[] = {
    {"--selftest", 0}, {"--arm", 1},
    {"--plan", 1}, {"--slab", 1}, {"--expect", 1}, {"--out", 1}, {"--report", 1},
    {"--out-floats", 1}, {"--expect-floats", 1}, {"--expect-units", 1}, {"--expect-rows", 1},
    {"--expect-cases", 1}, {"--expect-blocks", 1},
    {0, 0}
};

static void usage(void) {
    printf("usage: host_probe --plan P --slab S --expect E --out O --report R "
           "--out-floats N --expect-floats N --expect-units N --expect-rows N "
           "--expect-cases N --expect-blocks N | --selftest [--arm A]\n");
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
    char bad[128];
    if (mg_arg_validate(argc, argv, MG_ARGS, bad, sizeof(bad))) {
        printf("FAIL argv: %s\n", bad);
        usage();
        return 2;
    }
    if (mg_arg_has(argc, argv, "--selftest")) {
        int ok = 1;
        const long long arm_ll = mg_arg_ll(argc, argv, "--arm", -1, &ok);
        if (!ok) { printf("FAIL argv: --arm must be an integer\n"); usage(); return 2; }
        const int only = (int)arm_ll;               /* -1 = all three arms (the default) */
        if (only < -1 || only >= MG_ST_ARMS) {
            printf("FAIL argv: --arm %d outside [0,%d)\n", only, MG_ST_ARMS);
            return 2;
        }
        MgStResult rr[MG_ST_ARMS];
        const int bad3 = mg_selftest_all(1, only, rr);
        printf("HOSTPROBE SELFTEST arms=%d failures=%d\n", (only < 0) ? MG_ST_ARMS : 1, bad3);
        if (bad3) printf("RESULT FAIL host_probe selftest arms=%d\n", (only < 0) ? MG_ST_ARMS : 1);
        else printf("RESULT PASS host_probe selftest arms=%d\n", (only < 0) ? MG_ST_ARMS : 1);
        return bad3 ? 1 : 0;
    }

    const char *plan_path = mg_arg_value(argc, argv, "--plan", 0);
    const char *slab_path = mg_arg_value(argc, argv, "--slab", 0);
    const char *expect_path = mg_arg_value(argc, argv, "--expect", 0);
    const char *out_path = mg_arg_value(argc, argv, "--out", 0);
    const char *report_path = mg_arg_value(argc, argv, "--report", 0);
    int argv_ok = 1;
    const long long out_floats = mg_arg_ll(argc, argv, "--out-floats", 0, &argv_ok);
    const long long expect_floats = mg_arg_ll(argc, argv, "--expect-floats", 0, &argv_ok);
    const int exp_units = (int)mg_arg_ll(argc, argv, "--expect-units", -1, &argv_ok);
    const int exp_rows = (int)mg_arg_ll(argc, argv, "--expect-rows", -1, &argv_ok);
    const int exp_cases = (int)mg_arg_ll(argc, argv, "--expect-cases", -1, &argv_ok);
    const int exp_blocks = (int)mg_arg_ll(argc, argv, "--expect-blocks", -1, &argv_ok);
    if (!argv_ok) { printf("FAIL argv: a numeric flag is not a whole integer\n"); usage(); return 2; }
    if (!plan_path || !slab_path || !expect_path || !out_path || !report_path) {
        printf("FAIL argv: --plan/--slab/--expect/--out/--report are all required\n");
        usage();
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
    long long sum_bad = 0, sum_nan = 0, sum_route_bad = 0, sum_route_n = 0;
    for (int i = 0; i < pc; ++i) {
        /* the same ten columns the driver writes, so gate.py parses one record shape for both arms
         * and a per-case counter disagreement is a finding rather than a column-count artefact */
        fprintf(rf, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", i, p.cases[i].kind, p.cases[i].mode,
                p.cases[i].block, p.cases[i].row, p.cases[i].slot, cts[i].bad, cts[i].nan,
                cts[i].route_bad, cts[i].route_n);
        sum_bad += cts[i].bad;
        sum_nan += cts[i].nan;
        sum_route_bad += cts[i].route_bad;
        sum_route_n += cts[i].route_n;
    }
    fclose(rf);
    printf("REPORT file=%s cases=%d bad=%lld nan=%lld route_bad=%lld route_n=%lld\n",
           report_path, pc, sum_bad, sum_nan, sum_route_bad, sum_route_n);
    printf("RESULT PASS host_probe cases=%d out_floats=%lld\n", pc, out_floats);
    return 0;
}
