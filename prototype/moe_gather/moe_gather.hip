/* moe_gather.hip — the device side of prototype/moe_gather: the routed-expert gather on the iGPU.
 *
 * One workgroup per case, one launch per (block, phase): the gather launch of a block carries every
 * selected expert of that block (all K slots of every sampled row), which is the shape AGENTS.md
 * prediction 2 asks for — grouped launches rather than one dependent read per expert slot. The
 * combine is a second launch per block because it needs all K of the block's expert outputs before
 * any row's weighted sum is defined. A `--per-slot` mode emits the alternative shape (one launch per
 * (block, slot)) so the two are counted and compared, not merely asserted (see results.md).
 *
 * The arithmetic is in moe_gather_kernels.h and is shared verbatim with host_probe.cpp (g++); this
 * file owns only the HIP plumbing: allocation, the two launches, the counters, and the report.
 *
 * CORRECTNESS ONLY. No timer is constructed anywhere in this file, no duration is printed, and there
 * is no warm loop, burst or best-of-N: measurements belong to the main agent (AGENTS.md §"GPU
 * execution is serialized and belongs to the main agent"). The launch counters below count launches,
 * which is a structural fact, not a rate.
 */
#include <hip/hip_runtime.h>

#include <errno.h>

#include "moe_gather_kernels.h"
#include "moe_gather_selftest.h"
#include "moe_gather_args.h"

#define MG_BLOCK_THREADS 256

static int g_failures = 0;

#define CK(call)                                                              \
    do {                                                                      \
        const hipError_t _e = (call);                                         \
        if (_e != hipSuccess) {                                               \
            printf("FAIL hip: %s = %s (%s)\n", #call, hipGetErrorString(_e),  \
                   hipGetErrorName(_e));                                      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static const MgArgSpec MG_ARGS[] = {
    {"--selftest", 0}, {"--reach-table", 0}, {"--check-only", 0}, {"--per-slot", 0}, {"--arm", 1},
    {"--plan", 1}, {"--slab", 1}, {"--expect", 1}, {"--out", 1}, {"--report", 1},
    {"--out-floats", 1}, {"--expect-floats", 1}, {"--expect-units", 1}, {"--expect-rows", 1},
    {"--expect-cases", 1}, {"--expect-blocks", 1},
    {0, 0}
};

static void usage(void) {
    printf("usage: moe_gather --plan P --slab S --expect E --out O --report R "
           "--out-floats N --expect-floats N --expect-units N --expect-rows N "
           "--expect-cases N --expect-blocks N [--per-slot] | --check-only | --selftest "
           "[--arm A] | --reach-table\n");
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

/* ---- the kernel --------------------------------------------------------------------------------- *
 * blockIdx.x selects the case; one workgroup of 256 threads computes the whole record. Every write
 * goes through mg_run_case, the same function the g++ build calls.
 */
__global__ void mg_case_kernel(const MgPlan p, const int *case_idx, int n_cases,
                               float *out, MgCounters *cts, int *launch_count) {
    const int ci = case_idx[blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(launch_count, 1);
    const MgCase *c = &p.cases[ci];
    const float *rawbase = 0;
    int stride = 0, roff = 0;
    if (c->kind == MG_CASE_COMBINE) {
        if (mg_combine_raw_src(&p, c, out, &rawbase, &stride, &roff)) {
            /* undefined source: fill the record with the loud pattern rather than read garbage */
            for (int i = threadIdx.x; i < c->n_floats; i += blockDim.x) out[c->dst_off + i] = mg_nan();
            if (threadIdx.x == 0) atomicAdd(&cts[ci].nan, 1);
            return;
        }
    }
    mg_run_case(&p, c, rawbase, stride, roff, out + c->dst_off, (int)threadIdx.x, (int)blockDim.x,
                &cts[ci]);
}

static int run_group(const MgPlan *dp, const int *case_idx, int n_cases,
                     float *d_out, MgCounters *d_cts, int *d_launches) {
    if (n_cases <= 0) return 0;
    const int grid = n_cases;                       /* one workgroup per case */
    mg_case_kernel<<<grid, MG_BLOCK_THREADS>>>(*dp, case_idx, n_cases, d_out, d_cts, d_launches);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        printf("FAIL launch: %s\n", hipGetErrorString(e));
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    char badtok[128];
    if (mg_arg_validate(argc, argv, MG_ARGS, badtok, sizeof(badtok))) {
        printf("FAIL argv: %s\n", badtok);
        usage();
        return 2;
    }
    if (mg_arg_has(argc, argv, "--selftest")) {
        int ok = 1;
        const long long arm_ll = mg_arg_ll(argc, argv, "--arm", -1, &ok);
        if (!ok) { printf("FAIL argv: --arm must be an integer\n"); usage(); return 2; }
        const int only = (int)arm_ll;
        if (only < -1 || only >= MG_ST_ARMS) {
            printf("FAIL argv: --arm %d outside [0,%d)\n", only, MG_ST_ARMS);
            return 2;
        }
        const int bad3 = mg_selftest_all(1, only, 0);
        printf("DRIVER SELFTEST arms=%d failures=%d\n", (only < 0) ? MG_ST_ARMS : 1, bad3);
        if (bad3) printf("RESULT FAIL driver selftest arms=%d\n", (only < 0) ? MG_ST_ARMS : 1);
        else printf("RESULT PASS driver selftest arms=%d\n", (only < 0) ? MG_ST_ARMS : 1);
        return bad3 ? 1 : 0;
    }

    if (mg_arg_has(argc, argv, "--reach-table")) {
        /* the kernel's own claim about which fields a mode can move, printed so gate.py can compare
         * its expectations against the compiled predicate instead of mirroring it in Python */
        int recs = 0;
        for (int m = 0; m < MG_MODE_COUNT; ++m)
            for (int ck = 0; ck < MG_CASE_KIND_COUNT; ++ck)
                for (int f = 0; f < MG_FIELD_COUNT; ++f) {
                    printf("REACH mode=%d case_kind=%d field=%d value=%d\n", m, ck, f,
                           mg_reaches(m, ck, f));
                    recs++;
                }
        printf("REACHCOUNT records=%d\n", recs);
        printf("RESULT PASS reach table records=%d\n", recs);
        return 0;
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
    const int per_slot = mg_arg_has(argc, argv, "--per-slot");
    const int check_only = mg_arg_has(argc, argv, "--check-only");
    if (!argv_ok) { printf("FAIL argv: a numeric flag is not a whole integer\n"); usage(); return 2; }
    if (!plan_path || !slab_path || !expect_path || !out_path || !report_path) {
        printf("FAIL argv: --plan/--slab/--expect/--out/--report are all required\n");
        usage();
        return 2;
    }

    int cu = 0, cr = 0, cc = 0, cb = 0;
    {
        size_t n = 0;
        char *t = read_file(plan_path, &n);
        if (!t) return 1;
        for (size_t i = 0; i < n; ++i)
            if ((i == 0 || t[i - 1] == '\n')) {
                if (t[i] == 'U') cu++;
                else if (t[i] == 'R') cr++;
                else if (t[i] == 'B') cb++;
                else if (t[i] == 'C') cc++;
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
    char *slab = read_file(slab_path, &sb);
    if (!slab) return 1;
    p.slab = slab; p.slab_bytes = sb;
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
    if (check_only) {
        printf("RESULT PASS check plan units=%d cases=%d (no device touched)\n", pu, pc);
        return 0;
    }

    /* ---- device state --------------------------------------------------------------------------- */
    MgUnit *d_units = 0; MgRow *d_rows = 0; MgCase *d_cases = 0; MgBlock *d_blocks = 0;
    char *d_slab = 0; uint32_t *d_expect = 0; float *d_out = 0; MgCounters *d_cts = 0;
    int *d_launches = 0; int *d_caseidx = 0;
    CK(hipMalloc(&d_units, sizeof(MgUnit) * (size_t)pu));
    CK(hipMalloc(&d_rows, sizeof(MgRow) * (size_t)pr));
    CK(hipMalloc(&d_cases, sizeof(MgCase) * (size_t)pc));
    CK(hipMalloc(&d_blocks, sizeof(MgBlock) * (size_t)pb));
    CK(hipMalloc(&d_slab, sb ? sb : 1));
    CK(hipMalloc(&d_expect, (size_t)expect_floats * 4));
    CK(hipMalloc(&d_out, (size_t)(out_floats ? out_floats : 1) * 4));
    CK(hipMalloc(&d_cts, sizeof(MgCounters) * (size_t)(pc ? pc : 1)));
    CK(hipMalloc(&d_launches, sizeof(int)));
    CK(hipMalloc(&d_caseidx, sizeof(int) * (size_t)(pc ? pc : 1)));
    CK(hipMemcpy(d_units, p.units, sizeof(MgUnit) * (size_t)pu, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_rows, p.rows, sizeof(MgRow) * (size_t)pr, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_cases, p.cases, sizeof(MgCase) * (size_t)pc, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_blocks, p.blocks, sizeof(MgBlock) * (size_t)pb, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_slab, slab, sb, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_expect, expect, (size_t)expect_floats * 4, hipMemcpyHostToDevice));
    /* The record buffer is pre-filled with the SAME loud pattern the host arm pre-fills with
     * (mg_nan(), host_probe.cpp), not with zeros. hipMemset can only repeat one byte and the poison
     * is a four-byte pattern, so a zero-filled device buffer would make "this case never ran" and
     * "this case computed zero" read identically on the arm that has no mg_nan() of its own to
     * point at — the exact silence this lane exists to refuse. This is a host->device copy and so
     * does not touch HOST_SYNC, which counts device->host reads. */
    {
        const size_t out_bytes = (size_t)(out_floats ? out_floats : 1) * 4;
        float *poison = (float *)malloc(out_bytes);
        if (!poison) { printf("FAIL oom\n"); return 1; }
        for (size_t i = 0; i < out_bytes / 4; ++i) poison[i] = mg_nan();
        CK(hipMemcpy(d_out, poison, out_bytes, hipMemcpyHostToDevice));
        free(poison);
    }
    CK(hipMemset(d_cts, 0, sizeof(MgCounters) * (size_t)(pc ? pc : 1)));
    CK(hipMemset(d_launches, 0, sizeof(int)));

    MgPlan dp = p;                       /* same shape, device pointers */
    dp.units = d_units; dp.rows = d_rows; dp.cases = d_cases; dp.blocks = d_blocks;
    dp.slab = d_slab; dp.expect = d_expect;

    /* host-visible routing values: none. The ids, weights and pre-weight vectors the combine consumes
     * are read from the slab on the device, and the only device->host copies this program makes are
     * the two read-backs after the last launch. d2h_during counts any copy issued before that point,
     * which is the number that would have to be zero for specification/11-moe-block.md §11.7's claim
     * to hold in this shape. */
    int d2h_total = 0, d2h_during = 0, launches_issued = 0, launches_expected = 0;

    /* d_caseidx is ONE device buffer shared by the grouped and the per-slot path. That is safe for
     * exactly one reason, and it is worth naming because the two paths are otherwise independent
     * (HANDOFF §7.4): every hipMemcpy into it is a synchronous copy on the default stream, and each
     * launch that indexes it is enqueued on that same stream after the copy returns, so no two
     * launches can ever be reading different index arrays at once. If this lane ever moves to
     * streams or to asynchronous copies, this buffer must be per-launch (one allocation or one
     * offset per launch) — not because the arithmetic is wrong today, but because "shared index
     * buffer + concurrency" is the two-self-consistent-layout failure class in
     * pitfalls/40-gates-and-admissions.md.
     *
     * The index arrays are filled from the host case table before each launch, so no routing value
     * is read back from the device at any point (HOST_SYNC line below). */

    for (int b = 0; b < pb; ++b) {
        const int span = p.blocks[b].case_hi - p.blocks[b].case_lo + 1;
        int *idx_exp = (int *)malloc(sizeof(int) * (size_t)(span > 0 ? span : 1));
        int *idx_com = (int *)malloc(sizeof(int) * (size_t)(span > 0 ? span : 1));
        int *sub = (int *)malloc(sizeof(int) * (size_t)(span > 0 ? span : 1));
        if (!idx_exp || !idx_com || !sub) { printf("FAIL oom\n"); return 1; }
        int ne = 0, nc = 0;
        for (int i = p.blocks[b].case_lo; i <= p.blocks[b].case_hi; ++i) {
            if (p.cases[i].kind == MG_CASE_EXPERT) idx_exp[ne++] = i;
            else idx_com[nc++] = i;
        }
        if (per_slot) {
            /* the shape prediction 2 warns about: one launch per (block, slot). Counted, not used. */
            for (int s = 0; s < MG_K; ++s) {
                int ns = 0;
                for (int i = 0; i < ne; ++i)
                    if (p.cases[idx_exp[i]].slot == s) sub[ns++] = idx_exp[i];
                if (!ns) continue;
                CK(hipMemcpy(d_caseidx, sub, sizeof(int) * (size_t)ns, hipMemcpyHostToDevice));
                if (run_group(&dp, d_caseidx, ns, d_out, d_cts, d_launches)) return 1;
                launches_issued++; launches_expected++;
            }
            CK(hipMemcpy(d_caseidx, idx_com, sizeof(int) * (size_t)(nc ? nc : 1), hipMemcpyHostToDevice));
            if (run_group(&dp, d_caseidx, nc, d_out, d_cts, d_launches)) return 1;
            launches_issued++; launches_expected++;
        } else {
            /* the grouped shape: every selected expert of this block, in one launch */
            CK(hipMemcpy(d_caseidx, idx_exp, sizeof(int) * (size_t)(ne ? ne : 1), hipMemcpyHostToDevice));
            if (run_group(&dp, d_caseidx, ne, d_out, d_cts, d_launches)) return 1;
            launches_issued++; launches_expected++;
            CK(hipMemcpy(d_caseidx, idx_com, sizeof(int) * (size_t)(nc ? nc : 1), hipMemcpyHostToDevice));
            if (run_group(&dp, d_caseidx, nc, d_out, d_cts, d_launches)) return 1;
            launches_issued++; launches_expected++;
        }
        free(idx_exp); free(idx_com); free(sub);
    }
    CK(hipDeviceSynchronize());

    MgCounters *cts = (MgCounters *)calloc((size_t)(pc ? pc : 1), sizeof(MgCounters));
    float *out = (float *)malloc((size_t)(out_floats ? out_floats : 1) * 4);
    if (!cts || !out) { printf("FAIL oom\n"); return 1; }
    CK(hipMemcpy(cts, d_cts, sizeof(MgCounters) * (size_t)(pc ? pc : 1), hipMemcpyDeviceToHost));
    d2h_total += 1;
    CK(hipMemcpy(out, d_out, (size_t)out_floats * 4, hipMemcpyDeviceToHost));
    d2h_total += 1;
    int launches_seen = 0;
    CK(hipMemcpy(&launches_seen, d_launches, sizeof(int), hipMemcpyDeviceToHost));
    d2h_total += 1;

    printf("LAUNCHES mode=%s issued=%d expected=%d device_counted=%d groups=%d per_slot=%d\n",
           per_slot ? "per-slot" : "grouped", launches_issued, launches_expected, launches_seen,
           pb, per_slot ? 1 : 0);
    printf("HOST_SYNC d2h_total=%d d2h_during_launches=%d host_values_emitted=%d\n",
           d2h_total, d2h_during, 0);

    if (launches_seen != launches_expected) {
        printf("FAIL the device counted %d launches, the driver issued %d\n", launches_seen,
               launches_expected);
        return 1;
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
        fprintf(rf, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", i, p.cases[i].kind, p.cases[i].mode,
                p.cases[i].block, p.cases[i].row, p.cases[i].slot, cts[i].bad, cts[i].nan,
                cts[i].route_bad, cts[i].route_n);
        sum_bad += cts[i].bad; sum_nan += cts[i].nan;
        sum_route_bad += cts[i].route_bad; sum_route_n += cts[i].route_n;
    }
    fclose(rf);
    printf("REPORT file=%s cases=%d bad=%lld nan=%lld route_bad=%lld route_n=%lld\n",
           report_path, pc, sum_bad, sum_nan, sum_route_bad, sum_route_n);
    if (g_failures) { printf("RESULT FAIL driver failures=%d\n", g_failures); return 1; }
    printf("RESULT PASS device cases=%d launches=%d out_floats=%lld\n", pc, launches_issued, out_floats);
    return 0;
}
