/* lm_head.hip — the device implementation under test: the un-tied output projection on gfx1151.
 *
 * One launch per case, one item per (vocab row, k-slice lane), and every batch row computed from
 * each weight element as it is loaded — see lm_head_kernels.h §the item. The plan, the verifier, the
 * report writer and the arithmetic all live in headers shared with host_probe.cpp, so a green
 * device-vs-host comparison is a comparison of compilers and not of two drivers.
 *
 * CORRECTNESS ONLY. This driver takes no timer, runs no warm loop, no burst and no best-of-N, and
 * prints no ms / GB/s / tok/s figure (`pitfalls/00-measurement.md`; AGENTS.md §"GPU execution is
 * serialized"). The roofline numbers for this kernel belong to the main agent's exclusive window.
 *
 * Build:  ./build.sh            (hipcc, gfx1151, objects under build/)
 *         ./build.sh --check    (compile only; touches no device)
 * Run:    ./gate               (the wrapper alone owns the machine-wide GPU lock, then gate.py,
 *                               then this binary with --device-only)
 *         ./lm_head --selftest  host-only driver witnesses, no device
 *         ./lm_head --check     parse + verify the plan, no device
 */
#include <hip/hip_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "lm_head_plan.h"

static int g_hip_failures = 0;

#define CK(call)                                                                \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            ++g_hip_failures;                                                   \
            printf("FAIL hip: %s = %s (%s)\n", #call, hipGetErrorString(_e),    \
                   strerror(errno));                                            \
            return 2;                                                           \
        }                                                                       \
    } while (0)

/* grid-stride over the case's items; blockDim.x is a multiple of the case's lane count, so the
 * LMH_SPLIT lanes of one item are always consecutive threads inside one block and the combine reads
 * them out of shared memory in index order. __syncthreads() sits OUTSIDE every `if (g < n_items)`,
 * because a divergent barrier on this box presents as a busy stream sync with no output and no exit
 * (`pitfalls/50` §"Device faults present as hangs, not crashes"). */
__global__ void lm_head_kernel(const LmhCtx ctx, const LmhCase c, LmhTrace *tr,
                               long long n_items, int nlanes) {
    __shared__ float sp[LMH_BLOCK * LMH_MAX_BATCH];
    const int bs = (int)blockDim.x;
    for (long long base = (long long)blockIdx.x * bs; base < n_items;
         base += (long long)gridDim.x * bs) {
        const int g = (int)(base + threadIdx.x);
        const int active = (g < n_items);
        float acc[LMH_MAX_BATCH];
        for (int b = 0; b < LMH_MAX_BATCH; ++b) acc[b] = 0.0f;
        if (active) {
            const int lane = g % nlanes;
            const int item = g / nlanes;
            lmh_item(&ctx, &c, item, lane, nlanes, tr, acc);
        }
        for (int b = 0; b < LMH_MAX_BATCH; ++b) sp[threadIdx.x * LMH_MAX_BATCH + b] = acc[b];
        __syncthreads();
        if (active && (g % nlanes) == 0) {
            const int item = g / nlanes;
            float pp[LMH_SPLIT];
            for (int b = 0; b < c.bcount; ++b) {
                for (int i = 0; i < nlanes; ++i) pp[i] = sp[(threadIdx.x + i) * LMH_MAX_BATCH + b];
                lmh_store(&ctx, &c, b, item, lmh_combine(pp, nlanes), tr);
            }
        }
        __syncthreads();
    }
}

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

static long long read_long(const char *s, long long dflt) {
    if (!s) return dflt;
    char *e = 0;
    const long long v = strtoll(s, &e, 10);
    return (e && *e) ? dflt : v;
}

int main(int argc, char **argv) {
    if (have_flag(argc, argv, "--selftest")) {
        const char *work = arg_value(argc, argv, "--work", ".");
        const int fails = lmh_driver_selftest(work);
        if (fails) { printf("RESULT FAIL lm_head selftest failures=%d\n", fails); return 1; }
        printf("RESULT PASS lm_head selftest\n");
        return 0;
    }

    const char *plan_path = arg_value(argc, argv, "--plan", NULL);
    const char *slab_path = arg_value(argc, argv, "--slab", NULL);
    const char *hid_path = arg_value(argc, argv, "--hidden", NULL);
    const char *ids_path = arg_value(argc, argv, "--ids", NULL);
    const char *out_path = arg_value(argc, argv, "--out", NULL);
    const char *rep_path = arg_value(argc, argv, "--report", NULL);
    if (!plan_path || !slab_path || !hid_path || !ids_path || !out_path || !rep_path) {
        printf("usage: lm_head --plan P --slab S --hidden H --ids I --out O --report R\n"
               "                 [--expect-tables N] [--expect-cases N]\n"
               "                 [--expect-head-dim N] [--expect-h N] [--expect-vocab N]\n"
               "                 [--device-only] [--check] | --selftest [--work DIR]\n");
        return 2;
    }

    Plan plan;
    int np = 0, nt = 0, nc = 0;
    if (lmh_read_plan(plan_path, &plan, &np, &nt, &nc)) return 1;
    printf("PLAN file=%s p_records=%d tables_parsed=%d cases_parsed=%d h=%d head_dim=%d vocab=%d "
           "n_hidden=%d ids=%d out_floats=%lld slab_bytes=%lld\n",
           plan_path, np, nt, nc, plan.h, plan.head_dim, plan.vocab, plan.n_hidden, plan.ids_count,
           plan.out_floats, plan.slab_bytes);
    if (np != 1) {
        printf("FAIL plan has %d P records, expected exactly 1\n", np);
        return 1;
    }
    if (nt != (int)plan.tables.size() || nc != (int)plan.cases.size()) {
        printf("FAIL plan counted %d/%d rows but holds %zu/%zu records\n", nt, nc,
               plan.tables.size(), plan.cases.size());
        return 1;
    }

    std::vector<uint8_t> slab, hidb, idsb;
    if (lmh_load_file(slab_path, &slab)) return 1;
    if (lmh_load_file(hid_path, &hidb)) return 1;
    if (lmh_load_file(ids_path, &idsb)) return 1;
    const long long hidden_floats = (long long)hidb.size() / 4;
    const int ids_count = (int)(idsb.size() / 4);

    const int bad = lmh_verify_plan(plan, (int)read_long(arg_value(argc, argv, "--expect-tables", NULL), -1),
                                    (int)read_long(arg_value(argc, argv, "--expect-cases", NULL), -1),
                                    (long long)slab.size(), hidden_floats, ids_count,
                                    (int)read_long(arg_value(argc, argv, "--expect-head-dim", NULL), -1),
                                    (int)read_long(arg_value(argc, argv, "--expect-h", NULL), -1),
                                    (int)read_long(arg_value(argc, argv, "--expect-vocab", NULL), -1));
    if (bad) {
        printf("RESULT FAIL lm_head verify findings=%d\n", bad);
        return 1;
    }

    if (have_flag(argc, argv, "--check")) {
        printf("RESULT PASS check plan tables=%zu cases=%zu (no device touched)\n",
               plan.tables.size(), plan.cases.size());
        return 0;
    }

    /* ---- device work -------------------------------------------------------------------------- */
    const int n_dev = lmh_count_device_cases(plan);
    std::vector<float> out((size_t)plan.out_floats, 0.0f);
    std::vector<LmhTrace> traces(plan.cases.size());
    std::vector<long long> launches(plan.cases.size(), 0);
    for (size_t i = 0; i < plan.cases.size(); ++i) lmh_trace_init(&traces[i]);

    LmhTable *d_tabs = NULL;
    uint8_t *d_slab = NULL;
    float *d_hid = NULL, *d_out = NULL;
    int32_t *d_ids = NULL;
    LmhTrace *d_tr = NULL;
    CK(hipMalloc(&d_tabs, sizeof(LmhTable) * plan.tables.size()));
    CK(hipMalloc(&d_slab, slab.empty() ? 1 : slab.size()));
    CK(hipMalloc(&d_hid, hidb.empty() ? 1 : hidb.size()));
    CK(hipMalloc(&d_ids, idsb.empty() ? 1 : idsb.size()));
    CK(hipMalloc(&d_out, sizeof(float) * (size_t)(plan.out_floats > 0 ? plan.out_floats : 1)));
    CK(hipMalloc(&d_tr, sizeof(LmhTrace) * (plan.cases.empty() ? 1 : plan.cases.size())));
    CK(hipMemcpy(d_tabs, plan.tables.data(), sizeof(LmhTable) * plan.tables.size(),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_slab, slab.data(), slab.empty() ? 1 : slab.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_hid, hidb.data(), hidb.empty() ? 1 : hidb.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_ids, idsb.data(), idsb.empty() ? 1 : idsb.size(), hipMemcpyHostToDevice));
    CK(hipMemset(d_out, 0, sizeof(float) * (size_t)(plan.out_floats > 0 ? plan.out_floats : 1)));
    CK(hipMemset(d_tr, 0, sizeof(LmhTrace) * (plan.cases.empty() ? 1 : plan.cases.size())));

    LmhCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.tabs = d_tabs;
    ctx.n_tables = (int)plan.tables.size();
    ctx.slab = d_slab;
    ctx.hidden = d_hid;
    ctx.ids = d_ids;
    ctx.head_dim = plan.head_dim;
    ctx.out = d_out;
    ctx.out_floats = plan.out_floats;

    long long items = 0, cases_run = 0, skipped = 0;
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const LmhCase &c = plan.cases[i];
        if (have_flag(argc, argv, "--device-only") && !c.on_device) { ++skipped; continue; }
        const int nlanes = lmh_case_nlanes(c);
        const long long n_items = lmh_case_n_items(c);
        if (n_items == 0) { printf("FAIL case %d has 0 items\n", c.idx); return 1; }
        long long grid = (n_items + LMH_BLOCK - 1) / LMH_BLOCK;
        if (grid > 4096) grid = 4096;
        lm_head_kernel<<<(int)grid, LMH_BLOCK>>>(ctx, c, d_tr + c.idx, n_items, nlanes);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        items += n_items;
        ++cases_run;
        launches[c.idx] = grid;
    }
    CK(hipMemcpy(traces.data(), d_tr, sizeof(LmhTrace) * (plan.cases.empty() ? 1 : plan.cases.size()),
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(out.data(), d_out, sizeof(float) * (size_t)(plan.out_floats ? plan.out_floats : 1),
                 hipMemcpyDeviceToHost));
    CK(hipFree(d_tabs));
    CK(hipFree(d_slab));
    CK(hipFree(d_hid));
    CK(hipFree(d_ids));
    CK(hipFree(d_out));
    CK(hipFree(d_tr));
    printf("RUN arm=device cases=%lld skipped=%lld items=%lld planned=%d out_floats=%lld\n",
           cases_run, skipped, items, n_dev, plan.out_floats);

    FILE *of = fopen(out_path, "wb");
    if (!of) { printf("FAIL cannot write out %s\n", out_path); return 1; }
    if (plan.out_floats &&
        fwrite(out.data(), sizeof(float), (size_t)plan.out_floats, of) != (size_t)plan.out_floats) {
        printf("FAIL short write on out %s\n", out_path);
        fclose(of);
        return 1;
    }
    if (fflush(of) || ferror(of)) { printf("FAIL flush of %s\n", out_path); fclose(of); return 1; }
    fclose(of);
    printf("OUT file=%s floats=%lld bytes=%lld\n", out_path, plan.out_floats,
           4LL * plan.out_floats);

    if (lmh_write_report(rep_path, plan, traces.data(), out.data(), launches.data(), 1)) return 1;
    if (g_hip_failures) {
        printf("RESULT FAIL device hip-calls=%d\n", g_hip_failures);
        return 2;
    }
    printf("RESULT PASS device arm cases=%zu items=%lld\n", plan.cases.size(), items);
    return 0;
}
