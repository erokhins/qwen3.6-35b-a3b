/* host_probe.cpp — the host twin of the lm_head kernel: the SAME lm_head_kernels.h bodies, built by
 * g++ with no device, no ROCm and no lock.
 *
 * Two jobs:
 *   1. `--selftest`   — the lane's own instruments must be able to fail (shared with the device
 *                        binary through lmh_driver_selftest, so both print one roster);
 *   2. the ARM HOST run — every case in the plan, including the full-vocab ones the device never
 *                        sees, executed on the CPU through lmh_run_case_host. Its words are judged
 *                        against the torch CPU oracle with the derived band, and the device's words
 *                        are judged bit-exact against these.
 *
 * Correctness only: no timer, no warm loop, no burst, no best-of-N, and no ms / GB/s / tok/s figure
 * on any line this file prints.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "lm_head_plan.h"

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
        if (fails) { printf("RESULT FAIL host_probe selftest failures=%d\n", fails); return 1; }
        printf("RESULT PASS host_probe selftest\n");
        return 0;
    }

    const char *plan_path = arg_value(argc, argv, "--plan", NULL);
    const char *slab_path = arg_value(argc, argv, "--slab", NULL);
    const char *hid_path = arg_value(argc, argv, "--hidden", NULL);
    const char *ids_path = arg_value(argc, argv, "--ids", NULL);
    const char *out_path = arg_value(argc, argv, "--out", NULL);
    const char *rep_path = arg_value(argc, argv, "--report", NULL);
    if (!plan_path || !slab_path || !hid_path || !ids_path || !out_path || !rep_path) {
        printf("usage: host_probe --plan P --slab S --hidden H --ids I --out O --report R\n"
               "                  [--expect-tables N] [--expect-cases N]\n"
               "                  [--expect-head-dim N] [--expect-h N] [--expect-vocab N]\n"
               "                  [--check] | --selftest [--work DIR]\n");
        return 2;
    }

    Plan plan;
    int np = 0, nt = 0, nc = 0;
    if (lmh_read_plan(plan_path, &plan, &np, &nt, &nc)) return 1;
    /* One line, all four counts, both files: a parser that read only the header, or only the tables,
     * would otherwise report a plan with geometry and no work to do. */
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
        printf("RESULT FAIL host_probe verify findings=%d\n", bad);
        return 1;
    }

    if (have_flag(argc, argv, "--check")) {
        printf("RESULT PASS check plan tables=%zu cases=%zu (host twin, no device touched)\n",
               plan.tables.size(), plan.cases.size());
        return 0;
    }

    std::vector<float> out((size_t)plan.out_floats, 0.0f);
    std::vector<LmhTrace> traces(plan.cases.size());
    std::vector<long long> launches(plan.cases.size(), 0);
    for (size_t i = 0; i < plan.cases.size(); ++i) lmh_trace_init(&traces[i]);

    LmhCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.tabs = plan.tables.data();
    ctx.n_tables = (int)plan.tables.size();
    ctx.slab = slab.data();
    ctx.hidden = reinterpret_cast<const float *>(hidb.data());
    ctx.ids = reinterpret_cast<const int32_t *>(idsb.data());
    ctx.head_dim = plan.head_dim;
    ctx.out = out.data();
    ctx.out_floats = plan.out_floats;

    long long items = 0, cases_run = 0, skipped = 0;
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const LmhCase &c = plan.cases[i];
        if (have_flag(argc, argv, "--device-only") && !c.on_device) { ++skipped; continue; }
        lmh_run_case_host(ctx, c, &traces[c.idx], lmh_case_out_span(plan, c));
        items += lmh_case_n_items(c);
        ++cases_run;
        launches[c.idx] = 1;                    /* the host arm's "launch" is one host call */
    }
    printf("RUN arm=host cases=%lld skipped=%lld items=%lld out_floats=%lld\n", cases_run, skipped,
           items, plan.out_floats);

    FILE *of = fopen(out_path, "wb");
    if (!of) { printf("FAIL cannot write out %s\n", out_path); return 1; }
    if (!plan.out_floats ||
        fwrite(out.data(), sizeof(float), (size_t)plan.out_floats, of) != (size_t)plan.out_floats) {
        printf("FAIL short write on out %s\n", out_path);
        fclose(of);
        return 1;
    }
    if (fflush(of) || ferror(of)) { printf("FAIL flush of %s\n", out_path); fclose(of); return 1; }
    fclose(of);
    printf("OUT file=%s floats=%lld bytes=%lld\n", out_path, plan.out_floats,
           4LL * plan.out_floats);

    if (lmh_write_report(rep_path, plan, traces.data(), out.data(), launches.data(), 0)) return 1;
    printf("RESULT PASS host arm cases=%zu items=%lld\n", plan.cases.size(), items);
    return 0;
}
