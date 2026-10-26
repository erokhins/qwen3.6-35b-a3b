/* engine/device/m2_rt_main.inc — one main() for both compiled router arms.
 *
 * Included by `m2_rt_host.cpp` (g++) and `m2_rt_arm.hip` (hipcc); the includer defines
 * `M2_RT_COMPILER` and, for the HIP arm, `M2_RT_HAVE_HIP` plus `m2_rt_device_run()`. The arithmetic
 * is in `m2_rt_kernels.h`, the plan checker in `m2_rt_io.h`; what is here is the command line, the
 * order of acts and the report — the same report shape the shared-expert arm prints, so the driver
 * parses both with one parser.
 *
 * NO TIMER. Nothing here reads a clock and no rate appears in any line.
 */

static void m2_rt_usage(void) {
    fprintf(stderr,
            "usage: --plan P --slab S --out O [--device] [--fault NAME] [--check-only]\n"
            "       arm=%s compiler=%s device_path=%s faults=none|tie-forge|skip-renorm|skip-topk\n",
            M2_RT_ARM_NAME, M2_RT_COMPILER,
#ifdef M2_RT_HAVE_HIP
            "yes");
#else
            "no (g++ twin: no HIP symbol exists in this binary)");
#endif
}

static int m2_rt_host_run(const struct M2RtPlan *plan, const uint8_t *slab, float *out,
                          struct M2RtCounters *ctr, struct M2RtRunStats *st, char *err, size_t errn) {
    int bad = 0;
    (void)err;
    (void)errn;
    memset(st, 0, sizeof(*st));
    st->path = "host_cpubody";
    m2_rt_run(plan, slab, out, ctr, &bad);
    st->bad_index = bad;
    st->hip_calls = 0;                 /* a body executed on the CPU issued zero device calls */
    st->launches = 0;
    st->allocs = 0;
    st->bytes_in = 0;
    st->bytes_out = 0;
    return 0;
}

int main(int argc, char **argv) {
    const char *plan_path = 0, *slab_path = 0, *out_path = 0, *fault_cli = 0;
    int want_device = 0, check_only = 0, i, rc = 0;
    struct M2RtPlanFile pf;
    struct M2RtPlan plan;
    struct M2RtCounters ctr;
    struct M2RtRunStats st;
    char err[512], fields[512];
    uint8_t *slab = 0;
    float *out = 0;
    long slab_bytes = 0;

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--plan") && i + 1 < argc) plan_path = argv[++i];
        else if (!strcmp(argv[i], "--slab") && i + 1 < argc) slab_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--fault") && i + 1 < argc) fault_cli = argv[++i];
        else if (!strcmp(argv[i], "--device")) want_device = 1;
        else if (!strcmp(argv[i], "--check-only")) check_only = 1;
        else {
            fprintf(stderr, "FAIL tag=arg-unknown argv=%s\n", argv[i]);
            m2_rt_usage();
            return 2;
        }
    }
    if (!plan_path || !slab_path || !out_path) {
        fprintf(stderr, "FAIL tag=arg-missing note=--plan, --slab and --out are all required\n");
        m2_rt_usage();
        return 2;
    }
    err[0] = '\0';
    if (m2_rt_read_plan(plan_path, &pf, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_RT_ARM_NAME);
        return 1;
    }
    if (fault_cli && *fault_cli) snprintf(pf.fault, sizeof(pf.fault), "%s", fault_cli);
    if (!strcmp(pf.fault, "tie-forge")) pf.tie_forge = 1;
    else if (!strcmp(pf.fault, "skip-renorm")) snprintf(pf.skip_word, sizeof(pf.skip_word), "renorm");
    else if (!strcmp(pf.fault, "skip-topk")) snprintf(pf.skip_word, sizeof(pf.skip_word), "topk");
    else if (strcmp(pf.fault, "none")) {
        fprintf(stderr, "FAIL tag=fault-unknown note=fault `%s` is not none/tie-forge/skip-renorm/"
                        "skip-topk; an injection this arm does not implement cannot be a control\n",
                pf.fault);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_RT_ARM_NAME);
        return 1;
    }
    slab = m2_rt_read_file(slab_path, &slab_bytes, err, sizeof(err));
    if (!slab) {
        fprintf(stderr, "FAIL tag=slab-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_RT_ARM_NAME);
        return 1;
    }
    if (pf.slab_bytes && pf.slab_bytes != slab_bytes) {
        fprintf(stderr, "FAIL tag=slab-size note=%s is %ld B, the plan says %ld B; the arm refuses to "
                        "run a plan against a slab that is not the one it was written for\n",
                slab_path, slab_bytes, pf.slab_bytes);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_RT_ARM_NAME);
        free(slab);
        return 1;
    }
    if (m2_rt_plan_check(&pf, slab_bytes, &plan, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-rejected note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_RT_ARM_NAME);
        free(slab);
        return 1;
    }
    printf("HDR arm=%s compiler=%s rows=%d cols=%d top_k=%d floats=%d want_reads_w=%d want_reads_x=%d "
           "want_items=%d/%d\n",
           M2_RT_ARM_NAME, M2_RT_COMPILER, plan.rows, plan.cols, plan.top_k, M2_RT_FLOATS,
           plan.rows * plan.cols, plan.rows * plan.cols, plan.rows, 1);
    printf("PLAN path=%s version=%d slab_bytes=%ld off_w=%ld n_w=%ld off_x=%ld n_x=%ld elt=%d "
           "mode=%s cast=%s skip=%s tie_forge=%d fault=%s\n",
           plan_path, pf.version, slab_bytes, pf.off_w, pf.n_w, pf.off_x, pf.n_x, pf.elt,
           pf.mode_word, pf.cast_word, pf.skip_word, pf.tie_forge, pf.fault);
    if (check_only) {
        printf("RESULT arm=%s verdict=OK note=plan and slab verified; nothing computed no timing "
               "taken\n", M2_RT_ARM_NAME);
        free(slab);
        return 0;
    }
    out = (float *)malloc(sizeof(float) * (size_t)M2_RT_FLOATS);
    if (!out) {
        fprintf(stderr, "FAIL tag=oom note=record alloc failed\n");
        free(slab);
        return 1;
    }
    memset(out, 0, sizeof(float) * (size_t)M2_RT_FLOATS);
#ifdef M2_RT_HAVE_HIP
    if (want_device) {
        rc = m2_rt_device_run(&plan, slab, slab_bytes, out, &ctr, &st, err, sizeof(err));
        if (rc) {
            fprintf(stderr, "FAIL tag=device-run-failed note=%s\n", err);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n",
                    M2_RT_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
    } else
#endif
    {
        if (want_device) {
            fprintf(stderr, "FAIL tag=device-path-absent-in-this-binary note=%s was compiled by %s "
                            "and contains no HIP symbol; refusing to answer --device with a host run\n",
                    M2_RT_ARM_NAME, M2_RT_COMPILER);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n",
                    M2_RT_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (m2_rt_host_run(&plan, slab, out, &ctr, &st, err, sizeof(err))) {
            fprintf(stderr, "FAIL tag=host-run-failed note=%s\n", err);
            free(out);
            free(slab);
            return 1;
        }
    }
    m2_rt_field_report(out, fields, sizeof(fields));
#ifndef M2_RT_HAVE_HIP
    (void)rc;                            /* only the device branch has a return code to carry */
#endif
    if (m2_rt_write_f32(out_path, out, M2_RT_FLOATS, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=out-unsafe note=%s\n", err);
        free(out);
        free(slab);
        return 1;
    }
    printf("RUN arm=%s path=%s hip_calls=%d launches=%d allocs=%d bytes_to_device=%ld "
           "bytes_from_device=%ld slab_bytes=%ld out_floats=%d reads_w=%u reads_x=%u items=%u/%u "
           "bad_index=%d nan_fields=\"%s\"\n",
           M2_RT_ARM_NAME, st.path, st.hip_calls, st.launches, st.allocs,
           (long)st.bytes_in, (long)st.bytes_out, slab_bytes, M2_RT_FLOATS,
           ctr.reads_w, ctr.reads_x, ctr.items1, ctr.items2, st.bad_index, fields);
    printf("WROTE path=%s bytes=%d\n", out_path, (int)(sizeof(float) * M2_RT_FLOATS));
    printf("RESULT arm=%s verdict=OK no timing taken\n", M2_RT_ARM_NAME);
    free(out);
    free(slab);
    return 0;
}
