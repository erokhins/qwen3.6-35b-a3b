/* engine/device/m2_d_main.inc — one main() for both compiled GDN recurrent arms.
 *
 * Included by `m2_d_host.cpp` (g++) and `m2_d_arm.hip` (hipcc); the includer defines `M2D_ARM_NAME`,
 * `M2D_COMPILER` and, for the HIP arm, `M2D_HAVE_HIP` plus `m2d_device_run()`. The arithmetic is in
 * `m2_d_kernels.h`, the plan parser and its refusals in `m2_d_io.h`; what is here is the command line,
 * the order of acts and the report — the same report shape the shared-expert, router and gather arms
 * print, so the driver parses all four with one parser (`engine/device_arm.py::run_arm`).
 *
 * THE FORM IS PRINTED ON EVERY LINE THAT MATTERS. `HDR form=recurrent form_body=` names the program
 * this binary implements, because the GDN band licences rows per form and a driver that could not tell
 * the two programs apart from the outside could credit a recurrent row to chunked arithmetic. The
 * driver re-checks the arm's declared form against the licence pair it credits
 * (`engine/block_device.py`'s `gdn-form-not-merged`).
 *
 * NO TIMER. Nothing here reads a clock and no rate appears in any line (`pitfalls/00`).
 */

static void m2d_usage(void) {
    fprintf(stderr,
            "usage: --plan P --slab S --out O [--device] [--fault NAME] [--check-only]\n"
            "       arm=%s compiler=%s form=%s device_path=%s\n"
            "       faults=none|skip-carry|skip-out|zero-state|alloc-oversize\n",
            M2D_ARM_NAME, M2D_COMPILER, M2D_FORM_NAME(M2D_FORM_RECURRENT),
#ifdef M2D_HAVE_HIP
            "yes");
#else
            "no (g++ twin: no HIP symbol exists in this binary)");
#endif
}

static int m2d_host_run(const struct M2DPlan *plan, const uint8_t *slab, float *out,
                        struct M2DCounters *ctr, struct M2DRunStats *st, char *err, size_t errn) {
    struct M2DScratch sc;
    float *scratch;
    (void)err;
    (void)errn;
    memset(st, 0, sizeof(*st));
    st->path = "host_cpubody";
    scratch = (float *)calloc((size_t)m2d_scratch_floats(), sizeof(float));
    if (!scratch) { snprintf(err, errn, "scratch alloc failed"); return 1; }
    sc.v_w = scratch;
    sc.kv = scratch + GDN_H_V * GDN_D_V;
    sc.delta = sc.kv + GDN_H_V * GDN_D_V;
    sc.beta = sc.delta + GDN_H_V * GDN_D_V;
    m2d_run(plan, slab, out, &sc, ctr);
    free(scratch);
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
    struct M2DPlanFile pf;
    struct M2DPlan plan;
    struct M2DCounters ctr;
    struct M2DRunStats st;
    char err[1024], fields[512], census[1024];
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
            m2d_usage();
            return 2;
        }
    }
    if (!plan_path || !slab_path || !out_path) {
        fprintf(stderr, "FAIL tag=arg-missing note=--plan, --slab and --out are all required\n");
        m2d_usage();
        return 2;
    }
    err[0] = '\0';
    if (m2d_read_plan(plan_path, &pf, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
        return 1;
    }
    if (fault_cli && *fault_cli) snprintf(pf.fault_word, sizeof(pf.fault_word), "%s", fault_cli);
    slab = m2d_read_file(slab_path, &slab_bytes, err, sizeof(err));
    if (!slab) {
        fprintf(stderr, "FAIL tag=slab-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
        return 1;
    }
    if (pf.slab_bytes && pf.slab_bytes != slab_bytes) {
        fprintf(stderr, "FAIL tag=slab-size note=%s is %ld B, the plan says %ld B; the arm refuses to "
                        "run a plan against a slab that is not the one it was written for\n",
                slab_path, slab_bytes, pf.slab_bytes);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
        free(slab);
        return 1;
    }
    if (m2d_plan_check(&pf, slab_bytes, &plan, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-rejected note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
        free(slab);
        return 1;
    }
    printf("HDR arm=%s compiler=%s form=%s form_body=cached-recurrent-decode-step "
           "h_k=%d h_v=%d d_k=%d d_v=%d H=%d floats=%d stages=%d want_items=%ld "
           "want_state_words=%ld want_out_elems=%ld mode=%s mode_fields=%s\n",
           M2D_ARM_NAME, M2D_COMPILER, M2D_FORM_NAME(plan.form), GDN_H_K, GDN_H_V, GDN_D_K, GDN_D_V,
           GDN_H, M2D_FLOATS, M2D_FIELD_COUNT, m2d_items_total(), m2d_want_state_words(),
           m2d_want_out_elems(), pf.mode_word, m2d_mode_fields(plan.mode));
    printf("PLAN path=%s version=%d slab_bytes=%ld form=%s mode=%s fault=%s state_off=%ld "
           "state_bytes=%ld read_off=%ld read_len=%d\n",
           plan_path, pf.version, slab_bytes, pf.form_word, pf.mode_word, pf.fault_word,
           plan.off_in[M2D_F_STATE], plan.bytes_in[M2D_F_STATE], plan.read_off, plan.read_len);
    if (check_only) {
        printf("RESULT arm=%s verdict=OK note=plan and slab verified; nothing computed no timing "
               "taken\n", M2D_ARM_NAME);
        free(slab);
        return 0;
    }
    out = (float *)malloc(sizeof(float) * (size_t)M2D_FLOATS);
    if (!out) {
        fprintf(stderr, "FAIL tag=oom note=record alloc failed\n");
        free(slab);
        return 1;
    }
    memset(out, 0, sizeof(float) * (size_t)M2D_FLOATS);
    memset(&ctr, 0, sizeof(ctr));
#ifdef M2D_HAVE_HIP
    if (want_device) {
        rc = m2d_device_run(&plan, slab, slab_bytes, out, &ctr, &st, err, sizeof(err));
        if (rc) {
            fprintf(stderr, "FAIL tag=device-run-failed note=%s\n", err);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
    } else
#endif
    {
        if (want_device) {
            fprintf(stderr, "FAIL tag=device-path-absent-in-this-binary note=%s was compiled by %s and "
                            "contains no HIP symbol; refusing to answer --device with a host run\n",
                    M2D_ARM_NAME, M2D_COMPILER);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (plan.fault == M2D_FAULT_ALLOC_OVER) {
            fprintf(stderr, "FAIL tag=fault-device-only note=alloc-oversize is a device act; this "
                            "binary has no device path, so the leg cannot execute here and is not "
                            "faked\n");
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2D_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (m2d_host_run(&plan, slab, out, &ctr, &st, err, sizeof(err))) {
            fprintf(stderr, "FAIL tag=host-run-failed note=%s\n", err);
            free(out);
            free(slab);
            return 1;
        }
    }
    m2d_field_report(out, fields, sizeof(fields));
    m2d_census_report(&ctr, census, sizeof(census));
    if (plan.read_off >= 0 && plan.read_len > 0) {
        /* The read-back leg exists to bind what the DEVICE holds to what the file holds, so it prints
         * only where a device actually copied bytes back. On the host body the driver binds file bytes
         * to slab bytes by digest, and printing a host "read-back" as if it were one would be exactly
         * the substitution this lane refuses (`m2_g_main.inc` draws the same line). */
        if (st.read_hex[0])
            printf("DREADBACK path=%s off=%ld bytes=%d hex=%s\n", st.path, plan.read_off,
                   plan.read_len, st.read_hex);
        else if (st.path && strcmp(st.path, "device") == 0)
            printf("DREADBACK path=%s off=%ld bytes=%d hex=ABSENT\n", st.path, plan.read_off,
                   plan.read_len);
    }
    if (m2d_write_f32(out_path, out, M2D_FLOATS, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=out-unsafe note=%s\n", err);
        free(out);
        free(slab);
        return 1;
    }
    printf("RUN arm=%s path=%s form=%s hip_calls=%d launches=%d allocs=%d bytes_to_device=%ld "
           "bytes_from_device=%ld slab_bytes=%ld out_floats=%d nan_fields=\"%s\" census=\"%s\"\n",
           M2D_ARM_NAME, st.path, M2D_FORM_NAME(plan.form), st.hip_calls, st.launches, st.allocs,
           (long)st.bytes_in, (long)st.bytes_out, slab_bytes, M2D_FLOATS, fields, census);
    printf("WROTE path=%s bytes=%d\n", out_path, (int)(sizeof(float) * M2D_FLOATS));
    printf("RESULT arm=%s verdict=OK no timing taken\n", M2D_ARM_NAME);
#ifndef M2D_HAVE_HIP
    (void)rc;                            /* only the device branch has a return code to carry */
#endif
    free(out);
    free(slab);
    return 0;
}
