/* engine/device/m2_se_main.inc — one main() for both compiled arms.
 *
 * Included by `m2_se_host.cpp` (g++) and `m2_se_arm.hip` (hipcc). The includer defines
 * `M2_SE_ARM_NAME`, `M2_SE_COMPILER` and, for the HIP arm, `M2_SE_HAVE_HIP` plus
 * `m2_se_device_run()`. Everything that computes is in `m2_se_kernels.h`; everything that reads a
 * file is in `m2_se_io.h`; what is here is the command line, the order of acts, and the report.
 *
 * THE REPORT IS THE CONTRACT the driver parses, so every record is `key=value` with no spaces in a
 * value except the quoted field census, and the terminal line is one of:
 *     RESULT arm=<a> verdict=OK …  /  RESULT arm=<a> verdict=FAIL findings=<n> …
 * A non-zero exit and a FAIL verdict are the same event: the driver treats either as fatal and
 * prints no stage rows, which is what makes "falls back to the host twin silently" unimplementable.
 *
 * NO TIMER. Nothing here reads a clock, and no `ms`/`t/s`/`GB/s` appears in any line this prints.
 */

static void m2_se_usage(void) {
    fprintf(stderr,
            "usage: --plan P --slab S --out O [--device] [--fault NAME] [--check-only]\n"
            "       arm=%s compiler=%s device_path=%s\n",
            M2_SE_ARM_NAME, M2_SE_COMPILER,
#ifdef M2_SE_HAVE_HIP
            "yes");
#else
            "no (g++ twin: no HIP symbol exists in this binary)");
#endif
}

static int m2_se_host_run(const struct M2SePlan *plan, const uint8_t *slab, long slab_bytes,
                          float *out, struct M2SeCounters *ctr, struct M2SeRunStats *st,
                          char *err, size_t errn) {
    (void)slab_bytes;
    (void)err;
    (void)errn;
    memset(ctr, 0, sizeof(*ctr));
    memset(st, 0, sizeof(*st));
    st->path = "host_cpubody";
    m2_se_run(plan, slab, out, ctr);
    st->hip_calls = 0;                 /* a body executed on the CPU issued zero device calls */
    st->launches = 0;
    st->allocs = 0;
    st->bytes_in = 0;
    st->bytes_out = 0;
    return 0;
}

int main(int argc, char **argv) {
    const char *plan_path = 0, *slab_path = 0, *out_path = 0;
    const char *fault_cli = 0;
    int want_device = 0, check_only = 0;
    struct M2SePlanFile pf;
    struct M2SePlan plan;
    struct M2SeCounters ctr;
    struct M2SeRunStats st;
    char err[512];
    char fields[256];
    uint8_t *slab = 0;
    float *out = 0;
    long slab_bytes = 0;
    int i;
    int rc = 0;

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--plan") && i + 1 < argc) plan_path = argv[++i];
        else if (!strcmp(argv[i], "--slab") && i + 1 < argc) slab_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--fault") && i + 1 < argc) fault_cli = argv[++i];
        else if (!strcmp(argv[i], "--device")) want_device = 1;
        else if (!strcmp(argv[i], "--check-only")) check_only = 1;
        else {
            fprintf(stderr, "FAIL tag=arg-unknown argv=%s\n", argv[i]);
            m2_se_usage();
            return 2;
        }
    }
    if (!plan_path || !slab_path || !out_path) {
        fprintf(stderr, "FAIL tag=arg-missing note=--plan, --slab and --out are all required\n");
        m2_se_usage();
        return 2;
    }

    err[0] = '\0';
    if (m2_se_read_plan(plan_path, &pf, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=%s note=%s\n", "plan-unreadable", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_SE_ARM_NAME);
        return 1;
    }
    if (fault_cli && *fault_cli && strcmp(pf.fault, "none")) {
        snprintf(err, sizeof(err), "plan-and-cli-fault: the plan names fault=%s and --fault names %s; "
                                   "one injection per run, or the census judges an unknown build",
                 pf.fault, fault_cli);
        fprintf(stderr, "FAIL tag=fault-doubled note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_SE_ARM_NAME);
        return 1;
    }
    if (fault_cli && *fault_cli) snprintf(pf.fault, sizeof(pf.fault), "%s", fault_cli);
    pf.skip_stage3 = !strcmp(pf.fault, "skip-stage3");
    pf.alloc_oversize = !strcmp(pf.fault, "alloc-oversize");
    slab = m2_se_read_file(slab_path, &slab_bytes, err, sizeof(err));
    if (!slab) {
        fprintf(stderr, "FAIL tag=%s note=%s\n", "slab-unreadable", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_SE_ARM_NAME);
        return 1;
    }
    if (pf.slab_bytes != slab_bytes) {
        fprintf(stderr, "FAIL tag=slab-size note=%s is %ld B, the plan says %ld B; the arm refuses to "
                        "run a plan against a slab that is not the one it was written for\n",
                slab_path, slab_bytes, pf.slab_bytes);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_SE_ARM_NAME);
        free(slab);
        return 1;
    }
    if (m2_se_plan_check(&pf, slab_bytes, &plan, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=%s note=%s\n", "plan-rejected", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2_SE_ARM_NAME);
        free(slab);
        return 1;
    }

    printf("HDR arm=%s compiler=%s se_h=%d se_i=%d case_floats=%d "
           "want_reads_fused=%d want_reads_down=%d want_reads_sg=%d want_items=%d/%d/%d\n",
           M2_SE_ARM_NAME, M2_SE_COMPILER, SE_H, SE_I, M2_SE_FLOATS,
           M2_SE_WANT_READS_FUSED, M2_SE_WANT_READS_DOWN, M2_SE_WANT_READS_SG,
           M2_SE_ITEMS1, M2_SE_ITEMS2, M2_SE_ITEMS3);
    printf("PLAN path=%s version=%d slab_bytes=%ld off_fused=%ld off_down=%ld off_sg=%ld off_x=%ld "
           "mode=%d sgate_form=%d cast_form=%d fault=%s\n",
           plan_path, pf.version, pf.slab_bytes, pf.off_fused, pf.off_down, pf.off_sg, pf.off_x,
           pf.mode, pf.sgate_form, pf.cast_form, pf.fault);
    if (check_only) {
        printf("RESULT arm=%s verdict=OK note=plan and slab verified; nothing computed no timing taken\n",
               M2_SE_ARM_NAME);
        free(slab);
        return 0;
    }

    out = (float *)malloc(sizeof(float) * (size_t)M2_SE_FLOATS);
    if (!out) {
        fprintf(stderr, "FAIL tag=oom note=record alloc failed\n");
        free(slab);
        return 1;
    }
    memset(out, 0, sizeof(float) * (size_t)M2_SE_FLOATS);

#ifdef M2_SE_HAVE_HIP
    if (want_device) {
        rc = m2_se_device_run(&plan, slab, slab_bytes, out, &ctr, &st, err, sizeof(err));
        if (rc) {
            fprintf(stderr, "FAIL tag=%s note=%s\n", "device-run-failed", err);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n",
                    M2_SE_ARM_NAME);
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
                    M2_SE_ARM_NAME, M2_SE_COMPILER);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n",
                    M2_SE_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (m2_se_host_run(&plan, slab, slab_bytes, out, &ctr, &st, err, sizeof(err))) {
            fprintf(stderr, "FAIL tag=host-run-failed note=%s\n", err);
            free(out);
            free(slab);
            return 1;
        }
    }

    m2_se_field_report(out, fields, sizeof(fields));
    if (m2_se_write_f32(out_path, out, M2_SE_FLOATS, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=%s note=%s\n", "out-unsafe", err);
        free(out);
        free(slab);
        return 1;
    }
    printf("RUN arm=%s path=%s hip_calls=%d launches=%d allocs=%d bytes_to_device=%ld "
           "bytes_from_device=%ld slab_bytes=%ld out_floats=%d reads_fused=%u reads_down=%u "
           "reads_sg=%u items=%u/%u/%u nan_fields=\"%s\"\n",
           M2_SE_ARM_NAME, st.path, st.hip_calls, st.launches, st.allocs,
           (long)st.bytes_in, (long)st.bytes_out, slab_bytes, M2_SE_FLOATS,
           ctr.reads_fused, ctr.reads_down, ctr.reads_sg,
           ctr.items1, ctr.items2, ctr.items3, fields);
    printf("WROTE path=%s bytes=%d\n", out_path, (int)(sizeof(float) * M2_SE_FLOATS));
    printf("RESULT arm=%s verdict=OK no timing taken\n", M2_SE_ARM_NAME);
    free(out);
    free(slab);
    return 0;
}
