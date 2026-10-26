/* engine/device/m2_g_main.inc — one main() for both compiled gather arms.
 *
 * Included by `m2_g_host.cpp` (g++) and `m2_g_arm.hip` (hipcc). The includer defines `M2G_ARM_NAME`,
 * `M2G_COMPILER` and, for the HIP arm, `M2G_HAVE_HIP` plus `m2g_device_run()`. The arithmetic is in
 * `m2_g_kernels.h`, the plan parser and its refusals in `m2_g_io.h`; what is here is the command line,
 * the order of acts, and the report — the same report shape the shared-expert and router arms print, so
 * the driver parses all three with one parser (`engine/device_arm.py::run_arm`).
 *
 * WHAT THE REPORT CARRIES that the others' does not: the per-expert_index census line. AGENTS.md
 * prediction 2 says the gather's shape — 8 dependent ~6 MiB reads per token per layer — has to be
 * measured before the scheduler decision is made, so the arm prints, on every run and on both paths,
 * which expert indices were touched, how many matrix spans and how many bytes each one cost, how many
 * element reads the dots actually performed, and whether any of it was a whole tensor. A counter that
 * can only read 0 is not a guard, so `--fault whole-note` and `--fault skip-raw` exist to be witnessed
 * moving those two counters (`engine/block_device.py`'s controls).
 *
 * NO TIMER. Nothing here reads a clock and no rate appears in any line (`pitfalls/00`).
 */

static void m2g_usage(void) {
    fprintf(stderr,
            "usage: --plan P --slab S --out O [--device] [--fault NAME] [--check-only]\n"
            "       arm=%s compiler=%s device_path=%s\n"
            "       faults=none|skip-raw|skip-routed|whole-note|corrupt-byte|alloc-oversize\n",
            M2G_ARM_NAME, M2G_COMPILER,
#ifdef M2G_HAVE_HIP
            "yes");
#else
            "no (g++ twin: no HIP symbol exists in this binary)");
#endif
}

static int m2g_host_run(const struct M2GPlan *plan, const char *slab, float *out,
                        struct M2GCounters *ctr, struct M2GRunStats *st, char *err, size_t errn) {
    int bad = 0;
    (void)err;
    (void)errn;
    memset(st, 0, sizeof(*st));
    st->path = "host_cpubody";
    m2g_run(plan, slab, out, ctr, &bad);
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
    struct M2GPlanFile pf;
    struct M2GPlan plan;
    struct M2GCounters ctr;
    struct M2GRunStats st;
    char err[768], fields[256], census[2048], hex[129];
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
            m2g_usage();
            return 2;
        }
    }
    if (!plan_path || !slab_path || !out_path) {
        fprintf(stderr, "FAIL tag=arg-missing note=--plan, --slab and --out are all required\n");
        m2g_usage();
        return 2;
    }
    err[0] = '\0';
    if (m2g_read_plan(plan_path, &pf, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
        return 1;
    }
    if (fault_cli && *fault_cli) snprintf(pf.fault_word, sizeof(pf.fault_word), "%s", fault_cli);
    slab = m2g_read_file(slab_path, &slab_bytes, err, sizeof(err));
    if (!slab) {
        fprintf(stderr, "FAIL tag=slab-unreadable note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
        return 1;
    }
    if (pf.slab_bytes && pf.slab_bytes != slab_bytes) {
        fprintf(stderr, "FAIL tag=slab-size note=%s is %ld B, the plan says %ld B; the arm refuses to run "
                        "a plan against a slab that is not the one it was written for\n",
                slab_path, slab_bytes, pf.slab_bytes);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
        free(slab);
        return 1;
    }
    if (m2g_plan_check(&pf, slab_bytes, &plan, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=plan-rejected note=%s\n", err);
        fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
        free(slab);
        return 1;
    }
    printf("HDR arm=%s compiler=%s path=%s H=%d I=%d K=%d E=%d floats=%d want_units=%d want_spans=%d "
           "want_bytes=%ld mode=%s mode_fields=%s\n",
           M2G_ARM_NAME, M2G_COMPILER, M2G_PATH_NAME(plan.path), MG_H, MG_I, MG_K, MG_E, M2G_FLOATS,
           m2g_want_units(&plan), m2g_want_spans(&plan), m2g_want_bytes(&plan), pf.mode_word,
           m2g_mode_field(plan.mode));
    printf("PLAN path=%s version=%d slab_bytes=%ld plan_path=%s x_off=%ld x_elt=%d units=%d fused=%d "
           "mode=%s fault=%s corrupt_off=%ld read_off=%ld read_len=%d\n",
           plan_path, pf.version, slab_bytes, pf.path_word, plan.off_x, plan.elt_x, plan.n_units,
           (plan.n_units && plan.units[0].fused) ? 1 : 0, pf.mode_word, pf.fault_word,
           plan.corrupt_off, plan.read_off, plan.read_len);
    if (check_only) {
        printf("RESULT arm=%s verdict=OK note=plan and slab verified; nothing computed no timing taken\n",
               M2G_ARM_NAME);
        free(slab);
        return 0;
    }
    out = (float *)malloc(sizeof(float) * (size_t)M2G_FLOATS);
    if (!out) {
        fprintf(stderr, "FAIL tag=oom note=record alloc failed\n");
        free(slab);
        return 1;
    }
    memset(out, 0, sizeof(float) * (size_t)M2G_FLOATS);
    memset(&ctr, 0, sizeof(ctr));
    /* The corrupt-byte fault flips one byte of the slab on the side under test: on this path that is the
     * host copy of the slab bytes, before anything reads them. In a device window the same word drives a
     * flip inside the device's own copy, after the H2D — the only corruption a read-back can see. */
    if (!strcmp(pf.fault_word, "corrupt-byte")) {
        slab[plan.corrupt_off] = (uint8_t)(slab[plan.corrupt_off] ^ 0xFF);
        m2g_hex((const char *)slab, plan.corrupt_off, 8, hex, sizeof(hex));
        printf("CORRUPT side=host_cpubody off=%ld hex8=%s\n", plan.corrupt_off, hex);
    }
#ifdef M2G_HAVE_HIP
    if (want_device) {
        rc = m2g_device_run(&plan, (const char *)slab, slab_bytes, out, &ctr, &st,
                            &pf, err, sizeof(err));
        if (rc) {
            fprintf(stderr, "FAIL tag=device-run-failed note=%s\n", err);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
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
                    M2G_ARM_NAME, M2G_COMPILER);
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (!strcmp(pf.fault_word, "alloc-oversize")) {
            fprintf(stderr, "FAIL tag=fault-device-only note=alloc-oversize is a device act; this binary "
                            "has no device path, so the leg cannot execute here and is not faked\n");
            fprintf(stderr, "RESULT arm=%s verdict=FAIL findings=1 no timing taken\n", M2G_ARM_NAME);
            free(out);
            free(slab);
            return 1;
        }
        if (m2g_host_run(&plan, (const char *)slab, out, &ctr, &st, err, sizeof(err))) {
            fprintf(stderr, "FAIL tag=host-run-failed note=%s\n", err);
            free(out);
            free(slab);
            return 1;
        }
    }
    m2g_field_report(out, fields, sizeof(fields));
    m2g_census_report(&ctr, &plan, census, sizeof(census));
    if (plan.read_off >= 0 && plan.read_len > 0) {
        /* The unit read-back leg exists to bind what the DEVICE holds to what the file holds, so it prints
         * only where a device actually copied bytes back. On the host body the driver binds file bytes to
         * slab bytes by digest, and printing a host "read-back" as if it were one would be exactly the
         * substitution this lane refuses. */
        if (st.read_hex[0])
            printf("UNITREADBACK path=%s off=%ld bytes=%d hex=%s\n", st.path, plan.read_off,
                   plan.read_len, st.read_hex);
        else if (st.path && strcmp(st.path, "device") == 0)
            printf("UNITREADBACK path=%s off=%ld bytes=%d hex=ABSENT\n", st.path, plan.read_off,
                   plan.read_len);
    }
    if (m2g_write_f32(out_path, out, M2G_FLOATS, err, sizeof(err))) {
        fprintf(stderr, "FAIL tag=out-unsafe note=%s\n", err);
        free(out);
        free(slab);
        return 1;
    }
    printf("RUN arm=%s path=%s hip_calls=%d launches=%d allocs=%d bytes_to_device=%ld "
           "bytes_from_device=%ld slab_bytes=%ld out_floats=%d bad_index=%d nan_fields=\"%s\" "
           "census=\"%s\"\n",
           M2G_ARM_NAME, st.path, st.hip_calls, st.launches, st.allocs, (long)st.bytes_in,
           (long)st.bytes_out, slab_bytes, M2G_FLOATS, st.bad_index, fields, census);
    printf("WROTE path=%s bytes=%d\n", out_path, (int)(sizeof(float) * M2G_FLOATS));
    printf("RESULT arm=%s verdict=OK no timing taken\n", M2G_ARM_NAME);
    (void)rc;
    free(out);
    free(slab);
    return 0;
}
