/* shared_expert.hip — the device arm: the shared expert at policy-A widths (Q8_0 weights, exact F32
 * shared_expert_gate vector), one thread per output row, on gfx1151.
 *
 * The arithmetic is NOT spelled here. Every stage body comes from `shared_expert_kernels.h`, which
 * `host_probe.cpp` compiles with the same `-ffp-contract=off` over the same work decomposition, so a
 * host-vs-device difference is a device finding rather than a spelling difference. What this file owns
 * is the launch shape, the poison pre-fill, the device launch counter, and the occupancy probe.
 *
 *   ./shared_expert --selftest [--arm=N]                  three-arm hand-worked suite (no device)
 *   ./shared_expert --occupancy                           occupancy probe + canaries: a host-side
 *                                                         query, no allocation, no launch
 *   ./shared_expert --plan P --slab S --expect E --out O --report R \
 *                   --expect-weights N --expect-acts N --expect-cases N --out-floats N [--per-case]
 *
 * Correctness only: it prints no duration of anything. The wrapper that runs it (`gate`) is the sole
 * acquirer of the repo's GPU lock; this program never mentions the lock tool.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_expert_kernels.h"
#include "shared_expert_selftest.h"

#define SE_HIP(x)                                                                     \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                       \
            fprintf(stderr, "hip: %s -> %s\n", #x, hipGetErrorString(e_));             \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

/* Work-decomposition record: which case a block belongs to, where its work starts, and which case
 * index to read. Built on the host, uploaded once, read by every kernel, so both arms split the work
 * identically (one address expression shared by writer and reader, pitfalls/40). */
typedef struct { int32_t case_pos, base; } SeItem;

/* Per-case weight-read counters, accumulated with atomics: three slots per case (fused, down, gate
 * vector). A read-back published per thread is not a reduction (pitfalls/20), so nothing here lets
 * one thread stand in for the block; every thread adds what it actually read, and gate.py re-derives
 * the same totals from geometry and refuses a disagreement. */
enum { SE_RD_FUSED = 0, SE_RD_DOWN = 1, SE_RD_SG = 2, SE_RD_COUNT = 3 };

/* ---- the kernels ----------------------------------------------------------------------------- */
extern "C" __global__ void se_k_gateup(const SeItem *items, int item_base, const SeCase *cases,
                                       const SeWeight *weights, int n_w, const SeAct *acts, int n_a,
                                       const uint8_t *slab, float *out, unsigned long long *launches,
                                       unsigned int *reads_dev) {
    const SeItem it = items[item_base + blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(launches, 1ull);
    SePlan p;
    p.w = (SeWeight *)weights; p.n_w = n_w; p.a = (SeAct *)acts; p.n_a = n_a;
    p.slab = slab; p.slab_bytes = 0; p.expect = 0; p.n_expect = 0;
    const SeCase *c = &cases[it.case_pos];
    const int j = it.base + (int)threadIdx.x;
    int reads = 0;
    se_stage_gateup(&p, c, out, j, &reads);
    atomicAdd(&reads_dev[it.case_pos * SE_RD_COUNT + SE_RD_FUSED], (unsigned int)reads);
}

extern "C" __global__ void se_k_down(const SeItem *items, int item_base, const SeCase *cases,
                                     const SeWeight *weights, int n_w, const SeAct *acts, int n_a,
                                     const uint8_t *slab, float *out, unsigned long long *launches,
                                     unsigned int *reads_dev) {
    const SeItem it = items[item_base + blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(launches, 1ull);
    SePlan p;
    p.w = (SeWeight *)weights; p.n_w = n_w; p.a = (SeAct *)acts; p.n_a = n_a;
    p.slab = slab; p.slab_bytes = 0; p.expect = 0; p.n_expect = 0;
    const SeCase *c = &cases[it.case_pos];
    const int h = it.base + (int)threadIdx.x;
    int reads = 0;
    se_stage_down(&p, c, out, h, &reads);
    atomicAdd(&reads_dev[it.case_pos * SE_RD_COUNT + SE_RD_DOWN], (unsigned int)reads);
}

extern "C" __global__ void se_k_prob(const SeItem *items, int item_base, const SeCase *cases,
                                     const SeWeight *weights, int n_w, const SeAct *acts, int n_a,
                                     const uint8_t *slab, float *out, unsigned long long *launches,
                                     unsigned int *reads_dev) {
    const SeItem it = items[item_base + blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(launches, 1ull);
    if ((int)threadIdx.x != 0) return;
    SePlan p;
    p.w = (SeWeight *)weights; p.n_w = n_w; p.a = (SeAct *)acts; p.n_a = n_a;
    p.slab = slab; p.slab_bytes = 0; p.expect = 0; p.n_expect = 0;
    const SeCase *c = &cases[it.case_pos];
    if (c->sgate_form != SE_SGATE_SCALAR) return;      /* per-channel: the field stays POISON */
    int reads = 0;
    out[c->dst_off + SE_OFF_PROB] = se_stage_prob(&p, c, &reads);
    atomicAdd(&reads_dev[it.case_pos * SE_RD_COUNT + SE_RD_SG], (unsigned int)reads);
}

extern "C" __global__ void se_k_out(const SeItem *items, int item_base, const SeCase *cases,
                                    const SeWeight *weights, int n_w, const SeAct *acts, int n_a,
                                    const uint8_t *slab, float *out, unsigned long long *launches,
                                    unsigned int *reads_dev) {
    const SeItem it = items[item_base + blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(launches, 1ull);
    SePlan p;
    p.w = (SeWeight *)weights; p.n_w = n_w; p.a = (SeAct *)acts; p.n_a = n_a;
    p.slab = slab; p.slab_bytes = 0; p.expect = 0; p.n_expect = 0;
    const SeCase *c = &cases[it.case_pos];
    const int h = it.base + (int)threadIdx.x;
    const float prob = out[c->dst_off + SE_OFF_PROB];
    int reads = 0;
    out[c->dst_off + SE_OFF_OUT + h] = se_stage_out_channel(&p, c, out, prob, h, &reads);
    if (reads) atomicAdd(&reads_dev[it.case_pos * SE_RD_COUNT + SE_RD_SG], (unsigned int)reads);
}

/* Occupancy canaries: the probe must return 1 block/CU for the 64 KiB-LDS stub and the machine
 * maximum for the 0-LDS one, or its per-kernel number means nothing (pitfalls/20 §"A probe's own
 * summary column can be a mislabeled counter"). Both stubs WRITE the array before reading it, so
 * LLVM cannot delete the allocation and make the control vacuous. */
extern "C" __global__ void se_canary_empty(float *o) { if (o) *o = 1.0f; }
extern "C" __global__ void se_canary_full_lds(float *o) {
    __shared__ uint32_t pad[64 * 1024 / 4];
    pad[threadIdx.x & 1023u] = threadIdx.x;
    __syncthreads();
    if (o) *o = (float)pad[threadIdx.x & 1023u];
}
/* The census slicer's witness: 8 multiply-adds with contraction compiled out is 8 multiplies and 8
 * adds, so a sliced body with fewer than 16 vector-alu slots is a slicer bug, not a cheap kernel
 * (pitfalls/20 §"Before writing a probe or census tool" — the 57-slot-stub failure). */
extern "C" __global__ void se_canary_eight_fma(const float *a, float *o) {
    const unsigned t = threadIdx.x;
    float s = a[t & 63u];
    s += a[(t + 1u) & 63u] * 1.5f;  s += a[(t + 2u) & 63u] * 2.5f;
    s += a[(t + 3u) & 63u] * 3.5f;  s += a[(t + 4u) & 63u] * 4.5f;
    s += a[(t + 5u) & 63u] * 5.5f;  s += a[(t + 6u) & 63u] * 6.5f;
    s += a[(t + 7u) & 63u] * 7.5f;  s += a[(t + 8u) & 63u] * 8.5f;
    o[t] = s;
}
/* The second witness of the same slicer at twice the work. Two stubs of different size catch a
 * slicer that silently drops every other row, which an ordering test alone cannot see: gfx1151
 * peins VALU pairs into `v_dual_*` encodings, so a raw instruction count is not a linear function of
 * the source and the witness is stated over f32 VALU ops with the duals counted at two. */
extern "C" __global__ void se_canary_sixteen_fma(const float *a, float *o) {
    const unsigned t = threadIdx.x;
    float s = a[t & 63u];
    for (int i = 1; i <= 16; ++i) s += a[(t + (unsigned)i) & 63u] * (0.5f + (float)i);
    o[t] = s;
}

/* ---- occupancy: a host-side calculator, not a launch ----------------------------------------- */
static int occupancy_report(void) {
    int dev = 0;
    SE_HIP(hipGetDevice(&dev));
    int max_cu = 0, lds_cu = 0, warp = 0;
    SE_HIP(hipDeviceGetAttribute(&max_cu, hipDeviceAttributeMultiprocessorCount, dev));
    SE_HIP(hipDeviceGetAttribute(&lds_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
    SE_HIP(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev));
    /* HIP reports 20 CU on this package while KFD reports 40 SIMDs (pitfalls/20 §"The CU count from
     * KFD, not from HIP"); the census uses the KFD number, and printing both here keeps the gap from
     * being inherited silently by whoever reads this line next. */
    printf("OCCUPANCYDEV cu_hipreported=%d lds_per_cu=%d warp_size=%d\n", max_cu, lds_cu, warp);
    struct { const char *name; const void *fn; int threads; int canary; } ks[] = {
        {"se_canary_empty", (const void *)se_canary_empty, SE_THREADS, 1},
        {"se_canary_full_lds", (const void *)se_canary_full_lds, SE_THREADS, 1},
        {"se_canary_eight_fma", (const void *)se_canary_eight_fma, SE_THREADS, 1},
        {"se_canary_sixteen_fma", (const void *)se_canary_sixteen_fma, SE_THREADS, 1},
        {"se_k_gateup", (const void *)se_k_gateup, SE_THREADS, 0},
        {"se_k_down", (const void *)se_k_down, SE_THREADS, 0},
        {"se_k_prob", (const void *)se_k_prob, SE_THREADS, 0},
        {"se_k_out", (const void *)se_k_out, SE_THREADS, 0},
    };
    int n_empty = 0, n_full = 0;
    for (unsigned i = 0; i < sizeof(ks) / sizeof(ks[0]); ++i) {
        int blocks = -1;
        const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, ks[i].fn, ks[i].threads, 0);
        if (e != hipSuccess) {
            printf("OCCUPANCY kernel=%s error=%s\n", ks[i].name, hipGetErrorString(e));
            continue;
        }
        printf("OCCUPANCY kernel=%s blocks_per_cu=%d threads=%d wavefronts_per_cu=%d canary=%d\n",
               ks[i].name, blocks, ks[i].threads, blocks * (ks[i].threads / (warp > 0 ? warp : 32)), ks[i].canary);
        if (!strcmp(ks[i].name, "se_canary_empty")) n_empty = blocks;
        if (!strcmp(ks[i].name, "se_canary_full_lds")) n_full = blocks;
    }
    /* the canary verdict: the 64 KiB stub must be squeezed to 1 block/CU and the 0-LDS stub must be
     * above it, or the probe is blind to LDS and every number above is a decoration. */
    const int canary_ok = (n_full == 1) && (n_empty > 1);
    printf("OCCUPANCYCANARY empty=%d full_lds=%d verdict=%s\n", n_empty, n_full,
           canary_ok ? "probe-sees-lds" : "PROBE-BLIND");
    return canary_ok ? 0 : 1;
}

/* ---- driver ---------------------------------------------------------------------------------- */
typedef struct {
    const char *plan_path, *slab_path, *expect_path, *out_path, *report_path;
    int want_w, want_a, want_c, out_floats, expect_floats, per_case;
} Args;

static int arg_int(const char *s, const char *why) {
    char *end = 0;
    const long v = strtol(s, &end, 10);
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
    memset(a, 0, sizeof(*a));
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
        fprintf(stderr, "--plan --slab --expect --out --report --out-floats --expect-cases are "
                        "all required\n");
        exit(2);
    }
}

static int rows_of_mode(int mode) { return mode == SE_MODE_DOWN_HALF ? SE_H / 2 : SE_H; }

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
        printf("SELFTEST3 arms_run=1 arm=%d checks=%d changed=%d\n", only, rep.n_checks, rep.changed);
        printf("FAIL selftest single-arm: one arm alone is not a pass (arm=%d)\n", only);
        return 1;
    }
    if (argc > 1 && !strcmp(argv[1], "--occupancy")) {
        for (int i = 2; i < argc; ++i) { fprintf(stderr, "--occupancy takes no arguments\n"); return 2; }
        return occupancy_report();
    }

    Args a;
    parse(argc, argv, &a);
    size_t slab_n = 0, exp_n = 0;
    uint8_t *slab = read_file(a.slab_path, &slab_n);
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

    /* work decomposition, identical to host_probe.cpp's */
    int n1 = 0, n2 = 0, n3 = 0;
    for (int i = 0; i < p.n_c; ++i) {
        n1 += SE_I / SE_THREADS;
        n2 += rows_of_mode(p.c[i].mode) / SE_THREADS;
        n3 += 1 + SE_H / SE_THREADS;
    }
    SeItem *it1 = (SeItem *)malloc(sizeof(SeItem) * n1);
    SeItem *it2 = (SeItem *)malloc(sizeof(SeItem) * n2);
    SeItem *it3 = (SeItem *)malloc(sizeof(SeItem) * n3);
    int32_t lo1 = 0, lo2 = 0, lo3 = 0;
    int32_t *blo1 = (int32_t *)calloc(p.n_c, 4), *bhi1 = blo1 + p.n_c;
    int32_t *blo2 = (int32_t *)calloc(p.n_c, 4), *bhi2 = blo2 + p.n_c;
    int32_t *blo3 = (int32_t *)calloc(p.n_c, 4), *bhi3 = blo3 + p.n_c;
    for (int i = 0; i < p.n_c; ++i) {
        blo1[i] = lo1; for (int b = 0; b < SE_I / SE_THREADS; ++b) { it1[lo1].case_pos = i; it1[lo1].base = b * SE_THREADS; ++lo1; } bhi1[i] = lo1;
        const int rows = rows_of_mode(p.c[i].mode);
        blo2[i] = lo2; for (int b = 0; b < rows / SE_THREADS; ++b) { it2[lo2].case_pos = i; it2[lo2].base = b * SE_THREADS; ++lo2; } bhi2[i] = lo2;
        blo3[i] = lo3; it3[lo3].case_pos = i; it3[lo3].base = -1; ++lo3;
        for (int b = 0; b < SE_H / SE_THREADS; ++b) { it3[lo3].case_pos = i; it3[lo3].base = b * SE_THREADS; ++lo3; } bhi3[i] = lo3;
    }

    float *out = (float *)malloc(sizeof(float) * a.out_floats);
    for (int i = 0; i < a.out_floats; ++i) out[i] = se_float_of(SE_NAN_BITS);

    uint8_t *d_slab = 0; float *d_out = 0; SeCase *d_cases = 0; SeWeight *d_w = 0; SeAct *d_a = 0;
    SeItem *d_i1 = 0, *d_i2 = 0, *d_i3 = 0; unsigned long long *d_launch = 0;
    unsigned int *d_reads = 0;
    SE_HIP(hipMalloc((void **)&d_slab, slab_n));
    SE_HIP(hipMemcpy(d_slab, slab, slab_n, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_out, sizeof(float) * a.out_floats));
    SE_HIP(hipMemcpy(d_out, out, sizeof(float) * a.out_floats, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_cases, sizeof(SeCase) * p.n_c));
    SE_HIP(hipMemcpy(d_cases, p.c, sizeof(SeCase) * p.n_c, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_w, sizeof(SeWeight) * p.n_w));
    SE_HIP(hipMemcpy(d_w, p.w, sizeof(SeWeight) * p.n_w, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_a, sizeof(SeAct) * p.n_a));
    SE_HIP(hipMemcpy(d_a, p.a, sizeof(SeAct) * p.n_a, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_i1, sizeof(SeItem) * n1));
    SE_HIP(hipMemcpy(d_i1, it1, sizeof(SeItem) * n1, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_i2, sizeof(SeItem) * n2));
    SE_HIP(hipMemcpy(d_i2, it2, sizeof(SeItem) * n2, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_i3, sizeof(SeItem) * n3));
    SE_HIP(hipMemcpy(d_i3, it3, sizeof(SeItem) * n3, hipMemcpyHostToDevice));
    SE_HIP(hipMalloc((void **)&d_launch, sizeof(unsigned long long)));
    SE_HIP(hipMemset(d_launch, 0, sizeof(unsigned long long)));
    SE_HIP(hipMalloc((void **)&d_reads, sizeof(unsigned int) * SE_RD_COUNT * p.n_c));
    SE_HIP(hipMemset(d_reads, 0, sizeof(unsigned int) * SE_RD_COUNT * p.n_c));

    int issued = 0;
    if (!a.per_case) {
        se_k_gateup<<<n1, SE_THREADS>>>(d_i1, 0, d_cases, d_w, p.n_w, d_a, p.n_a, d_slab, d_out,
                                       d_launch, d_reads);
        se_k_down<<<n2, SE_THREADS>>>(d_i2, 0, d_cases, d_w, p.n_w, d_a, p.n_a, d_slab, d_out,
                                     d_launch, d_reads);
        se_k_prob<<<n3, SE_THREADS>>>(d_i3, 0, d_cases, d_w, p.n_w, d_a, p.n_a, d_slab, d_out,
                                     d_launch, d_reads);
        se_k_out<<<n3, SE_THREADS>>>(d_i3, 0, d_cases, d_w, p.n_w, d_a, p.n_a, d_slab, d_out,
                                    d_launch, d_reads);
        issued = 4;
    } else {
        for (int i = 0; i < p.n_c; ++i) {
            se_k_gateup<<<bhi1[i] - blo1[i], SE_THREADS>>>(d_i1, blo1[i], d_cases, d_w, p.n_w, d_a,
                                                          p.n_a, d_slab, d_out, d_launch, d_reads);
            se_k_down<<<bhi2[i] - blo2[i], SE_THREADS>>>(d_i2, blo2[i], d_cases, d_w, p.n_w, d_a,
                                                        p.n_a, d_slab, d_out, d_launch, d_reads);
            se_k_prob<<<bhi3[i] - blo3[i], SE_THREADS>>>(d_i3, blo3[i], d_cases, d_w, p.n_w, d_a,
                                                        p.n_a, d_slab, d_out, d_launch, d_reads);
            se_k_out<<<bhi3[i] - blo3[i], SE_THREADS>>>(d_i3, blo3[i], d_cases, d_w, p.n_w, d_a,
                                                       p.n_a, d_slab, d_out, d_launch, d_reads);
            issued += 4;
        }
    }
    SE_HIP(hipGetLastError());
    SE_HIP(hipDeviceSynchronize());
    unsigned long long counted = 0;
    SE_HIP(hipMemcpy(&counted, d_launch, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned int *rd = (unsigned int *)calloc(SE_RD_COUNT * (size_t)p.n_c, sizeof(unsigned int));
    SE_HIP(hipMemcpy(rd, d_reads, sizeof(unsigned int) * SE_RD_COUNT * p.n_c, hipMemcpyDeviceToHost));
    SE_HIP(hipMemcpy(out, d_out, sizeof(float) * a.out_floats, hipMemcpyDeviceToHost));
    for (int i = 0; i < p.n_c; ++i) {
        p.c[i].reads_fused = (int)rd[i * SE_RD_COUNT + SE_RD_FUSED];
        p.c[i].reads_down = (int)rd[i * SE_RD_COUNT + SE_RD_DOWN];
        p.c[i].reads_sg = (int)rd[i * SE_RD_COUNT + SE_RD_SG];
    }

    /* per-case counters over the case's own span, computed from the device's own output buffer */
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
    printf("LAUNCHES mode=%s issued=%d device_counted=%llu blocks_stage1=%d stage2=%d stage3=%d "
           "threads=%d\n", a.per_case ? "per-case" : "grouped", issued, counted, n1, n2, n3,
           SE_THREADS);
    printf("ARM device arch=gfx1151\nno timing taken\n");
    (void)hipFree(d_slab); (void)hipFree(d_out); (void)hipFree(d_cases); (void)hipFree(d_w);
    (void)hipFree(d_a); (void)hipFree(d_i1); (void)hipFree(d_i2); (void)hipFree(d_i3);
    (void)hipFree(d_launch); (void)hipFree(d_reads);
    free(rd);
    free(it1); free(it2); free(it3); free(blo1); free(blo2); free(blo3);
    return 0;
}
