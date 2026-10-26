/* m2_se_arm.hip — the hipcc arm of the M2 shared-expert stage: gfx1151 codegen, and the only one of
 * the two binaries that can launch anything.
 *
 * TWO ARMS, ONE SOURCE. `m2_se_host.cpp` compiles the same `m2_se_kernels.h` with g++; run both on
 * one host-visible plan and compare the records byte for byte (`engine/block_device.py` prints the
 * `PAIR` line). Without `--device` this binary executes the same body on the CPU and reports
 * `hip_calls=0` — that is the arm the compiled-pair comparison judges, and it is a numerics verdict,
 * not a device run (`prototype/gdn/gdn.hip` states the same discipline for its arm).
 *
 * WITH `--device` the acts are these, and every one of them is counted in `M2SeRunStats`: three
 * `hipMalloc`, one H2D of the slab, one `hipMemset` of the counters, four launches (poison, stage 1,
 * stage 2, stage 3), two D2H, one synchronize, three `hipFree`. A kernel launch is the only thing that
 * can move a stage row, so the counter is what the driver checks before it is allowed to print a
 * device-shaped line.
 *
 * THE POISON LAUNCH IS NOT DECORATION. The host body fills the record with the prototype's NaN pattern
 * before any stage runs, so an unwritten slot reads NaN; a `hipMemset` cannot write that pattern one
 * float at a time, so the device poisons it with a kernel of its own. Without it the `skip-stage3`
 * control would read 0.0 on the device and 0x7FC00000 on the host, and the census would mean two
 * different things on the two paths.
 *
 * CORRECTNESS ONLY. No timer is constructed, no duration printed, no warm loop: measurements belong
 * to the main agent (`AGENTS.md` §“GPU execution is serialized”). The counters below count calls,
 * which is a structural fact and not a rate.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_se_io.h"

#define M2_SE_ARM_NAME "hip"
#define M2_SE_COMPILER "hipcc"
#define M2_SE_HAVE_HIP 1

#define M2_SE_THREADS SE_THREADS          /* 128 threads per block, from the lane's own header */

/* One place where a HIP call becomes a counted call. The tag is the failure's name, so a device error
 * exits non-zero with a tag the driver can require — a half-staged run that still printed a record is
 * the bad number this lane exists to prevent (`engine/stage_run.py`::_chk says the same). */
static int g_hip_calls = 0;
static int g_launches = 0;
static int g_allocs = 0;
static long g_bytes_in = 0;
static long g_bytes_out = 0;

#define M2_CK(call, tag)                                                                \
    do {                                                                                \
        const hipError_t _e = (call);                                                   \
        ++g_hip_calls;                                                                  \
        if (_e != hipSuccess) {                                                         \
            snprintf(err, errn, "%s: %s -> %s", tag, #call, hipGetErrorString(_e));     \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

__global__ void m2_se_k0_poison(float *out) {
    if (!threadIdx.x && !blockIdx.x) m2_se_poison(out);   /* one thread, the whole record */
}

__global__ void m2_se_k1(const M2SePlan p, const uint8_t *slab, float *out, M2SeCounters *ctr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    struct M2SeTally t;
    SePlan sp;
    SeWeight w[3];
    SeAct a[1];
    SeCase c;
    if (j >= SE_I) return;                       /* the grid is derived, so the tail is a no-op */
    t.fused = t.down = t.sg = t.items = 0;
    m2_se_fill(&p, slab, &sp, w, a, &c);
    m2_se_item1(&sp, &c, out, j, &t);
    M2_ACC(&ctr->reads_fused, (uint32_t)t.fused);
    M2_ACC(&ctr->reads_sg, (uint32_t)t.sg);
    M2_ACC(&ctr->items1, (uint32_t)t.items);
}

__global__ void m2_se_k2(const M2SePlan p, const uint8_t *slab, float *out, M2SeCounters *ctr) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    struct M2SeTally t;
    SePlan sp;
    SeWeight w[3];
    SeAct a[1];
    SeCase c;
    if (h >= SE_H) return;
    t.fused = t.down = t.sg = t.items = 0;
    m2_se_fill(&p, slab, &sp, w, a, &c);
    m2_se_item2(&sp, &c, out, h, &t);
    M2_ACC(&ctr->reads_down, (uint32_t)t.down);
    M2_ACC(&ctr->items2, (uint32_t)t.items);
}

__global__ void m2_se_k3(const M2SePlan p, const uint8_t *slab, float *out, M2SeCounters *ctr) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    struct M2SeTally t;
    SePlan sp;
    SeWeight w[3];
    SeAct a[1];
    SeCase c;
    if (h >= SE_H) return;
    t.fused = t.down = t.sg = t.items = 0;
    m2_se_fill(&p, slab, &sp, w, a, &c);
    m2_se_item3(&sp, &c, out, out[c.dst_off + SE_OFF_PROB], h, &t);
    M2_ACC(&ctr->items3, (uint32_t)t.items);
}

/* Control-only: an allocation no pool can satisfy, so the launch-failure path has a device-side leg
 * the main agent can witness without editing the artifact or the plan. */
static int m2_se_probe_oversize(char *err, size_t errn) {
    void *p = 0;
    const hipError_t e = hipMalloc(&p, (size_t)1 << 44);
    ++g_hip_calls;
    ++g_allocs;
    if (e == hipSuccess && p) {
        const hipError_t f = hipFree(p);
        ++g_hip_calls;
        (void)f;
    }
    snprintf(err, errn, "alloc-oversize: hipMalloc(1 TiB) returned %s; a device run that survives it "
                        "has no allocation failure to test", hipGetErrorString(e));
    return e == hipSuccess ? 0 : 1;
}

static int m2_se_device_run(const struct M2SePlan *plan, const uint8_t *slab, long slab_bytes,
                            float *out, struct M2SeCounters *ctr, struct M2SeRunStats *st,
                            char *err, size_t errn) {
    uint8_t *d_slab = 0;
    float *d_out = 0;
    M2SeCounters *d_ctr = 0;
    const int b1 = (M2_SE_ITEMS1 + M2_SE_THREADS - 1) / M2_SE_THREADS;
    const int b2 = (M2_SE_ITEMS2 + M2_SE_THREADS - 1) / M2_SE_THREADS;
    const int b3 = (M2_SE_ITEMS3 + M2_SE_THREADS - 1) / M2_SE_THREADS;

    memset(st, 0, sizeof(*st));
    st->path = "device";
    g_hip_calls = 0;
    g_launches = 0;
    g_allocs = 0;
    g_bytes_in = 0;
    g_bytes_out = 0;
    memset(ctr, 0, sizeof(*ctr));

    if (plan->alloc_oversize) {
        const int r = m2_se_probe_oversize(err, errn);
        st->hip_calls = g_hip_calls;
        st->launches = g_launches;
        st->allocs = g_allocs;
        return r;
    }

    M2_CK(hipMalloc((void **)&d_slab, (size_t)slab_bytes), "alloc-slab");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_out, sizeof(float) * M2_SE_FLOATS), "alloc-out");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_ctr, sizeof(M2SeCounters)), "alloc-counters");
    ++g_allocs;
    M2_CK(hipMemcpy(d_slab, slab, (size_t)slab_bytes, hipMemcpyHostToDevice), "h2d-slab");
    g_bytes_in += slab_bytes;
    M2_CK(hipMemset(d_ctr, 0, sizeof(M2SeCounters)), "memset-counters");

    m2_se_k0_poison<<<1, 1>>>(d_out);
    M2_CK(hipGetLastError(), "launch-poison");
    ++g_launches;
    m2_se_k1<<<b1, M2_SE_THREADS>>>(*plan, d_slab, d_out, d_ctr);
    M2_CK(hipGetLastError(), "launch-stage1");
    ++g_launches;
    m2_se_k2<<<b2, M2_SE_THREADS>>>(*plan, d_slab, d_out, d_ctr);
    M2_CK(hipGetLastError(), "launch-stage2");
    ++g_launches;
    if (!plan->skip_stage3) {
        m2_se_k3<<<b3, M2_SE_THREADS>>>(*plan, d_slab, d_out, d_ctr);
        M2_CK(hipGetLastError(), "launch-stage3");
        ++g_launches;
    }
    M2_CK(hipDeviceSynchronize(), "sync");

    M2_CK(hipMemcpy(out, d_out, sizeof(float) * M2_SE_FLOATS, hipMemcpyDeviceToHost), "d2h-out");
    g_bytes_out += sizeof(float) * M2_SE_FLOATS;
    M2_CK(hipMemcpy(ctr, d_ctr, sizeof(M2SeCounters), hipMemcpyDeviceToHost), "d2h-counters");
    g_bytes_out += sizeof(M2SeCounters);

    st->hip_calls = g_hip_calls;
    st->launches = g_launches;
    st->allocs = g_allocs;
    st->bytes_in = g_bytes_in;
    st->bytes_out = g_bytes_out;

    M2_CK(hipFree(d_slab), "free-slab");
    M2_CK(hipFree(d_out), "free-out");
    M2_CK(hipFree(d_ctr), "free-counters");
    return 0;
}

#include "m2_se_main.inc"
