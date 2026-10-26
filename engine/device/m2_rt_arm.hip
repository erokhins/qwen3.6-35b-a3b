/* m2_rt_arm.hip — the hipcc arm of the M2 router stage: gfx1151 codegen, and the only one of the two
 * binaries that can launch anything.
 *
 * TWO ARMS, ONE SOURCE. `m2_rt_host.cpp` compiles the same `m2_rt_kernels.h` with g++; run both on
 * one host-visible plan and compare the records byte for byte. Without `--device` this binary
 * executes the same body on the CPU and reports `hip_calls=0` — the compiled-pair arm, a numerics
 * verdict and not a device run.
 *
 * WITH `--device` the acts are: three `hipMalloc` (slab, record, counters), one H2D of the slab, one
 * `hipMemset` (the counters only — the record is poisoned by a launch of its own, because a `hipMemset`
 * cannot write a NaN pattern one float at a time and a memset-to-zero would make the census mean one
 * thing on the host body and another on the device), three launches (poison, R1 over the 256 experts,
 * R2..R5 as the one serial item), two D2H, one synchronize, three `hipFree`. Every one of them is
 * counted in `M2RtRunStats`, because the counter — not the flag — is what lets the driver print anything
 * device-shaped.
 *
 * SHAPE, NOT SPEED. R1 is one thread per expert and R2..R5 is literally one thread, because that is
 * the shape the specification writes and the shape this slice can be judged for correctness on. The
 * router is the block's per-token serial point (AGENTS.md prediction 1), so `items=256/1` in the RUN
 * line is a statement about the algorithm, not a tuning claim, and no timing is taken here.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_rt_io.h"

#define M2_RT_ARM_NAME "rt_hip"
#define M2_RT_COMPILER "hipcc"
#define M2_RT_HAVE_HIP 1

#define M2_RT_THREADS 128

struct M2RtRunStats {
    const char *path;
    int hip_calls, launches, allocs, bad_index;
    long bytes_in, bytes_out;
};

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

/* The record is poisoned before any stage runs, on both arms: the host body does it inside
 * `m2_rt_run`, the device needs it as its own launch because a `hipMemset` cannot write the
 * 0x7FC00000 pattern one float at a time. An unwritten slot therefore reads NaN on the device too,
 * which is what makes the census leg of the controls mean the same thing on both paths. */
__global__ void m2_rt_k0_poison(float *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M2_RT_FLOATS) out[i] = mr_bits_to_f32(MR_NAN_BITS);
}

__global__ void m2_rt_k1(const M2RtPlan p, const uint8_t *slab, float *out, M2RtCounters *ctr,
                         int *bad) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    struct M2RtCounters t;
    int local_bad = 0;
    t.reads_w = t.reads_x = t.items1 = t.items2 = 0;
    if (e >= p.rows) return;                       /* the grid is derived, so the tail is a no-op */
    m2_rt_item1(&p, slab, out, e, &t, &local_bad);
    if (ctr) {
        atomicAdd(&ctr->reads_w, t.reads_w);
        atomicAdd(&ctr->reads_x, t.reads_x);
        atomicAdd(&ctr->items1, t.items1);
    }
    if (local_bad && bad) atomicExch(bad, 1);
}

__global__ void m2_rt_k2(const M2RtPlan p, float *out, M2RtCounters *ctr) {
    if (threadIdx.x || blockIdx.x) return;         /* the router's serial point, one thread */
    m2_rt_item2(&p, out + M2_RT_OFF_LOGITS, out + M2_RT_OFF_LOGITS_F32, out, ctr);
}

static int m2_rt_device_run(const struct M2RtPlan *plan, const uint8_t *slab, long slab_bytes,
                            float *out, struct M2RtCounters *ctr, struct M2RtRunStats *st,
                            char *err, size_t errn) {
    uint8_t *d_slab = 0;
    float *d_out = 0;
    M2RtCounters *d_ctr = 0;
    int *d_bad = 0;
    const int b1 = ((int)plan->rows + M2_RT_THREADS - 1) / M2_RT_THREADS;
    const int b0 = (M2_RT_FLOATS + M2_RT_THREADS - 1) / M2_RT_THREADS;

    memset(st, 0, sizeof(*st));
    st->path = "device";
    memset(ctr, 0, sizeof(*ctr));
    g_hip_calls = g_launches = g_allocs = 0;
    g_bytes_in = g_bytes_out = 0;

    M2_CK(hipMalloc((void **)&d_slab, (size_t)slab_bytes), "alloc-slab");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_out, sizeof(float) * M2_RT_FLOATS), "alloc-record");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_ctr, sizeof(M2RtCounters) + sizeof(int)), "alloc-counters");
    ++g_allocs;
    d_bad = (int *)(d_ctr + 1);
    M2_CK(hipMemcpy(d_slab, slab, (size_t)slab_bytes, hipMemcpyHostToDevice), "h2d-slab");
    g_bytes_in += slab_bytes;
    M2_CK(hipMemset(d_ctr, 0, sizeof(M2RtCounters) + sizeof(int)), "memset-counters");

    m2_rt_k0_poison<<<b0, M2_RT_THREADS>>>(d_out);
    M2_CK(hipGetLastError(), "launch-poison");
    ++g_launches;
    m2_rt_k1<<<b1, M2_RT_THREADS>>>(*plan, d_slab, d_out, d_ctr, d_bad);
    M2_CK(hipGetLastError(), "launch-stage1");
    ++g_launches;
    m2_rt_k2<<<1, 1>>>(*plan, d_out, d_ctr);
    M2_CK(hipGetLastError(), "launch-stage2");
    ++g_launches;
    M2_CK(hipDeviceSynchronize(), "sync");

    M2_CK(hipMemcpy(out, d_out, sizeof(float) * M2_RT_FLOATS, hipMemcpyDeviceToHost), "d2h-record");
    g_bytes_out += sizeof(float) * M2_RT_FLOATS;
    M2_CK(hipMemcpy(ctr, d_ctr, sizeof(M2RtCounters), hipMemcpyDeviceToHost), "d2h-counters");
    g_bytes_out += sizeof(M2RtCounters);
    {
        int bad = 0;
        M2_CK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost), "d2h-bad");
        g_bytes_out += sizeof(int);
        st->bad_index = bad;
    }
    st->hip_calls = g_hip_calls;
    st->launches = g_launches;
    st->allocs = g_allocs;
    st->bytes_in = g_bytes_in;
    st->bytes_out = g_bytes_out;
    M2_CK(hipFree(d_slab), "free-slab");
    M2_CK(hipFree(d_out), "free-record");
    M2_CK(hipFree(d_ctr), "free-counters");
    return 0;
}

#include "m2_rt_main.inc"
