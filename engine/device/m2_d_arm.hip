/* m2_d_arm.hip — the hipcc arm of the M2 GDN recurrent stage: gfx1151 codegen, and the only one of
 * the two binaries that can launch anything.
 *
 * TWO IMPLEMENTATIONS, ONE ARITHMETIC. `m2_d_host.cpp` compiles the same `m2_d_kernels.h` with g++ and
 * drives the items serially with `m2d_run`; this binary reaches the same items with seven launches and
 * never calls `m2d_run`. Running both on one host-visible plan and comparing the records byte for byte
 * is the compiled pair; a difference is a codegen finding at zero tolerance, not a band.
 *
 * WITH `--device` the acts are: four `hipMalloc` (slab, record, scratch, counters), one H2D of the slab,
 * one `hipMemset` (the counters only), eight launches (poison, then enter/carry/kvmem/write/out/norm/
 * outproj), two or three D2H, one synchronize, four `hipFree`. Every one of them is counted in
 * `M2DRunStats`, because the counter — not the flag — is what lets the driver print anything
 * device-shaped (`engine/block_device.py`'s `device-claims-come-from-device-only`).
 *
 * ONE LAUNCH PER STAGE, NOT ONE PER HEAD. §04b.1's five steps are sequentially dependent (step 3 needs
 * every thread's `kv_mem` before step 4 can write), so the stage boundary is a real barrier and the
 * launch is how a device gets one. Within a stage the items are independent: 4096 (head, row) items for
 * carry, 4096 (head, column) items for kvmem and out, 2048 out_proj rows — so the largest grid here is
 * 4096 threads and none of these launches is a serial loop in disguise. AGENTS.md prediction 2's point
 * about dependent reads applies to the MoE gather; here the dependency is between stages, and the shape
 * that respects it is grouped launches per stage, which is what this file does.
 *
 * SHAPE, NOT SPEED. No timing is taken anywhere in this file, and nothing reads a clock.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_d_io.h"

#define M2D_ARM_NAME "d_hip"
#define M2D_COMPILER "hipcc"
#define M2D_HAVE_HIP 1
#define M2D_THREADS 128

struct M2DRunStats {
    const char *path;
    int hip_calls, launches, allocs;
    long bytes_in, bytes_out;
    char read_hex[129];
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

/* The record is poisoned before any stage runs, on both arms: the host body does it inside `m2d_run`,
 * the device needs it as its own launch because a `hipMemset` cannot write the 0x7FC00000 pattern one
 * float at a time. An unwritten slot therefore reads NaN on the device too, which is what makes the
 * unwritten-slot census mean the same thing on both paths. */
__global__ void m2d_k0_poison(float *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M2D_FLOATS) out[i] = m2d_nan();
}

/* Fold one thread's counters into the shared block. Modular sums, so the fold order does not matter and
 * the device's atomic order cannot disagree with the host's serial `+=` (see M2DCounters). */
GDNDAY_DEV static inline void m2d_atomic_fold(M2DCounters *ctr, const M2DCounters *t) {
    int i;
    if (!ctr) return;
    for (i = 0; i < M2D_IT_N; ++i) if (t->items[i]) atomicAdd(&ctr->items[i], t->items[i]);
    if (t->items_total) atomicAdd(&ctr->items_total, t->items_total);
    if (t->reads_q) atomicAdd(&ctr->reads_q, t->reads_q);
    if (t->reads_k) atomicAdd(&ctr->reads_k, t->reads_k);
    if (t->reads_v) atomicAdd(&ctr->reads_v, t->reads_v);
    if (t->reads_state) atomicAdd(&ctr->reads_state, t->reads_state);
    if (t->writes_state) atomicAdd(&ctr->writes_state, t->writes_state);
    if (t->state_nonzero) atomicAdd(&ctr->state_nonzero, t->state_nonzero);
    if (t->state_sum) atomicAdd(&ctr->state_sum, t->state_sum);
    if (t->state_sumidx) atomicAdd(&ctr->state_sumidx, t->state_sumidx);
    if (t->out_elems) atomicAdd(&ctr->out_elems, t->out_elems);
    if (t->outproj_reads) atomicAdd(&ctr->outproj_reads, t->outproj_reads);
}

__global__ void m2d_k_enter(M2DPlan p, const uint8_t *slab, float *rec, M2DScratch sc,
                            M2DCounters *ctr) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (h >= GDN_H_V) return;
    memset(&t, 0, sizeof(t));
    m2d_item_enter(&p, slab, rec, &sc, h, &t);
    m2d_atomic_fold(ctr, &t);
}

/* The carry stage carries `zero_in`, the non-carried fault, so the device path can run the same leg the
 * host path runs and neither has to fake it. */
__global__ void m2d_k_carry(M2DPlan p, const uint8_t *slab, float *rec, M2DCounters *ctr, int zero_in) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (i >= GDN_H_V * GDN_D_K) return;
    memset(&t, 0, sizeof(t));
    m2d_item_carry(&p, slab, rec, i / GDN_D_K, i % GDN_D_K, zero_in, &t);
    m2d_atomic_fold(ctr, &t);
}

__global__ void m2d_k_kvmem(M2DPlan p, const float *rec, M2DScratch sc, M2DCounters *ctr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (i >= GDN_H_V * GDN_D_V) return;
    memset(&t, 0, sizeof(t));
    m2d_item_kvmem(&p, rec, &sc, i / GDN_D_V, i % GDN_D_V, &t);
    m2d_atomic_fold(ctr, &t);
}

__global__ void m2d_k_write(M2DPlan p, float *rec, const M2DScratch sc, M2DCounters *ctr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (i >= GDN_H_V * GDN_D_K) return;
    memset(&t, 0, sizeof(t));
    m2d_item_write(rec, &sc, i / GDN_D_K, i % GDN_D_K, &t);
    m2d_atomic_fold(ctr, &t);
}

__global__ void m2d_k_out(M2DPlan p, const float *rec, float *out, M2DCounters *ctr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (i >= GDN_H_V * GDN_D_V) return;
    memset(&t, 0, sizeof(t));
    m2d_item_out(&p, rec, out, i / GDN_D_V, i % GDN_D_V, &t);
    m2d_atomic_fold(ctr, &t);
}

__global__ void m2d_k_norm(M2DPlan p, const uint8_t *slab, const float *rec, float *out,
                           M2DCounters *ctr) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (h >= GDN_H_V) return;
    memset(&t, 0, sizeof(t));
    m2d_item_norm(&p, slab, rec, out, h, &t);
    m2d_atomic_fold(ctr, &t);
}

__global__ void m2d_k_outproj(M2DPlan p, const uint8_t *slab, const float *rec, float *out,
                              M2DCounters *ctr) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    M2DCounters t;
    if (r >= GDN_H) return;
    memset(&t, 0, sizeof(t));
    m2d_item_outproj(&p, slab, rec, out, r, &t);
    m2d_atomic_fold(ctr, &t);
}

#define M2D_GRID(n) (((int)(n) + M2D_THREADS - 1) / M2D_THREADS)

static int m2d_device_run(const struct M2DPlan *plan, const uint8_t *slab, long slab_bytes,
                          float *out, struct M2DCounters *ctr, struct M2DRunStats *st,
                          char *err, size_t errn) {
    uint8_t *d_slab = 0;
    float *d_out = 0, *d_scratch = 0;
    M2DCounters *d_ctr = 0;
    struct M2DScratch sc;
    const int zero_in = (plan->fault == M2D_FAULT_ZERO_STATE);

    memset(st, 0, sizeof(*st));
    st->path = "device";
    memset(ctr, 0, sizeof(*ctr));
    g_hip_calls = g_launches = g_allocs = 0;
    g_bytes_in = g_bytes_out = 0;

    /* The witnessed-capable-of-failing leg: an allocation no pool can satisfy has to fail here, and the
     * driver requires that failure. `pitfalls/10` §"~32–34 GiB pool, and nothing moves it" is why a
     * 1 TiB request is refused rather than silently served from somewhere else. */
    if (plan->fault == M2D_FAULT_ALLOC_OVER) {
        void *over = 0;
        const hipError_t e = hipMalloc(&over, (size_t)1 << 40);
        ++g_hip_calls;
        if (e == hipSuccess) {
            const hipError_t freed = hipFree(over);
            (void)freed;                 /* the leg's answer is the refusal, not the cleanup */
            snprintf(err, errn, "alloc-oversize: a 1 TiB hipMalloc SUCCEEDED, so this leg can no "
                                "longer show that an impossible allocation is refused");
            return 1;
        }
        snprintf(err, errn, "alloc-oversize refused by the allocator (hipMalloc of 1 TiB failed: %s), "
                            "so no stage ran and nothing is reported as a result",
                 hipGetErrorString(e));
        return 1;
    }

    M2_CK(hipMalloc((void **)&d_slab, (size_t)slab_bytes), "alloc-slab");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_out, sizeof(float) * M2D_FLOATS), "alloc-record");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_scratch, sizeof(float) * (size_t)m2d_scratch_floats()),
          "alloc-scratch");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_ctr, sizeof(M2DCounters)), "alloc-counters");
    ++g_allocs;
    sc.v_w = d_scratch;
    sc.kv = d_scratch + GDN_H_V * GDN_D_V;
    sc.delta = sc.kv + GDN_H_V * GDN_D_V;
    sc.beta = sc.delta + GDN_H_V * GDN_D_V;
    M2_CK(hipMemcpy(d_slab, slab, (size_t)slab_bytes, hipMemcpyHostToDevice), "h2d-slab");
    g_bytes_in += slab_bytes;
    M2_CK(hipMemset(d_ctr, 0, sizeof(M2DCounters)), "memset-counters");

    m2d_k0_poison<<<M2D_GRID(M2D_FLOATS), M2D_THREADS>>>(d_out);
    M2_CK(hipGetLastError(), "launch-poison");
    ++g_launches;
    m2d_k_enter<<<M2D_GRID(GDN_H_V), M2D_THREADS>>>(*plan, d_slab, d_out, sc, d_ctr);
    M2_CK(hipGetLastError(), "launch-enter");
    ++g_launches;
    if (plan->fault != M2D_FAULT_SKIP_CARRY) {
        m2d_k_carry<<<M2D_GRID(GDN_H_V * GDN_D_K), M2D_THREADS>>>(*plan, d_slab, d_out, d_ctr,
                                                                  zero_in);
        M2_CK(hipGetLastError(), "launch-carry");
        ++g_launches;
    }
    m2d_k_kvmem<<<M2D_GRID(GDN_H_V * GDN_D_V), M2D_THREADS>>>(*plan, d_out, sc, d_ctr);
    M2_CK(hipGetLastError(), "launch-kvmem");
    ++g_launches;
    m2d_k_write<<<M2D_GRID(GDN_H_V * GDN_D_K), M2D_THREADS>>>(*plan, d_out, sc, d_ctr);
    M2_CK(hipGetLastError(), "launch-write");
    ++g_launches;
    if (plan->fault != M2D_FAULT_SKIP_OUT) {
        m2d_k_out<<<M2D_GRID(GDN_H_V * GDN_D_V), M2D_THREADS>>>(*plan, d_out, d_out, d_ctr);
        M2_CK(hipGetLastError(), "launch-out");
        ++g_launches;
    }
    m2d_k_norm<<<M2D_GRID(GDN_H_V), M2D_THREADS>>>(*plan, d_slab, d_out, d_out, d_ctr);
    M2_CK(hipGetLastError(), "launch-norm");
    ++g_launches;
    m2d_k_outproj<<<M2D_GRID(GDN_H), M2D_THREADS>>>(*plan, d_slab, d_out, d_out, d_ctr);
    M2_CK(hipGetLastError(), "launch-outproj");
    ++g_launches;
    M2_CK(hipDeviceSynchronize(), "sync");

    M2_CK(hipMemcpy(out, d_out, sizeof(float) * M2D_FLOATS, hipMemcpyDeviceToHost), "d2h-record");
    g_bytes_out += sizeof(float) * M2D_FLOATS;
    M2_CK(hipMemcpy(ctr, d_ctr, sizeof(M2DCounters), hipMemcpyDeviceToHost), "d2h-counters");
    g_bytes_out += sizeof(M2DCounters);
    if (plan->read_off >= 0 && plan->read_len > 0) {
        /* The read-back leg binds what the DEVICE holds at the state's offset to what the file holds
         * there, which is the device-side half of "the arm read the state it claims to read". */
        uint8_t back[32];
        const int n = plan->read_len < 32 ? plan->read_len : 32;
        M2_CK(hipMemcpy(back, d_slab + plan->read_off, (size_t)n, hipMemcpyDeviceToHost),
              "d2h-state-readback");
        g_bytes_out += n;
        m2d_hex(back, 0, n, st->read_hex, sizeof(st->read_hex));
    }
    st->hip_calls = g_hip_calls;
    st->launches = g_launches;
    st->allocs = g_allocs;
    st->bytes_in = g_bytes_in;
    st->bytes_out = g_bytes_out;
    M2_CK(hipFree(d_slab), "free-slab");
    M2_CK(hipFree(d_out), "free-record");
    M2_CK(hipFree(d_scratch), "free-scratch");
    M2_CK(hipFree(d_ctr), "free-counters");
    return 0;
}

#include "m2_d_main.inc"
