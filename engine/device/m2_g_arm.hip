/* m2_g_arm.hip — the hipcc arm of the M2 routed-expert gather: gfx1151 codegen, and the only one of the
 * two binaries that can launch anything.
 *
 * TWO COMPILATIONS, ONE SOURCE. `m2_g_host.cpp` compiles the same `m2_g_kernels.h` with g++; run both on
 * one plan and compare the records byte for byte, per width path. Without `--device` this binary executes
 * the same body on the CPU and reports `hip_calls=0` — that is the compiled-pair arm, a numerics verdict
 * and not a device run.
 *
 * WITH `--device` the acts are: four `hipMalloc` (slab, record, counters+bad, plan), two H2D (the slab,
 * the plan), one `hipMemset` (counters only — the record is poisoned by a launch of its own, because
 * `hipMemset` cannot write a NaN pattern one float at a time and a memset-to-zero would make the
 * unwritten-slot census mean one thing on the host body and another on the device), FIVE launches (poison;
 * gate|up — one launch over every intermediate row of every selected expert of the row, which is
 * prediction 2's grouped shape rather than K dependent launches; the silu-product; `down` over every
 * hidden row of every selected expert; the combine), one unit read-back, one synchronize, three D2H, four
 * `hipFree`. Every one is counted in `M2GRunStats`, because the counter and not the flag is what lets the
 * driver print anything device-shaped.
 *
 * THE GROUPED SHAPE IS ONE LAUNCH PER PHASE, NOT ONE PER EXPERT. Each of the three body launches grids
 * over (slot, output element), so a token's K gathers are issued together and the machine has K independent
 * reads in flight instead of one at a time. That is the whole of what AGENTS.md prediction 2 asks to be
 * measured about the gather, and it is a fact about this launch grid and about the census's per-expert
 * bytes — never about a timing, which nothing here takes and nothing here may be inferred to exist.
 *
 * WHY FIVE LAUNCHES AND NO BARRIER. The four items are separate launches rather than phases inside one
 * kernel because `__syncthreads()` is a __device__ function and the item functions are compiled for the
 * host too (the g++ twin and this binary's own host path use the same source). A phase's inputs are the
 * previous launch's outputs, so no barrier is needed anywhere and the shared code stays callable from
 * both. The counter merge has the same shape: `atomicAdd` appears only in these `__global__` bodies, which
 * is the boundary `engine/device/m2_se_kernels.h` §counter merge already draws.
 *
 * SHAPE, NOT SPEED: one thread per output element, one dequant per MAC, no tiling, no shared-memory
 * staging, no vectorized loads. This is the shape a correctness gate can be judged on.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_g_io.h"

#define M2G_ARM_NAME "g_hip"
#define M2G_COMPILER "hipcc"
#define M2G_HAVE_HIP 1

#define M2G_THREADS 128

struct M2GRunStats {
    const char *path;
    int hip_calls, launches, allocs, bad_index;
    long bytes_in, bytes_out;
    char read_hex[129];
};

static int g_hip_calls = 0;
static int g_launches = 0;
static int g_allocs = 0;
static long g_bytes_in = 0;
static long g_bytes_out = 0;

#define M2_CK(call, tag)                                                              \
    do {                                                                              \
        const hipError_t _e = (call);                                                 \
        ++g_hip_calls;                                                                \
        if (_e != hipSuccess) {                                                       \
            snprintf(err, errn, "%s: %s -> %s", tag, #call, hipGetErrorString(_e));  \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

__global__ void m2g_k0_poison(float *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M2G_FLOATS) out[i] = mg_float_from_bits(MG_NAN_BITS);
}

/* The device twin of `m2g_item_fold`. `atomicAdd` is required here and illegal in the header: every
 * thread of a launch tallies into the same counters, and a __device__ function called from a
 * host-callable one is what the compiler rejects. */
__device__ static void m2g_fold_device(struct M2GCounters *ct, const struct M2GItem *it, int slots) {
    if (!ct || !it) return;
    atomicAdd(&ct->elts, it->elts);
    atomicAdd(&ct->route_n, it->route_n);
    atomicAdd(&ct->route_bad, it->route_bad);
    atomicAdd(&ct->note_conflict, it->note_conflict);
    atomicAdd(&ct->slots, (uint32_t)slots);
    if (it->n_note) {
        const int e = it->note_expert;
        atomicAdd(&ct->units, 1u);
        atomicAdd(&ct->units_by_expert[e], 1u);
        atomicAdd(&ct->bytes_by_expert[e], it->note_bytes);
        if (it->note_whole) atomicAdd(&ct->whole_reads, 1u);
    }
}

/* Phase 1: one thread per (slot, half, intermediate row). The grid covers every selected expert inside
 * one launch, and the census comes out per expert_index because the plan named the expert each thread's
 * bytes belong to. `slots` is counted by the thread that owns a slot's first row, so it counts slots
 * gathered rather than threads that ran. */
__global__ void m2g_k1_gateup(const struct M2GPlan *p, const char *slab, float *out,
                              struct M2GCounters *ctr, int *bad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = p->slots * 2 * MG_I;
    struct M2GItem it;
    int slot, which, row, local_bad = 0;
    if (idx >= n) return;                            /* the grid is derived, so the tail is a no-op */
    slot = idx / (2 * MG_I);
    which = (idx / MG_I) % 2;
    row = idx % MG_I;
    m2g_item_init(&it);
    m2g_item1(p, slab, out, slot, which, row, &it, &local_bad);
    if (ctr) {
        /* `slots` is counted once per slot by one thread; the whole-tensor note is per unit and comes from
         * `m2g_item_note`, so the counter counts units rather than a per-launch constant. */
        if (row == 0 && which == 0) atomicAdd(&ctr->slots, 1u);
        m2g_fold_device(ctr, &it, 0);
    }
    if (local_bad && bad) atomicAdd(bad, local_bad);
}

/* Phase 2 and 3 are separate launches, not barriers: a phase's inputs are the previous launch's outputs. */
__global__ void m2g_k2_silu(const struct M2GPlan *p, const float *rec, float *out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p->slots * MG_I) return;
    m2g_item2(p, rec, out, idx / MG_I, idx % MG_I);
}

__global__ void m2g_k3_down(const struct M2GPlan *p, const char *slab, const float *rec, float *out,
                            struct M2GCounters *ctr, int *bad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    struct M2GItem it;
    int local_bad = 0;
    if (idx >= p->slots * MG_H || p->fault_skip_raw) return;
    m2g_item_init(&it);
    m2g_item3(p, slab, rec, out, idx / MG_H, idx % MG_H, &it, &local_bad);
    if (ctr) m2g_fold_device(ctr, &it, 0);
    if (local_bad && bad) atomicAdd(bad, local_bad);
}

/* The combine, one hidden column per thread. Each thread derives the ascending-id walk itself with the
 * prototype's `mg_combine_order` (K=8), which costs less than a shared-memory staging round would and
 * keeps the ordering question in exactly one function on both paths. */
__global__ void m2g_k4_combine(const struct M2GPlan *p, const char *slab, const float *rec, float *out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= MG_H || p->fault_skip_routed) return;
    m2g_item4(p, slab, rec, out, k);
}

static int m2g_device_run(const struct M2GPlan *plan, const char *slab, long slab_bytes, float *out,
                          struct M2GCounters *ctr, struct M2GRunStats *st,
                          const struct M2GPlanFile *pf, char *err, size_t errn) {
    char *d_slab = 0;
    float *d_out = 0;
    struct M2GCounters *d_ctr = 0;
    struct M2GPlan *d_plan = 0;
    int *d_bad = 0;
    int bad = 0;
    const int b0 = (M2G_FLOATS + M2G_THREADS - 1) / M2G_THREADS;

    memset(st, 0, sizeof(*st));
    st->path = "device";
    memset(ctr, 0, sizeof(*ctr));
    g_hip_calls = g_launches = g_allocs = 0;
    g_bytes_in = g_bytes_out = 0;

    if (pf && !strcmp(pf->fault_word, "alloc-oversize")) {
        /* The leg a device window exists to witness: an allocation no pool can satisfy must end the run
         * with a named failure. The host twin refuses this fault with `fault-device-only` instead, and
         * neither path is allowed to answer with the other's numbers. */
        void *huge = 0;
        const hipError_t e = hipMalloc((void **)&huge, (size_t)1 << 40);
        ++g_hip_calls;
        st->hip_calls = g_hip_calls;
        if (e == hipSuccess) {
            hipFree(huge);
            snprintf(err, errn, "alloc-oversize: a 1 TiB hipMalloc SUCCEEDED, so this leg can no longer "
                   "prove anything and the window must stop and say so");
            return 1;
        }
        snprintf(err, errn, "alloc-oversize refused by the allocator (hipMalloc of 1 TiB failed), so no "
               "launch was issued and no host substitution is reachable");
        return 1;
    }

    M2_CK(hipMalloc((void **)&d_slab, (size_t)slab_bytes), "alloc-slab");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_out, sizeof(float) * M2G_FLOATS), "alloc-record");
    ++g_allocs;
    /* One allocation for the counters and the bad-index flag: `d_bad` is the word after the struct, and
     * both are zeroed by the same memset, so there is one place to look for whether the tail was cleared. */
    M2_CK(hipMalloc((void **)&d_ctr, sizeof(struct M2GCounters) + sizeof(int)), "alloc-counters");
    ++g_allocs;
    M2_CK(hipMalloc((void **)&d_plan, sizeof(struct M2GPlan)), "alloc-plan");
    ++g_allocs;
    d_bad = (int *)(d_ctr + 1);
    M2_CK(hipMemcpy(d_slab, slab, (size_t)slab_bytes, hipMemcpyHostToDevice), "h2d-slab");
    g_bytes_in += slab_bytes;
    M2_CK(hipMemcpy(d_plan, plan, sizeof(struct M2GPlan), hipMemcpyHostToDevice), "h2d-plan");
    g_bytes_in += (long)sizeof(struct M2GPlan);
    M2_CK(hipMemset(d_ctr, 0, sizeof(struct M2GCounters) + sizeof(int)), "memset-counters");

    if (plan->corrupt_off >= 0) {
        /* One deliberate 1-byte H2D into the device's own copy of the slab — `stage_run`'s
         * `device-corrupt` shape. It is the only corruption a read-back can see, and the read-back below
         * is what makes it visible rather than merely asserted. */
        const uint8_t flipped = (uint8_t)(((const uint8_t *)slab)[plan->corrupt_off] ^ 0xFF);
        M2_CK(hipMemcpy((char *)d_slab + plan->corrupt_off, &flipped, 1, hipMemcpyHostToDevice),
              "h2d-corrupt");
        g_bytes_in += 1;
        m2g_hex(slab, plan->corrupt_off, 8, st->read_hex, sizeof(st->read_hex));
        printf("CORRUPT side=device off=%ld host_hex8=%s\n", plan->corrupt_off, st->read_hex);
        st->read_hex[0] = '\0';
    }

    m2g_k0_poison<<<b0, M2G_THREADS>>>(d_out);
    M2_CK(hipGetLastError(), "launch-poison");
    ++g_launches;
    m2g_k1_gateup<<<(plan->slots * 2 * MG_I + M2G_THREADS - 1) / M2G_THREADS, M2G_THREADS>>>(
        d_plan, d_slab, d_out, d_ctr, d_bad);
    M2_CK(hipGetLastError(), "launch-gateup-grouped");
    ++g_launches;
    m2g_k2_silu<<<(plan->slots * MG_I + M2G_THREADS - 1) / M2G_THREADS, M2G_THREADS>>>(d_plan, d_out,
                                                                                        d_out);
    M2_CK(hipGetLastError(), "launch-silu-multiply");
    ++g_launches;
    m2g_k3_down<<<(plan->slots * MG_H + M2G_THREADS - 1) / M2G_THREADS, M2G_THREADS>>>(
        d_plan, d_slab, d_out, d_out, d_ctr, d_bad);
    M2_CK(hipGetLastError(), "launch-down-grouped");
    ++g_launches;
    m2g_k4_combine<<<(MG_H + M2G_THREADS - 1) / M2G_THREADS, M2G_THREADS>>>(d_plan, d_slab, d_out,
                                                                            d_out);
    M2_CK(hipGetLastError(), "launch-combine");
    ++g_launches;
    M2_CK(hipDeviceSynchronize(), "sync");

    if (plan->read_off >= 0 && plan->read_len > 0 && plan->read_len <= 64) {
        char win[64];
        M2_CK(hipMemcpy(win, (char *)d_slab + plan->read_off, (size_t)plan->read_len,
                        hipMemcpyDeviceToHost), "d2h-unit-window");
        g_bytes_out += plan->read_len;
        {
            int i, used = 0;
            for (i = 0; i < plan->read_len; ++i)
                used += snprintf(st->read_hex + used, sizeof(st->read_hex) - (size_t)used, "%02x",
                                 (unsigned)((uint8_t)win[i]));
        }
    }

    M2_CK(hipMemcpy(out, d_out, sizeof(float) * M2G_FLOATS, hipMemcpyDeviceToHost), "d2h-record");
    g_bytes_out += (long)sizeof(float) * M2G_FLOATS;
    M2_CK(hipMemcpy(ctr, d_ctr, sizeof(struct M2GCounters), hipMemcpyDeviceToHost), "d2h-counters");
    g_bytes_out += (long)sizeof(struct M2GCounters);
    M2_CK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost), "d2h-bad");
    g_bytes_out += (long)sizeof(int);

    st->bad_index = bad;
    st->hip_calls = g_hip_calls;
    st->launches = g_launches;
    st->allocs = g_allocs;
    st->bytes_in = g_bytes_in;
    st->bytes_out = g_bytes_out;
    M2_CK(hipFree(d_slab), "free-slab");
    M2_CK(hipFree(d_out), "free-record");
    M2_CK(hipFree(d_ctr), "free-counters");
    M2_CK(hipFree(d_plan), "free-plan");
    return 0;
}

#include "m2_g_main.inc"
