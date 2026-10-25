/* q4k_dequant.hip — device-side Q4_K / Q8_0 dequantization, plus the driver that feeds it.
 *
 * PLAN.md §"Step 2 and beyond", first line: the Q4_K/Q8_0 dequant gate on device, graduating from
 * step 1.5. This file is the implementation under test. Its reference is `tools/gguf_reader.py`
 * (the audited CPU reader); the comparison, the verdicts and the must-catch sets live in `gate.py`,
 * and `gate` is what acquires the GPU lock. Nothing in this file decides anything: it decodes bytes
 * and reports checksums and counts.
 *
 * Correctness only. This driver takes no timers, runs no warm loop, no burst and no best-of-N, and
 * prints no ms / GB/s / tok/s figure (`pitfalls/00-measurement.md`; AGENTS.md §"GPU execution is
 * serialized"). The first timing of this kernel is the main agent's, in its own window.
 *
 * Build:  ./build.sh           (hipcc, gfx1151, out-of-tree objects under build/)
 *         ./build.sh --check   (compile-only; no device is touched)
 * Run:   ./gate                (acquires tools/gpu_lock.sh, then gate.py, then this binary)
 *         ./q4k_dequant --selftest          host-only self-tests, no device, no GGUF
 *         ./q4k_dequant --check             parse the plan, verify structure, no device
 */
#include <hip/hip_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "q4k_dequant_kernels.h"

#define Q4KD_PLAN_FIELDS_U 18     /* 'U' + 17 values */
#define Q4KD_PLAN_FIELDS_C 6      /* 'C' + 5 values  */
#define Q4KD_BLOCK 256

static int g_failures = 0;

static void fail(const char *fmt) {
    ++g_failures;
    printf("FAIL %s\n", fmt);
}

#define CK(call)                                                              \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            printf("FAIL hip: %s = %s (%s)\n", #call, hipGetErrorString(_e), \
                   strerror(errno));                                          \
            return 2;                                                         \
        }                                                                     \
    } while (0)

/* --------------------------------------------------------------------------------------------- */
/* the kernel                                                                                     */
/* --------------------------------------------------------------------------------------------- */
__global__ void q4k_dequant_kernel(const Q4kUnit *__restrict__ units, const Q4kCase *__restrict__ cases,
                                   const int *__restrict__ gidx, int n_cases,
                                   const uint8_t *__restrict__ slab, float *__restrict__ out,
                                   float *__restrict__ csum, int *__restrict__ cbad,
                                   int *__restrict__ cnan, int mode) {
    __shared__ Q4kCase sc;
    __shared__ Q4kUnit su;
    __shared__ uint8_t sblk[Q4KD_MAX_BLOCK_BYTES];
    __shared__ float sdst[Q4KD_MAX_BLOCK_ELTS];

    for (int c = blockIdx.x; c < n_cases; c += gridDim.x) {
        if (threadIdx.x == 0) {
            sc = cases[c];
            su = units[sc.unit];
        }
        __syncthreads();                       /* sc and su visible to the whole block */
        for (int i = threadIdx.x; i < Q4KD_MAX_BLOCK_ELTS; i += blockDim.x) sdst[i] = 0.0f;
        /* Every branch below is block-uniform (it reads only sc/su), so the barriers inside it are
         * executed by all threads of the block. A divergent __syncthreads() is a hang, and on this
         * box a hang presents as a busy stream sync with no output (pitfalls/50). */
        int32_t src = su.src_off;
        const int bad = q4kd_resolve_src(units, sc.unit, &su, mode, &src);
        __syncthreads();
        if (!bad) {
            for (int i = threadIdx.x; i < su.block_bytes; i += blockDim.x) sblk[i] = slab[src + i];
        }
        __syncthreads();
        if (!bad && threadIdx.x < 32) {
            if (su.kind == Q4KD_KIND_Q4K) q4k_decode_lane(sblk, mode, (int)threadIdx.x, sdst);
            else q8_0_decode_lane(sblk, mode, (int)threadIdx.x, sdst);
        }
        __syncthreads();                       /* sdst complete */
        const int g = gidx[c];
        float sum = 0.0f;
        int nbad = 0, nnan = 0;
        for (int i = threadIdx.x; i < su.n_elts; i += blockDim.x) {
            const float v = sdst[i];
            out[sc.dst_off + i] = v;
            sum += v;
            const uint32_t b = __float_as_uint(v);
            if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) nnan = 1;
            if (sc.cmp_off >= 0 && !q4kd_word_eq(q4kd_as_float_bits(out[sc.cmp_off + i]), b)) ++nbad;
        }
        __syncthreads();
        /* Every thread owns a disjoint slice of the element range, so the reduction IS a per-thread
         * atomicAdd. Guarding these three adds with threadIdx.x == 0 (as this line used to) throws
         * away every partial but thread 0's: the driver then reported 1 bad element where all 32
         * differed, and a cell sum that was only element 0. gate.py recomputes both from the bytes,
         * which is how it was caught rather than believed. */
        if (sum != 0.0f) atomicAdd(&csum[g], sum);      /* a zero partial cannot change a sum */
        if (nbad) atomicAdd(&cbad[g], nbad);
        /* cnan is a per-case FLAG, not a count: one thread that saw a NaN sets it, and atomicMax (not
         * atomicAdd) keeps it 1 however many threads saw one. gate.py compares this column against its
         * own per-case scan, so Add-vs-Max is a disagreement it would report. */
        if (nnan) atomicMax(&cnan[g], nnan);
        __syncthreads();                       /* sblk/sdst may be reused by the next case */
    }
}

/* --------------------------------------------------------------------------------------------- */
/* plan parsing                                                                                   */
/* --------------------------------------------------------------------------------------------- */
struct Plan {
    std::vector<Q4kUnit> units;
    std::vector<Q4kCase> cases;
    std::vector<int> mode_of_case;
};

/* tab-separated fields, no collapsing, trailing newline stripped */
static int split(char *line, char **out, int max) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
    int c = 0;
    char *p = line;
    while (c < max) {
        out[c++] = p;
        char *t = strchr(p, '\t');
        if (!t) break;
        *t = '\0';
        p = t + 1;
    }
    return c;
}

static int read_plan(const char *path, Plan *plan, int *lu, int *lc) {
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        printf("FAIL cannot open plan %s: %s\n", path, strerror(errno));
        return 1;
    }
    char line[4096];
    int bad = 0;
    while (fgets(line, sizeof(line), fh)) {
        char *f[Q4KD_PLAN_FIELDS_U + 2];
        int nf = split(line, f, Q4KD_PLAN_FIELDS_U + 2);
        if (nf == 0 || f[0][0] == '#' || f[0][0] == '\n') continue;
        if (strcmp(f[0], "U") == 0) {
            if (nf < Q4KD_PLAN_FIELDS_U) {
                printf("FAIL unit row %d has %d fields, expected %d\n", *lu, nf,
                       Q4KD_PLAN_FIELDS_U);
                ++bad;
                continue;
            }
            Q4kUnit u;
            long v[17];
            for (int i = 0; i < 17; ++i) v[i] = strtol(f[1 + i], NULL, 10);
            u.kind = (int32_t)v[0];
            u.src_off = (int32_t)v[1];
            u.slot_bytes = (int32_t)v[2];
            u.dst_off = (int32_t)v[3];
            u.n_elts = (int32_t)v[4];
            u.n_blocks = (int32_t)v[5];
            u.block_bytes = (int32_t)v[6];
            u.nb_total = (int32_t)v[7];
            u.nb_per_row = (int32_t)v[8];
            u.n_rows = (int32_t)v[9];
            u.n_outer = (int32_t)v[10];
            u.grp_u0 = (int32_t)v[11];
            u.grp_u1 = (int32_t)v[12];
            u.xpose_ok = (int32_t)v[13];
            u.block_index = (int32_t)v[14];
            u.tensor_index = (int32_t)v[15];
            u.cls = (int32_t)v[16];
            plan->units.push_back(u);
            ++*lu;
        } else if (strcmp(f[0], "C") == 0) {
            if (nf < Q4KD_PLAN_FIELDS_C) {
                printf("FAIL case row %d has %d fields, expected %d\n", *lc, nf,
                       Q4KD_PLAN_FIELDS_C);
                ++bad;
                continue;
            }
            Q4kCase c;
            c.unit = (int32_t)strtol(f[2], NULL, 10);
            c.mode = (int32_t)strtol(f[3], NULL, 10);
            c.dst_off = (int32_t)strtol(f[4], NULL, 10);
            c.cmp_off = (int32_t)strtol(f[5], NULL, 10);
            plan->cases.push_back(c);
            plan->mode_of_case.push_back((int)strtol(f[1], NULL, 10));
            ++*lc;
        } else {
            printf("FAIL plan row %d has unknown tag '%s'\n", *lu + *lc, f[0]);
            ++bad;
        }
    }
    fclose(fh);
    if (bad) {
        printf("FAIL plan %s: %d malformed rows\n", path, bad);
        return 1;
    }
    return 0;
}

/* Structural verification of the plan. Every one of these is a check that can fire, and gate.py
 * asserts the number of checks it expected to see run. */
static int verify_plan(const Plan &plan, int expect_units, int expect_cases, long long slab_bytes,
                       long long out_floats) {
    int checks = 0;
    if (expect_units >= 0 && (int)plan.units.size() != expect_units) {
        printf("FAIL plan unit count %zu != expected %d\n", plan.units.size(), expect_units);
        ++g_failures;
    }
    if (expect_cases >= 0 && (int)plan.cases.size() != expect_cases) {
        printf("FAIL plan case count %zu != expected %d\n", plan.cases.size(), expect_cases);
        ++g_failures;
    }
    ++checks;
    ++checks;
    std::vector<int> mode0(plan.units.size(), 0);
    std::vector<int> ncase(plan.units.size(), 0);
    for (size_t i = 0; i < plan.units.size(); ++i) {
        const Q4kUnit &u = plan.units[i];
        const int bb = u.kind == Q4KD_KIND_Q4K ? Q4KD_Q4K_BYTES : Q4KD_Q8_0_BYTES;
        if (u.kind != Q4KD_KIND_Q4K && u.kind != Q4KD_KIND_Q8_0) {
            printf("FAIL unit %zu kind %d\n", i, u.kind);
            ++g_failures;
        }
        if (u.block_bytes != bb) {
            printf("FAIL unit %zu block_bytes %d != %d for kind %d\n", i, u.block_bytes, bb,
                   u.kind);
            ++g_failures;
        }
        if (u.slot_bytes != bb + 1) {
            printf("FAIL unit %zu slot_bytes %d != block_bytes+1\n", i, u.slot_bytes);
            ++g_failures;
        }
        if (u.n_blocks != 1) {
            printf("FAIL unit %zu n_blocks %d != 1 (a multi-block launch cannot be silently "
                   "1 block)\n", i, u.n_blocks);
            ++g_failures;
        }
        if (u.n_elts <= 0 || u.n_elts > Q4KD_MAX_BLOCK_ELTS) {
            printf("FAIL unit %zu n_elts %d out of (0,%d]\n", i, u.n_elts, Q4KD_MAX_BLOCK_ELTS);
            ++g_failures;
        }
        if ((long long)u.src_off + u.slot_bytes > slab_bytes) {
            printf("FAIL unit %zu slab read out of range\n", i);
            ++g_failures;
        }
        if ((long long)u.dst_off + u.n_elts > out_floats) {
            printf("FAIL unit %zu mode-0 write out of range\n", i);
            ++g_failures;
        }
        if (u.grp_u0 < 0 || u.grp_u1 < u.grp_u0 || u.grp_u1 >= (int)plan.units.size() ||
            u.grp_u0 > (int)i || (int)i > u.grp_u1) {
            printf("FAIL unit %zu group range [%d,%d] does not contain it\n", i, u.grp_u0,
                   u.grp_u1);
            ++g_failures;
        }
        for (int k = u.grp_u0; k <= u.grp_u1 && k < (int)plan.units.size(); ++k) {
            if (plan.units[k].tensor_index != u.tensor_index) {
                printf("FAIL unit %zu group member %d is a different tensor\n", i, k);
                ++g_failures;
                break;
            }
            if (k > u.grp_u0 && plan.units[k].block_index <= plan.units[k - 1].block_index) {
                printf("FAIL unit %zu group is not sorted by block_index (the transposed read "
                       "binary-searches it)\n", i);
                ++g_failures;
                break;
            }
        }
        ++checks;
    }
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const Q4kCase &c = plan.cases[i];
        if (c.unit < 0 || c.unit >= (int)plan.units.size()) {
            printf("FAIL case %zu names unit %d\n", i, c.unit);
            ++g_failures;
            continue;
        }
        if (c.mode < 0 || c.mode >= Q4KD_MODE_COUNT) {
            printf("FAIL case %zu mode %d out of range\n", i, c.mode);
            ++g_failures;
        }
        const Q4kUnit &u = plan.units[c.unit];
        if ((long long)c.dst_off + u.n_elts > out_floats) {
            printf("FAIL case %zu write out of range\n", i);
            ++g_failures;
        }
        if (c.mode == 0) {
            ++mode0[c.unit];
            if (c.cmp_off != -1) {
                printf("FAIL case %zu is the mode-0 case but names a comparison offset\n", i);
                ++g_failures;
            }
        } else {
            if (c.cmp_off < 0) {
                printf("FAIL case %zu is an injection but has no mode-0 offset to compare against\n",
                       i);
                ++g_failures;
            } else if ((long long)c.cmp_off + u.n_elts > out_floats) {
                printf("FAIL case %zu comparison read out of range\n", i);
                ++g_failures;
            }
        }
        ++ncase[c.unit];
        ++checks;
    }
    for (size_t i = 0; i < plan.units.size(); ++i) {
        if (mode0[i] != 1) {
            printf("FAIL unit %zu has %d mode-0 cases, expected exactly 1\n", i, mode0[i]);
            ++g_failures;
        }
        if (plan.units[i].cls != 2 && ncase[i] < 2) {
            printf("FAIL unit %zu (cls %d) has %d cases: a sampled unit must carry at least one "
                   "injection case\n", i, plan.units[i].cls, ncase[i]);
            ++g_failures;
        }
        if (plan.units[i].cls == 2 && ncase[i] != 1) {
            printf("FAIL support unit %zu carries %d cases; support blocks are slab filler only\n",
                   i, ncase[i]);
            ++g_failures;
        }
        ++checks;
    }
    printf("PLANCHECK units=%zu cases=%zu checks_run=%d\n", plan.units.size(), plan.cases.size(),
           checks);
    return g_failures ? 1 : 0;
}

/* --------------------------------------------------------------------------------------------- */
/* the driver's own self-tests: no device, no GGUF, and each one can fail                        */
/* --------------------------------------------------------------------------------------------- */
static int bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return (int)b;
}

static void expect_bits(const char *what, float got, uint32_t want) {
    uint32_t g = (uint32_t)bits(got);
    if (g != want) {
        printf("FAIL selftest %s: got bits %08x want %08x\n", what, g, want);
        ++g_failures;
    }
}

static int driver_selftest() {
    int n = 0;
    /* --- struct sizes are compile-time claims; re-assert them at run time so a stale build       */
    if (sizeof(block_q4_K) != Q4KD_Q4K_BYTES || sizeof(block_q8_0) != Q4KD_Q8_0_BYTES ||
        sizeof(block_iq4_nl) != Q4KD_Q4NL_BYTES) {
        fail("selftest block sizes drifted from the pinned header");
    }
    /* every counted check prints its own record: a check that increments the counter but prints
     * nothing makes `checks` and the parsed line count disagree, which is the vacuous-PASS shape. */
    printf("SELFTEST name=block_sizes ok=%d n=3\n", g_failures == 0 ? 1 : 0);
    ++n;
    /* --- f16 -> f32, hand-worked from IEEE binary16 ------------------------------------------- */
    expect_bits("f16 +1.0", q4kd_f16(0x3C00), 0x3F800000u);
    expect_bits("f16 -1.0", q4kd_f16(0xBC00), 0xBF800000u);
    expect_bits("f16 +0.0", q4kd_f16(0x0000), 0x00000000u);
    expect_bits("f16 -0.0", q4kd_f16(0x8000), 0x80000000u);
    expect_bits("f16 65504 (max)", q4kd_f16(0x7BFF), 0x477FE000u);
    expect_bits("f16 2^-14 (min normal)", q4kd_f16(0x0400), 0x38800000u);
    expect_bits("f16 2^-24 (min denormal)", q4kd_f16(0x0001), 0x33800000u);
    expect_bits("f16 1023*2^-24 (max denormal)", q4kd_f16(0x03FF), 0x387FC000u);
    expect_bits("f16 1024", q4kd_f16(0x6400), 0x44800000u);
    expect_bits("f16 inf", q4kd_f16(0x7C00), 0x7F800000u);
    expect_bits("f16 nan", q4kd_f16(0x7E00), 0x7FC00000u);
    /* an sNaN arrives as the oracle's canonical quiet NaN, sign kept: the payload is dropped on
     * purpose (see q4kd_f16), and host_probe's f16 table checks all 65536 patterns. */
    expect_bits("f16 sNaN 0x7C01", q4kd_f16(0x7C01), 0x7FC00000u);
    expect_bits("f16 -sNaN 0xFC01", q4kd_f16(0xFC01), 0xFFC00000u);
    printf("SELFTEST name=f16_bits ok=%d n=13\n", g_failures == 0 ? 1 : 0);
    ++n;
    /* --- get_scale_min_k4, hand-worked from the C source --------------------------------------- *
     * q = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C}
     * j<4 : sc = q[j]&63                 -> 1,2,3,4      m = q[j+4]&63 -> 5,6,7,8
     * j=4 : sc = (q[8]&0xF)|((q[0]>>6)<<4) = 9 | (0<<4)  = 9
     *       m  = (q[8]>>4)|( (q[4]>>6)<<4) = 0 | (0<<4)  = 0
     * j=5 : sc = (q[9]&0xF)|((q[1]>>6)<<4) = 10          m = (q[9]>>4)|((q[5]>>6)<<4) = 0
     * j=6 : sc = (q[10]&0xF)|((q[2]>>6)<<4)= 11          m = (q[10]>>4)|((q[6]>>6)<<4)= 0
     * j=7 : sc = (q[11]&0xF)|((q[3]>>6)<<4)= 12          m = (q[11]>>4)|((q[7]>>6)<<4)= 0
     * Then with the top two bits set: q[0]=0x41 (>>6 == 1) adds 16 to sc[4]; q[4]=0x45 adds 16 to
     * m[4]; q[3]=0x84 (>>6 == 2) adds 32 to sc[7].                                            */
    {
        uint8_t q[12] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                         0x09, 0x0A, 0x0B, 0x0C};
        const uint32_t wsc[8] = {1, 2, 3, 4, 9, 10, 11, 12};
        const uint32_t wmg[8] = {5, 6, 7, 8, 0, 0, 0, 0};
        for (int j = 0; j < 8; ++j) {
            uint32_t sc = 999, mg = 999;
            q4kd_scale_min(j, q, &sc, &mg);
            if (sc != wsc[j] || mg != wmg[j]) {
                printf("FAIL selftest scale_min j=%d got (%u,%u) want (%u,%u)\n", j, sc, mg,
                       wsc[j], wmg[j]);
                ++g_failures;
            }
        }
        q[0] = 0x41;   /* bits 6-7 set -> +16 into sc[4] */
        q[4] = 0x45;   /* bits 6-7 set -> +16 into m[4]  */
        q[3] = 0x84;   /* bits 7-6 = 2 -> +32 into sc[7] */
        uint32_t sc = 0, mg = 0;
        q4kd_scale_min(4, q, &sc, &mg);
        if (sc != 9 + 16 || mg != 16) {
            printf("FAIL selftest scale_min high bits j=4 got (%u,%u) want (25,16)\n", sc, mg);
            ++g_failures;
        }
        q4kd_scale_min(7, q, &sc, &mg);
        if (sc != 12 + 32 || mg != 0) {
            printf("FAIL selftest scale_min high bits j=7 got (%u,%u) want (44,0)\n", sc, mg);
            ++g_failures;
        }
        /* and the naive 6-bit read must NOT reproduce the correct route on this vector, or the
         * must-catch control that depends on it is decoration. */
        uint32_t nsc = 0, nmg = 0;
        int differs = 0;
        for (int j = 0; j < 8; ++j) {
            q4kd_scale_min_naive(j, q, &nsc, &nmg);
            uint32_t csc = 0, cmg = 0;
            q4kd_scale_min(j, q, &csc, &cmg);
            if (nsc != csc || nmg != cmg) differs = 1;
        }
        if (!differs) fail("selftest naive scale read is identical to the correct one");
        printf("SELFTEST name=get_scale_min_k4 ok=%d n=10\n", g_failures == 0 ? 1 : 0);
    }
    ++n;
    /* --- Q4_K, hand-worked: d = 1.0, dmin = 1.0, every 6-bit field 63 -> scale 63, min 63;
     *     qs all 0x00 gives q=0 -> value -63; qs all 0xFF gives q=15 -> 63*15 - 63 = 882.       */
    {
        uint8_t blk[Q4KD_Q4K_BYTES];
        float y[Q4KD_MAX_BLOCK_ELTS];
        memset(blk, 0, sizeof(blk));
        blk[0] = 0x00; blk[1] = 0x3C;         /* d    = 1.0 */
        blk[2] = 0x00; blk[3] = 0x3C;         /* dmin = 1.0 */
        memset(blk + 4, 0xFF, K_SCALE_SIZE);  /* every sc and mg = 63 */
        memset(blk + 16, 0x00, QK_K / 2);
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, Q4KD_KIND_Q4K, Q4KD_MODE_CORRECT, y);
        /* -63 = 1.96875 * 2^5 with the sign set; the hex is the IEEE-754 encoding of the value in
         * the label, not a number the kernel was asked to reproduce. */
        for (int i = 0; i < QK_K; ++i) expect_bits("q4k q=0 -> -63", y[i], 0xC27C0000u);
        memset(blk + 16, 0xFF, QK_K / 2);
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, Q4KD_KIND_Q4K, Q4KD_MODE_CORRECT, y);
        for (int i = 0; i < QK_K; ++i) expect_bits("q4k q=15 -> 882", y[i], 0x445C8000u);
        /* drop_min: same bytes, the -dmin*mg term gone -> 63*15 = 945 */
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, Q4KD_KIND_Q4K, Q4KD_MODE_Q4K_DROP_MIN, y);
        for (int i = 0; i < QK_K; ++i) expect_bits("q4k drop_min -> 945", y[i], 0x446C4000u);
        /* and the two must differ, or the drop_min control is decoration */
        q4kd_decode_block(blk, Q4KD_KIND_Q4K, Q4KD_MODE_CORRECT, y);
        if (bits(y[0]) == (int)0x446C4000u) fail("selftest drop_min produced the correct answer");
        printf("SELFTEST name=q4k_hand ok=%d n=%d\n", g_failures == 0 ? 1 : 0, 3 * QK_K + 1);
    }
    ++n;
    /* --- Q8_0, hand-worked: d = 2.0, qs = lane -> (-128..127 pattern); value = 2*qs ------------ */
    {
        uint8_t blk[Q4KD_Q8_0_BYTES];
        float y[Q4KD_MAX_BLOCK_ELTS];
        memset(y, 0, sizeof(y));
        blk[0] = 0x00; blk[1] = 0x40;         /* d = 2.0 */
        for (int i = 0; i < QK8_0; ++i) blk[2 + i] = (uint8_t)(int8_t)(i - 16);
        q4kd_decode_block(blk, Q4KD_KIND_Q8_0, Q4KD_MODE_CORRECT, y);
        for (int i = 0; i < QK8_0; ++i) {
            const int q = i - 16;
            expect_bits("q8_0 2*q", y[i], (uint32_t)bits(2.0f * (float)q));
        }
        /* d = 0 zeroes the block but does not erase the sign of the product (IEEE 754: +0 * y < 0
         * is -0), so the expected bit pattern is signed. gate.py compares 32-bit patterns, so a
         * comparator that folded -0.0 onto 0.0 would hide a real difference here. */
        blk[0] = 0x00; blk[1] = 0x00;
        q4kd_decode_block(blk, Q4KD_KIND_Q8_0, Q4KD_MODE_CORRECT, y);
        for (int i = 0; i < QK8_0; ++i) {
            const int q = i - 16;
            expect_bits("q8_0 d=0", y[i], q < 0 ? 0x80000000u : 0x00000000u);
        }
        printf("SELFTEST name=q8_0_hand ok=%d n=%d\n", g_failures == 0 ? 1 : 0, 2 * QK8_0);
    }
    ++n;
    /* --- the leak guard: an inapplicable (kind, mode) pair must write NaN, not a number -------- */
    {
        uint8_t blk[Q4KD_Q4K_BYTES];
        float y[Q4KD_MAX_BLOCK_ELTS];
        memset(blk, 0x5A, sizeof(blk));
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, Q4KD_KIND_Q4K, Q4KD_MODE_Q8_0_BYTE_SHIFT_1, y);
        int nan = 1;
        /* every slot, not a sample: an unwritten element reads 0.0 after the memset above, so a lane
         * that forgot to write its scattered slots shows up here instead of passing unseen. */
        for (int i = 0; i < QK_K; ++i) nan &= (bits(y[i]) == (int)Q4KD_NAN_BITS);
        if (!nan) fail("selftest q4k unit under a Q8_0 mode did not write NaN in every slot");
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, Q4KD_KIND_Q8_0, Q4KD_MODE_Q4K_DROP_MIN, y);
        nan = 1;
        for (int i = 0; i < QK8_0; ++i) nan &= (bits(y[i]) == (int)Q4KD_NAN_BITS);
        if (!nan) fail("selftest q8_0 unit under a Q4_K mode did not write NaN in every slot");
        printf("SELFTEST name=leak_guard ok=%d n=2\n", g_failures == 0 ? 1 : 0);
    }
    ++n;
    printf("DRIVER SELFTEST RESULT checks=%d failures=%d\n", n, g_failures);
    if (g_failures) printf("RESULT FAIL driver selftest failures=%d\n", g_failures);
    else printf("RESULT PASS driver selftest checks=%d\n", n);
    return g_failures ? 1 : 0;
}

/* --------------------------------------------------------------------------------------------- */
/* main                                                                                           */
/* --------------------------------------------------------------------------------------------- */
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

int main(int argc, char **argv) {
    if (have_flag(argc, argv, "--selftest")) return driver_selftest();

    const char *plan_path = arg_value(argc, argv, "--plan", NULL);
    const char *cases_path = arg_value(argc, argv, "--cases", NULL);
    const char *slab_path = arg_value(argc, argv, "--slab", NULL);
    const char *out_path = arg_value(argc, argv, "--out", NULL);
    const char *report_path = arg_value(argc, argv, "--report", NULL);
    const int expect_units = (int)strtol(arg_value(argc, argv, "--expect-units", "-1"), NULL, 10);
    const int expect_cases = (int)strtol(arg_value(argc, argv, "--expect-cases", "-1"), NULL, 10);

    if (!plan_path || !cases_path || !slab_path || !out_path || !report_path) {
        printf("usage: q4k_dequant --plan U --cases C --slab S --out O --report R "
               "[--expect-units N] [--expect-cases N]\n"
               "       q4k_dequant --selftest | --check\n");
        return 2;
    }
    Plan plan;
    int lu = 0, lc = 0;
    if (read_plan(plan_path, &plan, &lu, &lc)) return 1;
    if (read_plan(cases_path, &plan, &lu, &lc)) return 1;
    /* One line, both counts, both paths: units and cases arrive as two files, and a parser that read
     * only the first would report a plan with units and no work to do. gate.py requires this line to
     * carry all four fields and the counts to match the files it wrote. */
    printf("PLAN units_file=%s cases_file=%s units_parsed=%d cases_parsed=%d\n",
           plan_path, cases_path, lu, lc);
    if (lu != (int)plan.units.size() || lc != (int)plan.cases.size()) {
        printf("FAIL plan counted %d/%d rows but holds %zu/%zu records\n", lu, lc,
               plan.units.size(), plan.cases.size());
        return 1;
    }

    std::vector<uint8_t> slab;
    {
        FILE *fh = fopen(slab_path, "rb");
        if (!fh) { printf("FAIL cannot open slab %s: %s\n", slab_path, strerror(errno)); return 1; }
        fseek(fh, 0, SEEK_END);
        long long n = ftello(fh);
        fseek(fh, 0, SEEK_SET);
        slab.resize((size_t)n);
        if (n && fread(slab.data(), 1, (size_t)n, fh) != (size_t)n) {
            printf("FAIL short read on slab\n");
            fclose(fh);
            return 1;
        }
        fclose(fh);
    }
    long long out_floats = 0;
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        long long e = (long long)plan.cases[i].dst_off + plan.units[plan.cases[i].unit].n_elts;
        if (e > out_floats) out_floats = e;
    }
    if (verify_plan(plan, expect_units, expect_cases, (long long)slab.size(), out_floats)) return 1;
    if (g_failures) return 1;

    if (have_flag(argc, argv, "--check")) {
        printf("RESULT PASS check plan units=%zu cases=%zu (no device touched)\n",
               plan.units.size(), plan.cases.size());
        return 0;
    }

    /* ---- device work -------------------------------------------------------------------------- */
    std::vector<float> h_out((size_t)out_floats, 0.0f);
    std::vector<float> h_csum(plan.cases.size(), 0.0f);
    std::vector<int> h_cbad(plan.cases.size(), 0);
    std::vector<int> h_cnan(plan.cases.size(), 0);

    Q4kUnit *d_units = nullptr;
    Q4kCase *d_cases = nullptr;
    uint8_t *d_slab = nullptr;
    float *d_out = nullptr;
    float *d_csum = nullptr;
    int *d_cbad = nullptr;
    int *d_cnan = nullptr;
    CK(hipMalloc(&d_units, sizeof(Q4kUnit) * plan.units.size()));
    CK(hipMalloc(&d_cases, sizeof(Q4kCase) * plan.cases.size()));
    CK(hipMalloc(&d_slab, slab.size()));
    CK(hipMalloc(&d_out, sizeof(float) * (size_t)out_floats));
    CK(hipMalloc(&d_csum, sizeof(float) * plan.cases.size()));
    CK(hipMalloc(&d_cbad, sizeof(int) * plan.cases.size()));
    CK(hipMalloc(&d_cnan, sizeof(int) * plan.cases.size()));
    CK(hipMemcpy(d_units, plan.units.data(), sizeof(Q4kUnit) * plan.units.size(),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_cases, plan.cases.data(), sizeof(Q4kCase) * plan.cases.size(),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_slab, slab.data(), slab.size(), hipMemcpyHostToDevice));
    CK(hipMemset(d_out, 0, sizeof(float) * (size_t)out_floats));
    CK(hipMemset(d_csum, 0, sizeof(float) * plan.cases.size()));
    CK(hipMemset(d_cbad, 0, sizeof(int) * plan.cases.size()));
    CK(hipMemset(d_cnan, 0, sizeof(int) * plan.cases.size()));

    /* one launch per mode, modes ascending, mode 0 first so an injection case can be compared
     * against the same unit's correct output while it is still in device memory */
    std::vector<Q4kCase> subset;
    std::vector<int> subset_gidx;
    for (int mode = 0; mode < Q4KD_MODE_COUNT; ++mode) {
        subset.clear();
        subset_gidx.clear();
        for (size_t i = 0; i < plan.cases.size(); ++i)
            if (plan.cases[i].mode == mode) {
                subset.push_back(plan.cases[i]);
                subset_gidx.push_back((int)i);
            }
        if (subset.empty()) continue;
        Q4kCase *d_sub = nullptr;
        int *d_gidx = nullptr;
        CK(hipMalloc(&d_sub, sizeof(Q4kCase) * subset.size()));
        CK(hipMalloc(&d_gidx, sizeof(int) * subset.size()));
        CK(hipMemcpy(d_sub, subset.data(), sizeof(Q4kCase) * subset.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(d_gidx, subset_gidx.data(), sizeof(int) * subset.size(), hipMemcpyHostToDevice));
        const int grid = (int)(subset.size() < 128 ? subset.size() : 128);
        q4k_dequant_kernel<<<grid, Q4KD_BLOCK>>>(d_units, d_sub, d_gidx, (int)subset.size(), d_slab,
                                                d_out, d_csum, d_cbad, d_cnan, mode);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        CK(hipFree(d_sub));
        CK(hipFree(d_gidx));
    }
    CK(hipMemcpy(h_out.data(), d_out, sizeof(float) * (size_t)out_floats, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_csum.data(), d_csum, sizeof(float) * plan.cases.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_cbad.data(), d_cbad, sizeof(int) * plan.cases.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_cnan.data(), d_cnan, sizeof(int) * plan.cases.size(), hipMemcpyDeviceToHost));
    CK(hipFree(d_units));
    CK(hipFree(d_cases));
    CK(hipFree(d_slab));
    CK(hipFree(d_out));
    CK(hipFree(d_csum));
    CK(hipFree(d_cbad));
    CK(hipFree(d_cnan));

    /* The device's own words, dumped raw. This file — not any number the driver prints — is what
     * gate.py compares against the CPU reader, so it is written before the report and its size is
     * stated in a line the parser can hold to the plan. */
    FILE *of = fopen(out_path, "wb");
    if (!of) {
        printf("FAIL cannot write out %s: %s\n", out_path, strerror(errno));
        return 1;
    }
    if (out_floats > 0 &&
        fwrite(h_out.data(), sizeof(float), (size_t)out_floats, of) != (size_t)out_floats) {
        printf("FAIL short write on out %s\n", out_path);
        fclose(of);
        return 1;
    }
    if (fflush(of) || ferror(of)) {
        printf("FAIL flush of out %s failed\n", out_path);
        fclose(of);
        return 1;
    }
    fclose(of);
    printf("OUT file=%s floats=%lld bytes=%lld\n", out_path, out_floats, 4LL * out_floats);

    FILE *rep = fopen(report_path, "wb");
    if (!rep) { printf("FAIL cannot write report %s: %s\n", report_path, strerror(errno)); return 1; }
    fprintf(rep, "# case_id\tunit\tmode\tkind\tcls\tn_elts\tout_sum\tbad_vs_mode0\tnan\n");
    double agg[Q4KD_MODE_COUNT][Q4KD_KIND_COUNT][3];
    int agg_n[Q4KD_MODE_COUNT][Q4KD_KIND_COUNT][3];
    memset(agg, 0, sizeof(agg));
    memset(agg_n, 0, sizeof(agg_n));
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const Q4kCase &c = plan.cases[i];
        const Q4kUnit &u = plan.units[c.unit];
        const int cls = u.cls < 0 || u.cls > 2 ? 0 : u.cls;
        fprintf(rep, "%zu\t%d\t%d\t%d\t%d\t%d\t%.9g\t%d\t%d\n", i, c.unit, c.mode, u.kind, cls,
                u.n_elts, (double)h_csum[i], h_cbad[i], h_cnan[i]);
        agg[c.mode][u.kind][cls] += (double)h_csum[i];
        agg_n[c.mode][u.kind][cls] += 1;
    }
    fclose(rep);
    printf("REPORT file=%s rows=%zu\n", report_path, plan.cases.size());

    /* aggregates, printed after all device work so nothing can swallow them (pitfalls/50) */
    int cells = 0;
    for (int mode = 0; mode < Q4KD_MODE_COUNT; ++mode)
        for (int kind = 0; kind < Q4KD_KIND_COUNT; ++kind)
            for (int cls = 0; cls < 3; ++cls)
                if (agg_n[mode][kind][cls]) {
                    printf("DEQCASE mode=%d kind=%s cls=%d units=%d out_sum=%.9g\n", mode,
                           kind == Q4KD_KIND_Q4K ? "Q4_K" : "Q8_0", cls, agg_n[mode][kind][cls],
                           agg[mode][kind][cls]);
                    ++cells;
                }
    printf("DEQCELLS cells=%d cases=%zu\n", cells, plan.cases.size());
    printf("RESULT PASS device plan units=%zu cases=%zu cells=%d out_floats=%lld\n",
           plan.units.size(), plan.cases.size(), cells, out_floats);
    return 0;
}
