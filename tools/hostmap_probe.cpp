// hostmap_probe.cpp — D-02: can a kernel read a host buffer BIGGER than the device-visible
// pool, and at what bandwidth?
//
// Why this exists. devices/measured-facts.md records a ~32-34 GiB device-visible pool per
// process (a 40 GiB hipMalloc fails) and that host-pinned reads run at GTT speed, but nobody
// has measured a *single host mapping larger than the pool*. The answer decides whether
// bf16/Q8 residency is possible at all for a 64.6 GiB model, or whether quantization is
// mandatory. See DECISIONS.md D-01/D-02.
//
// What it does, in order (each step prints its own line and does not depend on the last):
//   A. hipMalloc <small>            - in-run baseline; expect ~237 GB/s (see pitfalls/00)
//   B. hipMalloc <big>              - expect FAILURE: re-confirms the pool ceiling in this window
//   C. hipHostMalloc <small>/<big>  - pinned + mapped host memory, read from the kernel
//   D. malloc + hipHostRegister     - the "register my own buffer" variant of C
//   E. plain malloc, unregistered   - exercises the system allocator, if prop.systemMemorySupport
//
// Each buffer is (1) filled on the host with a deterministic pattern while accumulating the
// expected checksum, (2) warmed with an untimed pass, (3) read N times, best-of-N by
// steady_clock (never hipEventElapsedTime - pitfalls/00 says it can return exactly 0), and
// (4) verified: the kernel returns a per-thread atomic double sum that must match the
// host-computed one, so a "fast" number that skipped bytes cannot be printed. The checksum
// expectation comes from the fill loop, never from the kernel, so it cannot pass by
// construction (pitfalls/40).
//
// Safety: it reads MemAvailable and refuses a size that would OOM the box, and it frees every
// allocation even on failure paths.
//
// Build: hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/hostmap_probe tools/hostmap_probe.cpp
// Run  : BLOCKING, gpu_lock held, zero subagents (see AGENTS.md GPU policy).
//   /tmp/hostmap_probe [big_gib=40] [small_gib=1] [passes=3]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#define CK(x)                                                                \
    do {                                                                     \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            fprintf(stderr, "  hip error %s @%d: %s\n", hipGetErrorString(e_),\
                    __LINE__, #x);                                           \
        }                                                                    \
    } while (0)

static double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Grid-stride float4 read, same shape as tools/bw_probe_hip.cpp so numbers are comparable.
// `sum` receives one atomicAdd per thread: a per-block store would be a fake reduction
// (pitfalls/40, the mixed_bw lesson) and this is only ~1e5 atomics for the whole buffer.
__global__ void read_sum_kernel(const float4* __restrict__ p, size_t n4, double* sum) {
    size_t stride = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step   = (size_t)gridDim.x * blockDim.x;
    double acc = 0.0;
    for (size_t i = stride; i < n4; i += step) {
        float4 v = p[i];
        acc += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }
    if (sum) atomicAdd(sum, acc);
}

// Deterministic pattern: exact binary fractions so a double sum is exactly reproducible.
__device__ __host__ static inline float4 pattern_at(size_t i) {
    uint32_t j = (uint32_t)(i & 255u);
    float v = (float)j * 0.5f;
    return make_float4(v, v, v, v);
}

static double expected_sum(size_t n4) {
    // sum over i of 4 * 0.5 * (i % 256) = 4*0.5*sum(i%256); closed form, no loop needed,
    // which is what makes it independent of the kernel.
    double per_cycle = 0.0;                 // sum of (i % 256) over one 256-cycle
    for (uint32_t j = 0; j < 256; ++j) per_cycle += (double)j;
    double cycles = (double)(n4 / 256);
    double rem = 0.0;
    for (size_t i = cycles * 256; i < n4; ++i) rem += (double)(i & 255u);
    return 2.0 * (cycles * per_cycle + rem);
}

static long mem_available_gib() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char k[64]; long kb; long avail = -1;
    while (fscanf(f, "%63s %ld kB\n", k, &kb) == 2) {
        if (!strcmp(k, "MemAvailable:")) { avail = kb / (1024 * 1024); break; }
    }
    fclose(f);
    return avail;
}

static long read_sysfs_num(const char* what) {
    // card index is not stable across boots; probe the usual candidates
    const char* cards[] = {"card0", "card1", "card2", "card3"};
    for (const char* c : cards) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/class/drm/%s/device/mem_info_%s", c, what);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        long v = -1;
        if (fscanf(f, "%ld", &v) != 1) v = -1;
        fclose(f);
        if (v >= 0) return v;
    }
    return -1;
}

static void print_aperture(const char* tag) {
    long vr = read_sysfs_num("vram_total"), vu = read_sysfs_num("vram_used");
    long gt = read_sysfs_num("gtt_total"), gu = read_sysfs_num("gtt_used");
    printf("  [%s] vram %.2f/%.2f GiB  gtt %.2f/%.2f GiB  MemAvailable %ld GiB\n", tag,
           vu / 1073741824.0, vr / 1073741824.0, gu / 1073741824.0, gt / 1073741824.0,
           mem_available_gib());
    fflush(stdout);
}

struct Result { const char* name; double gib; double gbs; double spread; bool ok; };

// Fill a host buffer with the pattern and return the checksum the kernel must reproduce.
static double fill_host(void* p, size_t bytes) {
    size_t n4 = bytes / sizeof(float4);
    float4* q = (float4*)p;
    double expect = expected_sum(n4);
    // write in chunks so progress is visible on a multi-minute fill
    const size_t CH = (size_t)512 * 1024 * 1024 / sizeof(float4);
    for (size_t base = 0; base < n4; base += CH) {
        size_t n = (n4 - base < CH) ? (n4 - base) : CH;
        for (size_t i = 0; i < n; ++i) q[base + i] = pattern_at(base + i);
        printf("  fill %5.1f %%\r", 100.0 * (base + n) / n4); fflush(stdout);
    }
    printf("  fill 100.0 %% (checksum of the written pattern computed independently)"
           "   \n");
    return expect;
}

// Fill a device buffer with the pattern using a kernel-side writer (cheap on device memory).
__global__ void fill_kernel(float4* p, size_t n4) {
    size_t stride = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step   = (size_t)gridDim.x * blockDim.x;
    for (size_t i = stride; i < n4; i += step) p[i] = pattern_at(i);
}

// One measurement: warm pass + best-of-N timed passes, with a checksum witness.
static Result measure(const char* name, const void* dev_ptr, size_t bytes, int blocks,
                      int threads, int passes, double expect) {
    Result r{name, bytes / 1073741824.0, 0.0, 0.0, false};
    if (!dev_ptr || !bytes) { printf("  %-22s SKIP (no pointer)\n", name); return r; }
    size_t n4 = bytes / sizeof(float4);
    double* dsum;
    if (hipMalloc(&dsum, sizeof(double)) != hipSuccess) { printf("  %-22s SKIP (sum alloc)\n", name); return r; }

    double best = 1e30, worst = 0.0;
    double got = 0.0;
    for (int p = 0; p < passes; ++p) {
        CK(hipMemset(dsum, 0, sizeof(double)));
        double t0 = now_s();
        hipLaunchKernelGGL(read_sum_kernel, dim3(blocks), dim3(threads), 0, 0,
                           (const float4*)dev_ptr, n4, dsum);
        CK(hipDeviceSynchronize());
        double dt = now_s() - t0;
        double gbs = (double)bytes / dt / 1e9;
        if (p > 0) {                       // pass 0 is the warm-up
            best = std::min(best, gbs);
            worst = std::max(worst, gbs);
        }
        if (p == passes - 1) {
            double h = 0.0;
            CK(hipMemcpy(&h, dsum, sizeof(double), hipMemcpyDeviceToHost));
            got = h;
        }
    }
    hipFree(dsum);
    double rel = expect != 0.0 ? (got - expect) / expect : 1.0;
    if (rel < 0) rel = -rel;
    r.gbs = best;
    r.spread = worst > 0 && best > 0 ? 100.0 * (worst / best - 1.0) : 0.0;
    r.ok = rel < 1e-6;
    printf("  %-22s %6.2f GiB  %8.1f GB/s  spread %5.1f %%  checksum %s (got %.6e want %.6e)\n",
           name, r.gib, r.gbs, r.spread, r.ok ? "OK" : "MISMATCH", got, expect);
    fflush(stdout);
    if (!r.ok) printf("  ^^ NOT A BANDWIDTH RESULT: the kernel did not read every byte\n");
    return r;
}

int main(int argc, char** argv) {
    double big_gib = (argc > 1) ? atof(argv[1]) : 40.0;
    double small_gib = (argc > 2) ? atof(argv[2]) : 1.0;
    int passes = (argc > 3) ? atoi(argv[3]) : 3;
    int try_unregistered = (argc > 4) ? atoi(argv[4]) : 0;   // opt-in: faults the context
    // Optional size sweep, e.g. "2,4,8,16,24,32": find the largest host mapping that can be
    // pinned and read, which is the number D-02 actually needs.
    std::string sweep = (argc > 5) ? argv[5] : "";

    rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        printf("  RLIMIT_MEMLOCK soft %s hard %s\n",
               rl.rlim_cur == RLIM_INFINITY ? "infinity" : std::to_string((long)rl.rlim_cur).c_str(),
               rl.rlim_max == RLIM_INFINITY ? "infinity" : std::to_string((long)rl.rlim_max).c_str());
    }

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s | hipGetDeviceProperties says %d CUs (halved on this SKU - use 40)\n",
           prop.name, prop.multiProcessorCount);
    printf("  canMapHostMemory %d | concurrentManagedAccess %d\n",
           prop.canMapHostMemory, prop.concurrentManagedAccess);
#if HIP_HOST_COHERENT_MEMORY
    printf("  HIP_HOST_COHERENT_MEMORY defined\n");
#endif
    print_aperture("start");
    fflush(stdout);

    int blocks = 40 * 16, threads = 256;   // 16 waves on the real 40 CU count
    size_t small = (size_t)(small_gib * 1073741824.0) & ~(size_t)4095;
    size_t big = (size_t)(big_gib * 1073741824.0) & ~(size_t)4095;
    double exp_small = expected_sum(small / sizeof(float4));
    double exp_big = expected_sum(big / sizeof(float4));

    // ---- A. device baseline ------------------------------------------------
    printf("\nA. hipMalloc %.2f GiB (in-run baseline, expect ~237 GB/s)\n", small_gib);
    void* da = nullptr;
    if (hipMalloc(&da, small) == hipSuccess) {
        size_t n4 = small / sizeof(float4);
        hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads), 0, 0, (float4*)da, n4);
        CK(hipDeviceSynchronize());
        measure("hipMalloc small", da, small, blocks, threads, passes, exp_small);
        hipFree(da);
    } else {
        printf("  hipMalloc %.2f GiB FAILED (unexpected)\n", small_gib);
    }

    // ---- B. does the pool ceiling still hold in this window? ---------------
    printf("\nB. hipMalloc %.2f GiB (expect failure: the pool ceiling)\n", big_gib);
    print_aperture("before B");
    void* db = nullptr;
    hipError_t eb = hipMalloc(&db, big);
    if (eb == hipSuccess) {
        printf("  UNEXPECTED SUCCESS: a %.2f GiB device allocation was granted\n", big_gib);
        size_t n4 = big / sizeof(float4);
        hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads), 0, 0, (float4*)db, n4);
        CK(hipDeviceSynchronize());
        measure("hipMalloc big", db, big, blocks, threads, passes, exp_big);
        hipFree(db);
    } else {
        printf("  failed as expected: %s\n", hipGetErrorString(eb));
        (void)hipGetLastError();
    }
    print_aperture("after B");

    // ---- C. pinned + mapped host memory ------------------------------------
    printf("\nC. hipHostMalloc pinned+mapped host memory\n");
    unsigned int hflags = hipHostMallocMapped;   // device-accessible mapping of host pages
    void* hc = nullptr;
    if (hipHostMalloc(&hc, small, hflags) == hipSuccess) {
        void* dc = nullptr;
        CK(hipHostGetDevicePointer(&dc, hc, 0));
        double e = fill_host(hc, small);
        measure("hipHostMalloc small", dc, small, blocks, threads, passes, e);
        (void)hipHostFree(hc);
    } else {
        printf("  hipHostMalloc %.2f GiB failed\n", small_gib);
        (void)hipGetLastError();
    }
    long avail = mem_available_gib();
    if (avail < (long)big_gib + 12) {
        printf("  SKIPPING the %.0f GiB host test: MemAvailable %ld GiB is not big_gib + 12\n",
               big_gib, avail);
    } else {
        print_aperture("before C big");
        void* hd = nullptr;
        double t0 = now_s();
        hipError_t ec = hipHostMalloc(&hd, big, hflags);
        double treg = now_s() - t0;
        if (ec == hipSuccess) {
            void* dd = nullptr;
            CK(hipHostGetDevicePointer(&dd, hd, 0));
            printf("  hipHostMalloc %.0f GiB ok in %.1f s\n", big_gib, treg);
            double e = fill_host(hd, big);
            print_aperture("C big filled");
            measure("hipHostMalloc BIG", dd, big, blocks, threads, passes, e);
            (void)hipHostFree(hd);
        } else {
            printf("  hipHostMalloc %.0f GiB FAILED after %.1f s: %s\n", big_gib, treg,
                   hipGetErrorString(ec));
            (void)hipGetLastError();
        }
        print_aperture("after C");

        // ---- D. my own malloc + hipHostRegister ----------------------------
        printf("\nD. malloc + hipHostRegister (the register-my-own-buffer variant)\n");
        avail = mem_available_gib();
        if (avail < (long)big_gib + 12) {
            printf("  SKIPPING: MemAvailable %ld GiB\n", avail);
        } else {
            void* hm = malloc(big);
            if (!hm) {
                printf("  malloc(%.0f GiB) returned NULL\n", big_gib);
            } else {
                double e = fill_host(hm, big);
                double t1 = now_s();
                hipError_t er = hipHostRegister(hm, big, hipHostRegisterMapped);
                printf("  hipHostRegister(%.0f GiB) -> %s in %.1f s\n", big_gib,
                       er == hipSuccess ? "ok" : hipGetErrorString(er), now_s() - t1);
                if (er == hipSuccess) {
                    void* dh = nullptr;
                    CK(hipHostGetDevicePointer(&dh, hm, 0));
                    print_aperture("D registered");
                    measure("malloc+Register BIG", dh, big, blocks, threads, passes, e);
                    CK(hipHostUnregister(hm));
                } else {
                    (void)hipGetLastError();
                }
                free(hm);
            }
        }
        print_aperture("after D");

        // ---- E. unregistered plain malloc (system allocator) ---------------
        printf("\nE. plain malloc, unregistered (system allocator path)\n");
        if (!try_unregistered) {
            printf("  skipped by default: an unregistered host pointer faults the queue, which\n"
                   "  poisons every later case in the process. Pass 1 as argv[4] to test it.\n");
        }
        void* he = try_unregistered ? malloc(small) : nullptr;
        if (he) {
            double e = fill_host(he, small);
            CK(hipDeviceSynchronize());
            measure("malloc unregistered", he, small, blocks, threads, passes, e);
            free(he);
        }
    }

    // ---- F. size sweep: where does pinning stop working? -------------------
    if (!sweep.empty()) {
        printf("\nF. size sweep of pinned host mappings (sizes in GiB: %s)\n", sweep.c_str());
        size_t pos = 0;
        while (pos < sweep.size()) {
            size_t nxt = sweep.find(',', pos);
            if (nxt == std::string::npos) nxt = sweep.size();
            double gib = atof(sweep.substr(pos, nxt - pos).c_str());
            pos = nxt + 1;
            if (gib <= 0) continue;
            size_t bytes = (size_t)(gib * 1073741824.0) & ~(size_t)4095;
            if (mem_available_gib() < (long)gib + 8) {
                printf("  %6.1f GiB SKIP (MemAvailable %ld GiB)\n", gib, mem_available_gib());
                continue;
            }
            print_aperture("pre");
            void* hp = nullptr;
            double t0 = now_s();
            hipError_t e1 = hipHostMalloc(&hp, bytes, hflags);
            double tpin = now_s() - t0;
            if (e1 == hipSuccess) {
                void* dp = nullptr;
                CK(hipHostGetDevicePointer(&dp, hp, 0));
                double e = fill_host(hp, bytes);
                Result r = measure("hipHostMalloc", dp, bytes, blocks, threads, passes, e);
                printf("  %6.1f GiB pin ok %.1f s -> %.1f GB/s\n", gib, tpin, r.gbs);
                (void)hipHostFree(hp);
            } else {
                printf("  %6.1f GiB hipHostMalloc FAILED in %.1f s: %s\n", gib, tpin,
                       hipGetErrorString(e1));
                (void)hipGetLastError();
                // try registering our own buffer instead
                void* hm = malloc(bytes);
                if (!hm) {
                    printf("  %6.1f GiB malloc itself returned NULL\n", gib);
                } else {
                    hipError_t e2 = hipHostRegister(hm, bytes, hipHostRegisterMapped);
                    printf("  %6.1f GiB hipHostRegister -> %s\n", gib,
                           e2 == hipSuccess ? "ok" : hipGetErrorString(e2));
                    if (e2 == hipSuccess) {
                        double e = fill_host(hm, bytes);
                        void* dp = nullptr;
                        CK(hipHostGetDevicePointer(&dp, hm, 0));
                        measure("malloc+Register", dp, bytes, blocks, threads, passes, e);
                        CK(hipHostUnregister(hm));
                    } else {
                        (void)hipGetLastError();
                    }
                    free(hm);
                }
            }
            fflush(stdout);
        }
    }

    // ---- G. can the TOTAL device-visible footprint exceed the pool? --------
    // The question D-02 was really asked for: one 40 GiB mapping is refused above, but if N
    // pinned mappings totalling more than the ~32-34 GiB pool can all be read, then residency
    // is a layout problem rather than a quantization mandate.
    {
        std::string multi = (argc > 6) ? argv[6] : "";
        if (!multi.empty()) {
            int nc = atoi(multi.substr(0, multi.find(',')).c_str());
            double sgib = atof(multi.substr(multi.find(',') + 1).c_str());
            size_t bytes = (size_t)(sgib * 1073741824.0) & ~(size_t)4095;
            printf("\nG1. %d pinned mappings of %.1f GiB (total %.1f GiB vs a ~32-34 GiB pool)\n",
                   nc, sgib, nc * sgib);
            std::vector<void*> hs;
            for (int i = 0; i < nc; ++i) {
                void* hp = nullptr;
                hipError_t e = hipHostMalloc(&hp, bytes, hflags);
                if (e != hipSuccess) {
                    printf("  mapping %d FAILED: %s (got %d of %d)\n", i,
                           hipGetErrorString(e), (int)hs.size(), nc);
                    (void)hipGetLastError();
                    break;
                }
                hs.push_back(hp);
            }
            printf("  %d of %d mappings granted (%.1f GiB total device-visible)\n",
                   (int)hs.size(), nc, hs.size() * sgib);
            print_aperture("G1 pinned");
            for (size_t i = 0; i < hs.size(); ++i) {
                void* dp = nullptr;
                CK(hipHostGetDevicePointer(&dp, hs[i], 0));
                double e = fill_host(hs[i], bytes);
                char nm[48];
                snprintf(nm, sizeof(nm), "pinned chunk %zu", i);
                measure(nm, dp, bytes, blocks, threads, passes, e);
            }
            for (void* p : hs) (void)hipHostFree(p);
        }
        std::string mix = (argc > 7) ? argv[7] : "";
        if (!mix.empty()) {
            double dg = atof(mix.substr(0, mix.find(',')).c_str());
            double pg = atof(mix.substr(mix.find(',') + 1).c_str());
            size_t db = (size_t)(dg * 1073741824.0) & ~(size_t)4095;
            size_t pb = (size_t)(pg * 1073741824.0) & ~(size_t)4095;
            printf("\nG2. device hipMalloc %.1f GiB + pinned host %.1f GiB together (%.1f GiB)\n",
                   dg, pg, dg + pg);
            void* dv = nullptr;
            hipError_t ed = hipMalloc(&dv, db);
            void* hv = nullptr;
            hipError_t ep = hipErrorInvalidValue;
            if (ed == hipSuccess) ep = hipHostMalloc(&hv, pb, hflags);
            if (ed != hipSuccess) {
                printf("  device %.1f GiB failed: %s\n", dg, hipGetErrorString(ed));
                (void)hipGetLastError();
            } else if (ep != hipSuccess) {
                printf("  pinned %.1f GiB failed: %s\n", pg, hipGetErrorString(ep));
                (void)hipGetLastError();
                hipFree(dv);
            } else {
                size_t n4 = db / sizeof(float4);
                hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads), 0, 0, (float4*)dv, n4);
                CK(hipDeviceSynchronize());
                print_aperture("G2 both");
                measure("device part", dv, db, blocks, threads, passes, expected_sum(n4));
                void* dp = nullptr;
                CK(hipHostGetDevicePointer(&dp, hv, 0));
                double e = fill_host(hv, pb);
                measure("pinned part", dp, pb, blocks, threads, passes, e);
                hipFree(dv);
                (void)hipHostFree(hv);
            }
        }
    }

    print_aperture("end");
    printf("\nDone. Any line printed as MISMATCH is not a bandwidth number.\n");
    return 0;
}
