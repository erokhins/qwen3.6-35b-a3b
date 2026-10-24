// pin_hold.cpp — pin N GiB of host memory for a device mapping and HOLD it, so two instances
// can be run concurrently to ask ONE question: is the ~32 GiB device-visible budget per
// process/client, or shared (per-UID / global, e.g. amdgpu.enforce_isolation dividing GTT among
// active clients)?
//
// Why this exists: the boundary measured on 2026-10-26 (~32 GiB granted, 34 GiB refused) was
// measured while flash_serve — the engine serving this very session — held ~23.66 GiB of GTT.
// That confounds "my process is capped at ~32 GiB" with "there is a shared pool and halogen
// already holds part of it". Two concurrent instances of this program, each asking for more than
// half the boundary, discriminates them:
//   both succeed  -> the budget is per process/client (total granted > the single-process cap)
//   second fails  -> the budget is shared, and a co-resident process' footprint may matter
//
// Allocation only: no timing is taken or printed, so this cannot contaminate a measurement window.
//
// Build: hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/pin_hold tools/pin_hold.cpp
// Run:   /tmp/pin_hold 24 20 & /tmp/pin_hold 24 20 & wait      (one blocking parent call)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char** argv) {
    double gib = (argc > 1) ? atof(argv[1]) : 24.0;
    int hold = (argc > 2) ? atoi(argv[2]) : 10;
    size_t bytes = (size_t)(gib * 1073741824.0) & ~(size_t)4095;

    void* p = nullptr;
    hipError_t e = hipHostMalloc(&p, bytes, hipHostMallocMapped);
    printf("pid %ld: hipHostMalloc %.1f GiB -> %s\n", (long)getpid(), gib,
           e == hipSuccess ? "OK" : hipGetErrorString(e));
    fflush(stdout);
    if (e != hipSuccess) return 1;
    sleep(hold);
    (void)hipHostFree(p);
    return 0;
}
