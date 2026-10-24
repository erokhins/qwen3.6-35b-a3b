# 10 — device memory and residency traps

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's
> `devices/strix-halo-unified-memory.md` (path relative to `../qwen3.5-0.8b/`; all figures are
> measurements of this box: Framework
> Desktop, Ryzen AI Max+ 395, `gfx1151`, 128 GB LPDDR5x-8000 unified). Re-measure only after a
> driver/BIOS/carve-out change, using `tools/mem_mode_probe_hip.cpp` and `tools/bw_probe_hip.cpp`
> (both copied into this repo).

## The device has no VRAM: allocation is system RAM reached through the GTT

`hipGetDeviceProperties().totalGlobalMem` reports ~110 GiB (a reservation and the GTT merged), which is
**not** what a process can actually use. There is no real VRAM to overflow, so the driver does not complain
the way you would expect — most failure modes here are silent, and the two loud ones are listed below.

## The pool is ~32–34 GiB per process, and no env var moves it

| test | result |
|---|---|
| one `hipMalloc` of 32 GiB | ok, 231 GB/s |
| one `hipMalloc` of 40 GiB | **fails** (`hipErrorOutOfMemory`) |
| 8 × 8 GiB | **4th ok (32 GiB), 5th fails**, mean 222 GB/s |
| 24 × 2 GiB | **17 ok (34 GiB), 18th fails**, mean 227 GB/s |

`GPU_MAX_HEAP_SIZE=100` and `GPU_MAX_ALLOC_SIZE` do not change it. It is a total-pool limit, not an
allocation-size limit, and bandwidth is flat right up to the edge. **This limit is the whole reason this
project has a capacity problem** (64.6 GiB of bf16 text weights against it), and `DECISIONS.md` D-02 is the
one untested escape: whether a >34 GiB *host-mapped* buffer can be read by a kernel at full speed. That has
never been measured on this box.

## `hipMallocManaged` degrades silently near the ceiling

4–24 GiB under `Managed` measured 215–239 GB/s; at 32 GiB the same access dropped to **51.6 GB/s with no
error at all**. `hipMalloc` fails loudly in the same situation. Therefore: **`hipMalloc` + explicit
`hipMemcpy` for the weight pool**; `Managed` only for small host-shared control buffers. A silently
half-slowest engine is indistinguishable from a model that is just slow.

## Host-pinned reads run at GTT speed, so residency is about address space, not bandwidth

Host-pinned and GTT pages measured the same read speed. So the choice between them is about control,
eviction and — crucially — **how much of it fits in the pool**, not about throughput. Zero-copy pointers are
still the wrong tool for weight streaming: they keep the host in the loop per access.

## Allocation recipe that is known to work on this box

```c
void* weights;                        // one slab for weights, one for state
hipMalloc(&weights, WEIGHTS_BYTES);   // never per-tensor allocations in the decode loop
void* state;
hipMalloc(&state, STATE_BYTES);

char* stage; hipHostMalloc(&stage, shard_bytes);          // stage the shard here
hipMemcpy(weights + offset, stage, shard_bytes, hipMemcpyHostToDevice);  // one bulk copy per shard
                                   // measured 54-58 GB/s incl. destination first touch,
                                   // 76-82 GB/s steady
launch_first_kernel(...); hipDeviceSynchronize();          // warm: this pass runs at ~35 GB/s
```

Per-tensor allocation and per-page first-touch are the ~35 GB/s regime. The warm pass is not optional before
any measurement or first request.

## The pool competes with everything else on the box

The pool *is* system RAM. `free -g` shows tens of GiB in `buff/cache`; the desktop's own ~2 GiB of the
reservation and page-cache pressure both come out of what is left. A kernel reading tens of GiB while the
page cache is full will be slower than an idle-machine probe said. Never benchmark while anything else is
touching memory, and re-measure after heavy I/O (a 67 GiB checkpoint load is exactly such heavy I/O).

## Check the memory split after any driver / BIOS / carve-out change

1. `cat /sys/class/drm/card*/device/mem_info_{vram,gtt}_{total,used}` — record the split.
2. `/tmp/mem_mode_probe 4.0 5` — read must stay 235–240 GB/s in every mode; note the first-pass column.
3. `/tmp/mem_mode_probe 2.0 1 1.565e9 many 24` — re-find the total-pool ceiling before sizing anything.
4. `free -g` — what the desktop is holding is unavailable to the pool.

Raising the carve-out (BIOS memory settings or the `rmem` module `ConfigInfo=`) is a root change and every
number above must be re-taken afterwards, not assumed unchanged.
