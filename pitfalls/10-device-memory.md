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
3. `/tmp/hostmap_probe 40 1 3` — re-find the total-pool ceiling before sizing anything (it also sweeps
   pinning sizes; see `tools/hostmap_probe.cpp`).
4. `free -g` — what the desktop is holding is unavailable to the pool.

Raising the carve-out (BIOS memory settings or the `rmem` module `ConfigInfo=`) is a root change and every
number above must be re-taken afterwards, not assumed unchanged.

## Traps found in this project (2026-10-26, D-02 residency probe)

Each of these was measured with `tools/hostmap_probe.cpp` / `tools/pin_hold.cpp` + `tools/pin_pair.sh`;
raw output in `devices/evidence/`.

- **A boundary measured while a co-resident process holds part of the pool is a LEFTOVER, not a limit.**
  This one is mine, caught the same day: the first reading of the ~32 GiB ceiling was "per process",
  because 32 GiB was granted and 34 GiB was refused. That is exactly what a leftover looks like.
  `flash_serve` (halogen, the engine serving this very session) was holding 23.5 GiB of the same
  device-visible pool, and 23.5 + 32 = 55.5 was the real number. The test that refutes per-process
  costs 20 seconds and does not need the co-resident process to move: **ask two separate processes for
  an amount each comfortably under the suspected per-process cap** (`tools/pin_pair.sh 20 2` → one
  granted, one refused). If they cannot coexist, the cap is shared and the single-process number is
  pool-minus-something. Correct form: whenever a ceiling is measured next to another process holding
  the same resource, measure it a second time *with a concurrent sibling* before writing the word
  "per-process" anywhere.
- **The device-visible pool is box-wide (≈55.5–56.5 GiB measured here) and `hipHostMalloc` spends it
  rather than escaping it.** Consequence for planning: the engine's budget is `pool − co-resident`, so
  it is an operational question (who else is resident?) as much as a technical one — ~32 GiB while
  halogen runs, ~55 GiB if the engine owns the box, which is the difference between "4-bit weights are
  mandatory" and "4-bit weights are a speed choice" (`DECISIONS.md` D-11). What is *not* the mechanism,
  established by measurement: `RLIMIT_MEMLOCK` (this shell's is 15.45 GiB and a 32 GiB pin was granted
  anyway; halogen's is `unlimited`), system RAM (MemAvailable 86–93 GiB at refusal), and GTT occupancy
  (refused at 43.66/112 GiB used).
- **A refused 34 GiB pin fails in 0.0 s, which is a different failure from exhaustion.** Symptom:
  `hipHostMalloc(34 GiB)` returns `out of memory` instantly while MemAvailable reads 87 GiB and most of
  the GTT aperture is free. Mechanism: an accounting refusal, not memory pressure (and
  `hipHostRegister` answers `invalid argument` for the same size). Correct form: report *time to
  failure* beside the error — instant is a budget/limit story, slow is pressure — and print
  MemAvailable and the aperture around every allocation attempt, which is what the probes do.
- **Pinned host memory reads at device speed or better, so "it has to be in VRAM" is a false premise
  on this box.** Measured in one window: device 20 GiB = 198.6 GB/s, pinned host 12 GiB = 215.8 GB/s,
  both checksum-verified. Mechanism: the iGPU is coherent with the same LPDDR5x; the carve-out buys
  address space, not bandwidth. Correct form: choose residency for layout/management reasons and prove
  bandwidth separately — and do not "fix" a slow kernel by moving its weights to device memory without
  re-measuring.
- **An unregistered host pointer handed to a kernel faults the queue, and the fault looks like a fast
  kernel if the harness is not watching.** Symptom: `malloc`'d, never-registered memory passed as a
  device pointer → `HSA_STATUS_ERROR_MEMORY_FAULT`, and the timing loop still printed
  "1,082,301 GB/s". Mechanism: no mapping exists, the kernel dies immediately, so elapsed time is
  near-zero. Correct form, twice over: register or `hipHostMalloc` before use; and make every bandwidth
  harness verify the kernel *read the bytes* — `hostmap_probe.cpp` accumulates a per-thread atomic
  double sum and compares it against a closed-form checksum derived from the fill, and prints
  "NOT A BANDWIDTH RESULT" instead of the number. This is the `mixed_bw` checksum lesson (this file,
  `pitfalls/40`) reproduced on a fresh path within a day.
- **A device fault poisons the rest of the process.** One faulted read made every later case in the
  same process print errors or zeros, so the unregistered-pointer test is opt-in and last in the
  probe's order. Correct form: one process per case for anything that can fault, exactly as
  `pitfalls/40` says for device selftests.
