# Device window — 2026-10-26, second window (`window-2026-10-26b`)

Run by the main agent, foreground and blocking, with **zero subagents active** (verified: the board's only `ALIVE`
row was its own legend text, not a worker), `tools/gpu_lock.sh` held under tag
`q36-main window-2026-10-26b igpu-measurements`, and halogen stopped through `tools/halogen_gate.sh down` — which
reported `VRAM Total Used Memory` falling from 24.3 GiB to **0.885 GB** before any measurement was taken. Session
endpoint was `tensorfold` (remote), so `down` was not self-decapitation. The A/B driver is committed as
`tools/device_window_ab_llama.sh` and the captured logs as `devices/evidence/device-window-2026-10-26b-*.log`; the
compiled `mem_mode_probe` binary and its raw per-size logs live in `work/device-window/` and are deliberately not
committed. Both drivers own **no lock**, per the self-locking-driver deadlock rule.

Window: `2026-10-26T04:34:42Z` (lock acquired) → halogen `ready: /health status=ok engine.responds=true after 79s`.

## 1. The D-02 pool bound — the premise of the quantization decision was capacity-shaped, and it is wrong

`tools/mem_mode_probe_hip.cpp`, invoked as `mem_mode_probe <GiB> <reps> <bytes_per_token>`. Box identity printed by
the probe itself: `device AMD Radeon 8060S Graphics | 20 CUs | visible device memory 110.0 GiB`, `carve-out total
2.00 GiB | GTT total 112.00 GiB`.

| requested | mode | `vram+` GiB | `gtt+` GiB | steady read GB/s | tok/s cap |
|---|---|---|---|---|---|
| 48.00 | `gtt` (plain `hipMalloc`) | 0.05 | **48.00** | 236.3 | 94 |
| **66.1337** | `gtt` | 0.02–0.05 | **66.13** | **190.4 / 219.2** (two runs) | 75 / 87 |
| 66.1337 | `man` (managed) | 0.01–0.02 | 0.00 | 237.5, 237.7 | 94 |
| 66.1337 | `pin` (pinned host) | 0.08 | 0.00 | 237.1, 237.1 | 94 |

**A 66.1337 GiB `hipMalloc` succeeds** — the entire bf16 model, draft block included — and sysfs says the pages came
from GTT (`gtt+` moved by exactly the requested amount while `vram+` moved ≤0.08 GiB), exactly as
`tools/mem_mode_probe_hip.cpp §header` predicts. Host headroom was not tight either: `MemAvailable` 119.89 GiB before,
119.85 GiB after the 66 GiB runs (the allocation is released between modes).

**Read this as grants-versus-allocations, not as a refutation of the pool bound.** The ≈55.5–57.5 GiB figure in
`devices/measured-facts.md §Residency above the pool` was measured by *KFD grant arithmetic* — 23.5 + 32 granted,
23.5 + 34 refused, 23.5 + 20 + 12 granted, 23.5 + 20 + 24 refused — so it bounds what the KFD will hand out as
accounted device-visible memory. This probe never asked for a 66 GiB grant: as
`tools/mem_mode_probe_hip.cpp §header` already documents, `hipMalloc` silently over-commits the 2.00 GiB carve-out and
hands back GTT pages, and that is exactly what the sysfs columns show. **The ceiling is therefore an accounting regime
and not an allocation cap.** A 66.1337 GiB `hipMalloc` does not get refused; it gets *slow* — 190.4 and 219.2 GB/s
against `pin`'s 237.1, both runs. So the honest D-02 answer is two-part: no allocation failure point was found up to
the full bf16 size on an idle box, and the thing that binds instead is bandwidth. Whether a 66 GiB **grant** would be
refused is the question the earlier bracket answers (probably yes, above ≈57.5 GiB) and this probe did not ask.

**This does not overturn D-01, and the reason is worth stating precisely.** Quantizing the routed experts was never
mainly a capacity argument, and §1 makes that harder to argue away rather than easier: bf16 *can* be allocated, and it
reads at 190–219 GB/s, so a 66.1337 GiB-per-token bf16 decode is **~2.8–3.4 t/s** — about 28× too slow, with no
refusal to blame. Policy A is 2.5252 GiB per token → ~95 t/s at the 240 GB/s ceiling. What the correction changes is
which sentence we are allowed to say out loud: *bf16 would be allocated into GTT and be ~28× too slow, and it also
cannot be granted co-resident with the other tenants* — not *bf16 is above the pool*.

## 2. The read-rate ladder, and the tier at exactly our GDN state size

Same probe, smaller buffers, five reps, device reads through HIP (this is the on-ruler version; the CPU-side
`tools/bw_probe.py` is a different ruler and is deliberately not mixed in):

| buffer | `carve` GB/s | `gtt` GB/s | `man` GB/s | `pin` GB/s |
|---|---|---|---|---|
| 0.0586 GiB (**60.0 MiB — one layer's GDN state**) | **239.9** | 185.5 | 105.9 | 112.0 |
| 0.25 GiB | 242.0 | 228.1 | 219.8 | 203.9 |
| 1.00 GiB | 240.3 | 239.4 | 238.5 | 231.6 |
| 48.00 GiB | 240.2 | 236.3 | 237.7 | 237.0 |

Two things follow. First, **`carve` is the only mode that never dips** — 239.9, 242.0, 240.3, 240.2, 237.8 across
four orders of magnitude of buffer size. That stability across sizes where the other modes swing from 106 to 237 is
why ~240 GB/s is treated as the machine's ceiling rather than one sample. Second, **at 60.0 MiB there is a real
~2.1× tier**: carve-out-resident pages read at 239.9 GB/s while host-backed pages read at 105.9–185.5. Our GDN state
is 60.0 MiB per layer, so this is the one size where placement — not kernel shape — decides whether the recurrent
read is a 240 GB/s read or a 106 GB/s read.

**Not committed as a number:** the `gtt` deficit at 66 GiB measured **190.4** on one run and **219.2** on the next,
and the `1st-run` column is order-dependent enough to be useless (carve first-run read 2.3 GB/s at 60 MiB and 30.4 at
1 GiB). A spread is recorded; a single figure is not. Re-measure with randomized mode order before anyone quotes a
66 GiB `gtt` rate.

## 3. The roofline, and where llama.cpp sits against it

- ceiling: **240 GB/s**
- policy A per-token read: **2.5252 GiB** → **cap ≈ 95 t/s**
- measured llama.cpp decode: **53.4 / 53.4 / 54.1 t/s** (warm run 53.5), `offloaded 41/41 layers` on every run,
  prompt rate 330.8–360.3 t/s
- implied effective read: 53.4 × 2.5252 GiB ≈ **135 GB/s = 56 % of ceiling**, so roughly **1.8× of headroom exists
  in principle** for the engine, and it is a streaming/occupancy question rather than a bytes question.

**A/B honesty.** The morning window quoted 44.9 / 49.1 / 51.4 t/s from "one fixed ~50-word prompt" whose text was
never written down, so it cannot be re-derived; this window's prompt is recorded verbatim in
`tools/device_window_ab_llama.sh` and echoed into `devices/evidence/device-window-2026-10-26b-llama-ab.log` (43
words). The two sets are therefore
*the same box, same binary, different prompt* — comparable in magnitude, **not** a strict A/B, and this figure does
not supersede the old one as a baseline until a run uses a recorded prompt on both sides. These are also best-case
numbers: warm page cache, idle box, halogen down.

The first attempt at this parse printed `NO_EVAL_LINE` while the run was plainly fine (41/41 offloaded, 25.89 GiB of
GTT). Cause: this build prints decimals with a **comma** — `eval time = 585,70 ms / 32 tokens ( 52,94 tokens per
second)` — so a `[0-9.]+` pattern matched nothing. An empty grep is not an absent measurement; the guard now prints
`NO_EVAL_LINE` rather than a blank, and the pattern accepts commas.

## 4. What the engine actually occupies during a full-offload run

`python3 tools/device_window_probe.py 128`, 30 samples:

```
PEAK  VmPeak=33.79 GiB  VmHWM=1.47 GiB  last_VmRSS=1.47 GiB
KFD   per-process vram=UNAVAILABLE (no kfd proc files)
AMDGPU gtt_used_peak=25.89 GiB  vram_used_peak=0.88 GiB  vis_vram_used_peak=0.88 GiB
      host MemAvailable before=119.89 GiB  min during=92.82 GiB  samples_with_value=30
OFFLOAD  41/41 layers  41/41 layers
```

`VmPeak` 33.79 GiB is address space, not residency (~1.5× the file). The residency instrument that works is
`gtt_used_peak` **25.89 GiB** ≈ the 22.13 GB GGuf plus staging, corroborated by host `MemAvailable` falling
119.89 → **92.82 GiB** — i.e. the weights are host pages read over GTT, and they charge the *host* pool, which is the
mechanism behind the co-residency number in §1.

**One counter produced no output and is recorded as broken, not skipped:** `KFD per-process vram=UNAVAILABLE (no kfd
proc files)`, while `/sys/class/kfd/kfd/proc/` demonstrably contains process directories (`729179` was present during
this window). A check that prints nothing cannot fail, so the KFD column is treated as absent evidence and the AMDGPU
sysfs counters are the load-bearing ones until `device_window_probe.py`'s KFD reader is repaired.

## 5. Restart cost — the assumption D-13's batching rule was standing on

`halogen_gate.sh down` → `stopped in 4s; flash_serve absent from rocm-smi --showpids`;
`halogen_gate.sh up 1200` → `ready: /health status=ok engine.responds=true after 79s`; appended to
`devices/evidence/halogen-restart.log`. So a down/up round trip on this box is **~83 s**, not the reload of a 66 GB
checkpoint the "batch, do not oscillate" rule was guarding against. The rule stays — one window, every timing in it —
but its justification is now measured rather than assumed, and this window is the second entry in that log.

## 6. Deliberately absent from this window

- No gather/GDN kernel timing: `prototype/moe_gather` and `prototype/gdn` gates are CPU-only by design (`grep -c
  gpu_lock` on their drivers returns 0), so there is no device-timing driver for them yet — the 1.8× headroom in §3 is
  a target, not an attribution of where llama.cpp loses it.
- No `lm_head` device roofline: its gate owns no lock and is not a timing harness.
- No CPU-side/device-side ruler mixing: `tools/bw_probe.py` (torch, 4/8/16 threads) reached 398 GB/s at 60 MiB and
  567 GB/s at 15 MiB with 16 threads, falling to 81–93 GB/s from 256 MiB up — those are **CPU threads reading
  LPDDR5x**, a different agent from the iGPU, and quoting them beside the 240 GB/s device ceiling would be the
  two-rulers error. They are recorded, labelled, and not compared.
- No >66.1337 GiB probe (no failure point located), and no probe while another tenant was resident — the co-resident
  case is precisely the one still unmeasured, and it is the one D-11's deployment budget needs.
