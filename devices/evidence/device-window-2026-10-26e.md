# Device window — 2026-10-26, fifth & sixth windows (`e`, `e2`) — first device execution of our kernels

Purpose: execute `engine/block_device.py --device`, the first time any kernel written in this project has run on the
iGPU. Exit-gate expectations were written by the implementing lane into `engine/HANDOFF.md`.

## Window `e` — **VOID, and void because of me**

The first attempt ran with **no `gpu_lock` held and the local engine not confirmed down**. Cause chain, recorded so
it is repeatable-in-the-wrong-direction and therefore fixable:

1. A `halogen` stream claim from the finished M2 lane (`/tmp/pi_halogen.lock`, keepalive pid 910720) was never
   released. I released it after the *first* of two resumptions but not after the resumed run completed — a claim
   outliving the worker it was minted for.
2. `tools/gpu_lock.sh acquire` refused (it sees the live stream claim), but my command line chained the steps with
   `&&` **through a pipeline** — `gpu_lock acquire && export … && halogen_gate.sh down | tail -1 && build && run` —
   and a pipeline's exit status is the last command's, so `tail`'s success papered over the refusal and the run
   proceeded.
3. Consequence: `RESULT … mode=device device=on … hip_calls=19+24 launches=7` was produced while residency of the
   engine was unknown and the machine-wide lock was not held.

The log is kept — `specification/evidence/blockdevice-device-2026-10-26-VOID-no-lock.log` — labelled void, and none
of its numbers is quotable. Two lessons, both general: **a claim lock must be released when the worker dies, not when
I remember**; and **never chain a device run behind a status command through a pipe**, because the pipe reports the
formatter's exit code, not the gate's.

## Window `e2` — the real run

`gpu_lock` HELD under `q36-main window-2026-10-26e2 m2-first-device-exec-clean` (status printed, not assumed),
`halogen_gate.sh down` → `stopped in 3s`, `VRAM Total Used Memory 831,705,088 B` before the run, `up` afterwards
(sixth/seventh entries in `evidence/halogen-restart.log`). Log:
`specification/evidence/blockdevice-device-2026-10-26-e2.log`.

```
RESULT mode=device checks=25 findings=2 controls=9 fired=8 rows=6 arms=policy_a device=on staged=admitted hip_calls=19+24 launches=7 bytes_to_device=10903552 bytes_from_device=10921168 band_digest=27a68c7e82c9aa05 no timing taken
```

**What the device actually did** (each item was on the must-hold list and is counted, not asserted): 2 `DEVALLOC`
slabs, 4 `STAGERUN … VERIFIED`, 9 `BIND … dev_sha256=<hex>`, `ARM arm=hip_device … launches=4`,
`ARM arm=rt_hip_device … launches=3`, `launches=7`, `hip_calls=19+24`, 10,903,552 B to the device and 10,921,168 B
back. The three licensed rows — `router_logits`, `topk_ids`, `topk_weights` — are **ACCEPTED at `ndiff=0` from the
device arm** (`licensed=3 accepted=3 rejected=0`), the three shared-expert rows remain honestly `UNLICENSED`, and
`compiled-pair-byte-identical` still PASSes for the shared-expert TU at 22,532 B / `ndiff_bytes=0`.

## The two findings are about the host/device pair, not about the pins

```
PAIR pair=rt_hip_device_vs_rt_hip_hostpath file=router_record.f32 bytes=3232 ndiff_bytes=237 claimed_fields=ids,ids_alt,wpre,wcast claimed_ndiff=ids=0/8,ids_alt=0/8,wcast=0/8,wpre=5/8 verdict=CLAIMED-DIFFER
CHECK name=rt-device-host-claimed-fields-identical verdict=FAIL … ids=0/8, ids_alt=0/8, wcast=0/8, wpr[e=5/8]
```

Everything a licence measures at exactness agrees bit-for-bit across the two compilers: `ids`, `ids_alt`, and `wcast`
— the cast-first bf16 weights that *are* `topk_weights` — all `0/8`. What differs is `wpre`, the **fp32 pre-cast**
renormalised weights, on 5 of 8 slots. That is the field `prototype/moe_router/moe_router_kernels.h` §R5 documents as
reproducing the pinned pre-cast weights "to single-digit ULP and **not** bit-exactly", with the host/device `exp()`
scope difference named as the mechanism. So the check was asserting bit-identity on a quantity no licence measures at
exactness, while its own detail text said a difference there is "the prototype header's documented `exp()` scope, not
a tolerance to absorb".

**That does not make it droppable.** Removing `wpre` from the claim would be a relaxation, which D-18 reserves to the
user; the admissible move is a re-derivation with witnessed rejection: split the claim so the licensed fields stay at
zero tolerance, and give `wpre` a **measured** bound derived from the documented `exp()` scope — with a control that
injects a larger-than-scope perturbation and must redden it. Attribution comes first: measure the actual ULP spread of
`wpre` between the two compilations before any wording changes. This is dispatched, not decided here.

## One control went SILENT on the device path

`CONTROL name=topk-selection-perturbations-redden-ids fired=SILENT` in `mode=device`, having fired `RED` in the CPU
witness. A control that fires on one path and not the other is a control that only guards half the code, and the
device path is the one we are about to build on. It must either execute its legs on the device arm or declare
explicitly which path it covers — silence is not an option.

## Not established, named

- **M2 is not green.** Terminal verdict is `FAIL` pending the `wpre` re-derivation above; even with it resolved the
  ceiling is `QUALIFIED`, because `block_output` has no pinned referent and the fixture that would supply one is a
  reserved (user) decision.
- **No timing of any kind.** Every figure here is a byte count, a launch count or a digest. D-22's ≥ 48 t/s and
  ladder rungs M4/M5 are untouched, and no rate from any window may be quoted beside another's ruler.
- GDN and routed-expert gather still have no engine-side kernel: they were the hard stop of the router slice and
  remain separately delegated.

## Window `f` — the device-path legs, witnessed

Preconditions asserted rather than piped (`engine/work/window_f.sh`, `set -uo pipefail`, no gate inside a pipeline,
lock status matched against the expected tag because `gpu_lock.sh status` prints `HELD: someone else` and still exits
0, engine-down read from `rocm-smi` → `VRAM_USED=891,719,680`):

```
RESULT mode=device checks=32 findings=0 controls=10 fired=10 rows=6 arms=policy_a device=on staged=admitted hip_calls=19+24 launches=7 bytes_to_device=10903552 bytes_from_device=10921168 band_digest=27a68c7e82c9aa05
FIRED=10 SILENT=0
CONTROL name=topk-selection-perturbations-redden-ids fired=RED legs=device:2,host_cpubody:2
WPRESCOPE field=wpre pair=rt_hip_device_vs_rt_hip_hostpath n=8 ndiff=5/8 ulp_per_slot=1,1,1,0,0,0,2,1 measured_max_ulp=2 scope_ulp=9 binding=documented
```

The two findings of window `e2` are gone: the `wpre` claim is split, licensed fields (`ids`, `ids_alt`, `wcast`) at
zero tolerance, `wpre` inside a bound taken from documentation (9 ULP, §R5) rather than from the measurement (2), and
the tie control now executes two legs **on the device path** — the thing only a window can witness. `checks` rose
32 vs the CPU witness's 28 because the device mode runs per-path legs. Log:
`specification/evidence/blockdevice-device-2026-10-26-f.log`. **M2's device arm is green; its terminal verdict remains
`QUALIFIED` pending the layer-0 residual fixture, and no timing exists anywhere in these windows.**
