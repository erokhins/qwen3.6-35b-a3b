# Device window `g` — 2026-10-27 — first device execution of the gather and GDN arms

Main-agent run. Preconditions asserted by `engine/work/window_f.sh` (`set -uo pipefail` before any
pipeline, no gate command inside a pipeline, `gpu_lock status` matched against the expected tag
because `status` prints `HELD: someone else` and still exits 0, engine-down read from `rocm-smi`).
Stale halogen stream claim from the finished `d63651e5` lane released before acquiring.
Correctness-only: **no timing of any kind was taken.**

## Outcome: the run aborted, and it did not reach a verdict

```
DEVICERUN rc=1
FIRED=15 SILENT=1
(no RESULT line, no CHECK verdict=FAIL lines)
```

`block_device.py` **crashed** at line 2055 in `control_gather_read_back`:

```
want = u["_bytes"][:16].hex()
KeyError: '_bytes'
```

The control assumes the unit dict came from the CPU host twin, where the driver keeps the raw
bytes it wrote. On the device path the unit is produced by the arm's read-back and has no
`_bytes`. This is the third instance of the same failure class in this repo — after
`KeyError: 'run'` in `cpu-witness-reports-no-device-acts` and the parser that printed
`parsed=0` while reporting PASS: **a control that only understands one path either crashes or
silently passes, and both read as "the gate ran".**

## What *was* established before the crash

Compiled pairs, `ndiff_bytes`, device included:

| pair | bytes | ndiff |
|---|---|---|
| `host_twin_vs_hip_hostpath` (shared expert) | 22,532 | **0** |
| `rt_host_twin_vs_rt_hip_hostpath` | 3,232 | **0** |
| `rt_hip_device_vs_rt_hip_hostpath` | 3,232 | 237 (documented `exp()` scope, `wpre` only) |
| `g_ref_bf16_host_twin_vs_hip_hostpath` | 122,880 | **0** |
| **`g_ref_bf16_hip_device_vs_hip_hostpath`** | 122,880 | **0** |
| `g_policy_a_host_twin_vs_hip_hostpath` | 122,880 | **0** |
| **`g_policy_a_hip_device_vs_hip_hostpath`** | 122,880 | **3,744 — unattributed** |
| `d_host_twin_vs_d_hip_hostpath` | 2,187,264 | **0** |
| **`d_hip_device_vs_d_hip_hostpath`** | 2,187,264 | **52,976 — unattributed** |

So the **reference-path gather arm is byte-exact across compilers on the device** — the first
device arm whose answer matches its host twin exactly. The two divergences are policy-A gather and
GDN, and neither is attributed yet.

Census survived the device path: `GATHERCENSUS path=ref_bf16 slot=device distinct_experts=8
arm_units=24 arm_total=50331648 whole_reads=0 disagree=none`. Tie control ran its device legs:
`legs=device:2,host_cpubody:2`. `WPRESCOPE … measured_max_ulp=2 scope_ulp=9 binding=documented`,
`RTATTRIB … r1_fields_bit_identical=True wcast_bit_identical=True`.
`band_digest=27a68c7e82c9aa05` unchanged.

## One control SILENT on the device path

```
CONTROL name=control-witness-guard-can-redden fired=SILENT
```

A guard whose job is to prove the witness can redden went silent precisely on the path where the
witness matters most. Silent is not neutral: it is the state in which a guard is not guarding.

## Deliberately not done here

The two device-vs-host divergences are **not** absorbed, explained away, or moved out of a claimed
field. Precedent is `wpre` (window `e2` → `f`): attribute first with measured ULP spread, then
either fix the kernel or re-derive a bound from a documented prior with a witnessed rejection.
Deleting `policy_a`/GDN fields from claimed-exactness would be the relaxation D-18 reserves to the
user. Until attribution, M2's device arm is **FAIL**, and the earlier statement that "M2's device
arm is green" applies only to the shared-expert and router pairs that were re-verified here.

Log: `specification/evidence/blockdevice-device-2026-10-27-g.log`.
