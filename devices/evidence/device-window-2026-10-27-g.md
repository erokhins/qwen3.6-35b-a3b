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

## Window `h` — 2026-10-27 — the attribution legs, and RUNG-1 says it is deterministic

Second main-agent window (zero workers, `gpu_lock` held with a declared tag, halogen down,
engine-down read from `rocm-smi`, two identical `--device` runs so a race cannot hide).
Correctness-only; **no timing taken**.

```
RUN1 rc=…  RUN2 rc=…   FIRED/SILENT per run recorded in the logs
legs still unexercised: 0
RUNG-1 constant across runs: ok
band_digest=27a68c7e82c9aa05
```

**Deterministic.** The attribution is identical across two independent device runs, so the
divergences are neither a race nor uninitialised scratch — they are a reproducible difference
between the compiled device arm and the host twin. That retires the cheapest explanations and
makes the remaining ones worth chasing.

Field-level, device-vs-host:

```
g_policy_a_hip_device_vs_hip_hostpath  ndiff_bytes=3,744  sum_of_fields=3,744  accounting=ok
  gate 0  up 0  s 58 (max_bit_step 2)  raw 2,846 (max_bit_step 24,576)  routed 840 (max_bit_step 768)

d_hip_device_vs_d_hip_hostpath         ndiff_bytes=52,976  sum_of_fields=52,976  accounting=ok
  q_hat 0  k_hat 0  core_attn 0  normed 0  y 0
  scan_out 481 (max_bit_step 5,632)   state 52,495 (max_bit_step 4,096)
  DHEAD heads=32 per_head_ndiff_bytes=0,0,0,0,0,0,0,0,0,10559,0,0,10906,0,10114,10080,…
```

What that rules in and out:

- `g_ref_bf16_hip_device_vs_hip_hostpath` is **0 on every field** — the reference gather is
  byte-exact, so the device toolchain is not broadly broken.
- The policy-A gather diff is confined to the quantized expert path and `raw`'s
  `max_bit_step=24,576` is **not ULP-scale**, so this is not rounding.
- The GDN diff is confined to `scan_out` and `state`, with `q_hat`, `k_hat`, `core_attn`,
  `normed` and `y` bit-identical, and `DHEAD` shows the state difference concentrated in a
  **subset of heads** at ~10 KB each rather than spread uniformly.

Both pairs account exactly (`sum_of_fields == ndiff_bytes`), which is what makes the per-field
claim checkable rather than a total with a story attached. Nothing here is absorbed: no field
leaves claimed-exactness, no band is authored, and the mechanism stays **unattributed** until an
injection reproduces both the magnitude and the field split.

## Window `i` — 2026-10-27 — both divergences attributed by whole-record byte identity

Third window, two identical `--device` runs, zero workers, lock tag asserted, engine-down from
`rocm-smi`. **No timing taken.**

```
GREPRO pair=g_policy_a_hip_device_vs_hip_hostpath probe=exp_ulp_table path=host_cpubody
       sites_perturbed=58 explained_by_exp_within_4ulp=58 unexplained=0
       min_steps_histogram=-1:42,+1:16 record_bytes=122,880 vs_device_record_ndiff_bytes=0
       verdict=EXACT-REPRODUCTION
DREPRO pair=d_hip_device_vs_d_hip_hostpath probe=decay_ulp_mask steps=-1 mask=0x80d200
       heads=9,12,14,15,23 record_bytes=2,187,264 vs_device_record_ndiff_bytes=0
       verdict=EXACT-REPRODUCTION
DREPRO pair=d_hip_device_vs_d_hip_hostpath probe=decay_ulp_mask steps=+1 mask=0x80d200
       heads=9,12,14,15,23 record_bytes=2,187,264 vs_device_record_ndiff_bytes=75,206
       verdict=RESIDUAL-LEFT
RUNG-1 constant across runs: ok     band_digest=27a68c7e82c9aa05
```

This is attribution, not a story. Perturbing 58 `exp()` sites on the **host** path by the device's
own measured ULP table reproduces the device's gather record **byte for byte**; applying **−1 ULP**
to `exp(g_decay)` on heads 9, 12, 14, 15, 23 reproduces the 2,187,264-byte GDN record exactly. The
`+1` arm is what makes it a test rather than a fit: the opposite direction leaves **75,206 B** of
residual, so the reproduction pins the *sign* and the *head subset*, and could not have been tuned
to any 1-ULP perturbation. `unexplained=0` with `min_steps_histogram=-1:42,+1:16`.

Conclusion: the device-vs-host difference is a last-bit library difference at `exp` sites
(`ocml` on the device, `glibc` on the host), and nothing else — the bf16 reference gather, which
touches no `exp`, is 0 on every field, and my width hypothesis was **refuted** (`width_misread`
injects 73,475 B, 19.6x, and cannot reach field `s`, which the device shows differing by 58 B).

## What is still wrong, stated before any celebration

1. **The run still crashes before printing `RESULT`** — `rc=1`, both runs. Now at line 4206 in
   `checks_d`: `r["run"]["launches"] > 0` → `TypeError: '>' not supported between 'str' and 'int'`.
   Fourth instance of the same class, and structurally unavoidable in a CPU-only workflow:
   `device=off` never executes these lines, so the lane's green `checks=54 findings=0` is *true and
   blind*. The two device-only defects found today (`_bytes` KeyError, tuple-into-dict-helper at
   2405, this `str`/`int` one) were all invisible to every gate the lane could run.
2. **One control contradicts the headline in the same log.**
   `CONTROL name=gdn-divergence-causes-priced fired=SILENT reason=no injected cause reproduces the
   GDN device divergence …` while `DREPRO … steps=-1 … verdict=EXACT-REPRODUCTION` says one does.
   A control whose cause table predates the probe that answered the question prints a confident
   falsehood; it must be re-pointed at the cause that now reproduces, and it must redden if that
   cause ever stops reproducing.
3. **Two controls print `fired=SILENT` with no `reason=`** (`gather-divergence-causes-priced`, with
   legs `host_cpubody:25,policy_a:13,ref_bf16:12`, and `gdn-state-read-is-the-arm-s-own-fingerprint`,
   legs `device:1,host_cpubody:2`) even though a check named
   `silent-control-names-its-reason` exists. So that check is not looking at controls, only at
   checks — which is exactly the gap it was written to close.

Until those three are closed the device arm is not "green": it is **attributed**, which is a
different and weaker thing. `band_digest` unchanged; no field left claimed-exactness; no band
authored; the `exp`-scope question this raises is a decision, not a tolerance to be slipped in.

## Window `j` — 2026-10-27 — the device run reaches a verdict for the first time, and it is RED

Fourth window. Scratch purged first (`find … -name '*.bin' -delete`) because a rehearsal child had
written fabricated device-shaped records into `engine/work/m2dev` — the lane disclosed this itself,
and the rehearsal now runs in its own `--work m2dev-rehearsal` directory behind a guard that aborts
if any HIP library is mapped. Arms rebuilt from source, then two identical `--device` runs.
**No timing taken.**

```
RESULT mode=device checks=63 findings=4 controls=32 fired=30 device=on staged=admitted
       hip_calls=19+448 launches=146 bytes_to_device=839,032,304 bytes_from_device=20,357,384
       counters_unparsed=0   (Traceback count in log: 0)
GREPRO … probe=exp_ulp_table sites_perturbed=58 unexplained=0 histogram=-1:42,+1:16 … EXACT-REPRODUCTION
DREPRO … steps=-1 heads=9,12,14,15,23 … vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION
DREPRO … steps=+1 … vs_device_record_ndiff_bytes=75,206 verdict=RESIDUAL-LEFT
DREPRO-WIN … residual_bytes=0 verdict=EXACT note=frozen winning combination: this leg reddens if the cause is withdrawn
RUNG-1 constant across runs: ok     band_digest=27a68c7e82c9aa05
```

The crash class is closed: the run prints a verdict, `counters_unparsed=0`, and the pass-9 typing
checks pass on real device output (`device-launch-counts-are-ints-and-positive` across 15 arm
records; `device-run-counters-are-typed-at-the-parse-site` with zero casts and zero inline quoted
comparisons; `device-only-paths-are-enumerated-not-assumed` with an AST census of 106 statements).
The attribution survived a rebuild from source and a purge of the contaminated scratch.

The verdict is **RED by exactly the two attributed divergences** — `gather-compiled-pair-byte-
identical-per-path` (policy_a ndiff 3,744, ref_bf16 0) and `gdn-compiled-pair-byte-identical`
(device vs host 52,976) — which is the honest state, and better than a PASS that cannot be believed.
Six of seven `fired=SILENT` lines carry a `reason=`; the seventh is the fabricated `quiet-no-reason`
whose whole purpose is to make the guard redden.

## Two defects the window exposed in the new instrumentation

1. **`no-control-contradicts-an-exact-reproduction` printed `verdict=PASS` claiming `0`
   reproduction legs, in a log that contains three.** It parses `ctx["text"]`, a buffer captured
   before the reproduction lines are printed. The check that exists to catch the window-`i`
   contradiction is itself vacuous in the window where it matters — the same failure mode as
   `parsed=0` printing PASS, arrived at by a different route (ordering, not parsing).
2. **Two tallies disagree**: `RESULT … findings=4` while a summary line in the same log says
   `findings=2`, and three `verdict=FAIL` lines are present. A terminal verdict must state its own
   finding count, and one number reaching four while another reaches two means neither is checked.

Nothing is licensed here, no field left claimed-exactness, and the `exp` scope remains a draft.
