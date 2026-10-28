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

## Window `k` — 2026-10-27 — the two check-layer defects closed on live device output

Fifth window: scratch purged, arms rebuilt, two identical `--device` runs under tag
`q36-main window-2026-10-27k pass9-device`, engine-down read from `rocm-smi`. **No timing taken.**

```
BLOCKDEVICE verdict=FAIL … checks=64 red=4 controls=32 fired=30
RESULT     mode=device     checks=64 findings=4 controls=32 fired=30 device=on staged=admitted
CHECK name=no-control-contradicts-an-exact-reproduction verdict=PASS 2 reproduction legs printed
      `verdict=EXACT-REPRODUCTION` (pairs: d_hip_device_vs_d_hip_hostpath,
      g_policy_a_hip_device_vs_hip_hostpath) and 1 printed a residual
GWITNESS guard=attribution-contradiction leg=this-run lines=434 exact_pairs=d_hip…,g_policy_a… findings=none
GWITNESS guard=tally-reconciliation findings=4 fail-check-records=2 unfired-control-records=2
      fabricated=missing-record flagged=True fabricated=extra-print flagged=True
RUNG-1 constant across runs: ok     band_digest=27a68c7e82c9aa05     Traceback count: 0
```

The ordering vacuity is gone: the check reads the complete printed buffer and reports **2 legs with
their pair names**, and the zero-count-on-device case is now itself a finding rather than a pass.
The two terminal lines agree with each other (`64/4` and `64/4`), and the reconciliation is stated
rather than asserted: `findings=4` is 2 `verdict=FAIL` records plus 2 registered-but-unfired
controls, with the guard's own fabrications flagged `True` so the equality cannot be vacuous. The
only reason-less `fired=SILENT` is `quiet-no-reason`, which exists to make the guard redden.

`findings=4` remains the honest RED of the two attributed divergences; nothing was licensed to
clear it. **My own error, recorded:** the script for this window was derived from window `j` with
`sed 's/2026-10-27j/2026-10-27k/g'`, which renamed the *tag* (so preconditions were correctly
declared and asserted) but not the log filenames, because those read `…-2026-10-27-j1.log` and the
pattern missed the hyphen. Window `k` therefore overwrote window `j`'s scratch logs; the committed
copies under `specification/evidence/` were untouched and the files have since been renamed and the
script corrected. A derived script has to be checked for what it did *not* substitute.

## Window `l` — 2026-10-27 — D-28's value bands on real device output: every measurement inside, one metric with no teeth

Sixth window: scratch purged, arms rebuilt, two identical `--device` runs, tag asserted, engine-down
from `rocm-smi`. **No timing taken.**

```
RESULT mode=device checks=71 findings=3 controls=32 fired=30 device=on staged=admitted counters_unparsed=0
gather-compiled-pair-value-agreement-per-path  verdict=PASS   (policy_a bytes=122,880 ndiff=3,744 [banded])
gdn-compiled-pair-value-agreement              verdict=PASS   (host pair ndiff 0; device byte count 52,976 reported not judged)
BOUNDWITNESS 36 cells, outside=none, inside=yes on all of them
MUSTCATCH 8 injected real defects, all reddens=RED
CHECK d28-escalation-clause-refuses-qualification verdict=PASS  (fabricated pass-with-foreign-mechanism is refused)
```

Measured against derived, with the margin printed rather than implied:

| field | metric | measured | derived | margin |
|---|---|---|---|---|
| gather `s` | rel-err | 1.368e-09 | 3.620e-08 | 26.5 |
| gather `raw` | rel-err | 8.455e-08 | 6.262e-07 | 7.41 |
| gather `raw` | ndiff_frac | 0.1700 | 0.8555 | 5.03 |
| gather `routed` | rel-err | 2.232e-07 | 3.249e-07 | **1.46** |
| gather `routed` | ndiff_frac | 0.4019 | 0.8589 | 2.14 |
| GDN `state` | relL2 | 3.136e-08 | 1.743e-07 | 5.56 |
| GDN `state` | ndiff_frac | 0.0994 | 0.6951 | 6.99 |

The tightest cell (`routed` rel-err, 1.46×) is only just inside, which is what a derived bound should
look like — a band with 100× headroom everywhere would be measuring nothing.

**The run is still FAIL, and on the rejection side, not the agreement side.**
`d28-value-band-witness-inside-and-edge-legs-reject verdict=FAIL` reports the edge legs as
`g/ndiff_frac at-limit=False` and `d/ndiff_frac at-limit=False` while the other four metrics behave
(`at-limit=True one-past=False`). So the `ndiff_frac` metric has **no demonstrated power to reject**:
its "sized at the bound" fabrication already lands outside the bound, most likely because the metric
is quantised in whole words (1/4096 = 2.44e-04) and the sizing crosses the boundary on the way up.
A band that cannot reject is not a band, so this stays red until the legs are sized so that one leg
sits at the last word inside and the next sits at the first word outside.

**One clause of D-28 is not satisfiable as written, stated here rather than buried.**
`REFUTED-CAUSE cause=decay_ulp1 … teeth=none outside=none` — in value space the refuted whole-record
`decay_ulp1` cause is no longer distinguishable, because value bands are deliberately looser than byte
identity. It was refuted on *byte* magnitude (264,704 against 52,976), and that axis still distinguishes
it — the `GATTRIB`/`DATTRIB` byte counts are still printed as diagnostics. So "the two refuted causes
remain live controls that must keep failing" is carried by the diagnostic byte axis for `decay_ulp1`,
not by the value band, and `silu_exp_ulp1` is carried by both (`teeth=yes outside=routed/rel-err
5.65e-07 vs 3.25e-07`). The clause is amended to say which axis carries which cause; it is not relaxed,
because the byte diagnostic still fails that cause.

## Window `m` — 2026-10-27 — a refusal aborts the suite: `checks=0` is not a pass, and it nearly read as one

Seventh window, same discipline. The CPU witness is green (`checks=69 findings=0 controls=31 fired=31`,
`BLOCKDEVICE verdict=QUALIFIED`), all six rejection summaries behave (`at-limit=True one-past=False
sized_correctly=yes`), `ndiff_frac` now lands exactly on the lattice (`at-limit measured=0.368164 ==
bound`, `one-past=0.368408`, `lattice_step=word`), and `granularity` is measured rather than typed.

The device run is **not** green, and the way it failed is worth recording:

```
RESULT mode=device checks=0 findings=1 controls=0 fired=0 rows=0 arms=policy_a device=on
       staged=admitted hip_calls=16+448 launches=146 counters_unparsed=0
last lines:  BOUNDEDGE … (6 legs, gather only)
             REFUSED tag=d28-edge-leg-wont-size note=[d28-edge-leg-wont-size] the scan_out/rel-err
             inside leg could not be placed on its own side of the bound 1.46299e-07 …
tracebacks=0   printed CHECK lines=0   printed CONTROL lines=36   printed MUSTCATCH=0   REFUTED-CAUSE=0
```

`grep -E "verdict=FAIL"` on this log returns **nothing**, and that is the oldest trap in this repo:
a run that executes no checks produces no failures. It is not a green, it is a run that died before
the check suite — `checks=0` with the tally check itself among the never-run checks. The device-pair
`BOUNDWITNESS` legs (36) and the gather edge legs printed; the GDN edge leg raised the lane's own
`REFUSED … wont-size` guard, the guard propagated out of `run()`, and everything after it — the
remaining edge legs, the whole `MUSTCATCH` population, the `REFUTED-CAUSE` lines, every `CHECK` —
never ran.

Two things follow, neither of which is a widening:

1. A refusal must be a **counted finding on the leg that refused**, never an exception that ends the
   run. And a device run with `checks == 0` must be fatal in a way that cannot be misread: the driver
   must say so and exit non-zero with a named reason, because "nothing failed" and "nothing ran" have
   been indistinguishable here before.
2. The refusal is itself evidence. `scan_out/rel-err` has bound `1.46299e-07`, and on this substrate
   the at-limit lattice point could not be reached by *adding* divergence — which is what happens when
   the base divergence already exceeds the bound. The CPU-only run placed that leg inside
   (`measured=1.097e-07` vs `1.463e-07`), so the question is what the device substrate changes. The
   answer must be measured and reported; if the device's `scan_out` really sits outside the derived
   bound, that is an escalation under D-28, **not** a licence to widen it.

## Window `n` — 2026-10-27 — the suite runs to completion on device: 75 checks, zero failures, two silent controls

Eighth window. CPU witness `checks=71 findings=0 controls=31 fired=31`, `BLOCKGATE checks=17 red=0`,
M1 `RESULT PASS checks=27 findings=0`. Device run:

```
RESULT mode=device checks=75 findings=2 controls=32 fired=30 device=on staged=admitted counters_unparsed=0
printed CHECK lines=75, of which verdict=FAIL=0
BOUNDEDGE=12  MUSTCATCH=10 (8 reddens=RED)  REFUTED-CAUSE=2  REFUSED=0
BOUNDWITNESS inside=yes on all cells, inside_no=0
scan_out/rel-err measured=2.43282e-08 vs derived=1.46299e-07  margin=6.014  inside=yes
gather routed/rel-err measured=2.23221e-07 vs 3.24921e-07  margin=1.456  inside=yes
```

Window `m`'s refusal is gone: `REFUSED` lines = 0, and every section that previously never ran now
does. The `scan_out` diagnosis came back **construction-wrong, not bound-breach** — the old closed form
sized the leg from `b_site` alone, i.e. from base 0, which is the only base a byte-identical host twin
can produce, so it landed `bound+base` (1,242 one-ULP steps past a 64-step budget). Sizing now solves
`bound − floor_metric`; it lands the same point on CPU and inside on window `m`'s real device bytes.
**Nothing was widened and no escalation was needed**; the device base sits inside the bound by 6.014×.

**Where the two findings actually come from, because "no FAIL lines but findings=2" is exactly the
shape I have been burned by twice.** They are not fabricated-witness residue (the stale-text witness
counts locally, and the tally line says so explicitly). They are **two registered controls that never
fired**: `control-witness-guard-can-redden` and `gdn-state-read-is-the-arm-s-own-fingerprint`, each
printed `fired=SILENT` and counted as a finding under the rule that a control which claims legs and
stays silent is not neutral. So the device arm's *checks* are clean and the arm is two silent controls
away from `QUALIFIED` — the same `control-witness-guard-can-redden` that was silent in window `g`, still
silent on the device path specifically.

## Window `o` — 2026-10-27 — **M2's device arm reaches `QUALIFIED`**

Ninth window, same discipline (scratch purged, arms rebuilt, two identical runs, tag asserted against
the expected string, engine-down read from `rocm-smi`). **No timing taken.**

```
RESULT mode=device checks=77 findings=0 controls=32 fired=32 device=on staged=admitted counters_unparsed=0
BLOCKDEVICE verdict=QUALIFIED … checks=77 red=0
CONTROL name=control-witness-guard-can-redden          fired=RED legs=cpu_predicate:6,device:10 claims=device
CONTROL name=gdn-state-read-is-the-arm-s-own-fingerprint fired=RED legs=device:1,host_cpubody:2 claims=host_cpubody,device
CHECK name=control-states-enumerate-and-claimed-and-silent-reddens verdict=PASS … fired=32, not-applicable=0, claimed-and-silent=0
CHECK name=control-claims-do-not-quietly-shrink       verdict=PASS (baseline: the 36 controls of n1)
BOUNDEDGE=12  MUSTCATCH=10 (8 RED)  REFUTED-CAUSE=2  REFUSED=0  printed CHECK verdict=FAIL=0
```

Both controls that window `n` left silent now execute on the device path, and the control whose
fingerprint leg had been inverted since window `h` judges each flip leg against **its own arm's**
honest record instead of a cross-path baseline — which is what window `n`'s "7 legs refused" really
was: four of those refusals were the control comparing a host flip against the device arm's number.

**Two hygiene defects found while verifying, neither fatal, both recorded.**

1. `status=not-exercised%s%s` at `engine/block_device.py:603` concatenates the next field with no
   separator, so two controls printed `status=not-exerciseddevice`. Both are fabrications in a scratch
   verifier, so nothing was miscounted **this** run — but a real control taking the not-applicable state
   would emit a token that is neither `not-exercised` nor absent, and the three-state rule depends on
   reading that token. A state machine whose state is parsed out of a formatted string must be tested
   against the formatter.
2. The terminal-tally check's own message contains the literal text `0 CHECK verdict=FAIL lines`. My
   verification grep `^CHECK name=.*verdict=FAIL` matched it and reported "one FAIL record" in a run with
   zero — a false alarm aimed at exactly the reader who is checking whether anything failed. A verdict
   line should not print the word it is counting; I counted with a stricter predicate and it went away,
   but the wording invites the mistake.

M2's exit gate is therefore met: rows ACCEPTED against pinned under licensed bands, CPU and device
halves both `QUALIFIED`, with D-28's value-space judgement replacing byte identity only for the
arm-against-its-own-CPU-twin relation. Everything that measures us against the oracle is unchanged.
