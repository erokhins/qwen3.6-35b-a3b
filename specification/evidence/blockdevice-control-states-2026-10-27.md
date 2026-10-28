# Control states: a silent control is either not-applicable or a finding (2026-10-27, CPU-only)

Two controls printed `fired=SILENT` while claiming legs in `specification/evidence/blockdevice-device-2026-10-27-n1.log`
(`RESULT mode=device checks=75 findings=2 controls=32 fired=30`, zero `verdict=FAIL` except those two controls).
This note records what each one's real defect was, the state rule that keeps a silent control from being a
reading exercise, and the legs that show each rule can still redden. No device act occurred here and no timing
is printed by anything in this file: no timing taken.

## The rule, in one place

`control_state` (`engine/block_device.py`, beside `silent_control_finding`) assigns every registered control
exactly one of three states, and `Verifier.control` stores the assignment on the line that prints it:

| state | printed shape | meaning |
|---|---|---|
| `fired` | `CONTROL name=X fired=RED legs=<path>:N claims=<path>` | it reddened |
| `not-applicable` | `fired=SILENT … status=not-exercised reason=<named>` and it claims none of those paths | it could not run here, and says which path |
| `claimed-and-silent` | `fired=SILENT` with a claim, or silent with no declaration at all | **a finding**, named with its sub-shape |

Two deliberate choices. An unfired control stays a counted finding whether or not it declared itself
not-applicable, so `status=not-exercised` lowers no counter and cannot be used to make a count go away; and
the state is decided once, by the printer, so the enumeration check and the `CONTROL` line cannot disagree
about who is in trouble. The sub-shapes are `claimed-and-silent`, `claims-a-path-it-declared-unexercised` and
`silent-without-declaration`.

Two checks enforce it, both in `check_control_states_and_declaims`, which runs from `check_control_paths`
after every control has registered:

* `control-states-enumerate-and-claimed-and-silent-reddens` — prints `fired=N, not-applicable=M,
  claimed-and-silent=K (offenders: …)` over the whole registered population and reddens on any offender. Six
  fabricated registrations (through the real `Verifier`) are the non-vacuity leg: the two honest states read
  clean and all four third-state shapes are flagged.
* `control-claims-do-not-quietly-shrink` — the baseline is the claim set parsed at run time out of the last
  recorded log of the same mode (`CONTROL_CLAIM_BASELINE`: `blockdevice-cpu-2026-10-27-p13.log`,
  `blockdevice-device-2026-10-27-n1.log`), not a list copied into the source. A claimed path that is not
  claimed here must appear as `declaimed=<path> reason=<named>` on the line that dropped it; a blank reason
  reddens, and so does a control that vanished. The parse must see every `CONTROL` line in that log
  (35 parsed / 35 printed in the CPU baseline) because a claim guard whose parser matched nothing guards
  nothing. Five fabricated cases are the non-vacuity leg (GWITNESS guard=claim-shrink).

## control-witness-guard-can-redden — SILENT on the device path since window `g`

Two separate defects, both in the control and neither in the predicate it witnesses.

1. One flat must-catch list over three predicates. It asked `launch_count_offences` — which reads `launches`
   and nothing else — to catch `device/bytes_to_device=1GiB`, so window `n` printed
   `uncaught=['device/bytes_to_device=1GiB']`. A shape a predicate cannot see is not that predicate's failure
   to catch it. `pitfalls/40-gates-and-admissions.md` §a-negative-control-with-an-empty-must-catch-set, read
   the other way: a must-catch set naming shapes outside the predicate's own field list is just as dead.
2. The fabrications were cloned from `sorted(honest)[0]`, which on a device window is the *device* record. So
   the host-half fabrications cloned a device record (wrong base, and vacuously caught), and the
   device-half predicate `device_act_absences` had no fabrication of its own — it only ever saw an act-free
   host base, which is why the CPU witness "passed" this control since window `g` for the wrong reason.

The fix gives each predicate the fabrications its own fields own: `no_device_act_offences` gets acts ADDED to
a host record (`launches=1`, a gigabyte each way, `hip_calls=1`, the counter unprinted),
`device_act_absences` gets acts REMOVED from a device record (`bytes_from_device=0`, `launches=0`, the counter
unprinted) and is judged against a real device record when one exists, and `launch_count_offences` gets the
three `launches` shapes plus the one shape that must stay clean (`launches=1`). Where no device record exists
(CPU witness) the base is labelled `synthetic-device-record reason=no-device-act-in-this-run`: bending a base
that was already an offence catches nothing, which is exactly how the CPU run was passing. Legs are counted
per path — `device:N` only for legs that read a record whose own `RUN path=` says device — and the control now
claims `device` when device records exist.

## gdn-state-read-is-the-arm-s-own-fingerprint — 7 legs refused, and 7 of the reasons were lies

Reported picture from window `n`: `fired=SILENT legs=device:1,host_cpubody:2 claims=host_cpubody,device`
with `reason=7 leg(s) refused`. All seven refusals were defects in the control:

* Three of them were `fingerprint-disagreed` printed next to `agrees=True`. The refusal table listed
  `("fingerprint-disagreed", not still_agree)` and appended when its value was *true*, so the control
  reddened the legs that agreed and stayed quiet about one that did not. The condition now names what went
  wrong: `still_agree is False`.
* Four were a cross-path baseline. `hm` was `ctx["d_honest_metrics"]` — the headline record, which in a window
  is the device arm's (`state` ndiff 120259) — while the g++ twin's own honest record is 85187. Judging the
  twin's flipped 85188 against 120259 printed `state-row-did-not-move (state_ndiff 120259->85188)` and
  `a-state-derived-row-got-closer-to-the-pin` for a flip that moved by exactly one word away from the pin.
  Every flip leg is now judged against the honest record of the arm that ran it (`d_out_host.bin`,
  `d_out_hip.bin`, `d_out_device.bin`), and each leg row prints which baseline it used (`honest=d_out_*.bin`).
  Same defect class `topk-selection-perturbations-redden-ids` names from window `e2`: a cross-path comparison
  printed as a refusal of the leg instead than of the baseline.

## Evidence

* CPU witness: `blockdevice-cpu-2026-10-27-p14.log` —
  `RESULT mode=cpu checks=73 findings=0 controls=31 fired=31 … band_digest=27a68c7e82c9aa05`. Both controls
  `fired=RED`; `control-witness-guard-can-redden` carries `legs=cpu_predicate:15 claims=none` (no device
  record exists to read in this mode) and `gdn-state-read-…` carries `legs=host_cpubody:2` with both flip legs
  reading their own baselines (`state_ndiff 85187->85188 honest=d_out_host.bin`, `… honest=d_out_hip.bin`).
  Gates beside it: `BLOCKGATE … checks=17 red=0 controls=11 fired=11`, `engine/gate.py` →
  `RESULT PASS checks=27 findings=0`, `tools/doc_ref_check.py --strict` → `findings: 0`. BOUNDWITNESS/BOUNDEDGE
  /MUSTCATCH line counts are unchanged from the pre-change baseline log (38/12/10; six
  `at-limit=True one-past=False sized_correctly=yes`; eight `MUSTCATCH name=` RED legs; `REFUSED=0`).
* Device-shaped rehearsal (no iGPU: stub device, host bodies, counters re-spelled from window `i`, every
  substitution labelled): `blockdevice-device-shape-rehearsal-2026-10-27-p14.log` —
  `RESULT mode=device checks=77 findings=7 controls=32 fired=28`, rehearsal `verdict=PASS`, and its
  `launches-abc` and `silent-control` mutation arms both reach `RESULT` with the target named on the RESULT
  line. `control-witness-guard-can-redden` prints
  `fired=RED legs=cpu_predicate:6,device:10 claims=device` — device-path legs, non-zero, on the path it claims.
* `gdn-state-read-…` in the rehearsal prints `fired=SILENT legs=device:1,host_cpubody:2 …
  reason=1 leg(s) refused: d_hip_device: the arm printed no DREADBACK path=device line` and the new state
  check reddens it by name — the honest answer for a rehearsal: the remaining leg is the device read-back,
  which a host body is not allowed to fake (`engine/device/m2_d_main.inc` §the read-back leg prints only where
  a device copied bytes back). In window `n` that leg already printed and matched
  (`DREADBACK control-leg=gdn-device-readback ran=yes … match=yes`), and with the per-arm baselines its three
  flip legs are the numbers that window already printed as refusals: host 85187→85188, hip host path
  85187→85188, device 120259→120260. The control's device-run `CONTROL` line therefore has to come from a real
  window; printing it here would be quoting a line this session is forbidden to produce.

## Residual risks

* The device leg of `gdn-state-read-…` is fixed by arithmetic on window `n`'s own printed numbers, not by a
  run in this session: the only thing that can still refuse it in a window is a real regression (a flipped
  state word moving a derived row nearer the pin on the device record), which is what the leg exists to catch.
* The state check treats a silent control with no declaration as an offender, so any future control that
  legitimately cannot run in a mode must print `status=not-exercised reason=<named>` and drop that claim; the
  de-claim guard is what makes dropping a claim cost a sentence.
