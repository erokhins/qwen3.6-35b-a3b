# Record — `engine/block_device.py` hygiene, PASS 16 (the two window-`o` defects)

Written by the worker at ~minute 32 (checkpoint) and completed at the end of the run. Scope:
`engine/block_device.py` plus new files under `specification/evidence/` only. **No iGPU: no device flag, no
kernel launch, no `hipcc` run, no GPU stream lock** (`grep -c gpu_lock engine/block_device.py` = 0, and this
file and the harness spawn nothing at all). Host wall and RSS only. **No timing taken.**

Both defects are the reader's, not the measurement's: window `o`
(`devices/evidence/device-window-2026-10-27-g.md` §"Window `o`") was verified by a reader and the two lines
that misled it were both lines whose job is to state a verdict.

## Fix 1 — the state-2 token, and the state machine tested against the formatter

Before / after, in `Verifier.control`:

```python
# before — engine/block_device.py §Verifier.control, the field window `o` named
            notecount = " status=not-exercised%s%s" % (
                ",".join(sorted(not_exercised)), notecount)
# after
            notecount = NOT_EXERCISED_FIELD % (",".join(sorted(not_exercised)), notecount)
NOT_EXERCISED_FIELD = " status=not-exercised paths=%s%s"          # §NOT_EXERCISED_FIELD
```

Window `o` printed `status=not-exerciseddevice` twice; the same registration now prints
`status=not-exercised paths=device reason=needs-a-device-window`.

The reader's half is new (§`control_state`): `control_state_of_line`, `control_states_from_lines`,
`control_state_contract_findings`, `malformed_not_exercised_lines`. `control_state_of_line` is deliberately
stricter than `control_state` — a `status=not-exercised` whose path list is not a well-formed `paths=` field
is **not** state 2, it is the third state with sub `malformed-not-exercised-token`, and `control_state_findings`
counts it as a finding.

New check `control-printed-lines-parse-back-to-their-registered-state` (§`check_control_line_formatter_contract`,
called at the end of `check_control_states_and_declaims`):

* one fabricated registration per state through the **shipped formatter**, read back out of this run's own
  printed output — `fired` / `not-applicable` / `claimed-and-silent` each parse to what they registered as;
* the not-applicable line with the separator deleted (window `o`'s exact token, rebuilt from the printed line
  so it cannot drift) → third state, `malformed-not-exercised-token`, named as a finding; and the same line
  with the whole field dropped → third state, `silent-without-declaration`, also a finding;
* the whole registered population re-derived from its own printed lines (31 in the CPU witness, 32 in the
  device-shaped rehearsal), plus a non-inertness witness that flips one real control's fire word and must
  make the same predicate name that control.

## Fix 2 — the tally no longer prints the word it counts, and a guard makes the class impossible

Re-worded (the same change in `reject_tally_findings`' three finding strings and in
`check_control_states_and_declaims`' detail, so no printed line carries either token):

```
before:  ...reconcile to it: %d CHECK verdict=FAIL lines (%s) plus %d registered unfired controls,
         each printed as `fired=SILENT` and counted as a finding (%s)...
after:   ...reconcile to it: %d F-A-I-L CHECK records (%s) plus %d registered unfired controls,
         each printed as a S-I-L-E-N-T control record and counted as a finding (%s)...
```

New guard `printed-verdict-tokens-match-reported-tallies` (§`check_printed_verdict_token_tally`), the last
check the run prints, over the pure predicate `verdict_self_match_findings`. It reads the run's own printed
output with the **reader's** loose patterns (`CHECK_FAIL_MENTION`, `CONTROL_SILENT_MENTION` — a reader will
not write the field-anchored form) and requires:

* every line those patterns match to also match the record form (no mention may wear a record's clothes);
* the record set to equal the set `check_terminal_tally` reports, both ways — an uncounted record and an
  unrecorded count both condemn, the silent half compared by registered name because scratch witnesses
  print silent records on purpose;
* no record line's message to contain either counted token;
* `printed-check-lines == registered checks` and non-zero, and `printed-control-lines >= registered controls`,
  so a parse that saw nothing cannot pass.

Non-inertness, six fabrications through the same predicate, printed in the run: the window-`o` wording
re-spliced onto the real tally line, prose quoting a silent control inside a fired record, a genuine failing
record the tally omits, an uncounted silent record for a registered control, a counted control with no
record, and the honest pair staying clean. `mask_counted_tokens` keeps the tokens out of the findings *about*
them, and the mask is witnessed catching 2/2 on a synthetic token-bearing line so `self-token-free` is not a
constant.

## Evidence

| file | what it is |
|---|---|
| `blockdevice-cpu-2026-10-27-p16.log` | the fixed CPU witness, `RESULT mode=cpu checks=75 findings=0 controls=31 fired=31 counters_unparsed=0` — run three times, byte-identical apart from pid, `work=` and the arm `out=` paths |
| `blockdevice-hygiene-p16-2026-10-27.py` | proof harness: 17 legs over window `o`'s own log and a suite a real `Verifier` printed |
| `blockdevice-hygiene-p16-proof-2026-10-27.log` | its output: `HARNESS-P16 verdict=PASS failures=0 legs=17 unique-legs=17` |
| `blockdevice-device-shape-rehearsal-2026-10-27-p16.log` | the device-SHAPE rehearsal after the change: `REHEARSAL verdict=PASS no device act occurred, no timing taken` |

Reader greps, before → after, on the CPU witness: `^CHECK name=.*verdict=FAIL` **1 → 0**;
`^CHECK name=.*fired=SILENT` **2 → 0**; `grep -oE "status=not-exercised[a-z_]*"` **17 clean / 2 malformed →
23 clean / 0 malformed**. In the device-shaped rehearsal the loose FAIL count (3) equals the tally's
`reported-fail=3`, i.e. the guard holds in a red run too.

Unchanged against the pre-change baseline run: BOUNDWITNESS 38 (all `inside=yes`), BOUNDEDGE 12
(`at-limit=True` ×6, `one-past=False` ×6, `sized_correctly=yes` ×18), MUSTCATCH 8 `reddens=RED`,
REFUTED-CAUSE 2, `REFUSED` 0 — all byte-identical lines. `band_digest=27a68c7e82c9aa05` everywhere.
`BLOCKGATE … checks=17 red=0 controls=11 fired=11`; `engine/gate.py` `RESULT PASS checks=27 findings=0`;
`tools/doc_ref_check.py --strict` `findings: 0`.

Nothing was widened, no field left claimed-exactness, no fixture replaced and no gate loosened: both new
checks only add predicates that can redden.
