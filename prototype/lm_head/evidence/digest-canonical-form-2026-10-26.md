# The R line's canonical text form, and the wrong-radix control (2026-10-26)

Scope: `prototype/lm_head/` only. CPU only — no device run, no timing, no `ms`/`tok/s`/`GB/s` figure
appears in this file or in any log under `tmp/worker-lm-head/`. Nothing staged, nothing committed.

## The bug this closes

`lmh_write_report` printed the two bit-image digests with `%u` (decimal) while `gate.py`'s R-line
consumer read the same columns as hex. `0xFF400000` and `4278190080` are one 32-bit value spelled two
ways, and **both parse to a plausible number**, so a check that only compares values cannot see the
disagreement in either direction — every digest comparison in the lane was meaningless whichever way it
erred. Same class as a ULP ruler being one step lax across zero, and the same two-rulers shape as the
gather lane's `CENSUS`/`ARMDIFFSUM` lines.

Before this change the consumer parsed **all** fields decimal (`int(r["oxor"])`, gate.py §"the report's
numeric fields are ALL decimal"), so a hex token would have raised `ValueError` — a traceback, not a
verdict — and a decimal-spelled digest was silently believed.

## The rule (one canonical form per field, declared once)

`lm_head_plan.h §lmh_r_fmt[]` is the single declaration. Three kinds:

| kind | spelling | fields |
|---|---|---|
| `hex`  | `0x` + exactly 8 UPPER-CASE hex digits | `oxor`, `xorb` — the two **bit images** |
| `real` | what `printf %.9g` emits (incl. `inf`/`nan`) | `osum` — the one sum of decoded VALUES |
| `dec`  | `0 \| [1-9][0-9]*` — no sign, no prefix, no leading zero | everything else, incl. `vsum`, `bsum`, `bytes`, `loads`, `lo`, `hi`, `cross`, `peer_hits`, `lanes`, `bover`, `launches` and every echoed plan field |

Why this form was picked, deliberately:

* a bit image is read as bits, so it gets the spelling that says so — the `0x` prefix **puts the radix
  in the string**, which is what turns a producer/consumer radix drift from a plausible wrong number
  into a token that matches no accepted pattern;
* `vsum` and `bsum` stay decimal because they are *integer sums* (of bit images and of raw storage
  bytes): `0x123456789ABC` invites a bit-wise reading of a quantity that only means anything summed,
  and the driver's own report-contract check parses that column with `strtoull(..., 10)`;
* the leading-zero/sign ban on `dec` closes the softer drifts (`007`, `-0`, `+4`) that a "looks like a
  number" test wave through.

## Both sides assert it

* **Producer** (`lm_head_plan.h`): `lmh_hex32()` renders `oxor`/`xorb`; every other numeric column is
  `%u`/`%llu`/`%.9g`. The self-test reads its OWN `selftest_report.tsv` back and requires every token
  to satisfy `lmh_fmt_ok(lmh_r_fmt[i], token)` — a validator written separately from the writer, so a
  drift to a bare `%u`, to lower case, or to a leading zero fails rather than agreeing with itself.
* **Alignment is derived, not counted**: `lmh_r_fmt_of(i)` derives the kind from the field NAME
  (`lmh_r_hex_fields[] = {"oxor","xorb"}`, `lmh_r_real_fields[] = {"osum"}`, else `dec`), and the new
  self-test check requires the hand-written `lmh_r_fmt[]` to agree field-by-field and the hex family to
  be populated (a digest rule over zero bit-image columns is a counter that reads zero and cannot fail).
* **Consumer** (`gate.py`): `contract()` **scrapes `lmh_r_fmt[]` out of the header — it is scraped, not
  mirrored; a mirrored table drifts.** `read_report()` then refuses any token not matching
  `CANON[kind]` (`gate.py §CANON`), printing `REFUSE <file>: case <n> field '<name>' is '<token>',
  which is not the canonical <kind> form ...` in full before raising, so the offender is named rather
  than truncated, and converts with the declared base (`int(tok, 16)` for hex, `int(tok, 10)` for dec).
  Count discipline is unchanged and was re-witnessed: `parsed == printed` is required
  (`parsed=192 printed=192 fields=33 forms=hex:2,dec:30,real:1`), a missing row is a `Fail`
  (`printed 191 R rows, the plan has 192 cases`), and an R-body-less report is a `Fail`
  (`has a name line and NO R rows: an empty parse cannot fail`).

## The controls, witnessed (each a separate `gate.py` process, into a file, never through a pipe)

Negative — the same 32-bit value re-spelled in the other radix, by `--control`:

```
$ python3 gate.py --control digest-radix --work work        # EXIT=1
CONTROL digest-radix file=host_report.tsv case=0 field=oxor value=0x0C0102A4 \
    re-spelled in the other radix as '201392804': the parser MUST refuse it by name, ...
REFUSE work/host_report.tsv: case 0 field 'oxor' is '201392804', which is not the canonical hex form \
    the writer declares in lmh_r_fmt[] (a digest spelled in the wrong radix is a finding, ...)
CHECK name=the run reached its adjudication ok=0 n=1 refusal=work/host_report.tsv: case 0 field ...
RESULT FAIL lm_head (refused) checks=8 findings=1

$ python3 gate.py --control count-radix --work work         # EXIT=1   (the mirror case)
CONTROL count-radix file=host_report.tsv case=0 field=loads value=0x00000004 \
    re-spelled in the other radix as '0x00000004': ...
REFUSE work/host_report.tsv: case 0 field 'loads' is '0x00000004', which is not the canonical dec form ...
RESULT FAIL lm_head (refused) checks=8 findings=1
```

This is the case the brief named: the gate **reports** the mismatch instead of comparing 201392804
against `0x0C0102A4` and calling them different values.

Positive — a parser that refused every token would also pass the two above, so the accepting arm is
witnessed too, and witnessed as a live comparison:

```
$ python3 gate.py --check-only --work work                  # EXIT=1, 4 pre-existing findings
REPORT file=work/host_report.tsv parsed=192 printed=192 fields=33 forms=hex:2,dec:30,real:1
CHECK name=host case=0 honest-bf16-lmhead-dense-batched oxor==recount ok=1 n=16 driver=0x0C0102A4 recount=0x0C0102A4
   (192/192 oxor==recount ok=1, 192/192 read-set == recount ok=1, 0 ok=0)

$ python3 gate.py --control tampered-out --work work        # EXIT=1
CONTROL tampered-out flipped the low bit of word 0 (case 0 honest-bf16-lmhead-dense-batched) ...
CHECK name=host case=0 honest-bf16-lmhead-dense-batched oxor==recount ok=0 n=16 driver=0x0C0102A4 recount=0x0C0102A5
RESULT FAIL lm_head host-only checks=1510 findings=10
```

Canonical spelling accepted and equal to the independent recount; one flipped bit in the output file
moves the recount and turns that same check red. The digest comparison is live, not decorative.

## Totals after the change

* `./host_probe --selftest --work work` → `RESULT PASS host_probe selftest`,
  **`checks=66 failures=0`**, rc=0. `./lm_head --selftest --work work` → identical roster,
  `checks=66 failures=0`, rc=0. HEAD was 64; the two added witnesses are
  `report-form-table-agrees-with-the-field-names` and
  `digest-radix-is-in-the-text-not-in-the-parser` (the latter asserts the decimal spelling of
  `0xDEADBEEF` is refused by BOTH kinds, that the two spellings differ as strings, and that reading it
  as hex does *not* give the digest — the mis-parse the parser must never be able to perform silently).
  `xor-digest-blind-to-uniform-rescale` remains green.
* `python3 gate.py --help` → rc=0. `python3 gate.py --check-only --work work` → rc=1,
  `CHECKS ran=1510 failed=4` (the four pre-existing findings, kept red — see below).
* `python3 gate.py --selftest --work work` → rc=1, `CHECKS ran=29 failed=2`. Both new controls pass
  their self-test entries (`rc=1`, marker present, named refusal). The two findings are pre-existing
  and unrelated to digests: `control q6k-geometry` exits 1 but never prints its `CONTROLARM host_probe`
  marker, and `control q6k-source` prints its two `REFUSE` lines but **exits 0**, so a control that
  refuses without failing its own run is accepted by the plan's expectation. Left red deliberately.

## Also fixed while here (one line, pre-existing at HEAD)

`--control tampered-out` died with `KeyError: 'derived'`: it read `c["derived"]` before
`adjudicate()` fills it (HEAD `gate.py:1467` vs `adjudicate()` at `:1482`), so it produced a traceback,
never printed its own `CONTROL tampered-out` line, and the `--selftest` entry for it could not have
passed. It now asks the same derivation `adjudicate()` will use (`derive_class(c, d)[0] == "BAND"`) and
produces the verdict above.

## Kept red, deliberately

The four host findings from `--check-only` are untouched and are the next lane's: `host injection
denominator matches the plan` (plan=77 vs judged=79 — the two `gather-row-off-1` cases whose successor
rows escape the table are appended to the firing roster twice, once by the cross-count witness at
gate.py §"only the successor rows that escape the table count as cross" and once by the class branch);
`every injection that must fire fired` (the four untied `*-cross-table-read` arms) and
`every derived-immune case really is immune` (`inj-bf16-tied-a-cross-table-read`) are both caused by one
roster line that files the arm under `ok = counters UNCHANGED` for both buckets, while the falsifiable
check three lines above it requires the untied arms' counters to MOVE by `n_out*nlanes` (they do:
`cross=4/4 peer_hits=4/4 ok=1`) and the tied arm's `cross` to stand still while `peer_hits` grows
(`cross=0/0 peer_hits=8/8 ok=1`) — the roster predicate is inverted for the untied arms; and
`no case reads outside its own table unless its mode says it must` fires on the honest `bf16-tied-a`
cases because the tied control's peer range IS its own range, so `peer_hits == loads` there by geometry
while the exemption list names only the three foreign-reading modes.

## NOT ESTABLISHED

* **No device run has ever happened in this lane.** No kernel number is attested; the device arm, the
  three-arm agreement and every launch count are unexecuted code paths.
* **Disjointness is not established.** The untied real tables are disjoint and come from different
  shards (green: `DISJOINT tag=host real_tables=5 overlaps=0 same_file=0`, and
  `lm_head and the embeddings come from DIFFERENT shard files`), and the tied control's aliasing is
  real, but the policy check that would certify "nobody read outside their table" is the one now red —
  so `peer_hits == 0` for untied cases and `> 0` for the tied control are each witnessed only by the
  recount agreement (`read-set == recount`, 192/192), not by a passing policy check.
* **Amortisation is not established as a claim about the device.** The host arm prints
  `batching amortises the weight read by exactly bcount` (10 groups, ratio == bcount exactly) and
  `addressed bytes are independent of bcount for the same row set` (5 shapes flat), and
  `batch-reloops-immune-at-bcount-1` is in the driver roster — all host-twin, all bytes-only, no timing.
* **Band derivation is not re-printed here.** `oracle_arm.py` reports `RESULT PASS oracle arm
  filled=1884/27488 words` and the gate requires `band_nonpositive=0`, but this pass did not re-derive
  or attribute `gamma_K = K*u/(1-K*u)`, `u = 2^-24`, `sum|h_k||w_k|` against the real
  `layer39.combined_out` rows. The bands in force are therefore not attributable to a floor probe and
  geometry in this report, and no verdict that rests on a `BAND` class should be read as certified.
* `results.md`, the `gate` wrapper and the GPU-lock acquisition do not exist yet; nothing here takes the
  lock and nothing here would be safe to read as a measurement.
