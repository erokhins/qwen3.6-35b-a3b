# `speccheck/gdn_ref.py` — the CPU GDN oracle: what is bit-exact, what is banded and from which probe, and what is UNVERIFIED by name

**Provenance.** Written 2026-10-26 by the W21 lane, which picked the lane up after two workers died on
run deadlines assembling its evidence. The MoE oracle has had `speccheck/results.md` since `4901b84`;
this is the GDN counterpart, and it is the file that says what that lane may be *read* as having
checked. Every figure below is read verbatim from a capture under `speccheck/evidence/` named beside
it, or is a product/sum of quoted figures with its factors given. **No device work and no clock are
involved anywhere**: every instrument cited is torch-CPU and prints counts, ratios and norms only, and
each capture ends with its own `no timing taken` (`AGENTS.md` §GPU execution).

**Who this is for.** A reader who has to decide one question: *can I use this as the bf16 authority for
a GDN kernel — the scan, the short conv, the gated norm, `out_proj` — or for a quantization gate over
those tensors?* §§What is bit-exact and §§What is banded answer the permission, §§What is UNVERIFIED
bounds it, and §§Findings and contradictions names the things that would make a careful reader stop.

**Citation form.** `path §label`, per `pitfalls/60-docs-and-citations.md`. Line-number cites are
avoided on purpose: a line cite is a claim about a file's line count, and this file has grown from
3,055 lines to 4,167 during this pass.

---

## §In one paragraph

`speccheck/gdn_ref.py` is a torch-CPU implementation of the token mixer of the 30 `linear_attention`
layers (40 = 30 + 10 `full_attention`, `README.md`'s layer table; Gated DeltaNet + short conv + gated
norm + `out_proj`), written from the committed config, the
shard index and `specification/04-gdn-linear-attention.md` / `specification/04b-gdn-chunked-form.md`,
with no HF import at module scope. Against the installed `transformers` **5.19.0**
`Qwen3_5MoeGatedDeltaNet` it is **bit-identical on all 150 comparison rows** — 76 stage rows
(19 stages × 2 dispatch shapes × 2 pinned blocks) + 18 per-step decode rows + 56 seam rows — with its
14 positive controls all REPORTED as differences and none blind, none leaking. Its injection matrix is
three arms over 300 derived cases per arm, with the blinded arm missing exactly the derived
**2084** allowed stages. Its floor probe (`--floor-probe`) measures 102 arm-pass pairs × 7 stages =
**714 records** over 26 pass shapes and 2 blocks, having sha256-verified **all 178 pinned dump files
(162,519,680 B)** first, and derives a band from those maxima by the triangle inequality — the band
licenses **1,136** cross-form rows, all of which were adjudicated inside it, and refuses the other 82
because a bf16 row that measured `ndiff 0` measured nothing. What it does *not* establish is listed by
name, not summarised: 12 of the 19 stages have no floor at all, and no verdict outside the probe
credits a band.

---

## §The pins are on-disk artifacts, and that is the first thing a fresh session needs to know

**The bytes are not in git and were never going to be.** `data/` is ignored at `.gitignore:20`
(the "Reference dumps / fixtures / bench output (regenerable)" block), and `git ls-files data/ | wc -l`
is **0**: no file under `data/` is tracked, so a fresh clone has this document, the oracle, the dumper
and the manifest, and no tensors.

| fact | value | where to check it |
|---|---|---|
| dump directory | `data/dumps/2026-10-26-gdn/` | `tools/dump_reference_gdn.py` §`DEFAULT_OUT` |
| manifest (tracked) | `checkpoints/dump-manifest-gdn-2026-10-26.json` | `git ls-files checkpoints/` |
| records | **178** = 2 blocks × (9 weights + 4 passes × (1 input + 19 stages)) | manifest `counts` |
| declared bytes | **162,519,680 B** | manifest `bytes.declared_records` |
| same, on disk | `du -sb` → 162,519,680 B; `du -sh` → **156M** (block-rounded, hence "156 MB") | the directory |
| integrity | **per-file sha256** in the manifest, plus size == elements × itemsize | `read_dump_tensor`, `verify()` |

**Integrity therefore lives in the manifest, not in git.** A tracked file is protected by the object
hash; an untracked 156 MB dump is protected by the digest the manifest carries for it and by the
verifier that re-hashes every one. That is why `tools/dump_reference_gdn.py --verify` refuses to run at
all without the manifest (`PINNED MODE REFUSED`): "no files were bad" over a directory nobody can name
is vacuous (`pitfalls/40` §a check that produces no output cannot fail).

**Recovery, for a fresh session — two commands, and the second is the proof:**

```
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/dump_reference_gdn.py \
    --blocks layer0,layer1 --tokens 32 --seed 20261026 --threads 8 \
    --out data/dumps/2026-10-26-gdn --manifest checkpoints/dump-manifest-gdn-2026-10-26.json
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/dump_reference_gdn.py \
    --verify --out data/dumps/2026-10-26-gdn \
    --manifest checkpoints/dump-manifest-gdn-2026-10-26.json
```

The write path is not a blind re-emit: it runs the installed reference over the same weights and the
same input and **refuses to write a byte the reference does not reproduce bit for bit**, all 19 stages
in prefill and at every cached decode step, and it cross-checks the prefill input against the MoE
dump's pinned input so "the two dumps describe one fixture draw" stays a checked claim. Capture:
`speccheck/evidence/gdn-dump-write-2026-10-26.txt`, `gdn-dump-repro-2026-10-26.txt`,
`gdn-dump-verify-2026-10-26.txt` (`VERIFY PASS`, 178 records / 178 verified / 0 bad, byte
reconciliation three ways at 162,519,680 B), and the refusals in `gdn-dump-verifier-controls-2026-10-26.txt`
and `gdn-dump-pinned-refusal-2026-10-26.txt`.

**The live paths are guarded, and that is load-bearing rather than tidy.** `refuse_live_paths()` refuses
any `--out` or `--manifest` that is, contains or is contained by `data/dumps/2026-10-26/` or
`checkpoints/dump-manifest-2026-10-26.json` — the MoE lanes' infrastructure — and it tests resolved path
*components*, not a string prefix, because `data/dumps/2026-10-26-gdn` is a sibling of the MoE dump and
shares its prefix. `git status` on both live paths is empty and must stay empty.

---

## §What is bit-exact, and against what

Against the installed reference (`--compare-hf`, which `--selftest` also runs), capture
`speccheck/evidence/gdn-compare-hf-2026-10-26.txt` and the rc=0 run this pass:

```
# rows compared 150 = 76 stage rows + 18 per-step rows + 56 seam rows; bit-identical 150, differing 0
# controls 14 run, 0 blind (must be 0), 0 leaking (must be 0)
```

That is every stage in `STAGES` × {chunked prefill, cached recurrent decode} × {layer0, layer1}, plus
`y`/`state`/`conv_window` at each of the 3 cached decode steps, plus the 7 seam rows per pass that
license the `derived-from-captured` and `reference-side-call` provenances. Each row prints the
**provenance of the HF-side tensor**, because "we compared against HF" is three claims, not one.

Against the pinned bytes, from the probe run (`speccheck/evidence/gdn-floor-probe-2026-10-26.txt`):

```
  recon    printed    170 parsed    170 derived    170  ok
  files    hashed 178 derived 178 (162,519,680 B read, every one sha256-checked before it was measured)
```

— 152 pinned stage tensors (19 × 4 passes × 2 blocks) and 9 pinned weight tensors per block, each
read back off disk through `tools/dump_reference_gdn.py:read_dump_tensor` and compared with the
oracle's own clean pass by `_bit_diff`. The reader's own contract (size-vs-shape, bf16 as a two-byte
bit image viewed `uint16` then `bitcast`, digest match) is exercised separately, and witnessed at
**152 pinned stage tensors read back, 0 not bit-identical** in
`speccheck/evidence/gdn-dump-reader-check-2026-10-26.txt` — a different instrument from `--verify`,
which proves the bytes *hash* to the manifest while this proves they *parse* to the tensors the
manifest names.

**What bit-exactness here does and does not mean.** Agreement with the reference on a same-form arm is
a *transcription* check — a mirror that shares a formula agrees with it because they share a formula
(`pitfalls/30` §a CPU mirror cannot see it). The independence evidence is the cross-form agreement
(§§What is banded) and the three-arm matrix; the bit-identity says the spec was read correctly and the
bytes on disk are the bytes the oracle produces. Both directions are witnessed:
`gdn-compare-hf-sabotage-read_pre_update_state-2026-10-26.txt` and
`gdn-compare-hf-sabotage-decay_after_write-2026-10-26.txt` show the comparison differing when the
oracle is injected, which is what makes the clean PASS a claim rather than a habit.

---

## §What is banded, and from which probe

**From `speccheck/gdn_ref.py --floor-probe`**, which implements
`specification/10-reference-dumps-and-gates.md §10.3` step 1 literally: the arms are the five legal F32
forms (`chunked`, `substitution` and `chunked32` for the A-16 branch/width pairs; `recurrent` and
`reassoc` — the recurrent form with its sums split in half, which *is* the re-association), plus the
pinned dump bytes themselves as the arm `pinned`, i.e. the reference's own F32 math over the pinned
rows. The anchor is `Form.named("exact")` in f64 with its scan **inputs** also f64, and its twin
`exact_matrix` is run against it as the anchor's credibility check. Band = per-stage max over the
folded records × `BAND_FACTORS` = `{one_minus_cos: 4, mean_rel: 2, max_abs: 2, per_head_max: 2}`,
where the 4 is `(1+1)²` from the angle metric and the 2s are the pointwise triangle inequality. **No
number in the file is typed and none is rounded up**, and `load_band()` re-derives `probe × factor` on
every read and refuses a file where the two disagree.

Capture `speccheck/evidence/gdn-floor-probe-2026-10-26.txt` (rc=0, `FLOOR PROBE PASS`), band file
`speccheck/evidence/gdn-band-2026-10-26.json`. **The band is reproducible, not a snapshot:** the probe
was run three times from the pinned bytes, with nothing cached between runs, and the file it wrote was
**byte-identical all three times** (`load_band()`'s re-read digest `27a68c7e82c9aa05` in each, and `cmp`
clean against the first run's bytes even for the run made after `--floor-probe`'s write path was edited)
— the
numbers below are a function of the pinned bytes and the arm list, not of run order or of what the
builder had already computed. A **fourth** run, made in the `--cross-form` pass after that section's own
band consumer had been added, left the committed file byte-identical again (`git status --porcelain
speccheck/evidence/gdn-band-2026-10-26.json` empty afterwards). **The digest is stated with its
canonicalisation, not as a bare hex string**: it is
`hashlib.sha256(json.dumps(band, sort_keys=True).encode()).hexdigest()` over the `band` member of the
JSON document
(`27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`), and `--cross-form` prints it
beside the command that regenerates it from the committed file.

| stage | probe maxabs | **band maxabs** | probe 1−cos | **band 1−cos** | probe meanrel | **band meanrel** | licensed rows |
|---|---|---|---|---|---|---|---|
| `q_hat` | 1.450e-08 | **2.900e-08** | 3.886e-14 | **1.554e-13** | 5.786e-08 | **1.157e-07** | 174 |
| `k_hat` | 1.301e-07 | **2.603e-07** | 3.875e-14 | **1.550e-13** | 5.652e-08 | **1.130e-07** | 174 |
| `scan_out` | 1.741e-08 | **3.483e-08** | 5.729e-14 | **2.292e-13** | 3.274e-07 | **6.548e-07** | 174 |
| `core_attn` | 7.629e-06 | **1.526e-05** | 7.689e-10 | **3.076e-09** | 9.924e-07 | **1.985e-06** | 152 |
| `state` | 5.766e-07 | **1.153e-06** | 5.396e-14 | **2.158e-13** | 2.195e-07 | **4.389e-07** | 174 |
| `normed` | 1.562e-02 | **3.125e-02** | 6.655e-09 | **2.662e-08** | 1.970e-06 | **3.941e-06** | 149 |
| `y` | 3.906e-03 | **7.812e-03** | 5.642e-08 | **2.257e-07** | 4.156e-05 | **8.311e-05** | 139 |

Two things a reader should notice rather than skim. `normed`'s band is **four orders of magnitude
wider than `state`'s** — that is the gated norm amplifying the scan's last bits through a reciprocal
square root and a SiLU gate, and it is printed rather than smoothed, because a band that disagrees
with the probe is a finding, not a thing to widen. And `core_attn`/`normed`/`y` have *fewer* licensed
rows (152/149/139 against 174) because a bf16 row that measured `ndiff 0` is excluded: the bf16 storage
quantum hides anything smaller, so such a row may not license a verdict
(`pitfalls/30` §byte-identity claims state the width they cover).

**This is the band `--cross-form` consumes**, and consumes honestly: §§The cross-form verdict is the
section that turns it into a verdict — the licence re-derived from the bytes rather than trusted, the
`normed` margin printed as three metrics plus its share of each licence, and a measurement of how much
of a real §04.7 gated-norm defect that margin swallows.

**The anchor's credibility is under test, not quoted.** `specification/04-gdn-linear-attention.md §04.6`
says the outer-product spelling is equivalent to the rank-1 one; the probe requires the f64 twin to sit
no further from the anchor than the *worst F32 arm* does, per stage and per aggregate:

```
  stage       twin maxabs   arm maxabs  twin meanrel   arm meanrel    ok  twin rows differing
  q_hat         0.000e+00    1.450e-08     0.000e+00     5.786e-08   YES                    0
  scan_out      1.145e-16    1.741e-08     2.506e-15     3.274e-07   YES                   26
  state         6.217e-15    5.766e-07     1.650e-15     2.195e-07   YES                   26
  y             0.000e+00    3.906e-03     0.000e+00     4.156e-05   YES                    0
```

`scan_out`/`state` have 26 differing twin rows each, which is also the non-vacuity witness: two f64
implementations that were bit-identical everywhere would prove nothing, and that is a FAIL, not a bonus.

**Population, and all three counts agreeing** (printed lines, parsed records, denominator derived from
the manifest's record set and the fixture list before anything ran):

```
  recon    printed    170 parsed    170 derived    170  ok
  probe    printed    714 parsed    714 derived    714  ok
  anchor   printed    182 parsed    182 derived    182  ok
  xform    printed   1136 parsed   1136 derived   1136  ok
  cred     printed      7 parsed      7 derived      7  ok
  band     printed      7 parsed      7 derived      7  ok
  sab      printed      4 parsed      4 derived      4  ok
  files    hashed 178 derived 178 (162,519,680 B read …)
  arm-pass pairs 102 over 26 passes on 2 blocks; stages probed 7 of 19; licensed cross-form rows
  adjudicated 1136 of derived 1136
```

`# population: files 178 x stages 7 x blocks 2 x passes 26 (pinned 4 + fixture 9 per block) … 102
arm-pass pairs, 714 probe records, 1218 cross-form rows, 182 anchor rows` — 1,218 candidate pairs
minus the 82 the bf16 rule unlicensed = the 1,136 adjudicated. The pass grid is not typed: the pinned
modes come from the manifest's own record set and are asserted against the geometry
(`prefill` + `decode0..k_conv-2`), and the fresh passes are the same `default_fixtures(g)` object the
injection matrix runs.

**Why only 7 of 19 stages get a floor.** The probed set is `SCAN_WIDTH_STAGES` =
`downstream_closure(SCAN_LOCAL_STAGES)` = `q_hat, k_hat, scan_out, core_attn, state, normed, y` — the
four the scan produces plus the three that are functions of them. A stage upstream of the scan-dtype
decision is computed by code the F32 arms and the f64 anchor **share**, so its distance to the anchor is
0 by construction; printing a floor for it would be printing a check that cannot fail. Those 12 are in
§§What is UNVERIFIED by name, with the condition that would make a floor measurable there.

**The band is derived from its input, and witnessed to be.** Two sabotages per block perturb a pinned
dump tensor in the field the band licenses (in memory, after `read_dump_tensor` has hashed it — nothing
writes to the dump):

```
  layer0  ulp     layer0/pinned/decode2   scan_out   4.657e-10   1/4096   2.075e-09  ACCEPT   ACCEPT
  layer0  exceed  layer0/pinned/decode2   scan_out   1.398e-07   1/4096   1.405e-07  REJECT   REJECT
  layer1  ulp     layer1/pinned/decode0   scan_out   4.657e-10   1/4096   1.558e-09  ACCEPT   ACCEPT
  layer1  exceed  layer1/pinned/decode0   scan_out   1.398e-07   1/4096   1.388e-07  REJECT   REJECT
```

1 ULP of the largest element (derived with `torch.nextafter`, not with a typed quantum) must still be
**accepted** — with `ndiff 1/4096` proving the comparator saw the perturbation, so this is not an
acceptance by blindness — and a value sized from the band itself must be **rejected**. One direction
alone proves nothing: an adjudicator that rejects everything passes the second and fails the first.
Beside each, the run prints what the band *would* have become had the poisoned row been folded in
(**8.1× and 8.0× looser** on `scan_out`), which is the argument for why the sha256 runs before any
measurement, and re-hashes the band to prove no sabotage relicensed anything.

The refusals the band rests on are all fired in
`speccheck/evidence/gdn-band-loadband-controls-2026-10-26.txt` — 17 controls, including a band number
widened 10× "for safety" (refused), probe maxima inflated without the band following (refused), a
licensed row for a pass that never ran (refused), every licence emptied (refused), and the bf16 rule in
*both* directions: a bf16 stage measuring all-zero floors loses its licence, and the same stage with
one real 3e-06 measurement earns it back.

---

## §The cross-form verdict — the band credited outside the probe

```
command: /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python speccheck/gdn_ref.py --cross-form
rc=0 — CROSS-FORM PASS
```

Capture `speccheck/evidence/gdn-cross-form-2026-10-26.txt`; the controls that fire its guards are in
`speccheck/evidence/gdn-cross-form-controls-2026-10-26.txt`. Before this section existed, the band was
credited inside one process only: `--floor-probe` derived it, adjudicated the licensed rows, and exited, so
outside the probe no PASS credited a band — `HF_UNVERIFIED` row 5 said so in `--floor-probe`'s own words.
This section is the consumer, and a consumer that re-derives everything it is asked to believe.

**1,136 rows, none of them read from a file.** Each row is a `(pass, arm-pair, stage)` triple on a selected
block, drawn from **1,218** candidate pairs (2 blocks × 21 unordered arm pairs per pass × 29 pass-stage
combinations), and its `max_abs`, `1 − cos`, `mean_rel` and per-head maxima are recomputed by `compare()`
from tensors read fresh off disk by `read_dump_tensor`, which sha256-verifies each of the **64** dump files
this section touches before anything is measured off it. The band file's role is narrower still: it is read
by `load_band(BAND_PATH)` and nowhere else, it supplies only `limits[stage]`, `licensed_rows[pass|a|b]` and
the `band_derivation` string, and no number in this section recomputes or restates a limit —
`--floor-probe`'s `probe × BAND_FACTORS` arithmetic is the only thing allowed to say what a limit is, and the
three refusals that enforce it (file missing, foreign derivation, a `probe × factor` mismatch) are each
witnessed firing in the controls capture. A scratch copy parked on the credited path by `--band-out` is
refused too, so no verdict is credited from a file someone hand-made.

**The counts agree three ways, for every table**, each denominator declared from geometry × licence before
any measurement and from the loaded band before any judgement:

```
  xf       printed   1136 parsed   1136 declared   1136  ok      (ACCEPT 1136, REJECT 0)
  audit    printed   1218 parsed   1218 declared   1218  ok      (licence/measurement disagreements 0)
  normed   printed    149 parsed    149 declared    149  ok      (ACCEPT 149, REJECT 0)
  margin   printed     72 parsed     72 declared     72  ok
  sab      printed      4 parsed      4 declared      4  ok
  bandctl  printed      4 parsed      4 declared      4  ok
  files hashed 64 declared 64
```

**The licence is audited, not believed — and in both directions.** `band_licenses()` answers "licensed by
probe records for both arms at this pass", which is a claim about measurements the *probe* made; a consumer
cannot re-run the probe and still be told the truth by a file. So the audit re-derives the probe's own
folding rule from the bytes: a bf16 arm whose record came back `ndiff 0` against the f64 anchor saw nothing,
and a pair where both arms saw nothing compared two blind men. All 1,218 candidates are checked that way and
disagreement is a failure either way round — *"licensed but measured nothing"* and *"licence withheld from a
row that measures a difference"*. Both strings are witnessed firing against the real section (controls 6 and
7), where `load_band()` is the thing that fails to notice — its coverage check is real, so a phantom licence
for a withheld `ndiff 0` pair passes it — and only the audit catches it. A licence for a case this section
could not run is a failure too (*"a licence whose case never ran lists as adjudicated evidence"*), which is
what makes 1,136 mean 1,136 measurements and not 1,136 lines of a file.

**The `normed` row, scrutinised.** `--floor-probe`'s own words are that `normed`'s band is four orders of
magnitude wider than `state`'s (MUFU-class disagreement through rsqrt, sigmoid and SiLU), and
`normed.max_abs = 3.125e-02` sits beside `state`'s `1.153e-06`. The section prints all three metrics for
every one of its 149 rows plus each metric's share of its own licence, because for this stage max_abs alone
cannot tell you anything: the licensed limit is **0.25–8.00 bf16 ULPs of a row's own largest element**, so
one ULP at the top of a tensor is sometimes inside the band and sometimes outside it. The worst row used
2.581e-01 of the 1−cos licence and 5.010e-01 of the mean-rel one; three rows sit above their max_abs licence
and pass on the relative metrics, at 1.25× and 2 × 1.38× — all prefill rows with `1 − cos ≤ 1.2e-09` and
`mean_rel ≤ 4e-08`, which is the `pitfalls/30` §a max-ULP band cannot see a whole-tensor 1-ULP error
situation — max-abs alone is the wrong ruler for a chain, and the aggregates are what bind — rather than an
accident of the band. `normed` is also one of the two stages that can move `y`, so every `normed` verdict is re-checked at
`y`'s own licence: 149 of 149, 0 failures.

**Can a defect hide inside that margin? Yes — measured, not argued.** The margin ladder takes the four §04.7
injections whose entire contract is `normed` + `y` (derived from `AFFECTED`, so the list is that table's and
not this section's opinion) and injects the oracle side on a real adjudicated row, at full strength and then
at 1/2**k. All four REJECT at full strength on both blocks, as they must. Then the ladder halves the defect
until the band stops noticing:

```
  ladder rows 72; ACCEPTed 15; of those, 3 left the arm's stored bytes changed — those are the defects the
  margin swallows
  the LEAST-scaled defect the band licenses: `norm_weight_f32` on layer0/pinned/prefill|reassoc|substitution
  at 1/2**2 — maxabs 3.906e-03, 1-cos 4.051e-09, meanrel 1.988e-06, with 190 of 131072 elements moved and
  ndiff 249/131072, against licensed 3.125e-02 / 2.662e-08 / 3.941e-06
  deepest accepted scaling: `norm_variance_flat` on layer0/pinned/prefill|reassoc|substitution at 1/2**8
  (still 144 elements moved)
```

So a gated-norm width defect at **one quarter strength** — the norm weight read at f32 instead of bf16 — is
accepted by this band while still leaving 190 stored elements changed. The consequence for the engine lane is
the reason the number is printed: **the cross-form band is a bound on legitimate cross-form spread, not a
defect detector.** A kernel whose only pass condition is these rows can ship a real gated-norm width bug at
25 % strength and print ACCEPT. The detector for that class is §04.7's injection matrix (`--selftest`; the
ladder above is its scaled twin), and a `normed`-width gate must be paired with a must-catch set the way
`--selftest` pairs them. The band was not tightened to make this answer comfortable, and it cannot be
tightened from here: a smaller `normed` limit would be a number nobody measured (`AGENTS.md` §Standing
rules).

**Both directions of the accept/reject sabotage, on the reference's own bytes.** The probe's witnesses
perturb an oracle-side arm; a consumer needs the same proof on the side it reads from disk. The perturbation
happens in memory, after `read_dump_tensor` has hashed the file — nothing here writes to the dump:

```
  layer0  ulp     layer0/pinned/decode2|pinned|reassoc  scan_out  4.657e-10  1/4096  ACCEPT  ACCEPT
  layer0  exceed  layer0/pinned/decode2|pinned|reassoc  scan_out  1.398e-07  1/4096  REJECT  REJECT
  layer1  ulp     layer1/pinned/decode0|pinned|reassoc  scan_out  4.657e-10  1/4096  ACCEPT  ACCEPT
  layer1  exceed  layer1/pinned/decode0|pinned|reassoc  scan_out  1.398e-07  1/4096  REJECT  REJECT
```

`1/4096` on the accept side is the proof the comparator was awake: exactly one element of the pinned
reference moved (`ndiff 1/4096`, one element per witness) and the band still accepted the row. The 1-ULP step
is derived with `torch.nextafter` at the row's largest element, not with a typed quantum, and the exceed
delta is `4 × band + 1 ULP` so the reject cannot be a rounding accident; a candidate row whose own 1-ULP step
would not fit its licence is excluded from the candidate set, and an empty candidate set is a failure rather
than a skip. A `--blocks layer0` run therefore carries layer0's two witnesses and `--blocks layer1` carries
layer1's — a per-block guarantee, not one witness standing in for two files.

**The verdict tracks the band.** Each key in `BAND_FACTORS` is zeroed in turn on a copy of the loaded band,
and the rows that flip must be *exactly* the rows whose measurement on that key is non-zero — and never zero
rows, which is `pitfalls/00-measurement.md` §A counter that reads exactly 0 is vacuous: 1,065
rows flip on `one_minus_cos`, 807 on each of `max_abs`, `mean_rel` and `per_head_max`. With `band_check_row`
stubbed to accept everything the reject direction fails; stubbed to refuse everything the accept direction
fails (controls 10 and 11), so neither witness is a constant.

**NOT ESTABLISHED by this section** — named, not summarised, and the same list `--cross-form` prints:

- **12 of the 19 stages have no floor**, so no cross-form row can be judged for the input projections,
  `conv_weight`, `dt_bias`, `norm_weight`, `A_cumsum`, `l2q`, `l2k`, `beta`, `gamma`, `q`, `k`, `v`. The band
  covers the seven stages `--floor-probe` measured; a wider claim would be about a band that does not exist.
- **82 of the 1,218 candidate pairs are unadjudicated**, withheld by the licence because an arm measured
  `ndiff 0` on bf16. Agreements the file licenses are checked; agreements it does not license are absent, and
  the 1,136 must not be read as the whole cross-form surface.
- **329 of the 1,136 adjudicated rows measured arm-to-arm bit-identity** (`ndiff 0`). Each is a true
  statement and a stronger one than the band, but it is not an independent test of the band, so the number of
  rows that actually stressed a limit is 807, not 1,136.
- **The reference is not judged.** Both arms are compared to one shared f64 anchor, so a bug in the oracle's
  anchor moves both arms together and is invisible to every row here. The independence evidence is §04.7's
  injections and §§What is bit-exact, not this table.
- **`conv_window` has no banded row.** It is upstream of the scan-dtype decision, so the probe refuses to
  floor it and `band_check_row` refuses to invent a limit; the two conv spellings are still checked
  bit-for-bit against the pinned bytes by `--compare-hf`. Closing this needs a probe measurement (a second
  legal conv path to floor against), not a licensor written by the consumer.

---

## §The `scan_in_bf16` arm — how it is disposed of, and the measurement that disposes of it

The arm stores the scan in bfloat16 where the reference stores it in float32
(`specification/04-gdn-linear-attention.md §04.4`'s scan-dtype row: `mamba_ssm_dtype: float32`). It
moves **seven** stages while declaring only that it touched the scan, and the lane owed an answer to
the obvious question: **is the declaration wrong, or does a bf16 scan really propagate past its own
boundary?** Both answers were on the table and one of them would have meant the injection matrix's
licence is wrong, so this was decided by measurement, not by editing the contract
(`speccheck/evidence/gdn-bf16-arm-disposition-2026-10-26.txt`, from the rc=0 `--selftest`):

```
# bf16-arm rows 76 = 19 stages x 2 modes x 2 blocks, reported separately and NOT adjudicated for
  agreement: bit-identical 48, differing 28 (of which 16 were REFUSED on storage width and measured
  after an explicit widening), 0 inert (must be 0), 0 outside contract (must be 0)
```

Splitting the blast radius into *storage changed* and *value moved* — which is what the run now
asserts, per block, per mode:

1. **Storage width changed on exactly the scan-local four**: `q_hat, k_hat, scan_out, state`. The run
   asserts that set equals `SCAN_LOCAL_STAGES`, so if bf16 ever reached a tensor §04.4 does not cover,
   the print says "the contract IS wrong, not merely wide" — which is the finding that would reopen
   this section.
2. **Value-only movers, dtype unchanged: exactly `core_attn, normed, y`** —
   `downstream_closure(SCAN_LOCAL_STAGES)`, derived from `STAGE_CONSUMERS` rather than typed.
3. **The other 12 stages bit-identical, and all 12 upstream of the scan** — separately asserted,
   because a pre-scan move would be bf16 propagating *backwards*, a different and worse finding.
4. **The attribution that closes it**: feed the *reference's own* `core_attn` bytes through the
   oracle's `gated_norm` + `out_proj` and they reproduce the reference's `normed` and `y`
   **bit for bit** — 8 printed rows, `ndiff 0` against their own element counts (prefill
   `0/393216` and `0/196608`, decode `0/4096` and `0/2048`, per block). The other half of the
   attribution — the *arm's* own `core_attn` through the clean downstream must return the arm's own
   `normed` and `y` — prints nothing on success and fails the run otherwise; it is counted, and
   `bf16_attrib` is asserted equal to its derived 16 (`4 × 2 modes × 2 blocks`), so the silent half
   cannot quietly stop running (`pitfalls/50` §a check that produces no output cannot fail). The
   downstream therefore contributes nothing to the disagreement; the four non-scan movers are pure
   dataflow from the four stored-narrow stages.

**Disposition: the arm is kept, not deleted, not skipped, not narrowed.** The contract was not wrong,
it was *untyped*: `AFFECTED["scan_in_bf16"]` is now derived from `SCAN_WIDTH_STAGES`, and the derived
set is the same seven stages the typed one listed — which is why the three-arm denominators below did
not move. The arm remains excluded from `MUST_CATCH` (forcing it would demand a bf16→f32 tolerance no
band in this repo was derived from a probe of) and remains reported separately from the bit-identity
rows, so a bf16 difference can never be booked as a bit-exact PASS.

**`torch`'s CPU backend has no BFloat16 kernel for `triangular_solve`.** This is not a quote from a
forum; it is the measured string from `bf16_solve_supported()`, re-printed with the case list and
required by `--selftest` to match the `NOT_ESTABLISHED` entry verbatim:

> `NotImplementedError: "triangular_solve_cpu" not implemented for 'BFloat16'`

So the arm does the solve the way the reference does it — F32, cast back to the scan's storage width —
per `specification/04-gdn-linear-attention.md §04.4` and `specification/04b-gdn-chunked-form.md §04b.4`
(`torch_chunk_gated_delta_rule` casts q/k/v/beta/decay to float32 at the scan entry and solves in
float32). **The width of the scan *storage* is the arm's measured effect; the numerics of a bf16 *solve*
remain `NOT ESTABLISHED`**, with that exception string as its measured reason and
`would_close: a backend with a bf16 solve_triangular, or the §04b.4 export branch (forward substitution)
implemented at bf16 and gated`.

---

## §What is UNVERIFIED, by name

An absent row reads as agreement, so this section is a list, not a summary. Two kinds: stages with no
floor, and claims the comparison cannot make.

**No banded floor — 12 of 19 stages**, each printed by name by `--floor-probe` and stored in the band
file's `unverified_stages`, with the same reason and the same closing condition: `mixed_qkv`, `z`,
`b_raw`, `a_raw`, `conv_pre`, `conv_out`, `conv_window`, `q_pre`, `k_pre`, `v_pre`, `beta`, `g_decay`.
They are computed by code the F32 arms and the f64 anchor **share** — all upstream of the scan-dtype
decision of §04.4 — so their distance to the anchor is 0 by construction and a band derived from them
would license nothing that bit-identity does not already. *Closes when* the stage acquires a second
legal implementation of its own (a kernel, or a re-associated producer).

**Claims the comparison cannot make** — `HF_UNVERIFIED`, printed with each entry's `why` and
`closes_when` by every `--compare-hf` run:

| item | why it is not established |
|---|---|
| agreement on a real activation | both sides are fed `gdn_input`, the A-01 RMS-1 seeded fixture; every agreement figure here is a synthetic-input statistic |
| the fused kernels the box does not have | `causal_conv1d` and `flash-linear-attention` are not installed, so the reference ran its own torch fallbacks (the function objects that ran are printed with the section) |
| chunked continuation (§04b.1 row 4) | no pinned case hands a live state to the chunked scan; only chunked-from-zero prefill and single-token recurrent decode are covered |
| the engine's conv-window policy (A-09) | `conv_window` is compared against the *reference's* k_conv-column cache, which is the reference's choice; the engine's 3-column window is a different tensor |
| banded agreement as a VERDICT | **narrowed, not closed.** `--cross-form` exists and credits the band (§§The cross-form verdict): `load_band()` only, `band_check_row()` as the sole comparator, denominators declared before the run, both sabotage directions witnessed on the reference's own bytes. What is still not established is the *reach* of that verdict — 12 unbanded stages, the 82 candidate pairs the licence withholds, the 329 adjudicated rows that measured arm-to-arm bit-identity, and the measured fact that a §04.7 gated-norm defect scaled to 1/2**2 sits inside the `normed` margin. It is a bound on legitimate cross-form spread, not a defect detector. Fresh capture of the two lists: `speccheck/evidence/gdn-hf-unverified-2026-10-26-r2.txt` |
| a defect cannot hide inside the `normed` margin | it can, and the ladder measures how much: `norm_weight_f32` at **1/2**2** is ACCEPTED with 190 of 131,072 elements still changed (maxabs 3.906e-03 vs 3.125e-02 licensed, meanrel 1.988e-06 vs 3.941e-06), and `norm_variance_flat` hides down to 1/2**8. The band may not be tightened from here to close this — a smaller `normed` limit would be a number nobody measured |
| the adjudicated rows are all evidence | **329 of 1,136** measured `ndiff 0` arm-to-arm: true, stronger than the band, and worth nothing as a test of it. The count that stressed a limit is 807 |
| a floor for the 12 upstream stages | as above, by name |

**`NOT_ESTABLISHED`, the injection matrix's own two entries:** the bf16 UT solve (§§above, with its
measured `NotImplementedError`), and `q_scaled_by_dv` — `d_k == d_v == 128` in the committed config, so
`1/sqrt(d_v)` *is* `1/sqrt(d_k)`, the injection and the reference are the same expression and no fixture
can distinguish them. It is excluded from the must-catch set as **DEGENERATE** and printed as a derived
absence, not run as a passing case.

---

## §CLOSED, and re-witnessed rather than deleted

**"file 09 is in scope with its own pinned dump" — closed as a non-item.** The open item asked this
lane to pin a GDN tensor family for the MTP draft layer that the checkpoint does not contain.
`specification/09-mtp-draft-layer.md §09.3` says the draft layer is a full-attention layer, and the
instrument is `specification/derive.py §--check`, which asserts no `mtp.*` key contains `linear_attn`.
`_witness_draft_block_has_no_gdn()` re-reads the **committed safetensors index** on every run and
requires both halves:

```
  CLOSED  the MTP draft block's mixer — 0 `mtp.*linear_attn*` keys of 1045; 6
          `mtp.layers.0.self_attn.*` keys present as the positive control
```

The positive half is the reason this is a witness and not a comment: an index that was empty or
mis-named would "prove" the negative just as happily (`pitfalls/40` §a check that produces no output
cannot fail). The entry stays in the file as `HF_CLOSED` with a `reopens_when` — a checkpoint whose
`mtp.*` keys do include `linear_attn` turns it back into a FAIL rather than a silent gap. **MTP itself
remains in scope** (`DECISIONS.md` D-05); what closed is the claim that the GDN lane owes it a mixer
dump.

---

## §The denominators this lane is held to

From the rc=0 `--selftest` (`speccheck/evidence/gdn-bf16-arm-disposition-2026-10-26.txt` carries the
tally block), 300 derived cases per arm over 2 blocks:

```
  honest   cases 300   caught 300   localised 300   stage_diffs 2084   must_catch_missed 0
           scan_moved_cases 252   pre_scan_moved_cases 122   must_not_catch_violations 0
  stub     cases 300   caught   0   localised   0   stage_diffs    0   must_catch_missed 2084
  crossed  cases 300   caught 300   localised   0   stage_diffs 2384   must_not_catch_violations 300
           crossed_breaches_injected 300   crossed_alarms_fired 300

  unconditional   2060   |AFFECTED[fault]| x derived cases x 2 blocks
  conditional total   24
  derived total     2084   the blinded differ must miss exactly this many allowed stages
```

`arm=stub must_catch_missed == derived total == 2084`, and 2084 = 2060 + 24. **This is the number the
brief carried as 2000, and it is not 2000.** It was 2084 in the pre-existing committed state, before
this pass touched the file, and it is 2084 after: the derived set did not move when
`AFFECTED["scan_in_bf16"]` was rewritten as a derivation, because the closure reproduces the typed
list stage-for-stage. A stub that missed 2083 or 2085 would fail the run — the equality is the gate,
which is the point of deriving a population instead of typing one (`pitfalls/40`).

The three arms are not decoration: `arm=honest` is what proves localisation (`localised == cases`,
`must_not_catch_violations == 0`), `arm=stub` is what proves the gate reads the differ and not a
constant (`caught == 0` while `must_catch_missed` equals a derived number > 0), `arm=crossed` is what
proves the must-not-catch side can fail at all (300/300 alarms fired), so the 0 in `arm=honest` is
earned rather than assumed. The `scan_moved_cases` / `pre_scan_moved_cases` split (252 / 122) is the
witness that both sides of the scan boundary can move — a matrix in which nothing upstream ever moved
would make the pre-scan assertion vacuous.

From the rc=0 `--cross-form` (`speccheck/evidence/gdn-cross-form-2026-10-26.txt`), the same discipline on
the consumer side — every denominator declared before the measurement it is meant to catch a silence in:

```
  xf       printed   1136 parsed   1136 declared   1136
  audit    printed   1218 parsed   1218 declared   1218
  normed   printed    149 parsed    149 declared    149
  margin   printed     72 parsed     72 declared     72      = 4 norm-contract faults x 2 blocks x 3
                                                               passes x 3 arms, x 3 rungs
  sab      printed      4 parsed      4 declared      4      = 2 directions x 2 blocks
  bandctl  printed      4 parsed      4 declared      4      = |BAND_FACTORS|
  files hashed 64 declared 64
  ladder rows 72; ACCEPTed 15; of those 3 left stored bytes changed; 4/4 faults REJECT at full strength
```

---

## §Findings and contradictions from this pass

1. **The brief's 2000 vs the measured 2084.** Reported, not reconciled by editing: see §§above. The
   strongest form of this is checkable without trusting this pass's output at all — the artifact
   committed *before* this pass touched anything already reads 2084:
   `git show HEAD:speccheck/evidence/gdn-selftest-2026-10-26.txt` contains
   `stub must_catch_missed 2084` and `derived total 2084 … no more and no fewer`. The number was
   never 2000 in a captured run of this file, and it did not move under this pass's edits.
2. **The brief listed `core_attn, k_hat, normed, q_hat` as the stages outside the bf16 contract; the
   measurement says otherwise.** The four whose *storage* the arm changes are `q_hat, k_hat, scan_out,
   state`, and the three that move by value are `core_attn, normed, y`. I could not reproduce that
   four-item set from the committed state. If it came from a run whose contract had already been
   narrowed, it is the control, not the arm — which is exactly the failure the split check was written
   to catch.
3. **`normed`'s band is 4 orders of magnitude wider than `state`'s** (3.125e-02 against 1.153e-06).
   That is the gated norm amplifying through `rsqrt` and a SiLU gate, and it is the stage where a
   quantization decision on the scan output would actually be felt. It is left as measured.
4. **Two f64 implementations are not bit-identical to each other.** `scan_out` and `state` each have 26
   rows where the outer-product twin differs from the rank-1 anchor. §04.6's "equivalent" is true to
   f64 precision, not bitwise; had the twin been bit-identical everywhere, the credibility check would
   have FAILED as vacuous.
5. **A 1 ULP perturbation is not distinguishable from legal spread, and that is the correct answer.**
   Both `ulp` sabotages are ACCEPTED by the band while `ndiff 1/4096` proves the comparator saw them.
   The sha256 — not the band — is what keeps a perturbed pinned file out.
6. **The `normed` margin swallows a real defect, measured.** Finding 3 said the margin is wide and left it
   there; the ladder quantifies what "wide" buys. `norm_weight_f32` — the §04.7 gated-norm width injection —
   is ACCEPTED at **1/2**2** with 190 of 131,072 stored elements still changed (maxabs 3.906e-03 against a
   licensed 3.125e-02, meanrel 1.988e-06 against 3.941e-06), and `norm_variance_flat` is accepted down to
   1/2**8 on both blocks with 144 elements still moved. All four norm faults REJECT at full strength on both
   blocks, so this is not the band being broken; it is the band being what it is — a bound on legitimate
   cross-form spread, priced against a probe of legal implementations, and therefore unable to see an
   illegal one that stays under it. The correction is in how the band is *used*, not in the band: a
   `normed` gate must be paired with a must-catch set, and the number to quote when someone proposes
   shipping a kernel on these rows alone is 1/2**2.
7. **The band-content digest the brief carried, `827b155820f8aa89`, is not reproducible from the committed
   artifact**, and `--cross-form` does not print a digest it cannot recompute. What the artifact does carry,
   reproducibly: `sha256sum` of the file is `dd68fd953896da09…`, and the `band` member's object digest over
   `json.dumps(band, sort_keys=True)` — the one `--floor-probe` itself prints — is
   `27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`. Candidates I computed and ruled out
   include the whole-file bytes, the file minus `files_read_sha256`, the `population`, `band_derivation`,
   `unverified_stages`, `generated_from` and `files_read_sha256` members individually, and `limits` alone.
   The section prints the canonicalisation and the command (`pitfalls/40` §provenance and attribution rules
   for reports: a digest is an input/output pairing claim, so it has to name how it was made), so this is a
   discrepancy to reconcile with the measurement lane, not a claim to
   inherit.
8. **`short_conv`'s docstring promised a verdict it cannot deliver.** It said the two conv spellings'
   equivalence was "the `--cross-form` section adjudicates as its own row" — but `conv_window` has no entry
   in the band, `--floor-probe` refuses to floor a stage upstream of the scan-dtype decision, and
   `band_check_row` refuses to invent a limit. Writing a `conv_window` licensor in the consumer would have
   been a band number typed by the section that consumes the band, which is the one thing this lane is not
   allowed to do. The docstring now states what is and is not adjudicated (§What is bit-exact keeps the
   bit-for-bit check `--compare-hf` does make); closing the row needs a second legal conv path for the probe
   to floor against.

---

## §How to re-run this lane

```
PY=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
$PY speccheck/gdn_ref.py --selftest          # three arms + HF comparison + witnesses; SELFTEST PASS
$PY speccheck/gdn_ref.py --compare-hf        # the 150-row bit-identity table and its 14 controls
$PY speccheck/gdn_ref.py --floor-probe       # derives and writes the band; FLOOR PROBE PASS
$PY speccheck/gdn_ref.py --floor-probe --band-out tmp/scratch-band.json   # scratch, never the pinned file
$PY speccheck/gdn_ref.py --cross-form        # credits that band; 1,136 rows, 6 tables, 4 witnesses;
                                              # CROSS-FORM PASS
$PY speccheck/gdn_ref.py --sabotage read_pre_update_state   # one Faults field, not a list; prints
                                              # `SABOTAGE RUN OK`, never `PASS`
$PY tools/dump_reference_gdn.py --verify --out data/dumps/2026-10-26-gdn \
     --manifest checkpoints/dump-manifest-gdn-2026-10-26.json             # VERIFY PASS, 178/178, 0 bad
```

`--floor-probe` refuses `--only-fixture` / `--only-fault` (witnessed: rc=1, `FLOOR PROBE FAIL —
--floor-probe does not honour --only-fixture/--only-fault: the band is derived from a population, so a
subset probe would size a band from rows that were never run`). `--band-out` outside
`speccheck/evidence/` writes a copy carrying a `scratch_artifact` record — the path written to, the
pinned band's path, and why no verdict is credited from a scratch file — and prints `SCRATCH COPY,
marked as one inside the file`; the default path carries no such record, and the section that credits a
band loads `BAND_PATH`, never `--band-out`. So a hand-made band cannot be mistaken for the pinned one,
and `load_band()` runs the same `probe × factor` arithmetic on either. `--cross-form` refuses
`--only-fixture` / `--only-fault` for the same reason the probe does (its population is the licensed grid,
not a subset of it), and refuses to run at all when the band is missing, was written by a foreign
derivation, disagrees with its own `probe × factor` arithmetic, or is a scratch copy — four of those
refusals, plus the seven the band file cannot cause, are witnessed in
`speccheck/evidence/gdn-cross-form-controls-2026-10-26.txt`, which also carries the driver's source in
full because the driver lives in the ignored `tmp/` (restore it to `tmp/w22_crossform_controls.py` and run
it with the venv python above to re-witness all eleven). All of it is CPU-only; none of it
takes a timing, and `tools/gpu_lock.sh` is never acquired
because nothing here touches the iGPU.

## §Open work owed by this lane

- **A `conv_window` licensor the consumer is not allowed to write.** `short_conv`'s two spellings are
  checked bit-for-bit by `--compare-hf` but have no banded row, because the stage is upstream of the
  scan-dtype decision and `--floor-probe` has no second legal implementation to floor against (§Findings
  item 8). Closing it means building that second path in the probe, not typing a limit in `--cross-form`.
- **A narrower `normed` floor, if one is wanted.** The current limit is the probe's maximum × 2 and the
  ladder measures what that tolerates (a real gated-norm width defect at 1/2**2). Tightening it from the
  consumer is forbidden; a tighter number has to come from a probe measurement — e.g. comparing the gated
  norm against a re-associated or MUFU-free implementation of §04.7 and letting THAT set the maximum — and
  until then the band stays where the triangle inequality puts it and the injection matrix stays the
  detector.
- **`--capability`** — the last missing section (`speccheck/gdn_ref.py`'s module docstring named three;
  `--floor-probe` and `--cross-form` are built).
- **A real activation.** Every agreement figure here is a synthetic-input statistic until the dense path
  lands and a real `hidden_states` is pinned.
- **The engine's conv-window policy (A-09)**, which is the engine's tensor, not the reference's.
