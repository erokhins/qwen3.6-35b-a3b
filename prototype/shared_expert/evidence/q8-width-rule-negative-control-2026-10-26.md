# Negative control — the Q8_0 scale rule, and what happens when it is wrong

This file exists so the numbers that came out of a **broken fixture** are kept as proof the check
bites, and never as a verdict about the kernel. The device arm was never run against these bytes (the
lane's owner refused it on 2026-10-26, correctly: a device verdict over bytes that are not Q8_0 would
be evidence about a fixture bug forever annotated as a kernel verdict).

## Commands that produced these numbers

```
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_published_q8_rule.py
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_q8_exactness.py
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python oracle_arm.py --mode derive --q8-rule 512
./gate            # the failing state, before any fix: RESULT FAIL: oracle --mode derive failed
```

## The artifact says which rule it uses

Decoding the **published** `blk.0.ffn_gate_shexp.weight` (`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`,
dtype `Q8_0`, 1048576 elements) with `tools/gguf_reader.py §dequant`, against the pinned HF bf16 dump
of the same tensor:

```
RULECHECK published_q8_0_decode_vs_pinned_hf      maxabs=0.00079155  relL2=0.00612683
RULECHECK repo_proxy(amax/127)_vs_pinned_hf       maxabs=0.00079155  relL2=0.00612396
RULECHECK published_decode_vs_repo_proxy          maxabs=0.00125313  relL2=0.00158685   (two independent implementations)
RULECHECK lane_pack(amax/512)_decoded_vs_pinned_hf maxabs=0.153076   relL2=0.58983      ← 96.3× worse
```

Two in-repo authorities state the rule and agree with the file: `tools/gguf_reader.py §_qref_q8_0`
(`d = amax / 127.0`) and `tools/moe_stats.py §q8_0_proxy` (`d = _f16(amax / 127.0)`).

## Why /512 is a damaged pack and not a finer one

With the normal `[-127, 127]` clamp, a step of `amax/512` clips every element above 127/512 = 24.8 %
of its block max:

```
Q8PACK family=gate divisor=127 clamp_eq127=33138/1048576 (3.1603 %) elements_equal_block_max=33121 (3.1587 %, 1/32=3.1250 %) relL2_vs_dump=0.006123959 f32_vs_f64_inexact=0 max_sig_bits=15
Q8PACK family=gate divisor=512 clamp_eq127=551575/1048576 (52.6023 %) elements_equal_block_max=33121 (3.1587 %) relL2_vs_dump=0.5911241 f32_vs_f64_inexact=0 max_sig_bits=16
```

(up and down behave the same: 3.164 %/3.163 % correct, 52.1 %/50.8 % clipped.)

The derived bound the gate now enforces is `elements_eq_own_blockmax × 1.05` — geometry, not
comfort: a conformant pack puts ±127 on an element only because that element **is** its block max, and
the 5 % slack is the measured effect of rounding the step to f16. 33138 ≤ 34778 passes; 551575 does not.

## The control, run and refused

```
$ oracle_arm.py --mode derive --q8-rule 512     # rc=1
Q8RULE family=gate  divisor=512 clamp_eq127=551575 clamp_bound_from_blockmax=34778 … relL2_vs_dump=0.5911241
q8_rule_check: divisor=512 is not Q8_0 (the published file and both in-repo references use amax/127);
refusing to pack a fixture that is not the width this lane exists to exercise
```

The refusal happens **after** the measurement is printed, on purpose: an injection that cannot print
its own bad number is an injection that cannot be seen firing. The full transcript is
`control-q8-rule-512-2026-10-26.log` in this directory.

## What the fixed state measured

`relL2_vs_dump` for the lane's own pack is now 0.006123959 / 0.005871895 / 0.006033145 (gate/up/down),
inside the published tensor's own 0.006126833 / 0.005874989 / 0.006036182, and the layout discriminator
went from "the wrong candidate is nearer on maxabs" to `nearest_wrong_ratio=53.661x` on relL2. The
header's `≤11+7 significand bits, EXACT in f32` claim measured true: `f32_vs_f64_inexact=0`,
`max_sig_bits=15` against f32's 24.
