# Unsloth Qwen3.5-35B-A3B GGUF benchmarks (external evidence)

**Provenance.** Fetched 2026-10-26 from `https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks`
(their "Qwen3.5 GGUF Benchmarks" page, March 5 2026 revision: >150 KL-divergence runs, ~9 TB of
GGUFs, and their research artifacts published as `unsloth/Qwen3.5-35B-A3B-Experiments-GGUF` with
121 configs). Numbers are theirs, transcribed so `quantization/README.md` can argue from them
offline. **Qwen3.5-35B-A3B is geometrically identical to this project's Qwen3.6-35B-A3B** (see
`DECISIONS.md` D-06), so a per-tensor sensitivity result measured on it transfers — this is the only
external evidence in the repo that is about *this* architecture rather than a nearby one.

## The size / quality table (their numbers)

PPL is wiki-text at 512 context; KLD 99.9 % is KL divergence against the bf16 model at the 99.9th
percentile, their headline metric. Lower is better in both.

| quantizer | level | GB | PPL | KLD 99.9 % | mean KLD |
|---|---|---|---|---|---|
| Unsloth | IQ2_XXS | 9.09 | 7.716 | 4.2221 | 0.1846 |
| Unsloth | Q2_K_XL | 12.04 | 7.0438 | 2.9092 | 0.097 |
| Unsloth | IQ3_XXS | 13.12 | 6.7829 | 1.5296 | 0.0501 |
| Unsloth | IQ3_S | 14.13 | 6.7715 | 1.4193 | 0.0457 |
| Unsloth | Q3_K_M | 15.54 | 6.732 | 0.9726 | 0.0324 |
| Unsloth | Q3_K_XL | 16.06 | 6.7245 | 0.9539 | 0.0308 |
| Unsloth | **MXFP4_MOE** | **18.17** | 6.600 | **0.7789** | 0.0272 |
| Unsloth | **Q4_K_M** | **18.49** | 6.6053 | **0.5478** | 0.0192 |
| Unsloth | Q4_K_L | 18.82 | 6.5905 | 0.4828 | 0.0150 |
| Unsloth | Q4_K_XL | 19.17 | 6.5918 | 0.4097 | 0.0137 |
| Unsloth | Q5_K_XL | 23.22 | 6.5489 | 0.236 | 0.0069 |
| Unsloth | Q6_K_S | 26.56 | 6.5456 | 0.2226 | 0.0065 |
| Unsloth | Q6_K_XL | 28.22 | 6.5392 | 0.1437 | 0.0041 |
| Unsloth | Q8_K_XL | 36.04 | 6.5352 | 0.1033 | 0.0026 |
| AesSedai | IQ4_XS | 16.40 | 6.6447 | 0.8067 | 0.0235 |
| AesSedai | IQ3_S | 12.65 | 6.9152 | 1.8669 | 0.0613 |
| AesSedai | Q4_K_M | 20.62 | 6.5665 | 0.3171 | 0.0096 |
| AesSedai | Q5_K_M | 24.45 | 6.5356 | 0.21 | 0.0058 |
| bartowski | Q4_K_M | 19.77 | 6.6097 | 0.5771 | 0.0182 |
| bartowski | IQ4_XS | 17.42 | 6.6234 | 0.7265 | 0.0234 |
| noctrex | MXFP4_MOE_BF16 | 20.55 | 6.5948 | 0.7939 | 0.0248 |
| Ubergarm | Q4_0 | 19.79 | 6.5784 | 0.4829 | 0.0142 |

Reads that matter for D-01/D-08:

- **Q4_K_M sits at ~18.5–20.6 GB on this model and is the knee.** Everything below Q3_K_XL degrades
  fast (KLD 0.95 → 1.53 → 2.91); everything above Q5_K_XL buys KLD for GB.
- **At equal size, Q4_K_M beats MXFP4_MOE** (0.5478 vs 0.7789 at 18.49 vs 18.17 GB). Their own
  conclusion, quoted: "MXFP4 uses 4.25 bits per weight, whilst Q4_K uses 4.5 bits per weight. It's
  better to use Q4_K than MXFP4 when choosing between them", and they have **retired MXFP4 from all
  their mixed quants** except the pure `MXFP4_MOE` type. This is the direct counter-evidence to the
  crowd claim that MXFP4 has lower perplexity than Q4_K_M (which exists too — a Reddit thread and
  secondary posts, perplexity-only, on other models); unsloth's own §3 of the same page warns that
  PPL and KLD move with the calibration set, which is exactly why a perplexity-only claim on a
  different model does not settle a KLD comparison on this one.
- Their speed table for this model (their rig, tokens/s generation): q4_k 90.38, mxfp4 90.67,
  q6_k 90.50, q8_0 90.33 — i.e. **the 4-bit vs 8-bit choice barely moved decode speed there**, while
  the i-quants cost ~5 % (iq3_xxs 85.68, iq2_s 85.04). Useful as a prior that a dequant inner loop
  is not automatically the bottleneck on a bandwidth-bound decode — but their rig is not this iGPU.
- Imatrix: "definitely helps reduce KLD & PPL, at the cost of 5-10 % slower inference".

## Their per-tensor sensitivity findings (the part that answers "what about attention")

From the same page, from their per-tensor bit-width sweep (the 121-config artifact):

- "**Quantizing any attn_\* is especially sensitive** for hybrid architectures, and so leaving them in
  higher precision works well." This model is the hybrid case (30 GDN + 10 full-attn layers).
- "**Quantizing ssm_out (Mamba layers) is not a good idea**" — and "ssm_out at q2_k does dramatically
  worse", i.e. the GDN output projection is the worst single tensor they found. Direct analogue here:
  `linear_attn.out_proj.weight`.
- Best tensors to compress: "`ffn_up_exps` and `ffn_gate_exps` are generally ok to quantize to 3 bit.
  `ffn_down_exps` is slightly more sensitive." So the routed experts are the right place for the bits,
  and `down_proj` is the one to watch there.
- IQ3_XXS on the ffn family "seems to be best compromise on disk space and 99.9 % KLD change. 2 bits
  cause more degradation."
- A March 2026 revision pushed *maximum* KLD down (not just 99.9 %): e.g. UD-Q4_K_XL 5.894 → 2.877 at
  +7.8 % size, which they present as the robustness fix for outliers.

Consequence for `quantization/README.md`: the recommendation (experts Q4_K, attention/GDN Q8_0) is
the direct transcription of the two sentences in bold above, and the "+40 % if attention goes to
Q4_K" option is exactly the trade their finding warns about — which is why it is proposed as a funded
experiment with decision-level gates, not as a default.
