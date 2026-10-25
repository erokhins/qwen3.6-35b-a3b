#!/usr/bin/env python3
"""tools/moe_stats.py — the 8-of-256 selection statistics, the near-tie analysis and the
floor probes the quantization bands will be sized from.

CPU only: torch-CPU, no device allocation, no clock read anywhere in this file. Every
number it prints is a count, a ratio or a norm; `no timing taken`.

Sections (each asserts its own population, `pitfalls/40`):

  --utilisation   per-expert selection counts over the sampled tokens, experts never hit,
                  and the spread against the uniform expectation T*K/E (derived).
  --near-tie      for every token: the 8/9 selection margin in probability and in logit
                  space, how many rows sit inside one bf16 ULP of the 8th logit, and the
                  CLOSED-FORM flip radius  Δ = (l8 - l9) / ||W8 - W9||_2  — for a linear
                  router with no bias this is the exact minimum ||δx|| that can change the
                  top-8 set. `--near-tie` also Monte-Carlo-tests the closed form, because a
                  counter that reads 0 is vacuous until it is witnessed non-zero.
  --floor         the disagreement between legal implementations of the same math (bf16
                  reference form, our independent form, f32/f64 accumulation, cast-after,
                  per-slot batching, rank order) measured against the fp64 oracle. This is
                  what any bf16 gate's band must sit ABOVE, and it is the number the
                  Q4_K/Q8_0 bands are sized against (`pitfalls/30` §A max-ULP band cannot
                  see a whole-tensor 1-ULP error — long dots need aggregates + a floor probe).
  --quant-proxy   the output-space error of the shipping widths, using the ggml packers:
                  routed experts Q4_K (DECISIONS.md D-01/D-08) and layer 39's attention
                  projections Q8_0 (D-01a). The packers are `tools/gguf_reader.py`'s, which
                  are transcribed from llama.cpp d327bf1f `quantize_row_*_ref`; this file
                  re-implements them vectorised and REQUIRES bit-equality with the reference
                  packers on a seeded sample before it prints a single proxy number.

Output of --floor and --quant-proxy is a PROPOSAL INPUT, not a band: a band is a
user-visible claim (`pitfalls/30` §What is a user decision vs a green gate), so the
derivation is printed next to every number and the decision is handed to the main agent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "speccheck"))
sys.path.insert(0, str(REPO / "tools"))

import moe_ref as R  # noqa: E402

BF16_REL_WIDTH = 2.0 ** -9      # 8 explicit mantissa bits -> half-ULP relative width


# --------------------------------------------------------------------- quantizers (vectorised)
def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


def _rhu(x: torch.Tensor) -> torch.Tensor:
    """C `nearest_int()`: round half AWAY FROM ZERO (`ggml-quants.c`). floor(x+0.5) is not
    it for negative x (-0.5 -> 0 instead of -1), and the packer validation below is
    bit-exact, so the difference is visible."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def q8_0_proxy(xs: torch.Tensor) -> torch.Tensor:
    """Q8_0: blocks of 32, one f16 scale per block, int8 round-to-nearest. 8.5 bpw."""
    if xs.shape[-1] % 32:
        raise AssertionError(f"Q8_0 needs a whole number of 32-element blocks, got "
                             f"{xs.shape[-1]}")
    v = xs.reshape(-1, 32).to(torch.float32)
    amax = v.abs().amax(dim=-1, keepdim=True)
    d = _f16(amax / 127.0)
    q = _rhu(torch.clamp(v / d.clamp(min=torch.finfo(torch.float32).tiny), -127, 127))
    out = torch.where(amax == 0, torch.zeros_like(v), d * q)
    return out.reshape(xs.shape)


def q4_k_proxy(xs: torch.Tensor) -> torch.Tensor:
    """Q4_K: 256-element super-blocks of eight 32-element sub-blocks, a 6-bit scale and a
    6-bit min per sub-block against a shared f16 d/dmin, 4.5 bpw. Mirrors
    `tools/gguf_reader.py` §_qref_k (from `quantize_row_q4_K_ref`), vectorised."""
    if xs.shape[-1] % 256:
        raise AssertionError(f"Q4_K needs whole 256-element super-blocks, got {xs.shape[-1]}")
    v = xs.reshape(-1, 256).to(torch.float32)
    sub = v.reshape(-1, 8, 32)
    lo = sub.amin(dim=-1)
    hi = sub.amax(dim=-1)
    scales = (hi - lo) / 15.0
    mins = lo.clamp(max=0.0).neg()          # max(0, -lo): the offset added before rounding
    max_scale = scales.amax(dim=-1, keepdim=True)
    max_min = mins.amax(dim=-1, keepdim=True)
    d = _f16(max_scale / 63.0)
    dmin = _f16(max_min / 63.0)
    inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_min = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
    ls = torch.clamp(_rhu(inv_scale * scales), max=63.0)
    lm = torch.clamp(_rhu(inv_min * mins), max=63.0)
    step = d * ls
    off = dmin * lm
    q = torch.clamp(_rhu((sub + off.unsqueeze(-1)) / step.unsqueeze(-1).clamp(min=1e-30)),
                    0.0, 15.0)
    out = step.unsqueeze(-1) * q - off.unsqueeze(-1)
    out = torch.where(step.unsqueeze(-1) == 0,
                      (-off).unsqueeze(-1).expand_as(out), out)
    return out.reshape(xs.shape)


def validate_quantizers(seed: int = 20261026) -> dict:
    """Bit-equality against `gguf_reader.quant_ref_deq`, the packers transcribed from
    llama.cpp d327bf1f. A vectorised re-implementation that agrees with itself proves
    nothing (`pitfalls/40` §A check whose reference side is derived from the thing under
    test cannot adjudicate it either), so this must pass before any proxy number prints."""
    import gguf_reader as GR

    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, mine in (("Q8_0", q8_0_proxy), ("Q4_K", q4_k_proxy)):
        n = 256 * 8
        # mixed magnitudes so both the tiny-scale and the large-scale paths are covered;
        # torch.logspace takes no generator, hence the explicit 10**linspace
        envelope = 10.0 ** torch.linspace(-3.0, 2.0, n)
        xs = torch.randn(n, generator=gen) * envelope
        xs[: n // 8] = 0.0                       # include the all-zero block case
        ref = torch.tensor(GR.quant_ref_deq(name, xs.tolist()))
        got = mine(xs)
        out[name] = {"n": int(n), "max_abs_diff": float((ref - got).abs().max()),
                     "bitexact": bool(torch.equal(ref, got))}
        if not out[name]["bitexact"]:
            raise AssertionError(f"{name} vectorised packer disagrees with "
                                 f"gguf_reader.quant_ref_deq: {out[name]}")
    return out


# ------------------------------------------------------------------------- helpers
def top9(logits: torch.Tensor, g: R.Geometry):
    """The 8 selected and the 9th (best unselected) per row, in logit space.

    Softmax is monotone, so the top-k of the probabilities is the top-k of the logits and
    the margin can be read in logit space — which is where the bf16 ULP comparison belongs."""
    vals, ids = torch.topk(logits.to(torch.float32), g.top_k + 1, dim=-1)
    return vals, ids


def pct(v: torch.Tensor, qs=(0.0, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)):
    """Percentiles, `qs` given in percent (torch wants fractions; the labels stay the
    reader's units)."""
    if v.numel() == 0:
        raise AssertionError("percentile over an empty tensor")
    q = torch.tensor([x / 100.0 for x in qs], dtype=torch.float64)
    r = torch.quantile(v.to(torch.float64), q)
    return {f"p{q_:g}": float(x) for q_, x in zip(qs, r)}


# ------------------------------------------------------------------------- sections
def section_utilisation(g, blocks, tokens, seed, command) -> dict:
    print("\n## utilisation — which of the 256 experts a block actually reads")
    out = {}
    expected_per_expert = tokens * g.top_k / g.num_experts
    print(f"  derived uniform expectation: {tokens} tokens x top_k {g.top_k} / E "
          f"{g.num_experts} = {expected_per_expert:.2f} selections per expert")
    for block in blocks:
        w = R.load_block_weights(block, g, roles=("router",))
        x = R.make_input(tokens, g, seed)
        r = R.route(x, w.tensors["router"], g, R.Form())
        ids = r["topk_ids"].flatten()
        counts = torch.bincount(ids, minlength=g.num_experts)
        never = int((counts == 0).sum().item())
        # derived population check: K*T draws over E experts must touch at least 1 and at
        # most E; a section that counted nothing would print zeros forever
        if int(counts.sum().item()) != tokens * g.top_k:
            raise AssertionError(f"{block.name}: selection count {int(counts.sum())} != "
                                 f"T*K={tokens * g.top_k}")
        chi2 = float(((counts.float() - expected_per_expert) ** 2
                      / expected_per_expert).sum())
        top = torch.topk(counts, 5)
        out[block.name] = {
            "tokens": tokens, "selections": int(counts.sum()),
            "experts_hit": int((counts > 0).sum()), "experts_never_hit": never,
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.float().mean()), "std": float(counts.float().std()),
            "chi2_vs_uniform": chi2,
            "busiest": [[int(i), int(c)] for i, c in zip(top.indices.tolist(),
                                                          top.values.tolist())],
            "entropy_bits": float(-(lambda p: (p * p.clamp(min=1e-30).log2()).sum())(
                counts.float() / counts.sum())),
        }
        o = out[block.name]
        print(f"  {block.name:<7} hit {o['experts_hit']:>3}/{g.num_experts}  "
              f"never {never:>3}  min {o['min']:>3} max {o['max']:>3} "
              f"mean {o['mean']:.2f} std {o['std']:.2f} chi2 {chi2:.0f} "
              f"H {o['entropy_bits']:.3f}/{math.log2(g.num_experts):.3f} bits")
        if o["experts_hit"] == 0 or o["selections"] == 0:
            raise AssertionError(f"{block.name}: utilisation counted nothing")
        del w
    return {"expected_per_expert": expected_per_expert, "per_block": out}


def section_near_tie(g, blocks, tokens, seed, command) -> dict:
    print("\n## near-tie — how close the 8/9 decision sits to the bf16 rounding of the "
          "router's own logits")
    out = {}
    for block in blocks:
        w = R.load_block_weights(block, g, roles=("router",))
        W = w.tensors["router"].to(torch.float32)
        x = R.make_input(tokens, g, seed)
        r = R.route(x, w.tensors["router"], g, R.Form())
        logits = r["logits"]
        vals, ids = top9(logits, g)                      # [T, K+1]
        sel, ninth = vals[:, :g.top_k], vals[:, g.top_k]
        l8, l9 = sel[:, -1], ninth                    # >= 0 by construction
        if bool((l9 > l8).any()):
            raise AssertionError("topk returned an unsorted tail: the 9th logit exceeds "
                                 "the 8th, so the margin definition below is wrong")
        margin = l8 - l9                                   # >= 0 by construction
        probs = r["probs"]
        p8 = probs.gather(-1, ids[:, :g.top_k]).amin(dim=-1)
        p9 = probs.gather(-1, ids[:, g.top_k:]).squeeze(-1)
        margin_p = p8 - p9
        # exact flip radius for a bias-free linear router: min ||d|| s.t. (x+d).(w8-w9)
        # changes sign of the 8/9 comparison.  ||d||_min = (l8-l9)/||w8-w9||_2 .
        dW = (W[ids[:, g.top_k - 1]] - W[ids[:, g.top_k]]).norm(dim=-1)
        radius = margin / dW.clamp(min=1e-30)
        xnorm = x.to(torch.float32).norm(dim=-1)
        radius_rel = radius / xnorm.clamp(min=1e-30)
        ulp8 = torch.tensor([R.ulp_of(float(v)) for v in l8.tolist()])
        inside_ulp = margin <= ulp8
        row = {
            "tokens": tokens,
            "margin_logit": pct(margin, (0.0, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)),
            "margin_prob": pct(margin_p, (0.0, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)),
            "flip_radius_rel": pct(radius_rel, (0.0, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)),
            "rows_margin_le_1ulp_of_l8": int(inside_ulp.sum()),
            "rows_flip_radius_rel_le": {
                f"{e:g}": int((radius_rel <= e).sum())
                for e in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)},
            "rows_total": tokens,
        }
        out[block.name] = row
        print(f"  {block.name:<7} margin(logit) p0 {row['margin_logit']['p0']:.3e} "
              f"p1 {row['margin_logit']['p1']:.3e} p50 {row['margin_logit']['p50']:.3e} "
              f"p99 {row['margin_logit']['p99']:.3e} | rows within 1 bf16 ULP of l8: "
              f"{row['rows_margin_le_1ulp_of_l8']}/{tokens}")
        print(f"          flip radius ||d||/||x|| p0 {row['flip_radius_rel']['p0']:.3e} "
              f"p1 {row['flip_radius_rel']['p1']:.3e} "
              f"p50 {row['flip_radius_rel']['p50']:.3e} "
              f"p99 {row['flip_radius_rel']['p99']:.3e} "
              f"p99.9 {row['flip_radius_rel']['p99.9']:.3e}")
        print(f"          rows with radius <= eps: "
              + "  ".join(f"{k}:{v}" for k, v in row["rows_flip_radius_rel_le"].items()))
        # Monte-Carlo witness for the closed form: perturb and re-route.
        mc = {}
        gen = torch.Generator().manual_seed(seed + 7)
        clean_ids = r["topk_ids"]
        for eps in (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            if eps == 0.0:
                flips = 0
            else:
                d = torch.randn(tokens, g.hidden_size, generator=gen)
                d = d / d.norm(dim=-1, keepdim=True) * (eps * xnorm).unsqueeze(-1)
                pert = (x.to(torch.float32) + d).to(torch.bfloat16)
                r2 = R.route(pert, w.tensors["router"], g, R.Form())
                flips = int((~(r2["topk_ids"] == clean_ids).all(dim=-1)).sum())
            pred = int((radius_rel <= eps).sum())
            mc[f"{eps:g}"] = {"observed_flips": flips, "predicted_from_radius": pred}
            print(f"          MC eps={eps:<6} flips {flips:>3}/{tokens}   "
                  f"predicted from closed form {pred:>3}/{tokens}")
        row["monte_carlo"] = mc
        if mc["0"]["observed_flips"] != 0:
            raise AssertionError(f"{block.name}: eps=0 flipped a decision — the router or "
                                 "the comparator is nondeterministic")
        if mc["0.1"]["observed_flips"] == 0:
            raise AssertionError(f"{block.name}: even a 10% input perturbation never "
                                 "changed a selection — the MC arm cannot adjudicate")
        del w
    return {"per_block": out}


def section_floor(g, blocks, tokens, seed, command) -> dict:
    print("\n## floor probe — the disagreement between legal implementations of the same "
          "math, against the fp64 oracle")
    print("  (a band must sit ABOVE these numbers; the numbers themselves do not set one)")
    arms = ("ref", "spec", "cast_after", "accum_f32", "accum_f64", "rank_order",
            "per_slot")
    stages = ("router_logits", "topk_weights", "expert_raw", "routed_sum", "shared_out",
              "combined_out")
    out = {}
    for block in blocks:
        w = R.load_block_weights(block, g)
        x = R.make_input(tokens, g, seed)
        exact = R.moe_block(x, w, g, "exact")
        out[block.name] = {}
        for arm in arms:
            o = R.moe_block(x, w, g, arm)
            for stage in stages:
                m = R.metrics(o[stage].to(torch.float64), exact[stage].to(torch.float64),
                              name=f"{block.name}/{arm}/{stage}")
                rec = {"max_abs": m["max_abs"], "rel_to_scale": m["max_abs_over_scale"],
                       "one_minus_cos": m["one_minus_cos"], "ndiff": m["ndiff"],
                       "n": m["n"], "n_gt_1ulp": m["n_gt_1ulp"],
                       "mean_abs": m["mean_abs"]}
                out[block.name][f"{arm}/{stage}"] = rec
                if stage in ("expert_raw", "routed_sum", "combined_out", "topk_weights"):
                    print(f"  {block.name:<7} {arm:<11} {stage:<14} "
                          f"maxabs {rec['max_abs']:.3e}  rel/scale "
                          f"{rec['rel_to_scale']:.3e}  1-cos {rec['one_minus_cos']:.3e}  "
                          f"ndiff {rec['ndiff']}/{rec['n']} ({100 * rec['ndiff'] / rec['n']:.1f}%)  "
                          f">1ulp {rec['n_gt_1ulp']}")
        del w
    if not out:
        raise AssertionError("floor probe over zero blocks")
    return {"arms": list(arms), "stages": list(stages), "per_block": out}


def section_quant_proxy(g, blocks, tokens, seed, command) -> dict:
    print("\n## quantization proxy — what the shipping widths do to the block OUTPUT "
          "(weight-space floors are W1's audit: quantization/evidence/"
          "audit-floor-probe-2026-10-26.json)")
    checked = validate_quantizers(seed)
    print(f"  packer validation vs tools/gguf_reader.py quant_ref_deq (llama.cpp d327bf1f): "
          f"{checked}")
    out = {}
    for block in blocks:
        w = R.load_block_weights(block, g)
        x = R.make_input(tokens, g, seed)
        clean = R.moe_block(x, w, g, "ref")
        exact = R.moe_block(x, w, g, "exact")
        ids = clean["topk_ids"]
        hit = sorted(set(int(e) for e in ids.flatten().tolist()))
        qgu = w.tensors["gate_up"].clone()
        qdn = w.tensors["down"].clone()
        for e in hit:                                     # only the experts this input reads
            qgu[e] = q4_k_proxy(w.tensors["gate_up"][e].to(torch.float32)).to(torch.bfloat16)
            qdn[e] = q4_k_proxy(w.tensors["down"][e].to(torch.float32)).to(torch.bfloat16)
        qw = R.BlockWeights(block, dict(w.tensors, gate_up=qgu, down=qdn), w.source_shards)
        qout = R.moe_block(x, qw, g, "ref")
        rec = {}
        for stage in ("expert_raw", "routed_sum", "combined_out"):
            m_bf16 = R.metrics(qout[stage].to(torch.float64),
                               clean[stage].to(torch.float64), name=stage)
            m_exact = R.metrics(qout[stage].to(torch.float64),
                                exact[stage].to(torch.float64), name=stage)
            rec[stage] = {"vs_bf16_ref": {k: m_bf16[k] for k in
                                          ("max_abs", "max_abs_over_scale", "one_minus_cos",
                                           "ndiff", "n")},
                          "vs_fp64_oracle": {k: m_exact[k] for k in
                                             ("max_abs", "max_abs_over_scale",
                                              "one_minus_cos", "ndiff", "n")}}
            print(f"  {block.name:<7} experts Q4_K {stage:<13} vs bf16 ref: maxabs "
                  f"{rec[stage]['vs_bf16_ref']['max_abs']:.3e} rel/scale "
                  f"{rec[stage]['vs_bf16_ref']['max_abs_over_scale']:.3e} 1-cos "
                  f"{rec[stage]['vs_bf16_ref']['one_minus_cos']:.3e} | vs fp64 oracle: "
                  f"rel/scale {rec[stage]['vs_fp64_oracle']['max_abs_over_scale']:.3e} "
                  f"1-cos {rec[stage]['vs_fp64_oracle']['one_minus_cos']:.3e}")
        # weight-space relative error of the proxy itself, so a reader can see the two
        # halves of the arithmetic separately
        rel_gu = float((qgu[hit].to(torch.float32) - w.tensors["gate_up"][hit].to(torch.float32)
                        ).norm() / w.tensors["gate_up"][hit].to(torch.float32).norm())
        rel_dn = float((qdn[hit].to(torch.float32) - w.tensors["down"][hit].to(torch.float32)
                        ).norm() / w.tensors["down"][hit].to(torch.float32).norm())
        rec["weight_space_rel_fro"] = {"gate_up_Q4_K": rel_gu, "down_Q4_K": rel_dn,
                                       "experts_quantised": len(hit)}
        print(f"  {block.name:<7} weight-space rel Frobenius: gate_up {rel_gu:.4e}  "
              f"down {rel_dn:.4e}  ({len(hit)} experts read by this input)")
        out[block.name] = rec
        del w, qgu, qdn

    # Q8_0 attention proxy: layer 39 is the deepest acquired full-attention layer.
    print("  Q8_0 attention proxy (linear-map only: no rope, no mask, no GDN — this is a "
          "projection-width number, not an attention-block number)")
    attn = {}
    names = {n: f"model.language_model.layers.39.self_attn.{n}.weight"
             for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
    for tag, name in names.items():
        t = R.load_tensor(R.SHARD_DIR / R.shard_of(name), name).to(torch.float32)
        deq = q8_0_proxy(t)
        gen = torch.Generator().manual_seed(seed + 11)
        n_rows = 256
        xx = torch.randn(n_rows, t.shape[1], generator=gen)
        xx = xx / xx.pow(2).mean(-1, keepdim=True).sqrt()
        y0 = xx @ t.t()
        y1 = xx @ deq.t()
        rel = float((y1 - y0).norm() / y0.norm())
        attn[tag] = {"shape": list(t.shape), "shard": R.shard_of(name),
                     "weight_space_rel_fro": float((deq - t).norm() / t.norm()),
                     "output_rel_fro": rel, "n_rows": n_rows}
        print(f"  layer39   attn {tag:<7} {str(attn[tag]['shape']):<16} weight rel "
              f"{attn[tag]['weight_space_rel_fro']:.4e}  output rel {rel:.4e}")
        del t, deq, y0, y1
    out["_q8_0_attention_layer39"] = attn
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--floor-tokens", type=int, default=32)
    ap.add_argument("--proxy-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--utilisation", action="store_true")
    ap.add_argument("--near-tie", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--quant-proxy", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    selected = [s for s in ("utilisation", "near_tie", "floor", "quant_proxy")
                if getattr(args, s.replace("-", "_")) or args.all]
    if not selected:
        print("MOE_STATS FAIL: no section selected — a run that computes nothing cannot pass")
        return 1

    threads = R.set_threads(args.threads)
    g = R.load_geometry()
    blocks = [R.Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    command = " ".join([sys.executable] + [str(Path(__file__).resolve())] + sys.argv[1:])
    print(f"# moe_stats — torch {torch.__version__} CPU threads={threads} seed={args.seed} "
          f"blocks={[b.name for b in blocks]}")
    print(f"# geometry: E={g.num_experts} top_k={g.top_k} I={g.moe_intermediate_size} "
          f"H={g.hidden_size} blocks_in_text_path={g.n_blocks} "
          f"expert_bytes={g.expert_bytes} ({g.expert_bytes / 2**20:.1f} MiB bf16)")
    print(f"# sections selected: {', '.join(selected)}")

    results = {"date": "2026-10-26", "command": command,
               "geometry": {"E": g.num_experts, "top_k": g.top_k,
                            "I": g.moe_intermediate_size,
                            "shared_I": g.shared_expert_intermediate_size,
                            "H": g.hidden_size, "n_blocks": g.n_blocks,
                            "expert_bytes": g.expert_bytes},
               "seed": args.seed, "sections": {}}
    if "utilisation" in selected:
        results["sections"]["utilisation"] = section_utilisation(g, blocks, args.tokens,
                                                                 args.seed, command)
    if "near_tie" in selected:
        results["sections"]["near_tie"] = section_near_tie(g, blocks, args.tokens,
                                                           args.seed, command)
    if "floor" in selected:
        results["sections"]["floor"] = section_floor(g, blocks, args.floor_tokens,
                                                     args.seed, command)
    if "quant_proxy" in selected:
        results["sections"]["quant_proxy"] = section_quant_proxy(g, blocks, args.proxy_tokens,
                                                                 args.seed, command)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\n# json written: {p} ({p.stat().st_size:,} B)")
    print("\nMOE_STATS PASS" if selected else "MOE_STATS FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
