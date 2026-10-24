#!/usr/bin/env python3
"""quant_scenarios.py — residency and decode-ceiling arithmetic for the D-01 quantization choice.

Why this exists: DECISIONS.md D-01 (experts at 4-bit, what about attention?) cannot be decided by
taste, and the two binding measured facts are both in this repo:
  * the device-visible budget is ~32 GiB TOTAL per process, and host-pinning does not escape it
    (devices/measured-facts.md section "Residency above the pool", measured 2026-10-26);
  * a warm pure-read at 190-237 GB/s is what a decode step is priced against (pitfalls/00).

The family census is derived from config.json shapes (element counts x bytes) and validated
against model.safetensors.index.json's total_size: derived 66.485 GiB vs 66.965 GiB actual,
0.72 % delta, which is what licenses using it. Re-run --census after any checkpoint change.

Sizes: 1 GiB = 2^30 B, and bits-per-weight includes block scales:
  bf16 16 | Q8_0 8.5 | Q6_K 6.5625 | Q5_K 5.5625 | Q4_K 4.5 | Q4_0 4.5 | IQ4_XS 4.25 | MXFP4 4.25
Ceilings are bytes-read-per-token divided into a GB/s figure and are CEILINGS: the 0.8B engine
reached ~60 % of its own (README), so quote a range, not the ceiling.

Usage: python3 tools/quant_scenarios.py [--census] [--read 237] [--read 190]
"""
import argparse
import json
import os

BPW = {"bf16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5625,
       "Q4_K": 4.5, "Q4_0": 4.5, "IQ4_XS": 4.25, "MXFP4": 4.25, "Q4_K_M": 4.5}

# ---- geometry, read from the checkpoint config, not from a blog post -----------------------
# The config and the shard index are committed under checkpoints/ (a few hundred KB, not
# weights) so this script and every sizing claim in this repo are reproducible offline.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.environ.get("Q36_CONFIG", os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json"))
IDX = os.environ.get("Q36_INDEX",
                     os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json"))


def census():
    """Return {family: bf16_bytes} plus the index's own total_size, or {} if files are absent."""
    if not (os.path.exists(CFG) and os.path.exists(IDX)):
        return {}, None
    c = json.load(open(CFG))
    t = c["text_config"]
    h, L, V = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    hd, nq, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    iv = t["full_attention_interval"]
    E, me, se = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    lk, lv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    kf = t.get("linear_conv_kernel_dim", 4)
    nfull, nlin = L // iv, L - L // iv
    B = 2  # bf16

    def routed(nb):
        return nb * (2 * me * h * E + h * me * E)

    fam = {
        "routed_experts": routed(L) + routed(1),                    # 40 layers + the MTP block
        "gdn_linear_attn": nlin * ((2 * lk * dk + lv * dv) * h + (lv * dv) * h + lv * h
                                   + lv * h + (2 * lk * dk + lv * dv) * kf + 2 * lv + 2 * h + dv),
        "full_attn": (nfull + 1) * ((2 * nq * hd) * h + 2 * (nkv * hd) * h + h * (nq * hd) + 2 * hd),
        "shared_experts": (L + 1) * (3 * se * h + h),
        "routers": (L + 1) * (h * E),
        "embed": V * h,
        "lm_head": V * h,
        "norms_misc": (2 * L + 4) * h + 3 * h * (3 * h),
    }
    return {k: v * B for k, v in fam.items()}, json.load(open(IDX))["metadata"]["total_size"]


FAM_BITS = {
    "routed_experts": "experts", "shared_experts": "shared", "routers": "router",
    "gdn_linear_attn": "attn", "full_attn": "attn", "embed": "embed", "lm_head": "lm_head",
    "norms_misc": "router",
}

# name -> {family-class: bits}. "router" is never quantized by any policy (llama.cpp refuses:
# gating tensors, norms and conv1d stay as they are).
POLICIES = {
    "bf16 (does not fit)":            dict(experts=16.0, shared=16.0, attn=16.0, lm_head=16.0, router=16.0),
    "A experts Q4_K, rest Q8_0":      dict(experts=4.5, shared=8.5, attn=8.5, lm_head=8.5, router=16.0),
    "B experts Q4_K, rest bf16":      dict(experts=4.5, shared=16.0, attn=16.0, lm_head=16.0, router=16.0),
    "C experts MXFP4, rest Q8_0":     dict(experts=4.25, shared=8.5, attn=8.5, lm_head=8.5, router=16.0),
    "D experts IQ4_XS, rest Q8_0":    dict(experts=4.25, shared=8.5, attn=8.5, lm_head=8.5, router=16.0),
    "E experts Q4_K, attn Q4_K too":  dict(experts=4.5, shared=4.5, attn=4.5, lm_head=6.5625, router=16.0),
    "F experts Q8_0, rest Q8_0":      dict(experts=8.5, shared=8.5, attn=8.5, lm_head=8.5, router=16.0),
    "G experts Q4_K, attn Q8, lm Q6": dict(experts=4.5, shared=8.5, attn=8.5, lm_head=6.5625, router=16.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--read", type=float, nargs="+", default=[],
                    help="GB/s figures to price against (one flag, space-separated list)")
    args = ap.parse_args()
    reads = args.read or [237.0, 210.0]

    fam, total = census()
    if not fam:
        print("no checkpoint config/index found; set Q36_CONFIG and Q36_INDEX")
        return 1
    GiB = 2.0 ** 30
    if args.census:
        got = sum(fam.values())
        print("family census (bf16), derived from config.json shapes")
        for k, v in sorted(fam.items(), key=lambda x: -x[1]):
            print("  %-18s %8.3f GiB  %5.2f %%" % (k, v / GiB, 100 * v / total))
        print("  %-18s %8.3f GiB" % ("TOTAL (text)", got / GiB))
        print("  index total_size   %8.3f GiB   delta %.3f GiB (%.2f %%)"
              % (total / GiB, (total - got) / GiB, 100 * (total - got) / total))
        print("  (delta is the vision tower + small tensors; text-only is the line above)")
        return 0

    n_active = 8            # num_experts_per_tok
    n_experts = 256
    print("residency = every weight at its policy width; per-token = attn + shared + %d/%d of"
          % (n_active, n_experts))
    print("the routed experts + lm_head (embed is one row) + router every token\n")
    hdr = "%-34s %9s %9s" % ("policy", "GiB", "GiB/tok")
    hdr += "".join("  %6s" % ("%d GB/s" % r) for r in reads)
    print(hdr)
    print("-" * len(hdr))
    for name, pol in POLICIES.items():
        resident = sum(v * BPW_F(BAM_BITS(k, pol)) for k, v in fam.items()) / GiB
        # per-token: routed experts are read only for the n_active selected
        per = 0.0
        for k, v in fam.items():
            cls = FAM_BITS[k]
            ratio = BAM_BITS(k, pol) / 16.0     # bytes this width costs per bf16 byte
            if cls == "experts":
                per += v * (n_active / n_experts) * ratio
            elif cls == "embed":
                per += 0.0            # one embedding row per token, ~4 KiB
            else:
                per += v * ratio
        per /= GiB
        line = "%-34s %9.2f %9.3f" % (name, resident, per)
        for r in reads:
            line += "  %6.0f" % (r / (per * GiB / 1e9))
        fits = "" if resident <= 30.0 else "   <- above the ~32 GiB budget"
        print(line + fits)
    print("\nBPW used:", BPW)
    return 0


def BAM_BITS(family, pol):
    cls = FAM_BITS[family]
    # token embeddings follow the output tensor's width: same role (a vocab x hidden table read
    # once per token at the ends of the pass), and llama.cpp applies the same rule to both.
    if cls not in pol and cls == "embed":
        cls = "lm_head"
    return pol[cls]


def BPW_F(bits):
    return bits / 16.0


if __name__ == "__main__":
    raise SystemExit(main())
