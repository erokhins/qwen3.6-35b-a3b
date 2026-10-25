#!/usr/bin/env python3
"""derive.py — the arithmetic witness for every number printed in specification/.

Why this exists: `AGENTS.md` and `pitfalls/40-gates-and-admissions.md` forbid a spec number that no
command reproduces. The committed shard index (`checkpoints/Qwen3.6-35B-A3B.safetensors.index.json`)
gives tensor *names and shards but no shapes* (`checkpoints/README.md` §"What they are good for"), so
the spec's shapes come from `checkpoints/Qwen3.6-35B-A3B.config.json` geometry, its name/dtype claims
come from the safetensors headers of the four shards on disk in `models/qwen36-src/` (which cover
layers 0, 1, 38, 39 and the whole `mtp.*` block — `README.md` §Download, `tools/mtp_shard_cost.py`),
and its "exactness" claims about a norm's stored values come from reading a few thousand elements of
those same shard files. stdlib only, CPU only; it never touches the iGPU and takes no timing.

Every figure in `specification/*.md` of the form "N tensors", "X GiB", "Y MiB/token" is transcribed
from one of these subcommands. If the spec and this script disagree, the script is right (it reads the
bytes) and the spec has rotted.

Usage:
    python3 specification/derive.py --header [--filter SUB]   # names/shapes/dtypes from the shard headers
    python3 specification/derive.py --families                # §01 inventory + census + index reconciliation
    python3 specification/derive.py --per-token               # decode read budget: bf16 / policy A / published
    python3 specification/derive.py --state                   # KV + GDN recurrent + conv state per request
    python3 specification/derive.py --mtp                     # the mtp.* inventory from the real headers
    python3 specification/derive.py --values NAME [NAME...]    # real stored values (stats) for a tensor
    python3 specification/derive.py --check                   # assert every header-witnessed claim §01-§09 makes
"""
import argparse
import json
import os
import struct
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")
IDX = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json")
SHARDS = os.path.join(REPO, "models/qwen36-src")
SHARD_NAMES = [
    "model-00001-of-00026.safetensors",
    "model-00002-of-00026.safetensors",
    "model-00025-of-00026.safetensors",
    "model-00026-of-00026.safetensors",
]
GiB = 2.0 ** 30
MiB = 2.0 ** 20
KiB = 2.0 ** 10
BYTES_PER_DTYPE = {"F32": 4, "BF16": 2, "F16": 2, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8}

# widths used by the two columns §10 has to print side by side (bpw includes block scales;
# same table as tools/quant_scenarios.py BPW and tools/gguf_header_types.py BPW)
BPW = {"bf16": 16.0, "F32": 32.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5625, "Q4_K": 4.5}


def cfg():
    return json.load(open(CFG))["text_config"]


def header(shard):
    """Parse a safetensors header without torch: 8-byte little-endian length, then JSON.

    Returns (tensor_map, data_base) where data_base = 8 + header_bytes: safetensors
    `data_offsets` are relative to the END of the header, not to the start of the file, and reading
    them as absolute offsets silently returns header bytes decoded as floats.
    """
    p = os.path.join(SHARDS, shard)
    if not os.path.exists(p):
        sys.exit("missing %s — see README.md §Download (D-03 step 1) before running this" % p)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def all_headers():
    """Merge the four shard headers; refuse if a tensor name appears twice.

    Returns (merged, shard_of, data_base_of) — data_base is per shard, because every offset is
    relative to its own file's header end.
    """
    merged, seen, base = {}, {}, {}
    for s in SHARD_NAMES:
        hdr, db = header(s)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if k in merged:
                sys.exit("tensor %s appears in two of the four shards — the shard set is wrong" % k)
            merged[k] = v
            seen[k] = s
            base[k] = db
    return merged, seen, base


def elems(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def layer_types(t):
    kinds = t["layer_types"]
    assert len(kinds) == t["num_hidden_layers"], "layer_types length != num_hidden_layers"
    return kinds


def sec(title):
    print("\n== %s" % title)


def expected_text_tensors(t):
    """The complete text-path inventory, derived from config geometry only.

    Returns {tensor_name: (family, shape)}. The name set is checked against the committed shard index
    by cmd_families, so a spec inventory error — a missing family, a wrong layer count, a forgotten
    MTP tensor — is a loud failure here instead of an argument in a later kernel review.
    """
    h, L, V = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    hd, nq, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    E, me, se = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    lk, lv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    kf = t["linear_conv_kernel_dim"]
    out = {}

    def put(name, fam, shape):
        assert name not in out, "duplicate derived name %s" % name
        out[name] = (fam, shape)

    P = "model.language_model.layers.%d."
    put("model.language_model.embed_tokens.weight", "embed", [V, h])
    put("lm_head.weight", "lm_head", [V, h])
    put("model.language_model.norm.weight", "norms", [h])
    for i, kind in enumerate(layer_types(t)):
        put(P % i + "input_layernorm.weight", "norms", [h])
        put(P % i + "post_attention_layernorm.weight", "norms", [h])
        put(P % i + "mlp.gate.weight", "router", [E, h])
        put(P % i + "mlp.experts.gate_up_proj", "routed_experts", [E, 2 * me, h])
        put(P % i + "mlp.experts.down_proj", "routed_experts", [E, h, me])
        put(P % i + "mlp.shared_expert.gate_proj.weight", "shared_experts", [se, h])
        put(P % i + "mlp.shared_expert.up_proj.weight", "shared_experts", [se, h])
        put(P % i + "mlp.shared_expert.down_proj.weight", "shared_experts", [h, se])
        put(P % i + "mlp.shared_expert_gate.weight", "shared_experts", [1, h])
        if kind == "linear_attention":
            cd = 2 * lk * dk + lv * dv          # conv channels = q|k|v width
            put(P % i + "linear_attn.in_proj_qkv.weight", "gdn", [cd, h])
            put(P % i + "linear_attn.in_proj_z.weight", "gdn", [lv * dv, h])
            put(P % i + "linear_attn.in_proj_a.weight", "gdn", [lv, h])
            put(P % i + "linear_attn.in_proj_b.weight", "gdn", [lv, h])
            put(P % i + "linear_attn.conv1d.weight", "gdn", [cd, 1, kf])
            put(P % i + "linear_attn.A_log", "gdn_control", [lv])
            put(P % i + "linear_attn.dt_bias", "gdn_control", [lv])
            put(P % i + "linear_attn.norm.weight", "gdn_control", [dv])
            put(P % i + "linear_attn.out_proj.weight", "gdn", [h, lv * dv])
        else:
            put(P % i + "self_attn.q_proj.weight", "full_attn", [2 * nq * hd, h])   # q|gate, §05.1
            put(P % i + "self_attn.k_proj.weight", "full_attn", [nkv * hd, h])
            put(P % i + "self_attn.v_proj.weight", "full_attn", [nkv * hd, h])
            put(P % i + "self_attn.o_proj.weight", "full_attn", [h, nq * hd])
            put(P % i + "self_attn.q_norm.weight", "norms", [hd])
            put(P % i + "self_attn.k_norm.weight", "norms", [hd])
    M = "mtp.layers.0."                          # the draft block: full-attn layer + MoE + fusion
    put(M + "input_layernorm.weight", "norms", [h])
    put(M + "post_attention_layernorm.weight", "norms", [h])
    put(M + "self_attn.q_proj.weight", "mtp_attn", [2 * nq * hd, h])
    put(M + "self_attn.k_proj.weight", "mtp_attn", [nkv * hd, h])
    put(M + "self_attn.v_proj.weight", "mtp_attn", [nkv * hd, h])
    put(M + "self_attn.o_proj.weight", "mtp_attn", [h, nq * hd])
    put(M + "self_attn.q_norm.weight", "norms", [hd])
    put(M + "self_attn.k_norm.weight", "norms", [hd])
    put(M + "mlp.gate.weight", "router", [E, h])
    put(M + "mlp.experts.gate_up_proj", "routed_experts", [E, 2 * me, h])
    put(M + "mlp.experts.down_proj", "routed_experts", [E, h, me])
    put(M + "mlp.shared_expert.gate_proj.weight", "shared_experts", [se, h])
    put(M + "mlp.shared_expert.up_proj.weight", "shared_experts", [se, h])
    put(M + "mlp.shared_expert.down_proj.weight", "shared_experts", [h, se])
    put(M + "mlp.shared_expert_gate.weight", "shared_experts", [1, h])
    put("mtp.fc.weight", "mtp_fusion", [h, 2 * h])
    put("mtp.pre_fc_norm_embedding.weight", "norms", [h])
    put("mtp.pre_fc_norm_hidden.weight", "norms", [h])
    put("mtp.norm.weight", "norms", [h])
    return out


def cmd_header(args):
    merged, seen, _ = all_headers()
    rows = sorted(merged.items())
    if args.filter:
        rows = [(k, v) for k, v in rows if args.filter in k]
    for k, v in rows:
        print("%-72s %-6s %-22s %12d elems  %s"
              % (k, v["dtype"], v["shape"], elems(v["shape"]), seen[k].replace("model-", "shard-")))
    print("\n%d tensors printed (headers of the 4 shards on disk)" % len(rows))


def cmd_families(args):
    """§01: the inventory derived from config, checked name-for-name against the index."""
    t = cfg()
    exp = expected_text_tensors(t)
    wm = json.load(open(IDX))["weight_map"]
    idx_text = {k for k in wm if not k.startswith("model.visual")}
    merged, _, _ = all_headers()

    print("§01 inventory: %d text tensors derived from config geometry vs %d in the committed index"
          % (len(exp), len(idx_text)))
    missing = sorted(idx_text - set(exp))
    extra = sorted(set(exp) - idx_text)
    if missing:
        print("  IN INDEX, NOT DERIVED (%d): %s" % (len(missing), missing[:10]))
    if extra:
        print("  DERIVED, NOT IN INDEX (%d): %s" % (len(extra), extra[:10]))
    if not missing and not extra:
        print("  name sets are IDENTICAL — every text tensor in the index is accounted for, and "
              "nothing is invented")

    fam_e, fam_n = defaultdict(int), defaultdict(int)
    for name, (f, shape) in exp.items():
        fam_e[f] += elems(shape)
        fam_n[f] += 1
    B = 2  # bf16 storage dtype, per text_config.dtype
    total_e = sum(fam_e.values())
    print("\n%-16s %6s %16s %10s %8s" % ("family", "count", "elements", "GiB", "share"))
    for k in sorted(fam_e, key=lambda x: -fam_e[x]):
        print("%-16s %6d %16d %10.4f %7.2f %%"
              % (k, fam_n[k], fam_e[k], fam_e[k] * B / GiB, 100.0 * fam_e[k] / total_e))
    print("%-16s %6d %16d %10.4f" % ("TOTAL text", sum(fam_n.values()), total_e, total_e * B / GiB))

    vis_b = vis_n = 0
    for k, v in merged.items():
        if k.startswith("model.visual"):
            vis_b += elems(v["shape"]) * BYTES_PER_DTYPE[v["dtype"]]
            vis_n += 1
    idx_total = json.load(open(IDX))["metadata"]["total_size"]
    print("\nindex total_size        %14d B  %8.4f GiB" % (idx_total, idx_total / GiB))
    print("vision, measured (%3d)  %14d B  %8.4f GiB   (every %d name is in the 4 shards held)"
          % (vis_n, vis_b, vis_b / GiB, vis_n))
    print("index minus vision      %14d B  %8.4f GiB" % (idx_total - vis_b, (idx_total - vis_b) / GiB))
    print("derived text x 2 B      %14d B  %8.4f GiB" % (total_e * B, total_e * B / GiB))
    d = (idx_total - vis_b) - total_e * B
    print("delta                   %14d B  (%d elements) — a complete bf16 inventory reads 0"
          % (d, d // B))

    sec("dtype witnesses from the shard headers (the CHECKPOINT, not the GGUF)")
    probes = ["model.language_model.layers.0.linear_attn.A_log",
              "model.language_model.layers.0.linear_attn.dt_bias",
              "model.language_model.layers.0.linear_attn.norm.weight",
              "model.language_model.layers.0.linear_attn.conv1d.weight",
              "model.language_model.layers.0.mlp.gate.weight",
              "model.language_model.layers.0.input_layernorm.weight",
              "model.language_model.layers.0.mlp.experts.gate_up_proj",
              "model.language_model.layers.39.self_attn.q_norm.weight",
              "model.language_model.layers.39.self_attn.q_proj.weight",
              "model.language_model.layers.39.self_attn.k_proj.weight",
              "model.language_model.layers.0.mlp.shared_expert_gate.weight",
              "model.language_model.embed_tokens.weight",
              "lm_head.weight"]
    for p in probes:
        v = merged.get(p)
        print("  %-62s %-6s %s" % (p, v["dtype"] if v else "not-held", v["shape"] if v else ""))

    sec("what tools/quant_scenarios.py --census prints for the same families (committed instrument)")
    print("  run: python3 tools/quant_scenarios.py --census")
    print("  its family grouping differs: gdn_linear_attn 1.416 GiB, norms_misc 0.071 GiB,")
    print("  text total 65.720 GiB, and it reports the index delta as 1.86 % 'vision tower + small")
    print("  tensors'. Compare against the rows above; specification/ASSUMPTIONS.md A-07 records")
    print("  the reconciliation this script produces and where the two instruments disagree.")


def _report_rows(w, mtp=True):
    """Yield (component, bytes/token) for a width assignment.

    `w` maps family -> bytes read per bf16 element: gate_up, down, router, control, shared, attn,
    lm_head, embed. Splitting gate_up from down is what lets the published column print next to
    policy A's, because the published file puts down_proj at Q5_K while gate/up stay Q4_K.
    """
    t = cfg()
    h, V = t["hidden_size"], t["vocab_size"]
    hd, nq, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    E, me, se = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    lk, lv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    kf = t["linear_conv_kernel_dim"]
    kinds = layer_types(t)
    nfull, nlin = kinds.count("full_attention"), kinds.count("linear_attention")
    nlayers_moe = nfull + nlin + (1 if mtp else 0)   # 40 stack layers (+ the MTP block)

    k = t["num_experts_per_tok"]
    gu_tok = 2 * me * h * k                    # gate_up_proj rows read per token per layer
    dn_tok = h * me * k                        # down_proj rows read per token per layer
    router_tok = E * h
    shared_tok = 3 * se * h + h
    fa_tok = (2 * nq * hd) * h + 2 * (nkv * hd) * h + h * (nq * hd) + 2 * hd
    gdn_tok = ((2 * lk * dk + lv * dv) * h + (lv * dv) * h + lv * h + lv * h
               + (2 * lk * dk + lv * dv) * kf + lv * dv * h)          # qkv,z,a,b,conv,out_proj
    ctrl_tok = 2 * lv + dv                                            # A_log, dt_bias, gdn norm
    return [
        ("routed experts gate|up (8 of 256)", gu_tok * nlayers_moe * w["gate_up"]),
        ("routed experts down (8 of 256)", dn_tok * nlayers_moe * w["down"]),
        ("routers (all 256 logits)", router_tok * nlayers_moe * w["router"]),
        ("shared experts + gate", shared_tok * nlayers_moe * w["shared"]),
        ("full attention (%d%s)" % (nfull, " + MTP" if mtp else ""),
         fa_tok * (nfull + (1 if mtp else 0)) * w["attn"]),
        ("GDN projections (%d layers)" % nlin, gdn_tok * nlin * w["attn"]),
        ("GDN control (A_log,dt_bias,norm)", ctrl_tok * nlin * w["control"]),
        ("lm_head (untied, read whole)", V * h * w["lm_head"]),
        ("embed (one row)", h * w["embed"]),
    ]


def _w(**kw):
    """bytes read per bf16 element, from a format name — so each policy column reads as the decision
    it is rather than as a magic number. Only known BPW keys are accepted (no silent default)."""
    out = {}
    for kk, vv in kw.items():
        if vv not in BPW:
            sys.exit("_w: unknown format %r (known: %s)" % (vv, sorted(BPW)))
        out[kk] = BPW[vv] / 8.0
    return out


def cmd_per_token(args):
    """Decode read budget: the bytes one greedy token touches. State/KV reads are in --state."""
    def show(label, widths, mtp=True):
        sec(label)
        tot = 0.0
        print("%-38s %10s %10s" % ("component", "GiB/token", "MiB/token"))
        for name, b in _report_rows(widths, mtp):
            print("%-38s %10.4f %10.3f" % (name, b / GiB, b / MiB))
            tot += b
        print("%-38s %10.4f %10.3f  <- weight bytes read per token, KV/state excluded"
              % ("TOTAL", tot / GiB, tot / MiB))
        return tot / GiB

    print("Decode read budget for ONE greedy token, weights only (no KV, no state, no logits write).")
    print("Experts count num_experts_per_tok=8 of num_experts=256 per MoE layer; the MTP draft block")
    print("adds an 41st MoE layer and an 11th attention layer when a draft step runs (D-05).\n")
    show("bf16 — every family at 16 bpw (does not fit the pool; this is the ORACLE column)",
         _w(gate_up="bf16", down="bf16", router="bf16", control="bf16", shared="bf16",
            attn="bf16", lm_head="bf16", embed="bf16"))
    show("policy A (D-01/D-01a/D-09/D-10): experts Q4_K, attn/GDN/shared/embed/lm_head Q8_0, "
         "router + GDN control F32",
         _w(gate_up="Q4_K", down="Q4_K", router="F32", control="F32", shared="Q8_0",
            attn="Q8_0", lm_head="Q8_0", embed="Q8_0"))
    show("the published unsloth UD-Q4_K_M as it stands on disk (quantization/evidence/"
         "q4km-published-width-map-2026-10-26.md): gate/up Q4_K, down Q5_K, lm_head Q6_K",
         _w(gate_up="Q4_K", down="Q5_K", router="F32", control="F32", shared="Q8_0",
            attn="Q8_0", lm_head="Q6_K", embed="Q8_0"))
    print("\nThe committed policy table is quantization/README.md §'The policies, priced'; reproduce")
    print("with: python3 tools/quant_scenarios.py --read 237 210 190   (policy A: 2.331 GiB/token).")
    print("Where the two disagree, the row is GDN out_proj — see specification/ASSUMPTIONS.md A-07.")
    print("Published column: down_proj is Q5_K on 38 of 41 layers and Q6_K on 3, so that line is a")
    print("lower bound on the file's real cost (the 3 Q6_K layers are counted at Q5_K).")
    return 0


def cmd_state(args):
    """Per-request state: KV over the full-attn layers, recurrent + conv over the linear ones."""
    t = cfg()
    hd, nkv = t["head_dim"], t["num_key_value_heads"]
    lv, dk, dv = t["linear_num_value_heads"], t["linear_key_head_dim"], t["linear_value_head_dim"]
    lk, kf = t["linear_num_key_heads"], t["linear_conv_kernel_dim"]
    kinds = layer_types(t)
    nfull, nlin = kinds.count("full_attention"), kinds.count("linear_attention")
    mtp_kv = 1  # the draft block is a full-attention layer with its own KV (§09)

    sec("KV cache — full_attention layers only, model dtype bf16")
    per_layer = 2 * nkv * hd * 2
    print("2 (K and V) x n_kv %d x head_dim %d x 2 B = %d B per layer per token"
          % (nkv, hd, per_layer))
    print("stack only:      %d layers x %d B = %d B = %.1f KiB per token"
          % (nfull, per_layer, nfull * per_layer, nfull * per_layer / KiB))
    print("with MTP layer:  %d layers x %d B = %d B = %.1f KiB per token"
          % (nfull + mtp_kv, per_layer, (nfull + mtp_kv) * per_layer,
             (nfull + mtp_kv) * per_layer / KiB))
    print("NOTE: README.md §'Memory and roofline arithmetic' says 10 KiB/token over 10 full-attn")
    print("layers; the same row then says that is ~1.7x the 0.8B's 12 KiB/token, and 12 x 1.67 = 20.")
    print("The arithmetic above is the one the geometry gives; ASSUMPTIONS A-08 records the delta.")

    sec("GDN recurrent state — dtype is mamba_ssm_dtype (%s)" % t["mamba_ssm_dtype"])
    rdt = {"float32": 4, "bfloat16": 2}[t["mamba_ssm_dtype"]]
    rec_layer = lv * dk * dv * rdt
    print("%d V heads x d_k %d x d_v %d x %d B = %d B per layer"
          % (lv, dk, dv, rdt, rec_layer))
    print("%d linear layers x %d B = %d B = %.2f MiB per request"
          % (nlin, rec_layer, nlin * rec_layer, nlin * rec_layer / MiB))

    sec("GDN short-conv state — the last k_conv inputs of mixed_qkv, per linear layer")
    conv_ch = 2 * lk * dk + lv * dv
    for taps, tag in ((kf, "HF cache_utils keeps conv_kernel_size columns"),
                      (kf - 1, "a minimal window (k_conv-1) is enough for the next output")):
        for cdt, dtname in ((rdt, t["mamba_ssm_dtype"]), (2, "model dtype bf16 (what HF-ref stores)")):
            c_layer = conv_ch * taps * cdt
            print("  %-42s %-22s %d ch x %d taps x %d B x %d layers = %8d B = %6.3f MiB"
                  % (tag, dtname, conv_ch, taps, cdt, nlin, nlin * c_layer, nlin * c_layer / MiB))
    print("  HF stores this window in the ACTIVATION dtype (mixed_qkv, i.e. bf16): cache_utils"
          " §LinearAttentionLayer.lazy_initialization allocates zeros_like(conv_states).")
    print("  The F32 row is printed because an engine that keeps the window in the scan's dtype pays"
          " 2x for it; that is a declared engine choice, not a config fact — §08.2, A-09.")

    sec("per-request totals")
    kv = lambda n: nfull * per_layer * n
    rec = nlin * rec_layer
    conv3 = nlin * conv_ch * (kf - 1) * rdt
    conv4 = nlin * conv_ch * kf * rdt
    for lanes in (1, 128, 1024, 4096, 32768):
        tot = rec + conv3 + kv(lanes)
        print("  %6d lanes: KV %8.2f MiB + recurrent %6.2f MiB + conv(k-1) %6.3f MiB = %8.2f MiB"
              % (lanes, kv(lanes) / MiB, rec / MiB, conv3 / MiB, tot / MiB))
    print("  draft-block KV adds %d B/token (%.1f KiB); with MTP active at 4096 lanes the total is %.2f MiB"
          % (mtp_kv * per_layer, mtp_kv * per_layer / KiB,
             (rec + conv3 + (nfull + mtp_kv) * per_layer * 4096) / MiB))
    print("  fit check: how many 4096-lane requests per 1 GiB of state pool = %.1f (conv k-1) / %.1f (conv k)"
          % (GiB / (rec + conv3 + kv(4096)), GiB / (rec + conv4 + kv(4096))))


def cmd_mtp(args):
    merged, seen, _ = all_headers()
    mtp = {k: v for k, v in merged.items() if k.startswith("mtp.")}
    if not mtp:
        sys.exit("no mtp.* tensors in the four shard headers — is models/qwen36-src populated?")
    sec("mtp.* tensors — real safetensors headers (ground truth rank 1)")
    tot_e = tot_b = 0
    for k in sorted(mtp):
        v = mtp[k]
        e = elems(v["shape"])
        b = e * BYTES_PER_DTYPE[v["dtype"]]
        tot_e += e
        tot_b += b
        print("  %-58s %-6s %-20s %12d elems %9.3f MiB" % (k, v["dtype"], v["shape"], e, b / MiB))
    print("  %-58s %-6s %-20s %12d elems %9.3f MiB"
          % ("TOTAL %d tensors" % len(mtp), "", "", tot_e, tot_b / MiB))
    print("  shards: %s" % sorted({seen[k] for k in mtp}))
    exp = {k: v for k, v in expected_text_tensors(cfg()).items() if k.startswith("mtp.")}
    print("\n  derived-from-config vs header: %s"
          % ("IDENTICAL names" if set(exp) == set(mtp) else "MISMATCH %s"
             % (sorted(set(exp) ^ set(mtp)))))
    bad = [k for k in exp if exp[k][1] != mtp[k]["shape"]]
    print("  shape agreement: %s" % ("every derived shape equals the header's" if not bad
                                     else "MISMATCH on %s" % bad))
    print("  PLAN.md's fetch table says 19 tensors / 41,943,040 elts / 79.9 MiB; the header says")
    print("  %d tensors / %d elts / %.3f MiB. The whole delta is the block's two fused expert"
          % (len(mtp), tot_e, tot_b / MiB))
    print("  tensors (805,306,368 elts = 1.5 GiB), which that estimate omitted — ASSUMPTIONS A-10.")


def read_values(name):
    """Decode one tensor's stored values from its shard file (stdlib, no torch, CPU only).

    Only the families the spec needs a witness for are decodable (BF16, F32); anything else exits
    rather than guessing a bit pattern, because a guessed decode reads like a measurement.
    """
    merged, seen, base = all_headers()
    if name not in merged:
        sys.exit("%s: not in the 4 shards held" % name)
    v = merged[name]
    dt = v["dtype"]
    if dt not in BYTES_PER_DTYPE:
        sys.exit("%s: dtype %s is not decodable here" % (name, dt))
    lo, hi = v["data_offsets"]
    with open(os.path.join(SHARDS, seen[name]), "rb") as fh:
        fh.seek(base[name] + lo)
        raw = fh.read(hi - lo)
    n = len(raw) // BYTES_PER_DTYPE[dt]
    if n > (1 << 20):
        sys.exit("%s: %d elements is too many to print as values; this instrument is for norms and "
                 "control tensors (a whole-tensor statistic belongs in a floor probe)" % (name, n))
    if dt == "BF16":
        # bf16 -> f32 is a 16-bit LEFT shift of the raw bits: on little-endian the two bf16 bytes
        # become the HIGH half of the f32 word, so the padding goes in front, not after.
        import array
        a = array.array("f")
        a.frombytes(b"".join(b"\x00\x00" + raw[2 * i:2 * i + 2] for i in range(n)))
        return [x for x in a if x == x]
    if dt == "F32":
        import array
        a = array.array("f")
        a.frombytes(raw)
        return [x for x in a if x == x]
    sys.exit("%s: dtype %s is not decodable here" % (name, dt))


def cmd_values(args):
    """Read a tensor's real stored values from the shard file.

    Witnesses the claims a shape cannot make: that the block norms are stored zero-centred (values
    near 0 for the (1+w) form), that the GDN output norm is stored plain (near 1), and that A_log
    holds log(A) rather than -exp(A).
    """
    merged, _, _ = all_headers()
    for name in args.values:
        if name not in merged:
            print("%s: NOT in the 4 shards held" % name)
            continue
        vals = read_values(name)
        v = merged[name]
        k = min(8, len(vals))
        print("%-64s %s %s" % (name, v["dtype"], v["shape"]))
        print("   n=%d  min %.6g  max %.6g  mean %.6g  max|x| %.6g"
              % (len(vals), min(vals), max(vals), sum(vals) / len(vals),
                 max(abs(x) for x in vals)))
        print("   first %d: %s" % (k, ["%.6g" % x for x in vals[:k]]))


def cmd_check(args):
    """Assert the header-witnessed claims §01-§09 make. Non-zero exit on any drift."""
    merged, _, _ = all_headers()
    t = cfg()
    fails = []

    def expect(cond, msg):
        print("%-5s %s" % ("ok" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    def shape_of(n):
        return merged.get(n, {}).get("shape")

    def dtype_of(n):
        return merged.get(n, {}).get("dtype")

    h = t["hidden_size"]
    E, me = t["num_experts"], t["moe_intermediate_size"]
    se = t["shared_expert_intermediate_size"]
    lk, lv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    kf = t["linear_conv_kernel_dim"]
    hd, nq, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    P = "model.language_model.layers.%d."
    kinds = layer_types(t)
    # witness layers must be ones whose shards are on disk: 0, 1, 38, 39 (README §Download)
    held = [0, 1, 38, 39]
    ilin = next(i for i in held if kinds[i] == "linear_attention")
    iful = next(i for i in held if kinds[i] == "full_attention")
    print("witness layers: linear %d, full %d (from config layer_types, restricted to the shards held)"
          % (ilin, iful))

    expect(shape_of(P % 0 + "mlp.experts.gate_up_proj") == [E, 2 * me, h],
           "experts.gate_up_proj shape == [256, 2*512, 2048]")
    expect(shape_of(P % 0 + "mlp.experts.down_proj") == [E, h, me],
           "experts.down_proj shape == [256, 2048, 512]")
    expect(dtype_of(P % 0 + "mlp.experts.gate_up_proj") == "BF16",
           "expert tensors are stored BF16 in the checkpoint")
    expect(shape_of(P % 0 + "mlp.gate.weight") == [E, h], "mlp.gate.weight shape == [256, 2048]")
    expect(dtype_of(P % 0 + "mlp.gate.weight") == "BF16",
           "the router is stored BF16 in the checkpoint (the F32 router is a GGUF-side widening)")
    expect(shape_of(P % 0 + "mlp.shared_expert.gate_proj.weight") == [se, h],
           "shared_expert.gate_proj shape == [512, 2048]")
    expect(shape_of(P % 0 + "mlp.shared_expert.down_proj.weight") == [h, se],
           "shared_expert.down_proj shape == [2048, 512]")
    expect(shape_of(P % 0 + "mlp.shared_expert_gate.weight") == [1, h],
           "mlp.shared_expert_gate shape == [1, 2048]")
    expect(shape_of(P % ilin + "linear_attn.in_proj_qkv.weight") == [2 * lk * dk + lv * dv, h],
           "GDN in_proj_qkv shape == [2*d_k*h_k + d_v*h_v, d] = [%d, %d]"
           % (2 * lk * dk + lv * dv, h))
    expect(shape_of(P % ilin + "linear_attn.in_proj_z.weight") == [lv * dv, h],
           "GDN in_proj_z shape == [d_v*h_v, d] = [%d, %d]" % (lv * dv, h))
    expect(shape_of(P % ilin + "linear_attn.in_proj_a.weight") == [lv, h],
           "GDN in_proj_a shape == [h_v, d] (beta is in_proj_b)")
    expect(shape_of(P % ilin + "linear_attn.conv1d.weight") == [2 * lk * dk + lv * dv, 1, kf],
           "GDN conv1d shape == [conv_dim, 1, k_conv] (depthwise, no bias)")
    expect(shape_of(P % ilin + "linear_attn.out_proj.weight") == [h, lv * dv],
           "GDN out_proj shape == [d, d_v*h_v] (this is the tensor the committed census omits)")
    expect(dtype_of(P % ilin + "linear_attn.A_log") == "BF16",
           "GDN A_log is stored BF16 in the checkpoint (mamba_ssm_dtype is the SCAN dtype)")
    expect(dtype_of(P % ilin + "linear_attn.dt_bias") == "BF16",
           "GDN dt_bias is stored BF16 in the checkpoint")
    expect(shape_of(P % ilin + "linear_attn.norm.weight") == [dv],
           "GDN output norm weight is per value-head-dim (%d)" % dv)
    expect(shape_of(P % iful + "self_attn.q_proj.weight") == [2 * nq * hd, h],
           "full-attn q_proj rows == 2*n_q*d_head = %d (the output gate is packed here, §05.1)"
           % (2 * nq * hd))
    expect(shape_of(P % iful + "self_attn.k_proj.weight") == [nkv * hd, h],
           "full-attn k_proj shape == [n_kv*d_head, d] = [%d, %d]" % (nkv * hd, h))
    expect(shape_of(P % iful + "self_attn.v_proj.weight") == [nkv * hd, h],
           "full-attn v_proj shape == k_proj (no separate value width)")
    expect(shape_of(P % iful + "self_attn.o_proj.weight") == [h, nq * hd],
           "full-attn o_proj shape == [d, n_q*d_head] = [%d, %d]" % (h, nq * hd))
    expect(shape_of(P % iful + "self_attn.q_norm.weight") == [hd]
           and shape_of(P % iful + "self_attn.k_norm.weight") == [hd],
           "q_norm/k_norm are per head_dim (%d), applied per head before rope" % hd)
    mtp = {k: v for k, v in merged.items() if k.startswith("mtp.")}
    expect(len(mtp) == 19, "mtp.* tensor count == 19 (got %d)" % len(mtp))
    expect(elems(mtp["mtp.fc.weight"]["shape"]) == 2 * h * h,
           "mtp.fc.weight == [d, 2d] = the (embedding|hidden) fusion (§09.2)")
    expect(any(k.startswith("mtp.layers.0.mlp.experts.") for k in mtp),
           "the draft block owns its own 256-expert MoE (1.5 GiB bf16)")
    expect(any(k.startswith("mtp.layers.0.self_attn.") for k in mtp),
           "the draft block is a full-attention layer (its own KV, §09.3)")
    expect(not any(k.startswith("mtp.layers.0.linear_attn.") for k in mtp),
           "the draft block has NO GDN block (config mtp block is attention+MoE)")
    expect(not any(k.startswith("mtp.") and "embed_tokens" in k for k in merged),
           "no dedicated MTP embedding table (config mtp_use_dedicated_embeddings: false)")

    sec("value witnesses (§03.1 which norm form, §04.6 the A_log sign)")
    for name, pred, why in [
        (P % 0 + "input_layernorm.weight", lambda a: max(abs(min(a)), abs(max(a))) < 0.5,
         "block norm stored near 0 => the form is out = x_hat * (1 + w), not out = w * x_hat"),
        (P % ilin + "linear_attn.norm.weight", lambda a: min(a) > 0.5 and max(a) < 2.5,
         "GDN output norm stored near 1 => plain out = w * x_hat (a DIFFERENT form, §04.7)"),
        (P % ilin + "linear_attn.A_log", lambda a: min(a) < 0.0 and max(a) > 0.0,
         "A_log spans negative and positive => it stores log(A), so the decay must be -exp(A_log)"),
        (P % ilin + "linear_attn.dt_bias", lambda a: min(a) < 0.0 < max(a),
         "dt_bias spans both signs (it is trained, not the ones-init) => softplus(a + dt_bias) "
         "must be computed on the sum; and max |x| sets the softplus linear-region question (§04.5)"),
    ]:
        try:
            vals = read_values(name)
        except SystemExit as e:
            expect(False, "%s: %s" % (name, e))
            continue
        print("  %s\n      n=%d min %.6g max %.6g mean %.6g — %s"
              % (name, len(vals), min(vals), max(vals), sum(vals) / len(vals), why))
        expect(pred(vals), "%s satisfies its value predicate" % name)

    print("\n%s (%d failures)" % ("ALL CHECKS PASSED" if not fails else "CHECKS FAILED", len(fails)))
    return 1 if fails else 0


def read_gguf_tensor_infos(path):
    """Minimal GGUF v2/v3 header reader: returns (infos, data_start, meta).

    infos maps tensor name -> (dims, type_id, offset), offset relative to `data_start`, which is the
    header end rounded up to the file's `general.alignment` (default 32). Treating `offset` as an
    absolute file position is the classic way to compare two unrelated byte ranges and call the
    agreement a pass. The fetch grows (16 -> 64 -> 256 MiB) because on a 248k-vocab model the
    metadata section alone is tens of MiB — the same ladder `tools/gguf_header_types.py` uses, and
    its `Cur` class is reused rather than reimplemented (one spelling per concept).
    """
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import gguf_header_types as gt

    size = 0
    for size in (16 << 20, 64 << 20, 256 << 20, 1 << 30):
        with open(path, "rb") as fh:
            head = fh.read(size)
        c = gt.Cur(head)
        try:
            if c.take(4) != b"GGUF":
                sys.exit("gguf: not a GGUF file")
            c.u32()
            n_tensors, n_kv = c.u64(), c.u64()
            meta = {}
            for _ in range(n_kv):
                k = c.s()
                vt = c.u32()
                meta[k] = gt.skip_array(c) if vt == 9 else c.value(vt)
            infos = {}
            for _ in range(n_tensors):
                name = c.s()
                dims = tuple(c.u64() for _ in range(c.u32()))
                dt = c.u32()
                off = c.u64()
                infos[name] = (dims, dt, off)
            break
        except EOFError:
            continue
    else:
        sys.exit("gguf: header never fit in %d bytes — refuse to guess" % size)
    align = int(meta.get("general.alignment", 32))
    return infos, ((c.p + align - 1) // align) * align, meta


def cmd_gguf_vs_ckpt(args):
    """Witness the converter's value transforms, byte for byte, on the unquantized families.

    Three claims the spec makes about the shipping artifact that a shape check cannot see:
    norms are stored as w+1 in F32, `ssm_a` is -exp(A_log), `ssm_dt.bias` is dt_bias unchanged.
    """
    import math
    path = args.gguf_vs_ckpt
    infos, data_start, meta = read_gguf_tensor_infos(path)
    print("file %s" % path)
    print("n_tensors %d  data_start %d  arch %s" % (len(infos), data_start, meta.get("general.architecture")))

    def rd_gguf_f32(name, n):
        dims, dt, off = infos[name]
        if dt != 0:
            sys.exit("%s: expected F32 (type 0), file has type %d" % (name, dt))
        with open(path, "rb") as fh:
            fh.seek(data_start + off)
            raw = fh.read(4 * n)
        import array
        a = array.array("f")
        a.frombytes(raw)
        return list(a)

    pairs = [
        ("blk.0.attn_norm.weight", "model.language_model.layers.0.input_layernorm.weight", 1.0,
         "GGUF attn_norm == HF input_layernorm + 1 (zero-centred norm baked in at conversion)"),
        ("blk.0.post_attention_norm.weight",
         "model.language_model.layers.0.post_attention_layernorm.weight", 1.0,
         "GGUF post_attention_norm == HF post_attention_layernorm + 1"),
        ("blk.39.attn_q_norm.weight", "model.language_model.layers.39.self_attn.q_norm.weight", 1.0,
         "GGUF attn_q_norm == HF q_norm + 1 (layer 39 is the full-attn witness held on disk)"),
        ("blk.0.ssm_norm.weight", "model.language_model.layers.0.linear_attn.norm.weight", 0.0,
         "GGUF ssm_norm == HF linear_attn.norm.weight UNCHANGED (the gated norm is plain w*x_hat)"),
        ("blk.0.ssm_a", "model.language_model.layers.0.linear_attn.A_log", None,
         "GGUF ssm_a == -exp(HF A_log), precomputed at conversion — do NOT exp/negate again. NOT"
         " owed bit-exactness: exp is evaluated by the producer (tolerance band printed)", "perm"),
        ("blk.0.ssm_dt.bias", "model.language_model.layers.0.linear_attn.dt_bias", 0.0,
         "GGUF ssm_dt.bias == HF dt_bias, value unchanged, order permuted", "perm"),
        ("blk.0.ssm_beta.weight", "model.language_model.layers.0.linear_attn.in_proj_b.weight", 0.0,
         "GGUF ssm_beta == HF in_proj_b with its V-head ROWS permuted (matrix witness, not just a"
         " 32-element vector)", "rows"),
        ("blk.0.ssm_alpha.weight", "model.language_model.layers.0.linear_attn.in_proj_a.weight", 0.0,
         "GGUF ssm_alpha == HF in_proj_a with its V-head ROWS permuted", "rows"),
    ]
    # rows-per-tensor for the matrix witnesses: GGUF stores dims reversed, so [2048, 32] here is
    # 32 rows of 2048 contiguous floats, which is the checkpoint's [32, 2048] row-major layout
    kind_cols_rows = {"blk.0.ssm_beta.weight": (2048, 32), "blk.0.ssm_alpha.weight": (2048, 32)}
    def vhead_perm(n_v, n_k, head_dim=1):
        """The GGUF V-head permutation, quoted from the converter and then tested, not assumed.

        `engines/llama.cpp` conversion/qwen.py §_LinearAttentionVReorderBase._reorder_v_heads builds
        `arange(n).reshape(n_k, n_v//n_k, head_dim).permute(1, 0, 2).reshape(n)` — "reorder V heads
        from grouped (by K head) to tiled order" — and applies it to the V rows of in_proj_qkv, the
        rows of in_proj_z / in_proj_a / in_proj_b, the A_log / dt_bias elements, the V channels of
        conv1d and the columns of out_proj. So the GGUF's bytes for those tensors are a PERMUTATION
        of the checkpoint's, which no shape check can see.
        """
        n = n_v * head_dim
        per_k = n_v // n_k
        return [i * per_k * head_dim + j * head_dim + d
                for j in range(per_k) for i in range(n_k) for d in range(head_dim)]

    perm = vhead_perm(cfg()["linear_num_value_heads"], cfg()["linear_num_key_heads"], 1)
    bad = 0
    for pair in pairs:
        gname, hname, plus1, why = pair[:4]
        kind = pair[4] if len(pair) > 4 else "vec"
        if gname not in infos:
            print("  SKIP  %-32s not in this GGUF" % gname)
            continue
        if kind == "rows":
            cols, n_rows = kind_cols_rows[gname]
            hv = read_values(hname)
            gv = rd_gguf_f32(gname, cols * n_rows)
            worst = 0.0
            for j in range(n_rows):
                src = perm[j]
                for c in range(cols):
                    worst = max(worst, abs(gv[j * cols + c] - hv[src * cols + c]))
            ok = worst == 0.0
            bad += 0 if ok else 1
            print("  %-5s %-30s vs %-52s rows=%d cols=%d max|diff|=%.3g\n        %s (row order = the"
                  " converter's tiled V-head order)"
                  % ("ok" if ok else "FAIL", gname, hname, n_rows, cols, worst, why))
            continue
        hv = read_values(hname)
        gv = rd_gguf_f32(gname, len(hv))
        if kind == "perm":
            hv = [hv[i] for i in perm]
        if plus1 is None:
            exp = [-math.exp(x) for x in hv]
        else:
            exp = [x + plus1 for x in hv]
        worst = max(abs(a - b) for a, b in zip(gv, exp))
        rel = max(abs(a - b) / max(1e-30, abs(b)) for a, b in zip(gv, exp))
        # A pure copy or a +1 is owed bit-exactness. A transcendental (exp) is not: the producer
        # evaluated it in some precision we do not control, so the honest band here is the residual
        # itself, printed rather than absorbed. PLAN.md step 1.4's "bit-exact, zero tolerance" on the
        # GDN control tensors is therefore achievable for dt_bias/alpha/beta and NOT for ssm_a.
        tol = 1e-3 if plus1 is None else 0.0
        ok = worst <= tol
        bad += 0 if ok else 1
        print("  %-5s %-30s vs %-52s n=%d max|diff|=%.3g max rel=%.3g (tol %g)%s\n        %s"
              % ("ok" if ok else "FAIL", gname, hname, len(hv), worst, rel, tol,
                 " (after the tiled V-head permutation)" if kind == "perm" else "", why))
        print("        gguf[:3] %s\n        exp [:3] %s" % (["%.6g" % x for x in gv[:3]],
                                                            ["%.6g" % x for x in exp[:3]]))
    print("\n%s (%d mismatches)" % ("every converter transform witnessed" if not bad
                                   else "TRANSFORM MISMATCHES — read the spec notes", bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--families", action="store_true")
    ap.add_argument("--per-token", action="store_true")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--values", nargs="+", metavar="NAME")
    ap.add_argument("--gguf-vs-ckpt", metavar="PATH",
                    help="witness the GGUF converter's value transforms against the checkpoint")
    ap.add_argument("--filter", default=None, help="substring filter for --header")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    did = False
    for name, fn in [("header", cmd_header), ("families", cmd_families), ("per_token", cmd_per_token),
                     ("state", cmd_state), ("mtp", cmd_mtp), ("check", cmd_check)]:
        if getattr(args, name, False):
            did = True
            rc = fn(args)
            if name == "check" and rc:
                return rc
    if args.values:
        did = True
        cmd_values(args)
    if args.gguf_vs_ckpt:
        did = True
        rc = cmd_gguf_vs_ckpt(args)
        if rc:
            return rc
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
