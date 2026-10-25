#!/usr/bin/env python3
"""bf16_census.py — the authoritative text-bf16 census, and the closures that keep it honest.

Three rulers once disagreed about "text weights, bf16", and the number is load-bearing (AGENTS.md
SCOPE, README.md, PLAN.md, DECISIONS.md D-01/D-11):

  tools/quant_scenarios.py --census      65.7198 GiB   a hand-written model of the architecture
  W7, quantization/residency-layout.md   66.1337 GiB   its own config-driven census
  the pinned GGUF's own records          64.5604 GiB   sum(elts) x 2 over 733 tensor records

The adjudication is quantization/evidence/bf16-census-adjudication-2026-10-26.md: 66.1337 GiB is the
text model *including* the MTP draft block, 64.5604 GiB is what the pinned artifact carries because
it has no draft block, and the 65.7198 GiB was a config model worth -0.4138 GiB of error. That tool
has since been rebuilt (commit a912ac9) to derive its census from the same files this one reads, so
its live answer now agrees with this one element for element. What is left of the dispute is not
forgotten: closure C4 keeps the retired model's defect decomposition alive against a frozen copy of
it, so a regression to the hand-written model is caught rather than remembered.

This tool does not pick the prettiest of the three. It builds five populations, each derived from a
file rather than typed, and then requires them to be equal where they ought to be equal:

  P_ARTIFACT  every tensor record in models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
              (733 names with real dims, read by tools/gguf_reader.py from the shipping header)
  P_INDEX     every name in checkpoints/Qwen3.6-35B-A3B.safetensors.index.json's weight_map,
              shaped from the config and *witnessed* against the safetensors headers of the four
              shards under models/qwen36-src/ (those headers carry real shapes and dtypes)
  P_MTP       the index names the artifact does not carry — derived from the name-set difference,
              not from a hard-coded list
  P_TOOL      quant_scenarios.census(), imported: the ruler this census was adjudicated against, now
              rebuilt from files; compared live, never retyped
  P_BROKEN    the retired hand-written census, frozen in quantization/evidence/
              bf16-census-broken-model-2026-10-26.json, because it no longer exists anywhere on disk

Closures enforced (each must close to 0 bytes, or this tool names the leftover):
  C1  P_ARTIFACT + P_MTP == P_INDEX restricted to text names   -> the artifact/text gap is the draft
  C2  sum(elts over all index names) x 2 == index total_size    -> nothing is missing from the census
  C3  per-name element equality for every HF name paired into the GGUF, an orphan check on both
      sides, and a half-shape check on the fused->split pair      -> no phantom, no double count
  C4a P_INDEX(text) - P_TOOL must be 0 elts on every text family -> the rebuilt tool has not drifted
  C4b P_INDEX(text) - P_BROKEN decomposed into named sub-terms   -> the retired model's
       -222,179,328-elt error stays attributed to terms *and witnessed non-zero*; C4 has to be a
       live comparison or a frozen one, and it is now both, because neither alone can fail
Policy arithmetic then recomputes the two numbers that decide D-01/D-11 — policy-A resident bytes and
per-token bytes — from P_ARTIFACT + P_MTP with widths read out of the pinned ggml header
(tools/ggml_block_geom.py, never retyped), and prints the same rows from P_TOOL so a number that
moved can be blamed on the census and not on the width rule.

CPU only: no device call, no timing, no ms figure. The GB/s figures used for ceilings are the ones
already in this repo (devices/measured-facts.md), used as divisors, not measured here.

Usage:
  python3 tools/bf16_census.py               # the adjudication report
  python3 tools/bf16_census.py --selftest    # injected faults, each one must fire
"""
import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_block_geom as GBG              # noqa: E402
import quant_scenarios as QS               # noqa: E402
from gguf_reader import Gguf               # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
CFG = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")
IDX = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json")
SHARD_DIR = os.path.join(REPO, "models/qwen36-src")
# The frozen retired census. C4b has nothing else to check the named-cause list against now that
# quant_scenarios.py derives its own census; if this file goes missing C4b cannot run, and the tool
# says so instead of printing a clean 0.
BROKEN_MODEL = os.path.join(REPO, "quantization/evidence/bf16-census-broken-model-2026-10-26.json")
GiB = 2.0 ** 30
READS = (237.0, 210.0, 190.0)              # GB/s already measured and recorded in this repo

# Quoted claims, for naming the drift. Not inputs to anything computed here. These are the figures the
# docs carried BEFORE the 2026-10-26 adjudication corrected them (README.md / AGENTS.md / PLAN.md /
# DECISIONS.md D-01/D-11 now carry 66.1337 / 19.7848 / 2.5252 / 35.16); they are kept because "how far
# did the correction move the answer" is a question a future reader has to ask, and the frozen answer is
# in the fixture alongside them.
CLAIM_DOCS_BF16 = 65.7198                   # the pre-correction text-bf16 figure (adjudication §1)
CLAIM_W7_BF16 = 66.1337                     # quantization/residency-layout.md (W7), now the authority
CLAIM_ARTIFACT_BF16 = 64.5604               # tmp/bf16_ruler.py (the supervisor's ruler); artifact only
CLAIM_A_RESIDENT = 19.59                    # policy A resident, pre-correction
CLAIM_A_PERTOKEN = 2.331                    # policy A per token, pre-correction
CLAIM_Q8_THROUGHOUT = 34.97                 # Q8-throughout resident, pre-correction

# Width *names* per policy class; the bits come from the pinned ggml header at run time.
# "exact_*" is deliberately two entries: the published artifact keeps router/norms/conv1d/A_log/
# dt_bias in F32, while quant_scenarios charges that bucket at bf16. Which one "exact" means is a
# width decision, so this tool prices both and chooses neither.
POLICIES = {
    "bf16": dict(experts="BF16", attn="BF16", shared="BF16", lm_head="BF16", embed="BF16",
                 exact="BF16"),
    "A": dict(experts="Q4_K", attn="Q8_0", shared="Q8_0", lm_head="Q8_0", embed="Q8_0",
              exact=None),        # None -> priced by the artifact's own stored type
    "F": dict(experts="Q8_0", attn="Q8_0", shared="Q8_0", lm_head="Q8_0", embed="Q8_0",
              exact=None),
    # The rest of quantization/README.md's table, so the whole policy table can be reprinted from the
    # corrected census instead of only the two rows this dispute turned on. embed follows lm_head's
    # width in E and G, which is quant_scenarios' own rule (BAM_BITS maps embed -> lm_head).
    "B": dict(experts="Q4_K", attn="BF16", shared="BF16", lm_head="BF16", embed="BF16",
              exact=None),
    "C": dict(experts="MXFP4", attn="Q8_0", shared="Q8_0", lm_head="Q8_0", embed="Q8_0",
              exact=None),
    "D": dict(experts="IQ4_XS", attn="Q8_0", shared="Q8_0", lm_head="Q8_0", embed="Q8_0",
              exact=None),
    "E": dict(experts="Q4_K", attn="Q4_K", shared="Q4_K", lm_head="Q6_K", embed="Q6_K",
              exact=None),
    "G": dict(experts="Q4_K", attn="Q8_0", shared="Q8_0", lm_head="Q6_K", embed="Q6_K",
              exact=None),
}
FAM_CLASS = {"routed_experts": "experts", "gdn_linear_attn": "attn", "full_attn": "attn",
             "shared_experts": "shared", "embed": "embed", "lm_head": "lm_head",
             "routers": "exact", "norms_misc": "exact", "vision": "exact"}

# HF name -> GGUF name(s). One table, read in both directions of the comparison, so the mapping
# cannot quietly drift from itself. `N` is the layer index; `mtp.*` names map to nothing because the
# published artifact carries no draft block — that absence is a finding, not a hole in the table.
MAP_HF_GGUF = {
    "model.language_model.embed_tokens.weight": ["token_embd.weight"],
    "model.language_model.norm.weight": ["output_norm.weight"],
    "lm_head.weight": ["output.weight"],
    "model.language_model.layers.N.input_layernorm.weight": ["blk.N.attn_norm.weight"],
    "model.language_model.layers.N.post_attention_layernorm.weight":
        ["blk.N.post_attention_norm.weight"],
    "model.language_model.layers.N.linear_attn.in_proj_qkv.weight": ["blk.N.attn_qkv.weight"],
    "model.language_model.layers.N.linear_attn.in_proj_z.weight": ["blk.N.attn_gate.weight"],
    "model.language_model.layers.N.linear_attn.in_proj_a.weight": ["blk.N.ssm_alpha.weight"],
    "model.language_model.layers.N.linear_attn.in_proj_b.weight": ["blk.N.ssm_beta.weight"],
    "model.language_model.layers.N.linear_attn.out_proj.weight": ["blk.N.ssm_out.weight"],
    "model.language_model.layers.N.linear_attn.conv1d.weight": ["blk.N.ssm_conv1d.weight"],
    "model.language_model.layers.N.linear_attn.A_log": ["blk.N.ssm_a"],
    "model.language_model.layers.N.linear_attn.dt_bias": ["blk.N.ssm_dt.bias"],
    "model.language_model.layers.N.linear_attn.norm.weight": ["blk.N.ssm_norm.weight"],
    "model.language_model.layers.N.self_attn.q_proj.weight": ["blk.N.attn_q.weight"],
    "model.language_model.layers.N.self_attn.k_proj.weight": ["blk.N.attn_k.weight"],
    "model.language_model.layers.N.self_attn.v_proj.weight": ["blk.N.attn_v.weight"],
    "model.language_model.layers.N.self_attn.o_proj.weight": ["blk.N.attn_output.weight"],
    "model.language_model.layers.N.self_attn.q_norm.weight": ["blk.N.attn_q_norm.weight"],
    "model.language_model.layers.N.self_attn.k_norm.weight": ["blk.N.attn_k_norm.weight"],
    # the only fused -> split pair: HF stores [E, 2I, H], GGUF stores two [E, I, H] matrices
    "model.language_model.layers.N.mlp.experts.gate_up_proj":
        ["blk.N.ffn_gate_exps.weight", "blk.N.ffn_up_exps.weight"],
    "model.language_model.layers.N.mlp.experts.down_proj": ["blk.N.ffn_down_exps.weight"],
    "model.language_model.layers.N.mlp.gate.weight": ["blk.N.ffn_gate_inp.weight"],
    "model.language_model.layers.N.mlp.shared_expert.gate_proj.weight":
        ["blk.N.ffn_gate_shexp.weight"],
    "model.language_model.layers.N.mlp.shared_expert.up_proj.weight":
        ["blk.N.ffn_up_shexp.weight"],
    "model.language_model.layers.N.mlp.shared_expert.down_proj.weight":
        ["blk.N.ffn_down_shexp.weight"],
    "model.language_model.layers.N.mlp.shared_expert_gate.weight":
        ["blk.N.ffn_gate_inp_shexp.weight"],
}


def strip_layer(name):
    """Fold every layer index to `N`, and fold `mtp.layers.k.` onto the ordinary layer pattern."""
    out = re.sub(r"^(model\.language_model\.layers\.)\d+\.", r"\g<1>N.", name)
    return re.sub(r"^mtp\.layers\.\d+\.", "model.language_model.layers.N.", out)


def geometry():
    t = json.load(open(CFG))["text_config"]
    return dict(h=t["hidden_size"], L=t["num_hidden_layers"], V=t["vocab_size"],
                hd=t["head_dim"], nq=t["num_attention_heads"], nkv=t["num_key_value_heads"],
                E=t["num_experts"], n_active=t["num_experts_per_tok"],
                me=t["moe_intermediate_size"], se=t["shared_expert_intermediate_size"],
                lk=t["linear_num_key_heads"], lv=t["linear_num_value_heads"],
                dk=t["linear_key_head_dim"], dv=t["linear_value_head_dim"],
                kf=t.get("linear_conv_kernel_dim", 4), layer_types=t["layer_types"],
                draft=t["mtp_num_hidden_layers"])


def shard_witnesses():
    """name -> (dtype, shape) read out of the safetensors headers on disk.

    Four of 26 shards were downloaded; a header is a few hundred KB and no weight bytes are touched.
    These are the only *real* shapes available offline, so every config-derived shape is checked
    against them where they overlap, and a name whose shape is not witnessed is printed as MODELLED.
    """
    files = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit("no safetensors shards under %s — nothing to witness shapes against"
                         % SHARD_DIR)
    out = {}
    for f in files:
        with open(os.path.join(SHARD_DIR, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        names = [k for k in hdr if k != "__metadata__"]
        assert names, "%s header parsed to zero tensor entries" % f
        for k in names:
            if k in out:
                raise AssertionError("tensor %s witnessed in two shards" % k)
            out[k] = (hdr[k]["dtype"], tuple(hdr[k]["shape"]))
    return out, files


_BROKEN_CACHE = {}


def broken_model():
    """The retired hand-written census, frozen — the only population C4b can decompose against.

    Loud by design. This fixture is the input that makes C4b a check rather than a formality, so an
    absent, unparseable or short file aborts the whole tool instead of letting C4 print a tidy 0 and a
    PASS nobody earned. It is reproducible from git (`tools/quant_scenarios.py::census()` at
    a912ac9^), which is the only reason it can be trusted now that the model that produced it is gone.
    """
    if "m" in _BROKEN_CACHE:
        return _BROKEN_CACHE["m"]
    rel = os.path.relpath(BROKEN_MODEL, REPO)
    try:
        raw = json.load(open(BROKEN_MODEL))
    except FileNotFoundError:
        raise SystemExit("the frozen retired census %s is absent, so closure C4b has nothing to "
                         "decompose against. Restore it from git; it is the only thing that witnesses "
                         "C4's named terms still do work." % rel)
    except ValueError as e:
        raise SystemExit("cannot parse the frozen retired census %s: %s" % (rel, e))
    fam = {k: int(v) for k, v in (raw.get("broken_census_elements_by_family") or {}).items()}
    if len(fam) != 8:
        raise SystemExit("%s carries %d families, C4b expects 8 — a short fixture checks "
                         "fewer families and still prints a clean total" % (rel, len(fam)))
    if [k for k, v in fam.items() if v <= 0]:
        raise SystemExit("%s has a non-positive family element count: %s" % (rel, fam))
    tot = int(raw.get("broken_census_total_elements") or 0)
    if sum(fam.values()) != tot:
        raise SystemExit("%s is internally inconsistent: families sum to %d, its own total says %d"
                         % (rel, sum(fam.values()), tot))
    terms = {}
    for row in raw.get("defect_decomposition_elements_authoritative_minus_broken") or []:
        terms[row["family"]] = terms.get(row["family"], 0) + int(row["elts"])
    fam_terms = {k: int(v) for k, v in
                 (raw.get("defect_decomposition_by_family_elements_authoritative_minus_broken")
                  or {}).items()}
    if not terms or fam_terms != terms:
        raise SystemExit("%s's per-term rows do not sum to its per-family decomposition (%s vs %s); a "
                         "fixture edited to float free of its own terms would agree with anything"
                         % (rel, sorted(fam_terms.items()), sorted(terms.items())))
    out = dict(fam=fam, terms=terms, total=tot,
               gib=float(raw.get("broken_census_total_bf16_gib_as_printed") or 0))
    _BROKEN_CACHE["m"] = out
    return out


def hf_shape(name, g):
    """Config-derived HF shape for any index name; None when this tool cannot model it.

    None is not silently skipped — build() aborts on it, because a census that drops what it cannot
    express reads low and still prints a total.
    """
    h, E, me, se = g["h"], g["E"], g["me"], g["se"]
    qkv = 2 * g["lk"] * g["dk"] + g["lv"] * g["dv"]
    if name.startswith("model.visual."):
        return None
    if name == "model.language_model.embed_tokens.weight" or name == "lm_head.weight":
        return (g["V"], h)
    if name == "model.language_model.norm.weight":
        return (h,)
    if name == "mtp.fc.weight":
        return (h, 2 * h)
    if name in ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                "mtp.pre_fc_norm_hidden.weight"):
        return (h,)
    s = strip_layer(name)
    kind = s.split(".layers.N.", 1)[1] if ".layers.N." in s else name
    if kind in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return (h,)
    if kind == "linear_attn.in_proj_qkv.weight":
        return (qkv, h)
    if kind == "linear_attn.in_proj_z.weight":
        return (g["lv"] * g["dv"], h)
    if kind in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
        return (g["lv"], h)
    if kind == "linear_attn.out_proj.weight":
        return (h, g["lv"] * g["dv"])
    if kind == "linear_attn.conv1d.weight":
        return (qkv, 1, g["kf"])
    if kind in ("linear_attn.A_log", "linear_attn.dt_bias"):
        return (g["lv"],)
    if kind == "linear_attn.norm.weight":
        return (g["dk"],)
    if kind == "self_attn.q_proj.weight":
        return (g["nq"] * g["hd"] * 2, h)          # attn_output_gate doubles q
    if kind in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        return (g["nkv"] * g["hd"], h)
    if kind == "self_attn.o_proj.weight":
        return (h, g["nq"] * g["hd"])
    if kind in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        return (g["hd"],)
    if kind == "mlp.experts.gate_up_proj":
        return (E, 2 * me, h)
    if kind == "mlp.experts.down_proj":
        return (E, h, me)
    if kind == "mlp.gate.weight":
        return (E, h)
    if kind in ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight"):
        return (se, h)
    if kind == "mlp.shared_expert.down_proj.weight":
        return (h, se)
    if kind == "mlp.shared_expert_gate.weight":
        return (1, h)
    return None


def family_of(name):
    """Bucket names mirroring quant_scenarios' families, so C4 can blame a specific term.

    Three conventions are inherited from quant_scenarios and stated rather than hidden:
    `self_attn.q_norm`/`k_norm` ride in `full_attn`, `linear_attn.norm` rides in `gdn_linear_attn`,
    and `mtp.fc` — a projection, not a norm — rides in `norms_misc`, which is the tool's own bucket
    name for "norms and misc".
    """
    if name.startswith("model.visual."):
        return "vision"
    if name in ("model.language_model.embed_tokens.weight",):
        return "embed"
    if name == "lm_head.weight":
        return "lm_head"
    s = strip_layer(name)
    kind = s.split(".layers.N.", 1)[1] if ".layers.N." in s else name
    if kind.startswith("mlp.experts."):
        return "routed_experts"
    if kind.startswith("linear_attn."):
        return "gdn_linear_attn"
    if kind.startswith("self_attn."):
        return "full_attn"
    if kind.startswith("mlp.shared_expert"):
        return "shared_experts"
    if kind == "mlp.gate.weight":
        return "routers"
    return "norms_misc"


def build(faults=()):
    """All four populations plus the C3 pairing. `faults` is a set of injected-fault names."""
    faults = set(faults)
    g = geometry()
    idx = json.load(open(IDX))
    wm = idx["weight_map"]
    index_total = int(idx["metadata"]["total_size"])
    wit, shard_files = shard_witnesses()

    # ---- P_INDEX: every index name; config-shaped, witnessed where a shard states the shape ----
    pop_index, shape_checks, unmodelled, mismatch = {}, 0, [], []
    for name in wm:
        sh = wit[name][1] if name in wit else hf_shape(name, g)
        if sh is None:
            unmodelled.append(name)
            continue
        if name in wit:
            shape_checks += 1
            modelled = hf_shape(name, g)
            if modelled is not None and modelled != sh:
                mismatch.append((name, sh, modelled))
        pop_index[name] = {"elts": 1, "shape": sh, "fam": family_of(name),
                           "witnessed": name in wit}
        for d in sh:
            pop_index[name]["elts"] *= d
    if mismatch:
        raise AssertionError("config-derived shapes disagree with real shard headers:\n  "
                             + "\n  ".join("%s: shard %s vs model %s" % m for m in mismatch))
    if unmodelled:
        raise AssertionError("cannot derive a shape for %d index names (first: %s) — a census that "
                             "silently drops what it cannot model reads low and still prints a total"
                             % (len(unmodelled), unmodelled[0]))

    # ---- P_ARTIFACT: the shipping file's own records, count asserted against its header --------
    gg = Gguf(GGUF)
    pop_gguf = {t.name: {"elts": t.n_elts, "shape": t.shape_hf, "type": t.type_name}
                for t in gg.tensors}

    def count_check(pop):
        assert len(pop) == gg.n_tensors, \
            "parsed %d GGUF tensor records, the header declares %d" % (len(pop), gg.n_tensors)
    count_check(pop_gguf)
    if "short-record" in faults:
        pop_gguf.pop(sorted(pop_gguf)[-1])
        count_check(pop_gguf)
    if "swap-half-shape" in faults:
        # The quiet version of a mis-pair: two expert tensors with *identical element counts* and
        # different shapes. Swapping their shapes is invisible to any element-sum check, so the only
        # thing that can catch it is the half-shape check — which is therefore witnessed alone.
        a, b = "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"
        assert pop_gguf[a]["elts"] == pop_gguf[b]["elts"], \
            "the swap fault needs two equal-elt tensors to be a fair test"
        pop_gguf[a]["shape"], pop_gguf[b]["shape"] = pop_gguf[b]["shape"], pop_gguf[a]["shape"]

    # ---- C3: pair every text name into the artifact, both directions ---------------------------
    pairs, claimer, unmatched_hf = [], {}, []
    for name in sorted(pop_index):
        if name.startswith("model.visual."):
            continue
        base = strip_layer(name)
        glist = [] if name.startswith("mtp.") else MAP_HF_GGUF.get(base, [])
        if "mispair" in faults and base.endswith("mlp.experts.gate_up_proj"):
            glist = ["blk.N.ffn_gate_exps.weight", "blk.N.ffn_down_exps.weight"]
        li = re.match(r"^(?:model\.language_model|mtp)\.layers\.(\d+)\.", name)
        if glist:
            if "N" in "".join(glist):
                if li is None:
                    raise AssertionError("mapping for %s wants a layer index this name does not "
                                         "carry" % name)
                glist = [x.replace("N", li.group(1)) for x in glist]
        if not glist:
            unmatched_hf.append(name)
            continue
        for gn in glist:
            if gn not in pop_gguf:
                raise AssertionError("mapping pairs %s -> %s, but the artifact carries no %s"
                                     % (name, gn, gn))
            if gn in claimer:
                raise AssertionError("GGUF record %s is claimed by both %s and %s — one record "
                                     "counted twice is how a census reads high"
                                     % (gn, claimer[gn], name))
            claimer[gn] = name
        pairs.append((name, glist))
    orphans = sorted(set(pop_gguf) - set(claimer))
    if orphans:
        raise AssertionError("%d artifact records are claimed by no HF index name (first: %s) — "
                             "either the map is incomplete or the artifact carries a tensor the "
                             "text model does not have" % (len(orphans), orphans[0]))
    bad = []
    for name, glist in pairs:
        if len(glist) == 2:                     # fused -> split: each half must be a real half
            e, two_i, h = pop_index[name]["shape"]
            want = (e, two_i // 2, h)
            for gn in glist:
                if pop_gguf[gn]["shape"] != want:
                    bad.append("%s: split half %s is %s, expected %s"
                               % (name, gn, pop_gguf[gn]["shape"], want))
        tot = sum(pop_gguf[gn]["elts"] for gn in glist)
        if tot != pop_index[name]["elts"]:
            bad.append("%s: index %d elts vs artifact %d (%s)"
                       % (name, pop_index[name]["elts"], tot, "+".join(glist)))
    if bad:
        raise AssertionError("name pairing does not preserve element counts or shapes:\n  "
                             + "\n  ".join(bad))

    # the pairing must also agree with the config's own layer_types, or a GDN layer is being compared
    # against a full-attention tensor and the element equality above would pass only by luck
    for i, lt in enumerate(g["layer_types"]):
        has_q = "blk.%d.attn_q.weight" % i in pop_gguf
        has_qkv = "blk.%d.attn_qkv.weight" % i in pop_gguf
        if has_q == has_qkv:
            raise AssertionError("blk.%d carries %s" % (i, "both attn_q and attn_qkv" if has_q
                                                        else "neither attn_q nor attn_qkv"))
        want = "full_attention" if has_q else "linear_attention"
        if want != lt:
            raise AssertionError("layer %d: config layer_types says %s, the artifact carries %s"
                                 % (i, lt, want))

    # ---- family sums, one per population; the artifact's families come from the claimer --------
    def famsum(pairs_or_pop, kind):
        out = {}
        if kind == "index":
            for n, d in pairs_or_pop.items():
                out[d["fam"]] = out.get(d["fam"], 0) + d["elts"]
        elif kind == "artifact":
            for gn, hf in claimer.items():
                fam = pop_index[hf]["fam"]
                out[fam] = out.get(fam, 0) + pop_gguf[gn]["elts"]
        else:
            for n in pairs_or_pop:
                fam = pop_index[n]["fam"]
                out[fam] = out.get(fam, 0) + pop_index[n]["elts"]
        return out

    fam_index = famsum(pop_index, "index")
    fam_art = famsum(claimer, "artifact")
    fam_mtp = famsum(unmatched_hf, "mtp")
    if "drop-family" in faults:
        nlin = sum(1 for x in g["layer_types"] if x == "linear_attention")
        fam_art["gdn_linear_attn"] -= nlin * g["lv"] * g["dv"] * g["h"]
    tool_bytes, _ = QS.census()                      # the ruler, imported — never retyped
    tool_fam = {k: v // 2 for k, v in tool_bytes.items()}
    if "model-regression" in faults:
        # The regression C4 was written to catch: the live tool population replaced by the retired
        # model's, i.e. what quant_scenarios would print if the rebuild were reverted. Nothing is
        # typed here — it is the same frozen population C4b decomposes against.
        tool_fam = dict(broken_model()["fam"])
    if "phantom-term" in faults:
        fam_index["norms_misc"] += 9 * g["h"] * g["h"]

    return dict(g=g, index=pop_index, gguf=pop_gguf, claimer=claimer, mtp=sorted(unmatched_hf),
                fam_index=fam_index, fam_art=fam_art, fam_mtp=fam_mtp, tool_fam=tool_fam,
                pairs=len(pairs), n_split=sum(1 for _, gl in pairs if len(gl) == 2),
                shape_checks=shape_checks, shards=shard_files,
                index_total=index_total, n_index=len(pop_index), n_wit=len(wit),
                decl_tensors=gg.n_tensors)


def closures(p):
    """The four closures. Returns (report lines, residuals in bytes, derived numbers)."""
    g = p["g"]
    text_index = sum(v for k, v in p["fam_index"].items() if k != "vision")
    art = sum(p["fam_art"].values())
    mtp = sum(p["fam_mtp"].values())
    all_index = sum(p["fam_index"].values())
    lines, res = [], {}

    d1 = (text_index - art - mtp) * 2
    lines.append("C1  text from index %d elts = artifact %d + draft block %d   residue %d B"
                 % (text_index, art, mtp, d1))
    if d1:
        res["C1 text-vs-artifact+draft"] = d1
    d2 = all_index * 2 - p["index_total"]
    lines.append("C2  all %d index names, sum(elts) x 2 = %d B vs index metadata total_size %d B   "
                 "residue %d B" % (p["n_index"], all_index * 2, p["index_total"], d2))
    if d2:
        res["C2 index total_size"] = d2
    lines.append("C3  %d text names paired into %d artifact records; 0 orphans on either side; "
                 "%d of %d shapes witnessed by real shard headers"
                 % (p["pairs"], len(p["gguf"]), p["shape_checks"], p["n_index"]))

    # ---- C4: quant_scenarios' census, in two arms ------------------------------------------------
    # C4 used to be one closure: the tool's hand-written config model differed from this census by
    # +222,179,328 elts and the five named terms below accounted for every one of them. The rebuild
    # (commit a912ac9) made quant_scenarios derive from the same files this census derives from, so the
    # live delta is 0 on every family and a decomposition of *it* would be a decomposition of nothing --
    # the exact state in which a closure stops being able to fail. So C4 is two arms:
    #   C4a  the live tool population must equal this census, 0 elts on every text family;
    #   C4b  the named-cause list must reproduce the frozen broken census, element for element, so the
    #        decomposition stays witnessed non-zero against
    #        quantization/evidence/bf16-census-broken-model-2026-10-26.json.
    # Neither arm can pass by being empty: C4a fires on any live drift, C4b fires if a term stops
    # carrying its magnitude, or if the fixture is edited to fit the terms.
    h, L = g["h"], g["L"]
    nlin = sum(1 for x in g["layer_types"] if x == "linear_attention")
    draft = g["draft"]
    real_norm_h = 2 * L + 1 + draft * 5          # in+post per layer, final norm, draft's 5 h-norms
    terms = [
        ("gdn out_proj / ssm_out [h, lv*dv] x %d — a real tensor the retired model did not model"
         % nlin, "gdn_linear_attn", nlin * g["lv"] * g["dv"] * g["h"]),
        ("the retired gdn term's `+ 2*h` per linear layer — no tensor behind it",
         "gdn_linear_attn", -2 * h * nlin),
        ("norms_misc's `3*h*(3*h)` = 9h^2 — no tensor family of this size exists",
         "norms_misc", -9 * h * h),
        ("norms: real %d h-sized norms vs the retired model's 2L+4 = %d (x h elements each)"
         % (real_norm_h, 2 * L + 4),
         "norms_misc", (real_norm_h - (2 * L + 4)) * h),
        ("mtp.fc [h, 2h] — the draft's fusion projection, absent from the retired model",
         "norms_misc", h * 2 * h * draft),
    ]
    named = {}
    for _, fam, d in terms:
        named[fam] = named.get(fam, 0) + d
    broken = broken_model()
    fam_broken = broken["fam"]

    lines.append("C4a live per-family delta (authoritative - quant_scenarios.census()), elements; "
                 "every text family must read 0:")
    unexplained_live, fam_delta = 0, {}
    for fam in sorted(set(p["fam_index"]) | set(p["tool_fam"])):
        if fam == "vision":
            continue
        d = p["fam_index"].get(fam, 0) - p["tool_fam"].get(fam, 0)
        fam_delta[fam] = d
        unexplained_live += d
        lines.append("      %-18s %14d vs %14d  delta %+12d  unexplained %+d"
                     % (fam, p["fam_index"].get(fam, 0), p["tool_fam"].get(fam, 0), d, d))
    live_total = sum(fam_delta.values())
    lines.append("      total live delta %+d elts = %+.4f GiB across %d text families "
                 "(0 is the rebuilt tool agreeing with this census to the element)"
                 % (live_total, live_total * 2 / GiB, len(fam_delta)))
    if unexplained_live:
        res["C4 quant_scenarios delta"] = unexplained_live * 2

    lines.append("C4b frozen retired model (%.4f GiB, %s), per-family delta "
                 "(authoritative - fixture), elements:" % (broken["gib"],
                                                           os.path.relpath(BROKEN_MODEL, REPO)))
    unexplained_fx, fam_fx_delta = 0, {}
    for fam in sorted(fam_broken):
        d = p["fam_index"].get(fam, 0) - fam_broken[fam]
        fam_fx_delta[fam] = d
        left = d - named.get(fam, 0)
        unexplained_fx += left
        lines.append("      %-18s %14d vs %14d  delta %+12d  named %+12d  unexplained %+d"
                     % (fam, p["fam_index"].get(fam, 0), fam_broken[fam], d, named.get(fam, 0), left))
    for label, _, d in terms:
        lines.append("      term %+12d elts  %s" % (d, label))
    fx_total = sum(fam_fx_delta.values())
    lines.append("      total broken-model delta %+d elts = %+.4f GiB; unexplained %+d elts"
                 % (fx_total, fx_total * 2 / GiB, unexplained_fx))
    if unexplained_fx:
        res["C4 broken-model decomposition"] = unexplained_fx * 2
    # The fixture carries its own frozen copy of the decomposition. The terms above are re-derived from
    # the checkpoint's geometry, so the two agreeing is a real cross-check: it is what stops a term
    # being quietly retuned until the fixture balanced.
    drift = {f: named.get(f, 0) - broken["terms"].get(f, 0)
             for f in set(named) | set(broken["terms"])
             if named.get(f, 0) != broken["terms"].get(f, 0)}
    lines.append("      derived terms vs the fixture's own frozen decomposition: %s"
                 % ("agree on every family" if not drift else "DISAGREE %s" % sorted(drift.items())))
    if drift:
        res["C4 broken-model terms disagree"] = sum(drift.values()) * 2
    return lines, res, dict(text_index=text_index, art=art, mtp=mtp, all_index=all_index,
                            terms=terms, fam_delta=fam_delta, live_total=live_total,
                            fam_fx_delta=fam_fx_delta, fx_total=fx_total,
                            broken_total=broken["total"], broken_gib=broken["gib"],
                            vision=p["fam_index"].get("vision", 0))


# ---------------------------------------------------------------------------------------------
# policy arithmetic
# ---------------------------------------------------------------------------------------------
def hf_kind(name):
    s = strip_layer(name)
    return s.split(".layers.N.", 1)[-1] if ".layers.N." in s else name


def f32_kinds(p):
    """Which HF name-kinds the published artifact stores as F32 — read off the artifact, not typed.

    This is what makes the `exact` bucket two-sided without anyone guessing: whatever llama.cpp kept
    unquantized in the shipping file is what the F32 charging variant prices at 4 bytes.
    """
    out = set()
    for gn, hf in p["claimer"].items():
        if p["gguf"][gn]["type"] in ("F32", "F16"):
            out.add(hf_kind(hf))
    return out


def width_for(p, polkey, exact_mode, fam, hf_name=None, art_type=None):
    spec = POLICIES[polkey]
    cls = FAM_CLASS[fam]
    tname = spec[cls]
    if tname is not None:
        return tname
    if exact_mode == "artifact":
        if art_type in ("F32", "F16"):
            return art_type
        if hf_name is not None and hf_kind(hf_name) in f32_kinds(p):
            return "F32"
    return "BF16"


def policy_rows(p, polkey, exact_mode="bf16"):
    """(main rows, draft rows) as {(family, ggml type): bytes}, priced by ggml struct sizes."""
    rows, draft = {}, {}
    for gn, hf in p["claimer"].items():
        fam = p["index"][hf]["fam"]
        t = width_for(p, polkey, exact_mode, fam, hf_name=hf, art_type=p["gguf"][gn]["type"])
        rows[(fam, t)] = rows.get((fam, t), 0) + GBG.row_bytes(p["gguf"][gn]["elts"], t)
    for name in p["mtp"]:
        fam = p["index"][name]["fam"]
        t = width_for(p, polkey, exact_mode, fam, hf_name=name)
        draft[(fam, t)] = draft.get((fam, t), 0) + GBG.row_bytes(p["index"][name]["elts"], t)
    return rows, draft


def per_token(p, rows, draft_rows, with_draft=True):
    """Bytes read per generated token. The rule is quant_scenarios' own, restated so it can be argued:

    routed experts are read only for the `n_active` of `E` that fire; the token embedding is one row
    of the vocab table, not the table; `lm_head` is the whole untied table; every other family is
    read in full. `with_draft` decides whether the MTP block is walked once per token — a draft-depth
    question this tool does not settle, so both numbers are printed.
    """
    g = p["g"]
    tot = 0
    pool = list(rows.items()) + (list(draft_rows.items()) if with_draft else [])
    for (fam, t), b in pool:
        if fam == "routed_experts":
            tot += b * g["n_active"] / g["E"]
        elif fam == "embed":
            tot += GBG.row_bytes(g["h"], t)
        else:
            tot += b
    return tot


def tool_policy_rows(p, polkey):
    """The same policy priced from quant_scenarios' census, as a control on the width rule.

    If policy A computed here from the tool's census lands on the published 19.59 / 2.331, then any
    difference in the artifact-census row is the census, not the width arithmetic.
    """
    spec = POLICIES[polkey]
    bpw = GBG.geom()
    fam_bits = {fam: bpw[spec[FAM_CLASS[fam]]]["bpw"] if spec[FAM_CLASS[fam]] is not None
                else 16.0 for fam in p["tool_fam"]}
    resident = sum(p["tool_fam"][f] * fam_bits[f] / 8.0 for f in p["tool_fam"])
    per = 0.0
    for f, elts in p["tool_fam"].items():
        if f == "routed_experts":
            per += elts * p["g"]["n_active"] / p["g"]["E"] * fam_bits[f] / 8.0
        elif f == "embed":
            per += 0.0
        else:
            per += elts * fam_bits[f] / 8.0
    return resident, per


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def report():
    p = build()
    g = p["g"]
    lines, res, sc = closures(p)
    ti, art, mtp = sc["text_index"], sc["art"], sc["mtp"]
    print("bf16_census — three rulers, five populations, five closures")
    print("sources: %s (%d weight_map names, total_size %d B); %d shards under %s (%d tensors of "
          "real shape/dtype); %s (%d records)"
          % (os.path.relpath(IDX, REPO), p["n_index"], p["index_total"], len(p["shards"]),
             os.path.relpath(SHARD_DIR, REPO), p["n_wit"], os.path.relpath(GGUF, REPO),
             p["decl_tensors"]))
    print("every count below is derived from those files; none of them is typed into this tool.\n")

    print("== 1. what each ruler measures ==")
    n_text_idx = sum(1 for n in p["index"] if not n.startswith("model.visual."))
    draft_claimed = sum(1 for hf in p["claimer"].values() if hf.startswith("mtp."))
    print("  P_ARTIFACT  %10.4f GiB  %d records, blocks 0..%d; %d of them claimed by a draft-block "
          "name (every record is accounted for, so 0 here means the artifact carries no draft block)"
          % (art * 2 / GiB, p["decl_tensors"], g["L"] - 1, draft_claimed))
    print("  P_INDEX text %9.4f GiB  %d names = %d body + %d draft-block (mtp.*)"
          % (ti * 2 / GiB, n_text_idx, n_text_idx - len(p["mtp"]), len(p["mtp"])))
    print("  W7 claim      %10.4f GiB  quantization/residency-layout.md — confirmed as the authority"
          % CLAIM_W7_BF16)
    print("  P_TOOL        %10.4f GiB  quant_scenarios.census(), imported (rebuilt 2026-10-26 to "
          "derive from files; C4a requires this line and the one above to agree to the element)"
          % (sum(p["tool_fam"].values()) * 2 / GiB))
    print("  retired model %10.4f GiB  the frozen hand-written census, quantization/evidence/"
          "bf16-census-broken-model-2026-10-26.json (C4b's input)" % sc["broken_gib"])
    print("  docs carried  %10.4f GiB  what the docs quoted before the correction (adjudication §1); "
          "they now carry the line above" % CLAIM_DOCS_BF16)
    print("  artifact ruler%9.4f GiB  tmp/bf16_ruler.py" % CLAIM_ARTIFACT_BF16)
    print("  index total   %10.4f GiB  all %d names: text + draft + vision (%d names, %.4f GiB)"
          % (p["index_total"] / GiB, p["n_index"],
             sum(1 for n in p["index"] if n.startswith("model.visual.")), sc["vision"] * 2 / GiB))

    print("\n== 1b. the authoritative text census by family, bf16 ==")
    print("  name arithmetic, all derived: %d artifact records = %d paired text names + %d extra "
          "halves from the one fused->split pair per layer. The index's %d text names = %d body + "
          "%d draft names. %d - %d = %d, which is the 'the two name lists differ by 21' the dispute "
          "quotes, and it is entirely the fused pair plus the absent draft block."
          % (p["decl_tensors"], p["pairs"], p["n_split"], n_text_idx, n_text_idx - len(p["mtp"]),
             len(p["mtp"]), p["decl_tensors"], n_text_idx, p["decl_tensors"] - n_text_idx))
    for fam, elts in sorted(((k, v) for k, v in p["fam_index"].items() if k != "vision"),
                            key=lambda kv: -kv[1]):
        print("  %-18s %16d elts  %8.4f GiB  %5.2f %%" % (fam, elts, elts * 2 / GiB,
                                                         100 * elts / ti))

    print("\n== 2. closures ==")
    for ln in lines:
        print("  " + ln)
    print("  residuals: %s" % ("NONE — every closure closed to 0 bytes" if not res else
                               "; ".join("%s: %d B" % kv for kv in sorted(res.items()))))

    print("\n== 3. the draft block, which is the whole artifact-vs-text gap ==")
    print("  %d names, %d elts = %.4f GiB bf16; %d of them shape-witnessed by the shard headers"
          % (len(p["mtp"]), mtp, mtp * 2 / GiB, sum(1 for n in p["mtp"] if p["index"][n]["witnessed"])))
    for n in p["mtp"]:
        print("      %-52s %-18s %12d elts  %s"
              % (n, "x".join(map(str, p["index"][n]["shape"])), p["index"][n]["elts"],
                 "witnessed" if p["index"][n]["witnessed"] else "MODELLED"))

    print("\n== 4. verdict on tools/quant_scenarios.py, family by family ==")
    for fam in ("routed_experts", "gdn_linear_attn", "full_attn", "shared_experts", "routers",
                "embed", "lm_head", "norms_misc"):
        d = p["fam_index"].get(fam, 0) - p["tool_fam"].get(fam, 0)
        print("  %-18s tool %14d  real %14d  delta %+12d elts (%+.5f GiB bf16)"
              % (fam, p["tool_fam"].get(fam, 0), p["fam_index"].get(fam, 0), d, d * 2 / GiB))
    print("  the live per-family delta in C4a is 0 everywhere: the rebuilt tool and this census read")
    print("  the same files. The named causes below are the retired model's error, which C4b keeps")
    print("  witnessed non-zero against the frozen fixture — their sum is %+.4f GiB and it is not 0,"
          % (sc["fx_total"] * 2 / GiB))
    print("  because a decomposition of a 0 delta would be a decomposition of nothing.")
    for label, fam, d in sc["terms"]:
        print("      %+12d elts  %s" % (d, label))

    print("\n== 5. the two deciding numbers, recomputed from the artifact census ==")
    ctrl_res, ctrl_per = tool_policy_rows(p, "A")
    print("  control: policy A priced from quant_scenarios' own census with this tool's width rule -> "
          "%.2f GiB resident, %.3f GiB/token. The hand-written model published %.2f / %.3f and the two "
          "rulers now derive the same census, so any difference left between this line and the rows "
          "below is the width rule and nothing else."
          % (ctrl_res / GiB, ctrl_per / GiB, CLAIM_A_RESIDENT, CLAIM_A_PERTOKEN))
    out = {}
    for polkey, modes in (("bf16", ("bf16",)), ("A", ("bf16", "artifact")),
                          ("B", ("bf16",)), ("C", ("bf16",)), ("D", ("bf16",)), ("E", ("bf16",)),
                          ("F", ("bf16",)), ("G", ("bf16",))):
        for mode in modes:
            rows, draft = policy_rows(p, polkey, mode)
            main_b, dft_b = sum(rows.values()), sum(draft.values())
            pt_w = per_token(p, rows, draft, True)
            pt_o = per_token(p, rows, draft, False)
            out[(polkey, mode)] = (main_b, dft_b, pt_w, pt_o)
            print("  policy %-4s exact-as-%-8s resident %8.4f GiB (main %8.4f + draft %6.4f)   "
                  "per-token with-draft %.4f  target-only %.4f GiB"
                  % (polkey, mode, (main_b + dft_b) / GiB, main_b / GiB, dft_b / GiB,
                     pt_w / GiB, pt_o / GiB))
    print("  ceilings are bytes divided by the GB/s figures already in devices/measured-facts.md;")
    print("  they are ceilings, and no timing was taken here:")
    hdr = "  %-30s" % "row" + "".join("%9s" % ("%d GB/s" % r) for r in READS)
    print(hdr)
    for (polkey, mode), (mb, db, ptw, pto) in out.items():
        print("  %-30s" % ("policy %s/%s with-draft" % (polkey, mode))
              + "".join("%9.0f" % (r * 1e9 / ptw) for r in READS))
        print("  %-30s" % ("policy %s/%s target-only" % (polkey, mode))
              + "".join("%9.0f" % (r * 1e9 / pto) for r in READS))
    a = out[("A", "bf16")]
    print("  vs the policy-A row the docs carried before the correction (%.2f resident / %.3f per "
          "token): resident %+.4f GiB (%+.2f %%), per-token %+.4f GiB (%+.1f %%) — that is the whole "
          "cost of the four census defects, and the docs now carry these figures"
          % (CLAIM_A_RESIDENT, CLAIM_A_PERTOKEN,
             (a[0] + a[1]) / GiB - CLAIM_A_RESIDENT, 100 * ((a[0] + a[1]) / GiB / CLAIM_A_RESIDENT
                                                            - 1), a[2] / GiB - CLAIM_A_PERTOKEN,
             100 * (a[2] / GiB / CLAIM_A_PERTOKEN - 1)))
    print("  vs the pre-correction Q8-throughout row (%.2f GiB): %+.4f GiB" %
          (CLAIM_Q8_THROUGHOUT, (out[("F", "bf16")][0] + out[("F", "bf16")][1]) / GiB
           - CLAIM_Q8_THROUGHOUT))

    print("\n== 6. consequence for quantization/residency-layout.md ==")
    # Constants quoted from residency-layout.md §1/§3 so the sensitivity is arithmetic against the
    # file being adjudicated, not a fresh model of it.
    W7_WEIGHTS, W7_DRAFT = 19.3170, 0.4686
    W7_FIXED_4LANE, W7_B_PER_TOKEN, W7_TOKENS = 20.027, 90112, 140755
    W7_BUDGET, W7_HEADROOM_4K, W7_HEADROOM_256K = 31.84, 11.907, 6.493
    tok_per_gib = GiB / W7_B_PER_TOKEN
    print("  witness on the quoted constants: (budget %.2f - fixed %.3f) GiB / %d B per token = "
          "%.0f tokens, where residency-layout.md prints %d — the row is reproduced before it is "
          "corrected" % (W7_BUDGET, W7_FIXED_4LANE, W7_B_PER_TOKEN,
                         (W7_BUDGET - W7_FIXED_4LANE) * GiB / W7_B_PER_TOKEN, W7_TOKENS))
    for mode in ("bf16", "artifact"):
        mb, db = out[("A", mode)][0] / GiB, out[("A", mode)][1] / GiB
        d_fixed = (mb - W7_WEIGHTS) + (db - W7_DRAFT)
        print("  exact-as-%-8s weights+draft row %8.4f GiB (W7 %.4f, bytes used %+.4f); headroom "
              "4k %.3f (W7 %.3f), 256k %.3f (W7 %.3f); crossover %+.0f tokens -> %d (W7 %d). "
              "1 GiB of fixed bytes = %.0f tokens."
              % (mode, mb + db, W7_WEIGHTS + W7_DRAFT, d_fixed,
                 W7_HEADROOM_4K - d_fixed, W7_HEADROOM_4K, W7_HEADROOM_256K - d_fixed,
                 W7_HEADROOM_256K, -d_fixed * tok_per_gib, W7_TOKENS - d_fixed * tok_per_gib,
                 W7_TOKENS, tok_per_gib))
    return 0


# ---------------------------------------------------------------------------------------------
# selftest: injected faults, each one required to fire
# ---------------------------------------------------------------------------------------------
def selftest():
    print("bf16_census --selftest")
    fired, missed, clean = [], [], []
    injected = [0]

    def expect(name, fn, why):
        """A fault counts as fired if the tool raises, or if fn() returns a detection note.

        Two shapes because two faults are structurally impossible to mis-read (a short parse, a
        double claim) while the others are only visible as a non-zero residual: raising is evidence,
        and so is a closure that came back with a named leftover. `injected` is counted here rather
        than derived from len(fired)+len(missed), because a failed clean-run condition is not an
        injected fault and must not be printed as one.
        """
        injected[0] += 1
        try:
            note = fn()
        except (AssertionError, ValueError) as e:
            fired.append("%s fired: %s" % (name, str(e).splitlines()[0][:140]))
            return
        if note:
            fired.append("%s fired: %s" % (name, note))
        else:
            missed.append("%s did NOT fire — %s" % (name, why))

    # 0. the clean run must be a genuine zero: closures that cannot fail cannot pass
    p = build()
    lines, res, sc = closures(p)
    if res:
        clean.append("clean run left residuals: %s" % res)
    if sc["mtp"] == 0:
        clean.append("clean run is vacuous: the draft block reads 0 elements")
    if sc["vision"] == 0:
        clean.append("clean run is vacuous: the vision family read 0")
    # C4a's clean value is 0 by construction now, so non-vacuity has to come from the other arm: the
    # frozen broken model must still carry a non-zero defect signature, and no named term may have
    # gone quiet. Without these two the C4 block could print all-zeros forever and still say PASS.
    if sc["fx_total"] == 0 or sc["fam_fx_delta"].get("gdn_linear_attn", 0) == 0:
        clean.append("clean run is vacuous: the frozen broken model's defect signature reads 0, so C4b "
                     "is decomposing nothing (fixture total %d elts)" % sc["broken_total"])
    if not sc["terms"] or [t for t in sc["terms"] if t[2] == 0]:
        clean.append("a C4 named term is missing or reads 0 — a term that reads 0 checks nothing")
    print("  clean run: text %.4f GiB, artifact %.4f GiB, draft %.4f GiB, vision %.4f GiB; "
          "C4a live delta %+d elts (must be 0), C4b frozen-model delta %+d elts (must NOT be 0)"
          % (sc["text_index"] * 2 / GiB, sc["art"] * 2 / GiB, sc["mtp"] * 2 / GiB,
             sc["vision"] * 2 / GiB, sc["live_total"], sc["fx_total"]))
    print("  witnesses that a 0 there would have meant something: %d names paired, %d shapes witnessed "
          "against real headers, %d draft names, %d named C4 terms worth %+.4f GiB against the frozen "
          "model (gdn %+d elts, norms %+d elts)"
          % (p["pairs"], p["shape_checks"], len(p["mtp"]), len(sc["terms"]),
             sc["fx_total"] * 2 / GiB, sc["fam_fx_delta"].get("gdn_linear_attn", 0),
             sc["fam_fx_delta"].get("norms_misc", 0)))
    for ln in clean:
        print("  FAIL " + ln)

    # 1. a dropped family: GDN out_proj removed from the artifact census
    def drop_family():
        pf = build(faults=("drop-family",))
        _, r, s = closures(pf)
        return ("C1 residual %+d B — a whole family missing from one population does not balance"
                % r["C1 text-vs-artifact+draft"]) if "C1 text-vs-artifact+draft" in r else None
    expect("dropped-family", drop_family, "C1 reported no residual for a missing family")

    # 2. a phantom term of the shape the retired model carried: 3h x 3h added to norms_misc. It now
    #    breaks both C4 arms at once — the live census no longer balances, and neither does the
    #    decomposition against the frozen model — so the note names both.
    def phantom():
        pf = build(faults=("phantom-term",))
        _, r, s = closures(pf)
        if "C4 quant_scenarios delta" not in r:
            return None
        return ("C4 left %+d B unexplained live and %+d B against the frozen model — the named-term "
                "decomposition refused to absorb a term with no tensor behind it"
                % (r["C4 quant_scenarios delta"],
                   r.get("C4 broken-model decomposition", 0)))
    expect("phantom-term", phantom, "C4 accepted an unexplained phantom term")

    # 2b. the regression C4 exists to catch: the live tool population replaced by the hand-written
    #     model it superseded. C4a must fire with the whole defect signature unexplained, and C4b must
    #     stay clean, because the frozen census really did carry that defect.
    def model_regression():
        pf = build(faults=("model-regression",))
        _, r, s = closures(pf)
        if "C4 quant_scenarios delta" not in r:
            return None
        other = sorted(k for k in r if k.startswith("C4 broken-model"))
        if other:
            return ("C4a fired but C4b broke with it (%s): the fixture and the live populations no "
                    "longer describe the same retired model, which is a fixture problem, not a "
                    "regression" % other)
        return ("C4a left %+d B unexplained — the tool's census had been swapped for the retired "
                "model and only the live arm noticed; C4b stayed clean, which is what 'the two agree' "
                "has to mean"
                % r["C4 quant_scenarios delta"])
    expect("tool-regression-to-model", model_regression,
           "the live C4a arm could not tell the rebuilt census from the hand-written model it replaced")

    # 3. a fused/split mis-pairing: gate_up_proj paired with gate_exps + down_exps. Element counts
    #    alone cannot catch this (both halves are 268,435,456 elts); the half-shape and double-claim
    #    checks can, which is exactly why they exist.
    def mispair():
        build(faults=("mispair",))
    expect("mispair", mispair, "a mis-paired fused/split split passed the pairing checks")

    # 3b. the quiet mis-pair: the fused halves' shapes swapped, element counts untouched, so only
    #     the half-shape check can see it.
    expect("swap-half-shape", lambda: build(faults=("swap-half-shape",)),
           "swapping two equal-element fused halves' shapes passed the half-shape check")

    # 4. a short parse: one artifact record dropped after the header said 733
    expect("short-record", lambda: build(faults=("short-record",)),
           "the record-count assert did not notice a missing record")

    for ln in fired:
        print("  PASS " + ln)
    for ln in missed:
        print("  FAIL " + ln)
    ok = not missed and not clean and len(fired) == injected[0]
    print("selftest: %d faults injected, %d fired, %d missed, %d clean-run condition(s) failed — %s"
          % (injected[0], len(fired), len(missed), len(clean), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    raise SystemExit(selftest() if args.selftest else report())
