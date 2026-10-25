#!/usr/bin/env python3
"""residency_budget.py — does policy A fit in the device-visible pool, and where does it stop fitting?

CPU arithmetic only. No torch, no numpy, no device call, no timing. Every device number it spends is
already in this repo and is cited at the point of use; nothing here is measured. The prose that goes
with it — the verdicts, the give-up ladder, the trap rows and the UNMEASURED queue — is
`quantization/residency-layout.md`.

Why this file exists: `tools/quant_scenarios.py` answers "how big is the weight set at width X"; it
deliberately ignores everything that grows with context (KV) and with concurrency (the GDN state
pool), and this is the first model in this project lineage where the pool binds (bf16 text weights
66.1337 GiB including the MTP draft block, 64.5604 GiB without it — the figure was 65.72 GiB until
quantization/evidence/bf16-census-adjudication-2026-10-26.md settled it, and this file now reconciles
against `tools/bf16_census.py` rather than quoting it; against a box-wide device-visible pool of
55.5-56.5 GiB, of which one process gets ~32 GiB while halogen is resident — `devices/measured-facts.md`
§"Device-visible memory" and §"Residency above the pool"). Nobody had written that budget down. This is
that budget, per component, derived.

Everything is derived from three committed sources, never typed:
  * `checkpoints/Qwen3.6-35B-A3B.config.json`   — the geometry (layers, heads, experts, state dims)
  * `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` — the 1,045-name tensor inventory
  * `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json` — the pinned artifact's authority (MTP element count)
plus, when the pinned file is on disk, `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`'s own
tensor-info header, which is the only *independent* witness of the served shape (733 real tensors).

Widths are `tools/ggml_block_geom.py`'s (derived from the pinned ggml-common.h structs), not a typed
bpw table — the same reason `quant_scenarios.py` stopped hand-typing them.

Usage:
  python3 tools/residency_budget.py                    # budget table; non-zero if a row cannot fit
  python3 tools/residency_budget.py --report           # same table, always exit 0
  python3 tools/residency_budget.py --give-up          # the savings ladder, cheapest first
  python3 tools/residency_budget.py --ctx 4k,128k --seqs 4 --kv-type q8_0 --policy F
  python3 tools/residency_budget.py --selftest         # witnesses + fault injections; see §selftest

Exit codes (main table): 0 every row fits the co-resident budget; 1 some row fits only with halogen
down; 2 some row fits neither budget; 3 a witness failed (the arithmetic is broken, so the table is
not trustworthy). `--report` downgrades 1 and 2 to 0; it never downgrades 3, because a broken
calculator printing a tidy table is the failure mode this repo has been bitten by.
"""
import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_block_geom as GBG
import gguf_header_types as GHT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")
IDX_PATH = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json")
PIN_PATH = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")
GGUF_PATH = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
MTP_PATH = os.path.join(REPO, "models/w1-mtp-extract/mtp.safetensors")
SHARD_GLOB = os.path.join(REPO, "models/qwen36-src/*.safetensors")

GiB = 2.0 ** 30
MiB = 2.0 ** 20

# ---- device-side facts. All measured, all cited; none invented, none estimated. --------------
# Pool/footprint bracket. devices/measured-facts.md §"Device-visible memory" (32 GiB one hipMalloc
# ok / 40 GiB fails; 8x8 GiB -> 4th ok; 24x2 GiB -> 17 ok = 34 GiB) and §"Residency above the pool"
# (23.5 + 32 granted, 23.5 + 34 refused; halogen's footprint is 22.62 GiB by `rocm-smi --showpids`
# and 23.53-23.66 GiB by gtt_used — two counters, both quoted). The pool is somewhere in
# 55.5-56.5 GiB and the footprint in 22.62-23.66 GiB, so what one co-resident process may map is
# itself a BRACKET, not a round 32:
POOL_LO, POOL_HI = 55.5, 56.5
HALOGEN_LO, HALOGEN_HI = 22.62, 23.66
CO_LO = POOL_LO - HALOGEN_HI          # 31.84 GiB — the pessimistic co-resident budget
CO_HI = POOL_HI - HALOGEN_LO          # 33.88 GiB — the optimistic one
DEDICATED_LO = POOL_LO                # what the engine gets if it owns the box (D-11, unconfirmed)
STAGING_GBS = (54.0, 58.0)            # hipHostMalloc stage -> hipMemcpy incl. dest first touch,
                                      # pitfalls/10-device-memory.md §"Allocation recipe"
FIRST_TOUCH_GBS = 35.0                # same file, §"Memory bandwidth"; devices/measured-facts.md
READ_GBS = (237.0, 210.0, 190.0)      # the repo's roofline denominators

# ---- the width model: policy A, spelled out per family, and the alternatives it can be varied to --
# Policy A is D-01 (routed experts Q4_K) + D-01a (attention/GDN Q8_0) + D-08 (Q4_K, not MXFP4/IQ4_XS)
# + D-09 (the MTP draft block is uniform with the stack it drafts for) + D-10 (lm_head/embeddings
# Q8_0). "exact" = stored as the checkpoint has it = BF16 (`dtype: bfloat16`); router/norms/GDN
# controls are never quantized (quantization/README.md §"How a quantized path gets gated" item 3).
POLICIES = {
    #                 experts shared attn   lmhead  exact-controls
    "A (DECIDED)": dict(experts="Q4_K", shared="Q8_0", attn="Q8_0", lm_head="Q8_0", ctrl="BF16"),
    "B attn bf16":  dict(experts="Q4_K", shared="BF16", attn="BF16", lm_head="BF16", ctrl="BF16"),
    "E attn Q4_K":  dict(experts="Q4_K", shared="Q4_K", attn="Q4_K", lm_head="Q6_K", ctrl="BF16"),
    "F Q8 through": dict(experts="Q8_0", shared="Q8_0", attn="Q8_0", lm_head="Q8_0", ctrl="BF16"),
    "G lm_head Q6": dict(experts="Q4_K", shared="Q8_0", attn="Q8_0", lm_head="Q6_K", ctrl="BF16"),
    "bf16":         dict(experts="BF16", shared="BF16", attn="BF16", lm_head="BF16", ctrl="BF16"),
}
# Which policy-A width class each inventory family belongs to. This is the one place it is spelled.
FAMILY_CLASS = {
    "routed_experts": "experts", "shared_experts": "shared", "full_attn": "attn",
    "gdn_matrices": "attn", "gdn_controls": "ctrl", "routers": "ctrl", "norms": "ctrl",
    "embed": "lm_head",       # D-10: embeddings follow lm_head's width (untied table, same role)
    "lm_head": "lm_head",
}
# Families a decode step reads in full, every token. Everything except the routed experts (8 of 256
# fire) and the token embedding (one row). `lm_head` is in here because `tie_word_embeddings` is
# false: an untied output table cannot reuse the embedding read, so it is a whole extra 0.50 GiB
# Q8_0 per token (AGENTS.md prediction 5 — "untied lm_head costs 1.0 GiB/token" at bf16).
READ_FULL = {"shared_experts", "full_attn", "gdn_matrices", "gdn_controls", "routers", "norms",
             "lm_head"}

# KV container choices. The container is a USER decision (the D-01/D-01a pattern covers weight
# widths and says nothing about KV), so this is a menu, not a recommendation. Bytes come from the
# ggml type's own derived bpw, which is what a block container actually costs (scales included).
KV_TYPES = {"bf16": "BF16", "f16": "F16", "q8_0": "Q8_0", "q4_k": "Q4_K", "q4_0": "Q4_0",
            "iq4_xs": "IQ4_XS"}

FAULTS = {
    "none":                 "baseline, nothing injected",
    "kv_zero":              "KV cache row computes to 0 bytes/token",
    "gdn_recurrent_dropped": "GDN recurrent state left out of the state pool",
    "gdn_conv_dropped":     "GDN conv window left out of the state pool",
    "lm_head_not_read":     "per-token read forgets the untied lm_head",
    "gdn_out_proj_dropped": "GDN out_proj missing from the weight census (the real defect found)",
    "draft_dropped":        "MTP draft block excluded from residency (contradicts D-05)",
    "draft_kv_missing":     "the draft layer's own KV rows are not reserved",
    "one_attn_layer_lost":  "one full-attention layer missing from the inventory",
    "lm_head_dup":          "lm_head counted twice in the census",
}


# -----------------------------------------------------------------------------------------------
# sources
class Sources:
    """The committed geometry, plus the two optional on-disk witnesses (shard headers, GGUF)."""

    def __init__(self, want_artifact=True):
        self.cfg = json.load(open(CFG_PATH))["text_config"]
        idx = json.load(open(IDX_PATH))
        self.names = [n for n in idx["weight_map"] if "visual" not in n]
        self.vision_names = [n for n in idx["weight_map"] if "visual" in n]
        self.index_total_size = idx["metadata"]["total_size"]
        self.pin = json.load(open(PIN_PATH))
        self.gguf = None
        if want_artifact and os.path.exists(GGUF_PATH):
            self.gguf = self.read_gguf()
        self.shard_shapes = self.read_shard_shapes()

    # --- config-derived shapes, verified against any real shard header that is present ---------
    def shape_rules(self):
        c = self.cfg
        H, E = c["hidden_size"], c["num_experts"]
        me = c["moe_intermediate_size"]
        hd, nq, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        lk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        lv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        kf = c["linear_conv_kernel_dim"]
        qkv = 2 * lk * dk + lv * dv          # conv channels = q | k | v rows
        return [
            (r"^lm_head\.weight$", [c["vocab_size"], H]),
            (r"embed_tokens\.weight$", [c["vocab_size"], H]),
            (r"embed_tokens\.weight$", [c["vocab_size"], H]),
            (r"\.input_layernorm\.weight$", [H]),
            (r"\.post_attention_layernorm\.weight$", [H]),
            (r"\.mlp\.experts\.down_proj$", [E, H, me]),
            (r"\.mlp\.experts\.gate_up_proj$", [E, 2 * me, H]),
            (r"\.mlp\.gate\.weight$", [E, H]),
            (r"\.mlp\.shared_expert\.down_proj\.weight$", [H, me]),
            (r"\.mlp\.shared_expert\.gate_proj\.weight$", [me, H]),
            (r"\.mlp\.shared_expert\.up_proj\.weight$", [me, H]),
            (r"\.mlp\.shared_expert_gate\.weight$", [1, H]),
            (r"\.linear_attn\.A_log$", [lv]),
            (r"\.linear_attn\.dt_bias$", [lv]),
            (r"\.linear_attn\.conv1d\.weight$", [qkv, 1, kf]),
            (r"\.linear_attn\.in_proj_a\.weight$", [lv, H]),
            (r"\.linear_attn\.in_proj_b\.weight$", [lv, H]),
            (r"\.linear_attn\.in_proj_qkv\.weight$", [qkv, H]),
            (r"\.linear_attn\.in_proj_z\.weight$", [lv * dv, H]),
            (r"\.linear_attn\.norm\.weight$", [dv]),
            (r"\.linear_attn\.out_proj\.weight$", [H, lv * dv]),
            (r"\.self_attn\.q_proj\.weight$", [2 * nq * hd, H]),   # attn_output_gate doubles it
            (r"\.self_attn\.k_proj\.weight$", [nkv * hd, H]),
            (r"\.self_attn\.v_proj\.weight$", [nkv * hd, H]),
            (r"\.self_attn\.o_proj\.weight$", [H, nq * hd]),
            (r"\.self_attn\.q_norm\.weight$", [hd]),
            (r"\.self_attn\.k_norm\.weight$", [hd]),
            (r"^mtp\.fc\.weight$", [H, 2 * H]),
            (r"\.pre_fc_norm_embedding\.weight$", [H]),
            (r"\.pre_fc_norm_hidden\.weight$", [H]),
            # generic last: `linear_attn.norm.weight` is [d_v] and must be matched by its own rule
            # above, while `language_model.norm.weight` and `mtp.norm.weight` are both [H].
            (r"\.norm\.weight$", [H]),
        ]

    def read_shard_shapes(self):
        """{name: shape} for whatever HF shards are on disk — used to verify the rules above."""
        import glob
        out = {}
        for p in sorted(glob.glob(SHARD_GLOB)):
            for k, v in safetensors_header(p).items():
                if k == "__metadata__" or "shape" not in v:
                    continue
                out[k] = list(v["shape"])
        return out

    def read_gguf(self):
        """(rows, sum_row_bytes, data_start, file_size) from the pinned GGUF's own header."""
        size = os.path.getsize(GGUF_PATH)
        data = GHT.fetch(GGUF_PATH, 32 << 20)
        c = GHT.Cur(data)
        c.take(4)
        c.u32()
        n_tensors, n_kv = c.u64(), c.u64()
        meta = {}
        for _ in range(n_kv):
            k = c.s()
            vt = c.u32()
            meta[k] = GHT.skip_array(c) if vt == 9 else c.value(vt)
        rows, total = [], 0
        for _ in range(n_tensors):
            name = c.s()
            nd = c.u32()
            dims = tuple(c.u64() for _ in range(nd))
            dt = c.u32()
            c.u64()
            el = prod(dims)
            tname = GHT.TYPES.get(dt, "type_%d" % dt)
            total += GBG.row_bytes(el, tname)
            rows.append((name, dims, tname, el))
        # data_start = align_up(header_end, alignment); alignment is the file's own key.
        align = 32
        for k, v in meta.items():
            if k.endswith("general.alignment"):
                align = int(v)
        header_end = c.p
        data_start = (header_end + align - 1) // align * align
        return dict(rows=rows, sum_row_bytes=total, data_start=data_start, file_size=size,
                    header_end=header_end, alignment=align, arch=meta.get("general.architecture"),
                    n_tensors=n_tensors)


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def prod(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def family_of(name):
    if name == "lm_head.weight":
        return "lm_head"
    if "embed_tokens" in name:
        return "embed"
    if ".experts." in name:
        return "routed_experts"
    if ".linear_attn." in name:
        return "gdn_matrices" if re.search(r"in_proj|out_proj", name) else "gdn_controls"
    if ".self_attn." in name:
        return "full_attn"
    if "shared_expert" in name:
        return "shared_experts"
    if name.endswith("mlp.gate.weight"):
        return "routers"
    return "norms"


# -----------------------------------------------------------------------------------------------
# the calculator. `fault` names one entry of FAULTS and is the ONLY way a value can be perturbed;
# selftest injects faults here and then asks whether the witnesses notice.
class Budget:
    def __init__(self, src, policy="A (DECIDED)", kv_type="bf16", fault="none"):
        self.src = src
        self.fault = fault
        self.policy = POLICIES[policy]
        self.policy_name = policy
        self.kv_type = kv_type
        c = src.cfg
        self.L = c["num_hidden_layers"]
        self.layer_types = c["layer_types"]
        self.n_full = sum(1 for t in self.layer_types if t == "full_attention")
        self.n_lin = sum(1 for t in self.layer_types if t == "linear_attention")
        self.hd = c["head_dim"]
        self.nkv = c["num_key_value_heads"]
        self.nq = c["num_attention_heads"]
        self.E = c["num_experts"]
        self.k = c["num_experts_per_tok"]
        self.V = c["vocab_size"]
        self.lk, self.dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        self.lv, self.dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        self.kf = c["linear_conv_kernel_dim"]
        self.mtp_layers = c["mtp_num_hidden_layers"]
        self.max_ctx = c["max_position_embeddings"]
        self.untied = not c["tie_word_embeddings"]
        self.state_dtype_bits = 32 if c["mamba_ssm_dtype"] == "float32" else 16
        self.build_census()
        self.build_state_geometry()

    # ---- weights -------------------------------------------------------------------------
    def build_census(self, fault=""):
        """{family: elements}, from the index inventory x config-derived shapes."""
        fault = fault or self.fault
        rules = [(re.compile(rx), sh) for rx, sh in self.src.shape_rules()]
        self.census = {}
        self.records = 0
        self.unshaped = []
        self.inventory_elements = 0
        self.draft_elements = 0
        for name in self.src.names:
            if fault == "one_attn_layer_lost" and name.endswith(".self_attn.v_proj.weight") \
                    and name.split(".")[3] == "3":
                continue
            shape = None
            for rx, sh in rules:
                if rx.search(name):
                    shape = sh
                    break
            if shape is None:
                self.unshaped.append(name)
                continue
            el = prod(shape)
            if fault == "gdn_out_proj_dropped" and name.endswith(".linear_attn.out_proj.weight"):
                continue
            if fault == "lm_head_dup" and name == "lm_head.weight":
                el *= 2
            fam = family_of(name)
            self.census[fam] = self.census.get(fam, 0) + el
            self.records += 1
            self.inventory_elements += el
            if name.startswith("mtp."):
                self.draft_elements += el
        # the draft block is its own row: D-05 has it in scope and the published GGUF carries zero
        # draft tensors (quantization/evidence/q4km-published-width-map-2026-10-26.md), so its bytes
        # can only come from the HF shards.
        self.main_elements = self.inventory_elements - self.draft_elements
        self.draft_census = {}
        for name in self.src.names:
            if not name.startswith("mtp."):
                continue
            for rx, sh in rules:
                if rx.search(name):
                    fam = family_of(name)
                    self.draft_census[fam] = self.draft_census.get(fam, 0) + prod(sh)
                    break

    def width_bits(self, fam):
        return GBG.bpw_of(self.policy[FAMILY_CLASS[fam]])

    def bytes_of(self, fam, elements, fault=""):
        fault = fault or self.fault
        return elements * self.width_bits(fam) / 8.0

    def weight_bytes(self, fault=""):
        """(main weights, draft weights) resident, everything at policy width."""
        fault = fault or self.fault
        main = sum(self.bytes_of(f, e) for f, e in self.census.items() if f in
                   ("routed_experts", "shared_experts", "full_attn", "gdn_matrices", "gdn_controls",
                    "routers", "norms", "embed", "lm_head"))
        # main census includes the draft tensors; take them out and re-add as its own row
        main -= sum(self.bytes_of(f, e) for f, e in self.draft_census.items())
        draft = sum(self.bytes_of(f, e) for f, e in self.draft_census.items())
        if fault == "draft_dropped":
            draft = 0.0
        return main, draft

    # ---- per-token traffic ------------------------------------------------------------
    def per_token_bytes(self, fault=""):
        """GiB read per decoded token: everything walked in full + k/E of the routed experts.

        Two routes on purpose: `direct` walks the family census, `by_name` re-walks the 712 tensor
        names independently. A calculator that drops a family in one place is caught by the other.
        """
        fault = fault or self.fault
        direct = 0.0
        for fam, el in self.census.items():
            if fam == "embed":
                direct += self.src.cfg["hidden_size"] * self.width_bits("embed") / 8.0  # one row
                continue
            if fam == "lm_head" and fault == "lm_head_not_read":
                continue
            if fam == "routed_experts":
                direct += self.bytes_of(fam, el) * self.k / self.E
            elif fam in READ_FULL:
                direct += self.bytes_of(fam, el)
        by_name = 0.0
        rules = [(re.compile(rx), sh) for rx, sh in self.src.shape_rules()]
        for name in self.src.names:
            fam = family_of(name)
            for rx, sh in rules:
                if rx.search(name):
                    el = prod(sh)
                    if fam == "embed":
                        by_name += self.src.cfg["hidden_size"] * self.width_bits("embed") / 8.0
                    elif fam == "routed_experts":
                        by_name += el * self.width_bits("routed_experts") / 8.0 * self.k / self.E
                    elif fam in READ_FULL:
                        by_name += el * self.width_bits(fam) / 8.0
                    break
        self.per_token_direct, self.per_token_by_name = direct, by_name
        return direct

    # ---- growing state ---------------------------------------------------------------
    def build_state_geometry(self):
        c = self.src.cfg
        self.kv_el_per_tok_layer = 2 * self.nkv * self.hd        # K row + V row
        self.kv_bits = GBG.bpw_of(KV_TYPES[self.kv_type])
        self.kv_el_per_tok = self.kv_el_per_tok_layer * self.n_full
        self.kv_bytes_per_tok = self.kv_el_per_tok * self.kv_bits / 8.0
        self.draft_kv_bytes_per_tok = self.kv_el_per_tok_layer * self.mtp_layers * self.kv_bits / 8.0
        # GDN state: the recurrent matrix is [h_v, d_k, d_v] per linear layer and the conv window is
        # [conv_dim, k_conv] (or k_conv-1); see specification/04-gdn-linear-attention.md §04.10,
        # which derives both from the HF reference and agrees with these numbers to the byte.
        self.rec_el_layer = self.lv * self.dk * self.dv
        self.rec_bytes_layer = self.rec_el_layer * self.state_dtype_bits / 8.0
        self.conv_dim = 2 * self.lk * self.dk + self.lv * self.dv
        self.conv_el_layer = self.conv_dim * (self.kf - 1)
        self.conv_bytes_layer = self.conv_el_layer * 16 / 8.0    # activation dtype = bf16 (A-09)
        self.logits_bytes = self.V * 4.0                        # fp32 logits, one token

    def kv_bytes(self, ctx, fault=""):
        fault = fault or self.fault
        if fault == "kv_zero":
            return 0.0
        return self.kv_bytes_per_tok * ctx

    def draft_kv_bytes(self, ctx, fault=""):
        fault = fault or self.fault
        if fault in ("kv_zero", "draft_kv_missing"):
            return 0.0
        return self.draft_kv_bytes_per_tok * ctx

    def state_bytes(self, nseq, fault=""):
        fault = fault or self.fault
        rec = 0.0 if fault == "gdn_recurrent_dropped" else self.rec_bytes_layer * self.n_lin
        conv = 0.0 if fault == "gdn_conv_dropped" else self.conv_bytes_layer * self.n_lin
        return (rec + conv) * nseq

    def other_bytes(self, nseq=1):
        # logits for the target pass plus a draft buffer of the same size; sampler scratch and any
        # compute/activation buffer are NOT here (UNMEASURED — see quantization/residency-layout.md).
        return 2 * self.logits_bytes

    # ---- one budget row ---------------------------------------------------------------
    def row(self, ctx, nseq=1, fault=""):
        main, draft = self.weight_bytes(fault)
        # weights are shared by the sequences; KV and the GDN state are per sequence
        kv = (self.kv_bytes(ctx, fault) + self.draft_kv_bytes(ctx, fault)) * nseq
        state = self.state_bytes(nseq, fault)
        other = self.other_bytes(nseq)
        total = main + draft + kv + state + other
        return dict(ctx=ctx, seqs=nseq, main=main, draft=draft, kv=kv, state=state, other=other,
                    total=total, per_token=self.per_token_bytes(fault),
                    verdict=verdict(total / GiB))


def verdict(total_gib):
    if total_gib <= CO_LO:
        return "FITS co-resident"
    if total_gib <= CO_HI:
        return "FITS co-resident only in the lucky half of the bracket"
    if total_gib <= DEDICATED_LO:
        return "FITS only with halogen down"
    return "DOES NOT FIT"


# -----------------------------------------------------------------------------------------------
# witnesses. Each returns (name, ok, detail). A witness that cannot run reports ok=False unless the
# reason is recorded and counted, because a check that cannot produce output cannot fail.
def witnesses(src, b, artifact_required):
    out = []
    # W1 record count: parsed from the index vs derived from the config's layer counts.
    c = src.cfg
    exp = (2 * b.L                                        # input/post_attention norms
           + sum(1 for t in b.layer_types if t == "linear_attention") * 9   # linear_attn tensors
           + sum(1 for t in b.layer_types if t == "full_attention") * 6     # self_attn tensors
           + b.L * 7                                      # MoE: 2 experts + gate + 3 shared + sh gate
           + 3                                            # embed, lm_head, final norm
           + 19)                                          # MTP block, one layer
    out.append(("record count", b.records == exp and not b.unshaped,
                "parsed %d tensor records from the index, %d derived from config layer counts%s"
                % (b.records, exp,
                   "" if not b.unshaped else ", %d names matched no shape rule: %s"
                   % (len(b.unshaped), b.unshaped[:4]))))
    # W2 shape rules vs real shard headers
    checked = mismatch = 0
    for name, shape in src.shard_shapes.items():
        if "visual" in name:
            continue
        for rx, sh in src.shape_rules():
            if re.search(rx, name):
                checked += 1
                mismatch += (list(sh) != list(shape))
                break
    out.append(("config-derived shapes vs real shard headers",
                checked > 0 and mismatch == 0,
                "%d tensor shapes checked against safetensors headers, %d mismatched"
                % (checked, mismatch)))
    # W3 the draft block: our inventory vs the pin's own authority
    pin_el = src.pin["authority"]["mtp_elements"]
    out.append(("MTP draft-block elements", b.draft_elements == pin_el,
                "derived %d, pin authority %s (=%s MiB bf16), extracted-file agree see --selftest log"
                % (b.draft_elements, pin_el, round(pin_el * 2 / MiB, 2))))
    # W4 layer split two ways
    out.append(("full/linear layer split", b.n_full == b.L // c["full_attention_interval"]
                and b.n_full + b.n_lin == b.L and b.n_full > 0,
                "layer_types gives %d full + %d linear; interval %d gives %d full"
                % (b.n_full, b.n_lin, c["full_attention_interval"], b.L // c["full_attention_interval"])))
    # W5 no component is silently zero
    comps = {"main weights": b.weight_bytes()[0], "draft weights": b.weight_bytes()[1],
             "KV @4k": b.kv_bytes(4096), "draft KV @4k": b.draft_kv_bytes(4096),
             "GDN state @1 seq": b.state_bytes(1), "per-token read": b.per_token_bytes()}
    zeros = [k for k, v in comps.items() if v == 0]
    out.append(("no component sums to zero", not zeros,
                "components in GiB: " + ", ".join("%s %.4f" % (k, v / GiB) for k, v in comps.items())
                + ("" if not zeros else "  <-- ZERO: %s" % zeros)))
    # W6 the two per-token routes agree
    d = abs(b.per_token_direct - b.per_token_by_name)
    out.append(("per-token read: two independent routes agree", d < 1.0,
                "family-census route %.4f GiB/tok, name-walk route %.4f GiB/tok, delta %.1f B"
                % (b.per_token_direct / GiB, b.per_token_by_name / GiB, d)))
    # W7 the artifact witnesses
    if src.gguf:
        g = src.gguf
        out.append(("GGUF size identity", g["sum_row_bytes"] + g["data_start"] == g["file_size"],
                    "sum(row_bytes) %d + data_start %d vs file %d"
                    % (g["sum_row_bytes"], g["data_start"], g["file_size"])))
        gg_el = sum(r[3] for r in g["rows"])
        # The published file carries 40 layers and no draft block, so it must equal the main part
        # of our census element-for-element. This is the one witness that does not share its inputs
        # with the config derivation.
        out.append(("artifact element total == our census minus the draft block",
                    gg_el == b.main_elements,
                    "GGUF %d elements (bf16 %.4f GiB) vs derived-minus-draft %d (%.4f GiB); delta %d"
                    % (gg_el, gg_el * 2 / GiB, b.main_elements, b.main_elements * 2 / GiB,
                       gg_el - b.main_elements)))
        out.append(("GGUF carries no draft tensors (so MTP bytes must come from HF)",
                    not any("mtp" in r[0] for r in g["rows"]),
                    "%d tensors whose name contains 'mtp' in %d"
                    % (sum(1 for r in g["rows"] if "mtp" in r[0]), len(g["rows"]))))
    else:
        out.append(("artifact witnesses", not artifact_required,
                    "SKIPPED: %s absent, so the census-vs-artifact witness is vacuous this run%s"
                    % (GGUF_PATH, "" if not artifact_required else
                       "  <-- pass --allow-missing-artifact to accept this")))
    # W8 reconcile against the census (tools/bf16_census.py), family by family and convention by
    # convention. Never against a remembered number: see census_reconcile's header comment.
    out.extend(census_reconcile(b))
    return out


_REFERENCE = {}


def census_reference():
    """tools/bf16_census.py's populations and policy-A row, derived once per process and then frozen.

    Memoised on purpose, for two reasons. The cheap one: building it parses the pinned GGUF, four
    safetensors headers and an imported quant_scenarios census, and a selftest asks ten times. The
    load-bearing one: the reference must NOT move when one of this file's own fault hooks fires, so
    the same frozen object is what every injection in the selftest is measured against.
    """
    if "d" not in _REFERENCE:
        import bf16_census as BC
        import quant_scenarios as QS
        p = BC.build()
        # The reference must be complete or it is not a reference: an empty population would let every
        # row below "agree" with nothing, so a short census aborts instead of reconciling against 0.
        fam = {k: v for k, v in p["fam_index"].items() if k != "vision"}
        if len(fam) != 8 or sum(fam.values()) <= 0:
            raise AssertionError("bf16_census gave %d text families totalling %d elements; a short "
                                 "reference census is an error, not a zero to reconcile against"
                                 % (len(fam), sum(fam.values())))
        rows, draft = BC.policy_rows(p, "A", "bf16")          # bf16_census census + bf16_census widths
        bc_a = (sum(rows.values()) + sum(draft.values())) / GiB
        if bc_a <= 0:
            raise AssertionError("bf16_census priced policy A at %.6f GiB; a zero policy row is not a "
                                 "figure" % bc_a)
        qs_a = QS.derived_numbers(QS.build_population())["A_resident"]   # theirs, both census and widths
        if qs_a <= 0:
            raise AssertionError("quant_scenarios priced policy A at %.6f GiB; a zero policy row is not "
                                 "a figure" % qs_a)
        _REFERENCE["d"] = dict(fam=fam, policy_a_gib=bc_a, qs_policy_a_gib=qs_a)
    return _REFERENCE["d"]


# ---- The 19.7856 / 19.7848 GiB difference: read this before re-litigating it. --------------------
# Two honest rulers print two policy-A resident figures for the same decided policy: this tool prints
# 19.7856 GiB and tools/bf16_census.py (and tools/quant_scenarios.py, and quantization/README.md,
# README.md and DECISIONS.md) print 19.7848 GiB. The difference is 927,000 bytes = 0.000863 GiB, it is
# not a disagreement about anything, and census_reconcile's third row below proves that claim rather
# than asserting it: it attributes the whole gap to one convention and requires the residue to be ~0 B.
#
# The convention is which width class the four 1-D GDN controls belong to — `ssm_conv1d`, `ssm_a`
# (A_log), `ssm_dt.bias` (dt_bias) and `ssm_norm`, 988,800 elements in total (30 linear layers x
# 32,960). quant_scenarios and bf16_census price by FAMILY, and those four tensors sit inside the GDN
# family, so policy A's attention width (Q8_0, 8.5 bpw) is applied to them. This tool splits the family
# into `gdn_matrices` and `gdn_controls` and keeps the controls exact (bf16, 16 bpw), which is what
# quantization/README §"How a quantized path gets gated" item 3 ("router, norms ... stay exact") reads
# like. That is the entire gap: 988,800 x (16 - 8.5)/8 = 927,000 B.
#
# So the two tools agree on the inventory (row 1, 0 elts on all eight families), on the policy, and on
# every other width, and they differ on the reading of one sentence of a doc — which is a width
# decision, and width decisions are the user's (the D-01/D-01a pattern). It is recorded as an open
# question in quantization/evidence/bf16-census-adjudication-2026-10-26.md §10.2 ("policy A's 'exact'
# bucket: bf16 or F32", with this 0.000863 GiB as its §5 item 7). 0.000863 GiB is 0.88 MiB: two orders
# of magnitude below anything that could move a fit verdict (the co-resident budget bracket is 31.84-33.88
# GiB, a 2.04 GiB window) and below the 0.0403 GiB that the F32-vs-bf16 reading of the same bucket is
# worth. Nothing in this file should be changed to chase the fourth decimal place; pick one convention
# when the user picks one for §10.2 and both tools will agree to the byte.
def census_reconcile(b):
    """Reconcile against `tools/bf16_census.py` — the census — naming every difference that survives.

    This used to reconcile against `tools/quant_scenarios.py`'s hand-written census and enumerate the
    four defects that census carried, and its own text said what to do the day they were fixed: "retire
    the reconciliation here, do not tune the tolerance." That day is 2026-10-26 (commit a912ac9), so the
    cause-by-cause rows are gone: a reconciliation whose cause list describes a census that no longer
    exists fires forever, and a witness that fires forever is worth exactly as much as one that never
    fires. What replaces them is a reconciliation against the *census*, which still has work to do:

      1. this tool's own inventory (config shape rules x the index name list) vs bf16_census's inventory
         (the artifact's records + the draft names it lacks, shard-witnessed). Two independently written
         shape models, required to agree element for element on all eight text families.
      2. policy A priced twice: bf16_census's census x bf16_census's widths vs quant_scenarios's own
         census x quant_scenarios's own derived row. Two width rules over two censuses; they must land
         on the same figure to within 0.0005 GiB.
      3. this tool's policy A vs bf16_census's, with the whole difference attributed to the one
         convention described above and required to have no residue.

    Row 1 is what `gdn_out_proj_dropped` and `one_attn_layer_lost` now trip, so the retirement did not
    quietly remove a live check; run `--selftest` and read the FIRED lines to see it.
    """
    try:
        ref = census_reference()
    except Exception as e:                                        # pragma: no cover
        return [("census reconciliation vs tools/bf16_census.py", False,
                 "cannot build the reference population: %r — a census tool that cannot run is not the "
                 "same thing as a census that agrees" % e)]
    # These rows are about policy A specifically (that is the figure the repo quotes), so they judge a
    # policy-A budget even when the run asked for a different policy row.
    if b.policy_name != "A (DECIDED)":
        b = Budget(b.src)
    out = []

    mine = {"routed_experts": b.census["routed_experts"],
            "gdn_linear_attn": b.census["gdn_matrices"] + b.census["gdn_controls"],
            "full_attn": b.census["full_attn"], "shared_experts": b.census["shared_experts"],
            "embed": b.census["embed"], "lm_head": b.census["lm_head"],
            "routers": b.census["routers"], "norms_misc": b.census["norms"]}
    theirs = ref["fam"]
    per_family = {f: mine.get(f, 0) - theirs.get(f, 0) for f in sorted(set(mine) | set(theirs))}
    drift = {f: d for f, d in per_family.items() if d}
    mine_total, theirs_total = sum(mine.values()), sum(theirs.values())
    # non-vacuity is part of the condition: eight populated families, neither side empty. A reconcile
    # against an empty census would "agree" with nothing at all.
    ok = not drift and len(theirs) == 8 and theirs_total > 0 and mine_total > 0
    out.append(("text census vs tools/bf16_census.py, family by family", ok,
                "ours %.4f GiB vs theirs %.4f GiB bf16 over %d families (%d elements each side): %s%s"
                % (mine_total * 2 / GiB, theirs_total * 2 / GiB, len(theirs), mine_total,
                   "0 elements of difference on every family" if not drift
                   else "DRIFT " + ", ".join("%s %+d el" % kv for kv in drift.items()),
                   "" if theirs_total else "  <-- the reference census is empty, which is not agreement")))

    bc_a, qs_a = ref["policy_a_gib"], ref["qs_policy_a_gib"]
    out.append(("policy A resident: bf16_census widths vs quant_scenarios' own derived row",
                abs(bc_a - qs_a) <= 5e-4 and qs_a > 0,
                "bf16_census %.4f GiB, quant_scenarios %.4f GiB (both derived here, tolerance 0.0005 "
                "GiB). Two independent width rules over two independent censuses, so a gap is a width "
                "rule and never a census. Neither side is a remembered number; if this fires, look at "
                "ggml_block_geom, not at a tolerance."
                % (bc_a, qs_a)))

    ours_a = sum(b.weight_bytes()) / GiB
    attn_bits = b.width_bits("gdn_matrices")
    ctrl_bits = b.width_bits("gdn_controls")
    ctrl_el = b.census["gdn_controls"]
    convention = ctrl_el * (ctrl_bits - attn_bits) / 8.0
    gap = (ours_a - bc_a) * GiB
    # The slack is block rounding, not fudge: this tool scales a family by bpw, bf16_census prices whole
    # ggml blocks per tensor. On this checkpoint the two agree to the byte; the slack is for a family
    # that stops dividing evenly into a block, and it is ~1e-6 of the 927,000 B this row attributes, so
    # it cannot absorb a real convention change.
    residual = gap - convention
    out.append(("policy A: this tool minus bf16_census is only the GDN-control width convention",
                abs(residual) < 1024.0 and ctrl_el > 0,
                "this tool %.4f GiB, bf16_census %.4f GiB, gap %.6f GiB = %d GDN 1-D control elements "
                "(ssm_conv1d/ssm_a/ssm_dt/ssm_norm) charged %g bpw here and %g bpw there = %.0f B, "
                "unattributed %.0f B. Convention, not disagreement — see the block comment above "
                "census_reconcile and adjudication §10.2; do not chase it."
                % (ours_a, bc_a, ours_a - bc_a, ctrl_el, ctrl_bits, attn_bits, convention, residual)))
    return out



# -----------------------------------------------------------------------------------------------
# geometry recomputed a second time, WITHOUT going through the assembly path the budget table uses.
# A dropped or zeroed row in the assembly is exactly what these catch; the fault hooks above are the
# only way any number can move, and none of them is visible here.
def geo_kv_per_tok(b):
    return 2 * b.nkv * b.hd * b.n_full * b.kv_bits / 8.0


def geo_draft_kv_per_tok(b):
    return 2 * b.nkv * b.hd * b.mtp_layers * b.kv_bits / 8.0


def geo_state_per_seq(b):
    return (b.lv * b.dk * b.dv * b.state_dtype_bits / 8.0
            + b.conv_dim * (b.kf - 1) * 16 / 8.0) * b.n_lin


def assembly_witnesses(b, ctx_a=4096, ctx_b=8192):
    """Row assembly vs geometry, at two context lengths so a constant-only error cannot hide."""
    out = []
    kv_a, kv_b = b.kv_bytes(ctx_a) + b.draft_kv_bytes(ctx_a), b.kv_bytes(ctx_b) + b.draft_kv_bytes(ctx_b)
    want_a = (geo_kv_per_tok(b) + geo_draft_kv_per_tok(b)) * ctx_a
    want_b = (geo_kv_per_tok(b) + geo_draft_kv_per_tok(b)) * ctx_b
    out.append(("KV rows: assembly == geometry, and grows with context",
                kv_a == want_a and kv_b == want_b and geo_kv_per_tok(b) > 0,
                "KV+draft KV %.4f GiB @%d tok (geometry %.4f), %.4f @%d; per-token geometry %.0f B, "
                "draft rows %.0f B%s"
                % (kv_a / GiB, ctx_a, want_a / GiB, kv_b / GiB, ctx_b, geo_kv_per_tok(b),
                   geo_draft_kv_per_tok(b),
                   "" if kv_a == want_a else
                   "  <-- KV computes to 0, which is not a size, it is a missing row" if kv_a == 0 else
                   "  <-- the draft layer's own rows are missing" if kv_a == geo_kv_per_tok(b) * ctx_a
                   else "  <-- KV is not scaling linearly with context")))
    out.append(("draft rows present for the MTP layer (D-05 in scope)",
                b.draft_kv_bytes(ctx_a) == geo_draft_kv_per_tok(b) * ctx_a,
                "draft KV %.6f GiB @%d tok, geometry %.6f (one full-attention layer, %d K+V el/token)"
                % (b.draft_kv_bytes(ctx_a) / GiB, ctx_a, geo_draft_kv_per_tok(b) * ctx_a / GiB,
                   2 * b.nkv * b.hd)))
    st = b.state_bytes(1)
    want = geo_state_per_seq(b)
    out.append(("GDN state pool: assembly == geometry", st == want and want > 0,
                "%.4f GiB/sequence (recurrent %.4f + conv %.6f) vs geometry %.4f"
                % (st / GiB, b.rec_bytes_layer * b.n_lin / GiB,
                   b.conv_bytes_layer * b.n_lin / GiB, want / GiB)))
    # the state geometry, re-derived from real tensor shapes rather than from the head-dim products
    zz = [n for n in b.src.names if n.endswith("linear_attn.in_proj_z.weight")]
    cv = [n for n in b.src.names if n.endswith("linear_attn.conv1d.weight")]
    ok = True
    detail = "no linear_attn tensors in the inventory to check against"
    if zz and cv and b.src.shard_shapes:
        zs = b.src.shard_shapes.get(zz[0])
        cs = b.src.shard_shapes.get(cv[0])
        if zs and cs:
            lv_dv = zs[0]                                     # in_proj_z rows are lv x dv
            dk_from_qkv = None
            for n in b.src.names:
                if n.endswith("linear_attn.in_proj_qkv.weight") and n in b.src.shard_shapes:
                    qkv_rows = b.src.shard_shapes[n][0]
                    dk_from_qkv = (qkv_rows - lv_dv) // (2 * b.lk)
            rec_from_shapes = lv_dv * (dk_from_qkv or b.dk)
            conv_from_shape = cs[0] * cs[2] // b.kf * (b.kf - 1)   # channels x (k_conv - 1)
            ok = (rec_from_shapes == b.rec_el_layer and conv_from_shape == b.conv_el_layer)
            detail = ("recurrent state %d el/layer from [h_v,d_k,d_v] vs %d el/layer from the "
                      "in_proj_z/in_proj_qkv tensor rows; conv state %d vs %d el/layer from the "
                      "conv1d tensor shape" % (b.rec_el_layer, rec_from_shapes, b.conv_el_layer,
                                               conv_from_shape))
    out.append(("GDN state shapes agree with the tensors that produce them", ok, detail))
    main, draft = b.weight_bytes()
    per_tok = b.per_token_bytes()
    lm = b.census.get("lm_head", 0) * b.width_bits("lm_head") / 8.0
    out.append(("per-token read pays for the untied output table",
                b.untied and per_tok >= lm,
                "tie_word_embeddings=false in the config, so lm_head (%.4f GiB at policy width) is a "
                "separate table read in full every token; per-token read is %.4f GiB"
                % (lm / GiB, per_tok / GiB)))
    out.append(("draft block is resident because D-05 has it in scope", draft > 0,
                "draft weights %.4f GiB for %d elements (pin authority)" % (draft / GiB,
                                                                            b.draft_elements)))
    return out


# -----------------------------------------------------------------------------------------------
def print_components(b):
    main, draft = b.weight_bytes()
    print("policy %s — widths from %s" % (b.policy_name, GBG.HEADER_REL))
    print("residency by component (derived, never typed):\n")
    print("  %-30s %12s %10s  %s" % ("component", "bytes", "GiB", "what fixes the number"))
    print("  " + "-" * 104)
    fam_label = {
        "routed_experts": "routed experts (41 x 256)", "shared_experts": "shared experts (41)",
        "full_attn": "full attention (10 + draft)", "gdn_matrices": "GDN matrices (30)",
        "gdn_controls": "GDN controls (conv1d/A_log/dt_bias/norm)", "routers": "routers (41)",
        "norms": "norms + mtp.fc", "embed": "token embeddings", "lm_head": "lm_head (untied)"}
    for fam in ["routed_experts", "shared_experts", "full_attn", "gdn_matrices", "gdn_controls",
                "routers", "norms", "embed", "lm_head"]:
        el = b.census.get(fam, 0)
        print("  %-30s %12d %10.4f  %d elements x %g bpw"
              % (fam_label[fam], b.bytes_of(fam, el), b.bytes_of(fam, el) / GiB, el,
                 b.width_bits(fam)))
    print("  %-30s %12.0f %10.4f  19 tensors, %d el (%.2f MiB bf16); the published GGUF has none"
          % ("  of which: MTP draft block", draft, draft / GiB, b.draft_elements,
             b.draft_elements * 2 / MiB))
    print("  %-30s %12.0f %10.4f  main stack, draft excluded" % ("main weights", main, main / GiB))
    print("  %-30s %12.0f %10.4f  weights total (main + draft)" % ("WEIGHTS", main + draft,
                                                                    (main + draft) / GiB))
    print("  %-30s %12.0f %10.6f  2 x %d kv-heads x %d dim x %d layers x %s"
          % ("KV per token", b.kv_bytes_per_tok, b.kv_bytes_per_tok / GiB, b.nkv, b.hd, b.n_full,
             KV_TYPES[b.kv_type]))
    print("  %-30s %12.0f %10.6f  the draft layer's own K/V rows"
          % ("draft KV per token", b.draft_kv_bytes_per_tok, b.draft_kv_bytes_per_tok / GiB))
    print("  %-30s %12.0f %10.4f  [h_v,d_k,dv]=[%d,%d,%d] x %d layers x %d bits (mamba_ssm_dtype)"
          % ("GDN recurrent state / sequence", b.rec_bytes_layer * b.n_lin,
             b.rec_bytes_layer * b.n_lin / GiB, b.lv, b.dk, b.dv, b.n_lin, b.state_dtype_bits))
    print("  %-30s %12.0f %10.6f  conv_dim %d x (k_conv %d - 1) x %d layers x 16 bits"
          % ("GDN conv window / sequence", b.conv_bytes_layer * b.n_lin,
             b.conv_bytes_layer * b.n_lin / GiB, b.conv_dim, b.kf, b.n_lin))
    print("  %-30s %12.0f %10.6f  vocab %d x f32, target + draft buffer"
          % ("logits buffers", b.other_bytes(1), b.other_bytes(1) / GiB, b.V))
    print("  %-30s %12.0f %10.4f  all-token families + %d/%d of the routed experts"
          % ("per-token read", b.per_token_bytes(), b.per_token_bytes() / GiB, b.k, b.E))
    pt_gb = b.per_token_bytes() / 1e9
    print("  %-30s %s tok/s at the repo's 237 / 210 / 190 GB/s — ceilings, not forecasts"
          % ("  decode ceiling from that", " / ".join("%.0f" % (r / pt_gb) for r in READ_GBS)))
    st = b.rec_bytes_layer * b.n_lin
    print("  %-30s %10.1f MiB/token  the recurrent state is read and rewritten every token: "
          "%.4f GiB out + %.4f GiB back in. Bytes, not time — a mixed read/write rate has never been "
          "measured here (devices/measured-facts.md §Memory bandwidth) and the chunked prefill form "
          "moves this number."
          % ("GDN state traffic per token", 2 * st / MiB, st / GiB, st / GiB))


def print_table(b, ctxs, seqs):
    print("\nbudget: %d sequence(s), KV as %s. Two device budgets, both from measured facts:"
          % (seqs, KV_TYPES[b.kv_type]))
    print("  co-resident with halogen  %.2f .. %.2f GiB  (pool %.1f-%.1f minus flash_serve %.2f-%.2f)"
          % (CO_LO, CO_HI, POOL_LO, POOL_HI, HALOGEN_LO, HALOGEN_HI))
    print("  engine owns the box       >= %.2f GiB     (D-11, not yet measured with halogen down)\n"
          % DEDICATED_LO)
    hdr = ("%-7s %8s %8s %9s %9s %7s %9s %10s  %s"
           % ("context", "weights", "draft", "KV", "GDN st", "other", "TOTAL", "headroom", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    worst = 0
    for ctx in ctxs:
        r = b.row(ctx, seqs)
        head = CO_LO - r["total"] / GiB
        print("%-7s %8.3f %8.3f %9.4f %9.4f %7.4f %9.3f %+10.3f  %s"
              % (fmt_ctx(ctx), r["main"] / GiB, r["draft"] / GiB, r["kv"] / GiB, r["state"] / GiB,
                 r["other"] / GiB, r["total"] / GiB, head, r["verdict"]))
        worst = max(worst, 1 if r["verdict"] != "FITS co-resident" else 0)
        worst = max(worst, 2 if r["verdict"] == "DOES NOT FIT" else 0)
    print("headroom is measured against the PESSIMISTIC co-resident budget %.2f GiB, not the round 32."
          % CO_LO)
    # Where the wall actually is, in the only axis that moves: context. Weights are shared between
    # sequences, KV and GDN state are not, so this is the row an allocator has to respect.
    per_tok = (b.kv_bytes_per_tok + b.draft_kv_bytes_per_tok) * seqs
    fixed = sum(b.weight_bytes()) + b.state_bytes(seqs) + b.other_bytes(seqs)
    room = CO_LO * GiB - fixed
    max_ctx = int(room // per_tok) if per_tok else 0
    print("largest context that still fits that budget at %d sequence(s): %s tokens (%s), weights + "
          "state fixed at %.3f GiB, %d B of KV+draft rows per token across the %d lane(s). The "
          "checkpoint's own window is %d tokens."
          % (seqs, "{:,}".format(max_ctx), fmt_ctx(max_ctx // 1024 * 1024), fixed / GiB, per_tok,
             seqs, b.max_ctx))
    return worst


def print_frontier(b, ctxs, seq_list):
    print("\nwhere it stops fitting: sequences x context, verdict only (co-resident budget %.2f GiB)"
          % CO_LO)
    print("%-9s%s" % ("", "".join("%-34s" % fmt_ctx(c) for c in ctxs)))
    for n in seq_list:
        cells = []
        for ctx in ctxs:
            r = b.row(ctx, n)
            short = {"FITS co-resident": "fits", "FITS co-resident only in the lucky half of the bracket":
                     "fits if pool is high", "FITS only with halogen down": "HALOGEN-DOWN-ONLY",
                     "DOES NOT FIT": "DOES NOT FIT"}[r["verdict"]]
            cells.append("%-34s" % ("%s (%.2f GiB)" % (short, r["total"] / GiB)))
        print("seqs=%-4d%s" % (n, "".join(cells)))
    print("the state pool and KV are per-sequence; weights are not, which is why concurrency is the")
    print("lever that crosses the line, not context.")


def print_give_up(b, ctx, seqs):
    r = b.row(ctx, seqs)
    over = r["total"] / GiB - CO_LO
    print("\nwhat to give up, cheapest first (row %s x %d seq(s): %.3f GiB, %.3f GiB %s the "
          "co-resident budget)" % (fmt_ctx(ctx), seqs, r["total"] / GiB, abs(over),
                                   "over" if over > 0 else "under"))
    opts = []
    for alt in ("q8_0", "q4_k", "q4_0", "iq4_xs"):
        if alt == b.kv_type:
            continue
        nb = Budget(b.src, b.policy_name, alt)
        kv_now = (b.kv_bytes_per_tok + b.draft_kv_bytes_per_tok) * ctx * seqs
        kv_alt = (nb.kv_bytes_per_tok + nb.draft_kv_bytes_per_tok) * ctx * seqs
        opts.append(("KV container %s -> %s" % (b.kv_type, alt), (kv_now - kv_alt) / GiB, "USER",
                     "width choice, same shape as the D-01/D-01a pattern; %s is %g bpw"
                     % (KV_TYPES[alt], GBG.bpw_of(KV_TYPES[alt]))))
    for half_ctx in (ctx // 2, ctx // 4):
        if half_ctx >= 1024:
            saving = (b.kv_bytes_per_tok + b.draft_kv_bytes_per_tok) * (ctx - half_ctx) * seqs
            opts.append(("context %s -> %s" % (fmt_ctx(ctx), fmt_ctx(half_ctx)), saving / GiB, "USER",
                         "capability cut, not an engineering one"))
    per_layer_bytes = (b.census.get("routed_experts", 0) / (b.L + b.mtp_layers)
                       * b.width_bits("routed_experts") / 8.0)
    streamed = per_layer_bytes * b.k / b.E
    opts.append(("stream one layer's routed experts instead of pinning them",
                 per_layer_bytes / GiB, "ENGINEERING",
                 "saves %.4f GiB per layer de-pinned. The same bytes do not get cheaper: %.4f GiB/token "
                 "of expert traffic would arrive through a %.0f-%.0f GB/s host->device copy instead of "
                 "a %.0f GB/s resident read, ~%.1fx slower for that traffic. Capacity lever, not a "
                 "speed lever."
                 % (per_layer_bytes / GiB, streamed / GiB, STAGING_GBS[0], STAGING_GBS[1],
                    READ_GBS[0], READ_GBS[0] / ((STAGING_GBS[0] + STAGING_GBS[1]) / 2))))
    opts.append(("drop the MTP draft block", r["draft"] / GiB, "USER",
                 "D-05 has MTP in scope; this reverses a user decision, it is not ours to spend"))
    opts.append(("MTP draft state + its KV rows", b.draft_kv_bytes_per_tok * ctx * seqs / GiB,
                 "ENGINEERING", "draft layer's rows only, keeps the draft weights"))
    opts.append(("one fewer concurrent sequence",
                 ((b.kv_bytes_per_tok + b.draft_kv_bytes_per_tok) * ctx
                  + geo_state_per_seq(b) + b.other_bytes(1)) / GiB, "OPS/USER",
                 "per-sequence marginal cost at %s context" % fmt_ctx(ctx)))
    for alt, what in (("G lm_head Q6", "lm_head + embeddings at Q6_K (policy G, D-10 not taken)"),
                      ("E attn Q4_K", "attention + shared expert at Q4_K (policy E, D-01a not taken)")):
        nb = Budget(b.src, alt)
        d = (sum(nb.weight_bytes()) - sum(b.weight_bytes())) / GiB
        if d >= 0:
            continue
        opts.append(("policy %s: %s" % (alt.split()[0], what), d, "USER",
                     "a width assignment is the user's, per the D-01/D-01a/D-10 pattern"))
    opts.append(("halogen down (budget, not bytes)", (POOL_LO - CO_LO), "OPS",
                 "raises the budget from %.2f to %.2f GiB by removing the %.2f-%.2f GiB co-resident "
                 "footprint; D-11, and the pool figure for that case is still UNMEASURED"
                 % (CO_LO, POOL_LO, HALOGEN_LO, HALOGEN_HI)))
    opts.sort(key=lambda o: -o[1])
    print("  %-46s %9s %-11s %s" % ("lever", "saves GiB", "whose call", "what it really costs"))
    print("  " + "-" * 118)
    for name, save, who, why in opts:
        print("  %-46s %+9.4f %-11s %s" % (name[:46], save, who, why))
    print("  ENGINEERING = this repo can do it without asking; USER = a width/scope decision that")
    print("  follows the D-01/D-01a/D-10 pattern and needs the user's words; OPS = who is resident.")


def fmt_ctx(c):
    return ("%dk" % (c // 1024)) if c % 1024 == 0 else str(c)


# -----------------------------------------------------------------------------------------------
def selftest(artifact_required):
    print("selftest: witnesses first, then injected faults. A fault nobody notices is a failure.")
    src = Sources(want_artifact=True)
    b = Budget(src)
    ws = witnesses(src, b, artifact_required) + assembly_witnesses(b)
    failing = [w for w in ws if not w[1]]
    for name, ok, detail in ws:
        print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    print("  %d/%d witnesses pass" % (len(ws) - len(failing), len(ws)))
    rc = 0 if not failing else 3
    print("\ninjections: each row perturbs exactly one number through the fault hook and must be")
    print("caught by at least one witness that does not read the perturbed value.\n")
    baseline_bad = {w[0] for w in failing}
    for code, what in FAULTS.items():
        if code == "none":
            continue
        fb = Budget(src, fault=code)
        fws = witnesses(src, fb, artifact_required) + assembly_witnesses(fb)
        caught = [w for w in fws if not w[1] and w[0] not in baseline_bad]
        r0, r1 = b.row(8192), fb.row(8192)
        drift = (r1["total"] - r0["total"]) / GiB
        tok = (r1["per_token"] - r0["per_token"]) / GiB
        if caught:
            print("  FIRED   %-24s (%+.4f GiB total, %+.4f GiB/token) by: %s"
                  % (code, drift, tok, "; ".join(
                      "%s -> %s" % (c[0], c[2][:110]) for c in caught[:2])))
        else:
            print("  MISSED  %-24s (%+.4f GiB total, %+.4f GiB/token) NOTHING CAUGHT IT"
                  % (code, drift, tok))
            rc = 3
    print("\nselftest: %s" % ("PASS" if rc == 0 else "FAIL"))
    return rc


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--ctx", default="4k,16k,32k,64k,128k,256k")
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--policy", default="A (DECIDED)", choices=sorted(POLICIES))
    ap.add_argument("--kv-type", default="bf16", choices=sorted(KV_TYPES))
    ap.add_argument("--give-up", action="store_true", help="print the savings ladder for the last row")
    ap.add_argument("--frontier", action="store_true", help="print the seqs x context verdict grid")
    ap.add_argument("--report", action="store_true", help="exit 0 even when a row does not fit")
    ap.add_argument("--no-artifact", action="store_true", help="do not read the pinned GGUF header")
    ap.add_argument("--allow-missing-artifact", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fault", default="none", choices=sorted(FAULTS))
    args = ap.parse_args()
    if args.selftest:
        return selftest(not args.allow_missing_artifact)
    src = Sources(want_artifact=not args.no_artifact)
    b = Budget(src, args.policy, args.kv_type, args.fault)
    ctxs = [parse_ctx(x) for x in args.ctx.split(",")]
    print("residency budget — %s, KV as %s, %d sequence(s)" % (args.policy, KV_TYPES[args.kv_type],
                                                               args.seqs))
    print("sources: config %s" % os.path.relpath(CFG_PATH, REPO))
    print("         index %s (%d text tensor names, %d vision names excluded per D-04)"
          % (os.path.relpath(IDX_PATH, REPO), len(src.names), len(src.vision_names)))
    print("         pin   %s (MTP authority %s elements)"
          % (os.path.relpath(PIN_PATH, REPO), src.pin["authority"]["mtp_elements"]))
    if args.fault != "none":
        print("         FAULT INJECTED: %s — %s" % (args.fault, FAULTS[args.fault]))
    print("")
    print_components(b)
    worst = print_table(b, ctxs, args.seqs)
    if args.frontier:
        print_frontier(b, [c for c in ctxs], [1, 2, 4, 8])
    if args.give_up:
        print_give_up(b, ctxs[-1], args.seqs)
    print("\nwitnesses (a budget table whose witnesses failed is not a budget table):")
    ws = witnesses(src, b, not args.allow_missing_artifact)
    for name, ok, detail in ws:
        print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    if any(not w[1] for w in ws):
        return 3
    if worst and not args.report:
        print("\nexit %d: at least one row does not fit the co-resident budget (see verdict column)"
              % worst)
        return worst
    return 0


def parse_ctx(s):
    s = s.strip().lower()
    mult = 1024 * 1024 if s.endswith("m") else (1024 if s.endswith("k") else 1)
    return int(float(s.rstrip("km")) * mult)


if __name__ == "__main__":
    raise SystemExit(main())
