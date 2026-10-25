#!/usr/bin/env python3
"""audit_published_quant.py — audit `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` against the HF bf16
checkpoint, on a sample, with our own reader. PLAN.md step 1.0-1.6.

Why this gate exists (PLAN.md §"Why an audit exists at all"): llama.cpp and our engine read the SAME
bytes, so a converter bug — wrong scale order, transposed expert stack, a router row that landed in
the wrong layer — makes both agree and makes every gate green. The only witness available is the
authority: the HF bf16 tensors. So the comparison here is GGUF-dequantized-by-our-own-reader versus
the checkpoint, never versus llama.cpp.

Design rules this file is built to (`pitfalls/40-gates-and-admissions.md`, `pitfalls/30-numerics.md`):
  * every section asserts its own parsed-record count against a count derived from geometry, and a
    section that parses zero cases FAILS rather than printing an empty table;
  * must-catch and must-not-catch sets are derived (from expert count 256, moe_intermediate 512,
    hidden 2048, 40 layers + 1 MTP layer, vocab 248320), never hand-listed;
  * the reader under test is never its own reference: the F32 families judge the loader, the bf16
    originals judge the dequant, and the floor probe is a separate quantizer whose only job is to
    size a band;
  * bands come from a floor probe run BEFORE any value is judged, and a family that falls outside a
    band is a finding, not something to accommodate;
  * the gate refuses to read a tensor-data byte until its own `--selftest` has passed, and records
    the ordering (`quantization/evidence/audit-ordering-2026-10-26.log`).

Usage (one section at a time, in this order):
  python3 tools/audit_published_quant.py --selftest
  python3 tools/audit_published_quant.py --hash
  python3 tools/audit_published_quant.py --shards [--no-shard-sha]
  python3 tools/audit_published_quant.py --structure
  python3 tools/audit_published_quant.py --exact
  python3 tools/audit_published_quant.py --floor-probe
  python3 tools/audit_published_quant.py --dequant
  python3 tools/audit_published_quant.py --shared
  python3 tools/audit_published_quant.py --summary
Everything is CPU-only, prints no timing, and writes its raw output under
quantization/evidence/audit-*.

Interpreter of record: system python3 (3.12), stdlib only. No numpy, no torch, no llama.cpp.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import struct
import sys
from decimal import Decimal, localcontext

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gguf_reader as gr        # noqa: E402
import ggml_block_geom as GBG   # noqa: E402  — width/block geometry read from the pinned ggml header

DAY = "2026-10-26"
# The day the supervisor authorised splitting 1.4 into a copied pass and a route predicate, and asked
# for it to be surfaced to the user for ratification. Named here so the summary can attribute the
# decision instead of implying the gate always said this.
DECISION_DAY = "2026-10-26"
GGUF_PATH = os.environ.get("Q36_GGUF", os.path.join(
    REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"))
GGUF_REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"   # unsloth/Qwen3.6-35B-A3B-GGUF
GGUF_HF_DIR = os.path.join(REPO, "models/unsloth-q4km")
SHARD_DIR = os.environ.get("Q36_SHARDS", os.path.join(REPO, "models/qwen36-src"))
SHARD_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"  # Qwen/Qwen3.6-35B-A3B
SHARDS = ["model-00001-of-00026.safetensors", "model-00002-of-00026.safetensors",
          "model-00025-of-00026.safetensors", "model-00026-of-00026.safetensors"]
MTP_OUT = os.path.join(REPO, "models/w1-mtp-extract/mtp.safetensors")
CFG = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")
IDX = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json")
EV = os.path.join(REPO, "quantization/evidence")
SELFTEST_MARK = os.path.join(EV, "audit-selftest-%s.json" % DAY)
ORDER_LOG = os.path.join(EV, "audit-ordering-%s.log" % DAY)
FLOOR_JSON = os.path.join(EV, "audit-floor-probe-%s.json" % DAY)
STRUCT_JSON = os.path.join(EV, "audit-structure-%s.json" % DAY)
EXACT_JSON = os.path.join(EV, "audit-exactness-%s.json" % DAY)
DEQ_JSON = os.path.join(EV, "audit-dequant-%s.json" % DAY)
HASH_JSON = os.path.join(EV, "audit-hash-%s.json" % DAY)
SHARD_JSON = os.path.join(EV, "audit-shards-%s.json" % DAY)
SHARED_JSON = os.path.join(EV, "audit-shared-%s.json" % DAY)
INJ_JSON = os.path.join(EV, "audit-injections-%s.json" % DAY)
SUMMARY_MD = os.path.join(EV, "audit-published-quant-%s.md" % DAY)
PIN_JSON = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")

SAMPLE_LAYERS = [0, 1, 39]      # PLAN.md step 1: 0 and 39 exercise the producer's upgraded path,
                                # 1 the baseline path (first 1/8, last 1/8, every third).
MTP_CLAIM_ELTS = 41943040       # PLAN.md step 1 fetch list: the figure to witness (or refute).
EXPECTED_TENSOR_RECORDS = 733   # from the header range-read already committed as evidence.

# Bands. Derived, not fitted: the floor probe (own quantizer, same grid, same data) is the floor;
# the margin below is the one taste-number in this file and it is justified by the injection table,
# which prints how far OUT the injected misreadings land. A family whose published error exceeds
# margin x floor is reported, never accommodated (AGENTS.md: never loosen a gate to make it pass).
RMS_MARGIN = 1.5
MAX_MARGIN = 3.0

SELFTEST_TOKEN = "audited-by-a-selftest-that-passed"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_order(msg):
    os.makedirs(EV, exist_ok=True)
    with open(ORDER_LOG, "a") as fh:
        fh.write("%s | tool | %s\n" % (now(), msg))


def sha256_file(path, chunk=1 << 23):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_download_meta(path, fname):
    """huggingface_hub stores <file>.metadata next to the blob: line1 = commit/revision,
    line2 = etag (for an LFS file that etag IS the sha256), line3 = fetch time."""
    p = os.path.join(path, ".cache/huggingface/download", fname + ".metadata")
    if not os.path.exists(p):
        return None
    lines = open(p).read().splitlines()
    return {"revision": lines[0] if lines else None,
            "etag": lines[1] if len(lines) > 1 else None,
            "fetched_at_epoch": lines[2] if len(lines) > 2 else None,
            "path": p}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def table(header, rows, out=sys.stdout):
    if not rows:
        # A check that can produce no output cannot fail (pitfalls/40). Loud, not empty.
        print("!! NO ROWS for table %s — that is a FAILURE, not an empty result" % (header[0],))
        return False
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join("%%-%ds" % w for w in widths)
    print(fmt % tuple(header))
    for r in rows:
        print(fmt % tuple(str(x) for x in r))
    return len(rows) == len(rows) and len(rows) > 0


def nines(n):
    return "{:,}".format(n)


# --------------------------------------------------------------------------------------------
# geometry, derived from checkpoints/ — nothing here is typed from memory
# --------------------------------------------------------------------------------------------
class Geo:
    def __init__(self):
        c = json.load(open(CFG))
        t = c["text_config"]
        self.H = t["hidden_size"]
        self.E = t["num_experts"]
        self.K = t["num_experts_per_tok"]
        self.I = t["moe_intermediate_size"]
        self.Ise = t["shared_expert_intermediate_size"]
        self.L = t["num_hidden_layers"]
        self.V = t["vocab_size"]
        self.iv = t["full_attention_interval"]
        self.nq = t["num_attention_heads"]
        self.nkv = t["num_key_value_heads"]
        self.hd = t["head_dim"]
        # attn_output_gate=True: q_proj emits the query rows AND the output-gate rows, so the row
        # count is 2*nq*hd. The converter maps self_attn.q_proj -> ATTN_Q unchanged, so the GGUF
        # tensor carries both halves in HF's own order; a reader that assumes nq*hd rows reads half
        # the tensor and never notices.
        self.attn_out_gate = bool(t.get("attn_output_gate", False))
        self.q_rows = self.nq * self.hd * (2 if self.attn_out_gate else 1)
        self.nk = t["linear_num_key_heads"]
        self.nv = t["linear_num_value_heads"]
        self.dk = t["linear_key_head_dim"]
        self.dv = t["linear_value_head_dim"]
        self.kf = t["linear_conv_kernel_dim"]
        self.mtp_layers = t["mtp_num_hidden_layers"]
        self.layer_types = t["layer_types"]
        self.mlp_gate = [k for k, v in self.layer_types.items()] if isinstance(
            self.layer_types, dict) else self.layer_types
        self.n_full = self.L // self.iv
        self.n_lin = self.L - self.n_full
        self.gdn_q_dim = self.nk * self.dk
        self.gdn_k_dim = self.nkv_gdn = self.nk * self.dk
        self.gdn_v_dim = self.nv * self.dv
        self.gdn_qkv_rows = self.gdn_q_dim + self.gdn_k_dim + self.gdn_v_dim
        self.nv_per_k = self.nv // self.nk

    def is_linear(self, layer):
        return self.layer_types[layer] == "linear_attention"

    def hf_shape_of(self, hf_name, layer=None):
        """Expected HF (slowest-first) shape for a checkpoint tensor name, derived from geometry.
        The structure pass uses this for the 37 layers whose shard bytes were not fetched."""
        H, I, E = self.H, self.I, self.E
        if hf_name.endswith("embed_tokens.weight"):
            return [self.V, H]
        if hf_name == "lm_head.weight":
            return [self.V, H]
        if hf_name.endswith("norm.weight") and ".layers." not in hf_name and "mtp." not in hf_name:
            return [H]
        s = {
            "input_layernorm.weight": [H],
            "post_attention_layernorm.weight": [H],
            "mlp.gate.weight": [E, H],
            "mlp.experts.gate_up_proj": [E, 2 * I, H],
            "mlp.experts.down_proj": [E, H, I],
            "mlp.shared_expert.gate_proj.weight": [self.Ise, H],
            "mlp.shared_expert.up_proj.weight": [self.Ise, H],
            "mlp.shared_expert.down_proj.weight": [H, self.Ise],
            "mlp.shared_expert_gate.weight": [1, H],
            "self_attn.q_proj.weight": [self.q_rows, H],
            "self_attn.k_proj.weight": [self.nkv * self.hd, H],
            "self_attn.v_proj.weight": [self.nkv * self.hd, H],
            "self_attn.o_proj.weight": [H, self.nq * self.hd],
            "self_attn.q_norm.weight": [self.hd],
            "self_attn.k_norm.weight": [self.hd],
            "linear_attn.in_proj_qkv.weight": [self.gdn_qkv_rows, H],
            "linear_attn.in_proj_z.weight": [self.gdn_v_dim, H],
            "linear_attn.in_proj_a.weight": [self.nv, H],
            "linear_attn.in_proj_b.weight": [self.nv, H],
            "linear_attn.out_proj.weight": [H, self.gdn_v_dim],
            "linear_attn.conv1d.weight": [self.gdn_qkv_rows, 1, self.kf],
            "linear_attn.norm.weight": [self.dv],
            "linear_attn.A_log": [self.nv],
            "linear_attn.dt_bias": [self.nv],
        }
        for suf, shape in s.items():
            if hf_name.endswith(suf):
                return list(shape)
        return None


# --------------------------------------------------------------------------------------------
# The HF -> GGUF name map, and the element-position map.
#
# Names come from the committed checkpoint census (checkpoints/*.index.json) and the GGUF header;
# the value transforms come from the pinned converter's own documented behaviour, read out of
# engines/llama.cpp (b11869) conversion/qwen.py:
#   * `mlp.experts.gate_up_proj` [E,2I,H] splits on dim=-2: gate = [...,:I,:], up = [...,I:,:]
#     (conversion/qwen.py, Qwen3NextModel.modify_tensors).
#   * `_LinearAttentionVReorderBase.modify_tensors` reorders the V heads of linear_attn tensors
#     from HF grouped-by-K order to ggml tiled order, because num_k_heads(16) != num_v_heads(32).
#     The audit models it explicitly and reports which order the published file actually carries:
#     a producer that converted with an older llama.cpp would have left it grouped, and a runtime
#     that assumes tiled would then be silently wrong. That is a finding, not a tolerance.
#   * `Qwen3NextModel.modify_tensors` also changes VALUES, not just names, and any exactness claim
#     that ignored it would report a clean file as corrupt:
#       - `*norm.weight` (except `linear_attn.norm.weight`) is stored as HF + 1;
#       - `linear_attn.A_log` is stored as -exp(A_log);
#       - `linear_attn.dt_bias` is renamed (`...dt_proj.bias` internally, `ssm_dt.bias` in the file);
#       - `linear_attn.conv1d.weight` is `squeeze()`d, so [8192,1,4] becomes [8192,4].
#     `model.norm.weight` ends in `norm.weight` too, so `output_norm` carries the +1 as well. Which
#     width that arithmetic happened in is a property of the file, not of the source text above, so
#     1.4 tries the transform in both f32 and bf16 and REPORTS which one is bit-exact.
#   * Names come from gguf-py/gguf/constants.py §MODEL_TENSOR_NAMES for the QWEN35MOE template
#     (§GGUF_TEMPLATE_LLM[QWEN35MOE]): `post_attention_norm` (ATTN_POST_NORM), `ssm_a` (SSM_A, no
#     `.weight`), `ssm_dt.bias` (SSM_DT + the bias suffix). The template lists no NEXTN_* tensor
#     that this checkpoint could fill, and the file carries none — the draft block is absent.
# --------------------------------------------------------------------------------------------
HF_TO_GGUF = [
    (r"^model\.language_model\.embed_tokens\.weight$", "token_embd.weight", None),
    (r"^lm_head\.weight$", "output.weight", None),
    (r"^model\.language_model\.norm\.weight$", "output_norm.weight", None),
]
LAYER_MAP = [
    (r"^input_layernorm\.weight$", "attn_norm.weight", None),
    (r"^post_attention_layernorm\.weight$", "post_attention_norm.weight", None),
    (r"^self_attn\.q_proj\.weight$", "attn_q.weight", None),
    (r"^self_attn\.k_proj\.weight$", "attn_k.weight", None),
    (r"^self_attn\.v_proj\.weight$", "attn_v.weight", None),
    (r"^self_attn\.o_proj\.weight$", "attn_output.weight", None),
    (r"^self_attn\.q_norm\.weight$", "attn_q_norm.weight", None),
    (r"^self_attn\.k_norm\.weight$", "attn_k_norm.weight", None),
    (r"^linear_attn\.in_proj_qkv\.weight$", "attn_qkv.weight", "v_qkv"),
    (r"^linear_attn\.in_proj_z\.weight$", "attn_gate.weight", "v_rows"),
    (r"^linear_attn\.in_proj_a\.weight$", "ssm_alpha.weight", "v_rows_h1"),
    (r"^linear_attn\.in_proj_b\.weight$", "ssm_beta.weight", "v_rows_h1"),
    (r"^linear_attn\.A_log$", "ssm_a", "v_1d_h1"),
    (r"^linear_attn\.dt_bias$", "ssm_dt.bias", "v_1d_h1"),
    (r"^linear_attn\.conv1d\.weight$", "ssm_conv1d.weight", "v_conv"),
    (r"^linear_attn\.norm\.weight$", "ssm_norm.weight", None),
    (r"^linear_attn\.out_proj\.weight$", "ssm_out.weight", "v_cols"),
    (r"^mlp\.gate\.weight$", "ffn_gate_inp.weight", None),
    (r"^mlp\.shared_expert_gate\.weight$", "ffn_gate_inp_shexp.weight", None),
    (r"^mlp\.shared_expert\.gate_proj\.weight$", "ffn_gate_shexp.weight", None),
    (r"^mlp\.shared_expert\.up_proj\.weight$", "ffn_up_shexp.weight", None),
    (r"^mlp\.shared_expert\.down_proj\.weight$", "ffn_down_shexp.weight", None),
    (r"^mlp\.experts\.down_proj$", "ffn_down_exps.weight", None),
]
# The fused expert pair is a 1-to-2 map, handled apart from the table above.
FUSED_GATE_UP = ("mlp.experts.gate_up_proj",
                 [("ffn_gate_exps.weight", 0), ("ffn_up_exps.weight", 1)])


def hf_layer_prefix(layer):
    return "model.language_model.layers.%d." % layer


def hf_to_gguf_names(hf_name):
    """-> list of (gguf_name_or_None, transform, extra) for one HF tensor name."""
    if hf_name.startswith("mtp."):
        return [(None, "absent-mtp", None)]
    if hf_name.startswith("model.visual.") or hf_name.startswith("visual."):
        return [(None, "absent-vision", None)]
    for pat, gg, tr in HF_TO_GGUF:
        if re.match(pat, hf_name):
            return [(gg, tr, None)]
    m = re.match(r"^model\.language_model\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        return []
    layer, rest = int(m.group(1)), m.group(2)
    out = []
    for pat, gg, tr in LAYER_MAP:
        if re.match(pat, rest):
            out.append(("blk.%d.%s" % (layer, gg), tr, None))
            return out
    if rest == FUSED_GATE_UP[0]:
        for gg, half in FUSED_GATE_UP[1]:
            out.append(("blk.%d.%s" % (layer, gg), "gate_up_split", half))
        return out
    return out


def gguf_family(name):
    """Family label used for counting and banding. Derived from the name pattern, not a table."""
    n = re.sub(r"\.(\d+)\.", ".N.", name)
    if "ffn_gate_inp" in name:
        return "router+shexp-gate"
    if "ffn_gate_exps" in name or "ffn_up_exps" in name:
        return "routed gate/up"
    if "ffn_down_exps" in name:
        return "routed down"
    if "shexp" in name:
        return "shared expert"
    if "attn_qkv" in name:
        return "GDN in_proj_qkv"
    if "attn_gate" in name:
        return "GDN z-gate"
    if "ssm_out" in name:
        return "GDN out_proj"
    if re.search(r"ssm_", name):
        return "GDN control"
    if re.search(r"attn_(q|k|v|output)\.", name):
        return "full attention"
    if name.startswith("token_embd"):
        return "embeddings"
    if name.startswith("output"):
        return "lm_head"
    if re.search(r"_norm\.|^norm\.", n):
        return "norms"
    return "other: " + n


# --------------------------------------------------------------------------------------------
# safetensors: header + element reads (the authority side). Reading a safetensors JSON header is
# metadata; reading its data section is gated by the same selftest token as the GGUF.
# --------------------------------------------------------------------------------------------
class Authority:
    """The bf16 checkpoint shards we have, as an element-addressed reader."""

    def __init__(self, token):
        self.token = token
        self.shards = {}
        for sh in SHARDS:
            p = os.path.join(SHARD_DIR, sh)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            meta = {}
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                meta[k] = (v["dtype"], tuple(v["shape"]), v["data_offsets"][0],
                           v["data_offsets"][1])
            self.shards[sh] = {"path": p, "tensors": meta, "fh": None, "json_len": n}
        self.index = {}
        for sh, d in self.shards.items():
            for k in d["tensors"]:
                self.index[k] = (sh,) + d["tensors"][k]
        self.shard_names = {sh: set(d["tensors"]) for sh, d in self.shards.items()}

    def close(self):
        for d in self.shards.values():
            if d["fh"] is not None:
                d["fh"].close()
                d["fh"] = None

    def has(self, name):
        return name in self.index

    def info(self, name):
        sh, dtype, shape, s0, s1 = self.index[name]
        nel = 1
        for x in shape:
            nel *= int(x)
        return {"shard": sh, "dtype": dtype, "shape": shape, "n_elts": nel,
                "bytes": s1 - s0, "start": s0}

    def _fh(self, shard):
        d = self.shards[shard]
        if d["fh"] is None:
            d["fh"] = open(d["path"], "rb")
        return d["fh"]

    def elements(self, name, indices):
        """-> list[float] of the given element indices (any order, duplicates allowed).

        The safetensors data section starts right after the 8-byte length prefix plus the JSON
        header; one window is read per element run, so the cost does not depend on the length of
        the index list but on how contiguous it is.
        """
        info = self.info(name)
        assert info["dtype"] == "BF16", "%s is %s, this audit assumes the authority is bf16" % (
            name, info["dtype"])
        base = 8 + self.shards[info["shard"]]["json_len"] + info["start"]
        fh = self._fh(info["shard"])
        idx = list(indices)
        order = sorted(range(len(idx)), key=lambda i: idx[i])
        vals = [None] * len(idx)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and idx[order[j + 1]] == idx[order[j]] + 1:
                j += 1
            lo, hi = idx[order[i]], idx[order[j]]
            fh.seek(base + 2 * lo)
            raw = fh.read(2 * (hi - lo + 1))
            got = gr.bf16_to_floats(raw)
            for k, v in enumerate(got):
                vals[order[i + k]] = v
            i = j + 1
        return vals

    def raw_bytes(self, name, start_elt, n_elts):
        info = self.info(name)
        fh = self._fh(info["shard"])
        fh.seek(8 + self.shards[info["shard"]]["json_len"] + info["start"] + 2 * start_elt)
        return fh.read(2 * n_elts)


# --------------------------------------------------------------------------------------------
# element-position mapping: GGUF linear index -> HF linear index (both row-major; GGUF dims are
# the reverse of the HF dims, so the plain case is the identity and everything else is a named,
# documented transform).
# --------------------------------------------------------------------------------------------
def v_tiled_to_grouped(i, nk, nv_per_k, hd):
    """Given a GGUF (tiled) index, return the HF (grouped) index the converter took it from.

    `_reorder_v_heads` (conversion/qwen.py) does `reshape(nk, v, d)` on the HF tensor, swaps the nk
    and v axes, and flattens again. So the GGUF element at p = (v*nk + k)*hd + d holds the HF
    element at (k*nv_per_k + v)*hd + d — V is the OUTER axis in the file, K the inner one. Getting
    this backwards is invisible for a 1-D head list only by accident, and this repo's first version
    had it backwards (it was caught by the exactness pass on `ssm_a`, not by the first witness).
    """
    v = i // (nk * hd)
    rem = i % (nk * hd)
    k = rem // hd
    d = rem % hd
    return (k * nv_per_k + v) * hd + d


def v_grouped_order(nk, nv_per_k, hd):
    """The reorder, spelled the way the converter spells it: nested loops over (k, v, d) writing at
    (v*nk + k)*hd + d, reading the HF element (k*nv_per_k + v)*hd + d. Independent formulation of
    `v_tiled_to_grouped`, so the witness is not the mapper checking itself."""
    out = {}
    for k in range(nk):
        for v in range(nv_per_k):
            for d in range(hd):
                out[(v * nk + k) * hd + d] = (k * nv_per_k + v) * hd + d
    return out


def make_mapper(geo, gguf_t, transform, extra=None, inj=None):
    """-> fn(gguf_linear_index) -> hf_linear_index, or None for the identity.

    Both files are row-major and GGUF dims are the reverse of the HF dims, so a straight copy maps a
    GGUF linear index to the SAME HF linear index; everything else here is a named, documented
    transform (fused gate/up split, the converter's V reorder) or a deliberate misreading used to
    prove the must-catch set fires on real data.
    """
    dims = gguf_t.dims                        # GGUF order: dims[0] is the fastest axis
    if inj == "gate_up_swap" and transform != "gate_up_split":
        # The swap means nothing on a tensor that is not the fused pair. "Means nothing" is a no-op,
        # NOT "drop the transform that does apply here": doing that made gate_up_swap appear to fire
        # on the GDN tensors, i.e. an injection the size of nothing at all looked like a fault.
        inj = None
    if inj == "expert_stack_transpose" and len(dims) >= 3:
        d0, d1, d2 = dims[0], dims[1], dims[2]

        def transpose_misread(g):
            i0 = g % d0
            r = g // d0
            i1 = r % d1
            i2 = r // d1
            # HF real dims are (E,N,K) with strides (N*K, K, 1). This builds the index as if the
            # axis order were (K,N,E) - the expert axis treated as trailing.
            return i0 * (d1 * d2) + i1 * d2 + i2
        return transpose_misread
    if transform == "gate_up_split":
        IH = geo.I * geo.H
        half = extra                          # 0 = gate (HF rows [0,I)), 1 = up (HF rows [I,2I))
        off = (1 - half) * IH if inj == "gate_up_swap" else half * IH

        def gs(g):
            return (g // IH) * (2 * IH) + (g % IH) + off
        return gs
    if inj == "gate_up_swap":
        return None                           # only meaningful on the fused pair

    nk, nvk, dv, H = geo.nk, geo.nv_per_k, geo.dv, geo.H
    if transform == "v_qkv":
        qk = geo.gdn_q_dim + geo.gdn_k_dim

        def mq(g):
            r = g // H
            if r < qk:
                return g
            c = g % H
            return (qk + v_tiled_to_grouped(r - qk, nk, nvk, dv)) * H + c
        return mq
    if transform == "v_rows":
        def mr(g):
            r = g // H
            c = g % H
            return v_tiled_to_grouped(r, nk, nvk, dv) * H + c
        return mr
    if transform == "v_rows_h1":
        def mr1(g):
            r = g // H
            c = g % H
            return v_tiled_to_grouped(r, nk, nvk, 1) * H + c
        return mr1
    if transform == "v_1d_h1":
        def m1(g):
            return v_tiled_to_grouped(g, nk, nvk, 1)
        return m1
    if transform == "v_conv":
        kf, qk = geo.kf, 2 * geo.gdn_q_dim

        def mc(g):
            c = g // kf
            if c < qk:
                return g
            t = g % kf
            return (qk + v_tiled_to_grouped(c - qk, nk, nvk, dv)) * kf + t
        return mc
    if transform == "v_cols":
        span = geo.gdn_v_dim

        def mcol(g):
            r = g // span
            c = g % span
            return r * span + v_tiled_to_grouped(c, nk, nvk, dv)
        return mcol
    if transform:
        raise NotImplementedError("no element map for transform %r" % (transform,))
    return None                               # None == identity


# --------------------------------------------------------------------------------------------
# sampling: which blocks of a GGUF tensor do we judge? Derived from the tensor's own dims, and for
# a rank>=3 expert stack stratified over the expert axis so the sample cannot be all-expert-0.
# --------------------------------------------------------------------------------------------
def even_spaced(n_total, n_want, include_ends=True):
    if n_want >= n_total:
        return list(range(n_total))
    if n_total <= 1:
        return [0]
    out = set()
    for i in range(n_want):
        out.add((i * n_total) // n_want)
    if include_ends:
        out.add(n_total - 1)
    return sorted(out)





def err_stats(deq, auth):
    """(n, max_abs, rms_diff, rms_auth, p999) plus a non-finite count. No timing, ever.

    The non-finite count exists because `NaN > band` is False: a dequant that produced garbage the
    float format cannot order would otherwise read as `inside the band` / `did not fire`, which is
    the `pitfalls` rule "a check that can produce no output cannot fail" wearing a different hat.
    """
    n = len(deq)
    if n == 0 or len(auth) != n:
        raise AssertionError("err_stats got dequant=%d authority=%d; a comparison that compares "
                             "nothing cannot fail" % (n, len(auth)))
    sse = 0.0
    ssa = 0.0
    mx = 0.0
    nonfinite = 0
    ad = []
    for u, v in zip(deq, auth):
        if not (math.isfinite(u) and math.isfinite(v)):
            nonfinite += 1
        d = u - v
        a = d if d >= 0 else -d
        if a > mx:
            mx = a
        sse += d * d
        ssa += v * v
        ad.append(a)
    ad.sort()
    i999 = int(0.999 * (n - 1))
    return {"n": n, "max_abs": mx, "rms_diff": math.sqrt(sse / n),
            "rms_auth": math.sqrt(ssa / n), "p999_abs": ad[i999], "n_nonfinite": nonfinite}


def stats_ok(st):
    """A comparison counts as a comparison only if it produced an orderable number for every
    element it claims to have compared."""
    return (st["n"] > 0 and st["n_nonfinite"] == 0 and math.isfinite(st["rms_diff"])
            and math.isfinite(st["max_abs"]))


# --------------------------------------------------------------------------------------------
# SELFTEST (PLAN.md 1.0). The gate is written from the spec; this is what proves the reader is the
# reader and not a wish.
#
# Triangle, deliberately three ways round:
#   * the READER (tools/gguf_reader.py) decodes bytes -> floats, transcribed from the C *dequant*
#     loops of ggml-quants.c;
#   * the PACKERS below encode intended quant indices -> bytes, transcribed from the C *quantize*
#     loops (quantize_row_q4_K_ref / _q5_K_ref / _q6_K_ref / _q8_0_ref) — a different function, a
#     different route through the same format definition;
#   * the ORACLE computes what each element must be from the intended (scale, min, index) triple and
#     the affine definition alone — it never looks at bytes at all.
# reader(bytes) == oracle(intended) is therefore a witness, not a tautology. And the oracle is not
# what the gate uses for the real comparison: there the reference is the bf16 checkpoint.
#
# Two of the five injections are INDEX misreadings (gate/up halves, expert-stack transpose), not
# byte misreadings, so they cannot fire on synthetic blocks alone; they get a structural witness
# here (the mapper must be the documented one, and the injected one must be a different permutation)
# and a value witness in the 1.5 section against the authority.
# --------------------------------------------------------------------------------------------
QK_K = 256


class Lcg:
    """Deterministic generator: the selftest must not depend on the interpreter's RNG version."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 16

    def u(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def f(self):
        return (self.next() % 2000001) / 1000000.0 - 1.0


def h16(x):
    return struct.pack("<e", x)


def f16v(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def i8v(b):
    return struct.unpack("<b", bytes([b & 0xFF]))[0]


def bf16_round(x):
    """f32 bits with the low 16 mantissa bits cleared = the bf16 value, widened back to f32."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bf16_stream(rng, n, scale=0.13):
    """n values that ARE exactly representable in bf16 — what a floor probe on a bf16 authority sees."""
    return [bf16_round(rng.f() * scale) for _ in range(n)]


VALUE_TR_LABEL = {"norm_plus_one": "norm+1", "neg_exp": "-exp"}


def value_transform(gguf_name):
    """Value the converter stores for a GGUF name, as a named transform of the HF value.

    Derived from conversion/qwen.py §Qwen3NextModel.modify_tensors: norms (every `*norm.weight`
    except `linear_attn.norm.weight`) are written as HF+1, and `A_log` as -exp(A_log). Everything
    else is stored as-is, only permuted or renamed. The width the producer did that arithmetic in is
    a property of the file, so it is measured (see `apply_value_transform`), not assumed.
    """
    if gguf_name.endswith(".ssm_a"):
        return "neg_exp"
    if gguf_name.endswith("_norm.weight") and not gguf_name.endswith("ssm_norm.weight"):
        return "norm_plus_one"
    return None


def f32_round(x):
    """round to nearest float32 (what a float32 tensor op would store)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_rne(x):
    """round to nearest-even bfloat16, widened back (what a bf16 tensor op would store)."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if (u & 0x7F800000) == 0x7F800000:          # inf / NaN: nothing to round
        return struct.unpack("<f", struct.pack("<I", u))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def apply_value_transform(tr, vals, arith="f32"):
    """The arithmetic the converter did, reproduced. `arith` says which width/route did it, because
    that is an empirical question about this file, not something to assume: `+1` on a bf16 tensor
    keeps 8 mantissa bits, `+1` in f32 keeps every bit of the widened original, and only one of
    them is bit-exact against what the file actually holds. `crl` (correctly rounded) exists for
    transcendental transforms: `math.exp` is a libm approximation, so a double-precision exp and a
    60-digit Decimal exp can disagree with each other, and only the correctly rounded one is a
    definition rather than an implementation.
    """
    rounder = bf16_rne if arith == "bf16" else f32_round
    if tr is None:
        return [f32_round(v) for v in vals]
    if tr == "norm_plus_one":
        return [rounder(v + 1.0) for v in vals]
    if tr == "neg_exp":
        if arith == "crl":
            with localcontext() as ctx:
                ctx.prec = 60
                return [f32_round(-float(Decimal(repr(v)).exp())) for v in vals]
        return [rounder(-math.exp(v)) for v in vals]
    raise NotImplementedError(tr)


# Which arithmetic routes each value transform is tried through. Derived: a plain additive transform
# has exactly two plausible widths, and both are bit-exact-or-not — that question belongs to 1.4a.
# `neg_exp` is deliberately NOT listed: a transcendental is not judged by trying widths until one
# fits, it is judged by the route predicate below.
ARITH_ROUTES = {"norm_plus_one": ("f32", "bf16")}


def _f32_ord(x):
    """f32 bit pattern mapped to a monotone unsigned integer (so distance == ulps)."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return (~u & 0xFFFFFFFF) if (u & 0x80000000) else (u | 0x80000000)


def ulp_dist(a, b):
    return abs(_f32_ord(a) - _f32_ord(b))


def f32_bits(x):
    """The bit pattern itself, so two values can be compared without trusting a decimal rendering."""
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


def f32_nextafter(x, y):
    """One representable f32 step from x toward y (C nextafterf, finite inputs)."""
    ox, oy = _f32_ord(x), _f32_ord(y)
    o = ox + 1 if oy > ox else (ox if oy == ox else ox - 1)
    # inverse of _f32_ord: the high half holds the POSITIVE floats (u | 0x80000000), the low half the
    # negatives (~u). Getting this backwards returns a plausible float that is nowhere near x, which
    # is exactly why f32_ulp has a witness in --selftest and is not trusted on sight.
    u = (o & 0x7FFFFFFF) if (o & 0x80000000) else (~o & 0xFFFFFFFF)
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f32_ulp(x):
    """The width of one representable step above |x|, from nextafter on x's own exponent.

    Reported for the reader. The PASS/FAIL bound itself is the neighbour set in judge_derived_route,
    because a step measured only upward is up to 2x the true one-hop distance just below a power of
    two — a bound has to be the tightest true statement, not the convenient one.
    """
    a = abs(x)
    if a == 0.0:
        return f32_nextafter(0.0, 1.0)
    return abs(f32_nextafter(a, float("inf")) - a)


# 1.4b exists because of one property of one tensor family. `ssm_a` is not stored by HF: HF computes
# A = -exp(A_log.float()) at module init, so a file's ssm_a is a RECOMPUTATION, and demanding it be
# bit-exact demands that the producer's libm agreed with ours — an unknowable property of an unknown
# build. It is not an engine-fidelity property either: this engine reads ssm_a from the file and never
# recomputes the exp. Zero tolerance against a recomputation route is a category error, not rigour.
#
# The predicate replaces tolerance with agreement between two independent implementations:
#   "f32"  exp computed in double (libm) and then rounded to f32;
#   "crl"  exp computed to 60 decimal digits and correctly rounded to f32 exactly once.
# They must agree with EACH OTHER bit-for-bit on every judged element; if they ever disagree the case
# FAILS, because a gate that absorbs a reference disagreement as "close enough" has stopped measuring
# anything. Only then may the file sit at most one f32 step (nextafter, on the agreed value's own
# exponent) from that agreed value. There is no ulp knob and no epsilon constant anywhere: the bound IS
# the spacing at that value, so changing the predicate means changing what it computes.
DERIVED_ROUTE = {"neg_exp": ("f32", "crl")}
DERIVED_ROUTE_LABEL = {"f32": "double exp, then round to f32",
                       "crl": "correctly-rounded exp (60 digits, one rounding)"}


def judge_derived_route(file_vals, refs):
    """Apply the 1.4b predicate. `refs` is {route_name: [values]}, exactly one list per DERIVED_ROUTE.

    Returns a verdict dict; never mutates its inputs. `fired` means "outside what the route allows",
    which is the question 1.5c's injection matrix asks.
    """
    names = sorted(refs)
    disagree, violations, within, exact = [], [], [], 0
    for i, fv in enumerate(file_vals):
        vals = {k: refs[k][i] for k in names}
        bits = {k: f32_bits(v) for k, v in vals.items()}
        if len(set(bits.values())) != 1:
            disagree.append({"index": i, "values": {k: repr(v) for k, v in vals.items()},
                             "bits": bits,
                             "ulp_between_references": ulp_dist(vals[names[0]], vals[names[-1]])})
            continue
        agreed = vals[names[0]]
        if fv == agreed:
            exact += 1
            continue
        # "Within one ulp" as a neighbour set: the file may be the agreed value, or the value one
        # nextafter hop either side of it. Nothing looser, and nothing that needs an epsilon.
        up = f32_nextafter(agreed, float("inf"))
        dn = f32_nextafter(agreed, float("-inf"))
        step = f32_ulp(agreed)
        d = abs(fv - agreed)
        rec = {"index": i, "file": repr(fv), "file_bits": f32_bits(fv),
               "agreed": repr(agreed), "agreed_bits": bits[names[0]],
               "reference_bits": dict(bits), "abs_error": d, "ulp_step_at_agreed": step,
               "neighbour_up": repr(up), "neighbour_up_bits": f32_bits(up),
               "neighbour_down": repr(dn), "neighbour_down_bits": f32_bits(dn),
               "steps_from_agreed": ulp_dist(fv, agreed)}
        (within if fv in (up, dn) else violations).append(rec)
    return {"elts": len(file_vals), "elements_bit_exact": exact,
            "elements_within_route": len(within), "elements_outside_route": len(violations),
            "reference_disagreements": len(disagree), "within_route_elements": within,
            "outside_route_elements": violations,
            "reference_disagreement_elements": disagree,
            "max_steps_from_agreed": max([w["steps_from_agreed"] for w in within] or [0]),
            "fired": bool(violations or disagree)}


def pack_k_scales(sc, mg):
    """Encoder route (quantize_row_q4_K_ref / _q5_K_ref): 8 6-bit scales + 8 6-bit mins in 12 B."""
    s = bytearray(12)
    for j in range(4):
        s[j] = sc[j] & 63
        s[j + 4] = mg[j] & 63
    for j in range(4, 8):
        s[j + 4] = (sc[j] & 0xF) | ((mg[j] & 0xF) << 4)
        s[j - 4] |= ((sc[j] >> 4) << 6)
        s[j] |= ((mg[j] >> 4) << 6)
    return s


def pack_q4_k(d, dmin, sc, mg, idx):
    out = bytearray()
    for i in range(len(idx)):
        b = bytearray(144)
        b[0:2] = h16(d)
        b[2:4] = h16(dmin)
        b[4:16] = pack_k_scales(sc[i], mg[i])
        L = idx[i]
        for j in range(0, QK_K, 64):
            q = 32 * (j // 64)
            for l in range(32):
                b[16 + q + l] = L[j + l] | (L[j + l + 32] << 4)
        out += b
    return bytes(out)


def expect_k(d, dmin, sc, mg, idx):
    """Oracle route: value = d*sc_j*q - dmin*m_j, sub-block j = element // 32."""
    out = []
    for i in range(len(idx)):
        blk = idx[i]
        for j in range(8):
            sd, md = d * sc[i][j], dmin * mg[i][j]
            out.extend([sd * q - md for q in blk[j * 32: j * 32 + 32]])
    return out


def pack_q5_k(d, dmin, sc, mg, idx):
    out = bytearray()
    for i in range(len(idx)):
        b = bytearray(176)
        b[0:2] = h16(d)
        b[2:4] = h16(dmin)
        b[4:16] = pack_k_scales(sc[i], mg[i])
        L = idx[i]
        qh = bytearray(32)
        ql = bytearray(128)
        m1, m2 = 1, 2
        for n in range(0, QK_K, 64):
            for j in range(32):
                l1 = L[n + j]
                if l1 > 15:
                    l1 -= 16
                    qh[j] |= m1
                l2 = L[n + j + 32]
                if l2 > 15:
                    l2 -= 16
                    qh[j] |= m2
                ql[(n // 2) + j] = l1 | (l2 << 4)
            m1 <<= 2
            m2 <<= 2
        b[16:48] = qh
        b[48:176] = ql
        out += b
    return bytes(out)


def pack_q6_k(d, sc, idx):
    out = bytearray()
    for i in range(len(idx)):
        b = bytearray(210)
        L = idx[i]
        ql = bytearray(128)
        qh = bytearray(64)
        for j in range(0, QK_K, 128):
            oq = 64 * (j // 128)
            oh = 32 * (j // 128)
            for l in range(32):
                ql[oq + l] = (L[j + l] & 0xF) | ((L[j + l + 64] & 0xF) << 4)
                ql[oq + l + 32] = (L[j + l + 32] & 0xF) | ((L[j + l + 96] & 0xF) << 4)
                qh[oh + l] = ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2)
                              | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6))
        b[0:128] = ql
        b[128:192] = qh
        for k in range(16):
            b[192 + k] = sc[i][k] & 0xFF
        b[208:210] = h16(d)
        out += b
    return bytes(out)


def expect_q6_k(d, sc, idx):
    out = []
    for i in range(len(idx)):
        blk = idx[i]
        for j in range(16):
            sd = d * sc[i][j]
            out.extend([sd * (q - 32) for q in blk[j * 16: j * 16 + 16]])
    return out


def pack_q8_0(ds, idx):
    out = bytearray()
    for i in range(len(idx)):
        out += h16(ds[i])
        out += bytes(x & 0xFF for x in idx[i])   # two.complement storage, read back as int8
    return bytes(out)


def expect_q8_0(ds, idx):
    out = []
    for i in range(len(idx)):
        out.extend([ds[i] * x for x in idx[i]])
    return out


def synth(tname, n_blocks, rng):
    """Build one synthetic block set -> (bytes, expected_dequant).

    The size guard is not decoration: this fixture returned ZERO elements for three of the five
    types on its first run (a block-list/flat-list mismatch), and the selftest read that as
    `mismatch 0` — `pitfalls/40-gates-and-admissions.md` §"A check that can produce no output
    cannot fail", witnessed rather than promised.
    """
    raw, exp = _synth_body(tname, n_blocks, rng)
    be, bb = gr.BLOCK[tname]
    n = n_blocks * be
    if len(exp) != n or len(raw) != n_blocks * bb:
        raise AssertionError("synthetic fixture for %s produced %d values / %d bytes, expected "
                             "%d / %d — a fixture that generates nothing cannot fail"
                             % (tname, len(exp), len(raw), n, n_blocks * bb))
    return raw, exp


def _synth_body(tname, n_blocks, rng):
    be, bb = gr.BLOCK[tname]
    n = n_blocks * be
    if tname == "F32":
        vals = [rng.f() * 3.0 for _ in range(n)]
        raw = struct.pack("<%df" % n, *vals)
        # the f32 stored IS the value; the oracle is what a widening of the stored bytes returns
        return raw, list(struct.unpack("<%df" % n, raw))
    if tname == "Q8_0":
        ds = [f16v(rng.f() * 0.1) for _ in range(n // 32)]
        idx = [[i8v(rng.u(0, 255)) for _ in range(32)] for _ in range(n // 32)]
        return pack_q8_0(ds, idx), expect_q8_0(ds, idx)
    if tname == "Q4_K":
        d, dmin = f16v(rng.f() * 0.02 + 0.001), f16v(rng.f() * 0.01 + 0.001)
        sc = [[rng.u(1, 63) for _ in range(8)] for _ in range(n // QK_K)]
        mg = [[rng.u(1, 63) for _ in range(8)] for _ in range(n // QK_K)]
        idx = [[rng.u(0, 15) for _ in range(QK_K)] for _ in range(n // QK_K)]
        return pack_q4_k(d, dmin, sc, mg, idx), expect_k(d, dmin, sc, mg, idx)
    if tname == "Q5_K":
        d, dmin = f16v(rng.f() * 0.02 + 0.001), f16v(rng.f() * 0.01 + 0.001)
        sc = [[rng.u(1, 63) for _ in range(8)] for _ in range(n // QK_K)]
        mg = [[rng.u(1, 63) for _ in range(8)] for _ in range(n // QK_K)]
        idx = [[rng.u(0, 31) for _ in range(QK_K)] for _ in range(n // QK_K)]
        return pack_q5_k(d, dmin, sc, mg, idx), expect_k(d, dmin, sc, mg, idx)
    if tname == "Q6_K":
        d = f16v(rng.f() * 0.02) or 0.0009765625
        sc = [[rng.u(-127, 127) for _ in range(16)] for _ in range(n // QK_K)]
        idx = [[rng.u(0, 63) for _ in range(QK_K)] for _ in range(n // QK_K)]
        return pack_q6_k(d, sc, idx), expect_q6_k(d, sc, idx)
    raise NotImplementedError(tname)


def selftest_shapes(geo):
    """Case list DERIVED from geometry (the shapes the audit will actually meet), not typed."""
    H, I, E, V = geo.H, geo.I, geo.E, geo.V
    return [
        ("routed gate/up", (H, I, E)),
        ("routed down", (I, H, E)),
        ("shared expert", (H, geo.Ise)),
        ("GDN in_proj_qkv", (H, geo.gdn_qkv_rows)),
        ("GDN out_proj", (geo.gdn_v_dim, H)),
        ("full attention", (H, geo.nq * geo.hd)),
        ("router", (H, E)),
        ("lm_head", (H, V)),
    ]


SELF_TYPES = ("F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K")
# Byte-level injections testable on synthetic blocks, and the types they must (and must not) move.
READER_INJS = {
    "q4k_as_q4nl": lambda tname: tname == "Q4_K",
    "q4k_drop_min": lambda tname: tname == "Q4_K",
    "block_shift_1": lambda tname: True,
}


def selftest(args):
    geo = Geo()
    shapes = selftest_shapes(geo)
    cases = [(fam, dims, tname) for fam, dims in shapes for tname in SELF_TYPES]
    n_blocks_used = 4
    results = []
    for fam, dims, tname in cases:
        rng = Lcg(hash((fam, tname)) & 0xFFFFFFFF)
        raw, exp = synth(tname, n_blocks_used, rng)
        got = gr.dequant(raw, tname, n_elts=len(exp))
        bad = sum(1 for a, b in zip(got, exp) if a != b)
        results.append({"family": fam, "shape": list(dims), "type": tname, "elts": len(exp),
                        "bytes": len(raw), "mismatch": bad,
                        "max_abs_dev": max((abs(a - b) for a, b in zip(got, exp)), default=0.0)})
    clean_fail = [r for r in results if r["mismatch"]]

    inj_rows = []
    for inj in sorted(READER_INJS):
        for tname in SELF_TYPES:
            if tname == "F32":
                continue
            rng = Lcg(0xC0FFEE)
            raw, exp = synth(tname, 2, rng)
            if inj == "block_shift_1":
                raw = b"\x00" + raw[:-1]
            got = gr.dequant(raw, tname, n_elts=len(exp), inj=inj)
            dev = max((abs(a - b) for a, b in zip(got, exp)), default=0.0)
            fired = dev > 0.0
            inj_rows.append({"injection": inj, "type": tname,
                             "expected": "FIRE" if READER_INJS[inj](tname) else "stay-clean",
                             "fired": fired, "max_abs_dev": dev})
    inj_bad = [r for r in inj_rows
               if (r["expected"] == "FIRE" and not r["fired"])
               or (r["expected"] == "stay-clean" and r["fired"])]

    gate_ok, gate_n = selftest_gate_up_split(geo)
    perm_ok, perm_n = selftest_v_perm(geo)
    tr_ok, tr_n = selftest_transpose_mapper(geo)
    floor_ok, floor_rows = selftest_floor_probe_honest(geo)
    auth_ok, auth_row = selftest_authority_widening()
    route_ok, route_rows = selftest_route_predicate()

    print("SELFTEST — reader decodes per the ggml-quants.c dequant loops; the packers encode per")
    print("the quantize_row_*_ref packing loops; the oracle uses the affine definition only.")
    print("cases: %d = %d shapes (derived from checkpoints/ geometry) x %d types %s"
          % (len(results), len(shapes), len(SELF_TYPES), list(SELF_TYPES)))
    table(["family", "shape", "type", "elts", "bytes", "mismatch", "max_abs_dev"],
          [(r["family"], r["shape"], r["type"], r["elts"], r["bytes"], r["mismatch"],
            "%.3e" % r["max_abs_dev"]) for r in results])
    print("\nmust-catch (FIRE) and must-not-catch (stay-clean), on synthetic bytes:")
    table(["injection", "type", "expected", "fired", "max_abs_dev"],
          [(r["injection"], r["type"], r["expected"], "YES" if r["fired"] else "no",
            "%.3e" % r["max_abs_dev"]) for r in inj_rows])
    print("\nclean decode: %d/%d cases bit-exact against the oracle (max dev %.3e)"
          % (len(results) - len(clean_fail), len(results),
             max(r["max_abs_dev"] for r in results)))
    print("injections that had to fire: %d, fired %d; wrongly fired outside their type: %d"
          % (sum(1 for r in inj_rows if r["expected"] == "FIRE"),
             sum(1 for r in inj_rows if r["expected"] == "FIRE" and r["fired"]),
             sum(1 for r in inj_rows if r["expected"] == "stay-clean" and r["fired"])))
    print("structural witnesses for the two INDEX injections:")
    print("  fused gate/up split + swapped halves: %s (%d positions)"
          % ("OK" if gate_ok else "BROKEN", gate_n))
    print("  GDN V tiled->grouped reorder:         %s (%d positions)"
          % ("OK" if perm_ok else "BROKEN", perm_n))
    print("  expert-stack transpose injection:     %s (%d positions, must differ from correct)"
          % ("OK" if tr_ok else "BROKEN", tr_n))
    print("floor-probe honesty (each width's own error inside its own step bound): %s"
          % ("OK" if floor_ok else "BROKEN"))
    for r in floor_rows:
        print("   %-6s bound=%.3e  own max err=%.3e  ratio=%.3f%s"
              % (r["type"], r["bound"], r["max_err"], r["ratio"],
                 "  " + r["note"] if "note" in r else ""))

    print("authority widening (bf16 -> f32, %d values, mismatches %d): %s"
          % (auth_row["n"], auth_row["bad"], "OK" if auth_ok else "BROKEN"))
    print("1.4b route predicate witnesses:")
    print("  nextafter/ulp against hand-computed neighbours (%d values): %s"
          % (len(route_rows["ulp_rows"]), "OK" if all(r["ok"] for r in route_rows["ulp_rows"])
             else "BROKEN"))
    print("  one step measured by spacing == one step measured in bits:              %s"
          % ("OK" if route_rows["steps_are_one_step"] else "BROKEN"))
    for r in route_rows["predicate_rows"]:
        print("  %-18s expected %-4s  predicate %-4s  (within-route %d, outside %d)  %s"
              % (r["case"], "FIRE" if r["must_fire"] else "clean",
                 "FIRE" if r["fired"] else "clean", r["within_route"], r["outside_route"],
                 "" if r["must_fire"] == r["fired"] else "MIS-SCHEDULED"))
    print("  two routes disagreeing fails the case even when the file is bit-exact:   %s"
          % ("OK" if route_rows["reference_disagreement_fires"] else "BROKEN"))
    ok = (not clean_fail and not inj_bad and gate_ok and perm_ok and tr_ok and floor_ok and auth_ok
          and route_ok
          and len(results) == len(shapes) * len(SELF_TYPES))
    print("\nSELFTEST %s" % ("PASS" if ok else "FAIL"))
    if ok:
        gr.grant_selftest_token(SELFTEST_TOKEN)
        first_pass = now()
        if os.path.exists(SELFTEST_MARK):
            # A later edit invalidates the marker and forces a re-run; the FIRST pass is the
            # ordering evidence and must survive it.
            first_pass = json.load(open(SELFTEST_MARK)).get("utc_first_selftest_pass", first_pass)
        write_json(SELFTEST_MARK, {
            "utc_first_selftest_pass": first_pass,
            "utc_latest_selftest_pass": now(),
            "gate_sha256": sha256_file(os.path.join(HERE, "audit_published_quant.py")),
            "reader_sha256": sha256_file(os.path.join(HERE, "gguf_reader.py")),
            "cases": len(results), "types": list(SELF_TYPES),
            "shapes_derived_from": "checkpoints/Qwen3.6-35B-A3B.config.json geometry",
            "must_catch_fired": sum(1 for r in inj_rows if r["expected"] == "FIRE" and r["fired"]),
            "must_catch_expected": sum(1 for r in inj_rows if r["expected"] == "FIRE"),
            "wrongly_fired": sum(1 for r in inj_rows if r["expected"] == "stay-clean"
                                 and r["fired"]),
            "max_abs_dev_clean": max(r["max_abs_dev"] for r in results),
            "index_injections_structural": {"gate_up_split": gate_ok, "v_reorder": perm_ok,
                                            "expert_transpose": tr_ok},
            "authority_widening": auth_row,
            "route_predicate": route_rows,
            "injections": sorted(list(READER_INJS) + ["gate_up_swap", "expert_stack_transpose"]),
            "python": sys.version.split()[0],
        })
        log_order("E1: SELFTEST PASS (first pass %s) — command: python3 "
                  "tools/audit_published_quant.py --selftest. No tensor-data byte had been read "
                  "before the first such line." % first_pass)
    else:
        if clean_fail:
            print("  failing clean cases: %s" % [(r["type"], r["family"], r["mismatch"])
                                                 for r in clean_fail])
        if inj_bad:
            print("  mis-scheduled injections: %s" % [(r["injection"], r["type"], r["expected"])
                                                      for r in inj_bad])
    return 0 if ok else 1


def selftest_route_predicate():
    """Witnesses for the 1.4b predicate, on values where the right answer is known independently.

    This exists because the first version of `f32_nextafter` inverted its own bit-mapping back and
    returned a float nowhere near the input, which made "within one step" mean "within 3.8" — a bound
    so loose it could not fail. A predicate built on a spacing function has to be tested against the
    spacing, not just used by it.
    """
    rows = []
    for x, expect_next, expect_prev in ((1.0, 1.0 + 2.0 ** -23, 1.0 - 2.0 ** -24),
                                       (0.5, 0.5 + 2.0 ** -24, 0.5 - 2.0 ** -25),
                                       (2.0, 2.0 + 2.0 ** -22, 2.0 - 2.0 ** -23),
                                       (-1.0, -(1.0 - 2.0 ** -24), -(1.0 + 2.0 ** -23)),
                                       (0.9096215963363647, 0.9096215963363647 + 2.0 ** -24,
                                        0.9096215963363647 - 2.0 ** -24)):
        # The expected neighbours are hand-worked from the binary representation — a float32 in [1,2)
        # is spaced 2**-23, in [0.5,1) it is 2**-24, in [2,4) it is 2**-22, and below a power of two the
        # step going up is half the step coming down — so they are an independent statement, not this
        # function echoed back. Writing them as powers of two instead of decimal literals is deliberate:
        # a decimal neighbour is only approximately the float you meant, and an approximate expectation
        # is how this witness first failed. (It caught a wrong spacing too: 2.0's next was written with
        # the [1,2) spacing.)
        # The expected neighbours are hand-worked from the binary representation (a float32 in
        # [1,2) is spaced 2**-23, in [0.5,1) it is 2**-24), so they are an independent statement and
        # not this function echoed back.
        _, e = math.frexp(abs(x))
        hand_ulp = math.ldexp(2.0 ** -23, e - 1)
        rows.append({"x": x,
                     "next": f32_nextafter(x, float("inf")), "expect_next": expect_next,
                     "prev": f32_nextafter(x, float("-inf")), "expect_prev": expect_prev,
                     "ulp": f32_ulp(x), "hand_ulp_from_frexp": hand_ulp,
                     "ok": (f32_nextafter(x, float("inf")) == expect_next
                            and f32_nextafter(x, float("-inf")) == expect_prev
                            and f32_ulp(x) == hand_ulp)})
    geom_ok = all(r["ok"] for r in rows)
    # A step must be exactly one step by the bit-distance measure too, on both sides, at both signs.
    step_ok = all(ulp_dist(f32_nextafter(x, float("inf")), x) == 1
                  and ulp_dist(f32_nextafter(x, float("-inf")), x) == 1
                  for x in (1.0, -1.0, 0.5, 2.0, 1e-38, 1e30, 0.9096215963363647))
    # Now the predicate itself, on a value the audited file really contains.
    a = 0.9096215963363647
    up1 = f32_nextafter(a, float("inf"))
    dn1 = f32_nextafter(a, float("-inf"))
    up2 = f32_nextafter(up1, float("inf"))
    cases = [("bit-exact", a, False), ("one step above", up1, False), ("one step below", dn1, False),
             ("two steps above", up2, True), ("two steps below", f32_nextafter(dn1, float("-inf")),
              True), ("sign flip", -a, True), ("permuted value", a * 1.5, True)]
    # The binade edge, where an upward-only "one ulp" would be twice as wide as one hop: a float32 just
    # below 1.0 is spaced 2**-24, one step above 1.0 is spaced 2**-23.
    dn_of_1 = f32_nextafter(1.0, float("-inf"))
    edge = [(1.0, dn_of_1, False),
            (1.0, f32_nextafter(dn_of_1, float("-inf")), True),
            (1.0, f32_nextafter(1.0, float("inf")), False),
            (1.0, f32_nextafter(f32_nextafter(1.0, float("inf")), float("inf")), True)]
    pred_rows = []
    for label, file_val, must_fire in cases:
        v = judge_derived_route([file_val], {"f32": [a], "crl": [a]})
        pred_rows.append({"case": label, "must_fire": must_fire, "fired": v["fired"],
                          "within_route": v["elements_within_route"],
                          "outside_route": v["elements_outside_route"]})
    for agreed, file_val, must_fire in edge:
        v = judge_derived_route([file_val], {"f32": [agreed], "crl": [agreed]})
        pred_rows.append({"case": "binade edge %r" % file_val, "must_fire": must_fire,
                          "fired": v["fired"], "within_route": v["elements_within_route"],
                          "outside_route": v["elements_outside_route"]})
    pred_ok = all(r["must_fire"] == r["fired"] for r in pred_rows)
    # The reference-agreement rule: if the two routes disagree, the case FAILS whatever the file says,
    # including when the file is bit-exact against one of them. Same file, two worlds.
    dis = judge_derived_route([a], {"f32": [a], "crl": [up1]})
    agree = judge_derived_route([a], {"f32": [a], "crl": [a]})
    disagree_ok = (dis["fired"] and dis["reference_disagreements"] == 1
                   and not agree["fired"] and agree["elements_bit_exact"] == 1)
    return (geom_ok and step_ok and pred_ok and disagree_ok), {
        "ulp_rows": rows, "steps_are_one_step": step_ok, "predicate_rows": pred_rows,
        "reference_disagreement_fires": disagree_ok,
        "reference_disagreement_note": "file bit-exact against route f32 while crl disagrees: still "
                                       "FAILS, because the gate may not absorb its own references"}


def selftest_gate_up_split(geo):
    """The fused pair is [E,2I,H] split on dim=-2 (conversion/qwen.py). Check the audit's inverse."""
    class T:
        dims = (geo.H, geo.I, geo.E)
        n_elts = geo.H * geo.I * geo.E
    t = T()
    IH = geo.I * geo.H
    bad = checked = 0
    for half in (0, 1):
        m = make_mapper(geo, t, "gate_up_split", half)
        ms = make_mapper(geo, t, "gate_up_split", half, inj="gate_up_swap")
        for g in (0, 1, IH - 1, IH, IH + 5, 2 * IH - 1, 3 * IH + 7, 6 * IH + 13, t.n_elts - 1):
            if g >= t.n_elts:
                continue
            exp_e, inner = g // IH, g % IH
            want = exp_e * 2 * IH + half * IH + inner
            swap = exp_e * 2 * IH + (1 - half) * IH + inner
            if m(g) != want or ms(g) != swap or swap == want:
                bad += 1
            checked += 1
    return bad == 0, checked


def selftest_v_perm(geo):
    """Check the audit's tiled->grouped map against the converter's own nested-loop formulation."""
    nk, nvk = geo.nk, geo.nv_per_k
    bad = checked = 0
    for hd in (geo.dv, 1):
        ref = v_grouped_order(nk, nvk, hd)
        seen = set()
        for i in sorted(ref):
            j = v_tiled_to_grouped(i, nk, nvk, hd)
            if j != ref[i]:
                bad += 1
            seen.add(j)
            checked += 1
        if len(seen) != len(ref):                 # a permutation must be a bijection
            bad += 1
        # and it must actually move something, or the witness proves nothing
        if all(v_tiled_to_grouped(i, nk, nvk, hd) == i for i in ref):
            bad += 1
    return bad == 0, checked


def selftest_transpose_mapper(geo):
    """The correct mapper is the identity; the transpose injection must be a different bijection."""
    class T:
        dims = (geo.H, geo.I, geo.E)
        n_elts = geo.H * geo.I * geo.E
    t = T()
    inj = make_mapper(geo, t, None, inj="expert_stack_transpose")
    bad = checked = 0
    seen = set()
    for g in (0, 1, geo.H - 1, geo.H, geo.H + 1, t.n_elts // 3, t.n_elts - 1):
        if inj(g) == g:
            pass                                  # a fixed point is allowed, not required
        if inj(g) >= t.n_elts:
            bad += 1
        seen.add(inj(g))
        checked += 1
    if len(seen) != checked:
        bad += 1
    if inj(geo.H) == geo.H:                       # the first stride step must move
        bad += 1
    return bad == 0, checked


def selftest_authority_widening():
    """The bf16 widening on the AUTHORITY side is load-bearing for 1.2/1.4/1.5 alike, so it is
    checked against an independent bit route (mask the f32 bits per element), not against itself.
    A wrong shift here would make every exactness row fail loudly, but a wrong shift that happened
    to agree with a rounding rule would not, and the authority is the thing being trusted."""
    rng = Lcg(0xA1)
    vals = [rng.f() * s for s in (1e-4, 1.0, 4.0e3, 3.0e-40) for _ in range(64)]
    vals += [0.0, -0.0]
    raw = b"".join(struct.pack("<f", v)[2:4] for v in vals)
    got = gr.bf16_to_floats(raw)
    want_bits = [struct.unpack("<I", struct.pack("<f", v))[0] & 0xFFFF0000 for v in vals]
    bad = sum(1 for g, w in zip(got, want_bits)
              if struct.unpack("<I", struct.pack("<f", g))[0] != w)
    ok = bad == 0 and len(got) == len(vals) and len(vals) > 0
    return ok, {"n": len(vals), "bad": bad, "route": "gr.bf16_to_floats vs per-element bit mask"}


def selftest_floor_probe_honest(geo):
    """A floor probe that lies is worse than none.

    Each width's own error, on data that is exactly representable in the authority width (bf16),
    must sit inside the grid that width actually produced. Bound = (realized step) x asym, and
    asym is a fact about the format, not a tolerance: Q8_0/Q4_K/Q5_K grids span the block range
    min..max, so half a step is the worst error; Q6_K's grid is zero-based with 64 levels
    [-32, 31], so the positive end of a symmetric block sits a whole step above its last level.
    The *realized* step is used because ggml's shared 6-bit scale grid can only approximate a
    sub-block's own min/max fit — that coarsening is part of what the floor honestly measures.
    """
    rng = Lcg(0xB0)
    rows = []
    ok = True
    for tname in ("Q8_0", "Q4_K", "Q5_K", "Q6_K"):
        be, _ = gr.BLOCK[tname]
        xs = bf16_stream(rng, 8 * be)
        steps = []
        dq = gr.quant_ref_deq(tname, xs, steps=steps)
        if len(dq) != len(xs) or len(steps) != len(xs):
            return False, [{"type": tname, "bound": 0.0, "max_err": 0.0, "ratio": float("inf"),
                            "note": "quantizer returned %d values / %d steps for %d"
                                    % (len(dq), len(steps), len(xs))}]
        err = max(abs(a - b) for a, b in zip(dq, xs))
        asym = 1.0 if tname == "Q6_K" else 0.5
        bound = asym * max(steps)
        rows.append({"type": tname, "bound": bound, "max_err": err,
                     "ratio": err / bound if bound else float("inf")})
        if not (0.0 < err <= bound * 1.02 + 1e-12):
            ok = False
    return ok, rows


# --------------------------------------------------------------------------------------------
# the selftest gate on every data-reading section
# --------------------------------------------------------------------------------------------
def require_selftest(args):
    if not os.path.exists(SELFTEST_MARK):
        print("REFUSING to read a single tensor-data byte: no selftest marker at %s.\n"
              "Run: python3 tools/audit_published_quant.py --selftest" % (SELFTEST_MARK,))
        return None
    mark = json.load(open(SELFTEST_MARK))
    cur_gate = sha256_file(os.path.join(HERE, "audit_published_quant.py"))
    cur_reader = sha256_file(os.path.join(HERE, "gguf_reader.py"))
    if mark.get("gate_sha256") != cur_gate or mark.get("reader_sha256") != cur_reader:
        print("REFUSING: the selftest marker is STALE — the gate or the reader changed since it "
              "passed (recorded %s / %s, current %s / %s). Re-run --selftest."
              % (mark.get("gate_sha256"), mark.get("reader_sha256"), cur_gate, cur_reader))
        return None
    gr.grant_selftest_token(SELFTEST_TOKEN)
    log_order("E2: selftest marker accepted (%s); tensor-data reads allowed from here. section=%s"
              % (mark["utc_first_selftest_pass"], args.section))
    return mark


# --------------------------------------------------------------------------------------------
# 1.1 — hash, revision, header identity
# --------------------------------------------------------------------------------------------
def sec_hash(args):
    out = {"utc": now(), "artifact": GGUF_PATH, "bytes_on_disk": os.path.getsize(GGUF_PATH)}
    print("1.1 — sha256, revision, file_type, quantization_version, tensor record count")
    if not os.path.exists(GGUF_PATH):
        print("MISSING artifact: %s" % GGUF_PATH)
        return 1
    s1 = sha256_file(GGUF_PATH)
    s2 = sha256_file(GGUF_PATH)
    meta = hf_download_meta(GGUF_HF_DIR, os.path.basename(GGUF_PATH))
    gg = gr.Gguf(GGUF_PATH)
    n_rec = len(gg.tensors)
    declared = gg.n_tensors
    rows = [("sha256 pass 1", s1), ("sha256 pass 2 (re-read)", s2),
            ("re-read identical", "YES" if s1 == s2 else "NO"),
            ("HF revision (download metadata)", meta and meta["revision"]),
            ("HF revision expected", GGUF_REVISION),
            ("HF etag (= LFS sha256)", meta and meta["etag"]),
            ("etag == computed sha256", "YES" if meta and meta["etag"] == s1 else "NO"),
            ("gguf version", gg.version),
            ("general.architecture", gg.meta.get("general.architecture")),
            ("general.file_type", gg.meta.get("general.file_type")),
            ("general.quantization_version", gg.meta.get("general.quantization_version")),
            ("n_tensors declared", declared),
            ("n_tensors parsed", n_rec),
            ("n_tensors expected", EXPECTED_TENSOR_RECORDS),
            ("n_kv metadata pairs", gg.n_kv),
            ("header_end / data_start / alignment",
             "%d / %d / %d" % (gg.header_end, gg.data_start, gg.alignment)),
            ("bytes pulled past tensor-info and never interpreted",
             gg.bytes_read_of_data_region_uninterpreted),
            ]
    weight_bytes = sum(t.row for t in gg.tensors)
    last_end = gg.data_start + max(t.offset + t.row for t in gg.tensors)
    rows.append(("sum(row_bytes) of all tensors", weight_bytes))
    rows.append(("data_start + last tensor end", last_end))
    rows.append(("file size", out["bytes_on_disk"]))
    rows.append(("size reconciles exactly", "YES" if last_end == out["bytes_on_disk"] else "NO"))
    table(["field", "value"], rows)
    verdict = (s1 == s2 and n_rec == declared == EXPECTED_TENSOR_RECORDS
               and last_end == out["bytes_on_disk"]
               and (meta or {}).get("revision") == GGUF_REVISION
               and (meta or {}).get("etag") == s1)
    print("\n1.1 %s" % ("PASS" if verdict else "FAIL"))
    out.update({"sha256_pass1": s1, "sha256_pass2": s2, "revision": (meta or {}).get("revision"),
                "etag": (meta or {}).get("etag"), "file_type": gg.meta.get("general.file_type"),
                "quantization_version": gg.meta.get("general.quantization_version"),
                "architecture": gg.meta.get("general.architecture"),
                "n_tensors": n_rec, "header_end": gg.header_end, "data_start": gg.data_start,
                "alignment": gg.alignment, "weight_bytes": weight_bytes,
                "size_reconciles": last_end == out["bytes_on_disk"],
                "verdict": "PASS" if verdict else "FAIL"})
    write_json(HASH_JSON, out)
    gg.close()
    return 0 if verdict else 1


# --------------------------------------------------------------------------------------------
# 1.2 — shards match the committed index; extract and witness mtp.*
# --------------------------------------------------------------------------------------------
def safetensors_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def sec_shards(args):
    print("1.2 — every fetched shard's declared tensor set vs the committed weight_map; "
          "extract mtp.* and witness its element count")
    idx = json.load(open(IDX))
    wm = idx["weight_map"]
    out = {"utc": now(), "shards": {}, "index_total_size": idx["metadata"]["total_size"],
           "index_tensor_count": len(wm)}
    bad = 0
    rows = []
    for sh in SHARDS:
        p = os.path.join(SHARD_DIR, sh)
        hdr, hlen = safetensors_header(p)
        declared = {k for k in hdr if k != "__metadata__"}
        want = {k for k, v in wm.items() if v == sh}
        meta = hf_download_meta(SHARD_DIR, sh)
        digest = None if args.no_shard_sha else sha256_file(p)
        ok_set = declared == want
        ok_sha = bool(digest) and (meta or {}).get("etag") == digest
        if not ok_set or (digest and not ok_sha):
            bad += 1
        dtypes = {v["dtype"] for k, v in hdr.items() if k != "__metadata__"}
        rows.append((sh, len(declared), len(want), "OK" if ok_set else "MISMATCH",
                     len(digest or "") and ("OK" if ok_sha else "MISMATCH") or "not-hashed",
                     ",".join(sorted(dtypes)), os.path.getsize(p), hlen))
        out["shards"][sh] = {"n_declared": len(declared), "n_index": len(want),
                             "set_matches": ok_set, "sha256": digest,
                             "etag_matches": (None if digest is None else ok_sha),
                             "etag": (meta or {}).get("etag"), "revision": (meta or {}).get("revision"),
                             "bytes": os.path.getsize(p), "header_len": hlen,
                             "dtypes": sorted(dtypes)}
    table(["shard", "n_declared", "n_index", "set==weight_map", "sha==HF etag", "dtypes",
           "bytes", "header_len"], rows)

    # --- mtp extraction and the element witness ---
    mtp_names = sorted(k for k in wm if k.startswith("mtp."))
    auth = Authority(SELFTEST_TOKEN)
    got = [n for n in mtp_names if auth.has(n)]
    elts = sum(auth.info(n)["n_elts"] for n in got)
    nbytes = sum(auth.info(n)["bytes"] for n in got)
    mtp_rows = []
    for n in mtp_names:
        i = auth.info(n) if auth.has(n) else None
        mtp_rows.append((n, i and i["shard"], i and i["dtype"], str(i and list(i["shape"])),
                         nines(i["n_elts"]) if i else "ABSENT", i and i["bytes"]))
    table(["mtp tensor", "shard", "dtype", "shape", "elements", "bytes"], mtp_rows)
    print("\nmtp tensors in the committed index: %d, all present in the fetched shards: %s"
          % (len(mtp_names), "YES" if len(got) == len(mtp_names) else "NO"))
    print("derived mtp element total = %s elts = %d B = %.4f MiB bf16"
          % (nines(elts), nbytes, nbytes / 2.0 ** 20))
    print("PLAN.md step 1 fetch list claims %s elts = 79.9 MiB bf16 -> %s"
          % (nines(MTP_CLAIM_ELTS),
             "CONFIRMED" if elts == MTP_CLAIM_ELTS else
             "REFUTED (actual is %.2fx the claim)" % (elts / float(MTP_CLAIM_ELTS))))
    claim_ok = elts == MTP_CLAIM_ELTS
    # write the extraction: a valid safetensors file the engine can load without the shards
    os.makedirs(os.path.dirname(MTP_OUT), exist_ok=True)
    hdr = {}
    blob = bytearray()
    off = 0
    for n in mtp_names:
        i = auth.info(n)
        raw = auth.raw_bytes(n, 0, i["n_elts"])
        hdr[n] = {"dtype": i["dtype"], "shape": list(i["shape"]),
                  "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(hj)) % 8
    hj = hj + b" " * pad
    with open(MTP_OUT, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj + bytes(blob))
    n2, _ = safetensors_header(MTP_OUT)
    wrote_ok = ({k for k in n2 if k != "__metadata__"} == set(mtp_names)
                and sum(v["data_offsets"][1] - v["data_offsets"][0]
                        for k, v in n2.items() if k != "__metadata__") == nbytes)
    print("extracted -> %s (%d tensors, %d B tensor data, re-parsed OK: %s)"
          % (os.path.relpath(MTP_OUT, REPO), len(mtp_names), off, "YES" if wrote_ok else "NO"))
    auth.close()
    out["mtp"] = {"names": len(mtp_names), "present": len(got), "elements": elts,
                  "bytes": nbytes, "plan_claim_elements": MTP_CLAIM_ELTS,
                  "plan_claim_verdict": "CONFIRMED" if claim_ok else "REFUTED",
                  "extracted_to": os.path.relpath(MTP_OUT, REPO), "extract_reparse_ok": wrote_ok}
    ok = bad == 0 and len(got) == len(mtp_names) and wrote_ok
    print("\n1.2 %s  (shard<->index reconciliation %s; mtp extraction %s; PLAN mtp element claim %s)"
          % ("PASS" if ok else "FAIL", "PASS" if bad == 0 else "FAIL",
             "PASS" if wrote_ok else "FAIL",
             "CONFIRMED" if claim_ok else "REFUTED — see the finding above"))
    write_json(SHARD_JSON, out)
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------
# 1.3 — structure pass over every tensor record (no shard bytes needed except the 4 we fetched)
# --------------------------------------------------------------------------------------------
def sec_structure(args):
    print("1.3 — structure pass: names, rank, dims (GGUF reversed-dim convention), dtype, "
          "per-family counts, tensor-for-tensor against the committed census")
    idx = json.load(open(IDX))
    wm = idx["weight_map"]
    geo = Geo()
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)   # headers only in this section; the token gates data reads
    if len(gg.tensors) != gg.n_tensors:
        print("PARSE SHORT: parsed %d of %d declared records" % (len(gg.tensors), gg.n_tensors))
        return 1
    # expected GGUF records, derived from the census
    expect = {}
    absent = {}
    for hf in wm:
        maps = hf_to_gguf_names(hf)
        if not maps:
            absent[hf] = "no mapping rule"
            continue
        for gname, tr, extra in maps:
            if gname is None:
                absent[hf] = tr
                continue
            expect[gname] = (hf, tr, extra)
    unreconciled_gguf = [t.name for t in gg.tensors if t.name not in expect]
    missing_gguf = [n for n in expect if n not in gg.by_name]
    dim_rows, dim_bad = [], []
    ranknorm = []
    fam_counts = {}
    dtype_by_family = {}
    for t in gg.tensors:
        hf_name, tr, extra = expect[t.name]
        if auth.has(hf_name):
            shape = list(auth.info(hf_name)["shape"])
            src = "safetensors header"
        else:
            shape = geo.hf_shape_of(hf_name)
            src = "config geometry"
        if shape is None:
            dim_bad.append((t.name, "no derived HF shape"))
            continue
        if tr == "gate_up_split":
            shape[1] //= 2
        if tr == "v_conv":
            # conversion/qwen.py §Qwen3NextModel.modify_tensors squeezes the conv kernel:
            # [8192, 1, 4] -> [8192, 4]. Modelled, not accommodated: the element count still has
            # to come out identical.
            shape = [d for d in shape if d != 1]
        want = list(reversed(shape))
        have = list(t.dims)
        n_want, n_have = _prod(want), _prod(have)
        fam = gguf_family(t.name)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        dtype_by_family.setdefault(fam, set()).add(t.type_name)
        if want == have:
            match = "OK"
        elif n_want == n_have and strip_trailing_ones(want) == strip_trailing_ones(have):
            # ggml_n_dims() (ggml/src/ggml.c) reports a tensor's rank with trailing singleton dims
            # dropped, so an HF [1, 2048] leaves the C++ writer as rank-1 [2048]. Same bytes, same
            # element count: a rank convention, recorded as its own class instead of being folded
            # into "OK" — a real dim error must not be able to hide in this bucket.
            match = "OK-ranknorm(trailing 1 dropped by ggml_n_dims)"
            ranknorm.append((t.name, want, have))
        else:
            match = "MISMATCH"
        dim_rows.append((t.name, fam, t.type_name, len(t.dims), str(have), str(want), match, src))
        if match == "MISMATCH":
            dim_bad.append((t.name, "dims %s != expected %s (HF %s %s, %d vs %d elements)"
                            % (have, want, hf_name, src, n_have, n_want)))
    n_hf_text = len([k for k in wm if not k.startswith("model.visual")
                     and not k.startswith("mtp.")])
    print("census: %d HF tensor names; %d not vision/mtp; mapped to %d expected GGUF names; "
          "file declares %d" % (len(wm), n_hf_text, len(expect), len(gg.tensors)))
    print("families (count, dtype(s) actually stored):")
    frows = [(f, fam_counts[f], ",".join(sorted(dtype_by_family[f]))) for f in sorted(fam_counts)]
    table(["family", "count", "types stored"], frows)
    n_fam_total = sum(fam_counts.values())
    print("family rows total %d, must equal %d records" % (n_fam_total, len(gg.tensors)))
    shown = dim_rows if args.all_rows else dim_rows[:12]
    print("\nfirst %d of %d per-tensor dim rows (full list is in the JSON):"
          % (len(shown), len(dim_rows)))
    table(["gguf tensor", "family", "type", "rank", "gguf dims", "expected dims", "match",
           "shape source"], shown)
    weight_bytes = sum(t.row for t in gg.tensors)
    size_ok = gg.data_start + max(t.offset + t.row for t in gg.tensors) == os.path.getsize(GGUF_PATH)
    print("\nblock-size witness: sum(row_bytes)=%d, data_start=%d, file=%d -> %s"
          % (weight_bytes, gg.data_start, os.path.getsize(GGUF_PATH),
             "exact" if size_ok else "NOT EXACT (the block table is wrong or a tensor is truncated)"))
    print("unreconciled GGUF names: %d; expected-but-absent: %d; dim mismatches: %d; "
          "rank-convention rows: %d%s"
          % (len(unreconciled_gguf), len(missing_gguf), len(dim_bad), len(ranknorm),
             "" if not ranknorm else " (all %s)" % sorted({r[0].split(".")[0] + "." + r[0].split(".")[-2] for r in ranknorm})))
    for r in (unreconciled_gguf + missing_gguf)[:20]:
        print("   UNRECONCILED: %s" % r)
    for r in dim_bad[:20]:
        print("   DIM: %s — %s" % r)
    print("HF names with no GGUF counterpart, by reason: %s"
          % json.dumps(_count_absent(absent), sort_keys=True))
    ok = (not unreconciled_gguf and not missing_gguf and not dim_bad and size_ok
          and n_fam_total == len(gg.tensors) == EXPECTED_TENSOR_RECORDS)
    print("\n1.3 %s%s" % ("PASS" if ok else "FAIL",
                          "" if ok else " — values are NOT judged until this reconciles"))
    out = {"utc": now(), "n_tensors": len(gg.tensors), "expected_from_census": len(expect),
           "families": {f: fam_counts[f] for f in sorted(fam_counts)},
           "types_by_family": {f: sorted(v) for f, v in sorted(dtype_by_family.items())},
           "hf_absent_reasons": _count_absent(absent),
           "unreconciled_gguf_names": unreconciled_gguf,
           "rank_convention_rows": ["%s: expected %s, stored %s" % r for r in ranknorm],
           "expected_but_absent": missing_gguf,
           "dim_mismatches": ["%s: %s" % r for r in dim_bad],
           "per_tensor": [{"name": r[0], "family": r[1], "type": r[2], "rank": r[3],
                           "dims": r[4], "expected": r[5], "match": r[6], "src": r[7]}
                          for r in dim_rows],
           "size_reconciles": size_ok, "verdict": "PASS" if ok else "FAIL"}
    write_json(STRUCT_JSON, out)
    auth.close()
    gg.close()
    return 0 if ok else 1


def _prod(dims):
    n = 1
    for d in dims:
        n *= int(d)
    return n


def strip_trailing_ones(dims):
    d = list(dims)
    while len(d) > 1 and d[-1] == 1:
        d.pop()
    return d


def _count_absent(absent):
    c = {}
    for k, v in absent.items():
        c[v] = c.get(v, 0) + 1
    return c


# --------------------------------------------------------------------------------------------
# 1.4 — exactness pass: the families stored F32 must be bit-exact against bf16 widened to F32
# --------------------------------------------------------------------------------------------
def sec_exact(args):
    """1.4 in two passes, because the two halves ask different questions about different things.

    1.4a COPIED — routers, every norm, dt_bias / alpha / beta, conv1d, and the two ordering questions
         (GDN V reorder, fused gate/up split). ZERO tolerance, byte-for-byte as before: each of these
         is a permutation or an exact arithmetic transform of bytes HF really stores, so bit-for-bit
         is the correct predicate and any deviation is a layout, dtype or ordering defect.

    1.4b DERIVED-TRANSCENDENTAL — only the tensors whose documented route is -exp(A_log). HF never
         stores ssm_a (it computes A = -exp(A_log.float()) at module init), so there is no stored
         authority for these bytes to be bit-exact against; the predicate is agreement between two
         independent correctly-rounded routes, plus a one-representable-step bound (DERIVED_ROUTE).
         Authorised as a route-derived predicate rather than a tolerance by the supervisor on
         2026-10-26, surfaced to the user for ratification. The zero-tolerance answer is still
         computed, still printed, and still stored beside the new verdict.
    """
    print("1.4 — exactness pass, split:")
    print("      1.4a COPIED families (routers, norms, dt_bias, alpha, beta, conv1d, order/reorder)")
    print("           — ZERO tolerance, bit-for-bit against the bf16 authority widened to f32;")
    print("      1.4b DERIVED-TRANSCENDENTAL (ssm_a = -exp(A_log)) — route predicate: two")
    print("           independent correctly-rounded routes must agree with each other, and the file")
    print("           may sit at most one f32 step from the value they agree on.")
    geo = Geo()
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)
    targets = build_targets(gg, geo, auth, kinds=("exact",))
    f32_targets = [tg for tg in targets if tg["t"].type_name == "F32"]
    derived_targets = [tg for tg in f32_targets if value_transform(tg["t"].name) in DERIVED_ROUTE]
    derived_ids = {id(tg) for tg in derived_targets}
    copied_targets = [tg for tg in f32_targets if id(tg) not in derived_ids]
    rows, fails, permuted_notes, near_misses = [], [], [], []
    zero_tol_fails = []
    routes_seen, v_order = {}, {}
    n_router = 0
    for tg in copied_targets:
        t, hf_name, tr, extra = tg["t"], tg["hf"], tg["tr"], tg["extra"]
        n = t.n_elts
        gbytes = gg.read(SELFTEST_TOKEN, gg.data_start + t.offset, t.row,
                         why="exactness pass on %s" % t.name)
        first_data_log(gg, "first tensor-data read (exactness pass): %s" % t.name)
        vtr = value_transform(t.name)
        # The candidate routes, all of them read off the pinned converter and none of them fitted:
        #   order      — HF order as stored, or the documented GDN V reorder;
        #   arithmetic — no value transform, or the transform done in f32, or done in bf16.
        # Reporting WHICH route matched is the point: "it matches" would hide whether the file
        # carries the grouped or the tiled V order, and whether the +1 on norms was done wide.
        routes = []
        for olabel, otr in ([("as stored", None)] if tr is None
                            else [("HF order", None), ("V-reorder", tr)]):
            m = (make_mapper(geo, t, otr, extra) or (lambda g: g)) if otr else (lambda g: g)
            vals0 = auth.elements(hf_name, [m(g) for g in range(n)])
            if vtr is None:
                routes.append((olabel, "no value transform", vals0))
            else:
                for arith in ARITH_ROUTES[vtr]:
                    routes.append((olabel, "%s(%s)" % (VALUE_TR_LABEL[vtr], arith),
                                   apply_value_transform(vtr, vals0, arith)))
        matched = None
        best = None
        gvals = struct.unpack("<%df" % n, gbytes)
        for olabel, vlabel, vals in routes:
            off = [i for i in range(n) if gvals[i] != vals[i]]
            ulp = max((ulp_dist(gvals[i], vals[i]) for i in off), default=0)
            if best is None or len(off) < best[2] or (len(off) == best[2] and ulp < best[3]):
                best = (olabel, vlabel, len(off), ulp, off[:4], n)
            if not off:
                matched = (olabel, vlabel)
                break
        which = "%s + %s" % matched if matched else "NEITHER"
        verdict = matched is not None
        if verdict:
            routes_seen[which] = routes_seen.get(which, 0) + 1
            if tr is not None:
                v_order[t.name] = matched[0]
            if matched not in (("as stored", "no value transform"),
                               ("HF order", "no value transform")):
                permuted_notes.append((t.name, which))
        else:
            fails.append(t.name)
            zero_tol_fails.append(t.name)
            near_misses.append({"name": t.name, "hf": hf_name, "pass": "1.4a", "elts": n,
                                "best_route": "%s + %s" % (best[0], best[1]),
                                "elements_off": best[2], "max_ulp": best[3],
                                "first_off_indices": best[4],
                                "routes_tried": ["%s+%s" % r[:2] for r in routes]})
        if "ffn_gate_inp" in t.name and "shexp" not in t.name:
            n_router += 1
        rows.append({"name": t.name, "family": gguf_family(t.name), "hf": hf_name, "elts": n,
                     "mapping": (tr or "-") + ("+" + vtr if vtr else ""), "matched_under": which,
                     "verdict": "PASS" if verdict else "FAIL", "pass": "1.4a"})
        print("  %-40s %-24s %9d elts  %-42s %s" % (t.name, gguf_family(t.name), n, which,
                                                    "PASS" if verdict else "FAIL"))
    # ---- 1.4b: the derived-transcendental route predicate ---------------------------------------
    derived_cases = []
    for tg in derived_targets:
        t, hf_name, tr, extra = tg["t"], tg["hf"], tg["tr"], tg["extra"]
        vtr = value_transform(t.name)
        n = t.n_elts
        orders = [("as stored", False)] if tr is None else [("HF order", False),
                                                            ("V-reorder", True)]
        trials, ref_disagree_total = [], 0
        for olabel, documented in orders:
            gvals, refs, n = derived_route_inputs(gg, geo, auth, tg, documented_order=documented,
                                                  why="derived-route pass on %s" % t.name)
            v = judge_derived_route(gvals, refs)
            v["order"] = olabel
            v["route_pairs"] = ["%s = %s" % (r, DERIVED_ROUTE_LABEL[r]) for r in sorted(refs)]
            ref_disagree_total += v["reference_disagreements"]
            v["elements"] = [{"index": i, "file_bits": f32_bits(gvals[i]),
                              "route_bits": {r: f32_bits(refs[r][i]) for r in refs},
                              "class": ("bit-exact" if gvals[i] == refs[sorted(refs)[0]][i]
                                        else "not bit-exact; see the within/outside lists")}
                             for i in range(n)]
            trials.append(v)
        # A reference disagreement anywhere, under any documented order, kills the case outright: the
        # gate must never absorb a disagreement between its own two references as "close enough".
        clean = [v for v in trials if not v["fired"]] if ref_disagree_total == 0 else []
        best = min(trials, key=lambda v: (v["reference_disagreements"], v["elements_outside_route"],
                                          v["elements_within_route"]))
        case = clean[0] if clean else best
        ok_case = bool(clean)
        finding = case["elements_within_route"] > 0
        verdict = ("PASS-with-finding" if ok_case and finding else "PASS" if ok_case else "FAIL")
        if finding:
            zero_tol_fails.append(t.name)        # what the old predicate said, kept on the record
            near_misses.append({"name": t.name, "hf": hf_name, "pass": "1.4b", "elts": n,
                                "best_route": "%s + %s" % (case["order"], VALUE_TR_LABEL[vtr]),
                                "elements_off": case["elements_within_route"],
                                "max_ulp": case["max_steps_from_agreed"],
                                "first_off_indices": [w["index"] for w in
                                                      case["within_route_elements"]][:4],
                                "routes_tried": case["route_pairs"]})
        if tr is not None:
            v_order[t.name] = case["order"]
        routes_seen["%s + %s (within-route)" % (case["order"], vtr)] = \
            routes_seen.get("%s + %s (within-route)" % (case["order"], vtr), 0) + 1
        derived_cases.append({"name": t.name, "hf": hf_name, "route": vtr, "order": case["order"],
                              "verdict": verdict,
                              "routes": case["route_pairs"],
                              "elts": case["elts"],
                              "elements_bit_exact": case["elements_bit_exact"],
                              "elements_within_route": case["elements_within_route"],
                              "elements_outside_route": case["elements_outside_route"],
                              "reference_disagreements": case["reference_disagreements"],
                              "max_steps_from_agreed": case["max_steps_from_agreed"],
                              "within_route_elements": case["within_route_elements"],
                              "outside_route_elements": case["outside_route_elements"],
                              "reference_disagreement_elements": case["reference_disagreement_elements"],
                              "every_element": case["elements"]})
        rows.append({"name": t.name, "family": gguf_family(t.name), "hf": hf_name, "elts": n,
                     "mapping": "-+" + vtr,
                     "matched_under": "%s + %s, %d within-route" % (case["order"], vtr,
                                                                    case["elements_within_route"]),
                     "verdict": verdict, "pass": "1.4b"})
        print("  %-40s %-24s %9d elts  order=%-10s %s" % (t.name, gguf_family(t.name), n,
                                                          case["order"], verdict))
        print("      %d elements judged by the route predicate: %d bit-exact, %d within-route (these "
              "are NOT bit-exact), %d outside, %d reference disagreements"
              % (case["elts"], case["elements_bit_exact"], case["elements_within_route"],
                 case["elements_outside_route"], case["reference_disagreements"]))
        for w in case["within_route_elements"]:
            print("      index %-4d file %s [%s] | route f32 %s [%s] | route crl %s [%s] | "
                  "%d step(s) away, one step here is %s"
                  % (w["index"], w["file"], w["file_bits"], w["agreed"],
                     w["reference_bits"]["f32"], w["agreed"], w["reference_bits"]["crl"],
                     w["steps_from_agreed"], w["ulp_step_at_agreed"]))
        for w in case["outside_route_elements"]:
            print("      OUTSIDE ROUTE index %d: file %s [%s] vs agreed %s [%s] — %d steps away, one "
                  "step is %s" % (w["index"], w["file"], w["file_bits"], w["agreed"],
                                  w["agreed_bits"], w["steps_from_agreed"], w["ulp_step_at_agreed"]))
        for w in case["reference_disagreement_elements"]:
            print("      REFERENCE DISAGREEMENT index %d: %s — the two correctly-rounded routes "
                  "disagree (%d ulp), so nothing here can be judged and the case FAILS"
                  % (w["index"], json.dumps(w["bits"]), w["ulp_between_references"]))
    table(["pass", "gguf tensor", "family", "hf authority", "elts", "mapping applied",
           "matched under", "verdict"],
          [(r["pass"], r["name"], r["family"], r["hf"], r["elts"], r["mapping"],
            r["matched_under"], r["verdict"]) for r in rows])
    n_within = sum(c["elements_within_route"] for c in derived_cases)
    n_bitexact_cases = sum(1 for r in rows if r["verdict"] in ("PASS", "PASS-with-finding")
                           and r["pass"] == "1.4a") + \
        sum(1 for c in derived_cases if c["verdict"] == "PASS")
    n_route_cases = sum(1 for c in derived_cases if c["verdict"] == "PASS-with-finding")
    print("\n1.4a copied: %d cases (routers among them: %d). Bit-exact with no transform at all: %d. "
          "Bit-exact only after a documented converter transform: %d. Outside: %d"
          % (len(rows) - len(derived_cases), n_router,
             len(rows) - len(derived_cases) - len(permuted_notes) - len(fails), len(permuted_notes),
             len(fails)))
    print("1.4b derived-transcendental: %d cases, %d elements judged: %d bit-exact, %d within-route, "
          "%d outside, %d reference disagreements"
          % (len(derived_cases), sum(c["elts"] for c in derived_cases),
             sum(c["elements_bit_exact"] for c in derived_cases), n_within,
             sum(c["elements_outside_route"] for c in derived_cases),
             sum(c["reference_disagreements"] for c in derived_cases)))
    print("cases judged: %d = %d bit-exact + %d within-route%s"
          % (len(rows), n_bitexact_cases, n_route_cases,
             "" if not n_route_cases else " (a within-route case is NOT a bit-exact case)"))
    print("routes that matched: %s" % json.dumps(routes_seen, sort_keys=True))
    if v_order:
        counts = {}
        for nm, o in v_order.items():
            counts[o] = counts.get(o, 0) + 1
        print("GDN V order the published file actually carries (the question the reorder exists to "
              "answer): %s" % json.dumps(counts, sort_keys=True))
    if fails:
        print("STOP-THE-LINE findings (a wrong router row is the worst silent failure this model "
              "has): %s" % ", ".join(fails))
    for nm in near_misses:
        print("NEAR MISS — %s (%s, %s): %d of %d element(s) differ from the closest documented route "
              "(%s) by at most %d f32 ulp (indices %s). 1.4a's zero-tolerance verdict on this tensor "
              "stays FAIL; %s is judged by the route predicate instead, and the reason is in the "
              "section docstring."
              % (nm["name"], nm["hf"], nm["pass"], nm["elements_off"], nm["elts"], nm["best_route"],
                 nm["max_ulp"], nm["first_off_indices"], nm["name"]))
    if permuted_notes:
        print("NOTE — these match ONLY after a transform the pinned converter documents "
              "(conversion/qwen.py: GDN V reorder in _LinearAttentionVReorderBase, norm+1 and "
              "-exp(A_log) in Qwen3NextModel.modify_tensors): %s"
              % ", ".join("%s[%s]" % (a, b) for a, b in permuted_notes))
    ok_copied = bool(copied_targets) and not fails and n_router > 0
    ok_derived = all(c["verdict"] != "FAIL" for c in derived_cases)
    ok = ok_copied and ok_derived
    verdict = ("PASS-with-finding" if ok and (n_within or derived_cases) else
               "PASS" if ok else "FAIL")
    print("\n1.4 %s   (zero-tolerance view, kept for the record: %s on %s)"
          % (verdict, "FAIL" if zero_tol_fails else "PASS",
             ", ".join(zero_tol_fails) if zero_tol_fails else "nothing"))
    write_json(EXACT_JSON, {"utc": now(), "cases": len(rows), "router_cases": n_router,
                            "copied_cases": len(copied_targets), "derived_cases": len(derived_cases),
                            "routes_matched": routes_seen,
                            "gdn_v_order": v_order,
                            "permuted_only": ["%s: %s" % (a, b) for a, b in permuted_notes],
                            "near_misses": near_misses,
                            "derived_route": derived_cases,
                            "zero_tolerance_fails": zero_tol_fails,
                            "zero_tolerance_verdict": "FAIL" if zero_tol_fails else "PASS",
                            "fails": fails,
                            "rows": rows,
                            "verdict": verdict})
    auth.close()
    gg.close()
    return 0 if ok else 1


_FIRST_DATA_LOGGED = []


def first_data_log(gg, msg):
    if not _FIRST_DATA_LOGGED:
        _FIRST_DATA_LOGGED.append(1)
        log_order("E3: FIRST TENSOR-DATA READ — %s. bytes now reachable: header data_start=%d."
                  % (msg, gg.data_start))


def build_targets(gg, geo, auth, kinds=None):
    """The sampled (GGUF tensor, HF authority) pairs. Derived: walk the census for the sampled
    layers, map each name through the documented converter rules, keep what both files have."""
    idx = json.load(open(IDX))
    wm = idx["weight_map"]
    names = []
    for L in SAMPLE_LAYERS:
        names += [k for k in wm if k.startswith(hf_layer_prefix(L))]
    names += ["lm_head.weight", "model.language_model.embed_tokens.weight",
              "model.language_model.norm.weight"]
    out = []
    for hf in sorted(names):
        for gname, tr, extra in hf_to_gguf_names(hf):
            if gname is None or gname not in gg.by_name or not auth.has(hf):
                continue
            t = gg.by_name[gname]
            quantized = t.type_name not in ("F32", "F16", "BF16")
            kind = "dequant" if quantized else "exact"
            if kinds and kind not in kinds:
                continue
            out.append({"t": t, "hf": hf, "tr": tr, "extra": extra, "kind": kind,
                        "family": gguf_family(gname)})
    return out


# --------------------------------------------------------------------------------------------
# 1.5 — dequant comparison. Band first, then values.
# --------------------------------------------------------------------------------------------
def sampled_blocks(t, want_elts):
    """-> sorted list of whole-block indices. For a rank>=3 stack, stratified over the expert axis
    so a sample can never be all-of-expert-0 (that would make a transposed stack look clean)."""
    be, bb = gr.BLOCK[t.type_name]
    n_total = t.n_elts // be
    want = max(1, -(-want_elts // be))
    if len(t.dims) >= 3 and n_total > want:
        outer = t.dims[-1]
        per_outer = n_total // outer
        outer_sel = even_spaced(outer, min(8, outer))
        inner = max(1, want // len(outer_sel))
        out = set()
        for e in outer_sel:
            out.update(e * per_outer + b for b in even_spaced(per_outer, inner))
        return sorted(b for b in out if b < n_total), be
    return sorted(even_spaced(n_total, want)), be


def read_sample(gg, t, blocks, be, shift=0, why=""):
    out = []
    for bi in blocks:
        rel, bb = t.block_bytes(bi)
        out.append(gg.read(SELFTEST_TOKEN, gg.data_start + rel + shift, bb, why=why))
    return out


def sample_pair(gg, geo, auth, tg, want_elts, inj=None, why=""):
    """(published dequant, authority elements) for the same logical positions, plus the count."""
    t, hf_name, tr, extra = tg["t"], tg["hf"], tg["tr"], tg["extra"]
    m = make_mapper(geo, t, tr, extra, inj=inj)
    blocks, be = sampled_blocks(t, want_elts)
    raws = read_sample(gg, t, blocks, be, shift=1 if inj == "block_shift_1" else 0, why=why)
    deq = []
    for raw in raws:
        deq.extend(gr.dequant(raw, t.type_name, n_elts=be, inj=inj))
    idx = []
    for bi in blocks:
        base = bi * be
        if m is None:
            idx.extend(range(base, base + be))
        else:
            idx.extend([m(base + k) for k in range(be)])
    first_data_log(gg, "tensor-data read on %s (type %s, %d blocks x %d elts)"
                   % (t.name, t.type_name, len(blocks), be))
    return deq, auth.elements(hf_name, idx), len(blocks), be


def derived_route_inputs(gg, geo, auth, tg, documented_order=True, why=""):
    """(file values, {route: reference values}, n_elts) for a 1.4b target.

    `documented_order=False` deliberately reads the authority through the WRONG order (identity where
    the converter documents a reorder). That is only ever used to witness that the reorder question is
    load-bearing for these tensors: if a wrong A_log read still passed, the reorder claim would be
    untested rather than true.
    """
    t, hf_name, tr, extra = tg["t"], tg["hf"], tg["tr"], tg["extra"]
    vtr = value_transform(t.name)
    n = t.n_elts
    gbytes = gg.read(SELFTEST_TOKEN, gg.data_start + t.offset, t.row, why=why or "route pass on %s" % t.name)
    gvals = struct.unpack("<%df" % n, gbytes)
    otr = tr if documented_order else None
    m = (make_mapper(geo, t, otr, extra) or (lambda g: g)) if otr else (lambda g: g)
    vals0 = auth.elements(hf_name, [m(g) for g in range(n)])
    refs = {r: apply_value_transform(vtr, vals0, r) for r in DERIVED_ROUTE[vtr]}
    first_data_log(gg, "tensor-data read on %s (route %s)" % (t.name, vtr))
    return list(gvals), refs, n


# The 1.4b predicate's own must-catch set, as a claim about the route rather than about the band: what
# the predicate must refuse. Each entry is (must_fire, how the file is built). The ±1-step pair is the
# must-NOT-catch side, and it is the only place in this gate where "did not fire" is the required
# answer — which is exactly why it has to be stated as a case and counted, not assumed.
DERIVED_INJS = {"ssma_step_1_up": (False, "one hop above the agreed value, every element"),
                "ssma_step_1_down": (False, "one hop below the agreed value, every element"),
                "ssma_step_2_up": (True, "two hops above the agreed value, every element"),
                "ssma_step_2_down": (True, "two hops below the agreed value, every element"),
                "ssma_sign_flip": (True, "the negation of the agreed value, every element"),
                "ssma_element_perm": (True, "the file's own elements rotated by one position"),
                "ssma_wrong_authority_read": (True, "reference built from the UNDOCUMENTED order "
                                                    "(identity where the converter reorders)")}


def sec_floor_probe(args):
    print("1.5a — FLOOR PROBE. Bands come from here and nowhere else (pitfalls/30 §\"cancellation")
    print("band plus aggregate metrics, both sized by a floor probe\").")
    print("For every sampled tensor: quantize the bf16 AUTHORITY with our own quantizer at the")
    print("published width (same grid, same block geometry), dequantize that, and measure the")
    print("error achievable on this data at this width. The published file is then judged against")
    print("that floor times a margin — never against what the file happens to produce.")
    geo = Geo()
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)
    targets = build_targets(gg, geo, auth, kinds=("dequant",))
    rows, fam = [], {}
    for tg in targets:
        t = tg["t"]
        _, authv, nb, be = sample_pair(gg, geo, auth, tg, args.sample_elts,
                                       why="floor probe sample")
        dqf = gr.quant_ref_deq(t.type_name, authv)
        st = err_stats(dqf, authv)
        if not stats_ok(st):
            print("DEGENERATE floor sample on %s: %d non-finite elements — a floor that is not a "
                  "number cannot size a band" % (t.name, st["n_nonfinite"]))
            return 1
        rows.append({"name": t.name, "family": tg["family"], "type": t.type_name, "hf": tg["hf"],
                     "mapping": tg["tr"] or "identity", "blocks": nb, "block_elts": be,
                     "elts": st["n"], "floor_rms": st["rms_diff"], "floor_p999": st["p999_abs"],
                     "floor_max": st["max_abs"], "rms_auth": st["rms_auth"]})
        f = fam.setdefault(tg["family"], {"type": t.type_name, "n": 0, "floor_rms": 0.0,
                                          "floor_p999": 0.0, "floor_max": 0.0, "rms_auth": 0.0})
        f["n"] += 1
        for k, kk in (("floor_rms", "rms_diff"), ("floor_p999", "p999_abs"),
                      ("floor_max", "max_abs"), ("rms_auth", "rms_auth")):
            f[k] = max(f[k], st[kk])
    table(["tensor", "family", "type", "mapping", "blocks", "elts", "floor_rms", "floor_p999",
           "floor_max", "rms_authority"],
          [(r["name"], r["family"], r["type"], r["mapping"], r["blocks"], nines(r["elts"]),
            "%.3e" % r["floor_rms"], "%.3e" % r["floor_p999"], "%.3e" % r["floor_max"],
            "%.3e" % r["rms_auth"]) for r in rows])
    print("\nBand per family. The worst tensor in the family sets it; the margin is the one taste")
    print("number in this file and the injection table (1.5c) is its witness: a real misreading")
    print("lands orders of magnitude outside it, so the band is between the floor and the fault.")
    table(["family", "type", "tensors", "floor_rms", "band rms (x%.1f)" % RMS_MARGIN, "floor_max",
           "band max (x%.1f)" % MAX_MARGIN, "band as %% of authority rms"],
          [(f, v["type"], v["n"], "%.3e" % v["floor_rms"], "%.3e" % (RMS_MARGIN * v["floor_rms"]),
            "%.3e" % v["floor_max"], "%.3e" % (MAX_MARGIN * v["floor_max"]),
            "%.3f %%" % (100.0 * RMS_MARGIN * v["floor_rms"] / v["rms_auth"]))
           for f, v in sorted(fam.items())])
    ok = bool(rows) and len(rows) == len(targets)
    print("\nfloor probe %s (%d tensors sampled, %d families, %s elements/tensor cap)"
          % ("RUN" if ok else "FAIL", len(rows), len(fam), nines(args.sample_elts)))
    write_json(FLOOR_JSON, {"utc": now(), "rms_margin": RMS_MARGIN, "max_margin": MAX_MARGIN,
                            "sample_elts": args.sample_elts, "per_tensor": rows,
                            "per_family": fam, "verdict": "RUN" if ok else "FAIL"})
    auth.close()
    gg.close()
    return 0 if ok else 1


def sec_dequant(args):
    print("1.5b — DEQUANT COMPARISON: our own reader against the bf16 authority, inside the bands")
    if not os.path.exists(FLOOR_JSON):
        print("REFUSING: no floor probe at %s. A band must exist before a value is judged.\n"
              "Run: python3 tools/audit_published_quant.py --floor-probe" % (FLOOR_JSON,))
        return 1
    floor = json.load(open(FLOOR_JSON))
    bands = {f: {"rms": RMS_MARGIN * v["floor_rms"], "max": MAX_MARGIN * v["floor_max"],
                 "floor_rms": v["floor_rms"]} for f, v in floor["per_family"].items()}
    geo = Geo()
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)
    targets = build_targets(gg, geo, auth, kinds=("dequant",))
    print("      bands from %s (margin rms x%.1f, max x%.1f)."
          % (os.path.relpath(FLOOR_JSON, REPO), RMS_MARGIN, MAX_MARGIN))
    rows, fails = [], []
    for tg in targets:
        t = tg["t"]
        deq, authv, nb, be = sample_pair(gg, geo, auth, tg, args.sample_elts)
        st = err_stats(deq, authv)
        b = bands[tg["family"]]
        inside = stats_ok(st) and (st["rms_diff"] <= b["rms"]) and (st["max_abs"] <= b["max"])
        ratio = st["rms_diff"] / b["floor_rms"] if b["floor_rms"] else float("inf")
        rows.append({"name": t.name, "family": tg["family"], "type": t.type_name, "hf": tg["hf"],
                     "mapping": tg["tr"] or "identity", "blocks": nb, "elts": st["n"],
                     "rms_diff": st["rms_diff"], "max_abs": st["max_abs"],
                     "p999_abs": st["p999_abs"], "rms_auth": st["rms_auth"],
                     "floor_rms": b["floor_rms"], "ratio_to_floor": ratio,
                     "inside_band": inside})
        if not inside:
            fails.append(t.name)
    table(["tensor", "family", "type", "hf authority", "mapping", "elts", "rms_diff", "floor_rms",
           "x floor", "max_abs", "verdict"],
          [(r["name"], r["family"], r["type"], r["hf"], r["mapping"], nines(r["elts"]),
            "%.3e" % r["rms_diff"], "%.3e" % r["floor_rms"], "%.2f" % r["ratio_to_floor"],
            "%.3e" % r["max_abs"], "PASS" if r["inside_band"] else "FAIL") for r in rows])
    fam_worst = {}
    for r in rows:
        w = fam_worst.setdefault(r["family"], {"worst_x": 0.0, "n": 0, "type": r["type"]})
        w["worst_x"] = max(w["worst_x"], r["ratio_to_floor"])
        w["n"] += 1
    table(["family", "type", "tensors", "worst ratio to floor", "band margin", "verdict"],
          [(f, v["type"], v["n"], "%.2f" % v["worst_x"], "%.1f" % RMS_MARGIN,
            "PASS" if v["worst_x"] <= RMS_MARGIN else "FAIL")
           for f, v in sorted(fam_worst.items())])
    print("\ncases judged: %d; outside band: %d" % (len(rows), len(fails)))
    for f in fails:
        print("   OUT OF BAND: %s — a family outside its band is a finding to report, never a"
              " band to widen" % f)
    ok = bool(rows) and not fails and len(rows) == len(targets)
    print("1.5b %s" % ("PASS" if ok else "FAIL"))
    write_json(DEQ_JSON, {"utc": now(), "bands": bands, "per_tensor": rows,
                          "per_family_worst": fam_worst, "fails": fails,
                          "sample_elts": args.sample_elts, "verdict": "PASS" if ok else "FAIL"})
    auth.close()
    gg.close()
    return 0 if ok else 1


INJ_EXPECT = {
    # Derived from the tensor's own type / rank / name pattern, never from a hand-listed family.
    "q4k_as_q4nl": lambda tg: tg["t"].type_name == "Q4_K",
    "q4k_drop_min": lambda tg: tg["t"].type_name == "Q4_K",
    "gate_up_swap": lambda tg: "ffn_gate_exps" in tg["t"].name or "ffn_up_exps" in tg["t"].name,
    "expert_stack_transpose": lambda tg: len(tg["t"].dims) >= 3,
    "block_shift_1": lambda tg: True,
}


def sec_injections(args):
    print("1.5c — INJECTION MATRIX. Each misreading must push its OWN families outside the band")
    print("and must leave every other family inside it — that is the derived must-not-catch side")
    print("(localisation), and it is what makes a fired injection mean something.")
    if not os.path.exists(FLOOR_JSON):
        print("REFUSING: no floor probe yet.")
        return 1
    floor = json.load(open(FLOOR_JSON))
    bands = {f: {"rms": RMS_MARGIN * v["floor_rms"], "floor_rms": v["floor_rms"]}
             for f, v in floor["per_family"].items()}
    geo = Geo()
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)
    targets = build_targets(gg, geo, auth, kinds=("dequant",))
    mat = []
    for inj in sorted(INJ_EXPECT):
        for tg in targets:
            want = INJ_EXPECT[inj](tg)
            deq, authv, nb, be = sample_pair(gg, geo, auth, tg, args.inject_sample_elts, inj=inj,
                                             why="injection %s on %s" % (inj, tg["t"].name))
            st = err_stats(deq, authv)
            b = bands[tg["family"]]
            ok_stats = stats_ok(st)
            fired = (not ok_stats) or st["rms_diff"] > b["rms"]
            # A non-finite deviation is the FARTHEST possible, so it is carried as "non-finite" and
            # counted as fired. Recorded as None rather than NaN/Infinity so the JSON stays JSON.
            ratio = (st["rms_diff"] / b["floor_rms"]) if (ok_stats and b["floor_rms"]) else None
            mat.append({"injection": inj, "tensor": tg["t"].name, "family": tg["family"],
                        "type": tg["t"].type_name, "must_fire": want, "fired": fired,
                        "rms_diff": st["rms_diff"] if ok_stats else None,
                        "ratio_to_floor": ratio,
                        "nonfinite": st["n_nonfinite"],
                        "case_ok": fired if want else (not fired)})
    table(["injection", "tensor", "type", "must_fire", "fired", "x floor", "case"],
          [(r["injection"], r["tensor"], r["type"], "YES" if r["must_fire"] else "no",
            "YES" if r["fired"] else "no",
            "non-finite" if r["ratio_to_floor"] is None else "%.1f" % r["ratio_to_floor"],
            "ok" if r["case_ok"] else "BAD") for r in mat])
    per_inj = {}
    for r in mat:
        d = per_inj.setdefault(r["injection"], {"must": 0, "fired": 0, "must_not": 0, "leaked": 0,
                                                "min_fire_x": None, "max_clean_x": 0.0})
        if r["must_fire"]:
            d["must"] += 1
            d["fired"] += 1 if r["fired"] else 0
            if r["ratio_to_floor"] is not None:
                d["min_fire_x"] = (r["ratio_to_floor"] if d["min_fire_x"] is None
                                   else min(d["min_fire_x"], r["ratio_to_floor"]))
        else:
            d["must_not"] += 1
            d["leaked"] += 1 if r["fired"] else 0
            if r["ratio_to_floor"] is not None:
                d["max_clean_x"] = max(d["max_clean_x"], r["ratio_to_floor"])
    table(["injection", "must-fire", "fired", "leaked into families it must not touch",
           "min x floor when it fires", "max x floor when it must not fire"],
          [(k, v["must"], "%d/%d" % (v["fired"], v["must"]),
            "%d/%d" % (v["leaked"], v["must_not"]),
            "non-finite" if v["min_fire_x"] is None else "%.1f" % v["min_fire_x"],
            "%.2f" % v["max_clean_x"]) for k, v in sorted(per_inj.items())])
    # ---- the 1.4b predicate's own must-catch set --------------------------------------------
    # The band matrix above asks "does a misreading move the numbers?". These ask the different
    # question the relaxed predicate raises: "is the relaxation actually narrow?" A route allowance that
    # nobody tested at two steps is indistinguishable from one that accepts anything, so the ±1-step
    # cases must NOT fire and the ±2-step, sign-flip, permutation and wrong-order cases MUST.
    d_targets = [tg for tg in build_targets(gg, geo, auth, kinds=("exact",))
                 if tg["t"].type_name == "F32" and value_transform(tg["t"].name) in DERIVED_ROUTE]
    dmat = []
    for inj in sorted(DERIVED_INJS):
        want, how = DERIVED_INJS[inj]
        for tg in d_targets:
            fvals, refs, n = derived_route_inputs(
                gg, geo, auth, tg, documented_order=(inj != "ssma_wrong_authority_read"),
                why="route injection %s on %s" % (inj, tg["t"].name))
            ref0 = refs[sorted(refs)[0]]
            if inj == "ssma_step_1_up":
                fvals = [f32_nextafter(v, float("inf")) for v in ref0]
            elif inj == "ssma_step_1_down":
                fvals = [f32_nextafter(v, float("-inf")) for v in ref0]
            elif inj == "ssma_step_2_up":
                fvals = [f32_nextafter(f32_nextafter(v, float("inf")), float("inf")) for v in ref0]
            elif inj == "ssma_step_2_down":
                fvals = [f32_nextafter(f32_nextafter(v, float("-inf")), float("-inf")) for v in ref0]
            elif inj == "ssma_sign_flip":
                fvals = [-v for v in ref0]
            elif inj == "ssma_element_perm":
                fvals = fvals[1:] + fvals[:1]
            v = judge_derived_route(fvals, refs)
            dmat.append({"injection": inj, "tensor": tg["t"].name, "how": how, "must_fire": want,
                         "fired": v["fired"], "elts": n, "bit_exact": v["elements_bit_exact"],
                         "within_route": v["elements_within_route"],
                         "outside_route": v["elements_outside_route"],
                         "reference_disagreements": v["reference_disagreements"],
                         "case_ok": v["fired"] == want})
    print("\nroute-predicate matrix (the 1.4b allowance probed at its own boundary):")
    table(["injection", "tensor", "must_fire", "fired", "bit-exact", "within-route", "outside",
           "refs disagree", "case"],
          [(r["injection"], r["tensor"], "YES" if r["must_fire"] else "no",
            "YES" if r["fired"] else "no", r["bit_exact"], r["within_route"], r["outside_route"],
            r["reference_disagreements"], "ok" if r["case_ok"] else "BAD") for r in dmat])
    d_per = {}
    for r in dmat:
        d = d_per.setdefault(r["injection"], {"must": 0, "fired": 0, "must_not": 0, "leaked": 0,
                                              "how": r["how"]})
        if r["must_fire"]:
            d["must"] += 1
            d["fired"] += 1 if r["fired"] else 0
        else:
            d["must_not"] += 1
            d["leaked"] += 1 if r["fired"] else 0
    table(["route injection", "must-fire", "fired", "fired when it must not", "what it does"],
          [(k, v["must"], "%d/%d" % (v["fired"], v["must"]), "%d/%d" % (v["leaked"], v["must_not"]),
            v["how"]) for k, v in sorted(d_per.items())])
    dbad = [r for r in dmat if not r["case_ok"]]
    bad = [r for r in mat if not r["case_ok"]] + dbad
    print("\n'non-finite' means the injected misreading made the dequant values unorderable (NaN).")
    print("That counts as FIRED: `NaN > band` is False, so a gate that compared naively would have")
    print("reported a byte-misaligned read as clean — the same trap as a parser that prints PASS on")
    print("an empty parse, reached from the float side.")
    for r in bad[:20]:
        print("   BAD: %s on %s — expected %s, got %s"
              % (r["injection"], r["tensor"], "fire" if r["must_fire"] else "clean",
                 "fired" if r["fired"] else "clean"))
    print("\ncases: %d band injections (%d x %d tensors) + %d route injections (%d x %d tensors); "
          "wrong: %d" % (len(mat), len(INJ_EXPECT), len(targets), len(dmat), len(DERIVED_INJS),
                         len(d_targets), len(bad)))
    ok = bool(mat) and bool(d_targets) and not bad
    print("1.5c %s" % ("PASS" if ok else "FAIL"))
    write_json(INJ_JSON, {"utc": now(), "matrix": mat, "per_injection": per_inj,
                          "route_matrix": dmat, "route_per_injection": d_per,
                          "cases": len(mat) + len(dmat),
                          "cases_band": len(mat), "cases_route": len(dmat),
                          "wrong": ["%s on %s" % (r["injection"], r["tensor"]) for r in bad],
                          "verdict": "PASS" if ok else "FAIL"})
    auth.close()
    gg.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------
# 1.6 — shared storage, then the pin
# --------------------------------------------------------------------------------------------
def find_alias_groups(items):
    """items: (name, start, end). -> groups of >=2 names claiming the identical byte range."""
    by = {}
    for name, start, end in items:
        by.setdefault((start, end), []).append(name)
    return [sorted(v) for v in by.values() if len(v) > 1]


def sec_shared(args):
    print("1.6 — shared storage: two names over the same bytes must agree; the untied pair must NOT")
    print("be aliased. Then write the pin, so a silent re-upload under the same name is a mismatch.")
    fixture = [("a", 0, 10), ("b", 0, 10), ("c", 5, 15)]
    witness = find_alias_groups(fixture)
    witness_ok = witness == [["a", "b"]]
    print("detector self-witness on a known-aliased fixture: %s (%s) — without this a 0 below is"
          " vacuous" % ("OK" if witness_ok else "BROKEN", witness))
    gg = gr.Gguf(GGUF_PATH)
    auth = Authority(SELFTEST_TOKEN)
    gg_groups = find_alias_groups([(t.name, t.offset, t.offset + t.row) for t in gg.tensors])
    overlaps = []
    ts = sorted(gg.tensors, key=lambda t: t.offset)
    for a, b in zip(ts, ts[1:]):
        if a.offset + a.row > b.offset:
            overlaps.append((a.name, b.name))
    sf_groups, sf_rows = [], []
    for sh in SHARDS:
        hdr, _ = safetensors_header(os.path.join(SHARD_DIR, sh))
        items = [(k, v["data_offsets"][0], v["data_offsets"][1])
                 for k, v in hdr.items() if k != "__metadata__"]
        g = find_alias_groups(items)
        sf_groups += g
        sf_rows.append((sh, len(items), len(g)))
    agree_rows = []
    for grp in gg_groups:
        raws = []
        for nm in grp:
            t = gg.by_name[nm]
            raws.append(gg.read(SELFTEST_TOKEN, gg.data_start + t.offset, min(t.row, 1 << 20),
                                why="shared-storage agreement %s" % nm))
        agree_rows.append((", ".join(grp),
                           "IDENTICAL" if all(r == raws[0] for r in raws)
                           else "DIFFERENT — aliased names with different bytes"))
    tie_rows = []
    ta, tb = gg.by_name.get("token_embd.weight"), gg.by_name.get("output.weight")
    if ta and tb:
        n = 1 << 16
        ra = gg.read(SELFTEST_TOKEN, gg.data_start + ta.offset, n, why="untied witness")
        rb = gg.read(SELFTEST_TOKEN, gg.data_start + tb.offset, n, why="untied witness")
        diff = sum(1 for x, y in zip(ra, rb) if x != y)
        tie_rows.append(("token_embd.weight", "output.weight", ta.offset, tb.offset,
                         "%d/%d bytes differ in the first 64 KiB" % (diff, n),
                         "DISTINCT" if diff else "ALIASED — config says tie_word_embeddings=False"))
    table(["shard", "tensors", "alias groups found"], [(r[0], r[1], r[2]) for r in sf_rows])
    print("safetensors alias groups in the 4 fetched shards: %d%s"
          % (len(sf_groups), "" if not sf_groups else " -> %s" % sf_groups[:5]))
    print("GGUF alias groups: %d; overlapping byte ranges: %d%s"
          % (len(gg_groups), len(overlaps), "" if not overlaps else " -> %s" % overlaps[:5]))
    for r in agree_rows:
        print("   aliased GGUF names %s: %s" % r)
    table(["tensor A", "tensor B", "offset A", "offset B", "witness", "verdict"], tie_rows)
    ok = (witness_ok and not overlaps and all("IDENTICAL" in r[1] for r in agree_rows)
          and bool(tie_rows) and all(r[5].startswith("DISTINCT") for r in tie_rows))
    briefs = {}
    for p in (HASH_JSON, SHARD_JSON, STRUCT_JSON, EXACT_JSON, FLOOR_JSON, DEQ_JSON, INJ_JSON):
        if os.path.exists(p):
            briefs[os.path.basename(p)] = _brief(json.load(open(p)))
    write_json(SHARED_JSON, {"utc": now(), "detector_witness": witness_ok,
                             "safetensors_alias_groups": len(sf_groups),
                             "gguf_alias_groups": gg_groups, "gguf_overlaps": overlaps,
                             "alias_byte_agreement": agree_rows, "untied_witness": tie_rows,
                             "verdict": "PASS" if ok else "FAIL"})
    if ok:
        h = briefs.get(os.path.basename(HASH_JSON), {})
        write_json(PIN_JSON, {
            "artifact": os.path.basename(GGUF_PATH),
            "source_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "revision": h.get("revision") or GGUF_REVISION,
            "sha256": h.get("sha256_pass1"),
            "bytes_on_disk": h.get("bytes_on_disk"),
            "gguf_header": {"architecture": h.get("architecture"),
                            "general.file_type": h.get("file_type"),
                            "general.quantization_version": h.get("quantization_version"),
                            "n_tensors": h.get("n_tensors"),
                            "general.alignment": h.get("alignment"),
                            "data_start": h.get("data_start")},
            "authority": {"repo": "Qwen/Qwen3.6-35B-A3B", "revision": SHARD_REVISION,
                          "shards_fetched": SHARDS,
                          "mtp_extracted_to": os.path.relpath(MTP_OUT, REPO),
                          "mtp_elements": briefs.get(os.path.basename(SHARD_JSON), {})
                              .get("mtp", {}).get("elements")},
            "audit_sections": briefs,
            "recorded_utc": now(),
            "note": "written by tools/audit_published_quant.py --shared; add a row for this file"
                    " to checkpoints/README.md when committing",
        })
    print("\n1.6 %s; pin -> %s" % ("PASS" if ok else "FAIL", os.path.relpath(PIN_JSON, REPO)))
    auth.close()
    gg.close()
    return 0 if ok else 1


def _brief(d):
    keep = ("utc", "verdict", "sha256_pass1", "sha256_pass2", "revision", "etag", "file_type",
            "quantization_version", "architecture", "n_tensors", "bytes_on_disk", "header_end",
            "data_start", "alignment", "weight_bytes", "size_reconciles", "cases", "router_cases",
            "copied_cases", "derived_cases", "fails", "zero_tolerance_verdict",
            "zero_tolerance_fails", "permuted_only", "per_family", "per_family_worst", "mtp",
            "sample_elts", "rms_margin", "max_margin", "index_tensor_count", "detector_witness",
            "safetensors_alias_groups", "gguf_alias_groups", "cases_band", "cases_route")
    out = {k: v for k, v in d.items() if k in keep}
    if "derived_route" in d:      # compact: the pin carries the finding, not every element dump
        out["derived_route_finding"] = [
            {"name": c["name"], "verdict": c["verdict"], "route": c["route"], "order": c["order"],
             "elements_within_route": c["elements_within_route"],
             "elements_bit_exact": c["elements_bit_exact"],
             "max_steps_from_agreed": c["max_steps_from_agreed"],
             "within_route_indices": [w["index"] for w in c["within_route_elements"]]}
            for c in d["derived_route"]]
    if "per_family_worst" in out:
        out["per_family_worst"] = {k: v["worst_x"] for k, v in out["per_family_worst"].items()}
    if "per_family" in out and isinstance(out["per_family"], dict):
        out["per_family"] = {k: {"type": v["type"], "n": v["n"], "floor_rms": v["floor_rms"],
                                 "floor_max": v["floor_max"]}
                             for k, v in out["per_family"].items()}
    return out


# --------------------------------------------------------------------------------------------
# the markdown summary (built from the section JSONs, never from memory)
# --------------------------------------------------------------------------------------------
def shard_layers():
    """Which decoder layers the fetched shards actually cover — read out of the shard headers."""
    layers = set()
    for sh in SHARDS:
        hdr, _ = safetensors_header(os.path.join(SHARD_DIR, sh))
        for k in hdr:
            m = re.match(r"^model\.language_model\.layers\.(\d+)\.", k)
            if m:
                layers.add(int(m.group(1)))
    return sorted("%d" % x for x in layers)


def shards_verdict(sh):
    if not sh:
        return "MISSING"
    ok = bool(sh.get("shards")) and all(v["set_matches"] and v["etag_matches"]
                                        for v in sh["shards"].values())
    m = sh.get("mtp", {})
    ok = ok and m.get("present") and m.get("extract_reparse_ok")
    return "PASS" if ok else "FAIL"


def findings(st, sh, ex, dq, inj, h, sd):
    out = []
    a = out.append
    m = sh.get("mtp", {})
    if m.get("plan_claim_elements"):
        a("* **PLAN's draft-block arithmetic is wrong by 20x.** The 19 `mtp.*` tensors are %s "
          "elements = %.1f MiB bf16, not the 41,943,040 elements / 79.9 MiB step 1's fetch list "
          "claims. Nothing downstream that budgets residency for the draft block can keep the "
          "smaller number. (Evidence: `%s` §mtp.)"
          % (nines(m["elements"]), m["bytes"] / 2.0 ** 20,
             os.path.basename(SHARD_JSON)))
    reasons = st.get("hf_absent_reasons", {})
    if reasons.get("absent-mtp"):
        a("* **The published GGUF carries no draft block at all.** %d checkpoint names have no "
          "counterpart, all of them `mtp.*` (plus %d vision tensors, out of scope by D-04). The "
          "QWEN35MOE template does list `NEXTN_*` tensors and none appear. MTP is in scope (D-05), "
          "so this file cannot serve it; the extracted `mtp.*` set at `%s` is what the engine has "
          "to work from."
          % (reasons["absent-mtp"], reasons.get("absent-vision", 0),
             m.get("extracted_to", "models/w1-mtp-extract/mtp.safetensors")))
    if ex.get("gdn_v_order"):
        tiled = sum(1 for o in ex["gdn_v_order"].values() if o == "V-reorder")
        a("* **The GDN V heads are stored in the converter's tiled order** (%d of %d judged GDN "
          "tensors), not HF's grouped order. The engine must read them tiled or permute on load; "
          "this is invisible to any test that does not compare against the checkpoint."
          % (tiled, len(ex["gdn_v_order"])))
    a("* **Norms are stored as `HF + 1`, computed in float32**, and `ssm_a` as `-exp(A_log)` "
      "(also float32): a loader that adds 1 in bf16 loses bits the file keeps. Routes are listed "
      "in `%s` §routes_matched." % os.path.basename(EXACT_JSON))
    a("* **`attn_q` is 2x the query rows**: `attn_output_gate=True`, so `self_attn.q_proj` is "
      "[2*nq*hd, H] = [8192, 2048] and the GGUF carries the gate half too. A reader sized for "
      "nq*hd rows reads half a tensor and sees no error.")
    a("* **`ffn_gate_inp_shexp` is rank 1 in the file, rank 2 in the checkpoint** ([1, 2048] -> "
      "[2048]) on all %d layers: `ggml_n_dims()` (ggml/src/ggml.c) drops trailing singleton dims, "
      "so element counts still match. Recorded as its own class in `%s` rather than folded into "
      "\"dims match\"." % (len(st.get("rank_convention_rows", [])), os.path.basename(STRUCT_JSON)))
    counts = {}
    for r in st.get("per_tensor", []):
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    dist = ", ".join("%s %d" % (t, counts[t]) for t in sorted(counts))
    a("* **The producer quantizes per tensor, not per file type.** Stored widths across the 733 "
      "records: %s. So `general.file_type = 15 (Q4_K_M)` describes the headline, not the file: it "
      "names the width of %d tensors out of 733, and every norm, router and GDN-control tensor is "
      "F32. Any residency or bandwidth model that multiplies 66.7 GB by one Q4_K_M ratio is wrong "
      "by construction (family table above; per-tensor list in `%s`)."
      % (dist, counts.get("Q4_K", 0), os.path.basename(STRUCT_JSON)))
    if inj:
        band_txt = ", ".join("`%s` %d/%d" % (k, v["fired"], v["must"])
                             for k, v in sorted(inj["per_injection"].items()))
        route = inj.get("route_per_injection", {})
        route_txt = ""
        if route:
            route_txt = ("; the 1.4b route allowance is probed the same way (" + ", ".join(
                "`%s` %d/%d" % (k, v["fired"], v["must"])
                for k, v in sorted(route.items()) if v["must"])
                + ", and the +/-1-step cases correctly stayed clean)")
        a("* **The gate can see the faults it claims to see**: %s%s, and none of them leaks into a "
          "family it must not touch (`%s`)."
          % (band_txt, route_txt, os.path.basename(INJ_JSON)))
    wd = width_drift(st, h)
    a("* **Two width tables in this repo were wrong on Q5_K and the audit's own prose was wrong about "
      "what that was worth.** The pinned struct is 176 B/256 elts = 5.5 bpw, not the hand-typed "
      "5.5625; on this file that is %.2f MiB over %d tensors, not the ~232 MiB an earlier draft "
      "claimed. Widths are now derived from the header at run time and both tools abort if the file "
      "size or a policy width stops matching. Nothing in the shipping arithmetic moves: no policy row "
      "uses Q5_K, so policy A stays 19.59 GiB resident / 2.331 GiB per token. Arithmetic and per-type "
      "table: §\"A hand-typed width\" below and `audit-block-geometry-%s.md`."
      % (sum(r.get("drift_mib", 0.0) for r in wd["per_type"]),
         sum(r["tensors"] for r in wd["per_type"] if r.get("drift_mib")), DAY))
    if dq:
        a("* **No family is even close to its band**: the worst published rms is %.2fx its own floor "
          "against a %.1fx band, i.e. the published quantization is as good as an honest "
          "single-pass quantizer at those widths."
          % (max(v["worst_x"] for v in dq["per_family_worst"].values()), RMS_MARGIN))
    return out


def not_claimed():
    return [
        "* This is a byte/structure fidelity audit of one file. It does not license the widths "
        "(policy A of `quantization/README.md` and D-01/D-01a already do) and it is not a quality "
        "measurement: nothing here says a Q4_K_M token is any good.",
        "* Only 4 of 26 shards were fetched, so the exactness and dequant passes judge layers %s "
        "and the router/GDN tensors of those layers; the structure pass is still tensor-for-tensor "
        "over all 733 records because the census carries every name and shape."
        % ", ".join(shard_layers()),
        "* Nothing here was checked against llama.cpp at runtime: no llama.cpp binary was built or "
        "run, and the reader here is this repo's own, written from `ggml-common.h` / "
        "`ggml-quants.c`. That is deliberate (PLAN §\"Why an audit exists at all\") — but it means "
        "\"the file is fine and llama.cpp disagrees\" is a failure mode this pass cannot see.",
        "* Element-count-per-block in `tools/ggml_block_geom.py` is a claim about each ggml type, not "
        "a reading of a struct; it is witnessed by the file-size identity over all 733 tensors, not "
        "by the header text. Every bytes-per-block figure is read from the pinned header.",
    ]


def width_drift(st, h):
    """The hand-typed-width story, computed from the file rather than recited from memory.

    This section exists because an earlier draft of this audit claimed the retired 5.5625 bpw row was
    worth "~232 MiB" here. It was worth 74.00 MiB, and the only way a wrong number in prose stays wrong
    is for nothing to recompute it. Everything below is derived from the tensor records (1.3) and the
    struct sizes `tools/ggml_block_geom.py` reads out of the pinned header; the single non-derived input
    is the retired hand value itself, named where it is used.
    """
    retired_bpw = {"Q5_K": 5.5625}          # what tools/quant_scenarios.py and gguf_header_types.py said
    g = GBG.geom()
    per, tot = {}, 0
    for r in st.get("per_tensor", []):
        tname = r["type"]
        n = 1
        for d in json.loads(r["dims"].replace("'", '"')):
            n *= int(d)
        rb = GBG.row_bytes(n, tname)
        tot += rb
        p = per.setdefault(tname, {"tensors": 0, "elts": 0, "bytes": 0})
        p["tensors"] += 1
        p["elts"] += n
        p["bytes"] += rb
    ident = {"sum_row_bytes": tot, "data_start": h.get("data_start"), "file_size": h.get("bytes_on_disk"),
             "reconciles": (h.get("data_start") is not None and h.get("bytes_on_disk") is not None
                            and tot + h["data_start"] == h["bytes_on_disk"])}
    rows = []
    for tname in sorted(per, key=lambda t: -per[t]["bytes"]):
        p = per[tname]
        row = {"type": tname, "tensors": p["tensors"], "elts": p["elts"], "bytes": p["bytes"],
               "gib": p["bytes"] / 2.0 ** 30, "bpw_derived": g[tname]["bpw"],
               "block": "%d B / %d elts" % (g[tname]["bytes"], g[tname]["elts"]),
               "retired_bpw": retired_bpw.get(tname)}
        if tname in retired_bpw:
            be, bb = g[tname]["elts"], g[tname]["bytes"]
            retired_bb = int(round(retired_bpw[tname] * be / 8.0))
            blocks = p["elts"] // be
            row["retired_bytes"] = blocks * retired_bb
            row["drift_bytes"] = blocks * (retired_bb - bb)
            row["drift_mib"] = row["drift_bytes"] / 2.0 ** 20
            row["blocks"] = blocks
            row["derived_block_bytes"] = bb
            row["retired_block_bytes"] = retired_bb
        rows.append(row)
    return {"identity": ident, "per_type": rows}


def sec_summary(args):
    parts = {}
    for tag, p in (("selftest", SELFTEST_MARK), ("hash", HASH_JSON), ("shards", SHARD_JSON),
                   ("structure", STRUCT_JSON), ("exact", EXACT_JSON), ("floor", FLOOR_JSON),
                   ("dequant", DEQ_JSON), ("injections", INJ_JSON), ("shared", SHARED_JSON)):
        if os.path.exists(p):
            parts[tag] = json.load(open(p))
    need = ("selftest", "hash", "shards", "structure", "exact", "floor", "dequant", "injections",
            "shared")
    missing = [k for k in need if k not in parts]
    h, st, ex = parts.get("hash", {}), parts.get("structure", {}), parts.get("exact", {})
    dq, inj, fl = parts.get("dequant", {}), parts.get("injections", {}), parts.get("floor", {})
    sh, sd = parts.get("shards", {}), parts.get("shared", {})
    L = []
    A = L.append
    A("# Audit of the published Q4_K_M against the bf16 checkpoint (%s)" % DAY)
    A("")
    A("Artifact: `models/unsloth-q4km/%s`, from `unsloth/Qwen3.6-35B-A3B-GGUF` revision"
      % os.path.basename(GGUF_PATH))
    A("`%s`. Authority: `Qwen/Qwen3.6-35B-A3B` revision `%s`; the %d fetched shards carry layers %s"
      % (GGUF_REVISION, SHARD_REVISION, len(SHARDS), ", ".join(shard_layers())))
    A("plus the draft block's 19 `mtp.*` tensors. Instrument: `tools/audit_published_quant.py` with")
    A("`tools/gguf_reader.py` (stdlib `python3` %s — no numpy, no torch, no llama.cpp binary)."
      % sys.version.split()[0])
    A("Raw output is the JSON set beside this file; the design-before-bytes ordering is evidenced in")
    A("`audit-ordering-%s.log`. CPU only: **no timing taken**." % DAY)
    A("")
    A("| step | pass condition (PLAN.md) | verdict |")
    A("|---|---|---|")
    A("| 1.0 gate authored from the spec | `--selftest` passes on synthetic blocks and every"
      " injected bug fires | %s |"
      % ("PASS at %s" % parts.get("selftest", {}).get("utc_first_selftest_pass", "-")
         if "selftest" in parts else "MISSING"))
    A("| 1.1 sha / revision / header | sha re-reads identical, header parses to %d records | %s |"
      % (EXPECTED_TENSOR_RECORDS, h.get("verdict", "MISSING")))
    A("| 1.2 shard sets vs committed weight_map | each shard matches; `mtp.*` extracted | %s |"
      % shards_verdict(sh))
    A("| 1.3 structure pass over every tensor record | reconciles tensor-for-tensor before any"
      " value is judged | %s |" % st.get("verdict", "MISSING"))
    A("| 1.4a copied families, zero tolerance | routers, norms, dt_bias, alpha, beta, conv1d and the "
      "order/reorder questions bit-exact against bf16 widened (with the converter's documented "
      "transforms) | %s |"
      % ("PASS" if ex.get("copied_cases") and not ex.get("fails") else
         (ex.get("verdict", "MISSING") if ex else "MISSING")))
    A("| 1.4b derived-transcendental route predicate | `ssm_a` = -exp(A_log): two correctly-rounded "
      "routes agree with each other, file within one f32 step of the agreed value | %s |"
      % ex.get("verdict", "MISSING"))
    A("| 1.5b dequant comparison | inside floor-probe bands | %s |" % dq.get("verdict", "MISSING"))
    A("| 1.5c injections | each fires only where it belongs | %s |"
      % inj.get("verdict", "MISSING"))
    A("| 1.6 shared storage + pin | identity holds, pin written | %s |"
      % sd.get("verdict", "MISSING"))
    A("")
    if missing:
        A("**Not run yet: %s.** Do not read this table as a finished audit." % ", ".join(missing))
        A("")
    if h:
        A("## 1.1 identity")
        A("")
        A("| field | value |")
        A("|---|---|")
        A("| sha256 (two independent reads, identical: %s) | `%s` |"
          % (h.get("sha256_pass1") == h.get("sha256_pass2"), h.get("sha256_pass1")))
        A("| HF revision / etag equals the computed sha | `%s` / %s |"
          % (h.get("revision"), h.get("etag") == h.get("sha256_pass1")))
        A("| `general.architecture` | %s |" % h.get("architecture"))
        A("| `general.file_type` / `general.quantization_version` | %s / %s |"
          % (h.get("file_type"), h.get("quantization_version")))
        A("| tensor records parsed (= declared = expected %d) | %s |"
          % (EXPECTED_TENSOR_RECORDS, h.get("n_tensors")))
        A("| header_end / data_start / alignment | %s / %s / %s |"
          % (h.get("header_end"), h.get("data_start"), h.get("alignment")))
        A("| sum(tensor row bytes) + data_start == file size | %s |" % h.get("size_reconciles"))
        A("")
    if sh.get("mtp"):
        m = sh["mtp"]
        A("## 1.2 the shards and the draft block")
        A("")
        A("| fact | value |")
        A("|---|---|")
        A("| shards fetched / sets matching the committed `weight_map` | %d / %d |"
          % (len(sh.get("shards", {})), sum(1 for v in sh.get("shards", {}).values()
                                             if v["set_matches"])))
        A("| `mtp.*` tensors | %d, all present |" % m["names"])
        A("| derived `mtp.*` element total | %s elts = %.4f MiB bf16 |"
          % (nines(m["elements"]), m["bytes"] / 2.0 ** 20))
        A("| PLAN.md step 1 claimed | %s elts = 79.9 MiB → **%s** |"
          % (nines(m["plan_claim_elements"]), m["plan_claim_verdict"]))
        A("| extraction written | `%s` |" % m["extracted_to"])
        A("")
    if st:
        A("## 1.3 what the file actually stores, tensor-for-tensor")
        A("")
        A("| family | tensors | types stored |")
        A("|---|---|---|")
        for f in sorted(st["families"]):
            A("| %s | %d | %s |" % (f, st["families"][f], ", ".join(st["types_by_family"][f])))
        A("| **total records** | **%d** | |" % st["n_tensors"])
        A("")
        A("Checkpoint names with no GGUF counterpart: %s"
          % json.dumps(st["hf_absent_reasons"], sort_keys=True))
        A("")
        wd = width_drift(st, h)
        ident = wd["identity"]
        A("### A hand-typed width: the 74 MiB, and where my 232 MiB went")
        A("")
        A("Two tools here priced this table from hand-typed bits-per-weight and one row disagreed with")
        A("ggml: Q5_K was written 5.5625 bpw where the pinned struct is 176 B per 256 elements = 5.5")
        A("bpw. Widths are now read from the header at run time by `tools/ggml_block_geom.py`, which")
        A("requires each struct's member sum and that struct's own `static_assert` to agree, and")
        A("`tools/gguf_header_types.py` now refuses to print a width table unless the byte identity")
        A("below holds. Derived widths and drift rows: `audit-block-geometry-%s.md`. The same diff "
          "re-audited every row of the old table and found exactly one wrong: Q4_K 4.5, Q6_K 6.5625, "
          "Q8_0 8.5, Q4_0 4.5, IQ4_XS 4.25 and MXFP4 4.25 all matched the header already. One wrong "
          "row in eight is precisely the shape that survives a spot-check, which is why the table is "
          "gone rather than corrected." % DAY)
        A("")
        A("| stored type | tensors | elements | bytes (block math) | GiB | bpw derived | block |")
        A("|---|---:|---:|---:|---:|---:|---|")
        for r in wd["per_type"]:
            A("| %s | %d | %s | %s | %.4f | %s | %s |"
              % (r["type"], r["tensors"], nines(r["elts"]), nines(r["bytes"]), r["gib"],
                 ("%.4f" % r["bpw_derived"]).rstrip("0").rstrip("."), r["block"]))
        A("")
        A("The identity that licenses those bytes: `sum(row_bytes) + data_start == file size` = "
          "`%s + %s == %s` -> **%s**. A width table that cannot reproduce the file's own byte count "
          "is not printed at all, which is the difference between a derived number and a plausible "
          "one."
          % (nines(ident["sum_row_bytes"]), nines(ident["data_start"]), nines(ident["file_size"]),
             "RECONCILES" if ident["reconciles"] else "MISMATCH"))
        A("")
        q5 = [r for r in wd["per_type"] if r["type"] == "Q5_K"]
        if q5:
            r = q5[0]
            A("What the wrong row was worth on this file, with the arithmetic shown rather than "
              "asserted: the Q5_K family is %s tensors of %s elements each = %s blocks of 256 per "
              "tensor, %s blocks in total. At the derived %d B/block that is %s bytes = **%.4f GiB**; "
              "at the retired %s bpw (= %d B/block) it would have been %s bytes. The overstatement is "
              "%d B per block over %s blocks = **%.2f MiB** (%.6f GiB). Cross-checking from the family "
              "bytes instead of the block count: %.4f GiB x 0.0625/5.5 = %.4f GiB, same answer. An "
              "earlier draft of this audit said ~232 MiB; that number was wrong, and the only reason "
              "it is not still in this file is that this section recomputes it instead of quoting it."
              % (r["tensors"], nines(r["elts"] // r["tensors"]),
                 nines(r["blocks"] // r["tensors"]), nines(r["blocks"]), r["derived_block_bytes"],
                 nines(r["bytes"]), r["gib"], ("%g" % r["retired_bpw"]), r["retired_block_bytes"],
                 nines(r["retired_bytes"]), r["retired_block_bytes"] - r["derived_block_bytes"],
                 nines(r["blocks"]), r["drift_mib"], r["drift_bytes"] / 2.0 ** 30, r["gib"],
                 r["drift_bytes"] / 2.0 ** 30))
        A("")
        A("**This does not move the shipping arithmetic.** `tools/quant_scenarios.py` policy rows use "
          "Q4_K / Q8_0 / bf16 / IQ4_XS / MXFP4, and Q6_K only for `lm_head` in E and G; **no policy row "
          "uses Q5_K**, so policy A's 19.59 GiB resident and 2.331 GiB/token (D-01/D-11) are unchanged, "
          "and the tool now aborts if any policy width stops matching a derived struct width. Only the "
          "published-file width-map GiB column above changed. Anyone tempted to \"correct\" a policy row "
          "with 5.5 bpw has corrected a row that was never Q5_K.")
        A("")
        wctl = os.path.join(EV, "audit-width-negative-controls-%s.json" % DAY)
        if os.path.exists(wctl):
            wc = json.load(open(wctl))
            A("A derived table nobody has seen refuse anything is a hand-typed table with extra steps, "
              "so the machinery is probed four ways (`python3 tools/audit_published_quant.py "
              "--selfcheck-widths`, verdict %s):" % wc.get("verdict"))
            A("")
            A("| control | fired? | detail |")
            A("|---|---|---|")
            for c in wc["controls"]:
                A("| %s | %s | %s |" % (c["control"], c["did"], c["detail"]))
            A("")
    if ex:
        A("## 1.4 exactness: 1.4a copied (zero tolerance) and 1.4b derived-transcendental "
          "(route predicate)")
        A("")
        A("| what | value |")
        A("|---|---|")
        A("| cases judged (F32 families, layers %s) | %d = %d bit-exact + %d within-route |"
          % (SAMPLE_LAYERS, ex["cases"],
             ex["cases"] - sum(1 for c in ex.get("derived_route", [])
                               if c["verdict"] == "PASS-with-finding"),
             sum(1 for c in ex.get("derived_route", [])
                 if c["verdict"] == "PASS-with-finding")))
        A("| 1.4a copied cases / router tensors among them | %d / %d |"
          % (ex.get("copied_cases", 0), ex["router_cases"]))
        A("| 1.4b derived-transcendental cases (`ssm_a` = -exp(A_log)) | %d |"
          % ex.get("derived_cases", 0))
        A("| matched bit-exactly as stored | %d |"
          % sum(1 for r in ex["rows"]
                if r["matched_under"].startswith("as stored + no value transform")))
        A("| matched bit-exactly only after a documented converter transform | %d |"
          % len(ex["permuted_only"]))
        A("| 1.4a zero-tolerance failures | %d |" % len(ex["fails"]))
        A("| zero-tolerance verdict on the whole set (kept, not replaced) | %s — %s |"
          % (ex.get("zero_tolerance_verdict", "-"), ", ".join(ex.get("zero_tolerance_fails", []))
             or "nothing"))
        A("| route-predicate totals | %d elements bit-exact, %d within-route, %d outside, %d "
          "reference disagreements |"
          % (sum(c["elements_bit_exact"] for c in ex.get("derived_route", [])),
             sum(c["elements_within_route"] for c in ex.get("derived_route", [])),
             sum(c["elements_outside_route"] for c in ex.get("derived_route", [])),
             sum(c["reference_disagreements"] for c in ex.get("derived_route", []))))
        A("")
        if ex.get("gdn_v_order"):
            orders = {}
            for nm, o in ex["gdn_v_order"].items():
                orders[o] = orders.get(o, 0) + 1
            A("The GDN tensors carry the **%s** order of the V heads (%d tensors) — the question the "
              "reorder exists to answer, because a runtime that assumes the other order reads the "
              "wrong heads (`conversion/qwen.py §_LinearAttentionVReorderBase`)."
              % ("tiled (converter-reordered)" if any(o == "V-reorder"
                                                     for o in ex["gdn_v_order"].values())
                 else "grouped (HF)", len(ex["gdn_v_order"])))
            A("")
        if ex["permuted_only"]:
            A("These are bit-exact ONLY after a transform the pinned converter documents "
              "(GDN V reorder, `norm+1`, `-exp(A_log)`):")
            A("")
            for s in ex["permuted_only"]:
                A("* `%s` — %s" % tuple(s.split(": ", 1)))
            A("")
        for nm in ex.get("near_misses", []):
            A("**Within-route, not bit-exact** — `%s`: %d of %d element(s) sit %d f32 step from the "
              "value both correctly-rounded routes agree on (route `%s`), indices %s. The "
              "zero-tolerance comparison still calls this a FAIL and that line is kept in `%s` "
              "beside the new verdict; the reason 1.4b does not demand bit-exactness is in the next "
              "section."
              % (nm["name"], nm["elements_off"], nm["elts"], nm["max_ulp"], nm["best_route"],
                 nm["first_off_indices"], os.path.basename(EXACT_JSON)))
            A("")
        for c in ex.get("derived_route", []):
            if not c["within_route_elements"]:
                continue
            A("| element index | file (f32) | file bits | route `f32` value | its bits | route `crl` "
              "value | its bits | steps away | one step is |")
            A("|---:|---|---|---|---|---|---|---:|---|")
            for w in c["within_route_elements"]:
                A("| %d | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %d | `%s` |"
                  % (w["index"], w["file"], w["file_bits"], w["agreed"],
                     w["reference_bits"]["f32"], w["agreed"], w["reference_bits"]["crl"],
                     w["steps_from_agreed"], w["ulp_step_at_agreed"]))
            A("")
        nm_names = {n["name"] for n in ex.get("near_misses", [])}
        for f in ex.get("fails", []):
            if f not in nm_names:
                A("**Failure** — `%s`: no documented route reproduces it bit-exactly (see `%s`)."
                  % (f, os.path.basename(EXACT_JSON)))
        A("")
        A("### Why 1.4 has two halves, and who said so")
        A("")
        A("The supervisor authorised this split on %s, as a **route-derived predicate, not a "
          "tolerance**: bit-exactness was the wrong predicate for `ssm_a` because HF never stores it "
          "(it computes `A = -exp(A_log.float())` at module init), so there is no stored authority to "
          "be bit-exact against — demanding it would demand that the producer's libm agreed with ours. "
          "It is not an engine-fidelity property either: this engine reads `ssm_a` from the file and "
          "never recomputes the exp. The underlying defect was PLAN 1.4 lumping a derived "
          "transcendental in with the copied families. This is surfaced to the user for ratification; "
          "if they overrule it the finding above is intact and the verdict flips back with one flag."
          % DECISION_DAY)
        A("")
        A("What the predicate is, and what keeps it honest:")
        A("")
        A("* two independent correctly-rounded routes — `exp` in double then rounded to f32, and `exp` "
          "to 60 digits correctly rounded to f32 once — must agree with **each other** bit-for-bit on "
          "every judged element; if they ever disagree the case FAILS rather than being absorbed;")
        A("* the file may then be the agreed value or one `nextafter` hop either side of it, no looser. "
          "There is no ulp knob and no epsilon constant in the tool: the bound is the spacing at that "
          "value, and `%s §selftest_route_predicate` witnesses the spacing against hand-worked "
          "neighbours and the predicate at 1, 2, sign-flipped, permuted and binade-edge offsets;"
          % os.path.basename(os.path.join("tools", os.path.basename(__file__))));
        A("* the must-catch set grew with the relaxation: `%s` carries %d route cases where +/-2 steps, "
          "sign flip, element permutation and a wrong-order `A_log` read must all fire and +/-1 step "
          "must not. A relaxation nobody probes at two steps is indistinguishable from one that "
          "accepts anything."
          % (os.path.basename(INJ_JSON), len(inj.get("route_matrix", []) or [])))
        A("")
    if dq and fl:
        A("## 1.5 dequant comparison, bands sized by the floor probe")
        A("")
        A("| family | type | tensors | floor rms | band rms (x%.1f) | worst published rms |"
          " worst x floor | verdict |" % RMS_MARGIN)
        A("|---|---|---|---|---|---|---|---|")
        for f in sorted(dq["per_family_worst"]):
            fv, w = fl["per_family"][f], dq["per_family_worst"][f]
            A("| %s | %s | %d | %.3e | %.3e | %.3e | %.2f | %s |"
              % (f, fv["type"], fv["n"], fv["floor_rms"], RMS_MARGIN * fv["floor_rms"],
                 max(r["rms_diff"] for r in dq["per_tensor"] if r["family"] == f), w["worst_x"],
                 "PASS" if w["worst_x"] <= RMS_MARGIN else "FAIL"))
        A("")
        A("Floor = error achievable on this data at this width, measured by re-quantizing the")
        A("bf16 authority with this repo's own quantizer at the published width")
        A("(`tools/gguf_reader.py` §\"Reference quantizers for the FLOOR PROBE\").")
        A("")
    if inj:
        A("## 1.5c injection matrix")
        A("")
        A("| injection | must-fire | fired | leaked into families it must not touch |"
          " min x floor when it fires |")
        A("|---|---|---|---|---|")
        for k in sorted(inj["per_injection"]):
            v = inj["per_injection"][k]
            A("| `%s` | %d | %d/%d | %d/%d | %s |"
              % (k, v["must"], v["fired"], v["must"], v["leaked"], v["must_not"],
                 "non-finite" if v["min_fire_x"] is None else "%.1f" % v["min_fire_x"]))
        A("")
        if inj.get("route_per_injection"):
            A("")
            A("The 1.4b allowance probed at its own boundary — the must-catch set grew with the "
              "relaxation, which is the only thing that distinguishes a one-step rule from \"whatever "
              "we later call one step\":")
            A("")
            A("| route injection | must-fire | fired | fired when it must not | what it does |")
            A("|---|---:|---|---|---|")
            for k in sorted(inj["route_per_injection"]):
                v = inj["route_per_injection"][k]
                A("| `%s` | %d | %d/%d | %d/%d | %s |"
                  % (k, v["must"], v["fired"], v["must"], v["leaked"], v["must_not"], v["how"]))
            A("")
        A("Cases: %d band + %d route; wrong: %d." % (inj.get("cases_band", 0),
                                                     inj.get("cases_route", 0),
                                                     len(inj.get("wrong", []))))
    A("## 1.6 shared storage, and the pin")
    A("")
    if sd:
        uw = (sd.get("untied_witness") or [["-", "-", "-", "-", "-", "-"]])[0]
        A("| check | result |")
        A("|---|---|")
        A("| alias detector self-witness (a known-aliased fixture must be found) | %s |"
          % ("OK" if sd.get("detector_witness") else "BROKEN"))
        A("| GGUF name pairs claiming identical bytes | %d |"
          % len(sd.get("gguf_alias_groups", [])))
        A("| GGUF tensors with overlapping byte ranges | %d |"
          % len(sd.get("gguf_overlaps", [])))
        A("| safetensors alias groups in the fetched shards | %d |"
          % sd.get("safetensors_alias_groups"))
        A("| untied `token_embd` vs `output` | %s |" % uw[4])
        A("| verdict | %s |" % sd.get("verdict"))
        A("")
        A("Nothing in this file shares storage with anything else, and the untied pair is witnessed")
        A("DISTINCT rather than assumed: `tie_word_embeddings=False` is a config claim, the byte")
        A("comparison is the evidence. The pin is `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`")
        A("(sha256 `%s`, %s bytes, revision `%s`, %s tensors, plus this run's section verdicts), so"
          % (h.get("sha256_pass1"), nines(h.get("bytes_on_disk")), h.get("revision"),
             h.get("n_tensors")))
        A("a silent re-upload under the same filename is a mismatch instead of a new truth. It also")
        A("carries the authority revision and the `mtp.*` element count, and it needs a row in")
        A("`checkpoints/README.md` when this is committed.")
        A("")
    A("## What this audit changes for the engine (findings)")
    A("")
    for line in findings(st, sh, ex, dq, inj, h, sd):
        A(line)
    A("")
    A("## What is NOT claimed")
    A("")
    for line in not_claimed():
        A(line)
    A("")
    with open(SUMMARY_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines; sections present: %s)"
          % (os.path.relpath(SUMMARY_MD, REPO), len(L), ",".join(sorted(parts))))
    return 0 if not missing else 1


# --------------------------------------------------------------------------------------------
def selfcheck_negative(args):
    """Prove the width machinery can fail. A derived table nobody has seen refuse anything is just a
    hand-typed table with extra steps.

    Four controls, each run against the real audited file: a stale hand-typed row must be reported as
    drift; a 2-byte-per-block geometry error must break the size identity; the width-map tool must
    refuse to print its table over a failing identity; and a policy row whose literal stops matching a
    derived struct width must abort. Run with `--selfcheck-widths`; it opens headers only and mutates
    nothing on disk.
    """
    print("width negative controls (each must ABORT or report, not print a plausible number)")
    import gguf_header_types as GHT
    import quant_scenarios as QS
    out = []

    def ctl(name, fired, detail):
        out.append({"control": name, "expected": "abort/report", "did": "yes" if fired else "NO",
                    "detail": detail})

    drift = GBG.assert_matches({"Q5_K": 5.5625, "Q4_K": 4.5, "Q8_0": 8.5, "Q6_K": 6.5625})
    ctl("stale hand row (Q5_K 5.5625) diffed against the header", [d["type"] for d in drift] == ["Q5_K"],
        "drift rows: %s; the three correct rows stayed silent" % [d["type"] for d in drift])

    ok_clean, d_clean = GBG.check_file(GGUF_PATH)
    orig = GBG.row_bytes

    def bad_row_bytes(n, tname):
        return orig(n, tname) + 2 * (n // 256) if tname == "Q5_K" else orig(n, tname)

    GBG.row_bytes = bad_row_bytes
    caught = ""
    try:
        ok_bad, d_bad = GBG.check_file(GGUF_PATH)
        caught = ("size identity rejected it (delta %+d B)" % d_bad["delta"]) if not ok_bad else ""
    except ValueError as e:
        # check_file cross-checks every derived row against gguf_reader's own table, so a geometry
        # edit can be caught there instead. Either way it does not survive to a printed number.
        caught = "cross-check against gguf_reader's row sizes rejected it: %s" % str(e)[:70]
    GBG.row_bytes = orig
    ok_after, d_after = GBG.check_file(GGUF_PATH)
    ctl("Q5_K block 2 B too big cannot survive to a printed number",
        bool(caught) and ok_clean and ok_after, "clean delta %+d B; %s" % (d_clean["delta"], caught))

    real_getsize = os.path.getsize
    real_argv = list(sys.argv)
    try:
        os.path.getsize = lambda p: real_getsize(p) + 4096
        # gguf_header_types.main() reads sys.argv itself; this drives it in-process so the control
        # needs no subprocess and cannot be mistaken for a successful external run.
        sys.argv = ["gguf_header_types.py", GGUF_PATH, "--bytes", str(64 << 20)]
        rc = GHT.main()
    finally:
        sys.argv = real_argv
        os.path.getsize = real_getsize
    ctl("width-map tool refuses to print over a failing identity", rc == 1,
        "gguf_header_types.py exit code %d with a lying file size (no width table printed)" % rc)

    key = "A experts Q4_K, rest Q8_0"
    saved = dict(QS.POLICIES[key])
    QS.POLICIES[key]["experts"] = 5.5625
    aborted = ""
    try:
        QS.assert_policy_widths_derived()
    except SystemExit as e:
        aborted = str(e).strip().splitlines()[-1]
    QS.POLICIES[key].clear()
    QS.POLICIES[key].update(saved)
    QS.assert_policy_widths_derived()
    ctl("policy row literal that is no derived width aborts", bool(aborted), aborted[:120])

    table(["control", "expected", "did", "detail"],
          [(c["control"], c["expected"], c["did"], c["detail"]) for c in out])
    bad = [c for c in out if c["did"] != "yes"]
    print("\n%d controls, %d failed to fire" % (len(out), len(bad)))
    ok = bool(out) and not bad
    print("SELFCHK-WIDTHS %s" % ("PASS" if ok else "FAIL"))
    write_json(os.path.join(EV, "audit-width-negative-controls-%s.json" % DAY),
               {"utc": now(), "controls": out, "verdict": "PASS" if ok else "FAIL"})
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--shards", action="store_true")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--floor-probe", action="store_true")
    ap.add_argument("--dequant", action="store_true")
    ap.add_argument("--injections", action="store_true")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--selfcheck-widths", action="store_true",
                    help="negative controls for the derived-width machinery (headers only, no writes)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-shard-sha", action="store_true",
                    help="skip re-hashing the fetched shards (they are still set-checked)")
    ap.add_argument("--sample-elts", type=int, default=131072,
                    help="elements sampled per tensor (whole blocks, spread over the dims)")
    ap.add_argument("--inject-sample-elts", type=int, default=16384)
    ap.add_argument("--all-rows", action="store_true", help="print every structure row")
    args = ap.parse_args()
    order = [("selftest", selftest), ("hash", sec_hash), ("shards", sec_shards),
             ("structure", sec_structure), ("exact", sec_exact), ("floor_probe", sec_floor_probe),
             ("dequant", sec_dequant), ("injections", sec_injections), ("shared", sec_shared),
             ("selfcheck_widths", selfcheck_negative), ("summary", sec_summary)]
    chosen = [name for name, _ in order
              if getattr(args, name, False) or getattr(args, name.replace("_", "-"), False)]
    if args.all:
        chosen = [name for name, _ in order]
    if not chosen:
        ap.print_help()
        return 2
    rc = 0
    verdicts = []
    for name, fn in order:
        if name not in chosen:
            continue
        args.section = name
        if name not in ("selftest", "hash", "shards", "structure"):
            if require_selftest(args) is None:
                verdicts.append((name, "REFUSED (no selftest marker)"))
                rc = rc or 3
                continue
        r = fn(args)
        verdicts.append((name, "rc=%d" % r))
        rc = rc or r
    print("\n--- section verdicts ---")
    for name, v in verdicts:
        print("  %-12s %s" % (name, v))
    return rc


if __name__ == "__main__":
    sys.exit(main())
