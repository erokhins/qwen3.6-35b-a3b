#!/usr/bin/env python3
"""weights_report.py — what a safetensors shard of THIS checkpoint actually holds, byte-witnessed.

Why this file exists and what it is not
---------------------------------------
`tools/bf16_census.py` owns the element census (it closes GGUF + draft block + vision to 0 bytes
against the shard index's `total_size`), and `tools/mtp_shard_cost.py` sizes the question "which
shards would I have to fetch" from the committed index alone, with shard bytes *estimated* as
`total_size / shard_count`. Neither one reads a shard header against itself. This tool does the
per-shard axis:

  * it parses each shard's own header and requires the file to be self-consistent —
    every tensor's offset span equals `numel * itemsize`, the spans tile `[0, max_end)` with no
    gap and no overlap, and `8 + header_bytes + max_end == os.path.getsize(path)`;
  * it reconciles the shard's name set with the committed index in BOTH directions (a tensor the
    index maps to another shard is a mixed-checkpoint directory, and a name the index says this
    shard holds but the header lacks is a truncated download that still parses);
  * it checks every shape it can see against the config-derived shape, and every family that the
    config says must exist, so a tool run against the wrong checkpoint aborts instead of printing
    a plausible table.

Shapes and family names are NOT retyped here. They come from `tools/bf16_census.py`
(`geometry()`, `hf_shape(name, g)`, `family_of(name)`), which is this repo's authority for the
arithmetic — a third copy of a formula is not a second ruler, and this repo has already paid for
one (`quantization/evidence/bf16-census-adjudication-2026-10-26.md`: a hand-written config model
was worth -0.4138 GiB). `--selftest` therefore builds a *synthetic* 4-layer / 3-expert checkpoint
in `tmp/` and prices it through the same code path: nothing in this file may hard-code 40 layers,
256 experts or 2048 hidden, or the synthetic arm fails.

What breaks if you port a 0.8B-shaped weights report onto this checkpoint (all four were checked
against the real headers, and all four are refused here):

  1. dense MLP assumptions — 96.4 % of the text weights are `mlp.experts.gate_up_proj`
     `[E, 2I, H] = [256, 1024, 2048]` and `mlp.experts.down_proj` `[E, H, I]`, one pair per layer,
     1.07 GiB + 0.54 GiB per layer. A report with no routed-expert family reports nothing.
  2. tied embeddings — this checkpoint is `tie_word_embeddings: false`, so `lm_head.weight`
     `[248320, 2048]` (0.9473 GiB) exists as a separate tensor and must be required, not assumed
     absent.
  3. `attn_output_gate: true` doubles `self_attn.q_proj` to `[nq*hd*2, H] = [8192, 2048]` while
     `k_proj`/`v_proj` stay `[nkv*hd, H] = [512, 2048]`; `head_dim` is 256, not `hidden/nq`.
  4. family spelling differs between the two artifacts of the SAME model. The HF shards say
     `linear_attn.in_proj_qkv` / `in_proj_z` / `conv1d` / `A_log` / `dt_bias` / `norm` /
     `out_proj`; the published GGUF says `attn_qkv` / `attn_gate` / `ssm_out`. A bucket written for
     one spelling silently collects nothing from the other while still printing a correct grand
     total, which is the failure this repo was bitten by once (`tools/gguf_suffix_census.py`).
     So this file refuses GGUF spellings outright (`GGUF_SPELLING`) rather than guessing.

Provenance: transcribed and adapted 2026-10-26 from the 0.8B engine repo's weights report (looked
at, copied in, never cited — rule: `pitfalls/60-docs-and-citations.md`). That instrument's summary
block prints "embed (tied, no lm_head)", groups layers by `linear_attn.A_log` presence and has no
routed-expert family; on this model every one of those lines is wrong, so they were rewritten
against the headers under `models/qwen36-src/` rather than kept. Row: `tools/PROVENANCE.md`.

Usage
-----
    python3 tools/weights_report.py                      # every shard under models/qwen36-src
    python3 tools/weights_report.py --shard model-00026-of-00026.safetensors --filter experts
    python3 tools/weights_report.py --require-complete   # all 26 shards must be on disk
    python3 tools/weights_report.py --selftest           # fault injection, no real file touched

Exit 0 only when every witness above holds. No timing is taken and no device is touched: this file
reads headers (a few hundred KB) and never maps a weight byte.
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bf16_census as CEN      # noqa: E402  — the shapes and family names live THERE, not here

SRC_DEFAULT = os.path.join(REPO, "models/qwen36-src")
IDX_DEFAULT = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json")
CFG_DEFAULT = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")

# safetensors dtype -> bytes per element. An unlisted dtype aborts; it is never priced as zero.
ELEMENT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                "U8": 1, "BOOL": 1}

# Names that belong to the GGUF spelling of this model. Seeing one inside a safetensors header is
# not a naming variant, it is the wrong artifact, so it aborts with a pointer to the right tool.
GGUF_SPELLING = ("blk.", "token_embd", "output.weight", "attn_qkv", "attn_gate", "ssm_out",
                 "ssm_a", "ssm_beta", "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_exps",
                 "ffn_up_exps", "ffn_down_exps", "attn_norm", "v_tiled", "w_v_tiled")

# The routed experts are stored FUSED in the HF checkpoint. The split pair is the GGUF's layout
# (and the layout llama.cpp hands a kernel), so a safetensors carrying it is a different checkpoint
# or a converter's half-finished output — either way, not the artifact this tool's arithmetic is for.
SPLIT_EXPERT_NAMES = ("mlp.experts.gate_proj", "mlp.experts.up_proj")


class Fatal(Exception):
    """A witness failed. Every refusal in this file raises this, never prints-and-continues."""


def die(fmt, *a):
    raise Fatal(fmt % a if a else fmt)


def read_header(path):
    """(header_len, header_dict, file_size) for one safetensors file, with the file-class guard.

    The GGUF check is not decoration: `models/` holds both artifact classes, a GGUF header also
    starts with a length-like field, and a tool that parsed a GGUF as safetensors would fail with
    a JSON error that blames the file rather than the caller.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic == b"GGUF":
            die("%s is a GGUF, not a safetensors shard. The published file spells its families "
                "differently (attn_qkv / attn_gate / ssm_out, split ffn_gate_exps/ffn_up_exps); "
                "use tools/gguf_suffix_census.py or tools/bf16_census.py for it.", path)
        if len(magic) != 4:
            die("%s is %d bytes long: too short to carry a safetensors header", path, size)
        fh.seek(0)
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            die("%s: no 8-byte header length", path)
        (hlen,) = struct.unpack("<Q", raw_len)
        if hlen == 0 or 8 + hlen > size:
            die("%s: header length %s is not consistent with a %d-byte file (truncated download?)",
                path, hlen, size)
        try:
            hdr = json.loads(fh.read(hlen).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            die("%s: header is not valid JSON (%s) — this is not a safetensors file", path, e)
    if not isinstance(hdr, dict):
        die("%s: header parsed to %s, expected an object", path, type(hdr).__name__)
    return hlen, hdr, size


def tensor_bytes(name, meta, path):
    """Validate one header record and return its byte count.

    Two independent claims are checked here, and both have to hold: the span the header reserves,
    and `numel * itemsize` implied by the declared shape and dtype. A truncated shard, an
    endianness mistake in a converter, and a dtype relabel all show up as a mismatch between them.
    """
    if meta.get("dtype") not in ELEMENT_SIZE:
        die("%s: tensor %s has dtype %r, which this tool has no itemsize for. Refusing to price it "
            "as 0 bytes — add it to ELEMENT_SIZE only with a source.",
            path, name, meta.get("dtype"))
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape or any(not isinstance(d, int) for d in shape):
        die("%s: tensor %s has shape %r; every tensor in this checkpoint is at least 1-D and a "
            "0-D or non-integer shape means the header was not written by the converter we expect",
            path, name, shape)
    off = meta.get("data_offsets")
    if (not isinstance(off, list) or len(off) != 2
            or not all(isinstance(x, int) for x in off) or off[1] < off[0]):
        die("%s: tensor %s has data_offsets %r", path, name, off)
    numel = 1
    for d in shape:
        numel *= d
    want = numel * ELEMENT_SIZE[meta["dtype"]]
    span = off[1] - off[0]
    if span != want:
        die("%s: tensor %s declares shape %s dtype %s (= %d elements, %d bytes) but reserves %d "
            "bytes. One of the shape, the dtype and the offsets is a lie, so no number printed from "
            "this header can be trusted.",
            path, name, shape, meta["dtype"], numel, want, span)
    return span


def check_shard(path, index_map, shard_name, allow_other_dtype=()):
    """Full witness over one shard header. Returns the parsed records.

    `index_map` is the committed index's `weight_map`. Any disagreement between the file and the
    index is fatal, in either direction, because the failure mode being guarded is "a directory
    holding shards from two different checkpoints" — which parses fine and prints fine.
    """
    hlen, hdr, size = read_header(path)
    recs = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not recs:
        die("%s: header parsed to zero tensor records. A check that has nothing to check cannot "
            "fail, so this is an error, not an empty pass.", path)
    spans = []
    dtypes = {}
    for name in sorted(recs):
        meta = recs[name]
        spans.append((meta["data_offsets"][0], meta["data_offsets"][1], name))
        nbytes = tensor_bytes(name, meta, path)
        dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + nbytes
        mapped = index_map.get(name)
        if mapped is None:
            die("%s: tensor %s is not in the committed index. Either the shard came from another "
                "checkpoint or the index is stale; both mean the geometry printed here is a "
                "guess.", path, name)
        if mapped != shard_name:
            die("%s: index maps %s to %s. Two shards in one directory claiming the same tensor is "
                "how a mixed-checkpoint directory reads as a complete model.",
                path, name, mapped)
        if meta["dtype"] not in ("BF16",) and meta["dtype"] not in allow_other_dtype:
            die("%s: tensor %s is %s, not BF16. The bf16 checkpoint is the reference of record "
                "(`pitfalls/30-numerics.md`); a silently wider tensor is a 2x residency surprise. "
                "Pass --allow-other-dtype %s if this shard is deliberately not the bf16 artifact.",
                path, name, meta["dtype"], meta["dtype"])
    spans.sort()
    if spans[0][0] != 0:
        die("%s: first tensor starts at offset %d, not 0", path, spans[0][0])
    total_data = 0
    for i, (lo, hi, name) in enumerate(spans):
        if i:
            if lo < prev_hi:
                die("%s: %s overlaps the preceding tensor's span (%d < %d)", path, name, lo, prev_hi)
            if lo > prev_hi:
                die("%s: %d bytes of gap before %s — offsets must tile the data section",
                    path, lo - prev_hi, name)
        prev_hi = hi
        total_data += hi - lo
    max_end = spans[-1][1]
    if total_data != max_end:
        die("%s: spans sum to %d but the data section ends at %d", path, total_data, max_end)
    if 8 + hlen + max_end != size:
        die("%s: file is %d bytes but 8 + header(%d) + data(%d) = %d. Truncated, padded, or still "
            "downloading; every byte total from this shard would be wrong.",
            path, size, hlen, max_end, 8 + hlen + max_end)
    declared = {k for k, v in index_map.items() if v == shard_name}
    missing = sorted(declared - set(recs))
    if missing:
        die("%s: the index says this shard holds %d tensors, %d are absent from its header (e.g. "
            "%s). A partial download parses happily and prints a confident, short table.",
            path, len(declared), len(missing), ", ".join(missing[:3]))
    extra = sorted(set(recs) - declared)
    if extra:
        die("%s: %d tensors in the header that the index does not attribute to this shard (e.g. %s)",
            path, len(extra), ", ".join(extra[:3]))
    return dict(path=path, shard=shard_name, header_bytes=hlen, file_bytes=size,
                data_bytes=max_end, records=recs, dtypes=dtypes)


def required_names(g, cfg):
    """The name population the config demands, derived — never a typed list.

    This is the wrong-model check. `hf_shape()` returns None for anything the census cannot model,
    and that is fatal here too: an unmodellable required name is a checkpoint this tool has no
    business summarising.
    """
    lm = "model.language_model."
    want = [lm + "embed_tokens.weight", lm + "norm.weight"]
    if not cfg.get("tie_word_embeddings", cfg["text_config"].get("tie_word_embeddings", True)):
        want.append("lm_head.weight")
    for i in range(g["L"]):
        p = f"{lm}layers.{i}."
        want += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        if g["layer_types"][i] == "linear_attention":
            want += [p + "linear_attn." + n for n in
                     ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                      "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight",
                      "out_proj.weight")]
        elif g["layer_types"][i] == "full_attention":
            want += [p + "self_attn." + n for n in
                     ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                      "q_norm.weight", "k_norm.weight")]
        else:
            die("config layer_types[%d] = %r: expected linear_attention or full_attention",
                i, g["layer_types"][i])
        want += [p + "mlp.gate.weight", p + "mlp.experts.gate_up_proj", p + "mlp.experts.down_proj",
                 p + "mlp.shared_expert.gate_proj.weight", p + "mlp.shared_expert.up_proj.weight",
                 p + "mlp.shared_expert.down_proj.weight", p + "mlp.shared_expert_gate.weight"]
    if g["draft"]:
        want += ["mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                 "mtp.pre_fc_norm_hidden.weight"]
        for j in range(g["draft"]):
            p = f"mtp.layers.{j}."
            want += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
            want += [p + "self_attn." + n for n in
                     ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                      "q_norm.weight", "k_norm.weight")]
            want += [p + "mlp.gate.weight", p + "mlp.experts.gate_up_proj",
                     p + "mlp.experts.down_proj", p + "mlp.shared_expert.gate_proj.weight",
                     p + "mlp.shared_expert.up_proj.weight", p + "mlp.shared_expert.down_proj.weight",
                     p + "mlp.shared_expert_gate.weight"]
    return want


def check_config(cfg, g):
    """Internal consistency of the config the geometry was read out of.

    `full_attention_interval` and `layer_types` are two independent statements of the same fact in
    this config; if they ever disagree, every per-family count below is a coin flip.
    """
    L, interval = g["L"], cfg["text_config"].get("full_attention_interval")
    if len(g["layer_types"]) != L:
        die("config: layer_types has %d entries but num_hidden_layers is %d",
            len(g["layer_types"]), L)
    nfull = sum(1 for t in g["layer_types"] if t == "full_attention")
    if interval:
        if nfull * interval != L:
            die("config: %d full_attention layers x full_attention_interval %s != %d layers",
                nfull, interval, L)
        for i, t in enumerate(g["layer_types"]):
            expect = "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            if t != expect:
                die("config: layer_types[%d] = %r but full_attention_interval %s says %r",
                    i, t, interval, expect)
    if g["n_active"] > g["E"]:
        die("config: num_experts_per_tok %d exceeds num_experts %d",
            g["n_active"], g["E"])


def geometry_from(cfg_path):
    """The config-derived geometry dict, built by `tools/bf16_census.py` — never retyped here.

    `bf16_census.geometry()` reads its own module-level `CFG`, so pointing that name at the caller's
    config for the duration of one call is how `--config` stays honest. The alternative — a local
    copy of the config-field mapping — is the third copy of a formula this repo already forbids, and
    it is how a tool ends up accepting `--config` while pricing from a different file. The control
    set in `tmp/w17_inventory/wr_controls.py` is what proves this matters: with the geometry read from
    the census's own constant instead, four wrong-config mutations ran clean.
    """
    saved = CEN.CFG
    CEN.CFG = cfg_path
    try:
        return dict(CEN.geometry())
    finally:
        CEN.CFG = saved


def report(src, index_path, cfg_path, shards=None, filt=None, require_complete=False,
           allow_other_dtype=(), out=print, geometry=None, show_tensors=False):
    """Run every witness and print the table. Returns a dict of the numbers for the selftest."""
    cfg = json.load(open(cfg_path))
    g = dict(geometry) if geometry else geometry_from(cfg_path)
    check_config(cfg, g)
    idx = json.load(open(index_path))
    wm = idx["weight_map"]
    total_size = int(float(idx["metadata"]["total_size"]))

    entries = sorted(os.listdir(src))
    files = [f for f in entries if f.endswith(".safetensors")]
    others = [f for f in entries if not f.endswith(".safetensors")]
    if others:
        out("note: ignored non-safetensors entries under %s: %s" % (src, ", ".join(others)))
    if shards:
        unknown = [s for s in shards if s not in files]
        if unknown:
            die("--shard %s not found under %s (present: %s)", ", ".join(unknown), src,
                ", ".join(files) or "nothing")
        files = [f for f in files if f in shards]
    if not files:
        die("no .safetensors shards under %s — refusing to print an empty census as a pass", src)

    index_shards = sorted(set(wm.values()))
    strays = sorted(set(files) - set(index_shards))
    if strays:
        die("shard(s) %s under %s are not named by the committed index (%d shards). A directory "
            "with a foreign shard in it is not this checkpoint.", ", ".join(strays), src,
            len(index_shards))
    if require_complete and len(files) != len(index_shards):
        die("--require-complete: index names %d shards, %d present (missing %s)",
            len(index_shards), len(files),
            ", ".join(sorted(set(index_shards) - set(files))[:5]))

    out("== %s" % src)
    out("config %s: L=%d E=%d top_k=%d I=%d H=%d V=%d head_dim=%d nq=%d nkv=%d "
        "linear(lk,lv,dk,dv,kf)=(%d,%d,%d,%d,%d) draft=%d"
        % (os.path.relpath(cfg_path, REPO), g["L"], g["E"], g["n_active"], g["me"], g["h"], g["V"],
           g["hd"], g["nq"], g["nkv"], g["lk"], g["lv"], g["dk"], g["dv"], g["kf"], g["draft"]))
    out("index  %s: %d names, %d shards, total_size %d B"
        % (os.path.relpath(index_path, REPO), len(wm), len(index_shards), total_size))

    # --- identity first, bytes second. These are the wrong-model checks: they read the index and the
    # config only, and they are the ones that answer "is this even the right checkpoint". Running them
    # before the per-shard byte witnesses matters for diagnosis — a wrong-model directory usually
    # violates several invariants at once, and the informative refusal should be the identity one, not
    # "the file is the wrong length".
    want = required_names(g, cfg)
    absent = [n for n in want if n not in wm]
    if absent:
        die("the config requires %d tensor name(s) the index does not carry (e.g. %s). This is not "
            "the checkpoint the config describes.", len(absent), ", ".join(absent[:4]))
    for n in want:
        if not n.startswith("model.visual.") and CEN.hf_shape(n, g) is None:
            die("required name %s has no config-derived shape in tools/bf16_census.py::hf_shape", n)
    gguf_named = [k for k in wm if any(s in k for s in GGUF_SPELLING)]
    if gguf_named:
        die("the index carries GGUF-spelled name(s) %s — that is the published artifact's naming, "
            "not the safetensors checkpoint's (see tools/gguf_suffix_census.py)",
            ", ".join(gguf_named[:4]))
    split_experts = [k for k in wm for s in SPLIT_EXPERT_NAMES if s in k]
    if split_experts:
        die("the index carries split routed-expert weights %s. THIS checkpoint stores the fused "
            "`mlp.experts.gate_up_proj` [E, 2I, H] and `mlp.experts.down_proj` [E, H, I]; the split "
            "pair is the GGUF/kernel layout. Reading one as the other is the gate/up swap this "
            "repo's README prediction 3 warns about.", ", ".join(split_experts[:4]))
    tied = cfg.get("tie_word_embeddings", cfg["text_config"].get("tie_word_embeddings", True))
    if tied and "lm_head.weight" in wm:
        die("config says tie_word_embeddings=true but the index carries lm_head.weight")
    if not tied and "lm_head.weight" not in wm:
        die("config says tie_word_embeddings=false but the index has no lm_head.weight — the "
            "0.9473 GiB untied head is where a fifth of the per-token read goes "
            "(README.md prediction 5), so its absence is a different checkpoint, not a saving")
    if g["draft"] and not any(k.startswith("mtp.") for k in wm):
        die("config says mtp_num_hidden_layers=%d but the index carries no mtp.* name (D-05 puts the "
            "draft block in scope)" % g["draft"])
    ndraft_names = sum(1 for k in wm if k.startswith("mtp."))
    if not g["draft"] and ndraft_names:
        die("the index carries %d mtp.* tensor(s) but the config says mtp_num_hidden_layers=0. The "
            "draft block is either real or it is not; a census that quietly drops 1.5733 GiB of "
            "names because a config field was edited reads low and still prints a total",
            ndraft_names)
    nvision = sum(1 for k in wm if k.startswith("model.visual."))
    if ("vision_config" in cfg) != bool(nvision):
        die("config has vision_config=%s but the index carries %d model.visual.* tensors",
            "vision_config" in cfg, nvision)
    out("\n== identity (config x index, checked before any shard byte)")
    out("   required names present in the index            OK (%d checked)" % len(want))
    out("   GGUF-spelled names refused                     OK (0 of %d index names)" % len(wm))
    out("   split routed experts refused                   OK (fused gate_up_proj is the layout)")
    out("   untied lm_head / draft block / vision presence OK")

    fam_bytes, fam_count, seen_layers, parsed_records = {}, {}, set(), 0
    bytes_present = 0
    shape_checks = 0
    for f in files:
        sh = check_shard(os.path.join(src, f), wm, f, allow_other_dtype)
        parsed_records += len(sh["records"])
        bytes_present += sh["data_bytes"]
        out("\n-- %s  %d tensors  file %s B (header %s + data %s)"
            % (f, len(sh["records"]), format(sh["file_bytes"], ","), format(sh["header_bytes"], ","),
               format(sh["data_bytes"], ",")))
        out("   8 + header + max_end == file size            OK")
        out("   spans tile [0, %s) with no gap or overlap    OK" % format(sh["data_bytes"], ","))
        out("   every span == numel x itemsize               OK (%d records checked)"
            % len(sh["records"]))
        out("   dtype bytes: %s" % ", ".join("%s %s B" % (k, format(v, ","))
                                             for k, v in sorted(sh["dtypes"].items())))
        layers = sorted({int(m.group(1)) for k in sh["records"]
                         if k.startswith("model.language_model.layers.")
                         and (m := re.search(r"layers\.(\d+)\.", k))})
        mtp_layers = sorted({int(m.group(1)) for k in sh["records"]
                             if k.startswith("mtp.layers.") and (m := re.search(r"layers\.(\d+)\.", k))})
        seen_layers.update(layers)
        out("   text layers touched: %s   draft-block layers: %s"
            % (",".join(map(str, layers)) or "none", ",".join(map(str, mtp_layers)) or "none"))
        rows = {}
        for name in sorted(sh["records"]):
            if filt and filt not in name:
                continue
            meta = sh["records"][name]
            numel = 1
            for d in meta["shape"]:
                numel *= d
            fam = CEN.family_of(name)
            key = (fam, tuple(meta["shape"]), meta["dtype"])
            r = rows.setdefault(key, [0, 0, 0, name])
            r[0] += 1
            r[1] += numel
            r[2] += numel * ELEMENT_SIZE[meta["dtype"]]
            fb = fam_bytes.setdefault(fam, [0, 0])
            fb[0] += 1
            fb[1] += numel * ELEMENT_SIZE[meta["dtype"]]
            if name.startswith("model.visual."):
                continue        # out of scope (DECISIONS.md D-04); counted, never shaped
            want_shape = CEN.hf_shape(name, g)
            if want_shape is None:
                die("%s: %s is a text-path name that tools/bf16_census.py cannot model. A census "
                    "that drops what it cannot express reads low and still prints a total.", f, name)
            if tuple(meta["shape"]) != tuple(want_shape):
                die("%s: %s has shape %s in the shard header; the config says %s "
                    "(via tools/bf16_census.py::hf_shape). Wrong checkpoint, wrong config, or a "
                    "converter that moved a tensor — not a rounding difference.",
                    f, name, tuple(meta["shape"]), tuple(want_shape))
            shape_checks += 1
            if show_tensors:
                out("     %-62s %-5s %-22s %s B"
                    % (name, meta["dtype"], "x".join(map(str, meta["shape"])),
                       format(numel * ELEMENT_SIZE[meta["dtype"]], ",")))
        out("   %-16s %8s %16s %16s" % ("family", "tensors", "elements", "bytes"))
        for (fam, shape, dt), (n, el, by, _) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
            out("   %-16s %8d %16s %16s   %s%s %s"
                % (fam, n, format(el, ","), format(by, ","), dt, list(shape), ""))
        if filt:
            out("   (filtered on %r; family totals above are the filtered subset)" % filt)

    out("\n== shard witnesses")
    out("   shard records parsed == records attributed     OK (%d == %d)"
        % (parsed_records, sum(1 for k, v in wm.items() if v in set(files))))
    out("   shapes witnessed against the config            OK (%d non-vision tensors; %d vision "
        "tensors are counted, never shaped — DECISIONS.md D-04)"
        % (shape_checks, parsed_records - shape_checks))
    out("   layers touched by present shards               %s of %d"
        % (len(seen_layers), g["L"]))
    out("   bytes present / index total_size               %s / %s  (%.2f %%)  [%d of %d shards]"
        % (format(bytes_present, ","), format(total_size, ","),
           100.0 * bytes_present / total_size, len(files), len(index_shards)))
    if len(files) == len(index_shards) and bytes_present != total_size:
        die("all %d shards are present but their data sections total %d B against an index "
            "total_size of %d B (delta %+d). The census cannot close, so nothing downstream may "
            "quote it.", len(files), bytes_present, total_size, bytes_present - total_size)
    if len(files) != len(index_shards):
        out("   MISSING shards (named, not estimated): %s"
            % ", ".join(sorted(set(index_shards) - set(files))))
        out("   (a partial directory is legal for a sampled audit — tools/mtp_shard_cost.py sizes "
            "the fetch; but no total quoted from this run covers the whole model)")
    out("\n== family totals over the PRESENT shards")
    text_bytes = sum(b for f, (_, b) in fam_bytes.items() if f != "vision")
    out("   %-16s %8s %18s" % ("family", "tensors", "bytes"))
    for fam in sorted(fam_bytes, key=lambda k: -fam_bytes[k][1]):
        c, by = fam_bytes[fam]
        out("   %-16s %8d %18s%s" % (fam, c, format(by, ","),
                                     "   (out of scope, counted only)" if fam == "vision" else ""))
    out("   %-16s %8d %18s" % ("TOTAL", sum(c for c, _ in fam_bytes.values()),
                               format(sum(b for _, b in fam_bytes.values()), ",")))
    out("   text (non-vision) bytes present %s B = %s GiB, over %d of %d shards"
        % (format(text_bytes, ","), format(text_bytes / 2.0 ** 30, ".4f"), len(files),
           len(index_shards)))
    out("   (the authoritative whole-model text-bf16 figure is tools/bf16_census.py: 66.1337 GiB "
        "with the draft block; this row is only what is on disk here)")
    return dict(shards=len(files), records=parsed_records, bytes_present=bytes_present,
                required=len(want), shape_checks=shape_checks, families=fam_bytes,
                text_bytes=text_bytes)


# ---------------------------------------------------------------------------------------------
# selftest — faults injected into a SYNTHETIC checkpoint in tmp/, so the real artifact is never
# mutated and no 40/256/2048 literal can hide in this file.
# ---------------------------------------------------------------------------------------------
def synth(root):
    """Write a small checkpoint with the same family structure and a matching index + config."""
    os.makedirs(root, exist_ok=True)
    g = dict(h=32, L=4, V=64, hd=8, nq=2, nkv=1, E=3, n_active=2, me=16, se=16, lk=1, lv=2,
             dk=8, dv=8, kf=4, layer_types=["linear_attention", "linear_attention",
                                            "linear_attention", "full_attention"], draft=1)
    cfg = {"tie_word_embeddings": False,
           "text_config": {"hidden_size": g["h"], "num_hidden_layers": g["L"],
                           "vocab_size": g["V"], "head_dim": g["hd"],
                           "num_attention_heads": g["nq"], "num_key_value_heads": g["nkv"],
                           "num_experts": g["E"], "num_experts_per_tok": g["n_active"],
                           "moe_intermediate_size": g["me"],
                           "shared_expert_intermediate_size": g["se"],
                           "linear_num_key_heads": g["lk"], "linear_num_value_heads": g["lv"],
                           "linear_key_head_dim": g["dk"], "linear_value_head_dim": g["dv"],
                           "linear_conv_kernel_dim": g["kf"], "layer_types": g["layer_types"],
                           "full_attention_interval": 4, "mtp_num_hidden_layers": g["draft"]},
           "vision_config": {"hidden_size": 8}}
    names = required_names(g, cfg) + ["model.visual.patch_embed.proj.weight"]
    shapes = {n: (CEN.hf_shape(n, g) or (8,)) for n in names}
    weight_map, headers = {}, {}
    order = sorted(names)
    buckets = [order[i::2] for i in range(2)]           # two shards, interleaved names
    for si, bucket in enumerate(buckets, start=1):
        shard = "model-%05d-of-00002.safetensors" % si
        off, recs = 0, {}
        for n in bucket:
            sh = list(shapes[n])
            numel = 1
            for d in sh:
                numel *= d
            recs[n] = {"dtype": "BF16", "shape": sh, "data_offsets": [off, off + numel * 2]}
            off += numel * 2
            weight_map[n] = shard
        headers[shard] = (recs, off)
    for shard, (recs, data) in headers.items():
        hdr = dict(recs)
        hdr["__metadata__"] = {"format": "pt"}
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        # Real safetensors writers align the data section by padding the JSON *inside* the declared
        # header length (trailing spaces), which is what makes `8 + header + data == file size` an
        # exact identity on the four shards under models/qwen36-src/. Padding AFTER the length field
        # would break that identity, so this mirrors the real convention rather than inventing one.
        pad = (-len(blob)) % 8
        blob += b" " * pad
        with open(os.path.join(root, shard), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            fh.write(b"\xab" * data)
    json.dump({"metadata": {"total_size": float(sum(v[1] for v in headers.values()))},
               "weight_map": weight_map},
              open(os.path.join(root, "model.safetensors.index.json"), "w"))
    json.dump(cfg, open(os.path.join(root, "config.json"), "w"))
    return g, os.path.join(root, "model.safetensors.index.json"), os.path.join(root, "config.json")


def selftest(root):
    import shutil
    import io
    if os.path.isdir(root):
        shutil.rmtree(root)
    g, idx_path, cfg_path = synth(root)
    src = root

    def run(**kw):
        buf = io.StringIO()
        kw.setdefault("geometry", g)
        return report(src, idx_path, cfg_path, out=buf.write, **kw), buf.getvalue()

    fired, missed = [], []

    def expect_fire(label, fn):
        def why(e):
            return " ".join(str(e).split())[-118:]
        try:
            fn()
        except Fatal as e:
            fired.append((label, why(e)))
            print("  FIRED   %-38s ...%s" % (label, why(e)))
            return
        except SystemExit as e:
            fired.append((label, "SystemExit " + why(e)))
            print("  FIRED   %-38s ...SystemExit %s" % (label, why(e)))
            return
        missed.append(label)
        print("  NOT FIRED %-36s the fault ran clean — the guard is decorative" % label)

    def patch_json(path, mutate):
        obj = json.load(open(path))
        mutate(obj)
        json.dump(obj, open(path, "w"))

    def patch_header(shard, mutate):
        p = os.path.join(src, shard)
        hlen, hdr, size = read_header(p)
        mutate(hdr)
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)          # same alignment convention as synth()/the real shards
        with open(p, "rb") as fh:
            fh.seek(8 + hlen)
            data = fh.read(size - 8 - hlen)
        with open(p, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            fh.write(data)

    print("weights_report --selftest: fault injection over a synthetic 4-layer/3-expert checkpoint")
    res, _ = run()
    if res["required"] < 40 or res["records"] < 40 or len(res["families"]) < 6:
        print("SELFTEST ABORT: the synthetic checkpoint priced %d required names / %d records / %d "
              "families, too few to exercise the guards — the population being checked is vacuous"
              % (res["required"], res["records"], len(res["families"])))
        return 1
    print("  CONTROL   clean synthetic run priced %d records, %d B"
          % (res["records"], res["bytes_present"]))
    if res["bytes_present"] <= 0:
        die("selftest: clean run reported 0 bytes, so the byte accounting is vacuous")

    good = json.load(open(idx_path))
    def gguf_as_shard():
        p = os.path.join(src, "model-00001-of-00002.bak")
        with open(p, "wb") as fh:
            fh.write(b"GGUF" + b"\0" * 64)
        check_shard(p, json.load(open(idx_path))["weight_map"], "model-00001-of-00002.bak")
    expect_fire("GGUF fed as a shard", gguf_as_shard)
    expect_fire("shard claims an index name it is not mapped to",
                lambda: check_shard(os.path.join(src, "model-00001-of-00002.safetensors"),
                                    good["weight_map"], "model-00002-of-00002.safetensors"))
    expect_fire("tensor absent from the index",
                lambda: check_shard(os.path.join(src, "model-00001-of-00002.safetensors"), {},
                                    "model-00001-of-00002.safetensors"))

    stash = json.load(open(idx_path))
    def drop_lmhead(obj):
        obj["weight_map"].pop("lm_head.weight")
    expect_fire("lm_head.weight dropped (untied model)",
                lambda: (patch_json(idx_path, drop_lmhead), run())[1])
    json.dump(stash, open(idx_path, "w"))

    def rename_gdn(obj):
        wm = obj["weight_map"]
        k = next(k for k in wm if k.endswith("linear_attn.in_proj_qkv.weight"))
        wm[k.replace("linear_attn.in_proj_qkv.weight", "attn_qkv")] = wm.pop(k)
    expect_fire("GGUF spelling of GDN in the index",
                lambda: (patch_json(idx_path, rename_gdn), run())[1])
    json.dump(stash, open(idx_path, "w"))

    def split_experts(obj):
        wm = obj["weight_map"]
        k = next(k for k in wm if k.endswith("mlp.experts.gate_up_proj"))
        wm[k.replace("gate_up_proj", "gate_proj")] = wm.pop(k)
    expect_fire("split routed experts in the index",
                lambda: (patch_json(idx_path, split_experts), run())[1])
    json.dump(stash, open(idx_path, "w"))

    def forget_a_name(obj):
        wm = obj["weight_map"]
        k = next(k for k in wm if k.endswith("mtp.pre_fc_norm_hidden.weight"))
        wm[k + ".gone"] = wm.pop(k)
    expect_fire("required draft-block name missing",
                lambda: (patch_json(idx_path, forget_a_name), run())[1])
    json.dump(stash, open(idx_path, "w"))

    shard1 = "model-00001-of-00002.safetensors"
    p1 = os.path.join(src, shard1)
    size0 = os.path.getsize(p1)
    def truncate():
        with open(p1, "r+b") as fh:
            fh.truncate(size0 - 64)
    expect_fire("truncated shard file", lambda: (truncate(), run())[1])
    with open(p1, "r+b") as fh:
        fh.seek(size0 - 64)
        fh.write(b"\xab" * 64)
    ctl, _ = run()
    print("  CONTROL   restored after truncation, %d B again" % ctl["bytes_present"])

    def grow_shape(hdr):
        k = next(k for k in hdr if k.endswith("mlp.experts.gate_up_proj"))
        hdr[k]["shape"] = [hdr[k]["shape"][0], hdr[k]["shape"][1] * 2, hdr[k]["shape"][2]]
    expect_fire("shape lies about its span", lambda: (patch_header(shard1, grow_shape), run())[1])
    # restoring is cheapest by re-synth, but the header fault also breaks the file identity, so
    # rebuild the whole synthetic tree and re-stash the index.
    shutil.rmtree(root)
    g2, idx_path, cfg_path = synth(root)
    assert g2 == g

    def wrong_dtype(hdr):
        k = next(k for k in hdr if k.endswith("mlp.gate.weight"))
        hdr[k]["dtype"] = "F32"
    expect_fire("dtype relabelled BF16 -> F32", lambda: (patch_header(shard1, wrong_dtype), run())[1])
    shutil.rmtree(root)
    g2, idx_path, cfg_path = synth(root)

    def bogus_dtype(hdr):
        k = next(k for k in hdr if k.endswith("mlp.gate.weight"))
        hdr[k]["dtype"] = "F128"
    expect_fire("unknown dtype", lambda: (patch_header(shard1, bogus_dtype), run())[1])
    shutil.rmtree(root)
    g2, idx_path, cfg_path = synth(root)

    def overlap(hdr):
        ks = sorted(k for k in hdr if k != "__metadata__")
        a, b = ks[0], ks[1]
        lo_a, hi_a = hdr[a]["data_offsets"]
        lo_b, hi_b = hdr[b]["data_offsets"]
        if lo_a == lo_b:
            die("selftest fixture: the first two tensors already share a start offset, so the "
                "overlap fault cannot be injected without also changing a span")
        # b keeps claiming its own correct LENGTH but starts where a starts: the spans stay
        # numel-consistent, so the only witness left able to fire is the tiling check.
        hdr[b]["data_offsets"] = [lo_a, lo_a + (hi_b - lo_b)]
    expect_fire("overlapping spans", lambda: (patch_header(shard1, overlap), run())[1])
    shutil.rmtree(root)
    g2, idx_path, cfg_path = synth(root)

    def empty_header(hdr):
        for k in list(hdr):
            if k != "__metadata__":
                del hdr[k]
    expect_fire("header with zero records", lambda: (patch_header(shard1, empty_header), run())[1])
    shutil.rmtree(root)
    g2, idx_path, cfg_path = synth(root)

    expect_fire("geometry drift: num_experts 3 -> 4",
                lambda: run(geometry=dict(g, E=4))[1])
    expect_fire("geometry drift: head_dim 8 -> 16",
                lambda: run(geometry=dict(g, hd=16))[1])
    expect_fire("layer_types contradict full_attention_interval",
                lambda: run(geometry=dict(g, layer_types=["linear_attention"] * 4))[1])

    cfg = json.load(open(cfg_path))
    cfg["text_config"]["full_attention_interval"] = 2
    json.dump(cfg, open(cfg_path, "w"))
    expect_fire("config contradicts itself (interval)", lambda: run()[1])

    if missed:
        print("\nSELFTEST FAIL: %d of %d injected faults ran clean: %s"
              % (len(missed), len(missed) + len(fired), ", ".join(missed)))
        return 1
    print("\nSELFTEST PASS: %d injected faults each fired, and the clean synthetic run priced "
          "%d records / %d bytes" % (len(fired), ctl["records"], ctl["bytes_present"]))
    return 0


def main(argv):
    a = argv[:]
    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    def flag(name):
        return name in a
    src = opt("--src", SRC_DEFAULT)
    index_path = opt("--index", IDX_DEFAULT)
    cfg_path = opt("--config", CFG_DEFAULT)
    shards = opt("--shard")
    shards = shards.split(",") if shards else None
    filt = opt("--filter")
    if flag("--selftest"):
        return selftest(os.path.join(REPO, "tmp", "w17_weights_report_selftest"))
    try:
        report(src, index_path, cfg_path, shards=shards, filt=filt,
               require_complete=flag("--require-complete"),
               allow_other_dtype=tuple(opt("--allow-other-dtype", "").split(",")) if
               flag("--allow-other-dtype") else (),
               show_tensors=flag("--show-tensors"))
    except Fatal as e:
        print("REFUSED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
