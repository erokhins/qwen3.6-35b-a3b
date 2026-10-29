#!/usr/bin/env python3
"""tools/hf_full_forward_d29.py — HF logits and per-block activations for all 40 blocks, on the
aligned 32-id stimulus. CPU only.

D-29 makes installed `transformers` (`qwen3_5_moe`, the published bf16 checkpoint) the
AUTHORITATIVE reference where it can run, and it can now run everywhere: all 26 shards are on disk
and `tools/checkpoint_integrity_d29.py` tied their bytes to the Hub's LFS etags. This file produces
the CANDIDATE fixture; registering it is the user's act (D-18), so every artifact it writes says
`status=candidate`, `registration=not-registered`.

WHAT IS ON THE STIMULUS. The chain manifest's own 32 ids
(`checkpoints/dump-manifest-chain-2026-10-26.json input.token_ids`), the same 32 the llama.cpp
candidate consumed. `token_ids_sha256` is recomputed here with the encoding the chain itself uses
(sha256 over each id as a little-endian int64 — verified against the chain's own field before it is
used), compared elementwise against BOTH the chain manifest and the llama candidate, and a
mismatch is a refusal, not a warning.

STAGES are the seven §03.2 rows `tools/dump_chain.py` pins (`x, h_in, mixer_out, x1, h_mlp, moe_out,
out`), captured by forward hooks on the reference's own modules — the dumper never computes a stage
value, it only observes one — plus `normed` (final RMSNorm output, all 32 positions) and
`logits-last-token` (the row greedy sampling reads). Hook sources are the same module list
`tools/dump_chain.py::STAGE_SOURCE` names; they are re-declared here rather than imported so that
this file can be deleted without taking the pinned fixture's producer with it.

MEMORY. The text tower is ~63 GiB bf16 (derived from the safetensors headers, not typed) against
`MemAvailable`. `preflight` proves the name/shape/dtype mapping on the meta device and refuses
before a byte is read if the host cannot hold it; `run` re-checks `MemAvailable` at load time and
refuses rather than thrashing. The layer-by-layer alternative (`tools/dump_chain.py`'s one-block-
resident regime) is the fallback and is named in the refusal.

CPU-only guards: `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`, and an asserted
`torch.cuda.is_available() is False` printed before loading. No device clock is read; the clocks
here are host wall-clock and host RSS. **no timing taken (device).**

MODES
  preflight                     mapping + geometry + RAM, no weights read
  run --tag TAG                 load, forward, write records + run manifest
  pair TAG_A TAG_B              determinism pair (separate processes), + the perturbation control
  perturb --tag TAG             same as run with one expert weight zeroed downstream; the control
  candidate TAG_A TAG_B         write the candidate fixture manifest (write-once)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import resource
import shutil
import struct
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CKPT_DIR = os.path.join(REPO, "models", "qwen36-src")
INDEX = os.path.join(CKPT_DIR, "model.safetensors.index.json")
CONFIG = os.path.join(CKPT_DIR, "config.json")
CHAIN = os.path.join(REPO, "checkpoints", "dump-manifest-chain-2026-10-26.json")
LLAMA_CAND = os.path.join(REPO, "checkpoints", "m3-candidate-llama-fixture-2026-10-27.json")
INTEGRITY = os.path.join(REPO, "checkpoints", "full-checkpoint-manifest-2026-10-27.json")
OUT_ROOT = os.environ.get("HFDUMP_ROOT", "/tmp/m3-hf-d29")
CANDIDATE = os.path.join(REPO, "checkpoints", "m3-candidate-hf-fixture-2026-10-27.json")
CONTROL_CLASS_MIN = 248044
STAGES = ["x", "h_in", "mixer_out", "x1", "h_mlp", "moe_out", "out"]
STAGE_SOURCE = {
    "x": "forward PRE-hook on the decoder layer's input (== input_layernorm's input)",
    "h_in": "forward hook on input_layernorm's output (§03.2 step 1)",
    "mixer_out": "forward hook on linear_attn/self_attn's output (§03.2 step 2)",
    "x1": "forward PRE-hook on post_attention_layernorm's input (§03.2 step 3's addend pair)",
    "h_mlp": "forward hook on post_attention_layernorm's output (§03.2 step 4)",
    "moe_out": "forward hook on mlp's output (§03.2 step 5)",
    "out": "the decoder layer's return value (§03.2 step 6)",
}
MIXER_ATTR = {"linear_attention": "linear_attn", "full_attention": "self_attn"}


class Refuse(Exception):
    def __init__(self, tag, msg):
        super().__init__(msg)
        self.tag = tag


FINDINGS: list[str] = []


def check(name, ok, detail):
    print(f"{'OK  ' if ok else 'FAIL'} {name} :: {detail}")
    if not ok:
        FINDINGS.append(name)
    return bool(ok)


def mem_available_bytes():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise Refuse("meminfo", "no MemAvailable in /proc/meminfo")


def ids_sha256(ids):
    return hashlib.sha256(b"".join(struct.pack("<q", int(t)) for t in ids)).hexdigest()


def stimulus():
    """The 32 ids, with the equality legs the task demands. Reddens on inequality."""
    chain = json.load(open(CHAIN))
    ids = [int(t) for t in chain["input"]["token_ids"]]
    want = chain["input"]["token_ids_sha256"]
    got = ids_sha256(ids)
    print(f"stim    chain_ids={len(ids)} encoding=le64-per-id recomputes={got == want} "
          f"token_ids_sha256={got}")
    if got != want:
        raise Refuse("stim-recompute", f"recomputed {got} != chain {want}")
    llama = json.load(open(LLAMA_CAND))
    lids = [int(t) for t in llama["stimulus"]["token_ids"]]
    eq = sum(1 for a, b in zip(ids, lids) if a == b)
    print(f"stim    HF-stimulus vs llama-candidate: elementwise {eq}/{len(ids)} equal, "
          f"length_equal={len(ids) == len(lids)}, "
          f"llama_sha={llama['stimulus']['token_ids_sha256']}")
    if not (eq == len(ids) and len(ids) == len(lids)
            and llama["stimulus"]["token_ids_sha256"] == got):
        raise Refuse("stim-mismatch", f"HF stimulus and the llama candidate disagree ({eq}/{len(ids)})")
    ctrl = [t for t in ids if t >= CONTROL_CLASS_MIN]
    print(f"stim    control-class ids (>={CONTROL_CLASS_MIN}) among the {len(ids)}: {ctrl or 'none'}")
    if ctrl:
        raise Refuse("stim-template", f"control-class ids present: {ctrl}")
    return ids


def cpu_only_guards():
    import torch
    print(f"env     HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}")
    avail = torch.cuda.is_available()
    print(f"assert  torch.cuda.is_available() is False -> assertion_holds={avail is False} "
          f"(torch {torch.__version__}, cuda build {torch.version.cuda})")
    if avail is not False:
        raise Refuse("device-visible", "torch.cuda.is_available() is True; this run is CPU-only")


def weight_map_and_families():
    wm = json.load(open(INDEX))["weight_map"]
    fam = {}
    for t, f in wm.items():
        key = "visual" if t.startswith("model.visual.") else "mtp" if t.startswith("mtp.") else "text"
        fam[key] = fam.get(key, 0) + 1
    return wm, fam


def shard_tensor_index():
    """checkpoint tensor name -> (shard file, shape, dtype-string), from the shards' own headers."""
    import struct as _struct
    out = {}
    for f in sorted({v for v in json.load(open(INDEX))["weight_map"].values()}):
        with open(os.path.join(CKPT_DIR, f), "rb") as fh:
            (n,) = _struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(n).decode("utf-8"))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (f, list(v["shape"]), v["dtype"])
    return out


def build_meta_model():
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    cfg = Qwen3_5MoeTextConfig(**json.load(open(CONFIG))["text_config"])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = M.Qwen3_5MoeForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)
    return model, cfg, M


def checkpoint_to_model_names(wm):
    """The prefix rule, stated as data so the strictness check below is a real bijection test.
    `Qwen3_5MoeForCausalLM` puts the text tower at `model.*`; the published checkpoint names the
    same tensors `model.language_model.*`. Visual and mtp towers have no slot here by design."""
    mapped = {}
    for t in wm:
        if t.startswith("model.language_model."):
            mapped["model." + t[len("model.language_model."):]] = t
        elif t == "lm_head.weight":
            mapped[t] = t
    return mapped


def family_bytes():
    """Per-family weight bytes, derived from the safetensors headers."""
    import json as _json
    import struct as _struct
    fam = {}
    for f in sorted({v for v in json.load(open(INDEX))["weight_map"].values()}):
        with open(os.path.join(CKPT_DIR, f), "rb") as fh:
            (n,) = _struct.unpack("<Q", fh.read(8))
            hdr = _json.loads(fh.read(n).decode("utf-8"))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            key = "visual" if k.startswith("model.visual.") else "mtp" if k.startswith("mtp.") else "text"
            fam[key] = fam.get(key, 0) + (int(v["data_offsets"][1]) - int(v["data_offsets"][0]))
    return fam


def ram_statement(needed_note):
    avail = mem_available_bytes()
    fam = family_bytes()
    gib = 1024 ** 3
    print(f"ram     MemAvailable={avail} B = {avail / gib:.2f} GiB   "
          f"text_weights={fam['text']} B = {fam['text'] / gib:.2f} GiB "
          f"(visual={fam['visual'] / gib:.2f} GiB not loaded, mtp={fam['mtp'] / gib:.2f} GiB not loaded)")
    headroom = avail - fam["text"]
    print(f"ram     headroom_after_text_weights={headroom} B = {headroom / gib:.2f} GiB  note={needed_note}")
    return avail, fam["text"], headroom



# ---------------------------------------------------------------- loading
def integrity_gate():
    """D-29: no HF-derived fixture may be trusted unless the checkpoint bytes are tied to the Hub.
    Refuses (with a named reason) rather than producing a fixture on unverified bytes."""
    if not os.path.exists(INTEGRITY):
        raise Refuse("integrity-missing", f"{INTEGRITY} absent — run tools/checkpoint_integrity_d29.py build")
    doc = json.load(open(INTEGRITY))
    rows = doc["shards"]
    n_true = sum(1 for r in rows if r["etag_equals_sha256"] is True)
    bad = [r["file"] for r in rows if r["etag_equals_sha256"] is not True]
    if len(rows) != 26 or n_true != len(rows):
        raise Refuse("integrity-unverified",
                     f"shards_listed={len(rows)} etag_equals_sha256_true={n_true} bad={bad[:3]}")
    sha = hashlib.sha256(open(INTEGRITY, "rb").read()).hexdigest()
    print(f"integ   manifest={os.path.basename(INTEGRITY)} sha256={sha[:16]}… shards={len(rows)} "
          f"etag_equals_sha256_true={n_true}/{len(rows)} verdict=CHECKPOINT-VERIFIED")
    return {"path": os.path.relpath(INTEGRITY, REPO), "sha256": sha, "shards": len(rows),
            "etag_equals_sha256_true": n_true}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def set_module_tensor(model, name, tensor):
    import torch
    parent, _, leaf = name.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    if leaf in mod._parameters:
        mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in mod._buffers:
        mod._buffers[leaf] = tensor
    else:
        raise Refuse("no-slot", f"{name}: no parameter or buffer of that name")


def load_weights(model, mapped, tensor_index):
    """One tensor at a time from the safetensors files, into the meta-built skeleton.

    Memory: the model's own bytes plus one tensor transient, never a whole shard copy. Anti-
    circularity (D-29): every path read here is under models/qwen36-src and no `.gguf` can be,
    which the loop asserts rather than asserts-in-prose."""
    from safetensors.torch import safe_open
    by_shard = {}
    for mname, cname in mapped.items():
        by_shard.setdefault(tensor_index[cname][0], []).append((mname, cname))
    loaded, read_bytes = 0, 0
    for shard in sorted(by_shard):
        path = os.path.join(CKPT_DIR, shard)
        if ".gguf" in path or "unsloth" in path:
            raise Refuse("anti-circularity", f"refusing to read weights from {path}")
        with safe_open(path, framework="pt") as sf:
            for mname, cname in sorted(by_shard[shard]):
                t = sf.get_tensor(cname)
                read_bytes += t.numel() * t.element_size()
                set_module_tensor(model, mname, t)
                loaded += 1
    return loaded, read_bytes, len(by_shard)


def slot_sets(model):
    """Which model slots a checkpoint tensor fills, and which the model computes for itself.

    `state_dict()` excludes non-persistent buffers, and the two rope-frequency buffers are exactly
    that: constants the reference derives from the config, not weights the checkpoint ships. The
    honest predicate is therefore `mapped == state_dict keys`, plus a named restriction that
    anything with no checkpoint source lives under the rotary module. Typed-out names would be a
    claim; this set is read off the model."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    sd = set(model.state_dict().keys())
    no_src = sorted((set(params) | set(buffers)) - sd)
    return params, buffers, sd, no_src


def load_checks(model, mapped, tensor_index, loaded):
    import torch
    params, buffers, sd_keys, no_src = slot_sets(model)
    check("slot_bijection", set(mapped) == sd_keys,
          f"checkpoint_text_tensors={len(mapped)} state_dict_slots={len(sd_keys)} "
          f"tensors_read={loaded} unfilled_slots={len(sd_keys - set(mapped))} "
          f"unused_checkpoint_tensors={len(set(mapped) - sd_keys)} "
          f"no_checkpoint_source={len(no_src)}{no_src}")
    check("built_not_loaded_is_rotary_only",
          all(n.startswith("model.rotary_emb.") for n in no_src),
          f"names_with_no_checkpoint_source={no_src} (only rope constants may be computed by the "
          "reference; anything else here means a weight is missing and would be initialised at "
          "random, which would make the whole fixture fictitious)")
    meta_left = sorted(n for n, p in list(params.items()) + list(buffers.items())
                       if p.device.type == "meta")
    check("nothing_left_on_meta", not meta_left,
          f"meta_tensors={len(meta_left)} {meta_left[:6]}")
    dtypes = sorted({str(p.dtype) for p in params.values()})
    check("every_param_bf16", dtypes == ["torch.bfloat16"], f"param_dtypes={dtypes}")
    ptrs = {}
    for n, p in params.items():
        ptrs.setdefault(p.data_ptr(), []).append(n)
    dupes = {k: v for k, v in ptrs.items() if len(v) > 1}
    check("no_aliased_storage", not dupes,
          f"distinct_data_ptrs={len(ptrs)} params={len(params)} aliased_groups={len(dupes)} "
          f"{list(dupes.values())[:2]}")
    shape_bad = []
    for mname, cname in mapped.items():
        want_shape, want_dtype = tensor_index[cname][1], tensor_index[cname][2]
        got = params.get(mname)
        if got is None or list(got.shape) != want_shape or (
                want_dtype == "BF16" and got.dtype != torch.bfloat16):
            shape_bad.append(mname)
    check("shapes_dtypes_match_headers", not shape_bad,
          f"checked={len(mapped)} mismatches={len(shape_bad)} {shape_bad[:3]}")
    return {"params": len(params), "buffers": len(buffers), "state_dict_slots": len(sd_keys),
            "no_checkpoint_source": no_src, "meta_left": meta_left,
            "distinct_data_ptrs": len(ptrs), "shapes_dtypes_mismatches": len(shape_bad)}


def read_back_checks(model, mapped, tensor_index, stride=97):
    """After every `safe_open` context has closed, re-read a derived sample of the checkpoint and
    compare digests with what the model is holding.

    Why this is not busywork: safetensors hands back tensors backed by the shard's mapping, so a
    dangling mapping, a stale page or a wrong-slot assignment would show up here as a digest
    mismatch — and a fixture whose bytes are not the checkpoint's is fiction. The sample is
    derived: all of layer 0, all of layer 39, `embed_tokens`, `lm_head`, and every `stride`-th of
    the rest, so no block is unrepresented and the population is printed, not asserted."""
    from safetensors.torch import safe_open
    import torch
    params = dict(model.named_parameters())
    sample = []
    for i, (mname, cname) in enumerate(sorted(mapped.items())):
        if (".layers.0." in mname or ".layers.39." in mname or mname in
                ("model.embed_tokens.weight", "lm_head.weight") or i % stride == 0):
            sample.append((mname, cname))
    cache, bad, hashed = {}, [], 0
    for mname, cname in sample:
        shard = tensor_index[cname][0]
        if shard not in cache:
            cache.clear()
            cache[shard] = safe_open(os.path.join(CKPT_DIR, shard), framework="pt")
        fresh = cache[shard].get_tensor(cname)
        a = hashlib.sha256(tensor_raw(params[mname])).hexdigest()
        b = hashlib.sha256(tensor_raw(fresh)).hexdigest()
        hashed += fresh.numel() * fresh.element_size()
        if a != b:
            bad.append(mname)
        del fresh
    for k in list(cache):
        del cache[k]
    check("read_back_matches_checkpoint", not bad,
          f"sample={len(sample)} of {len(mapped)} mapped tensors bytes_rehashed={hashed} "
          f"mismatches={len(bad)} {bad[:3]} (layer0={sum(1 for m,_ in sample if '.layers.0.' in m)} "
          f"layer39={sum(1 for m,_ in sample if '.layers.39.' in m)} stride={stride})")
    return {"sample": len(sample), "mapped": len(mapped), "bytes_rehashed": hashed,
            "mismatches": len(bad), "stride": stride}


def perturbation(model):
    """D-25 guard 5's shape: zero one expert row downstream and prove the comparator can see it.
    Counted, not narrated: the number of elements zeroed is derived from the parameter's shape."""
    import torch
    layer = model.model.layers[PERTURB_LAYER]
    dp = layer.mlp.experts.down_proj
    n = dp.shape[-1]
    before = hashlib.sha256(tensor_raw(dp[PERTURB_EXPERT])).hexdigest()
    with torch.no_grad():
        dp.data[PERTURB_EXPERT, PERTURB_ROW, :] = 0
    after = hashlib.sha256(tensor_raw(dp[PERTURB_EXPERT])).hexdigest()
    print(f"perturb layer={PERTURB_LAYER} param=mlp.experts.down_proj expert={PERTURB_EXPERT} "
          f"row={PERTURB_ROW} elements_zeroed={n} expert_sha_before={before[:16]}… "
          f"after={after[:16]}… changed={before != after}")
    if before == after:
        raise Refuse("perturb-noop", "the injected perturbation did not change the weight")
    return {"layer": PERTURB_LAYER, "expert": PERTURB_EXPERT, "row": PERTURB_ROW,
            "elements_zeroed": int(n), "sha_before": before, "sha_after": after}


# ---------------------------------------------------------------- records
def tensor_raw(tensor):
    import torch
    t = tensor.detach().cpu().contiguous().reshape(-1)
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def write_record(outdir, store, name, block, stage, tensor, derivation, dtype_str=None):
    raw = tensor_raw(tensor)
    path = os.path.join(outdir, name.replace(".", "_") + ".bin")
    with open(path, "wb") as fh:
        fh.write(raw)
    rec = {"name": name, "block": block, "stage": stage, "file": path,
           "bytes": len(raw), "elements": len(raw) // tensor.element_size(),
           "dtype": dtype_str or str(tensor.dtype), "shape": list(tensor.shape),
           "sha256": hashlib.sha256(raw).hexdigest(), "derivation": derivation,
           "source": STAGE_SOURCE.get(stage, "model-level record")}
    store.append(rec)
    return rec


def aggregate_digest(records):
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["name"]):
        h.update(f"{r['name']} {r['sha256']}\n".encode())
    return h.hexdigest()


def top5_line(row):
    import torch
    vals, idx = torch.topk(row.float(), 5)
    return " ".join(f"{int(i)}:{float(v):.8f}" for v, i in zip(vals, idx))


# ---------------------------------------------------------------- run
def do_run(tag, perturb=False):
    import torch
    t0 = time.monotonic()
    outdir = os.path.join(OUT_ROOT, "dumps", tag)
    if os.path.isdir(outdir):
        raise Refuse("write-once", f"{outdir} already exists; a run never overwrites its own bytes")
    os.makedirs(outdir)
    torch.set_num_threads(8)
    ids = stimulus()
    integ = integrity_gate()
    avail, need, headroom = ram_statement("text tower must fit in MemAvailable with margin")
    if headroom < 6 * 1024 ** 3:
        raise Refuse("ram-refusal",
                     f"MemAvailable={avail} B minus text weights={need} B leaves {headroom} B; "
                     "refusing to thrash. Fallback: tools/dump_chain.py's one-block-resident chain "
                     "(layer-by-layer), which is the approved cheaper regime")
    wm, fam = weight_map_and_families()
    fb = family_bytes()
    tensor_index = shard_tensor_index()
    mapped = checkpoint_to_model_names(wm)
    model, cfg, M = build_meta_model()
    params_before = len(dict(model.named_parameters()))
    loaded, read_bytes, n_shards = load_weights(model, mapped, tensor_index)
    with torch.device("cpu"):
        model.model.rotary_emb = M.Qwen3_5MoeTextRotaryEmbedding(config=cfg)
    lchecks = load_checks(model, mapped, tensor_index, loaded)
    rchecks = read_back_checks(model, mapped, tensor_index)
    print(f"load    shards_read={n_shards} tensors_read={loaded} bytes_read={read_bytes} "
          f"params_before_meta={params_before} MemAvailable_now={mem_available_bytes()}")
    if perturb:
        pert = perturbation(model)
    else:
        pert = None
    model.eval()
    store = {}
    handles = attach_hooks(model, store, cfg)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    for h in handles:
        h.remove()
    logits_all = out.logits.detach()
    logits_last = logits_all[0, -1, :]
    argmax = int(torch.argmax(logits_last).item())
    print(f"forward blocks={len(model.model.layers)} logits_shape={list(logits_all.shape)} "
          f"last_position_row={list(logits_last.shape)} argmax_id={argmax} "
          f"argmax_logit={float(logits_last.float()[argmax]):.8f} top5={top5_line(logits_last)}")

    records = []
    layer_types = json.load(open(CONFIG))["text_config"]["layer_types"]
    missing_stages = []
    for i in range(len(model.model.layers)):
        blk = f"layer{i}"
        got = store.get(blk, {})
        miss = [s for s in STAGES if s not in got]
        if miss:
            missing_stages.append(f"{blk}:{miss}")
            continue
        for stage in STAGES:
            write_record(outdir, records, f"hf40.{blk}.{stage}", blk, stage, got[stage],
                         STAGE_SOURCE[stage])
    if missing_stages:
        raise Refuse("hook-gaps", f"stages never captured: {missing_stages[:4]}")
    normed = store["__model__"]["normed"]
    write_record(outdir, records, "hf40.normed", "__model__", "normed", normed,
                 "forward hook on model.norm's output (all 32 positions; the last row is the "
                 "one greedy sampling reads)")
    write_record(outdir, records, "hf40.logits-last-token", "__model__", "logits", logits_last,
                 "lm_head output at the last position of the 32-token prompt, all "
                 f"{logits_last.numel()} vocabulary rows, no penalties/mask/grammar")
    n_stage = sum(1 for r in records if r["block"].startswith("layer"))
    check("record_population", n_stage == len(STAGES) * len(model.model.layers),
          f"per_block_records={n_stage} stages={len(STAGES)} blocks={len(model.model.layers)} "
          f"identity {len(STAGES)}x{len(model.model.layers)}={len(STAGES) * len(model.model.layers)}; "
          f"model_level_records={len(records) - n_stage}")
    agg = aggregate_digest(records)
    host = {"peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mem_available_before_load_bytes": avail, "headroom_bytes": headroom,
            "host_wall_s": round(time.monotonic() - t0, 3),
            "threads": torch.get_num_threads()}
    doc = {"kind": "hf40-run", "tag": tag, "date": "2026-10-27",
           "producing_command": " ".join(sys.argv),
           "interpreter": {"executable": sys.executable, "version": sys.version.split()[0],
                           "torch": torch.__version__},
           "transformers": __import__("transformers").__version__,
           "attn_implementation": getattr(cfg, "_attn_implementation", None),
           "model": {"config": os.path.relpath(CONFIG, REPO), "config_sha256": sha_file(CONFIG),
                     "class": "Qwen3_5MoeForCausalLM (text tower only)",
                     "num_hidden_layers": cfg.num_hidden_layers,
                     "layer_types": layer_types,
                     "linear_layers": [i for i, t in enumerate(layer_types) if t == "linear_attention"],
                     "full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"]},
           "weights_read_from": os.path.relpath(CKPT_DIR, REPO),
           "shares_no_bytes_with_engine_input": {"engine_input": "models/unsloth-q4km/*.gguf",
                                                "hf_input": "models/qwen36-src/*.safetensors",
                                                "asserted_in_load_weights": True},
           "checkpoint_integrity": integ, "stimulus": {"token_ids": ids, "n_tokens": len(ids),
                                                      "token_ids_sha256": ids_sha256(ids),
                                                      "source_manifest": os.path.relpath(CHAIN, REPO)},
           "families_not_loaded": {"visual_bytes": fb["visual"], "mtp_bytes": fb["mtp"]},
           "load_checks": lchecks, "read_back_checks": rchecks,
           "perturbation": pert, "records": records,
           "logits": {"argmax_id": argmax, "n_vocab": int(logits_last.numel()),
                      "sha256": [r for r in records if r["stage"] == "logits"][0]["sha256"],
                      "dtype": str(logits_last.dtype), "top5": top5_line(logits_last),
                      "definition": "greedy argmax over all vocabulary rows at the last position; "
                                    "no penalties, no bad-token mask, no grammar, no temperature"},
           "aggregate_digest": agg, "host": host,
           "counts": {"records": len(records), "stages_per_block": len(STAGES),
                      "blocks": len(model.model.layers)},
           "clocks": "host wall-clock and host RSS only; no device clock read; no timing taken (device)"}
    mpath = os.path.join(OUT_ROOT, "manifests", f"hf40-{tag}.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"records wrote={len(records)} files bytes={sum(r['bytes'] for r in records)} "
          f"aggregate_digest={agg}")
    print(f"RUN tag={tag} records={len(records)} findings={len(FINDINGS)}{FINDINGS[:3]} "
          f"argmax_id={argmax} aggregate_digest={agg} "
          f"peak_rss_kb={host['peak_rss_kb']} mem_available_before_load={avail} "
          f"host_wall_s={host['host_wall_s']} manifest={mpath} no timing taken")
    return 1 if FINDINGS else 0



# ---------------------------------------------------------------- hooks
def _first_arg(args, kwargs):
    """A pre-hook must find the activation whether the caller passed it positionally or by name;
    transformers' own layer loop has used both forms across versions."""
    if args:
        return args[0]
    return kwargs["hidden_states"]


def attach_hooks(model, store, cfg):
    """Every captured value is the reference's own, observed by a hook. Nothing here computes a
    stage; a stage that the module list does not produce is a refusal in do_run, not a guess."""
    layer_types = json.load(open(CONFIG))["text_config"]["layer_types"]

    def put(blk, stage, t):
        store.setdefault(blk, {})[stage] = t

    handles = []
    for i, layer in enumerate(model.model.layers):
        blk = f"layer{i}"
        handles.append(layer.register_forward_pre_hook(
            lambda m, args, kwargs, blk=blk: put(blk, "x", _first_arg(args, kwargs)),
            with_kwargs=True))
        handles.append(layer.register_forward_hook(
            lambda m, inp, out, blk=blk: put(blk, "out", out[0] if isinstance(out, tuple) else out)))
        mixer = getattr(layer, MIXER_ATTR[layer_types[i]])
        handles.append(mixer.register_forward_hook(
            lambda m, inp, out, blk=blk: put(blk, "mixer_out",
                                             out[0] if isinstance(out, tuple) else out)))
        handles.append(layer.input_layernorm.register_forward_hook(
            lambda m, inp, out, blk=blk: put(blk, "h_in", out)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, args, kwargs, blk=blk: put(blk, "x1", _first_arg(args, kwargs)),
            with_kwargs=True))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            lambda m, inp, out, blk=blk: put(blk, "h_mlp", out)))
        handles.append(layer.mlp.register_forward_hook(
            lambda m, inp, out, blk=blk: put(blk, "moe_out", out)))
    handles.append(model.model.norm.register_forward_hook(
        lambda m, inp, out: put("__model__", "normed", out)))
    return handles


# ---------------------------------------------------------------- pair / control
def do_pair(tag_a, tag_b, control_tag):
    a = json.load(open(os.path.join(OUT_ROOT, "manifests", f"hf40-{tag_a}.json")))
    b = json.load(open(os.path.join(OUT_ROOT, "manifests", f"hf40-{tag_b}.json")))
    ra = {r["name"]: r for r in a["records"]}
    rb = {r["name"]: r for r in b["records"]}
    common = sorted(set(ra) & set(rb))
    differing = [n for n in common if ra[n]["sha256"] != rb[n]["sha256"]]
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    print(f"pair    {tag_a}: records={len(ra)} aggregate_digest={a['aggregate_digest']}")
    print(f"pair    {tag_b}: records={len(rb)} aggregate_digest={b['aggregate_digest']}")
    print(f"pair    keys compared={len(common)} differing={len(differing)} only_in_first={len(only_a)} "
          f"only_in_second={len(only_b)} aggregate_equal={a['aggregate_digest'] == b['aggregate_digest']}")
    ok = check("determinism_pair", not differing and not only_a and not only_b
               and a["aggregate_digest"] == b["aggregate_digest"] and len(common) > 0,
               f"records_compared={len(common)} differing={len(differing)} {differing[:3]} "
               f"aggregate_equal={a['aggregate_digest'] == b['aggregate_digest']}")
    det = {"runs": [tag_a, tag_b], "run_manifests": [os.path.join(OUT_ROOT, "manifests",
             f"hf40-{tag_a}.json"), os.path.join(OUT_ROOT, "manifests", f"hf40-{tag_b}.json")],
           "aggregate_digest_run_1": a["aggregate_digest"], "aggregate_digest_run_2": b["aggregate_digest"],
           "aggregate_digests_equal": a["aggregate_digest"] == b["aggregate_digest"],
           "records_compared": len(common), "records_differing": len(differing),
           "clause": "D-26 clause 4 (determinism is a precondition) re-established on HF under D-29"}
    # The control that makes the verdict above mean something: a run with one expert row zeroed
    # must be seen by the SAME comparator, and the change must be downstream-only.
    cpath = os.path.join(OUT_ROOT, "manifests", f"hf40-{control_tag}.json")
    if not os.path.exists(cpath):
        raise Refuse("control-missing", f"{cpath} absent — run `perturb --tag {control_tag}`")
    c = json.load(open(cpath))
    rc = {r["name"]: r for r in c["records"]}
    diff_c = [n for n in common if ra[n]["sha256"] != rc.get(n, {}).get("sha256")]
    up = [n for n in diff_c if re.match(r"hf40\.layer(\d+)\.", n)
          and int(re.match(r"hf40\.layer(\d+)\.", n).group(1)) < c["perturbation"]["layer"]]
    print(f"control perturb={control_tag} vs {tag_a}: records_compared={len(common)} "
          f"differing={len(diff_c)} first_changed={min(diff_c) if diff_c else 'none'} "
          f"upstream_of_perturb_layer_{c['perturbation']['layer']}_changed={len(up)}")
    ok = check("comparator_not_vacuous", len(diff_c) > 0 and not up,
               f"a real weight change moves {len(diff_c)} of {len(common)} records and 0 upstream "
               f"ones; if this read 0 the pair verdict above would prove nothing") and ok
    det["non_vacuity_control"] = {"comparison_against": control_tag,
                                 "perturbation": c["perturbation"],
                                 "records_compared": len(common), "differing": len(diff_c),
                                 "upstream_of_perturbation_changed": len(up),
                                 "first_changed": min(diff_c) if diff_c else None}
    return 0 if ok else 1, det


# ---------------------------------------------------------------- candidate fixture
def do_candidate(tag_a, tag_b, control_tag):
    if os.path.exists(CANDIDATE):
        raise Refuse("write-once", f"{CANDIDATE} exists; registering a fixture is the user's act "
                                   "(D-18) and this file never overwrites one")
    rc, det = do_pair(tag_a, tag_b, control_tag)
    if rc:
        raise Refuse("pair-red", "refusing to write a candidate whose determinism pair is red")
    a = json.load(open(os.path.join(OUT_ROOT, "manifests", f"hf40-{tag_a}.json")))
    llama = json.load(open(LLAMA_CAND))
    ids = a["stimulus"]["token_ids"]
    doc = {
        "kind": "m3-candidate-hf-fixture", "status": "candidate",
        "registration": "not-registered — adding or replacing pinned fixtures is reserved to the "
                        "user (D-18); this file is a proposal with its evidence attached",
        "decision": "DECISIONS.md D-29 (HF is the authoritative reference where it can run; it now "
                    "runs all 40 blocks)",
        "date": "2026-10-27", "purpose": "M3 material from an UNQUANTIZED reference: per-block "
                 "activations for all 40 blocks, the final-norm row, the logits row and the greedy "
                 "argmax, on the same 32-id stimulus the llama.cpp candidate consumed.",
        "authoritative_for": ["per-block activations (all 40 blocks)",
                              "the 40-block greedy argmax",
                              "the 40-block logits row"],
        "produced_by": {"tool": "tools/hf_full_forward_d29.py", "run_manifest": os.path.join(
                            OUT_ROOT, "manifests", f"hf40-{tag_a}.json"),
                        "run_tags": [tag_a, tag_b],
                        "interpreter": a["interpreter"], "transformers": a["transformers"],
                        "attn_implementation": a["attn_implementation"],
                        "class": a["model"]["class"]},
        "stimulus": {"token_ids": ids, "n_tokens": len(ids),
                     "token_ids_sha256": ids_sha256(ids),
                     "token_ids_sha256_matches_chain": True,
                     "token_ids_sha256_matches_llama_candidate": True,
                     "chain_token_ids_sha256": json.load(open(CHAIN))["input"]["token_ids_sha256"],
                     "llama_candidate_sha": llama["stimulus"]["token_ids_sha256"],
                     "source_manifest": os.path.relpath(CHAIN, REPO),
                     "template_applied": "no — raw ids in, no tokenizer, no chat template; no id "
                                         f">= {CONTROL_CLASS_MIN} is present"},
        "checkpoint": {"directory": os.path.relpath(CKPT_DIR, REPO), "dtype": "bfloat16",
                       "integrity_manifest": a["checkpoint_integrity"],
                       "config": a["model"]["config"], "config_sha256": a["model"]["config_sha256"],
                       "families_not_loaded": a["families_not_loaded"],
                       "layers": a["model"]["num_hidden_layers"]},
        "anti_circularity": {"hf_reads": "models/qwen36-src/*.safetensors (bf16)",
                             "engine_reads": "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                             "shared_weight_bytes": "none",
                             "asserted_where": "load_weights() refuses any path containing .gguf"},
        "records": a["records"], "counts": a["counts"],
        "aggregate_digest": a["aggregate_digest"],
        "logits": a["logits"], "determinism": det, "host": a["host"],
        "coverage": {"layers": list(range(a["model"]["num_hidden_layers"])),
                     "stages_per_layer": STAGES,
                     "model_level_records": ["hf40.normed", "hf40.logits-last-token"],
                     "layers_covered": a["counts"]["blocks"],
                     "comparison_with_llama": "tools/hf_llama_crosscheck_d29.py"},
        "not_licensed": {"bands": "none authored here",
                         "band_question": "the logits band is the user's (§PASS 15 §6); D-18/D-20 "
                                          "forbid reading a band off the build",
                         "hf_vs_llama": "any HF-vs-llama disagreement is an escalation to report, "
                                        "never a side to adopt (D-26 clause 3, D-29)"},
    }
    with open(CANDIDATE, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"CANDIDATE wrote={os.path.relpath(CANDIDATE, REPO)} records={len(a['records'])} "
          f"aggregate_digest={a['aggregate_digest']} argmax_id={a['logits']['argmax_id']} "
          f"status=candidate registration=not-registered")
    return 0


def do_preflight():
    import torch
    t0 = time.monotonic()
    cpu_only_guards()
    ids = stimulus()
    integrity_gate()
    avail, need, headroom = ram_statement("preflight: no weights read yet")
    fits = headroom > 6 * 1024 ** 3
    wm, fam = weight_map_and_families()
    tensor_index = shard_tensor_index()
    model, cfg, M = build_meta_model()
    mapped = checkpoint_to_model_names(wm)
    params, buffers, sd_keys, no_src = slot_sets(model)
    check("name_bijection_on_meta", set(mapped) == sd_keys,
          f"mapped={len(mapped)} state_dict_slots={len(sd_keys)} "
          f"model_only={len(sd_keys - set(mapped))} mapped_only={len(set(mapped) - sd_keys)}")
    check("built_not_loaded_is_rotary_only",
          all(n.startswith("model.rotary_emb.") for n in no_src),
          f"names_with_no_checkpoint_source={no_src}")
    bad = [m for m, c in mapped.items()
           if tensor_index[c][1] != list(params[m].shape) or tensor_index[c][2] != "BF16"]
    check("header_shapes_dtypes", not bad, f"checked={len(mapped)} mismatches={len(bad)} {bad[:3]}")
    print(f"meta    slots_built_on_meta={len(params) + len(buffers)} "
          f"state_dict_slots={len(sd_keys)} checkpoint_text_tensors={len(mapped)} "
          f"buffers={len(buffers)} no_checkpoint_source={no_src}")
    print(f"preflight layers={cfg.num_hidden_layers} stages={len(STAGES)} "
          f"linear_layers={len([i for i, t in enumerate(cfg.layer_types) if t == 'linear_attention'])} "
          f"full_attention_layers={len([i for i, t in enumerate(cfg.layer_types) if t == 'full_attention'])} "
          f"fits_in_MemAvailable={fits}")
    if not fits:
        print("preflight REFUSAL route=layer-by-layer — tools/dump_chain.py runs one block resident; "
              f"MemAvailable={avail} B cannot hold text weights={need} B with margin")
    print(f"PREFLIGHT findings={len(FINDINGS)}{FINDINGS[:4]} "
          f"host_wall_s={round(time.monotonic() - t0, 3)} "
          f"peak_rss_kb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} no timing taken")
    return 1 if FINDINGS else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["preflight", "run", "perturb", "pair", "candidate"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    if args.mode == "preflight":
        return do_preflight()
    cpu_only_guards()
    if args.mode == "run":
        return do_run(args.tag, perturb=False)
    if args.mode == "perturb":
        return do_run(args.tag, perturb=True)
    if args.mode == "pair":
        a, b, c = (args.args + ["p1"])[:3]
        return do_pair(a, b, c)[0]
    if args.mode == "candidate":
        a, b, c = (args.args + ["p1"])[:3]
        return do_candidate(a, b, c)
    raise Refuse("bad-mode", args.mode)


PERTURB_LAYER, PERTURB_EXPERT, PERTURB_ROW = 20, 7, 3


if __name__ == "__main__":
    sys.exit(main())
