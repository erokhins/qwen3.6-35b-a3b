#!/usr/bin/env python3
"""tools/dump_chain.py — pin a CHAINED bf16 forward through real decoder layers, one block
resident at a time, for `Qwen/Qwen3.6-35B-A3B`.

CPU only: torch-CPU, no device allocation, no HIP, no lock, and no device clock is read.
The only clocks here are host wall-clock (`time.monotonic`) and host RSS
(`resource.getrusage`), both of which the task asks for; **no timing taken (device)**.

WHY THIS FILE EXISTS. `DECISIONS.md` §D-25 found, and this file re-proves, that
`layer0.input`, `layer1.input`, `layer39.input` in `data/dumps/2026-10-26/` are
byte-identical: `tools/dump_reference.py` feeds EVERY block the same
`R.make_input(seed, T)` independently (`dump_reference.py::dump_block`, the
`x = R.make_input(tokens, g, seed)` line). The pinned set is a per-block unit-test regime.
Nothing pinned so far has ever validated a residual chain, and there is no embedding,
final-norm, `lm_head` or `logits` record at all — so M3 (pinned logits, greedy argmax) was
unreachable with the current fixture set. This file adds a chained fixture **additively**;
it never touches the existing pins, and it imports their machinery rather than forking it.

WHAT A "BLOCK" IS HERE, and the two substitutions the supervisor approved on 2026-10-27.

1. **Block = the whole decoder layer**, `specification/03 §03.2`'s six steps, run by the
   installed `transformers` reference's own `Qwen3_5MoeDecoderLayer`. A MoE-only chain
   (`h <- h + MoE(h)`) was rejected because it would manufacture a residual stream the
   model does not have, and the final-norm + `lm_head` logits built on it would be
   meaningless while *looking* like D-25 was honoured. Seven stage records per block:
   `x, h_in, mixer_out, x1, h_mlp, moe_out, out` — the seven rows `engine/HANDOFF.md` §4
   has been asking for. None of them is computed by this file: each is a forward hook on a
   reference module (`STAGES`/`STAGE_SOURCE` below says which), so the dumper never
   produces a number it then has to trust.
2. **The block set is the acquired blocks, not 0..3.** Only 4 of 26 shards were acquired
   (README §Download, D-03's sampled audit): layers 0, 1, 38, 39 exist on disk, layers
   2..37 do not. The chain is therefore `layer0 -> layer1 -> layer38 -> layer39`, every
   record carries `contiguous: false` while a gap exists, and `missing_layers` is DERIVED
   from `safetensors.index.json` minus what `models/qwen36-src/` actually holds — a typed
   "2..37" would be a claim, a derived set is a measurement. `checks.gap_declared_for_gap`
   and `checks.no_contiguity_claim_across_gap` redden if either property is broken, and
   both are witnessed against mutated copies of the manifest so they cannot be vacuous.

THE THREE VIEWS (D-25 guard 6: the dumper is never its own oracle).
  A. the chained forward: meta-built reference modules with substituted parameters
     (`build_layer(..., mode="meta")`), one block resident;
  B. `speccheck/moe_ref.py` and `speccheck/gdn_ref.py`, the independent implementations,
     re-deriving one or more stage values per block in the units those files already print
     (`moe_ref.metrics`/`fmt_metrics`, `gdn_ref.aggregates`/`fmt_ag`);
  C. the SAME four layers built the ordinary way — real materialised parameters, no meta,
     no substitution, all four resident — so the meta path has a peer rather than a mirror.
  B and C are separate processes with their own peak-RSS figures; a `--mode all` run of A
  alone is what the residency claim is about.

APPEND-ONLY, WRITE-ONCE. `data/dumps/2026-10-26-chain/` and
`checkpoints/dump-manifest-chain-2026-10-26.json` are created by this file and by nothing
else; either existing already is a refusal with a named tag, not an overwrite. Every write
goes through `open_for_write()`, which asserts the target is inside this file's own output
tree and is not, does not contain and is not contained by, the live per-block dump or
manifest (`dump_reference_gdn.refuse_live_paths`, imported so there is one rule).

PHASES (each its own process, so a peak-RSS figure belongs to the phase that earned it):
    CHAIN     python tools/dump_chain.py --mode chain   --tokens 32 --seed 20261026 --threads 8
    GUARDS    python tools/dump_chain.py --mode guards
    PERTURB   python tools/dump_chain.py --mode perturb --perturb layer38:mlp.experts.down_proj:row=7
    PEER      python tools/dump_chain.py --mode peer
    MANIFEST  python tools/dump_chain.py --mode manifest   (writes the manifest, once)
    VERIFY    python tools/dump_chain.py --mode verify
`--mode manifest` refuses until the four evidence files exist, so a manifest that claims
witnesses nobody ran cannot be written.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "speccheck"))
sys.path.insert(0, str(REPO / "tools"))

import moe_ref as R           # noqa: E402  geometry, shard reader, Block, set_threads
import gdn_ref as G           # noqa: E402  the GDN oracle + its comparator
import cross_check_hf as C    # noqa: E402  the structural bug detector, canon/canon_raw
import dump_reference as D    # noqa: E402  ONE byte contract: sha256, raw payload, dtype-by-filesize
import dump_reference_gdn as DG  # noqa: E402 refuse_live_paths — one path rule, imported not forked

DEFAULT_OUT = "data/dumps/2026-10-26-chain"
DEFAULT_MANIFEST = "checkpoints/dump-manifest-chain-2026-10-26.json"
DEFAULT_BLOCKS = "layer0,layer1,layer38,layer39"
DEFAULT_PERTURB = "layer38:mlp.experts.down_proj:row=7"

# The live per-block infrastructure. Read-only for this file, asserted, not merely avoided.
LIVE_PATHS = DG.LIVE_PATHS
OLD_MOE_MANIFEST = REPO / "checkpoints" / "dump-manifest-2026-10-26.json"

# §The stages — `specification/03 §03.2`'s six steps plus the layer input, in pipeline order.
# The names are the HANDOFF §4 table's names, so the rows that print `referent=absent` in
# engine/block_gate.py have something to be pointed at. Nothing here is computed by this
# file: `STAGE_SOURCE` names the reference module whose hook produced the value.
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
# Which mixer module a layer type carries; read out of the reference's own branch in
# `modeling_qwen3_5_moe.py::Qwen3_5MoeDecoderLayer.__init__`, not typed from memory.
MIXER_ATTR = {"linear_attention": "linear_attn", "full_attention": "self_attn"}
MIXER_ORACLE = {"linear_attention": "speccheck/gdn_ref.py::gdn_block(arm=chunked)",
                "full_attention": None}


def rel(p) -> str:
    """Repo-relative when it is inside the repo, absolute when it is not. A scratch `--out`
    outside the tree must still produce readable records."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- write guards
def refuse_live_paths(*paths: Path) -> None:
    """Imported rule, extra message: this file writes only its own two paths."""
    DG.refuse_live_paths(*paths)


def refuse_existing(path: Path, tag: str) -> None:
    """Write-once. An existing output is a REFUSAL, never an overwrite or an append.

    The tag is named so a reader can tell which guard stopped them, and so a control can
    assert on the tag rather than on an exit code (`pitfalls/40` §name the refusal)."""
    if path.exists():
        raise Refusal(f"{tag}: refusing to run, {path} already exists. This dump is "
                      "append-only/write-once; delete it by hand only if you are sure it "
                      "was never pinned.")


class Refusal(Exception):
    """A guard that stopped the run. Printed as a named tag, not as a traceback."""


def open_for_write(path: Path, out_dir: Path, manifest: Path):
    """The single funnel every write goes through: asserts the destination, then opens.

    Asserting rather than "just avoiding" the live paths is the D-25 guard-1 requirement.
    Two conditions, both checked on RESOLVED paths (a string prefix would be wrong:
    `data/dumps/2026-10-26-chain` is a sibling of the live `data/dumps/2026-10-26`):
    the target must be inside this run's output tree or be this run's manifest, and it must
    survive `refuse_live_paths`."""
    p = Path(path).resolve()
    refuse_live_paths(p)
    inside = out_dir.resolve() in p.parents
    if not (inside or p == manifest.resolve()):
        raise Refusal(f"write-outside-own-tree: {p} is neither inside {out_dir} nor this "
                      "run's manifest; dump_chain.py writes nothing else.")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise Refusal(f"write-once: {p} already exists.")
    return open(p, "wb")


# ------------------------------------------------------------------------ derived census
def shard_census() -> dict:
    """What is actually on disk vs what the committed index says must exist.

    Everything here is DERIVED: the expected layer set is `range(num_hidden_layers)` from
    the config geometry, the expected tensor set per layer is the index's own key list
    under that layer's prefix, and "present" is a `Path.exists()` on the shard file that
    index names. The missing-layer population — the number that drives the download
    decision — is therefore a measurement, not a sentence in a note."""
    index = R.load_index()
    g = R.load_geometry()
    wm = index["weight_map"]
    shards_total = sorted({s for s in wm.values()})
    shards_present = sorted({s for s in shards_total if (R.SHARD_DIR / s).exists()})
    tensors_total = len(wm)
    tensors_present = sum(1 for s in wm.values() if (R.SHARD_DIR / s).exists())

    layers_total = list(range(g.num_hidden_layers))
    layer_tensors = {i: sorted(n for n in wm if n.startswith(f"{R.Block('layer', i).prefix}"))
                     for i in layers_total}
    layers_complete = [i for i in layers_total
                       if all((R.SHARD_DIR / wm[n]).exists() for n in layer_tensors[i])]
    missing = [i for i in layers_total if i not in layers_complete]
    # Witness the census against the config's own layer_types, so "30 linear + 10 full"
    # is checked rather than assumed (pitfalls/40 §derived populations).
    types = json.loads(R.CONFIG_PATH.read_text())["text_config"]["layer_types"]
    if len(types) != g.num_hidden_layers:
        raise Refusal(f"layer_types has {len(types)} entries, geometry says "
                      f"{g.num_hidden_layers} layers")
    return {
        "shards_present": len(shards_present), "shards_total": len(shards_total),
        "shard_files_present": shards_present, "shard_files_total": shards_total,
        "tensors_present": tensors_present, "tensors_total": tensors_total,
        "layers_total": len(layers_total), "layers_complete": layers_complete,
        "missing_layers": missing, "n_missing_layers": len(missing),
        "layer_tensor_count": {str(i): len(layer_tensors[i]) for i in layers_total},
        "layer_types": {str(i): types[i] for i in layers_total},
    }


# ----------------------------------------------------------------------------- host costs
def peak_rss_gib() -> float:
    """Peak host RSS of THIS process, in GiB. A host quantity; not a device measurement."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def current_rss_gib() -> float:
    with open("/proc/self/statm", "r") as fh:
        return int(fh.read().split()[1]) * 4096 / 1073741824.0


class Cost:
    """Host wall-clock and RSS bookkeeping. No device clock is read anywhere near this."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.marks = {}
        self.rss_marks = {}

    def mark(self, name: str) -> float:
        dt = time.monotonic() - self.t0
        self.marks[name] = dt
        self.rss_marks[name] = current_rss_gib()
        return dt

    def span(self, name: str) -> float:
        return self.mark(name) - self.t0

    def since(self, a: str, b: str) -> float:
        return self.marks[b] - self.marks[a]

    def rows(self) -> list:
        return [{"phase": k, "t_s": round(v, 3), "rss_gib": round(self.rss_marks[k], 3)}
                for k, v in self.marks.items()]


# --------------------------------------------------------------------- reference modules
def hf():
    """The lazy HF import, mirroring `gdn_ref._hf_modules` so there is one way in."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    import transformers
    cfg = json.loads(R.CONFIG_PATH.read_text())["text_config"]
    cfg_text = Qwen3_5MoeTextConfig(**cfg)
    cfg_text._experts_implementation = "eager"   # [REF] pin the reference loop, as cross_check_hf does
    return M, cfg_text, transformers.__version__


def sha256_tensor(t: torch.Tensor) -> str:
    """Digest of a tensor's raw little-endian bytes — the same payload `write_tensor`
    writes, so a digest in a witness row and a digest in a dump row are the same ruler."""
    raw = t.detach().contiguous()
    if raw.dtype in (torch.bfloat16, torch.float16):
        payload = raw.view(torch.uint16).numpy().tobytes()   # bitcast, never cast (pitfalls/30)
    else:
        payload = raw.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_layer_params(block: R.Block) -> tuple[dict, dict, int]:
    """Every tensor the reference's own `Qwen3_5MoeDecoderLayer` needs, one at a time.

    The EXPECTED set is derived twice and cross-checked: from the module's
    `named_parameters()` (what the reference would allocate) and from the committed shard
    index (what the checkpoint stores). A tensor on one list and not the other is a refusal
    — this is the census that makes "the substitution covers every parameter" a count
    rather than a belief.
    """
    M, cfg_text, _v = hf()
    with torch.device("meta"):
        probe = M.Qwen3_5MoeDecoderLayer(cfg_text, layer_idx=block.index)
    want = sorted(n for n, _ in probe.named_parameters())
    buffers = sorted(n for n, _ in probe.named_buffers())
    if buffers:
        raise Refusal(f"{block.name}: the decoder layer carries buffers {buffers}; this file's "
                      "substitution witness enumerates parameters only, so a buffer left on "
                      "meta would go unwitnessed")
    index = R.load_index()
    stored = sorted(n[len(block.prefix):] for n in index["weight_map"]
                    if n.startswith(block.prefix))
    if set(want) != set(stored):
        raise Refusal(f"{block.name}: the reference's parameter set and the shard index "
                      f"disagree — module_only={sorted(set(want) - set(stored))} "
                      f"index_only={sorted(set(stored) - set(want))}")
    del probe
    tensors, sources = {}, {}
    for pname in want:
        tname = block.prefix + pname
        shard = R.shard_of(tname, index)
        path = R.SHARD_DIR / shard
        if not path.exists():
            c = shard_census()
            raise Refusal(f"{block.name}.{pname}: lives in {shard}, which is not in "
                          f"{R.SHARD_DIR}. Shard census: {c['shards_present']}/"
                          f"{c['shards_total']} shards, {c['tensors_present']}/"
                          f"{c['tensors_total']} tensors, {c['n_missing_layers']} of "
                          f"{c['layers_total']} layers incomplete. That is the download gap "
                          "the census derives, not a defect in this file")
        tensors[pname] = R.load_tensor(path, tname)
        sources[pname] = shard
    return tensors, sources, len(want)


def build_layer(block: R.Block, mode: str, tensors: dict, sources: dict, *,
                break_substitution: str | None = None, perturb: tuple | None = None) -> tuple:
    """Construct one decoder layer and put the checkpoint's tensors in it.

    `mode="meta"` — build on meta, then substitute each parameter (the cheap path, and the
    one that needs witnesses: a parameter that never got substituted stays a meta tensor or
    a random init and the forward still runs, confidently, on garbage).
    `mode="materialised"` — construct normally, letting the reference initialise itself,
    then hand the whole dict to `load_state_dict(assign=True)`. No meta, no substitution:
    this is the PEER that the meta path is compared against. It gets the SAME per-parameter
    digest witness, because "the peer is a different implementation" is only worth saying
    if the peer is also shown to be holding the checkpoint's bytes.

    `break_substitution` is the control knob for `substitution-is-load-bearing`: name a
    parameter and it is left at ZEROS instead of substituted. It must change the block
    output, or nothing above it was ever reading it.
    """
    M, cfg_text, _v = hf()
    import torch.nn as nn
    if mode not in ("meta", "materialised"):
        raise Refusal(f"unknown build mode {mode!r}; expected 'meta' or 'materialised'")
    if mode == "meta":
        with torch.device("meta"):
            layer = M.Qwen3_5MoeDecoderLayer(cfg_text, layer_idx=block.index)
    else:
        if break_substitution:
            raise Refusal("the break-substitution control belongs to the meta path only; the "
                          "peer view must never be sabotaged")
        with default_dtype(torch.bfloat16):
            layer = M.Qwen3_5MoeDecoderLayer(cfg_text, layer_idx=block.index)  # self-initialises
        sd = {n: tensors[n] for n in tensors}
        missing, unexpected = layer.load_state_dict(sd, assign=True, strict=True)
        if missing or unexpected:
            raise Refusal(f"{block.name}: load_state_dict(missing={missing}, "
                          f"unexpected={unexpected}) — the peer is not holding the "
                          "checkpoint's tensors")
    substituted = []
    if mode == "meta":
        for pname, want_shape in sorted(((n, tuple(p.shape))
                                         for n, p in layer.named_parameters())):
            t = tensors[pname]
            if tuple(t.shape) != want_shape:
                raise Refusal(f"{block.name}.{pname}: stored shape {tuple(t.shape)} != the "
                              f"reference's own {want_shape}")
            obj = layer
            for part in pname.split(".")[:-1]:
                obj = getattr(obj, part)
            setattr(obj, pname.split(".")[-1],
                    nn.Parameter(torch.zeros_like(t) if break_substitution == pname else t,
                                 requires_grad=False))
    live_params = dict(layer.named_parameters())
    left = [n for n, p in live_params.items() if p.device.type == "meta"]
    if left:
        raise Refusal(f"{block.name}: {len(left)} parameter(s) still on meta after "
                      f"{mode} construction: {left}. A meta parameter is the exact failure "
                      "mode a param-substituted model is famous for.")
    witness = []
    for pname, want_shape in sorted(((n, tuple(p.shape)) for n, p in live_params.items())):
        substituted.append(pname)
        # The comparison must be against the SHARD, not against the tensor we just put in:
        # hashing the live parameter against the same object it came from is an identity,
        # not a test. `R.load_tensor` reads the slice again through safetensors.
        live = live_params[pname]
        fresh = R.load_tensor(R.SHARD_DIR / sources[pname], block.prefix + pname)
        witness.append({"parameter": pname, "shape": list(want_shape),
                        "dtype": str(live.dtype), "source_shard": sources[pname],
                        "source_tensor": block.prefix + pname,
                        "substituted": break_substitution != pname,
                        "deliberately_broken": break_substitution == pname,
                        "expected_to_differ_from_shard": bool(perturb and perturb[0] == block.name
                                                              and perturb[1] == pname),
                        "sha256_live": sha256_tensor(live), "sha256_shard": sha256_tensor(fresh)})
        del fresh, live
    return layer, substituted, witness


def substitution_verdict(block: R.Block, witness: list, expected: int) -> dict:
    """`substituted=N/expected=M`, with M from the index and every N individually hashed.

    A substituted parameter whose digest does not equal an independently-read safetensors
    slice is not substituted, it is something else that happens to have the right shape.
    `deliberately_broken` entries are the `substitution-is-load-bearing` control's own
    doing and are counted separately, so the control can leave one parameter at zeros
    without making the census read like a harness defect."""
    broken = [w["parameter"] for w in witness if w.get("deliberately_broken")]
    pert = [w for w in witness if w.get("expected_to_differ_from_shard")]
    n_ok = [w for w in witness if w["substituted"] and w["sha256_live"] == w["sha256_shard"]
            and not w.get("expected_to_differ_from_shard")]
    bad = [w["parameter"] for w in witness if w["substituted"]
           and w["sha256_live"] != w["sha256_shard"]
           and not w.get("expected_to_differ_from_shard")]
    # A perturbed tensor MUST differ from the shard: if its live digest still equals the
    # freshly-read slice, the perturbation never reached the forward and the downstream
    # change-map would be measuring nothing.
    moved = [w["parameter"] for w in pert if w["sha256_live"] != w["sha256_shard"]]
    silent = [w["parameter"] for w in pert if w["sha256_live"] == w["sha256_shard"]]
    return {"block": block.name, "substituted": len(n_ok) + len(moved),
            "expected": int(expected),
            "deliberately_broken": broken, "perturbed": moved,
            "digest_mismatch": bad, "perturbation_did_not_reach_the_forward": silent,
            "ok": (len(n_ok) + len(moved) + len(silent) + len(broken) == int(expected)
                   and not bad and not silent)}


# --------------------------------------------------------------------- the chained forward
@dataclass
class Capture:
    """Forward hooks on the reference's own modules. The dumper computes nothing.

    Every stage is a module's input or output as the reference produced it. `x1` in
    particular is the PRE-hook argument of `post_attention_layernorm` — it is the value the
    reference itself carries into the second norm, not this file re-adding the residual."""
    stages: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)

    def install(self, layer, block_type: str) -> None:
        mixer = getattr(layer, MIXER_ATTR[block_type])

        def keep(key):
            def hook(_mod, _inp, out):
                self.stages[key] = out[0] if isinstance(out, tuple) else out
                self.calls[key] = self.calls.get(key, 0) + 1
            return hook

        def keep_pre(key):
            def hook(_mod, inp):
                self.stages[key] = inp[0]
                self.calls[key] = self.calls.get(key, 0) + 1
            return hook

        layer.register_forward_pre_hook(keep_pre("x"))
        layer.register_forward_hook(keep("out"))
        layer.input_layernorm.register_forward_hook(keep("h_in"))
        mixer.register_forward_hook(keep("mixer_out"))
        layer.post_attention_layernorm.register_forward_pre_hook(keep_pre("x1"))
        layer.post_attention_layernorm.register_forward_hook(keep("h_mlp"))
        layer.mlp.register_forward_hook(keep("moe_out"))


def run_layer(layer, x: torch.Tensor, rotary, cfg_text, tokens: int,
              block_type: str) -> torch.Tensor:
    """One decoder-layer forward through the reference, exactly as `Qwen3_5MoeTextModel`
    calls it (same 4-D position_ids expand, same mask mapping, same kwargs).

    The mask helpers are HF's own: `create_causal_mask` for a full_attention layer,
    `create_recurrent_attention_mask` for a linear_attention layer. This file does not
    invent a mask."""
    import transformers.masking_utils as MU
    with torch.no_grad():
        position_ids = torch.arange(tokens).view(1, 1, -1).expand(4, 1, -1)
        text_position_ids = position_ids[0]
        position_embeddings = rotary(x, position_ids[1:])
        mask_kwargs = {"config": cfg_text, "inputs_embeds": x, "attention_mask": None,
                       "past_key_values": None, "position_ids": text_position_ids}
        mask = {"full_attention": MU.create_causal_mask(**mask_kwargs),
                "linear_attention": MU.create_recurrent_attention_mask(**mask_kwargs)}
        return layer(x, position_embeddings=position_embeddings,
                     attention_mask=mask[block_type],
                     position_ids=text_position_ids, use_cache=False)


def claim_for_write(path: Path, out_dir: Path, manifest: Path) -> None:
    """Assert the destination, then take it exclusively, before `dump_reference` writes it.

    `dump_reference.write_tensor` is the byte contract (raw little-endian payload, dtype by
    file size, sha256) and is NOT forked. Enforcement lives here: the path is checked
    against the live-path rule and this run's own tree, then created with O_EXCL so a second
    claim of the same name is a refusal. `write_tensor` then fills the file we own."""
    p = Path(path).resolve()
    refuse_live_paths(p)
    if not (out_dir.resolve() in p.parents or p == manifest.resolve()):
        raise Refusal(f"write-outside-own-tree: {p} is neither inside {out_dir} nor this "
                      "run's manifest.")
    if p.exists():
        raise Refusal(f"write-once: {p} already exists; this dump never overwrites.")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)


def write_stage(t: torch.Tensor, path: Path, out_dir: Path, manifest: Path) -> dict:
    claim_for_write(path, out_dir, manifest)
    return D.write_tensor(t, path, base=REPO)


def read_stage(path: Path, shape, dtype) -> torch.Tensor:
    """Read a dump back by SHAPE AND DTYPE FROM THE CALLER, after asserting the file size
    can hold it — `pitfalls/30` §Read a dump's dtype off its file size, in the direction a
    chained reader needs: a [T,H] bf16 file read as f32 still prints plausible numbers."""
    n = 1
    for d in shape:
        n *= int(d)
    want = n * D.dtype_size(str(dtype))
    size = path.stat().st_size
    if size != want:
        raise Refusal(f"{path.name}: {size} B cannot hold {list(shape)} x {dtype} "
                      f"({want} B); refusing to guess a dtype")
    raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    if dtype is torch.bfloat16:
        return raw.view(torch.uint16).view(dtype).reshape(list(shape))   # bitcast, not cast
    return raw.view(dtype).reshape(list(shape))


def token_ids(tokens: int, seed: int, vocab_size: int) -> torch.Tensor:
    """Deterministic token ids from a private generator, like `moe_ref.make_input`'s draw.

    Drawn from the first 100k vocabulary rows so no chat-template special id can be pulled
    into the fixture or onto a terminal (this family: eos/bos 248044, end-of-turn 248046,
    think-boundary 248068/248069 — `README.md` §Token ids). The exclusion is ASSERTED
    against the config's own bos/eos ids, not assumed, and the ids are recorded in the
    manifest so the prompt is reproducible without a tokenizer on this box."""
    cfg = json.loads(R.CONFIG_PATH.read_text())["text_config"]
    reserved = {cfg.get("bos_token_id"), cfg.get("eos_token_id"), cfg.get("pad_token_id"),
                248046, 248068, 248069}
    reserved.discard(None)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, min(vocab_size, 100_000), (tokens,), generator=gen)
    clash = sorted(int(i) for i in ids if int(i) in reserved)
    if clash:
        raise Refusal(f"token-ids-hit-reserved: {clash} are structural ids; widen the draw "
                      "range or change the seed rather than shipping them")
    return ids


# --------------------------------------------------------------------- independent views
def moe_weights(block: R.Block, tensors: dict, sources: dict, g: R.Geometry) -> R.BlockWeights:
    """`speccheck/moe_ref.py`'s weight bundle, from the tensors ALREADY loaded for the
    forward. Re-reading the 1.5 GiB expert stack for the oracle would double peak RSS for
    no additional independence: independence here is a second IMPLEMENTATION
    (`moe_ref.expert_gather`'s grouped/expert-ascending form), not a second file read."""
    w = R.BlockWeights(block, {role: tensors[suf] for role, suf in R.SUFFIXES.items()},
                       {role: sources[suf] for role, suf in R.SUFFIXES.items()})
    w.shapes_ok(g)                       # config-derived shapes before anything consumes them
    return w


def gdn_weights(block: R.Block, tensors: dict, sources: dict,
                gg: G.GdnGeometry) -> G.GdnWeights:
    if block.index not in gg.linear_layers:
        raise Refusal(f"layer {block.index} is not a linear_attention layer; the GDN oracle "
                      "does not serve it (gdn_ref.load_gdn_block_weights makes the same refusal)")
    w = G.GdnWeights(block, {role: tensors[suf] for role, suf in G.SUFFIXES.items()},
                     {role: sources[suf] for role, suf in G.SUFFIXES.items()})
    w.shapes_ok(gg)
    return w


def oracle_rows(block: R.Block, block_type: str, out_dir: Path, tokens: int,
                tensors: dict, sources: dict, g: R.Geometry, gg, cfg_text) -> list:
    """View B: re-derive stage values with `speccheck/moe_ref.py` and `speccheck/gdn_ref.py`.

    Both sides are read back off the FILES on disk, so this compares the independent
    implementation against what was pinned rather than against a tensor still in this
    process's memory. Units are the ones those files already print: `moe_ref.fmt_metrics`
    and `gdn_ref.fmt_ag`. No band is authored here; the only threshold applied is
    `cross_check_hf.BUG_DETECTOR_ONE_MINUS_COS`, the structural bug detector that file
    derives from bf16's 2^-9 relative width."""
    rows = []
    h_mlp = read_stage(out_dir / f"{block.name}.h_mlp.bin", (tokens, g.hidden_size),
                       torch.bfloat16)
    moe_out = read_stage(out_dir / f"{block.name}.moe_out.bin", (tokens, g.hidden_size),
                         torch.bfloat16)
    w = moe_weights(block, tensors, sources, g)
    mine = R.moe_block(h_mlp, w, g, "ref")["combined_out"]
    m = R.metrics(mine, moe_out, name=f"{block.name}.moe_out vs moe_ref.combined_out")
    rows.append({"view": "B/moe_ref", "block": block.name, "stage": "moe_out",
                 "referent": "moe_ref.moe_block(ref).combined_out",
                 "line": R.fmt_metrics(m), "metrics": m,
                 "bit_exact": bool((mine == moe_out).all().item())})
    if m["one_minus_cos"] > C.BUG_DETECTOR_ONE_MINUS_COS:
        raise Refusal(f"{block.name}: moe_ref-vs-chain 1-cos={m['one_minus_cos']:.3e} is above "
                      "the structural bug detector; that is a broken read, not a form gap")

    if MIXER_ORACLE[block_type] is None:
        rows.append({"view": "B/gdn_ref", "block": block.name, "stage": "mixer_out",
                     "referent": "absent",
                     "line": (f"{block.name}.mixer_out{'':<28} independent_oracle=none: "
                              f"layer_types[{block.index}]=full_attention and speccheck has no "
                              "full_attention oracle; this block's independent witness is its "
                              "moe_out row above"),
                     "metrics": None, "bit_exact": None})
        return rows

    h_in = read_stage(out_dir / f"{block.name}.h_in.bin", (tokens, g.hidden_size),
                      torch.bfloat16)
    mixer_out = read_stage(out_dir / f"{block.name}.mixer_out.bin", (tokens, g.hidden_size),
                           torch.bfloat16)
    wg = gdn_weights(block, tensors, sources, gg)
    got = G.gdn_block(h_in.unsqueeze(0), wg, gg, "chunked", G.NO_FAULTS, mode="prefill")["y"]
    got = got.reshape(tokens, g.hidden_size)     # gdn_ref returns [B,T,H]; the dump is [T,H]
    if got.numel() != mixer_out.numel():
        raise Refusal(f"{block.name}: gdn_ref returned {got.numel()} elements, the pinned "
                      f"mixer_out has {mixer_out.numel()}")
    ag = G.aggregates(got, mixer_out, name=f"{block.name}.mixer_out vs gdn_ref.y",
                      stage="y")
    rows.append({"view": "B/gdn_ref", "block": block.name, "stage": "mixer_out",
                 "referent": "gdn_ref.gdn_block(chunked,prefill).y",
                 "line": G.fmt_ag(ag), "metrics": ag,
                 "bit_exact": bool((got == mixer_out).all().item())})
    if ag["one_minus_cos"] > C.BUG_DETECTOR_ONE_MINUS_COS:
        raise Refusal(f"{block.name}: gdn_ref-vs-chain 1-cos={ag['one_minus_cos']:.3e} above "
                      "the structural bug detector")
    return rows


def residual_identity(rows_out: dict) -> list:
    """The hook wiring's own negative control: the two §03.2 adds must hold bit for bit.

    `x1 = x + mixer_out` and `out = x1 + moe_out`, both in the model dtype — the same
    operations the reference performs inline, so a capture that was mis-wired (say `x1`
    actually holding `h_mlp`) reddens here instead of quietly pinning the wrong tensor.
    `wired_wrong` is the witnessed-RED arm: it swaps two captures and must be rejected."""
    def check(name, a, b):
        eq = bool(torch.equal(a, b))
        nd = int((a.to(torch.float64) != b.to(torch.float64)).sum().item())
        return {"check": name, "bit_exact": eq, "ndiff": nd, "n": int(a.numel())}

    good = [check("x1 == x + mixer_out", rows_out["x1"], rows_out["x"] + rows_out["mixer_out"]),
            check("out == x1 + moe_out", rows_out["out"], rows_out["x1"] + rows_out["moe_out"])]
    swapped = dict(rows_out)
    swapped["x1"] = rows_out["h_mlp"]
    bad = check("CONTROL swapped x1<-h_mlp", swapped["x1"],
                rows_out["x"] + rows_out["mixer_out"])
    if bad["bit_exact"]:
        raise Refusal("residual-identity-control-did-not-fire: swapping the x1 capture for "
                      "h_mlp still satisfied x1 == x + mixer_out, so the check is vacuous")
    good.append({"check": "CONTROL swapped x1<-h_mlp (must be rejected)",
                 "bit_exact": False, "ndiff": bad["ndiff"], "n": bad["n"], "control_fired": True})
    if not all(c["bit_exact"] for c in good[:2]):
        raise Refusal(f"residual-identity failed: {[c for c in good[:2] if not c['bit_exact']]}")
    return good


@contextlib.contextmanager
def default_dtype(dt):
    """Scope a `torch.set_default_dtype` change.

    Needed because the peer's self-initialisation should not allocate fp32 [256,1024,2048]
    scratch it is about to throw away, while the rotary's `inv_freq` MUST be computed in
    f32 in both views — it is formed by `torch.pow` at construction time, so building it
    under a bf16 default would round the frequencies and make the two views disagree for a
    reason that has nothing to do with the substitution."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ------------------------------------------------------------------------- evidence files
def evidence_path(out_dir: Path, name: str) -> Path:
    return out_dir / "evidence" / f"{name}.json"


def write_evidence(out_dir: Path, manifest: Path, name: str, obj: dict) -> Path:
    p = evidence_path(out_dir, name)
    claim_for_write(p, out_dir, manifest)
    obj = dict(obj)
    obj.setdefault("interpreter", {"executable": sys.executable,
                                   "version": sys.version.split()[0], "torch": torch.__version__})
    obj.setdefault("producing_command", COMMAND)
    obj["peak_rss_gib_at_write"] = round(peak_rss_gib(), 3)
    with open(p, "w") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return p


def read_evidence(out_dir: Path, name: str) -> dict:
    p = evidence_path(out_dir, name)
    if not p.exists():
        raise Refusal(f"missing-evidence: {p} does not exist. Run `--mode {name}` first; a "
                      "manifest must not be able to claim a witness that never ran.")
    return json.loads(p.read_text())


def env_block(threads: int) -> dict:
    return {"interpreter": {"executable": sys.executable, "version": sys.version.split()[0],
                            "torch": torch.__version__, "threads": threads},
            "producing_command": COMMAND,
            "peak_rss_gib": round(peak_rss_gib(), 3)}


# ------------------------------------------------------------------------------ the chain
def run_chain(blocks, tokens, seed, out_dir, manifest, *, threads=8, build="meta",
              perturb=None, break_substitution=None, break_block=None, oracles=True,
              tag="chain", subdir=None, resident=False):
    """Run the chain: one block resident, each block's input read from the previous block's
    output FILE. Returns the rows, the per-block cost table and the witnesses.

    `perturb=(block_name, tensor_name, row, scale)` scales one row of one weight tensor
    after loading — the D-25 guard-5 witness. `break_substitution=<param name>` leaves that
    parameter at zeros — the `substitution-is-load-bearing` control. Both are ordinary
    arguments of the SAME code path the pinned chain runs, which is the point: the
    perturbation witness tests the chaining, not a parallel copy of it.
    """
    R.set_threads(threads)
    g = R.load_geometry()
    gg = G.load_gdn_geometry()
    M, cfg_text, tver = hf()
    census = shard_census()
    cost = Cost()
    work_dir = out_dir if subdir is None else (out_dir / subdir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for b in blocks:
        if b.index not in census["layers_complete"]:
            raise Refusal(f"{b.name} is not complete on disk: layers_complete="
                          f"{census['layers_complete']}. The shard census is derived from "
                          "the committed index; there is nothing to load.")

    with default_dtype(torch.float32):
        rotary = M.Qwen3_5MoeTextRotaryEmbedding(config=cfg_text)
    ids = token_ids(tokens, seed, json.loads(R.CONFIG_PATH.read_text())["text_config"]["vocab_size"])
    emb_name = "model.language_model.embed_tokens.weight"
    emb_shard = R.shard_of(emb_name)
    cost.mark("embed_load_start")
    emb = R.load_tensor(R.SHARD_DIR / emb_shard, emb_name)
    cost.mark("embed_loaded")
    x = emb[ids].clone()
    # §Embedding provenance: the rows must come from the shard's own slice, so a second,
    # independent read of the same tensor must produce the same digest. A stand-in would not.
    emb2 = R.load_tensor(R.SHARD_DIR / emb_shard, emb_name)
    rows_sha = sha256_tensor(emb2[ids])
    del emb, emb2
    gc.collect()
    emb_prov = {"token_ids": ids.tolist(), "n_tokens": tokens, "seed": seed,
                "source_shard": emb_shard, "source_tensor": emb_name,
                "rows_sha256": rows_sha,
                "token_ids_sha256": hashlib.sha256(
                    ids.numpy().tobytes()).hexdigest(),
                "dtype": str(x.dtype), "shape": list(x.shape)}
    cost.mark("embed_sliced")

    def load_and_build(block):
        """Load one block's tensors, apply any perturbation, substitute them into a layer.
        `resident=True` calls this for every block BEFORE the forward loop, which is how the
        peer view gets all four blocks materialised at once (no meta device anywhere in it)."""
        tensors, sources, expected = load_layer_params(block)
        if perturb and perturb[0] == block.name:
            _bname, tname, row, scale = perturb
            t = tensors[tname]
            if not (t.dim() >= 1 and 0 <= row < t.shape[0]):
                raise Refusal(f"perturbation row {row} is outside {tname}'s {tuple(t.shape)}")
            with torch.no_grad():
                t[row] = t[row].to(torch.float32).mul(scale).to(t.dtype)
        layer, _subs, witness = build_layer(
            block, build, tensors, sources,
            break_substitution=(break_substitution if break_block in (None, block.name)
                                else None), perturb=perturb)
        return layer, tensors, sources, witness, expected

    prebuilt = {}
    if resident:
        for b in blocks:
            cost.mark(f"{b.name}.load_start")
            prebuilt[b.name] = load_and_build(b)
            cost.mark(f"{b.name}.substituted")

    rows, per_block, substitution, oracle, identities = [], [], [], [], []
    prev_out_path, prev_out_sha, prev_index = None, None, None
    for i, block in enumerate(blocks):
        btype = census["layer_types"][str(block.index)]
        rec = {"block": block.name, "index": block.index, "block_type": btype}
        # --- the input arrives FROM DISK, not from a variable in this loop
        if i == 0:
            x_in, x_from = x, dict(
                emb_prov, kind="embedding_slice",
                file=rel(R.SHARD_DIR / emb_shard),
                sha256=emb_prov["rows_sha256"])
        else:
            x_in = read_stage(prev_out_path, (tokens, g.hidden_size), torch.bfloat16)
            x_from = {"kind": "previous_block_output", "file": rel(prev_out_path),
                      "sha256": prev_out_sha, "predecessor_block": blocks[i - 1].name,
                      "predecessor_stage": "out"}
            if x_from["sha256"] != sha256_tensor(x_in):
                raise Refusal(f"{block.name}: the file we read back does not hash to the "
                              "digest the previous block recorded for it")
        cost.mark(f"{block.name}.load_start")
        if resident:
            layer, tensors, sources, witness, expected = prebuilt[block.name]
        else:
            layer, tensors, sources, witness, expected = load_and_build(block)
        cost.mark(f"{block.name}.substituted")
        if perturb and perturb[0] == block.name:
            rec["perturbed"] = {"tensor": perturb[1], "row": perturb[2], "scale": perturb[3]}
        sub = substitution_verdict(block, witness, expected)
        substitution.append(sub)
        if not sub["ok"]:
            raise Refusal(f"{block.name}: substituted={sub['substituted']}/{sub['expected']} "
                          "is not the full parameter set "
                          f"(digest_mismatch={sub['digest_mismatch']}, "
                          f"perturbation_did_not_reach_the_forward="
                          f"{sub['perturbation_did_not_reach_the_forward']})")
        if break_substitution and block.name == break_block \
                and sub["deliberately_broken"] != [break_substitution]:
            raise Refusal(f"{block.name}: the control asked to leave {break_substitution!r} "
                          f"at zeros and the census reports {sub['deliberately_broken']}")
        cap = Capture()
        cap.install(layer, btype)
        out = run_layer(layer, x_in.unsqueeze(0), rotary, cfg_text, tokens, btype)
        cost.mark(f"{block.name}.forward")
        got = {k: v.detach() for k, v in cap.stages.items()}
        missing = [s for s in STAGES if s not in got]
        if missing:
            raise Refusal(f"{block.name}: hooks produced no value for {missing}; the stage "
                          "list and the module wiring disagree, so nothing here is pinnable")
        extra = [k for k in got if k not in STAGES]
        if extra:
            raise Refusal(f"{block.name}: hooks captured unlisted stages {extra}")
        got["out"] = out.detach()
        if break_substitution:
            identities.append({"block": block.name, "control": break_substitution,
                               "out_sha256": sha256_tensor(got["out"])})
        else:
            identities += [{"block": block.name, **c} for c in residual_identity(got)]
        for stage in STAGES:
            t = got[stage].reshape(tokens, g.hidden_size).contiguous()
            name = f"{tag}.{block.name}.{stage}"
            rec_row = write_stage(t, work_dir / f"{block.name}.{stage}.bin", out_dir, manifest)
            rec_row.update({
                "name": name, "block": block.name, "stage": stage, "pass": tag,
                "derivation": STAGE_SOURCE[stage],
                "block_type": btype,
                "contiguous": (len(census["missing_layers"]) == 0),
                "chain_gap": (None if i == 0 else
                              {"from_index": prev_index, "to_index": block.index,
                               "layers_between_missing":
                                   [m for m in census["missing_layers"]
                                    if prev_index < m < block.index]}),
                "command": COMMAND,
                "interpreter": {"executable": sys.executable, "torch": torch.__version__},
            })
            if stage == "x":
                rec_row["input_from"] = x_from
            else:
                pred = next((r for r in rows if r.get("name") == f"{tag}.{block.name}.x"), None)
                if pred is None:
                    raise Refusal(f"{block.name}.{stage}: its predecessor "
                                  f"{tag}.{block.name}.x is not in the row set; a chained "
                                  "record may not name a predecessor that was never written")
                rec_row["input_from"] = {"kind": "same_block_stage", "block": block.name,
                                         "stage": "x", "file": pred["file"],
                                         "sha256": pred["sha256"]}
            rows.append(rec_row)
        prev_out_path = work_dir / f"{block.name}.out.bin"
        prev_out_sha = sha256_tensor(got["out"])
        prev_index = block.index
        if oracles:
            oracle += oracle_rows(block, btype, work_dir, tokens, tensors, sources, g, gg,
                                  cfg_text)
        cost.mark(f"{block.name}.oracled")
        cost.mark(f"{block.name}.written")
        per_block.append({"block": block.name, "block_type": btype,
                          "load_s": round(cost.since(f"{block.name}.load_start",
                                                     f"{block.name}.substituted"), 3),
                          "forward_s": round(cost.since(f"{block.name}.substituted",
                                                        f"{block.name}.forward"), 3),
                          "write_s": round(cost.since(f"{block.name}.oracled",
                                                      f"{block.name}.written"), 3),
                          "oracle_s": round(cost.since(f"{block.name}.forward",
                                                       f"{block.name}.oracled"), 3),
                          "rss_gib_after": round(current_rss_gib(), 3),
                          "bytes_written": sum(r["bytes"] for r in rows
                                               if r.get("block") == block.name)})
        del layer, tensors, sources, cap, got, out
        if resident:
            prebuilt.pop(block.name, None)
        gc.collect()
    return {"rows": rows, "per_block": per_block, "substitution": substitution,
            "oracle": oracle, "identities": identities, "census": census,
            "embedding": emb_prov, "cost": cost.rows(),
            "peak_rss_gib": round(peak_rss_gib(), 3),
            "blocks": [b.name for b in blocks], "tokens": tokens, "seed": seed,
            "build": build, "transformers": tver, "tag": tag}


# --------------------------------------------------------------------------------- checks
class Report:
    """Prints one line per check and refuses to call an empty run green.

    `pitfalls/40` §a check whose predicate is True, and §a check that can produce no output
    cannot fail: the verdict is computed from the population it parsed, the population is
    derived from the records rather than typed, and `finish()` compares the number of
    records parsed with the number printed and asserts at least one check actually ran."""

    def __init__(self, title: str):
        self.title = title
        self.rows, self.parsed, self.printed = [], 0, 0

    def emit(self, name: str, ok, detail: str = "", parsed: int = 1,
               word: str | None = None) -> bool:
        """One verdict, plus how many records it was computed over.

        `pitfalls/40` §a check that can produce no output cannot fail: a verdict over a
        population of zero is not a verdict, so `parsed <= 0` is turned into a RED here
        rather than trusted, and `finish()` re-checks it across the whole run."""
        population = int(parsed)
        if population <= 0:
            ok, detail = False, f"EMPTY POPULATION (parsed={parsed}); {detail}"
        verdict = ("PASS" if ok else "RED") if word is None else word
        print(f"{name:<46} {verdict:<9} {detail}")
        self.rows.append({"check": name, "verdict": verdict, "detail": detail,
                          "population": population})
        self.parsed += population
        self.printed += 1
        return bool(ok)

    def note(self, name: str, detail: str = "") -> None:
        """A line that asserts nothing — printed, counted, and EXCLUDED from the
        parsed-versus-printed comparison, which is only meaningful for verdict lines."""
        print(f"{name:<46} ----   {detail}")
        self.rows.append({"check": name, "verdict": "NOTE", "detail": detail})
        self.printed += 1

    def finish(self) -> list:
        bad = [r for r in self.rows if r["verdict"].startswith("RED")]
        verdicts = [r for r in self.rows
                    if r["verdict"].startswith("PASS") or r["verdict"].startswith("RED")]
        if self.printed == 0:
            raise Refusal(f"{self.title}: no check printed anything; an empty run cannot pass")
        if not verdicts:
            raise Refusal(f"{self.title}: {self.printed} lines printed but none of them is a "
                          "verdict; notes cannot carry a run")
        empty = [r["check"] for r in verdicts if r["population"] <= 0]
        if empty:
            raise Refusal(f"{self.title}: verdicts computed over an empty population: {empty}")
        print(f"# {self.title}: {self.printed} lines, {len(verdicts)} verdicts over "
              f"{self.parsed} records parsed, "
              f"{sum(1 for r in self.rows if r['verdict'].startswith('PASS'))} PASS, {len(bad)} RED")
        return self.rows


def pairwise_duplicates(digests_by_block: list) -> list:
    """Every pair of blocks whose input digests are equal. Derived, O(n^2), no typed list:
    the population is whatever blocks are in the record set."""
    out = []
    for i in range(len(digests_by_block)):
        for j in range(i + 1, len(digests_by_block)):
            (ba, da), (bb, db) = digests_by_block[i], digests_by_block[j]
            if da == db:
                out.append((ba, bb, da))
    return out


def check_inputs_not_collapsed(rows: list, rep: Report, where: str, expect: str,
                               selector: tuple) -> bool:
    """`inputs_not_collapsed` — the block inputs must be pairwise distinct by digest.

    Run against BOTH fixtures: PASS on the chain, RED on
    `checkpoints/dump-manifest-2026-10-26.json`. The old per-block set is the negative
    control, which makes this check non-vacuous on the chain and keeps the D-25 finding
    permanently in the output. The population is SELECTED from the records handed in
    (`selector=(key, value)`), never typed: the caller says "records whose `stage` is `x`"
    or "records whose `tensor_key` is `input`" and the count that comes back is the
    denominator printed with the verdict."
    """
    key, value = selector
    pairs = [(r.get("block") or r.get("name"), r["sha256"]) for r in rows
             if r.get(key) == value]
    if len(pairs) < 2:
        rep.emit(f"inputs_not_collapsed[{where}]", False,
                 f"only {len(pairs)} input records parsed from {len(rows)} records — a "
                 "distinctness check over fewer than two inputs is vacuous, so it reports RED")
        return False
    dups = pairwise_duplicates(pairs)
    ok = not dups
    got = "RED" if dups else "PASS"
    # The verdict column reports whether the CHECK behaved as expected, so the label says
    # so explicitly: a PASS on the negative control means it correctly REJECTED the old
    # per-block fixture, which is the finding D-25 recorded.
    if got == expect:
        fired = ("the existing per-block pins are one and the same tensor" if "existing" in where
                 else "the mutated copy has one and the same tensor in every slot")
        lead = (f"{expect} AS EXPECTED"
                + (f" — the negative control FIRED: {fired}" if expect == "RED" and dups else "")
                + ": ")
    else:
        lead = f"UNEXPECTED {got}, wanted {expect}: "
    detail = (lead + f"{len(pairs)} inputs, {len({d for _, d in pairs})} distinct digests"
              + (f", COLLAPSED: {dups[:4]}" if dups else ""))
    rep.emit(f"inputs_not_collapsed[{where}]", got == expect, detail, parsed=len(pairs),
             word=("PASS-RED" if (got == expect == "RED") else None))
    return ok


def check_chained_input_equals_previous_output(rows: list, out_dir: Path, rep: Report) -> None:
    """Bit-exact, verified by RE-HASHING THE FILES ON DISK, never by comparing tensors that
    are still in this process's memory. A chained fixture that only agrees in RAM proves
    nothing about what a later reader will find in the dump."""
    by_name = {r["name"]: r for r in rows}
    n = 0
    for r in rows:
        if r["stage"] != "x":
            continue
        src = r["input_from"]
        if src["kind"] != "previous_block_output":
            rep.note(f"chained_input[{r['block']}]",
                     f"input_from={src['kind']} ({src.get('source_tensor', src.get('file'))}); "
                     "no predecessor block to compare against")
            n += 1
            continue
        x_path = (REPO / r["file"]).resolve()
        p_path = (REPO / src["file"]).resolve()
        hx, hp = D.sha256_file(x_path), D.sha256_file(p_path)
        ok = hx == hp == src["sha256"] == r["sha256"]
        rep.emit(f"chained_input_equals_previous_output[{r['block']}]", ok,
                 f"{p_path.name} sha={hp[:16]}… == {x_path.name} sha={hx[:16]}… "
                 f"== manifest {src['sha256'][:16]}…")
        n += 1
    if n == 0:
        rep.emit("chained_input_equals_previous_output", False,
                 "no input records parsed; the check ran over an empty population")


def check_gap_is_declared(rows: list, census: dict, rep: Report) -> None:
    """The gap must be structural. Two separate predicates, each witnessed against a mutated
    copy of the records so neither can be vacuous.

    `gap_declared_for_gap` reddens if a gap exists and any record is missing
    `contiguous: false`; `no_contiguity_claim_across_gap` reddens if any record claims
    contiguity while missing layers sit between its predecessor and itself."""
    gap = bool(census["missing_layers"])
    if not gap:
        rep.emit("gap_declared_for_gap", False,
                 "census reports NO missing layers, so this check has nothing to witness; "
                 "it should not be running on a complete checkpoint")
        return

    def judge(rs) -> bool:
        return all(r.get("contiguous", True) is False for r in rs)

    ok = judge(rows)
    rep.emit("gap_declared_for_gap", ok,
             f"{len(census['missing_layers'])} layers missing ({census['missing_layers'][:4]}…"
             f"), {len(rows)} records, all carry contiguous=false: {ok}", parsed=len(rows))
    mutated = [dict(r) for r in rows]
    mutated[0].pop("contiguous", None)
    fired = not judge(mutated)
    rep.emit("CONTROL gap_declared_for_gap catches a stripped record", fired,
             f"removed `contiguous` from {mutated[0]['name']}; check must reject: "
             f"{not judge(mutated)}")

    def judge_claims(rs) -> bool:
        for r in rs:
            g = r.get("chain_gap")
            if g and g["layers_between_missing"] and r.get("contiguous") is not False:
                return False
        return True

    ok2 = judge_claims(rows)
    gapped = [r["name"] for r in rows if r.get("chain_gap")
              and r["chain_gap"]["layers_between_missing"]]
    rep.emit("no_contiguity_claim_across_gap", ok2,
             f"{len(gapped)} records straddle the gap ({gapped[:4]}), none claims contiguity",
             parsed=len(rows))
    mutated = [dict(r) for r in rows]
    if not gapped:
        rep.emit("CONTROL no_contiguity_claim catches a lying record", False,
                 "no record straddles the gap, so the predicate cannot be witnessed")
    else:
        for r in mutated:
            if r["name"] == gapped[0]:
                r["contiguous"] = True
        rep.emit("CONTROL no_contiguity_claim catches a lying record",
                 not judge_claims(mutated),
                 f"flipped `contiguous: true` on {gapped[0]}; check rejects: "
                 f"{not judge_claims(mutated)}")


def check_substitution(sub_rows: list, rep: Report) -> None:
    for s in sub_rows:
        rep.emit(f"substitution_complete[{s['block']}]", s["ok"],
                 f"substituted={s['substituted']}/{s['expected']} parameters, each hashed "
                 f"against its safetensors slice; mismatches={s['digest_mismatch']}",
                 parsed=s["expected"])
    if not sub_rows:
        rep.emit("substitution_complete", False, "no block reported a substitution census")


def check_oracle_rows(oracle: list, blocks: list, rep: Report) -> None:
    """Every block must have at least one stage re-derived by an implementation other than
    the dumper. The denominator is the block list, so a block that silently acquired no
    independent witness fails rather than going unmentioned."""
    for o in oracle:
        m = o["metrics"]
        if m is None:
            rep.note(f"independent_oracle[{o['block']}/{o['stage']}/{o['view']}]", o["line"])
            continue
        ok = m["one_minus_cos"] <= C.BUG_DETECTOR_ONE_MINUS_COS
        rep.emit(f"independent_oracle[{o['block']}/{o['stage']}/{o['view']}]", ok,
                 o["line"] + f"  bit_exact={o['bit_exact']}")
    witnessed = {o["block"] for o in oracle
                 if o["metrics"] is not None and o["view"].startswith("B/")}
    unproved = [b for b in blocks if b not in witnessed]
    rep.emit("independent_oracle_covers_every_block", not unproved,
             f"{len(witnessed)}/{len(blocks)} blocks re-derived by speccheck; unproved="
             f"{unproved}", parsed=len(blocks))


# ---------------------------------------------------------------------------------- modes
def parse_blocks(spec: str) -> list:
    blocks = [R.Block.parse(s) for s in spec.split(",") if s.strip()]
    if not blocks:
        raise Refusal(f"--blocks {spec!r} selected nothing; an empty chain cannot be pinned")
    if len({b.name for b in blocks}) != len(blocks):
        raise Refusal(f"--blocks {spec!r} repeats a block; a chain may not visit one twice")
    return blocks


def parse_perturb(spec: str) -> tuple:
    """`layer38:mlp.experts.down_proj:row=7` (optionally `:scale=1.5`)."""
    try:
        block, tensor, row = spec.split(":")
    except ValueError:
        raise Refusal(f"--perturb {spec!r} must look like layer38:mlp.experts.down_proj:row=7"
                      ) from None
    scale = 1.5
    if "=" in row:
        k, v = row.split("=", 1)
        if k != "row":
            raise Refusal(f"--perturb {spec!r}: only `row=` is supported, got {k!r}")
        row = int(v)
    else:
        row = int(row)
    if scale == 1.0:
        raise Refusal("perturbation-scale-is-one: scaling by 1.0 cannot move anything")
    return block, tensor, row, scale


def cost_table(per_block: list, peak: float) -> None:
    print(f"{'block':<9} {'type':<16} {'load s':>8} {'forward s':>10} {'write s':>8} "
          f"{'oracle s':>9} {'rss GiB':>8} {'bytes':>10}")
    for p in per_block:
        print(f"{p['block']:<9} {p['block_type']:<16} {p['load_s']:>8.2f} "
              f"{p['forward_s']:>10.2f} {p['write_s']:>8.2f} {p['oracle_s']:>9.2f} "
              f"{p['rss_gib_after']:>8.2f} {p['bytes_written']:>10,}")
    print(f"peak host RSS for this process: {peak:.2f} GiB  "
          f"(one block + embeddings; no device, no timing taken)")


def digest_map(rows: list, stage: str) -> dict:
    return {r["block"]: r["sha256"] for r in rows if r["stage"] == stage}


def mode_chain(args, out_dir: Path, manifest: Path) -> int:
    refuse_existing(out_dir, "chain-dump-exists")
    refuse_existing(manifest, "chain-manifest-exists")
    blocks = parse_blocks(args.blocks)
    res = run_chain(blocks, args.tokens, args.seed, out_dir, manifest,
                    threads=args.threads, tag="chain")
    write_evidence(out_dir, manifest, "chain", res)
    print(f"\n# chain: {len(res['rows'])} records over {res['blocks']} blocks "
          f"(T={args.tokens}, seed={args.seed})")
    cost_table(res["per_block"], res["peak_rss_gib"])
    c = res["census"]
    print(f"# shard census: shards {c['shards_present']}/{c['shards_total']}, tensors "
          f"{c['tensors_present']}/{c['tensors_total']}, layers complete "
          f"{c['layers_complete']}, missing_layers count={c['n_missing_layers']}")
    return 0


def mode_guards(args, out_dir: Path, manifest: Path) -> int:
    chain = read_evidence(out_dir, "chain")
    rep = Report("guards")
    rows = chain["rows"]
    census = chain["census"]

    check_inputs_not_collapsed(rows, rep, "chain", "PASS", ("stage", "x"))
    collapse = [dict(r) for r in rows if r["stage"] == "x"]
    if len(collapse) >= 2:
        fake = [dict(r) for r in rows]
        first = collapse[0]["sha256"]
        for r in fake:
            if r["stage"] == "x":
                r["sha256"] = first
        check_inputs_not_collapsed(fake, rep, "CONTROL all-chain-inputs-collapsed", "RED",
                                   ("stage", "x"))
    old = json.loads(OLD_MOE_MANIFEST.read_text())
    rep.note("negative_control_source",
             f"{rel(OLD_MOE_MANIFEST)} — the sanctioned per-block pins, read "
             "only, used as the negative control")
    check_inputs_not_collapsed(old["dumps"], rep, "existing per-block manifest", "RED",
                               ("tensor_key", "input"))

    check_chained_input_equals_previous_output(rows, out_dir, rep)
    check_gap_is_declared(rows, census, rep)
    check_substitution(chain["substitution"], rep)
    check_oracle_rows(chain["oracle"], chain["blocks"], rep)

    wired = [i for i in chain["identities"] if not i.get("control_fired")]
    for i in wired:
        rep.emit(f"residual_identity[{i['block']}/{i['check']}]",
                 bool(i["bit_exact"]), f"ndiff={i['ndiff']}/{i['n']} (must be 0)",
                 parsed=i["n"])
    controls = [i for i in chain["identities"] if i.get("control_fired")]
    for i in controls:
        rep.emit(f"CONTROL residual_identity catches a swapped capture[{i['block']}]",
                 not i["bit_exact"], f"swapped x1<-h_mlp moved {i['ndiff']}/{i['n']} values")

    emb = chain["embedding"]
    x0 = next(r for r in rows if r["block"] == chain["blocks"][0] and r["stage"] == "x")
    on_disk = D.sha256_file(REPO / x0["file"])
    rep.emit("embedding_rows_come_from_the_shard",
             emb["rows_sha256"] == on_disk == x0["sha256"],
             f"rows_sha256={emb['rows_sha256'][:16]}… (second independent read of "
             f"{emb['source_tensor']}) == file {on_disk[:16]}… == record {x0['sha256'][:16]}…; "
             f"token_ids_sha256={emb['token_ids_sha256'][:16]}… n={emb['n_tokens']}")
    rows_out = write_evidence(out_dir, manifest, "guards", {"checks": rep.finish()})
    print(f"# evidence: {rel(rows_out)}")
    bad = [r for r in rep.rows if r["verdict"] == "RED"]
    return 1 if bad else 0


def mode_perturb(args, out_dir: Path, manifest: Path) -> int:
    chain = read_evidence(out_dir, "chain")
    perturb = parse_perturb(args.perturb)
    tag = "perturb-" + args.perturb.replace(":", "_").replace("=", "")
    rep = Report("perturbation witness")
    res = run_chain(parse_blocks(args.blocks), args.tokens, args.seed, out_dir, manifest,
                    threads=args.threads, tag="perturb", perturb=perturb,
                    subdir=tag, oracles=False)
    clean_x = digest_map(chain["rows"], "x")
    clean_out = digest_map(chain["rows"], "out")
    new_x = digest_map(res["rows"], "x")
    new_out = digest_map(res["rows"], "out")
    order = chain["blocks"]
    try:
        at = order.index(perturb[0])
    except ValueError:
        raise Refusal(f"--perturb names {perturb[0]}, which is not in the chain {order}") from None
    rep.note("perturbation", f"{perturb[0]}.{perturb[1]} row {perturb[2]} x {perturb[3]}")
    # An input moves only DOWNSTREAM of the perturbed block (that block's own input was read
    # from disk before its weights were touched); an output moves from the perturbed block
    # onwards, inclusive. Getting this table the other way round is how a chaining bug is
    # blessed rather than caught.
    for i, b in enumerate(order):
        want_in, want_out = i > at, i >= at
        moved_in = clean_x[b] != new_x[b]
        rep.emit(f"perturbation_input[{'downstream' if want_in else 'upstream'}:{b}]",
                 moved_in == want_in,
                 f"input {'CHANGED' if moved_in else 'unchanged'}, expected "
                 f"{'change' if want_in else 'no change'}: {clean_x[b][:12]}… -> "
                 f"{new_x[b][:12]}…")
        moved_out = clean_out[b] != new_out[b]
        rep.emit(f"perturbation_output[{'from_here' if want_out else 'upstream'}:{b}]",
                 moved_out == want_out,
                 f"output {'changed' if moved_out else 'unchanged'}, expected "
                 f"{'change' if want_out else 'no change'}: {clean_out[b][:12]}… -> "
                 f"{new_out[b][:12]}…")
    if all(clean_x[b] == new_x[b] for b in order):
        rep.emit("perturbation_moved_anything", False,
                 "NO block input changed — the chain is not chained, or the perturbation was "
                 "a no-op. This is a real bug and is reported as one.")
    cost_table(res["per_block"], res["peak_rss_gib"])
    p = write_evidence(out_dir, manifest, "perturb",
                       {"spec": {"block": perturb[0], "tensor": perturb[1],
                                 "row": perturb[2], "scale": perturb[3]},
                        "clean_input_digests": clean_x, "perturbed_input_digests": new_x,
                        "clean_output_digests": clean_out, "perturbed_output_digests": new_out,
                        "per_block": res["per_block"], "checks": rep.finish()})
    print(f"# evidence: {rel(p)}")
    return 1 if any(r["verdict"] == "RED" for r in rep.rows) else 0


def mode_peer(args, out_dir: Path, manifest: Path) -> int:
    """View C: the same layers, built the ordinary way — real materialised parameters, all
    four resident, no meta device, no substitution. A peer, not a mirror."""
    chain = read_evidence(out_dir, "chain")
    rep = Report("peer view — installed transformers, materialised parameters")
    res = run_chain(parse_blocks(args.blocks), args.tokens, args.seed, out_dir, manifest,
                    threads=args.threads, tag="peer", build="materialised", resident=True,
                    oracles=False, subdir="peer-hf-materialised")
    for r in res["rows"]:
        want = next((c for c in chain["rows"] if c["block"] == r["block"]
                     and c["stage"] == r["stage"]), None)
        if want is None:
            rep.emit(f"peer[{r['block']}/{r['stage']}]", False,
                     "the pinned chain has no such record to compare against")
            continue
        rep.emit(f"peer[{r['block']}/{r['stage']}]", want["sha256"] == r["sha256"],
                 f"materialised={r['sha256'][:16]}… vs meta-substituted={want['sha256'][:16]}…",
                 parsed=1)
    cost_table(res["per_block"], res["peak_rss_gib"])
    p = write_evidence(out_dir, manifest, "peer",
                       {"rows": res["rows"], "per_block": res["per_block"],
                        "peak_rss_gib": res["peak_rss_gib"], "checks": rep.finish()})
    print(f"# evidence: {rel(p)}")
    return 1 if any(r["verdict"] == "RED" for r in rep.rows) else 0


def mode_control(args, out_dir: Path, manifest: Path) -> int:
    """`substitution-is-load-bearing`: leave one parameter at ZEROS and show the block's
    output moves. If it does not, the forward was never reading that tensor and every
    witness above it is decoration."""
    chain = read_evidence(out_dir, "chain")
    block, param = args.control.split(":", 1)
    tag = "control-broken-substitution"
    rep = Report("control — substitution is load-bearing")
    res = run_chain(parse_blocks(args.blocks), args.tokens, args.seed, out_dir, manifest,
                    threads=args.threads, tag="control", break_substitution=param,
                    break_block=block, subdir=tag, oracles=False)
    clean = digest_map(chain["rows"], "out")
    broken = digest_map(res["rows"], "out")
    order = chain["blocks"]
    at = order.index(block)
    for i, b in enumerate(order):
        moved = clean[b] != broken[b]
        if b == block:
            rep.emit(f"broken_substitution_moves_output[{b}]", moved,
                     "output CHANGED, the substitution is load-bearing" if moved else
                     "output UNCHANGED — the forward never read this tensor, so every witness "
                     "above it is decoration", parsed=1)
        elif i < at:
            rep.emit(f"broken_substitution_is_local[{b}]", not moved,
                     f"upstream output {'unchanged' if not moved else 'CHANGED — the break leaked upstream'}",
                     parsed=1)
        else:
            rep.emit(f"broken_substitution_propagates[{b}]", moved,
                     f"downstream output {'changed, as it must' if moved else 'UNCHANGED — the chain is not carrying the break'}",
                     parsed=1)
    p = write_evidence(out_dir, manifest, "control",
                       {"block": block, "parameter": param,
                        "clean_output_digests": clean, "broken_output_digests": broken,
                        "checks": rep.finish()})
    print(f"# evidence: {rel(p)}")
    return 1 if any(r["verdict"] == "RED" for r in rep.rows) else 0


# ------------------------------------------------------------- the live paths stay frozen
def live_path_mtimes() -> dict:
    """mtime of every file under the live dump dir plus the live manifest, keyed by path.

    Snapshot before the run, compare after: "never opened for writing" becomes an
    observation about mtimes rather than a promise about code paths."""
    out = {}
    for p in LIVE_PATHS:
        q = (REPO / p)
        if q.is_file():
            out[str(q)] = q.stat().st_mtime_ns
        elif q.is_dir():
            for f in sorted(q.iterdir()):
                if f.is_file():
                    out[str(f)] = f.stat().st_mtime_ns
    return out


def assert_live_paths_untouched(before: dict, tag: str) -> None:
    after = live_path_mtimes()
    changed = sorted(k for k in set(before) | set(after)
                     if before.get(k) != after.get(k))
    if changed:
        raise Refusal(f"live-path-mutated[{tag}]: {changed[:5]} changed under this run. "
                      f"dump_chain.py may read {LIVE_PATHS} and must never write them.")


def mode_manifest(args, out_dir: Path, manifest: Path) -> int:
    chain = read_evidence(out_dir, "chain")
    guards = read_evidence(out_dir, "guards")
    perturb = read_evidence(out_dir, "perturb")
    peer = read_evidence(out_dir, "peer")
    control = read_evidence(out_dir, "control")
    refuse_existing(manifest, "chain-manifest-exists")
    blocks = chain["blocks"]
    census = chain["census"]
    expected = len(blocks) * len(STAGES)
    rows = chain["rows"]
    if len(rows) != expected:
        raise Refusal(f"wrote {len(rows)} records, expected {expected} "
                      f"({len(blocks)} blocks x {len(STAGES)} stages)")
    for r in rows:
        if "input_from" not in r or "file" not in r["input_from"] or "sha256" not in r["input_from"]:
            raise Refusal(f"{r['name']}: a chained record without input_from.file/sha256 is "
                          "not auditable, so it may not be pinned")
        if "contiguous" not in r:
            raise Refusal(f"{r['name']}: no `contiguous` field; a gappy chain must say so on "
                          "every record")
    gapped = any(r.get("chain_gap") and r["chain_gap"]["layers_between_missing"] for r in rows)
    manifest_body = {
        "manifest_version": 1,
        "kind": "chained-forward",
        "date": "2026-10-27",
        "producing_command": COMMAND,
        "interpreter": chain.get("interpreter"),
        "transformers": chain["transformers"],
        "producer": "tools/dump_chain.py (imported machinery: dump_reference.write_tensor, "
                    "dump_reference_gdn.refuse_live_paths, moe_ref, gdn_ref)",
        "model": {"config": rel(R.CONFIG_PATH),
                  "geometry": {"hidden_size": json.loads(R.CONFIG_PATH.read_text())
                               ["text_config"]["hidden_size"],
                               "num_hidden_layers": R.load_geometry().num_hidden_layers,
                               "num_experts": R.load_geometry().num_experts,
                               "top_k": R.load_geometry().top_k}},
        "block_set": {
            "blocks": blocks, "block_indices": [b.index for b in parse_blocks(
                ",".join(blocks))],
            "contiguous": False if census["missing_layers"] else True,
            "why_not_contiguous": (
                "layers 2..37 are listed in "
                "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json but their shards were "
                "never acquired (README.md §Download, D-03's sampled audit). The chain is "
                "therefore layer0 -> layer1 -> layer38 -> layer39, and a record read as "
                "'the activation after layer 37' would be wrong: layer38.input is layer1's "
                "output, not the model's post-layer-37 state."),
            "missing_layers": census["missing_layers"],
            "n_missing_layers": census["n_missing_layers"],
            "shard_census": census,
        },
        "input": dict(chain["embedding"], kind="embedding_slice",
                      method="torch.randint over the first 100k vocabulary rows from a private "
                             "torch.Generator('cpu').manual_seed(seed); structural ids "
                             "(config bos/eos, 248046, 248068, 248069) are excluded by "
                             "assertion. The rows are the real embed_tokens slice, sha-witnessed "
                             "against a second independent read of the shard."),
        "stages": STAGES, "stage_source": STAGE_SOURCE,
        "reading_rules": {
            "payload": "raw little-endian element bytes, no header; bytes == elements x "
                       "itemsize, and `--mode verify` refuses a record whose size cannot hold "
                       "its own declared shape (pitfalls/30 §Read a dump's dtype off its file "
                       "size). Same contract as checkpoints/dump-manifest-2026-10-26.json.",
            "bf16": "two bytes per element, bit image — read with a uint16 view then bitcast, "
                    "never cast the uint16 to a float (pitfalls/30)",
            "chaining": "each record's `input_from` names the predecessor FILE and its sha256; "
                        "the chain is auditable from this manifest alone, not assumed",
            "contiguous": "`contiguous: false` on a record means NO record in this manifest may "
                          "be read as a true model activation while any layer is missing; it is "
                          "a global property, deliberately the same on every record so no "
                          "reader can pick a row and assume it is contiguous. The per-hop truth "
                          "is `chain_gap.layers_between_missing`: [] means the predecessor is "
                          "the immediately preceding layer (layer0->layer1, layer38->layer39), "
                          "a list means the predecessor is NOT the previous layer and the "
                          "activation is missing those layers' transformations.",
            "authority": "these are reference activations from the installed transformers "
                         "reference under the acquired bf16 weights, not engine output. They "
                         "license nothing by themselves: a band over a chained row needs its "
                         "own floor probe (D-25 guard 7).",
        },
        "dumps": rows,
        "witnesses": {
            "substitution_census": chain["substitution"],
            "independent_oracle": chain["oracle"],
            "residual_identity": chain["identities"],
            "guards": guards["checks"],
            "perturbation": {k: perturb[k] for k in
                             ("spec", "clean_input_digests", "perturbed_input_digests",
                              "clean_output_digests", "perturbed_output_digests", "checks")},
            "peer_view": {"checks": peer["checks"], "peak_rss_gib": peer["peak_rss_gib"],
                          "per_block": peer["per_block"]},
            "substitution_control": {k: control[k] for k in
                                     ("block", "parameter", "clean_output_digests",
                                      "broken_output_digests", "checks")},
        },
        "cost": {"per_block": chain["per_block"], "peak_rss_gib": chain["peak_rss_gib"],
                 "phases": chain["cost"],
                 "note": "host wall-clock and host RSS only; no device clock is read anywhere "
                         "in this tool (no timing taken)"},
        "counts": {"dumps": len(rows), "expected": expected, "blocks": len(blocks),
                   "stages_per_block": len(STAGES),
                   "witness_lines": len(guards["checks"]) + len(peer["checks"])
                                    + len(perturb["checks"]) + len(control["checks"])},
    }
    if manifest_body["counts"]["dumps"] != manifest_body["counts"]["expected"]:
        raise Refusal("count mismatch before the manifest could be written")
    claim_for_write(manifest, out_dir, manifest)
    with open(manifest, "w") as fh:
        fh.write(json.dumps(manifest_body, indent=1, sort_keys=False) + "\n")
    print(f"# manifest: {rel(manifest)} "
          f"({manifest.stat().st_size:,} B), {len(rows)} chained records, "
          f"{manifest_body['counts']['witness_lines']} witness lines")
    return mode_verify(args, out_dir, manifest)


def mode_verify(args, out_dir: Path, manifest: Path) -> int:
    if not manifest.exists():
        raise Refusal(f"verify-refused: {manifest} does not exist. This tool does not verify a "
                      "chain it has not written; there is no sample-and-hope mode.")
    m = json.loads(manifest.read_text())
    rep = Report("verify")
    bad, checked = [], 0
    for rec in m["dumps"]:
        p = Path(rec["file"])
        p = p if p.is_absolute() else REPO / p
        if not p.exists():
            bad.append(f"MISSING {rec['file']}")
            continue
        got = D.sha256_file(p)
        size = p.stat().st_size
        if got != rec["sha256"]:
            bad.append(f"HASH {rec['file']}")
        elif size != rec["bytes"]:
            bad.append(f"SIZE {rec['file']} {size} != {rec['bytes']}")
        elif size != rec["elements"] * D.dtype_size(rec["dtype"]):
            bad.append(f"BYTES-vs-SHAPE {rec['file']}: {size} B cannot hold "
                       f"{rec['elements']} x {rec['dtype']}")
        else:
            checked += 1
    rep.emit("every_file_hashes_and_sizes", not bad,
             f"{checked}/{len(m['dumps'])} verified, {len(bad)} bad: {bad[:3]}",
             parsed=len(m["dumps"]))
    rep.emit("verified_count_equals_denominator",
             checked == len(m["dumps"]) == m["counts"]["expected"],
             f"checked={checked} records={len(m['dumps'])} expected={m['counts']['expected']} "
             "(pitfalls/40)", parsed=len(m["dumps"]))
    # A chained fixture is only auditable if every row says where its bytes came from. This
    # is asserted at write time too; asserting it here as well is what stops a hand-edited
    # manifest from silently losing its `input_from`.
    unauditable = [r.get("name", "?") for r in m["dumps"]
                   if not (isinstance(r.get("input_from"), dict)
                           and r["input_from"].get("file") and r["input_from"].get("sha256"))
                   or not r.get("derivation") or not r.get("command")]
    rep.emit("every_record_names_its_predecessor_and_producer", not unauditable,
             f"{len(m['dumps']) - len(unauditable)}/{len(m['dumps'])} records carry "
             f"input_from.file+sha256, a hook derivation and the producing command; "
             f"missing: {unauditable[:4]}", parsed=len(m["dumps"]))
    check_inputs_not_collapsed(m["dumps"], rep, "chain(manifest)", "PASS", ("stage", "x"))
    check_chained_input_equals_previous_output(m["dumps"], out_dir, rep)
    check_gap_is_declared(m["dumps"], m["block_set"]["shard_census"], rep)
    rep.finish()
    ok = all(r["verdict"] == "PASS" for r in rep.rows if r["verdict"] in ("PASS", "RED"))
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    return 0 if ok else 1


COMMAND = "not set"


def main(argv=None) -> int:
    global COMMAND
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="chain",
                    choices=["chain", "guards", "perturb", "peer", "control", "manifest",
                             "verify"])
    ap.add_argument("--blocks", default=DEFAULT_BLOCKS)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--perturb", default=DEFAULT_PERTURB)
    ap.add_argument("--control", default="layer38:mlp.experts.down_proj")
    ap.add_argument("--print-stages", action="store_true",
                    help="print the stage table and exit (no weights are read)")
    args = ap.parse_args(argv)
    COMMAND = " ".join([sys.executable] + [str(Path(__file__).resolve())]
                       + [a for a in (argv or sys.argv[1:])])

    out_dir = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    manifest = (REPO / args.manifest) if not Path(args.manifest).is_absolute() \
        else Path(args.manifest)
    if args.print_stages:
        for s in STAGES:
            print(f"{s:<11} {STAGE_SOURCE[s]}")
        return 0
    # Every mode: the live per-block fixture must come out of this run exactly as it went in.
    before = live_path_mtimes()
    if not before:
        raise Refusal(f"live-path-audit-vacuous: {LIVE_PATHS} yielded no files, so the "
                      "untouched-check could not fail")
    try:
        rc = {"chain": mode_chain, "guards": mode_guards, "perturb": mode_perturb,
              "peer": mode_peer, "control": mode_control, "manifest": mode_manifest,
              "verify": mode_verify}[args.mode](args, out_dir, manifest)
    except Refusal as r:
        print(f"REFUSED: {r}")
        rc = 2
    finally:
        assert_live_paths_untouched(before, args.mode)
    print(f"# mode={args.mode} rc={rc} peak_rss={peak_rss_gib():.2f} GiB  "
          f"no timing taken (device)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
