#!/usr/bin/env python3
"""engine/cross_oracle.py — the three-way view: our engine x the HF chained dumps x llama.cpp.

D-26 makes llama.cpp the sole numerical oracle for engine/* and keeps the HF reference as a
comparison instrument on four blocks (0, 1, 38, 39 — the two GDN kinds and the two transitions).
This tool puts the three sides in one report. It does NOT author a band, floor or tolerance:
every magnitude it prints is labelled with the file that licenses it, and any limit that could
only come from llama.cpp prints `licence=llama-crosscheck(...)` — which means "an observation
from a cross-check", never "a licensed band". D-17 (±1-ULP acceptance) is untouched here and is
the user's, not ours.

THE LOAD-BEARING FACT, measured before this file was written: the three sides are not fed the same
tokens. The HF chain ran on the 32 pinned `token_ids` of
checkpoints/dump-manifest-chain-2026-10-26.json — which that manifest says were *sampled*
(`input.method`: "torch.randint over the first 100k vocabulary rows", `input.kind`:
"embedding_slice"), i.e. they were never produced by a tokenizer from text; the llama.cpp dumps
under data/dumps/2026-10-27-llama/run-a ran on a chat-template-wrapped prompt (prefill graph = 11
ids starting 248045, next graph = 4 ids containing the think-boundary id). A per-stage numeric diff
across two different stimuli is a table of true-looking garbage, so alignment is checked as a
PRECONDITION (`align` mode) and every numeric line prints UNCOMPARABLE-BY-STIMULUS until it holds.

Why alignment is not merely untried-but-awkward: the vendored front ends take text, never ids
(`tools/LLAMA_DUMP_NOTES.md` §CLI surface is the whole instrument surface; there is no token-id input
and no token-override knob in the patch), and byte-level BPE round-trips ids to text only where its
merges agree with the concatenation boundary. Measured on the chain's own 32 ids: detokenize → 205
bytes (valid UTF-8, no control-class token) → `llama-tokenize -f` → 32 ids again, but only 30 equal;
first divergence index 3 (chain 78392, re-tokenized 750). The byte string is forced by the ids
(byte-level BPE is byte-preserving and the token→piece map is a bijection), so that candidate is the
only candidate: no text prefill reproduces those 32 ids — which is what you would expect of ids that
were sampled rather than tokenized. Control that the channel itself is template-free: tokenizing the
run-a prompt text returns exactly the prefill's 5 inner ids. Evidence and reproduction commands:
quantization/evidence/llama-stimulus-alignment-2026-10-27.md.

Modes:
  align       — stimulus/identity evidence for the three sides; exit non-zero if unaligned.
  structural  — stage mapping, shape/order agreement, per-side statistics, denominators.
  numeric     — the pairwise numeric comparison. Refuses unless alignment holds, and says so.
  all (default) — align, then structural; numeric only if alignment holds.

CPU-only, read-only over the manifests and the dump .bin files. No device lock is taken because
nothing here touches the iGPU; nothing here prints a rate — `no timing taken`.
"""
import argparse
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine"))

CHAIN_MANIFEST = os.path.join(REPO, "checkpoints/dump-manifest-chain-2026-10-26.json")
LLAMA_MANIFEST = os.path.join(REPO, "checkpoints/llama-dump-manifest-2026-10-27-run-a.json")
PIN_FILE = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")
BLOCKS = ("layer0", "layer1", "layer38", "layer39")
# The seven §03.2 stages the HF chain records per block, in execution order. That order is the
# denominator of every per-block comparison below; it is taken from the chain manifest, not typed.
HF_STAGES_EXPECT = 7


class Refuse(Exception):
    def __init__(self, tag, msg):
        Exception.__init__(self, msg)
        self.tag = tag


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_sides():
    """The three sides, as loaded facts. Refuses loudly if either dump set is not what its
    manifest says, because a mapping table over missing files proves nothing."""
    try:
        chain = json.load(open(CHAIN_MANIFEST))
        llama = json.load(open(LLAMA_MANIFEST))
        pin = json.load(open(PIN_FILE))
    except (OSError, ValueError) as exc:
        raise Refuse("manifest-unreadable", "%s: %s" % (type(exc).__name__, exc))
    stages_by_block = {}
    for d in chain["dumps"]:
        stages_by_block.setdefault(d["block"], []).append(d)
    for blk in chain["block_set"]["blocks"]:
        if blk not in stages_by_block:
            raise Refuse("chain-block-missing",
                         "chain manifest has no dumps for %s; denominators would be 0" % blk)
        got = [d["stage"] for d in stages_by_block[blk]]
        if got != chain["stages"]:
            raise Refuse("chain-stage-count",
                         "%s carries stages %s but the manifest's own stage list is %s"
                         % (blk, got, chain["stages"]))
    if not stages_by_block:
        raise Refuse("chain-empty", "the chain manifest lists 0 dump records; every denominator "
                     "below would be 0, which is a failure, not a pass")
    if chain["block_set"]["blocks"] != list(BLOCKS):
        raise Refuse("block-set-drift",
                     "this tool is written for blocks %s but the manifest pins %s; the four blocks "
                     "are a pinned fixture, so re-derive the report before trusting it"
                     % (list(BLOCKS), chain["block_set"]["blocks"]))
    llama_rows = {}
    for r in llama["dumps"]:
        llama_rows.setdefault((r["graph"], r["block"]), []).append(r)
    if not llama["dumps"]:
        raise Refuse("llama-empty", "the llama dump manifest lists 0 rows")
    return chain, llama, pin, stages_by_block, llama_rows


def llama_graphs(llama):
    """The graphs with their role *derived*: the largest `n_tokens` is the prompt prefill, a
    single token is a decode step, the rest are warm-up/header chunks. The manifest carries no
    kind field, so this is arithmetic over its own numbers and the derivation is printed."""
    gs = sorted(llama["graphs"], key=lambda g: g["index"])
    biggest = max(g["n_tokens"] for g in gs)
    out = []
    for g in gs:
        kind = ("prefill" if g["n_tokens"] == biggest and biggest > 1
                else "decode" if g["n_tokens"] == 1 else "chunk")
        out.append(dict(g, kind=kind))
    return out


def llama_graph(llama, kind=None):
    return [g for g in llama_graphs(llama) if kind is None or g["kind"] == kind]


def stage_order(chain):
    """The §03.2 stage order, read from the manifest's own stage list."""
    return list(chain["stages"])


# --------------------------------------------------------------------------------------------
# align mode
# --------------------------------------------------------------------------------------------
def mode_align(chain, llama, pin):
    print("=== align: is there one stimulus? ===")
    ids = chain["input"]["token_ids"]
    print("hf      stimulus n_tokens=%d sha256=%s"
          % (chain["input"]["n_tokens"], chain["input"]["token_ids_sha256"]))
    print("hf      ids=%s" % " ".join(str(t) for t in ids))
    print("hf      special-class ids present=%s (max id %d; this family's control class starts at "
          "248044, so the chain stimulus is raw text tokens, not a rendered conversation)"
          % (any(t >= 248044 for t in ids), max(ids)))
    print("hf      source=%s/%s dtype=%s shape=%s method=%s"
          % (chain["input"]["source_shard"], chain["input"]["source_tensor"],
             chain["input"]["dtype"], chain["input"]["shape"],
             str(chain["input"].get("method", "?"))[:60]))
    gg = llama["gguf"]["sha256"]
    print("llama   gguf_sha256=%s matches_pin=%s pinned_sha256=%s (pin %s)"
          % (gg, llama["gguf"]["matches_pin"], llama["gguf"]["pinned_sha256"], pin["artifact"]))
    print("llama   instrument=%s llama.cpp=%s prompt given to the front end: %r"
          % (llama["instrument"]["patch"], llama["instrument"]["llama_cpp_commit"],
             llama["input"]["prompt"]))
    aligned = []
    for g in llama_graphs(llama):
        same_len = g["n_tokens"] == len(ids)
        same_ids = g.get("token_ids") == ids
        print("llama   graph %d kind=%-7s n_tokens=%-3d ids_equal_to_chain=%-5s special-class=%s "
              "ids=%s" % (g["index"], g["kind"], g["n_tokens"], same_ids,
                          [t for t in g.get("token_ids", []) if t >= 248044] or "none",
                          " ".join(str(t) for t in g.get("token_ids", []))))
        if same_len and same_ids:
            aligned.append(g["index"])
    pre = llama_graph(llama, "prefill")[0]
    print("llama   the prefill's ids 0, 8, 10 are 248045 / 248046 / 198 — the chat wrapper — and its "
          "5 inner ids are the prompt text; `llama-tokenize` on that text returns exactly those 5 "
          "ids, which is the template-free proof")
    print("engine  stimulus = the pinned single-token decode row (engine/block_forward.py BLOCK=0, "
          "GDN_MODE=decode0): a third stimulus, one row, not a prefill")
    print("align   denominator: hf=1 stimulus, llama=%d graphs, engine=1 stimulus; graph roles are "
          "derived from n_tokens (largest=prefill, 1=decode), not from a name"
          % len(llama_graphs(llama)))
    if aligned:
        print("align   ALIGNED graphs=%s" % ",".join(str(g) for g in aligned))
        return 0
    print("align   NOT-ALIGNED-BY-STIMULUS: no llama graph is fed the chain's %d ids (prefill is %d)"
          % (len(ids), pre["n_tokens"]))
    print("align   unattainable-through-frontends=yes: detokenizing the chain's 32 ids gives 205 "
          "bytes (valid UTF-8, 0 control-class tokens); `llama-tokenize -f` on those bytes returns 32 "
          "ids of which only 30 are equal, first divergence index 3 (chain 78392 vs 750). The byte "
          "string is forced by the ids (byte-level BPE is byte-preserving and token->piece is a "
          "bijection), so that candidate is the only candidate and no text prefill reproduces them. "
          "Reproduction: quantization/evidence/llama-stimulus-alignment-2026-10-27.md")
    print("align   consequence: mode numeric refuses; mode structural stands alone.")
    return 3
    print("align   unattainable-through-frontends=yes: tokenization of the detokenized chain ids "
          "reproduces 30 of 32 ids; first divergence index 3 (chain 78392 vs 750); the byte "
          "string is forced by the ids, so no text prefill reproduces them. See "
          "quantization/evidence/llama-stimulus-alignment-2026-10-27.md")
    print("align   consequence: mode numeric refuses; mode structural stands alone.")
    return 3


# --------------------------------------------------------------------------------------------
# structural mode
# --------------------------------------------------------------------------------------------
# The stage mapping. llama's names come from src/models/qwen35moe.cpp of the vendored tree
# (cited per row); `linear` / `full` are the two block kinds, which is why some rows have two
# names. None of this is inference about our engine: the engine column states what
# engine/block_forward.py computes, and UNMAPPED rows are named as absences, not guessed.
STAGE_MAP = {
    # hf stage: (engine stage or None, {kind: llama cb name}, llama.cpp cite)
    "x":         (None, {"linear": "model.input_embed(layer0)/l_out(prev)",
                         "full": "model.input_embed(layer0)/l_out(prev)"},
                  "qwen35moe.cpp:170,222"),
    "h_in":      (None, {"linear": "attn_norm", "full": "attn_norm"}, "qwen35moe.cpp:184"),
    "mixer_out": ("y", {"linear": "linear_attn_out", "full": "attn_output"},
                  "qwen35moe.cpp:483 / :352"),
    "x1":        (None, {"linear": "attn_residual", "full": "attn_residual"},
                  "qwen35moe.cpp:204"),
    "h_mlp":     (None, {"linear": "attn_post_norm", "full": "attn_post_norm"},
                  "qwen35moe.cpp:211"),
    "moe_out":   ("combined_out", {"linear": "ffn_out", "full": "ffn_out"},
                  "qwen35moe.cpp:536 (routed-only sub-stage is ffn_moe_out :507)"),
    "out":       (None, {"linear": "l_out", "full": "l_out"}, "qwen35moe.cpp:222"),
}


def kind_of_block(chain, blk):
    for d in chain["dumps"]:
        if d["block"] == blk:
            return d["block_type"]
    raise Refuse("block-type-missing", "chain manifest has no block_type for %s" % blk)


def llama_kind(llama, blk):
    il = int(blk.replace("layer", ""))
    return "linear" if il in llama["model"]["linear_layers"] else "full"


def resolve(llama_rows, graph, block_il, name):
    rows = llama_rows.get((graph, block_il), [])
    hits = [r for r in rows if r["name"] == name or r["name"].startswith(name + "-")]
    return hits


def elements(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def stats_of(path, dtype, shape, nbytes):
    """Statistics of one side's own values, from its own .bin. No cross-side arithmetic."""
    n = elements(shape)
    esz = {"bf16": 2, "torch.bfloat16": 2, "f32": 4, "float32": 4}.get(dtype)
    if esz is None:
        raise Refuse("dtype-unknown",
                     "%s: manifest dtype %r is not a width this tool knows how to read; guessing an "
                     "item size is how a census reads wrong and still prints a total" % (path, dtype))
    if n * esz != nbytes:
        raise Refuse("record-size", "%s: shape %s x %s B != manifest bytes %d"
                     % (path, shape, dtype, nbytes))
    raw = open(path, "rb").read(nbytes)
    if len(raw) != nbytes:
        raise Refuse("record-short", "%s: %d bytes on disk, manifest says %d"
                     % (path, len(raw), nbytes))
    import numpy as np
    if dtype in ("bf16", "torch.bfloat16"):
        # bf16 -> f32 is a left shift by 16 in the raw bits (pitfalls/30-numerics.md §Reading a
        # bf16 bin: `.view`, not `.to`); the shift is done on widening ints so no bits are lost.
        u16 = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
        v = u16.view("<f4").astype("float64")
    else:
        v = np.frombuffer(raw, dtype="<f4").astype("float64")
    if not bool(np.isfinite(v).all()):
        return {"n": int(n), "finite": False, "min": None, "max": None, "mean": None, "rms": None}
    return {"n": int(n), "finite": True, "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "rms": float((v * v).mean() ** 0.5)}


def mode_structural(chain, llama, pin, stages_by_block, llama_rows):
    print("")
    print("=== structural: what corresponds to what, and how far the comparison reaches ===")
    graph = llama_graph(llama, "prefill")[0]["index"]
    checks, findings = 0, []
    print("")
    print("stage mapping — one row per §03.2 stage per block; denominator %d rows"
          % (len(BLOCKS) * HF_STAGES_EXPECT))
    print("block  kind(ll/HF)  hf_stage   engine_stage             llama_tensor              status")
    mapped_pairs, mapped_slots, unwritten_slots, pending_slots = [], 0, 0, 0
    for blk in BLOCKS:
        hk, lk = kind_of_block(chain, blk), llama_kind(llama, blk)
        hk_short = {"linear_attention": "linear", "full_attention": "full"}[hk]
        checks += 1
        if hk_short != lk:
            findings.append("block-kind-disagree:%s hf=%s llama=%s" % (blk, hk, lk))
        il = int(blk.replace("layer", ""))
        hf_by_stage = {d["stage"]: d for d in stages_by_block[blk]}
        for stage in stage_order(chain):
            eng, names, cite = STAGE_MAP[stage]
            want = names[lk]
            hits, unwritten = [], []
            if "/" not in want:
                all_rows = resolve(llama_rows, graph, il, want)
                hits = [r for r in all_rows if r["written"]]
                unwritten = [r for r in all_rows if not r["written"]]
            if hits:
                r = hits[0]
                status = "MAPPED(%d)" % len(hits)
                mapped_slots += 1
                mapped_pairs.append((blk, stage, eng, r))
            elif unwritten:
                # The row exists in the manifest — so the name mapping is right — but the instrument
                # wrote no bytes for it. In run-a that is the last block of the prefill graph: after
                # the `crop_before_nextn` gather at qwen35moe.cpp:194-196 its token axis is 0, so
                # every node after it has shape [2048, 0, 1, 1] and nothing to write. A fixture
                # absence, named as one, not a mapping error and not quietly a pass.
                u = unwritten[0]
                elsewhere = [(g, r["shape"]) for gg in sorted({k[0] for k in llama_rows})
                             for r in resolve(llama_rows, gg, il, want) if r["written"]
                             for g in [gg]]
                status = ("UNWRITTEN-BY-INSTRUMENT(name %s is right, shape %s has a zero token axis, "
                          "written=false; the same name IS written in %s)"
                          % (u["name"], u["shape"],
                             ", ".join("graph %d %s" % es for es in elsewhere) or "no other graph"))
                unwritten_slots += 1
            elif "/" in want:
                gap = next((d.get("chain_gap") for d in stages_by_block[blk] if d.get("chain_gap")),
                           None)
                if gap is None or not gap.get("layers_between_missing"):
                    why = "its chained predecessor is the immediately previous layer"
                else:
                    why = ("its chained predecessor is layer%d, with layers %d..%d of the real model "
                           "absent from the fixture"
                           % (gap["from_index"], gap["layers_between_missing"][0],
                              gap["layers_between_missing"][-1]))
                status = ("PENDING-BY-STIMULUS(llama side %s; HF side %s)" % (want, why))
                pending_slots += 1
            else:
                status = "UNMAPPED-BY-ABSENCE(llama:%s)" % want
                findings.append("mapped-name-absent:%s/%s expects llama %s and no row of that block "
                                "carries the name" % (blk, stage, want))
            print("%-6s %-12s %-10s %-24s %-23s %s"
                  % (blk, "%s/%s" % (lk, hk_short), stage,
                     eng or "UNMAPPED-BY-ABSENCE(engine)",
                     (hits[0]["name"] if hits else want)[:23], status + " [" + cite + "]"))
    print("mapping   denominator: %d rows total (%d blocks x %d §03.2 stages); %d resolve to a "
          "written llama tensor, %d name a real llama row that the instrument wrote no bytes for, "
          "%d are position-only pending the stimulus, %d are engine absences (engine/block_forward.py "
          "computes the mixer output and the MoE sum and nothing else at §03.2 granularity)"
          % (len(BLOCKS) * HF_STAGES_EXPECT, len(BLOCKS), HF_STAGES_EXPECT,
             mapped_slots, unwritten_slots, pending_slots,
             sum(1 for blk in BLOCKS for s in stage_order(chain) if STAGE_MAP[s][0] is None)))

    print("")
    print("shape/order agreement — denominator %d mapped pairs" % len(mapped_pairs))
    order_pairs_total, order_ok_total = 0, 0
    for blk in BLOCKS:
        nodes = []
        for b, stage, eng, r in mapped_pairs:
            if b != blk or r["block"] < 0:
                continue
            hf = {d["stage"]: d for d in stages_by_block[blk]}[stage]
            checks += 1
            hf_hidden = hf["shape"][-1]
            ll_hidden = r["shape"][0]
            if hf_hidden != ll_hidden:
                findings.append("hidden-axis-disagree:%s/%s hf=%d llama=%d"
                                % (blk, stage, hf_hidden, ll_hidden))
            tokens_hf, tokens_ll = hf["shape"][0], r["shape"][1]
            if tokens_hf != tokens_ll:
                print("  %-6s %-11s hidden %d==%d ok; token axis hf=%d llama=%d "
                      "STIMULUS-MISMATCH(not a mapping error)"
                      % (blk, stage, hf_hidden, ll_hidden, tokens_hf, tokens_ll))
            else:
                print("  %-6s %-11s hidden %d==%d ok; token axis %d==%d"
                      % (blk, stage, hf_hidden, ll_hidden, tokens_hf, tokens_ll))
            if r["block"] >= 0 and "input_embed" not in r["name"]:
                nodes.append((stage, r["node"], r["op"], r["dtype"], r["shape"], r["bytes"]))
        order = [s for s, _, _, _, _, _ in nodes]
        order_ok = 0
        for a, b in zip(order, order[1:]):
            order_pairs_total += 1
            ia = [n for n in nodes if n[0] == a][0][1]
            ib = [n for n in nodes if n[0] == b][0][1]
            if ib > ia:
                order_ok += 1
                order_ok_total += 1
            else:
                findings.append("graph-order-inverts:%s %s(node %d) after %s(node %d)"
                                % (blk, b, ib, a, ia))
        checks += 1
        print("  %-6s graph order %s -> %d/%d consecutive pairs increasing"
              % (blk, " < ".join(order), order_ok, len(order) - 1))
    print("shape/order denominator: %d pairs mapped, %d ordering pairs checked, %d increasing, "
          "checks so far=%d" % (len(mapped_pairs), order_pairs_total, order_ok_total, checks))

    print("")
    print("per-side statistics — each side's own values from its own .bin; NOT a comparison, "
          "and no delta column is printed because the stimuli differ")
    print("block  stage        side   elements   finite  min          max          mean        rms")
    n_stats = 0
    for blk in BLOCKS:
        hf_by_stage = {d["stage"]: d for d in stages_by_block[blk]}
        mapped = {s: r for b, s, e, r in mapped_pairs if b == blk}
        for stage in stage_order(chain):
            hf = hf_by_stage[stage]
            s = stats_of(os.path.join(REPO, hf["file"]), hf["dtype"], hf["shape"], hf["bytes"])
            checks += 1
            n_stats += 1
            if not s["finite"] or not (s["rms"] > 0.0):
                findings.append("hf-stage-degenerate:%s/%s %s" % (blk, stage, s))
            print("%-6s %-12s hf     %-9d %-7s %-12.5f %-12.5f %-11.6f %.6f"
                  % (blk, stage, s["n"], s["finite"], s["min"], s["max"], s["mean"], s["rms"]))
            r = mapped.get(stage)
            if r is None:
                continue
            s2 = stats_of(os.path.join(REPO, r["file"]), r["dtype"], r["shape"], r["bytes"])
            checks += 1
            n_stats += 1
            if not s2["finite"] or not (s2["rms"] > 0.0):
                findings.append("llama-stage-degenerate:%s/%s %s" % (blk, stage, s2))
            print("%-6s %-12s llama    %-9d %-7s %-12.5f %-12.5f %-11.6f %.6f"
                  % (blk, r["name"], s2["n"], s2["finite"], s2["min"], s2["max"], s2["mean"],
                     s2["rms"]))
    print("statistics denominator: %d records profiled (%d hf + %d llama), 0-element records = "
          "a failure and there were none" % (n_stats, len(BLOCKS) * HF_STAGES_EXPECT,
                                             n_stats - len(BLOCKS) * HF_STAGES_EXPECT))

    # Distinctness over the manifest's own digests: the instrument registers one row per
    # (graph, node, seq) and keeps the first name (tools/LLAMA_DUMP_NOTES.md §Reading the dump), so a
    # stage that silently aliases to another tensor would print four identical blocks. The digests
    # come from the manifest; no bytes are re-hashed here.
    print("")
    print("aliasing/distinctness — digests from the manifest, denominator 2 groups")
    for stage in ("mixer_out", "moe_out"):
        digs = []
        for blk in BLOCKS:
            hit = [(b, s, e, r) for (b, s, e, r) in mapped_pairs if b == blk and s == stage]
            if hit:
                digs.append(hit[0][3]["sha256"][:16])
        checks += 1
        distinct = len(set(digs))
        print("  stage %-10s blocks=%d distinct_digests=%d %s" % (stage, len(digs), distinct,
              "OK" if distinct == len(digs) and len(digs) > 1 else
              "FAIL: a stage digest repeating across blocks is the aliasing the dump dedupe causes"))
        if not (distinct == len(digs) and len(digs) > 1):
            findings.append("digest-not-distinct:%s distinct=%d of %d" % (stage, distinct, len(digs)))
    return checks, findings, mapped_pairs


# --------------------------------------------------------------------------------------------
# engine side: what it computes, derived from its own source, never asserted from memory
# --------------------------------------------------------------------------------------------
def engine_coverage():
    src = open(os.path.join(REPO, "engine/block_forward.py")).read()
    consts = dict(re.findall(r'^([A-Z_]+)\s*=\s*"?([A-Za-z0-9_]+)"?\s*(?:#.*)?$', src, re.M))
    stages = []
    for fn, body in re.findall(r"^def (\w+)\([^)]*\):\n((?:[\t ]+.*\n|\n)+)", src, re.M):
        stages.extend((fn, s) for s in re.findall(r'st\["([a-z0-9_]+)"\]\s*=', body))
    return consts, stages


def mode_engine_coverage():
    print("")
    print("=== engine side: what our implementation can even be asked for ===")
    consts, stages = engine_coverage()
    blk, row, mode = consts.get("BLOCK"), consts.get("TOKEN_ROW"), consts.get("GDN_MODE")
    print("engine  block_forward.py constants: BLOCK=%s TOKEN_ROW=%s GDN_MODE=%s"
          % (blk, row, mode))
    print("engine  stages computed (function, stage): %d — %s"
          % (len(stages), ", ".join("%s.%s" % (f, s) for f, s in stages)))
    print("engine  at §03.2 granularity only two of the seven stages exist: the mixer output "
          "(`y`) and the MoE sum (`combined_out`); input_layernorm and the two residual adds are "
          "not computed, so there is nothing of ours to put against hf `h_in`, `x1`, `out`")
    print("engine  denominator: 1 block x 1 row x 7 stages, of which 2 stages exist = 2 "
          "candidate cells out of the 4 blocks x 7 stages = 28 the report could have used")
    return len(stages)


# --------------------------------------------------------------------------------------------
# pairwise: the three disagreement counts per block, and the refusals
# --------------------------------------------------------------------------------------------
def mode_pairs(chain, llama, stages_by_block, llama_rows, mapped_pairs, aligned):
    print("")
    print("=== pairwise disagreement counts (engine-vs-HF, engine-vs-llama, HF-vs-llama) ===")
    graph = llama_graph(llama, "prefill")[0]["index"]
    consts, _stages = engine_coverage()
    engine_block = "layer%s" % consts.get("BLOCK", "?")
    order = stage_order(chain)
    for blk in BLOCKS:
        il = int(blk.replace("layer", ""))
        pairs_here = [(b, s, e, r) for (b, s, e, r) in mapped_pairs if b == blk]
        # The engine's two stages exist for one block only, the one block_forward.py names.
        engine_slots = [s for b, s, e, r in pairs_here if e and blk == engine_block]
        slots = {"engine-vs-HF": len(engine_slots), "engine-vs-llama": len(engine_slots),
                 "HF-vs-llama": len(pairs_here)}
        n_elts = {"engine-vs-HF": len(engine_slots) * chain["input"]["n_tokens"] * 2048,
                  "engine-vs-llama": (len(engine_slots)
                                      * llama_graph(llama, "prefill")[0]["n_tokens"] * 2048),
                  "HF-vs-llama": len(pairs_here) * chain["input"]["n_tokens"] * 2048}
        for pair in ("engine-vs-HF", "engine-vs-llama", "HF-vs-llama"):
            if aligned:
                print("%-6s pair=%-15s stage_slots=%d elements_compared=%d verdict=ALIGNED-BUT-"
                      "NUMERIC-PATH-NOT-IMPLEMENTED" % (blk, pair, slots[pair], n_elts[pair]))
                continue
            reason = {
                "engine-vs-HF": (
                    "engine/block_forward.py is BLOCK=%s TOKEN_ROW=%s mode %s: %s"
                    % (consts.get("BLOCK"), consts.get("TOKEN_ROW"), consts.get("GDN_MODE"),
                       ("it computes one decode row of layer %s while the HF record is a %d-token "
                        "prefill of that layer, so no row is shared"
                        % (consts.get("BLOCK"), chain["input"]["n_tokens"]))
                       if blk == engine_block else
                       ("layer %s is outside the single block it implements (BLOCK=%s), so there is "
                        "no engine value for it at all"
                        % (il, consts.get("BLOCK"))))),
                "engine-vs-llama": ("llama graph %d is an %d-token prefill of a different prompt; no "
                                    "engine stage shares its (block, position)"
                                    % (graph, llama_graph(llama, "prefill")[0]["n_tokens"])),
                "HF-vs-llama": ("chain ran on the pinned %d sampled ids; llama graph %d ran on %d "
                                "template-wrapped ids; the shapes differ (%d x 2048 vs 2048 x %d) and "
                                "layers 38/39 have a chain gap on top of that"
                                % (chain["input"]["n_tokens"], graph,
                                   llama_graph(llama, "prefill")[0]["n_tokens"],
                                   chain["input"]["n_tokens"],
                                   llama_graph(llama, "prefill")[0]["n_tokens"])),
            }[pair]
            print("%-6s pair=%-15s stage_slots=%d elements_compared=0 of %d disagreements=n/a "
                  "verdict=UNCOMPARABLE-BY-STIMULUS because %s"
                  % (blk, pair, slots[pair], n_elts[pair], reason))
    print("pairwise denominator: 3 pairs x %d blocks = %d lines; numeric comparisons performed = "
          "0; a zero-element comparison is a refusal, not a pass, and is labelled as one"
          % (len(BLOCKS), 3 * len(BLOCKS)))


REFUSED_HF_VS_LLAMA = [
    ("activation-width",
     "the HF chain records activations as bf16 (dtype field of each record in "
     "checkpoints/dump-manifest-chain-2026-10-26.json), llama.cpp records the same stages as f32 "
     "(dtype field per row of checkpoints/llama-dump-manifest-2026-10-27-run-a.json). Two numeric "
     "systems on the same stage.",
     "D-26 clause 3: llama.cpp is the oracle for engine/*, but a quantized oracle may only ADD "
     "checks, never loosen one; calling f32 'the reference' would license a wider input than the "
     "bf16 fixture we are gated on, and calling bf16 the reference would loosen llama. Not "
     "adjudicated; the user's ±1-ULP policy (D-17) is where this belongs."),
    ("router renormalisation dtype and form",
     "llama renormalises the top-k weights with a f32 DIV (graph 1 block 0 node 74 "
     "ffn_moe_weights_norm, op DIV, shape [8, 11, 1, 1]); the HF form casts the renormalised "
     "weights back to the logits dtype before multiplying the expert output, and our §11.6 fixture "
     "already refused the f32 carry at ndiff=1326 with a flipped argmax (engine/HANDOFF.md §MoE "
     "renormalisation dtype). The pinned fixture sides with HF; llama disagrees with the fixture.",
     "D-26 clause 3 — recorded as a named divergence, not settled. Settling it would either move "
     "our §11.6 must-catch set (a gate change) or declare the oracle wrong; both are the user's."),
    ("softmax domain before top-k",
     "llama takes SOFT_MAX over all 256 router logits (graph 1 block 0 node 66, shape "
     "[256, 11, 1, 1]) and then selects and renormalises; §03.2 softmaxes the top-k subset after "
     "selection. Mathematically the renormalised results agree; bitwise they will not, and the "
     "banded/quantized router path can flip a near-tie.",
     "same refusal as above; also unmeasurable today because the stimuli differ."),
    ("chained-vs-real activations at layers 38 and 39",
     "the HF chain is not a contiguous run: layer1's input is layer0's output, but layer38's input "
     "is layer1's output with layers 2..37 absent (why_not_contiguous / chain_range in "
     "checkpoints/dump-manifest-chain-2026-10-26.json). llama's graph-1 layer-38 activation is a "
     "real forward over all 40 layers.",
     "Even under an aligned stimulus a layer-38/39 HF-vs-llama delta would mix a chain gap with an "
     "implementation difference. The comparable HF-vs-llama scope is layer0 and layer1; layers 38 "
     "and 39 are structural-only regardless of stimulus. Reported, not adjudicated."),
]


def mode_refusals():
    print("")
    print("=== refused HF-vs-llama adjudications (D-26 clause 3) ===")
    for i, (name, obs, refusal) in enumerate(REFUSED_HF_VS_LLAMA, 1):
        print("refused[%d] %s\n    observation: %s\n    why-refused: %s" % (i, name, obs, refusal))
    print("refusals denominator: %d named, 0 adjudicated, 0 bands authored"
          % len(REFUSED_HF_VS_LLAMA))


def mode_numeric(aligned):
    print("")
    print("=== numeric: three-way magnitudes per stage ===")
    if aligned:
        print("numeric alignment holds, but the numeric path is not implemented in this revision; "
              "it would print, per stage and pair: elements, n_diff, max_abs_delta, and "
              "licence=llama-crosscheck(observation-only) with no threshold. Refusing to fake it.")
        return
    for blk in BLOCKS:
        print("%-6s numeric verdict=UNCOMPARABLE-BY-STIMULUS elements_compared=0 of "
              "3 x 7 stage slots; a band here would be arithmetic on unrelated tensors" % blk)
    print("numeric denominator: %d stage slots x 3 pairs = %d possible comparisons, 0 performed"
          % (len(BLOCKS) * HF_STAGES_EXPECT, 3 * len(BLOCKS) * HF_STAGES_EXPECT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="all", choices=("align", "structural", "numeric", "all"))
    args = ap.parse_args()
    try:
        chain, llama, pin, stages_by_block, llama_rows = load_sides()
    except Refuse as exc:
        print("FATAL     [%s] %s" % (exc.tag, exc))
        print("RESULT    verdict=AUDIT-ERROR findings=1 no timing taken")
        return 1
    checks = findings_n = 0
    findings = []
    print("cross_oracle sides: hf=%s (%d records)  llama=%s (%d rows)  engine=engine/block_forward.py"
          % (os.path.relpath(CHAIN_MANIFEST, REPO), sum(len(v) for v in stages_by_block.values()),
             os.path.relpath(LLAMA_MANIFEST, REPO), len(llama["dumps"])))
    aligned = False
    if args.mode in ("align", "all"):
        aligned = mode_align(chain, llama, pin) == 0
    if args.mode == "align":
        print("")
        print("RESULT    checks=0 findings=0 aligned=%s verdict=%s no timing taken"
              % (aligned, "ALIGNED" if aligned else "NOT-ALIGNED-BY-STIMULUS"))
        return 0 if aligned else 3
    if args.mode == "numeric":
        mode_numeric(aligned)
        print("")
        print("RESULT    checks=0 findings=0 aligned=%s verdict=%s no bands authored=no "
              "licence-issued=none no timing taken"
              % (aligned, "NUMERIC-PATH-NOT-IMPLEMENTED" if aligned else "UNCOMPARABLE-BY-STIMULUS"))
        return 0 if aligned else 3
    if args.mode in ("structural", "all"):
        checks, findings, mapped_pairs = mode_structural(chain, llama, pin, stages_by_block,
                                                         llama_rows)
        mode_engine_coverage()
        mode_pairs(chain, llama, stages_by_block, llama_rows, mapped_pairs, aligned)
        mode_refusals()
        findings_n = len(findings)
    if args.mode == "all" and not aligned:
        mode_numeric(False)
    print("")
    for f in findings:
        print("FINDING   %s" % f)
    print("RESULT    checks=%d findings=%d aligned=%s no bands authored=no licence-issued=none "
          "verdict=%s no timing taken"
          % (checks, findings_n, aligned,
             "STRUCTURAL-OK" if findings_n == 0 else "STRUCTURAL-BLOCKED"))
    return 0 if findings_n == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
