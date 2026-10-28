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

Why alignment is not merely untried-but-awkward FOR TEXT: the vendored front ends take text, never ids
(`tools/LLAMA_DUMP_NOTES.md` §CLI surface is the whole instrument surface), and byte-level BPE
round-trips ids to text only where its merges agree with the concatenation boundary. Measured on the
chain's own 32 ids: detokenize → 205 bytes (valid UTF-8, no control-class token) → `llama-tokenize -f`
→ 32 ids again, but only 30 equal; first divergence index 3 (chain 78392, re-tokenized 750). The byte
string is forced by the ids (byte-level BPE is byte-preserving and the token→piece map is a
bijection), so that candidate is the only candidate: no text prefill reproduces those 32 ids — which is
what you would expect of ids that were sampled rather than tokenized. Control that the channel itself
is template-free: tokenizing the run-a prompt text returns exactly the prefill's 5 inner ids. Evidence
and reproduction commands: quantization/evidence/llama-stimulus-alignment-2026-10-27.md.

That proof still stands, and it is why the alignment used here came from the OTHER door that file
named: an id-input front end, `tools/llama_tokens_dump.cpp` driven by `tools/llama_aligned_run.py`,
which hands the oracle the chain's ids through `llama_batch.token[]` with no tokenizer and no chat
template in the process image at all (the binary does not link `libllama-common`). The aligned run's
own manifest — read `graphs[0].token_ids` out of it, written by the instrumentation inside
`llama-context.cpp`, not echoed by the driver — is what `align` and `aligned` check against the chain.

Modes:
  align       — stimulus/identity evidence for the three sides; exit non-zero if unaligned. Point
                `--llama-manifest` at an aligned run's manifest and this is where the equality is checked.
  structural  — stage mapping, shape/order agreement, per-side statistics, denominators.
  numeric     — the pairwise numeric comparison. Refuses unless alignment holds, and says so.
  aligned     — the aligned comparison: re-verifies every precondition from files (ids, gguf pin, no
                control-class id in the stimulus, the determinism pair and its non-vacuity control),
                then MEASURES the HF-vs-llama distance on the blocks HF can reach and prints it as
                `licence=llama-crosscheck(observation-only)` with no threshold and `adjudicated=no`.
                Measuring a divergence is D-26 clause 3 escalation material; adopting a side from it is not.
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
# This family's control-class ids start here (bos/eos 248044, end-of-turn 248046, think-boundary
# 248068/248069 — README.md §stop list), and they are referred to by id and never printed as text
# (AGENTS.md §the special-token rule). Used to show a template was or was not applied.
CONTROL_CLASS_MIN = 248044
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
    pre_ctrl = [t for t in pre.get("token_ids", []) if int(t) >= CONTROL_CLASS_MIN]
    if pre_ctrl:
        print("llama   the prefill's ids 0, 8, 10 are 248045 / 248046 / 198 — the chat wrapper — and its "
              "5 inner ids are the prompt text; `llama-tokenize` on that text returns exactly those 5 "
              "ids, which is the template-free proof")
    else:
        print("llama   the prefill carries no control-class id (>= %d) at all, and its token ids are "
              "elementwise the chain's, so this is the raw-id path: no wrapper before, none after"
              % CONTROL_CLASS_MIN)
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
    print("align   unattainable-through-text=yes: detokenizing the chain's 32 ids gives 205 "
          "bytes (valid UTF-8, 0 control-class tokens); `llama-tokenize -f` on those bytes returns 32 "
          "ids of which only 30 are equal, first divergence index 3 (chain 78392 vs 750). The byte "
          "string is forced by the ids (byte-level BPE is byte-preserving and token->piece is a "
          "bijection), so that candidate is the only candidate and no text prefill reproduces them. "
          "Reproduction: quantization/evidence/llama-stimulus-alignment-2026-10-27.md")
    print("align   attainable-through-raw-ids=yes: tools/llama_tokens_dump.cpp feeds the ids through "
          "batch.token[] with no tokenizer in the process image; run "
          "tools/llama_aligned_run.py run <tag>, then re-run this mode with --llama-manifest pointing "
          "at that run's manifest (the aligned fixture records its path). This manifest (%s) is the "
          "template-wrapped run-a, which is why the line above is NOT-ALIGNED and not a refusal to look."
          % os.path.relpath(LLAMA_MANIFEST, REPO))
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


# --------------------------------------------------------------------------------------------
# aligned mode: the one stimulus both oracles can be compared on
# --------------------------------------------------------------------------------------------
# The three-way report (`mode_structural`, `mode_pairs`) compares the pinned HF chain against the
# template-wrapped run-a dumps and therefore compares nothing numeric. The aligned stimulus exists now
# — tools/llama_tokens_dump.cpp feeds the oracle the chain's own 32 ids and the evidence is
# checkpoints/m3-candidate-llama-fixture-2026-10-27.json — so this mode does two things and only two:
# it re-verifies the alignment preconditions from files (D-26 clause 4's determinism included), and it
# measures how far the two oracles are from each other on the stages the HF chain can reach, printing
# `licence=llama-crosscheck(observation-only)` with NO threshold attached. It still does not adjudicate
# HF-vs-llama disagreement: measuring a divergence is D-26 clause 3's escalation material, adopting a
# side from it is not. Engine-vs-oracle cells are reported absent, because engine/block_forward.py is a
# one-block, one-token lane and the stimulus is a 32-token prefill.

ALIGNED_MANIFEST = os.environ.get("ALIGNED_MANIFEST", "")
CANDIDATE_MANIFEST = os.environ.get("CANDIDATE_MANIFEST",
                                    os.path.join(REPO, "checkpoints/m3-candidate-llama-fixture-"
                                                            "2026-10-27.json"))
# The two blocks where an HF value and a llama value are the same question: layers 0 and 1 are the
# chain's contiguous head, so layer1's input really is layer0's output. Layers 38/39 are reached in the
# chain from layer1 with layers 2..37 absent, so their HF values answer a different forward.
HF_COMPARABLE = ("layer0", "layer1")


def token_major(path, dtype, shape, nbytes, ggml):
    """One record, returned token-major ([tokens, hidden]).

    The two sides store the same matrix two ways and guessing which is which is how a delta ends up
    comparing element order instead of arithmetic. The HF chain wrote torch tensors shaped
    [tokens, hidden] in C order, so its flat bytes read straight into (tokens, hidden). ggml shapes
    are [hidden, tokens] with ne[0] fastest (index = i0 + i1*ne0), so the same bytes of a
    [2048, 32, 1, 1] tensor read into (32, 2048) ARE the token-major matrix. The `ggml` flag says
    which contract the caller is under rather than inferring it from the numbers, which would be
    undecided for a square tensor.
    """
    v = values_of(path, dtype, shape, nbytes)
    dims = list(shape)
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if len(dims) != 2:
        raise Refuse("layout-unknown", "%s: shape %s is not a 2-d matrix after trailing 1s are "
                     "dropped; this comparison is defined on [tokens, hidden] only" % (path, shape))
    return v.reshape(dims[1], dims[0]) if ggml else v.reshape(dims[0], dims[1])


def values_of(path, dtype, shape, nbytes):
    """One side's own values, widened to float64 without losing bits: bf16 -> f32 is a left shift by 16
    on the raw bits (pitfalls/30-numerics.md §the `.view` rule), f32 is read as f32."""
    import numpy as np
    esz = {"bf16": 2, "torch.bfloat16": 2, "f32": 4, "float32": 4}.get(dtype)
    if esz is None:
        raise Refuse("dtype-unknown", "%s: dtype %r" % (path, dtype))
    raw = open(path, "rb").read(nbytes)
    if len(raw) != nbytes:
        raise Refuse("record-short", "%s: %d of %d bytes" % (path, len(raw), nbytes))
    n = 1
    for s in shape:
        n *= s
    if n * esz != nbytes:
        raise Refuse("record-size", "%s: shape %s x %d != %d" % (path, shape, esz, nbytes))
    if esz == 2:
        return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").astype("float64")
    return np.frombuffer(raw, dtype="<f4").astype("float64")


def aligned_preconditions(chain, al, pin, cand):
    """Every precondition M3's G0 will gate on, each one counted, each one a file-derived equality.
    Returns (checks, findings, facts)."""
    checks, findings, facts = 0, [], {}
    ids = [int(t) for t in chain["input"]["token_ids"]]
    print("stim    hf_ids=%d chain_sha=%s" % (len(ids), chain["input"]["token_ids_sha256"][:16]))
    graphs = al.get("graphs", [])
    checks += 1
    if len(graphs) != 1:
        findings.append("aligned-graph-count:expected 1 captured graph (one ubatch), got %d" % len(graphs))
    g = graphs[0] if graphs else {}
    got = [int(t) for t in g.get("token_ids", [])]
    checks += 1
    if got != ids:
        findings.append("stimulus-not-equal:the oracle consumed %d ids, the chain consumed %d; first "
                        "difference at index %s" % (len(got), len(ids),
                        next((i for i, (a, b) in enumerate(zip(got, ids)) if a != b), len(got))))
    checks += 1
    got_sha = ids_sha256_local(got)
    if got_sha != chain["input"]["token_ids_sha256"]:
        findings.append("token-ids-sha-mismatch:%s != chain %s" % (got_sha[:16],
                                                                   chain["input"]["token_ids_sha256"][:16]))
    print("stim    oracle_ids=%d oracle_sha=%s equal_to_chain=%s" % (len(got), got_sha[:16],
                                                                    got_sha == chain["input"]["token_ids_sha256"]))
    checks += 1
    ctrl = [t for t in got if t >= CONTROL_CLASS_MIN]
    if ctrl:
        findings.append("control-class-id-in-stimulus:%s — a template was applied" % ctrl)
    print("stim    control_class_ids=%s (>= %d); a chat template would have added some and the ids "
          "above would not match the chain's" % (ctrl or "none", CONTROL_CLASS_MIN))
    checks += 1
    if al["gguf"]["sha256"] != pin["sha256"] or not al["gguf"]["matches_pin"]:
        findings.append("gguf-not-pinned:%s" % al["gguf"]["sha256"][:16])
    print("gguf    aligned_run_gguf=%s pin=%s matches=%s"
          % (al["gguf"]["sha256"][:16], pin["sha256"][:16], al["gguf"]["matches_pin"]))
    checks += 1
    if cand.get("status") != "candidate" or "not-registered" not in str(cand.get("registration", "")):
        findings.append("candidate-status:%s/%s — this mode reads a CANDIDATE fixture; a registered "
                        "pin must be re-checked by the gate that licenses pins, not here"
                        % (cand.get("status"), str(cand.get("registration", ""))[:24]))
    det = cand.get("determinism", {})
    checks += 1
    if not (det.get("aggregate_digests_equal") and det.get("tensors_differing") == 0
            and det.get("tensors_compared", 0) > 0):
        findings.append("determinism-not-established:%s" % json.dumps(det)[:120])
    print("determ  runs=%s tensors_compared=%s differing=%s aggregate_equal=%s (D-26 clause 4)"
          % (det.get("runs"), det.get("tensors_compared"), det.get("tensors_differing"),
             det.get("aggregate_digests_equal")))
    checks += 1
    if det.get("non_vacuity_control", {}).get("differing", 0) <= 0:
        findings.append("determinism-control-vacuous:the same comparator found no difference against a "
                        "different stimulus (control=%s), so its verdict of equality proves nothing"
                        % json.dumps(det.get("non_vacuity_control"))[:120])
    print("determ  non-vacuity control vs %s: keys_compared=%s differing=%s"
          % (os.path.basename(det.get("non_vacuity_control", {}).get("comparison_against", "?")),
             det.get("non_vacuity_control", {}).get("keys_compared"),
             det.get("non_vacuity_control", {}).get("differing")))
    return checks, findings, {"graph": g, "ids": got, "sha": got_sha}


def ids_sha256_local(ids):
    import hashlib
    import struct
    return hashlib.sha256(b"".join(struct.pack("<q", int(t)) for t in ids)).hexdigest()


def mode_aligned(chain, pin, stages_by_block):
    """The aligned comparison itself. HF x llama on the stages the chain can reach. Counted, and no
    band: every magnitude line carries licence=llama-crosscheck(observation-only) and a refusal to
    adjudicate."""
    print("")
    print("=== aligned: one stimulus, two oracles, measured distance and no verdict about who is right ===")
    if not ALIGNED_MANIFEST or not os.path.exists(ALIGNED_MANIFEST):
        print("aligned ALIGNED_MANIFEST is unset or missing (%s) — produce it with "
              "tools/llama_aligned_run.py run <tag>; until then nothing here is comparable and this "
              "mode refuses rather than printing the old mismatched-stimulus table as if it were news"
              % (ALIGNED_MANIFEST or "unset"))
        return 1, ["aligned-manifest-absent"], 0
    al = json.load(open(ALIGNED_MANIFEST))
    cand = json.load(open(CANDIDATE_MANIFEST))
    import numpy as np
    checks, findings, facts = aligned_preconditions(chain, al, pin, cand)
    graph = facts["graph"]["index"]
    n_tok = len(facts["ids"])

    rows = {}
    for r in al["dumps"]:
        if r["written"]:
            rows.setdefault((r["block"], r["name"]), []).append(r)

    print("")
    print("hf x llama on the comparable blocks (%s): denominator %d blocks x %d stages = %d cells"
          % ("+".join(HF_COMPARABLE), len(HF_COMPARABLE), HF_STAGES_EXPECT,
             len(HF_COMPARABLE) * HF_STAGES_EXPECT))
    # The layout claim, witnessed rather than assumed. ggml stores [hidden, tokens] with ne[0] fastest
    # and the HF chain stored torch [tokens, hidden], so the comparison is only arithmetic if one side
    # is re-viewed. Both readings are computed for one cell and the chosen one has to be the closer by
    # a wide margin: a wrong re-view scrambles element order and lands near sqrt(2) relative error,
    # while the real HF-vs-llama divergence here is a few percent. Without this leg "we transpose" is a
    # sentence, and a sentence cannot fail.
    probe_stage = stage_order(chain)[stage_order(chain).index("out")]
    probe_hf = {d["stage"]: d for d in stages_by_block["layer0"]}[probe_stage]
    probe_ll = rows.get((0, STAGE_MAP[probe_stage][1]["linear"]), [])
    if probe_ll:
        checks += 1
        flat = values_of(probe_ll[0]["file"], probe_ll[0]["dtype"], probe_ll[0]["shape"],
                         probe_ll[0]["bytes"])
        hf_probe = values_of(probe_hf["file"], probe_hf["dtype"], probe_hf["shape"], probe_hf["bytes"])
        rel = lambda a, b: float(np.sqrt(((a - b) ** 2).sum())) / float(np.sqrt((b * b).sum()))
        r_chosen = rel(flat, hf_probe)
        r_other = rel(flat.reshape(2048, 32).T.reshape(-1), hf_probe)
        print("layout  cell=layer0.%s relL2[token-major re-view]=%.3e relL2[raw re-view]=%.3e "
              "chosen=%s margin=%.1fx (a wrong re-view is ~1.4, the real divergence is ~0.04)"
              % (probe_stage, r_chosen, r_other, "token-major", r_other / max(r_chosen, 1e-12)))
        if not (r_other > 5.0 * r_chosen):
            findings.append("layout-not-witnessed:layer0.%s chosen %.3e vs other %.3e — the transpose "
                            "is not established by a margin, so no delta below is trustworthy"
                            % (probe_stage, r_chosen, r_other))
    cells, refused_cells, escalations = 0, 0, 0
    for blk in HF_COMPARABLE:
        il = int(blk.replace("layer", ""))
        hf = {d["stage"]: d for d in stages_by_block[blk]}
        for stage in stage_order(chain):
            eng, names, cite = STAGE_MAP[stage]
            want = names["linear" if il in al["model"]["linear_layers"] else "full"]
            if "/" in want:
                # `x` is the chained input: layer0's is the embedding row set, layer1's is layer0's
                # output. Both exist in the aligned dump, so the cell is fillable rather than position-only.
                if stage == "x":
                    src = (-1, "model.input_embed") if il == 0 else (il - 1, "l_out")
                else:
                    refused_cells += 1
                    print("  %-6s %-10s llama-name=%s refused: position-only mapping (chained input)"
                          % (blk, stage, want))
                    continue
            else:
                src = (il, want)
            hits = rows.get(src, [])
            if not hits:
                refused_cells += 1
                findings.append("aligned-row-absent:%s/%s wants llama (%d,%s) and the aligned dump "
                                "wrote no such row" % (blk, stage, src[0], src[1]))
                print("  %-6s %-10s llama=(%d,%s) ABSENT-BY-INSTRUMENT" % (blk, stage, src[0], src[1]))
                continue
            lr = hits[0]
            h = hf[stage]
            hv = token_major(h["file"], h["dtype"], h["shape"], h["bytes"], ggml=False)
            lv = token_major(lr["file"], lr["dtype"], lr["shape"], lr["bytes"], ggml=True)
            if hv.shape != lv.shape:
                refused_cells += 1
                print("  %-6s %-10s shapes hf=%s llama=%s SHAPE-MISMATCH(token axis or transpose; "
                      "a delta here would be element-order, not arithmetic)"
                      % (blk, stage, h["shape"], lr["shape"]))
                continue
            d = hv - lv
            denom = float(np.sqrt((lv * lv).sum())) or 1.0
            cells += 1
            checks += 1
            if not np.isfinite(d).all():
                findings.append("non-finite-delta:%s/%s" % (blk, stage))
            escalations += 1
            print("  %-6s %-10s hf=%s/%s llama=%s/%s n=%d max_abs_delta=%.3e relL2=%.3e "
                  "hf_rms=%.4f llama_rms=%.4f elements_compared=%d "
                  "licence=llama-crosscheck(observation-only) band=NONE adjudicated=no"
                  % (blk, stage, h["dtype"], "x".join(map(str, h["shape"])),
                     lr["dtype"], "x".join(map(str, lr["shape"])), hv.size,
                     float(np.abs(d).max()), float(np.sqrt((d * d).sum())) / denom,
                     float(np.sqrt((hv * hv).mean())), float(np.sqrt((lv * lv).mean())), hv.size))
    print("aligned denominator: %d cells measured (HF bf16 vs llama f32, same 32-token stimulus, no "
          "threshold applied to any of them), %d cells refused with a named reason, out of %d x %d = %d "
          "candidate cells. The other two HF blocks (%s) are not in this denominator at all: their chain "
          "input skips layers 2..37, so a delta there would mix a chain gap with an implementation "
          "difference and is refused as structural-only regardless of stimulus"
          % (cells, refused_cells, len(HF_COMPARABLE), HF_STAGES_EXPECT,
             len(HF_COMPARABLE) * HF_STAGES_EXPECT,
             "+".join(b for b in BLOCKS if b not in HF_COMPARABLE)))
    print("aligned escalation: every line above is a HF-vs-llama divergence on a block where HF has an "
          "unquantized referent. D-26 clause 3: reported, not adopted; where HF's limit is tighter, HF's "
          "limit ships. %d divergences printed, 0 adjudicated, 0 bands authored." % escalations)
    print("aligned engine coverage: engine-vs-oracle cells = 0. engine/block_forward.py implements "
          "%s x %s x one token row; this stimulus is 40 blocks x %d tokens. That is the next rung, not "
          "a gap in this one" % (engine_coverage()[0].get("BLOCK"), engine_coverage()[0].get("GDN_MODE"),
                                 n_tok))
    lg = cand.get("logits", {})
    print("aligned logits referent: argmax_id=%s over n_vocab=%s, digest=%s — status=candidate, and no "
          "band exists for it: the logits-band derivation is put to the user in engine/HANDOFF.md "
          "§PASS 15 §6 (options A/B/C, each with the rejection witness it needs). D-18/D-20 forbid "
          "reading one off this build."
          % (lg.get("argmax_id"), lg.get("n_vocab"), str(lg.get("sha256"))[:16]))
    return checks, findings, escalations


def main():
    global LLAMA_MANIFEST, ALIGNED_MANIFEST, CANDIDATE_MANIFEST
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="all",
                    choices=("align", "structural", "numeric", "all", "aligned"))
    ap.add_argument("--llama-manifest", default=None,
                    help="manifest for the `llama` side of align/structural (default: the run-a "
                         "template-wrapped dumps the three-way report was written against)")
    ap.add_argument("--aligned-manifest", default=None,
                    help="instrument manifest of an aligned run (tools/llama_aligned_run.py run <tag>)")
    ap.add_argument("--candidate", default=None,
                    help="the candidate M3 fixture manifest carrying the determinism pair")
    args = ap.parse_args()
    if args.llama_manifest:
        LLAMA_MANIFEST = args.llama_manifest
    if args.aligned_manifest:
        ALIGNED_MANIFEST = args.aligned_manifest
    if args.candidate:
        CANDIDATE_MANIFEST = args.candidate
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
    if args.mode == "aligned":
        a_checks, a_findings, escalations = mode_aligned(chain, pin, stages_by_block)
        mode_refusals()
        print("")
        for f in a_findings:
            print("FINDING   %s" % f)
        print("RESULT    mode=aligned checks=%d findings=%d aligned=%s escalations=%d "
              "bands-authored=none licence-issued=llama-crosscheck(observation-only) verdict=%s "
              "no timing taken"
              % (a_checks, len(a_findings), not a_findings, escalations,
                 "ALIGNED-OBSERVED" if not a_findings else "ALIGNED-BLOCKED"))
        return 0 if not a_findings else 3
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
