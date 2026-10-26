#!/usr/bin/env python3
"""Independent audit of the D-25 chained-forward fixture.

Deliberately does NOT import tools/dump_chain.py: an artifact that is its own auditor proves
nothing. Everything here is re-derived from bytes on disk and from the safetensors index, and
every check prints the population it measured, because a check that produces no output cannot
fail. Exits non-zero on any FAILED line.
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "checkpoints/dump-manifest-chain-2026-10-26.json"
OLD_MANIFEST = "checkpoints/dump-manifest-2026-10-26.json"
OLD_DIR = "data/dumps/2026-10-26"
INDEX = "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json"
SHARD_DIR = "models/qwen36-src"
OLD_PIN_SHA = "f82d402d574a785b0d310a043ff6f3d8d6a03c19bd4ef46f3fb5b48f1f052ac3"

findings = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, ok, detail):
    print(f"{'OK  ' if ok else 'FAIL'} {name} :: {detail}")
    if not ok:
        findings.append(name)


def records(path):
    with open(path) as fh:
        doc = json.load(fh)
    rows = None
    for key in ("dumps", "records", "tensors", "rows"):
        if isinstance(doc, dict) and key in doc:
            rows = doc[key]
            break
    if rows is None:
        raise SystemExit(f"FAIL no record list in {path}; keys={sorted(doc) if isinstance(doc, dict) else type(doc)}")
    named = [r for r in rows if isinstance(r, dict) and r.get("name")]
    check(f"records_parsed[{os.path.basename(path)}]", len(named) == len(rows),
          f"parsed={len(named)} printed_list={len(rows)}")
    return rows, doc


def main():
    # --- 1. the old pins are still the old pins: this audit must not have written anything ---
    old = []
    for blk in ("layer0", "layer1", "layer39"):
        p = os.path.join(REPO, OLD_DIR, f"{blk}.input.bin")
        old.append((blk, sha(p) if os.path.exists(p) else "MISSING"))
    same = all(s == OLD_PIN_SHA for _, s in old)
    check("old_pins_untouched", same, " ".join(f"{b}={s[:12]}" for b, s in old))

    # --- 2. new manifest: parse, and count what it claims ---
    rows, doc = records(os.path.join(REPO, MANIFEST))
    print(f"MANIFEST records={len(rows)} top_keys={sorted(k for k in doc if isinstance(doc, dict))[:10]}")

    # --- 3. byte-level integrity: every record's file, length and digest, re-hashed here ---
    bad_hash, bad_len, missing = [], [], []
    for r in rows:
        rel = r.get("file") or r.get("path")
        if not rel:
            missing.append(r["name"] + ":no-file-field")
            continue
        p = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        if not os.path.exists(p):
            missing.append(r["name"])
            continue
        if "bytes" in r and os.path.getsize(p) != int(r["bytes"]):
            bad_len.append(f"{r['name']}({os.path.getsize(p)}!={r['bytes']})")
        want = r.get("sha256")
        if want and sha(p) != want:
            bad_hash.append(r["name"])
    check("record_digests_rehashed", not (bad_hash or bad_len or missing),
          f"n={len(rows)} bad_sha={len(bad_hash)} bad_len={len(bad_len)} missing={len(missing)}"
          + (f" first={bad_hash[:2] + bad_len[:2] + missing[:2]}" if (bad_hash or bad_len or missing) else ""))

    # --- 4. chaining verified from files, per block, on the stage pair the manifest names ---
    by_block = defaultdict(dict)
    for r in rows:
        blk = r.get("block") or (r["name"].split(".")[0] if "." in r["name"] else "?")
        stage = r.get("stage") or r["name"].rsplit(".", 1)[-1]
        by_block[blk][stage] = r
    blocks = [b for b in by_block if re.fullmatch(r"layer\d+", b)]
    order = sorted(blocks, key=lambda b: int(b[5:]))
    print(f"BLOCKS order={order} stages_per_block={ {b: sorted(by_block[b]) for b in order[:1]} }")

    pairs, broken = 0, []
    for prev, cur in zip(order, order[1:]):
        po = next((by_block[prev][s] for s in ("out", "output") if s in by_block[prev]), None)
        ci = next((by_block[cur][s] for s in ("x", "in", "input") if s in by_block[cur]), None)
        if po is None or ci is None:
            broken.append(f"{prev}->{cur}:missing-stage")
            continue
        pairs += 1
        a = sha(os.path.join(REPO, po["file"]))
        b = sha(os.path.join(REPO, ci["file"]))
        if a != b:
            broken.append(f"{prev}.out!={cur}.x")
    check("chained_input_equals_previous_output", pairs > 0 and not broken,
          f"pairs={pairs} broken={broken if broken else 'none'}")

    # --- 5. inputs must not have collapsed into one tensor (the defect D-25 was written for) ---
    ins = {}
    for b in order:
        for s in ("x", "in", "input"):
            if s in by_block[b]:
                ins[b] = sha(os.path.join(REPO, by_block[b][s]["file"]))
                break
    distinct = len(set(ins.values()))
    check("inputs_not_collapsed", len(ins) >= 2 and distinct == len(ins),
          f"blocks_with_input={len(ins)} distinct_digests={distinct}")

    # negative control: the SAME predicate over the old per-block fixture must reject it
    old_ins = {}
    for blk in ("layer0", "layer1", "layer39"):
        old_ins[blk] = sha(os.path.join(REPO, OLD_DIR, f"{blk}.input.bin"))
    old_distinct = len(set(old_ins.values()))
    check("negative_control_old_fixture_rejected", old_distinct == 1,
          f"old_blocks={len(old_ins)} old_distinct={old_distinct} (expected 1 -> old set must fail the predicate)")

    # --- 6. the gap must be structural and independently derived, not typed ---
    idx = json.load(open(os.path.join(REPO, INDEX)))
    weight_map = idx.get("weight_map", {})
    indexed_shards = set(weight_map.values())
    have = {f for f in os.listdir(os.path.join(REPO, SHARD_DIR)) if f.endswith(".safetensors")}
    tensors_by_layer = defaultdict(set)
    for t in weight_map:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", t)
        if m:
            tensors_by_layer[int(m.group(1))].add(t)
    all_layers = set(tensors_by_layer)
    complete = set()
    for lyr, ts in tensors_by_layer.items():
        shards = {weight_map[t] for t in ts}
        if shards <= have:
            complete.add(lyr)
    missing_layers = sorted(all_layers - complete)
    print(f"GEOMETRY layers_indexed={len(all_layers)} complete_on_disk={sorted(complete)} "
          f"missing_count={len(missing_layers)} shards_have={len(have)}/indexed={len(indexed_shards)}")
    claimed = doc.get("missing_layers") or doc.get("gap") or doc.get("gap_layers")
    bs = doc.get("block_set") if isinstance(doc.get("block_set"), dict) else {}
    if claimed is None:
        claimed = bs.get("missing_layers")
    check("gap_recorded_structurally", claimed is not None and sorted(claimed) == missing_layers,
          f"manifest_count={None if claimed is None else len(claimed)} derived_count={len(missing_layers)} "
          f"equal={claimed is not None and sorted(claimed) == missing_layers} "
          f"block_set_contiguous={bs.get('contiguous')} why_len={len(str(bs.get('why_not_contiguous', '')))}")
    n_false = sum(1 for r in rows if r.get("contiguous") is False)
    n_true = sum(1 for r in rows if r.get("contiguous") is True)
    check("contiguity_flag_present",
          (n_false + n_true) == len(rows) and n_true == 0 and n_false == len(rows),
          f"records={len(rows)} contiguous_false={n_false} contiguous_true={n_true} "
          f"(a gap exists, so every record must say false and none may claim true)")

    # --- 7. provenance: the dumper must name itself, its command and the interpreter ---
    prov = [k for k in doc if k in ("producing_command", "command", "argv", "producer", "interpreter", "transformers")]
    check("provenance_present", len(prov) >= 2, f"provenance_keys={prov}")

    # --- 8. token ids and embedding seed must be pinned, wherever the producer chose to put them ---
    inp = doc.get("input") if isinstance(doc.get("input"), dict) else {}
    tids = inp.get("token_ids")
    seed_fields = {k: inp[k] for k in ("seed", "tokens", "embedding_slice", "embedding_rows_come_from_the_shard") if k in inp}
    tok_count = len(tids) if isinstance(tids, list) else None
    geom_t = None
    for src in (doc.get("input"), doc.get("counts")):
        if isinstance(src, dict) and isinstance(src.get("tokens"), int):
            geom_t = src["tokens"]
            break
    named_seed = [r["name"] for r in rows if "embed" in r["name"].lower() or "token" in r["name"].lower()]
    ok_seed = bool(tids) and bool(seed_fields) and (geom_t is None or tok_count == geom_t)
    check("embedding_seed_pinned", ok_seed,
          f"token_ids_count={tok_count} expected_tokens={geom_t} seed_fields={sorted(seed_fields)} "
          f"rows_named_embed_or_token={named_seed} (absence is allowed only because the seed is pinned "
          f"in manifest.input, which is why this check reads that field rather than record names)")

    print(f"\nVERDICT audit=findings={len(findings)} records={len(rows)} blocks={len(order)} chain_pairs={pairs}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
