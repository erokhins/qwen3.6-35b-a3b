#!/usr/bin/env python3
"""check_dump_reproduction.py — prove a reference dump is what its manifest says, and that a fresh
dump reproduces the pinned one bit for bit before anyone thinks about replacing it.

Why this file exists and what it is not
---------------------------------------
The pinned dumps under `data/dumps/<date>/` are the oracle the MoE gates are judged against
(`tools/dump_reference.py` writes them, `speccheck/moe_ref.py` computes them, AGENTS.md §Standing
rules makes `speccheck/` the independent implementation). Two failure classes invalidate every
verdict downstream, and neither is visible from the dump directory alone:

  * **the dump drifted.** A hook change in the dumper, a torch/transformers bump, or a re-run after
    a seed change can move bits. A fresh directory must reproduce the pinned one exactly — equal
    names, equal bytes.
  * **the dump is short or extra.** These files are RAW little-endian payloads with no header: a
    `.npy` carries its shape and dtype in the file, a `.bin` carries nothing. Everything a reader
    needs about shape, dtype and element count lives in
    `checkpoints/dump-manifest-2026-10-26.json`. So a missing tensor is invisible in the directory
    listing, and a mislabelled dtype is invisible in the bytes. The manifest is therefore the thing
    under test here, and it is tested against the files, the config, and the model geometry — never
    against a copy of itself.

Every check asserts its own population before it reports anything (`pitfalls/40-gates-and-admissions.md`
§"A check that can produce no output cannot fail"): the declared row count must equal the file count
on disk in both directions, each section prints how many rows it examined, and an empty population is
an error rather than a pass.

Provenance: adapted 2026-10-26 from the 0.8B engine repo's dump-reproduction checker (looked at,
copied in, never cited — `pitfalls/60-docs-and-citations.md`). That version compares two directories
of `.npy` files by digest and treats names present only in the fresh dump as "additions (fine,
reported)". Three things do not transfer, and all three were rewritten:

  1. `.npy` self-describes; these dumps do not, so dtype/shape/element identities are checked here
     against the manifest rather than trusted from a file header.
  2. "additions are fine" is wrong for a pinned oracle: an undeclared file is exactly the shape of
     defect this repo has been bitten by (a directory whose population and manifest disagree).
     Additions are a failure unless `--allow-additions` is passed, and the flag prints what it is
     excusing.
  3. the 0.8B dump has no MoE block, so it has no per-stage population to check. Here the stage set
     (router_logits … combined_out) and the shape identity per stage ARE the geometry check, which is
     what makes this refuse a dump made against a different model rather than printing a table.

Usage
-----
    # audit the committed dump against the committed manifest + config
    python3 tools/check_dump_reproduction.py data/dumps/2026-10-26

    # plus: does a candidate dump reproduce it?
    python3 tools/check_dump_reproduction.py data/dumps/2026-10-26 --fresh tmp/dump_candidate

    python3 tools/check_dump_reproduction.py --selftest

Exit 0 only when every section below holds. CPU only: reads bytes, hashes them, takes no timing and
touches no device (`pitfalls/00-measurement.md`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO, "checkpoints/dump-manifest-2026-10-26.json")
DEFAULT_CONFIG = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")

# torch dtype name -> bytes per element. An unlisted dtype aborts rather than being priced as 0.
ITEMSIZE = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8,
            "torch.int64": 8, "torch.int32": 4, "torch.int16": 2, "torch.int8": 1,
            "torch.uint8": 1, "torch.bool": 1}

# The stages a MoE block dump must contain for a block to count as dumped. Names are the manifest's
# `tensor_key` values, i.e. `tools/dump_reference.py`'s stage names.
REQUIRED_STAGES = ("input", "router_logits", "topk_ids", "topk_weights", "expert_raw",
                   "routed_sum", "shared_raw", "shared_gate_prob", "shared_out", "combined_out",
                   "mlp.gate.weight", "mlp.shared_expert_gate.weight")

# shape identity per stage, as a function of the manifest's own geometry/input block. Anything the
# dumper adds beyond these must be declared with --known-stage or the run aborts: a stage this tool
# cannot shape-check is a stage nobody is checking.
def stage_expect(key, T, G):
    """(shape, legal dtypes) the manifest must declare for tensor_key `key`, or None if unmodellable.

    Why a legal SET per stage rather than one dtype per arm name: the same stage legitimately exists
    as the bf16 reference arm and as a wider container for the same numbers (`topk_weights_f32` is the
    pre-cast value; `layer0.routed_sum_exact` carries tensor_key `routed_sum`). The arm lives in the
    NAME, not in `tensor_key`, so an earlier version that matched arm suffixes on `tensor_key` refused
    the committed dump outright. What IS enforced, and is the check that matters, is (i) the dtype is
    one this stage can legally have, and (ii) where bf16 is legal for a stage, the bf16 reference row
    must actually be present — so a dump that quietly went all-f32 (2x the bytes, a different fixture)
    cannot pass. A cross-check on equal element counts across arms was tried and deleted: given (i) and
    the shape identity below it cannot fire, and a guard that cannot fire is not a guard
    (`pitfalls/40-gates-and-admissions.md`).
    """
    H, E, K, SI = G["hidden_size"], G["num_experts"], G["top_k"], G["shared_expert_intermediate_size"]
    BF, F32, I64 = "torch.bfloat16", "torch.float32", "torch.int64"
    table = {
        "input": ([T, H], {BF}),
        "router_logits": ([T, E], {BF, F32}),
        "topk_ids": ([T, K], {I64}),                 # an index is an index in every arm
        "topk_weights": ([T, K], {BF, F32}),         # f32 = the pre-cast value (README prediction 4)
        "expert_raw": ([T, K, H], {BF, F32}),
        "routed_sum": ([T, H], {BF, F32}),
        "shared_raw": ([T, H], {BF, F32}),
        "shared_out": ([T, H], {BF, F32}),
        "combined_out": ([T, H], {BF, F32}),
        "shared_gate_prob": ([T, 1], {BF, F32}),
        "mlp.gate.weight": ([E, H], {BF}),           # as stored in the bf16 checkpoint
        "mlp.shared_expert_gate.weight": ([1, H], {BF}),
        "mlp.shared_expert.gate_proj.weight": ([SI, H], {BF}),
        "mlp.shared_expert.up_proj.weight": ([SI, H], {BF}),
        "mlp.shared_expert.down_proj.weight": ([H, SI], {BF}),
    }
    if key not in table:
        return None
    shape, dts = table[key]
    return list(shape), dts


class Fatal(Exception):
    pass


def die(fmt, *a):
    raise Fatal(fmt % a if a else fmt)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def audit_manifest(pinned, manifest, config, out, allow_additions=False, known_stages=()):
    """Every witness over the pinned directory + its manifest + the model config."""
    if not os.path.isdir(pinned):
        die("not a directory: %s", pinned)
    if not os.path.isfile(manifest):
        die("manifest %s is absent. The dumps are headerless: without the manifest there is no "
            "shape, dtype or element count to check, so this tool has nothing to adjudicate.",
            manifest)
    m = json.load(open(manifest))
    rows = m.get("dumps")
    if not isinstance(rows, list) or not rows:
        die("manifest %s has %r `dumps`; a census over an empty list prints a clean sheet and "
            "catches nothing", manifest, type(rows).__name__)
    G, I = m.get("geometry"), m.get("input")
    for name, blk in (("geometry", G), ("input", I)):
        if not isinstance(blk, dict) or not blk:
            die("manifest %s has no %r block, so the shape identity has no denominators",
                manifest, name)
    cfg = json.load(open(config))
    t = cfg["text_config"]
    for key, cfgkey in (("num_experts", "num_experts"), ("top_k", "num_experts_per_tok"),
                        ("moe_intermediate_size", "moe_intermediate_size"),
                        ("shared_expert_intermediate_size", "shared_expert_intermediate_size"),
                        ("hidden_size", "hidden_size"), ("num_hidden_layers", "num_hidden_layers"),
                        ("mtp_num_hidden_layers", "mtp_num_hidden_layers")):
        if key not in G:
            die("manifest geometry has no %r", key)
        if G[key] != t[cfgkey]:
            die("manifest geometry says %s=%s, the config %s says %s. The dump was made against a "
                "different model (or a different config file): every band sized from this dump would "
                "be sized against the wrong geometry.", key, G[key], os.path.relpath(config, REPO),
                t[cfgkey])
    if G.get("config") and os.path.basename(str(G["config"])) != os.path.basename(config):
        out("   note: manifest geometry.config names %s; this run was told %s"
            % (os.path.basename(str(G["config"])), os.path.relpath(config, REPO)))
    T = int(I["n_tokens"])
    if T <= 0:
        die("manifest input.n_tokens = %r", I["n_tokens"])

    files = sorted(f for f in os.listdir(pinned) if f.endswith(".bin"))
    others = [f for f in os.listdir(pinned) if not f.endswith(".bin")]
    if others:
        die("%s holds %d non-.bin file(s) (%s). The dumper writes .bin plus the manifest only; an "
            "extra file type means something else wrote into the pinned directory.",
            pinned, len(others), ", ".join(sorted(others)[:4]))
    declared = {}
    for r in rows:
        if "name" not in r or "file" not in r:
            die("manifest row without name/file: %r", sorted(r)[:6])
        if r["name"] in declared:
            die("manifest declares %s twice", r["name"])
        declared[r["name"]] = r
    on_disk = {f[:-4] for f in files}
    missing = sorted(set(declared) - on_disk)
    extra = sorted(on_disk - set(declared))
    out("== population")
    out("   manifest rows                     %d" % len(rows))
    out("   .bin files on disk                %d" % len(files))
    if missing:
        die("%d tensor(s) the manifest declares are absent from %s (e.g. %s). A headerless dump "
            "whose population silently shrank looks exactly like a complete one in a directory "
            "listing.", len(missing), pinned, ", ".join(missing[:3]))
    if extra:
        if not allow_additions:
            die("%d .bin file(s) on disk are not declared by the manifest (e.g. %s). Pass "
                "--allow-additions only when the dumper legitimately gained stages, and say which.",
                len(extra), ", ".join(extra[:3]))
        out("   UNDECLARED, excused by --allow-additions: %d (%s)"
            % (len(extra), ", ".join(extra[:4])))
    out("   declared == present               OK (%d == %d, both directions)"
        % (len(declared), len(on_disk)))

    blocks = sorted({r.get("block") for r in rows if r.get("block")})
    want_blocks = G.get("blocks_dumped")
    if not want_blocks:
        die("manifest geometry has no blocks_dumped, so per-block population cannot be checked")
    if sorted(want_blocks) != blocks:
        die("geometry.blocks_dumped = %s but the rows cover blocks %s",
            sorted(want_blocks), blocks)
    out("   blocks                            %s" % ", ".join(blocks))
    for b in blocks:
        have = {r.get("tensor_key") for r in rows if r.get("block") == b}
        short = [s for s in REQUIRED_STAGES if s not in have]
        if short:
            die("block %s is missing required stage(s) %s. A dump that lost a stage silently turns "
                "the gate that reads it into a check over nothing.", b, ", ".join(short))
    out("   required stages per block         OK (%d stages x %d blocks)"
        % (len(REQUIRED_STAGES), len(blocks)))

    out("== per-tensor identities (bytes == elements x itemsize, shape x dtype == the geometry)")
    unknown_keys = set()
    n_bytes = 0
    core_seen = {}
    for r in rows:
        name, rel = r["name"], r["file"]
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        if not os.path.isfile(path):
            path = os.path.join(pinned, os.path.basename(rel))
        if not os.path.isfile(path):
            die("manifest row %s points at %s, which is not a file", name, rel)
        dt = r.get("dtype")
        if dt not in ITEMSIZE:
            die("row %s declares dtype %r, which this tool has no itemsize for. Refusing to treat "
                "it as 0 bytes; add it to ITEMSIZE only with the torch name spelled exactly.",
                name, dt)
        shape, elements, nbytes = r.get("shape"), r.get("elements"), r.get("bytes")
        if not isinstance(shape, list) or not shape:
            die("row %s declares shape %r", name, shape)
        if numel(shape) != elements:
            die("row %s declares shape %s (%d elements) but says elements=%s. One of the two is a "
                "typo and the reader cannot tell which", name, shape, numel(shape), elements)
        if elements * ITEMSIZE[dt] != nbytes:
            die("row %s: elements %d x itemsize %d (%s) = %d, but the manifest declares bytes=%s. "
                "The dtype label and the payload disagree.", name, elements, ITEMSIZE[dt], dt,
                elements * ITEMSIZE[dt], nbytes)
        real = os.path.getsize(path)
        if real != nbytes:
            die("row %s: file %s is %d bytes, the manifest declares %d. Truncated, or overwritten by "
                "a run with different geometry.", name, os.path.basename(path), real, nbytes)
        want = stage_expect(r.get("tensor_key", ""), T, G)
        if want is None:
            unknown_keys.add(r.get("tensor_key"))
        else:
            wshape, wdts = want
            if list(shape) != list(wshape):
                die("row %s: stage %r at n_tokens=%d must be %s by the manifest's own geometry "
                    "(H=%s, E=%s, top_k=%s); the manifest says %s. The dump does not have the shape "
                    "the geometry implies.", name, r.get("tensor_key"), T, wshape,
                    G["hidden_size"], G["num_experts"], G["top_k"], shape)
            if dt not in wdts:
                die("row %s: stage %r may only be %s; the manifest says %s",
                    name, r.get("tensor_key"), " or ".join(sorted(wdts)), dt)
            if r.get("tensor_key") in REQUIRED_STAGES:
                seen = core_seen.setdefault((r.get("block"), r.get("tensor_key")),
                                            {"legal": set(wdts), "have": set()})
                seen["have"].add(dt)
        want_sha = r.get("sha256")
        if not want_sha or len(want_sha) != 64:
            die("row %s has sha256 %r; a pinned dump without a digest cannot be reproduced or "
                "distinguished", name, want_sha)
        got = sha256(path)
        if got != want_sha:
            die("row %s: sha256 mismatch. manifest %s… file %s… — the file is not the bytes the "
                "manifest pinned.", name, want_sha[:12], got[:12])
        n_bytes += nbytes
    if unknown_keys and not known_stages:
        die("the manifest declares stage(s) %s that this tool cannot shape-check. Either add them to "
            "stage_expect() with their geometry-derived shape, or pass --known-stage for each one and "
            "say why; an unchecked stage is not a checked one.", ", ".join(sorted(map(str, unknown_keys))))
    # The bf16 reference arm must actually be there. This is the check a directory listing cannot
    # make: a stage can be present in every block, with every file declared, and still have lost its
    # bf16 row so only the f32 arm remains — a different fixture with twice the bytes. It applies only
    # where bf16 is a legal dtype for the stage: `topk_ids` is int64 in every arm, and demanding a
    # bf16 row there would refuse the committed dump for carrying indices in the right container.
    for (block, stage), seen in sorted(core_seen.items()):
        if "torch.bfloat16" in seen["legal"] and "torch.bfloat16" not in seen["have"]:
            die("block %s stage %r has no bf16 reference arm (%s). The bf16 reference is the "
                "comparison of record (`pitfalls/30-numerics.md`); a dump whose only arm is f32 is a "
                "different fixture with twice the bytes, not a variation on this one.",
                block, stage, ", ".join(sorted(seen["have"])))
    out("   bf16 reference arm present for every required stage  OK (%d block x stage pairs)"
        % len(core_seen))
    out("   dtypes known, shapes and digests  OK (%d rows, %s bytes hashed)"
        % (len(rows), format(n_bytes, ",")))
    return m, n_bytes


def compare_fresh(pinned, fresh, manifest, out, allow_additions=False):
    """Byte-for-byte reproduction: same names, same digests, and the non-volatile manifest fields."""
    if not os.path.isdir(fresh):
        die("not a directory: %s", fresh)
    m = json.load(open(manifest))
    pinned_rows = {r["name"]: r["sha256"] for r in m["dumps"]}
    new_files = {f[:-4] for f in os.listdir(fresh) if f.endswith(".bin")}
    others = [f for f in os.listdir(fresh) if not f.endswith(".bin")
              and f not in ("manifest.json",)]
    if others:
        die("%s holds non-.bin files (%s); the dumper writes .bin only", fresh, ", ".join(others[:4]))
    missing = sorted(set(pinned_rows) - new_files)
    extra = sorted(new_files - set(pinned_rows))
    out("== reproduction against the pinned dump")
    out("   fresh .bin files                  %d (pinned declares %d)", len(new_files), len(pinned_rows))
    if missing:
        die("fresh dump is missing %d pinned tensor(s) (e.g. %s)", len(missing), ", ".join(missing[:3]))
    if extra and not allow_additions:
        die("fresh dump adds %d undeclared tensor(s) (e.g. %s). New stages are a decision, not a "
            "side effect — land them with the manifest change that declares them.",
            len(extra), ", ".join(extra[:3]))
    differ = []
    for name in sorted(pinned_rows):
        if sha256(os.path.join(fresh, name + ".bin")) != pinned_rows[name]:
            differ.append(name)
    if differ:
        die("%d of %d tensors differ from the pinned dump (e.g. %s). A fresh run must reproduce the "
            "pinned bits; if the reference legitimately moved, the pin is re-taken as a decision, not "
            "overwritten by a gate.", len(differ), len(pinned_rows), ", ".join(differ[:3]))
    out("   every pinned tensor reproduced    OK (%d digests equal)" % len(pinned_rows))
    fp = os.path.join(fresh, "manifest.json")
    if os.path.isfile(fp):
        # Volatile fields are named, not guessed: paths, digests of the environment and the record of
        # when it ran are expected to differ; anything carrying numbers is not.
        VOLATILE = {"producing_command", "date"}
        nf = json.load(open(fp))
        bad = sorted(k for k in set(m) | set(nf)
                     if k not in VOLATILE and json.dumps(m.get(k), sort_keys=True)
                     != json.dumps(nf.get(k), sort_keys=True))
        if bad:
            die("fresh manifest fields differ from the pinned one: %s. Those fields carry the "
                "geometry, the input seed and the reading rules — a dump made under different ones is "
                "a different fixture.", ", ".join(bad))
        out("   manifest fields (minus %s)     OK" % ", ".join(sorted(VOLATILE)))
    else:
        out("   no manifest.json in the fresh dir: fields not compared (the file comparison above "
            "still stands)")


# ---------------------------------------------------------------------------------------------
# selftest — synthetic tiny dump, faults injected one at a time, each must fire by name.
# ---------------------------------------------------------------------------------------------
def synth(root, n_tokens=3, G=None):
    G = G or dict(num_experts=4, top_k=2, moe_intermediate_size=8,
                  shared_expert_intermediate_size=8, hidden_size=16, num_hidden_layers=2,
                  mtp_num_hidden_layers=1, blocks_dumped=["layer0", "mtp0"])
    os.makedirs(root, exist_ok=True)
    rows = []
    payloads = {}

    def add(block, key, shape, dt, fill, suffix=""):
        n = numel(shape)
        blob = bytes([fill % 251] * (n * ITEMSIZE[dt]))
        name = "%s.%s%s" % (block, key, suffix)
        payloads[name] = blob
        rows.append(dict(file=os.path.join(root, name + ".bin"), bytes=len(blob), shape=list(shape),
                         dtype=dt, elements=n, sha256=hashlib.sha256(blob).hexdigest(),
                         name=name, tensor_key=key, block=block,
                         arm="ref" if not suffix else suffix.lstrip("_")))

    for b in G["blocks_dumped"]:
        for s in REQUIRED_STAGES:
            shape, dts = stage_expect(s, n_tokens, G)
            add(b, s, shape, sorted(dts)[0], len(s))
        for s in ("router_logits", "expert_raw", "routed_sum", "combined_out", "topk_weights"):
            shape, dts = stage_expect(s, n_tokens, G)
            # the arm spelling the dumper actually uses: `tensor_key` carries the STAGE and `name`
            # carries the arm (layer0.topk_weights_f32 is tensor_key `topk_weights`), which is why
            # the dtype rule is a legal set per stage and not a dtype per arm name.
            add(b, s, shape, "torch.float32", 7, suffix="_exact")
        for s in ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                  "mlp.shared_expert.down_proj.weight"):
            shape, dts = stage_expect(s, n_tokens, G)
            add(b, s, shape, sorted(dts)[0], 11)
    for name, blob in payloads.items():
        with open(os.path.join(root, name + ".bin"), "wb") as fh:
            fh.write(blob)
    man = dict(manifest_version=1, date="2026-10-26", producing_command="synthetic",
               geometry=dict(G), input=dict(n_tokens=n_tokens, seed=1, dtype="bfloat16"),
               dumps=rows)
    # The manifest lives OUTSIDE the dump directory, exactly as the committed one does
    # (`checkpoints/dump-manifest-2026-10-26.json` for `data/dumps/2026-10-26/`): a pinned dump dir
    # holding a stray file type is itself a defect the tool must be able to see.
    mp = root + ".manifest.json"
    json.dump(man, open(mp, "w"))
    return man


def selftest(root):
    cfg = json.load(open(DEFAULT_CONFIG))
    t = cfg["text_config"]
    G = dict(num_experts=t["num_experts"], top_k=t["num_experts_per_tok"],
             moe_intermediate_size=t["moe_intermediate_size"],
             shared_expert_intermediate_size=t["shared_expert_intermediate_size"],
             hidden_size=t["hidden_size"], num_hidden_layers=t["num_hidden_layers"],
             mtp_num_hidden_layers=t["mtp_num_hidden_layers"],
             blocks_dumped=["layer0", "mtp0"])
    n_tokens = 3
    fired, missed = [], []

    def build():
        if os.path.isdir(root):
            shutil.rmtree(root)
        if os.path.isfile(root + ".manifest.json"):
            os.remove(root + ".manifest.json")
        synth(root, n_tokens=n_tokens, G=G)
        return root + ".manifest.json"

    def expect(label, fn, needle=None):
        try:
            fn()
        except Fatal as e:
            msg = " ".join(str(e).split())
            if needle and needle.lower() not in msg.lower():
                fired.append((label, msg[-120:]))
                print("  WRONG REASON %-36s ...%s" % (label, msg[-120:]))
                missed.append(label + " (wrong reason: wanted %r)" % needle)
                return
            fired.append((label, msg[-120:]))
            print("  FIRED   %-40s ...%s" % (label, msg[-118:]))
            return
        missed.append(label)
        print("  NOT FIRED %-38s the fault ran clean — the guard is decorative" % label)

    print("check_dump_reproduction --selftest: synthetic dump (2 blocks x %d stages), faults injected"
          % len(REQUIRED_STAGES))
    mp = build()
    audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None)
    fresh = root + "_fresh"
    if os.path.isdir(fresh):
        shutil.rmtree(fresh)
    shutil.copytree(root, fresh)
    compare_fresh(root, fresh, mp, lambda *a, **k: None)
    shutil.copyfile(mp, os.path.join(fresh, "manifest.json"))
    compare_fresh(root, fresh, mp, lambda *a, **k: None)
    print("  CONTROL   clean synthetic dump audits and reproduces exactly, with and without a fresh "
          "manifest present (%d rows)" % len(json.load(open(mp))["dumps"]))

    def touch(base_dir, name, mutate):
        p = os.path.join(base_dir, name + ".bin")
        with open(p, "r+b") as fh:
            data = bytearray(fh.read())
            mutate(data)
            fh.seek(0)
            fh.write(bytes(data))

    expect("one byte flipped in the fresh dump",
           lambda: (touch(fresh, "layer0.routed_sum", lambda d: d.__setitem__(3, d[3] ^ 0xFF)),
                    compare_fresh(root, fresh, mp, lambda *a, **k: None))[1],
           "differ from the pinned dump")
    shutil.copyfile(os.path.join(root, "layer0.routed_sum.bin"),
                    os.path.join(fresh, "layer0.routed_sum.bin"))

    expect("the pinned file itself drifted from its manifest",
           lambda: (touch(root, "layer0.routed_sum", lambda d: d.__setitem__(5, d[5] ^ 0xFF)),
                    audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None))[1],
           "sha256 mismatch")
    mp = build()

    expect("a pinned .bin deleted",
           lambda: (os.remove(os.path.join(root, "layer0.combined_out.bin")),
                    audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None))[1],
           "absent from")
    mp = build()

    open(os.path.join(root, "layer0.unexpected.bin"), "wb").write(b"\0" * 8)
    expect("an undeclared .bin in the pinned dir",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "not declared by the manifest")
    os.remove(os.path.join(root, "layer0.unexpected.bin"))

    m = json.load(open(mp))
    m["dumps"][0]["sha256"] = "0" * 64
    json.dump(m, open(mp, "w"))
    expect("manifest digest rewritten", lambda: audit_manifest(root, mp, DEFAULT_CONFIG,
                                                               lambda *a, **k: None), "sha256 mismatch")
    mp = build()

    m = json.load(open(mp))
    r = m["dumps"][0]
    r["bytes"] = r["bytes"] + 2
    json.dump(m, open(mp, "w"))
    expect("row bytes inconsistent with elements x itemsize",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "The dtype label and the payload disagree")
    mp = build()

    m = json.load(open(mp))
    m["geometry"]["num_experts"] = t["num_experts"] // 2
    json.dump(m, open(mp, "w"))
    expect("manifest geometry disagrees with the config",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "different model")
    mp = build()

    m = json.load(open(mp))
    for row in m["dumps"]:
        if row["tensor_key"] == "shared_gate_prob":
            # the stage whose shape is [T, 1]: widening it to [T, 2] is geometrically impossible for
            # one sigmoid per token, and the file + digest are rewritten so the ONLY witness left
            # able to fire is the shape-vs-geometry identity.
            row["shape"] = [n_tokens, 2]
            row["elements"] = numel(row["shape"])
            row["bytes"] = row["elements"] * ITEMSIZE[row["dtype"]]
            blob = b"\x01" * row["bytes"]
            open(os.path.join(root, row["name"] + ".bin"), "wb").write(blob)
            row["sha256"] = hashlib.sha256(blob).hexdigest()
    json.dump(m, open(mp, "w"))
    expect("a stage's declared shape contradicts the geometry",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "does not have the shape the geometry implies")
    mp = build()

    m = json.load(open(mp))
    keep, dropped = [], []
    for row in m["dumps"]:
        if row["tensor_key"] == "shared_gate_prob":
            dropped.append(row["name"])
        else:
            keep.append(row)
    m["dumps"] = keep
    for name in dropped:
        os.remove(os.path.join(root, name + ".bin"))
    json.dump(m, open(mp, "w"))
    expect("a required stage silently lost (rows and files, consistently)",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "missing required stage")
    mp = build()

    m = json.load(open(mp))
    for r in m["dumps"]:
        if r["tensor_key"] == "topk_ids":
            r["dtype"] = "torch.bfloat16"
            r["bytes"] = r["elements"] * 2
            blob = bytes(r["bytes"])
            open(os.path.join(root, r["name"] + ".bin"), "wb").write(blob)
            r["sha256"] = hashlib.sha256(blob).hexdigest()
    json.dump(m, open(mp, "w"))
    expect("topk_ids relabelled int64 -> bfloat16",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "may only be torch.int64")
    mp = build()

    m = json.load(open(mp))
    keep, gone = [], []
    for row in m["dumps"]:
        if row["tensor_key"] == "routed_sum" and row["dtype"] == "torch.bfloat16":
            gone.append(row["name"])
        else:
            keep.append(row)
    m["dumps"] = keep
    for name in gone:
        os.remove(os.path.join(root, name + ".bin"))
    json.dump(m, open(mp, "w"))
    expect("the bf16 reference arm of a stage deleted, only its f32 arm left",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "no bf16 reference arm")
    mp = build()

    m = json.load(open(mp))
    for r in m["dumps"]:
        if r["tensor_key"] == "topk_ids":
            r["dtype"] = "torch.float8_e4m3fn"
    json.dump(m, open(mp, "w"))
    expect("an unknown dtype spelling", lambda: audit_manifest(root, mp, DEFAULT_CONFIG,
                                                               lambda *a, **k: None),
           "no itemsize for")
    mp = build()

    m = json.load(open(mp))
    for r in m["dumps"]:
        if r["tensor_key"] == "routed_sum":
            # the shape claims one row more than the payload holds, while elements/bytes/file stay
            # mutually consistent — so `numel(shape) == elements` is the only lie left to catch.
            r["shape"] = [n_tokens + 1, G["hidden_size"]]
    json.dump(m, open(mp, "w"))
    expect("shape does not multiply out to its element count",
           lambda: audit_manifest(root, mp, DEFAULT_CONFIG, lambda *a, **k: None),
           "One of the two is a typo")
    mp = build()

    m = json.load(open(mp))
    m["input"]["seed"] = 999
    json.dump(m, open(os.path.join(fresh, "manifest.json"), "w"))
    expect("fresh manifest re-seeded (a different fixture, not a reproduction)",
           lambda: compare_fresh(root, fresh, mp, lambda *a, **k: None), "fields differ")
    shutil.copyfile(mp, os.path.join(fresh, "manifest.json"))
    mp = build()

    m = json.load(open(mp))
    tmpl = m["dumps"][0]
    key, shape, dt = "mystery_stage", [n_tokens, G["hidden_size"]], "torch.bfloat16"
    blob = b"\x02" * (numel(shape) * ITEMSIZE[dt])
    name = "layer0.%s" % key
    open(os.path.join(root, name + ".bin"), "wb").write(blob)
    m["dumps"].append(dict(file=os.path.join(root, name + ".bin"), bytes=len(blob), shape=shape,
                           dtype=dt, elements=numel(shape),
                           sha256=hashlib.sha256(blob).hexdigest(), name=name, tensor_key=key,
                           block="layer0", arm="ref"))
    json.dump(m, open(mp, "w"))
    expect("a stage this tool cannot shape-check", lambda: audit_manifest(root, mp, DEFAULT_CONFIG,
                                                                          lambda *a, **k: None),
           "cannot shape-check")
    mp = build()

    m = json.load(open(mp))
    m["geometry"]["blocks_dumped"] = ["layer0", "layer1", "mtp0"]
    json.dump(m, open(mp, "w"))
    expect("blocks_dumped over-promises", lambda: audit_manifest(root, mp, DEFAULT_CONFIG,
                                                                 lambda *a, **k: None),
           "blocks_dumped")
    mp = build()

    if missed:
        print("\nSELFTEST FAIL: %d of %d faults did not fire for the right reason: %s"
              % (len(missed), len(missed) + len(fired), ", ".join(missed)))
        return 1
    print("\nSELFTEST PASS: %d injected faults each fired at its own magnitude" % len(fired))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pinned", nargs="?", help="directory of .bin dumps (data/dumps/<date>)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--fresh", help="a candidate dump to compare against the pinned one")
    ap.add_argument("--allow-additions", action="store_true",
                    help="excuse undeclared files, printing what it excuses")
    ap.add_argument("--known-stage", action="append", default=[],
                    help="declare a stage as intentionally not shape-checked (repeatable)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(os.path.join(REPO, "tmp", "w17_dump_selftest"))
    if not a.pinned:
        ap.error("a pinned dump directory is required (or --selftest)")
    try:
        audit_manifest(a.pinned, a.manifest, a.config, print, allow_additions=a.allow_additions,
                       known_stages=set(a.known_stage))
        if a.fresh:
            compare_fresh(a.pinned, a.fresh, a.manifest, print,
                          allow_additions=a.allow_additions)
    except Fatal as e:
        print("REFUSED: %s" % e)
        return 1
    print("\nPASS: the dump agrees with its manifest, the manifest with the config, and every byte "
          "hashed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
