#!/usr/bin/env python3
"""tools/checkpoint_integrity_d29.py — tie `models/qwen36-src/` to the published Hub checkpoint.

DECISIONS.md D-29's integrity obligation, verbatim in intent: "All 26 shard digests get recorded
and verified against the Hub's LFS etags in a new manifest before any HF-derived fixture is
trusted — a fixture whose bytes cannot be tied to the published checkpoint is a liability, not a
reference. The 4 shards already on disk are re-hashed under the same manifest rather than
grandfathered."

This file imports NEITHER the producer (`tools/dump_chain.py`, `tools/hf_full_forward_d29.py`) NOR
the consumer (`engine/*`), following `tools/verify_chain_d25.py`: an artifact that is its own
auditor proves nothing. Everything is re-derived from bytes on disk plus one read of the Hub tree.

Read-only with respect to `models/`: every open is `"rb"`, and `refuse_write_into_models()` guards
the only writes this file performs (the manifest, the scratch witness tree under /tmp).

CPU-only. No device, no HIP, no device lock of any kind (this file names no lock file), no device
clock.
The only clocks are host wall-clock, and they are printed as `host_wall_s` — a host figure, not a
rate. **no timing taken (device).**

MODES
  hub-tree            fetch + cache the Hub tree (the only network call this file makes)
  build               hash all 26 shards, parse every safetensors header, write the manifest ONCE
  verify [--dir D]    re-hash and re-derive against a manifest (offline; --dir points at a copy
                      for the perturbation witness)
  witness             non-vacuity: perturb a shard copy in scratch and delete one row from a
                      manifest copy, and FAIL if the verifier does NOT redden on the named shard

The etag semantics, recorded because they are subtle: for an LFS-tracked file the Hub's etag is
`tree[].lfs.oid`, which is the sha256 of the file's bytes, so `etag_equals_sha256` is a real
equality and not a platitude; for a small git-tracked file the etag is the git blob sha1, which is
a different hash function, so this file asserts `local_sha1 == hub_oid` there and refuses to call
it a sha256 match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD_DIR = os.path.join(REPO, "models", "qwen36-src")
INDEX_LOCAL = os.path.join(SHARD_DIR, "model.safetensors.index.json")
INDEX_PIN = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.safetensors.index.json")
MANIFEST = os.path.join(REPO, "checkpoints", "full-checkpoint-manifest-2026-10-27.json")
HUB_CACHE = os.path.join(REPO, "specification", "evidence",
                         "full-checkpoint-hub-tree-2026-10-27.json")
HUB_REPO = "Qwen/Qwen3.6-35B-A3B"
HUB_TREE_URL = f"https://huggingface.co/api/models/{HUB_REPO}/tree/main?recursive=true"
EXPECTED_SHARDS = 26
SCRATCH = "/tmp/d29-integrity-witness"

findings: list[str] = []


# ---------------------------------------------------------------- primitives
def sha_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, ok, detail):
    print(f"{'OK  ' if ok else 'FAIL'} {name} :: {detail}")
    if not ok:
        findings.append(name)


def refuse_write_into_models(path):
    real = os.path.realpath(path)
    root = os.path.realpath(os.path.join(REPO, "models"))
    if real == root or real.startswith(root + os.sep):
        raise SystemExit(f"REFUSED tag=write-guard tried to write inside models/: {path}")


def safetensors_header(path):
    """Parse a safetensors file's own header. Returns (header_json_bytes, {tensor: nbytes}).

    A corrupted header raises rather than returning junk: the witness that flips a byte can land
    in the header, and a verifier that crashes instead of reddening that shard by name is a check
    that cannot fail on purpose."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n > (1 << 26):
            raise ValueError(f"implausible header length {n}")
        raw = fh.read(n)
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"header unreadable: {exc}") from None
    sizes = {k: int(v["data_offsets"][1]) - int(v["data_offsets"][0])
             for k, v in hdr.items() if k != "__metadata__"}
    dtypes = {v.get("dtype") for k, v in hdr.items() if k != "__metadata__"}
    return {"header_bytes": 8 + n, "n_tensors": len(sizes),
            "tensor_bytes": sum(sizes.values()), "dtypes": sorted(d for d in dtypes if d),
            "header_error": None}, sizes


def hub_tree(cache_only=False):
    """The Hub tree, with pagination followed. Cached to HUB_CACHE so `verify` stays offline."""
    if os.path.exists(HUB_CACHE) and cache_only:
        doc = json.load(open(HUB_CACHE))
        return doc["entries"]
    entries, url, pages = [], HUB_TREE_URL, 0
    while url:
        pages += 1
        req = urllib.request.Request(url, headers={"User-Agent": "qwen36-d29-integrity/1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            batch = json.load(resp)
            link = resp.headers.get("Link", "")
        entries.extend(batch)
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1:part.find(">")]
        if pages > 20:
            raise SystemExit("REFUSED tag=hub-pagination exceeded 20 pages")
    lfs = sum(1 for e in entries if isinstance(e.get("lfs"), dict))
    if not entries:
        raise SystemExit("REFUSED tag=hub-empty-tree the API returned nothing")
    os.makedirs(os.path.dirname(HUB_CACHE), exist_ok=True)
    refuse_write_into_models(HUB_CACHE)
    with open(HUB_CACHE, "w") as fh:
        json.dump({"repo": HUB_REPO, "url": HUB_TREE_URL, "pages_followed": pages,
                   "entries_printed": len(entries), "lfs_tracked": lfs, "entries": entries}, fh,
                  indent=1, sort_keys=True)
    print(f"hub    entries={len(entries)} pages={pages} lfs_tracked={lfs} cached={HUB_CACHE}")
    return entries


def hub_index(entries):
    out = {}
    for e in entries:
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs") if isinstance(e.get("lfs"), dict) else None
        out[e["path"]] = {
            "hub_etag": (lfs or e)["oid"],
            "hub_etag_kind": "sha256-lfs" if lfs else "sha1-git-blob",
            "hub_size": int(e["size"]),
            "git_blob_sha1": e["oid"],
            "xet_hash": e.get("xetHash"),
        }
    return out


# ---------------------------------------------------------------- the derived facts
def derive(dirpath):
    """Everything derivable from the directory itself: shards on disk, their headers, the index."""
    shards = sorted(f for f in os.listdir(dirpath) if f.endswith(".safetensors"))
    per_shard = {}
    header_errors = []
    for name in shards:
        try:
            hdr, sizes = safetensors_header(os.path.join(dirpath, name))
        except ValueError as exc:
            header_errors.append(f"{name}({exc})")
            hdr = {"header_bytes": 0, "n_tensors": 0, "tensor_bytes": 0,
                   "dtypes": ["HEADER-UNREADABLE"], "header_error": str(exc)}
            sizes = {}
        per_shard[name] = (hdr, sizes)
    print(f"derive  shards_on_disk={len(shards)} header_errors={len(header_errors)}"
          f"{header_errors[:3]}")
    index = json.load(open(os.path.join(dirpath, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    index_shards = sorted(set(weight_map.values()))
    tensors_by_layer, families = {}, {}
    for t in weight_map:
        fam = ("visual" if t.startswith("model.visual.") or ".visual." in t
               else "mtp" if t.startswith("mtp.") else "text")
        families[fam] = families.get(fam, 0) + 1
        parts = t.split(".")
        layer = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer = int(parts[i + 1])
                if parts[0] == "mtp":
                    layer = None
                break
        if layer is not None:
            tensors_by_layer.setdefault(layer, set()).add(t)
    seen = {}
    for name, (_, sizes) in per_shard.items():
        for t in sizes:
            seen[t] = name
    return {"shards": shards, "per_shard": per_shard, "weight_map": weight_map,
            "index_shards": index_shards, "index_total_size": index["metadata"]["total_size"],
            "tensors_by_layer": tensors_by_layer, "families": families, "seen": seen,
            "header_errors": header_errors}


def count_identities(d, shard_rows, dirpath):
    """Every count identity, printed with its operands. A bare 0 is vacuous and will not be
    accepted here: each identity prints expected, seen and the derived difference."""
    exp_shards = set(d["index_shards"])
    got_shards = set(d["shards"])
    listed = {r["file"] for r in shard_rows}
    listed_shards = {f for f in listed if f.endswith(".safetensors")}
    n_exp = len(exp_shards)
    print(f"count   shards_index={n_exp} shards_on_disk={len(got_shards)} "
          f"shards_listed={len(listed_shards)} expected_constant={EXPECTED_SHARDS}")
    check("shard_count_equals_index", len(got_shards) == n_exp and n_exp == EXPECTED_SHARDS,
          f"on_disk={len(got_shards)} index={n_exp} constant={EXPECTED_SHARDS} "
          f"difference_on_disk_minus_index={len(got_shards) - n_exp}")
    only_idx = sorted(exp_shards - got_shards)
    only_disk = sorted(got_shards - exp_shards)
    check("shard_sets_identical", not only_idx and not only_disk,
          f"in_index_not_on_disk={len(only_idx)}{only_idx[:3]} on_disk_not_in_index="
          f"{len(only_disk)}{only_disk[:3]}")
    unlisted = sorted(exp_shards - listed)
    extra_listed = sorted(listed_shards - exp_shards)
    check("manifest_lists_exactly_the_index_shards", not unlisted and not extra_listed,
          f"index={n_exp} listed={len(listed_shards)} unlisted={len(unlisted)}{unlisted[:3]} "
          f"listed_extra={len(extra_listed)}{extra_listed[:3]} "
          f"identity: {n_exp} - {len(listed_shards)} = {n_exp - len(listed_shards)}")

    expected_tensors = set(d["weight_map"])
    seen_tensors = set(d["seen"])
    missing = sorted(expected_tensors - seen_tensors)
    extra = sorted(seen_tensors - expected_tensors)
    n_exp_t, n_seen_t = len(expected_tensors), len(seen_tensors)
    check("tensor_count_identity", not missing and not extra and n_exp_t == n_seen_t,
          f"tensor_expected={n_exp_t} tensor_seen={n_seen_t} "
          f"missing = {n_exp_t} - |seen in shards| = {n_exp_t - len(seen_tensors)} "
          f"in_index_not_in_shards={len(missing)}{missing[:3]} in_shards_not_in_index={len(extra)}"
          f"{extra[:3]}")

    data_bytes = sum(h["tensor_bytes"] for h, _ in d["per_shard"].values())
    want_bytes = int(d["index_total_size"])
    check("data_bytes_equal_index_total_size", data_bytes == want_bytes,
          f"sum_of_shard_tensor_bytes={data_bytes} index.metadata.total_size={want_bytes} "
          f"delta={data_bytes - want_bytes}")
    hdr_bytes = sum(h["header_bytes"] for h, _ in d["per_shard"].values())
    file_bytes = sum(os.path.getsize(os.path.join(dirpath, f)) for f in d["shards"])
    bad_size = [f for f in d["shards"]
                if os.path.getsize(os.path.join(dirpath, f))
                != d["per_shard"][f][0]["header_bytes"] + d["per_shard"][f][0]["tensor_bytes"]]
    check("file_bytes_identity", not bad_size,
          f"per shard: filesize == header_bytes + tensor_bytes; violations={len(bad_size)}"
          f"{bad_size[:3]} total_header_bytes={hdr_bytes} total_file_bytes={file_bytes} "
          f"tensor_bytes={data_bytes} identity: {file_bytes} - {hdr_bytes} = "
          f"{file_bytes - hdr_bytes} (must equal tensor_bytes {data_bytes})")

    layers_indexed = sorted(d["tensors_by_layer"])
    complete = sorted(lyr for lyr, ts in d["tensors_by_layer"].items()
                      if {d["weight_map"][t] for t in ts} <= set(d["shards"]))
    missing_layers = sorted(set(layers_indexed) - set(complete))
    check("layers_complete", len(layers_indexed) == 40 and not missing_layers,
          f"layers_indexed={len(layers_indexed)} complete={len(complete)} "
          f"missing_layers = {len(layers_indexed)} - {len(complete)} = {len(missing_layers)}"
          f"{missing_layers[:5]}")
    fam = d["families"]
    check("family_partition_covers_the_index", sum(fam.values()) == n_exp_t,
          f"families={fam} sum={sum(fam.values())} index_tensors={n_exp_t}")
    return {"n_expected_tensors": n_exp_t, "n_seen_tensors": n_seen_t,
            "data_bytes": data_bytes, "header_bytes": hdr_bytes,
            "layers_indexed": len(layers_indexed), "layers_complete": len(complete),
            "missing_layers": missing_layers, "families": fam,
            "tensors_per_shard": {n: h["n_tensors"] for n, (h, _) in d["per_shard"].items()},
            "dtypes": sorted({dt for h, _ in d["per_shard"].values() for dt in h["dtypes"]})}


# ---------------------------------------------------------------- modes
def cmd_hub_tree(_args):
    hub_tree()
    return 0


def cmd_build(_args):
    if os.path.exists(MANIFEST):
        raise SystemExit(f"REFUSED tag=write-once manifest exists: {MANIFEST}")
    refuse_write_into_models(MANIFEST)
    entries = hub_tree()
    hubs = hub_index(entries)
    d = derive(SHARD_DIR)
    rows = []
    for name in d["index_shards"]:
        path = os.path.join(SHARD_DIR, name)
        h = hubs.get(name)
        if h is None:
            raise SystemExit(f"REFUSED tag=hub-missing-shard {name} is in the index but not in "
                             "the Hub tree")
        digest = sha_file(path)
        rows.append({
            "file": name, "path": path, "bytes": os.path.getsize(path), "sha256": digest,
            "hub_etag": h["hub_etag"], "hub_etag_kind": h["hub_etag_kind"],
            "hub_size": h["hub_size"], "git_blob_sha1": h["git_blob_sha1"], "xet_hash": h["xet_hash"],
            "etag_equals_sha256": h["hub_etag_kind"] == "sha256-lfs" and h["hub_etag"] == digest,
            "bytes_equal_hub": os.path.getsize(path) == h["hub_size"],
            "local_mtime_epoch": int(os.path.getmtime(path)),
        })
    ids = count_identities(d, rows, SHARD_DIR)
    n_true = sum(1 for r in rows if r["etag_equals_sha256"])
    n_len_ok = sum(1 for r in rows if r["bytes_equal_hub"])
    check("every_shard_etag_equals_sha256", n_true == EXPECTED_SHARDS,
          f"etag_equal = {n_true} of {len(rows)} listed shards (expected {EXPECTED_SHARDS}); "
          f"length_equal_to_hub = {n_len_ok} of {len(rows)}")
    doc = {
        "kind": "full-checkpoint-integrity", "decision": "DECISIONS.md D-29",
        "date": "2026-10-27", "hub_repo": HUB_REPO, "hub_tree_url": HUB_TREE_URL,
        "hub_tree_cache": os.path.relpath(HUB_CACHE, REPO),
        "directory_verified": os.path.relpath(SHARD_DIR, REPO),
        "index_local": os.path.relpath(INDEX_LOCAL, REPO),
        "index_pinned_copy": os.path.relpath(INDEX_PIN, REPO),
        "index_sha256_local": sha_file(INDEX_LOCAL),
        "index_sha256_pinned": sha_file(INDEX_PIN) if os.path.exists(INDEX_PIN) else None,
        "pre_existing_shards_rehashed": "all 26 rows above were hashed in this run; the 4 that "
                                       "were already on disk (shards 1, 2, 25, 26 — older "
                                       "local_mtime_epoch) are re-hashed, not grandfathered",
        "etag_rule": "LFS etag == sha256 of the file bytes; small git-tracked files carry a sha1 "
                     "git-blob etag and are checked as sha1, never called a sha256 match",
        "shards": rows, "derived": ids,
        "supersedes": {
            "chain_manifest": "checkpoints/dump-manifest-chain-2026-10-26.json",
            "note": "the chain manifest's block_set.shard_census recorded 4 of 26 shards and 36 "
                    "missing layers; D-29 fetched the rest, so that census is stale by design and "
                    "is recorded here rather than edited (pinned fixtures are append-only, D-25)",
        },
        "provenance": "fetched unauthenticated (no HF_TOKEN) — D-29 Decision 1; this manifest is "
                      "what makes the bytes attributable to the published checkpoint",
    }
    with open(MANIFEST, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"\nMANIFEST wrote={os.path.relpath(MANIFEST, REPO)} shards={len(rows)} "
          f"findings={len(findings)}")
    for r in rows:
        print(f"  {r['file']} bytes={r['bytes']} sha256={r['sha256'][:16]}… "
              f"etag={r['hub_etag'][:16]}… etag_equals_sha256={r['etag_equals_sha256']}")
    print(("\nVERDICT integrity=findings=%d etag_equal=%d/%d\n" % (len(findings), n_true, len(rows)))
          + ("CHECKPOINT TRUSTWORTHY: every shard's sha256 equals the Hub's LFS etag"
             if n_true == EXPECTED_SHARDS and not findings else
             "\nCHECKPOINT NOT VERIFIED — do not use it as a reference"))
    return 1 if findings else 0


def cmd_verify(args):
    manifest = json.load(open(args.manifest))
    rows = manifest["shards"]
    listed = len(rows)
    print(f"verify  manifest={os.path.relpath(args.manifest, REPO)} dir={args.dir} "
          f"shards_listed={listed}")
    d = derive(args.dir)
    red = []
    for r in rows:
        path = os.path.join(args.dir, r["file"])
        if not os.path.exists(path):
            red.append(f"{r['file']}:missing-file")
            print(f"FAIL  shard {r['file']} :: file absent from {args.dir}")
            continue
        hdr_err = (d["per_shard"].get(r["file"]) or ({"header_error": "not on disk"}, {}))[0]["header_error"]
        digest = sha_file(path)
        size = os.path.getsize(path)
        ok_sha = digest == r["sha256"]
        ok_len = size == r["bytes"]
        ok_etag = r["hub_etag"] == digest if r["hub_etag_kind"] == "sha256-lfs" else None
        line = (f"{'OK  ' if ok_sha and ok_len else 'FAIL'} shard {r['file']} :: bytes={size} "
                f"(manifest {r['bytes']}, equal={ok_len}) sha256={digest[:16]}… "
                f"(manifest {r['sha256'][:16]}…, equal={ok_sha}) etag={r['hub_etag'][:16]}… "
                f"etag_equals_sha256={ok_etag}")
        print(line)
        if hdr_err:
            red.append(f"{r['file']}:header")
            print(f"FAIL shard {r['file']} :: safetensors header unreadable — {hdr_err}")
        if not (ok_sha and ok_len):
            red.append(f"{r['file']}:sha_or_len")
        if ok_etag is False:
            red.append(f"{r['file']}:etag")
    ids = count_identities(d, rows, args.dir)
    n_red = len(red)
    check("verify_clean", n_red == 0, f"red_shards={n_red} of listed={listed} {red[:4]}")
    print(f"\nVERDICT verify findings={len(findings)} red_shards={n_red} shards_listed={listed}")
    return 1 if findings else 0


# ---------------------------------------------------------------- non-vacuity
def run_verify(manifest_path, dirpath, expect, needle, label):
    """expect is 'red' or 'green'. A witness that cannot fail is not a witness, so the expectation
    is asserted, not narrated."""
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "verify",
                           "--manifest", manifest_path, "--dir", dirpath],
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    reddened = proc.returncode != 0
    want_red = expect == "red"
    check(f"witness[{label}].verifier_verdict_is_{expect}", reddened == want_red,
          f"returncode={proc.returncode} expected_{'nonzero' if want_red else 'zero'} "
          f"output_lines={len(out.splitlines())}")
    named = needle in out
    check(f"witness[{label}].names_the_case", named,
          f"needle={needle!r} present_in_output={named}")
    if not named:
        print(out[-1500:])
    return out


def cmd_witness(_args):
    if not os.path.exists(MANIFEST):
        raise SystemExit("REFUSED tag=build-first run `build` before `witness`")
    manifest = json.load(open(MANIFEST))
    rows = manifest["shards"]
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)
    pristine = os.path.join(SCRATCH, "pristine")
    perturbed = os.path.join(SCRATCH, "perturbed")
    os.makedirs(pristine)
    os.makedirs(perturbed)
    victim = sorted(rows, key=lambda r: r["bytes"])[0]["file"]
    print(f"witness scratch={SCRATCH} victim_shard={victim} "
          f"bytes={next(r['bytes'] for r in rows if r['file'] == victim)}")
    for r in rows:
        src = os.path.join(SHARD_DIR, r["file"])
        os.symlink(src, os.path.join(pristine, r["file"]))
        # Only the victim is copied; the rest are symlinks, so the scratch tree costs ~1 GiB and
        # models/ is still never written (a symlink read is a read of the original bytes).
        if r["file"] == victim:
            shutil.copyfile(src, os.path.join(perturbed, r["file"]))
        else:
            os.symlink(src, os.path.join(perturbed, r["file"]))
    for extra in ("model.safetensors.index.json",):
        os.symlink(os.path.join(SHARD_DIR, extra), os.path.join(pristine, extra))
        os.symlink(os.path.join(SHARD_DIR, extra), os.path.join(perturbed, extra))
    thin_manifest = json.loads(json.dumps(manifest))
    thin_manifest["shards"] = [r for r in thin_manifest["shards"] if r["file"] != victim]
    thin_path = os.path.join(SCRATCH, "manifest-missing-one-shard.json")
    with open(thin_path, "w") as fh:
        json.dump(thin_manifest, fh, indent=1, sort_keys=True)
    perturbed_path = os.path.join(perturbed, victim)
    victim_header = safetensors_header(os.path.join(SHARD_DIR, victim))[0]["header_bytes"]
    with open(perturbed_path, "r+b") as fh:
        fh.seek(victim_header + 4096)
        b = fh.read(1)
        fh.seek(victim_header + 4096)
        fh.write(bytes([b[0] ^ 0xFF]))
    print(f"witness perturbation: one byte flipped at offset header_bytes+4096={victim_header + 4096} "
          f"of {victim} (inside the tensor data, so the sha256 must move and the length must not)")
    full = os.path.join(SCRATCH, "manifest-full.json")
    shutil.copyfile(MANIFEST, full)

    print("--- witness 0: the pristine scratch tree (symlinks) must stay green, or the two "
          "witnesses below prove nothing")
    out0 = run_verify(full, pristine, expect="green", needle=victim,
                      label="pristine-symlink-tree")
    check("witness[pristine].reports_all_shards_green",
          proc_green(out0) and f"shard {victim}" in out0,
          "the symlinked tree hashes identically to the real one, so the perturbation below is "
          "the only difference; the victim shard is named here as a green row, not as a failure")
    print("--- witness 1: one byte of one shard perturbed (models/ untouched) must redden THAT "
          "shard by name")
    run_verify(full, perturbed, expect="red", needle=victim, label="perturbed-shard")
    print("--- witness 2: one shard deleted from the listing must fail the COUNT identity, "
          "not pass quietly")
    run_verify(thin_path, pristine, expect="red", needle=victim, label="listing-deleted")
    print("--- witness 3: header corruption must redden that shard by name, not crash the "
          "verifier")
    broken = os.path.join(SCRATCH, "header-broken")
    os.makedirs(broken, exist_ok=True)
    for r in rows:
        os.symlink(os.path.join(SHARD_DIR, r["file"]), os.path.join(broken, r["file"]))
    os.symlink(os.path.join(SHARD_DIR, "model.safetensors.index.json"),
               os.path.join(broken, "model.safetensors.index.json"))
    broken_copy = os.path.join(broken, victim)
    os.unlink(broken_copy)
    shutil.copyfile(os.path.join(SHARD_DIR, victim), broken_copy)
    with open(broken_copy, "r+b") as fh:
        fh.seek(64)
        fh.write(b"\xbd\xbd\xbd\xbd")
    run_verify(full, broken, expect="red", needle=victim, label="header-corrupted")
    n_rows = len(thin_manifest["shards"])
    print(f"witness listing had {EXPECTED_SHARDS} rows, now has {n_rows}; "
          f"identity printed above must read {EXPECTED_SHARDS} - {n_rows} = 1")
    print(f"\nVERDICT witness findings={len(findings)} scratch_kept={SCRATCH}")
    return 1 if findings else 0


def proc_green(out):
    return "VERDICT verify findings=0" in out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["hub-tree", "build", "verify", "witness"])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--dir", default=SHARD_DIR)
    args = ap.parse_args()
    return {"hub-tree": cmd_hub_tree, "build": cmd_build, "verify": cmd_verify,
            "witness": cmd_witness}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
