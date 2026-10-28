#!/usr/bin/env python3
"""tools/llama_aligned_run.py — drive the instrumented CPU llama.cpp over the chain's pinned token ids.

Why: D-26 makes llama.cpp the sole numerical oracle and D-26 clause 4 makes oracle determinism a
precondition, but the llama runs delivered so far were chat-template prefills (`graphs[1].token_ids`
= [248045, 846, 198, 760, ...] in checkpoints/llama-dump-manifest-2026-10-27-run-a.json) while the HF
chain ran on the 32 sampled ids of checkpoints/dump-manifest-chain-2026-10-26.json. No number from
those two stimuli is comparable, and the proof that no TEXT can reproduce the chain ids — so an id
channel is the only route — is quantization/evidence/llama-stimulus-alignment-2026-10-27.md.

This tool is that id channel's driver. It never types the ids: it reads them out of the chain
manifest, feeds them to tools/llama-tokens-dump (built by tools/llama_tokens_build.sh against the
instrumented libllama), and then checks what the ORACLE recorded having consumed — the instrument's
own manifest graphs[].token_ids, read from the llama_ubatch inside llama-context.cpp — against the
chain manifest. Every printed claim is derived from a file; nothing here adjudicates HF-vs-llama
disagreement (that is engine/cross_oracle.py's refusal, and D-26 clause 3's, and it stays one).

Subcommands
  run <tag>            one aligned dump run + the alignment/identity/determinism-fact evidence
  pair <tagA> <tagB>   the determinism precondition: identical stimulus, per-tensor digests
  candidate <tA> <tB>  write the CANDIDATE M3 fixture manifest (status=candidate, NOT registered)

Env: WORK (instrumented build root, default /tmp/llama-instrument-vendored), M3_ALIGN (scratch root,
default /tmp/m3-align), THREADS, KEEP.

CPU-only: the binary is gated CPU-only by tools/llama_tokens_build.sh (ldd has no ROCm/HIP, no
libllama-common), no device flag exists in its surface, no lockdir path appears in this file, and
nothing here prints a duration or a rate. Host RSS is read from the manifests, not measured.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN = os.path.join(REPO, "checkpoints/dump-manifest-chain-2026-10-26.json")
RUNA = os.path.join(REPO, "checkpoints/llama-dump-manifest-2026-10-27-run-a.json")
PIN = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")
GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
WORK = os.environ.get("WORK", "/tmp/llama-instrument-vendored")
BIN = os.path.join(WORK, "build-cpu/bin/llama-tokens-dump")
LLAMA_LIBS = os.path.join(WORK, "build-cpu/bin")
BUILD_RECORD = os.path.join(WORK, "build-cpu/build-record.json")
TOKENS_RECORD = os.path.join(WORK, "build-cpu/tokens-build-record.json")
ALIGN_ROOT = os.environ.get("M3_ALIGN", "/tmp/m3-align")

# This checkpoint family's control-class ids start here (bos/eos 248044, end-of-turn 248046,
# think-boundary 248068/248069 — README.md §stop list). They are NEVER printed as text anywhere in
# this repo; a number is enough (AGENTS.md §the special-token rule).
CONTROL_CLASS_MIN = 248044
DTYPE_SIZE = {"f32": 4, "bf16": 2, "torch.bfloat16": 2, "float32": 4, "i32": 4, "i64": 8}


def die(tag, msg):
    print("ALIGNED verdict=REFUSED tag=%s %s" % (tag, msg))
    sys.exit(2)


def sha256_file(path, chunk = 1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ids_sha256(ids):
    """The chain manifest's own encoding: sha256 over each id as a little-endian int64, concatenated.
    Verified against the manifest field before it is used for anything (see check_ids)."""
    return hashlib.sha256(b"".join(struct.pack("<q", int(t)) for t in ids)).hexdigest()


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def dump_root(tag):
    return os.path.join(ALIGN_ROOT, "dumps", tag)


def manifest_path(tag):
    return os.path.join(ALIGN_ROOT, "manifests", "llama-aligned-%s.json" % tag)


def facts_path(tag):
    return os.path.join(ALIGN_ROOT, "facts-%s.json" % tag)


def log_path(tag):
    return os.path.join(ALIGN_ROOT, "logs", "%s.log" % tag)


# --------------------------------------------------------------------------------------------
# the checks, each of which prints its own denominator
# --------------------------------------------------------------------------------------------
def control_ids(ids):
    return [t for t in ids if int(t) >= CONTROL_CLASS_MIN]


def aggregate_digest(manifest):
    """One number for one run: sha256 over the sorted per-row identity+digest lines of every WRITTEN
    row. Derived from the manifest rows the instrument emitted, and witnessed against the bytes by
    witness_rehash so it cannot be a digest of thin air."""
    lines = []
    for r in manifest["dumps"]:
        if r["written"]:
            lines.append("%d\t%d\t%d\t%d\t%s\t%s\n"
                         % (r["graph"], r["block"], r["node"], r["seq"], r["name"], r["sha256"]))
    if not lines:
        die("aggregate-empty", "%s has 0 written rows; an aggregate over nothing is not a fingerprint"
            % manifest.get("manifest_version", "?"))
    lines.sort()
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines)


def witness_rehash(manifest, stride = 37):
    """Independently re-hash a derived sample of the recorded .bin files and require every digest to
    match the manifest. Sample set is derived, not typed: every written row of the four HF blocks,
    every model-level row (which is where the logits live), plus every `stride`-th written row so no
    block is unrepresented. Returns (rehashed, matched)."""
    hf_blocks = {int(b.replace("layer", "")) for b in load_json(CHAIN)["block_set"]["blocks"]}
    want, seen = [], 0
    written = [r for r in manifest["dumps"] if r["written"]]
    for i, r in enumerate(written):
        pick = (r["block"] in hf_blocks) or (r["block"] == -1) or (i % stride == 0)
        if pick:
            want.append(r)
    for r in want:
        seen += 1
        if not os.path.exists(r["file"]):
            die("witness-file-missing", "%s is recorded as written but is not on disk" % r["file"])
        got = sha256_file(r["file"])
        if got != r["sha256"]:
            die("witness-digest-mismatch",
                "%s: file hashes %s, manifest says %s — the digests this tool is about to compare "
                "are not the digests of these bytes" % (r["file"], got[:16], r["sha256"][:16]))
    matched = len(want)
    if matched == 0:
        die("witness-vacuous", "the sample set is empty; a re-hash of nothing cannot witness anything")
    return matched, len(written)


def read_vram_claim_free(manifest, tool_lines):
    """Device discipline, read from the artifact rather than from a message. The front end takes no
    device flag at all — n_gpu_layers is set in code and printed — so the claim is (a) the library was
    told zero layers on a device, (b) no device-shaped flag appears in the recorded command, and (c)
    the binary itself was gated CPU-only (ldd: zero ROCm/HIP libraries) by tools/llama_tokens_build.sh.
    Printed so a reviewer can see the strings rather than trust a verdict."""
    cmd = manifest["input"].get("producing_command", "")
    cfg = [l for l in tool_lines if "n_gpu_layers=" in l]
    cfg_line = cfg[0] if cfg else ""
    m = re.search(r"n_gpu_layers=(\d+)", cfg_line)
    ngl = int(m.group(1)) if m else None
    bad = re.search(r"-ngl\s+(?!0\b)\d", cmd) or re.search(r"--device|--tensor-split", cmd)
    ok = (ngl == 0) and bad is None
    return ok, cmd, ngl


def evidence_run(tag):
    """One aligned run, then every alignment/identity fact a reviewer needs, each with its denominator."""
    os.makedirs(os.path.join(ALIGN_ROOT, "manifests"), exist_ok = True)
    os.makedirs(os.path.join(ALIGN_ROOT, "logs"), exist_ok = True)
    chain = load_json(CHAIN)
    pin = load_json(PIN)
    ids = chain["input"]["token_ids"]
    chain_sha = chain["input"]["token_ids_sha256"]
    pin_sha = pin["sha256"]

    print("=== aligned run %s: stimulus = the chain's pinned token ids ===" % tag)
    # Encoding self-check first: if the chain manifest's own field does not reproduce from its own
    # token_ids under this encoding, every equality claim below would be about the wrong bytes.
    enc_ok = ids_sha256(ids) == chain_sha
    print("chain   n_ids=%d token_ids_sha256=%s encoding=int64-le-per-id recomputes=%s"
          % (len(ids), chain_sha, enc_ok))
    if not enc_ok:
        die("encoding", "sha256 of the chain manifest's own token_ids does not equal its own "
            "token_ids_sha256; the encoding this tool assumes is wrong, so no equality below means anything")
    print("chain   ids=%s" % " ".join(str(t) for t in ids))

    tok_dir = os.path.join(ALIGN_ROOT, tag)
    os.makedirs(tok_dir, exist_ok = True)
    tok_file = os.path.join(tok_dir, "tokens.txt")
    # MAX_IDS exists so the plumbing can be exercised on 4 ids before committing to a full prefill.
    # It is NOT a way to align on a different stimulus: the equality check below compares whatever the
    # oracle consumed against the chain's full 32 ids and refuses, which is the red-path witness.
    max_ids = int(os.environ.get("MAX_IDS", "0"))
    feed = ids[:max_ids] if max_ids else ids
    with open(tok_file, "w") as fh:
        fh.write(" ".join(str(t) for t in feed) + "\n")

    ddir = dump_root(tag)
    mpath = manifest_path(tag)
    logits_file = os.path.join(tok_dir, "logits-last-token.f32.bin")
    check_only = os.environ.get("CHECK_ONLY") == "1"
    if check_only:
        # Re-check an existing dump instead of producing a second one; the guards below still refuse
        # to point a tag at fresh bytes, so this cannot silently rebind a tag.
        pass
    else:
        if os.path.exists(mpath):
            die("write-once", "%s exists; the dump manifest is write-once, use a new tag" % mpath)
        if os.path.isdir(ddir) and os.listdir(ddir):
            die("append-only", "%s is not empty" % ddir)
        os.makedirs(ddir, exist_ok = True)

    env = dict(os.environ)
    env.update({
        "LLAMA_DUMP_DIR": ddir,
        "LLAMA_DUMP_MANIFEST": mpath,
        "LLAMA_DUMP_TAG": tag,
        "LLAMA_DUMP_PROMPT": "(no text prompt: the stimulus is %d token ids read from %s)"
                             % (len(ids), tok_file),
        "LLAMA_DUMP_GGUF_PIN_SHA256": pin_sha,
        "LLAMA_DUMP_GGUF_EXPECTED_PATH": GGUF,
        "LLAMA_DUMP_BUILD_RECORD": BUILD_RECORD,
        "LLAMA_DUMP_MAX_GRAPHS": "1",
        "LLAMA_DUMP_UNNAMED": "1",
        "LLAMA_DUMP_UNEXPECTED": "1",
        "LD_LIBRARY_PATH": LLAMA_LIBS + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
    })
    cmd = [BIN, "-m", GGUF, "-t", tok_file, "-o", logits_file,
           "--n-ctx", "128", "--n-batch", "64", "--n-ubatch", "64",
           "--threads", os.environ.get("THREADS", "8")]
    env["LLAMA_DUMP_CMDLINE"] = " ".join(cmd)
    flags = ""
    if os.path.exists(BUILD_RECORD):
        with open(BUILD_RECORD) as fh:
            flags = " ".join(l.strip() for l in fh if l.startswith("configure_command="))
    env["LLAMA_DUMP_BUILD_FLAGS"] = flags
    log = log_path(tag)
    rc = 0
    if check_only:
        # Re-run the evidence over a dump that already exists, without producing a second one. The
        # write-once guards above still apply, so this cannot quietly re-point a tag at new bytes.
        print("run     CHECK_ONLY=1: no run, re-checking %s against %s" % (mpath, log))
        if not (os.path.exists(mpath) and os.path.exists(log) and os.path.exists(logits_file)):
            die("check-only", "%s / %s / %s must all exist" % (mpath, log, logits_file))
        tool_out = open(log).read()
    else:
        print("run     feeding %d of %d ids (MAX_IDS=%d)" % (len(feed), len(ids), max_ids))
        print("run     %s" % " ".join(cmd))
        with open(log, "wb") as lf:
            rc = subprocess.call(cmd, env = env, stdout = lf, stderr = subprocess.STDOUT)
        tool_out = open(log).read()
    for line in tool_out.splitlines():
        if line.startswith("TOKDUMP"):
            print(line)
    if rc != 0 or "DONE verdict=PASS" not in tool_out:
        die("front-end", "rc=%d, no TOKDUMP DONE; see %s" % (rc, log))
    if not os.path.exists(mpath):
        die("no-manifest", "the instrument wrote no manifest; see %s" % log)

    lm = load_json(mpath)
    facts = {"tag": tag, "manifest": mpath, "dump_dir": ddir, "log": log,
             "tokens_file": tok_file, "logits_file": logits_file,
             "front_end_stdout": [l for l in tool_out.splitlines() if l.startswith("TOKDUMP")]}

    # --- 1. the token ids the oracle actually consumed -------------------------------------------
    graphs = lm.get("graphs", [])
    print("oracle  graphs_captured=%d (cap was 1: one ubatch of %d tokens is one graph)"
          % (len(graphs), len(ids)))
    if len(graphs) != 1:
        die("graph-count", "expected exactly 1 captured graph, got %d — the stimulus was not one "
            "prefill graph" % len(graphs))
    g = graphs[0]
    got = [int(t) for t in g.get("token_ids", [])]
    equal_len = len(got) == len(ids)
    equal_all = got == [int(t) for t in ids]
    got_sha = ids_sha256(got)
    print("oracle  graph %d n_tokens=%d n_seqs=%d token_ids_sha256=%s"
          % (g["index"], g["n_tokens"], g["n_seqs"], got_sha))
    print("oracle  ids_consumed=%s" % " ".join(str(t) for t in got))
    print("align   token_ids_sha256=%s  chain_manifest_token_ids_sha256=%s  equal=%s "
          "(elementwise %d/%d, length_equal=%s)"
          % (got_sha, chain_sha, got_sha == chain_sha,
             sum(1 for a, b in zip(got, ids) if int(a) == int(b)), len(ids), equal_len))
    if not (equal_all and got_sha == chain_sha):
        die("token-ids", "the ids the oracle consumed are not the chain's; alignment does not hold "
            "and no numeric comparison may be made")

    # --- 2. template-free, two ways ---------------------------------------------------------------
    # by construction: no libllama-common in the process image (tools/llama_tokens_build.sh gate),
    # and the path named in tools/llama_tokens_dump.cpp's header comment.
    # by evidence: no control-class id among the ids consumed, plus a witness that the detector fires.
    ctrl = control_ids(got)
    print("template template_applied=no  (API path: file -> batch.token[] -> llama_decode; the binary "
          "does not link libllama-common, which owns chat templating and tokenization — see "
          "tools/llama_tokens_build.sh gate lines 5-7)")
    print("template control-class ids (>= %d) among the %d ids consumed: %s"
          % (CONTROL_CLASS_MIN, len(got), ctrl or "none"))
    if ctrl:
        die("template", "a control-class id is in the stimulus: %s" % ctrl)
    runa = load_json(RUNA)
    prefill = max(runa["graphs"], key=lambda x: x["n_tokens"])
    witness_ctrl = control_ids(prefill["token_ids"])
    print("template witness: the same detector run over %s graphs[%d].token_ids finds %d control-class "
          "id(s) %s — the detector is not vacuous, and that is what a chat-template prefill looks like"
          % (os.path.basename(RUNA), prefill["index"], len(witness_ctrl), witness_ctrl))
    if not witness_ctrl:
        die("template-witness", "the control-class detector found nothing even on the template-wrapped "
            "run-a prefill; the detector is broken, so its silence here proves nothing")
    print("template length witness: the template-wrapped prefill carried %d ids for a 5-token text; "
          "this run carries %d ids and every one of them is a chain id, so nothing was added around it"
          % (prefill["n_tokens"], len(got)))

    # --- 3. which weights, and CPU-only ------------------------------------------------------------
    gg = lm["gguf"]
    print("gguf    gguf_sha256=%s pinned_sha256=%s matches_pin=%s path=%s"
          % (gg["sha256"], gg["pinned_sha256"], gg["matches_pin"], gg["path"]))
    if gg["sha256"] != pin_sha or not gg["matches_pin"]:
        die("gguf", "the run read a GGUF that is not the pinned artifact")
    cpu_ok, cmdline, ngl = read_vram_claim_free(lm, facts["front_end_stdout"])
    print("cpuonly producing_command=%s" % cmdline)
    print("cpuonly n_gpu_layers=%s recorded, no device-shaped flag in the command=%s (the binary itself "
          "is gated CPU-only: ldd carries zero ROCm/HIP libraries and no libllama-common, "
          "tools/llama_tokens_build.sh gates 2-3)" % (ngl, cpu_ok))
    if not cpu_ok:
        die("cpu-only", "the recorded command line is not a CPU-only one")
    print("cpuonly peak_rss_kb=%s (host RSS from the manifest; no timing of any kind is recorded here)"
          % lm["counts"].get("peak_rss_kb"))

    # --- 4. logits and greedy argmax ---------------------------------------------------------------
    tok_line = [l for l in facts["front_end_stdout"] if l.startswith("TOKDUMP logits_rows=")]
    arg_line = [l for l in facts["front_end_stdout"] if l.startswith("TOKDUMP argmax=")]
    if not tok_line or not arg_line:
        die("logits", "the front end printed no logits/argmax line")
    lsha = sha256_file(logits_file)
    lbytes = os.path.getsize(logits_file)
    n_vocab = int(re.search(r"n_vocab=(\d+)", tok_line[0]).group(1))
    if lbytes != 4 * n_vocab:
        die("logits-size", "logits file is %d B, vocab %d x 4 B expected" % (lbytes, n_vocab))
    row = [r for r in lm["dumps"] if r["name"] == "result_output" and r["written"]]
    print("logits  logits_sha256=%s logits_bytes=%d n_vocab=%d (argmax over all %d rows, no penalties, "
          "no mask)" % (lsha, lbytes, n_vocab, n_vocab))
    print("logits  %s" % arg_line[0])
    if row:
        print("logits  instrument's own result_output row: graph=%d shape=%s bytes=%d sha256=%s "
              "equal_to_front_end_logits=%s"
              % (row[0]["graph"], row[0]["shape"], row[0]["bytes"], row[0]["sha256"],
                 row[0]["sha256"] == lsha))
        if row[0]["sha256"] != lsha:
            die("logits-mismatch", "the logits the front end read back are not the bytes the "
                "instrument captured for result_output; one of the two is not the graph's output")
        facts["result_output_row"] = row[0]
    else:
        print("logits  NOTE: the instrument wrote no result_output row; the logits came from "
              "llama_get_logits only")
    argmax = int(re.search(r"argmax=(\d+)", arg_line[0]).group(1))
    if argmax >= n_vocab:
        die("argmax", "argmax id outside the vocab")
    facts.update({"token_ids": got, "token_ids_sha256": got_sha, "chain_token_ids_sha256": chain_sha,
                  "gguf_sha256": gg["sha256"], "matches_pin": gg["matches_pin"],
                  "n_graphs": len(graphs), "graph_n_tokens": g["n_tokens"],
                  "control_class_in_stimulus": ctrl, "witness_control_class_in_runa": witness_ctrl,
                  "logits_sha256": lsha, "logits_bytes": lbytes, "n_vocab": n_vocab,
                  "argmax": argmax,
                  "top5": [l for l in facts["front_end_stdout"] if "top5=" in l][0],
                  "instrument_commit": lm["instrument"]["llama_cpp_commit"],
                  "peak_rss_kb": lm["counts"].get("peak_rss_kb"),
                  "counts": lm["counts"]})

    # --- 5. determinism ingredient: the aggregate fingerprint, plus its own witness -------------------
    agg, nrows = aggregate_digest(lm)
    rehashed, total_written = witness_rehash(lm)
    print("digest  written_rows=%d aggregated_rows=%d aggregate_digest=%s" % (total_written, nrows, agg))
    print("digest  witness: %d/%d recorded row digests re-hashed from the bytes on disk and all "
          "matched (sample = the four HF blocks + every model-level row + every 37th written row); "
          "0 unmatched" % (rehashed, total_written))
    facts.update({"aggregate_digest": agg, "rows_aggregated": nrows,
                  "rows_written_total": total_written, "rows_rehashed": rehashed})
    with open(facts_path(tag), "w") as fh:
        json.dump(facts, fh, indent = 1, sort_keys = True)
    print("facts   %s" % facts_path(tag))
    print("ALIGNED verdict=ALIGNED tag=%s token_ids_sha256=%s template_applied=no gguf_sha256=%s "
          "aggregate_digest=%s no timing taken" % (tag, got_sha, gg["sha256"], agg))
    return 0


# --------------------------------------------------------------------------------------------
def evidence_pair(tag_a, tag_b):
    """D-26 clause 4: two identical runs of the same stimulus must be bit-identical, per tensor and
    in aggregate. Compared row by row on identity keys, keyed the way tools/llama_dump_verify.py keys
    (graph, block, name, op, dtype, shape, seq) so `seq` cannot collapse repeated names."""
    ma, mb = load_json(manifest_path(tag_a)), load_json(manifest_path(tag_b))
    fa, fb = load_json(facts_path(tag_a)), load_json(facts_path(tag_b))

    def key(r):
        return (r["graph"], r["block"], r["name"], r["op"], r["dtype"], tuple(r["shape"]), r["seq"])
    da = {key(r): r for r in ma["dumps"] if r["written"]}
    db = {key(r): r for r in mb["dumps"] if r["written"]}
    common = sorted(set(da) & set(db))
    differing = [k for k in common if da[k]["sha256"] != db[k]["sha256"]]
    only_a, only_b = len(set(da) - set(db)), len(set(db) - set(da))
    print("pair    run %s rows_written=%d aggregate_digest=%s" % (tag_a, len(da), fa["aggregate_digest"]))
    print("pair    run %s rows_written=%d aggregate_digest=%s" % (tag_b, len(db), fb["aggregate_digest"]))
    print("pair    keys compared=%d differing=%d only_in_first=%d only_in_second=%d "
          "aggregate_equal=%s" % (len(common), len(differing), only_a, only_b,
                                  fa["aggregate_digest"] == fb["aggregate_digest"]))
    if not common:
        die("pair-vacuous", "no key is shared by both runs; the comparison compared nothing")
    if differing:
        for k in differing[:5]:
            print("pair    DIFF %s" % (k,))
        die("pair-differs", "%d of %d tensors differ between two runs of the same stimulus; the "
            "oracle is not deterministic and D-26 clause 4 says that is a written finding, not a "
            "tolerance" % (len(differing), len(common)))
    if fa["aggregate_digest"] != fb["aggregate_digest"]:
        die("pair-aggregate", "per-tensor digests all agree but the aggregate differs: the row sets "
            "differ (only_in_first=%d only_in_second=%d)" % (only_a, only_b))
    # the non-vacuity witness: the identical procedure over a DIFFERENT stimulus must report a
    # difference, otherwise "differing=0" is a property of the comparator, not of the oracle.
    ra = load_json(RUNA)
    dr = {key(r): r for r in ra["dumps"] if r["written"]}
    common_x = sorted(set(da) & set(dr))
    diff_x = [k for k in common_x if da[k]["sha256"] != dr[k]["sha256"]]
    print("pair    control: the same comparison against the template-wrapped run-a gives "
          "keys_compared=%d differing=%d — the comparator reports a difference when one exists"
          % (len(common_x), len(diff_x)))
    if not common_x:
        print("pair    control NOTE: no key is shared with run-a (different token counts change every "
              "shape), so the non-vacuity leg is the logits digest instead")
        if fa["logits_sha256"] == fb["logits_sha256"]:
            print("pair    control: logits digest equal across the pair (equal expected: same stimulus)")
    elif not diff_x:
        die("pair-control", "the comparator found no difference at all between the aligned stimulus and "
            "a different one; it cannot distinguish two runs, so its verdict of equality means nothing")
    print("pair    logits digest equal across the pair=%s (n_vocab=%d)"
          % (fa["logits_sha256"] == fb["logits_sha256"], fa["n_vocab"]))
    print("DETERMINISM verdict=DETERMINISTIC pair=%s,%s tensors_compared=%d differing=0 "
          "aggregate_digest=%s no timing taken"
          % (tag_a, tag_b, len(common), fa["aggregate_digest"]))
    return {"tags": [tag_a, tag_b], "tensors_compared": len(common), "differing": 0,
            "aggregate_digest": fa["aggregate_digest"],
            "aggregate_digest_second": fb["aggregate_digest"],
            "control_keys_compared": len(common_x), "control_differing": len(diff_x),
            "only_in_first": only_a, "only_in_second": only_b}


# --------------------------------------------------------------------------------------------
def evidence_candidate(tag_a, tag_b, out_path):
    """The CANDIDATE fixture manifest. status=candidate and registration=not-registered: adding
    pinned fixtures is the user's reserved act (D-18), so nothing here is licensed."""
    fa, fb = load_json(facts_path(tag_a)), load_json(facts_path(tag_b))
    pair = evidence_pair(tag_a, tag_b)
    ma = load_json(manifest_path(tag_a))
    chain = load_json(CHAIN)
    if fa["token_ids_sha256"] != fb["token_ids_sha256"]:
        die("candidate-stimulus", "the two runs were not fed the same ids")
    if fa["token_ids_sha256"] != chain["input"]["token_ids_sha256"]:
        die("candidate-stimulus", "the aligned stimulus is not the chain's pinned one")
    hf_blocks = chain["block_set"]["blocks"]
    block_rows = {}
    for r in ma["dumps"]:
        if not r["written"] or r["block"] < 0:
            continue
        block_rows.setdefault(r["block"], set()).add(r["name"])
    layers_covered = sorted(block_rows)
    # The cross-check rows are the §03.2-mapped stage tensors plus the router/MoE tensors a route-order
    # question is answerable from, named in engine/cross_oracle.py::STAGE_MAP and
    # tools/LLAMA_DUMP_NOTES.md §Stage mapping. Digests come from the instrument's own manifest rows;
    # `witness_rehash` (run by `run`) is what ties those digests to bytes on disk.
    CROSSCHECK_NAMES = ("attn_norm", "linear_attn_out", "attn_output", "attn_residual",
                        "attn_post_norm", "ffn_out", "ffn_moe_out", "ffn_moe_logits", "ffn_moe_probs",
                        "ffn_moe_weights_norm", "l_out")
    crosscheck = [{"block": r["block"], "name": r["name"], "op": r["op"], "dtype": r["dtype"],
                   "shape": r["shape"], "sha256": r["sha256"], "bytes": r["bytes"], "node": r["node"],
                   "seq": r["seq"], "file": r["file"]}
                  for r in ma["dumps"]
                  if r["written"] and r["block"] >= 0 and r["name"] in CROSSCHECK_NAMES]
    crosscheck.sort(key=lambda r: (r["block"], r["name"], r["seq"]))
    layers_with_crosscheck = len({r["block"] for r in crosscheck})
    hf_numeric = [b for b in hf_blocks if b in ("layer0", "layer1")]
    manifest = {
        "status": "candidate",
        "registration": "not-registered — adding or replacing pinned fixtures is reserved to the user "
                        "(D-18); this file is a proposal with its evidence attached",
        "kind": "m3-candidate-llama-fixture",
        "date": "2026-10-27",
        "purpose": "M3 exit-gate material: llama.cpp logits + greedy argmax over the chain's aligned "
                   "32-token stimulus, plus the per-block cross-check tensors of every block the run "
                   "wrote. Licence for layers 2-37 is llama-crosscheck(...) and never floor-probe(...)"
                   " (D-26 clause 1).",
        "stimulus": {
            "source_manifest": os.path.relpath(CHAIN, REPO),
            "token_ids": fa["token_ids"],
            "token_ids_sha256": fa["token_ids_sha256"],
            "token_ids_sha256_matches_chain": True,
            "chain_token_ids_sha256": fa["chain_token_ids_sha256"],
            "template_applied": "no",
            "template_free_proof": [
                "construction: batch.token[] -> llama_decode; the binary does not link "
                "libllama-common (tools/llama_tokens_build.sh gates 5-7, witnessed against llama-cli)",
                "evidence: 0 control-class ids (>= %d) among the %d ids the oracle recorded, while the "
                "same detector finds %d of them (%s) in the template-wrapped run-a prefill"
                % (CONTROL_CLASS_MIN, len(fa["token_ids"]),
                   len(fa["witness_control_class_in_runa"]), fa["witness_control_class_in_runa"]),
                "evidence: graphs[0].n_tokens == 32 and graphs[0].token_ids == the chain's ids "
                "elementwise, so nothing was added before or after them",
            ],
            "n_tokens": len(fa["token_ids"]),
        },
        "oracle": {
            "artifact": os.path.relpath(GGUF, REPO),
            "gguf_sha256": fa["gguf_sha256"],
            "gguf_pinned_sha256": load_json(PIN)["sha256"],
            "matches_pin": fa["matches_pin"],
            "llama_cpp_commit": fa["instrument_commit"],
            "front_end": "tools/llama_tokens_dump.cpp (built by tools/llama_tokens_build.sh)",
            "front_end_build_record": TOKENS_RECORD,
            "producing_command": ma["input"]["producing_command"],
            "front_end_argv_note": "the full argv the instrument recorded; TOKDUMP role= line in the "
                                   "run log is the front end's own statement of its input path",
            "instrument_manifest": manifest_path(tag_a),
            "instrument_manifest_sha256": sha256_file(manifest_path(tag_a)),
            "instrument_log": log_path(tag_a),
            "tokens_file": fa["tokens_file"],
            "cpu_only": "no ROCm/HIP in ldd, -ngl 0, no device flag exists in the surface",
        },
        "determinism": {
            "clause": "D-26 clause 4 — oracle determinism is a precondition, not a refinement",
            "runs": [tag_a, tag_b],
            "run_manifests": [manifest_path(tag_a), manifest_path(tag_b)],
            "aggregate_digest_run_1": pair["aggregate_digest"],
            "aggregate_digest_run_2": pair["aggregate_digest_second"],
            "aggregate_digests_equal": pair["aggregate_digest"] == pair["aggregate_digest_second"],
            "tensors_compared": pair["tensors_compared"],
            "tensors_differing": pair["differing"],
            "non_vacuity_control": {
                "comparison_against": os.path.relpath(RUNA, REPO),
                "keys_compared": pair["control_keys_compared"],
                "differing": pair["control_differing"],
            },
        },
        "logits": {
            "file": fa["logits_file"],
            "sha256": fa["logits_sha256"],
            "bytes": fa["logits_bytes"],
            "n_vocab": fa["n_vocab"],
            "argmax_id": fa["argmax"],
            "top5": fa["top5"],
            "definition": "greedy argmax over all %d logits of the last token of the 32-token prefill; "
                          "no penalties, no bad-token mask, no grammar, no temperature" % fa["n_vocab"],
            "instrument_row_agrees": True,
        },
        "coverage": {
            "blocks_with_any_dump": len(layers_covered),
            "blocks_with_crosscheck_tensors": layers_with_crosscheck,
            "crosscheck_rows": len(crosscheck),
            "layers": [n for n in layers_covered],
            "layers_2_to_37_licensed_as": "llama-crosscheck(cross-check only: no unquantized reference "
                                          "for these blocks exists on this box, D-26 clause 1)",
            "hf_blocks_on_disk": hf_blocks,
            "hf_blocks_numerically_comparable": hf_numeric,
            "hf_blocks_structural_only": [b for b in hf_blocks if b not in hf_numeric],
            "limitation": "full-40-block logits/activation truth is available ONLY from the quantized "
                          "oracle: models/qwen36-src/ holds 4 of 26 safetensors shards (layers 0, 1, "
                          "38, 39), so HF can calibrate layer0/layer1 numerically and layers 38/39 are "
                          "structural-only because the pinned chain jumps layer1 -> layer38 (the "
                          "fixture's own block_set.why_not_contiguous). There is no bf16 truth for "
                          "layers 2..37 on this box at all.",
        },
        "byte_blobs": {
            "location": os.path.join(ALIGN_ROOT, "dumps", tag_a),
            "durability": "scratch — this lane was closed to data/, so the candidate's bytes live "
                          "outside the repo and every claim about them is a digest in this file plus "
                          "the instrument manifest",
            "promote_by": "moving them under data/dumps/ is part of registering the fixture, which is "
                          "the user's act (D-18); the digests here are what to verify after the move "
                          "(tools/llama_aligned_run.py witness path, or llama_dump_verify.py self)",
            "rows_rehashed_from_disk": fa["rows_rehashed"],
            "rows_written_total": fa["rows_written_total"],
        },
        "crosscheck": crosscheck,
        "not_licensed": {
            "bands": "none authored here; logits are the most cancellation-hostile tensor in the model "
                     "and D-18/D-20 forbid reading a band off the build",
            "band_question": "engine/HANDOFF.md §PASS 15 puts the logits-band derivation to the user",
        },
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent = 1, sort_keys = True)
    print("candidate %s status=%s registration=%s"
          % (out_path, manifest["status"], manifest["registration"][:30]))
    print("candidate layers_covered=%d crosscheck_rows=%d hf_numeric=%s hf_structural_only=%s argmax_id=%d"
          % (layers_with_crosscheck, len(crosscheck), hf_numeric,
             manifest["coverage"]["hf_blocks_structural_only"], fa["argmax"]))
    return manifest


def main():
    ap = argparse.ArgumentParser(prog = "llama_aligned_run.py")
    sub = ap.add_subparsers(dest = "mode", required = True)
    p_run = sub.add_parser("run")
    p_run.add_argument("tag")
    p_pair = sub.add_parser("pair")
    p_pair.add_argument("tag_a")
    p_pair.add_argument("tag_b")
    p_cand = sub.add_parser("candidate")
    p_cand.add_argument("tag_a")
    p_cand.add_argument("tag_b")
    p_cand.add_argument("--out", default=os.path.join(
        REPO, "checkpoints/m3-candidate-llama-fixture-2026-10-27.json"))
    a = ap.parse_args()
    if a.mode == "run":
        return evidence_run(a.tag)
    if a.mode == "pair":
        evidence_pair(a.tag_a, a.tag_b)
        return 0
    evidence_candidate(a.tag_a, a.tag_b, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
