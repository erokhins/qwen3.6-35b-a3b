#!/usr/bin/env python3
"""llama_dump_verify.py — the two controls over an instrumented llama.cpp dump.

Reading guide: tools/LLAMA_DUMP_NOTES.md. Instrument: tools/llama-instrument.patch.

Three subcommands, each of which prints records and ends in a verdict line:

  self          <manifest> [<manifest> ...]
      every manifest row's file exists, has the recorded length, and hashes to the recorded digest
      (so the manifest is a claim about bytes on disk, not about a buffer we once had); the gguf
      digest equals the pin; the census arithmetic adds up in every block; and the number of parsed
      census rows equals the number the instrument printed in the run log.

  determinism   <manifestA> <manifestB>
      same prompt, two runs: every tensor present in both must hash the same. compared == 0 is a
      failure, not a pass (pitfalls/40: a check that can produce no output cannot fail).

  prompt-change <manifestBaseline> <manifestChanged>
      different prompt: at least one tensor must change, the logits must change, and the last block
      must change. compared == 0 or differing == 0 is a failure.

Nothing here measures how long anything took: every number printed is a count, a byte length or a
digest.
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENSUS_RE = re.compile(
    r"LLAMA-DUMP CENSUS graph=(?P<graph>-?\d+) block=(?P<block>-?\d+) kind=(?P<kind>\w+) "
    r"dumped=(?P<dumped>\d+) expected=(?P<expected>\d+) matched=(?P<matched>\d+) "
    r"unexpected=(?P<unexpected>\d+) anonymous=(?P<anonymous>\d+) state=(?P<state>\d+) "
    r"weights_excluded=(?P<weights_excluded>\d+) weights_unknown=(?P<weights_unknown>\d+) "
    r"duplicates=(?P<duplicates>\d+) not_written=(?P<not_written>\d+) "
    r"too_big=(?P<too_big>\d+) missing=(?P<missing>\d+)"
)
MISSING_NAMES_RE = re.compile(
    r"LLAMA-DUMP CENSUS-NAMES graph=(?P<graph>-?\d+) block=(?P<block>-?\d+) missing=(?P<missing>.*)$")
HEADER_ARRAY_RE = re.compile(
    r"^static const char \* const LLAMA_DUMP_EXPECTED_(\w+)\[\] = \{(.*?)^\};", re.M | re.S)
WEIGHT_ARRAY_RE = re.compile(
    r"^static const char \* const LLAMA_DUMP_WEIGHT_FAMILIES\[\] = \{(.*?)^\};", re.M | re.S)

EXPECTED_HEADER = os.path.join(ROOT, "tools", "llama-dump-expected.h")
CONFIG_PATH = os.path.join(ROOT, "checkpoints", "Qwen3.6-35B-A3B.config.json")


def expected_sets_from_header():
    """The denominator, read out of the generated header by this script alone.

    The point of re-deriving it here is that the instrument expands its own expected set at runtime
    from the same header; if the two derivations disagree, the census denominator was not what the
    census claims to be. A check that reuses the producer's own arithmetic proves nothing.
    """
    if not os.path.isfile(EXPECTED_HEADER):
        die(f"missing {EXPECTED_HEADER}: run tools/llama_dump_expected_gen.py --write")
    text = open(EXPECTED_HEADER).read()
    sets = {k.lower(): set(re.findall(r'"([^"]*)"', body)) for k, body in HEADER_ARRAY_RE.findall(text)}
    for needed in ("main", "attn_full", "attn_linear", "ffn", "model_level"):
        if needed not in sets:
            die(f"{EXPECTED_HEADER} has no LLAMA_DUMP_EXPECTED_{needed.upper()} array")
    wf = WEIGHT_ARRAY_RE.findall(text)
    return sets, set(re.findall(r'"([^"]*)"', wf[0])) if wf else set()


def param_family(nm):
    """Mirror of llama-dump.cpp::ld_param_family: "blk.17.ffn_gate_inp.weight" -> "ffn_gate_inp".

    Derived here independently on purpose: the instrument classifies parameters with that rule at run
    time, so if the two drift a parameter could be excluded as a weight without ever matching a family
    table, and weights_unknown == 0 would then be a lie.
    """
    if not nm.startswith("blk."):
        return None
    rest = nm[4:]
    i = 0
    while i < len(rest) and rest[i].isdigit():
        i += 1
    if i == 0 or i >= len(rest) or rest[i] != ".":
        return None
    tail = rest[i + 1:]
    return tail.rsplit(".", 1)[0] if "." in tail else tail


def expand(sets, kind):
    if kind == "model":
        return set(sets["model_level"])
    if kind == "linear":
        return sets["main"] | sets["attn_linear"] | sets["ffn"]
    if kind == "full":
        return sets["main"] | sets["attn_full"] | sets["ffn"]
    die(f"unknown block kind {kind!r}")


def die(msg):
    print(f"llama_dump_verify: {msg}", file=sys.stderr)
    sys.exit(1)


def load(manifest_path):
    if not os.path.isfile(manifest_path):
        die(f"no manifest at {manifest_path}")
    with open(manifest_path) as f:
        m = json.load(f)
    # the dumps live in data/dumps/<date>-llama/<tag>/, the manifest in checkpoints/
    tag = m["input"]["tag"]
    date = m["date"]
    dump_dir = os.path.join(ROOT, "data", "dumps", f"{date}-llama", tag)
    if not os.path.isdir(dump_dir):
        die(f"dump directory for {manifest_path} is missing: {dump_dir}")
    return m, dump_dir


_SHA_CACHE = {}


def sha256_file(path):
    # memoised: the 22 GB gguf is read once per verifier invocation, not once per manifest
    if path in _SHA_CACHE:
        return _SHA_CACHE[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def key_of(row):
    # seq is the instrument's per-(graph, block, name) occurrence counter, which makes a key unique
    # per written file: without it the ~1270 tensors that share a name inside a block (the two
    # state-view copies per block, the residual adds) would collapse to one key and never be compared
    return (row["graph"], row["block"], row["name"], row["op"], row["dtype"], tuple(row["shape"]),
            row["seq"])


def log_of(manifest_path, m):
    """the run log that produced this manifest, where tools/llama_dump_run.sh put it"""
    cand = os.path.join(ROOT, "data", "dumps", f"{m['date']}-llama", "logs", m["input"]["tag"] + ".log")
    return cand if os.path.isfile(cand) else None


def census_from_log(manifest_path, m):
    lp = log_of(manifest_path, m)
    if lp is None:
        return None, None, None
    rows = []
    missing = {}
    with open(lp, errors="replace") as f:
        for line in f:
            line = line.strip()
            mm = CENSUS_RE.match(line)
            if mm:
                rows.append(mm.groupdict())
                continue
            mn = MISSING_NAMES_RE.match(line)
            if mn:
                names = mn.group("missing")
                missing[(int(mn.group("graph")), int(mn.group("block")))] = (
                    set() if names == "none" else set(names.split("|")))
    return rows, lp, missing


def cmd_self(paths):
    failures = 0
    for mp in paths:
        m, dump_dir = load(mp)
        print(f"== {mp}")
        # --- the gguf binding -------------------------------------------------------------
        pin = m["gguf"]["pinned_sha256"]
        got = m["gguf"]["sha256"]
        gguf_path = m["gguf"]["path"]
        ok = got == pin
        print(f"  gguf path={gguf_path}")
        print(f"  gguf sha256={got}")
        print(f"  gguf pin   ={pin} matches_pin={int(ok)}")
        if not ok:
            failures += 1
            print("  FAIL gguf digest does not equal the pin")
        if os.path.isfile(gguf_path):
            on_disk = sha256_file(gguf_path)
            print(f"  gguf on-disk digest={on_disk} matches_manifest={int(on_disk == got)}")
            if on_disk != got:
                failures += 1
                print("  FAIL gguf bytes on disk do not hash to the manifest digest")
        else:
            failures += 1
            print("  FAIL gguf not present on disk, cannot bind the dump to a weight file")

        # --- every dumped row: file exists, length and digest as recorded -----------------
        n_rows = len(m["dumps"])
        n_checked = n_missing = n_len = n_hash = n_outside = 0
        for row in m["dumps"]:
            if not row.get("written"):
                continue
            p = row["file"]
            if os.path.dirname(os.path.abspath(p)) != os.path.abspath(dump_dir):
                n_outside += 1
                continue
            n_checked += 1
            if not os.path.isfile(p):
                n_missing += 1
                continue
            if os.path.getsize(p) != row["bytes"]:
                n_len += 1
                continue
            if sha256_file(p) != row["sha256"]:
                n_hash += 1
        print(f"  rows={n_rows} files_checked={n_checked} missing={n_missing} "
              f"length_mismatch={n_len} digest_mismatch={n_hash} outside_dump_dir={n_outside}")
        print("  census arithmetic, as asserted and checked below: every node lands in exactly one "
              "of three write classes — matched (a name in the derived expected set), unexpected "
              "(a name the builder chose that is not in it), anonymous (no name the builder chose: "
              "ggml's node_%d/leaf_%d, or a view derived from one). So "
              "dumped + not_written == matched + unexpected + anonymous, where dumped counts bytes "
              "written and not_written counts the class members skipped (zero bytes, size cap, or "
              "anonymous/unexpected writing turned off). Separately `missing` counts DISTINCT "
              "expected names never seen, so missing == expected - |seen_expected|; `matched` counts "
              "WRITES including duplicates, so matched >= |seen_expected| and matched + missing != "
              "expected whenever a name is written twice (duplicates > 0). `state` is not a fourth "
              "bucket: it counts writes whose class is state, drawn from all three")
        if n_checked == 0:
            failures += 1
            print("  FAIL no dumped file to check: the manifest claims bytes that are not there")
        if n_outside:
            failures += 1
            print(f"  FAIL {n_outside} file(s) recorded outside the run's dump directory")
        for label, cnt in (("missing", n_missing), ("length_mismatch", n_len),
                           ("digest_mismatch", n_hash)):
            if cnt:
                failures += 1
                print(f"  FAIL {label}={cnt}")

        bad = 0

        # --- the geometry, read from the checkpoint config, not from llama.cpp ----------------
        # The instrument reports which blocks are recurrent from llama.cpp's hparams. That is a
        # second opinion, not an authority: if llama.cpp's is_recr() and the pinned config disagreed,
        # every per-block census row would still be internally consistent and wrongly labelled.
        if not os.path.isfile(CONFIG_PATH):
            bad += 1
            print(f"  FAIL pinned config {CONFIG_PATH} not found: the block kinds cannot be checked "
                  f"independently of llama.cpp")
        else:
            tc = json.load(open(CONFIG_PATH))["text_config"]
            types = tc["layer_types"]
            mdl = m["model"]
            if len(types) != mdl["n_layer"]:
                bad += 1
                print(f"  FAIL config has {len(types)} layer_types, the run reports "
                      f"n_layer={mdl['n_layer']}")
            want_linear = [il for il, t in enumerate(types) if t == "linear_attention"]
            want_full = [il for il, t in enumerate(types) if t == "full_attention"]
            if sorted(want_linear + want_full) != list(range(len(types))):
                bad += 1
                print("  FAIL config layer_types is not a partition into the two known kinds")
            for label, want in (("linear_layers", want_linear), ("full_attention_layers", want_full)):
                got_l = sorted(mdl.get(label, []))
                if got_l != sorted(want):
                    bad += 1
                    print(f"  FAIL model.{label} = {got_l[:6]}… != config {sorted(want)[:6]}… "
                          f"({len(got_l)} vs {len(want)} entries)")
            for il, t in enumerate(types):
                want_recr = 0 if t == "full_attention" else 1
                if mdl["is_recr_by_layer"][il] != want_recr:
                    bad += 1
                    print(f"  FAIL is_recr_by_layer[{il}] = {mdl['is_recr_by_layer'][il]}, config "
                          f"says {t}")
                    break
            for nm, want_n in (("n_expert", tc["num_experts"]),
                               ("n_expert_used", tc["num_experts_per_tok"]),
                               ("n_embd", tc["hidden_size"])):
                if mdl[nm] != want_n:
                    bad += 1
                    print(f"  FAIL model.{nm} = {mdl[nm]}, config says {want_n}")
            for c in m["census"]:
                if c["block"] >= 0:
                    want_kind = "linear" if types[c["block"]] == "linear_attention" else "full"
                    if c["kind"] != want_kind:
                        bad += 1
                        print(f"  FAIL census graph={c['graph']} block={c['block']}: kind "
                              f"{c['kind']}, config says {want_kind}")
                        break

        # --- census arithmetic, against the log the instrument printed ---------------------
        # The manifest carries the class counts and the name sets; the log additionally carries
        # `missing`. Both are produced by the same run, so they must agree row for row, and the log
        # rows must satisfy matched + missing == expected in every block.
        rows, lp, missing = census_from_log(mp, m)
        sets, weights = expected_sets_from_header()
        n_census_manifest = len(m["census"])
        print(f"  census rows={n_census_manifest} log_rows={'none' if rows is None else len(rows)} "
              f"log={lp}")
        if rows is None:
            bad += 1
            print("  FAIL run log not found; the census cannot be cross-checked against a second "
                  "source (the log is written by tools/llama_dump_run.sh next to the dump)")
        else:
            if len(rows) != n_census_manifest:
                bad += 1
                print("  FAIL the manifest census and the log census disagree on row count")
            by_key = {(int(r["graph"]), int(r["block"])): r for r in rows}
            if len(by_key) != len(rows):
                bad += 1
                print("  FAIL duplicate (graph, block) census rows in the log")
            for c in m["census"]:
                k = (c["graph"], c["block"])
                r = by_key.get(k)
                if r is None:
                    bad += 1
                    print(f"  FAIL census graph={c['graph']} block={c['block']}: in the manifest, "
                          f"absent from the log")
                    continue
                for f_ in ("kind", "dumped", "expected", "matched", "unexpected", "anonymous",
                           "state", "weights_excluded", "weights_unknown", "duplicates",
                           "not_written", "too_big"):
                    a, b = str(c[f_]), r[f_]
                    if a != b:
                        bad += 1
                        print(f"  FAIL census graph={k[0]} block={k[1]} field {f_}: "
                              f"manifest={a} log={b}")
                if int(r["missing"]) != int(r["expected"]) - len(c["seen_expected"]):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: missing {r['missing']} != "
                          f"expected {r['expected']} - {len(c['seen_expected'])} distinct names seen")
                # the independent denominator: this script's own expansion of the generated header
                want = expand(sets, c["kind"])
                if len(want) != int(r["expected"]):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]} kind={c['kind']}: the instrument "
                          f"printed expected={r['expected']}, the header expands to {len(want)}")
                seen_or_missing = set(c["seen_expected"]) | missing.get(k, set())
                if seen_or_missing != want:
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: seen_expected ∪ missing is not "
                          f"the header's expected set (only in census: "
                          f"{sorted(seen_or_missing - want)[:5]}; only in header: "
                          f"{sorted(want - seen_or_missing)[:5]})")
                if int(r["dumped"]) + int(r["not_written"]) != (
                        int(r["matched"]) + int(r["unexpected"]) + int(r["anonymous"])):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: dumped {r['dumped']} + "
                          f"not_written {r['not_written']} != matched {r['matched']} + unexpected "
                          f"{r['unexpected']} + anonymous {r['anonymous']} — the three write "
                          f"classes are meant to be disjoint and exhaustive over graph nodes")
                if int(r["state"]) > int(r["dumped"]):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: state {r['state']} exceeds "
                          f"dumped {r['dumped']}; state is a class of written tensors")
                if int(r["duplicates"]) > int(r["matched"]) + int(r["unexpected"]) + int(r["anonymous"]):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: duplicates exceeds the write "
                          f"count, which cannot happen if duplicates counts repeats")
                for nm in c["parameters"]:
                    fam = param_family(nm)
                    if fam is None or fam not in weights:
                        bad += 1
                        print(f"  FAIL graph={k[0]} block={k[1]}: parameter {nm} is not a derived "
                              f"weight family, so its exclusion was unclassified")
                if c["matched"] < len(c["seen_expected"]):
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: matched {c['matched']} is below "
                          f"the {len(c['seen_expected'])} distinct expected names it saw")
                if c["expected"] == 0:
                    bad += 1
                    print(f"  FAIL census graph={k[0]} block={k[1]}: the expected denominator is 0")
        for c in m["census"]:
            if c["block"] >= 0 and c["dumped"] == 0:
                bad += 1
                print(f"  FAIL census graph={c['graph']} block={c['block']}: dumped 0")
            if c["block"] >= 0 and c["weights_excluded"] == 0:
                bad += 1
                print(f"  FAIL census graph={c['graph']} block={c['block']}: the parameter guard "
                      f"never fired, so nothing proves weights were excluded")
            if c["weights_unknown"]:
                bad += 1
                print(f"  FAIL census graph={c['graph']} block={c['block']}: a parameter carried a "
                      f"name outside the derived weight-family set")
        if n_census_manifest == 0:
            bad += 1
            print("  FAIL empty census")
        # ---- aliasing must-catch ------------------------------------------------------------
        # A name the builder emits once per block cannot carry the same bytes in every block. This is
        # the check that the capture reads live memory: reading an intermediate after ggml recycled
        # its compute-buffer region produces exactly this signature — one digest shared by all 40
        # blocks, stable across runs, and still sensitive to the prompt, so neither the determinism
        # control nor the prompt-change control can see it. It fired on the first version of the
        # instrument, which walked the graph after graph_compute() returned: 39 blocks of ffn_moe_out
        # shared one digest and ffn_moe_probs held logits-range values.
        MIN_BLOCKS = 5
        zero_memo = {}
        def all_zero(digest, nbytes):
            if nbytes <= 0:
                return True
            if digest not in zero_memo:
                zero_memo[digest] = (digest == hashlib.sha256(b"\0" * nbytes).hexdigest())
            return zero_memo[digest]
        by_name = {}
        for row in m["dumps"]:
            if not row.get("written"):
                continue
            by_name.setdefault((row["graph"], row["name"], row["seq"]), {})[row["block"]] = row
        aliased = []
        for key in sorted(by_name):
            per_block = by_name[key]
            if len(per_block) < MIN_BLOCKS:
                continue
            digs = {r["sha256"] for r in per_block.values()}
            if len(digs) != 1:
                continue
            d = next(iter(digs))
            nbytes = max(int(r["bytes"]) for r in per_block.values())
            if all_zero(d, nbytes):
                continue
            aliased.append((key, len(per_block), d))
        for key, nb, d in aliased[:10]:
            bad += 1
            print(f"  FAIL aliasing graph={key[0]} name={key[1]!r} seq={key[2]}: {nb} blocks share "
                  f"digest {d[:16]}... and those bytes are not all zero — the capture read a buffer "
                  f"ggml had already recycled")
        print(f"  aliasing: {len(aliased)} group(s) of >={MIN_BLOCKS} blocks sharing one non-zero "
              f"digest (witness: this counter is non-zero on a post-compute walk)")

        # graph count consistency between the log, the manifest and the records
        graphs_manifest = len(m["graphs"])
        if graphs_manifest == 0:
            bad += 1
            print("  FAIL no graph records: nothing was captured")
        for g in m["graphs"]:
            if g["n_tokens"] != len(g["token_ids"]):
                bad += 1
                print(f"  FAIL graph {g['index']}: n_tokens {g['n_tokens']} != "
                      f"len(token_ids) {len(g['token_ids'])}")
        failures += bad
        verdict = "PASS" if bad == 0 else "FAIL"
        print(f"  verdict={verdict} gguf_sha256={got}")
    print(f"VERDICT {'PASS' if failures == 0 else 'FAIL'} self: {len(paths)} manifest(s), "
          f"{failures} failure(s)")
    return 0 if failures == 0 else 1


def compare(ma_path, mb_path, mode):
    ma = load(ma_path)[0]
    mb = load(mb_path)[0]
    if ma["gguf"]["sha256"] != mb["gguf"]["sha256"]:
        die(f"the two runs read different weight files: {ma['gguf']['sha256'][:12]} vs "
            f"{mb['gguf']['sha256'][:12]}")
    same_prompt = ma["input"]["prompt"] == mb["input"]["prompt"]
    print(f"A={ma_path}\n  prompt={ma['input']['prompt']!r} tag={ma['input']['tag']}")
    print(f"B={mb_path}\n  prompt={mb['input']['prompt']!r} tag={mb['input']['tag']}")
    print(f"  same_prompt={int(same_prompt)}")

    if mode == "determinism" and not same_prompt:
        die("determinism control needs two runs of the SAME prompt")
    if mode == "prompt-change" and same_prompt:
        die("prompt-change control needs two DIFFERENT prompts")

    # index by (graph, block, name, op, dtype, shape); a digest is only comparable to the digest of
    # a tensor with the same op, dtype and shape
    ia = {key_of(r): r for r in ma["dumps"] if r.get("written")}
    ib = {key_of(r): r for r in mb["dumps"] if r.get("written")}
    common = sorted(set(ia) & set(ib))
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))

    # only rows that were written as bytes are hashable; for the rest the manifest digest field is
    # empty and comparison falls back to the recorded length
    compared = 0
    differing = []
    for k in common:
        ra, rb = ia[k], ib[k]
        if ra["written"] and rb["written"]:
            # trust the file on disk, not the manifest's memory of it
            ha = sha256_file(ra["file"])
            hb = sha256_file(rb["file"])
        else:
            ha, hb = ra.get("sha256", ""), rb.get("sha256", "")
            if ra["bytes"] != rb["bytes"]:
                differing.append((k, "length", ra["bytes"], rb["bytes"]))
                compared += 1
                continue
        compared += 1
        if ha != hb:
            differing.append((k, "digest", ha[:12], hb[:12]))

    print(f"  comparable={len(common)} compared={compared} differing={len(differing)} "
          f"only_in_A={len(only_a)} only_in_B={len(only_b)}")
    print("  note: comparison is file-vs-file on disk, keyed on (graph, block, name, op, dtype, "
          "shape, seq) so tensors sharing a name inside a block are each compared; a tensor written "
          "without bytes is compared on its recorded length")
    if only_a[:5]:
        print("  only_in_A sample: " + ", ".join("/".join(map(str, k[:3])) for k in only_a[:5]))
    if only_b[:5]:
        print("  only_in_B sample: " + ", ".join("/".join(map(str, k[:3])) for k in only_b[:5]))

    fail = 0
    if compared == 0:
        print("  FAIL compared == 0: the comparison was vacuous")
        fail += 1

    def first_differences(limit=8):
        for k, kind, a, b in differing[:limit]:
            print(f"    graph={k[0]} block={k[1]} name={k[2]} op={k[3]} dtype={k[4]} "
                  f"shape={k[5]} {kind}: A={a} B={b}")

    if mode == "determinism":
        if differing:
            print(f"  FAIL {len(differing)} tensor(s) differ between two runs of one prompt")
            first_differences()
            fail += 1
        else:
            print("  OK every compared tensor is byte-identical across the two runs")
        verdict = 0 if fail == 0 else 1
        print(f"VERDICT {'PASS' if verdict == 0 else 'FAIL'} determinism: compared={compared}, "
              f"differing={len(differing)}, {fail} failure(s)")
        return verdict
    else:
        if not differing:
            print("  FAIL no tensor changed when the prompt changed: the dump is not reading the graph")
            fail += 1
        else:
            blocks_changed = sorted({k[1] for k, _, _, _ in differing})
            graphs_changed = sorted({k[0] for k, _, _, _ in differing})
            print(f"  blocks with a change: {blocks_changed}")
            print(f"  graphs with a change: {graphs_changed}")
            first_differences()
            logits = [k for k, _, _, _ in differing
                      if k[2] in ("model.result_output", "result_output")]
            if not logits:
                print("  FAIL the logits did not change under a changed prompt")
                fail += 1
            else:
                print(f"  OK logits changed: {['/'.join(map(str, k[:3])) for k in logits]}")
            n_layer = ma["model"]["n_layer"]
            last = n_layer - 1
            if not any(k[1] == last for k, _, _, _ in differing):
                print(f"  FAIL nothing in the last block (block {last}) changed")
                fail += 1
            else:
                print(f"  OK block {last} changed")
        fail = 0 if fail == 0 else 1
        print(f"VERDICT {'PASS' if fail == 0 else 'FAIL'} prompt-change: compared={compared}, "
              f"differing={len(differing)}, {fail} failure(s)")
        return 0 if fail == 0 else 1


def main():
    if len(sys.argv) < 2:
        die(__doc__)
    mode = sys.argv[1]
    args = sys.argv[2:]
    if mode == "self":
        if not args:
            die("usage: self <manifest> [...]")
        return cmd_self(args)
    if mode in ("determinism", "prompt-change"):
        if len(args) != 2:
            die(f"usage: {mode} <manifestA> <manifestB>")
        return compare(args[0], args[1], mode)
    die(f"unknown subcommand {mode!r}")


if __name__ == "__main__":
    sys.exit(main())
