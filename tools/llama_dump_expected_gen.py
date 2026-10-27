#!/usr/bin/env python3
"""llama_dump_expected_gen.py — derive the expected named-intermediate set for the llama.cpp
instrumentation (tools/LLAMA_DUMP_NOTES.md) from the vendored sources, never by typing names.

Why this file exists: tools/llama-instrument.patch installs a dumper into the vendored
engines/llama.cpp checkout that reads named graph intermediates. Its census ("how many tensors did
block N produce, how many of them are the ones we expect, what else is in the graph") is only
non-vacuous if the *expected* set comes from somewhere other than the dumper itself
(AGENTS.md §A check that can produce no output cannot fail, pitfalls/40 §derived must-catch sets).
Two independent derivations happen here:

  1. activations — the `cb(<tensor>, "<name>", <il>)` calls in the vendored graph builders
     (`src/models/qwen35moe.cpp`, plus the helper bodies it transitively calls in
     `src/models/delta-net-base.cpp` and `src/llama-graph.cpp`). A name the vendored source can
     emit for a block but that does not appear in the graph is reported as `missing`, so a
     renamed-or-dropped intermediate is caught rather than silently skipped.
  2. weights — the `blk.%d.<family>` strings in `src/llama-arch.cpp`'s tensor-name table, filtered
     to the `LLM_TENSOR_*` ids the qwen35moe loader actually asks for. This is the guard that
     keeps 22 GiB of parameters out of the dump: a name that matches a table-derived weight
     family is counted `weights_excluded` and its bytes are never written.

Usage:
    tools/llama_dump_expected_gen.py [--src engines/llama.cpp] [--out FILE] [--check]

`--check` regenerates and compares against the file on disk, exiting 4 on drift — that is the
witness that the committed header is the derived one and not a hand-edited one.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The four sources the derivation reads. Names are file roles, not name lists.
SRC_MODEL = "src/models/qwen35moe.cpp"
SRC_DELTA = "src/models/delta-net-base.cpp"
SRC_GRAPH = "src/llama-graph.cpp"
SRC_ARCH = "src/llama-arch.cpp"
SRC_BASE = "src/llama-model.cpp"

# Graph-builder entry points, named by what the vendored file calls them.
ENTRY_MAIN = "llama_model_qwen35moe::graph::graph"
ENTRY_ATTN_FULL = "llama_model_qwen35moe::graph::build_layer_attn"
ENTRY_ATTN_LINEAR = "llama_model_qwen35moe::graph::build_layer_attn_linear"
ENTRY_FFN = "llama_model_qwen35moe::graph::build_layer_ffn"
ENTRY_LOADERS = [
    "llama_model_qwen35moe::load_arch_tensors",
]

# A qualified name immediately followed by '(' — a definition candidate, confirmed by what
# follows the matching ')': '{' means definition, ';' means call or declaration.
QNAME_RE = re.compile(r"(?<![.\w>])(\w[\w:]*::)?(\w[\w~]*)\s*\(")
CB_RE = re.compile(r"(?m)(?<![.\w])cb\s*\(\s*[A-Za-z_]\w*\s*,\s*"
                   r'"((?:[^"\\]|\\.)*)"\s*,\s*([^()]*?)\)')
IDENT_RE = re.compile(r"(?<![.\w>])([A-Za-z_]\w*)\s*\(")
TN_RE = re.compile(r"(?<![.\w])tn\s*\(\s*(LLM_TENSOR_\w+)")
ARCH_NAME_RE = re.compile(r"(?m)\{\s*(LLM_TENSOR_\w+)\s*,\s*"
                          r'"((?:blk\.%d\.)?[^"]*)"\s*\}')


def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments so commented-out cb() calls cannot inflate a set."""
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(c if c == "\n" else " " for c in text[i:j]))
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def brace_end(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def close_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parens")


def find_definitions(text: str) -> dict[str, str]:
    """qualified name -> body, found by a top-level scan.

    Scanning is sequential and skips every body it accepts, so nested calls, lambdas and
    template arguments inside a function cannot be mistaken for definitions. A candidate is a
    definition only if the first '{' or ';' after its closing paren is the '{'.
    """
    defs: dict[str, str] = {}
    i = 0
    while i < len(text):
        m = QNAME_RE.search(text, i)
        if not m:
            break
        try:
            close = close_paren(text, m.end() - 1)
        except ValueError:
            i = m.end()
            continue
        j = close + 1
        while j < len(text) and text[j] not in "{;":
            j += 1
        if j >= len(text) or text[j] == ";":
            i = m.end()
            continue
        # A definition's own line carries only its return type and qualified name; an `if (f(x)) {`
        # or `while (g()) {` line has a paren before the name and is not a definition.
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start:m.start()]
        if any(c in prefix for c in "(){};"):
            i = m.end()
            continue
        key = (m.group(1) or "") + m.group(2)
        try:
            end = brace_end(text, j)
        except ValueError:
            i = m.end()
            continue
        body = text[j:end + 1]
        if key not in defs or len(body) > len(defs[key]):
            defs[key] = body
        i = end + 1
    return defs


def cb_calls(body: str) -> list[tuple[str, str]]:
    """[(name, il_expr)] from one function body."""
    return [(m.group(1), m.group(2).strip()) for m in CB_RE.finditer(body)]


def callees(body: str, known: set[str]) -> set[str]:
    return {m.group(1) for m in IDENT_RE.finditer(body) if m.group(1) in known}


def closure(entry: str, defs: dict[str, str], known: set[str]) -> list[str]:
    """Transitive set of definition keys reachable from `entry` (by bare function name)."""
    def key_of(name: str) -> str | None:
        cands = [k for k in defs if k == name or k.endswith("::" + name)]
        return cands[0] if cands else None

    seen: set[str] = set()
    stack = [key_of(entry)]
    if stack[0] is None:
        raise SystemExit(f"entry point not found in parsed definitions: {entry}")
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        for c in callees(defs[key], known):
            k2 = key_of(c)
            if k2 and k2 not in seen:
                stack.append(k2)
    return sorted(seen)


def names_from(keys: list[str], defs: dict[str, str], want: str) -> list[str]:
    """cb names in these bodies; want='block' (il is not a literal -1) or 'model'."""
    out: set[str] = set()
    for k in keys:
        for name, il in cb_calls(defs[k]):
            is_model = il == "-1"
            if (want == "model") == is_model:
                out.add(name)
    return sorted(out)


def weight_families(arch_src: str, model_src: str, base_src: str) -> tuple[list[str], dict[str, str]]:
    """arch-table families the qwen35moe loader asks for, keyed by the table string."""
    table = dict((m.group(1), m.group(2)) for m in ARCH_NAME_RE.finditer(arch_src))
    asked: set[str] = set()
    for m in TN_RE.finditer(model_src):
        asked.add(m.group(1))
    # helpers like create_tensor_qkv / create_tensor_gate_up_exps live in llama-model.cpp and are
    # reached from the loader, so their tn() ids are the loader's too.
    loader_body = "".join(
        body for name, body in find_definitions(model_src).items()
        if any(name.endswith(e.split("::")[-1]) for e in ENTRY_LOADERS))
    helper_names = sorted(callees(loader_body,
                                  {k.split("::")[-1] for k in find_definitions(base_src)}))
    base_defs = find_definitions(base_src)
    for h in helper_names:
        for k, body in base_defs.items():
            if k.endswith("::" + h):
                for m in TN_RE.finditer(body):
                    asked.add(m.group(1))
    fams: dict[str, str] = {}
    for tid in sorted(asked):
        s = table.get(tid)
        if not s:
            continue
        if not s.startswith("blk.%d."):
            continue
        fams[s[len("blk.%d."):]] = tid
    return sorted(fams), fams


def emit(sets: dict[str, list[str]], digests: dict[str, str], src: Path,
         fam_ids: dict[str, str]) -> str:
    L = []
    L.append("// GENERATED by tools/llama_dump_expected_gen.py — do not edit by hand.")
    L.append("// Regenerate (and --check) from the repo root: tools/llama_dump_expected_gen.py")
    L.append("// Derived from the vendored checkout at %s:" % src)
    for f, h in sorted(digests.items()):
        L.append("//   %-40s sha256 %s" % (f, h))
    L.append("// Activation names come from cb() calls in the vendored graph builders; weight")
    L.append("// families come from the src/llama-arch.cpp tensor-name table, filtered to the")
    L.append("// LLM_TENSOR_* ids the qwen35moe loader asks for. See the generator docstring.")
    L.append("#pragma once")
    L.append("#define LLAMA_DUMP_EXPECTED_HEADER_VERSION 1")
    for key in ("main", "attn_full", "attn_linear", "ffn", "model_level"):
        L.append("")
        L.append(f"static const char * const LLAMA_DUMP_EXPECTED_{key.upper()}[] = {{")
        for n in sets[key]:
            L.append(f'    "{n}",')
        L.append("};")
        L.append(f"static const unsigned LLAMA_DUMP_EXPECTED_{key.upper()}_N ="
                 f" (unsigned) (sizeof(LLAMA_DUMP_EXPECTED_{key.upper()}) /"
                 f" sizeof(LLAMA_DUMP_EXPECTED_{key.upper()}[0]));")
    L.append("")
    L.append("// blk.%d.<family> strings from the arch tensor table the loader actually creates.")
    L.append("static const char * const LLAMA_DUMP_WEIGHT_FAMILIES[] = {")
    for f in sets["weight_families"]:
        L.append(f'    "{f}",   // {fam_ids[f]}')
    L.append("};")
    L.append("static const unsigned LLAMA_DUMP_WEIGHT_FAMILIES_N ="
             " (unsigned) (sizeof(LLAMA_DUMP_WEIGHT_FAMILIES) /"
             " sizeof(LLAMA_DUMP_WEIGHT_FAMILIES[0]));")
    L.append("")
    return "\n".join(L) + "\n"


def build(src: Path) -> tuple[str, dict[str, list[str]]]:
    texts = {}
    for rel in (SRC_MODEL, SRC_DELTA, SRC_GRAPH, SRC_ARCH, SRC_BASE):
        p = src / rel
        if not p.is_file():
            raise SystemExit(f"missing vendored source: {p}")
        texts[rel] = strip_comments(p.read_text())

    defs: dict[str, str] = {}
    for rel in (SRC_MODEL, SRC_DELTA, SRC_GRAPH):
        defs.update(find_definitions(texts[rel]))
    known = {k.split("::")[-1] for k in defs}

    sets: dict[str, list[str]] = {}
    # `main` is the block-body of the arch graph itself — no closure, or the two attention paths and
    # the whole MoE helper would all be folded into one undifferentiated set.
    main_key = [k for k in defs if k == ENTRY_MAIN or k.endswith("::" + ENTRY_MAIN)][0]
    sets["main"] = names_from([main_key], defs, "block")
    sets["attn_full"] = names_from(closure(ENTRY_ATTN_FULL, defs, known), defs, "block")
    sets["attn_linear"] = names_from(closure(ENTRY_ATTN_LINEAR, defs, known), defs, "block")
    sets["ffn"] = names_from(closure(ENTRY_FFN, defs, known), defs, "block")
    sets["model_level"] = names_from(closure(ENTRY_MAIN, defs, known), defs, "model")

    # A cb name containing '-' would be ambiguous against the "<name>-<il>" spelling that
    # llama_context::graph_get_cb() puts on tensors, which the dumper parses. Refuse, do not guess.
    for key, names in sets.items():
        for n in names:
            if "-" in n:
                raise SystemExit(f"refusing: derived cb name {n!r} (set {key}) contains '-'")
            if not n:
                raise SystemExit(f"refusing: empty cb name in set {key}")

    fams, fam_ids = weight_families(texts[SRC_ARCH], texts[SRC_MODEL], texts[SRC_BASE])
    sets["weight_families"] = fams

    digests = {rel: hashlib.sha256((src / rel).read_bytes()).hexdigest()
               for rel in (SRC_MODEL, SRC_DELTA, SRC_GRAPH, SRC_ARCH, SRC_BASE)}
    return emit(sets, digests, src, fam_ids), sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "engines" / "llama.cpp"))
    ap.add_argument("--out", default=str(REPO / "tools" / "llama-dump-expected.h"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print-sets", action="store_true")
    a = ap.parse_args()

    text, sets = build(Path(a.src))

    if a.print_sets:
        for k in ("main", "attn_full", "attn_linear", "ffn", "model_level", "weight_families"):
            print(f"== {k} ({len(sets[k])})")
            for n in sets[k]:
                print(f"   {n}")

    out = Path(a.out)
    if a.check:
        have = out.read_text() if out.is_file() else ""
        if have != text:
            print(f"DRIFT: {out} is not what the vendored sources derive to."
                  f" Re-run tools/llama_dump_expected_gen.py", file=sys.stderr)
            return 4
        print(f"EXPECTED-HEADER current: {out} ({len(sets['weight_families'])} weight families,"
              f" {len(sets['attn_linear'])} linear-attn names)")
        return 0

    out.write_text(text)
    print(f"wrote {out}")
    for k in ("main", "attn_full", "attn_linear", "ffn", "model_level", "weight_families"):
        print(f"  {k:14s} {len(sets[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
