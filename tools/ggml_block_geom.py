#!/usr/bin/env python3
"""ggml_block_geom.py — block geometry (elements, bytes, bpw) read out of the pinned ggml source.

Why this exists: two tools in this repo priced a GGUF's width map from hand-typed bits-per-weight
constants and one row disagreed with ggml by 0.0625 bpw (Q5_K was written 5.5625, where the pinned
struct is 176 B per 256 elements = 5.5). A hand-typed table is exactly the shape that survives a
spot-check: four rows right, the reviewer never lands on the fifth. So widths used to estimate a
file's size are read from the pinned header at run time, and any hand-typed table still carried here
is *diffed* against that derivation (assert_matches) instead of trusted.

Two independent readings of the same header must agree, per type:
  1. the sum of the struct's own member sizes (nested block_* resolved recursively);
  2. the header's own `static_assert(sizeof(block_qX) == <expr>, ...)` line, evaluated with the same
     member/macro rules.
Structs this scanner cannot read (they hold an extended `union { struct {...} ...; }`, which needs a
real C parser) are reported as not derived and are never priced by guesswork. Whether that matters is
the caller's call: geom(require=...) aborts when a type it needs is missing.

Element-count-per-block (ELTS) is a claim about the *type*, not a block-overhead reading, and it is
witnessed by data rather than by parsing: a wrong elts breaks `sum(row_bytes) + data_start == file
size` for every tensor of that type, which check_file() and the audit both enforce and abort on.

Derived from engines/llama.cpp/ggml/src/ggml-common.h §block structs (pinned llama.cpp
d327bf1f0780f87b8c01b367b61853b0c291d7c4). `--emit` writes the table and the drift rows to
quantization/evidence/audit-block-geometry-2026-10-26.md so the numbers are checkable statically.

Usage:
  python3 tools/ggml_block_geom.py                    # derived table (rc 0 if it derived)
  python3 tools/ggml_block_geom.py --require Q8_0,Q4_K,Q5_K,Q6_K
  python3 tools/ggml_block_geom.py --emit             # also write the evidence markdown
  python3 tools/ggml_block_geom.py --check PATH       # also assert a GGUF's size reconciles
  python3 tools/ggml_block_geom.py --check-history    # exit non-zero on the retired table's drift
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "engines/llama.cpp/ggml/src/ggml-common.h")
HEADER_REL = "engines/llama.cpp/ggml/src/ggml-common.h"
EMIT = os.path.join(ROOT, "quantization/evidence/audit-block-geometry-2026-10-26.md")

# ggml block struct -> stored ggml_type name (enum read out of ggml.h at the same pin).
STRUCT_TO_TYPE = {
    "block_q1_0": "Q1_0", "block_q2_0": "Q2_0", "block_q3_0": "Q3_0", "block_q4_0": "Q4_0",
    "block_q4_1": "Q4_1", "block_q5_0": "Q5_0", "block_q5_1": "Q5_1", "block_q2_K": "Q2_K",
    "block_q3_K": "Q3_K", "block_q4_K": "Q4_K", "block_q5_K": "Q5_K", "block_q6_K": "Q6_K",
    "block_q8_0": "Q8_0", "block_q8_1": "Q8_1", "block_q8_K": "Q8_K", "block_iq2_xxs": "IQ2_XXS",
    "block_iq2_xs": "IQ2_XS", "block_iq2_s": "IQ2_S", "block_iq3_s": "IQ3_S",
    "block_iq3_xxs": "IQ3_XXS", "block_iq1_s": "IQ1_S", "block_iq1_m": "IQ1_M",
    "block_iq4_nl": "IQ4_NL", "block_iq4_xs": "IQ4_XS", "block_q4_0_q8_0": "Q4_0_Q8_0",
    "block_mxfp4": "MXFP4", "block_nvfp4": "NVFP4",
}
# Elements per block per type — a claim about the type, witnessed by the file-size identity.
ELTS = {"Q1_0": 32, "Q2_0": 16, "Q3_0": 16, "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
        "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_0": 32, "Q8_1": 32,
        "Q8_K": 256, "IQ2_XXS": 256, "IQ2_XS": 256, "IQ2_S": 256, "IQ3_S": 256, "IQ3_XXS": 256,
        "IQ1_S": 256, "IQ1_M": 256, "IQ4_NL": 32, "IQ4_XS": 256, "Q4_0_Q8_0": 32, "MXFP4": 32,
        "NVFP4": 64, "F32": 1, "F16": 1, "BF16": 1}

PLAIN = {"uint8_t": 1, "int8_t": 1, "char": 1, "uint16_t": 2, "int16_t": 2, "ggml_half": 2,
         "_Float16": 2, "uint32_t": 4, "int32_t": 4, "float": 4, "ggml_half2": 4,
         "ggml_half4": 8, "uint64_t": 8, "int64_t": 8, "unsigned short": 2, "short": 2,
         "size_t": 8}
SCALAR_TYPES = {"F32": 4, "F16": 2, "BF16": 2}
# What a width map for a llama.cpp-quantized checkpoint actually needs. geom(require=...) aborts if
# one of these cannot be read, because a missing row here silently under-prices a whole family.
REQUIRED = ("F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS")


def _macros(text):
    """Every `#define NAME <integer expression>` in the header, kept as its source text.

    Kept as text because they chain — `IQ3S_N_SCALE` is `QK_K/64`, `QI4_K` is `(QK_K / (4*QR4_K))` —
    so evaluating eagerly would need the same machinery, in the same order, anyway.
    """
    out = {}
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+([A-Z][A-Z0-9_]*)([ \t]+[^\n]*)?$", text, re.M):
        val = re.sub(r"//.*", "", m.group(2) or "").strip()
        if val and re.fullmatch(r"[A-Za-z0-9_+\-*/()\s]+", val) and re.search(r"\d", val):
            out[m.group(1)] = val
    return out


_TOK = re.compile(r"\s*(?:([A-Za-z_]\w*)|(\d+)|(.))")


class _Expr:
    """Recursive-descent evaluator for the C integer expressions these structs and static_asserts use.

    A real parser rather than splitting on operators: `QK_K/64`, `sizeof(ggml_half2) + ...` and
    `sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2` do not split left-to-right, and a mis-split
    size is a silently wrong width — the exact failure this module exists to prevent.
    """

    def __init__(self, text, macros, sizes):
        self.toks, self.i, s = [], 0, text
        while s.strip():
            m = _TOK.match(s)
            if not m:
                raise ValueError("bad token in %r" % text)
            self.toks.append(m.group(1) or m.group(2) or m.group(3))
            s = s[m.end():]
        self.macros, self.sizes = macros, sizes

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, want=None):
        t = self.peek()
        if t is None or (want and t != want):
            raise ValueError("expected %r in %r" % (want, self.toks))
        self.i += 1
        return t

    def expr(self, top=True):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            r = self.term()
            v = v + r if op == "+" else v - r
        if top and self.peek() is not None:
            raise ValueError("trailing %r in %r" % (self.peek(), self.toks))
        return v

    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.eat()
            r = self.factor()
            v = v * r if op == "*" else v // r
        return v

    def factor(self):
        t = self.peek()
        if t == "(":
            self.eat("(")
            v = self.expr(top=False)      # a parenthesised sub-expression ends at ')', on purpose
            self.eat(")")
            return v
        if t == "-":
            self.eat()
            return -self.factor()
        self.eat()
        if t.isdigit():
            return int(t)
        if t == "sizeof":
            self.eat("(")
            inner = self.eat()
            if self.peek() == "*":
                self.eat()
            self.eat(")")
            return self.sym(inner)
        return self.sym(t)

    def sym(self, name):
        if name in self.sizes:
            return self.sizes[name]
        if name in PLAIN:
            return PLAIN[name]
        if name in self.macros:
            return _Expr(self.macros[name], self.macros, self.sizes).expr()
        raise ValueError("unknown symbol %r" % name)


def _eval(expr, macros, sizes):
    """Evaluate a size expression. Unresolvable input raises: a guessed member size is how a derived
    table slides back into being a hand-typed one."""
    return _Expr(expr, macros, sizes).expr()


def _scan_structs(text):
    """Brace-balanced scan for `typedef struct {...} block_x;`.

    A `[^{}]*` regex silently skips every struct that holds an extended `union { struct {...} };` —
    six of them here, including all three K-quants this repo prices — which is the failure mode that
    matters, because a skipped struct becomes a missing row and a missing row is priced as nothing. So:
    scan by brace depth and size inline struct/union members properly; whatever is still unreadable is
    returned as `opaque`, reported and never priced.
    """
    bodies, opaque = {}, {}
    for m in re.finditer(r"typedef\s+struct[^{;]*\{", text):
        i, depth, j = m.end() - 1, 0, m.end() - 1
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        nm = re.match(r"\s*(block_[A-Za-z0-9_]+)\s*;", text[j + 1:j + 56])
        if not nm:
            continue
        sname = nm.group(1)
        body = text[i + 1:j]
        bodies[sname] = (body, text[:m.start()].count("\n") + 1)
    return bodies, opaque


def _split_top(body):
    """Split a member list on ';' that is not inside braces or parens."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == ";" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _decl_members(base, rest, macros, sizes):
    """Sizes of the (possibly comma-separated) declarators of one member declaration."""
    vals = []
    for part in rest.split(","):
        part = re.sub(r"//.*", "", part).strip()
        if not part:
            continue
        am = re.match(r"(\w+)\s*\[\s*([^\]]+)\s*\]", part)
        if am:
            n = _eval(am.group(2), macros, sizes)
        elif re.fullmatch(r"[\w,\s*]+", part):
            n = 1
        else:
            raise ValueError("cannot parse declarator %r" % part)
        item = sizes.get(base) if base.startswith("block_") else PLAIN.get(base)
        if item is None:
            raise ValueError("unknown member type %r" % base)
        vals.append(n * item)
    return vals


def _inline_size(kind, inner, macros, sizes):
    """Size of an inline `struct {...}` (sum of members) or `union {...}` (its largest member).

    llama.cpp aliases a struct field against a packed vector with an anonymous union (d/dmin vs dm), so
    this is not exotic: block_q4_K, block_q5_K and block_q6_K are all built this way at this pin, and
    each one's own static_assert is what proves the sum came out right.
    """
    vals = []
    for decl in _split_top(inner):
        decl = re.sub(r"//[^\n]*", "", decl).strip()
        if not decl:
            continue
        m = re.match(r"(struct|union)\s*\{(.*)\}\s*[\w,\s]*$", decl, re.S)
        if m:
            vals.append(_inline_size(m.group(1), m.group(2), macros, sizes))
            continue
        dm = re.match(r"([A-Za-z_]\w*)(\s+.+)", decl, re.S)
        if not dm:
            raise ValueError("cannot parse member %r" % decl[:60])
        vals.extend(_decl_members(dm.group(1), dm.group(2), macros, sizes))
    if not vals:
        raise ValueError("empty %s body" % kind)
    return sum(vals) if kind == "struct" else max(vals)


def _member_sum(body, macros, sizes):
    """Sum the byte sizes of every member of one struct body (inline unions included)."""
    clean = re.sub(r"(?:GGML_EXTENSION|__extension__)\s*\b(struct|union)\b", r"\1", body)
    total = 0
    for decl in _split_top(clean):
        decl = re.sub(r"//[^\n]*", "", decl).strip()
        if not decl:
            continue
        m = re.match(r"(struct|union)\s*\{(.*)\}\s*[\w,\s]*$", decl, re.S)
        if m:
            total += _inline_size(m.group(1), m.group(2), macros, sizes)
            continue
        dm = re.match(r"([A-Za-z_]\w*)(\s+.+)", decl, re.S)
        if not dm:
            raise ValueError("cannot parse member %r" % decl[:60])
        total += sum(_decl_members(dm.group(1), dm.group(2), macros, sizes))
    return total


def block_geom(header_path=None, require=REQUIRED):
    """{type: {"struct","elts","bytes","bpw","line","claim"}} read from the pinned header.

    Raises if the header is missing, if a `require`d type cannot be read, or if a struct's member sum
    disagrees with that header's own static_assert for it.
    """
    path = header_path or HEADER
    if not os.path.exists(path):
        raise FileNotFoundError("%s missing: the pinned llama.cpp checkout is what prices a width "
                                "here. Clone it at the pin recorded in checkpoints/README.md; do "
                                "not hand-type the numbers back in." % path)
    text = open(path, encoding="utf-8", errors="replace").read()
    macros = _macros(text)
    bodies, opaque = _scan_structs(text)
    claims = {}
    for m in re.finditer(r"static_assert\(\s*sizeof\((block_[A-Za-z0-9_]+)\)\s*==\s*([^,]+),", text):
        claims[m.group(1)] = m.group(2).strip()
    sizes, order, failed = {}, [], {}
    for _ in range(len(bodies) + 1):
        progressed = False
        for sname, (body, _) in list(bodies.items()):
            if sname in sizes or sname in failed:
                continue
            if any(n not in sizes for n in set(re.findall(r"\b(block_[A-Za-z0-9_]+)\b", body))):
                continue                       # sibling not summed yet; next pass
            try:
                sizes[sname] = _member_sum(body, macros, sizes)
            except ValueError as e:
                failed[sname] = str(e)
                continue
            order.append(sname)
            progressed = True
        if not progressed:
            break
    unread = sorted(set(bodies) - set(sizes) - set(failed))
    for sname, why in sorted(opaque.items()):
        print("note: %s (line %d) not derived — %s" % (sname, why[1], why[0]), file=sys.stderr)
    for sname in sorted(failed):
        print("note: %s not derived — %s" % (sname, failed[sname]), file=sys.stderr)
    for sname in unread:
        print("note: %s not derived — unresolved members" % sname, file=sys.stderr)
    out = {}
    for sname in order:
        tname = STRUCT_TO_TYPE.get(sname, sname.upper())
        if tname not in ELTS:
            continue
        claim = claims.get(sname)
        if claim:
            cv = _eval(claim, macros, sizes)
            if cv != sizes[sname]:
                raise ValueError("%s: member sum %d disagrees with the header's own static_assert "
                                 "(%s = %d) — do not price widths from a header you cannot read"
                                 % (sname, sizes[sname], claim, cv))
        out[tname] = {"struct": sname, "elts": ELTS[tname], "bytes": sizes[sname],
                      "bpw": 8.0 * sizes[sname] / ELTS[tname], "line": bodies[sname][1],
                      "claim": claim or "(no static_assert)",
                      "claim_value": _eval(claim, macros, sizes) if claim else None}
    for t, nb in SCALAR_TYPES.items():
        out[t] = {"struct": None, "elts": 1, "bytes": nb, "bpw": 8.0 * nb, "line": None,
                  "claim": "(scalar type)", "claim_value": None}
    miss = sorted(t for t in require if t not in out)
    if miss:
        raise ValueError("could not derive the width of %s from %s — refusing to price them by "
                         "guessing" % (miss, HEADER_REL))
    return out


_GEOM = None


def geom(require=()):
    global _GEOM
    if _GEOM is None:
        _GEOM = block_geom(require=tuple(require))
    elif any(t not in _GEOM for t in require):
        _GEOM = block_geom(require=tuple(require))
    return _GEOM


def _lookup(tname):
    g = geom()
    for key in (tname, tname.upper(), tname.replace("-", "_").upper()):
        if key in g:
            return g[key]
    raise KeyError("no derived block geometry for %r (derived rows: %s)"
                   % (tname, ",".join(sorted(g))))


def block_elts(tname):
    return _lookup(tname)["elts"]


def block_bytes(tname):
    return _lookup(tname)["bytes"]


def bpw_of(tname):
    """bits per weight including block overhead, for exact-sizing a tensor of `tname`."""
    return _lookup(tname)["bpw"]


def row_bytes(n_elts, tname):
    """exact on-disk bytes for one tensor: whole blocks, so a partial block still costs a block."""
    g = _lookup(tname)
    return -(-n_elts // g["elts"]) * g["bytes"]


def assert_matches(table):
    """Diff a hand-typed {type: bpw} table against the derived geometry; returns the drift rows.

    Empty return means the table is honest. A non-empty return is a finding to act on, not a line to
    print and forget: a stale row silently misprices every GiB figure built from it.
    """
    g = geom()
    drift = []
    for tname, bpw in sorted(table.items()):
        if tname not in g:
            drift.append({"type": tname, "stated_bpw": float(bpw), "derived_bpw": None,
                          "delta_bpw": None, "note": "not derived from " + HEADER_REL})
        elif abs(g[tname]["bpw"] - float(bpw)) > 1e-9:
            drift.append({"type": tname, "stated_bpw": float(bpw), "derived_bpw": g[tname]["bpw"],
                          "delta_bpw": round(float(bpw) - g[tname]["bpw"], 6), "note": ""})
    return drift


def check_file(path):
    """sum(row_bytes) + data_start == file size, using the derived geometry. Caller aborts on False."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gguf_reader as gr
    g = gr.Gguf(path)                              # header parse only; no tensor-data byte is read
    tot, per = 0, {}
    for t in g.tensors:
        rb = row_bytes(t.n_elts, t.type_name)
        if rb != t.row:
            raise ValueError("%s: derived row_bytes %d != gguf_reader's %d"
                             % (t.name, rb, t.row))
        tot += rb
        p = per.setdefault(t.type_name, [0, 0, 0])
        p[0] += 1
        p[1] += t.n_elts
        p[2] += rb
    ok = tot + g.data_start == g.size
    return ok, {"sum_row_bytes": tot, "data_start": g.data_start, "file_size": g.size,
                "delta": g.size - (tot + g.data_start),
                "per_type": {k: tuple(v) for k, v in per.items()}}


# The table this repo used to hand-type, kept only so the drift stays visible and re-auditable. It is
# not a live input: no arithmetic reads it. `--check-history` exits non-zero on its drift rows, so the
# diff is still assertable without making the ordinary "print the geometry" run red.
RETIRED_HAND_TABLE = {"Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5625, "Q4_K": 4.5, "Q4_0": 4.5,
                      "IQ4_XS": 4.25, "MXFP4": 4.25}


def emit_markdown():
    g = geom()
    rows = assert_matches(RETIRED_HAND_TABLE)
    out = [
        "# Block geometry, read out of the pinned ggml header",
        "",
        "Generated by `python3 tools/ggml_block_geom.py --emit`. Why: `tools/quant_scenarios.py` and "
        "`tools/gguf_header_types.py` both priced width maps from hand-typed bits-per-weight and one "
        "row (Q5_K) sat at 5.5625 where the pinned struct is 176 B per 256 elements = 5.5 bpw. That "
        "shape of error survives a spot-check — four rows right, the reviewer never lands on the "
        "fifth — so widths now come from `%s` (llama.cpp pin "
        "`d327bf1f0780f87b8c01b367b61853b0c291d7c4`) at run time: each struct's member sizes *and* "
        "the header's own `static_assert`, which must agree or the tool aborts. `assert_matches()` "
        "diffs any hand-typed table instead of trusting it." % HEADER_REL,
        "",
        "| stored type | struct (line at the pin above) | elts/block | bytes/block | bpw | the header's own second reading |",
        "|---|---|---:|---:|---:|---|",
    ]
    for t in sorted(g, key=lambda x: (g[x]["struct"] is None, x)):
        e = g[t]
        second = ("`%s` = %d ✓" % (e["claim"], e["claim_value"])) if e["claim_value"] is not None \
            else e["claim"]
        out.append("| %s | %s | %d | %d | %s | %s |" % (
            t, "%s (`ggml-common.h:%d`" % (e["struct"], e["line"]) + ")" if e["line"] else
            (e["struct"] or "(scalar)"), e["elts"], e["bytes"],
            ("%.4f" % e["bpw"]).rstrip("0").rstrip("."), second))
    out += ["", "## The hand-typed row that was wrong", "",
            "| type | was | derived from the header | delta |", "|---|---:|---:|---:|"]
    for r in rows:
        out.append("| %s | %s | %s | %s |" % (
            r["type"], r["stated_bpw"],
            r["derived_bpw"] if r["derived_bpw"] is not None else "not derivable",
            ("%+g" % r["delta_bpw"]) if r["delta_bpw"] is not None else "n/a — " + r["note"]))
    out += ["",
            "`MXFP4` derives from member sizes only (17 B per 32 elements = 4.25 bpw): this header "
            "carries no `static_assert` for it, so that row is a single reading, unlike every K-quant "
            "row above it. It is a policy-row width, so it is worth knowing which rows are doubled and "
            "which are not.",
            "",
            "What the wrong row was worth on the audited file, computed from the file's own tensor "
            "records and not from the bpw delta alone: **74.00 MiB** over its 37 Q5_K tensors — "
            "arithmetic in quantization/evidence/audit-published-quant-2026-10-26.md, section \"A "
            "hand-typed width: the 74 MiB, and where my 232 MiB went\". It touched only the "
            "published-file width map. No `quant_scenarios.py` policy row uses Q5_K, so policy A's "
            "19.59 GiB resident and 2.331 GiB/token do not move.",
            "",
            "Element-count-per-block is a claim about the type, not a reading of a struct, and it is "
            "witnessed by data instead: `check_file()` fails unless `sum(row_bytes) + data_start` "
            "reproduces a real file's size exactly. On the audited file it does: 22,123,538,944 + "
            "10,990,048 = 22,134,528,992 bytes.",
            ""]
    os.makedirs(os.path.dirname(EMIT), exist_ok=True)
    open(EMIT, "w", encoding="utf-8").write("\n".join(out))
    # A check that cannot produce output cannot fail: count what was actually written and make the
    # table row count equal the geometry it claims to report, so a silently-empty emit is an error.
    back = open(EMIT, encoding="utf-8").read()
    rows_back = sum(1 for line in back.splitlines() if line.startswith("|"))
    want = len(g) + len(rows) + 4          # geom rows + drift rows + 2 headers + 2 separator rows
    if rows_back < want:
        raise ValueError("emitted table has %d table rows, expected at least %d (geom %d + drift %d "
                         "+ headers) — the emit is not reporting what it claims" % (rows_back, want,
                                                                                   len(g), len(rows)))
    return EMIT


def main():
    args = sys.argv[1:]
    if "--emit" in args:
        print("wrote", emit_markdown())
        return 0
    require = tuple(args[args.index("--require") + 1].split(",")) if "--require" in args else REQUIRED
    try:
        g = block_geom(require=require)
    except (FileNotFoundError, ValueError) as e:
        print("ABORT: %s" % e)
        return 1
    print("# derived from %s" % HEADER_REL)
    print("%-10s %-16s %9s %11s %9s  %s" % ("type", "struct", "elts/blk", "bytes/blk", "bpw",
                                            "header's own claim"))
    for t in sorted(g, key=lambda x: (g[x]["struct"] is None, x)):
        e = g[t]
        print("%-10s %-16s %9d %11d %9s  %s" % (
            t, e["struct"] or "(scalar)", e["elts"], e["bytes"],
            ("%.4f" % e["bpw"]).rstrip("0").rstrip("."), e["claim"]))
    if "--check" in args:
        path = args[args.index("--check") + 1]
        ok, d = check_file(path)
        print("\nsize identity on %s: sum(row_bytes)=%d + data_start=%d vs file_size=%d -> %s"
              % (os.path.basename(path), d["sum_row_bytes"], d["data_start"], d["file_size"],
                 "RECONCILES" if ok else "MISMATCH delta=%+d B" % d["delta"]))
        for t in sorted(d["per_type"]):
            n, e, b = d["per_type"][t]
            print("  %-6s tensors=%3d elts=%14d bytes=%14d %9.4f GiB bpw=%.4f"
                  % (t, n, e, b, b / 2.0 ** 30, 8.0 * b / e))
        return 0 if ok else 1
    if "--check-history" in args:
        drift = assert_matches(RETIRED_HAND_TABLE)
        bad = [r for r in drift if r["delta_bpw"]]
        print("\nretired hand-typed table diffed against the header: %d drift row(s)" % len(bad))
        for r in bad:
            print("  %s: table said %s, header says %s (%+g bpw)"
                  % (r["type"], r["stated_bpw"], r["derived_bpw"], r["delta_bpw"]))
        return 1 if bad else 0
    print("\nhistorical note: the retired hand table differs from the header by 1 row (Q5_K). "
          "`--check-history` turns that into an exit code; no live arithmetic reads the retired table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
