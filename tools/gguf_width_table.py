#!/usr/bin/env python3
"""gguf_width_table.py — derive a GGUF's true stored bytes per family and PROVE the byte table.

Why this exists, and why it is a checker rather than a table. On 2026-10-26 the audit found that
`tools/gguf_header_types.py` and `tools/quant_scenarios.py` both carried Q5_K at 5.5625 bpw, while
GGML's own `block_q5_K` is 176 B per 256 weights = **5.5 bpw** (`ggml/src/ggml-common.h`, pinned
revision, whose `static_assert` fixes it as `2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8`
= 4 + 12 + 128 + 32 = 176). Exactly one row of a hand-maintained table was wrong, and it survived
review because the rows that get spot-checked are the common ones (Q4_K, Q8_0), which were right.
A table whose entries are typed by hand can fail that way silently forever; this one cannot,
because every number it prints is checked against the file's own size.

THE WITNESS (the reason to trust any of it):
    sum(tensor row bytes) + data_start == file size
If a block size is wrong for any type that occurs in the file, that identity breaks and this tool
exits non-zero. `--selftest` proves the witness is not vacuous by injecting a wrong Q5_K size and
requiring the identity to FAIL.

Usage:
  python3 tools/gguf_width_table.py <file.gguf>                # table + identity assertion
  python3 tools/gguf_width_table.py <file.gguf> --check-header # also re-derive sizes from ggml-common.h
  python3 tools/gguf_width_table.py --selftest                  # proves the identity can fail

Honesty rules baked in (AGENTS.md: "a check that can produce no output cannot fail"):
  * an unmapped ggml_type is a hard failure, not a 0-byte row — a family silently counted as 0
    makes the identity false in a direction that looks like a rounding difference;
  * printing a family table with zero rows exits non-zero;
  * the parsed record count must equal the header's declared n_tensors, or it aborts;
  * `--selftest` must witness the injected fault firing, so the guard is demonstrated, not assumed.
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_header_types as G  # tokenizer only; its BPW table is deliberately NOT used

# (weights per block, bytes per block), transcribed from the pinned ggml-common.h struct fields.
# Each carries the expression it came from so --check-header can re-derive it from the source.
#   Q4_K: half2 dm(4) + scales[12]                          + qs[128]              = 144
#   Q5_K: half2 dm(4) + scales[12] + qh[QK_K/8](32)         + qs[128]              = 176
#   Q6_K: dm[16] + scales[16] + qh[QK_K/4](64)              + ql[128]              = 210
#   Q8_0: half d(2)                                          + qs[QK8_0](32)        = 34  (per 32)
BLK = {
    "F32":    (1,   4,  "4 = sizeof(float)"),
    "F16":    (1,   2,  "2 = sizeof(ggml_half)"),
    "BF16":   (1,   2,  "2 = sizeof(ggml_bfloat16)"),
    "Q8_0":   (32,  34, "2*sizeof(ggml_half) + QK8_0"),
    "Q4_0":   (32,  34, "sizeof(ggml_half) + QK"),
    "Q4_K":   (256, 144, "2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2"),
    "Q5_K":   (256, 176, "2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8"),
    "Q6_K":   (256, 210, "2*sizeof(ggml_half) + QK_K/16 + QK_K/4 + QK_K/2"),
    "Q2_K":   (256,  84, "sizeof(ggml_half) + QK_K/16 + QK_K/4"),
    "Q3_K":   (256, 110, "QK_K*3/8 + QK_K/16 + 2*(QK_K/32) + sizeof(ggml_half)"),
    "IQ4_NL": (256, 144, "sizeof(ggml_half)*2 + QK_K/2"),
    "IQ4_XS": (256, 136, "sizeof(ggml_half)*2 + QK_K/2 + 8"),
    "MXFP4":  (32,   17, "1 scale byte per 32 + 16 nibble bytes"),
}
# The two rows the 2026-10-26 audit disagrees with the older tools about, kept explicit so a
# future "tidy" cannot silently restore them.
KNOWN_TOOL_DISAGREEMENT = {"Q5_K": 5.5625}


def parse_gguf(path):
    """Return (n_tensors, rows, data_start, header_end) using the proven tokenizer."""
    with open(path, "rb") as fh:
        head = fh.read(64 << 20)
    if head[:4] != b"GGUF":
        raise SystemExit("not a GGUF (magic=%r)" % head[:4])
    c = G.Cur(head)
    c.take(4)
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    for _ in range(n_kv):
        k = c.s()
        vt = c.u32()
        G.skip_array(c) if vt == 9 else c.value(vt)
    rows = []
    for _ in range(n_tensors):
        name = c.s()
        nd = c.u32()
        dims = tuple(c.u64() for _ in range(nd))
        dt = c.u32()
        c.u64()
        rows.append((name, dims, dt))
    header_end = c.p
    align = 32
    data_start = (header_end + align - 1) // align * align
    return version, n_tensors, rows, header_end, data_start


def family_bytes(rows, table):
    """(per-type [count,bytes], total, unmapped-set)."""
    fam = {}
    total = 0
    unmapped = set()
    for name, dims, dt in rows:
        tn = G.TYPES.get(dt, "type_%d" % dt)
        if tn not in table:
            unmapped.add(tn)
            continue
        ne = 1
        for d in dims:
            ne *= d
        qk, rb, _expr = table[tn]
        total += ((ne + qk - 1) // qk) * rb
        e = fam.setdefault(tn, [0, 0])
        e[0] += 1
        e[1] += rb * ((ne + qk - 1) // qk)
    return fam, total, unmapped


def check_header(table, header_path):
    """Re-derive K-quant block sizes from ggml-common.h's own static_assert, no hand typing."""
    if not os.path.exists(header_path):
        print("FAIL: no ggml-common.h at %s" % header_path)
        return 1
    src = open(header_path, encoding="utf-8", errors="replace").read()
    consts = {}
    for m in re.finditer(r"#define\s+(QK_K|QK|QK8_0|K_SCALE_SIZE|QI4_K|QI6_K)\s+(\d+)", src):
        consts[m.group(1)] = int(m.group(2))
    bad = 0
    for tn in ("Q4_K", "Q5_K", "Q6_K"):
        # ggml names the struct block_q4_K / block_q5_K / block_q6_K (lower-case q); building this
        # from the type name is where a `%%`-precedence slip put `block_4_kK` on 2026-10-26, and a
        # checker whose pattern cannot match reports "cannot verify" forever without ever being wrong.
        sname = "block_q%s" % tn[1:]
        m = re.search(r"static_assert\(\s*sizeof\(%s\)\s*==\s*([^,]+)," % sname, src)
        if not m:
            print("FAIL: no static_assert for %s — cannot verify, refusing to assume" % sname)
            bad += 1
            continue
        expr = m.group(1).strip()
        try:
            pyexpr = (expr.replace("sizeof(ggml_half)", "2")
                          .replace("sizeof(ggml_half2)", "4")
                          .replace("sizeof(int8_t)", "1")
                          .replace("sizeof(uint8_t)", "1"))
            derived = eval(pyexpr, {"__builtins__": {}}, dict(consts))
        except Exception as e:
            print("FAIL: cannot evaluate %s for %s: %s" % (expr, tn, e))
            bad += 1
            continue
        if derived != table[tn][1]:
            print("FAIL: %s derived %d B from the header, table says %d B" % (tn, derived, table[tn][1]))
            bad += 1
        else:
            print("ok: %s = %d B/256 = %.4f bpw, re-derived from `%s`"
                  % (tn, derived, 8.0 * derived / 256, expr))
    if len(consts) < 2:
        print("FAIL: read no block constants from the header (%s) — the derivation above is vacuous"
              % sorted(consts))
        bad += 1
    else:
        print("ok: block constants read from the header: %s" % dict(sorted(consts.items())))
    return bad


def run(path, header=None):
    version, n_tensors, rows, header_end, data_start = parse_gguf(path)
    size = os.path.getsize(path)
    if len(rows) != n_tensors:
        print("FAIL: parsed %d tensor records, header declares %d — a partial parse must not print a table"
              % (len(rows), n_tensors))
        return 1
    fam, total, unmapped = family_bytes(rows, BLK)
    if unmapped:
        print("FAIL: unmapped ggml_types %s would be counted as 0 bytes and quietly falsify the identity"
              % sorted(unmapped))
        return 1
    if not fam:
        print("FAIL: zero families to print — an empty table cannot pass")
        return 1
    print("GGUF v%d  %s" % (version, path))
    print("n_tensors declared=%d parsed=%d  header_end=%d  data_start=%d  file=%d"
          % (n_tensors, len(rows), header_end, data_start, size))
    print("\n%-8s %-6s %14s %10s %10s   %s" % ("type", "count", "bytes", "GiB", "bpw", "block"))
    print("-" * 92)
    shown = 0
    for tn in sorted(fam, key=lambda t: -fam[t][1]):
        cnt, b = fam[tn]
        qk, rb, expr = BLK[tn]
        bpw = 8.0 * rb / qk
        note = ""
        if tn in KNOWN_TOOL_DISAGREEMENT:
            note = "   <- tools/gguf_header_types.py+quant_scenarios.py say %.4f; file says %.4f" % (
                KNOWN_TOOL_DISAGREEMENT[tn], bpw)
        print("%-8s %-6d %14d %10.4f %10.4f   %3d B / %d wt%s"
              % (tn, cnt, b, b / 2**30, bpw, rb, qk, note))
        shown += 1
    print("-" * 92)
    print("%-8s %-6d %14d %10.4f" % ("TOTAL", sum(v[0] for v in fam.values()), total, total / 2**30))
    if shown == 0 or sum(v[0] for v in fam.values()) != n_tensors:
        print("FAIL: printed %d rows covering %d tensors against %d declared"
              % (shown, sum(v[0] for v in fam.values()), n_tensors))
        return 1
    ok = data_start + total == size
    print("\nWITNESS  sum(row_bytes) + data_start == file size : %s" % ok)
    if not ok:
        print("         off by %d bytes — some block size in BLK is wrong for a type present here"
              % (size - (data_start + total)))
        return 1
    if header:
        print("\nre-deriving K-quant sizes from %s" % header)
        rc = check_header(BLK, header)
        if rc:
            return 1
    print("\nPASS: every printed byte is pinned by the file's own length.")
    return 0


def selftest():
    """Prove the witness can fail. An identity that never fails is not a witness."""
    print("selftest: the identity must FIRE when one block size is wrong")
    path = os.environ.get("Q36_GGUF", "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    if not os.path.exists(path):
        print("SKIP: %s absent — a skipped check is not a passed one, so this is a FAIL" % path)
        return 1
    _, n_tensors, rows, header_end, data_start = parse_gguf(path)
    size = os.path.getsize(path)
    good, total, unmapped = family_bytes(rows, BLK)
    if data_start + total != size:
        print("FAIL: the table does not reproduce the file size even before injection")
        return 1
    fired = 0
    for tn, wrong in (("Q5_K", 178), ("Q4_K", 145), ("Q8_0", 33)):
        bad_tbl = dict(BLK)
        bad_tbl[tn] = (BLK[tn][0], wrong, BLK[tn][2])
        _, bad_total, _ = family_bytes(rows, bad_tbl)
        fires = (data_start + bad_total) != size
        n_touched = good.get(tn, [0, 0])[0]
        print("  inject %-5s -> %3d B/block: identity %s  (%d tensors of that type present)"
              % (tn, wrong, "FIRED" if fires else "did NOT fire", n_touched))
        if n_touched == 0:
            print("  FAIL: %s never appears, so injecting it proves nothing" % tn)
            return 1
        fired += 1 if fires else 0
    if fired != 3:
        print("FAIL: only %d/3 injections fired — the guard is not watching the table" % fired)
        return 1
    print("selftest PASS: 3/3 injected byte-table faults fired, and every injected type is present.")
    return 0


def main():
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    if not a:
        print(__doc__)
        return 2
    path = a[0]
    header = None
    if "--check-header" in a:
        header = a[a.index("--check-header") + 1] if len(a) > a.index("--check-header") + 1 \
            and not a[a.index("--check-header") + 1].startswith("--") else \
            "engines/llama.cpp/ggml/src/ggml-common.h"
    return run(path, header)


if __name__ == "__main__":
    sys.exit(main())
