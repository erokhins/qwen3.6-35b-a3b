#!/usr/bin/env python3
"""gguf_header_types.py — read a GGUF's *header only* and report which ggml type every tensor
actually got, grouped by tensor-name family.

Why this exists: "what quantization does the published Q4_K_M actually use?" is not answerable from
quantize-tool prose or a model card — a K-quant is a *policy over tensors*, and the only
authoritative record of that policy in a published file is the per-tensor type stored in the file's
own header. And we want that without downloading a ~19 GiB file, so this reads a byte range
(default: grow 4 -> 16 -> 32 -> 64 MiB until the tensor-info array parses) and stops.

Layout it relies on (GGUF v2/v3): magic "GGUF", u32 version, u64 n_tensors, u64 n_kv, then n_kv
metadata pairs, then n_tensors tensor-info records (name, nd, dims[], type, offset). On a
248k-vocab model the metadata section carries the whole vocab + merges, so it is many MiB — that is
why a naive 512 KiB read fails, and why the fetch grows instead of guessing.

Usage:
  python3 tools/gguf_header_types.py <url-or-path> [--bytes N] [--raw [K]] [--grep SUBSTR]

Honesty notes:
  * type ids are printed by name, from the ggml_type enum read out of ggml.h (see tools/PROVENANCE.md);
    an id not in the table prints as `type_<id>` — never guessed.
  * widths, and therefore every GiB figure below, are derived each run by tools/ggml_block_geom.py from
    the pinned ggml-common.h struct sizes. They used to be a hand-typed BPW dict in this file and one
    row of it was wrong — Q5_K at 5.5625 bpw where the struct is 176 B/256 = 5.5 — which overpriced
    this table by 74.00 MiB on the audited file. See quantization/evidence/audit-block-geometry-2026-10-26.md.
  * if the fetch never covers the tensor-info array, this exits non-zero and says so; it does not
    print a partial census, because a partial census of a width policy reads like a complete one.
  * the width table prints only after `sum(row_bytes) + data_start == file size` has been checked; a
    mismatch aborts instead of printing a plausible column. A source that will not report its total
    size still prints the table, under an explicit NOT-CHECKED banner.
"""
import os
import re
import struct
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_block_geom as GBG

TYPES = {  # ggml_type enum values, read out of ggml.h
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16", 39: "MXFP4",
}
# There is deliberately no BPW dict here any more. A hand-typed bits-per-weight table is what put
# Q5_K at 5.5625 in this file and in tools/quant_scenarios.py at the same time, and a spot-check of
# such a table finds every right row and misses the one wrong one. Widths come from the pinned
# ggml-common.h structs instead — tools/ggml_block_geom.py §why.
GIB = 2.0 ** 30
SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
# NOTE: GGUF BOOL is ONE byte (gguf-py maps it to np.bool_), not 8. Reading it as 8 desyncs the
# whole metadata walk and then produces an absurd string length later - which is exactly the
# failure this script hit on its first run against a real file.


class Cur:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise EOFError("need %d more bytes (fetched %d)" % (self.p + n - len(self.b), len(self.b)))
        v, self.p = self.b[self.p:self.p + n], self.p + n
        return v

    u32 = lambda self: struct.unpack("<I", self.take(4))[0]
    u64 = lambda self: struct.unpack("<Q", self.take(8))[0]

    def s(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def value(self, vt):
        if vt == 8:
            return self.s()
        if vt in SCALAR:
            f, n = SCALAR[vt]
            v = struct.unpack(f, self.take(n))[0]
            return bool(v) if vt == 7 else v
        raise ValueError("unknown gguf value type %d at %d" % (vt, self.p))


def read_meta(c):
    """Unused stub kept out of the parse path; metadata is walked inline in parse()."""
    raise NotImplementedError


def skip_array(c):
    et = c.u32()
    n = c.u64()
    if et in SCALAR and et != 8:
        c.take(SCALAR[et][1] * n)
        return "%s x %d" % ({0: "U8", 1: "I8", 2: "U16", 3: "I16", 4: "U32", 5: "I32", 6: "F32",
                             7: "BOOL", 10: "U64", 11: "I64", 12: "F64"}.get(et, et), n)
    samples = []
    for i in range(n):
        v = c.value(et)
        if i < 3:
            samples.append(str(v))
    return "%s x %d [%s...]" % ("STRING" if et == 8 else et, n, ", ".join(samples))


FAMILIES = [
    (r"^mtp|\.mtp\b|mtp\.", "MTP draft block"),
    (r"^token_embd\.", "token_embd (embeddings)"),
    (r"^output\.", "output (lm_head)"),
    (r"ffn_gate_inp", "router (ffn_gate_inp)"),
    (r"ffn_gate_exps|ffn_up_exps", "routed experts gate/up"),
    (r"ffn_down_exps", "routed experts down"),
    (r"ffn_shared_expert|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp", "shared expert"),
    (r"ffn_gate\b|ffn_up\b|ffn_down\b", "non-expert ffn"),
    (r"conv1d|conv\.weight", "GDN conv1d"),
    (r"ssm_out|linear_attn\.out_proj", "GDN out_proj"),
    (r"ssm_in|linear_attn\.in_proj", "GDN in_proj"),
    (r"linear_attn\.|ssm_", "GDN other (a/b/gate/beta/A_log/dt_bias)"),
    (r"attn_qkv", "linear-attn in_proj (attn_qkv)"),
    (r"attn_gate", "linear-attn z-gate (attn_gate)"),
    (r"attn_q\.", "full-attn q"),
    (r"attn_k\.", "full-attn k"),
    (r"attn_v\.", "full-attn v"),
    (r"attn_output\.", "full-attn output"),
    (r"_norm\.|norm\.weight|^norm\.", "norms"),
    (r"^rope", "rope freqs"),
    (r"^blk\.0\.", "layer-0 misc (grouped for visibility)"),
]


def family(name):
    n = re.sub(r"\.(\d+)\.", ".N.", name)
    for pat, lab in FAMILIES:
        if re.search(pat, name):
            return lab
    return "other: " + n


def fetch(src, nbytes):
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"Range": "bytes=0-%d" % (nbytes - 1),
                                                 "User-Agent": "gguf_header_types/1.0"})
        return urllib.request.urlopen(req, timeout=300).read()
    with open(src, "rb") as fh:
        return fh.read(nbytes)


def remote_total_size(src):
    """Total size of a remote GGUF, from the Content-Range of a 1-byte ranged GET (HF serves it).

    Needed because the size identity is the only witness that the derived block geometry is this
    file's real layout; a header on its own cannot show it.
    """
    try:
        req = urllib.request.Request(src, headers={"Range": "bytes=0-0",
                                                 "User-Agent": "gguf_header_types/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            cr = r.headers.get("Content-Range") or ""
            return int(cr.rsplit("/", 1)[1]) if "/" in cr else None
    except Exception:
        return None


def parse(data, raw_k=0, grep=None, keys=None, file_size=None):
    c = Cur(data)
    c.take(4)
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    meta = {}
    for _ in range(n_kv):
        k = c.s()
        vt = c.u32()
        meta[k] = skip_array(c) if vt == 9 else c.value(vt)
    rows = []
    for _ in range(n_tensors):
        name = c.s()
        nd = c.u32()
        dims = tuple(c.u64() for _ in range(nd))
        dt = c.u32()
        c.u64()
        rows.append((name, dims, dt))
    arch = str(meta.get("general.architecture", "?"))
    print("GGUF v%d  arch=%s  n_tensors=%d  n_kv=%d  header bytes parsed=%d/%d"
          % (version, arch, n_tensors, n_kv, c.p, len(data)))
    for k in sorted(meta):
        if k in ("general.name", "general.quantization_version", "general.file_type") or \
           (k.startswith(arch + ".") and re.search(
               r"attention\.layer|full_attention|hidden_size|head_dim|head_length|value_length|layers|embedding|feed_forward_length|expert", k)):
            print("  %-52s %s" % (k, meta[k]))

    if keys:
        print("\nmetadata keys matching %r:" % keys)
        for k in sorted(meta):
            if re.search(keys, k):
                v = str(meta[k])
                print("  %-52s %s" % (k, v[:150] + ("..." if len(v) > 150 else "")))
        # A GGUF is only runnable if it carries the generation contract; check for it, don't assume it.
        for must in ("tokenizer.ggml.tokens", "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
                     "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
            print("  %-52s %s" % (must, "present" if must in meta else "** ABSENT **"))
        ct = [k for k in meta if "chat_template" in k]
        print("  chat template: %s" % (ct[0] if ct else "** ABSENT ** (chatml parity must then come from elsewhere)"))

    if raw_k:
        shown = [r for r in rows if (grep is None or grep in r[0])]
        for name, dims, dt in shown[:raw_k]:
            print("  %-56s %-7s %s" % (name, TYPES.get(dt, "type_%d" % dt), list(dims)))
        print("  (%d of %d tensors matched)" % (len(shown), len(rows)))

    agg, rank3, gbytes, nbytes, unknown = {}, 0, {}, {}, set()
    # Widths come from the pinned header. A stored type this repo cannot price stops the run: counting
    # it as 0 would understate the one column this tool exists to report honestly.
    for name, dims, dt in rows:
        tname = TYPES.get(dt, "type_%d" % dt)
        try:
            GBG.bpw_of(tname)
        except KeyError:
            unknown.add(tname)
    if unknown:
        print("ABORT: no derived block geometry for %s (from %s). Those tensors cannot be priced, "
              "and pricing them as 0 would understate the table below." % (sorted(unknown),
                                                                          GBG.HEADER_REL))
        return None
    # Size identity first. If the derived block geometry does not reproduce this file's byte count,
    # every GiB figure below is plausible and wrong, so the table is not printed at all.
    align = int(meta.get("general.alignment", 32))
    data_start = -(-c.p // align) * align
    if file_size:
        tot = 0
        for name, dims, dt in rows:
            n = 1
            for d in dims:
                n *= d
            tot += GBG.row_bytes(n, TYPES.get(dt, "type_%d" % dt))
        ok = tot + data_start == file_size
        print("\nsize identity: sum(row_bytes)=%d + data_start=%d (header_end %d, alignment %d) "
              "vs file size %d -> %s" % (tot, data_start, c.p, align, file_size,
                                         "RECONCILES" if ok else "MISMATCH by %+d B"
                                         % (file_size - (tot + data_start))))
        if not ok:
            print("ABORT: the file does not reconcile against the derived geometry. No width table, "
                  "because a plausible GiB column built on a geometry that does not fit the file is "
                  "worse than no column.")
            return None
    else:
        print("\n** size identity NOT CHECKED: this source would not report a total size, so the GiB "
              "column below is arithmetic on the header alone. **")
    for name, dims, dt in rows:
        if len(dims) >= 3:
            rank3 += 1
        tname = TYPES.get(dt, "type_%d" % dt)
        n = 1
        for d in dims:
            n *= d
        fam = family(name)
        agg.setdefault(fam, {}).setdefault(tname, 0)
        agg[fam][tname] += 1
        rb = GBG.row_bytes(n, tname)
        gbytes[fam] = gbytes.get(fam, 0.0) + rb
        nbytes[tname] = nbytes.get(tname, 0) + rb
    print("\n%-44s %-26s %s" % ("family", "type(s) actually stored (count)", "weight GiB"))
    print("-" * 96)
    total = 0.0
    for fam in sorted(agg):
        total += gbytes[fam]
        print("%-44s %-26s %8.3f" % (fam, ", ".join("%s x%d" % (t, n)
                                                     for t, n in sorted(agg[fam].items())), gbytes[fam] / GIB))
    print("%-44s %-26s %8.3f" % ("TOTAL weight bytes", "", total / GIB))
    print("(%d bytes = %.2f GB decimal)" % (total, total / 1e9))
    print("\n%-8s %8s %16s %10s %10s" % ("type", "tensors", "bytes", "GiB", "bpw"))
    for tname in sorted(nbytes, key=lambda t: -nbytes[t]):
        cnt = sum(1 for _, _, dt in rows if TYPES.get(dt, "type_%d" % dt) == tname)
        print("%-8s %8d %16d %10.4f %10s" % (tname, cnt, nbytes[tname], nbytes[tname] / GIB,
                                             ("%.4f" % GBG.bpw_of(tname)).rstrip("0").rstrip(".")))
    print("bpw source: %s — struct sizes, each cross-checked against that struct's own static_assert"
          % GBG.HEADER_REL)
    layers = sorted({int(m.group(1)) for n, _, _ in rows
                     for m in [re.match(r"^(?:mtp\.)?blk\.(\d+)\.", n)] if m})
    print("layer indices present: %s%s" % ("%d..%d (n=%d)" % (layers[0], layers[-1], len(layers))
                                           if layers else "none",
                                           "  [blk.%d present => MTP/extra block]" % layers[-1]
                                           if layers and layers[-1] >= 40 else ""))
    mtp = [n for n, _, _ in rows if "mtp" in n.lower()]
    print("tensors whose name contains 'mtp': %d%s" % (len(mtp), "" if mtp else
          "  -> this file carries no draft block by that name"))
    print("\nrank-3+ tensors (expert stacks / grouped proj): %d" % rank3)
    print("all %d tensor-info records parsed." % n_tensors)
    return {"records": n_tensors, "weight_bytes": total, "size_identity_checked": bool(file_size)}


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 1
    src = a[0]
    sizes = [int(a[a.index("--bytes") + 1])] if "--bytes" in a else [4 << 20, 16 << 20, 32 << 20, 64 << 20]
    raw_k = 30 if "--raw" not in a else (int(a[a.index("--raw") + 1]) if len(a) > a.index("--raw") + 1
                                        and a[a.index("--raw") + 1].isdigit() else 30)
    grep = a[a.index("--grep") + 1] if "--grep" in a else None
    keys = a[a.index("--keys") + 1] if "--keys" in a else None
    err = None
    if src.startswith("http://") or src.startswith("https://"):
        fsize = remote_total_size(src)
    else:
        fsize = os.path.getsize(src) if os.path.exists(src) else None
    for n in sizes:
        data = fetch(src, n)
        if data[:4] != b"GGUF":
            print("not a GGUF (magic=%r)" % data[:4])
            return 1
        try:
            out = parse(data, raw_k, grep, keys, file_size=fsize)
            return 0 if out is not None else 1
        except (EOFError, struct.error) as e:
            err = e
            print("  %d MiB not enough: %s — growing the range" % (n >> 20, e), file=sys.stderr)
    print("FAILED to reach the tensor-info array: %s" % err)
    return 1


if __name__ == "__main__":
    sys.exit(main())
