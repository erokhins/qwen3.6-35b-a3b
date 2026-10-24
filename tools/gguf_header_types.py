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
  * if the fetch never covers the tensor-info array, this exits non-zero and says so; it does not
    print a partial census, because a partial census of a width policy reads like a complete one.
"""
import re
import struct
import sys
import urllib.request

TYPES = {  # ggml_type enum values, read out of ggml.h
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16", 39: "MXFP4",
}
# bits per weight incl. scale/zero overhead, so bytes = n_elems * bpw / 8. Values are the standard
# ggml block sizes (e.g. Q4_K = 144 B per 256 weights = 4.5 bpw), and anything unmapped prints
# `type_<id>` and is counted as 0 with a warning line rather than being guessed.
BPW = {"F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q4_0": 4.5, "Q4_1": 4.5, "Q5_0": 5.5, "Q5_1": 5.5,
       "Q8_0": 8.5, "Q8_1": 9.0, "Q2_K": 2.625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5625,
       "Q6_K": 6.5625, "IQ2_XS": 2.3125, "IQ3_XXS": 3.0625, "IQ1_S": 1.5625, "IQ4_NL": 4.5,
       "IQ3_S": 3.4375, "IQ2_S": 2.5, "IQ4_XS": 4.25, "MXFP4": 4.25}
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


def parse(data, raw_k=0, grep=None):
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

    if raw_k:
        shown = [r for r in rows if (grep is None or grep in r[0])]
        for name, dims, dt in shown[:raw_k]:
            print("  %-56s %-7s %s" % (name, TYPES.get(dt, "type_%d" % dt), list(dims)))
        print("  (%d of %d tensors matched)" % (len(shown), len(rows)))

    agg, rank3, gbytes, unknown = {}, 0, {}, set()
    for name, dims, dt in rows:
        if len(dims) >= 3:
            rank3 += 1
        tname = TYPES.get(dt, "type_%d" % dt)
        if tname not in BPW:
            unknown.add(tname)
        n = 1
        for d in dims:
            n *= d
        fam = family(name)
        agg.setdefault(fam, {}).setdefault(tname, 0)
        agg[fam][tname] += 1
        gbytes[fam] = gbytes.get(fam, 0.0) + n * BPW.get(tname, 0.0) / 8.0
    if unknown:
        print("WARNING: no bpw known for %s; those bytes are counted as 0." % sorted(unknown))
    print("\n%-44s %-26s %s" % ("family", "type(s) actually stored (count)", "weight GiB"))
    print("-" * 96)
    total = 0.0
    for fam in sorted(agg):
        total += gbytes[fam]
        print("%-44s %-26s %8.3f" % (fam, ", ".join("%s x%d" % (t, n)
                                                     for t, n in sorted(agg[fam].items())), gbytes[fam] / GIB))
    print("%-44s %-26s %8.3f" % ("TOTAL weight bytes", "", total / GIB))
    print("(%d bytes = %.2f GB decimal)" % (total, total / 1e9))
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
    err = None
    for n in sizes:
        data = fetch(src, n)
        if data[:4] != b"GGUF":
            print("not a GGUF (magic=%r)" % data[:4])
            return 1
        try:
            parse(data, raw_k, grep)
            return 0
        except (EOFError, struct.error) as e:
            err = e
            print("  %d MiB not enough: %s — growing the range" % (n >> 20, e), file=sys.stderr)
    print("FAILED to reach the tensor-info array: %s" % err)
    return 1


if __name__ == "__main__":
    sys.exit(main())
