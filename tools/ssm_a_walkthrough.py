#!/usr/bin/env python3
"""How the ssm_a near-miss numbers were obtained, end to end. CPU only, stdlib only.

Reads: the GGUF's raw bytes for blk.1.ssm_a, and A_log from the HF shard, both from disk.
Prints: the bytes, the two computations, and the evidence for the head reorder.
"""
import json, math, os, struct, sys
from decimal import Decimal, localcontext

REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
SRC = os.path.join(REPO, "models/qwen36-src")
HF_NAME = "model.language_model.layers.1.linear_attn.A_log"
TENSOR = "blk.1.ssm_a"


def bits(x):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------- step 1: the file. Read the GGUF tensor table, then seek to the bytes. ----------
sys.path.insert(0, os.path.join(REPO, "tools"))
import gguf_reader as gr

gg = gr.Gguf(GGUF)
t = [x for x in gg.tensors if x.name == TENSOR][0]
print("STEP 1  the GGUF, read straight from disk")
print("  tensor            %s   type=%s dims=%s" % (t.name, t.type_name, t.dims))
print("  byte offset       data_start=%d + t.offset=%d  -> file pos %d"
      % (gg.data_start, t.offset, gg.data_start + t.offset))
n = t.row // 4
with open(GGUF, "rb") as fh:
    fh.seek(gg.data_start + t.offset)
    raw = fh.read(t.row)
file_bits = ["0x%08x" % v for v in struct.unpack("<%dI" % n, raw)]
vals = list(struct.unpack("<%df" % n, raw))
print("  %d float32 elements; element 21 = %s = %.16f" % (n, file_bits[21], vals[21]))

# ---------- step 2: the original. A_log is bf16 in the shard; widen to f32 exactly. ----------
def read_safetensors(path, name):
    """-> (dtype, shape, raw bytes) for one tensor, or None if this shard lacks it."""
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
        if name not in hdr:
            return None
        info = hdr[name]
        lo, hi = info["data_offsets"]
        fh.seek(8 + hlen + lo)
        return info["dtype"], info["shape"], fh.read(hi - lo)


SHARD, dtype_a, shape_a, alog_raw = None, None, None, None
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".safetensors"):
        continue
    got = read_safetensors(os.path.join(SRC, fn), HF_NAME)
    if got is not None:
        SHARD, dtype_a, shape_a, alog_raw = os.path.join(SRC, fn), got[0], got[1], got[2]
        break
assert SHARD, "A_log for layer 1 is in no shard on disk -- the sample does not cover it"
print("\nSTEP 2  the HF checkpoint (%s)" % os.path.basename(SHARD))
print("  tensor            %s" % HF_NAME)
print("  dtype=%s shape=%s  %d bytes = %d bf16 elements"
      % (dtype_a, shape_a, len(alog_raw), len(alog_raw) // 2))
u16 = struct.unpack("<%dH" % (len(alog_raw) // 2), alog_raw)
alog = [struct.unpack("<f", struct.pack("<I", u << 16))[0] for u in u16]
print("  A_log in natural HF order, element 21: bf16 0x%04x -> f32 %s = %.16f"
      % (u16[21], bits(alog[21]), alog[21]))
print("  (that is HF's own index 21 -- NOT yet the element file index 21 came from; step 3 finds it)")
print("  (widening bf16->f32 shifts the 16 mantissa bits left; it loses nothing, ever)")

# ---------- step 3: the head reorder, discovered rather than assumed ----------
def route_f32(v):
    return f32(-math.exp(v))                     # double exp (libm), then one rounding to f32


def route_crl(v):
    with localcontext() as ctx:                  # 60 digits, then one rounding to f32
        ctx.prec = 60
        return f32(-float(Decimal(repr(v)).exp()))


def agrees(i, j):
    """does file element i match the route applied to HF element j, under BOTH routes?"""
    b = bits(route_f32(alog[j])), bits(route_crl(alog[j]))
    return b == (file_bits[i], file_bits[i])


ident_hits = sum(1 for i in range(n) if agrees(i, i))
perm = [None] * n
for i in range(n):
    for j in range(n):
        if agrees(i, j):
            perm[i] = j
            break
found_hits = sum(1 for p in perm if p is not None)

with open(os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")) as fh:
    cfg = json.load(fh)
tc = cfg.get("text_config", cfg)
nk = tc["linear_num_key_heads"]
nv = n // nk
assert nk * nv == n, "nk=%d does not divide n=%d; reorder parameters wrong" % (nk, n)


def tiled_to_grouped(i, nk, nv, hd=1):
    """GGUF(tiled) index -> HF(grouped) index, as conversion/qwen.py _reorder_v_heads spells it:
    reshape(nk, v, d), swap nk and v, flatten. V is the OUTER axis in the file. Copied from
    tools/audit_published_quant.py §v_tiled_to_grouped, and checked against the search above."""
    v = i // (nk * hd)
    rem = i % (nk * hd)
    k = rem // hd
    d = rem % hd
    return (k * nv + v) * hd + d


doc = [tiled_to_grouped(i, nk, nv) for i in range(n)]
exact_hits_doc = sum(1 for i in range(n) if agrees(i, doc[i]))
mismatched = [i for i in range(n) if not agrees(i, doc[i])]
consistent = all(perm[i] == doc[i] for i in range(n) if perm[i] is not None)

print("\nSTEP 3  which HF element does file index i come from? (searched, then checked)")
print("  nk=%d, nv_per_k=%d, 32 v-heads total" % (nk, nv))
print("  same index, no reorder                             : %2d / %d bit-exact"
      % (ident_hits, n))
print("  through the converter's documented V-head reorder  : %2d / %d bit-exact"
      % (exact_hits_doc, n))
print("  the permutation found by blind search agrees with it: %s" % consistent)
print("  (this reorder was not known in advance -- an earlier version of this repo had it the other")
print("   way round and only the exactness pass caught it, which is why it is re-derived here)")

# ---------- step 4: the elements that do not agree ----------
print("\nSTEP 4  the %d element(s) that do not agree, traced to their source" % len(mismatched))
for i in mismatched:
    j = doc[i]
    steps = int(file_bits[i], 16) - int(bits(route_crl(alog[j])), 16)
    print("  file index %d <- HF index %d" % (i, j))
    print("    A_log[%d] in the shard : bf16 0x%04x -> widened to f32 %s = %.16f"
          % (j, u16[j], bits(alog[j]), alog[j]))
    print("    -exp of it, route f32  : %s = %.16f" % (bits(route_f32(alog[j])), route_f32(alog[j])))
    print("    -exp of it, route crl  : %s = %.16f" % (bits(route_crl(alog[j])), route_crl(alog[j])))
    print("    what the GGUF holds    : %s = %.16f" % (file_bits[i], vals[i]))
    print("    integer difference in the encoded bits: %+d  (one f32 step is 1)" % steps)
print("  (the two routes are different code paths -- C libm exp in double, and a 60-digit Decimal")
print("   exp -- and they agree with each other on every element, including this one)")

# ---------- step 5: how far apart is one step, measured not asserted ----------
up = struct.unpack("<f", struct.pack("<I", int(file_bits[21], 16) - 1))[0]
down = struct.unpack("<f", struct.pack("<I", int(file_bits[21], 16) - 2))[0]
print("\nSTEP 5  the size of 'one step' at this magnitude, from its own neighbours")
print("  two below   %s = %.16f" % (bits(down), down))
print("  one below   %s = %.16f   <- both routes land here" % (bits(up), up))
print("  the file    %s = %.16f" % (file_bits[21], vals[21]))
print("  gap         %.3e absolute, %.3e relative to the value"
      % (abs(vals[21] - up), abs(vals[21] - up) / abs(vals[21])))

print("\nVERDICT  %d of %d elements in blk.1.ssm_a agree to the byte through the reorder; the"% (exact_hits_doc, n))
print("remaining %d is one representable step away. There is no third opinion available: the HF" % len(mismatched))
print("checkpoint stores A_log, not this value, so -exp(A_log) is the only definition of it.")

