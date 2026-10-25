"""Is the row's x present in the slab? Zero x explains a zero expert span end to end."""
import struct

lines = [l.rstrip("\n") for l in open("work/plan.tsv") if l.strip() and not l.startswith("#")]
units, rows = [], []
for l in lines:
    f = l.split("\t")
    if f[0] == "U":
        units.append(f)
    elif f[0] == "R":
        rows.append(f)
slab = open("work/slab.bin", "rb").read()
print("slab_bytes=%d rows=%d units=%d" % (len(slab), len(rows), len(units)))
for r in rows[:3]:
    x_off = int(r[7])
    vals = struct.unpack_from("<8f", slab, x_off)
    nz = sum(1 for i in range(2048) if struct.unpack_from("<f", slab, x_off + 4 * i)[0] != 0.0)
    print("row block=%s token=%s fixture=%s x_off=%d first8=%s nonzero_of_2048=%d"
          % (r[1], r[2], r[3], x_off, ["%.4g" % v for v in vals], nz))
for u in units[:3]:
    off = int(u[6])
    nz = sum(1 for i in range(64) if slab[off + i] != 0)
    print("unit role=%s expert=%s src_off=%d nonzero_bytes_of_64=%d" % (u[4], u[5], off, nz))
