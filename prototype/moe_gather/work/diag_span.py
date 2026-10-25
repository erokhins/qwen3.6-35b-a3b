"""One-shot diagnostic: is the expert span unwritten, written elsewhere, or written wrong?"""
from array import array


def words(p):
    a = array("I")
    a.frombytes(open(p, "rb").read())
    return a


def cases():
    lines = [l.rstrip("\n") for l in open("work/cases.tsv") if l.strip()]
    hdr = lines[0].lstrip("# ").split("\t")
    return [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]


out = words("work/host_out.f32")
exp = words("work/expect.f32")
c = [r for r in cases() if int(r["case_index"]) == 0][0]
s, n, eo = int(c["dst_off"]), int(c["n_floats"]), int(c["exp_off"])
print("case0 dst_off=%d n=%d exp_off=%d mode=%s kind=%s src=%s"
      % (s, n, eo, c["mode_name"], c["kind"], c["src"]))
print("out span all zero?", all(out[s + i] == 0 for i in range(n)))
print("out first 4 ", [hex(x) for x in out[s:s + 4]])
print("exp first 4 ", [hex(x) for x in exp[eo:eo + 4]])
print("out words=%d nonzero=%d | exp words=%d nonzero=%d"
      % (len(out), sum(1 for w in out if w), len(exp), sum(1 for w in exp if w)))
key = list(exp[eo:eo + 8])
print("expectation first 8 words found in out at:", [i for i in range(len(out) - 8)
                                                    if list(out[i:i + 8]) == key][:5])
key2 = list(out[s:s + 8])
print("out span first 8 words found in exp at:", [i for i in range(len(exp) - 8)
                                                  if list(exp[i:i + 8]) == key2][:5])
