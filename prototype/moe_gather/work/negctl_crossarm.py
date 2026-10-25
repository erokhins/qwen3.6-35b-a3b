"""Negative controls for the D-16 cross-arm check: each of its four arms must be able to fail.

A check that cannot fail is a comment wearing a counter (`pitfalls/40` §"a check whose predicate is
True"). This mutates the words **in memory only** — the files on disk are never touched, so there is
nothing to restore — and requires each cross-arm check to go red for its own reason:

  A  one `gate` word of the device arm changed          -> the declared-exact arm must fire
  B  the device arm made to MATCH the oracle in `s`    -> the attribution arm must fire (one arm
                                                          agreeing with the oracle while the other
                                                          does not is a device defect, not a route)
  C  every `s` element of one case shifted by 1 ULP    -> the isolation arm must fire
  D  one word outside the 18 declared cases changed    -> the outside-the-declared-set arm must fire
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
sys.path.insert(0, LANE)
os.environ["MG_I"] = "512"
os.environ["MG_H"] = "2048"
import gate as G                                              # noqa: E402

work = os.path.join(LANE, "work")
cases = G.read_cases(os.path.join(work, "cases.tsv"))
NF = sum(c["n_floats"] for c in cases)
H0 = G.read_words(os.path.join(work, "host_out.f32"), NF)
D0 = G.read_words(os.path.join(work, "out.f32"), NF)
E0 = G.read_words(os.path.join(work, "expect.f32"), NF)
I, H = 512, 2048
FIELDS = [("gate", 0, I), ("up", I, I), ("s", 2 * I, I), ("raw", 3 * I, H)]
gu = [c for c in cases if c["kind"] == 0 and c["field_exact"] == "gate_up"]
nongu = [c for c in cases if c["field_exact"] != "gate_up"]


def run(mut, label, expect_bad):
    host, dev, exp = list(H0), list(D0), list(E0)
    mut(host, dev, exp)
    v = G.V()
    G.cross_arm(v, cases, host, dev, exp)
    got = set(v.bad)
    print("NEGCTL %s checks=%d failed=%d bad=%s" % (label, v.n, len(v.bad), sorted(v.bad)))
    ok = all(any(w in b for b in got) for w in expect_bad)
    print("NEGCTL %s verdict=%s (expected a failure naming: %s)"
          % (label, "OK" if ok else "WRONG", expect_bad))
    return ok


results = []


def mut_a(host, dev, exp):
    dev[gu[0]["dst_off"] + 5] ^= 0x00000001                    # one gate word, 1 ULP from the arm


def mut_b(host, dev, exp):
    c = gu[0]
    for i in range(FIELDS[2][2]):                              # make device `s` == the oracle's
        dev[c["dst_off"] + FIELDS[2][1] + i] = exp[c["exp_off"] + FIELDS[2][1] + i]


def mut_c(host, dev, exp):
    c = gu[0]
    for i in range(FIELDS[2][2]):                              # every s element of one case, +1 ULP
        dev[c["dst_off"] + FIELDS[2][1] + i] = host[c["dst_off"] + FIELDS[2][1] + i] + 1


def mut_d(host, dev, exp):
    dev[nongu[0]["dst_off"] + 3] ^= 0x00000002                 # outside the declared cases


results.append(run(mut_a, "A-one-gate-word-moved", ["declared-exact field"]))
results.append(run(mut_b, "B-device-matches-oracle-in-s", ["attributed"]))
results.append(run(mut_c, "C-every-s-element-shifted", ["isolated"]))
results.append(run(mut_d, "D-outside-the-declared-cases", ["outside the declared counted cases"]))
print("NEGCTLSUM arms=4 witnessed_firing=%d verdict=%s"
      % (sum(1 for r in results if r), "PASS" if all(results) else "FAIL"))
sys.exit(0 if all(results) else 1)
