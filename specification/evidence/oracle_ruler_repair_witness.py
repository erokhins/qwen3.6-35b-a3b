#!/usr/bin/env python3
"""oracle_ruler_repair_witness.py — the router's third sign-magnitude ruler, repaired and witnessed.

`prototype/moe_router/oracle_arm.py::ulpdist` was the third sign-magnitude ULP ruler in this repo and
the only one that DERIVES bands: every figure in `work/oracle_floor.tsv` — and therefore every
allowance `gate.py` judges the kernel against — is a `.max()` of its output. Its old body was

    ai = np.where(ai < 0, 0x80000000 - ai, ai)          # on a SIGNED int32 view

For a negative image the shared ruler's key is `~b`; the old body's key was `2^31 + 1 + key(b)`. On a
same-sign pair the two offsets cancel, so the old body was EXACT — but on a pair straddling zero they
do not, and it returns the COMPLEMENT of the true distance, `2^32 + 1 - d`, instead of `d`. That is an
order reversal about zero, not an off-by-one: `+0`/`-0` (d = 1, the one distance fixed by the format)
read 4,294,967,296 where the ruler of record (`tools/ulp_policy.py` §Ruler) says 1, and `+1.0`/`-1.0`
(d = 2,130,706,433) read 2,164,260,864. Any small cross-zero rounding step read as ~4.29e9, so a
straddling pair in a probe population would have inflated a band rather than quietly tightening it —
and a band that reads 65,536 is therefore not a band a straddle went into, which §4 below measures
instead of asserting (`pitfalls/30` §A). It has been repaired against `tools/ulp_policy.py`, which is
not edited by this or anything else here.

This file is the evidence, in the order a reviewer should read it:

  1. AGREEMENT. On a derived population (every bf16 image widened the way the lane stores them, with
     neighbours and sign flips, plus a boundary-sampled f32 population: every exponent crossed with a
     mantissa-edge set), the repaired `ulpdist` equals `Ruler.classify`'s step count on every pair the
     ruler counts — and REFUSES exactly the pairs the ruler answers `None` for, which is what a probe
     that takes `.max()` needs: a NaN must not become a big number.
  2. THE OLD BODY, WITNESSED. The pre-repair formula is kept here verbatim as `historic_ulpdist` and
     run over the same population: its error law, `old = 2^32 + 1 - d` on straddling pairs, is checked
     pair by pair (not sampled), and it is checked to be the IDENTITY on same-sign pairs. Those class
     counters must be non-zero — an agreement check between two rulers that never disagreed would be
     checking nothing (`pitfalls/40` §"a negative control with an empty must-catch set is not a
     control").
  3. THE BANDS, MEASURED BOTH WAYS. `probe_inputs` + `floor_probe` are run TWICE over the same
     inputs, once with the repaired ruler and once with the old body, and every band is printed as
     `BANDDELTA name=… old=… new=… delta=… pairs=… straddling=… signed_zero=…`. Nothing is
     hand-edited: `work/oracle_floor.tsv` is re-derived by the probe, and this file's job is to show
     which bands the repair could have moved and which it did.
  4. THE DORMANCY CLAIM, PROVEN. `logit_dot_route_ulp=65536` was reported as "a same-sign measurement,
     which is why the bug has been dormant". Here it is recomputed independently of `pair_census`: the
     straddling-pair count over that band's own population, the widest pair's two bit images, and —
     the part that makes it a proof rather than a restatement — the same pair with one member negated,
     showing what the old body *would* have printed and by how much it was short.

CPU only, no device, no clock: **no timing taken**. Run under the CPU venv (it imports torch):

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \
        specification/evidence/oracle_ruler_repair_witness.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "speccheck"),
           os.path.join(REPO, "prototype", "moe_router")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ulp_policy                                          # noqa: E402
import moe_ref                                             # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


oa = load_module(os.path.join(REPO, "prototype/moe_router/oracle_arm.py"), "router_oracle_arm")
RULER = ulp_policy.Ruler("f32")
SIGN = RULER.sign
FLOOR_TSV = os.path.join(REPO, "prototype/moe_router/work/oracle_floor.tsv")
BEFORE_TSV = os.path.join(HERE, "oracle-floor-before-ruler-repair-2026-10-26.tsv")


def historic_ulpdist(a, b):
    """The body `oracle_arm.ulpdist` carried before 2026-10-26, verbatim — the thing to refuse.

    Kept so the repair is a comparison and not an assertion: a bug you cannot re-run is a bug you
    cannot show fixed, and a fix nobody can reproduce is a story.
    """
    ai = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    bi = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 0x80000000 - ai, ai)
    bi = np.where(bi < 0, 0x80000000 - bi, bi)
    return np.abs(ai - bi)


def f32_of_image(image):
    """One f32 BIT IMAGE as a one-element float32 array — the form `ulpdist` is called with."""
    return np.array([int(image)], dtype=np.uint32).view(np.float32)


class V(object):
    def __init__(self):
        self.n = 0
        self.f = 0

    def check(self, name, ok, n, detail=""):
        self.n += 1
        if not ok:
            self.f += 1
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if ok else 0, n, detail))
        return ok


def pair_population():
    """(a_images, b_images) as f32 BIT IMAGES — derived, not hand-listed.

    Half one: every one of the 2^16 bf16 images widened the way the lane stores a bf16 value
    (`image << 16`), each paired with its two neighbours, its sign flip and the next binade up,
    because R1's band is a bf16 field measured in f32 images. Half two: the f32 boundary set — every
    exponent crossed with a mantissa-edge set (0, 1, 2, 3, half-1, half, half+1, max-1, max, and two
    interior points), both signs, plus ±1/±2 neighbourhoods and the sign flip — so subnormals, both
    infinities, both NaN ends and every binade edge are in it by construction rather than by taste.
    """
    m = RULER.mant
    a, b = [], []

    def add(x, y):
        if 0 <= x <= RULER.mask and 0 <= y <= RULER.mask:
            a.append(int(x))
            b.append(int(y))

    for h in range(1 << 16):                                   # every bf16 image
        x = h << 16
        for y in (x + 1, x - 1, x ^ (1 << 15), x + (1 << 7)):   # neighbours, flip, next binade
            add(x, y)
    mants = sorted({0, 1, 2, 3, (1 << m) // 2 - 1, (1 << m) // 2, (1 << m) // 2 + 1,
                    (1 << m) - 2, (1 << m) - 1, (1 << m) // 3, (1 << m) // 5})
    base = []
    for e in range(1 << RULER.format.exp):
        for mm in mants:
            base.append((e << m) | mm)
            base.append(SIGN | ((e << m) | mm))
    for x in base:
        add(x, x ^ SIGN)
        for d in (-2, -1, 1, 2):
            add(x, x + d)
    return a, b


def read_floor(path):
    rows = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        rows[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4])) if len(p) == 5 else (
            int(p[1]), int(p[2]), None, None)
    return rows


def main():
    v = V()
    print("RULER of_record=tools/ulp_policy.py format=f32 bits=%d mant=%d exp=%d bias=%d sign=0x%08x"
          % (RULER.format.bits, RULER.format.mant, RULER.format.exp, RULER.format.bias, SIGN))

    # ---- 1. the population, and agreement with the ruler of record ------------------------------
    ap, bp = pair_population()
    a_img = np.array(ap, dtype=np.uint32)
    b_img = np.array(bp, dtype=np.uint32)
    nan_mask = oa.is_nan_image(a_img.astype(np.int64)) | oa.is_nan_image(b_img.astype(np.int64))
    clean = ~nan_mask
    a_neg, b_neg = (a_img & SIGN) != 0, (b_img & SIGN) != 0
    a_zero = (a_img == 0) | (a_img == SIGN)
    b_zero = (b_img == 0) | (b_img == SIGN)
    abs_mask = RULER.mask ^ SIGN                       # 0x7FFFFFFF: the image without its sign bit
    inf_image = RULER.exp_mask << RULER.mant           # 0x7F800000, from the ruler's own field widths
    straddle = int(((a_neg != b_neg) & clean).sum())
    signed_zero = int(((a_zero & b_zero & (a_img != b_img)) & clean).sum())
    inf_pairs = int(((((a_img & abs_mask) == inf_image)
                      | ((b_img & abs_mask) == inf_image)) & clean).sum())
    print("POPULATION pairs=%d clean=%d nan=%d straddling=%d signed_zero=%d involving_inf=%d "
          "derived_from=every_bf16_image(widened) x {+-1,flip,binade} + f32 exponents x mants x {flip,+-1,+-2}"
          % (len(ap), int(clean.sum()), int(nan_mask.sum()), straddle, signed_zero, inf_pairs))
    v.check("the pair population contains every class the repair is about",
            len(ap) > 100000 and straddle > 0 and signed_zero > 0 and inf_pairs > 0
            and int(nan_mask.sum()) > 0, len(ap),
            "a population with no straddling pair could not show the old body's complement law at "
            "all, and one with no NaN could not show the refusal")
    ca, cb = a_img[clean], b_img[clean]
    av = np.array([RULER.value(int(x)) for x in ca], dtype=np.float32)
    bv = np.array([RULER.value(int(x)) for x in cb], dtype=np.float32)
    d_new = oa.ulpdist(av, bv)
    disagree, first = 0, None
    for i in range(len(av)):
        kind, steps = RULER.classify(int(ca[i]), int(cb[i]))
        if steps is None or kind in (ulp_policy.NAN_PAIR, ulp_policy.NAN_OTHER) \
                or int(d_new[i]) != int(steps):
            disagree += 1
            first = first or (int(ca[i]), int(cb[i]), kind, steps, int(d_new[i]))
    v.check("the repaired ulpdist equals the shared ruler on every pair it counts",
            disagree == 0, int(clean.sum()),
            "%d finite-and-infinite pairs compared against Ruler.classify; first disagreement: %s"
            % (int(clean.sum()), first))
    refused, counted = 0, []
    for j in np.flatnonzero(nan_mask)[:400]:
        try:
            oa.ulpdist(f32_of_image(a_img[j]), f32_of_image(b_img[j]))
            counted.append((int(a_img[j]), int(b_img[j])))
        except AssertionError:
            refused += 1
    v.check("the repaired ulpdist REFUSES exactly the pairs the ruler answers None for",
            refused > 0 and not counted, refused,
            "%d NaN-involving pairs asked one at a time, all raised (a sentinel here would widen a "
            "band with a number that means nothing); pairs it counted anyway: %s"
            % (refused, counted[:3] or "none"))

    # ---- 2. the old body, witnessed against its error law ---------------------------------------
    d_old = historic_ulpdist(av, bv)
    COMPLEMENT = (1 << 32) + 1                    # old = COMPLEMENT - d across zero; = d otherwise
    straddle_ok, straddle_bad = 0, []
    signed_zero_ok, signed_zero_bad = 0, []
    same_sign, same_sign_bad, examples = 0, [], []
    for i in range(len(av)):
        aa, bb = int(ca[i]), int(cb[i])
        want = int(RULER.classify(aa, bb)[1])
        got = int(d_old[i])
        zs = (aa in (0, SIGN)) and (bb in (0, SIGN)) and aa != bb
        crosses = ((aa & SIGN) != 0) != ((bb & SIGN) != 0)
        if zs:
            if got == COMPLEMENT - want:
                signed_zero_ok += 1
            else:
                signed_zero_bad.append("a=0x%08x b=0x%08x ruler=%d historic=%d" % (aa, bb, want, got))
        elif crosses:
            if got == COMPLEMENT - want:
                straddle_ok += 1
            else:
                straddle_bad.append("a=0x%08x b=0x%08x ruler=%d historic=%d law_says=%d"
                                    % (aa, bb, want, got, COMPLEMENT - want))
        else:
            same_sign += 1
            if got != want:
                same_sign_bad.append("a=0x%08x b=0x%08x ruler=%d historic=%d" % (aa, bb, want, got))
    for line in (signed_zero_bad + straddle_bad + same_sign_bad)[:4]:
        print("LAWVIOLATION %s" % line)
    print("LAW old = 2^32+1-d on straddling pairs: holds on %d of %d (of which the signed-zero pair "
          "class accounts for %d of %d); old == d on %d same-sign pairs, violations %d"
          % (straddle_ok, straddle_ok + len(straddle_bad), signed_zero_ok,
             signed_zero_ok + len(signed_zero_bad), same_sign, len(same_sign_bad)))
    for label, x, y in (("+0/-0", 0x00000000, 0x80000000),
                        ("+1.0/-1.0", 0x3F800000, 0xBF800000),
                        ("+smallest-subnormal/-0", 0x00000001, 0x80000000)):
        want = RULER.classify(x, y)[1]
        got = int(historic_ulpdist(f32_of_image(x), f32_of_image(y))[0])
        now = int(oa.ulpdist(f32_of_image(x), f32_of_image(y))[0])
        print("HEADLINE pair=%-24s ruler=%d historic=%d repaired=%d repaired_matches_ruler=%d"
              % (label, want, got, now, int(now == want)))
        v.check("the repaired ruler states %s the way the ruler of record does" % label,
                now == want and got != want, now,
                "historic body said %d where tools/ulp_policy.py says %d — this is the pair class the "
                "bands were blind to" % (got, want))
    v.check("the old body's straddling-pair error law fires and has no exception",
            straddle_ok > 0 and signed_zero_ok > 0 and not straddle_bad and not signed_zero_bad,
            straddle_ok + signed_zero_ok,
            "every pair whose members have opposite signs was returned as 2^32+1-d rather than d, "
            "signed zeros included (they are the pair whose true distance the format fixes at 1, so "
            "this class is checkable without trusting either ruler)")
    v.check("the old body is the IDENTITY on same-sign pairs, which is why no band moved",
            same_sign > 0 and not same_sign_bad, same_sign,
            "%d same-sign pairs, %d of them measured wrong by the old body — if that second number "
            "were not 0 the bands would have been wrong for a second, unrelated reason" 
            % (same_sign, len(same_sign_bad)))

    # ---- 3. the probe, run twice over the same inputs -------------------------------------------
    man, D = oa.load_manifest()
    thr = int(man["interpreter"]["threads"])
    oa.torch.set_num_threads(thr)
    moe_ref.set_threads(thr)
    g = moe_ref.load_geometry()
    rows = oa.probe_inputs(D, g)
    new_bands, pop = oa.floor_probe(rows, g.top_k)
    saved = oa.ulpdist
    try:
        oa.ulpdist = historic_ulpdist
        old_bands, _p = oa.floor_probe(rows, g.top_k)
    finally:
        oa.ulpdist = saved
    moved = []
    for name in sorted(new_bands):
        delta = new_bands[name] - old_bands[name]
        if delta:
            moved.append(name)
        print("BANDDELTA name=%s old=%d new=%d delta=%+d pairs=%d straddling=%d signed_zero=%d "
              "verdict=%s"
              % (name, old_bands[name], new_bands[name], delta, pop[name][0], pop[name][1],
                 pop[name][2], "MOVED — re-derive, do not widen" if delta
                 else "unchanged (no straddling pair in its population)"))
    v.check("re-running the probe with the old body moves no band on this artifact",
            not moved, len(new_bands),
            "bands that moved: %s — a non-empty list means the figure contained a pair whose members "
            "have opposite signs, and the fix is to re-derive and re-examine the verdicts that quote "
            "it, never to widen a band by hand" % (moved or "none"))
    on_disk = read_floor(FLOOR_TSV)
    v.check("the table on disk is the table this probe just derived",
            all(on_disk.get(k) == (pop[k][0], new_bands[k], pop[k][1], pop[k][2])
                for k in new_bands) and len(on_disk) == len(new_bands), len(on_disk),
            "work/oracle_floor.tsv must be the probe's own output (name, comparisons, max_ulp, "
            "straddling, signed-zero) and not a hand-edited band file")
    before = read_floor(BEFORE_TSV)
    changed = [k for k in sorted(new_bands) if before.get(k) is None
               or before[k][1] != new_bands[k]]
    print("DELTA archived_before=%s bands=%d changed_vs_before=%s"
          % (os.path.basename(BEFORE_TSV), len(before), changed or "none"))
    v.check("every band delta against the archived pre-repair table is printed and zero",
            len(before) == len(new_bands) and not changed, len(new_bands),
            "bands whose max_ulp differs from the archived pre-repair table: %s — that table was "
            "written by the old body, so a non-empty list is a real band move and the verdicts "
            "quoting those bands must be re-examined, not the band widened" % (changed or "none"))

    # ---- 4. the dormancy claim, recomputed and shown as a would-have-printed ---------------------
    worst = (0, None, None, "")
    straddle_total = 0
    for blk, _lg, _twf, _pr, _pb, x, wg, ref_bits in rows:
        bits = oa.dot_orders(x, wg, ref_bits)
        ref = bits["f32_seq"]
        for nm, v2 in bits.items():
            if nm == "f32_seq":
                continue
            f2, fref = oa.u16_to_f32(v2), oa.u16_to_f32(ref)
            a2 = f2.view(np.uint32).astype(np.int64)
            b2 = fref.view(np.uint32).astype(np.int64)
            straddle_total += int((((a2 & SIGN) != 0) != ((b2 & SIGN) != 0)).sum())
            dist = oa.ulpdist(f2, fref).ravel()
            i = int(np.argmax(dist))
            if int(dist[i]) > worst[0]:
                a2f, b2f = a2.ravel(), b2.ravel()
                worst = (int(dist[i]), int(a2f[i]), int(b2f[i]), "%s/%s" % (blk, nm))
    a_w, b_w = worst[1], worst[2]
    twin = a_w ^ SIGN
    twin_new = RULER.classify(twin, b_w)[1]
    twin_old = int(historic_ulpdist(f32_of_image(twin), f32_of_image(b_w))[0])
    twin_now = int(oa.ulpdist(f32_of_image(twin), f32_of_image(b_w))[0])
    print("DORMANT band=logit_dot_route_ulp max_ulp=%d pairs=%d straddling_pairs_in_population=%d "
          "(recomputed here, independently of oracle_arm.pair_census)"
          % (new_bands["logit_dot_route_ulp"], pop["logit_dot_route_ulp"][0], straddle_total))
    print("DORMANT widest_pair=%s a=0x%08x(%r) b=0x%08x(%r) steps=%d same_sign=%d"
          % (worst[3], a_w, RULER.value(a_w), b_w, RULER.value(b_w), worst[0],
             int(((a_w & SIGN) != 0) == ((b_w & SIGN) != 0))))
    print("WOULDPRINT negate one member: a=0x%08x(%r) b=0x%08x straddles_zero=%d ruler=%d "
          "repaired=%d historic=%d law_says=%d law_holds=%d"
          % (twin, RULER.value(twin), b_w, int(((twin & SIGN) != 0) != ((b_w & SIGN) != 0)),
             twin_new, twin_now, twin_old, (1 << 32) + 1 - twin_new,
             int(twin_old == (1 << 32) + 1 - twin_new)))
    print("WOULDPRINT had a straddling pair of the band's own size existed, the old body would have "
          "printed 2^32+1-%d = %d where the ruler says %d — five orders of magnitude above the band "
          "that was actually reported (%d), which is the sense in which the bug was dormant"
          % (worst[0], (1 << 32) + 1 - worst[0], worst[0], new_bands["logit_dot_route_ulp"]))
    v.check("logit_dot_route_ulp is a same-sign measurement, shown rather than stated",
            straddle_total == 0 and worst[0] == new_bands["logit_dot_route_ulp"]
            and ((a_w & SIGN) != 0) == ((b_w & SIGN) != 0), straddle_total,
            "0 straddling pairs in %d comparisons of that band's own population, recomputed here from "
            "the bit images rather than read from oracle_arm.pair_census; the band's own value is "
            "second evidence, because a straddle of size d would have been reported as 2^32+1-d"
            % pop["logit_dot_route_ulp"][0])
    v.check("the straddling twin prints what the old body WOULD have said, by the complement law",
            twin_new == twin_now and twin_old == (1 << 32) + 1 - twin_new, twin_old,
            "the band's own widest pair with one member negated: the ruler and the repaired body say "
            "%d, the historic body said %d = 2^32+1-%d — an order reversal about zero, not a rounding "
            "nudge" % (twin_new, twin_old, twin_new))
    print("RESULT %s oracle ruler repair witness checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
