#!/usr/bin/env python3
"""ulp_policy_equiv.py — does `tools/ulp_policy.py`'s ruler AGREE with the rulers already in this repo?

This is the pre-flight for a migration, not the migration. `DECISIONS.md` D-17 counts five files that
each carry their own ULP logic (`tools/audit_published_quant.py`, `tools/moe_stats.py`,
`speccheck/moe_ref.py`, `speccheck/selftest_moe.py`, `prototype/moe_router/gate.py`). Folding them onto
one ruler is a patch to a live lane, so it is staged and reviewed rather than slipped into this pass:
this harness is the evidence a reviewer needs to say the fold is safe, and
`specification/evidence/ulp-policy-migration-2026-10-26.md` holds the patches.

It is an equivalence harness, not a re-implementation: every foreign ruler is IMPORTED and CALLED, so
the comparison is between the code that ships and the new ruler, not between two readings of a docstring
(`pitfalls/40` §"a check whose reference side is derived from the thing under test").

The fixture population is DERIVED, not hand-listed (`pitfalls/40` §"an admission over a fixture list
typed from memory"): for the 16-bit formats every image is used, and for f32 the population is every
exponent crossed with a mantissa set that includes the boundaries (0, 1, 2, half, half±1, max), each
with both signs and with the ±1/±2 neighbourhood. The pairs cover the cases the two rulers are reported
to disagree on: powers of two, subnormals, the subnormal/normal edge, signed zeros, ±inf, NaN payloads,
sign flips, and pairs straddling zero.

The harness asserts its own non-vacuity: a class that MUST be observed (because it is the reason the
migration matters) is required to be non-empty. A harness that finds no disagreement between two rulers
that are documented to disagree has tested nothing.

Run under the CPU venv (it imports torch via `speccheck/moe_ref.py`):
  /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \
      specification/evidence/ulp_policy_equiv.py
Prints counts only. No timing.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "speccheck"))
import ulp_policy as UP                                       # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


audit = load_module(os.path.join(REPO, "tools/audit_published_quant.py"), "audit_pub_quant")
router = load_module(os.path.join(REPO, "prototype/moe_router/gate.py"), "router_gate")
import moe_ref                                                # noqa: E402  (speccheck oracle)
moe_stats_src = open(os.path.join(REPO, "tools/moe_stats.py")).read()


# ---------------------------------------------------------------------------------------------
# fixture populations
# ---------------------------------------------------------------------------------------------
def image_population(fmt):
    """Every image for a 16-bit format; a derived boundary sample for wider ones.

    Derived, not listed: the exponent loop is `0 .. 2^exp - 1` (so ±0, both subnormal regimes, every
    binade, both infinities and both NaN ends are in it by construction), and the mantissa set names the
    edges of the field rather than a taste for them.
    """
    m, e_bits = fmt.mant, fmt.exp
    if fmt.bits == 16:
        return list(range(0, 1 << fmt.bits))
    mants = sorted({0, 1, 2, 3, (1 << m) // 2 - 1, (1 << m) // 2, (1 << m) // 2 + 1,
                    (1 << m) - 2, (1 << m) - 1, (1 << m) // 3, (1 << m) // 5})
    out = []
    for e in range(0, 1 << e_bits):
        for mm in mants:
            out.append((e << m) | mm)
            out.append(fmt.sign_bit | ((e << m) | mm))
    extra = []
    for x in out:                       # neighbourhoods, so a power-of-two edge is hit from both sides
        for d in (-2, -1, 1, 2):
            y = x + d
            if 0 <= y <= fmt.mask:
                extra.append(y)
    seen, pop = set(), []
    for x in out + extra:
        if x not in seen:
            seen.add(x)
            pop.append(x)
    return pop


def pair_population(fmt, pop):
    """(index, a, b) pairs: next/below/two-steps, the sign flip, and the magnitude doubling."""
    pairs = []
    for x in pop:
        for y in (x + 1, x - 1, x + 2, x ^ fmt.sign_bit, x + (1 << fmt.mant)):
            if 0 <= y <= fmt.mask:
                pairs.append((len(pairs), x, y))
    return pairs


class Tally(object):
    def __init__(self, must_fire=()):
        self.n = 0
        self.agree = 0
        self.classes = {}
        self.examples = {}
        self.must_fire = set(must_fire)

    def record(self, agreed, cls, example):
        self.n += 1
        if agreed:
            self.agree += 1
            return
        self.classes[cls] = self.classes.get(cls, 0) + 1
        self.examples.setdefault(cls, example)

    def line(self, ruler, fmt, verdict, note=""):
        return ("EQUIV ruler=%s format=%s pairs=%d agree=%d disagree=%d classes[%s] verdict=%s%s"
                % (ruler, fmt, self.n, self.agree, self.n - self.agree,
                   " ".join("%s=%d" % (k, v) for k, v in sorted(self.classes.items())) or "none",
                   verdict, (" | " + note) if note else ""))


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


# ---------------------------------------------------------------------------------------------
# the comparisons
# ---------------------------------------------------------------------------------------------
def cmp_step_ruler(name, ruler, pairs, foreign, fmt, must_fire=()):
    """Compare a foreign step-counting ruler against ours on the same pairs."""
    t = Tally(must_fire)
    for _, a, b in pairs:
        ours_kind, ours = ruler.classify(a, b)
        theirs = foreign(a, b)
        if ours_kind in (UP.NAN_PAIR, UP.NAN_OTHER):
            t.record(False, "nan-arm:the shared ruler refuses to count a step here",
                     "a=0x%x b=0x%x theirs=%s" % (a, b, theirs))
            continue
        if theirs == ours:
            t.record(True, "", "")
        elif theirs is None:
            t.record(False, "foreign-ruler-returns-nothing", "a=0x%x b=0x%x" % (a, b))
        elif theirs == 0 and ours == 1:
            # a distinct defect from the straddle below, so it is tested first: this foreign key maps
            # +0 and -0 to the SAME integer, so the flip between them reads as no difference at all
            t.record(False, "identifies-plus-zero-with-minus-zero",
                     "a=0x%x b=0x%x ours=%d theirs=%d" % (a, b, ours, theirs))
        elif theirs == ours - 1:
            t.record(False, "one-step-fewer-where-the-pair-straddles-zero",
                     "a=0x%x b=0x%x ours=%d theirs=%d" % (a, b, ours, theirs))
        else:
            t.record(False, "different-count", "a=0x%x b=0x%x ours=%d theirs=%d"
                     % (a, b, ours, theirs))
    for cls in sorted(t.classes):
        print("EQUIVCLASS ruler=%s class=%s count=%d example: %s"
              % (name, cls, t.classes[cls], t.examples[cls]))
    unmet = sorted(t.must_fire - set(t.classes))
    verdict = ("EQUIVALENT" if not t.classes else
               ("DIFFERS-BY-DESIGN(nan arms)" if set(t.classes) <= {"nan-arm:the shared ruler refuses "
                                                                   "to count a step here"}
                else "DIVERGES — migration must decide each class"))
    return t, verdict, unmet


def main():
    v = V()
    f32 = UP.format_facts("f32")
    bf16 = UP.format_facts("bf16")
    r32, r16 = UP.Ruler(f32), UP.Ruler(bf16)

    pop32 = image_population(f32)
    pop16 = image_population(bf16)
    p32 = pair_population(f32, pop32)
    p16 = pair_population(bf16, pop16)
    print("POPULATION f32 images=%d pairs=%d (derived: 2^%d exp x boundary mants + neighbourhoods)"
          % (len(pop32), len(p32), f32.exp))
    print("POPULATION bf16 images=%d pairs=%d (exhaustive: every one of the 2^16 images)"
          % (len(pop16), len(p16)))
    v.check("f32 population is non-trivial", len(p32) > 10000, len(p32))
    v.check("bf16 population is exhaustive", len(pop16) == (1 << 16), len(pop16))
    # the pair list must actually contain each nasty class, or the harness is comparing nothing
    def has(pred):
        return any(pred(a, b) for _, a, b in p32)
    v.check("pairs contain signed zeros",
            has(lambda a, b: (a, b) in ((0, f32.sign_bit), (f32.sign_bit, 0))), 1)
    v.check("pairs contain subnormal-to-normal edges",
            has(lambda a, b: (a, b) == ((1 << f32.mant) - 1, 1 << f32.mant)), 1)
    v.check("pairs contain powers of two",
            has(lambda a, b: a == (f32.bias << f32.mant)
                and b == ((f32.bias + 1) << f32.mant)), 1,
            "the 1.0-to-2.0 pair, which no float-quantum ruler can count as steps")
    v.check("pairs contain infinities and NaN images",
            has(lambda a, b: r32.is_inf(a) or r32.is_inf(b)) and
            has(lambda a, b: r32.is_nan(a) or r32.is_nan(b)), 1)
    v.check("pairs straddle zero",
            has(lambda a, b: (a < f32.sign_bit) != (b < f32.sign_bit)
                and not (r32.is_zero(a) and r32.is_zero(b))), 1)

    # --- tools/audit_published_quant.py §ulp_dist (f32) --------------------------------------
    t, verdict, unmet = cmp_step_ruler(
        "tools/audit_published_quant.py:ulp_dist", r32, p32,
        lambda a, b: audit.ulp_dist(r32.value(a), r32.value(b)), "f32",
        must_fire=("nan-arm:the shared ruler refuses to count a step here",))
    print(t.line("tools/audit_published_quant.py:ulp_dist", "f32", verdict,
                 "ulp_dist takes floats and bitcasts the same way; the only documented gap is the "
                 "NaN arms"))
    v.check("ulp_dist agrees with the shared ruler on every finite pair", not unmet and
            set(t.classes) <= {"nan-arm:the shared ruler refuses to count a step here"},
            t.n - t.agree, "unmet must-fire classes: %s" % (unmet or "none"))

    # --- prototype/moe_router/gate.py §bf16_ulpdist (bf16, widened words) ---------------------
    t, verdict, unmet = cmp_step_ruler(
        "prototype/moe_router/gate.py:bf16_ulpdist", r16, p16,
        lambda a, b: router.bf16_ulpdist(a << 16, b << 16), "bf16",
        must_fire=("nan-arm:the shared ruler refuses to count a step here",))
    print(t.line("prototype/moe_router/gate.py:bf16_ulpdist", "bf16", verdict,
                 "same sign-magnitude transform as the shared ruler"))
    v.check("bf16_ulpdist agrees with the shared ruler on every finite pair",
            not unmet and set(t.classes) <= {"nan-arm:the shared ruler refuses to count a step here"},
            t.n - t.agree, "unmet must-fire classes: %s" % (unmet or "none"))

    # --- prototype/moe_router/gate.py §ulpdist_bits (f32 images) ------------------------------
    t, verdict, unmet = cmp_step_ruler(
        "prototype/moe_router/gate.py:ulpdist_bits", r32, p32,
        lambda a, b: router.ulpdist_bits(a, b), "f32-images",
        must_fire=("one-step-fewer-where-the-pair-straddles-zero",
                   "identifies-plus-zero-with-minus-zero",
                   "nan-arm:the shared ruler refuses to count a step here"))
    print(t.line("prototype/moe_router/gate.py:ulpdist_bits", "f32-images", verdict,
                 "its map is `a if a < 0x80000000 else 0x80000000 - a`, which folds +0 and -0 onto "
                 "one key: every pair straddling zero is counted ONE STEP SHORT of the true number of "
                 "representable steps, and +0/-0 read as identical. The direction matters — a band "
                 "measured with this ruler is one step LAXER across zero than the band it claims to be"))
    v.check("ulpdist_bits disagreement classes are the ones the migration note names",
            not unmet, t.n - t.agree,
            "unmet must-fire classes: %s — if this reads none, the harness is vacuous"
            % (unmet or "none"))

    # --- speccheck/moe_ref.py §ulp_of (bf16 float quantum, not a step count) -------------------
    t = Tally(must_fire=("returns-zero-at-zero-or-nonfinite",))
    for _, a, _b in p16:
        va = bf16_val(a)
        ours = upward_step(r16, a)
        theirs = moe_ref.ulp_of(va)
        if ours is None:
            t.record(True, "", "")                       # NaN image: ulp_of is not asked about these
            continue
        if theirs == 0.0 and ours != 0.0:
            t.record(False, "returns-zero-at-zero-or-nonfinite",
                     "value=%r ulp_of=%r true upward step=%r" % (va, theirs, ours))
        elif theirs == ours:
            t.record(True, "", "")
        else:
            t.record(False, "quantum-miscount (subnormal or binade edge)",
                     "image=0x%04x value=%r ulp_of=%r true upward step=%r ratio=%r"
                     % (a, va, theirs, ours, (theirs / ours) if ours else None))
    for cls in sorted(t.classes):
        print("EQUIVCLASS ruler=speccheck/moe_ref.py:ulp_of class=%s count=%d example: %s"
              % (cls, t.classes[cls], t.examples[cls]))
    unmet_ulpof = sorted(t.must_fire - set(t.classes))
    print(t.line("speccheck/moe_ref.py:ulp_of", "bf16",
                 "NOT A STEP COUNT — a length, so it cannot be the acceptance ruler",
                 "; ".join("%s=%d" % (k, v2) for k, v2 in sorted(t.classes.items()))))
    v.check("ulp_of disagreements are the documented subnormal/zero classes and are witnessed",
            not unmet_ulpof, t.n - t.agree,
            "unmet must-fire classes: %s" % (unmet_ulpof or "none"))

    # --- the prose-vs-code width claim in the two files that typed a bf16 width ------------------
    v.check("bf16 explicit mantissa bits derive to 7 (moe_ref §ulp_of prose says 8)",
            bf16.mant == 7, bf16.mant,
            "ulp_of's formula 2^(exp-7) is right; its prose is not. Derived twice: %s + %s"
            % (",".join(bf16.sources), bf16.crosscheck))
    half_ulp = 2.0 ** -(bf16.mant - 1)
    v.check("moe_stats' BF16_REL_WIDTH is the typed-8 value, not the derived one",
            "BF16_REL_WIDTH = 2.0 ** -9" in moe_stats_src, half_ulp,
            "the constant is 2^-9 under the comment '8 explicit mantissa bits'; with mant=7 the "
            "half-ULP relative width is %.3e = 2^-8. NOTE: grep shows the constant is never used, so "
            "this is dead code carrying a wrong width, not a live wrong number" % half_ulp)

    print("RESULT %s ulp_policy equivalence checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


def bf16_val(image):
    return UP._value_of("bf16", image)


def upward_step(ruler, image):
    """The value distance to the next image, from the shared ruler's own image arithmetic (None for
    the top image, where there is no next)."""
    if ruler.is_nan(image) or image >= ruler.mask:
        return None
    return abs(bf16_val(image + 1) - bf16_val(image))


if __name__ == "__main__":
    sys.exit(main())
