#!/usr/bin/env python3
"""ulp_policy_equiv.py — does `tools/ulp_policy.py`'s ruler AGREE with the rulers already in this repo?

Two jobs, because the fold this harness was written to pre-flight has landed.

(1) It is still the STANDING equivalence evidence. `DECISIONS.md` D-17 counts five files that each
    carried their own ULP logic (`tools/audit_published_quant.py`, `tools/moe_stats.py`,
    `speccheck/moe_ref.py`, `speccheck/selftest_moe.py`, `prototype/moe_router/gate.py`); this is what
    says the shipped arms agree with the one ruler of record. The patches are in
    `ulp-policy-migration-2026-10-26.md`, their re-verification in
    `ulp-policy-migration-verified-2026-10-26.md`.
(2) It is now also where the PRE-migration bodies live. That half is not nostalgia: the must-catch set
    that made this harness worth running named classes
    (`one-step-fewer-where-the-pair-straddles-zero`, `identifies-plus-zero-with-minus-zero`) that a
    migrated function can no longer produce, so demanding them of a correct ruler yields an EMPTY
    must-fire set — which reads as green and proves nothing (`pitfalls/40` §"a negative control with an
    empty must-catch set is not a control"). The classes are therefore demanded of the bodies that
    carried the defects, and the shipped arms are demanded to show NONE of them: a regression that puts
    a fold back turns the shipped arm red, which is a stronger position than HEAD's, where the classes
    were required of a function that had already been fixed and so could only ever read none.

It is an equivalence harness, not a re-implementation: every foreign ruler is IMPORTED and CALLED, so
the comparison is between the code that ships and the new ruler, not between two readings of a docstring
(`pitfalls/40` §"a check whose reference side is derived from the thing under test"). The one deliberate
exception is `legacy_fold_f32`, which is a body that no longer ships anywhere and is kept precisely so
it can still be executed.

The fixture population is DERIVED, not hand-listed (`pitfalls/40` §"an admission over a fixture list
typed from memory"): for the 16-bit formats every image is used, and for f32 the population is every
exponent crossed with a mantissa set that includes the boundaries (0, 1, 2, half, half±1, max), each
with both signs and with the ±1/±2 neighbourhood. The pairs cover the cases the two rulers are reported
to disagree on: powers of two, subnormals, the subnormal/normal edge, signed zeros, ±inf, NaN payloads,
sign flips, and pairs straddling zero.

The harness asserts its own non-vacuity twice over: the must-fire classes of §2 must be non-empty, and
each arm's OBLIGATION census (§obligations) must find a non-empty population in every class it judges —
a signed-zero pair, a straddling pair, an overflow step, a NaN pair. A class with no population in it is
a finding, not a pass.

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
moe_ref_src = open(os.path.join(REPO, "speccheck/moe_ref.py")).read()
# The width check below is about a VALUE, so the module is imported rather than grepped: a string test
# can be satisfied by a comment, and a constant that was deleted fails a value test loudly.
mstats = load_module(os.path.join(REPO, "tools/moe_stats.py"), "moe_stats_under_test")


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
# the body that no longer ships, and the obligations that only a corrected ruler satisfies
# ---------------------------------------------------------------------------------------------
F32_SIGN = UP.format_facts("f32").sign_bit


def legacy_fold_f32(a, b):
    """`gate.py`'s pre-D-17 f32 step body, verbatim: `a if a < 0x80000000 else 0x80000000 - a`.

    The shipped `ulpdist_bits` was folded onto `tools/ulp_policy.py`, so the two classes this harness
    used to demand of it can no longer fire against it. Deleting the classes would have turned a real
    control into a green row, so the body that HAD the defect lives here and the classes are demanded
    of it — the same shape as `oracle_ruler_repair_witness.py::historic_ulpdist`: a bug you cannot
    re-run is a bug you cannot show stays fixed.

    Do not confuse this with `prototype/moe_router/oracle_arm.py`'s old body. That one tested a SIGNED
    int32 view and returned the COMPLEMENT `2^32 + 1 - d` across zero; this one tests the UNSIGNED
    image against the sign bit, folds `+0` onto `-0`, and is short by exactly one on every straddling
    pair. Quoting the repair to the wrong bug is how a fix gets believed for something it never was
    (`specification/evidence/oracle-ruler-repair-witness-2026-10-26.log` is the other one's evidence).
    """
    def key(w):
        return w if w < F32_SIGN else F32_SIGN - w
    return abs(key(a) - key(b))


def obligation_population(fmt, ruler, pairs):
    """Select, in one pass, the pairs each obligation is about — so every denominator is derived.

    `overflow` is the pair an overflowing product actually produces: one member infinite, the other the
    adjacent finite image. It is NOT "any pair involving an infinity", which would let a ruler that
    counts `1.0 → +inf` as one step pass a check that is about the step above the largest finite.
    """
    sign, mask = fmt.sign_bit, fmt.mask
    abs_mask = mask ^ sign
    inf_image = ((1 << fmt.exp) - 1) << fmt.mant
    sel = {"signed_zero": [], "straddling": [], "overflow": [], "nan": [], "inf_finite": 0}
    for _, a, b in pairs:
        if ruler.is_nan(a) or ruler.is_nan(b):
            # A NaN pair is not also a straddling pair or an overflow pair: no arm counts it, and the
            # fourth obligation is about exactly that. Counting it twice would make the straddling
            # census depend on how the arm answers a question it is not supposed to answer.
            sel["nan"].append((a, b))
            continue
        a_inf, b_inf = (a & abs_mask) == inf_image, (b & abs_mask) == inf_image
        if a_inf != b_inf:
            sel["inf_finite"] += 1
            if abs(ruler.order_key(a) - ruler.order_key(b)) == 1:
                sel["overflow"].append((a, b))
        if ((a & sign) != 0) != ((b & sign) != 0):
            if a in (0, sign) and b in (0, sign):
                sel["signed_zero"].append((a, b))
            else:
                sel["straddling"].append((a, b))
    return sel


def obligations(v, arm, fmt, ruler, answer, sel, nan_mode, control=None):
    """Four things an arm must be SEEN to do, each over a population that has to be non-empty.

    `answer(a, b)` is the arm's step answer for two BIT IMAGES. `nan_mode` is "refuses" where the arm
    answers None on the pairs the ruler of record answers None for, and "numeric" where it keeps
    printing a number — the latter is documented and tolerated for a printing tool, but it is counted
    here rather than waved through, because "tolerated" and "unmeasured" are the same thing in a log.

    These are the mirror image of the must-catch classes: not "the old body must fail" but "the ruler
    that replaced it must be observed succeeding", each pinned to the specific way it could break.

    `control` names the classes that must FIRE rather than stay clean, and is used exactly once — on
    `legacy_fold_f32`. An obligation that has never gone red on anything is an obligation nobody has
    tested, so the pre-D-17 body is run through the identical probes and required to fail three of the
    four and pass the fourth (same-sign pairs were always measured correctly, which is what makes the
    defect a crossing-zero defect and not a broken ruler).
    """
    sign = ruler.sign
    tag = " [control]" if control else ""

    def emit(cls, what, ok, n, detail):
        if control is None:
            return v.check("%s: %s" % (arm, what), ok, n, detail)
        fired = not ok
        return v.check("%s%s: %s — %s" % (arm, tag, what,
                                          "MUST FIRE" if cls in control else "MUST STAY CLEAN"),
                       fired == (cls in control), n,
                       "the shipped-arm obligations above can only be believed if they are capable of "
                       "going red; this runs the pre-D-17 body through the same probes and requires "
                       "%s" % ("it to fire here, otherwise the probe is decoration" if cls in control
                               else "it NOT to fire, which scopes the defect to crossing zero instead "
                                  "of indicting the whole ruler")
                       + "; " + detail)

    n = len(sel["signed_zero"])
    off = ["a=0x%08x b=0x%08x answer=%r" % (a, b, answer(a, b))
           for a, b in sel["signed_zero"] if answer(a, b) != 1]
    print("OBLIGATION%s arm=%s class=signed_zero pairs=%d not_one_step=%d"
          % (tag, arm, n, len(off)))
    emit("signed_zero", "a signed-zero pair counts ONE step, never zero", n > 0 and not off, n,
         "+0/-0 is the one distance fixed by the format itself, so it is checkable without trusting "
         "either ruler; offenders: %s" % (off[:3] or "none"))

    n = len(sel["straddling"])
    off = []
    for a, b in sel["straddling"]:
        za = sign if (a & sign) else 0
        zb = sign if (b & sign) else 0
        got = answer(a, b)
        # a -> zero-of-a -> the OTHER signed zero -> b. The lone +1 is the -0/+0 adjacency, and it is
        # the whole step the pre-D-17 fold left out (one key for both zeros), which is why this
        # identity is the check and not a tautology about the ruler.
        want = answer(a, za) + 1 + answer(zb, b)
        if got != want or not isinstance(got, int) or got < 1:
            off.append("a=0x%08x b=0x%08x answer=%r via_own_zeros=%r" % (a, b, got, want))
    print("OBLIGATION%s arm=%s class=straddling pairs=%d not_additive_thru_own_zeros=%d"
          % (tag, arm, n, len(off)))
    emit("straddling", "a pair straddling zero counts through its own signed zeros, plus the "
         "-0/+0 step", n > 0 and not off, n,
         "steps(a,b) must equal steps(a, zero-of-a) + 1 + steps(zero-of-b, b); the pre-D-17 fold "
         "returned one fewer on 13,771 pairs, so a band measured with it was one step laxer across "
         "zero than it claimed. Offenders: %s" % (off[:3] or "none"))

    n = len(sel["overflow"])
    off = ["a=0x%08x b=0x%08x answer=%r" % (a, b, answer(a, b))
           for a, b in sel["overflow"] if answer(a, b) != 1]
    # `within()` is the ACCEPT side, and it must not accept what the count side counts: an overflow is
    # one representable step from the largest finite, and a ±1-ULP band is not the thing that forgives it
    accepted = ["a=0x%08x b=0x%08x" % (a, b) for a, b in sel["overflow"] if ruler.within(a, b, 1)]
    print("OBLIGATION%s arm=%s class=overflow_step pairs=%d(of %d inf-finite) not_one_step=%d "
          "accepted_by_within=%d" % (tag, arm, n, sel["inf_finite"], len(off), len(accepted)))
    emit("overflow", "largest finite to +/-inf is one step, and within() still refuses it",
         n > 0 and not off and not accepted, n,
         "the counting side must see 1 (an overflow is a rounding step, not an unrelated number) and "
         "the accepting side must see a refusal (an infinity is not within 1 ULP of anything); "
         "offenders: not_one=%s within_accepted=%s" % (off[:2] or "none", accepted[:2] or "none"))

    n = len(sel["nan"])
    if nan_mode == "refuses":
        counted = ["a=0x%08x b=0x%08x answer=%r" % (a, b, answer(a, b))
                   for a, b in sel["nan"] if answer(a, b) is not None]
        print("OBLIGATION%s arm=%s class=nan pairs=%d counted_instead_of_refused=%d"
              % (tag, arm, n, len(counted)))
        emit("nan", "every pair the ruler of record answers None for is REFUSED, not counted",
             n > 0 and not counted, n,
             "a sentinel here becomes the maximum of a band, so a NaN in an input would widen the "
             "lane's allowance with a number that means nothing; offenders: %s"
             % (counted[:3] or "none"))
    else:
        numeric = [(a, b, answer(a, b)) for a, b in sel["nan"] if answer(a, b) is not None]
        print("OBLIGATION%s arm=%s class=nan pairs=%d counted=%d (documented numeric arms) "
              "example=%s" % (tag, arm, n, len(numeric),
                              ("a=0x%08x b=0x%08x answer=%r" % numeric[0]) if numeric else "none"))
        emit("nan", "its gap is EXACTLY the NaN arms, and they are counted",
             n > 0 and len(numeric) == n, n,
             "this arm still prints a number where the ruler of record answers None; tolerated "
             "because the arm is a printing tool, but the population is measured, not assumed — if "
             "it ever disagrees with the ruler on a NON-NaN pair the equivalence checks above catch "
             "it")


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

    # --- prototype/moe_router/gate.py §ulpdist_bits (f32 images) — TWO arms, on purpose ---------
    # The pre-D-17 body first, because it is the only thing left that can produce the classes the
    # migration note names. Its expected counts are not invented: `ulp-policy-migration-verified-2026-10-26.md`
    # §1 records 13,771 straddling pairs plus the two signed-zero images out of the same 69,089, so a
    # drift here means the corpus or the legacy body changed and the note's number went stale.
    t, verdict, unmet = cmp_step_ruler(
        "legacy pre-D-17 body: a if a < 0x80000000 else 0x80000000 - a", r32, p32,
        legacy_fold_f32, "f32-images",
        must_fire=("one-step-fewer-where-the-pair-straddles-zero",
                   "identifies-plus-zero-with-minus-zero",
                   "nan-arm:the shared ruler refuses to count a step here"))
    print(t.line("legacy pre-D-17 folded key", "f32-images", verdict,
                 "its map folds +0 and -0 onto one key: every pair straddling zero is counted ONE STEP "
                 "SHORT of the true number of representable steps, and +0/-0 read as identical. The "
                 "direction matters — a band measured with this ruler is one step LAXER across zero "
                 "than the band it claims to be"))
    v.check("the pre-D-17 folded key still produces every class the migration note names",
            not unmet, t.n - t.agree,
            "unmet must-fire classes: %s — if this reads none the harness is vacuous, which is exactly "
            "what it became when the shipped ruler was migrated and the classes were left pointing at "
            "it" % (unmet or "none"))
    v.check("the legacy arm reproduces the verified migration note's own denominators",
            t.classes.get("one-step-fewer-where-the-pair-straddles-zero") == 13771
            and t.classes.get("identifies-plus-zero-with-minus-zero") == 2,
            t.classes.get("one-step-fewer-where-the-pair-straddles-zero", 0)
            + t.classes.get("identifies-plus-zero-with-minus-zero", 0),
            "expected 13,771 straddling + 2 signed-zero out of 69,089 "
            "(ulp-policy-migration-verified-2026-10-26.md §1); got %s — this is the figure that makes "
            "the must-catch set non-vacuous, so it is pinned rather than admired"
            % (" ".join("%s=%d" % kv for kv in sorted(t.classes.items())) or "none"))

    t, verdict, unmet = cmp_step_ruler(
        "prototype/moe_router/gate.py:ulpdist_bits", r32, p32,
        lambda a, b: router.ulpdist_bits(a, b), "f32-images",
        must_fire=("nan-arm:the shared ruler refuses to count a step here",))
    legacy_seen = sorted(set(t.classes) -
                         {"nan-arm:the shared ruler refuses to count a step here"})
    print(t.line("prototype/moe_router/gate.py:ulpdist_bits", "f32-images", verdict,
                 "the SHIPPED body now delegates to tools/ulp_policy.py, so the only class it may show "
                 "is the NaN arm; the two fold classes appearing here would mean the fold came back"))
    v.check("the shipped ulpdist_bits shows none of the fold classes and meets its must-fire arm",
            not unmet and not legacy_seen, t.n - t.agree,
            "unmet must-fire classes: %s; fold classes present: %s — the first is a vacuous harness, "
            "the second a regression, and they are reported together because a check that cannot be "
            "green for two different reasons must not be red for one" % (unmet or "none",
                                                                          legacy_seen or "none"))

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

    # --- the obligations only a CORRECTED ruler can satisfy --------------------------------------
    # One census per shipped arm. The classes are the ones the repairs were FOR, so each must have a
    # non-empty population: a ruler that never sees a signed-zero pair cannot be credited with counting
    # it as 1, and a ruler that never sees an overflow step cannot be credited with counting it as 1
    # while `within()` refuses it (`pitfalls/40` §"count reachability").
    sel32 = obligation_population(f32, r32, p32)
    sel16 = obligation_population(bf16, r16, p16)
    print("OBLIGATIONSEL f32 signed_zero=%d straddling=%d overflow=%d(of %d inf-finite) nan=%d"
          % (len(sel32["signed_zero"]), len(sel32["straddling"]), len(sel32["overflow"]),
             sel32["inf_finite"], len(sel32["nan"])))
    print("OBLIGATIONSEL bf16 signed_zero=%d straddling=%d overflow=%d(of %d inf-finite) nan=%d"
          % (len(sel16["signed_zero"]), len(sel16["straddling"]), len(sel16["overflow"]),
             sel16["inf_finite"], len(sel16["nan"])))
    obligations(v, "tools/audit_published_quant.py:ulp_dist", f32, r32,
                lambda a, b: audit.ulp_dist(r32.value(a), r32.value(b)), sel32, "numeric")
    obligations(v, "prototype/moe_router/gate.py:ulpdist_bits", f32, r32,
                lambda a, b: router.ulpdist_bits(a, b), sel32, "refuses")
    obligations(v, "speccheck/moe_ref.py:ulpdist_bf16", bf16, r16,
                lambda a, b: moe_ref.ulpdist_bf16(a, b)[1], sel16, "refuses")
    obligations(v, "prototype/moe_router/gate.py:bf16_ulpdist", bf16, r16,
                lambda a, b: router.bf16_ulpdist(a << 16, b << 16), sel16, "numeric")
    # The obligations, witnessed firing. Three of the four must go red on the pre-D-17 folded body and
    # one must stay clean: same-sign pairs were always measured correctly, which is what makes this a
    # crossing-zero defect rather than a broken ruler — the same asymmetry
    # `oracle_ruler_repair_witness.py` measures on the router's other old body. A control that failed
    # everything would tell us nothing about where the corrected ruler is trusted.
    obligations(v, "legacy pre-D-17 folded key", f32, r32, legacy_fold_f32, sel32, "refuses",
                control=("signed_zero", "straddling", "nan"))

    # --- the prose-vs-code width claims, in the two files that once typed a bf16 width -----------
    v.check("bf16 explicit mantissa bits derive to 7, not 8 (8 is the precision INCLUDING the implicit)",
            bf16.mant == 7, bf16.mant,
            "ulp_of's formula 2^(exp-7) was always right; what was wrong was the prose calling the "
            "field 8 bits. Both files now state 7 and keep the old wording only as a citation, and the "
            "check after this one is what proves that rather than asserting it. Derived twice: %s + %s"
            % (",".join(bf16.sources), bf16.crosscheck))

    # The prose trap, still witnessed rather than deleted. Both files still contain the words
    # "8 explicit", but only while saying that is what they USED to say (bf16 is 1 sign + 8 exponent +
    # 7 mantissa; 8 is the precision INCLUDING the implicit bit). A line that says "8 explicit" without
    # naming itself as the old claim has re-asserted the wrong width, and requiring the citation to be
    # present keeps this from being a check that can produce no output (`pitfalls/40`).
    def eight_bit_lines(src, path):
        offenders, cited = [], 0
        for i, line in enumerate(src.splitlines(), 1):
            if "8 explicit" not in line:
                continue
            if any(w in line for w in ("said", "called", "old comment", "used to")):
                cited += 1
            else:
                offenders.append("%s:%d %s" % (path, i, line.strip()))
        return offenders, cited
    for src, path, right in ((moe_ref_src, "speccheck/moe_ref.py", "7 explicit"),
                             (moe_stats_src, "tools/moe_stats.py", "7 EXPLICIT")):
        off, cited = eight_bit_lines(src, path)
        print("PROSE file=%s cites_the_old_8_bit_claim=%d reasserts_it=%d states_7_explicit=%d"
              % (path, cited, len(off), int(right in src)))
        v.check("%s states 7 explicit bits and cites the 8-bit claim as history" % path,
                right in src and cited > 0 and not off, cited,
                "the wrong width must still appear (as history) and the right one as the claim; lines "
                "re-asserting 8: %s" % (off or "none"))

    # The constant HEAD's check asserted the WRONG STATE of. It used to require the literal string
    # `BF16_REL_WIDTH = 2.0 ** -9`, which was the pre-correction file; once the correction landed, that
    # check could only be satisfied by reverting the file, which is a green nobody wants. The obligation
    # is the one that is true in both worlds: the constant exists, and its VALUE is half an ULP of
    # relative width. Half, not one: the spacing above 1.0 is 2**-mant (ulp_of(1.0) == 0.0078125), so
    # half of it is 2**-(mant+1). The staged patch wrote 2**-(mant-1) = 2**-6, four times too big, which
    # is why three values are named here and only one is accepted.
    half_ulp = 2.0 ** -(bf16.mant + 1)
    typed8 = 2.0 ** -9
    staged_patch = 2.0 ** -(bf16.mant - 1)
    got_width = getattr(mstats, "BF16_REL_WIDTH", None)
    print("VALUE moe_stats.BF16_REL_WIDTH got=%r accepted=%r rejected=[typed8 %r, staged_patch %r] "
          "mant=%d ulp_of(1.0)=%r"
          % (got_width, half_ulp, typed8, staged_patch, bf16.mant, moe_ref.ulp_of(1.0)))
    v.check("moe_stats.BF16_REL_WIDTH is the derived half-ULP relative width 2^-(mant+1)",
            got_width == half_ulp and got_width not in (typed8, staged_patch), got_width,
            "a value test, not a string test: a comment can satisfy a grep and cannot satisfy this, and "
            "deleting the constant makes getattr answer None and fail here rather than pass silently. "
            "The before/after of the patch itself is witnessed by ulp_policy_smoke_staged.py and "
            "ulp_policy_migration_witness.py; this holds the shipped figure")
    v.check("no bf16 width literal is typed in tools/moe_stats.py any more",
            not any(s in moe_stats_src for s in ("2.0 ** -9", "2.0 ** -6", "2.0 ** -8")),
            moe_stats_src.count("** -"),
            "the constant must come from format_facts('bf16').mant; the exponents it must NOT carry "
            "appear in its comments as prose (2**-9, 2**-6, 2**-8) and are excluded because those are "
            "not Python float literals — this check is about what would evaluate, not what is mentioned")

    # "It is dead code" is a claim about the whole repo, so it is counted over the whole repo, and the
    # counter is witnessed non-zero on the evidence files first: a search that finds nothing anywhere
    # proves nothing about the one place it must (`pitfalls/40` §"a counter that reads 0").
    readers, witnesses = [], []
    for base, _dirs, files in os.walk(REPO):
        if "/.git" in base or "__pycache__" in base:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, REPO)
            for i, line in enumerate(open(p, errors="ignore").read().splitlines(), 1):
                if "BF16_REL_WIDTH" not in line:
                    continue
                if rel.startswith("specification" + os.sep):
                    witnesses.append("%s:%d" % (rel, i))
                elif not line.strip().startswith("BF16_REL_WIDTH ="):
                    readers.append("%s:%d" % (rel, i))
    print("READERS name=BF16_REL_WIDTH outside_its_definition_in_shipped_code=%d "
          "mentions_in_evidence=%d %s"
          % (len(readers), len(witnesses), witnesses[:3]))
    v.check("BF16_REL_WIDTH still has no reader in shipped code, and the search can see the name",
            not readers and len(witnesses) > 0, len(readers),
            "the note under the value claim is that no published number moves when this constant "
            "changes; that is only safe while it is true, and the second half of the check is the "
            "witness that the walker is not simply blind (the evidence files must mention the name). "
            "Readers found: %s" % (readers[:4] or "none"))

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
