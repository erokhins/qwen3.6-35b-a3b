#!/usr/bin/env python3
"""ulp_policy_stage_patches.py — generate the STAGED migration patches, and prove each one applies.

`DECISIONS.md` D-17's one-ruler rule reaches five files, and four of them belong to other lanes
(`speccheck/` has a live worker in it right now). So the migration is prepared here, and the main agent
applies it when each lane is quiet. Each patch is generated from the file AS IT IS NOW — one exact-string
rewrite per patch, refused if the anchor text is not unique — and then verified two ways: `patch -p1
--dry-run` in a scratch tree, and `ast.parse` of the patched text. A hand-written diff that has never
been applied is a wish, not a patch.

Writes `ulp-policy-migrate-<label>.patch` next to itself (git/`patch -p1` applicable), and prints per
patch: the hunk count, whether it applies, whether the patched file still parses, and the behavioural
consequence a reviewer has to accept or reject. Patches that change a measurement are marked as belonging
to their lane; staging is not applying.

Run with the CPU venv python, the interpreter of record for CPU work in this repo.
"""
import ast
import difflib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# (target, label, exact old text, new text, consequence for the reviewer)
EDITS = [
    (
        "tools/audit_published_quant.py", "audit-ulp-dist",
        '''def ulp_dist(a, b):
    return abs(_f32_ord(a) - _f32_ord(b))
''',
        '''def ulp_dist(a, b):
    """f32 ULP distance, delegated to the repo's one ruler (DECISIONS.md D-17).

    `specification/evidence/ulp_policy_equiv.py` proves the delegated ruler agrees with the body this
    replaced on every finite pair of a 69,089-pair derived corpus. The NaN arms differ on purpose: the
    shared ruler refuses to call a NaN a rounding step, while this tool prints a number for every pair
    it judges, so the numeric answer is kept there and named rather than silently large.
    """
    import sys as _sys
    _sys.path.insert(0, HERE)
    import ulp_policy
    r = getattr(ulp_dist, "_ruler", None)
    if r is None:
        r = ulp_dist._ruler = ulp_policy.Ruler("f32")   # f32 derives from struct: stdlib only
    kind, steps = r.classify(struct.unpack("<I", struct.pack("<f", a))[0],
                             struct.unpack("<I", struct.pack("<f", b))[0])
    if kind in (ulp_policy.NAN_PAIR, ulp_policy.NAN_OTHER):
        return abs(_f32_ord(a) - _f32_ord(b))
    return steps
''',
        "No behaviour change on finite pairs (proved by the harness); NaN pairs keep the old numeric "
        "answer deliberately. Safe to apply whenever this tool is next quiet.",
    ),
    (
        "prototype/moe_router/gate.py", "router-import",
        '''HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
''',
        '''HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
import ulp_policy                                   # noqa: E402  the repo's one ULP ruler (D-17)

_RULER32 = ulp_policy.Ruler("f32")                  # f32 facts derive from `struct`, stdlib only
''',
        "Import only; no measurement changes.",
    ),
    (
        "prototype/moe_router/gate.py", "router-ulpdist-bits",
        '''def ulpdist_bits(a, b):
    ai = a if a < 0x80000000 else 0x80000000 - a
    bi = b if b < 0x80000000 else 0x80000000 - b
    return abs(ai - bi)
''',
        '''def ulpdist_bits(a, b):
    """Steps between two f32 images — delegated to the repo's one ruler (DECISIONS.md D-17).

    The body this replaced (`a if a < 0x80000000 else 0x80000000 - a`) folds +0 and -0 onto one key, so
    EVERY pair straddling zero was counted one step short of the true number of representable steps and
    a signed-zero flip read as no difference at all: 13,771 of 69,089 pairs in the derived corpus of
    `specification/evidence/ulp_policy_equiv.py`. A NaN pair now answers None; every call site here
    compares the result against a band, and `None <= band` is a TypeError rather than a silent pass,
    which is the point — NaN reaching a band is a finding, not a small value.

    CONSEQUENCE THIS LANE OWNS: the bands measured with the short ruler
    (`weights_vs_pinned_f32_ulp`, `softmax_denominator_reassoc_ulp`) were measured with the short ruler,
    so a straddling pair now reads one step larger. Re-derive them before quoting the lane's verdicts
    again. Staged, not pushed.
    """
    kind, steps = _RULER32.classify(a, b)
    if kind in (ulp_policy.NAN_PAIR, ulp_policy.NAN_OTHER):
        return None
    return steps
''',
        "CHANGES measurements: straddling pairs read +1 step, and a NaN now raises at the band "
        "comparison instead of passing. The router lane must re-derive its two bands first.",
    ),
    (
        "speccheck/moe_ref.py", "moe-ref-ulp-of",
        '''def ulp_of(v: float) -> float:
    """One bf16 ULP at magnitude |v| — 8 explicit mantissa bits, so ULP(v) = 2^(exp-7)."""
    v = abs(float(v))
    if v == 0 or not math.isfinite(v):
        return 0.0
    return math.ldexp(1.0, math.floor(math.log2(v)) - 7)
''',
        '''_BF16_RULER = None


def _bf16_facts():
    """bf16's format facts, derived once by tools/ulp_policy.py (mantissa width, bias, edges)."""
    global _BF16_RULER
    if _BF16_RULER is None:
        import sys
        sys.path.insert(0, str(REPO / "tools"))
        import ulp_policy
        _BF16_RULER = ulp_policy.format_facts("bf16")
    return _BF16_RULER


def ulp_of(v: float) -> float:
    """The width of one bf16 step ABOVE |v| — a LENGTH, not a count of steps.

    The prose here said "8 explicit mantissa bits". bf16 has 7 explicit bits (1 sign + 8 exponent + 7
    mantissa; 8 is the precision *including* the implicit bit), and the formula's `-7` was always the
    right one — the number now comes from the derived facts instead of a literal.

    What this function cannot do, and the ruler can: it is a length at |v|, so it cannot count the
    representable steps between two values. It returns 0.0 at +/-0 and for non-finite inputs, and it
    mis-states the step across the whole subnormal range (1,280 of 327,548 bf16 images in
    `specification/evidence/ulp_policy_equiv.py`'s corpus, by up to 128x). A caller that is *scaling* a
    tolerance is fine; a caller that is *counting* steps must use `tools/ulp_policy.py`.
    """
    v = abs(float(v))
    if v == 0 or not math.isfinite(v):
        return 0.0
    return math.ldexp(1.0, math.floor(math.log2(v)) - _bf16_facts().mant)


def ulpdist_bf16(bits_a: int, bits_b: int):
    """Steps between two bf16 images, as (category, steps) — the one ruler's answer (D-17)."""
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    import ulp_policy
    return ulp_policy.Ruler("bf16").classify(bits_a, bits_b)
''',
        "`ulp_of`'s numeric answer is unchanged for normal inputs (mant derives to exactly the 7 it "
        "already typed). The prose defect is fixed, the limits are written where a caller can read "
        "them, and the lane gains a step-counting entry point. `selftest_moe.py:263` calls `ulp_of` only "
        "to scale a printed number, so it is unaffected.",
    ),
    (
        "tools/moe_stats.py", "moe-stats-rel-width",
        '''BF16_REL_WIDTH = 2.0 ** -9      # 8 explicit mantissa bits -> half-ULP relative width
''',
        '''# bf16 has 7 EXPLICIT mantissa bits (1+8+7); the precision including the implicit bit is 8, which is
# what the old comment here called "8 explicit". Half-ULP relative width is therefore 2**-8, not 2**-9.
# Derived rather than typed: see tools/ulp_policy.py and DECISIONS.md D-17. grep shows this constant has
# no reader anywhere in the file, so no number this tool ever printed changes — deleting it is the better
# move, and this patch is the fallback if it is kept.
sys.path.insert(0, str(REPO / "tools"))
import ulp_policy as _UP                            # noqa: E402
BF16_REL_WIDTH = 2.0 ** -(_UP.format_facts("bf16").mant - 1)
''',
        "Dead constant: no live number changes either way. Reported because it is a second copy of the "
        "same typed width, not because it is wrong in a result.",
    ),
]


def unified(target, new_text):
    orig = open(os.path.join(REPO, target)).read().splitlines(keepends=True)
    return "".join(difflib.unified_diff(orig, new_text.splitlines(keepends=True),
                                       fromfile="a/" + target, tofile="b/" + target))


def verify(label, target, patch_path, patched_text):
    """`patch -p1 --dry-run` against a pristine copy, and `ast.parse` of the patched text."""
    d = "/tmp/ulp_stage_check_%s" % label
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, os.path.dirname(target)), exist_ok=True)
    shutil.copyfile(os.path.join(REPO, target), os.path.join(d, target))
    p = subprocess.run(["patch", "-p1", "--dry-run", "-i", patch_path], cwd=d,
                       capture_output=True, text=True)
    applies = p.returncode == 0
    tmp = os.path.join(d, "patched_copy.py")
    shutil.copyfile(os.path.join(d, target), tmp)
    shutil.copyfile(os.path.join(REPO, target), tmp)
    # parse the PATCHED text, not the pristine copy: write it out and hand the file to ast
    with open(tmp, "w") as fh:
        fh.write(patched_text)
    a = subprocess.run([sys.executable, "-c",
                        "import ast,sys;ast.parse(open(sys.argv[1]).read())", tmp],
                       capture_output=True, text=True)
    return applies, (p.stdout + p.stderr).strip(), a.returncode == 0, a.stderr.strip()


def main():
    n_ok = 0
    problems = []
    for target, label, old, new, note in EDITS:
        src = open(os.path.join(REPO, target)).read()
        if src.count(old) != 1:
            problems.append("%s: anchor text occurs %d times in %s — a patch generated against a "
                            "guess is worse than no patch" % (label, src.count(old), target))
            print("STAGE label=%s target=%s SKIP anchor not unique (%d occurrences)"
                  % (label, target, src.count(old)))
            continue
        patched = src.replace(old, new, 1)
        pf = os.path.join(HERE, "ulp-policy-migrate-%s.patch" % label)
        with open(pf, "w") as fh:
            fh.write(unified(target, patched))
        applies, msg, parses, perr = verify(label, target, pf, patched)
        hunks = sum(1 for line in open(pf) if line.startswith("@@"))
        print("STAGE label=%s target=%s patch=%s bytes=%d hunks=%d dry_run_applies=%s "
              "patched_file_parses=%s" % (label, target, os.path.basename(pf), os.path.getsize(pf),
                                          hunks, "yes" if applies else "NO",
                                          "yes" if parses else "NO " + perr[:120]))
        print("STAGENOTE label=%s %s" % (label, note))
        if applies and parses:
            n_ok += 1
        else:
            problems.append("%s: verification failed (%s)" % (label, msg[:200]))
    print("RESULT %s staged=%d of %d  (%s)"
          % ("PASS" if not problems else "FAIL", n_ok, len(EDITS),
             "; ".join(problems) if problems else "every patch applies and every patched file parses"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
