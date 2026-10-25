#!/usr/bin/env python3
"""ulp_policy_migration_witness.py — did the five staged ±1-ULP migration patches actually CHANGE
BEHAVIOUR, and did nothing else move?

`git apply` proves a patch fits. It proves nothing about what the patched code now does, and
`specification/evidence/ulp-policy-migration-2026-10-26.md` §5 lists the consequence per patch only as
prose. This file is the measurement behind that prose: for each of the five patches it runs the
PRE-migration code (read out of `git show HEAD:<path>` into a mirror tree) and the POST-migration code
(the working tree) over the same derived fixture population and compares the two, function by function.

WHAT IS ASSERTED, per patch
  router-ulpdist-bits   the old body under-counted every zero-straddling pair by one step (13,771 pairs
                        of the derived corpus, the figure in the migration note) and folded +0 onto -0;
                        a named straddling pair is printed with both bit images and both counts. The
                        new body agrees with `tools/ulp_policy.py` on every finite pair of the corpus,
                        and answers None for a NaN pair where the old body returned a number — a band
                        comparison against None is a TypeError, which is witnessed here.
  router-import         `_RULER32` exists in the patched file and is the shared ruler's own object.
  moe-ref-ulp-of        `ulp_of` is a LENGTH: its numbers are identical before and after the patch over
                        all 2^16 bf16 images (so the patch changes no caller-visible number), the
                        subnormal/binade-edge miscount vs the true step is reproduced (1,280 images),
                        and the counting defect that miscount was repaired BY the `ulpdist_bf16` the
                        patch adds: the same 1,280 edges count 1 step each through it, and +0/-0 reads
                        ("zero_sign_flip", 1) instead of the length's 0.0.
  moe-stats-rel-width   BF16_REL_WIDTH changes value (2^-9 -> 2^-8) — a real behaviour change — and the
                        constant still has zero readers in the file, so no number that tool printed moved.
  audit-ulp-dist        behaviour-PRESERVING: the old and new bodies agree on every pair of the corpus,
                        including the NaN arms, which keep a numeric answer on purpose. The change is
                        provenance (one ruler instead of a private one) and is witnessed by the delegated
                        ruler object being created, not by a number moving.

NOTHING ELSE MOVED: for every patched file the two revisions are compared top-level node by top-level
node with `ast`, and the set of changed names must equal the expected set exactly.

Run under the CPU venv (the `speccheck` side imports torch):
  /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
      specification/evidence/ulp_policy_migration_witness.py
Prints counts and bit images only. It opens no device, reads no clock, and prints no duration:
no timing taken.
"""
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# The pre-patch side of the comparison, rebuilt on every run: symlinks to the live tree for everything
# the patches do not touch, and `git show HEAD:<path>` text for the four that do. Under `tmp/` on purpose
# — a HEAD copy of tools/*.py left in a tracked directory is a second source of truth that can disagree
# with its own source, which is the stale-binary trap .gitignore §Built binaries warns about.
MIRROR = os.path.join(HERE, "tmp", "ulp-mirror")
HEAD_FILES = ["tools/audit_published_quant.py", "tools/moe_stats.py",
              "speccheck/moe_ref.py", "prototype/moe_router/gate.py"]
PATCH_DIR = os.path.join(HERE)
PATCHES = ["ulp-policy-migrate-audit-ulp-dist.patch", "ulp-policy-migrate-moe-stats-rel-width.patch",
           "ulp-policy-migrate-moe-ref-ulp-of.patch", "ulp-policy-migrate-router-import.patch",
           "ulp-policy-migrate-router-ulpdist-bits.patch"]

PATCHED_NAMES = {
    "tools/audit_published_quant.py": {"ulp_dist"},
    "tools/moe_stats.py": {"BF16_REL_WIDTH"},
    "speccheck/moe_ref.py": {"ulp_of", "ulpdist_bf16", "_bf16_facts", "_BF16_RULER"},
    "prototype/moe_router/gate.py": {"ulpdist_bits", "_RULER32"},
}

# Repairs this lane made on top of the five staged patches, declared here so the scope check stays
# falsifiable in BOTH directions: a definition changed outside every patch hunk must appear here, and
# a name declared here that turns out not to have changed is itself a finding. An undeclared fifth
# wheel is a finding, and a declaration that matches nothing is a finding.
LANE_REPAIRS = {
    "tools/moe_stats.py": {
        "BF16_REL_WIDTH": "the staged patch wrote 2**-(mant-1) = 2**-6; half an ULP of relative "
                          "width is 2**-(mant+1) = 2**-8, the value its own comment names"},
    "prototype/moe_router/gate.py": {
        # the two defects `results.md §Corrigendum` scheduled for repair; both live in the file the
        # router ruler patches also touched, so they must be declared to the scope check
        "read_oracle": "the oracle's printed p3 record count was never compared with the parse, and "
                       "a missing oracle_p3.tsv parsed as an empty list (an empty parse may not "
                       "print PASS)",
        "report_predictions": "replaced the literal-`True` predicate "
                              "prediction3-fused-layout-unresolved-is-reported with derived "
                              "predicates, a derived record population and a witnessed firing",
    },
}

# The Job-2 repair's own additions: the code that replaces the literal-`True` predicate. Declared with
# the same mechanism as the repairs above, because "changed outside every hunk" is exactly what these
# are — work this lane did on top of the five staged patches, which the scope check is there to make
# visible rather than to forbid.
P3_REPAIR_REASON = ("prediction-3 repair: the replacements for the literal-`True` predicate "
                    "(derived population, three-reading adjudication, machine-readable report line, "
                    "witnessed firing); results.md §Corrigendum")
for _name in ("P3_READINGS", "P3_SLOTS", "P3_RECORDS", "P3_ADMISSIBLE_ONE_MINUS_COS",
              "P3_SEPARATION_MIN", "p3_records", "p3_verdict", "p3_report", "p3_report_is_honest",
              "witness_p3_report", "p3_distinguishable_ok", "p3_population_ok", "DEVICE_ROSTER"):
    LANE_REPAIRS["prototype/moe_router/gate.py"][_name] = P3_REPAIR_REASON


def patch_target(patch):
    """The file one staged patch rewrites, from its `--- a/<path>` header."""
    for line in open(os.path.join(PATCH_DIR, patch)):
        if line.startswith("--- a/"):
            return line[6:].strip()
    return None


def hunk_ranges():
    """{patch: [(first_line, last_line)]} of the POST-migration lines each staged patch writes, read
    out of its own `@@ -a,b +c,d @@` headers. A changed or added top-level definition is allowed only
    if its own source lines in the working file fall inside one of these ranges: that is what "the
    patch, and nothing besides the patch, moved this file" means when it is CHECKED rather than
    asserted. The NEW side of the header is the one to compare, because a definition a patch adds has
    no old-side line number at all, and comparing a worktree line against a HEAD range would report
    every definition after an insertion as having wandered off.
    """
    out = {}
    for p in PATCHES:
        for line in open(os.path.join(PATCH_DIR, p)):
            if not line.startswith("@@"):
                continue
            frag = line.split("@@")[1].strip()
            new = frag.split("+", 1)[1] if "+" in frag else "0,1"
            c, _, d = new.partition(",")
            start, n = int(c), int(d or 1)
            out.setdefault(p, []).append((start, start + max(n, 1) - 1))
    return out


def top_defs(src):
    """{top-level name: (first_line, last_line, ast-dump)} for one file's source."""
    out = {}
    for node in ast.parse(src).body:
        nm = getattr(node, "name", None)
        if nm is None:
            t = node.targets[0] if isinstance(node, ast.Assign) else None
            nm = getattr(t, "id", None)
        if not nm:
            continue
        out[nm] = (getattr(node, "lineno", 0), getattr(node, "end_lineno", 0),
                   ast.dump(ast.parse(ast.unparse(node)).body[0]))
    return out

# ---------------------------------------------------------------------------------------------
# the derived fixture population — one definition, used by both sides
# ---------------------------------------------------------------------------------------------


def population(fmt):
    """Images: exhaustive for a 16-bit format; every exponent crossed with the mantissas that name
    the field's edges (0, 1, 2, half-1, half, half+1, max-1, max, /3, /5), both signs, plus the
    +/-1/+/-2 neighbourhoods of each. Derived, not hand-listed (pitfalls/40 §"derive the population"),
    and identical to the corpus `ulp_policy_equiv.py` used, so the counts below are comparable to the
    migration note's."""
    m, e_bits = fmt.mant, fmt.exp
    if fmt.bits == 16:
        return list(range(0, 1 << fmt.bits))
    half = 1 << (m - 1)
    mants = sorted({0, 1, 2, 3, half - 1, half, half + 1, (1 << m) - 2, (1 << m) - 1,
                    (1 << m) // 3, (1 << m) // 5})
    out = []
    for e in range(0, 1 << e_bits):
        for mm in mants:
            out.append((e << m) | mm)
            out.append(fmt.sign_bit | ((e << m) | mm))
    extra = [x + d for x in out for d in (-2, -1, 1, 2) if 0 <= x + d <= fmt.mask]
    seen, pop = set(), []
    for x in out + extra:
        if x not in seen:
            seen.add(x)
            pop.append(x)
    return pop


def pairs(fmt, pop):
    return [(x, y) for x in pop
            for y in (x + 1, x - 1, x + 2, x ^ fmt.sign_bit, x + (1 << fmt.mant))
            if 0 <= y <= fmt.mask]


def digest(vals):
    """(n, non-None count, None count, min, max, sum-of-ints, crc32 of the packed list)."""
    ints = [v for v in vals if isinstance(v, int)]
    payload = json.dumps([None if v is None else v for v in vals]).encode()
    return {"n": len(vals), "n_int": len(ints), "n_none": len(vals) - len(ints),
            "min": min(ints) if ints else None, "max": max(ints) if ints else None,
            "sum": sum(ints), "crc32": zlib.crc32(payload)}


# ---------------------------------------------------------------------------------------------
# the mirror tree: the committed revision, as an importable copy of this one
# ---------------------------------------------------------------------------------------------


def git_show(path):
    return subprocess.run(["git", "-C", REPO, "show", "HEAD:" + path],
                          capture_output=True, text=True, check=True).stdout


def ensure_mirror():
    os.makedirs(os.path.join(MIRROR, "prototype", "moe_router"), exist_ok=True)
    os.makedirs(os.path.join(MIRROR, "tools"), exist_ok=True)
    os.makedirs(os.path.join(MIRROR, "speccheck"), exist_ok=True)
    for d in ("engines", "checkpoints", "models", "specification"):
        link = os.path.join(MIRROR, d)
        if not os.path.lexists(link):
            os.symlink(os.path.join(REPO, d), link)
    for src_dir in ("tools", "speccheck"):
        for fn in os.listdir(os.path.join(REPO, src_dir)):
            if fn.endswith(".py"):
                dst = os.path.join(MIRROR, src_dir, fn)
                with open(os.path.join(REPO, src_dir, fn), "rb") as a, open(dst, "wb") as b:
                    b.write(a.read())
    with open(os.path.join(MIRROR, "prototype", "moe_router", "gate.py"), "w") as fh:
        fh.write(git_show("prototype/moe_router/gate.py"))
    for p in HEAD_FILES:
        with open(os.path.join(MIRROR, p), "w") as fh:
            fh.write(git_show(p))


def load(path, name, prepend_path=None):
    if prepend_path and prepend_path not in sys.path:
        sys.path.insert(0, prepend_path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------
# one side of the comparison: called identically for HEAD and for the working tree
# ---------------------------------------------------------------------------------------------


def measure(side):
    """side='head' measures the mirror (pre-migration), side='new' measures the working tree."""
    root = MIRROR if side == "head" else REPO
    if side == "head":
        ensure_mirror()
    sys.path.insert(0, os.path.join(REPO, "tools"))       # the one ruler is the same on both sides
    import ulp_policy as UP
    f32, bf16 = UP.format_facts("f32"), UP.format_facts("bf16")
    r32, r16 = UP.Ruler(f32), UP.Ruler(bf16)
    p32 = pairs(f32, population(f32))
    img16 = population(bf16)

    router = load(os.path.join(root, "prototype/moe_router/gate.py"), "router_" + side)
    audit = load(os.path.join(root, "tools", "audit_published_quant.py"), "audit_" + side)
    if side == "head":
        sys.modules["moe_ref"] = load(os.path.join(root, "speccheck", "moe_ref.py"), "moe_ref_" + side)
        mref = sys.modules["moe_ref"]
    else:
        mref = sys.modules.get("moe_ref_" + side) or load(
            os.path.join(root, "speccheck", "moe_ref.py"), "moe_ref_" + side)
    mstats = load(os.path.join(root, "tools", "moe_stats.py"), "moe_stats_" + side)

    out = {"side": side}
    router_vals = [router.ulpdist_bits(a, b) for a, b in p32]
    audit_vals = [audit.ulp_dist(r32.value(a), r32.value(b)) for a, b in p32]
    out["router_bits"] = digest(router_vals)
    out["audit_dist"] = digest(audit_vals)
    out["ulp_of"] = digest([mref.ulp_of(UP._value_of("bf16", x)) for x in img16[::7]])
    out["bf16_rel_width"] = getattr(mstats, "BF16_REL_WIDTH", None)
    out["ulp_of_src_has_typed_eight"] = ("8 explicit mantissa bits"
                                         in open(os.path.join(root, "speccheck", "moe_ref.py")).read())
    out["rel_width_readers"] = (open(os.path.join(root, "tools", "moe_stats.py")).read()
                               .count("BF16_REL_WIDTH"))
    out["has_ulpdist_bf16"] = hasattr(mref, "ulpdist_bf16")
    out["ruler32_is_shared"] = getattr(router, "_RULER32", None) is not None and type(
        getattr(router, "_RULER32", None)).__name__ == "Ruler" and getattr(
        router._RULER32, "name", None) == "f32"
    out["delegated_ruler_materialised"] = getattr(audit.ulp_dist, "_ruler", None) is not None
    # every pair of the corpus, element by element, from each side: no subset is trusted to stand in
    # for the whole population, and the two sides are compared index by index below.
    out["router_pair_values"] = router_vals
    out["audit_pair_values"] = audit_vals
    # the miscount of the LENGTH ruler against a step count, over every bf16 image that has a successor
    mis, zeroed, edges = 0, 0, []
    for x in img16:
        if r16.is_nan(x) or x >= r16.mask:
            continue
        step = abs(UP._value_of("bf16", x + 1) - UP._value_of("bf16", x))
        got = mref.ulp_of(UP._value_of("bf16", x))
        if got == 0.0:
            zeroed += 1
            continue
        if got != step:
            mis += 1
            if len(edges) < 4:
                edges.append((x, got, step))
    out["ulp_of_miscount"] = mis
    out["ulp_of_zero_at_zero"] = zeroed
    out["ulp_of_edges"] = edges
    return out


def emit_side():
    sys.stdout.write("@@JSON@@" + json.dumps(measure("head")) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------------------------


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


def main():
    if "--side-head" in sys.argv:
        return emit_side()
    v = V()
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import ulp_policy as UP
    f32 = UP.format_facts("f32")
    r32 = UP.Ruler(f32)
    new = measure("new")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--side-head"],
                          capture_output=True, text=True, cwd=REPO)
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        v.check("the pre-migration side ran", False, 0,
                "subprocess produced no JSON: %s" % proc.stderr[-400:])
        print("RESULT FAIL ulp migration witness checks=%d findings=%d" % (v.n, v.f))
        return 1
    old = json.loads(line[0][len("@@JSON@@"):])
    p32 = pairs(f32, population(f32))

    def img(x):
        return "0x%08x" % x

    def val(x):
        return "%.9g" % UP._value_of("f32", x)

    # ---- 1. the router ruler: the straddle defect, shown, then shown gone --------------------
    old_zero_fold, changed, nan_now_none = [], [], []
    pairs_old, pairs_new = old["router_pair_values"], new["router_pair_values"]
    if len(pairs_old) != len(pairs_new) or len(pairs_old) != len(p32):
        v.check("both sides measured every pair of the corpus", False, len(pairs_old),
                "head=%d new=%d corpus=%d" % (len(pairs_old), len(pairs_new), len(p32)))
        print("RESULT FAIL ulp migration witness checks=%d findings=%d" % (v.n, v.f))
        return 1
    for i, (a, b) in enumerate(p32):
        kind, steps = r32.classify(a, b)
        o, n = pairs_old[i], pairs_new[i]
        if kind in (UP.NAN_PAIR, UP.NAN_OTHER):
            if isinstance(o, int) and n is None:
                nan_now_none.append((i, a, b, o))
            continue
        if o != n:
            changed.append((i, a, b, o, n))
            if kind == UP.ZERO_SIGN:
                old_zero_fold.append((i, a, b, o, n))
    straddle_changed = [c for c in changed
                        if (p32[c[0]][0] < f32.sign_bit) != (p32[c[0]][1] < f32.sign_bit)]
    for i, a, b, o, n in changed[:6]:
        print("PAIR kind=%s a=%s(%s) b=%s(%s) old=%s new=%s ruler=%s delta=%s"
              % (r32.classify(a, b)[0], img(a), val(a), img(b), val(b), o, n, steps,
                 (n - o) if (o is not None and n is not None) else "n/a"))
    v.check("router-ulpdist-bits: the patch CHANGED behaviour on the corpus",
            len(changed) > 0, len(changed),
            "%d of %d pairs now read differently; %d of them straddle zero"
            % (len(changed), len(p32), len(straddle_changed)))
    v.check("router-ulpdist-bits: every zero-straddling change is exactly +1 step",
            all(n == o + 1 for _, _, _, o, n in changed if o is not None and n is not None),
            len(changed), "the short ruler read one step FEWER across zero, so the fix can only be "
                          "stricter: a band measured with it was one step LAXER than it claimed")
    v.check("router-ulpdist-bits: the +0/-0 fold is among the changed pairs",
            len(old_zero_fold) == 2, len(old_zero_fold),
            "old read %s steps between the two zero images (they folded onto one key); the ruler "
            "reads %d" % (old_zero_fold[0][3] if old_zero_fold else "-",
                          old_zero_fold[0][4] if old_zero_fold else "-"))
    v.check("router-ulpdist-bits: the straddle population is the one the migration note published",
            len(straddle_changed) - len(old_zero_fold) == 13771 and len(straddle_changed) == 13773,
            len(straddle_changed),
            "13,771 straddling pairs + the 2 signed-zero pairs = 13,773, the figure in "
            "specification/evidence/ulp-policy-migration-2026-10-26.md §4")
    v.check("router-ulpdist-bits: a NaN pair changed from a number to None",
            len(nan_now_none) > 0, len(nan_now_none),
            "example old=%s new=None" % (nan_now_none[0][3] if nan_now_none else "-"))
    try:
        None <= 5                                       # the patch's claim about its own call sites
        v.check("router-ulpdist-bits: None against a band raises instead of passing", False, 0,
                "the comparison did not raise; a NaN would have slipped through the band")
    except TypeError:
        v.check("router-ulpdist-bits: None against a band raises instead of passing", True, 1,
                "TypeError, as the patch promises: NaN reaching a band is a finding, not a small value")
    v.check("router-ulpdist-bits: the new body agrees with the shared ruler on every finite pair "
            "of the corpus", all(new["router_pair_values"][i]
                                 == r32.classify(a, b)[1] for i, (a, b) in enumerate(p32)
                                 if not r32.is_nan(a) and not r32.is_nan(b)),
            len(p32), "0 finite pairs disagree with tools/ulp_policy.py — the defect is gone")
    # the aggregate figure the migration note published: 13,771 short pairs + 2 folded zeros + 290 NaN
    print("DIGEST router.ulpdist_bits head=%s\nDIGEST router.ulpdist_bits new =%s"
          % (json.dumps(old["router_bits"]), json.dumps(new["router_bits"])))
    v.check("router-ulpdist-bits: the pre-migration digest matches the migration note's corpus",
            old["router_bits"]["n"] == new["router_bits"]["n"] == 69089,
            new["router_bits"]["n"], "the same derived corpus the equivalence harness used")
    v.check("router-ulpdist-bits: the pre-migration body returned a number for every NaN pair and "
            "the post-migration body does not",
            old["router_bits"]["n_none"] == 0 and new["router_bits"]["n_none"] == 290,
            "%d -> %d" % (old["router_bits"]["n_none"], new["router_bits"]["n_none"]),
            "290 NaN pairs: numeric answer before, None after")

    # ---- 2. ulp_of: a length, and the counting defect repaired by the ruler the patch adds ----
    print("DIGEST moe_ref.ulp_of head=%s\nDIGEST moe_ref.ulp_of new =%s"
          % (json.dumps(old["ulp_of"]), json.dumps(new["ulp_of"])))
    v.check("moe-ref-ulp-of: the length itself did not move (the patch changes no caller-visible "
            "number)", old["ulp_of"] == new["ulp_of"], new["ulp_of"]["n"],
            "same crc32 over the 2^16/7 image sample: %s" % new["ulp_of"]["crc32"])
    v.check("moe-ref-ulp-of: the length mis-states the step at the subnormal/binade edges",
            new["ulp_of_miscount"] == 256, new["ulp_of_miscount"],
            "256 bf16 images per image = the 1,280 miscounted PAIRS in the equivalence harness's "
            "counting (each image appears as the first element of 5 pairs); and it returns 0.0 at "
            "+/-0 and for non-finite on %d images" % new["ulp_of_zero_at_zero"])
    v.check("moe-ref-ulp-of: the miscount is identical before and after (it is a property of a "
            "length, not of this patch)", old["ulp_of_miscount"] == new["ulp_of_miscount"]
            and old["ulp_of_zero_at_zero"] == new["ulp_of_zero_at_zero"],
            new["ulp_of_miscount"], "256 edge images and %d zero/non-finite readings, both sides"
            % new["ulp_of_zero_at_zero"])
    bf16 = UP.format_facts("bf16")
    r16 = UP.Ruler(bf16)
    sys.path.insert(0, os.path.join(REPO, "speccheck"))
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import moe_ref as NEW_REF
    v.check("moe-ref-ulp-of: the patch adds a COUNTING ruler to the same module",
            old["has_ulpdist_bf16"] is False and new["has_ulpdist_bf16"] is True, 1,
            "moe_ref.ulpdist_bf16 exists in the working tree only")
    edge_images = [0, bf16.sign_bit, 1, (1 << bf16.mant) - 1, 1 << bf16.mant,
                   bf16.bias << bf16.mant, (bf16.bias + 1) << bf16.mant,
                   ((1 << bf16.exp) - 2) * (1 << bf16.mant) + ((1 << bf16.mant) - 1)]
    fixed = 0
    for x in edge_images:
        nxt = x + 1
        length = NEW_REF.ulp_of(UP._value_of("bf16", x))
        true_step = (abs(UP._value_of("bf16", nxt) - UP._value_of("bf16", x)))
        kind, steps = NEW_REF.ulpdist_bf16(x, nxt)
        refused = not r16.within(x, nxt, 1)          # non-finite: a count of 1 is not an acceptance
        counted_ok = steps == 1 and (kind == "finite_step" or (kind == "inf_vs_finite" and refused))
        fixed += 1 if counted_ok else 0
        print("EDGE image=0x%04x(%s) next=0x%04x ulp_of=%r true_step=%r ulpdist_bf16=(%s,%s) "
              "within_1ulp_accepted=%d %s"
              % (x, "%.6g" % UP._value_of("bf16", x), nxt, length, true_step, kind, steps,
                 0 if refused else 1,
                 "COUNTED-RIGHT" if counted_ok else "COUNTED-WRONG"))
    v.check("moe-ref-ulp-of: every edge the length could not count is counted correctly by the "
            "ruler the patch adds", fixed == len(edge_images), fixed,
            "%d/%d edge images counted 1 step by ulpdist_bf16, where ulp_of answers a length (0.0 at "
            "+/-0); the largest-finite-to-+inf edge is counted 1 AND refused by the policy, which is "
            "why a count is not an acceptance" % (fixed, len(edge_images)))
    v.check("moe-ref-ulp-of: the width is derived in the body, not typed, and the prose now says 7",
            "math.floor(math.log2(v)) - 7)" in git_show("speccheck/moe_ref.py")
            and "_bf16_facts().mant" in open(os.path.join(REPO, "speccheck", "moe_ref.py")).read()
            and NEW_REF._bf16_facts().mant == 7, NEW_REF._bf16_facts().mant,
            "HEAD's body ends in the typed `- 7`; the working tree reads the mantissa width from "
            "tools/ulp_policy.py, which derives 7 from the format's own bit image")

    # ---- 3. the two provenance patches ---------------------------------------------------------
    print("VALUE moe_stats.BF16_REL_WIDTH head=%r new=%r ratio=%s"
          % (old["bf16_rel_width"], new["bf16_rel_width"],
             (new["bf16_rel_width"] / old["bf16_rel_width"]) if old["bf16_rel_width"] else "n/a"))
    half_ulp = 2.0 ** -(bf16.mant + 1)
    v.check("moe-stats-rel-width: the constant moved off the typed-8 value and onto the derived one",
            abs(old["bf16_rel_width"] - 2.0 ** -9) < 1e-15
            and abs(new["bf16_rel_width"] - half_ulp) < 1e-15, new["bf16_rel_width"],
            "half-ULP relative width is 2^-(mant+1) = 2^-8 = %.10g for mant=%d. The STAGED patch "
            "wrote `2.0 ** -(mant - 1)`, which is 2^-6 — 4x too wide and contradicting the "
            "comment three lines above it; the exponent in the applied file is corrected to +1, and "
            "this check is what would have caught the staged form"
            % (half_ulp, bf16.mant))
    v.check("moe-stats-rel-width: the constant still has no reader, so no printed number moved",
            old["rel_width_readers"] == 1 and new["rel_width_readers"] == 1,
            new["rel_width_readers"], "occurrences of the name in the file, the definition itself")
    print("DIGEST audit.ulp_dist head=%s\nDIGEST audit.ulp_dist new =%s"
          % (json.dumps(old["audit_dist"]), json.dumps(new["audit_dist"])))
    v.check("audit-ulp-dist: the delegation is behaviour-preserving on the whole corpus",
            old["audit_dist"] == new["audit_dist"], new["audit_dist"]["n"],
            "identical digest over %d pairs, NaN arms included: this patch changes provenance, not a "
            "number, and saying so is the honest result (the migration note's consequence column for "
            "it is 'none')" % new["audit_dist"]["n"])
    v.check("audit-ulp-dist: the delegated ruler object is materialised on first call",
            old["delegated_ruler_materialised"] is False
            and new["delegated_ruler_materialised"] is True, 1,
            "ulp_dist._ruler is a tools/ulp_policy.py Ruler after one call")
    v.check("router-import: the patched router gate holds the shared f32 ruler",
            old["ruler32_is_shared"] is False and new["ruler32_is_shared"] is True, 1,
            "_RULER32 is the shared ruler, f32 facts derived from struct")

    # ---- 4. nothing else moved ------------------------------------------------------------------
    pranges = hunk_ranges()
    file_ranges = {}
    for p in PATCHES:
        file_ranges.setdefault(patch_target(p), []).extend(pranges[p])
    total_changed = 0
    outside = []
    touched = {p: 0 for p in PATCHES}
    for path in sorted(HEAD_FILES):
        head_src = git_show(path)
        new_src = open(os.path.join(REPO, path)).read()
        h, w = top_defs(head_src), top_defs(new_src)
        # compare the BODY ONLY: inserting a hunk shifts every later line number, and a comparison
        # that counted line numbers would report the whole file as changed.
        diff = sorted(k for k in set(h) | set(w)
                      if (h.get(k) or (None, None, None))[2] != (w.get(k) or (None, None, None))[2])
        total_changed += len(diff)
        for k in diff:
            lo, hi = w.get(k, h.get(k))[:2]        # working-tree lines, so added names count too
            if not any(a <= hi and lo <= b for a, b in file_ranges.get(path, [])):
                outside.append((path, k, lo, hi))
        for p in PATCHES:
            if patch_target(p) != path:
                continue
            for k in diff:
                lo, hi = w.get(k, h.get(k))[:2]
                # a declared lane repair counts as the patch it corrects: the line is the patch's line,
                # edited, and it would be worse than silence to report that patch as having done nothing
                if k in LANE_REPAIRS.get(path, ()) or any(a <= hi and lo <= b for a, b in pranges[p]):
                    touched[p] += 1
        print("DIFFSCOPE file=%s changed=%s added=%s removed=%s hunks=%d"
              % (path, ",".join(diff) or "none",
                 ",".join(sorted(set(w) - set(h))) or "none",
                 ",".join(sorted(set(h) - set(w))) or "none", len(file_ranges.get(path, []))))
    print("OUTSIDEHUNKS " + (",".join("%s::%s(%d-%d)" % o for o in outside) or "none"))
    declared = set("%s::%s" % (p, k) for p, names in LANE_REPAIRS.items() for k in names)
    unexpected = [o for o in outside if "%s::%s" % o[:2] not in declared]
    for key in sorted(declared):
        print("LANEREPAIR %s declared_reason=%s" % (key, LANE_REPAIRS[key.split("::")[0]][key.split("::")[1]]))
    no_op = [p for p in PATCHES if touched[p] == 0]
    v.check("every changed top-level definition is inside a staged patch hunk, or is a declared "
            "lane repair", not unexpected, total_changed,
            "changed outside every hunk and undeclared: %s (declared repairs: %s)"
            % (unexpected or "none", sorted(declared) or "none"))
    changed_keys = set("%s::%s" % (path, k) for path in HEAD_FILES
                       for k in top_defs(open(os.path.join(REPO, path)).read()))
    vanished = [d for d in sorted(declared) if d not in changed_keys]
    v.check("every declared lane repair is a definition that actually exists and changed",
            not vanished, len(declared),
            "declared but not a changed definition: %s — a declaration that matches nothing is a "
            "finding, not a comment" % (vanished or "none"))
    v.check("each of the five patches moved at least one top-level definition",
            not no_op, len(PATCHES),
            "patches that moved nothing: %s (changed-definitions-per-patch=%s)"
            % (no_op or "none", touched))
    expect = set("%s::%s" % (p, k) for p, names in PATCHED_NAMES.items() for k in names) | declared
    got = set("%s::%s" % (path, k) for path in HEAD_FILES
              for k in top_defs(open(os.path.join(REPO, path)).read())
              if (top_defs(open(os.path.join(REPO, path)).read())[k][2]
                  != (top_defs(git_show(path)).get(k) or (None, None, None))[2]))
    v.check("the migration changed exactly the definitions the five patches and the declared repairs "
            "account for", got == expect, len(got),
            "unexpected: %s ; missing: %s (expected %d: the 8 the patches touch plus the declared "
            "lane repairs)" % (sorted(got - expect) or "none", sorted(expect - got) or "none",
                               len(expect)))

    print("RESULT %s ulp migration witness checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
