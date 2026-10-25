#!/usr/bin/env python3
"""ulp_policy.py — ONE ULP ruler, and the ONE policy for accepting ±1 ULP (DECISIONS.md D-17).

Why a file and not twenty lines. `DECISIONS.md` D-17 records the user's request — stop escalating
isolated last-bit disagreements for approval, accept ±1 from the target **only at Q8 and above; for
Q4 that could be fatal** — and it records that this repo carries two incompatible rulers:

| ruler | what "one ULP" means | where it lives |
|---|---|---|
| float quantum | `2^(exp-7)`, a *value* at |v|, bf16 width typed in prose | `speccheck/moe_ref.py` §ulp_of |
| bit-image adjacency | an *integer count of representable steps* between two bit patterns after sign-magnitude ordering | `prototype/moe_router/gate.py` §ulpdist_bits, §bf16_ulpdist; `tools/audit_published_quant.py` §ulp_dist (f32) |

**The ruler is the bit-image one.** It is the only one of the two that is a *count*, so it is the only
one that can answer "is this element within 1 ULP of that element" at a power of two, at a subnormal,
at ±0, or across zero: the float quantum is a length, and a length cannot say how many representable
steps lie between 1.0 and 2.0 (it answers "1"; the image answers 2^mantissa). Every format fact is
derived at run time from a real source — `struct`'s own round-trip through the format for f16/f32/f64,
`torch` for bf16, cross-checked against `numpy.finfo`/`torch.finfo` metadata when that module is
importable — and the tool **refuses to run** rather than fall back to a typed `8`. A typed width is the
exact error class `tools/ggml_block_geom.py` exists to eliminate, and `pitfalls/30` §A documents what a
wrong sign transform in exactly this comparator does: the gate passes vacuously.

The policy, exactly as D-17 states it — an acceptance requires ALL of:
  (a) the field is in the lane's declared field set;
  (b) the tensor's OWN ggml type, read from the pinned file with `tools/gguf_reader.py`, is ≥ Q8 —
      bits-per-weight derived from the pinned `ggml-common.h` structs by `tools/ggml_block_geom.py`
      and cross-checked against `tools/gguf_width_table.py` as the independent ruler. The Q4 veto is
      the user's, is structural, and fires **even at 1 ULP**;
  (c) the divergences are ISOLATED: some elements at exactly 1 ULP, **zero** elements at ≥ 2 ULP, and
      a strict majority of the judged elements bit-identical;
  (d) the census is printed — element counts, fields, the ULP histogram — and the lane carries an
      attribution, computed from data, for why the divergence is explained rather than merely small.
Outside the declared field set exactness is required and the count must stay zero.

Why (c) is written the way it is, and why the isolation test is a *rejection*: a wrong rounding mode or
an off-by-one in a reduction produces ±1 on **every** element. That is a different function, not
rounding noise, and a naive "accept ±1" would bless it. `--selftest` therefore builds the
systematic-shift fixture and requires the refusal — an acceptance rule whose tripwire was never
watched to fire has been hoped for, not tested (`pitfalls/40` §"a negative control with an empty
must-catch set is not a control", §"count reachability, not just checks").

Interpreter of record: `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python` (CPU-only
torch and numpy) for `--selftest` and for anything judging bf16. The f32/f16 ruler and the width veto
also work under plain `python3`, because those derivations need only `struct`. This file never opens a
device context and prints no timing of any kind.

Usage:
  ulp_policy.py --selftest                  every arm, witnessed, with branch reachability
  ulp_policy.py --formats                   derived format facts and where each came from
  ulp_policy.py --width <gguf> [--tensor T] the derived width class of tensors in a real file

Evidence for this file and the staged (unapplied) migration of the four older rulers lives in
`specification/evidence/ulp-policy-migration-2026-10-26.md`, with its runs in the `-selftest`,
`-equiv` and `-staged-patches` logs beside it.
"""
import argparse
import math
import os
import struct
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VENV_PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"
PINNED_GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

# ---------------------------------------------------------------------------------------------
# 1. format facts, derived at run time from a real source
# ---------------------------------------------------------------------------------------------


class FormatSourceError(Exception):
    """Raised when a format's width cannot be derived. There is no fallback and no default: a typed
    mantissa width is the failure this file was written to make impossible."""


# struct's own format chars for the CPU-native IEEE binary formats; bf16 has none and is derived from
# torch, the only installed source for it (numpy 2.5 has no bfloat16 dtype).
STRUCT_FMT = {"f16": "e", "f32": "f", "f64": "d"}
TORCH_DTYPE = {"bf16": "bfloat16"}
NUMPY_DTYPE = {"f16": "float16", "f32": "float32", "f64": "float64"}
# "Q8 and above" == at least 8 stored bits per weight. One place, one spelling, and every type's own
# bpw is derived (see WidthClass); this constant is the *threshold*, not a width.
Q8_MIN_BPW = 8.0


class Format(object):
    """Derived facts about one binary floating-point format.

    `mant` is the number of EXPLICIT mantissa bits — bf16: 7, not 8. 8 is the precision including the
    implicit bit, which is what `speccheck/moe_ref.py` §ulp_of's prose calls "8 explicit mantissa
    bits" while its formula correctly uses 7. The formula was right and the prose was the typed guess.
    """

    __slots__ = ("name", "bits", "mant", "exp", "bias", "smallest_subnormal", "smallest_normal",
                 "max_finite", "image_one", "sources", "crosscheck")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def sign_bit(self):
        return 1 << (self.bits - 1)

    @property
    def mask(self):
        return (1 << self.bits) - 1

    def line(self):
        return ("FORMAT name=%s bits=%d mant=%d exp=%d bias=%d subnormal_min=%.3e normal_min=%.3e "
                "max=%.4e derived_from=%s crosscheck=%s"
                % (self.name, self.bits, self.mant, self.exp, self.bias, self.smallest_subnormal,
                   self.smallest_normal, self.max_finite, ",".join(self.sources), self.crosscheck))


def _image_of(fmt, value):
    """The bit image of `value` in format `fmt`, from the format's own encoder."""
    if fmt in TORCH_DTYPE:
        import torch
        return int(torch.tensor([value], dtype=getattr(torch, TORCH_DTYPE[fmt]))
                   .view(torch.uint16).item())
    return int.from_bytes(struct.pack("<" + STRUCT_FMT[fmt], value), "little")


def _value_of(fmt, image):
    """The value whose image is `image`, from the format's own decoder."""
    if fmt in TORCH_DTYPE:
        import torch
        return float(torch.tensor([int(image)], dtype=torch.uint16)
                     .view(getattr(torch, TORCH_DTYPE[fmt])).item())
    return struct.unpack("<" + STRUCT_FMT[fmt],
                         int(image).to_bytes(struct.calcsize(STRUCT_FMT[fmt]), "little"))[0]


def _derive_bitimage(fmt):
    """(bits, mant, bias) out of the format's OWN arithmetic.

    image(1.0) = bias << mant and image(2.0) = (bias+1) << mant, so their difference is exactly
    2^mant: the width comes out of the format rather than out of a header or a typed constant.
    """
    if fmt not in STRUCT_FMT and fmt not in TORCH_DTYPE:
        raise FormatSourceError("no source for format %r; this tool knows %s and will not guess "
                                "another one" % (fmt, ",".join(sorted(list(STRUCT_FMT)
                                                                       + list(TORCH_DTYPE)))))
    try:
        i1, i2 = _image_of(fmt, 1.0), _image_of(fmt, 2.0)
    except ImportError as e:
        raise FormatSourceError("cannot derive %s: its source is not importable (%s). This tool "
                                "never substitutes a typed mantissa width; run it with %s"
                                % (fmt, e, VENV_PY))
    if i2 <= i1:
        raise FormatSourceError("derivation degenerate for %s: image(1.0)=%x image(2.0)=%x"
                                % (fmt, i1, i2))
    diff = i2 - i1
    mant = int(round(math.log2(diff)))
    if (1 << mant) != diff:
        raise FormatSourceError("derivation of %s produced a non-power-of-two image gap %d"
                                % (fmt, diff))
    bias = i1 >> mant
    nbytes = 2 if fmt in TORCH_DTYPE else struct.calcsize(STRUCT_FMT[fmt])
    bits = 8 * nbytes
    exp = bits - 1 - mant
    if not (0 < mant < bits and 0 < exp < bits):
        raise FormatSourceError("derived mant=%d exp=%d bits=%d is not a sign/exp/mant partition"
                                % (mant, exp, bits))
    return {"bits": bits, "mant": mant, "exp": exp, "bias": bias, "image_one": i1}


def _finfo_facts(fmt):
    """The second, independent source: library format metadata. None when the library is absent."""
    facts = {}
    src = []
    if fmt in TORCH_DTYPE:
        try:
            import torch
            fi = torch.finfo(getattr(torch, TORCH_DTYPE[fmt]))
        except ImportError:
            fi = None
        if fi is not None:
            facts = {"bits": int(fi.bits), "mant": int(round(-math.log2(fi.eps))),
                     "bias": 1 - int(round(math.log2(fi.tiny)))}
            src.append("torch.finfo")
    if fmt in NUMPY_DTYPE:
        try:
            import numpy as np
            fi = np.finfo(getattr(np, NUMPY_DTYPE[fmt]))
            facts = {"bits": int(fi.bits), "mant": int(fi.nmant), "bias": 1 - int(fi.minexp)}
            src.append("numpy.finfo")
        except ImportError:
            pass
    return (facts, "+".join(src)) if src else ({}, "absent")


# Named sabotage hooks, in the style of tools/gguf_width_table.py's injected stale Q5_K row: a guard
# is only credible when an injected fault is witnessed being caught. --selftest uses both.
_SABOTAGE_SOURCE_UNAVAILABLE = False


def format_facts(fmt, _inject_mant=None):
    """Facts for `fmt`, derived. `_inject_mant` exists ONLY for the selftest's sabotage (a typed
    width) and is validated against the second source, so an injected guess is caught rather than
    shipped. Nothing outside --selftest passes it."""
    if _SABOTAGE_SOURCE_UNAVAILABLE:
        raise FormatSourceError("no source for format %r (sabotage): refusing to run rather than "
                                "fall back to a typed width" % fmt)
    d = _derive_bitimage(fmt)
    srcs = ["bitimage(torch)" if fmt in TORCH_DTYPE else "bitimage(struct)"]
    finfo, xchk_src = _finfo_facts(fmt)
    if finfo:
        for k in ("bits", "mant", "bias"):
            if int(finfo[k]) != int(d[k]):
                raise FormatSourceError("%s: %s says %s=%s, the format's own bit image says %s=%s"
                                        % (fmt, xchk_src, k, finfo[k], k, d[k]))
    if _inject_mant is not None:
        typed = int(_inject_mant)
        srcs.append("SABOTAGE:typed-mant=%d" % typed)
        if not finfo:
            raise FormatSourceError("%s: an injected mantissa width cannot be checked without a "
                                    "second source; refusing instead of trusting the injection"
                                    % fmt)
        if typed != int(finfo["mant"]):
            raise FormatSourceError("%s: injected mantissa width %d contradicts %s (mant=%d) and "
                                    "the bit image (mant=%d)"
                                    % (fmt, typed, xchk_src, finfo["mant"], d["mant"]))
    sub = 2.0 ** (1 - d["bias"] - d["mant"])
    normal = 2.0 ** (1 - d["bias"])
    max_image = ((1 << d["exp"]) - 2) * (1 << d["mant"]) + ((1 << d["mant"]) - 1)
    return Format(name=fmt, bits=d["bits"], mant=d["mant"], exp=d["exp"], bias=d["bias"],
                  smallest_subnormal=sub, smallest_normal=normal,
                  max_finite=_value_of(fmt, max_image), image_one=d["image_one"],
                  sources=tuple(srcs), crosscheck=xchk_src)


# ---------------------------------------------------------------------------------------------
# 2. the ruler: distance in the sign-magnitude-ordered bit image
# ---------------------------------------------------------------------------------------------
# Every pair classifies into exactly ONE category. `steps` is the distance in the ordered image; it is
# None for the NaN arms, where "how many steps" has no answer this policy would act on.
EQUAL = "equal"
STEP = "finite_step"
ZERO_SIGN = "zero_sign_flip"
INF_FINITE = "inf_vs_finite"
INF_PAIR = "inf_vs_inf"
NAN_PAIR = "nan_payload"
NAN_OTHER = "nan_vs_other"
CATEGORIES = (EQUAL, STEP, ZERO_SIGN, INF_FINITE, INF_PAIR, NAN_PAIR, NAN_OTHER)
NONFINITE = (INF_FINITE, INF_PAIR, NAN_PAIR, NAN_OTHER)


class Ruler(object):
    """Distance between two bit images, in representable steps."""

    def __init__(self, fmt):
        self.format = fmt if isinstance(fmt, Format) else format_facts(fmt)
        self.name = self.format.name
        self.sign = self.format.sign_bit
        self.mask = self.format.mask
        self.mant = self.format.mant
        self.exp_mask = (1 << self.format.exp) - 1

    # -- classification -------------------------------------------------------------------------
    def order_key(self, bits):
        """Sign-magnitude image -> monotone unsigned key (IEEE totalOrder, which puts -0 below +0).

        This is `pitfalls/30` §A's corrected form: the test is on the BIT (`bits & sign`), never on
        `x < 0` of an unsigned view — which is never true, so every negative takes the positive branch
        and +v and -v collide in the key. That is how a ULP gate passes vacuously.
        """
        b = int(bits) & self.mask
        return (~b & self.mask) if (b & self.sign) else (b | self.sign)

    def is_nan(self, bits):
        b = int(bits) & self.mask
        return ((b >> self.mant) & self.exp_mask) == self.exp_mask \
            and (b & ((1 << self.mant) - 1)) != 0

    def is_inf(self, bits):
        b = int(bits) & self.mask
        return ((b >> self.mant) & self.exp_mask) == self.exp_mask \
            and (b & ((1 << self.mant) - 1)) == 0

    def is_zero(self, bits):
        return (int(bits) & self.mask) in (0, self.sign)

    def is_subnormal(self, bits):
        b = int(bits) & self.mask
        return (b & ((1 << self.mant) - 1)) != 0 and (b >> self.mant) == 0

    def classify(self, a, b):
        """(category, steps) for one pair of bit images. A total function: nothing raises, because a
        comparator that throws on a NaN is a comparator a caller will wrap in `except: pass`."""
        a, b = int(a) & self.mask, int(b) & self.mask
        if a == b:
            return EQUAL, 0
        na, nb = self.is_nan(a), self.is_nan(b)
        if na and nb:
            return NAN_PAIR, None                  # different payloads: not a rounding step
        if na or nb:
            return NAN_OTHER, None                 # NaN is within 0 ULP of nothing
        steps = abs(self.order_key(a) - self.order_key(b))
        ia, ib = self.is_inf(a), self.is_inf(b)
        if ia and ib:
            return INF_PAIR, steps
        if ia or ib:
            return INF_FINITE, steps               # an overflow reads 1 step from the largest finite
        if self.is_zero(a) and self.is_zero(b):
            return ZERO_SIGN, steps                # +0 and -0 are adjacent in the ordered image
        return STEP, steps

    def within(self, a, b, ulps=1):
        kind, steps = self.classify(a, b)
        return kind == EQUAL or (kind == STEP and steps <= ulps)

    def value(self, bits):
        return _value_of(self.name, bits)

    def census(self, pairs, limit_indices=6):
        c = Census(self, limit_indices)
        c.consume(pairs)
        return c

    def dead_branch_vacuity(self, pairs):
        """How many pairs the `x < 0`-on-an-unsigned-array form would call identical.

        Returns None when numpy is absent, so the witness reports itself unreachable rather than
        silently passing. The form is `pitfalls/30` §A's verbatim wrong line; on unsigned images the
        condition is never true, so +v and -v collide and a sign flip reads as a zero difference.
        """
        try:
            import numpy as np
        except ImportError:
            return None
        dt = np.uint16 if self.format.bits == 16 else np.uint32
        a = np.array([int(x) for x, _ in pairs], dtype=dt)
        b = np.array([int(y) for _, y in pairs], dtype=dt)
        sign = dt(self.sign)
        # the wrong form, as written in the pitfall: on an UNSIGNED array `x < 0` is never true, so
        # every element takes the positive branch and +v / -v collide in the key
        naive_a = np.where(a < 0, ~a, a | sign)
        naive_b = np.where(b < 0, ~b, b | sign)
        return int(np.count_nonzero(naive_a != naive_b))


class Census(object):
    """Counts per category, the ULP histogram, and where the divergences sit.

    This is the artifact D-17 guardrail 5 demands: an acceptance that is not counted is an acceptance
    nobody can re-check.
    """

    def __init__(self, ruler, limit_indices=6):
        self.ruler = ruler
        self.judged = 0
        self.cats = Counter()
        self.steps_hist = Counter()
        self.first_idx = {}
        self.limit = limit_indices
        self.subnormal_diverging = 0

    def consume(self, pairs):
        for idx, a, b in pairs:
            self.add(idx, a, b)

    def add(self, idx, a, b):
        self.judged += 1
        kind, steps = self.ruler.classify(a, b)
        self.cats[kind] += 1
        if kind != EQUAL:
            self.first_idx.setdefault(kind, [])
            if len(self.first_idx[kind]) < self.limit:
                self.first_idx[kind].append(idx)
            if steps is not None:
                self.steps_hist[steps] += 1
            if self.ruler.is_subnormal(a) or self.ruler.is_subnormal(b):
                self.subnormal_diverging += 1

    @property
    def diverging(self):
        return self.judged - self.cats[EQUAL]

    def n_at(self, steps):
        return self.steps_hist[steps]

    def n_ge(self, steps):
        return sum(v for k, v in self.steps_hist.items() if k >= steps)

    @property
    def max_steps(self):
        return max(self.steps_hist) if self.steps_hist else 0

    def nonfinite(self):
        return sum(self.cats[k] for k in NONFINITE)

    def count(self, cat):
        return self.cats[cat]

    def hist(self):
        return " ".join("%d:%d" % (k, v) for k, v in sorted(self.steps_hist.items())) or "none"

    def where(self):
        return " ".join("%s@%s" % (k, ",".join(str(i) for i in v))
                        for k, v in sorted(self.first_idx.items())) or "none"

    def line(self, field, extra=""):
        return ("CENSUS format=%s field=%s judged=%d identical=%d differing=%d "
                "categories[%s] ulp_histogram[%s] max_ulp=%d subnormal_diverging=%d%s"
                % (self.ruler.name, field, self.judged, self.cats[EQUAL], self.diverging,
                   " ".join("%s=%d" % (k, self.cats[k]) for k in CATEGORIES if self.cats[k]),
                   self.hist(), self.max_steps, self.subnormal_diverging,
                   (" " + extra) if extra else ""))


# ---------------------------------------------------------------------------------------------
# 3. the width class — read from the artifact, never from a string
# ---------------------------------------------------------------------------------------------


class WidthError(Exception):
    pass


class WidthClass(object):
    """A stored width class, from a ggml type and the pinned geometry.

    `bpw` comes from `tools/ggml_block_geom.py` (read out of the pinned `ggml-common.h` structs) and
    is cross-checked against `tools/gguf_width_table.py`'s independently transcribed block table —
    two rulers, because a hand-copied width is exactly the bug that file exists to catch.
    `at_least_q8` is D-17's width test; below it the ±1 acceptance is refused **even at 1 ULP**.
    """

    __slots__ = ("tensor", "type_name", "type_id", "bpw", "bpw_independent", "gguf", "at_least_q8",
                 "n_elts", "shape", "ruler_agreement")

    def __init__(self, type_name, type_id, label, gguf_label, geom, table,
                 n_elts=0, shape=()):
        self.type_name = type_name
        self.type_id = type_id
        self.tensor = label
        self.gguf = gguf_label
        self.n_elts = n_elts
        self.shape = shape
        try:
            self.bpw = float(geom.bpw_of(type_name))
        except Exception as e:
            raise WidthError("ggml_block_geom cannot derive %s (%s): refusing to guess a width"
                             % (type_name, e))
        row = table.get(type_name)
        if row is None:
            raise WidthError("gguf_width_table has no row for %s: an unmapped type is a failure, "
                             "not a 0-byte row" % type_name)
        self.bpw_independent = 8.0 * row[1] / row[0]
        if abs(self.bpw_independent - self.bpw) > 1e-9:
            raise WidthError("two width rulers disagree for %s: ggml_block_geom says %.6f bpw, "
                             "gguf_width_table says %.6f bpw"
                             % (type_name, self.bpw, self.bpw_independent))
        self.ruler_agreement = "ggml_block_geom==gguf_width_table"
        self.at_least_q8 = self.bpw >= Q8_MIN_BPW

    def line(self):
        return ("WIDTH tensor=%s ggml_type=%s(id=%s) bpw=%.4f bpw_independent=%.4f q8_or_above=%d "
                "rule=bpw>=%.1f source=%s"
                % (self.tensor, self.type_name, self.type_id, self.bpw, self.bpw_independent,
                   1 if self.at_least_q8 else 0, Q8_MIN_BPW, self.gguf))

    @staticmethod
    def _geom_table():
        import ggml_block_geom as GBG
        import gguf_width_table as GWT
        return GBG, GWT.BLK

    @classmethod
    def from_type(cls, type_name, type_id=None):
        """Width of a ggml TYPE, from the pinned header (no file needed). Honest about it: `tensor`
        says `type-derived`."""
        geom, table = cls._geom_table()
        return cls(type_name, type_id if type_id is not None else "?",
                   "<type-derived:%s>" % type_name, "pinned ggml-common.h", geom, table)

    @classmethod
    def from_gguf(cls, gguf_path, tensor_name, _parsed=None):
        """Width of one real tensor, from its OWN ggml type in a real file (D-17 guardrail b)."""
        if not os.path.exists(gguf_path):
            raise WidthError("no pinned file at %s: the width class has to be read from an artifact, "
                             "so there is nothing to fall back to" % gguf_path)
        geom, table = cls._geom_table()
        g = _parsed if _parsed is not None else _open_gguf(gguf_path)
        t = g.by_name.get(tensor_name)
        if t is None:
            raise WidthError("tensor %r is not in %s (%d tensors parsed)"
                             % (tensor_name, os.path.basename(gguf_path), len(g.tensors)))
        return cls(t.type_name, t.type_id, t.name, os.path.basename(gguf_path), geom, table,
                   t.n_elts, t.shape_hf)

    @classmethod
    def first_of_type(cls, gguf_path, type_name, _parsed=None):
        """First tensor in a real file whose own ggml type is `type_name` — how a selftest builds a
        Q4_K / Q6_K / Q8_0 fixture without typing a width anywhere."""
        if not os.path.exists(gguf_path):
            raise WidthError("no pinned file at %s" % gguf_path)
        geom, table = cls._geom_table()
        g = _parsed if _parsed is not None else _open_gguf(gguf_path)
        hit = [t for t in g.tensors if t.type_name == type_name]
        if not hit:
            raise WidthError("no tensor of type %s in %s: a veto test with no fixture tensor cannot "
                             "fail" % (type_name, os.path.basename(gguf_path)))
        t = hit[0]
        return cls(t.type_name, t.type_id, t.name, os.path.basename(gguf_path), geom, table,
                   t.n_elts, t.shape_hf)


_GGUF_CACHE = {}


def _open_gguf(path):
    import gguf_reader as GR
    if path not in _GGUF_CACHE:
        _GGUF_CACHE[path] = GR.Gguf(path)
    return _GGUF_CACHE[path]


# ---------------------------------------------------------------------------------------------
# 4. the policy
# ---------------------------------------------------------------------------------------------
# Branch labels: every decision path is named, and --selftest asserts each one was actually taken. A
# policy branch nobody executed has been hoped for, not tested (`pitfalls/40` §"count reachability").
BRANCHES = (
    "judged-empty",
    "exact-nothing-diverges",
    "refuse-field-not-declared",
    "refuse-width-below-q8",
    "refuse-width-missing",
    "refuse-nonfinite-divergence",
    "refuse-signed-zero-flip",
    "refuse-beyond-one-ulp",
    "refuse-systematic-shift",
    "refuse-no-attribution",
    "refuse-attribution-false",
    "accept-isolated-one-ulp",
    "ruler-equal", "ruler-step", "ruler-zero-sign", "ruler-inf-finite", "ruler-inf-pair",
    "ruler-nan-payload", "ruler-nan-other",
    "format-derived-bf16", "format-derived-f16", "format-derived-f32", "format-crosscheck",
    "format-source-missing", "format-sabotage-caught",
    "width-fixtures-from-file", "width-rulers-disagree", "dead-branch-witness",
)
REACHED = set()


def branch(label):
    REACHED.add(label)
    return label


class Attribution(object):
    """A lane's claim that an accepted divergence is EXPLAINED, not merely small (D-16 condition 3).

    `ok` must be computed from data by the lane — the gather lane derives it from both arms' distances
    from the pinned oracle words — and the policy requires its presence and its truth for every
    acceptance of a real divergence, printing `detail` beside the census so the reason sits on the same
    line as the number it justifies. A free-text reason nobody could falsify would be a comment
    wearing a counter (`pitfalls/40`), so the boolean is the checked part and the text is the record.
    """

    __slots__ = ("ok", "detail")

    def __init__(self, ok, detail):
        self.ok = bool(ok)
        self.detail = detail


class Verdict(object):
    __slots__ = ("accepted", "reasons", "field", "width", "census", "attribution")

    def __init__(self, accepted, reasons, field, width, census, attribution):
        self.accepted = accepted
        self.reasons = reasons
        self.field = field
        self.width = width
        self.census = census
        self.attribution = attribution

    @property
    def vetoed_by_width(self):
        return any("D-17 width veto" in r for r in self.reasons)

    def line(self):
        w = self.width.type_name if self.width is not None else "no-artifact-width"
        att = self.attribution.detail if self.attribution is not None else "none"
        return ("POLICY field=%s width=%s verdict=%s reasons[%s] attribution=%s"
                % (self.field, w, "ACCEPT" if self.accepted else "REJECT",
                   ";".join(self.reasons) or "none", att))


def finish(accepted, reasons, field, width, census, attribution):
    print(census.line(field))
    v = Verdict(accepted, reasons, field, width, census, attribution)
    print(v.line())
    return v


def judge(field, declared_fields, width, census, attribution=None, require_attribution=True):
    """D-17's acceptance rule, over one field of one tensor. Prints the census and the verdict.

    `declared_fields` — the lane's declared set (a). `width` — a WidthClass read from an artifact (b);
    None is a refusal, not a default (c). `census` — this ruler's census of the lane's pairs (c). The
    census print is (d).
    """
    reasons = []
    if census.judged == 0:
        branch("judged-empty")
        reasons.append("nothing judged: a census over an empty population cannot fail")
        return finish(False, reasons, field, width, census, attribution)
    if census.diverging == 0:
        branch("exact-nothing-diverges")
        reasons.append("exact: %d/%d identical, no acceptance was needed"
                       % (census.judged, census.judged))
        return finish(True, reasons, field, width, census, attribution)
    # (a) outside the declared field set exactness is required, and "the count must stay zero" is
    #     checked here rather than hoped for.
    if field not in set(declared_fields):
        branch("refuse-field-not-declared")
        reasons.append("field %r is not in the lane's declared set (%s): exactness required and %d "
                       "elements moved" % (field, sorted(declared_fields), census.diverging))
    # (b) the width veto, from the artifact
    if width is None:
        branch("refuse-width-missing")
        reasons.append("no width class was read from an artifact: the veto cannot be evaluated, and "
                       "silence is not permission")
    elif not width.at_least_q8:
        branch("refuse-width-below-q8")
        reasons.append("D-17 width veto: ggml type %s stores %.4f bpw, below Q8 (%.1f). The Q4 veto "
                       "is the user's, is structural, and applies even at 1 ULP"
                       % (width.type_name, width.bpw, Q8_MIN_BPW))
    # (c) isolation, in its three separate teeth
    if census.nonfinite():
        branch("refuse-nonfinite-divergence")
        reasons.append("non-finite divergence refused: nan_payload=%d nan_vs_other=%d "
                       "inf_vs_finite=%d inf_vs_inf=%d — NaN is never within 1 ULP of a finite value, "
                       "and a one-step overflow of a single element is exactly the defect an "
                       "acceptance must never bless"
                       % (census.count(NAN_PAIR), census.count(NAN_OTHER),
                          census.count(INF_FINITE), census.count(INF_PAIR)))
    if census.count(ZERO_SIGN):
        branch("refuse-signed-zero-flip")
        reasons.append("signed-zero flip refused: +0 and -0 are one ordered step apart, but the sign "
                       "of a zero propagates (reciprocal, copysign, a silu denominator), so exactness "
                       "is required there — %d element(s)" % census.count(ZERO_SIGN))
    if census.n_ge(2):
        branch("refuse-beyond-one-ulp")
        reasons.append("%d element(s) at >=2 ULP (max %d): isolation requires zero"
                       % (census.n_ge(2), census.max_steps))
    if census.n_at(1) * 2 >= census.judged:
        branch("refuse-systematic-shift")
        reasons.append("systematic shift: %d of %d judged elements sit at exactly 1 ULP, so a strict "
                       "majority is not even bit-identical — a wrong rounding mode or an off-by-one in "
                       "a reduction looks exactly like this, and that is a different function, not "
                       "rounding noise" % (census.n_at(1), census.judged))
    # (d) attribution: an acceptance arrives with a reason that was computed, or it is not an acceptance
    if not reasons and require_attribution and attribution is None:
        branch("refuse-no-attribution")
        reasons.append("no attribution supplied: an accepted divergence must be explained by data the "
                       "lane computed, not merely observed to be small")
    if not reasons and attribution is not None and not attribution.ok:
        branch("refuse-attribution-false")
        reasons.append("the lane's own attribution does not hold: %s" % attribution.detail)
    if reasons:
        return finish(False, reasons, field, width, census, attribution)
    branch("accept-isolated-one-ulp")
    reasons.append("isolated ±1 ULP: %d of %d elements at exactly 1 ULP, 0 at >=2, field declared, "
                   "width %s (%.4f bpw) is >= Q8"
                   % (census.n_at(1), census.judged, width.type_name, width.bpw))
    return finish(True, reasons, field, width, census, attribution)


# ---------------------------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------------------------


class V(object):
    """Check tally in the repo's house form: every check named, counted, printed; the verdict line
    refuses a run whose finding count is non-zero."""

    def __init__(self):
        self.n = 0
        self.f = 0

    def check(self, name, ok, n, detail=""):
        self.n += 1
        good = bool(ok)
        if not good:
            self.f += 1
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if good else 0, n, detail))
        return good


def selftest_formats(v):
    fmts = {}
    for name in ("bf16", "f16", "f32"):
        try:
            f = format_facts(name)
        except FormatSourceError as e:
            v.check("format-%s-derived" % name, False, 0, "DERIVATION FAILED: %s" % e)
            continue
        fmts[name] = f
        branch("format-derived-" + name)
        print(f.line())
        two = "finfo" in f.crosscheck
        if two:
            branch("format-crosscheck")
        v.check("format-%s two independent sources agree" % name, two and "bitimage" in
                ",".join(f.sources), f.mant,
                "mant=%d exp=%d bias=%d derived_from=%s crosscheck=%s"
                % (f.mant, f.exp, f.bias, ",".join(f.sources), f.crosscheck))
        step_up = _value_of(name, f.image_one + 1)
        v.check("format-%s one-image-step-above-one-is-2^-mant" % name,
                abs(step_up - 1.0) <= 2.0 ** -f.mant, 1,
                "next image above 1.0 decodes to %.10g; 1+2^-%d is %.10g"
                % (step_up, f.mant, 1.0 + 2.0 ** -f.mant))
    v.check("bf16 mantissa derives to 7, not the typed 8 of §ulp_of's prose",
            fmts.get("bf16") and fmts["bf16"].mant == 7,
            fmts["bf16"].mant if fmts.get("bf16") else "no source",
            "speccheck/moe_ref.py §ulp_of says 8 explicit bits in prose and uses 7 in the formula; "
            "the formula is the one that is right, and this check is what stops a typed 8 coming back")
    v.check("f32 mantissa derives to 23", fmts.get("f32") and fmts["f32"].mant == 23,
            fmts["f32"].mant if fmts.get("f32") else "no source")
    # a missing source must be a loud failure, never a default
    global _SABOTAGE_SOURCE_UNAVAILABLE
    _SABOTAGE_SOURCE_UNAVAILABLE = True
    try:
        try:
            format_facts("bf16")
            msg = None
        except FormatSourceError as e:
            msg = str(e)
    finally:
        _SABOTAGE_SOURCE_UNAVAILABLE = False
    branch("format-source-missing")
    v.check("a missing format source fails loudly instead of typing a width",
            msg is not None, 1, "message: %s" % (msg[:140] if msg else
                                                 "NO EXCEPTION — the tool guessed a width"))
    try:
        format_facts("nope")
        v.check("an unknown format is refused", False, 0, "no exception")
    except FormatSourceError:
        v.check("an unknown format is refused", True, 1)
    try:
        format_facts("bf16", _inject_mant=8)
        v.check("sabotage: a typed bf16 mantissa of 8 is caught", False, 0,
                "no exception — the injected width shipped as if derived")
    except FormatSourceError as e:
        branch("format-sabotage-caught")
        v.check("sabotage: a typed bf16 mantissa of 8 is caught", True, 1, str(e)[:140])
    return fmts


RULER_ARM_BRANCH = {EQUAL: "ruler-equal", STEP: "ruler-step", ZERO_SIGN: "ruler-zero-sign",
                    INF_FINITE: "ruler-inf-finite", INF_PAIR: "ruler-inf-pair",
                    NAN_PAIR: "ruler-nan-payload", NAN_OTHER: "ruler-nan-other"}


def selftest_ruler(v, fmts):
    for name in sorted(fmts):
        r = Ruler(fmts[name])
        m = r.format.mant
        img = lambda x: _image_of(name, x)                       # noqa: E731
        big = ((1 << r.format.exp) - 2) * (1 << m) + ((1 << m) - 1)
        pinf = big + 1
        ninf = r.sign | pinf
        qnan = pinf | 1                                          # the image one step above +inf
        zero, nzero = 0, r.sign
        neg1 = r.sign | img(1.0)
        arms = [
            ("identical", img(1.0), img(1.0), EQUAL, 0),
            ("one-step-above-one", img(1.0), img(1.0) + 1, STEP, 1),
            ("power-of-two-1-to-2", img(1.0), img(2.0), STEP, 1 << m),
            ("signed-zero", zero, nzero, ZERO_SIGN, 1),
            ("zero-to-smallest-subnormal", zero, 1, STEP, 1),
            ("largest-subnormal-to-smallest-normal", (1 << m) - 1, 1 << m, STEP, 1),
            ("largest-finite-to-infinity", big, pinf, INF_FINITE, 1),
            ("plus-inf-to-minus-inf", pinf, ninf, INF_PAIR, "any"),
            ("nan-vs-finite", qnan, img(3.7), NAN_OTHER, None),
            ("nan-vs-adjacent-plus-inf", qnan, pinf, NAN_OTHER, None),
            ("nan-payload-move", qnan, qnan | 2, NAN_PAIR, None),
            ("negative-one-step", neg1, neg1 + 1, STEP, 1),
            ("across-zero-minus-one-to-plus-one", neg1, img(1.0), STEP,
             2 * (r.format.bias << m) + 1),
        ]
        for aname, x, y, want_cat, want_steps in arms:
            kind, steps = r.classify(x, y)
            branch(RULER_ARM_BRANCH[want_cat])
            steps_ok = (want_steps == "any" and steps is not None) or steps == want_steps
            v.check("ruler-%s-%s" % (name, aname), kind == want_cat and steps_ok,
                    1 if steps is None else steps,
                    "got %s/%s want %s/%s" % (kind, steps, want_cat, want_steps))
        v.check("ruler-%s-nan-is-never-within-one" % name,
                not r.within(qnan, img(3.7)) and not r.within(img(3.7), qnan)
                and not r.within(qnan, pinf), 1,
                "even the NaN pattern one ordered step above +inf is refused, in both directions")
        v.check("ruler-%s-subnormal-boundary-is-one-step" % name,
                r.classify((1 << m) - 1, 1 << m)[1] == 1, 1,
                "the float-quantum ruler cannot say this: it reads the gap as 2 different quanta")
        v.check("ruler-%s-one-ulp-at-a-power-of-two-is-the-binade-below" % name,
                r.classify(img(1.0) - 1, img(1.0))[1] == 1
                and r.classify(img(1.0), img(1.0) + 1)[1] == 1, 1,
                "the step below 1.0 and the step above it are both one step, which is why a length "
                "cannot be the ruler")


def selftest_negatives(v, fmts):
    """The negatives fixture, and the dead branch witnessed on it.

    `pitfalls/30` §A: `np.where(x < 0, …)` is never true for an unsigned image array, so the sign
    transform degenerates and a sign flip reads as a zero difference. The fixture below is negative
    enough for that to matter, and is checked two ways: the real ruler must see every pair as a
    divergence, and the naive form must be witnessed reporting zero of them.
    """
    for name in sorted(fmts):
        r = Ruler(fmts[name])
        vals = [-1.0, -3.5, -8.0, -0.00390625, -123.0, -0.5]
        neg = [_image_of(name, x) for x in vals]
        pos = [_image_of(name, abs(x)) for x in vals]
        pairs = list(enumerate(zip(neg, pos)))
        cen = r.census((i, a, b) for i, (a, b) in pairs)
        v.check("negatives-%s-sign-flips-are-real-divergences" % name,
                cen.diverging == len(vals) and cen.n_ge(2) == len(vals), cen.diverging,
                "%d of %d negative/positive pairs diverge, all of them by >=2 ULP"
                % (cen.diverging, len(vals)))
        vac = r.dead_branch_vacuity([(a, b) for _, (a, b) in pairs])
        if vac is None:
            v.check("dead-branch witness is reachable", False, 0,
                    "numpy is not importable here, so the pitfall could not be reproduced — run "
                    "--selftest under %s" % VENV_PY)
        else:
            branch("dead-branch-witness")
            v.check("dead-branch witness: the naive `x<0` form reports a vacuous zero",
                    vac == 0 and cen.diverging > 0, cen.diverging,
                    "naive form: %d differing of %d; the real ruler: %d. That is %s §A firing on "
                    "this fixture, not a hypothetical" % (vac, len(vals), cen.diverging,
                                                          "pitfalls/30"))
        # the same negatives at exactly 1 ULP, isolated: the accept path on the side of the number
        # line where the sign transform lives, which is where a wrong transform hides
        n = 16
        base = [_image_of(name, -(1.0 + 0.01 * i)) for i in range(n)]
        got = list(base)
        got[3] = base[3] + 1
        got[9] = base[9] + 1
        ciso = r.census((i, a, b) for i, (a, b) in enumerate(zip(base, got)))
        d = judge("s", ["s"], WidthClass.from_type("F32"), ciso,
                  Attribution(True, "fixture: two negative elements crossed a rounding boundary"))
        v.check("POLICY isolated ±1 on NEGATIVE values is ACCEPTED", d.accepted, ciso.diverging,
                ";".join(d.reasons)[:150])
        # and the systematic version of the same fixture must be refused
        csys = r.census((i, a, a + 1) for i, a in enumerate(base))
        d2 = judge("s", ["s"], WidthClass.from_type("F32"), csys, Attribution(True, "fixture"))
        v.check("POLICY a systematic shift on NEGATIVE values is REJECTED", not d2.accepted,
                csys.diverging, ";".join(d2.reasons)[:150])


def selftest_policy(v, gguf_path, fmts):
    """The width arms. Every fixture is a REAL tensor of the type it stands for, taken from the pinned
    file — the veto test must not be able to pass on a typed string."""
    w = {}
    parsed = None
    try:
        parsed = _open_gguf(gguf_path)
    except Exception as e:
        v.check("pinned gguf readable for the width fixtures", False, 0, str(e))
    # the population is the file's own type census, so a type that is not in the file is reported as
    # absent rather than quietly missing from the denominator (pitfalls/40 §"derive the population")
    present = {}
    if parsed is not None:
        for t in parsed.tensors:
            present.setdefault(t.type_name, t.name)
    want = {"Q4_K": False, "Q5_K": False, "Q6_K": False, "Q8_0": True, "F32": True,
            "F16": True, "BF16": True}
    for tname, expected in sorted(want.items()):
        if tname not in present:
            print("WIDTHABSENT type=%s is not a tensor type in %s: no veto fixture can be built for "
                  "it here, and this is reported as absent rather than silently skipped"
                  % (tname, os.path.basename(gguf_path)))
            continue
        try:
            w[tname] = WidthClass.first_of_type(gguf_path, tname, _parsed=parsed)
        except WidthError as e:
            v.check("width fixture %s comes from the pinned file" % tname, False, 0, str(e))
            continue
    if all(k in w for k in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32")):
        branch("width-fixtures-from-file")
    v.check("width fixtures are real tensors of their own ggml type",
            all(k in w for k in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32")), len(w),
            " | ".join(w[k].line() for k in sorted(w)) if w else "none")
    for label in sorted(w):
        wc = w[label]
        v.check("width-%s is %s Q8" % (label, ">=" if want[label] else "<"),
                wc.at_least_q8 == want[label], wc.bpw, wc.line())
    # The two width rulers must agree on every type this file can actually ask the policy about.
    # Types the file does not contain are still DIFFED and PRINTED, because a disagreement that no
    # tensor in this file exercises is exactly the kind a file-size witness cannot see.
    try:
        import gguf_width_table as GWT
        import ggml_block_geom as GBG
        universe = sorted(set(list(present) + list(GBG.REQUIRED)))
        drift = []
        for t in universe:
            if t in GWT.BLK:
                try:
                    if abs(8.0 * GWT.BLK[t][1] / GWT.BLK[t][0] - GBG.bpw_of(t)) > 1e-9:
                        drift.append(t)
                except Exception:
                    drift.append(t)
        in_file = [t for t in drift if t in present]
        for t in drift:
            print("WIDTHDRIFT type=%s ggml_block_geom=%.6f bpw gguf_width_table=%.6f bpw in_pinned_"
                  "file=%d note=%s"
                  % (t, GBG.bpw_of(t), 8.0 * GWT.BLK[t][1] / GWT.BLK[t][0], 1 if t in present else 0,
                     "live: the policy would REFUSE this type until the two rulers agree" if t
                     in present else "latent: no tensor of this type in the pinned file, so the "
                     "file-size witness in gguf_width_table.py cannot see it"))
        v.check("the two width rulers agree on every type the pinned file contains", not in_file,
                len(present), "drift inside the file=%s; drift outside it=%s"
                % (in_file or "none", [t for t in drift if t not in present] or "none"))
        saved = GWT.BLK["Q8_0"]
        try:
            GWT.BLK["Q8_0"] = (32, 35, "SABOTAGE: one extra byte")
            try:
                WidthClass.from_type("Q8_0")
                fired = False
            except WidthError:
                fired = True
        finally:
            GWT.BLK["Q8_0"] = saved
        if fired:
            branch("width-rulers-disagree")
        v.check("sabotage: an injected stale width-table row is caught", fired, 1,
                "the injected Q8_0 row must make WidthClass refuse, not quietly ship")
    except Exception as e:
        v.check("the two width rulers agree on every type the pinned file contains", False, 0,
                repr(e))

    if "f32" not in fmts:
        v.check("f32 ruler available for the policy arms", False, 0, "no f32 format facts")
        return
    r = Ruler(fmts["f32"])
    img = lambda x: _image_of("f32", x)                          # noqa: E731
    base = [img(0.5 + 0.01 * i) for i in range(16)]
    q8 = w.get("Q8_0") or WidthClass.from_type("F32")

    def cen(pairs):
        return r.census(pairs)

    isolated = list(base)
    isolated[3] = base[3] + 1
    isolated[9] = base[9] - 1
    c_iso = cen((i, a, b) for i, (a, b) in enumerate(zip(base, isolated)))
    att = Attribution(True, "fixture: two elements crossed a rounding boundary")
    d = judge("s", ["s", "raw"], q8, c_iso, att)
    v.check("POLICY isolated ±1 in a declared field at Q8 is ACCEPTED", d.accepted, c_iso.diverging,
            ";".join(d.reasons)[:160])

    c_sys = cen((i, a, a + 1) for i, a in enumerate(base))
    d = judge("s", ["s"], q8, c_sys, att)
    v.check("POLICY systematic ±1 on EVERY element is REJECTED (the mandatory tripwire)",
            not d.accepted, c_sys.diverging, ";".join(d.reasons)[:160])
    c_half = cen((i, a, a + 1 if i < 8 else a) for i, a in enumerate(base))
    d = judge("s", ["s"], q8, c_half, att)
    v.check("POLICY half-the-field-at-1-ULP is REJECTED (isolation is not the >=2 rule)",
            not d.accepted, c_half.diverging, ";".join(d.reasons)[:160])

    c_two = list(base)
    c_two[7] = base[7] + 2
    d = judge("s", ["s"], q8, cen((i, a, b) for i, (a, b) in enumerate(zip(base, c_two))), att)
    v.check("POLICY a single element at 2 ULP is REJECTED", not d.accepted, 1,
            ";".join(d.reasons)[:160])

    qnan32 = 0x7F800001
    c_nan = list(base)
    c_nan[5] = qnan32
    d = judge("s", ["s"], q8, cen((i, a, b) for i, (a, b) in enumerate(zip(base, c_nan))), att)
    v.check("POLICY NaN-vs-finite is REJECTED", not d.accepted, 1, ";".join(d.reasons)[:160])
    c_nanpay = list(base)
    c_nanpay[5] = qnan32 | 4
    base_np = list(base)
    base_np[5] = 0x7FC00000
    c_nanpay[5] = 0x7FC00004
    d = judge("s", ["s"], q8,
              cen((i, a, b) for i, (a, b) in enumerate(zip(base_np, c_nanpay))), att)
    v.check("POLICY a NaN payload move is REJECTED (not a rounding step)", not d.accepted, 1,
            ";".join(d.reasons)[:160])

    # the user's veto: the same fixture, the same 1 ULP, the tensor's own type is Q4_K
    if w.get("Q4_K"):
        d = judge("s", ["s", "raw"], w["Q4_K"], c_iso, att)
        v.check("POLICY Q4_K at exactly 1 ULP is REJECTED — and by the veto, not by accident",
                (not d.accepted) and d.vetoed_by_width, c_iso.diverging,
                "%s | %s" % (w["Q4_K"].line(), ";".join(d.reasons)[:160]))
    if w.get("Q6_K"):
        d = judge("s", ["s", "raw"], w["Q6_K"], c_iso, att)
        v.check("POLICY Q6_K at exactly 1 ULP is REJECTED (6.5625 bpw is below Q8)",
                (not d.accepted) and d.vetoed_by_width, c_iso.diverging,
                "%s" % ";".join(d.reasons)[:160])
    if w.get("Q5_K"):
        d = judge("s", ["s", "raw"], w["Q5_K"], c_iso, att)
        v.check("POLICY Q5_K at exactly 1 ULP is REJECTED (5.5 bpw is below Q8)",
                (not d.accepted) and d.vetoed_by_width, c_iso.diverging,
                "%s" % ";".join(d.reasons)[:160])
    if w.get("F32"):
        d = judge("s", ["s", "raw"], w["F32"], c_iso, att)
        v.check("POLICY F32 (32 bpw) accepts the same isolated ±1", d.accepted, c_iso.diverging,
                ";".join(d.reasons)[:160])

    d = judge("gate", ["up", "s", "raw"], q8, c_iso, att)
    v.check("POLICY a divergence in an undeclared field is REJECTED (count must stay zero)",
            (not d.accepted) and any("not in the lane's declared set" in rr for rr in d.reasons),
            c_iso.diverging, ";".join(d.reasons)[:160])

    big32 = ((1 << 8) - 2) * (1 << 23) + ((1 << 23) - 1)
    of_a, of_b = list(base), list(base)
    of_a[2] = big32                       # the largest finite f32 ...
    of_b[2] = big32 + 1                   # ... and +inf, one ordered step away
    d = judge("s", ["s"], q8, cen((i, a, b) for i, (a, b) in enumerate(zip(of_a, of_b))), att)
    v.check("POLICY a one-step overflow to +inf is REJECTED (a step count is not permission)",
            (not d.accepted) and any("non-finite" in rr for rr in d.reasons), 1,
            ";".join(d.reasons)[:160])

    z_a, z_b = list(base), list(base)
    z_a[4] = 0
    z_b[4] = 0x80000000
    d = judge("s", ["s"], q8, cen((i, a, b) for i, (a, b) in enumerate(zip(z_a, z_b))), att)
    v.check("POLICY a signed-zero flip is REJECTED and named, not silently absorbed",
            not d.accepted, 1, ";".join(d.reasons)[:120])

    c_same = cen((i, a, a) for i, a in enumerate(base))
    d = judge("s", ["s"], q8, c_same)
    v.check("POLICY all-identical is ACCEPTED as exact, having judged something",
            d.accepted and c_same.judged == 16 and c_same.diverging == 0, c_same.judged)
    d = judge("s", ["s"], q8, r.census([]))
    v.check("POLICY an empty census is REJECTED (a check that cannot fail)", not d.accepted, 0,
            ";".join(d.reasons)[:120])
    d = judge("s", ["s"], q8, c_iso)
    v.check("POLICY an isolated ±1 with no attribution is REJECTED", not d.accepted, 1,
            ";".join(d.reasons)[:160])
    d = judge("s", ["s"], q8, c_iso, Attribution(False, "the lane says the divergence is unexplained"))
    v.check("POLICY a failing attribution is REJECTED", not d.accepted, 1,
            ";".join(d.reasons)[:160])
    d = judge("s", ["s"], None, c_iso, att)
    v.check("POLICY a missing width class is REJECTED (never default to permissive)", not d.accepted,
            1, ";".join(d.reasons)[:160])


def selftest(v, gguf_path=PINNED_GGUF):
    fmts = selftest_formats(v)
    selftest_ruler(v, fmts)
    selftest_negatives(v, fmts)
    selftest_policy(v, gguf_path, fmts)
    missed = sorted(set(BRANCHES) - REACHED)
    bogus = sorted(REACHED - set(BRANCHES))
    print("BRANCHES declared=%d reached=%d unreached=%s bogus_labels=%s"
          % (len(BRANCHES), len(REACHED), ",".join(missed) or "none", ",".join(bogus) or "none"))
    v.check("every declared branch was actually executed by this selftest", not missed,
            len(BRANCHES), "unreached: %s — an unexecuted branch has been hoped for, not tested"
            % (",".join(missed) or "none"))
    v.check("every branch reached is a declared one", not bogus, len(BRANCHES),
            "labels not in BRANCHES: %s" % (",".join(bogus) or "none"))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--formats", action="store_true")
    ap.add_argument("--width", nargs="?", const="__any__", metavar="GGUF")
    ap.add_argument("--tensor", default=None)
    ap.add_argument("--gguf", default=PINNED_GGUF)
    a = ap.parse_args()
    if a.formats:
        for name in ("bf16", "f16", "f32", "f64"):
            try:
                print(format_facts(name).line())
            except FormatSourceError as e:
                print("FORMAT name=%s UNAVAILABLE %s" % (name, e))
        return 0
    if a.width:
        if a.tensor:
            print(WidthClass.from_gguf(a.width, a.tensor).line())
            return 0
        g = _open_gguf(a.width)
        seen = {}
        for t in g.tensors:
            seen.setdefault(t.type_name, t.name)
        for tn in sorted(seen):
            print(WidthClass.from_gguf(a.width, seen[tn], _parsed=g).line())
        return 0
    if a.selftest:
        v = V()
        selftest(v, a.gguf)
        print("RESULT %s ulp_policy selftest checks=%d findings=%d"
              % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
        return 0 if v.f == 0 else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
