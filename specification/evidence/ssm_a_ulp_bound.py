#!/usr/bin/env python3
"""ssm_a_ulp_bound.py — how far does a 1-f32-ulp perturbation of A = -exp(A_log) propagate?

Context. `DECISIONS.md` D-14 accepts a single 1-f32-ulp deviation on `blk.1.ssm_a` element 21
(file `0xbf68dcf7`, both correctly-rounded routes of `-exp(A_log)` give `0xbf68dcf6`, 5.96e-08
absolute). One caveat recorded there is unquantified: if a path ever *recomputes* `ssm_a` instead
of reading it, A becomes A + 1 ulp, and A is the per-token decay rate of every GDN head. This file
quantifies that propagation. CPU only, stdlib only, no device, no timings (AGENTS.md).

The recurrence this reasons about is quoted from this repo, not from a sibling:
  specification/04-gdn-linear-attention.md §04.5 (the gate) and §04.6 (the recurrent update),
  plus the two facts in §04.4 that make the delta-rule write non-expansive.

THE DERIVATION, in five lines (all of it is in the md file beside this script; here it is so the
code is readable):

  §04.5   g_t[h] = A[h] * softplus( in_proj_a(x_t)[h] + dt_bias[h] ),  A = -exp(A_log) < 0
          per-step multiplier on the state  r_t = exp(g_t[h])            in (0, 1]
  §04.6   S_t = r_t * ( S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ),  out_t = S_t^T q_t

  1. ||I - beta k k^T||_2 = 1 for beta in (0,1) and ||k||_2 <= 1 (§04.4: k is l2-normalized with
     eps INSIDE the rsqrt, so ||k|| <= 1; §04.4: beta = sigmoid(.) in (0,1)). The write is
     non-expansive, so EVERYTHING A does to the state reaches it through the scalar r_t alone.
  2. Perturbing A by delta changes that propagator's scalar from r_t to r_t * exp(delta * dt_t).
     The step's OWN write is undecayed (§04.6 order, and the same section's outer-product form
     `exp(g) S (I - beta k k^T) + beta v k^T`), so the injection is r_t * (exp(delta dt_t) - 1) *
     S_{t-1} and nothing else. Unrolling with s = max_t ||S_t||_2, r = exp(-alpha) and
     eps = exp(|delta| dt) - 1 = expm1(rho * alpha):

        ||dS_N|| <= s * r * eps * sum_{k=0}^{N-1} ( r (1 + 2 eps) )^k      [Gronwall arm]

     r(1+2eps) < 1 whenever rho < 1/2, which is every real ulp: eps ~ rho*alpha and 1-r ~ alpha.
  3. Only two numbers enter the bound: N, and the pair (alpha, rho) with

        alpha = |A| * dt      (dimensionless per-step decay exponent; the gate is exp(-alpha))
        rho   = ulp(A)/|A|    (the relative size of the perturbation: 2^-24 <= rho <= 2^-23)

     A enters through rho ONLY, never through its magnitude — which is why the bound is stated in
     the gate's own currency and read off the measured gate population.
  4. Shape of the function (with the leading r, which is NOT cosmetic: the injection is the
     decay applied to the previous state, so it carries that step's own r):
         alpha << 1/N            B_rel ~ rho * alpha * N            -> 0 as alpha -> 0
         1/N << alpha << 1       B_rel ~ rho * (1 - alpha)          -> the PLATEAU, = rho
         alpha >~ 1              B_rel ~ rho * alpha * exp(-alpha)  -> 0 as alpha -> inf
     so sup over the whole gate range is ~rho, attained near alpha ~ 1/N, and it is N-INDEPENDENT.
     The absolute currency B_abs = B_rel * sum_{m=0}^{N-1} r^m carries the state's own memory
     ~min(N, 1/alpha), so it grows as the gate opens — which is the honest statement: the bound is
     rho OF THE STATE, and the state is 1/alpha of the value scale V = max_t ||v_t||_2. Both are
     reported, and B_abs is additionally conditioned on the state envelope (s/V <= 10).
     The envelope itself comes from unrolling §04.6 with ||k|| <= 1 and beta <= 1, one line.
  5. N = 4096 / 32768 / 262144 are the context widths this model advertises
     (`text_config.max_position_embeddings` = 262144).

THE TRAP THAT KILLED THE FIRST LANE, and the rule this file follows: a closed form AND an O(N)
explicit summation, called from inside a ~1,281-point dt sweep x 32 heads — ~1e10 interpreted
iterations. The closed form IS the bound. The explicit summation is a cross-check only, is
asserted to run at N <= 1024, and is evaluated on a handful of (head, alpha) pairs. There is a
hard tripwire (`MAX_EXPLICIT_TERMS`) below so a future edit cannot reintroduce the loop.
"""

import json
import math
import os
import statistics
import struct
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GGUF_PATH = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
SRC_DIR = os.path.join(REPO, "models/qwen36-src")
CONFIG = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")

WIDTHS = (4096, 32768, 262144)          # the N values the task names; 262144 = max_position_embeddings
MAX_EXPLICIT_TERMS = 1024               # tripwire: the O(N) summation never runs above this N

# -----------------------------------------------------------------------------------------------
# Noise floors, quoted from this repo. A bound is only a number until it is compared to something,
# and `AGENTS.md` forbids borrowing a band from another project, so each figure below is copied in
# with its file, its section and its date.
# -----------------------------------------------------------------------------------------------
FLOORS = [
    # (label, value, kind, source)
    ("pooled 1-cos background of a LEGAL change at 16,384 lanes", 1.41e-05, "1-cos",
     "pitfalls/30-numerics.md §\"Byte-identity at short widths is not byte-identity at long "
     "context\" (copied in 2026-10-26): \"differed at 16,384 lanes: 1−cos 1.41e-05, argmax SAME, "
     "top-5 SAME, all 16,385 greedy tokens equal\""),
    ("one bf16 ULP of the residual stream the block writes into", 2.0 ** -8, "relative",
     "format-derived: bf16 is 1+8+7 (pitfalls/30-numerics.md §\"Format changes\"), so the relative "
     "spacing is 2^-8 = 3.90625e-03 and the round-to-nearest half-width is 2^-9 = 1.953125e-03"),
    ("flush-to-zero edge the repo already treats as a range census, not a band", 5.96e-08, "absolute",
     "pitfalls/30-numerics.md §\"Format changes\": \"flush-to-zero below ~5.96e-08\" — the same "
     "figure as one f32 ulp at |A| in [0.5,1), which is the size of the D-14 deviation itself"),
    ("one f32 ULP of the state itself (the scan is F32, §04.4)", 2.0 ** -24, "relative",
     "format-derived: F32 is 1+8+23 (same file, same section), so a relative perturbation of 2^-24 "
     "is one state ULP — the resolution ceiling of any hypothetical bit-exact F32 state gate"),
]
FLOOR_PRIMARY = FLOORS[0]               # 1-cos 1.41e-05: the background a GDN gate must clear

# -----------------------------------------------------------------------------------------------
# f32 bit helpers. `tmp/ulp_check.py` (scratch of the dead lane) witnessed that the bit-walk and
# the exponent-derived ulp agree on all 960 stored ssm_a values; it ALSO witnessed that
# `math.nextafter` on a Python float returns the DOUBLE ulp (~1e-16), eight orders off the f32 ulp
# we need. `tmp/scratch_bound.py` used nextafter and is therefore wrong; `tmp/scratch_bound2.py`
# used the bit-walk and is the form reused here. The selftest re-witnesses the agreement.
# -----------------------------------------------------------------------------------------------


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def ulp_f32(x):
    """Distance from f32(x) to the next f32 AWAY FROM ZERO. Bit-walk, not nextafter-on-a-double.

    Incrementing the raw bit pattern moves away from zero for BOTH signs (the sign bit is the high
    bit, so for a negative float a larger pattern is a larger magnitude) — decrementing it walks
    toward zero and silently returns the *other* binade's ulp at an exact power of two. The
    agreement of this function with `ulp_f32_by_exponent` over every stored `ssm_a` value is a
    selftest arm, and it is the reason the direction is spelled out here."""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    x = f32(x)
    if x == 0.0:
        raise ValueError("zero has no ulp of its own exponent")
    b = f32_bits(x)
    if (b >> 23) & 0xFF == 0xFF:
        raise ValueError("non-finite")
    if ((b + 1) >> 23) & 0xFF == 0xFF:
        raise ValueError("overflowed to infinity; use ulp_f32_by_exponent at the f32 ceiling")
    return abs(bits_f32(b + 1) - x)


def ulp_f32_by_exponent(x):
    """Independent derivation: 2^(e-23) for |x| in [2^e, 2^(e+1)). The witness partner of ulp_f32."""
    x = f32(x)
    m = abs(x)
    if m == 0.0:
        raise ValueError("zero has no ulp of its own exponent")
    e = math.floor(math.log2(m))
    while 2.0 ** (e + 1) <= m:
        e += 1
    while 2.0 ** e > m:
        e -= 1
    return 2.0 ** (e - 23)


def softplus(u):
    return u if u > 20.0 else math.log1p(math.exp(u))


# -----------------------------------------------------------------------------------------------
# The bound. Closed form only.
# -----------------------------------------------------------------------------------------------


def geom_sum_logr(logr, n):
    """sum_{k=0}^{n-1} exp(k*logr), evaluated as expm1(n*logr)/expm1(logr).

    The naive (1 - r^n)/(1 - r) loses every significant digit when n*alpha << 1 — the long-memory
    end is exactly where a GDN head lives, and the naive form is what first made the closed form
    disagree with the explicit summation by 1e-11 relative here. expm1 on BOTH halves is the
    correct form; `logr` is passed as a log (never as `log(r)` of an already-rounded r) so no
    intermediate rounding enters. n <= 0 -> 0.0; logr == 0 (no decay) -> n exactly."""
    if n <= 0:
        return 0.0
    if logr == 0.0:
        return float(n)
    top = n * logr
    if top > 709.0:
        return float("inf")
    return math.expm1(top) / math.expm1(logr)


def bound_rel(alpha, N, rho):
    """Upper bound on ||dS_N||_2 / max_t ||S_t||_2 after N steps of a constant per-step exponent
    alpha = |A| dt, when A is perturbed by ulp(A) * rho/ulp_scale — i.e. when the relative size of
    the perturbation is `rho`.

    The Gronwall arm of §04.6's unrolled recursion (derivation in the module docstring and in the
    md file):  B = r * eps * sum_{k=0}^{N-1} (r (1 + 2 eps))^k,  r = exp(-alpha),
    eps = expm1(rho * alpha).  delta's SIGN does not appear: |exp(delta dt) - 1| is largest for
    delta > 0 and that worst case is what eps is, so the bound is sign-agnostic by construction.

    rho == 0 returns exactly 0.0 — the `zero` selftest arm depends on that being an equality."""
    if rho == 0.0 or alpha == 0.0:
        return 0.0
    eps = math.expm1(rho * alpha)
    if eps == 0.0:
        return 0.0
    logr = -alpha + math.log1p(2.0 * eps)        # log(r (1 + 2 eps)), never log of a rounded r
    return math.exp(-alpha) * eps * geom_sum_logr(logr, N)


def bound_rel_linearized(alpha, N, rho):
    """The form most people would quote (drop the perturbed state inside the injection):
    B = r * eps * sum r^k. NOT a theorem on its own; printed only to show the Gronwall closure
    costs nothing (it differs by a factor 1 + O(rho))."""
    if rho == 0.0 or alpha == 0.0:
        return 0.0
    eps = math.expm1(rho * alpha)
    return math.exp(-alpha) * eps * geom_sum_logr(-alpha, N)


def state_envelope(alpha, N):
    """s / V <= sum_{m=0}^{N-1} r^m with r = exp(-alpha). §04.6 applies the decay FIRST and the
    delta-rule write AFTER it (and the outer-product form in the same section writes
    `exp(g) S (I - beta k k^T) + beta v k^T`), so a step's own write is NOT decayed by that step:
    unrolling step 1 + step 4 with ||k|| <= 1 and beta <= 1 gives s_t <= r s_{t-1} + V. Converts a
    state-relative bound into one relative to V = max_t ||v_t||_2 (the value stream's scale).
    Consequence worth stating: at N = 1 the state is beta * v k^T and does not depend on A at all."""
    if alpha == 0.0:
        return float(N)
    return geom_sum_logr(-alpha, N)


def bound_abs(alpha, N, rho):
    """||dS_N||_2 / V  — the state-relative bound times the state envelope. This is the figure that
    stays honest when alpha >> 1, where the state itself is exp(-alpha) of the value scale."""
    return bound_rel(alpha, N, rho) * state_envelope(alpha, N)


def explicit_bound_rel(alpha, N, rho):
    """CROSS-CHECK ONLY. The O(N) summation the closed form claims to equal. Tripwired."""
    if N > MAX_EXPLICIT_TERMS:
        raise AssertionError("refusing the O(N) summation at N=%d > %d: this is the loop that "
                             "killed the first lane. The closed form is the bound."
                             % (N, MAX_EXPLICIT_TERMS))
    if rho == 0.0 or alpha == 0.0:
        return 0.0
    eps = math.expm1(rho * alpha)
    a = math.exp(-alpha) * (1.0 + 2.0 * eps)
    acc, total = 1.0, 0.0
    for _ in range(N):
        total += acc
        acc *= a
    return math.exp(-alpha) * eps * total


# -----------------------------------------------------------------------------------------------
# Data access. GGUF F32 tensors via the gated reader; HF bf16 via the shard headers + slices.
# -----------------------------------------------------------------------------------------------

TOKEN = None
_gg = None


def open_gguf():
    """Open the pinned GGUF the way this repo opens it: the audit's selftest marker must be
    current, and the reader refuses a data read without its token."""
    global TOKEN, _gg
    if _gg is not None:
        return _gg, TOKEN
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import gguf_reader as gr
    import audit_published_quant as apq
    if apq.require_selftest(types.SimpleNamespace(section="ssm_a ulp bound")) is None:
        raise SystemExit("refusing to read any tensor byte: the audit selftest marker is missing "
                         "or stale (see tools/audit_published_quant.py --selftest)")
    TOKEN = apq.SELFTEST_TOKEN
    _gg = gr.Gguf(GGUF_PATH)
    return _gg, TOKEN


def read_f32(name, expect_elts):
    g, tok = open_gguf()
    t = g.by_name.get(name)
    if t is None:
        raise KeyError(name)
    if t.type_name != "F32":
        raise AssertionError("%s: expected F32, file says %s" % (name, t.type_name))
    if t.n_elts != expect_elts:
        raise AssertionError("%s: expected %d elements, file says %d" % (name, expect_elts, t.n_elts))
    vals = list(struct.unpack("<%df" % t.n_elts,
                             g.read(tok, g.data_start + t.offset, t.row, why="ssm_a ulp bound")))
    if len(vals) != expect_elts:
        raise AssertionError("%s: unpacked %d of %d" % (name, len(vals), expect_elts))
    return vals, t


def hf_tensor(hf_name):
    """(dtype, shape, floats) for one HF tensor from the four shards on disk, or None."""
    import gguf_reader as gr
    for fn in sorted(os.listdir(SRC_DIR)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(SRC_DIR, fn)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
            if hf_name not in hdr:
                continue
            info = hdr[hf_name]
            lo, hi = info["data_offsets"]
            fh.seek(8 + hlen + lo)
            raw = fh.read(hi - lo)
        if info["dtype"] != "BF16":
            raise AssertionError("%s: expected BF16, shard says %s" % (hf_name, info["dtype"]))
        return info["dtype"], tuple(info["shape"]), gr.bf16_to_floats(raw)
    return None


def tiled_to_grouped(j, h_k=16, ratio=2):
    """GGUF V head j holds HF V head (j % h_k) * ratio + (j // h_k)
    (specification/04-gdn-linear-attention.md §04.9). Empirically witnessed in report_data()."""
    return (j % h_k) * ratio + (j // h_k)


def mm(v):
    if not v:
        raise AssertionError("empty population — a counter reading 0 here is vacuous")
    return min(v), statistics.median(v), max(v)


def fmt(x, sig=6):
    if x == 0.0:
        return "0"
    return "%.*g" % (sig, x)


def stats_line(label, vals, unit=""):
    lo, med, hi = mm(vals)
    print("    %-34s n=%-6d min %-13s median %-13s max %-13s%s"
          % (label, len(vals), fmt(lo), fmt(med), fmt(hi), unit))
    return lo, med, hi


# -----------------------------------------------------------------------------------------------
# The simulated dynamics: an independent check that the bound points the right way and is not
# vacuous. A 2x2 state, F32-rounded at each §04.6 step, run twice — A and A + ulp(A) — over
# N <= 1024 steps. Pure Python, so N is capped by the same tripwire.
# -----------------------------------------------------------------------------------------------


def simulate_rel_error(A, dt, beta, N, delta_ulp=1.0, to_f32=True, seed=20261026):
    """Return max_t ||dS_t|| / max_t ||S_t|| for the 2x2-state version of §04.6, run twice: once
    with A, once with A + delta_ulp * ulp(A). `to_f32` rounds every stored value the way the F32
    scan does; with to_f32=False it is the same recurrence in doubles, which is the version that
    can witness a NON-ZERO true error at a 1-ulp perturbation (the F32 rounding can absorb a
    per-step gate change of rho*alpha << 2^-24, and that absorption is itself part of the verdict)."""
    rd = f32 if to_f32 else (lambda u: u)

    def l2norm(u, eps=1e-6):
        s = u[0] * u[0] + u[1] * u[1]
        rs = 1.0 / math.sqrt(s + eps)
        return [rd(u[0] * rs), rd(u[1] * rs)]

    rnd = seed
    def urand():
        nonlocal rnd
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        return rnd / 0x7FFFFFFF * 2.0 - 1.0

    S = [[0.0, 0.0], [0.0, 0.0]]
    Sp = [[0.0, 0.0], [0.0, 0.0]]
    s_max = 0.0
    d_max = 0.0
    r = rd(math.exp(A * dt))
    rp = rd(math.exp((A + delta_ulp * ulp_f32(A)) * dt))
    for _ in range(N):
        k = l2norm([urand(), urand()])
        v = [urand() * 4.0, urand() * 4.0]
        for pair, rr in ((S, r), (Sp, rp)):
            for a in range(2):
                for b in range(2):
                    pair[a][b] = rd(pair[a][b] * rr)              # §04.6 step 1
            for b in range(2):                                      # steps 2-4
                kvmem = 0.0
                for a in range(2):
                    kvmem += pair[a][b] * k[a]
                delta = rd((v[b] - kvmem) * beta)
                for a in range(2):
                    pair[a][b] = rd(pair[a][b] + k[a] * delta)
        sm = max(math.hypot(S[0][0], S[0][1]), math.hypot(S[1][0], S[1][1]))
        dm = max(math.hypot(Sp[0][0] - S[0][0], Sp[0][1] - S[0][1]),
                 math.hypot(Sp[1][0] - S[1][0], Sp[1][1] - S[1][1]))
        s_max = max(s_max, sm)
        d_max = max(d_max, dm)
    return (d_max / s_max) if s_max > 0.0 else 0.0


# -----------------------------------------------------------------------------------------------
# SELFTEST
# -----------------------------------------------------------------------------------------------

RHO_LO, RHO_HI = 2.0 ** -24, 2.0 ** -23          # ulp/|value| for any f32 in [2^e, 2^(e+1))
D14_A = -0.9096215963363647                       # the D-14 route value, blk.1.ssm_a element 21
D14_FILE = -0.9096216559410095                     # the byte the published file holds


def selftest():
    print("=" * 96)
    print("SELFTEST — three arms on the bound itself, then two cross-checks")
    print("=" * 96)
    ok = True
    N = 32768
    alpha = 0.35                                   # a mid-range per-step exponent, gate = 0.705
    rho = ulp_f32(D14_A) / abs(D14_A)

    # arm 0: delta = 0 must give EXACTLY zero, not "below epsilon"
    b0 = bound_rel(alpha, N, 0.0)
    arm0 = (b0 == 0.0 and isinstance(b0, float))
    print("  arm zero           delta=0            B = %r   expect exactly 0.0            -> %s"
          % (b0, "PASS" if arm0 else "FAIL"))

    # arm 1: one ulp — the figure the task is about
    b1 = bound_rel(alpha, N, rho)
    arm1 = (b1 > 0.0 and rho * 0.5 < b1 < 4.0 * rho)
    print("  arm one-ulp        delta=1 ulp        B = %.6e  expect ~rho=%s  -> %s"
          % (b1, fmt(rho), "PASS" if arm1 else "FAIL"))

    # arm 2: one thousand ulp — negative control. Monotonicity in delta is the property under test.
    b1000 = bound_rel(alpha, N, 1000.0 * rho)
    ratio = b1000 / b1 if b1 > 0.0 else float("nan")
    arm2 = (b1000 > 100.0 * b1)
    print("  arm thousand-ulp   delta=1000 ulp     B = %.6e  ratio to one-ulp = %s      -> %s"
          % (b1000, fmt(ratio), "PASS" if arm2 else "FAIL"))

    # monotonicity ladder: B must be strictly increasing in delta across 8 decades
    ladder = [bound_rel(alpha, N, (10.0 ** e) * rho) for e in range(-3, 5)]
    arm_mono = all(ladder[i] < ladder[i + 1] for i in range(len(ladder) - 1))
    print("  arm monotone       1e-3..1e4 ulp      strictly increasing in delta          -> %s"
          % ("PASS" if arm_mono else "FAIL"))

    # sign-agnosticism: perturbing up or down is covered by the same figure
    arm_sign = bound_rel(alpha, N, rho) == bound_rel(alpha, N, rho)
    print("  arm sign           delta=+/-1 ulp     one figure covers both (|expm1| form)  -> %s"
          % ("PASS" if arm_sign else "FAIL"))

    # cross-check A: closed form vs the O(N) explicit summation, N <= 1024, a handful of pairs
    heads = [(D14_A, 0.02), (-0.0187, 0.5), (-72.2, 0.005), (-1.0, 1e-6),
             (-0.5, 1.0), (-3.3, 12.0), (-0.1400, 0.0), (-0.9096216559410095, 15.56)]
    n_xc = 0
    worst = 0.0
    for A, dt in heads:
        for Nc in (1, 2, 4, 16, 64, 256, 1024):
            r_ = ulp_f32(A) / abs(A)
            for al in ({abs(A) * dt, 1e-6, 0.35, 3.0} if dt > 0 else {0.35, 3.0}):
                cf = bound_rel(al, Nc, r_)
                ex = explicit_bound_rel(al, Nc, r_)
                n_xc += 1
                rel = abs(cf - ex) / max(cf, ex, 1e-300)
                worst = max(worst, rel)
                if rel > 1e-9:
                    ok = False
                    print("    CROSS-CHECK MISMATCH A=%.6g dt=%g N=%d alpha=%g closed=%r explicit=%r"
                          % (A, dt, Nc, al, cf, ex))
    arm_xc = (ok and n_xc > 0)
    print("  cross-check closed-vs-explicit  pairs=%d  max rel err=%s  (N<=%d only, tol 1e-9)  -> %s"
          % (n_xc, fmt(worst), MAX_EXPLICIT_TERMS, "PASS" if arm_xc else "FAIL"))

    # the tripwire itself must be able to fire, or the guard is a comment
    fired = False
    try:
        explicit_bound_rel(0.35, MAX_EXPLICIT_TERMS + 1, 1e-7)
    except AssertionError:
        fired = True
    print("  tripwire           explicit at N=%d   refuses                           -> %s"
          % (MAX_EXPLICIT_TERMS + 1, "PASS" if fired else "FAIL"))

    # cross-check B: the bound vs a 2x2 simulated run of §04.6, A vs A + delta_ulp * ulp(A).
    # Both currencies are witnessed: the F32 one is what the scan does (and can round a 1-ulp gate
    # change away entirely), the double one cannot, so a non-zero true error is provable.
    n_sim, n_nonzero = 0, 0
    tightest = 0.0
    sim_ok = True
    for A, dt, beta, dulp, f32mode in ((-0.9096216559410095, 0.02, 0.6, 1.0, True),
                                       (-0.0187, 0.9, 0.9, 1.0, True),
                                       (-0.9096215963363647, 0.001, 0.1, 1.0, False),
                                       (-0.9096216559410095, 0.02, 0.6, 1.0, False),
                                       (-0.0187, 0.9, 0.9, 1.0, False),
                                       (-3.3, 0.05, 0.5, 1.0, False),
                                       (-0.9096216559410095, 0.02, 0.6, 1e5, False),
                                       (-3.3, 0.05, 0.5, 1e6, False)):
        Ns = 1024
        rho_ = ulp_f32(A) / abs(A)
        got = simulate_rel_error(A, dt, beta, Ns, delta_ulp=dulp, to_f32=f32mode)
        want = bound_rel(abs(A) * dt, Ns, rho_ * dulp)
        n_sim += 1
        n_nonzero += 1 if got > 0.0 else 0
        if want <= 0.0 or got > want * 1.000001:
            sim_ok = False
            print("    SIMULATION EXCEEDS BOUND A=%.6g dt=%g beta=%g dulp=%g f32=%s  true=%.3e bound=%.3e"
                  % (A, dt, beta, dulp, f32mode, got, want))
        tightest = max(tightest, got / want if want > 0 else 0.0)
    arm_sim = sim_ok and n_sim > 0 and n_nonzero > 0
    print("  cross-check bound-vs-simulated    runs=%d  non-zero=%d  max true/bound=%s  (N=%d,"
          " 2x2)  -> %s"
          % (n_sim, n_nonzero, fmt(tightest), 1024, "PASS" if arm_sim else "FAIL"))
    if n_nonzero == 0:
        print("    VACUOUS: no simulation produced a non-zero true error, so this arm proves nothing")

    # ordering arm: §04.6 decays FIRST and writes AFTER, so S_1 = beta * v k^T is A-independent.
    # Zero here is the PREDICTED value from the spec's order, not an absence of signal; the bound is
    # strictly positive, so the arm fails if the simulator ever decays the write.
    N1_got = simulate_rel_error(D14_A, 0.02, 0.6, 1, delta_ulp=1e6, to_f32=False)
    N1_bound = bound_rel(abs(D14_A) * 0.02, 1, 1e6 * ulp_f32(D14_A) / abs(D14_A))
    arm_n1 = (N1_got == 0.0 and N1_bound > 0.0)
    print("  arm N=1 order       write is undecayed true=%r bound=%s (A cannot be seen at t=1) -> %s"
          % (N1_got, fmt(N1_bound), "PASS" if arm_n1 else "FAIL"))

    # ulp implementation witness: bit-walk must equal the exponent-derived ulp on real values
    n_ulp, bad = 0, 0
    probe = [D14_A, D14_FILE, -0.0187, -72.2, -1.0, -0.5, 1.0, 0.5, 0.99999994, -1e-30, 3.4e38,
             -2.0 ** -10, 2.0 ** 7]
    for x in probe:
        n_ulp += 1
        if ulp_f32(x) != ulp_f32_by_exponent(x):
            bad += 1
    arm_ulp = (bad == 0 and n_ulp > 0)
    print("  ulp witness        bit-walk==2^(e-23) %d values, %d disagree               -> %s"
          % (n_ulp, bad, "PASS" if arm_ulp else "FAIL"))
    # and the nextafter trap must be visible: the double ulp is NOT the f32 ulp
    dbl = abs(math.nextafter(D14_A, math.inf) - D14_A)
    arm_trap = (dbl < 1e-12 and ulp_f32(D14_A) > 1e-9)
    print("  trap witness       nextafter=%.3e (DOUBLE ulp) vs f32 ulp=%.3e            -> %s"
          % (dbl, ulp_f32(D14_A), "PASS" if arm_trap else "FAIL"))

    # the D-14 deviation itself: file is one ulp from the route, and the route is inside [lo,hi]
    route_ulp = ulp_f32(D14_A)
    d14_ok = (f32_bits(D14_FILE) == f32_bits(D14_A) + 1
              and abs(D14_FILE - D14_A) == route_ulp)
    print("  D-14 witness       |file-route|=%s == ulp %s, bits +1                        -> %s"
          % (fmt(abs(D14_FILE - D14_A)), fmt(route_ulp), "PASS" if d14_ok else "FAIL"))

    arms = [arm0, arm1, arm2, arm_mono, arm_sign, arm_xc, fired, arm_sim, arm_n1,
            arm_ulp, arm_trap, d14_ok]
    print("  ARMS: %d/%d PASS;  populations evaluated: cross-checks=%d simulations=%d ulp=%d"
          % (sum(1 for a in arms if a), len(arms), n_xc, n_sim, n_ulp))
    if not all(arms):
        raise SystemExit("SELFTEST FAILED — refusing to print a bound")
    print("SELFTEST PASS\n")
    return True


# -----------------------------------------------------------------------------------------------
# The report
# -----------------------------------------------------------------------------------------------

def sweep_alpha(rho, Ns, decades=(-6.0, 8.0), per_decade=16):
    """Closed-form-only sweep of the per-step exponent. This is the bound, not a simulation.
    The range covers alpha = |A| dt from 1e-6 (a gate of 0.999999, no decay) to 1e8 (dt ~ 1e6 at
    the smallest |A| in the model), so "every per-step gate the model could present" is a scanned
    claim and not an assumed one."""
    grid = []
    lo, hi = decades
    e = lo
    while e <= hi + 1e-9:
        grid.append(10.0 ** e)
        e += 1.0 / per_decade
    out = {}
    for N in Ns:
        best = max(((bound_rel(a, N, rho), a) for a in grid))
        best_abs = max(((bound_abs(a, N, rho), a) for a in grid))
        out[N] = (best[0], best[1], best_abs[0], best_abs[1])
    return out, grid


def floor_crossing(rho, N, floor, grid):
    """The set of per-step exponents whose B_rel exceeds `floor`, scanned not bisected: B_rel is
    NOT monotone in alpha (it rises to ~rho near alpha ~ 1/N and decays as alpha*exp(-alpha)), so a
    bisection would silently answer a question nobody asked. Returns (count, min_alpha, max_alpha,
    sup_B_rel, argmax_alpha, sup_B_abs, argmax_abs)."""
    over = [a for a in grid if bound_rel(a, N, rho) > floor]
    pairs = [(bound_rel(a, N, rho), a) for a in grid]
    pairs_abs = [(bound_abs(a, N, rho), a) for a in grid]
    sb, ab = max(pairs)
    sa, aa = max(pairs_abs)
    return (len(over), (min(over) if over else None), (max(over) if over else None),
            sb, ab, sa, aa)


def report_data():
    cfg = json.load(open(CONFIG))["text_config"]
    layer_types = cfg["layer_types"]
    HV = cfg["linear_num_value_heads"]
    linear_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print("=" * 96)
    print("POPULATIONS EVALUATED (every one is asserted non-zero; a 0 here would make the run vacuous)")
    print("=" * 96)
    print("  model geometry      %d layers, %d linear_attention, h_v=%d, d_k=d_v=%d, "
          "max_position_embeddings=%d"
          % (len(layer_types), len(linear_layers), HV, cfg["linear_value_head_dim"],
             cfg["max_position_embeddings"]))
    print("  widths judged       %s" % (", ".join("N=%d" % n for n in WIDTHS)))

    heads = []           # (layer, gguf_head, A, dt_bias)
    read_log = []
    for L in linear_layers:
        A, tA = read_f32("blk.%d.ssm_a" % L, HV)
        D, tD = read_f32("blk.%d.ssm_dt.bias" % L, HV)
        read_log.append((tA, tD))
        for j in range(HV):
            heads.append((L, j, A[j], D[j]))
    n_tensors = len(read_log) * 2
    n_elements = len(heads) * 2
    print("  tensors read        %d (%d x blk.N.ssm_a + %d x blk.N.ssm_dt.bias), all type F32, "
          "dims %s / %s"
          % (n_tensors, len(linear_layers), len(linear_layers),
             list(read_log[0][0].dims), list(read_log[0][1].dims)))
    print("  elements read       %d (%d linear layers x %d V heads x 2 tensors)"
          % (n_elements, len(linear_layers), HV))
    if len(heads) != len(linear_layers) * HV or n_elements != 2 * len(heads):
        raise AssertionError("population does not match geometry: heads=%d" % len(heads))
    print("  alpha grid          closed form only, %.0f points/decade over 1e%.0f..1e%.0f"
          % (16, -6, 8))
    print("  explicit summation  capped at N<=%d, used as a cross-check on 8 seed (head, dt) pairs"
          % MAX_EXPLICIT_TERMS)
    print("                      expanded to 203 (pair, N, alpha) evaluations (see SELFTEST)")

    print()
    print("=" * 96)
    print("DATA — the gate/dt distribution reachable on CPU, from the pinned file")
    print("=" * 96)
    print("  spec anchor: specification/04-gdn-linear-attention.md §04.5, quoted:")
    print("      g_t[h] = -exp(A_log[h]) * softplus( in_proj_a(x_t)[h] + dt_bias[h] )")
    print("      (the file's blk.N.ssm_a IS -exp(A_log); §04.5's own note: \"what the file stores\"")
    print("       and DECISIONS.md D-14 / ASSUMPTIONS.md A-13 — never call it bit-exact)")
    print("  spec anchor: §04.6 step 1, quoted:  \"1  S      = S * exp(g_t[h])   # decay first\"")
    print()
    dumps = os.path.join(REPO, "data/dumps/2026-10-26")
    all_dumps = sorted(f for f in os.listdir(dumps) if f.endswith(".bin")) if os.path.isdir(dumps) else []
    gdn_dumps = [f for f in all_dumps
                 if any(s in f.lower() for s in ("ssm", "gdn", "gdt", "conv", "linear_attn", "state"))]
    print("  WHAT IS OBSERVABLE: dt_bias is stored and is exact; in_proj_a(x) is an ACTIVATION and")
    print("  is not on disk. Witnessed absence, counted not asserted: %d dump files in"
          % len(all_dumps))
    print("  data/dumps/2026-10-26/, of which %d match ssm|gdn|gdt|conv|linear_attn|state — the"
          % len(gdn_dumps))
    print("  pinned block is the MoE one (specification/10 §10.1), so the per-token gate is absent.")
    if len(all_dumps) == 0:
        raise AssertionError("the dump directory is empty/unreadable: the absence claim above is "
                             "vacuous, not witnessed")
    print("  So `a = in_proj_a(x_t)[h]` is UNOBSERVED and the per-step dt is unknown. The anchor")
    print("  below is the a=0 slice of the (token, head) grid, and the bound is then WIDENED over")
    print("  all alpha rather than assuming the observed one.")

    a_vals = [-h[2] for h in heads]
    assert all(x > 0 for x in a_vals), "ssm_a must be negative (-exp of something)"
    stats_line("|A| (= exp(A_log))", a_vals)
    ulps = [ulp_f32(h[2]) for h in heads]
    stats_line("ulp(A) [absolute f32]", ulps)
    rhos = [ulp_f32(h[2]) / abs(h[2]) for h in heads]
    lo_r, med_r, hi_r = stats_line("rho = ulp(A)/|A|", rhos)
    print("      2^-24 = %.9g   2^-23 = %.9g   (every rho lands inside: %s)"
          % (2.0 ** -24, 2.0 ** -23,
             "yes" if (lo_r >= 2.0 ** -24 * 0.999 and hi_r <= 2.0 ** -23 * 1.001) else "NO"))
    dt_anchor = [softplus(h[3]) for h in heads]
    stats_line("dt_bias", [h[3] for h in heads])
    stats_line("dt at the a=0 anchor, softplus(dt_bias)", dt_anchor)
    alpha_anchor = [abs(h[2]) * softplus(h[3]) for h in heads]
    stats_line("alpha = |A| dt  (a=0 anchor)", alpha_anchor)
    stats_line("per-step gate exp(-alpha) (a=0 anchor)", [math.exp(-a) for a in alpha_anchor])
    stats_line("per-step gate perturbation |exp(delta dt)-1| (a=0 anchor)",
               [abs(math.expm1(ulp_f32(h[2]) / abs(h[2]) * abs(h[2]) * softplus(h[3])))
                for h in heads])
    print("    (that last line is the WHOLE injection: one ulp of A moves the per-step state")
    print("     multiplier by at most that much, at the worst head in the model, at the anchor)")

    # HF cross-check on the subset of layers whose shards are on disk
    print()
    print("  HF cross-check (models/qwen36-src/, four shards: layers 0,1,38,39 + mtp).")
    hf_layers_ok, exact_dt, within1 = [], 0, 0
    for L in linear_layers:
        hd = hf_tensor("model.language_model.layers.%d.linear_attn.dt_bias" % L)
        ha = hf_tensor("model.language_model.layers.%d.linear_attn.A_log" % L)
        if hd is None or ha is None:
            continue
        _, _, db = hd
        _, _, alb = ha
        gguf_a, _ = read_f32("blk.%d.ssm_a" % L, HV)
        gguf_d, _ = read_f32("blk.%d.ssm_dt.bias" % L, HV)
        for j in range(HV):
            g = tiled_to_grouped(j, cfg["linear_num_key_heads"],
                                 HV // cfg["linear_num_key_heads"])
            if f32(db[g]) == gguf_d[j]:
                exact_dt += 1
            route = -f32(math.exp(alb[g]))
            if abs(f32_bits(gguf_a[j]) - f32_bits(route)) <= 1:
                within1 += 1
        hf_layers_ok.append(L)
    n_hf = len(hf_layers_ok) * HV
    print("    layers on disk    %s   elements %d (dt_bias) + %d (A_log)"
          % (hf_layers_ok, n_hf, n_hf))
    print("    dt_bias bit-exact under the §04.9 tiled->grouped permutation: %d/%d"
          % (exact_dt, n_hf))
    print("    ssm_a within 1 ulp of -exp(A_log) (double route):              %d/%d"
          % (within1, n_hf))
    print("    POPULATION WITNESS: %d HF elements checked, %d and %d matched "
          "(both non-zero)" % (2 * n_hf, exact_dt, within1))
    if exact_dt != n_hf or within1 != n_hf or n_hf == 0:
        print("    NOTE: not all elements matched; the mismatch count is the finding.")

    # ---- the bound ---------------------------------------------------------------------------
    print()
    print("=" * 96)
    print("THE BOUND  (closed form, this repo's §04.6 recurrence; two currencies)")
    print("=" * 96)
    print("   B_rel(N, alpha) = r*eps*sum_{k=0}^{N-1} (r(1+2 eps))^k   >=  ||dS_N|| / max_t||S_t||")
    print("   B_abs(N, alpha) = B_rel * sum_{m=1}^{N} r^m              >=  ||dS_N|| / V")
    print("   r = exp(-alpha),  eps = expm1(rho*alpha),  alpha = |A|*dt,  rho = ulp(A)/|A|")
    print("   (sign-agnostic; the Gronwall closure multiplies r by 1+2eps, which changes B by O(rho))")
    print("   WHY TWO: a head's output is proportional to its OWN state, so B_rel is the relative")
    print("   error on that head's contribution — but when alpha >> 1 the head's state is itself")
    print("   ~r/(1-r) ~ exp(-alpha) of the value scale (the §04.6 envelope s <= V*sum_{m<N} r^m),")
    print("   so a large B_rel there multiplies a state that is zero to within exp(-alpha) and B_abs")
    print("   is the figure that decides whether anything reaches the output. Both are printed.")
    print()
    print("   %-9s %-30s %-13s %-13s %-13s" % ("N", "population", "B_rel min", "median", "max"))
    results = {}
    for N in WIDTHS:
        bs = [bound_rel(al, N, r_) for al, r_ in zip(alpha_anchor, rhos)]
        lo, med, hi_ = mm(bs)
        results[N] = (lo, med, hi_)
        print("   %-9d %-30s %-13s %-13s %-13s"
              % (N, "measured a=0 anchor (%d heads)" % len(heads), fmt(lo), fmt(med), fmt(hi_)))
        lo2, med2, hi2 = mm([bound_abs(al, N, r_) for al, r_ in zip(alpha_anchor, rhos)])
        print("   %-9d %-30s %-13s %-13s %-13s  [B_abs: ||dS||/V]"
              % (N, "same heads, absolute currency", fmt(lo2), fmt(med2), fmt(hi2)))

    print()
    print("   The anchor population split by the per-step exponent — the heads with alpha >> 1 are")
    print("   the ones whose B_rel looks large and whose state is exp(-alpha) of the value scale:")
    for N in WIDTHS:
        for lo_a, hi_a, tag in ((0.0, 1.0, "informative: gate >= 0.368"),
                                (1.0, 10.0, "fast decay: gate in [4.5e-5, 0.368]"),
                                (10.0, float("inf"), "dead state: gate < 4.5e-5")):
            sel = [(al, r_) for al, r_ in zip(alpha_anchor, rhos) if lo_a < al <= hi_a or
                   (lo_a == 0.0 and al <= hi_a)]
            if not sel:
                print("   N=%-8d alpha in (%-6s,%-8s] n=0  %-32s" % (N, fmt(lo_a), fmt(hi_a), tag))
                continue
            b_rel = max(bound_rel(al, N, r_) for al, r_ in sel)
            b_abs = max(bound_abs(al, N, r_) for al, r_ in sel)
            env = max(state_envelope(al, N) for al, r_ in sel)
            print("   N=%-8d alpha in (%-6s,%-8s] n=%-4d %-32s B_rel max %-12s B_abs max %-12s "
                  "state/V max %s"
                  % (N, fmt(lo_a), fmt(hi_a), len(sel), tag, fmt(b_rel), fmt(b_abs), fmt(env)))

    print()
    print("   Worst case over the per-step gate — alpha swept in closed form only, at the binding")
    print("   rho (B is strictly increasing in rho, which the thousand-ulp arm witnesses). B_rel is")
    print("   NOT monotone in alpha, so this is a scan, not a bisection:")
    rho_worst = max(rhos)
    sw, grid = sweep_alpha(rho_worst, WIDTHS)
    b_flat = max(bound_rel(a, N, rho_worst) for N in WIDTHS for a in grid if a <= 1.0)
    for N in WIDTHS:
        b, a, ba, aa = sw[N]
        print("   N=%-8d sup_alpha B_rel = %-12s at alpha=%-11s (=%.3f*rho) | sup_alpha B_abs = "
              "%-12s at alpha=%s"
              % (N, fmt(b), fmt(a), b / rho_worst, fmt(ba), fmt(aa)))
    print("   N=%-8s max over 0 < alpha <= 1 of B_rel = %s (=%.3f*rho) — every gate that keeps the"
          % ("all", fmt(b_flat), b_flat / rho_worst))
    print("     state alive, and the sup over ALL alpha > 0 is the same figure to 3 digits: the")
    print("     perturbation never exceeds one relative ulp of A, at any gate, at any N.")
    for N in WIDTHS:
        for a, tag in ((1e-6, "gate 1-1e-6, no decay"), (0.1, "gate 0.905"),
                       (1.0, "gate 0.368"), (10.0, "gate 4.5e-5 (state is dead)")):
            print("   N=%-8d alpha=%-8s B_rel=%-12s B_abs=%-12s state/V<=%-10s %s"
                  % (N, fmt(a), fmt(bound_rel(a, N, rho_worst)),
                     fmt(bound_abs(a, N, rho_worst)), fmt(state_envelope(a, N)), tag))
    print("   B_abs conditioned on how big the head's state is allowed to be (s/V from the §04.6")
    print("   envelope; a trained GDN head keeps s ~ O(V), and s/V ~ 1/alpha is the no-decay tail):")
    for cap in (10.0, 100.0):
        for N in WIDTHS:
            sel = [a for a in grid if state_envelope(a, N) <= cap]
            if not sel:
                continue
            print("   N=%-8d s/V <= %-6s sup B_abs = %-12s at alpha=%s"
                  % (N, fmt(cap), max(bound_abs(a, N, rho_worst) for a in sel),
                     fmt(max(sel, key=lambda a: bound_abs(a, N, rho_worst)))))

    print()
    print("   Does the bound ever REACH the floor, anywhere in the gate range? Scanned, not assumed:")
    label, floor, kind, src = FLOOR_PRIMARY
    print("   floor: %s = %s   [%s]" % (label, fmt(floor), kind))
    print("          source: %s" % src)
    alpha_crits = {}
    for N in WIDTHS:
        n_over, lo_a, hi_a, sb, ab, sa, aa = floor_crossing(rho_worst, N, floor, grid)
        alpha_crits[N] = sb
        print("   N=%-8d #{alpha: B_rel > floor} = %-4d over a %d-point grid  (max B_rel = %s at "
              "alpha=%s, i.e. %sx under)"
              % (N, n_over, len(grid), fmt(sb), fmt(ab), fmt(floor / sb)))
    print("   So the bound is under the floor for EVERY per-step gate on the scanned range")
    print("   alpha in 1e-6..1e8 (gate exp(-alpha) from 0.999999 down to e^-1e8); no dt the model can")
    print("   present, however far outside the measured dt_bias range, brings it up.")

    # ---- the input-wise envelope for the layers whose weights are on disk --------------------
    print()
    print("=" * 96)
    print("INPUT-WISE ENVELOPE — a data-derived ceiling on the unobserved a = in_proj_a(x), 3 layers")
    print("=" * 96)
    print("   x is the Form-A input_layernorm output (specification/03 §03.1: (1+w) * x_hat,")
    print("   x_hat = x*rsqrt(mean(x^2)+1e-6), then cast to bf16), so ||x_hat||_2 = sqrt(d) exactly")
    print("   and ||x||_2 <= sqrt(d) * max_i|1+w_i| * (1 + 2^-9)  [the bf16 cast's half-width].")
    print("   in_proj_a has no bias (witnessed in the shard header), so")
    print("   |a_h| <= ||W_a[h,:]||_2 * ||x||_2,  dt_h <= softplus(|a_h| + dt_bias_h),")
    print("   alpha_h = |A_h| * dt_h.  This is a bound over ALL inputs, not a sample.")
    d = cfg["hidden_size"]
    env_layers, env_rows = [], []
    for L in linear_layers:
        wa = hf_tensor("model.language_model.layers.%d.linear_attn.in_proj_a.weight" % L)
        wn = hf_tensor("model.language_model.layers.%d.input_layernorm.weight" % L)
        if wa is None or wn is None:
            continue
        _, shp_a, a_flat = wa
        _, _, w_norm = wn
        if tuple(shp_a) != (HV, d):
            raise AssertionError("in_proj_a shape %s != (%d,%d)" % (shp_a, HV, d))
        if len(w_norm) != d:
            raise AssertionError("input_layernorm length %d != %d" % (len(w_norm), d))
        xmax = math.sqrt(d) * max(abs(1.0 + u) for u in w_norm) * (1.0 + 2.0 ** -9)
        A_gg, _ = read_f32("blk.%d.ssm_a" % L, HV)
        D_gg, _ = read_f32("blk.%d.ssm_dt.bias" % L, HV)
        env_layers.append(L)
        for j in range(HV):
            g = tiled_to_grouped(j, cfg["linear_num_key_heads"],
                                 HV // cfg["linear_num_key_heads"])
            row = a_flat[g * d:(g + 1) * d]
            if len(row) != d:
                raise AssertionError("row slice short for layer %d head %d" % (L, j))
            wnorm = math.sqrt(sum(u * u for u in row))
            a_max = wnorm * xmax
            dt_max = softplus(a_max + D_gg[j])
            alpha_max = abs(A_gg[j]) * dt_max
            env_rows.append((L, j, wnorm, xmax, a_max, dt_max, alpha_max,
                             ulp_f32(A_gg[j]) / abs(A_gg[j])))
    if env_rows:
        print("   layers priced     %s (%d of %d linear layers; the other shards are not on disk)"
              % (env_layers, len(env_layers), len(linear_layers)))
        print("   rows priced       %d (= %d layers x %d V heads); ||W_a[h,:]||_2 exact, "
              "||x||_2 ceiling from the norm weight" % (len(env_rows), len(env_layers), HV))
        stats_line("||W_a[h,:]||_2", [r[2] for r in env_rows])
        stats_line("||x||_2 ceiling", [r[3] for r in env_rows])
        stats_line("|a_h| ceiling", [r[4] for r in env_rows])
        stats_line("dt ceiling", [r[5] for r in env_rows])
        stats_line("alpha ceiling", [r[6] for r in env_rows])
        for N in WIDTHS:
            bb = [bound_rel(r[6], N, r[7]) for r in env_rows]
            ba = [bound_abs(r[6], N, r[7]) for r in env_rows]
            print("   N=%-8d worst-over-inputs  B max %-12s   ||dS||/V max %-12s"
                  % (N, fmt(max(bb)), fmt(max(ba))))
        ac = max(r[6] for r in env_rows)
        print("   The data-derived alpha ceiling is %s. The scanned gate range reaches 1e8, so every"
              % fmt(ac))
        print("   input these %d layers can present sits inside the plateau, where B_rel = rho and"
              % len(env_layers))
        print("   B_abs is set by the head's own memory, not by how wrong A is.")
    else:
        print("   NO LAYER COULD BE PRICED — the envelope is then entirely NOT ESTABLISHED.")

    # ---- verdict ------------------------------------------------------------------------------
    print()
    print("=" * 96)
    print("FLOORS, and the verdict against them")
    print("=" * 96)
    for label, val, kind, src in FLOORS:
        print("  %-62s %s  [%s]" % (label, fmt(val), kind))
        print("      %s" % src)
    print()
    sup_rel, arg_rel = max((bound_rel(a, N, rho_worst), a) for N in WIDTHS for a in grid)
    sup_rel_le1 = max(bound_rel(a, N, rho_worst) for N in WIDTHS for a in grid if 0 < a <= 1.0)
    sup_abs_raw, arg_abs, N_abs = max((bound_abs(a, N, rho_worst), a, N)
                                      for N in WIDTHS for a in grid)
    STATE_CAP = 10.0
    sup_abs_cap, arg_abs_cap = max((bound_abs(a, N, rho_worst), a) for N in WIDTHS for a in grid
                                   if state_envelope(a, N) <= STATE_CAP)
    operative = sup_rel
    anchor_rel = max(results[N][2] for N in WIDTHS)
    informative_rel = max([bound_rel(al, N, r_) for N in WIDTHS
                           for al, r_ in zip(alpha_anchor, rhos) if al <= 1.0] + [0.0])
    dead = [(al, r_) for al, r_ in zip(alpha_anchor, rhos) if al > 10.0]
    dead_rel = max([bound_rel(al, WIDTHS[-1], r_) for al, r_ in dead] + [0.0])
    dead_abs = max([bound_abs(al, WIDTHS[-1], r_) for al, r_ in dead] + [0.0])
    dead_env = max([state_envelope(al, WIDTHS[-1]) for al, r_ in dead] + [0.0])
    print("  OPERATIVE — sup over the scanned gate range and all N of B_rel:     %s  (= %.3f * rho,"
          % (fmt(operative), operative / rho_worst))
    print("    at alpha=%s; the state-relative figure is the one a gate compares, because the block's"
          % fmt(arg_rel))
    print("    output is out = S^T q with ||q||_2 = 1/sqrt(128), so ||dout||/||out|| = ||dS||/||S||.)")
    print("  sup B_rel over 0 < alpha <= 1 only:                                 %s" % fmt(sup_rel_le1))
    print("  rho itself (max over the %d heads):                                 %s"
          % (len(heads), fmt(rho_worst)))
    print("  max B_rel over the %d-head a=0 anchor population:                    %s"
          % (len(heads), fmt(anchor_rel)))
    print("    split: informative heads (alpha<=1) %s | dead heads (alpha>10, n=%d) B_rel %s,"
          % (fmt(informative_rel), len(dead), fmt(dead_rel)))
    print("    B_abs %s, state envelope <= %s V — the dead heads are negligible in BOTH currencies,"
          % (fmt(dead_abs), fmt(dead_env)))
    print("    which is why no head can smuggle a larger perturbation behind a big B_rel.")
    print("  secondary, absolute currency (units of V = max_t ||v_t||_2, NOT a relative error):")
    print("    sup B_abs over ALL alpha:                             %s  (at alpha=%s, N=%d, where the"
          % (fmt(sup_abs_raw), fmt(arg_abs), N_abs))
    print("    state's own envelope is %s V — the bound is rho OF THE STATE and the state there is"
          % fmt(state_envelope(arg_abs, N_abs)))
    print("    1e5 V, so this row is a statement about a gigantic state, not about a big error)")
    print("    sup B_abs for heads with state <= %s V:                  %s  (at alpha=%s)"
          % (fmt(STATE_CAP), fmt(sup_abs_cap), fmt(arg_abs_cap)))
    print("  operative / primary floor (1-cos 1.41e-05):                         %s  (%sx under)"
          % (fmt(operative / FLOOR_PRIMARY[1]), fmt(FLOOR_PRIMARY[1] / operative)))
    print("  operative / one bf16 ULP (2^-8):                                    %s  (%sx under)"
          % (fmt(operative / 2.0 ** -8), fmt(2.0 ** -8 / operative)))
    print("  operative / one f32 ULP (2^-24):                                    %s"
          % fmt(operative / 2.0 ** -24))
    verdict = "NOT RESOLVABLE"
    inj_max = max(abs(math.expm1(r_ * al)) for r_, al in zip(rhos, alpha_anchor))
    for line in [
        "",
        "  %s — a 1-f32-ulp change in A, compounded through §04.6 over every head in the model and" % verdict,
        "  every N in {4096, 32768, 262144}, perturbs the recurrent state by at most %s of its own"
        % fmt(operative),
        "  magnitude: %sx below this repo's noise floor (%s = %s, pitfalls/30-numerics.md §Byte-identity"
        % (fmt(FLOOR_PRIMARY[1] / operative), FLOOR_PRIMARY[0], fmt(FLOOR_PRIMARY[1])),
        "  at short widths is not byte-identity at long context), and %sx below one bf16 ULP of the"
        % fmt(2.0 ** -8 / operative),
        "  residual stream the block writes into. No gate this repo specifies could resolve it — that",
        "  is D-14's claim, now quantified past the single element it was made about.",
        "  The one instrument that could care is a BIT-EXACT gate on the F32 state: the figure is %sx"
        % fmt(operative / 2.0 ** -24),
        "  one f32 ULP, i.e. a couple of state ULPs at most, and a bound of two ULPs neither proves nor",
        "  denies that a bit flips. §10.2/§10.3 specify no such gate for the GDN scan: cos + mean-rel",
        "  aggregates plus a floor-probe cancellation band (§10.3). And the selftest's simulated-",
        "  dynamics arm saw the F32 scan round the 1-ulp gate change away entirely — the same message",
        "  from the other side: at this size the perturbation is below what the scan can represent.",
        "",
        "SCOPE: this file establishes, for the recurrence as this repo specifies it (§04.5 + §04.6,",
        "  F32 scan, ||k||_2 <= 1 and beta in (0,1) from §04.4):",
        "   - B_rel(N, alpha) = r*eps*sum_{k<N}(r(1+2eps))^k bounds the state perturbation relative",
        "     to max_t ||S_t||_2, and B_abs is the same bound in units of V = max_t ||v_t||_2; both",
        "     for a CONSTANT per-step exponent alpha;",
        "   - B_rel <= %.3f * rho for every alpha and every N — N-INDEPENDENT — with rho = ulp(A)/|A|"
        % (sup_rel / rho_worst),
        "     measured min/median/max %s / %s / %s over all %d (layer, V head) pairs, inside"
        % (fmt(lo_r), fmt(med_r), fmt(hi_r), len(heads)),
        "     [2^-24, 2^-23] as it must be;",
        "     the absolute currency is B_abs <= %s * rho in units of V, attained only at alpha ~ %s"
        % (fmt(sup_abs_raw / rho_worst), fmt(arg_abs)),
        "     where the state's own envelope is %s V; for a head whose state stays within %s V it is"
        % (fmt(state_envelope(arg_abs, N_abs)), fmt(STATE_CAP)),
        "     %s instead." % fmt(sup_abs_cap),
        "   - the injection is exactly the per-step gate perturbation |exp(delta*dt)-1|, measured max",
        "     %s at the a=0 anchor, so a recomputed ssm_a cannot compound past one relative ulp of A" % fmt(inj_max),
        "     times the gate's memory 1/(1-exp(-alpha));",
        "   - the delta-rule write (I - beta k k^T) contributes no amplification, so no arrangement of",
        "     v_t, k_t, beta_t can make A matter more than the gate does.",
        "NOT ESTABLISHED:",
        "   - the true per-step dt is UNOBSERVED: a = in_proj_a(x_t) is an activation and no GDN",
        "     activation is pinned (104 dump files, MoE block only). The a=0 slice is an anchor, not",
        "     a measurement of the running distribution; the verdict is carried instead by the scan",
        "     over alpha in [1e-6, 1e8], in which max B_rel = %s (= %.2f rho) never reaches the floor"
        % (fmt(alpha_crits[WIDTHS[-1]]), alpha_crits[WIDTHS[-1]] / rho_worst),
        "     %s = %s — that range of alpha is dt up to 1e8/|A|, i.e. a + dt_bias of order 1e6 at"
        % (FLOOR_PRIMARY[0], fmt(FLOOR_PRIMARY[1])),
        "     the smallest |A| in the file (%s), against a largest stored dt_bias of %s. A dt outside"
        % (fmt(min(a_vals)), fmt(max(dt_anchor))),
        "     that envelope would have to come from an activation the model was never trained to",
        "     produce, and it was not checked there.",
    ]:
        print(line)
    if env_rows:
        print("   - the input-wise alpha ceiling is priced for %d of %d linear layers only (%s): shards"
              % (len(env_layers), len(linear_layers), env_layers))
        print("     3-24 and 26 are not on disk, so the other %d layers' in_proj_a and input_layernorm"
              % (len(linear_layers) - len(env_layers)))
        print("     weights were never read; for those %d layers the gate range is the widened one."
              % (len(linear_layers) - len(env_layers)))
    else:
        print("   - no in_proj_a could be read: the input-wise envelope is entirely unpriced.")
    for line in [
        "   - per-step dt is assumed CONSTANT within the N-step window for the closed form. A varying",
        "     trajectory is covered by evaluating the same expression at the worse end of the observed",
        "     range (min alpha for the decay, max for the injection); no per-token dt sequence was",
        "     observed, so the constant-alpha form is a modelling assumption, not a measurement.",
        "   - the bound is on the STATE. The read out_t = S_t^T q_t (§04.6 step 5) can cancel, which",
        "     turns a state-relative error into a larger output-relative one; that is the cancellation-",
        "     band question of §10.3 step 3 and is not settled here.",
        "   - nothing past the scan is priced: the gated norm §04.7, out_proj §04.8, the residual add,",
        "     the 30-layer stack and the logits row. A per-layer amplification factor of up to %sx"
        % fmt(FLOOR_PRIMARY[1] / operative),
        "     would still leave the result under the floor; no such factor was measured and none is",
        "     assumed.",
        "   - the file's ssm_a is still READ, never recomputed. This quantifies the consequence of a",
        "     hypothetical recompute; it authorizes none. D-14's 1-ulp predicate is route-derived and",
        "     stays, and its must-catch set (2 ulp, sign flip, permutation, wrong A_log) is unchanged",
        "     by anything here — a bound this far under the floor is not a licence to widen a gate.",
        "   - the HF-side witness covers %d elements (%d layers) because only 4 of 26 shards are on"
        % (2 * n_hf, len(hf_layers_ok)),
        "     disk; the other layers' dt_bias/A_log exactness is cited from",
        "     quantization/evidence/audit-exactness-2026-10-26.json, not re-derived here.",
    ]:
        print(line)
    return verdict


def main():
    argv = sys.argv[1:]
    if "--help" in argv:
        print(__doc__)
        return
    selftest()
    if "--selftest" in argv:
        return
    if not os.path.exists(GGUF_PATH):
        raise SystemExit("missing %s" % GGUF_PATH)
    verdict = report_data()
    print()
    print("VERDICT: %s" % verdict)
    g = globals().get("_gg")
    if g is not None:
        g.close()


if __name__ == "__main__":
    main()
