"""speccheck/gdn_ref.py — the independent torch-CPU implementation of the
Qwen3.6-35B-A3B GDN block (Gated DeltaNet + short conv), i.e. the token mixer of the 30
`linear_attention` layers. This module is that block's oracle, the counterpart of
`speccheck/moe_ref.py` for the MoE block.

Rules it obeys (same lane rules as `moe_ref.py`):
  * torch-CPU only. No `.cuda()`, no HIP, no device allocation, **no clock is read**. This
    file prints counts and verdicts and cannot produce a timing (`AGENTS.md` §GPU execution).
  * It does not import any HF model class at module scope. The math here is written from
    the geometry in `checkpoints/Qwen3.6-35B-A3B.config.json`, the tensor names in
    `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` and the spec files named below;
    the installed `transformers` reference is imported lazily inside the `--cross-check-hf`
    and `--capability` drivers, which are the only sections allowed to know about it.
  * Every non-obvious behaviour carries `[SPEC]` (derived from the committed config or
    geometry), `[REF]` (a form choice the pinned reference takes, recorded so the choice is
    visible rather than inherited) or `[ASSUMPTION A-nn]` (our reading — `specification/
    ASSUMPTIONS.md` owns those). Same vocabulary as `speccheck/results.md` §Tag vocabulary.

§The recurrence this implements, and where it is specified
------------------------------------------------------------------------------
Authoritative form: `specification/04-gdn-linear-attention.md` §04.6 (recurrent) and
`specification/04b-gdn-chunked-form.md` §04b.3–§04b.5 (chunked). The recurrent update is
transcribed from §04.6 steps 1–5, quoted here so the file states its own contract:

    1  S          = S * exp(g_t[h])                              # decay first
    2  kv_mem[v]  = sum_k S[k, v] * k_t[k]                       # what the state predicts
    3  delta[v]   = (v_t[v] - kv_mem[v]) * beta_t[h]             # correction x write strength
    4  S[k, v]    = S[k, v] + k_t[k] * delta[v]                  # delta-rule write
    5  out_t[v]   = sum_k S[k, v] * q_t[k]                       # read AFTER the write

with, from §04.4/§04.5,

    beta_t[h] = sigmoid(in_proj_b(x_t)[h])                                   (activation dtype)
    g_t[h]    = -exp(A_log[h]) * softplus(in_proj_a(x_t)[h] + dt_bias[h])    (F32)
    q = q * rsqrt(sum(q*q, -1) + 1e-6);  k = k * rsqrt(sum(k*k, -1) + 1e-6);  q /= sqrt(d_k)

and the chunked rewrite from §04b.3–§04b.5 (`pairwise`, the UT solve, the per-chunk scan),
the short conv from §04.3, the gated norm from §04.7 and `out_proj` from §04.8.

§Why the cross-form agreement is the evidence, not the HF agreement
------------------------------------------------------------------------------
`pitfalls/30-numerics.md` §"A hoisted scalar constant is a different number, and a CPU
mirror cannot see it": a mirror of the reference's formula agrees with it *because* they
share a formula. So agreement with the installed reference on a same-form arm is a
**transcription** check (it catches a misread spec), never an independence proof. What this
oracle offers as evidence instead:

  * **four implementations of one function**: the recurrent scan, the chunked UT-transform
    scan, and two f64 forms that never share an expression with either (`exact`: rank-1
    updates with matvec reads; `exact_matrix`: `S <- exp(g) * (P @ S) + beta k v^T` with
    `P = I - beta k k^T`, a per-token `d_k x d_k` matmul). Cross-form agreement is banded,
    and the band is derived from the floor probe by the triangle inequality (§Bands).
  * **the geometry-derived injection matrix** (`--selftest`), three arms, every sabotage
    watched to fire.
  * the reference's own two triangular-solve branches, both pinned and named (A-16).

Run (CPU-only, PASS/FAIL, no timing):
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        speccheck/gdn_ref.py --selftest

§What this file does, and what it does not do yet
------------------------------------------------------------------------------
Implemented and running: the oracle, the comparators, the band derivation, the three-arm
injection matrix (`--selftest`), and the installed-reference comparison (`--compare-hf`,
which `--selftest` also runs): 19 stages x 2 modes x 2 pinned blocks, every row printed with
`bit_identical / ndiff / maxabs` and with the provenance of the HF-side tensor (`captured`,
`derived-from-captured`, or `reference-side-call`), per-step rows for `y`, `state` and
`conv_window`, seam rows that license the non-`captured` provenances, seven positive controls
per block that must be REPORTED as differences, and an explicit UNVERIFIED list. A sabotage
run (`--sabotage FAULT`) injects the oracle side and FAILS unless the comparison differs.
**Not implemented yet — no flag exists for them, so they cannot print a PASS:** the floor probe
and its evidence file (`--floor-probe`), the cross-form rows of §04b.6 (`--cross-form`), the
capability/dispatch probe (`--capability`) and `--against-dumps` (reading the pinned dumps that
`tools/dump_reference_gdn.py` writes). `aggregates`, `band_from_probe` and `load_band` are
written and exercised by scratch scripts under `tmp/w20_*`, but until their sections exist
no gate credits them. This is recorded here rather than advertised in the usage block
because a flag that cannot fail is worse than a missing one (`pitfalls/40`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import moe_ref as R  # noqa: E402  (read-only reuse: geometry loader, shard reader, fixture,
#                                    Block, ulp_of, set_threads. Forked nothing — pitfalls/40
#                                    §one address expression shared by writer and reader.)

CONFIG_PATH = R.CONFIG_PATH
INDEX_PATH = R.INDEX_PATH
EVIDENCE = HERE / "evidence"

# §Tensor names, read out of the committed index by these suffixes (nine per linear layer,
# 270 over the 30 linear layers — witnessed by `load_gdn_block_weights` + `n_linear_layers`).
SUFFIXES = {
    "qkv": "linear_attn.in_proj_qkv.weight",
    "z": "linear_attn.in_proj_z.weight",
    "a": "linear_attn.in_proj_a.weight",
    "b": "linear_attn.in_proj_b.weight",
    "conv": "linear_attn.conv1d.weight",
    "A_log": "linear_attn.A_log",
    "dt_bias": "linear_attn.dt_bias",
    "norm": "linear_attn.norm.weight",
    "out": "linear_attn.out_proj.weight",
}
# The GGUF names, for manifest rows and for the A-12 head-order list.
GGUF_NAMES = {
    "qkv": "blk.N.attn_qkv.weight", "z": "blk.N.attn_gate.weight",
    "a": "blk.N.ssm_alpha.weight", "b": "blk.N.ssm_beta.weight",
    "conv": "blk.N.ssm_conv1d.weight", "A_log": "blk.N.ssm_a (recomputed, A-13)",
    "dt_bias": "blk.N.ssm_dt.bias", "norm": "blk.N.ssm_norm.weight",
    "out": "blk.N.ssm_out.weight",
}

# [REF] `chunk_size = 64` is `torch_chunk_gated_delta_rule`'s default, not a config field
# (file 04b §04b.2). Registering it here is what makes "this prefill was run at width 64"
# a checkable claim rather than a coincidence, and what the manifest must record (§04b.7).
CHUNK_WIDTH_REF = 64
# [REF] file 04 §04.4: the l2norm eps is 1e-6 added INSIDE the rsqrt of a SUM of squares.
# It is not the RMSNorm eps; the two merely share the value.
L2NORM_EPS = 1e-6

# §The stages, in pipeline order. A stage is a named intermediate a gate can attribute a
# failure to; the names are the file-04 section that defines the quantity.
STAGES = ["mixed_qkv", "z", "b_raw", "a_raw", "conv_pre", "conv_out", "conv_window",
          "q_pre", "k_pre", "v_pre", "beta", "g_decay", "q_hat", "k_hat",
          "scan_out", "core_attn", "state", "normed", "y"]
FLOAT_STAGES = list(STAGES)          # every stage here is numeric (no ids)
DISCRETE_FACTS = ["state_identical", "step_y_moved"]   # computed by torch.equal, not by the
#                                                          differ — see compute_facts()


# ------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class GdnGeometry:
    """Every number the gates derive, read from the committed config. Nothing here is
    typed by hand (`pitfalls/40` §an admission over a fixture list typed from memory)."""

    hidden_size: int
    h_k: int                  # linear_num_key_heads
    h_v: int                  # linear_num_value_heads
    d_k: int                  # linear_key_head_dim
    d_v: int                  # linear_value_head_dim
    k_conv: int               # linear_conv_kernel_dim
    num_hidden_layers: int
    linear_layers: tuple      # indices with layer_types == "linear_attention"
    hidden_act: str
    mamba_ssm_dtype: str
    rms_norm_eps: float
    chunk_width: int = CHUNK_WIDTH_REF
    l2norm_eps: float = L2NORM_EPS
    config_path: str = "checkpoints/Qwen3.6-35B-A3B.config.json"

    @property
    def key_dim(self) -> int:      # [SPEC]
        return self.d_k * self.h_k

    @property
    def value_dim(self) -> int:    # [SPEC]
        return self.d_v * self.h_v

    @property
    def conv_dim(self) -> int:     # [SPEC] key_dim*2 + value_dim
        return self.key_dim * 2 + self.value_dim

    @property
    def expansion(self) -> int:    # [SPEC] h_v / h_k, the grouped K->V repeat
        return self.h_v // self.h_k

    @property
    def n_linear_layers(self) -> int:
        return len(self.linear_layers)

    @property
    def n_gdn_tensors(self) -> int:      # [SPEC] 9 per linear layer
        return 9 * self.n_linear_layers

    @property
    def state_shape(self) -> tuple:      # [SPEC] file 04 §04.10
        return (self.h_v, self.d_k, self.d_v)

    @property
    def state_bytes_f32(self) -> int:
        return self.h_v * self.d_k * self.d_v * 4

    @property
    def conv_window_shape(self) -> tuple:   # [REF] the reference's cache keeps k_conv taps
        return (self.conv_dim, self.k_conv)

    @property
    def scan_dtype(self) -> torch.dtype:    # [SPEC] mamba_ssm_dtype is the SCAN dtype
        return {"float32": torch.float32}[self.mamba_ssm_dtype]

    def degeneracies(self) -> dict:
        """Injections that this geometry cannot adjudicate, with the geometric reason.
        Printed, never silently dropped (`pitfalls/40` §a route no fixture reaches has no
        tests — here the route is unreachable *by construction*, which is a fact to print).
        """
        d = {}
        if self.d_k == self.d_v:
            d["q_scaled_by_dv"] = (f"d_k == d_v == {self.d_k}: scaling q by 1/sqrt(d_v) is "
                                   "the same expression as 1/sqrt(d_k), so the injection "
                                   "cannot be distinguished. Excluded from the must-catch "
                                   "set as degenerate, not as passing.")
        if self.expansion == 1:
            d["head_pairing_tiled"] = (f"h_v == h_k == {self.h_v}: no grouped/tiled pairing "
                                       "distinction exists (file 04 §04.9 head-expansion row)")
            d["v_head_order_tiled"] = "see head_pairing_tiled"
        if self.k_conv < 2:
            d["conv_tap_rotate"] = f"k_conv={self.k_conv}: nothing to rotate"
        return d


def load_gdn_geometry(config_path: Path = CONFIG_PATH) -> GdnGeometry:
    cfg = json.loads(Path(config_path).read_text())
    t = cfg["text_config"]
    linear = tuple(i for i, kind in enumerate(t["layer_types"]) if kind == "linear_attention")
    g = GdnGeometry(
        hidden_size=t["hidden_size"],
        h_k=t["linear_num_key_heads"],
        h_v=t["linear_num_value_heads"],
        d_k=t["linear_key_head_dim"],
        d_v=t["linear_value_head_dim"],
        k_conv=t["linear_conv_kernel_dim"],
        num_hidden_layers=t["num_hidden_layers"],
        linear_layers=linear,
        hidden_act=t["hidden_act"],
        mamba_ssm_dtype=t["mamba_ssm_dtype"],
        rms_norm_eps=t["rms_norm_eps"],
        config_path=str(config_path),
    )
    # Guards: each of these is a spec claim that the code below depends on. If the config
    # drifts, the oracle refuses rather than quietly computing a different block.
    if t["hidden_act"] != "silu":
        raise AssertionError(f"hidden_act={t['hidden_act']!r}: file 04 §04.3 pins SiLU on the "
                             "conv output only")
    if t["mamba_ssm_dtype"] != "float32":
        raise AssertionError(f"mamba_ssm_dtype={t['mamba_ssm_dtype']!r}: file 04 §04.4's scan "
                             "dtype row, and every f32-vs-f64 band in this file, break")
    if g.h_v % g.h_k:
        raise AssertionError(f"h_v={g.h_v} not a multiple of h_k={g.h_k}: the grouped repeat "
                             "of §04.4/§04.9 is undefined")
    if g.d_v != g.d_k:
        raise AssertionError("d_v != d_k: `Qwen3_5MoeRMSNormGated` is constructed over "
                             "head_v_dim and §04.7's variance axis changes")
    interval = t.get("full_attention_interval")
    expect = tuple(i for i in range(g.num_hidden_layers) if (i + 1) % interval != 0)
    if linear != expect:
        raise AssertionError(f"layer_types linear indices {linear[:4]}... disagree with "
                             f"(N+1) %% {interval} != 0 — file 04's header claim")
    return g


# ------------------------------------------------------------------- block + weights
Block = R.Block      # [SPEC] same "layerN"/"mtpN" grammar as the MoE lane


@dataclass
class GdnWeights:
    """The nine weight tensors of one GDN block, bf16, as stored in the checkpoint."""

    block: Block
    tensors: dict
    source_shards: dict = field(default_factory=dict)

    def shapes_ok(self, g: GdnGeometry) -> None:
        want = {
            "qkv": (g.conv_dim, g.hidden_size),
            "z": (g.value_dim, g.hidden_size),
            "a": (g.h_v, g.hidden_size),
            "b": (g.h_v, g.hidden_size),
            "conv": (g.conv_dim, 1, g.k_conv),
            "A_log": (g.h_v,),
            "dt_bias": (g.h_v,),
            "norm": (g.d_v,),
            "out": (g.hidden_size, g.value_dim),
        }
        for role, shape in want.items():
            if role not in self.tensors:
                raise AssertionError(f"block {self.block.name} is missing {role}")
            t = self.tensors[role]
            if tuple(t.shape) != tuple(shape):
                raise AssertionError(f"{self.block.name}.{role}: stored shape "
                                     f"{tuple(t.shape)} != config-derived {tuple(shape)}")
            if t.dtype is not torch.bfloat16:
                raise AssertionError(f"{self.block.name}.{role}: stored dtype {t.dtype}, "
                                     "expected bf16 (dtype-by-filesize, pitfalls/30)")


def load_gdn_block_weights(block: Block, g: GdnGeometry,
                           shard_dir: Path = R.SHARD_DIR) -> GdnWeights:
    """Load the nine tensors of one *linear_attention* block, one at a time, and verify
    their shapes against the config-derived geometry before anything consumes them.

    The block-kind refusal is load-bearing: a `full_attention` layer has no `linear_attn`
    tensors at all (40 layers, 30 of them linear — file 04's header), so asking for layer 3
    is a caller bug and must not be answered with a KeyError from the shard index."""
    if block.kind != "layer":
        raise AssertionError(f"{block.name}: the GDN block exists only in decoder layers; the "
                             "MTP draft block's mixer is file 09's business, not this one")
    if block.index not in g.linear_layers:
        raise AssertionError(f"layer {block.index} is layer_types="
                             f"{json.load(open(REPO / g.config_path))['text_config']['layer_types'][block.index]}"
                             f"; this oracle serves the {g.n_linear_layers} linear_attention "
                             "layers only (file 04's header)")
    index = R.load_index()
    tensors, sources = {}, {}
    for role, suffix in SUFFIXES.items():
        name = block.prefix + suffix
        shard = R.shard_of(name, index)
        path = Path(shard_dir) / shard
        if not path.exists():
            raise AssertionError(f"{name} lives in {shard}, which is not in {shard_dir}; only "
                                 "4 of 26 shards were acquired (README.md §Download)")
        tensors[role] = R.load_tensor(path, name)
        sources[role] = shard
    w = GdnWeights(block, tensors, sources)
    w.shapes_ok(g)
    return w


def gdn_input(n_tokens: int, g: GdnGeometry, seed: int) -> torch.Tensor:
    """[ASSUMPTION A-01, the dump-manifest namespace] The block's input is
    `input_layernorm(hidden)` (file 03 §03.3), so the fixture rows are RMS-1 exactly as in
    `moe_ref.make_input` — and it does NOT carry the `(1+w)` modulation of a real
    activation. Consequence: every agreement figure in this lane is a synthetic-input
    statistic, and the follow-up that makes it a real one needs the dense path.

    `moe_ref.make_input` is reused rather than copied so the GDN and MoE fixtures are the
    same generator under the same seed; it reads only `geometry.hidden_size` off its second
    argument, which is the shim below."""
    return R.make_input(n_tokens, _HiddenShim(g.hidden_size), seed)


@dataclass(frozen=True)
class _HiddenShim:
    hidden_size: int


# ------------------------------------------------------------ forms, faults, fixtures
@dataclass(frozen=True)
class Form:
    """The LEGAL arms. Nothing in here is a bug; the bugs are `Faults`.

    `solve_branch` is A-16: the reference carries two legal UT solves and they are not the
    same number, so a record must name which one made it. `chunk_width` is A-16's sibling
    rule §04b.7: `C` changes the association of the sums, so changing it is a numerics
    change that owes a floor probe, not a tuning knob."""

    scan: str = "chunked"               # chunked|recurrent|exact|exact_matrix|reassoc
    solve_branch: str = "triangular"    # triangular|substitution      [REF] A-16
    chunk_width: int = 0                # 0 => geometry.chunk_width (64) [REF] §04b.2
    q_scale: str = "auto"               # auto|mul|div — see §the q scaling footnote

    @property
    def is_f64(self) -> bool:
        return self.scan in ("exact", "exact_matrix")

    @staticmethod
    def named(arm: str) -> "Form":
        arms = {
            "recurrent": Form(scan="recurrent", q_scale="div"),      # [REF] torch_recurrent
            "chunked": Form(scan="chunked", q_scale="mul"),          # [REF] torch_chunk
            "substitution": Form(scan="chunked", solve_branch="substitution", q_scale="mul"),
            "chunked32": Form(scan="chunked", chunk_width=32, q_scale="mul"),
            "exact": Form(scan="exact"),                              # f64 rank-1
            "exact_matrix": Form(scan="exact_matrix"),                # f64, P @ S per token
            "reassoc": Form(scan="reassoc", q_scale="div"),           # f32, sums split
        }
        if arm not in arms:
            raise AssertionError(f"unknown arm {arm!r}; known: {sorted(arms)}")
        return arms[arm]


@dataclass(frozen=True)
class Faults:
    """Every injection is one field here, default off. The selftest's case list is derived
    from `dataclasses.fields(Faults)`, so an injection cannot be added without entering it
    in the `AFFECTED` contract table too (`AFFECTED` is asserted to cover these fields).

    Each field names the spec line it violates. Any path that produces a pin passes
    `NO_FAULTS`."""

    # §04.2 the packed qkv layout is three contiguous blocks, not interleaved by head
    qkv_split_interleaved: bool = False
    # §04.3 SiLU is after the conv and on all conv_dim channels
    silu_before_conv: bool = False
    # §04.3 the conv equation's tap index, rotated (count = k_conv - 1)
    conv_tap_rotate: int = 0
    # §04.3 consequence 3 / §10.4: the conv state is a window of INPUTS, not outputs
    conv_state_is_outputs: bool = False
    # §04.3 the tap index of W[c, s] * x[t - k_conv + 1 + s]; reversed tap order is the
    # producer-side mirror error (only distinguishable when k_conv >= 3)
    conv_tap_reverse: bool = False
    # §04.4 beta = sigmoid(b)
    beta_no_sigmoid: bool = False
    # §04.5 softplus must be evaluated on the SUM a + dt_bias
    softplus_on_a_only: bool = False
    # §04.5 A_log is a log, the decay is -exp(A_log)
    a_log_used_directly: bool = False
    # §04.5 g <= 0 always; the per-step multiplier is exp(g) in (0, 1]
    decay_sign_flip: bool = False
    # §04.4 l2norm: eps inside the rsqrt of a SUM of squares
    l2norm_eps_outside: bool = False
    l2norm_uses_mean: bool = False
    # §04.4 q scaled by 1/sqrt(d_k), once, before the scan
    q_scale_omitted: bool = False
    q_scaled_by_dv: bool = False                      # degenerate when d_k == d_v
    # §04.9 / A-12 grouped pairing vs tiled V-head order
    head_pairing_tiled: bool = False
    v_head_order_tiled: bool = False
    # §04.6 step 5: the output reads the POST-update state
    read_pre_update_state: bool = False
    # §04.6 steps 1-5 order: decay precedes the write
    decay_after_write: bool = False
    # §04.4 the scan is F32 (mamba_ssm_dtype), not the activation dtype
    scan_in_bf16: bool = False
    # §04.7 the gated norm: plain w*x_hat (no +1), norm before gate, weight applied in the
    # INPUT dtype, variance over d_v
    norm_plus_one: bool = False
    norm_gate_first: bool = False
    norm_weight_f32: bool = False
    norm_variance_flat: bool = False
    # §04b.3-§04b.5 the chunked form
    chunk_decay_off_by_one: bool = False
    pairwise_mask_after_exp: bool = False
    unitriangular_dropped: bool = False
    beta_on_v_not_k: bool = False
    q_decayed_twice: bool = False
    state_stored_transposed: bool = False
    # §04b.2 padding: beta and g are ZERO-padded (a pad column writes nothing and does not
    # decay the state)
    pad_beta_ones: bool = False


NO_FAULTS = Faults()
FAULT_FIELDS = tuple(f.name for f in dataclasses.fields(Faults))


@dataclass(frozen=True)
class Fixture:
    """One pass shape. The counts that decide which injections can fire on it —
    `n_chunks`, `n_pad`, `n_steps` — are derived from the geometry, never typed.

    `n_steps = k_conv - 1` is geometric: the conv window holds `k_conv - 1` columns, so a
    window read corrupted is only *observable* once a previous step has written a column,
    and every column has been rewritten after `k_conv - 1` steps (§10.4's "and every
    subsequent step" is then adjudicated over the whole window)."""

    name: str
    kind: str                 # "prefill" | "decode"
    n_tokens: int
    n_steps: int = 0
    seed: int = 20261026
    # Non-zero = the chunk-padding columns of q/k/v carry a deterministic nonzero fill, the
    # way a kernel that reuses a buffer leaves them. §04b.2 requires padding q/k/v **and**
    # beta/g with zeros; with q/k/v zeroed, `pad_beta_ones` is provably inert (a pad row's
    # write is `k_pad^T @ 0`), so the only fixture on which that §10.4 line can fire is one
    # where the padding is dirty. Measured: inert on all three zero-padded fixtures.
    pad_noise: int = 0

    def n_chunks(self, g: GdnGeometry, width: int) -> int:
        width = width or g.chunk_width
        return -(-self.n_tokens // width)

    def n_pad(self, g: GdnGeometry, width: int) -> int:
        width = width or g.chunk_width
        return (-(-self.n_tokens // width)) * width - self.n_tokens


def default_fixtures(g: GdnGeometry) -> list:
    """The fixture set, sized from the chunk width and the conv kernel — not from memory.

    | fixture | why it must exist |
    |---|---|
    | T=96  | `n_chunks = 2` (chunk-boundary injections are unreachable at 1 chunk) and `n_pad = 32 > 0` (the padding rows of §04b.2 are unreachable at `T % C == 0`) |
    | T=128 | `n_pad = 0`: the zero-padding path must not be *silently load-bearing*, so a run with no padding is required too |
    | T=4   | `n_chunks = 1`, `n_pad = 60`: a single padded chunk, the shape a short prompt takes |
    | decode | prefill 96 then `k_conv - 1` cached steps: the §04b.1 table's recurrent dispatch, and the only shape the conv-state injection can fire on |
    """
    C = g.chunk_width
    return [Fixture("prefill_2chunk_padded", "prefill", C + C // 2),
            Fixture("prefill_2chunk_exact", "prefill", 2 * C),
            # Three chunks, and the third padded: the only shape on which a wrong
            # `chunk_decay` can reach an output at all (see CONDITIONAL_AFFECTED).
            Fixture("prefill_3chunk_padded", "prefill", 2 * C + C // 2),
            Fixture("prefill_1chunk_padded", "prefill", max(1, g.k_conv)),
            # Three chunks AND dirty chunk-padding: the only shape on which §10.4's "pad beta
            # 1 instead of 0" can be seen at all, because with q/k/v zero-padded a pad row's
            # write is `k_pad^T @ 0` whatever beta claims — measured inert on all three
            # zero-padded fixtures.
            Fixture("prefill_3chunk_dirty_pad", "prefill", 2 * C + C // 2, pad_noise=917),
            Fixture("decode_3step", "decode", C + C // 2, n_steps=max(2, g.k_conv - 1))]


# ------------------------------------------------------------- the block, stage by stage
def _l2norm(x: torch.Tensor, eps: float, faults: Faults) -> torch.Tensor:
    """[REF] file 04 §04.4 / `l2norm` in the reference: `x * rsqrt((x*x).sum(-1) + eps)`.
    The eps is inside the rsqrt and the reduction is a SUM, not a mean — both are injections
    here because the two forms are one edit apart and produce plausible tensors."""
    if faults.l2norm_eps_outside:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True)) + eps
    if faults.l2norm_uses_mean:
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _perm_v(t: torch.Tensor, axis: int, g: GdnGeometry) -> torch.Tensor:
    """A-12: read a V-axis as if the file were tiled when it is grouped. `GGUF V head j
    holds HF V head (j % h_k) * (h_v/h_k) + j // h_k` (file 01 §01.6 item 1, file 04 §04.9).
    Head order and pairing must come from one assumption; this is the mirror error, and it
    raises nothing, which is exactly why it is an injection."""
    idx = torch.tensor([(j % g.h_k) * g.expansion + (j // g.h_k) for j in range(g.h_v)])
    return t.index_select(axis, idx)


def projections(x: torch.Tensor, w: GdnWeights, g: GdnGeometry, faults: Faults) -> dict:
    """file 04 §04.2. `x` is [B, T, hidden_size] in the activation dtype."""
    if x.ndim != 3 or x.shape[-1] != g.hidden_size:
        raise AssertionError(f"gdn block expects [B,T,{g.hidden_size}], got {tuple(x.shape)}")
    if x.shape[0] != 1:
        raise AssertionError("only batch 1 is implemented; the engine is single-request")
    mixed_qkv = F.linear(x, w.tensors["qkv"])                      # [B,T,conv_dim]
    # §04.2: `z = in_proj_z(x).view(B, T, h_v, d_v)` — the gate is head-shaped from here on,
    # which is what makes §04.7's variance axis `d_v` and not the flat 4096.
    z = F.linear(x, w.tensors["z"]).reshape(x.shape[0], x.shape[1], g.h_v, g.d_v)
    b_raw = F.linear(x, w.tensors["b"])                            # [B,T,h_v]
    a_raw = F.linear(x, w.tensors["a"])                            # [B,T,h_v]
    if faults.v_head_order_tiled:
        # A-12: the V heads read in tiled/grouped order instead of the checkpoint's paired
        # order. V is the LAST third of the packed 8192 (§04.2), so the split is
        # [q|k, v] — splitting it the other way round permutes q|k and the fault then moves
        # q_pre/k_pre, which the depthwise conv (§04.3) makes impossible for a V-only bug.
        rest, v = mixed_qkv.split([g.conv_dim - g.value_dim, g.value_dim], dim=-1)
        v = _perm_v(v.reshape(x.shape[0], x.shape[1], g.h_v, g.d_v), 2, g).reshape(
            x.shape[0], x.shape[1], g.value_dim)
        mixed_qkv = torch.cat([rest, v], dim=-1)                   # V is the LAST third
        z = _perm_v(z, 2, g)
        a_raw, b_raw = _perm_v(a_raw, -1, g), _perm_v(b_raw, -1, g)
    return {"mixed_qkv": mixed_qkv, "z": z, "b_raw": b_raw, "a_raw": a_raw}


def short_conv(mixed_qkv: torch.Tensor, w: GdnWeights, g: GdnGeometry, faults: Faults,
               *, window: torch.Tensor | None, mode: str) -> dict:
    """file 04 §04.3, both reference paths.

    prefill: `F.conv1d(x, W.unsqueeze(1), padding=k_conv-1, groups=conv_dim)[:, :, :T]`
    then SiLU  [REF: causal_conv1d_fn].
    decode: concatenate the stored window with the new column, keep the last `k_conv`
    columns as the new window, convolve with no padding, keep the last output, then SiLU
    [REF: causal_conv1d_update] — "the same arithmetic as the equation above", which the
    `--cross-form` section adjudicates as its own row.

    `conv_pre` is the conv output BEFORE the activation and `conv_window` is a window of
    **inputs**: pinning the output window instead reproduces a real-looking tensor that is
    the wrong quantity (`pitfalls/30` §a hook that reads `output` can pin the answer)."""
    W = w.tensors["conv"]                                          # [conv_dim,1,k_conv]
    # file 01 §01.6 item 4: the checkpoint stores [conv_dim, 1, k_conv] and the reference
    # squeezes the middle axis (`self.conv1d.weight.squeeze(1)`) before `unsqueeze(1)`.
    # Squeezing here is the transcription; skipping it hands conv1d a 4-D weight.
    if tuple(W.shape) != (g.conv_dim, 1, g.k_conv):
        raise AssertionError(f"conv1d.weight stored {tuple(W.shape)}, expected "
                             f"({g.conv_dim},1,{g.k_conv})")
    W = W.squeeze(1)                                               # [conv_dim, k_conv]
    if faults.conv_tap_rotate:                                     # §04.3 tap correspondence
        W = torch.roll(W, shifts=faults.conv_tap_rotate, dims=-1)
    if faults.conv_tap_reverse:
        W = torch.flip(W, dims=(-1,))
    xin = mixed_qkv.transpose(1, 2)                                # [B,conv_dim,T]
    if faults.silu_before_conv:                                    # §04.3 consequence 1
        xin = F.silu(xin)
    if mode == "decode":
        if xin.shape[-1] != 1:
            raise AssertionError(f"decode conv needs T==1, got {xin.shape[-1]}")
        if window is None or window.shape[-1] != g.k_conv:
            raise AssertionError(f"decode conv needs a [{g.conv_dim},{g.k_conv}] window")
        cat = torch.cat([window, xin], dim=-1).to(W.dtype)         # [B,conv_dim,k_conv+1]
        window_next = cat[:, :, -g.k_conv:]                        # the INPUT window
        conv_in = cat
        keep = -1
        pad = 0
    else:
        if window is not None:
            raise AssertionError("prefill conv takes no window")
        conv_in = xin.to(W.dtype)
        window_next = None
        keep = xin.shape[-1]
        pad = g.k_conv - 1
    conv_pre = F.conv1d(conv_in, weight=W.unsqueeze(1), bias=None,
                        padding=pad, groups=g.conv_dim)
    conv_pre = conv_pre[:, :, :keep] if keep > 0 else conv_pre[:, :, -1:]
    conv_out = F.silu(conv_pre)                                    # [REF] silu, all channels
    return {"conv_pre": conv_pre.to(mixed_qkv.dtype),
            "conv_out": conv_out.to(mixed_qkv.dtype),
            "conv_window": window_next}


def split_heads(conv_out: torch.Tensor, g: GdnGeometry, faults: Faults) -> dict:
    """file 04 §04.2's three contiguous blocks, then §04.2's head reshapes. `q_pre`/`k_pre`
    are still `h_k` heads: the grouping happens in `enter_scan`, so a pairing fault cannot
    hide behind a q_pre that already moved."""
    if faults.qkv_split_interleaved:
        # The other reading of the packed 8192: interleaved per K head, which is what an
        # engine that assumes [h_k, (d_k + d_k + expansion*d_v)] writes. Shape-compatible.
        per = 2 * g.d_k + g.expansion * g.d_v
        view = conv_out.reshape(conv_out.shape[0], conv_out.shape[1], g.h_k, per)
        q = view[..., :g.d_k].reshape(conv_out.shape[0], conv_out.shape[1], g.h_k, g.d_k)
        k = view[..., g.d_k:2 * g.d_k].reshape(conv_out.shape[0], conv_out.shape[1],
                                               g.h_k, g.d_k)
        v = view[..., 2 * g.d_k:].reshape(conv_out.shape[0], conv_out.shape[1], g.h_v, g.d_v)
    else:
        q, k, v = torch.split(conv_out, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], g.h_k, g.d_k)        # [REF] torch.split order
        k = k.reshape(k.shape[0], k.shape[1], g.h_k, g.d_k)
        v = v.reshape(v.shape[0], v.shape[1], g.h_v, g.d_v)
    return {"q_pre": q, "k_pre": k, "v_pre": v}


def decay_and_beta(b_raw: torch.Tensor, a_raw: torch.Tensor, w: GdnWeights,
                   g: GdnGeometry, faults: Faults) -> dict:
    """file 04 §04.4/§04.5. `beta` stays in the activation dtype; `g_decay` is F32 with the
    two explicit casts (`A_log.float()`, `a.float()`) that §04.5 calls out — computing
    softplus in bf16 and upcasting afterwards is a different number."""
    beta = b_raw if faults.beta_no_sigmoid else b_raw.sigmoid()
    a_log = w.tensors["A_log"].float()
    dt = w.tensors["dt_bias"]
    if faults.v_head_order_tiled:
        a_log, dt = _perm_v(a_log, 0, g), _perm_v(dt, 0, g)
    u = a_raw.float() if faults.softplus_on_a_only else a_raw.float() + dt
    # §04.5: `A_log[h]` is a LOG and the decay is `-exp(A_log)`, so `g_t[h] <= 0` always and
    # the per-step multiplier on the state is `exp(g_t[h])` in (0, 1].
    factor = a_log if faults.a_log_used_directly else a_log.exp()
    factor = factor if faults.decay_sign_flip else -factor
    g_decay = factor * F.softplus(u)                               # [REF] §04.5 exactly
    return {"beta": beta, "g_decay": g_decay}


def enter_scan(heads: dict, beta: torch.Tensor, g_decay: torch.Tensor,
               g: GdnGeometry, form: Form, faults: Faults) -> dict:
    """file 04 §04.4: cast to the scan dtype, transpose to `[B, heads, T, ...]`, l2norm q
    and k, scale q. The grouped K->V expansion happens here, in the activation dtype, as
    the reference does it (`repeat_interleave(2, dim=2)` before the scan call)."""
    dtype = torch.bfloat16 if faults.scan_in_bf16 else g.scan_dtype
    q, k, v = heads["q_pre"], heads["k_pre"], heads["v_pre"]
    if g.expansion > 1:
        if faults.head_pairing_tiled:                              # §04.9 tiled pairing
            idx = torch.arange(g.h_v) % g.h_k
            q, k = q[:, :, idx, :], k[:, :, idx, :]
        else:
            q = q.repeat_interleave(g.expansion, dim=2)             # [REF] grouped
            k = k.repeat_interleave(g.expansion, dim=2)
    # The expanded, still-activation-dtype tensors the block hands to the scan — exactly the
    # arguments the reference's own scan call receives, so the cross-check can compare the
    # block/scan seam against the installed module rather than against ourselves.
    expanded = {"q_exp": q, "k_exp": k, "v_exp": v}
    q = q.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)
    k = k.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)
    v = v.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)
    q = _l2norm(q, g.l2norm_eps, faults)
    k = _l2norm(k, g.l2norm_eps, faults)
    # §the q scaling footnote. The two reference scan functions do NOT spell this the same:
    # `torch_chunk_gated_delta_rule` multiplies by `d ** -0.5`, `torch_recurrent_...` divides
    # by `d ** 0.5`. Both are "q / sqrt(d_k)" in §04.4's words and they are not the same
    # product tree at finite precision, so the arm chooses the spelling and the manifest
    # records it. `auto` (the f64 arms) divides, which is the algebraic reading.
    if not faults.q_scale_omitted:
        d = float(g.d_v if faults.q_scaled_by_dv else g.d_k)
        if form.q_scale == "div" or form.q_scale == "auto":
            q = q / (d ** 0.5)
        else:
            q = q * (d ** -0.5)
    return {"q_hat": q, "k_hat": k, "v_scan": v,
            "beta_scan": beta.transpose(1, 2).to(dtype),
            "decay": g_decay.transpose(1, 2).to(dtype), **expanded}


# --------------------------------------------------------------------- the scans
def scan_recurrent(q, k, v, beta, decay, g: GdnGeometry, form: Form, faults: Faults,
                   initial_state=None) -> tuple:
    """file 04 §04.6 steps 1–5, in that order, all in the scan dtype.

    `state_stored_transposed` keeps the state in `[d_v, d_k]` and reads it where `[d_k, d_v]`
    is meant. It is only definable because `d_k == d_v` here (128 == 128) — with different
    widths torch raises, which is the catcher. The transposed implementation *agrees with
    itself*, which is the point: it is the two-self-consistent-layouts shape at tensor level
    (`pitfalls/40`)."""
    dtype = torch.bfloat16 if faults.scan_in_bf16 else g.scan_dtype
    B, h, T, dk = k.shape
    dv = v.shape[-1]
    if faults.state_stored_transposed:
        S = torch.zeros(B, h, dv, dk, dtype=dtype)
    else:
        S = torch.zeros(B, h, dk, dv, dtype=dtype)
    if initial_state is not None:
        S = initial_state.to(dtype)
    out = torch.zeros(B, h, T, dv, dtype=dtype)
    split = form.scan == "reassoc"
    half = dk // 2
    for i in range(T):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]            # §04.6 step 1
        if faults.decay_after_write:                               # §04.6 order, violated
            S_decayed = S
        else:
            S = S * decay_t
            S_decayed = S
        kv_mem = _contract_read(S, k_t, faults, dk, half, split)   # step 2
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)       # step 3
        S = _rank_one_write(S, k_t, delta, faults)                 # step 4
        if faults.decay_after_write:
            S = S * decay_t
            S_decayed = S
        read_from = S_decayed if faults.read_pre_update_state else S
        out[:, :, i] = _contract_read(read_from, q_t, faults, dk, half, split)   # step 5
    if faults.state_stored_transposed:
        S = S.transpose(-1, -2)                                    # handed on transposed
    return out, S


def _contract_read(S, vec, faults: Faults, dk: int, half: int, split: bool) -> torch.Tensor:
    """`sum_k S[k, v] * vec[k]` — three spellings, one per arm/fault.
    default: [REF] the reference's elementwise-multiply-then-sum.
    `split`: the same sum taken as two halves — the floor probe's re-association
             (`pitfalls/30` §step 1 of §10.3: re-run the reference's own F32 math with its
             summation re-associated).
    transposed fault: the contraction over the wrong axis of a self-consistent layout."""
    if faults.state_stored_transposed:                             # S is [.., dv, dk]
        prod = S * vec.unsqueeze(-2)
        return prod.sum(dim=-1) if not split else (
            prod[..., :half].sum(dim=-1) + prod[..., half:].sum(dim=-1))
    prod = S * vec.unsqueeze(-1)
    if not split:
        return prod.sum(dim=-2)
    return prod[:, :, :half].sum(dim=-2) + prod[:, :, half:].sum(dim=-2)


def _rank_one_write(S, k_t, delta, faults: Faults) -> torch.Tensor:
    if faults.state_stored_transposed:                             # S is [.., dv, dk]
        return S + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
    return S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)


def scan_exact(q, k, v, beta, decay, g: GdnGeometry, form: Form, initial_state=None) -> tuple:
    """Two f64 implementations that share no expression with either f32 arm.

    `exact`: §04.6's five steps, but the two reads are matrix-vector products and the
    state is f64 — the error reference for the floor probe.
    `exact_matrix`: the outer-product form file 04 §04.6 names as equivalent,
    `S_t = exp(g_t) S_{t-1} (I - beta k k^T) + beta v k^T` read as `S^T q`, evaluated as a
    per-token `d_k x d_k` matmul. A different product tree, a different op mix, and the
    witness that `exact` is not merely "the same formula in wider type": their mutual
    distance must stay far below either one's distance to the f32 arms."""
    dtype = torch.float64
    q, k, v, beta, decay = [t.to(dtype) for t in (q, k, v, beta, decay)]
    B, h, T, dk = k.shape
    dv = v.shape[-1]
    S = (torch.zeros(B, h, dk, dv, dtype=dtype) if initial_state is None
         else initial_state.to(dtype))
    out = torch.zeros(B, h, T, dv, dtype=dtype)
    eye = torch.eye(dk, dtype=dtype)
    for i in range(T):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]
        beta_t = beta[:, :, i].unsqueeze(-1)
        if form.scan == "exact_matrix":
            # beta is per (batch, head): shape [B,h,1,1] so it scales the [d_k,d_k] matrix
            # rather than broadcasting against its last axis (a [B,h,1] beta silently reads
            # as "head axis vs feature axis" here and raises, which is the good outcome).
            bm = beta[:, :, i].unsqueeze(-1).unsqueeze(-1)
            P = eye - bm * (k_t.unsqueeze(-1) * k_t.unsqueeze(-2))    # I - b k k^T
            S = decay_t * (P @ S)                                      # [REF] §04.6 equiv
            S = S + bm * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
            out[:, :, i] = (S.transpose(-1, -2) @ q_t.unsqueeze(-1)).squeeze(-1)
        else:
            S = S * decay_t
            kv_mem = (S.transpose(-1, -2) @ k_t.unsqueeze(-1)).squeeze(-1)
            delta = (v_t - kv_mem) * beta_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, i] = (S.transpose(-1, -2) @ q_t.unsqueeze(-1)).squeeze(-1)
    return out, S


def _pad_cols(t: torch.Tensor, pad: int, seed: int) -> torch.Tensor:
    """Zero-pad the time axis, or fill the pad columns deterministically when the fixture
    models a reused buffer (`Fixture.pad_noise`). Same seed for the clean and the injected
    run, so the only difference between the two arms is the fault."""
    if pad == 0:
        return t
    if not seed:
        return F.pad(t, (0, 0, 0, pad))
    gen = torch.Generator(device="cpu").manual_seed(seed)          # CPU only, no device ctx
    noise = torch.randn((t.shape[0], t.shape[1], pad, t.shape[-1]), generator=gen,
                        dtype=torch.float64).to(t.dtype)
    return torch.cat([t, noise], dim=-2)


def _ut_solve(ut: torch.Tensor, rhs: torch.Tensor, dtype: torch.dtype, unitri: bool,
              faults: Faults) -> torch.Tensor:
    """`solve_triangular(ut, rhs, upper=False, unitriangular=...)` at the scan's dtype.

    Under `scan_in_bf16` the solve is done in F32 and cast back, which is *not* a softening of
    the injection: §04.4's scan-dtype row and §04b.4 put the solve in F32 in the first place,
    and the reference casts q/k/v/beta/decay to float32 at the scan entry before it solves
    (`torch_chunk_gated_delta_rule`). This CPU backend has no bf16 `solve_triangular` kernel at
    all (`bf16_solve_supported()`), so the alternative to this route is an arm that never runs.
    The bf16-solve numerics that this therefore does NOT establish are printed in
    `NOT_ESTABLISHED` with the probed error rather than left to silence."""
    if dtype is torch.float32:
        return torch.linalg.solve_triangular(ut, rhs, upper=False, unitriangular=unitri)
    if faults.unitriangular_dropped and dtype is not torch.float32:
        raise AssertionError("unitriangular_dropped under a non-F32 scan would silently test "
                             "the F32 diagonal, not the bf16 one; refuse")
    solved = torch.linalg.solve_triangular(ut.to(torch.float32), rhs.to(torch.float32),
                                           upper=False, unitriangular=unitri)
    return solved.to(dtype)


def scan_chunked(q, k, v, beta, decay, g: GdnGeometry, form: Form, faults: Faults,
                 initial_state=None, pad_noise: int = 0) -> tuple:
    """file 04b §04b.2–§04b.5, transcribed. `[REF: torch_chunk_gated_delta_rule]`.

    The `pairwise` construction masks the strictly-upper triangle to `-inf` **before** the
    exp (§04b.3: "Any implementation that masks after `exp` ... is not this form"), and the
    solve is `unitriangular=True`, i.e. `I + strictly_lower(ut_system)` (§04b.4). The two
    legal solve branches are both here and are **not the same number** — A-16, and the
    manifest must say which one made a record."""
    dtype = torch.bfloat16 if faults.scan_in_bf16 else g.scan_dtype
    B, h, T, dk = k.shape
    dv = v.shape[-1]
    C = form.chunk_width or g.chunk_width
    if C < 2:
        raise AssertionError("chunk_width < 2 makes chunk_decay_off_by_one degenerate")
    pad = (-(-T // C)) * C - T
    q_, k_, v_ = (_pad_cols(t, pad, pad_noise) for t in (q, k, v))
    if pad and faults.pad_beta_ones:
        beta_ = F.pad(beta, (0, pad), value=1.0)     # §04b.2: pads must be ZERO
        decay_ = F.pad(decay, (0, pad), value=0.0)
    else:
        beta_, decay_ = (F.pad(t, (0, pad)) for t in (beta, decay))
    # [REF] beta is applied to K *and* V here (§04b.3); dropping it from K is §10.4's row.
    v_beta = v_ * beta_.unsqueeze(-1)
    k_beta = k_ if faults.beta_on_v_not_k else k_ * beta_.unsqueeze(-1)
    reshape = lambda t: t.reshape(t.shape[0], t.shape[1], -1, C, t.shape[-1])
    q_, k_, k_beta, v_beta = (reshape(t) for t in (q_, k_, k_beta, v_beta))
    decay_ = decay_.reshape(decay_.shape[0], decay_.shape[1], -1, C)
    strictly_upper = torch.ones(C, C, dtype=torch.bool).triu(1)
    cum = decay_.cumsum(dim=3)                                       # §04b.3
    if faults.pairwise_mask_after_exp:
        # §04b.3: "Any implementation that masks after `exp` ... is not this form." The
        # difference is NOT a rounding difference, and `masked_fill(0)` after the exp does not
        # show it — overwriting an `inf` with `0` is identical to never having computed it. The
        # failure mode is that the *unmasked* triangle overflows: the exponent on the masked
        # side is `+|sum of g|`, and `g` reaches -91.6 per token on this checkpoint (§04.5), so
        # `exp` = inf in f32 — 617 inf entries at T=4 and 31,679 at T=160 on layer0
        # (tmp/w20_primitives.py). Multiplying that by a 0/1 mask yields `inf * 0 = nan`, which
        # is exactly the corruption the spec forbids the order for, so the injection is written
        # as the multiplication and not as a fill (`pitfalls/40` §a control that cannot fail).
        keep_mask = (~strictly_upper).to(dtype)
        pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).exp() * keep_mask
    else:
        pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(strictly_upper,
                                                                      float("-inf")).exp()
    ut_system = (k_beta @ k_.transpose(-1, -2)) * pairwise
    intra_attn = (q_ @ k_.transpose(-1, -2)) * pairwise
    dec_k_beta = k_beta * cum.exp().unsqueeze(-1)
    unitri = not faults.unitriangular_dropped
    if form.solve_branch == "triangular":
        new_values = _ut_solve(ut_system, v_beta, dtype, unitri, faults)
        k_cumdecay = _ut_solve(ut_system, dec_k_beta, dtype, unitri, faults)
    elif form.solve_branch == "substitution":
        ut = -ut_system.tril(-1)
        for i in range(1, C):
            row = ut[..., i, :i].clone()
            sub = ut[..., :i, :i].clone()
            ut[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut = ut + torch.eye(C, dtype=ut.dtype)
        if not unitri:                     # the fault in this branch: the identity is the diag
            ut = ut - torch.eye(C, dtype=ut.dtype)
        new_values, k_cumdecay = ut @ v_beta, ut @ dec_k_beta
    else:
        raise AssertionError(form.solve_branch)
    S = (torch.zeros(B, h, dk, dv, dtype=dtype) if initial_state is None
         else initial_state.to(dtype))
    if faults.state_stored_transposed:
        S = S.transpose(-1, -2).contiguous().transpose(-1, -2)      # layout, same values
    core = torch.zeros(B, h, q_.shape[2], C, dv, dtype=dtype)
    q_ = q_ * cum.exp().unsqueeze(-1)                                # §04b.4, once ...
    if faults.q_decayed_twice:
        q_ = q_ * cum.exp().unsqueeze(-1)                            # ... not twice
    k_ = k_ * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    last = cum[..., -1] if not faults.chunk_decay_off_by_one else cum[..., -2]
    chunk_decay = last.exp()[..., None, None]                        # §04b.5
    for i in range(q_.shape[2]):
        Sc = S.transpose(-1, -2) if faults.state_stored_transposed else S
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ Sc
        core[:, :, i] = q_[:, :, i] @ Sc + intra_attn[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k_[:, :, i].transpose(-1, -2) @ v_new
    core = core.reshape(B, h, -1, dv)[:, :, :T]
    return core, S


def gated_norm(core_attn: torch.Tensor, z: torch.Tensor, w: GdnWeights,
               g: GdnGeometry, faults: Faults) -> dict:
    """file 04 §04.7 `[REF: Qwen3_5MoeRMSNormGated.forward]`, transcribed line for line
    including the intermediate cast back to the input dtype:

        dtype0 = out.dtype
        var    = mean(out.float()^2, -1)
        out    = out.float() * rsqrt(var + eps)        # Norm before gate
        out    = w * cast_to_dtype0(out)               # weight applied in the INPUT dtype
        out    = out * silu(z.float())                 # gate in F32
        return cast_to_dtype0(out)

    Two facts make this a different RMSNorm from §03.1: the weight is plain `w * x_hat`
    (no `+1`), and the variance is over `d_v = 128`, never over the flat 4096-wide vector."""
    # `z` was already permuted in `projections` — permuting it twice would be a different
    # fault, so the V-side permutation lives in exactly one place (pitfalls/40 §one address
    # expression shared by writer and reader).
    dtype0 = core_attn.dtype
    x = core_attn.to(torch.float32)
    if faults.norm_variance_flat:
        var = x.pow(2).flatten(-2).mean(-1, keepdim=True).unsqueeze(-1)
    else:
        var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + g.rms_norm_eps)
    if faults.norm_gate_first:
        # §04.7 "Norm before gate", violated: gate, then take the variance of the gated
        # value. Algebraically different, and it raises nothing.
        xg = x * F.silu(z.to(torch.float32))
        var_g = xg.pow(2).mean(-1, keepdim=True)
        xg = xg * torch.rsqrt(var_g + g.rms_norm_eps)
        weight = w.tensors["norm"]
        if faults.norm_plus_one:
            weight = 1.0 + weight.to(torch.float32)
        return {"normed": (weight * xg.to(dtype0)).to(dtype0)}
    weight = w.tensors["norm"]
    if faults.norm_plus_one:
        weight = 1.0 + weight.to(torch.float32)
    if faults.norm_weight_f32:
        x = weight * x                                       # no cast back to dtype0
    else:
        x = weight * x.to(dtype0)                            # [REF] the intermediate cast
    x = x * F.silu(z.to(torch.float32))
    return {"normed": x.to(dtype0)}


def out_proj(normed: torch.Tensor, w: GdnWeights, g: GdnGeometry, faults: Faults) -> dict:
    """file 04 §04.8: `out_proj` is `[hidden_size, value_dim]`, no bias, its columns in the
    same V-head order as everything else (§04.9 / A-12)."""
    W = w.tensors["out"]
    if faults.v_head_order_tiled:
        # A-12 names the out_proj COLUMNS as a V-side family. They are laid out as
        # [hidden_size, h_v, d_v], so the permutation is over the head axis, not over the
        # 4096 columns — permuting the flat column axis would index 32 of 4096 and raise.
        W = _perm_v(W.reshape(g.hidden_size, g.h_v, g.d_v), 1, g).reshape(
            g.hidden_size, g.value_dim)
    flat = normed.reshape(normed.shape[0], normed.shape[1], g.value_dim)
    if flat.shape[-1] != g.value_dim:
        raise AssertionError(f"normed flat width {flat.shape[-1]} != value_dim={g.value_dim}: "
                             "§04.8's reshape before out_proj is wrong")
    return {"y": F.linear(flat, W)}


# ------------------------------------------------------------------- the whole block
def gdn_block(x: torch.Tensor, w: GdnWeights, g: GdnGeometry, arm: str | Form = "chunked",
              faults: Faults = NO_FAULTS, *, mode: str = "prefill",
              state: torch.Tensor | None = None,
              conv_window: torch.Tensor | None = None,
              pad_noise: int = 0) -> dict:
    """One pass of the GDN block, returning every stage in `STAGES` under its own name.

    `mode="prefill"` runs the chunked dispatch of §04b.1; `mode="decode"` is the
    `use_precomputed_states and seq_len == 1` branch, which takes the conv *update* and the
    **recurrent** scan. The caller passes the carried state and conv window; this function
    returns the next ones, so a sequence is a fold over it."""
    form = Form.named(arm) if isinstance(arm, str) else arm
    out = {}
    out.update(projections(x, w, g, faults))
    out.update(short_conv(out["mixed_qkv"], w, g, faults,
                          window=conv_window, mode=mode))
    # `conv_pre`/`conv_out` are kept in the reference's conv orientation [B, conv_dim, T],
    # which is where §10.4's conv injections must fire; the reference transposes back
    # (`mixed_qkv = mixed_qkv.transpose(1, 2)`) before the split, and so do we.
    heads = split_heads(out["conv_out"].transpose(1, 2), g, faults)
    out.update(heads)
    out.update(decay_and_beta(out["b_raw"], out["a_raw"], w, g, faults))
    scan_in = enter_scan(heads, out["beta"], out["g_decay"], g, form, faults)
    out["q_hat"], out["k_hat"] = scan_in["q_hat"], scan_in["k_hat"]
    if form.scan in ("exact", "exact_matrix"):
        scan_out, state_next = scan_exact(scan_in["q_hat"], scan_in["k_hat"],
                                          scan_in["v_scan"], scan_in["beta_scan"],
                                          scan_in["decay"], g, form, state)
    elif form.scan in ("chunked",):
        scan_out, state_next = scan_chunked(scan_in["q_hat"], scan_in["k_hat"],
                                            scan_in["v_scan"], scan_in["beta_scan"],
                                            scan_in["decay"], g, form, faults, state,
                                            pad_noise=pad_noise)
    else:
        scan_out, state_next = scan_recurrent(scan_in["q_hat"], scan_in["k_hat"],
                                              scan_in["v_scan"], scan_in["beta_scan"],
                                              scan_in["decay"], g, form, faults, state)
    out["scan_out"] = scan_out
    # [REF] the scan returns in the activation dtype; the gated norm's `dtype0` is that
    # dtype, so the f32 scan output the state-comparison uses is NOT what the norm reads.
    out["core_attn"] = scan_out.transpose(1, 2).to(x.dtype,
                                                   memory_format=torch.contiguous_format)
    out["state"] = state_next
    if out["conv_window"] is None:                    # prefill: the reference's cache keeps
        mi = out["mixed_qkv"].transpose(1, 2)         # the last k_conv INPUT columns
        pad = g.k_conv - mi.shape[-1]
        out["conv_window"] = F.pad(mi, (pad, 0)) if pad > 0 else mi[:, :, -g.k_conv:]
    elif faults.conv_state_is_outputs:
        # §04.3 consequence 3 / §10.4: the state is a window of previous mixed_qkv **inputs**,
        # never of conv outputs. A producer with this bug keeps a rolling window of the last
        # `k_conv` output columns, so the corruption is visible only to the NEXT step — which
        # is why this fault's contract is a deferred fire rather than an immediate one
        # (`pitfalls/30` §a hook that reads `output` can pin the answer under the name of an
        # intermediate).
        out["conv_window"] = torch.cat([conv_window[:, :, 1:], out["conv_out"][:, :, -1:]],
                                       dim=-1)
    out.update(gated_norm(out["core_attn"], out["z"], w, g, faults))
    out.update(out_proj(out["normed"], w, g, faults))
    out["form"] = form
    out["faults"] = faults
    missing = [s for s in STAGES if s not in out]
    if missing:
        raise AssertionError(f"stage(s) {missing} were never produced — the stage list and "
                             "the pipeline disagree")
    return out


def gdn_sequence(x_prefill: torch.Tensor, x_steps: torch.Tensor, w: GdnWeights,
                 g: GdnGeometry, prefill_arm: str | Form, decode_arm: str | Form,
                 faults: Faults = NO_FAULTS, decode_faults: Faults | None = None,
                 prefill_stages: dict | None = None, pad_noise: int = 0) -> dict:
    """Prefill then cached decode (§04b.1's table), threaded state and conv window.

    Returns the prefill stages, the LAST decode step's stages, and every step's `y` so the
    §10.4 "and every subsequent step" claim about the conv state is adjudicated per step
    rather than asserted. `decode_faults` exists so a decode-path injection is not smuggled
    into the prefill that produced the states it is meant to corrupt."""
    decode_faults = faults if decode_faults is None else decode_faults
    if prefill_stages is not None:
        pf = prefill_stages
    else:
        pf = gdn_block(x_prefill, w, g, prefill_arm, faults, mode="prefill",
                       pad_noise=pad_noise)
    state, window = pf["state"], pf["conv_window"]
    steps, step_y = [], []
    for s in range(x_steps.shape[1]):
        st = gdn_block(x_steps[:, s:s + 1], w, g, decode_arm, decode_faults, mode="decode",
                       state=state, conv_window=window, pad_noise=pad_noise)
        state, window = st["state"], st["conv_window"]
        steps.append(st)
        step_y.append(st["y"])
    pf["n_chunks_used"] = pf["form"].chunk_width or g.chunk_width
    return {"prefill": pf, "decode": steps[-1], "all_steps": steps,
            "decode_steps": len(steps),
            "step_y": step_y, "state_final": state, "conv_window_final": window}


# --------------------------------------------------------------------- comparators
# Which axis of each staged tensor is the V-head axis, so "plus a per-head maximum"
# (§04b.6) folds the right axes. `None` = the tensor is not head-indexed, so its per-head
# maximum degenerates to its max and is printed as such rather than pretending otherwise.
HEAD_AXIS = {"q_hat": 1, "k_hat": 1, "scan_out": 1, "state": 1, "core_attn": 2,
             "normed": 2, "beta": 2, "g_decay": 2, "v_pre": 2, "z": 2}


def aggregates(a: torch.Tensor, b: torch.Tensor, *, name: str = "",
               stage: str | None = None,
               anchor: torch.Tensor | None = None) -> dict:
    """Aggregate comparator for one stage of two implementations.

    Widening is always an explicit `.to(torch.float64)` — never `.view()`, which bitcasts
    and turns an f32/bf16 mix into garbage that still prints a plausible `ndiff`
    (`pitfalls/30` §Tensor.to casts, `.view` bitcasts; §Read a dump's dtype).

    `anchor` is the f64 exact arm. When given, `mean_rel`'s denominator is
    `max(|anchor|, 1e-3 max|anchor|)` — a *common* denominator, which is what makes
    `mean_rel` a metric and lets the band be derived from the triangle inequality rather
    than typed (§Bands). The 1e-3 scale floor is the cancellation floor: `pitfalls/30`
    §A max-ULP band cannot see a whole-tensor 1-ULP error, and a chained dot's legal
    `ndiff` is 29–47 %, so an unanchored relative error near a cancellation is noise.

    Per-head maximum is printed because §04b.6 requires it: "cos + mean-rel over the whole
    [h_v, d_k, d_v] block, **plus** a per-head maximum"."""
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    # A bf16/f32 mix is the `pitfalls/30` dump-reading bug (an F32 dump read as BF16 still
    # prints a plausible ndiff), so it is refused. Widening f32 against the f64 anchor is
    # not that bug and is what a floor probe is, so it is allowed.
    if (a.dtype is torch.bfloat16) != (b.dtype is torch.bfloat16):
        raise AssertionError(f"{name}: dtype {a.dtype} vs {b.dtype} — a bf16/f32 mix is the "
                             "dump-read bug, refuse to print (pitfalls/30)")
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise AssertionError(f"{name}: non-finite input to aggregates()")
    scale = y.abs().max().item()
    if scale == 0:
        raise AssertionError(f"{name}: the reference side is all zero; a comparator over "
                             "zeros cannot fail")
    diff = (x - y).abs()
    if anchor is None:
        denom = y.abs()
        denom = torch.clamp(denom, min=1e-3 * scale)
    else:
        an = anchor.detach().to(torch.float64)
        denom = torch.clamp(an.abs(), min=1e-3 * an.abs().max().item())
    mean_rel = (diff / denom).mean().item()
    max_abs = diff.max().item()
    # per-head maximum: fold every axis except the head axis (dim 1 in [B,h,T/dk,dv] shapes,
    # dim -3 for the [B,h,dk,dv] state) and take the worst head
    axis = HEAD_AXIS.get(stage) if stage is not None else None
    if axis is not None and 0 <= axis < diff.ndim:
        per_head = diff.amax(dim=tuple(d for d in range(diff.ndim) if d != axis))
    else:
        per_head = diff.flatten()
    n_heads = int(per_head.numel())
    worst_head = int(per_head.argmax().item())
    per_head_max = per_head.max().item()
    denom_cos = (x.norm() * y.norm()).item()
    if denom_cos == 0:
        raise AssertionError(f"{name}: zero-norm side; cos is undefined, refuse to print")
    cos = (x.flatten() @ y.flatten()).item() / denom_cos
    one_minus_cos = max(0.0, 1.0 - cos)
    for v in (max_abs, one_minus_cos, mean_rel):
        if v != v:
            raise AssertionError(f"{name}: NaN aggregate; the comparator is broken")
    ulp_ref = R.ulp_of(scale)
    return {"name": name, "shape": list(a.shape), "dtype": str(a.dtype), "n": int(a.numel()),
            "max_abs": max_abs, "one_minus_cos": one_minus_cos, "mean_rel": mean_rel,
            "per_head_max": per_head_max, "worst_head": worst_head, "n_heads": n_heads,
            "ndiff": int((diff > 0).sum().item()), "ref_absmax": scale,
            "ulp_of_refmax": ulp_ref}


def fmt_ag(m: dict) -> str:
    return (f"{m['name']:<46} {m['dtype']:<15} {str(m['shape']):<22} "
            f"maxabs={m['max_abs']:.3e} 1-cos={m['one_minus_cos']:.3e} "
            f"meanrel={m['mean_rel']:.3e} perhead={m['per_head_max']:.3e}"
            f"(h{m['worst_head']}/{m['n_heads']}) ndiff={m['ndiff']}/{m['n']}")


def stage_moved(a: torch.Tensor, b: torch.Tensor, predicate=None) -> tuple:
    """(did the stage move, max|a-b|). The SINGLE comparison used by both directions of the
    localisation contract, so the must-not-catch side is witnessed by the must-catch side
    firing on the same machinery (`selftest_moe.py` §differ). A dtype change IS a move:
    `scan_in_bf16` is a policy violation whose whole point is the dtype.

    Every verdict — including the two structural ones, shape and dtype — is routed through
    `predicate`. The stub arm exists to prove the gate reads the differ and not a constant, and
    a structural short-circuit that answers before the differ is asked leaves the stub blind to
    exactly the signal `scan_in_bf16` produces (witnessed before this was routed: arm=stub
    reported `stage_diffs=48` and still `caught` 12 cases, and its must_catch_missed read 2036
    against a derived 2084)."""
    blind = predicate is not None
    if a.shape != b.shape:
        return ((False if blind else True), float("inf"))
    if a.dtype != b.dtype:
        return ((False if blind else True), float("nan"))
    x, y = a.detach().to(torch.float64), b.detach().to(torch.float64)
    n = int((x != y).sum().item())
    changed = n > 0
    if predicate is not None:
        changed = predicate(changed)
    if n == 0:
        return changed, 0.0
    return changed, (x - y).abs().max().item()


# ------------------------------------------------------------------------- the band
# §Bands — derived, never typed. `specification/10-reference-dumps-and-gates.md` §10.3 step 1
# requires a floor probe: "re-run the reference's own F32 math with its summation
# re-associated, over the pinned rows, per family". Steps 2–3 assign the metric per family:
# stage-local near-exact ops get ndiff with no cancellation band, and the long dots (the GDN
# scan is named in that row) get cos + mean-rel aggregates plus a cancellation band, "both
# sized by the probe".
#
# The probe measures, per stage, the distance of every LEGAL arm to the f64 anchor
# (`exact`). The cross-form band is then the triangle inequality, with no fitted factor:
#
#   1-cos:  with d(a,b) = sqrt(1 - cos(a,b)) — a true metric (the angle),
#           d(A,B) <= d(A,E) + d(E,B) <= 2 sqrt(F)  =>  1-cos(A,B) <= 4 F,
#           where F = max over arms of 1-cos(arm, anchor). The 4 is (1+1)^2, not a taste.
#   mean_rel: the denominators are all the anchor's, so |A-B|/den <= |A-E|/den + |E-B|/den
#           pointwise  =>  mean_rel(A,B) <= 2 R,  R = max over arms of mean_rel(arm, anchor).
#   max_abs, per-head max: the same pointwise triangle  =>  2 M.
#
# A band is therefore a *statement about the probe*, and if the probe is missing or its
# maximum is exactly 0 the gate refuses to run rather than falling back to a number.
BAND_FACTORS = {"one_minus_cos": 4.0, "mean_rel": 2.0, "max_abs": 2.0, "per_head_max": 2.0}


def band_from_probe(probe_rows: list, stages: tuple = tuple(STAGES)) -> dict:
    """Fold probe records into per-stage bands. `probe_rows` is the list of
    `{arm, stage, aggregates}` records from `--floor-probe`.

    Refusals, all of them loud: no records for a stage; a maximum of exactly 0 (a probe
    that measured nothing cannot license a band); a non-finite maximum."""
    worst = {}
    for rec in probe_rows:
        st, m = rec["stage"], rec["aggregates"]
        if st not in stages:
            raise AssertionError(f"probe record for stage {st!r} outside the stage list")
        cur = worst.setdefault(st, {k: 0.0 for k in BAND_FACTORS})
        for key in BAND_FACTORS:
            v = m[key]
            if v != v or v == float("inf"):
                raise AssertionError(f"probe {rec['arm']}/{st}: {key}={v} is not a number")
            cur[key] = max(cur[key], v)
    missing = [s for s in stages if s not in worst]
    if missing:
        raise AssertionError(f"no floor-probe record for stage(s) {missing}; the band cannot "
                             "be derived and this gate refuses to invent one "
                             "(specification/10 §10.3, pitfalls/30)")
    band = {}
    for st, cur in worst.items():
        band[st] = {"probe": dict(cur),
                    **{k: cur[k] * BAND_FACTORS[k] for k in BAND_FACTORS},
                    "derived_from": "floor probe (f64 anchor) x the triangle inequality"}
        for k in BAND_FACTORS:
            if cur[k] == 0.0:
                band[st]["vacuous"] = (band[st].get("vacuous", []) + [k])
    return band


def load_band(path: Path) -> dict:
    """Read a band written by `--floor-probe`. Missing file is an ERROR, not a default."""
    if not Path(path).exists():
        raise AssertionError(f"{path} does not exist: run "
                             "`speccheck/gdn_ref.py --floor-probe` first. A gate with no "
                             "cited band source is not a gate (AGENTS.md §Standing rules).")
    doc = json.loads(Path(path).read_text())
    if doc.get("band_derivation") != "triangle-inequality-on-floor-probe":
        raise AssertionError(f"{path} was written by a different derivation "
                             f"({doc.get('band_derivation')!r}); refuse to read it")
    return doc["band"]


# ================================================================== the HF witness
# §What the installed reference is allowed to witness here, and what it is not.
#
# `pitfalls/30` §"A hoisted scalar constant is a different number, and a CPU mirror cannot
# see it": a mirror that shares a formula with the reference agrees with it *because* they
# share a formula. So every row this section prints is a **transcription** check (it catches
# a misread §04), never an independence proof — the independence evidence is the cross-form
# agreement and the three-arm injection matrix. What this section does establish, and is the
# reason it exists, is that the oracle and the pinned reference produce the *same bytes* on
# the same weights and the same input, stage by stage, in both dispatch shapes.
#
# Where each HF-side tensor comes from is recorded per stage in `HF_PROVENANCE` and printed
# with the row, because "we compared against HF" is not one claim but three:
#   captured               read off the installed module at one of its own boundaries (a
#                          forward hook, or a spy on a module-level function it calls);
#   derived-from-captured  a spec-quoted reshape/split applied to a captured tensor — here
#                          §04.2's `torch.split(mixed_qkv, [key_dim, key_dim, value_dim])`,
#                          written out on the capture side rather than reused from
#                          `split_heads`, so the two sides do not share the expression, and
#                          witnessed against HF's own scan arguments (the `seam` rows);
#   reference-side-call    the reference's own function re-run on the captured arguments so
#                          that a value it never returns gets materialised (only `scan_out`,
#                          whose F32 `core_attn_out` is cast to the activation dtype inside
#                          `torch_chunk_gated_delta_rule` before the function returns).
# A stage missing from `HF_PROVENANCE` fails the section: an unclaimed stage is not a
# checked one (`pitfalls/40` §a check that produces no output cannot fail).
#
# The section is only meaningful if it can fail, so it carries its own positive controls:
# the oracle's recurrence is sabotaged (`read_pre_update_state`, `decay_after_write` — the
# §04.6 step-5 read position and the step-1/step-4 order) and the chunked form's decay
# ordering (`decay_sign_flip`, `chunk_decay_off_by_one`, `pairwise_mask_after_exp`,
# `beta_no_sigmoid`) and each control must be *reported as a difference* by this comparison,
# on its contracted stages and on no others. A comparison that has never differed is a
# comparison that was hoped in, not tested.

# The nine parameters the witness block is built with. They are set to the oracle's own
# tensors (`build_hf_gdn_block` witnesses the `data_ptr()` identity per tensor), so a
# disagreement here is about the maths and never about two different weight loads.
HF_PARAM_MAP = {
    "in_proj_qkv.weight": "qkv", "in_proj_z.weight": "z", "in_proj_a.weight": "a",
    "in_proj_b.weight": "b", "conv1d.weight": "conv", "A_log": "A_log",
    "dt_bias": "dt_bias", "norm.weight": "norm", "out_proj.weight": "out",
}

HF_PROVENANCE = {
    "mixed_qkv": ("captured", "forward hook on in_proj_qkv's output"),
    "z": ("captured", "forward hook on in_proj_z's output, reshaped to [B,T,h_v,d_v]"),
    "b_raw": ("captured", "forward hook on in_proj_b's output"),
    "a_raw": ("captured", "forward hook on in_proj_a's output"),
    "conv_pre": ("captured", "the bf16 input to ACT2FN['silu'] inside causal_conv1d_fn/"
                          "_update: the pre-activation conv output, which the function never"
                          " returns"),
    "conv_out": ("captured", "the return of causal_conv1d_fn / causal_conv1d_update"),
    "conv_window": ("captured", "the layer's conv_states cache after the pass (the reference"
                              "'s own k_conv-column INPUT window, A-09)"),
    "q_pre": ("derived-from-captured", "§04.2's torch.split+reshape of the captured conv_out"),
    "k_pre": ("derived-from-captured", "§04.2's torch.split+reshape of the captured conv_out"),
    "v_pre": ("derived-from-captured", "§04.2's torch.split+reshape of the captured conv_out"),
    "beta": ("captured", "the `beta` argument of the reference's own scan call"),
    "g_decay": ("captured", "the `g` argument of the reference's own scan call"),
    "q_hat": ("reference-side-call", "M.l2norm + the scan's own q-scaling line, applied to the"
                                     " captured scan argument (F32, per §04.4)"),
    "k_hat": ("reference-side-call", "M.l2norm applied to the captured scan argument (F32, per"
                                     " §04.4); unscaled, §04.4's l2norm-scope row"),
    "scan_out": ("reference-side-call", "torch_chunk/recurrent_gated_delta_rule re-run on the"
                                        " captured args with F32 inputs, so initial_dtype is"
                                        " F32 and the internal core_attn_out is returned"
                                        " uncast; witnessed by its bf16 cast equalling the"
                                        " captured core_attn"),
    "core_attn": ("captured", "the scan's returned core_attn_out (activation dtype)"),
    "state": ("captured", "the scan's returned last_recurrent_state (F32)"),
    "normed": ("captured", "forward hook on Qwen3_5MoeRMSNormGated's output, reshaped back"
                           " to [B,T,h_v,d_v]"),
    "y": ("captured", "the return of Qwen3_5MoeGatedDeltaNet.forward / the out_proj hook"),
}

# §The explicit UNVERIFIED list, and it is a rule that this stays non-empty: the section
# fails if it is ever emptied, because a lane with nothing unverified has stopped looking
# rather than finished looking.
HF_UNVERIFIED = [
    {"item": "agreement on a real activation",
     "why": "both sides are fed `gdn_input`, the A-01 RMS-1 seeded fixture; it does not carry "
            "the (1+w) modulation of a real input_layernorm output, so every agreement figure "
            "here is a synthetic-input statistic",
     "closes_when": "the dense path lands and a real hidden_states is pinned by "
                    "tools/dump_reference.py's successor"},
    {"item": "the fused kernels the box does not have",
     "why": "causal_conv1d and flash-linear-attention are not installed, so the reference ran "
            "its own torch fallbacks; which function objects ran is printed with the section",
     "closes_when": "a kernel-bearing build is compared on the same pinned rows"},
    {"item": "chunked continuation (T > 1 with a state present, §04b.1 row 4)",
     "why": "the comparison covers chunked-from-zero prefill and single-token recurrent decode; "
            "no pinned case hands a live state to the chunked scan",
     "closes_when": "a rollback/chunked-continuation fixture is pinned"},
    {"item": "the engine's conv-window policy (A-09: k_conv vs k_conv-1 columns)",
     "why": "conv_window is compared against the reference's own k_conv-column cache, which is "
            "the reference's choice and not the engine's; the engine's 3-column window is a "
            "different tensor and is not checked here",
     "closes_when": "the engine's residency layout is gated against a dump of its own window"},
    {"item": "the MTP draft block's mixer",
     "why": "load_gdn_block_weights refuses a non-`layer` block; file 09's draft mixer is a "
            "different task",
     "closes_when": "file 09 is in scope with its own pinned dump"},
    {"item": "banded (not bit-exact) agreement on any stage",
     "why": "no --floor-probe section exists yet, so no band file exists to load and "
            "load_band() refuses to invent one; every credited row here is bit-exact or it is a "
            "failure",
     "closes_when": "--floor-probe writes the per-stage probe and band_from_probe licenses the "
                    "cross-form rows"},
]


class _HfCapture:
    """Spies and hooks on the installed reference, installed per call and removed after it.

    Every field counts its own calls: a capture that produced nothing would otherwise read
    as an empty tensor comparison that trivially 'agrees' (`pitfalls/40` §a check that
    produces no output cannot fail)."""

    def __init__(self):
        self.calls = {}
        self.hook_io = {}
        self.conv_fn_in, self.conv_out = [], []
        self.silu_in = []
        self.scan_args, self.scan_return, self.scan_which = [], [], []
        self.window_after = []
        self._handles, self._saved = [], {}

    def _rec(self, key, value):
        self.calls[key] = self.calls.get(key, 0) + 1
        (self.hook_io.setdefault(key, [])).append(value)

    def install(self, M, blk):
        def hook(name):
            def _h(mod, inputs, output):
                self._rec("hook:" + name, {"inputs": tuple(inputs), "output": output})
            return _h

        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "norm",
                     "out_proj"):
            self._handles.append(getattr(blk, name).register_forward_hook(hook(name)))

        orig = {"causal_conv1d_fn": M.causal_conv1d_fn,
                "causal_conv1d_update": M.causal_conv1d_update,
                "torch_chunk_gated_delta_rule": M.torch_chunk_gated_delta_rule,
                "torch_recurrent_gated_delta_rule": M.torch_recurrent_gated_delta_rule,
                "ACT2FN": M.ACT2FN}
        self._saved = orig

        def conv_fn(x, weight, bias=None, activation=None, **kw):
            self._rec("conv_fn", {"inputs": (x, weight, bias, activation)})
            out = orig["causal_conv1d_fn"](x, weight, bias, activation, **kw)
            self._rec("conv_fn_out", {"output": out})
            return out

        def conv_update(x, conv_state, weight, bias=None, activation=None):
            self._rec("conv_upd", {"inputs": (x, conv_state, weight, bias, activation)})
            out = orig["causal_conv1d_update"](x, conv_state, weight, bias, activation)
            self._rec("conv_upd_out", {"output": out})
            return out

        def make_scan(which):
            def _scan(*a, **kw):
                # initial_state is the cache's own buffer, and the block copies the NEW state
                # into it before this call returns — clone it here or the side call below would
                # start from the state this step produced (witnessed: 417,407 of 524,288 state
                # elements and 3,090 of 4,096 outputs off by one step's decay).
                kw_seen = dict(kw)
                init = kw_seen.get("initial_state")
                if torch.is_tensor(init):
                    kw_seen["initial_state"] = init.detach().clone()
                self._rec("scan", {"output": (which, a, kw_seen)})
                out = orig["torch_chunk_gated_delta_rule" if which == "chunked" else
                          "torch_recurrent_gated_delta_rule"](*a, **kw)
                self._rec("scan_out", {"output": out})
                return out
            return _scan

        def spy_silu(x):
            self._rec("silu", x)
            return orig["ACT2FN"]["silu"](x)

        M.causal_conv1d_fn = conv_fn
        M.causal_conv1d_update = conv_update
        M.torch_chunk_gated_delta_rule = make_scan("chunked")
        M.torch_recurrent_gated_delta_rule = make_scan("recurrent")
        M.ACT2FN = {**orig["ACT2FN"], "silu": spy_silu}

    def restore(self, M):
        for h in self._handles:
            h.remove()
        self._handles = []
        for k, v in self._saved.items():
            setattr(M, k, v)
        self._saved = {}

    def one(self, key):
        """Exactly-one-of: a capture that fired zero times, or twice, is a broken capture."""
        if self.calls.get(key, 0) != 1:
            raise AssertionError(f"HF capture: expected exactly 1 `{key}` call, got "
                                 f"{self.calls.get(key, 0)} — the spy is not on the path that "
                                 "ran, so nothing was compared")
        return self.hook_io[key][0]


def _hf_modules():
    """The lazy HF import — the only place in this file allowed to know the reference
    exists, and it is reached only from `--compare-hf` (which `--selftest` calls)."""
    import transformers
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.cache_utils import DynamicCache
    cfg = json.loads(Path(CONFIG_PATH).read_text())["text_config"]
    return M, Qwen3_5MoeTextConfig(**cfg), DynamicCache, transformers.__version__


def build_hf_gdn_block(block: Block, w: GdnWeights, g: GdnGeometry, cfg_text, M):
    """Instantiate the reference's own class and hand it the oracle's weight tensors.

    Meta-device construction, then `nn.Parameter(our tensor)`: the nine parameters must be
    the *same storage* as `w.tensors`, which is witnessed here rather than assumed, because
    an agreement between two different weight loads would prove nothing about the maths."""
    import torch.nn as nn
    with torch.device("meta"):
        blk = M.Qwen3_5MoeGatedDeltaNet(cfg_text, layer_idx=block.index)
    for pname, role in HF_PARAM_MAP.items():
        obj = blk
        for part in pname.split(".")[:-1]:
            obj = getattr(obj, part)
        setattr(obj, pname.split(".")[-1],
                nn.Parameter(w.tensors[role], requires_grad=False))
    share = []
    for pname, role in HF_PARAM_MAP.items():
        obj = blk
        for part in pname.split(".")[:-1]:
            obj = getattr(obj, part)
        got = getattr(obj, pname.split(".")[-1])
        if got.data_ptr() != w.tensors[role].data_ptr():
            raise AssertionError(f"{block.name}.{role}: the witness block's parameter is a "
                                 "different tensor from the oracle's; the comparison would be "
                                 "two weight loads, not two implementations")
        if tuple(got.shape) != tuple(w.tensors[role].shape):
            raise AssertionError(f"{block.name}.{role}: {tuple(got.shape)} vs "
                                 f"{tuple(w.tensors[role].shape)}")
        share.append(role)
    return blk, share


def _bit_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """(bit_identical, ndiff, maxabs) for two tensors, with shape and dtype checked first.

    Widening is an explicit `.to(torch.float64)`; `.view()` would bitcast and turn an
    f32/bf16 mix into garbage that still prints a plausible ndiff (`pitfalls/30` §Tensor.to
    casts, .view bitcasts). A non-finite value is reported, not refused — an injected
    `pairwise_mask_after_exp` is *supposed* to produce `inf`/`nan`, and printing it is the
    point (`pitfalls/40` §a control that cannot fail)."""
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"shape {tuple(a.shape)} vs {tuple(b.shape)} — a shape mismatch is "
                             "a mis-wired capture, not a disagreement")
    if a.dtype is not b.dtype:
        raise AssertionError(f"dtype {a.dtype} vs {b.dtype} — a bf16/f32 mix is the dump-read "
                             "bug; refuse to print a number for it (`pitfalls/30`)")
    x, y = a.detach().to(torch.float64), b.detach().to(torch.float64)
    n = int((x != y).sum().item())
    max_abs = 0.0 if n == 0 else float((x - y).abs().max().item())
    return {"bit_identical": n == 0, "ndiff": n, "n": int(x.numel()), "max_abs": max_abs,
            "dtype": str(a.dtype), "shape": list(a.shape)}


def hf_reference_stages(block: Block, w: GdnWeights, g: GdnGeometry, x_prefill: torch.Tensor,
                        x_steps: torch.Tensor) -> tuple:
    """Run the installed reference over prefill-then-cached-decode and harvest every stage.

    Returns `({"prefill": {stage: tensor}, "decode": [{stage: tensor}, ...],
    "prefill_seam": {...}, "decode_seam": [...]}, ran)` where `ran` records which reference
    code actually executed. The reference dispatches itself (§04b.1: chunked prefill, then
    the `use_precomputed_states and seq_len == 1` recurrent branch); this driver never
    chooses a scan for it, and refuses if the scan it called is not the one §04b.1's table
    says should have run.
    """
    M, cfg_text, DynamicCache, hf_version = _hf_modules()
    blk, _share = build_hf_gdn_block(block, w, g, cfg_text, M)
    cache = DynamicCache(config=cfg_text)
    if len(cache.layers) <= block.index:
        raise AssertionError(f"the reference cache has {len(cache.layers)} layers, asked for "
                             f"{block.index}")
    layer = cache.layers[block.index]
    if type(layer).__name__ != "LinearAttentionLayer":
        raise AssertionError(f"layer {block.index} is {type(layer).__name__} in the reference "
                             "cache, not LinearAttentionLayer — the comparison would be running "
                             "the full-attention mixer")
    if layer.record_past:
        # §04.3's cached conv *update* path (causal_conv1d_update), not the full-window one.
        layer.record_past = False
    B = x_prefill.shape[0]
    if B != 1:
        raise AssertionError("batch 1 only, as in gdn_block")
    if x_prefill.ndim != 3 or x_steps.ndim != 3:
        raise AssertionError("expects a [B,T,d] prefill and [B,n_steps,d] steps")

    ran = {"transformers": hf_version,
           "scan_impl": {"chunked": M.torch_chunk_gated_delta_rule.__qualname__,
                         "recurrent": M.torch_recurrent_gated_delta_rule.__qualname__,
                         "module": M.torch_chunk_gated_delta_rule.__module__},
           "conv_impl": {"fn": M.causal_conv1d_fn.__qualname__,
                         "update": M.causal_conv1d_update.__qualname__},
           "cache_layer": type(layer).__name__, "record_past": bool(layer.record_past),
           "scans_seen": [], "l2norm_in_kernel": []}

    def episode(x):
        cap = _HfCapture()
        cap.install(M, blk)
        try:
            with torch.no_grad():
                y = blk(x, cache_params=cache)
        finally:
            cap.restore(M)
        return cap, y

    def harvest(cap, y, T, mode):
        want_scan = "chunked" if mode == "prefill" else "recurrent"
        st, seam = {}, {}
        st["mixed_qkv"] = cap.one("hook:in_proj_qkv")["output"]
        st["z"] = cap.one("hook:in_proj_z")["output"].reshape(B, T, g.h_v, g.d_v)
        st["b_raw"] = cap.one("hook:in_proj_b")["output"]
        st["a_raw"] = cap.one("hook:in_proj_a")["output"]
        # The pre-activation conv output is the tensor handed to ACT2FN["silu"] *inside* the
        # conv (the function returns only the post-SiLU value); §04.7's gate is the other
        # silu call and is F32/2-D, so the two are told apart by dtype — and either being
        # absent or duplicated is a broken capture, not an empty comparison.
        conv_silu = [t for t in cap.hook_io.get("silu", []) if t.dtype is torch.bfloat16]
        gate_silu = [t for t in cap.hook_io.get("silu", []) if t.dtype is not torch.bfloat16]
        if len(conv_silu) != 1 or len(gate_silu) != 1:
            raise AssertionError(f"HF capture: {len(conv_silu)} conv-silu and "
                                 f"{len(gate_silu)} gate-silu calls in one {mode} episode; "
                                 "expected exactly one of each")
        st["conv_pre"] = conv_silu[0]
        st["conv_out"] = (cap.one("conv_fn_out")["output"] if mode == "prefill"
                          else cap.one("conv_upd_out")["output"])
        # The window is the reference cache's own buffer, cloned immediately after the pass:
        # prefill fills it in `update_conv_state`, decode in `causal_conv1d_update`'s in-place
        # copy_, and the buffer has a static address that the next step overwrites.
        if not layer.is_conv_states_initialized[0]:
            raise AssertionError("the reference cache holds no conv window; `conv_window` would "
                                 "have no HF side to compare against")
        st["conv_window"] = layer.conv_states[0].detach().clone()
        if mode == "prefill" and cap.calls.get("conv_upd", 0):
            raise AssertionError("a prefill episode called causal_conv1d_update")
        if mode == "decode" and cap.calls.get("conv_fn", 0):
            raise AssertionError("a decode episode called causal_conv1d_fn — §04b.1's table says "
                                 "the cached path is causal_conv1d_update")
        # §04.2's split, transcribed on THIS side from the reference's own source line rather
        # than reused from `split_heads`, so the two sides do not share an expression; the
        # `seam` rows below are what witness it against the reference's own scan arguments.
        flat = st["conv_out"].transpose(1, 2)
        q, k, v = torch.split(flat, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        st["q_pre"] = q.reshape(B, T, g.h_k, g.d_k)
        st["k_pre"] = k.reshape(B, T, g.h_k, g.d_k)
        st["v_pre"] = v.reshape(B, T, g.h_v, g.d_v)

        scan_which, scan_a, scan_kw = cap.one("scan")["output"]
        ran["scans_seen"].append(scan_which)
        if scan_which != want_scan:
            raise AssertionError(f"{mode}: the reference called the {scan_which} scan; §04b.1's "
                                 f"table says {want_scan}, so the dispatch is not what this "
                                 "section is comparing")
        ran["l2norm_in_kernel"].append(bool(scan_kw.get("use_qk_l2norm_in_kernel")))
        if not scan_kw.get("use_qk_l2norm_in_kernel"):
            raise AssertionError("the reference did not ask its scan for the q/k l2norm, so "
                                 "q_hat/k_hat would be compared against nothing")
        q_arg, k_arg, v_arg = scan_a[0], scan_a[1], scan_a[2]
        g_arg, beta_arg = scan_kw["g"], scan_kw["beta"]
        st["beta"], st["g_decay"] = beta_arg, g_arg
        f32 = torch.float32

        def widen(t):
            return t.transpose(1, 2).to(f32, memory_format=torch.contiguous_format)

        q_f32, k_f32, v_f32 = widen(q_arg), widen(k_arg), widen(v_arg)
        st["k_hat"] = M.l2norm(k_f32, dim=-1, eps=g.l2norm_eps)
        st["q_hat"] = M.l2norm(q_f32, dim=-1, eps=g.l2norm_eps)
        # §04.4's q-scaling footnote: the two reference scans spell 1/sqrt(d_k) differently
        # (chunk multiplies by d**-0.5, recurrent divides by d**0.5) and the reference side
        # mirrors whichever one RAN, not whichever we prefer.
        if scan_which == "chunked":
            st["q_hat"] = st["q_hat"] * (st["q_hat"].shape[-1] ** -0.5)
        else:
            st["q_hat"] = st["q_hat"] / (st["q_hat"].shape[-1] ** 0.5)

        core_out, state_out = cap.one("scan_out")["output"]
        st["core_attn"] = core_out
        if state_out is None:
            raise AssertionError("the reference returned no recurrent state; "
                                 "output_final_state was not honoured, so `state` has no HF side")
        st["state"] = state_out
        st["normed"] = cap.one("hook:norm")["output"].reshape(B, T, g.h_v, g.d_v)
        st["y"] = y
        if cap.one("hook:out_proj")["output"].data_ptr() != y.data_ptr():
            raise AssertionError("the out_proj hook and the block return are different tensors "
                                 "— `y` is not what the block handed back")

        # The F32 `core_attn_out` never leaves the reference's scan: it is cast to the
        # activation dtype before the function returns. Materialise it by calling the
        # reference's OWN scan with F32 inputs, which makes its `initial_dtype` F32 and the
        # returned tensor uncast — and witness that it really is that tensor, by checking its
        # bf16 cast against the captured `core_attn`. Without that witness this row would be a
        # second implementation being compared to the first.
        fn = (M.torch_chunk_gated_delta_rule if scan_which == "chunked"
              else M.torch_recurrent_gated_delta_rule)
        # The side call takes the SAME argument shapes the block used ([B,T,h,d] and
        # [B,T,h_v]); only the dtype differs, and bf16 -> f32 is an exact widening, so the
        # function's own transpose+cast afterwards reproduces its own internals bit for bit.
        kw = {"g": g_arg.to(f32), "beta": beta_arg.to(f32),
              "use_qk_l2norm_in_kernel": True, "output_final_state": True}
        init = scan_kw.get("initial_state")
        kw["initial_state"] = None if init is None else init.detach().clone()
        if scan_which == "chunked":
            C = scan_kw.get("chunk_size", CHUNK_WIDTH_REF)
            if C != g.chunk_width:
                raise AssertionError(f"the reference chunked at C={C}, the oracle at "
                                     f"C={g.chunk_width}; §04b.2's width claim is broken")
            kw["chunk_size"] = C
        side_out, side_state = fn(q_arg.to(f32), k_arg.to(f32), v_arg.to(f32), **kw)
        act = st["core_attn"].dtype
        # The scan returns [B,T,h_v,d_v] in `initial_dtype`; with F32 inputs that dtype is F32,
        # so `side_out` is the reference's internal core_attn_out before its exit cast. The
        # witness is that casting it back reproduces the captured bf16 tensor exactly.
        seam["scan_out_cast_reproduces_core_attn"] = _bit_diff(
            side_out.to(act, memory_format=torch.contiguous_format), st["core_attn"])
        seam["scan_out_reproduces_state"] = _bit_diff(side_state, st["state"])
        # The oracle keeps the scan output in the scan's own [B,h_v,T,d_v] layout
        # (`gdn_block` transposes it to form `core_attn`), so the comparison reads it there.
        st["scan_out"] = side_out.transpose(1, 2).contiguous()

        # Seam rows: the boundaries *between* captured tensors. These license the
        # `derived-from-captured` provenance of q_pre/k_pre/v_pre — a misread §04.2 split
        # could not expand to the reference's own argument.
        exp = g.expansion
        seam["q_pre_expanded_is_the_scan_argument"] = _bit_diff(
            st["q_pre"].repeat_interleave(exp, dim=2) if exp > 1 else st["q_pre"], q_arg)
        seam["k_pre_expanded_is_the_scan_argument"] = _bit_diff(
            st["k_pre"].repeat_interleave(exp, dim=2) if exp > 1 else st["k_pre"], k_arg)
        seam["v_pre_is_the_scan_argument"] = _bit_diff(st["v_pre"], v_arg)
        seam["norm_reads_the_scan_output"] = _bit_diff(
            st["core_attn"].reshape(-1, g.d_v), cap.one("hook:norm")["inputs"][0])
        seam["norm_gate_is_the_captured_z"] = _bit_diff(
            st["z"].reshape(-1, g.d_v), cap.one("hook:norm")["inputs"][1])
        if set(seam) != set(HF_SEAM_CHECKS):
            raise AssertionError(f"the seam checks computed were {sorted(seam)}, expected "
                                 f"{sorted(HF_SEAM_CHECKS)} — a missing seam row is a missing "
                                 "check, not a smaller denominator")
        missing = [s for s in STAGES if s not in st]
        if missing:
            raise AssertionError(f"the HF harvest produced no tensor for {missing}")
        if set(st) != set(STAGES):
            raise AssertionError(f"the HF harvest produced stage(s) {sorted(set(st) - set(STAGES))} "
                                 "outside the stage list")
        return st, seam

    cap, y = episode(x_prefill)
    prefill, prefill_seam = harvest(cap, y, x_prefill.shape[1], "prefill")
    decode, decode_seam = [], []
    for s in range(x_steps.shape[1]):
        x = x_steps[:, s:s + 1]
        cap, y = episode(x)
        st, seam = harvest(cap, y, x.shape[1], "decode")
        decode.append(st)
        decode_seam.append(seam)
    if ran["scans_seen"] != ["chunked"] + ["recurrent"] * x_steps.shape[1]:
        raise AssertionError(f"§04b.1 dispatch not observed: {ran['scans_seen']}")
    if not all(ran["l2norm_in_kernel"]):
        raise AssertionError("the reference stopped asking for the in-kernel l2norm on some "
                             "pass; q_hat/k_hat have no HF side there")
    return {"prefill": prefill, "decode": decode, "prefill_seam": prefill_seam,
            "decode_seam": decode_seam}, ran


# ========================================================================= the contract
# §The contract tables. One table per fact about an injection, and every one of them is
# keyed by a `Faults` field name so `dataclasses.fields(Faults)` is the single source of the
# case list: an injection cannot be added without appearing in all of them, and `main()`
# refuses to run if the key sets disagree (`pitfalls/40` §two self-consistent layouts joined
# by a branch are invisible to both sides).

_D_CONV = ["conv_pre", "conv_out", "q_pre", "k_pre", "v_pre", "q_hat", "k_hat",
           "scan_out", "core_attn", "state", "normed", "y"]
_D_SCAN = ["scan_out", "core_attn", "state", "normed", "y"]
# The stages a q-side fault can reach. `state` is deliberately ABSENT: no read of q writes
# the state (§04.6 steps 1-5), so a fault that moves the state here is the injection leaking.
_D_SCAN_NO_STATE = ["scan_out", "core_attn", "normed", "y"]
_D_NORM = ["normed", "y"]

# The UNCONDITIONAL half of each case's contract: which stages the injection moves whatever
# the data does. Everything not listed must stay bit-identical, and that is checked.
AFFECTED = {
    "qkv_split_interleaved": ["q_pre", "k_pre", "v_pre", "q_hat", "k_hat",
                              "scan_out", "core_attn", "state", "normed", "y"],
    # Not `conv_window` for prefill: the prefill window is the raw `mixed_qkv` tail, which the
    # misplaced activation never touches — only the decode window is assembled from the columns
    # handed to the conv, so that half is CONDITIONAL_AFFECTED.
    "silu_before_conv": ["conv_pre", "conv_out", "q_pre", "k_pre", "v_pre",
                         "q_hat", "k_hat", "scan_out", "core_attn", "state",
                         "normed", "y"],
    "conv_tap_rotate": _D_CONV,
    "conv_tap_reverse": _D_CONV,
    # §10.4: "read the conv state as outputs instead of inputs -> the conv window, and every
    # subsequent step". The window moving is the fired witness, not a leak.
    "conv_state_is_outputs": ["conv_window"] + _D_CONV,
    "beta_no_sigmoid": ["beta"] + _D_SCAN,
    "softplus_on_a_only": ["g_decay"] + _D_SCAN,
    "a_log_used_directly": ["g_decay"] + _D_SCAN,
    "decay_sign_flip": ["g_decay"] + _D_SCAN,
    "l2norm_eps_outside": ["q_hat", "k_hat"] + _D_SCAN,
    "l2norm_uses_mean": ["q_hat", "k_hat"] + _D_SCAN,
    "q_scale_omitted": ["q_hat"] + _D_SCAN_NO_STATE,
    "q_scaled_by_dv": ["q_hat"] + _D_SCAN_NO_STATE,
    "head_pairing_tiled": ["q_hat", "k_hat"] + _D_SCAN,
    # A-12's tensor list: attn_qkv V third, attn_gate, ssm_alpha/beta, ssm_a, ssm_dt.bias,
    # ssm_conv1d V channels, ssm_out columns. Q/K are NOT in that list, so q_pre/k_pre are
    # must-not-catch here — which is what makes this distinguishable from a whole-tensor bug.
    # `conv_window` is in the blast radius: the window covers all `conv_dim` channels, V
    # included, so a V-head reordering changes the columns the next decode step reads.
    "v_head_order_tiled": ["mixed_qkv", "conv_pre", "conv_out", "conv_window", "v_pre", "z",
                           "a_raw", "b_raw", "beta", "g_decay"] + _D_SCAN,
    "read_pre_update_state": _D_SCAN_NO_STATE,
    "decay_after_write": _D_SCAN,
    # §04.4's scan-dtype row is "F32 for the state, the decay, **the norms** and the output of
    # the scan": the two l2norms that produce `q_hat`/`k_hat` are inside the scan dtype, so a
    # bf16 scan moves them and they are must-CATCH here, not a leak. `beta`/`g_decay` stay
    # must-not-catch: they are produced before the scan, `g_decay` in F32 by §04.5's own
    # explicit casts, so the scan's storage dtype cannot reach either.
    "scan_in_bf16": ["q_hat", "k_hat"] + _D_SCAN,
    "norm_plus_one": _D_NORM,
    "norm_gate_first": _D_NORM,
    "norm_weight_f32": _D_NORM,
    "norm_variance_flat": _D_NORM,
    # Its base is EMPTY on purpose: what this injection can reach is entirely positional —
    # `chunk_decay[i]` only matters where a later chunk or the final state reads it. Both
    # routes are CONDITIONAL_AFFECTED, each with a measured reason, so no fixture can be
    # judged against a stage it structurally cannot move.
    "chunk_decay_off_by_one": [],
    # No `state`: the strictly-upper entries of `ut_system` are never read, because
    # `solve_triangular(upper=False)` reads only the triangle it is solving — measured, the
    # state is bit-identical under this injection on every fixture while the outputs move by
    # up to 6.3e-02. `intra_attn`, by contrast, is consumed by a dense matmul, so its NaNs
    # reach `core_attn`, `scan_out`, `normed` and `y`.
    "pairwise_mask_after_exp": ["scan_out", "core_attn", "normed", "y"],
    "unitriangular_dropped": _D_SCAN,
    "beta_on_v_not_k": _D_SCAN,
    "q_decayed_twice": _D_SCAN_NO_STATE,
    "state_stored_transposed": _D_SCAN,
    # `state` only. The pad columns are the tail of the FINAL chunk by construction, so their
    # corrupted write lands in the state that is returned and never read back: measured on
    # `prefill_3chunk_dirty_pad`, `state` moves by 2.105e+30 (against a clean absmax of 0.89)
    # while `scan_out`, `core_attn`, `normed` and `y` are bit-identical.
    "pad_beta_ones": ["state"],
}

# Which pass shape an injection is definable on. `read_pre_update_state` and
# `decay_after_write` are orderings *inside one recurrent step* (§04.6): the chunked form
# has no per-token step to reorder, so running them there would be a fault against nothing.
APPLIES = {
    "qkv_split_interleaved": "both", "silu_before_conv": "both",
    "conv_tap_rotate": "both", "conv_tap_reverse": "both",
    "conv_state_is_outputs": "decode",
    "beta_no_sigmoid": "both", "softplus_on_a_only": "both", "a_log_used_directly": "both",
    "decay_sign_flip": "both", "l2norm_eps_outside": "both", "l2norm_uses_mean": "both",
    "q_scale_omitted": "both", "q_scaled_by_dv": "both", "head_pairing_tiled": "both",
    "v_head_order_tiled": "both", "read_pre_update_state": "decode",
    "decay_after_write": "decode", "scan_in_bf16": "both",
    "norm_plus_one": "both", "norm_gate_first": "both", "norm_weight_f32": "both",
    "norm_variance_flat": "both",
    "chunk_decay_off_by_one": "prefill", "pairwise_mask_after_exp": "prefill",
    "unitriangular_dropped": "prefill", "beta_on_v_not_k": "prefill",
    "q_decayed_twice": "prefill", "state_stored_transposed": "both",
    "pad_beta_ones": "prefill",
}


def _always(g, f, C):
    return True, ""


# The geometric preconditions, each with the reason a case would be dropped. A case whose
# precondition fails is UNREACHABLE, printed as such — never silently absent.
_BF16_SOLVE_PROBE = None


def bf16_solve_supported() -> tuple:
    """Witnessed, not assumed: does this interpreter's CPU backend implement a bf16
    triangular solve? `§04b.4` runs the UT solve in `mamba_ssm_dtype` (float32 per the
    committed config), so the honest arm never needs bf16 here. The `scan_in_bf16` *injection*
    now runs its solve in F32 and casts back (see `REQUIRES["scan_in_bf16"]`), which is the
    reference's own width — so a False here no longer skips the arm, it bounds what the arm
    can claim, and is printed in `NOT_ESTABLISHED` with the measured error. A skip whose
    reason was never observed is a guess (`pitfalls/40`)."""
    global _BF16_SOLVE_PROBE
    if _BF16_SOLVE_PROBE is None:
        try:
            a = torch.eye(2, dtype=torch.bfloat16)
            torch.linalg.solve_triangular(a, torch.ones(2, 1, dtype=torch.bfloat16),
                                          upper=False, unitriangular=True)
            _BF16_SOLVE_PROBE = (True, "bf16 solve_triangular ran")
        except Exception as exc:                                   # noqa: BLE001
            _BF16_SOLVE_PROBE = (False, f"{type(exc).__name__}: {exc}")
    return _BF16_SOLVE_PROBE


REQUIRES = {
    # The arm RUNS: the reference keeps the UT solve in F32 whatever the scan's storage dtype
    # (§04.4's scan-dtype row, §04b.4, and the reference casts q/k/v/beta/decay to float32 at
    # the scan entry before it solves — `modeling_qwen3_5_moe.py`, in
    # `torch_chunk_gated_delta_rule`). So a bf16 scan that solves in F32 and casts back is the
    # arm that can be executed on this backend; what it cannot establish — the numerics of a
    # bf16 SOLVE — is printed in NOT_ESTABLISHED with the probed error, never folded into the
    # arm as if it had run. See `bf16_solve_supported()`.
    "scan_in_bf16": (lambda g, f, C: True,
                     "runs with the UT solve in F32 (the reference's own width) and cast back; "
                     "the bf16 triangular solve itself is probed and reported in "
                     "NOT_ESTABLISHED"),
    "conv_tap_rotate": (lambda g, f, C: g.k_conv >= 2,
                        "k_conv >= 2: a rotation needs at least two taps"),
    "conv_tap_reverse": (lambda g, f, C: g.k_conv >= 3,
                         "k_conv >= 3: for k_conv == 2 a reversal is a rotation"),
    "head_pairing_tiled": (lambda g, f, C: g.expansion > 1,
                          "h_v > h_k: grouped vs tiled pairing needs an expansion"),
    "v_head_order_tiled": (lambda g, f, C: g.expansion > 1,
                           "h_v > h_k: see head_pairing_tiled"),
    "q_scaled_by_dv": (lambda g, f, C: g.d_k != g.d_v,
                       "needs d_k != d_v; measured d_k == d_v (config: 128 == 128, see "
                       "geometry.degeneracies and NOT_ESTABLISHED), so 1/sqrt(d_v) IS "
                       "1/sqrt(d_k) and no fixture can distinguish the two"),
    # With d_k != d_v torch raises on the transposed read, and that raise is the catcher;
    # the injection is only a numerics test in the square-head geometry this model has.
    "state_stored_transposed": (lambda g, f, C: g.d_k == g.d_v,
                               "d_k == d_v: otherwise the shape error catches it"),
    "chunk_decay_off_by_one": (lambda g, f, C: (f.n_chunks(g, C) >= 2
                                                and f.n_pad(g, C) <= 1)
                               or f.n_chunks(g, C) >= 3,
                              "either a final chunk whose last two columns are real tokens "
                              "(the state can see the off-by-one) or three chunks (an output "
                              "can). Otherwise it is a no-op on this fixture, as measured — "
                              "see CONDITIONAL_AFFECTED and tmp/w20_chunkdecay.py"),
    "pairwise_mask_after_exp": (lambda g, f, C: f.n_tokens >= (C or g.chunk_width),
                                "n_tokens >= chunk_width: only real-token columns carry a "
                                "nonzero g, so the masked triangle has to overflow before the "
                                "two spellings differ, and a 4-token fixture cannot make it "
                                "do so on every block (witnessed: 617 inf entries at T=4 on "
                                "layer0, none on layer1) — see tmp/w20_primitives.py"),
    "pad_beta_ones": (lambda g, f, C: f.n_pad(g, C) > 0 and f.pad_noise != 0,
                      "n_pad > 0 and a fixture whose padding columns are non-zero: with "
                      "q/k/v zero-padded (§04b.2) a pad row's write is k_pad^T @ 0 for any "
                      "beta, so the injection is provably inert there — measured on all three "
                      "zero-padded fixtures"),
    "q_decayed_twice": (lambda g, f, C: f.n_chunks(g, C) >= 2,
                        "n_chunks >= 2: inside the chunked form q is used twice — undecayed in "
                        "`intra_attn`, decayed against the carried state — and with one chunk "
                        "that state is the zero matrix, so the second decay cannot be seen "
                        "(§04b.4)"),
    "state_stored_transposed": (lambda g, f, C: f.n_chunks(g, C) >= 2,
                                "n_chunks >= 2: the transposed layout is only read by the "
                                "inter-chunk state read; one chunk reads the zero state"),
}
for _f in FAULT_FIELDS:
    REQUIRES.setdefault(_f, (_always, ""))

# The CONDITIONAL half, and it is conditional on **geometry**, not on data: stages a case may
# move only when a structural fact about the fixture holds. Read by BOTH `allowed_stages()`
# and `derive_expected_allowed_stages()`, one table, two readers (pitfalls/40 §must-catch AND
# must-not-catch sets must be derived from the geometry). Value: fault -> [(stages, predicate,
# why), ...].
#
# `chunk_decay_off_by_one` is the case that forced this table, and both of its clauses are
# witnessed by `tmp/w20_chunkdecay.py` rather than asserted:
#   * `chunk_decay[i]` multiplies the state that chunk `i+1` reads. With **two** chunks,
#     `chunk_decay[0]` multiplies the zero initial state, so it is a no-op and no output can
#     see the off-by-one — measured on T=128: `state` moves by 7.05e-03 while `scan_out`,
#     `core_attn`, `normed` and `y` are bit-identical (ndiff 0/524288).
#   * the final chunk's decay is `exp(cum[-1])`; if its last column is a pad row, `cum[-1]`
#     equals `cum[-2]` (a pad row contributes `g = 0`, §04b.2) and the off-by-one cannot even
#     reach the state — measured on T=96 (`n_pad = 32`): every stage bit-identical.
# A fixture set with no three-chunk case would therefore contain **no test at all** for this
# §10.4 injection, which is why `prefill_3chunk_padded` exists.
def _n_chunks(g, fx, C):
    return fx.n_chunks(g, C)


CONDITIONAL_AFFECTED = {
    "silu_before_conv": [
        (["conv_window"], lambda g, fx, C: fx.kind == "decode",
         "decode only: the decode window is assembled from the columns handed to the conv, "
         "which under this fault are already SiLU'd; the prefill window is the raw mixed_qkv "
         "tail (§04.3) and cannot move"),
    ],
    "chunk_decay_off_by_one": [
        (["state"],
         lambda g, fx, C: (_n_chunks(g, fx, C) >= 3
                           or (_n_chunks(g, fx, C) >= 2 and fx.n_pad(g, C) <= 1)),
         "a differing chunk_decay with index >= 1 survives into the final state: three chunks "
         "(chunk 1 is full, so its decay differs and is multiplied in even when the last "
         "chunk is padded), or two chunks whose final column pair is real"),
        (["scan_out", "core_attn", "normed", "y"],
         lambda g, fx, C: _n_chunks(g, fx, C) >= 3,
         "n_chunks >= 3: with two chunks chunk_decay[0] multiplies the zero initial state"),
    ],
}


def _cond_groups(fault: str):
    return CONDITIONAL_AFFECTED.get(fault, [])


def _state_must_be_untouched(fault: str) -> bool:
    """True for the faults whose whole point is that the state does not move: no read of `q`
    writes the state (§04.6 steps 1-5), and the pre/post-update read is a read. If any of
    them moves `state`, the injection leaked and that is a bug in this oracle, not in the
    contract."""
    return (fault in ("q_scale_omitted", "q_scaled_by_dv", "q_decayed_twice",
                      "read_pre_update_state")
            and "state" not in AFFECTED[fault])


def cases_for(g: GdnGeometry, fixtures: list, width: int = 0, only=None) -> tuple:
    """The derived case list: (fault, param, fixture) triples. Every multiplicity is a
    function of the geometry — the tap rotations are `k_conv - 1` because a rotation by
    `k_conv` is the identity, exactly as `selftest_moe.py` derives `two_i / I - 1`.

    `only` is the debug filter and is applied *here and in `populations`*, so a subset run
    still checks its own two derivations against each other instead of failing on a
    denominator it was never asked to cover."""
    cases, skipped = [], []
    for fault in FAULT_FIELDS:
        if only is not None and fault not in only:
            continue
        for fx in fixtures:
            kind = APPLIES[fault]
            if kind != "both" and kind != fx.kind:
                continue
            ok = REQUIRES[fault][0](g, fx, width)
            reason = REQUIRES[fault][1]
            if not ok:
                skipped.append((fault, fx.name, reason))
                continue
            if fault == "conv_tap_rotate":
                for shift in range(1, g.k_conv):          # rotation by k_conv is identity
                    cases.append((fault, shift, fx))
            else:
                cases.append((fault, None, fx))
    return cases, skipped


def populations(g: GdnGeometry, fixtures: list, width: int = 0, only=None) -> dict:
    """The expected size of every section, as independent arithmetic on the geometry and the
    fixture list. `cases_for` builds the list; this counts what the list must contain, and
    `main()` fails the run when the two disagree — the 192-vs-184 lesson
    (`speccheck/results.md` §The 192-vs-184 dispute)."""
    pop = {}
    for fault in FAULT_FIELDS:
        if only is not None and fault not in only:
            continue
        ok_any = any(REQUIRES[fault][0](g, f, width)
                     for f in fixtures if APPLIES[fault] in ("both", f.kind))
        if not ok_any:
            pop[fault] = 0
            continue
        mult = 1
        if fault == "conv_tap_rotate":
            mult = g.k_conv - 1
        n = 0
        for fx in fixtures:
            if APPLIES[fault] not in ("both", fx.kind):
                continue
            if not REQUIRES[fault][0](g, fx, width):
                continue
            n += mult
        pop[fault] = n
        assert n >= 0
    return pop


# ================================================================== running one case
def run_pass(g: GdnGeometry, w: GdnWeights, fx: Fixture, arm: str, faults: Faults,
             cache: dict | None = None) -> dict:
    """One fixture under one arm with one fault set. Returns the stage dict plus `_facts`.

    The decode fixture replays §04b.1's real dispatch — chunked prefill, recurrent cached
    decode — because that is the path the engine takes and the path a conv-state corruption
    can only be seen on. The prefill that produces the states is always run CLEAN: a
    decode-path injection must not be smuggled into the state it is meant to corrupt."""
    if fx.kind == "prefill":
        x = gdn_input(fx.n_tokens, g, fx.seed).unsqueeze(0)
        st = dict(gdn_block(x, w, g, arm, faults, mode="prefill", pad_noise=fx.pad_noise))
        st["_facts"] = {}
        return st
    x = gdn_input(fx.n_tokens + fx.n_steps, g, fx.seed).unsqueeze(0)
    key = (w.block.name, fx.name, "clean-prefill")
    if cache is not None and key in cache:
        pf = cache[key]
    else:
        pf = gdn_block(x[:, :fx.n_tokens], w, g, "chunked", NO_FAULTS, mode="prefill",
                       pad_noise=fx.pad_noise)
        if cache is not None:
            cache[key] = pf
    seq = gdn_sequence(x[:, :fx.n_tokens], x[:, fx.n_tokens:], w, g, "chunked", "recurrent",
                       faults=NO_FAULTS, decode_faults=faults, prefill_stages=pf,
                       pad_noise=fx.pad_noise)
    st = dict(seq["decode"])
    st["_facts"] = {"step_y": seq["step_y"], "n_steps": seq["decode_steps"],
                    "state_after_prefill": pf["state"]}
    return st


def compute_facts(clean: dict, dirty: dict) -> dict:
    """The discrete facts, computed with `torch.equal` — never through the differ's
    predicate, so the stub arm cannot blind them (`selftest_moe.py` §verdict_flips)."""
    facts = {}
    a, b = clean["state"], dirty["state"]
    facts["state_identical"] = bool(a.dtype == b.dtype and a.shape == b.shape
                                   and torch.equal(a, b))
    ys = clean.get("_facts", {}).get("step_y")
    if ys is not None:
        facts["step_y_moved"] = [bool(not torch.equal(u, v))
                                 for u, v in zip(ys, dirty["_facts"]["step_y"])]
    return facts


def allowed_stages(fault: str, g: GdnGeometry, fx: Fixture, width: int = 0) -> tuple:
    """The contract for one case. The conditional stages are resolved from the fixture's
    geometry — evaluated before anything runs, so the differ cannot license its own result."""
    allowed = set(AFFECTED[fault])
    licensed = []
    for stages, pred, _why in _cond_groups(fault):
        if pred(g, fx, width):
            allowed |= set(stages)
            licensed += [s for s in stages if s not in licensed]
    return allowed, licensed


def derive_expected_allowed_stages(cases: list, g: GdnGeometry, width: int = 0) -> tuple:
    """The (case x allowed-stage) pairs the contract licenses — the stub arm's non-zero
    expectation, never read back off a tally."""
    base = conditional = 0
    detail = []
    for fault, param, fx in cases:
        base += len(AFFECTED[fault])
        for stages, pred, _why in _cond_groups(fault):
            if pred(g, fx, width):
                conditional += len(stages)
                detail.append(f"{len(stages)} stage(s) {','.join(stages)} licensed by "
                              f"geometry on {fault}/{fx.name}")
    return base, conditional, detail


def case_key(fault: str, param, fx: Fixture) -> str:
    """One key shape, shared by the fact record and the verdict that re-derives the case
    set — the same expression on both sides."""
    return f"{fault}|{param}|{fx.name}"


# =================================================================== the three arms
ARM_BLIND = "stub"


def run_matrix(g, blocks, cases, args, arm, cache):
    """One pass over the derived case matrix under one differ policy.

    honest   the differ as it ships; carries the localisation claim.
    stub     the "did this stage move" predicate replaced by `lambda: False`; every case
             must then lose its must-catch, so a `caught` count above 0 here means the
             assertion is not reading the differ.
    crossed  a deliberate must-not-catch breach per case, on a stage its contract says must
             stay identical; the localisation check must fire on exactly that stage.
    """
    predicate = (lambda changed: False) if arm == ARM_BLIND else None
    crossed = arm == "crossed"
    tally = {"cases": 0, "caught": 0, "localised": 0, "stage_diffs": 0,
             "scan_moved_cases": 0, "pre_scan_moved_cases": 0,
             "must_not_catch_violations": 0, "must_catch_missed": 0,
             "crossed_breaches_injected": 0, "crossed_alarms_fired": 0}
    rows, fails, facts_by_case = [], [], {}
    breach_cycle = 0
    for block in blocks:
        w = load_gdn_block_weights(block, g)
        for fault, param, fx in cases:
            faults = _faults_from(fault, param)
            clean = run_pass(g, w, fx, _arm_for(fx), NO_FAULTS, cache)
            dirty = run_pass(g, w, fx, _arm_for(fx), faults, cache)
            facts = compute_facts(clean, dirty)
            facts_by_case[case_key(fault, param, fx)] = facts
            allowed, licensed = allowed_stages(fault, g, fx, 0)

            breach = None
            if crossed:
                cands = [s for s in STAGES if s not in allowed
                         and stage_moved(clean[s], dirty[s])[0] is False]
                if not cands:
                    fails.append(f"{arm}/{block.name}/{case_key(fault, param, fx)}: no "
                                 "must-not-catch stage is currently identical, so the "
                                 "breach cannot be injected")
                    continue
                breach = cands[breach_cycle % len(cands)]      # rotated, not always [0]
                breach_cycle += 1
                dirty = dict(dirty)
                step = R.ulp_of(float(clean[breach].abs().max().item())) or 1.0
                dirty[breach] = clean[breach] + step
                tally["crossed_breaches_injected"] += 1

            moved_now, violations, max_seen = set(), [], 0.0
            for stage in STAGES:
                moved, max_abs = stage_moved(clean[stage], dirty[stage], predicate)
                if max_abs == max_abs and max_abs != float("inf"):
                    max_seen = max(max_seen, max_abs)
                if moved:
                    moved_now.add(stage)
                    tally["stage_diffs"] += 1
                if (stage in allowed) != (stage in moved_now):
                    violations.append(f"{stage}:"
                                      + ("silent" if stage in allowed else "moved"))
            tally["cases"] += 1
            if moved_now & allowed:
                tally["caught"] += 1
            tally["must_catch_missed"] += len([v for v in violations
                                               if v.endswith("silent")])
            if moved_now & {"scan_out", "state", "core_attn"}:
                tally["scan_moved_cases"] += 1
            if moved_now & {"mixed_qkv", "conv_pre", "conv_out", "beta", "g_decay"}:
                tally["pre_scan_moved_cases"] += 1
            if not violations:
                tally["localised"] += 1
            else:
                tally["must_not_catch_violations"] += len(violations)
            if crossed and breach is not None:
                if any(v.startswith(breach + ":moved") for v in violations):
                    tally["crossed_alarms_fired"] += 1
                else:
                    fails.append(f"{arm}/{block.name}/{case_key(fault, param, fx)}: injected "
                                 f"a breach of {breach} and the localisation check did not "
                                 "notice")
            if violations and not (crossed or predicate is not None):
                fails.append(f"{arm}/{block.name}/{case_key(fault, param, fx)}: "
                             + "; ".join(violations))
            if not moved_now and not crossed and predicate is None:
                fails.append(f"{arm}/{block.name}/{case_key(fault, param, fx)}: NO stage "
                             "moved — the injection is vacuous")
            # §10.4's "and every subsequent step", adjudicated rather than asserted.
            if fault == "conv_state_is_outputs" and not crossed and predicate is None:
                mask = facts.get("step_y_moved", [])
                if len(mask) < 2 or mask[0] or not all(mask[1:]):
                    fails.append(f"{arm}/{block.name}/conv_state_is_outputs: the deferred-fire "
                                 f"claim failed — step_y_moved={mask}, expected "
                                 "[False, True, ...]")
            rows.append((block.name, fault, param, fx.name, fx.n_tokens, sorted(moved_now),
                         len(violations), max_seen, breach or "",
                         ",".join(licensed)))
        del w
    return tally, rows, fails, facts_by_case


def _arm_for(fx: Fixture) -> str:
    return "chunked" if fx.kind == "prefill" else "recurrent"


def _faults_from(fault: str, param) -> Faults:
    if fault == "conv_tap_rotate":
        return Faults(conv_tap_rotate=int(param))
    return Faults(**{fault: True})


TALLY_KEYS = ("cases", "caught", "localised", "stage_diffs",
              "scan_moved_cases", "pre_scan_moved_cases",
              "must_not_catch_violations", "must_catch_missed",
              "crossed_breaches_injected", "crossed_alarms_fired")

# The non-vacuity witnesses, and the expectation each is held to — derived from the case
# list, never `> 0`. A witness asserted against `> 0` is a check a debug subset run can only
# fail spuriously, and a counter asserted against nothing is a comment wearing a counter
# (pitfalls/40 §a check whose predicate is `True`).
WITNESS_SETS = {"scan_moved_cases": {"scan_out", "state", "core_attn"},
                "pre_scan_moved_cases": {"mixed_qkv", "conv_pre", "conv_out", "beta",
                                         "g_decay"}}


def expected_witnesses(cases: list, g: GdnGeometry, n_blocks: int) -> dict:
    """One derivation with `allowed_stages` — base stages AND the conditionals its geometry
    licenses. Deriving the expectation from the base alone under-counts by exactly the cases
    whose only reachable scan stages are conditional, which is how `chunk_decay_off_by_one`
    on a three-chunk fixture came to move `scan_out` while the expectation called it
    unexpected."""
    return {k: sum(1 for f, _p, fx in cases
                   if allowed_stages(f, g, fx, 0)[0] & s) * n_blocks
            for k, s in WITNESS_SETS.items()}


def print_table(arm, rows):
    print(f"\n## firing table — arm={arm} (the comparison is bit-level; no band is involved)")
    print(f"  {'block':<7} {'fault':<24} {'param':>5} {'fixture':<21} {'T':>4} "
          f"{'moved stages':<58} {'viol':>4} maxabs {'breach':<22} {'licensed':<8}")
    for r in rows:
        print(f"  {r[0]:<7} {r[1]:<24} {str(r[2]):>5} {r[3]:<21} {r[4]:>4} "
              f"{(','.join(r[5]) if r[5] else '(NOTHING MOVED)'):<58} {r[6]:>4} "
              f"{r[7]:.2e} {r[8]:<22} {r[9]:<8}")


def print_tally(arm, tally):
    print(f"\n## tallies — arm={arm}")
    for k in TALLY_KEYS:
        print(f"  {arm:<8} {k:<28} {tally[k]}")


def verdict_honest(tally, expected_cases, expected_wit, fails, tag):
    if tally["cases"] != expected_cases:
        fails.append(f"{tag}: case count {tally['cases']} != derived {expected_cases}")
    if tally["caught"] != tally["cases"]:
        fails.append(f"{tag}: {tally['cases'] - tally['caught']} cases moved no allowed "
                     "stage — an injection that changes nothing is not a control")
    if tally["must_not_catch_violations"] != 0:
        fails.append(f"{tag}: {tally['must_not_catch_violations']} must-not-catch violations "
                     "(an injection moved a stage its contract says it cannot reach)")
    if tally["localised"] != tally["cases"]:
        fails.append(f"{tag}: only {tally['localised']}/{tally['cases']} cases localised")
    # Non-vacuity: each witness is held to a count derived from the case list, so a debug
    # subset is adjudicated against the cases it actually contains and the full gate against
    # all of them. Fewer than derived means a side of the check never fired.
    for k, want in sorted(expected_wit.items()):
        if tally[k] != want:
            fails.append(f"{tag}: witness {k} reads {tally[k]}, derived {want} from the case "
                         "list — that side of the check did not fire on every case whose "
                         "contract says it must")


def verdict_stub(tally, expected_cases, expected_allowed, fails, tag):
    if tally["stage_diffs"] != 0:
        fails.append(f"{tag}: the stub did not blind the gate (stage_diffs="
                     f"{tally['stage_diffs']})")
    if tally["caught"] != 0:
        fails.append(f"{tag}: the stubbed gate still caught {tally['caught']} cases — the "
                     "must-catch assertion is not reading the differ")
    if tally["must_catch_missed"] != expected_allowed:
        fails.append(f"{tag}: stub missed {tally['must_catch_missed']} allowed stages, "
                     f"derived {expected_allowed}")
    if expected_allowed == 0:
        fails.append(f"{tag}: derived allowed-stage population is 0; the stub arm is vacuous")


def verdict_crossed(tally, expected_cases, fails, tag):
    if tally["crossed_breaches_injected"] != expected_cases:
        fails.append(f"{tag}: injected {tally['crossed_breaches_injected']} breaches, "
                     f"expected {expected_cases}")
    if tally["crossed_alarms_fired"] != tally["crossed_breaches_injected"]:
        fails.append(f"{tag}: the localisation check missed "
                     f"{tally['crossed_breaches_injected'] - tally['crossed_alarms_fired']} "
                     "injected breaches — the must-not-catch side cannot fail, so its 0 in "
                     "arm=honest is unmeasured rather than earned")
    if tally["crossed_alarms_fired"] == 0:
        fails.append(f"{tag}: crossed arm fired 0 alarms: vacuous positive control")


def verdict_facts(arms_facts, cases, blocks, g, fails):
    """The discrete facts, computed with `torch.equal` outside the differ, are part of the
    contract and are asserted — not consulted for permission.

      - every derived case must carry a fact record, or the expectation is for a different
        case set than the tally (an unnamed denominator);
      - a fault contracted to leave `state` alone must leave it alone. If it moves, that is
        the injection leaking into a stage it cannot reach, and the must-not-catch check is
        not enough on its own because a leak can also *look* like a pass when `state` is in
        the allowed set for some other reason;
      - §10.4's "and every subsequent step" is adjudicated per step, not asserted;
      - every conditional section must be licensed by its geometry somewhere in the fixture
        set, or that half of the contract is untested rather than satisfied."""
    keys = {case_key(f, p, fx) for _b in blocks for f, p, fx in cases}
    for arm, fl in sorted(arms_facts.items()):
        if set(fl) != keys:
            fails.append(f"arm={arm}: recorded {len(fl)} facts, derived {len(keys)} — the "
                         "stub expectation would be counting a different case set")
    honest = arms_facts.get("honest", {})
    for fault, param, fx in cases:
        key = case_key(fault, param, fx)
        facts = honest.get(key, {})
        if _state_must_be_untouched(fault) and not facts.get("state_identical", False):
            fails.append(f"arm=honest/{key}: {fault} is contracted to leave `state` "
                         "bit-identical (no read of q writes the state, §04.6 steps 1-5) but "
                         "the state moved — the injection leaked")
        if fault == "conv_state_is_outputs":
            mask = facts.get("step_y_moved", [])
            if len(mask) < 2 or mask[0] or not all(mask[1:]):
                fails.append(f"arm=honest/{key}: the deferred-fire claim failed — "
                             f"step_y_moved={mask}, expected [False, True, ...] (the window "
                             "is read by the NEXT step, §10.4)")
    for fault, groups in sorted(CONDITIONAL_AFFECTED.items()):
        for stages, pred, why in groups:
            if not any(fault == f and pred(g, fx, 0) for f, _p, fx in cases):
                fails.append(f"arm=honest: no case licensed the conditional stages "
                             f"{'/'.join(stages)} of {fault} ({why}) — that half of the "
                             "contract is untested, not satisfied")
    for arm, fl in sorted(arms_facts.items()):
        if arm == "honest":
            continue
        for k in sorted(set(fl) & set(honest)):
            if fl[k] != honest[k]:
                fails.append(f"arm={arm}: fact for {k} reads {fl[k]} but arm=honest read "
                             f"{honest[k]} — the arms are not comparing one fixture")
                break


# =========================================================================== the driver
def section_selftest(g, args) -> list:
    fails = []
    blocks = [Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    if not blocks:
        raise AssertionError("no blocks selected; an empty matrix cannot pass")
    fixtures = default_fixtures(g)
    if args.only_fixture:
        fixtures = [f for f in fixtures if f.name in args.only_fixture.split(",")]
    if args.only_fault:
        keep = set(args.only_fault.split(","))
    else:
        keep = None
    width = 0
    cases, skipped = cases_for(g, fixtures, width, keep)
    pops = populations(g, fixtures, width, keep)
    if not cases:
        raise AssertionError("the derived case list is empty; nothing to gate")

    print(f"# command: {' '.join(sys.argv)}")
    print(f"# selftest — torch {torch.__version__} CPU threads={args.threads_used} "
          f"blocks={','.join(b.name for b in blocks)} "
          f"fixtures={','.join(f.name for f in fixtures)}")
    print(f"# geometry: h_k={g.h_k} h_v={g.h_v} d_k={g.d_k} d_v={g.d_v} k_conv={g.k_conv} "
          f"C={g.chunk_width} conv_dim={g.conv_dim} linear layers={g.n_linear_layers} "
          f"(gdn tensors in the index: {g.n_gdn_tensors})")
    if args.only_fault or args.only_fixture:
        print("# [SUBSET — NOT THE FULL GATE] --only-fault/--only-fixture in effect; the "
              "counts below are not the lane's verdict")
    print(f"# fault fields in Faults: {len(FAULT_FIELDS)}; derived cases: {len(cases)} "
          f"per block, {len(cases) * len(blocks)} total")

    # The contract tables must cover exactly the fault fields — before any comparison runs.
    for name, table in (("AFFECTED", AFFECTED), ("APPLIES", APPLIES), ("REQUIRES", REQUIRES)):
        if set(table) != set(FAULT_FIELDS):
            fails.append(f"contract: {name} keys != Faults fields: missing "
                         f"{sorted(set(FAULT_FIELDS) - set(table))}, extra "
                         f"{sorted(set(table) - set(FAULT_FIELDS))}")
    for fault, stages in AFFECTED.items():
        bad = [s for s in stages if s not in STAGES]
        if bad:
            fails.append(f"contract: AFFECTED[{fault}] names stages outside STAGES: {bad}")
    if fails:
        return fails

    print("# derived population per section (cases_for vs populations, two derivations):")
    print(f"#   {'fault':<24} {'cases_for':>9} {'populations':>11}  applies / note")
    from collections import Counter
    built = Counter(f for f, _p, _fx in cases)
    for fault in [f for f in FAULT_FIELDS if keep is None or f in keep]:
        n_built, n_pop = built.get(fault, 0), pops[fault]
        note = APPLIES[fault]
        if n_pop == 0 and fault not in g.degeneracies() and REQUIRES[fault][1]:
            note += f" — unreachable: {REQUIRES[fault][1]}"
        if fault in g.degeneracies():
            note += " — DEGENERATE for this geometry, excluded (see geometry.degeneracies)"
        print(f"#   {fault:<24} {n_built:>9} {n_pop:>11}  {note}")
        if n_built != n_pop:
            fails.append(f"population: section {fault}: cases_for {n_built} != populations "
                         f"{n_pop}")
    if sum(pops.values()) != len(cases):
        fails.append(f"population: case list ({len(cases)}) and population table "
                     f"({sum(pops.values())}) disagree")
    for fault, fx_name, reason in skipped:
        extra = ""
        if fault == "scan_in_bf16":
            ok, detail = bf16_solve_supported()
            extra = f" [probed, not assumed: bf16 solve_triangular supported={ok}; {detail}]"
        print(f"#   skipped (unreachable, printed not dropped): {fault} on {fx_name}: "
              f"{reason}{extra}")
    print(f"# skipped cases: {len(skipped)} of {len(cases) + len(skipped)} built")

    # §The arms that RAN, and the ones this geometry cannot adjudicate — printed from the
    # tables, so an absence is a derived claim with a reason rather than a dropped test.
    ok, detail = bf16_solve_supported()
    print(f"# scan_in_bf16: RAN on {built.get('scan_in_bf16', 0)} case(s) per block. Every "
          "tensor of the scan (q_hat/k_hat, the decay cumsum, pairwise, the state, the outputs) "
          "is bfloat16; the UT solve is F32 and cast back, which is the reference's OWN width "
          "(§04.4's scan-dtype row, §04b.4 — torch_chunk_gated_delta_rule casts q/k/v/beta/decay "
          "to float32 at the scan entry and solves in float32).")
    print(f"#   bf16 solve_triangular on this backend: supported={ok}; probe says: {detail} — "
          "the bf16-SOLVE numerics are therefore NOT ESTABLISHED, printed below rather than "
          "folded into the arm as if it had run")
    print("# NOT ESTABLISHED — the arms and claims this lane may not be read as checked:")
    for entry in NOT_ESTABLISHED:
        for key in ("item", "measured", "what_ran_instead", "would_close"):
            if not str(entry.get(key, "")).strip():
                fails.append(f"NOT_ESTABLISHED entry {entry.get('item')!r} has an empty "
                             f"{key!r}; an entry with no reason is a dropped test wearing an "
                             "entry")
        print(f"#   - {entry['item']}")
        print(f"#       measured: {entry['measured']}")
        print(f"#       instead : {entry['what_ran_instead']}")
        print(f"#       closes    : {entry['would_close']}")
    # The list is checked against the geometry: every degeneracy and every section whose
    # derived population is 0 must appear in it, or an arm has vanished quietly.
    listed = {e["item"].split(":")[0].strip() for e in NOT_ESTABLISHED}
    for fault in sorted(g.degeneracies()):
        if fault not in listed:
            fails.append(f"NOT_ESTABLISHED does not carry the geometry degeneracy {fault} "
                         f"({g.degeneracies()[fault]}) — a degenerate arm must be listed, not "
                         "silently excluded")
    for fault, n in pops.items():
        if n == 0 and fault not in listed:
            fails.append(f"section {fault} has derived population 0 and no NOT_ESTABLISHED "
                         "entry — a skipped arm is not an arm")
    # The probe and the recorded claim must agree about whether the solve exists.
    probe_name = detail.split(":")[0]
    for entry in NOT_ESTABLISHED:
        if "solve_triangular" in entry["item"] or "UT solve" in entry["item"]:
            if ok is False and probe_name not in entry["measured"]:
                fails.append(f"NOT_ESTABLISHED entry {entry['item']!r} records a different "
                             f"probe result than the live one ({detail})")

    if fails:
        return fails

    print("# derived (case x allowed-stage) population, unconditional half — printed before "
          "any comparison runs:")
    per_block = sum(len(AFFECTED[f]) for f, _p, _fx in cases)
    print(f"#   {per_block} per block x {len(blocks)} blocks = {per_block * len(blocks)}; "
          "the conditional half is licensed by fixture geometry (CONDITIONAL_AFFECTED) and "
          "printed with its denominator after the arms")

    arms, arms_facts = {}, {}
    cache = {}
    for arm in ("honest", ARM_BLIND, "crossed"):
        t, rows, f, facts = run_matrix(g, blocks, cases, args, arm, cache)
        arms[arm], arms_facts[arm] = t, facts
        print_table(arm, rows)
        print_tally(arm, t)
        fails += f

    base, conditional, detail = derive_expected_allowed_stages(cases, g, width)
    base *= len(blocks)
    conditional *= len(blocks)
    expected_allowed = base + conditional
    print("\n## derived (case x allowed-stage) population — the arm=stub expectation")
    print(f"  unconditional   {base:>4}   |AFFECTED[fault]| x derived cases x {len(blocks)} "
          "blocks")
    for line in detail[:40]:
        print(f"  conditional     {line}")
    print(f"  conditional total {conditional:>4}")
    print(f"  derived total     {expected_allowed:>4}   the blinded differ must miss exactly "
          "this many allowed stages, no more and no fewer")

    expected_cases = len(cases) * len(blocks)
    wit = expected_witnesses(cases, g, len(blocks))
    verdict_honest(arms["honest"], expected_cases, wit, fails, "arm=honest")
    verdict_stub(arms[ARM_BLIND], expected_cases, expected_allowed, fails, f"arm={ARM_BLIND}")
    verdict_crossed(arms["crossed"], expected_cases, fails, "arm=crossed")
    verdict_facts(arms_facts, cases, blocks, g, fails)

    print("\n## what each witness is evidence for")
    print("  arm=honest   scan_moved_cases / pre_scan_moved_cases == counts derived from the "
          "case list — the two sides that must be able to fire")
    print("  arm=honest   must_not_catch_violations==0 and localised==cases — localisation")
    print(f"  arm={ARM_BLIND}     caught==0, stage_diffs==0, must_catch_missed==derived>0 "
          "— the gate reads the differ, not a constant")
    print("  facts        `state` untouched and the deferred conv-state fire are asserted "
          "outright from torch.equal, and agreed across all three arms")
    print("  arm=crossed  crossed_alarms_fired==cases — the must-not-catch side can fail, so "
          "its 0 in arm=honest is earned")
    return fails


# ================================================================= the HF comparison
# §The `--compare-hf` section, which `--selftest` calls, and what it takes to PASS.
#
# PASS requires: (a) every one of the 19 stages, in both dispatch shapes, on both pinned
# blocks, agrees with the installed reference BIT FOR BIT; (b) the per-step rows for `y`,
# `state` and `conv_window` agree at every cached decode step, not just the last one
# (§10.4's "and every subsequent step"); (c) the seam rows agree, which is what licenses the
# `derived-from-captured` and `reference-side-call` provenances; (d) every positive control in
# `HF_CONTROLS` is REPORTED as a difference by this very comparison, on its contracted stages
# and on no others; (e) every count equals its own derived denominator; and (f) the
# UNVERIFIED list is non-empty and complete.
#
# What it does NOT claim: see the section docstring above and `HF_UNVERIFIED` — agreement here
# is a transcription check against the reference's own maths, printed with the provenance of
# every HF-side tensor so no reader has to take "we compared against HF" as one claim.

# The positive controls, by pass shape. `read_pre_update_state` and `decay_after_write` are
# the two §04.6 orderings the section is specifically required to be able to see (the step-5
# read position, and decay-before-write); the rest are there because a control that only ever
# runs on one shape proves the harness, not the maths.
HF_CONTROLS = {
    "prefill": ["decay_sign_flip", "beta_no_sigmoid", "chunk_decay_off_by_one",
                "pairwise_mask_after_exp"],
    "decode": ["read_pre_update_state", "decay_after_write", "decay_sign_flip"],
}

# The seam checks, named, so a seam that stopped being computed shows up as a denominator
# failure instead of quietly shrinking the expectation to match what ran (`pitfalls/40` §a
# check that produces no output cannot fail).
HF_SEAM_CHECKS = ["scan_out_cast_reproduces_core_attn", "scan_out_reproduces_state",
                  "q_pre_expanded_is_the_scan_argument",
                  "k_pre_expanded_is_the_scan_argument", "v_pre_is_the_scan_argument",
                  "norm_reads_the_scan_output", "norm_gate_is_the_captured_z"]

# §NOT ESTABLISHED — what this lane may not be read as having checked. Each entry carries the
# measured reason it is not established. `section_selftest` requires that every geometry
# degeneracy and every zero-population fault appears here, so an arm cannot vanish quietly
# (`pitfalls/40` §a skipped case is counted, and a skip whose reason was never observed is a
# guess); entries that are not forced by that rule are here because someone measured something.
NOT_ESTABLISHED = [
    {"item": "scan_in_bf16: the UT solve evaluated in bfloat16",
     "measured": 'NotImplementedError: "triangular_solve_cpu" not implemented for '
                 "'BFloat16' — measured by bf16_solve_supported(), re-printed with the case "
                 "list; the selftest requires this entry to name the same exception the live "
                 "probe returns",
     "what_ran_instead": "the bf16 scan in full — q_hat/k_hat, the decay cumsum, pairwise, the "
                         "state and the scan outputs are all bfloat16 — with the UT solve done "
                         "in F32 and cast back, which is the reference's own width (§04.4's "
                         "scan-dtype row, §04b.4)",
     "measured_cost": "printed per stage, per mode, per block by the bf16-arm table in "
                      "--compare-hf, and deliberately not copied here where it would rot: the "
                      "four scan tensors the arm stores bf16 where the reference keeps f32 "
                      "(q_hat, k_hat, scan_out, state) are REFUSED by the bit comparator and "
                      "then measured again after an explicit widening; core_attn, normed and y "
                      "differ numerically. What IS adjudicated is containment — every moved "
                      "stage must sit inside AFFECTED['scan_in_bf16'] and at least one must "
                      "move — so the arm is neither inert nor leaking",
     "would_close": "a backend with a bf16 solve_triangular, or the §04b.4 export branch "
                    "(forward substitution) implemented at bf16 and gated"},
    {"item": "q_scaled_by_dv",
     "measured": "d_k == d_v == 128 in the committed config, so 1/sqrt(d_v) IS 1/sqrt(d_k): the "
                 "injection and the reference are the same expression, and no fixture can "
                 "distinguish them",
     "what_ran_instead": "nothing — excluded from the must-catch set as DEGENERATE and printed "
                         "as a derived absence in the population table, not as a passing case",
     "would_close": "a checkpoint with d_k != d_v; not this one"},
]


def _fmt_maxabs(v: float) -> str:
    return "-" if v == 0.0 else f"{v:.3e}"


def fixture_input(g: GdnGeometry, fx: Fixture) -> tuple:
    """The reference/oracle input pair for one fixture: (prefill [1,T,d], steps [1,n,d]).

    One generator draw over `n_tokens + n_steps` rows, split, so the decode steps continue the
    same sequence the prefill covered — the same `gdn_input` call, the same seed, no second
    fixture the two sides could disagree about."""
    n = fx.n_tokens + (fx.n_steps if fx.kind == "decode" else 0)
    x = gdn_input(n, g, fx.seed).unsqueeze(0)
    if fx.kind == "prefill":
        return x, x[:, :0]
    return x[:, :fx.n_tokens], x[:, fx.n_tokens:]


def _compare_rows(tag, stages, ours, theirs, fails, counts, print_rows=True,
                  adjudicate=True, row_key="rows", ident_key="identical",
                  diff_key="differing"):
    """Print and adjudicate one (block, mode) table of per-stage bit comparisons.

    `adjudicate=False` is the sabotage run and the bf16-arm table: differences are printed and
    counted but not turned into failures, because producing them is the point of those runs.
    The three counter keys are named by the caller so an extra table cannot quietly inflate the
    main table's denominator (pitfalls/40 §an unnamed denominator)."""
    if print_rows:
        print(f"  {'stage':<12} {'provenance':<22} {'dtype':<6} {'shape':<20} "
              f"{'bit_identical':>13} {'ndiff':>16} maxabs")
    differing = []
    for stage in stages:
        kind, _how = HF_PROVENANCE[stage]
        a, b = ours[stage], theirs[stage]
        counts[row_key] = counts.get(row_key, 0) + 1
        try:
            d = _bit_diff(a, b)
        except AssertionError as exc:
            if adjudicate:
                fails.append(f"{tag}/{stage}: {exc}")
            if print_rows:
                print(f"  {stage:<12} {kind:<22} {'REFUSED':<6} {str(tuple(a.shape)):<20} "
                      f"{'NO':>13} {'refused':>16} {str(exc)[:60]}")
            differing.append(stage)
            counts[diff_key] = counts.get(diff_key, 0) + 1
            continue
        if d["bit_identical"]:
            counts[ident_key] = counts.get(ident_key, 0) + 1
        else:
            differing.append(stage)
            counts[diff_key] = counts.get(diff_key, 0) + 1
            if adjudicate:
                fails.append(f"{tag}/{stage}: NOT bit-identical against the reference — "
                             f"ndiff={d['ndiff']}/{d['n']} maxabs={d['max_abs']:.3e} "
                             f"({d['dtype']}, provenance={kind})")
        if print_rows:
            print(f"  {stage:<12} {kind:<22} {d['dtype'].replace('torch.', ''):<6} "
                  f"{str(tuple(d['shape'])):<20} "
                  f"{'YES' if d['bit_identical'] else 'NO':>13} "
                  f"{str(d['ndiff']) + '/' + str(d['n']):>16} {_fmt_maxabs(d['max_abs'])}")
    return differing


def section_compare_hf(g: GdnGeometry, args, blocks=None, sabotage: str = "") -> list:
    fails = []
    blocks = blocks if blocks is not None else [
        Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    if not blocks:
        raise AssertionError("no blocks selected; an empty comparison cannot pass")
    if set(HF_PROVENANCE) != set(STAGES):
        fails.append(f"HF_PROVENANCE keys != STAGES: missing "
                     f"{sorted(set(STAGES) - set(HF_PROVENANCE))}, extra "
                     f"{sorted(set(HF_PROVENANCE) - set(STAGES))} — an unclaimed stage is not a "
                     "checked one")
        return fails
    if not HF_UNVERIFIED:
        fails.append("HF_UNVERIFIED is empty: a lane that has established everything has "
                     "stopped looking, so this section refuses to print PASS")
    for e in HF_UNVERIFIED:
        for key in ("item", "why", "closes_when"):
            if not str(e.get(key, "")).strip():
                fails.append(f"UNVERIFIED entry {e.get('item')!r} has an empty {key!r}")
    if sabotage:
        if sabotage not in FAULT_FIELDS:
            fails.append(f"--sabotage {sabotage!r} is not a Faults field; the controls are "
                         f"{sorted(FAULT_FIELDS)}")
            return fails

    fixtures = default_fixtures(g)
    pf_cands = [f for f in fixtures if f.kind == "prefill" and f.n_chunks(g, 0) == 2
                and f.n_pad(g, 0) > 0 and not f.pad_noise]
    dec_cands = [f for f in fixtures if f.kind == "decode"]
    if len(pf_cands) != 1 or len(dec_cands) != 1:
        fails.append(f"the fixture set does not determine exactly one 2-chunk padded prefill "
                     f"and one decode fixture (got {len(pf_cands)}, {len(dec_cands)}); the "
                     "comparison's shape must be derived, not typed")
        return fails
    fx_pf, fx_dec = pf_cands[0], dec_cands[0]

    print(f"# command: {' '.join(sys.argv)}")
    print(f"# compare-hf — torch {torch.__version__} CPU threads={getattr(args, 'threads_used', '?')} "
          f"blocks={','.join(b.name for b in blocks)}")
    print(f"# geometry: h_k={g.h_k} h_v={g.h_v} d_k={g.d_k} d_v={g.d_v} k_conv={g.k_conv} "
          f"C={g.chunk_width} scan_dtype={g.mamba_ssm_dtype}")
    print(f"# shapes: prefill {fx_pf.name} T={fx_pf.n_tokens} "
          f"(n_chunks={fx_pf.n_chunks(g, 0)}, n_pad={fx_pf.n_pad(g, 0)}); decode {fx_dec.name} "
          f"prefill {fx_dec.n_tokens} then {fx_dec.n_steps} cached steps "
          f"(n_steps = k_conv-1, geometric)")
    print("# the installed reference is the witness. It is a TRANSCRIPTION check, not an "
          "independence proof (pitfalls/30 §a CPU mirror cannot see a shared formula); the "
          "independence evidence is the cross-form agreement and the three-arm matrix.")
    if sabotage:
        print(f"# [SABOTAGE — NOT A VERDICT] the ORACLE side is injected with Faults({sabotage}"
              f"=True); the reference side is untouched. A PASS here is impossible by "
              f"construction and a run that differs nowhere is a FAILURE of this section.")

    # §The control plan, built BEFORE anything runs. Each control gets a fixture its own
    # REQUIRES admits AND whose contract licenses at least one allowed stage: a control run on a
    # fixture where the fault is provably inert is not a control. Measured, not assumed — on a
    # 2-chunk PADDED prefill `chunk_decay_off_by_one` moves nothing at all (the final chunk's
    # `cum[-1]` equals `cum[-2]` because a pad row contributes g = 0, and with two chunks
    # `chunk_decay[0]` multiplies the zero initial state; see CONDITIONAL_AFFECTED and
    # tmp/w20_chunkdecay.py), so it has to run on a three-chunk fixture to prove anything.
    plan = []
    for mode in sorted(HF_CONTROLS):
        for fault in HF_CONTROLS[mode]:
            # The mode fixes the pass shape (prefill = the chunked scan, decode = the cached
            # recurrent one), so the fixture must be of that kind; among those, the reachable
            # one that licenses the most stages, then the deepest chunk count, then the name —
            # deterministic, and never a fixture on which the fault is provably inert.
            cands = [fx for fx in fixtures if fx.kind == mode
                     and APPLIES[fault] in ("both", fx.kind)
                     and REQUIRES[fault][0](g, fx, 0)
                     and bool(allowed_stages(fault, g, fx, 0)[0])]
            cands.sort(key=lambda fx: (-len(allowed_stages(fault, g, fx, 0)[0]),
                                       -fx.n_chunks(g, 0), fx.name))
            if not cands:
                fails.append(f"control {fault} ({mode}) has no fixture that reaches any allowed "
                             "stage — the control is vacuous by geometry and must move to "
                             "NOT_ESTABLISHED, not be run and reported as a pass")
                continue
            plan.append((mode, fault, cands[0]))

    counts, control_rows = {}, []
    for block in blocks:
        w = load_gdn_block_weights(block, g)
        hf_cache: dict = {}

        def hf_for(fx, _w=w):
            key = (fx.name, fx.kind, fx.n_tokens, fx.n_steps, fx.seed, fx.pad_noise)
            if key not in hf_cache:
                xp, xs = fixture_input(g, fx)
                hf_cache[key] = hf_reference_stages(block, _w, g, xp, xs)
                counts["captures"] = counts.get("captures", 0) + 1
            return hf_cache[key]

        x_pf, _none = fixture_input(g, fx_pf)
        x_dec_pf, x_steps = fixture_input(g, fx_dec)
        if x_steps.shape[1] != fx_dec.n_steps:
            raise AssertionError("the decode steps are not the fixture's own row count")
        hf_pf, ran = hf_for(fx_pf)
        hf_dec, ran_dec = hf_for(fx_dec)
        print(f"\n## {block.name}: which reference code ran — transformers {ran['transformers']}, "
              f"module {ran['scan_impl']['module']}")
        print(f"   scans: chunked={ran['scan_impl']['chunked']} recurrent="
              f"{ran['scan_impl']['recurrent']} | conv: fn={ran['conv_impl']['fn']} "
              f"update={ran['conv_impl']['update']}")
        print(f"   cache layer={ran['cache_layer']} record_past={ran['record_past']} "
              f"(§04.3's cached conv update path) | dispatch seen: {ran_dec['scans_seen']}")

        faults_pf = faults_dec = NO_FAULTS
        if sabotage:
            faults_pf = (NO_FAULTS if APPLIES[sabotage] == "decode"
                         else Faults(**{sabotage: True}))
            faults_dec = (NO_FAULTS if APPLIES[sabotage] == "prefill"
                          else Faults(**{sabotage: True}))
        orc_pf = gdn_block(x_pf, w, g, "chunked", faults_pf, mode="prefill")
        seq = gdn_sequence(x_dec_pf, x_steps, w, g, "chunked", "recurrent",
                           faults=NO_FAULTS, decode_faults=faults_dec)
        steps = seq["all_steps"]
        if len(steps) != fx_dec.n_steps or len(hf_dec["decode"]) != fx_dec.n_steps:
            raise AssertionError("step count disagreement between the oracle, the reference and "
                                 "the fixture — nothing below is comparable")

        _compare_rows(f"{block.name}/prefill", STAGES, orc_pf, hf_pf["prefill"], fails, counts,
                      adjudicate=not sabotage)
        print(f"  (decode rows are the FINAL cached step; every step is in the per-step table "
              f"below)")
        _compare_rows(f"{block.name}/decode", STAGES, steps[-1], hf_dec["decode"][-1], fails,
                      counts, adjudicate=not sabotage)

        # Every step, not just the last: §10.4's conv-state claim is a claim about all of them.
        print(f"\n### {block.name} per-step rows (cached decode, {fx_dec.n_steps} steps)")
        print(f"  {'step':<5} {'stage':<12} {'provenance':<22} {'bit_identical':>13} "
              f"{'ndiff':>16} maxabs")
        for i, (ours, theirs) in enumerate(zip(steps, hf_dec["decode"])):
            for stage in ("y", "state", "conv_window"):
                kind, _how = HF_PROVENANCE[stage]
                counts["per_step_rows"] = counts.get("per_step_rows", 0) + 1
                try:
                    d = _bit_diff(ours[stage], theirs[stage])
                except AssertionError as exc:
                    fails.append(f"{block.name}/decode{i}/{stage}: {exc}")
                    print(f"  {i:<5} {stage:<12} {kind:<22} {'NO':>13} "
                          f"{'refused':>16} {str(exc)[:60]}")
                    continue
                if d["bit_identical"]:
                    counts["identical"] = counts.get("identical", 0) + 1
                else:
                    counts["differing"] = counts.get("differing", 0) + 1
                    if not sabotage:
                        fails.append(f"{block.name}/decode{i}/{stage}: NOT bit-identical "
                                     f"at step {i} — ndiff={d['ndiff']}/{d['n']} "
                                     f"maxabs={d['max_abs']:.3e}")
                print(f"  {i:<5} {stage:<12} {kind:<22} "
                      f"{'YES' if d['bit_identical'] else 'NO':>13} "
                      f"{str(d['ndiff']) + '/' + str(d['n']):>16} {_fmt_maxabs(d['max_abs'])}")

        # The seam rows: the boundaries between captured tensors. They are what make the
        # `derived-from-captured` / `reference-side-call` rows mean what they say.
        print(f"\n### {block.name} seam rows — these license the non-`captured` provenances")
        seam_groups = [("prefill", hf_pf["prefill_seam"])] + [
            (f"decode{i}", s) for i, s in enumerate(hf_dec["decode_seam"])]
        for tag, seam in seam_groups:
            for name, d in sorted(seam.items()):
                counts["seam_rows"] = counts.get("seam_rows", 0) + 1
                if d["bit_identical"]:
                    counts["identical"] = counts.get("identical", 0) + 1
                else:
                    counts["differing"] = counts.get("differing", 0) + 1
                    fails.append(f"{block.name}/{tag} seam {name}: NOT bit-identical — "
                                 f"ndiff={d['ndiff']}/{d['n']} maxabs={d['max_abs']:.3e}; the "
                                 "capture wiring, not the maths, is what this row checks")
                print(f"  {tag:<10} {name:<40} "
                      f"{'YES' if d['bit_identical'] else 'NO':>3} "
                      f"{str(d['ndiff']) + '/' + str(d['n']):>16} {_fmt_maxabs(d['max_abs'])}")

        if sabotage:
            continue
        # §The `scan_in_bf16` arm, against the reference — an OBSERVATION with one adjudication.
        # The arm stores the scan in bf16, which is not the committed config (§04.4's scan-dtype
        # row and §04b.4 put the scan in `mamba_ssm_dtype`, float32; the reference widens
        # q/k/v/beta/decay at the scan entry). So it is EXPECTED to differ from the reference and
        # bit-identity is the wrong demand — this table does not adjudicate agreement. What it
        # does adjudicate is containment: the arm may only differ on the stages its own contract
        # licenses (`allowed_stages`, the same table the injection matrix reads), and it must
        # differ on at least one of them, or the arm is inert and its green matrix means nothing.
        # The measured size of the disagreement is printed stage by stage, including the four
        # stages the arm stores bf16 where the reference stores f32 — those numbers, not an
        # adjective, are what the NOT_ESTABLISHED entry for this arm carries.
        runs, why = bf16_solve_supported()
        print(f"\n### {block.name} the scan_in_bf16 arm — bf16 scan storage, F32 UT solve. "
              f"Adjudicated: containment inside the contract. NOT adjudicated: agreement.")
        print(f"   a bf16 `solve_triangular` on this backend: "
              f"{'runs' if runs else 'UNAVAILABLE (' + why + ')'}; the arm therefore solves in "
              f"F32 and casts back (§04.4, §04b.4 — the reference's own width). The width of the "
              f"scan STORAGE is the arm's real effect and is measured below.")
        for mode, fx, theirs in (("prefill", fx_pf, hf_pf["prefill"]),
                                 ("decode", fx_dec, hf_dec["decode"][-1])):
            xp, xs = fixture_input(g, fx)
            if mode == "prefill":
                ours_arm = gdn_block(xp, w, g, "chunked", Faults(scan_in_bf16=True),
                                     mode="prefill", pad_noise=fx.pad_noise)
            else:
                ours_arm = gdn_sequence(xp, xs, w, g, "chunked", "recurrent",
                                        faults=NO_FAULTS,
                                        decode_faults=Faults(scan_in_bf16=True))["all_steps"][-1]
            print(f"  {mode} on {fx.name}")
            diff = _compare_rows(f"{block.name}/{mode}/scan_in_bf16", STAGES, ours_arm, theirs,
                                 fails, counts, adjudicate=False, row_key="arm_rows",
                                 ident_key="arm_identical", diff_key="arm_differing")
            allowed, _lic = allowed_stages("scan_in_bf16", g, fx, 0)
            caught = sorted(set(diff) & allowed)
            leak = sorted(set(diff) - allowed)
            counts["arm_checks"] = counts.get("arm_checks", 0) + 1
            if not caught:
                counts["arm_inert"] = counts.get("arm_inert", 0) + 1
                fails.append(f"{block.name}/{mode}/scan_in_bf16: the arm moved NO stage against "
                             "the reference. An arm that cannot differ is inert and its matrix "
                             "rows are decoration, not evidence")
            if leak:
                counts["arm_leaks"] = counts.get("arm_leaks", 0) + 1
                fails.append(f"{block.name}/{mode}/scan_in_bf16: the arm moved stages outside "
                             f"its contract: {leak} (allowed={sorted(allowed)}) — bf16 scan "
                             "storage reached a stage the §04.4 scan-dtype row does not cover")
            # The width cost, measured rather than asserted: a stage the arm stores bf16 where
            # the reference stores f32 is REFUSED by the bit comparator, which is correct but
            # says nothing about magnitude. Widen explicitly, once, and label it — this is a
            # measurement of the storage error, not a bit-identity claim (pitfalls/30 §a bf16/f32
            # mix is refused by design; `.to` casts, it does not reinterpret).
            for stage in STAGES:
                a, b = ours_arm[stage], theirs[stage]
                if a.shape == b.shape and a.dtype != b.dtype and a.is_floating_point() \
                        and b.is_floating_point():
                    d = _bit_diff(a.to(b.dtype), b)
                    counts["arm_widened"] = counts.get("arm_widened", 0) + 1
                    print(f"    widened {stage}: {str(a.dtype).replace('torch.', '')} storage "
                          f"vs {str(b.dtype).replace('torch.', '')} reference, widened for the "
                          f"measurement only — ndiff {d['ndiff']}/{d['n']}, maxabs "
                          f"{_fmt_maxabs(d['max_abs'])} (NOT a bit-identity claim)")
            print(f"    containment: moved {len(caught)}/{len(STAGES)} stages, all inside the "
                  f"contract: {bool(caught) and not leak}; outside: {','.join(leak) or '-'}")

        # §The positive controls. Each must be REPORTED as a difference by the comparison above,
        # confined to the stages its contract licenses — the same `allowed_stages` table the
        # injection matrix reads, so one contract has two readers (pitfalls/40).
        print(f"\n### {block.name} positive controls — the comparison must be able to differ")
        print(f"  {'mode':<8} {'control':<24} {'fixture':<22} {'caught (allowed)':<44} "
              f"{'LEAK':<20} maxabs")
        for mode, fault, fx in plan:
            f = Faults(**{fault: True})
            xp, xs = fixture_input(g, fx)
            hf_fx = hf_for(fx)[0]
            if mode == "prefill":
                ours = gdn_block(xp, w, g, "chunked", f, mode="prefill")
                theirs = hf_fx["prefill"]
            else:
                ours = gdn_sequence(xp, xs, w, g, "chunked", "recurrent",
                                    faults=NO_FAULTS, decode_faults=f)["all_steps"][-1]
                theirs = hf_fx["decode"][-1]
            allowed, _lic = allowed_stages(fault, g, fx, 0)
            diff, worst, nonfinite = [], 0.0, False
            for stage in STAGES:
                try:
                    d = _bit_diff(ours[stage], theirs[stage])
                except AssertionError:
                    diff.append(stage)
                    nonfinite = True
                    continue
                if not d["bit_identical"]:
                    diff.append(stage)
                    v = d["max_abs"]
                    if v != v or v == float("inf"):
                        nonfinite = True
                    else:
                        worst = max(worst, v)
            caught = sorted(set(diff) & allowed)
            leak = sorted(set(diff) - allowed)
            counts["controls"] = counts.get("controls", 0) + 1
            if not caught:
                counts["controls_blind"] = counts.get("controls_blind", 0) + 1
                fails.append(f"{block.name}/{mode}/{fault} on {fx.name}: the HF comparison "
                             "reported NO difference on an allowed stage — this comparison "
                             "cannot fail, so its clean PASS is hoped in, not tested")
            if leak:
                counts["controls_leaks"] = counts.get("controls_leaks", 0) + 1
                fails.append(f"{block.name}/{mode}/{fault}: the comparison reported a difference "
                             f"outside the contract: {leak} (allowed={sorted(allowed)})")
            control_rows.append((block.name, mode, fault, fx.name, caught, leak))
            print(f"  {mode:<8} {fault:<24} {fx.name:<22} "
                  f"{(','.join(caught) or 'NONE'):<44} {(','.join(leak) or '-'):<20} "
                  f"{'nan/inf' if nonfinite else format(worst, '.2e')}")
        del w

    # Counts against their own denominators (pitfalls/40 §an unnamed denominator).
    exp_main = len(STAGES) * 2 * len(blocks)
    exp_steps = 3 * fx_dec.n_steps * len(blocks)
    exp_seam = len(HF_SEAM_CHECKS) * (1 + fx_dec.n_steps) * len(blocks)
    exp_arm = 0 if sabotage else len(STAGES) * 2 * len(blocks)
    exp_arm_checks = 0 if sabotage else 2 * len(blocks)
    exp_controls = len(plan) * len(blocks)
    if sabotage:
        # A sabotage run is the main tables only: the controls are what a clean run uses to
        # prove it can differ, and this run proves it by being injected instead.
        exp_controls, exp_captures = 0, 2 * len(blocks)
    else:
        exp_captures = len({(fx.name, fx.kind, fx.n_tokens, fx.n_steps, fx.seed, fx.pad_noise)
                            for _m, _f, fx in plan} | {
                            (fx_pf.name, fx_pf.kind, fx_pf.n_tokens, fx_pf.n_steps,
                             fx_pf.seed, fx_pf.pad_noise),
                            (fx_dec.name, fx_dec.kind, fx_dec.n_tokens, fx_dec.n_steps,
                             fx_dec.seed, fx_dec.pad_noise)}) * len(blocks)
    for key, want in (("rows", exp_main), ("per_step_rows", exp_steps),
                      ("seam_rows", exp_seam), ("controls", exp_controls),
                      ("arm_rows", exp_arm), ("arm_checks", exp_arm_checks),
                      ("captures", exp_captures)):
        got = counts.get(key, 0)
        print(f"# count {key:<14} {got:>5} derived {want:>5}")
        if got != want:
            fails.append(f"count {key}: printed {got}, derived {want} — a section whose rows do "
                         "not equal its own denominator compared nothing for the rest")
    n_rows = exp_main + exp_steps + exp_seam
    print(f"# rows compared {counts.get('rows', 0) + counts.get('per_step_rows', 0) + counts.get('seam_rows', 0)} "
          f"= {exp_main} stage rows + {exp_steps} per-step rows + {exp_seam} seam rows; "
          f"bit-identical {counts.get('identical', 0)}, differing {counts.get('differing', 0)}")
    if not sabotage:
        print(f"# bf16-arm rows {counts.get('arm_rows', 0)} = {len(STAGES)} stages x 2 modes x "
              f"{len(blocks)} blocks, reported separately and NOT adjudicated for agreement: "
              f"bit-identical {counts.get('arm_identical', 0)}, differing "
              f"{counts.get('arm_differing', 0)} (of which {counts.get('arm_widened', 0)} were "
              f"REFUSED on storage width and measured after an explicit widening), "
              f"{counts.get('arm_inert', 0)} inert (must be 0), {counts.get('arm_leaks', 0)} "
              f"outside contract (must be 0)")
    print(f"# controls {counts.get('controls', 0)} run, {counts.get('controls_blind', 0)} blind "
          f"(must be 0), {counts.get('controls_leaks', 0)} leaking (must be 0)")
    if not sabotage and counts.get("differing", 0) != 0:
        fails.append(f"{counts['differing']} row(s) differ against the reference on a clean run")
    if sabotage:
        if counts.get("differing", 0) == 0:
            fails.append(f"SABOTAGE Faults({sabotage}=True) on the oracle moved NO row anywhere: "
                         "this comparison cannot fail, so its clean PASS was hoped in, not "
                         "tested")
        else:
            print(f"# sabotage witness: Faults({sabotage}=True) moved "
                  f"{counts['differing']}/{n_rows} row(s) against the reference — the comparison "
                  "can differ, which is what makes the clean PASS a claim")

    if sabotage:
        print(f"\n## NOT ESTABLISHED by a sabotage run — this run is evidence that the "
              f"comparison can differ, it is not an agreement verdict")
    print("\n## UNVERIFIED — explicitly, because silence would read as agreement")
    for e in HF_UNVERIFIED:
        print(f"  NOT ESTABLISHED: {e['item']}")
        print(f"      why: {e['why']}")
        print(f"      closes when: {e['closes_when']}")
    if not HF_UNVERIFIED:
        fails.append("the UNVERIFIED list printed nothing")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="the geometry-derived injection matrix, three arms, plus the HF "
                         "comparison section")
    ap.add_argument("--compare-hf", action="store_true",
                    help="the installed reference compared stage by stage, with its own "
                         "positive controls (--selftest runs it too)")
    ap.add_argument("--sabotage", default="", metavar="FAULT",
                    help="evidence only: run the HF comparison with the ORACLE side injected; "
                         "a run that differs nowhere is a FAILURE, and it never prints PASS")
    ap.add_argument("--blocks", default="layer0,layer1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only-fault", default="", help="debug only; prints a SUBSET banner")
    ap.add_argument("--only-fixture", default="", help="debug only; prints a SUBSET banner")
    args = ap.parse_args()
    if not (args.selftest or args.compare_hf or args.sabotage):
        print("SELFTEST FAIL: nothing selected (--selftest / --compare-hf / --sabotage)")
        return 1
    args.threads_used = R.set_threads(args.threads)
    g = load_gdn_geometry()
    fails = []
    if args.selftest:
        fails += section_selftest(g, args)
    if args.selftest or args.compare_hf or args.sabotage:
        fails += section_compare_hf(g, args, sabotage=args.sabotage)
    if fails:
        tag = "SABOTAGE RUN FAIL" if args.sabotage else "SELFTEST FAIL"
        print(f"\n{tag}")
        for f in fails:
            print("  - " + f)
        return 1
    if args.sabotage:
        print(f"\nSABOTAGE RUN OK — Faults({args.sabotage}=True) was reported as a difference; "
              "this is not an agreement verdict and no timing was taken")
        return 0
    print("\nSELFTEST PASS — three arms over the derived matrix, and every HF row bit-identical "
          "with its positive controls firing; no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
