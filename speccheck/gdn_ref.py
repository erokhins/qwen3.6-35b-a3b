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
Implemented and running: the oracle, the comparators, the band derivation, and the
three-arm injection matrix (`--selftest`).
**Not implemented yet — no flag exists for them, so they cannot print a PASS:** the HF
cross-check (`--cross-check-hf`), the floor probe and its evidence file (`--floor-probe`),
the cross-form rows of §04b.6 (`--cross-form`), the capability/dispatch probe
(`--capability`) and `--against-dumps`. `aggregates`, `band_from_probe` and `load_band` are
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
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False,
                                                   unitriangular=unitri)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, dec_k_beta, upper=False,
                                                   unitriangular=unitri)
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
    return {"prefill": pf, "decode": steps[-1], "decode_steps": len(steps),
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
    `scan_in_bf16` is a policy violation whose whole point is the dtype."""
    if a.shape != b.shape:
        return True, float("inf")
    if a.dtype != b.dtype:
        return True, float("nan")
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
    "scan_in_bf16": _D_SCAN,
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
    committed config), so the honest arm never needs bf16 here — but the `scan_in_bf16`
    *injection* does, and a case that cannot run must be printed with the error that proves
    it rather than dropped (`pitfalls/40` §a skipped case is counted, and a skip whose reason
    was never observed is a guess)."""
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
    "scan_in_bf16": (lambda g, f, C: bf16_solve_supported()[0],
                     "the CPU backend needs a bf16 triangular solve for the chunked form; "
                     "probed result is printed with the case list"),
    "conv_tap_rotate": (lambda g, f, C: g.k_conv >= 2,
                        "k_conv >= 2: a rotation needs at least two taps"),
    "conv_tap_reverse": (lambda g, f, C: g.k_conv >= 3,
                         "k_conv >= 3: for k_conv == 2 a reversal is a rotation"),
    "head_pairing_tiled": (lambda g, f, C: g.expansion > 1,
                          "h_v > h_k: grouped vs tiled pairing needs an expansion"),
    "v_head_order_tiled": (lambda g, f, C: g.expansion > 1,
                           "h_v > h_k: see head_pairing_tiled"),
    "q_scaled_by_dv": (lambda g, f, C: g.d_k != g.d_v,
                       "d_k != d_v: otherwise 1/sqrt(d_v) IS 1/sqrt(d_k)"),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="the geometry-derived injection matrix, three arms")
    ap.add_argument("--blocks", default="layer0,layer1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only-fault", default="", help="debug only; prints a SUBSET banner")
    ap.add_argument("--only-fixture", default="", help="debug only; prints a SUBSET banner")
    args = ap.parse_args()
    if not args.selftest:
        print("SELFTEST FAIL: nothing selected (--selftest)")
        return 1
    args.threads_used = R.set_threads(args.threads)
    g = load_gdn_geometry()
    fails = section_selftest(g, args)
    if fails:
        print("\nSELFTEST FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nSELFTEST PASS — three arms over the derived matrix; no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
