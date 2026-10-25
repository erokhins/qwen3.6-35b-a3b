"""speccheck/moe_ref.py — the independent torch-CPU implementation of the
Qwen3.6-35B-A3B MoE block. This module is the oracle (§What this is of
speccheck/README.md).

Rules it obeys:
  * torch-CPU only. No `.cuda()`, no HIP, no device allocation, no timing call.
  * It must not import any HF model class. The math is written here from the
    geometry in `checkpoints/Qwen3.6-35B-A3B.config.json` plus the tensor names in
    `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` — both committed in this repo.
  * Every non-obvious behaviour is tagged `[SPEC]` (derived from the config/geometry),
    `[REF]` (a form choice the pinned reference takes, recorded here so the choice is
    visible rather than inherited) or `[ASSUMPTION A-nn]` (our reading, not yet
    adjudicated). The tag list lives in speccheck/results.md §Tag vocabulary.

§The block, in words is the authority for order-of-operations questions:

    router_logits = x @ Wg.T                       # [T, E]   model dtype (bf16)
    router_probs  = softmax_f32(router_logits)     # [T, E]   fp32 in, fp32 out
    vals, ids     = topk(router_probs, K)          # [T, K]   descending prob order
    vals         /= vals.sum(-1)                   # renormalise over the K, fp32
    weights       = vals.to(logits dtype)          # [REF] cast BEFORE the multiply
    expert(t,k)   = down_e @ (silu(gate_e(h)) * up_e(h)),  h = x[t], e = ids[t,k]
    routed[t]     = sum_k expert(t,k) * weights[t,k]       # [REF] bf16 accumulate
    shared[t]     = sigmoid(x @ wsg.T) * (down_s @ (silu(gate_s(x)) * up_s(x)))
    out[t]        = routed[t] + shared[t]          # [REF] added in the model dtype

The two reference-form choices that are *not* algebra — the cast position above, and the
accumulation dtype/order of the routed sum — are switchable in `Form` so both can be
measured rather than argued about.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "checkpoints" / "Qwen3.6-35B-A3B.config.json"
INDEX_PATH = REPO / "checkpoints" / "Qwen3.6-35B-A3B.safetensors.index.json"
SHARD_DIR = REPO / "models" / "qwen36-src"

# §Tensor names. The names are read out of the committed index, never typed into a gate;
# these are the suffix templates used to build them (one per weight family).
SUFFIXES = {
    "router": "mlp.gate.weight",
    "gate_up": "mlp.experts.gate_up_proj",
    "down": "mlp.experts.down_proj",
    "shared_gate": "mlp.shared_expert.gate_proj.weight",
    "shared_up": "mlp.shared_expert.up_proj.weight",
    "shared_down": "mlp.shared_expert.down_proj.weight",
    "shared_scalar_gate": "mlp.shared_expert_gate.weight",
}


# --------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """Every number the gates derive. Nothing here is typed by hand (§Tensor names)."""

    hidden_size: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_hidden_layers: int
    mtp_num_hidden_layers: int
    dtype: torch.dtype
    rms_norm_eps: float
    config_path: str = "checkpoints/Qwen3.6-35B-A3B.config.json"

    @property
    def n_blocks(self) -> int:
        """MoE blocks in the text path: every decoder layer is sparse (this config has
        no `intermediate_size` and no `mlp_only_layers`) plus the MTP draft block. [SPEC]"""
        return self.num_hidden_layers + self.mtp_num_hidden_layers

    @property
    def expert_bytes(self) -> int:
        """bf16 bytes of one expert: [2I,H] + [H,I]. [SPEC]"""
        return (self.two_i * self.hidden_size + self.hidden_size * self.moe_intermediate_size) * 2

    @property
    def two_i(self) -> int:
        return 2 * self.moe_intermediate_size


def load_geometry(config_path: Path = CONFIG_PATH) -> Geometry:
    cfg = json.loads(Path(config_path).read_text())
    t = cfg["text_config"]
    for key in ("intermediate_size", "mlp_only_layers"):
        if t.get(key) not in (None, []):
            raise AssertionError(f"text_config carries {key}={t[key]!r}: the all-MoE "
                                 "assumption in README.md §Target no longer holds")
    g = Geometry(
        hidden_size=t["hidden_size"],
        num_experts=t["num_experts"],
        top_k=t["num_experts_per_tok"],
        moe_intermediate_size=t["moe_intermediate_size"],
        shared_expert_intermediate_size=t["shared_expert_intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"],
        mtp_num_hidden_layers=t["mtp_num_hidden_layers"],
        dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[t["dtype"]],
        rms_norm_eps=t["rms_norm_eps"],
        config_path=str(config_path),
    )
    # The fused pair must be laid out along the OUTPUT dim, so the rank-3 expert tensor
    # is [E, 2I, H] and E != 2I. If they were equal, a transposed-stack read could not be
    # told apart from the correct read by shape and the selftest case would be vacuous.
    if g.two_i == g.num_experts:
        raise AssertionError("E == 2I: the transposed-expert-stack injection is "
                             "degenerate for this geometry")
    return g


# ------------------------------------------------------------------- safetensors access
def read_header(shard: Path) -> tuple[dict, int]:
    """Header-first read: 8-byte length + JSON header, no tensor bytes. [SPEC]"""
    with open(shard, "rb") as fh:
        nbytes = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(nbytes))
    return header, nbytes


def load_index(index_path: Path = INDEX_PATH) -> dict:
    return json.loads(Path(index_path).read_text())


def shard_of(tensor_name: str, index: dict | None = None) -> str:
    index = load_index() if index is None else index
    try:
        return index["weight_map"][tensor_name]
    except KeyError:
        raise AssertionError(f"{tensor_name} is not in the committed shard index") from None


def load_tensor(shard_path: Path, tensor_name: str, dtype=None) -> torch.Tensor:
    """One tensor at a time out of one shard. Never load a whole shard: the four acquired
    shards are 10.3 GiB and three other workers share this box (task rule 2026-10-26)."""
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt") as f:
        names = set(f.keys())
        if tensor_name not in names:
            raise AssertionError(f"{tensor_name} not in {shard_path.name} "
                                 f"({len(names)} tensors); check the shard index")
        t = f.get_tensor(tensor_name)
    return t.to(dtype) if dtype is not None and t.dtype != dtype else t


@dataclass(frozen=True)
class Block:
    kind: str          # "layer" | "mtp"
    index: int

    @property
    def name(self) -> str:
        return f"{self.kind}{self.index}"

    @property
    def prefix(self) -> str:
        if self.kind == "layer":
            return f"model.language_model.layers.{self.index}."
        if self.kind == "mtp":
            return f"mtp.layers.{self.index}."
        raise AssertionError(f"unknown block kind {self.kind!r}")

    @staticmethod
    def parse(spec: str) -> "Block":
        spec = spec.strip()
        if spec.startswith("mtp"):
            return Block("mtp", int(spec[3:]))
        if spec.startswith("layer"):
            return Block("layer", int(spec[5:]))
        raise AssertionError(f"block spec must be layer<N> or mtp<N>, got {spec!r}")


@dataclass
class BlockWeights:
    """The seven weight tensors of one MoE block, bf16, as stored."""

    block: Block
    tensors: dict
    source_shards: dict = field(default_factory=dict)

    def shapes_ok(self, g: Geometry, roles: tuple[str, ...] | None = None) -> None:
        want = {
            "router": (g.num_experts, g.hidden_size),
            "gate_up": (g.num_experts, g.two_i, g.hidden_size),
            "down": (g.num_experts, g.hidden_size, g.moe_intermediate_size),
            "shared_gate": (g.shared_expert_intermediate_size, g.hidden_size),
            "shared_up": (g.shared_expert_intermediate_size, g.hidden_size),
            "shared_down": (g.hidden_size, g.shared_expert_intermediate_size),
            "shared_scalar_gate": (1, g.hidden_size),
        }
        for role, shape in want.items():
            if roles is not None and role not in roles:
                continue                       # not requested by this caller
            if role not in self.tensors:
                raise AssertionError(f"block {self.block.name} is missing {role}")
            t = self.tensors[role]
            if tuple(t.shape) != tuple(shape):
                raise AssertionError(f"{self.block.name}.{role}: stored shape "
                                     f"{tuple(t.shape)} != config-derived {tuple(shape)}")
            if t.dtype is not torch.bfloat16:
                raise AssertionError(f"{self.block.name}.{role}: stored dtype {t.dtype}, "
                                     "expected bf16 (dtype-by-filesize, pitfalls/30)")


def load_block_weights(block: Block, g: Geometry,
                       shard_dir: Path = SHARD_DIR,
                       roles: tuple[str, ...] | None = None) -> BlockWeights:
    """Load (exactly) the weight tensors of one block, one at a time, and verify their
    shapes against the config-derived geometry before anything consumes them."""
    index = load_index()
    roles = tuple(SUFFIXES) if roles is None else tuple(roles)
    tensors, sources = {}, {}
    for role in roles:
        name = block.prefix + SUFFIXES[role]
        shard = shard_of(name, index)
        path = Path(shard_dir) / shard
        if not path.exists():
            raise AssertionError(f"{name} lives in {shard}, which is not in {shard_dir}; "
                                 "see README.md §Download — only 4 shards were acquired")
        tensors[role] = load_tensor(path, name)
        sources[role] = shard
    w = BlockWeights(block, tensors, sources)
    w.shapes_ok(g, roles)
    return w


# ------------------------------------------------------------------------ input fixture
def make_input(n_tokens: int, g: Geometry, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    """§Input generation. Deterministic given the seed; the method is part of the pin.

    fp32 `torch.randn` drawn from a private `torch.Generator('cpu')` (so the global RNG
    state cannot change the answer), each row rescaled to RMS exactly 1.0, then cast to
    the model dtype. [ASSUMPTION A-01] rows are RMS-1 because the block's input is
    `post_attention_layernorm(hidden)`, and this family's RMSNorm is
    `x * rsqrt(mean(x^2) + eps) * (1 + w)` — RMS 1 before the weight. Real activations
    modulate that per element by `(1 + w)`; that modulation is *not* in this fixture, so
    every near-tie / utilisation figure in speccheck/results.md is labelled a
    synthetic-input statistic and the follow-up that makes it a real one is named there.
    """
    if n_tokens <= 0:
        raise AssertionError("n_tokens must be positive")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_tokens, g.hidden_size, generator=gen, dtype=torch.float32)
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    if not torch.isfinite(rms).all():
        raise AssertionError("non-finite RMS in the fixture generator")
    x = x / rms
    return x.to(dtype)


# ------------------------------------------------------------------------- the router
@dataclass(frozen=True)
class Form:
    """The switchable form choices. Four of them are `[REF]` (what the pinned reference
    does); the rest exist so an injection is expressed as a form change, not as a fork."""

    softmax_dtype: str = "f32"       # "f32" | "bf16"          [REF] f32
    renorm: bool = True              #                          [REF] True
    weight_cast: str = "before"      # "before" | "after"       [REF] before
    accumulate: str = "bf16"         # "bf16" | "f32" | "f64"   [REF] bf16
    order: str = "expert_asc"        # "expert_asc" | "rank"    [REF] expert_asc
    batching: str = "grouped"        # "grouped" | "per_slot"   [REF] grouped
    router_dtype: str = "model"      # "model" | "f64"

    @staticmethod
    def named(arm: str) -> "Form":
        arms = {
            # the reference form we claim the pinned implementation takes
            "ref": Form(),
            # our own math form: independent of the reference's batching, accumulation
            # dtype and addend order; routing weights never rounded below f32
            "spec": Form(accumulate="f32", order="rank", batching="per_slot",
                         weight_cast="after"),
            # the same math with no bf16 rounding anywhere: the error reference
            "exact": Form(accumulate="f64", order="rank", batching="per_slot",
                          weight_cast="after", router_dtype="f64"),
            # single-switch arms, used for the prediction-4 quantification and the floor
            "cast_after": Form(weight_cast="after"),
            "softmax_bf16": Form(softmax_dtype="bf16"),
            "no_renorm": Form(renorm=False),
            "accum_f32": Form(accumulate="f32"),
            "accum_f64": Form(accumulate="f64"),
            "rank_order": Form(order="rank"),
            "per_slot": Form(batching="per_slot"),
        }
        if arm not in arms:
            raise AssertionError(f"unknown arm {arm!r}; known: {sorted(arms)}")
        return arms[arm]


def expected_dtypes(form: Form) -> dict:
    """The dtype each stage of a given `Form` is *defined* to carry — derived from the form
    switches, never typed out per arm. A gate that guesses a tensor's dtype from the file
    it read is the `pitfalls/30` failure (an F32 dump read as BF16 still prints a plausible
    ndiff), so every consumer asserts against this table before printing an aggregate."""
    cd = {"bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[form.accumulate]
    logits = torch.float64 if form.router_dtype == "f64" else torch.bfloat16
    probs = {"f32": torch.float32, "bf16": torch.bfloat16}[form.softmax_dtype]
    if form.router_dtype == "f64":
        probs = torch.float64
    weights = logits if form.weight_cast == "before" else (
        torch.float64 if logits is torch.float64 else torch.float32)
    return {"router_logits": logits, "router_probs": probs, "topk_weights": weights,
            "expert_raw": cd, "routed_sum": cd, "shared_raw": cd,
            "shared_gate_prob": cd, "shared_out": cd, "combined_out": cd}


def route(x: torch.Tensor, w_router: torch.Tensor, g: Geometry, form: Form) -> dict:
    """Router: logits -> softmax(dtype) -> topk(K) -> renormalise over the K -> cast.

    Returns logits, probs, ids, weights and `topk_probs_pre_renorm` (so the renorm
    injection is auditable) and the logits dtype the cast-before target is read from.
    """
    if x.ndim != 2 or x.shape[1] != g.hidden_size:
        raise AssertionError(f"route expects [T, {g.hidden_size}], got {tuple(x.shape)}")
    if tuple(w_router.shape) != (g.num_experts, g.hidden_size):
        raise AssertionError(f"router weight shape {tuple(w_router.shape)} != [E, H]="
                             f"{(g.num_experts, g.hidden_size)}")
    if form.router_dtype == "f64":
        logits = x.double() @ w_router.double().transpose(0, 1)
        sdtype = torch.float64
    elif form.router_dtype == "model":
        logits = F.linear(x, w_router)
        sdtype = torch.float32 if form.softmax_dtype == "f32" else torch.bfloat16
    else:
        raise AssertionError(form.router_dtype)

    probs = F.softmax(logits, dim=-1, dtype=sdtype)
    if probs.shape != (x.shape[0], g.num_experts):
        raise AssertionError(f"probs shape {tuple(probs.shape)} != [T, E]="
                             f"{(x.shape[0], g.num_experts)}")
    vals, ids = torch.topk(probs, g.top_k, dim=-1)      # descending; ties by lower index
    if ids.dtype is not torch.int64:
        raise AssertionError(f"topk returned {ids.dtype}, expected int64")
    pre = vals.clone()
    if form.renorm:
        vals = vals / vals.sum(dim=-1, keepdim=True)
    # [REF] cast-before: the renormalised weights are rounded to the LOGITS dtype here.
    # cast-after keeps them at f32 (f64 for the exact arm) so the product is formed wide.
    if form.weight_cast == "before":
        weights = vals.to(logits.dtype)
    elif form.weight_cast == "after":
        weights = vals.to(torch.float64 if logits.dtype is torch.float64 else torch.float32)
    else:
        raise AssertionError(form.weight_cast)
    return {
        "logits": logits,
        "probs": probs,
        "topk_ids": ids,
        "topk_weights": weights,
        "topk_probs_pre_renorm": pre,
        "logits_dtype": logits.dtype,
    }


# ------------------------------------------------------------------- the expert gather
def _expert_apply(rows: torch.Tensor, gu: torch.Tensor, dn: torch.Tensor,
                  g: Geometry, compute_dtype: torch.dtype, swap_halves: bool) -> torch.Tensor:
    """rows [n,H] -> [n,H] through one expert.

    `gate, up = chunk(2, dim=-1)` on the fused `[2I, H]` read: the fused pair is laid out
    along the OUTPUT dim, so chunking the GEMM output is the correct read and there is no
    stride/offset swap to invent. README.md prediction 3 says check, do not re-derive — the
    check is the `out.shape[-1] == 2I` assertion plus the `swapped_gate_up_half` case in
    `selftest_moe.py`, which is only caught if the halves really are output-dim blocks.
    `swap_halves` is an injection knob and must be False on any path that produces a pin.
    """
    gu = gu.to(compute_dtype)
    dn = dn.to(compute_dtype)
    rows = rows.to(compute_dtype)
    out = F.linear(rows, gu)
    if out.shape[-1] != g.two_i:
        raise AssertionError(f"fused read produced {out.shape[-1]} outputs, expected "
                             f"2I={g.two_i}: gate_up_proj is not [E, 2I, H]")
    gate, up = out.chunk(2, dim=-1)
    if swap_halves:
        gate, up = up, gate
    h = F.silu(gate) * up
    if h.shape[-1] != g.moe_intermediate_size:
        raise AssertionError(f"SwiGLU produced {h.shape[-1]}, expected I="
                             f"{g.moe_intermediate_size}")
    return F.linear(h, dn)


def expert_gather(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                  gate_up: torch.Tensor, down: torch.Tensor, g: Geometry, form: Form,
                  *, expert_index_shift: int = 0, swap_halves: bool = False,
                  transpose_stack: bool = False) -> dict:
    """The 8-of-256 gather, the SwiGLU and the weighted sum.

    `expert_raw[t, k]` is the selected expert's output **before** the routing weight is
    applied — `down_proj`'s output, which is both its producer's output and the weighted
    sum's input. `selftest_moe.py` re-derives it and asserts it is *not* the post-weight
    tensor (pitfalls/30 §A hook that reads output can pin the answer).

    The three keyword knobs exist only so the gate can inject one specific class of
    addressing/ordering bug at a time; every path that produces a pin passes the defaults.
    """
    if ids.ndim != 2:
        raise AssertionError(f"ids must be [T, K], got {tuple(ids.shape)}")
    T, K = ids.shape
    if K != g.top_k or x.shape[0] != T or x.shape[1] != g.hidden_size:
        raise AssertionError(f"ids {tuple(ids.shape)} / x {tuple(x.shape)} disagree with "
                             f"geometry (T, K={g.top_k}, H={g.hidden_size})")
    cd = {"bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[form.accumulate]
    if form.accumulate == "bf16" and x.dtype is not torch.bfloat16:
        raise AssertionError("bf16 accumulation requires a bf16 input")

    read_ids = ids if expert_index_shift == 0 else (ids + expert_index_shift) % g.num_experts
    if transpose_stack:
        # Emulate "the expert stack is stored transposed": index a [2I, E, H] view where
        # an [E, 2I, H] one belongs. Shape-compatible either way, so only numerics see it.
        gu_view = gate_up.reshape(g.two_i, g.num_experts, g.hidden_size)
        dn_view = down.reshape(g.hidden_size, g.num_experts, g.moe_intermediate_size)

        def pick(e: int):
            return gu_view[:, e, :], dn_view[:, e, :]
    else:
        def pick(e: int):
            return gate_up[e], down[e]

    raw = torch.zeros(T, K, g.hidden_size, dtype=cd)
    if form.batching == "grouped":
        # One GEMM per hit expert, over all rows that hit it — the grouped-launch form.
        for e_t in torch.unique(read_ids):
            e = int(e_t)
            tok, slot = torch.where(read_ids == e)
            gu, dn = pick(e)
            raw[tok, slot] = _expert_apply(x[tok], gu, dn, g, cd, swap_halves)
    elif form.batching == "per_slot":
        # One GEMM per (token, slot): the form a decode kernel emits, and the one whose
        # reduction order is independent of how many tokens share an expert.
        for k in range(K):
            for e_t in torch.unique(read_ids[:, k]):
                e = int(e_t)
                tok = torch.where(read_ids[:, k] == e)[0]
                gu, dn = pick(e)
                raw[tok, k] = _expert_apply(x[tok], gu, dn, g, cd, swap_halves)
    else:
        raise AssertionError(form.batching)

    # Weighted sum. `order` decides the association of the K addends, `accumulate` the
    # dtype they are added in; both are [REF] choices and both are measurable.
    # The product is always formed in f32 (f64 for the exact arm): a bf16*bf16 product is
    # exact in f32 (8+8 mantissa bits), so this reproduces the reference's single rounding
    # without depending on which internal opmath torch uses for bf16 elementwise mul.
    # cast-before rounds the weight to the model dtype first; cast-after does not.
    mul_cd = torch.float64 if cd is torch.float64 else torch.float32
    w = (weights.to(torch.bfloat16) if form.weight_cast == "before"
         else weights.to(mul_cd)).to(mul_cd)
    if form.order == "expert_asc":
        # [REF] the reference loops over experts ascending, so a token's K addends are
        # applied in ascending expert-id order — not in top-k rank order.
        order = torch.argsort(ids.to(torch.int64), dim=-1, stable=True)
    elif form.order == "rank":
        order = torch.arange(K).expand(T, K).contiguous()
    else:
        raise AssertionError(form.order)

    acc = torch.zeros(T, g.hidden_size, dtype=cd)
    for t in range(T):
        for j in range(K):
            k = int(order[t, j])
            prod = raw[t, k].to(mul_cd) * w[t, k]
            acc[t] += prod.to(cd)
    return {"expert_raw": raw, "routed_sum": acc, "compute_dtype": cd}


# ----------------------------------------------------------------------- shared expert
def shared_expert(x: torch.Tensor, w: BlockWeights, g: Geometry,
                  compute_dtype: torch.dtype) -> dict:
    x = x.to(compute_dtype)
    gw = w.tensors["shared_gate"].to(compute_dtype)
    uw = w.tensors["shared_up"].to(compute_dtype)
    dw = w.tensors["shared_down"].to(compute_dtype)
    sg = w.tensors["shared_scalar_gate"].to(compute_dtype)
    if gw.shape[0] != g.shared_expert_intermediate_size:
        raise AssertionError(f"shared expert I={gw.shape[0]} != config "
                             f"{g.shared_expert_intermediate_size}")
    raw = F.linear(F.silu(F.linear(x, gw)) * F.linear(x, uw), dw)
    if raw.shape != (x.shape[0], g.hidden_size):
        raise AssertionError(f"shared expert output {tuple(raw.shape)} != [T, H]")
    gate_prob = torch.sigmoid(F.linear(x, sg))
    return {"shared_raw": raw, "shared_gate_prob": gate_prob,
            "shared_out": gate_prob * raw}


# ------------------------------------------------------------------------- whole block
def moe_block(x: torch.Tensor, w: BlockWeights, g: Geometry, form: Form | str = "ref",
              *, expert_index_shift: int = 0, swap_halves: bool = False,
              transpose_stack: bool = False) -> dict:
    """The whole MoE block, returning every intermediate a gate needs under a name that
    says which stage it is (§The block, in words)."""
    form = Form.named(form) if isinstance(form, str) else form
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise AssertionError("only batch 1 is implemented; the engine is single-request")
        x = x.reshape(-1, x.shape[-1])
    r = route(x, w.tensors["router"], g, form)
    e = expert_gather(x, r["topk_ids"], r["topk_weights"], w.tensors["gate_up"],
                      w.tensors["down"], g, form,
                      expert_index_shift=expert_index_shift, swap_halves=swap_halves,
                      transpose_stack=transpose_stack)
    s = shared_expert(x, w, g, {"bf16": torch.bfloat16, "f32": torch.float32,
                               "f64": torch.float64}[form.accumulate])
    routed = e["routed_sum"]
    shared_out = s["shared_out"].to(routed.dtype)
    combined = routed + shared_out          # [REF] added in the model dtype
    return {
        "input": x,
        "router_logits": r["logits"],
        "router_probs": r["probs"],
        "topk_ids": r["topk_ids"],
        "topk_weights": r["topk_weights"],
        "topk_probs_pre_renorm": r["topk_probs_pre_renorm"],
        "expert_raw": e["expert_raw"],
        "routed_sum": routed,
        "shared_raw": s["shared_raw"],
        "shared_gate_prob": s["shared_gate_prob"],
        "shared_out": shared_out,
        "combined_out": combined,
        "form": form,
    }


# ------------------------------------------------------------------------- comparators
_BF16_RULER = None


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


def metrics(a: torch.Tensor, b: torch.Tensor, *, name: str = "") -> dict:
    """Aggregate comparator for two stages of the same tensor family.

    Both sides are widened with an explicit `.double()` — never `.view()`, which bitcasts
    and turns a bf16/f32 mix into garbage that still prints a plausible `ndiff`
    (`pitfalls/30` §Tensor.to casts, .view bitcasts and §Read a dump's dtype). Any NaN
    aggregate is an assertion, not a printed number (§the same file)."""
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype {a.dtype} vs {b.dtype} — read the dtype off "
                             "the tensor, not off the assumption (pitfalls/30)")
    x = a.detach().to(torch.float64).flatten()
    y = b.detach().to(torch.float64).flatten()
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise AssertionError(f"{name}: non-finite input to metrics()")
    scale = y.abs().max().item()
    if scale == 0:
        raise AssertionError(f"{name}: the reference side is all zero; a comparator over "
                             "zeros cannot fail")
    diff = (x - y).abs()
    eps_rel = 1e-3 * scale                       # cancellation floor for max_rel
    big = y.abs() > eps_rel
    denom = (x.norm() * y.norm()).item()
    if denom == 0:
        raise AssertionError(f"{name}: zero-norm side; cos is undefined, refuse to print")
    cos = (x @ y).item() / denom
    max_abs = diff.max().item()
    for v in (max_abs, cos):
        if v != v:
            raise AssertionError(f"{name}: NaN aggregate; the comparator is broken")
    ulp_ref = ulp_of(scale)
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "n": int(a.numel()),
        "max_abs": max_abs,
        "max_abs_over_scale": max_abs / scale,
        "max_rel_big": diff[big].max().item() / eps_rel if bool(big.any()) else 0.0,
        "n_over_1e3_rel": int((diff > 1e-3 * y.abs().clamp(min=eps_rel)).sum().item()),
        "mean_abs": diff.mean().item(),
        # clamped at 0: two bit-identical tensors give cos = 1 + 2e-16 from the f64
        # reduction order, and a negative 1-cos in a report reads like a broken comparator
        "one_minus_cos": max(0.0, 1.0 - cos),
        "cos_raw": cos,
        "ndiff": int((diff > 0).sum().item()),
        "ndiff_frac": (diff > 0).double().mean().item(),
        "n_gt_1ulp": int((diff > ulp_ref).sum().item()),
        "n_gt_100ulp": int((diff > 100 * ulp_ref).sum().item()),
        "ref_absmax": scale,
        "ulp_of_refmax": ulp_ref,
    }


def fmt_metrics(m: dict) -> str:
    return (f"{m['name']:<40} {m['dtype']:<15} {str(m['shape']):<18} "
            f"maxabs={m['max_abs']:.3e} rel/scale={m['max_abs_over_scale']:.3e} "
            f"1-cos={m['one_minus_cos']:.3e} ndiff={m['ndiff']}/{m['n']} "
            f"({100 * m['ndiff_frac']:.2f}%) >1ulp={m['n_gt_1ulp']}")


# ------------------------------------------------------------------- host-sync counting
class SyncCounter:
    """Counts host-synchronising calls made inside the block. §Host syncs.

    README.md §Predictions item 1 claims the router is the first per-token serial point
    and that `torch.where`/`nonzero` in the HF reference is what would make decode
    host-bound. The *count* of those calls is a CPU-only, timing-free adjudication of the
    structural half of that claim, so the instrument lives here and `cross_check_hf.py`
    drives it. It never reads a clock, so it cannot contaminate a measurement window."""

    TARGETS = (("torch", "where"), ("torch", "nonzero"), ("tensor", "nonzero"),
               ("tensor", "item"), ("tensor", "tolist"), ("tensor", "__bool__"))

    def __init__(self):
        self.counts: dict[str, int] = {}
        self._saved: list[tuple[object, str, object]] = []

    def __enter__(self):
        for kind, attr in self.TARGETS:
            obj = torch if kind == "torch" else torch.Tensor
            key = f"{kind}.{attr}"
            self.counts[key] = 0
            orig = getattr(obj, attr)
            self._saved.append((obj, attr, orig))
            outer, k = self, key

            def wrap(*a, _o=orig, _outer=outer, _k=k, **kw):
                _outer.counts[_k] += 1
                return _o(*a, **kw)

            setattr(obj, attr, wrap)
        return self

    def __exit__(self, *exc):
        for obj, attr, orig in self._saved:
            setattr(obj, attr, orig)
        self._saved = []
        return False

    def total(self) -> int:
        return sum(self.counts.values())

    def __str__(self):
        return " ".join(f"{k}={v}" for k, v in sorted(self.counts.items()) if v) + \
               f" total={self.total()}"


def set_threads(n: int | None) -> int:
    """Pin the intra-op thread count and report it. CPU matmul reduction order follows the
    thread partitioning, so a bf16 pin is only reproducible at a fixed count;
    `tools/dump_reference.py` records the value it used in the manifest."""
    if n is not None:
        if n < 1:
            raise AssertionError("threads must be >= 1")
        torch.set_num_threads(n)
    return torch.get_num_threads()
