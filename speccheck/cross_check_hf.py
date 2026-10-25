"""speccheck/cross_check_hf.py — feed the same inputs through the installed
`transformers` `qwen3_5_moe` router/experts/shared-expert and report agreement with
`speccheck/moe_ref.py`, the independent implementation.

CPU only: this script never touches a device and never reads a clock. It prints
comparisons, not timings (§What this is of speccheck/README.md).

Run:
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        speccheck/cross_check_hf.py --blocks layer0,layer1,layer39,mtp0

What it adjudicates (§Findings, recorded in speccheck/results.md):
  * can `transformers 5.19.0` build `Qwen/Qwen3.6-35B-A3B` from
    `checkpoints/Qwen3.6-35B-A3B.config.json`, and can it load what is on disk? The
    answer is printed verbatim into `--capability-log`, including where it fails.
  * does our independent math agree with the installed reference, per stage, and do the
    router top-k ids agree *exactly* (a discrete check — no band involved)?
  * how many host-synchronising calls does the reference's MoE forward make per block?
    That is README.md §Predictions item 1, adjudicated structurally (§Host syncs).
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import moe_ref as R  # noqa: E402  (our independent implementation — no HF imports in it)

# The continuous stages a gate compares, in pipeline order. `topk_ids` is discrete and is
# handled by an exact-equality check instead (§What this is).
STAGES = ["router_logits", "router_probs", "topk_weights", "expert_raw", "routed_sum",
          "shared_raw", "shared_gate_prob", "shared_out", "combined_out"]

# Structural bug detector, not a band: two implementations of the SAME math in the same
# dtype differ by at most a few bf16 ULPs, and bf16's relative width is 2^-9, so a
# 1-cos this far above it is a broken read, not a form gap. Derived, not fitted; the Q4_K
# and Q8_0 bands proper come from the floor probe in tools/moe_stats.py and are the
# user's to set (DECISIONS.md). Derived from: pitfalls/30 §A max-ULP band cannot see a
# whole-tensor 1-ULP error (long dots need aggregates) + bf16 2^-9 relative width.
BUG_DETECTOR_ONE_MINUS_COS = 1e-2


def canon(t: torch.Tensor) -> torch.Tensor:
    """Compare in f64 with an explicit widening cast — never `.view()` (pitfalls/30)."""
    return t.detach().to(torch.float64)


def canon_raw(raw: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Sort the K slot-axis of `expert_raw` by expert id before comparing.

    `expert_raw[t, k]` is indexed by *slot*, and the slot order is the top-k rank order —
    which is a tie-break, not a fact. Two correct implementations whose routers disagree
    about a tie hold the same SET of experts and put them in different slots, and an
    unsorted elementwise comparison then reads a slot permutation as a 3e-2 1-cos error on
    the tensor. Sorting along K by expert id makes the comparison about which experts ran,
    and the slot order is adjudicated separately by the exact-id check."""
    if raw.shape[1] != ids.shape[1]:
        raise AssertionError(f"raw {tuple(raw.shape)} and ids {tuple(ids.shape)} disagree")
    order = torch.argsort(ids.to(torch.int64), dim=-1, stable=True)
    return canon(torch.gather(raw, 1, order.unsqueeze(-1).expand_as(raw)))


def rows_same_set(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row mask: both routers chose the same SET of experts. Comparing a slot-indexed
    tensor across a selection difference compares two different sums, so this mask names the
    denominator of every `expert_raw` figure (pitfalls/40 §the unnamed denominator). A row
    whose set differs is not a numerics failure — it is the near-tie population, printed as
    such in the disagreement table."""
    if a.shape != b.shape:
        raise AssertionError(f"ids shapes {tuple(a.shape)} vs {tuple(b.shape)}")
    return (torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(dim=-1)


# Arms that are claimed-correct *forms* of the same math: the structural bug detector
# applies to them. Arms that are deliberate errors are the selftest's business and are
# required to be caught, not tolerated (§selftest_moe.py).
LEGAL_ARMS = ("ref", "spec", "exact", "cast_after", "accum_f32", "accum_f64",
              "rank_order", "per_slot")
WRONG_ARMS = ("no_renorm", "softmax_bf16")


def check_dtypes(out: dict, want: dict, where: str) -> None:
    for stage, dtype in want.items():
        got = out[stage].dtype
        if got != dtype:
            raise AssertionError(f"{where}: stage {stage} carries {got}, the form declares "
                                 f"{dtype} — refuse to print an aggregate (pitfalls/30)")


def build_hf_block(block: R.Block, w: R.BlockWeights, cfg_text):
    """Instantiate the installed reference's own `SparseMoeBlock` and load the real
    tensors into it.

    The module is built on the meta device and every parameter is then *replaced* by the
    loaded tensor: instantiating normally would allocate 3.2 GiB of fp32 [256,1024,2048]
    scratch per block before we overwrite it, and this box has 86 GiB free with three
    other workers running.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    import torch.nn as nn

    cfg_text._experts_implementation = "eager"   # [REF] pin the reference loop
    with torch.device("meta"):
        blk = M.Qwen3_5MoeSparseMoeBlock(cfg_text)
    params = {n for n, _ in blk.named_parameters()}
    want = {"gate.weight", "experts.gate_up_proj", "experts.down_proj",
            "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight", "shared_expert_gate.weight"}
    if params != want:
        raise AssertionError(f"SparseMoeBlock parameters changed: extra={sorted(params - want)} "
                             f"missing={sorted(want - params)} — the loader below is stale")
    blk.gate.weight = nn.Parameter(w.tensors["router"], requires_grad=False)
    blk.experts.gate_up_proj = nn.Parameter(w.tensors["gate_up"], requires_grad=False)
    blk.experts.down_proj = nn.Parameter(w.tensors["down"], requires_grad=False)
    blk.shared_expert.gate_proj.weight = nn.Parameter(w.tensors["shared_gate"], requires_grad=False)
    blk.shared_expert.up_proj.weight = nn.Parameter(w.tensors["shared_up"], requires_grad=False)
    blk.shared_expert.down_proj.weight = nn.Parameter(w.tensors["shared_down"], requires_grad=False)
    blk.shared_expert_gate.weight = nn.Parameter(w.tensors["shared_scalar_gate"], requires_grad=False)
    for n, p in blk.named_parameters():
        if p.device.type != "cpu":
            raise AssertionError(f"{n} is still on {p.device}")
    return blk, M


def hf_block_outputs(blk, x: torch.Tensor) -> dict:
    """Run the reference block and pull every stage out.

    `routed_sum`, `topk_ids` and the routing weights are taken from the reference's own
    `experts` forward by a hook on its inputs/outputs; `router_probs` and the softmax
    *dtype argument* are captured by wrapping `torch.nn.functional.softmax` for the
    duration of the call, because the reference does not return them and re-softmaxing its
    logits here would only prove that our own softmax agrees with itself. `expert_raw` is
    filled by the caller (§recompute_expert_raw): the reference multiplies the routing
    weight inside the expert loop and never returns the pre-weight tensor, and pinning the
    layer output instead of that intermediate is the pitfalls/30 hook trap."""
    import torch.nn.functional as F

    x2 = x.reshape(-1, x.shape[-1])
    captured: dict[str, object] = {}

    def experts_hook(mod, inp, out):
        # inp = (hidden_states, top_k_index, top_k_weights); out = the weighted sum.
        captured["routed_sum"] = out
        captured["hf_ids"] = inp[1]
        captured["hf_weights"] = inp[2]

    softmax_calls: list[tuple] = []
    orig_softmax = F.softmax

    def spy_softmax(*a, **kw):
        r = orig_softmax(*a, **kw)
        softmax_calls.append((a, kw, r))
        return r

    h = blk.experts.register_forward_hook(experts_hook)
    F.softmax = spy_softmax
    counter = R.SyncCounter()
    try:
        with counter, torch.no_grad():
            combined = blk(x2.unsqueeze(0))
    finally:
        F.softmax = orig_softmax
        h.remove()
    with torch.no_grad():
        logits, scores, ids = blk.gate(x2)
        shared_raw = blk.shared_expert(x2)
        sgate = torch.sigmoid(blk.shared_expert_gate(x2))

    if len(softmax_calls) != 1:
        raise AssertionError(f"the reference block made {len(softmax_calls)} softmax calls "
                             "in one forward; the capture below is stale")
    a, kw, probs = softmax_calls[0]
    dtype_arg = kw.get("dtype", a[2] if len(a) > 2 else None)
    if probs.shape != (x2.shape[0], blk.experts.num_experts):
        raise AssertionError(f"captured router_probs {tuple(probs.shape)} != [T, E]")
    print(f"  reference softmax call: args={len(a)} kwargs={ {k: str(v) for k, v in kw.items()} } "
          f"-> dtype {probs.dtype}, {probs.numel()} elements")

    shared_out = sgate * shared_raw
    return {
        "router_logits": logits,
        "router_probs": probs,
        "topk_ids": ids,
        "topk_weights": scores,
        "expert_raw": None,                      # filled by recompute_expert_raw
        "routed_sum": captured["routed_sum"],
        "shared_raw": shared_raw,
        "shared_gate_prob": sgate,
        "shared_out": shared_out,
        "combined_out": combined.reshape(x2.shape),
        "hf_ids": captured["hf_ids"],
        "hf_weights": captured["hf_weights"],
        "softmax_dtype_arg": dtype_arg,
        "sync_summary": str(counter),
        "sync_counts": dict(counter.counts),
    }


def recompute_expert_raw(x: torch.Tensor, ids: torch.Tensor, w: R.BlockWeights,
                         g: R.Geometry) -> torch.Tensor:
    """The reference's own per-expert output, recomputed with the reference's own ops
    (`chunk(2, dim=-1)` on the fused read, SwiGLU, down_proj) but WITHOUT the routing
    weight — the quantity the task pins as "each selected expert's output before
    weighting". [REF] form: grouped by expert, exactly as the reference batches it."""
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    act = ACT2FN["silu"]
    T, K = ids.shape
    raw = torch.zeros(T, K, g.hidden_size, dtype=x.dtype)
    for e in torch.unique(ids):
        e = int(e)
        tok, slot = torch.where(ids == e)
        rows = x[tok]
        gate, up = F.linear(rows, w.tensors["gate_up"][e]).chunk(2, dim=-1)
        raw[tok, slot] = F.linear(act(gate) * up, w.tensors["down"][e])
    return raw


# --------------------------------------------------------------- capability probe
PROBE = HERE / "hf_capability_probe.py"
# One CAP line per probe in hf_capability_probe.py; the driver refuses to call the section
# complete unless it parsed exactly this many (pitfalls/40 §A driver whose parser matches
# nothing prints PASS). This list is the probe file's own set of `show(...)` names, and it
# is checked against that file below, so it cannot drift silently.
CAP_NAMES = [
    "config_from_json", "build_conditional_generation_meta", "build_text_causallm_meta",
    "mtp_class_in_module", "keys_to_ignore_on_load_unexpected",
    "default_experts_implementation_bare_text_config",
    "default_experts_implementation_nested", "experts_implementation_after_model_build",
    "from_pretrained_real_index_26_shards",
    "from_pretrained_truncated_index_4_shards",
]


def capability_probe(log_path: Path) -> str:
    """Run the capability probe in a subprocess with an address-space limit, so an
    attempted 66 GiB materialisation fails loudly inside the probe instead of swapping the
    box the other three workers are running on.

    The parse is counted: the driver derives the expected probe list from the probe file's
    own `show("...")` calls and refuses to continue unless every one of them printed
    (pitfalls/40 §A driver whose parser matches nothing prints PASS)."""
    import re

    src = PROBE.read_text()
    declared = re.findall(r'show\("([^"]+)"', src)
    if sorted(declared) != sorted(CAP_NAMES):
        raise AssertionError(f"the probe file and CAP_NAMES disagree: "
                             f"file={sorted(declared)} driver={sorted(CAP_NAMES)}")
    code = PROBE
    p = subprocess.run([sys.executable, str(code), str(R.CONFIG_PATH), str(R.SHARD_DIR),
                        str(R.INDEX_PATH)],
                       capture_output=True, text=True, timeout=900)
    text = p.stdout + (("\nSTDERR:\n" + p.stderr[-2000:]) if p.stderr else "")
    parsed = [ln for ln in text.splitlines() if ln.startswith("CAP ")]
    got = [ln.split(":", 1)[0][len("CAP "):].strip() for ln in parsed]
    if len(parsed) != len(CAP_NAMES) or sorted(got) != sorted(CAP_NAMES):
        raise AssertionError(f"capability probe printed {len(parsed)} of {len(CAP_NAMES)} "
                             f"expected records (got {got}); the section cannot be trusted")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "# Capability probe of the installed transformers against this checkpoint.\n"
        f"# driver: {Path(__file__).resolve()}\n"
        f"# argv: {sys.executable} {code} {R.CONFIG_PATH} {R.SHARD_DIR} {R.INDEX_PATH}\n"
        f"# return code: {p.returncode}\n"
        "# (subprocess, RLIMIT_AS=24 GB so a 66 GiB materialisation fails inside the probe "
        "rather than swapping the box the other workers run on)\n\n"
        + text)
    return text


# ------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", default="ref,spec,exact,cast_after,softmax_bf16,no_renorm")
    ap.add_argument("--capability-log",
                    default=str(REPO / "speccheck" / "evidence" / "hf-capability-2026-10-26.txt"))
    ap.add_argument("--skip-capability-probe", action="store_true")
    args = ap.parse_args()

    threads = R.set_threads(args.threads)
    torch.manual_seed(args.seed)
    g = R.load_geometry()
    print(f"# cross_check_hf — torch {torch.__version__} CPU, threads={threads}, "
          f"seed={args.seed}, tokens={args.tokens}, blocks={args.blocks}")
    print(f"# geometry: E={g.num_experts} top_k={g.top_k} I={g.moe_intermediate_size} "
          f"shared_I={g.shared_expert_intermediate_size} H={g.hidden_size} "
          f"blocks={g.n_blocks} expert_bytes={g.expert_bytes}")

    if not args.skip_capability_probe:
        print("\n## capability probe (installed transformers vs this checkpoint)")
        txt = capability_probe(Path(args.capability_log))
        print(txt.rstrip())
        print(f"# written: {args.capability_log}")
    else:
        print("\n## capability probe SKIPPED (--skip-capability-probe)")

    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    cfg_text = Qwen3_5MoeTextConfig(**json.loads(R.CONFIG_PATH.read_text())["text_config"])

    blocks = [R.Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    # Derived denominators: a table with an empty population cannot fail (pitfalls/40).
    n_expected_stages = len(STAGES)
    printed_records = 0
    id_mismatch = 0
    id_checked = 0
    set_disagreements: dict[str, list[int]] = {}
    sync_rows: list[tuple[str, dict, int]] = []
    failures: list[str] = []

    for block in blocks:
        x = R.make_input(args.tokens, g, args.seed)
        w = R.load_block_weights(block, g)
        hf, _M = build_hf_block(block, w, cfg_text)
        print(f"\n## block {block.name} ({block.prefix[:-1]})")

        with R.SyncCounter() as _sc:
            h = hf_block_outputs(hf, x)
        print(f"  host-sync calls during one reference MoE block forward: {h['sync_summary']}")
        sync_rows.append((block.name, h["sync_counts"], args.tokens))

        # discrete check first: exact top-k id agreement, ordered and as a set
        ref_out = R.moe_block(x, w, g, "ref")
        ordered = int((ref_out["topk_ids"] == h["topk_ids"]).all(dim=-1).sum())
        as_set = int(sum(
            1 for t in range(args.tokens)
            if sorted(ref_out["topk_ids"][t].tolist()) == sorted(h["topk_ids"][t].tolist())))
        id_checked += args.tokens
        id_mismatch += args.tokens - as_set
        print(f"  topk ids vs reference: ordered {ordered}/{args.tokens}, "
              f"as-set {as_set}/{args.tokens}")
        if ordered != args.tokens:
            first = int((ref_out["topk_ids"] != h["topk_ids"]).any(dim=-1).nonzero()[0])
            print(f"    FIRST ORDERED MISMATCH row {first}: "
                  f"mine {ref_out['topk_ids'][first].tolist()} "
                  f"hf {h['topk_ids'][first].tolist()}")
        if as_set != args.tokens:
            failures.append(f"{block.name}: top-k id SET disagreement on "
                            f"{args.tokens - as_set}/{args.tokens} rows")

        # the reference's own pre-weight expert outputs
        h["expert_raw"] = recompute_expert_raw(x, h["topk_ids"], w, g)
        check_dtypes(h, R.expected_dtypes(R.Form()), f"{block.name}/hf_eager")

        for arm in [a for a in args.arms.split(",") if a.strip()]:
            form = R.Form.named(arm)
            out = R.moe_block(x, w, g, form)
            check_dtypes(out, R.expected_dtypes(form), f"{block.name}/{arm}")
            ids_same = int((out["topk_ids"] == h["topk_ids"]).all(dim=-1).sum())
            rows_mask = rows_same_set(out["topk_ids"], h["topk_ids"])
            ids_set_same = int(rows_mask.sum())
            set_disagreements.setdefault(arm, []).append(args.tokens - ids_set_same)
            note = ""
            if ids_set_same == args.tokens and ids_same != args.tokens:
                note = "  <-- slot order canonicalised in expert_raw below"
            elif ids_set_same != args.tokens:
                note = "  <-- SELECTION DIFFERS: expert_raw compared on agreeing rows only"
            print(f"  {block.name}/{arm}: top-k ordered {ids_same}/{args.tokens}, "
                  f"as-set {ids_set_same}/{args.tokens}{note}")
            if ids_set_same == 0:
                failures.append(f"{block.name}/{arm}: no row shares an expert SET with the "
                                "reference; expert_raw is not comparable at all")
            for stage in STAGES:
                if stage == "expert_raw":
                    a = canon_raw(out["expert_raw"][rows_mask], out["topk_ids"][rows_mask])
                    b = canon_raw(h["expert_raw"][rows_mask], h["topk_ids"][rows_mask])
                else:
                    a, b = canon(out[stage]), canon(h[stage])
                m = R.metrics(a, b, name=f"{block.name}/{arm} vs hf_eager/{stage}")
                print("  " + R.fmt_metrics(m)
                      + ("" if stage != "expert_raw" else f" rows={ids_set_same}/{args.tokens}"))
                printed_records += 1
                if arm in LEGAL_ARMS and m["one_minus_cos"] > BUG_DETECTOR_ONE_MINUS_COS:
                    failures.append(f"{block.name}/{arm}/{stage}: 1-cos="
                                    f"{m['one_minus_cos']:.3e} above the structural bug "
                                    f"detector {BUG_DETECTOR_ONE_MINUS_COS}")
                if arm in WRONG_ARMS and m["ndiff"] == 0 and stage in (
                        "topk_weights", "routed_sum", "combined_out"):
                    failures.append(f"{block.name}/{arm}/{stage}: a deliberate error left "
                                    "this stage untouched — the injection is vacuous")
            # ids per arm, discrete
            print(f"  {block.name}/{arm} vs hf_eager/topk_ids              int64"
                  f"{' ':<23} ordered {ids_same}/{args.tokens}")
            printed_records += 1
            if arm in ("ref", "spec") and ids_same != args.tokens:
                failures.append(f"{block.name}/{arm}: top-k ordered ids differ on "
                                f"{args.tokens - ids_same} rows")
        del hf, w

    expected = len(blocks) * len(args.arms.split(",")) * (n_expected_stages + 1)
    print(f"\n## coverage: printed comparison records {printed_records}, "
          f"expected {expected} ({len(blocks)} blocks x "
          f"{len(args.arms.split(','))} arms x {n_expected_stages + 1} stages)")
    if printed_records != expected or printed_records == 0:
        failures.append(f"record count {printed_records} != derived {expected}")
    print(f"## top-k id rows checked {id_checked}, set-disagreements {id_mismatch}")
    print("## selection disagreement vs the reference, by arm — rows whose top-8 SET differs."
          " This is the near-tie population; a Q4_K expert path cannot recover it (the router"
          " is exact by policy, DECISIONS.md D-01), so it bounds what an expert-side gate can"
          " promise at the decision level")
    for arm, lst in sorted(set_disagreements.items()):
        tot = sum(lst)
        print(f"  {arm:<14} " + " ".join(f"{n}/{args.tokens}" for n in lst)
              + f"   total {tot}/{args.tokens * len(blocks)} rows "
                f"({100 * tot / (args.tokens * len(blocks)):.2f}%)")
    if not set_disagreements:
        failures.append("no arm was checked for selection disagreement: the section is vacuous")
    if sync_rows:
        keys = sorted({k for _, c, _ in sync_rows for k in c})
        print("## host-sync call counts per reference MoE block forward "
              f"(README prediction 1; {len(sync_rows)} blocks: "
              f"{', '.join(n for n, _, _ in sync_rows)})")
        for n, c, toks in sync_rows:
            print(f"  {n:<8} T={toks:<4} " + " ".join(f"{k}={c[k]}" for k in keys))
        # The decode-shaped case is the one prediction 1 is about: one token, so the
        # reference loop runs once per distinct expert selected by that single token, i.e.
        # exactly top_k times. Count only — the tok/s half of that prediction is a device
        # measurement and belongs to the main agent (AGENTS.md §GPU execution).
        x1 = R.make_input(1, g, args.seed)
        b0 = blocks[0]
        w1 = R.load_block_weights(b0, g)
        hf1, _ = build_hf_block(b0, w1, cfg_text)
        h1 = hf_block_outputs(hf1, x1)
        c1 = h1["sync_counts"]
        hits1 = int(len(torch.unique(h1["topk_ids"])))
        print(f"  {b0.name:<8} T=1     " + " ".join(f"{k}={c1[k]}" for k in keys)
              + f"   [decode shape: {hits1} distinct experts hit = top_k={g.top_k}]")
        if hits1 != g.top_k:
            failures.append(f"decode-shape probe hit {hits1} experts, expected top_k={g.top_k}")
        per_tok = sum(c1.values())
        print(f"  decode shape: {per_tok} host-synchronising calls per MoE block forward; "
              f"x {g.n_blocks} MoE blocks in the text path = {per_tok * g.n_blocks} per token "
              "if the reference form were shipped. COUNT ONLY — no timing taken, and the "
              "tok/s consequence is the main agent's measurement.")
        if per_tok == 0:
            failures.append("decode-shape host-sync count is 0: the counter is vacuous")
        del hf1, w1
    else:
        failures.append("no block was measured for host-sync calls: the section is vacuous")

    if failures:
        print("\nCROSS_CHECK FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nCROSS_CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
