#!/usr/bin/env python3
"""speccheck/selftest_moe.py — the gate for `speccheck/moe_ref.py`, the pinned dumps and
the two form choices README.md §Predictions raises as questions (3 and 4).

CPU only: torch-CPU, no device allocation, and no clock is read. It prints counts and
verdicts; `no timing taken`.

Three ARMS, and every printed number names the arm it belongs to. A tally without an arm
label is the failure mode this file exists to prevent: an earlier revision printed
`must_not_catch_violations 192` beside `SELFTEST PASS ... 0 must-not-catch violations`,
which is two truths about one run and therefore neither of them verifiable.

  honest   the differ as it ships. The must-catch set must fire and the must-not-catch set
           must stay bit-identical. THIS arm carries the localisation claim, so a witness
           that reads 0 here fails the run: `stage_diffs`, `shared_moved_cases`,
           `router_moved_cases` and `ids_flips` must all be > 0, and
           `must_not_catch_violations` must be exactly 0.
  stub     the differ's "did this stage move" predicate replaced by `lambda: False` —
           "assume no difference exists". Every case must then lose its must-catch, so
           `caught` and `stage_diffs` must be 0 and `must_catch_missed` must equal the
           derived, non-zero number of allowed stages. If the numbers do not move, the
           check was reading nothing (pitfalls/40 §witness it by stubbing the predicate).
  crossed  a deliberate must-not-catch BREACH: one stage the case's contract says must stay
           identical is perturbed, per case. The localisation check must fire on exactly
           that stage, so `must_not_catch_violations` must equal the derived, non-zero case
           count. Without this arm the must-not-catch side is never given the chance to
           fail and "0 violations" is unmeasured, not earned.

The exit status is computed from the same tallies that are printed, per arm, in one place
at the end of the file.

Usage:
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        speccheck/selftest_moe.py --selftest
    ... speccheck/selftest_moe.py --against-dumps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import moe_ref as R  # noqa: E402

# The block's stages, in pipeline order. `shared_*` sits outside the router, so no
# router-side or gather-side injection may move them — and that is checked, not assumed.
STAGES = ["router_logits", "router_probs", "topk_ids", "topk_weights", "expert_raw",
          "routed_sum", "shared_raw", "shared_gate_prob", "shared_out", "combined_out"]
FLOAT_STAGES = [s for s in STAGES if s != "topk_ids"]

# The UNCONDITIONAL half of the contract: which stages each injection is allowed to move,
# derived from §The block, in words in `moe_ref.py`: the gather feeds expert_raw, expert_raw
# feeds routed_sum, both feed combined_out; the routing weights feed only the weighted sum;
# the softmax feeds the probabilities the top-k is taken from. Anything not listed here or
# in CONDITIONAL_AFFECTED must stay bit-identical.
AFFECTED = {
    "wrong_expert_index": ["expert_raw", "routed_sum", "combined_out"],
    "swapped_gate_up_half": ["expert_raw", "routed_sum", "combined_out"],
    "transposed_expert_stack": ["expert_raw", "routed_sum", "combined_out"],
    "renorm_omitted": ["topk_weights", "routed_sum", "combined_out"],
    "cast_after_multiply": ["topk_weights", "routed_sum", "combined_out"],
    "router_softmax_bf16": ["router_probs", "topk_weights", "routed_sum", "combined_out"],
    "swapped_shared_gate_up": ["shared_raw", "shared_out", "combined_out"],
}

# The CONDITIONAL half: stages a case may additionally move when a discrete fact holds
# rather than always. `route()` takes the top-k out of softmax(softmax_dtype), so a bf16
# softmax can reorder the selection; when it does, the reported ids themselves move and,
# because `expert_gather` indexes the [E, 2I, H] stack by those ids, the gathered experts
# move with them. So {topk_ids, expert_raw} are licensed for the bf16-softmax arm exactly
# on a flip — not always (a value-only change must leave both identical), never by guess.
#
# This table is named, and read by BOTH `allowed_stages()` and
# `derive_expected_allowed_stages()`, on the pitfalls/40 rule that a fact held in two
# places can be committed in only one of them. While the conditional pair lived as a bare
# literal inside `allowed_stages()`, the contract said 48 stages per block and the header
# derivation said 46; the stub arm reported that disagreement for weeks as
# "missed 192 allowed stages, derived 184" while the differ was innocent of both numbers.
CONDITIONAL_AFFECTED = {
    "router_softmax_bf16": ("topk_ids", "expert_raw"),
}


def cases_for(g: R.Geometry):
    """The derived case list. Every count here is a function of the config geometry."""
    cases = []
    for slot in range(g.top_k):                             # one per gather slot
        cases.append(("wrong_expert_index", {"slot": slot}))
    cases.append(("wrong_expert_index", {"slot": None}))    # all slots at once
    n_halves = g.two_i // g.moe_intermediate_size           # == 2 for this model
    for _ in range(n_halves - 1):                # permutations of the halves minus identity
        cases.append(("swapped_gate_up_half", {}))
        cases.append(("swapped_shared_gate_up", {}))        # same count, shared side
    cases.append(("transposed_expert_stack", {}))  # swap dims 0 and 1 of the rank-3 read
    cases.append(("renorm_omitted", {}))
    cases.append(("cast_after_multiply", {}))
    cases.append(("router_softmax_bf16", {}))
    return cases


def populations(g: R.Geometry) -> dict:
    """The expected size of every section, as a function of the geometry."""
    return {
        "wrong_expert_index": g.top_k + 1,
        "swapped_gate_up_half": g.two_i // g.moe_intermediate_size - 1,
        "swapped_shared_gate_up": g.two_i // g.moe_intermediate_size - 1,
        "transposed_expert_stack": 1,
        "renorm_omitted": 1,
        "cast_after_multiply": 1,
        "router_softmax_bf16": 1,
    }


def run_case(g, w, x, kind, kw):
    """Recompute the block with exactly one injection applied."""
    if kind == "wrong_expert_index":
        return _with_slot_shift(g, w, x, kw["slot"])
    if kind == "swapped_gate_up_half":
        return R.moe_block(x, w, g, "ref", swap_halves=True)
    if kind == "transposed_expert_stack":
        return R.moe_block(x, w, g, "ref", transpose_stack=True)
    if kind == "renorm_omitted":
        return R.moe_block(x, w, g, R.Form(renorm=False))
    if kind == "cast_after_multiply":
        return R.moe_block(x, w, g, R.Form(weight_cast="after"))
    if kind == "router_softmax_bf16":
        return R.moe_block(x, w, g, R.Form(softmax_dtype="bf16"))
    if kind == "swapped_shared_gate_up":
        swapped = dict(w.tensors)
        swapped["shared_gate"], swapped["shared_up"] = (w.tensors["shared_up"],
                                                        w.tensors["shared_gate"])
        return R.moe_block(x, R.BlockWeights(w.block, swapped, w.source_shards), g, "ref")
    raise AssertionError(f"unknown injection {kind}")


def _with_slot_shift(g, w, x, slot):
    """The addressing bug: the router says expert e, the gather reads expert e+1 and still
    reports e. Modelled by gathering on mutated ids while the *reported* ids stay clean —
    which is exactly what makes `topk_ids` a must-not-catch stage."""
    form = R.Form()
    r = R.route(x, w.tensors["router"], g, form)
    ids = r["topk_ids"]
    read_ids = ids.clone()
    if slot is None:
        read_ids = (read_ids + 1) % g.num_experts
    else:
        read_ids[:, slot] = (read_ids[:, slot] + 1) % g.num_experts
    e = R.expert_gather(x, read_ids, r["topk_weights"], w.tensors["gate_up"],
                        w.tensors["down"], g, form)
    s = R.shared_expert(x, w, g, torch.bfloat16)
    shared_out = s["shared_out"].to(e["routed_sum"].dtype)
    return {"router_logits": r["logits"], "router_probs": r["probs"], "topk_ids": ids,
            "topk_weights": r["topk_weights"], "expert_raw": e["expert_raw"],
            "routed_sum": e["routed_sum"], "shared_raw": s["shared_raw"],
            "shared_gate_prob": s["shared_gate_prob"], "shared_out": shared_out,
            "combined_out": e["routed_sum"] + shared_out}


def differ(a: torch.Tensor, b: torch.Tensor, predicate=None) -> tuple[bool, float]:
    """(did the stage move, max|a-b|). The SINGLE comparison used by both directions, so the
    must-not-catch side is witnessed by the must-catch side firing on the same machinery."""
    if a.shape != b.shape:
        raise AssertionError(f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    n = int((x != y).sum().item())
    changed = n > 0
    if predicate is not None:
        changed = predicate(changed)
    if n == 0:
        return changed, 0.0
    m = R.metrics(x, y, name="")
    return changed, m["max_abs"]


def allowed_stages(kind, clean, dirty):
    """The contract for one case, with the conditional stages resolved from the discrete
    fact (did the selection change), not from a guess. Reads the same two tables the derived
    stub-arm expectation reads; neither may carry a stage the other does not."""
    allowed = set(AFFECTED[kind])
    flipped = not torch.equal(dirty["topk_ids"], clean["topk_ids"])
    if flipped:
        allowed |= set(CONDITIONAL_AFFECTED.get(kind, ()))
    return allowed, flipped


def case_key(block_name, kind, kw):
    """One key shape for one case, shared by the flip-fact record and the verdict that
    re-derives the case set — the same expression on both sides (pitfalls/40 §two
    self-consistent layouts joined by a branch)."""
    return (block_name, kind, json.dumps(kw, sort_keys=True))


def derive_expected_allowed_stages(cases, n_blocks, flips):
    """The (case x allowed-stage) pairs the contract licenses over the whole matrix — the
    stub arm's non-zero expectation. Never read back off a tally.

    Base: |AFFECTED[kind]| for every case in the derived list, x the number of blocks.
    Conditional: |CONDITIONAL_AFFECTED[kind]| for every case whose discrete fact holds —
    `flips[key]`, the plain `torch.equal` on the reported topk_ids that the differ's
    predicate does not touch. A conditional population of 0 is a real, reportable outcome
    (nothing flipped, nothing licensed), and `verdict_flips` fails the run in that case
    rather than letting the arm pass on an untested half of its contract.
    """
    base = sum(len(AFFECTED[kind]) for kind, _ in cases) * n_blocks
    conditional, detail = 0, []
    for kind, extra in sorted(CONDITIONAL_AFFECTED.items()):
        keys = [k for k in flips if k[1] == kind]
        n_flip = sum(1 for k in keys if flips[k])
        conditional += len(extra) * n_flip
        detail.append(f"{len(extra) * n_flip:>4}   {len(extra)} stage(s) "
                      f"{','.join(sorted(extra))} x {n_flip}/{len(keys)} "
                      f"{kind} case(s) flipped topk_ids")
    return base, conditional, detail


# --------------------------------------------------------------------------- the matrix
def run_matrix(g, blocks, cases, args, arm, predicate, crossed):
    """One pass over the derived case matrix under one differ policy.

    Returns (tally, rows, fails, flips), where `flips` maps each case key to the discrete
    verdict (did the reported topk_ids change) that licenses the conditional stages. Every
    tally key is printed with `arm` beside it by the caller; nothing here prints a number
    that the verdict does not also read."""
    flips = {}
    tally = {"cases": 0, "caught": 0, "localised": 0, "stage_diffs": 0, "ids_flips": 0,
             "shared_moved_cases": 0, "router_moved_cases": 0,
             "must_not_catch_violations": 0, "must_catch_missed": 0,
             "crossed_breaches_injected": 0, "crossed_alarms_fired": 0}
    rows, fails = [], []
    for block in blocks:
        w = R.load_block_weights(block, g)
        for kind, kw in cases:
            toks = args.softmax_tokens if kind == "router_softmax_bf16" else args.tokens
            x = R.make_input(toks, g, args.seed)
            clean = R.moe_block(x, w, g, "ref")
            dirty = run_case(g, w, x, kind, kw)
            allowed, flipped = allowed_stages(kind, clean, dirty)
            tally["ids_flips"] += int(flipped)
            flips[case_key(block.name, kind, kw)] = flipped

            breach = None
            if crossed:
                # Positive control for the must-not-catch side: pick a stage this case's
                # contract says must stay bit-identical, confirm it currently is, then move
                # it. The localisation check must see exactly this.
                cands = [s for s in FLOAT_STAGES if s not in allowed
                         and torch.equal(clean[s], dirty[s])]
                if not cands:
                    fails.append(f"{arm}/{block.name}/{kind}: no must-not-catch stage is "
                                 "currently identical, so the breach cannot be injected")
                    continue
                breach = cands[0]
                step = R.ulp_of(float(clean[breach].abs().max())) or 1.0
                dirty = dict(dirty)
                dirty[breach] = clean[breach] + step
                tally["crossed_breaches_injected"] += 1

            affected_now, violations, max_seen = set(), [], 0.0
            for stage in STAGES:
                changed, max_abs = differ(clean[stage], dirty[stage], predicate)
                max_seen = max(max_seen, max_abs)
                if changed:
                    affected_now.add(stage)
                    tally["stage_diffs"] += 1
                if (stage in allowed) != (stage in affected_now):
                    violations.append(f"{stage}:" + ("silent" if stage in allowed else "moved"))
            tally["cases"] += 1
            if affected_now & allowed:
                tally["caught"] += 1
            tally["must_catch_missed"] += len([v for v in violations if v.endswith("silent")])
            if affected_now & {"shared_raw", "shared_out"}:
                tally["shared_moved_cases"] += 1
            if affected_now & {"router_logits", "router_probs", "topk_weights"}:
                tally["router_moved_cases"] += 1
            if not violations:
                tally["localised"] += 1
            else:
                tally["must_not_catch_violations"] += len(violations)
            if crossed and breach is not None:
                if any(v.startswith(breach + ":moved") for v in violations):
                    tally["crossed_alarms_fired"] += 1
                else:
                    fails.append(f"{arm}/{block.name}/{kind}: injected a breach of "
                                 f"{breach} and the localisation check did not notice")
            if violations and not (crossed or predicate is not None):
                fails.append(f"{arm}/{block.name}/{kind}"
                             f"{'' if not kw else json.dumps(kw, sort_keys=True)}: "
                             + "; ".join(violations))
            if not affected_now and not crossed and predicate is None:
                fails.append(f"{arm}/{block.name}/{kind}: NO stage moved — injection vacuous")
            rows.append((block.name, kind, json.dumps(kw, sort_keys=True), toks,
                         sorted(affected_now), len(violations), max_seen,
                         breach or ""))
        del w
    return tally, rows, fails, flips


def print_table(arm, rows, args):
    print(f"\n## firing table — arm={arm} (the comparison is bit-level; no band is involved)")
    print(f"  {'block':<7} {'injection':<24} {'param':<14} {'T':>4} "
          f"{'moved stages':<62} {'viol':>5} maxabs {'breach':<16}")
    for r in rows:
        print(f"  {r[0]:<7} {r[1]:<24} {r[2]:<14} {r[3]:>4} "
              f"{','.join(r[4]) if r[4] else '(NOTHING MOVED)':<62} {r[5]:>5} "
              f"{r[6]:.3e} {r[7]:<16}")


TALLY_KEYS = ("cases", "caught", "localised", "stage_diffs", "ids_flips",
              "shared_moved_cases", "router_moved_cases", "must_not_catch_violations",
              "must_catch_missed", "crossed_breaches_injected", "crossed_alarms_fired")


def print_tally(arm, tally, extra=""):
    print(f"\n## tallies — arm={arm}{extra}")
    for k in TALLY_KEYS:
        print(f"  {arm:<8} {k:<28} {tally[k]}")


def verdict_honest(tally, expected_cases, fails, tag):
    """The arm that carries the real claim: must-catch fires, must-not-catch holds, and
    every non-vacuity witness is strictly positive here."""
    if tally["cases"] != expected_cases:
        fails.append(f"{tag}: case count {tally['cases']} != derived {expected_cases}")
    if tally["caught"] != tally["cases"]:
        fails.append(f"{tag}: {tally['cases'] - tally['caught']} cases moved no allowed stage")
    if tally["must_not_catch_violations"] != 0:
        fails.append(f"{tag}: {tally['must_not_catch_violations']} must-not-catch violations "
                     "(an injection moved a stage its contract says it cannot reach)")
    if tally["localised"] != tally["cases"]:
        fails.append(f"{tag}: only {tally['localised']}/{tally['cases']} cases localised")
    # non-vacuity: a witness that reads 0 in the arm that is supposed to fire has not
    # witnessed anything (pitfalls/40, pitfalls/20 §a counter that can only be 0 cannot fail)
    for k in ("stage_diffs", "shared_moved_cases", "router_moved_cases", "ids_flips"):
        if tally[k] == 0:
            fails.append(f"{tag}: witness {k} reads 0 — that side of the check never fired, "
                         "so it proves nothing about the case it is claimed to prove")


def verdict_stub(tally, expected_cases, expected_allowed_stages, fails, tag):
    """The stubbed arm: with 'assume no difference', nothing may be caught and every allowed
    stage must be missed. Zeroes here are the POINT, which is why they are asserted against
    a derived non-zero expectation (missed stages) rather than admired."""
    if tally["stage_diffs"] != 0:
        fails.append(f"{tag}: the stub did not blind the gate (stage_diffs="
                     f"{tally['stage_diffs']})")
    if tally["caught"] != 0:
        fails.append(f"{tag}: the stubbed gate still caught {tally['caught']} cases — the "
                     "must-catch assertion is not reading the differ")
    if tally["must_catch_missed"] != expected_allowed_stages:
        fails.append(f"{tag}: stub missed {tally['must_catch_missed']} allowed stages, "
                     f"derived {expected_allowed_stages}")
    if expected_allowed_stages == 0:
        fails.append(f"{tag}: derived allowed-stage population is 0; the stub arm is vacuous")


def verdict_flips(arm_flips, cases, blocks, fails):
    """The flip facts feed the stub arm's derived expectation, so they must be one fact and
    not three per-arm opinions, and they must not be free:

      - every derived case must carry a verdict, or the expectation is for a different
        case set than the tally (an unnamed denominator);
      - only a section that declares conditional stages may flip at all. An unlicensed flip
        would inflate the derived population with stages the injection cannot reach, and
        `wrong_expert_index` in particular is contracted to *report* clean ids while reading
        mutated ones — a flip there is the injection leaking, not a fact;
      - the conditional half of the contract must actually have been licensed somewhere, or
        those stages are untested rather than satisfied;
      - every arm must read the same discrete fact. The predicate is applied inside `differ`
        and never to `torch.equal`, and the crossed arm draws its breach from FLOAT_STAGES
        (which excludes topk_ids), so disagreement between arms means the fixtures or the
        breach moved something that should not have moved."""
    keys = {case_key(b.name, kind, kw) for b in blocks for kind, kw in cases}
    for arm, fl in sorted(arm_flips.items()):
        if set(fl) != keys:
            fails.append(f"arm={arm}: recorded {len(fl)} flip facts, derived {len(keys)} "
                         "— the stub expectation would be counting a different case set")
    honest = arm_flips.get("honest", {})
    for (block, kind, param), flipped in sorted(honest.items()):
        if flipped and kind not in CONDITIONAL_AFFECTED:
            fails.append(f"arm=honest/{block}/{kind}{param}: topk_ids changed but {kind} "
                         "declares no conditional stages — the injection moved the selection "
                         "it is contracted to report unchanged")
    for kind, extra in sorted(CONDITIONAL_AFFECTED.items()):
        if not any(k[1] == kind and v for k, v in honest.items()):
            fails.append(f"arm=honest: no {kind} case flipped topk_ids, so the conditional "
                         f"stages {'/'.join(sorted(extra))} were never licensed by a real "
                         "fact — that half of the contract is untested, not satisfied")
    for arm, fl in sorted(arm_flips.items()):
        if arm == "honest":
            continue
        for k in sorted(set(fl) & set(honest)):
            if fl[k] != honest[k]:
                fails.append(f"arm={arm}: flip fact for {k} reads {fl[k]} but arm=honest "
                             f"read {honest[k]} — the arms are not comparing one fixture")
                break


def verdict_crossed(tally, expected_cases, fails, tag):
    """The positive-control arm: the must-not-catch check must fire on every injected breach."""
    if tally["crossed_breaches_injected"] != expected_cases:
        fails.append(f"{tag}: injected {tally['crossed_breaches_injected']} breaches, "
                     f"expected {expected_cases}")
    if tally["crossed_alarms_fired"] != expected_cases:
        fails.append(f"{tag}: the localisation check missed "
                     f"{tally['crossed_breaches_injected'] - tally['crossed_alarms_fired']} "
                     "injected breaches — the must-not-catch side cannot fail, so it cannot "
                     "be trusted when it reports 0")
    if tally["crossed_alarms_fired"] == 0:
        fails.append(f"{tag}: crossed arm fired 0 alarms: vacuous positive control")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--softmax-tokens", type=int, default=64,
                    help="tokens for the router-softmax case, which must be shown able to "
                         "change a decision, not only a value")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--against-dumps", action="store_true")
    ap.add_argument("--manifest", default="checkpoints/dump-manifest-2026-10-26.json")
    args = ap.parse_args()

    if not args.selftest and not args.against_dumps:
        print("SELFTEST FAIL: nothing selected (--selftest or --against-dumps)")
        return 1

    threads = R.set_threads(args.threads)
    g = R.load_geometry()
    blocks = [R.Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    cases = cases_for(g)
    print(f"# selftest_moe — torch {torch.__version__} CPU threads={threads} seed={args.seed}")
    print(f"# geometry from {g.config_path}: E={g.num_experts} top_k={g.top_k} "
          f"I={g.moe_intermediate_size} shared_I={g.shared_expert_intermediate_size} "
          f"H={g.hidden_size} layers={g.num_hidden_layers} +mtp={g.mtp_num_hidden_layers} "
          f"=> {g.n_blocks} MoE blocks in the text path")
    print(f"# blocks available in the acquired shards: {len(blocks)} of {g.n_blocks} "
          f"({', '.join(b.name for b in blocks)}) — coverage stated, not implied")
    expected_cases = len(cases) * len(blocks)
    print(f"# derived cases per block: {len(cases)}; total {expected_cases}")

    fails: list[str] = []
    pops = populations(g)
    if {k for k, _ in cases} != set(AFFECTED):
        fails.append(f"derived injections {sorted({k for k, _ in cases})} != declared "
                     f"{sorted(AFFECTED)}")
    for k, want in sorted(pops.items()):
        n = len([c for c in cases if c[0] == k])
        if n != want or n == 0:
            fails.append(f"section {k} derived {n} cases, expected {want}")
            continue
        print(f"#   section {k}: {n} case(s) per block (derived)")
    if sum(pops.values()) != len(cases):
        fails.append("case list and population table disagree")

    # The table the stub arm's expectation comes from. The unconditional half is a function
    # of the geometry and the case list and is printed here, before any comparison runs; the
    # conditional half depends on a discrete fact that does not exist until the fixtures do,
    # so it is derived after the arms and printed with its denominator beside it.
    print("# derived stage population per section, per block "
          "(AFFECTED / CONDITIONAL_AFFECTED x the case list):")
    print(f"#   {'section':<24} {'cases':>5} {'stages/case':>11} {'stages/blk':>10}"
          f"  conditional (licensed by a topk_ids flip)")
    per_block_base = 0
    for kind in list(dict.fromkeys(k for k, _ in cases)):
        n = pops[kind]
        extra = CONDITIONAL_AFFECTED.get(kind, ())
        per_block_base += n * len(AFFECTED[kind])
        print(f"#   {kind:<24} {n:>5} {len(AFFECTED[kind]):>11} "
              f"{n * len(AFFECTED[kind]):>10}  "
              + (f"+{len(extra)} {','.join(sorted(extra))}" if extra else "-"))
    print(f"#   {'unconditional total':<24} {len(cases):>5} {'':>11} {per_block_base:>10}")
    print(f"# derived (case x allowed-stage) pairs, unconditional: "
          f"{per_block_base * len(blocks)} (= {per_block_base} per block x {len(blocks)} "
          f"blocks); the conditional pairs are derived after the arms, from the flip facts")
    if fails:
        print("\nSELFTEST FAIL (derived population)")
        for f in fails:
            print("  - " + f)
        return 1

    arms, arm_flips = {}, {}
    if args.selftest:
        for arm, predicate, crossed in (("honest", None, False),
                                        ("stub", (lambda c: False), False),
                                        ("crossed", None, True)):
            t, rows, f, fl = run_matrix(g, blocks, cases, args, arm, predicate, crossed)
            arms[arm], arm_flips[arm] = t, fl
            print_table(arm, rows, args)
            print_tally(arm, t)
            fails += f
        # The stub arm's expectation, derived from the contract tables and this arm's own
        # discrete facts — never from the tally it is about to be compared with.
        base, conditional, cond_detail = derive_expected_allowed_stages(
            cases, len(blocks), arm_flips["stub"])
        expected_allowed_stages = base + conditional
        print("\n## derived (case x allowed-stage) population — the arm=stub expectation")
        print(f"  unconditional   {base:>4}   |AFFECTED[kind]| x {len(cases)} cases x "
              f"{len(blocks)} blocks")
        for line in cond_detail:
            print(f"  conditional   {line}")
        print(f"  derived total   {expected_allowed_stages:>4}   the blinded differ must "
              f"miss exactly this many allowed stages, no more and no fewer")
        verdict_honest(arms["honest"], expected_cases, fails, "arm=honest")
        verdict_stub(arms["stub"], expected_cases, expected_allowed_stages, fails,
                     "arm=stub")
        verdict_crossed(arms["crossed"], expected_cases, fails, "arm=crossed")
        verdict_flips(arm_flips, cases, blocks, fails)
        print("\n## what each witness is evidence for")
        print("  arm=honest   stage_diffs>0, shared_moved_cases>0, router_moved_cases>0, "
              "ids_flips>0  — the four sides of the check that must be able to fire")
        print("  arm=honest   must_not_catch_violations==0                  — localisation")
        print("  arm=stub     caught==0, stage_diffs==0, must_catch_missed==derived>0 "
              "— the gate reads the differ, not a constant")
        print("  flip facts   the conditional stages are licensed case-by-case by a real "
              "topk_ids flip, agreed across all three arms")
        print("  arm=crossed  crossed_alarms_fired==cases                   — the "
              "must-not-catch side can fail, so its 0 in arm=honest is earned")
        print("  arm labels are printed with every count above; no number in this file is "
              "unlabelled, and the verdict reads these same tallies")

    if args.against_dumps:
        fails += check_against_dumps(g, Path(args.manifest))

    if fails:
        print("\nSELFTEST FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    h = arms.get("honest")
    if h:
        print(f"\nSELFTEST PASS — honest arm: {h['cases']}/{expected_cases} derived cases "
              f"caught, {h['stage_diffs']} stage differences, "
              f"{h['must_not_catch_violations']} must-not-catch violations; "
              f"stub arm: {arms['stub']['caught']} caught (expected 0), "
              f"{arms['stub']['must_catch_missed']}/{expected_allowed_stages} allowed stages "
              f"missed; crossed arm: {arms['crossed']['crossed_alarms_fired']} alarms on "
              f"{arms['crossed']['crossed_breaches_injected']} injected breaches. "
              "no timing taken")
    else:
        print("\nSELFTEST PASS (against-dumps only). no timing taken")
    return 0


def check_against_dumps(g: R.Geometry, manifest_path: Path) -> list[str]:
    """Recompute every `arm=ref` tensor in the manifest and require bit-equality. Without
    this the dumps are an assertion by the program that wrote them."""
    if not manifest_path.exists():
        return [f"{manifest_path} does not exist; run tools/dump_reference.py first"]
    m = json.loads(manifest_path.read_text())
    wanted = defaultdict(list)
    for rec in m["dumps"]:
        if rec["arm"] == "ref" and rec["tensor_key"] != "stored":
            wanted[rec["block"]].append(rec)
    fails, checked, total = [], 0, sum(len(v) for v in wanted.values())
    for name, recs in wanted.items():
        block = R.Block.parse(name)
        x = R.make_input(m["input"]["n_tokens"], g, m["input"]["seed"])
        w = R.load_block_weights(block, g)
        out = R.moe_block(x, w, g, "ref")
        for rec in recs:
            p = Path(rec["file"])
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                fails.append(f"arm=dumps/{rec['name']}: dump file {p} missing")
                continue
            got = _read_dump(p, rec)
            t = out[rec["tensor_key"]]
            if tuple(got.shape) != tuple(rec["shape"]):
                fails.append(f"arm=dumps/{rec['name']}: dump shape {tuple(got.shape)} != "
                             f"manifest shape {tuple(rec['shape'])}")
            elif tuple(t.shape) != tuple(rec["shape"]):
                fails.append(f"arm=dumps/{rec['name']}: recomputed shape {tuple(t.shape)} != "
                             f"manifest shape {tuple(rec['shape'])} — the fixture drifted")
            elif str(got.dtype) != rec["dtype"]:
                fails.append(f"arm=dumps/{rec['name']}: dump dtype {got.dtype} != manifest "
                             f"{rec['dtype']}")
            elif not torch.equal(got, t):
                m2 = R.metrics(got.to(torch.float64), t.to(torch.float64), name=rec["name"])
                fails.append(f"arm=dumps/{rec['name']}: dump not bit-exact with the oracle "
                             f"(max_abs={m2['max_abs']:.3e}, ndiff={m2['ndiff']}/{m2['n']})")
            else:
                checked += 1
        del w
    print(f"\n## arm=dumps: {checked}/{total} pinned ref tensors reproduced bit-exactly "
          "from the oracle")
    if checked != total:
        fails.append(f"arm=dumps: checked {checked} of {total} ref-arm tensors")
    if total == 0:
        fails.append("arm=dumps: the manifest carries no ref-arm tensors; vacuous section")
    return fails


def _read_dump(p: Path, rec: dict) -> torch.Tensor:
    """Read a dump back by shape and dtype from the manifest record — the dtype is taken
    from the record and cross-checked against the file size, never assumed (pitfalls/30)."""
    raw = p.read_bytes()
    dt = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32,
          "torch.int64": torch.int64}[rec["dtype"]]
    itemsize = {torch.bfloat16: 2, torch.float32: 4, torch.int64: 8}[dt]
    if len(raw) != rec["elements"] * itemsize:
        raise AssertionError(f"{p.name}: {len(raw)} B cannot hold {rec['elements']} "
                             f"{rec['dtype']} ({rec['elements'] * itemsize} B)")
    if dt is torch.bfloat16:
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
    else:
        t = torch.frombuffer(bytearray(raw), dtype=dt)
    return t.reshape(rec["shape"])


if __name__ == "__main__":
    sys.exit(main())
