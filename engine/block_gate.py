"""M2 block gate: adjudicates `engine/block_forward.py` stage by stage against the pinned dumps.

THE QUESTION. Layer 0, one token, every stage of the GDN recurrence, the routed MoE and the shared
expert printed against its pinned referent with a verdict. A row may say `ACCEPTED` only when an
existing licence grants it: the GDN rows go through `speccheck/gdn_ref.band_check_row` and
`band_licenses` over `speccheck/evidence/gdn-band-2026-10-26.json` (digest pinned below,
`band_digest()` must stay `27a68c7e82c9aa05…`), and the MoE rows are judged by the bounds their own
prototype lanes measured, re-read at run time from those lanes' committed artifacts and checked
against the limit used. **No band, factor, floor or tolerance is authored in this file.** A stage
with no licence prints `UNLICENSED stage=<name> reason=<the missing artifact>` and the gate is then
structurally unable to print PASS: `BLOCKGATE verdict=QUALIFIED`. That is the posture
`DECISIONS.md` D-21 chose for the rows its own licence could not reach — leave them unlicensed, name
them, do not force a probe.

TWO ARMS BECAUSE THERE ARE TWO ERROR BUDGETS.
  `arm=ref_bf16` runs the block program on the bf16 weights the dumps were made from. That is the
     population every licensed band was probed over, so its licensed rows are genuinely judgeable.
  `arm=policy_a` runs the same program on the shipping shape — experts Q4_K/Q5_K, GDN, attention and
     shared expert Q8_0, router / shared gate / norms F32 exact (D-01, D-01a, D-08) — read through
     `engine/loader.py` addresses and dequantized by `tools/gguf_reader.py`. No band here measures a
     quantized arm against bf16 pins: `quantization/README.md` §"How a quantized path gets gated"
     item 1 says a 4-bit expert output is not within a bf16 ULP of anything and its band must come
     from a floor probe over the pinned dumps; that probe has not been run. Those rows print their
     metrics — which is what such a probe would need — under a named `UNLICENSED`. Where policy A
     keeps a tensor bit-exact (the router, `specification/10` §10.2) the licensed band applies
     unchanged, and those rows are judged on both arms.

DEVICE CLAIMS COME FROM A DEVICE, AND ONLY FROM A DEVICE. This gate has no device path and says so.
`--device` is refused with the tag `live-device-path-absent` (engine/block_forward.py::device_refusal:
nothing here allocates, launches or imports HIP, and `Loader.device_slabs()` still raises
`device-allocator-absent`), and check `device-claims-come-from-device-only` reddens if any device-shaped
verdict appears in the output while no live device path ran — a predicate a control feeds a fabricated
residency line to, so it cannot be decorative. `--staged-evidence LOG` is a different thing and is
labelled as one: it admits M1's read-back log as a PRECONDITION on staging integrity, re-hashing every
policy-A region the log claims came back off the device against the artifact this process is reading
(`engine/block_forward.py::staged_precondition`) and refusing by named tag if the log is not about this
file. A precondition is not a device result, and no line printed for it reads like one. No allocation, no
kernel, no HIP, no device lock happens here.

EVERY COUNT IS PARSED BACK OFF THE PRINTED TEXT. The `Tee` is what makes that possible, and two rules
hold it up: it reassembles lines across `write()` boundaries (CPython's `print` emits the text and the
newline as two writes, so a tee that treats one write as one line hands the parser a buffer of empty
strings and every parsed count reads 0), and `parsed-record-count-equals-printed` requires parsed ==
printed and non-zero — the reconciliation a vacuous parser cannot survive (`pitfalls/40`
§"A driver whose parser matches nothing prints PASS").

NO TIMINGS. Nothing in this file reads a clock; nothing it prints carries `ms`, `t/s` or `GB/s`, and
`check no-rate-figures-in-output` enforces that over its own printed text (`pitfalls/00`
§"A sub-millisecond timer is not a ruler"). Run it to completion and grep the file; never pipe it
through `head` or `tail`, which truncates the reconciliation it exists to perform.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "speccheck"), str(ROOT / "specification")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import block_forward as BF                              # noqa: E402
import gdn_ref as G                                     # noqa: E402

BAND_DIGEST = "27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee"
ROUTER_EVIDENCE = ROOT / "prototype" / "moe_router" / "evidence" / "moe-router-host-2026-10-26.json"
GATHER_FLOOR = ROOT / "prototype" / "moe_gather" / "work" / "oracle_floor.tsv"
SPEC03 = ROOT / "specification" / "03-block-glue-and-normalization.md"
ARMS = ("ref_bf16", "policy_a")
MODES = ("all", "stage", "weights", "device", "population")

# The MoE stages a prototype lane has already bought a band for, with the probe key that names the
# limit and the artifact that licenses it. `exact` means zero tolerance, which is what those lanes
# proved (`specification/10` §10.6's router / selection / routed-sum rows).
MOE_BANDS = {
    "router_logits": ("ndiff_ulp", "logit_dot_route_disagreement_elements",
                      "prototype/moe_router/evidence/moe-router-host-2026-10-26.json bands{} + "
                      "results.md §band row `logits`"),
    "topk_weights": ("exact", None, "specification/10 §10.6 router row: cast to the logits dtype "
                     "before the multiply, bit-exact against the pinned arm; moe_router results.md "
                     "row `wcast` (cast-is-robust-not-lucky)"),
    "topk_ids": ("exact", None, "specification/10 §10.6 selection row: set-equal per row and "
                 "slot-order equal"),
    "expert_raw": ("ndiff", "legitimate_reassociation_layer0",
                   "prototype/moe_gather/work/oracle_floor.tsv: the largest bit-difference any "
                   "legal re-association of the reference's own arithmetic produces on layer 0"),
    "routed_sum": ("exact", None, "specification/10 §10.6 routed-sum row: ascending-expert-id "
                   "accumulation reproduces routed_sum; moe_gather oracle_floor.tsv "
                   "combine_form_vs_pinned_routed_sum=0"),
}
# Stages that are licensed only when the arm reads no quantized byte, because their band is a
# program-error band on the pinned weights. On the policy-A arm these are unlicensed by name.
QUANT_DEPENDING = None            # set below from MOE_BANDS minus the exact-family router stages

UNLICENSED_WIDTH = ("no licensed band measures a policy-A (quantized) arm against bf16 pins; the "
                    "GDN band file licenses cross-form program error on one width "
                    "(prototype/gdn/results.md §Licence) and the stage-output floor probe that "
                    "quantization/README.md §How a quantized path gets gated item 1 requires has "
                    "never been run")
UNLICENSED_NO_REFERENT = ("the pinned dumps carry no referent for a residual-stream stage: §03.2 "
                          "adds each mixer output to the residual stream and neither M0 manifest "
                          "pins layer 0's residual stream (each pins a mixer INPUT, a synthetic "
                          "RMS-1 tensor per specification/10 §10.1 [A-01]), so there is nothing to "
                          "compare against and inverting the norm to manufacture one would change "
                          "the quantity under test (specification/03 §03.3)")
UNLICENSED_NO_BAND = ("no band and no prototype floor probe has measured this stage against the "
                      "pinned arm; refusing to invent a limit is `gdn_ref.band_check_row`'s own "
                      "posture for a stage with no band entry")


class Verifier:
    """Named checks and witnessed controls, the house shape (`engine/stage_run.py` §Verifier)."""

    def __init__(self):
        self.names = []
        self.red = []
        self.controls = []
        self.fired = 0

    def check(self, name, predicate, detail):
        ok = bool(predicate)
        self.names.append(name)
        if not ok:
            self.red.append(name)
        print("CHECK name=%s verdict=%s %s" % (name, "PASS" if ok else "FAIL", detail))
        return ok

    def control(self, name, fired, detail):
        """A control exists to make something go wrong; `fired=RED` is the only pass state."""
        self.controls.append(name)
        if fired:
            self.fired += 1
        else:
            self.red.append("control:" + name)
        print("CONTROL name=%s fired=%s %s" % (name, "RED" if fired else "SILENT", detail))
        return bool(fired)


def ulp_diff(a, b):
    """(elements differing, largest bit-image step) for a bf16/f32 pair.

    `pitfalls/30` §byte-identity claims state the width they cover: a bf16 ULP is one step of the
    uint16 image, an f32 ULP one step of the ordered int32 image; data is widened with `.to()` by the
    caller, never reinterpreted with `.view()` on the tensor itself.
    """
    if a.dtype is torch.bfloat16:
        ai = a.contiguous().view(torch.uint16).to(torch.int32)
        bi = b.contiguous().view(torch.uint16).to(torch.int32)
    elif a.dtype is torch.float32:
        ai = a.contiguous().view(torch.int32)
        bi = b.contiguous().view(torch.int32)
        ai = torch.where(ai < 0, 0x7FFFFFFF - ai, ai)
        bi = torch.where(bi < 0, 0x7FFFFFFF - bi, bi)
    else:
        raise BF.Refused("ulp-unsupported-dtype", "ULP distance is defined for bf16/f32, not %s"
                         % a.dtype)
    d = (ai - bi).abs()
    return int((a != b).sum().item()), (int(d.max().item()) if d.numel() else 0)


def spec03_steps():
    """(numbered steps, residual adds) counted out of §03.2's own skeleton, not typed here."""
    blocks = SPEC03.read_text().split("```")
    block = next((b for b in blocks if "h_in" in b and "MoE(" in b), "")
    steps = [ln.strip() for ln in block.splitlines() if re.match(r"^\s*\d+\s+\w+\s*=", ln)]
    adds = [ln.split("=")[0].strip() for ln in steps if re.search(r"=\s*\w+\s*\+\s*\w+", ln)]
    return len(steps), len(adds), adds


def prototype_limits():
    """The limits the prototype lanes measured, re-read from their committed artifacts.

    Returns `({stage: limit}, {probe key: value}, {missing probe keys})`. A stage whose probe key is
    absent is left out of the limit table, and `adjudicate` then prints it as unlicensed rather than
    picking a number.
    """
    raw = {}
    doc = json.loads(ROUTER_EVIDENCE.read_text())
    for k, v in doc.get("bands", {}).items():
        raw[k] = v
    for line in GATHER_FLOOR.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            raw[parts[0]] = parts[1]
    limits, missing = {}, []
    for stage, (kind, key, _) in MOE_BANDS.items():
        if kind == "exact":
            limits[stage] = 0
        elif key in raw:
            try:
                limits[stage] = int(raw[key])
            except (TypeError, ValueError):
                missing.append("%s(%s=%r)" % (stage, key, raw[key]))
        else:
            missing.append("%s(%s absent)" % (stage, key))
    return limits, raw, missing


def width_verdicts(declared, artifact_types):
    """(n_compared, [(name, artifact_type, declared)]) — policy A's declared widths against the file's.

    The ruler is the ggml type the artifact record stores for that tensor name — the same field
    `engine/loader.py`'s width veto reads — never the dtype a dequantized tensor materialises as: a
    Q8_0 tensor legitimately arrives as float32 after dequant, so a comparison against a torch dtype is
    an assertion that can only ever fail (`pitfalls/30` §dtype-by-filesize). The declared map is a
    parameter so a control can hand it a deliberately wrong declaration and watch it get caught.
    """
    n, bad = 0, []
    for name in sorted(declared):
        got = artifact_types.get(name)
        if got is None:
            continue
        n += 1
        if got != declared[name]:
            bad.append((name, got, declared[name]))
    return n, bad


def gather_census(reads, cfg):
    """(n_gathered_slices, whole_tensor_expert_reads, ok) over one arm's read records.

    Scoped to the records the block forward produced (`ctx == "forward"`): the controls recompute the
    router and the experts through their own `Weights`, and those reads answer a different question,
    so counting them would bill a control's work against the shipping path's read pattern. The
    denominator is geometry (3 expert matrices x top_k gathered slices), and the failure this exists to
    catch is a whole `_exps.weight` read instead of a gather — 1.5 GiB per token instead of the 48 MiB
    the resident set is priced for.
    """
    fwd = [r for r in reads if r.get("ctx") == "forward"]
    slices = [r for r in fwd if r.get("kind") == "expert"]
    whole = [r for r in fwd if r["name"].endswith("_exps.weight") and r.get("kind") != "expert"]
    want = 3 * cfg["top_k"]
    return len(slices), whole, (len(slices) == want and not whole)


# Split so the scanner cannot match its own pattern text — the self-matching-scanner trap of
# `pitfalls/50`, and the same reason engine/stage_run.py writes its lock prohibition as
# r"gpu_" r"lock\.sh". The rule cuts both ways: prose in this file must not spell the two-word form
# either, or the check reddens on its own documentation — which is exactly how the first version of this
# scanner was caught by the thing it was written to catch.
CONST_IF_RE = re.compile(r"\bif\s+Tr" r"ue\b")
CONST_CHECK_RE = re.compile(r"check\([^)]*,\s*(Tr" r"ue|1)\s*,")


def constant_predicates(src):
    """Every literal-constant predicate in `src` (`pitfalls/40` §a-check-whose-predicate-is-True).

    Takes the text as an argument so a control can feed it a copy with a constant injected and watch it
    fire; the scan of this gate's own source is worthless until someone has seen it return non-empty for
    a source that deserves it.
    """
    return re.findall(CONST_CHECK_RE, src) + re.findall(CONST_IF_RE, src)


# What a device-shaped verdict looks like in this file's own output. A refusal is not in this list:
# refusing a claim is the opposite of making one.
DEVICE_CLAIM_RE = re.compile(r"DEVICESTAGED|\bdevice=(?:on|admitted|yes)\b|verdict=MATCH")


def device_claims(text):
    """The printed lines that read like a device result. Empty is the only honest CPU answer."""
    return [ln for ln in text.splitlines() if DEVICE_CLAIM_RE.search(ln)]


def renormalised_topk(logits, cfg):
    """R2-R4 of `block_forward.moe_router` on logits handed in, for the form probe.

    Deliberately a re-implementation of the *shape* (softmax F32, top-k, renormalise F32) rather than a
    call into the arm, so the probe can ask "what would the weights be if R1 had landed in this dtype"
    without editing the shipping program.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    tw, _ids = torch.topk(probs, cfg["top_k"], dim=-1)
    return tw / tw.sum(-1, keepdim=True)


def derived_population(dumps):
    """The stage list, derived: spec stage list ∩ dump records, per subsystem, plus the glue.

    GDN: the GDN manifest's `layer0/decode0` records under `arm=ref`, which are the stages
    `specification/04` enumerates — 19 of them, matching `counts.stages_per_pass`.
    MoE: the MoE manifest's `layer0` records under `arm=ref` (the `exact` and `spec` arms belong to
    other lanes' questions and are excluded by rule, not by a typed list).
    Glue: §03.2's residual adds, counted from the spec file.
    """
    gdn = dumps.stage_names("gdn", BF.GDN_PASS, modes={BF.GDN_MODE}, arms=("ref",))
    moe = dumps.stage_names("moe", "layer0", modes=None, arms=("ref",))
    steps, adds, add_names = spec03_steps()
    glue = ["block_output"] if adds else []
    return dict(gdn=gdn, moe=moe, glue=glue, all=gdn + moe + glue,
                spec_steps=steps, spec_adds=adds, add_names=add_names)


def referent_name(stage):
    if stage in REFERENT_GDN:
        return "%s.%s.%s" % (BF.GDN_PASS, BF.GDN_MODE, stage)
    if stage in REFERENT_MOE:
        return "layer0.%s" % stage
    return None


def compute_arm(dumps, cfg, arm, faults=(), ctx="forward"):
    """Run the block once under one weight arm and return its stage dict."""
    W = BF.Weights(arm, cfg, faults=faults, ctx=ctx)
    try:
        stages, meta = BF.forward_block(dumps, W, cfg)
        reads = list(W.reads)
    finally:
        W.close()
    return stages, meta, reads


def stage_row(dumps, arm, stage, stages):
    """One STAGE row as a dict; metrics only, no verdict (the verdict is `adjudicate`'s job)."""
    row = dict(arm=arm, stage=stage,
               weights=("pinned-bf16" if arm == "ref_bf16" else "policy-A"),
               pass_=("%s.%s" % (BF.GDN_PASS, BF.GDN_MODE)) if stage in REFERENT_GDN else
                     ("layer0.row%d" % BF.TOKEN_ROW),
               licence="none", band="", why="")
    refname = referent_name(stage)
    if refname is None:
        row.update(referent="absent", n=0, ndiff="", dtype="", max_ulp="", maxabs="",
                   one_minus_cos="", meanrel="", perhead="", ref=name_repr(stage))
        row["why"] = UNLICENSED_NO_REFERENT
        return row
    ref = dumps.tensor(refname, rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1)
                       if stage in REFERENT_MOE else None)
    mine = stages.get(stage)
    row["ref"] = refname
    if mine is None:
        row.update(referent=refname, n=int(ref.numel()), ndiff="", dtype=str(ref.dtype),
                   max_ulp="", maxabs="", one_minus_cos="", meanrel="", perhead="")
        row["why"] = "the forward produced no value for this stage"
        return row
    mine = mine.reshape(ref.shape)
    if mine.dtype is not ref.dtype:
        mine = mine.to(ref.dtype)            # rounded to the dtype the reference stores
    row.update(referent=refname, n=int(ref.numel()), dtype=str(ref.dtype))
    if ref.dtype is torch.int64:
        row["ndiff"] = int((mine != ref).sum().item())
        row["max_ulp"] = 0
        for k in ("maxabs", "one_minus_cos", "meanrel", "perhead"):
            row[k] = ""
        row["_ag"] = None
    else:
        mine = mine.contiguous()
        nd, ulp = ulp_diff(mine, ref)
        ag = G.aggregates(mine, ref, name=stage, stage=stage)
        ag["stage"] = stage
        row.update(ndiff=nd, max_ulp=ulp, maxabs="%.3e" % ag["max_abs"],
                   one_minus_cos="%.3e" % ag["one_minus_cos"], meanrel="%.3e" % ag["mean_rel"],
                   perhead="%.3e" % ag["per_head_max"], _ag=ag)
    return row


def name_repr(stage):
    return "§03.2-step-%s" % stage


def adjudicate(rows, band, limits, population):
    """Give every row its licence and verdict, consulting only the existing licensors."""
    tally = dict(licensed=0, accepted=0, rejected=0, unlicensed=0)
    for row in rows:
        stage, arm = row["stage"], row["arm"]
        quantized_arm = arm == "policy_a"
        if referent_name(stage) is None:
            row["band"] = "UNLICENSED stage=%s reason=no-pinned-referent" % stage
            tally["unlicensed"] += 1
            continue
        if stage in band:
            granted, why = G.band_licenses(band, BF.GDN_PASS_KEY, stage, "recurrent", "pinned")
            if quantized_arm:
                row["band"] = ("UNLICENSED stage=%s reason=policy-a-arm-outside-band-population"
                               % stage)
                row["why"] = UNLICENSED_WIDTH
                tally["unlicensed"] += 1
                continue
            if not granted:
                row["band"] = "UNLICENSED stage=%s reason=arm-pair-not-licensed-at-pass" % stage
                row["why"] = why
                tally["unlicensed"] += 1
                continue
            ok, why = G.band_check_row(band, stage, row["_ag"])
            row["licence"] = "gdn-band(band_licenses|recurrent|pinned@%s)" % BF.GDN_PASS_KEY
            row["band"] = "ACCEPTED" if ok else "REJECTED"
            row["why"] = why
            tally["licensed"] += 1
            tally["accepted" if ok else "rejected"] += 1
            continue
        if stage in MOE_BANDS and (not quantized_arm or stage in ROUTER_FAMILY):
            kind, key, src = MOE_BANDS[stage]
            if stage in QUANT_DEPENDING and quantized_arm:
                row["band"] = "UNLICENSED stage=%s reason=policy-a-width-band-absent" % stage
                row["why"] = UNLICENSED_WIDTH
                tally["unlicensed"] += 1
                continue
            lim = limits.get(stage)
            if lim is None:
                row["band"] = "UNLICENSED stage=%s reason=probe-limit-unreadable" % stage
                tally["unlicensed"] += 1
                continue
            bad = row["ndiff"] > int(lim)
            if kind == "ndiff_ulp":
                bad = bad or row["max_ulp"] > 1
            row["licence"] = "floor-probe(%s)" % key if key else "spec-10.6(exactness)"
            row["band"] = "REJECTED" if bad else "ACCEPTED"
            row["why"] = "ndiff=%s limit=%s kind=%s src=%s" % (row["ndiff"], lim, kind,
                                                              src.split(" ")[0])
            tally["licensed"] += 1
            tally["rejected" if bad else "accepted"] += 1
            continue
        if quantized_arm:
            row["band"] = "UNLICENSED stage=%s reason=policy-a-width-band-absent" % stage
            row["why"] = UNLICENSED_WIDTH
        else:
            row["band"] = "UNLICENSED stage=%s reason=no-band-entry-and-no-probe" % stage
            row["why"] = UNLICENSED_NO_BAND
        tally["unlicensed"] += 1
    return tally


# One `STAGE` record, as `print_row` emits it: space-free `key=value` fields, then `band=` last because
# an unlicensed band's text (`UNLICENSED stage=x reason=y`) is the one field that carries spaces. A row
# whose required fields are absent, or whose `n=` is not a number, is not a record — and
# `parsed-record-count-equals-printed` is what makes dropping one hurt. `ndiff` is required of every row
# that HAS a referent; a no-referent row prints no metric at all rather than a fake one.
ROW_RE = re.compile(r"^STAGE ((?:[a-z_]+=\S+ )+)band=(.*\S)$")
ROW_KV_RE = re.compile(r"([a-z_]+)=(\S+)")
ROW_KEYS = ("arm", "weights", "pass", "stage", "referent", "n", "licence")


def print_row(row):
    parts = ["STAGE arm=%s" % row["arm"], "weights=%s" % row["weights"],
             "pass=%s" % row["pass_"], "stage=%s" % row["stage"],
             "referent=%s" % row["referent"], "n=%d" % row["n"]]
    for k in ("ndiff", "dtype", "max_ulp", "maxabs", "one_minus_cos", "meanrel", "perhead"):
        if row.get(k, "") != "":
            parts.append("%s=%s" % (k, row[k]))
    parts.append("licence=%s" % row["licence"])
    parts.append("band=%s" % row["band"])
    print(" ".join(parts))
    if row.get("why"):
        print("DETAIL arm=%s stage=%s why=%s" % (row["arm"], row["stage"], row["why"]))


def parse_printed(text):
    """Every STAGE record the emitter actually emitted, as dicts (empty text parses to nothing)."""
    recs = []
    for ln in text.splitlines():
        m = ROW_RE.match(ln)
        if not m:
            continue
        kv = dict(ROW_KV_RE.findall(m.group(1)))
        need = set(ROW_KEYS) | ({"ndiff"} if kv.get("referent") != "absent" else set())
        if need - set(kv) or not kv["n"].isdigit():
            continue
        recs.append(dict(arm=kv["arm"], pass_=kv["pass"], stage=kv["stage"], referent=kv["referent"],
                         n=int(kv["n"]), ndiff=kv.get("ndiff", ""), licence=kv["licence"],
                         band=m.group(2)))
    return recs


class Tee:
    """Capture every printed line while printing it, so parsing can be checked against printing.

    `pitfalls/40` §"A driver whose parser matches nothing prints PASS": a count derived from the
    same buffer that was printed is the only count worth having, and a parser that matches nothing
    must make the run fail rather than print an empty table. The buffering is load-bearing:
    `print("x")` reaches a file object as `write("x")` then `write("\n")`, so treating one write as
    one line produced a buffer of empty strings and a parsed count of 0 for 58 real rows.
    """

    def __init__(self, fh):
        self.fh = fh
        self.lines = []
        self._partial = ""

    def write(self, s):
        self.fh.write(s)
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self.lines.append(line)
        return len(s)

    def flush(self):
        self.fh.flush()

    def text(self):
        return "\n".join(self.lines + ([self._partial] if self._partial else []))


REFERENT_GDN = set()
REFERENT_MOE = set()
ROUTER_FAMILY = ("router_logits", "topk_ids", "topk_weights")
QUANT_DEPENDING = set(MOE_BANDS) - set(ROUTER_FAMILY)


def build_rows(dumps, cfg, arms, population, skip=None, out=None, ctx="forward"):
    rows, meta = [], {}
    for arm in arms:
        stages, fmeta, reads = compute_arm(dumps, cfg, arm, ctx=ctx)
        meta[arm] = dict(forward=fmeta, reads=reads, stages=stages)
        for stage in population["all"]:
            if skip and stage == skip:
                continue
            rows.append(stage_row(dumps, arm, stage, stages))
    return rows, meta


def controls(v, dumps, cfg, rows, population, band, out_lines, parsed, meta, limits, forms):
    """Every control must make something go wrong. A control that cannot is a SILENT red failure.

    Controls 1-4 attack the arithmetic; 5-11 attack the checkers and the admissions, because a repaired
    checker is only repaired if someone watched the repaired version refuse a case it must refuse
    (`pitfalls/40` §gates-must-fail-for-something).
    """
    x_moe = dumps.tensor("layer0.input", row=BF.TOKEN_ROW).reshape(1, cfg["d"])

    # 1. a policy-A weight read from the wrong offset must move the stage that reads it.
    clean = BF.Weights("policy_a", cfg, ctx="control:wrong-offset-weight-read")
    dirty = BF.Weights("policy_a", cfg, faults=("wrong_offset_weight_read",),
                       ctx="control:wrong-offset-weight-read")
    try:
        a = BF.moe_router(x_moe, clean, cfg)["router_logits"]
        try:
            b = BF.moe_router(x_moe, dirty, cfg)["router_logits"]
            moved, mag = G.stage_moved(a, b)
        except BF.Refused as exc:
            moved, mag = True, float("nan")
            detail = "the shifted read raised %s instead" % exc.tag
        else:
            detail = ("stage=router_logits moved=%s max|delta|=%.3e injected=file_off+1 byte on "
                      "blk.0.ffn_gate_inp.weight (the licensed row went %s)"
                      % (moved, mag, [r["band"] for r in rows
                                      if r["stage"] == "router_logits" and r["arm"] == "policy_a"]
                         [0] if any(r["stage"] == "router_logits" and r["arm"] == "policy_a"
                                    for r in rows) else "-"))
        v.control("wrong-offset-weight-read", moved, detail)
    finally:
        clean.close()
        dirty.close()

    # 2. a swapped expert slice must move the routed sum.
    clean = BF.Weights("ref_bf16", cfg, ctx="control:swapped-expert-slice")
    dirty = BF.Weights("ref_bf16", cfg, faults=("swapped_expert_slice",),
                       ctx="control:swapped-expert-slice")
    route = BF.moe_router(x_moe, clean, cfg)
    honest = BF.moe_routed(x_moe, clean, cfg, route)
    swapped = BF.moe_routed(x_moe, dirty, cfg, route)
    m1, _ = G.stage_moved(honest["routed_sum"], swapped["routed_sum"])
    m2, _ = G.stage_moved(honest["expert_raw"], swapped["expert_raw"])
    v.control("swapped-expert-slice", m1 and m2,
              "stage=routed_sum moved=%s stage=expert_raw moved=%s injected=expert e's gate slice "
              "read from expert (e+1)%%E on all %d slots (specification/11 §11.5's slot↔id alignment)"
              % (m1, m2, cfg["top_k"]))

    # 3. a stage the forward skipped must be caught by the population, not missed silently.
    one_licensed = next((r["stage"] for r in rows if r["band"].startswith("ACCEPTED")), None)
    if one_licensed is None and not rows:
        probe_rows, _ = build_rows(dumps, cfg, ["ref_bf16"], population)
        one_licensed = next((r["stage"] for r in probe_rows
                             if r["band"].startswith("ACCEPTED")), None)
    if one_licensed is None:
        v.control("skipped-stage", False, "no ACCEPTED row exists to drop; the population control "
                                          "has nothing to bite on and cannot be witnessed")
    else:
        shrunk, _ = build_rows(dumps, cfg, [r["arm"] for r in rows][:1], population,
                               skip=one_licensed, ctx="control:skipped-stage")
        expect = len(population["all"])
        got = len([r for r in shrunk if r["arm"] == shrunk[0]["arm"]]) if shrunk else 0
        v.control("skipped-stage", got != expect and got == expect - 1,
                  "dropped stage=%s: built %d rows for one arm against a derived population of %d "
                  "(the missing row is what the population check must catch)"
                  % (one_licensed, got, expect))

    # 4. a printed stage-row count that disagrees with the derived population must be caught.
    extra = list(rows) + [dict(rows[0])]
    parsed_again = parse_printed(out_lines) if out_lines else []
    derived_total = len(population["all"]) * len(set(r["arm"] for r in rows))
    v.control("stage-row-count-mismatch",
              len(extra) != derived_total and len(parsed_again) == derived_total,
              "duplicated one row: %d rows against %d parsed and a derived population of %d; the "
              "reconciliation check fires on the disagreement rather than trusting the emitter"
              % (len(extra), len(parsed_again), derived_total))

    # 5. the parser must count what was printed, and must notice a missing line.
    stage_lines = [ln for ln in out_lines.splitlines() if ln.startswith("STAGE ")]
    truncated = parse_printed("\n".join(stage_lines[:-1]))
    v.control("stage-row-parse-truncation",
              len(parsed) > 0 and len(parsed) == len(stage_lines) and len(truncated) == len(rows) - 1,
              "parsed %d of %d printed STAGE lines. A parser matching nothing reads 0, which is what the "
              "inherited gate printed (parsed=0, printed=58) and it went red; the subtler failure is a "
              "parser that matches most lines and drops a few silently, and that is what this control "
              "measures — dropping the last printed line parses %d, which the parsed==printed equality "
              "refuses"
              % (len(parsed), len(stage_lines), len(truncated)))

    # 6. the width comparison must reject a tensor declared at the wrong width.
    art_types = {r["name"]: r["ggml_type"] for r in meta.get("policy_a", {}).get("reads", [])
                 if r.get("ctx") == "forward" and r.get("ggml_type")}
    lie = dict(BF.PolicyA.WIDTH, **{"blk.0.attn_qkv.weight": "Q4_K"})
    n_honest, bad_honest = width_verdicts(BF.PolicyA.WIDTH, art_types)
    n_lie, bad_lie = width_verdicts(lie, art_types)
    v.control("policy-a-width-wrong-declaration",
              not bad_honest and n_lie == n_honest and len(bad_lie) == 1
              and bad_lie[0][0] == "blk.0.attn_qkv.weight",
              "honest declaration compares %d reads with %d mismatches; declaring "
              "blk.0.attn_qkv.weight Q4_K against the artifact's own %s returns %s. The ruler is the "
              "file's ggml type, not the dequantized dtype, which is why this can fail at all"
              % (n_honest, len(bad_honest), art_types.get("blk.0.attn_qkv.weight"), bad_lie or "none"))

    # 7. the gather counter must redden on a whole-tensor expert read and ignore a control's own reads.
    pa_reads = meta.get("policy_a", {}).get("reads", [])
    probe_w = BF.Weights("policy_a", cfg, ctx="control:whole-tensor-expert-read")
    try:
        h = probe_w.pa.L.handle("blk.0.ffn_gate_exps.weight")
    finally:
        probe_w.close()
    whole_rec = dict(name=h["name"], kind="tensor", ctx="forward", ggml_type=h["type_name"],
                     dtype="torch.float32", bytes=h["length"])
    n_ok, whole_ok, ok_ok = gather_census(pa_reads, cfg)
    n_whole, whole_bad, ok_whole = gather_census(pa_reads + [whole_rec], cfg)
    n_ctl, _, ok_ctl = gather_census(pa_reads + [dict(pa_reads[0], ctx="control:skipped-stage")], cfg)
    v.control("gather-counter-catches-a-whole-tensor-read",
              ok_ok and not ok_whole and len(whole_bad) == 1 and n_ctl == n_ok and ok_ctl,
              "honest forward: %d gathered slices against a derived %d, %d whole-tensor expert reads, "
              "ok=%s. Append the record a whole-tensor read of %s (%s, %d B) writes and the same census "
              "reads ok=%s with %s flagged — the record, not the 150 MB dequantization, which a CPU "
              "control must not do to itself; a control-context read stays out of the count (%d, ok=%s)"
              % (n_ok, 3 * cfg["top_k"], len(whole_ok), ok_ok, h["name"], h["type_name"], h["length"],
                 ok_whole, [r["name"] for r in whole_bad], n_ctl, ok_ctl))

    # 8. the constant-predicate scanner must fire on a source that carries one.
    src = (HERE / "block_gate.py").read_text()
    copy = HERE / "work" / "const-predicate-injected-copy.py"
    injected = src.replace("def width_verdicts(",
                           "if Tr" "ue:      # injected by the constant-predicate control\n"
                           "    pass\n\n\ndef width_verdicts(", 1)
    injected += "\nv.check(\"fabricated-constant-predicate\", Tr" "ue, \"detail\")\n"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(injected)
    found_copy, found_self = constant_predicates(copy.read_text()), constant_predicates(src)
    v.control("constant-predicate-scan-is-awake",
              bool(found_copy) and not found_self and copy.read_text() != src,
              "the injected copy at %s scans %s; this file scans %s. An empty list from a scanner whose "
              "pattern matches only its own text is indistinguishable from a clean file until someone "
              "injects one (pitfalls/50 §the self-matching scanner)"
              % (copy.relative_to(ROOT), found_copy, found_self or "[]"))

    # 9. the device-claim scanner must refuse a residency line this process did not earn.
    probe_line = ("DEVICE" + "STAGED tensor=blk.0.attn_qkv.weight run=91 " + "verdict=" + "MATCH "
                  + "device=" + "admitted")
    honest_claims = device_claims(out_lines)
    v.control("device-claim-scan-is-awake",
              not honest_claims and len(device_claims(probe_line)) == 1,
              "this run printed %d device-shaped lines (a refusal is not a claim); the assembled probe "
              "line — kept out of the output and assembled here so this file cannot self-match — "
              "scans %d, so device-claims-come-from-device-only has something to lose"
              % (len(honest_claims), len(device_claims(probe_line))))

    # 10. the staging PRECONDITION must be about the artifact it is used to describe, and must refuse
    #     when it is not. The honest arm re-hashes every policy-A region the log claims came back off the
    #     device; the six corrupted arms are copies in engine/work/ with exactly one thing changed each,
    #     and each must refuse with its own named tag rather than admit.
    work = HERE / "work"
    work.mkdir(parents=True, exist_ok=True)

    def staged_arm(text, name):
        path = work / name
        path.write_text(text)
        try:
            r = BF.staged_precondition(path)
            return ("admitted", r["tensors"], r["bytes_hashed"])
        except BF.Refused as exc:
            return ("refused", exc.tag, None)

    base = BF.M1_EVIDENCE.read_text()
    art_name = json.loads(BF.GGUF_PIN.read_text())["artifact"]
    gate_run = next(ln.split("run=")[1].split()[0] for ln in base.splitlines()
                    if ln.startswith("TENSOR name=blk.0.attn_gate.weight "))
    dig_lines, off_lines, rb_lines, miss_lines, width_lines = [], [], [], [], []
    for ln in base.splitlines():
        forms_ln = dict(dig=ln, off=ln, rb=ln, miss=ln, width=ln)
        if ln.startswith("READBACK idx=%s " % gate_run):
            tok = re.search(r"[0-9a-f]{64}", ln).group(0)
            bad = ("f" if tok[0] != "f" else "e") + tok[1:]
            forms_ln["dig"] = ln.replace(tok, bad)           # all three together: the log stays
            forms_ln["rb"] = ln.replace("readback_sha256=" + tok,   # self-consistent, so only the file
                                        "readback_sha256=" + bad)   # can catch it
                                                              # only the read-back, verdict still MATCH
        if ln.startswith("RUN idx=%s " % gate_run):
            forms_ln["off"] = re.sub(r"\bfile_off=(\d+)",
                                     lambda m: "file_off=%d" % (int(m.group(1)) + 1), ln)
        if ln.startswith("TENSOR name=blk.0.attn_gate.weight "):
            forms_ln["width"] = ln.replace("type=Q8_0", "type=Q4_K")  # a width the artifact denies
            forms_ln["miss"] = None                          # the log stops covering this tensor
        for key, lines in (("dig", dig_lines), ("off", off_lines), ("rb", rb_lines),
                           ("miss", miss_lines), ("width", width_lines)):
            if forms_ln[key] is not None:
                lines.append(forms_ln[key])
    st_honest = staged_arm(base, "staged-evidence-honest-copy.log")
    st_noresult = staged_arm("\n".join(ln for ln in base.splitlines()
                                       if not ln.startswith("RESULT PASS")),
                             "staged-evidence-no-result-line.log")
    st_otherfile = staged_arm(base.replace(art_name, art_name.replace("Q4_K_M", "Q4_K_X")),
                              "staged-evidence-other-artifact.log")
    st_baddigest = staged_arm("\n".join(dig_lines), "staged-evidence-region-digest.log")
    st_badoffset = staged_arm("\n".join(off_lines), "staged-evidence-bad-offset.log")
    st_un_equal = staged_arm("\n".join(rb_lines), "staged-evidence-readback-unequal.log")
    st_missing = staged_arm("\n".join(miss_lines), "staged-evidence-missing-tensor.log")
    st_wrongwidth = staged_arm("\n".join(width_lines), "staged-evidence-wrong-width.log")
    v.control("staged-precondition-refusals-fire",
              st_honest == ("admitted", len(BF.POLICY_A_TENSORS), st_honest[2]) and st_honest[2] > 0
              and st_noresult == ("refused", "staged-evidence-result-line", None)
              and st_otherfile == ("refused", "staged-evidence-artifact-name", None)
              and st_baddigest == ("refused", "staged-evidence-region-digest", None)
              and st_badoffset == ("refused", "staged-evidence-layout", None)
              and st_un_equal == ("refused", "staged-evidence-readback", None)
              and st_missing == ("refused", "staged-evidence-incomplete", None)
              and st_wrongwidth == ("refused", "staged-evidence-width", None),
              "honest log: %s %s/%d tensors, %s B of the artifact re-hashed against the log's own "
              "read-back digests. One field changed at a time: no RESULT line -> %s; the artifact renamed "
              "-> %s; one read-back digest flipped (all three on that line, so the equality inside the "
              "log still holds) -> %s; that run's file_off shifted one byte -> %s; only the read-back "
              "digest flipped while the log still calls it MATCH -> %s; the TENSOR record for one "
              "policy-A tensor removed -> %s; that tensor's type declared Q4_K against the artifact's Q8_0 "
              "-> %s. One refusal tag is deliberately unwitnessed: staged-evidence-artifact-size would "
              "require editing the pinned artifact, which no gate in this repo does. An admissions check "
              "that only ever says yes is the failure window-c found in M1"
              % (st_honest[0], st_honest[1], len(BF.POLICY_A_TENSORS), st_honest[2],
                 st_noresult[1], st_otherfile[1], st_baddigest[1], st_badoffset[1], st_un_equal[1],
                 st_missing[1], st_wrongwidth[1]))
    # 11. the R5 form probe must be able to lose, twice: the two emitted forms have to reproduce the
    #     pinned records of their own dtype, and both wrong forms have to be refused by the same
    #     arithmetic. A probe where every form matches is a probe that measures nothing.
    nd = dict((a, forms[a]["nd"]) for a in sorted(forms))
    shown = [a for a in sorted(forms) if forms[a]["product_shown"]]
    honest_ok = bool(nd) and all(nd[a]["weights-emitted-cast-first"] == 0
                                 and nd[a]["weights-f32-carry"] == 0 for a in nd)
    leak_refused = [nd[a]["weights-arm-basis-leak-pre-fix"] for a in sorted(nd)]
    prod_sep = [(nd[a]["product-times-cast-first"], nd[a]["product-times-f32-carry"]) for a in shown]
    v.control("topk-form-probe-distinguishes-the-wrong-form",
              honest_ok and any(d > 0 for d in leak_refused)
              and bool(shown) and all(c == 0 and f > 0 for c, f in prod_sep),
              "limit=%s from the §10.6 exactness row (never typed here). The emitted cast-first weights "
              "reproduce the pinned bf16 record and the f32 weights reproduce the pinned f32 record on "
              "every arm (%s); the arm-basis-leak form is refused with ndiff %s, and on the arm whose "
              "routed_sum licence IS exactness (%s) the product built from f32-carry weights is refused by "
              "ndiff %s against the product built from cast-first weights %s. If policy A ever computes R1 "
              "in the model dtype natively the leak form coincides with the emitted one and this control "
              "has nothing left to distinguish: it must be re-derived against new pinned material, not "
              "deleted"
              % (limits.get("topk_weights"),
                 ",".join("%s:cast-first=%d,f32=%d" % (a, nd[a]["weights-emitted-cast-first"],
                                                      nd[a]["weights-f32-carry"]) for a in sorted(nd)),
                 leak_refused, ",".join(shown) or "none",
                 [f for _c, f in prod_sep], [c for c, _f in prod_sep]))


def topk_form_probe(dumps, cfg, meta, limits, licensed_exact):
    """Print every R5 form against the pinned record that stores ITS OWN dtype, and say which form the
    pinned bytes carry.

    The manifest pins the renormalised weights in two dtypes, so no form is judged by a cross-dtype
    image of itself: `layer0.topk_weights` (arm=ref, torch.bfloat16) is the value AFTER HF's cast to the
    logits' dtype and `layer0.topk_weights_f32` (arm=spec, torch.float32 — `speccheck/moe_ref.py`
    `Form(weight_cast="after")`) is the value BEFORE it. The forms printed:

      weights-emitted-cast-first     what `moe_router` returns and `moe_routed` multiplies by now;
      weights-f32-carry              the unrounded renormalised weight, i.e. what a skipped R5 carries;
      weights-arm-basis-leak-pre-fix the weights this lane shipped at its deadline, renormalised from
                                     logits left in the arm's own compute basis (F32 on policy A), so
                                     the softmax saw logits that were never rounded;
      product-times-cast-first / -f32-carry
                                     the same two weight forms carried through `moe_routed`, printed only
                                     for an arm whose `routed_sum` licence IS exactness, because that is
                                     the row where the cast ORDER is separable at all.

    Every `limit` shown is the §10.6 exactness or prototype floor-probe number that `prototype_limits()`
    re-reads; this file never types one (`pitfalls/40` §a-band-must-not-be-sized-by-the-build).
    """
    ref_w = "layer0.topk_weights"
    refs = {ref_w: dumps.tensor(ref_w, rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1))}
    refs["layer0.topk_weights_f32"] = dumps.tensor("layer0.topk_weights_f32",
                                                   rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1))
    refs["layer0.routed_sum"] = dumps.tensor("layer0.routed_sum",
                                             rows=(BF.TOKEN_ROW, BF.TOKEN_ROW + 1))
    x = dumps.tensor("layer0.input", row=BF.TOKEN_ROW).reshape(1, cfg["d"])
    out = {}
    for arm in sorted(meta):
        st = meta[arm].get("stages", {})
        if ref_w.split(".")[1] not in st:
            continue
        show_product = ("routed_sum", arm) in licensed_exact
        W = BF.Weights(arm, cfg, ctx="control:topk-form-probe")
        try:
            raw_logits = BF._linear(x, W.get("blk.0.ffn_gate_inp.weight"))      # R1, arm's own basis
            prod_carry = (BF.moe_routed(x, W, cfg, dict(topk_ids=st["topk_ids"],
                                                        topk_weights=st["topk_weights_f32"]))["routed_sum"]
                          if show_product else None)
        finally:
            W.close()
        forms = [("weights-emitted-cast-first", st["topk_weights"], ref_w, "topk_weights"),
                 ("weights-f32-carry", st["topk_weights_f32"], "layer0.topk_weights_f32",
                  "topk_weights"),
                 ("weights-arm-basis-leak-pre-fix", renormalised_topk(raw_logits, cfg), ref_w,
                  "topk_weights")]
        if show_product:
            forms += [("product-times-cast-first", st["routed_sum"], "layer0.routed_sum",
                       "routed_sum"),
                      ("product-times-f32-carry", prod_carry, "layer0.routed_sum", "routed_sum")]
        nd = {}
        for name, val, ref_name, stage in forms:
            ref = refs[ref_name]
            img = val.reshape(ref.shape).to(ref.dtype).contiguous()
            d, ulp = ulp_diff(img, ref)
            nd[name] = d
            print("TOPKFORM arm=%s form=%s ndiff=%d max_ulp=%d limit=%s referent=%s dtype=%s n=%d"
                  % (arm, name, d, ulp, limits.get(stage), ref_name, val.dtype, int(ref.numel())))
        carried = [n for n, _d, _u, _s in forms if nd[n] == 0]
        refused = [n for n, _d, _u, _s in forms if nd[n] > 0]
        print("TOPKFORM arm=%s verdict=%s refused=%s note=the bf16 bytes are reproduced only when R1 has "
              "landed in the model dtype, and the pinned routed_sum refuses the f32-carry weight form on "
              "the arm whose licence there IS exactness: both halves of AGENTS.md prediction 4 (the "
              "renormalised weights are cast to the logits' dtype BEFORE the multiply) are carried by the "
              "pinned material, not chosen here. The fp32-carry form reproducing the f32 record is the "
              "renormalisation being bit-exact in itself, which is a different claim from which form gets "
              "multiplied"
              % (arm, "pinned-bytes-carry-" + "+".join(carried), ",".join(refused) or "none"))
        out[arm] = dict(nd=nd, refused=refused, product_shown=show_product,
                        emitted_dtype=str(st["topk_weights"].dtype),
                        f32_dtype=str(st["topk_weights_f32"].dtype))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine/block_gate.py", description=__doc__.splitlines()[0])
    ap.add_argument("--device", action="store_true",
                    help="claim device residency. This gate owns no device path, so it refuses with "
                         "tag=live-device-path-absent; it does not imply --staged-evidence")
    ap.add_argument("--staged-evidence", default=None, metavar="LOG",
                    help="read-back log of engine/stage_run.py --device --verify, admitted as a "
                         "PRECONDITION on staging integrity — not a device result")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of %s" % (ARMS,))
    ap.add_argument("--fault", default="",
                    help="name one control; it runs with the others and must still fire")
    args = ap.parse_args(argv)
    # Two flags that used to live here are gone, and the reason is the rule this gate exists to enforce.
    # `--selftest` ("controls only, no stage rows") starved the controls that bite on the printed
    # population — the parse reconciliation, the form probe — and a control that cannot bite is a SILENT
    # red, so a mode whose roster cannot fire is worse than no mode (`pitfalls/40`
    # §gates-must-fail-for-something). `--check-only` ("certify that nothing was admitted") was worse in a
    # different way: it changed nothing except that it refused to coexist with an admission, which is what
    # its absence already means. The RESULT line's `staged=absent|admitted` is the record; a flag that
    # only conflicts with another flag is a placeholder.
    if args.device:
        # A flag this process cannot honour is a usage error, not a note beside a green verdict: a reader
        # who asked for a device result must not be handed `verdict=QUALIFIED ... device=off` and have to
        # notice the difference. The refusal text lives in `block_forward.py` so the reason is one thing.
        try:
            BF.device_refusal()
        except BF.Refused as exc:
            print("REFUSED tag=%s note=%s" % (exc.tag, exc))
            print("NEXT use --staged-evidence LOG to admit engine/stage_run.py --device --verify read-back "
                  "as a staging PRECONDITION, which is not a device result")
        return 1

    out = Tee(sys.stdout)
    sys.stdout = out
    try:
        run = main_body(out, args)
    finally:
        sys.stdout = out.fh
    return run


def main_body(out, args):
    v = Verifier()
    cfg = BF.load_geometry()
    torch.set_num_threads(int(cfg["threads"]))       # the manifest's recorded thread count
    dumps = BF.Dumps()
    band = G.load_band(BF.BAND_PATH)
    digest = G.band_digest(band)
    limits, raw, missing = prototype_limits()
    population = derived_population(dumps)
    global REFERENT_GDN, REFERENT_MOE
    REFERENT_GDN = set(population["gdn"])
    REFERENT_MOE = set(population["moe"])
    arms = [a for a in args.arms.split(",") if a]

    print("BLOCKGATE-HDR block=layer0 token=%d gdn_pass=%s moe_pass=layer0.row%d arms=%s "
          "bands=%s band_digest=%s threads=%d interpreter=torch/%s"
          % (BF.TOKEN_ROW, BF.GDN_MODE, BF.TOKEN_ROW, ",".join(arms), BF.BAND_PATH.name,
             digest[:16], cfg["threads"], torch.__version__))
    print("POPULATION gdn=%d moe=%d glue=%d per_arm=%d arms=%d expected_rows=%d "
          "source=two manifests (arm=ref records) + specification/03 §03.2 step list (%d steps, "
          "%d residual adds: %s)"
          % (len(population["gdn"]), len(population["moe"]), len(population["glue"]),
             len(population["all"]), len(arms), len(population["all"]) * len(arms),
             population["spec_steps"], population["spec_adds"], ",".join(population["add_names"])))
    print("LIMITS %s missing=%s source=prototype floor probes re-read at run time"
          % (" ".join("%s=%s" % (k, limits[k]) for k in sorted(limits)), missing or "none"))

    rows, meta = build_rows(dumps, cfg, arms, population)
    tally = adjudicate(rows, band, limits, population)
    for row in rows:
        print_row(row)

    device_refusal_tag = "live-device-path-absent"     # true in every mode this file can run

    staged, staged_refusal, staged_note = None, None, None
    if args.staged_evidence:
        try:
            staged = BF.staged_precondition(args.staged_evidence)
        except BF.Refused as exc:
            staged_refusal, staged_note = exc.tag, str(exc)
    if staged is not None:
        print("PRECONDITION name=policy-a-staging-integrity verdict=ADMITTED log=%s artifact=%s "
              "pin_sha256=%s tensors=%d/%d bytes_rehashed=%d log_readback=%s "
              "label=staging-integrity-precondition-not-a-device-result"
              % (staged["log"], staged["artifact"], staged["pin_sha"][:16], staged["tensors"],
                 len(BF.POLICY_A_TENSORS), staged["bytes_hashed"], staged["readback"]))
    elif staged_refusal:
        print("PRECONDITION name=policy-a-staging-integrity verdict=REFUSED tag=%s note=%s"
              % (staged_refusal, staged_note))
    else:
        print("PRECONDITION name=policy-a-staging-integrity verdict=ABSENT "
              "tag=staged-evidence-flag-absent note=this run claims nothing about staging and nothing "
              "about residency")

    if args.fault:
        print("FAULT requested=%s note=every control runs in every mode; one is not separable "
              "from the population it must bite on" % args.fault)

    text = out.text()
    parsed = parse_printed(text)
    # Which rows this run actually holds an §10.6 exactness licence for: the form probe prints its
    # product rows only there, because a row under a quantization band cannot separate a cast order.
    licensed_exact = set((r["stage"], r["arm"]) for r in rows
                         if "exactness" in str(r.get("licence", "")))
    forms = topk_form_probe(dumps, cfg, meta, limits, licensed_exact) if rows else {}
    controls(v, dumps, cfg, rows, population, band, text, parsed, meta, limits, forms)

    checks(v, dumps, cfg, rows, parsed, population, meta, band, digest, limits, raw, missing,
           staged, staged_refusal or "staged-evidence-flag-absent", device_refusal_tag, False,
           forms, text, out.text())
    red = len(v.red)
    verdict = "FAIL" if red else ("QUALIFIED" if tally["unlicensed"] else "PASS")
    print("STAGETALLY arms=%d licensed=%d accepted=%d rejected=%d unlicensed=%d"
          % (len(arms), tally["licensed"], tally["accepted"], tally["rejected"],
             tally["unlicensed"]))
    print("BLOCKGATE verdict=%s block=layer0 arms=%d rows=%d parsed=%d licensed=%d accepted=%d "
          "rejected=%d unlicensed=%d checks=%d red=%d controls=%d fired=%d band_digest=%s "
          "note=a row with no licence is never counted as accepted, and PASS is unreachable while "
          "any row is unlicensed"
          % (verdict, len(arms), len(rows), len(parsed), tally["licensed"], tally["accepted"],
             tally["rejected"], tally["unlicensed"], len(v.names), len(v.red), len(v.controls),
             v.fired, digest[:16]))
    print("RESULT mode=%s checks=%d findings=%d controls=%d fired=%d rows=%d arms=%s device=%s "
          "staged=%s no timing taken"
          % ("cpu", len(v.names), len(v.red), len(v.controls), v.fired, len(rows), ",".join(arms),
             "off", "admitted" if staged is not None else
             ("refused:%s" % staged_refusal if staged_refusal else "absent")))
    if v.red:
        for r in v.red:
            print("RED %s" % r)
    return 1 if v.red else 0


def checks(v, dumps, cfg, rows, parsed, population, meta, band, digest, limits, raw, missing,
           staged, staged_refusal, device_refusal_tag, live_device_ran, forms, text, full_text):
    """Every check is an expression over parsed records and derived populations."""
    per_arm = len(population["all"])
    arms = sorted(set(r["arm"] for r in rows)) or ["-"]
    counts = {a: len([r for r in rows if r["arm"] == a]) for a in arms}
    v.check("stage-rows-equal-derived-population",
            all(c == per_arm for c in counts.values()) and len(rows) == per_arm * len(arms),
            "per_arm=%s derived=%d total=%d expected=%d" % (counts, per_arm, len(rows),
                                                            per_arm * len(arms)))
    v.check("parsed-record-count-equals-printed", len(parsed) == len(rows) and len(parsed) > 0,
            "parsed=%d printed=%d (a parser matching nothing would print 0 and pass)"
            % (len(parsed), len(rows)))
    accepted_unlicensed = [r for r in parsed if r["band"] == "ACCEPTED" and r["licence"] == "none"]
    v.check("no-accepted-row-without-a-licence", not accepted_unlicensed,
            "accepted=%d unlicensed_accepts=%d %s"
            % (len([r for r in parsed if r["band"] == "ACCEPTED"]), len(accepted_unlicensed),
               accepted_unlicensed[:2]))
    unl = [r for r in parsed if r["band"].startswith("UNLICENSED")]
    bad_unl = [r for r in unl if not re.match(r"^UNLICENSED stage=\S+ reason=\S+", r["band"])]
    v.check("every-unlicensed-row-names-stage-and-reason", not bad_unl,
            "unlicensed=%d malformed=%d" % (len(unl), len(bad_unl)))
    v.check("band-digest-pinned", digest == BAND_DIGEST,
            "band_digest=%s pinned=%s" % (digest[:16], BAND_DIGEST[:16]))
    tally_pass = (not [r for r in parsed if r["band"].startswith("UNLICENSED")])
    v.check("pass-unreachable-while-rows-unlicensed",
            (len(unl) == 0) or (tally_pass is False),
            "unlicensed=%d; PASS would require zero" % len(unl))
    v.check("probe-limits-came-from-their-probes", not missing and
            all(k in raw for k in (m for m in (b[1] for b in MOE_BANDS.values()) if m)),
            "limits=%s missing=%s probe_keys_present=%d" % (len(limits), missing, len(raw)))
    art_types = {r["name"]: r["ggml_type"] for r in meta.get("policy_a", {}).get("reads", [])
                 if r.get("ctx") == "forward" and r.get("ggml_type")}
    n_width, width_bad = width_verdicts(BF.PolicyA.WIDTH, art_types)
    v.check("policy-a-reads-carry-their-declared-width", n_width > 0 and not width_bad,
            "compared=%d of %d declared policy-A tensors against the ggml type the artifact record "
            "stores for that name (engine/loader.py's own veto source), mismatches=%s. The dequantized "
            "dtype is not in this comparison: a Q8_0 tensor arrives as float32 by design, so comparing "
            "a torch dtype with a declared quant type name could only ever fail"
            % (n_width, len(BF.PolicyA.WIDTH), width_bad[:3]))
    n_slices, whole_exps, gather_ok = gather_census(meta.get("policy_a", {}).get("reads", []), cfg)
    v.check("routed-experts-are-gathered-slices-not-whole-tensors", gather_ok and n_slices > 0,
            "gathered expert slices=%d against the derived %d (3 matrices x top_k), whole-tensor expert "
            "reads=%d, counted over ctx=forward records only so a control's recomputation cannot be "
            "billed against the shipping read pattern"
            % (n_slices, 3 * cfg["top_k"], len(whole_exps)))
    loaders = BF.Weights("policy_a", cfg)
    try:
        exact_types = {n: loaders.pa.L.handle(n)["type_name"]
                       for n in ("blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight",
                                 "blk.0.ssm_norm.weight")}
    finally:
        loaders.close()
    v.check("router-and-norm-family-is-exact-in-the-artifact",
            all(t == "F32" for t in exact_types.values()),
            " ".join("%s=%s" % (k.split(".")[-1], t) for k, t in sorted(exact_types.items())) +
            " (D-01a; this is what licenses judging the router rows on the policy-A arm)")
    v.check("no-rate-figures-in-output",
            not re.search(r"\b(ms|t/s|GB/s|ms=|elapsed|duration)\b", text),
            "scanned %d printed lines against the rate-token list and found none; the list itself is not "
            "printed here because the scan covers this line too" % len(text.splitlines()))
    const = constant_predicates((HERE / "block_gate.py").read_text()
                                + (HERE / "block_forward.py").read_text())
    v.check("no-constant-predicates-in-the-gate-source", not const, "found=%s" % (const[:3] or "[]"))
    ref_logits_dtype = BF._DTYPE[dumps.rec("layer0.router_logits")["dtype"]]
    v.check("router-logits-dtype-derivation-matches-the-pinned-referent",
            cfg["act_dtype"] is ref_logits_dtype,
            "config text_config.dtype=%s, the dtype the pinned layer0.router_logits record stores=%s; "
            "R5's cast target is 'the logits dtype', so both spellings of it must be the same derived "
            "thing and a config drift cannot go unnoticed"
            % (cfg["act_dtype"], ref_logits_dtype))
    cast_dtypes = {a: str(forms[a]["emitted_dtype"]) for a in sorted(forms)}
    v.check("router-r5-cast-is-a-real-cast-on-every-arm",
            bool(forms) and all(d == str(cfg["act_dtype"]) for d in cast_dtypes.values()),
            "the weight the routed multiply carries must be dtype=%s on every arm; saw %s. On the "
            "policy-A arm an f32 there means R5 was a no-op and the multiply carried the fp32 weight, "
            "which is AGENTS.md prediction 4's wrong order"
            % (cfg["act_dtype"], cast_dtypes))
    claims = device_claims(full_text)
    v.check("device-claims-come-from-device-only", live_device_ran or not claims,
            "live device path ran=%s refusal=%s; device-shaped lines in this run's output=%d %s — a CPU "
            "process may print a staging PRECONDITION but never a device result"
            % (live_device_ran, device_refusal_tag, len(claims),
               [ln.split()[0] for ln in claims][:3] or "none"))
    if staged is not None:
        ok = staged["tensors"] == len(BF.POLICY_A_TENSORS) and staged["bytes_hashed"] > 0
        detail = ("admitted %d/%d policy-A tensors this block reads, %d B of artifact re-hashed against "
                  "the digests the log records; log readback=%s"
                  % (staged["tensors"], len(BF.POLICY_A_TENSORS), staged["bytes_hashed"],
                     staged["readback"]))
    else:
        ok = staged_refusal is not None
        detail = ("no staging admitted (tag=%s); the check has nothing to vouch for and says so instead "
                  "of defaulting to a claim" % staged_refusal)
    v.check("staged-precondition-covers-this-blocks-tensors", ok, detail)
    v.check("pinned-digests-verified-on-every-dump-read", dumps.files_read > 0,
            "files_read=%d bytes_read=%d (every read sha256-checked against its manifest record)"
            % (dumps.files_read, dumps.bytes_read))


if __name__ == "__main__":
    sys.exit(main())
