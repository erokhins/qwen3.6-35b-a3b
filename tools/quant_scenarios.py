#!/usr/bin/env python3
"""quant_scenarios.py — residency and decode-ceiling arithmetic for the D-01 quantization choice.

Why this exists: DECISIONS.md D-01 (experts at 4-bit, what about attention?) cannot be decided by
taste, and the two binding measured facts are both in this repo:
  * the device-visible budget is ~32 GiB TOTAL per process, and host-pinning does not escape it
    (devices/measured-facts.md §"Residency above the pool", measured 2026-10-26);
  * a warm read at 190-237 GB/s is what a decode step is priced against (devices/measured-facts.md
    §"Memory bandwidth (iGPU, warm, best-of-N)", the same numbers copied in pitfalls/00).

THE CENSUS IS DERIVED, NOT MODELLED. Until 2026-10-26 this file hand-wrote eight algebraic terms
(`(2*L+4)*h`, `3*h*(3*h)`, `+ 2*h` per linear layer, ...) and was wrong by −0.4138 GiB: it omitted the
GDN `out_proj`/`ssm_out` family (251,658,240 elts) and `mtp.fc` (8,388,608), carried two phantom terms
(−37,748,736 and −122,880) and counted 84 h-sized norms where the checkpoint has 86 (+4,096). The
decomposition to the element is in quantization/evidence/bf16-census-adjudication-2026-10-26.md
§3/§4/§5 — read that file before changing anything here; it is the authority this file is checked
against, and its §8 holds the doc replacements the corrected numbers require.

So no element count comes from an expression any more. The census is built from files:
  * the pinned GGUF's own tensor records — models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (733);
  * every name in checkpoints/Qwen3.6-35B-A3B.safetensors.index.json (1,045), which is also where
    `metadata.total_size` comes from and is what the whole census is closed against;
  * the safetensors headers of the four shards under models/qwen36-src/ (424 real shapes/dtypes), and
    tools/bf16_census.py — the reference implementation of the shapes, the HF↔GGUF name map and the
    family buckets. This file IMPORTS those (see `_src()`); it does not restate them, because a third
    copy of a formula is not a ruler.
Two independent counts of the same weight set are kept and required to agree to 0 bytes: the census
side (the artifact's records + the element counts of the index names the artifact does not carry, i.e.
the MTP draft block) and the index side (every index name, shard-witnessed where a shard states the
shape). Every width comes from tools/ggml_block_geom.py (the pinned ggml-common.h structs) as a *type
name* applied with `row_bytes()`; no bits-per-weight literal prices anything here.

Corrected figures this file prints and *requires* (REQUIRED_FIGURES; a disagreement aborts instead of
quietly adopting this tool's own number): text bf16 **66.1337 GiB** — 712 tensors = the 40 blocks + the
19 `mtp.*` draft tensors — **64.5604 GiB** without the draft block, which is all the pinned artifact
carries; policy A **19.7848 GiB** resident, **2.5252 GiB/token** with one draft pass per token,
**2.4653** target-only; policy F (Q8_0 throughout) **35.16 GiB**. The figures this file used to print —
65.7198 / 19.59 / 2.331 / 34.97 — are wrong; README.md, PLAN.md, DECISIONS.md D-01/D-11 and
quantization/README.md still quote them until they are amended. Provenance for each corrected number:
adjudication §3 (census) and §6 (policy rows), each recomputed there from the artifact + draft census
with widths from the pinned ggml header.

Ceilings are PREDICTIONS, not measurements: bytes-read-per-token divided into a GB/s figure already
recorded in devices/measured-facts.md (237-240 pure read, 210 copy read+write, 190 the low end of the
measured window). Policy A at 2.5252 GiB/token predicts ~87 tok/s at 237 GB/s, ~77 at 210, ~70 at 190;
the ~95/84/76 this file used to imply came out of the wrong census. Nothing here is measured: no timing,
no `ms` figure, no device call, and the 0.8B engine's ~60 %-of-own-ceiling discount still applies on
top (README.md).

Conventions inherited, not decided (adjudication §10):
  * policy A's "exact" bucket (routers, norms, the GDN 1-D controls) is charged **bf16**, as this file
    always did and as quantization/residency-layout.md does; charging it F32-as-published is worth
    +0.0403 GiB resident and +0.0403 GiB/token and is an open user decision (D-01's "stay exact"), so
    neither variant is added to the table;
  * the quoted per-token figure charges **one draft pass per token** — what the old table did, so the
    apples-to-apples row — and the target-only row is printed beside it;
  * `Q36_CONFIG` / `Q36_INDEX` are refused rather than silently ignored (see `_src()`).

Usage:
  python3 tools/quant_scenarios.py                     # policy table at 237/210/190 GB/s
  python3 tools/quant_scenarios.py --census            # the derived census + the closures
  python3 tools/quant_scenarios.py --read 237 210 190  # GB/s figures to price against
  python3 tools/quant_scenarios.py --selftest          # 8 injected faults, all 8 must fire
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_block_geom as GBG                              # noqa: E402
from gguf_reader import Gguf                               # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GiB = 2.0 ** 30
BF16_BYTES = 2
# The GB/s figures already measured and recorded in devices/measured-facts.md. Divisors only: this tool
# measures nothing.
READS = (237.0, 210.0, 190.0)

# Width reference for printing, derived from the pinned header rather than typed. Nothing here computes
# a policy number from it: policy rows take their widths from POLICY_TYPES, which is checked against
# this table in assert_policy_widths_derived().
BPW_NOTE = ("widths derived from %s struct sizes" % GBG.HEADER_REL)

TEXT_FAMILIES = ("routed_experts", "gdn_linear_attn", "full_attn", "shared_experts", "routers",
                 "embed", "lm_head", "norms_misc")


class CensusFault(Exception):
    """The census refuses to print a number it cannot stand behind.

    Every abort goes through here, because the failure being defended against is a census that silently
    reads low — a dropped family, a short parse, an empty population — and still prints a tidy total.
    A zero here is an error, never a result."""


def _src():
    """tools/bf16_census.py — the reference implementation of shapes, the HF↔GGUF name map, families.

    Imported inside functions, not at module level: bf16_census imports THIS module to price its control
    arm, so a module-level import here would be a cycle. And nothing here calls bf16_census.build(),
    which would call back into `census()` and recurse; what is reused is the naming/shape/family layer
    (`geometry`, `hf_shape`, `family_of`, `strip_layer`, `hf_kind`, `MAP_HF_GGUF`, `shard_witnesses`) —
    which is exactly where the retired hand-written shape model used to live.
    """
    try:
        import bf16_census as BC
    except ImportError as e:
        raise CensusFault("cannot import the census's reference implementation "
                          "(tools/bf16_census.py): %s" % e)
    for var in ("Q36_CONFIG", "Q36_INDEX"):
        if os.environ.get(var):
            raise CensusFault("%s is set but no longer honoured: the census is derived from the pinned "
                              "GGUF, the safetensors index and the shard headers read by "
                              "tools/bf16_census.py (%s, %s), not from a config-only shape model, and a "
                              "config-only override is the hand-modelled census whose −0.4138 GiB this "
                              "file fixed" % (var, BC.CFG, BC.IDX))
    return BC


# Quoted claims, for naming the drift — not inputs to anything computed here. A figure is only "required"
# because an independent ruler published it first (quantization/evidence/bf16-census-adjudication-
# 2026-10-26.md §3 and §6). If a derived figure disagrees by more than its tolerance the tool aborts and
# says so: it does not adopt its own number and does not edit the evidence file.
REQUIRED_FIGURES = (
    ("text bf16 GiB (712 names, incl. the MTP draft block)", "text_gib", 66.1337, 5e-4),
    ("text bf16 GiB without the draft block (733 records)", "artifact_gib", 64.5604, 5e-4),
    ("draft block bf16 GiB (19 mtp.* names)", "draft_gib", 1.5733, 5e-4),
    ("vision tower bf16 GiB (333 names)", "vision_gib", 0.8318, 5e-4),
    ("policy A resident GiB", "A_resident", 19.7848, 5e-4),
    ("policy A GiB/token, one draft pass per token", "A_per_token", 2.5252, 5e-4),
    ("policy A GiB/token, target only", "A_per_token_only", 2.4653, 5e-4),
    ("policy F (Q8_0 throughout) resident GiB", "F_resident", 35.16, 5e-3),
)


# ---------------------------------------------------------------------------------------------
# the census: two independent counts of the same weight set, closed against each other and against the
# index's own total_size.
# ---------------------------------------------------------------------------------------------
def build_population(faults=()):
    """Both counts, the pairing between them, and the witnesses that make a 0 mean something.

    `faults` is a set of injected-fault names. Faults are applied to the CENSUS side only, so the index
    side stays the anchor that catches them; FAULTS documents what each one is and which closure must
    name it.
    """
    BC = _src()
    g = BC.geometry()
    idx = json.load(open(BC.IDX))
    wm = idx.get("weight_map")
    if not wm:
        raise CensusFault("%s has no weight_map; refusing to print a census of nothing" % BC.IDX)
    try:
        index_total = int(idx["metadata"]["total_size"])
    except KeyError:
        raise CensusFault("%s carries no metadata.total_size, so the census has nothing to close "
                          "against" % BC.IDX)

    # ---- index side: every name; shard-witnessed shape where a shard states one -----------------
    wit, shard_files = BC.shard_witnesses()
    index_pop, shape_checks, mismatch = {}, 0, []
    for name in wm:
        sh = wit[name][1] if name in wit else BC.hf_shape(name, g)
        if sh is None:
            raise CensusFault("cannot derive a shape for index name %s — a census that silently drops "
                              "what it cannot model reads low and still prints a total" % name)
        if name in wit:
            shape_checks += 1
            modelled = BC.hf_shape(name, g)
            if modelled is not None and modelled != sh:
                mismatch.append((name, sh, modelled))
        elts = 1
        for d in sh:
            elts *= d
        index_pop[name] = {"elts": elts, "fam": BC.family_of(name), "witnessed": name in wit}
    if mismatch:
        raise CensusFault("config-derived shapes disagree with the real shard headers:\n  "
                          + "\n  ".join("%s: shard %s vs model %s" % m for m in mismatch))
    if not shape_checks:
        raise CensusFault("no shard header under %s witnessed any index name, so every shape in this "
                          "census is modelled and nothing checks it" % BC.SHARD_DIR)

    # ---- census side: the artifact's records + the draft names it does not carry -----------------
    if not os.path.exists(BC.GGUF):
        raise CensusFault("the pinned artifact %s is absent, so there is no census to derive" % BC.GGUF)
    gg = Gguf(BC.GGUF)                                     # header parse only; no weight byte is read
    art = {t.name: {"elts": t.n_elts, "type": t.type_name} for t in gg.tensors}
    if "short-record" in faults:
        # the fault: one record vanishes after the header already said how many there are
        art.pop(sorted(art)[-1])
    if len(art) != int(gg.n_tensors):
        raise CensusFault("parsed %d GGUF tensor records, the header declares %d"
                          % (len(art), gg.n_tensors))
    if not art:
        raise CensusFault("the pinned artifact carries 0 tensor records; refusing to print a census of "
                          "nothing")

    claim, draft_names = {}, []
    for name, d in index_pop.items():
        if d["fam"] == "vision":
            continue                                  # the artifact carries no vision tower either
        base = BC.strip_layer(name)
        # bf16_census' own naming rule, restated so the two tools pair identically: `mtp.layers.0.*`
        # folds onto the ordinary layer pattern and must not be allowed to claim blk.0's tensors.
        glist = [] if name.startswith("mtp.") else BC.MAP_HF_GGUF.get(base, [])
        if not glist:
            draft_names.append(name)                   # an index name the artifact does not carry
            continue
        li = re.match(r"^(?:model\.language_model|mtp)\.layers\.(\d+)\.", name)
        if "N" in "".join(glist):
            if li is None:
                raise CensusFault("the mapping for %s wants a layer index this name does not carry"
                                  % name)
            glist = [x.replace("N", li.group(1)) for x in glist]
        for gn in glist:
            if gn not in art:
                raise CensusFault("the mapping pairs %s -> %s, but the artifact carries no %s"
                                  % (name, gn, gn))
            if gn in claim:
                raise CensusFault("artifact record %s is claimed by both %s and %s — one record counted "
                                  "twice is how a census reads high" % (gn, claim[gn], name))
            claim[gn] = name
    orphans = sorted(set(art) - set(claim))
    if orphans:
        raise CensusFault("%d artifact records are claimed by no HF index name (first: %s) — either the "
                          "name map is incomplete or the artifact carries a tensor the text model does "
                          "not have" % (len(orphans), orphans[0]))
    if not draft_names:
        raise CensusFault("no index name was left unclaimed by the artifact, so the MTP draft block "
                          "vanished from the census — that is the −1.5733 GiB error of 2026-10-26")

    census_recs = [{"name": gn, "fam": index_pop[hf]["fam"], "elts": art[gn]["elts"],
                    "kind": "artifact", "hf": hf} for gn, hf in claim.items()]
    census_recs += [{"name": n, "fam": index_pop[n]["fam"], "elts": index_pop[n]["elts"],
                     "kind": "draft", "hf": n} for n in draft_names]
    if "empty-census" in faults:
        census_recs = []
    census_recs, injected = _apply_faults(census_recs, g, faults)
    if not census_recs:
        raise CensusFault("the census side is empty (0 records, 0 elements); an empty census is an "
                          "error, not a zero")

    cen_fam, index_fam = {}, {}
    for r in census_recs:
        cen_fam[r["fam"]] = cen_fam.get(r["fam"], 0) + r["elts"]
    for name, d in index_pop.items():
        index_fam[d["fam"]] = index_fam.get(d["fam"], 0) + d["elts"]
    for fam in TEXT_FAMILIES:
        if not cen_fam.get(fam):
            raise CensusFault("census family %s is empty — every text family must be populated or the "
                              "printed total is silently low" % fam)

    return dict(g=g, census=census_recs, index=index_pop, claim=claim, draft=sorted(draft_names),
                cen_fam=cen_fam, index_fam=index_fam, index_total=index_total,
                n_index=len(index_pop), n_wit=shape_checks, decl_tensors=int(gg.n_tensors),
                n_art=len(art), shards=shard_files, injected=injected)


def _apply_faults(recs, g, faults):
    """Inject the census defects into the census side and return what each one was expected to be worth.

    Each fault is the shape of an error this file actually made (adjudication §4/§5). Its magnitude is
    computed from the checkpoint's own geometry, never copied from the evidence file, so a fault whose
    size drifted with the config still fires with the right number.
    """
    BC = _src()
    h = g["h"]
    nlin = sum(1 for x in g["layer_types"] if x == "linear_attention")
    # how many h-sized norms the census actually carries, counted rather than asserted; the h-norm fault
    # is defined against this, so it cannot drift into a magic number
    real_h_norms = sum(1 for r in recs if r["fam"] == "norms_misc" and r["elts"] == h)
    injected, out = [], list(recs)
    for fault in sorted(set(faults)):
        if fault in STRUCT_FAULTS:
            continue                       # applied in build_population, where the record count is read
        fam, fn, what, match = FAULTS[fault]
        delta = fn(g, nlin, g["L"], real_h_norms)
        if delta < 0:
            want, dropped, keep = -delta, 0, []
            for r in out:
                if r["fam"] == fam and match(BC, r, h) and dropped < want:
                    dropped += r["elts"]
                    continue
                keep.append(r)
            if dropped != want:
                raise FaultInjectionError("cannot inject %s: wanted to drop %d elts of %s, dropped %d"
                                          % (fault, want, what, dropped))
            out = keep
        else:
            out.append({"name": "INJECTED." + fault, "fam": fam, "elts": delta, "kind": "phantom",
                        "hf": "INJECTED." + fault})
        injected.append((fault, fam, delta, what))
    return out, injected


def _m_out_proj(BC, r, h):
    return BC.hf_kind(r["hf"]) == "linear_attn.out_proj.weight"


def _m_mtp_fc(BC, r, h):
    return BC.hf_kind(r["hf"]) == "mtp.fc.weight"


def _m_h_norm(BC, r, h):
    # only a real h-sized norm counts. Dropping any norms_misc tensor would be the lazier, wrong fault:
    # mtp.fc sits in that bucket too and is 2h^2 elts on its own.
    return r["elts"] == h


# The five census defects of quantization/evidence/bf16-census-adjudication-2026-10-26.md §4/§5, as
# injections, and the two structural faults below. A magnitude is computed from the checkpoint's own
# geometry (`geometry()` is the config the shard headers were checked against) and from what the census
# actually carries, so a fault whose size drifted with the checkpoint still fires with the right number.
def _f_out_proj(g, nlin, L, real_h_norms=None):
    return -(g["lv"] * g["dv"] * g["h"]) * nlin


def _f_mtp_fc(g, nlin, L, real_h_norms=None):
    return -(g["h"] * 2 * g["h"]) * max(1, g["draft"])


def _f_phantom_9h2(g, nlin, L, real_h_norms=None):
    return 3 * g["h"] * (3 * g["h"])


def _f_phantom_2h(g, nlin, L, real_h_norms=None):
    return 2 * g["h"] * nlin


def _f_norm_count(g, nlin, L, real_h_norms):
    # the old expression counted 2L+4 h-vectors; `real_h_norms` is how many the census side actually
    # carries (40 input + 40 post + the final norm + the draft block's five), so this is exactly the set
    # of h-norms the old census never counted
    return -(real_h_norms - (2 * L + 4)) * g["h"]


FAULTS = {
    "drop-gdn-out-proj": ("gdn_linear_attn", _f_out_proj,
                          "drop the GDN `out_proj`/`ssm_out` family ([h, lv*dv] per linear layer) from "
                          "the census, which is what this file did until 2026-10-26", _m_out_proj),
    "drop-mtp-fc": ("norms_misc", _f_mtp_fc,
                    "drop `mtp.fc.weight` [h, 2h], the draft block's fusion projection, absent from this "
                    "file's old norms bucket", _m_mtp_fc),
    "phantom-9h2": ("norms_misc", _f_phantom_9h2,
                    "add the old `3*h*(3*h)` term to norms_misc: no tensor family of that size exists",
                    None),
    "phantom-2h-per-linear": ("gdn_linear_attn", _f_phantom_2h,
                              "add the old `+ 2*h` per linear layer: nothing is behind it, and the tensor "
                              "left with no summand was out_proj", None),
    "norm-count-short": ("norms_misc", _f_norm_count,
                         "count `2L+4` h-sized norms instead of counting them: the h-norms the old "
                         "expression never included", _m_h_norm),
}
STRUCT_FAULTS = {
    "short-record": "drop one artifact record after the header declared how many there are",
    "empty-census": "drop every record from the census side and require a refusal, not a zero total",
}


def closures(pop):
    """The closures. Returns (report lines, non-zero residues in bytes, derived numbers)."""
    cen, idxp = pop["cen_fam"], pop["index_fam"]
    lines, res = [], {}

    index_all = sum(d["elts"] for d in pop["index"].values())
    d = index_all * BF16_BYTES - pop["index_total"]
    lines.append("K1  index side: %d names, sum(elts) x 2 = %d B vs index metadata total_size %d B   "
                 "residue %d B" % (pop["n_index"], index_all * BF16_BYTES, pop["index_total"], d))
    if d:
        res["K1 index total_size"] = d

    fam_res = {}
    lines.append("K2  census side (artifact records + draft names) vs index side, per text family:")
    for fam in sorted(set(cen) | set(idxp)):
        if fam == "vision":
            continue        # index-only by construction: the artifact carries no vision tower
        delta = cen.get(fam, 0) - idxp.get(fam, 0)
        fam_res[fam] = delta
        if delta:
            res["K2 %s" % fam] = delta * BF16_BYTES
        lines.append("      %-18s census %14d  index %14d   residue %+d elts"
                     % (fam, cen.get(fam, 0), idxp.get(fam, 0), delta))
    if cen.get("vision"):
        res["K2 vision"] = cen["vision"] * BF16_BYTES
        lines.append("      vision is index-only; the census side carries %d elts of it" % cen["vision"])
    cen_text = sum(cen.get(f, 0) for f in TEXT_FAMILIES)
    idx_text = sum(idxp.get(f, 0) for f in TEXT_FAMILIES)
    d = (cen_text - idx_text) * BF16_BYTES
    lines.append("K3  text total: census %d elts vs index %d elts   residue %d B; index-only vision "
                 "%d elts" % (cen_text, idx_text, d, idxp.get("vision", 0)))
    if d:
        res["K3 text total"] = d
    lines.append("K4  artifact records parsed %d, header declares %d, claimed %d, orphans 0"
                 % (pop["n_art"], pop["decl_tensors"], len(pop["claim"])))
    if pop["n_art"] != pop["decl_tensors"]:
        res["K4 record count"] = 1

    art = sum(r["elts"] for r in pop["census"] if r["kind"] == "artifact")
    draft = sum(r["elts"] for r in pop["census"] if r["kind"] == "draft")
    phantom = sum(r["elts"] for r in pop["census"] if r["kind"] == "phantom")
    return lines, res, dict(text=cen_text, art=art, draft=draft, phantom=phantom,
                            vision=idxp.get("vision", 0), fam_res=fam_res)


def census(faults=()):
    """{family: bf16 bytes} for the text tensors, plus the index's own total_size.

    The module's stable API (tools/bf16_census.py and tools/residency_budget.py import it): same keys,
    same units, correct values. Text families only — vision is counted by the closures but is in no
    policy row.
    """
    pop = build_population(faults)
    _, res, _ = closures(pop)
    if res:
        raise CensusFault("the census does not close: "
                          + "; ".join("%s = %+d B" % kv for kv in sorted(res.items())))
    return {f: pop["cen_fam"][f] * BF16_BYTES for f in TEXT_FAMILIES}, pop["index_total"]

class FaultInjectionError(Exception):
    """A selftest fault could not be injected. Loud, and never counted as the fault having fired."""


# ---------------------------------------------------------------------------------------------
# policy widths. A policy row names ggml TYPES per family class; the bytes come from
# tools/ggml_block_geom.py's struct sizes via row_bytes(). No bits-per-weight literal prices anything:
# POLICIES (bits, kept because other tools read it) is derived from POLICY_TYPES at import.
# ---------------------------------------------------------------------------------------------
# family -> policy class. "router" is never quantized by any policy (llama.cpp refuses: gating tensors,
# norms and conv1d stay as they are), and "embed" follows "lm_head" — same role, a vocab x hidden table
# read once per token at the ends of the pass.
FAM_BITS = {
    "routed_experts": "experts", "shared_experts": "shared", "routers": "router",
    "gdn_linear_attn": "attn", "full_attn": "attn", "embed": "embed", "lm_head": "lm_head",
    "norms_misc": "router",
}

# policy -> {class: ggml type}. The only width table in this file.
POLICY_TYPES = {
    "bf16 (does not fit)":           dict(experts="BF16", shared="BF16", attn="BF16",
                                         lm_head="BF16", router="BF16"),
    "A experts Q4_K, rest Q8_0":     dict(experts="Q4_K", shared="Q8_0", attn="Q8_0",
                                         lm_head="Q8_0", router="BF16"),
    "B experts Q4_K, rest bf16":     dict(experts="Q4_K", shared="BF16", attn="BF16",
                                         lm_head="BF16", router="BF16"),
    "C experts MXFP4, rest Q8_0":    dict(experts="MXFP4", shared="Q8_0", attn="Q8_0",
                                         lm_head="Q8_0", router="BF16"),
    "D experts IQ4_XS, rest Q8_0":   dict(experts="IQ4_XS", shared="Q8_0", attn="Q8_0",
                                         lm_head="Q8_0", router="BF16"),
    "E experts Q4_K, attn Q4_K too": dict(experts="Q4_K", shared="Q4_K", attn="Q4_K",
                                         lm_head="Q6_K", router="BF16"),
    "F experts Q8_0, rest Q8_0":     dict(experts="Q8_0", shared="Q8_0", attn="Q8_0",
                                         lm_head="Q8_0", router="BF16"),
    "G experts Q4_K, attn Q8, lm Q6": dict(experts="Q4_K", shared="Q8_0", attn="Q8_0",
                                         lm_head="Q6_K", router="BF16"),
}
# The names the widths stand for, so a row can be checked against a struct rather than against a
# reviewer's memory. Q5_K is absent on purpose: no policy row uses it.
POLICY_WIDTH_NAMES = {"BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q4_K": 4.5, "IQ4_XS": 4.25,
                      "MXFP4": 4.25}


def _derive_policy_bits():
    need = sorted({t for row in POLICY_TYPES.values() for t in row.values()})
    g = GBG.geom(require=need)
    return {name: {cls: g[t]["bpw"] for cls, t in row.items()}
            for name, row in POLICY_TYPES.items()}


try:
    POLICIES = _derive_policy_bits()
except (FileNotFoundError, ValueError) as e:
    raise CensusFault("cannot price a policy width without the pinned ggml header %s: %s"
                      % (GBG.HEADER_REL, e))


def assert_policy_widths_derived():
    """Every width a policy row can spend must be a width the pinned ggml header produces.

    Two separate checks, because they catch different slips: the width a POLICIES row carries must still
    be the derived bpw of the ggml type POLICY_TYPES names for that class (so a row cannot be quietly
    retyped to a plausible number — that is how Q5_K once sat at 5.5625 in two tools), and the
    name->width claims in POLICY_WIDTH_NAMES must match the header row for that name. Aborts either
    way: a policy table is the one place a plausible number does the most damage, since it is quoted
    into DECISIONS.md.
    """
    g = GBG.geom()
    real = {t: g[t]["bpw"] for t in g}
    bad = []
    for pol, row in POLICIES.items():
        for cls, bits in row.items():
            tname = POLICY_TYPES[pol][cls]
            want = g.get(tname, {}).get("bpw")
            if want is None:
                bad.append("%s/%s: %s is not a width the pinned header derives" % (pol, cls, tname))
            elif abs(bits - want) > 1e-9:
                bad.append("%s/%s: row carries %g bpw but %s derives %g" % (pol, cls, bits, tname,
                                                                           want))
            if not any(abs(bits - v) < 1e-9 for v in real.values()):
                bad.append("%s/%s: %g bpw is no ggml type's derived width" % (pol, cls, bits))
    for bad_row in GBG.assert_matches(POLICY_WIDTH_NAMES):
        if bad_row["delta_bpw"] is not None:
            bad.append("%s: claimed %g, header derives %g" % (bad_row["type"], bad_row["stated_bpw"],
                                                             bad_row["derived_bpw"]))
        else:
            bad.append("%s: cannot be derived from %s, so the policy row that uses it is unverified"
                       % (bad_row["type"], GBG.HEADER_REL))
    if bad:
        raise SystemExit("ABORT: policy widths not derivable from %s:\n  %s" % (GBG.HEADER_REL,
                                                                               "\n  ".join(bad)))
    return {t: g[t]["bpw"] for t in ("BF16", "Q8_0", "Q6_K", "Q5_K",
                                     "Q4_K", "Q4_0", "IQ4_XS", "MXFP4") if t in g}


def BAM_BITS(family, pol):
    """bits/weight a policy spends on a family — the API tools/residency_budget.py reads.

    New arithmetic in this file uses POLICY_TYPE() + row_bytes() instead: bits/16 scaling approximates an
    on-disk tensor by a bf16-byte ratio, and the two differ whenever a family's element count is not a
    whole number of blocks.
    """
    cls = FAM_BITS[family]
    if cls not in pol and cls == "embed":
        cls = "lm_head"
    return pol[cls]


def POLICY_TYPE(polkey, family):
    cls = FAM_BITS[family]
    spec = POLICY_TYPES[polkey]
    if cls not in spec and cls == "embed":
        cls = "lm_head"
    return spec[cls]


def policy_rows(pop, polkey):
    """(main rows, draft rows) as {(family, ggml type): bytes}, priced by ggml struct sizes.

    The same width rule as tools/bf16_census.py's policy_rows, restated for this file's population so the
    two policy tables can be compared row for row; the widths themselves are single-sourced in
    POLICY_TYPES and the bytes in ggml_block_geom.row_bytes.
    """
    rows, draft = {}, {}
    for r in pop["census"]:
        if r["kind"] == "phantom":
            continue                      # an injected phantom is a closure test, never a priced tensor
        t = POLICY_TYPE(polkey, r["fam"])
        bucket = rows if r["kind"] == "artifact" else draft
        bucket[(r["fam"], t)] = bucket.get((r["fam"], t), 0) + GBG.row_bytes(r["elts"], t)
    return rows, draft


def per_token(pop, rows, draft_rows, with_draft=True):
    """Bytes read per generated token.

    The rule, stated so it can be argued: routed experts are read only for the `n_active` of `E` that
    fire; the token embedding is one row of the vocab table, not the table; `lm_head` is the whole untied
    table (1.0 GiB/token at bf16 — an untied table is a fifth of the per-token read, README.md);
    every other family is read in full. `with_draft` decides whether the MTP block is walked once per
    token — D-09/D-12 territory, undecided, so both numbers are printed and the row the old table quoted
    is the one-draft-pass one.
    """
    g = pop["g"]
    tot = 0
    pool = list(rows.items()) + (list(draft_rows.items()) if with_draft else [])
    for (fam, t), b in pool:
        if fam == "routed_experts":
            tot += b * g["n_active"] / g["E"]
        elif fam == "embed":
            tot += GBG.row_bytes(g["h"], t)
        else:
            tot += b
    return tot


def derived_numbers(pop):
    """The figures REQUIRED_FIGURES names, computed from the population and nothing else."""
    _, _, sc = closures(pop)
    out = {"text_gib": sc["text"] * BF16_BYTES / GiB,
           "artifact_gib": sc["art"] * BF16_BYTES / GiB,
           "draft_gib": sc["draft"] * BF16_BYTES / GiB,
           "vision_gib": sc["vision"] * BF16_BYTES / GiB}
    rows, draft = policy_rows(pop, "A experts Q4_K, rest Q8_0")
    out["A_resident"] = (sum(rows.values()) + sum(draft.values())) / GiB
    out["A_per_token"] = per_token(pop, rows, draft, True) / GiB
    out["A_per_token_only"] = per_token(pop, rows, draft, False) / GiB
    rows, draft = policy_rows(pop, "F experts Q8_0, rest Q8_0")
    out["F_resident"] = (sum(rows.values()) + sum(draft.values())) / GiB
    return out


def check_required(numbers, figures=REQUIRED_FIGURES):
    """Drift rows between what this tool derived and what the adjudication published. Empty = agreement.

    A drift row is a finding for a human, not a tolerance to widen: this tool's own numbers may be the
    wrong ones, which is the only reason it is checked against another ruler's published figures.
    """
    drift = []
    for label, key, quoted, tol in figures:
        got = numbers.get(key)
        if got is None:
            drift.append("%s: not derived at all (published %s)" % (label, quoted))
        elif abs(got - quoted) > tol:
            drift.append("%s: this tool derives %.4f, the adjudication published %s (delta %+.4f, "
                         "tolerance %g)" % (label, got, quoted, got - quoted, tol))
    return drift


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def print_census(pop):
    lines, res, sc = closures(pop)
    BC = _src()
    text = sc["text"]
    print("family census (bf16), derived from the pinned artifact's tensor records + the safetensors")
    print("index, with shapes witnessed against the shard headers — not from a hand-written shape model")
    print("  sources: %s (%d records); %s (%d names, total_size %d B); %d shards under %s (%d real"
          % (os.path.relpath(BC.GGUF, REPO), pop["n_art"], os.path.relpath(BC.IDX, REPO),
             pop["n_index"], pop["index_total"], len(pop["shards"]),
             os.path.relpath(BC.SHARD_DIR, REPO), pop["n_wit"]))
    print("  shapes witnessed)\n")
    for fam in sorted(TEXT_FAMILIES, key=lambda f: -pop["cen_fam"].get(f, 0)):
        elts = pop["cen_fam"].get(fam, 0)
        print("  %-18s %10.4f GiB %16d elts %6.2f %%" % (fam, elts * BF16_BYTES / GiB, elts,
                                                        100.0 * elts / text))
    print("  %-18s %10.4f GiB %16d elts" % ("TOTAL (text)", text * BF16_BYTES / GiB, text))
    print("    without draft   %10.4f GiB %16d elts   (the artifact alone: %d records, no draft block)"
          % (sc["art"] * BF16_BYTES / GiB, sc["art"], pop["n_art"]))
    print("    draft block     %10.4f GiB %16d elts   (%d mtp.* names the artifact does not carry)"
          % (sc["draft"] * BF16_BYTES / GiB, sc["draft"], len(pop["draft"])))
    print("    vision tower    %10.4f GiB %16d elts   (outside every policy row)"
          % (sc["vision"] * BF16_BYTES / GiB, sc["vision"]))
    print("  %-18s %10.4f GiB %16d elts" % ("index total_size", pop["index_total"] / GiB,
                                            pop["index_total"] // BF16_BYTES))
    print("  the index-minus-text delta is the vision tower and nothing else: %.4f + %.4f = %.4f GiB,"%(
        sc["text"] * BF16_BYTES / GiB, sc["vision"] * BF16_BYTES / GiB, pop["index_total"] / GiB))
    print("  and the closures say so to 0 bytes. The line this file used to print blamed a 1.246 GiB")
    print("  delta on \"the vision tower plus small tensors\": there are no small tensors left over, and")
    print("  0.4138 GiB of that delta was this census's own error (adjudication §3).")
    print("\nclosures (each must read 0 B, or the two counts are not the same weight set):")
    for ln in lines:
        print("  " + ln)
    print("  residuals: %s" % ("NONE — every closure closed to 0 bytes" if not res else
                               "; ".join("%s: %+d B" % kv for kv in sorted(res.items()))))
    print("\ndraft block, the whole artifact-vs-text gap (%d names): %s"
          % (len(pop["draft"]), ", ".join(pop["draft"])))
    return res


def print_policies(pop, reads):
    widths = assert_policy_widths_derived()        # aborts before any policy number is printed
    n_active, n_experts = pop["g"]["n_active"], pop["g"]["E"]
    print("residency = every weight at its policy width (whole ggml blocks, per tensor); per-token =")
    print("attn + shared + %d/%d of the routed experts + lm_head (embed is one row) + router every"
          % (n_active, n_experts))
    print("token, plus one MTP draft pass\n")
    print("PREDICTED ceilings: bytes divided by the GB/s figures already measured in")
    print("devices/measured-facts.md §\"Memory bandwidth (iGPU, warm, best-of-N)\" (237-240 pure read,")
    print("210 copy read+write, 190 the low end of the measured window). These are PREDICTIONS against")
    print("those measured rates, NOT measurements: no timing was taken here, and the 0.8B engine's ~60")
    print("%-of-its-own-ceiling discount still applies on top (README.md).")
    hdr = "%-34s %9s %10s %12s" % ("policy", "GiB", "GiB/tok", "GiB/tok-tgt")
    hdr += "".join("  %6s" % ("%d GB/s" % r) for r in reads)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in POLICY_TYPES:
        rows, draft = policy_rows(pop, name)
        resident = (sum(rows.values()) + sum(draft.values())) / GiB
        pt = per_token(pop, rows, draft, True)
        pto = per_token(pop, rows, draft, False)
        line = "%-34s %9.2f %10.4f %12.4f" % (name, resident, pt / GiB, pto / GiB)
        for r in reads:
            line += "  %6.0f" % (r * 1e9 / pt)
        print(line + ("" if resident <= 30.0 else "   <- above the ~32 GiB budget"))
    print("\nthe same ceilings for the target-only row (no draft pass charged per token):")
    hdr = "%-34s %12s" % ("policy", "GiB/tok-tgt")
    hdr += "".join("  %6s" % ("%d GB/s" % r) for r in reads)
    print(hdr)
    print("-" * len(hdr))
    for name in POLICY_TYPES:
        rows, draft = policy_rows(pop, name)
        pto = per_token(pop, rows, draft, False)
        line = "%-34s %12.4f" % (name, pto / GiB)
        for r in reads:
            line += "  %6.0f" % (r * 1e9 / pto)
        print(line)
    print("\nBPW used (derived, %s):" % BPW_NOTE)
    for t in sorted(widths, key=lambda x: -widths[x]):
        print("  %-8s %s bpw%s" % (t, ("%.4f" % widths[t]).rstrip("0").rstrip("."),
                                   "   <- not used by any policy row" if t == "Q5_K" else ""))
    print("\nPolicy rows use Q4_K/Q8_0/bf16/IQ4_XS/MXFP4, and Q6_K only for lm_head (embeddings follow")
    print("lm_head) in E and G. No row uses Q5_K, so the Q5_K width correction (5.5625 was hand-typed")
    print("where the struct derives 5.5) changes no figure in this table; anyone tempted to \"fix\" a")
    print("policy row with the corrected width has fixed a row that was never Q5_K.")
    print("Policy A charges the exact bucket (routers, norms, the GDN 1-D controls) bf16, as this file")
    print("always has; charging it F32 as the published artifact stores it is +0.0403 GiB resident and")
    print("+0.0403 GiB/token and is an open decision (adjudication §5 items 6-7, §10.2).")
    return 0


# ---------------------------------------------------------------------------------------------
# selftest: injected faults, each one required to fire
#
# The rule this exists to satisfy (AGENTS.md §"A check that can produce no output cannot fail"): a
# counter reading 0 is vacuous until it has been witnessed non-zero where it must fire. So every fault
# below prints the injected case *and* the caught discrepancy, the clean run prints its own population
# witnesses before any PASS can mean anything, and a fault that cannot be injected is a MISS rather than
# a disguised pass.
# ---------------------------------------------------------------------------------------------
def selftest():
    print("quant_scenarios --selftest — the five census defects of quantization/evidence/"
          "bf16-census-adjudication-2026-10-26.md §4/§5, plus a short parse, an empty census and a")
    print("required figure that stops matching. Each fault must be caught, by name, at its own size.")
    fired, missed = [], []

    def expect(name, fn, why):
        """Fired = the tool refused, or returned a note naming the discrepancy it caught."""
        try:
            note = fn()
        except FaultInjectionError as e:
            missed.append("%s could not be injected: %s" % (name, e))
            return
        except (CensusFault, AssertionError, ValueError) as e:
            fired.append("%s fired: %s" % (name, str(e).splitlines()[0][:150]))
            return
        if note:
            fired.append("%s fired: %s" % (name, note))
        else:
            missed.append("%s did NOT fire — %s" % (name, why))

    pop = build_population()
    _, res, sc = closures(pop)
    numbers = derived_numbers(pop)
    drift = check_required(numbers)
    g = pop["g"]
    nlin = sum(1 for x in g["layer_types"] if x == "linear_attention")
    print("  clean run: text %.4f GiB, artifact %.4f GiB, draft %.4f GiB, vision %.4f GiB; policy A "
          "%.4f GiB resident, %.4f GiB/token (target-only %.4f)"
          % (numbers["text_gib"], numbers["artifact_gib"], numbers["draft_gib"],
             numbers["vision_gib"], numbers["A_resident"], numbers["A_per_token"],
             numbers["A_per_token_only"]))
    print("  witnesses that make a 0 above mean something: %d artifact records (header declares %d), "
          "%d index names of which %d are shape-witnessed by real shard headers, %d draft names, "
          "%d text families, %d linear layers, %d phantom elts"
          % (pop["n_art"], pop["decl_tensors"], pop["n_index"], pop["n_wit"], len(pop["draft"]),
             len(TEXT_FAMILIES), nlin, sc["phantom"]))
    if res:
        missed.append("clean run left residues: %s" % sorted(res.items()))
    for row in drift:
        missed.append("clean run drifted from the adjudication: %s" % row)
    if sc["phantom"]:
        missed.append("clean run carries %d phantom elements" % sc["phantom"])
    if not pop["draft"] or not sc["vision"] or not pop["n_wit"]:
        missed.append("clean run is vacuous: draft names, vision elements or shard witnesses read 0")

    def fault_check(fault):
        """Inject one census defect; require the closures to catch it at exactly the injected size."""
        pf = build_population(faults=(fault,))
        _, resf, scf = closures(pf)
        inj = {f: (fam, delta, what) for f, fam, delta, what in pf["injected"]}
        if fault not in inj:
            return None
        fam, want, what = inj[fault]
        got = scf["fam_res"].get(fam, 0)
        if not resf or got != want:
            return None
        return ("injected %s (%s); caught as a %s residue of %+d elts (%+.5f GiB bf16) and %d "
                "non-zero closure residue(s)" % (fault, what, fam, got, got * BF16_BYTES / GiB,
                                                 len(resf)))

    for fault in ("drop-gdn-out-proj", "drop-mtp-fc", "phantom-9h2", "phantom-2h-per-linear",
                  "norm-count-short"):
        expect(fault, (lambda f=fault: fault_check(f)),
               "the census-vs-index closures stayed at 0 with %s injected" % fault)

    expect("short-record", lambda: build_population(faults=("short-record",)),
           STRUCT_FAULTS["short-record"] + "; the header's own record count did not notice")
    expect("empty-census", lambda: build_population(faults=("empty-census",)),
           STRUCT_FAULTS["empty-census"] + "; a zero total was printed instead of a refusal")

    def figure_drift():
        bad = dict(numbers)
        bad["A_per_token"] = 2.331          # the figure this file printed on the broken census
        rows = check_required(bad)
        return ("required-figure check rejected the pre-fix 2.331 GiB/token: %s" % rows[0]) if rows \
            else None
    expect("required-figure-drift", figure_drift,
           "the required-figure check could not notice a wrong policy-A row")

    for ln in fired:
        print("  PASS " + ln)
    for ln in missed:
        print("  FAIL " + ln)
    want_n = len(FAULTS) + len(STRUCT_FAULTS) + 1
    print("selftest: %d faults injected, %d fired, %d missed — %s"
          % (len(fired) + len(missed), len(fired), len(missed),
             "PASS" if not missed and len(fired) == want_n else "FAIL"))
    return 0 if (not missed and len(fired) == want_n) else 1


def main():
    ap = argparse.ArgumentParser(description="derived bf16 census + policy residency/ceiling arithmetic")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="inject the five census defects (plus a short parse, an empty census and a "
                         "figure drift); each must fire")
    ap.add_argument("--read", type=float, nargs="+", default=[],
                    help="GB/s figures to price against (one flag, space-separated list)")
    args = ap.parse_args()
    reads = args.read or list(READS)

    if args.selftest:
        try:
            return selftest()
        except CensusFault as e:
            print("ABORT census: %s" % e)
            return 1
    try:
        pop = build_population()
    except CensusFault as e:
        print("ABORT census: %s" % e)
        return 1
    try:
        if args.census:
            res = print_census(pop)
            numbers = derived_numbers(pop)
            drift = check_required(numbers)
            print("\nrequired figures, published in quantization/evidence/bf16-census-adjudication-"
                  "2026-10-26.md §3/§6:")
            for label, key, quoted, tol in REQUIRED_FIGURES:
                bad = any(line.startswith(label) for line in drift)
                print("  %-56s derived %9.4f  published %8s  %s"
                      % (label, numbers[key], quoted, "DRIFT" if bad else "match"))
            if drift or res:
                for ln in drift:
                    print("  DRIFT %s" % ln)
                print("\nABORT: this tool's derivation disagrees with the adjudication, or a closure "
                      "left a residue. Do not adopt this tool's number and do not edit the evidence "
                      "file — report the discrepancy.")
                return 1
            print("  all %d required figures agree with the adjudication within their tolerance"
                  % len(REQUIRED_FIGURES))
            return 0
        return print_policies(pop, reads)
    except CensusFault as e:
        print("ABORT census: %s" % e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
