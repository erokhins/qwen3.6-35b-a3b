#!/usr/bin/env python3
"""layout.py — where policy A's bytes go, and which budget "it fits" is a claim about.

Two questions get answered here and nowhere else in the engine:

1. **The residency layout.** Policy A ships as a published quant, so the bytes that actually become
   resident are `artifact.py`'s records — 733 from the GGUF plus the 19 draft tensors from the HF
   shards. They are laid into a *small number of slabs*, not 752 allocations:
   `pitfalls/10-device-memory.md` §"Allocation recipe that is known to work on this box" says one
   slab for weights and one for state and "never per-tensor allocations in the decode loop", and
   `quantization/residency-layout.md` §7 item 2 records that allocation *count* at 712 allocations
   has never been probed on this box (UNMEASURED). So the layout's job is to make the per-tensor
   view free and the allocation count small, and to print the copy plan (maximal runs that one
   `hipMemcpy` can carry) so the bulk-copy recipe has something to consume.

2. **Which budget.** D-11 split residency into two regimes and the repo's rule is that a residency
   figure that does not say which one it belongs to is ambiguous, not imprecise. `BUDGETS` spells
   them; `verdict()` raises rather than printing "fits" if a caller leaves the budget unnamed. The
   two figures that must never be conflated are also spelled out here: policy A *models* at
   19.7848 GiB of weights, and the measured llama.cpp engine demand against the same artifact was
   ≈26.9 GiB (`devices/evidence/device-window-2026-10-26.md` §2, `mem_info_gtt_used` 26.86-26.88 GiB
   over 5 runs, corroborated by a 27.33 GiB fall in host `MemAvailable`), i.e. **≥6.24 GiB of
   non-weight footprint that the modelled figure does not contain.**

Byte identities this module is obliged to close at 0, all against tools that already exist:

  * `sum(row_bytes) + data_start == file size`, and per-family bytes identical to
    `tools/gguf_width_table.py`'s independent parse (done in `artifact.py`, re-checked here through
    the layout so a slab bug cannot hide it);
  * policy A resident identical to `tools/quant_scenarios.py`'s `derived_numbers()["A_resident"]`
    **in bytes**, not to 4 decimals;
  * the resident element census identical to `tools/bf16_census.py` /
    `tools/residency_budget.py` (66.1337 GiB bf16 with the draft block, 64.5604 without);
  * the gap between this layout's policy-A figure and `tools/residency_budget.py`'s 19.7856 GiB
    attributed in full to the one documented GDN-control width convention, residue 0 B
    (`tools/residency_budget.py` §"The 19.7856 / 19.7848 GiB difference"). A residue that is not
    attributed is a mismatch, and a mismatch aborts.

CPU-only. No timer, no device call, no ms or tok/s printed.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
for _p in (TOOLS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bf16_census as BC                 # noqa: E402
import ggml_block_geom as GBG            # noqa: E402
import quant_scenarios as QS             # noqa: E402
import residency_budget as RB            # noqa: E402
import artifact as ART                   # noqa: E402

GiB = 2.0 ** 30
MiB = 2.0 ** 20
POLICY_A = "A experts Q4_K, rest Q8_0"          # quant_scenarios' own key for the decided policy
POLICY_A_RB = "A (DECIDED)"                      # residency_budget's own key for the same decision

# -----------------------------------------------------------------------------------------------
# the two budgets (D-11). Names, not numbers typed here: the brackets come from
# tools/residency_budget.py, which reads them off devices/measured-facts.md §"Device-visible
# memory" and §"Residency above the pool".
# -----------------------------------------------------------------------------------------------
BUDGETS = {
    "development": dict(
        lo=RB.CO_LO, hi=RB.CO_HI,
        label="development, co-resident with the local serving container (halogen up)",
        citation="tools/residency_budget.py CO_LO/CO_HI <- devices/measured-facts.md "
                 "§Device-visible memory + §Residency above the pool (pool 55.5-56.5 GiB minus "
                 "halogen's 22.62-23.66 GiB footprint). DECISIONS.md D-11: development on this box "
                 "is co-resident almost by definition, because D-15 keeps a worker on halogen.",
        rule="Design against 31.84 GiB, the pessimistic end. A row inside 31.84-33.88 GiB has no "
             "determinate answer today and must be treated as NOT fitting "
             "(quantization/residency-layout.md §1)."),
    "deployment": dict(
        lo=RB.DEDICATED_LO, hi=None,
        label="deployment, with the serving container stopped (D-11: the engine owns the box)",
        citation="tools/residency_budget.py DEDICATED_LO <- devices/measured-facts.md "
                 "§Device-visible memory; DECISIONS.md D-11 (user, 2026-10-26: \"I propose to stop "
                 "halogen when we will be running our engine\").",
        rule=">=55.5 GiB is an inference from D-02's allocation probes, not an observation: the "
             "48 GiB single-hipMalloc probe is still queued (quantization/residency-layout.md §7 "
             "item 1). Say \"expected\", never \"measured\", for this budget."),
}

# The measured footprint of an engine running this artifact, and what it implies about the
# non-weight half of residency. Quoted, cited, and NOT produced here.
MEASURED_ENGINE = dict(
    gtt_peak_gib=26.9, gtt_range="26.86-26.88 GiB over 5 runs",
    file_gib=20.624, instrument="mem_info_gtt_used, corroborated by host MemAvailable (114.80 -> 87.47 GiB)",
    source="devices/evidence/device-window-2026-10-26.md §2",
    condition="llama.cpp, shipping artifact, 41/41 layers offloaded, n_ctx=26214, halogen down",
)
MEASURED_NON_WEIGHT_FLOOR_GIB = MEASURED_ENGINE["gtt_peak_gib"] - MEASURED_ENGINE["file_gib"]

# Which slab a tensor belongs to, and why that grouping. Six slabs for 752 tensors, which is the
# point: the count is what this box has never been probed at (residency-layout §7 item 2).
SLAB_ORDER = ("weights.experts", "weights.attention", "weights.shared", "weights.tables",
              "weights.exact", "weights.draft")
SLAB_WHY = {
    "weights.experts": "routed gate/up (Q4_K) and down (Q5_K/Q6_K). 82.6 % of the file's bytes and "
                       "the only family read sparsely (8 of 256 per token), so it is the slab an "
                       "expert-placement or streaming policy acts on.",
    "weights.attention": "full attention (attn_q/k/v/output) and the GDN matrices (attn_qkv, "
                         "attn_gate, ssm_out), all Q8_0. Read in full by every token.",
    "weights.shared": "the shared expert's gate/up/down, Q8_0. Read in full by every token.",
    "weights.tables": "token_embd (Q8_0) and the untied output table (Q6_K). One row per token from "
                      "the first, the whole table per token from the second (AGENTS.md prediction 5).",
    "weights.exact": "everything the artifact stores unquantized: the router (ffn_gate_inp), the "
                     "shared-expert gate, every norm, and the GDN controls (ssm_conv1d, ssm_a, "
                     "ssm_dt, ssm_norm, ssm_alpha, ssm_beta). Small, and a 1-ULP change here is not "
                     "a rounding difference (quantization/README.md §\"How a quantized path gets "
                     "gated\" item 3), so it stays byte-exact and gets its own slab.",
    "weights.draft": "the MTP draft block. NOT in the GGUF (zero mtp.* tensors); its bytes come from "
                     "the HF shards, which store them bf16 — see NOT WIRED in loader.py.",
}
# Roles that must be stored unquantized. Anything F32 outside this set, or any quantized tensor
# inside it, is a layout error: it means the never-quantized bucket and the exact slab disagree.
EXACT_ROLES = {
    "blk.N.attn_norm.weight", "blk.N.post_attention_norm.weight", "output_norm.weight",
    "blk.N.attn_q_norm.weight", "blk.N.attn_k_norm.weight",
    "blk.N.ffn_gate_inp.weight", "blk.N.ffn_gate_inp_shexp.weight",
    "blk.N.ssm_conv1d.weight", "blk.N.ssm_a", "blk.N.ssm_dt.bias", "blk.N.ssm_norm.weight",
    "blk.N.ssm_alpha.weight", "blk.N.ssm_beta.weight",
}
FAMILIES_TO_SLAB = {"routed_experts": "weights.experts", "full_attn": "weights.attention",
                    "gdn_linear_attn": "weights.attention", "shared_experts": "weights.shared",
                    "embed": "weights.tables", "lm_head": "weights.tables"}
READ_CLASSES = ("per_token_all", "per_token_gathered", "one_row")


class LayoutError(Exception):
    """A byte identity did not close, or a caller asked for something this module cannot honour."""


def slab_of(rec):
    if rec.source == "hf_shards":
        return "weights.draft"
    if rec.type_name in ("F32", "F16", "BF16") and rec.role in EXACT_ROLES:
        return "weights.exact"
    if rec.type_name in ("F32", "F16"):
        raise LayoutError("%s is stored %s but is not in the never-quantized role set %s: the exact "
                          "slab and the artifact's own stored widths disagree, so either the width "
                          "map moved or the role set is stale."
                          % (rec.name, rec.type_name, "quantization/README.md §How-a-quantized-path-"
                             "gets-gated item 3"))
    if rec.role in EXACT_ROLES:
        raise LayoutError("%s is a never-quantized role but is stored %s; the router, the norms and "
                          "the GDN controls stay exact "
                          "(quantization/README.md §How a quantized path gets gated item 3)"
                          % (rec.name, rec.type_name))
    try:
        return FAMILIES_TO_SLAB[rec.family]
    except KeyError:
        raise LayoutError("no slab for family %r (tensor %s): a family with no slab is bytes that go "
                          "nowhere and still print a total" % (rec.family, rec.name))


def read_class_of(rec):
    """Who reads a tensor per decoded token. The rule is `tools/quant_scenarios.py`'s `per_token()`,
    restated as a tag on the record so a mis-tagged family shows up in the layout and not only in a
    rate figure nobody is allowed to print here."""
    if rec.source == "hf_shards":
        return "per_token_all"                # D-05: the draft block is walked on a draft pass
    if rec.family == "routed_experts":
        return "per_token_gathered"
    if rec.family == "embed":
        return "one_row"
    return "per_token_all"


class Placement:
    __slots__ = ("rec", "slab", "slab_off", "offsets", "read_class", "policy_type",
                 "policy_row")

    def __init__(self, rec, slab):
        self.rec = rec
        self.slab = slab
        self.slab_off = None
        self.offsets = None                   # (slab_off, length) once the slab is closed
        self.read_class = read_class_of(rec)
        self.policy_type = None
        self.policy_row = None

    def __repr__(self):
        return "<Placement %s %s slab=%s off=%s len=%d>" % (self.rec.name, self.rec.type_name,
                                                            self.slab, self.slab_off, self.rec.row)


class Slab:
    """One contiguous destination buffer. The allocator's unit, and the unit the copy plan is cut on."""
    __slots__ = ("name", "placements", "bytes", "content_bytes", "pad_bytes", "runs", "source")

    def __init__(self, name):
        self.name = name
        self.placements = []
        self.bytes = 0
        self.content_bytes = 0
        self.pad_bytes = 0
        self.runs = []
        self.source = None

    def __repr__(self):
        return "<Slab %s n=%d bytes=%d pad=%d runs=%d>" % (self.name, len(self.placements),
                                                           self.bytes, self.pad_bytes, len(self.runs))


class ResidencyLayout:
    """The placement of every weight tensor, plus the closures that prove the placement is honest.

    `align` is the artifact's own `general.alignment` (32): a quantized tensor's row must start on it
    for the block grid to stay addressable, so the layout uses the file's number rather than inventing
    a device-side one. Larger device alignment is the allocator's business and is asserted where the
    slab is actually allocated (loader.py, stubbed)."""

    CLOSURES = ("row-bytes-vs-ruler", "file-size-identity", "slab-coverage", "run-coverage",
                "no-overlap", "element-census-vs-census", "policy-a-vs-quant_scenarios",
                "policy-a-vs-residency_budget", "expert-slices", "read-class-coverage",
                "exact-slab-is-never-quantized", "draft-block-size")

    def __init__(self, art=None, align=None, faults=()):
        self.art = art if art is not None else ART.open_artifact()
        self.align = align or self.art.alignment
        self.faults = set(faults)
        if self.align <= 0:
            raise LayoutError("alignment %r is not a positive byte count" % (self.align,))
        self.placements = []
        self.slabs = {}
        self._build()
        self._close_slabs()
        # faults land AFTER the slabs are closed: injected before, _close_slabs would recompute the
        # offsets and every closure would pass over the very number the fault moved.
        self._apply_faults()
        self.closure_results = None

    # ---- build -----------------------------------------------------------------------------
    def _build(self):
        for rec in list(self.art.records) + list(self.art.draft):
            s = slab_of(rec)
            p = Placement(rec, s)
            if rec.source == "gguf":
                p.policy_type = QS.POLICY_TYPE(POLICY_A, rec.family)
            else:
                p.policy_type = QS.POLICY_TYPE(POLICY_A, rec.family)
            p.policy_row = GBG.row_bytes(rec.n_elts, p.policy_type)
            self.slabs.setdefault(s, Slab(s)).placements.append(p)
            self.placements.append(p)
        missing = [s for s in SLAB_ORDER if s not in self.slabs]
        if missing:
            raise LayoutError("no tensor was placed in %s: an empty slab is a family that vanished "
                              "from the layout while the totals still printed" % missing)
        extra = [s for s in self.slabs if s not in SLAB_ORDER]
        if extra:
            raise LayoutError("unexpected slab(s) %s" % extra)
        for s in self.slabs.values():
            s.source = s.placements[0].rec.source
            if any(p.rec.source != s.source for p in s.placements):
                raise LayoutError("slab %s mixes sources %s; a slab is staged from one file, so "
                                  "mixing them makes the bulk-copy plan a lie"
                                  % (s.name, sorted({p.rec.source for p in s.placements})))
        # every slab keeps file order, so staging is ascending reads
        for s in self.slabs.values():
            s.placements.sort(key=lambda p: (p.rec.file_offset, p.rec.name))

    def _close_slabs(self):
        for s in self.slabs.values():
            off = 0
            for p in s.placements:
                pad = (-off) % self.align
                off += pad
                s.pad_bytes += pad
                p.slab_off = off
                p.offsets = (off, p.rec.row)
                off += p.rec.row
            s.bytes = off
            s.content_bytes = sum(p.rec.row for p in s.placements)
            if s.bytes - s.content_bytes != s.pad_bytes:
                raise LayoutError("slab %s: bytes %d - content %d != accounted padding %d; padding "
                                  "that is not accounted is bytes no one can explain"
                                  % (s.name, s.bytes, s.content_bytes, s.pad_bytes))
            s.runs = self._runs(s)

    @staticmethod
    def _runs(s):
        """Maximal runs: consecutive tensors that are consecutive in the slab AND in their source file.

        This is the bulk-copy plan `pitfalls/10-device-memory.md` §"Allocation recipe" wants — one
        `hipMemcpy` per run instead of one per tensor — and the reason it is printed rather than
        assumed is that the number of memcpys is a property of the layout, not of the intent."""
        runs = []
        for p in s.placements:
            if runs:
                last = runs[-1]
                if (last["file_end"] == p.rec.file_offset
                        and last["slab_end"] == p.slab_off):
                    last["length"] += p.rec.row
                    last["file_end"] = p.rec.file_offset + p.rec.row
                    last["slab_end"] = p.slab_off + p.rec.row
                    last["n"] += 1
                    continue
            runs.append(dict(slab=s.name, slab_off=p.slab_off, file_off=p.rec.file_offset,
                             length=p.rec.row, file_end=p.rec.file_offset + p.rec.row,
                             slab_end=p.slab_off + p.rec.row, n=1))
        return runs

    # ---- injections ------------------------------------------------------------------------
    def _apply_faults(self):
        for f in sorted(self.faults):
            fn = FAULTS.get(f)
            if fn is None:
                raise LayoutError("unknown layout fault %r; known: %s" % (f, sorted(FAULTS)))
            fn(self)

    # ---- views -----------------------------------------------------------------------------
    def placement(self, name):
        for p in self.placements:
            if p.rec.name == name:
                return p
        raise LayoutError("no placement for %r" % name)

    def slab(self, name):
        if name not in self.slabs:
            raise LayoutError("no slab %r; slabs are %s" % (name, SLAB_ORDER))
        return self.slabs[name]

    def copy_plan(self):
        runs = []
        for s in SLAB_ORDER:
            runs.extend(self.slabs[s].runs)
        return runs

    def expert_slice(self, name, expert):
        """Byte range of one expert's matrix inside a `[*, *, E]` GGUF record.

        GGUF order has dims[0] fastest, so the expert axis — the trailing HF axis — is the SLOWEST
        one, and expert e occupies the contiguous element range [e*inner, (e+1)*inner). That is only
        addressable when `inner` is a whole number of ggml blocks; when it is not, the slice would
        split a block and this raises instead of handing back a half-block. The closure that the E
        slices tile the row exactly is `expert-slices` below, and `expert_stride_fused` in the gate
        injects the `[E,2I,H]` stride (AGENTS.md prediction 3: the fused gate|up pair is laid out
        along the OUTPUT dim, so a kernel that takes the fused stride for a split GGUF record is
        wrong, and this is the check that says so)."""
        rec = self.art.by_name.get(name)
        if rec is None:
            for r in self.art.draft:
                if r.name == name:
                    rec = r
                    break
        if rec is None:
            raise LayoutError("no tensor %r" % name)
        if not rec.dims or rec.dims[-1] <= 1:
            raise LayoutError("%s has trailing dim %s, which is not an expert axis"
                              % (name, rec.dims[-1] if rec.dims else "()"))
        e = int(rec.dims[-1])
        if not 0 <= expert < e:
            raise LayoutError("expert %d out of range for %s (%d experts)" % (expert, name, e))
        inner = 1
        for d in rec.dims[:-1]:
            inner *= int(d)
        return self._axis_slice(rec, inner, expert, e, name)

    def _axis_slice(self, rec, inner, idx, n, name):
        elts_per_block = GBG.block_elts(rec.type_name)
        if inner % elts_per_block:
            raise LayoutError("%s: a slice is %d elements, which is not a whole number of %s blocks "
                              "of %d; the slice would split a quantized block and dequantize into "
                              "garbage" % (name, inner, rec.type_name, elts_per_block))
        row = GBG.row_bytes(inner, rec.type_name)
        if row * n != rec.row:
            raise LayoutError("%s: %d slices of %d B = %d B, the record's row is %d B. Either the "
                              "axis being split is not the expert axis or the block geometry does "
                              "not divide the tensor." % (name, n, row, row * n, rec.row))
        return dict(name=name, index=idx, n_slices=n, file_off=rec.file_offset + idx * row,
                    slab_off=self.placement(name).slab_off + idx * row, length=row,
                    elts=inner, type_name=rec.type_name)

    # ---- closures --------------------------------------------------------------------------
    def closures(self):
        """Every byte identity, as (name, ok, n, detail). `run_closures()` turns a non-closure into abort.

        Nothing here re-derives a width, a shape or a policy: widths come from
        `tools/ggml_block_geom.py`, the ruler is `tools/gguf_width_table.py`, the census is
        `tools/bf16_census.py`, the policy arithmetic is `tools/quant_scenarios.py` and the residency
        is `tools/residency_budget.py`. This function only asks whether what the layout produced
        agrees with them, to the byte."""
        out = []
        art = self.art
        # C1 the ruler's per-family bytes vs the layout's
        ruler = {k: v[1] for k, v in art.width_table_family.items()}
        mine = {}
        for p in self.placements:
            if p.rec.source == "gguf":
                mine[p.rec.type_name] = mine.get(p.rec.type_name, 0) + p.rec.row
        drift = {t: mine.get(t, 0) - ruler.get(t, 0) for t in sorted(set(mine) | set(ruler))}
        out.append(("row-bytes-vs-ruler", all(v == 0 for v in drift.values()), len(ruler),
                    "per-family bytes vs tools/gguf_width_table.py: %s"
                    % ("identical on every type" if not any(drift.values())
                       else "DRIFT %s" % {k: v for k, v in drift.items() if v})))
        # C2 file size
        stored = sum(p.rec.row for p in self.placements if p.rec.source == "gguf")
        out.append(("file-size-identity", stored + art.data_start == art.size, 1,
                    "sum(row_bytes) %d + data_start %d vs file %d: delta %+d B"
                    % (stored, art.data_start, art.size, art.size - (stored + art.data_start))))
        # C3 slab coverage
        slab_bytes = sum(s.bytes for s in self.slabs.values())
        content = sum(p.rec.row for p in self.placements)
        pad = sum(s.pad_bytes for s in self.slabs.values())
        out.append(("slab-coverage", slab_bytes == content + pad, len(self.slabs),
                    "slabs %d B == content %d B + accounted padding %d B (residue %+d B); padding is "
                    "%0.4f MiB, %.5f %% of the weights"
                    % (slab_bytes, content, pad, slab_bytes - content - pad, pad / MiB,
                       100.0 * pad / max(1, content))))
        # C4 copy plan covers every byte exactly once
        plan = self.copy_plan()
        planned = sum(r["length"] for r in plan)
        per_slab_ok = True
        for s in self.slabs.values():
            covered = sum(r["length"] for r in s.runs)
            offs = sorted((r["slab_off"], r["slab_off"] + r["length"]) for r in s.runs)
            for i in range(1, len(offs)):
                if offs[i][0] < offs[i - 1][1]:
                    per_slab_ok = False
            if covered != s.content_bytes:
                per_slab_ok = False
        out.append(("run-coverage", planned == content and per_slab_ok, len(plan),
                    "copy plan carries %d run(s), %d B, against %d B of tensor content; runs are "
                    "disjoint and inside their slabs: %s"
                    % (len(plan), planned, content, per_slab_ok)))
        # C5 no two placements share slab bytes
        seen, overlap = {}, []
        for p in self.placements:
            key = (p.slab, p.slab_off // self.align)
            if key in seen:
                overlap.append("%s and %s both start in block %d of %s"
                               % (seen[key], p.rec.name, key[1], p.slab))
            seen[key] = p.rec.name
        out.append(("no-overlap", not overlap, len(self.placements),
                    "no two tensors share an aligned block of a slab" if not overlap
                    else "OVERLAP %s" % overlap[:3]))
        # C6 element census vs the authoritative census
        pop = QS.build_population()
        _, _, sc = QS.closures(pop)
        art_elts = sum(p.rec.n_elts for p in self.placements if p.rec.source == "gguf")
        draft_elts = sum(p.rec.n_elts for p in self.placements if p.rec.source == "hf_shards")
        out.append(("element-census-vs-census",
                    art_elts == sc["art"] and draft_elts == sc["draft"], 3,
                    "layout %d gguf elements + %d draft elements vs tools/bf16_census.py via "
                    "quant_scenarios %d + %d: delta %+d / %+d elements (%+d B / %+d B bf16); text "
                    "bf16 %.4f GiB incl. draft, %.4f GiB without"
                    % (art_elts, draft_elts, sc["art"], sc["draft"], art_elts - sc["art"],
                       draft_elts - sc["draft"], (art_elts - sc["art"]) * 2,
                       (draft_elts - sc["draft"]) * 2,
                       (art_elts + draft_elts) * 2 / GiB, art_elts * 2 / GiB)))
        # C7 policy A, to the byte, against quant_scenarios
        mine_a_main = sum(p.policy_row for p in self.placements if p.rec.source == "gguf")
        mine_a_draft = sum(p.policy_row for p in self.placements if p.rec.source == "hf_shards")
        rows, draft_rows = QS.policy_rows(pop, POLICY_A)
        qs_main, qs_draft = sum(rows.values()), sum(draft_rows.values())
        out.append(("policy-a-vs-quant_scenarios",
                    mine_a_main == qs_main and mine_a_draft == qs_draft, len(rows) + len(draft_rows),
                    "layout prices policy A at %d B (main) + %d B (draft) = %.4f GiB; "
                    "tools/quant_scenarios.py prices %d B + %d B = %.4f GiB; delta %+d B / %+d B"
                    % (mine_a_main, mine_a_draft, (mine_a_main + mine_a_draft) / GiB,
                       qs_main, qs_draft, (qs_main + qs_draft) / GiB,
                       mine_a_main - qs_main, mine_a_draft - qs_draft)))
        # C8 policy A against residency_budget, with the one documented convention attributed
        src = RB.Sources()
        b = RB.Budget(src, policy=POLICY_A_RB)
        rb_main, rb_draft = b.weight_bytes()
        attn_bits = b.width_bits("gdn_matrices")
        ctrl_bits = b.width_bits("gdn_controls")
        ctrl_el = b.census["gdn_controls"]
        convention = ctrl_el * (ctrl_bits - attn_bits) / 8.0
        gap_main = mine_a_main - rb_main
        gap_draft = mine_a_draft - rb_draft
        # residency_budget charges the 1-D GDN controls at its own width, so it sits ABOVE the
        # quant_scenarios policy by exactly the convention term. The direction is part of the claim.
        residue = gap_main + convention
        out.append(("policy-a-vs-residency_budget",
                    abs(residue) < 1.0 and gap_draft == 0 and ctrl_el > 0 and convention > 0, 2,
                    "layout minus tools/residency_budget.py: main %+d B, draft %+d B; the whole main "
                    "gap must be the GDN 1-D control convention (%d elements x (%g-%g) bpw / 8 = "
                    "%.0f B), unattributed %.0f B"
                    % (gap_main, gap_draft, ctrl_el, ctrl_bits, attn_bits, convention, residue)))
        # C9 expert slices tile every routed record exactly
        checked, bad = 0, []
        for p in self.placements:
            if p.rec.family != "routed_experts":
                continue
            inner = 1
            for d in p.rec.dims[:-1]:
                inner *= int(d)
            e = int(p.rec.dims[-1])
            try:
                self._axis_slice(p.rec, inner, 0, e, p.rec.name)
            except LayoutError as ex:
                bad.append(str(ex))
                continue
            checked += 1
            seen_off = []
            for i in (0, 1, e // 2, e - 2, e - 1):
                sl = self._axis_slice(p.rec, inner, i, e, p.rec.name)
                want_off = p.rec.file_offset + i * sl["length"]
                if sl["file_off"] != want_off or sl["length"] * e != p.rec.row:
                    bad.append("%s expert %d: file_off %d, expected %d; slice %d B x %d != row %d"
                               % (p.rec.name, i, sl["file_off"], want_off, sl["length"], e, p.rec.row))
                seen_off.append((sl["slab_off"], sl["slab_off"] + sl["length"]))
            for i in range(1, len(seen_off)):
                if seen_off[i][0] < seen_off[i - 1][1]:
                    bad.append("%s: sampled expert slices overlap at %s" % (p.rec.name, seen_off[i]))
        out.append(("expert-slices", not bad and checked > 0, checked,
                    "%d routed records: the %d expert slices tile each row exactly and each slice is "
                    "a whole number of %s blocks" % (checked, 256, "Q4_K/Q5_K/Q6_K")
                    if not bad else "BAD %s" % bad[:3]))
        # C10 every read class is populated (a family that silently stops being read is the bug)
        counts = {c: 0 for c in READ_CLASSES}
        for p in self.placements:
            counts[p.read_class] = counts.get(p.read_class, 0) + 1
        out.append(("read-class-coverage", all(counts.get(c, 0) > 0 for c in READ_CLASSES),
                    len(counts), "read classes: %s" % counts))
        # C11 the exact slab holds exactly the never-quantized roles
        exact_roles = {p.rec.role for p in self.placements if p.slab == "weights.exact"}
        stray = sorted(r for r in exact_roles
                       if r not in EXACT_ROLES)
        quantized = sorted({p.rec.name for p in self.placements
                            if p.slab == "weights.exact" and p.rec.type_name not in ("F32", "F16")})
        out.append(("exact-slab-is-never-quantized",
                    not stray and not quantized and len(exact_roles) == len(EXACT_ROLES),
                    len(exact_roles),
                    "exact slab carries %d roles, all in the never-quantized set; %d quantized "
                    "tensor(s) inside it" % (len(exact_roles), len(quantized))))
        # C12 draft block size against the pin's authority
        want = int(self.art.pin["authority"]["mtp_elements"]) if self.art.pin else draft_elts
        out.append(("draft-block-size", draft_elts == want and draft_elts > 0, len(self.slabs["weights.draft"].placements),
                    "draft block %d elements vs pin authority %s (%s); %d B bf16 on disk, %.4f GiB "
                    "at policy-A widths" % (draft_elts, want, ART.MTP_AUTHORITY,
                                            sum(p.rec.row for p in self.slabs["weights.draft"].placements),
                                            mine_a_draft / GiB)))
        return out

    def run_closures(self):
        res = self.closures()
        self.closure_results = res
        bad = [r for r in res if not r[1]]
        if bad:
            raise LayoutError("the layout does not close; refusing to print a table over it:\n  %s"
                              % "\n  ".join("%s: %s" % (r[0], r[3]) for r in bad))
        return res

    # ---- budgets ---------------------------------------------------------------------------
    def weights_total_bytes(self, view="as-stored"):
        """`as-stored` is what becomes resident when this artifact is loaded; `policy-A` is the width
        assignment we would build if we converted from f16 (quantization/README.md §status). They are
        different numbers and quoting one as the other is the conflation this function exists to stop."""
        if view == "as-stored":
            return sum(p.rec.row for p in self.placements)
        if view == "policy-A":
            return sum(p.policy_row for p in self.placements)
        raise LayoutError("unknown view %r; the two views are 'as-stored' and 'policy-A'" % (view,))

    def verdict(self, resident_gib, budget, basis, ctx=4096, nseq=1):
        """The only legal way to print a fit claim. `budget` must name a regime; `basis` must name what
        the number contains, because \"policy A fits\" has been said about 19.78 GiB of weights while
        the engine measured 26.9 GiB of footprint (devices/evidence/device-window-2026-10-26.md §2)."""
        if budget not in BUDGETS:
            raise LayoutError(
                "residency claim without a named budget: pass budget='development' (co-resident with "
                "the serving container, %s) or budget='deployment' (that container stopped, %s). "
                "D-11: a residency figure that does not say which budget it belongs to is ambiguous, "
                "not merely imprecise." % (BUDGETS_TEXT["development"], BUDGETS_TEXT["deployment"]))
        if basis not in BASES:
            raise LayoutError("residency claim without a named basis: pass one of %s. The modelled "
                              "weight figure and the measured engine footprint differ by >=6.24 GiB, "
                              "so which one is being quoted is part of the claim."
                              % (sorted(BASES),))
        bd = BUDGETS[budget]
        allowance = BASES[basis](self, ctx=ctx, nseq=nseq)
        total = resident_gib + allowance["extra_gib"]
        lines = ["budget=%s (%s)" % (budget, bd["label"]),
                 "basis=%s" % basis,
                 "resident=%.4f GiB + allowance=%.4f GiB = %.4f GiB" % (resident_gib,
                                                                       allowance["extra_gib"], total),
                 "budget bound=%s GiB" % ("%.2f (pessimistic; bracket to %.2f)" % (bd["lo"], bd["hi"])
                                          if bd["hi"] else ">=%.2f GiB, UNMEASURED (D-02 probes)"
                                          % bd["lo"]),
                 "verdict=%s" % RB.verdict(total)]
        for k, v in sorted(allowance["detail"].items()):
            lines.append("  %s=%s" % (k, v))
        return total, lines

    def report(self, ctx=4096, nseq=1):
        """The residency table. Every printed row carries its budget and its basis, and the gate
        re-parses these lines and requires the count to match what was printed."""
        print("residency layout — policy A, artifact as stored, %d tensors in %d slabs"
              % (len(self.placements), len(self.slabs)))
        print("%-18s %6s %14s %14s %8s %6s  %s"
              % ("slab", "n", "content B", "slab B", "padding", "runs", "why"))
        for s in SLAB_ORDER:
            sl = self.slabs[s]
            print("%-18s %6d %14d %14d %8d %6d  %s"
                  % (s, len(sl.placements), sl.content_bytes, sl.bytes, sl.pad_bytes, len(sl.runs),
                     SLAB_WHY[s][:56] + "..."))
        print("%-18s %6d %14d %14d %8d %6d"
              % ("TOTAL", len(self.placements),
                 sum(p.rec.row for p in self.placements),
                 sum(s.bytes for s in self.slabs.values()),
                 sum(s.pad_bytes for s in self.slabs.values()), len(self.copy_plan())))
        print("\nweights, two views. They are not interchangeable:")
        print("  as-stored (what this artifact loads)   %14d B  %8.4f GiB"
              % (self.weights_total_bytes("as-stored"), self.weights_total_bytes("as-stored") / GiB))
        print("  policy-A modelled (what we would build) %13d B  %8.4f GiB"
              % (self.weights_total_bytes("policy-A"), self.weights_total_bytes("policy-A") / GiB))
        print("    of which the GGUF main stack          %14d B  %8.4f GiB"
              % (sum(p.policy_row for p in self.placements if p.rec.source == "gguf"),
                 sum(p.policy_row for p in self.placements if p.rec.source == "gguf") / GiB))
        print("    of which the MTP draft block          %14d B  %8.4f GiB"
              % (sum(p.policy_row for p in self.placements if p.rec.source == "hf_shards"),
                 sum(p.policy_row for p in self.placements if p.rec.source == "hf_shards") / GiB))
        print("\n%s" % MEASURED_ENGINE["source"])
        print("  measured engine demand %s GiB GTT (%s), %s"
              % (MEASURED_ENGINE["gtt_peak_gib"], MEASURED_ENGINE["gtt_range"],
                 MEASURED_ENGINE["condition"]))
        print("  => non-weight footprint (KV at n_ctx, compute buffers, scratch) >= %.2f GiB; the "
              "modelled weight figure omits it entirely and is that much optimistic."
              % MEASURED_NON_WEIGHT_FLOOR_GIB)
        print("\nfit claims — every line names its budget and its basis:")
        printed = 0
        for view in ("as-stored", "policy-A"):
            for budget in ("development", "deployment"):
                _, lines = self.verdict(self.weights_total_bytes(view) / GiB, budget,
                                        basis="weights-only", ctx=ctx, nseq=nseq)
                for ln in lines:
                    print("  [%s] %s" % (view, ln))
                    printed += 1
        modelled = RB.Budget(RB.Sources(), policy=POLICY_A_RB)
        row = modelled.row(ctx, nseq)
        for budget in ("development", "deployment"):
            _, lines = self.verdict(row["total"] / GiB, budget, basis="weights-plus-modelled-state",
                                    ctx=ctx, nseq=nseq)
            for ln in lines:
                print("  [modelled @%d tok x %d lane(s)] %s" % (ctx, nseq, ln))
                printed += 1
        return printed

    def check_list_matches(self):
        got = [r[0] for r in (self.closure_results or [])]
        if tuple(got) != self.CLOSURES:
            raise LayoutError("the closures that ran (%s) are not the closures this module "
                              "advertises (%s)" % (got, list(self.CLOSURES)))
        return len(got)


BUDGETS_TEXT = {"development": "%.2f-%.2f GiB, halogen co-resident" % (RB.CO_LO, RB.CO_HI),
                "deployment": ">= %.2f GiB, halogen stopped" % RB.DEDICATED_LO}


def _allowance_weights_only(layout, ctx=4096, nseq=1):
    return dict(extra_gib=0.0, detail={"note": "the number quoted is weights and nothing else"})


def _allowance_modelled_state(layout, ctx=4096, nseq=1):
    b = RB.Budget(RB.Sources(), policy=POLICY_A_RB)
    row = b.row(ctx, nseq)
    fixed = (row["main"] + row["draft"]) / GiB
    return dict(extra_gib=(row["total"] / GiB) - fixed,
                detail={"kv_gib": "%.4f" % (row["kv"] / GiB),
                        "draft_kv_gib": "%.4f" % (row["draft_kv"] / GiB),
                        "gdn_state_gib": "%.4f" % (row["state"] / GiB),
                        "other_gib": "%.4f" % (row["other"] / GiB),
                        "source": "tools/residency_budget.py @ ctx=%d seqs=%d" % (ctx, nseq)})


def _allowance_measured_footprint(layout, ctx=4096, nseq=1):
    return dict(extra_gib=MEASURED_NON_WEIGHT_FLOOR_GIB,
                detail={"measured_non_weight_gib": "%.2f" % MEASURED_NON_WEIGHT_FLOOR_GIB,
                        "source": MEASURED_ENGINE["source"],
                        "note": "lower bound: only the expert pages a generation actually touched "
                                "get faulted into the GTT, so weight bytes counted are <= the file "
                                "and the residue is correspondingly larger"})


BASES = {"weights-only": _allowance_weights_only,
         "weights-plus-modelled-state": _allowance_modelled_state,
         "weights-plus-measured-non-weight-floor": _allowance_measured_footprint}


# ---- injections ----------------------------------------------------------------------------
def _f_drop_a_slab_tensor(l):
    """One tensor vanishes from the layout after the artifact was validated: the slab/run coverage
    closures must not silently follow it."""
    s = l.slabs["weights.attention"]
    gone = s.placements.pop(3)
    l.placements.remove(gone)


def _f_overlap(l):
    """Two tensors placed at the same slab offset."""
    s = l.slabs["weights.experts"]
    s.placements[1].slab_off = s.placements[0].slab_off


def _f_pad_unaccounted(l):
    """Padding computed but not added to the slab size: the closure that catches it is
    `slab-coverage`, and this is the injection that proves that closure is awake."""
    l.slabs["weights.exact"].bytes -= l.slabs["weights.exact"].pad_bytes


def _f_run_length_short(l):
    for r in l.slabs["weights.experts"].runs:
        r["length"] -= 4096
        break


def _f_read_class_mis_tag(l):
    """The routed experts tagged as read-in-full: every byte total still closes, and the per-token
    read goes 4.3x wrong, which is exactly why the tag is checked rather than trusted."""
    for p in l.placements:
        if p.read_class == "per_token_gathered":
            p.read_class = "per_token_all"


def _f_expert_stride_fused(l):
    """Price the split GGUF expert records with the fused HF `[E,2I,H]` stride (AGENTS.md prediction
    3). The slice arithmetic then no longer tiles the row, and `expert-slices` must fire."""
    for p in l.placements:
        if p.rec.role == "blk.N.ffn_gate_exps.weight":
            p.rec.dims = (p.rec.dims[0], p.rec.dims[1] * 2, p.rec.dims[-1])


def _f_policy_width_swap(l):
    """Policy A mislabelled: experts charged at the attention width. `policy-a-vs-quant_scenarios`
    is the closure that catches it, because the tools price the same population and disagree."""
    for p in l.placements:
        if p.rec.family == "routed_experts" and p.rec.source == "gguf":
            p.policy_type = "Q8_0"
            p.policy_row = GBG.row_bytes(p.rec.n_elts, "Q8_0")


FAULTS = {
    "drop_a_slab_tensor": _f_drop_a_slab_tensor,
    "slab_overlap": _f_overlap,
    "pad_unaccounted": _f_pad_unaccounted,
    "run_length_short": _f_run_length_short,
    "read_class_mis_tag": _f_read_class_mis_tag,
    "expert_stride_fused": _f_expert_stride_fused,
    "policy_width_swap": _f_policy_width_swap,
}


def build(**kw):
    return ResidencyLayout(**kw)


def main():
    lay = ResidencyLayout()
    res = lay.run_closures()
    print("closures — every one must close at 0 bytes, and a non-closure aborts this run:")
    for name, ok, n, detail in res:
        print("  [%s] %-34s n=%-4d %s" % ("PASS" if ok else "FAIL", name, n, detail))
    print()
    printed = lay.report()
    print("\nprinted residency claim lines: %d (re-parsed by engine/gate.py, which requires the "
          "count to match)" % printed)
    print("closures run: %d/%d" % (lay.check_list_matches(), len(ResidencyLayout.CLOSURES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
