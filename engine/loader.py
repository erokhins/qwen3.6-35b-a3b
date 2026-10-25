#!/usr/bin/env python3
"""loader.py — policy A's resident set: admit it, plan it, stage it, hand a kernel lane a handle.

`artifact.py` answers *what the file claims to be*; `layout.py` answers *where the bytes go*. This
module is the load path between them, and it is the only place in `engine/` that is allowed to say
"no" to a tensor. Four things are checks here rather than comments:

1. **The width veto (D-17).** A tensor is admitted from its **own ggml type**, read out of the
   header's type id and priced by `tools/ggml_block_geom.py`'s geometry, against the width policy A
   declares for its family — which is also derived (`quant_scenarios.POLICY_TYPE` -> `ggml_block_geom.bpw_of`),
   never typed in bits. Narrower than the family floor is a refusal **by name**, never a coercion and
   never a rounding-to-fit. Two families in the published artifact legitimately sit outside the
   floor and are carried as *named, exercised* waivers (`WIDTH_WAIVERS`); a waiver that nothing
   exercises fails the check, so the waiver cannot rot into dead code.
2. **The two budgets (D-11), printed separately, each with its own bound, its own evidence state and
   its own verdict** — see `layout.BUDGETS`. A budget whose bound has never been measured gets
   `layout.NOT_ADJUDICABLE`, and the check `budget-lines-separate` requires the *pairing*
   (unmeasured bound => no fit claim), not either field alone: `budgets_conflated` is the injection
   that proves it awake.
3. **Independent re-derivation, not self-agreement.** The records below are parsed by
   `tools/gguf_width_table.py`'s own tokenizer — a second one, with its own block table, kept here
   in a variant that *retains the per-tensor `offset` field* `gguf_width_table.parse_gguf` discards —
   and every byte total, per-family sum, per-tensor offset and the `data_start` arithmetic is then
   compared against `artifact.py`/`layout.py`, which came through `tools/gguf_reader.py` and
   `tools/ggml_block_geom.py` instead. Two tokenizers, two block tables, one file: agreement is
   evidence, self-agreement is not.
4. **A must-catch set derived from geometry.** Every routed record must refuse an expert index
   outside `[0, E)` *for every one of the 122 routed records*, so the refusal cannot be vacuous; the
   other refusals are injected by `FAULTS` below and `engine/gate.py` requires the *named* check to
   turn red for each.

Staging is honest about what does not exist: the draft block's bytes are bf16 on disk and policy A
prices them at Q4_K/Q8_0, so staging them would require a requantizer this repo has not written —
`stage_draft()` refuses instead of handing back bf16 bytes labelled Q4_K, and the byte gap it refuses
across is printed. Device residency is a plan: no allocator exists, so `device_slabs()` raises
(see `engine/README.md` §Stubbed and `quantization/residency-layout.md` §7 item 2).

CPU-only. No HIP import, no timer, no `ms` / `tok/s` / `GB/s` figure printed. Repetitions report
counts. Data-section reads are gated by `tools/audit_published_quant.py`'s own selftest marker and
are opt-in (`--probe`), so a default run of this module touches header bytes only.
"""
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
for _p in (TOOLS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import artifact as ART                     # noqa: E402  the pinned file, its records, its refusals
import layout as LAY                       # noqa: E402  the slabs, the copy plan, the two budgets
import bf16_census as BC                   # noqa: E402  config geometry (E, layer_types)
import ggml_block_geom as GBG              # noqa: E402  widths derived from the pinned ggml header
import gguf_header_types as G              # noqa: E402  the tokenizer gguf_width_table.py drives
import gguf_width_table as GWT             # noqa: E402  the independent byte ruler (own block table)
import quant_scenarios as QS               # noqa: E402  policy A's own width table
import residency_budget as RB              # noqa: E402  committed paths

GiB = 2.0 ** 30
MiB = 2.0 ** 20
POLICY_A = LAY.POLICY_A
POLICY_A_RB = LAY.POLICY_A_RB
AUDIT_HASH_EVIDENCE = "quantization/evidence/audit-hash-2026-10-26.json"


class LoaderError(Exception):
    """The load path refused. `tag` names the refusal so `engine/gate.py` can require a *specific*
    check or exception, not merely that something raised."""

    def __init__(self, tag, msg):
        super().__init__("[%s] %s" % (tag, msg))
        self.tag = tag


class WidthVeto(LoaderError):
    """A tensor's own stored ggml type is not what its family's policy-A width allows."""


class NotWired(LoaderError):
    """The plan is complete and the machinery to execute it does not exist. Raising is the point:
    a stub that returns something plausible is how an unbuilt path becomes a wrong number."""


# -----------------------------------------------------------------------------------------------
# 1. the width veto (D-17): from the tensor's ggml type, against the family's policy-A width
# -----------------------------------------------------------------------------------------------
# What "exact" is spelled as, per source. The artifact stores the never-quantized families as F32;
# the HF shards store the draft's as BF16. Either is unquantized; a quantized width in an exact role
# is a veto whatever the floor arithmetic would otherwise allow, because a 1-ULP change to a router
# scale or an `A_log` is not a rounding difference
# (quantization/README.md §"How a quantized path gets gated" item 3).
EXACT_KINDS = {"gguf": {"F32"}, "hf_shards": {"F32", "BF16", "F16"}}

# policy A's floor for a family, in bits per weight, DERIVED: quant_scenarios owns which width policy
# A gives which family class, ggml_block_geom owns what that width costs per weight. Nothing here
# types a bpw, which is the exact mistake tools/gguf_width_table.py §why exists to kill.
def policy_floor_bpw(family):
    return GBG.bpw_of(QS.POLICY_TYPE(POLICY_A, family))


def stored_bpw(type_name):
    try:
        return GBG.bpw_of(type_name)
    except KeyError:
        raise WidthVeto("width-unpriceable",
                        "no derived block geometry for stored type %r; a type this loader cannot "
                        "price is bytes it would silently count as zero, so it is refused rather "
                        "than guessed (tools/ggml_block_geom.py derives from the pinned ggml header)"
                        % (type_name,))


# The only tensor policy A would store wider than the artifact does, and therefore the only one the
# floor rule cannot admit on its own. Named, cited, and REQUIRED to be exercised by a real tensor:
# a waiver nothing exercises is dead code, and `width-veto-admits-artifact` fails if one is declared
# and unused.
WIDTH_WAIVERS = {
    "output.weight": dict(
        accepted=("Q6_K",),
        authority="DECISIONS.md D-10 as amended by inheritance: the user decided lm_head/embeddings "
                  "at Q8_0 and the published artifact stores lm_head at Q6_K "
                  "(quantization/README.md \u00a7Independent confirmation; policy G is the row that "
                  "prices it). This is the ONLY narrower-than-policy tensor the loader admits.",
        note="narrower than the Q8_0 floor: 6.5625 bpw, 1.0037 GiB stored vs 1.2960 GiB modelled"),
}

# The other place the artifact and policy A disagree: expert `down_proj`, which the producer priced
# ABOVE policy A's Q4_K (Q5_K on 37 layers, Q6_K on 3). The floor rule admits it without a waiver,
# so this table is not a permission slip -- it is the bucket the byte attribution needs, because "we
# are 1.3260 GiB heavier than our own policy and nobody knows where it went" is the wrong answer.
WIDTH_WIDER_ROLES = {
    "blk.N.ffn_down_exps.weight": dict(
        stored=("Q5_K", "Q6_K"),
        authority="DECISIONS.md D-08 as amended by inheritance (quantization/README.md \u00a7Independent "
                  "confirmation of the attention decision): 5.5 bpw on 37 layers, 6.5625 bpw on 3. "
                  "Which three is read out of the file, never typed here -- see the q6k-layer-set check.",
        note="wider than the Q4_K floor: +1.3502 GiB across 40 layers"),
}


def admit(rec, waivers=None):
    """Admit one record into the resident set, or refuse it by name. Returns a note naming the rule.

    The predicate is arithmetic over the tensor's own type, so the refusal is a fact about the file
    and not about a reviewer's memory of what a family is supposed to be."""
    wa = WIDTH_WAIVERS if waivers is None else waivers
    role = rec.role if rec.source == "gguf" else None
    bpw = stored_bpw(rec.type_name)
    if role is not None and role in LAY.EXACT_ROLES:
        if rec.type_name not in EXACT_KINDS.get(rec.source, set()):
            raise WidthVeto(
                "width-veto",
                "%s is a never-quantized role (%s) but the file stores it %s at %.4f bpw; the "
                "router, the norms, conv1d, A_log and dt_bias stay exact and are never coerced to "
                "fit a slab (quantization/README.md \u00a7How a quantized path gets gated item 3)"
                % (rec.name, role, rec.type_name, bpw))
        return "exact:%s" % rec.type_name
    floor = policy_floor_bpw(rec.family)
    if bpw + 1e-9 >= floor:
        return "at-or-above-floor:%s>=%.4f" % (rec.type_name, floor)
    wv = wa.get(role) if role else None
    if wv and rec.type_name in wv["accepted"]:
        return "waived:%s:%s" % (role, rec.type_name)
    raise WidthVeto(
        "width-veto",
        "%s is family %r, whose policy-A floor is %.4f bpw (%s), but the file stores it %s at "
        "%.4f bpw. Refusing by name: this loader does not requantize, does not round to fit, and "
        "does not accept a width claim that is not the tensor's own (DECISIONS.md D-17: the width "
        "veto comes from the artifact, not from a string)"
        % (rec.name, rec.family, floor, QS.POLICY_TYPE(POLICY_A, rec.family), rec.type_name, bpw))


# -----------------------------------------------------------------------------------------------
# 2. the independent parse: gguf_width_table's own tokenizer, with the offset field kept
# -----------------------------------------------------------------------------------------------
def parse_records(path):
    """[(name, dims, type_name, offset)] read by the ruler's tokenizer, not `artifact.py`'s.

    `tools/gguf_width_table.parse_gguf` walks these very fields and throws the 8-byte `offset` away,
    which is enough for a byte census and useless for a load path. This function reuses that
    tokenizer (`gguf_header_types.Cur`, the same one) and keeps the field, so the loader's offsets are
    read from the file by a different parser than the one `artifact.py` used — while the block sizes
    come from `gguf_width_table.BLK`, which is also not the table `artifact.py` priced with."""
    with open(path, "rb") as fh:
        head = fh.read(64 << 20)
    if head[:4] != b"GGUF":
        raise LoaderError("ruler-magic", "%s is not a GGUF for the ruler (magic=%r)"
                          % (path, head[:4]))
    c = G.Cur(head)
    c.take(4)
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    meta = {}
    for _ in range(n_kv):
        k = c.s()
        vt = c.u32()
        meta[k] = G.skip_array(c) if vt == 9 else c.value(vt)
    rows = []
    for i in range(int(n_tensors)):
        name = c.s()
        nd = c.u32()
        dims = tuple(c.u64() for _ in range(nd))
        dt = c.u32()
        off = c.u64()
        rows.append((i, name, dims, G.TYPES.get(dt, "type_%d" % dt), off))
    header_end = c.p
    align = int(meta.get("general.alignment", 32)) or 32
    data_start = -(-header_end // align) * align
    return dict(version=version, n_tensors=int(n_tensors), n_kv=int(n_kv), meta=meta, rows=rows,
                header_end=header_end, alignment=align, data_start=data_start)


def ruler_record(row, data_start):
    """A `TensorRecord` (the repo's one record type) priced by the ruler's block table."""
    i, name, dims, type_name, off = row
    if type_name not in GWT.BLK:
        raise LoaderError("ruler-unmapped-type",
                          "%s is stored %s, which tools/gguf_width_table.py's table does not carry; "
                          "an unmapped type counted as zero bytes is how a census reads low and "
                          "still prints a total" % (name, type_name))
    elts = 1
    for d in dims:
        elts *= int(d)
    qk, blk_b, _expr = GWT.BLK[type_name]
    row_bytes = -(-elts // qk) * blk_b
    return ART.TensorRecord(i, name, dims, type_name, elts, row_bytes, off, data_start, "gguf")


def ruler_draft(path):
    """The draft block re-read: spans from the safetensors header, rows from the ruler's table.

    `artifact.py` derives the same records with `ggml_block_geom`'s block sizes; this derives them
    with `gguf_width_table.BLK`'s. Agreement is then a fact about two block tables, not one."""
    if not os.path.exists(path):
        raise LoaderError("draft-absent", "the draft block %s is absent; the GGUF carries zero "
                                          "mtp.* tensors, so it is the engine's only draft source "
                                          "(quantization/residency-layout.md \u00a72)" % path)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    out, spans = [], {}
    for i, (name, d) in enumerate(sorted(hdr.items())):
        if name == "__metadata__":
            continue
        tname = {"BF16": "BF16", "F32": "F32", "F16": "F16"}.get(d["dtype"])
        if tname is None:
            raise LoaderError("draft-dtype", "%s says %s is %s; nothing here prices a dtype it has "
                                             "to guess" % (path, name, d["dtype"]))
        dims = tuple(d["shape"])
        elts = 1
        for x in dims:
            elts *= int(x)
        qk, blk_b, _expr = GWT.BLK[tname]
        lo, hi = d["data_offsets"]
        r = ART.TensorRecord(i, name, dims, tname, elts, -(-elts // qk) * blk_b, lo, base,
                             "hf_shards", hf_name=name)
        spans[name] = (lo, hi)
        out.append(r)
    return out, base, spans


# -----------------------------------------------------------------------------------------------
# 3. the loader
# -----------------------------------------------------------------------------------------------
class Loader:
    """Admit the artifact's tensors into a resident set, plan the staging, refuse what is not built.

    Every check returns `(ok, n, detail)` instead of raising, so one run enumerates every outcome and
    `engine/gate.py` can require a *named* check to have turned red under an injection. `n` is always
    a population size the check actually walked — a check that walked nothing prints n=0 and cannot
    pass (pitfalls/40 §empty-parse-PASS)."""

    CHECKS = ("ruler-parse-complete", "ruler-family-bytes-vs-artifact", "ruler-records-vs-artifact",
              "data-start-arithmetic", "file-size-identity", "residency-set-complete",
              "width-veto-admits-artifact", "offsets-in-range", "file-ranges-disjoint",
              "final-range-complete", "sha-vs-pin", "q6k-layer-set", "width-delta-attributed",
              "staging-plan-matches-ruler", "expert-slices-addressable",
              "draft-requantize-not-wired", "budget-lines-separate")

    WAIVER_ROLES = tuple(WIDTH_WAIVERS)
    WIDER_ROLES = tuple(WIDTH_WIDER_ROLES)

    def __init__(self, art=None, faults=(), sha_provider=None, verify_sha256=False):
        self.art = art if art is not None else ART.open_artifact(verify_sha256=verify_sha256)
        self.cfg = BC.geometry()
        self.ruler_hdr = parse_records(self.art.path)
        self.data_start = self.ruler_hdr["data_start"]
        self.size = os.path.getsize(self.art.path)
        self.ruler = [ruler_record(r, self.data_start) for r in self.ruler_hdr["rows"]]
        self.draft, self.draft_base, self.draft_spans = ruler_draft(RB.MTP_PATH)
        self.draft_size = os.path.getsize(RB.MTP_PATH)
        self.records = self.ruler + self.draft
        self._reindex()
        self.sha_provider = sha_provider or (ART.stream_sha256 if verify_sha256 else None)
        self.sha_override = None
        self.waivers = dict(WIDTH_WAIVERS)
        self.stub_requantizer = False
        self.faults = set(faults)
        # Built BEFORE the faults so an injection can reach the copy plan (runs), and so a fault that
        # makes a slab unloadable surfaces as a red check here rather than as a raised LayoutError
        # escaping the enumeration.
        self.lay = LAY.ResidencyLayout(self.art)
        self._apply_faults()
        self.results = None

    def _reindex(self):
        self.by_name = {r.name: r for r in self.records}

    def _apply_faults(self):
        for f in sorted(self.faults):
            fn = FAULTS.get(f)
            if fn is None:
                raise LoaderError("unknown-fault", "unknown loader fault %r; known: %s"
                                  % (f, sorted(FAULTS)))
            fn(self)
            self._reindex()

    # ---- the expert axis, named ----------------------------------------------------------------------------
    def expert_axis(self, rec):
        """Which axis is the expert axis, and why it is not "the last one".

        GGUF stores dims fastest-first, so the HF trailing expert axis is `dims[-1]` and one expert's
        bytes are contiguous. A safetensors record is HF order, so the expert axis is `dims[0]` and
        one expert is a STRIDED set of rows — there is no contiguous byte range to hand a kernel, and
        pretending otherwise is how a draft-path kernel reads the wrong matrix. `layout.py`'s
        `expert-slices` closure cannot see this: ANY axis that divides a row tiles it exactly, so it
        is axis-blind by construction. This function is what names the axis."""
        if rec.source == "gguf":
            return len(rec.dims) - 1
        return 0

    def expert_range(self, name, idx, want=None):
        """One expert's byte range in a routed record, or a refusal that names the reason.

        Addresses computed here and in `layout.expert_slice` must agree byte for byte — one address
        expression per concept, checked against each other rather than trusted (pitfalls/40
        §two-self-consistent-layouts).
        """
        rec = self.by_name.get(name)
        if rec is None:
            raise LoaderError("unknown-tensor", "%r is not in the resident set" % name)
        if rec.family != "routed_experts":
            raise LoaderError("not-routed", "%s is family %r; it has no expert axis to slice"
                              % (name, rec.family))
        if rec.source == "hf_shards":
            raise NotWired(
                "expert-slice-strided",
                "%s comes from %s, whose element order makes the expert axis dims[0]: one expert is "
                "%d strided rows, not a contiguous range, so no bulk copy can stage it. Staging the "
                "draft block's experts needs a transposing pack that does not exist yet "
                "(engine/README.md \u00a7Stubbed)." % (name, RB.MTP_PATH, rec.dims[0]))
        ax = self.expert_axis(rec)
        e = int(rec.dims[ax])
        if e <= 1:
            raise LoaderError("not-an-expert-axis", "%s has dims %s: axis %d is %d wide, which is "
                                                   "not an expert axis" % (name, rec.dims, ax, e))
        idx = int(idx)
        if not 0 <= idx < e:
            raise LoaderError("expert-index",
                              "%s: expert %d is outside [0,%d). Refusing rather than wrapping: an "
                              "out-of-range expert id is the signature of a router that picked "
                              "something this tensor does not contain (AGENTS.md prediction 1)"
                              % (name, idx, e))
        inner = 1
        for i, d in enumerate(rec.dims):
            if i != ax:
                inner *= int(d)
        qk, blk_b, _ = GWT.BLK[rec.type_name]
        if inner % qk:
            raise LoaderError("expert-slice-splits-block",
                              "%s: a slice is %d elements, not a whole number of %s blocks of %d; "
                              "the slice would split a quantized block" % (name, inner, rec.type_name, qk))
        length = -(-inner // qk) * blk_b
        if length * e != rec.row:
            raise LoaderError("expert-axis-does-not-tile",
                              "%s: %d slices of %d B = %d B but the record's row is %d B, so axis %d "
                              "of dims %s is not the expert axis (AGENTS.md prediction 3: the fused "
                              "gate|up pair is laid out along the OUTPUT dim)"
                              % (name, e, length, length * e, rec.row, ax, rec.dims))
        mine = dict(name=name, index=idx, n_slices=e, file_off=rec.file_offset + idx * length,
                    length=length, elts=inner, type_name=rec.type_name)
        if want is None:
            want = self.lay.expert_slice(name, idx)
        if (want["file_off"], want["length"]) != (mine["file_off"], mine["length"]):
            raise LoaderError("expert-address-disagreement",
                              "%s expert %d: loader says file_off %d len %d, layout.py says %d/%d"
                              % (name, idx, mine["file_off"], mine["length"],
                                 want["file_off"], want["length"]))
        return mine

    # ---- staging -----------------------------------------------------------------------------------------
    def stage_runs(self):
        """The bulk-copy plan for the admitted set, largest-first-bytes-last, exactly as `layout` cut it."""
        return self.lay.copy_plan()

    def _size_of_source(self, slab_name):
        return self.size if self.lay.slab(slab_name).source == "gguf" else self.draft_size

    def probe(self, window=512):
        """Earn the audit selftest token, then pread a window at each end of every sampled run.

        This is the step that turns "the header's offsets are self-consistent" into "those bytes are
        in the file". The sample is derived, not hand-picked: first and last run of every slab, plus
        the runs holding each layer's routed gate/up and down records (the sparse-read family). Bytes
        are read and counted, never interpreted, and no timing is taken.
        """
        import audit_published_quant as APQ

        class _Section:
            section = "engine/loader.py staging probe"

        mark = APQ.require_selftest(_Section())
        if mark is None:
            raise NotWired("data-read-not-granted",
                           "tools/audit_published_quant.py refused the selftest token, so not one "
                           "tensor-data byte is read (that is the reader's own gate, reused here "
                           "rather than a second one: %s)" % APQ.SELFTEST_MARK)
        tok = APQ.SELFTEST_TOKEN
        runs = self.stage_runs()
        sample, seen = [], set()
        for s in LAY.SLAB_ORDER:
            rs = [r for r in runs if r["slab"] == s]
            for r in (rs[:1] + rs[-1:]):
                if id(r) not in seen:
                    seen.add(id(r))
                    sample.append(r)
        # A tensor's slab coordinate lives on its PLACEMENT, not on the TensorRecord (layout.py
        # §Placement.__slots__ = rec, slab, slab_off, offsets, read_class, policy_type, policy_row),
        # so resolve name -> (slab_off, row) through the slabs themselves. This was an AttributeError
        # on the arm's first-ever execution: the loop below read rec.slab_off, which does not exist,
        # and the arm had only ever printed SKIPPED, so nothing had ever reached this line.
        placed = {}
        for s in LAY.SLAB_ORDER:
            for p in self.lay.slab(s).placements:
                placed[p.rec.name] = (p.slab_off, p.rec.row)
        for li in range(self.cfg["L"]):
            for role in ("blk.N.ffn_gate_exps.weight", "blk.N.ffn_down_exps.weight"):
                rec = self.by_name.get(role.replace("N", str(li)))
                if rec is None:
                    continue
                pos = placed.get(rec.name)
                if pos is None:
                    continue
                off, row = pos
                for r in runs:
                    if r["slab_off"] <= off and r["slab_off"] + r["length"] >= off + row:
                        if id(r) not in seen:
                            seen.add(id(r))
                            sample.append(r)
                        break
        reads = bytes_read = short = 0
        try:
            for r in sample:
                n = min(window, r["length"])
                for where in (r["file_off"], r["file_off"] + r["length"] - n):
                    got = self.art._hdr.read(tok, where, n, why="staging probe")
                    reads += 1
                    bytes_read += len(got)
                    if len(got) != n:
                        short += 1
        finally:
            self.art._hdr.close()
        return dict(runs=len(runs), sampled=len(sample), reads=reads, bytes_read=bytes_read,
                    short=short, marker=mark.get("utc_latest_selftest_pass"))

    def handle(self, name):
        """What a kernel lane gets: one tensor's address in its slab, and nothing it did not ask for."""
        rec = self.by_name[name]
        p = self.lay.placement(name)
        return dict(name=name, slab=p.slab, slab_off=p.slab_off, length=rec.row,
                    file_off=rec.file_offset, type_name=rec.type_name, family=rec.family,
                    read_class=p.read_class, policy_type=p.policy_type, n_elts=rec.n_elts,
                    dims=tuple(rec.dims), source=rec.source)

    def stage_draft(self, name):
        """Refuse to stage a draft tensor. The bytes on disk are bf16; policy A prices them at Q4_K/Q8_0.

        Handing back the bf16 bytes and labelling them Q4_K is the failure this function exists to
        make impossible, so the refusal is the honest return value and the byte gap is printed."""
        rec = self.by_name.get(name)
        if rec is None or rec.source != "hf_shards":
            raise LoaderError("not-draft", "%s is not a draft-block tensor" % name)
        if self.stub_requantizer:
            return b"\x00" * rec.row
        want = GBG.row_bytes(rec.n_elts, QS.POLICY_TYPE(POLICY_A, rec.family))
        raise NotWired(
            "draft-requantizer-absent",
            "%s is %d B of %s in %s, which policy A would hold as %d B of %s. Converting it needs a "
            "requantizer this repo has not written, so this loader will not hand back the bf16 bytes "
            "under the label %s (DECISIONS.md D-09 prices the draft block uniform with the stack it "
            "drafts for; engine/README.md \u00a7Stubbed)"
            % (name, rec.row, rec.type_name, os.path.basename(RB.MTP_PATH), want,
               QS.POLICY_TYPE(POLICY_A, rec.family), QS.POLICY_TYPE(POLICY_A, rec.family)))

    def device_slabs(self):
        raise NotWired(
            "device-allocator-absent",
            "the plan allocates %d slabs totalling %d B; nothing here can allocate them because the "
            "engine's allocator does not exist, and the allocation COUNT has never been probed on "
            "this box (quantization/residency-layout.md \u00a77 item 2). The recipe is hipMalloc + bulk "
            "copy, never Managed (pitfalls/10-device-memory.md \u00a7Allocation-recipe)."
            % (len(self.lay.slabs), sum(s.bytes for s in self.lay.slabs.values())))

    # ---- the checks ---------------------------------------------------------------------------------
    def checks(self):
        return [(name,) + fn() for name, fn in (
            ("ruler-parse-complete", self._c_ruler_parse),
            ("ruler-family-bytes-vs-artifact", self._c_ruler_family),
            ("ruler-records-vs-artifact", self._c_ruler_records),
            ("data-start-arithmetic", self._c_data_start),
            ("file-size-identity", self._c_file_size),
            ("residency-set-complete", self._c_residency_set),
            ("width-veto-admits-artifact", self._c_width_admit),
            ("offsets-in-range", self._c_offsets_in_range),
            ("file-ranges-disjoint", self._c_ranges_disjoint),
            ("final-range-complete", self._c_final_range),
            ("sha-vs-pin", self._c_sha),
            ("q6k-layer-set", self._c_q6k_layers),
            ("width-delta-attributed", self._c_width_delta),
            ("staging-plan-matches-ruler", self._c_staging_plan),
            ("expert-slices-addressable", self._c_expert_slices),
            ("draft-requantize-not-wired", self._c_draft_not_wired),
            ("budget-lines-separate", self._c_budget_lines))]

    def run_checks(self):
        self.results = self.checks()
        names = [r[0] for r in self.results]
        if tuple(names) != self.CHECKS:
            raise LoaderError("check-list-drift", "ran %s, advertised %s" % (names, list(self.CHECKS)))
        return self.results

    def _c_ruler_parse(self):
        hdr, n = self.ruler_hdr, len(self.ruler)
        uniq = len({r.name for r in self.ruler})
        ok = (n == hdr["n_tensors"] == ART.EXPECTED_N_TENSORS == uniq)
        return ok, n, ("second tokenizer parsed %d records of %d declared, %d unique names, %d printed "
                       "by --records; header_end %d alignment %d (artifact.py read the same file with "
                       "tools/gguf_reader.py and got %d)"
                       % (n, hdr["n_tensors"], uniq, n, hdr["header_end"], hdr["alignment"],
                          len(self.art.records)))

    def _c_ruler_family(self):
        mine, theirs = {}, dict(self.art.width_table_family)
        for r in self.ruler:
            e = mine.setdefault(r.type_name, [0, 0])
            e[0] += 1
            e[1] += r.row
        drift = []
        for t in sorted(set(mine) | set(theirs)):
            m, o = mine.get(t, [0, 0]), theirs.get(t, (0, 0))
            if list(m) != list(o):
                drift.append("%s: ruler %d tensors/%d B, artifact.py %d/%d"
                             % (t, m[0], m[1], o[0], o[1]))
        self.family_table = mine
        return not drift, len(mine), ("per-family count and byte sum identical to artifact.py's for all "
                                      "%d stored types" % len(mine) if not drift else "; ".join(drift[:4]))

    def _c_ruler_records(self):
        bad, n = [], 0
        theirs = self.art.by_name
        for r in self.ruler:
            n += 1
            o = theirs.get(r.name)
            if o is None:
                bad.append("%s: present in the ruler parse, absent from artifact.py's" % r.name)
                continue
            if (tuple(r.dims), r.type_name, r.offset, r.row) != (tuple(o.dims), o.type_name,
                                                                o.offset, o.row):
                bad.append("%s: ruler dims=%s %s off=%d row=%d vs artifact dims=%s %s off=%d row=%d"
                           % (r.name, r.dims, r.type_name, r.offset, r.row,
                              o.dims, o.type_name, o.offset, o.row))
        return not bad and n == len(theirs), n, (
            "all %d records agree on dims, stored type, data-section offset and row bytes across two "
            "tokenizers and two block tables" % n if not bad
            else "%d record(s) disagree, first: %s" % (len(bad), bad[0]))

    def _c_data_start(self):
        pin = ART.pin_header(self.art.pin) if self.art.pin else {}
        recomputed = -(-self.ruler_hdr["header_end"] // self.ruler_hdr["alignment"]) * self.ruler_hdr["alignment"]
        rows = [("recomputed here", recomputed), ("ruler header claim", self.data_start),
                ("artifact.py", self.art.data_start), ("pin", pin.get("data_start"))]
        bad = ["%s=%s" % (k, v) for k, v in rows if v != recomputed]
        return not bad, len(rows), ("data_start %d = ceil(header_end %d / %d) * %d and every reader "
                                    "agrees: %s" % (recomputed, self.ruler_hdr["header_end"],
                                                     self.ruler_hdr["alignment"],
                                                     self.ruler_hdr["alignment"],
                                                     ", ".join("%s %s" % (k, v) for k, v in rows))
                                    if not bad else "DISAGREE: %s (recomputed %d)"
                                    % ("; ".join(bad), recomputed))

    def _c_file_size(self):
        stored = sum(r.row for r in self.ruler)
        residue = self.size - (stored + self.data_start)
        return residue == 0, len(self.ruler), (
            "ruler sum(row_bytes) %d + data_start %d == file size %d (residue %+d B); artifact.py "
            "reports the same identity on its own parse" % (stored, self.data_start, self.size, residue))

    def _c_residency_set(self):
        g = self.cfg
        mine = {r.name for r in self.ruler}
        missing = sorted(set(self.art.expected) - mine)
        extra = sorted(mine - set(self.art.expected))
        by_layer = {}
        for r in self.ruler:
            if r.layer is not None:
                by_layer.setdefault(r.layer, set()).add(r.role)
        groups = {}
        for li in sorted(by_layer):
            groups.setdefault(g["layer_types"][li], set()).add(li)
        bad = []
        for ltype, lset in sorted(groups.items()):
            role_sets = {frozenset(by_layer[li]) for li in lset}
            if len(role_sets) != 1:
                bad.append("%s layers disagree about their own role set (%d variants)" % (ltype, len(role_sets)))
        if len(by_layer) != g["L"]:
            bad.append("%d layers carry tensors, the config has %d" % (len(by_layer), g["L"]))
        routed = [r for r in self.ruler if r.family == "routed_experts"]
        e_bad = [r.name for r in routed if int(r.dims[self.expert_axis(r)]) != g["E"]]
        if e_bad:
            bad.append("%d routed records do not carry %d experts, first %s" % (len(e_bad), g["E"], e_bad[0]))
        d_missing = sorted(set(self.art.draft_index_names) - {r.name for r in self.draft})
        if d_missing:
            bad.append("draft block is missing %d names, first %s" % (len(d_missing), d_missing[0]))
        n = len(self.ruler) + len(by_layer) + len(routed)
        return (not missing and not extra and not bad), n, (
            "%d names match the set derived from the safetensors index x config shapes; %d layers, each "
            "carrying one role set (%s); %d routed records all carry %d experts; %d draft names"
            % (len(self.ruler), len(by_layer),
               ", ".join("%s x%d" % (k, len(v)) for k, v in sorted(groups.items())),
               len(routed), g["E"], len(self.draft))
            if not (missing or extra or bad) else "MISSING %s / EXTRA %s / %s"
            % (missing[:3], extra[:3], bad[:3]))

    def _c_width_admit(self):
        notes, refused = {}, []
        for r in self.records:
            try:
                k = admit(r, self.waivers)
            except WidthVeto as ex:
                refused.append(str(ex))
                continue
            notes[k] = notes.get(k, 0) + 1
        exercised = {k.split(":")[1] for k in notes if k.startswith("waived:")}
        unexercised = sorted(set(self.waivers) - exercised)
        self.admit_notes, self.refusals = notes, refused
        ok = (not refused and not unexercised and sum(notes.values()) == len(self.records)
              and len(self.records) > 0)
        return ok, len(self.records), (
            "%d/%d tensors admitted; waivers exercised by %s; refusals %d"
            % (sum(notes.values()), len(self.records), sorted(exercised), len(refused))
            if ok else "refused %d (first: %s); unexercised waivers %s"
            % (len(refused), refused[0] if refused else "-", unexercised))

    def _c_offsets_in_range(self):
        bad, n = [], 0
        span = {"gguf": (self.data_start, self.size, os.path.basename(self.art.path)),
                "hf_shards": (self.draft_base, self.draft_size, os.path.basename(RB.MTP_PATH))}
        for src, (lo, hi, what) in sorted(span.items()):
            for r in [x for x in self.records if x.source == src]:
                n += 1
                if not (lo <= r.file_offset and r.file_offset + r.row <= hi):
                    bad.append("%s [%d,%d) is outside %s's [%d,%d)"
                               % (r.name, r.file_offset, r.file_offset + r.row, what, lo, hi))
        return not bad, n, ("every admitted tensor's byte range lies inside the file it is staged "
                            "from (%d tensors, %d bytes of the %d B gguf)"
                            % (n, sum(r.row for r in self.ruler), self.size) if not bad
                            else "%d out-of-range: %s" % (len(bad), bad[:2]))

    def _c_ranges_disjoint(self):
        bad, pairs = [], 0
        for src in ("gguf", "hf_shards"):
            rs = sorted([r for r in self.records if r.source == src], key=lambda r: r.offset)
            for a, b in zip(rs, rs[1:]):
                pairs += 1
                if b.offset < a.offset + a.row:
                    bad.append("%s [%d,%d) overlaps %s [%d,%d)" % (a.name, a.offset,
                                                                   a.offset + a.row, b.name, b.offset,
                                                                   b.offset + b.row))
        return not bad, pairs, ("%d adjacent record pairs in file order, no two claim the same byte"
                                % pairs if not bad else "%d overlaps: %s" % (len(bad), bad[:2]))

    def _c_final_range(self):
        rows, ok = [], True
        gg = max(self.ruler, key=lambda r: r.offset + r.row)
        res = self.size - (gg.file_offset + gg.row)
        rows.append((os.path.basename(self.art.path), gg.name, res))
        ok = ok and res == 0
        dmax = max(self.draft_spans[r.name][1] for r in self.draft)
        dres = self.draft_size - (self.draft_base + dmax)
        rows.append((os.path.basename(RB.MTP_PATH), "last draft span", dres))
        ok = ok and dres == 0
        return ok, len(rows), ("last record ends exactly at EOF in both files: %s"
                               % "; ".join("%s via %s residue %+d B" % r for r in rows))

    def _c_sha(self):
        ev_path = os.path.join(REPO, AUDIT_HASH_EVIDENCE)
        if not os.path.exists(ev_path):
            raise LoaderError("sha-evidence-absent",
                              "%s is absent, so the pin's digest has no second witness and this "
                              "loader will not pretend otherwise" % AUDIT_HASH_EVIDENCE)
        ev = json.load(open(ev_path))
        dig = {"pin sha256": self.art.pin.get("sha256"),
               "audit pass1": ev.get("sha256_pass1"), "audit pass2": ev.get("sha256_pass2")}
        how = "three committed digests"
        if self.sha_override:
            dig["injected"] = self.sha_override
            how = "three committed digests plus an injected one"
        elif self.sha_provider:
            dig["recomputed from the file"] = self.sha_provider(self.art.path)
            how = "three committed digests plus a fresh full-file hash"
        bad = [k for k, v in dig.items() if not (isinstance(v, str) and len(v) == 64)]
        if bad:
            return False, len(dig), "malformed digest(s): %s" % bad
        ok = len(set(dig.values())) == 1 and ev.get("verdict") == "PASS"
        return ok, len(dig), ("%s agree (%s\u2026) and the audit record's verdict is %s"
                              % (how, sorted(dig.values())[0][:12], ev.get("verdict"))
                              if ok else "DIGESTS DISAGREE: %s" % dig)

    def _c_q6k_layers(self):
        downs = [r for r in self.ruler if r.role == "blk.N.ffn_down_exps.weight"]
        cnt = {}
        for r in downs:
            cnt[r.type_name] = cnt.get(r.type_name, 0) + 1
        mine = sorted(r.layer for r in downs if r.type_name == "Q6_K")
        theirs = sorted(getattr(self.art, "down_proj_q6_layers", []))
        ok = mine == theirs and cnt.get("Q5_K") == 37 and cnt.get("Q6_K") == 3 and len(downs) == 40
        return ok, len(downs), ("ffn_down_exps: Q5_K x%d, Q6_K x%d on layers %s, independently derived "
                                "from this parse and identical to artifact.py's %s"
                                % (cnt.get("Q5_K"), cnt.get("Q6_K"), mine, theirs)
                                if ok else "MISMATCH: this parse says %s on %s, artifact.py says %s, "
                                           "census %s" % (cnt, mine, theirs, ART.EXPECTED_CENSUS))

    def _c_width_delta(self):
        stored = sum(r.row for r in self.ruler)
        modelled = sum(GBG.row_bytes(r.n_elts, QS.POLICY_TYPE(POLICY_A, r.family)) for r in self.ruler)
        per_role = {}
        for r in self.ruler:
            pt = QS.POLICY_TYPE(POLICY_A, r.family)
            per_role[r.role] = per_role.get(r.role, 0) + r.row - GBG.row_bytes(r.n_elts, pt)
        exact = sum(v for k, v in per_role.items() if k in LAY.EXACT_ROLES)
        waiver = sum(v for k, v in per_role.items() if k in self.WAIVER_ROLES)
        wider = sum(v for k, v in per_role.items() if k in self.WIDER_ROLES)
        unexplained = {k: v for k, v in per_role.items()
                       if k not in LAY.EXACT_ROLES and k not in self.WAIVER_ROLES
                       and k not in self.WIDER_ROLES and v}
        residue = (stored - modelled) - exact - waiver - wider
        ok = residue == 0 and not unexplained and waiver != 0 and wider != 0
        self.width_delta = dict(stored=stored, modelled=modelled, exact=exact, waiver=waiver,
                               wider=wider, residue=residue)
        return ok, len(per_role), (
            "as-stored %d B minus policy-A modelled %d B = %+d B, attributed in full: never-quantized "
            "roles stored wider %+d B, priced-above-policy roles %+d B, the below-policy waiver %+d B, "
            "unexplained %+d B across %d roles"
            % (stored, modelled, stored - modelled, exact, wider, waiver, residue, len(unexplained)))

    def _c_staging_plan(self):
        bad, per_slab = [], {}
        for r in self.records:
            try:
                s = LAY.slab_of(r)
            except LAY.LayoutError as ex:
                # A width the layout will not place is exactly what the width veto is for; report it
                # as a red check rather than letting the exception end the enumeration.
                bad.append("%s: layout.py refuses to place it -- %s" % (r.name, ex))
                continue
            per_slab[s] = per_slab.get(s, 0) + r.row
        for s in LAY.SLAB_ORDER:
            sl = self.lay.slab(s)
            if per_slab.get(s, 0) != sl.content_bytes:
                bad.append("slab %s: this loader's own bytes %d, layout.py's %d"
                           % (s, per_slab.get(s, 0), sl.content_bytes))
        runs = self.stage_runs()
        outside = [r for r in runs
                   if r["file_off"] < 0 or r["file_off"] + r["length"] > self._size_of_source(r["slab"])]
        planned = sum(r["length"] for r in runs)
        content = sum(r.row for r in self.records)
        ok = not bad and not outside and planned == content and len(runs) > 0
        return ok, len(runs), ("%d copy runs carry %d B, equal to the %d B this loader priced itself; "
                               "every run lies inside the file it reads from (maximal runs, so the "
                               "allocation recipe gets %d memcpys and not %d)"
                               % (len(runs), planned, content, len(runs), len(self.records))
                               if ok else "BAD %s; %d run(s) outside their file" % (bad[:2], len(outside)))

    def _c_expert_slices(self):
        routed = [r for r in self.records if r.family == "routed_experts"]
        bad, sliced, refused, strided = [], 0, 0, 0
        for r in routed:
            if r.source == "hf_shards":
                try:
                    self.expert_range(r.name, 0)
                    bad.append("%s: a strided draft expert was handed back as a byte range" % r.name)
                except NotWired:
                    strided += 1
                except LoaderError as ex:
                    bad.append("%s: %s" % (r.name, ex))
                continue
            e = int(r.dims[self.expert_axis(r)])
            for idx in (0, 1, e // 2, e - 2, e - 1):
                try:
                    self.expert_range(r.name, idx)
                except LoaderError as ex:
                    bad.append("%s expert %d: %s" % (r.name, idx, ex))
                    continue
                sliced += 1
            for idx in (e, e + 1, -1, -e - 1):
                try:
                    self.expert_range(r.name, idx)
                    bad.append("%s ACCEPTED expert %d outside [0,%d)" % (r.name, idx, e))
                except LoaderError:
                    refused += 1
        n_gguf = len([r for r in routed if r.source == "gguf"])
        n_draft = len(routed) - n_gguf
        ok = (not bad and sliced == 5 * n_gguf and refused == 4 * n_gguf and strided == n_draft
              and n_gguf > 0)
        return ok, len(routed), (
            "%d gguf routed records x 5 sampled experts addressed and agreed with layout.py to the "
            "byte; %d out-of-range refusals (4 per record, so the bound is witnessed, not assumed); "
            "%d draft routed records refused as strided"
            % (n_gguf, refused, strided) if ok else "BAD %s (sliced %d, refused %d, strided %d)"
            % (bad[:3], sliced, refused, strided))

    def _c_draft_not_wired(self):
        refused, leaked = 0, []
        gap = 0
        for r in self.draft:
            want = GBG.row_bytes(r.n_elts, QS.POLICY_TYPE(POLICY_A, r.family))
            gap += r.row - want
            try:
                self.stage_draft(r.name)
                leaked.append(r.name)
            except NotWired:
                refused += 1
            except LoaderError as ex:
                leaked.append("%s (%s)" % (r.name, ex))
        ok = refused == len(self.draft) and not leaked and len(self.draft) > 0
        self.draft_gap = gap
        return ok, len(self.draft), (
            "all %d draft tensors refused staging: %d B of bf16 on disk would have to become %d B of "
            "policy-A widths (%+.4f GiB the requantizer owes) and no bf16 bytes were handed back"
            % (len(self.draft), sum(r.row for r in self.draft),
               sum(GBG.row_bytes(r.n_elts, QS.POLICY_TYPE(POLICY_A, r.family)) for r in self.draft),
               gap / GiB) if ok else "LEAKED %s" % leaked[:3])

    def resident_gib(self, view="policy-A"):
        return self.lay.weights_total_bytes(view) / GiB

    def _c_budget_lines(self):
        groups = {}
        for budget in ("development", "deployment"):
            _, lines = self.lay.verdict(self.resident_gib(), budget, basis="weights-only")
            groups[budget] = dict(ln.split("=", 1) for ln in lines if "=" in ln)
        bad = []
        for budget, kv in sorted(groups.items()):
            ev, v = kv.get("budget-bound-evidence"), kv.get("verdict", "")
            if ev == "unmeasured" and v != LAY.NOT_ADJUDICABLE:
                bad.append("%s: bound is %s but a fit verdict was printed (%s)"
                           % (budget, ev, v))
            if ev != "unmeasured" and v == LAY.NOT_ADJUDICABLE:
                bad.append("%s: bound is %s yet the verdict was withheld" % (budget, ev))
        if groups["development"]["budget-bound"] == groups["deployment"]["budget-bound"]:
            bad.append("both budgets print the same bound, which is the conflation D-11 forbids")
        self.budget_groups = groups
        return not bad, len(groups), (
            "2 budgets printed with different bounds and their own evidence states: development bound "
            "%s [%s] -> %s; deployment bound %s [%s] -> %s"
            % (groups["development"]["budget-bound"], groups["development"]["budget-bound-evidence"],
               groups["development"]["verdict"], groups["deployment"]["budget-bound"],
               groups["deployment"]["budget-bound-evidence"],
               "no verdict (unmeasured bound)" if groups["deployment"]["verdict"] == LAY.NOT_ADJUDICABLE
               else groups["deployment"]["verdict"])
            if not bad else "; ".join(bad))

    # ---- printing ------------------------------------------------------------------------------------
    def echo_records(self):
        """One line per resident tensor, then a trailer that states the count it printed.

        The trailer exists so a reader (and `engine/gate.py`) can tell 733 lines from 0 lines: a check
        that prints nothing cannot fail, and a table with no row count cannot be checked for one."""
        for r in self.records:
            print("RECORD %s|%s|%d|%d|%s|%s|%s" % (r.name, r.type_name, r.row, r.file_offset,
                                                   r.family, r.source, r.layer))
        print("RECORDS parsed=%d printed=%d" % (len(self.records), len(self.records)))
        return len(self.records)

    def report(self):
        print("artifact     %s (%d B), data_start %d, alignment %d"
              % (os.path.relpath(self.art.path, REPO), self.size, self.data_start,
                 self.ruler_hdr["alignment"]))
        print("draft block  %s (%d B), %d tensors"
              % (os.path.relpath(RB.MTP_PATH, REPO), self.draft_size, len(self.draft)))
        print("\nstored families (second tokenizer):")
        for t in sorted(self.family_table, key=lambda k: -self.family_table[k][1]):
            n, b = self.family_table[t]
            print("  %-8s %5d tensors %14d B  %8.4f GiB  %.4f bpw"
                  % (t, n, b, b / GiB, GBG.bpw_of(t)))
        print("  %-8s %5d tensors %14d B  %8.4f GiB"
              % ("TOTAL", len(self.ruler), sum(v[1] for v in self.family_table.values()),
                 sum(v[1] for v in self.family_table.values()) / GiB))
        d = getattr(self, "width_delta", None)
        if d:
            print("\nwidth attribution: as-stored %d B (%.4f GiB) vs policy-A modelled %d B (%.4f GiB); "
                  "delta %+d B = never-quantized %+d B + priced-above-policy %+d B + below-policy "
                  "waiver %+d B + unexplained %+d B"
                  % (d["stored"], d["stored"] / GiB, d["modelled"], d["modelled"] / GiB,
                     d["stored"] - d["modelled"], d["exact"], d["wider"], d["waiver"], d["residue"]))
        print("\nwidth rules: floor per family (policy A, derived bits/weight) %s"
              % ", ".join("%s>=%.4f" % (f, policy_floor_bpw(f))
                          for f in ("routed_experts", "gdn_linear_attn", "full_attn",
                                     "shared_experts", "embed", "lm_head", "routers", "norms_misc")))
        print("  below-floor waiver admitted by name: %s"
              % "; ".join("%s (%s)" % (k, v["note"]) for k, v in WIDTH_WAIVERS.items()))
        print("  priced-above-policy, no waiver needed: %s"
              % "; ".join("%s (%s)" % (k, v["note"]) for k, v in WIDTH_WIDER_ROLES.items()))
        print("\nresident set, per budget (D-11: two questions, never one answer):")
        for budget in ("development", "deployment"):
            total, lines = self.lay.verdict(self.resident_gib(), budget, basis="weights-only")
            print("  [%s]" % budget)
            for ln in lines:
                print("    %s" % ln)
            _, lines = self.lay.verdict(self.resident_gib(), budget,
                                        basis="weights-plus-measured-non-weight-floor")
            keep = [ln for ln in lines if ln.split("=")[0] in ("resident", "verdict")]
            print("    (weights + the measured non-weight floor) %s" % " | ".join(keep))
        plan = self.stage_runs()
        print("\nstaging plan: %d runs, %d B, slabs %s"
              % (len(plan), sum(r["length"] for r in plan),
                 ", ".join("%s=%d B/%d runs" % (s, self.lay.slab(s).bytes, len(self.lay.slab(s).runs))
                            for s in LAY.SLAB_ORDER)))


def open_loader(**kw):
    return Loader(**kw)


# -----------------------------------------------------------------------------------------------
# the injections. Each perturbs THIS loader's view in exactly one way, and each is required to turn a
# named check red in engine/gate.py -- a control nobody watched fire is a decoration (pitfalls/40).
# -----------------------------------------------------------------------------------------------
def _reprice(rec, type_name):
    qk, blk_b, _ = GWT.BLK[type_name]
    rec.type_name = type_name
    rec.row = -(-rec.n_elts // qk) * blk_b


def _f_wrong_width(l):
    r = l.by_name["blk.0.attn_qkv.weight"]
    _reprice(r, "Q4_K")


def _f_exact_role_quantized(l):
    _reprice(l.by_name["blk.0.ffn_gate_inp.weight"], "Q8_0")


def _f_lm_head_waiver_withdrawn(l):
    l.waivers.pop("output.weight", None)


def _f_missing_tensor(l):
    gone = l.by_name["blk.17.ffn_up_exps.weight"]
    l.records.remove(gone)
    l.ruler.remove(gone)


def _f_offset_past_eof(l):
    r = l.by_name["blk.3.ffn_up_exps.weight"]
    r.offset += 1 << 40
    r.file_offset = l.data_start + r.offset


def _f_ranges_overlap(l):
    a = l.by_name["blk.5.ffn_gate_exps.weight"]
    b = l.by_name["blk.5.ffn_up_exps.weight"]
    b.offset = a.offset
    b.file_offset = l.data_start + b.offset


def _f_data_start_mismatch(l):
    l.data_start += 32


def _f_file_truncated(l):
    l.size -= 4096


def _f_sha_mismatch(l):
    l.sha_override = "0" * 64


def _f_q6k_extra_layer(l):
    _reprice(l.by_name["blk.7.ffn_down_exps.weight"], "Q6_K")


def _f_expert_axis_tampered(l):
    """The AGENTS.md prediction-3 mistake: slice the split GGUF record along the wrong axis."""
    r = l.by_name["blk.3.ffn_gate_exps.weight"]
    r.dims = tuple(r.dims[:-1]) + (int(r.dims[-1]) - 1,)


def _f_unattributed_width(l):
    l.by_name["token_embd.weight"].row += 4096


def _f_run_past_file(l):
    """A copy plan whose first run reads past the end of the file it reads from."""
    l.lay.slab(LAY.SLAB_ORDER[0]).runs[0]["file_off"] += 1 << 40


def _f_draft_staged_anyway(l):
    l.stub_requantizer = True


def _f_budgets_conflated(l):
    LAY.BUDGETS["deployment"]["bound_measured"] = True


FAULTS = {
    "wrong_width": _f_wrong_width,
    "exact_role_quantized": _f_exact_role_quantized,
    "lm_head_waiver_withdrawn": _f_lm_head_waiver_withdrawn,
    "missing_tensor": _f_missing_tensor,
    "offset_past_eof": _f_offset_past_eof,
    "ranges_overlap": _f_ranges_overlap,
    "data_start_mismatch": _f_data_start_mismatch,
    "file_truncated": _f_file_truncated,
    "sha_mismatch": _f_sha_mismatch,
    "q6k_extra_layer": _f_q6k_extra_layer,
    "expert_axis_tampered": _f_expert_axis_tampered,
    "unattributed_width": _f_unattributed_width,
    "run_past_file": _f_run_past_file,
    "draft_staged_anyway": _f_draft_staged_anyway,
    "budgets_conflated": _f_budgets_conflated,
}


def main(argv):
    faults = argv[argv.index("--fault") + 1].split(",") if "--fault" in argv else []
    faults = [f for f in faults if f]
    L = Loader(faults=faults, verify_sha256="--sha256" in argv)
    res = L.run_checks()
    print("engine/loader.py — policy A's resident set%s"
          % (" [faults: %s]" % ",".join(faults) if faults else ""))
    print("\nchecks — every registered check prints an outcome; n=0 cannot pass:")
    for name, ok, n, detail in res:
        print("  [%s] %-32s n=%-5d %s" % ("PASS" if ok else "FAIL", name, n, detail))
    print()
    L.report()
    if "--records" in argv:
        print()
        L.echo_records()
    if "--probe" in argv:
        p = L.probe()
        print("\nstaging probe: marker selftest passed %s; %d runs planned, %d sampled, %d preads, "
              "%d B read, %d short" % (p["marker"], p["runs"], p["sampled"], p["reads"],
                                       p["bytes_read"], p["short"]))
        if p["short"]:
            print("FAIL: a planned range did not read its full window")
    failed = [r for r in res if not r[1]]
    print("\nLOADER checks=%d passed=%d failed=%d faults=%d"
          % (len(res), len(res) - len(failed), len(failed), len(faults)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
