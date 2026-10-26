#!/usr/bin/env python3
"""cross_check_loader.py — a SECOND ruler for `engine/loader.py`'s expert addressing.

Why this file exists. `engine/gate.py`'s `expert-slices-addressable` arm proves `loader.expert_range`
and `layout.expert_slice` agree — two expressions inside one directory, priced by one block table
(`tools/gguf_width_table.py §BLK`). That is self-agreement wearing a second name. This file brings in
a ruler that was built and gated somewhere else: `prototype/moe_gather/oracle_arm.py`'s
`read_expert_role()`, which derives one expert's byte span out of `tools/gguf_reader.py`'s header parse
and the block byte counts `moe_gather_kernels.h` defines (`MG_Q4K_BYTES / MG_Q5K_BYTES / MG_Q6_K_BYTES`,
mirrored into Python by `oracle_arm.read_header_constants()`). Two tokenizers, three width tables
(gguf_width_table.BLK, gguf_reader.BLOCK<-ggml_block_geom, the kernel header), one file: agreement is
evidence, self-agreement is not (pitfalls/40 §two-self-consistent-layouts).

What is compared, and to what precision — always to the BYTE, never to a tolerance:
  * `expert-address-bytes`   — `loader.expert_range(name, idx)`'s (file_off, length) against the span
                               `oracle_arm.read_expert_role` derives for the same (tensor, expert).
  * `expert-refusal-paired`  — an index outside [0,E) must be refused by BOTH, or the bound is a story.
  * `expert-axis-agreement`  — which axis is the expert axis (AGENTS.md prediction 3's whole point).
  * `ruler-header-agreement` — every one of the 733 gguf records and the 19 draft tensors: name, dims,
                               stored type, absolute file offset.
  * `data-span-closure`      — data_start + Sigma row_bytes == file size, residue 0 B, priced by ruler B.
  * `slab-offsets-repriced`  — every placement's `slab_off`, and every slab's bytes/pad/content,
                               re-packed from ruler B's row sizes and ruler B's own `general.alignment`.
  * `copy-runs-agree`        — the 464-run bulk-copy plan, re-cut from ruler B numbers, run for run.

No population is typed in: layers and E come from `checkpoints/Qwen3.6-35B-A3B.config.json` (through
`oracle_arm.reconcile_geometry()`), roles come from `oracle_arm.GGUF_TENSOR`, and the expert index set
is the four boundaries plus seven interior points derived as `(j*E)//8` — which contains the `E//2`
`engine/gate.py` samples, so the two gates overlap on a real index instead of each sampling its own.

Disagreement is a FINDING, never a reconciliation: both numbers print, `loser=` names which side lost
and `owner=` names the gate that owns the losing claim, and the run exits 1.

`NotWired` is a boundary, not a defect. `engine/loader.py` refuses to address a draft (hf_shards)
expert at all — `expert-slice-strided` — because a safetensors record's expert axis is `dims[0]` and one
expert is strided rows, not a byte range. This harness therefore compares draft experts only as a
*paired refusal* (engine raises `NotWired`, ruler B prices strided rows) and says so in its detail line;
it does not wire the path, which is `engine/`'s owner and device territory.

CPU-only, and provably so: the harness never opens the data section. The object handed to
`oracle_arm.read_expert_role` is a capture shim whose `read()` records the (start, length) the prototype
derived and returns zeros; `gr.Gguf.read`/`open_data` are replaced by a tripwire that raises, so an
accidental data read is a hard error rather than a silent 22 GiB. Takes no device lock. Prints no
duration, no rate, no `ms`.

    cross_check_loader.py                     the honest comparison
    cross_check_loader.py --selftest          every injected disagreement, each witnessed firing
    cross_check_loader.py --arm expert_off_by_one    one injected arm (what --selftest drives)
    cross_check_loader.py --keep-evidence PATH       also write this run's stdout to PATH
"""
import argparse
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "engine"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gguf_reader as gr                 # noqa: E402  ruler B's header parse (own type-id table)
import loader as ENG                     # noqa: E402  the code under cross-check (READ-ONLY here)
import layout as LAY                     # noqa: E402  the slabs and the copy plan (READ-ONLY)
import oracle_arm as OA                  # noqa: E402  the separately gated prototype's own derivation
import residency_budget as RB            # noqa: E402  committed paths

GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
OWNER_ENGINE = "engine/gate.py expert-slices-addressable / staging-plan-matches-ruler"
OWNER_PROTO = "prototype/moe_gather/gate.py (oracle_arm.read_expert_role)"


class XCError(Exception):
    pass


# -----------------------------------------------------------------------------------------------
# ruler B: the prototype's ruler, plus the injections that make the comparator falsifiable
# -----------------------------------------------------------------------------------------------
class RulerB:
    """`gguf_reader`'s header parse + the kernel header's block sizes, with named perturbations.

    Every injection moves RULER B, never the engine: the point is to prove the comparator sees a
    difference, not to prove the engine can be broken from out here.
    """

    INJECTIONS = ("expert_off_by_one", "expert_len_by_one", "axis_swap", "refusal_ignored",
                  "width_table_stale", "alignment_wrong", "file_off_shift", "type_swap",
                  "population_shrunk", "empty_ruler")

    def __init__(self, inject=None):
        self.inject = inject
        self._tripwire()
        self.gg = gr.Gguf(GGUF)
        self.align = int(self.gg.alignment)
        # Ruler B's OWN block table, copied. `width_table_stale` moves this copy and nothing else:
        # oracle_arm.reconcile_geometry() compares gr.BLOCK against the kernel header, so editing the
        # shared table would be caught by the prototype's own geometry gate before this harness could
        # show what a stale row does to a comparison. tools/gguf_width_table.py §BLK records that this
        # class of stale row (Q5_K at 5.0 bpw) was a real bug in three tools at once.
        self.block = dict(gr.BLOCK)
        if inject == "width_table_stale":
            self.block["Q5_K"] = (256, 160)
        if inject == "alignment_wrong":
            self.align = self.align + 1      # 33: guarantees a packing delta on any odd-sized row
        self.n_tensors = int(self.gg.n_tensors)         # the count the header DECLARES
        self.spans = {}                                # name -> (file_off, row, type_name, dims)
        for t in self.gg.tensors:
            tname = t.type_name
            if inject == "type_swap" and tname in ("Q5_K", "Q6_K"):
                # The kind of bug two type-id tables make possible: gguf_reader.GGML_TYPES and
                # gguf_header_types.TYPES are separate transcriptions of ggml.h, and if one mislabels
                # id 13/14 the widths trade places. Prices different bytes, so it must be caught.
                tname = "Q6_K" if tname == "Q5_K" else "Q5_K"
            row = self.row_bytes(tname, t.n_elts)
            foff = self.gg.data_start + t.offset
            if inject == "file_off_shift" and t.name == self.SHIFT_TENSOR:
                foff += self.align
            self.spans[t.name] = (foff, row, tname, tuple(t.dims))
        self._draft_spans()
        if inject == "empty_ruler":
            self.spans = {}

    SHIFT_TENSOR = "blk.17.ffn_gate_exps.weight"

    def row_bytes(self, tname, n_elts):
        be, bb = self.block[tname]
        return (bb * int(n_elts) + be - 1) // be

    def close(self):
        self.gg.close()

    def _tripwire(self):
        """Any read of the data section from here on is a bug in THIS harness, so make it loud."""
        def _no_read(*a, **k):
            raise XCError("cross_check tried to read tensor data; the capture shim failed")
        gr.Gguf.read = _no_read
        gr.Gguf.open_data = _no_read

    def _draft_spans(self):
        """The draft tensors' spans, read from the safetensors header by this file's own parse."""
        with open(RB.MTP_PATH, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        self.draft_base = base
        self.draft_size = os.path.getsize(RB.MTP_PATH)
        self.draft_keys = len([k for k in hdr if k != "__metadata__"])
        for name, d in hdr.items():
            if name == "__metadata__":
                continue
            lo, hi = d["data_offsets"]
            self.spans[name] = (base + lo, hi - lo, d["dtype"], tuple(d["shape"]))

    def span(self, name):
        if name not in self.spans:
            raise XCError("ruler B has no tensor %r" % name)
        return self.spans[name]

    def expert_axis(self, dims):
        """Ruler B's own answer to "which axis is the expert axis" — the prototype's unpack order."""
        return 0 if self.inject == "axis_swap" else len(dims) - 1


# -----------------------------------------------------------------------------------------------
# the prototype's live derivation, captured instead of re-typed
# -----------------------------------------------------------------------------------------------
class Capture:
    """Stand-in for `gr.Gguf` that keeps the byte span `read_expert_role` derives and reads nothing."""

    def __init__(self, gg):
        self.by_name = gg.by_name
        self.data_start = gg.data_start
        self.got = None
        self.reads = 0
        self.zeros = 0

    def read(self, token, start, length, why=""):
        if self.got is not None:
            raise XCError("read_expert_role read twice; the capture would keep only one span")
        self.got = (int(start), int(length))
        self.reads += 1
        self.zeros += int(length)
        return b"\x00" * int(length)


def proto_span(cap, block_index, role, expert, inject=None):
    """(file_off, length, tname, rows, cols) for one expert, as `oracle_arm` derives it right now."""
    cap.got = None
    try:
        raw, tname, rows, cols, _idx = OA.read_expert_role(cap, block_index, role, expert)
    except OA.GateError as ex:
        return None, None, None, None, None, "refused: %s" % ex
    if cap.got is None:
        raise XCError("oracle_arm derived no address for blk.%d %s expert %d" % (block_index, role, expert))
    start, length = cap.got
    if len(raw) != length:
        raise XCError("oracle_arm returned %d bytes for the %d B it addressed" % (len(raw), length))
    if inject == "expert_off_by_one":
        start += 1
    if inject == "expert_len_by_one":
        length += 1
    return start, length, tname, rows, cols, None


# -----------------------------------------------------------------------------------------------
# the population, derived from geometry
# -----------------------------------------------------------------------------------------------
def derive_population(ruler, inject=None):
    """{engine_tensor_name: (block_index, role, [expert indices])} for every routed gguf record.

    L and E are read from the pinned config, roles from the prototype's own table, and the index set is
    four boundaries plus the derived interior lattice `(j*E)//8`. Nothing here is a hand-picked list.
    """
    cfg = OA.reconcile_geometry()
    L = int(cfg["num_hidden_layers"])
    E = int(OA.HC["MG_E"])
    if int(cfg["num_experts"]) != E:
        raise XCError("config num_experts %d != MG_E %d" % (cfg["num_experts"], E))
    boundaries = [0, 1, E - 2, E - 1]
    interior = sorted({(j * E) // 8 for j in range(1, 8)} - set(boundaries) - {2})
    idxs = sorted(set(boundaries) | set(interior))
    if inject == "population_shrunk":
        idxs = [0]
    pop = []
    for li in range(L):
        for role in sorted(OA.GGUF_TENSOR):
            name = OA.GGUF_TENSOR[role] % li
            pop.append((name, li, role, idxs))
    refusals = [E, E + 1, -1, -E - 1]
    return dict(pop=pop, L=L, E=E, idxs=idxs, refusals=refusals,
                boundaries=len(boundaries), interior=len(interior))


# -----------------------------------------------------------------------------------------------
# the comparison
# -----------------------------------------------------------------------------------------------
class CrossCheck:
    """One pass of the two rulers over one population. `findings` is what a disagreement becomes."""

    CAP_PRINT = 5           # disagreement lines printed per check; the count is always exact

    def __init__(self, inject=None, verbose=True):
        self.inject = inject
        self.ld = ENG.Loader()
        self.b = RulerB(inject)
        self.cap = Capture(self.b.gg)
        self.pop = derive_population(self.b, inject)
        self.findings = []
        self.fired = set()
        self.file_ok = True        # ruler B closes against the file, so it may arbitrate a tie

    def close(self):
        self.b.close()

    # ---- findings ----------------------------------------------------------------------
    def disagree(self, check, what, eng_val, proto_val, loser, owner):
        """A disagreement is recorded with BOTH numbers and the gate that owns the losing claim."""
        self.findings.append(dict(check=check, what=what, engine=eng_val, proto=proto_val,
                                  loser=loser, owner=owner))
        self.fired.add(check)

    def _arbiter(self, name, rec, idx, eng, proto):
        """Who lost? The file decides, by two facts neither side authored.

        The header DECLARES each tensor's base offset, and the row that closes
        `data_start + Sigma row == file size` (residue 0) fixes how many bytes one of E equal experts
        must be. A side is 'consistent with the file' if its length x E is that row and its start is
        base + idx x its own length. If both sides or neither are consistent, say `unadjudicated` --
        that is a real state, it happens whenever ruler B itself is the injected side, and it stays red.
        """
        foff_b, row_b, _tn, _dims = self.b.span(name)
        if not self.file_ok:
            return "unadjudicated"
        e = int(rec.dims[self.ld.expert_axis(rec)])

        def consistent(off, length):
            return length * e == row_b and off == foff_b + idx * length

        eng_ok = consistent(eng["file_off"], eng["length"])
        pro_ok = consistent(proto[0], proto[1])
        if eng_ok and not pro_ok:
            return "prototype"
        if pro_ok and not eng_ok:
            return "engine"
        return "unadjudicated"

    # ---- the checks --------------------------------------------------------------------
    def c_population(self):
        p = self.pop
        missing_eng = [n for n, _b, _r, _i in p["pop"] if n not in self.ld.by_name]
        missing_b = [n for n, _b, _r, _i in p["pop"] if n not in self.b.spans]
        bad_axis = [i for i in p["idxs"] if not 0 <= i < p["E"]]
        expected = len(p["idxs"]) * len(p["pop"])
        n = expected
        ok = (p["L"] > 0 and p["E"] > 1 and len(p["idxs"]) == p["boundaries"] + p["interior"]
              and len(p["idxs"]) >= 5 and not bad_axis and not missing_eng and not missing_b
              and expected > 0)
        if not ok:
            self.disagree("population-derived", "derived population", "%s" % expected,
                          "missing_engine=%d missing_ruler_b=%d idxs=%d bad=%s"
                          % (len(missing_eng), len(missing_b), len(p["idxs"]), bad_axis[:4]),
                          "comparator", "cross_check_loader.py --selftest population_shrunk")
        return ok, n, ("%d layers x %d routed roles x %d expert indices (%d boundary + %d "
                       "interior-derived) = %d address comparisons over %d records; every name "
                       "present in BOTH rulers"
                       % (p["L"], len(p["pop"]) // max(p["L"], 1), len(p["idxs"]), p["boundaries"],
                          p["interior"], expected, len(p["pop"])))

    def c_ruler_header(self):
        bad = []
        n = 0
        for rec in self.ld.records:
            if rec.name not in self.b.spans:
                bad.append("%s absent from ruler B" % rec.name)
                continue
            foff, row, tname, dims = self.b.span(rec.name)
            n += 1
            if (rec.file_offset, rec.row, rec.type_name, tuple(rec.dims)) != (foff, row, tname, dims):
                if not bad:
                    bad.append("%s engine=(%s,%s,%s,%s) rulerB=(%s,%s,%s,%s)"
                               % (rec.name, rec.file_offset, rec.row, rec.type_name, rec.dims,
                                  foff, row, tname, dims))
                self.disagree("ruler-header-agreement", rec.name,
                              "%s/%s/%s/%s" % (rec.file_offset, rec.row, rec.type_name, rec.dims),
                              "%s/%s/%s/%s" % (foff, row, tname, dims),
                              "unadjudicated", "%s vs %s" % (OWNER_ENGINE, OWNER_PROTO))
        want = self.b.n_tensors + self.b.draft_keys
        ok = (n > 0 and not bad and n == len(self.b.spans) and want == len(self.ld.records))
        return ok, n, ("%d records agree on name, dims, stored width and absolute file offset across "
                       "two tokenizers (%d gguf declared by ruler B's header + %d draft tensors = "
                       "%d the engine parsed)" % (n, self.b.n_tensors, self.b.draft_keys,
                                                  len(self.ld.records)) if ok else
                       "BAD %d of %d: %s" % (len(bad), n, bad[0] if bad else "no record parsed"))

    def c_data_closure(self):
        gg_sum = sum(self.b.span(r.name)[1] for r in self.ld.records if r.source == "gguf"
                     and r.name in self.b.spans)
        draft_sum = sum(self.b.span(r.name)[1] for r in self.ld.records if r.source != "gguf"
                        and r.name in self.b.spans)
        gg_size = os.path.getsize(GGUF)
        residue_g = gg_size - self.b.gg.data_start - gg_sum
        residue_d = self.b.draft_size - self.b.draft_base - draft_sum
        n = len(self.ld.records)
        ok = n > 0 and residue_g == 0 and residue_d == 0
        if not ok:
            self.disagree("data-span-closure", "file-size identity",
                          "engine rows %d" % sum(r.row for r in self.ld.records),
                          "rulerB rows %d/%d residue %d/%d" % (gg_sum, draft_sum, residue_g, residue_d),
                          "prototype", OWNER_PROTO)
        return ok, n, ("ruler B prices %d B of gguf data + %d B of draft with residue %d B and %d B "
                       "(both must be 0: %d records, two files)"
                       % (gg_sum, draft_sum, residue_g, residue_d, n))

    def c_expert_bytes(self):
        cmps = agree = 0
        for name, blk, role, idxs in self.pop["pop"]:
            rec = self.ld.by_name.get(name)
            if rec is None:
                continue
            for idx in idxs:
                cmps += 1
                poff, plen, ptname, prows, pcols, refusal = proto_span(
                    self.cap, blk, role, idx, self.inject)
                try:
                    eng = self.ld.expert_range(name, idx)
                except ENG.LoaderError as ex:
                    self.disagree("expert-address-bytes", "%s expert %d" % (name, idx),
                                  "refused: %s" % ex, "%s/%s" % (poff, plen),
                                  "engine", OWNER_ENGINE)
                    continue
                if refusal is not None:
                    self.disagree("expert-address-bytes", "%s expert %d" % (name, idx),
                                  "%d/%d" % (eng["file_off"], eng["length"]), refusal,
                                  "prototype", OWNER_PROTO)
                    continue
                if (eng["file_off"], eng["length"]) == (poff, plen):
                    agree += 1
                    continue
                loser = self._arbiter(name, rec, idx, eng, (poff, plen))
                self.disagree("expert-address-bytes", "%s expert %d" % (name, idx),
                              "%d/%d" % (eng["file_off"], eng["length"]),
                              "%d/%d" % (poff, plen), loser,
                              OWNER_ENGINE if loser == "engine" else
                              (OWNER_PROTO if loser == "prototype" else
                               "%s AND %s" % (OWNER_ENGINE, OWNER_PROTO)))
        ok = cmps > 0 and agree == cmps
        return ok, cmps, ("%d of %d expert byte ranges agree EXACTLY (file_off and length) between "
                          "engine/loader.expert_range and oracle_arm.read_expert_role"
                          % (agree, cmps) if ok else
                          "%d of %d disagree; first: %s" % (len(self.finded("expert-address-bytes")),
                                                            cmps, self.first("expert-address-bytes")))

    def c_refusals(self):
        p = self.pop
        paired = 0
        for name, blk, role, _idxs in p["pop"]:
            for idx in p["refusals"]:
                eref = None
                try:
                    self.ld.expert_range(name, idx)
                except ENG.LoaderError as ex:
                    eref = ex.tag
                _po, _pl, _tn, _r, _c, pref = proto_span(self.cap, blk, role, idx, self.inject)
                if pref and self.inject == "refusal_ignored":
                    # The injected failure mode this check exists for: a ruler that wraps an
                    # out-of-range expert id instead of refusing it. It hands back an address, and a
                    # silently-wrapped -1 reads a legal expert and produces a plausible wrong sum.
                    pref = None
                if eref and pref:
                    paired += 1
                else:
                    self.disagree("expert-refusal-paired", "%s expert %d" % (name, idx),
                                  "refused(%s)" % eref, pref or "ADDRESSED a wrapped index",
                                  "prototype" if not pref else "engine",
                                  OWNER_PROTO if not pref else OWNER_ENGINE)
        draft = [r for r in self.ld.records if r.family == "routed_experts" and r.source != "gguf"]
        draft_note = []
        for r in draft:
            try:
                self.ld.expert_range(r.name, 0)
                self.disagree("expert-refusal-paired", "%s draft expert 0" % r.name,
                              "ADDRESSED", "no gguf record", "engine", OWNER_ENGINE)
            except ENG.NotWired as ex:
                draft_note.append("%s:%s" % (r.name.split(".")[-1], ex.tag))
        need = len(p["pop"]) * len(p["refusals"])
        ok = paired > 0 and paired == need and not self.finded("expert-refusal-paired")
        return ok, paired, ("%d out-of-range refusals paired across both rulers (%d records x %d "
                            "indices), and %d draft routed records refused as strided [%s] — the "
                            "NotWired boundary, stated not bridged"
                            % (paired, len(p["pop"]), len(p["refusals"]), len(draft),
                               ",".join(draft_note)) if ok else
                            "BAD paired=%d need=%d %s" % (paired, need,
                                                          self.first("expert-refusal-paired")))

    def c_axis(self):
        agree = 0
        for name, _blk, _role, _idxs in self.pop["pop"]:
            rec = self.ld.by_name.get(name)
            if rec is None or name not in self.b.spans:
                continue
            e_ax = self.ld.expert_axis(rec)
            b_ax = self.b.expert_axis(self.b.span(name)[3])
            e_e = int(rec.dims[e_ax])
            b_e = int(self.b.span(name)[3][b_ax])
            if e_ax == b_ax and e_e == b_e == self.pop["E"]:
                agree += 1
            else:
                self.disagree("expert-axis-agreement", name, "axis %d wide %d" % (e_ax, e_e),
                              "axis %d wide %d" % (b_ax, b_e),
                              "prototype" if b_e != self.pop["E"] else "engine",
                              OWNER_PROTO if b_e != self.pop["E"] else OWNER_ENGINE)
        n = len(self.pop["pop"])
        ok = n > 0 and agree == n
        return ok, n, ("both rulers name axis %d (the trailing, slowest GGUF axis) as the expert axis "
                       "on all %d routed records, E=%d" % (self.ld.expert_axis(self.ld.by_name[
                           self.pop["pop"][0][0]]), n, self.pop["E"]) if ok else
                       "BAD %d of %d: %s" % (agree, n, self.first("expert-axis-agreement")))

    def c_slab_offsets(self):
        bad = 0
        n = 0
        per_slab = {}
        for s in LAY.SLAB_ORDER:
            sl = self.ld.lay.slab(s)
            off = 0
            pad_sum = content = 0
            for p in sl.placements:
                n += 1
                if p.rec.name not in self.b.spans:
                    bad += 1
                    self.disagree("slab-offsets-repriced", p.rec.name, "slab_off %d" % p.slab_off,
                                  "absent from ruler B", "unadjudicated",
                                  "%s vs %s" % (OWNER_ENGINE, OWNER_PROTO))
                    continue
                row_b = self.b.span(p.rec.name)[1]
                pad = (-off) % self.b.align
                want = off + pad
                if want != p.slab_off:
                    bad += 1
                    if bad == 1:
                        self.disagree("slab-offsets-repriced", "%s placement" % p.rec.name,
                                      "slab_off %d" % p.slab_off, "repriced %d (pad %d, row_B %d)"
                                      % (want, pad, row_b), "unadjudicated",
                                      "%s vs %s" % (OWNER_ENGINE, OWNER_PROTO))
                off = want + row_b
                pad_sum += pad
                content += row_b
            n += 1
            for field, want, got in (("bytes", off, sl.bytes), ("pad_bytes", pad_sum, sl.pad_bytes),
                                     ("content_bytes", content, sl.content_bytes)):
                if want != got:
                    bad += 1
                    self.disagree("slab-offsets-repriced", "%s %s" % (s, field), "%d" % got,
                                  "repriced %d at align %d" % (want, self.b.align), "unadjudicated",
                                  "%s vs %s" % (OWNER_ENGINE, OWNER_PROTO))
            per_slab[s] = (off, len(sl.placements))
        ok = n > 0 and bad == 0
        return ok, n, ("%d placements and %d slabs re-packed from ruler B's row sizes at align %d "
                       "(ruler B's own general.alignment) land on every slab_off, pad and total the "
                       "engine planned: %s" % (n - len(LAY.SLAB_ORDER), len(LAY.SLAB_ORDER),
                                               self.b.align,
                                               " ".join("%s=%d/%d" % (k, v[0], v[1])
                                                        for k, v in sorted(per_slab.items())))
                       if ok else "BAD %d: %s" % (bad, self.first("slab-offsets-repriced")))

    def c_runs(self):
        mine = []
        for s in LAY.SLAB_ORDER:
            sl = self.ld.lay.slab(s)
            off = 0
            runs = []
            for p in sl.placements:
                if p.rec.name not in self.b.spans:
                    continue
                foff, row_b, _tn, _d = self.b.span(p.rec.name)
                pad = (-off) % self.b.align
                off += pad
                if runs and runs[-1]["file_end"] == foff and runs[-1]["slab_end"] == off:
                    last = runs[-1]
                    last["length"] += row_b
                    last["file_end"] = foff + row_b
                    last["slab_end"] = off + row_b
                    last["n"] += 1
                else:
                    runs.append(dict(slab=s, slab_off=off, file_off=foff, length=row_b,
                                     file_end=foff + row_b, slab_end=off + row_b, n=1))
                off += row_b
            mine.extend(runs)
        theirs = self.ld.stage_runs()
        keys = ("slab", "slab_off", "file_off", "length", "n")
        bad = []
        if len(mine) != len(theirs):
            bad.append("run count %d vs %d" % (len(mine), len(theirs)))
        else:
            for a, b in zip(mine, theirs):
                if any(a[k] != b[k] for k in keys):
                    bad.append("%s engine=%s repriced=%s" % (
                        b["slab"], [b[k] for k in keys], [a[k] for k in keys]))
        planned = sum(r["length"] for r in theirs)
        my_total = sum(r["length"] for r in mine)
        ok = len(theirs) > 0 and not bad and planned == my_total
        if bad:
            self.disagree("copy-runs-agree", "bulk-copy plan",
                          "%d runs / %d B" % (len(theirs), planned),
                          "%d runs / %d B; first diff %s" % (len(mine), my_total, bad[0]),
                          "unadjudicated", "%s vs %s" % (OWNER_ENGINE, OWNER_PROTO))
        return ok, len(theirs), ("%d copy runs, %d B, re-cut from ruler B's file offsets, rows and "
                                 "alignment: identical run for run in slab, slab_off, file_off, "
                                 "length and tensors-per-run" % (len(theirs), planned)
                                 if ok else "BAD %s" % bad[0])

    # ---- plumbing ----------------------------------------------------------------------
    def finded(self, check):
        return [f for f in self.findings if f["check"] == check]

    def first(self, check):
        f = self.finded(check)
        if not f:
            return "none"
        return "%s engine=%s proto=%s loser=%s owner=%s" % (f[0]["what"], f[0]["engine"],
                                                           f[0]["proto"], f[0]["loser"], f[0]["owner"])

    CHECKS = (("population-derived", "c_population"),
              ("ruler-header-agreement", "c_ruler_header"),
              ("data-span-closure", "c_data_closure"),
              ("expert-address-bytes", "c_expert_bytes"),
              ("expert-refusal-paired", "c_refusals"),
              ("expert-axis-agreement", "c_axis"),
              ("slab-offsets-repriced", "c_slab_offsets"),
              ("copy-runs-agree", "c_runs"))

    def run(self):
        out = []
        for name, fn in self.CHECKS:
            ok, n, detail = getattr(self, fn)()
            out.append((name, ok, n, detail))
            # The file can only arbitrate a disagreement if ruler B itself still closes against it.
            if name in ("ruler-header-agreement", "data-span-closure") and not ok:
                self.file_ok = False
        # Anti-vacuity: if the prototype never derived anything, nothing was compared. A comparison
        # that called no live code is self-agreement with an extra step.
        if self.cap.reads == 0:
            out.append(("capture-nonvacuous", False, 0,
                        "BAD oracle_arm.read_expert_role derived no address: the capture read 0 spans"))
        else:
            out.append(("capture-nonvacuous", True, self.cap.reads,
                        "%d spans came out of the live prototype code (0 file bytes; %d B of synthetic "
                        "zeros handed back so the prototype's own length check stayed honest)"
                        % (self.cap.reads, self.cap.zeros)))
        return out


# -----------------------------------------------------------------------------------------------
# driver — and the injections that make every check above falsifiable
# -----------------------------------------------------------------------------------------------
# Each arm moves ONE input, and names the check that must turn red because of it. An arm that turns
# nothing red is a check that cannot fail, which is the same as no check (pitfalls/40).
MUST_FIRE = {
    "expert_off_by_one": ("expert-address-bytes",),
    "expert_len_by_one": ("expert-address-bytes",),
    "axis_swap": ("expert-axis-agreement",),
    "refusal_ignored": ("expert-refusal-paired",),
    "width_table_stale": ("data-span-closure",),
    "alignment_wrong": ("slab-offsets-repriced",),
    "file_off_shift": ("ruler-header-agreement",),
    "type_swap": ("ruler-header-agreement",),
    "population_shrunk": ("population-derived",),
    "empty_ruler": ("ruler-header-agreement",),
}


def run_arm(arm, keep=None):
    """One pass, printed. Exit 1 unless every check is green AND every population is non-empty."""
    xc = CrossCheck(arm)
    try:
        results = xc.run()
    finally:
        xc.close()
    lines = []
    for name, ok, n, detail in results:
        lines.append("XCHECK name=%s ok=%d n=%d %s" % (name, 1 if ok else 0, n, detail))
    for f in xc.findings[:CrossCheck.CAP_PRINT]:
        lines.append("XDISAGREE check=%s what=%s engine=%s proto=%s loser=%s owner=%s"
                     % (f["check"], f["what"], f["engine"], f["proto"], f["loser"], f["owner"]))
    if len(xc.findings) > CrossCheck.CAP_PRINT:
        lines.append("XDISAGREE_TRUNCATED hidden=%d" % (len(xc.findings) - CrossCheck.CAP_PRINT))
    empty = [name for name, ok, n, _d in results if n <= 0]
    failing = [name for name, ok, _n, _d in results if not ok]
    passed = not failing and not empty and not xc.findings
    lines.append("XSUM arm=%s checks=%d findings=%d disagree_lines_printed=%d failing=%d "
                 "empty_parse=%d data_reads=%d capture_reads=%d"
                 % (arm or "honest", len(results), len(xc.findings),
                    min(len(xc.findings), CrossCheck.CAP_PRINT), len(failing), len(empty),
                    0, xc.cap.reads))
    lines.append("RESULT %s cross-check-engine-loader arm=%s checks=%d findings=%d"
                 % ("PASS" if passed else "FAIL", arm or "honest", len(results), len(xc.findings)))
    lines.append("NOTE cpu-only: no device lock taken; tensor-data reads=0 (the tripwire on "
                 "gguf_reader's data window raised 0 times); no timing taken")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if keep:
        open(keep, "w").write(text)
    return 0 if passed else 1


# what the selftest parent requires of each injected arm: exit 1, and the checks that must be red.
# That asymmetry is the whole point -- the honest arm is green and every injection turns something red.
ARM_WANT = dict((k, (1, v)) for k, v in MUST_FIRE.items())


def selftest(keep=None):
    """Run the honest arm and every injected arm; require each injection to turn a NAMED check red.

    The parent parses the child's own printed records and requires parsed == printed for every arm, so
    an arm that printed nothing cannot pass and an arm whose parser dropped a line cannot pass.
    """
    keep = keep or os.path.join(HERE, "work", "cross_check_selftest.log")
    if not os.path.isdir(os.path.dirname(keep)):
        os.makedirs(os.path.dirname(keep))
    log = []
    ok = True
    rc, out = _child(None)
    log.append("===== arm=honest =====\n" + out)
    sys.stdout.write(out)                       # the honest detail is the detail worth reading
    parsed, printed = out.count("XCHECK "), _xsum(out, "checks")
    good = rc == 0 and "RESULT PASS" in out and parsed == printed and parsed > 0
    line = "XSELFTEST arm=honest exit=%d parsed_xcheck=%d printed_checks=%s ok=%d"
    line = line % (rc, parsed, printed, 1 if good else 0)
    print(line)
    log.append(line + "\n")
    ok = ok and good
    for arm in RulerB.INJECTIONS:
        rc, out = _child(arm)
        log.append("===== arm=%s =====\n" % arm + out)
        fired = set()
        for ln in out.splitlines():
            if ln.startswith("XCHECK ") and " ok=0 " in ln + " ":
                fired.add(ln.split()[1].split("=", 1)[1])
            elif ln.startswith("XDISAGREE check="):
                fired.add(ln.split()[1].split("=", 1)[1])
        p2, pr2 = out.count("XCHECK "), _xsum(out, "checks")
        good = (rc == 1 and set(MUST_FIRE[arm]) <= fired and p2 == pr2 and pr2 is not None
                and "RESULT FAIL" in out)
        line = ("XSELFTEST arm=%s exit=%d fired=%s must=%s parsed_xcheck=%d printed_checks=%s ok=%d"
                % (arm, rc, ",".join(sorted(fired)) or "NONE", ",".join(MUST_FIRE[arm]), p2, pr2,
                   1 if good else 0))
        print(line)
        log.append(line + "\n")
        ok = ok and good
    open(keep, "w").write("".join(log))
    print("EVIDENCE file=%s bytes=%d" % (keep, os.path.getsize(keep)))
    print("RESULT %s cross-check-selftest arms=%d injections=%d"
          % ("PASS" if ok else "FAIL", 1 + len(RulerB.INJECTIONS), len(RulerB.INJECTIONS)))
    return 0 if ok else 1


def _child(arm):
    cmd = [PY, os.path.abspath(__file__)]
    if arm:
        cmd += ["--arm", arm]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + (("\n" + p.stderr) if p.stderr.strip() else "")


def _xsum(text, key):
    for line in text.splitlines():
        if line.startswith("XSUM "):
            for tok in line.split():
                if tok.startswith(key + "="):
                    return int(tok.split("=", 1)[1])
    return None


def main(argv):
    ap = argparse.ArgumentParser(prog="cross_check_loader.py")
    ap.add_argument("--arm", choices=sorted(RulerB.INJECTIONS), default=None,
                    help="inject one named perturbation into ruler B (what --selftest drives)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the honest arm plus every injected arm and require each to fire")
    ap.add_argument("--keep-evidence", default=None, help="write this run's output to PATH as well")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.keep_evidence)
    return run_arm(a.arm, a.keep_evidence)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
