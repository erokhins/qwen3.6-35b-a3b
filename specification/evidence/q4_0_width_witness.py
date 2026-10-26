#!/usr/bin/env python3
"""q4_0_width_witness.py — the two width rulers on Q4_0 and IQ4_NL: adjudicated, corrected, witnessed.

`tools/gguf_width_table.py` and `tools/ggml_block_geom.py` are the repo's two independent width rulers;
`tools/ulp_policy.py` §WidthClass refuses to price any type they disagree about, which is fail-closed but
silent. On 2026-10-26 `specification/evidence/ulp-policy-migration-2026-10-26.md` §1 found they disagreed
on Q4_0 — 8.5 vs 4.5 bpw — and nothing caught it, because gguf_width_table.py's authority is the
file-size identity `sum(tensor row bytes) + data_start == file size` and the pinned GGUF contains no
Q4_0 tensor, so the identity is vacuous for that row. That is `pitfalls/40` §"two self-consistent
layouts that nothing compares", in the wild. The witness found a second row the same way: IQ4_NL.

**State as of 2026-10-26 (later the same day): both rows are corrected**, locally and with provenance —
`BLK["Q4_0"] = (32, 18, "sizeof(ggml_half) + QK4_0/2")` and
`BLK["IQ4_NL"] = (32, 18, "sizeof(ggml_half) + QK4_NL/2")`, the arithmetic being
`QK4_0 = 32` (`ggml-common.h:194`) with `block_q4_0` = `ggml_half d` (2 B) + `uint8_t qs[QK4_0/2]`
(16 B) = 18 B per 32 elts = 4.5 bpw, which is also what that struct's own
`static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2)` (:199) evaluates to; and the same
shape for `block_iq4_nl` under `QK4_NL = 32` (:447, `static_assert` at :452). The correction and its
arithmetic are recorded in `tools/PROVENANCE.md` §gguf_width_table.py, because an unrecorded divergence
in a copied executable is how a silent fork starts. So the check this file was written to make red —
`two-width-rulers-agree-on-every-type` — is now expected GREEN, and it is green on its own merits: the
table's expression string is no longer a hand-typed guess, it is the header's own expression and it
evaluates against the header's own macros to the header's own byte count (§3 below).

What this file still owes, and now carries:

  1. NAMED AGREEMENT. `diff_rulers` diffs the two rulers over the whole type universe (the pinned
     file's own type census, plus every type `ggml_block_geom.REQUIRED` demands, plus every row of
     `BLK`) and prints a `WIDTHDRIFT` line per disagreement. The check requires the drift list to be
     empty AND requires every type in the universe to have actually been compared, so a type skipped
     by `continue` cannot agree by silence.
  2. ITS NON-VACUITY. The two pre-correction rows are kept here (`STALE_ROWS`, named, with what each
     got wrong) and injected into `GWT.BLK` at run time, one type at a time, to show the agreement
     check firing again — same code path, opposite expectation — plus a check that each injection
     really differs from both the geometry and the shipped row, and a check that every injected row
     was restored. A counter that reads zero is vacuous until it has been witnessed non-zero where it
     must fire (`pitfalls/40` §"count reachability, not just checks").
  3. SOMETHING TO BITE ON. Synthetic GGUFs containing Q4_0 and IQ4_NL tensors whose byte length is
     derived from the pinned struct geometry, run through gguf_width_table.py's OWN `parse_gguf` and
     `family_bytes` under three tables: the derived geometry (must reproduce the file), the corrected
     `BLK` row (must reproduce it too — that is the regression catcher, if someone retypes 34 this
     fires), and the injected stale row (must break it at every size). The block-count population is
     derived from `ggml_block_geom.block_elts`, not typed from memory.
  4. ARITHMETIC, NOT TASTE. The struct members, the header's own `static_assert`, the shipped
     table's expression string and the stale expression string, all evaluated with ggml_block_geom's
     evaluator against the header's macros.
  5. THE D-17 Q4 VETO on a Q4_0 and an IQ4_NL width — the pinned file has no tensor of either type,
     so before this file the veto had only ever been seen firing on Q4_K/Q5_K/Q6_K fixtures — plus the
     fail-closed leg: with a stale row injected the policy must REFUSE to price the type at all.

CPU only, no device, no clock: no timing taken.

    python3 specification/evidence/q4_0_width_witness.py
Run log: `specification/evidence/q4-0-width-witness-after-blk-correction-2026-10-26.log`; the log of
the run that found the drift (and correctly printed `RESULT FAIL ... findings=1`) is kept beside it as
`q4-0-width-witness-2026-10-26.log`.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))

import gguf_width_table as GWT                                 # noqa: E402
import ggml_block_geom as GBG                                  # noqa: E402
import gguf_header_types as G                                  # noqa: E402
import ulp_policy as UP                                        # noqa: E402

PINNED = UP.PINNED_GGUF
WORK = os.path.join(HERE, "work")
SUBJECTS = ("Q4_0", "IQ4_NL")
# The two rows exactly as tools/gguf_width_table.py carried them before the 2026-10-26 correction
# (tools/PROVENANCE.md §gguf_width_table.py). They live HERE and nowhere else, so the agreement check
# can be shown refusing them: Q4_0 priced 32 elements at 34 B (8.5 bpw) from an expression naming a
# symbol the pinned header does not define, and IQ4_NL carried 144 B per 256 elements — block_q4_K's
# byte count — behind an expression that evaluates to 132, i.e. neither block_iq4_nl (18/32) nor
# block_iq4_xs (136/256), at a bpw (4.5) that a bpw-only comparison would have called agreement.
STALE_ROWS = {
    "Q4_0": (32, 34, "sizeof(ggml_half) + QK"),
    "IQ4_NL": (256, 144, "sizeof(ggml_half)*2 + QK_K/2"),
}
# The macro each struct's nibble array is sized by, for the arithmetic section.
QK_MACRO = {"Q4_0": "QK4_0", "IQ4_NL": "QK4_NL"}


class V(object):
    def __init__(self):
        self.n = 0
        self.f = 0

    def check(self, name, ok, n, detail=""):
        self.n += 1
        if not ok:
            self.f += 1
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if ok else 0, n, detail))
        return ok


def type_id(name):
    for tid, tn in G.TYPES.items():
        if tn == name:
            return tid
    raise AssertionError("gguf_header_types has no id for %s: an unmapped type is a failure, not a "
                         "0-byte row" % name)


def write_gguf(path, tensor_name, type_name, n_elts, row_bytes):
    """One GGUF, one tensor, no metadata: enough header for `gguf_width_table.parse_gguf` to read the
    row back, and the data region is exactly `row_bytes` long.

    The bytes written are NOT what adjudicates anything — the file's LENGTH is, because that is the
    quantity the identity under test compares against.
    """
    name = tensor_name.encode()
    body = b""
    body += struct.pack("<I", 3)                       # version
    body += struct.pack("<Q", 1)                       # n_tensors
    body += struct.pack("<Q", 0)                       # n_kv
    body += struct.pack("<Q", len(name)) + name        # tensor name
    body += struct.pack("<I", 1) + struct.pack("<Q", n_elts)
    body += struct.pack("<I", type_id(type_name))
    body += struct.pack("<Q", 0)                       # offset within the data region
    head = b"GGUF" + body
    align = 32
    data_start = (len(head) + align - 1) // align * align
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(b"\0" * (data_start - len(head)))
        fh.write(b"\xa5" * row_bytes)
    return data_start


def block_population(type_name):
    """How many whole blocks each synthetic tensor holds: 1, 2, 3, a prime, and every power of two up
    to one block-count-per-element, plus the block count that makes exactly one block per ELTS of the
    type itself. Derived from the geometry (`ggml_block_geom.block_elts`) rather than hand-listed."""
    elts = GBG.block_elts(type_name)
    return sorted({1, 2, 3, 7} | {1 << k for k in range(0, 8)} | {elts})


def diff_rulers(universe, present, table, geom, tag):
    """(drift, compared) for one pass of the two width rulers over `universe`.

    ONE implementation, called twice — on the shipped `BLK` (drift must be empty) and on `BLK` with a
    stale row injected at run time (drift must name that type). The injected arm is what makes the
    agreement check a witness instead of a hope, and `compared` is what stops the honest arm from
    agreeing by silence: a type neither ruler priced would otherwise produce no drift line at all.
    """
    drift, compared = [], 0
    for t in universe:
        g, row = geom.get(t), table.get(t)
        if g is None or row is None:
            print("WIDTHUNMAPPED tag=%s type=%s geom_derived=%d table_row=%d — a type one ruler "
                  "cannot price is not agreement" % (tag, t, g is not None, row is not None))
            drift.append(t)
            continue
        compared += 1
        bpw_table = 8.0 * row[1] / row[0]
        if abs(bpw_table - g["bpw"]) > 1e-9 or row[0] != g["elts"]:
            drift.append(t)
            print("WIDTHDRIFT tag=%s type=%s elts_geom=%d elts_table=%d bytes_geom=%d "
                  "bytes_table=%d bpw_geom=%.6f bpw_table=%.6f struct=%s header_line=%s "
                  "static_assert=%s table_expression=%r in_pinned_file=%d note=%s"
                  % (tag, t, g["elts"], row[0], g["bytes"], row[1], g["bpw"], bpw_table, g["struct"],
                     g["line"], g["claim"], row[2], 1 if t in present else 0,
                     ("LIVE: a tensor of this type is in the pinned file, so the file-size witness "
                      "would have caught it" if t in present else
                      "LATENT: no tensor of this type in the pinned file, so the file-size witness "
                      "in tools/gguf_width_table.py cannot see it — this check is the only witness")))
    return drift, compared


def census_one_ulp(r):
    """A 16-element f32 census with exactly one element moved by exactly one step."""
    base = [UP._image_of("f32", 0.5 + 0.01 * i) for i in range(16)]
    moved = list(base)
    moved[3] = base[3] + 1
    return r.census((i, a, b) for i, (a, b) in enumerate(zip(base, moved)))


def synth_legs(v, tname, geom):
    """Synthetic tensors of `tname`, priced three ways by the tool's OWN parsers. Every size prints a
    SYNTHETIC line so the population is auditable, and each size where the stale row's price happens
    to coincide with the struct's prints IDENTITYBLIND."""
    tid = type_id(tname)
    g = geom[tname]
    pop = block_population(tname)
    held_geom, held_table, broke_stale, blind = [], [], [], []
    for n_blocks in pop:
        n_elts = n_blocks * g["elts"]
        good_bytes = n_blocks * g["bytes"]
        path = os.path.join(WORK, "%s_probe_%d_%d.gguf" % (tname.lower(), n_blocks, g["bytes"]))
        data_start = write_gguf(path, "%s.probe.weight" % tname.lower(), tname, n_elts, good_bytes)
        size = os.path.getsize(path)
        _, declared, parsed_rows, _he, ds2 = GWT.parse_gguf(path)
        if declared != len(parsed_rows) or ds2 != data_start:
            v.check("synthetic file %s parses back" % os.path.basename(path), False, declared,
                    "declared=%d parsed=%d data_start=%d/%d" % (declared, len(parsed_rows), ds2,
                                                                data_start))
            continue
        totals = {}
        for tag, tbl in (("geom", {tname: (g["elts"], g["bytes"], "geom")}),
                         ("table", {tname: GWT.BLK[tname]}),
                         ("stale", {tname: STALE_ROWS[tname]})):
            _fam, total, unmapped = GWT.family_bytes(parsed_rows, tbl)
            if unmapped:
                raise AssertionError("unmapped type in a file this tool wrote: %s" % unmapped)
            totals[tag] = total
        ok_geom = data_start + totals["geom"] == size
        ok_table = data_start + totals["table"] == size
        # Whether the stale row prices the tensor differently at all. Where it happens to price it
        # the same, the file-size identity is BLIND to the wrong row — that is not a failed
        # injection, it is the finding: block_iq4_nl under block_q4_K's (256, 144) costs the same per
        # tensor whenever the block count is a multiple of 8, which is one reason the row survived.
        prices_differently = totals["stale"] != totals["geom"]
        ok_stale = data_start + totals["stale"] != size
        held_geom.append(ok_geom)
        held_table.append(ok_table)
        if prices_differently:
            broke_stale.append(ok_stale)
        else:
            blind.append(n_blocks)
        print("SYNTHETIC type=%s file=%s elts=%d blocks=%d bytes_written=%d identity_geom=%d "
              "identity_table=%d identity_stale=%d stale_prices_differently=%d off_by_stale=%d"
              % (tname, os.path.basename(path), n_elts, n_blocks, good_bytes, ok_geom, ok_table,
                 ok_stale, prices_differently, size - (data_start + totals["stale"])))
    for n_blocks in blind:
        print("IDENTITYBLIND type=%s blocks=%d — the stale row %r prices this tensor exactly as the "
              "struct does, so tools/gguf_width_table.py's file-size witness cannot see the row; the "
              "elts-AND-bytes comparison in two-width-rulers-agree-on-every-type is the only thing "
              "that can (a bpw-only comparison could not either: both are 4.5 bpw)"
              % (tname, n_blocks, STALE_ROWS[tname]))
    v.check("synthetic %s tensors: the derived geometry reproduces every file length" % tname,
            all(held_geom) and len(held_geom) == len(pop), len(held_geom),
            "population %s derived from block_elts(%s)=%d" % (pop, tname, g["elts"]))
    v.check("synthetic %s tensors: the CORRECTED BLK row reproduces the same identity (the "
            "regression catcher)" % tname,
            all(held_table) and len(held_table) == len(pop), len(held_table),
            "BLK[%s]=%r priced every synthetic tensor to the file's own length" % (tname,
                                                                                   GWT.BLK[tname]))
    v.check("synthetic %s tensors: wherever the injected stale row prices a tensor differently the "
            "identity BREAKS" % tname,
            all(broke_stale) and len(broke_stale) + len(blind) == len(pop) and len(broke_stale) > 0,
            len(broke_stale),
            "BLK[%s] would have been %r = %.4f bpw; it bites at %d of the %d sizes and is blind at "
            "%s (block counts where its per-tensor price coincides with the struct's) — a witness "
            "that bites at no size is not a witness, and one that is claimed to bite at every size "
            "when it does not is a false claim"
            % (tname, STALE_ROWS[tname], 8.0 * STALE_ROWS[tname][1] / STALE_ROWS[tname][0],
               len(broke_stale), len(pop), blind or "none"))


def arithmetic_legs(v, tname, geom, macros):
    """The arithmetic that adjudicated the row, shown rather than asserted, and re-checked against
    the shipped row: the header's static_assert, the member sum, and both expression strings."""
    g = geom[tname]
    qk = QK_MACRO[tname]
    try:
        qk_val = int(GBG._eval(qk, macros, {}))
    except Exception as e:
        v.check("%s: the block macro evaluates from the header" % tname, False, 0, str(e))
        return
    half = qk_val // 2
    member_sum = 2 + half                       # ggml_half d (2 B) + uint8_t qs[QK/2]
    try:
        tbl_val = int(GBG._eval(GWT.BLK[tname][2], macros, {}))
        tbl_note = "evaluates to %d" % tbl_val
    except Exception as e:
        tbl_val, tbl_note = None, "does NOT evaluate against the pinned header: %s" % e
    try:
        stale_val = GBG._eval(STALE_ROWS[tname][2], macros, {})
    except Exception as e:
        stale_val = "raises %s" % e.__class__.__name__
    print("ARITHMETIC type=%s struct=%s header_line=%s elts=%d bytes=%d bpw=%.6f "
          "static_assert=%r evaluates_to=%s"
          % (tname, g["struct"], g["line"], g["elts"], g["bytes"], g["bpw"], g["claim"],
             g["claim_value"]))
    print("ARITHMETIC members: ggml_half d = 2 B, uint8_t qs[%s/2] = %d/2 = %d B; total %d B per "
          "%d elts = %.4f bpw" % (qk, qk_val, half, member_sum, g["elts"], g["bpw"]))
    print("ARITHMETIC shipped_row=(%d,%d,%r) %s" % (GWT.BLK[tname][0], GWT.BLK[tname][1],
                                                    GWT.BLK[tname][2], tbl_note))
    print("ARITHMETIC stale_row=(%d,%d,%r) evaluates_to=%s (struct says %d B / %d elts)"
          % (STALE_ROWS[tname][0], STALE_ROWS[tname][1], STALE_ROWS[tname][2], stale_val,
             g["bytes"], g["elts"]))
    agrees = (g["claim_value"] == g["bytes"] == member_sum
              and GWT.BLK[tname][:2] == (g["elts"], g["bytes"])
              and tbl_val == g["bytes"])
    v.check("%s: the arithmetic, the header's static_assert and the shipped BLK row all agree" % tname,
            agrees, g["bytes"],
            "block is `ggml_half d` + `uint8_t qs[%s/2]` = 2 + %d = %d B per %d elts = %.4f bpw; the "
            "shipped expression string %r evaluates to %s and the header's own static_assert to %s "
            "(the expression is no longer hand-typed: it is the header's, and it evaluates)"
            % (qk, half, member_sum, g["elts"], g["bpw"], GWT.BLK[tname][2], tbl_val,
               g["claim_value"]))
    could_not = (stale_val != g["bytes"]) and (STALE_ROWS[tname][1] != g["bytes"])
    v.check("%s: the stale row could not have come from the header" % tname, could_not, 1,
            "the pre-correction expression %r gives %s and the pre-correction byte count was %d; "
            "the struct is %d B — no reading of the header produces the stale row, which is why this "
            "row survived a review that spot-checks bpw"
            % (STALE_ROWS[tname][2], stale_val, STALE_ROWS[tname][1], g["bytes"]))


def main():
    v = V()
    geom = GBG.geom()
    if not os.path.isdir(WORK):
        os.makedirs(WORK)

    # ---- 0. what the pinned file actually contains -------------------------------------------
    present = {}
    if os.path.exists(PINNED):
        for name, dims, dt in GWT.parse_gguf(PINNED)[2]:
            present.setdefault(G.TYPES.get(dt, "type_%d" % dt), name)
    # The census population is what the universe is built from, so an empty parse must be a finding
    # and not a small universe. Q4_K and Q8_0 are what a Q4_K_M artifact of this checkpoint contains
    # by construction; if the parse produced neither, the parse is broken, not the file.
    v.check("the pinned GGUF's type census was parsed",
            bool(present) and {"Q4_K", "Q8_0"} <= set(present), len(present),
            ("%d distinct tensor types in %s: %s" % (len(present), os.path.basename(PINNED),
                                                      ",".join(sorted(present))))
            if present else "no file at %s, or the parse yielded no tensor records — an empty census "
                            "would make every population below vacuous" % PINNED)
    universe = sorted(set(list(present) + list(GBG.REQUIRED) + list(GWT.BLK)))
    v.check("the type universe contains both subject types",
            all(t in universe for t in SUBJECTS) and len(universe) > 5, len(universe),
            "%d types: %d in the pinned file, %d required by the geometry, %d rows in BLK; "
            "in_pinned_file for the subjects: %s"
            % (len(universe), len(present), len(GBG.REQUIRED), len(GWT.BLK),
               ",".join("%s=%d" % (t, 1 if t in present else 0) for t in SUBJECTS)))

    # ---- 1. the named check: the two rulers, diffed over the whole universe -------------------
    drift, compared = diff_rulers(universe, present, GWT.BLK, geom, "shipped")
    v.check("two-width-rulers-agree-on-every-type", not drift, len(universe),
            "types where the rulers disagree: %s — this is the check that was RED on 2026-10-26 with "
            "drift=[Q4_0, IQ4_NL]; both rows were corrected the same day (tools/PROVENANCE.md "
            "§gguf_width_table.py), so a non-empty list here is a new finding, not the old one"
            % (drift or "none"))
    v.check("every type in the universe was actually compared (no agreement by silence)",
            compared == len(universe), compared,
            "compared=%d of %d — a type one ruler cannot price prints WIDTHUNMAPPED and counts as "
            "drift, so a short pass cannot print agreement" % (compared, len(universe)))

    # ---- 2. its non-vacuity: the two stale rows, injected at run time -------------------------
    fired, vacuous, unrestored = [], [], []
    for t in SUBJECTS:
        g, stale, saved = geom[t], STALE_ROWS[t], GWT.BLK[t]
        distinct = ((stale[0], stale[1]) != (g["elts"], g["bytes"]) and stale != saved)
        try:
            GWT.BLK[t] = stale
            d2, n2 = diff_rulers(universe, present, GWT.BLK, geom, "injected:%s" % t)
        finally:
            GWT.BLK[t] = saved
        ok = distinct and (t in d2) and d2 != drift and n2 == compared
        (fired if ok else vacuous).append(t)
        if GWT.BLK[t] != saved:
            unrestored.append(t)
        print("INJECTION type=%s stale=%r shipped=%r geom=(%d,%d) drift_shipped=%s drift_injected=%s "
              "restored=%d" % (t, stale, saved, g["elts"], g["bytes"], drift or "none", d2 or "none",
                               int(GWT.BLK[t] == saved)))
    v.check("the agreement check still FIRES when a stale row is injected (it is not vacuous)",
            len(fired) == len(SUBJECTS) and not vacuous and not unrestored, len(fired),
            "subjects that fired: %s; injections that were not a real perturbation: %s; rows left "
            "tampered: %s — the honest pass reads agreement ONLY because this arm reads disagreement"
            % (fired or "none", vacuous or "none", unrestored or "none"))

    # ---- 3. something to bite on: synthetic tensors, priced by the tool's own identity --------
    for tname in SUBJECTS:
        synth_legs(v, tname, geom)

    # ---- 4. the arithmetic, shown ------------------------------------------------------------
    src = open(GBG.HEADER, encoding="utf-8", errors="replace").read()
    macros = GBG._macros(src)
    for tname in SUBJECTS:
        arithmetic_legs(v, tname, geom, macros)

    # ---- 5. the D-17 Q4 veto, witnessed on both corrected widths, and the fail-closed leg -----
    r = UP.Ruler("f32")
    cen = census_one_ulp(r)
    v.check("the ±1-ULP fixture judged below actually judged something", cen.judged == 16
            and cen.diverging == 1 and cen.n_ge(2) == 0, cen.judged,
            "16 elements judged, exactly 1 at exactly 1 ULP, 0 at >=2 — if this moved, the veto arms "
            "below prove nothing about 1-ULP acceptance")
    # `WidthClass` cross-checks the two rulers by BITS PER WEIGHT only (`abs(bpw_independent - bpw) >
    # 1e-9`), not by element count. Both arms are witnessed here because they go opposite ways, and
    # `tools/ulp_policy.py` is byte-clean and stays that way: a bpw disagreement must make the policy
    # refuse the type, and a disagreement that is only in ELTS is invisible to it — which is precisely
    # why the agreement check above compares elts AND bytes, and why a bpw-only comparison would have
    # called the IQ4_NL row agreement. If ulp_policy ever starts comparing element counts, the second
    # arm goes red on purpose; re-read this leg then and say so in the commit message.
    arms = []
    for tname in SUBJECTS:
        saved, stale, g = GWT.BLK[tname], STALE_ROWS[tname], geom[tname]
        stale_bpw = 8.0 * stale[1] / stale[0]
        visible = abs(stale_bpw - g["bpw"]) > 1e-9
        try:
            GWT.BLK[tname] = stale
            try:
                UP.WidthClass.from_type(tname)
                raised, reason = False, "it priced the type anyway"
            except UP.WidthError as e:
                raised, reason = True, str(e)[:150]
        finally:
            GWT.BLK[tname] = saved
        print("FAILCLOSED type=%s stale=%r stale_bpw=%.4f geom_bpw=%.4f bpw_disagreement_visible_to_"
              "widthclass=%d widthclass_raised=%d said=%s"
              % (tname, stale, stale_bpw, g["bpw"], visible, raised, reason))
        arms.append((tname, visible, raised))
    seen = [t for t, vis, _ in arms if vis]
    blind = [t for t, vis, _ in arms if not vis]
    v.check("an injected bpw disagreement makes the policy REFUSE to price the type (fail closed)",
            bool(seen) and all(r for t, vis, r in arms if vis), len(seen),
            "types whose injected stale row moves bpw, and must be refused: %s" % (seen or "none"))
    v.check("an injected ELTS-only disagreement is INVISIBLE to the policy's bpw-only cross-check "
            "(witnessed limitation, not a pass)",
            bool(blind) and all(not r for t, vis, r in arms if not vis), len(blind),
            "types whose stale row keeps bpw and changes only the element count, and which "
            "WidthClass therefore still prices: %s — this is why two-width-rulers-agree-on-every-type "
            "compares elts AND bytes. RED HERE MEANS THE LIMITATION IS GONE: ulp_policy would have "
            "started comparing element counts, so update this leg and say so"
            % (blind or "none"))
    for tname in SUBJECTS:
        wc = UP.WidthClass.from_type(tname)
        d = UP.judge("s", ["s"], wc, census_one_ulp(r),
                     UP.Attribution(True, "fixture: one element crossed a rounding boundary"))
        v.check("the Q4 veto fires at exactly 1 ULP on a %s width, and names the veto" % tname,
                (not d.accepted) and d.vetoed_by_width, d.census.diverging,
                "%s | %s" % (wc.line(), ";".join(d.reasons)[:150]))
    q8 = UP.WidthClass.from_type("Q8_0")
    d8 = UP.judge("s", ["s"], q8, census_one_ulp(r),
                  UP.Attribution(True, "fixture: one element crossed a rounding boundary"))
    v.check("the same census at Q8_0 is ACCEPTED, so the refusals above are the width and nothing "
            "else", d8.accepted, d8.census.diverging, ";".join(d8.reasons)[:150])
    v.check("no injected row outlived its control (a tampered table must not be left behind)",
            all(GWT.BLK[t] != STALE_ROWS[t] for t in SUBJECTS), len(SUBJECTS),
            " | ".join("BLK[%s]=%r" % (t, GWT.BLK[t]) for t in SUBJECTS))

    for tname in SUBJECTS:
        g = geom[tname]
        print("VERDICT type=%s source_of_record=tools/ggml_block_geom.py (%s = %d B / %d elts = "
              "%.4f bpw) shipped_row=tools/gguf_width_table.py BLK['%s']=%r agreement=%s"
              % (tname, g["struct"], g["bytes"], g["elts"], g["bpw"], tname, GWT.BLK[tname],
                 "the two rulers now price this type identically"
                 if GWT.BLK[tname][:2] == (g["elts"], g["bytes"]) else "STILL DISAGREEING"))
    print("RESULT %s q4_0 width witness checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
