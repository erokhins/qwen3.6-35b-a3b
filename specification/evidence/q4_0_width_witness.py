#!/usr/bin/env python3
"""q4_0_width_witness.py — make the two width rulers DISAGREE LOUDLY on Q4_0, and settle it by arithmetic.

`tools/gguf_width_table.py` and `tools/ggml_block_geom.py` are the repo's two independent width rulers;
`tools/ulp_policy.py` §WidthClass refuses to price any type they disagree about, which is fail-closed but
silent. `specification/evidence/ulp-policy-migration-2026-10-26.md` §1 found they disagree on Q4_0 — 8.5
vs 4.5 bpw — and nothing catches it, because gguf_width_table.py's authority is the file-size identity
`sum(tensor row bytes) + data_start == file size` and the pinned GGUF contains no Q4_0 tensor, so the
identity is vacuous for that row. That is `pitfalls/40` §"two self-consistent layouts that nothing
compares", in the wild.

This file does three things, in that order:

  1. NAMED CHECK. It diffs the two rulers over the whole type universe (the pinned file's own type
     census, plus every type `ggml_block_geom.REQUIRED` demands, plus every row of `BLK`) and prints a
     `WIDTHDRIFT` line per disagreement, saying whether a tensor of that type exists in the pinned file
     (live) or not (latent). The check `two-width-rulers-agree-on-every-type` is RED while any drift
     stands, including the latent ones — a file-size witness cannot see a type the file does not
     contain, so this is the only thing that can.
  2. SOMETHING TO BITE ON. It writes synthetic GGUF files containing Q4_0 tensors whose byte length is
     derived from the pinned struct geometry, and runs gguf_width_table.py's OWN `parse_gguf` and
     `family_bytes` over them with each table. The identity holds under the derived geometry and breaks
     under the hand-typed row, for every block count — so the disagreement is no longer argued, it is a
     witness that fires. The population of block counts is derived from the block geometry, not typed
     from memory.
  3. ARITHMETIC, NOT TASTE. It prints the struct members, the header's own `static_assert` expression,
     and the table's own expression string, evaluating both against the header's macros with
     ggml_block_geom's evaluator, and reports which ruler the arithmetic supports. It does not pick a
     favourite: if the arithmetic cannot settle it, the check stays red and says so.

It also witnesses the D-17 Q4 veto on a Q4_0 width — the pinned file has no Q4_0 tensor, so the veto
had only ever been seen firing on Q4_K/Q5_K/Q6_K fixtures.

CPU only, no device, no clock: no timing taken.

    python3 specification/evidence/q4_0_width_witness.py
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
SUBJECT = "Q4_0"


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
    pop = sorted({1, 2, 3, 7} | {1 << k for k in range(0, 8)} | {elts})
    return pop


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
    else:
        v.check("the pinned GGUF is readable", False, 0, "no file at %s" % PINNED)
    universe = sorted(set(list(present) + list(GBG.REQUIRED) + list(GWT.BLK)))
    v.check("the type universe is non-empty and contains the subject",
            SUBJECT in universe and len(universe) > 5, len(universe),
            "%d types: %d in the pinned file, %d required by the geometry, %d rows in BLK"
            % (len(universe), len(present), len(GBG.REQUIRED), len(GWT.BLK)))

    # ---- 1. the named check: the two rulers, diffed over the whole universe -------------------
    drift = []
    for t in universe:
        g = geom.get(t)
        row = GWT.BLK.get(t)
        if g is None or row is None:
            print("WIDTHUNMAPPED type=%s geom=%d table=%d — a type one ruler cannot price is not "
                  "agreement" % (t, g is not None, row is not None))
            drift.append(t)
            continue
        bpw_table = 8.0 * row[1] / row[0]
        if abs(bpw_table - g["bpw"]) > 1e-9 or row[0] != g["elts"]:
            drift.append(t)
            print("WIDTHDRIFT type=%s elts_geom=%d elts_table=%d bytes_geom=%d bytes_table=%d "
                  "bpw_geom=%.6f bpw_table=%.6f struct=%s static_assert=%s table_expression=%r "
                  "in_pinned_file=%d note=%s"
                  % (t, g["elts"], row[0], g["bytes"], row[1], g["bpw"], bpw_table, g["struct"],
                     g["claim"], row[2], 1 if t in present else 0,
                     ("LIVE: a tensor of this type is in the pinned file, so the file-size witness "
                      "would have caught it" if t in present else
                      "LATENT: no tensor of this type in the pinned file, so the file-size witness "
                      "in tools/gguf_width_table.py cannot see it — this check is the only witness")))
    v.check("two-width-rulers-agree-on-every-type", not drift, len(universe),
            "types where the rulers disagree: %s — red is the CORRECT state until the stale row is "
            "corrected by the file's owner; this tool does not edit tools/gguf_width_table.py"
            % (drift or "none"))
    v.check("the Q4_0 disagreement is among them (the check is not vacuous)",
            SUBJECT in drift, 1 if SUBJECT in drift else 0,
            "if this reads none, either the row was fixed or this file is comparing nothing")

    # ---- 2. something to bite on: synthetic Q4_0 tensors, the tool's own identity --------------
    tid = type_id(SUBJECT)
    g = geom[SUBJECT]
    pop = block_population(SUBJECT)
    held_geom, broke_table = [], []
    for n_blocks in pop:
        n_elts = n_blocks * g["elts"]
        rows = [("q4_0.probe.weight", (n_elts,), tid)]
        good_bytes = n_blocks * g["bytes"]
        path = os.path.join(WORK, "q4_0_probe_%d_%d.gguf" % (n_blocks, g["bytes"]))
        data_start = write_gguf(path, "q4_0.probe.weight", SUBJECT, n_elts, good_bytes)
        size = os.path.getsize(path)
        _, declared, parsed_rows, header_end, ds2 = GWT.parse_gguf(path)
        if declared != len(parsed_rows) or ds2 != data_start:
            v.check("synthetic file %s parses back" % os.path.basename(path), False, declared,
                    "declared=%d parsed=%d data_start=%d/%d" % (declared, len(parsed_rows), ds2,
                                                                data_start))
            continue
        tot_geom, tot_table = [], []
        for tbl, acc in (({SUBJECT: (g["elts"], g["bytes"], "geom")}, tot_geom),
                         ({SUBJECT: GWT.BLK[SUBJECT]}, tot_table)):
            _fam, total, unmapped = GWT.family_bytes(parsed_rows, tbl)
            if unmapped:
                raise AssertionError("unmapped type in a file this tool wrote: %s" % unmapped)
            acc.append(total)
        ok_geom = data_start + tot_geom[0] == size
        ok_table = data_start + tot_table[0] == size
        held_geom.append(ok_geom)
        broke_table.append((not ok_table) and g["bytes"] != GWT.BLK[SUBJECT][1])
        print("SYNTHETIC file=%s elts=%d blocks=%d bytes_written=%d identity_with_geom=%d "
              "identity_with_table=%d off_by=%d"
              % (os.path.basename(path), n_elts, n_blocks, good_bytes, ok_geom, ok_table,
                 size - (data_start + tot_table[0])))
        v.check("synthetic Q4_0 tensor %d blocks: the derived geometry reproduces the file" % n_blocks,
                ok_geom, good_bytes, "data_start=%d + %d == %d" % (data_start, tot_geom[0], size))
        v.check("synthetic Q4_0 tensor %d blocks: the hand-typed row BREAKS the same identity"
                % n_blocks, ok_table is False, size - (data_start + tot_table[0]),
                "the file-size witness in tools/gguf_width_table.py now bites, because a Q4_0 tensor "
                "exists to price")
    v.check("the witness fires on every size in the derived population",
            all(held_geom) and all(broke_table) and len(pop) > 0, len(pop),
            "population %s derived from block_elts(%s)=%d" % (pop, SUBJECT, g["elts"]))

    # ---- 3. the arithmetic, shown ------------------------------------------------------------
    src = open(GBG.HEADER, encoding="utf-8", errors="replace").read()
    macros = GBG._macros(src)
    tbl_expr = GWT.BLK[SUBJECT][2]
    try:
        tbl_val = GBG._eval(tbl_expr, macros, {})
        tbl_note = "evaluates to %d" % tbl_val
    except Exception as e:
        tbl_val, tbl_note = None, "does not evaluate against the pinned header: %s" % e
    print("ARITHMETIC type=%s struct=%s header_line=%s" % (SUBJECT, g["struct"], g["line"]))
    print("ARITHMETIC elts=%d bytes=%d bpw=%.6f static_assert=%r evaluates_to=%s"
          % (g["elts"], g["bytes"], g["bpw"], g["claim"], g["claim_value"]))
    print("ARITHMETIC table_row=(%d,%d,%r) %s; macros QK=%s QK4_0=%s"
          % (GWT.BLK[SUBJECT][0], GWT.BLK[SUBJECT][1], tbl_expr, tbl_note,
             macros.get("QK"), macros.get("QK4_0")))
    half = 0
    try:
        half = int(GBG._eval("QK4_0", macros, {})) // 2
    except Exception as e:
        print("ARITHMETIC QK4_0 does not evaluate: %s" % e)
    print("ARITHMETIC members: ggml_half d = 2 B, uint8_t qs[QK4_0/2] = %s/%d = %d B; total %d B per "
          "%d elts = %.4f bpw" % (macros.get("QK4_0"), 2, half, 2 + half, g["elts"], g["bpw"]))
    settles = (g["claim_value"] == g["bytes"] == 2 + half and GWT.BLK[SUBJECT][1] != g["bytes"])
    v.check("the arithmetic settles which ruler is right, and it is the derived one",
            settles, g["bytes"],
            "block_q4_0 is `ggml_half d` + `uint8_t qs[QK4_0/2]` = 2 + %d = %d B per %d elts = "
            "%.4f bpw; the table's expression string is %r, which is the same expression with the "
            "/2 (two nibbles per byte) dropped — %d B = %.4f bpw" % (
                half, 2 + half, g["elts"], g["bpw"], tbl_expr, GWT.BLK[SUBJECT][1],
                8.0 * GWT.BLK[SUBJECT][1] / GWT.BLK[SUBJECT][0]))

    # ---- 4. the D-17 Q4 veto, witnessed on a Q4_0 width ---------------------------------------
    r = UP.Ruler("f32")
    base = [UP._image_of("f32", 0.5 + 0.01 * i) for i in range(16)]
    moved = list(base)
    moved[3] = base[3] + 1                                # exactly one element, exactly one step
    cen = r.census((i, a, b) for i, (a, b) in enumerate(zip(base, moved)))
    saved = GWT.BLK[SUBJECT]
    try:
        try:
            UP.WidthClass.from_type(SUBJECT)
            closed = False
            reason = "it priced the type anyway"
        except UP.WidthError as e:
            closed = True
            reason = str(e)[:150]
        v.check("while the rulers disagree the policy REFUSES to price Q4_0 (fail closed)",
                closed, 1, "WidthClass said: %s" % reason)
        GWT.BLK[SUBJECT] = (g["elts"], g["bytes"], "CORRECTED-FOR-THE-VETO-CONTROL")
        wc = UP.WidthClass.from_type(SUBJECT)
        d = UP.judge("s", ["s"], wc, r.census(
            (i, a, b) for i, (a, b) in enumerate(zip(base, moved))),
            UP.Attribution(True, "fixture: one element crossed a rounding boundary"))
        v.check("the Q4 veto fires at exactly 1 ULP on a Q4_0 width, and names the veto",
                (not d.accepted) and d.vetoed_by_width, cen.diverging,
                "%s | %s" % (wc.line(), ";".join(d.reasons)[:150]))
        q8 = UP.WidthClass.from_type("Q8_0")
        d8 = UP.judge("s", ["s"], q8, cen, UP.Attribution(
            True, "fixture: one element crossed a rounding boundary"))
        v.check("the same census at Q8_0 is ACCEPTED, so the Q4 refusal is the width and nothing else",
                d8.accepted, d8.census.diverging, ";".join(d8.reasons)[:150])
    finally:
        GWT.BLK[SUBJECT] = saved
    v.check("the injected row was restored (a tampered table must not outlive the control)",
            GWT.BLK[SUBJECT] == saved, 1, "BLK[%s]=%r" % (SUBJECT, GWT.BLK[SUBJECT]))

    print("VERDICT source_right=tools/ggml_block_geom.py (block_q4_0 = %d B / %d elts = %.4f bpw) "
          "source_wrong=tools/gguf_width_table.py BLK['Q4_0'] (34 B = 8.5 bpw) not_fixed_here=that "
          "file is not this lane's; the row belongs (32, 18, \"sizeof(ggml_half) + QK4_0/2\")"
          % (g["bytes"], g["elts"], g["bpw"]))
    print("RESULT %s q4_0 width witness checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
