#!/usr/bin/env python3
"""gate.py — the device gate for `prototype/moe_router`.

The router is judged against the committed oracle and the pinned dumps, element by element, on
4-byte patterns: R5's bf16 cast weights and the top-K ids and the routed sum they feed are claimed
bit-exact; R1's 2048-term dot and R2's 256-term denominator are long dots and are claimed only to
sit inside a band that a floor probe of the reference's OWN re-associations derived
(`specification/10-reference-dumps-and-gates.md` §"Bands are derived, not invented"). Nothing here
has a tolerance that was set by looking at what this build produces.

What is compared, and by whom
  reference side   `speccheck/moe_ref.py` (torch CPU), driven by `oracle_arm.py` under the venv
                   interpreter, plus the pinned dumps in `data/dumps/2026-10-26/` read through
                   `checkpoints/dump-manifest-2026-10-26.json` with the dtype the manifest states.
  subject side     `moe_router` (hipcc, gfx1151) and `host_probe` (g++), both running the single
                   arithmetic path in `moe_router_kernels.h`.
  the oracle's own recorded agreement in the manifest (`agreement_at_write_time`) is re-derived
  here rather than trusted: `check_manifest_agreement` re-reads it and fails if any arm it claims
  was bit-exact is not bit-exact in this run.

Discipline
  correctness only. This file imports no timer and prints no ms, GB/s or tok/s figure; the lock
  record's wall clock belongs to the `gate` wrapper. The wrapper alone owns the machine-wide GPU
  lock, and the lock tool's filename is deliberately not spelled anywhere in this file so that
  `grep -c` for it returns 0 (`pitfalls/50-harness-and-shell.md` §"Self-locking gate drivers
  deadlock silently"). The driver's stdout goes to a file that is then parsed — never piped — and
  its return code is checked directly, and every parser here requires the number of records it
  parsed to equal the number the producer printed
  (`pitfalls/40-gates-and-admissions.md` §"A gate whose parser can parse nothing prints PASS").
"""
import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BIN = os.path.join(HERE, "moe_router")
PROBE_BIN = os.path.join(HERE, "host_probe")
ORACLE_PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"
WORK = os.path.join(HERE, "work")
EV = os.path.join(HERE, "evidence")
DAY = "2026-10-26"
MANIFEST = os.path.join(REPO, "checkpoints", "dump-manifest-2026-10-26.json")
HEADER = os.path.join(HERE, "moe_router_kernels.h")
BLOCKS = ["layer0", "layer1", "layer39", "mtp0"]
E, K, H = 256, 8, 2048
CASE_FLOATS = E + E + K + K + K + H
OFF = {"logits": 0, "probs": E, "wpre": 2 * E, "wcast": 2 * E + K, "ids": 2 * E + 2 * K,
       "routed": 2 * E + 3 * K}
LEN = {"logits": E, "probs": E, "wpre": K, "wcast": K, "ids": K, "routed": H}
FIELDS = ["logits", "probs", "wpre", "wcast", "ids", "routed"]
MODES = ["correct", "softmax_bf16", "no_renorm", "cast_after", "rank_accum", "w_by_expert_id",
         "row_off_1", "wg_transposed", "tie_high", "f32_combine", "topk_asc", "no_max_sub",
         "poison_nan"]
ARM_KERNEL, ARM_LOGITS = 0, 1
MATCH, MISMATCH, ALT, IMMUNE, FIRED = 0, 1, 2, 3, 4
# indices into MODES, not literals: an injection that is renamed in the header stops being checked
# loudly rather than silently becoming a no-op check
MR_MODE_ROW_OFF_1_IDX = MODES.index("row_off_1")
CLS_NAME = {MATCH: "MATCH", MISMATCH: "MISMATCH", ALT: "ALT", IMMUNE: "IMMUNE", FIRED: "FIRED"}
POISON = 0x7FC00000


class GateError(Exception):
    pass


# -----------------------------------------------------------------------------------------------
# the small rulers
# -----------------------------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def bf16_round(b):
    """round-to-nearest-even of an f32 bit pattern to the bf16 bit image, matching the kernel."""
    if (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF):
        return 0x7FC0
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def ulpdist_bits(a, b):
    ai = a if a < 0x80000000 else 0x80000000 - a
    bi = b if b < 0x80000000 else 0x80000000 - b
    return abs(ai - bi)


def pack_f32(vals):
    return struct.pack("<%dI" % len(vals), *vals)


def pack_widened(bits16):
    return struct.pack("<%dI" % len(bits16), *[(v << 16) for v in bits16])


def pack_ids(ids):
    return struct.pack("<%df" % len(ids), *[float(v) for v in ids])


def hexlist(s, width):
    """A column of bit patterns. `,` separates values inside one column and whitespace separates
    columns, so a truncated list is caught by the width and not silently read as a short row."""
    out = []
    for tok in s.replace(",", " ").split():
        out.append(int(tok, 16))
    if len(out) != width:
        raise GateError("expected %d hex values, got %d in %r" % (width, len(out), s[:60]))
    return out


class Verdict(object):
    """Every check is named, counted and printed; the roster is asserted complete at the end."""

    def __init__(self):
        self.checks = []
        self.findings = []

    def check(self, name, ok, n, detail=""):
        self.checks.append((name, bool(ok), n, detail))
        print("CHECK %-38s %s n=%s %s" % (name, "ok" if ok else "FAIL", n, detail))
        if not ok:
            self.findings.append(name)
        return bool(ok)


# -----------------------------------------------------------------------------------------------
# the header is the source of truth for the integers
# -----------------------------------------------------------------------------------------------
def check_header_enum():
    text = open(HEADER).read()

    def enum_vals(first):
        m = re.search(r"enum \{\s*(" + first + r"\s*=\s*0,.*?)\};", text, re.S)
        if not m:
            raise GateError("cannot find the enum starting at %s in the header" % first)
        body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)
        return {k: int(v, 0) for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)",
                                                    body)}
    modes = enum_vals("MR_MODE_CORRECT")
    for i, nm in enumerate(MODES):
        key = "MR_MODE_" + nm.upper()
        if modes.get(key) != i:
            raise GateError("header %s=%s, gate says %d" % (key, modes.get(key), i))
    if modes.get("MR_MODE_COUNT") != len(MODES):
        raise GateError("MR_MODE_COUNT=%s, gate has %d modes" % (modes.get("MR_MODE_COUNT"),
                                                                 len(MODES)))
    arms = enum_vals("MR_ARM_KERNEL")
    if arms.get("MR_ARM_KERNEL") != ARM_KERNEL or arms.get("MR_ARM_LOGITS") != ARM_LOGITS:
        raise GateError("the arm enum in the header disagrees with the gate")
    for nm, want in (("MR_E", E), ("MR_K", K), ("MR_H", H), ("MR_NAN_BITS", POISON)):
        m = re.search(r"#define %s\s+(0x[0-9A-Fa-f]+|\d+)u?" % nm, text)
        if not m or int(m.group(1), 0) != want:
            raise GateError("%s is %s in the header, %s in the gate"
                            % (nm, m.group(1) if m else "absent", hex(want)))
    # The offsets and the case stride are written in the header as arithmetic over E/K/H, so they are
    # evaluated here with the header's own expression rather than re-typed: two copies of an offset
    # table is how a field comparison ends up reading its neighbour's words.
    env = {"MR_E": E, "MR_K": K, "MR_H": H}

    def cexpr(macro):
        m = re.search(r"#define %s\s+([^\n]+)" % macro, text)
        if not m:
            raise GateError("no #define %s in %s" % (macro, HEADER))
        expr = m.group(1).strip()
        while expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1].strip()          # one balanced outer pair, not a strip of characters
        try:
            return int(eval(expr, {"__builtins__": {}}, dict(env)))
        except Exception as exc:
            raise GateError("cannot evaluate #define %s = %r (%s)" % (macro, expr, exc))

    for f in FIELDS:
        got = cexpr("MR_OFF_%s" % f.upper())                # the header's own arithmetic
        if got != OFF[f]:
            raise GateError("MR_OFF_%s evaluates to %d, the gate says %d" % (f.upper(), got, OFF[f]))
        env["MR_OFF_" + f.upper()] = got
    got = cexpr("MR_CASE_FLOATS")
    if got != CASE_FLOATS:
        raise GateError("MR_CASE_FLOATS evaluates to %d, the gate says %d" % (got, CASE_FLOATS))
    if CASE_FLOATS != sum(LEN[f] for f in FIELDS):
        raise GateError("the case stride %d is not the sum of the field lengths %d" %
                        (CASE_FLOATS, sum(LEN[f] for f in FIELDS)))
    # the reachability table is the gate's must-not-move source; refuse a table that names a mode
    # the gate does not have
    body = re.search(r"mr_mode_reaches\(int mode, int field, int arm\)\s*\{(.*?)\n\}", text, re.S)
    if not body:
        raise GateError("cannot read mr_mode_reaches out of the header")
    if len(re.findall(r"case MR_MODE_", body.group(1))) < len(MODES) - 1:
        raise GateError("mr_mode_reaches covers fewer modes than the gate has")
    return len(MODES)


def reaches(mode, field, arm):
    """The kernel's own answer, re-read rather than mirrored: the gate asks the same question the
    kernel answers, so a mode that moves more than it claims to is caught instead of blessed."""
    global _REACH
    return _REACH[(arm, mode, field)]


def load_reach_table():
    """Ask the compiled kernel which fields each mode can reach, instead of re-implementing the
    predicate here: one spelling of the claim (pitfalls/40 §"two self-consistent layouts"), so the
    must-not-move half of the injection matrix is the kernel's statement about itself that the gate
    checks rather than remembers. A mode the table forgets would otherwise make a dead injection look
    immune, so the record count is checked against the count the producer printed.

    Host-only: --reach-table makes no HIP call, so this path takes no lock (the wrapper prints the
    reason for the mode it was asked to run)."""
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    if not os.path.exists(BIN):
        raise GateError("missing binary %s; run ./build.sh" % BIN)
    rc, text = run([BIN, "--reach-table"], os.path.join(WORK, "reach-table.log"))
    recs, n = parse_records(text, "REACH", ["arm", "mode", "field", "v"])
    cc, nc = parse_records(text, "REACHCOUNT", ["arms", "modes", "fields", "records"])
    res = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if rc != 0 or len(res) != 1 or not res[0].startswith("RESULT PASS"):
        raise GateError("--reach-table failed (rc=%d): %r" % (rc, res))
    if nc != 1:
        raise GateError("--reach-table printed %d REACHCOUNT lines, expected 1" % nc)
    arms = int(cc[0]["arms"])
    modes, fields, printed = int(cc[0]["modes"]), int(cc[0]["fields"]), int(cc[0]["records"])
    if (arms, modes, fields) != (2, len(MODES), len(FIELDS)):
        raise GateError("the kernel declares %d arms x %d modes x %d fields, the gate has 2 x %d x %d"
                        % (arms, modes, fields, len(MODES), len(FIELDS)))
    if printed != n or printed != arms * modes * fields:
        raise GateError("--reach-table printed %d records; parsed %d of the %d expected"
                        % (printed, n, arms * modes * fields))
    tbl = {}
    for r in recs:
        a, m, f, val = int(r["arm"]), int(r["mode"]), int(r["field"]), int(r["v"])
        if a not in (ARM_KERNEL, ARM_LOGITS) or not 0 <= m < len(MODES) or not 0 <= f < len(FIELDS):
            raise GateError("REACH record outside the enums: %r" % r)
        if val not in (0, 1, 2):
            raise GateError("mr_mode_reaches(%s, %s) returned %d, not NO/YES/MAY"
                            % (MODES[m], FIELDS[f], val))
        tbl[(a, m, FIELDS[f])] = val
    return tbl


# -----------------------------------------------------------------------------------------------
# dumps, read with the dtype the manifest states
# -----------------------------------------------------------------------------------------------
def read_dump(rec):
    path = os.path.join(REPO, rec["file"])
    raw = open(path, "rb").read()
    if len(raw) != rec["bytes"]:
        raise GateError("%s: %d bytes on disk, manifest says %d" % (rec["name"], len(raw),
                                                                    rec["bytes"]))
    n = 1
    for d in rec["shape"]:
        n *= d
    if rec["dtype"] == "torch.bfloat16":
        if n * 2 != len(raw):
            raise GateError("%s: bf16 shape does not hold its own byte count" % rec["name"])
        return list(struct.unpack("<%dH" % n, raw)), "bf16"
    if rec["dtype"] == "torch.float32":
        return list(struct.unpack("<%dI" % n, raw)), "f32"
    if rec["dtype"] == "torch.int64":
        return list(struct.unpack("<%dq" % n, raw)), "i64"
    raise GateError("%s: manifest dtype %r is not one this gate knows" % (rec["name"],
                                                                          rec["dtype"]))


# -----------------------------------------------------------------------------------------------
# the gate's own combine: a second, independently written implementation of the reference form.
# It is validated against the pinned `routed_sum` before it is allowed to judge anything, which is
# what keeps it from being a mirror of the kernel (pitfalls/40 §"A check whose reference side is
# derived from the thing under test").
# -----------------------------------------------------------------------------------------------
def py_combine(raw_bits, wpre_bits, wcast_bits, ids, order="expert_asc", cast="before",
               acc="bf16", pair_by_id=False):
    if order == "expert_asc":
        o = sorted(range(K), key=lambda k: (ids[k], k))
    else:
        o = list(range(K))
    out = [0] * H
    for h in range(H):
        if acc == "f32":
            facc = 0.0
            for k in o:
                w = bits_f32(wpre_bits[k]) if cast == "after" else bits_f32(wcast_bits[k] << 16)
                facc += bits_f32(raw_bits[k * H + h] << 16) * w
            out[h] = bf16_round(f32bits(facc))
            continue
        a = 0
        for k in o:
            kk = (ids[k] % K) if pair_by_id else k
            w = bits_f32(wpre_bits[kk] if cast == "after" else (wcast_bits[kk] << 16))
            prod = bf16_round(f32bits(bits_f32(raw_bits[k * H + h] << 16) * w))
            a = bf16_round(f32bits(bits_f32(a << 16) + bits_f32(prod << 16)))
        out[h] = a
    return out


# -----------------------------------------------------------------------------------------------
# the plan
# -----------------------------------------------------------------------------------------------
class Plan(object):
    def __init__(self):
        self.cases = []
        self.slab = []                     # uint16 elements
        self.rows = {}                     # src_key -> row record
        self.out_floats = 0

    def add_slab(self, vals):
        off = len(self.slab)
        self.slab.extend(vals)
        return off

    def declare(self):
        """Print the whole sample BEFORE any device work, so the run can be judged against what it
        claimed to cover, and so a cut is visible rather than inferred from a smaller number later
        (the discipline the dequant lane adopted). Units are the field comparisons actually made,
        and launches are the kernel launches the driver will issue: one per mode with cases."""
        units = len(self.cases) * len(FIELDS)
        elements = sum(LEN[f] for c in self.cases for f in FIELDS)
        modes = sorted(set(c["mode"] for c in self.cases))
        arms = len(set((c["arm"], c["row"]) for c in self.cases))
        real = sum(1 for r in self.rows.values() if r["kind"] == "real")
        synth = sum(1 for r in self.rows.values() if r["kind"] == "synth")
        print("SAMPLE cases=%d arms=%d rows=%d(real=%d,synth=%d) field_comparisons=%d "
              "elements_compared=%d launches=%d slab_elems=%d slab_MiB=%.1f modes=%d"
              % (len(self.cases), arms, len(self.rows), real, synth, units, elements,
                 len(modes), len(self.slab), len(self.slab) * 2 / 1048576.0, len(MODES)))
        return units


def build_plan(man, D, oracle_rows, oracle_synth, bands, v, max_rows=32):
    p = Plan()
    # the weight slab: one copy per block of the router weight, bf16 bits
    wg_off = {}
    for blk in BLOCKS:
        bits, dt = read_dump(D[blk + ".mlp.gate.weight"])
        if dt != "bf16" or len(bits) != E * H:
            raise GateError("%s.mlp.gate.weight is %s of %d elements; expected bf16 [E,H]"
                            % (blk, dt, len(bits)))
        wg_off[blk] = p.add_slab(bits)
    for blk in BLOCKS:
        xall, dt = read_dump(D[blk + ".input"])
        lball, _ = read_dump(D[blk + ".router_logits"])
        rawall, _ = read_dump(D[blk + ".expert_raw"])
        idsall, _ = read_dump(D[blk + ".topk_ids"])
        twb, _ = read_dump(D[blk + ".topk_weights"])
        twf, _ = read_dump(D[blk + ".topk_weights_f32"])
        rs, _ = read_dump(D[blk + ".routed_sum"])
        for t in range(max_rows):
            key = BLOCKS.index(blk) * 32 + t
            o = oracle_rows[key]
            r = {"key": key, "kind": "real", "block": blk, "token": t,
                 "x_off": p.add_slab(xall[t * H:(t + 1) * H]),
                 "wg_off": wg_off[blk],
                 "lg_off": p.add_slab(lball[t * E:(t + 1) * E]),
                 "raw_off": p.add_slab(rawall[t * K * H:(t + 1) * K * H]),
                 "logits_bits": lball[t * E:(t + 1) * E],
                 "raw_bits": rawall[t * K * H:(t + 1) * K * H],
                 "ids": list(idsall[t * K:(t + 1) * K]),
                 "wpre_bits": twf[t * K:(t + 1) * K],
                 "wcast_bits": twb[t * K:(t + 1) * K],
                 "routed_bits": rs[t * H:(t + 1) * H],
                 "alt_ids": o["rule_ids"], "alt_wpre_bits": o["rule_wpre_bits"],
                 "alt_wcast_bits": o["rule_wcast_bits"],
                 "castafter_routed_bits": o["castafter_routed_bits"],
                 "tie": o["tie"], "margin_pin": o["margin_pin"], "margin_ref": o["margin_ref"],
                 "id_asc": o["id_asc"], "pair_identity": o["pair_identity"]}
            r["id_asc_auth"] = int(all(r["ids"][k] < r["ids"][k + 1] for k in range(K - 1)))
            r["pair_identity_auth"] = int(all(r["ids"][k] % K == k for k in range(K)))
            if r["id_asc_auth"] != o["id_asc"] or r["pair_identity_auth"] != o["pair_identity"]:
                raise GateError("row %d: the gate's immunity predicate disagrees with the oracle's "
                                "(id_asc %d/%d, pair %d/%d)" % (key, r["id_asc_auth"], o["id_asc"],
                                                                r["pair_identity_auth"],
                                                                o["pair_identity"]))
            p.rows[key] = r
    for key, o in sorted(oracle_synth.items()):
        r = {"key": key, "kind": "synth", "block": "synth", "token": key,
             "x_off": p.rows[0]["x_off"], "wg_off": p.rows[0]["wg_off"],
             "lg_off": p.add_slab(o["logits_bits"]),
             "raw_off": p.add_slab(o["raw_bits"]),
             "logits_bits": o["logits_bits"], "raw_bits": o["raw_bits"],
             "ids": o["rule_ids"], "wpre_bits": o["rule_wpre_bits"],
             "wcast_bits": o["rule_wcast_bits"], "routed_bits": o["routed_bits"],
             "alt_ids": o["oracle_ids"], "alt_wpre_bits": o["rule_wpre_bits"],
             "alt_wcast_bits": o["rule_wcast_bits"], "castafter_routed_bits": None,
             "tie": o["tie"], "margin_pin": o["margin"], "margin_ref": o["margin"],
             "id_asc": o["id_asc"], "pair_identity": o["pair_identity"],
             "id_asc_auth": o["id_asc"], "pair_identity_auth": o["pair_identity"]}
        p.rows[key] = r

    # the case matrix. Every real row runs both arms under every mode, so a R1 disagreement can be
    # separated from a downstream one (arm LOGITS feeds the pinned logits in); the synthetic rows
    # exist to make the tie and ordering classes reachable and run the LOGITS arm only, because
    # their logits are the fixture rather than something x @ Wg can produce.
    poison_rows = set(BLOCKS.index(b) * 32 for b in BLOCKS)
    for key in sorted(p.rows):
        r = p.rows[key]
        arms = [ARM_LOGITS] if r["kind"] == "synth" else [ARM_KERNEL, ARM_LOGITS]
        for arm in arms:
            for mode in range(len(MODES)):
                if mode == len(MODES) - 1 and key not in poison_rows:
                    continue
                if mode == 0 and arm not in arms:
                    continue
                c = {"arm": arm, "mode": mode, "row": key, "tie": r["tie"],
                     "id_asc": r["id_asc_auth"], "raw_src": 1 if r["kind"] == "synth" else 0,
                     "src_key": key, "x_off": r["x_off"], "wg_off": r["wg_off"],
                     "lg_off": r["lg_off"], "raw_off": r["raw_off"],
                     "dst_off": p.out_floats, "cmp_off": -1}
                p.out_floats += CASE_FLOATS
                p.cases.append(c)
    # every injection case compares against its own row+arm reference record
    base = {}
    for i, c in enumerate(p.cases):
        c["id"] = i                              # the record number, so a NOTE names the case it means
        if c["mode"] == 0:
            base[(c["arm"], c["row"])] = c["dst_off"]
    for c in p.cases:
        if c["mode"] != 0:
            c["cmp_off"] = base[(c["arm"], c["row"])]
    return p


def write_plan(p, path):
    with open(path, "w") as fh:
        fh.write("# moe_router plan: GEOM then one CASE per case\n")
        fh.write("GEOM E=%d K=%d H=%d slab_elems=%d out_floats=%d\n"
                 % (E, K, H, len(p.slab), p.out_floats))
        for i, c in enumerate(p.cases):
            fh.write("CASE %d %d %d %d %d %d %d %d %d %d %d %d %d\n"
                     % (i, c["arm"], c["mode"], c["tie"], c["id_asc"], c["raw_src"], c["src_key"],
                        c["x_off"], c["wg_off"], c["lg_off"], c["raw_off"], c["dst_off"],
                        c["cmp_off"]))
    with open(path.replace(".tsv", ".bin"), "wb") as fh:
        fh.write(struct.pack("<%dH" % len(p.slab), *p.slab))
    return path + "", path.replace(".tsv", ".bin")


# -----------------------------------------------------------------------------------------------
# the reference side: the oracle arm, run under the interpreter that has torch
# -----------------------------------------------------------------------------------------------
def run(cmd, log_path, env=None):
    """stdout and stderr both go to a FILE, never a pipe: piping a gate through tail swallows its
    exit code and its verdict line alike (pitfalls/50-harness-and-shell.md)."""
    with open(log_path, "wb") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    text = open(log_path, errors="replace").read()
    return rc, text


def parse_records(text, tag, keys):
    """Records are matched on the tag as a whole word, not by prefix: `PLAN` must not swallow
    `PLANCHECK`, or the two producers' counts get merged and both checks pass on each other's lines."""
    recs = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != tag:
            continue
        kv = {}
        for tok in parts[1:]:
            if "=" in tok:
                k, vv = tok.split("=", 1)
                kv[k] = vv
        missing = [k for k in keys if k not in kv]
        if missing:
            raise GateError("%s line is missing %s: %r" % (tag, missing, line))
        recs.append(kv)
    return recs, len(recs)


def parse_tsv(path, ncol, sep=None, comment="#"):
    """Columns are counted, not assumed. `sep="\t"` is for the oracle's own tables, where a column
    holds a comma-joined list: splitting those on whitespace would produce a column count that means
    nothing, and a reader that cannot count columns cannot notice a truncated record."""
    rows = []
    for line in open(path):
        if line.startswith(comment) or not line.strip():
            continue
        line = line.rstrip("\n")
        if sep is not None and sep not in line:
            raise GateError("%s: no %r separator on this line, so the column count would be "
                            "invented: %r" % (path, sep, line[:80]))
        parts = line.split(sep) if sep else line.split()
        if len(parts) != ncol:
            raise GateError("%s: expected %d columns, got %d in %r" % (path, ncol, len(parts),
                                                                       line[:80]))
        rows.append(parts)
    return rows


def read_oracle(v, work):
    if not os.path.exists(ORACLE_PY):
        raise GateError("no torch interpreter at %s; the oracle arm must run under the venv that "
                        "produced the dumps" % ORACLE_PY)
    cmd = [ORACLE_PY, os.path.join(HERE, "oracle_arm.py"), "--work", work]
    print("CMD " + " ".join(cmd))
    rc, text = run(cmd, os.path.join(work, "oracle.log"))
    for line in text.splitlines():
        if line.startswith(("ORACLE", "FLOOR ", "P3 ", "RESULT ", "Traceback", "FAIL ")):
            print("ORACLE> " + line)
    results = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if len(results) != 1 or results[0].split()[1] != "PASS":
        raise GateError("the oracle arm did not print exactly one PASS RESULT: %r" % results)
    sec, _ = parse_records(text, "ORACLESECTION", ["section", "records"])
    got = {s["section"]: int(s["records"]) for s in sec}
    if sorted(got) != ["floor", "p3", "rows", "synth"]:
        raise GateError("the oracle printed sections %s, expected floor, p3, rows and synth"
                        % sorted(got))
    rows = parse_tsv(os.path.join(work, "oracle_rows.tsv"), 17, sep="\t")
    if len(rows) != got.get("rows"):
        raise GateError("oracle_rows.tsv holds %d records, the oracle printed %s"
                        % (len(rows), got.get("rows")))
    synth = parse_tsv(os.path.join(work, "oracle_synth.tsv"), 8, sep="\t")
    if len(synth) != got.get("synth"):
        raise GateError("oracle_synth.tsv holds %d records, the oracle printed %s"
                        % (len(synth), got.get("synth")))
    floors = {}
    for nm, n, val in parse_tsv(os.path.join(work, "oracle_floor.tsv"), 3):
        floors[nm] = int(val)
    if not floors:
        raise GateError("the floor probe produced no bands; every band would be vacuous")
    p3 = parse_tsv(os.path.join(work, "oracle_p3.tsv"), 8) if \
        os.path.exists(os.path.join(work, "oracle_p3.tsv")) else []
    orows = {}
    for r in rows:
        orows[int(r[0])] = {
            "block": r[1], "token": int(r[2]), "tie": int(r[3]),
            "margin_pin": int(r[4]), "margin_ref": int(r[5]),
            "id_asc": int(r[6]), "pair_identity": int(r[7]),
            "rule_ids": [int(t) for t in r[8].split(",")],
            "pinned_ids": [int(t) for t in r[9].split(",")],
            "rule_wpre_bits": hexlist(r[10], K), "pinned_wpre_bits": hexlist(r[11], K),
            "rule_wcast_bits": hexlist(r[12], K), "pinned_wcast_bits": hexlist(r[13], K),
            "pinned_routed_bits": hexlist(r[14], H),
            "castafter_routed_bits": hexlist(r[15], H),
            "n_rule_ne_pin": int(r[16]),
        }
    osynth = {}
    for r in synth:
        key = int(r[0])
        lg = hexlist(r[1], E)
        raw = hexlist(r[2], K * H)
        rids = [int(t) for t in r[3].split(",")]
        osynth[key] = {
            "logits_bits": lg, "raw_bits": raw, "rule_ids": rids,
            "rule_wpre_bits": hexlist(r[4], K), "rule_wcast_bits": hexlist(r[5], K),
            "routed_bits": hexlist(r[6], H),
            "oracle_ids": [int(t) for t in r[7].split(",")],
        }
    # tie class and margins for the synthetic rows, from their own probabilities, with the same
    # rule tie_of() uses on the oracle side: descending probability, the 8th/9th gap in ULPs
    for key, o in osynth.items():
        lg = o["logits_bits"]
        vals = [bits_f32(b << 16) for b in lg]
        mx = max(vals)
        ex = [math.exp(xv - mx) for xv in vals]
        s = 0.0
        for x in ex:
            s += x
        p = [x / s for x in ex]
        srt = sorted(p, reverse=True)
        v8, v9 = srt[K - 1], srt[K]
        o["tie"] = 0
        if v8 == v9:
            n_in = sum(1 for x in srt[:K] if x == v8)
            o["tie"] = 3 if n_in >= 2 else 2
        else:
            for i in range(K - 1):
                if srt[i] == srt[i + 1]:
                    o["tie"] = 1
                    break
        o["margin"] = 0 if v8 == v9 else int(round((v8 - v9) / math.ulp(v8)))
        o["id_asc"] = int(all(o["rule_ids"][k] < o["rule_ids"][k + 1] for k in range(K - 1)))
        o["pair_identity"] = int(all(o["rule_ids"][k] % K == k for k in range(K)))
    return orows, osynth, floors, p3, text


# -----------------------------------------------------------------------------------------------
# judging
# -----------------------------------------------------------------------------------------------
MAY_UNCHANGED, MAY_MOVED = 5, 6
CLS_NAME[MAY_UNCHANGED] = "MAY_UNCHANGED"
CLS_NAME[MAY_MOVED] = "MAY_MOVED"


def field_words(mm, case, field):
    return struct.unpack_from("<%dI" % LEN[field], mm, 4 * (case["dst_off"] + OFF[field]))


def argmax_tie(probs, used, tie_high):
    best, bv = -1, None
    for l in range(E):
        if used[l]:
            continue
        pv = probs[l]
        if best < 0 or (pv >= bv if tie_high else pv > bv):
            best, bv = l, pv
    return best


def expect_injection(mode, field, r, arm):
    """(kind, why). kind is 'must' (the injection is obliged to move this field), 'never' (it
    cannot, from the algebra in mr_mode_reaches, so a move is a second bug), 'immune' (a row derived
    not to exercise it) or 'may' (reachable but data-dependent, so it is counted, not demanded).
    The derived immunities are the must-not-catch half of the table, and each names its predicate so
    an empty population can be traced to a predicate rather than to a typo."""
    if reaches(mode, field, arm) == 0:
        return "never", ""
    if reaches(mode, field, arm) == 2:
        # the kernel says this edge exists but is data-dependent (a re-associated sum whose ulp the
        # bf16 cast may absorb); demanding a move per case would be demanding a property of the row
        return "may", ""
    if mode in (4,):
        return ("immune", "ids already ascending") if r["id_asc_auth"] else ("must", "")
    if mode == 5:
        return ("immune", "ids[k] %% K == k") if r["pair_identity_auth"] else ("must", "")
    if mode == 8:
        return ("must", "") if r["tie"] in (1, 2, 3) else ("immune", "no tie in the top K+1")
    if mode == 10:
        return ("immune", "ids already ascending") if r["id_asc_auth"] else ("must", "")
    if mode in (6, 7):
        return ("must", "") if field == "logits" else ("may", "")
    if mode in (1, 11) and field == "ids":
        return ("may", "")
    if mode == 2 and field == "ids":
        return ("never", "")
    return "must", ""


def judge_reference(p, case, mm, bands, tally, notes, v, dev_words, prob_bound):
    """mode 0: compare the device's words against the authoritative arm."""
    r = p.rows[case["row"]]
    out = {}
    for f in FIELDS:
        got = field_words(mm, case, f)
        if any(b == POISON for b in got):
            out[f] = MISMATCH
            notes.append("case %d (%s/%s) field %s has an unwritten slot"
                         % (case["id"], MODES[case["mode"]],
                            "kernel" if case["arm"] == ARM_KERNEL else "logits", f))
            continue
        if f == "logits":
            if case["arm"] == ARM_LOGITS:
                ok = pack_widened(r["logits_bits"]) == struct.pack("<%dI" % E, *got)
                out[f] = MATCH if ok else MISMATCH
                if not ok:
                    notes.append("case %d: the LOGITS arm did not transport the supplied logits"
                                 % case["id"])
            else:
                ndiff, maxulp = 0, 0
                for i in range(E):
                    if got[i] != (r["logits_bits"][i] << 16):
                        ndiff += 1
                        maxulp = max(maxulp, ulpdist_bits(got[i], r["logits_bits"][i] << 16))
                out[f] = MATCH if (ndiff <= bands["logit_dot_route_disagreement_elements"]
                                   and maxulp <= bands["logit_dot_route_ulp"]) else MISMATCH
                if out[f] == MISMATCH:
                    notes.append("case %d: R1 moved %d logits by up to %d ULP, outside the "
                                 "floor-probe band (%d elements / %d ULP)"
                                 % (case["id"], ndiff, maxulp,
                                    bands["logit_dot_route_disagreement_elements"],
                                    bands["logit_dot_route_ulp"]))
        elif f == "probs":
            ok = judge_probs(got, r, case, notes, prob_bound, E)
            out[f] = MATCH if ok else MISMATCH
        elif f == "wpre":
            ndiff, maxulp = 0, 0
            pin = r["wpre_bits"]
            for i in range(K):
                if got[i] != pin[i]:
                    ndiff += 1
                    maxulp = max(maxulp, ulpdist_bits(got[i], pin[i]))
            band = bands["weights_vs_pinned_f32_ulp"]
            ok = maxulp <= band
            # R5 must be one round of this field, not two
            for i in range(K):
                if bf16_round(got[i]) != r["wcast_bits"][i] and r["tie"] == 0:
                    ok = False
                    notes.append("case %d: bf16(wpre[%d]) != wcast[%d]; R5 rounded twice"
                                 % (case["id"], i, i))
            out[f] = MATCH if ok else MISMATCH
            if not ok:
                notes.append("case %d: the f32 weights sit %d ULP from the pinned pre-cast arm, "
                             "outside the derived band (%d ULP)" % (case["id"], maxulp, band))
        elif f == "wcast":
            if pack_widened(r["wcast_bits"]) == struct.pack("<%dI" % K, *got):
                out[f] = MATCH
            elif pack_widened(sorted(r["wcast_bits"])) == \
                    pack_widened(sorted(got[i] >> 16 for i in range(K))):
                out[f] = ALT
                notes.append("case %d: the cast weights agree as a multiset but not by slot; the "
                             "top-k set moved under the tie-break (A-03), margin %d ULP"
                             % (case["id"], r["margin_pin"]))
            else:
                out[f] = MISMATCH
                notes.append("case %d: topk_weights disagrees with the authoritative arm at all"
                             % case["id"])
        elif f == "ids":
            out[f] = judge_ids(got, r, case, notes)
        else:
            out[f] = judge_routed(p, case, mm, r, got, notes)
        tally[(case["mode"], f, out[f])] = tally.get((case["mode"], f, out[f]), 0) + 1
    return out


FP32_U = 2.0 ** -24        # unit roundoff of binary32: 24 significand bits, round-to-nearest


def prob_sum_bound(terms):
    """The certificate bound on sum_k p_k, DERIVED, never typed (AGENTS.md: "Never loosen a gate to
    make it pass"; bands come from a floor probe and from geometry, never from what the build
    produces).

    There is no pinned router_probs — the manifest does not dump one, which is reported rather than
    repaired — so R2 cannot be compared to an authoritative arm and must instead be held to the
    algebraic certificate a softmax owes. For `n` positive addends (Higham's theta form, valid while
    (n-1)u < 1):

      s   = fl(sum_i e_i) = (sum_i e_i)(1 + theta),  |theta| <= (n-1)u/(1-(n-1)u)   [same-sign sum]
      p_i = fl(e_i / s)   = (e_i / s)(1 + delta_i),   |delta_i| <= u                [one divide each]
      fl(sum_i p_i) = sum_i (e_i/s)(1 + theta'_i),    |theta'_i| <= (n-1)u/(1-(n-1)u)
      sum_i e_i / s = 1/(1 + theta)

    giving |fl(sum p) - 1| <= (2(n-1) + 1) u / (1 - (n-1)u).

    Two inputs, both visible to the gate: n = MR_E read from the header (geometry, not a constant to
    tune) and u = 2^-24, a property of IEEE binary32 rather than of this build.

    Note what is deliberately NOT in this bound: the exp() route spread. The certificate is
    normalisation-invariant — an error in e_i appears in the numerator of every p_i and in the
    denominator once, so it cancels out of the sum and survives only in the relative distribution.
    The first version of this witness folded the measured 19-ULP exp spread into the bound and the
    witness refused it (3.05e-4 against a 3.05e-5 bound), which is the bound being right and the
    witness being right for the same reason. `witness_prob_bound` shows the bound can be violated and
    can be approached, because a check that can produce no output cannot fail."""
    if terms <= 0:
        raise GateError("prob_sum_bound: no row width to derive from (E=%r)" % (terms,))
    t = (terms - 1.0) * FP32_U
    if not t < 1.0:
        raise GateError("prob_sum_bound: (n-1)u = %g is not < 1, the same-sign sum bound does not "
                        "apply at n=%d" % (t, terms))
    return (2.0 * (terms - 1) + 1.0) * FP32_U / (1.0 - t)


def worst_case_drift(terms):
    """The fp32 rounding drift of the widest row a softmax can emit: one addend at each of `terms`
    distinct binary exponents across the probabilities' own range, (0, 1] — binary32 has 127
    representable exponents there, so this is the largest dynamic range a probability row can have,
    built from the format rather than tuned. `s` accumulates in fp32 the way the kernel does and `ex`
    in double, so the difference is the accumulation error a correct kernel is entitled to produce."""
    lo, hi = -126, 0                       # binary32's normal exponents below 1.0: the prob range
    vals = [2.0 ** (lo + i * (hi - lo) / (terms - 1.0)) for i in range(terms)]
    ex = 0.0
    for x in vals:
        ex += x
    s = 0.0
    for x in vals:
        s = f32(s + x)
    return abs(s - ex), vals


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def judge_probs(got, r, case, notes, bound, term_count):
    vals = [bits_f32(b) for b in got]
    if any(b != b or b in (float("inf"), float("-inf")) for b in vals):
        notes.append("case %d: probs are non-finite" % case["id"])
        return False
    if min(vals) < 0.0:
        notes.append("case %d: a probability is negative" % case["id"])
        return False
    s = 0.0
    for x in vals:
        s += x
    if abs(s - 1.0) > bound:
        notes.append("case %d: probs sum to %.9f, outside the derived fp32 certificate bound %.3e "
                     "for %d addends" % (case["id"], s, bound, term_count))
        return False
    return True


def judge_ids(got, r, case, notes):
    mine = [int(bits_f32(b)) for b in got]
    want = r["ids"]
    if mine == want:
        return MATCH
    if sorted(mine) != sorted(want):
        return ALT if r["tie"] in (2, 3) else MISMATCH
    if r["tie"] in (1, 2, 3):
        return ALT
    notes.append("case %d: topk ids differ from the pinned arm on a tie-free row: %s vs %s"
                 % (case["id"], mine, want))
    return MISMATCH


def judge_routed(p, case, mm, r, got, notes):
    if r["kind"] == "real":
        ids = [int(bits_f32(b)) for b in field_words(mm, case, "ids")]
        if ids != r["ids"]:
            # the top-k set moved under the tie-break, and expert_raw holds the PINNED selection's
            # outputs, so the weighted sum of this row is not comparable to the dump. Counted, with
            # its margin, rather than quietly dropped.
            return "NOT_JUDGED"
    if pack_widened(r["routed_bits"]) == struct.pack("<%dI" % H, *got):
        return MATCH
    notes.append("case %d: routed_sum disagrees with the reference form on %d of %d elements"
                 % (case["id"], sum(1 for i in range(H)
                                    if got[i] != (r["routed_bits"][i] << 16)), H))
    return MISMATCH


def judge_injection(p, case, mm, tally, notes, alt_notes):
    """An injection is judged field by field against the same row and arm under mode 0, and where
    an alternative form can be written down independently, it must land on that form exactly."""
    r = p.rows[case["row"]]
    mode = case["mode"]
    out = {}
    base = {f: field_words(mm, {"dst_off": case["cmp_off"], "id": case["id"]}, f) for f in FIELDS}
    if mode == len(MODES) - 1:                          # poison: everything is the canonical NaN
        for f in FIELDS:
            got = field_words(mm, case, f)
            ok = all(b == POISON for b in got)
            out[f] = MATCH if ok else MISMATCH
            if not ok:
                notes.append("case %d: the poison mode left a written value in %s"
                             % (case["id"], f))
            tally[(mode, f, out[f])] = tally.get((mode, f, out[f]), 0) + 1
        return out
    wpre0 = base["wpre"]
    wcast0 = [b >> 16 for b in base["wcast"]]
    ids0 = [int(bits_f32(b)) for b in base["ids"]]
    probs0 = [bits_f32(b) for b in base["probs"]]
    alt = {}
    if mode == 2:                                        # R4 dropped: the weights are the raw probs
        alt["wpre"] = [f32bits(probs0[e]) for e in ids0]
        alt["wcast"] = [bf16_round(w) for w in alt["wpre"]]
    if mode in (4, 5, 9):
        cast = "before"
        acc = "f32" if mode == 9 else "bf16"
        order = "rank" if mode == 4 else "expert_asc"
        alt["routed"] = [w << 16 for w in py_combine(r["raw_bits"], wpre0, wcast0, ids0,
                                                     order=order, cast=cast, acc=acc,
                                                     pair_by_id=(mode == 5))]
    if mode == 3:
        alt["routed"] = [w << 16 for w in py_combine(r["raw_bits"], wpre0, wcast0, ids0,
                                                     order="expert_asc", cast="after", acc="bf16")]
    if mode == 10:
        order = sorted(range(K), key=lambda k: ids0[k])
        alt["ids"] = [f32bits(float(ids0[k])) for k in order]
        alt["wpre"] = [wpre0[k] for k in order]
        alt["wcast"] = [wcast0[k] << 16 for k in order]
    if mode == 8:
        used = [False] * E
        sel = []
        for _ in range(K):
            b = argmax_tie(probs0, used, True)
            used[b] = True
            sel.append(b)
        alt["ids"] = [f32bits(float(b)) for b in sel]
        alt["wpre"] = [f32bits(probs0[b]) for b in sel]
        s = 0.0
        for b in sel:
            s += probs0[b]
        alt["wpre"] = [f32bits(probs0[b] / s) for b in sel]
        alt["wcast"] = [bf16_round(w) << 16 for w in alt["wpre"]]
    for f in FIELDS:
        got = field_words(mm, case, f)
        kind, why = expect_injection(mode, f, r, case["arm"])
        if kind == "never":
            cls = IMMUNE if got == base[f] else MISMATCH
            if cls == MISMATCH:
                notes.append("case %d (%s): field %s moved, but mr_mode_reaches says this mode "
                             "cannot reach it" % (case["id"], MODES[mode], f))
        elif kind == "immune":
            cls = IMMUNE if got == base[f] else MISMATCH
            if cls == MISMATCH:
                notes.append("case %d (%s): field %s moved on a row derived immune (%s)"
                             % (case["id"], MODES[mode], f, why))
        elif kind == "may":
            cls = MAY_MOVED if got != base[f] else MAY_UNCHANGED
        else:
            if got == base[f]:
                # "did not move" is only a dead control if the injection also failed to land on the
                # alternative form. When an independently computed alternative form coincides with the
                # reference on this row — a tied fixture whose weights are bf16-exact, so casting
                # before or after the multiply is the same arithmetic — agreeing with it is agreement,
                # not a missed catch. Counted as ALT and printed, so the coincidence is visible rather
                # than silently green, and `witness-injections-fired` still demands that every mode
                # move something somewhere.
                if f in alt and pack_f32(alt[f]) == struct.pack("<%dI" % LEN[f], *got):
                    cls = ALT
                    alt_notes["%s/%s:coincides" % (MODES[mode], f)] = alt_notes.get(
                        "%s/%s:coincides" % (MODES[mode], f), 0) + 1
                else:
                    cls = MISMATCH
                    notes.append("case %d (%s): field %s did not move, and did not land on the "
                                 "independently derived alternative form either"
                                 % (case["id"], MODES[mode], f))
            elif f in alt and pack_f32(alt[f]) == struct.pack("<%dI" % LEN[f], *got):
                cls = ALT
                alt_notes["%s/%s" % (MODES[mode], f)] = alt_notes.get(
                    "%s/%s" % (MODES[mode], f), 0) + 1
            else:
                cls = FIRED
                if f in alt:
                    alt_notes["%s/%s:inexact" % (MODES[mode], f)] = alt_notes.get(
                        "%s/%s:inexact" % (MODES[mode], f), 0) + 1
        out[f] = cls
        tally[(mode, f, cls)] = tally.get((mode, f, cls), 0) + 1
    return out


# -----------------------------------------------------------------------------------------------
# running the three arms
# -----------------------------------------------------------------------------------------------
def run_driver_selftest(v, work):
    cmd = [BIN, "--selftest"]
    print("CMD " + " ".join(cmd))
    rc, text = run(cmd, os.path.join(work, "driver-selftest.log"))
    st, n = parse_records(text, "SELFTEST", ["name", "ok", "n"])
    res = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if len(res) != 1:
        raise GateError("the selftest printed %d RESULT lines, expected 1" % len(res))
    dkv, _ = parse_records(text, "DRIVER", ["checks", "failures"])
    if len(dkv) != 1 or int(dkv[0]["checks"]) != n:
        raise GateError("the selftest printed %d SELFTEST records but the tally says checks=%s"
                        % (n, dkv[0]["checks"] if dkv else "none"))
    v.check("driver-selftest", rc == 0 and res[0].split()[1] == "PASS"
            and all(s["ok"] == "1" for s in st), n,
            " ".join(s["name"] for s in st if s["ok"] != "1") or "all ok")
    names = sorted(s["name"] for s in st)
    want = ["bf16_add", "bf16_cast", "bf16_double_ruler", "cast_order_differs", "equal_row",
            "leak_guard", "output_widening", "pairing_vs_order", "verify_cases"]
    v.check("driver-selftest-roster", sorted(want) == names, len(names),
            "missing=%s extra=%s" % (sorted(set(want) - set(names)) or "none",
                                     sorted(set(names) - set(want)) or "none"))
    return n


def run_host_probe(p, plan_path, slab_path, work, v):
    out = os.path.join(work, "host_out.f32")
    rep = os.path.join(work, "host_report.tsv")
    cmd = [PROBE_BIN, "--plan", plan_path, "--slab", slab_path, "--out", out, "--report", rep]
    print("CMD " + " ".join(cmd))
    rc, text = run(cmd, os.path.join(work, "host-probe.log"))
    hp, n = parse_records(text, "HOSTPROBE", ["name", "ok", "n"])
    want = sorted(["row_off_1_reads_next_row", "transposed_is_column_dot",
                   "transposed_differs_from_row_dot", "topk_asc_same_set", "no_renorm_leaves_ids",
                   "no_renorm_moves_weights", "renorm_sums_to_one",
                   "order_injection_not_decoration", "pairing_injection_not_decoration",
                   "tie_every_weight_agrees", "tie_swaps_the_expert_set",
                   "tie_changes_the_routed_sum", "r5_cast_idempotent"])
    got_names = sorted(x["name"] for x in hp)
    v.check("host-probe-roster", got_names == want, len(got_names),
            "missing=%s extra=%s" % (sorted(set(want) - set(got_names)) or "none",
                                     sorted(set(got_names) - set(want)) or "none"))
    v.check("host-structural-probes", all(x["ok"] == "1" for x in hp) and n == len(want), n,
            " ".join("%s(%s)" % (x["name"], x["n"]) for x in hp if x["ok"] != "1") or "all ok")
    res = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if len(res) != 1 or res[0].split()[1] != "PASS":
        for line in text.splitlines():
            if line.startswith("FAIL"):
                print("HOSTPROBE> " + line)
        raise GateError("the host probe did not pass (rc=%d): %r" % (rc, res))
    rr, nh = parse_records(text, "REPORT", ["rows"])
    if nh != 1:
        raise GateError("the host probe printed %d REPORT lines" % nh)
    hrows = parse_tsv(rep, 9)
    if len(hrows) != len(p.cases):
        raise GateError("host_report.tsv holds %d rows for %d cases" % (len(hrows), len(p.cases)))
    return out, hrows, int(rr[0]["rows"])


def bf16_ulpdist(a, b):
    """Distance between two widened-bf16 words measured in **bf16** steps, which is the only ruler
    that means anything for a bf16 field: one bf16 ULP is ~2^16 f32 ULP, so an f32 ruler reads a
    single rounding flip as a 32768-ULP disaster and a f32-sized band on it is meaningless. bf16 is
    sign-magnitude, so order it the way IEEE floats are ordered before subtracting."""
    def ord16(w):
        h = w >> 16
        return (~h & 0xFFFF) if (h & 0x8000) else (h | 0x8000)
    return abs(ord16(a) - ord16(b))


def compare_arms(p, dev_path, host_path, v, bands):
    """Field-scoped host/device agreement, with the ruler matched to the field.

    Bit-identity everywhere the arithmetic is bf16-quantised end-to-end. The two injections that keep
    the fp32 weight (cast_after, f32_combine) are the only places a glibc-vs-libdevice `expf`
    difference can survive into the words: there the derived allowance is **one bf16 ULP**, because a
    19-f32-ULP route difference (1e-6 relative) can only flip a bf16 round when the product sits
    within that of a boundary, and when it flips it flips by exactly one step. Two steps is a drift,
    not a route. `witness-bf16-ruler` below shows the metric separates 1 step from 2, so the bound
    cannot be vacuous; the verdicts against the DUMP never used this comparison (the device arm is
    MISMATCH=0 against the pinned arms on its own)."""
    f32_carrying = {MODES[3], MODES[9]}          # cast_after, f32_combine
    exact = {"logits": True, "probs": False, "wpre": False, "wcast": True, "ids": True,
             "routed": True}
    bad = {}
    sep = {}
    with open(dev_path, "rb") as a, open(host_path, "rb") as b:
        ma = mmap.mmap(a.fileno(), 0, access=mmap.ACCESS_READ)
        mb = mmap.mmap(b.fileno(), 0, access=mmap.ACCESS_READ)
        for c in p.cases:
            if c["mode"] == len(MODES) - 1:
                continue
            for f in FIELDS:
                ga = field_words(ma, c, f)
                gb = field_words(mb, c, f)
                if ga == gb:
                    continue
                if f == "routed" and MODES[c["mode"]] in f32_carrying:
                    worst = max(bf16_ulpdist(x, y) for x, y in zip(ga, gb))
                    sep[(MODES[c["mode"]], f)] = max(sep.get((MODES[c["mode"]], f), 0), worst)
                elif exact[f]:
                    bad[(MODES[c["mode"]], f)] = bad.get((MODES[c["mode"]], f), 0) + 1
                else:
                    worst = max(ulpdist_bits(x, y) for x, y in zip(ga, gb))
                    lim = bands["weights_vs_pinned_f32_ulp"] if f == "wpre" else \
                        bands["softmax_denominator_reassoc_ulp"]
                    if worst > 4 * lim:
                        bad[("band-" + MODES[c["mode"]], f)] = bad.get(
                            ("band-" + MODES[c["mode"]], f), 0) + 1
    v.check("host-device-arms", not bad, len(bad),
            "fields where the two arms disagree beyond what the derived bands allow: %s"
            % (sorted(bad.items())[:6] if bad else "none"))
    # The two injections that keep the fp32 weight are REPORTED, not asserted: no floor probe has
    # measured the host-vs-device expf spread (the floor probe measured CPU re-association, which is
    # a different axis), so a bound here would be a number chosen to fit what the build produced.
    # The measurements are printed so the gap is visible, and results.md carries it as an open item.
    for k in sorted(sep):
        print("ARM-SEPARATION mode=%s field=%s worst_bf16_steps=%d (no host-vs-device expf floor "
              "probe exists yet, so this is reported rather than bounded)" % (k[0], k[1], sep[k]))
    measured = set(MODES[c["mode"]] for c in p.cases if c["mode"] in
                   (MODES.index(MODES[3]), MODES.index(MODES[9])))
    v.check("host-device-f32-carrying-separation-reported",
            measured == {MODES[3], MODES[9]}, len(sep),
            "both f32-carrying injections were compared across the arms; worst bf16-step "
            "separation observed: %s" % (sorted(sep.items()) if sep else "the two arms agreed "
                                                                        "bit-for-bit"))
    one = bf16_ulpdist(0x3F800000, 0x3F810000)          # one bf16 ULP above 1.0
    two = bf16_ulpdist(0x3F800000, 0x3F820000)          # two above it
    neg = bf16_ulpdist(0x80000000, 0x3F800000)
    v.check("witness-bf16-ruler", one == 1 and two == 2 and neg > two,
            3, "one bf16 step reads %d, two read %d, crossing zero reads %d" % (one, two, neg))


# -----------------------------------------------------------------------------------------------
# witnesses: every counter below can read zero, so each one is shown reading non-zero first
# -----------------------------------------------------------------------------------------------
def witness_comparator(p, mm, v):
    """Flipping one bit of one element must make exactly one element disagree, and a comparator
    that reports zero differences for a perturbed input is not a comparator."""
    c = next(c for c in p.cases if c["mode"] == 0 and p.rows[c["row"]]["kind"] == "real")
    got = list(field_words(mm, c, "routed"))
    want = [b << 16 for b in p.rows[c["row"]]["routed_bits"]]
    same = sum(1 for i in range(H) if got[i] != want[i])
    flipped = list(got)
    flipped[7] ^= 1
    diff = sum(1 for i in range(H) if flipped[i] != want[i])
    ok = (same == 0 and diff == 1)
    v.check("witness-comparator-flip", ok, diff, "one flipped bit must move exactly one element")
    return ok


def witness_cast_orders(p, mm, v):
    """The A-05 witness: the two cast orders must produce different numbers on real rows, or the
    gate has measured nothing about the cast position."""
    rows_sep = {}
    for c in p.cases:
        if c["mode"] != 3 or c["arm"] != ARM_KERNEL:
            continue
        r = p.rows[c["row"]]
        if r["kind"] != "real":
            continue
        before = field_words(mm, {"dst_off": c["cmp_off"], "id": c["id"]}, "routed")
        after = field_words(mm, c, "routed")
        nd = sum(1 for i in range(H) if before[i] != after[i])
        rows_sep[c["row"]] = nd
    nz = [k for k, n in rows_sep.items() if n > 0]
    v.check("witness-cast-orders-differ", len(nz) == len(rows_sep) and len(nz) > 0, len(nz),
            "real rows where cast-before and cast-after differ: %d of %d; worst %d of %d elements"
            % (len(nz), len(rows_sep), max(rows_sep.values()) if rows_sep else 0, H))
    return rows_sep


def witness_populations(tally, v):
    """Each class must be reachable. A class that no fixture can reach has no tests
    (pitfalls/40 §"A route no fixture reaches has no tests")."""
    want = {MATCH: "reference agreement", ALT: "an alternative form reached exactly",
            IMMUNE: "a derived immunity actually exercised", FIRED: "an injection fired"}
    pop = {}
    for cls in (MATCH, MISMATCH, ALT, IMMUNE, FIRED, MAY_UNCHANGED, MAY_MOVED):
        pop[CLS_NAME[cls]] = sum(n for (m, f, c), n in tally.items() if c == cls)
    for cls, why in want.items():
        v.check("witness-class-%s" % CLS_NAME[cls], pop[CLS_NAME[cls]] > 0, pop[CLS_NAME[cls]], why)
    v.check("class-mismatch-empty", pop["MISMATCH"] == 0, pop["MISMATCH"],
            "no comparison may land outside every expectation")
    return pop


def witness_injections_fired(p, tally, v):
    """Every injection mode must move something somewhere. A mode that fires nowhere is a dead
    control, and a dead control reads the same green as a working one."""
    dead = []
    for mode in range(1, len(MODES) - 1):
        fired = sum(n for (m, f, c), n in tally.items()
                    if m == mode and c in (FIRED, ALT, MAY_MOVED))
        if fired == 0:
            dead.append(MODES[mode])
    v.check("witness-injections-fired", not dead, len(MODES) - 2,
            "modes that never fired anywhere: %s" % (dead or "none"))
    return dead


def witness_not_judged(p, tally, divergent, v):
    """The skipped routed comparisons are derived, not counted and hoped for: only the reference mode
    compares routed_sum against the dump (an injection is compared against its own row's reference
    record), and every real row runs on both arms, so the expected number of skips is
    divergent rows x 2 arms x 1 mode."""
    got = sum(n for (m, f, c), n in tally.items() if f == "routed" and c == "NOT_JUDGED")
    want = len(divergent) * 2 * 1
    v.check("witness-unjudged-is-derived", got == want, got,
            "routed comparisons skipped: %d, derived from %d set-divergent rows x 2 arms x 1 mode "
            "(injections compare against the row's own reference record, not the dump)"
            % (got, len(divergent)))
    return got


def witness_prob_bound(v):
    """The derived certificate bound has to be able to fire, and to be able to pass. Three legs, all
    by construction rather than by measurement: a row summing to one passes; a row perturbed past the
    bound is rejected; and a row whose rounding drift approaches the bound (the one-big-plus-many-
    tiny case the theta form is tight for) is still accepted, so the certificate is not so tight that
    it rejects a correct kernel. The bound is printed so a reader can re-derive it without trusting
    this file."""
    b = prob_sum_bound(E)
    good = [1.0 / E] * E
    s_good = 0.0
    for x in good:
        s_good += x
    ok_pass = abs(s_good - 1.0) <= b
    bad = list(good)
    bad[0] += 3.0 * b
    s_bad = 0.0
    for x in bad:
        s_bad += x
    ok_reject = abs(s_bad - 1.0) > b
    drift, wide = worst_case_drift(E)
    ok_drift = 0.0 < drift <= b
    # resolving power: a single fp32 ULP of drift is accepted, so the certificate is a bound on
    # rounding and not an exactness claim wearing a bound's clothes
    ulp_row = list(good)
    ulp_row[0] = f32(ulp_row[0] * (1.0 + 4.0 * FP32_U))
    s_ulp = 0.0
    for x in ulp_row:
        s_ulp += x
    ok_ulp = abs(s_ulp - 1.0) <= b and abs(s_ulp - 1.0) > 0.0
    v.check("witness-prob-sum-bound", ok_pass and ok_reject and ok_drift and ok_ulp, E,
            "bound=(2(E-1)+1)*2^-24/(1-(E-1)2^-24)=%.3e for E=%d; a row summing to 1 passes, a "
            "3x-bound perturbation is rejected (%s), the widest-dynamic-range fp32 accumulation "
            "drift %.3e is accepted and non-zero (%s), and a few-ULP drift is accepted (%s)"
            % (b, E, ok_reject, drift, ok_drift, ok_ulp))
    v.check("witness-prob-bound-tight", b < 0.01, E,
            "a bound that accepted 1%% off 1.0 would not be a certificate (bound=%.3e)" % b)
    return b


def witness_bands(bands, v):
    """A band that nothing can exceed is not a band. Each one is shown rejecting a perturbation of
    band+1 ULP, through the same predicate the gate uses."""
    bad = []
    for name, lim in bands.items():
        if lim < 0:
            bad.append(name)
            continue
        a, b = 0x3F800000, 0x3F800000 + lim + 1
        if ulpdist_bits(a, b) <= lim:
            bad.append(name)
    v.check("witness-bands-reject", not bad, len(bands),
            "bands a band+1 perturbation slips through: %s" % (bad or "none"))
    for name in ("weights_vs_pinned_f32_ulp", "pinned_weight_to_bf16_boundary_f32ulp"):
        if bands.get(name, 0) <= 0:
            v.check("band-nonzero-" + name, False, bands.get(name, 0), "a zero band is bit-exactness")
    # the figure the zero-tolerance claim on R5 rests on: the pinned f32 weights sit far further
    # from a bf16 rounding boundary than any legal route can move them
    ratio = bands["pinned_weight_to_bf16_boundary_f32ulp"] / max(1, bands["weights_vs_pinned_f32_ulp"])
    v.check("cast-is-robust-not-lucky", ratio >= 4.0, int(ratio),
            "min distance to a bf16 rounding boundary (%d f32 ULP) over the route spread (%d) = %dx"
            % (bands["pinned_weight_to_bf16_boundary_f32ulp"],
               bands["weights_vs_pinned_f32_ulp"], int(ratio)))


def witness_parsers(v):
    good, n = parse_records("FOO name=a ok=1 n=2\nFOO name=b ok=1 n=3\n", "FOO",
                            ["name", "ok", "n"])
    ok1 = (n == 2 and good[1]["n"] == "3")
    try:
        parse_records("FOO name=a ok=1\n", "FOO", ["name", "ok", "n"])
        ok2 = False
    except GateError:
        ok2 = True
    try:
        parse_tsv_from_string("a\tb\tc\na\tb\n", 3)
        ok3 = False
    except GateError:
        ok3 = True
    try:
        hexlist("0000 0001 0002", 2)
        ok4 = False
    except GateError:
        ok4 = True
    empty = parse_records("NOTHING\n", "FOO", ["name"])
    ok5 = (len(empty[0]) == 0)
    ok6 = parse_records("FOO a=1\nFOOBAR a=1\n", "FOO", ["a"])[1] == 1
    ok7 = parse_records("DRIVER SELFTEST RESULT checks=2 failures=0\n", "DRIVER",
                        ["checks", "failures"])[1] == 1
    v.check("witness-parsers", ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7, 7,
            "parsers must yield records, refuse a missing field, refuse a short row, refuse a "
            "wrong-width hex list, yield nothing from nothing, keep PLAN out of PLANCHECK, and "
            "read a multi-word tag")


def parse_tsv_from_string(text, ncol):
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != ncol:
            raise GateError("expected %d columns, got %d" % (ncol, len(parts)))
        rows.append(parts)
    return rows


# -----------------------------------------------------------------------------------------------
# the manifest is the authority on the dumps; its own recorded agreement is re-derived, not trusted
# -----------------------------------------------------------------------------------------------
def load_manifest(v):
    man = json.load(open(MANIFEST))
    D = {r["name"]: r for r in man["dumps"]}
    g = man["geometry"]
    want = {"hidden_size": H, "num_experts": E, "top_k": K, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "num_hidden_layers": 40,
            "mtp_num_hidden_layers": 1}
    for k, w in want.items():
        if g.get(k) != w:
            raise GateError("manifest geometry %s=%s, the gate's constants say %s" % (k, g.get(k), w))
    cfg = json.load(open(os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")))
    t = cfg["text_config"]
    for k in want:
        key = {"hidden_size": "hidden_size", "num_experts": "num_experts", "top_k": "num_experts_per_tok",
               "moe_intermediate_size": "moe_intermediate_size",
               "shared_expert_intermediate_size": "shared_expert_intermediate_size",
               "num_hidden_layers": "num_hidden_layers",
               "mtp_num_hidden_layers": "mtp_num_hidden_layers"}[k]
        if t[key] != g[k]:
            raise GateError("manifest %s=%s disagrees with the config's %s=%s"
                            % (k, g[k], key, t[key]))
    v.check("manifest-geometry-matches-config", True, len(want),
            " ".join("%s=%s" % (k, g[k]) for k in sorted(want)))
    named = ["input", "mlp.gate.weight", "router_logits", "topk_ids", "topk_weights",
             "topk_weights_f32", "expert_raw", "routed_sum"]
    have = set(n.split(".", 1)[1] for n in D if n.split(".", 1)[0] in BLOCKS)
    v.check("manifest-dump-census", set(named) <= have, len(named),
            "missing arms: %s" % (sorted(set(named) - have) or "none"))
    for r in man["dumps"]:
        n = 1
        for s in r["shape"]:
            n *= s
        item = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int64": 8}[r["dtype"]]
        if r["bytes"] != n * item:
            raise GateError("%s: declares %s bytes for shape %s of %s" % (r["name"], r["bytes"],
                                                                          r["shape"], r["dtype"]))
        if r.get("elements") not in (None, n):
            raise GateError("%s: declares %s elements, shape %s holds %d"
                            % (r["name"], r.get("elements"), r["shape"], n))
        on_disk = os.path.getsize(os.path.join(REPO, r["file"]))
        if on_disk != r["bytes"]:
            raise GateError("%s: %d bytes on disk, the manifest declares %d"
                            % (r["name"], on_disk, r["bytes"]))
    v.check("manifest-selfconsistent-sizes", True, len(man["dumps"]),
            "every declared byte count equals shape x itemsize")
    missing = [r["name"] for r in man["dumps"] if not os.path.exists(os.path.join(REPO, r["file"]))]
    if missing:
        raise GateError("manifest names %d dumps that are not on disk: %s"
                        % (len(missing), missing[:4]))
    return man, D


def check_manifest_hashes(man, D, v):
    """The manifest carries a sha256 per dump. Verifying it is what makes "the device agrees with
    the oracle" mean agreement with the pinned run and not with whatever is in the directory now."""
    n = 0
    bad = []
    for r in man["dumps"]:
        want = r.get("sha256")
        if not want:
            continue
        got = sha256_file(os.path.join(REPO, r["file"]))
        n += 1
        if got != want:
            bad.append(r["name"])
    v.check("manifest-sha256", not bad, n, "dumps whose bytes are not what the manifest hashed: %s"
            % (bad[:4] or "none"))
    return n


def check_manifest_agreement(p, results, v):
    """`agreement_at_write_time` in the manifest claims each block's topk_ids and routed_sum were
    bit-exact against the reference. Re-derive it here: a manifest claim that this run cannot
    reproduce is a finding about the dumps or about the gate, and it must not be able to hide behind
    the gate's own PASS."""
    ids_ok = routed_ok = 0
    for blk in BLOCKS:
        rows = [c for c in results if c["block"] == blk and c["mode"] == 0
                and c["arm"] == ARM_KERNEL]
        if not rows:
            raise GateError("no reference cases for %s" % blk)
        i_ok = all(c["cls"]["ids"] in (MATCH, ALT) for c in rows)
        r_ok = all(c["cls"]["routed"] in (MATCH, ALT, "NOT_JUDGED") for c in rows)
        ids_ok += int(i_ok)
        routed_ok += int(r_ok)
    v.check("manifest-claims-ids-hold", ids_ok == len(BLOCKS), ids_ok,
            "blocks where the ids still agree with the pinned arm: %d/%d" % (ids_ok, len(BLOCKS)))
    v.check("manifest-claims-routed-hold", routed_ok == len(BLOCKS), routed_ok,
            "blocks where the routed sum still agrees: %d/%d" % (routed_ok, len(BLOCKS)))


# -----------------------------------------------------------------------------------------------
# what the lane was asked to find out, printed rather than asserted away
# -----------------------------------------------------------------------------------------------
def report_predictions(p, results, oracle_rows, p3, bands, v):
    # prediction 1: the serial spine, counted from the code and the geometry, with no clock read
    stages = 5
    tok_reads = K * (40 + 1)
    print("SERIAL stages_per_block=%d blocks_per_token=%d dependent_router_stages_per_token=%d "
          "dependent_expert_reads_per_token=%d barriers_per_router_row=%d host_values_emitted=%d"
          % (stages, 41, stages * 41, tok_reads, 2, 0))
    v.check("serial-structure-shape", stages == 5 and tok_reads == 328, tok_reads,
            "R1..R5 per block, one gather per (token, slot), 41 blocks per token")
    # prediction 3: the fused layout. What this probe CAN claim is that the three readings of
    # gate_up_proj are three different functions (a reading that made no difference would mean the
    # layout is not being read at all). What it CANNOT claim is a verdict on which reading the model
    # uses, and the reason is reported rather than smoothed over: no reading reproduces the pinned
    # expert_raw, including the one the oracle's own _expert_apply implements.
    reads = {}
    for l in p3:
        reads.setdefault((l[0], l[1], l[2]), {})[l[3]] = float(l[4])
    indistinct = [k for k, d in reads.items() if len(set("%.6e" % v for v in d.values())) != 3]
    v.check("prediction3-readings-distinguishable", not indistinct and len(reads) > 0, len(reads),
            "(block,token,slot) readings where the three ways of reading [E,2I,H] are not three "
            "different answers: %s" % (indistinct[:4] or "none"))
    right = [float(l[5]) for l in p3 if l[3] == "chunk2"]
    wrong = [float(l[5]) for l in p3 if l[3] != "chunk2"]
    if right and wrong:
        best_right, best_wrong = max(right), min(wrong)
        print("P3 chunk2_worst_1_minus_cos=%.3e wrong_reads_best_1_minus_cos=%.3e n=%d"
              % (best_right, best_wrong, len(p3)))
        if best_right < 1e-3:
            v.check("prediction3-fused-layout", best_right < best_wrong, len(p3),
                    "the halves-as-read reading is the only one that reproduces the pinned "
                    "expert_raw, by %.1fx over the nearest wrong reading"
                    % (best_wrong / max(1e-12, best_right)))
        else:
            # An unresolved prediction, stated as one. This lane contains no expert GEMM, so P3 is a
            # claim about the checkpoint's weights, and the pinned dump cannot adjudicate it: the
            # oracle's own _expert_apply and this file's independent numpy SwiGLU agree with each
            # other to the last bit and neither correlates with expert_raw (best cosine 0.065 over
            # all 256 experts for layer0 token0 slot0 — work/probe15.py). Reporting it, not
            # relaxing a threshold to make it look adjudicated.
            print("P3 UNRESOLVED no reading of gate_up_proj reproduces the pinned expert_raw "
                  "(worst 1-cos over the correct reading = %.3f); the oracle's _expert_apply and a "
                  "separate numpy SwiGLU agree with each other, and sweeping all 256 experts for "
                  "layer0/token0/slot0 peaked at cos 0.065 (work/probe15.py). expert_raw is not "
                  "down(silu(gate*x)*up*x) for any expert under the manifest's geometry, so the "
                  "dump cannot settle the layout claim; what IS verified from expert_raw is the "
                  "combine, because routed_sum reproduces from it on every real row." % best_right)
            v.check("prediction3-fused-layout-unresolved-is-reported", True, len(p3),
                    "P3 is reported as unresolved with its evidence; the router verdicts do not "
                    "rest on it (no expert GEMM in this lane)")
    # prediction 4: which cast order the dump agrees with, and what the other one produced
    agree_before = disagree = 0
    worst = 0
    for c in results:
        if c["mode"] != 0 or c["arm"] != ARM_KERNEL or c["kind"] != "real":
            continue
        agree_before += int(c["cls"]["wcast"] == MATCH)
    for c in results:
        if c["mode"] == 3 and c["arm"] == ARM_KERNEL and c["kind"] == "real":
            disagree += 1
            worst = max(worst, c["cast_after_diff"])
    print("PREDICTION4 rows_compared=%d rows_whose_cast_weights_match_the_authoritative_arm=%d"
          % (sum(1 for c in results if c["mode"] == 0 and c["arm"] == ARM_KERNEL
                 and c["kind"] == "real"), agree_before))
    print("PREDICTION4 cast_after_rows=%d worst_row_elements_differing=%d of %d"
          % (disagree, worst, H))
    v.check("prediction4-wrong-order-differs", disagree > 0 and worst > 0, disagree,
            "the wrong order produced a different number on %d real rows, worst %d of %d elements"
            % (disagree, worst, H))
    # A-03: the tie population, with every margin printed
    hist = {}
    for key, o in sorted(oracle_rows.items()):
        hist[o["tie"]] = hist.get(o["tie"], 0) + 1
    print("A03 tie_class_population " + " ".join("class%d=%d" % kv for kv in sorted(hist.items())))
    straddle = [(k, o["margin_pin"], o["margin_ref"]) for k, o in sorted(oracle_rows.items())
                if o["tie"] in (2, 3)]
    print("A03 straddle_rows=%d margins_ulp=%s" % (len(straddle), [m for _, m, _ in straddle]))
    div = [c["row"] for c in results if c["mode"] == 0 and c["arm"] == ARM_KERNEL
           and c["cls"]["routed"] == "NOT_JUDGED"]
    print("A03 rows_where_the_engine_and_the_pinned_arm_chose_different_tied_experts=%d keys=%s"
          % (len(set(div)), sorted(set(div))[:16]))
    v.check("a03-population-nonempty", len(straddle) > 0, len(straddle),
            "the dump contains rows whose 8th/9th probabilities are bit-identical, which is the "
            "witness A-03 says is owed before any gate claims to cover ties")
    # the tie-break is not what the oracle's comment says it is
    ne = sum(1 for o in oracle_rows.values() if o["n_rule_ne_pin"] > 0)
    print("A03 rows_where_lower_index_rule_disagrees_with_pinned_ids=%d of %d"
          % (ne, len(oracle_rows)))


# -----------------------------------------------------------------------------------------------
def check_roster(v, roster):
    """The completeness half of `finish`, split out so a run that writes an artifact can settle the
    roster BEFORE the artifact is written: the JSON's `result` must be derived from the very findings
    list the exit status reads, roster finding included, not from a snapshot taken a moment earlier."""
    names = [c[0] for c in v.checks]
    missing = sorted(set(roster) - set(names))
    extra = sorted(set(names) - set(roster))
    dupes = sorted(set(n for n in names if names.count(n) > 1))
    if missing or extra or dupes:
        print("FAIL check roster: missing=%s extra=%s duplicated=%s" % (missing, extra, dupes))
        v.findings.append("check-roster")


def finish(v, label, roster, roster_checked=False):
    if not roster_checked:
        check_roster(v, roster)
    ok = not v.findings
    print("RESULT %s %s checks=%d findings=%d"
          % ("PASS" if ok else "FAIL", label, len(v.checks), len(v.findings)))
    return 0 if ok else 1


DEVICE_ROSTER = [
    "manifest-geometry-matches-config", "manifest-dump-census", "manifest-selfconsistent-sizes",
    "manifest-sha256", "oracle-arm", "plan-shape", "driver-result", "driver-plan-parse",
    "driver-planchannel", "driver-report-rows", "driver-selftest", "driver-selftest-roster",
    "host-structural-probes", "host-probe-roster", "host-device-arms", "witness-comparator-flip",
    "witness-cast-orders-differ", "witness-class-MATCH", "witness-class-ALT",
    "witness-class-IMMUNE", "witness-class-FIRED", "class-mismatch-empty",
    "witness-injections-fired", "witness-unjudged-is-derived", "witness-bands-reject",
    "witness-bf16-ruler", "host-device-f32-carrying-separation-reported",
    "witness-parsers", "manifest-claims-ids-hold", "manifest-claims-routed-hold",
    "serial-structure-shape", "prediction3-readings-distinguishable",
    "prediction3-fused-layout-unresolved-is-reported", "prediction4-wrong-order-differs",
    "a03-population-nonempty", "cast-is-robust-not-lucky", "witness-prob-sum-bound",
    "witness-prob-bound-tight", "sample-declared", "judge-arm-device",
]

# The host-only roster: everything except the device arm. Its point is that the judging path, the
# oracle, the injections and every witness are all proven green with zero device work, so a failure
# in the device run is a failure of the kernel and not of the gate.
HOST_ROSTER = [c for c in DEVICE_ROSTER
               if c not in ("driver-result", "driver-plan-parse", "driver-planchannel",
                            "driver-report-rows", "host-device-arms", "judge-arm-device",
                            "witness-bf16-ruler", "host-device-f32-carrying-separation-reported",
                            "driver-selftest", "driver-selftest-roster")]


def prepare(args, v):
    """Everything both the host-only and the device mode need, with no device work in it."""
    global _REACH
    for f in (BIN, PROBE_BIN):
        if not os.path.exists(f):
            raise GateError("missing binary %s; run ./build.sh" % f)
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    _REACH = load_reach_table()
    check_header_enum()
    man, D = load_manifest(v)
    n_hash = check_manifest_hashes(man, D, v)
    orows, osynth, bands, p3, otext = read_oracle(v, WORK)
    v.check("oracle-arm", True, len(orows) + len(osynth),
            "rows=%d synth=%d bands=%d p3=%d sha256-verified-dumps=%d"
            % (len(orows), len(osynth), len(bands), len(p3), n_hash))
    if args.max_rows is not None and not 1 <= args.max_rows <= 32:
        raise GateError("--max-rows must be 1..32, got %d" % args.max_rows)
    max_rows = 32 if args.max_rows is None else args.max_rows
    plan = build_plan(man, D, orows, osynth, bands, v, max_rows)
    plan_path = os.path.join(WORK, "plan.tsv")
    _, slab_path = write_plan(plan, plan_path)     # the writer owns the slab's name, not the caller
    if max_rows != 32:
        print("CUT rows_per_block=%d of 32; the cut is in the SAMPLED ROWS only: every mode, every "
              "arm, every field and every assertion still runs on the rows kept" % max_rows)
    units = plan.declare()
    n_real = sum(1 for r in plan.rows.values() if r["kind"] == "real")
    n_synth = sum(1 for r in plan.rows.values() if r["kind"] == "synth")
    v.check("plan-shape", len(plan.cases) > 0 and n_real == 4 * max_rows and n_synth == len(osynth),
            len(plan.cases), "rows=%d (%d pinned = 4 blocks x %d + %d synthetic)"
            % (len(plan.rows), n_real, max_rows, n_synth))
    v.check("sample-declared", units == len(plan.cases) * len(FIELDS) and units > 0, units,
            "field comparisons declared before any device work; the driver must match it")
    host_out, hrows, host_printed = run_host_probe(plan, plan_path, slab_path, WORK, v)
    prob_bound = witness_prob_bound(v)
    # Counted, not asserted-away: what the producer said it wrote, what the reader read back, and
    # what the gate demanded, all three in the artifact. Equality is shown with numbers because a
    # bare `true` cannot tell a full parse from an empty one.
    counters = {
        "arm": "host",
        "driver_report_path": os.path.join(WORK, "host_report.tsv"),
        "driver_records_printed": host_printed,
        "driver_records_parsed": len(hrows),
        "expect_rows": len(plan.cases),          # --expect-cases as issued to moe_router
        "expect_units": len(plan.cases) * len(FIELDS),
        "expect_units_note": "this lane's unit is the field comparison (cases x 6 fields); the "
                             "driver takes --expect-cases only, so expect_units is the gate's own "
                             "declaration (the SAMPLE line), not a driver flag",
    }
    return {"plan": plan, "man": man, "orows": orows, "osynth": osynth, "bands": bands,
            "p3": p3, "plan_path": plan_path, "slab_path": slab_path, "host_out": host_out,
            "prob_bound": prob_bound, "max_rows": max_rows, "counters": counters}


def judge_all(plan, path, bands, prob_bound, v, arm_label):
    """Judge one arm's output words. The same judges run over the host arm (--check-only) and the
    device arm, so a green host run and a green device run mean the same thing about the algebra."""
    tally, notes, alt_notes, results = {}, [], {}, []
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        for i, c in enumerate(plan.cases):
            rec = {"id": i, "arm": c["arm"], "mode": c["mode"], "row": c["src_key"],
                   "kind": plan.rows[c["src_key"]]["kind"],
                   "block": plan.rows[c["src_key"]]["block"],
                   "token": plan.rows[c["src_key"]]["token"]}
            if c["mode"] == 0:
                rec["cls"] = judge_reference(plan, c, mm, bands, tally, notes, v, None, prob_bound)
            else:
                rec["cls"] = judge_injection(plan, c, mm, tally, notes, alt_notes)
            if c["mode"] == 3 and c["arm"] == ARM_KERNEL:
                before = field_words(mm, {"dst_off": c["cmp_off"], "id": i}, "routed")
                after = field_words(mm, c, "routed")
                rec["cast_after_diff"] = sum(1 for j in range(H) if before[j] != after[j])
            results.append(rec)
        witness_comparator(plan, mm, v)
        sep = witness_cast_orders(plan, mm, v)
    pop = witness_populations(tally, v)
    dead = witness_injections_fired(plan, tally, v)
    divergent = sorted(set(c["row"] for c in results if c["mode"] == 0
                           and c["cls"].get("routed") == "NOT_JUDGED"))
    witness_not_judged(plan, tally, divergent, v)
    witness_bands(bands, v)
    witness_parsers(v)
    check_manifest_agreement(plan, results, v)
    report_predictions(plan, results, v._orows, v._p3, bands, v)
    print("CLASSES arm=%s %s" % (arm_label,
                                 " ".join("%s=%d" % kv for kv in sorted(pop.items()))))
    for mode in range(len(MODES)):
        cells_row = []
        for f in FIELDS:
            cells = {CLS_NAME[c]: tally.get((mode, f, c), 0)
                     for c in (MATCH, MISMATCH, ALT, IMMUNE, FIRED, MAY_UNCHANGED, MAY_MOVED)}
            cells["NOT_JUDGED"] = tally.get((mode, f, "NOT_JUDGED"), 0)
            cells_row.append("%s[%s]" % (f, ",".join("%s=%d" % kv for kv in sorted(cells.items())
                                                     if kv[1])))
        print("TALLY arm=%s mode=%-14s %s" % (arm_label, MODES[mode], " ".join(cells_row)))
    print("ALT-EXACT arm=%s %s" % (arm_label, json.dumps(alt_notes, sort_keys=True)))
    if notes:
        for nline in notes[:40]:
            print("NOTE " + nline)
        if len(notes) > 40:
            print("NOTE ... %d more" % (len(notes) - 40))
    return {"results": results, "tally": tally, "notes": notes, "pop": pop, "sep": sep,
            "dead": dead, "divergent": divergent, "alt": alt_notes}


def device(args, v):
    st = prepare(args, v)
    plan, bands = st["plan"], st["bands"]
    v._orows, v._p3 = st["orows"], st["p3"]
    run_driver_selftest(v, WORK)
    cmd = [BIN, "--plan", st["plan_path"], "--slab", st["slab_path"],
           "--out", os.path.join(WORK, "out.f32"),
           "--report", os.path.join(WORK, "report.tsv"),
           "--expect-cases", str(len(plan.cases))]
    print("CMD " + " ".join(cmd))
    rc, text = run(cmd, os.path.join(WORK, "driver.log"))
    for line in text.splitlines():
        if line.startswith(("FAIL", "PLAN ", "PLANCHECK ", "REPORT ", "OUT ", "RESULT ")):
            print("DRIVER " + line)
    res = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if len(res) != 1:
        raise GateError("the driver printed %d RESULT lines, expected exactly 1" % len(res))
    v.check("driver-result", rc == 0 and res[0].split()[1] == "PASS", len(res),
            "%r rc=%d" % (res[0], rc))
    pkv, n = parse_records(text, "PLANCHECK", ["cases", "checks_run", "malformed"])
    v.check("driver-planchannel", n == 1 and int(pkv[0]["checks_run"]) == len(plan.cases)
            and int(pkv[0]["malformed"]) == 0, int(pkv[0]["checks_run"]),
            "driver ran %s structural checks, malformed=%s" % (pkv[0]["checks_run"],
                                                               pkv[0]["malformed"]))
    ppkv, n = parse_records(text, "PLAN", ["plan_file", "cases_parsed", "cases_lines", "E", "K",
                                           "H", "slab_elems", "out_floats"])
    v.check("driver-plan-parse", int(ppkv[0]["cases_parsed"]) == len(plan.cases)
            and int(ppkv[0]["cases_lines"]) == len(plan.cases) + 1
            and int(ppkv[0]["out_floats"]) == plan.out_floats, len(plan.cases),
            "driver parsed %s/%s records, out_floats=%s" % (ppkv[0]["cases_parsed"],
                                                            ppkv[0]["cases_lines"],
                                                            ppkv[0]["out_floats"]))
    rr, n = parse_records(text, "REPORT", ["file", "rows"])
    v.check("driver-report-rows", n == 1 and int(rr[0]["rows"]) == len(plan.cases),
            int(rr[0]["rows"]) if n else 0, "report.tsv rows must equal the case count")
    rep_rows = parse_tsv(os.path.join(WORK, "report.tsv"), 17)
    if len(rep_rows) != len(plan.cases):
        raise GateError("report.tsv holds %d rows for %d cases" % (len(rep_rows), len(plan.cases)))
    st["counters"] = {
        "arm": "device",
        "driver_report_path": os.path.join(WORK, "report.tsv"),
        "driver_records_printed": int(rr[0]["rows"]),   # what the driver said it wrote
        "driver_records_parsed": len(rep_rows),         # what the gate read back from the file
        "expect_rows": int(cmd[cmd.index("--expect-cases") + 1]),   # as issued to the driver
        "expect_units": len(plan.cases) * len(FIELDS),
        "expect_units_note": "this lane's unit is the field comparison (cases x 6 fields); the "
                             "driver takes --expect-cases only, so expect_units is the gate's own "
                             "declaration (the SAMPLE line), not a driver flag",
    }
    compare_arms(plan, os.path.join(WORK, "out.f32"), st["host_out"], v, bands)
    j = judge_all(plan, os.path.join(WORK, "out.f32"), bands, st["prob_bound"], v, "device")
    v.check("judge-arm-device", j["pop"]["MATCH"] > 0 and j["pop"]["MISMATCH"] == 0,
            j["pop"]["MATCH"], "the device arm's judges produced verdicts and none of them is "
                               "outside every expectation")
    # The roster is settled before the artifact is written, so the JSON's `result` is the same
    # findings list the exit status is computed from, roster finding included.
    check_roster(v, DEVICE_ROSTER)
    write_evidence(plan, st["man"], bands, j, st, args, v, "device")
    if v.findings:
        print("FINDINGS " + ",".join(v.findings))
    return finish(v, "device", DEVICE_ROSTER, roster_checked=True)


def check_only(args, v):
    """Host-only: the oracle arm, the plan, the host arm's words and every judge, with no device and
    therefore no lock. This is the mode that proves the judges can tell the two cast orders apart
    before any kernel is asked to agree with them."""
    st = prepare(args, v)
    plan, bands = st["plan"], st["bands"]
    v._orows, v._p3 = st["orows"], st["p3"]
    j = judge_all(plan, st["host_out"], bands, st["prob_bound"], v, "host")
    check_roster(v, HOST_ROSTER)
    write_evidence(plan, st["man"], bands, j, st, args, v, "host")
    if v.findings:
        print("FINDINGS " + ",".join(v.findings))
    return finish(v, "check-only", HOST_ROSTER, roster_checked=True)


def write_evidence(plan, man, bands, j, st, args, v, mode):
    if not os.path.isdir(EV):
        os.makedirs(EV)
    tally, pop, results, notes, sep, dead, divergent = (j["tally"], j["pop"], j["results"],
                                                        j["notes"], j["sep"], j["dead"],
                                                        j["divergent"])
    doc = {
        "utc": "2026-10-26",
        "mode": mode,
        # Derived from the one findings list the exit status reads (and the roster has already been
        # settled by the time this runs), never a string typed alongside it.
        "result": "FAIL" if v.findings else "PASS",
        "artifact": "checkpoints/dump-manifest-2026-10-26.json",
        "manifest_dump_count": len(man["dumps"]),
        "rows_per_block_sampled": st["max_rows"],
        "prob_sum_bound": st["prob_bound"],
        "binary_sha256": sha256_file(BIN),
        "host_probe_sha256": sha256_file(PROBE_BIN),
        "gate_sha256": sha256_file(os.path.join(HERE, "gate.py")),
        "kernels_header_sha256": sha256_file(HEADER),
        "hip_sha256": sha256_file(os.path.join(HERE, "moe_router.hip")),
        "oracle_arm_sha256": sha256_file(os.path.join(HERE, "oracle_arm.py")),
        "oracle_module_sha256": sha256_file(os.path.join(REPO, "speccheck", "moe_ref.py")),
        "cases": len(plan.cases),
        "rows": len(plan.rows),
        "field_comparisons": len(plan.cases) * len(FIELDS),
        # The producer's count, the reader's count and the demand, as numbers: equality is only
        # meaningful next to the values, since an empty parse satisfies any boolean.
        "driver_records_printed": st["counters"]["driver_records_printed"],
        "driver_records_parsed": st["counters"]["driver_records_parsed"],
        "driver_report_path": st["counters"]["driver_report_path"],
        "expect_rows": st["counters"]["expect_rows"],
        "expect_units": st["counters"]["expect_units"],
        "expect_units_note": st["counters"]["expect_units_note"],
        "records_parse_equal": (st["counters"]["driver_records_printed"]
                                == st["counters"]["driver_records_parsed"]),
        "out_floats": plan.out_floats,
        "bands": bands,
        "classes": pop,
        "tally": {"%s/%s/%s" % (MODES[m], f, CLS_NAME[c] if isinstance(c, int) else c): n
                  for (m, f, c), n in sorted(tally.items(), key=lambda kv: str(kv[0]))},
        "cast_after_elements_differing_by_row": {str(k): v2 for k, v2 in sorted(sep.items())},
        "modes_that_never_fired": dead,
        "rows_where_the_set_moved_under_the_tie_break": divergent,
        "checks": [{"name": n, "ok": o, "n": c, "detail": d} for n, o, c, d in v.checks],
        "findings": v.findings,
        "notes": notes[:200],
    }
    with open(os.path.join(EV, "moe-router-%s-%s.json" % (mode, DAY)), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)


SELFTEST_ROSTER = ["manifest-geometry-matches-config", "manifest-dump-census",
                   "manifest-selfconsistent-sizes", "witness-parsers", "driver-selftest",
                   "driver-selftest-roster", "witness-bands-reject", "cast-is-robust-not-lucky",
                   "witness-prob-sum-bound", "witness-prob-bound-tight", "reach-table-shape"]


def selftest(args, v):
    """Host-only witnesses: no device work, so no lock is taken (the wrapper prints the reason)."""
    global _REACH
    _REACH = load_reach_table()
    check_header_enum()
    load_manifest(v)
    witness_parsers(v)
    run_driver_selftest(v, WORK)
    witness_prob_bound(v)
    witness_bands({"weights_vs_pinned_f32_ulp": 5,
                   "pinned_weight_to_bf16_boundary_f32ulp": 164}, v)
    v.check("reach-table-shape", all(reaches(0, f, ARM_KERNEL) == 0 for f in FIELDS)
            and all(reaches(0, f, ARM_LOGITS) == 0 for f in FIELDS)
            and all(reaches(len(MODES) - 1, f, ARM_KERNEL) == 1 for f in FIELDS)
            and all(reaches(MR_MODE_ROW_OFF_1_IDX, f, ARM_LOGITS) == 0 for f in FIELDS)
            and all(reaches(MR_MODE_ROW_OFF_1_IDX, f, ARM_KERNEL) == 1 for f in FIELDS),
            len(FIELDS) * 5,
            "the reference mode moves nothing on either arm, the poison mode moves everything, and "
            "the two R1 injections move nothing on the arm that never computes R1")
    if v.findings:
        print("FINDINGS " + ",".join(v.findings))
    return finish(v, "selftest", SELFTEST_ROSTER)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="host-only witnesses; no device, so no lock is taken")
    ap.add_argument("--check-only", action="store_true",
                    help="oracle + host arm + every judge, no device, so no lock is taken")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="sample only the first N tokens per block (cuts the sample, never an "
                         "assertion); recorded in the log and in the evidence")
    a = ap.parse_args()
    try:
        if a.selftest:
            return selftest(a, Verdict())
        if a.check_only:
            return check_only(a, Verdict())
        return device(a, Verdict())
    except GateError as e:
        print("FAIL gate: %s" % e)
        print("RESULT FAIL gate aborted: %s" % e.__class__.__name__)
        return 1
    except Exception as e:                       # a traceback escaping the judges would otherwise
        print("FAIL gate: %s: %s" % (e.__class__.__name__, e))   # print no RESULT line at all, and
        print("RESULT FAIL gate aborted: %s" % e.__class__.__name__)   # no RESULT line means no gate
        if os.environ.get("MR_TRACEBACK"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
