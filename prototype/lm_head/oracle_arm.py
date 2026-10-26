#!/usr/bin/env python3
"""oracle_arm.py — the reference answer for prototype/lm_head, and the arm that MATERIALISES the run.

    <venv>/bin/python oracle_arm.py --work WORK [--full-table] [--control NAME]

It is the third arm and the only one allowed to touch numpy/torch. The two drivers are C++; this file
shares NO code with them. Where the kernel decodes a Q8_0 block with its own f16 rebias, this file
decodes it with `tools/gguf_reader.py`'s audited reader; where the kernel sums K=hidden_size terms in
a written order, this file sums the SAME terms with `math.fsum` (exactly rounded) and then rounds once
to f32. A match is therefore a real agreement between three implementations, not two spellings of one
(`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the thing under
test").

WHAT IT READS (all written by gate.py, which owns the population):
  work/plan.tsv      the case/table contract both drivers also read (P/T/C records)
  work/sources.tsv   per table: which FILE and which ROWS the bytes come from, and the packing
  work/hidden.tsv    per hidden vector: dump row or engineered synthetic, with the derivation written
  work/ids.tsv       the gather index lists
WHAT IT WRITES:
  work/slab.bin      the weight bytes, at the bases the T records declare
  work/hidden.f32    the hidden vectors, f32 (bf16 dumps widened by tools/gguf_reader.bf16_to_f32_bytes)
  work/expect.f32    one f32 per word of the drivers' `out`, filled ONLY under the regions of the
                     mode==CORRECT cases; every region must be filled exactly once or this exits non-zero
  work/band.f64      the per-element derived band, gamma_index * sum_k |h_k||w_k|, where K = terms
                     actually summed and gamma_index is (K-1) for a family whose dot products are all
                     exactly representable in binary32 and (2K-1) for one the measurement finds inexact

WIDTH POLICY, and the refusal that goes with it: policy A (`quantization/README.md`, DECISIONS.md D-01a)
stores `lm_head` and the embeddings at Q8_0. The published GGUF stores `output.weight` as Q6_K — that
is the recorded D-10 deviation, which this file PRINTS from the artifact's own tensor records via
`tools/gguf_width_table.py` rather than assuming away. Q6_K kernel support is a later lane
(`tools/gguf_reader.py`'s audited `_deq_q6_k` is the reuse point). Until then a Q6_K tensor arriving as
the source of a Q8_0 table is a HARD REFUSAL, not a silent re-interpretation: `--control q6k-source`
exists to prove that refusal fires.

Correctness only: no timer, no warm loop, no rate. Nothing here touches the device or the GPU lock.
"""
import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np            # only for the band-form measurement below, never for the expectation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
import gguf_reader as GR                                            # audited reader/packer

CONFIG = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.config.json")
INDEX = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.safetensors.index.json")
SHARD_DIR = os.path.join(REPO, "models", "qwen36-src")
GGUF = os.path.join(REPO, "models", "unsloth-q4km", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DUMP_DIR = os.path.join(REPO, "data", "dumps", "2026-10-26")
PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"

ENC_NAME = {0: "BF16", 1: "Q8_0"}
Q8_BYTES, Q8_ELTS = 34, 32
U32 = 2.0 ** -24                      # binary32 unit roundoff


def gamma_of_index(n):
    """gamma_n = n*u/(1 - n*u), u = 2^-24: Higham's bound on the accumulated rounding of n f32 adds.

    This is deliberately the ORACLE'S OWN spelling of the rule. gate.py spells it too and
    floor_probe.py imports gate's; this file shares no code with either (that is what makes a match
    between the three an agreement rather than a tautology), so the rule is written down twice on
    purpose and the equality is checked in work/band.f64 by gate.verify_band_form().
    """
    if n < 1 or n * U32 >= 1.0:
        raise Refuse("gamma_%d is not a band factor (n >= 1 and n*u < 1 are both required)" % n)
    return n * U32 / (1.0 - n * U32)


def gamma_index(K, products_inexact):
    """The band index this family's own products license: K-1 roundings when a K-term dot has exactly
    representable products, 2K-1 when any product carries a rounding of its own.

    `products_inexact` is measured, never typed: it counts the dot terms h_k*w_k that do not survive a
    round trip through the binary32 grid, over every word this file bands for that family. A bf16 table
    multiplies 8 significand bits by 8 and is exact; a Q8_0 table's term is h_k*(d_k*q_k), an 8-bit
    hidden element times a 19-bit decoded value, and is not. User ratification 2026-10-26, recorded in
    `prototype/lm_head/evidence/band-form-gamma-index-2026-10-26.md`.
    """
    if products_inexact is None or int(products_inexact) < 0:
        raise Refuse("gamma_index(%d, %r): the index comes from a measurement, not a default"
                     % (K, products_inexact))
    return (K - 1) if int(products_inexact) == 0 else (2 * K - 1)

# The widths this lane is allowed to see as a table source. Anything else — and Q6_K above all — is
# refused by name below.
ALLOWED_SOURCE_DTYPES = ("BF16",)


class Refuse(Exception):
    """A refusal this file is REQUIRED to be able to make. Printed as REFUSE and exits non-zero."""


def contract_from_header():
    """The record field counts and the enum values, scraped out of the C++ the drivers compile.

    Nothing here is typed: if `LMH_F_C` or `LMH_MODE_HALF_ACCUM` changes in the header, this file sees
    the new number or dies saying so. A default in a gate driver and a default in the binary under test
    must move in one commit; scraping is how the driver side of that rule is enforced.
    """
    src = open(os.path.join(HERE, "lm_head_kernels.h")).read()
    plan = open(os.path.join(HERE, "lm_head_plan.h")).read()
    fcounts = {}
    for tag in ("LMH_F_P", "LMH_F_T", "LMH_F_C", "LMH_BLOCK", "LMH_SPLIT", "LMH_MAX_BATCH",
                "LMH_Q8_0_ELTS", "LMH_Q8_0_BYTES", "LMH_BF16_BYTES", "LMH_NAN_BITS"):
        m = re.search(r"#define\s+%s\s+(0x[0-9A-Fa-f]+|\d+)" % tag, plan + "\n" + src)
        if not m:
            raise Refuse("no #define %s with a numeric value found in the lane headers" % tag)
        fcounts[tag] = int(m.group(1), 0)
    enums = {}
    for body in re.findall(r"enum\s*[A-Za-z_]*\s*\{([^}]*)\}", src, re.S):
        for name, val in re.findall(r"\b(LMH_[A-Z0-9_]+)\s*=\s*(\d+)", body):
            enums[name] = int(val)
    for want in ("LMH_ENC_BF16", "LMH_ENC_Q8_0", "LMH_MODE_CORRECT", "LMH_MODE_COUNT",
                 "LMH_EXP_EXACT", "LMH_EXP_BAND", "LMH_EXP_FIRES", "LMH_EXP_IMMUNE",
                 "LMH_EXP_BYTES", "LMH_PATH_DENSE", "LMH_PATH_GATHER", "LMH_BATCH_BATCHED",
                 "LMH_BATCH_PERROW", "LMH_ASSOC_SEQ", "LMH_ASSOC_REV", "LMH_ASSOC_SPLIT8"):
        if want not in enums:
            raise Refuse("enum name %s not found in the lane headers" % want)
    return fcounts, enums


def artifact_widths(gguf_path):
    """(widths, n_tensors_declared, n_rows_parsed) for the published artifact.

    The stored type of every tensor, read from the file's own tensor-info table through
    tools/gguf_width_table.py's proven tokenizer and named by tools/gguf_header_types.py's TYPES map
    (the enum read out of ggml.h). This is the ONLY place the lane asks what the artifact actually
    stores, because policy A says Q8_0 and the file says otherwise for `output.weight`
    (`DECISIONS.md` D-10) — an assumption here would be the exact way Q6_K gets read as Q8_0.
    """
    if not os.path.exists(gguf_path):
        return {}, 0, 0
    import gguf_width_table as GWT
    import gguf_header_types as G
    try:
        _version, n_tensors, rows, _hend, _dstart = GWT.parse_gguf(gguf_path)
    except SystemExit as e:                            # parse_gguf raises SystemExit on a bad magic
        raise Refuse("cannot parse the published artifact %s: %s" % (gguf_path, e))
    widths = dict((name, G.TYPES.get(dt, "type_%d" % dt)) for name, _dims, dt in rows)
    return widths, int(n_tensors), len(rows)


def split_record(line, nfields, tag, ln, path):
    """Tab-separated, no collapsing, trailing newline stripped — the same rule lmh_split applies.
    A record that does not have EXACTLY the contracted field count is refused here as well as in the
    drivers, because a parser that reads 19 of 20 columns and defaults the 20th is the bug this whole
    section exists to prevent."""
    f = line.rstrip("\n").split("\t")
    if len(f) != nfields:
        raise Refuse("%s line %d: %s record has %d fields, the header contract says %d"
                     % (path, ln, tag, len(f), nfields))
    return f


def read_plan(path, fc, en):
    plan = {"P": None, "T": [], "C": []}
    n_written = 0
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            if line.startswith("#") or not line.strip():
                continue
            n_written += 1
            tag = line.split("\t", 1)[0]
            if tag == "P":
                f = split_record(line, fc["LMH_F_P"], "P", ln, path)
                plan["P"] = dict(head_dim=int(f[1]), h=int(f[2]), vocab=int(f[3]),
                                 n_hidden=int(f[4]), ids_count=int(f[5]), out_floats=int(f[6]),
                                 slab_bytes=int(f[7]), n_tables_decl=int(f[8]),
                                 n_cases_decl=int(f[9]))
            elif tag == "T":
                f = split_record(line, fc["LMH_F_T"], "T", ln, path)
                plan["T"].append(dict(idx=int(f[1]), enc=int(f[2]), role=int(f[3]), base=int(f[4]),
                                      nbytes=int(f[5]), v_rows=int(f[6]), h=int(f[7]),
                                      peer=int(f[8]), real=int(f[9])))
            elif tag == "C":
                f = split_record(line, fc["LMH_F_C"], "C", ln, path)
                plan["C"].append(dict(idx=int(f[1]), table=int(f[2]), path=int(f[3]),
                                      batching=int(f[4]), b0=int(f[5]), bcount=int(f[6]),
                                      row_lo=int(f[7]), nrows=int(f[8]), ids_off=int(f[9]),
                                      n_ids=int(f[10]), assoc=int(f[11]), mode=int(f[12]),
                                      out_off=int(f[13]), n_out=int(f[14]), cmp_off=int(f[15]),
                                      grp=int(f[16]), expect=int(f[17]), on_device=int(f[18]),
                                      trace=int(f[19])))
            else:
                raise Refuse("%s line %d: unknown record type %r" % (path, ln, tag))
    if plan["P"] is None:
        raise Refuse("%s holds no P record: a plan with no geometry adjudicates nothing" % path)
    if len(plan["T"]) != plan["P"]["n_tables_decl"]:
        raise Refuse("%s declares %d tables, holds %d" % (path, plan["P"]["n_tables_decl"],
                                                           len(plan["T"])))
    if len(plan["C"]) != plan["P"]["n_cases_decl"]:
        raise Refuse("%s declares %d cases, holds %d" % (path, plan["P"]["n_cases_decl"],
                                                          len(plan["C"])))
    print("ORACLEPLAN file=%s records=%d p=1 tables=%d cases=%d h=%d head_dim=%d vocab=%d "
          "n_hidden=%d ids=%d out_floats=%d slab_bytes=%d"
          % (path, n_written, len(plan["T"]), len(plan["C"]), plan["P"]["h"], plan["P"]["head_dim"],
             plan["P"]["vocab"], plan["P"]["n_hidden"], plan["P"]["ids_count"],
             plan["P"]["out_floats"], plan["P"]["slab_bytes"]))
    return plan


def read_kv_lines(path):
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            tag, d = f[0], {}
            for tok in f[1:]:
                k, _, v = tok.partition("=")
                d[k] = v
            out.append((tag, d))
    return out


def q8_row_bytes(h):
    return (h // Q8_ELTS) * Q8_BYTES


def pack_q8_0(vals):
    """bf16-widened f32 values -> Q8_0 storage bytes, from the FORMAT.

    d = f16(amax/127), q = clamp(round-half-away(v/d), -127, 127), 34 bytes per 32 elements — the
    documented ggml Q8_0 packing. This function is NOT trusted by anything: the caller requires that
    feeding these bytes back through `gguf_reader.dequant(..., "Q8_0")` reproduces, element for
    element, what the AUDITED packer `gguf_reader.quant_ref_deq("Q8_0", ...)` produces. An off-by-one
    here therefore fails the run rather than quietly becoming the kernel's new reference.
    """
    if len(vals) % Q8_ELTS:
        raise Refuse("pack_q8_0 needs a whole number of 32-element blocks, got %d values" % len(vals))
    out = bytearray()
    for i in range(len(vals) // Q8_ELTS):
        blk = vals[i * Q8_ELTS:(i + 1) * Q8_ELTS]
        amax = max(abs(v) for v in blk)
        d = struct.unpack("<e", struct.pack("<e", amax / 127.0))[0]
        out += struct.pack("<e", d)
        if d == 0.0:
            out += bytes(Q8_ELTS)
            continue
        qs = []
        for v in blk:
            x = v / d
            q = math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)   # C nearest_int(): half away
            qs.append(max(-127, min(127, q)))
        out += struct.pack("<32b", *qs)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--full-table", action="store_true",
                    help="also run the real-size (full-vocab) amortisation witness on the CPU")
    ap.add_argument("--control", default=None,
                    help="negative-control mode for gate.py's selftest: q6k-source")
    ap.add_argument("--gamma-form", default="", choices=("", "wide", "narrow"),
                    help="WITNESS ONLY: price the band with a gamma index the measurement did not "
                         "select, so the checks that refuse such a band can be shown refusing.")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--gguf", default=GGUF)
    a = ap.parse_args()
    work = a.work

    fc, en = contract_from_header()
    print("ORACLECONTRACT fields P=%d T=%d C=%d modes=%d exp=%d q8=%dB/%d elts"
          % (fc["LMH_F_P"], fc["LMH_F_T"], fc["LMH_F_C"],
             en["LMH_MODE_COUNT"], en["LMH_EXP_COUNT"] if "LMH_EXP_COUNT" in en else 5,
             fc["LMH_Q8_0_BYTES"], fc["LMH_Q8_0_ELTS"]))
    plan = read_plan(os.path.join(work, "plan.tsv"), fc, en)
    P = plan["P"]
    tables, cases = plan["T"], plan["C"]

    # ---- the width the artifact actually stores, read from the file and cited, never assumed -------
    widths, n_declared, n_parsed = artifact_widths(a.gguf)
    for nm in ("output.weight", "token_embd.weight"):
        print("ARTIFACTWIDTH tensor=%s stored=%s policy_a=Q8_0 note=D-10-deviation"
              % (nm, widths.get(nm, "absent-from-artifact")))
    print("ARTIFACTFILE file=%s declared_tensors=%d parsed_rows=%d"
          % (os.path.basename(a.gguf), n_declared, n_parsed))
    if a.control == "q6k-source":
        # The refusal must be reachable, so the control points a Q8_0 table at the Q6_K tensor.
        q6 = [n for n, t in widths.items() if t == "Q6_K"]
        if not q6:
            print("REFUSE-CONTROL-SETUP no Q6_K tensor in %s: the control cannot fire" % a.gguf)
            return 1
        print("REFUSE_CONTROL injecting Q6_K tensor %s as the source of a Q8_0 table" % q6[0])

    # ---- materialise the tables -------------------------------------------------------------------
    slab = bytearray(P["slab_bytes"])
    for i in range(len(slab)):
        slab[i] = 0xCD                                        # LMH_PAD_BYTE: gaps are not table bytes
    src_rows = read_kv_lines(os.path.join(work, "sources.tsv"))
    by_idx = {}
    for tag, d in src_rows:
        if tag != "T":
            raise Refuse("sources.tsv: unknown record %r" % tag)
        by_idx[int(d["idx"])] = d
    if len(by_idx) != len(tables):
        raise Refuse("sources.tsv names %d tables, the plan has %d (a table with no source is a table "
                     "of zeros)" % (len(by_idx), len(tables)))

    import numpy as np
    n_bytes_written = 0
    n_q8_blocks = 0
    shard_witnessed = set()
    for t in tables:
        d = by_idx[t["idx"]]
        want_enc = ENC_NAME[t["enc"]]
        if d["enc"] != want_enc:
            raise Refuse("table %d: plan says enc=%s, sources.tsv says %s"
                         % (t["idx"], want_enc, d["enc"]))
        if int(d["real"]) != t["real"]:
            raise Refuse("table %d: plan says real=%d, sources.tsv says %s"
                         % (t["idx"], t["real"], d["real"]))
        rows = [int(x) for x in d["rows"].split(",")] if d.get("rows") else []
        if t["enc"] == en["LMH_ENC_Q8_0"]:
            row_b = q8_row_bytes(t["h"])
        else:
            row_b = t["h"] * fc["LMH_BF16_BYTES"]
        # The rows-times-row-width identity is a statement about a FILE read. A tied_copy has no rows of
        # its own (it copies a byte range another table already read; its own geometry check is the
        # nbytes == nbytes-copied comparison in that branch) and a synth table carries a byte payload,
        # so requiring `rows` of them would demand 0 rows x 4096 B and refuse a legal plan.
        if d["kind"] == "real" and t["real"] and t["nbytes"] != len(rows) * row_b:
            raise Refuse("table %d: nbytes=%d is not %d rows x %d B/row"
                         % (t["idx"], t["nbytes"], len(rows), row_b))
        base = t["base"]
        if base + t["nbytes"] > len(slab):
            raise Refuse("table %d [%d,%d) does not fit the slab (%d B)"
                         % (t["idx"], base, base + t["nbytes"], len(slab)))
        if d["kind"] == "synth":
            payload = bytes.fromhex(d["bytes"])
            if len(payload) != t["nbytes"]:
                raise Refuse("synthetic table %d: %d bytes handed over, plan wants %d"
                             % (t["idx"], len(payload), t["nbytes"]))
            slab[base:base + t["nbytes"]] = payload
            n_bytes_written += len(payload)
            # POPULATION WITNESSES for a handed-over payload. (a) No row may decode to all zeros: an
            # all-zero synthetic table makes every case over it a zero-output case, which is how a bf16
            # image written from the wrong half of the f32 bits once shipped here — the value test on
            # the way in passed, the bytes were still zeros. (b) The rows declared `dup=` must really
            # be byte-identical, because the IMMUNE arms over this table rest on "two rows, one value".
            zero_rows = [r for r in range(t["v_rows"])
                         if not any(x != 0.0 for x in decode_row(slab, t, r, t["h"], fc, en))]
            if zero_rows:
                raise Refuse("synthetic table %d: rows %s decode to all zeros. A case over a zero row "
                             "agrees with anything; it is a hollow check, not a cheap one."
                             % (t["idx"], zero_rows[:8]))
            dup = [int(x) for x in d["dup"].split(",")] if d.get("dup") else []
            if dup:
                imgs = dict((r, bytes(slab[base + r * row_b:base + (r + 1) * row_b])) for r in dup)
                if len(set(imgs.values())) != 1:
                    raise Refuse("synthetic table %d declares dup=%s but their bytes differ"
                                 % (t["idx"], dup))
            print("SYNTHWITNESS table=%d rows=%d all_rows_nonzero=1 dup=%s dup_identical=%d"
                  % (t["idx"], t["v_rows"], dup or "none", 1 if dup else 0))
            if t["enc"] == en["LMH_ENC_Q8_0"]:
                n_q8_blocks += t["nbytes"] // Q8_BYTES
            print("TABLE idx=%d kind=synth enc=%s bytes=%d rows=%d base=%d"
                  % (t["idx"], want_enc, t["nbytes"], len(rows), base))
            continue
        if d["kind"] == "tied_copy":
            src_base = int(d["from_base"])
            n = int(d["nbytes"])
            if n != t["nbytes"]:
                raise Refuse("tied control table %d: copies %d B but declares %d B"
                             % (t["idx"], n, t["nbytes"]))
            slab[base:base + n] = slab[src_base:src_base + n]
            n_bytes_written += n
            print("TABLE idx=%d kind=tied_copy from_base=%d bytes=%d base=%d (ONE buffer, two roles)"
                  % (t["idx"], src_base, n, base))
            continue
        # a real safetensors read: the file, the tensor and the rows are named, never inferred
        shard = d["file"]
        tname = d["tensor"]
        if not os.path.exists(shard):
            raise Refuse("table %d: source shard %s does not exist" % (t["idx"], shard))
        hdr, header_len = read_safetensors_header(shard)
        if shard not in shard_witnessed:
            # WITNESS for header_len, not a trust: safetensors tiles every tensor from the data start,
            # so the largest data_offset in the file must end exactly at EOF. An 8-byte slip in
            # read_safetensors_header (returning the JSON length instead of 8 + it) makes this false by
            # 8 for every shard — which is how a wrong offset becomes a refusal here instead of quietly
            # becoming the reference the whole lane is judged against.
            hi_max = max([v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"])
            sz = os.path.getsize(shard)
            if header_len + hi_max != sz:
                raise Refuse("%s: header_len=%d + max data_offset=%d = %d, but the file is %d B. The "
                             "data section must end at EOF; the header length is wrong."
                             % (os.path.basename(shard), header_len, hi_max,
                                header_len + hi_max, sz))
            print("SHARDHDR file=%s header_len=%d data_bytes=%d tensors=%d identity=header_plus_data_eq_filesize"
                  % (os.path.basename(shard), header_len, hi_max, len(hdr) - 1))
            shard_witnessed.add(shard)
        if tname not in hdr:
            raise Refuse("table %d: %s is not in %s" % (t["idx"], tname, shard))
        rec = hdr[tname]
        dtype = rec["dtype"]
        if dtype not in ALLOWED_SOURCE_DTYPES:
            # THE Q6_K REFUSAL. A tensor stored at a width this lane has no kernel for is refused by
            # name rather than reinterpreted: the byte geometry of Q6_K (210 B / 256 elts) is not
            # Q8_0's (34 B / 32 elts), so accepting it "as if Q8_0" would either fail the plan's own
            # geometry check or, worse, read 1680 bytes where 2176 are needed.
            raise Refuse("table %d: source %s is stored %s in %s. This lane has kernels for BF16 and "
                         "Q8_0 only (policy A / D-01a); Q6_K support is deferred and this is the D-10 "
                         "deviation, not a licence to reinterpret the bytes."
                         % (t["idx"], tname, dtype, shard))
        if a.control == "q6k-source" and d["enc"] == "Q8_0":
            d["file"] = a.gguf                                # the control's tampering is printed
            raise Refuse("table %d: sources.tsv points a Q8_0 table at %s, which stores output.weight "
                         "as Q6_K (artifact widths printed above). Q6_K is not a Q8_0 source; refusing "
                         "to read it as one." % (t["idx"], a.gguf))
        lo, hi = rec["data_offsets"]
        shape = rec["shape"]
        if len(shape) != 2 or shape[1] != t["h"]:
            raise Refuse("table %d: %s is %s, not [rows, %d]" % (t["idx"], tname, shape, t["h"]))
        if (hi - lo) != shape[0] * shape[1] * 2:
            raise Refuse("table %d: %s spans %d B, shape %s x bf16 says %d"
                         % (t["idx"], tname, hi - lo, shape, shape[0] * shape[1] * 2))
        mm = np.memmap(shard, dtype=np.uint16, mode="r", offset=header_len + lo,
                       shape=(shape[0], shape[1]))
        payload = bytearray()
        for r in rows:
            if not 0 <= r < shape[0]:
                raise Refuse("table %d: row %d is outside %s's %d rows" % (t["idx"], r, tname, shape[0]))
            raw = np.asarray(mm[r], dtype=np.uint16)
            f32 = list(GR.bf16_to_floats(raw.tobytes()))       # the audited bf16 -> f32 widening
            if t["enc"] == en["LMH_ENC_Q8_0"]:
                blk = pack_q8_0(f32)
                n_q8_blocks += len(blk) // Q8_BYTES
                # the audited pair-agreement: these bytes must dequantize to what the audited packer
                # produces from the same source values, or the packer above is wrong
                got = GR.dequant(blk, "Q8_0")
                want = GR.quant_ref_deq("Q8_0", f32)
                if [round(g, 12) for g in got] != [round(w, 12) for w in want]:
                    bad = next(i for i in range(len(got)) if abs(got[i] - want[i]) > 1e-12)
                    raise Refuse("table %d row %d: pack_q8_0 disagrees with tools/gguf_reader at "
                                 "element %d (%r vs %r)" % (t["idx"], r, bad, got[bad], want[bad]))
                payload += blk
            else:
                payload += raw.tobytes()                        # the BIT IMAGE, not a re-quantised copy
        if len(payload) != t["nbytes"]:
            raise Refuse("table %d: read %d B of rows, plan declares %d"
                         % (t["idx"], len(payload), t["nbytes"]))
        slab[base:base + t["nbytes"]] = payload
        n_bytes_written += len(payload)
        print("TABLE idx=%d kind=real enc=%s tensor=%s shard=%s rows=%d bytes=%d base=%d "
              "stored_dtype=%s"
              % (t["idx"], want_enc, tname, os.path.basename(shard), len(rows), t["nbytes"], base,
                 dtype))
    with open(os.path.join(work, "slab.bin"), "wb") as fh:
        fh.write(bytes(slab))
    print("SLAB file=slab.bin bytes=%d pad=0xCD tables=%d q8_blocks=%d table_bytes=%d"
          % (len(slab), len(tables), n_q8_blocks, n_bytes_written))
    if len(slab) != P["slab_bytes"]:
        raise Refuse("slab is %d B, the P record says %d" % (len(slab), P["slab_bytes"]))

    # ---- the hidden vectors -----------------------------------------------------------------------
    spec = read_kv_lines(os.path.join(work, "hidden.tsv"))
    H, hd = P["h"], P["head_dim"]
    hidden = [None] * P["n_hidden"]
    n_real = n_synth = 0
    for tag, d in spec:
        if tag != "H":
            raise Refuse("hidden.tsv: unknown record %r" % tag)
        i = int(d["i"])
        if not 0 <= i < P["n_hidden"]:
            raise Refuse("hidden.tsv: index %d outside n_hidden=%d" % (i, P["n_hidden"]))
        if hidden[i] is not None:
            raise Refuse("hidden.tsv: index %d specified twice" % i)
        if d["src"] == "dump":
            path = os.path.join(DUMP_DIR, d["file"])
            sz = os.path.getsize(path)
            want = int(d["n_rows"]) * H * 2
            if sz != want:
                raise Refuse("%s holds %d B, %d rows x %d x bf16 says %d"
                             % (path, sz, int(d["n_rows"]), H, want))
            r = int(d["row"])
            raw = open(path, "rb").read()
            row = raw[r * H * 2:(r + 1) * H * 2]
            hidden[i] = list(GR.bf16_to_floats(row))           # bits << 16, exact — the audited route
            n_real += 1
            print("HIDDEN i=%d src=dump file=%s row=%d widened=bits<<16" % (i, d["file"], r))
        else:
            kind = d["kind"]
            v = [0.0] * H
            if kind == "ones":
                v = [1.0] * H
            elif kind == "onehot":
                v[int(d["k"])] = 1.0
            elif kind == "onehot_inexact":
                v[int(d["k"])] = 1.001953125                   # NOT a bf16 value: HALF_ACCUM's fuel
            elif kind == "head_only":
                v = [1.0] * hd + [0.0] * (H - hd)
            elif kind == "tail_zero":
                v = [1.0] * (H - 1) + [0.0]
            elif kind == "head_zero":
                v = [0.0] + [1.0] * (H - 1)
            elif kind == "zeros":
                pass
            else:
                raise Refuse("hidden.tsv: unknown synthetic kind %r" % kind)
            hidden[i] = v
            n_synth += 1
            print("HIDDEN i=%d src=synth kind=%s" % (i, kind))
    if any(v is None for v in hidden):
        raise Refuse("hidden.tsv leaves %d of %d hidden vectors unspecified"
                     % (sum(1 for v in hidden if v is None), P["n_hidden"]))
    with open(os.path.join(work, "hidden.f32"), "wb") as fh:
        fh.write(struct.pack("<%df" % (len(hidden) * H), *[x for row in hidden for x in row]))
    print("HIDDEN vectors=%d real=%d synth=%d h=%d head_dim=%d"
          % (len(hidden), n_real, n_synth, H, hd))
    if len(hidden) != P["n_hidden"]:
        raise Refuse("hidden.tsv specifies %d vectors, P says n_hidden=%d"
                     % (len(hidden), P["n_hidden"]))

    # ---- the reference logits, region by region ----------------------------------------------------
    ids = array_from_file(os.path.join(work, "ids.tsv"), P["ids_count"])
    out_floats = P["out_floats"]
    expect = [0.0] * out_floats
    band = [float("nan")] * out_floats
    filled = [0] * out_floats
    n_fill = 0
    owner = [None] * out_floats
    band_seen = set()
    band_deriv = []
    BAND_PRINT = 16
    # The band is NOT computed inside this loop. A family's gamma index is a property of the whole
    # family — one inexact product anywhere in it licenses gamma_{2K-1} for all of it — so the index
    # cannot be known until every word has been measured. This loop measures; the pass after it prices.
    word_sig = []
    for c in cases:
        if c["mode"] != en["LMH_MODE_CORRECT"]:
            continue                                     # injections are judged against their honest twin
        t = tables[c["table"]]
        h_used = t["h"]                       # an honest case never takes the wrong stride: mode is CORRECT
        rows = ([c["row_lo"] + j for j in range(c["nrows"])] if c["path"] == en["LMH_PATH_DENSE"]
                else [ids[c["ids_off"] + j] for j in range(c["n_ids"])])
        slot = H if c["mode"] == en["LMH_MODE_BATCH_SLOT_STRIDE"] else c["n_out"]
        for b in range(c["bcount"]):
            if not 0 <= c["b0"] + b < len(hidden):
                raise Refuse("case %d: batch row %d addresses hidden[%d] but the plan has %d "
                             "vectors — an out-of-range hidden index, not a wrong number"
                             % (c["idx"], b, c["b0"] + b, len(hidden)))
            hv = hidden[c["b0"] + b]
            for jj, r in enumerate(rows):
                vals = decode_row(slab, t, r, h_used, fc, en)
                terms = [hv[k] * vals[k] for k in range(h_used)]
                exact = struct.unpack("<f", struct.pack("<f", math.fsum(terms)))[0]
                at = c["out_off"] + b * slot + jj
                if at >= out_floats:
                    raise Refuse("case %d: word %d is outside out_floats=%d" % (c["idx"], at, out_floats))
                if filled[at]:
                    raise Refuse("expectation word %d is filled by two cases (case %d and an earlier "
                                 "one): the layout is ambiguous and nothing here can be trusted"
                                 % (at, c["idx"]))
                expect[at] = exact
                K = h_used
                sigma = math.fsum(abs(x) for x in terms)
                # Is each product exactly a binary32 number? The products are formed here in f64 from
                # two values that are each exactly f32, so the comparison below is a measurement of the
                # term, not a round trip through the ruler being tested.
                t64 = np.asarray(terms, dtype=np.float64)
                inexact_here = int(np.count_nonzero(
                    t64.astype(np.float32).astype(np.float64) != t64))
                word_sig.append(dict(table=c["table"], K=K, at=at, sigma=sigma, inexact=inexact_here))
                filled[at] = 1
                owner[at] = c["idx"]
                if c["idx"] not in band_seen:
                    band_seen.add(c["idx"])
                    band_deriv.append(dict(idx=c["idx"], K=K, sigma=sigma, band=None, exact=exact,
                                           table=c["table"], path=c["path"], assoc=c["assoc"],
                                           mode=c["mode"], b0=c["b0"], bcount=c["bcount"],
                                           h_used=h_used, word=at))
                n_fill += 1
    if n_fill == 0:
        raise Refuse("the oracle filled no expectation words: there is nothing to judge against")

    # ---- THE BAND FORM: priced per family by what that family's products measurably did ------------
    forms = {}
    for w in word_sig:
        f = forms.setdefault(w["table"], dict(K=w["K"], words=0, terms=0, inexact=0))
        if f["K"] != w["K"]:
            raise Refuse("table %d is summed with %d terms in one honest case and %d in another"
                         % (w["table"], f["K"], w["K"]))
        f["words"] += 1
        f["terms"] += w["K"]
        f["inexact"] += w["inexact"]
    if len(word_sig) != n_fill:
        raise Refuse("measured %d words but filled %d expectations: the band would be priced over a "
                     "different population than the answer"
                     % (len(word_sig), n_fill))
    for ti in sorted(forms):
        f = forms[ti]
        f["gidx"] = gamma_index(f["K"], f["inexact"])
        f["gamma"] = gamma_of_index(f["gidx"])
    if a.gamma_form:
        # WITNESS ONLY, reached by `--gamma-form` (gate.py forwards it under --selftest). Overwrite the
        # licensed index with one the measurement did not select, so the checks that refuse a form nobody
        # measured can be SHOWN refusing. Nothing else in the repo may set this; the band that comes out
        # of a forced run is by construction not the band the bytes license.
        for ti in sorted(forms):
            f = forms[ti]
            if a.gamma_form == "wide":
                f["gidx"] = 2 * f["K"] - 1
            elif a.gamma_form == "narrow":
                f["gidx"] = f["K"] - 1
            else:
                raise Refuse("--gamma-form wants wide|narrow, got %r" % (a.gamma_form,))
            f["gamma"] = gamma_of_index(f["gidx"])
            print("FORCEDFORM table=%d measured_products_inexact=%d licensed_without_force=%d "
                  "forced_gamma_index=%d forced_gamma=%.9g "
                  "note=forced_by_--gamma-form_not_by_measurement"
                  % (ti, f["inexact"], gamma_index(f["K"], f["inexact"]), f["gidx"], f["gamma"]))
    print("BANDCENSUS tag=oracle families=%d words_banded=%d term_pairs=%d products_inexact=%d "
          "u=2^-24 rule=gamma_index(K,products_inexact)=(K-1 if inexact==0 else 2K-1)"
          % (len(forms), len(word_sig), sum(f["terms"] for f in forms.values()),
             sum(f["inexact"] for f in forms.values())))
    fam_lines = []
    for ti in sorted(forms):
        f = forms[ti]
        fam_lines.append("BANDFAMILY tag=oracle table=%d enc=%d K=%d words=%d term_pairs=%d "
                         "products_inexact=%d gamma_index=%d gamma=%.9g"
                         % (ti, tables[ti]["enc"], f["K"], f["words"], f["terms"], f["inexact"],
                            f["gidx"], f["gamma"]))
    for l in fam_lines:
        print(l)
    if len(fam_lines) != len(forms):
        raise Refuse("printed %d BANDFAMILY records for %d families: the print is the derivation, so "
                     "a record that did not print is a band nobody can recompute" % (len(fam_lines),
                                                                                    len(forms)))
    for w in word_sig:
        band[w["at"]] = forms[w["table"]]["gamma"] * w["sigma"]
    for b in band_deriv:
        b["gamma"] = forms[b["table"]]["gamma"]
        b["gidx"] = forms[b["table"]]["gidx"]
        b["band"] = band[b["word"]]
    with open(os.path.join(work, "expect.f32"), "wb") as fh:
        fh.write(struct.pack("<%df" % out_floats, *expect))
    with open(os.path.join(work, "band.f64"), "wb") as fh:
        fh.write(struct.pack("<%dd" % out_floats, *band))
    n_band = sum(1 for i in range(out_floats) if filled[i] and not (band[i] > 0.0))
    print("ORACLEFILL regions_filled=%d words=%d words_total=%d band_nonzero=%d band_nonpositive=%d "
          "gamma_index_by_table=%s"
          % (n_fill, n_fill, out_floats, n_fill - n_band, n_band,
             ",".join("%d:%d" % (ti, forms[ti]["gidx"]) for ti in sorted(forms))))
    # THE DERIVATION, printed. A band printed as a bare number is a number the reader has to trust;
    # these lines carry K, the licensed gamma index and its value n*u/(1-n*u) with u = 2^-24, the sum
    # of |h_k||w_k|, the product, and the exact f32 expectation, so an independent reader can recompute
    # the tolerance from the print alone.
    print("BANDFORMULA u=2^-24 gamma_n=n*u/(1-n*u) band=gamma_index*fsum(|h_k||w_k|) "
          "gamma_index=(K-1 if the family's products are all exact else 2K-1) K=h_used terms=h_used")
    for b in band_deriv[:BAND_PRINT]:
        print("BAND idx=%d K=%d gamma_index=%d gamma=%.9g sigma=%.9g band=%.9g exact=%.9g table=%d "
              "path=%d assoc=%d mode=%d b0=%d bcount=%d word=%d"
              % (b["idx"], b["K"], b["gidx"], b["gamma"], b["sigma"], b["band"], b["exact"],
                 b["table"], b["path"], b["assoc"], b["mode"], b["b0"], b["bcount"], b["word"]))
    print("BANDSUMMARY cases_with_honest_words=%d printed=%d"
          % (len(band_seen), min(len(band_seen), BAND_PRINT)))
    for i in [w for w in range(out_floats) if filled[w] and not (band[w] > 0.0)][:20]:
        print("BANDZERO word=%d case=%d exact=%.9g band=%.9g note=zero-band-word"
              % (i, owner[i], expect[i], band[i]))
    if n_band:
        raise Refuse("%d expectation words have a non-positive band: the band is vacuous and a "
                     "must-fire injection could never exceed it (case list above)" % n_band)

    if a.full_table:
        full_table_witness(P, tables, hidden, fc, en)

    print("RESULT PASS oracle arm filled=%d/%d words slab=%d B hidden=%d"
          % (n_fill, out_floats, len(slab), len(hidden)))
    return 0


def read_safetensors_header(shard):
    """(header dict, byte offset where the data section begins). The 8 comes from the format: the file
    opens with a little-endian u64 tensor-info-JSON length, then that JSON, then the tensors. Returning
    `n` instead of `8 + n` shifts every read 8 B early, which is eight bytes of the header's own JSON
    read as bf16 weights — the reason this function's return is the second value and not a bare dict."""
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def array_from_file(path, want):
    """ids.tsv is `I\t<off>\tvalue=<int>` lines: the gather indices as a FILE, so the drivers, the
    oracle and gate.py all read the same integers rather than each generating them."""
    vals = [None] * want
    for tag, d in read_kv_lines(path):
        if tag != "I":
            raise Refuse("ids.tsv: unknown record %r" % tag)
        off = int(d["off"])
        if not 0 <= off < want:
            raise Refuse("ids.tsv: off %d outside ids_count=%d" % (off, want))
        if vals[off] is not None:
            raise Refuse("ids.tsv: off %d given twice" % off)
        vals[off] = int(d["value"])
    if any(v is None for v in vals):
        raise Refuse("ids.tsv leaves %d of %d indices unspecified"
                     % (sum(1 for v in vals if v is None), want))
    return vals


def decode_row(slab, t, row, h_used, fc, en):
    """One row's decoded values, through the AUDITED reader only, over the exact byte range the row's
    own addressing says. Returns h_used floats."""
    if t["enc"] == en["LMH_ENC_Q8_0"]:
        row_b = (h_used // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"]
        at = t["base"] + row * row_b
        buf = bytes(slab[at:at + row_b])
        vals = GR.dequant(buf, "Q8_0", h_used)
    else:
        row_b = h_used * fc["LMH_BF16_BYTES"]
        at = t["base"] + row * row_b
        buf = bytes(slab[at:at + row_b])
        vals = GR.bf16_to_floats(buf)
    if len(vals) != h_used:
        raise Refuse("decoded %d values for row %d, expected %d" % (len(vals), row, h_used))
    return vals


def full_table_witness(P, tables, hidden, fc, en):
    """Prediction 5 at the size the prediction is about: the whole untied projection table.

    This is an ARITHMETIC-AND-BYTES witness on the CPU, printed and never band-judged: it says what one
    token of `lm_head` costs and what batching does to that cost at vocab_size = 248320, from the shard
    header's own byte span. The kernel's own read-set proof lives in gate.py at the sampled row count;
    nothing claimed here is a claim about the device.
    """
    import numpy as np
    lmh = [t for t in tables if t["enc"] == en["LMH_ENC_BF16"] and t["real"] == 1]
    if not lmh:
        raise Refuse("--full-table found no real bf16 table to price")
    idx = json.load(open(INDEX))
    name = "lm_head.weight"
    shard = os.path.join(SHARD_DIR, idx["weight_map"][name])
    hdr, header_len = read_safetensors_header(shard)
    lo, hi = hdr[name]["data_offsets"]
    rows, h = hdr[name]["shape"]
    bytes_per_token = (hi - lo)
    print("FULLTABLE tensor=%s shard=%s rows=%d h=%d bytes=%d per_token=%d header_shape=%s"
          % (name, os.path.basename(shard), rows, h, bytes_per_token, bytes_per_token,
             hdr[name]["shape"]))
    if bytes_per_token != rows * h * 2:
        raise Refuse("lm_head.weight spans %d B, [rows,h] x bf16 says %d"
                     % (bytes_per_token, rows * h * 2))
    B = min(4, len(hidden))
    X = np.asarray(hidden[:B], dtype=np.float32)                # [B, h]
    mm = np.memmap(shard, dtype=np.uint16, mode="r", offset=header_len + lo, shape=(rows, h))
    # batched: ONE pass over the table for all B hidden vectors. per-row: B passes.
    batched = np.zeros((B, rows), dtype=np.float32)
    CH = 4096                                                    # rows per chunk, memory-bounded
    for s in range(0, rows, CH):
        e = min(rows, s + CH)
        W = np.asarray(mm[s:e], dtype=np.uint16)
        Wf = np.zeros((e - s, h), dtype=np.float32)
        Wf.view(np.uint32)[:, :] = (W.astype(np.uint32)[:, :] << 16)     # bits << 16, exact
        batched[:, s:e] = (Wf @ X.T).T
    passes_batched = 1
    per_row = np.zeros((B, rows), dtype=np.float32)
    for b in range(B):
        for s in range(0, rows, CH):
            e = min(rows, s + CH)
            W = np.asarray(mm[s:e], dtype=np.uint16)
            Wf = np.zeros((e - s, h), dtype=np.float32)
            Wf.view(np.uint32)[:, :] = (W.astype(np.uint32)[:, :] << 16)
            per_row[b, s:e] = Wf @ X[b]
    passes_perrow = B
    apart = int(np.count_nonzero(batched.view(np.uint32) != per_row.view(np.uint32)))
    print("FULLTABLE amortise batch_rows=%d bytes_batched=%d bytes_perrow=%d passes_batched=%d "
          "passes_perrow=%d words=%d words_apart=%d"
          % (B, bytes_per_token * passes_batched, bytes_per_token * passes_perrow, passes_batched,
             passes_perrow, batched.size, apart))
    print("FULLTABLE note=CPU arithmetic and byte counts from the shard header; no device, no timing, "
          "and no claim about the kernel (its read-set proof is gate.py's)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as e:
        print("REFUSE %s" % e)
        sys.exit(1)
    except Exception as e:                                     # a crash is a refusal too
        print("REFUSE oracle arm crashed: %r" % (e,))
        import traceback
        traceback.print_exc()
        sys.exit(1)
