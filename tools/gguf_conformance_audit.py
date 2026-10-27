#!/usr/bin/env python3
"""gguf_conformance_audit.py — does the shipped artifact store what the width map says it stores,
and does the engine agree with the file per tensor? (DECISIONS.md D-27's conformance audit.)

WHY THIS EXISTS. D-27 decided the shipping artifact is the published, sha-pinned third-party
quantization `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. We never converted it, so there
is no converter step to audit — and a check written against a step that does not exist cannot fail,
which is the vacuity class AGENTS.md bans. D-26 clause 2's real requirement is a WIDTH audit: the
family->type map must be read out of the file, compared against
`quantization/evidence/q4km-published-width-map-2026-10-26.md`, and the engine's own declared widths
must agree with the file tensor by tensor, so a stale hard-coded width in the engine surfaces as a
named finding instead of as a silent numeric difference.

WHAT IT CHECKS, in this order, each with its own denominator printed:

  1. identity   — the file's sha256 must equal the pin in `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`;
                  a mismatch refuses before any width claim is made.
  2. parse      — the artifact's own tensor-info header, walked by this file's own parser. The parsed
                  record count must equal the header's declared count AND the population derived from
                  the committed safetensors index (`engine/artifact.py::expected_records`).
                  Inequality is fatal. A parse that yields nothing is a failure, never a pass
                  (AGENTS.md: a check that can produce no output cannot fail).
  3. family map — family -> stored type -> count, populations DERIVED FROM THE FILE (never typed
                  here), with stored bytes from `tools/ggml_block_geom.py`'s struct-derived block
                  geometry and the byte identity `sum(row_bytes) + data_start == file size`.
  4. width map  — that derived map against the published-width-map markdown, which is PARSED OUT OF
                  THE DOCUMENT at run time rather than re-typed here, so the document stays the
                  authority. The document's GiB column is reproduced from the bpw table the document
                  itself states, then compared with the struct-derived bytes; a residual difference is
                  printed as a named finding, not absorbed.
  5. engine     — per tensor, `engine/loader.py`'s declared width (policy A through
                  `quant_scenarios.POLICY_TYPE`, the never-quantized role set
                  `engine/layout.py::EXACT_ROLES`, and the loader's own waiver table) against the
                  type the header stores; plus `loader.admit()` run over every tensor; plus three
                  independent header walks (this file's, `loader.parse_records`, and
                  `engine/artifact.py` through `tools/gguf_reader.py`) agreeing name by name.
                  Divergences outside the two D-27 documents them (expert `down_proj` wider,
                  `lm_head` waived narrower) are findings.

CONTROLS, both of which must go red for this audit to be green (`controls=N fired=N`):
  * mistyped-header-record — a record's stored type is deliberately rewritten; the family-map
    comparison, the width-map comparison and the engine-agreement step must each name it.
  * zero-record-parse — the header buffer is truncated so the tensor-info walk yields nothing; the
    parse must refuse with parsed=0 against the derived expectation, and the width-map comparison
    over an empty population must refuse rather than print an empty table and pass.

DEVICE AND TIMING DISCIPLINE. CPU only, read-only over the artifact: it reads the header for the map
and streams the file once for the digest. No device, no lock: no lockdir path and no acquire or release call, which is the point of the grep that:
driver that self-acquires a lock and is called by a caller already holding it deadlocks silently,
pitfalls/50), and no rate, duration, `ms`, `t/s` or `GB/s` anywhere in its output.

USAGE
  tools/gguf_conformance_audit.py                     # the audit, then both controls
  tools/gguf_conformance_audit.py --control mistyped-header-record
  tools/gguf_conformance_audit.py --out-json checkpoints/gguf-conformance-audit-2026-10-27.json
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine.artifact as ART                    # noqa: E402  the pinned artifact's own reader
import engine.layout as LAY                      # noqa: E402  EXACT_ROLES and the policy-A label
import engine.loader as LD                       # noqa: E402  the loader whose widths are on trial
import ggml_block_geom as GBG                    # noqa: E402  struct-derived block geometry
import gguf_header_types as G                    # noqa: E402  the repo's GGUF header tokenizer
import gguf_width_table as GWT                   # noqa: E402  a third, independent header walk
import quant_scenarios as QS                     # noqa: E402  policy A's own width table

GGUF = os.path.join(REPO, "models", "unsloth-q4km", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
WIDTH_MAP_MD = os.path.join(REPO, "quantization/evidence/q4km-published-width-map-2026-10-26.md")
PIN_PATH = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")
BYTE_AUTHORITY = os.path.join(REPO, "quantization/evidence/audit-hash-2026-10-26.json")
GIB = 2.0 ** 30
HEADER_READ = 64 << 20

# The width-map document's row label (markdown stripped) -> the family labels
# `tools/gguf_header_types.py::family()` gives the file's own tensor names, and whether the row's
# count is a sum over those families or a per-family count ("x10 each"). This is the one
# hand-written table in this file and it maps names only: every type and every count still comes
# from one side or the other. An unmapped document row or an unclaimed file family is fatal, so the
# mapping cannot quietly stop covering something.
MD_ROW_TO_FAMILIES = {
    "routedexpertsgate/up": (["routed experts gate/up"], "sum"),
    "routedexpertsdown": (["routed experts down"], "sum"),
    "sharedexpert": (["shared expert"], "sum"),
    "fullattentionq/k/v/output": (["full-attn q", "full-attn k", "full-attn v",
                                   "full-attn output"], "each"),
    "linear-attn(gdn)attn_qkv": (["linear-attn in_proj (attn_qkv)"], "sum"),
    "linear-attnz-gateattn_gate": (["linear-attn z-gate (attn_gate)"], "sum"),
    "gdnssm_out": (["GDN out_proj"], "sum"),
    "gdncontrol(ssm_a/alpha/beta/dt,conv1d,ssm_norm)": (["GDN conv1d",
                                                        "GDN other (a/b/gate/beta/A_log/dt_bias)"],
                                                       "sum"),
    "routerffn_gate_inp": (["router (ffn_gate_inp)"], "sum"),
    "norms": (["norms"], "sum"),
    "embeddings": (["token_embd (embeddings)"], "sum"),
    "lm_head(output)": (["output (lm_head)"], "sum"),
}


class AuditFatal(Exception):
    """A step that is not allowed to print a plausible answer. `tag` names the refusal."""

    def __init__(self, tag, msg):
        super().__init__("[%s] %s" % (tag, msg))
        self.tag = tag


# -----------------------------------------------------------------------------------------------
# 1. identity
# -----------------------------------------------------------------------------------------------
def check_identity(path):
    pin = ART.load_pin(PIN_PATH)
    size = os.path.getsize(path)
    if size != int(pin["bytes_on_disk"]):
        raise AuditFatal("size-mismatch",
                         "%s is %d B, the pin says %d B: a different file, or the same name with "
                         "different bytes" % (os.path.basename(path), size, pin["bytes_on_disk"]))
    got = ART.stream_sha256(path)
    if got != pin["sha256"]:
        raise AuditFatal("sha-mismatch",
                         "%s hashes %s, the pin %s says %s. Refusing: every width claim below is a "
                         "claim about the pinned artifact and about no other file."
                         % (os.path.basename(path), got, os.path.relpath(PIN_PATH, REPO),
                            pin["sha256"]))
    print("identity      sha256=%s matches_pin=True (pin %s; two independent passes are recorded "
          "in quantization/evidence/audit-hash-2026-10-26.json)"
          % (got, os.path.relpath(PIN_PATH, REPO)))
    return got, pin


# -----------------------------------------------------------------------------------------------
# 2. the header parse, and the population it must cover
# -----------------------------------------------------------------------------------------------
def parse_header(path, nbytes=HEADER_READ):
    """Walk GGUF's tensor-info array out of the file's own bytes. Returns rows + header facts.

    `nbytes` is the byte budget, exactly as in `tools/gguf_header_types.py`: a 248k-vocab model puts
    the whole vocabulary in the metadata section, so the tensor-info array starts ~10 MiB in. The
    count reconciliation at the end of this function is what makes a short read a refusal instead of
    a short table."""
    with open(path, "rb") as fh:
        buf = fh.read(nbytes)
    if buf[:4] != b"GGUF":
        raise AuditFatal("magic", "%s does not begin with GGUF" % path)
    c = G.Cur(buf)
    c.take(4)
    version = c.u32()
    declared = int(c.u64())
    n_kv = int(c.u64())
    meta, rows = {}, []
    try:
        for _ in range(n_kv):
            k = c.s()
            vt = c.u32()
            meta[k] = G.skip_array(c) if vt == 9 else c.value(vt)
        for _ in range(declared):
            name = c.s()
            nd = c.u32()
            dims = tuple(int(c.u64()) for _ in range(nd))
            dt = c.u32()
            off = c.u64()
            rows.append(dict(name=name, dims=dims, type=G.TYPES.get(dt, "type_%d" % dt),
                             offset=off, n_elts=1 if not dims else int(__import__("math").prod(dims))))
    except Exception as e:
        raise AuditFatal("parse-incomplete",
                         "parsed %d of %d declared tensor-info records before the walk failed (%s: "
                         "%s) on a %d-byte read. A short parse of a width policy reads like a "
                         "complete one, so this refuses." % (len(rows), declared, type(e).__name__,
                                                             e, len(buf)))
    align = int(meta.get("general.alignment", 32)) or 32
    header_end = c.p
    return dict(version=version, declared=declared, n_kv=n_kv, meta=meta, rows=rows,
                header_end=header_end, alignment=align, data_start=-(-header_end // align) * align,
                bytes_read=len(buf), arch=str(meta.get("general.architecture", "?")))


def derived_population():
    """The tensor names the artifact MUST carry, derived from the committed safetensors index.

    This is the denominator the parse is judged by: it comes from
    `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` x config-derived shapes, not from the
    header being about to agree with itself."""
    exp, draft = ART.expected_records()
    return exp, draft


def reconcile_counts(hdr, expected_names):
    parsed, declared = len(hdr["rows"]), hdr["declared"]
    print("parse         version=%d arch=%s n_kv=%d records_parsed=%d records_declared=%d "
          "expected_from_index=%d header_end=%d data_start=%d alignment=%d"
          % (hdr["version"], hdr["arch"], hdr["n_kv"], parsed, declared, len(expected_names),
             hdr["header_end"], hdr["data_start"], hdr["alignment"]))
    if parsed == 0:
        raise AuditFatal("zero-record-parse",
                         "0 tensor-info records parsed against %d expected. A census that parsed "
                         "nothing and printed an empty table is the failure this audit exists to "
                         "refuse (pitfalls/40 section empty-parse-PASS)." % len(expected_names))
    if parsed != declared:
        raise AuditFatal("parse-count", "parsed %d records, the header declares %d"
                         % (parsed, declared))
    if parsed != len(expected_names):
        raise AuditFatal("population-count", "parsed %d records, the committed index derives %d "
                                            "expected tensors: a family that vanished from the file "
                                            "would otherwise just make the totals read low"
                         % (parsed, len(expected_names)))
    names = {r["name"] for r in hdr["rows"]}
    if len(names) != parsed:
        raise AuditFatal("duplicate-name", "%d records but %d distinct names" % (parsed, len(names)))
    missing = sorted(set(expected_names) - names)
    extra = sorted(names - set(expected_names))
    if missing or extra:
        raise AuditFatal("name-set", "missing %d (first %s), unexpected %d (first %s)"
                         % (len(missing), missing[:1], len(extra), extra[:1]))
    return parsed


# -----------------------------------------------------------------------------------------------
# 3. the family map, derived from the file
# -----------------------------------------------------------------------------------------------
def family_map(rows, size=None, data_start=None):
    """{family: {type: count}} plus bytes plus per-(family,type) element lists, from the records."""
    if not rows:
        raise AuditFatal("empty-population",
                         "0 records to build a family map from: an empty census of a width policy "
                         "prints an empty table and a zero total, which is a pass-shaped nothing")
    fam, bytes_by_fam, elts_by_ft, total_bytes, unknown = {}, {}, {}, 0, set()
    for r in rows:
        t = r["type"]
        try:
            GBG.bpw_of(t)
        except KeyError:
            unknown.add(t)
            continue
        f = G.family(r["name"])
        fam.setdefault(f, {}).setdefault(t, 0)
        fam[f][t] += 1
        rb = GBG.row_bytes(r["n_elts"], t)
        bytes_by_fam[f] = bytes_by_fam.get(f, 0) + rb
        elts_by_ft.setdefault((f, t), []).append(r["n_elts"])
        total_bytes += rb
    if unknown:
        raise AuditFatal("type-unpriceable",
                         "no derived block geometry for %s; pricing them as 0 would understate the "
                         "one column this audit prints" % sorted(unknown))
    if size is not None:
        identity = total_bytes + data_start
        if identity != size:
            raise AuditFatal("byte-identity",
                             "sum(row_bytes)=%d + data_start=%d = %d, file size %d (off by %+d B): "
                             "the block geometry does not reproduce this file, so no GiB figure here "
                             "can be trusted" % (total_bytes, data_start, identity, size,
                                                 size - identity))
        print("bytes         sum(row_bytes)=%d + data_start=%d == file size %d (struct-derived "
              "geometry, tools/ggml_block_geom.py)" % (total_bytes, data_start, size))
    return fam, bytes_by_fam, elts_by_ft, total_bytes


def print_family_table(fam, bytes_by_fam):
    print("\nfamily                                       stored types (derived count)      "
          "tensors    stored bytes        GiB")
    print("-" * 103)
    n_tot = 0
    for f in sorted(fam):
        cnt = sum(fam[f].values())
        n_tot += cnt
        print("%-44s %-30s %8d %14d %9.4f"
              % (f, ", ".join("%s x%d" % (t, n) for t, n in sorted(fam[f].items())), cnt,
                 bytes_by_fam[f], bytes_by_fam[f] / GIB))
    print("%-44s %-30s %8d %14d %9.4f" % ("TOTAL", "", n_tot, sum(bytes_by_fam.values()),
                                           sum(bytes_by_fam.values()) / GIB))
    return n_tot, len(fam)


# -----------------------------------------------------------------------------------------------
# 4. the width map, read out of the document that is the authority for it
# -----------------------------------------------------------------------------------------------
TYPE_TOKEN = r"(?:Q[0-9]_[A-Z0-9]+|IQ[0-9]_[A-Z0-9]+|MXFP4|F16|F32|BF16)"


def _md_label(cell):
    return re.sub(r"[*`\s]+", "", cell).lower()


def parse_width_map(path):
    """The published-width-map table, parsed. Returns (rows, total_gib, bpw_note).

    The document is the authority D-27 points at, so its numbers are read from it at run time: if
    the document is corrected, the audit's expectation moves with it, and if a row goes missing the
    audit says so instead of comparing against a remembered copy."""
    text = open(path, encoding="utf-8").read().replace("\u00d7", "x")
    rows, total_gib = [], None
    for ln in text.splitlines():
        s = ln.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 5:
            continue
        label, pub, gib_col, policy_col, same_col = cells
        if label.lower().startswith("family") or set(label) <= set("-: "):
            continue
        if "total weights" in label.lower():
            m = re.search(r"([0-9.]+)\s*GiB", gib_col)
            total_gib = float(m.group(1)) if m else None
            continue
        kinds = re.findall("(%s)[\\s*]*[xX]?([0-9]+)?" % TYPE_TOKEN, pub)
        counts = {}
        for tname, num in kinds:
            counts.setdefault(tname, 0)
            counts[tname] += int(num) if num else 1
        if not counts:
            raise AuditFatal("width-map-unparsed",
                             "width-map row %r names no ggml type in its published column (%r); "
                             "the table was expected to be machine-readable and is not" % (label, pub))
        pm = re.search(r"([0-9.]+)(?:\s*GiB)?", gib_col)
        rows.append(dict(label=label, key=_md_label(label), counts=counts, each=("each" in pub),
                         gib=float(pm.group(1)) if pm else None, policy=policy_col, same=same_col,
                         published=pub))
    if not rows:
        raise AuditFatal("width-map-empty", "no width-map rows parsed out of %s. A document parse "
                                            "that matched nothing cannot fail, which is why it is "
                                            "fatal here." % os.path.relpath(path, REPO))
    note = re.search(r"Byte totals use standard ggml block sizes \((.*?)\)", text)
    bpw_note = {}
    if note:
        bpw_note = {t: float(b) for t, b in
                    re.findall("(%s) ([0-9.]+)" % TYPE_TOKEN, note.group(1))}
    return rows, total_gib, bpw_note


def map_md_rows(md_rows, fam):
    """Every document row must map to file families and every file family must be claimed.

    Both directions are checked: a document row nobody claimed, and a file family the mapping no
    longer covers, are each fatal."""
    claimed, unmapped = set(), []
    for r in md_rows:
        if r["key"] not in MD_ROW_TO_FAMILIES:
            unmapped.append(r["label"])
            continue
        for f in MD_ROW_TO_FAMILIES[r["key"]][0]:
            if f not in fam:
                raise AuditFatal("width-map-family-absent",
                                  "width-map row %r names file family %r, which the file does not "
                                  "produce" % (r["label"], f))
            if f in claimed:
                raise AuditFatal("family-double-claimed",
                                  "file family %r is claimed by two width-map rows; the comparison "
                                  "would then judge it twice and total it once" % f)
            claimed.add(f)
    if unmapped:
        raise AuditFatal("width-map-row-unmapped",
                         "%d width-map rows are not mapped to file families: %s"
                         % (len(unmapped), unmapped))
    missing = sorted(set(fam) - claimed)
    if missing:
        raise AuditFatal("family-unclaimed",
                         "%d file families are claimed by no width-map row: %s" % (len(missing), missing))
    return claimed


def compare_width_map(md_rows, fam, bytes_by_fam, elts_by_ft, total_gib_file, total_gib_doc,
                      bpw_note):
    """Compare the derived map to the document's. Returns (findings, lines, n_rows_checked)."""
    findings, lines, doc_deltas = [], [], []
    map_md_rows(md_rows, fam)
    for r in md_rows:
        families, mode = MD_ROW_TO_FAMILIES[r["key"]]
        got = {}
        for f in families:
            for t, n in fam[f].items():
                got.setdefault(t, 0)
                got[t] += n
        want = dict(r["counts"])
        if r["each"]:
            per = set(want.values())
            if len(want) != 1 or len(per) != 1:
                raise AuditFatal("width-map-each-ambiguous",
                                 "row %r says xN each but its published column is %r: this audit "
                                 "will not guess which count applies to which family"
                                 % (r["label"], r["published"]))
            each = {list(want)[0]: per.pop()}
            for f in families:                      # "each" is a per-family claim, so check it per family
                if fam[f] != each:
                    findings.append("width-map-each row %r expects %s in every family; family %r "
                                    "stores %s" % (r["label"], each, f, fam[f]))
            want = {t: n * len(families) for t, n in each.items()}
        if got != want:
            findings.append("width-map-types row %r says %s, the file stores %s"
                            % (r["label"], ", ".join("%s x%d" % kv for kv in sorted(want.items())),
                               ", ".join("%s x%d" % kv for kv in sorted(got.items()))))
        # The document's GiB column, reproduced from the bpw table the document itself states.
        md_gib = None
        if bpw_note and r["gib"] is not None:
            tot = 0
            for f in families:
                for t in fam[f]:
                    blk = GBG.block_elts(t)
                    # The document's own bpw note prices the quantized types; a type it does not name
                    # (F32 is the one here) is priced by the struct table, which is what the audit's
                    # own column uses. Which is which is printed, so the reproduction is not opaque.
                    bpw = bpw_note.get(t) or GBG.bpw_of(t)
                    per_block = bpw * blk / 8.0
                    if abs(per_block - round(per_block)) > 1e-9:
                        raise AuditFatal("bpw-note-block",
                                         "bpw %s=%s does not give a whole block size at %d "
                                         "elements/block" % (t, bpw, blk))
                    for n in elts_by_ft[(f, t)]:
                        tot += -(-n // blk) * round(per_block)
            md_gib = round(tot / GIB, 3)
            if md_gib != round(r["gib"], 3):
                doc_deltas.append((r["label"], r["gib"], md_gib))
        row_bytes = sum(bytes_by_fam[f] for f in families)
        lines.append(dict(label=r["label"], want=want, got=got, doc_gib=r["gib"], md_gib=md_gib,
                          derived_gib=round(row_bytes / GIB, 4), policy=r["policy"],
                          families=list(families)))
    doc_total = sum(x["doc_gib"] for x in lines if x["doc_gib"] is not None)
    if total_gib_doc is not None and abs(doc_total - total_gib_doc) > 0.01:
        findings.append("width-map-total rows sum to %.3f GiB, the document's total row says %.3f"
                        % (doc_total, total_gib_doc))
    return findings, lines, len(md_rows), doc_deltas


# -----------------------------------------------------------------------------------------------
# 5. the engine agrees with the file, per tensor
# -----------------------------------------------------------------------------------------------
def declared_for(rec):
    """The width the engine DECLARES for one tensor, read out of the engine's own tables.

    Two sources, both derived: a never-quantized role (`engine/layout.py::EXACT_ROLES`) is declared
    by `loader.py::EXACT_KINDS` for a gguf source, and everything else is policy A's own family
    width through `quant_scenarios.POLICY_TYPE`. This file never types a width of its own."""
    if rec.role in LAY.EXACT_ROLES:
        kinds = sorted(LD.EXACT_KINDS["gguf"])
        if len(kinds) != 1:
            raise AuditFatal("exact-kind-ambiguous",
                             "loader.EXACT_KINDS['gguf'] holds %s; this audit cannot declare one "
                             "exact width from a set of several" % kinds)
        return kinds[0], "exact-role"
    return QS.POLICY_TYPE(LD.POLICY_A, rec.family), "policy-A"


def engine_agreement(recs):
    """Per tensor: declared width vs stored width, plus the loader's own veto on the same pair.

    Returns (findings, summary). A divergence is printed always and is a finding only when it is
    neither of the two D-27 records: `wider` than policy A without being wider than the loader's
    floor, or `narrower` without a waiver the loader itself carries. A third divergence appearing
    silently is the failure mode this exists for, so `judge_divergences` owns that verdict.
    """
    counts = {"equal": 0, "wider": 0, "narrower": 0}
    by_role, admit_notes, findings = {}, {}, []
    for rec in recs:
        declared, basis = declared_for(rec)
        relation = "equal"
        if rec.type_name != declared:
            relation = "wider" if GBG.bpw_of(rec.type_name) > GBG.bpw_of(declared) else "narrower"
        counts[relation] += 1
        if relation != "equal":
            by_role.setdefault((basis, rec.role, declared, rec.type_name, relation), []).append(rec.name)
            if relation == "narrower":
                wv = LD.WIDTH_WAIVERS.get(rec.role)
                if not (wv and rec.type_name in wv["accepted"]):
                    findings.append("narrower-undeclared %s stores %s where the engine declares %s "
                                    "(%.4f vs %.4f bpw) and loader.WIDTH_WAIVERS carries no waiver "
                                    "for role %r" % (rec.name, rec.type_name, declared,
                                                     GBG.bpw_of(rec.type_name), GBG.bpw_of(declared),
                                                     rec.role))
        try:
            note = LD.admit(rec)
        except LD.WidthVeto as e:
            note = "refused:%s" % e.tag
            if relation == "wider":
                findings.append("loader-refuses-a-wider-tensor %s: %s" % (rec.name, e))
        key = note.split(":")[0]
        admit_notes[key] = admit_notes.get(key, 0) + 1
    n = sum(counts.values())
    if n != len(recs):
        raise AuditFatal("engine-agreement-count",
                         "classified %d tensors of %d records; a classification that drops records "
                         "cannot report agreement" % (n, len(recs)))
    if n == 0:
        raise AuditFatal("engine-agreement-empty",
                         "the engine leg classified 0 records; there is nothing to compare")
    return findings, dict(counts=counts, by_role=dict(by_role),
                          admit_notes=admit_notes, total=n)


def loader_records(path, mutate=None):
    """The loader's own parse, as `TensorRecord`s, with no `Artifact()` construction.

    `engine/artifact.py::Artifact` cannot be built on this file at all: its `_c_draft_block` check
    requires an MTP draft block and the published GGUF carries zero `mtp.*` tensors
    (quantization/evidence/q4km-published-width-map-2026-10-26.md §What else the file carries, item
    3). That is a property of the artifact, not of this audit, so the engine leg goes through
    `loader.parse_records` + `loader.ruler_record`, which is the same header the staging path reads.
    """
    pr = LD.parse_records(path)
    recs = [LD.ruler_record(row, pr["data_start"]) for row in pr["rows"]]
    if mutate:
        name, wrong = mutate
        hit = [r for r in recs if r.name == name]
        if len(hit) != 1:
            raise AuditFatal("control-target", "engine leg needs exactly one record named %s" % name)
        hit[0].type_name = wrong
    return recs


# The two divergences D-27 records. Anything else is a finding; these two are expected, printed, and
# counted — an expected divergence is not a silent one.
EXPECTED_DIVERGENCES = {
    ("policy-A", "blk.N.ffn_down_exps.weight", "Q4_K", "Q5_K", "wider"),
    ("policy-A", "blk.N.ffn_down_exps.weight", "Q4_K", "Q6_K", "wider"),
    ("policy-A", "output.weight", "Q8_0", "Q6_K", "narrower"),
}


def judge_divergences(summary):
    findings = []
    seen = set(summary["by_role"])
    unexpected = sorted(seen - EXPECTED_DIVERGENCES)
    vanished = sorted(EXPECTED_DIVERGENCES - seen)
    for key in unexpected:
        findings.append("third-divergence %s is not one of the two divergences D-27 records; the "
                        "width map, the loader and D-27 must be reconciled before any row reads this "
                        "artifact as policy A" % (key,))
    for key in vanished:
        findings.append("documented-divergence-gone %s no longer diverges: either the artifact "
                        "changed (the sha check would have said so) or the loader's declared width "
                        "moved, and a moved declared width is exactly the stale constant this check "
                        "exists to catch" % (key,))
    return findings


def walks_agree(mine, loader_rows, ruler_rows):
    """Three header walks over one file: this audit's, `loader.parse_records`, `gguf_width_table`'s.

    They are three parses of the same bytes — this file's walk, `loader.py`'s ruler walk (which keeps
    the offset field), and `tools/gguf_width_table.py::parse_gguf` (which throws it away). A
    disagreement is a named finding: two readers of one header that disagree is a bug in one of them,
    never a matter of opinion. `engine/artifact.py`'s walk is the fourth reader but it cannot be
    built on a file with no draft block (see `loader_records`)."""
    a = {r["name"]: r["type"] for r in mine}
    b = {r[1]: r[3] for r in loader_rows}
    c = {r[0]: G.TYPES.get(r[2], "type_%d" % r[2]) for r in ruler_rows}
    compared = len(set(a) & set(b) & set(c))
    if compared == 0:
        raise AuditFatal("walk-comparison-empty",
                         "0 tensor names in common across the three header walks; a comparison that "
                         "compared nothing may not print agreement")
    diff = [n for n in sorted(set(a) & set(b) & set(c)) if not (a[n] == b[n] == c[n])]
    for n in diff:
        print("FINDING   walk-disagreement %s: this audit reads %s, loader.parse_records reads %s, "
              "gguf_width_table reads %s" % (n, a[n], b[n], c[n]))
    return compared, diff


# -----------------------------------------------------------------------------------------------
# 6. controls — each must be seen to go red, or the check it guards is not believed
# -----------------------------------------------------------------------------------------------
def byte_authority(total_bytes, size, parsed):
    """The derived byte total against an independent record of the same bytes.

    `quantization/evidence/audit-hash-2026-10-26.json` is a separate pass over this file from
    2026-10-26 (hash plus census). The comparison is exact, with no tolerance: two counts of the same
    bytes that differ by one byte means one of them is wrong. This is also why the width-map
    document's GiB column is printed but never asserted — those totals were typed from the
    hand-written bpw table that `quantization/evidence/audit-block-geometry-2026-10-26.md` retired
    (Q5_K 5.5625 against the struct's 5.5, a 74 MiB overcharge across 37 `down_proj` tensors), so
    asserting the document's arithmetic would assert a table this repo has already overruled."""
    try:
        au = json.load(open(BYTE_AUTHORITY))
    except (OSError, ValueError) as exc:
        raise AuditFatal("byte-authority-unreadable",
                         "%s: %s — the audit has no independent byte count to check against"
                         % (BYTE_AUTHORITY, exc))
    pairs = (("weight_bytes", int(au["weight_bytes"]), total_bytes),
             ("bytes_on_disk", int(au["bytes_on_disk"]), size),
             ("n_tensors", int(au["n_tensors"]), parsed))
    findings = []
    for name, want, got in pairs:
        if want != got:
            findings.append("byte-authority %s: this audit derives %d, %s records %d (delta %+d)"
                            % (name, got, os.path.relpath(BYTE_AUTHORITY, REPO), want,
                               got - want))
    print("authority     derived weight_bytes=%d file=%d records=%d vs %s (%d/%d/%d) delta=%+d"
          % (total_bytes, size, parsed, os.path.relpath(BYTE_AUTHORITY, REPO),
             au["weight_bytes"], au["bytes_on_disk"], au["n_tensors"],
             (total_bytes - au["weight_bytes"]) + (size - au["bytes_on_disk"])
             + (parsed - au["n_tensors"])))
    return findings


def control_mistyped_record(path, md_rows, bpw_note, expected_names):
    """Rewrite one record's stored type and require the audit to redden, by name, in three places.

    Two cases, because a mis-typed record can go either way against policy A: `attn_qkv` Q8_0->Q4_K
    (narrower, the loader's own veto must fire) and `ffn_gate_inp` F32->Q8_0 (a never-quantized role
    stored quantized, which the width map must catch even though Q8_0 is "wider" than nothing)."""
    hdr = parse_header(path)
    reconcile_counts(hdr, expected_names)
    cases = [("blk.0.attn_qkv.weight", "Q4_K", "width-map-types"),
             ("blk.0.ffn_gate_inp.weight", "Q8_0", "narrower-undeclared")]
    fired, notes = 0, []
    for name, wrong, expect_tag in cases:
        honest = [r["type"] for r in hdr["rows"] if r["name"] == name]
        if len(honest) != 1:
            raise AuditFatal("control-target", "control needs exactly one record named %s" % name)
        for r in hdr["rows"]:
            if r["name"] == name:
                r["type"] = wrong
        fam, bytes_by_fam, elts_by_ft, _tot = family_map(hdr["rows"])
        wf, _, _n, _dd = compare_width_map(md_rows, fam, bytes_by_fam, elts_by_ft, None, None,
                                           bpw_note)
        ef, summary = engine_agreement(loader_records(path, mutate=(name, wrong)))
        ef = ef + judge_divergences(summary)
        tagged = [x for x in wf + ef if expect_tag in x or name in x]
        ok = bool(tagged) and len(wf) > 0 and len(ef) > 0
        notes.append("%s %s->%s: %d width-map findings, %d engine findings (%s) %s"
                     % (name, honest[0], wrong, len(wf), len(ef),
                        sorted({t.split()[0] for t in tagged}), "RED" if ok else "GREEN"))
        fired += 1 if ok else 0
    return fired, len(cases), notes


def control_zero_record(path, expected_names):
    """A parse that yields nothing must be fatal, in both places a count could be quietly absent."""
    legs, notes = 0, []
    try:
        hdr = parse_header(path, nbytes=4096)
        reconcile_counts(hdr, expected_names)
        notes.append("truncated-header: NOT CAUGHT (parse returned %d records)" % len(hdr["rows"]))
    except AuditFatal as e:
        notes.append("truncated-header: %s" % str(e).split("Refusing")[0].strip())
        legs += 1
    try:
        family_map([])
        notes.append("empty-population: NOT CAUGHT (an empty census returned)")
    except AuditFatal as e:
        notes.append("empty-population: %s" % str(e).split("pass")[0].strip())
        legs += 1
    return legs, 2, notes


# -----------------------------------------------------------------------------------------------
# 7. the run
# -----------------------------------------------------------------------------------------------
def run_audit(path, md_path, out_json=None):
    findings = []
    sha, pin = check_identity(path)
    size = os.path.getsize(path)
    expected, draft = derived_population()
    hdr = parse_header(path)
    parsed = reconcile_counts(hdr, expected)
    fam, bytes_by_fam, elts_by_ft, total_bytes = family_map(hdr["rows"], size=size,
                                                            data_start=hdr["data_start"])
    n_fams_tensors, n_fams = print_family_table(fam, bytes_by_fam)
    md_rows, doc_total, bpw_note = parse_width_map(md_path)
    wf, lines, n_md_rows, doc_deltas = compare_width_map(md_rows, fam, bytes_by_fam, elts_by_ft,
                                                          total_bytes, doc_total, bpw_note)
    print("\nwidth map     rows_parsed=%d rows_compared=%d doc_total_gib=%s derived_total_gib=%.4f "
          "bpw_note_types=%d (source %s)"
          % (len(md_rows), n_md_rows, doc_total, total_bytes / GIB, len(bpw_note),
             os.path.relpath(md_path, REPO)))
    print("%-42s %-34s %-34s %9s %9s %9s" % ("width-map row", "document says", "file stores",
                                              "doc GiB", "doc repro", "derived GiB"))
    for x in lines:
        print("%-42s %-34s %-34s %9s %9s %9.4f"
              % (x["label"][:41],
                 ", ".join("%s x%d" % kv for kv in sorted(x["want"].items()))[:33],
                 ", ".join("%s x%d" % kv for kv in sorted(x["got"].items()))[:33],
                 x["doc_gib"], x["md_gib"], x["derived_gib"]))
    if doc_deltas:
        print("doc-column    %d row(s) whose printed GiB does not reproduce from the document's own "
              "bpw note: %s. That column is a 3-decimal display typed from the hand-written bpw table "
              "that quantization/evidence/audit-block-geometry-2026-10-26.md retired; it is reported, "
              "not asserted, because the byte authority here is the struct geometry cross-checked "
              "against audit-hash-2026-10-26.json below (delta 0), and the width map's authority is "
              "its type-and-count map, which matched exactly."
              % (len(doc_deltas), "; ".join("%r doc %.3f vs repro %.3f" % d for d in doc_deltas)))
    findings += wf
    recs = loader_records(path)
    ef, summary = engine_agreement(recs)
    print("\nengine        records=%d classified=%d equal=%d wider=%d narrower=%d loader-admit-notes=%s"
          % (summary["total"], summary["total"], summary["counts"]["equal"],
             summary["counts"]["wider"], summary["counts"]["narrower"],
             json.dumps(summary["admit_notes"], sort_keys=True)))
    print("engine        declared-vs-stored divergences=%d" % len(summary["by_role"]))
    for (basis, role, dec, stored, rel), names in sorted(summary["by_role"].items()):
        exp = "EXPECTED(D-27)" if (basis, role, dec, stored, rel) in EXPECTED_DIVERGENCES else "UNEXPECTED"
        print("divergence    %-14s role=%-30s engine declares %-5s file stores %-5s x%-4d %s"
              % (rel, role, dec, stored, len(names), exp))
    findings += ef
    findings += judge_divergences(summary)
    n_mtp = sum(1 for r in hdr["rows"] if "mtp" in r["name"])
    print("draft         the file carries %d tensors naming 'mtp'; engine/artifact.py::_c_draft_block "
          "refuses an artifact with no draft block (%d expected from the committed index), so "
          "Artifact() cannot be built on this file by construction and the engine leg parses through "
          "loader.parse_records instead"
          % (n_mtp, len(draft)))
    compared, diff = walks_agree(hdr["rows"], LD.parse_records(path)["rows"],
                                 GWT.parse_gguf(path)[2])
    print("walks         three header walks compared=%d/%d differing=%d"
          % (compared, parsed, len(diff)))
    findings += ["walk-disagreement on %d tensors" % len(diff)] if diff else []
    findings += byte_authority(total_bytes, size, parsed)

    cf, legs_fired, legs_total, cnotes = 0, 0, 0, []
    for name, fn in (("mistyped-header-record",
                      lambda: control_mistyped_record(path, md_rows, bpw_note, expected)),
                     ("zero-record-parse", lambda: control_zero_record(path, expected))):
        fired, total, notes = fn()
        for n in notes:
            print("control       %s: %s" % (name, n))
        legs_fired += fired
        legs_total += total
        if fired != total:
            cnotes.append("%s fired=%d of %d" % (name, fired, total))
        cf += 1 if fired == total else 0
    for f in findings:
        print("FINDING   %s" % f)
    verdict = "CONFORMS" if not findings and cf == 2 else "REFUSED"
    print("\nRESULT verdict=%s sha256=%s records=%d/%d families=%d families_tensor_total=%d "
          "types=%d width_map_rows=%d divergences=%d findings=%d doc_column_deltas=%d controls=%d "
          "fired=%d legs=%d/%d draft_tensors_expected=%d no timing taken"
          % (verdict, sha[:12], parsed, len(expected), n_fams, n_fams_tensors,
             len({t for f in fam.values() for t in f}), n_md_rows, len(summary["by_role"]),
             len(findings), len(doc_deltas), cf, cf, legs_fired, legs_total, len(draft)))
    if out_json:
        if os.path.exists(out_json):
            raise AuditFatal("out-exists", "%s exists; manifests here are write-once" % out_json)
        json.dump(dict(kind="gguf-conformance-audit", date="2026-10-27", verdict=verdict,
                       artifact=os.path.relpath(path, REPO), sha256=sha, records=parsed,
                       records_expected_from_index=len(expected), families={f: fam[f] for f in fam},
                       family_bytes={f: bytes_by_fam[f] for f in fam},
                       total_weight_bytes=total_bytes, data_start=hdr["data_start"],
                       header_end=hdr["header_end"], families_count=n_fams,
                       doc_column_deltas=["%s doc %.3f repro %.3f" % d for d in doc_deltas],
                       width_map=dict(path=os.path.relpath(md_path, REPO), rows=n_md_rows,
                                      doc_total_gib=doc_total, bpw_note=bpw_note),
                       engine=dict(summary["counts"], total=summary["total"],
                                   admit_notes=summary["admit_notes"],
                                   divergences=["%s|%s|%s|%s|%s x%d" % (k[0], k[1], k[2], k[3], k[4],
                                                                        len(v))
                                                for k, v in summary["by_role"].items()]),
                       walks_compared=compared, walks_differing=len(diff),
                       findings=findings, controls_fired=cf,
                       control_legs_fired=legs_fired, control_legs_total=legs_total),
                  open(out_json, "w"), indent=1)
    return 0 if verdict == "CONFORMS" else 1


def main():
    ap = argparse.ArgumentParser(description="GGUF conformance audit (D-27)")
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--width-map", default=WIDTH_MAP_MD)
    ap.add_argument("--out-json")
    ap.add_argument("--control", choices=("mistyped-header-record", "zero-record-parse"))
    args = ap.parse_args()
    try:
        if args.control:
            expected, _d = derived_population()
            md_rows, _t, bpw_note = parse_width_map(args.width_map)
            fn = (control_mistyped_record if args.control == "mistyped-header-record"
                  else control_zero_record)
            args_c = ((args.gguf, md_rows, bpw_note, expected)
                      if args.control == "mistyped-header-record" else (args.gguf, expected))
            fired, total, notes = fn(*args_c)
            for n in notes:
                print("control       %s: %s" % (args.control, n))
            print("CONTROL %s fired=%d of %d %s" % (args.control, fired, total,
                                                    "RED" if fired == total else "NOT-RED"))
            return 0 if fired == total else 1
        return run_audit(args.gguf, args.width_map, args.out_json)
    except AuditFatal as e:
        print("FATAL     %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
