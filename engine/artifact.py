#!/usr/bin/env python3
"""artifact.py — what the shipping file claims to be, and the refusal to load anything else.

The engine has exactly one weight source it is allowed to read: the pinned published artifact
`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, plus the MTP draft block extracted to
`models/w1-mtp-extract/mtp.safetensors`. A loader that accepts "a GGUF" rather than *this* file is
the bug this repo has been bitten by: `tools/gguf_suffix_census.py` §"why it exists" records that an
earlier hand-written bucketing of these very records mislabelled the GDN family entirely, because
this model spells GDN `attn_qkv` / `attn_gate` / `ssm_out` while the classifier was looking for the
HF spellings `in_proj` / `linear_attn`. Name-substring buckets are the thing not to trust, so this
module derives the expected name set from committed sources and aborts on anything else.

Every expectation is read from a committed source, none is remembered:

  | expectation                        | source                                                  |
  |------------------------------------|---------------------------------------------------------|
  | file size, sha256, header fields   | `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`             |
  | tensor NAME set, element counts,   | the safetensors index x config shapes, paired through   |
  | dims                               | `tools/bf16_census.py`'s `MAP_HF_GGUF` / `hf_shape`     |
  |                                    | (imported, not restated)                                |
  | stored WIDTH per tensor            | `quantization/evidence/q4km-published-width-map-2026-10-26.md` |
  |                                    | §family table, plus the file's own tensor records       |
  | zero `mtp.*` tensors               | the same evidence; `quantization/residency-layout.md` §2 |
  |                                    | makes it load-bearing (draft bytes come from HF shards) |
  | row bytes, data-section closure    | `tools/ggml_block_geom.py` (widths derived from the     |
  |                                    | pinned `ggml-common.h`) + `tools/gguf_width_table.py`   |
  |                                    | as the independent second ruler                         |

Width is a **per-family, in fact per-tensor** fact. `ffn_down_exps` is Q5_K on 37 layers and Q6_K on
three; the evidence records the 37/3 split, and *which three* is read out of the file because no
document in this repo names them. A loader that prices a whole group at one width is wrong by the
difference and still closes against itself, so the split is checked explicitly and
`down_proj_group_width` in `engine/gate.py` is the injection that proves the check is awake.

CPU-only: reads the header (and, on request, the data section through `loader.py`). No device call,
no timer import, no ms or tok/s figure printed anywhere in this file.
"""
import hashlib
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

import bf16_census as BC                 # noqa: E402  the name map and shape model, reused
import ggml_block_geom as GBG            # noqa: E402  widths derived from the pinned ggml header
import gguf_reader as gr                 # noqa: E402  GGUF header parse (structure only)
import gguf_width_table as GWT           # noqa: E402  the independent byte ruler (own tokenizer)
import residency_budget as RB            # noqa: E402  committed paths and budget brackets

PIN_PATH = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json")
WIDTH_MAP_EVIDENCE = "quantization/evidence/q4km-published-width-map-2026-10-26.md"
MTP_AUTHORITY = "checkpoints/Qwen3.6-35B-A3B.gguf-pin.json §authority.mtp_elements"

TYPE_IDS = {v: k for k, v in gr.GGML_TYPES.items()}

# The published width map, keyed by the file's own suffix with the layer index folded to N — the same
# fold `tools/gguf_suffix_census.py` prints by, so the two tools cannot be describing two different
# files. The values are SETS because one family legitimately carries two widths.
EXPECTED_TYPE_BY_ROLE = {
    "token_embd.weight": {"Q8_0"},                        # D-10
    "output.weight": {"Q6_K"},                             # published is NARROWER than D-10
    "output_norm.weight": {"F32"},
    "blk.N.attn_norm.weight": {"F32"},
    "blk.N.post_attention_norm.weight": {"F32"},
    "blk.N.attn_q.weight": {"Q8_0"},
    "blk.N.attn_k.weight": {"Q8_0"},
    "blk.N.attn_v.weight": {"Q8_0"},
    "blk.N.attn_output.weight": {"Q8_0"},
    "blk.N.attn_q_norm.weight": {"F32"},
    "blk.N.attn_k_norm.weight": {"F32"},
    # GDN, spelled the way THIS checkpoint spells it. The HF spellings are on the reject list.
    "blk.N.attn_qkv.weight": {"Q8_0"},
    "blk.N.attn_gate.weight": {"Q8_0"},
    "blk.N.ssm_out.weight": {"Q8_0"},
    "blk.N.ssm_conv1d.weight": {"F32"},
    "blk.N.ssm_a": {"F32"},
    "blk.N.ssm_dt.bias": {"F32"},
    "blk.N.ssm_norm.weight": {"F32"},
    "blk.N.ssm_alpha.weight": {"F32"},
    "blk.N.ssm_beta.weight": {"F32"},
    "blk.N.ffn_gate_inp.weight": {"F32"},                  # the router is never quantized
    "blk.N.ffn_gate_inp_shexp.weight": {"F32"},
    "blk.N.ffn_gate_exps.weight": {"Q4_K"},                # D-01 / D-08
    "blk.N.ffn_up_exps.weight": {"Q4_K"},
    "blk.N.ffn_down_exps.weight": {"Q5_K", "Q6_K"},        # per-tensor, NOT one width per group
    "blk.N.ffn_gate_shexp.weight": {"Q8_0"},
    "blk.N.ffn_up_shexp.weight": {"Q8_0"},
    "blk.N.ffn_down_shexp.weight": {"Q8_0"},
}
# The per-family counts the width-map evidence records. Checked against the file, not quoted from it.
EXPECTED_CENSUS = {"Q4_K": 80, "Q5_K": 37, "Q8_0": 251, "Q6_K": 4, "F32": 361}
EXPECTED_N_TENSORS = 733
EXPECTED_DRAFT_TENSORS = 19               # PLAN.md step 1.2 / audit-shards

# Spellings a GGUF of this checkpoint never carries. Their presence is the signature of a file from a
# different naming convention, or of a header being mis-parsed — and it is precisely the confusion
# that cost tools/gguf_suffix_census.py its first cut.
REJECT_NAME_SUBSTRINGS = ("in_proj", "linear_attn", "self_attn", "gate_up_proj", "down_proj",
                          "embed_tokens", "language_model", "mlp.gate", "shared_expert")
DRAFT_NAME_PREFIX = "mtp."

# role (file suffix, layer folded to N) -> the HF name it is paired with. Read out of the census's
# own map, so family bucketing goes through tools/bf16_census.py's classifier instead of a second
# substring guess living here.
HF_BY_ROLE = {}
for _hf, _glist in BC.MAP_HF_GGUF.items():
    for _gn in _glist:
        HF_BY_ROLE[_gn] = _hf


class ArtifactError(Exception):
    """The file is not the pinned artifact, or does not say what the pin says it says.

    `tag` names the check that fired, so `engine/gate.py` can require an injection to fire a *specific*
    check rather than merely to produce a failure — a control that accepts any exception cannot tell a
    caught bug from a crash.

    The attribute used to be only a promise: the class body documented `tag` and never set it, so every
    refusal unwrapped to a two-tuple and a gate could not name the check that fired. Three lines make
    the documented contract true."""

    def __init__(self, tag, message):
        super().__init__("[%s] %s" % (tag, message))
        self.tag = tag
        self.message = message


# -----------------------------------------------------------------------------------------------
# the pin, read not remembered
# -----------------------------------------------------------------------------------------------
def load_pin(path=PIN_PATH):
    if not os.path.exists(path):
        raise ArtifactError("pin-absent",
                            "%s is absent: the committed pin is where this loader gets its identity "
                            "expectations (size, sha256, header fields, the draft block's element "
                            "count). Refusing to fall back to remembered numbers." % path)
    try:
        pin = json.load(open(path))
    except ValueError as e:
        raise ArtifactError("pin-unparseable", "cannot parse the pin %s: %s" % (path, e))
    for k in ("artifact", "bytes_on_disk", "sha256", "gguf_header", "authority"):
        if k not in pin:
            raise ArtifactError("pin-short", "pin %s has no %r; the loader will not guess it"
                                % (path, k))
    if "mtp_elements" not in pin["authority"]:
        raise ArtifactError("pin-short", "pin %s has no authority.mtp_elements, which is the only "
                                         "authority this repo has for the draft block's size (%s)"
                                         % (path, MTP_AUTHORITY))
    return pin


# -----------------------------------------------------------------------------------------------
# the expected shape, derived — names, dims and element counts from committed sources
# -----------------------------------------------------------------------------------------------
def expected_records():
    """{gguf_name: {"elts","shape_hf","hf"}} from the safetensors index x config-derived shapes.

    Reuses `tools/bf16_census.py`'s `MAP_HF_GGUF` and `hf_shape` so there is one name map and one
    shape model in the repo. The fused -> split pair (`mlp.experts.gate_up_proj` ->
    `ffn_gate_exps` + `ffn_up_exps`) is the only place a GGUF record is half an HF tensor; each half
    must carry exactly half the elements, which is what makes `gate_up_half_swap` a checkable claim.
    """
    g = BC.geometry()
    idx = json.load(open(BC.IDX))
    out, unmodelled, draft_names = {}, [], []
    for name in idx["weight_map"]:
        if name.startswith("model.visual."):
            continue                                   # the artifact carries no vision tower either
        if name.startswith(DRAFT_NAME_PREFIX):
            draft_names.append(name)
            continue
        base = BC.strip_layer(name)
        glist = BC.MAP_HF_GGUF.get(base, [])
        sh = BC.hf_shape(name, g)
        if not glist or sh is None:
            unmodelled.append(name)
            continue
        li = re.match(r"^model\.language_model\.layers\.(\d+)\.", name)
        for gn in glist:
            real = gn.replace("N", li.group(1)) if "N" in gn else gn
            s = tuple(sh)
            if len(glist) == 2:                        # fused gate|up: each half is [E, I, H]
                e, two_i, h = sh
                if two_i % 2:
                    raise ArtifactError("odd-fused-dim",
                                        "%s is fused with an odd output dim %d; the halves would not "
                                        "be equal and this loader does not guess which way to round"
                                        % (name, two_i))
                s = (e, two_i // 2, h)
            elts = 1
            for x in s:
                elts *= int(x)
            out[real] = {"elts": elts, "shape_hf": s, "hf": name}
    if unmodelled:
        raise ArtifactError("unmodelled-names",
                            "cannot derive an expected shape for %d text names (first: %s) — a census "
                            "that silently drops what it cannot model reads low and still prints a "
                            "total (pitfalls/40 §empty-parse-PASS)"
                            % (len(unmodelled), unmodelled[0]))
    return out, sorted(draft_names)


PIN_HEADER_FIELDS = ("architecture", "n_tensors", "header_end", "data_start", "alignment",
                     "file_type", "quantization_version")


def pin_header(pin):
    """The pinned header fields, merged from the two committed places that carry them.

    `gguf_header` carries architecture / n_tensors / data_start / general.alignment /
    general.file_type / general.quantization_version; `header_end` lives in
    `audit_sections["audit-hash-2026-10-26.json"]`, the record of PLAN.md step 1.1. Both are read, and
    a field neither carries is a hard error rather than a field quietly left unchecked."""
    gg = pin.get("gguf_header", {})
    audit = (pin.get("audit_sections") or {}).get("audit-hash-2026-10-26.json", {})
    out, missing = {}, []
    for k in PIN_HEADER_FIELDS:
        for src in (k, "general." + k):
            if src in gg:
                out[k] = gg[src]
                break
        else:
            if k in audit:
                out[k] = audit[k]
            else:
                missing.append(k)
    if missing:
        raise ArtifactError("pin-short", "the pin %s carries no %s, so this loader cannot check that "
                                         "header field — and it is not checked anywhere else either, "
                                         "which is the failure being refused here"
                            % (PIN_PATH, ", ".join(missing)))
    return out


def role_of(name):
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def layer_of(name):
    m = re.match(r"^blk\.(\d+)\.", name) or re.match(r"^mtp\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def family_of_role(name):
    """Family via the census's own classifier, fed the HF name the file's role is paired with."""
    hf = HF_BY_ROLE.get(role_of(name))
    return BC.family_of(hf if hf else name)


def dims_match(actual, expected_hf):
    """True when a GGUF dim tuple is the HF shape reversed, modulo the singletons GGUF drops.

    `ssm_conv1d` is HF [qkv,1,kf] and GGUF [kf,qkv]; `ffn_gate_inp_shexp` is HF [1,H] and GGUF [H].
    Rank is therefore not the equality that matters — element count is, and it is checked separately
    and exactly, so widening this comparison cannot hide a real dim error."""
    a = tuple(int(x) for x in actual if int(x) != 1)
    e = tuple(int(x) for x in reversed(expected_hf) if int(x) != 1)
    return a == e


# -----------------------------------------------------------------------------------------------
# records
# -----------------------------------------------------------------------------------------------
class TensorRecord:
    """One tensor record plus the two byte facts every consumer needs.

    `offset` is the file's own number, bytes from the start of the data section. `file_offset` is
    computed as `data_start + offset` by this loader and then *compared* against a pread of the same
    range in `loader.py` — the staging check is what stops a header that lies about its offsets from
    producing a slab of plausible-looking garbage."""
    __slots__ = ("index", "name", "role", "layer", "dims", "shape_hf", "type_name", "n_elts",
                 "row", "offset", "file_offset", "source", "family", "hf_name")

    def __init__(self, index, name, dims, type_name, n_elts, row, offset, abs_base, source,
                 hf_name=None):
        self.index = index
        self.name = name
        self.role = role_of(name)
        self.layer = layer_of(name)
        self.dims = tuple(int(x) for x in dims)
        self.shape_hf = tuple(reversed(self.dims))
        self.type_name = type_name
        self.n_elts = int(n_elts)
        self.row = int(row)
        self.offset = int(offset)
        self.file_offset = abs_base + self.offset if offset is not None else None
        self.source = source                       # "gguf" | "hf_shards"
        self.hf_name = hf_name or HF_BY_ROLE.get(self.role, name)
        self.family = BC.family_of(self.hf_name)

    def n_blocks(self):
        """Whole ggml blocks this tensor occupies. A slice is addressable only at this granularity."""
        return -(-self.n_elts // GBG.block_elts(self.type_name))

    def block_bytes(self):
        return GBG.block_bytes(self.type_name)

    def __repr__(self):
        return "<%s %s %s elts=%d row=%d src=%s>" % (type(self).__name__, self.name, self.type_name,
                                                     self.n_elts, self.row, self.source)


class Artifact:
    """The pinned GGUF plus the extracted draft block, validated before anything can use it.

    Checks run in a fixed order and every one is named: `checks_run` records which ones actually
    executed, and `engine/gate.py` requires the printed count to equal the registered count, so a
    check that silently stopped running is a red result rather than a shorter list."""

    CHECKS = ("size-vs-pin", "header-fields", "record-count", "name-set", "element-counts",
              "dims", "rejected-spellings", "width-census", "down-proj-split", "draft-absent",
              "row-bytes-vs-ruler", "file-size-identity", "data-section-coverage", "draft-block",
              "sha256")

    def __init__(self, path=None, want_draft=True, verify_sha256=False, require_pin=True,
                 faults=()):
        self.path = path or RB.GGUF_PATH
        self.faults = set(faults)
        self.checks_run = []
        self.pin = load_pin() if require_pin else {}
        if not os.path.exists(self.path):
            raise ArtifactError("artifact-absent",
                                "the pinned artifact %s is absent; the engine has no weight source"
                                % self.path)
        self.size = os.path.getsize(self.path)
        self._hdr = gr.Gguf(self.path)          # header parse only; touches no data byte
        self.header_bytes_read = self._hdr.bytes_read
        self.meta = self._hdr.meta
        self.alignment = self._hdr.alignment
        self.data_start = self._hdr.data_start
        self.header_end = self._hdr.header_end
        self.n_tensors_declared = int(self._hdr.n_tensors)
        self.records = [TensorRecord(t.index, t.name, t.dims, t.type_name, t.n_elts, t.row,
                                     t.offset, self.data_start, "gguf")
                        for t in self._hdr.tensors]
        self._reindex()
        self.draft = self._read_draft() if want_draft else []
        self.sha256 = None
        self._apply_faults()
        self._run_checks(verify_sha256=verify_sha256)

    def _reindex(self):
        self.by_name = {r.name: r for r in self.records}

    def _run_checks(self, verify_sha256):
        steps = [("size-vs-pin", self._c_size),
                 ("header-fields", self._c_header),
                 ("record-count", self._c_record_count),
                 ("name-set", self._c_name_set),
                 ("element-counts", self._c_elements),
                 ("dims", self._c_dims),
                 ("rejected-spellings", self._c_rejected_spellings),
                 ("width-census", self._c_width_census),
                 ("down-proj-split", self._c_down_proj_split),
                 ("draft-absent", self._c_draft_absent),
                 ("row-bytes-vs-ruler", self._c_row_bytes_vs_ruler),
                 ("file-size-identity", self._c_file_size_identity),
                 ("data-section-coverage", self._c_data_coverage),
                 ("draft-block", self._c_draft_block)]
        if verify_sha256:
            steps.append(("sha256", self._c_sha256))
        names = [n for n, _ in steps]
        if tuple(names) != tuple(self.CHECKS if verify_sha256 else self.CHECKS[:-1]):
            raise ArtifactError("check-list-drift",
                                "the checks that would run (%s) are not the checks this module "
                                "advertises (%s); a gate cannot judge a list it has to guess"
                                % (names, self.CHECKS))
        for name, fn in steps:
            try:
                fn()
            except ArtifactError:
                self.checks_run.append(name)
                raise
            self.checks_run.append(name)

    # ---- the checks ------------------------------------------------------------------------
    def _c_size(self):
        if self.pin and self.size != int(self.pin["bytes_on_disk"]):
            raise ArtifactError("size", "%s is %d B; the pin says %d B. Different file, or the same "
                                        "name with different bytes."
                                % (os.path.basename(self.path), self.size, self.pin["bytes_on_disk"]))

    def _c_header(self):
        if not self.pin:
            return
        h = pin_header(self.pin)
        want = [("general.architecture", self.meta.get("general.architecture"), h["architecture"]),
                ("n_tensors", self.n_tensors_declared, h["n_tensors"]),
                ("header_end", self.header_end, h["header_end"]),
                ("data_start", self.data_start, h["data_start"]),
                ("general.alignment", self.alignment, h["alignment"]),
                ("general.file_type", self.meta.get("general.file_type"), h["file_type"]),
                ("general.quantization_version", self.meta.get("general.quantization_version"),
                 h["quantization_version"])]
        bad = [(k, got, exp) for k, got, exp in want if got != exp]
        if bad:
            raise ArtifactError("header", "%s does not match the committed pin %s on: %s. This is "
                                          "not the pinned artifact, or it has been replaced under the "
                                          "same name." % (os.path.basename(self.path), PIN_PATH,
                                                          "; ".join("%s: file %s, pin %s" % b
                                                                    for b in bad)))

    def _c_record_count(self):
        if len(self.records) != self.n_tensors_declared:
            raise ArtifactError("record-count", "parsed %d tensor records, the header declares %d; a "
                                                "partial parse must not print a table"
                                % (len(self.records), self.n_tensors_declared))
        if len(self.records) != EXPECTED_N_TENSORS:
            raise ArtifactError("record-count", "parsed %d tensor records, the derived name set has "
                                                "%d and the pin declares %s"
                                % (len(self.records), EXPECTED_N_TENSORS,
                                   self.pin.get("gguf_header", {}).get("n_tensors")
                                   if self.pin else "unpinned"))
        if len(self.by_name) != len(self.records):
            raise ArtifactError("record-count", "%d duplicate tensor names in the header"
                                % (len(self.records) - len(self.by_name)))

    def _c_name_set(self):
        self.expected, self.draft_index_names = expected_records()
        missing = sorted(set(self.expected) - set(self.by_name))
        extra = sorted(set(self.by_name) - set(self.expected))
        if missing or extra:
            raise ArtifactError("name-set",
                                "the file's tensor names do not match the set derived from the "
                                "safetensors index x the config's shapes (%d expected, %d parsed): "
                                "%d missing (first 3: %s), %d unexpected (first 3: %s). Wrong names "
                                "mean a wrong checkpoint or a wrong name map — see %s."
                                % (len(self.expected), len(self.by_name), len(missing), missing[:3],
                                   len(extra), extra[:3], WIDTH_MAP_EVIDENCE))

    def _c_elements(self):
        bad = []
        for r in self.records:
            e = self.expected[r.name]
            if r.n_elts != e["elts"]:
                bad.append("%s: file %d elts, derived %d from %s"
                           % (r.name, r.n_elts, e["elts"], e["hf"]))
        if bad:
            raise ArtifactError("element-counts", "element counts disagree with the derived shapes on "
                                                  "%d tensors:\n  %s" % (len(bad), "\n  ".join(bad[:6])))

    def _c_dims(self):
        bad = [r for r in self.records if not dims_match(r.dims, self.expected[r.name]["shape_hf"])]
        if bad:
            raise ArtifactError("dims", "dims disagree with the derived shapes (GGUF order = reversed "
                                        "HF order, singleton dims dropped) on %d tensors:\n  %s"
                                % (len(bad), "\n  ".join("%s: file %s, derived %s"
                                                         % (r.name, r.dims,
                                                            self.expected[r.name]["shape_hf"])
                                                         for r in bad[:6])))

    def _c_rejected_spellings(self):
        hits = sorted({r.name for r in self.records
                       for s in REJECT_NAME_SUBSTRINGS if s in r.name})
        if hits:
            raise ArtifactError("rejected-spellings",
                                "%d tensor name(s) use an HF-side spelling this checkpoint's GGUF "
                                "never carries: %s. This model spells GDN `attn_qkv`/`attn_gate`/"
                                "`ssm_out`, not `in_proj_*`/`linear_attn`/`out_proj`, and spells the "
                                "router `ffn_gate_inp`, not `mlp.gate` — a file that spells them the "
                                "HF way is not this artifact, and a header that yields them here is "
                                "being mis-parsed. Evidence: %s §why it exists."
                                % (len(hits), hits[:4], WIDTH_MAP_EVIDENCE))

    def _c_width_census(self):
        census = {}
        for r in self.records:
            census[r.type_name] = census.get(r.type_name, 0) + 1
        if census != EXPECTED_CENSUS:
            raise ArtifactError("width-census", "the stored-width census is %s; %s §family table pins "
                                                "%s. A census that disagrees is a different file, not "
                                                "a different reading of this one."
                                % (census, WIDTH_MAP_EVIDENCE, EXPECTED_CENSUS))
        unknown = sorted({r.role for r in self.records} - set(EXPECTED_TYPE_BY_ROLE))
        if unknown:
            raise ArtifactError("width-census", "%d tensor role(s) have no expected width, first: %s. "
                                                "An unmapped family priced at zero is how a census "
                                                "reads low and still prints a total."
                                % (len(unknown), unknown[:4]))
        wrong = ["%s (%s): stored %s, expected one of %s"
                 % (r.name, r.role, r.type_name, sorted(EXPECTED_TYPE_BY_ROLE[r.role]))
                 for r in self.records if r.type_name not in EXPECTED_TYPE_BY_ROLE[r.role]]
        if wrong:
            raise ArtifactError("width-census", "%d tensor(s) carry a width the published width map "
                                                "does not:\n  %s" % (len(wrong), "\n  ".join(wrong[:6])))

    def _c_down_proj_split(self):
        downs = [r for r in self.records if r.role == "blk.N.ffn_down_exps.weight"]
        by_type = {}
        for r in downs:
            by_type.setdefault(r.type_name, []).append(r.layer)
        if sorted(len(v) for v in by_type.values()) != [3, 37]:
            raise ArtifactError("down-proj-split",
                                "ffn_down_exps splits %s; the published width map says Q5_K x37 and "
                                "Q6_K x3 (%s). Reading the group as one width changes the resident "
                                "total by 3 x (176-144)/176 of its bytes and still closes against "
                                "itself, which is why it is checked here and not assumed."
                                % ({k: sorted(v) for k, v in sorted(by_type.items())},
                                   WIDTH_MAP_EVIDENCE))
        self.down_proj_q6_layers = sorted(by_type.get("Q6_K", []))
        self.down_proj_q5_layers = sorted(by_type.get("Q5_K", []))

    def _c_draft_absent(self):
        hits = [r.name for r in self.records if r.name.startswith(DRAFT_NAME_PREFIX)]
        self.draft_tensors_in_gguf = len(hits)
        if hits:
            raise ArtifactError("draft-absent",
                                "the published GGUF carries %d mtp.* tensors (first 3: %s). Every "
                                "committed source says it carries zero (%s; the witness \"GGUF carries "
                                "no draft tensors\" in quantization/residency-layout.md §6), and the "
                                "engine's draft bytes are sourced from the HF shards on that basis."
                                % (len(hits), hits[:3], WIDTH_MAP_EVIDENCE))

    def _c_row_bytes_vs_ruler(self):
        rows = [(r.name, r.dims, TYPE_IDS[r.type_name]) for r in self.records]
        fam, total, unmapped = GWT.family_bytes(rows, GWT.BLK)
        if unmapped:
            raise ArtifactError("row-bytes-vs-ruler", "unmapped ggml types %s would be counted as "
                                                      "zero bytes" % sorted(unmapped))
        self.width_table_family = {k: tuple(v) for k, v in fam.items()}
        mine = sum(r.row for r in self.records)
        if total != mine:
            raise ArtifactError("row-bytes-vs-ruler",
                                "the independent ruler tools/gguf_width_table.py totals %d B for "
                                "these records and this loader totals %d B (delta %+d B). Two parsers "
                                "and two block tables, one header: check tools/ggml_block_geom.py, do "
                                "not widen anything." % (total, mine, total - mine))

    def _c_file_size_identity(self):
        self.weight_bytes = sum(r.row for r in self.records)
        if self.weight_bytes + self.data_start != self.size:
            raise ArtifactError("file-size-identity",
                                "sum(row_bytes) %d + data_start %d = %d, file size %d, off by %+d B. "
                                "This identity is the witness that every block size in use is right "
                                "(tools/gguf_width_table.py §THE WITNESS); it does not round."
                                % (self.weight_bytes, self.data_start,
                                   self.weight_bytes + self.data_start, self.size,
                                   self.size - (self.weight_bytes + self.data_start)))

    def _c_data_coverage(self):
        srt = sorted(self.records, key=lambda r: r.offset)
        end, gaps, overlap, unaligned = 0, 0, [], []
        for r in srt:
            if r.offset < end:
                overlap.append("%s at %d overlaps the previous record, which ends at %d"
                               % (r.name, r.offset, end))
            elif (r.offset - end) % self.alignment:
                unaligned.append("%s starts at %d, %d B past the previous record: not a multiple of "
                                 "the file's alignment (%d)"
                                 % (r.name, r.offset, r.offset - end, self.alignment))
            gaps += max(0, r.offset - end)
            end = max(end, r.offset + r.row)
        if overlap:
            raise ArtifactError("data-section-coverage", "tensor records overlap in the data section: "
                                                         "%s" % overlap[:3])
        if unaligned:
            raise ArtifactError("data-section-coverage", "%d record(s) sit at an unaligned offset, "
                                                         "first: %s" % (len(unaligned), unaligned[0]))
        self.data_gap_bytes = gaps
        self.data_end = end
        if self.data_start + end != self.size:
            raise ArtifactError("data-section-coverage",
                                "the last record ends at %d (absolute %d) but the file is %d B: %d "
                                "bytes are claimed by no record"
                                % (end, self.data_start + end, self.size,
                                   self.size - (self.data_start + end)))

    def _c_draft_block(self):
        if not self.draft:
            raise ArtifactError("draft-block", "no draft block was loaded. The GGUF carries zero mtp.* "
                                               "tensors, so without it the engine has no MTP weights at "
                                               "all and D-05 is silently unimplemented.")
        elts = sum(r.n_elts for r in self.draft)
        if self.pin:
            want = int(self.pin["authority"]["mtp_elements"])
            if elts != want:
                raise ArtifactError("draft-block",
                                    "the draft extract has %d elements; %s says %d. Delta %+d "
                                    "elements = %+d B bf16."
                                    % (elts, MTP_AUTHORITY, want, elts - want, (elts - want) * 2))
        if len(self.draft) != EXPECTED_DRAFT_TENSORS:
            raise ArtifactError("draft-block", "the draft extract carries %d tensors; PLAN.md step 1.2 "
                                               "and quantization/residency-layout.md §2 pin 19"
                                % len(self.draft))
        if sorted(r.name for r in self.draft) != self.draft_index_names:
            raise ArtifactError("draft-block", "the draft extract's names are not the %d mtp.* names "
                                               "the safetensors index carries (missing %s, extra %s)"
                                % (len(self.draft_index_names),
                                   sorted(set(self.draft_index_names) - {r.name for r in self.draft})[:3],
                                   sorted({r.name for r in self.draft} - set(self.draft_index_names))[:3]))
        for r in self.draft:
            if not r.name.startswith(DRAFT_NAME_PREFIX):
                raise ArtifactError("draft-block", "%s is in the draft extract but is not an mtp.* "
                                                   "tensor" % r.name)
        self.draft_bytes = sum(r.row for r in self.draft)

    def _c_sha256(self):
        if not self.pin:
            return
        want = self.pin["sha256"]
        got = stream_sha256(self.path)
        self.sha256 = got
        if got != want:
            raise ArtifactError("sha256", "sha256 mismatch on %s: file is %s, the pin (recorded %s) "
                                          "says %s. A silent re-upload under the same name is exactly "
                                          "what the pin exists to catch."
                                % (os.path.basename(self.path), got, self.pin.get("recorded_utc"),
                                   want))

    # ---- draft block -----------------------------------------------------------------------
    def _read_draft(self):
        """The 19 mtp.* tensors, from the HF extract — the GGUF has none of them.

        Byte geometry comes from the safetensors header's own `data_offsets`, and the row bytes are
        re-derived from `tools/ggml_block_geom.py` and required to agree: a safetensors file is only
        trusted to say where its bytes are, not how big a quantized block is."""
        path = RB.MTP_PATH
        if not os.path.exists(path):
            raise ArtifactError("draft-block",
                                "the draft block %s is absent, and the published GGUF carries zero "
                                "mtp.* tensors — so this file is the engine's only draft-block source "
                                "(quantization/residency-layout.md §2)." % path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        recs, bad_span = [], []
        for i, (name, d) in enumerate(sorted(hdr.items())):
            if name == "__metadata__":
                continue
            dims = tuple(d["shape"])
            elts = 1
            for x in dims:
                elts *= int(x)
            tname = {"BF16": "BF16", "F32": "F32", "F16": "F16"}.get(d["dtype"])
            if tname is None:
                raise ArtifactError("draft-block", "%s says %s is %s; this loader will not price a "
                                                   "dtype it has to guess" % (path, name, d["dtype"]))
            lo, hi = d["data_offsets"]
            row = GBG.row_bytes(elts, tname)
            if hi - lo != row:
                bad_span.append("%s: declared span %d B, derived row %d B at %d elts of %s"
                                % (name, hi - lo, row, elts, tname))
            r = TensorRecord(i, name, dims, tname, elts, row, lo, base, "hf_shards",
                             hf_name=name)
            recs.append(r)
        if bad_span:
            raise ArtifactError("draft-block", "the draft extract's declared byte spans do not match "
                                               "the derived ggml row sizes:\n  %s"
                                % "\n  ".join(bad_span[:4]))
        return recs

    # ---- injections ------------------------------------------------------------------------
    def _apply_faults(self):
        for f in sorted(self.faults):
            fn = FAULTS.get(f)
            if fn is None:
                raise ArtifactError("unknown-fault", "unknown fault %r; known: %s"
                                    % (f, sorted(FAULTS)))
            fn(self)
            self._reindex()


# Each fault perturbs THIS object's view of the file in exactly one way — the convention
# tools/residency_budget.py and tools/quant_scenarios.py use, so the fault hook is the only way a
# number can move and engine/gate.py can require the named check to notice.
def _f_rename_gdn(a):
    r = a.by_name["blk.0.attn_qkv.weight"]
    r.name = "blk.0.in_proj_qkv.weight"


def _f_hf_spelling(a):
    r = a.by_name["blk.0.ssm_out.weight"]
    r.name = "blk.0.linear_attn.out_proj.weight"


def _f_add_draft_tensor(a):
    r = a.records[0]
    a.records.append(TensorRecord(len(a.records), "mtp.layers.0.ffn_gate_exps.weight", r.dims,
                                  "Q4_K", r.n_elts, r.row, 0, a.data_start, "gguf"))


def _f_group_width(a):
    """The trap named in the brief: read the down_proj group as ONE width.

    Implemented as the loader mistake rather than as a typo — every Q6_K down_proj is relabelled and
    re-priced as Q5_K, which is internally consistent and therefore invisible to every closure except
    the one that knows the published split."""
    for r in a.records:
        if r.role == "blk.N.ffn_down_exps.weight":
            r.type_name = "Q5_K"
            r.row = GBG.row_bytes(r.n_elts, r.type_name)


def _f_stale_q5k_bpw(a):
    """This repo's own historical width error, re-injected: Q5_K at 5.5625 bpw where the pinned struct
    is 176 B / 256 elts = 5.5. quantization/evidence/audit-block-geometry-2026-10-26.md is the record
    of what the stale row cost (74.00 MiB on these very tensors)."""
    for r in a.records:
        if r.type_name == "Q5_K":
            r.row = -(-r.n_elts // 256) * 178


def _f_short_parse(a):
    a.records.pop()


def _f_lm_head_width(a):
    r = a.by_name["output.weight"]
    r.type_name = "Q8_0"
    r.row = GBG.row_bytes(r.n_elts, "Q8_0")


def _f_draft_from_gguf(a):
    """Assume the draft bytes come from the GGUF, which carries none of them (the D-05 gap)."""
    a.draft = []


def _f_offset_shift(a):
    """A header whose offsets lie by one byte — the misalignment the coverage check is for."""
    for r in a.records:
        r.offset += 1


FAULTS = {
    "rename_gdn_tensor": _f_rename_gdn,
    "hf_spelling_in_gguf": _f_hf_spelling,
    "add_draft_tensor": _f_add_draft_tensor,
    "down_proj_group_width": _f_group_width,
    "stale_q5k_bpw": _f_stale_q5k_bpw,
    "short_parse": _f_short_parse,
    "lm_head_width": _f_lm_head_width,
    "draft_from_gguf": _f_draft_from_gguf,
    "offset_shift_1": _f_offset_shift,
}


def stream_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def open_artifact(**kw):
    """The one entry point: validate, then let the caller have the records."""
    return Artifact(**kw)


def main(argv):
    faults = [f for f in (argv[argv.index("--fault") + 1].split(",") if "--fault" in argv else []) if f]
    try:
        a = open_artifact(verify_sha256="--sha256" in argv, faults=faults)
    except ArtifactError as ex:
        # The refusal prints the check that fired and the tensor or field it fired on, then exits 1:
        # engine/gate.py requires a *named* check per injection, so "it raised" is not enough.
        print("ARTIFACT REFUSED tag=%s fault=%s\n%s" % (ex.tag, ",".join(faults) or "-", ex.message))
        return 1
    print("artifact      %s" % os.path.relpath(a.path, REPO))
    print("size          %d B (pin %s)" % (a.size, a.pin.get("bytes_on_disk")))
    print("header        version=%d header_end=%d data_start=%d alignment=%d"
          % (a._hdr.version, a.header_end, a.data_start, a.alignment))
    print("metadata      arch=%s file_type=%s quant_version=%s block_count=%s"
          % (a.meta.get("general.architecture"), a.meta.get("general.file_type"),
             a.meta.get("general.quantization_version"), a.meta.get("qwen35moe.block_count")))
    print("tensors       parsed=%d declared=%d expected=%d   checks run=%d/%d"
          % (len(a.records), a.n_tensors_declared, EXPECTED_N_TENSORS, len(a.checks_run),
             len(a.CHECKS) - 1 + (1 if a.sha256 else 0)))
    print("data section  sum(row_bytes)=%d gaps=%d last_end=%d -> file %d B (delta %d B)"
          % (a.weight_bytes, a.data_gap_bytes, a.data_end, a.size,
             a.size - (a.data_start + a.data_end)))
    print("\nstored-width census (file vs %s):" % WIDTH_MAP_EVIDENCE)
    print("%-8s %8s %8s %16s %12s" % ("type", "count", "pin", "bytes", "GiB"))
    for t in sorted(a.width_table_family, key=lambda k: -a.width_table_family[k][1]):
        n, b = a.width_table_family[t]
        print("%-8s %8d %8d %16d %12.4f" % (t, n, EXPECTED_CENSUS.get(t, -1), b, b / 2.0 ** 30))
    print("%-8s %8d %8d %16d %12.4f" % ("TOTAL", len(a.records), EXPECTED_N_TENSORS, a.weight_bytes,
                                        a.weight_bytes / 2.0 ** 30))
    print("\ndown_proj width is per-tensor: Q5_K on %d layers, Q6_K on layers %s"
          % (len(a.down_proj_q5_layers), a.down_proj_q6_layers))
    print("draft block   %d tensors, %d elements (pin authority %s), %d B bf16 on disk, from %s"
          % (len(a.draft), sum(r.n_elts for r in a.draft),
             a.pin.get("authority", {}).get("mtp_elements"), a.draft_bytes,
             os.path.relpath(RB.MTP_PATH, REPO)))
    print("bytes read    %d B (header only; the data section is opened only by loader.py, on request)"
          % a.header_bytes_read)
    print("ARTIFACT checks=%d/%d faults=%d" % (len(a.checks_run), len(a.CHECKS) - (0 if a.sha256 else 1),
                                               len(faults)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
