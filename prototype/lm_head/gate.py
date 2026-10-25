#!/usr/bin/env python3
"""gate.py — the judge for prototype/lm_head. It never touches the GPU lock and never touches a timer.

    ./gate --selftest            host-only controls: the instruments that must be able to fail
    ./gate --check-only          population + oracle + host arm + all judgements, no device, no lock
    ./gate                       the same, plus the device arm and the three-arm agreement
    ./gate --control NAME        a deliberately broken run (used by --selftest; must exit non-zero)
    ./gate --max-rows N          cut the SAMPLED row count only; no class, mode or assertion moves

`gate` (the shell wrapper) is the only acquirer of the repo's GPU lock tool, and it greps this file for
any mention of that tool at run time: a driver that self-acquires and is called by a caller already
holding the lock parks forever with no CPU, no GPU, no message and no exit
(`pitfalls/50-harness-and-shell.md` §"Self-locking gate drivers deadlock silently").

WHAT IS JUDGED, AND BY WHOM
  * each arm's words against the oracle's words, inside the band the DATA derives
    (gamma_K * sum_k |h_k||w_k|, K = terms summed, u = 2^-24) — `BAND`;
  * bit identity, zero tolerance, between cases that share an association — `EXACT`:
    batched vs per-row, gather vs dense, tied role A vs tied role B;
  * injections: `FIRES` must move at least one word FURTHER than that band from its honest twin, and
    `IMMUNE` must leave every word bit-identical to it. Which of the two a case is is DERIVED from the
    materialised bytes here (`derive_class`), compared against what the plan declares, and a
    disagreement is a finding. An immunity true by construction is not evidence, so every mode with an
    immune arm must also have a firing arm somewhere in the same plan (`ALTREACH`);
  * `BYTES` cases are judged on the read-set: bytes/loads/xorb/vsum/bsum/min/max/cross/peer_hits/lanes
    are recomputed here from the slab bytes — the audited decoder for values, this file's own address
    derivation — and required to equal what the driver printed. No counter is believed on the
    driver's authority;
  * host arm vs device arm vs oracle: the three-arm agreement, per class.

WHAT EACH DIGEST PROVABLY SEES (results.md carries this table; one of them is provably blind):
  oxor  XOR of the f32 bit images this case wrote        blind to any uniform XOR-cancelling change
  osum  real sum of those words                          blind to a permutation of them
  xorb  XOR of the decoded weight elements it loaded     BLIND to a uniform power-of-two rescale
                                                         (measured: d=0.5 and d=1.0 both give 0xFF400000,
                                                         tmp/worker-lm-head/probe10_digest.cpp)
  vsum  INTEGER SUM of those same bit images             sees that rescale; carries the value verdict
  bsum  sum of the RAW STORAGE byte values               sees layout, blind to any decode change
Every parser asserts parsed == printed, and `--selftest` refuses a run whose judgement cannot go red.
"""
import argparse
import ast
import json
import math
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRIVER = os.path.join(HERE, "lm_head")
PROBE = os.path.join(HERE, "host_probe")
ORACLE = os.path.join(HERE, "oracle_arm.py")
KERNELS_H = os.path.join(HERE, "lm_head_kernels.h")
PLAN_H = os.path.join(HERE, "lm_head_plan.h")
PY = "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python"
CONFIG = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.config.json")
INDEX = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.safetensors.index.json")
SHARD_DIR = os.path.join(REPO, "models", "qwen36-src")
GGUF = os.path.join(REPO, "models", "unsloth-q4km", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DUMP_FILE = "layer39.combined_out.bin"          # [32, 2048] bf16, the pinned dump
DUMP_DIR = os.path.join(REPO, "data", "dumps", "2026-10-26")
LMH_TENSOR = "lm_head.weight"
EMBD_TENSOR = "model.language_model.embed_tokens.weight"

sys.path.insert(0, os.path.join(REPO, "tools"))
import gguf_reader as GR                                       # audited decoder, used for the recount


class Fail(Exception):
    pass


def run(cmd, log=None):
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if log:
        with open(log, "w") as fh:
            fh.write(p.stdout + (("\n" + p.stderr) if p.stderr.strip() else ""))
    return p.returncode, p.stdout


def kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def parse_tag(text, tag, keys, want=None):
    """key=value records for a line tag. The value is any non-space token, so a word-valued field can
    never be silently dropped (pitfalls/40 §"a key=value parser whose value pattern is numeric")."""
    recs = []
    for line in text.splitlines():
        if line.startswith(tag + " ") or line == tag:
            d = kv(line)
            if not all(k in d for k in keys):
                raise Fail("line tagged %s is missing %s: %r" % (tag, keys, line))
            recs.append(d)
    if want is not None and len(recs) != want:
        raise Fail("parsed %d %s records, expected %d (a parser that matches nothing prints PASS)"
                   % (len(recs), tag, want))
    if not recs:
        raise Fail("no %s records at all: this run adjudicated nothing" % tag)
    return recs


# ---------------------------------------------------------------------------------------------
# the contract, scraped from the C++ the drivers compile. Nothing here is typed: a renumbered mode or
# an added report column has to move in one commit with the gate or this file says so.
# ---------------------------------------------------------------------------------------------
def contract():
    src = open(KERNELS_H).read()
    plan = open(PLAN_H).read()
    fc = {}
    for tag in ("LMH_F_P", "LMH_F_T", "LMH_F_C", "LMH_F_R", "LMH_BLOCK", "LMH_SPLIT",
                "LMH_MAX_BATCH", "LMH_Q8_0_ELTS", "LMH_Q8_0_BYTES", "LMH_BF16_BYTES",
                "LMH_NAN_BITS", "LMH_PAD_BYTE"):
        m = re.search(r"#define\s+%s\s+(0x[0-9A-Fa-f]+|\d+)" % tag, plan + "\n" + src)
        if not m:
            raise Fail("no #define %s with a numeric value in the lane headers" % tag)
        fc[tag] = int(m.group(1), 0)
    en = {}
    for body in re.findall(r"enum\s*[A-Za-z_]*\s*\{([^}]*)\}", src, re.S):
        for name, val in re.findall(r"\b(LMH_[A-Z0-9_]+)\s*=\s*(\d+)", body):
            en[name] = int(val)
    for want in ("LMH_ENC_BF16", "LMH_ENC_Q8_0", "LMH_ROLE_LMHEAD", "LMH_ROLE_EMBD",
                 "LMH_ROLE_LMHEAD_Q8", "LMH_ROLE_TIED_LMHEAD", "LMH_ROLE_TIED_EMBD",
                 "LMH_ROLE_SYNTH_DUP", "LMH_PATH_DENSE", "LMH_PATH_GATHER", "LMH_BATCH_BATCHED",
                 "LMH_BATCH_PERROW", "LMH_ASSOC_SEQ", "LMH_ASSOC_REV", "LMH_ASSOC_SPLIT8",
                 "LMH_MODE_CORRECT", "LMH_MODE_GATHER_ROW_OFF_1", "LMH_MODE_BATCH_SLOT_STRIDE",
                 "LMH_MODE_TIED_ALIAS", "LMH_MODE_ROW_STRIDE_HEAD_DIM", "LMH_MODE_HALF_ACCUM",
                 "LMH_MODE_DROP_LAST_K", "LMH_MODE_Q8_DROP_SCALE", "LMH_MODE_CROSS_TABLE_READ",
                 "LMH_MODE_BATCH_RELOOPS", "LMH_MODE_COUNT", "LMH_EXP_EXACT", "LMH_EXP_BAND",
                 "LMH_EXP_FIRES", "LMH_EXP_IMMUNE", "LMH_EXP_BYTES"):
        if want not in en:
            raise Fail("enum name %s not found in %s" % (want, KERNELS_H))
    m = re.search(r"lmh_r_fields\[\]\s*=\s*\{([^}]*)\}", plan, re.S)
    if not m:
        raise Fail("no lmh_r_fields[] array in lm_head_plan.h: the report layout has no single source")
    fields = re.findall(r"\"([^\"]+)\"", m.group(1))
    if len(fields) != fc["LMH_F_R"] - 1:
        raise Fail("lmh_r_fields has %d names, LMH_F_R says %d data fields"
                   % (len(fields), fc["LMH_F_R"] - 1))
    # The canonical text form of each field is scraped from the SAME header the writer compiles, so
    # producer and parser cannot each hold a ruler: lmh_r_fmt[] is the one spelling of that rule.
    m = re.search(r"lmh_r_fmt\[\]\s*=\s*\{([^}]*)\}", plan, re.S)
    if not m:
        raise Fail("no lmh_r_fmt[] array in lm_head_plan.h: no report field has a declared text form")
    kinds = dict(re.findall(r"#define\s+(LMH_RF_[A-Z]+)\s+\"([a-z]+)\"", plan))
    fmts = []
    for tok in re.findall(r"LMH_RF_[A-Z]+|\"[a-z]+\"", m.group(1)):
        if tok.startswith("LMH_RF_"):
            if tok not in kinds:
                raise Fail("lmh_r_fmt[] uses %s, which has no #define in lm_head_plan.h" % tok)
            f = kinds[tok]
        else:
            f = tok.strip('"')
        if f not in ("dec", "hex", "real"):
            raise Fail("lmh_r_fmt[] declares form %r, which is not dec/hex/real" % f)
        fmts.append(f)
    if len(fmts) != len(fields):
        raise Fail("lmh_r_fmt has %d forms, lmh_r_fields has %d names" % (len(fmts), len(fields)))
    if "hex" not in fmts:
        raise Fail("no report field is declared a bit image: the digest columns would have no ruler")
    modes = sorted(n for n in en if n.startswith("LMH_MODE_") and n != "LMH_MODE_COUNT"
                   and n != "LMH_MODE_CORRECT")
    return fc, en, fields, modes, fmts


def mode_token(mode_name):
    """LMH_MODE_HALF_ACCUM -> 'half-accum': the fragment a driver self-test line must contain for that
    mode. Derived from the header enum, so a mode nobody wrote a witness for is a red check rather than
    a hole in a hand-typed list."""
    return mode_name[len("LMH_MODE_"):].lower().replace("_", "-")


def mode_token_id(mid, en):
    """The same fragment from the numeric id a plan record carries. The ids come from the plan and the
    names from the header; this is the only place that translates, so a renumbered enum moves both."""
    for k in sorted(en):
        if k.startswith("LMH_MODE_") and k != "LMH_MODE_COUNT" and en[k] == mid:
            return mode_token(k)
    raise Fail("no LMH_MODE_* name in %s has id %d" % (KERNELS_H, mid))


def read_config():
    cfg = json.load(open(CONFIG))
    tc = cfg["text_config"]
    if tc.get("tie_word_embeddings") or cfg.get("tie_word_embeddings"):
        raise Fail("the config says the tables are TIED; this lane exists to test the untied case "
                   "(README.md prediction 5) and refuses to gate a tied model by accident")
    return dict(h=tc["hidden_size"], head_dim=tc["head_dim"], vocab=tc["vocab_size"],
                n_layers=tc["num_hidden_layers"])


# ---------------------------------------------------------------------------------------------
# the population. One builder, one spelling of the geometry; both drivers, the oracle and every
# judgement read what this writes rather than each re-deciding it.
# ---------------------------------------------------------------------------------------------
class Pop(object):
    def __init__(self, cfg, fc, en):
        self.cfg, self.fc, self.en = cfg, fc, en
        self.tables = []
        self.cases = []
        self.sources = []
        self.ids = []
        self._idmap = {}
        self.out_cursor = 0
        self.n_hidden = 0
        self.hidden_spec = []
        self.slab_bytes = 0

    def add_table(self, enc, role, real, v_rows, peer, kind, file=None, tensor=None, rows=None,
                  nbytes=None, extra=None):
        idx = len(self.tables)
        h = self.cfg["h"]
        if enc == self.en["LMH_ENC_Q8_0"]:
            row_b = (h // self.fc["LMH_Q8_0_ELTS"]) * self.fc["LMH_Q8_0_BYTES"]
        else:
            row_b = h * self.fc["LMH_BF16_BYTES"]
        t = dict(idx=idx, enc=enc, role=role, base=-1,
                 nbytes=nbytes if nbytes is not None else v_rows * row_b, v_rows=v_rows, h=h,
                 peer=peer, real=real, row_b=row_b, kind=kind, file=file, tensor=tensor,
                 rows=list(rows or []), extra=dict(extra or {}))
        self.tables.append(t)
        src = dict(idx=idx, kind=kind, enc=("Q8_0" if enc == self.en["LMH_ENC_Q8_0"] else "BF16"),
                   real=real)
        if file:
            src.update(file=file, tensor=tensor, rows=",".join(str(r) for r in t["rows"]))
        for k, v in t["extra"].items():
            src[k] = v
        self.sources.append(src)
        return idx

    def place_tables(self):
        """Assign slab bases: every table's range is disjoint from every other unless it is the tied
        control, which is deliberately the SAME bytes (lmh_verify_plan refuses a partial overlap)."""
        at = 0
        for t in self.tables:
            if t["kind"] == "tied_copy":
                continue
            t["base"] = at
            at += t["nbytes"]
            if at % 64:                                   # keep rows on a cache-line boundary
                at += 64 - (at % 64)
        for t in self.tables:
            if t["kind"] != "tied_copy":
                continue
            src = self.tables[t["extra"]["from_idx"]]
            t["base"], t["nbytes"] = src["base"], src["nbytes"]
            t["extra"]["from_base"], t["extra"]["nbytes"] = src["base"], src["nbytes"]
            self.sources[t["idx"]]["from_base"] = src["base"]
            self.sources[t["idx"]]["nbytes"] = src["nbytes"]
        return at

    def slot(self, c):
        return self.cfg["h"] if c["mode"] == self.en["LMH_MODE_BATCH_SLOT_STRIDE"] else c["n_out"]

    def span(self, c):
        return (c["bcount"] - 1) * self.slot(c) + c["n_out"]

    def add_case(self, name, table, path, batching, b0, bcount, assoc, mode, expect,
                 row_lo=0, nrows=0, ids=None, cmp_off=-1, grp=-1, on_device=1, trace=1):
        c = dict(idx=len(self.cases), name=name, table=table, path=path, batching=batching,
                 b0=b0, bcount=bcount,
                 row_lo=row_lo, nrows=nrows, ids_off=-1, n_ids=0, assoc=assoc, mode=mode,
                 out_off=self.out_cursor, cmp_off=cmp_off, grp=grp, expect_name=expect,
                 on_device=on_device, trace=trace)
        if path == self.en["LMH_PATH_GATHER"]:
            key = tuple(ids)
            if key in self._idmap:
                off = self._idmap[key]
            else:
                off = len(self.ids)
                self._idmap[key] = off
                self.ids.extend(ids)
            c["ids_off"], c["n_ids"], c["n_out"] = off, len(ids), len(ids)
        else:
            c["n_out"] = nrows
        self.cases.append(c)
        self.out_cursor += self.span(c)
        return c

    def geo_key(self, c):
        return (c["table"], c["path"], c["batching"], c["b0"], c["bcount"], c["row_lo"], c["nrows"],
                c["ids_off"], c["n_ids"], c["assoc"])

    def honest_twin(self, c):
        """The case that is this one in every respect except the injection. Keyed on the geometry, so
        an injection can never be judged against a different piece of work."""
        key = self.geo_key(c)
        for o in self.cases:
            if o["mode"] == self.en["LMH_MODE_CORRECT"] and self.geo_key(o) == key:
                return o
        return None

    @property
    def out_floats(self):
        return self.out_cursor


def _wbits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def synth_bf16_bits(h, rows):
    """The synthetic bf16 rows, as hex. Every value is required to be a bf16 value, so the
    HALF_ACCUM derivation below has something exact to be immune about.

    The bf16 image is the HIGH half of the f32 image — a bf16 value is an f32 whose low 16 mantissa
    bits are zero, so writing `bits & 0xFF` and `bits >> 8` writes two zero bytes for every legal
    value. That is how this function shipped a table of all zeros while still passing its own
    is-a-bf16-value test; the payload's non-zero witness at the end is what makes the encoding
    checked rather than assumed."""
    out = bytearray(len(rows) * h * 2)
    for ri, row in enumerate(rows):
        for k, v in row:
            bits = _wbits(v)
            if bits & 0xFFFF:
                raise Fail("synthetic weight %r at row %d k=%d is not a bf16 value" % (v, ri, k))
            at = ri * h * 2 + k * 2
            out[at] = (bits >> 16) & 0xFF
            out[at + 1] = (bits >> 24) & 0xFF
    if not any(out):
        raise Fail("synth_bf16_bits produced an all-zero payload: every case over it would be a "
                   "zero-output case, which is a hollow check rather than a cheap one")
    return bytes(out).hex()


def synth_q8_bits(fc, h, rows):
    """The synthetic Q8_0 rows, packed through the same documented packing oracle_arm.py uses and
    independently required (there) to dequantize to what tools/gguf_reader's audited packer produces."""
    payload = bytearray()
    for row in rows:
        vals = [0.0] * h
        for k, v in row:
            vals[k] = float(v)
        for i in range(h // fc["LMH_Q8_0_ELTS"]):
            blk = vals[i * fc["LMH_Q8_0_ELTS"]:(i + 1) * fc["LMH_Q8_0_ELTS"]]
            amax = max(abs(v) for v in blk)
            d = struct.unpack("<e", struct.pack("<e", amax / 127.0))[0]
            payload += struct.pack("<e", d)
            qs = []
            for v in blk:
                if d == 0.0:
                    qs.append(0)
                    continue
                x = v / d
                q = math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)
                qs.append(max(-127, min(127, q)))
            payload += struct.pack("<32b", *qs)
    return bytes(payload).hex()


# ---------------------------------------------------------------------------------------------
# the population itself
# ---------------------------------------------------------------------------------------------
def build(cfg, fc, en, max_rows, deep_rows):
    """One list of cases, built from the config and the files. `max_rows` cuts the SAMPLED row count
    and nothing else: no class, no mode and no assertion is removed by the cut (the cut is printed)."""
    p = Pop(cfg, fc, en)
    H, hd = cfg["h"], cfg["head_dim"]
    BATCH = 4                                        # the bcount of every batched case, and the number
    #                                                of copies each synthetic hidden shape gets below
    path = os.path.join(DUMP_DIR, DUMP_FILE)
    sz = os.path.getsize(path)
    if sz % (H * 2):
        raise Fail("%s is %d B, not a whole number of %d-element bf16 rows" % (path, sz, H))
    n_real = sz // (H * 2)

    idx = json.load(open(INDEX))
    lmh_shard = os.path.join(SHARD_DIR, idx["weight_map"][LMH_TENSOR])
    emb_shard = os.path.join(SHARD_DIR, idx["weight_map"][EMBD_TENSOR])
    if not os.path.exists(lmh_shard) or not os.path.exists(emb_shard):
        raise Fail("a source shard is missing: %s | %s" % (lmh_shard, emb_shard))
    if os.path.basename(lmh_shard) == os.path.basename(emb_shard):
        raise Fail("the untiedness this lane tests has gone: %s and %s live in ONE shard"
                   % (LMH_TENSOR, EMBD_TENSOR))
    rows = list(range(deep_rows))
    t_lmh = p.add_table(en["LMH_ENC_BF16"], en["LMH_ROLE_LMHEAD"], 1, deep_rows, -1, "real",
                        file=lmh_shard, tensor=LMH_TENSOR, rows=rows)
    t_emb = p.add_table(en["LMH_ENC_BF16"], en["LMH_ROLE_EMBD"], 1, deep_rows, -1, "real",
                        file=emb_shard, tensor=EMBD_TENSOR, rows=rows)
    t_lmh_q8 = p.add_table(en["LMH_ENC_Q8_0"], en["LMH_ROLE_LMHEAD_Q8"], 1, deep_rows, -1, "real",
                           file=lmh_shard, tensor=LMH_TENSOR, rows=rows)
    t_emb_q8 = p.add_table(en["LMH_ENC_Q8_0"], en["LMH_ROLE_EMBD"], 1, deep_rows, -1, "real",
                           file=emb_shard, tensor=EMBD_TENSOR, rows=rows)
    p.tables[t_lmh]["peer"], p.tables[t_emb]["peer"] = t_emb, t_lmh
    p.tables[t_lmh_q8]["peer"], p.tables[t_emb_q8]["peer"] = t_emb_q8, t_lmh_q8
    # the tied control: two roles over ONE buffer, which is the assumption the batching precedent
    # was validated on and the thing an untied model must not be allowed to keep
    t_tied_a = p.add_table(en["LMH_ENC_BF16"], en["LMH_ROLE_TIED_LMHEAD"], 1, deep_rows, -1, "real",
                           file=lmh_shard, tensor=LMH_TENSOR, rows=rows)
    t_tied_b = p.add_table(en["LMH_ENC_BF16"], en["LMH_ROLE_TIED_EMBD"], 1, deep_rows, -1,
                           "tied_copy", extra=dict(from_idx=t_tied_a))
    p.tables[t_tied_a]["peer"], p.tables[t_tied_b]["peer"] = t_tied_b, t_tied_a

    W = [1.0, 2.0, -0.5, 3.0, 0.25, 1.5, -1.0, 4.0]

    def shaped(scale, shift):
        return [(k, W[(k + shift) % len(W)] * scale) for k in range(H)]

    srows = [shaped(1.0, 0), shaped(1.0, 0), shaped(2.0, 3), shaped(0.5, 5)]
    t_synth = p.add_table(en["LMH_ENC_BF16"], en["LMH_ROLE_SYNTH_DUP"], 0, 4, -1, "synth",
                          nbytes=4 * H * fc["LMH_BF16_BYTES"],
                          extra=dict(bytes=synth_bf16_bits(H, srows), dup="0,1"))
    t_synth_q8 = p.add_table(en["LMH_ENC_Q8_0"], en["LMH_ROLE_SYNTH_DUP"], 0, 4, -1, "synth",
                             nbytes=4 * (H // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"],
                             extra=dict(bytes=synth_q8_bits(fc, H, srows), dup="0,1"))
    p.tables[t_synth]["peer"], p.tables[t_synth_q8]["peer"] = t_synth_q8, t_synth
    p.slab_bytes = p.place_tables()

    hidden_spec = [dict(i=i, src="dump", file=DUMP_FILE, row=i, n_rows=n_real)
                   for i in range(n_real)]
    kinds = [("ones", {}), ("onehot", dict(k=0)), ("onehot", dict(k=H - 1)),
             ("onehot", dict(k=hd - 1)), ("onehot", dict(k=hd)),
             ("onehot_inexact", dict(k=H // 2)), ("tail_zero", {}), ("head_zero", {}),
             ("head_only", {})]
    # Each synthetic shape is materialised BATCH times, because a case addresses hidden[b0 + b] for
    # b in [0, bcount) and every batched case uses bcount=BATCH. One copy per shape made an injection
    # built on tail_zero/head_zero/head_only read past the end of the pool — the out-of-range index is
    # caught at build time below, so the pool size and the indexing rule cannot drift apart again.
    h_index, n_h = {}, n_real
    for kind, extra in kinds:
        if kind not in h_index:
            h_index[kind] = n_h                      # the FIRST of that kind's group
        for _ in range(BATCH):
            d = dict(i=n_h, src="synth", kind=kind)
            d.update(extra)
            hidden_spec.append(d)
            n_h += 1
    p.hidden_spec, p.n_hidden, p.h_index = hidden_spec, n_h, h_index

    SEQ, REV, S8 = en["LMH_ASSOC_SEQ"], en["LMH_ASSOC_REV"], en["LMH_ASSOC_SPLIT8"]
    DENSE, GATH = en["LMH_PATH_DENSE"], en["LMH_PATH_GATHER"]
    BATCHED, PERROW = en["LMH_BATCH_BATCHED"], en["LMH_BATCH_PERROW"]
    CORRECT, M = en["LMH_MODE_CORRECT"], en
    B = BATCH
    real0, real1 = 0, 1
    anchors = {}
    for t, nm, nr in [(t_lmh, "bf16-lmhead", max_rows), (t_emb, "bf16-embd", max_rows),
                      (t_lmh_q8, "q8-lmhead", max_rows), (t_emb_q8, "q8-embd", max_rows),
                      (t_tied_a, "bf16-tied-a", max_rows), (t_tied_b, "bf16-tied-b", max_rows),
                      (t_synth, "bf16-synth", 4), (t_synth_q8, "q8-synth", 4)]:
        a = anchors[nm] = {}
        a["batched"] = p.add_case("honest-%s-dense-batched" % nm, t, path=DENSE, batching=BATCHED,
                                  b0=real0, bcount=B, row_lo=0, nrows=nr, assoc=SEQ, mode=CORRECT,
                                  expect="BAND")
        a["perrow"] = []
        for b in range(B):
            a["perrow"].append(p.add_case("honest-%s-perrow%d" % (nm, b), t, path=DENSE,
                                          batching=PERROW, b0=real0 + b, bcount=1, row_lo=0,
                                          nrows=nr, assoc=SEQ, mode=CORRECT, expect="EXACT",
                                          cmp_off=a["batched"]["out_off"], grp=a["batched"]["idx"]))
        gids = [0, 1, 2, 3] if "synth" in nm else list(range(nr))
        same_as_dense = gids == list(range(nr))
        a["gather"] = p.add_case("honest-%s-gather" % nm, t, path=GATH, batching=BATCHED, b0=real0,
                                 bcount=B, ids=gids, assoc=SEQ, mode=CORRECT,
                                 expect="EXACT" if same_as_dense else "BAND",
                                 cmp_off=(a["batched"]["out_off"] if same_as_dense else -1))
        p.add_case("honest-%s-gather-shuffled" % nm, t, path=GATH, batching=BATCHED, b0=real1,
                   bcount=B, ids=list(reversed(gids)), assoc=SEQ, mode=CORRECT, expect="BAND")
        if t in (t_lmh, t_lmh_q8):
            p.add_case("honest-%s-dense-rev" % nm, t, path=DENSE, batching=BATCHED, b0=real0,
                       bcount=B, row_lo=0, nrows=nr, assoc=REV, mode=CORRECT, expect="BAND")
            p.add_case("honest-%s-dense-split8" % nm, t, path=DENSE, batching=BATCHED, b0=real0,
                       bcount=B, row_lo=0, nrows=nr, assoc=S8, mode=CORRECT, expect="BAND")
        if t == t_tied_b:
            # The tied control's whole claim in one case: the SAME geometry (same hidden vectors, same
            # id list, same path) issued against the PEER table. The peer's bytes are the anchor's bytes,
            # so this is bit-exact by construction and any difference is the aliasing assumption breaking.
            # It is not the shuffled case renamed: shuffled reads different hidden, which cannot be EXACT.
            p.add_case("honest-bf16-tied-b-gather-vs-tied-a", t, path=GATH, batching=BATCHED,
                       b0=real0, bcount=B, ids=gids, assoc=SEQ, mode=CORRECT, expect="EXACT",
                       cmp_off=anchors["bf16-tied-a"]["gather"]["out_off"])

    p.add_case("honest-bf16-lmhead-dense-batched-b2", t_lmh, path=DENSE, batching=BATCHED,
               b0=real0, bcount=2, row_lo=0, nrows=max_rows, assoc=SEQ, mode=CORRECT,
               expect="EXACT", cmp_off=anchors["bf16-lmhead"]["batched"]["out_off"])
    deep = p.add_case("honest-bf16-lmhead-deep-batched", t_lmh, path=DENSE, batching=BATCHED,
                      b0=real0, bcount=B, row_lo=0, nrows=deep_rows, assoc=SEQ, mode=CORRECT,
                      expect="BAND", on_device=0)
    for b in range(B):
        p.add_case("honest-bf16-lmhead-deep-perrow%d" % b, t_lmh, path=DENSE, batching=PERROW,
                   b0=real0 + b, bcount=1, row_lo=0, nrows=deep_rows, assoc=SEQ, mode=CORRECT,
                   expect="EXACT", cmp_off=deep["out_off"], grp=deep["idx"], on_device=0)
    deep8 = p.add_case("honest-bf16-lmhead-deep-batched-b8", t_lmh, path=DENSE, batching=BATCHED,
                       b0=0, bcount=8, row_lo=0, nrows=deep_rows, assoc=SEQ, mode=CORRECT,
                       expect="BAND", on_device=0)
    for b in range(8):
        p.add_case("honest-bf16-lmhead-deep-perrow-b8-%d" % b, t_lmh, path=DENSE, batching=PERROW,
                   b0=b, bcount=1, row_lo=0, nrows=deep_rows, assoc=SEQ, mode=CORRECT,
                   expect="EXACT", cmp_off=deep8["out_off"], grp=deep8["idx"], on_device=0)

    def inj(nm, table, mode_name, **kw):
        kw.setdefault("path", DENSE)
        kw.setdefault("batching", BATCHED)
        kw.setdefault("assoc", SEQ)
        kw.setdefault("row_lo", 0)
        kw.setdefault("nrows", max_rows)
        tok = mode_token(mode_name)
        c = p.add_case("inj-%s-%s" % (nm, tok), table, mode=M[mode_name], expect="AUTO", **kw)
        # Every injection is judged against the SAME work done honestly. Hand-pairing 73 injections
        # with 73 baselines is the pairing that goes silently wrong, so the baseline is added here from
        # the injection's own geometry when the population did not already contain one, and cmp_off is
        # filled from it (lm_head_plan.h refuses a traced BYTES case with no baseline). adjudicate does
        # NOT read cmp_off: it re-derives the twin from the geometry and the R line, so a mis-paired
        # baseline is a red check rather than a silent agreement.
        tw = p.honest_twin(c)
        if tw is None:
            tw = p.add_case("honest-%s-baseline" % c["name"], c["table"], path=c["path"],
                            batching=c["batching"], b0=c["b0"], bcount=c["bcount"],
                            row_lo=c["row_lo"], nrows=c["nrows"], ids=kw.get("ids"),
                            assoc=c["assoc"], mode=CORRECT, expect="BAND")
        if c["cmp_off"] < 0:
            c["cmp_off"] = tw["out_off"]
        return c

    for t, nm in [(t_lmh, "bf16-lmhead"), (t_lmh_q8, "q8-lmhead"), (t_emb, "bf16-embd"),
                  (t_emb_q8, "q8-embd")]:
        inj(nm, t, "LMH_MODE_HALF_ACCUM", b0=h_index["onehot"], bcount=B)
        inj(nm, t, "LMH_MODE_HALF_ACCUM", b0=real0, bcount=B)
        # The firing arm needs a product that is NOT already a bf16 value by a margin the band cannot
        # swallow: one hidden entry of 1.001953125 makes that single product's bf16 rounding about
        # 2^-9 of itself, against a band of gamma_2048 = 1.2e-4 of the absolute sum.
        inj(nm, t, "LMH_MODE_HALF_ACCUM", b0=h_index["onehot_inexact"], bcount=B)
        inj(nm, t, "LMH_MODE_DROP_LAST_K", b0=h_index["tail_zero"], bcount=B)
        inj(nm, t, "LMH_MODE_DROP_LAST_K", b0=real0, bcount=B)
        inj(nm, t, "LMH_MODE_DROP_LAST_K", b0=h_index["head_zero"], bcount=B, assoc=REV)
        inj(nm, t, "LMH_MODE_DROP_LAST_K", b0=real0, bcount=B, assoc=REV)
        inj(nm, t, "LMH_MODE_ROW_STRIDE_HEAD_DIM", b0=h_index["head_only"], bcount=B, row_lo=0)
        inj(nm, t, "LMH_MODE_ROW_STRIDE_HEAD_DIM", b0=real0, bcount=B, row_lo=1)
        inj(nm, t, "LMH_MODE_BATCH_SLOT_STRIDE", b0=real0, bcount=1)
        inj(nm, t, "LMH_MODE_BATCH_SLOT_STRIDE", b0=real0, bcount=B)
        inj(nm, t, "LMH_MODE_BATCH_RELOOPS", b0=real0, bcount=1)
        inj(nm, t, "LMH_MODE_BATCH_RELOOPS", b0=real0, bcount=B)
        inj(nm, t, "LMH_MODE_TIED_ALIAS", b0=real0, bcount=B)
        inj(nm, t, "LMH_MODE_CROSS_TABLE_READ", b0=real0, bcount=B)
        inj(nm, t, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH, ids=[0, 1, 2, 3],
            nrows=0)
        if p.tables[t]["enc"] == en["LMH_ENC_Q8_0"]:
            inj(nm, t, "LMH_MODE_Q8_DROP_SCALE", b0=real0, bcount=B)
    inj("bf16-tied-a", t_tied_a, "LMH_MODE_TIED_ALIAS", b0=real0, bcount=B)
    inj("bf16-tied-a", t_tied_a, "LMH_MODE_CROSS_TABLE_READ", b0=real0, bcount=B)
    inj("bf16-lmhead", t_lmh, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH,
        ids=[deep_rows - 1], nrows=0)
    inj("q8-lmhead", t_lmh_q8, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH,
        ids=[deep_rows - 1], nrows=0)
    inj("bf16-synth", t_synth, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH,
        ids=[0], nrows=0)
    inj("q8-synth", t_synth_q8, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH,
        ids=[0], nrows=0)
    inj("bf16-synth", t_synth, "LMH_MODE_GATHER_ROW_OFF_1", b0=real0, bcount=B, path=GATH,
        ids=[1], nrows=0)
    for t, nm in [(t_synth, "bf16-synth"), (t_synth_q8, "q8-synth")]:
        inj(nm, t, "LMH_MODE_HALF_ACCUM", b0=h_index["onehot"], bcount=B, nrows=4)
        inj(nm, t, "LMH_MODE_DROP_LAST_K", b0=h_index["tail_zero"], bcount=B, nrows=4)
    p.t_of = dict(lmh=t_lmh, emb=t_emb, lmh_q8=t_lmh_q8, emb_q8=t_emb_q8, tied_a=t_tied_a,
                  tied_b=t_tied_b, synth=t_synth, synth_q8=t_synth_q8)
    p.real_hidden, p.synth_hidden = n_real, n_h - n_real
    # A case addresses hidden[b0 .. b0+bcount): checked here, at build time, rather than as an
    # IndexError from inside the oracle or a garbage row from a driver.
    for c in p.cases:
        if c["b0"] < 0 or c["bcount"] < 1 or c["b0"] + c["bcount"] > p.n_hidden:
            raise Fail("case %r (idx %d) addresses hidden[%d,%d) but n_hidden=%d"
                       % (c["name"], c["idx"], c["b0"], c["b0"] + c["bcount"], p.n_hidden))
    return p


def declared(name):
    """The plan's `expect` field as an int. `AUTO` means gate.py will derive it from the bytes and
    require the drivers to have carried whatever the derivation said."""
    return name


# ---------------------------------------------------------------------------------------------
# the materialised bytes, and what they FORCE each case to be
# ---------------------------------------------------------------------------------------------
class Data(object):
    """The files the oracle materialised, read back. Every judgement below reads THESE bytes rather
    than trusting the intent that produced them: a population builder that asked for a duplicated row
    and did not get one is exactly what this catches."""

    def __init__(self, work, p, fc, en):
        self.p, self.fc, self.en = p, fc, en
        with open(os.path.join(work, "slab.bin"), "rb") as fh:
            self.slab = fh.read()
        if len(self.slab) != p.slab_bytes:
            raise Fail("slab.bin is %d B, the plan says %d" % (len(self.slab), p.slab_bytes))
        with open(os.path.join(work, "hidden.f32"), "rb") as fh:
            hb = fh.read()
        want = 4 * p.n_hidden * p.cfg["h"]
        if len(hb) != want:
            raise Fail("hidden.f32 is %d B, n_hidden=%d x h=%d x 4 says %d" % (len(hb), p.n_hidden,
                                                                              p.cfg["h"], want))
        self.hidden = [list(struct.unpack_from("<%df" % p.cfg["h"], hb, i * 4 * p.cfg["h"]))
                       for i in range(p.n_hidden)]
        self.ids = list(p.ids)
        self._cache = {}

    def row_bytes(self, t, row, h_used=None):
        """(start, length) of one row under the stride the case believes. Written here, independently
        of lmh_row_ptr, so a disagreement between the two is a finding and not an agreement."""
        fc, en = self.fc, self.en
        h_used = t["h"] if h_used is None else h_used
        if t["enc"] == en["LMH_ENC_Q8_0"]:
            at = t["base"] + (row * (h_used // fc["LMH_Q8_0_ELTS"])) * fc["LMH_Q8_0_BYTES"]
            return at, (h_used // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"]
        return (t["base"] + row * h_used * fc["LMH_BF16_BYTES"],
                h_used * fc["LMH_BF16_BYTES"])

    def row_values(self, t, row, h_used):
        """One row decoded through the AUDITED reader over exactly the range the addressing says."""
        key = (t["idx"], row, h_used, t["enc"])
        at, n = self.row_bytes(t, row, h_used)
        ck = self._cache.get(key)
        if ck is not None:
            return ck
        buf = self.slab[at:at + n]
        vals = (GR.dequant(buf, "Q8_0", h_used) if t["enc"] == self.en["LMH_ENC_Q8_0"]
                else GR.bf16_to_floats(buf))
        if len(vals) != h_used:
            raise Fail("decoded %d values for table %d row %d, expected %d"
                       % (len(vals), t["idx"], row, h_used))
        self._cache[key] = vals
        return vals

    def rows_of(self, c):
        if c["path"] == self.en["LMH_PATH_DENSE"]:
            return [c["row_lo"] + j for j in range(c["nrows"])]
        return [self.ids[c["ids_off"] + j] for j in range(c["n_ids"])]

    def h_used(self, c):
        t = self.p.tables[c["table"]]
        return (self.p.cfg["head_dim"] if c["mode"] == self.en["LMH_MODE_ROW_STRIDE_HEAD_DIM"]
                else t["h"])

    def nlanes(self, c):
        return self.fc["LMH_SPLIT"] if c["assoc"] == self.en["LMH_ASSOC_SPLIT8"] else 1


def _f32(x):
    """Round a Python float to the binary32 value it would become, exactly as the kernel's float does."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bf16_round(x):
    """The kernel's `__float2bfloat16(x)` then back: ties-to-even on the discarded 16 bits, so the
    same rule as lm_head_kernels.h's bf16 product, not a truncation."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    if (b & 0x7FFF_FFFF) > 0x7F80_0000:
        return x                                    # NaN stays NaN
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000
    return struct.unpack("<f", struct.pack("<I", r))[0]


def derive_class(c, d):
    """Which class a case IS, derived from the materialised bytes — never typed.

    (class, predicate_that_was_tested). `AUTO` in the population means this function decided it, and
    `adjudicate` then prints the predicate beside the verdict, so an immunity is always accompanied by
    the fact in the bytes that earned it."""
    p, fc, en = d.p, d.fc, d.en
    mode, t = c["mode"], p.tables[c["table"]]
    if mode == en["LMH_MODE_CORRECT"]:
        if c["expect_name"] != "AUTO":
            return c["expect_name"], "not an injection"
        # AUTO on a CORRECT case is only reachable by neutralising an injection (--control no-fire):
        # the case now believes it is honest, so it is judged as one — bit-identical to the twin it
        # names, otherwise against the oracle's band. Its expectation words were never filled (the
        # oracle saw an injection when it ran), so the band it is bounded by reads NaN and the run
        # goes red with a named word. That is the control's whole point, reached by a judgement.
        if c["cmp_off"] >= 0:
            return "EXACT", ("mode CORRECT with cmp_off=%d: bit-identity against the named twin"
                             % c["cmp_off"])
        return "BAND", "mode CORRECT with no declared class: bounded by the oracle band"
    h_used = d.h_used(c)
    rows = d.rows_of(c)
    peer = p.tables[t["peer"]] if t["peer"] >= 0 else None
    nlanes = d.nlanes(c)
    if mode == en["LMH_MODE_GATHER_ROW_OFF_1"]:
        same = True
        for r in rows:
            a, an = d.row_bytes(t, r, h_used)
            b, bn = d.row_bytes(t, r + 1, h_used)
            if not (t["base"] <= b and b + bn <= t["base"] + t["nbytes"]):
                # The successor of the LAST row is not this table's bytes at all, so BOTH rulers must
                # move: the words (they come from a foreign range) and the cross counter. Calling that
                # BYTES would demand "the words still equal the twin's" of a case whose whole content is
                # reading somebody else's row.
                return "FIRES", ("successor row %d of table %d leaves the table: the words come from a "
                                 "foreign range and the cross counter must move" % (r + 1, t["idx"]))
            if d.slab[a:a + an] != d.slab[b:b + bn]:
                same = False
        return ("IMMUNE" if same else "FIRES"), \
            "successor rows byte-identical for all %d gathered rows: %d" % (len(rows), same)
    if mode == en["LMH_MODE_BATCH_SLOT_STRIDE"]:
        immune = c["bcount"] == 1 or p.cfg["h"] == c["n_out"]
        return ("IMMUNE" if immune else "FIRES"), \
            "bcount=%d, injected slot=h=%d, honest slot=n_out=%d" % (c["bcount"], p.cfg["h"],
                                                                     c["n_out"])
    if mode == en["LMH_MODE_BATCH_RELOOPS"]:
        immune = (c["batching"] != en["LMH_BATCH_BATCHED"]) or c["bcount"] == 1
        return ("IMMUNE" if immune else "BYTES"), \
            "batched with bcount=%d: the row set is walked %d time(s)" % (c["bcount"], c["bcount"])
    if mode == en["LMH_MODE_TIED_ALIAS"]:
        same = peer is not None and peer["base"] == t["base"] and peer["nbytes"] == t["nbytes"]
        return ("IMMUNE" if same else "FIRES"), \
            "peer table %d is %s" % (t["peer"], "THE SAME BYTES (tied control)" if same
                                     else "a disjoint range (untied)")
    if mode == en["LMH_MODE_CROSS_TABLE_READ"]:
        same = peer is not None and peer["base"] == t["base"] and peer["nbytes"] == t["nbytes"]
        return "BYTES", ("peer IS the own range: cross can never move" if same
                         else "peer is a disjoint range: cross and peer_hits must move")
    if mode == en["LMH_MODE_ROW_STRIDE_HEAD_DIM"]:
        hd = p.cfg["head_dim"]
        tail_zero = all(d.hidden[c["b0"] + b][k] == 0.0
                        for b in range(c["bcount"]) for k in range(hd, t["h"]))
        first = all(r == 0 for r in rows)
        return ("IMMUNE" if (tail_zero and first) else "FIRES"), \
            "all rows are row 0 (%d) and h[k]==0 for k in [%d,%d) (%d)" % (first, hd, t["h"], tail_zero)
    if mode == en["LMH_MODE_HALF_ACCUM"]:
        # Derived from the bytes, not from the arm: round every product to bf16 here and ask whether the
        # word would move FURTHER than that word's own band. "Some product is not a bf16 value" is not
        # the claim — an injection whose every effect sits inside the rounding budget is not observable
        # at this width, and calling it FIRES would certify a check that cannot go any colour but green.
        u = 2.0 ** -24
        beyond_any = False
        moved_any = False
        worst = 0.0
        first = None
        for r in rows:
            vals = d.row_values(t, r, h_used)
            for b in range(c["bcount"]):
                hv = d.hidden[c["b0"] + b]
                honest, rounded, scale = [], [], 0.0
                for k in range(h_used):
                    p = hv[k] * vals[k]
                    honest.append(p)
                    rounded.append(_f32(_bf16_round(p)))
                    scale += abs(p)
                K = h_used
                band = (K * u / (1.0 - K * u)) * math.fsum(abs(x) for x in honest)
                dd = abs(math.fsum(honest) - math.fsum(rounded))
                if band > 0.0 and dd > band:
                    beyond_any = True
                if band > 0.0 and dd != 0.0:
                    moved_any = True
                if band > 0.0 and first is None:
                    first = (dd, band)
                worst = max(worst, dd / band if band > 0.0 else 0.0)
        if beyond_any:
            return "FIRES", "bf16-rounding every product moves a word beyond its own band (worst %.3g x)" % worst
        if moved_any:
            # The injection really happened (the words are not the twin's) and every word still sits
            # inside the band. At h=2048 the fp32 summation band, gamma_2048 = 1.221e-4 of the absolute
            # sum, is WIDER than the per-term error of rounding a product to bf16 after cancellation,
            # so a bf16-valued pair of operands is not observable through the band. Declaring this
            # FIRES would certify an arm that cannot be anything but red, and declaring it IMMUNE would
            # claim bit-identity that the bytes contradict; INSIDE is the class that says both facts.
            return "INSIDE", ("bf16-rounding every product moves every word but never past its own "
                              "band (worst %.3f x)" % worst)
        return "IMMUNE", "bf16-rounding every product changes no word at all"
    if mode == en["LMH_MODE_DROP_LAST_K"]:
        per = h_used // nlanes
        dropped = []
        for lane in range(nlanes):
            k0 = lane * per
            k1 = h_used if lane == nlanes - 1 else k0 + per
            dropped.append(k0 if c["assoc"] == en["LMH_ASSOC_REV"] else k1 - 1)
        zero = all(d.hidden[c["b0"] + b][k] == 0.0 for b in range(c["bcount"]) for k in dropped)
        return ("IMMUNE" if zero else "FIRES"), \
            "the dropped k %s has h[k]==0 for every batch row: %d" % (sorted(set(dropped)), zero)
    if mode == en["LMH_MODE_Q8_DROP_SCALE"]:
        ones = True
        for r in rows:
            at, n = d.row_bytes(t, r, h_used)
            for i in range(n // fc["LMH_Q8_0_BYTES"]):
                if GR.f16(d.slab, at + i * fc["LMH_Q8_0_BYTES"]) != 1.0:
                    ones = False
                    break
            if not ones:
                break
        return ("IMMUNE" if ones else "FIRES"), "every block delta equals 1.0: %d" % ones
    raise Fail("derive_class: mode %d has no derivation, so it cannot be judged" % mode)


# ---------------------------------------------------------------------------------------------
# the independent read-set recount
# ---------------------------------------------------------------------------------------------
def recount(c, d, words, skip=False):
    """Every field the R line prints for this case, recomputed HERE.

    The address derivation is this file's own (mirroring, in a second language and a second author,
    what lmh_row_ptr/lmh_lane_ptr must do); the VALUES come from the audited decoder, so the digest
    columns are not the kernel's own arithmetic quoted back at it. oxor/osum come from the arm's
    output FILE, not from the driver's memory.

    `skip` is the device arm leaving a host-only case alone: every counter must then read zero, which
    is a claim, not an exemption — a device that ran a case it was told to skip shows up here."""
    p, fc, en = d.p, d.fc, d.en
    t = p.tables[c["table"]]
    peer = p.tables[t["peer"]] if t["peer"] >= 0 else None
    h_used = d.h_used(c)
    nlanes = d.nlanes(c)
    per = h_used // nlanes
    rows = d.rows_of(c)
    if c["mode"] == en["LMH_MODE_GATHER_ROW_OFF_1"]:
        # The injection's whole content is "gather the successor row". A recount that read the honest
        # row would agree with the kernel only if the kernel had not done what the mode says, so the
        # +1 lives HERE as well as in the kernel; derive_class decided FIRES/BYTES from the same +1.
        rows = [r + 1 for r in rows]
    src = t
    if c["mode"] == en["LMH_MODE_TIED_ALIAS"] and peer is not None:
        src = peer
    own_lo, own_hi = t["base"], t["base"] + t["nbytes"]
    p_lo = peer["base"] if peer else 0
    p_hi = peer["base"] + peer["nbytes"] if peer else 0
    r = dict(bytes=0, loads=0, xorb=0, vsum=0, bsum=0, min_off=0xFFFFFFFF, max_off=0, cross=0,
             peer_hits=0, lanes=0, bover=0)
    reloop = (not skip and c["mode"] == en["LMH_MODE_BATCH_RELOOPS"]
              and c["batching"] == en["LMH_BATCH_BATCHED"] and c["bcount"] > 1)
    reps = c["bcount"] if reloop else 1
    mode = c["mode"]

    def touch(at, n):
        r["bytes"] += n
        r["loads"] += 1
        if at < own_lo or at + n > own_hi:
            r["cross"] += 1
        else:
            r["min_off"] = min(r["min_off"], at - own_lo)
            r["max_off"] = max(r["max_off"], at + n - own_lo)
        if peer is not None and p_lo <= at and at + n <= p_hi:
            r["peer_hits"] += 1
        r["bsum"] += sum(d.slab[at:at + n])

    def slice_at(base, k0, kn):
        if t["enc"] == en["LMH_ENC_Q8_0"]:
            return base + (k0 // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"], \
                (kn // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"]
        return base + k0 * fc["LMH_BF16_BYTES"], kn * fc["LMH_BF16_BYTES"]

    for jj in range(0 if skip else c["n_out"]):
        row = rows[jj]
        wr, _wn = d.row_bytes(src, row, h_used)
        for lane in range(nlanes):
            k0 = lane * per
            k1 = h_used if lane == nlanes - 1 else k0 + per
            kn = k1 - k0
            at, n = slice_at(wr, k0, kn)
            for _ in range(reps):
                touch(at, n)
            if mode == en["LMH_MODE_CROSS_TABLE_READ"] and peer is not None:
                pr, _pn = d.row_bytes(peer, row, h_used)
                pat, pn = slice_at(pr, k0, kn)
                touch(pat, pn)
            r["lanes"] += 1
            # the VALUES come from the table the addressing says was read (`src`), not from the table
            # the case was named after: a tied-alias arm loads through the peer's base, and quoting the
            # own table's values back would make the digest agree with the driver only if the alias
            # had not happened.
            vals = d.row_values(src, row, h_used)
            if mode == en["LMH_MODE_Q8_DROP_SCALE"]:
                qbytes = d.slab
                for k in range(k0, k1):
                    blk = wr + (k // fc["LMH_Q8_0_ELTS"]) * fc["LMH_Q8_0_BYTES"]
                    q = struct.unpack_from("<b", qbytes, blk + 2 + (k % fc["LMH_Q8_0_ELTS"]))[0]
                    bits = _wbits(float(q))
                    r["xorb"] ^= bits
                    r["vsum"] = (r["vsum"] + bits) & 0xFFFFFFFFFFFFFFFF
            else:
                for k in range(k0, k1):
                    bits = _wbits(vals[k])
                    r["xorb"] ^= bits
                    r["vsum"] = (r["vsum"] + bits) & 0xFFFFFFFFFFFFFFFF
    slot = p.slot(c)
    ox, osum, bover = 0, 0.0, 0
    for b in range(0 if skip else c["bcount"]):
        for j in range(c["n_out"]):
            at = c["out_off"] + b * slot + j
            if at < 0 or at >= p.out_floats:
                bover += 1
                continue
            w = words[at]
            ox ^= w
            osum += f32(w)
    r["bover"], r["oxor"], r["osum"] = bover, ox, osum
    return r


# ---------------------------------------------------------------------------------------------
# the verdict machinery
# ---------------------------------------------------------------------------------------------
class V(object):
    """The check ledger. `check()` refuses a constant predicate outright: the router lane certified 40
    checks that included one literal True, and it took two investigations to find it
    (`pitfalls/40` §"A check whose predicate is True is a comment wearing a counter")."""

    def __init__(self):
        self.n = 0
        self.bad = []

    def check(self, name, ok, n, note=""):
        if not isinstance(ok, bool):
            raise Fail("verdict-bearing check %r has a non-boolean predicate (%r); a check over a "
                       "string or a list silently counts the empty case as a finding" % (name, ok))
        self.n += 1
        if not ok:
            self.bad.append(name)
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if ok else 0, n, note))
        return bool(ok)

    def note(self, name, note):
        print("NOTE name=%s %s" % (name, note))

    def refusal(self, e):
        """An explicit refusal IS a finding. main() catches Fail from anywhere in the pipeline; before
        this, a run that died in the oracle printed `RESULT PASS ... checks=2 findings=0` because the
        two checks that had run were both green. A verdict over a run that adjudicated nothing is the
        same class of hole as a check that can produce no output (pitfalls/40)."""
        self.n += 1
        self.bad.append("the run refused: %s" % str(e)[:120])
        print("CHECK name=the run reached its adjudication ok=0 n=1 refusal=%s" % str(e)[:200])

    def verdict(self, label):
        print("CHECKS ran=%d failed=%d %s" % (self.n, len(self.bad),
                                              ",".join(self.bad[:8]) if self.bad else ""))
        print("RESULT %s %s checks=%d findings=%d" % ("PASS" if not self.bad else "FAIL", label,
                                                      self.n, len(self.bad)))
        return 0 if not self.bad else 1


def words_of(path, n):
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) != 4 * n:
        raise Fail("%s holds %d bytes, the plan says %d floats" % (path, len(buf), n))
    return list(struct.unpack("<%dI" % n, buf))


def doubles_of(path, n):
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) != 8 * n:
        raise Fail("%s holds %d bytes, the plan says %d doubles" % (path, len(buf), n))
    return list(struct.unpack("<%dd" % n, buf))


def word_eq(a, b):
    if a == b:
        return True
    ma, mb = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    return ma > 0x7F800000 and mb > 0x7F800000          # NaN of any payload, as lmh_word_eq does


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


# The consumer half of the report's canonical text form; lmh_r_fmt[] in lm_head_plan.h is the producer
# half and contract() scrapes it, so a field's radix is declared once and enforced twice. Each pattern
# is anchored with \Z, so a token only passes if the WHOLE token is the canonical spelling.
CANON = {
    "dec": re.compile(r"(?:0|[1-9][0-9]*)\Z"),
    "hex": re.compile(r"0x[0-9A-F]{8}\Z"),
    "real": re.compile(r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]inf(?:inity)?)\Z",
                       re.I),
}


def read_report(path, p, fields, fc, fmts):
    """Parse the R lines, require the name line to be the writer's own array, require one row per
    case, and require EVERY numeric token to be the canonical spelling its declared form says. A report
    that parsed to zero rows cannot fail, so it is refused here rather than judged.

    The form check is the consumer half of the digest rule (lmh_r_fmt[] in lm_head_plan.h is the
    producer half). A bit-image column accepts ONLY `0x` + 8 upper-case hex digits and a count column
    ONLY canonical decimal, so a producer that spells a digest in the other radix is a REFUSAL naming
    the field, the case and the offending token — never a mis-parse into a plausible number. Parsing
    4278190080 where 0xFF400000 was written, or the reverse, is exactly the two-rulers bug that makes
    every digest comparison in the lane meaningless in whichever direction it errs."""
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    header = [l for l in lines if l.startswith("#R")]
    body = [l for l in lines if not l.startswith("#")]
    if len(header) != 1:
        raise Fail("%s has %d '#R' name lines, expected exactly 1" % (path, len(header)))
    hf = header[0].split("\t")
    if len(hf) != fc["LMH_F_R"]:
        raise Fail("%s: the name line has %d tokens, LMH_F_R says %d" % (path, len(hf), fc["LMH_F_R"]))
    if hf[1:] != fields:
        raise Fail("%s: the name line %s is not the writer's lmh_r_fields[] %s"
                   % (path, hf[1:][:5], fields[:5]))
    if not body:
        raise Fail("%s has a name line and NO R rows: an empty parse cannot fail" % path)
    rows = {}
    for l in body:
        f = l.split("\t")
        if len(f) != fc["LMH_F_R"]:
            raise Fail("%s: an R row has %d fields, the contract says %d" % (path, len(f), fc["LMH_F_R"]))
        rec = {}
        for nm, kind, tok in zip(fields, fmts, f[1:]):
            if not CANON[kind].match(tok):
                msg = ("%s: case %s field '%s' is %r, which is not the canonical %s form the writer "
                       "declares in lmh_r_fmt[] (a digest spelled in the wrong radix is a finding, not "
                       "a plausible number to be compared)"
                       % (path, f[1], nm, tok, kind))
                # Printed in full BEFORE the raise: main() truncates a refusal to 200 chars in its CHECK
                # line, and a non-zero exit that does not name the offending token is a crash, not a
                # finding.
                print("REFUSE %s" % msg)
                raise Fail(msg)
            rec[nm] = (int(tok, 16) if kind == "hex" else
                       int(tok, 10) if kind == "dec" else float(tok))
        rows[int(f[1])] = rec
    if len(body) != len(p.cases):
        raise Fail("%s printed %d R rows, the plan has %d cases (a missing row would leave that case "
                   "entirely unjudged)" % (path, len(body), len(p.cases)))
    if sorted(rows) != list(range(len(p.cases))):
        raise Fail("%s: the idx column is not 0..%d exactly once" % (path, len(p.cases) - 1))
    print("REPORT file=%s parsed=%d printed=%d fields=%d forms=hex:%d,dec:%d,real:%d"
          % (path, len(rows), len(body), len(fields), fmts.count("hex"), fmts.count("dec"),
             fmts.count("real")))
    return rows


COUNTER_FIELDS = ("bytes", "loads", "xorb", "vsum", "bsum", "lo", "hi", "cross", "peer_hits",
                  "lanes", "bover")


def legal_class(c, cls):
    """A declared class is only worth printing if the derivation could have contradicted it. Honest
    cases declare BAND or EXACT (this file's population builder, and the derivation echoes it because
    there is nothing to derive); injections declare AUTO and must land in one of the three derived
    classes. Anything else means the population asked for a class the derivation has no rule for."""
    if c["expect_name"] == "AUTO":
        return cls in ("FIRES", "IMMUNE", "BYTES", "INSIDE")
    return cls == c["expect_name"] and cls in ("BAND", "EXACT")


def adjudicate(v, p, d, report, words, expw, bandw, tag, fc, en):
    rep = report
    counts = {}
    # Keyed by CASE INDEX, not appended to a list. The gather branch and the FIRES branch both register
    # the same case, so the old list counted those arms twice: `injection denominator matches the plan`
    # read 79 judged against 77 declared, and a duplicate is exactly what lets one case that was never
    # judged hide behind a case that was judged twice.
    fired, immunes, subband = {}, {}, {}
    results = {}
    results = {}
    skip_all = (tag != "host")
    want_inj = sum(1 for c in p.cases if c["mode"] != en["LMH_MODE_CORRECT"]
                   and (not skip_all or c["on_device"]))
    want_idx = set(c["idx"] for c in p.cases if c["mode"] != en["LMH_MODE_CORRECT"]
                   and (not skip_all or c["on_device"]))
    for c in p.cases:
        cls, pred = derive_class(c, d)
        c["derived"], c["predicate"] = cls, pred
        counts[cls] = counts.get(cls, 0) + 1
    for c in p.cases:
        cls, pred = c["derived"], c["predicate"]
        skip = skip_all and not c["on_device"]
        r = rep[c["idx"]]
        slot = p.slot(c)
        nlanes = d.nlanes(c)
        # 1. the R line must echo the plan it was given
        want_echo = dict(table=c["table"], role=p.tables[c["table"]]["role"],
                         enc=p.tables[c["table"]]["enc"], path=c["path"], batching=c["batching"],
                         b0=c["b0"], bcount=c["bcount"], row_lo=c["row_lo"], nrows=c["nrows"],
                         n_ids=c["n_ids"], assoc=c["assoc"], mode=c["mode"],
                         on_device=c["on_device"], trace=c["trace"], n_out=c["n_out"],
                         n_items=c["n_out"] * nlanes, nlanes=nlanes)
        bad_echo = [(k, int(r[k]), want) for k, want in want_echo.items() if int(r[k]) != want]
        v.check("%s case=%d %s R line echoes the plan" % (tag, c["idx"], c["name"]),
                not bad_echo, len(want_echo), "mismatches=%s" % (bad_echo[:4],))
        # 2. the roll-ups of the words it wrote, recomputed from the output FILE
        rc = recount(c, d, words, skip)
        # The report's numeric fields are ALL decimal — lmh_write_report prints oxor/xorb with %u and
        # vsum/bsum with %llu. Parsing a bit image as hex here is the two-rulers bug this lane exists to
        # catch: 29163520 read as hex is 689321248, which disagrees with the recount for every case.
        # The 0x%08X below is how THIS file displays a bit image, not how the report spells it.
        v.check("%s case=%d %s oxor==recount" % (tag, c["idx"], c["name"]),
                int(r["oxor"]) == rc["oxor"], c["bcount"] * c["n_out"],
                "driver=0x%08X recount=0x%08X" % (int(r["oxor"]) & 0xFFFFFFFF, rc["oxor"]))
        osum_ok = abs(float(r["osum"]) - rc["osum"]) <= 1e-6 * max(1.0, abs(rc["osum"]))
        v.check("%s case=%d %s osum==recount" % (tag, c["idx"], c["name"]), osum_ok,
                c["bcount"] * c["n_out"], "driver=%s recount=%.9g" % (r["osum"], rc["osum"]))
        # 3. the read-set, field by field, against this file's own derivation
        bad_ctr = []
        for fld in COUNTER_FIELDS:
            got = int(r["xorb"]) if fld == "xorb" else int(r[fld])
            want = rc[{"lo": "min_off", "hi": "max_off"}.get(fld, fld)]
            if got != want:
                bad_ctr.append((fld, got, want))
        v.check("%s case=%d %s read-set == recount" % (tag, c["idx"], c["name"]),
                not bad_ctr, len(COUNTER_FIELDS),
                "mismatch=%s trace=%d" % (bad_ctr[:4], c["trace"]))
        v.check("%s case=%d %s no write left `out`" % (tag, c["idx"], c["name"]),
                rc["bover"] == 0, 1, "bover=%d" % rc["bover"])
        if skip:
            # the device was told this case is host-only. That is a claim about the bytes it left
            # alone, not an exemption from being looked at.
            nz = sum(1 for b in range(c["bcount"]) for j in range(c["n_out"])
                     if words[c["out_off"] + b * slot + j] != 0)
            v.check("%s case=%d %s host-only case was not written by this arm"
                    % (tag, c["idx"], c["name"]), nz == 0, c["bcount"] * c["n_out"],
                    "non-zero words=%d (the arm ran a case it was told to skip)" % nz)
            v.check("%s case=%d %s declared class is legal and matches the derivation"
                    % (tag, c["idx"], c["name"]), legal_class(c, cls), 1,
                    "plan=%s derived=%s (host-only case)" % (c["expect_name"], cls))
            continue
        # 4. the verdict
        twin = p.honest_twin(c)
        res = dict(cls=cls, name=c["name"], mode=c["mode"], ok=True, detail="")
        if cls == "BAND":
            outside, worst = [], 0.0
            for b in range(c["bcount"]):
                for j in range(c["n_out"]):
                    at = c["out_off"] + b * slot + j
                    got, want = f32(words[at]), f32(expw[at])
                    bb = bandw[at]
                    if not bb > 0.0:
                        raise Fail("case %d word %d has band %r: a non-positive band bounds nothing"
                                   % (c["idx"], at, bb))
                    rel = abs(got - want) / max(bb, 1e-300)
                    worst = max(worst, rel)
                    if abs(got - want) > bb:
                        outside.append((at, got, want, bb))
            res["ok"] = v.check("%s case=%d %s BAND inside the derived band" % (tag, c["idx"], c["name"]),
                                not outside, c["bcount"] * c["n_out"],
                                "worst=%.3f x band outside=%d %s"
                                % (worst, len(outside), ("first=%r" % (outside[0],)) if outside else ""))
        elif cls == "EXACT":
            owner = next((o for o in p.cases if o["out_off"] == c["cmp_off"] and o["idx"] != c["idx"]),
                         None)
            legal_owner = (owner is not None and owner["mode"] == en["LMH_MODE_CORRECT"]
                           and owner["assoc"] == c["assoc"] and owner["n_out"] == c["n_out"])
            same_table = bool(owner is not None and owner["table"] == c["table"])
            if owner is not None and not same_table:
                # Two roles over ONE buffer is the tied control, and only the tied control: a cmp_off
                # across tables whose bytes are not identical is the assumption this lane exists to
                # break, so it stays a finding.
                ot, ct = p.tables[owner["table"]], p.tables[c["table"]]
                same_table = (ot["base"] == ct["base"] and ot["nbytes"] == ct["nbytes"]
                              and ot["enc"] == ct["enc"] and ot["h"] == ct["h"])
            res["ok"] = v.check("%s case=%d %s cmp_off is a same-association honest case"
                                % (tag, c["idx"], c["name"]),
                                legal_owner and same_table, 1,
                                "owner=%s same_table=%d" % (owner["name"] if owner else None,
                                                            int(same_table)))
            if legal_owner:
                # A per-row case holds ONE hidden vector of the owner's batch, so the words it must
                # equal are the owner's batch (b0 - owner.b0), not the owner's batch 0. Comparing every
                # per-row arm against batch 0 makes three of four arms disagree with themselves.
                oslot = p.slot(owner)
                boff = (0 if owner["batching"] == en["LMH_BATCH_PERROW"]
                        else c["b0"] - owner["b0"])
                if not 0 <= boff < owner["bcount"]:
                    raise Fail("case %d is EXACT against case %d's batch %d, which the owner "
                               "(bcount=%d) does not have"
                               % (c["idx"], owner["idx"], boff, owner["bcount"]))
                nd = sum(1 for b in range(c["bcount"]) for j in range(c["n_out"])
                         if not word_eq(words[c["out_off"] + b * slot + j],
                                        words[c["cmp_off"] + (boff + b) * oslot + j]))
                res["ok"] = v.check("%s case=%d %s EXACT vs cmp_off=%d batch=%d"
                                    % (tag, c["idx"], c["name"], c["cmp_off"], boff),
                                    nd == 0, c["bcount"] * c["n_out"], "differing=%d" % nd) and res["ok"]
        elif cls in ("FIRES", "IMMUNE", "BYTES", "INSIDE"):
            if not v.check("%s case=%d %s has an honest twin" % (tag, c["idx"], c["name"]),
                           twin is not None, 1,
                           "no mode==CORRECT case with this geometry exists"):
                continue
            ts = p.slot(twin)
            # The judgement reads the words where a CONSUMER reads them: batch b at b*n_out. Reading a
            # slot-stride injection at the stride it injected would find the right numbers in the wrong
            # places and record that as agreement — the placement bug is exactly the thing being judged.
            js = c["n_out"]
            nd = sum(1 for b in range(c["bcount"]) for j in range(c["n_out"])
                     if not word_eq(words[c["out_off"] + b * js + j],
                                    words[twin["out_off"] + b * ts + j]))
            res["nd"] = nd          # read back by the amortisation witness: bytes alone must be what
            #                       catches the re-looping kernel, so the words-identical half needs
            #                       to be visible outside this function without being recomputed
            beyond = 0
            worst = 0.0
            for b in range(c["bcount"]):
                for j in range(c["n_out"]):
                    at = c["out_off"] + b * js + j
                    bt = twin["out_off"] + b * ts + j
                    dd = abs(f32(words[at]) - f32(words[bt]))
                    if dd > bandw[bt]:
                        beyond += 1
                    worst = max(worst, dd / max(bandw[bt], 1e-300))
            # GATHER_ROW_OFF_1's read-set claim is a MODE claim, so it is judged whatever the words did:
            # the cross count says exactly which successor rows escaped the table. A case that read
            # nothing outside and still produced plausible numbers would otherwise sail past the word
            # judgement, because an out-of-range successor can look like an unlucky value.
            if c["mode"] == en["LMH_MODE_GATHER_ROW_OFF_1"]:
                # lmh_case_row_direct adds 1 to the id and deliberately does NOT clamp, so the escape
                # set is computable from the plan alone: an item's successor reads outside
                # [base, base + v_rows*row_b) exactly when id+1 >= v_rows, and lmh_trace_range is called
                # once per (item, lane). Zero escapes must mean zero cross reads, which is what stops a
                # kernel from padding its read-set and still looking immune on the words.
                tv = p.tables[c["table"]]["v_rows"]
                ids = (p.ids[c["ids_off"]:c["ids_off"] + c["n_ids"]] if c["n_ids"]
                       else list(range(c["row_lo"], c["row_lo"] + c["n_out"])))
                esc = sum(1 for i in ids if i + 1 >= tv)
                want_cross = esc * nlanes
                res["ok"] = v.check(
                    "%s case=%d %s only the successor rows that escape the table count as cross"
                    % (tag, c["idx"], c["name"]), int(r["cross"]) == want_cross, want_cross,
                    "cross=%s expected=%s (%d of %d ids escape v_rows=%d)"
                    % (r["cross"], want_cross, esc, len(ids), tv)) and res["ok"]
                if esc:
                    fired[c["idx"]] = (c["mode"], c["name"], int(r["cross"]) == want_cross)
            if cls == "FIRES":
                ok = v.check("%s case=%d %s FIRED beyond the band" % (tag, c["idx"], c["name"]),
                             beyond > 0, c["bcount"] * c["n_out"],
                             "differing=%d beyond-band=%d worst=%.3g x band | %s"
                             % (nd, beyond, worst, pred))
                fired[c["idx"]] = (c["mode"], c["name"], ok)
            elif cls == "IMMUNE":
                ok = v.check("%s case=%d %s IMMUNE (bit-identical to its twin)"
                             % (tag, c["idx"], c["name"]), nd == 0, c["bcount"] * c["n_out"],
                             "differing=%d | %s" % (nd, pred))
                immunes[c["idx"]] = (c["mode"], c["name"], ok)
            elif cls == "INSIDE":
                # Two falsifiable halves: the words are NOT the twin's (the injection happened), and no
                # word left the band (the perturbation is smaller than fp32 summation noise at this h).
                moved = nd > 0
                ok = v.check("%s case=%d %s INSIDE: moved but never beyond the band"
                             % (tag, c["idx"], c["name"]), moved and beyond == 0,
                             c["bcount"] * c["n_out"],
                             "differing=%d beyond-band=%d worst=%.3g x band | %s"
                             % (nd, beyond, worst, pred))
                subband[c["idx"]] = (c["mode"], c["name"], ok)
            else:
                ok = v.check("%s case=%d %s BYTES: output still equals its twin"
                             % (tag, c["idx"], c["name"]), nd == 0, c["bcount"] * c["n_out"],
                             "differing=%d | %s" % (nd, pred))
                hh = rep[twin["idx"]]
                if c["mode"] not in (en["LMH_MODE_BATCH_RELOOPS"], en["LMH_MODE_CROSS_TABLE_READ"]):
                    raise Fail("case %d is BYTES-classed but mode %d has no counter rule"
                               % (c["idx"], c["mode"]))
                if c["mode"] == en["LMH_MODE_BATCH_RELOOPS"]:
                    f = c["bcount"] if c["bcount"] > 1 else 1
                    ok = v.check("%s case=%d %s reloops bytes x%d loads x%d digest unchanged"
                                 % (tag, c["idx"], c["name"], f, f),
                                 int(r["bytes"]) == f * int(hh["bytes"])
                                 and int(r["loads"]) == f * int(hh["loads"])
                                 and r["lanes"] == hh["lanes"] and r["xorb"] == hh["xorb"]
                                 and r["vsum"] == hh["vsum"], c["bcount"],
                                 "bytes %s/%s loads %s/%s xorb %s/%s vsum %s/%s"
                                 % (r["bytes"], hh["bytes"], r["loads"], hh["loads"], r["xorb"],
                                    hh["xorb"], r["vsum"], hh["vsum"])) and ok
                    fired[c["idx"]] = (c["mode"], c["name"],
                                       f > 1 and int(r["bytes"]) == f * int(hh["bytes"]))
                elif c["mode"] == en["LMH_MODE_CROSS_TABLE_READ"]:
                    t = p.tables[c["table"]]
                    peer = p.tables[t["peer"]]
                    tied = peer["base"] == t["base"] and peer["nbytes"] == t["nbytes"]
                    n_add = c["n_out"] * nlanes
                    want_cross = int(hh["cross"]) if tied else int(hh["cross"]) + n_add
                    want_peer = int(hh["peer_hits"]) + n_add
                    extents_ok = (int(r["cross"]) == want_cross
                                  and int(r["peer_hits"]) == want_peer)
                    ok = v.check("%s case=%d %s cross/peer_hits as the extents say (%s)"
                                 % (tag, c["idx"], c["name"],
                                    "tied control: cross cannot move" if tied
                                    else "untied: every peer range is a cross"),
                                 extents_ok, 1,
                                 "cross=%s/%s peer_hits=%s/%s" % (r["cross"], want_cross,
                                                                 r["peer_hits"], want_peer)) and ok
                    # ONE predicate for the verdict and for the roll-up. They were two, and the roll-up
                    # asked whether the counters had NOT moved: the four untied arms, whose entire claim
                    # is that cross and peer_hits DO move and which the check above confirms at 4/4,
                    # were registered `fired=0`, so `every injection that must fire fired` could never
                    # go green. A must-fire flag that is the negation of the must-fire claim is a broken
                    # check, not a strict one.
                    #
                    # The tied arm keeps a must-not-move claim, but over `cross` only. For a tied pair
                    # the peer extent IS the own extent, so the witness read lands inside it: peer_hits
                    # must move by n_add while cross stays 0 (lmh_trace_range tests the own range first;
                    # the driver self-test's tied-control-peer-read-is-not-a-cross-read is the witness,
                    # and the host report says cross=0/0 peer_hits=8/8 against the twin's 4). Requiring
                    # peer_hits to stand still there demanded a counter that cannot behave that way, and
                    # it is why `every derived-immune case really is immune` was red on that one case.
                    if tied:
                        immunes[c["idx"]] = (c["mode"], c["name"],
                                             int(r["cross"]) == int(hh["cross"])
                                             and int(r["peer_hits"]) == int(r["loads"]))
                    else:
                        fired[c["idx"]] = (c["mode"], c["name"], extents_ok)
                elif c["mode"] == en["LMH_MODE_GATHER_ROW_OFF_1"]:
                    want_cross = c["n_out"] * nlanes
        else:
            raise Fail("case %d derived class %r is not judgable" % (c["idx"], cls))
        ok_cls = v.check("%s case=%d %s declared class is legal and matches the derivation"
                         % (tag, c["idx"], c["name"]), legal_class(c, cls), 1,
                         "plan=%s derived=%s" % (c["expect_name"], cls))
        res["ok"] = res["ok"] and ok_cls
        results[c["idx"]] = res
    print("POPULATION %s %s inj=%d" % (tag, " ".join("%s=%d" % (k, counts.get(k, 0))
                                                     for k in ("BAND", "EXACT", "FIRES", "IMMUNE",
                                                               "BYTES", "INSIDE")), want_inj))
    print("INJECTIONS %s judged=%d fired=%d immune=%d immune-held=%d"
          % (tag, len(fired) + len(immunes), sum(1 for _, _, o in fired.values() if o),
             len(immunes), sum(1 for _, _, o in immunes.values() if o)))
    for mode, nm, ok in fired.values():
        print("INJECTION arm=%s fired=%d" % (nm, 1 if ok else 0))
    for mode, nm, ok in immunes.values():
        print("IMMUNITY arm=%s held=%d" % (nm, 1 if ok else 0))
    for mode, nm, ok in subband.values():
        print("SUBBAND arm=%s inside=%d" % (nm, 1 if ok else 0))
    judged = set(fired) | set(immunes) | set(subband)
    never = sorted(want_idx - judged)
    not_inj = sorted(judged - want_idx)
    v.check("%s injection denominator matches the plan" % tag,
            want_inj > 0 and len(judged) == want_inj and not never and not not_inj, want_inj,
            "plan=%d judged=%d (fired=%d immune=%d inside=%d) never_judged=%s judged_but_not_injection=%s"
            % (want_inj, len(judged), len(fired), len(immunes), len(subband), never[:6], not_inj[:6]))
    v.check("%s every arm judged INSIDE the band really moved and really stayed inside" % tag,
            all(o for _, _, o in subband.values()), len(subband),
            "wrong: %s" % (",".join(nm for _, nm, o in subband.values() if not o) or "none"))
    v.check("%s every injection that must fire fired" % tag,
            len(fired) > 0 and all(o for _, _, o in fired.values()), len(fired),
            "did not fire: %s" % (",".join(nm for _, nm, o in fired.values() if not o) or "none"))
    v.check("%s every derived-immune case really is immune" % tag,
            len(immunes) > 0 and all(o for _, _, o in immunes.values()), len(immunes),
            "moved anyway: %s" % (",".join(nm for _, nm, o in immunes.values() if not o) or "none"))
    # an immunity that is true by construction is not evidence: each mode with an immune arm has to be
    # demonstrably fireable somewhere in this same plan, and each mode has to be exercised at all.
    reach = {}
    for c in p.cases:
        if c["mode"] == en["LMH_MODE_CORRECT"] or (skip_all and not c["on_device"]):
            continue
        e = reach.setdefault(c["mode"], dict(immune=0, fires=0, fired=0))
        if c["derived"] == "IMMUNE":
            e["immune"] += 1
        if c["derived"] in ("FIRES", "BYTES"):
            e["fires"] += 1
    for mode, nm, ok in fired.values():
        if mode in reach and ok:
            reach[mode]["fired"] += 1
    untested = sorted(mode_token_id(m, en) for m, e in reach.items()
                      if e["immune"] and not e["fired"])
    print("ALTREACH modes_with_injections=%d immune-somewhere=%d fireable-somewhere=%d "
          "immune-only=%s"
          % (len(reach), sum(1 for e in reach.values() if e["immune"]),
             sum(1 for e in reach.values() if e["fired"]), ",".join(untested) or "none"))
    v.check("%s every mode that is immune somewhere is fireable somewhere" % tag,
            not untested, len(reach), "immune-only modes: %s" % (",".join(untested) or "none"))
    return results


# ---------------------------------------------------------------------------------------------
# writing the run: plan.tsv, sources.tsv, hidden.tsv, ids.tsv
# ---------------------------------------------------------------------------------------------
def write_plan(path, p, expect_map):
    """The P/T/C contract, in the field order lmh_read_plan parses. Field counts come from the header."""
    cfg, en, fc = p.cfg, p.en, p.fc
    lines = []
    lines.append("\t".join(["P", str(cfg["head_dim"]), str(cfg["h"]), str(cfg["vocab"]),
                            str(p.n_hidden), str(len(p.ids)), str(p.out_floats), str(p.slab_bytes),
                            str(len(p.tables)), str(len(p.cases))]))
    if len(lines[-1].split("\t")) != fc["LMH_F_P"]:
        raise Fail("the P record I built has %d fields, LMH_F_P says %d"
                   % (len(lines[-1].split("\t")), fc["LMH_F_P"]))
    for t in p.tables:
        lines.append("\t".join(["T", str(t["idx"]), str(t["enc"]), str(t["role"]), str(t["base"]),
                                str(t["nbytes"]), str(t["v_rows"]), str(t["h"]), str(t["peer"]),
                                str(t["real"])]))
        if len(lines[-1].split("\t")) != fc["LMH_F_T"]:
            raise Fail("a T record has %d fields, LMH_F_T says %d"
                       % (len(lines[-1].split("\t")), fc["LMH_F_T"]))
    exp_id = {"EXACT": en["LMH_EXP_EXACT"], "BAND": en["LMH_EXP_BAND"],
              "FIRES": en["LMH_EXP_FIRES"], "IMMUNE": en["LMH_EXP_IMMUNE"],
              "BYTES": en["LMH_EXP_BYTES"],
              # INSIDE is a judgement only this file can make — it needs the band, which the driver
              # does not carry. To the driver it is an ordinary injection, so it travels under the
              # FIRES id; the class the arm is actually held to is the derived one printed beside the
              # verdict, and legal_class is what refuses a mismatch.
              "INSIDE": en["LMH_EXP_FIRES"]}
    for c in p.cases:
        cls = expect_map[c["idx"]]
        if cls not in exp_id:
            raise Fail("case %d has class %r, which is not one of the header's LMH_EXP_* names"
                       % (c["idx"], cls))
        lines.append("\t".join(["C", str(c["idx"]), str(c["table"]), str(c["path"]),
                                str(c["batching"]), str(c["b0"]), str(c["bcount"]), str(c["row_lo"]),
                                str(c["nrows"]), str(c["ids_off"]), str(c["n_ids"]), str(c["assoc"]),
                                str(c["mode"]), str(c["out_off"]), str(c["n_out"]), str(c["cmp_off"]),
                                str(c["grp"]), str(exp_id[cls]), str(c["on_device"]),
                                str(c["trace"])]))
        if len(lines[-1].split("\t")) != fc["LMH_F_C"]:
            raise Fail("case %d has %d fields, LMH_F_C says %d"
                       % (c["idx"], len(lines[-1].split("\t")), fc["LMH_F_C"]))
    with open(path, "w") as fh:
        fh.write("# P head_dim h vocab n_hidden ids_count out_floats slab_bytes n_tables n_cases\n")
        fh.write("# T idx enc role base nbytes v_rows h peer real\n")
        fh.write("# C idx table path batching b0 bcount row_lo nrows ids_off n_ids assoc mode "
                 "out_off n_out cmp_off grp expect on_device trace\n")
        for l in lines:
            fh.write(l + "\n")
    return lines


def write_sidecars(work, p):
    with open(os.path.join(work, "sources.tsv"), "w") as fh:
        fh.write("# T idx kind enc real [file tensor rows] [bytes] [from_idx from_base nbytes]\n")
        for s in p.sources:
            fh.write("\t".join(["T"] + ["%s=%s" % (k, v) for k, v in sorted(s.items())]) + "\n")
    with open(os.path.join(work, "hidden.tsv"), "w") as fh:
        fh.write("# H i src=dump file row n_rows | src=synth kind [k]\n")
        for s in p.hidden_spec:
            fh.write("\t".join(["H"] + ["%s=%s" % (k, v) for k, v in sorted(s.items())]) + "\n")
    with open(os.path.join(work, "ids.tsv"), "w") as fh:
        fh.write("# I off value\n")
        for off, v in enumerate(p.ids):
            fh.write("I\toff=%d\tvalue=%d\n" % (off, v))
    with open(os.path.join(work, "ids.i32"), "wb") as fh:
        fh.write(struct.pack("<%di" % len(p.ids), *p.ids) if p.ids else b"")


def driver_args(work, p, cfg, plan, slab, out, report, extra=()):
    """One driver invocation. The `--expect-*` values come from the CONFIG and the plan this file
    built: a plan that quietly lost or gained a table, a case, or a width is refused, not judged."""
    return ["--plan", plan, "--slab", slab, "--hidden", os.path.join(work, "hidden.f32"),
            "--ids", os.path.join(work, "ids.i32"),
            "--expect-tables", len(p.tables), "--expect-cases", len(p.cases),
            "--expect-head-dim", cfg["head_dim"], "--expect-h", cfg["h"],
            "--expect-vocab", cfg["vocab"],
            "--out", out, "--report", report] + list(extra)


def run_arm(bin_path, work, p, cfg, out_name, report_name, log_name, extra=()):
    cmd = [bin_path] + driver_args(
        work, p, cfg, os.path.join(work, "plan.tsv"), os.path.join(work, "slab.bin"),
        os.path.join(work, out_name), os.path.join(work, report_name), extra)
    rc, out = run(cmd, os.path.join(work, log_name))
    return rc, out


# ---------------------------------------------------------------------------------------------
# the three-arm agreement, the amortisation proof and the disjointness proof
# ---------------------------------------------------------------------------------------------
def three_way(v, p, d, host_rep, dev_rep, hostw, devw, expw, bandw, fc, en):
    """host twin vs device vs torch, per case.

    lm_head_kernels.h states the contract this implements: bit-exact where the two arms run the SAME
    written association (every SEQ case, and both arms of every case for the fields that are pure
    addressing), and band-judged where the association is a legal re-association (REV, SPLIT8). Both
    are printed either way, so 'the arms were in fact bit-identical even on REV' is a reported result
    rather than a relaxed assertion.

    WHERE A RE-ASSOCIATION'S BOUND COMES FROM. The oracle derives a band only for the words it has an
    expectation for, i.e. the honest BAND cases (oracle_arm.py fills `band.f64` for those and prints
    ORACLEFILL words=1884 of 27488). A re-associated INJECTION therefore has no band at its own words,
    and it is the case this function exists to judge — the bound is taken at its honest twin's
    corresponding word, which is exactly how adjudicate judges an injection's words today
    (`bt = twin["out_off"] + b * ts + j`). No new formula and no widened band: the same derived number
    the same repo already uses for the same (h, w) row. A word with no bound in either place is a
    refusal, not a skip — measured 2026-10-26, the first version of this function skipped nothing and
    refused at `case 84 word 1496 has no band`, because eight REV DROP_LAST_K injections (84, 86, 107,
    109, 131, 133, 155, 157) had never been executed on a device arm before."""
    n_exact = n_banded = 0
    n_word_self = n_word_twin = 0
    nd_bad, band_bad = [], []
    identical_rev = 0
    # The family table: device vs host twin per (encoding, association, path). The FMA-contraction
    # prediction this lane carries is a claim about ONE cell of it — Q8_0 under SEQ — so that cell has
    # to be readable on its own rather than dissolved into an aggregate. The keys come from the PLAN's
    # own population (nothing is typed here, so a family cannot be forgotten) and every family the plan
    # populates prints, which makes "a family stopped being judged" visible as an absent row.
    enc_name = {en["LMH_ENC_BF16"]: "bf16", en["LMH_ENC_Q8_0"]: "Q8_0"}
    assoc_name = {en["LMH_ASSOC_SEQ"]: "SEQ", en["LMH_ASSOC_REV"]: "REV",
                  en["LMH_ASSOC_SPLIT8"]: "SPLIT8"}
    fam = {}
    for c in p.cases:
        if not c["on_device"]:
            continue                                  # the device never ran it; see adjudicate
        slot = p.slot(c)
        words = [(c["out_off"] + b * slot + j) for b in range(c["bcount"]) for j in range(c["n_out"])]
        nd = sum(1 for at in words if not word_eq(hostw[at], devw[at]))
        f = fam.setdefault((enc_name[p.tables[c["table"]]["enc"]], assoc_name[c["assoc"]],
                            "gather" if c["path"] == en["LMH_PATH_GATHER"] else "dense"),
                           [0, 0, 0, 0])
        f[0] += 1
        f[1] += len(words)
        f[2] += nd
        f[3] += 1 if nd else 0
        if c["assoc"] == en["LMH_ASSOC_SEQ"]:
            n_exact += 1
            if nd:
                nd_bad.append("case %d %s: %d of %d words differ" % (c["idx"], c["name"], nd, len(words)))
        else:
            n_banded += 1
            worst = 0.0
            twin = None if c["mode"] == en["LMH_MODE_CORRECT"] else p.honest_twin(c)
            ts = p.slot(twin) if twin is not None else 0
            for b in range(c["bcount"]):
                for j in range(c["n_out"]):
                    at = c["out_off"] + b * slot + j
                    dd = abs(f32(hostw[at]) - f32(devw[at]))
                    bt, whence = at, "own"
                    if not bandw[at] > 0.0 and twin is not None:
                        bt, whence = twin["out_off"] + b * ts + j, "twin%d" % twin["idx"]
                    if not bandw[bt] > 0.0:
                        raise Fail("case %d word %d has no band (looked for it at %s word %d): a "
                                   "re-association cannot be bounded"
                                   % (c["idx"], at, whence, bt))
                    if whence == "own":
                        n_word_self += 1
                    else:
                        n_word_twin += 1
                    worst = max(worst, dd / bandw[bt])
            if worst > 1.0:
                band_bad.append("case %d %s: arms apart by %.3g x band" % (c["idx"], c["name"], worst))
            if nd == 0:
                identical_rev += 1
    print("THREEWAY reassociated_words=%d bound_at_own_word=%d bound_at_honest_twin=%d"
          % (n_word_self + n_word_twin, n_word_self, n_word_twin))
    fam_words = sum(f[1] for f in fam.values())
    fam_diff = sum(f[2] for f in fam.values())
    for key in sorted(fam):
        f = fam[key]
        print("THREEWAYFAM enc=%s assoc=%s path=%s cases=%d words_compared=%d words_differing=%d "
              "cases_diverging=%d" % (key[0], key[1], key[2], f[0], f[1], f[2], f[3]))
    print("THREEWAYFAM families=%d words_compared=%d words_differing=%d (aggregate of the rows above; "
          "a row count that does not sum to these is a parser bug, not a clean run)"
          % (len(fam), fam_words, fam_diff))
    print("THREEWAY seq_cases=%d reassociated_cases=%d reassociated_bit_identical=%d "
          "seq_diverging_cases=%d reassociated_outside_band=%d"
          % (n_exact, n_banded, identical_rev, len(nd_bad), len(band_bad)))
    v.check("three-arm: the host twin and the device agree bit-for-bit on every SEQ case",
            n_exact > 0 and not nd_bad, n_exact, "; ".join(nd_bad[:4]) or "0 diverging cases")
    v.check("three-arm: the arms stay inside the derived band on every re-association",
            not band_bad, n_banded, "; ".join(band_bad[:4]) or "all inside")
    v.note("re-associations-were-also-bit-identical",
           "%d of %d REV/SPLIT8 cases had ZERO words apart between the arms (band judgement was "
           "available but not needed)" % (identical_rev, n_banded))
    # both arms against the oracle, per class: the third side of the triangle
    def n_outside(c, words):
        slot = p.slot(c)
        out = 0
        for b in range(c["bcount"]):
            for j in range(c["n_out"]):
                at = c["out_off"] + b * slot + j
                if abs(f32(words[at]) - f32(expw[at])) > bandw[at]:
                    out += 1
        return out

    per_arm = {}
    excl = [0, 0, 0]                                 # cases, words, would-be-outside words
    for c in p.cases:
        if c["derived"] != "BAND":
            continue
        for nm, words in (("host", hostw), ("device", devw)):
            # The device is not asked to be inside the band of a value it was told not to compute. The
            # 14 on_device=0 `--deep-rows` cases are host-only BY PLAN, and their device words are the
            # zeroed output buffer; judging them here asked the device to reproduce an oracle value it
            # never wrote, and produced 382 outside-band words out of 382 (measured 2026-10-26, first
            # device run this function ever executed). The claim that DOES cover those words is
            # adjudicate's `host-only case was not written by this arm`, which requires every one of
            # them to be exactly zero — so nothing is let off, and the exclusion is counted and printed
            # rather than silently dropped.
            if nm == "device" and not c["on_device"]:
                excl[0] += 1
                excl[1] += c["bcount"] * c["n_out"]
                excl[2] += n_outside(c, words)
                continue
            e = per_arm.setdefault(nm, [0, 0])
            e[0] += c["bcount"] * c["n_out"]
            e[1] += n_outside(c, words)
    print("THREEWAY oracle words=%s outside_band=%s device_skipped_host_only cases=%d words=%d "
          "outside_there=%d"
          % ({k: val[0] for k, val in per_arm.items()}, {k: val[1] for k, val in per_arm.items()},
             excl[0], excl[1], excl[2]))
    v.check("three-arm: neither arm leaves the derived band against the torch oracle",
            per_arm and all(val[1] == 0 for val in per_arm.values()) and sum(
                val[0] for val in per_arm.values()) > 0,
            sum(val[0] for val in per_arm.values()),
            "outside=%s" % ({k: val[1] for k, val in per_arm.items()},))


def amortisation_and_disjointness(v, p, d, rep, tag, fc, en, results):
    """The two claims this lane can make WITHOUT a timer, both read off the traced read-set."""
    # ---- amortisation: batching must buy weight reads, and only weight reads ------------------
    # The families are the per-row cases' OWN declarations (their grp names the batched case they are
    # the control for); a batched case with no per-row control is not an amortisation claim, it is one
    # of the injections' honest baselines. Completeness is then a finding: a family that lost an arm is
    # reported, and the ratio is only asserted over families that are whole.
    groups = {}
    orphans = []
    for c in p.cases:
        if c["batching"] == en["LMH_BATCH_PERROW"]:
            if c["grp"] < 0:
                orphans.append("case %d %s is per-row but names no batched control" % (c["idx"],
                                                                                        c["name"]))
                continue
            groups.setdefault(c["grp"], {"batched": None, "perrow": []})["perrow"].append(c)
    by_idx = dict((c["idx"], c) for c in p.cases)
    for gi in sorted(groups):
        b = by_idx.get(gi)
        if b is not None and b["mode"] == en["LMH_MODE_CORRECT"] \
                and b["batching"] == en["LMH_BATCH_BATCHED"]:
            groups[gi]["batched"] = b
    n_groups = 0
    bad = []
    skipped_host_only = []
    for gi, g in sorted(groups.items()):
        b, per = g["batched"], g["perrow"]
        if b is None:
            bad.append("group %d: no batched honest case to amortise (the control points at %s)"
                       % (gi, "nothing"))
            continue
        # An amortisation ratio is a claim about a READ SET, so it can only be read off the arm that
        # produced it. The `--deep-rows` scaling families (groups 62, 67) are on_device=0 BY PLAN, so on
        # the device arm every one of their counters is the untouched init state — measured 2026-10-26
        # on the first device run, the device read `batched bytes 0 != geometry 131072` for both of them
        # and, worse, the bcount-independence witness below would have called two all-zero families
        # "flat". Excluding them is not a relaxation: the exclusion is counted and printed, the host
        # arm still judges all ten families, and a family that lost ONE arm to on_device still fails the
        # completeness test right here (same convention adjudicate and the reloop/disjointness loops use).
        if tag != "host" and not b["on_device"]:
            skipped_host_only.append(gi)
            continue
        if tag != "host":
            per = [c for c in per if c["on_device"]]
        if len(per) != b["bcount"]:
            bad.append("group %d: %d per-row controls for bcount=%d" % (gi, len(per), b["bcount"] if b else -1))
            continue
        n_groups += 1
        bb = int(rep[b["idx"]]["bytes"])
        pb = sum(int(rep[c["idx"]]["bytes"]) for c in per)
        t = p.tables[b["table"]]
        want_rows = b["nrows"] if b["path"] == en["LMH_PATH_DENSE"] else b["n_ids"]
        geometry_bytes = want_rows * t["row_b"]
        if pb != b["bcount"] * bb:
            bad.append("group %d: per-row bytes %d != %d x batched %d" % (gi, pb, b["bcount"], bb))
        if bb != geometry_bytes:
            bad.append("group %d: batched bytes %d != geometry %d" % (gi, bb, geometry_bytes))
        print("AMORTISE tag=%s group=%d rows=%d bcount=%d batched_bytes=%d perrow_bytes_sum=%d "
              "ratio=%.2f geometry=%d"
              % (tag, gi, want_rows, b["bcount"], bb, pb,
                 (pb / bb) if bb else 0.0, geometry_bytes))
    v.check("%s batching amortises the weight read by exactly bcount" % tag,
            n_groups > 0 and not bad and not orphans, n_groups,
            "; ".join((bad + orphans)[:4]) or "all groups exact")
    # bytes must not depend on how many hidden vectors share the walk
    curve_all = [c for c in p.cases if c["mode"] == en["LMH_MODE_CORRECT"]
                 and c["batching"] == en["LMH_BATCH_BATCHED"] and c["path"] == en["LMH_PATH_DENSE"]
                 and c["n_out"] > 0]
    curve = [c for c in curve_all if tag == "host" or c["on_device"]]
    print("AMORTISE tag=%s families_judged=%d families_host_only_excluded=%s "
          "batched_dense_cases=%d of %d (on_device filter)"
          % (tag, n_groups, skipped_host_only or "none", len(curve), len(curve_all)))
    by_shape = {}
    for c in curve:
        by_shape.setdefault((c["table"], c["row_lo"], c["nrows"], c["assoc"]), []).append(c)
    flat_bad = []
    tested = 0
    for key, cs in sorted(by_shape.items()):
        bs = set(int(rep[c["idx"]]["bytes"]) for c in cs)
        bc = set(c["bcount"] for c in cs)
        if len(bc) < 2:
            continue                      # one bcount cannot falsify "independent of bcount": not a
            #                               claim yet, so it is not counted and not called a pass
        tested += 1
        if len(bs) != 1:
            flat_bad.append("shape %s: bytes %s over bcounts %s" % (key, sorted(bs), sorted(bc)))
        else:
            print("AMORTISE tag=%s shape=table%d/rows%d+n%d bytes=%d constant across bcounts %s"
                  % (tag, key[0], key[1], key[2], sorted(bs)[0], sorted(bc)))
    v.check("%s addressed bytes are independent of bcount for the same row set" % tag,
            tested > 0 and not flat_bad, tested, "; ".join(flat_bad[:4]) or "flat")
    # ---- the re-looping batched kernel: caught on BYTES ALONE -----------------------------------
    # LMH_MODE_BATCH_RELOOPS walks the row set once per batch row while still writing one correct
    # output per batch row, so every arithmetic field is unchanged and only `bytes`/`loads` can see it.
    # The driver self-test's `batch-reloops-immune-at-bcount-1` is the immunity half (at bcount=1 the
    # injection is invisible BY CONSTRUCTION and must be classed IMMUNE there); this is the firing half,
    # and the claim being witnessed is that the words are bit-identical to the twin while bytes is
    # exactly bcount x the twin's -- an amortisation claim with no timer in it.
    rl_bad, rl_witnessed, rl_checked = [], 0, 0
    for c in p.cases:
        if c["mode"] != en["LMH_MODE_BATCH_RELOOPS"] or c["batching"] != en["LMH_BATCH_BATCHED"]:
            continue
        if tag != "host" and not c["on_device"]:
            continue                      # a host-only case has no device trace to witness
        if c["bcount"] <= 1:
            continue                      # immune by construction; the driver self-test owns that arm
        rl_checked += 1
        r, res = rep[c["idx"]], results.get(c["idx"])
        twin = p.honest_twin(c)
        if twin is None or res is None:
            rl_bad.append("case %d %s: no honest twin / no judgement to read the words-identical "
                          "claim from" % (c["idx"], c["name"]))
            continue
        tr = rep[twin["idx"]]
        tb, rb = int(tr["bytes"]), int(r["bytes"])
        if res.get("nd", -1) != 0:
            rl_bad.append("case %d %s: %d words differ from the twin, so this is not a bytes-only "
                          "catch" % (c["idx"], c["name"], res["nd"]))
        elif tb <= 0 or rb != c["bcount"] * tb:
            rl_bad.append("case %d %s: bytes %d != %d x twin %d" % (c["idx"], c["name"],
                                                                    rb, c["bcount"], tb))
        elif int(r["xorb"]) != int(tr["xorb"]) or int(r["vsum"]) != int(tr["vsum"]):
            rl_bad.append("case %d %s: a value digest moved (%s/%s, %s/%s), so the injection was not "
                          "confined to the read-set" % (c["idx"], c["name"], r["xorb"], tr["xorb"],
                                                        r["vsum"], tr["vsum"]))
        else:
            rl_witnessed += 1
            print("AMORTISE tag=%s reloops case=%d bcount=%d bytes=%d twin_bytes=%d ratio=%d "
                  "words_identical=1 xorb_unchanged=1 vsum_unchanged=1"
                  % (tag, c["idx"], c["bcount"], rb, tb, rb // tb))
    v.check("%s the re-looping batched kernel is caught on bytes alone (words bit-identical)" % tag,
            rl_checked > 0 and rl_witnessed == rl_checked, rl_checked,
            "witnessed=%d/%d; %s" % (rl_witnessed, rl_checked, "; ".join(rl_bad[:3]) or "clean"))
    # ---- disjointness: nobody read outside the table they were given ---------------------------
    # The predicate has to know which table is a TIED pair. `peer_hits` counts every read that lands
    # inside the peer EXTENT (lm_head_kernels.h §lmh_trace_range), and for a tied pair the peer extent
    # IS the own extent -- so requiring peer_hits == 0 of every case whose mode is not a foreign reader
    # was a claim the tied control could never satisfy (cases 32-35, honest-bf16-tied-a-*, peer_hits=4
    # =loads). A policy check that can only ever be red is a broken check, not a strict one; the strict
    # claims are (a) an UNTIED case reads neither its peer nor any foreign range, and (b) a TIED case
    # never counts a cross, because a read of the peer extent of a tied pair is a read of its own bytes.
    # Both are witnessed non-zero below, so neither can be satisfied by a counter that never moves.
    stray, tied_idx, tied_witness, untied_checked, tied_cross_max = [], [], [], 0, 0
    for c in p.cases:
        r = rep[c["idx"]]
        cross, peer, loads = int(r["cross"]), int(r["peer_hits"]), int(r["loads"])
        t = p.tables[c["table"]]
        pt = p.tables[t["peer"]] if t["peer"] >= 0 else None
        tied = pt is not None and pt["base"] == t["base"] and pt["nbytes"] == t["nbytes"]
        # The modes whose own rule says a foreign range must be touched: the two peer readers, and the
        # successor-row gather, whose successor of the last row is by construction outside the table
        # (and can land in the peer range). Exempting them here costs nothing: adjudicate requires
        # cross == n_out*nlanes for the gather and the exact extents for the peer readers, so a kernel
        # that read further than its one sanctioned range is still caught there.
        reads_foreign = c["mode"] in (en["LMH_MODE_TIED_ALIAS"], en["LMH_MODE_CROSS_TABLE_READ"],
                                      en["LMH_MODE_GATHER_ROW_OFF_1"])
        if tied:
            tied_idx.append(c["idx"])
            tied_cross_max = max(tied_cross_max, cross)
            if cross:
                stray.append("case %d %s tied pair: cross=%d, but a read of a tied peer extent is a "
                             "read of the own extent" % (c["idx"], c["name"], cross))
            if loads and peer != loads:
                stray.append("case %d %s tied pair: peer_hits=%d != loads=%d (every read of a tied "
                             "pair lands inside the peer extent)"
                             % (c["idx"], c["name"], peer, loads))
            if loads and peer > 0:
                tied_witness.append(c["idx"])
            continue
        untied_checked += 1
        if cross and not reads_foreign:
            stray.append("case %d %s cross=%d without a foreign-reading mode" % (c["idx"], c["name"],
                                                                                 cross))
        if peer and not reads_foreign:
            stray.append("case %d %s peer_hits=%d without a foreign-reading mode" % (c["idx"],
                                                                                      c["name"], peer))
    v.check("%s no untied case reads outside its own table unless its mode says it must" % tag,
            not stray, len(p.cases), "; ".join(stray[:4]) or "no stray reads")
    # the positive halves: a counter that reads 0 everywhere is vacuous, so the tied peer read and the
    # witness mode's cross must be seen MOVING, over a population that contains the cases at all.
    cross_w = [(c["idx"], int(rep[c["idx"]]["cross"]), int(rep[c["idx"]]["peer_hits"]))
               for c in p.cases
               if c["mode"] == en["LMH_MODE_CROSS_TABLE_READ"]
               and not (tag != "host" and not c["on_device"])
               and (p.tables[p.tables[c["table"]]["peer"]]["base"]
                    != p.tables[c["table"]]["base"])]
    print("DISJOINT tag=%s witness untied_cases=%d tied_cases=%d tied_peer_nonzero=%d "
          "tied_cross_max=%d untied_cross_arms=%d arms_with_cross>0=%d"
          % (tag, untied_checked, len(tied_idx), len(tied_witness), tied_cross_max,
             len(cross_w), sum(1 for _, cr, _ph in cross_w if cr > 0)))
    v.check("%s the tied control's peer read is witnessed non-zero (peer_hits is not vacuous)" % tag,
            len(tied_idx) > 0 and len(tied_witness) == len(tied_idx) > 0, len(tied_idx),
            "tied cases=%d with peer_hits>0=%d" % (len(tied_idx), len(tied_witness)))
    v.check("%s the cross-table-read witness moves `cross` on every untied arm" % tag,
            len(cross_w) > 0 and all(cr > 0 for _i, cr, _ph in cross_w), len(cross_w),
            "arms=%s" % (cross_w[:4],))
    # the two real tables: disjoint in the slab AND from different files, which is the untiedness
    real = [t for t in p.tables if t["real"] == 1 and t["kind"] == "real"]
    pairs = []
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, b = real[i], real[j]
            if a["base"] + a["nbytes"] <= b["base"] or b["base"] + b["nbytes"] <= a["base"]:
                continue
            if not (a["base"] == b["base"] and a["nbytes"] == b["nbytes"]):
                pairs.append("tables %d/%d overlap" % (a["idx"], b["idx"]))
    lmh_files = set(t["file"] for t in real if t["tensor"] == LMH_TENSOR)
    emb_files = set(t["file"] for t in real if t["tensor"] == EMBD_TENSOR)
    print("DISJOINT tag=%s real_tables=%d overlaps=%d lmhead_files=%s embd_files=%s same_file=%d"
          % (tag, len(real), len(pairs), sorted(os.path.basename(f) for f in lmh_files),
             sorted(os.path.basename(f) for f in emb_files),
             1 if lmh_files & emb_files else 0))
    v.check("%s the real tables occupy disjoint slab ranges" % tag, not pairs, len(real),
            "; ".join(pairs) or "disjoint")
    v.check("%s lm_head and the embeddings come from DIFFERENT shard files (the untiedness)" % tag,
            len(lmh_files) == 1 and len(emb_files) == 1 and not (lmh_files & emb_files), 2,
            "lm_head=%s embeddings=%s" % (sorted(lmh_files), sorted(emb_files)))


# ---------------------------------------------------------------------------------------------
# the pipeline, and the controls that prove it can go red
# ---------------------------------------------------------------------------------------------
def artifact_widths(gguf_path):
    if not os.path.exists(gguf_path):
        return {}
    import gguf_width_table as GWT
    try:
        # parse_gguf returns (version, n_tensors, rows, header_end, data_start) and each row is the
        # tuple (name, dims, type_id) — NOT dicts. The type id is resolved through the audited
        # decoder's own name table (tools/gguf_reader.py GGML_TYPES, the superset of the header
        # tokenizer's 22-entry map) so a width claim in this lane names the same spelling as the
        # dequantiser that would have to read the tensor.
        _ver, n_tensors, rows, _he, _ds = GWT.parse_gguf(gguf_path)
    except Exception as e:                            # an unreadable artifact is a finding, not a skip
        raise Fail("cannot read the published artifact %s: %r" % (gguf_path, e))
    if len(rows) != n_tensors:
        raise Fail("gguf_width_table parsed %d tensor rows, header declares %d"
                   % (len(rows), n_tensors))
    widths = {}
    for name, _dims, dt in rows:
        tn = GR.GGML_TYPES.get(dt)
        if tn is None:                                # an unmapped type id is a failure, not an absent width
            raise Fail("tensor %s has unmapped ggml type id %d in %s"
                       % (name, dt, os.path.join(REPO, "tools", "gguf_reader.py")))
        widths[name] = tn
    return widths


def tamper_q8_scale(slab_bytes, tables, en, fc, table_idx):
    """Double every Q8_0 block delta in one table: a UNIFORM POWER-OF-TWO rescale of the decoded
    values, which is exactly the perturbation an XOR of bit patterns can cancel. Byte geometry is
    unchanged, so nothing but the numbers move."""
    buf = bytearray(slab_bytes)
    t = tables[table_idx]
    if t["enc"] != en["LMH_ENC_Q8_0"]:
        raise Fail("tamper_q8_scale wants a Q8_0 table, got enc=%d" % t["enc"])
    nblk = t["nbytes"] // fc["LMH_Q8_0_BYTES"]
    for i in range(nblk):
        at = t["base"] + i * fc["LMH_Q8_0_BYTES"]
        d = struct.unpack_from("<e", buf, at)[0]
        struct.pack_into("<e", buf, at, d * 2.0)
    return bytes(buf), nblk


def do_run(a, v, device, fc, en, fields, modes, fmts):
    cfg = read_config()
    work = a.work
    p = build(cfg, fc, en, a.max_rows, a.deep_rows)
    print("POPULATION tables=%d cases=%d injections=%d hidden=%d (real=%d synth=%d) ids=%d "
          "out_floats=%d slab_bytes=%d"
          % (len(p.tables), len(p.cases),
             sum(1 for c in p.cases if c["mode"] != en["LMH_MODE_CORRECT"]), p.n_hidden,
             p.real_hidden, p.synth_hidden, len(p.ids), p.out_floats, p.slab_bytes))
    expect1 = dict((c["idx"], c["expect_name"] if c["expect_name"] != "AUTO" else "FIRES")
                   for c in p.cases)
    plan_path = os.path.join(work, "plan.tsv")
    lines1 = write_plan(plan_path, p, expect1)
    write_sidecars(work, p)

    # ---- the artifact's own widths, printed and cited, with Q6_K refused as a Q8_0 source -------
    widths = artifact_widths(GGUF)
    for nm in ("output.weight", "token_embd.weight"):
        print("ARTIFACTWIDTH tensor=%s stored=%s policy_a=Q8_0 citation=DECISIONS.md#D-10"
              % (nm, widths.get(nm, "absent")))
    q6 = sorted(n for n, t in widths.items() if t == "Q6_K")
    print("ARTIFACTWIDTH q6_k_tensors=%d first=%s note=Q6_K kernel support deferred "
          "(tools/gguf_reader.py's audited _deq_q6_k is the reuse point)"
          % (len(q6), q6[:1]))
    v.check("the stored width of the two tables this lane gates is read from the artifact, not assumed",
            all(nm in widths for nm in ("output.weight", "token_embd.weight")) and len(widths) > 10,
            len(widths), "output.weight=%s token_embd.weight=%s"
            % (widths.get("output.weight"), widths.get("token_embd.weight")))
    q8_sources = set(s.get("file", "") for s in p.sources
                     if s["enc"] == "Q8_0" and s["kind"] == "real")
    v.check("no Q8_0 table is sourced from the published GGUF (Q6_K would be read as Q8_0)",
            all(f.endswith(".safetensors") for f in q8_sources) and len(q8_sources) > 0,
            len(q8_sources), "Q8_0 sources=%s" % sorted(os.path.basename(f) for f in q8_sources))

    # ---- the oracle arm ------------------------------------------------------------------------
    if a.control == "q6k-source":
        rc, out = run([PY, ORACLE, "--work", work, "--control", "q6k-source"],
                      os.path.join(work, "control-q6k-source.log"))
        for line in out.splitlines():
            print(line)
        lines = out.splitlines()
        # The witness is the READER'S refusal, which exists only if the width check executed. Three
        # lines in that log start with REFUSE and name Q6_K: the control's own announcement
        # (`REFUSE_CONTROL injecting ...'), the setup-cannot-fire line (`REFUSE-CONTROL-SETUP no Q6_K
        # tensor in ...'), and the reader's verdict (`REFUSE table N: ... Q6_K is not a Q8_0 source').
        # Counting any REFUSE line let the control certify its own injection, and a control that
        # cannot find a Q6_K tensor in the artifact passed. Only the third line is evidence.
        announced = [l for l in lines if l.startswith("REFUSE_CONTROL ")]
        cannot = [l for l in lines if l.startswith("REFUSE-CONTROL-SETUP")]
        reader = [l for l in lines
                  if l.startswith("REFUSE table") and "Q6_K is not a Q8_0 source" in l]
        v.check("the q6k-source injection reaches the reader (an announcement is not a refusal)",
                len(announced) == 1 and not cannot, 1,
                "announced=%d setup_cannot_fire=%d" % (len(announced), len(cannot)))
        v.check("a Q6_K source tensor is refused rather than read as Q8_0",
                rc != 0 and len(reader) >= 1, 1, "rc=%d reader_refusal=%s" % (rc, reader[:1]))
        print("CONTROL q6k-source rc=%d reader_refusals=%d announced=%d setup_blocked=%d"
              % (rc, len(reader), len(announced), len(cannot)))
        # A control whose expected outcome IS a refusal has to leave the run red. Until now this one
        # printed two REFUSE lines, checked them, and returned PASS/exit 0 -- so `gate.py --control
        # q6k-source` was indistinguishable, by exit code alone, from a run in which the Q6_K tensor
        # had been silently accepted. That is the empty-parse-PASS trap in a refusal's clothes, and it
        # is why --selftest carried a finding for it.
        raise Fail("CONTROL q6k-source refused as designed (the reader printed %d refusal line(s)); "
                   "a control that achieved its refusal still exits non-zero"
                   % len(reader))
    cmd = [PY, ORACLE, "--work", work]
    if a.full_table:
        cmd.append("--full-table")
    rc, out = run(cmd, os.path.join(work, "oracle.log"))
    for line in out.splitlines():
        print(line)
    if rc != 0:
        raise Fail("oracle_arm.py failed rc=%d (see work/oracle.log)" % rc)
    fill = parse_tag(out, "ORACLEFILL", ["regions_filled", "words", "words_total",
                                         "band_nonzero", "band_nonpositive"], want=1)[0]
    want_words = sum(c["bcount"] * c["n_out"] for c in p.cases
                     if c["mode"] == en["LMH_MODE_CORRECT"])
    v.check("the oracle filled exactly the honest cases' words", int(fill["words"]) == want_words
            and int(fill["band_nonpositive"]) == 0
            and int(fill["words_total"]) == p.out_floats
            and int(fill["regions_filled"]) == want_words, want_words,
            "oracle=%s/%s total=%s expected=%d band_nonpositive=%s"
            % (fill["words"], fill["regions_filled"], fill["words_total"], want_words,
               fill["band_nonpositive"]))
    slab_bands = parse_tag(out, "ARTIFACTWIDTH", ["tensor", "stored"], want=2)
    v.check("the oracle and the gate agree on the artifact widths (parsed == printed)",
            all(d["stored"] == widths.get(d["tensor"]) for d in slab_bands), len(slab_bands),
            "oracle=%s gate=%s" % ([d["stored"] for d in slab_bands],
                                   [widths.get(d["tensor"]) for d in slab_bands]))

    d = Data(work, p, fc, en)

    # A control may legitimately move a column outside the expect column. When it does, it declares
    # which case and which column here, and the witness below requires that tampering to be VISIBLE in
    # the plan the drivers are handed: a control that silently did nothing stays red.
    control_cols = {}
    if a.control == "no-fire":
        victim = next(c for c in reversed(p.cases) if c["name"].startswith("inj-"))
        print("CONTROL no-fire neutralising case=%d %s (mode %d -> CORRECT): the run MUST go red"
              % (victim["idx"], victim["name"], victim["mode"]))
        victim["mode"] = en["LMH_MODE_CORRECT"]
        control_cols[victim["idx"]] = {12, 17}      # 12 is the mode column: the control's tampering

    # ---- derive the classes from the bytes, then rewrite the plan with what they said -----------
    expect2 = {}
    for c in p.cases:
        cls, pred = derive_class(c, d)
        expect2[c["idx"]] = cls
    lines2 = write_plan(plan_path + ".derived", p, expect2)
    # The derivation may move the expect column (field 17) and nothing else. Column INDICES, not
    # values: comparing the moved values would let a column whose new value happens to equal the old
    # expect value pass as "nothing but expect moved".
    diff = [i for i in range(len(lines1)) if lines1[i] != lines2[i]]
    moved = []
    for i in diff:
        if not lines1[i].startswith("C\t"):
            moved.append(("non-case line", i, None))
            continue
        f1, f2 = lines1[i].split("\t"), lines2[i].split("\t")
        allow = control_cols.get(int(f1[1]), set()) | {17}
        cols = [k for k in range(len(f1)) if f1[k] != f2[k]]
        if set(cols) - allow:
            moved.append((int(f1[1]), i, cols))
    v.check("deriving the classes changed the expect column and nothing else", not moved,
            len(diff), "%d records differ, unexpected columns=%s" % (len(diff), moved[:6]))
    for cidx in sorted(control_cols):
        seen = [i for i in diff if lines1[i].startswith("C\t")
                and int(lines1[i].split("\t")[1]) == cidx]
        v.check("the control's tampering reaches the plan the drivers run", bool(seen), 1,
                "case %d lines differing=%d" % (cidx, len(seen)))
    write_plan(plan_path, p, expect2)

    # ---- the host arm ---------------------------------------------------------------------------
    expw = words_of(os.path.join(work, "expect.f32"), p.out_floats)
    bandw = doubles_of(os.path.join(work, "band.f64"), p.out_floats)
    if a.control == "q6k-geometry":
        return control_q6k_geometry(v, p, cfg, fc, en, work)
    rc, out = run_arm(PROBE, work, p, cfg, "host_out.f32", "host_report.tsv", "host-probe.log")
    for line in out.splitlines():
        print(line)
    if rc != 0:
        raise Fail("host_probe rc=%d" % rc)
    prec = parse_tag(out, "PLAN", ["p_records", "tables_parsed", "cases_parsed"], want=1)[0]
    v.check("host arm parsed the plan exactly", int(prec["p_records"]) == 1
            and int(prec["tables_parsed"]) == len(p.tables)
            and int(prec["cases_parsed"]) == len(p.cases), len(p.cases),
            "p=%s tables=%s cases=%s" % (prec["p_records"], prec["tables_parsed"],
                                         prec["cases_parsed"]))
    runrec = parse_tag(out, "RUN", ["arm", "cases", "skipped", "items"], want=1)[0]
    v.check("host arm ran every case (nothing skipped)",
            runrec["arm"] == "host" and int(runrec["skipped"]) == 0
            and int(runrec["cases"]) == len(p.cases), len(p.cases),
            "cases=%s skipped=%s items=%s" % (runrec["cases"], runrec["skipped"], runrec["items"]))
    if a.control == "tampered-out":
        op = os.path.join(work, "host_out.f32")
        words = words_of(op, p.out_floats)
        # `c["derived"]` is only filled in by adjudicate(), which runs AFTER this control has tampered
        # the file, so reading it here was a KeyError traceback (present at HEAD: `tgt = next(c for c in
        # p.cases if c["derived"] == "BAND")` with adjudicate() 15 lines later). A traceback is not a
        # verdict either, so the control derived nothing and never even printed its own CONTROL line.
        # Ask the same derivation adjudicate will use, from the bytes as they are now.
        tgt = next(c for c in p.cases if derive_class(c, d)[0] == "BAND")
        at = tgt["out_off"]
        words[at] ^= 0x00000001
        with open(op, "wb") as fh:
            fh.write(struct.pack("<%dI" % p.out_floats, *words))
        print("CONTROL tampered-out flipped the low bit of word %d (case %d %s): the run MUST go red"
              % (at, tgt["idx"], tgt["name"]))
    if a.control == "no-report":
        rp = os.path.join(work, "host_report.tsv")
        keep = open(rp).read().splitlines(True)
        with open(rp, "w") as fh:
            fh.writelines(keep[:-1])
        print("CONTROL no-report dropped the last report row: parsed != printed MUST go red")
    if a.control in ("digest-radix", "count-radix"):
        # THE WRONG-RADIX CONTROL. One numeric token of one real R line is re-spelled in the OTHER
        # radix — same 32-bit value, different text — which is exactly the producer/consumer drift that
        # makes every digest comparison in the lane meaningless in whichever direction it errs: a bit
        # image written `%u` and read as hex (or a count written as hex and read as decimal) yields a
        # plausible number either way, so a value-only comparison cannot see it. The canonical form lives
        # in lmh_r_fmt[] (lmh_head_plan.h) and is SCRAPED here, not mirrored; read_report refuses any
        # token that is not the canonical spelling of its declared kind and names file, case, field and
        # token. An injection that found nothing to tamper is itself a refusal, never a pass.
        rp = os.path.join(work, "host_report.tsv")
        fld_name = "oxor" if a.control == "digest-radix" else "loads"
        want_hex = a.control == "digest-radix"          # a hex column spelled decimal, or vice versa
        at = fields.index(fld_name) + 1                  # +1: token 0 is the record letter
        if fmts[fields.index(fld_name)] != ("hex" if want_hex else "dec"):
            raise Fail("control %s wants %s to be a %s column; lmh_r_fmt[] says %s"
                       % (a.control, fld_name, "hex" if want_hex else "dec",
                          fmts[fields.index(fld_name)]))
        lines = open(rp).read().splitlines(True)
        hit = None
        for i, l in enumerate(lines):
            if not l.startswith("R\t"):
                continue
            f = l.rstrip("\n").split("\t")
            if len(f) != fc["LMH_F_R"]:
                raise Fail("%s: an R row has %d fields, the contract says %d"
                           % (rp, len(f), fc["LMH_F_R"]))
            val = int(f[at], 16) if f[at].startswith("0x") else int(f[at], 10)
            if (val & 0xFFFFFFFF) == 0:
                continue          # 0 spells the same in both radices: no witness, wrong or right
            bad = ("%d" % (val & 0xFFFFFFFF)) if want_hex else ("0x%08X" % (val & 0xFFFFFFFF))
            f[at] = bad
            lines[i] = "\t".join(f) + "\n"
            hit = (f[1], fld_name, bad, val & 0xFFFFFFFF)
            break
        if hit is None:
            raise Fail("control %s found no R row whose %s is non-zero: an injection that cannot fire "
                       "is a finding, not a pass" % (a.control, fld_name))
        with open(rp, "w") as fh:
            fh.writelines(lines)
        print("CONTROL %s file=%s case=%s field=%s value=0x%08X re-spelled in the other radix as %r: "
              "the parser MUST refuse it by name, not compare a plausible number"
              % (a.control, os.path.basename(rp), hit[0], hit[1], hit[3], hit[2]))
    hostw = words_of(os.path.join(work, "host_out.f32"), p.out_floats)
    host_rep = read_report(os.path.join(work, "host_report.tsv"), p, fields, fc, fmts)
    host_res = adjudicate(v, p, d, host_rep, hostw, expw, bandw, "host", fc, en)
    amortisation_and_disjointness(v, p, d, host_rep, "host", fc, en, host_res)

    if a.control == "scale-q8":
        return control_scale_q8(v, p, d, cfg, fc, en, work, host_rep, expw, bandw, fields, fmts)

    if a.check_only or not device:
        return v

    # ---- the device arm -------------------------------------------------------------------------
    rc, out = run_arm(DRIVER, work, p, cfg, "out.f32", "report.tsv", "driver.log",
                      extra=["--device-only"])
    for line in out.splitlines():
        print(line)
    if rc != 0:
        raise Fail("device driver rc=%d" % rc)
    drec = parse_tag(out, "RUN", ["arm", "cases", "skipped", "items", "planned"], want=1)[0]
    n_off = sum(1 for c in p.cases if not c["on_device"])
    v.check("device arm ran exactly the on_device cases",
            drec["arm"] == "device" and int(drec["planned"]) == len(p.cases) - n_off
            and int(drec["skipped"]) == n_off and int(drec["cases"]) == len(p.cases) - n_off,
            len(p.cases) - n_off, "planned=%s skipped=%s cases=%s (host-only cases=%d)"
            % (drec["planned"], drec["skipped"], drec["cases"], n_off))
    devw = words_of(os.path.join(work, "out.f32"), p.out_floats)
    dev_rep = read_report(os.path.join(work, "report.tsv"), p, fields, fc, fmts)
    dev_res = adjudicate(v, p, d, dev_rep, devw, expw, bandw, "device", fc, en)
    amortisation_and_disjointness(v, p, d, dev_rep, "device", fc, en, dev_res)
    launch_bad = []
    for c in p.cases:
        got = int(dev_rep[c["idx"]]["launches"])
        if c["on_device"] and got < 1:
            launch_bad.append("case %d launched %d blocks" % (c["idx"], got))
        if not c["on_device"] and got != 0:
            launch_bad.append("host-only case %d reports %d launches" % (c["idx"], got))
        hg = int(host_rep[c["idx"]]["launches"])
        if hg != 1:
            launch_bad.append("host case %d reports %s launches" % (c["idx"], hg))
    v.check("launch counts: one host call per case, at least one grid per device case",
            not launch_bad, len(p.cases), "; ".join(launch_bad[:4]) or "consistent")
    three_way(v, p, d, host_rep, dev_rep, hostw, devw, expw, bandw, fc, en)
    return v


def control_q6k_geometry(v, p, cfg, fc, en, work):
    """A Q6_K tensor silently accepted as Q8_0 must be refused by BOTH sides.

    The byte geometry is derived from the artifact's own block table (210 B per 256 weights, via
    tools/gguf_width_table.py's BLK), never typed here. If the plan's Q8_0 table is given that
    geometry, the C++ verifier refuses (`FAIL verify ... the encoding says`) and the oracle refuses;
    a plan that walked through either would mean the width was never actually checked."""
    import gguf_width_table as GWT
    # BLK rows are (elts per block, bytes per block, the ggml-common.h expression it came from) -- a
    # 3-tuple (tools/gguf_width_table.py §BLK). Unpacking two names from it raised ValueError before
    # the CONTROL line could print, so this control had never executed at all: rc=1 with no marker is a
    # traceback, not a verdict, and the Q6_K geometry was unproven on both drivers.
    elts, blk, geom_expr = GWT.BLK["Q6_K"]
    h = p.cfg["h"]
    q6_row = (h // elts) * blk
    t = next(t for t in p.tables if t["enc"] == en["LMH_ENC_Q8_0"] and t["kind"] == "real")
    want = t["nbytes"]
    print("CONTROL q6k-geometry table=%d enc=Q8_0 honest_bytes=%d q6k_geometry=%d (%d B / %d elts "
          "from tools/gguf_width_table.py BLK, expression %s)"
          % (t["idx"], want, q6_row * t["v_rows"], blk, elts, geom_expr))
    if q6_row * t["v_rows"] == want:
        raise Fail("the Q6_K and Q8_0 geometries coincide at h=%d: this control cannot fire and the "
                   "refusal is unproven" % h)
    plan_path = os.path.join(work, "plan.tsv")
    lines = open(plan_path).read().splitlines()
    outl = []
    for l in lines:
        f = l.split("\t")
        if f[0] == "T" and int(f[1]) == t["idx"]:
            f[5] = str(q6_row * t["v_rows"])
        outl.append("\t".join(f))
    with open(os.path.join(work, "plan_q6k.tsv"), "w") as fh:
        fh.write("\n".join(outl) + "\n")
    bad = []
    for nm, binp in (("host_probe", PROBE), ("device", DRIVER)):
        cmd = [binp] + driver_args(work, p, cfg, os.path.join(work, "plan_q6k.tsv"),
                                   os.path.join(work, "slab.bin"),
                                   os.path.join(work, "out_q6k.f32"),
                                   os.path.join(work, "rep_q6k.tsv"))
        rc, out = run(cmd, os.path.join(work, "q6k-%s.log" % nm))
        refused = rc != 0 and any(l.startswith("FAIL verify") or l.startswith("FAIL plan")
                                  for l in out.splitlines())
        print("CONTROLARM %s rc=%d refused=%d first=%s" % (nm, rc, 1 if refused else 0,
                                                           [l for l in out.splitlines()
                                                            if l.startswith("FAIL")][:1]))
        if not refused:
            bad.append(nm)
    v.check("a Q6_K-shaped table declared Q8_0 is refused by both drivers", not bad, 2,
            "accepted it silently: %s" % (",".join(bad) or "none"))
    if not bad:
        # Printed only when BOTH arms actually refused, which is what makes it a marker rather than a
        # line that prints whatever happened; --selftest requires it and requires rc != 0.
        print("CONTROLARM-REFUSED host_probe=1 device=1 honest_bytes=%d q6k_geometry=%d "
              "geom=%d B/%d elts expression=%s" % (want, q6_row * t["v_rows"], blk, elts, geom_expr))
        raise Fail("CONTROL q6k-geometry: both drivers refused the Q6_K-shaped Q8_0 table as designed "
                   "(%d B vs %d B); a control that achieved its refusal still exits non-zero"
                   % (want, q6_row * t["v_rows"]))
    return v


def control_scale_q8(v, p, d, cfg, fc, en, work, host_rep, expw, bandw, fields, fmts):
    """A uniform power-of-two rescale of the Q8_0 deltas: the class of value bug an XOR of bit
    patterns provably cancels (the driver self-test's `xor-digest-blind-to-uniform-rescale` is the
    witness). The gate must go red on it, and `vsum` must be the field that moves — `bsum` cannot see
    a decode change at all and `xorb` is the one that has been measured to coincide."""
    t = next(tab for tab in p.tables if tab["enc"] == en["LMH_ENC_Q8_0"] and tab["kind"] == "real")
    clean = d.slab
    scaled, nblk = tamper_q8_scale(clean, p.tables, en, fc, t["idx"])
    with open(os.path.join(work, "slab_scaled.bin"), "wb") as fh:
        fh.write(scaled)
    cmd = [PROBE] + driver_args(work, p, cfg, os.path.join(work, "plan.tsv"),
                                os.path.join(work, "slab_scaled.bin"),
                                os.path.join(work, "out_scaled.f32"),
                                os.path.join(work, "report_scaled.tsv"))
    rc, out = run(cmd, os.path.join(work, "host-probe-scaled.log"))
    for line in out.splitlines():
        print(line)
    if rc != 0:
        raise Fail("the scaled slab broke a structural check instead of a value check (rc=%d); "
                   "the control expects to be caught by the judge" % rc)
    rep2 = read_report(os.path.join(work, "report_scaled.tsv"), p, fields, fc, fmts)
    words2 = words_of(os.path.join(work, "out_scaled.f32"), p.out_floats)
    moved_v = moved_x = same_x = n_traced = moved_nonexempt = 0
    must_move, exempt, blind = [], [], []
    for c in p.cases:
        if c["table"] != t["idx"] or not c["trace"]:
            continue
        n_traced += 1
        aa, bb = host_rep[c["idx"]], rep2[c["idx"]]
        vm = aa["vsum"] != bb["vsum"]
        xm = aa["xorb"] != bb["xorb"]
        moved_v += 1 if vm else 0
        moved_x += 1 if xm else 0
        same_x += 1 if not xm else 0
        # Why one case may legitimately not move, DERIVED from its own record rather than typed: the
        # tamper touched the block deltas of THIS table only, and only a decode that multiplies by the
        # delta can see them. So a case is exempt when part of its read-set lies outside the tampered
        # table (cross > 0: the untied tied-alias arm reads the PEER's bytes, and the successor-row
        # gather escapes into the 0xCD pad), or when the injection it carries throws the delta away
        # (LMH_MODE_Q8_DROP_SCALE digests the raw quants, so doubling the delta moves nothing).
        # Measured on this plan the derived exempt set is cases 119 (mode 3, reads the PEER table's
        # bytes), 120 (mode 8, the cross-table witness -- which moves anyway), 122 (mode 7) and 176
        # (mode 1, the successor row that escapes into the pad): 4 of 35 traced cases, and the 3 that
        # vsum also failed to move ({119, 122, 176}) are a subset of it. Anything else that fails to
        # move is a finding, not an exemption.
        escapes = int(aa["cross"]) > 0 or c["mode"] == en["LMH_MODE_Q8_DROP_SCALE"]
        (exempt if escapes else must_move).append(c["idx"])
        if vm and not escapes:
            moved_nonexempt += 1
        if not vm and not escapes:
            blind.append("case %d mode=%d cross=%s vsum=%s->%s xorb=%s->%s"
                         % (c["idx"], c["mode"], aa["cross"], aa["vsum"], bb["vsum"],
                            aa["xorb"], bb["xorb"]))
    print("SCALEDIFF table=%d blocks=%d traced_cases=%d vsum_moved=%d xorb_moved=%d "
          "xorb_unchanged=%d must_move=%d moved_nonexempt=%d exempt=%d exempt_cases=%s"
          % (t["idx"], nblk, n_traced, moved_v, moved_x, same_x, len(must_move), moved_nonexempt,
             len(exempt), exempt))
    inner = V()
    adjudicate(inner, p, d, rep2, words2, expw, bandw, "scaled", fc, en)
    print("CONTROL scale-q8 red_checks=%d of %d checks" % (len(inner.bad), inner.n))
    v.check("a uniform x2 rescale of the Q8_0 deltas makes the judge red",
            len(inner.bad) > 0, inner.n, "red checks=%d (%s)"
            % (len(inner.bad), ",".join(inner.bad[:3])))
    # `moved_v <= n_traced` was the second conjunct here, and it is true by construction: moved_v
    # counts a subset of n_traced, so the predicate was `moved_v > 0 and True` -- a decoration, in the
    # same family as a literal-True predicate. The claim that can actually go red is per-case: every
    # traced case whose read-set lies inside the tampered table, and whose decode uses the delta, moved
    # its vsum.
    v.check("vsum moves on every traced case of the tampered table that is not exempt by geometry",
            len(must_move) > 0 and moved_nonexempt == len(must_move) and not blind, len(must_move),
            "moved=%d/%d non-exempt (exempt=%d) vsum_moved_all=%d xorb_moved=%d xorb_unchanged=%d "
            "blind=%s"
            % (moved_nonexempt, len(must_move), len(exempt), moved_v, moved_x, same_x,
               blind[:4] or "none"))
    v.check("the rescale control has a non-empty population to be red about",
            n_traced > 0 and len(exempt) < n_traced, n_traced,
            "traced_cases=%d must_move=%d exempt=%d" % (n_traced, len(must_move), len(exempt)))
    print("CONTROL scale-q8 inner_red=%d/%d vsum_moved=%d moved_nonexempt=%d/%d exempt=%d"
          % (len(inner.bad), inner.n, moved_v, moved_nonexempt, len(must_move), len(exempt)))
    # Same class as the two q6k controls: this control's DESIGNED outcome is a red judge, so rc=0 on
    # success was a control a driver walked past (it printed SCALEDIFF with 103 red inner checks and
    # the outer run said RESULT PASS). The red now comes from the control itself, and the SCALEDIFF
    # marker is what distinguishes "caught it" from "did not catch it" -- in the latter the two checks
    # above are red and no FAIL names the control.
    raise Fail("CONTROL scale-q8 caught the uniform x2 delta rescale (%d red inner checks of %d; vsum "
               "moved on %d/%d non-exempt traced cases); a control that achieved its red still exits "
               "non-zero" % (len(inner.bad), inner.n, moved_nonexempt, len(must_move)))


# ---------------------------------------------------------------------------------------------
# --selftest: the driver rosters, the source scan, and the controls that must go red
# ---------------------------------------------------------------------------------------------
def selftest(v, a, fc, en, modes):
    here = HERE
    rosters = {}
    for nm, binp in (("host_probe", PROBE), ("device", DRIVER)):
        if not os.path.exists(binp):
            raise Fail("%s is missing; run build.sh" % binp)
        rc, out = run([binp, "--selftest", "--work", a.work],
                      os.path.join(a.work, "selftest-%s.log" % nm))
        lines = [l for l in out.splitlines() if l.startswith("SELFTEST name=")]
        recs = [kv(l) for l in lines]
        fin = parse_tag(out, "DRIVER", ["checks", "failures"], want=1)[0]
        for r in recs:
            if r["ok"] != "1":
                v.check("driver %s selftest %s" % (nm, r["name"]),
                        r["ok"] == "1" and r.get("note", "") == "",
                        1, "ok=%s note=%s" % (r["ok"], r.get("note", "")))
        v.check("%s --selftest exits 0" % nm, rc == 0, len(recs), "rc=%d" % rc)
        v.check("%s selftest has zero failures" % nm,
                int(fin["failures"]) == 0 and int(fin["checks"]) > 0, int(fin["checks"]),
                "checks=%s failures=%s" % (fin["checks"], fin["failures"]))
        v.check("%s selftest parsed == printed" % nm, len(recs) == int(fin["checks"]),
                int(fin["checks"]), "printed=%d parsed=%d" % (len(recs), int(fin["checks"])))
        names = [r["name"] for r in recs]
        rosters[nm] = names
        # every injection mode the header declares must have a FIRING witness in the roster: an
        # injection nobody ever demonstrated firing is the hole that lets a green mean nothing.
        missing = [mode_token(m) for m in modes
                   if not any(mode_token(m) in x for x in names)]
        v.check("%s roster shows a firing witness for every declared mode" % nm,
                len(modes) > 1 and not missing, len(modes),
                "no witness for: %s" % (",".join(missing) or "none"))
        witness = [r for r in recs if "xor-digest-blind" in r["name"]]
        v.check("%s carries the measured XOR-blindness witness for the digest" % nm,
                len(witness) == 1 and witness[0]["ok"] == "1", 1,
                "found=%d ok=%s" % (len(witness), [w["ok"] for w in witness]))
        report_w = [r for r in recs if "report-field-count" in r["name"]]
        v.check("%s verifies its own report field count against LMH_F_R" % nm,
                len(report_w) == 1 and report_w[0]["ok"] == "1", 1,
                "found=%d ok=%s" % (len(report_w), [r["ok"] for r in report_w]))
    v.check("both drivers run the identical self-test roster",
            rosters["host_probe"] == rosters["device"] and len(rosters["device"]) > 20,
            len(rosters["device"]), "host=%d device=%d" % (len(rosters["host_probe"]),
                                                           len(rosters["device"])))
    # no verdict-bearing check in this file or the oracle may carry a constant predicate. AST, not
    # grep: a check whose predicate sits on a continuation line is exactly what a line scan misses.
    hits = []
    for f in ("gate.py", "oracle_arm.py"):
        src = open(os.path.join(here, f)).read()
        for nd in ast.walk(ast.parse(src)):
            if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute) \
                    and nd.func.attr == "check" and len(nd.args) >= 2:
                aa = nd.args[1]
                if isinstance(aa, ast.Constant) and aa.value in (True, False):
                    hits.append("%s:%d -> %r" % (f, nd.lineno, aa.value))
    v.check("no verdict-bearing check carries a constant predicate", not hits, 0,
            "offenders=%s" % (hits[:6],))
    # the controls: each must make the gate red, and say why
    for name, marker in (("no-fire", "CONTROL no-fire"),
                         ("tampered-out", "CONTROL tampered-out"),
                         ("no-report", "CONTROL no-report"),
                         ("digest-radix", "CONTROL digest-radix"),
                         ("count-radix", "CONTROL count-radix"),
                         ("q6k-geometry", "CONTROLARM-REFUSED"),
                         ("q6k-source", "REFUSE table"),
                         ("scale-q8", "SCALEDIFF")):
        rc, out = run([PY, os.path.join(here, "gate.py"), "--control", name,
                       "--work", a.work, "--max-rows", str(a.max_rows),
                       "--deep-rows", str(a.deep_rows)],
                      os.path.join(a.work, "control-%s.log" % name))
        text = out
        saw = [l for l in text.splitlines() if l.startswith(marker)]
        print("CONTROL %s rc=%d marker=%r lines=%d first=%s" % (name, rc, marker, len(saw),
                                                                saw[:1] and saw[0][:150]))
        v.check("control %s is refused or goes red" % name, rc != 0 and len(saw) >= 1, 1,
                "rc=%d marker_lines=%d" % (rc, len(saw)))
    for name in ("no-fire", "tampered-out", "no-report", "digest-radix", "count-radix",
                 "scale-q8", "q6k-geometry", "q6k-source"):
        log = os.path.join(a.work, "control-%s.log" % name)
        text = open(log).read() if os.path.exists(log) else ""
        bad = [l for l in text.splitlines() if l.startswith("CHECK ") and " ok=0 " in l]
        refused = [l for l in text.splitlines() if l.startswith("FAIL ")]
        print("CONTROLRED %s red_checks=%d refusals=%d first=%s"
              % (name, len(bad), len(refused), (bad + refused)[:1] and (bad + refused)[0][:140]))
        v.check("control %s was caught by a named check or an explicit refusal" % name,
                len(bad) + len(refused) >= 1, len(bad) + len(refused),
                "red checks: %d, refusals: %d" % (len(bad), len(refused)))
    # a malformed plan must be refused by the parser rather than judged
    rc, out = run([PROBE, "--plan", os.path.join(a.work, "plan.tsv"), "--slab",
                   os.path.join(a.work, "slab.bin"), "--hidden", os.path.join(a.work, "hidden.f32"),
                   "--ids", os.path.join(a.work, "ids.i32"), "--out", os.path.join(a.work, "x.f32"),
                   "--report", os.path.join(a.work, "x.tsv"), "--expect-cases", "99999"],
                  os.path.join(a.work, "control-wrong-count.log"))
    v.check("a plan whose case count disagrees with --expect-cases is refused", rc != 0, 1,
            "rc=%d" % rc)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="oracle + host twin only; no iGPU (the CI mode)")
    ap.add_argument("--device", action="store_true", help="add the device arm (wrapper only)")
    ap.add_argument("--control", default="")
    ap.add_argument("--max-rows", type=int, default=4,
                    help="rows per table that device-judged cases may touch")
    ap.add_argument("--deep-rows", type=int, default=32,
                    help="rows materialised for the host-only scaling cases")
    ap.add_argument("--full-table", action="store_true",
                    help="ask the oracle for the full-table CPU amortisation witness")
    a = ap.parse_args()
    if not os.path.isdir(a.work):
        os.makedirs(a.work)
    fc, en, fields, modes, fmts = contract()
    v = V()
    if a.selftest:
        return selftest(v, a, fc, en, modes).verdict("lm_head host selftest")
    try:
        do_run(a, v, a.device, fc, en, fields, modes, fmts)
    except Fail as e:
        print("FAIL %s" % e)
        v.refusal(e)
        return v.verdict("lm_head (refused)")
    return v.verdict("lm_head %s" % ("host+device" if a.device else "host-only"))


if __name__ == "__main__":
    sys.exit(main())
