#!/usr/bin/env python3
"""gguf_fast_dequant.py — vectorized dequantizers whose output is BIT-IDENTICAL to
`tools/gguf_reader.py`'s audited decoder on the same bytes, plus the gated byte-read path they share.

Why a second decoder exists. `gguf_reader.dequant` walks blocks in Python and returns a `list[float]`
(16.85 M elements/s measured on this box, Q4_K). `engine/m3_forward.py` dequantizes on the order of
10^10 elements per 40-block pass, so a vector decoder is a precondition of the lane rather than a
micro-optimization — and a second decoder is exactly what `pitfalls/40` §two-self-consistent-layouts
forbids *unless it is compared to the first*. So it is compared: `--xcheck` re-decodes real bytes of the
pinned artifact with both decoders and requires the two float32 arrays to agree **bit for bit**, on a
sample derived by index arithmetic from the file's own record list (never hand-listed), with a
non-vacuity leg that re-decodes under the audited reader's own `q4k_drop_min` injection and requires the
comparator to see the difference. Two decoders that were never shown able to disagree would be
self-agreement in a new costume.

Bit-identity is deliberate. The audited reader evaluates each element in double precision — `d*s1*q - m1`
with `m1 = dm*mg`, f16 deltas exact as doubles — and rounds to float32 once, at `np.array(vals, f32)`.
Every expression below keeps that association (per-sub-block `ds = d*sc` in float64, then `ds*q - dm*mg`,
then one cast), so the two decoders return the same bits and not two near-equal numbers. Block tables and
element order come from `tools/gguf_reader.py`'s module docstring, which transcribes them from
`ggml/src/ggml-common.h` and `ggml-quants.c` at the pinned checkout.

The second product, `half`, is the one the M3 band *proposal* needs and the audited decoder has no
interest in: the per-element half-width of the storage grid, i.e. the largest weight error the stored
width can hide at that element. Every family here is a uniform grid inside its sub-block:

    Q4_K  q in [0,15],    value = d*sc*q - dm*m    step = |d*sc|   half = |d*sc| / 2
    Q5_K  q in [0,31],    value = d*sc*q - dm*m    step = |d*sc|   half = |d*sc| / 2
    Q6_K  q in [0,63],    value = d*sc*(q - 32)    step = |d*sc|   half = |d*sc| / 2
    Q8_0  q in [-127,127], value = d*q             step = |d|      half = |d| / 2
    F32   the stored value IS the reference value  half = 0, exact family (D-01/D-01a/D-08 "exact")

`half` is a width statement read out of the file's own scales. It is not a tolerance, it licenses
nothing, and `engine/m3_gate.py` prints it only as an operand of a band PROPOSAL marked
`proposal=awaiting-main-agent`.

CPU-only. No device import, no timer, no `ms`/`tok/s`/`GB/s` anywhere, and no device-lock reference of
any kind appears in this file (`grep -c` for the lock script's name is 0 by design — `pitfalls/50`
§self-locking drivers).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE, os.path.join(REPO, "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gguf_reader as GR            # noqa: E402  the audited decoder — the reference side

QK = {"Q8_0": 32, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256}
BLK_BYTES = {"Q8_0": 34, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
SUPPORTED = ("F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K")


def _f16(raw, col):
    """One little-endian binary16 column of a block-strided uint8 view, widened to float64 exactly."""
    return np.ascontiguousarray(raw[:, col:col + 2]).view("<f2").astype(np.float64).reshape(-1)


def _blocks(buf, tname, n_elts):
    nb = -(-int(n_elts) // QK[tname])
    want = nb * BLK_BYTES[tname]
    if len(buf) < want:
        raise ValueError("short buffer: %s x %d elements needs %d B, got %d"
                         % (tname, n_elts, want, len(buf)))
    return np.frombuffer(memoryview(buf)[:want], dtype=np.uint8).reshape(nb, BLK_BYTES[tname])


def _scale_min_qk(scales):
    """(sc[8], mg[8]) from the 12 scale bytes, by `gguf_reader.qk_scale_min`'s route, vectorized."""
    nb = scales.shape[0]
    sc = np.empty((nb, 8), dtype=np.float64)
    mg = np.empty((nb, 8), dtype=np.float64)
    sc[:, 0:4] = scales[:, 0:4] & 63
    mg[:, 0:4] = scales[:, 4:8] & 63
    sc[:, 4:8] = (scales[:, 8:12] & 0xF) | ((scales[:, 0:4] >> 6) << 4)
    mg[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
    return sc, mg


def dequant_np(buf, tname, n_elts, want_half=False):
    """-> float32 array of `n_elts` values (and, with `want_half`, the per-element half-width).

    Everything is computed in the block's own multi-dimensional layout — `(nb, group, half, pos)` for
    the K-quant families — and flattened once at the end, because the flatten order IS the stored
    element order (`gguf_reader`: elements 32l..32l+31 of group g are the low nibbles of bytes
    32g..32g+31, and the high nibbles follow). Keeping the layout is also what avoids the
    sub-block-index gathers that made a first draft of this function run at 4 M elements/s; the
    sub-block scales broadcast from shape `(..., 1)` instead.
    """
    n = int(n_elts)
    if tname not in SUPPORTED:
        raise NotImplementedError("no vector decoder for %r; SUPPORTED=%s" % (tname, SUPPORTED))
    if tname == "F32":
        vals = np.frombuffer(memoryview(buf)[:4 * n], dtype="<f4")
        return (vals.copy(), np.zeros(n, dtype=np.float32)) if want_half else vals.copy()
    raw = _blocks(buf, tname, n)
    if tname == "Q8_0":
        d = _f16(raw, 0)
        q = np.ascontiguousarray(raw[:, 2:34]).astype(np.int8).astype(np.float64)
        vals = d[:, None] * q
        half = np.broadcast_to(np.abs(d)[:, None] / 2.0, (d.size, 32))
    elif tname in ("Q4_K", "Q5_K"):
        # (nb, 4 sub-blocks, 32 bytes) -> stack the two nibble halves on a new axis, giving
        # (nb, 4, 2, 32) whose flatten order is the stored element order. The K-quant scales broadcast
        # from (nb, 4, 2, 1) against exactly this shape.
        qs_off, qh_off = (16, None) if tname == "Q4_K" else (48, 16)
        qs = np.ascontiguousarray(raw[:, qs_off:qs_off + 128]).reshape(-1, 4, 32)
        q = np.stack([qs & 0xF, (qs >> 4) & 0xF], axis=2).astype(np.float64)   # (nb,4,2,32)
        if tname == "Q5_K":
            qh = np.ascontiguousarray(raw[:, qh_off:qh_off + 32])                 # (nb,32)
            bits = (qh[:, None, :] >> np.arange(8, dtype=np.uint8)[:, None]) & 1   # (nb,8,32)
            q = q + 16.0 * bits.reshape(-1, 4, 2, 32)
        sc, mg = _scale_min_qk(np.ascontiguousarray(raw[:, 4:16]))
        sc = sc.reshape(-1, 4, 2, 1)
        mg = mg.reshape(-1, 4, 2, 1)
        ds = _f16(raw, 0).reshape(-1, 1, 1, 1) * sc                             # audited: d*sc first
        vals = ds * q - (_f16(raw, 2).reshape(-1, 1, 1, 1) * mg)
        half = np.broadcast_to(np.abs(ds) / 2.0, q.shape)
    else:                                                    # Q6_K: d is the LAST two bytes.
        ql = np.ascontiguousarray(raw[:, 0:128])
        qh = np.ascontiguousarray(raw[:, 128:192])
        s8 = np.ascontiguousarray(raw[:, 192:208]).astype(np.int8).astype(np.float64)
        g, k, l = np.arange(2), np.arange(4), np.arange(32)
        # The index vectors carry no leading singleton: `ql[:, idx]` with a 4-D index would put the
        # advanced-index result at 5 dimensions and broadcast against the 4-D scale into an
        # (nb, nb, 2, 4, 32) product. A single-block buffer hides that completely, so the shape is
        # pinned here and the element-count check below is what catches a regression.
        ql_idx = g[:, None, None] * 64 + (k & 1)[None, :, None] * 32 + l
        qh_idx = g[:, None, None] * 32 + l
        base = (ql[:, ql_idx] >> (4 * ((k >> 1) & 1)).reshape(1, 1, 4, 1)).reshape(-1, 2, 4, 32) & 0xF
        ext = (qh[:, qh_idx] >> (2 * k).reshape(1, 1, 4, 1)).reshape(-1, 2, 4, 32) & 3
        q = (base | (ext << 4)).astype(np.float64)                             # (nb,2,4,32)
        # scale index is 8g + 2k + (l>>4): two values per 32-element chunk, so a 16-wide repeat
        # reproduces the per-element scale without a gather.
        sc_e = np.repeat(s8.reshape(-1, 2, 4, 2), 16, axis=-1)                  # (nb,2,4,32)
        ds = _f16(raw, 208).reshape(-1, 1, 1, 1) * sc_e
        vals = ds * (q - 32.0)
        half = np.abs(ds) / 2.0
    if vals.size != n:
        raise ValueError("%s decoder produced %d elements for %d requested: a layout or broadcast "
                         "slip, and a silent truncation here would decode a tensor into its own "
                         "prefix (pitfalls/40 §two-self-consistent-layouts)"
                         % (tname, vals.size, n))
    out = vals.reshape(-1).astype(np.float32)
    return (out, half.reshape(-1).astype(np.float32)) if want_half else out


# ---------------------------------------------------------------------------------------------
# The gated byte-read path — one implementation, shared by `--xcheck` and by engine/m3_forward.py.
# Bytes only ever come from `gguf_reader.Gguf.read`, which refuses until the audit selftest token is
# handed to it (`tools/audit_published_quant.py::require_selftest`, the marker block_forward's
# policy-A arm earns too), and addresses only ever come from `engine/loader.py`.
# ---------------------------------------------------------------------------------------------
class Artifact:
    """Byte ranges of the pinned GGUF, addressed by `engine/loader.py`, read under the audit token."""

    def __init__(self, path=None, ledger=None, why="policy-A weight reads"):
        import audit_published_quant as APQ
        import engine.loader as LD

        self.why = why
        self.ledger = ledger
        self.L = LD.open_loader()
        if path is not None and os.path.realpath(self.L.art.path) != os.path.realpath(path):
            raise ValueError("loader opened %s, caller asked for %s" % (self.L.art.path, path))

        class _Section:
            section = why

        if APQ.require_selftest(_Section()) is None:
            raise RuntimeError("%s refused the selftest token, so not one tensor-data byte may be read"
                               % APQ.SELFTEST_MARK)
        self.tok = APQ.SELFTEST_TOKEN
        self.hdr = self.L.art._hdr
        self.path = self.L.art.path
        self.bytes_read = 0
        self.reads = 0

    def close(self):
        self.hdr.close()

    def raw(self, file_off, length, name):
        if self.ledger is not None:
            self.ledger.note(self.path, "gguf-range")
        buf = self.hdr.read(self.tok, int(file_off), int(length), why=self.why)
        if len(buf) != int(length):
            raise RuntimeError("%s: asked %d B at %d, got %d" % (name, length, file_off, len(buf)))
        self.bytes_read += len(buf)
        self.reads += 1
        return buf

    def handle(self, name):
        return self.L.handle(name)

    def expert_range(self, name, idx):
        return self.L.expert_range(name, idx)

    def dequant(self, file_off, length, type_name, n_elts, name, want_half=False):
        return dequant_np(self.raw(file_off, length, name), type_name, n_elts, want_half=want_half)

    def tensor(self, name, want_half=False):
        """One whole tensor, dequantized to HF axis order (GGUF `ne` is fastest-first)."""
        h = self.handle(name)
        got = self.dequant(int(h["file_off"]), int(h["length"]), h["type_name"], int(h["n_elts"]),
                           name, want_half=want_half)
        vals, half = got if want_half else (got, None)
        dims = tuple(reversed(h["dims"]))
        out = vals.reshape(dims)
        return (out, half.reshape(dims), h) if want_half else (out, h)

    def expert(self, name, idx, shape_hf, want_half=False):
        """One routed expert's slab, contiguous in the file (GGUF's expert axis is its slowest)."""
        r = self.expert_range(name, idx)
        got = self.dequant(int(r["file_off"]), int(r["length"]), r["type_name"], int(r["elts"]),
                           name, want_half=want_half)
        vals, half = got if want_half else (got, None)
        if int(np.prod(shape_hf)) != int(r["elts"]):
            raise ValueError("%s expert %d: range holds %s elements, %s wants %s"
                             % (name, idx, r["elts"], tuple(shape_hf), int(np.prod(shape_hf))))
        out = vals.reshape(tuple(shape_hf))
        return (out, half.reshape(tuple(shape_hf)), r) if want_half else (out, r)


    def verify(self, rec, name):
        """Per-tensor byte-identity leg: this module's fast decode against the AUTHORITATIVE one.

        Authority, stated once and enforced: `tools/gguf_reader.py::dequant` is the authoritative
        decoder — it is what `engine/loader.py`-addressed bytes were gated with in M2
        (`engine/block_forward.py::WeightSource.get` calls it directly, and `loader.py` itself has no
        decoder, it supplies addresses). This file's decoder exists only because a Python per-element
        loop cannot decode 10^10 elements in a CPU pass, and it is admitted per run only by agreeing
        with the authoritative decoder on the sample below, for every tensor the pass reads. Any
        nonzero `ndiff_bytes` makes the caller refuse the read, so the fast path can never be used
        silently because it happened to be faster this time.

        The sample is derived, not typed: unquantized families are decoded **in full** (they are the
        never-quantized bucket — norms, router, GDN controls — where a 1-ULP difference is not a
        rounding difference, so block-sampling them would be the wrong strength), and quantized
        families on their first, middle and last block, indices by arithmetic from the record.
        """
        off, tname = int(rec["file_off"]), rec["type_name"]
        n = int(rec["n_elts"]) if "n_elts" in rec else int(rec["elts"])
        if tname == "F32":
            spans, n_sub, mode = [(off, 4 * n)], n, "full-tensor"
        else:
            nb = -(-n // QK[tname])
            picks = sorted({0, nb // 2, nb - 1})
            spans = [(off + i * BLK_BYTES[tname], BLK_BYTES[tname]) for i in picks]
            n_sub, mode = QK[tname], "derived-blocks(%d-of-%d)" % (len(picks), nb)
        ndiff = el = 0
        for (o, ln) in spans:
            buf = self.raw(o, ln, name)
            mine = dequant_np(buf, tname, n_sub)
            theirs = np.array(GR.dequant(buf, tname, n_sub), dtype=np.float32)
            el += n_sub
            ndiff += 4 * int((mine.view(np.uint32) != theirs.view(np.uint32)).sum())
        return dict(name=name, width=tname, mode=mode, blocks=len(spans), elements=el,
                    ndiff_bytes=ndiff)

    def row_gather(self, name, rows, k_elts):
        """Selected rows of a 2-D tensor, read row by row — the engine's gather read class.

        The per-row byte span is derived, not typed: `length * k_elts / n_elts`, exact for every type
        this file decodes because a row of 2048 elements is a whole number of 32/256-element blocks.
        Reading only the rows asked for is what keeps an embedding lookup at 32 rows instead of 248,320.
        """
        h = self.handle(name)
        if int(h["length"]) * int(k_elts) % int(h["n_elts"]) != 0:
            raise ValueError("%s: rows of %d elements do not tile its %d B record exactly"
                             % (name, k_elts, h["length"]))
        per_row = int(h["length"]) * int(k_elts) // int(h["n_elts"])
        base = int(h["file_off"])
        out = np.empty((len(rows), int(k_elts)), dtype=np.float32)
        half = np.empty((len(rows), int(k_elts)), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i], half[i] = self.dequant(base + int(r) * per_row, per_row, h["type_name"],
                                           k_elts, name, want_half=True)
        return out, half, h


# ---------------------------------------------------------------------------------------------
# --xcheck — the equivalence leg. Population derived from the file's own record list.
# ---------------------------------------------------------------------------------------------
def _sample_names(rows, tname, per_type=5):
    names = sorted(r[0] for r in rows if r[3] == tname)
    if not names:
        return []
    picks = sorted({0, len(names) // 4, len(names) // 2, (3 * len(names)) // 4, len(names) - 1})
    return [names[i] for i in picks[:per_type]]


def xcheck(path):
    import engine.loader as LD

    hdr = LD.parse_records(path)
    rows = [(r[1], r[2], r[4], r[3]) for r in hdr["rows"]]          # name, dims, offset, type
    types = sorted({r[3] for r in rows})
    art = Artifact(path, ledger=None, why="tools/gguf_fast_dequant.py --xcheck")
    findings, tot_el, tot_blk = [], 0, 0
    try:
        for tname in [t for t in types if t in SUPPORTED]:
            names = _sample_names(rows, tname)
            el = blk = bad = ctrl_caught = ctrl_el = 0
            ctrl_kind = "none-no-tensor-of-this-type-sampled"
            for nm in names:
                h = art.L.handle(nm)
                off, length, n = int(h["file_off"]), int(h["length"]), int(h["n_elts"])
                if tname == "F32":
                    step = min(n, 4096)
                    spans = [(off, step), (off + length - step, step)]
                else:
                    nb = -(-n // QK[tname])
                    picks = sorted({0, nb // 3, nb // 2, (2 * nb) // 3, nb - 1})
                    spans = [(off + i * BLK_BYTES[tname], BLK_BYTES[tname]) for i in picks]
                    # Single-block spans hide every nb-dependent broadcast slip (a first draft of
                    # dequant_np broadcast a 5-D q against a 4-D scale into an (nb,nb,2,4,32) product
                    # and still matched at nb=1), so multi-block runs are compared too.
                    big = min(64, nb)
                    spans += [(off, big * BLK_BYTES[tname]),
                              (off + (nb // 2) * BLK_BYTES[tname], big * BLK_BYTES[tname])]
                for (o, ln) in spans:
                    buf = art.raw(o, ln, nm)
                    n_sub = (ln // BLK_BYTES[tname]) * QK[tname] if tname != "F32" else ln // 4
                    mine = dequant_np(buf, tname, n_sub)
                    theirs = np.array(GR.dequant(buf, tname, n_sub), dtype=np.float32)
                    blk += 1
                    el += n_sub
                    if mine.shape != (n_sub,):
                        bad += 1
                        print("  BADSHAPE %s off=%d asked=%d got=%s" % (nm, o, n_sub, mine.shape))
                    same = (mine.shape == theirs.shape
                            and np.array_equal(mine.view(np.uint32), theirs.view(np.uint32)))
                    if not same:
                        bad += 1
                        where = ("shape" if mine.shape != theirs.shape
                                 else int(np.argmax(mine.view(np.uint32) != theirs.view(np.uint32))))
                        print("  MISMATCH %s off=%d n=%d first_element=%s" % (nm, o, n_sub, where))
                # Non-vacuity of the comparison itself. "0 bit mismatches" says nothing unless the
                # comparator can register a difference, so every type carries a control where the two
                # decoders are handed byte streams that differ: Q4_K gets the audited reader's own
                # `q4k_drop_min` re-decode, the rest get a one-byte-misaligned read
                # (`block_shift_1` in gguf_reader.INJECTIONS), which is a real converter bug.
                if tname == "Q4_K":
                    buf = art.raw(off, 8 * BLK_BYTES[tname], nm)
                    other = np.array(GR.dequant(buf, tname, 8 * QK[tname], inj="q4k_drop_min"),
                                     dtype=np.float32)
                    what = "q4k_drop_min re-decode"
                else:
                    span = 8 * BLK_BYTES[tname] if tname != "F32" else 4096
                    buf = art.raw(off, span, nm)
                    shv = art.raw(off + 1, span, nm)
                    other = np.array(GR.dequant(shv, tname, span // 4 if tname == "F32"
                                                else 8 * QK[tname]), dtype=np.float32)
                    what = "one-byte-shifted read"
                n_sub = 8 * QK[tname] if tname != "F32" else 1024
                ctrl_el += n_sub
                if not np.array_equal(other.view(np.uint32),
                                      dequant_np(buf, tname, n_sub).view(np.uint32)):
                    ctrl_caught += 1
                ctrl_kind = what
            tot_el += el
            tot_blk += blk
            ok = bool(names) and bad == 0 and el > 0 and ctrl_caught > 0
            print("  [%s] %-5s tensors=%d spans_compared=%d elements_compared=%d bit_mismatches=%d "
                  "control_seen_to_differ=%d/%d control=%s"
                  % ("OK  " if ok else "FAIL", tname, len(names), blk, el, bad, ctrl_caught,
                     len(names), ctrl_kind))
            if not ok:
                findings.append(tname)
    finally:
        art.close()
    print("XCHECK types_compared=%d of_stored_types=%d (%s) spans=%d elements_compared=%d "
          "bit_mismatches=%d verdict=%s no timing taken"
          % (len(SUPPORTED), len(types), ",".join(types), tot_blk, tot_el, len(findings),
             "BIT-IDENTICAL-TO-AUDITED-READER" if not findings else "FAIL %s" % findings))
    return 1 if findings else 0


def main(argv):
    if "--xcheck" in argv:
        path = os.environ.get("Q36_GGUF", os.path.join(
            REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"))
        return xcheck(path)
    print(__doc__)
    print("usage: gguf_fast_dequant.py --xcheck")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
