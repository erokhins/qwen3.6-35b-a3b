#!/usr/bin/env python3
"""tools/hf_llama_crosscheck_d29.py — compare the HF (bf16, unquantized) candidate against the
llama.cpp (quantized) candidate, and refuse to take a side.

D-26 clause 3, re-affirmed by D-29: where HF and llama.cpp disagree, the disagreement is a finding
to report, never a side to adopt. This file therefore measures distances and counts them; it authors
no band, no floor and no tolerance, and prints `licence=llama-crosscheck(observation-only)
band=NONE adjudicated=no` on every magnitude line.

It imports NEITHER producer (tools/hf_full_forward_d29.py, tools/llama_aligned_run.py) NOR the
engine (engine/*). The stage mapping and the two numeric helpers are re-declared here, with their
source named, so that a fault in one tool cannot hide in the other:
  * stage map          — `engine/cross_oracle.py::STAGE_MAP` (llama names come from the vendored
                         `src/models/qwen35moe.cpp`)
  * widening a bf16 file to float64 — `engine/cross_oracle.py::values_of` (raw bits << 16, never a
                         `.to()` that would round, pitfalls/30 §the `.view` rule)
  * the ggml re-view   — `engine/cross_oracle.py::token_major` (ggml stores [hidden, tokens] with
                         ne[0] fastest, so the same bytes read into [tokens, hidden] ARE the
                         token-major matrix; the leg below witnesses that rather than trusting it)

CPU-only, no device, no lock, no device clock. **no timing taken.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_FIX = os.environ.get("HF_FIX", os.path.join(REPO, "checkpoints",
                                              "m3-candidate-hf-fixture-2026-10-27.json"))
LLAMA_FIX = os.environ.get("LLAMA_FIX", os.path.join(REPO, "checkpoints",
                                                    "m3-candidate-llama-fixture-2026-10-27.json"))
PIN = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.gguf-pin.json")
STAGES = ["x", "h_in", "mixer_out", "x1", "h_mlp", "moe_out", "out"]
# (hf stage) -> (llama name for a linear_attention block, llama name for a full_attention block)
STAGE_MAP = {
    "x": ("model.input_embed", "l_out"),
    "h_in": ("attn_norm", "attn_norm"),
    "mixer_out": ("linear_attn_out", "attn_output"),
    "x1": ("attn_residual", "attn_residual"),
    "h_mlp": ("attn_post_norm", "attn_post_norm"),
    "moe_out": ("ffn_out", "ffn_out"),
    "out": ("l_out", "l_out"),
}
FINDINGS: list[str] = []


def check(name, ok, detail):
    print(f"{'OK  ' if ok else 'FAIL'} {name} :: {detail}")
    if not ok:
        FINDINGS.append(name)
    return bool(ok)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def widen(path, dtype, nbytes):
    """One side's own bytes, widened to float64 without rounding."""
    raw = open(path, "rb").read(nbytes)
    if len(raw) != nbytes:
        raise SystemExit(f"REFUSED record-short {path}: {len(raw)} of {nbytes}")
    if "bfloat16" in dtype:
        return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").astype("float64")
    if dtype in ("f32", "float32"):
        return np.frombuffer(raw, dtype="<f4").astype("float64")
    raise SystemExit(f"REFUSED dtype-unknown {path}: {dtype}")


def token_major(rec, ggml):
    """(values, shape) with the row order the comparison is defined on: [tokens, hidden]."""
    v = widen(rec["file"], rec["dtype"], rec["bytes"])
    dims = list(rec["shape"])
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if len(dims) == 3 and dims[0] == 1:      # HF wrote [1, tokens, hidden]
        dims = dims[1:]
    if len(dims) != 2:
        raise SystemExit(f"REFUSED layout-unknown {rec['name']}: shape {rec['shape']}")
    n_tok, n_hid = (dims[1], dims[0]) if ggml else (dims[0], dims[1])
    if n_tok * n_hid != v.size:
        raise SystemExit(f"REFUSED record-size {rec['name']}: {rec['shape']} != {v.size}")
    return (v.reshape(n_tok, n_hid), (n_tok, n_hid))


def row_of(path, dtype, nbytes, row, n_rows, width):
    """One row of a [tokens, hidden] file, read from disk without loading the whole thing."""
    esz = 2 if "bfloat16" in dtype else 4
    if (row + 1) * width * esz > nbytes:
        raise SystemExit(f"REFUSED row-out-of-range {path}: row {row} of {nbytes // esz // width}")
    with open(path, "rb") as fh:
        fh.seek(row * width * esz)
        raw = fh.read(width * esz)
    if len(raw) != width * esz:
        raise SystemExit(f"REFUSED row-short {path}: row {row}")
    return widen_bytes(raw, dtype)


def widen_bytes(raw, dtype):
    if "bfloat16" in dtype:
        return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").astype("float64")
    return np.frombuffer(raw, dtype="<f4").astype("float64")


def rel(a, b):
    d = a - b
    denom = float(np.sqrt((b * b).sum())) or 1.0
    return (float(np.abs(d).max()), float(np.sqrt((d * d).sum())) / denom,
            float(np.sqrt((a * a).mean())), float(np.sqrt((b * b).mean())), int(a.size))


def main():
    hf = json.load(open(HF_FIX))
    ll = json.load(open(LLAMA_FIX))
    instr = json.load(open(ll["oracle"]["instrument_manifest"]))
    print(f"inputs  hf={os.path.relpath(HF_FIX, REPO)} records={len(hf['records'])} "
          f"aggregate_digest={hf['aggregate_digest']}")
    print(f"inputs  llama={os.path.relpath(LLAMA_FIX, REPO)} crosscheck_rows={len(ll['crosscheck'])} "
          f"argmax_id={ll['logits']['argmax_id']}")
    # --- stimulus identity: the precondition under which every number below means anything
    h_sha, l_sha = hf["stimulus"]["token_ids_sha256"], ll["stimulus"]["token_ids_sha256"]
    h_ids, l_ids = hf["stimulus"]["token_ids"], ll["stimulus"]["token_ids"]
    eq = sum(1 for a, b in zip(h_ids, l_ids) if a == b)
    check("stimulus_identical", h_sha == l_sha and eq == len(h_ids) == len(l_ids),
          f"elementwise {eq}/{len(h_ids)} equal length_equal={len(h_ids) == len(l_ids)} "
          f"hf_sha={h_sha} llama_sha={l_sha}")
    n_tok = len(h_ids)
    # --- integrity of the rows being compared: re-hash, do not read-and-believe
    hf_by = {r["name"]: r for r in hf["records"]}
    ll_by = {(r["block"], r["name"]): r for r in ll["crosscheck"]}
    for r in instr["dumps"]:
        if r.get("written") and r["block"] < 0:
            ll_by.setdefault((r["block"], r["name"]), r)
    h_bad = [r["name"] for r in hf["records"] if sha_file(r["file"]) != r["sha256"]]
    l_bad = [f"{b}/{n}" for (b, n), r in ll_by.items()
             if os.path.exists(r["file"]) and sha_file(r["file"]) != r["sha256"]]
    l_missing = [f"{b}/{n}" for (b, n), r in ll_by.items() if not os.path.exists(r["file"])]
    check("hf_row_digests_rehashed", not h_bad,
          f"rehashed={len(hf['records'])} mismatches={len(h_bad)} {h_bad[:3]}")
    check("llama_row_digests_rehashed", not l_bad and not l_missing,
          f"rehashed={len(ll_by) - len(l_missing)} of {len(ll_by)} mismatches={len(l_bad)} "
          f"missing={len(l_missing)} {(l_bad + l_missing)[:3]}")
    pin = json.load(open(PIN))
    check("oracle_reads_the_pinned_gguf", ll["oracle"]["gguf_sha256"] == pin["sha256"],
          f"llama_gguf={ll['oracle']['gguf_sha256'][:16]}… pin={pin['sha256'][:16]}…")
    # --- the layout witness: chosen re-view vs the other reading, on one cell, with a margin
    lin = set(instr["model"]["linear_layers"])
    probe = ll_by[(0, STAGE_MAP["out"][0])]
    lv = token_major(probe, ggml=True)[0]
    hv = token_major(hf_by["hf40.layer0.out"], ggml=False)[0]
    r_chosen = rel(lv, hv)[1]
    flat = widen(probe["file"], probe["dtype"], probe["bytes"])
    r_other = rel(flat.reshape(probe["shape"][0], -1).T.reshape(-1), hv.reshape(-1))[1]
    print(f"layout  cell=layer0.out relL2[token-major re-view]={r_chosen:.3e} "
          f"relL2[raw re-view]={r_other:.3e} chosen=token-major margin={r_other / max(r_chosen, 1e-12):.1f}x")
    check("layout_witnessed", r_other > 5.0 * r_chosen,
          f"chosen={r_chosen:.3e} other={r_other:.3e} margin_needed=5x (a wrong re-view scrambles "
          "element order and lands near sqrt(2); without this leg every delta below could be "
          "element order rather than arithmetic)")
    # --- the table: 40 blocks x 7 stages, every cell measured or refused with a named reason
    measured, refused, escalations = 0, [], 0
    print("")
    print("hf(bf16, unquantized) x llama(f32 from Q4_K) — denominator 40 blocks x 7 stages = 280 "
          "cells; every line is an observation, no threshold is attached to any of them")
    worst = {}
    for blk in range(40):
        for stage in STAGES:
            hname = f"hf40.layer{blk}.{stage}"
            hrec = hf_by.get(hname)
            if stage == "x":
                want = (-1, "model.input_embed") if blk == 0 else (blk - 1, "l_out")
            else:
                want = (blk, STAGE_MAP[stage][0] if blk in lin else STAGE_MAP[stage][1])
            lrec = ll_by.get(want)
            if hrec is None or lrec is None:
                refused.append(f"layer{blk}.{stage}:llama-row-absent{want}")
                continue
            a = token_major(hrec, ggml=False)
            b = token_major(lrec, ggml=True)
            if a[1] != b[1]:
                refused.append(f"layer{blk}.{stage}:shape hf={hrec['shape']} llama={lrec['shape']}")
                continue
            mx, rl, hr, lr, n = rel(a[0], b[0])
            measured += 1
            escalations += 1
            worst[stage] = max(worst.get(stage, 0.0), rl)
            print(f"  layer{blk:<3d} {stage:<10s} hf={hrec['dtype']}/{hrec['shape']} "
                  f"llama={lrec['dtype']}/{lrec['shape']} n={n} max_abs_delta={mx:.3e} "
                  f"relL2={rl:.3e} hf_rms={hr:.4f} llama_rms={lr:.4f} "
                  f"licence=llama-crosscheck(observation-only) band=NONE adjudicated=no")
    print(f"table   cells_measured={measured} cells_refused={len(refused)} "
          f"identity: {measured} + {len(refused)} = {measured + len(refused)} of 40x7=280")
    for r in refused:
        print(f"  refused {r}")
    check("cell_denominator", measured + len(refused) == 40 * len(STAGES),
          f"measured={measured} refused={len(refused)} expected=280")
    check("cells_actually_measured", measured > 0,
          f"measured={measured} (a 0 here would make the whole table vacuous)")
    print(f"worst-relL2-per-stage  " + " ".join(f"{s}={worst[s]:.3e}" for s in STAGES if s in worst))
    print(f"escalation  {escalations} HF-vs-llama divergences printed, 0 adjudicated, "
          "0 bands authored. D-26 clause 3: reported, not adopted; where HF's limit is tighter, "
          "HF's limit ships.")
    # --- logits and the argmax: the number M3 hangs on
    hl = next(r for r in hf["records"] if r["stage"] == "logits")
    ln_norm = ll_by[(-1, "result_norm")]
    ln_logits = ll_by[(-1, "result_output")]
    h_norm = hf_by["hf40.normed"]
    hn = row_of(h_norm["file"], h_norm["dtype"], h_norm["bytes"], n_tok - 1, 1, 2048)
    nn = widen(ln_norm["file"], ln_norm["dtype"], ln_norm["bytes"])
    mx, rl, hr, lr, n = rel(hn, nn)
    print(f"normed  hf_normed[last]={h_norm['shape']}[row {n_tok - 1}] llama_result_norm="
          f"{ln_norm['shape']} n={n} max_abs_delta={mx:.3e} relL2={rl:.3e} hf_rms={hr:.4f} "
          f"llama_rms={lr:.4f} licence=llama-crosscheck(observation-only) band=NONE adjudicated=no")
    hv_l = widen(hl["file"], hl["dtype"], hl["bytes"])
    lv_l = widen(ln_logits["file"], ln_logits["dtype"], ln_logits["bytes"])
    mx, rl, hr, lr, n = rel(hv_l, lv_l)
    top_h = np.argsort(-hv_l)[:5]
    print(f"logits  hf={hl['dtype']}/{hl['shape']} llama={ln_logits['dtype']}/{ln_logits['shape']} "
          f"n={n} max_abs_delta={mx:.3e} relL2={rl:.3e} hf_rms={hr:.4f} llama_rms={lr:.4f} "
          f"licence=llama-crosscheck(observation-only) band=NONE adjudicated=no")
    print(f"logits  hf_top5={' '.join(f'{int(i)}:{hv_l[int(i)]:.8f}' for i in top_h)}")
    print(f"logits  llama_top5={ll['logits']['top5']}  (llama row sha={ll['logits']['sha256'][:16]}… "
          f"hf row sha={hl['sha256'][:16]}…)")
    margin_h = float(np.sort(hv_l)[-1] - np.sort(hv_l)[-2])
    print(f"argmax  HF argmax_id={hf['logits']['argmax_id']}  llama argmax_id={ll['logits']['argmax_id']}  "
          f"equal={hf['logits']['argmax_id'] == ll['logits']['argmax_id']}  "
          f"hf_rank1_minus_rank2={margin_h:.6f}  llama_definition={ll['logits']['definition'][:60]}…")
    check("argmax_agrees", hf["logits"]["argmax_id"] == ll["logits"]["argmax_id"],
          f"hf={hf['logits']['argmax_id']} llama={ll['logits']['argmax_id']} "
          f"(if this is red it is the headline of the report, not something to smooth)")
    print(f"\nCROSSCHECK findings={len(FINDINGS)}{FINDINGS[:4]} cells_measured={measured} "
          f"cells_refused={len(refused)} escalations={escalations} "
          f"hf_argmax={hf['logits']['argmax_id']} llama_argmax={ll['logits']['argmax_id']} "
          f"bands_authored=0 no timing taken")
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    sys.exit(main())
