#!/usr/bin/env python3
"""probe_repro.py — scratch: can a torch CPU reference reproduce the pinned shared-expert pins?

Run with the venv interpreter (it has torch + numpy):
  /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python prototype/shared_expert/work/probe_repro.py
Read-only on the dumps. Prints no timing.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))

MAN = json.load(open(os.path.join(REPO, "checkpoints/dump-manifest-2026-10-26.json")))
REC = {r["name"]: r for r in MAN["dumps"]}


def arr(name):
    r = REC[name]
    raw = open(os.path.join(REPO, r["file"]), "rb").read()
    assert len(raw) == r["bytes"], (name, len(raw), r["bytes"])
    dt = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32,
          "torch.int64": torch.int64}[r["dtype"]]
    if dt is torch.bfloat16:
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
    else:
        t = torch.frombuffer(bytearray(raw), dtype=dt)
    return t.reshape(list(r["shape"])).clone()


def bits(t):
    return t.reshape(-1).view(torch.uint16).numpy() if t.dtype is torch.bfloat16 \
        else t.reshape(-1).view(torch.int32).numpy()


import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402

cfg = Qwen3_5MoeTextConfig(**json.load(
    open(os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")))["text_config"])
print("config hidden=%d shared_I=%d act=%s" % (cfg.hidden_size, cfg.shared_expert_intermediate_size,
                                               cfg.hidden_act))
torch.set_num_threads(8)
for block in ("layer0", "layer1", "layer39", "mtp0"):
    x = arr(block + ".input")
    gw = arr(block + ".mlp.shared_expert.gate_proj.weight")
    uw = arr(block + ".mlp.shared_expert.up_proj.weight")
    dw = arr(block + ".mlp.shared_expert.down_proj.weight")
    sg = arr(block + ".mlp.shared_expert_gate.weight")
    mlp = M.Qwen3_5MoeMLP(cfg, intermediate_size=cfg.shared_expert_intermediate_size)
    mlp = mlp.to(torch.bfloat16)
    gate = torch.nn.Linear(cfg.hidden_size, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gw); mlp.up_proj.weight.copy_(uw); mlp.down_proj.weight.copy_(dw)
        gate.weight.copy_(sg)
    x = x.reshape(-1, cfg.hidden_size)
    raw = mlp(x)
    prob = torch.nn.functional.sigmoid(gate(x))
    out = prob * raw
    for nm, mine, pinned in (("shared_raw", raw, block + ".shared_raw"),
                             ("shared_gate_prob", prob, block + ".shared_gate_prob"),
                             ("shared_out", out, block + ".shared_out")):
        p = arr(pinned)
        same = int((bits(mine) == bits(p)).sum())
        n = int(p.numel())
        mx = float((mine.reshape(-1).float() - p.reshape(-1).float()).abs().max())
        print("%-8s %-16s n=%6d bit-exact=%6d maxabs=%.6g" % (block, nm, n, same, mx))
    ex = arr(block + ".shared_out_exact")
    d = (out.reshape(-1).float() - ex.reshape(-1)).abs()
    rel = (d / ex.reshape(-1).abs().clamp_min(1e-30)).max()
    print("%-8s shared_out vs shared_out_exact  maxabs=%.6g maxrel=%.6g" % (block, d.max(), rel))
