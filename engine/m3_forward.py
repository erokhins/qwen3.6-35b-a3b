#!/usr/bin/env python3
"""m3_forward.py — the engine's own 40-block forward over the aligned 32-token prompt.

WHAT THIS IS. M3's producer side (`AGENTS.md` §Milestone ladder; the gate design in
`engine/HANDOFF.md` §PASS 15 §5): embedding → 40 decoder blocks → final RMSNorm → `lm_head`, for the
pinned 32 ids, reading the **shipping artifact** `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
through `engine/loader.py` addresses and `tools/gguf_fast_dequant.py`'s decoder (shown bit-identical to
the audited `tools/gguf_reader.py` by that file's `--xcheck`). It emits the same stage set the
authoritative HF fixture carries — 7 stages × 40 blocks plus `normed` plus the logits row = 282
records — as f32 files with a per-record digest. Adjudication lives in `engine/m3_gate.py`; this file
prints records and operands, never a distance and never a band.

THE INDEPENDENCE RULE, enforced rather than asserted (D-29: "HF reads `models/qwen36-src/*.safetensors`,
the engine reads the GGUF; the two share no weight bytes"):
  * **no `transformers` import.** The GDN scan, the gated attention, the MoE and both RMSNorm forms are
    written here from `specification/03`, §04/§04b, §05, §06 and §11, cited where each is used. `torch`
    is arithmetic, not a producer.
  * **no fixture byte is an input.** The only prompt bytes read are the 32 ids at
    `checkpoints/dump-manifest-chain-2026-10-26.json::input.token_ids` — the pinned stimulus, not an
    activation. Every `open()` after import lands in a ledger the gate audits against both candidate
    fixtures' record lists; `--fault peek-hf-dump` exists so the gate can watch that counter FIRE
    instead of taking a 0 on faith (`pitfalls/40` §a-counter-that-reads-0).
  * **the comparator is a separate file.** `engine/m3_gate.py` reads the HF fixture; this file never
    reads it, never imports the gate, and prints no distance.

WHAT THE NUMBERS ARE. Every activation is F32, computed from the dequantized artifact values — the same
compute basis as `engine/block_forward.py`'s `policy_a` arm, kept identical so the two lanes are
comparable — while the reference's stream is BF16. Of the reference's dtype roundings, exactly two are
applied here because they change *values* rather than precision: R1's cast of the router logits and R5's
cast of the renormalised top-k weights (§11.2, AGENTS.md prediction 4). All the others are
`bf16_stream_rounding=not-applied`, printed because they are part of what a distance measures. No band,
floor, tolerance, factor or licence is authored here (D-18, D-20); the band operands this file prints are
consumed by a PROPOSAL in the gate marked `proposal=awaiting-main-agent`.

NO DEVICE, NO TIMING. `torch.cuda.is_available()` must be False or this file refuses to run; there is no
`--device`, no allocator call and no device-lock reference of any kind (`grep -c` for the lock script's
name is 0 here by design — `pitfalls/50` §self-locking drivers). Nothing prints `ms`, `tok/s` or
`GB/s`; host RSS and `MemAvailable` only. **no timing taken.**
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gguf_fast_dequant as FD                            # noqa: E402

CONFIG_PATH = os.path.join(REPO, "checkpoints", "Qwen3.6-35B-A3B.config.json")
CHAIN_PATH = os.path.join(REPO, "checkpoints", "dump-manifest-chain-2026-10-26.json")
STAGES = ("x", "h_in", "mixer_out", "x1", "h_mlp", "moe_out", "out")
L2NORM_EPS = 1e-6        # §04.4's l2norm eps: a spec constant that happens to equal rms_norm_eps
CHUNK = 64               # §04b.2: the prefill chunk width the reference's scan uses
PROBE_LAYERS = (0, 20, 39)   # derived-in-spirit probe sites for the band operands; named, not hidden
F32 = torch.float32
CFG = None                                                # set by load_geometry()


# ------------------------------------------------------------------ read ledger (independence witness)
class Ledger:
    """Every path this process opens after import, so the gate can prove what the engine read."""

    def __init__(self):
        self.paths = {}
        self._real = builtins.open

    def note(self, path, kind):
        try:
            self.paths.setdefault(os.path.realpath(str(path)), kind)
        except Exception:
            pass

    def _open(self, file, *a, **kw):
        self.note(file, "open")
        return self._real(file, *a, **kw)

    def install(self):
        builtins.open = self._open

    def remove(self):
        builtins.open = self._real

    def names(self):
        """Sorted `path::kind` strings — the ledger the gate audits against the fixture record lists."""
        return sorted("%s::%s" % (p, k) for p, k in self.paths.items())


# ------------------------------------------------------------------ geometry, all read from files
def load_geometry():
    """Every shape and constant, from `checkpoints/Qwen3.6-35B-A3B.config.json`. Nothing typed by hand."""
    global CFG
    tc = json.loads(open(CONFIG_PATH).read())["text_config"]
    h_k, h_v = int(tc["linear_num_key_heads"]), int(tc["linear_num_value_heads"])
    d_k, d_v = int(tc["linear_key_head_dim"]), int(tc["linear_value_head_dim"])
    rp = tc["rope_parameters"]
    CFG = dict(d=int(tc["hidden_size"]), V=int(tc["vocab_size"]), L=int(tc["num_hidden_layers"]),
               layer_types=list(tc["layer_types"]), act_dtype=getattr(torch, tc["dtype"]),
               scan_dtype=getattr(torch, tc["mamba_ssm_dtype"]), eps_rms=float(tc["rms_norm_eps"]),
               h_k=h_k, h_v=h_v, d_k=d_k, d_v=d_v, expansion=h_v // h_k,
               key_dim=d_k * h_k, value_dim=d_v * h_v,
               conv_dim=d_k * h_k + d_k * h_k + d_v * h_v, k_conv=int(tc["linear_conv_kernel_dim"]),
               n_experts=int(tc["num_experts"]), top_k=int(tc["num_experts_per_tok"]),
               i_e=int(tc["moe_intermediate_size"]), i_s=int(tc["shared_expert_intermediate_size"]),
               n_q=int(tc["num_attention_heads"]), n_kv=int(tc["num_key_value_heads"]),
               head_dim=int(tc["head_dim"]), partial_rotary=float(rp["partial_rotary_factor"]),
               rope_theta=float(rp["rope_theta"]), l2norm_eps=L2NORM_EPS, chunk=CHUNK)
    return CFG


def token_ids():
    """The pinned 32 ids, from the chain manifest, with that manifest's own encoding re-derived."""
    doc = json.loads(open(CHAIN_PATH).read())
    ids = [int(i) for i in doc["input"]["token_ids"]]
    sha = hashlib.sha256(b"".join(struct.pack("<q", i) for i in ids)).hexdigest()
    return ids, sha, doc["input"].get("token_ids_sha256")


def v_order_gguf_to_hf(h_v, h_k, expansion, head_dim):
    """GGUF tiled V axis -> HF grouped V axis. Same map as
    `engine/block_forward.py::v_order_gguf_to_hf` (§04.9, A-12); re-declared so this file imports no
    band machinery, and `engine/m3_gate.py` compares the two spellings bit for bit instead of trusting
    the comment."""
    idx = torch.arange(h_v)
    order = torch.empty(h_v, dtype=torch.long)
    order[(idx % h_k) * expansion + (idx // h_k)] = idx
    return order.repeat_interleave(head_dim)


# The tensors carrying a V axis: (axis, first row/col of the V block, elements per V head). §04.9's
# tensor list; same content as engine/block_forward.py::V_ORDERED.
V_ROLES = {"attn_qkv": ("row", 4096, 128), "attn_gate": ("row", 0, 128),
           "ssm_alpha": ("row", 0, 1), "ssm_beta": ("row", 0, 1),
           "ssm_dt_bias": ("flat", 0, 1), "ssm_a": ("flat", 0, 1),
           "ssm_conv1d": ("row", 4096, 128), "ssm_out": ("col", 0, 128)}


def v_keep(cfg, spec, n):
    """The index_select vector that moves one role's V axis to HF grouped order and leaves the rest of
    the axis alone (§04.9). `n` is that axis's length, so the same formula covers a matrix whose V block
    starts mid-axis (`attn_qkv`, `ssm_conv1d`: q and k rows precede it) and one that is all V
    (`attn_gate`, `ssm_out`, the flat control vectors).
    """
    axis, start, head = spec
    order = v_order_gguf_to_hf(cfg["h_v"], cfg["h_k"], cfg["expansion"], head)
    if axis == "flat":
        return order
    return torch.cat([torch.arange(start), order + start,
                      torch.arange(start + cfg["h_v"] * head, n)])


class Refused(Exception):
    """A claim this lane may not make. Carries a tag so the gate can name the refusal."""

    def __init__(self, tag, msg):
        super().__init__("[%s] %s" % (tag, msg))
        self.tag = tag


class Engine:
    """Policy-A weights: engine addresses, audited decode, one read path, every read counted."""

    ROLE_WIDTH = {"attn_qkv": "Q8_0", "attn_gate": "Q8_0", "ssm_out": "Q8_0", "ssm_conv1d": "F32",
                  "ssm_alpha": "F32", "ssm_beta": "F32", "ssm_dt_bias": "F32", "ssm_norm": "F32",
                  "ssm_a": "F32", "ffn_gate_inp": "F32", "ffn_gate_inp_shexp": "F32",
                  "ffn_gate_shexp": "Q8_0", "ffn_up_shexp": "Q8_0", "ffn_down_shexp": "Q8_0",
                  "ffn_gate_exps": "Q4_K", "ffn_up_exps": "Q4_K", "attn_q": "Q8_0", "attn_k": "Q8_0",
                  "attn_v": "Q8_0", "attn_output": "Q8_0", "attn_q_norm": "F32", "attn_k_norm": "F32"}

    def __init__(self, ledger, fault=None):
        self.led = ledger
        self.art = FD.Artifact(ledger=ledger, why="engine/m3_forward.py policy-A reads")
        self.cfg = load_geometry()
        self.cache = {}
        self.reads = 0
        self.bytes = 0
        self.types = {}
        self.probe_ops = []       # dot products whose width feeds a band PROPOSAL operand
        self.exact_ops = []       # stages with no quantized read: association floor only
        self.verify_lines = []    # per-tensor fast-vs-authoritative decode identity, one line each
        self.verify_groups = {}   # routed-expert slabs: grouped per (layer, role), endpoints named
        self.verify = True        # never off: a fast decoder that is not checked is a second layout
        # Host-wall cost accounting, printed as M3COST. This is not a device timing and no rate is
        # derived from it: it exists to say which term a 40-block pass spends its minutes in.
        self.t_read = self.t_read_expert = self.t_verify = 0.0
        self.fault = fault
        self.perturb = None
        self.blocked = []

    def name_for(self, role, layer=None):
        if layer is None:
            return {"norm_final": "output_norm.weight", "embd": "token_embd.weight",
                    "lm_head": "output.weight"}[role]
        if role.endswith("_bias"):
            return "blk.%d.%s.bias" % (layer, role[:-5])
        if role == "ssm_a":
            return "blk.%d.ssm_a" % layer        # the one stored name without a `.weight` suffix
        return "blk.%d.%s.weight" % (layer, role)

    def verify_tensor(self, name, rec, group=None):
        """Run the per-tensor decode-identity leg and refuse the read if the two decoders disagree."""
        t0 = time.perf_counter()
        v = self.art.verify(rec, name)
        self.t_verify += time.perf_counter() - t0
        if group is None:
            self.verify_lines.append(v)
        else:
            g = self.verify_groups.setdefault(group, dict(group=group, width=v["width"], mode=v["mode"],
                                                         tensors=0, blocks=0, elements=0, ndiff_bytes=0,
                                                         first=name, last=name))
            g["tensors"] += 1
            g["blocks"] += v["blocks"]
            g["elements"] += v["elements"]
            g["ndiff_bytes"] += v["ndiff_bytes"]
            g["last"] = name
        if v["ndiff_bytes"]:
            raise Refused("dequant-disagrees",
                          "%s: fast decode differs from authoritative gguf_reader.dequant in %d bytes "
                          "of its %s sample" % (name, v["ndiff_bytes"], v["mode"]))
        return v

    def read(self, role, layer=None):
        """One whole tensor in HF axis order, V axis un-permuted to HF grouped order, counted.

        Returns `(values, half_widths, handle)`. Values are cached until the caller clears the cache —
        the driver clears it per block, so peak RSS stays at one block's weights, not forty's.
        """
        key = ("t", role, layer)
        if key in self.cache:
            return self.cache[key]
        name = self.name_for(role, layer)
        t0 = time.perf_counter()
        vals, half, h = self.art.tensor(name, want_half=True)
        self.t_read += time.perf_counter() - t0
        self.reads += 1
        self.bytes += int(h["length"])
        self.types[h["type_name"]] = self.types.get(h["type_name"], 0) + 1
        self.verify_tensor(name, h)
        want = self.ROLE_WIDTH.get(role)
        if want is not None and h["type_name"] != want:
            raise Refused("role-width", "%s stores %s; the documented policy-A width (D-01/D-01a/D-08) "
                                        "is %s" % (name, h["type_name"], want))
        spec = V_ROLES.get(role)
        if (spec is not None and layer is not None
                and self.cfg["layer_types"][layer] == "linear_attention"):
            axis, _start, _head = spec
            ax = 0 if axis in ("row", "flat") else 1
            idx = v_keep(self.cfg, spec, int(vals.shape[ax]))
            vals = torch.from_numpy(vals).index_select(ax, idx).numpy()
            half = torch.from_numpy(half).index_select(ax, idx).numpy()
        out = (torch.from_numpy(np.ascontiguousarray(vals)), np.ascontiguousarray(half), h)
        self.cache[key] = out
        return out

    def expert(self, role, layer, idx, shape_hf):
        """One routed expert slab — the `per_token_gathered` read class, one range per expert.

        Deliberately NOT cached: each expert is touched once per layer and a 4 MB slab times ~170
        experts times 40 layers is how a CPU pass runs out of RAM.
        """
        t0 = time.perf_counter()
        vals, half, r = self.art.expert(self.name_for(role, layer), idx, shape_hf, want_half=True)
        self.t_read += time.perf_counter() - t0
        self.t_read_expert += time.perf_counter() - t0
        self.reads += 1
        self.bytes += int(r["length"])
        self.types[r["type_name"]] = self.types.get(r["type_name"], 0) + 1
        self.verify_tensor(self.name_for(role, layer) + "[expert %d]" % idx, r,
                           group="blk.%d.%s.weight" % (layer, role))
        return torch.from_numpy(vals), np.ascontiguousarray(half), r

    def rows(self, role, row_ids, k):
        """A row gather — the embedding's read class, and `lm_head`'s when streamed in row blocks."""
        name = self.name_for(role)
        t0 = time.perf_counter()
        got = self.art.row_gather(name, row_ids, k)
        self.t_read += time.perf_counter() - t0
        self.reads += 1
        self.verify_tensor(name, got[2])
        return got

    def note_type(self, tname, n=1):
        self.types[tname] = self.types.get(tname, 0) + n

    def close(self):
        self.art.close()


# ------------------------------------------------------------------ the arithmetic, each step cited
def rmsnorm_a(x, w):
    """§03.1 Form A, F32 core. The published file already carries `1 + w`, so there is NO `+1` here —
    the form is a property of the artifact being read (§03.1's artifact rule)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + CFG["eps_rms"])) * w


def chunked_delta_scan(q, k, v, beta, decay, T, cfg):
    """§04b.3–§04b.5: pairwise decay with the `-inf` mask BEFORE the exp, the unitriangular UT solve,
    and one sequential scan over chunks. Pads are zero so a pad column writes nothing (§04b.2)."""
    C = cfg["chunk"]
    pad = (-T) % C
    q, k, v = [F.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]
    beta, decay = F.pad(beta, (0, pad)), F.pad(decay, (0, pad))
    Tp = T + pad
    v_beta, k_beta = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)

    def chunked(t):
        return t.reshape(t.shape[0], t.shape[1], -1, C, t.shape[-1])

    q, k, k_beta, v_beta = chunked(q), chunked(k), chunked(k_beta), chunked(v_beta)
    dec = decay.reshape(decay.shape[0], decay.shape[1], -1, C)
    cum = dec.cumsum(dim=3)
    upper = torch.ones(C, C, dtype=torch.bool).triu(1)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    dec_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, dec_k_beta, upper=False, unitriangular=True)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    S = torch.zeros(v.shape[0], v.shape[1], cfg["d_k"], cfg["d_v"], dtype=torch.float32)
    out = torch.zeros_like(new_values)
    for i in range(Tp // C):                                    # §04b.5, the only sequential loop
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ S
        out[:, :, i] = q[:, :, i] @ S + intra[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    return out.reshape(v.shape[0], v.shape[1], Tp, cfg["d_v"])[:, :, :T], S


def gdn_block(xn, E, layer):
    """§04: projections, causal short conv, chunked delta scan, gated RMSNorm (Form B), out_proj."""
    cfg = E.cfg
    qkv = E.read("attn_qkv", layer)[0]
    z_w = E.read("attn_gate", layer)[0]
    wa, wb = E.read("ssm_alpha", layer)[0], E.read("ssm_beta", layer)[0]
    conv, dtb, a_eff = E.read("ssm_conv1d", layer)[0], E.read("ssm_dt_bias", layer)[0], \
        E.read("ssm_a", layer)[0]
    wnorm = E.read("ssm_norm", layer)[0]
    wout, half_out = E.read("ssm_out", layer)[:2]
    B, T, d = xn.shape
    mixed = F.linear(xn, qkv)                                          # [B,T,conv_dim] §04.2
    zv = F.linear(xn, z_w).view(B, T, cfg["h_v"], cfg["d_v"])
    b_raw, a_raw = F.linear(xn, wb), F.linear(xn, wa)
    ci = mixed.transpose(1, 2)                                         # §04.3 causal conv, silu after
    conv_out = F.silu(F.conv1d(F.pad(ci, (cfg["k_conv"] - 1, 0)),
                               conv.view(cfg["conv_dim"], 1, cfg["k_conv"]),
                               groups=cfg["conv_dim"])[:, :, :T])
    cq = conv_out.transpose(1, 2)
    q, k, v = torch.split(cq, [cfg["key_dim"], cfg["key_dim"], cfg["value_dim"]], dim=-1)
    q = q.reshape(B, T, cfg["h_k"], cfg["d_k"])
    k = k.reshape(B, T, cfg["h_k"], cfg["d_k"])
    v = v.reshape(B, T, cfg["h_v"], cfg["d_v"])
    beta = torch.sigmoid(b_raw)                                        # §04.5
    dec = a_eff.view(1, 1, cfg["h_v"]) * F.softplus(a_raw.float() + dtb.float())
    if cfg["expansion"] > 1:                                           # §04.4 grouped expansion
        q, k = q.repeat_interleave(cfg["expansion"], dim=2), k.repeat_interleave(cfg["expansion"], dim=2)
    q, k, v = [t.transpose(1, 2).to(torch.float32) for t in (q, k, v)]
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + cfg["l2norm_eps"])
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + cfg["l2norm_eps"])
    q = q / (cfg["d_k"] ** 0.5)
    out, S = chunked_delta_scan(q, k, v, beta.float(), dec.float(), T, cfg)
    f = out.float()                                                    # §04.7 Form B, variance over d_v
    gated = wnorm.view(1, 1, 1, cfg["d_v"]) * (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True)
                                                               + cfg["eps_rms"]))
    gated = (gated * F.silu(zv.float())).to(F32)
    core = gated.reshape(B, T, cfg["value_dim"])
    E.probe_ops.append(dict(stage="mixer_out", layer=layer, op="ssm_out", K=cfg["value_dim"],
                            width="Q8_0", x=core[-1, -1], half=half_out, w=wout))
    return F.linear(core, wout), S                                     # §04.8


def rope_tables(T, cfg):
    """§05.4: R = int(d_head * partial_rotary) = 64, 32 frequencies written twice, F32 tables,
    positions 0..T-1 (text-only makes the mrope sections inert, §05.4)."""
    R = int(cfg["head_dim"] * cfg["partial_rotary"])
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, R, 2, dtype=torch.float32) / R))
    fr = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([fr.cos(), fr.cos()], -1), torch.cat([fr.sin(), fr.sin()], -1)


def attn_block(xn, E, layer, cos, sin):
    """§05: interleaved q|gate head split, per-head QK norm before rope, causal attention, out gate."""
    cfg = E.cfg
    wq, wk, wv = (E.read("attn_q", layer)[0], E.read("attn_k", layer)[0], E.read("attn_v", layer)[0])
    wo, half_o = E.read("attn_output", layer)[:2]
    nq, nkv, dh = cfg["n_q"], cfg["n_kv"], cfg["head_dim"]
    B, T, _ = xn.shape
    qg = F.linear(xn, wq).view(B, T, nq, 2 * dh)
    q_raw, gate = qg.chunk(2, dim=-1)                                  # §05.1 interleaved, not halves
    q = rmsnorm_a(q_raw, E.read("attn_q_norm", layer)[0]).transpose(1, 2)  # §05.2 norm BEFORE rope
    k = rmsnorm_a(F.linear(xn, wk).view(B, T, nkv, dh),
                  E.read("attn_k_norm", layer)[0]).transpose(1, 2)
    v = F.linear(xn, wv).view(B, T, nkv, dh).transpose(1, 2)
    R = cos.shape[-1]

    def rot(t):
        tr, tp = t[..., :R], t[..., R:]
        c, s = cos.view(1, 1, T, R), sin.view(1, 1, T, R)
        half = torch.cat([-tr[..., R // 2:], tr[..., :R // 2]], -1)
        return torch.cat([tr * c + half * s, tp], -1)

    q, k = rot(q), rot(k)
    kk = k.repeat_interleave(nq // nkv, dim=1)                         # §05.3 GQA expansion
    vv = v.repeat_interleave(nq // nkv, dim=1)
    scores = (q @ kk.transpose(-1, -2)) * (dh ** -0.5)
    scores = scores + torch.full((T, T), float("-inf")).triu(1)
    p = torch.softmax(scores.float(), dim=-1)                          # §05.3 F32 softmax
    attn = (p @ vv).transpose(1, 2).reshape(B, T, nq * dh)
    gated = attn * torch.sigmoid(gate.reshape(B, T, nq * dh))           # §05.5 gate, then o_proj
    E.probe_ops.append(dict(stage="mixer_out", layer=layer, op="attn_output", K=nq * dh,
                            width="Q8_0", x=gated[-1, -1], half=half_o, w=wo))
    return F.linear(gated, wo)


def moe_block(g, E, layer, probe_row=None):
    """§11: shared expert (independent of the router), router R1–R5, K=8 gathered experts summed in
    ascending expert-id order, plain add. Returns the block output and the routing inventory."""
    cfg = E.cfg
    B, T, d = g.shape
    x = g.reshape(B * T, d)
    wg = E.read("ffn_gate_shexp", layer)[0]
    wu = E.read("ffn_up_shexp", layer)[0]
    wd, half_wd = E.read("ffn_down_shexp", layer)[:2]
    up = F.linear(x, wg)
    shared = F.linear(F.silu(up) * F.linear(x, wu), wd)
    shgate = torch.sigmoid(F.linear(x, E.read("ffn_gate_inp_shexp", layer)[0])).view(x.shape[0], 1)
    shared = shgate * shared                                            # §11.6 one scalar per token
    logits = F.linear(x, E.read("ffn_gate_inp", layer)[0]).to(cfg["act_dtype"])              # R1
    probs = torch.softmax(logits.float(), dim=-1)                                             # R2
    tw, ids = torch.topk(probs, cfg["top_k"], dim=-1)                                          # R3
    wcast = (tw / tw.sum(-1, keepdim=True)).to(logits.dtype)                                   # R4, R5
    routed = torch.zeros_like(x)
    used = {}
    for tok in range(x.shape[0]):
        for slot in range(cfg["top_k"]):
            used.setdefault(int(ids[tok, slot]), []).append((tok, slot))
    for e in sorted(used):                                                    # §11.5 ascending id
        gate_w = E.expert("ffn_gate_exps", layer, e, (cfg["i_e"], d))[0]
        up_w = E.expert("ffn_up_exps", layer, e, (cfg["i_e"], d))[0]
        if E.fault and layer == E.fault["layer"]:
            want = E.fault["expert"]
            E.fault["resolved"] = want if want in used else max(used, key=lambda e: len(used[e]))
        down_w, half_d, dr = E.expert("ffn_down_exps", layer, e, (cfg["d"], cfg["i_e"]))
        rows = [t for t, _ in used[e]]
        if E.fault and layer == E.fault["layer"] and e == E.fault["resolved"]:
            E.perturb = apply_down_fault(E, layer, e, down_w, rows)
        s = F.silu(F.linear(x[rows], gate_w)) * F.linear(x[rows], up_w)
        out_e = F.linear(s, down_w)
        wts = torch.stack([wcast[t, slot].to(F32) for t, slot in used[e]])
        routed[rows] += out_e * wts[:, None]
        if probe_row is not None and probe_row in rows:
            i = rows.index(probe_row)
            E.probe_ops.append(dict(stage="moe_out", layer=layer, op="ffn_down_exps", K=cfg["i_e"],
                                    width=dr["type_name"],
                                    x=s[i], half=half_d, w=down_w, weight=float(wts[i])))
    if probe_row is not None:
        E.probe_ops.append(dict(stage="moe_out", layer=layer, op="ffn_down_shexp", K=cfg["i_s"],
                                width="Q8_0", x=F.silu(F.linear(x[probe_row:probe_row + 1], wg))[0],
                                half=half_wd, w=wd, weight=1.0))
    return routed.reshape(B, T, d) + shared.reshape(B, T, d), dict(used=len(used),
                                                                 ids=ids.reshape(B, T, -1))


def apply_down_fault(E, layer, expert, down_w, rows):
    """The perturbation: zero `n` elements of one expert's dequantized `down_proj` row, mirroring the
    HF fixture's own `p1` (512 elements of layer 20, expert 7, row 3) so the two witnesses compare."""
    n = E.fault["elements"]
    before = hashlib.sha256(down_w.detach().numpy().tobytes()).hexdigest()
    row = E.fault["row"]
    down_w[row, :n] = 0.0
    after = hashlib.sha256(down_w.detach().numpy().tobytes()).hexdigest()
    return dict(layer=layer, expert=expert, requested=E.fault["requested"], row=row,
                elements_zeroed=n, sha_before=before, sha_after=after, tokens_touching=len(rows),
                same_sha=before == after)


# ------------------------------------------------------------------ band operands (never a band)
def gamma(n, u=2.0 ** -24):
    """Higham's γ_n = n·u/(1−n·u) in double, u = 2⁻²⁴ for f32 (the form D-20 settled on)."""
    return n * u / (1.0 - n * u)


def _spread(stage, layer, op, K, width, sum_abs_x, quant_mean, round_mean, y32, y64):
    d = (y32.double() if torch.is_tensor(y32) else torch.as_tensor(y32).double()) - (
        y64 if torch.is_tensor(y64) else torch.as_tensor(y64))
    return dict(kind="exact", stage=stage, layer=layer, op=op, K=K, width=width,
                sum_abs_x=float(sum_abs_x), quant_term_mean=float(quant_mean),
                round_term_mean=float(round_mean),
                assoc_max_abs_delta=float(d.abs().max()),
                assoc_relL2=float(d.pow(2).sum().sqrt()
                                  / (y64 if torch.is_tensor(y64) else torch.as_tensor(y64)
                                      ).double().pow(2).sum().sqrt()),
                out_rms_f64=float((y64 if torch.is_tensor(y64) else torch.as_tensor(y64)
                                   ).double().pow(2).mean().sqrt()), weight=1.0)


def norm_spread(stage, layer, xr, w, cfg):
    """A Form-A norm reads no quantized byte (D-01a/D-08 keep norms exact), so its only floor is
    association — measured by recomputing the same row in float64."""
    y32 = rmsnorm_a(xr, w)
    y64 = rmsnorm_a(xr.double(), w.double())
    K = cfg["d"]
    return _spread(stage, layer, "rmsnorm_a-form-A", K, "F32-exact",
                   float(xr.double().abs().sum()), 0.0,
                   gamma(K - 1) * float((xr.double().abs() * w.double()).mean()), y32, y64)


def add_spread(stage, layer, a, b, cfg):
    """§03.2's two residual adds: activation-dtype adds, no quantized read at this step."""
    K = cfg["d"]
    return _spread(stage, layer, "residual-add", K, "f32-stream",
                   float(a.double().abs().sum() + b.double().abs().sum()), 0.0,
                   gamma(1) * float((a.double().abs() + b.double().abs()).mean()), a + b, a.double()
                   + b.double())


def band_operand(entry):
    """Turn one (activation row, weight, per-element grid half-width) triple into printed operands.

    `quant_term_mean` is Σ_j |x_j|·half_ij averaged over output rows i — the worst-case weight-width
    error the stored grid can contribute to that dot, with no cancellation credit. `round_term_mean` is
    γ_{2K−1}·Σ_j|x_j w_ij|, the f32 rounding term at D-20's wide index. `assoc_*` re-runs the same dot
    in float64: that spread is what no implementation can remove by being careful. All operands, no
    threshold — the gate turns them into a PROPOSAL and marks it `proposal=awaiting-main-agent`.
    """
    x = entry["x"].double().numpy().astype(np.float64)
    w = entry["w"].double().numpy().astype(np.float64)
    half = np.asarray(entry["half"], dtype=np.float64)
    K = int(entry["K"])
    quant = (np.abs(x)[None, :] * half).sum(axis=1)
    absw = (np.abs(x)[None, :] * np.abs(w)).sum(axis=1)
    y32 = entry["x"] @ entry["w"].t()
    y64 = np.asarray(x @ w.T, dtype=np.float64)
    delta = float(np.abs(y32.double().numpy().astype(np.float64) - y64).max())
    rel = float(np.sqrt(((y32.double().numpy().astype(np.float64) - y64) ** 2).sum())
                / np.sqrt((y64 ** 2).sum()))
    return dict(stage=entry["stage"], layer=entry["layer"], op=entry["op"], K=K, width=entry["width"],
                kind="dot",
                sum_abs_x=float(np.abs(x).sum()),
                quant_term_mean=float(quant.mean()), quant_term_max=float(quant.max()),
                round_term_mean=float(gamma(2 * K - 1) * absw.mean()),
                assoc_max_abs_delta=delta, assoc_relL2=rel,
                out_rms_f64=float(np.sqrt((y64 ** 2).mean())),
                weight=float(entry.get("weight", 1.0)))


def exact_operand(stage, layer, op, x_f32, w_f32, K):
    """Kept for callers that want the spread of a norm in one call; `norm_spread` is the same thing."""
    return norm_spread(stage, layer, x_f32, w_f32, CFG)


# ------------------------------------------------------------------ records


def record_emit(recs, outdir, name, stage, block, tensor):
    """Write one stage row as raw little-endian f32 and return its manifest record."""
    arr = np.ascontiguousarray(tensor.detach().to(torch.float32).numpy())
    raw = arr.tobytes()
    path = os.path.join(outdir, name + ".bin")
    with open(path, "wb") as fh:
        fh.write(raw)
    recs.append(dict(name=name, stage=stage, block=block, file=path, basename=name + ".bin",
                     sha256=hashlib.sha256(raw).hexdigest(), dtype="f32", shape=list(arr.shape),
                     elements=int(arr.size), bytes=len(raw)))


def aggregate_digest(recs):
    """sha256 over the sorted `name:sha256` lines — the same construction the HF fixture uses, over
    basenames only so two runs in two directories are comparable."""
    blob = "\n".join("%s:%s" % (r["name"], r["sha256"]) for r in sorted(recs, key=lambda r: r["name"]))
    return hashlib.sha256(blob.encode()).hexdigest(), len(recs)


def host_report():
    rss = mem = -1
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    rss = int(line.split()[1])
                elif line.startswith("VmRSS"):
                    mem = int(line.split()[1])
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return rss, mem, int(line.split()[1]) * 1024
    except OSError:
        pass
    return rss, mem, -1


# ------------------------------------------------------------------ the pass itself
def lm_head_row(E, xrow, chunk_rows=512):
    """§03.4: one row of logits, `lm_head` streamed in row blocks so the untied 1.0 GiB table never
    has to be resident at once.

    The band operands are NOT computed over all 248,320 vocabulary rows: an f64 `Σ_j |x_j|·half_ij` for
    the whole table is 2 GB of temporaries per chunk and dominated a first draft of this pass. The mean
    comes from a row sample derived by arithmetic from `V` (printed, not typed), and the rows that decide
    an argmax get their operands exactly — see `lm_head_row_operands`.
    """
    cfg = CFG
    h = E.art.handle("output.weight")
    E.verify_tensor("output.weight", h)
    E.reads += 1
    V, d = cfg["V"], cfg["d"]
    per_row = int(h["length"]) * d // int(h["n_elts"])
    out = np.empty(V, dtype=np.float32)
    xv = xrow.numpy().astype(np.float64)
    xv32 = xv.astype(np.float32)
    sample = sorted(set(range(0, V, max(1, V // 64))) | {0, V // 4, V // 2, 3 * V // 4, V - 1})
    seen = set()
    quant_sum = round_sum = 0.0
    t0 = time.perf_counter()
    for lo in range(0, V, chunk_rows):
        n = min(chunk_rows, V - lo)
        buf = E.art.raw(int(h["file_off"]) + lo * per_row, n * per_row, "output.weight")
        vals, half = FD.dequant_np(buf, h["type_name"], n * d, want_half=True)
        vals = vals.reshape(n, d)
        out[lo:lo + n] = vals @ xv32
        picked = [i for i in sample if lo <= i < lo + n]
        if picked:
            seen.update(picked)
            loc = np.asarray(picked) - lo
            hv = np.asarray(half, dtype=np.float64).reshape(n, d)[loc]
            quant_sum += float((np.abs(xv)[None, :] * hv).sum())
            round_sum += float((np.abs(xv)[None, :]
                                * np.abs(vals.astype(np.float64)[loc])).sum())
        E.types[h["type_name"]] = E.types.get(h["type_name"], 0) + 1
    E.t_read += time.perf_counter() - t0
    E.bytes += int(h["length"])
    ns = len(seen)
    operands = dict(stage="logits", layer=CFG["L"] - 1, op="lm_head", K=d, width=h["type_name"],
                    kind="dot", rows_sampled=ns, rows_total=V,
                    sum_abs_x=float(np.abs(xv).sum()),
                    quant_term_mean=quant_sum / ns, quant_term_max=None,
                    round_term_mean=gamma(2 * d - 1) * round_sum / ns,
                    assoc_max_abs_delta=None, assoc_relL2=None, out_rms_f64=None, weight=1.0,
                    rows_streamed=-(-V // chunk_rows))
    return out, operands


def lm_head_row_operands(E, xrow, rows):
    """Exact width/rounding operands for the vocabulary rows that can decide the argmax.

    A band on a logits row is only interesting at the contenders: `argmax` moves if the top pair moves,
    so these rows get the operand computed exactly rather than sampled.
    """
    cfg = CFG
    h = E.art.handle("output.weight")
    V, d = cfg["V"], cfg["d"]
    per_row = int(h["length"]) * d // int(h["n_elts"])
    xv = xrow.numpy().astype(np.float64)
    got = []
    for r in rows:
        buf = E.art.raw(int(h["file_off"]) + int(r) * per_row, per_row, "output.weight")
        vals, half = FD.dequant_np(buf, h["type_name"], d, want_half=True)
        got.append(dict(stage="logits-contender", layer=cfg["L"] - 1, op="lm_head row %d" % r, K=d,
                        width=h["type_name"], kind="dot", sum_abs_x=float(np.abs(xv).sum()),
                        quant_term_mean=float((np.abs(xv) * np.asarray(half, np.float64)).sum()),
                        quant_term_max=None,
                        round_term_mean=gamma(2 * d - 1) * float((np.abs(xv)
                                                                  * np.abs(vals.astype(np.float64))).sum()),
                        assoc_max_abs_delta=None, assoc_relL2=None, out_rms_f64=None, weight=1.0,
                        row=int(r)))
    return got


def run_forward(args):
    led = Ledger()
    cfg = load_geometry()
    torch.set_num_threads(args.threads)
    cuda = torch.cuda.is_available()
    led.install()
    outdir = os.path.join(args.out, args.tag, "dumps")
    os.makedirs(outdir, exist_ok=True)
    fault = None
    if args.perturb_layer is not None:
        fault = dict(layer=args.perturb_layer, expert=args.perturb_expert, requested=args.perturb_expert,
                     row=args.perturb_row, elements=args.perturb_elements, resolved=None)
    E = Engine(led, fault=fault)
    if args.peek:
        with open(args.peek, "rb") as fh:      # CONTROL ONLY: the ledger must catch this read
            fh.read(64)
    ids, sha, man_sha = token_ids()
    T = len(ids)
    recs, blocks, blocked = [], [], []
    try:
        x_emb, half_emb = E.rows("embd", ids, cfg["d"])[:2]
        x = torch.from_numpy(x_emb).view(1, T, cfg["d"])
        E.exact_ops.append(dict(kind="exact", stage="x", layer=0, op="embedding-row-gather", K=0,
                                width="Q8_0", sum_abs_x=float(np.abs(x_emb.astype(np.float64)).sum()),
                                quant_term_mean=float(np.asarray(half_emb, dtype=np.float64).mean()),
                                round_term_mean=0.0, assoc_max_abs_delta=0.0, assoc_relL2=0.0,
                                out_rms_f64=float(np.sqrt((x_emb.astype(np.float64) ** 2).mean())),
                                weight=1.0))
        cos, sin = rope_tables(T, cfg)
        limit = args.limit_blocks if args.limit_blocks else cfg["L"]
        for layer in range(limit):
            try:
                E.cache.clear()
                t_block0 = time.perf_counter()
                r0, re0, v0 = E.t_read, E.t_read_expert, E.t_verify
                before_reads = E.reads
                stage = "layer%d" % layer
                probe = T - 1 if layer in PROBE_LAYERS else None
                x_row = x[0, probe] if probe is not None else None      # block input, before any add
                record_emit(recs, outdir, "eng40_layer%d_x" % layer, "x", stage, x)
                h_in = rmsnorm_a(x, E.read("attn_norm", layer)[0])
                record_emit(recs, outdir, "eng40_layer%d_h_in" % layer, "h_in", stage, h_in)
                if cfg["layer_types"][layer] == "linear_attention":
                    mixer, _S = gdn_block(h_in, E, layer)
                else:
                    mixer = attn_block(h_in, E, layer, cos, sin)
                record_emit(recs, outdir, "eng40_layer%d_mixer_out" % layer, "mixer_out", stage, mixer)
                x1 = x + mixer                                              # §03.2 step 3
                record_emit(recs, outdir, "eng40_layer%d_x1" % layer, "x1", stage, x1)
                h_mlp = rmsnorm_a(x1, E.read("post_attention_norm", layer)[0])
                record_emit(recs, outdir, "eng40_layer%d_h_mlp" % layer, "h_mlp", stage, h_mlp)
                moe_out, info = moe_block(h_mlp, E, layer, probe_row=probe)
                record_emit(recs, outdir, "eng40_layer%d_moe_out" % layer, "moe_out", stage, moe_out)
                x = x1 + moe_out                                          # §03.2 step 6
                record_emit(recs, outdir, "eng40_layer%d_out" % layer, "out", stage, x)
                if probe is not None:
                    E.exact_ops.append(norm_spread("h_in", layer, x_row,
                                                  E.read("attn_norm", layer)[0], cfg))
                    E.exact_ops.append(add_spread("x1", layer, x[0, probe], mixer[0, probe], cfg))
                    E.exact_ops.append(norm_spread("h_mlp", layer, x1[0, probe],
                                                  E.read("post_attention_norm", layer)[0], cfg))
                    E.exact_ops.append(add_spread("out", layer, x1[0, probe], moe_out[0, probe], cfg))
                blocks.append(dict(layer=layer, layer_type=cfg["layer_types"][layer],
                                   reads=E.reads - before_reads, experts_used=info["used"],
                                   total_s=time.perf_counter() - t_block0,
                                   dequant_s=E.t_read - r0, dequant_expert_s=E.t_read_expert - re0,
                                   verify_s=E.t_verify - v0))
                b = blocks[-1]
                print("M3COST block=%d total_s=%.2f dequant_s=%.2f dequant_expert_s=%.2f "
                      "verify_s=%.2f matmul_other_s=%.2f tensors_read=%d basis=host-wall-not-device"
                      % (layer, b["total_s"], b["dequant_s"], b["dequant_expert_s"], b["verify_s"],
                         b["total_s"] - b["dequant_s"] - b["verify_s"], b["reads"]))
            except Refused as ex:
                blocked.append(dict(role="layer%d" % layer, tag=ex.tag, detail=str(ex)))
                break
        if blocked or limit < cfg["L"]:
            reason = blocked[0]["tag"] if blocked else "limit_blocks=%d" % limit
            print("M3-BLOCKED role=%s reason=%s blocks_completed=%d of=%d missing=%s"
                  % (blocked[0]["role"] if blocked else "layer%d" % limit, reason, len(blocks),
                     cfg["L"], ",".join("layer%d" % i for i in range(len(blocks), cfg["L"]))))
        complete = not blocked and limit == cfg["L"]
        if not complete:
            # An honest prefix emits no `normed`, no logits and no argmax: a number computed from a
            # truncated residual stream is not a partial result, it is a different quantity.
            head_ops, contender_ops, argmax, margin, top5 = None, [], -1, float("nan"), "none"
        else:
            t_final0 = time.perf_counter()
            normed = rmsnorm_a(x, E.read("norm_final")[0])                       # §03.4 Form A
            record_emit(recs, outdir, "eng40_normed", "normed", "__model__", normed)
            E.exact_ops.append(norm_spread("normed", CFG["L"] - 1, normed[0, T - 1],
                                           E.read("norm_final")[0], cfg))
            logits, head_ops = lm_head_row(E, normed[0, T - 1])
            record_emit(recs, outdir, "eng40_logits-last-token", "logits", "__model__",
                        torch.from_numpy(logits))
            order = np.argsort(-logits)
            argmax = int(order[0])
            margin = float(logits[order[0]] - logits[order[1]])
            top5 = " ".join("%d:%.6f" % (int(i), float(logits[int(i)])) for i in order[:5])
            contender_ops = lm_head_row_operands(E, normed[0, T - 1], [int(i) for i in order[:5]])
            t_final = time.perf_counter() - t_final0
    finally:
        art_path = E.art.path
        E.close()
        led.remove()
    agg, n = aggregate_digest(recs)
    if complete:
        print("M3COST phase=final-norm-and-lm_head total_s=%.2f rows=%d basis=host-wall-not-device"
              % (t_final, CFG["V"]))
    manifest = dict(tag=args.tag, stage_records=recs, aggregate_digest=agg, records=n,
                    argmax=dict(id=argmax, top5=top5, rank1_minus_rank2=margin, n_vocab=CFG["V"],
                                definition="greedy argmax over all vocabulary rows at the last "
                                            "position; no penalties, no mask, no temperature"),
                    stimulus=dict(token_ids=ids, token_ids_sha256=sha,
                                  manifest_token_ids_sha256=man_sha, n_tokens=T,
                                  source=os.path.relpath(CHAIN_PATH, REPO) + "::input.token_ids",
                                  positions="0..%d" % (T - 1)),
                    blocks=blocks, blocked=blocked, reads=dict(reads=E.reads, bytes=E.bytes,
                                                               types=E.types),
                    band_operands=[band_operand(e) for e in E.probe_ops] + E.exact_ops
                    + ([head_ops] if head_ops else []) + contender_ops,
                    complete=complete,
                    ledger=led.names(), cuda_available=bool(cuda),
                    dequant=dict(tensors=E.verify_lines, groups=list(E.verify_groups.values()),
                                 reads=E.reads,
                                 verified=len(E.verify_lines) + sum(g["tensors"] for g in
                                                                    E.verify_groups.values())),
                    perturb=E.perturb, artifact=os.path.relpath(art_path, REPO),
                    compute_basis="f32-from-dequant", act_dtype=str(CFG["act_dtype"]),
                    dtype_casts_applied="router-logits-R1,topk-weights-R5",
                    bf16_stream_rounding="not-applied")
    with open(os.path.join(args.out, args.tag, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


def main(argv):
    ap = argparse.ArgumentParser(prog="engine/m3_forward.py",
                                 description="engine's own 40-block forward; CPU-only, no timings")
    ap.add_argument("--out", default="/tmp/m3-engine-d30")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit-blocks", type=int, default=0)
    ap.add_argument("--perturb-layer", type=int, default=None)
    ap.add_argument("--perturb-expert", type=int, default=7)
    ap.add_argument("--perturb-row", type=int, default=3)
    ap.add_argument("--perturb-elements", type=int, default=512)
    ap.add_argument("--peek", default=None, help="control only: open one path so the ledger must see it")
    args = ap.parse_args(argv)
    if torch.cuda.is_available():
        print("M3FWD-REFUSE tag=device-visible reason=a device is present; this lane is CPU-only "
              "by contract")
        return 2
    led_note = "installed-in-run_forward"
    man = run_forward(args)
    print("M3FWD-HDR tag=%s gguf=%s n_tokens=%d layers=%d layers_completed=%d stages=%d "
          "records_expected=%d records_emitted=%d compute_basis=%s act_dtype=%s "
          "dtype_casts_applied=%s bf16_stream_rounding=%s device=off hip_calls=0 threads=%d"
          % (man["tag"], os.path.basename(man["artifact"]), man["stimulus"]["n_tokens"],
             CFG["L"], len(man["blocks"]), len(STAGES), CFG["L"] * len(STAGES) + 2,
             man["records"], man["compute_basis"], man["act_dtype"], man["dtype_casts_applied"],
             man["bf16_stream_rounding"], args.threads))
    print("M3FWD-CHECK cuda_available=%s ledger_install=%s" % (str(man["cuda_available"]).lower(),
                                                               led_note))
    s = man["stimulus"]
    print("TOKENS n=%d token_ids_sha256=%s manifest_token_ids_sha256=%s equal=%s source=%s "
          "positions=%s" % (s["n_tokens"], s["token_ids_sha256"], s["manifest_token_ids_sha256"],
                            s["token_ids_sha256"] == s["manifest_token_ids_sha256"], s["source"],
                            s["positions"]))
    for b in man["blocks"]:
        print("STAGE-BLOCK layer=%d type=%s reads=%d experts_used=%d stages=7" %
              (b["layer"], b["layer_type"], b["reads"], b["experts_used"]))
    print("READS-TOTAL blocks_printed=%d blocks_expected=%d reads=%d bytes=%d types=%s missing=%s"
          % (len(man["blocks"]), CFG["L"], man["reads"]["reads"], man["reads"]["bytes"],
             ",".join("%s:%d" % kv for kv in sorted(man["reads"]["types"].items())),
             len(man["blocks"]) == CFG["L"] and "none" or "see M3-BLOCKED"))
    for r in man["stage_records"]:
        print("RECORD name=%s stage=%s block=%s sha256=%s shape=%s dtype=f32 elements=%d bytes=%d"
              % (r["name"], r["stage"], r["block"], r["sha256"], ",".join(map(str, r["shape"])),
                 r["elements"], r["bytes"]))
    print("RECORDS printed=%d expected=%d aggregate_digest=%s" %
          (man["records"], CFG["L"] * len(STAGES) + 2, man["aggregate_digest"]))
    a = man["argmax"]
    print("ENGINE-ARGMAX complete=%s id=%d n_vocab=%d rank1_minus_rank2=%.6f top5=%s"
          % (man["complete"], a["id"], a["n_vocab"], a["rank1_minus_rank2"], a["top5"]))
    if man["perturb"]:
        p = man["perturb"]
        print("PERTURB tensor=%s layer=%d expert=%d expert_requested=%d row=%d elements_zeroed=%d "
              "tokens_touching=%d sha_before=%s sha_after=%s tensor_changed=%s"
              % (p["tensor"], p["layer"], p["expert"], p["requested"], p["row"], p["elements_zeroed"],
                 p["tokens_touching"], p["sha_before"], p["sha_after"],
                 str(p["sha_before"] != p["sha_after"]).lower()))
    for op in man["band_operands"]:
        print("BANDOP " + " ".join("%s=%s" % (k, ("%.6e" % v if isinstance(v, float) else v))
                                    for k, v in sorted(op.items()) if v is not None))
    for v in man["dequant"]["tensors"]:
        print("M3DEQUANT-CROSS tensor=%s fast_vs_loader ndiff_bytes=%d width=%s sample=%s blocks=%d "
              "elements=%d" % (v["name"], v["ndiff_bytes"], v["width"], v["mode"], v["blocks"],
                               v["elements"]))
    for g in man["dequant"]["groups"]:
        print("M3DEQUANT-CROSS tensor=%s fast_vs_loader ndiff_bytes=%d width=%s sample=%s tensors=%d "
              "named_first=%s named_last=%s blocks=%d elements=%d"
              % (g["group"], g["ndiff_bytes"], g["width"], g["mode"], g["tensors"], g["first"],
                 g["last"], g["blocks"], g["elements"]))
    dq = man["dequant"]
    print("M3DEQUANT-CROSS-TRAILER tensors_verified=%d reads=%d reads_verified_equal=%s widths=%s "
          "elements_compared=%d ndiff_bytes_total=%d authority=tools/gguf_reader.py::dequant "
          "fast_path=%s refusal=on-any-nonzero-ndiff"
          % (dq["verified"], dq["reads"], dq["verified"] == dq["reads"],
             ",".join(sorted({v["width"] for v in dq["tensors"]})),
             sum(v["elements"] for v in dq["tensors"]) + sum(g["elements"] for g in dq["groups"]),
             sum(v["ndiff_bytes"] for v in dq["tensors"])
             + sum(g["ndiff_bytes"] for g in dq["groups"]),
             "tools/gguf_fast_dequant.py::dequant_np"))
    print("M3COST-TOTAL blocks=%d total_s=%.2f dequant_s=%.2f dequant_expert_s=%.2f verify_s=%.2f "
          "other_s=%.2f basis=host-wall-not-device no-rate-derived"
          % (len(man["blocks"]), sum(b["total_s"] for b in man["blocks"]),
             sum(b["dequant_s"] for b in man["blocks"]),
             sum(b["dequant_expert_s"] for b in man["blocks"]),
             sum(b["verify_s"] for b in man["blocks"]),
             sum(b["total_s"] - b["dequant_s"] - b["verify_s"] for b in man["blocks"])))
    rss, vm, avail = host_report()
    print("HOST peak_rss_kb=%d vm_rss_kb=%d mem_available_bytes=%d (host total 123 GiB; no device "
          "memory touched)" % (rss, vm, avail))
    print("M3FWD-END tag=%s records=%d aggregate_digest=%s argmax_id=%d no timing taken"
          % (man["tag"], man["records"], man["aggregate_digest"], man["argmax"]["id"]))
    return 0 if len(man["stage_records"]) == CFG["L"] * len(STAGES) + 2 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


