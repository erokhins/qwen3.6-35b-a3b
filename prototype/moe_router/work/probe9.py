import json, os, sys
import numpy as np, torch
REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
sys.path.insert(0, REPO + "/prototype/moe_router")
import oracle_arm as oa
sys.path.insert(0, REPO + "/speccheck")
import moe_ref
g = moe_ref.load_geometry(); E,H,I = g.num_experts, g.hidden_size, g.moe_intermediate_size
man, D = oa.load_manifest()
x, xb = oa.read_arr(D["layer0.input"]); raw, _ = oa.read_arr(D["layer0.expert_raw"])
ids = oa.read_arr(D["layer0.topk_ids"])[0]
idx = moe_ref.load_index()
blk = moe_ref.Block.parse("layer0")
gu_n = blk.prefix + moe_ref.SUFFIXES["gate_up"]; dn_n = blk.prefix + moe_ref.SUFFIXES["down"]
gu_s = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(gu_n, idx))
dn_s = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(dn_n, idx))
guh, guhn = moe_ref.read_header(gu_s); dnh, dnhn = moe_ref.read_header(dn_s)
print("gu hdr", guh[gu_n], "base", guhn, "dn hdr", dnh[dn_n], "base", dnhn)
t, ks = 0, 0
e = int(ids[t, ks])
print("expert", e, "ids row", ids[t])
gu_off = guhn + guh[gu_n]["data_offsets"][0]
dn_off = dnhn + dnh[dn_n]["data_offsets"][0]
print("byte offsets", gu_off, dn_off, "aligned", gu_off % 2, dn_off % 2)
gu = np.memmap(gu_s, dtype=np.uint16, mode="r", offset=gu_off, shape=(E, 2*I, H))
dn = np.memmap(dn_s, dtype=np.uint16, mode="r", offset=dn_off, shape=(E, H, I))
want = raw[t, ks].astype(np.float64)
print("want absmax", np.abs(want).max(), "first", want[:4])
xt = x[t].astype(np.float32)
def go(gate_w, up_w, dn_m, label):
    h = gate_w @ xt; u = up_w @ xt
    act = (h / (1.0 + np.exp(-h))) * u
    y = (dn_m @ act).astype(np.float64)
    c = float(y @ want / (np.linalg.norm(y)*np.linalg.norm(want)))
    print("%-18s absmax(y)=%9.4f 1-cos=%9.5f maxabs diff=%.4e" % (label, np.abs(y).max(), max(0,1-c), np.abs(y-want).max()))
gs = (np.array(gu[e], dtype=np.uint32) << 16).view(np.float32)
ds = (np.array(dn[e], dtype=np.uint32) << 16).view(np.float32)
go(gs[:I].reshape(I,H), gs[I:].reshape(I,H), ds.reshape(H,I), "chunk2")
go(gs[I:].reshape(I,H), gs[:I].reshape(I,H), ds.reshape(H,I), "swap")
go(gs[:I].reshape(I,H), gs[I:].reshape(I,H), ds.reshape(H,I).T, "dn transposed")
# does the slice even look like weights?
print("gu[e] absmax", np.abs(gs).max(), "dn[e] absmax", np.abs(ds).max())
# second token as a control
t2 = 1; e2 = int(ids[t2, 0])
gs2 = (np.array(gu[e2], dtype=np.uint32) << 16).view(np.float32)
ds2 = (np.array(dn[e2], dtype=np.uint32) << 16).view(np.float32)
want2 = raw[t2, 0].astype(np.float64)
xt2 = x[t2].astype(np.float32)
h = gs2[:I].reshape(I,H) @ xt2; u = gs2[I:].reshape(I,H) @ xt2
act = (h/(1+np.exp(-h)))*u
y2 = (ds2.reshape(H,I) @ act).astype(np.float64)
print("t2 cos", float(y2@want2/(np.linalg.norm(y2)*np.linalg.norm(want2))), "absmax", np.abs(y2).max(), np.abs(want2).max())
