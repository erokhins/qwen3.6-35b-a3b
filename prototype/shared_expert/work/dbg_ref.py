import json, os, sys
import numpy as np, torch
REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
D = os.path.join(REPO, "data/dumps/2026-10-26")
torch.set_num_threads(8)

def bf16(path):
    r = np.fromfile(path, dtype=np.uint16)
    return r.astype(np.uint32).view(np.float32).copy(), r

x, xb = bf16(os.path.join(D, "layer0.input.bin"))
sg, sgb = bf16(os.path.join(D, "layer0.mlp.shared_expert_gate.weight.bin"))
gp, gpb = bf16(os.path.join(D, "layer0.shared_gate_prob.bin"))
print("x", x.shape, "sg", sg.shape, "prob", gp.shape)
tx = torch.from_numpy(x.reshape(32, 2048).astype(np.float32)).to(torch.bfloat16)
tsg = torch.from_numpy(sg.reshape(1, 2048).astype(np.float32)).to(torch.bfloat16)
t = torch.nn.functional.linear(tx, tsg)
prob = torch.sigmoid(t)
mine = prob.to(torch.float32).numpy().reshape(-1)
want = gp.reshape(-1)
print("mine[:6]", mine[:6])
print("want[:6]", want[:6])
mb = (mine.view(np.uint32) >> 16) if hasattr(mine.view(np.uint32), "__rshift__") else None
mb = (mine.view(np.uint32) >> np.uint32(16))
wb = (want.view(np.uint32) >> np.uint32(16))
print("bit-exact", int((mb == wb).sum()), "of", wb.size)
# the pinned prob bit image straight off the file, for comparison
print("pinned bits", [hex(int(v)) for v in gpb[:6]])
print("mine  bits", [hex(int(v)) for v in (prob.to(torch.uint16) if hasattr(torch,'uint16') else torch.zeros(0))][:6] if False else [hex(int(v)) for v in (prob.numpy().view(np.uint32)>>16)[:6]])
