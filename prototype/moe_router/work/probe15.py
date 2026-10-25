import sys, os, numpy as np, torch
sys.path.insert(0, os.path.abspath('..')); sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0,'../../speccheck')
os.chdir('../..')
import oracle_arm as O, moe_ref
man = O.json.load(open(O.MANIFEST)); D = {r['name']: r for r in man['dumps']}
g = O.load_geometry() if hasattr(O, 'load_geometry') else None
import glob
idx = moe_ref.load_index()
blk='layer0'; bo = moe_ref.Block.parse(blk)
gu_n = bo.prefix + moe_ref.SUFFIXES['gate_up']; dn_n = bo.prefix + moe_ref.SUFFIXES['down']
gu_s = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(gu_n, idx))
dn_s = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(dn_n, idx))
gu_h, gu_hn = moe_ref.read_header(gu_s); dn_h, dn_hn = moe_ref.read_header(dn_s)
E, twoI, H, I = 256, 1024, 2048, 512
gu = np.memmap(gu_s, dtype=np.uint16, mode='r', offset=gu_hn+gu_h[gu_n]['data_offsets'][0], shape=(E,twoI,H))
dn = np.memmap(dn_s, dtype=np.uint16, mode='r', offset=dn_hn+dn_h[dn_n]['data_offsets'][0], shape=(E,H,I))
raw_f, raw_u = O.read_arr(D['layer0.expert_raw'])
xb_f, xb_u = O.read_arr(D['layer0.input'])
ids = O.read_arr(D['layer0.topk_ids'])[0]
geom = O.geometry_from(man) if hasattr(O,'geometry_from') else None
print('ids[0]=', ids[0])
xt = O.u16_to_torch_bf16(np.asarray(xb_u[0], dtype=np.uint16)).reshape(1,H)
want = raw_f[0,0].astype(np.float64)
best=(-2,None)
for e in range(0, E):
    y = moe_ref._expert_apply(xt, O.u16_to_torch_bf16(np.asarray(gu[e],dtype=np.uint16)),
                              O.u16_to_torch_bf16(np.asarray(dn[e],dtype=np.uint16)),
                              moe_ref.load_geometry(),
                              torch.float32, swap_halves=False).numpy().reshape(-1).astype(np.float64)
    c = float(y@want/(np.linalg.norm(y)*np.linalg.norm(want)))
    if c>best[0]: best=(c,e)
    if e<2: print('e=%d y[:3]=%s want[:3]=%s'%(e, y[:3], want[:3]))
print('best cos=%.6f at expert %d (pinned slot0 expert=%d)'%(best[0],best[1],ids[0,0]))
