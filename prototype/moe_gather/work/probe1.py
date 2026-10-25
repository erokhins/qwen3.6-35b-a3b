import json, os, sys, collections
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
sys.path.insert(0, REPO+'/tools')
idx=json.load(open(REPO+'/checkpoints/Qwen3.6-35B-A3B.safetensors.index.json'))['weight_map']
have=set(os.listdir(REPO+'/models/qwen36-src'))
suf={'router':'mlp.gate.weight','gate_up':'mlp.experts.gate_up_proj','down':'mlp.experts.down_proj',
 'shared_gate':'mlp.shared_expert.gate_proj.weight','shared_up':'mlp.shared_expert.up_proj.weight',
 'shared_down':'mlp.shared_expert.down_proj.weight','shared_scalar_gate':'mlp.shared_expert_gate.weight'}
for blk,pref in [('layer0','model.language_model.layers.0.'),('layer1','model.language_model.layers.1.'),
                 ('layer2','model.language_model.layers.2.'),('layer39','model.language_model.layers.39.'),
                 ('mtp0','mtp.layers.0.')]:
    row=[]
    for r,s in suf.items():
        sh=idx.get(pref+s)
        row.append("%s:%s%s"%(r,sh,"" if sh in have else " MISSING"))
    print(blk, " ".join(row))
print("have:", sorted(have))
import gguf_reader as gr
import audit_published_quant as apq
print("SELFTEST_MARK", apq.SELFTEST_MARK, os.path.exists(apq.SELFTEST_MARK))
print("GGUF_PATH", apq.GGUF_PATH, os.path.exists(apq.GGUF_PATH))
gg=gr.Gguf(apq.GGUF_PATH)
print("n_tensors",gg.n_tensors,"data_start",gg.data_start,"align",gg.alignment)
cnt=collections.Counter(); down={}
for t in gg.tensors:
    if t.name.endswith('ffn_down_exps.weight'): down[t.name.split('.')[1]]=t.type_name
    if t.name.endswith(('ffn_gate_exps.weight','ffn_up_exps.weight')): cnt[(t.name.split('.')[-2],t.type_name)]+=1
print("gate/up types:",dict(cnt))
print("down widths:",collections.Counter(down.values()))
print("q6 layers:",[k for k,v in down.items() if v=='Q6_K'])
want={'blk.0.ffn_gate_exps.weight','blk.0.ffn_up_exps.weight','blk.0.ffn_down_exps.weight','blk.19.ffn_down_exps.weight'}
for t in gg.tensors:
    if t.name in want: print(t.name, t.dims, t.shape_hf, t.type_name, t.n_elts, t.offset, gr.BLOCK[t.type_name])
