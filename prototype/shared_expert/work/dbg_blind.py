from array import array
def words(p):
    a=array('I'); a.frombytes(open(p,'rb').read()); return a
G=words('work/host_out.f32'); E=words('work/inputs.bin')
CASE=5633; OFF={'gate':0,'up':512,'act':1024,'raw':1536,'out':3584,'prob':5632}; LEN=dict(gate=512,up=512,act=512,raw=2048,out=2048,prob=1)
def fdiff(a,b,fi,fn):
    return sum(1 for i in range(fn) if G[a*CASE+fi+i]!=G[b*CASE+fi+i])
labels={}
for line in open('work/cases.tsv'):
    if line.startswith('#'): continue
    f=line.split('\t'); labels[int(f[0])]=f[1]
print('labels', labels)
honest0=0
for ci,lab in sorted(labels.items()):
    if ci==0: continue
    d={k:fdiff(honest0,ci,OFF[k],LEN[k]) for k in OFF}
    print(f"case{ci:3d} {lab:16s} vs honest0: " + " ".join(f"{k}={v}" for k,v in d.items()))
