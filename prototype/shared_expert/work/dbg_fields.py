import struct, sys
from array import array
NAN = 0x7FC00000
def words(p):
    a = array('I'); a.frombytes(open(p,'rb').read()); return a
E = words('work/inputs.bin'); G = words('work/host_out.f32')
CASE=5633; OFF={'gate':0,'up':512,'act':1024,'raw':1536,'out':3584,'prob':5632}
LEN={'gate':512,'up':512,'act':512,'raw':2048,'out':2048,'prob':1}
for ci in (0,2,4,11):
    line = []
    for k in ('gate','up','act','raw','out','prob'):
        d = sum(1 for i in range(LEN[k]) if G[ci*CASE+OFF[k]+i] != E[ci*CASE+OFF[k]+i])
        line.append(f"{k}={d}/{LEN[k]}")
    # show a couple of mismatching gate words
    ex = [(i, hex(E[ci*CASE+i]), hex(G[ci*CASE+i])) for i in range(4) if G[ci*CASE+i] != E[ci*CASE+i]]
    print(f"case {ci}: " + " ".join(line), "first gate diffs:", ex[:3])
