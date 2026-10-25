"""Which tensors does one dump step actually carry? Written because the shared expert's *input* is
what decides whether the oracle can be validated against the reference: shared_raw is the output of
down_proj, so gate/up/SwiGLU can only be checked against a dump that carries the block input."""
import sys
from safetensors.numpy import load_file

d = load_file(sys.argv[1])
for k in sorted(d):
    v = d[k]
    print(f"{k:56s} {str(v.shape):16s} {str(v.dtype):8s} n={v.size}")
