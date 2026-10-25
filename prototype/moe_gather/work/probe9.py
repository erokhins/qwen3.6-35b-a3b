"""What does the audited reader cost per element, and is its selftest marker fresh?

Prices two things gate.py has to budget before it can declare a sample (AGENTS.md: budget
verification to what a change can affect; pitfalls/50: no heredoc into an interpreter, so this is a
file):
  1. the audit selftest marker — gguf_reader refuses tensor data unless it is current against the
     bytes of tools/audit_published_quant.py and tools/gguf_reader.py;
  2. tools/gguf_reader.dequant's throughput on real published Q4_K / Q5_K / Q6_K bytes, because the
     Python expectation side pays it once per sampled expert and the sample size follows from it.

CPU only, structure + a bounded window of tensor data. No timing of the device: prints
`no timing taken`. The wall figures here price PYTHON, not the kernel.
"""
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)
import gguf_reader as gr            # noqa: E402
import audit_published_quant as apq  # noqa: E402

GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def marker_state():
    if not os.path.exists(apq.SELFTEST_MARK):
        return "missing", None
    mark = json.load(open(apq.SELFTEST_MARK))
    ok = (mark.get("gate_sha256") == sha(os.path.join(REPO, "tools/audit_published_quant.py"))
          and mark.get("reader_sha256") == sha(os.path.join(REPO, "tools/gguf_reader.py")))
    return ("fresh" if ok else "STALE"), mark


def main():
    state, mark = marker_state()
    print("marker file=%s state=%s" % (os.path.basename(apq.SELFTEST_MARK), state))
    if state != "fresh":
        print("VERDICT cannot open GGUF data: marker is %s" % state)
        return 1
    gr.grant_selftest_token(apq.SELFTEST_TOKEN)
    gg = gr.Gguf(GGUF)
    print("gguf tensors=%d data_start=%d" % (gg.n_tensors, gg.data_start))
    want = ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
            "blk.39.ffn_down_exps.weight"]
    for name in want:
        t = gg.by_name.get(name)
        if t is None:
            print("MISSING %s" % name)
            continue
        # one expert's window: expert 0 for gate, expert 0 for down
        per = t.n_elts // 256
        nb = per // gr.BLOCK[t.type_name][0]
        bb = gr.BLOCK[t.type_name][1]
        t0 = time.time()
        buf = gg.read(apq.SELFTEST_TOKEN, gg.data_start + t.offset, nb * bb, "probe9 pricing")
        t1 = time.time()
        vals = gr.dequant(buf, t.type_name, per)
        t2 = time.time()
        print("%-30s type=%-5s expert0_elts=%d read=%.3fs dequant=%.3fs rate=%.2f Melem/s"
              % (name, t.type_name, per, t1 - t0, t2 - t1, per / max(t2 - t1, 1e-9)))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
