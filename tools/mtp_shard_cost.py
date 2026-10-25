#!/usr/bin/env python3
"""Which safetensors shards would we actually have to fetch for a given ask?

Answers two live questions for D-03 without guessing:
  * MTP without the checkpoint: the published GGUF carries no draft block, so what does keeping D-05
    cost in bytes?
  * A *sampled* provenance audit of a third-party quant: instead of all 67 GiB, fetch the shards that
    hold a chosen set of layers and check those tensors against the GGUF. Shards are independent
    files, so this is a real partial download - the catch is that shards run over contiguous layer
    ranges, so a witness spread over all 40 layers touches all 26 shards, while a sampled witness
    (layer 0, layer 39, mtp) does not. That is exactly what this prints.

Reads only the committed index. Estimated shard bytes = index total_size / shard count; the real
number comes from the file listing at download time, so this sizes a decision, it is not a measurement.

Usage:
  python3 tools/mtp_shard_cost.py [--index PATH] [--layers 0,39] [--mtp] [--all-text] [--router-norm]
"""
import json
import re
import sys

GIB = 2.0 ** 30


def main():
    a = sys.argv[1:]
    idx_path = a[a.index("--index") + 1] if "--index" in a else \
        "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    per = idx.get("metadata", {}).get("total_size", 0) / max(1, len(shards))
    print("index: %d tensors in %d shards, total_size %.2f GiB, ~%.2f GiB/shard (estimate)"
          % (len(wm), len(shards), idx.get("metadata", {}).get("total_size", 0) / GIB, per / GIB))

    want = set()
    asked = []
    if "--mtp" in a or not any(x in a for x in ("--layers", "--all-text", "--router-norm")):
        want |= {k for k in wm if k.startswith("mtp.")}
        asked.append("mtp.* (the draft block)")
    if "--layers" in a:
        ls = [int(x) for x in a[a.index("--layers") + 1].split(",") if x != ""]
        for L in ls:
            want |= {k for k in wm if re.match(r"^model\.language_model\.layers\.%d\." % L, k)}
        asked.append("layers %s" % ls)
    if "--router-norm" in a:
        want |= {k for k in wm if re.search(r"gate\.weight$|\.norm\.weight$|linear_attn\.(norm|conv1d|A_log|dt_bias)", k)}
        asked.append("all routers + all norms + GDN control tensors (every layer)")
    if "--all-text" in a:
        want |= {k for k in wm if not k.startswith("model.visual")}
        asked.append("the whole text stack")

    need = sorted({wm[k] for k in want})
    print("\nasked for: %s" % "; ".join(asked))
    print("tensors: %d   distinct shards to fetch: %d of %d   ~%.2f GiB"
          % (len(want), len(need), len(shards), len(need) * per / GIB))
    if len(need) <= 6:
        for s in need:
            print("   %s" % s)
    else:
        print("   (more than 6 shards - a witness spread this thin is a full download in disguise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
