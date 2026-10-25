"""speccheck/hf_capability_probe.py — can the installed `transformers` build and load
this checkpoint, or only its MoE sub-blocks?

Run as a subprocess with an address-space limit so an attempted 66 GiB materialisation
fails inside the probe instead of swapping the box the other workers are running on:

    resource.setrlimit(RLIMIT_AS, 24 GB)          # set below, before anything loads

`speccheck/cross_check_hf.py` drives this and archives the output under
`speccheck/evidence/`. Each probe prints one `CAP <name>: OK|FAIL <detail>` line; the
driver asserts it saw the expected number of lines (pitfalls/40 §A driver whose parser
matches nothing prints PASS).

argv: <config.json> <shard dir> <safetensors index.json>
"""

import json
import os
import resource
import sys
import tempfile

resource.setrlimit(resource.RLIMIT_AS, (int(24e9), int(24e9)))

import torch  # noqa: E402
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeConfig, Qwen3_5MoeTextConfig)

cfg = json.load(open(sys.argv[1]))


def show(tag, fn):
    try:
        print(f"CAP {tag}: OK {fn()}")
    except Exception as e:
        print(f"CAP {tag}: FAIL {type(e).__name__}: {str(e)[:400]}")


show("config_from_json", lambda: type(Qwen3_5MoeConfig(**cfg)).__name__)


def build_full():
    with torch.device("meta"):
        m = M.Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(**cfg))
    return f"{sum(1 for _ in m.named_parameters())} params"


show("build_conditional_generation_meta", build_full)


def build_text():
    with torch.device("meta"):
        m = M.Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**cfg["text_config"]))
    return f"{len(m.model.layers)} layers"


show("build_text_causallm_meta", build_text)

# The MTP block is 19 tensors in the checkpoint (`mtp.fc`, `mtp.norm`,
# `mtp.pre_fc_norm_*`, `mtp.layers.0.*`). transformers 5.19 ships no class for it and the
# loader is told to ignore exactly those keys, which is the reason this project's MTP MoE
# block has to be instantiated from the submodules rather than loaded.
show("mtp_class_in_module", lambda: sorted(
    a for a in dir(M) if "mtp" in a.lower()) or "NO_MTP_CLASS_IN_MODULE")
show("keys_to_ignore_on_load_unexpected", lambda: (
    getattr(M.Qwen3_5MoeForCausalLM, "_keys_to_ignore_on_load_unexpected", None),
    getattr(M.Qwen3_5MoeModel, "_keys_to_ignore_on_load_unexpected", None)))
show("default_experts_implementation_bare_text_config", lambda: Qwen3_5MoeTextConfig(
    **cfg["text_config"])._experts_implementation)
show("default_experts_implementation_nested", lambda: Qwen3_5MoeConfig(
    **cfg).text_config._experts_implementation)


def experts_impl_after_build():
    """The dispatch key is *unset* on a freshly built config and only gets its default
    when the model is constructed, so a `from_pretrained` model does not run the eager
    expert loop this project's spec mirrors unless it is pinned. That is why
    `cross_check_hf.py` pins `eager` explicitly and reports the unpinned default."""
    c = Qwen3_5MoeConfig(**cfg)
    with torch.device("meta"):
        M.Qwen3_5MoeForConditionalGeneration(c)
    return f"bare={Qwen3_5MoeTextConfig(**cfg['text_config'])._experts_implementation} " \
           f"after_build={c.text_config._experts_implementation}"


show("experts_implementation_after_model_build", experts_impl_after_build)


def local_load(mode):
    """Two honest attempts to `from_pretrained` from what is on disk. `real-index` uses
    the committed 26-shard index (22 files were never downloaded — DECISIONS.md D-03
    deferred the full checkpoint); `truncated` rewrites the index to name only the four
    acquired shards, so the loader gets past file discovery and must then meet the 38
    decoder layers we do not have."""
    from transformers import AutoModelForCausalLM
    d = tempfile.mkdtemp(prefix="cap-")
    import shutil
    shutil.copy(sys.argv[1], os.path.join(d, "config.json"))
    idx = json.load(open(sys.argv[3]))
    wm = idx["weight_map"]
    have = set(os.listdir(sys.argv[2]))
    if mode == "real-index":
        json.dump(idx, open(os.path.join(d, "model.safetensors.index.json"), "w"))
        want = sorted(set(wm.values()))
    else:
        keep = {k: v for k, v in wm.items() if v in have}
        json.dump({"metadata": idx["metadata"], "weight_map": keep},
                  open(os.path.join(d, "model.safetensors.index.json"), "w"))
        want = sorted(set(keep.values()))
        print(f"POP truncated_index_population: OK kept {len(keep)} of {len(wm)} tensors "
              f"in {len(want)} shards")
    for s in want:
        src = os.path.join(sys.argv[2], s)
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), os.path.join(d, s))
    return type(AutoModelForCausalLM.from_pretrained(
        d, torch_dtype=torch.bfloat16)).__name__


show("from_pretrained_real_index_26_shards", lambda: local_load("real-index"))
show("from_pretrained_truncated_index_4_shards", lambda: local_load("truncated"))
