#!/usr/bin/env python3
"""ulp_policy_smoke_staged.py — apply the staged patches in a mirror tree and RUN the patched code.

`ulp_policy_stage_patches.py` proves each patch applies alone and that the patched file parses. That is
not enough: two patches for the same file have to compose, and code that parses is not code that runs.
This script builds a throwaway mirror of the files the patches touch, applies every staged patch to it,
and then imports the patched modules and checks the patched functions against the shared ruler over a
corpus. The repo is never modified — this is the dress rehearsal, not the performance.

  /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \
      specification/evidence/ulp_policy_smoke_staged.py

Counts only; no timing; no device.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
import ulp_policy as UP                                       # noqa: E402

MIRROR = "/tmp/ulp_stage_mirror"
TARGETS = ["tools/audit_published_quant.py", "tools/moe_stats.py", "tools/gguf_reader.py",
           "tools/gguf_width_table.py", "tools/gguf_header_types.py", "tools/ggml_block_geom.py",
           "speccheck/moe_ref.py", "prototype/moe_router/gate.py",
           # the two files the imported modules resolve through REPO at import time:
           "engines/llama.cpp/ggml/src/ggml-common.h",
           "checkpoints/Qwen3.6-35B-A3B.config.json",
           "checkpoints/Qwen3.6-35B-A3B.safetensors.index.json"]
PATCHES = sorted(f for f in os.listdir(HERE) if f.startswith("ulp-policy-migrate-"))


def build_mirror():
    shutil.rmtree(MIRROR, ignore_errors=True)
    for t in TARGETS:
        dst = os.path.join(MIRROR, t)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, t), dst)
    out = []
    for p in PATCHES:
        r = subprocess.run(["patch", "-p1", "--silent", "-i", os.path.join(HERE, p)], cwd=MIRROR,
                           capture_output=True, text=True)
        out.append((p, r.returncode == 0, (r.stdout + r.stderr).strip()[:120]))
    return out


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def corpus_f32(n=4000, seed=20261026):
    import random
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        a = rnd.randrange(0, 1 << 32)
        b = a + rnd.choice([-3, -2, -1, 1, 2, 3, 1 << 23])
        if 0 <= b <= 0xFFFFFFFF:
            out.append((a, b))
    return out


def main():
    print("PATCHES staged=%d" % len(PATCHES))
    allok = True
    for p, ok, msg in build_mirror():
        print("APPLY patch=%s ok=%d %s" % (p, 1 if ok else 0, msg))
        allok = allok and ok
    if not allok:
        print("RESULT FAIL staged patches do not compose in the mirror tree")
        return 1
    r32 = UP.Ruler("f32")
    pairs = corpus_f32()

    audit = load(os.path.join(MIRROR, "tools/audit_published_quant.py"), "patched_audit")
    bad = 0
    nan_numeric = 0
    for a, b in pairs:
        fa = UP._value_of("f32", a)
        fb = UP._value_of("f32", b)
        kind, steps = r32.classify(a, b)
        got = audit.ulp_dist(fa, fb)
        if kind in (UP.NAN_PAIR, UP.NAN_OTHER):
            nan_numeric += 1 if isinstance(got, int) else 0
        elif got != steps:
            bad += 1
    print("SMOKE target=audit_published_quant.ulp_dist pairs=%d mismatches=%d nan_pairs_kept_numeric=%d"
          % (len(pairs), bad, nan_numeric))
    allok = allok and bad == 0

    router = load(os.path.join(MIRROR, "prototype/moe_router/gate.py"), "patched_router")
    bad = none_ct = 0
    for a, b in pairs:
        kind, steps = r32.classify(a, b)
        got = router.ulpdist_bits(a, b)
        if kind in (UP.NAN_PAIR, UP.NAN_OTHER):
            none_ct += 1 if got is None else 0
        elif got != steps:
            bad += 1
    straddle = sum(1 for a, b in pairs if (a < 0x80000000) != (b < 0x80000000)
                   and router.ulpdist_bits(a, b) == r32.classify(a, b)[1])
    print("SMOKE target=moe_router.ulpdist_bits pairs=%d mismatches=%d nan_answered_None=%d "
          "straddling_pairs_now_agree=%d" % (len(pairs), bad, none_ct, straddle))
    allok = allok and bad == 0 and none_ct > 0 and straddle > 0

    mref = load(os.path.join(MIRROR, "speccheck/moe_ref.py"), "patched_moe_ref")
    r16 = UP.Ruler("bf16")
    import math
    bad = same = 0
    for img in range(0, 1 << 16):
        kind, _ = r16.classify(img, img)
        if r16.is_nan(img):
            continue
        v = UP._value_of("bf16", img)
        old = (0.0 if (v == 0 or not math.isfinite(v))
               else math.ldexp(1.0, math.floor(math.log2(abs(v))) - 7))
        new = mref.ulp_of(v)
        if new == old:
            same += 1
        else:
            bad += 1
    print("SMOKE target=moe_ref.ulp_of images=%d identical_to_the_typed-7_body=%d changed=%d "
          "(bf16 mant derives to %d, so it must be unchanged)"
          % ((1 << 16), same, bad, UP.format_facts("bf16").mant))
    allok = allok and bad == 0

    stats = load(os.path.join(MIRROR, "tools/moe_stats.py"), "patched_moe_stats")
    want = 2.0 ** -(UP.format_facts("bf16").mant - 1)
    ok = abs(stats.BF16_REL_WIDTH - want) < 1e-15
    print("SMOKE target=moe_stats.BF16_REL_WIDTH value=%.6g derived_half_ulp=%.6g ok=%d "
          "(was 0.0019531 = the typed-8 value)" % (stats.BF16_REL_WIDTH, want, 1 if ok else 0))
    allok = allok and ok

    print("RESULT %s staged patches compose and the patched code runs and agrees"
          % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
