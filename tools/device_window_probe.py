#!/usr/bin/env python3
"""Device-window probe: peak RSS + per-process VRAM during a full-offload llama.cpp run.

Owns no lock (the caller holds tools/gpu_lock.sh). Prints no wall-clock timing except the two
figures this window exists to obtain, and both are labelled with what they are.
"""
import json, os, re, subprocess, sys, threading, time

REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
CLI = os.path.join(REPO, "engines/llama.cpp/build-hip/bin/llama-cli")
GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ("Explain, in order, how a mixture-of-experts transformer routes a token: "
          "the router logits, the top-k selection, the renormalisation of the weights, and the "
          "weighted sum of expert outputs. Then give one reason this is hard to keep fast at "
          "batch size one.")


def read_status(pid):
    out = {}
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                for key in ("VmPeak", "VmHWM", "VmRSS"):
                    if line.startswith(key):
                        out[key] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def read_kfd(pid):
    """Per-process VRAM from the amdgpu KFD, when the kernel exposes it. On this box the
    /sys/class/kfd/kfd/proc/<pid>/ files do not exist, so the caller falls back to the device-wide
    amdgpu sysfs counter (read_drm_vram) and says which one it used."""
    out = {}
    for name, fn in (("vram_used", "mem_info_vram_used"),
                     ("vram_vis_used", "mem_info_vram_vis_used")):
        p = "/sys/class/kfd/kfd/proc/%d/%s" % (pid, fn)
        try:
            with open(p) as fh:
                out[name] = int(fh.read().strip())
        except (OSError, ValueError):
            pass
    return out


def read_drm_vram():
    """Device-wide memory in use, straight from amdgpu sysfs -- no subprocess, cheap enough to poll.

    On this APU `mem_info_vram_used` tracks only the small dedicated/visible window (it read 1.68 GiB
    while 21 GiB of weights were being read by the GPU), so the number that actually sees the weights is
    `mem_info_gtt_used`: the GGUF is mmap'd and the device reads it over the GTT, which is why VmHWM
    stayed at 1.47 GiB. All three are recorded rather than just the one that moves."""
    import glob
    out = {}
    for key, pat in (("drm_vram_used", "/sys/class/drm/card*/device/mem_info_vram_used"),
                     ("drm_vis_vram_used", "/sys/class/drm/card*/device/mem_info_vis_vram_used"),
                     ("drm_gtt_used", "/sys/class/drm/card*/device/mem_info_gtt_used")):
        best = 0
        for p in glob.glob(pat):
            try:
                with open(p) as fh:
                    v = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            if v > best:
                best = v
        out[key] = best
    return out


def read_mem_available():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def main():
    n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    cmd = [CLI, "-m", GGUF, "-p", PROMPT, "-n", str(n_tok), "-ngl", "99", "-s", "42",
           "--temp", "0", "--no-display-prompt", "--no-warmup", "-st", "-v"]
    peak = {"VmPeak": 0, "VmHWM": 0, "VmRSS": 0, "vram_used": 0, "vram_vis_used": 0}
    stop = threading.Event()
    samples = [0]

    def sampler(pid):
        while not stop.is_set():
            s = read_status(pid)
            k = read_kfd(pid)
            k.update(read_drm_vram())
            k["mem_available_min"] = read_mem_available()
            for d in (s, k):
                for kk, vv in d.items():
                    if vv > peak.get(kk, 0):
                        peak[kk] = vv
            if k["mem_available_min"] and (low["mem_available_min"] is None or k["mem_available_min"] < low["mem_available_min"]):
                low["mem_available_min"] = k["mem_available_min"]
            samples[0] += 1
            time.sleep(0.2)

    low = {"mem_available_min": None}
    for kk in ("drm_vram_used", "drm_vis_vram_used", "drm_gtt_used"):
        peak[kk] = 0
    mem_available_before = read_mem_available()

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    th = threading.Thread(target=sampler, args=(proc.pid,), daemon=True)
    th.start()
    out = proc.communicate(timeout=1800)[0]
    stop.set()
    th.join(timeout=5)
    load_s = None
    m = re.search(r"model load time = ([0-9.]+) ms", out)
    if m:
        load_s = float(m.group(1)) / 1000.0
    perf = {}
    # llama.cpp's summary line uses a COMMA decimal separator: [ Prompt: 358,0 t/s | Generation: 51,9 t/s ]
    m = re.search(r"\[\s*Prompt:\s*([0-9]+(?:,[0-9]+)?)\s*t/s\s*\|\s*Generation:\s*([0-9]+(?:,[0-9]+)?)\s*t/s\s*\]", out)
    if m:
        perf["pp_t_s"] = float(m.group(1).replace(",", "."))
        perf["tg_t_s"] = float(m.group(2).replace(",", "."))
    off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", out)
    GiB = 1024 ** 3
    print("PROBE n_tok=%d rc=%d samples=%d" % (n_tok, proc.returncode, samples[0]))
    print("PEAK  VmPeak=%.2f GiB  VmHWM=%.2f GiB  last_VmRSS=%.2f GiB" %
          (peak["VmPeak"] / GiB, peak["VmHWM"] / GiB, peak["VmRSS"] / GiB))
    print("KFD   per-process vram=%s" % ("%.2f GiB" % (peak["vram_used"] / GiB) if peak["vram_used"] else "UNAVAILABLE (no kfd proc files)"))
    print("AMDGPU gtt_used_peak=%.2f GiB  vram_used_peak=%.2f GiB  vis_vram_used_peak=%.2f GiB" % (
        peak["drm_gtt_used"] / GiB, peak["drm_vram_used"] / GiB, peak["drm_vis_vram_used"] / GiB))
    print("      host MemAvailable before=%.2f GiB  min during=%s GiB  samples_with_value=%s (the mmap'd weights are host pages read over GTT)" % (
        mem_available_before / GiB,
        "%.2f" % (low["mem_available_min"] / GiB) if low["mem_available_min"] else "NOT MEASURED",
        samples[0] if low["mem_available_min"] else 0))
    print("OFFLOAD  %s" % ("  ".join("%s/%s layers" % o for o in off) if off else "NOT REPORTED (-v gave nothing)"))
    print("RULER   pp_t_s=%s  tg_t_s=%s  model_load_s=%s" % (perf.get("pp_t_s"), perf.get("tg_t_s"), load_s))
    print("      (llama.cpp prints its own timings too; the lines above are the ones this window is for)")
    with open(os.path.join(REPO, "tmp/evidence/llama-full-%d.txt" % n_tok), "w") as fh:
        fh.write(out)


main()
