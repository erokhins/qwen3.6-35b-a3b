"""Non-vacuity witness for the two textual prohibitions in engine/stage_run.py.

`copy-path-owns-no-lock` passes when LOCK_INVOKED and HIP_IMPORT find nothing in the driver. A
pattern that can never match anything produces the same PASS, so this script feeds each pattern
strings it MUST fire on. If a leg prints FIRES=NO for a should-fire probe, the prohibition is
decorative and the PASS in the gate means nothing.

Run: /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/work/probe_lock_check_vacuity.py
Exit 0 only if every should-fire probe fired and neither fired on the driver's own text.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, os.pardir, "stage_run.py")

body = open(DRIVER).read()


def grab(name):
    # Trailing comments and a flags argument are both legal on these lines, so grab everything
    # inside the outer re.compile(...) and let eval decide whether it is one arg or (pattern, flags).
    m = re.search(r"^%s\s*=\s*re\.compile\((.*)\)\s*(?:#.*)?$" % name, body, re.M)
    if not m:
        raise SystemExit("PATTERN_ABSENT name=%s -- the check cannot be evaluated at all" % name)
    arg = eval("(" + m.group(1) + ")", {"re": re})
    if isinstance(arg, str):
        arg = (arg,)
    return re.compile(*arg)


LOCK = grab("LOCK_INVOKED")
IMP = grab("HIP_IMPORT")

CASES = [
    ("LOCK_INVOKED", LOCK, "bash tools/gpu_lock.sh acquire tag", True),
    ("LOCK_INVOKED", LOCK, "/tmp/pi_gpu.lock", False),
    ("LOCK_INVOKED", LOCK, "gpu" + "_lock.sh", True),
    ("HIP_IMPORT", IMP, "import hip", True),
    ("HIP_IMPORT", IMP, "from pyrhip import x", False),
    ("HIP_IMPORT", IMP, "from rocm import util", True),
    ("HIP_IMPORT", IMP, "import ctypes", False),
    ("HIP_IMPORT", IMP, 'HIP_LIBRARY = "/opt/rocm/lib/libamdhip64.so"', False),
    ("HIP_IMPORT", IMP, "import amdsmi", True),
]

bad = 0
for name, rx, probe, want in CASES:
    got = bool(rx.search(probe))
    ok = (got == want)
    if not ok:
        bad += 1
    print("PROBE pattern=%s want_fire=%s got_fire=%s ok=%s probe=%r"
          % (name, want, got, "YES" if ok else "NO", probe[:48]))

for name, rx in (("LOCK_INVOKED", LOCK), ("HIP_IMPORT", IMP)):
    hits = [m.group(0) for m in rx.finditer(body)]
    print("SELFSCAN pattern=%s hits_in_driver=%d values=%s"
          % (name, len(hits), sorted(set(hits))[:3] or "none"))
    if hits:
        bad += 1
        print("  BAD: the driver's own text trips the prohibition it asserts")

print("VERDICT %s bad=%d cases=%d" % ("PASS" if bad == 0 else "FAIL", bad, len(CASES)))
sys.exit(0 if bad == 0 else 1)
