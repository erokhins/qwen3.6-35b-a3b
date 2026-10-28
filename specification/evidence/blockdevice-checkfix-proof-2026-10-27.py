#!/usr/bin/env python3
"""Proof harness for the two window-`j` check-layer defects — no device act, no timing taken.

WHAT THIS IS. Window `j` (`devices/evidence/device-window-2026-10-27-g.md` §"Two defects the window
exposed") found that `no-control-contradicts-an-exact-reproduction` printed `verdict=PASS 0 reproduction
legs` in a log that had printed three — `parsed=0 → PASS` reached through ORDERING: the check parsed
`ctx["text"]`, a snapshot taken before the reproduction legs existed. And that the terminal tallies
disagreed: `RESULT ... findings=4` beside an attribution-guard detail counting `2 findings` (its own
fabricated stale text, a different denominator nobody named) and three `verdict=FAIL` lines of which the
third is the `BLOCKDEVICE` verdict echo, not a divergence.

WHAT IS REAL HERE. The REAL, UNMODIFIED functions from `engine/block_device.py` —
`check_control_paths`, `control_attribution_contradiction_guard`, `reject_tally_findings`,
`reproduction_leg_finding`, `Verifier` — driven over the printed text of window `j`'s own REAL device
run (`blockdevice-device-2026-10-27-j1.log`, `RESULT mode=device ... findings=4 ... counters_unparsed=0`).
That log is the "device-shaped run": the check's rule is defined over printed lines, and this feeds it
the printed lines of a device run in full, the way the fixed code reads them (live buffer at check time).
Nothing here touches a device, loads HIP, launches a kernel, or takes a timing.

WHAT IT PROVES, in order, each expectation asserted (any miss exits non-zero):

  1. the fixed check over the device-shaped text prints a NON-ZERO leg count and PASSes;
  2. the guard's `GWITNESS ... leg=this-run` line shows NON-EMPTY `exact_pairs` in the device-shaped run
     (window `j` printed `exact_pairs=none` against three printed legs);
  3. non-vacuity (a): a fabricated SILENT control denying a pair the run reproduced reddens the check;
  4. non-vacuity (b): the same text with the reproduction lines REMOVED reddens the check through the
     explicit zero-count assertion instead of the check passing over an empty list;
  5. the CPU witness keeps its honest exemption (no legs, no device act, not a finding);
  6. `_REJECT_TALLY`: `reject_tally_findings` reconciles window `j`'s `findings=4` against the printed
     records exactly — 2 CHECK `verdict=FAIL` lines + 2 unfired controls printed `fired=SILENT` — and
     names the third `verdict=FAIL` line as the `BLOCKDEVICE` verdict echo, while two fabrications
     (finding-without-record, record-without-finding) redden the predicate.

`no timing taken`.
"""
import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import block_device as BD  # noqa: E402

J1 = (HERE / "blockdevice-device-2026-10-27-j1.log").read_text().splitlines()
CPU = (HERE / "blockdevice-cpu-2026-10-27-p7.log").read_text().splitlines()
failures = []


def expect(name, cond, evidence):
    print("PROOF name=%s verdict=%s %s" % (name, "PASS" if cond else "FAIL", evidence))
    if not cond:
        failures.append(name)


def check_line(text_lines, device):
    """Run the REAL contradiction check over a text, and hand back its printed CHECK line."""
    v = BD.Verifier()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        BD.check_control_paths(v, {"device": device, "text": "\n".join(text_lines)})
    lines = [ln for ln in buf.getvalue().splitlines()
             if ln.startswith("CHECK name=no-control-contradicts-an-exact-reproduction")]
    return lines[0], buf.getvalue()


print("HARNESS substrate=%s lines=%d note=printed text of window j's real device run through the "
      "unmodified driver functions; no device act" % ((HERE / "blockdevice-device-2026-10-27-j1.log").name,
                                                      len(J1)))

# 1 — the fixed check line, device-shaped, with its non-zero leg count.
line, _ = check_line(J1, True)
print("FIXED-CHECK %s" % line)
expect("device-shaped-nonzero-legs",
       "verdict=PASS" in line and line.startswith(
           "CHECK name=no-control-contradicts-an-exact-reproduction verdict=PASS 2 reproduction legs")
       and "d_hip_device_vs_d_hip_hostpath,g_policy_a_hip_device_vs_hip_hostpath" in line,
       "expected PASS with 2 exact legs (GREPRO + DREPRO steps=-1) and both pair names")

# 2 — the guard's this-run leg with non-empty exact_pairs (printed verbatim by the real control).
v = BD.Verifier()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    BD.control_attribution_contradiction_guard(v, {"device": True, "text": "\n".join(J1)})
guard_out = buf.getvalue().splitlines()
for ln in guard_out:
    if ln.startswith("GWITNESS guard=attribution-contradiction") or ln.startswith("CONTROL name=attribution"):
        print("GUARD %s" % ln)
witness = [ln for ln in guard_out if "leg=this-run" in ln]
expect("guard-exact-pairs-nonempty",
       bool(witness) and "exact_pairs=none" not in witness[0]
       and "exact_pairs=d_hip_device_vs_d_hip_hostpath,g_policy_a_hip_device_vs_hip_hostpath" in witness[0],
       "window j printed `exact_pairs=none`; the fixed guard reads the live text")

# 3 — non-vacuity (a): a fabricated denial of a pair THIS run reproduced reddens the check.
deny = J1 + ["CONTROL name=fabricated-supervisor-denial fired=SILENT reason=no injected cause reproduces "
             "the device divergence of d_hip_device_vs_d_hip_hostpath"]
line_a, _ = check_line(deny, True)
print("PROOF-A %s" % line_a)
expect("fabricated-denial-reddens",
       "verdict=FAIL" in line_a and "fabricated-supervisor-denial denies d_hip_device_vs_d_hip_hostpath"
       in line_a,
       "a SILENT control denying the reproduced d pair must redden the check and name the denial")

# 4 — non-vacuity (b): repro lines REMOVED -> the zero-count assertion reddens, the check does not pass.
stripped = [ln for ln in J1 if not ln.startswith(BD.REPRO_HEADS)]
line_b, _ = check_line(stripped, True)
print("PROOF-B %s" % line_b)
expect("zero-leg-assertion-reddens",
       "verdict=FAIL" in line_b and "0 reproduction legs" in line_b
       and "device run parsed 0 exact reproduction legs" in line_b,
       "window j's own text with the repro lines removed is the window-j shape; it must redden")

# 5 — the CPU exemption stays honest (zero legs is NOT a finding when there is no device act).
line_c, _ = check_line(CPU, False)
print("CPU-EXEMPT %s" % line_c[:260])
expect("cpu-zero-is-exempt", "verdict=PASS" in line_c and "exempt in a CPU run" in line_c,
       "device=False with 0 legs must pass by the named exemption, not by vacuity")

# 6 — `_REJECT_TALLY` reconciliation against window j's own printed records.
fails = [ln for ln in J1 if "verdict=FAIL" in ln]
for ln in fails:
    print("FAIL-LINE %s" % ln[:200])
red = [ln[len("RED "):] for ln in J1 if ln.startswith("RED ")]
print("REDRECORDS %s" % " | ".join(red))
print("TALLY-LINES %s" % [ln[:120] for ln in J1 if "findings=" in ln and ("RESULT" in ln or "findings=2" in ln)][:3])
recon = BD.reject_tally_findings(red, J1)
expect("tally-reconciles-on-window-j", not recon,
       "findings=4 = 2 CHECK FAIL lines + 2 unfired controls printed fired=SILENT; recon=%s" % (recon or "clean"))
expect("third-fail-line-is-the-verdict-echo",
       len(fails) == 3 and any(ln.startswith("BLOCKDEVICE verdict=FAIL") for ln in fails),
       "the third verdict=FAIL line is the BLOCKDEVICE terminal verdict — the tally echoed, not a third divergence")
fab_missing = BD.reject_tally_findings(["fabricated-finding-without-a-printed-record"], [])
fab_extra = BD.reject_tally_findings([], ["CHECK name=fabricated-verdict-failure verdict=FAIL witness only"])
fab_clean = BD.reject_tally_findings([], [])
print("TALLY-WITNESS missing-flagged=%s extra-flagged=%s clean=%s"
      % (bool(fab_missing), bool(fab_extra), not fab_clean))
expect("tally-predicate-not-inert", fab_missing and fab_extra and not fab_clean,
       "the reconciliation predicate reddens on both fabrications and stays clean on an honest empty pair")

print("HARNESS verdict=%s substrates=window-j device log + cpu witness log no device act occurred, "
      "no timing taken" % ("PASS" if not failures else "FAIL findings=%s" % ",".join(failures)))
sys.exit(1 if failures else 0)
