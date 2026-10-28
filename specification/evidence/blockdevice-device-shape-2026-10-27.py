#!/usr/bin/env python3
"""Device-SHAPE rehearsal of `engine/block_device.py --device`, on a box that must not touch its iGPU.

WHY THIS EXISTS. PASS 9's defect 2 is that the `--device` arm of `engine/block_device.py` has never printed
its `RESULT` line: window `i` (`specification/evidence/blockdevice-device-2026-10-27-i1.log`) died with
`TypeError: '>' not supported between instances of 'str' and 'int'` inside a device-only statement, after
170 lines of otherwise-good output, so the run reported nothing and looked like a hang. A CPU session cannot
answer "does the device arm now reach `RESULT`" by asserting it — that is the same move as a green
`checks=54 findings=0` over a population the run never reached. It can answer it by running the driver's
device shape end to end with the two things a CPU session is forbidden to do — allocate on the iGPU and
launch a kernel — replaced by substitutes that are LOUD about being substitutes.

WHAT IS REAL HERE, and it is most of it.

  * the driver itself: `block_device.run(out, args)` with `args.device=True`, every layout parser, the
    licensor and band from `speccheck/`, all 58 checks and 31 controls, the `Verifier`, the refusals;
  * the arms' compiled bodies: a `--device` request runs the SAME binary without the flag, i.e. its host
    body — the very path the CPU witness runs as `hip_hostpath` — so every perturbation leg (`DREPRO`,
    `GREPRO`, `DDIAL`, `DMAGMATCH`, `DEXP`) computes the arithmetic the driver judges, on the real pinned
    inputs, through the real plan/slab/record formats;
  * the device record bytes: where window `i` left a saved device output on disk, those bytes are copied
    over the rehearsal's `--out`, so `device` vs `host` comparisons are the real device record against the
    real host record and the headline numbers must come out exactly as window `i` printed them;
  * the staging machinery: `stage_run.stage_run_to_device`, `device_read_back`, `device_verdict`, the span
    digest binds and the corrupt-and-restore controls all execute, against a device that is a buffer.

WHAT IS SUBSTITUTE, printed at every use and summarised at the end.

  * `StubHip` — a byte buffer with `CountedHip`'s call surface. No `libamdhip64` is loaded, and the child
    greps its own `/proc/self/maps` for it and refuses if it is there.
  * the path word and the launch counters of an arm report: the twin prints `path=host hip_calls=0
    launches=0`; the rehearsal re-spells `path=device` and copies the counters of the same lane's window-`i`
    `ARM` line, because `device_arm.run_arm` refuses a device run that cannot count a launch. The counters
    are a stand-in for the non-zero rule; they are not a measurement and no timing is printed.

WHAT THE REHEARSAL DOES NOT LET ANYONE BELIEVE. A substituted leg can pass while the thing it stands for is
broken, which is precisely how `checks=54 findings=0` was true and blind in window `i`. So the child also
derives, from the driver's own AST (via `block_device.device_census_derive`), every statement the file
executes only when the device flag is on, traces the run, and prints `DEVONLY path=… status=exercised` or
`status=not-exercised reason=…` for each one. In a rehearsal the honest answer for a kernel-shaped leg is
often `not-exercised`, and printing that is the deliverable.

Usage: `python specification/evidence/blockdevice-device-shape-2026-10-27.py` (parent; runs the children and
audits their logs). `--child [--mutate NAME]` is the child mode the parent uses. Host wall and RSS only:
no timing appears in any line this script prints. no timing taken.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess as SUBPROCESS
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "engine"
WORK = ENGINE / "work"
DEVLOG = ROOT / "specification" / "evidence" / "blockdevice-device-2026-10-27-i1.log"
HIP_LIBRARY_HINTS = ("libamdhip64", "librocm", "hipblas")
MUTATE = None
MUTATE_CONTROL = "control-witness-guard-can-redden"


def log(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# -----------------------------------------------------------------------------------------------
# 1. the substitute device
# -----------------------------------------------------------------------------------------------

class StubHip(object):
    """`device_arm.CountedHip`'s call surface over byte buffers, with no HIP and no kernel.

    Addresses are fake, monotonically increasing, and owned by this object, which lets `_shift` and
    `device_arm.device_ptr` do real pointer arithmetic on them: a control that writes one byte into the
    resident set and re-reads it gets a real different byte back, so `gdn-state-read-is-the-arm-s-own-fingerprint`
    and the short-read-back legs exercise their logic instead of being skipped. What this cannot do is
    compute: nothing here runs on a GPU, and any digest a caller folds over these bytes is a digest of the
    bytes the caller put in, which is exactly the half the staging checks already hold the driver to.
    """

    BASE = 0x7f0000000000

    def __init__(self, *a, **kw):
        self.ptrs = {}
        self.stage = None
        self.stage_bytes = 0
        self.allocs = []
        self.buf = bytearray()
        self.regions = []
        self.next_addr = self.BASE
        self.tally = {"allocs": 0, "slab_bytes": 0, "h2d_calls": 0, "h2d_bytes": 0,
                      "d2h_calls": 0, "d2h_bytes": 0, "frees": 0}
        log("STUBHIP note=byte-buffer device in use; no libamdhip64 loaded, no kernel launched; "
            "allocations and copies are counted, timings are not taken")

    def _locate(self, addr):
        for base, mem in self.regions:
            if base <= addr < base + len(mem):
                return mem, addr - base
        raise AssertionError("stub-device-overread: 0x%x is not inside any allocated region "
                             "(regions=%s); a device read outside what was allocated is what this stub "
                             "would have to hide, so it refuses instead"
                             % (addr, [hex(b) for b, _ in self.regions]))

    def alloc_slab(self, name, nbytes):
        nbytes = int(nbytes)
        mem = bytearray(nbytes)
        base = self.next_addr
        self.next_addr += (nbytes + 0xFFF) & ~0xFFF
        self.regions.append((base, mem))
        p = ctypes.c_void_p(base)
        self.allocs.append(dict(slab=name, bytes=nbytes, rc=0))
        self.ptrs[name] = p
        self.tally["allocs"] += 1
        self.tally["slab_bytes"] += nbytes
        log("STUBHIP op=alloc_slab slab=%s bytes=%d addr=0x%x note=substitute, not a hipMalloc"
            % (name, nbytes, base))
        return p

    def alloc_stage(self, nbytes):
        nbytes = int(nbytes)
        self.buf = bytearray(nbytes)
        self.stage_bytes = nbytes
        self.stage = ctypes.c_void_p(self.next_addr)
        self.next_addr += (nbytes + 0xFFF) & ~0xFFF
        log("STUBHIP op=alloc_stage bytes=%d note=substitute for hipHostMalloc" % nbytes)
        return self.stage

    def fill(self, offset, data):
        n = len(data)
        if offset + n > self.stage_bytes:
            raise AssertionError("staging-overflow: wrote %d B at +%d into a %d B staging buffer"
                                 % (n, offset, self.stage_bytes))
        self.buf[offset:offset + n] = data

    def view(self, offset, n):
        if offset + n > self.stage_bytes:
            raise AssertionError("staging-overread: read %d B at +%d from a %d B staging buffer"
                                 % (n, offset, self.stage_bytes))
        return bytes(self.buf[offset:offset + n])

    def to_device(self, dev_ptr, nbytes):
        mem, off = self._locate(dev_ptr.value)
        nbytes = int(nbytes)
        mem[off:off + nbytes] = self.buf[:nbytes]
        self.tally["h2d_calls"] += 1
        self.tally["h2d_bytes"] += nbytes

    def from_device(self, dev_ptr, nbytes):
        mem, off = self._locate(dev_ptr.value)
        nbytes = int(nbytes)
        self.buf[:nbytes] = mem[off:off + nbytes]
        self.tally["d2h_calls"] += 1
        self.tally["d2h_bytes"] += nbytes

    def read_back(self, dev_ptr, nbytes):
        self.from_device(dev_ptr, nbytes)
        data = self.view(0, int(nbytes))
        return data, hashlib.sha256(data).hexdigest(), len(data)

    def free_all(self):
        self.tally["frees"] = len(self.ptrs) + (1 if self.stage is not None else 0)
        self.ptrs = {}
        self.regions = []
        self.stage, self.stage_bytes = None, 0


# -----------------------------------------------------------------------------------------------
# 2. the substitute arm report
# -----------------------------------------------------------------------------------------------

def window_i_counters():
    """`{out_basename: {counter: value}}` read off window `i`'s own `ARM … path=device` lines.

    The rehearsal must put *some* non-zero launch count into a re-spelled device report, because
    `device_arm.run_arm` refuses a device run that cannot count a launch. The honest source for that number
    is the real device run's own line, so it is parsed from the window-`i` log in this repo rather than
    typed. A lane with no window-`i` line falls back to the shared-expert lane's counters and says so.
    """
    rows = {}
    for line in DEVLOG.read_text().splitlines():
        m = re.match(r"^ARM arm=(\S+) path=device (.*\bout=(\S+))\s*$", line)
        if not m:
            continue
        counters = dict((k, int(v)) for k, v in re.findall(r"\b(hip_calls|launches|allocs|"
                                                           r"bytes_to_device|bytes_from_device)=(\d+)",
                                                           m.group(2)))
        rows[Path(m.group(3)).name] = dict(counters, arm=m.group(1),
                                           source="window-i ARM arm=%s out=%s" % (m.group(1),
                                                                                   Path(m.group(3)).name))
    if not rows:
        raise SystemExit("window-i device ARM lines not found in %s; the rehearsal has no honest source "
                         "for a launch counter and refuses to invent one" % DEVLOG)
    return rows


def window_i_device_bytes():
    """Real device records this repo still holds, keyed by path under a read-only snapshot dir.

    THE HONEST STATE OF THIS LEG. The rehearsal's first child run pointed `--work` at the driver's default
    `engine/work/m2dev`, and because an arm asked for `--device` writes its `--out` there, it overwrote the
    window-`i` scratch device records (`out_device.bin`, `rt_out_device.bin`, `d_out_device.bin`,
    `gather_*/out_device.bin`) with host-body bytes. They are gitignored scratch (`*.bin`), not git-tracked, so
    they are gone; every number taken from them survives in
    `specification/evidence/blockdevice-device-2026-10-27-i1.log` and
    `specification/evidence/m2-device-pair-attribution-2026-10-27.md`, and a device window regenerates the
    bytes. The rehearsal now (a) runs against an isolated work dir, (b) reads records only from a snapshot dir
    outside the work dir, and (c) refuses any copy whose source is its destination. What survives elsewhere on
    this box — window `e2`'s router records — is used if present, and every line that uses it says which window
    it came from, because a record from `e2` cannot adjudicate `i`.
    """
    out = {}
    for snap in sorted(WORK.glob("*-device-records")):
        if not snap.is_dir():
            continue
        for p in sorted(snap.rglob("*device*.bin")):
            if p.is_file():
                out[str(p.relative_to(snap))] = p
    return out


class ArmShim(object):
    """`device_arm`'s subprocess door, re-aimed at the same binary's host body.

    A `--device` request on an `_arm` binary runs that binary without the flag — its own host body, the same
    compiled translation unit the CPU witness compares as `hip_hostpath` — and the report's `path=host` is
    re-spelled `path=device` with the window-`i` launch counters. A `--device` request on a `_host` binary is
    passed through untouched: `control_gather_device_refusal` and `control_d_device_ask_refuses` exist to
    prove that body refuses the flag by name, and substituting that would delete the only honest device claim
    those controls make.
    """

    def __init__(self, counters, saved):
        self.counters, self.saved = counters, saved
        self.se_fallback = counters.get("out_device.bin") or sorted(counters.values())[0]

    def key_of(self, out_path):
        """The saved-record key for an `--out` path: its path under `engine/work/m2dev`, not its basename."""
        p = Path(out_path).resolve()
        root = (WORK / "m2dev").resolve()
        try:
            return str(p.relative_to(root))
        except ValueError:
            return p.name

    def run(self, argv, **kw):
        argv = [str(a) for a in argv]
        binary = Path(argv[0])
        want_device = "--device" in argv
        if not want_device or "_host" in binary.name:
            return SUBPROCESS.run(argv, **kw)
        rest = [a for a in argv[1:] if a != "--device"]
        out_path = rest[rest.index("--out") + 1] if "--out" in rest else None
        proc = SUBPROCESS.run([str(binary)] + rest, **kw)
        counters = self.counters.get(Path(out_path).name, self.se_fallback) if out_path else None
        if counters is None:
            raise SystemExit("no window-i launch counters for %s and no fallback lane" % out_path)
        respelled, rewrote = self.respell(proc.stdout, counters, binary.name, out_path)
        key = self.key_of(out_path) if out_path else None
        if key and key in self.saved:
            src = self.saved[key]
            if src.resolve() == Path(out_path).resolve():
                rewrote.append("record-bytes=REFUSED-self-copy")
            else:
                data = src.read_bytes()
                if len(data) == Path(out_path).stat().st_size:
                    Path(out_path).write_bytes(data)
                    rewrote.append("record-bytes=copied-from(%s,%d B,window=%s)"
                                   % (src.name, len(data), src.parent.name))
                else:
                    rewrote.append("record-bytes=REFUSED-size-mismatch(%d!=%d)"
                                   % (len(data), Path(out_path).stat().st_size))
        log("REHEARSAL-FABRICATION arm=%s out=%s key=%s path-field=device(counters %s) rewrites=%s "
            "note=the computation is the arm's own host body; the device act is substituted, so no number "
            "here is a device measurement and none is printed as one"
            % (binary.name, out_path, key or "none",
               ",".join("%s=%s" % (k, counters[k]) for k in ("hip_calls", "launches", "allocs",
                                                             "bytes_to_device", "bytes_from_device")),
               ";".join(rewrote) or "path only"))
        return SUBPROCESS.CompletedProcess([str(binary)] + rest, proc.returncode, respelled, proc.stderr)

    @staticmethod
    def respell(stdout, counters, arm, out_path):
        rewrote = []
        lines = []
        for line in stdout.splitlines():
            if line.startswith(("RUN ", "DREADBACK ", "READBACK ", "UNITREADBACK ", "PLAN ", "HDR ")):
                # The host body spells its path word `host_cpubody`, not `host`; replacing the short form first
                # produced `device_cpubody`, which is neither path word and made the driver's own
                # `path=device` bookkeeping read zero device legs. Longest spelling first, then the bare one.
                for src_word in ("path=host_cpubody", "path=host"):
                    n = line.count(src_word)
                    if n:
                        rewrote.append("%s->device x%d on %s" % (src_word, n, line.split(" ", 1)[0]))
                        line = line.replace(src_word, "path=device")
            if line.startswith("RUN "):
                line = re.sub(r"\bhip_calls=\d+", "hip_calls=%d" % counters["hip_calls"], line)
                line = re.sub(r"\blaunches=\d+", "launches=%d" % counters["launches"], line)
                line = re.sub(r"\ballocs=\d+", "allocs=%d" % counters["allocs"], line)
                line = re.sub(r"\bbytes_to_device=\d+",
                              "bytes_to_device=%d" % counters["bytes_to_device"], line)
                line = re.sub(r"\bbytes_from_device=\d+",
                              "bytes_from_device=%d" % counters["bytes_from_device"], line)
            lines.append(line)
        return "\n".join(lines) + "\n", rewrote


# -----------------------------------------------------------------------------------------------
# 3. the child: run the driver's device shape under a line tracer
# -----------------------------------------------------------------------------------------------

def assert_no_hip(tag):
    maps = open("/proc/self/maps").read()
    hits = sorted(set(h for h in HIP_LIBRARY_HINTS if h in maps))
    if hits:
        raise SystemExit("REHEARSAL-ABORT %s: this process has %s mapped; the rehearsal is not allowed to "
                         "reach a HIP runtime at all" % (tag, ",".join(hits)))
    log("REHEARSAL-CHECK arm=no-hip-library stage=%s maps_grep=0 libs=%s" % (tag, ",".join(HIP_LIBRARY_HINTS)))


def child_main(argv):
    sys.path.insert(0, str(ENGINE))
    import device_arm as DA
    import block_device as BD
    counters, saved = window_i_counters(), window_i_device_bytes()
    DA.CountedHip = StubHip
    DA.subprocess = ArmShim(counters, saved)
    log("REHEARSAL-STUB installed CountedHip=%s subprocess=%s device-records-available=%d lanes-with-counters=%s"
        % (DA.CountedHip.__name__, type(DA.subprocess).__name__, len(saved),
           ",".join(sorted(counters))))
    log("REHEARSAL-DEVICE-BYTES status=%s sources=%s note=without a real device record the rehearsal's "
        "device-vs-host pairs compare two host paths; their ndiff_bytes=0 is the absence of a second device "
        "record, not an agreement, and no GREPRO/DREPRO line is claimed from it"
        % ("present" if saved else "absent",
           ",".join(sorted("%s/%s" % (v.parent.name, v.name) for v in saved.values())) or "none"))
    assert_no_hip("before-run")

    traced = set()
    target = str(Path(BD.__file__).resolve())

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == target:
            traced.add(frame.f_lineno)
        return tracer

    if MUTATE == "launches-abc":
        real_run_arm = DA.run_arm

        def run_arm(*a, **kw):
            rec = real_run_arm(*a, **kw)
            if kw.get("device"):
                rec["run"]["launches"] = "abc"
                log("REHEARSAL-MUTATE arm=%s field=launches value=abc stage=injected-after-the-parse-site "
                    "note=this is the shape window `i`'s report had for EVERY field; the parse site is the "
                    "door now, so a consumer that casts would raise here instead of reddening a check"
                    % (kw.get("label") or "?"))
            return rec
        DA.run_arm = run_arm
    if MUTATE == "silent-control":
        real_control = BD.Verifier.control

        def control(self, name, fired, detail, legs=None, claims=None, reason=None, **kw):
            if name == MUTATE_CONTROL:
                fired, reason = False, "   "
                log("REHEARSAL-MUTATE control=%s field=reason value=whitespace-only fired=False "
                    "note=a quiet control whose reason is whitespace is a quiet control with no reason"
                    % name)
            # `**kw` forwards `not_exercised`/`declaim`, which `Verifier.control` grew when the three control
            # states became machine-readable; a wrapper that dropped them would make the rehearsal's own
            # mutation arm crash for a reason that has nothing to do with what it is testing.
            return real_control(self, name, fired, detail, legs=legs, claims=claims, reason=reason, **kw)
        BD.Verifier.control = control

    rc = None
    try:
        sys.settrace(tracer)
        rc = BD.main(["--device", "--work", str(WORK / "m2dev-rehearsal")])
    except Exception:
        log("REHEARSAL-TRACEBACK " + traceback.format_exc().replace("\n", "\n  ")[-4000:])
        rc = 99
    finally:
        sys.settrace(None)
    log("REHEARSAL-CHILD rc=%s mutate=%s traced-lines-in-block_device=%d"
        % (rc, MUTATE or "none", len(traced)))
    census = BD.device_census_derive()
    statements = census["statements"]
    exercised = quiet = 0
    for lineno in sorted(statements):
        st = statements[lineno]
        end = st["lineno"] + len(st["text"].splitlines()) - 1
        hit = any(l in traced for l in range(st["lineno"], end + 1))
        if hit:
            exercised += 1
            log("DEVONLY path=%s:%d status=exercised kind=%s guard=%s note=line hit under the tracer"
                % (st["func"], st["lineno"], st["kind"], st["guard"][:90]))
        else:
            quiet += 1
            log("DEVONLY path=%s:%d status=not-exercised kind=%s reason=guard-held-but-enclosing-condition"
                "-false(%s) note=reported unexercised, not counted as passed"
                % (st["func"], st["lineno"], st["kind"], st["guard"][:90]))
    log("DEVONLY census statements=%d exercised=%d not-exercised=%d traced=%d mutation=%s"
        % (len(statements), exercised, quiet, len(traced), MUTATE or "none"))
    assert_no_hip("after-run")
    return 0 if rc in (0, 1) else 2


# -----------------------------------------------------------------------------------------------
# 4. the parent: run the children and audit what they printed
# -----------------------------------------------------------------------------------------------

def child(extra, logfile):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child"] + extra
    with open(logfile, "w") as fh:
        proc = SUBPROCESS.run(cmd, stdout=fh, stderr=SUBPROCESS.STDOUT)
    return logfile, proc.returncode


def audit(lines, pattern):
    return [ln for ln in lines if re.search(pattern, ln)]


def parent_main():
    work = ENGINE / "work"
    findings = []
    shape_log, rc = child([], work / "p9-shape.log")
    lines = Path(shape_log).read_text().splitlines()
    result = audit(lines, r"^RESULT mode=device")
    checks = audit(lines, r"^RESULT .*checks=\d+")
    headline = {
        "device-bytes-stated": r"^REHEARSAL-DEVICE-BYTES status=(present|absent)",
        "stub-counted": r"^STUBHIP op=alloc_slab",
        "fabrication-labelled": r"^REHEARSAL-FABRICATION ",
        "devonly-census": r"^DEVONLY census statements=\d+ exercised=\d+ not-exercised=\d+",
        "no-hip-library": r"^REHEARSAL-CHECK arm=no-hip-library stage=after-run maps_grep=0",
    }
    # The reproduction legs are only claimable when the GDN and gather device records are on the box to compare
    # against; a router record from another window cannot adjudicate them.
    have_repro_basis = audit(lines, r"^REHEARSAL-DEVICE-BYTES status=present sources=\S*d_out_device\.bin")
    if have_repro_basis:
        headline["grepro-exact"] = r"^GREPRO .*vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION"
        headline["drepro-exact"] = (r"^DREPRO .*steps=-1 mask=0x80d200 heads=9,12,14,15,23 "
                                   r".*vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION")
        headline["drepro-residual"] = (r"^DREPRO .*steps=\+1 "
                                      r".*vs_device_record_ndiff_bytes=75206 verdict=RESIDUAL-LEFT")
    else:
        log("REHEARSAL arm=grepro-drepro status=not-claimed note=no real GDN/gather device record was "
            "available on this box, so the device-vs-host pairs in this rehearsal compare host paths; their "
            "ndiff_bytes=0 is the absence of a second record and NOT an agreement, and the reproduction legs "
            "are not claimed from here - a device window owns them")
    if audit(lines, r"^RESULT mode=device .*counters_unparsed=0"):
        log("REHEARSAL arm=counters-unparsed status=zero note=every arm record folded an integer into the "
            "witness totals")
    else:
        findings.append("counters-unparsed: a RESULT line reported counters the driver could not read")
    log("REHEARSAL arm=child-rc rc=%s log=%s" % (rc, Path(shape_log).name))
    if not result:
        findings.append("RESULT-line: the device-shaped run printed no RESULT line — defect 2 is unfixed")
    else:
        log("REHEARSAL arm=result-line %s" % result[-1][:400])
    for line in checks[-1:]:
        log("REHEARSAL arm=counters %s" % line[:400])
    for name, pat in headline.items():
        hits = audit(lines, pat)
        if hits:
            log("REHEARSAL arm=%s status=present count=%d first=%s" % (name, len(hits), hits[0][:200]))
        else:
            findings.append("%s: expected line %r absent from the rehearsal log" % (name, pat))
    for name, pat in (("launches-abc-red", r"^CHECK name=device-launch-counts-are-ints-and-positive "
                                           r"verdict=FAIL"),
                      ("silent-control-red", r"^CHECK name=silent-control-names-its-reason verdict=FAIL")):
        mut = "launches-abc" if name == "launches-abc-red" else "silent-control"
        mlog, mrc = child(["--mutate", mut], work / ("p9-mutate-%s.log" % mut))
        mlines = Path(mlog).read_text().splitlines()
        red = audit(mlines, pat)
        reached = audit(mlines, r"^RESULT mode=device")
        crash = audit(mlines, r"^REHEARSAL-TRACEBACK")
        named = audit(mlines, r"^RESULT mode=device .*counters_unparsed=[1-9]") if mut == "launches-abc" \
            else audit(mlines, r"^RED silent-control-names-its-reason")
        log("MUTATION arm=%s mutate=%s rc=%s target-red=%s result-reached=%s crash=%s named-on-result-line=%s "
            "log=%s"
            % (name, mut, mrc, bool(red), bool(reached), bool(crash), bool(named), Path(mlog).name))
        if not red:
            findings.append("%s: the fabrication did not redden its target check" % name)
        if not reached:
            findings.append("%s: the fabrication crashed or silenced the driver instead of reddening it"
                            % name)
        if not named:
            findings.append("%s: the fabrication was not named on the RESULT line" % name)
        if crash:
            findings.append("%s: uncaught exception in the driver (%s)" % (name, crash[0][:200]))
    census = audit(lines, r"^DEVONLY census ")
    if census:
        log("REHEARSAL arm=devonly-census %s" % census[-1][:300])
    else:
        findings.append("devonly-census: no census line printed")
    if audit(lines, r"^DEVONLY path=\S+ status=not-exercised"):
        log("REHEARSAL arm=devonly-named not-exercised legs carry reason= (lines above)")
    if findings:
        log("REHEARSAL verdict=FAIL findings=%d detail=%s" % (len(findings), " | ".join(findings)))
        return 1
    log("REHEARSAL verdict=PASS no device act occurred, no timing taken")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog=__file__, description=__doc__.splitlines()[0])
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mutate", default=None)
    a = ap.parse_args()
    MUTATE = a.mutate
    sys.exit(child_main(sys.argv[1:]) if a.child else parent_main())
