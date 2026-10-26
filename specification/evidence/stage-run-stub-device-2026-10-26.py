"""Structural test of the device branch against a STUB hip: no libamdhip64, no iGPU, no lock.

STUB SCOPE — read before quoting anything from this file's log. Every arm here runs against
Python dicts pretending to be device memory; no real `hipMalloc`/`hipMemcpy` was called and no
iGPU was touched. NOTHING witnessed here says the hardware path works; the real `--device` run
belongs to the main agent's clean window (AGENTS.md §GPU-execution-is-serialized). What is
established: the Python side of `stage_device` + the read-back digests + the device controls +
the denominator checks execute and fail-or-pass for the right reasons.

Arms, each printed with an `ARM` line:
  A  guard — stage_device(enabled=False) refuses and constructs nothing (pre-existing arm).
  B  faithful D2H — a bounded sample of runs staged and read back through the stub; per-run
     read-back digests exist and equal the two host digests (stub capability (a)).
  C  device-buffer corruption via the driver's OWN control `device-corrupt`: it writes one wrong
     byte INTO the device buffer with a deliberate 1-byte H2D (the mechanism a real
     `hipMemcpy` H2D provides too), the read-back digest comparison reddens, a clean re-stage
     matches again (stub capability (b)).
  D  short D2H via the driver's OWN control `device-short-d2h`: the read-back requests
     length-1 bytes and the returned-count rule reddens (stub capability (c)). A real
     hipMemcpy bulk copy either copies the whole request or errors — it has no short-without-
     error — so a short read-back can only ever mean a short REQUEST, against stub and hardware
     alike; the count rule is what makes that observable.
  E  row-verdict falsifiability: corrupting a device byte through the stub back door makes the
     driver's own `device_verdict` say STAGE-MISMATCH, a restore says STAGED, a short read-back
     says STAGE-MISMATCH.
  F  falsifiability of the denominator: with all sample rows' digests present
     `device_readback_count_eq_plan` PASSes; strip one row's device fields and the same check
     FAILs. (Checks that can only pass are not checks — pitfalls/40.)
  G  the whole plan through `stage_run.py main(--device --verify)` with the stub hip injected in
     a CHILD process: verbatim `RESULT ... mode=device` line, 464 READBACK records each carrying
     `readback_sha256=`, all 10 controls fired=RED. The child needs ~30 GiB of free RAM for the
     stub slabs and REFUSES below that.

All ten controls (the eight CPU ones and the two device ones) fire `fired=RED` against the stub
in device mode — that is what closes `controls=0 fired=0` from the window-c finding.
"""
import os
import re
import subprocess
import sys

REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
sys.path.insert(0, os.path.join(REPO, "engine"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import stage_run as SR  # noqa

MEM = {}          # device address -> bytearray (stub device memory)
HOST = {}         # host/pinned address -> bytearray
CALLS = []
NEXT = [0x1000_0000]
STUB_NOTE = ("arm witnessed against the STUB hip only; no real hip call was made and the iGPU "
             "was not touched")


def _new(n):
    addr = NEXT[0]
    NEXT[0] += max(n, 1) + 4096
    return addr


def _resolve(table, addr):
    """Which allocation contains `addr`, and the offset inside it — hipMemcpy is handed
    slab_base + slab_off, not a base address, which is the whole point of this test."""
    for base in sorted(table):
        if base <= addr < base + len(table[base]):
            return base, addr - base
    raise AssertionError("address %d is inside no allocation; the driver indexed a slab wrongly"
                         % addr)


def _as_int(x):
    return int(x.value) if hasattr(x, "value") else int(x)


class StubHip(SR.HipStaging):
    """Same method names and call shape the driver uses, backed by dicts of bytearrays.

    Only the memcpy backends are swapped: the driver still goes through `fill`, `to_device`,
    `from_device` and `view` — the same four calls the real `HipStaging` exposes — so no
    stub-only API is on any control's path. `corrupt_device` is a stub-side back door used ONLY
    by this harness's row-level verdict demo (arm E), never by a driver control; it exists
    because a Python dict has no other way to look like a mutated device byte."""

    def __init__(self):
        SR.DEVICE_EXERCISED["calls"] += 1
        self.fn = {}
        self.ptrs = {}
        self.stage = None
        self.stage_bytes = 0
        self.allocs = []
        for s in SR._HIP_SYMBOLS:
            self.fn[SR._HIP + s] = {"Malloc": self._malloc, "Free": self._free,
                                    "Memcpy": self._memcpy, "HostMalloc": self._hostmalloc,
                                    "HostFree": self._hostfree,
                                    "GetErrorString": lambda rc: b"stub",
                                    "DeviceSynchronize": lambda: 0}[s]

    def _malloc(self, pp, n):
        n = _as_int(n)
        addr = _new(n)
        MEM[addr] = bytearray(n)
        pp._obj.value = addr
        CALLS.append(("hipMalloc", n))
        return 0

    def _free(self, p):
        MEM.pop(int(p.value), None)
        CALLS.append(("hipFree", int(p.value)))
        return 0

    def _hostmalloc(self, pp, n, flags):
        n = _as_int(n)
        addr = _new(n)
        HOST[addr] = bytearray(n)
        pp._obj.value = addr
        CALLS.append(("hipHostMalloc", n))
        return 0

    def _hostfree(self, p):
        HOST.pop(int(p.value), None)
        return 0

    def _memcpy(self, dst, src_, n, kind):
        dst, src_, n = _as_int(dst), _as_int(src_), _as_int(n)
        if kind == SR.HIP_MEMCPY_HOST_TO_DEVICE:
            base, off = _resolve(MEM, dst)
            MEM[base][off:off + n] = bytes(HOST[src_][:n])
            CALLS.append(("H2D", dst, n))
        else:
            base, off = _resolve(MEM, src_)
            HOST[dst][:n] = bytes(MEM[base][off:off + n])   # faithful D2H (capability (a)):
            CALLS.append(("D2H", src_, n))                  # copies exactly `n`, any n
        return 0

    def fill(self, offset, data):
        n = len(data)
        if offset + n > self.stage_bytes:
            raise SR.StageError("staging-overflow", "stub")
        HOST[self.stage.value][offset:offset + n] = data

    def view(self, offset, n):
        if offset + n > self.stage_bytes:
            raise SR.StageError("staging-overread", "stub")
        return bytes(HOST[self.stage.value][offset:offset + n])

    def corrupt_device(self, dev_ptr, at, value):
        """Stub back door (capability (b)): write one byte INTO device memory at an offset.
        Harness-demo only — driver controls corrupt via a real 1-byte H2D instead."""
        base, off = _resolve(MEM, _as_int(dev_ptr))
        old = MEM[base][off + at]
        MEM[base][off + at] = value
        CALLS.append(("stub-corrupt", base, off + at))
        return old


def sample_idxs(stager):
    idxs = sorted(range(stager.plan_count), key=lambda i: stager.plan[i]["length"])[:10]
    idxs += [max(range(stager.plan_count), key=lambda i: stager.plan[i]["length"])]   # the draft
    idxs += [i for i in sorted(range(stager.plan_count), key=lambda i: -stager.plan[i]["length"])
             if stager.plan[i]["slab"] != "weights.draft"][:2]                         # the tables
    return sorted(set(idxs))


def run_parent():
    SR.HIP_LIBRARY = "/nonexistent/stub.so"          # the real library is never loaded here
    print("HARNESS stage_run stub device witness — %s" % STUB_NOTE)
    print("HARNESS driver=%s hip_library_claimed=%s (does not exist: %s)"
          % (os.path.relpath(SR.__file__, REPO), SR.HIP_LIBRARY,
             not os.path.exists(SR.HIP_LIBRARY)))
    L = SR.LO.Loader()
    st = SR.Stager(L)
    src = SR.ByteSource(L, why="stub device witness")
    idxs = sample_idxs(st)
    rows = st.verify(src, idxs=idxs)
    assert all(r["verdict"] == "VERIFIED" for r in rows), [r["reasons"] for r in rows]
    rc = 0

    # ---- arm A: the guard ---------------------------------------------------------------------
    before = SR.DEVICE_EXERCISED["calls"]
    try:
        SR.stage_device(L, st, rows, src, False)
        print("ARM A name=guard verdict=FAIL did not raise")
        rc = 1
    except SR.Refused as ex:
        ok = ex.tag == "device-flag-absent" and SR.DEVICE_EXERCISED["calls"] == before
        print("ARM A name=guard verdict=%s refused tag=%s device-calls unchanged at %d (%s)"
              % ("PASS" if ok else "FAIL", ex.tag, before, STUB_NOTE))
        rc |= 0 if ok else 1

    # ---- arm B: faithful D2H over a bounded sample --------------------------------------------
    hip = StubHip()
    hip, slabs, peak = SR.stage_device(L, st, rows, src, True, hip=hip)
    staged = [r for r in rows if r["staged"] == "STAGED"]
    refused = [r for r in rows if r["staged"] == "refused-not-wired"]
    n_ref = sum(1 for r in rows
                if L.lay.slab(r["slab"]).source == "gguf")
    n_sha = sum(1 for r in staged if len(r.get("dev_sha") or "") == 64)
    ok_b = len(staged) == n_ref and n_sha == n_ref and len(refused) == len(rows) - n_ref
    print("ARM B name=faithful-d2h verdict=%s staged=%d/%d readback_sha256_present=%d "
          "draft-refused=%d H2D=%d D2H=%d peak_staging=%d (%s)"
          % ("PASS" if ok_b else "FAIL", len(staged), n_ref, n_sha, len(refused),
             sum(1 for c in CALLS if c[0] == "H2D"), sum(1 for c in CALLS if c[0] == "D2H"),
             peak, STUB_NOTE))
    rc |= 0 if ok_b else 1

    # ---- arms C+D: the driver's own device controls, fired against the stub --------------------
    ctl_idxs, ctl_bytes = SR.choose_sample(st)
    p2, v2 = SR.Print(), SR.Verifier()
    SR.run_controls(p2, v2, L, st, src, ctl_idxs, None, mode="device", hip=hip, rows=rows)
    line = {ln.split(" ")[1]: ln for ln in p2.lines if ln.startswith("CONTROL name=")}
    for key in ("device-corrupt", "device-short-d2h"):
        hit = [ln for ln in p2.lines if ln.startswith("CONTROL name=%s " % key)]
        ok = bool(hit) and "fired=RED" in hit[0]
        print("ARM %s name=%s verdict=%s %s"
              % ("C" if key == "device-corrupt" else "D", key, "PASS" if ok else "FAIL",
                 STUB_NOTE))
        for ln in hit:
            print("CONTROL-VERBATIM %s" % ln)
        rc |= 0 if ok else 1
    cpu_fired = sum(1 for c in v2.controls if c[1] and c[0] not in
                    ("device-corrupt", "device-short-d2h"))
    print("ARM C+D name=controls-in-device-mode controls_declared=%d fired=%d "
          "(CPU controls %d/8 also fired in device mode) %s"
          % (len(v2.controls), sum(1 for c in v2.controls if c[1]), cpu_fired, STUB_NOTE))
    rc |= 0 if len(v2.controls) == 10 and all(c[1] for c in v2.controls) else 1
    for ln in p2.lines:
        if ln.startswith("CONTROL name="):
            print("CONTROL-LINE %s" % ln)

    # ---- arm E: row-level reddening through the driver's own verdict function -------------------
    t = next(r["idx"] for r in staged if r["length"] >= 16 and r["length"] <= (4 << 20))
    row = next(r for r in rows if r["idx"] == t)
    dev = SR._shift(hip.ptrs[row["slab"]], row["slab_off"])
    was0 = hip.view(0, 1)[0]                      # staging still holds the run's byte 0
    hip.corrupt_device(dev, 0, was0 ^ 0xFF)       # stub back door: one wrong byte IN device memory
    sha_c, back_c = SR.device_read_back(hip, row, dev)
    v_c = SR.device_verdict(dict(row, dev_sha=sha_c, dev_read=back_c))
    src2 = SR.load_windows(st, src, [t])
    SR.stage_run_to_device(hip, src2, row, src2.path_for(row["slab"]))
    sha_r, back_r = SR.device_read_back(hip, row, dev)
    v_r = SR.device_verdict(dict(row, dev_sha=sha_r, dev_read=back_r))
    sha_s, back_s = SR.device_read_back(hip, row, dev, row["length"] - 1)
    v_s = SR.device_verdict(dict(row, dev_sha=sha_s, dev_read=back_s))
    ok_e = v_c == "STAGE-MISMATCH" and v_r == "STAGED" and v_s == "STAGE-MISMATCH"
    print("ARM E name=verdict-falsifiable verdict=%s corrupted-digest->%s restored->%s "
          "short(%d/%d B)->%s; tags: short=%s digest=%s (%s)"
          % ("PASS" if ok_e else "FAIL", v_c, v_r, back_s, row["length"], v_s,
             SR.jdev([dict(row, reasons=["read-back-short:x"])])["readback-short"],
             SR.jdev([dict(row, reasons=["read-back-digest:x"])])["readback-digest"],
             STUB_NOTE))
    rc |= 0 if ok_e else 1
    hip.free_all()
    src.close()
    return rc, L, st


def run_parent_f_checks(L, st):
    """Arm F (in-parent part): the denominator check must be falsifiable, not decorative."""
    src = SR.ByteSource(L, why="stub denominator falsification")
    idxs = [i for i in sorted(range(st.plan_count), key=lambda i: st.plan[i]["length"])[:6]]
    rows = st.verify(src, idxs=idxs)
    for r in rows:
        if L.lay.slab(r["slab"]).source == "gguf":
            r["dev_sha"], r["dev_read"] = r["plan_sha"], r["length"]   # pretend clean read-backs
    n_ref = sum(1 for r in rows if L.lay.slab(r["slab"]).source == "gguf")
    out = {}
    for label, strip in (("all-digests", False), ("one-stripped", True)):
        rr = [dict(r) for r in rows]
        if strip:
            for r in rr:
                if L.lay.slab(r["slab"]).source == "gguf":
                    r.pop("dev_sha", None)
                    r["dev_read"] = 0
                    break
        pp, vv = SR.Print(), SR.Verifier()
        rd = SR.print_readbacks(pp, st, rr)
        cov = st.coverage(idxs, rr)
        cov_addr = st.coverage(list(range(st.plan_count)),
                               [dict(idx=i, verdict="VERIFIED")
                                for i in range(st.plan_count)])
        SR.add_checks(vv, L, st, pp, "device", idxs, rr, cov, cov_addr, src, True,
                      (idxs, sum(st.plan[i]["length"] for i in idxs)),
                      SR._order_log_lines(), SR._order_log_lines(), 0, rd)
        out[label] = [r for r in vv.rows
                      if r[0] in ("device_readback_count_eq_plan",
                                  "device_readback_digest_eq_file_digest")]
    ok = (out["all-digests"] and all(r[1] for r in out["all-digests"])
          and out["one-stripped"] and sum(1 for r in out["one-stripped"] if not r[1]) == 2)
    print("ARM F name=denominator-falsifiable verdict=%s all-digests:%s one-stripped:%s (%s)"
          % ("PASS" if ok else "FAIL",
             " ".join("%s=%s" % (r[0], "PASS" if r[1] else "FAIL") for r in out["all-digests"]),
             " ".join("%s=%s" % (r[0], "PASS" if r[1] else "FAIL")
                      for r in out["one-stripped"]),
             STUB_NOTE))
    src.close()
    return 0 if ok else 1


def mem_available_bytes():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024
    return 0


def run_full_plan_child(evidence):
    """Arm G: the driver's own main() over the whole 464-run plan, stub hip injected, so the
    verbatim mode=device RESULT line and the READBACK records come from the shipped code path."""
    SR.HIP_LIBRARY = "/nonexistent/stub.so"
    SR.HipStaging = StubHip
    return SR.main(["--device", "--verify", "--keep-evidence", evidence])


def verify_child_evidence(path):
    txt = open(path, errors="replace").read()
    n_read = len(re.findall(r"(?m)^READBACK idx=", txt))
    n_sha = len(re.findall(r"(?m)^READBACK .* readback_sha256=[0-9a-f]{64}", txt))
    n_window_c = len(re.findall(r"dev_sha256|device_sha256|readback_sha256|dev_digest", txt))
    n_ctl = len(re.findall(r"(?m)^CONTROL name=", txt))
    n_red = len(re.findall(r"(?m)^CONTROL name=\S+ fired=RED", txt))
    result = [ln for ln in txt.splitlines() if ln.startswith("RESULT ")]
    ok = (n_read == 464 and n_sha == 463 and n_window_c >= 463 and n_ctl == 10 and n_red == 10
          and result and "mode=device" in result[0] and "checks=0" not in result[0])
    print("ARM G name=evidence-audit verdict=%s readback_records=%d readback_sha256_hex=%d "
          "window-c-grep(readback_sha256 et al)=%d control_lines=%d fired=RED=%d result=%s"
          % ("PASS" if ok else "FAIL", n_read, n_sha, n_window_c, n_ctl, n_red,
             result[-1] if result else "(none)"))
    return 0 if ok else 1


def run():
    if len(sys.argv) >= 3 and sys.argv[1] == "--child-full-plan":
        return run_full_plan_child(sys.argv[2])
    rc, L, st = run_parent()
    rc |= run_parent_f_checks(L, st)
    if os.environ.get("SR_STUB_SKIP_CHILD"):
        print("ARM G name=full-plan-child verdict=SKIPPED (SR_STUB_SKIP_CHILD is set, so the "
              "full-plan child did not run; a complete witness run leaves it unset)")
        return rc
    avail = mem_available_bytes()
    child_ev = os.path.join(REPO, "specification", "evidence",
                            "stage-run-stub-device-fullplan-2026-10-26.log")
    if avail < 30 * (1 << 30):
        print("ARM G name=full-plan-child verdict=REFUSED mem_available_bytes=%d < %d; the stub "
              "slabs need ~20.5 GiB of dirty pages plus the reader's page cache (no partial child "
              "was started)" % (avail, 30 * (1 << 30)))
        return rc or 1
    print("ARM G name=full-plan-child start child=%s mem_available_bytes=%d (%s)"
          % (os.path.relpath(child_ev, REPO), avail, STUB_NOTE))
    rcp = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-full-plan",
                          child_ev], cwd=REPO).returncode
    print("ARM G name=full-plan-child child_exit=%d evidence=%s" % (rcp, child_ev))
    rc |= rcp
    rc |= verify_child_evidence(child_ev)
    print("HARNESS verdict=%s — %s; the REAL hip path is only exercised by the main agent's "
          "clean-window run of `engine/stage_run.py --device --verify`"
          % ("PASS" if rc == 0 else "FAIL", STUB_NOTE))
    return rc


raise SystemExit(run())
