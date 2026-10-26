"""Structural test of the device branch, against a STUB hip: no libamdhip64, no iGPU, no lock.

Not part of the driver's selftest (the task forbids the selftest invoking the device path); this is a
lane-internal check that the Python side of `stage_device` is executable at all, since the first real
device run must not be the first time the branch is entered.
"""
import ctypes
import os
import sys

REPO = "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b"
sys.path.insert(0, os.path.join(REPO, "engine"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import stage_run as SR  # noqa

MEM = {}          # device address -> bytearray
CALLS = []
NEXT = [0x1000_0000]
ALLOC_FAIL_AT = None


class FakeLib(object):
    def __getattr__(self, name):
        raise AttributeError(name)


def _new(n):
    addr = NEXT[0]
    NEXT[0] += max(n, 1) + 4096
    MEM[addr] = bytearray(n)
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


def fake_malloc(pp, n):
    n = _as_int(n)
    if ALLOC_FAIL_AT is not None and n >= ALLOC_FAIL_AT:
        return 2                      # hipErrorOutOfMemory
    addr = _new(n)
    pp._obj.value = addr
    CALLS.append(("hipMalloc", n))
    return 0


def fake_free(p):
    MEM.pop(int(p.value), None)
    CALLS.append(("hipFree", int(p.value)))
    return 0


def fake_h2d(dst, src, n, kind):
    dst, src, n = int(dst.value), int(src.value), int(n)
    if kind != SR.HIP_MEMCPY_HOST_TO_DEVICE:
        return 7
    if dst not in MEM or dst + n > max(MEM) + 10 ** 9:
        return 7                      # invalid device address
    MEM[dst][:n] = HOST[src:src + n]
    CALLS.append(("H2D", dst, n))
    return 0


def fake_d2h(dst, src, n, kind):
    dst, src, n = int(dst.value), int(src.value), int(n)
    if kind != SR.HIP_MEMCPY_DEVICE_TO_HOST:
        return 7
    if src not in MEM:
        return 7
    HOST[dst:dst + n] = MEM[src][:n]
    CALLS.append(("D2H", src, n))
    return 0


def fake_hostmalloc(pp, n, flags):
    n = _as_int(n)
    addr = _new(n)
    global HOST
    HOST[addr] = bytearray(n)
    pp._obj.value = addr
    CALLS.append(("hipHostMalloc", n))
    return 0


HOST = {}


def install(monkey_host=True):
    SR.ctypes.CDLL = lambda path: FakeLib()

    def sym(self, name):
        return {"hipMalloc": fake_malloc, "hipFree": fake_free,
                "hipMemcpy": lambda *a: 0, "hipHostMalloc": fake_hostmalloc,
                "hipHostFree": lambda p: 0, "hipGetErrorString": lambda rc: b"stub",
                "hipDeviceSynchronize": lambda: 0}[name]
    return sym


def run():
    SR.HIP_LIBRARY = "/nonexistent/stub.so"
    L = SR.LO.Loader()
    st = SR.Stager(L)
    src = SR.ByteSource(L, why="stub device test")
    # only the small runs, so the stub memory stays small
    idxs = sorted(range(st.plan_count), key=lambda i: st.plan[i]["length"])[:10]
    idxs += [max(range(st.plan_count), key=lambda i: st.plan[i]["length"])]          # the draft
    idxs += [i for i in sorted(range(st.plan_count), key=lambda i: -st.plan[i]["length"])
             if st.plan[i]["slab"] != "weights.draft"][:2]                           # the tables
    idxs = sorted(set(idxs))
    rows = st.verify(src, idxs=idxs)
    assert all(r["verdict"] == "VERIFIED" for r in rows), [r["reasons"] for r in rows]
    # --- guard first: enabled=False must raise and must not construct anything
    before = SR.DEVICE_EXERCISED["calls"]
    try:
        SR.stage_device(L, st, rows, src, False)
        print("FAIL guard did not raise")
        return 1
    except SR.Refused as ex:
        assert ex.tag == "device-flag-absent", ex.tag
        assert SR.DEVICE_EXERCISED["calls"] == before
        print("guard refused tag=%s, device calls unchanged at %d" % (ex.tag, before))
    # --- now the branch itself, against the stub
    class StubHip(SR.HipStaging):
        def __init__(self):
            SR.DEVICE_EXERCISED["calls"] += 1
            self.fn = {}
            self.ptrs = {}
            self.stage = None
            self.stage_bytes = 0
            self.allocs = []
            for s in SR._HIP_SYMBOLS:
                self.fn[SR._HIP + s] = {"Malloc": fake_malloc, "Free": fake_free,
                                        "Memcpy": self._memcpy, "HostMalloc": self._hostmalloc,
                                        "HostFree": self._hostfree,
                                        "GetErrorString": lambda rc: b"stub",
                                        "DeviceSynchronize": lambda: 0}[s]

        def _memcpy(self, dst, src_, n, kind):
            dst, src_, n = _as_int(dst), _as_int(src_), _as_int(n)
            if kind == SR.HIP_MEMCPY_HOST_TO_DEVICE:
                base, off = _resolve(MEM, dst)
                MEM[base][off:off + n] = HOST[src_][:n]
                CALLS.append(("H2D", dst, n))
            else:
                base, off = _resolve(MEM, src_)
                HOST[dst][:n] = MEM[base][off:off + n]
                CALLS.append(("D2H", src_, n))
            return 0

        def _hostmalloc(self, pp, n, flags):
            addr = _new(_as_int(n))
            HOST[addr] = bytearray(_as_int(n))
            pp._obj.value = addr
            CALLS.append(("hipHostMalloc", _as_int(n)))
            return 0

        def _hostfree(self, p):
            HOST.pop(int(p.value), None)
            return 0

        def fill(self, offset, data):
            n = len(data)
            if offset + n > self.stage_bytes:
                raise SR.StageError("staging-overflow", "stub")
            HOST[self.stage.value][offset:offset + n] = data
            return None

        def view(self, offset, n):
            if offset + n > self.stage_bytes:
                raise SR.StageError("staging-overread", "stub")
            return bytes(HOST[self.stage.value][offset:offset + n])

    hip, slabs, peak = SR.stage_device(L, st, rows, src, True, hip=StubHip())
    print("slabs=%s peak=%d" % (slabs, peak))
    staged = [r for r in rows if r["staged"] == "STAGED"]
    print("staged=%d of %d rows; draft rows refused=%d"
          % (len(staged), len(rows), sum(1 for r in rows if r["staged"] == "refused-not-wired")))
    bad = [r["idx"] for r in rows if r["staged"] not in ("STAGED", "refused-not-wired")]
    print("not-staged-and-not-refused: %s" % (bad or "none"))
    print("device calls=%d; H2D=%d D2H=%d malloc=%d"
          % (SR.DEVICE_EXERCISED["calls"],
             sum(1 for c in CALLS if c[0] == "H2D"), sum(1 for c in CALLS if c[0] == "D2H"),
             sum(1 for c in CALLS if c[0] == "hipMalloc")))
    # slab offsets really were used: every H2D destination is one staged row's slab base + slab_off
    bases = {name: int(p.value) for name, p in hip.ptrs.items()}
    expected = {bases[r["slab"]] + r["slab_off"] for r in rows if r["staged"] == "STAGED"}
    for c in [c for c in CALLS if c[0] == "H2D"]:
        if c[1] not in expected:
            print("FAIL H2D to %x is not slab_base+slab_off of any staged row" % c[1])
            return 1
    print("every H2D destination == slab_base + slab_off of a staged row: True "
          "(%d copies, %d distinct addresses)" % (sum(1 for c in CALLS if c[0] == "H2D"),
                                                  len(expected)))
    hip.free_all()
    print("after free_all: device allocations left=%d pinned=%d" % (len(MEM) - len(HOST), len(HOST)))
    src.close()
    return 0


raise SystemExit(run())
