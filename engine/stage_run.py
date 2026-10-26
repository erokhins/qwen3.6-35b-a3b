#!/usr/bin/env python3
"""stage_run.py — EXECUTE policy A's staging plan and certify it byte for byte (M1's rung).

`loader.py` plans the staging (`stage_runs()`: 464 bulk-copy runs) and `engine/README.md` §Not
established is explicit that the plan "has never been executed". This file is the execution and the
verification. It adds no second ruler: the plan comes from `loader.Loader.stage_runs` (i.e.
`layout.ResidencyLayout.copy_plan`) verbatim, the tensor coordinates come from
`layout.ResidencyLayout.placements`, and the *reference* byte regions come from `loader`'s own
re-parse (`Loader.ruler`/`Loader.draft`, priced by `tools/gguf_width_table.py`'s block table while
the plan's come from `tools/ggml_block_geom.py`'s). Two block tables, one plan, one file.

What a run must do to be recorded VERIFIED — all four, or it is recorded with the reason it failed:

1. **Plan digest.** the planned byte range `[file_off, file_off+length)` of its source file is read
   once, in chunks, through the audit reader, and hashed (sha256) as it is read.
2. **Header digest.** the same bytes, hashed again by walking the *member tensors'* own records —
   `record.file_offset .. +record.row` for each tensor the run covers, one read per tensor. A run
   whose merged range and whose per-tensor ranges disagree by even one byte hashes differently, and
   nothing here assumes the two coincide: `run-file-ranges-consecutive` proves the coincidence.
3. **Span and membership.** `sum(member.row) == run["length"]`, `|members| == run["n"]`, and every
   member lies inside exactly one run of its slab. All derived from `layout`'s placements, never
   re-derived from the merge logic that produced the runs.
4. **Bound.** `[file_off, file_end)` lies inside the file the run reads from, by that file's actual
   `st_size` — the check llama.cpp makes in `src/llama-model-loader.h::llama_tensor_weight`.

The device path is a third digest, not a fourth ruler: read the run into one pinned host staging
buffer, one bulk H2D copy into its slab, one bulk D2H read-back into the same buffer, hash what comes
back, and a run is STAGED only when the read-back hash equals the two host hashes it was staged from.

llama.cpp is the shape reference (AGENTS.md §GOAL, D-19). Read before writing this file:
`engines/llama.cpp/src/llama-model-loader.cpp::llama_model_loader::init_mappings` (the denominator
`size_data` is summed from the header's tensor sizes, never from what was read), `::load_all_data`
(one pass; per tensor one contiguous `seek`+`read_raw` into one staging buffer sized like the tensor,
then one bulk `ggml_backend_tensor_set`, then `ggml_validate_row_data`), `::load_data_range`
(seek+read of the whole row, then validate), and `merge_ranges`/`sub_ranges` (byte ranges merged into
a sorted non-overlapping list — the same object this file calls a run and uses for its coverage
union). Adopted: bulk copy per planned run, offsets and sizes taken from the header, one pass, no
per-tensor re-open of the file, and a completeness total derived from the header. Divergences, each
with its reason where it is implemented: llama.cpp verifies *structurally* (`ggml_validate_row_data`)
and trusts `read_raw` plus `ggml_backend_tensor_set`; this file verifies *by digest*, because M1's
claim is "staged on device and read back byte-exact", which a structural check cannot see. And
llama.cpp's non-mmap pass loads biggest-first to bound the staging buffer's peak; this file keeps the
plan's order (ascending file offsets per slab, which is what `layout._runs` guarantees) because the
staging buffer is allocated once at the largest staged run's size, so the peak does not depend on
order. The brief's `ggml/src/ggml-llama.cpp` does not exist in `engines/llama.cpp`; the paths above
are the ones that do.

Modes are mutually exclusive and an unknown flag is refused rather than ignored:

    --plan-only     print the plan, the slab table and the coverage bookkeeping. Reads NO data byte.
    --verify        the CPU path: all 464 runs, both host digests.
    --selftest      the CPU-only controls. Injections land on an in-memory copy of a bounded window
                    set, never on the artifact; the device path is never invoked here.
    --check-only    --verify plus --selftest, then one RESULT line. Refuses --device outright and
                    goes red if the device path was exercised in this process.
    --device        CPU path plus the staging/read-back copy. Without this flag the device path
                    refuses by name, so a copy can never happen by accident.
    --fault NAME    run one control by name (what --selftest runs, one control at a time)

CPU-only by default: no HIP import, no device call, no lock of any kind (this file owns none — a
driver that self-acquires a lock and is called by a holder deadlocks silently,
`pitfalls/50-harness-and-shell.md` §Self-locking-gate-drivers). Reads of the data section go through
`tools/audit_published_quant.py`'s own selftest gate, earned once per process, so the ordering log
gets one line per process that reads tensor bytes and the count is reported, not assumed. No timing
is taken or printed anywhere in this file: counts and bytes only.
"""
import ctypes
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
for _p in (TOOLS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ggml_block_geom as GBG              # noqa: E402  the width the wrong-bpw control mis-prices
import layout as LAY                       # noqa: E402  the slabs, the runs, the placements
import loader as LO                        # noqa: E402  the plan, the ruler records, the refusals

MiB = 2.0 ** 20
GiB = 2.0 ** 30
CHUNK = 4 << 20                 # bytes per pread into the hash; 4 MiB, not a tuned number
SELFTEST_CAP = 32 << 20         # the selftest digests at most this many bytes per side
HIP_LIBRARY = "/opt/rocm/lib/libamdhip64.so"
ORDER_LOG_FALLBACK = os.path.join(REPO, "quantization", "evidence",
                                  "audit-ordering-2026-10-26.log")
# The device-copy symbols, spelled as separate literals so this file's own no-lock/no-device-import
# scan (CHECK copy-path-owns-no-lock) can be honest about a file that names the API it calls.
_HIP = "hip"
_HIP_SYMBOLS = ("Malloc", "Free", "Memcpy", "HostMalloc", "HostFree", "GetErrorString",
                "DeviceSynchronize")
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


class StageError(Exception):
    """The staging run refused. `tag` names the refusal so a gate can require a *specific* one."""

    def __init__(self, tag, msg):
        super().__init__("[%s] %s" % (tag, msg))
        self.tag = tag


class Refused(StageError):
    """A caller asked for a path this file will not take without being told to, explicitly."""


# The device-path witness. Anything that touches the iGPU must bump this, and --check-only goes red
# when it is non-zero: the CPU mode's whole claim is that no device call happened in this process.
DEVICE_EXERCISED = {"calls": 0, "notes": []}
# One token grant per process is the audit reader's own rule; the ordering log grows by exactly this
# many lines and CHECK ordering-log-append-witnessed holds the two together.
TOKEN_GRANTS = {"n": 0}


# -----------------------------------------------------------------------------------------------
# 1. byte sources — the plan's reads, and an in-memory copy for the controls
# -----------------------------------------------------------------------------------------------
class ByteSource(object):
    """Sequential-range reader over the plan's two source files.

    One open per file per process, chunked reads, and the GGUF's data bytes reachable only through
    the audit reader's own token (`engine/README.md` §Not established: that token is the engine
    layer's gate on tensor-data reads, and reusing it here is deliberate — a second gate would be a
    second ruler that can go stale on its own schedule).
    """

    def __init__(self, L, why="staging verify"):
        self.L = L
        self.why = why
        self.reads = 0
        self.bytes_read = 0
        self.copy_bytes = 0                 # bytes pulled into control copies
        self.data_read = False
        self.token = None
        self._plain = {}                    # path -> file handle, for non-GGUF sources only
        self._mm = None                     # the audit reader, opened on first data read

    def path_for(self, slab_name):
        """Which file a run reads from, from the slab's own `source` field."""
        return self.L.art.path if self.L.lay.slab(slab_name).source == "gguf" else LO.RB.MTP_PATH

    def size_of(self, slab_name):
        return self.L.size if self.L.lay.slab(slab_name).source == "gguf" else self.L.draft_size

    def _earn_token(self):
        if self.token is not None:
            return self.token
        import audit_published_quant as APQ

        class _Section:
            section = "engine/stage_run.py staging execution"

        mark = APQ.require_selftest(_Section())
        if mark is None:
            raise Refused(
                "data-read-not-granted",
                "%s refused the selftest token, so not one tensor-data byte is read. This file uses "
                "the audit reader's own gate rather than a second one (engine/README.md \u00a7Not "
                "established)" % APQ.SELFTEST_MARK)
        self.token = APQ.SELFTEST_TOKEN
        TOKEN_GRANTS["n"] += 1
        self.grant_utc = mark.get("utc_latest_selftest_pass")
        return self.token

    def raw_read(self, path, off, n, lane="plan"):
        """Read `n` bytes at absolute offset `off` of `path`. May return fewer: a short read is a
        fact the caller must see, never a retry loop that hides it.

        `lane` is "plan" for the run's own range and "header" for the per-tensor walk; the GGUF does
        not care which lane asked, and the controls do (see `MemSource.flip`)."""
        self.reads += 1
        if path == self.L.art.path:
            tok = self._earn_token()
            self.data_read = True
            if self._mm is None:
                self._mm = self.L.art._hdr
            # the reader's own token-checked read, not a raw mmap peek: gguf_reader.Gguf.read is the
            # only door its data section has, and this file does not build a second one
            got = self._mm.read(tok, off, n, why=self.why)
        else:
            fh = self._plain.get(path)
            if fh is None:
                fh = self._plain[path] = open(path, "rb")
            fh.seek(off)
            got = fh.read(n)
        self.bytes_read += len(got)
        return got

    def close(self):
        for fh in self._plain.values():
            try:
                fh.close()
            except OSError:
                pass
        self._plain = {}
        if self._mm is not None:
            self._mm.close()
            self._mm = None


class MemSource(object):
    """An in-memory copy of the byte windows the controls need. The controls mutate THIS.

    `pitfalls/50` §A-selftest-that-writes-fixtures-into-the-directory-a-real-run-reads-from is the
    reason this exists: every injection lands on a copy of bytes already read, never on the artifact,
    and never on the loader's records. The bytes are loaded once through `ByteSource`, so the control
    still judges real file bytes — it just judges a copy of them.
    """

    def __init__(self, base, windows):
        self.base = base
        self.reads = 0
        self.bytes_read = 0
        self.data_read = False
        self.blocks = {}                    # (path, off) -> (end, bytearray)
        for path, off, length in windows:
            key = (path, off)
            if key in self.blocks and self.blocks[key][0] >= off + length:
                continue                    # a wider window already covers this one
            got = bytearray(base.raw_read(path, off, length))
            if len(got) != length:
                raise StageError("window-short",
                                 "loading the control window %s+%d,%d returned %d B; a control whose "
                                 "fixture is short cannot judge anything" % (path, off, length,
                                                                             len(got)))
            self.blocks[key] = (off + length, got)
            self.bytes_read += len(got)
        if hasattr(base, "copy_bytes"):
            base.copy_bytes += self.bytes_read
        self.token = getattr(base, "token", None)
        self.data_read = getattr(base, "data_read", False)
        self.short = set()                  # (path, off, n) -> return one byte fewer
        self.corrupt = {}                   # (lane, path, off) -> byte, for one-lane corruption
        self.flips_applied = 0

    def path_for(self, slab_name):
        return self.base.path_for(slab_name)

    def size_of(self, slab_name):
        return self.base.size_of(slab_name)

    def _cover(self, path, off, n):
        for (bpath, boff), (end, buf) in self.blocks.items():
            if bpath == path and boff <= off and off + n <= end:
                return buf, off - boff
        return None, None

    def raw_read(self, path, off, n, lane="plan"):
        """Serve from the copy. A queued short read is honoured once, exactly as the file would.

        `lane` matters here and nowhere else: the two lanes read the same file through the same
        source, so a corruption the controls plant must be plantable in ONE of them, or the digest
        comparison would see two identical wrong answers and call it verified."""
        self.reads += 1
        buf, at = self._cover(path, off, n)
        if buf is None:
            raise StageError("window-miss",
                             "control source was asked for %s+%d,%d, which no loaded window covers; "
                             "the control's run set and its window set disagree" % (path, off, n))
        out = bytes(buf[at:at + n])
        for (l, pth, o), nb in self.corrupt.items():
            if l not in (lane, "*") or pth != path:
                continue
            i = o - off
            if 0 <= i < len(out) and out[i] != nb:
                out = out[:i] + bytes([nb]) + out[i + 1:]
        self.bytes_read += len(out)
        if (path, off, n) in self.short:
            out = out[:-1]
            self.bytes_read -= 1
        return out

    def flip(self, path, off, lane="*"):
        """Corrupt ONE byte of the in-memory copy, in one lane or both. The artifact is never written
        to; this is the only mutation any control performs, and it is reported so the control can name
        what it changed."""
        buf, at = self._cover(path, off, 1)
        if buf is None:
            raise StageError("window-miss",
                             "cannot flip %s+%d: no loaded window covers it, so the control would "
                             "silently corrupt nothing and still print its verdict" % (path, off))
        before = buf[at]
        if lane == "*":
            buf[at] = before ^ 0xFF
            after = buf[at]
        else:
            self.corrupt[(lane, path, off)] = before ^ 0xFF
            after = self.corrupt[(lane, path, off)]
        self.flips_applied += 1
        return before, after

    def close(self):
        self.blocks = {}


# -----------------------------------------------------------------------------------------------
# 2. the device path — one pinned staging buffer, one bulk copy per run, one bulk read-back
# -----------------------------------------------------------------------------------------------
class HipStaging(object):
    """hipMalloc per slab + hipHostMalloc staging buffer + one bulk copy per run.

    `pitfalls/10-device-memory.md` §Allocation-recipe is the recipe this follows: `hipMalloc` and an
    explicit `hipMemcpy`, never `Managed` (32 GiB under `Managed` measured 51.6 GB/s with no error at
    all, while `hipMalloc` fails loudly), one allocation per slab rather than per tensor, and a single
    bulk copy per planned run. The buffer is pinned once at the largest staged run's size, which is
    why this file copies in the plan's own order instead of llama.cpp's biggest-first
    (`src/llama-model-loader.cpp::load_all_data` sorts biggest-first only to bound that buffer).

    No device call happens unless an instance of this class is constructed, and constructing it is
    the only thing that bumps DEVICE_EXERCISED.
    """

    def __init__(self, chunk=CHUNK):
        DEVICE_EXERCISED["calls"] += 1
        DEVICE_EXERCISED["notes"].append("HipStaging.__init__")
        if not os.path.exists(HIP_LIBRARY):
            raise StageError("device-library-absent",
                             "%s is not on this box; the device path cannot copy a single byte"
                             % HIP_LIBRARY)
        try:
            lib = ctypes.CDLL(HIP_LIBRARY)
        except OSError as ex:
            raise StageError("device-library-unloadable",
                             "dlopen(%s) failed: %s" % (HIP_LIBRARY, ex))
        self.lib = lib
        self.fn = {}
        for s in _HIP_SYMBOLS:
            name = _HIP + s
            try:
                self.fn[name] = getattr(lib, name)
            except AttributeError:
                raise StageError("device-symbol-absent",
                                 "%s has no %s(); the device path refuses rather than guessing a "
                                 "signature" % (os.path.basename(HIP_LIBRARY), name))
        self.fn["hipMalloc"].argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.fn["hipMalloc"].restype = ctypes.c_int
        self.fn["hipFree"].argtypes = [ctypes.c_void_p]
        self.fn["hipMemcpy"].argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_int]
        self.fn["hipMemcpy"].restype = ctypes.c_int
        self.fn["hipHostMalloc"].argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                             ctypes.c_uint]
        self.fn["hipHostMalloc"].restype = ctypes.c_int
        self.fn["hipHostFree"].argtypes = [ctypes.c_void_p]
        self.fn["hipGetErrorString"].argtypes = [ctypes.c_int]
        self.fn["hipDeviceSynchronize"].restype = ctypes.c_int
        self.ptrs = {}
        self.stage = None
        self.stage_bytes = 0
        self.allocs = []

    def _chk(self, rc, what):
        if rc != 0:
            why = self.fn["hipGetErrorString"](rc)
            why = why.decode() if isinstance(why, bytes) else str(why)
            raise StageError("device-copy-failed", "%s returned hip error %d (%s); failing loudly, "
                                                   "because a half-staged slab that still prints a "
                                                   "digest is the bad number this file exists to "
                                                   "prevent" % (what, rc, why))

    @staticmethod
    def _proc(tag):
        """Bytes from /proc and the amdgpu sysfs around an allocation. Bytes only, never a rate:
        pitfalls/10 §"print MemAvailable and the aperture around every allocation attempt"."""
        out = {}
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemAvailable:"):
                    out["mem_available_bytes"] = int(ln.split()[1]) * 1024
                    break
        except OSError:
            pass
        best = None
        for path in _gtt_paths():
            try:
                best = max(best or 0, int(open(path).read().strip()))
            except (OSError, ValueError):
                continue
        if best is not None:
            out["gtt_used_bytes"] = best
        out["probe"] = tag
        return out

    def alloc_slab(self, name, nbytes):
        p = ctypes.c_void_p()
        rc = self.fn["hipMalloc"](ctypes.byref(p), ctypes.c_size_t(nbytes))
        rec = dict(slab=name, bytes=nbytes, rc=rc, before=self._proc("before hipMalloc"),
                   after=self._proc("after hipMalloc"))
        self.allocs.append(rec)
        self._chk(rc, "hipMalloc(%s, %d B)" % (name, nbytes))
        self.ptrs[name] = p
        return p

    def alloc_stage(self, nbytes):
        p = ctypes.c_void_p()
        rc = self.fn["hipHostMalloc"](ctypes.byref(p), ctypes.c_size_t(nbytes), 0)
        self._chk(rc, "hipHostMalloc(%d B staging)" % nbytes)
        self.stage, self.stage_bytes = p, nbytes
        return p

    def fill(self, offset, data):
        n = len(data)
        if offset + n > self.stage_bytes:
            raise StageError("staging-overflow",
                             "wrote %d B at +%d into a %d B staging buffer" % (n, offset,
                                                                               self.stage_bytes))
        ctypes.memmove(ctypes.c_void_p(self.stage.value + offset), data, ctypes.c_size_t(n))

    def to_device(self, dev_ptr, nbytes):
        self._chk(self.fn["hipMemcpy"](dev_ptr, self.stage, ctypes.c_size_t(nbytes),
                                       HIP_MEMCPY_HOST_TO_DEVICE), "hipMemcpy H2D")
        self._chk(self.fn["hipDeviceSynchronize"](), "hipDeviceSynchronize after H2D")

    def from_device(self, dev_ptr, nbytes):
        self._chk(self.fn["hipMemcpy"](self.stage, dev_ptr, ctypes.c_size_t(nbytes),
                                       HIP_MEMCPY_DEVICE_TO_HOST), "hipMemcpy D2H")
        self._chk(self.fn["hipDeviceSynchronize"](), "hipDeviceSynchronize after D2H")

    def view(self, offset, n):
        if offset + n > self.stage_bytes:
            raise StageError("staging-overread",
                             "read %d B at +%d from a %d B staging buffer" % (n, offset,
                                                                              self.stage_bytes))
        return ctypes.string_at(self.stage.value + offset, n)

    def free_all(self):
        for name, p in list(self.ptrs.items()):
            self._chk(self.fn["hipFree"](p), "hipFree(%s)" % name)
            del self.ptrs[name]
        if self.stage is not None:
            self.fn["hipHostFree"](self.stage)
            self.stage, self.stage_bytes = None, 0


def _gtt_paths():
    import glob
    return sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"))


# -----------------------------------------------------------------------------------------------
# 3. interval arithmetic — the same two helpers llama.cpp uses for its mapping ranges
# -----------------------------------------------------------------------------------------------
def merge_ranges(ranges):
    """Sort and merge `[first, last)` intervals. Port of
    `engines/llama.cpp/src/llama-model-loader.cpp::merge_ranges`, which exists there to keep a
    tensor's mapped byte ranges non-overlapping; here it keeps *verified runs* honest."""
    out = []
    for first, last in sorted(ranges):
        if last <= first:
            raise StageError("empty-interval",
                             "[%d,%d) is not an interval; an empty run cannot cover anything and a "
                             "counter over it cannot fail" % (first, last))
        if out and first <= out[-1][1]:
            out[-1][1] = max(out[-1][1], last)
        else:
            out.append([first, last])
    return [(a, b) for a, b in out]


def intersect_len(a, b):
    """|A \u2229 B| for two already-merged interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def overlap_len(merged, first, last):
    return sum(min(last, hi) - max(first, lo) for lo, hi in merged if min(last, hi) > max(first, lo))


# -----------------------------------------------------------------------------------------------
# 4. the stager
# -----------------------------------------------------------------------------------------------
class Stager(object):
    """Executes the loader's plan and judges it. Nothing here plans: `self.plan` IS
    `loader.Loader.stage_runs()` and the tensor coordinates are `layout`'s placements."""

    def __init__(self, L, out=None):
        self.L = L
        self.out = out if out is not None else []
        self.plan = L.stage_runs()
        self.plan_count = len(self.plan)
        self.plan_bytes = sum(r["length"] for r in self.plan)
        self.by_slab = {}
        for i, r in enumerate(self.plan):
            self.by_slab.setdefault(r["slab"], []).append((i, r))
        self.members = self._resolve_members()
        self.rows = []
        self.device_rows = {}

    # ---- member resolution, by coordinate, never by re-running the merge -------------------
    def _resolve_members(self):
        """run index -> the placements whose slab byte range lies inside it.

        Membership is *asked of the coordinates*, not recomputed from the merge logic that cut the
        runs: if this file re-derived runs the same way `layout._runs` does, a bug in that merge
        would be invisible to it (pitfalls/40 \u00a7a-check-whose-reference-side-is-derived-from-the-
        thing-under-test). The count is then compared against the `n` layout itself recorded, which
        is a second, independent statement about the same run."""
        members = {i: [] for i in range(self.plan_count)}
        stray = []
        for p in self.L.lay.placements:
            hits = [i for i, r in self.by_slab.get(p.slab, [])
                    if r["slab_off"] <= p.slab_off and p.slab_off + p.rec.row <= r["slab_end"]]
            if len(hits) != 1:
                stray.append("%s covered by %d runs of %s" % (p.rec.name, len(hits), p.slab))
                continue
            members[hits[0]].append(p)
        if stray:
            raise StageError("placement-coverage",
                             "%d placement(s) are not inside exactly one run: %s" % (len(stray),
                                                                                      stray[:4]))
        return members

    def ref_regions(self, idx, faults=None):
        """The member tensors' byte ranges as the loader's OWN re-parse states them.

        `self.L.by_name` is priced by `tools/gguf_width_table.py`'s block table while the plan and the
        placements are priced by `tools/ggml_block_geom.py`'s, so this is the second ruler \u2014 and
        `faults['reprice']` is how the selftest shows the ruler is load-bearing."""
        reprice = (faults or {}).get("reprice") or {}
        out = []
        for p in self.members[idx]:
            rec = self.L.by_name.get(p.rec.name)
            if rec is None:
                raise StageError("ref-ruler-missing",
                                 "%s is in the layout but not in the loader's re-parse; the reference "
                                 "side has lost a tensor and cannot adjudicate the plan" % p.rec.name)
            row = int(rec.row)
            wrong = reprice.get(rec.name)
            if wrong is not None:
                row = GBG.row_bytes(rec.n_elts, wrong)
            out.append(dict(name=rec.name, off=int(rec.file_offset), row=row, row_ruler=row,
                            row_record=int(p.rec.row), type_name=rec.type_name,
                            repriced=wrong))
        return out

    # ---- digests --------------------------------------------------------------------------
    def digest_range(self, source, path, off, n, lane="plan"):
        h = hashlib.sha256()
        got_total = 0
        pos = 0
        while pos < n:
            want = min(CHUNK, n - pos)
            got = source.raw_read(path, off + pos, want, lane=lane)
            h.update(got)
            got_total += len(got)
            if len(got) != want:
                break
            pos += len(got)
        return h.hexdigest(), got_total

    def digest_ranges(self, source, path, regions, lane="header"):
        """One hash over several ranges, in order — the header-side read.

        The digests are of the BYTES, not of per-tensor digests: chaining member digests would still
        print two hex strings and would then be equal for any partition of the same bytes while being
        unequal for identical bytes read in a different order, which is backwards for what this
        compares (the two addressing paths, plan coordinates vs tensor records)."""
        h = hashlib.sha256()
        total = 0
        for off, n in regions:
            pos = 0
            while pos < n:
                want = min(CHUNK, n - pos)
                got = source.raw_read(path, off + pos, want, lane=lane)
                h.update(got)
                total += len(got)
                if len(got) != want:
                    break
                pos += len(got)
        return h.hexdigest(), total

    def verify_run(self, source, idx, faults=None):
        faults = faults or {}
        r = self.plan[idx]
        path = source.path_for(r["slab"])
        fsize = source.size_of(r["slab"])
        shift = faults["shift"][1] if faults.get("shift") and faults["shift"][0] == idx else 0
        drop = idx in set(faults.get("drop") or ())
        regs = self.ref_regions(idx, faults)
        row = dict(idx=idx, slab=r["slab"], slab_off=r["slab_off"], file_off=r["file_off"],
                   read_off=r["file_off"] + shift, length=r["length"], n=r["n"],
                   slab_end=r["slab_end"],
                   members=len(regs), plan_sha="-", ref_sha="-", dev_sha="-", read=0, ref_read=0,
                   verdict="DROPPED" if drop else "PENDING", reasons=[], staged="not-attempted")
        if drop:
            return row
        # --- structural, before a single byte is hashed
        if not (0 <= row["read_off"] and row["read_off"] + row["length"] <= fsize):
            row["reasons"].append("bound:past-eof(read_off=%d+length=%d in a %d B file)"
                                  % (row["read_off"], row["length"], fsize))
        if sum(m["row_record"] for m in regs) != row["length"]:
            row["reasons"].append("span:records %d B != run %d B"
                                  % (sum(m["row_record"] for m in regs), row["length"]))
        if sum(m["row"] for m in regs) != row["length"]:
            row["reasons"].append("span:ruler %d B != run %d B"
                                  % (sum(m["row"] for m in regs), row["length"]))
        if len(regs) != row["n"]:
            row["reasons"].append("membership:%d placements in a run that says n=%d"
                                  % (len(regs), row["n"]))
        pos = r["file_off"]
        for m in regs:
            if m["off"] != pos:
                row["reasons"].append("consecutive:%s starts at %d, previous range ended at %d"
                                      % (m["name"], m["off"], pos))
                break
            pos += m["row"]
        for m in regs:
            if m["row"] != m["row_record"]:
                row["reasons"].append("two-rulers:%s is %d B by gguf_width_table and %d B by "
                                      "ggml_block_geom" % (m["name"], m["row"], m["row_record"]))
                break
        # --- the two digests
        try:
            row["plan_sha"], row["read"] = self.digest_range(source, path, row["read_off"],
                                                             row["length"])
            row["ref_sha"], row["ref_read"] = self.digest_ranges(
                source, path, [(m["off"], m["row"]) for m in regs])
        except StageError as ex:
            row["reasons"].append("read:%s" % ex.tag)
        if row["read"] != row["length"]:
            row["reasons"].append("short-read:planned %d B, read %d B" % (row["length"], row["read"]))
        want_ref = sum(m["row"] for m in regs)
        if row["ref_read"] != want_ref:
            row["reasons"].append("short-read-ref:expected %d B, read %d B" % (want_ref,
                                                                               row["ref_read"]))
        if row["plan_sha"] != row["ref_sha"]:
            row["reasons"].append("digest:plan-side %s.. != header-side %s.."
                                  % (row["plan_sha"][:12], row["ref_sha"][:12]))
        if not row["reasons"]:
            row["verdict"] = "VERIFIED"
        else:
            row["verdict"] = "NOT-VERIFIED"
        return row

    def verify(self, source, idxs=None, faults=None):
        idxs = range(self.plan_count) if idxs is None else list(idxs)
        rows = [self.verify_run(source, i, faults) for i in idxs]
        self.last_rows = rows
        return rows

    # ---- the completeness invariant --------------------------------------------------------
    def coverage(self, idxs, rows):
        """residue / uncovered bytes over a run set, derived from the plan and the placements.

        residue  = bytes the verified runs cover that are no admitted tensor's bytes (double-counted
                   or stray), plus any byte counted twice;
        uncovered= bytes of admitted tensors in THIS run set that no verified run covers.
        Both are set arithmetic over `layout`'s coordinates; neither is typed, and neither is
        computed from the merge that made the runs. The tensor side is the placements of the runs in
        `idxs`, so a bounded selftest sample and the whole 464-run plan are judged the same way."""
        rows = [r for r in rows if r["verdict"] == "VERIFIED"]
        keep = {r["idx"] for r in rows}
        per_slab_runs, per_slab_recs = {}, {}
        for i in idxs:
            if i in keep:
                r = self.plan[i]
                per_slab_runs.setdefault(r["slab"], []).append((r["slab_off"], r["slab_end"]))
            for pl in self.members[i]:
                per_slab_recs.setdefault(pl.slab, []).append((pl.slab_off,
                                                              pl.slab_off + pl.rec.row))
        residue = uncovered = counted = planned = 0
        for slab in sorted(set(per_slab_runs) | set(per_slab_recs)):
            runs = merge_ranges(per_slab_runs.get(slab, []))
            recs = merge_ranges(per_slab_recs.get(slab, []))
            raw = sum(e - s for s, e in per_slab_runs.get(slab, []))
            union = sum(e - s for s, e in runs)
            planned += raw
            counted += union
            residue += raw - union                              # bytes counted more than once
            residue += union - intersect_len(runs, recs)         # bytes no tensor owns
            for first, last in per_slab_recs.get(slab, []):
                uncovered += (last - first) - overlap_len(runs, first, last)
        return dict(runs=len(rows), run_set=len(idxs), planned_bytes=planned, union_bytes=counted,
                    residue=residue, uncovered=uncovered)


# -----------------------------------------------------------------------------------------------
# 5. the device path — guarded, and never reached from the selftest
# -----------------------------------------------------------------------------------------------
def stage_device(L, stager, rows, source, enabled, hip=None):
    """Copy each VERIFIED run into its slab and read it back, then hash what came back.

    The guard is a parameter, not a CLI check, so an in-process caller that forgot the flag is
    refused by the same sentence the command line is. `enabled` False raises; it does not quietly
    return an empty result, because a device mode that silently did nothing is how a null A/B reads
    like a pass (pitfalls/50 §A-flag-whose-premise-does-not-hold).
    """
    if not enabled:
        raise Refused("device-flag-absent",
                      "the device path copies bytes into a slab and reads them back; that is a device "
                      "run and it happens only when --device is on the command line. Nothing here "
                      "takes the iGPU on its own initiative (AGENTS.md §GPU-execution-is-serialized)")
    staged = [r for r in rows if r["verdict"] == "VERIFIED" and r["slab"] != "weights.draft"]
    refused = [r for r in rows if r["verdict"] == "VERIFIED" and r["slab"] == "weights.draft"]
    if not staged:
        raise StageError("device-nothing-to-stage",
                         "%d verified runs and none of them is staged; a device mode that copies "
                         "nothing has measured nothing" % len(rows))
    hip = hip or HipStaging()
    slabs = []
    for s in LAY.SLAB_ORDER:
        if any(r["slab"] == s for r in staged) and s not in slabs:
            slabs.append(s)
    for s in slabs:
        hip.alloc_slab(s, L.lay.slab(s).bytes)
    peak = max(r["length"] for r in staged)
    hip.alloc_stage(peak)
    for r in staged:
        path = source.path_for(r["slab"])
        pos = 0
        while pos < r["length"]:
            want = min(CHUNK, r["length"] - pos)
            got = source.raw_read(path, r["read_off"] + pos, want)
            if len(got) != want:
                raise StageError("stage-short-read",
                                 "run %d (%s) asked for %d B at +%d and got %d B; a partial staging "
                                 "buffer would still copy and still hash" % (r["idx"], r["slab"],
                                                                             want, pos, len(got)))
            hip.fill(pos, got)
            pos += want
        dev = _shift(hip.ptrs[r["slab"]], r["slab_off"])
        hip.to_device(dev, r["length"])
        hip.from_device(dev, r["length"])
        h = hashlib.sha256()
        pos = 0
        while pos < r["length"]:
            want = min(CHUNK, r["length"] - pos)
            h.update(hip.view(pos, want))
            pos += want
        r["dev_sha"] = h.hexdigest()
        r["staged"] = "STAGED" if r["dev_sha"] == r["plan_sha"] == r["ref_sha"] else "STAGE-MISMATCH"
        if r["staged"] != "STAGED":
            r["reasons"].append("read-back:device digest %s.. != host %s../%s.."
                                % (r["dev_sha"][:12], r["plan_sha"][:12], r["ref_sha"][:12]))
    for r in refused:
        # The draft block is bf16 on disk and policy A prices it at Q4_K/Q8_0; loader.stage_draft
        # refuses to hand those bytes back under a quantized label, and this file obeys that refusal
        # rather than staging a slab of bf16 that the gate would then certify as if it were Q4_K.
        names = [m["name"] for m in stager.ref_regions(r["idx"])]
        try:
            L.stage_draft(names[0])
            r["staged"] = "REFUSED-LEAKED"          # the loader handed bytes back: that is a finding
            r["reasons"].append("draft-refusal:stage_draft returned bytes for %s" % names[0])
        except LO.NotWired as ex:
            r["staged"] = "refused-not-wired"
            r["reasons"].append("draft:%s" % ex.tag)
    return hip, slabs, peak


def _shift(dev_ptr, delta):
    """`dev_ptr + delta` on a raw address, never on a typed pointer (pitfalls/40 §geometry-that-grows:
    a scaled offset at one slot is identical to the correct expression and wrong at the second)."""
    return ctypes.c_void_p(dev_ptr.value + delta)


# -----------------------------------------------------------------------------------------------
# 6. judgements — one verdict per case, computed once, reused by the controls
# -----------------------------------------------------------------------------------------------
# Which rule a run's reason string belongs to. The control that must fire a named rule asks for that
# name, so "something went red" is never enough (pitfalls/40 §a-negative-control-with-an-empty-
# must-catch-set): each of these is a rule the CPU path also enforces on the real bytes.
JUDGEMENTS = ("bound", "span-records", "span-ruler", "membership", "consecutive", "two-rulers",
              "short-read", "short-read-ref", "digest", "read-error")
_REASON_TAG = (("short-read-ref", "short-read-ref"), ("short-read", "short-read"),
               ("span:records", "span-records"), ("span:ruler", "span-ruler"),
               ("two-rulers", "two-rulers"), ("membership", "membership"),
               ("consecutive", "consecutive"), ("bound", "bound"), ("digest", "digest"),
               ("read:", "read-error"))


def _tag_of(reason):
    for pre, tag in _REASON_TAG:
        if reason.startswith(pre):
            return tag
    return None


def _has_tag(row, tag):
    return any(_tag_of(x) == tag for x in row["reasons"])


def judge(rows):
    """{rule: ok} over a set of run rows, read off the reasons the rows themselves carry."""
    red = set()
    for r in rows:
        for reason in r["reasons"]:
            tag = _tag_of(reason)
            if tag:
                red.add(tag)
    return {name: name not in red for name in JUDGEMENTS}


class Verifier(object):
    """Named checks with denominators, and controls that are required to turn something red.

    `check` never takes a literal predicate (see CHECK self-source-has-no-constant-predicates) and
    `control` counts as a finding when it does NOT fire: a control that cannot fail is the same dead
    instrument as a parse that prints nothing (pitfalls/40 §a-check-that-can-produce-no-output)."""

    def __init__(self):
        self.rows = []
        self.controls = []
        self.notes = []

    def check(self, name, ok, n, detail):
        self.rows.append((name, bool(ok), int(n), detail))
        return bool(ok)

    def control(self, name, fired, injected, detail):
        self.controls.append((name, bool(fired), injected, detail))
        return bool(fired)

    def note(self, name, detail):
        self.notes.append((name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]] + [c for c in self.controls if not c[1]]


# -----------------------------------------------------------------------------------------------
# 7. printing — every line a counter, and every counter re-parsed from the text it was printed in
# -----------------------------------------------------------------------------------------------
RUN_RE = re.compile(r"^RUN idx=(\d+) slab=(\S+) slab_off=(\d+) ", re.M)
TENSOR_RE = re.compile(r"^TENSOR name=(\S+) slab=(\S+) run=(\d+) ", re.M)
SLAB_RE = re.compile(r"^SLAB name=(\S+) source=(\S+) ", re.M)
CONTROL_RE = re.compile(r"^CONTROL name=(\S+) fired=(RED|SILENT) ", re.M)
RATE = re.compile(r"(\btok/s\b|\bGB/s\b|\bMB/s\b|[0-9](\.[0-9]+)?\s*ms\b|elapsed|duration)", re.I)


class Print(object):
    """Collects every printed line, so a count can be checked against the text rather than against a
    variable that was incremented while printing."""

    def __init__(self):
        self.lines = []

    def say(self, line):
        self.lines.append(line)
        print(line)

    def text(self):
        return "\n".join(self.lines)


def print_slabs(p, L, stager, idxs):
    keep = set(idxs)
    for s in LAY.SLAB_ORDER:
        sl = L.lay.slab(s)
        rs = [(i, r) for i, r in stager.by_slab.get(s, [])]
        here = [(i, r) for i, r in rs if i in keep]
        p.say("SLAB name=%s source=%s runs=%d runs_in_set=%d slab_bytes=%d content=%d pad=%d "
              "set_bytes=%d staged=%s"
              % (s, sl.source, len(rs), len(here), sl.bytes, sl.content_bytes, sl.pad_bytes,
                 sum(r["length"] for _, r in here),
                 "yes" if L.lay.slab(s).source == "gguf" else "no:draft-block"))
    p.say("TRAILER slabs_planned=%d slabs_printed=%d"
          % (len(LAY.SLAB_ORDER), len(SLAB_RE.findall(p.text()))))


def print_rows(p, stager, idxs, rows):
    for r in rows:
        p.say("RUN idx=%d slab=%s slab_off=%d file_off=%d read_off=%d length=%d n=%d members=%d "
              "read=%d ref_read=%d plan=%s ref=%s dev=%s verdict=%s staged=%s reasons=%s"
              % (r["idx"], r["slab"], r["slab_off"], r["file_off"], r["read_off"], r["length"],
                 r["n"], r["members"], r["read"], r["ref_read"], r["plan_sha"], r["ref_sha"],
                 r["dev_sha"], r["verdict"], r["staged"],
                 ";".join(r["reasons"]) or "-"))
    printed = len(RUN_RE.findall(p.text()))
    p.say("TRAILER runs_in_set=%d runs_printed=%d verified=%d not_verified=%d dropped=%d "
          "bytes_verified=%d"
          % (len(idxs), printed, sum(1 for r in rows if r["verdict"] == "VERIFIED"),
             sum(1 for r in rows if r["verdict"] == "NOT-VERIFIED"),
             sum(1 for r in rows if r["verdict"] == "DROPPED"),
             sum(r["length"] for r in rows if r["verdict"] == "VERIFIED")))


def print_tensors(p, stager, L, idxs, rows):
    """One line per admitted tensor of the run set, with how many verified runs cover it.

    This is the completeness invariant at tensor granularity: `covered_runs=1` and
    `covered_bytes=row` for every one of them is what residue 0 / uncovered 0 mean in a form a reader
    can grep, and the population is the placements of the runs in the set — derived, never typed."""
    keep = {r["idx"] for r in rows if r["verdict"] == "VERIFIED"}
    per_slab = {}
    for r in rows:
        if r["verdict"] == "VERIFIED":
            per_slab.setdefault(r["slab"], []).append((r["slab_off"], r["slab_end"]))
    merged = {s: merge_ranges(v) for s, v in per_slab.items()}
    shown = 0
    for i in idxs:
        for pl in stager.members[i]:
            ref = L.by_name[pl.rec.name]
            cov_bytes = overlap_len(merged.get(pl.slab, []), pl.slab_off,
                                    pl.slab_off + pl.rec.row)
            p.say("TENSOR name=%s slab=%s run=%d slab_off=%d row=%d ref_row=%d covered_runs=%d "
                  "covered_bytes=%d/%d type=%s"
                  % (pl.rec.name, pl.slab, i, pl.slab_off, pl.rec.row, ref.row,
                     1 if i in keep else 0, cov_bytes, pl.rec.row, pl.rec.type_name))
            shown += 1
    parsed = len(TENSOR_RE.findall(p.text()))
    p.say("TRAILER tensors_planned=%d tensors_printed=%d tensors_parsed=%d"
          % (shown, shown, parsed))
    return shown, parsed


# -----------------------------------------------------------------------------------------------
# 8. the selftest's sample, and the windows it needs
# -----------------------------------------------------------------------------------------------
def choose_sample(stager, cap=SELFTEST_CAP):
    """A derived, bounded sample of runs — not a hand-list, and it says which rule picked it.

    Three runs are required by rule so the paths that matter are never optional: the largest-`n`
    run that fits the cap (the only shape that exercises the multi-tensor reference walk), the
    smallest, and the largest that fits. The rest is smallest-first until the byte cap, which is what
    keeps a CPU selftest out of the artifact's 22,134,528,992 B."""
    n = stager.plan_count
    fits = [i for i in range(n) if stager.plan[i]["length"] <= cap]
    if not fits:
        raise StageError("sample-empty", "no run is smaller than the selftest cap %d B" % cap)
    required = [min(fits, key=lambda i: (stager.plan[i]["length"], i)),
                max(fits, key=lambda i: (stager.plan[i]["n"], -stager.plan[i]["length"], i)),
                max(fits, key=lambda i: (stager.plan[i]["length"], i))]
    chosen, total = [], 0
    for i in sorted(set(required), key=lambda i: (stager.plan[i]["length"], i)):
        chosen.append(i)
        total += stager.plan[i]["length"]
    for i in sorted(range(n), key=lambda i: (stager.plan[i]["length"], stager.plan[i]["slab"],
                                             stager.plan[i]["slab_off"])):
        if i in chosen:
            continue
        if total + stager.plan[i]["length"] > cap:
            continue
        chosen.append(i)
        total += stager.plan[i]["length"]
    return sorted(chosen), total


def load_windows(stager, source, idxs, shift_idx=None, margin=4096):
    """Load the sample's bytes into a copy the controls may corrupt.

    The plan-side window is loaded at its planned range plus `margin` when a control is going to read
    it from the wrong offset, so the wrong-offset control reads real neighbouring bytes rather than
    tripping the window guard for the wrong reason."""
    windows = []
    for i in idxs:
        r = stager.plan[i]
        path = source.path_for(r["slab"])
        extra = margin if i == shift_idx else 0
        limit = source.size_of(r["slab"])
        length = min(r["length"] + extra, limit - r["file_off"])
        windows.append((path, r["file_off"], length))
        for m in stager.ref_regions(i):
            wl = min(m["row"], limit - m["off"])
            if (path, m["off"], wl) not in windows:
                windows.append((path, m["off"], wl))
    return MemSource(source, windows)


# -----------------------------------------------------------------------------------------------
# 9. the controls — each one injects a specific fault and must turn a NAMED rule red
# -----------------------------------------------------------------------------------------------
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
WRONG_WIDTHS = ("Q4_K", "Q8_0", "F32", "BF16", "Q5_K", "Q6_K")


def _pick(stager, idxs, pred, why):
    """Pick a run from the sample by a predicate, smallest first, or refuse out loud.

    A control whose target does not exist must not print `fired=RED` over an empty case list
    (pitfalls/40 §a-negative-control-with-an-empty-must-catch-set), so no target is a hard finding."""
    cands = [i for i in sorted(idxs, key=lambda i: (stager.plan[i]["length"], i)) if pred(i)]
    if not cands:
        raise StageError("control-target", "no sampled run satisfies %s (%s)" % (why, sorted(idxs)))
    return cands[0]


def c_short_read(ctx):
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    t = _pick(stager, idxs, lambda i: stager.plan[i]["length"] >= 2, "length >= 2")
    src = load_windows(stager, source, [t])
    r = stager.plan[t]
    path = source.path_for(r["slab"])
    src.short.add((path, r["file_off"], min(CHUNK, r["length"])))
    row = stager.verify(src, idxs=[t])[0]
    red = judge([row])
    fired = (not red["short-read"]) and row["verdict"] == "NOT-VERIFIED" \
        and row["read"] == row["length"] - 1
    return fired, "run %d (%s) served %d of %d B" % (t, r["slab"], row["read"], row["length"]), \
        "read=%d/%d verdict=%s short-read-rule=%s reasons=%s" % (
            row["read"], row["length"], row["verdict"], red["short-read"],
            ";".join(row["reasons"]) or "-")


def c_wrong_offset(ctx):
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    t = _pick(stager, idxs,
              lambda i: stager.plan[i]["file_off"] + stager.plan[i]["length"] + 2
              <= source.size_of(stager.plan[i]["slab"]),
              "a run with two bytes of room after it")
    src = load_windows(stager, source, [t], shift_idx=t)
    row = stager.verify(src, idxs=[t], faults={"shift": (t, 1)})[0]
    red = judge([row])
    fired = (not red["digest"]) and row["verdict"] == "NOT-VERIFIED"
    r = stager.plan[t]
    return fired, "run %d (%s) read from file_off %d+1 instead of %d" % (
        t, r["slab"], r["file_off"], r["file_off"]), \
        "plan=%s.. ref=%s.. digest-rule=%s verdict=%s reasons=%s" % (
            row["plan_sha"][:12], row["ref_sha"][:12], red["digest"], row["verdict"],
            ";".join(row["reasons"]) or "-")


def c_wrong_width(ctx):
    """Price one member tensor at the wrong bpw on the RULER side only.

    The plan is left alone, which is the point: if the fault moved both sides together the check would
    be self-consistent and blind (pitfalls/40 §two-self-consistent-layouts). What must go red is the
    span rule, the two-rulers rule and the digest, because the header says one width and the plan was
    cut on another."""
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    target = None
    for i in sorted(idxs, key=lambda i: (stager.plan[i]["length"], i)):
        for m in stager.ref_regions(i):
            rec = stager.L.by_name[m["name"]]
            for wrong in WRONG_WIDTHS:
                if wrong == rec.type_name:
                    continue
                nb = GBG.row_bytes(rec.n_elts, wrong)
                if 0 < nb < m["row"]:
                    target = (i, m["name"], rec.type_name, wrong, m["row"], nb)
                    break
            if target:
                break
        if target:
            break
    if target is None:
        raise StageError("control-target",
                         "no sampled tensor can be repriced narrower, so the wrong-width control has "
                         "nothing to inject and must not print a verdict")
    i, name, good, wrong, good_row, bad_row = target
    src = load_windows(stager, source, [i])
    row = stager.verify(src, idxs=[i], faults={"reprice": {name: wrong}})[0]
    red = judge([row])
    fired = (not red["span-ruler"]) and (not red["two-rulers"]) and (not red["digest"])
    return fired, "%s priced %s (%d B) as %s (%d B) on the ruler side while the plan kept %s" % (
        name, good, good_row, wrong, bad_row, good), \
        "span-ruler=%s two-rulers=%s digest=%s verdict=%s reasons=%s" % (
            red["span-ruler"], red["two-rulers"], red["digest"], row["verdict"],
            ";".join(row["reasons"]) or "-")


def c_skipped_run(ctx):
    """Drop one run of 464 and require the uncovered counter to move by exactly that run's size."""
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    t = _pick(stager, idxs, lambda i: True, "any sampled run")
    src = load_windows(stager, source, idxs)
    base = stager.verify(src, idxs=idxs)
    cov0 = stager.coverage(idxs, base)
    rows = stager.verify(src, idxs=idxs, faults={"drop": [t]})
    cov1 = stager.coverage(idxs, rows)
    delta = cov1["uncovered"] - cov0["uncovered"]
    fired = (cov0["uncovered"] == 0 and delta == stager.plan[t]["length"]
             and cov1["uncovered"] != 0
             and sum(1 for r in rows if r["verdict"] == "DROPPED") == 1)
    return fired, "run %d (%s, %d B) dropped from a %d-run set" % (
        t, stager.plan[t]["slab"], stager.plan[t]["length"], len(idxs)), \
        "uncovered %d -> %d (delta %d, run length %d) dropped_rows=%d" % (
            cov0["uncovered"], cov1["uncovered"], delta, stager.plan[t]["length"],
            sum(1 for r in rows if r["verdict"] == "DROPPED"))


def c_corrupted_byte(ctx):
    """Corrupt one byte of the copy in the PLAN lane only.

    A corruption both lanes see is invisible to a digest comparison by construction — both sides hash
    the same wrong bytes. Planting it in one lane is what makes this a control: it proves the second
    read is a real second read, and that a divergence between the two addressing paths cannot be
    certified. If anyone "optimises" the header walk into reusing the plan buffer, this goes SILENT."""
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    t = _pick(stager, idxs, lambda i: stager.plan[i]["length"] >= 1, "length >= 1")
    src = load_windows(stager, source, [t])
    r = stager.plan[t]
    path = source.path_for(r["slab"])
    before, after = src.flip(path, r["file_off"], lane="plan")
    row = stager.verify(src, idxs=[t])[0]
    red = judge([row])
    fired = (not red["digest"]) and src.flips_applied == 1 and row["verdict"] == "NOT-VERIFIED"
    return fired, "one byte of the in-memory copy flipped in the plan lane at %s+%d: 0x%02X -> 0x%02X" % (
        os.path.basename(path), r["file_off"], before, after), \
        "plan=%s.. ref=%s.. digest-rule=%s verdict=%s flips=%d reasons=%s" % (
            row["plan_sha"][:12], row["ref_sha"][:12], red["digest"], row["verdict"],
            src.flips_applied, ";".join(row["reasons"]) or "-")


def c_duplicated_run(ctx):
    """Witness the RESIDUE counter live: the same run counted twice must move it by exactly its size.

    Every other control leaves residue at 0, so without this one `residue=0` would be a counter that
    has never been seen to be anything else."""
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    t = _pick(stager, idxs, lambda i: True, "any sampled run")
    src = load_windows(stager, source, idxs)
    rows = stager.verify(src, idxs=idxs)
    cov0 = stager.coverage(idxs, rows)
    dup = [r for r in rows if r["idx"] == t][0]
    idxs2 = list(idxs) + [t]
    rows2 = list(rows) + [dup]
    cov1 = stager.coverage(idxs2, rows2)
    delta = cov1["residue"] - cov0["residue"]
    fired = cov0["residue"] == 0 and delta == stager.plan[t]["length"] and cov1["residue"] != 0
    return fired, "run %d (%s, %d B) counted twice in the %d-run set" % (
        t, stager.plan[t]["slab"], stager.plan[t]["length"], len(idxs)), \
        "residue %d -> %d (delta %d, run length %d)" % (
            cov0["residue"], cov1["residue"], delta, stager.plan[t]["length"])


def c_window_guard(ctx):
    """A control whose fixture can silently be absent is no control: ask for bytes no window covers."""
    stager, idxs, source = ctx["stager"], ctx["idxs"], ctx["source"]
    src = load_windows(stager, source, idxs)
    path = source.path_for(stager.plan[idxs[0]]["slab"])
    size = source.size_of(stager.plan[idxs[0]]["slab"])
    off = size - 8
    while src._cover(path, off, 8) != (None, None) and off > 0:
        off -= 1
        if off < size - (1 << 22):
            raise StageError("window-guard", "no uncovered offset found within 4 MiB of EOF; the "
                                             "sample is too wide for this control to be meaningful")
    try:
        src.raw_read(path, off, 8)
    except StageError as ex:
        return ex.tag == "window-miss", "read of 8 B at %s+%d, outside every loaded window" % (
            os.path.basename(path), off), "refused with tag=%s" % ex.tag
    return False, "read of 8 B at %s+%d" % (os.path.basename(path), off), \
        "the control source SERVED bytes no window covers, so every control would judge real bytes " \
        "or nothing and print the same verdict either way"


def c_device_guard(ctx):
    """The device path must refuse when the flag is absent, and must not have touched anything."""
    stager, idxs, source, L = ctx["stager"], ctx["idxs"], ctx["source"], ctx["L"]
    rows = stager.verify(load_windows(stager, source, idxs), idxs=idxs)
    before = DEVICE_EXERCISED["calls"]
    try:
        stage_device(L, stager, rows, source, False)
    except Refused as ex:
        return ex.tag == "device-flag-absent" and DEVICE_EXERCISED["calls"] == before, \
            "stage_device(enabled=False) on %d verified rows" % len(rows), \
            "refused with tag=%s; device calls before=%d after=%d" % (
                ex.tag, before, DEVICE_EXERCISED["calls"])
    except StageError as ex:
        return False, "stage_device(enabled=False)", "wrong refusal tag=%s" % ex.tag
    return False, "stage_device(enabled=False)", \
        "the device path RAN without --device; that is the accident the guard exists for"


CONTROLS = (
    ("short-read", "one run's planned range served one byte short", c_short_read),
    ("wrong-offset", "one run read from file_off+1 instead of its planned file_off", c_wrong_offset),
    ("wrong-width", "one member tensor repriced at the wrong bpw on the ruler side", c_wrong_width),
    ("skipped-run", "one run dropped from the plan", c_skipped_run),
    ("corrupted-byte", "one byte of the in-memory copy flipped", c_corrupted_byte),
    ("duplicated-run", "one run counted twice", c_duplicated_run),
    ("window-guard", "the control source asked for bytes outside every window", c_window_guard),
    ("device-guard", "the device path called with the flag absent", c_device_guard),
)


# -----------------------------------------------------------------------------------------------
# 10. the checks
# -----------------------------------------------------------------------------------------------
LOCK_INVOKED = re.compile(r"gpu_" r"lock\.sh")        # split literal: the file must not contain the
HIP_IMPORT = re.compile(r"^\s*(import|from)\s+(hip|pyhip|rocm|rocmsmi|amdsmi)", re.M)
CONSTANT_PREDICATE = re.compile(r"\.check\(\s*[\"'][^\"']+[\"']\s*,\s*(?:True|False|1|0)\s*[,)]")


def _order_log_lines():
    try:
        import audit_published_quant as APQ
        path = APQ.ORDER_LOG
    except Exception:
        path = ORDER_LOG_FALLBACK
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def add_checks(v, L, stager, p, mode, idxs, rows, cov, cov_addr, source, ran_selftest, sample,
               order_before, order_after, staged_bytes):
    full = mode in ("verify", "check-only", "device")
    digests = mode != "plan-only"
    # ---- A. the plan this file executed is the loader's plan, field for field --------------------
    fresh = L.stage_runs()
    same = (len(fresh) == len(stager.plan)
            and all(a == b for a, b in zip(fresh, stager.plan)))
    v.check("plan-is-the-loaders-plan", same, stager.plan_count,
            "the %d runs staged are `loader.Loader.stage_runs()` verbatim (%d B); no plan table is "
            "copied or re-derived here" % (stager.plan_count, stager.plan_bytes))
    v.check("record-population-derived",
            len(L.records) == len(L.lay.placements)
            == sum(len(stager.members[i]) for i in range(stager.plan_count)), len(L.records),
            "%d records = %d placements = sum of per-run member counts; the 752 is read off the "
            "artifact and the layout, never typed" % (len(L.records), len(L.lay.placements)))
    v.check("plan-bytes-equals-record-bytes",
            stager.plan_bytes == sum(r.row for r in L.records), stager.plan_count,
            "sum(run.length) %d B vs sum(record.row) %d B"
            % (stager.plan_bytes, sum(r.row for r in L.records)))
    # ---- B. every printed counter reconciled against the text it was printed in -----------------
    runs_parsed = len(RUN_RE.findall(p.text()))
    tensors_parsed = len(TENSOR_RE.findall(p.text()))
    slabs_parsed = len(SLAB_RE.findall(p.text()))
    v.check("run-lines-reconciled",
            runs_parsed == len(rows) and len(rows) == (len(idxs) if digests else 0), len(idxs),
            "%d runs in the set, %d RUN lines printed, %d rows returned%s"
            % (len(idxs), runs_parsed, len(rows),
               " (plan-only reads no data byte, so it verifies and prints no run row)"
               if not digests else ""))
    v.check("tensor-lines-reconciled",
            tensors_parsed == sum(len(stager.members[i]) for i in idxs) > 0, tensors_parsed,
            "%d TENSOR lines parsed; %d is the derived population (placements of this run set), so "
            "a parser that matches nothing cannot pass" % (tensors_parsed,
                                                           sum(len(stager.members[i])
                                                               for i in idxs)))
    v.check("slab-lines-reconciled", slabs_parsed == len(LAY.SLAB_ORDER), slabs_parsed,
            "%d SLAB lines parsed against layout.SLAB_ORDER's %d" % (slabs_parsed,
                                                                      len(LAY.SLAB_ORDER)))
    if full:
        v.check("run-set-is-the-whole-plan",
                len(idxs) == stager.plan_count and sorted(idxs) == list(range(stager.plan_count)),
                stager.plan_count,
                "the plan's own count is %d (%d from loader.stage_runs, %d from "
                "layout.ResidencyLayout.copy_plan) and this run staged that many"
                % (stager.plan_count, len(L.stage_runs()), len(L.lay.copy_plan())))
    # ---- C. the structural rules, from the reasons the rows carry --------------------------------
    v.check("runs-disjoint-in-slab", cov["planned_bytes"] == cov["union_bytes"], len(idxs),
            "%d B summed over runs vs %d B after merging (a run counted twice is residue, and the "
            "duplicated-run control shows this counter moves)" % (cov["planned_bytes"],
                                                                   cov["union_bytes"]))
    v.check("plan-addresses-every-tensor-once",
            cov_addr["residue"] == 0 and cov_addr["uncovered"] == 0, stager.plan_count,
            "addressing only, no bytes read: the %d runs tile the %d placements with residue %d and "
            "uncovered %d" % (stager.plan_count, len(L.lay.placements), cov_addr["residue"],
                              cov_addr["uncovered"]))
    if digests:
        red = judge(rows)
        for name in JUDGEMENTS:
            bad = [str(r["idx"]) for r in rows if _has_tag(r, name)]
            v.check("run-rule-" + name, red[name], len(rows),
                    "%d/%d runs clean; offending run indices: %s" % (
                        len(rows) - len(bad), len(rows), ",".join(bad[:12]) or "none"))
        v.check("residue-bytes-zero", cov["residue"] == 0, len(idxs),
                "%d B covered by verified runs that is no admitted tensor's bytes"
                % cov["residue"])
        v.check("uncovered-bytes-zero", cov["uncovered"] == 0, len(idxs),
                "%d B of admitted tensors covered by no verified run of this %d-run set" % (
                    cov["uncovered"], len(idxs)))
        v.check("every-run-verified",
                all(r["verdict"] == "VERIFIED" for r in rows) and bool(rows), len(rows),
                "%d VERIFIED, %d NOT-VERIFIED, %d DROPPED"
                % (sum(1 for r in rows if r["verdict"] == "VERIFIED"),
                   sum(1 for r in rows if r["verdict"] == "NOT-VERIFIED"),
                   sum(1 for r in rows if r["verdict"] == "DROPPED")))
        v.check("digests-are-not-the-hash-of-nothing",
                all(r["plan_sha"] == r["ref_sha"] and len(r["plan_sha"]) == 64
                    and r["plan_sha"] != EMPTY_SHA256 for r in rows if r["verdict"] == "VERIFIED")
                and any(r["verdict"] == "VERIFIED" for r in rows),
                sum(1 for r in rows if r["verdict"] == "VERIFIED"),
                "a verified run's two digests are 64 hex chars, equal to each other, and not "
                "sha256(b'') (%s..)" % (rows[0]["plan_sha"][:12] if rows else "-"))
        draft_runs = [i for i in idxs if L.lay.slab(stager.plan[i]["slab"]).source != "gguf"]
        v.check("draft-slab-staging-refused",
                all(r["staged"] in ("refused-not-wired", "not-attempted")
                    for r in rows if r["idx"] in draft_runs), len(draft_runs),
                "%d draft run(s) (bf16 on disk, policy A prices Q4_K/Q8_0): loader.stage_draft "
                "refuses them and this file obeys that refusal instead of staging them"
                % len(draft_runs))
    expected_bytes = (sum(r["read"] + r["ref_read"] for r in rows) + staged_bytes
                      + source.copy_bytes)
    v.check("bytes-read-reconciled", source.bytes_read == expected_bytes, source.reads,
            "source read %d B in %d calls; the rows account for %d B (plan + header + staging) plus "
            "%d B pulled into control copies"
            % (source.bytes_read, source.reads, sum(r["read"] + r["ref_read"] for r in rows)
               + staged_bytes, source.copy_bytes))
    # ---- D. the controls ---------------------------------------------------------------------------
    if ran_selftest:
        printed = len(CONTROL_RE.findall(p.text()))
        fired = len([c for c in v.controls if c[1]])
        v.check("controls-population-derived",
                printed == len(v.controls) == len(CONTROLS), len(CONTROLS),
                "%d controls declared, %d printed a CONTROL line, %d parsed back off the text"
                % (len(CONTROLS), len(v.controls), printed))
        v.check("controls-all-fired", fired == len(CONTROLS) and bool(CONTROLS), len(CONTROLS),
                "%d/%d fired; silent: %s" % (fired, len(CONTROLS),
                                             [c[0] for c in v.controls if not c[1]] or "none"))
        v.check("selftest-window-bounded", sample is not None and sample[1] <= SELFTEST_CAP,
                sample[1] if sample else 0,
                "the controls digested %d B of a %d B cap, so the CPU selftest is not reading the "
                "artifact end to end" % (sample[1] if sample else 0, SELFTEST_CAP))
    # ---- E. the guards -------------------------------------------------------------------------------
    if mode in ("plan-only", "verify", "selftest", "check-only"):
        v.check("device-path-not-exercised", DEVICE_EXERCISED["calls"] == 0,
                DEVICE_EXERCISED["calls"],
                "mode=%s took %d device calls; %s" % (mode, DEVICE_EXERCISED["calls"],
                                                      DEVICE_EXERCISED["notes"] or "no notes"))
    if mode == "device":
        v.check("device-path-was-exercised", DEVICE_EXERCISED["calls"] >= 1,
                DEVICE_EXERCISED["calls"],
                "--device asked for a copy and %d device construction(s) happened" % (
                    DEVICE_EXERCISED["calls"]))
    v.check("data-reads-went-through-the-gated-reader",
            (source.data_read and source.token is not None) or not source.data_read, 1,
            "data_read=%s token_granted=%s grants=%d (tools/audit_published_quant.require_selftest "
            "is the only door to the data section)" % (source.data_read, source.token is not None,
                                                       TOKEN_GRANTS["n"]))
    expect_lines = TOKEN_GRANTS["n"] if order_before is not None else None
    v.check("ordering-log-append-witnessed",
            order_before is not None and order_after is not None
            and order_after - order_before == (expect_lines or 0),
            TOKEN_GRANTS["n"],
            "audit-ordering log %d -> %d lines, and %d token grant(s) happened; a run that read "
            "tensor bytes and left the count unchanged would be a dead counter" % (
                order_before, order_after, TOKEN_GRANTS["n"]))
    body = open(os.path.abspath(__file__)).read()
    consts = CONSTANT_PREDICATE.findall(body)
    v.check("self-source-has-no-constant-predicates", not consts, body.count("v.check("),
            "%d check calls in this file, %d with a literal True/False/1/0 predicate "
            "(pitfalls/40 \u00a7a-check-whose-predicate-is-True)" % (body.count("v.check("),
                                                                      len(consts)))
    v.check("copy-path-owns-no-lock", not LOCK_INVOKED.search(body) and not HIP_IMPORT.search(body),
            2, "no stream-lock invocation and no HIP module import in this file; the lock it does "
               "not own is the one it must never acquire (pitfalls/50 \u00a7self-locking-gate-drivers)")
    rates = sorted({m.group(0) for m in RATE.finditer(p.text())})
    v.check("no-rate-figures-in-output", not rates, len(p.lines),
            "%d printed lines scanned for tok/s, GB/s, MB/s, ms, elapsed, duration: %s" % (
                len(p.lines), rates or "none"))


# -----------------------------------------------------------------------------------------------
# 11. CLI
# -----------------------------------------------------------------------------------------------
MODES = ("--plan-only", "--verify", "--selftest", "--check-only")
FLAGS = ("--device",)                        # a modifier on --verify, never a mode of its own
OPTS = ("--keep-evidence", "--fault", "--cap")
USAGE = __doc__.split("CPU-only by default")[0]


def run_controls(p, v, L, stager, source, idxs, only=None):
    chosen = [c for c in CONTROLS if only is None or c[0] == only]
    if only is not None and not chosen:
        raise StageError("unknown-control", "%r is not a control; the population is %s"
                         % (only, [c[0] for c in CONTROLS]))
    for name, what, fn in chosen:
        ctx = dict(stager=stager, L=L, source=source, idxs=idxs)
        try:
            fired, injected, detail = fn(ctx)
        except StageError as ex:
            fired, injected, detail = False, "(never injected)", "raised tag=%s %s" % (ex.tag, ex)
        p.say("CONTROL name=%s fired=%s injected=%s %s" % (
            name, "RED" if fired else "SILENT", injected, detail))
        v.control(name, fired, injected, detail)


def parse(argv):
    """Refuse anything unknown: a tool that accepts every flag by silence is how a gate judges the
    wrong binary (pitfalls/50 §A-tool-with-no-argument-handling-accepts-any-flag-by-silence)."""
    opt, out = {}, {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in OPTS:
            if i + 1 >= len(argv):
                raise StageError("missing-argument", "%s needs a value" % a)
            out[a] = argv[i + 1]
            i += 2
            continue
        if a in FLAGS:
            out.setdefault("--device", True)
            i += 1
            continue
        if a in MODES:
            opt.setdefault("modes", []).append(a)
            i += 1
            continue
        raise StageError("unknown-argument", "%r is not one of %s / %s / %s; it is not ignored" % (
            a, ",".join(MODES), ",".join(FLAGS), ",".join(OPTS)))
    if len(opt.get("modes", [])) != 1:
        raise StageError("no-mode" if not opt.get("modes") else "two-modes",
                         "exactly one of %s is required (%s given); a default that silently did "
                         "something would be a mode nobody chose"
                         % (", ".join(MODES), opt.get("modes")))
    return opt["modes"][0].strip("-"), out


def main(argv):
    try:
        base_mode, out = parse(argv)
    except StageError as ex:
        print("REFUSED tag=%s %s" % (ex.tag, ex))
        return 2
    device = bool(out.get("--device"))
    if device and base_mode == "check-only":
        print("REFUSED tag=check-only-device-conflict --check-only certifies that no device call "
              "happened in this process; it will not do that and copy into a slab in the same breath")
        return 2
    if device and base_mode == "selftest":
        print("REFUSED tag=device-not-in-selftest the controls judge an in-memory copy and must never "
              "reach the iGPU; a selftest that could touch the device is not a CPU selftest")
        return 2
    if device and base_mode != "verify":
        print("REFUSED tag=device-needs-verify --device copies the verified runs, so it modifies "
              "--verify and nothing else")
        return 2
    mode = "device" if device else base_mode
    only = out.get("--fault")
    if only is not None and mode != "selftest":
        print("REFUSED tag=fault-needs-selftest --fault runs one control, which is the selftest")
        return 2
    cap = int(out.get("--cap", SELFTEST_CAP))
    ev = out.get("--keep-evidence")

    p, v = Print(), Verifier()
    order_before = _order_log_lines()
    L = LO.Loader()
    stager = Stager(L)
    p.say("PLAN runs=%d bytes=%d records=%d placements=%d sources=2 alignment=%d"
          % (stager.plan_count, stager.plan_bytes, len(L.records), len(L.lay.placements),
             L.art.alignment))
    source = ByteSource(L, why="staging execution mode=%s" % mode)
    hip = None
    staged_bytes = 0
    ran_selftest = mode in ("selftest", "check-only")
    sample = None
    rows, cov = [], stager.coverage([], [])
    idxs = []
    failure = None
    try:
        if mode == "selftest":
            idxs, sample_bytes = choose_sample(stager, cap)
            sample = (idxs, sample_bytes)
            p.say("SAMPLE runs=%d bytes=%d cap=%d rule=smallest,largest-n,largest,then-smallest-first"
                  % (len(idxs), sample_bytes, cap))
        else:
            idxs = list(range(stager.plan_count))
            if ran_selftest:
                # the controls always judge a bounded, derived sample; --check-only is the whole plan
                # digested plus this, never the whole plan digested twice over
                ctl_idxs, sample_bytes = choose_sample(stager, cap)
                sample = (ctl_idxs, sample_bytes)
                p.say("SAMPLE runs=%d bytes=%d cap=%d rule=smallest,largest-n,largest,"
                      "then-smallest-first (controls only; the CPU path above is the whole plan)"
                      % (len(ctl_idxs), sample_bytes, cap))
        print_slabs(p, L, stager, idxs)
        if mode != "plan-only":
            rows = stager.verify(source, idxs=idxs)
        print_rows(p, stager, idxs, rows)
        cov = stager.coverage(idxs, rows)
        print_tensors(p, stager, L, idxs, rows)
        if mode == "device":
            hip, slabs, peak = stage_device(L, stager, rows, source, True)
            for a in hip.allocs:
                p.say("ALLOC slab=%s bytes=%d rc=%d gtt_after=%s mem_available_after=%s"
                      % (a["slab"], a["bytes"], a["rc"], a["after"].get("gtt_used_bytes"),
                         a["after"].get("mem_available_bytes")))
            p.say("STAGED slabs=%d peak_staging=%d staged=%d mismatched=%d refused=%d"
                  % (len(slabs), peak,
                     sum(1 for r in rows if r["staged"] == "STAGED"),
                     sum(1 for r in rows if r["staged"] == "STAGE-MISMATCH"),
                     sum(1 for r in rows if r["staged"] == "refused-not-wired")))
            staged_bytes = sum(r["length"] for r in rows if r["staged"] == "STAGED")
        if ran_selftest:
            run_controls(p, v, L, stager, source, sample[0], only)
        cov_addr = stager.coverage(list(range(stager.plan_count)),
                                   [dict(idx=i, verdict="VERIFIED")
                                    for i in range(stager.plan_count)])
        p.say("COVERAGE run_set=%d planned_bytes=%d union_bytes=%d residue=%d uncovered=%d "
              "addressing_residue=%d addressing_uncovered=%d"
              % (cov["run_set"], cov["planned_bytes"], cov["union_bytes"], cov["residue"],
                 cov["uncovered"], cov_addr["residue"], cov_addr["uncovered"]))
        add_checks(v, L, stager, p, mode, idxs, rows, cov, cov_addr, source, ran_selftest, sample,
                   order_before, _order_log_lines(), staged_bytes)
    except StageError as ex:
        failure = ex
    finally:
        if hip is not None:
            try:
                hip.free_all()
            except StageError as ex:
                print("NOTE  free-failed %s" % ex)
        source.close()
    order_after = _order_log_lines()
    print()
    for name, detail in v.notes:
        print("NOTE  %-44s %s" % (name, detail))
    for name, ok, n, detail in v.rows:
        print("CHECK name=%-44s verdict=%s n=%s %s" % (name, "PASS" if ok else "FAIL", n, detail))
    bad = v.failed
    fired = sum(1 for c in v.controls if c[1])
    result = ("RESULT %s mode=%s checks=%d findings=%d controls=%d fired=%d run_set=%d plan=%d "
              "verified=%d residue=%d uncovered=%d device=%s ordering_log=%s->%s"
              % ("FAIL" if (bad or failure) else "PASS", mode, len(v.rows), len(bad) + (failure and 1 or 0),
                 len(v.controls), fired, len(idxs), stager.plan_count,
                 sum(1 for r in rows if r["verdict"] == "VERIFIED"), cov["residue"],
                 cov["uncovered"], "on" if mode == "device" else "off",
                 order_before, order_after))
    print()
    if failure is not None:
        print("REFUSED tag=%s %s" % (failure.tag, failure))
    print(result)
    for row in bad:
        print("  FINDING %s: %s" % (row[0], row[3]))
    if ev:
        os.makedirs(os.path.dirname(ev) or ".", exist_ok=True)
        with open(ev, "w") as fh:
            fh.write("RUN STARTED pid=%d argv=%s\n" % (os.getpid(), " ".join(argv)))
            fh.write("\n".join(p.lines) + "\n")
            for name, ok, n, detail in v.rows:
                fh.write("CHECK name=%-44s verdict=%s n=%s %s\n"
                         % (name, "PASS" if ok else "FAIL", n, detail))
            fh.write(result + "\n")
    return 1 if (bad or failure) else 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
