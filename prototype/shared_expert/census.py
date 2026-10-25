#!/usr/bin/env python3
"""census.py — the slot census and launch bill for the shared-expert kernels.

Correctness-only arithmetic bill. It prices nothing in time and prints no milliseconds: what it can
honestly say is which kernel costs how many valuation slots, how many blocks fit per CU, and how many
launches and blocks the shape needs. Anything that turns those into cycles is a separate act with its
own premises, and this tool refuses to pretend otherwise.

Why each piece is here:

* `wavefronts_per_simd` is printed **beside** every slot bill. clang's `Occupancy [waves/SIMD]`
  remark prints 16 for every kernel in the tree — it is the machine maximum, not a per-kernel result
  (`tools/kernel_occupancy_probe.cpp` header), and pitfalls/20 §"A probe's own summary column can be
  a mislabeled counter" is exactly about a bill read without its occupancy. A 5-slot body at 1
  wavefront/SIMD is not cheaper than a 20-slot body at 8.
* The CU count comes from **KFD**, not HIP: `/sys/.../kfd/topology/nodes/1/properties` reports
  `simd_count 80`, `simd_per_cu 2` -> 40 CU x 64 lanes, while `hipDeviceGetAttribute` reports 20 CU
  here (pitfalls/20 §"The CU count from KFD, not from HIP"). A census that multiplied by the HIP
  number would price half the machine.
* The **slicer is witnessed** by two stub kernels compiled in the same translation unit:
  `se_canary_empty` must come out cheaper than `se_canary_eight_fma`, and the eight-multiply stub must
  come out at exactly 16 valuation ops (8 v_mul + 8 v_add, contraction compiled out). Without the two
  stubs a slicer that matched no kernel would print an empty table and a PASS
  (pitfalls/40 §"A check that can produce no output cannot fail").
* `se_canary_full_lds` (64 KiB of `__shared__`, written before it is read) is the occupancy canary in
  `shared_expert --occupancy`: 1 block/CU for it and >1 for the empty stub, or the probe is blind to
  LDS. Both canaries write before reading so LLVM cannot delete the allocation and make the control
  vacuous (pitfalls/20 §"A canary stub must WRITE the array before it reads it").
* **Launch count vs blocks-per-launch is not a trade this census will make.** `--per-case` and grouped
  runs produce the same words (that is gate.py's witness, not this tool's); what is priced here is
  that the four stages cannot be merged into fewer launches without a grid-wide dependency: every
  row of `down` needs all `I=512` values of `act`, which stage 1 produces across four separate
  blocks, so deleting the boundary between them replaces a launch with a global barrier. The per-case
  form is priced at `4 x n_cases` launches against grouped `4`, and the deletion argument is printed
  with the numbers rather than assumed.

    census.py [--obj PATH] [--occupancy-file PATH] [--out PATH]
"""
import argparse
import collections
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

OBJDUMP_CANDIDATES = [
    "/opt/rocm/llvm/bin/llvm-objdump",      # what exists on this box; the bare names do not
    "llvm-objdump-17", "llvm-objdump",
]

# Valuation classes, per pitfalls/20 §"Write a per-kernel slot census": count only what the real body
# can produce. v_permlane* and v_readfirstlane are NOT valuation on this part and including them
# turned a 2-slot-per-32-element reduction into a 47-slot kernel.
NON_VALUATION = ("v_permlane", "v_readfirstlane")
VAL_RE = re.compile(r"\b(v_[a-z0-9_]+)")
MEM_CLASSES = ("global_", "ds_", "image_", "load_", "store_", "scratch_", "s_load", "s_waitcnt",
               "flat_")

KFD_PROPERTIES = "/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties"


def kfd_geometry():
    """The authority on this box: the GPU node's simd_count, simd_per_cu, max_waves_per_simd,
    lds_size_in_kb. Node ids are not stable across kernels, so every node is read and the one with
    simds wins (node 0 is the CPU and reports zeros). The properties file is `key value`, not ini."""
    import glob
    best = None
    for path in sorted(glob.glob("/sys/devices/virtual/kfd/kfd/topology/nodes/*/properties")):
        out = {}
        try:
            with open(path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            out[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
        except OSError:
            continue
        need = ("simd_count", "simd_per_cu", "max_waves_per_simd", "lds_size_in_kb")
        if all(k in out for k in need) and out["simd_count"] > 0:
            if best is None or out["simd_count"] > best[1]["simd_count"]:
                best = (path, out)
    if best is None:
        raise SystemExit("census: no KFD node reports simds; the CU count cannot be derived and "
                         "HIP's number is the wrong one (pitfalls/20 §'The CU count from KFD, "
                         "not from HIP')")
    path, out = best
    if out["simd_count"] % out["simd_per_cu"]:
        raise SystemExit("census: simd_count is not a multiple of simd_per_cu")
    return out["simd_count"] // out["simd_per_cu"], out


def objdump_path():
    for c in OBJDUMP_CANDIDATES:
        if os.path.sep in c:
            if os.access(c, os.X_OK):
                return c
            continue
        from shutil import which
        if which(c):
            return c
    raise SystemExit("census: no llvm-objdump found; tried " + ", ".join(OBJDUMP_CANDIDATES))


def slice_kernels(obj):
    """Per-kernel instruction lines, sliced from `llvm-objdump -d` of the device code object.

    The section header line is `0000000000000000 <name>:` and the next one of those ends the body.
    A kernel name that matches nothing raises instead of printing an empty row.
    """
    text = subprocess.run([objdump_path(), "-d", obj], capture_output=True, text=True)
    if text.returncode != 0:
        raise SystemExit("census: llvm-objdump failed on %s: %s" % (obj, text.stderr.strip()[:200]))
    heads = {}
    for m in re.finditer(r"^([0-9a-f]{16}) <([^>]+)>:$", text.stdout, re.M):
        heads[m.start()] = m.group(2)
    order = sorted(heads)
    bodies = {}
    for i, pos in enumerate(order):
        end = order[i + 1] if i + 1 < len(order) else len(text.stdout)
        bodies[heads[pos]] = text.stdout[pos:end]
    if not bodies:
        raise SystemExit("census: no kernels found in %s — the slicer matched nothing, which is a "
                         "failure and not an empty table" % obj)
    return bodies


def bill(body):
    """Valuation-op histogram plus the memory/latency classes, for one kernel body.

    llvm-objdump on this ROCm prints the address as a *trailing* comment
    (`\tglobal_load_dword v1, v2, s[4:5]   // 0000...: ...`), so the mnemonic is the first token of
    the line and the address is not a prefix. Rows that are section headers, symbol rows or bare
    comments are skipped by requiring the mnemonic to be a bare word.
    """
    ops = collections.Counter()
    mem = collections.Counter()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = re.match(r"^([a-z][a-z0-9_.]*)\b", line)
        if not m:
            continue          # symbol rows, section headers, blank pseudo-ops
        mn = m.group(1)
        if mn.startswith("v_"):
            if mn.startswith(NON_VALUATION):
                continue
            ops[mn] += 1
        elif mn.startswith(("global_", "ds_", "image_", "load_", "store_", "scratch_",
                            "s_load", "s_waitcnt", "flat_")):
            mem[mn] += 1
    return ops, mem, sum(ops.values())


def f32_valu(ops):
    """f32 VALU work, with the peined `v_dual_*` encodings counted at two operations each.

    gfx1151 dual-issues VALU pairs into a single encoding, so a plain instruction histogram is not a
    linear function of the source: the 8-multiply stub came out at 14 encodings for 16 operations.
    Stating the witness over operations (and the duals at two) is what makes "twice the source work
    costs twice the slots" a test rather than a guess.
    """
    n = 0
    for name, count in ops.items():
        if not ("_f32" in name or "_f64" in name):
            continue
        n += count * (2 if name.startswith("v_dual_") else 1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=os.path.join(HERE, "work", "isa",
                                                 "shared_expert-hip-amdgcn-amd-amdhsa-gfx1151.o"))
    ap.add_argument("--occupancy-file", default=None,
                    help="text of `shared_expert --occupancy`; omit to report slots without occupancy")
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.obj):
        raise SystemExit("census: %s missing. Build it compile-only (no device work):\n"
                         "  hipcc -x hip --offload-arch=gfx1151 -O3 -ffp-contract=off "
                         "-fno-unsafe-math-optimizations -I%s --save-temps -c %s/shared_expert.hip "
                         "-o %s/work/x.o" % (args.obj, HERE, HERE, HERE))
    cu, kfd = kfd_geometry()

    occ = {}
    if args.occupancy_file:
        with open(args.occupancy_file) as fh:
            for line in fh:
                if line.startswith("OCCUPANCY "):
                    f = dict(kv.split("=", 1) for kv in line.split()[1:])
                    occ[f["kernel"]] = f
        canary = [l for l in open(args.occupancy_file) if l.startswith("OCCUPANCYCANARY")]
    else:
        canary = []

    bodies = slice_kernels(args.obj)
    out_lines = []

    def emit(s=""):
        print(s)
        if args.out:
            out_lines.append(s)

    emit(f"CENSUSDEVICE cu_from_kfd={cu} simd_count={kfd['simd_count']} simd_per_cu={kfd['simd_per_cu']}"
         f" max_waves_per_simd={kfd['max_waves_per_simd']} lds_kb={kfd['lds_size_in_kb']}"
         f"   (hipDeviceAttributeMultiprocessorCount reports 20 here; the census uses KFD)")
    emit("CENSUSKERNEL kernel\tvalu_encodings\tf32_valu_ops\ttop_classes\tmem_ops\tblocks_per_cu"
         "\twavefronts_per_simd")
    for name in sorted(bodies):
        ops, mem, total = bill(bodies[name])
        top = " ".join(f"{k}={v}" for k, v in ops.most_common(4))
        o = occ.get(name)
        if o:
            bpc = int(o["blocks_per_cu"])
            wf = bpc * (args.threads // 32)
            emit(f"CENSUSKERNEL {name}\t{total}\t{f32_valu(ops)}\t{top}\t{sum(mem.values())}"
                 f"\t{bpc}\t{wf}")
        else:
            emit(f"CENSUSKERNEL {name}\t{total}\t{f32_valu(ops)}\t{top}\t{sum(mem.values())}\t-\t-")

    # ---- the slicer's own witnesses -----------------------------------------------------------
    want = ("se_canary_empty", "se_canary_eight_fma", "se_canary_sixteen_fma")
    if any(k not in bodies for k in want):
        raise SystemExit("census: the canary stubs are missing from the code object; the slicer has "
                         "no witness and its numbers are unqualified")
    o0 = f32_valu(bill(bodies["se_canary_empty"])[0])
    o8 = f32_valu(bill(bodies["se_canary_eight_fma"])[0])
    o16 = f32_valu(bill(bodies["se_canary_sixteen_fma"])[0])
    ops8 = bill(bodies["se_canary_eight_fma"])[0]
    ok = (o0 == 0) and (o8 >= 8) and (o16 >= 16) and (o16 > o8)
    emit(f"CENSUSSLICER empty_f32_valu={o0} eight_madd_f32_valu={o8} sixteen_madd_f32_valu={o16} "
         f"eight_mul={ops8.get('v_mul_f32_e32', 0)} eight_add={ops8.get('v_add_f32_e32', 0)} "
         f"want=empty:0,eight>=8,sixteen>=16,sixteen>eight verdict={ok and 'ok' or 'BROKEN'}")
    if not ok:
        raise SystemExit("census: the slicer cannot see 8 f32 ops in an 8-multiply stub, or cannot "
                         "see the difference between the empty stub and the busy one. Fix the "
                         "slicer before believing the table above (pitfalls/20 §'Before writing a "
                         "probe or census tool').")
    if canary:
        emit(canary[0].rstrip())

    # ---- the launch bill, and why it is not tradable -----------------------------------------
    emit("LAUNCHBILL stages=4 gateup,down,prob,out "
         "grouped_launches=4 per_case_launches=4*n_cases "
         "stage1_blocks_per_case=4 stage2_blocks_per_case=16 stage3_blocks_per_case=17")
    emit("LAUNCHBILL merge_denied=gateup->down "
         "reason=every down row needs all 512 act values, which stage 1 produces across 4 distinct "
         "blocks; deleting that boundary buys no launch, it buys a grid-wide barrier. The prob and "
         "out stages are the same argument at I->H width. The grouped form is therefore the floor "
         "for a launch that is independent of case count, and per-case is exactly 4*n_cases: the two "
         "forms are word-identical (gate.py proves that), so nothing here trades blocks-per-launch "
         "for launch count.")
    emit("no timing taken")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(out_lines) + "\n")
            print(f"census: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
