/* m2_d_host.cpp — the g++ twin of the M2 GDN recurrent arm: the engine's own host-side GDN body.
 *
 * This is the second of the two implementations the slice is required to have. It is not a Python
 * reference and not a transcription of one: it executes `m2d_run` from `m2_d_kernels.h`, the serial
 * host-side drive of the same item bodies `m2_d_arm.hip` reaches with seven launches. `m2_d_arm.hip`
 * never calls `m2d_run`, so neither binary's answer is produced by the other — the pair compares two
 * compilations of one arithmetic, which is the only comparison that can show codegen moved a number.
 *
 * Same plan, same slab, same record, same arithmetic header as `m2_d_arm.hip`, compiled by a different
 * compiler: the other half of a zero-tolerance pair (`engine/block_device.py` prints
 * `PAIR pair=d_host_twin_vs_d_hip_hostpath file=gdn_record.f32 … ndiff_bytes=…`). A non-zero there is a
 * codegen finding, never a band to absorb: the licensed GDN rows are the four whose bands the engine's
 * own torch body already clears at `ndiff=0`, so two compilers disagreeing on them would mean the
 * arithmetic as written is not stable under a compiler, and that fact has to surface before any row
 * built on it can be believed.
 *
 * No HIP symbol exists in this binary. Asked for `--device` it fails with
 * `tag=device-path-absent-in-this-binary`, and asked for the device-only allocation fault it fails with
 * `tag=fault-device-only`, rather than answering either with a host run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_d_io.h"

#define M2D_ARM_NAME "d_host_twin"
#define M2D_COMPILER "g++"

struct M2DRunStats {
    const char *path;
    int hip_calls, launches, allocs;
    long bytes_in, bytes_out;
    char read_hex[129];
};

#include "m2_d_main.inc"
