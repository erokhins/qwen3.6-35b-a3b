/* m2_g_host.cpp — the g++ twin of the M2 routed-expert gather arm.
 *
 * Same plan, same slab, same record, same arithmetic header as `m2_g_arm.hip`, compiled by a different
 * compiler: the other half of a zero-tolerance pair (`engine/block_device.py` prints
 * `PAIR pair=g_* file=gather_record.f32 … ndiff_bytes=…` for each width path). A non-zero there is a
 * codegen finding, never a band to absorb — `routed_sum` on the reference path is licensed at exactness,
 * so "the two compilers disagree on the combine" is exactly the fact that has to come out before any of
 * the rest could be believed. `expert_raw` is banded by a floor probe, and even there a pair difference
 * is a finding: the band is the licensor's re-association bound, not a container for compiler drift.
 *
 * No HIP symbol exists in this binary. Asked for `--device` it fails with
 * `tag=device-path-absent-in-this-binary`, and asked for the device-only allocation fault it fails with
 * `tag=fault-device-only`, rather than answering either with a host run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_g_io.h"

#define M2G_ARM_NAME "g_host_twin"
#define M2G_COMPILER "g++"

struct M2GRunStats {
    const char *path;
    int hip_calls, launches, allocs, bad_index;
    long bytes_in, bytes_out;
    char read_hex[129];
};

#include "m2_g_main.inc"
