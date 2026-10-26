/* m2_rt_host.cpp — the g++ twin of the M2 router arm.
 *
 * Same plan, same slab, same record, same arithmetic header as `m2_rt_arm.hip`, compiled by a
 * different compiler: the other half of a zero-tolerance pair (`engine/block_device.py` prints
 * `PAIR … file=router_record.f32 … ndiff_bytes=…`). A non-zero there is a codegen finding, never a
 * band to absorb — the router's `topk_weights` row is licensed at exactness, so "the two compilers
 * disagree on the weights" is precisely the fact that would have to come out before anything else
 * could be believed.
 *
 * No HIP symbol exists in this binary. Asked for `--device` it fails with
 * `tag=device-path-absent-in-this-binary` rather than answering with a host run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_rt_io.h"

#define M2_RT_ARM_NAME "rt_host_twin"
#define M2_RT_COMPILER "g++"

struct M2RtRunStats {
    const char *path;
    int hip_calls, launches, allocs, bad_index;
    long bytes_in, bytes_out;
};

#include "m2_rt_main.inc"
