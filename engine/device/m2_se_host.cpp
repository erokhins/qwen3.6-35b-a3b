/* m2_se_host.cpp — the g++ twin of the M2 shared-expert arm.
 *
 * Same plan, same slab, same record, same arithmetic header as `m2_se_arm.hip`, compiled by a
 * different compiler. Its whole purpose is to be the other half of a zero-tolerance pair:
 * `engine/block_device.py` runs both binaries on one host-visible plan and requires
 * `PAIR … ndiff_bytes=0`, so a disagreement is a codegen finding and not something a band may
 * absorb (`prototype/gdn/gate.py` §the compiled pair is the pattern this copies).
 *
 * No HIP symbol exists in this binary. Asked for `--device` it fails with
 * `tag=device-path-absent-in-this-binary` rather than quietly answering with a host run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_se_io.h"

#define M2_SE_ARM_NAME "host_twin"
#define M2_SE_COMPILER "g++"

#include "m2_se_main.inc"
