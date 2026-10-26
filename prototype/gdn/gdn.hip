/* gdn.hip — the device translation unit for slice 1.
 *
 * Two things are true of this file and both are checked at run time:
 *   * it is compiled by hipcc for gfx1151, so its arithmetic is the DEVICE compiler's codegen and
 *     not g++'s — this is the arm the zero-tolerance compiled-pair comparison exists for;
 *   * this arm EXECUTES ON THE CPU. It issues no HIP runtime call: `hip_calls=0` is printed on the
 *     RUN line and `gate.py` refuses the arm without it. The iGPU arm proper — real launches, real
 *     buffers — is a later slice's business and is not here; slice 1's verdict is a numerics
 *     verdict and takes no device run and no timing.
 *
 * A kernel launch that read these stage arrays would need [h_v,C,d] scratch in device memory and an
 * item->thread map over the eleven items in gdn_case.h; writing it now, before the arithmetic has
 * agreed with the oracle on one chunk, would be the placeholder this lane is explicitly not allowed
 * to leave behind.
 */
#include "gdn_arm.h"

#ifndef __HIPCC__
#error "gdn.hip must be compiled by hipcc; the g++ arm is host_probe.cpp"
#endif

int main(int argc, char **argv) {
    return gdn_arm_main(argc, argv, "hip_hostpath", "hipcc");
}
