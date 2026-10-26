/* host_probe.cpp — the host twin arm: the same three headers compiled by g++ with no ROCm at all.
 * Built by build.sh at -O2 -ffp-contract=off, and compared at ZERO tolerance against the device
 * translation unit's host path. No timer is read; see gdn_case.h. */
#include "gdn_arm.h"

int main(int argc, char **argv) {
    return gdn_arm_main(argc, argv, "host_twin", "g++");
}
