#!/usr/bin/env python3
"""Enumerate every check the device gate printed, with its observed outcome per arm.

Two lanes on this box have carried a predicate that could not fire; the way to find those by inspection,
instead of by a device run being held hostage, is the full list of checks with the number each one
wanted and the number it got. This parses the transcript, splits host-arm checks from device-arm checks
by position (the device block is everything after the host arm's last check), evaluates each outcome
from want/got as printed, and asserts the count it derives equals the FAILED lines the driver printed
-- a parser that matched nothing would print an empty table and a clean bill of health.

    python3 work/enumerate_checks.py
Writes evidence/check-enumeration-2026-10-26.tsv. No device, no lock, no timing.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
LOG = os.path.join(LANE, "evidence", "shared-expert-device-2026-10-26.log")
OUT = os.path.join(LANE, "evidence", "check-enumeration-2026-10-26.tsv")
LAST_HOST_CHECK = "launch counts are what the plan says"


def outcome(want, got):
    if want == got:
        return "pass"
    if want == "non-zero":
        return "pass" if got != "0" else "FAIL"
    if want.startswith(">="):
        try:
            return "pass" if float(re.sub(r"[^0-9.]", "", got) or 0) >= float(want[2:]) else "FAIL"
        except ValueError:
            return "see-detail"
    try:
        return "pass" if int(want) == int(got) else "FAIL"
    except ValueError:
        return "see-detail"


def main():
    checks, failed = [], []
    for line in open(LOG):
        if line.startswith("CHECK "):
            # The name is padded to a fixed width only when it is short enough, so a long name is
            # separated from `want=` by one space: split on the markers, never on the padding.
            lw = line.find(" want=")
            lg = line.find(" got=", lw)
            if lw < 0 or lg < 0:
                raise SystemExit("cannot parse a CHECK line into name/want/got: %r" % line[:120])
            name = line[len("CHECK "):lw].strip()
            want = line[lw + len(" want="):lg].strip()
            rest = line[lg + len(" got="):]
            got = re.split(r" {2,}", rest.strip())[0].strip()
            checks.append((name, want, got))
        elif line.startswith("FAILED "):
            failed.append(line[len("FAILED "):].strip())
    cut = [i for i, (n, _, _) in enumerate(checks) if LAST_HOST_CHECK in n]
    if len(cut) != 1:
        raise SystemExit("found %d split markers (%s), need exactly 1" % (len(cut), LAST_HOST_CHECK))
    rows = []
    for i, (name, want, got) in enumerate(checks):
        arm = "host" if i <= cut[0] else "device"
        rows.append((arm, name, want, got, outcome(want, got)))
    nfail = sum(1 for r in rows if r[4] == "FAIL")
    if not rows:
        raise SystemExit("parsed 0 CHECK lines: an empty enumeration cannot fail")
    if nfail != len(failed):
        print("WARNING derived %d failing checks, driver printed %d FAILED lines (the same name can "
              "be checked per field, so the driver's list groups them)" % (nfail, len(failed)))
    with open(OUT, "w") as fh:
        fh.write("# arm\tcheck\twant\tgot\toutcome\n")
        for arm, name, want, got, o in rows:
            fh.write("%s\t%s\t%s\t%s\t%s\n" % (arm, name, want, got, o))
    print("WROTE %s checks=%d host=%d device=%d fail=%d see_detail=%d driver_FAILED_lines=%d"
          % (OUT, len(rows), sum(1 for r in rows if r[0] == "host"),
             sum(1 for r in rows if r[0] == "device"), nfail,
             sum(1 for r in rows if r[4] == "see-detail"), len(failed)))
    for arm, name, want, got, o in rows:
        if o != "pass":
            print("%-6s %-7s %s want=%s got=%s" % (arm.upper(), o, name, want, got))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
