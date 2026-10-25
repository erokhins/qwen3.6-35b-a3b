#!/usr/bin/env python3
"""Splice work/new_route_discriminates.txt over route_discriminates() in oracle_arm.py.

One-shot patch tool, kept next to the other patch scripts this lane already carries. It refuses to
write unless it found exactly one start line and exactly one end marker, because a splice that matched
nothing and wrote the file anyway is the same failure as a gate that parses nothing and prints PASS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
TARGET = os.path.join(LANE, "oracle_arm.py")
NEW = os.path.join(HERE, "new_route_discriminates.txt")
START = "def route_discriminates("
END = "# ---- the declared route, vectorised over rows but never across k -------------------------------"

src = open(TARGET).read().splitlines(True)
new = open(NEW).read()
starts = [i for i, l in enumerate(src) if l.startswith(START)]
ends = [i for i, l in enumerate(src) if l.startswith(END)]
if len(starts) != 1 or len(ends) != 1:
    raise SystemExit("refusing: found %d start lines and %d end markers, need exactly 1 and 1"
                     % (len(starts), len(ends)))
s, e = starts[0], ends[0]
if not (e > s):
    raise SystemExit("refusing: end marker (line %d) is before the start (line %d)" % (e + 1, s + 1))
out = "".join(src[:s]) + new + "\n" + "".join(src[e:])
open(TARGET, "w").write(out)
print("spliced oracle_arm.py lines %d..%d (%d lines out, %d lines in) -> %d lines total"
      % (s + 1, e, e - s, new.count("\n"), out.count("\n")))
