#!/usr/bin/env python3
"""Report unreachable statements: a statement sitting after a return/raise/break/continue in the same
block is dead code, and in this tree that shape has a history (a botched edit once left a stale second
copy of a census body below its `return`, which still referenced a retired CASCADE_BAND key and so could
not have raised its way to attention). CPU only, stdlib only.

    .venv/bin/python tools/dead_after_return.py engine/gate.py engine/wave_flip_census.py
    .venv/bin/python tools/dead_after_return.py --selftest   # fires on 3 fixtures, stays clean on 3
"""
from __future__ import annotations

import ast
import sys

TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def walk(body, path, out):
    seen = None
    for stmt in body:
        if seen is not None:
            out.append(f"{path}: line {stmt.lineno}: unreachable after "
                       f"{type(seen).__name__.lower()} at line {seen.lineno}")
        # Only the first offender per block is reported; the rest are the same mistake.
            continue
        if isinstance(stmt, TERMINATORS):
            seen = stmt
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                walk(inner, f"{path}:{stmt.lineno}", out)
        for field in ("handlers",):
            for h in getattr(stmt, field, []) or []:
                walk(h.body, f"{path}:{stmt.lineno}:handler", out)


def selftest() -> int:
    """Watch the checker fire, and watch it NOT fire. Both directions are evidence.

    Fixtures are source *strings*, so this file contains no dead code of its own and the arms cannot
    drift into the tree. `A/C` are firing controls (a defect must be reported); `B/D/E` are
    false-positive controls (a legal program must be clean) -- a checker that reports everything
    passes the firing arms and is still worthless. Non-vacuity is asserted, not assumed: if no arm
    produced any finding, the run FAILS, because a counter that reads exactly 0 everywhere has not
    been witnessed non-zero anywhere it must fire (pitfalls/40).
    """
    arms = [
        ("A return-then-statement", "def f(x):\n    return x\n    print(x)\n", "must",
         "unreachable after return"),
        ("B raise guarded by if", "def g(x):\n    if x:\n        raise ValueError()\n    return 1\n",
         "must_not", None),
        ("C continue-then-statement", "def h(xs):\n    for v in xs:\n        if v:\n            continue\n            print(v)\n",
         "must", "unreachable after continue"),
        ("D clean function", "def k(x):\n    if x:\n        return 1\n    return 2\n", "must_not", None),
        ("E try/finally clean", "def m(x):\n    try:\n        v = x\n    finally:\n        pass\n    return v\n",
         "must_not", None),
        ("F dead after raise in try", "def n():\n    try:\n        raise ValueError()\n        print(1)\n    except ValueError:\n        pass\n",
         "must", "unreachable after raise"),
    ]
    bad = 0
    fired_any = 0
    for name, src, kind, needle in arms:
        out: list[str] = []
        walk(ast.parse(src).body, name, out)
        if kind == "must":
            ok = bool(out) and any(needle and needle in line for line in out)
            fired_any += 1 if out else 0
        else:
            ok = not out
        print("  %-24s findings=%d %s" % (name, len(out), "PASS" if ok else "FAIL"))
        if not ok:
            for line in out:
                print("      reported: %s" % line)
        bad += 0 if ok else 1
    if fired_any == 0:
        print("  FAIL non-vacuity: no firing arm produced a finding -- the checker never worked")
        bad += 1
    print("SELFTEST %s" % ("PASS" if bad == 0 else "FAIL arms=%d" % bad))
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selftest":
        return selftest()
    if not argv:
        print(__doc__)
        print("usage: dead_after_return.py <file>... | --selftest")
        return 2
    bad = 0
    for name in argv:
        src = open(name).read()
        out: list[str] = []
        walk(ast.parse(src).body, name, out)
        for line in out:
            print(line)
        bad += len(out)
    print(f"checked {len(argv)} file(s): {bad} unreachable statement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
