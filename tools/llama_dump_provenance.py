#!/usr/bin/env python3
"""llama_dump_provenance.py — write provenance.json into one llama dump directory.

Written by tools/llama_dump_run.sh; kept as a file rather than an inline interpreter invocation
(the CC Safety Net blocks heredocs into an interpreter, pitfalls/50, and a named script is
reviewable on its own).

Usage: llama_dump_provenance.py OUT TAG PROMPT BINARY GGUF PINFILE RC CMDLINE [BUILD_RECORD...]
"""

import hashlib
import json
import os
import platform
import sys


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) < 9:
        print(__doc__, file=sys.stderr)
        return 2
    out, tag, prompt, binary, gguf, pinfile, rc, cmdline = argv[1:9]
    build_records = argv[9:]
    pin = json.load(open(pinfile))
    rec = {
        "tag": tag,
        "prompt_verbatim": prompt,
        "command_line": cmdline,
        "exit_code": int(rc),
        "binary": {"path": os.path.realpath(binary), "sha256": sha256(binary)},
        "gguf": {
            "path": gguf,
            "sha256": sha256(gguf),
            "pin_file": pinfile,
            "pin_sha256": pin["sha256"],
        },
        "build_records": [{"path": p, "sha256": sha256(p)} for p in build_records
                          if os.path.isfile(p)],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "note": "no timing taken; this file records what ran, not how long it took",
    }
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(f"provenance -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
