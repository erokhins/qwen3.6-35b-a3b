#!/usr/bin/env python3
"""Check the in-repo citations and table integrity of this project's markdown.

Why this exists: `pitfalls/60-docs-and-citations.md` makes this repo self-contained, which
only holds if the paths it cites exist here. Three sibling lessons shaped this script:

* a checker's clean bill measures only the population it scanned, so it prints what it walked
  and what it skipped, and treats zero matches as a question rather than a pass;
* a `path:NNN` line cite is a claim about a file's line count, so line numbers are reported,
  never trusted (and never used to write anything);
* a check that can produce no output cannot fail, so `--strict` turns any finding into a
  non-zero exit and `run` asserts the file count is above a floor.

Checks, per markdown file:
 1. every `path` in backticks that starts with `pitfalls/`, `devices/` or `tools/` resolves
    here (with or without a trailing `.md`). Blockquote lines (`> ...`) are provenance headers
    and are exempt: they name the sibling project they came from.
 2. markdown table rows start and end with `|` (an unterminated row silently drops a column).
 3. information only: counts of `path:NNN` line-form cites, of cross-repo cites (`../qwen...`),
    and of bare-section cites (`§N` with no file named), so a reader can see whether the rule is
    holding rather than whether a tool happened to look.

Usage:  python3 tools/doc_ref_check.py [--strict] [--root PATH]
"""
import argparse
import os
import re
import sys

INREPO = ("pitfalls/", "devices/", "tools/")
CITE_RE = re.compile(r"`((?:%s)[A-Za-z0-9_./-]+)`" % "|".join(re.escape(p) for p in INREPO))
LINE_CITE_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:md|py|cpp|h|hpp|sh|json)):(\d+)`")
CROSS_RE = re.compile(r"\.\./qwen[\w.-]*")
BARE_SECT_RE = re.compile(r"§\d")


def markdown_files(root):
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", ".idea", ".junie", "__pycache__",
                                                "node_modules", "models", "data", "engines")]
        for f in sorted(fs):
            if f.endswith(".md"):
                out.append(os.path.join(d, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if any finding")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--min-files", type=int, default=10)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    files = markdown_files(root)
    if len(files) < args.min_files:
        print("POPULATION SUSPECT: only %d markdown files under %s (floor %d)"
              % (len(files), root, args.min_files))
        return 1

    problems = []
    n_line = n_cross = n_sect = 0
    for p in files:
        rel = os.path.relpath(p, root)
        lines = open(p, encoding="utf-8").read().splitlines()
        body = "\n".join(l for l in lines if not l.startswith(">"))
        for m in CITE_RE.finditer(body):
            t = m.group(1)
            if any(c in t for c in "{}*<>"):
                continue
            if not any(os.path.exists(os.path.join(root, cand))
                       for cand in (t, t + ".md")):
                problems.append("MISSING  %-40s cites %s" % (rel, t))
        for i, l in enumerate(lines, 1):
            s = l.strip()
            if s.startswith("|") and not s.endswith("|"):
                problems.append("TABLE    %s:%d row does not end with |" % (rel, i))
        n_line += len(LINE_CITE_RE.findall("\n".join(lines)))
        n_cross += len(CROSS_RE.findall("\n".join(lines)))
        n_sect += len(BARE_SECT_RE.findall("\n".join(lines)))

    print("walked %d markdown files under %s (provenance blockquotes exempt from path checks)"
          % (len(files), root))
    print("info: %d line-form `path:NNN` cites (each is a claim about a line count), "
          "%d cross-repo `../qwen…` mentions, %d bare `§N` cites" % (n_line, n_cross, n_sect))
    for x in problems:
        print(x)
    print("findings: %d" % len(problems))
    if problems and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
