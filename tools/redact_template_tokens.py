#!/usr/bin/env python3
"""Replace literal chat-template special tokens with a printable placeholder.

WHY THIS EXISTS (2026-10-26, witnessed twice the same day).
Every agent on this box is served by a checkpoint whose stop list includes the template's
end-of-turn token, so a program that prints template control tokens *literally* can truncate the
tool call that captured it -- silently, mid-sentence. Worker W17's final report was cut off
exactly there, and the loss was unrecoverable: the artifact on disk holds only the fragment
before the token. Any tool whose job is to print a rendered chat template (``chat_render.py``)
or to capture such a render into ``evidence/`` therefore has to be run **through** a redactor,
or must mask internally, before its output can be logged or echoed by an agent.

The placeholder keeps the token's *name* (which is not itself control-significant) and drops
the delimiters, so a reader still sees which token was there: ``<|im_start|>`` becomes
``[TK:im_start]``. Token ids are printed instead where an id is the better citation
(``AGENTS.md``: refer to special tokens by id; re-derive per checkpoint rather than trusting a
remembered list).

Contract:
  * reads stdin (or the files given), writes stdout;
  * ALWAYS prints a ``REDACTED <n>`` line on stderr, so a zero count is a witnessed zero
    rather than a silent pass (``pitfalls/40``: a check that can produce no output cannot fail);
  * exits 2 if the pattern population is zero and ``--require-hit`` was given -- that is how a
    capture script proves the redactor was armed and needed, instead of assuming it.

The pattern is assembled from character codes rather than written as a literal token, so this
source file is itself safe to cat into an agent's output.

Usage:
  ./tools/chat_render.py ... | tools/redact_template_tokens.py
  tools/redact_template_tokens.py --require-hit < capture.txt
"""

import re
import sys

# "<" "|" NAME "|" ">" -- built from codes so no literal control token appears in this file.
_LT = chr(60)
_GT = chr(62)
_BAR = "|"
PATTERN = re.compile(
    re.escape(_LT) + re.escape(_BAR) + r"([A-Za-z0-9_\-\.]{1,64})" + re.escape(_BAR) + re.escape(_GT)
)


def redact(text):
    """Return (redacted_text, hits, names) with every template token replaced."""
    names = []

    def sub(m):
        names.append(m.group(1))
        return "[TK:%s]" % m.group(1)

    return PATTERN.sub(sub, text), len(names), names


def main(argv):
    require_hit = False
    paths = []
    for a in argv:
        if a == "--require-hit":
            require_hit = True
        elif a == "--selftest":
            return selftest()
        elif a == "-h" or a == "--help":
            sys.stdout.write(__doc__)
            return 0
        else:
            paths.append(a)

    if paths:
        chunks = []
        for p in paths:
            with open(p, "r", errors="replace") as fh:
                chunks.append(fh.read())
        data = "".join(chunks)
    else:
        data = sys.stdin.read()

    out, hits, names = redact(data)
    sys.stdout.write(out)
    sys.stdout.flush()
    sys.stderr.write("REDACTED %d\n" % hits)
    if names:
        uniq = {}
        for n in names:
            uniq[n] = uniq.get(n, 0) + 1
        sys.stderr.write("NAMES %s\n" % " ".join("%s:%d" % kv for kv in sorted(uniq.items())))
    if require_hit and hits == 0:
        sys.stderr.write("FAIL require-hit saw zero template tokens\n")
        return 2
    return 0


def selftest():
    """Watch the redactor fire. A redactor that has never been seen to redact is decoration."""
    # Built from codes so this source stays safe to print.
    samples = [
        (_LT + _BAR + "im_start" + _BAR + _GT, "im_start"),
        (_LT + _BAR + "im_end" + _BAR + _GT, "im_end"),
        (_LT + _BAR + "think" + _BAR + _GT, "think"),
        (_LT + _BAR + "no-think" + _BAR + _GT, "no-think"),
    ]
    bad = 0
    for lit, name in samples:
        out, hits, names = redact("A" + lit + "B")
        ok = hits == 1 and names == [name] and lit not in out and ("[TK:%s]" % name) in out
        print("  %-24s hits=%d placeholder=%s %s" % (name, hits, ("[TK:%s]" % name) in out, "PASS" if ok else "FAIL"))
        bad += 0 if ok else 1
    # Negative control: text that must NOT be touched, and must report a witnessed zero.
    out, hits, _ = redact("not a token: <| bare, x|y, <im_sep>, [TK:im_start]")
    ok = hits == 0
    print("  %-24s hits=%d (expect 0) %s" % ("negative-control", hits, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
    # Idempotence: running twice must not create or destroy anything.
    once, n1, _ = redact("x" + samples[0][0] + "y")
    twice, n2, _ = redact(once)
    ok = n1 == 1 and n2 == 0 and once == twice
    print("  %-24s first=%d second=%d %s" % ("idempotent", n1, n2, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
    print("SELFTEST %s" % ("PASS" if bad == 0 else "FAIL arms=%d" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
