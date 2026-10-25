#!/usr/bin/env python3
"""Is a quiet worker thinking, or wedged? Elapsed time is NOT the discriminator.

Why this exists. `worker_lanes.sh` answers "is this lane alive" for the board; it cannot answer the
question that actually precedes an intervention — *should I interrupt this run at all?* Interrupting a
worker that is mid-computation destroys work; leaving a wedged one burns wall-clock and its endpoint
slot. The two look identical from the outside: both are "no events for five minutes".

They are not identical, and `pitfalls/50` §a live worker is not a reachable worker says why: a worker
blocked inside one long tool call cannot consume steering, so a queued message never delivers; but a
worker in a long *generation* is producing tokens and must be left alone. The signals that separate them
are printed here:

  open_tool_call   an open tool call  -> it is running something; steer will NOT be consumed until it ends
  open_assistant_message + no tool    -> generating, OR wedged. Decided by the next line.
  last activity Xs ago                -> 0-30s = generating (LEAVE IT); hundreds of s = WEDGED
  children + worker %CPU              -> a compiler/torch/python child, or real CPU, = computing
  artifacts so far                    -> progress with no events = a long single tool call, not a stall

Verdict logic deliberately errs toward "leave it": the cost of killing a thinking worker is lost work and
a lost hour, the cost of waiting on a wedged one is an interruption I can issue any time.

usage: python3 tools/worker_probe.py <run-dir>
       (run dirs live under /tmp/pi-subagents-uid-<uid>/async-subagent-runs/<run-id>)
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The files a GDN-oracle lane is expected to produce; edit for other lanes or ignore the section.
GDN_ARTIFACTS = (
    "speccheck/gdn_ref.py",
    "tools/dump_reference_gdn.py",
    "checkpoints/dump-manifest-gdn-2026-10-26.json",
    "data/dumps/2026-10-26-gdn",
)


def classify(open_msg, open_tool, quiet, kids, cpu, state="running"):
    """The verdict, as a pure function so --selftest can exercise every branch.

    Errs toward leaving a worker alone: killing a thinking worker loses work and an hour; waiting on a
    wedged one costs only an interruption that can be issued at any moment.

    TERMINAL is checked FIRST and on `state`, not on message/tool shape: after an interrupt the assistant
    message closes, so a dead run otherwise reads as "IDLE at a turn boundary, steer is consumable" --
    which this tool said about a corpse minutes after writing itself, and steering a failed run only
    ever produces `child completed before consuming steering`.
    """
    if state not in ("running", "queued", "starting"):
        return "TERMINAL", "state=%s -- steering cannot be delivered; use resume (it revives) or stop" % state
    try:
        cpu_f = float(cpu) if cpu else 0.0
    except (TypeError, ValueError):
        cpu_f = 0.0
    if kids or cpu_f > 25.0:
        return "COMPUTING", "do not interrupt; steering may not be consumed until the call ends"
    if open_msg and quiet is not None and quiet < 45:
        return "GENERATING", "tokens flowing; LEAVE IT (this is what a long think looks like)"
    if open_msg and quiet is not None and quiet > 180:
        return "WEDGED", "assistant message open, nothing computing, silent %ds; interrupt then resume" % quiet
    if not open_msg and quiet is not None and quiet > 180:
        return "IDLE", "at a turn boundary -- steer IS consumable; nudge it"
    return "UNDECIDED", "look at children/artifacts before acting"


def selftest():
    """Each branch must be reachable, and the two that matter must not be confusable.

    The dangerous pair is WEDGED vs GENERATING: a tool that calls a wedged stream 'generating' gets its
    worker abandoned until it times out, and a tool that calls a long think 'wedged' gets live work
    destroyed. Both directions are asserted, and a branch nobody reached is a FAIL -- a classifier with
    an unreachable branch has not been tested, it has been hoped for (pitfalls/40).
    """
    cases = [
        ("child running -> COMPUTING", True, None, 400, ["python ref.py"], "3", "COMPUTING"),
        ("high cpu -> COMPUTING", True, None, 400, [], "80", "COMPUTING"),
        ("tokens flowing -> GENERATING", True, None, 5, [], "1.2", "GENERATING"),
        ("silent+open msg -> WEDGED", True, None, 400, [], "1.2", "WEDGED"),
        ("silent+closed -> IDLE", False, None, 400, [], "1.0", "IDLE"),
        ("ambiguous -> UNDECIDED", False, None, 60, [], "0.5", "UNDECIDED"),
        # A dead run must never be mistaken for a nudgable one -- this was the tool's first real bug.
        ("dead run closed msg -> TERMINAL", False, None, 446, [], "1.0", "TERMINAL"),
        ("dead run open msg -> TERMINAL", True, None, 446, [], "1.0", "TERMINAL"),
    ]
    bad = 0
    hit = set()
    for name, om, ot, q, kids, cpu, want in cases:
        st = "failed" if want == "TERMINAL" else "running"
        got, _why = classify(om, ot, q, kids, cpu, st)
        hit.add(got)
        ok = got == want
        print("  %-30s got=%-11s %s" % (name, got, "PASS" if ok else "FAIL want=%s" % want))
        bad += 0 if ok else 1
    missing = {c[6] for c in cases} - hit
    if missing:
        print("  FAIL branches never reached: %s" % ",".join(sorted(missing)))
        bad += 1
    if classify(True, None, 400, [], "1.0")[0] == "GENERATING":
        print("  FAIL a wedged stream was classified GENERATING")
        bad += 1
    if classify(False, None, 446, [], "1.0", "failed")[0] == "IDLE":
        print("  FAIL a dead run was offered as nudgable")
        bad += 1
    print("SELFTEST %s" % ("PASS" if bad == 0 else "FAIL arms=%d" % bad))
    return 0 if bad == 0 else 1


def main(argv):
    if argv and argv[0] == "--selftest":
        return selftest()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    run = argv[1] if len(argv) > 1 else argv[0]
    if not os.path.isdir(run):
        # allow a bare run id
        cand = "/tmp/pi-subagents-uid-%d/async-subagent-runs/%s" % (os.getuid(), run)
        if not os.path.isdir(cand):
            print("no such run dir: %s" % run)
            return 2
        run = cand

    st = json.load(open(os.path.join(run, "status.json")))
    pid = st.get("pid")
    print("state=%s activity=%s step=%s pid=%s"
          % (st.get("state"), st.get("activityState"), st.get("currentStep"), pid))
    la = st.get("lastActivityAt")
    quiet = None
    if la:
        quiet = time.time() - la / 1000.0
        print("last activity %.0fs ago" % quiet)

    open_msg = False
    open_tool = None
    n = 0
    with open(os.path.join(run, "events.jsonl")) as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            n += 1
            t = e.get("type")
            if t == "message_start":
                open_msg = True
            elif t == "message_end":
                open_msg = False
            elif t in ("tool_use", "tool_call", "tool_use_start"):
                open_tool = (e.get("toolName") or e.get("name") or "?")
            elif t in ("tool_result", "tool_use_end"):
                open_tool = None
    print("events=%d  open_assistant_message=%s  open_tool_call=%s" % (n, open_msg, open_tool))

    kids = []
    cpu = None
    if pid:
        try:
            out = subprocess.run(["ps", "-o", "pid,etime,pcpu,rss,args", "--ppid", str(pid)],
                                 capture_output=True, text=True, timeout=10).stdout
            kids = [l for l in out.splitlines()[1:] if l.strip()]
            self_ = subprocess.run(["ps", "-o", "pcpu,etime", "-p", str(pid)],
                                   capture_output=True, text=True, timeout=10).stdout.splitlines()
            if len(self_) > 1:
                cpu = self_[1].split()[0]
        except Exception as exc:                                    # noqa: BLE001
            print("ps failed: %s" % exc)
    if kids:
        print("children (this is what 'thinking' looks like):")
        for l in kids[:4]:
            print("  %s" % l[:110])
    else:
        print("no child processes -- nothing is computing on this worker's behalf")
    if cpu is not None:
        print("worker pid cpu=%s" % cpu)

    verdict, why = classify(open_msg, open_tool, quiet, kids, cpu, st.get("state", "running"))
    print("VERDICT: %s -- %s" % (verdict, why))

    print("artifacts so far (GDN lane expectations; ignore for other lanes):")
    for rel in GDN_ARTIFACTS:
        p = os.path.join(REPO, rel)
        if os.path.isdir(p):
            print("  %-46s dir, %d files" % (rel, len(os.listdir(p))))
        elif os.path.isfile(p):
            print("  %-46s %d B" % (rel, os.path.getsize(p)))
        else:
            print("  %-46s (absent)" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
