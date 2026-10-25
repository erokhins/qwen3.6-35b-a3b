#!/usr/bin/env bash
# worker_lanes.sh — is a quiet worker THINKING or is it STUCK, answered by counters not by silence.
#
# Why this exists. On 2026-10-26 the harness raised "no observed activity for 300s" twice, and both
# times the lane was healthy: a `thinking high` worker composing a long file emits NO event until the
# write tool call lands, so a big markdown document or a long thinking block reads as a stall. The
# tempting response — `steer` / `interrupt` — is the harmful one: it injects into a live stream, and
# the fleet deliberately runs 4 of the 5 advertised LAN slots, so the injection also spends the
# operator's stability margin. Conversely a genuinely dead lane (the `Connection error` deaths when
# the LAN server was restarted) DOES need a resume, and sitting on it wastes wall time. So the
# question has to be answerable in one command, and answerable HONESTLY.
#
# WHAT COUNTS AS EVIDENCE. Two positive signals only, both narrow:
#   (a) the lane's own events.jsonl moved within the age limit — it emitted something recently; or
#   (b) its last event is an unclosed message_start (no "stopReason":"stop") AND its serving endpoint
#       currently reports a non-zero live stream — an in-flight generation that simply has not
#       produced an event yet.
# Deliberately NOT evidence: "some file in the repo changed". The first version of this script had
# that signal and it marked 10-hour-old corpses from earlier sessions ALIVE, because *other* workers
# are always writing — a check that cannot report STALE is not a check. Endpoint stream counts are
# fleet-wide and cannot be attributed to one lane, which is why (b) requires the lane's own pending
# message and not merely a busy server. Anything else prints UNKNOWN, and UNKNOWN is never ALIVE
# ("cannot observe" is not evidence of life — same rule that stops an empty parse reporting PASS,
# pitfalls/40).
#
# Usage: tools/worker_lanes.sh [age-limit-seconds]      (default 300; lanes quieter than 6 h are
#        reported as HISTORICAL and excluded from the verdict, since they belong to old sessions)
set -uo pipefail

# --selftest: fabricate runs in a throwaway rundir and require every verdict this tool can give to
# actually be produced. Same rule this repo applies to every gate: a branch that cannot fire is
# decorative, and a verdict nobody has seen printed is not evidence the tool works. TERMINAL used to
# be unreachable-by-construction the other way (a finished run read ALIVE); STALE became
# unreachable after TERMINAL landed, because 28 of 29 real runs emit a terminal marker even when
# killed -- so both branches are now witnessed here rather than assumed.
if [ "${1:-}" = "--selftest" ]; then
  fx=$(mktemp -d /tmp/wl-selftest.XXXXXX) || { echo "selftest: cannot build fixtures"; exit 2; }
  mk() { mkdir -p "$fx/$1" 2>/dev/null; printf '%s\n' "$2" > "$fx/$1/events.jsonl"; touch -d "$3" "$fx/$1/events.jsonl"; }
  mk 0a0a0001 '{"type":"x"}
{"type":"subagent.run.completed","status":"complete"}' '2 minutes ago'
  mk 0a0a0002 '{"provider":"tensorfold","type":"turn_end"}' '40 minutes ago'
  mk 0a0a0003 '{"provider":"tensorfold","type":"message_start"}' '20 seconds ago'
  mk 0a0a0004 '{"provider":"tensorfold","type":"turn_end"}' '7 hours ago'
  mk 0a0a0005 '{"type":"turn_end"}' '40 minutes ago'
  out=$(PI_ASYNC_RUNS="$fx" bash "${BASH_SOURCE[0]}" 300 2>&1)
  rm -rf "$fx"
  # Match the verdict COLUMN only. The first version cased against $out, and $out contains the verdict
  # guide, whose text names every verdict -- so every want() matched no matter what the classifier
  # printed, and the "selftest" could not fail. A negative control (every fixture given a terminal
  # marker, which makes STALE unreachable) still printed 5/5 PASS, which is how this was found.
  # length()==8 rather than /^[0-9a-f]{8}$/: this box's awk is mawk, which does not implement POSIX
  # interval expressions, so {8} matches nothing at all and the extraction silently returned empty --
  # a selftest that reported 5/5 FAIL for a reason unrelated to the classifier.
  # tr '\n' ' ' is load-bearing, not cosmetic: want() matches " $verd " against "* $1 *", and a
  # newline-separated list has no space around any verdict, so every want() failed even with all five
  # verdicts present. Three separate ways this one selftest could silently lie, all found only because
  # a negative control was actually run: (1) matching $out matched the legend that names every verdict,
  # (2) mawk has no {8} interval expressions, (3) newlines are not spaces.
  verd=$(printf '%s\n' "$out" | awk '$1 ~ /^[0-9a-f]+$/ && length($1)==8 && $2 ~ /^prov=/ {print $5}' | tr '\n' ' ')
  fail=0
  want() { case " $verd " in *" $1 "*) echo "  PASS verdict $1 produced by the classifier" ;; *) echo "  FAIL verdict $1 never produced"; fail=1 ;; esac; }
  want TERMINAL; want STALE; want ALIVE; want HISTORICAL; want UNKNOWN
  case "$out" in *"alive=1"*) echo "  PASS exactly one fixture alive" ;; *) echo "  FAIL alive count wrong"; fail=1 ;; esac
  [ "$fail" = 0 ] && { echo "worker_lanes selftest PASS — every verdict reachable"; exit 0; }
  echo "worker_lanes selftest FAIL"; exit 1
fi

AGE_LIMIT=${1:-300}
HIST=$((6 * 3600))
RUNDIR=${PI_ASYNC_RUNS:-/tmp/pi-subagents-uid-1000/async-subagent-runs}
TF_URL=${TF_HEALTH_URL:-http://192.168.10.27:8888/health}
HG_URL=${HG_HEALTH_URL:-http://127.0.0.1:8731/health}

field() { tr '{,' '\n\n' | grep -o "\"$1\": *[0-9.]*" | head -1 | tr -d ' "' | cut -d: -f2; }
get()   { curl -s -m 5 "$1" 2>/dev/null | field "$2"; }

echo "probe: $(date -u +%FT%TZ)   age limit ${AGE_LIMIT}s   rundir $RUNDIR"
echo

tf_conn=$(get "$TF_URL" connections);  tf_dec=$(get "$TF_URL" decode_tokens_per_second)
h_in=$(get "$HG_URL" in_flight);       h_proc=$(get "$HG_URL" requests_proc)
h_busy=$(get "$HG_URL" busy_for_s)
printf 'endpoint tensorfold: connections=%s decode_t/s=%s\n' "${tf_conn:-UNKNOWN}" "${tf_dec:-UNKNOWN}"
printf 'endpoint halogen   : in_flight=%s requests_proc=%s busy_for_s=%s\n' \
       "${h_in:-UNKNOWN}" "${h_proc:-UNKNOWN}" "${h_busy:-UNKNOWN}"
[ -n "${tf_conn:-}" ] || echo "  (tensorfold unreadable — lanes served by it can only be UNKNOWN here, never ALIVE)"
[ -n "${h_in:-}" ]    || echo "  (halogen unreadable — same rule applies)"
echo

total=0; n_alive=0; n_stale=0; n_unk=0; n_hist=0; n_term=0
for ev in "$RUNDIR"/[0-9a-f]*/events.jsonl; do
  [ -f "$ev" ] || continue
  id=$(basename "$(dirname "$ev")"); id=${id:0:8}
  mt=$(stat -c %Y "$ev" 2>/dev/null || echo 0)
  age=$(( $(date +%s) - mt ))
  prov=$(grep -o '"provider":"[a-z]*"' "$ev" 2>/dev/null | tail -1 | cut -d'"' -f4)
  [ -n "$prov" ] || prov=unknown
  last=$(tail -c 6000 "$ev" 2>/dev/null | tail -1)

  sig=""
  if [ "$age" -le "$AGE_LIMIT" ]; then sig="$sig own-event-recent"; fi
  case "$last" in
    *'"message_start"'*)
      case "$last" in
        *'"stopReason":"stop"'*|*'"stopReason":"stop"'*) : ;;                    # closed: not in flight
        *) case "$prov" in
             tensorfold) [ "${tf_conn:-0}" != "0" ] && [ -n "${tf_conn:-}" ] && sig="$sig pending-message+tf-stream" ;;
             halogen)    [ "${h_in:-0}"    != "0" ] && [ -n "${h_in:-}" ]    && sig="$sig pending-message+hal-stream" ;;
           esac ;;
      esac ;;
  esac
  sig=${sig# }

  # A run that has already reported itself terminal is TERMINAL no matter how fresh its last event
  # is. Without this, a run that completed 60 s ago reads ALIVE -- same verdict as a live one -- and
  # the first thing that does with that verdict is refuse to start a device window because "a worker
  # is still active", i.e. wait on a corpse. Observed: a completed halogen lane reported ALIVE at
  # age 94 s, and only the supervisor's own completion notice contradicted it.
  term=0
  if grep -qE '"type":"subagent\.run\.(completed|process_terminal)"' "$ev" 2>/dev/null; then term=1; fi
  tstat=$(grep -oE '"status":"[a-z_]+"' "$ev" 2>/dev/null | tail -1 | cut -d'"' -f4)

  if [ "$age" -gt "$HIST" ]; then
    v=HISTORICAL; n_hist=$((n_hist+1))
  elif [ "$term" = 1 ];    then v=TERMINAL;  n_term=$((n_term+1))
  elif [ -n "$sig" ];      then v=ALIVE;    n_alive=$((n_alive+1))
  elif [ "$age" -gt "$AGE_LIMIT" ]; then
    if [ "$prov" = unknown ]; then v=UNKNOWN; n_unk=$((n_unk+1))
    else v=STALE; n_stale=$((n_stale+1)); fi
  else v=quiet; fi
  total=$((total+1))
  printf '%-9s prov=%-11s evts=%-5s age=%-6s %-11s %s\n' "$id" "$prov" \
    "$(wc -l < "$ev" 2>/dev/null)" "${age}s" "$v" "$sig"
done
echo
echo "lanes=$total  alive=$n_alive  terminal=$n_term  stale=$n_stale  unknown=$n_unk  historical(excluded)=$n_hist"
cat <<'GUIDE'
verdict guide
  TERMINAL   the run reported itself complete or failed -- do NOT wait on it, and do NOT count it as
             an active worker when deciding whether a device window is clear. A fresh event age does
             not make a finished run live; this verdict exists because that exact confusion would
             block a measurement for no reason.
  ALIVE      do NOT steer or interrupt: that injects into a live stream and spends the fifth-slot margin. Wait for the completion notification instead.
  STALE      no event within the limit and no in-flight generation on its endpoint: likely dead.
             Check events.jsonl for 'Connection error' / compaction failure, then resume.
  UNKNOWN    provider or endpoint unreadable from here — ASK, do not assume it is alive.
  HISTORICAL belongs to an earlier session; ignored.
GUIDE
