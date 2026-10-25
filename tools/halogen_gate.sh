#!/usr/bin/env bash
# halogen_gate.sh — put the halogen server DOWN for a measurement window, and bring it back.
#
# Why this exists (numbers measured on this box, 2026-10-26; see
# devices/measured-facts.md §"Two model endpoints, and where each one's GPU is"):
#
#   halogen is not a remote endpoint, it is a container ON THIS BOX — `halogen-flash-server`
#   (image ghcr.io/peonist-ai/halogen-flash-server:0.18.9), API on 127.0.0.1:8731, engine
#   flash_serve on 8730. While it is up, `rocm-smi --showpids` charges it 24.29 GB against the
#   device-visible pool, and that residency is the whole difference between "≈32 GiB for one
#   process" and "≈55 GiB if the engine owns the box" (pitfalls/10-device-memory.md). Its polling
#   loop also holds the iGPU at high utilisation independently of whether anyone is streaming.
#
#   So a halogen stream is a DEVICE ACTIVITY, not a token stream: same hazard class as a running
#   gate. A measurement taken with halogen up is a measurement taken in a smaller pool with a
#   second client on the memory controller, and the 0.8B project's clean-vs-contended spread
#   (2.61-3.59 ms against a documented 2.650 ± 0.02) is what that buys.
#
# WHAT THIS SCRIPT IS NOT
#   It never acquires tools/gpu_lock.sh, and never will: a driver that self-acquires the lock and
#   is called by a caller already holding it deadlocks silently — no CPU, no GPU, no message, no
#   exit (pitfalls/50-harness-and-shell.md, "self-locking driver deadlock"). `down` therefore
#   REFUSES unless the caller already holds the lock, declared via GPU_LOCK_TAG.
#
# USE (the main agent only — a subagent does not stop the box's server)
#   tag="35b cold sweep window"
#   tools/gpu_lock.sh acquire "$tag"; trap "tools/gpu_lock.sh release '$tag'" EXIT
#   GPU_LOCK_TAG="$tag" tools/halogen_gate.sh down     # admit + stop + verify residency gone
#   ./prototype/... --time ...                          # the measurements, blocking
#   tools/halogen_gate.sh up                            # restart + wait for /health ok
#
# VERBS
#   status                     container, stream counters, resident bytes, both locks
#   admit [N]                  exit 0 iff nothing is streaming (N probes, default 3, 2 s apart)
#   down                       admit -> ~/halogen-models/stop.sh -> verify flash_serve is gone
#   up                         start (detached) -> wait for /health status:ok, print wall time
#   wait                       wait for readiness only
#   pause | unpause            EXPERIMENTAL (docker pause): freezes the polling loop but keeps the
#                              24.29 GB mapped. Unproven as a measurement-isolation mechanism —
#                              do not build a policy on it until the paired A/B in
#                              pitfalls/00-measurement.md says what the contamination actually is.
#
# The restart cost is NOT yet measured. Loading the 66.69 GB checkpoint is disk + pin work, so
# `up` prints its wall time and appends it to devices/evidence/halogen-restart.log; until that log
# has entries, assume a restart is expensive and batch every measurement of a session into ONE
# down-window rather than stopping per gate.
set -uo pipefail

NAME=halogen-flash-server
API=http://127.0.0.1:8731
MODEL_DIR=${HALOGEN_MODEL_DIR:-$HOME/halogen-models}
START=$MODEL_DIR/start.sh
STOP=$MODEL_DIR/stop.sh
SELF_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU_LOCK=$SELF_DIR/gpu_lock.sh
GPU_TAGFILE=/tmp/pi_gpu.lock/holder
STREAM_LOCK=${HALOGEN_STREAM_LOCK:-/tmp/pi_halogen.lock}
EVIDENCE=$SELF_DIR/../devices/evidence
START_LOG=${HALOGEN_START_LOG:-$EVIDENCE/halogen-restart.log}

now() { date -u +%FT%TZ; }
epoch() { date -u +%s; }
die() { echo "halogen_gate: $*" >&2; exit 1; }

running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; }

health_field() {  # $1 = json key; empty output if unreachable
  curl -s --max-time 8 "$API/health" 2>/dev/null \
    | grep -o "\"$1\":[0-9.]*" | head -1 | cut -d: -f2
}
processing() {  # requests in flight at the engine, from its own counter
  curl -s --max-time 8 "$API/metrics" 2>/dev/null \
    | awk '$1=="llamacpp:requests_processing" {print $2; found=1} END{if(!found) print ""}'
}
resident_bytes() {  # what the engine holds against the device-visible pool
  rocm-smi --showpids 2>/dev/null | awk '/flash_serve/ {print $4; exit}'
}
gpu_lock_state() { "$GPU_LOCK" status 2>/dev/null || echo "gpu_lock.sh missing"; }
gpu_lock_tag()   { [ -f "$GPU_TAGFILE" ] && cut -f2 "$GPU_TAGFILE" || true; }

stream_claim() {  # who is running a halogen-backed worker, if anyone
  local f=$STREAM_LOCK/holder
  [ -f "$f" ] && cat "$f" || true
}

cmd_status() {
  echo "time:            $(now)"
  if running; then
    echo "container:       running ($NAME)"
    echo "  in_flight:     $(health_field in_flight)"
    echo "  queued:        $(health_field queued)"
    echo "  busy_for_s:    $(health_field busy_for_s)"
    echo "  requests_proc: $(processing)"
    # instantaneous decode rate; reads 0 when the server is idle, 58.5 under a real stream (2026-10-26)
    echo "  decode s/tok:  $(curl -s --max-time 8 "$API/metrics" 2>/dev/null | awk '$1=="llamacpp:predicted_tokens_seconds"{print $2}') (0 = idle)"
  else
    echo "container:       not running"
  fi
  local rb; rb=$(resident_bytes)
  echo "resident:        ${rb:-0} bytes ($(LC_ALL=C awk -v b="${rb:-0}" 'BEGIN{printf "%.2f GiB", b/1073741824}'))"
  echo "gpu_lock:        $(gpu_lock_state)"
  echo "stream claim:    ${CL:-none}"
  echo "model dir:       $MODEL_DIR"
}

# --- admission: is it safe to take the box's server away? -------------------------------
# The counters are the gate, not an assumption. Someone else's session is allowed to be *idle*
# on halogen (in_flight 0 between turns), so this is a "no work in flight right now" test, and
# it is deliberately not a claim of exclusive ownership — that is what the stream lock is for.
cmd_admit() {
  local n=${1:-3} i busy=0
  if ! running; then echo "admit: ok (already down)"; return 0; fi
  local claim; claim=$(stream_claim)
  if [ -n "$claim" ]; then
    die "refused: a halogen-backed worker is claimed ($claim) — stop or park it first"
  fi
  for i in $(seq 1 "$n"); do
    local p q f
    p=$(processing); q=$(health_field queued); f=$(health_field in_flight)
    echo "probe $i/$n: requests_processing=${p:-?} queued=${q:-?} in_flight=${f:-?}"
    [ "${p:-1}" = "0" ] && [ "${q:-1}" = "0" ] && [ "${f:-1}" = "0" ] || busy=1
    [ "$i" -lt "$n" ] && sleep 2
  done
  [ "$busy" = "0" ] || die "refused: halogen is streaming (or unreachable) — someone is using it"
  echo "admit: ok — $n/$n probes idle"
}

cmd_down() {
  # Guard 1: never pull the server out from under the session that is being served by it.
  [ "${PI_PROVIDER:-}" != "halogen" ] \
    || die "refused: this session is served by halogen (PI_PROVIDER=halogen) — stopping it kills this session"
  # Guard 2: the caller must already hold the GPU lock, declared, not inferred (no self-acquire).
  [ -n "${GPU_LOCK_TAG:-}" ] \
    || die "refused: set GPU_LOCK_TAG to the tag you acquired tools/gpu_lock.sh with"
  [ "$(gpu_lock_tag)" = "$GPU_LOCK_TAG" ] \
    || die "refused: gpu_lock is not held as '$GPU_LOCK_TAG' (it reads: $(gpu_lock_state))"
  # Guard 3: docker must be reachable without a password here, or the recipe is a lie.
  docker ps --format '{{.Names}}' >/dev/null 2>&1 || die "refused: docker not usable as $(id -un)"
  [ -x "$STOP" ] || die "refused: $STOP not found/executable"

  if ! running; then echo "down: already down ($(now))"; return 0; fi
  cmd_admit "${ADMIT_PROBES:-3}" || exit 1

  local t0 wall
  t0=$(epoch)
  echo "down: stopping $NAME (stop.sh allows a 60 s grace to flush the KV tier)"
  "$STOP" || die "stop.sh failed"
  while running; do
    sleep 2
    [ $(( $(epoch) - t0 )) -lt 120 ] || die "container still present after 120 s — inspect: docker ps -a"
  done
  wall=$(( $(epoch) - t0 ))
  local rb; rb=$(resident_bytes)
  [ -z "$rb" ] || echo "down: WARNING flash_serve still charged ${rb} bytes — residency did not drop"
  echo "down: stopped in ${wall}s; flash_serve absent from rocm-smi --showpids"
  echo "down: the pool is now the engine's — record this window with tools/gpu_lock.sh util for the log"
}

cmd_up() {
  [ -x "$START" ] || die "refused: $START not found/executable"
  if running; then echo "up: already running"; cmd_wait "${1:-120}"; return $?; fi
  # start.sh is `docker run --rm` WITHOUT -d, so it blocks; detach it properly. This is a server,
  # not a measurement — the blocking-run rule (pitfalls/00) is about timings, not daemons.
  mkdir -p "$EVIDENCE" 2>/dev/null || true
  local t0 wall
  t0=$(epoch)
  echo "up: launching $START (detached; log /tmp/halogen_start.log)"
  setsid nohup "$START" >/tmp/halogen_start.log 2>&1 &
  cmd_wait "${1:-1800}" || { echo "up: FAILED to become ready; last 20 log lines:"; tail -20 /tmp/halogen_start.log; exit 1; }
  wall=$(( $(epoch) - t0 ))
  echo "up: ready in ${wall}s"
  printf '%s\tstart_to_ready=%ss\n' "$(now)" "$wall" >> "$START_LOG" 2>/dev/null || true
  echo "up: appended to $START_LOG (this is the restart-cost record the policy is waiting on)"
}

cmd_wait() {
  local tmo=${1:-1800} t0 now_s st eng
  t0=$(epoch)
  while :; do
    st=$(curl -s --max-time 8 "$API/health" 2>/dev/null | grep -o '"status":"[a-z]*"' | head -1 | cut -d'"' -f4)
    eng=$(curl -s --max-time 8 "$API/health" 2>/dev/null | grep -o '"responds":\(true\|false\)' | head -1 | cut -d: -f2)
    now_s=$(( $(epoch) - t0 ))
    if [ "$st" = "ok" ] && [ "$eng" = "true" ]; then
      echo "ready: /health status=ok engine.responds=true after ${now_s}s"
      return 0
    fi
    [ "$now_s" -lt "$tmo" ] || { echo "wait: timeout after ${tmo}s (status=${st:-none} engine=${eng:-none})" >&2; return 1; }
    if [ $(( now_s % 30 )) -lt 3 ]; then echo "  waiting: status=${st:-none} engine=${eng:-none} (${now_s}s)"; fi
    sleep 3
  done
}

cmd_pause() {
  running || die "not running"
  local p; p=$(processing)
  [ "${p:-1}" = "0" ] || die "refused: $p request(s) in flight — pausing with a live request hangs it to the 3600 s queue timeout"
  echo "pause: EXPERIMENTAL — freezes the polling loop, keeps $(resident_bytes) bytes mapped."
  docker pause "$NAME" >/dev/null && echo "pause: done (verify with 'status'; resume with 'unpause', never leave it paused)"
}

cmd_unpause() { running || die "not running"; docker unpause "$NAME" >/dev/null && echo "unpause: done"; }

case "${1:-status}" in
  status)   CL=$(stream_claim) cmd_status ;;
  admit)    shift; cmd_admit "${1:-3}" ;;
  down)     shift; cmd_down ;;
  up)       shift; cmd_up "${1:-1800}" ;;
  wait)     shift; cmd_wait "${1:-1800}" ;;
  pause)    shift; cmd_pause ;;
  unpause)  shift; cmd_unpause ;;
  *) sed -n '2,40p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
