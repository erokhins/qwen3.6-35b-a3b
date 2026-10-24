#!/usr/bin/env bash
# pin_pair.sh — run N concurrent pin_hold instances to ask whether the device-visible budget is
# per process or shared. Companion to tools/pin_hold.cpp; see that file's header for why.
#
# Usage: tools/pin_pair.sh [gib=24] [count=2] [hold_s=18] [bin=/tmp/pin_hold]
# Allocation only: no timing is printed. Run blocking, with tools/gpu_lock.sh held.
set -u
gib=${1:-24}
count=${2:-2}
hold=${3:-18}
bin=${4:-/tmp/pin_hold}

if [ ! -x "$bin" ]; then
  echo "no binary at $bin (build: hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/pin_hold tools/pin_hold.cpp)"
  exit 1
fi

echo "launching $count concurrent pins of $gib GiB each (total $(echo "$count * $gib" | bc) GiB)"
for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
  [ -r "$f" ] && echo "gtt_used before = $(cat "$f") bytes ($(echo "scale=2; $(cat $f)/1073741824" | bc) GiB)"
done
free -k | awk '/MemAvailable/ {print "MemAvailable = " int($2/1048576) " GiB"}'
pids=()
for i in $(seq 1 "$count"); do
  "$bin" "$gib" "$hold" &
  pids+=($!)
done
ok=0
fail=0
for p in "${pids[@]}"; do
  if wait "$p"; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
done
echo "granted $ok of $count  (refused $fail)"
if [ "$fail" -eq 0 ]; then
  echo "VERDICT: total granted exceeds the single-process boundary -> the budget is PER PROCESS."
else
  echo "VERDICT: a concurrent pin was refused -> the budget is SHARED (per-UID/global or client-divided)."
fi
