#!/usr/bin/env bash
# Device-window A/B: llama.cpp full-offload decode rate, one fixed prompt.
# Owns NO lock -- the caller holds tools/gpu_lock.sh (pitfalls/50: a self-locking driver
# called by a caller already holding the lock deadlocks silently).
# Prompt is recorded verbatim below and echoed here, because the 2026-10-26 morning window
# quoted 44.9/49.1/51.4 t/s from "one fixed ~50-word prompt" whose text was never written
# down -- so that figure cannot be re-derived, and a second window must not repeat that.
set -u
# Committed location is tools/, so this goes up ONE level. It previously lived in
# work/device-window/ and went up two -- copying it without fixing this would have run
# every path below outside the repo.
cd "$(dirname "$0")/.." || exit 2
P='The capital of France is Paris, and the capital of Germany is Berlin. Write a short paragraph describing why river valleys were fertile ground for early agriculture, mentioning irrigation, silt, and seasonal flooding in the Nile and Tigris regions, and note two crops.'
echo "PROMPT_WORDS=$(echo $P | wc -w)"
echo "PROMPT=${P}"
for i in warm 1 2 3; do
  t0=$(date +%s)
  out=$(engines/llama.cpp/build-hip/bin/llama-cli \
        -m models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        -ngl 99 -s 42 --temp 0 --no-warmup -st -v -n 128 -p "$P" 2>&1)
  rc=$?
  t1=$(date +%s)
  # This llama.cpp build prints decimals with a COMMA (locale), e.g.
  #   eval time = 585,70 ms / 32 tokens ( 18,89 ms per token, 52,94 tokens per second)
  # and a summary line. A [0-9.]+ pattern matches neither and silently yields an empty
  # field -- the empty-grep trap again, so both the pattern and the emptiness are guarded.
  ev=$(echo "$out" | grep -oE 'Prompt: [0-9,]+ t/s \| Generation: [0-9,]+ t/s' | tail -1)
  off=$(echo "$out" | grep -oE 'offloaded [0-9]+/[0-9]+ layers' | tail -1)
  printf 'RUN %-5s rc=%s wall=%ss | %s | %s\n' "$i" "$rc" "$((t1 - t0))" "${ev:-NO_EVAL_LINE}" "${off:-NO_OFFLOAD_LINE}"
done
