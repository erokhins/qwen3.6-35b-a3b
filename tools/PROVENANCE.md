# tools/ — provenance of the copied probes

Every file here except `doc_ref_check.py`, `hostmap_probe.cpp`, `pin_hold.cpp`, `pin_pair.cpp`,
`quant_scenarios.py` and `gguf_header_types.py` (all six are this
repo's own) is a **verbatim
copy** made on 2026-10-26 from the 0.8B engine repo, so that this project can re-measure and re-gate without
depending on that repo at read time (rule: `pitfalls/60-docs-and-citations.md`).
Keeping the copies pristine means they can be diffed against upstream later; **do not edit them to remove old
references** — record the caveats here instead.

| file | what it measures | caveat when using it **here** |
|---|---|---|
| `gpu_lock.sh` | machine-wide GPU lock (`/tmp/pi_gpu.lock`), acquire/release by TAG, `status` shows a stale holder | unchanged, and intentionally shared with the sibling project so the two repos exclude each other on the one GPU. Check `grep -c gpu_lock <script>` before a caller acquires it (see `pitfalls/50-harness-and-shell.md`) |
| `bw_probe_hip.cpp` | iGPU grid-stride read + copy sweep over blocks × threads, H2D/D2H, size–bandwidth curve | the numbers are model-independent; build `hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/bw_probe_hip bw_probe_hip.cpp` |
| `mem_mode_probe_hip.cpp` | allocation modes (`hipMalloc` / `Managed` / host-pinned) + the total device-visible pool ceiling | use it for `DECISIONS.md` D-02 (can a >34 GiB host mapping be read at full speed?) — that is exactly what it is close to |
| `gemm_peak.cpp`, `gemm_common.h` | dense matrix rate at chosen shapes/classes | its docstring quotes the 0.8B's shapes; pass the shapes this engine will actually emit (see `pitfalls/20-rates-and-occupancy.md`, "price the width the engine emits") |
| `kernel_occupancy_probe.cpp` | blocks/CU and wavefronts/CU per kernel, with a canary that proves the query discriminates 0-LDS vs 64-KiB stubs | its embedded table lists the 0.8B's kernel names; re-run it on this project's kernels |
| `launch_overhead_hip.cpp` | per-launch cost floor | dispatch-bound here, so bimodal — always report the spread |
| `bw_probe.py` | CPU-side read/copy bandwidth → decode ceiling | **`--bytes-per-token` defaults to `1.565e9`, which is the 0.8B.** For this model pass `--bytes-per-token 5.89e9` (bf16 active bytes) or the printed ceiling is wrong by ~4× |
| `isa_probe.sh` | ISA/class inventory of a code object (matrix/dot instructions, WMMA roofs) | remember the census rule: name which files it could actually read (`pitfalls/20-rates-and-occupancy.md`) |
| `doc_ref_check.py` | **not a probe — this repo's own tool**: checks that in-repo path cites resolve, that markdown table rows are terminated, and reports how many line-form, cross-repo and bare-`§` cites exist | it walks only `*.md` and exempts provenance blockquotes, so a clean run measures that population and nothing else. Run `python3 tools/doc_ref_check.py --strict` before committing a doc change (`pitfalls/60-docs-and-citations.md`) |
| `hostmap_probe.cpp` | **written here (2026-10-26)**: pool ceiling, host-pin sweep, multi-mapping totals, device-vs-pinned A/B — the D-02 measurement | every bandwidth line is checksum-witnessed and prints `MISMATCH` + "NOT A BANDWIDTH RESULT" instead of a number; the unregistered-pointer case (argv[4]=1) faults the queue, so it is opt-in and last |
| `pin_hold.cpp` | **written here (2026-10-26)**: pin N GiB and hold it, so a second process can be run concurrently — the instrument that refuted the "pool is per-process" reading | allocation only: it prints no timing, so it cannot contaminate a measurement window; evidence `devices/evidence/d02-two-process-*.txt` |
| `pin_pair.sh` | **written here (2026-10-26)**: run N concurrent `pin_hold` instances and print the grant/refusal split plus `gtt_used` and MemAvailable | its verdict line names the two hypotheses it is adjudicating; run it blocking with `tools/gpu_lock.sh` held, and keep total pinned bytes well under MemAvailable — 2×24 GiB next to halogen's 23.5 GiB is the edge of comfortable |
| `gguf_header_types.py` | **written here (2026-10-26)**: HTTP-Range-reads a GGUF header and tabulates the ggml type actually stored per tensor family, with byte totals | answers "what quantization is this file?" without downloading it. Fails loudly rather than printing a partial table (a truncated width map reads like a complete one); `BOOL` is one byte; unknown type ids print as `type_<id>` and count as zero bytes with a warning. Evidence: `quantization/evidence/q4km-published-width-map-2026-10-26.md` |
| `quant_scenarios.py` | **written here (2026-10-26)**: family census from `checkpoints/*.config.json` validated against the index `total_size`, then residency + per-token + tok/s ceilings per quantization policy | ceilings are bytes ÷ bandwidth, not engine measurements; `--census` must print a delta against `total_size` of a couple of %, otherwise the shapes drifted (`checkpoints/README.md`) |

Not copied (deliberately, so far): `mixed_bw.cpp` (1,759 lines, and the mixed read/write question it answers was
still open upstream), the dumper and gate harnesses (`reference_dump.py`, `weights_report.py`, `chat_render.py`
and friends) — those get written for this model rather than ported, because their shape assumptions are the
0.8B's. Add them here with a row above when they are.
