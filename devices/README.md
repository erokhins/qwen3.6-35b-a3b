# devices/ — this box, measured

| file | contents |
|---|---|
| `measured-facts.md` | bandwidth, geometry, compute rates, the box-wide device-visible pool (≈55.5–56.5 GiB total; ≈32 GiB free while halogen is resident), and the roofline arithmetic for `Qwen3.6-35B-A3B`. **This file is the authority**; nothing here cites another project. |
| `../pitfalls/00-measurement.md` | when a number taken here may be trusted (blocking runs, bimodality, warm-up, two rulers, canaries) |
| `../pitfalls/10-device-memory.md` | allocation/residency consequences (pool ceiling, silent `Managed` degradation, first touch, staging rates, the recipe) |
| `../tools/` | the probes that produced these numbers, so a re-take is possible in-repo |

Two standing points:

- **Do not re-measure to confirm.** Re-take only after a driver / BIOS / carve-out change, using the checklist at
  the end of `measured-facts.md`. Re-verifying a number that nothing changed is a budget leak, and it produces a
  *new window*, which is how a 5–10 % bimodal swing gets mistaken for a regression.
- **Always pair a number with its instrument and its window.** Median without the spread is not a result on this
  box (see `pitfalls/00-measurement.md`, §the protocol).
