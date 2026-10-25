"""engine/ — the deliverable: policy A's weight residency and load path.

Three pieces, one per question:

  * `artifact.py` — what the shipping file claims to be, and the refusal to load anything else
    (identity, derived name/element census, per-tensor width census, the zero-`mtp.*` fact);
  * `layout.py` — where policy A's bytes go: slabs, per-tensor offsets, read classes, and the
    two budgets D-11 split apart (development co-resident / deployment halogen-down);
  * `loader.py` — the load path: plan → validate → stage bytes → hand a kernel lane a handle,
    with the parts that are not built raising instead of quietly returning something plausible.

`gate.py` is the CPU gate over all three, with the negative controls that show each check can
fail. `README.md` says what is wired and what is stubbed.

Nothing in this package touches the iGPU: no HIP import, no timing call, no `ms` or tok/s figure
printed. Weight *bytes* are read from disk; device residency is a plan until the engine's own
allocator exists (see README §Stubbed).
"""

__all__ = ["artifact", "layout", "loader"]
