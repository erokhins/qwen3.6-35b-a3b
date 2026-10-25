import re
p = "oracle_arm.py"
s = open(p).read()
start = s.index("def layout_discriminator(")
end = s.index("# ---- the declared route")
new = '''def fused_of(wg, wu):
    return np.concatenate([wg, wu], axis=0)


def layout_row_order(fused, gate_vals, up_vals, n):
    """Does our assembly + Q8_0 round-trip keep row j where HF put it? Answered per candidate row map
    by nearest row, not bit equality: the fused image is a Q8_0 round-trip, so it is never bit-equal to
    the bf16 original and an equality test here would grade the quantizer instead of the row order.

    What this can and cannot say, stated plainly because the difference matters: the [2I,H] view is OUR
    assembly (the published file stores two separate tensors, printed by the WIDTH lines above), and
    the HF bf16 weight shards are not on this box, only the dump's copies are. So this test cannot
    discover a row order in a file; it verifies that the quantizer and the assembly are row-preserving,
    which is the failure a transposed Q8 pack would produce. The question "does the kernel read the row
    map the slab was written with" is answered by route_discriminates below and by the injected
    wrong-candidate cases, which is where an actual disagreement would show up.
    """
    norms = np.linalg.norm(fused, axis=1)
    for lay in LAYOUTS:
        hits = 0
        for src, which in ((gate_vals, 0), (up_vals, 1)):
            for j in range(n):
                want = layout_rows(lay, j)[which]
                if int(np.argmin(np.abs(norms - np.linalg.norm(src[j])))) == want:
                    hits += 1
        print("LAYOUTROW candidate=%-16s rows_nearest_own_hf_row=%d/%d (2I=%d, %d rows each side)"
              % (lay, hits, 2 * n, TWO_I, n))
    marg = []
    for src in (gate_vals[:n], up_vals[:n]):
        for row in src:
            d = np.sqrt(((fused - row) ** 2).sum(axis=1))
            o = np.sort(d)
            marg.append(float(o[1] / o[0]) if o[0] > 0 else float("inf"))
    print("LAYOUTROW margin_nearest_vs_runner_up min=%.3f median=%.3f (the nearest-row test is only a "
          "test if the winner is clearly nearer; this is by how much)"
          % (min(marg), float(np.median(marg))))
    return min(marg)


def route_discriminates(x, ref_raw, wg, wu, wd):
    """Which kernel row map reproduces the pinned HF eager `shared_raw`, measured against HF's own
    bytes. Distances, not bit-identity: this route runs on Q8_0-decoded weights, so it cannot be
    bit-equal to an eager bf16 reference, and pretending otherwise would be a band typed to fit what the
    build produces. The claim that carries the answer is the ORDER of the distances: the honest map must
    be far closer than every wrong one, including the one-row-early control, or a real layout bug and an
    off-by-one would be indistinguishable. Vectorised matmul on purpose, because this half measures a
    distance to HF and does not produce expectations; the declared association order is not what is under
    test here."""
    import torch
    xt = torch.from_numpy(x.astype(np.float32))

    def bf(a):
        return a.to(torch.bfloat16).to(torch.float32)

    def raw_of(grows, urows):
        g = bf(xt @ torch.from_numpy(wg[grows]))
        u = bf(xt @ torch.from_numpy(wu[urows]))
        act = bf(bf(g * torch.sigmoid(g)) * u)
        return bf(act @ torch.from_numpy(wd.T))

    rows = {lay: np.array([layout_rows(lay, j) for j in range(I)]) for lay in LAYOUTS}
    out = {}
    for lay in LAYOUTS:
        raw = raw_of(rows[lay][:, 0], rows[lay][:, 1])
        d = (raw - torch.from_numpy(ref_raw)).abs()
        cos = float((raw * torch.from_numpy(ref_raw)).sum()
                    / (raw.norm() * torch.from_numpy(ref_raw).norm()))
        out[lay] = (float(d.max()), cos)
        print("LAYOUTRT candidate=%-16s shared_raw_vs_pinned_eager maxabs=%.6g cos=%.9f"
              % (lay, float(d.max()), cos))
    lim = TWO_I - 1
    ctl = raw_of(np.minimum(rows["gate_first"][:, 0] + 1, lim),
                 np.minimum(rows["gate_first"][:, 1] + 1, lim))
    dctl = float((ctl - torch.from_numpy(ref_raw)).abs().max())
    honest = out["gate_first"][0]
    worst_wrong = min(v[0] for l, v in out.items() if l != "gate_first")
    print("LAYOUTRT candidate=row_shift1(control) shared_raw_vs_pinned_eager maxabs=%.6g"
          "  (control: reading one row early must not be the best reading either)" % dctl)
    print("LAYOUTRT winner=gate_first nearest_wrong_ratio=%.1fx control_also_worse=%s"
          % (worst_wrong / honest if honest else float("inf"), "yes" if dctl > honest else "no"))
    if worst_wrong <= honest:
        raise SystemExit("the row map the kernel implements is not distinguishable from a wrong one: "
                         "honest maxabs=%.6g, closest wrong=%.6g" % (honest, worst_wrong))
    if dctl <= honest:
        raise SystemExit("the one-row-early control is not worse than the honest map, so this "
                         "discriminator cannot tell a layout bug from an off-by-one")


'''
open(p, "w").write(s[:start] + new + s[end:])
print("patched: replaced %d chars with %d" % (end - start, len(new)))
