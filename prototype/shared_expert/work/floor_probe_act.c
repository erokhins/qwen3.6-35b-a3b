/* floor_probe_act.c — the C side of the `act` floor probe: the route the shipping kernel runs,
 * evaluated over the domain the field ACTUALLY lives in, beside an f64 anchor for the same route.
 *
 * Why this file exists: `work/floor_probe_exp.c` measures expf over every bf16 input pattern, which is
 * the domain of the bf16 route (`se_swiglu` with f32_act=0 feeds the transcendental a bf16-rounded
 * value). The `act` field of the f32_act case is `se_swiglu(gate, up, 1)`, whose transcendental input
 * is a raw f32 DOT PRODUCT — a domain the bf16-grid probe never touches. Judging that field with the
 * bf16-grid band is a comparison measured by a ruler calibrated somewhere else, and the honest verdict
 * for it was `UNVERIFIED by probe domain`. This probe replaces that verdict with a measurement, or
 * leaves the finding standing if the measurement says the same number.
 *
 * What it computes per input pair, and why each one:
 *   act_c    = se_swiglu(gate, up, 1)   — the kernel's own inline function, from the header both arms
 *                                        compile, with the same -ffp-contract=off. Not a model of the
 *                                        route; the route.
 *   anchor   = the same expression evaluated in double and rounded to float once at the end. Its own
 *                                        error is ~1e-16 relative, i.e. ~2e-9 of one f32 ULP, so it is
 *                                        the "exactly rounded f32" for every distance this probe can
 *                                        produce. This is the anchor role the gdn lane's floor probe
 *                                        plays (`speccheck/gdn_ref.py` §the triangle inequality, copied
 *                                        here as a method, 2026-10-26).
 * The distances themselves are taken by the repo's ruler in `floor_probe_act_np.py`, which supplies
 * the numpy leg; this file only publishes the C leg and the anchor so the two sides are compared on
 * identical inputs.
 *
 * Population: every (gate, up) pair the fixture's own f32 dot fields carry, written by
 * `floor_probe_act_np.py` out of `work/cases.tsv` + the oracle's expect dump. That population is
 * EXHAUSTIVE over what the judged field consumes, which is what lets the band be a measurement; the
 * whole f32 continuum inside the hull is ~1e9 patterns and no gate may rest on sampling it.
 *
 * Build (same flags as both arms, because a rounding claim is a claim about the flags):
 *   g++ -O2 -std=c++11 -ffp-contract=off -I<prototype/shared_expert> work/floor_probe_act.c -o work/floor_probe_act
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_expert_kernels.h"

/* Signed, monotone key over f32 bit images — the same order_key the repo's ruler uses
 * (`tools/ulp_policy.py §Ruler.order_key`, the corrected form of `pitfalls/30 §A`): flip the whole
 * image for a negative, set the sign bit for a positive. XOR-ing only the sign bit looks equivalent and
 * is not: it orders the negatives backwards, so a hull taken that way reports the wrong domain for any
 * field whose dot products are negative — which is most of them. */
static unsigned long long se_f32_rank(uint32_t b) {
    return (b & 0x80000000u) ? (unsigned long long)(~b) : ((unsigned long long)b | 0x80000000ull);
}

int main(int argc, char **argv) {
    const char *pairs_path = 0, *emit_path = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs_path = argv[++i];
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit_path = argv[++i];
        else { fprintf(stderr, "usage: floor_probe_act --pairs P --emit E\n"); return 2; }
    }
    if (!pairs_path || !emit_path) {
        fprintf(stderr, "usage: floor_probe_act --pairs P --emit E\n");
        return 2;
    }
    FILE *fh = fopen(pairs_path, "r");
    if (!fh) { fprintf(stderr, "cannot read %s\n", pairs_path); return 1; }
    FILE *oh = fopen(emit_path, "w");
    if (!oh) { fprintf(stderr, "cannot write %s\n", emit_path); return 1; }

    fprintf(oh, "idx\tgate\tup\tc_act\tanchor_f32\n");
    char line[128];
    long n = 0, skipped = 0, bad = 0;
    int have_hull = 0;
    uint32_t hull_lo_bits = 0, hull_hi_bits = 0;
    unsigned long long hull_lo_rank = 0, hull_hi_rank = 0;
    while (fgets(line, (int)sizeof(line), fh)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') continue;
        unsigned gb = 0, ub = 0;
        if (sscanf(line, "%x %x", &gb, &ub) != 2) { ++bad; continue; }
        const float g = se_float_of((uint32_t)gb);
        const float u = se_float_of((uint32_t)ub);
        /* NaN and inf are not inputs this field can carry (they are poison markers, not dots), and an
         * anchor over them is meaningless, so they are counted and skipped rather than silently banded.
         */
        if (!isfinite(g) || !isfinite(u)) { ++skipped; continue; }
        const float act_c = se_swiglu(g, u, 1);         /* the shipping route, verbatim */
        const double gd = (double)g, ud = (double)u;
        const double sd = gd * (1.0 / (1.0 + exp(-gd))); /* the same route, one rounding at the end */
        const float anchor = (float)(sd * ud);
        fprintf(oh, "%ld\t%08X\t%08X\t%08X\t%08X\n", n, gb, ub,
                se_bits_of(act_c), se_bits_of(anchor));
        /* the hull is over the transcendental's input, because that is the domain being claimed */
        const unsigned long long rg = se_f32_rank(se_bits_of(g));
        if (!have_hull || rg < hull_lo_rank) { hull_lo_bits = se_bits_of(g); hull_lo_rank = rg; }
        if (!have_hull || rg > hull_hi_rank) { hull_hi_bits = se_bits_of(g); hull_hi_rank = rg; }
        have_hull = 1;
        ++n;
    }
    fclose(fh);
    fclose(oh);
    if (n == 0) {
        fprintf(stderr, "floor_probe_act: %s held no usable pair (bad=%ld skipped=%ld); a probe that "
                        "measured nothing may not license a band\n", pairs_path, bad, skipped);
        return 1;
    }
    printf("FLOOREXP_ACT_C pairs_read=%ld pairs_emitted=%ld malformed=%ld nonfinite_skipped=%ld "
           "hull_gate=%08X..%08X hull_gate_min=%.9g hull_gate_max=%.9g\n",
           n, n, bad, skipped, hull_lo_bits, hull_hi_bits,
           se_float_of(hull_lo_bits), se_float_of(hull_hi_bits));
    printf("route=se_swiglu(gate,up,f32_act=1) from shared_expert_kernels.h contract=off anchor=f64-"
           "route-rounded-to-f32-once\n");
    printf("no timing taken\n");
    return 0;
}
