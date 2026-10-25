/* host_probe.cpp — the host arm of prototype/moe_router: the SAME moe_router_kernels.h bodies,
 * compiled by g++, run over the SAME plan and slab as the device driver.
 *
 * Two jobs:
 *   1. run every case on the host and dump host_out.f32, so gate.py can attribute a host/device
 *      disagreement to the device compiler rather than to two spellings of the arithmetic. The
 *      claim is field-scoped and says so: bit-identity on the fields whose exactness is claimed
 *      (the cast weights, the ids, the routed sum, and the logits of an arm=LOGITS case), band
 *      agreement on the long dots (R1's dot, R2's denominator), where glibc's expf and ROCm's
 *      libdevice expf are permitted to differ by an ULP or two.
 *   2. structural probes: predicates about the arithmetic that do not need the GPU, the dumps or a
 *      plan, each of which can fail and each of which prints a record gate.py counts.
 *
 * No timer: correctness only.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <vector>

#include "moe_router_kernels.h"

static int g_fail = 0;

/* --------------------------------------------------------------------------------------------- */
/* plan + slab, the same files the device driver reads                                             */
/* --------------------------------------------------------------------------------------------- */
static int read_plan(const char *path, std::vector<struct MrCase> *cases, long long *slab_elems,
                     long long *out_floats) {
    FILE *fh = fopen(path, "r");
    if (!fh) { printf("FAIL cannot open plan %s: %s\n", path, strerror(errno)); return 1; }
    char line[1024];
    int printed = 0;
    while (fgets(line, sizeof(line), fh)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == 0) continue;
        ++printed;
        if (strncmp(line, "GEOM", 4) == 0) {
            int e = 0, k = 0, h = 0, se = 0, of = 0;
            if (sscanf(line + 4, " E=%d K=%d H=%d slab_elems=%d out_floats=%d", &e, &k, &h, &se,
                       &of) != 5 || e != MR_E || k != MR_K || h != MR_H) {
                printf("FAIL host probe: GEOM line malformed or off-geometry: %s", line);
                fclose(fh); return 1;
            }
            *slab_elems = se; *out_floats = of;
            continue;
        }
        struct MrCase c;
        memset(&c, 0, sizeof(c));
        if (strncmp(line, "CASE", 4) != 0 ||
            sscanf(line + 4, " %d %d %d %d %d %d %d %d %d %d %d %d %d", &c.id, &c.arm, &c.mode,
                   &c.tie_class, &c.id_asc, &c.raw_src, &c.src_key, &c.x_off, &c.wg_off, &c.lg_off,
                   &c.raw_off, &c.dst_off, &c.cmp_off) != 13) {
            printf("FAIL host probe: malformed plan line: %s", line);
            fclose(fh); return 1;
        }
        cases->push_back(c);
    }
    fclose(fh);
    if (printed != (int)cases->size() + 1) {
        printf("FAIL host probe: %d printed records, %zu cases parsed\n", printed, cases->size());
        return 1;
    }
    return 0;
}

/* --------------------------------------------------------------------------------------------- */
/* structural probes. Each one states a property of the arithmetic and can fail.                  */
/* --------------------------------------------------------------------------------------------- */
static uint32_t fb(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

static void probe(const char *name, int ok, int n, const char *detail) {
    printf("HOSTPROBE name=%s ok=%d n=%d%s%s\n", name, ok ? 1 : 0, n, detail && *detail ? " " : "",
           detail ? detail : "");
    if (!ok) ++g_fail;
}

/* a deterministic filler so the structural probes do not depend on the dumps. The value is kept in
 * [0.5,2.0): an arbitrary 16-bit pattern would put the exponent field on 0xFF sometimes, and a
 * probe that passes because both sides produced the same NaN proves nothing. */
static uint16_t fill(int a, int b) {
    uint32_t h = (uint32_t)(a * 2654435761u) ^ (uint32_t)(b * 40503u);
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    const uint16_t hi = (h & 1u) ? 0x3F80u : 0x3F00u;        /* exponent 127 or 126 */
    return (uint16_t)(hi | (h & 0x7Fu));
}

static int structural_probes() {
    static uint16_t wg[(size_t)MR_E * MR_H];
    static uint16_t x[MR_H];
    for (int l = 0; l < MR_E; ++l)
        for (int j = 0; j < MR_H; ++j) wg[(size_t)l * MR_H + j] = fill(l, j);
    for (int j = 0; j < MR_H; ++j) x[j] = fill(7, j);

    /* --- the row-off-by-one injection reads exactly the next expert's row --------------------- */
    {
        int bad_index = 0, ok = 1, distinct = 0;
        for (int l = 0; l < MR_E; l += 37) {
            const uint16_t a = mr_logit_row(x, wg, l, 0, 0, &bad_index);
            const uint16_t b = mr_logit_row(x, wg, (l + 1) % MR_E, 0, 0, &bad_index);
            const uint16_t s = mr_logit_row(x, wg, l, 1, 0, &bad_index);
            if (s != b) ok = 0;
            distinct += (a != b);
        }
        char det[96];
        sprintf(det, "adjacent rows giving different logits=%d", distinct);
        probe("row_off_1_reads_next_row", ok && !bad_index && distinct > 0,
              MR_E * MR_H * 3 / 37 + 3, det);
    }
    /* --- the transposed read is a column dot, not a row dot ----------------------------------- */
    {
        int bad_index = 0;
        float acc = 0.0f;
        for (int j = 0; j < MR_H; ++j)
            acc += mr_bf16_to_f32(x[j]) * mr_bf16_to_f32(wg[(size_t)j * MR_E + 5]);
        const uint16_t want = mr_f32_to_bf16(acc);
        const uint16_t got = mr_logit_row(x, wg, 5, 0, 1, &bad_index);
        char det[96];
        sprintf(det, "want %04x got %04x", want, got);
        probe("transposed_is_column_dot", got == want && !bad_index, MR_H, det);
        const uint16_t normal = mr_logit_row(x, wg, 5, 0, 0, &bad_index);
        probe("transposed_differs_from_row_dot", normal != got, 1,
              normal == got ? "the two reads are indistinguishable; the injection is decoration" : "");
    }
    /* --- top-k ascending returns the same SET, permuted --------------------------------------- */
    {
        float probs[MR_E];
        int a[MR_K], b[MR_K];
        for (int l = 0; l < MR_E; ++l) probs[l] = 1.0f + (float)((l * 37) % 101) / 1024.0f;
        mr_topk_row(probs, MR_MODE_CORRECT, a);
        mr_topk_row(probs, MR_MODE_TOPK_ASC, b);
        int sa[MR_E] = {0}, sb[MR_E] = {0};
        for (int k = 0; k < MR_K; ++k) { sa[a[k]] = 1; sb[b[k]] = 1; }
        int same = 1, asc = 1;
        for (int l = 0; l < MR_E; ++l) same &= (sa[l] == sb[l]);
        for (int k = 1; k < MR_K; ++k) asc &= (b[k] > b[k - 1]);
        int desc = 1;
        for (int k = 1; k < MR_K; ++k) desc &= (probs[a[k]] <= probs[a[k - 1]]);
        probe("topk_asc_same_set", same && asc && desc, 3 * MR_K,
              same ? (asc ? "" : "not ascending") : "the set changed");
    }
    /* --- dropping R4 cannot move the ids: dividing a row by one positive constant is monotone -- */
    {
        static uint16_t lg[MR_E];
        float p0[MR_E], p1[MR_E], w0[MR_K], w1[MR_K];
        uint16_t c0[MR_K], c1[MR_K];
        int i0[MR_K], i1[MR_K];
        for (int l = 0; l < MR_E; ++l) lg[l] = fill(l, 99);
        mr_softmax_row(lg, MR_MODE_CORRECT, p0);
        mr_softmax_row(lg, MR_MODE_NO_MAX_SUB, p1);
        mr_topk_row(p0, MR_MODE_CORRECT, i0);
        mr_topk_row(p0, MR_MODE_NO_RENORM, i1);
        int same_ids = 1;
        for (int k = 0; k < MR_K; ++k) same_ids &= (i0[k] == i1[k]);
        mr_renorm_row(p0, i0, MR_MODE_CORRECT, w0, c0);
        mr_renorm_row(p0, i0, MR_MODE_NO_RENORM, w1, c1);
        int wdiff = 0;
        for (int k = 0; k < MR_K; ++k) wdiff += (fb(w0[k]) != fb(w1[k]));
        probe("no_renorm_leaves_ids", same_ids, MR_K, same_ids ? "" : "ids moved under R4");
        probe("no_renorm_moves_weights", wdiff > 0, MR_K,
              wdiff ? "" : "the renorm injection is decoration");
        /* and the reference arm's weights must sum to 1 within a derived fp32 slack */
        double s = 0.0;
        for (int k = 0; k < MR_K; ++k) s += (double)w0[k];
        probe("renorm_sums_to_one", s > 0.99999 && s < 1.00001, MR_K, "");
    }
    /* --- the two order injections, each provoked by its own construction ----------------------
     * One probe covering both could only report "something failed"; they are separated so a failure
     * names the leg. The data is chosen, not decorative: bf16 accumulation order only shows up when
     * the addends sit within a few ULP of each other (a 2^8 spread across the addends separates
     * nothing, a 3-ULP step across the same eight values does — measured in work/probe10.py and
     * work/probe13.py), so raw[k] = 0x3d00 + 3k is the near-ULP row that makes rank-vs-ascending-id
     * accumulation a visible difference. The pairing leg needs the opposite: distinct weights per
     * slot, because indexing the weight array by expert id instead of by rank is invisible when all
     * the weights are equal. Both legs carry their immunity half: the same construction with ids in
     * ascending order, or with ids[k] % K == k, must produce bit-identical words. */
    {
        static uint16_t raw[(size_t)MR_K * MR_H];
        float wpre[MR_K];
        uint16_t wcast[MR_K];
        int ids[MR_K];
        std::vector<uint16_t> a(MR_H), b(MR_H), c(MR_H);
        for (int k = 0; k < MR_K; ++k)
            for (int h = 0; h < MR_H; ++h) raw[(size_t)k * MR_H + h] = (uint16_t)(0x3d00u + 3 * k);
        for (int k = 0; k < MR_K; ++k) { wpre[k] = 1.0f; wcast[k] = mr_f32_to_bf16(wpre[k]); }
        for (int k = 0; k < MR_K; ++k) ids[k] = k;                    /* ascending: immune */
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, a.data());
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_RANK_ACCUM, b.data());
        int d_asc = 0;
        for (int h = 0; h < MR_H; ++h) if (a[h] != b[h]) ++d_asc;
        for (int k = 0; k < MR_K; ++k) ids[k] = MR_K - 1 - k;         /* descending: must fire */
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, a.data());
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_RANK_ACCUM, b.data());
        int d_desc = 0;
        for (int h = 0; h < MR_H; ++h) if (a[h] != b[h]) ++d_desc;
        char det[128];
        snprintf(det, sizeof det, "ascending-id immunity moved %d of %d; descending-id order "
                 "injection moved %d of %d", d_asc, MR_H, d_desc, MR_H);
        probe("order_injection_not_decoration", d_asc == 0 && d_desc > 0, MR_H, det);

        for (int k = 0; k < MR_K; ++k) { wpre[k] = 1.0f + 0.25f * (float)k;
                                         wcast[k] = mr_f32_to_bf16(wpre[k]); }
        for (int k = 0; k < MR_K; ++k) ids[k] = k;                    /* ids[k] % K == k: immune */
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, a.data());
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_W_BY_EXPERT_ID, c.data());
        int p_ok = 0;
        for (int h = 0; h < MR_H; ++h) if (a[h] != c[h]) ++p_ok;
        for (int k = 0; k < MR_K; ++k) ids[k] = MR_K - 1 - k;         /* ids[k] % K != k: must fire */
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_CORRECT, a.data());
        mr_combine_row(raw, wpre, wcast, ids, MR_MODE_W_BY_EXPERT_ID, c.data());
        int p_bad = 0;
        for (int h = 0; h < MR_H; ++h) if (a[h] != c[h]) ++p_bad;
        snprintf(det, sizeof det, "pairing immunity moved %d of %d; mis-pairing moved %d of %d",
                 p_ok, MR_H, p_bad, MR_H);
        probe("pairing_injection_not_decoration", p_ok == 0 && p_bad > 0, MR_H, det);
    }
    /* --- an exact 8/9 boundary tie: the weights agree, the SET does not (A-03's shape) --------- */
    {
        static uint16_t lg[MR_E];
        float p[MR_E], wa[MR_K], wb[MR_K];
        uint16_t ca[MR_K], cb[MR_K], rwa[MR_H], rwb[MR_H];
        int ia[MR_K], ib[MR_K];
        static uint16_t raw[(size_t)MR_K * MR_H];
        for (int i = 0; i < MR_K * MR_H; ++i) raw[i] = fill(i, 5);
        /* nine experts share one logit value; the rest are strictly below it */
        const int tie[9] = {11, 47, 90, 121, 150, 177, 200, 233, 251};
        for (int l = 0; l < MR_E; ++l) lg[l] = mr_f32_to_bf16(-8.0f);
        for (int i = 0; i < 9; ++i) lg[tie[i]] = mr_f32_to_bf16(4.0f);
        for (int i = 0; i < 4; ++i) lg[3 + i] = mr_f32_to_bf16(9.0f);   /* four clear winners */
        mr_softmax_row(lg, MR_MODE_CORRECT, p);
        mr_topk_row(p, MR_MODE_CORRECT, ia);
        mr_topk_row(p, MR_MODE_TIE_HIGH, ib);
        mr_renorm_row(p, ia, MR_MODE_CORRECT, wa, ca);
        mr_renorm_row(p, ib, MR_MODE_CORRECT, wb, cb);
        int w_same = 1;
        for (int k = 0; k < MR_K; ++k) w_same &= (fb(wa[k]) == fb(wb[k]));
        int set_a[MR_E] = {0}, set_b[MR_E] = {0};
        for (int k = 0; k < MR_K; ++k) { set_a[ia[k]] = 1; set_b[ib[k]] = 1; }
        int sets_differ = 0;
        for (int l = 0; l < MR_E; ++l) sets_differ |= (set_a[l] != set_b[l]);
        mr_combine_row(raw, wa, ca, ia, MR_MODE_CORRECT, rwa);
        mr_combine_row(raw, wb, cb, ib, MR_MODE_CORRECT, rwb);
        int routed_diff = 0;
        for (int h = 0; h < MR_H; ++h) routed_diff += (rwa[h] != rwb[h]);
        probe("tie_every_weight_agrees", w_same, MR_K,
              w_same ? "" : "tied experts gave different weights; the fixture is not tied");
        probe("tie_swaps_the_expert_set", sets_differ, MR_K,
              sets_differ ? "" : "the tie-break did not change the set; this fixture does not "
                                "exercise A-03 and must be rebuilt, not dropped");
        probe("tie_changes_the_routed_sum", routed_diff > 0, MR_H, "");
    }
    /* --- R5's cast is idempotent: bf16(wpre) == wcast for every mode that produces both -------- */
    {
        static uint16_t lg[MR_E];
        float p[MR_E], w[MR_K];
        uint16_t c[MR_K];
        for (int l = 0; l < MR_E; ++l) lg[l] = fill(l, 3);
        int modes[4] = {MR_MODE_CORRECT, MR_MODE_SOFTMAX_BF16, MR_MODE_NO_RENORM, MR_MODE_NO_MAX_SUB};
        int bad = 0, tried = 0;
        for (int mi = 0; mi < 4; ++mi) {
            mr_softmax_row(lg, modes[mi], p);
            int ids[MR_K];
            mr_topk_row(p, modes[mi] == MR_MODE_SOFTMAX_BF16 ? MR_MODE_CORRECT : modes[mi], ids);
            mr_renorm_row(p, ids, modes[mi], w, c);
            for (int k = 0; k < MR_K; ++k) bad += (mr_f32_to_bf16(w[k]) != c[k]);
            tried += MR_K;
        }
        probe("r5_cast_idempotent", bad == 0, tried, "");
    }
    return g_fail;
}

/* --------------------------------------------------------------------------------------------- */
static const char *arg_value(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}
static int have_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int rc = 0;
    if (!have_flag(argc, argv, "--no-struct")) rc |= (structural_probes() != 0);

    const char *plan_path = arg_value(argc, argv, "--plan", NULL);
    const char *slab_path = arg_value(argc, argv, "--slab", NULL);
    const char *out_path = arg_value(argc, argv, "--out", NULL);
    if (!plan_path || !slab_path || !out_path) {
        if (!rc) printf("RESULT PASS host structural probes only (no plan given)\n");
        else printf("RESULT FAIL host structural probes\n");
        return rc ? 1 : 0;
    }
    std::vector<struct MrCase> cases;
    long long slab_elems = 0, out_floats = 0;
    if (read_plan(plan_path, &cases, &slab_elems, &out_floats)) return 1;
    std::vector<uint16_t> slab;
    {
        FILE *fh = fopen(slab_path, "rb");
        if (!fh) { printf("FAIL cannot open slab %s: %s\n", slab_path, strerror(errno)); return 1; }
        fseek(fh, 0, SEEK_END);
        long long nb = ftello(fh);
        fseek(fh, 0, SEEK_SET);
        slab.resize((size_t)(nb / 2));
        if (nb && fread(slab.data(), 1, (size_t)nb, fh) != (size_t)nb) {
            printf("FAIL short read on slab\n"); fclose(fh); return 1;
        }
        fclose(fh);
    }
    int checks = 0;
    const int bad = mr_verify_cases(cases.data(), (int)cases.size(), (long long)slab.size(),
                                    out_floats, &checks);
    printf("PLANCHECK host=1 cases=%zu checks_run=%d malformed=%d\n", cases.size(), checks, bad);
    if (bad) { printf("RESULT FAIL host plan verification malformed=%d\n", bad); return 1; }

    std::vector<float> out((size_t)out_floats, mr_bits_to_f32(MR_NAN_BITS));
    struct MrRowState st;
    /* mode 0 first, then the injections, so each injection compares against the same row's
     * reference record exactly as the device driver does */
    for (int mode = 0; mode < MR_MODE_COUNT; ++mode) {
        for (size_t i = 0; i < cases.size(); ++i) {
            if (cases[i].mode != mode) continue;
            struct MrCase c = cases[i];
            mr_row(slab.data(), &c, &st, out.data());   /* buf base: mr_row indexes by dst_off */
            cases[i] = c;
        }
    }
    FILE *of = fopen(out_path, "wb");
    if (!of) { printf("FAIL cannot write host out %s: %s\n", out_path, strerror(errno)); return 1; }
    if (out_floats && fwrite(out.data(), sizeof(float), (size_t)out_floats, of) !=
                          (size_t)out_floats) {
        printf("FAIL short write on host out\n"); fclose(of); return 1;
    }
    if (fflush(of) || ferror(of)) { printf("FAIL flush host out\n"); fclose(of); return 1; }
    fclose(of);
    printf("OUT host=1 file=%s floats=%lld bytes=%lld\n", out_path, out_floats,
           4LL * out_floats);

    FILE *rp = fopen(arg_value(argc, argv, "--report", NULL), "wb");
    if (!rp) { printf("FAIL cannot write host report\n"); return 1; }
    fprintf(rp, "# id\tarm\tmode\tbad_logits\tbad_probs\tbad_wpre\tbad_wcast\tbad_ids\tbad_routed\n");
    int nrows = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const struct MrCase &c = cases[i];
        fprintf(rp, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", c.id, c.arm, c.mode, c.bad[0],
                c.bad[1], c.bad[2], c.bad[3], c.bad[4], c.bad[5]);
        ++nrows;
    }
    fclose(rp);
    printf("REPORT host=1 rows=%d\n", nrows);
    if (nrows != (int)cases.size()) {
        printf("FAIL host report holds %d rows for %zu cases\n", nrows, cases.size());
        return 1;
    }
    if (rc) { printf("RESULT FAIL host structural probes\n"); return 1; }
    printf("RESULT PASS host arm cases=%zu\n", cases.size());
    return 0;
}
