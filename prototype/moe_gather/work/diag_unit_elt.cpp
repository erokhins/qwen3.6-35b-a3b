/* diag_unit_elt.cpp — CPU-only witness for WHICH weight-read path is silent.
 *
 * Three candidate causes for an expert record with no value in it, and the words alone cannot tell
 * them apart (mg_nan() is both the poison the driver pre-fills and the value a refused read writes):
 *   (a) mg_resolve_unit returns NULL           -> the case writes NaN and never touches a byte
 *   (b) mg_unit_elt refuses (returns non-zero) -> same NaN, and the width census stays at zero
 *   (c) the read succeeds and decodes 0.0      -> acc really is a zero dot
 * This program calls each function directly on the REAL plan/slab the gate uses and prints the
 * return codes, so the cause is a measurement. It is a diagnostic, not part of the gate.
 *
 * Build (CPU only, no device context, no lock):
 *   g++ -O2 -std=c++11 -Wall -Wextra -ffp-contract=off -I. work/diag_unit_elt.cpp -o work/diag_unit_elt
 */
#include "moe_gather_kernels.h"

int main(int argc, char **argv) {
    const char *plan_path = argc > 1 ? argv[1] : "work/plan.tsv";
    const char *slab_path = argc > 2 ? argv[2] : "work/slab.bin";
    const char *expect_path = argc > 3 ? argv[3] : "work/expect.f32";
    size_t sb = 0, eb = 0;
    FILE *fh = fopen(slab_path, "rb");
    if (!fh) { printf("no slab\n"); return 1; }
    fseek(fh, 0, SEEK_END); const long sn = ftell(fh); fseek(fh, 0, SEEK_SET);
    char *slab = (char *)malloc((size_t)sn);
    if (fread(slab, 1, (size_t)sn, fh) != (size_t)sn) { return 1; }
    fclose(fh);
    fh = fopen(expect_path, "rb");
    if (!fh) { return 1; }
    fseek(fh, 0, SEEK_END); const long en = ftell(fh); fseek(fh, 0, SEEK_SET);
    char *expb = (char *)malloc((size_t)en);
    if (fread(expb, 1, (size_t)en, fh) != (size_t)en) { return 1; }
    fclose(fh);

    MgPlan p;
    memset(&p, 0, sizeof(p));
    int cu = 0, cr = 0, cc = 0, cb = 0;
    {
        FILE *pf = fopen(plan_path, "rb");
        if (!pf) return 1;
        char line[4096];
        while (fgets(line, sizeof(line), pf)) {
            if (line[0] == 'U') cu++; else if (line[0] == 'R') cr++;
            else if (line[0] == 'B') cb++; else if (line[0] == 'C') cc++;
        }
        fclose(pf);
    }
    p.units = (MgUnit *)calloc((size_t)cu, sizeof(MgUnit));
    p.rows = (MgRow *)calloc((size_t)cr, sizeof(MgRow));
    p.cases = (MgCase *)calloc((size_t)cc, sizeof(MgCase));
    p.blocks = (MgBlock *)calloc((size_t)cb, sizeof(MgBlock));
    int pu = 0, pr = 0, pc = 0, pb = 0;
    if (mg_read_plan(plan_path, &p, cu, cr, cc, cb, &pu, &pr, &pc, &pb)) { printf("plan parse fail\n"); return 1; }
    p.n_units = pu; p.n_rows = pr; p.n_cases = pc; p.n_blocks = pb;
    p.slab = slab; p.slab_bytes = (size_t)sn;
    p.expect = (const uint32_t *)(void *)expb;
    printf("PLAN units=%d rows=%d cases=%d blocks=%d slab_bytes=%zu\n", pu, pr, pc, pb, sn);

    int resolved = 0, refused = 0, decoded = 0, decoded_zero = 0;
    int cases_checked = 0;
    for (int i = 0; i < pc; ++i) {
        const MgCase *c = &p.cases[i];
        if (c->kind != MG_CASE_EXPERT || c->mode != MG_MODE_CORRECT) continue;
        cases_checked++;
        if (cases_checked > 3) break;
        const int roles[3] = {MG_ROLE_GATE, MG_ROLE_UP, MG_ROLE_DOWN};
        for (int ri = 0; ri < 3; ++ri) {
            const MgUnit *u = mg_resolve_unit(&p, c, roles[ri], c->mode);
            if (!u) { printf("CASE %d role %d RESOLVE=NULL\n", i, roles[ri]); continue; }
            resolved++;
            const int n_rows = u->rows, n_cols = (roles[ri] == MG_ROLE_DOWN) ? MG_I : MG_H;
            int rr = 0, dd = 0, zz = 0;
            for (int row = 0; row < n_rows; ++row) {
                for (int col = 0; col < n_cols; ++col) {
                    float w = 12345.0f, w32 = 12345.0f;
                    const int rc = mg_unit_elt(&p, u, roles[ri], row, col, n_cols, c->mode, &w, &w32);
                    if (rc) { rr++; continue; }
                    dd++;
                    if (mg_bits_of(w) == 0u || mg_bits_of(w) == 0x80000000u) zz++;
                }
            }
            printf("CASE %d role %d unit=%d expert=%d kind=%d rows=%d cols=%d "
                   "refused=%d decoded=%d decoded_zero=%d\n",
                   i, roles[ri], (int)(u - p.units), u->expert, u->kind, n_rows, n_cols, rr, dd, zz);
            refused += rr; decoded += dd; decoded_zero += zz;
        }
    }
    printf("WITNESS expert_correct_cases=%d resolved=%d refused=%d decoded=%d decoded_zero=%d\n",
           cases_checked, resolved, refused, decoded, decoded_zero);
    /* and what one element really decodes to, printed as a number, for eyeball sanity */
    {
        const MgCase *c = &p.cases[0];
        const MgUnit *u = mg_resolve_unit(&p, c, MG_ROLE_GATE, c->mode);
        float w = 0.0f;
        const int rc = mg_unit_elt(&p, u, MG_ROLE_GATE, 0, 0, MG_H, c->mode, &w, 0);
        printf("SAMPLE case0 gate unit elt(0,0) rc=%d value=%.9g\n", rc, (double)w);
    }

    /* The closing witness: the element reads above succeed, so run the SAME case through the
     * function the drivers call and count what it writes. If the elements decode but the record is
     * all MG_NAN_BITS, the defect is in the caller's use of the return code, not in the reader. */
    {
        MgCounters ct;
        memset(&ct, 0, sizeof(ct));
        const MgCase *c = &p.cases[0];
        float *rec = (float *)malloc((size_t)MG_EXPERT_FLOATS * 4);
        for (int i = 0; i < MG_EXPERT_FLOATS; ++i) rec[i] = mg_nan();
        mg_run_case(&p, c, 0, 0, 0, rec, 0, 1, &ct);
        int nanw = 0, zerow = 0, other = 0;
        for (int i = 0; i < MG_EXPERT_FLOATS; ++i) {
            const uint32_t b = mg_bits_of(rec[i]);
            if (b == MG_NAN_BITS) nanw++;
            else if (b == 0u || b == 0x80000000u) zerow++;
            else other++;
        }
        printf("RUNCASE case0 span=%d nan=%d zero=%d other=%d counter_bad=%d counter_nan=%d "
               "counter_route_n=%d\n", MG_EXPERT_FLOATS, nanw, zerow, other,
               ct.bad, ct.nan, ct.route_n);
        free(rec);
    }
    return 0;
}
