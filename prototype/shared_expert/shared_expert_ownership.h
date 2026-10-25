/* shared_expert_ownership.h — the write-ownership probe, one code path shared by BOTH arms.
 *
 * "The last column is wrong" is not a diagnosis; "stage 4 wrote a word it does not own" is. Each stage
 * of the pipeline owns exactly one region of a case's record, and a kernel that processes an item it was
 * not given — the leaked `base = -1` sentinel being the real historical case — writes one word across a
 * field boundary and corrupts a value that another stage owns. The stage-3 version of this probe earned
 * its keep by firing 43 times (once per case, deterministically) on the build where `se_k_out` let the
 * sentinel through; this file generalises it to every stage so the same instrument covers stages 1 and 2.
 *
 * HOW IT WORKS
 *   The launch set is split into four groups — gateup, down, prob, out — and the output buffer is read
 *   back after every group. Between two consecutive snapshots exactly one group could have written, so
 *   every changed word is attributable to that group, and every changed word outside the group's owned
 *   region is a named violation: stage, case, field, index, and both bit patterns.
 *   Attributable means the group boundaries are real synchronisations, not a hope: the device arm reads
 *   back after each launch (which implies the device is idle), and the host arm's stage loop is already
 *   sequential. It also means GROUPED launch mode only: in per-case mode stage 1 of the next case runs
 *   after stage 4 of this one, so no snapshot separates the stages and the probe says
 *   `attributable=no` rather than printing a tidy zero. A probe that can only report zero is a hope.
 *
 * THE OWNERSHIP TABLE IS DERIVED FROM THE LAYOUT, not typed beside it: the four owned spans must tile
 * one case record exactly (`partition=ok`), so a new field added to the layout without a new table row
 * makes the probe refuse to claim coverage instead of quietly watching less than it appears to.
 *
 * THE INJECTION (`--inject-ownership=N`) is a control, not a defect in the shipping path: it hands stage
 * N a foreign item — a second `base = -1` sentinel for the stages that have none, a channel item for
 * the prob stage — and lets that stage write ONE word outside its own region. Each stage's injection is
 * the same class of mistake the fixed bug was, so each stage proves its own half of the probe can fire.
 */
#ifndef SHARED_EXPERT_OWNERSHIP_H
#define SHARED_EXPERT_OWNERSHIP_H

#include "shared_expert_kernels.h"

#define SE_OWN_STAGES 4
#define SE_OWN_LEAK_BITS 0x4C45414Bu        /* "LEAK": a pattern no stage could compute */

static const char *const se_own_stage[SE_OWN_STAGES] = {"1_gateup", "2_down", "3_prob", "4_out"};
static const char *const se_own_owns[SE_OWN_STAGES] = {"gate|up|act", "raw", "prob", "out"};
static const int se_own_off[SE_OWN_STAGES] = {SE_OFF_GATE, SE_OFF_RAW, SE_OFF_PROB, SE_OFF_OUT};
static const int se_own_len[SE_OWN_STAGES] = {
    SE_OFF_RAW - SE_OFF_GATE,              /* 1536: gate|up|act            */
    SE_OFF_OUT - SE_OFF_RAW,               /* 2048: raw                    */
    SE_CASE_FLOATS - SE_OFF_PROB,          /* 1:    prob                   */
    SE_OFF_PROB - SE_OFF_OUT               /* 2048: out                    */
};

/* the six fields of a record, in layout order, for naming a word in a report */
static const char *const se_own_field[6] = {"gate", "up", "act", "raw", "out", "prob"};
static const int se_own_field_off[6] = {SE_OFF_GATE, SE_OFF_UP, SE_OFF_ACT, SE_OFF_RAW,
                                        SE_OFF_OUT, SE_OFF_PROB};
static const int se_own_field_len[6] = {SE_N_GATE, SE_N_UP, SE_N_ACT, SE_N_RAW, SE_N_OUT, SE_N_PROB};

/* Does stage `st` own word `k` of a case record? */
SE_DEV static int se_owns(int st, int k) {
    return k >= se_own_off[st] && k < se_own_off[st] + se_own_len[st];
}

/* The field a word index falls in, or "?" if the layout and the field table disagree. */
SE_DEV static const char *se_own_where(int k) {
    for (int i = 0; i < 6; ++i)
        if (k >= se_own_field_off[i] && k < se_own_field_off[i] + se_own_field_len[i])
            return se_own_field[i];
    return "?";
}
SE_DEV static int se_own_where_idx(int k) {
    for (int i = 0; i < 6; ++i)
        if (k >= se_own_field_off[i] && k < se_own_field_off[i] + se_own_field_len[i]) return i;
    return -1;
}

/* The table is a claim about the layout: the four owned spans must tile one case record with no gap
 * and no overlap, and every one of the six fields must appear in exactly one stage's owned region. */
SE_DEV static int se_own_partition_ok(void) {
    int seen[6], total = 0;
    for (int i = 0; i < 6; ++i) seen[i] = 0;
    for (int st = 0; st < SE_OWN_STAGES; ++st) {
        total += se_own_len[st];
        for (int k = se_own_off[st]; k < se_own_off[st] + se_own_len[st]; ++k) {
            const int f = se_own_where_idx(k);
            if (f < 0) return 0;                       /* a word in no field: layout drift */
            if (seen[f] != 0 && seen[f] != st + 1) return 0;   /* a field owned by two stages */
            seen[f] = st + 1;
        }
    }
    if (total != SE_CASE_FLOATS) return 0;
    for (int i = 0; i < 6; ++i) if (seen[i] == 0) return 0;
    return 1;
}

typedef struct { int violations, printed; long compared; } SeOwnTally;

/* One group's diff: every word that CHANGED between the two snapshots is attributable to the group that
 * just ran; a changed word it does not own is printed by name. `lines` is counted, never trusted from
 * the caller — a check that printed nothing cannot fail, so the driver re-derives both numbers. */
SE_DEV static void se_own_diff(SeOwnTally *t, const int stage, const SeCase *cases, const int n_c,
                               const float *before, const float *after, FILE *out) {
    for (int i = 0; i < n_c; ++i) {
        const int d = cases[i].dst_off;
        for (int k = 0; k < cases[i].n_floats; ++k) {
            const uint32_t ba = se_bits_of(before[d + k]);
            const uint32_t bb = se_bits_of(after[d + k]);
            ++t->compared;
            if (ba == bb) continue;
            if (se_owns(stage, k)) continue;             /* this stage owns it: a legitimate write */
            fprintf(out, "OWNERSHIP stage=%d group=%s case=%d field=%s idx=%d before=0x%08X "
                         "after=0x%08X\n", stage + 1, se_own_stage[stage], cases[i].index,
                    se_own_where(k), k, ba, bb);
            ++t->violations;
            ++t->printed;
        }
    }
}

/* The per-stage line the driver prints after each group, so the totals can be re-derived by hand. */
SE_DEV static void se_own_stage_line(const int stage, const SeOwnTally *t, FILE *out) {
    fprintf(out, "OWNERSHIPSTAGE stage=%d group=%s owns=%s compared=%ld violations=%d lines=%d\n",
            stage + 1, se_own_stage[stage], se_own_owns[stage], t->compared, t->violations,
            t->printed);
}

/* The injected write: one word, outside the stage's own region, carrying a pattern nothing computes. */
SE_DEV static void se_own_leak(const SeCase *c, float *out, const int stage) {
    static const int leak_at[SE_OWN_STAGES] = {SE_OFF_RAW, SE_OFF_ACT, SE_OFF_RAW, SE_OFF_OUT - 1};
    out[c->dst_off + leak_at[stage]] = se_float_of(SE_OWN_LEAK_BITS);
}

#endif /* SHARED_EXPERT_OWNERSHIP_H */
