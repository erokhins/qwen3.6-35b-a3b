/* lm_head_plan.h — plan parsing, structural verification and the host-side runner, shared by
 * `lm_head.hip` (hipcc) and `host_probe.cpp` (g++).
 *
 * WHY this is a header and not two copies: the plan is the contract between gate.py and both
 * drivers. If the device driver and the host twin each parsed it their own way, a green
 * host-vs-device agreement would only prove the two parsers agreed with each other
 * (`pitfalls/40-gates-and-admissions.md` §"Two self-consistent layouts joined by a branch are
 * invisible to both sides" — there the fix was ONE address expression shared by writer and reader;
 * this is the same rule applied to a plan reader). So: one parser, one verifier, one host runner,
 * included by both builds. The only thing the two drivers differ in is how they execute a case.
 *
 * Correctness only. Nothing in this file takes a timer, runs a warm loop, or prints a rate.
 */
#ifndef LM_HEAD_PLAN_H
#define LM_HEAD_PLAN_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "lm_head_kernels.h"

/* Record shapes, as asserted by gate.py. A field-count change is a contract change and must move in
 * one commit with the gate (`pitfalls/40` §"a default in a gate driver and the default in the binary
 * under test must move in one commit"). */
#define LMH_F_P 10     /* 'P' + 9  */
#define LMH_F_T 10     /* 'T' + 9  */
#define LMH_F_C 20     /* 'C' + 19 */
#define LMH_BLOCK 256  /* threads per block; a multiple of LMH_SPLIT */

struct Plan {
    /* the 'P' header — the geometry both drivers must agree the run has */
    int have_p;
    int head_dim, h, vocab, n_hidden, ids_count;
    long long out_floats, slab_bytes;
    int n_tables_decl, n_cases_decl;

    std::vector<LmhTable> tables;
    std::vector<LmhCase> cases;

    Plan() : have_p(0), head_dim(0), h(0), vocab(0), n_hidden(0), ids_count(0), out_floats(0),
             slab_bytes(0), n_tables_decl(0), n_cases_decl(0) {}
};

/* tab-separated, no collapsing of adjacent tabs, trailing newline stripped */
static inline int lmh_split(char *line, char **out, int max) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
    int c = 0;
    char *p = line;
    while (c < max) {
        out[c++] = p;
        char *t = strchr(p, '\t');
        if (!t) break;
        *t = '\0';
        p = t + 1;
    }
    /* trailing empties are not counted: a line that stopped early must fail the field count, not
     * silently look like a line with empty tail fields */
    while (c > 1 && out[c - 1][0] == '\0') --c;
    return c;
}

static inline int lmh_read_plan(const char *path, Plan *plan, int *n_p, int *n_t, int *n_c) {
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("FAIL cannot open plan %s: %s\n", path, strerror(errno));
        return 1;
    }
    char line[4096];
    char *fld[32];
    int ln = 0;
    while (fgets(line, sizeof(line), f)) {
        ++ln;
        if (line[0] == '#' || line[0] == '\n') continue;
        int nf = lmh_split(line, fld, 32);
        if (strcmp(fld[0], "P") == 0) {
            if (nf != LMH_F_P) {
                printf("FAIL plan %s line %d: P has %d fields, expected %d\n", path, ln, nf, LMH_F_P);
                fclose(f);
                return 1;
            }
            if (plan->have_p) {
                printf("FAIL plan %s line %d: a second P record\n", path, ln);
                fclose(f);
                return 1;
            }
            plan->have_p = 1;
            plan->head_dim = (int)strtol(fld[1], 0, 10);
            plan->h = (int)strtol(fld[2], 0, 10);
            plan->vocab = (int)strtol(fld[3], 0, 10);
            plan->n_hidden = (int)strtol(fld[4], 0, 10);
            plan->ids_count = (int)strtol(fld[5], 0, 10);
            plan->out_floats = strtoll(fld[6], 0, 10);
            plan->slab_bytes = strtoll(fld[7], 0, 10);
            plan->n_tables_decl = (int)strtol(fld[8], 0, 10);
            plan->n_cases_decl = (int)strtol(fld[9], 0, 10);
            ++*n_p;
        } else if (strcmp(fld[0], "T") == 0) {
            if (nf != LMH_F_T) {
                printf("FAIL plan %s line %d: T has %d fields, expected %d\n", path, ln, nf, LMH_F_T);
                fclose(f);
                return 1;
            }
            LmhTable t;
            memset(&t, 0, sizeof(t));
            t.idx = (int)strtol(fld[1], 0, 10);
            t.enc = (int)strtol(fld[2], 0, 10);
            t.role = (int)strtol(fld[3], 0, 10);
            t.base = strtoll(fld[4], 0, 10);
            t.nbytes = strtoll(fld[5], 0, 10);
            t.v_rows = strtoll(fld[6], 0, 10);
            t.h = (int)strtol(fld[7], 0, 10);
            t.peer = (int)strtol(fld[8], 0, 10);
            t.real = (int)strtol(fld[9], 0, 10);
            plan->tables.push_back(t);
            ++*n_t;
        } else if (strcmp(fld[0], "C") == 0) {
            if (nf != LMH_F_C) {
                printf("FAIL plan %s line %d: C has %d fields, expected %d\n", path, ln, nf, LMH_F_C);
                fclose(f);
                return 1;
            }
            LmhCase c;
            memset(&c, 0, sizeof(c));
            c.idx = (int)strtol(fld[1], 0, 10);
            c.table = (int)strtol(fld[2], 0, 10);
            c.path = (int)strtol(fld[3], 0, 10);
            c.batching = (int)strtol(fld[4], 0, 10);
            c.b0 = (int)strtol(fld[5], 0, 10);
            c.bcount = (int)strtol(fld[6], 0, 10);
            c.row_lo = (int)strtol(fld[7], 0, 10);
            c.nrows = (int)strtol(fld[8], 0, 10);
            c.ids_off = (int)strtol(fld[9], 0, 10);
            c.n_ids = (int)strtol(fld[10], 0, 10);
            c.assoc = (int)strtol(fld[11], 0, 10);
            c.mode = (int)strtol(fld[12], 0, 10);
            c.out_off = (int)strtol(fld[13], 0, 10);
            c.n_out = (int)strtol(fld[14], 0, 10);
            c.cmp_off = (int)strtol(fld[15], 0, 10);
            c.grp = (int)strtol(fld[16], 0, 10);
            c.expect = (int)strtol(fld[17], 0, 10);
            c.on_device = (int)strtol(fld[18], 0, 10);
            c.trace = (int)strtol(fld[19], 0, 10);
            plan->cases.push_back(c);
            ++*n_c;
        } else {
            printf("FAIL plan %s line %d: unknown record type %s\n", path, ln, fld[0]);
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

/* The output slot stride for a batch row, in ONE place. lmh_store writes with it, lmh_case_out_span
 * sizes the region with it, the report rolls the region up with it, and gate.py mirrors the same
 * expression — so "batched GEMV gave batch b its own row of logits" is one formula in four places
 * rather than four formulas that happen to agree today. */
static inline int lmh_case_slot(const Plan &plan, const LmhCase &c) {
    return (c.mode == LMH_MODE_BATCH_SLOT_STRIDE) ? plan.h : c.n_out;
}

/* The row stride a case believes in, and the lane split — both derived here so the verifier and the
 * runner cannot disagree, and so `h_used % nlanes` is a checked fact rather than an assumption. */
static inline int lmh_case_h_used(const Plan &plan, const LmhCase &c) {
    return (c.mode == LMH_MODE_ROW_STRIDE_HEAD_DIM) ? plan.head_dim : plan.h;
}
static inline int lmh_case_nlanes(const LmhCase &c) {
    return (c.assoc == LMH_ASSOC_SPLIT8) ? LMH_SPLIT : 1;
}
static inline long long lmh_case_n_items(const LmhCase &c) {
    return (long long)c.n_out * lmh_case_nlanes(c);
}
/* How much of the plan the device arm is expected to execute. Printed by both drivers and required
 * by gate.py to equal the number of on_device=1 cases, so a cut is a declared population and not a
 * kernel that quietly never ran (pitfalls/40 §"a route no fixture reaches has no tests"). */
static inline int lmh_count_device_cases(const Plan &p) {
    int n = 0;
    for (size_t i = 0; i < p.cases.size(); ++i)
        if (p.cases[i].on_device) ++n;
    return n;
}
static inline long long lmh_case_out_span(const Plan &plan, const LmhCase &c) {
    return (long long)(c.bcount - 1) * lmh_case_slot(plan, c) + c.n_out;
}

static inline void lmh_trace_init(LmhTrace *t) {
    memset(t, 0, sizeof(*t));
    t->min_off = 0xFFFFFFFFu;
}

/* Every structural fact a plan must satisfy, checked before any byte is read. `why` names the
 * failure so a red run says which invariant broke instead of exiting 1. */
static inline int lmh_verify_plan(const Plan &plan, int expect_tables, int expect_cases,
                                  long long slab_bytes, long long hidden_floats, int ids_count,
                                  int expect_head_dim, int expect_h, int expect_vocab) {
    int bad = 0;
#define LMH_BAD(...) do { ++bad; printf("FAIL verify "); printf(__VA_ARGS__); printf("\n"); } while (0)
    if (!plan.have_p) LMH_BAD("no P record");
    if ((int)plan.tables.size() != plan.n_tables_decl)
        LMH_BAD("P declares %d tables, parsed %zu", plan.n_tables_decl, plan.tables.size());
    if ((int)plan.cases.size() != plan.n_cases_decl)
        LMH_BAD("P declares %d cases, parsed %zu", plan.n_cases_decl, plan.cases.size());
    if (expect_tables >= 0 && (int)plan.tables.size() != expect_tables)
        LMH_BAD("--expect-tables says %d, parsed %zu", expect_tables, plan.tables.size());
    if (expect_cases >= 0 && (int)plan.cases.size() != expect_cases)
        LMH_BAD("--expect-cases says %d, parsed %zu", expect_cases, plan.cases.size());
    if (plan.slab_bytes != slab_bytes)
        LMH_BAD("P says slab_bytes=%lld, the file is %lld", plan.slab_bytes, slab_bytes);
    if (plan.ids_count != ids_count)
        LMH_BAD("P says ids_count=%d, the file holds %d", plan.ids_count, ids_count);
    /* The hidden slab's size is a fact about the plan AND about the file, and the batch-slice
     * arithmetic (b0 + bcount <= n_hidden) is only meaningful if the slab really holds n_hidden rows
     * of h floats. Checked here so a truncated hidden file cannot silently shrink the batch. */
    if (hidden_floats != (long long)plan.n_hidden * (long long)plan.h)
        LMH_BAD("hidden file holds %lld floats; P says n_hidden=%d x h=%d = %lld", hidden_floats,
                plan.n_hidden, plan.h, (long long)plan.n_hidden * (long long)plan.h);
    if (expect_head_dim >= 0 && plan.head_dim != expect_head_dim)
        LMH_BAD("P head_dim=%d, the caller's config says %d", plan.head_dim, expect_head_dim);
    if (expect_h >= 0 && plan.h != expect_h)
        LMH_BAD("P h=%d, the caller's config says %d", plan.h, expect_h);
    if (expect_vocab >= 0 && plan.vocab != expect_vocab)
        LMH_BAD("P vocab=%d, the caller's config says %d", plan.vocab, expect_vocab);
    if (plan.h <= 0 || plan.head_dim <= 0 || plan.vocab <= 0)
        LMH_BAD("non-positive geometry h=%d head_dim=%d vocab=%d", plan.h, plan.head_dim,
                plan.vocab);
    /* head_dim is the wrong-geometry injection's fuel; if it ever equalled h that mode would be
     * unfireable, and the gate would be holding a control that cannot fire. */
    if (plan.head_dim >= plan.h)
        LMH_BAD("head_dim=%d must be < h=%d or LMH_MODE_ROW_STRIDE_HEAD_DIM cannot fire",
                plan.head_dim, plan.h);

    for (size_t i = 0; i < plan.tables.size(); ++i) {
        const LmhTable &t = plan.tables[i];
        if (t.idx != (int)i) LMH_BAD("table %zu has idx=%d", i, t.idx);
        if (t.enc < 0 || t.enc >= LMH_ENC_COUNT) LMH_BAD("table %d bad enc %d", t.idx, t.enc);
        if (t.role < 0 || t.role >= LMH_ROLE_COUNT) LMH_BAD("table %d bad role %d", t.idx, t.role);
        if (t.h != plan.h) LMH_BAD("table %d h=%d != P h=%d", t.idx, t.h, plan.h);
        if (t.v_rows <= 0) LMH_BAD("table %d v_rows=%lld", t.idx, t.v_rows);
        const long long want = (t.enc == LMH_ENC_BF16)
                                   ? t.v_rows * (long long)t.h * LMH_BF16_BYTES
                                   : (t.v_rows * ((long long)t.h / LMH_Q8_0_ELTS)) * LMH_Q8_0_BYTES;
        if (t.nbytes != want)
            LMH_BAD("table %d nbytes=%lld, the encoding says %lld for %lld rows of %d",
                    t.idx, t.nbytes, want, t.v_rows, t.h);
        if (t.base < 0 || t.base + t.nbytes > plan.slab_bytes)
            LMH_BAD("table %d [%lld,%lld) outside the slab (%lld)", t.idx, t.base,
                    t.base + t.nbytes, plan.slab_bytes);
        if (t.h % LMH_Q8_0_ELTS != 0 && t.enc == LMH_ENC_Q8_0)
            LMH_BAD("table %d h=%d is not a multiple of the Q8_0 block size", t.idx, t.h);
        if (t.peer >= (int)plan.tables.size()) LMH_BAD("table %d peer=%d out of range", t.idx, t.peer);
    }
    /* Peer extents: either the SAME bytes (the tied control) or DISJOINT from the table. A peer that
     * partially overlaps would make "did this read leave its table" unanswerable, and gate.py builds
     * both shapes from the file offsets, so this is a fact about the plan, not a wish. */
    for (size_t i = 0; i < plan.tables.size(); ++i) {
        const LmhTable &t = plan.tables[i];
        if (t.peer < 0) continue;
        const LmhTable &p = plan.tables[t.peer];
        const int same = (p.base == t.base) && (p.nbytes == t.nbytes);
        const int disjoint = (p.base + p.nbytes <= t.base) || (t.base + t.nbytes <= p.base);
        if (!same && !disjoint)
            LMH_BAD("table %d and its peer %d partially overlap: [%lld,%lld) vs [%lld,%lld)",
                    t.idx, p.idx, t.base, t.base + t.nbytes, p.base, p.base + p.nbytes);
    }

    long long max_out = 0;
    int napplicable_bad = 0;
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const LmhCase &c = plan.cases[i];
        if (c.idx != (int)i) LMH_BAD("case %zu has idx=%d", i, c.idx);
        if (c.table < 0 || c.table >= (int)plan.tables.size()) {
            LMH_BAD("case %d table=%d out of range", c.idx, c.table);
            continue;
        }
        const LmhTable &t = plan.tables[c.table];
        if (c.path < 0 || c.path >= LMH_PATH_COUNT) LMH_BAD("case %d bad path %d", c.idx, c.path);
        if (c.batching < 0 || c.batching >= LMH_BATCH_COUNT)
            LMH_BAD("case %d bad batching %d", c.idx, c.batching);
        if (c.assoc < 0 || c.assoc >= LMH_ASSOC_COUNT) LMH_BAD("case %d bad assoc %d", c.idx, c.assoc);
        if (c.mode < 0 || c.mode >= LMH_MODE_COUNT) LMH_BAD("case %d bad mode %d", c.idx, c.mode);
        if (c.expect < 0 || c.expect >= LMH_EXP_COUNT)
            LMH_BAD("case %d bad expect %d", c.idx, c.expect);
        if (c.on_device != 0 && c.on_device != 1) LMH_BAD("case %d on_device=%d", c.idx, c.on_device);
        if (c.trace != 0 && c.trace != 1) LMH_BAD("case %d trace=%d", c.idx, c.trace);
        if (c.bcount < 1 || c.bcount > LMH_MAX_BATCH)
            LMH_BAD("case %d bcount=%d outside [1,%d]", c.idx, c.bcount, LMH_MAX_BATCH);
        if (c.b0 < 0 || c.b0 + c.bcount > plan.n_hidden)
            LMH_BAD("case %d batch slice [%d,%d) outside n_hidden=%d", c.idx, c.b0,
                    c.b0 + c.bcount, plan.n_hidden);
        if (c.path == LMH_PATH_DENSE) {
            if (c.n_out != c.nrows) LMH_BAD("case %d DENSE n_out=%d != nrows=%d", c.idx, c.n_out, c.nrows);
            if (c.nrows <= 0) LMH_BAD("case %d DENSE nrows=%d", c.idx, c.nrows);
            /* DENSE rows must be inside the table: the successor-row injection is a GATHER case, so a
             * DENSE out-of-range row is always a plan bug. */
            if ((long long)c.row_lo + c.nrows > t.v_rows)
                LMH_BAD("case %d DENSE rows [%d,%d) outside v_rows=%lld", c.idx, c.row_lo,
                        c.row_lo + c.nrows, t.v_rows);
        } else {
            if (c.n_out != c.n_ids) LMH_BAD("case %d GATHER n_out=%d != n_ids=%d", c.idx, c.n_out, c.n_ids);
            if (c.n_ids <= 0) LMH_BAD("case %d GATHER n_ids=%d", c.idx, c.n_ids);
            if (c.ids_off < 0 || c.ids_off + c.n_ids > plan.ids_count)
                LMH_BAD("case %d ids [%d,%d) outside ids_count=%d", c.idx, c.ids_off,
                        c.ids_off + c.n_ids, plan.ids_count);
        }
        const int h_used = lmh_case_h_used(plan, c);
        const int nlanes = lmh_case_nlanes(c);
        if (h_used % nlanes != 0)
            LMH_BAD("case %d h_used=%d not divisible by nlanes=%d", c.idx, h_used, nlanes);
        if (t.enc == LMH_ENC_Q8_0 && (h_used / nlanes) % LMH_Q8_0_ELTS != 0)
            LMH_BAD("case %d Q8_0 lane slice %d is not a whole number of 32-element blocks",
                    c.idx, h_used / nlanes);
        if (!lmh_mode_applicable(&c, &t)) ++napplicable_bad;
        const long long end = c.out_off + lmh_case_out_span(plan, c);
        if (c.out_off < 0 || end > plan.out_floats)
            LMH_BAD("case %d out region [%d,%lld) outside out_floats=%lld", c.idx, c.out_off, end,
                    plan.out_floats);
        if (end > max_out) max_out = end;
        if (c.cmp_off >= 0 && (long long)c.cmp_off + c.n_out > plan.out_floats)
            LMH_BAD("case %d cmp_off=%d + n_out outside out", c.idx, c.cmp_off);
        if (c.trace && c.expect == LMH_EXP_BYTES && c.cmp_off < 0)
            LMH_BAD("case %d is judged on bytes but has no baseline to compare against", c.idx);
    }
    if (napplicable_bad)
        LMH_BAD("%d case(s) carry a mode their encoding/path does not support", napplicable_bad);
    if (plan.out_floats < max_out)
        LMH_BAD("out_floats=%lld is below the largest region any case needs, %lld", plan.out_floats,
                max_out);
#undef LMH_BAD
    return bad ? 1 : 0;
}

/* ---- the host runner --------------------------------------------------------------------------- *
 * The same lmh_item / lmh_combine / lmh_store the device kernel calls, driven by an ordinary loop.
 * Called by host_probe.cpp for every case and by lm_head.hip for the --check path's arithmetic
 * self-tests, so "the host arm" and "the driver's own idea of the answer" are the same code. */
static inline void lmh_run_case_host(const LmhCtx &ctx, const LmhCase &c, LmhTrace *tr,
                                     long long span) {
    const int nlanes = lmh_case_nlanes(c);
    if (!lmh_mode_applicable(&c, &ctx.tabs[c.table])) {
        /* Loud, not stale: every word this case owns becomes NaN, so an inapplicable (mode, encoding,
         * path) pair can never be mistaken for a correct result. */
        lmh_nan_fill(ctx.out + (long long)c.out_off, span);
        return;
    }
    for (int jj = 0; jj < c.n_out; ++jj) {
        float part[LMH_MAX_BATCH][LMH_SPLIT];
        for (int lane = 0; lane < nlanes; ++lane) {
            float acc[LMH_MAX_BATCH];
            lmh_item(&ctx, &c, jj, lane, nlanes, tr, acc);
            for (int b = 0; b < c.bcount; ++b) part[b][lane] = acc[b];
        }
        for (int b = 0; b < c.bcount; ++b) {
            const float v = lmh_combine(part[b], nlanes);
            lmh_store(&ctx, &c, b, jj, v, tr);
        }
    }
}

/* Load a whole file into a vector. Returns 1 on failure. */
static inline int lmh_load_file(const char *path, std::vector<uint8_t> *dst) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("FAIL cannot open %s: %s\n", path, strerror(errno));
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long long n = ftello(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        printf("FAIL ftello %s: %s\n", path, strerror(errno));
        fclose(f);
        return 1;
    }
    dst->resize((size_t)n);
    if (n && fread(dst->data(), 1, (size_t)n, f) != (size_t)n) {
        printf("FAIL short read on %s\n", path);
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

/* ---- the report ------------------------------------------------------------------------------- *
 * One 'R' line per case: the plan fields echoed back (so gate.py can catch a driver that read the
 * plan differently from the gate), the driver's own roll-up of the words it wrote, and the read-set
 * trace. gate.py recomputes oxor/osum from out.f32 and every trace field from the slab bytes it
 * wrote itself, so no number in this file is believed on the driver's authority.
 */
/* LMH_F_R is the tab-separated TOKEN COUNT of one R line, record letter included — the same
 * convention LMH_F_P documents as "'P' + 9". It used to be a number only gate.py checked, which is
 * how it drifted: the value in the last revision (31) was neither the token count (33) nor the data
 * field count (32). It is now witnessed, not believed: lmh_driver_selftest writes one real R line,
 * counts its fields, and requires LMH_F_R; and the names printed in the header come from
 * lmh_r_fields[], whose length is asserted against the same constant. Add a column to the writer
 * without adding its name, or to the constant without the writer, and the self-test goes red. */
#define LMH_F_R 34     /* 'R' + 33 */
static const char *const lmh_r_fields[] = {
    "idx", "table", "role", "enc", "path", "batching", "b0", "bcount", "row_lo", "nrows",
    "n_ids", "assoc", "mode", "expect", "on_device", "trace", "n_out", "n_items", "nlanes",
    "oxor", "osum", "bytes", "loads", "xorb", "vsum", "bsum", "lo", "hi", "cross", "peer_hits",
    "lanes", "bover", "launches"};

/* THE CANONICAL TEXT FORM OF ONE R FIELD — the producer half of the one rule the digest columns live
 * by. Two spellings exist and each field is assigned exactly one of them:
 *
 *   "hex"  0x + exactly 8 UPPER-CASE hex digits   — for a BIT IMAGE (oxor, xorb).
 *   "dec"  0 | [1-9][0-9]*                        — for a COUNT or an INTEGER SUM (vsum, bsum, bytes,
 *                                                   loads, lo, hi, cross, peer_hits, lanes, bover,
 *                                                   launches, and every echoed plan field).
 *   "real" what printf %.9g emits                  — for osum, which is a sum of VALUES.
 *
 * Why a bit image is hex and everything else is decimal, and why the prefix is mandatory: the bug
 * this closes is a producer printing `%u` while the parser read the column as hex — 4278190080 and
 * 0xFF400000 are the SAME 32 bits spelled two ways, and both parse to a plausible number, so the
 * disagreement is invisible to any check that only compares values (`pitfalls/40` §"two self-consistent
 * layouts"). An explicit `0x` puts the radix IN the string: a decimal bit image is then not a
 * plausible-but-wrong number, it is a token the parser's form check rejects. vsum and bsum stay
 * decimal because they are read as numbers, not as bits — `0x123456789ABC` invites a bit-wise reading
 * of a value that only means anything summed, and the driver's own contract check parses that column
 * with strtoull(base 10) below. gate.py scrapes THIS array and refuses any token that does not match
 * its declared form, so the writer and the parser cannot each hold their own ruler; the driver's
 * self-test witnesses the writer against the same table (block 11). */
#define LMH_RF_HEX "hex"
#define LMH_RF_DEC "dec"
#define LMH_RF_REAL "real"

/* The bit-image columns, named. These two are the only fields whose text is a BIT PATTERN, so they
 * are the only fields whose radix a reader can get wrong without noticing: 0xFF400000 spelled as
 * 4278190080 still looks like a number. Everything else in the R line is a count or a sum read as a
 * number, and the one exception among the sums is osum, which adds decoded VALUES rather than bits. */
static const char *const lmh_r_hex_fields[] = {"oxor", "xorb"};
static const char *const lmh_r_real_fields[] = {"osum"};

/* The form table, DERIVED from the field names rather than counted out by hand: a 33-entry literal
 * list is exactly the constant that silently misaligns when a column is added, and LMH_F_R drifted
 * once already. The rule is one line per family — a named bit image is hex, a named value-sum is a
 * real, anything else is canonical decimal. lmh_r_fmt[] below is the text the gate scrapes, and the
 * driver self-test requires the two to agree field by field, so neither the list nor the rule can
 * drift alone. */
static inline int lmh_name_in(const char *const *v, int n, const char *name) {
    for (int i = 0; i < n; ++i) if (strcmp(v[i], name) == 0) return 1;
    return 0;
}
static inline const char *lmh_r_fmt_of(int i) {
    const char *nm = lmh_r_fields[i];
    if (lmh_name_in(lmh_r_hex_fields, (int)(sizeof(lmh_r_hex_fields) / sizeof(lmh_r_hex_fields[0])),
                    nm)) return LMH_RF_HEX;
    if (lmh_name_in(lmh_r_real_fields,
                    (int)(sizeof(lmh_r_real_fields) / sizeof(lmh_r_real_fields[0])), nm))
        return LMH_RF_REAL;
    return LMH_RF_DEC;
}

static const char *const lmh_r_fmt[] = {
    LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC,
    LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC,
    LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC,
    LMH_RF_HEX,                                     /* oxor: a bit image */
    LMH_RF_REAL,                                    /* osum: a sum of values */
    LMH_RF_DEC, LMH_RF_DEC,
    LMH_RF_HEX,                                     /* xorb: a bit image */
    LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC,
    LMH_RF_DEC, LMH_RF_DEC, LMH_RF_DEC};            /* lanes, bover, launches */

/* One uint32 rendered in the canonical bit-image form. Hand-rolled rather than sprintf("%08X") so the
 * prefix and the digit width are one expression in one place the self-test can also call. */
static inline const char *lmh_hex32(char *buf, uint32_t v) {
    static const char digits[16] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
    buf[0] = '0'; buf[1] = 'x';
    for (int i = 0; i < 8; ++i) {
        const unsigned shift = 4u * (7u - (unsigned)i);
        buf[2 + i] = digits[(v >> shift) & 0xFu];
    }
    buf[10] = '\0';
    return buf;
}

/* The validator the self-test runs its OWN output through. Deliberately not the writer: if the
 * witness called lmh_hex32 and asked whether the result was non-empty it would prove nothing. This
 * accepts a token only if it is the canonical spelling of its kind, so a producer that drifted to a
 * bare `%u` (or to lower case, or to a leading zero) fails here rather than agreeing with itself. */
static inline int lmh_fmt_ok(const char *kind, const char *s) {
    if (!kind || !s || !*s) return 0;
    if (strcmp(kind, LMH_RF_HEX) == 0) {
        if (strlen(s) != 10 || s[0] != '0' || s[1] != 'x') return 0;
        for (int i = 2; i < 10; ++i) {
            const char c = s[i];
            if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F'))) return 0;
        }
        return 1;
    }
    if (strcmp(kind, LMH_RF_DEC) == 0) {
        if (s[0] == '0') return s[1] == '\0';        /* no 007, no -0, no +0 */
        if (s[0] < '1' || s[0] > '9') return 0;      /* no sign, no prefix, no leading zero */
        for (const char *p = s + 1; *p; ++p) if (*p < '0' || *p > '9') return 0;
        return 1;
    }
    if (strcmp(kind, LMH_RF_REAL) == 0) {
        char *end = NULL;
        const double d = strtod(s, &end);
        if (!end || *end != '\0' || end == s) return 0;
        return d == d || d != d;                     /* a finite number, or a NaN: both are printable */
    }
    return 0;                                        /* an undeclared kind is never legal */
}

static inline int lmh_write_report(const char *path, const Plan &plan, const LmhTrace *tr,
                                   const float *out, const long long *n_launches, int device_run) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        printf("FAIL cannot write report %s: %s\n", path, strerror(errno));
        return 1;
    }
    /* one header line generated from the field names, so a column and its name cannot drift apart */
    fputs("#R", f);
    for (int i = 0; i < (int)(sizeof(lmh_r_fields) / sizeof(lmh_r_fields[0])); ++i) {
        fputs("\t", f);
        fputs(lmh_r_fields[i], f);
    }
    fputs("\n", f);
    fputs("# oxor|osum roll the words this case wrote; xorb|vsum|bsum roll the weight elements it "
          "loaded; lo|hi|cross|peer_hits|lanes are the read-set; bover counts writes that would "
          "have left `out`; launches is blocks (device) or 1 (host)\n", f);
    long long printed = 0;
    for (size_t i = 0; i < plan.cases.size(); ++i) {
        const LmhCase &c = plan.cases[i];
        const LmhTable &t = plan.tables[c.table];
        const int slot = lmh_case_slot(plan, c);
        uint32_t oxor = 0;
        double osum = 0.0;
        for (int b = 0; b < c.bcount; ++b)
            for (int j = 0; j < c.n_out; ++j) {
                const float v = out[(long long)c.out_off + (long long)b * slot + j];
                oxor ^= lmh_float_to_bits(v);
                osum += (double)v;
            }
        const LmhTrace &r = tr[c.idx];
        /* The two bit-image columns go through lmh_hex32 in the canonical 0x + 8 upper-case hex form;
         * every other numeric column is plain decimal. The radix is IN the text, so the parser
         * downstream cannot guess it and cannot silently mis-read one. */
        char hxo[16], hxb[16];
        fprintf(f, "R\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t"
                  "%s\t%.9g\t%llu\t%llu\t%s\t%llu\t%llu\t%u\t%u\t%u\t%u\t%u\t%u\t%lld\n",
                c.idx, c.table, t.role, t.enc, c.path, c.batching, c.b0, c.bcount, c.row_lo,
                c.nrows, c.n_ids, c.assoc, c.mode, c.expect, c.on_device, c.trace, c.n_out,
                (int)lmh_case_n_items(c), lmh_case_nlanes(c), lmh_hex32(hxo, oxor), osum,
                (unsigned long long)r.bytes, (unsigned long long)r.loads, lmh_hex32(hxb, r.xorb),
                (unsigned long long)r.vsum, (unsigned long long)r.bsum, r.min_off, r.max_off, r.cross, r.peer_hits, r.lanes,
                r.bover, n_launches[c.idx]);
        ++printed;
    }
    if (fflush(f) || ferror(f)) {
        printf("FAIL flush of report %s failed\n", path);
        fclose(f);
        return 1;
    }
    fclose(f);
    printf("REPORT file=%s rows=%lld cases=%zu arm=%s\n", path, printed, plan.cases.size(),
           device_run ? "device" : "host");
    return 0;
}

/* ---- the driver self-tests --------------------------------------------------------------------- *
 * Host-only, no device, no plan file, one argument: the directory to write its three scratch plans
 * into. These are the witnesses that the lane's OWN instruments can fail — the parsers, the
 * comparator, the two bit-wideners, the combine order, the address arithmetic, and the counters that
 * must be able to leave zero. Shared by both binaries, so the roster the host twin prints and the
 * roster the device binary prints are the same roster and gate.py can require it.
 *
 * Every CK line prints, including the ones that pass, because a selftest that prints nothing on
 * success cannot be distinguished from a selftest that ran nothing.
 */
/* Helpers for the driver self-test only — host-only, and named lmh_st_* so nobody mistakes them for
 * kernel policy. lmh_st_one runs one (item, lane 0) through the REAL lmh_item, so every firing and
 * every immunity in block 12 is witnessed through the same code path the device kernel calls. */
static inline void lmh_st_one(const LmhCtx &ctx, const LmhCase *c, int jj, LmhTrace *tr, float *acc) {
    lmh_trace_init(tr);
    lmh_item(&ctx, c, jj, 0, 1, tr, acc);
}

static inline void lmh_st_put_row(std::vector<uint8_t> &slab, long long base, long long row, int h,
                                  const uint16_t *bits) {
    for (int k = 0; k < h; ++k) {
        const long long at = base + (row * h + k) * LMH_BF16_BYTES;
        slab[(size_t)at] = (uint8_t)(bits[k] & 0xFF);
        slab[(size_t)at + 1] = (uint8_t)(bits[k] >> 8);
    }
}

static inline int lmh_driver_selftest(const char *workdir) {
    int fails = 0, checks = 0;
#define CK(cond, name)                                                        \
    do {                                                                      \
        ++checks;                                                             \
        if (!(cond)) { ++fails; printf("FAIL selftest %s\n", name); }         \
        printf("SELFTEST name=%s ok=%d\n", name, (cond) ? 1 : 0);             \
    } while (0)

    /* 1. the comparator: NaN == NaN whatever the sign, +0 != -0, and a 1-ulp neighbour is unequal.
     *    If any of these three were wrong, "bit-exact" would mean something else on one arm. */
    CK(lmh_word_eq(0x7FC00000u, 0xFFC00000u), "word-eq-nan-nan");
    CK(!lmh_word_eq(0x00000000u, 0x80000000u), "word-eq-signed-zero");
    CK(!lmh_word_eq(0x3F800000u, 0x3F800001u), "word-eq-ulp-neighbour");

    /* 2. bf16 widening against hand-written literals. The reader's route is bits << 16
     *    (tools/gguf_reader.py §bf16_to_f32_bytes), and these are the values that argument rests on. */
    CK(lmh_float_to_bits(lmh_bf16_to_float(0x3F80u)) == 0x3F800000u, "bf16-one");
    CK(lmh_float_to_bits(lmh_bf16_to_float(0xBF80u)) == 0xBF800000u, "bf16-minus-one");
    CK(lmh_float_to_bits(lmh_bf16_to_float(0x4000u)) == 0x40000000u, "bf16-two");
    CK(lmh_float_to_bits(lmh_bf16_to_float(0x8000u)) == 0x80000000u, "bf16-minus-zero");
    CK(lmh_float_to_bits(lmh_bf16_to_float(0x0001u)) == 0x00010000u, "bf16-min-subnormal");

    /* 3. f16 (the Q8_0 block scale) including the denormal route and the NaN canonicalisation */
    CK(lmh_float_to_bits(lmh_f16_to_float(0x3C00u)) == 0x3F800000u, "f16-one");
    CK(lmh_float_to_bits(lmh_f16_to_float(0x3800u)) == 0x3F000000u, "f16-half");
    CK(lmh_float_to_bits(lmh_f16_to_float(0x0001u)) == 0x33800000u, "f16-min-subnormal");
    {
        const uint32_t nb = lmh_float_to_bits(lmh_f16_to_float(0x7C01u));
        CK((nb & 0x7F800000u) == 0x7F800000u && (nb & 0x007FFFFFu) != 0u, "f16-nan-is-nan");
    }

    /* 4. bf16 rounding is round-to-nearest-EVEN, not half-up. The two TIES below (discarded half ==
     *    0x8000) land on different neighbours because their kept LSBs differ in parity, and a half-up
     *    implementation gets the even one wrong. Measured, not assumed, in
     *    tmp/worker-lm-head/probe10_digest.cpp and tmp/worker-lm-head/probe11_round.cpp
     *    (2026-10-26): 0x3F808000 -> 0x3F80, 0x3F818000 -> 0x3F82. The two non-ties witness that a
     *    discarded half below or above 0x8000 rounds by magnitude alone (disc=0x4000 -> down,
     *    disc=0xC000 -> up), so the parity rule is only tested where it can change an answer.
     *    (The first revision asserted 1.003906250f -> 0x3F81; that value IS the even tie, so the
     *    right answer was 0x3F80 and the check was red for the right reason.) */
    CK(lmh_float_to_bf16_bits(1.003906250f) == 0x3F80u, "bf16-round-tie-to-even-down");
    CK(lmh_float_to_bf16_bits(1.011718750f) == 0x3F82u, "bf16-round-tie-to-even-up");
    CK(lmh_float_to_bf16_bits(1.001953125f) == 0x3F80u, "bf16-round-below-half-down");
    CK(lmh_float_to_bf16_bits(1.005859375f) == 0x3F81u, "bf16-round-above-half-up");

    /* 5. lmh_combine runs in index order and is the IDENTITY at n == 1, including -0.0. The -0.0 half
     *    is what a restored `acc = 0.0f` seed would break (0.0f + -0.0f is +0.0f), and the
     *    order-sensitivity half is what proves the SEQ/REV/SPLIT8 distinction is real. */
    {
        float p4[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        CK(lmh_combine(p4, 4) == 15.0f, "combine-index-order");
        float pz = -0.0f;
        CK(lmh_float_to_bits(lmh_combine(&pz, 1)) == 0x80000000u, "combine-identity-neg-zero");
        float pa[3] = {1.0e8f, 1.0f, -1.0e8f};      /* (1e8+1) loses the 1 -> 0 */
        float pb[3] = {1.0e8f, -1.0e8f, 1.0f};      /* (1e8-1e8)+1 keeps the 1 -> 1 */
        CK(lmh_combine(pa, 3) != lmh_combine(pb, 3), "combine-is-order-sensitive");
        CK(lmh_combine(pa, 3) == 0.0f, "combine-witness-left-to-right");
        /* the other arm of the pair, pinned by value: the != above is satisfied by any two results,
         * including two wrong ones. A combine that sorted, or that summed in another order on one
         * arm, disagrees with one of these two lines even when it agrees with the !=. */
        CK(lmh_combine(pb, 3) == 1.0f, "combine-witness-other-order");
    }

    /* 6. address arithmetic, re-derived on the host from the same structs the launch is built from
     *    (pitfalls/20 §"A device selftest must verify bytes touched, not arithmetic"). The per-lane
     *    ranges must tile the row exactly and start where the k index says, for both encodings, both
     *    lane counts and the wrong-stride mode. */
    {
        int tile_ok = 1, lane_ok = 1, nonnull_ok = 1;
        for (int enc = 0; enc < LMH_ENC_COUNT; ++enc) {
            LmhTable t;
            memset(&t, 0, sizeof(t));
            t.enc = enc; t.h = 2048; t.v_rows = 4; t.base = 0; t.peer = -1;
            t.nbytes = (enc == LMH_ENC_BF16) ? t.v_rows * 2048LL * LMH_BF16_BYTES
                                             : (t.v_rows * (2048 / LMH_Q8_0_ELTS)) * LMH_Q8_0_BYTES;
            std::vector<uint8_t> dummy((size_t)t.nbytes, 0);
            const int strides[2] = {2048, 256};
            for (int hs = 0; hs < 2; ++hs) {
                const int h_used = strides[hs];
                const int lanes[2] = {1, LMH_SPLIT};
                for (int li = 0; li < 2; ++li) {
                    const int nl = lanes[li];
                    if (h_used % nl || (h_used / nl) % LMH_Q8_0_ELTS) continue;
                    long long sum = 0;
                    for (int lane = 0; lane < nl; ++lane) {
                        long long row_nb = 0, ln_nb = 0;
                        const uint8_t *wr = lmh_row_ptr(&t, dummy.data(), 2, h_used, &row_nb);
                        if (!wr) nonnull_ok = 0;
                        const int per = h_used / nl;
                        const int k0 = lane * per, k1 = (lane == nl - 1) ? h_used : k0 + per;
                        const uint8_t *pl = lmh_lane_ptr(&t, wr, k0, k1 - k0, &ln_nb);
                        const long long want =
                            (wr - dummy.data()) +
                            (enc == LMH_ENC_BF16 ? (long long)k0 * LMH_BF16_BYTES
                                                 : ((long long)k0 / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES);
                        if (pl != dummy.data() + want) lane_ok = 0;
                        sum += ln_nb;
                    }
                    const long long want_row = (enc == LMH_ENC_BF16)
                                                   ? (long long)h_used * LMH_BF16_BYTES
                                                   : ((long long)h_used / LMH_Q8_0_ELTS) *
                                                         LMH_Q8_0_BYTES;
                    if (sum != want_row) tile_ok = 0;
                }
            }
        }
        CK(tile_ok, "lane-ranges-tile-the-row");
        CK(lane_ok, "lane-address-matches-k-index");
        CK(nonnull_ok, "row-pointer-nonnull");
    }

    /* 7. the row selectors, including the successor injection and the deliberate non-clamp */
    {
        int32_t ids[2] = {5, 7};
        CK(lmh_case_row_direct(ids, 0, 0, LMH_MODE_CORRECT) == 5 &&
           lmh_case_row_direct(ids, 1, 0, LMH_MODE_CORRECT) == 7, "gather-row-identity");
        CK(lmh_case_row_direct(ids, 0, 0, LMH_MODE_GATHER_ROW_OFF_1) == 6 &&
           lmh_case_row_direct(ids, 1, 0, LMH_MODE_GATHER_ROW_OFF_1) == 8, "gather-row-successor");
        LmhCase d;
        memset(&d, 0, sizeof(d));
        d.row_lo = 100;
        CK(lmh_case_row_dense(&d, 0) == 100 && lmh_case_row_dense(&d, 2) == 102,
           "dense-row-contiguous");
        int32_t last[1] = {248319};
        CK(lmh_case_row_direct(last, 0, 248320, LMH_MODE_GATHER_ROW_OFF_1) == 248320,
           "gather-successor-of-last-row-is-not-clamped");
    }

    /* 8. the counters that must be able to leave zero: a store outside `out` is counted, not lost.
     *    Without this the driver's bover column is a decoration (pitfalls/00 §"A counter that reads
     *    exactly 0 is vacuous"). */
    {
        Plan plan;
        plan.have_p = 1; plan.h = 2048; plan.head_dim = 256; plan.vocab = 4;
        plan.n_hidden = 1; plan.ids_count = 0; plan.out_floats = 4; plan.slab_bytes = 4 * 2048 * 2;
        plan.n_tables_decl = 1; plan.n_cases_decl = 1;
        LmhTable t;
        memset(&t, 0, sizeof(t));
        t.idx = 0; t.enc = LMH_ENC_BF16; t.role = LMH_ROLE_SYNTH_DUP; t.base = 0;
        t.nbytes = plan.slab_bytes; t.v_rows = 4; t.h = 2048; t.peer = -1;
        plan.tables.push_back(t);
        LmhCase c;
        memset(&c, 0, sizeof(c));
        c.idx = 0; c.table = 0; c.path = LMH_PATH_DENSE; c.batching = LMH_BATCH_BATCHED;
        c.b0 = 0; c.bcount = 2; c.row_lo = 0; c.nrows = 4; c.n_out = 4; c.assoc = LMH_ASSOC_SEQ;
        c.mode = LMH_MODE_CORRECT; c.out_off = 0; c.expect = LMH_EXP_EXACT; c.trace = 1;
        std::vector<uint8_t> slab((size_t)plan.slab_bytes, 0);
        std::vector<float> outv(4, -12345.0f);
        LmhCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.tabs = plan.tables.data(); ctx.n_tables = 1; ctx.slab = slab.data();
        ctx.hidden = NULL; ctx.ids = NULL; ctx.head_dim = 256;
        ctx.out = outv.data(); ctx.out_floats = 4;
        LmhTrace tr;
        lmh_trace_init(&tr);
        CK(tr.min_off == 0xFFFFFFFFu && tr.max_off == 0u, "trace-init-sentinels");
        /* batch row 1 at the injected slot stride (h = 2048) lands outside `out` */
        LmhCase bad = c;
        bad.mode = LMH_MODE_BATCH_SLOT_STRIDE;
        lmh_store(&ctx, &bad, 1, 0, 7.0f, &tr);
        CK(tr.bover == 1, "store-bound-counter-can-fire");
        CK(outv[0] == -12345.0f, "out-of-range-store-writes-nothing");
        lmh_store(&ctx, &bad, 0, 3, 8.0f, &tr);
        CK(tr.bover == 1 && outv[3] == 8.0f, "in-range-store-still-lands");
    }

    /* 9. the dots, against hand computations the file does not share with them. Weights and hidden
     *    are exactly representable, so the true result is exact and this witnesses the TERM formula
     *    (bf16 widening, Q8_0 d*q) rather than the summation. */
    {
        const int H = 32;
        std::vector<uint8_t> slab(4096, 0);
        const uint16_t wbf[4] = {0x3F80u, 0xC000u, 0x3F00u, 0x4080u};   /* 1, -2, 0.5, 4 */
        for (int k = 0; k < 4; ++k) {
            slab[2 * k] = (uint8_t)(wbf[k] & 0xFF);
            slab[2 * k + 1] = (uint8_t)(wbf[k] >> 8);
        }
        const long long q8_base = 512;
        const uint16_t dbits = 0x3800u;                                  /* f16 0.5 */
        slab[q8_base] = (uint8_t)(dbits & 0xFF);
        slab[q8_base + 1] = (uint8_t)(dbits >> 8);
        const int8_t q8v[4] = {1, 2, -3, 4};
        for (int k = 0; k < 4; ++k) slab[q8_base + 2 + k] = (uint8_t)q8v[k];

        LmhTable t0, t1;
        memset(&t0, 0, sizeof(t0));
        t0.idx = 0; t0.enc = LMH_ENC_BF16; t0.role = LMH_ROLE_SYNTH_DUP; t0.base = 0;
        t0.nbytes = 4 * H * LMH_BF16_BYTES; t0.v_rows = 4; t0.h = H; t0.peer = -1;
        memset(&t1, 0, sizeof(t1));
        t1.idx = 1; t1.enc = LMH_ENC_Q8_0; t1.role = LMH_ROLE_SYNTH_DUP; t1.base = q8_base;
        t1.nbytes = 4 * (H / LMH_Q8_0_ELTS) * LMH_Q8_0_BYTES; t1.v_rows = 4; t1.h = H; t1.peer = 0;

        Plan plan;
        plan.have_p = 1; plan.h = H; plan.head_dim = 8; plan.vocab = 4; plan.n_hidden = 2;
        plan.ids_count = 1; plan.out_floats = 8; plan.slab_bytes = (long long)slab.size();
        plan.n_tables_decl = 2; plan.n_cases_decl = 2;
        plan.tables.push_back(t0);
        plan.tables.push_back(t1);

        std::vector<float> hid(2 * H, 0.0f), outv(8, -1.0f);
        const float hvals[4] = {1.0f, 0.5f, -2.0f, 2.0f};
        for (int k = 0; k < 4; ++k) { hid[k] = hvals[k]; hid[H + k] = 2.0f * hvals[k]; }

        LmhCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.tabs = plan.tables.data(); ctx.n_tables = 2; ctx.slab = slab.data();
        ctx.hidden = hid.data(); ctx.ids = NULL; ctx.head_dim = 8;
        ctx.out = outv.data(); ctx.out_floats = 8;

        LmhCase c;
        memset(&c, 0, sizeof(c));
        c.idx = 0; c.table = 0; c.path = LMH_PATH_DENSE; c.batching = LMH_BATCH_BATCHED;
        c.b0 = 0; c.bcount = 2; c.row_lo = 0; c.nrows = 1; c.n_out = 1; c.assoc = LMH_ASSOC_SEQ;
        c.mode = LMH_MODE_CORRECT; c.out_off = 0; c.expect = LMH_EXP_EXACT; c.trace = 1;
        LmhTrace tr;
        lmh_trace_init(&tr);
        float acc[LMH_MAX_BATCH];
        lmh_item(&ctx, &c, 0, 0, 1, &tr, acc);
        /* 1*1 + 0.5*(-2) + (-2)*0.5 + 2*4 = 1 - 1 - 1 + 8 = 7 exactly; batch row 1 doubles it */
        CK(acc[0] == 7.0f && acc[1] == 14.0f, "dot-bf16-against-hand-computation");
        CK(tr.bytes == (uint64_t)H * LMH_BF16_BYTES && tr.loads == 1 && tr.lanes == 1,
           "dot-bf16-trace-bytes-loads-lanes");

        LmhCase cq = c;
        cq.idx = 1; cq.table = 1; cq.out_off = 4;
        LmhTrace tq;
        lmh_trace_init(&tq);
        lmh_item(&ctx, &cq, 0, 0, 1, &tq, acc);
        /* 0.5*(1*1 + 2*0.5 + (-3)*(-2) + 4*2) = 0.5*(1+1+6+8) = 8 exactly; batch row 1 doubles it */
        CK(acc[0] == 8.0f && acc[1] == 16.0f, "dot-q8-against-hand-computation");
        CK(tq.bytes == LMH_Q8_0_BYTES && tq.loads == 1 && tq.lanes == 1,
           "dot-q8-trace-bytes-loads-lanes");

        /* Which digest carries the verdict is a measured fact, not a preference: dropping d = 0.5 is
         * a uniform x2 rescale of every decoded weight, and an XOR over an even-count set is provably
         * blind to exactly that. Measured on this fixture (tmp/worker-lm-head/probe10_digest.cpp,
         * 2026-10-26): xorb = 0xFF400000 for BOTH arms, vsum = 0x17E400000 vs 0x180400000. So vsum
         * carries the verdict; xorb's coincidence is witnessed as the blind spot it is (and gate.py
         * states this per case); and bsum — which sums the RAW STORAGE bytes — cannot see a decode
         * change at all, so it must not be asked to. */
        LmhCase cd = cq;
        cd.mode = LMH_MODE_Q8_DROP_SCALE;
        LmhTrace td;
        lmh_trace_init(&td);
        lmh_item(&ctx, &cd, 0, 0, 1, &td, acc);
        CK(acc[0] == 16.0f && acc[1] == 32.0f, "dot-q8-drop-scale-fires");
        CK(td.vsum != tq.vsum, "drop-scale-moves-the-value-digest");
        CK(td.xorb == tq.xorb, "xor-digest-blind-to-uniform-rescale");
        CK(td.bsum == tq.bsum, "byte-sum-digest-blind-to-decode");
        CK(td.bytes == tq.bytes && td.lanes == tq.lanes, "drop-scale-reads-the-same-bytes");

        /* the re-loop injection: outputs identical, bytes doubled, lanes unchanged. This pair is the
         * whole reason this lane traces addresses at all. */
        LmhCase cb = cq;
        cb.bcount = 2;
        LmhTrace tb;
        lmh_trace_init(&tb);
        lmh_item(&ctx, &cb, 0, 0, 1, &tb, acc);
        const uint64_t bytes_once = tb.bytes;
        LmhCase cl = cb;
        cl.mode = LMH_MODE_BATCH_RELOOPS;
        LmhTrace tl;
        lmh_trace_init(&tl);
        lmh_item(&ctx, &cl, 0, 0, 1, &tl, acc);
        CK(acc[0] == 8.0f && acc[1] == 16.0f, "reloops-leaves-outputs-identical");
        CK(tl.bytes == bytes_once * 2 && tl.loads == tb.loads * 2 && tl.lanes == tb.lanes &&
           tl.xorb == tb.xorb, "reloops-doubles-bytes-and-cancels-the-digest");

        /* the tied alias reads the PEER: untied it must not move cross or peer_hits (the peer is a
         * disjoint range), and on the tied control a peer read cannot be a cross read because the
         * peer range IS the own range. Built here so the witness does not depend on the plan file
         * gate.py writes. */
        LmhCase ct = cq;
        ct.bcount = 1;
        LmhTrace tt0;
        lmh_trace_init(&tt0);
        lmh_item(&ctx, &ct, 0, 0, 1, &tt0, acc);
        CK(tt0.cross == 0 && tt0.peer_hits == 0, "untied-read-stays-in-its-table");
        Plan tied = plan;
        tied.tables[0].base = tied.tables[1].base;      /* the control: one buffer, two roles */
        tied.tables[0].nbytes = tied.tables[1].nbytes;
        LmhCtx tctx = ctx;
        tctx.tabs = tied.tables.data();
        LmhTrace tt1;
        lmh_trace_init(&tt1);
        lmh_item(&tctx, &ct, 0, 0, 1, &tt1, acc);
        CK(tt1.peer_hits >= 1 && tt1.cross == 0, "tied-control-peer-read-is-not-a-cross-read");
        CK(tt1.bytes == tt0.bytes && tt1.lanes == tt0.lanes, "tied-control-reads-one-row-set");
    }

    /* 10. the parser: a good plan parses with the declared counts, and three malformed shapes
     *     refuse. A parser that cannot fail is the empty-parse-PASS bug waiting to happen. */
    {
        const char *good =
            "P\t256\t2048\t248320\t1\t1\t8\t16384\t1\t1\n"
            "T\t0\t0\t0\t0\t16384\t4\t2048\t-1\t0\n"
            "C\t0\t0\t0\t0\t0\t1\t0\t1\t0\t0\t0\t0\t0\t1\t-1\t0\t0\t1\t1\n";
        const char *bad_count =
            "P\t256\t2048\t248320\t1\t1\t8\t16384\t1\t1\n"
            "T\t0\t0\t0\t0\t16384\t4\t2048\t-1\t0\n"
            "C\t0\t0\t0\t0\t0\t1\t0\t1\t0\t0\t0\t0\t0\t1\t-1\t0\t0\t1\n";
        const char *bad_rec =
            "P\t256\t2048\t248320\t1\t1\t8\t16384\t1\t1\n"
            "X\t0\n";
        const char *bad_two_p =
            "P\t256\t2048\t248320\t1\t1\t8\t16384\t1\t1\n"
            "P\t256\t2048\t248320\t1\t1\t8\t16384\t1\t1\n";
        const char *bodies[4] = {good, bad_count, bad_rec, bad_two_p};
        int want_ok[4] = {1, 0, 0, 0};
        int parser_ok = 1, saw_good = 0, n_written = 0;
        for (int i = 0; i < 4; ++i) {
            char name[512];
            snprintf(name, sizeof(name), "%s/selftest_plan_%d.tsv", workdir, i);
            FILE *w = fopen(name, "wb");
            if (!w) { parser_ok = 0; break; }
            fputs(bodies[i], w);
            fclose(w);
            ++n_written;
            Plan p;
            int np = 0, nt = 0, nc = 0;
            const int ok = (lmh_read_plan(name, &p, &np, &nt, &nc) == 0);
            if (ok != want_ok[i]) parser_ok = 0;
            if (i == 0) {
                if (np != 1 || nt != 1 || nc != 1) parser_ok = 0;
                if (p.head_dim != 256 || p.h != 2048 || p.vocab != 248320) parser_ok = 0;
                if (p.tables.empty() || p.tables[0].nbytes != 16384) parser_ok = 0;
                if (p.cases.empty() || p.cases[0].n_out != 1) parser_ok = 0;
                saw_good = 1;
            }
            if (i > 0 && !ok && (int)p.cases.size() + (int)p.tables.size() != (i == 1 ? 1 : 0))
                parser_ok = 0;
        }
        CK(parser_ok && saw_good && n_written == 4, "plan-parser-fires-on-malformed-input");
    }

    /* 11. the report contract. The R line the writer emits and the name header above it must both
     *     carry LMH_F_R tab-separated tokens, the names must be exactly lmh_r_fields[], and the vsum
     *     column must hold the trace's own vsum printed as an integer — not a second copy of xorb,
     *     which is what a copy-paste column bug produces and which every count-only check would
     *     wave through. gate.py parses THIS header to find its column indices, so this block is the
     *     one place the R layout is decided; a constant a writer and a parser agree on by luck is how
     *     LMH_F_R drifted once already. */
    {
        Plan plan;
        plan.have_p = 1; plan.h = 32; plan.head_dim = 8; plan.vocab = 4; plan.n_hidden = 1;
        plan.ids_count = 0; plan.out_floats = 4; plan.slab_bytes = 4096;
        plan.n_tables_decl = 1; plan.n_cases_decl = 1;
        LmhTable t;
        memset(&t, 0, sizeof(t));
        t.idx = 0; t.enc = LMH_ENC_BF16; t.role = LMH_ROLE_SYNTH_DUP; t.base = 0;
        t.nbytes = plan.slab_bytes; t.v_rows = 4; t.h = 32; t.peer = -1;
        plan.tables.push_back(t);
        LmhCase c;
        memset(&c, 0, sizeof(c));
        c.idx = 0; c.table = 0; c.path = LMH_PATH_DENSE; c.batching = LMH_BATCH_PERROW;
        c.b0 = 0; c.bcount = 1; c.row_lo = 0; c.nrows = 2; c.n_out = 2; c.assoc = LMH_ASSOC_SEQ;
        c.mode = LMH_MODE_CORRECT; c.out_off = 0; c.expect = LMH_EXP_EXACT; c.trace = 1;
        plan.cases.push_back(c);
        std::vector<float> outv(4, 0.25f);
        std::vector<LmhTrace> trv(1);
        lmh_trace_init(&trv[0]);
        trv[0].bytes = 64; trv[0].loads = 2; trv[0].xorb = 0xDEADBEEFu;
        trv[0].vsum = 0x123456789ABCull; trv[0].bsum = 7;
        trv[0].min_off = 4; trv[0].max_off = 68; trv[0].cross = 0; trv[0].peer_hits = 0;
        trv[0].lanes = 2; trv[0].bover = 0;
        long long launches[1] = {3};

        const int n_names = (int)(sizeof(lmh_r_fields) / sizeof(lmh_r_fields[0]));
        int vcol = -1, xcol = -1;
        for (int i = 0; i < n_names; ++i) {
            if (strcmp(lmh_r_fields[i], "vsum") == 0) vcol = i + 1;   /* +1: token 0 is the letter */
            if (strcmp(lmh_r_fields[i], "xorb") == 0) xcol = i + 1;
        }

        int contract_ok = 1;
        if (n_names != LMH_F_R - 1) contract_ok = 0;
        if ((int)(sizeof(lmh_r_fmt) / sizeof(lmh_r_fmt[0])) != n_names) contract_ok = 0;
        /* The hand-written form table and the name-derived rule must say the same thing for EVERY
         * column, and the hex family must be populated: a digest rule over zero bit-image columns is a
         * counter that reads zero and cannot fail (pitfalls/40). Its own counter, not contract_ok, so a
         * red here names the form table and nothing else. */
        int form_ok = ((int)(sizeof(lmh_r_fmt) / sizeof(lmh_r_fmt[0])) == n_names);
        int n_hex = 0;
        for (int i = 0; i < n_names; ++i) {
            if (strcmp(lmh_r_fmt[i], lmh_r_fmt_of(i)) != 0) form_ok = 0;
            if (strcmp(lmh_r_fmt[i], LMH_RF_HEX) == 0) ++n_hex;
        }
        if (n_hex != (int)(sizeof(lmh_r_hex_fields) / sizeof(lmh_r_hex_fields[0]))) form_ok = 0;
        CK(form_ok, "report-form-table-agrees-with-the-field-names");
        if (vcol < 0 || xcol < 0) contract_ok = 0;
        char rpath[512];
        snprintf(rpath, sizeof(rpath), "%s/selftest_report.tsv", workdir);
        if (lmh_write_report(rpath, plan, trv.data(), outv.data(), launches, 0)) contract_ok = 0;
        FILE *rf = fopen(rpath, "rb");
        if (!rf) {
            contract_ok = 0;
        } else {
            char line[4096];
            int saw_header = 0, saw_r = 0;
            while (fgets(line, sizeof(line), rf)) {
                char *fld[64];
                if (line[0] == '#') {
                    if (line[1] != 'R') continue;          /* the prose note line is not a contract */
                    ++saw_header;
                    const int nh = lmh_split(line, fld, 64);
                    if (nh != LMH_F_R) contract_ok = 0;    /* '#R' + the 33 names */
                    for (int i = 0; i + 1 < LMH_F_R && i + 1 < nh; ++i)
                        if (strcmp(fld[i + 1], lmh_r_fields[i]) != 0) contract_ok = 0;
                    continue;
                }
                if (line[0] != 'R') continue;
                ++saw_r;
                const int nf = lmh_split(line, fld, 64);
                if (nf != LMH_F_R) contract_ok = 0;
                if (vcol < 0 || vcol >= nf || xcol >= nf) { contract_ok = 0; continue; }
                if (strtoull(fld[vcol], 0, 10) != trv[0].vsum) contract_ok = 0;
                if (strcmp(fld[vcol], fld[xcol]) == 0) contract_ok = 0;
                /* Every printed token must be the canonical spelling of the kind lmh_r_fmt[] declares
                 * for it — the writer is checked against the table the parser scrapes, so the two
                 * sides cannot each keep their own ruler. */
                for (int i = 0; i + 1 < LMH_F_R && i + 1 < nf; ++i)
                    if (!lmh_fmt_ok(lmh_r_fmt[i], fld[i + 1])) contract_ok = 0;
            }
            fclose(rf);
            if (!saw_header || !saw_r) contract_ok = 0;
        }
        CK(contract_ok, "report-field-count-names-and-vsum-column");
        /* THE WRONG-RADIX CONTROL. The bug this lane was handed is a producer that printed the digest
         * with %u while the parser read the same column as hex: 0xDEADBEEF and 3735928559 are one
         * 32-bit value in two radices, each of which parses to a plausible number, so a value-only
         * comparison cannot see the disagreement. The rule that kills it is that the radix lives IN
         * THE TEXT, and this block is the witness that the rule has teeth: the canonical token passes,
         * the other radix's spelling of the SAME value is refused by both kinds, the two spellings
         * differ as strings, and reading the decimal token as hex yields a number that is not the
         * digest (which is exactly the mis-parse the parser must never be able to perform silently). */
        {
            int radix_ok = (xcol >= 0);
            char hb[16];
            char db[32];
            const char *canon = lmh_hex32(hb, trv[0].xorb);
            snprintf(db, sizeof(db), "%u", trv[0].xorb);
            if (!lmh_fmt_ok(LMH_RF_HEX, canon)) radix_ok = 0;
            if (lmh_fmt_ok(LMH_RF_HEX, db)) radix_ok = 0;
            if (lmh_fmt_ok(LMH_RF_DEC, canon)) radix_ok = 0;
            if (!lmh_fmt_ok(LMH_RF_DEC, db)) radix_ok = 0;
            if (strcmp(canon, db) == 0) radix_ok = 0;
            if (strtoull(canon, NULL, 16) != (unsigned long long)trv[0].xorb) radix_ok = 0;
            if (strtoull(db, NULL, 10) != (unsigned long long)trv[0].xorb) radix_ok = 0;
            if (strtoull(db, NULL, 16) == (unsigned long long)trv[0].xorb) radix_ok = 0;
            CK(radix_ok, "digest-radix-is-in-the-text-not-in-the-parser");
        }
    }

    /* 12. every injection this lane claims an immunity for, witnessed on BOTH arms — a firing arm and
     *     the immune arm the plan relies on, from one fixture, through the same lmh_item the device
     *     kernel calls. `AGENTS.md` §"Never loosen a gate" plus the standing rule that an immunity
     *     true by construction is not evidence: each immune arm therefore sits beside a firing arm of
     *     the SAME mode, and the counters that must be able to leave zero are read, not assumed.
     *
     *     The fixture is small but not degenerate: h = 32 with head_dim = 8 (so the wrong-stride
     *     injection has a stride to be wrong about), four OWN rows of which rows 0 and 1 are
     *     BYTE-IDENTICAL (the duplicated pair the GATHER_ROW_OFF_1 immunity is derived from), and a
     *     disjoint PEER table with its own values (the untied case) next to a tied control built by
     *     pointing PEER at OWN's bytes. Every weight and hidden value here is exactly representable,
     *     so each expected sum is a hand number and not a second implementation. */
    {
        const int H = 32, HD = 8;
        const uint16_t W_ONE = 0x3F80u, W_TWO = 0x4000u, W_THREE = 0x4040u, W_FOUR = 0x4080u;
        const long long OWN_BASE = 0, PEER_BASE = 1024;
        const long long ROW_B = (long long)H * LMH_BF16_BYTES;
        std::vector<uint8_t> slab(4096, 0);
        std::vector<uint16_t> rw(H, W_ONE);
        std::vector<LmhTable> tabs(2);

        rw[H - 1] = W_TWO;                                   /* the k=31 term DROP_LAST_K removes */
        lmh_st_put_row(slab, OWN_BASE, 0, H, rw.data());
        lmh_st_put_row(slab, OWN_BASE, 1, H, rw.data());     /* byte-identical to row 0 */
        for (int k = 0; k < H; ++k) rw[k] = (k < HD) ? W_THREE : W_ONE;
        lmh_st_put_row(slab, OWN_BASE, 2, H, rw.data());     /* differs from rows 0/1 in the head */
        for (int k = 0; k < H; ++k) rw[k] = W_ONE;
        lmh_st_put_row(slab, OWN_BASE, 3, H, rw.data());
        for (int k = 0; k < H; ++k) rw[k] = W_FOUR;
        lmh_st_put_row(slab, PEER_BASE, 0, H, rw.data());
        lmh_st_put_row(slab, PEER_BASE, 1, H, rw.data());

        memset(&tabs[0], 0, sizeof(tabs[0]));
        tabs[0].idx = 0; tabs[0].enc = LMH_ENC_BF16; tabs[0].role = LMH_ROLE_SYNTH_DUP;
        tabs[0].base = OWN_BASE; tabs[0].nbytes = 4 * ROW_B; tabs[0].v_rows = 4; tabs[0].h = H;
        tabs[0].peer = 1; tabs[0].real = 0;
        memset(&tabs[1], 0, sizeof(tabs[1]));
        tabs[1].idx = 1; tabs[1].enc = LMH_ENC_BF16; tabs[1].role = LMH_ROLE_LMHEAD;
        tabs[1].base = PEER_BASE; tabs[1].nbytes = 4 * ROW_B; tabs[1].v_rows = 4; tabs[1].h = H;
        tabs[1].peer = 0; tabs[1].real = 0;

        /* hidden rows: 0 = all ones, 1 = a one-hot that gets moved below, 2 = a one-hot whose value is
         * NOT a bf16 value, 3 = ones with h[31] == 0, 4 = non-zero only on k < head_dim */
        std::vector<float> hid(5 * H, 0.0f);
        for (int k = 0; k < H; ++k) hid[k] = 1.0f;
        hid[2 * H] = 1.001953125f;
        for (int k = 0; k < H - 1; ++k) hid[3 * H + k] = 1.0f;
        for (int k = 0; k < HD; ++k) hid[4 * H + k] = 1.0f;
        std::vector<int32_t> idsv(2);
        idsv[0] = 0; idsv[1] = 1;                            /* rows 0/1 are the duplicate pair */
        std::vector<float> outv(64, 0.0f);

        LmhCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.tabs = tabs.data(); ctx.n_tables = 2; ctx.slab = slab.data();
        ctx.hidden = hid.data(); ctx.ids = idsv.data(); ctx.head_dim = HD;
        ctx.out = outv.data(); ctx.out_floats = (long long)outv.size();

        /* the tied control: two roles, one buffer — the assumption the batching precedent was built on */
        std::vector<LmhTable> tied_tabs = tabs;
        tied_tabs[1].base = tabs[0].base;
        tied_tabs[1].nbytes = tabs[0].nbytes;
        LmhCtx tctx = ctx;
        tctx.tabs = tied_tabs.data();

        LmhCase proto;
        memset(&proto, 0, sizeof(proto));
        proto.idx = 0; proto.table = 0; proto.path = LMH_PATH_DENSE;
        proto.batching = LMH_BATCH_PERROW; proto.b0 = 0; proto.bcount = 1; proto.row_lo = 0;
        proto.nrows = 1; proto.n_out = 1; proto.assoc = LMH_ASSOC_SEQ; proto.mode = LMH_MODE_CORRECT;
        proto.out_off = 0; proto.expect = LMH_EXP_EXACT; proto.trace = 1;

        LmhCase c, c2;
        LmhTrace tr, tr2;
        float acc[LMH_MAX_BATCH], acc2[LMH_MAX_BATCH];

        /* --- LMH_MODE_HALF_ACCUM: every partial is rounded to bf16 ----------------------------- */
        c = proto; c.b0 = 2;                                  /* a product that is not a bf16 value */
        c2 = c; c2.mode = LMH_MODE_HALF_ACCUM;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(lmh_float_to_bits(acc[0]) == lmh_float_to_bits(1.001953125f) &&
           lmh_float_to_bits(acc2[0]) == lmh_float_to_bits(1.0f),
           "half-accum-fires-on-an-inexact-product");
        int ha_immune = 1;
        for (int j = 0; j < H; ++j) {                          /* one-hot at EVERY position */
            for (int k = 0; k < H; ++k) hid[H + k] = 0.0f;
            hid[H + j] = 1.0f;
            c = proto; c.b0 = 1;
            c2 = c; c2.mode = LMH_MODE_HALF_ACCUM;
            lmh_st_one(ctx, &c, 0, &tr, acc);
            lmh_st_one(ctx, &c2, 0, &tr2, acc2);
            if (!lmh_word_eq(lmh_float_to_bits(acc[0]), lmh_float_to_bits(acc2[0]))) ha_immune = 0;
        }
        CK(ha_immune, "half-accum-immune-for-every-one-hot-hidden");

        /* --- LMH_MODE_DROP_LAST_K: the last term of the slice is undone ------------------------ */
        c = proto; c.b0 = 0;                                  /* h[31] = 1 -> the term is live */
        c2 = c; c2.mode = LMH_MODE_DROP_LAST_K;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 33.0f && acc2[0] == 31.0f,
           "drop-last-k-fires-when-the-last-term-is-live");
        c = proto; c.b0 = 3;                                  /* h[31] = 0 -> it removes 0 * w[31] */
        c2 = c; c2.mode = LMH_MODE_DROP_LAST_K;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 31.0f && acc2[0] == 31.0f,
           "drop-last-k-immune-where-that-hidden-element-is-zero");

        /* --- LMH_MODE_ROW_STRIDE_HEAD_DIM: rows are laid out 8 elements apart ------------------ */
        c = proto; c.b0 = 4; c.row_lo = 2;                    /* row 2's head is 3.0, row 0's is 1.0 */
        c2 = c; c2.mode = LMH_MODE_ROW_STRIDE_HEAD_DIM;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 24.0f && acc2[0] == 8.0f,
           "row-stride-head-dim-fires-on-a-row-beyond-the-first");
        CK(tr.bytes == (uint64_t)H * LMH_BF16_BYTES &&
           tr2.bytes == (uint64_t)HD * LMH_BF16_BYTES && tr2.cross == 0u,
           "row-stride-head-dim-reads-head_dim-bytes-and-stays-in-its-table");
        c = proto; c.b0 = 4; c.row_lo = 0;
        c2 = c; c2.mode = LMH_MODE_ROW_STRIDE_HEAD_DIM;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 8.0f && acc2[0] == 8.0f && tr2.bytes == (uint64_t)HD * LMH_BF16_BYTES,
           "row-stride-head-dim-immune-numerically-for-a-head-only-hidden-at-row-0");

        /* --- LMH_MODE_TIED_ALIAS: the projection's rows read out of the embedding table --------- */
        c = proto; c.b0 = 0;
        c2 = c; c2.mode = LMH_MODE_TIED_ALIAS;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 33.0f && acc2[0] == 128.0f,
           "tied-alias-untied-reads-the-embedding-rows-not-the-projection");
        CK(tr.cross == 0u && tr.peer_hits == 0u && tr2.cross >= 1u && tr2.peer_hits >= 1u,
           "tied-alias-untied-is-counted-as-a-cross-read");
        lmh_st_one(tctx, &c, 0, &tr, acc);                    /* the tied control: one buffer */
        lmh_st_one(tctx, &c2, 0, &tr2, acc2);
        CK(lmh_word_eq(lmh_float_to_bits(acc[0]), lmh_float_to_bits(acc2[0])) &&
           tr2.cross == 0u && tr2.peer_hits >= 1u,
           "tied-alias-on-the-tied-control-is-a-no-op-and-never-a-cross-read");

        /* --- LMH_MODE_CROSS_TABLE_READ: touch the peer's bytes, compute from own ---------------- */
        c = proto; c.b0 = 0;
        c2 = c; c2.mode = LMH_MODE_CROSS_TABLE_READ;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(acc[0] == 33.0f && acc2[0] == 33.0f,
           "cross-table-read-leaves-the-output-alone");
        CK(tr2.cross >= 1u && tr2.peer_hits >= 1u &&
           tr.bytes == (uint64_t)H * LMH_BF16_BYTES &&
           tr2.bytes == 2ull * (uint64_t)H * LMH_BF16_BYTES,
           "cross-table-read-untied-moves-both-byte-counters");
        lmh_st_one(tctx, &c, 0, &tr, acc);
        lmh_st_one(tctx, &c2, 0, &tr2, acc2);
        CK(tr.cross == 0u && tr2.cross == 0u && tr2.peer_hits >= 2u,
           "cross-table-read-on-the-tied-control-can-never-count-a-cross");

        /* --- LMH_MODE_GATHER_ROW_OFF_1 against the duplicated row pair -------------------------- */
        c = proto; c.path = LMH_PATH_GATHER; c.ids_off = 0; c.n_ids = 1; c.n_out = 1;
        c2 = c; c2.mode = LMH_MODE_GATHER_ROW_OFF_1;
        lmh_st_one(ctx, &c, 0, &tr, acc);                     /* ids[0]=0 -> row 0 */
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);                  /* successor row 1: BYTE-IDENTICAL */
        const int dup_same =
            lmh_word_eq(lmh_float_to_bits(acc[0]), lmh_float_to_bits(acc2[0])) &&
            tr.bytes == tr2.bytes && tr.xorb == tr2.xorb && tr.vsum == tr2.vsum &&
            tr.cross == 0u && tr2.cross == 0u;
        lmh_st_one(ctx, &c, 1, &tr, acc);                     /* ids[1]=1 -> row 1 */
        lmh_st_one(ctx, &c2, 1, &tr2, acc2);                  /* successor row 2: DIFFERENT */
        CK(dup_same && acc[0] != acc2[0] && acc[0] == 33.0f && acc2[0] == 48.0f,
           "gather-row-off-1-immune-on-the-duplicate-and-live-next-to-it");

        /* --- the two batching injections at the one batch row, where they cannot fire ------------ *
         * Their firing arms are block 8 (bover == 1 at bcount == 2) and block 9 (bytes double at
         * bcount == 2 with bit-identical outputs). Both are named here so an immune plan cannot be
         * mistaken for an untested mode. */
        c = proto; c.mode = LMH_MODE_BATCH_SLOT_STRIDE; c.bcount = 1; c.out_off = 7;
        for (size_t i = 0; i < outv.size(); ++i) outv[i] = -3.0f;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_store(&ctx, &c, 0, 0, acc[0], &tr);
        CK(tr.bover == 0u && outv[7] == acc[0] && outv[7 + H] == -3.0f,
           "batch-slot-stride-immune-at-bcount-1");
        c = proto; c.batching = LMH_BATCH_BATCHED; c.bcount = 1;
        c2 = c; c2.mode = LMH_MODE_BATCH_RELOOPS;
        lmh_st_one(ctx, &c, 0, &tr, acc);
        lmh_st_one(ctx, &c2, 0, &tr2, acc2);
        CK(tr.bytes == tr2.bytes && tr.loads == tr2.loads && tr.lanes == tr2.lanes,
           "batch-reloops-immune-at-bcount-1");
    }

#undef CK
    printf("DRIVER SELFTEST RESULT checks=%d failures=%d\n", checks, fails);
    return fails;
}

#endif /* LM_HEAD_PLAN_H */
