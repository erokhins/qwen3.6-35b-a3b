/* engine/device/m2_se_io.h — host-only plan/slab IO for the M2 shared-expert arm.
 *
 * Host-only by necessity, not by taste: a `__host__ __device__` function may not call `fopen`
 * (`prototype/moe_gather/moe_gather_kernels.h` says the same about its plan parser at the same
 * reason), so the compiled pair shares this file from the host side of each arm and shares
 * `m2_se_kernels.h` for everything the device touches.
 *
 * THE PLAN IS ENGINE-SHAPED AND SMALL ON PURPOSE. It names four byte spans and three form words.
 * Every geometry number it carries is a claim the arm checks against the constants in
 * `shared_expert_kernels.h` and refuses on disagreement, so a plan typed from memory cannot be
 * judged — which is the empty-denominator failure `pitfalls/40` names.
 *
 * A REFUSAL IS AN EXIT CODE, NOT A NOTE. `m2_se_plan_check` returns a named tag and the caller
 * exits non-zero. The driver treats a non-zero arm as fatal (`engine/block_device.py`), so a bad
 * plan cannot produce a partial record that a later stage row might be computed from — the silent
 * host fallback this lane exists to make impossible.
 */
#ifndef M2_SE_IO_H
#define M2_SE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_se_kernels.h"

#define M2_SE_PLAN_VERSION 1

struct M2SePlanFile {
    int version;
    long slab_bytes;
    long off_fused, bytes_fused, rows_fused, cols_fused;
    long off_down, bytes_down, rows_down, cols_down;
    long off_sg, bytes_sg, rows_sg, cols_sg;
    long off_x, bytes_x, elems_x;
    int mode, sgate_form, cast_form, skip_stage3, alloc_oversize;
    char fault[64];
};

/* `key=value` within a line; value domain is "no space", because a numeric-only value pattern drops
 * every word-valued field (pitfalls/40 §a-driver-whose-parser-matches-nothing). */
/* What one run of one arm did, as that arm reports it. The HIP arm fills it from counted calls;
 * the g++ twin fills it with zeros and the path name `host_cpubody`, and says so in the RUN line — a
 * device claim that has to be earned by a non-zero count is `pitfalls/40`'s device-claim rule. */
struct M2SeRunStats {
    int hip_calls;
    int launches;
    int allocs;
    long bytes_in;
    long bytes_out;
    const char *path;
};

static inline int m2_se_field(const char *line, const char *key, char *val, size_t n) {
    const size_t kl = strlen(key);
    const char *p = line;
    while (*p) {
        if (!strncmp(p, key, kl) && p[kl] == '=') {
            size_t i = 0;
            p += kl + 1;
            while (*p && *p != ' ' && *p != '\t' && *p != '\n' && i + 1 < n) val[i++] = *p++;
            val[i] = '\0';
            return 1;
        }
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') ++p;
        while (*p == ' ' || *p == '\t') ++p;
    }
    return 0;
}

static inline long m2_se_long(const char *s) { return strtol(s, 0, 10); }

static inline int m2_se_mode_of(const char *w) {
    if (!strcmp(w, "correct")) return SE_MODE_CORRECT;
    if (!strcmp(w, "up_first")) return SE_MODE_LAYOUT_UP_FIRST;
    if (!strcmp(w, "pair_swap")) return SE_MODE_LAYOUT_PAIR_SWAP;
    if (!strcmp(w, "row_inter")) return SE_MODE_LAYOUT_ROW_INTER;
    if (!strcmp(w, "row_shift1")) return SE_MODE_ROW_SHIFT1;
    if (!strcmp(w, "f32_act")) return SE_MODE_F32_ACT;
    if (!strcmp(w, "tail_drop")) return SE_MODE_TAIL_DROP;
    if (!strcmp(w, "cols_half")) return SE_MODE_COLS_HALF;
    if (!strcmp(w, "wide_misread")) return SE_MODE_WIDE_MISREAD;
    return -1;
}

static inline int m2_se_read_plan(const char *path, struct M2SePlanFile *pf, char *err, size_t errn) {
    FILE *fh = fopen(path, "rb");
    char line[1024];
    int saw_header = 0, n_span = 0;
    memset(pf, 0, sizeof(*pf));
    snprintf(pf->fault, sizeof(pf->fault), "none");
    if (!fh) { snprintf(err, errn, "plan-absent: %s cannot be opened", path); return 1; }
    while (fgets(line, sizeof(line), fh)) {
        char v[64];
        if (!strncmp(line, "M2SEPLAN", 8)) {
            saw_header = 1;
            if (!m2_se_field(line, "version", v, sizeof(v))) {
                snprintf(err, errn, "plan-version-absent: the M2SEPLAN line carries no version=");
                fclose(fh);
                return 1;
            }
            pf->version = (int)m2_se_long(v);
        } else if (!strncmp(line, "SLAB", 4)) {
            if (m2_se_field(line, "bytes", v, sizeof(v))) pf->slab_bytes = m2_se_long(v);
        } else if (!strncmp(line, "SPAN", 4)) {
            long off = -1, bytes = -1, rows = -1, cols = -1;
            char role[32];
            role[0] = '\0';
            if (m2_se_field(line, "role", role, sizeof(role))) {
                if (m2_se_field(line, "off", v, sizeof(v))) off = m2_se_long(v);
                if (m2_se_field(line, "bytes", v, sizeof(v))) bytes = m2_se_long(v);
                if (m2_se_field(line, "rows", v, sizeof(v))) rows = m2_se_long(v);
                if (m2_se_field(line, "cols", v, sizeof(v))) cols = m2_se_long(v);
                if (!strcmp(role, "fused")) {
                    pf->off_fused = off; pf->bytes_fused = bytes;
                    pf->rows_fused = rows; pf->cols_fused = cols;
                } else if (!strcmp(role, "down")) {
                    pf->off_down = off; pf->bytes_down = bytes;
                    pf->rows_down = rows; pf->cols_down = cols;
                } else if (!strcmp(role, "sg")) {
                    pf->off_sg = off; pf->bytes_sg = bytes;
                    pf->rows_sg = rows; pf->cols_sg = cols;
                } else if (!strcmp(role, "x")) {
                    pf->off_x = off; pf->bytes_x = bytes; pf->elems_x = cols;
                } else {
                    snprintf(err, errn, "plan-role-unknown: SPAN role=%s is not one of "
                                        "fused, down, sg, x", role);
                    fclose(fh);
                    return 1;
                }
                ++n_span;
            }
        } else if (!strncmp(line, "FORM", 4)) {
            if (m2_se_field(line, "mode", v, sizeof(v))) {
                pf->mode = m2_se_mode_of(v);
                if (pf->mode < 0) {
                    snprintf(err, errn, "plan-mode-unknown: mode=%s is not a mode the lane ships", v);
                    fclose(fh);
                    return 1;
                }
            }
            pf->sgate_form = m2_se_field(line, "sgate", v, sizeof(v)) && !strcmp(v, "perchan")
                             ? SE_SGATE_PERCHAN : SE_SGATE_SCALAR;
            pf->cast_form = m2_se_field(line, "cast", v, sizeof(v)) && !strcmp(v, "after")
                            ? SE_CAST_AFTER : SE_CAST_BEFORE;
        } else if (!strncmp(line, "FAULT", 5)) {
            if (m2_se_field(line, "name", v, sizeof(v))) snprintf(pf->fault, sizeof(pf->fault), "%s", v);
        }
    }
    fclose(fh);
    if (!saw_header) { snprintf(err, errn, "plan-header-absent: %s has no M2SEPLAN line", path); return 1; }
    if (pf->version != M2_SE_PLAN_VERSION) {
        snprintf(err, errn, "plan-version: file says %d, this arm is %d",
                 pf->version, M2_SE_PLAN_VERSION);
        return 1;
    }
    if (n_span != 4) {
        snprintf(err, errn, "plan-spans: %d SPAN records parsed, the arm needs exactly 4 "
                            "(fused, down, sg, x); a plan that parses fewer cannot be judged", n_span);
        return 1;
    }
    pf->skip_stage3 = !strcmp(pf->fault, "skip-stage3");
    pf->alloc_oversize = !strcmp(pf->fault, "alloc-oversize");
    return 0;
}

/* Every span must lie inside the slab file AND match the geometry the prototype header declares.
 * The second half is the one that matters: the plan is allowed to *address*, never to *re-shape*. */
static inline int m2_se_plan_check(const struct M2SePlanFile *pf, long slab_bytes,
                                   struct M2SePlan *p, char *err, size_t errn) {
    const long need_fused = (long)SE_TWO_I * SE_GATE_BLOCKS * SE_Q8_BYTES;
    const long need_down = (long)SE_H * SE_DOWN_BLOCKS * SE_Q8_BYTES;
    const long need_sg = (long)SE_H * 4;
    const long need_x = (long)SE_H * 2;
    if (pf->rows_fused != SE_TWO_I || pf->cols_fused != SE_H || pf->bytes_fused != need_fused) {
        snprintf(err, errn, "plan-geometry: fused span declared rows=%ld cols=%ld bytes=%ld, the "
                            "header's geometry needs rows=%d cols=%d bytes=%ld", pf->rows_fused,
                 pf->cols_fused, pf->bytes_fused, SE_TWO_I, SE_H, need_fused);
        return 1;
    }
    if (pf->rows_down != SE_H || pf->cols_down != SE_I || pf->bytes_down != need_down) {
        snprintf(err, errn, "plan-geometry: down span declared rows=%ld cols=%ld bytes=%ld, needs "
                            "%d/%d/%ld", pf->rows_down, pf->cols_down, pf->bytes_down,
                 SE_H, SE_I, need_down);
        return 1;
    }
    if (pf->rows_sg != 1 || pf->cols_sg != SE_H || pf->bytes_sg != need_sg) {
        snprintf(err, errn, "plan-geometry: sg span declared %ld/%ld/%ld, needs 1/%d/%ld",
                 pf->rows_sg, pf->cols_sg, pf->bytes_sg, SE_H, need_sg);
        return 1;
    }
    if (pf->elems_x != SE_H || pf->bytes_x != need_x) {
        snprintf(err, errn, "plan-geometry: x span declared elems=%ld bytes=%ld, needs %d/%ld",
                 pf->elems_x, pf->bytes_x, SE_H, need_x);
        return 1;
    }
    {
        struct { const char *name; long off, bytes; } sp[4];
        int i;
        sp[0].name = "fused"; sp[0].off = pf->off_fused; sp[0].bytes = pf->bytes_fused;
        sp[1].name = "down";  sp[1].off = pf->off_down;  sp[1].bytes = pf->bytes_down;
        sp[2].name = "sg";    sp[2].off = pf->off_sg;    sp[2].bytes = pf->bytes_sg;
        sp[3].name = "x";     sp[3].off = pf->off_x;     sp[3].bytes = pf->bytes_x;
        for (i = 0; i < 4; ++i) {
            if (sp[i].off < 0 || sp[i].bytes <= 0 || sp[i].off + sp[i].bytes > slab_bytes) {
                snprintf(err, errn, "plan-span-out-of-range: %s wants %ld B at +%ld, the slab is "
                                    "%ld B; refusing rather than reading past the resident bytes",
                         sp[i].name, sp[i].bytes, sp[i].off, slab_bytes);
                return 1;
            }
        }
    }
    if (pf->mode >= SE_MODE_COUNT) {
        snprintf(err, errn, "plan-mode: %d is outside the lane's mode table", pf->mode);
        return 1;
    }
    memset(p, 0, sizeof(*p));
    p->off_fused = (int32_t)pf->off_fused;
    p->off_down = (int32_t)pf->off_down;
    p->off_sg = (int32_t)pf->off_sg;
    p->off_x = (int32_t)pf->off_x;
    p->mode = pf->mode;
    p->sgate_form = pf->sgate_form;
    p->cast_form = pf->cast_form;
    p->skip_stage3 = pf->skip_stage3;
    p->alloc_oversize = pf->alloc_oversize;
    return 0;
}

static inline uint8_t *m2_se_read_file(const char *path, long *n_out, char *err, size_t errn) {
    FILE *fh = fopen(path, "rb");
    long n;
    uint8_t *buf;
    if (!fh) { snprintf(err, errn, "slab-absent: %s cannot be opened", path); return 0; }
    fseek(fh, 0, SEEK_END);
    n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    buf = (uint8_t *)malloc((size_t)(n > 0 ? n : 1));
    if (!buf) { snprintf(err, errn, "oom: %ld B", n); fclose(fh); return 0; }
    if (fread(buf, 1, (size_t)n, fh) != (size_t)n) {
        snprintf(err, errn, "slab-short: %s is %ld B on paper and short on disk", path, n);
        free(buf);
        fclose(fh);
        return 0;
    }
    fclose(fh);
    *n_out = n;
    return buf;
}

static inline int m2_se_write_f32(const char *path, const float *buf, int n, char *err, size_t errn) {
    FILE *fh = fopen(path, "wb");
    if (!fh) { snprintf(err, errn, "out-unwritable: %s", path); return 1; }
    if (fwrite(buf, sizeof(float), (size_t)n, fh) != (size_t)n) {
        snprintf(err, errn, "out-short: %s", path);
        fclose(fh);
        return 1;
    }
    fclose(fh);
    return 0;
}

static inline int m2_se_count_nan(const float *buf, int n) {
    int i, k = 0;
    for (i = 0; i < n; ++i) if (se_is_nan(buf[i])) ++k;
    return k;
}

/* Which field of the record a slot belongs to, so the report can say where the holes are. Derived
 * from the engine's single reading of the lane's `SE_OFF_*`/`SE_N_*` pairs. */
static inline void m2_se_field_report(const float *out, char *buf, size_t n) {
    int f;
    size_t used = 0;
    for (f = 0; f < SE_FIELD_COUNT; ++f) {
        const int off = M2_SE_FIELD_OFF[f], len = M2_SE_FIELD_LEN[f];
        int i, nan = 0;
        for (i = 0; i < len; ++i) if (se_is_nan(out[off + i])) ++nan;
        used += (size_t)snprintf(buf + used, n - used, "%s%s=%d/%d", f ? ";" : "",
                                 M2_SE_FIELD_NAME[f], nan, len);
    }
}

#endif /* M2_SE_IO_H */
