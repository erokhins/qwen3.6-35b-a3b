/* engine/device/m2_rt_io.h — the router arm's plan file and record IO. HOST SIDE ONLY.
 *
 * The arm is handed an offset, a byte count and an element size; it is not handed a tensor name, a
 * dtype enum or a path. Everything it can legitimately refuse it refuses here, by name, before a
 * single weight is read — `m2_se_io.h` is the same discipline for the shared-expert arm and the tags
 * are what a control can require.
 *
 * The width check lives here on purpose. `elt=4` is not taken on faith: the element size has to
 * divide the span, the span has to be exactly rows*cols*elt, and the offset has to be aligned for the
 * access the kernel is about to make. A plan that describes a bf16 or Q8_0 copy of the router matrix
 * cannot be satisfied by this arm, and `router-span-is-f32` in `engine/block_device.py` is the
 * control that shows that refusal is load-bearing rather than decorative.
 */
#ifndef M2_RT_IO_H
#define M2_RT_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_rt_kernels.h"

struct M2RtPlanFile {
    int version;
    long slab_bytes;
    long off_w, n_w;
    long off_x, n_x;
    int rows, cols, top_k, elt;
    int mode, cast_form, skip_stage, tie_forge;
    char mode_word[32], cast_word[32], skip_word[32], fault[64];
};

/* The plan is `TAG key=value key=value ...`; a key is a whole token, so `bytes=` never matches inside
 * `slab_bytes=`. Matching on token starts is the difference between a parser and a substring. */
static inline const char *m2_rt_tok(const char *line, const char *key) {
    size_t n = strlen(key);
    const char *p = line;
    while (p && *p) {
        if (!strncmp(p, key, n) && p[n] == '=') return p + n + 1;
        p = strchr(p, ' ');
        while (p && *p == ' ') ++p;
    }
    return 0;
}

static inline int m2_rt_kv_i(const char *line, const char *key, long *out) {
    const char *p = m2_rt_tok(line, key);
    char *end = 0;
    long v;
    if (!p) return 0;
    v = strtol(p, &end, 10);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static inline int m2_rt_kv_s(const char *line, const char *key, char *out, size_t n) {
    const char *p = m2_rt_tok(line, key);
    size_t i = 0;
    if (!p) return 0;
    while (*p && *p != ' ' && i + 1 < n) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

static inline int m2_rt_read_plan(const char *path, struct M2RtPlanFile *pf, char *err, size_t errn) {
    FILE *f = fopen(path, "r");
    char line[512];
    int saw_plan = 0, saw_router = 0, saw_x = 0, saw_form = 0;
    if (!f) { snprintf(err, errn, "%s cannot be opened", path); return 1; }
    memset(pf, 0, sizeof(*pf));
    snprintf(pf->mode_word, sizeof(pf->mode_word), "correct");
    snprintf(pf->cast_word, sizeof(pf->cast_word), "act_dtype");
    snprintf(pf->skip_word, sizeof(pf->skip_word), "none");
    snprintf(pf->fault, sizeof(pf->fault), "none");
    pf->top_k = MR_K;
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (!line[0] || line[0] == '#') continue;
        if (!strncmp(line, "M2RTPLAN", 8)) {
            long v;
            saw_plan = 1;
            if (!m2_rt_kv_i(line, "version", &v)) {
                snprintf(err, errn, "M2RTPLAN line carries no version=");
                fclose(f);
                return 1;
            }
            pf->version = (int)v;
            if (pf->version != 1) {
                snprintf(err, errn, "plan version %d; this arm speaks 1", pf->version);
                fclose(f);
                return 1;
            }
            if (m2_rt_kv_i(line, "slab_bytes", &v)) pf->slab_bytes = v;
        } else if (!strncmp(line, "ROUTER", 6)) {
            long v;
            saw_router = 1;
            if (!m2_rt_kv_i(line, "span_off", &v)) { snprintf(err, errn, "ROUTER: no span_off");
                                                    fclose(f); return 1; }
            pf->off_w = v;
            if (!m2_rt_kv_i(line, "bytes", &v)) { snprintf(err, errn, "ROUTER: no bytes");
                                                  fclose(f); return 1; }
            pf->n_w = v;
            if (!m2_rt_kv_i(line, "rows", &v)) { snprintf(err, errn, "ROUTER: no rows");
                                                 fclose(f); return 1; }
            pf->rows = (int)v;
            if (!m2_rt_kv_i(line, "cols", &v)) { snprintf(err, errn, "ROUTER: no cols");
                                                 fclose(f); return 1; }
            pf->cols = (int)v;
            if (!m2_rt_kv_i(line, "elt", &v)) { snprintf(err, errn, "ROUTER: no elt");
                                                fclose(f); return 1; }
            pf->elt = (int)v;
            if (m2_rt_kv_i(line, "top_k", &v)) pf->top_k = (int)v;
        } else if (!strncmp(line, "X ", 2)) {
            long v;
            saw_x = 1;
            if (!m2_rt_kv_i(line, "span_off", &v)) { snprintf(err, errn, "X: no span_off");
                                                    fclose(f); return 1; }
            pf->off_x = v;
            if (!m2_rt_kv_i(line, "bytes", &v)) { snprintf(err, errn, "X: no bytes");
                                                  fclose(f); return 1; }
            pf->n_x = v;
        } else if (!strncmp(line, "FORM", 4)) {
            long v;
            saw_form = 1;
            m2_rt_kv_s(line, "mode", pf->mode_word, sizeof(pf->mode_word));
            m2_rt_kv_s(line, "cast", pf->cast_word, sizeof(pf->cast_word));
            m2_rt_kv_s(line, "skip", pf->skip_word, sizeof(pf->skip_word));
            m2_rt_kv_s(line, "fault", pf->fault, sizeof(pf->fault));
            if (m2_rt_kv_i(line, "tie_forge", &v)) pf->tie_forge = (int)v;
        } else {
            snprintf(err, errn, "unrecognised plan line `%.24s`", line);
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    if (!(saw_plan && saw_router && saw_x && saw_form)) {
        snprintf(err, errn, "plan is missing %s%s%s%s", saw_plan ? "" : "M2RTPLAN ",
                 saw_router ? "" : "ROUTER ", saw_x ? "" : "X ", saw_form ? "" : "FORM ");
        return 1;
    }
    return 0;
}

static inline int m2_rt_plan_check(const struct M2RtPlanFile *pf, long slab_bytes,
                                   struct M2RtPlan *p, char *err, size_t errn) {
    if (pf->elt != 4) {
        snprintf(err, errn, "router plan declares elt=%d bytes per element; the policy-A router "
                            "matrix is F32-exact (DECISIONS.md D-01a) and this arm reads it as 4-byte "
                            "floats, so a plan naming any other width is a plan for a different "
                            "tensor. Refusing rather than reading %lld B as %d-wide",
                 pf->elt, (long long)pf->n_w, pf->elt);
        return 1;
    }
    if (pf->rows != MR_E || pf->cols != MR_H || pf->top_k != MR_K) {
        snprintf(err, errn, "plan geometry rows=%d cols=%d top_k=%d does not match the prototype "
                            "header's MR_E=%d MR_H=%d MR_K=%d; the record layout is derived from "
                            "those constants, so a plan that disagrees would write fields over each "
                            "other", pf->rows, pf->cols, pf->top_k, MR_E, MR_H, MR_K);
        return 1;
    }
    if ((long)pf->rows * (long)pf->cols * (long)pf->elt != pf->n_w) {
        snprintf(err, errn, "router span is %lld B but rows*cols*elt is %lld B; those are two "
                            "different matrices", (long long)pf->n_w,
                 (long long)((long)pf->rows * (long)pf->cols * (long)pf->elt));
        return 1;
    }
    if (pf->off_w % 4 || pf->off_x % 4) {
        snprintf(err, errn, "router span at %lld and activation span at %lld are not 4-byte aligned; "
                            "the arm reads floats from both, and an unaligned read would be a second "
                            "bug pretending to be the first",
                 (long long)pf->off_w, (long long)pf->off_x);
        return 1;
    }
    if (pf->n_x != (long)pf->cols * 2) {
        snprintf(err, errn, "activation span is %lld B, which is not cols=%d bf16 elements",
                 (long long)pf->n_x, pf->cols);
        return 1;
    }
    if (pf->off_w < 0 || pf->n_w < 0 || pf->off_w + pf->n_w > slab_bytes
        || pf->off_x < 0 || pf->n_x < 0 || pf->off_x + pf->n_x > slab_bytes) {
        snprintf(err, errn, "a span runs past the %ld B slab (router %ld+%lld, activation %ld+%lld)",
                 slab_bytes, (long)pf->off_w, (long long)pf->n_w, (long)pf->off_x,
                 (long long)pf->n_x);
        return 1;
    }
    p->mode = m2_rt_mode_of(pf->mode_word);
    if (p->mode < 0) {
        snprintf(err, errn, "mode word `%s` is not one of the prototype lane's named forms "
                            "(correct softmax_bf16 no_renorm row_off1 wg_transposed tie_high "
                            "topk_ascending no_max_sub poison_nan cast_after rank_accum "
                            "w_by_expert_id f32_combine)", pf->mode_word);
        return 1;
    }
    if (!m2_rt_mode_reaches_anything(p->mode)) {
        snprintf(err, errn, "mode-unreachable-here: %s reaches no field this slice emits, per the "
                            "prototype's own reach table `mr_mode_reaches`. This arm computes no "
                            "routed sum, so accepting the mode would be accepting a no-op and calling "
                            "it a control", pf->mode_word);
        return 1;
    }
    if (!strcmp(pf->cast_word, "act_dtype")) p->cast_form = M2_RT_CAST_ACT_DTYPE;
    else if (!strcmp(pf->cast_word, "arm_basis")) p->cast_form = M2_RT_CAST_ARM_BASIS;
    else {
        snprintf(err, errn, "cast word `%s` is neither act_dtype nor arm_basis; those are the two "
                            "answers to what dtype R1 lands in, and there is no third",
                 pf->cast_word);
        return 1;
    }
    if (!strcmp(pf->skip_word, "none")) p->skip_stage = M2_RT_SKIP_NONE;
    else if (!strcmp(pf->skip_word, "renorm")) p->skip_stage = M2_RT_SKIP_RENORM;
    else if (!strcmp(pf->skip_word, "topk")) p->skip_stage = M2_RT_SKIP_TOPK;
    else {
        snprintf(err, errn, "skip word `%s` is not none/renorm/topk", pf->skip_word);
        return 1;
    }
    if (pf->tie_forge != 0 && pf->tie_forge != 1) {
        snprintf(err, errn, "tie_forge=%d is neither 0 nor 1", pf->tie_forge);
        return 1;
    }
    p->off_w = pf->off_w;
    p->off_x = pf->off_x;
    p->n_w_bytes = pf->n_w;
    p->rows = pf->rows;
    p->cols = pf->cols;
    p->top_k = pf->top_k;
    p->elt = pf->elt;
    p->tie_forge = pf->tie_forge;
    return 0;
}

static inline uint8_t *m2_rt_read_file(const char *path, long *n, char *err, size_t errn) {
    FILE *f = fopen(path, "rb");
    long size;
    uint8_t *buf;
    if (!f) { snprintf(err, errn, "%s cannot be opened", path); return 0; }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (uint8_t *)malloc((size_t)(size > 0 ? size : 1));
    if (!buf) { snprintf(err, errn, "out of memory reading %s", path); fclose(f); return 0; }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        snprintf(err, errn, "%s: short read", path);
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);
    *n = size;
    return buf;
}

static inline int m2_rt_write_f32(const char *path, const float *out, int n, char *err, size_t errn) {
    FILE *f = fopen(path, "wb");
    if (!f) { snprintf(err, errn, "%s cannot be written", path); return 1; }
    if (fwrite(out, sizeof(float), (size_t)n, f) != (size_t)n) {
        snprintf(err, errn, "%s: short write", path);
        fclose(f);
        return 1;
    }
    if (fclose(f)) { snprintf(err, errn, "%s: close failed", path); return 1; }
    (void)errn;
    return 0;
}

/* Per-field poison census: `name=n/len;...`, space-free so one report line stays one record. */
static inline size_t m2_rt_field_report(const float *out, char *buf, size_t n) {
    size_t used = 0;
    int f;
    for (f = 0; f < M2_RT_FIELD_COUNT; ++f) {
        if (used + 1 >= n) break;
        used += (size_t)snprintf(buf + used, n - used, "%s%s=%d/%d", f ? ";" : "",
                                 M2_RT_FIELD_NAME[f], m2_rt_field_nan(out, f),
                                 M2_RT_FIELD_LEN[f]);
    }
    return used;
}

#endif /* M2_RT_IO_H */
