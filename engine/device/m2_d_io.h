/* engine/device/m2_d_io.h — the GDN recurrent arm's plan file and record IO. HOST SIDE ONLY.
 *
 * The arm is handed byte offsets, byte counts and element widths; it is never handed a tensor name, a
 * dtype enum or a path. Everything it can legitimately refuse, it refuses here, by name, before one
 * activation is read — `m2_rt_io.h` and `m2_g_io.h` are the same discipline for their arms.
 *
 * THREE REFUSALS THAT ONLY EXIST FOR THIS ARM, each of which is a control's reason to exist.
 *   1. `form-not-implemented`. The band licenses GDN rows per FORM: `band_licenses` grants
 *      `recurrent|pinned` at the pinned decode pass and grants `chunked|pinned` at no decode pass at
 *      all. An arm that accepted `form=chunked` would silently earn a recurrent licence for chunked
 *      arithmetic, so the form word is checked here and a plan naming any other form is rejected.
 *   2. `field-elt`. The scan dtype is f32 and the state is f32, while q_pre/k_pre/v_pre/beta/z and the
 *      two weights are bf16 bit images. The element width is asserted per field, not inferred from a
 *      file size (`pitfalls/30` §dtype-by-filesize), so a plan describing a Q8_0 or bf16 copy of the
 *      state cannot be satisfied by this arm.
 *   3. `spans-overlap`. d_k == d_v == 128 and the state is [h_v, d_k, d_v], so a transposed state is
 *      the SAME bytes in a different order. That makes a layout error invisible to every shape check
 *      the machine has, so this arm at least guarantees the spans it was given are disjoint and
 *      inside the slab; the transposition itself is `engine/block_device.py`'s control, not a bug.
 */
#ifndef M2_D_IO_H
#define M2_D_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_d_kernels.h"

struct M2DFieldFile {
    char name[32];
    long off, bytes, words;
    int elt;
    int seen;
};

struct M2DPlanFile {
    int version;
    long slab_bytes;
    char form_word[32], mode_word[32], fault_word[64];
    long read_off;
    int read_len;
    struct M2DFieldFile f[M2D_F_COUNT];
};

/* `TAG key=value ...`, matched on whole token starts so `bytes=` can never match inside
 * `slab_bytes=` — the difference between a parser and a substring (`pitfalls/40`). */
static inline const char *m2d_tok(const char *line, const char *key) {
    size_t n = strlen(key);
    const char *p = line;
    while (p && *p) {
        if (!strncmp(p, key, n) && p[n] == '=') return p + n + 1;
        p = strchr(p, ' ');
        while (p && *p == ' ') ++p;
    }
    return 0;
}

static inline int m2d_kv_i(const char *line, const char *key, long *out) {
    const char *p = m2d_tok(line, key);
    char *end = 0;
    long v;
    if (!p) return 0;
    v = strtol(p, &end, 10);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static inline int m2d_kv_s(const char *line, const char *key, char *out, size_t n) {
    const char *p = m2d_tok(line, key);
    size_t i = 0;
    if (!p) return 0;
    while (*p && *p != ' ' && i + 1 < n) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

static inline int m2d_read_plan(const char *path, struct M2DPlanFile *pf, char *err, size_t errn) {
    FILE *f = fopen(path, "r");
    char line[512];
    int saw_plan = 0, saw_form = 0, i;
    if (!f) { snprintf(err, errn, "%s cannot be opened", path); return 1; }
    memset(pf, 0, sizeof(*pf));
    snprintf(pf->form_word, sizeof(pf->form_word), "recurrent");
    snprintf(pf->mode_word, sizeof(pf->mode_word), "correct");
    snprintf(pf->fault_word, sizeof(pf->fault_word), "none");
    pf->read_off = -1;
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (!line[0] || line[0] == '#') continue;
        if (!strncmp(line, "M2DPLAN", 7)) {
            long v;
            saw_plan = 1;
            if (!m2d_kv_i(line, "version", &v)) {
                snprintf(err, errn, "M2DPLAN line carries no version=");
                fclose(f);
                return 1;
            }
            pf->version = (int)v;
            if (pf->version != 1) {
                snprintf(err, errn, "plan version %d; this arm speaks 1", pf->version);
                fclose(f);
                return 1;
            }
            if (!m2d_kv_i(line, "slab_bytes", &v)) {
                snprintf(err, errn, "M2DPLAN line carries no slab_bytes=; the arm will not run a plan "
                                    "against a slab whose size it has not been told");
                fclose(f);
                return 1;
            }
            pf->slab_bytes = v;
            m2d_kv_s(line, "form", pf->form_word, sizeof(pf->form_word));
        } else if (!strncmp(line, "FIELD", 5)) {
            long v;
            int slot = -1;
            char nm[32];
            if (!m2d_kv_s(line, "name", nm, sizeof(nm))) {
                snprintf(err, errn, "FIELD line with no name=");
                fclose(f);
                return 1;
            }
            for (i = 0; i < M2D_F_COUNT; ++i)
                if (!strcmp(nm, M2D_FIELD_IN_NAME[i])) slot = i;
            if (slot < 0) {
                snprintf(err, errn, "field-unknown: `%s` is not one of %s %s %s %s %s %s %s %s %s; "
                                    "this arm reads exactly these nine spans and no others",
                         nm, M2D_FIELD_IN_NAME[0], M2D_FIELD_IN_NAME[1], M2D_FIELD_IN_NAME[2],
                         M2D_FIELD_IN_NAME[3], M2D_FIELD_IN_NAME[4], M2D_FIELD_IN_NAME[5],
                         M2D_FIELD_IN_NAME[6], M2D_FIELD_IN_NAME[7], M2D_FIELD_IN_NAME[8]);
                fclose(f);
                return 1;
            }
            if (pf->f[slot].seen) {
                snprintf(err, errn, "field `%s` declared twice; two spans for one input is two "
                                    "answers to one question", nm);
                fclose(f);
                return 1;
            }
            pf->f[slot].seen = 1;
            snprintf(pf->f[slot].name, sizeof(pf->f[slot].name), "%s", nm);
            if (!m2d_kv_i(line, "span_off", &v)) { snprintf(err, errn, "FIELD %s: no span_off", nm);
                                                    fclose(f); return 1; }
            pf->f[slot].off = v;
            if (!m2d_kv_i(line, "bytes", &v)) { snprintf(err, errn, "FIELD %s: no bytes", nm);
                                                 fclose(f); return 1; }
            pf->f[slot].bytes = v;
            if (!m2d_kv_i(line, "elt", &v)) { snprintf(err, errn, "FIELD %s: no elt", nm);
                                               fclose(f); return 1; }
            pf->f[slot].elt = (int)v;
            if (!m2d_kv_i(line, "words", &v)) { snprintf(err, errn, "FIELD %s: no words", nm);
                                                 fclose(f); return 1; }
            pf->f[slot].words = v;
        } else if (!strncmp(line, "FORM", 4)) {
            long v;
            saw_form = 1;
            m2d_kv_s(line, "mode", pf->mode_word, sizeof(pf->mode_word));
            m2d_kv_s(line, "fault", pf->fault_word, sizeof(pf->fault_word));
            if (m2d_kv_i(line, "read_off", &v)) pf->read_off = v;
            if (m2d_kv_i(line, "read_len", &v)) pf->read_len = (int)v;
        } else {
            snprintf(err, errn, "unrecognised plan line `%.24s`", line);
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    if (!saw_plan) { snprintf(err, errn, "plan has no M2DPLAN line"); return 1; }
    if (!saw_form) { snprintf(err, errn, "plan has no FORM line"); return 1; }
    for (i = 0; i < M2D_F_COUNT; ++i)
        if (!pf->f[i].seen) {
            snprintf(err, errn, "field-missing: the plan never declares `%s`, which this arm has to "
                                "read; a missing input would otherwise become a plausible row computed "
                                "from whatever the slab happened to hold there",
                     M2D_FIELD_IN_NAME[i]);
            return 1;
        }
    return 0;
}

static inline int m2d_plan_check(const struct M2DPlanFile *pf, long slab_bytes, struct M2DPlan *p,
                                 char *err, size_t errn) {
    int i, j;
    if (strcmp(pf->form_word, "recurrent")) {
        snprintf(err, errn, "form-not-implemented: the plan declares form=`%s` and this arm implements "
                            "the cached single-token RECURRENT step (specification/04 §04b.1) and "
                            "nothing else. The chunked scan is a different program with its own "
                            "arithmetic (UT system, two solves, intra-chunk attention) and "
                            "speccheck/gdn_ref.py::band_licenses grants chunked|pinned at NO decode "
                            "pass, so running this body under a `chunked` label would spend a "
                            "recurrent licence on arithmetic nobody probed. Either run the chunked "
                            "arm (prototype/gdn) or ask for form=recurrent",
                 pf->form_word);
        return 1;
    }
    for (i = 0; i < M2D_F_COUNT; ++i) {
        const struct M2DFieldFile *ff = &pf->f[i];
        if (ff->elt != M2D_FIELD_IN_ELT[i]) {
            snprintf(err, errn, "field-elt: `%s` is declared elt=%d but this arm reads it as %d-byte "
                                "%s. The scan dtype is f32 and the recurrent state is f32; q_pre, "
                                "k_pre, v_pre, beta, z and both weights are bf16 bit images. A plan "
                                "naming another width is a plan for a different tensor, and reading "
                                "%lld B as %d-wide would be a silent requantization",
                     ff->name, ff->elt, M2D_FIELD_IN_ELT[i],
                     M2D_FIELD_IN_ELT[i] == 4 ? "float" : "bf16 image",
                     (long long)ff->bytes, ff->elt);
            return 1;
        }
        if (ff->words != M2D_FIELD_IN_WORDS[i]) {
            snprintf(err, errn, "field-words: `%s` declares %lld elements, the geometry says %ld "
                                "(h_k=%d h_v=%d d_k=%d d_v=%d H=%d). The record and every index in "
                                "the kernel are derived from the header's constants, so a field of a "
                                "different shape is a different tensor, not a view",
                     ff->name, (long long)ff->words, M2D_FIELD_IN_WORDS[i], GDN_H_K, GDN_H_V,
                     GDN_D_K, GDN_D_V, GDN_H);
            return 1;
        }
        if (ff->bytes != ff->words * ff->elt) {
            snprintf(err, errn, "field-bytes: `%s` is %lld B, which is not words=%lld x elt=%d",
                     ff->name, (long long)ff->bytes, (long long)ff->words, ff->elt);
            return 1;
        }
        if (ff->off < 0 || ff->bytes < 0 || ff->off + ff->bytes > slab_bytes) {
            snprintf(err, errn, "span-past-slab: `%s` at %lld+%lld does not fit the %ld B slab",
                     ff->name, (long long)ff->off, (long long)ff->bytes, slab_bytes);
            return 1;
        }
        if (ff->off % ff->elt) {
            snprintf(err, errn, "span-unaligned: `%s` at %lld is not aligned for the %d-byte reads "
                                "the kernel makes from it", ff->name, (long long)ff->off, ff->elt);
            return 1;
        }
    }
    for (i = 0; i < M2D_F_COUNT; ++i)
        for (j = i + 1; j < M2D_F_COUNT; ++j) {
            const struct M2DFieldFile *a = &pf->f[i], *b = &pf->f[j];
            if (a->off < b->off + b->bytes && b->off < a->off + a->bytes) {
                snprintf(err, errn, "spans-overlap: `%s` [%lld,%lld) and `%s` [%lld,%lld) share bytes. "
                                    "Every stage of this step reads two or three of these spans and "
                                    "writes a fourth, so overlapping spans make the answer depend on "
                                    "stage order rather than on the mathematics",
                         a->name, (long long)a->off, (long long)(a->off + a->bytes), b->name,
                         (long long)b->off, (long long)(b->off + b->bytes));
                return 1;
            }
        }
    p->form = M2D_FORM_RECURRENT;                    /* the only value reachable past the check above */
    p->mode = m2d_mode_of(pf->mode_word);
    if (p->mode < 0) {
        snprintf(err, errn, "mode-unknown: `%s` is not correct/state_xpose/decay_head_off1; those are "
                            "the honest run and the two state-reading mistakes this arm can spell "
                            "(transposed state contraction, wrong head's decay index)", pf->mode_word);
        return 1;
    }
    p->fault = m2d_fault_of(pf->fault_word);
    if (p->fault < 0) {
        snprintf(err, errn, "fault-unknown: `%s` is not none/skip-carry/skip-out/zero-state/"
                            "alloc-oversize", pf->fault_word);
        return 1;
    }
    if (pf->read_len > 0) {
        if (pf->read_off < 0 || (long)pf->read_len > slab_bytes || pf->read_off + pf->read_len > slab_bytes) {
            snprintf(err, errn, "readback-range: off=%ld len=%d is not inside the %ld B slab",
                     pf->read_off, pf->read_len, slab_bytes);
            return 1;
        }
    }
    p->slab_bytes = slab_bytes;
    for (i = 0; i < M2D_F_COUNT; ++i) {
        p->off_in[i] = pf->f[i].off;
        p->bytes_in[i] = pf->f[i].bytes;
    }
    p->read_off = pf->read_off;
    p->read_len = pf->read_len;
    return 0;
}

static inline uint8_t *m2d_read_file(const char *path, long *n, char *err, size_t errn) {
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

static inline int m2d_write_f32(const char *path, const float *out, int n, char *err, size_t errn) {
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

/* Per-field unwritten-slot census: `name=n/len;...`, space-free so one report line stays one record. */
static inline size_t m2d_field_report(const float *out, char *buf, size_t n) {
    size_t used = 0;
    int f;
    for (f = 0; f < M2D_FIELD_COUNT; ++f) {
        if (used + 1 >= n) break;
        used += (size_t)snprintf(buf + used, n - used, "%s%s=%d/%d", f ? ";" : "",
                                 M2D_FIELD_NAME[f], m2d_field_nan(out, f), M2D_FIELD_LEN[f]);
    }
    return used;
}

/* The census the driver reconciles. The state fingerprint is printed in decimal, modular, so both
 * compilers print the same number; `want_*` is printed beside it so a reader can see the denominator
 * the arm was measured against without opening the header (`pitfalls/40` §an unnamed denominator). */
static inline size_t m2d_census_report(const struct M2DCounters *ct, char *buf, size_t n) {
    size_t used = 0;
    int i;
    for (i = 0; i < M2D_IT_N && used + 40 < n; ++i)
        used += (size_t)snprintf(buf + used, n - used, "%s%s=%u/%ld", i ? ";" : "",
                                 M2D_ITEM_NAME[i], ct->items[i], m2d_item_count(i));
    snprintf(buf + used, n - used,
             ";items_total=%u/%ld;reads_q=%u/%d;reads_k=%u/%d;reads_v=%u/%d;reads_state=%u/%ld"
             ";writes_state=%u/%ld;state_nonzero=%u;state_sum=%u;state_sumidx=%u"
             ";out_elems=%u/%ld;outproj_reads=%u/%d",
             ct->items_total, m2d_items_total(),
             ct->reads_q, GDN_H_V * GDN_D_K, ct->reads_k, GDN_H_V * GDN_D_K,
             ct->reads_v, GDN_H_V * GDN_D_V, ct->reads_state, m2d_want_state_words(),
             ct->writes_state, m2d_want_state_words(),
             ct->state_nonzero, ct->state_sum, ct->state_sumidx,
             ct->out_elems, m2d_want_out_elems(), ct->outproj_reads,
             GDN_H * GDN_H_V * GDN_D_V);
    return used;
}

static inline void m2d_hex(const uint8_t *slab, long off, int len, char *buf, size_t n) {
    int i, used = 0;
    buf[0] = '\0';
    for (i = 0; i < len && (size_t)used + 3 < n; ++i)
        used += snprintf(buf + used, (size_t)(n - used), "%02x", (unsigned)slab[off + i]);
}

#endif /* M2_D_IO_H */
