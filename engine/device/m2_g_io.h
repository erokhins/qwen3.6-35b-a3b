/* engine/device/m2_g_io.h — the gather arm's plan file, census report and file IO. HOST SIDE ONLY.
 *
 * The arm is handed spans: an offset, a byte count, a kind, a geometry. It is never handed a tensor name,
 * a dtype enum or a path, and everything it can legitimately refuse it refuses here, by name, before one
 * weight element is decoded. The tags are what a control can require, and a control that cannot make the
 * arm refuse is reporting a decoration.
 *
 * TWO REFUSALS DO REAL WORK HERE and both are the shape of a claim this slice must not assume:
 *   * `path-kinds` — the plan's declared width path fixes which kinds are admissible (policy A: Q4_K
 *     gate/up and Q5_K down, 4-byte activation; ref_bf16: a bf16 matrix image, 2-byte activation). A plan
 *     that hands the shipping path a bf16 copy of an expert, or the reference path a quantized one, is
 *     refused instead of quietly decoded. That is the same discipline `m2_rt_io.h::elt=4` enforces for the
 *     router, and `gather-path-kinds` is the control that shows it bites.
 *   * `whole-tensor-unit-refused` — a unit whose span is not one expert's matrix cannot be addressed at
 *     all, so the gather arm is structurally incapable of the 1.5 GiB-per-token read
 *     (`engine/block_gate.py::gather_census`'s torch-side twin is the other half). The census counter for
 *     a whole-tensor read is witnessed non-zero by a fault leg, because a counter that can only read 0 is
 *     not a guard.
 */
#ifndef M2G_IO_H
#define M2G_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "m2_g_kernels.h"

struct M2GPlanFile {
    int version;
    long slab_bytes;
    char path_word[16];
    int slots;
    long off_x; int n_x, elt_x;
    long off_ids, off_wcast, off_wpre;
    char mode_word[24], fault_word[24];
    long corrupt_off;
    long read_off; int read_len;
    int n_units;
    struct M2GUnit units[M2G_MAX_UNITS];
};

/* Token-aware key lookup: matching on token starts is the difference between a parser and a substring
 * (`bytes=` must never match inside `span_bytes=`). */
static inline const char *m2g_tok(const char *line, const char *key) {
    size_t n = strlen(key);
    const char *p = line;
    while (p && *p) {
        if (!strncmp(p, key, n) && p[n] == '=') return p + n + 1;
        p = strchr(p, ' ');
        while (p && *p == ' ') ++p;
    }
    return 0;
}

static inline int m2g_kv_i(const char *line, const char *key, long *out) {
    const char *p = m2g_tok(line, key);
    char *end = 0;
    long v;
    if (!p) return 0;
    v = strtol(p, &end, 10);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static inline int m2g_kv_s(const char *line, const char *key, char *out, size_t n) {
    const char *p = m2g_tok(line, key);
    size_t i = 0;
    if (!p) return 0;
    while (*p && *p != ' ' && i + 1 < n) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

static inline int m2g_kind_of(const char *w) {
    if (!strcmp(w, "q4k")) return MG_KIND_Q4K;
    if (!strcmp(w, "q5k")) return MG_KIND_Q5K;
    if (!strcmp(w, "q6k")) return MG_KIND_Q6K;
    if (!strcmp(w, "bf16")) return M2G_KIND_BF16;
    return -1;
}

static inline int m2g_mode_of(const char *w) {
    if (!strcmp(w, "correct"))         return MG_MODE_CORRECT;
    if (!strcmp(w, "halves_swapped"))  return MG_MODE_HALVES_SWAPPED;
    if (!strcmp(w, "input_chunk"))     return MG_MODE_INPUT_CHUNK;
    if (!strcmp(w, "expert_rotate"))   return MG_MODE_EXPERT_ROTATE;
    if (!strcmp(w, "row_off_1"))       return MG_MODE_ROW_OFF_1;
    if (!strcmp(w, "block_off_1"))     return MG_MODE_BLOCK_OFF_1;
    if (!strcmp(w, "byte_shift_1"))    return MG_MODE_BYTE_SHIFT_1;
    if (!strcmp(w, "width_misread"))   return MG_MODE_WIDTH_MISREAD;
    if (!strcmp(w, "scale_min_naive")) return MG_MODE_SCALE_MIN_NAIVE;
    if (!strcmp(w, "q6k_d_byte_0"))    return MG_MODE_Q6K_D_BYTE_0;
    if (!strcmp(w, "w_by_expert_id"))  return MG_MODE_W_BY_EXPERT_ID;
    if (!strcmp(w, "rank_accum"))      return MG_MODE_RANK_ACCUM;
    if (!strcmp(w, "cast_after"))      return MG_MODE_CAST_AFTER;
    if (!strcmp(w, "f32_combine"))     return MG_MODE_F32_COMBINE;
    if (!strcmp(w, "poison_nan"))      return MG_MODE_POISON_NAN;
    if (!strcmp(w, "acc_bf16"))        return M2G_MODE_ACC_BF16;
    if (!strcmp(w, "acc_f32"))         return M2G_MODE_ACC_F32;
    if (!strcmp(w, "silu_exp_ulp1"))   return M2G_MODE_EXP_ULP1;
    return -1;
}

static inline int m2g_read_plan(const char *path, struct M2GPlanFile *pf, char *err, size_t errn) {
    FILE *f = fopen(path, "r");
    char line[768];
    int saw_plan = 0, saw_x = 0, saw_ids = 0, saw_form = 0;
    if (!f) { snprintf(err, errn, "%s cannot be opened", path); return 1; }
    memset(pf, 0, sizeof(*pf));
    pf->corrupt_off = -1;
    pf->read_off = -1;
    snprintf(pf->mode_word, sizeof(pf->mode_word), "correct");
    snprintf(pf->fault_word, sizeof(pf->fault_word), "none");
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (!line[0] || line[0] == '#') continue;
        if (!strncmp(line, "M2GPLAN", 7)) {
            long v;
            saw_plan = 1;
            if (!m2g_kv_i(line, "version", &v)) { fclose(f); snprintf(err, errn, "no version="); return 1; }
            pf->version = (int)v;
            if (pf->version != 1) { fclose(f); snprintf(err, errn, "plan version %d; this arm speaks 1",
                                                        pf->version); return 1; }
            if (!m2g_kv_s(line, "path", pf->path_word, sizeof(pf->path_word))) {
                fclose(f); snprintf(err, errn, "plan-rejected path=… is required: the arm will not guess "
                                    "which width it is being asked to run"); return 1;
            }
            if (m2g_kv_i(line, "slots", &v)) pf->slots = (int)v;
            if (m2g_kv_i(line, "slab_bytes", &v)) pf->slab_bytes = v;
        } else if (!strncmp(line, "X ", 2)) {
            long v;
            saw_x = 1;
            if (!m2g_kv_i(line, "span_off", &v)) { fclose(f); snprintf(err, errn, "X: no span_off"); return 1; }
            pf->off_x = v;
            if (!m2g_kv_i(line, "bytes", &v)) { fclose(f); snprintf(err, errn, "X: no bytes"); return 1; }
            pf->n_x = (int)v;
            if (!m2g_kv_i(line, "elt", &v)) { fclose(f); snprintf(err, errn, "X: no elt"); return 1; }
            pf->elt_x = (int)v;
        } else if (!strncmp(line, "IDS", 3)) {
            long v;
            saw_ids = 1;
            if (!m2g_kv_i(line, "span_off", &v)) { fclose(f); snprintf(err, errn, "IDS: no span_off"); return 1; }
            pf->off_ids = v;
        } else if (!strncmp(line, "WCAST", 5)) {
            long v;
            if (!m2g_kv_i(line, "span_off", &v)) { fclose(f); snprintf(err, errn, "WCAST: no span_off"); return 1; }
            pf->off_wcast = v;
        } else if (!strncmp(line, "WPRE", 4)) {
            long v;
            if (!m2g_kv_i(line, "span_off", &v)) { fclose(f); snprintf(err, errn, "WPRE: no span_off"); return 1; }
            pf->off_wpre = v;
        } else if (!strncmp(line, "UNIT", 4)) {
            long v;
            char w[16];
            struct M2GUnit *u;
            if (pf->n_units >= M2G_MAX_UNITS) {
                fclose(f); snprintf(err, errn, "more than %d units on one plan", M2G_MAX_UNITS); return 1;
            }
            u = &pf->units[pf->n_units];
            saw_ids = saw_ids;                       /* UNIT lines are counted separately */
            if (!m2g_kv_s(line, "role", w, sizeof(w))) { fclose(f); snprintf(err, errn, "UNIT: no role"); return 1; }
            u->role = (!strcmp(w, "gate")) ? 0 : (!strcmp(w, "up")) ? 1 : (!strcmp(w, "down")) ? 2 : -1;
            if (u->role < 0) { fclose(f); snprintf(err, errn, "UNIT role=%s is not gate/up/down", w); return 1; }
            if (!m2g_kv_i(line, "slot", &v)) { fclose(f); snprintf(err, errn, "UNIT: no slot"); return 1; }
            u->slot = (int)v;
            if (!m2g_kv_i(line, "expert", &v)) { fclose(f); snprintf(err, errn, "UNIT: no expert"); return 1; }
            u->expert = (int)v;
            if (!m2g_kv_i(line, "off", &v)) { fclose(f); snprintf(err, errn, "UNIT: no off"); return 1; }
            u->off = v;
            if (!m2g_kv_i(line, "bytes", &v)) { fclose(f); snprintf(err, errn, "UNIT: no bytes"); return 1; }
            u->bytes = v;
            if (!m2g_kv_i(line, "rows", &v)) { fclose(f); snprintf(err, errn, "UNIT: no rows"); return 1; }
            u->rows = (int)v;
            if (!m2g_kv_i(line, "cols", &v)) { fclose(f); snprintf(err, errn, "UNIT: no cols"); return 1; }
            u->cols = (int)v;
            if (!m2g_kv_s(line, "kind", w, sizeof(w))) { fclose(f); snprintf(err, errn, "UNIT: no kind"); return 1; }
            u->kind = m2g_kind_of(w);
            if (u->kind < 0) { fclose(f); snprintf(err, errn, "UNIT kind=%s is unknown to this arm", w); return 1; }
            u->elt = (u->kind == M2G_KIND_BF16) ? 2 : 1;
            if (m2g_kv_i(line, "elt", &v)) u->elt = (int)v;
            u->span_off = u->off;
            u->span_bytes = u->bytes;
            if (m2g_kv_i(line, "span_off", &v)) u->span_off = v;
            if (m2g_kv_i(line, "span_bytes", &v)) u->span_bytes = v;
            u->blocks = 0;
            if (m2g_kv_i(line, "blocks", &v)) u->blocks = (int)v;
            u->fused = 0;
            if (m2g_kv_i(line, "fused", &v)) u->fused = (int)v;
            u->whole = 0;
            if (m2g_kv_i(line, "whole", &v)) u->whole = (int)v;
            pf->n_units++;
        } else if (!strncmp(line, "FORM", 4)) {
            long v;
            saw_form = 1;
            if (m2g_kv_s(line, "mode", pf->mode_word, sizeof(pf->mode_word)) == 0) {
                fclose(f); snprintf(err, errn, "FORM: no mode="); return 1;
            }
            if (m2g_kv_s(line, "fault", pf->fault_word, sizeof(pf->fault_word)) == 0)
                snprintf(pf->fault_word, sizeof(pf->fault_word), "none");
            if (m2g_kv_i(line, "corrupt_off", &v)) pf->corrupt_off = v;
            if (m2g_kv_i(line, "read_off", &v)) pf->read_off = v;
            if (m2g_kv_i(line, "read_len", &v)) pf->read_len = (int)v;
        }
    }
    fclose(f);
    if (!saw_plan || !saw_x || !saw_ids || !saw_form) {
        snprintf(err, errn, "plan is missing %s%s%s%s; a plan this arm parses nothing from is a plan it "
                 "cannot run", saw_plan ? "" : "M2GPLAN ", saw_x ? "" : "X ",
                 saw_ids ? "" : "IDS ", saw_form ? "" : "FORM ");
        return 1;
    }
    return 0;
}

/* The check: every claim the plan makes that this arm can disprove. Returns non-zero and fills `err` with
 * the refusal tag plus the reason. `plan` is filled only on success. */
static inline int m2g_plan_check(const struct M2GPlanFile *pf, long slab_bytes, struct M2GPlan *plan,
                                 char *err, size_t errn) {
    int path, i, bad_units = 0;
    long need_x;
    if (!strcmp(pf->path_word, "ref_bf16")) path = M2G_PATH_REF_BF16;
    else if (!strcmp(pf->path_word, "policy_a")) path = M2G_PATH_POLICY_A;
    else {
        snprintf(err, errn, "path-unknown: path=%s is neither ref_bf16 nor policy_a", pf->path_word);
        return 1;
    }
    if (pf->slots != MG_K) {
        snprintf(err, errn, "slots-not-geometry: the plan declares %d slots and the geometry read from "
                 "prototype/moe_gather says MG_K=%d; the row has no other width", pf->slots, MG_K);
        return 1;
    }
    if (pf->elt_x != (path == M2G_PATH_POLICY_A ? 4 : 2)) {
        snprintf(err, errn, "elt-mismatch-for-path: path=%s stores the activation as %d B/elt and the plan "
                 "declares %d; those are two different shapes of x", pf->path_word,
                 path == M2G_PATH_POLICY_A ? 4 : 2, pf->elt_x);
        return 1;
    }
    memset(plan, 0, sizeof(*plan));
    plan->path = path;
    plan->slots = pf->slots;
    plan->off_x = pf->off_x;
    plan->elt_x = pf->elt_x;
    plan->off_ids = pf->off_ids;
    plan->off_wcast = pf->off_wcast;
    plan->off_wpre = pf->off_wpre;
    plan->corrupt_off = pf->corrupt_off;
    plan->read_off = pf->read_off;
    plan->read_len = pf->read_len;
    plan->slab_bytes = slab_bytes;
    plan->mode = m2g_mode_of(pf->mode_word);
    if (plan->mode < 0 || !m2g_mode_known(plan->mode)) {
        snprintf(err, errn, "mode-unknown: mode=%s is not a word this arm implements; an injection it "
                 "cannot run cannot be a control", pf->mode_word);
        return 1;
    }
    if (!strcmp(pf->fault_word, "none")) ;
    else if (!strcmp(pf->fault_word, "skip-raw")) plan->fault_skip_raw = 1;
    else if (!strcmp(pf->fault_word, "skip-routed")) plan->fault_skip_routed = 1;
    else if (!strcmp(pf->fault_word, "whole-note")) plan->fault_whole_note = 1;
    else if (!strcmp(pf->fault_word, "corrupt-byte")) {
        if (pf->corrupt_off < 0 || pf->corrupt_off >= slab_bytes) {
            snprintf(err, errn, "corrupt-off: the fault wants to flip byte %ld, outside the %ld B slab",
                     pf->corrupt_off, slab_bytes);
            return 1;
        }
    } else if (strcmp(pf->fault_word, "alloc-oversize")) {
        snprintf(err, errn, "fault-unknown: fault=%s is not none/skip-raw/skip-routed/whole-note/"
                 "corrupt-byte/alloc-oversize", pf->fault_word);
        return 1;
    }
    need_x = (long)pf->n_x;
    if (pf->off_x < 0 || pf->off_x + need_x > slab_bytes) {
        snprintf(err, errn, "span-out-of-range: the activation span is %ld B at %ld and the slab is %ld B",
                 need_x, pf->off_x, slab_bytes);
        return 1;
    }
    if (pf->off_ids < 0 || pf->off_ids + (long)sizeof(int32_t) * pf->slots > slab_bytes
        || pf->off_wcast < 0 || pf->off_wcast + (long)sizeof(float) * pf->slots > slab_bytes
        || pf->off_wpre < 0 || pf->off_wpre + (long)sizeof(float) * pf->slots > slab_bytes) {
        snprintf(err, errn, "span-out-of-range: the router arm's selection or weights run past the slab");
        return 1;
    }
    if (pf->n_units != pf->slots * 3) {
        snprintf(err, errn, "unit-count: %d units for %d slots; the gather reads 3 matrices per selected "
                 "expert and anything else is a different op", pf->n_units, pf->slots);
        return 1;
    }
    for (i = 0; i < pf->n_units; ++i) {
        const struct M2GUnit *u = &pf->units[i];
        char tag[64];
        snprintf(tag, sizeof(tag), "unit[slot=%d role=%s]", u->slot,
                 u->role == 0 ? "gate" : (u->role == 1 ? "up" : "down"));
        if (u->slot < 0 || u->slot >= pf->slots) {
            snprintf(err, errn, "slot-index: %s has slot %d, outside [0,%d)", tag, u->slot, pf->slots);
            return 1;
        }
        if (u->expert < 0 || u->expert >= MG_E) {
            snprintf(err, errn, "expert-index: %s names expert %d, outside [0,%d); an out-of-range expert "
                     "is the signature of a router that picked something this tensor does not contain",
                     tag, u->expert, MG_E);
            return 1;
        }
        if (u->whole) {
            snprintf(err, errn, "whole-tensor-unit-refused: %s declares whole=1 over %ld B. The gather "
                     "arm addresses one expert per unit and will not decode a whole tensor: reading "
                     "ffn_gate_exps end to end per token is the failure the census exists to catch",
                     tag, u->bytes);
            return 1;
        }
        if (u->off < 0 || u->off + u->bytes > slab_bytes
            || u->span_off < 0 || u->span_off + u->span_bytes > slab_bytes) {
            snprintf(err, errn, "span-out-of-range: %s runs past the %ld B slab (off=%ld bytes=%ld "
                     "span_off=%ld span_bytes=%ld)", tag, slab_bytes, u->off, u->bytes, u->span_off,
                     u->span_bytes);
            return 1;
        }
        if (path == M2G_PATH_POLICY_A) {
            if (u->kind == M2G_KIND_BF16) {
                snprintf(err, errn, "path-kinds: %s declares a bf16 span on path=policy_a. The shipping "
                         "shape stores gate/up Q4_K and down Q5_K (D-01); a bf16 copy is a different "
                         "artifact and must not be judged as policy A", tag);
                return 1;
            }
            if (u->role != 2 && u->kind != MG_KIND_Q4K) {
                snprintf(err, errn, "path-kinds: %s declares %s on the gate/up family, which policy A "
                         "stores Q4_K", tag, pf->units[i].kind == MG_KIND_Q5K ? "q5k" : "q6k");
                return 1;
            }
            if (u->role == 2 && u->kind != MG_KIND_Q5K && u->kind != MG_KIND_Q6K) {
                snprintf(err, errn, "path-kinds: %s declares q4k on the down family, which policy A "
                         "stores Q5_K (layer 0) — a width swap is not a substitution this arm accepts", tag);
                return 1;
            }
        } else {
            if (u->kind != M2G_KIND_BF16) {
                snprintf(err, errn, "path-kinds: %s declares a quantized span on path=ref_bf16. The "
                         "reference path reads the bf16 matrix image the pinned dumps were made from", tag);
                return 1;
            }
        }
        if (u->role == 2) {
            if (u->rows != MG_H || u->cols != MG_I) {
                snprintf(err, errn, "unit-geometry: %s is [%d,%d]; down is [%d,%d]", tag, u->rows,
                         u->cols, MG_H, MG_I);
                return 1;
            }
        } else {
            if (u->rows != MG_I || u->cols != MG_H) {
                snprintf(err, errn, "unit-geometry: %s is [%d,%d]; gate/up are [%d,%d]", tag, u->rows,
                         u->cols, MG_I, MG_H);
                return 1;
            }
        }
        if (u->kind == M2G_KIND_BF16) {
            if (u->elt != 2 || u->bytes != (long)u->rows * u->cols * 2) {
                snprintf(err, errn, "unit-span-width: %s is a bf16 [%d,%d] matrix and spans %ld B with "
                         "elt=%d; the image is rows*cols*2 bytes, so a shorter span is a strided read "
                         "wearing the name of a slice", tag, u->rows, u->cols, u->bytes, u->elt);
                return 1;
            }
        } else {
            const int bb = mg_block_bytes(u->kind);
            if (u->elt != 1 || u->blocks <= 0
                || (long)u->blocks * MG_QK_K != (long)u->rows * u->cols
                || u->bytes != (long)u->blocks * bb) {
                snprintf(err, errn, "unit-span-width: %s is a %s [%d,%d] matrix: %d blocks of %d elements "
                         "at %d B is %ld B, the plan says %ld B (elt=%d). One block cannot be split, so a "
                         "span that disagrees with its own geometry is refused here, not misdecoded",
                         tag, u->kind == MG_KIND_Q4K ? "Q4_K" : (u->kind == MG_KIND_Q5K ? "Q5_K" : "Q6_K"),
                         u->rows, u->cols, u->blocks, MG_QK_K, bb,
                         (long)u->blocks * bb, u->bytes, u->elt);
                return 1;
            }
        }
        if (u->fused) {
            const long one = (u->kind == M2G_KIND_BF16)
                                 ? (long)u->rows * u->cols * 2
                                 : ((long)u->rows * u->cols / MG_QK_K) * mg_block_bytes(u->kind);
            if (u->role == 2 || u->span_bytes != 2 * one) {
                snprintf(err, errn, "fused-span: %s declares fused=1 over a %ld B span. The gate|up pair "
                         "is [%d,%d] of one element kind — %ld B for two halves of this kind — and a fused "
                         "down is not a layout anyone stores", tag, u->span_bytes, MG_TWO_I, MG_H, 2 * one);
                return 1;
            }
        }
        plan->units[plan->n_units++] = *u;
        ++bad_units;
    }
    /* One fused pair must be ONE span, named identically by its two views: two self-consistent layouts
     * joined by a branch is the defect class this repo refuses twice (pitfalls/40). */
    for (i = 0; i < plan->n_units; ++i) {
        int j, mate = -1;
        if (plan->units[i].role != 0 || !plan->units[i].fused) continue;
        for (j = 0; j < plan->n_units; ++j)
            if (plan->units[j].slot == plan->units[i].slot && plan->units[j].role == 1) mate = j;
        if (mate < 0 || plan->units[mate].span_off != plan->units[i].span_off
            || plan->units[mate].span_bytes != plan->units[i].span_bytes
            || plan->units[mate].off == plan->units[i].off) {
            snprintf(err, errn, "fused-pair: slot %d's gate and up views of the fused span do not name the "
                     "same span_off/span_bytes, or they name the same offset — the pair is two halves of "
                     "one [2I,H] block, and gate-first is what §11.4 and prediction 3 say",
                     plan->units[i].slot);
            return 1;
        }
    }
    if (plan->mode == MG_MODE_INPUT_CHUNK) {
        int any = 0;
        for (i = 0; i < plan->n_units; ++i) if (plan->units[i].fused) any = 1;
        if (!any) {
            snprintf(err, errn, "mode-unreachable-here: mode=input_chunk re-chunks a fused [2I,H] span and "
                     "this plan has none (the artifact stores gate and up as two tensors), so the leg "
                     "cannot execute here and must not be reported as fired");
            return 1;
        }
    }
    if (plan->mode == MG_MODE_WIDTH_MISREAD || plan->mode == MG_MODE_Q6K_D_BYTE_0) {
        int any = 0;
        for (i = 0; i < plan->n_units; ++i)
            if (plan->units[i].role == 2 && plan->units[i].kind != MG_KIND_Q4K) any = 1;
        if (!any) {
            snprintf(err, errn, "mode-unreachable-here: mode=%s is a down-family width misreading and no "
                     "down unit here is wider than Q4_K", pf->mode_word);
            return 1;
        }
    }
    (void)bad_units;
    return 0;
}

/* ---- file IO and report strings ------------------------------------------------------------------ */
static inline uint8_t *m2g_read_file(const char *path, long *n_out, char *err, size_t errn) {
    FILE *f = fopen(path, "rb");
    long n;
    uint8_t *buf;
    if (!f) { snprintf(err, errn, "%s cannot be opened", path); return 0; }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (uint8_t *)malloc((size_t)(n > 0 ? n : 1));
    if (!buf) { fclose(f); snprintf(err, errn, "out of memory reading %s", path); return 0; }
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(buf); snprintf(err, errn, "%s short read", path); return 0;
    }
    fclose(f);
    *n_out = n;
    return buf;
}

static inline int m2g_write_f32(const char *path, const float *v, long n, char *err, size_t errn) {
    FILE *f = fopen(path, "wb");
    if (!f) { snprintf(err, errn, "%s cannot be opened for writing", path); return 1; }
    if (fwrite(v, sizeof(float), (size_t)n, f) != (size_t)n) {
        fclose(f); snprintf(err, errn, "%s short write", path); return 1;
    }
    fclose(f);
    return 0;
}

/* The unwritten-slot census, printed so a control can require it non-zero. */
static inline void m2g_field_report(const float *out, char *buf, size_t n) {
    int i, used = 0;
    buf[0] = '\0';
    for (i = 0; i < M2G_FIELD_COUNT; ++i)
        used += snprintf(buf + used, n - (size_t)used, "%s%s=%d", i ? "," : "", M2G_FIELD_NAME[i],
                         m2g_field_nan(out, i));
}

/* The census, in the form a scheduler decision gets read out of: per expert_index, the number of matrix
 * spans gathered and the bytes they cover, plus the two counters a lie would have to move. `;` between
 * keys and no spaces anywhere: `engine/device_arm.py::run_arm` parses the report as whitespace-separated
 * key=value pairs, and a census carrying a space would be read as two facts, neither of them true. */
static inline void m2g_census_report(const struct M2GCounters *ct, const struct M2GPlan *plan,
                                     char *buf, size_t n) {
    int i, used = 0, distinct = 0;
    long want_bytes = m2g_want_bytes(plan);
    for (i = 0; i < MG_E; ++i) if (ct->units_by_expert[i]) distinct++;
    buf[0] = '\0';
    used += snprintf(buf + used, n - (size_t)used,
                     "slots=%u;unit_reads=%u;want_units=%d;distinct_experts=%d;want_experts=%d"
                     ";whole_reads=%u;elt_reads=%u;route_n=%u;route_bad=%u;note_conflict=%u;"
                     "bytes_gathered=",
                     ct->slots, ct->units, m2g_want_units(plan), distinct, plan->slots,
                     ct->whole_reads, ct->elts, ct->route_n, ct->route_bad, ct->note_conflict);
    for (i = 0; i < MG_E && (size_t)used + 40 < n; ++i)
        if (ct->units_by_expert[i])
            used += snprintf(buf + used, n - (size_t)used, "%d:%u/%uB,", i, ct->units_by_expert[i],
                             ct->bytes_by_expert[i]);
    {
        long tot = 0;
        for (i = 0; i < MG_E; ++i) tot += (long)ct->bytes_by_expert[i];
        snprintf(buf + used, n - (size_t)used, ";total=%ldB;want_bytes=%ldB;spans=%d", tot, want_bytes,
                 m2g_want_spans(plan));
    }
}

/* A small hex window of the slab, so a control can compare what the device holds against what the file
 * holds without the arm needing a hash function of its own. */
static inline void m2g_hex(const char *slab, long off, int len, char *buf, size_t n) {
    int i, used = 0;
    buf[0] = '\0';
    for (i = 0; i < len && (size_t)used + 3 < n; ++i)
        used += snprintf(buf + used, n - (size_t)used, "%02x",
                         (unsigned)((uint8_t)slab[off + i]));
}

#endif                                                 /* M2G_IO_H */
