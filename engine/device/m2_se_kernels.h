/* engine/device/m2_se_kernels.h — the shared expert's layer-0 decode step, composed from the
 * gated prototype arithmetic, for ONE source compiled by TWO compilers.
 *
 * WHY THIS FILE IS THIS SMALL. Every arithmetic decision belongs to
 * `prototype/shared_expert/shared_expert_kernels.h`, which is included verbatim below and is NOT
 * edited by anyone (it is read-only material for this lane). What M2 adds is only the *address
 * arrangement* the engine's policy-A resident set needs and the *item order* one decode step runs
 * in: the prototype's stage bodies take `SeWeight`/`SeAct`/`SeCase` records and read through
 * `p->slab + byte_off`, so the engine can hand them the published Q8_0 bytes at engine-chosen
 * offsets without restating one multiply. If that had required changing their arithmetic, the
 * correct outcome would have been to report the gap, not to fork the lane.
 *
 * ONE SOURCE, TWO COMPILERS, ZERO TOLERANCE. `m2_se_host.cpp` (g++) and `m2_se_arm.hip` (hipcc,
 * `--offload-arch=gfx1151`) both include this header and both run the same item order; the compiled
 * pair is compared byte-for-byte by `engine/block_device.py` and prints
 * `PAIR … ndiff_bytes=0`. That claim is only meaningful while both compilers are forbidden from
 * reassociating or contracting the declared dot, so `build.sh` passes `-ffp-contract=off
 * -fno-fast-math` to both arms (`prototype/shared_expert/build.sh` does the same for the same
 * reason, and `shared_expert_kernels.h` §FMA states the dependency).
 *
 * WHAT THE DEVICE ARM IS ALLOWED TO TOUCH. Nothing in this file calls the HIP runtime: the runtime
 * lives in `m2_se_arm.hip` and in `engine/stage_run.py`'s `HipStaging`, and every act of either is
 * counted. No timer appears here or in any file that includes it.
 *
 * THE ADDRESS ARRANGEMENT, STATED PLAINLY (it is an address fact, not a conversion). Policy A
 * stores `blk.0.ffn_gate_shexp.weight` and `blk.0.ffn_up_shexp.weight` as two separate Q8_0 tensors
 * of `SE_I` rows each (`engine/loader.py::handle` — both 1114112 B, both `weights.shared`). The
 * prototype's stage-1 body reads ONE fused weight with gate row j at fused row j and up row j at
 * fused row `SE_I + j` (`se_layout_rows` §GATE_FIRST, §11.4 / AGENTS.md prediction 3). The engine
 * therefore places the two staged spans adjacently, gate first, and hands the pair to the prototype
 * as one `SE_W_FUSED` weight. No byte is re-ordered, re-packed or re-quantized: the two halves of
 * the fused span are byte-identical to the two resident spans, and
 * `engine/block_device.py` proves that by digest on every run rather than asserting it.
 */
#ifndef M2_SE_KERNELS_H
#define M2_SE_KERNELS_H

#include "shared_expert_kernels.h"      /* the gated lane's arithmetic, unedited */

/* Counter merge, used ONLY in `__global__` bodies: `atomicAdd` is a __device__ function and may not
 * be called from a `__host__ __device__` one, which is also why `prototype/moe_gather` puts its
 * MG_ACC at the same place. The item functions below therefore accumulate into caller-owned ints and
 * whoever runs them merges: a plain `+=` on the host twin, an atomic in the kernel. Every item adds a
 * count derived from geometry, so the merged census is reconcilable against a number the driver
 * computes from `SE_H`/`SE_I` without knowing the thread map. */
#if defined(__HIPCC__) || defined(__CUDACC__)
#define M2_ACC(dst, v) atomicAdd((dst), (v))
#else
#define M2_ACC(dst, v) (*(dst) += (v))
#endif

/* One decode step's worth of addresses, all relative to the slab base the caller passes in. The
 * struct is a kernel argument, so it stays POD and carries no pointers of its own. */
struct M2SePlan {
    int32_t off_fused;                  /* gate|up: SE_TWO_I rows x SE_GATE_BLOCKS x SE_Q8_BYTES  */
    int32_t off_down;                   /* SE_H rows x SE_DOWN_BLOCKS x SE_Q8_BYTES                */
    int32_t off_sg;                     /* SE_H f32 (policy A keeps the shared gate exact)        */
    int32_t off_x;                      /* SE_H bf16 bit images (the block's mixer input)         */
    int32_t mode;                       /* SE_MODE_* — the honest run is SE_MODE_CORRECT          */
    int32_t sgate_form;                 /* SE_SGATE_SCALAR for this model                         */
    int32_t cast_form;                  /* SE_CAST_BEFORE — §11.6 / prediction 4's position       */
    int32_t skip_stage3;                /* control only: leaves stage 3 unwritten so the poison
                                         * census is witnessed non-zero, never a shipping mode    */
    int32_t alloc_oversize;             /* control only, device arm: asks for an allocation no pool
                                         * can satisfy, so "a failed launch exits red" has a leg  */
};

/* The census the driver reconciles. Every field is a count the kernel itself accumulates, so a
 * device run that launched a short grid shows up as a census disagreement instead of a hole. */
struct M2SeCounters {
    uint32_t reads_fused;
    uint32_t reads_down;
    uint32_t reads_sg;
    uint32_t items1;
    uint32_t items2;
    uint32_t items3;
};

/* The record one case writes: the prototype lane's own field layout, so `SE_OFF_*` is the single
 * spelling of the field map and this file never re-types an offset. The lane exposes no field table
 * of its own (only `moe_router_kernels.h` has `mr_field_off`), so this is the engine's one reading
 * of the lane's `SE_OFF_*`/`SE_N_*` pairs: names, offsets and lengths from one place. */
#define M2_SE_FLOATS SE_CASE_FLOATS
static const int M2_SE_FIELD_OFF[SE_FIELD_COUNT] = {SE_OFF_GATE, SE_OFF_UP, SE_OFF_ACT, SE_OFF_RAW,
                                                   SE_OFF_OUT, SE_OFF_PROB};
static const int M2_SE_FIELD_LEN[SE_FIELD_COUNT] = {SE_N_GATE, SE_N_UP, SE_N_ACT, SE_N_RAW,
                                                   SE_N_OUT, SE_N_PROB};
static const char *const M2_SE_FIELD_NAME[SE_FIELD_COUNT] = {"gate", "up", "act", "raw", "out",
                                                            "prob"};

SE_DEV static inline void m2_se_poison(float *out) {
    for (int i = 0; i < M2_SE_FLOATS; ++i) out[i] = se_float_of(SE_NAN_BITS);
}

SE_DEV static inline int m2_se_is_nan(const float *v) {
    return se_is_nan(*v);
}

/* Fill the prototype's plan records from the engine's addresses. `w`, `a`, `c` are caller-owned
 * storage (registers or local memory on the device arm) so nothing here allocates. */
SE_DEV static inline void m2_se_fill(const struct M2SePlan *p, const uint8_t *slab,
                                     SePlan *sp, SeWeight *w, SeAct *a, SeCase *c) {
    w[0].id = 0; w[0].role = SE_W_FUSED; w[0].n_rows = SE_TWO_I; w[0].n_cols = SE_H;
    w[0].kind = 0; w[0].byte_off = (int64_t)p->off_fused;
    w[0].n_bytes = (int64_t)SE_TWO_I * SE_GATE_BLOCKS * SE_Q8_BYTES;
    w[1].id = 1; w[1].role = SE_W_DOWN; w[1].n_rows = SE_H; w[1].n_cols = SE_I;
    w[1].kind = 0; w[1].byte_off = (int64_t)p->off_down;
    w[1].n_bytes = (int64_t)SE_H * SE_DOWN_BLOCKS * SE_Q8_BYTES;
    w[2].id = 2; w[2].role = SE_W_SG; w[2].n_rows = 1; w[2].n_cols = SE_H;
    w[2].kind = 1; w[2].byte_off = (int64_t)p->off_sg;
    w[2].n_bytes = (int64_t)SE_H * 4;
    a[0].id = 0; a[0].n_elems = SE_H; a[0].byte_off = (int64_t)p->off_x;
    c->index = 0; c->w_fused = 0; c->w_down = 1; c->w_sg = 2; c->a = 0;
    c->mode = p->mode; c->sgate_form = p->sgate_form; c->cast_form = p->cast_form;
    c->dst_off = 0;
    sp->w = w; sp->n_w = 3;
    sp->a = a; sp->n_a = 1;
    sp->c = c; sp->n_c = 1;
    sp->slab = slab; sp->slab_bytes = 0;
    sp->expect = 0; sp->n_expect = 0;
}

/* ---- the three items of one decode step -------------------------------------------------------
 * §11.6's order, and each item is ONE output element of one field, so the thread map cannot show up
 * in the answer: no item reduces across another item. Stage 2 reads stage 1's `act` and stage 3
 * reads stage 2's `raw`, which is why they are three launches and not one — the host twin runs the
 * same three loops in the same order for the same reason.                                            */

/* Per-item tally: plain ints, filled by the item bodies, merged by whoever ran them. */
struct M2SeTally {
    int fused;
    int down;
    int sg;
    int items;
};

/* Item 1: fused row j -> gate, up, act. `j` runs 0..SE_I-1. The scalar gate probability is one
 * number for the whole token, so exactly one item (j == 0) also runs stage 3a. Counters are the
 * caller's, so this body never needs an atomic. */
SE_DEV static inline void m2_se_item1(const SePlan *sp, const SeCase *c, float *out, int j,
                                      struct M2SeTally *t) {
    int rf = 0, rs = 0;
    se_stage_gateup(sp, c, out, j, &rf);
    t->fused += rf;
    t->items += 1;
    if (j == 0) {
        const float prob = se_stage_prob(sp, c, &rs);
        out[c->dst_off + SE_OFF_PROB] = prob;
        t->sg += rs;
    }
}

/* Item 2: output row h of `down`, reading stage 1's act out of the record. */
SE_DEV static inline void m2_se_item2(const SePlan *sp, const SeCase *c, float *out, int h,
                                      struct M2SeTally *t) {
    int reads_h = 0;
    se_stage_down(sp, c, out, h, &reads_h);
    t->down += reads_h;
    t->items += 1;
}

/* Item 3: one channel of the gated output, `prob` from stage 1's field. */
SE_DEV static inline void m2_se_item3(const SePlan *sp, const SeCase *c, const float *out,
                                      float prob, int h, struct M2SeTally *t) {
    int reads_h = 0;
    const float v = se_stage_out_channel(sp, c, out, prob, h, &reads_h);
    ((float *)out)[c->dst_off + SE_OFF_OUT + h] = v;
    t->items += 1;
    (void)reads_h;                     /* SGATE_SCALAR reads no weight: nothing to bill */
}

/* The host twin's whole run: the same three loops the device launches, in the same order. */
SE_DEV static inline void m2_se_run(const struct M2SePlan *p, const uint8_t *slab, float *out,
                                    struct M2SeCounters *ctr) {
    SePlan sp;
    SeWeight w[3];
    SeAct a[1];
    SeCase c;
    int j, h;
    m2_se_fill(p, slab, &sp, w, a, &c);
    m2_se_poison(out);
    for (j = 0; j < SE_I; ++j) {
        struct M2SeTally t;
        t.fused = t.down = t.sg = t.items = 0;
        m2_se_item1(&sp, &c, out, j, &t);
        ctr->reads_fused += (uint32_t)t.fused;
        ctr->reads_sg += (uint32_t)t.sg;
        ctr->items1 += (uint32_t)t.items;
    }
    for (h = 0; h < SE_H; ++h) {
        struct M2SeTally t;
        t.fused = t.down = t.sg = t.items = 0;
        m2_se_item2(&sp, &c, out, h, &t);
        ctr->reads_down += (uint32_t)t.down;
        ctr->items2 += (uint32_t)t.items;
    }
    if (!p->skip_stage3) {
        const float prob = out[c.dst_off + SE_OFF_PROB];
        for (h = 0; h < SE_H; ++h) {
            struct M2SeTally t;
            t.fused = t.down = t.sg = t.items = 0;
            m2_se_item3(&sp, &c, out, prob, h, &t);
            ctr->items3 += (uint32_t)t.items;
        }
    }
}

/* What the census must come to, from the geometry in the prototype header. The driver derives the
 * same numbers from `engine/block_forward.py::load_geometry()` and requires agreement, so a header
 * constant that drifts from the checkpoint config is caught here and not in a stage row. */
#define M2_SE_WANT_READS_FUSED (2 * SE_I * SE_H)
#define M2_SE_WANT_READS_DOWN (SE_H * SE_I)
#define M2_SE_WANT_READS_SG (SE_H)
#define M2_SE_ITEMS1 SE_I
#define M2_SE_ITEMS2 SE_H
#define M2_SE_ITEMS3 SE_H

#endif /* M2_SE_KERNELS_H */
