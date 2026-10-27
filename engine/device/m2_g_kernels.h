/* engine/device/m2_g_kernels.h — the routed-expert gather: K selected experts, their SwiGLU bodies,
 * and the combine that folds them into one row. ONE translation unit, TWO width paths.
 *
 * WHY TWO PATHS THROUGH ONE KERNEL. The rows this slice can license live on the reference path
 * (`engine/block_gate.py::MOE_BANDS`: `expert_raw` under the moe_gather floor probe, `routed_sum` under
 * `specification/10` §10.6's exactness row). The shape the engine ships is the quantized one, which no
 * band measures — D-01 authorized the widths, and the stage-output floor probe that
 * `quantization/README.md` §“How a quantized path gets gated” item 1 requires has never been run. An arm
 * built only for the quantized path would be plumbing with no licensed evidence; an arm built only for
 * the reference path would gate a shape the engine does not ship. Both therefore run through this one
 * file, and what differs between them is exactly three things, each declared once, here:
 *
 *   1. which decoder answers — a bf16 matrix image, or the artifact's Q4_K (gate, up) and Q5_K (down)
 *      block spans through the audited prototype decoders;
 *   2. where the roundings fall — on `path=ref_bf16`, gate, up, `silu(gate)` and the gate*up product are
 *      each rounded to bf16 before the next step. Those positions are measured, not assumed:
 *      `engine/work/probe_m2dev_gather2.py` shows that dropping any one of them moves
 *      6 837..10 146 of the pinned row's 16 384 elements, so the pins fix them;
 *   3. which accumulator combines — a bf16 accumulator over products rounded to bf16 on the reference
 *      path (the form `prototype/moe_gather/work/oracle_floor.tsv` records as
 *      `combine_form_vs_pinned_routed_sum = 0`, re-measured on this row by
 *      `engine/work/probe_m2dev_gather3.py`), or an f32 accumulator rounded once on policy A, because a
 *      dequantized weight arrives as f32 and `engine/block_forward.py::moe_routed` keeps it that way.
 *
 * WHO OWNS WHICH ARITHMETIC — read this before trusting a row.
 *   The Q4_K/Q5_K/Q6_K decoders (`mg_deq_dbl`, `mg_deq_f32`, `mg_scale_min`), the bf16 primitives
 *   (`mg_bf16_of`, `mg_bf16_to_f32`, `mg_bf16_add`), `mg_silu`, the ascending-expert-id walk
 *   (`mg_combine_order`) and the mistake vocabulary (`MG_MODE_*`) are called VERBATIM from
 *   `prototype/moe_gather/moe_gather_kernels.h` — the lane whose own gate passed 911 checks with 0
 *   findings on this op (`prototype/moe_gather/evidence/moe-gather-device-2026-10-26.log`, and
 *   `prototype/moe_gather/results.md` §4 for prediction 3). They are not re-derived here.
 *   Engine-new, i.e. the part no prototype gate vouches for: the span addressing in `m2g_unit_elt` (the
 *   engine addresses experts as artifact/shard spans, not as the prototype's sampled unit table), the
 *   bf16 width path (`M2G_KIND_BF16`; the prototype has no bf16 kind), the three reference-path
 *   roundings, the path-dependent combine accumulator, and reading `MG_MODE_HALVES_SWAPPED` as "exchange
 *   the two spans" (the prototype exchanges two unit *names*, inert on a fused span). `engine/HANDOFF.md`
 *   PASS 5 states the boundary in prose.
 *
 * NO ATOMICS AND NO BARRIERS IN THIS FILE. Same reason `engine/device/m2_se_kernels.h` states it: the
 * counters merge with `atomicAdd` and phases would need `__syncthreads()`, both `__device__` functions,
 * and every function here is `__host__ __device__` because the host twin and the hipcc arm's host path
 * compile the same source. So each `m2g_item*` accumulates into a caller-owned `struct M2GItem` and the
 * caller folds it — with plain `+=` on a host path, with `atomicAdd` inside a `__global__` body. The
 * device therefore parallelizes over *items* (one thread per output element of one slot), which needs no
 * barrier at all: gate/up, s, raw and routed are separate launches, and a phase's inputs are the previous
 * launch's outputs.
 *
 * WHAT THIS FILE DOES NOT DO: it does not route. The K expert ids and both weight vectors arrive in the
 * slab from the ROUTER arm's own record, so the gather is judged on the selection the compiled router
 * actually produced. It reads no clock and prints no rate (`pitfalls/00`).
 */
#ifndef M2G_KERNELS_H
#define M2G_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "moe_gather_kernels.h"          /* via -I prototype/moe_gather */

#define M2G_MAX_UNITS 32

/* ---- the width paths ---------------------------------------------------------------------------- *
 * `path` is a plan word, never inferred from a byte count. The element size the plan declares for the
 * activation has to agree with it — 2 B on ref_bf16, 4 B on policy_a — and `m2_g_io.h` refuses a plan
 * where they disagree, so "which shape is this arm running?" is a checked fact, not a convention. */
#define M2G_PATH_REF_BF16 0
#define M2G_PATH_POLICY_A 1
#define M2G_PATH_NAME(p) ((p) == M2G_PATH_POLICY_A ? "policy_a" : "ref_bf16")

/* A kind id outside MG_KIND_*, so `mg_kind_ok()` still refuses it and no prototype code path can be
 * reached with it by accident. Its element addressing is engine code, inside `m2g_unit_elt`. */
#define M2G_KIND_BF16 8

/* The engine's two extra words: force the accumulator the OTHER path uses, so "is the accumulator
 * choice load-bearing?" is a leg the arm can execute rather than a comment. Numbered above the
 * prototype's enum so a renumbered prototype enum cannot collide with it. */
#define M2G_MODE_ACC_BF16 100
#define M2G_MODE_ACC_F32  101
/* The rounding injection. `mg_silu` is one `expf`, and a transcendental is the one part of this body that
 * two libms are entitled to answer differently: `DECISIONS.md` D-16 records exactly that for this lane
 * (`expf` glibc on the host, ROCm libdevice on the device, declared for `s`/`raw` and counted, never
 * banded), and `pitfalls/00-measurement.md` §"A divergence count that changes between identical runs is a
 * race, not rounding" says the test is whether the differing bit images are ADJACENT. `silu_exp_ulp1`
 * makes that difference explicit and measurable on the CPU: one ULP up on the exp() result, with nothing
 * else changed, so the byte count and the field locations a libdevice difference implies can be compared
 * with what a device window actually observes. */
#define M2G_MODE_EXP_ULP1 102

static inline int m2g_mode_known(int m) {
    if (m >= MG_MODE_CORRECT && m < MG_MODE_COUNT) return 1;
    return m == M2G_MODE_ACC_BF16 || m == M2G_MODE_ACC_F32 || m == M2G_MODE_EXP_ULP1;
}

/* Which record fields a mode can move, in the prototype's reach-table sense. The driver uses it to
 * refuse a control that asks for a leg the mode cannot produce — an inert mode reported as a fired
 * control is the exact failure this lane exists to refuse (window e2's tie control). */
static inline const char *m2g_mode_field(int m) {
    if (m == MG_MODE_POISON_NAN) return "gate,up,s,raw,routed";
    /* `s` is the first field the exp() result can reach, `raw` is the dot over it, `routed` the combine
     * over that; gate and up are computed before any transcendental, so a difference in them CANNOT be
     * this cause and the reach table is what makes that a refusal instead of an opinion. */
    if (m == M2G_MODE_EXP_ULP1) return "s,raw,routed";
    if (m == M2G_MODE_ACC_BF16 || m == M2G_MODE_ACC_F32) return "routed";
    switch (m) {
        case MG_MODE_W_BY_EXPERT_ID:
        case MG_MODE_RANK_ACCUM:
        case MG_MODE_CAST_AFTER:
        case MG_MODE_F32_COMBINE:   return "routed";
        case MG_MODE_WIDTH_MISREAD:
        case MG_MODE_Q6K_D_BYTE_0:  return "raw";
        /* `byte_shift_1` shifts the span the operands are read from, so it reaches `routed` through `raw`
         * like nothing else in this branch does. Measured on the host path of the hipcc arm, both widths:
         * policy_a gate 12,444 up 12,475 s 11,817 raw 65,331 routed 8,145 B; ref_bf16 7,642/7,637/6,302/
         * 29,641/3,702 B. The default claim omitted `routed`, and an under-claimed reach is how a control
         * starts believing a mode moved less than it did — `gather-divergence-causes-priced`'s GREACH leg
         * holds every cause it runs to its own claim now, in both directions. */
        case MG_MODE_BYTE_SHIFT_1:   return "gate,up,s,raw,routed";
        default:                     return "gate,up,s,raw";
    }
}

/* ---- the record ---------------------------------------------------------------------------------- *
 * fp32 words throughout; a bf16-carrying field is stored widened, the repo's convention, so the
 * comparator compares 4-byte patterns and never guesses a dtype off a file. gate/up/s are kept rather
 * than discarded: prediction 3's swap has to be visible on the halves themselves and not only in the
 * sum, and an unwritten slot has to be attributable to the stage that skipped it. */
#define M2G_OFF_GATE    0
#define M2G_OFF_UP     (M2G_OFF_GATE + MG_K * MG_I)
#define M2G_OFF_S      (M2G_OFF_UP + MG_K * MG_I)
#define M2G_OFF_RAW    (M2G_OFF_S + MG_K * MG_I)
#define M2G_OFF_ROUTED (M2G_OFF_RAW + MG_K * MG_H)
#define M2G_FLOATS     (M2G_OFF_ROUTED + MG_H)

#define M2G_FIELD_COUNT 5
static const char *const M2G_FIELD_NAME[M2G_FIELD_COUNT] = { "gate", "up", "s", "raw", "routed" };
static const int M2G_FIELD_OFF[M2G_FIELD_COUNT] = {
    M2G_OFF_GATE, M2G_OFF_UP, M2G_OFF_S, M2G_OFF_RAW, M2G_OFF_ROUTED
};
static const int M2G_FIELD_LEN[M2G_FIELD_COUNT] = {
    MG_K * MG_I, MG_K * MG_I, MG_K * MG_I, MG_K * MG_H, MG_H
};

/* ---- the plan ------------------------------------------------------------------------------------ */
struct M2GUnit {
    int role;            /* 0 gate, 1 up, 2 down — the prototype's MG_ROLE_* values */
    int slot;            /* the top-k slot whose expert these bytes are */
    int expert;          /* the expert_index they belong to; the census is keyed on this */
    long off;            /* byte offset of this unit's matrix inside the slab */
    long bytes;          /* the span's own byte length, from the address that produced it */
    long span_off;       /* the span this unit lives in (== off unless the gate|up pair is fused) */
    long span_bytes;
    int rows, cols;      /* the matrix it encodes: gate/up [I,H], down [H,I] */
    int kind;            /* MG_KIND_Q4K / MG_KIND_Q5K / MG_KIND_Q6K / M2G_KIND_BF16 */
    int elt;             /* 1 for a quantized block span, 2 for a bf16 matrix image */
    int blocks;          /* 256-element blocks in the unit (0 for a bf16 image) */
    int fused;           /* 1 = gate and up share one [2I,H] span: the reference layout */
    int whole;           /* 1 = staged as a whole tensor; the census has to refuse it */
};

struct M2GPlan {
    int path;
    int slots;
    long off_x; int elt_x;               /* activation: 2 B/elt ref_bf16, 4 B/elt policy_a */
    long off_ids, off_wcast, off_wpre;   /* the router arm's own selection and weights */
    int mode;
    int fault_skip_raw, fault_skip_routed, fault_whole_note;
    long corrupt_off;                    /* >= 0: flip one slab byte before any launch */
    long read_off; int read_len;         /* the device unit read-back leg */
    int n_units;
    struct M2GUnit units[M2G_MAX_UNITS];
    long slab_bytes;
};

/* Which unit a slot's role resolves to. HALVES_SWAPPED exchanges the gate and up UNITS, which is the
 * swap in the form each layout actually has: on policy A they are two artifact tensors, on the reference
 * path two views of one fused span. EXPERT_ROTATE walks the slot ring inside this row's own K-set, so a
 * neighbour's bytes answer under this expert's id. */
MG_DEV static inline const struct M2GUnit *m2g_resolve(const struct M2GPlan *p, int slot, int role,
                                                       int mode) {
    int i, want_slot = slot, want_role = role;
    if (mode == MG_MODE_EXPERT_ROTATE) want_slot = (slot + 1) % p->slots;
    if (mode == MG_MODE_HALVES_SWAPPED && role != 2) want_role = (role == 0) ? 1 : 0;
    for (i = 0; i < p->n_units; ++i)
        if (p->units[i].slot == want_slot && p->units[i].role == want_role) return &p->units[i];
    return 0;
}

/* ---- one element read ---------------------------------------------------------------------------- *
 * The value of element (row, col) of a unit's matrix, or 1 when the plan cannot answer for it — the
 * same contract as `mg_unit_elt` (0 = defined, non-zero = refuse), so a refused read becomes NaN in the
 * record and stays loud. Quantized kinds delegate to the audited prototype decoder on the same block
 * bytes; the bf16 kind is engine addressing over a matrix image. All addressing is byte-offset
 * arithmetic, never a typed-pointer offset (`pitfalls/40` §two-self-consistent-layouts).
 *
 * `v32` is the counted second route, exactly as the prototype counts it: Q4_K and Q5_K are exact in f32,
 * so the two routes must agree bit for bit and a divergence is a finding to report, never a tolerance to
 * absorb. The wrong-spelling words are the prototype's own enum where they exist. */
MG_DEV static inline int m2g_unit_elt(const char *slab, const struct M2GUnit *u, int row, int col,
                                      int mode, float *v, float *v32) {
    long base = u->off, extent = u->bytes, flat;
    int blocks = u->blocks;
    if (row < 0 || row >= u->rows || col < 0 || col >= u->cols) return 1;
    flat = (long)row * u->cols + col;
    if (mode == MG_MODE_INPUT_CHUNK && u->fused) {
        /* the [2I,H] pair chunked along the INPUT dim instead of the output dim: row i, column c of a
         * half reads element (2i + which)*H + c of the SAME span. The prototype's own spelling of that
         * mistake, expressible only where the pair really is one span, and addressed from the span's
         * start rather than the half's, or it would run off the end instead of moving. */
        base = u->span_off;
        extent = u->span_bytes;
        flat = ((long)row * 2 + (u->role == 1 ? 1 : 0)) * u->cols + col;
        if (u->kind != M2G_KIND_BF16) blocks = (int)(extent / MG_QK_K);
    } else if (mode == MG_MODE_ROW_OFF_1) {
        flat = (long)((row + 1) % u->rows) * u->cols + col;
    } else if (mode == MG_MODE_BYTE_SHIFT_1) {
        flat = (flat + 1) % ((long)u->rows * u->cols);   /* one element late, wraps inside the unit */
    }
    if (u->kind == M2G_KIND_BF16) {
        const uint8_t *q;
        if (flat < 0 || (flat + 1) * 2 > extent) return 1;
        q = (const uint8_t *)slab + base + 2 * flat;
        *v = mg_bf16_to_f32((uint16_t)((uint32_t)q[0] | ((uint32_t)q[1] << 8)));
        if (v32) *v32 = *v;                             /* a bf16 image has no second route */
        return 0;
    }
    if (!mg_kind_ok(u->kind)) return 1;
    {
        const int bb = mg_block_bytes(u->kind);
        int kind = u->kind;
        long blk = flat / MG_QK_K, off;
        const int e = (int)(flat % MG_QK_K);
        const uint8_t *bp;
        if (mode == MG_MODE_WIDTH_MISREAD && u->role == 2) {
            if (kind == MG_KIND_Q5K) kind = MG_KIND_Q6K;
            else if (kind == MG_KIND_Q6K) kind = MG_KIND_Q5K;
            else return 1;                              /* Q4_K has no defined width swap */
        }
        if (blk < 0 || blk >= blocks) return 1;
        if (mode == MG_MODE_BLOCK_OFF_1) blk = (blk + 1) % blocks;
        off = base + blk * bb;
        if (mode == MG_MODE_BYTE_SHIFT_1) off += 1;
        if (off < 0 || off + bb > base + extent) return 1;
        bp = (const uint8_t *)slab + off;
        *v = mg_deq_dbl(bp, kind, e, mode);
        if (v32) *v32 = mg_deq_f32(bp, kind, e, mode);
    }
    return 0;
}

/* ---- the census the scheduler decision gets made on ---------------------------------------------- *
 * AGENTS.md prediction 2 prices the gather as 8 dependent ~6 MiB reads per token per layer, and the
 * scheduler choice it has to make (grouped vs per-slot, what stays resident) needs the real numbers:
 * how many spans, of what size, per which expert_index, and whether any of them was a whole tensor.
 * `units_by_expert` and `bytes_by_expert` are indexed by the expert_index the plan named, so the answer
 * is per expert rather than in aggregate, and `whole_reads` is the counter a whole-tensor read moves —
 * the failure that check exists to catch is reading the whole stacked tensor per token instead of the
 * slice the resident set is priced for (`engine/block_gate.py::gather_census` is the torch-side twin).
 * `route_n`/`route_bad` is the prototype's counted-not-banded width census.
 *
 * Two structures, because of the no-atomics rule above: `M2GItem` is what one item call tallies into
 * (caller-owned, one per thread, one per UNIT of the plan), `M2GCounters` is the run's total.
 * `m2g_item_fold` is plain `+=` and is what a host path uses; `engine/device/m2_g_arm.hip` has the
 * `atomicAdd` version for a `__global__` body. `distinct_experts` is derived at report time from
 * `units_by_expert`, never counted, so it cannot be raced by two threads that both saw a zero.
 *
 * ONE NOTE PER ITEM, AND THE COUNT IS NOT ALLOWED TO BE QUIET ABOUT IT. An earlier version let one item
 * accumulate every unit of the row into a fixed-size note array and `return`ed silently once it was full:
 * the host body tallied all 24 units of a row into 4 slots, so the census printed `unit_reads=4` and
 * `total=8 MiB` for a plan that reads 50 MiB, and the first thing that number was used to claim —
 * `whole_reads=0`, "the gather reads slices" — was read off a counter that had silently dropped 20 of the
 * 24 units. A dropped note is worse than a zero, because a zero is at least visibly zero. The device never
 * had this bug (a thread owns at most the unit whose row 0 it computed), so the fix is to make the host
 * body match the device's note shape — one item per unit — and to keep a conflict counter for a caller that
 * breaks that shape, reported in the census rather than dropped. */
struct M2GItem {
    uint32_t elts, route_n, route_bad;
    int note_expert;
    uint32_t note_bytes;
    int note_whole;
    int n_note;                  /* 0 or 1: the geometry of a unit, not a capacity */
    uint32_t note_conflict;      /* a second, different unit noted into this item: caller-shape error */
};

struct M2GCounters {
    uint32_t units, elts, whole_reads, slots;
    uint32_t route_n, route_bad, note_conflict;
    uint32_t units_by_expert[MG_E];
    uint32_t bytes_by_expert[MG_E];      /* bytes, not blocks: a scheduler prices bytes */
};

MG_DEV static inline void m2g_item_init(struct M2GItem *it) {
    if (!it) return;
    memset(it, 0, sizeof(*it));
    it->note_expert = -1;
}

/* Note the unit this item is reading. Called by the item that owns the unit's first row, so the count is
 * units gathered, not elements read, on either path. `whole` is the plan's own `whole` flag ORed with the
 * declared whole-note fault: the parser will not accept a whole-tensor unit at all (`m2_g_io.h`
 * ::whole-tensor-unit-refused), so the counter's non-zero leg has to come from a fault the plan declares
 * by name, and it is counted per unit rather than once per run so it cannot be a constant someone forgot
 * to update. */
MG_DEV static inline void m2g_item_note(struct M2GItem *it, const struct M2GUnit *u, int whole) {
    if (!it || !u) return;
    if (u->expert < 0 || u->expert >= MG_E) return;      /* counted by `bad_index`, not silently */
    if (it->n_note) {
        if (it->note_expert == u->expert && it->note_bytes == (uint32_t)u->bytes
            && it->note_whole == whole) return;         /* the same unit, revisited */
        it->note_conflict += 1u;                        /* loud in the census, never dropped */
        return;
    }
    it->note_expert = u->expert;
    it->note_bytes = (uint32_t)u->bytes;
    it->note_whole = whole;
    it->n_note = 1;
}

/* Fold one unit's tallies into the run's counters. */
static inline void m2g_item_fold(struct M2GCounters *ct, const struct M2GItem *it, uint32_t slots) {
    if (!ct || !it) return;
    ct->elts += it->elts;
    ct->route_n += it->route_n;
    ct->route_bad += it->route_bad;
    ct->note_conflict += it->note_conflict;
    ct->slots += slots;
    if (it->n_note) {
        const int e = it->note_expert;
        ct->units += 1u;
        ct->units_by_expert[e] += 1u;
        ct->bytes_by_expert[e] += it->note_bytes;
        if (it->note_whole) ct->whole_reads += 1u;
    }
}

MG_DEV static inline void m2g_poison(float *out) {
    int i;
    const float nan = mg_float_from_bits(MG_NAN_BITS);
    for (i = 0; i < M2G_FLOATS; ++i) out[i] = nan;
}

MG_DEV static inline int m2g_field_nan(const float *out, int field) {
    int i, n = 0;
    for (i = 0; i < M2G_FIELD_LEN[field]; ++i)
        if ((mg_bits_of(out[M2G_FIELD_OFF[field] + i]) & 0x7FFFFFFFu) == MG_NAN_BITS) n++;
    return n;
}

/* The activation element, read the way the path stores it: a bf16 image on the reference path (the same
 * bytes the pinned dump was made from) and f32-widened values on policy A (where the shipping program
 * holds the dequantized f32, per `engine/block_forward.py::Weights.basis`). */
MG_DEV static inline float m2g_x(const struct M2GPlan *p, const char *slab, int j) {
    if (p->elt_x == 2) {
        const uint8_t *q = (const uint8_t *)slab + p->off_x + 2 * (long)j;
        return mg_bf16_to_f32((uint16_t)((uint32_t)q[0] | ((uint32_t)q[1] << 8)));
    }
    return ((const float *)(slab + p->off_x))[j];
}

/* ---- the four items ------------------------------------------------------------------------------- *
 * One output element each, one thread each, no cross-thread dependence: `item1` is one row of gate or of
 * up (a dot over H), `item2` is one element of `s`, `item3` is one element of `raw` (a dot over I),
 * `item4` is one column of the combined row. Each dot walks its index ascending with one accumulator —
 * the shape `mg_expert_case` uses — because a wrong pairing has to show up rather than average away.
 * Nothing here is timed, and the shape is a correctness shape, not a throughput claim.
 *
 * `row`/`t` are element indices inside the slot; `which` (item1 only) says gate or up. */
MG_DEV static inline void m2g_item1(const struct M2GPlan *p, const char *slab, float *rec, int slot,
                                    int which, int row, struct M2GItem *it, int *bad) {
    const int mode = p->mode;
    const int ref = (p->path == M2G_PATH_REF_BF16);
    const struct M2GUnit *un = m2g_resolve(p, slot, which, mode);
    float *dst = rec + (which ? M2G_OFF_UP : M2G_OFF_GATE) + (long)slot * MG_I + row;
    float acc = 0.0f;
    int j;
    if (!un) { if (bad) *bad += 1; *dst = mg_float_from_bits(MG_NAN_BITS); return; }
    if (mode == MG_MODE_POISON_NAN) { *dst = mg_float_from_bits(MG_NAN_BITS); return; }
    /* One census note per unit, by the thread that owns its first row — on the fused reference span the
     * up unit is a view of the gate's span and is still counted, which is what makes "24 units over 16
     * spans" the printed fact rather than an assumption. */
    if (it && row == 0) m2g_item_note(it, un, un->whole || p->fault_whole_note);
    for (j = 0; j < un->cols; ++j) {
        float w, w32 = 0.0f;
        if (m2g_unit_elt(slab, un, row, j, mode, &w, ref ? 0 : &w32)) {
            acc = mg_float_from_bits(MG_NAN_BITS); break;
        }
        if (it) {
            it->elts += 1u;
            if (!ref) {
                it->route_n += 1u;
                if (mode == MG_MODE_CORRECT && !mg_word_eq(mg_bits_of(w), mg_bits_of(w32)))
                    it->route_bad += 1u;
            }
        }
        acc += m2g_x(p, slab, j) * w;
    }
    /* rounding positions 1 and 2: the GEMV result lands in the model dtype before anything touches it */
    *dst = ref ? mg_bf16_to_f32(mg_bf16_of(acc)) : acc;
}

/* The exp() injection, spelled as the arithmetic it models: same expf, one ULP up on its result. Every
 * other operation is `mg_silu` verbatim, so a control that runs this mode is measuring the size of a
 * last-bit transcendental difference and nothing else. `nextafterf` is the standard one on both sides
 * (C99 on the host, the libdevice route on the device), so "one ULP" means the same thing in both. */
MG_DEV static inline float m2g_silu_ulp1(float z) {
    return z / (1.0f + nextafterf(expf(-z), INFINITY));
}

MG_DEV static inline float m2g_silu_mode(float z, int mode) {
    return mode == M2G_MODE_EXP_ULP1 ? m2g_silu_ulp1(z) : mg_silu(z);
}

/* s = silu(gate) * up, in the reference path's two further roundings (the silu, then the product).
 * Both positions are pinned by the dump; `gather-rounding-forms-redden-s` is the leg that drops one at a
 * time and watches the row redden. */
MG_DEV static inline void m2g_item2(const struct M2GPlan *p, const float *rec, float *out, int slot,
                                    int i) {
    const int ref = (p->path == M2G_PATH_REF_BF16);
    const float g = rec[M2G_OFF_GATE + (long)slot * MG_I + i];
    const float u = rec[M2G_OFF_UP + (long)slot * MG_I + i];
    const float sl = ref ? mg_bf16_to_f32(mg_bf16_of(m2g_silu_mode(g, p->mode)))
                         : m2g_silu_mode(g, p->mode);
    const float prod = sl * u;
    out[M2G_OFF_S + (long)slot * MG_I + i] = ref ? mg_bf16_to_f32(mg_bf16_of(prod)) : prod;
}

MG_DEV static inline void m2g_item3(const struct M2GPlan *p, const char *slab, const float *rec,
                                    float *out, int slot, int t, struct M2GItem *it, int *bad) {
    const int mode = p->mode;
    const int ref = (p->path == M2G_PATH_REF_BF16);
    const struct M2GUnit *un = m2g_resolve(p, slot, 2, mode);
    const float *s = rec + M2G_OFF_S + (long)slot * MG_I;
    float *dst = out + M2G_OFF_RAW + (long)slot * MG_H + t;
    float acc = 0.0f;
    int j;
    if (!un) { if (bad) *bad += 1; *dst = mg_float_from_bits(MG_NAN_BITS); return; }
    if (mode == MG_MODE_POISON_NAN) { *dst = mg_float_from_bits(MG_NAN_BITS); return; }
    if (it && t == 0) m2g_item_note(it, un, un->whole || p->fault_whole_note);
    for (j = 0; j < un->cols; ++j) {
        float w, w32 = 0.0f;
        if (m2g_unit_elt(slab, un, t, j, mode, &w, ref ? 0 : &w32)) {
            acc = mg_float_from_bits(MG_NAN_BITS); break;
        }
        if (it) {
            it->elts += 1u;
            if (!ref) {
                it->route_n += 1u;
                if (mode == MG_MODE_CORRECT && !mg_word_eq(mg_bits_of(w), mg_bits_of(w32)))
                    it->route_bad += 1u;
            }
        }
        acc += s[j] * w;
    }
    *dst = ref ? mg_bf16_to_f32(mg_bf16_of(acc)) : acc;
}

/* The combine: `mg_combine_order` picks the walk (ascending expert id, with the prototype's rank-order
 * injection available); the accumulator is the path's, with `M2G_MODE_ACC_*` forcing the other one so the
 * choice is a leg and not a comment. Cast-before is the reference (R5); `MG_MODE_CAST_AFTER` keeps the
 * fp32 weight the reference threw away — prediction 4's different number, this time on the PRODUCT, the
 * half pass 3 §8.4 named as unreachable until a `routed_sum` row existed. */
MG_DEV static inline void m2g_item4(const struct M2GPlan *p, const char *slab, const float *rec,
                                    float *out, int k) {
    const int mode = p->mode;
    const int *ids = (const int *)(slab + p->off_ids);
    const float *wcast = (const float *)(slab + p->off_wcast);
    const float *wpre = (const float *)(slab + p->off_wpre);
    const float *rawbase = rec + M2G_OFF_RAW;
    int order[MG_K], i;
    int bf16_acc = (p->path == M2G_PATH_REF_BF16);
    if (mode == M2G_MODE_ACC_F32 || mode == MG_MODE_F32_COMBINE) bf16_acc = 0;
    if (mode == M2G_MODE_ACC_BF16) bf16_acc = 1;
    mg_combine_order(ids, mode, order);
    if (bf16_acc) {
        uint16_t acc = 0u;                          /* +0.0 in the model dtype */
        for (i = 0; i < p->slots; ++i) {
            const int slot = order[i];
            float w = (mode == MG_MODE_CAST_AFTER) ? wpre[slot] : wcast[slot];
            if (mode == MG_MODE_W_BY_EXPERT_ID) w = wcast[ids[slot] % MG_K];
            acc = mg_bf16_add(acc, mg_bf16_of(rawbase[(long)slot * MG_H + k] * w));
        }
        out[M2G_OFF_ROUTED + k] = mg_bf16_to_f32(acc);
    } else {
        float acc = 0.0f;                           /* policy A: f32 accumulation, one rounding */
        for (i = 0; i < p->slots; ++i) {
            const int slot = order[i];
            float w = (mode == MG_MODE_CAST_AFTER) ? wpre[slot] : wcast[slot];
            if (mode == MG_MODE_W_BY_EXPERT_ID) w = wcast[ids[slot] % MG_K];
            acc += rawbase[(long)slot * MG_H + k] * w;
        }
        out[M2G_OFF_ROUTED + k] = acc;
    }
}

/* The fault that makes a counter's non-zero leg witnessable: a poisoned field must read NaN, which the
 * driver's own census sees, instead of the counter being asserted from a branch no run takes. */
MG_DEV static inline void m2g_poison_field(float *out, int field) {
    int i;
    const float nan = mg_float_from_bits(MG_NAN_BITS);
    for (i = 0; i < M2G_FIELD_LEN[field]; ++i) out[M2G_FIELD_OFF[field] + i] = nan;
}

/* One unit's rows, run and tallied by an item of its own.
 *
 * The census note belongs to the unit, not to the row. These two helpers exist because the host body used to
 * run a whole row with one item and lose every note past that array's capacity (see the census block above):
 * giving each unit its own item makes the host's note shape the device's note shape — a thread owns at most
 * the unit whose row 0 it computed — so neither path can tally two units into one item, and the helpers run
 * a unit's whole row before returning, which is what `item2` below depends on. */
static inline void m2g_unit_gateup(const struct M2GPlan *p, const char *slab, float *out,
                                   struct M2GCounters *ct, int slot, int which, int *bad) {
    struct M2GItem it;
    int t;
    m2g_item_init(&it);
    for (t = 0; t < MG_I; ++t) m2g_item1(p, slab, out, slot, which, t, &it, bad);
    if (ct) m2g_item_fold(ct, &it, 0);
}

static inline void m2g_unit_down(const struct M2GPlan *p, const char *slab, const float *rec, float *out,
                                 struct M2GCounters *ct, int slot, int *bad) {
    struct M2GItem it;
    int t;
    m2g_item_init(&it);
    for (t = 0; t < MG_H; ++t) m2g_item3(p, slab, rec, out, slot, t, &it, bad);
    if (ct) m2g_item_fold(ct, &it, 0);
}

/* ---- the whole row, serial ---------------------------------------------------------------------- *
 * The host body's order of acts. The device reaches the same four items from four launches instead
 * (`m2_g_arm.hip`), which is why the items, not this loop, are the arithmetic. One launch covering every
 * selected expert of the row is prediction 2's grouped shape: the K gathers of a token are issued
 * together rather than as K dependent visits, and nothing here is timed. */
static inline void m2g_run(const struct M2GPlan *p, const char *slab, float *out,
                           struct M2GCounters *ct, int *bad) {
    int slot, t, k;
    m2g_poison(out);
    if (ct) ct->slots += (uint32_t)p->slots;
    for (slot = 0; slot < p->slots; ++slot) {
        m2g_unit_gateup(p, slab, out, ct, slot, 0, bad);
        m2g_unit_gateup(p, slab, out, ct, slot, 1, bad);
        if (p->mode == MG_MODE_POISON_NAN) continue;    /* the leak leg: nothing below is written */
        for (t = 0; t < MG_I; ++t) m2g_item2(p, out, out, slot, t);
        if (!p->fault_skip_raw) m2g_unit_down(p, slab, out, out, ct, slot, bad);
    }
    if (p->fault_skip_routed) {
        m2g_poison_field(out, 4);
    } else if (p->mode != MG_MODE_POISON_NAN) {
        for (k = 0; k < MG_H; ++k) m2g_item4(p, slab, out, out, k);
    }
}

/* The census expectations, derived from the plan rather than typed by a caller: three matrices per slot,
 * over as many distinct expert ids as the row selected. `want_units` counts matrices (24), `want_bytes`
 * counts the bytes of all of them (a fused gate|up pair is one span but its two halves are both read, so
 * it still costs 2I*H*elt bytes), and `want_spans` counts distinct spans — 16 on the reference layout and
 * 24 on policy A. That the last two differ is prediction 3's fact about the layout, and the census prints
 * both so a reader can see which one a number came from. */
static inline int m2g_want_units(const struct M2GPlan *p) { return p->slots * 3; }
static inline int m2g_want_spans(const struct M2GPlan *p) {
    int i, j, n = 0;
    for (i = 0; i < p->n_units; ++i) {
        int dup = 0;
        for (j = 0; j < i; ++j)
            if (p->units[j].fused && p->units[i].fused && p->units[j].role != 2
                && p->units[i].role != 2 && p->units[j].span_off == p->units[i].span_off) dup = 1;
        if (!dup) n++;
    }
    return n;
}
static inline long m2g_want_bytes(const struct M2GPlan *p) {
    int i;
    long b = 0;
    for (i = 0; i < p->n_units; ++i) b += p->units[i].bytes;
    return b;
}

#endif                                                 /* M2G_KERNELS_H */
