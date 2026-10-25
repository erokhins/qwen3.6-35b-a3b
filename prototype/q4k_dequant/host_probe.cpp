/* host_probe.cpp — the same decoders, built by the host compiler.
 *
 * `q4k_dequant_kernels.h` is compiled twice: by hipcc for the device, and here by g++. The probe
 * file carries, per row, the block bytes and the f32 bit patterns the audited CPU reader produced for
 * them; this program decodes the bytes with the shared functions and reports where the bit patterns
 * differ. It is not the gate: it is the control that shows the shared code and the comparator both
 * work before any device verdict is trusted, and it is the arm that makes a device-vs-host difference
 * attributable to the device compiler/ISA rather than to two spellings of the decoder
 * (`pitfalls/40-gates-and-admissions.md` §"A check whose reference side is derived from the thing
 * under test").
 *
 * No device, no timer, no GGUF. Exit 0 only if every row's verdict matched its expectation class, the
 * row count is non-zero, and at least one MISMATCH row actually fired (a comparator that has never
 * seen a failure is not evidence — pitfalls/40).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "q4k_dequant_kernels.h"

static int hexnib(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* tab-separated fields, no collapsing, newline stripped */
static int split(char *line, char **out, int max) {
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
    return c;
}

static int unhex(const char *s, uint8_t *dst, int want) {
    if ((int)strlen(s) != 2 * want) return -1;
    for (int i = 0; i < want; ++i) {
        const int a = hexnib(s[2 * i]), b = hexnib(s[2 * i + 1]);
        if (a < 0 || b < 0) return -1;
        dst[i] = (uint8_t)((a << 4) | b);
    }
    return want;
}

static int unhex_words(const char *s, float *dst, int want) {
    int n = 0;
    while (*s && n < want) {
        uint32_t b = 0;
        for (int k = 0; k < 8; ++k) {
            const int v = hexnib(s[k]);
            if (v < 0) return -1;
            b = (b << 4) | (uint32_t)v;
        }
        memcpy(&dst[n++], &b, sizeof(b));
        s += 8;
        if (*s == ',') ++s;
    }
    while (*s == ',') ++s;
    return *s ? -1 : n;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: host_probe PROBEFILE [F16TABLE]\n"
               "  rows: P\\t<case_id>\\t<kind>\\t<mode>\\t<expect>\\t<n_elts>\\t<hexbytes>\\t<hexbits>\n"
               "  f16:  F\\t<hex4>\\t<hex8>   (the reader's own f16->f32 answer, all 65536 inputs)\n");
        return 2;
    }
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) { printf("FAIL cannot open %s\n", argv[1]); return 2; }
    char line[262144];
    int rows = 0, bad_rows = 0, unexpected = 0, mismatch_fired = 0, match_rows = 0;
    while (fgets(line, sizeof(line), fh)) {
        char *f[8];
        const int nf = split(line, f, 8);
        if (nf == 0 || f[0][0] == '#' || f[0][0] == '\0') continue;
        if (strcmp(f[0], "P") != 0) {
            printf("FAIL probe row %d has unknown tag '%s'\n", rows, f[0]);
            ++bad_rows;
            continue;
        }
        if (nf < 8) {
            printf("FAIL probe row %d has %d fields, expected 8\n", rows, nf);
            ++bad_rows;
            continue;
        }
        const int id = atoi(f[1]);
        const int kind = atoi(f[2]);
        const int mode = atoi(f[3]);
        const int expect = atoi(f[4]);
        const int n_elts = atoi(f[5]);
        const int bb = kind == Q4KD_KIND_Q4K ? Q4KD_Q4K_BYTES : Q4KD_Q8_0_BYTES;
        if (kind != Q4KD_KIND_Q4K && kind != Q4KD_KIND_Q8_0) {
            printf("FAIL probe row %d kind %d\n", id, kind);
            ++bad_rows;
            continue;
        }
        if (n_elts != (kind == Q4KD_KIND_Q4K ? QK_K : QK8_0)) {
            printf("FAIL probe row %d n_elts %d is not one %s block\n", id, n_elts,
                   kind == Q4KD_KIND_Q4K ? "Q4_K" : "Q8_0");
            ++bad_rows;
            continue;
        }
        if (mode < 0 || mode >= Q4KD_MODE_COUNT) {
            printf("FAIL probe row %d mode %d out of range\n", id, mode);
            ++bad_rows;
            continue;
        }
        uint8_t blk[Q4KD_MAX_BLOCK_BYTES];
        memset(blk, 0, sizeof(blk));
        if (unhex(f[6], blk, bb) != bb) {
            printf("FAIL probe row %d byte field is not %d hex bytes\n", id, bb);
            ++bad_rows;
            continue;
        }
        float want[Q4KD_MAX_BLOCK_ELTS];
        if (unhex_words(f[7], want, n_elts) != n_elts) {
            printf("FAIL probe row %d expected-value field is not %d hex f32 words\n", id, n_elts);
            ++bad_rows;
            continue;
        }
        float y[Q4KD_MAX_BLOCK_ELTS];
        memset(y, 0, sizeof(y));
        q4kd_decode_block(blk, kind, mode, y);
        int nbad = 0;
        for (int i = 0; i < n_elts; ++i) {
            uint32_t a, b;
            memcpy(&a, &y[i], sizeof(a));
            memcpy(&b, &want[i], sizeof(b));
            /* the same q4kd_word_eq the device counters use: if this file compared with != and the
             * kernel compared with the NaN rule, the host arm and the device arm would disagree on
             * which rows are clean, and that disagreement is not attributable to anything. */
            if (!q4kd_word_eq(a, b)) ++nbad;
        }
        int ok;
        const char *verdict;
        if (expect == Q4KD_EXP_MISMATCH) {
            ok = nbad > 0;
            verdict = ok ? "FIRED" : "NO-FIRE";
            if (ok) ++mismatch_fired;
        } else {
            ok = nbad == 0;
            verdict = ok ? "MATCH" : "DEVIATE";
            if (ok) ++match_rows;
        }
        if (!ok) ++unexpected;
        printf("PROBE id=%d kind=%s mode=%d expect=%d n_elts=%d n_bad=%d verdict=%s\n", id,
               kind == Q4KD_KIND_Q4K ? "Q4_K" : "Q8_0", mode, expect, n_elts, nbad, verdict);
        ++rows;
    }
    fclose(fh);
    printf("PROBE RESULT rows=%d bad_rows=%d unexpected=%d match_rows=%d mismatch_fired=%d\n",
           rows, bad_rows, unexpected, match_rows, mismatch_fired);

    /* The exhaustive f16 arm: every one of the 65536 half-precision patterns, judged against the CPU
     * reader's own answer. The hand-worked f16 cases in the driver's self-test are 13 points; a
     * bit-rebias that mishandles sNaN quieting or a denormal is exactly the kind of defect that
     * lives between the points, and it moves a whole 32-element sub-block of the as_q4nl control.
     * The expectation side is struct.unpack('<e') in gate.py, not a formula from this file. */
    int f_given = 0, f_rows = 0, f_bad = 0;
    if (argc >= 3) {
        f_given = 1;
        FILE *ff = fopen(argv[2], "rb");
        if (!ff) { printf("FAIL cannot open %s\n", argv[2]); return 2; }
        while (fgets(line, sizeof(line), ff)) {
            char *f[3];
            const int nf = split(line, f, 3);
            if (nf == 0 || f[0][0] == '#' || f[0][0] == '\n' || f[0][0] == '\0') continue;
            if (strcmp(f[0], "F") != 0 || nf < 3) {
                printf("FAIL f16 table row %d is not F<hex4>\\t<hex8>: %s\n", f_rows, line);
                ++f_bad;
                continue;
            }
            uint32_t hb = 0, wb = 0;
            for (int k = 0; k < 4; ++k) { const int vc = hexnib(f[1][k]); if (vc < 0) { hb = 0xFFFFu; break; } hb = (hb << 4) | (uint32_t)vc; }
            for (int k = 0; k < 8; ++k) { const int vc = hexnib(f[2][k]); if (vc < 0) { wb = 0xFFFFFFFFu; break; } wb = (wb << 4) | (uint32_t)vc; }
            float got = q4kd_f16(hb);
            uint32_t gb;
            memcpy(&gb, &got, sizeof(gb));
            /* STRICT bit equality here, deliberately: unlike a decoded product, the f16->f32
             * conversion has a defined NaN form on both routes (canonical +/-0x7FC00000), so a
             * payload or sign drift is a real disagreement and must be caught. q4kd_word_eq would
             * wave the sNaN-quieting defect through, which is the exact bug this table exists to
             * catch. */
            if (gb != wb) {
                if (f_bad < 8) printf("FAIL f16 0x%04x: got %08x want %08x\n", hb, gb, wb);
                ++f_bad;
            }
            ++f_rows;
        }
        fclose(ff);
    }
    printf("F16 given=%d rows=%d bad=%d\n", f_given, f_rows, f_bad);
    const int pass = rows > 0 && bad_rows == 0 && unexpected == 0 && match_rows > 0 &&
                     mismatch_fired > 0 && (!f_given || (f_rows > 0 && f_bad == 0));
    printf("PROBE RESULT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
