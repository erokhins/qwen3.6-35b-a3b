// llama_tokens_dump.cpp — a template-free llama.cpp front end that is fed token IDS, not text.
//
// Why this file exists. The pinned HF chain fixture (checkpoints/dump-manifest-chain-2026-10-26.json,
// input.token_ids = 32 sampled ids) and every existing llama.cpp dump run use different stimuli, and no
// text prefill can reproduce those 32 ids: detokenize -> 205 bytes -> re-tokenize reproduces 30 of 32
// (first divergence index 3), and for a byte-level BPE the byte string is forced by the ids, so that
// candidate is the only candidate. Evidence and proof:
// quantization/evidence/llama-stimulus-alignment-2026-10-27.md. The only remaining way to give the
// oracle that stimulus is an id input channel, and the vendored front ends have none: text is their only
// input. This program is that channel, built as a *client* of the vendored library rather than a patch
// to a vendored front end, so engines/llama.cpp stays pristine.
//
// THE API PATH THAT BYPASSES TEMPLATING, named because "template-free" has to be falsifiable:
//   ids are read from a file as decimal integers -> batch.token[i] (llama_batch_init) ->
//   llama_decode(ctx, batch).
// That is the whole input path. It deliberately never calls llama_chat_apply_template /
// common_chat_apply_template, common_tokenize / common_tokenize_text, llama_tokenize, llama_detokenize,
// or llama_add_bos_token, and it never links libllama-common (the library that owns chat templating and
// tokenization in this tree). Because no tokenizer runs, no BOS can be prepended and no conversation
// wrapper can be inserted: the ids the graph is built from are the ids in the file, and the
// instrumentation's own manifest graphs[].token_ids — read from the llama_ubatch inside
// llama-context.cpp::process_ubatch(), not echoed from here — is what a caller checks that against.
//
// The dump itself is tools/llama-instrument.patch, which arms inside llama_context::process_ubatch();
// this program needs no dump call of its own. Plan of record: tools/LLAMA_DUMP_NOTES.md.
//
// CPU-only by construction and by gate: n_gpu_layers = 0, and tools/llama_tokens_build.sh builds it
// against the CPU-only libllama whose ldd contains no ROCm/HIP library. No lock is taken, nothing is
// timed, and no_perf=true so the library prints no timings either.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char * fmt, const char * arg) {
    fprintf(stderr, "TOKDUMP verdict=REFUSED reason=%s %s\n", fmt, arg ? arg : "");
    exit(2);
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m <gguf> -t <tokens-file> [-o <logits-out>] [--n-ctx N] [--n-batch N]\n"
            "          [--n-ubatch N] [--threads N]\n"
            "  <tokens-file>: decimal token ids, whitespace or comma separated, one batch, positions 0..n-1\n",
            argv0);
    exit(2);
}

int main(int argc, char ** argv) {
    const char * path_model  = nullptr;
    const char * path_tokens = nullptr;
    const char * path_out    = nullptr;
    uint32_t n_ctx = 128, n_batch = 64, n_ubatch = 64;
    int32_t n_threads = 8;

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        auto next = [&](const char * name) -> const char * {
            (void) name;
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };
        if (!strcmp(a, "-m"))               { path_model = next(a); }
        else if (!strcmp(a, "-t"))          { path_tokens = next(a); }
        else if (!strcmp(a, "-o"))          { path_out = next(a); }
        else if (!strcmp(a, "--n-ctx"))     { n_ctx = (uint32_t) atoi(next(a)); }
        else if (!strcmp(a, "--n-batch"))    { n_batch = (uint32_t) atoi(next(a)); }
        else if (!strcmp(a, "--n-ubatch"))   { n_ubatch = (uint32_t) atoi(next(a)); }
        else if (!strcmp(a, "--threads"))    { n_threads = (int32_t) atoi(next(a)); }
        else usage(argv[0]);
    }
    if (!path_model || !path_tokens) {
        usage(argv[0]);
    }

    // ---- the id channel --------------------------------------------------------------
    // Numbers in, numbers out. Nothing here is a string that could be templated or tokenized.
    std::vector<llama_token> ids;
    {
        FILE * f = fopen(path_tokens, "r");
        if (!f) {
            die("tokens-file-unreadable", path_tokens);
        }
        long v;
        while (fscanf(f, "%ld", &v) == 1) {
            ids.push_back((llama_token) v);
        }
        fclose(f);
    }
    if (ids.empty()) {
        die("tokens-file-empty", path_tokens);
    }
    if ((uint32_t) ids.size() > n_ubatch) {
        die("n-ids-exceeds-n-ubatch", "one ubatch is required for one graph");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;          // CPU-only: no layer is offloaded, and no device is enumerated
    mparams.load_mtp     = false;      // the 40-block main path, not the draft block
    llama_model * model = llama_model_load_from_file(path_model, mparams);
    if (!model) {
        die("model-load-failed", path_model);
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            die("token-id-out-of-vocab", "an id is not a row of the embedding table");
        }
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = n_ctx;
    cparams.n_batch        = n_batch;
    cparams.n_ubatch       = n_ubatch;
    cparams.n_seq_max      = 1;
    cparams.n_threads      = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings     = false;
    cparams.no_perf        = true;     // no timings printed, by flag and not by hoping
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        die("context-init-failed", path_model);
    }

    llama_batch batch = llama_batch_init((int32_t) ids.size(), /* embd */ 0, /* n_seq_max */ 1);
    batch.n_tokens = (int32_t) ids.size();
    for (size_t i = 0; i < ids.size(); i++) {
        batch.token[i]    = ids[i];
        batch.pos[i]      = (llama_pos) i;      // positions 0..n-1, the same fixture positions use
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // llama_batch_init mallocs logits[] and does not fill it; setting it is not optional. The last
        // row is the only output, which is what makes result_output one row of logits.
        batch.logits[i]   = (i + 1 == ids.size()) ? 1 : 0;
    }
    int n_outputs_marked = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        n_outputs_marked += batch.logits[i] != 0;
    }

    printf("TOKDUMP role=raw-token-prefill api_path=batch.token[]->llama_decode template_path_not_called=yes\n");
    printf("TOKDUMP ids_file=%s n_ids=%zu\n", path_tokens, ids.size());
    printf("TOKDUMP ids_consumed=");
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%s%d", i ? " " : "", (int) ids[i]);
    }
    printf("\n");
    printf("TOKDUMP n_ctx=%u n_batch=%u n_ubatch=%u n_threads=%d n_gpu_layers=0 load_mtp=0 "
           "n_vocab=%d vocab_wants_bos=%d n_tokens=%zu n_outputs_marked=%d (the last token only)\n",
           cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, (int) cparams.n_threads,
           (int) n_vocab, llama_vocab_get_add_bos(vocab) ? 1 : 0, ids.size(), n_outputs_marked);
    fflush(stdout);

    const int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        die("decode-failed", "llama_decode returned non-zero");
    }

    const float * logits = llama_get_logits(ctx);
    if (!logits) {
        die("no-logits", "llama_get_logits returned null: the graph produced no output row");
    }
    size_t n_logits = (size_t) n_vocab;

    // greedy argmax over the whole vocab row, no penalties, no bad-token mask, no grammar — that is the
    // referent "greedy argmax" for a fixture. Top-5 are printed so a near-tie is visible, not hidden.
    size_t best = 0;
    for (size_t i = 1; i < n_logits; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    int top_idx[5] = { 0, 0, 0, 0, 0 };
    float top_val[5] = { -1e30f, -1e30f, -1e30f, -1e30f, -1e30f };
    for (size_t i = 0; i < n_logits; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > top_val[k]) {
                for (int j = 4; j > k; j--) {
                    top_val[j] = top_val[j - 1];
                    top_idx[j] = top_idx[j - 1];
                }
                top_val[k] = logits[i];
                top_idx[k] = (int) i;
                break;
            }
        }
    }

    const char * out = path_out ? path_out : "logits.f32";
    FILE * fo = fopen(out, "wb");
    if (!fo) {
        die("logits-out-unwritable", out);
    }
    const size_t wrote = fwrite(logits, sizeof(float), n_logits, fo);
    fclose(fo);
    if (wrote != n_logits) {
        die("logits-out-short", out);
    }

    printf("TOKDUMP logits_rows=1 n_vocab=%d logits_file=%s logits_bytes=%zu\n",
           (int) n_vocab, out, n_logits * sizeof(float));
    printf("TOKDUMP argmax=%d argmax_logit=%.9g top5=", (int) best, (double) logits[best]);
    for (int k = 0; k < 5; k++) {
        printf("%s%d:%.9g", k ? " " : "", top_idx[k], (double) top_val[k]);
    }
    printf("\n");
    printf("TOKDUMP DONE verdict=PASS n_ids=%zu\n", ids.size());
    fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
