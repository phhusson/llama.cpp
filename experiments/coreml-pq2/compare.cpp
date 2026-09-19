#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (const char * path = std::getenv("PQ2_COMPARE_STDOUT")) { if (!std::freopen(path, "w", stdout)) { return 9; } }
    if (const char * path = std::getenv("PQ2_COMPARE_STDERR")) { if (!std::freopen(path, "w", stderr)) { return 9; } }
    if (argc < 5) {
        fprintf(stderr, "usage: compare MODEL PROMPT OUTPUT_PREFIX UBATCH [TEACHER_TOKENS]\n");
        return 1;
    }
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 2; }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_owner(model, llama_model_free);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream prompt_file(argv[2]);
    std::string prompt((std::istreambuf_iterator<char>(prompt_file)), std::istreambuf_iterator<char>());
    int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, false);
    std::vector<llama_token> tokens(count);
    count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, false);
    if (count < 4096) { fprintf(stderr, "need at least 4096 prompt tokens\n"); return 3; }
    tokens.resize(4096);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4352;
    cp.n_batch = 4096;
    cp.n_ubatch = std::stoi(argv[4]);
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_owner(ctx, llama_free);
    if (!ctx || llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) { return 4; }
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[3])+".logits.f32", std::ios::binary);
    std::ofstream emitted(std::string(argv[3])+".tokens.i32", std::ios::binary);
    std::ifstream teacher;
    if (argc > 5) { teacher.open(argv[5], std::ios::binary); }
    for (int step = 0; step <= 8; ++step) {
        const float * values = llama_get_logits_ith(ctx, -1);
        for (int i = 0; i < nv; ++i) { if (!std::isfinite(values[i])) { return 5; } }
        logits.write((const char *) values, nv*sizeof(float));
        llama_token next = std::max_element(values, values+nv)-values;
        printf("step %d top1 %d\n", step, next);
        if (step == 8) { break; }
        if (teacher.is_open()) {
            teacher.read((char *) &next, sizeof(next));
            if (!teacher) { return 6; }
        }
        emitted.write((const char *) &next, sizeof(next));
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { return 7; }
    }
    printf("vocab %d, prompt 4096, decode 8, ubatch %u\n", nv, cp.n_ubatch);
    llama_perf_context_print(ctx);
    return 0;
}
