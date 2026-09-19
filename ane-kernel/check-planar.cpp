// Check lossless load/readback and the planar matvec/matmul readers without ANE.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--ane") != 0)) {
        fprintf(stderr, "usage: %s [--ane]\n", argv[0]); return 1;
    }
    const bool split = argc == 2;
    if (!split) { setenv("GGML_METAL_ANE", "0", 1); setenv("GGML_METAL_ANE_PLANAR", "1", 1); }
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { return 1; }
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto factory = (ggml_backend_dev_get_alt_type_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_alt_type");
    if (!factory) { return 2; }
    auto backend = ggml_backend_dev_init(dev, nullptr);
    const int rows = split ? 10240 : 2048;
    const double fraction = getenv("GGML_METAL_ANE_FRACTION") ? atof(getenv("GGML_METAL_ANE_FRACTION")) : .65;
    const int boundary = int(rows*fraction/256)*256;
    double worst = 0;
    int cases = 0;
    for (int k : (split ? std::vector<int>{5120} : std::vector<int>{128,5120,6144,17408})) {
        ggml_init_params params = {32*1024*1024, nullptr, true};
        auto weights = ggml_init(params);
        auto w = ggml_new_tensor_2d(weights, GGML_TYPE_PQ2_0, k, rows);
        auto wb = ggml_backend_alloc_ctx_tensors_from_buft(weights, factory(dev, w->type));
        std::vector<unsigned char> packed(ggml_nbytes(w)), back(packed.size());
        for (int row=0; row<rows; ++row) {
            for (int b=0; b<k/128; ++b) {
                auto block = packed.data()+(row*(k/128)+b)*34;
                ggml_fp16_t d = ggml_fp32_to_fp16(float(1+(row+b)%7)/256);
                memcpy(block, &d, 2);
                for (int j=0; j<128; ++j) {
                    unsigned q = (row*11+b*7+j*13+j/5)%3;
                    block[2+j/4] |= q << (2*(j%4));
                }
            }
        }
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
        ggml_backend_tensor_get(w, back.data(), 0, back.size());
        if (back != packed) { fprintf(stderr,"readback mismatch k=%d\n",k); return 3; }
        for (int tokens : (split ? std::vector<int>{2048} : std::vector<int>{1,2,8,32,129})) {
            auto ctx = ggml_init(params);
            auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
            auto y = ggml_mul_mat(ctx, w, a);
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<float> input(k*tokens), output(rows*tokens);
            for (size_t i=0;i<input.size();++i) { input[i]=float(int((i*17+i/31)%97)-48)/64; }
            ggml_backend_tensor_set(a, input.data(), 0, input.size()*4);
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 4; }
            ggml_backend_tensor_get(y, output.data(), 0, output.size()*4);
            for (float v : output) { if (!std::isfinite(v)) { return 5; } }
            for (int sample=0;sample<64;++sample) {
                int row = sample==1 ? rows-1 : (sample*997)%rows, t=(sample*73)%tokens;
                if (split && boundary>0 && boundary<rows && sample==2) { row=boundary-1; }
                if (split && boundary>0 && boundary<rows && sample==3) { row=boundary; }
                double ref=0;
                for (int j=0;j<k;++j) {
                    auto block=packed.data()+(row*(k/128)+j/128)*34;
                    ggml_fp16_t d; memcpy(&d, block, 2);
                    int q=((block[2+(j%128)/4]>>(2*(j%4)))&3)-1;
                    ref += double(ggml_fp16_to_fp32(d))*q*input[t*k+j];
                }
                double error=std::abs(output[t*rows+row]-ref);
                worst=std::max(worst,error);
                if(error>(split ? 1e-3+std::abs(ref)*.02 : 1e-4+std::abs(ref)*1e-4)) {
                    fprintf(stderr,"mismatch k=%d t=%d row=%d ref=%g got=%g\n",k,tokens,row,ref,output[t*rows+row]);
                    return 6;
                }
            }
            ++cases;
            ggml_backend_buffer_free(buf); ggml_free(ctx);
        }
        ggml_backend_buffer_free(wb); ggml_free(weights);
    }
    printf("PASS: %d matmuls, all packed weights round-trip exactly, max scalar error %.9g\n",cases,worst);
    ggml_backend_free(backend);
}
