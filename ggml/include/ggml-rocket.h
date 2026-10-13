#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// RK3588 NPU backend. Runs BF16 matmuls on the three NPU cores, using the
// DMA_BUF buffer type for shared memory and a helper backend's OP_CPY to
// convert weights.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rocket_reg(void);

#ifdef __cplusplus
}
#endif
