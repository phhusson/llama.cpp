#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// GGML_ANE_MTLQUANTS selects comma-separated quant types for Metal-to-F16 conversion (default: all supported by Metal).
// Names are case-insensitive; an empty value disables quantized ANE matmul. Read once at startup.
// Explicitly selected quants must support Metal CPY to F16 or initialization asserts.
// GGML_ANE_MIN_MACS and GGML_ANE_MIN_INTENSITY set minimum MACs and MACs per estimated byte moved.
// Defaults are 6e9 MACs and 45 MACs/byte, calibrated on M4. Zero disables that threshold.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ane_reg(void);

// Borrowed Metal objects remain valid for the lifetime of the ggml buffer.
struct ggml_backend_ane_buffer_api {
    bool (*supports_buft)(ggml_backend_buffer_type_t buft, void * metal_device);
    bool (*get_buffer)(const struct ggml_tensor * tensor, void ** metal_buffer, size_t * offset);
};

typedef const struct ggml_backend_ane_buffer_api * (*ggml_backend_ane_get_buffer_api_t)(void);

#ifdef __cplusplus
}
#endif
