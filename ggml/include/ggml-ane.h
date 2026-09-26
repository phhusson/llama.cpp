#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

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
