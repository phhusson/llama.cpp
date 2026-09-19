#pragma once
#include "ggml-backend.h"
#include "ggml-metal-device.h"
#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_metal_pq2_planar_buffer_type(ggml_backend_dev_t dev);
bool ggml_metal_buffer_is_pq2_planar(ggml_backend_buffer_t buffer);
bool ggml_metal_pq2_is_planar(const struct ggml_tensor * tensor);
bool ggml_metal_pq2_supports_op(const struct ggml_tensor * op);
struct ggml_metal_buffer_id ggml_metal_pq2_buffer(const struct ggml_tensor * tensor, bool scales);
void * ggml_metal_pq2_surface(const struct ggml_tensor * tensor, bool scales);
float ggml_metal_pq2_value(const struct ggml_tensor * tensor, int64_t row, int64_t col);
#ifdef __cplusplus
}
#endif
