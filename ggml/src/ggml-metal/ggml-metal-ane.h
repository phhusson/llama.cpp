#pragma once
#include "ggml.h"
#include "ggml-metal-device.h"

bool ggml_metal_ane_enabled(void);
void * ggml_metal_ane_init(void);
// Query the embedded shapes without compiling or loading programs.
int64_t ggml_metal_ane_rows(void * opaque, const struct ggml_tensor * op);
void ggml_metal_ane_check(void * opaque, const struct ggml_tensor * op, int64_t rows);
void ggml_metal_ane_free(void * opaque);
bool ggml_metal_ane_load(void * opaque, const struct ggml_tensor * op, int64_t rows);
bool ggml_metal_ane_begin(void * opaque, ggml_metal_device_t dev);
bool ggml_metal_ane_prepare(void * opaque, ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb, const struct ggml_tensor * op, int64_t rows);
bool ggml_metal_ane_submit(void * opaque, const struct ggml_tensor * op, int64_t rows);
void ggml_metal_ane_join(void * opaque, ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb, const struct ggml_tensor * op, int64_t rows);
bool ggml_metal_ane_finish(void * opaque);
