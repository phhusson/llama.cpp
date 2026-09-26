#pragma once

#include "ggml-backend-impl.h"
#import <Metal/Metal.h>

struct ggml_apple_fence {
    id<MTLSharedEvent> event;
    uint64_t value;
};

static ggml_backend_fence_t ggml_apple_fence_init(id<MTLSharedEvent> event, uint64_t value);

static ggml_backend_fence_t ggml_apple_fence_dup(ggml_backend_fence_t fence) {
    struct ggml_apple_fence * ctx = fence->context;
    return ggml_apple_fence_init(ctx->event, ctx->value);
}

static void ggml_apple_fence_free(ggml_backend_fence_t fence) {
    struct ggml_apple_fence * ctx = fence->context;
    [ctx->event release];
    free(ctx);
}

static void ggml_apple_fence_sync(ggml_backend_fence_t fence) {
    struct ggml_apple_fence * ctx = fence->context;
    GGML_ASSERT([ctx->event waitUntilSignaledValue:ctx->value timeoutMS:60000]);
}

static ggml_backend_fence_t ggml_apple_fence_init(id<MTLSharedEvent> event, uint64_t value) {
    struct ggml_apple_fence * ctx = malloc(sizeof(*ctx));
    GGML_ASSERT(ctx);
    ctx->event = [event retain];
    ctx->value = value;
    struct ggml_backend_fence_i iface = { ggml_apple_fence_dup, ggml_apple_fence_free, ggml_apple_fence_sync };
    return ggml_backend_fence_init_native(GGML_BACKEND_FENCE_METAL, iface, ctx);
}
