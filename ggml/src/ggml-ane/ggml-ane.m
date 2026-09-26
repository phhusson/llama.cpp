#include "ggml-ane.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-apple-fence.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <unistd.h>

@interface NSObject (GGMLANE)
+ (id)modelWithMILText:(NSData *)mil weights:(NSDictionary *)weights optionsPlist:(id)options;
+ (id)inMemoryModelWithDescriptor:(id)descriptor;
- (NSString *)hexStringIdentifier;
- (BOOL)compileWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)loadWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)unloadWithQoS:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithQoS:(unsigned int)qos options:(NSDictionary *)options request:(id)request error:(NSError **)error;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats procedureIndex:(NSNumber *)index sharedEvents:(id)events;
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(uint32_t)index eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
- (id)IOSurfaceSharedEvent;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end

static id<MTLDevice> ane_device;

static size_t ane_page_align(size_t size) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    return (size + page - 1) & ~(page - 1);
}

static NSDictionary * ane_surface_properties(size_t size) {
    return @{
        (id) kIOSurfaceWidth:           @(size),
        (id) kIOSurfaceHeight:          @1,
        (id) kIOSurfaceBytesPerElement: @1,
        (id) kIOSurfaceBytesPerRow:     @(size),
        (id) kIOSurfaceAllocSize:       @(size),
        (id) kIOSurfacePixelFormat:     @0,
    };
}

struct ane_buffer {
    IOSurfaceRef surface;
    id<MTLBuffer> metal;
    NSMutableDictionary * surfaces;
};

static ggml_backend_buffer_t ane_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size);

static bool ane_is_buft(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.alloc_buffer == ane_buffer_alloc;
}

static struct ane_buffer * ane_tensor_buffer(const struct ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buffer && ane_is_buft(buffer->buft));
    return buffer->context;
}

static void ane_buffer_free(ggml_backend_buffer_t buffer) {
    struct ane_buffer * ctx = buffer->context;
    [ctx->surfaces release];
    [ctx->metal release];
    CFRelease(ctx->surface);
    free(ctx);
}

static void * ane_buffer_base(ggml_backend_buffer_t buffer) {
    return IOSurfaceGetBaseAddress(((struct ane_buffer *) buffer->context)->surface);
}

static void ane_buffer_set(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void ane_buffer_get(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void ane_buffer_memset(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void ane_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(ane_buffer_base(buffer), value, buffer->size);
}

static void ane_buffer_reset(ggml_backend_buffer_t buffer) {
    struct ane_buffer * ctx = buffer->context;
    [ctx->surfaces removeAllObjects];
}

static ggml_backend_buffer_t ane_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    @autoreleasepool {
        if (size > ane_device.maxBufferLength) {
            return NULL;
        }
        size_t allocated = ane_page_align(MAX(size, 1));
        if (allocated > ane_device.maxBufferLength) {
            return NULL;
        }
        struct ane_buffer * ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            return NULL;
        }
        ctx->surface = IOSurfaceCreate((CFDictionaryRef) ane_surface_properties(allocated));
        if (!ctx->surface) {
            free(ctx);
            return NULL;
        }
        ctx->metal = [ane_device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(ctx->surface) length:allocated options:MTLResourceStorageModeShared deallocator:nil];
        if (!ctx->metal) {
            CFRelease(ctx->surface);
            free(ctx);
            return NULL;
        }
        ctx->surfaces = [NSMutableDictionary new];
        struct ggml_backend_buffer_i iface = {
            .free_buffer   = ane_buffer_free,
            .get_base      = ane_buffer_base,
            .memset_tensor = ane_buffer_memset,
            .set_tensor    = ane_buffer_set,
            .get_tensor    = ane_buffer_get,
            .clear         = ane_buffer_clear,
            .reset         = ane_buffer_reset,
        };
        return ggml_backend_buffer_init(buft, iface, ctx, size);
    }
}

static const char * ane_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "ANE";
}

static size_t ane_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return (size_t) sysconf(_SC_PAGESIZE);
}

static size_t ane_buft_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ane_device.maxBufferLength;
}

static bool ane_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static struct ggml_backend_buffer_type ane_buft = {
    .iface         = {
        .get_name      = ane_buft_name,
        .alloc_buffer  = ane_buffer_alloc,
        .get_alignment = ane_buft_alignment,
        .get_max_size  = ane_buft_max_size,
        .is_host       = ane_buft_is_host,
    },
};

static id ane_tensor_surface(const struct ggml_tensor * tensor) {
    struct ane_buffer * ctx = ane_tensor_buffer(tensor);
    size_t offset = (const char *) tensor->data - (const char *) IOSurfaceGetBaseAddress(ctx->surface);
    size_t size = ane_page_align(ggml_nbytes(tensor));
    GGML_ASSERT(offset % (size_t) sysconf(_SC_PAGESIZE) == 0);
    GGML_ASSERT(offset <= ctx->metal.length && size <= ctx->metal.length - offset);

    // Wrap only this tensor; ANE child-surface offsets wrap above 4 GiB.
    NSString * key = [NSString stringWithFormat:@"%zu:%zu", offset, size];
    id surface = ctx->surfaces[key];
    if (!surface) {
        NSMutableDictionary * props = [ane_surface_properties(size) mutableCopy];
        props[@"IOSurfaceAddress"] = @((uintptr_t) tensor->data);
        surface = (id) IOSurfaceCreate((CFDictionaryRef) props);
        [props release];
        if (!surface) {
            return nil;
        }
        GGML_ASSERT(IOSurfaceGetBaseAddress((IOSurfaceRef) surface) == tensor->data);
        GGML_ASSERT(IOSurfaceGetAllocSize((IOSurfaceRef) surface) == size);
        ctx->surfaces[key] = surface;
        [surface release];
    }
    return [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:(IOSurfaceRef) surface];
}

static bool ane_metal_supports_buft(ggml_backend_buffer_type_t buft, void * device) {
    return ane_is_buft(buft) && [(id<MTLDevice>) device registryID] == ane_device.registryID;
}

static bool ane_metal_buffer(const struct ggml_tensor * tensor, void ** metal, size_t * offset) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || !ane_is_buft(buffer->buft)) {
        return false;
    }
    struct ane_buffer * ctx = buffer->context;
    *metal = ctx->metal;
    *offset = (const char *) tensor->data - (const char *) IOSurfaceGetBaseAddress(ctx->surface);
    GGML_ASSERT(*offset + ggml_nbytes(tensor) <= ctx->metal.length);
    return true;
}

static const struct ggml_backend_ane_buffer_api * ane_get_buffer_api(void) {
    static const struct ggml_backend_ane_buffer_api api = { ane_metal_supports_buft, ane_metal_buffer };
    return &api;
}

@interface GGMLANEModel : NSObject {
@public
    id model;
}
@end

@implementation GGMLANEModel
- (void)dealloc {
    [model unloadWithQoS:21 error:NULL];
    [model release];
    [super dealloc];
}
@end

@interface GGMLANETicket : NSObject {
@public
    atomic_bool completed;
}
@end

@implementation GGMLANETicket
@end

struct ane_context {
    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> ready;
    id<MTLSharedEvent> done;
    id<MTLComputePipelineState> cast;
    id<MTLComputePipelineState> convert;
    uint64_t sequence;
    NSMutableDictionary * models;
    NSMutableArray * requests;
    NSMutableArray * commands;
    dispatch_group_t pending;
    atomic_bool failed;
};

static void ane_fail(struct ane_context * ctx, NSString * error) {
    GGML_LOG_ERROR("ANE: %s\n", error.UTF8String);
    atomic_store(&ctx->failed, true);
    // Drain device waits on failure; the graph is no longer usable.
    ctx->ready.signaledValue = UINT64_MAX;
    ctx->done.signaledValue = UINT64_MAX;
}

static GGMLANEModel * ane_model(struct ane_context * ctx, int64_t k, int64_t m, int64_t n) {
    NSString * key = [NSString stringWithFormat:@"%lld:%lld:%lld", k, m, n];
    GGMLANEModel * cached = ctx->models[key];
    if (cached) {
        return cached;
    }

    NSMutableString * mil = [NSMutableString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},"
        "{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},"
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16,[1,1,%lld,%lld]> a, tensor<fp16,[1,1,%lld,%lld]> w) {\n"
        "        bool no = const()[name = string(\"no\"), val = bool(false)];\n"
        "        bool yes = const()[name = string(\"yes\"), val = bool(true)];\n", n, k, m, k];

    // Compute W * A^T with smaller reductions; Metal transposes the output.
    const int64_t chunk = 2048;
    for (int64_t first = 0, i = 0; first < k; first += chunk, ++i) {
        int64_t count = MIN(chunk, k - first);
        [mil appendFormat:@"        tensor<int32,[4]> b%lld = const()[name = string(\"b%lld\"), val = tensor<int32,[4]>([0,0,0,%lld])];\n", i, i, first];
        [mil appendFormat:@"        tensor<int32,[4]> sa%lld = const()[name = string(\"sa%lld\"), val = tensor<int32,[4]>([1,1,%lld,%lld])];\n", i, i, n, count];
        [mil appendFormat:@"        tensor<int32,[4]> sw%lld = const()[name = string(\"sw%lld\"), val = tensor<int32,[4]>([1,1,%lld,%lld])];\n", i, i, m, count];
        [mil appendFormat:@"        tensor<fp16,[1,1,%lld,%lld]> a%lld = slice_by_size(x = a, begin = b%lld, size = sa%lld)[name = string(\"a%lld\")];\n", n, count, i, i, i, i];
        [mil appendFormat:@"        tensor<fp16,[1,1,%lld,%lld]> w%lld = slice_by_size(x = w, begin = b%lld, size = sw%lld)[name = string(\"w%lld\")];\n", m, count, i, i, i, i];
        [mil appendFormat:@"        tensor<fp16,[1,1,%lld,%lld]> p%lld = matmul(x = w%lld, y = a%lld, transpose_x = no, transpose_y = yes)[name = string(\"p%lld\")];\n", m, n, i, i, i, i];
        if (i) {
            [mil appendFormat:@"        tensor<fp16,[1,1,%lld,%lld]> r%lld = add(x = %@%lld, y = p%lld)[name = string(\"r%lld\")];\n", m, n, i, i == 1 ? @"p" : @"r", i - 1, i, i];
        }
    }
    int64_t last = (k - 1)/chunk;
    [mil appendFormat:@"    } -> (%@%lld);\n}\n", last ? @"r" : @"p", last];

    NSData * data = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData * dummy = [NSData dataWithBytes:"\0\0\0\0" length:4];
    id descriptor = [NSClassFromString(@"_ANEInMemoryModelDescriptor") modelWithMILText:data weights:@{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": dummy}} optionsPlist:nil];
    id model = [NSClassFromString(@"_ANEInMemoryModel") inMemoryModelWithDescriptor:descriptor];
    if (!model) {
        ane_fail(ctx, @"model creation failed");
        return nil;
    }

    NSString * directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[model hexStringIdentifier]];
    NSError * error = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error] ||
        ![data writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error] ||
        ![model compileWithQoS:21 options:@{} error:&error] || ![model loadWithQoS:21 options:@{} error:&error]) {
        ane_fail(ctx, [NSString stringWithFormat:@"matmul K=%lld M=%lld N=%lld compile/load: %@", k, m, n, error ? error.description : @"failed"]);
        return nil;
    }

    cached = [[GGMLANEModel new] autorelease];
    cached->model = [model retain];
    ctx->models[key] = cached;
    GGML_LOG_DEBUG("ANE: compiled F16 matmul %lld x %lld x %lld\n", k, m, n);
    return cached;
}

static const char * ane_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "ANE";
}

static void ane_synchronize(ggml_backend_t backend) {
    struct ane_context * ctx = backend->context;
    @autoreleasepool {
        if (dispatch_group_wait(ctx->pending, dispatch_time(DISPATCH_TIME_NOW, 60*NSEC_PER_SEC))) {
            ane_fail(ctx, [NSString stringWithFormat:@"request timeout: submitted=%llu ready=%llu done=%llu pending=%lu", ctx->sequence, ctx->ready.signaledValue, ctx->done.signaledValue, (unsigned long) ctx->requests.count]);
        }
        for (id<MTLCommandBuffer> cb in ctx->commands) {
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                ane_fail(ctx, cb.error.description ? cb.error.description : @"Metal command failed");
            }
        }
        [ctx->commands removeAllObjects];
        [ctx->requests removeAllObjects];
    }
    GGML_ASSERT(!atomic_load(&ctx->failed));
}

static void ane_free(ggml_backend_t backend) {
    ane_synchronize(backend);
    struct ane_context * ctx = backend->context;
    [ctx->models release];
    [ctx->requests release];
    [ctx->commands release];
    [ctx->queue release];
    [ctx->ready release];
    [ctx->done release];
    [ctx->cast release];
    [ctx->convert release];
    dispatch_release(ctx->pending);
    free(ctx);
    free(backend);
}

static bool ane_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op->op == GGML_OP_NONE || (op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE)) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const int32_t acc = ggml_get_op_params_i32(op, 0);
    const int32_t src = ggml_get_op_params_i32(op, 3);
    if ((acc != GGML_PREC_UNDEFINED && acc < GGML_PREC_F16) || (src != GGML_PREC_UNDEFINED && src < GGML_PREC_F16)) {
        return false;
    }
    const struct ggml_tensor * w = op->src[0], * a = op->src[1];
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (w->view_offs % page || (a->type == GGML_TYPE_F16 && a->view_offs % page)) {
        return false;
    }
    return w->type == GGML_TYPE_F16 && (a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
        w->ne[0] >= 32 && w->ne[1] >= 32 && a->ne[1] >= 32 &&
        w->ne[0] % 32 == 0 && w->ne[1] % 32 == 0 && a->ne[1] % 32 == 0 &&
        w->ne[0] <= 16384 && w->ne[1] <= 16384 && a->ne[1] <= 4096 &&
        w->ne[2] == 1 && w->ne[3] == 1 && a->ne[2] == 1 && a->ne[3] == 1 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(a) && ggml_is_contiguous(op);
}

static void ane_encode_convert(struct ane_context * ctx, id<MTLCommandBuffer> cb, const struct ggml_tensor * tensor, id<MTLBuffer> scratch, id<MTLBuffer> scales, bool output) {
    struct ane_buffer * buffer = ane_tensor_buffer(tensor);
    size_t offset = (const char *) tensor->data - (const char *) buffer->metal.contents;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:output ? ctx->convert : ctx->cast];
    [enc setBuffer:buffer->metal offset:offset atIndex:0];
    [enc setBuffer:scratch offset:0 atIndex:1];
    [enc setBuffer:scales offset:0 atIndex:3];
    if (output) {
        const uint32_t shape[] = { (uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1] };
        [enc setBytes:shape length:sizeof(shape) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(tensor->ne[1]/16, tensor->ne[0]/16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    } else {
        const uint32_t shape[] = { (uint32_t) tensor->ne[0], tensor->type == GGML_TYPE_F16 };
        [enc setBytes:shape length:sizeof(shape) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(tensor->ne[1], 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
}

static enum ggml_status ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph) {
    struct ane_context * ctx = backend->context;
    @autoreleasepool {
        if (atomic_load(&ctx->failed)) {
            return GGML_STATUS_FAILED;
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            struct ggml_tensor * op = graph->nodes[i];
            if (op->op == GGML_OP_NONE || (op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE)) {
                continue;
            }
            if (!ane_supports_op(backend->device, op)) {
                return GGML_STATUS_FAILED;
            }
            // Bound temporary surfaces and pending requests across asynchronous submissions.
            if (ctx->requests.count >= 8) {
                ane_synchronize(backend);
            }
            const struct ggml_tensor * w = op->src[0], * a = op->src[1];
            GGMLANEModel * model = ane_model(ctx, w->ne[0], w->ne[1], a->ne[1]);
            if (!model) {
                return GGML_STATUS_FAILED;
            }

            id ws = ane_tensor_surface(w);
            IOSurfaceRef act = NULL;
            id<MTLBuffer> act_metal = nil;
            id as = nil;
            id<MTLCommandBuffer> prepare = [ctx->queue commandBuffer];
            size_t size = ane_page_align(ggml_nelements(a)*sizeof(ggml_fp16_t));
            act = IOSurfaceCreate((CFDictionaryRef) ane_surface_properties(size));
            if (!act) {
                return GGML_STATUS_ALLOC_FAILED;
            }
            act_metal = [ane_device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(act) length:size options:MTLResourceStorageModeShared deallocator:nil];
            if (!act_metal) {
                CFRelease(act);
                return GGML_STATUS_ALLOC_FAILED;
            }
            as = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:act];

            size_t out_size = ane_page_align(ggml_nelements(op)*sizeof(ggml_fp16_t));
            IOSurfaceRef out = IOSurfaceCreate((CFDictionaryRef) ane_surface_properties(out_size));
            if (!out) {
                [act_metal release];
                if (act) {
                    CFRelease(act);
                }
                return GGML_STATUS_ALLOC_FAILED;
            }
            id<MTLBuffer> out_metal = [ane_device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(out) length:out_size options:MTLResourceStorageModeShared deallocator:nil];
            if (!out_metal) {
                [act_metal release];
                if (act) {
                    CFRelease(act);
                }
                CFRelease(out);
                return GGML_STATUS_ALLOC_FAILED;
            }

            id<MTLBuffer> scales = [ane_device newBufferWithLength:a->ne[1]*sizeof(float) options:MTLResourceStorageModePrivate];
            if (!scales) {
                [act_metal release];
                [out_metal release];
                CFRelease(act);
                CFRelease(out);
                return GGML_STATUS_ALLOC_FAILED;
            }
            ane_encode_convert(ctx, prepare, a, act_metal, scales, false);
            id os = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:out];
            GGML_ASSERT(ws && as && os && out_metal);

            // Metal prepares the input before ANE reads it.
            uint64_t sequence = ++ctx->sequence;
            [prepare encodeSignalEvent:ctx->ready value:sequence];
            id wait = [NSClassFromString(@"_ANESharedWaitEvent") waitEventWithValue:sequence sharedEvent:[(id) ctx->ready IOSurfaceSharedEvent]];
            id signal = [NSClassFromString(@"_ANESharedSignalEvent") signalEventWithValue:sequence symbolIndex:0 eventType:0 sharedEvent:[(id) ctx->done IOSurfaceSharedEvent]];
            id events = [NSClassFromString(@"_ANESharedEvents") sharedEventsWithSignalEvents:@[signal] waitEvents:@[wait]];
            id request = [NSClassFromString(@"_ANERequest") requestWithInputs:@[as, ws] inputIndices:@[@0, @1] outputs:@[os] outputIndices:@[@0] weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:events];
            GGML_ASSERT(request);

            GGMLANETicket * ticket = [[GGMLANETicket new] autorelease];
            atomic_init(&ticket->completed, false);
            NSString * label = [NSString stringWithFormat:@"%s (K=%lld M=%lld N=%lld, request=%llu)", op->name, w->ne[0], w->ne[1], a->ne[1], sequence];
            dispatch_group_enter(ctx->pending);
            [request setCompletionHandler:^(BOOL success, NSError * error) {
                if (!success) {
                    ane_fail(ctx, [NSString stringWithFormat:@"%@: execution: %@", label, error ? error.description : @"failed"]);
                }
                if (!atomic_exchange(&ticket->completed, true)) {
                    dispatch_group_leave(ctx->pending);
                }
            }];
            [ctx->requests addObject:request];
            [prepare addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    ane_fail(ctx, cb.error.description ? cb.error.description : @"input conversion failed");
                }
            }];
            [prepare commit];
            [ctx->commands addObject:prepare];

            NSError * error = nil;
            BOOL ok = [model->model evaluateWithQoS:21 options:@{} request:request error:&error];
            if (!ok) {
                ane_fail(ctx, [NSString stringWithFormat:@"%@: submission: %@", label, error ? error.description : @"failed"]);
                if (!atomic_exchange(&ticket->completed, true)) {
                    dispatch_group_leave(ctx->pending);
                }
            }

            // The exported backend fence must include this output conversion.
            id<MTLCommandBuffer> finish = [ctx->queue commandBuffer];
            [finish encodeWaitForEvent:ctx->done value:sequence];
            ane_encode_convert(ctx, finish, op, out_metal, scales, true);
            // Keep no-copy surface storage alive through the GPU conversion.
            [finish addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    ane_fail(ctx, cb.error.description ? cb.error.description : @"output conversion failed");
                }
                if (act) {
                    CFRelease(act);
                }
                CFRelease(out);
            }];
            [finish commit];
            [ctx->commands addObject:finish];
            [act_metal release];
            [out_metal release];
            [scales release];
            if (!ok || atomic_load(&ctx->failed)) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

struct ane_event {
    id<MTLSharedEvent> event;
    uint64_t value;
};

static void ane_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    struct ane_context * ctx = backend->context;
    struct ane_event * ev = event->context;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        [cb encodeSignalEvent:ev->event value:++ev->value];
        [cb commit];
        [ctx->commands addObject:cb];
    }
}

static bool ane_fence_wait(ggml_backend_t backend, ggml_backend_fence_t fence) {
    if (fence->type != GGML_BACKEND_FENCE_METAL) {
        return false;
    }
    struct ane_context * ctx = backend->context;
    struct ggml_apple_fence * ev = fence->context;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        [cb encodeWaitForEvent:ev->event value:ev->value];
        [cb commit];
        [ctx->commands addObject:cb];
    }
    return true;
}

static ggml_backend_fence_t ane_event_export(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    struct ane_event * ev = event->context;
    return ggml_apple_fence_init(ev->event, ev->value);
}

static void ane_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_fence_t fence = ane_event_export(backend, event);
    ane_fence_wait(backend, fence);
    ggml_backend_fence_free(fence);
}

static struct ggml_backend_i ane_backend_i = {
    .get_name           = ane_name,
    .free               = ane_free,
    .synchronize        = ane_synchronize,
    .graph_compute      = ane_graph_compute,
    .event_record       = ane_event_record,
    .event_wait         = ane_event_wait,
    .event_export_fence = ane_event_export,
    .fence_wait         = ane_fence_wait,
};

static ggml_backend_t ane_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    @autoreleasepool {
        struct ane_context * ctx = calloc(1, sizeof(*ctx));
        GGML_ASSERT(ctx);
        ctx->queue = [ane_device newCommandQueue];
        ctx->ready = [ane_device newSharedEvent];
        ctx->done = [ane_device newSharedEvent];
        ctx->models = [NSMutableDictionary new];
        ctx->requests = [NSMutableArray new];
        ctx->commands = [NSMutableArray new];
        ctx->pending = dispatch_group_create();
        atomic_init(&ctx->failed, false);

        // Scale each token to limit ANE loss of F16 subnormals; undo the scale after evaluation.
        NSString * source =
            @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "float input_value(device const uchar * x, uint i, uint is_half) {\n"
            "    return is_half ? float(((device const half *) x)[i]) : ((device const float *) x)[i];\n"
            "}\n"
            "\n"
            "// One threadgroup per token; shape contains K and the F16 input flag.\n"
            "kernel void cast(device const uchar * x [[buffer(0)]],\n"
            "                 device half * y [[buffer(1)]],\n"
            "                 constant uint2 & shape [[buffer(2)]],\n"
            "                 device float * scales [[buffer(3)]],\n"
            "                 uint t [[thread_index_in_threadgroup]],\n"
            "                 uint g [[threadgroup_position_in_grid]]) {\n"
            "    threadgroup float peaks[8];\n"
            "    threadgroup float scale;\n"
            "    float peak = 0.0f;\n"
            "\n"
            "    for (uint i = t; i < shape.x; i += 256) {\n"
            "        peak = max(peak, abs(input_value(x, g*shape.x + i, shape.y)));\n"
            "    }\n"
            "    peak = simd_max(peak);\n"
            "    if (t%32 == 0) {\n"
            "        peaks[t/32] = peak;\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "\n"
            "    // Combine the eight SIMD-group maxima and choose a power-of-two scale.\n"
            "    if (t == 0) {\n"
            "        for (uint i = 0; i < 8; i++) {\n"
            "            peak = max(peak, peaks[i]);\n"
            "        }\n"
            "        scale = peak > 0.0f && isfinite(peak) ? exp2(clamp(5.0f - floor(log2(peak)), -126.0f, 126.0f)) : 1.0f;\n"
            "        scales[g] = scale;\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "\n"
            "    for (uint i = t; i < shape.x; i += 256) {\n"
            "        y[g*shape.x + i] = half(input_value(x, g*shape.x + i, shape.y)*scale);\n"
            "    }\n"
            "}\n"
            "\n"
            "// Transpose [M,N] to [N,M] and remove each token's scale; shape contains M and N.\n"
            "kernel void convert(device float * y [[buffer(0)]],\n"
            "                    device const half * x [[buffer(1)]],\n"
            "                    constant uint2 & shape [[buffer(2)]],\n"
            "                    device const float * scales [[buffer(3)]],\n"
            "                    uint2 t [[thread_position_in_threadgroup]],\n"
            "                    uint2 g [[threadgroup_position_in_grid]]) {\n"
            "    threadgroup half tile[16][17];\n"
            "    tile[t.y][t.x] = x[(g.y*16 + t.y)*shape.y + g.x*16 + t.x];\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    y[(g.x*16 + t.y)*shape.x + g.y*16 + t.x] = float(tile[t.x][t.y])/scales[g.x*16 + t.y];\n"
            "}\n";

        NSError * error = nil;
        id<MTLLibrary> library = [ane_device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> cast = [library newFunctionWithName:@"cast"], convert = [library newFunctionWithName:@"convert"];
        ctx->cast = cast ? [ane_device newComputePipelineStateWithFunction:cast error:&error] : nil;
        ctx->convert = convert ? [ane_device newComputePipelineStateWithFunction:convert error:&error] : nil;
        [cast release];
        [convert release];
        [library release];

        static ggml_guid guid = { 0xa1, 0x9e, 0x80, 0x12, 0x2f, 0x42, 0x4a, 0xe2, 0xb0, 0x57, 0xaa, 0x65, 0x81, 0x3c, 0x94, 0x07 };
        ggml_backend_t backend = malloc(sizeof(*backend));
        GGML_ASSERT(backend);
        *backend = (struct ggml_backend) {
            .guid    = &guid,
            .iface   = ane_backend_i,
            .device  = dev,
            .context = ctx,
        };
        if (!ctx->queue || !ctx->ready || !ctx->done || !ctx->cast || !ctx->convert || ![(id) ctx->ready respondsToSelector:@selector(IOSurfaceSharedEvent)]) {
            GGML_LOG_ERROR("ANE: initialization failed: %s\n", error.description.UTF8String);
            ane_free(backend);
            return NULL;
        }
        return backend;
    }
}

static const char * ane_dev_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "ANE";
}

static const char * ane_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Apple Neural Engine (F16)";
}

static enum ggml_backend_dev_type ane_dev_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ane_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *total = ane_device.recommendedMaxWorkingSetSize;
    *free = *total > ane_device.currentAllocatedSize ? *total - ane_device.currentAllocatedSize : 0;
}

static void ane_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name = ane_dev_name(dev);
    props->description = ane_description(dev);
    props->type = ane_dev_type(dev);
    ane_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async = true;
    props->caps.events = true;
}

static ggml_backend_buffer_type_t ane_get_buft(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return &ane_buft;
}

static bool ane_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ane_is_buft(buft);
}

static bool ane_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && ane_supports_op(dev, op);
}

static ggml_backend_event_t ane_event_new(ggml_backend_dev_t dev) {
    struct ane_event * ev = calloc(1, sizeof(*ev));
    GGML_ASSERT(ev);
    ev->event = [ane_device newSharedEvent];
    if (!ev->event) {
        free(ev);
        return NULL;
    }
    ggml_backend_event_t event = malloc(sizeof(*event));
    GGML_ASSERT(event);
    *event = (struct ggml_backend_event) {
        .device  = dev,
        .context = ev,
    };
    return event;
}

static void ane_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    struct ane_event * ev = event->context;
    [ev->event release];
    free(ev);
    free(event);
}

static void ane_event_sync(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    struct ane_event * ev = event->context;
    GGML_ASSERT([ev->event waitUntilSignaledValue:ev->value timeoutMS:60000]);
}

static struct ggml_backend_device_i ane_device_i = {
    .get_name          = ane_dev_name,
    .get_description   = ane_description,
    .get_memory        = ane_memory,
    .get_type          = ane_dev_type,
    .get_props         = ane_props,
    .init_backend      = ane_init,
    .get_buffer_type   = ane_get_buft,
    .supports_op       = ane_supports_op,
    .supports_buft     = ane_supports_buft,
    .offload_op        = ane_offload_op,
    .event_new         = ane_event_new,
    .event_free        = ane_event_free,
    .event_synchronize = ane_event_sync,
};

static const char * ane_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "ANE";
}

static size_t ane_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        void * handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
        if (handle && NSClassFromString(@"_ANEInMemoryModel") && NSClassFromString(@"_ANEInMemoryModelDescriptor") &&
            NSClassFromString(@"_ANEIOSurfaceObject") && NSClassFromString(@"_ANERequest") && NSClassFromString(@"_ANESharedEvents") &&
            NSClassFromString(@"_ANESharedWaitEvent") && NSClassFromString(@"_ANESharedSignalEvent")) {
            ane_device = MTLCreateSystemDefaultDevice();
        }
    });
    return ane_device ? 1 : 0;
}

static ggml_backend_dev_t ane_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static struct ggml_backend_device dev;
    dev.iface = ane_device_i;
    dev.reg = reg;
    ane_buft.device = &dev;
    return &dev;
}

static void * ane_proc(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (!strcmp(name, "ggml_backend_ane_get_buffer_api")) {
        return (void *) ane_get_buffer_api;
    }
    return NULL;
}

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    static struct ggml_backend_reg reg = {
        .api_version = GGML_BACKEND_API_VERSION,
        .iface       = { ane_reg_name, ane_device_count, ane_get_device, ane_proc },
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)
