#import "ggml-metal-ane.h"
#import "ggml-metal-pq2.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <math.h>
#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <unistd.h>

#ifdef GGML_METAL_ANE
#include "ggml-metal-ane-data.h"
#else
#define GGML_METAL_ANE_DEFAULT_FRACTION .65
#endif

@interface NSObject (GGMLPrivateANE)
+ (id)modelWithMILText:(NSData *)mil weights:(NSDictionary *)weights optionsPlist:(id)options;
+ (id)inMemoryModelWithDescriptor:(id)descriptor;
- (NSString *)hexStringIdentifier;
- (BOOL)compileWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)loadWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)unloadWithQoS:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithQoS:(unsigned int)qos options:(NSDictionary *)options request:(id)request error:(NSError **)error;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices
               outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices
         weightsBuffer:(id)weights perfStats:(id)stats procedureIndex:(NSNumber *)index sharedEvents:(id)events;
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(uint32_t)index eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
- (id)IOSurfaceSharedEvent;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end

enum { ANE_ACT, ANE_Y, ANE_IO_COUNT };

@interface GGMLANEProgram : NSObject {
@public
    id model;
    IOSurfaceRef surfaces[ANE_IO_COUNT];
    id<MTLBuffer> buffers[ANE_IO_COUNT];
    NSArray * inputs;
    NSArray * outputs;
    NSString * temporary;
}
@end
@implementation GGMLANEProgram
- (void)dealloc {
    if (model) { [model unloadWithQoS:21 error:NULL]; }
    [model release];
    [inputs release]; [outputs release];
    for (int i = 0; i < ANE_IO_COUNT; ++i) {
        [buffers[i] release];
        if (surfaces[i]) { CFRelease(surfaces[i]); }
    }
    if (temporary) { [[NSFileManager defaultManager] removeItemAtPath:temporary error:NULL]; }
    [temporary release];
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

struct ggml_ane {
    NSMutableDictionary * programs;
    NSMutableArray * requests;
    id<MTLSharedEvent> input_ready;
    id<MTLSharedEvent> output_ready;
    struct ggml_metal_pipeline_with_params cast;
    struct ggml_metal_pipeline_with_params join;
    dispatch_group_t pending;
    dispatch_semaphore_t credits;
    atomic_bool failed;
    uint64_t sequence;
    bool profile;
    double fraction;
    NSMutableDictionary * mps_ops;
    struct ggml_metal_pipeline_with_params expand;
    id<MTLBuffer> scratch_w, scratch_a;
};

static NSString * ane_key(const struct ggml_tensor * op, int64_t rows) {
    return [NSString stringWithFormat:@"pq2_k%lld_r%lld_t%lld", op->src[0]->ne[0], rows, op->ne[1]];
}

static void ane_fail(struct ggml_ane * ctx, NSString * error) {
    GGML_LOG_ERROR("ANE events: %s\n", error.UTF8String);
    atomic_store(&ctx->failed, true);
    // Release queued GPU waits on failure, then return an error for the graph.
    ctx->input_ready.signaledValue = UINT64_MAX;
    ctx->output_ready.signaledValue = UINT64_MAX;
}

bool ggml_metal_ane_enabled(void) {
#ifdef GGML_METAL_ANE
    const char * enabled = getenv("GGML_METAL_ANE");
    if (enabled && strcmp(enabled, "0") != 0 && strcmp(enabled, "1") != 0) {
        GGML_ABORT("GGML_METAL_ANE now accepts 0 or 1. Unset it to use the embedded ANE kernels; no directory is needed.");
    }
    return !enabled || strcmp(enabled, "0") != 0;
#else
    return false;
#endif
}

void * ggml_metal_ane_init(void) {
    if (!ggml_metal_ane_enabled()) { return NULL; }
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        GGML_ABORT("ANE events: AppleNeuralEngine unavailable");
    }
    struct ggml_ane * ctx = calloc(1, sizeof(*ctx));
    ctx->programs = [NSMutableDictionary new];
    ctx->requests = [NSMutableArray new];
    ctx->pending = dispatch_group_create();
    ctx->credits = dispatch_semaphore_create(8);
    ctx->profile = getenv("GGML_METAL_ANE_PROFILE") != NULL;
    ctx->fraction = getenv("GGML_METAL_ANE_FRACTION") ? atof(getenv("GGML_METAL_ANE_FRACTION")) : GGML_METAL_ANE_DEFAULT_FRACTION;
    ctx->mps_ops = [NSMutableDictionary new];
    atomic_init(&ctx->failed, false);
    return ctx;
}

void ggml_metal_ane_free(void * opaque) {
    struct ggml_ane * ctx = opaque;
    if (!ctx) { return; }
    ggml_metal_ane_finish(ctx);
    [ctx->requests release]; [ctx->programs release];
    [ctx->input_ready release]; [ctx->output_ready release];
    [ctx->mps_ops release];
    [ctx->scratch_w release]; [ctx->scratch_a release];
    dispatch_release(ctx->pending); dispatch_release(ctx->credits);
    free(ctx);
}

static bool ane_supports_op(const struct ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && op->src[0]->type == GGML_TYPE_PQ2_0 &&
        ggml_metal_pq2_is_planar(op->src[0]) &&
        op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->ne[1] >= 128 &&
        op->ne[0] % 64 == 0 && op->ne[0] >= 2048 && op->ne[0] <= 32768 &&
        op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

bool ggml_metal_ane_mps_only(void * opaque, const struct ggml_tensor * op) {
    struct ggml_ane * ctx = opaque;
    return ctx && ctx->fraction == 0 && !getenv("GGML_METAL_ANE_MPS_DISABLE") && ane_supports_op(op);
}

int64_t ggml_metal_ane_rows(void * opaque, const struct ggml_tensor * op) {
    struct ggml_ane * ctx = opaque;
    if (!ctx || !ane_supports_op(op)) { return 0; }
    int rows = (int)(op->ne[0]*ctx->fraction/256)*256;
    if (rows <= 0 || rows > op->ne[0]) { return 0; }
#ifdef GGML_METAL_ANE
    char key[96];
    snprintf(key, sizeof(key), "pq2_k%lld_r%d_t%lld", op->src[0]->ne[0], rows, op->ne[1]);
    for (size_t i = 0; i < sizeof(ggml_metal_ane_resources)/sizeof(ggml_metal_ane_resources[0]); ++i) {
        const struct ggml_metal_ane_resource * resource = &ggml_metal_ane_resources[i];
        if (strcmp(resource->program, key) == 0 && strcmp(resource->path, "model.mil") == 0) { return rows; }
    }
#endif
    return 0;
}

static NSData * ane_program_data(NSString * key, NSMutableDictionary * weights) {
    NSData * mil = nil;
#ifdef GGML_METAL_ANE
    for (size_t i = 0; i < sizeof(ggml_metal_ane_resources)/sizeof(ggml_metal_ane_resources[0]); ++i) {
        const struct ggml_metal_ane_resource * resource = &ggml_metal_ane_resources[i];
        if (strcmp(resource->program, key.UTF8String) != 0) { continue; }
        NSData * data = [NSData dataWithBytes:resource->data length:resource->size];
        if (strcmp(resource->path, "model.mil") == 0) {
            mil = data;
        } else {
            NSString * path = [@"@model_path/" stringByAppendingString:[NSString stringWithUTF8String:resource->path]];
            weights[path] = @{@"offset": @0, @"data": data};
        }
    }
#else
    (void)key; (void)weights;
#endif
    return mil;
}

bool ggml_metal_ane_load(void * opaque, const struct ggml_tensor * op, int64_t rows) {
    struct ggml_ane * ctx = opaque;
    if (atomic_load(&ctx->failed)) { return false; }
    NSString * key = ane_key(op, rows);
    id known = ctx->programs[key];
    if (known) { return known != [NSNull null]; }
    NSMutableDictionary * weights = [NSMutableDictionary dictionary];
    NSData * mil = ane_program_data(key, weights);
    if (!mil) {
        GGML_LOG_WARN("ANE events: no embedded kernel for %s; use Metal\n", key.UTF8String);
        ctx->programs[key] = [NSNull null];
        return false;
    }
    GGMLANEProgram * p = [[GGMLANEProgram new] autorelease];
    id desc = [NSClassFromString(@"_ANEInMemoryModelDescriptor") modelWithMILText:mil weights:weights optionsPlist:nil];
    p->model = [[NSClassFromString(@"_ANEInMemoryModel") inMemoryModelWithDescriptor:desc] retain];
    if (!p->model) { ane_fail(ctx, @"model descriptor rejected"); return false; }
    p->temporary = [[NSTemporaryDirectory() stringByAppendingPathComponent:[p->model hexStringIdentifier]] retain];
    NSFileManager * fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:p->temporary withIntermediateDirectories:YES attributes:nil error:NULL];
    [mil writeToFile:[p->temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString * name in weights) {
        NSString * dest = [p->temporary stringByAppendingPathComponent:[name substringFromIndex:12]];
        [fm createDirectoryAtPath:[dest stringByDeletingLastPathComponent] withIntermediateDirectories:YES attributes:nil error:NULL];
        [weights[name][@"data"] writeToFile:dest atomically:YES];
    }
    NSError * error = nil;
    if (![p->model compileWithQoS:21 options:@{} error:&error] || ![p->model loadWithQoS:21 options:@{} error:&error]) {
        ane_fail(ctx, error ? error.description : @"compile/load failed");
        ctx->programs[key] = [NSNull null];
        return false;
    }
    const int64_t k = op->src[0]->ne[0], t = op->ne[1];
    const size_t sizes[] = {t*k*2, t*rows*2};
    NSMutableArray * inputs = [NSMutableArray array], * outputs = [NSMutableArray array];
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    for (int i = 0; i < ANE_IO_COUNT; ++i) {
        size_t bytes = (sizes[i]+page-1) & ~(page-1);
        p->surfaces[i] = IOSurfaceCreate((CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1, (id)kIOSurfaceBytesPerElement: @1,
            (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
        GGML_ASSERT(p->surfaces[i]);
        memset(IOSurfaceGetBaseAddress(p->surfaces[i]), 0, bytes);
        id wrapped = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:p->surfaces[i]];
        [(i == ANE_Y ? outputs : inputs) addObject:wrapped];
    }
    p->inputs = [inputs copy]; p->outputs = [outputs copy];
    ctx->programs[key] = p;
    GGML_LOG_INFO("ANE events: loaded %s\n", key.UTF8String);
    return true;
}

bool ggml_metal_ane_begin(void * opaque, ggml_metal_device_t dev) {
    struct ggml_ane * ctx = opaque;
    if (atomic_load(&ctx->failed)) { return false; }
    if (ctx->cast.pipeline) { return true; }
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    ctx->input_ready = [device newSharedEvent]; ctx->output_ready = [device newSharedEvent];
    GGML_ASSERT([(id)ctx->input_ready respondsToSelector:@selector(IOSurfaceSharedEvent)]);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    ctx->cast = ggml_metal_library_compile_pipeline(lib, "kernel_ane_cast", "kernel_ane_cast", NULL);
    ctx->join = ggml_metal_library_compile_pipeline(lib, "kernel_ane_join", "kernel_ane_join", NULL);
    if (!ctx->cast.pipeline || !ctx->join.pipeline) { ane_fail(ctx, @"Metal helper pipeline compilation failed"); return false; }
    return true;
}

static struct ggml_metal_buffer_id ane_buffer(const struct ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    return ggml_metal_buffer_get_id(buffer->context, t);
}

static void ane_watch(struct ggml_ane * ctx, id<MTLCommandBuffer> cb, bool credit) {
    dispatch_group_enter(ctx->pending);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status != MTLCommandBufferStatusCompleted) { ane_fail(ctx, done.error ? done.error.description : @"Metal command failed"); }
        if (credit) { dispatch_semaphore_signal(ctx->credits); }
        dispatch_group_leave(ctx->pending);
    }];
}

bool ggml_metal_ane_prepare(void * opaque, ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb_raw, const struct ggml_tensor * op, int64_t rows) {
    struct ggml_ane * ctx = opaque;
    if (dispatch_semaphore_wait(ctx->credits, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC))) {
        ane_fail(ctx, @"submission queue timeout"); return false;
    }
    if (atomic_load(&ctx->failed)) { dispatch_semaphore_signal(ctx->credits); return false; }
    ++ctx->sequence;
    GGMLANEProgram * p = ctx->programs[ane_key(op, rows)];
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    for (int i = 0; i < ANE_IO_COUNT; ++i) {
        if (!p->buffers[i]) {
            p->buffers[i] = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(p->surfaces[i])
                length:IOSurfaceGetAllocSize(p->surfaces[i]) options:MTLResourceStorageModeShared deallocator:nil];
            GGML_ASSERT(p->buffers[i]);
        }
    }
    const int64_t k = op->src[0]->ne[0], t = op->ne[1];
    struct ggml_metal_buffer_id a = ane_buffer(op->src[1]);
    id<MTLCommandBuffer> cb = cb_raw;
    ggml_metal_encoder_t enc = ggml_metal_encoder_init(cb, false);
    ggml_metal_encoder_set_pipeline(enc, ctx->cast);
    ggml_metal_encoder_set_buffer(enc, a, 0);
    ggml_metal_encoder_set_buffer(enc, (struct ggml_metal_buffer_id){p->buffers[ANE_ACT], 0}, 1);
    ggml_metal_encoder_dispatch_threads(enc, t*k/4, 1, 1, 256, 1, 1);
    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);
    [cb encodeSignalEvent:ctx->input_ready value:ctx->sequence];
    ane_watch(ctx, cb, false);
    return true;
}

bool ggml_metal_ane_submit(void * opaque, const struct ggml_tensor * op, int64_t rows) {
    struct ggml_ane * ctx = opaque;
    GGMLANEProgram * p = ctx->programs[ane_key(op, rows)];
    uint64_t seq = ctx->sequence;
    id wait = [NSClassFromString(@"_ANESharedWaitEvent") waitEventWithValue:seq sharedEvent:[(id)ctx->input_ready IOSurfaceSharedEvent]];
    id signal = [NSClassFromString(@"_ANESharedSignalEvent") signalEventWithValue:seq symbolIndex:0
        eventType:0 sharedEvent:[(id)ctx->output_ready IOSurfaceSharedEvent]];
    id events = [NSClassFromString(@"_ANESharedEvents") sharedEventsWithSignalEvents:@[signal] waitEvents:@[wait]];
    id d = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:ggml_metal_pq2_surface(op->src[0], true)];
    id q = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:ggml_metal_pq2_surface(op->src[0], false)];
    // The compiled MIL signature is ordered act, d, q.
    NSArray * inputs = @[p->inputs[0], d, q];
    id request = [NSClassFromString(@"_ANERequest") requestWithInputs:inputs inputIndices:@[@0,@1,@2]
        outputs:p->outputs outputIndices:@[@0] weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:events];
    if (!request) { ane_fail(ctx, @"request rejected"); return false; }
    GGMLANETicket * ticket = [[GGMLANETicket new] autorelease];
    atomic_init(&ticket->completed, false);
    NSString * name = [NSString stringWithUTF8String:op->name];
    dispatch_group_enter(ctx->pending);
    [request setCompletionHandler:^(BOOL success, NSError * error) {
        if (!success) { ane_fail(ctx, error ? error.description : @"ANE execution failed"); }
        if (ctx->profile) { GGML_LOG_INFO("ANE events complete %s: value %llu, ok %d\n", name.UTF8String, seq, success); }
        if (!atomic_exchange(&ticket->completed, true)) { dispatch_group_leave(ctx->pending); }
    }];
    [ctx->requests addObject:request];
    NSError * error = nil;
    BOOL ok = [p->model evaluateWithQoS:21 options:@{} request:request error:&error];
    if (!ok) {
        ane_fail(ctx, error ? error.description : @"ANE submission failed");
        if (!atomic_exchange(&ticket->completed, true)) { dispatch_group_leave(ctx->pending); }
    }
    return ok;
}

void ggml_metal_ane_join(void * opaque, ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb_raw, const struct ggml_tensor * op, int64_t rows) {
    (void)dev;
    struct ggml_ane * ctx = opaque;
    GGMLANEProgram * p = ctx->programs[ane_key(op, rows)];
    id<MTLCommandBuffer> cb = cb_raw;
    [cb encodeWaitForEvent:ctx->output_ready value:ctx->sequence];
    struct ggml_metal_buffer_id y = ane_buffer(op);
    uint32_t s[] = {(uint32_t)op->ne[0], (uint32_t)rows};
    ggml_metal_encoder_t enc = ggml_metal_encoder_init(cb, false);
    ggml_metal_encoder_set_pipeline(enc, ctx->join);
    ggml_metal_encoder_set_buffer(enc, (struct ggml_metal_buffer_id){p->buffers[ANE_Y], 0}, 0);
    ggml_metal_encoder_set_buffer(enc, y, 1);
    ggml_metal_encoder_set_bytes(enc, s, sizeof(s), 2);
    ggml_metal_encoder_dispatch_threads(enc, op->ne[1]*rows/4, 1, 1, 256, 1, 1);
    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);
    ane_watch(ctx, cb, true);
}

bool ggml_metal_ane_finish(void * opaque) {
    struct ggml_ane * ctx = opaque;
    if (dispatch_group_wait(ctx->pending, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC))) {
        ane_fail(ctx, @"completion timeout");
        if (dispatch_group_wait(ctx->pending, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC))) {
            GGML_ABORT("ANE events: failed to drain requests");
        }
    }
    [ctx->requests removeAllObjects];
    return !atomic_load(&ctx->failed);
}

void ggml_metal_ane_check(void * opaque, const struct ggml_tensor * op, int64_t rows) {
    (void)opaque;
    if (!getenv("GGML_METAL_ANE_VALIDATE")) { return; }
    const int64_t k = op->src[0]->ne[0], n = op->ne[0], t = op->ne[1];
    const float * dst = op->data;
    for (int part = 0; part < 2; ++part) {
        int64_t count = part ? rows : n - rows;
        if (!count) { continue; }
        double e2 = 0, r2 = 0, maxe = 0, maxr = 0;
        for (int s = 0; s < 64; ++s) {
            int64_t row = (s*997) % count, token = (s*73) % t;
            if (s == 1) { row = count - 1; }
            if (!part) { row += rows; }
            const float * a = (const float *) op->src[1]->data + token*k;
            double ref = 0;
            for (int64_t j = 0; j < k; ++j) {
                float weight = ggml_metal_pq2_value(op->src[0], row, j);
                ref += (double)ggml_fp16_to_fp32(ggml_fp32_to_fp16(a[j]))*(double)weight;
            }
            double val = (double) dst[token*n + row], e = val-ref;
            GGML_ASSERT(isfinite(val));
            e2 += e*e; r2 += ref*ref; maxe = fmax(maxe, fabs(e)); maxr = fmax(maxr, fabs(ref));
        }
        double rel = sqrt(e2 / fmax(r2, 1e-30));
        GGML_LOG_INFO("ANE PQ2 check %s %s: relative RMS %.5g, max error %.5g, max ref %.5g\n", op->name, part ? "ANE" : "Metal", rel, maxe, maxr);
        GGML_ASSERT(rel < .02 && maxe < .04*maxr + .001);
    }
}

bool ggml_metal_ane_mps(void * opaque, ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb_raw, const struct ggml_tensor * op, int64_t rows) {
    struct ggml_ane * ctx = opaque;
    if (getenv("GGML_METAL_ANE_MPS_DISABLE")) { return false; }
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandBuffer> cb = cb_raw;
    if (!ctx->expand.pipeline) {
        ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
        ctx->expand = ggml_metal_library_compile_pipeline(lib, "kernel_pq2_planar_expand", "kernel_pq2_planar_expand", NULL);
        GGML_ASSERT(ctx->expand.pipeline && ctx->cast.pipeline);
    }
    const int64_t k = op->src[0]->ne[0], t = op->ne[1];
    const int64_t first = op->ne[0]-rows;
    size_t nw = rows*k*2, na = t*k*2;
    if (ctx->scratch_w.length < nw) {
        [ctx->scratch_w release];
        ctx->scratch_w = [device newBufferWithLength:nw options:MTLResourceStorageModePrivate];
    }
    if (ctx->scratch_a.length < na) {
        [ctx->scratch_a release];
        ctx->scratch_a = [device newBufferWithLength:na options:MTLResourceStorageModePrivate];
    }
    GGML_ASSERT(ctx->scratch_w && ctx->scratch_a);
    struct ggml_metal_buffer_id w = ane_buffer(op->src[0]), a = ane_buffer(op->src[1]), y = ane_buffer(op);
    ggml_metal_encoder_t enc = ggml_metal_encoder_init(cb, false);
    ggml_metal_encoder_set_pipeline(enc, ctx->expand);
    w.offs += first*(k/4);
    ggml_metal_encoder_set_buffer(enc, w, 0);
    ggml_metal_encoder_set_buffer(enc, (struct ggml_metal_buffer_id){ctx->scratch_w, 0}, 1);
    struct ggml_metal_buffer_id d = ggml_metal_pq2_buffer(op->src[0], true);
    uint32_t shape[] = {(uint32_t)k, (uint32_t)((k/128+31)/32*32)};
    d.offs += first*shape[1]*2;
    ggml_metal_encoder_set_buffer(enc, d, 2);
    ggml_metal_encoder_set_bytes(enc, shape, sizeof(shape), 3);
    ggml_metal_encoder_dispatch_threads(enc, rows*k/4, 1, 1, 256, 1, 1);
    ggml_metal_encoder_set_pipeline(enc, ctx->cast);
    ggml_metal_encoder_set_buffer(enc, a, 0);
    ggml_metal_encoder_set_buffer(enc, (struct ggml_metal_buffer_id){ctx->scratch_a, 0}, 1);
    ggml_metal_encoder_dispatch_threads(enc, t*k/4, 1, 1, 256, 1, 1);
    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);
    NSString * key = [NSString stringWithFormat:@"%lld_%lld_%lld", k, rows, t];
    MPSMatrixMultiplication * mm = ctx->mps_ops[key];
    if (!mm) {
        mm = [[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:t resultColumns:rows interiorColumns:k alpha:1 beta:0] autorelease];
        ctx->mps_ops[key] = mm;
    }
    MPSMatrix * mw = [[[MPSMatrix alloc] initWithBuffer:ctx->scratch_w descriptor:
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:k rowBytes:k*2 dataType:MPSDataTypeFloat16]] autorelease];
    MPSMatrix * ma = [[[MPSMatrix alloc] initWithBuffer:ctx->scratch_a descriptor:
        [MPSMatrixDescriptor matrixDescriptorWithRows:t columns:k rowBytes:k*2 dataType:MPSDataTypeFloat16]] autorelease];
    MPSMatrix * my = [[[MPSMatrix alloc] initWithBuffer:y.metal offset:y.offs+first*4 descriptor:
        [MPSMatrixDescriptor matrixDescriptorWithRows:t columns:rows rowBytes:op->nb[1] dataType:MPSDataTypeFloat32]] autorelease];
    [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mw resultMatrix:my];
    if (rows == op->ne[0]) { ane_watch(ctx, cb, false); }
    return true;
}
