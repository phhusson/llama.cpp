#import "ggml-metal-pq2.h"
#import "ggml-metal-ane.h"
#import "ggml-backend-impl.h"
#import "ggml-impl.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <sys/mman.h>
#include <unistd.h>

@interface GGMLPQ2Planes : NSObject {
@public
    IOSurfaceRef q, d;
    id<MTLBuffer> mq, md;
    int64_t k, rows, ds;
}
@end
@implementation GGMLPQ2Planes
- (void)dealloc {
    [mq release]; [md release];
    if (q) { CFRelease(q); }
    if (d) { CFRelease(d); }
    [super dealloc];
}
@end

struct pq2_buffer {
    void * address;
    size_t size;
    NSMutableArray * tensors;
    int64_t repack_us;
    size_t packed_bytes;
};

static IOSurfaceRef pq2_surface(size_t bytes) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    bytes = (bytes + page - 1) & ~(page - 1);
    return IOSurfaceCreate((CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static void pq2_free(ggml_backend_buffer_t buffer) {
    struct pq2_buffer * ctx = buffer->context;
    if (ctx->packed_bytes && getenv("GGML_METAL_ANE_PROFILE")) {
        GGML_LOG_INFO("PQ2 planar load total: %lu tensors, %.2f MiB, %.3f ms conversion\n",
            (unsigned long)ctx->tensors.count, ctx->packed_bytes/1048576.0, ctx->repack_us/1000.0);
    }
    [ctx->tensors release];
    if (ctx->size) { munmap(ctx->address, ctx->size); }
    free(ctx);
}

static ggml_backend_buffer_t pq2_alloc(ggml_backend_buffer_type_t buft, size_t size);

bool ggml_metal_buffer_is_pq2_planar(ggml_backend_buffer_t buffer) {
    return buffer && buffer->buft->iface.alloc_buffer == pq2_alloc;
}

bool ggml_metal_pq2_is_planar(const struct ggml_tensor * tensor) {
    return tensor && ggml_metal_buffer_is_pq2_planar(tensor->buffer);
}

bool ggml_metal_pq2_supports_op(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[0];
    return op->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_PQ2_0 &&
        w->ne[0] % 128 == 0 && w->ne[1] % 64 == 0 && w->ne[1] >= 2048 && w->ne[1] <= 32768 &&
        w->ne[2] == 1 && w->ne[3] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
        op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

static void * pq2_base(ggml_backend_buffer_t buffer) {
    return ((struct pq2_buffer *)buffer->context)->address;
}

static enum ggml_status pq2_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_PQ2_0 && !tensor->view_src);
    struct pq2_buffer * ctx = buffer->context;
    GGMLPQ2Planes * p = [GGMLPQ2Planes new];
    p->k = tensor->ne[0]; p->rows = tensor->ne[1]; p->ds = ((p->k/128+31)/32)*32;
    p->q = pq2_surface(p->rows*p->k/4);
    p->d = pq2_surface(p->rows*p->ds*2);
    GGML_ASSERT(p->q && p->d);
    id<MTLDevice> device = ggml_metal_device_get_obj(buffer->buft->device->context);
    p->mq = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(p->q) length:IOSurfaceGetAllocSize(p->q)
        options:MTLResourceStorageModeShared deallocator:nil];
    p->md = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(p->d) length:IOSurfaceGetAllocSize(p->d)
        options:MTLResourceStorageModeShared deallocator:nil];
    GGML_ASSERT(p->mq && p->md);
    memset(IOSurfaceGetBaseAddress(p->d), 0, IOSurfaceGetAllocSize(p->d));
    tensor->extra = p;
    [ctx->tensors addObject:p]; [p release];
    return GGML_STATUS_SUCCESS;
}

static void pq2_set(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    struct pq2_buffer * ctx = buffer->context;
    int64_t started = ggml_time_us();
    GGML_ASSERT(offset % 34 == 0 && size % 34 == 0);
    GGMLPQ2Planes * p = tensor->extra;
    const uint8_t * src = data;
    uint8_t * q = IOSurfaceGetBaseAddress(p->q);
    uint16_t * d = IOSurfaceGetBaseAddress(p->d);
    for (size_t block = offset/34; block < (offset+size)/34; ++block, src += 34) {
        size_t row = block/(p->k/128), ib = block%(p->k/128);
        memcpy(d + row*p->ds + ib, src, 2);
        for (int i = 0; i < 32; ++i) {
            int at = 2+i/4, shift = 2*(i%4);
            q[block*32+i] = ((src[at]>>shift)&3) | (((src[at+8]>>shift)&3)<<2) |
                (((src[at+16]>>shift)&3)<<4) | (((src[at+24]>>shift)&3)<<6);
        }
    }
    ctx->repack_us += ggml_time_us()-started; ctx->packed_bytes += size;
    if (getenv("GGML_METAL_ANE_PROFILE")) { GGML_LOG_INFO("PQ2 planar load: %s, %zu bytes\n", tensor->name, size); }
}

static void pq2_get(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    (void)buffer;
    GGMLPQ2Planes * p = tensor->extra;
    const uint8_t * q = IOSurfaceGetBaseAddress(p->q);
    const uint16_t * d = IOSurfaceGetBaseAddress(p->d);
    uint8_t * dst = data;
    for (size_t first = offset; first < offset+size;) {
        size_t block = first/34, row = block/(p->k/128), ib = block%(p->k/128);
        uint8_t packed[34];
        memcpy(packed, d + row*p->ds + ib, 2);
        for (int i = 0; i < 32; ++i) {
            unsigned v = 0;
            for (int l = 0; l < 4; ++l) {
                int j = i*4+l;
                v |= ((q[block*32+j%32] >> (2*(j/32))) & 3) << (2*l);
            }
            packed[i+2] = v;
        }
        size_t count = MIN(34-first%34, offset+size-first);
        memcpy(dst, packed+first%34, count);
        first += count; dst += count;
    }
}

static void pq2_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct pq2_buffer * ctx = buffer->context;
    for (GGMLPQ2Planes * p in ctx->tensors) {
        memset(IOSurfaceGetBaseAddress(p->q), value, IOSurfaceGetAllocSize(p->q));
        memset(IOSurfaceGetBaseAddress(p->d), value, IOSurfaceGetAllocSize(p->d));
    }
}

static const char * pq2_name(ggml_backend_buffer_type_t buft) { (void)buft; return "Metal_PQ2_PLANAR"; }
static size_t pq2_alignment(ggml_backend_buffer_type_t buft) { (void)buft; return 64; }
static size_t pq2_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    (void)buft;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t q = t->ne[1]*t->ne[0]/4, d = t->ne[1]*((t->ne[0]/128+31)/32)*64;
    return ((q+page-1)&~(page-1)) + ((d+page-1)&~(page-1));
}
static ggml_backend_buffer_t pq2_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    struct pq2_buffer * ctx = calloc(1, sizeof(*ctx));
    ctx->size = size;
    // ggml uses these addresses as tensor identities. Only the IOSurfaces hold data.
    ctx->address = size ? mmap(NULL, size, PROT_NONE, MAP_PRIVATE|MAP_ANON, -1, 0) : (void *)0x1000;
    if (ctx->address == MAP_FAILED) { free(ctx); return NULL; }
    ctx->tensors = [NSMutableArray new];
    struct ggml_backend_buffer_i iface = {
        .free_buffer=pq2_free, .get_base=pq2_base, .init_tensor=pq2_init_tensor,
        .set_tensor=pq2_set, .get_tensor=pq2_get, .clear=pq2_clear,
    };
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

ggml_backend_buffer_type_t ggml_metal_pq2_planar_buffer_type(ggml_backend_dev_t dev) {
    if (!getenv("GGML_METAL_ANE_PLANAR") && !ggml_metal_ane_enabled()) { return NULL; }
    static struct ggml_backend_buffer_type types[16];
    int index = ggml_metal_device_get_props(dev->context)->device;
    GGML_ASSERT(index >= 0 && index < 16);
    types[index] = (struct ggml_backend_buffer_type){
        .iface={.get_name=pq2_name, .alloc_buffer=pq2_alloc, .get_alignment=pq2_alignment, .get_alloc_size=pq2_alloc_size},
        .device=dev,
    };
    return &types[index];
}

struct ggml_metal_buffer_id ggml_metal_pq2_buffer(const struct ggml_tensor * tensor, bool scales) {
    GGMLPQ2Planes * p = tensor->extra;
    GGML_ASSERT(p);
    return (struct ggml_metal_buffer_id){scales ? p->md : p->mq, 0};
}
void * ggml_metal_pq2_surface(const struct ggml_tensor * tensor, bool scales) {
    GGMLPQ2Planes * p = tensor->extra;
    return scales ? p->d : p->q;
}
float ggml_metal_pq2_value(const struct ggml_tensor * tensor, int64_t row, int64_t col) {
    GGMLPQ2Planes * p = tensor->extra;
    const uint8_t * q = IOSurfaceGetBaseAddress(p->q);
    const ggml_fp16_t * d = IOSurfaceGetBaseAddress(p->d);
    int digit = (q[row*p->k/4+(col/128)*32+col%32] >> (2*((col%128)/32))) & 3;
    return ggml_fp16_to_fp32(d[row*p->ds+col/128]) * (digit-1);
}
