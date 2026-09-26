#include "ggml-fakenpu.h"
#include "ggml-dmabuf.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// A fake NPU used to exercise the scheduler and the dma-buf sharing path.
// - accepts only the DMA_BUF buffer type (no host pointer)
// - claims MUL_MAT and delegates the work to an internal CPU backend
// - runs the work on a worker thread with a configurable test latency
//
// Fences are not implemented yet.

//
// helpers
//

static int64_t ggml_backend_fakenpu_latency_us(void) {
    const char * env = getenv("GGML_FAKENPU_LATENCY_US");
    return env ? atoll(env) : 1000;
}

static void ggml_backend_fakenpu_collect_fds(const struct ggml_tensor * t, std::vector<int> & fds) {
    if (t == nullptr || t->buffer == nullptr ||
        strcmp(ggml_backend_buft_name(t->buffer->buft), GGML_DMABUF_NAME) != 0) {
        return;
    }
    const int fd = ggml_backend_dmabuf_buffer_fd(t->buffer);
    if (fd < 0) {
        return;
    }
    for (int f : fds) {
        if (f == fd) {
            return;
        }
    }
    fds.push_back(fd);
}

static void ggml_backend_fakenpu_collect_graph_fds(const struct ggml_cgraph * cgraph, std::vector<int> & fds) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        ggml_backend_fakenpu_collect_fds(node, fds);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_backend_fakenpu_collect_fds(node->src[j], fds);
        }
    }
    for (int i = 0; i < cgraph->n_leafs; i++) {
        ggml_backend_fakenpu_collect_fds(cgraph->leafs[i], fds);
    }
}

// bracket CPU access of the dma-bufs so the cache is coherent with the other devices
static void ggml_backend_fakenpu_buf_sync(const std::vector<int> & fds, bool start) {
#if defined(__linux__)
    for (int fd : fds) {
        struct dma_buf_sync sync = { (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | DMA_BUF_SYNC_RW };
        ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
    }
#else
    GGML_UNUSED(fds);
    GGML_UNUSED(start);
#endif
}

//
// backend (stream)
//

struct ggml_backend_fakenpu_context {
    ggml_backend_t cpu = nullptr;

    std::thread        worker;
    enum ggml_status   status = GGML_STATUS_SUCCESS;
};

static ggml_guid_t ggml_backend_fakenpu_guid(void) {
    static ggml_guid guid = { 0x9d, 0x4b, 0x2f, 0x76, 0x18, 0xac, 0x43, 0xe1, 0x8a, 0x5d, 0x71, 0x0c, 0x36, 0xf9, 0xb2, 0x40 };
    return &guid;
}

static const char * ggml_backend_fakenpu_get_name(ggml_backend_t backend) {
    return "NPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_fakenpu_free(ggml_backend_t backend) {
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;

    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }
    if (ctx->cpu) {
        ggml_backend_free(ctx->cpu);
    }

    delete ctx;
    delete backend;
}

static void ggml_backend_fakenpu_synchronize(ggml_backend_t backend) {
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;
    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }
}

static enum ggml_status ggml_backend_fakenpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;

    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }

    const int64_t latency_us = ggml_backend_fakenpu_latency_us();

    // no fences yet: run the work on a worker thread but wait for it before returning,
    // the scheduler does not synchronize an async backend before a zero-copy read
    ctx->status = GGML_STATUS_SUCCESS;
    ctx->worker = std::thread([ctx, cgraph, latency_us]() {
        std::vector<int> fds;
        ggml_backend_fakenpu_collect_graph_fds(cgraph, fds);

        if (latency_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
        }

        // TODO: wait for incoming fences

        ggml_backend_fakenpu_buf_sync(fds, true);
        ctx->status = ggml_backend_graph_compute(ctx->cpu, cgraph);
        ggml_backend_fakenpu_buf_sync(fds, false);

        // TODO: signal outgoing fences

        if (latency_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
        }
    });
    ctx->worker.join();

    return ctx->status;
}

static const struct ggml_backend_i ggml_backend_fakenpu_i = {
    /* .get_name          = */ ggml_backend_fakenpu_get_name,
    /* .free              = */ ggml_backend_fakenpu_free,
    /* .set_tensor_async  = */ NULL,
    /* .get_tensor_async  = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async  = */ NULL,
    /* .synchronize       = */ ggml_backend_fakenpu_synchronize,
    /* .graph_plan_create = */ NULL,
    /* .graph_plan_free   = */ NULL,
    /* .graph_plan_update = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute     = */ ggml_backend_fakenpu_graph_compute,
    /* .event_record      = */ NULL,
    /* .event_wait        = */ NULL,
    /* .graph_optimize    = */ NULL,
};

//
// device
//

static const char * ggml_backend_fakenpu_device_get_name(ggml_backend_dev_t dev) {
    return "NPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_fakenpu_device_get_description(ggml_backend_dev_t dev) {
    return "fake NPU (dma-buf only)";

    GGML_UNUSED(dev);
}

static void ggml_backend_fakenpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_fakenpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_fakenpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_fakenpu_device_get_name(dev);
    props->description = ggml_backend_fakenpu_device_get_description(dev);
    props->type        = ggml_backend_fakenpu_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_fakenpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_fakenpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_fakenpu_context * ctx = new ggml_backend_fakenpu_context;

    ctx->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (ctx->cpu == nullptr) {
        delete ctx;
        return nullptr;
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_fakenpu_guid(),
        /* .iface   = */ ggml_backend_fakenpu_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_fakenpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_dmabuf_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_fakenpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_MUL_MAT:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_fakenpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // only dma-buf, no host pointer
    return buft != nullptr && strcmp(ggml_backend_buft_name(buft), GGML_DMABUF_NAME) == 0;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_fakenpu_device_i = {
    /* .get_name             = */ ggml_backend_fakenpu_device_get_name,
    /* .get_description      = */ ggml_backend_fakenpu_device_get_description,
    /* .get_memory           = */ ggml_backend_fakenpu_device_get_memory,
    /* .get_type             = */ ggml_backend_fakenpu_device_get_type,
    /* .get_props            = */ ggml_backend_fakenpu_device_get_props,
    /* .init_backend         = */ ggml_backend_fakenpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fakenpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_fakenpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fakenpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// reg
//

static const char * ggml_backend_fakenpu_reg_get_name(ggml_backend_reg_t reg) {
    return "FAKENPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_fakenpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_fakenpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static struct ggml_backend_device device = {
        /* .iface   = */ ggml_backend_fakenpu_device_i,
        /* .reg     = */ nullptr, // set below
        /* .context = */ nullptr,
    };
    device.reg = reg;

    return &device;
}

static ggml_backend_buffer_type_t * ggml_backend_fakenpu_device_get_extra_bufts(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t bufts[] = { ggml_backend_dmabuf_buffer_type(), nullptr };
    GGML_UNUSED(dev);
    return bufts;
}

static void * ggml_backend_fakenpu_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_fakenpu_device_get_extra_bufts;
    }
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_fakenpu_reg_i = {
    /* .get_name         = */ ggml_backend_fakenpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_fakenpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fakenpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fakenpu_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_fakenpu_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fakenpu_reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fakenpu_reg)
