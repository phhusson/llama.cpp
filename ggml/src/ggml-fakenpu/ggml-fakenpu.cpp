#include "ggml-fakenpu.h"
#include "ggml-dmabuf.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// A fake NPU used to exercise the scheduler and the dma-buf sharing path.
// - accepts only the DMA_BUF buffer type (no host pointer)
// - claims MUL_MAT and delegates the work to an internal CPU backend
// - runs the work on a worker thread with a configurable test latency
// - exports/waits on sync_file fences via sw_sync (required, no fallback)

#if defined(__linux__)
struct sw_sync_create_fence_data {
    uint32_t value;
    char name[32];
    int32_t fence;
};

#define SW_SYNC_IOC_MAGIC 'W'
#define SW_SYNC_IOC_CREATE_FENCE _IOWR(SW_SYNC_IOC_MAGIC, 0, struct sw_sync_create_fence_data)
#define SW_SYNC_IOC_INC _IOW(SW_SYNC_IOC_MAGIC, 1, uint32_t)

#define FAKENPU_SW_SYNC_PATH "/sys/kernel/debug/sync/sw_sync"
#endif

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

struct ggml_backend_fakenpu_event {
    int fd = -1; // owned sync_file, -1 if none
};

struct ggml_backend_fakenpu_context {
    ggml_backend_t cpu = nullptr;

    std::thread        worker;
    enum ggml_status   status = GGML_STATUS_SUCCESS;

#if defined(__linux__)
    int              pending_fence = -1; // sync_file for the running graph, moved to the event on record
    std::vector<int> in_fences;          // sync_files to wait for before the next graph
#endif
    // stats
    uint64_t t_poll_us = 0;
    uint64_t t_bufsync_us = 0;
    uint64_t t_compute_us = 0;
    uint64_t n_graphs = 0;
};

static uint64_t fakenpu_now_us(void) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

//
// fences (sw_sync)
//

#if defined(__linux__)
// each graph gets its own sw_sync timeline, so the worker signals exactly the fence
// that was exported for that graph, independent of ordering with other workers.
// returns an owned sync_file fence (value 1) and outputs the timeline fd to signal it
static int ggml_backend_fakenpu_sw_sync_fence_new(int * out_tl_fd) {
    *out_tl_fd = -1;

    const int tl_fd = open(FAKENPU_SW_SYNC_PATH, O_RDWR);
    if (tl_fd < 0) {
        GGML_LOG_ERROR("%s: cannot open %s\n", __func__, FAKENPU_SW_SYNC_PATH);
        return -1;
    }

    struct sw_sync_create_fence_data data;
    memset(&data, 0, sizeof(data));
    data.value = 1;
    snprintf(data.name, sizeof(data.name), "npu");
    if (ioctl(tl_fd, SW_SYNC_IOC_CREATE_FENCE, &data) != 0) {
        GGML_LOG_ERROR("%s: sw_sync CREATE_FENCE failed\n", __func__);
        close(tl_fd);
        return -1;
    }

    *out_tl_fd = tl_fd;
    return data.fence;
}

static void ggml_backend_fakenpu_sw_sync_signal(int tl_fd) {
    uint32_t inc = 1;
    if (ioctl(tl_fd, SW_SYNC_IOC_INC, &inc) != 0) {
        GGML_LOG_ERROR("%s: sw_sync INC failed\n", __func__);
    }
}

static void ggml_backend_fakenpu_fence_poll(int fd) {
    struct pollfd pfd = { fd, POLLIN, 0 };
    while (poll(&pfd, 1, -1) < 0) {
        if (errno != EINTR) {
            GGML_LOG_ERROR("%s: poll() failed\n", __func__);
            return;
        }
    }
}
#endif

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

#if defined(__linux__)
    if (ctx->pending_fence >= 0) {
        close(ctx->pending_fence);
    }
    for (int fd : ctx->in_fences) {
        close(fd);
    }
#endif

    if (getenv("GGML_FAKENPU_STATS") != NULL) {
        fprintf(stderr, "[NPU stats] graphs=%llu poll=%.1fms bufsync=%.1fms compute=%.1fms\n",
            (unsigned long long) ctx->n_graphs, ctx->t_poll_us / 1000.0, ctx->t_bufsync_us / 1000.0, ctx->t_compute_us / 1000.0);
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

static enum ggml_status ggml_backend_fakenpu_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;

    if (ctx->worker.joinable()) {
        ctx->worker.join();
    }

    const int64_t latency_us = ggml_backend_fakenpu_latency_us();

#if defined(__linux__)
    // reserve a fence and its timeline for this graph, the worker signals exactly this one
    if (ctx->pending_fence >= 0) {
        close(ctx->pending_fence);
    }
    int tl_fd = -1;
    ctx->pending_fence = ggml_backend_fakenpu_sw_sync_fence_new(&tl_fd);
#endif

    ctx->status = GGML_STATUS_SUCCESS;
#if defined(__linux__)
    ctx->worker = std::thread([ctx, cgraph, latency_us, tl_fd]() {
#else
    ctx->worker = std::thread([ctx, cgraph, latency_us]() {
#endif
        const uint64_t t0 = fakenpu_now_us();
#if defined(__linux__)
        // wait for the producers before touching the shared buffers
        for (int fd : ctx->in_fences) {
            ggml_backend_fakenpu_fence_poll(fd);
            close(fd);
        }
        ctx->in_fences.clear();
#endif
        const uint64_t t1 = fakenpu_now_us();

        if (latency_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
        }

        std::vector<int> fds;
        ggml_backend_fakenpu_collect_graph_fds(cgraph, fds);

        const uint64_t tb0 = fakenpu_now_us();
        ggml_backend_fakenpu_buf_sync(fds, true);
        const uint64_t tb1 = fakenpu_now_us();
        ctx->status = ggml_backend_graph_compute(ctx->cpu, cgraph);
        const uint64_t t2 = fakenpu_now_us();
        ggml_backend_fakenpu_buf_sync(fds, false);
        const uint64_t tb2 = fakenpu_now_us();

        ctx->t_poll_us += t1 - t0;
        ctx->t_bufsync_us += (tb1 - tb0) + (tb2 - t2);
        ctx->t_compute_us += t2 - tb1;
        ctx->n_graphs++;

        if (latency_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
        }

#if defined(__linux__)
        if (tl_fd >= 0) {
            ggml_backend_fakenpu_sw_sync_signal(tl_fd);
            close(tl_fd);
        }
#endif
    });

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_fakenpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // async: the worker signals the graph fence when it is done
    return ggml_backend_fakenpu_graph_compute_async(backend, cgraph);
}

static void ggml_backend_fakenpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;
    ggml_backend_fakenpu_event * ev = (ggml_backend_fakenpu_event *) event->context;

#if defined(__linux__)
    if (ev->fd >= 0) {
        close(ev->fd);
    }
    // take ownership of the fence for the last submitted graph
    ev->fd = ctx->pending_fence;
    ctx->pending_fence = -1;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(ev);
#endif
}

static void ggml_backend_fakenpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_fakenpu_event * ev = (ggml_backend_fakenpu_event *) event->context;

#if defined(__linux__)
    if (ev->fd >= 0) {
        ggml_backend_fakenpu_fence_poll(ev->fd);
    }
#else
    GGML_UNUSED(ev);
#endif

    GGML_UNUSED(backend);
}

static ggml_backend_fence_t ggml_backend_fakenpu_event_export_fence(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_fakenpu_event * ev = (ggml_backend_fakenpu_event *) event->context;

#if defined(__linux__)
    if (ev->fd >= 0) {
        return ggml_backend_fence_init(dup(ev->fd));
    }
#else
    GGML_UNUSED(ev);
#endif

    GGML_UNUSED(backend);
    return NULL;
}

static bool ggml_backend_fakenpu_fence_wait(ggml_backend_t backend, ggml_backend_fence_t fence) {
#if defined(__linux__)
    ggml_backend_fakenpu_context * ctx = (ggml_backend_fakenpu_context *) backend->context;

    const int fd = ggml_backend_fence_fd(fence);
    if (fd < 0) {
        return false;
    }
    // defer the wait to the worker so the host can keep scheduling
    ctx->in_fences.push_back(dup(fd));
    return true;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(fence);
    return false;
#endif
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
    /* .event_record      = */ ggml_backend_fakenpu_event_record,
    /* .event_wait        = */ ggml_backend_fakenpu_event_wait,
    /* .graph_optimize    = */ NULL,
    /* .event_export_fence = */ ggml_backend_fakenpu_event_export_fence,
    /* .fence_wait        = */ ggml_backend_fakenpu_fence_wait,
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
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
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

#if defined(__linux__)
    // require sw_sync, probe that the node is usable
    const int sw_probe = open(FAKENPU_SW_SYNC_PATH, O_RDWR);
    if (sw_probe < 0) {
        GGML_LOG_ERROR("%s: cannot open %s: %s\n", __func__, FAKENPU_SW_SYNC_PATH, strerror(errno));
        ggml_backend_free(ctx->cpu);
        delete ctx;
        return nullptr;
    }
    close(sw_probe);
#endif

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

static ggml_backend_event_t ggml_backend_fakenpu_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_event_t event = new ggml_backend_event;
    event->device  = dev;
    event->context = new ggml_backend_fakenpu_event;
    return event;
}

static void ggml_backend_fakenpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_fakenpu_event * ev = (ggml_backend_fakenpu_event *) event->context;

#if defined(__linux__)
    if (ev->fd >= 0) {
        close(ev->fd);
    }
#endif

    delete ev;
    delete event;

    GGML_UNUSED(dev);
}

static void ggml_backend_fakenpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_fakenpu_event * ev = (ggml_backend_fakenpu_event *) event->context;

#if defined(__linux__)
    if (ev->fd >= 0) {
        ggml_backend_fakenpu_fence_poll(ev->fd);
    }
#endif

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
    /* .event_new            = */ ggml_backend_fakenpu_device_event_new,
    /* .event_free           = */ ggml_backend_fakenpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_fakenpu_device_event_synchronize,
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
