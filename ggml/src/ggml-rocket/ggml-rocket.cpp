#include "ggml-rocket.h"
#include "ggml-dmabuf.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "rocket_accel.h"
#include "rocket-regs.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <linux/sync_file.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static const int ROCKET_CORES = 3;

// One NPU task computes h output rows for at most k_tile input channels and
// n_tile output channels. The K tile is capped by the hardware at 8192.
static const int ROCKET_K_TILE = 768;
static const int ROCKET_N_TILE = 4096;
static const size_t ROCKET_EXPAND_ELEMS = 1024*1024;
static const size_t ROCKET_MAX_TASKS = 32;

using rocket_fence = std::unique_ptr<ggml_backend_fence, decltype(&ggml_backend_fence_free)>;

static const char * rocket_path() {
    const char * path = getenv("GGML_ROCKET_DEVICE");
    return path ? path : "/dev/accel/accel0";
}

static int rocket_min_batch() {
    const char * value = getenv("GGML_ROCKET_MIN_BATCH");
    return value ? std::max(1, atoi(value)) : 32;
}

static void rocket_ioctl(int fd, unsigned long request, void * arg) {
    int ret;
    do {
        ret = ioctl(fd, request, arg);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
        throw std::runtime_error(std::string("Rocket ioctl: ") + strerror(errno));
    }
}

static void rocket_wait(int fd) {
    if (fd < 0) {
        return;
    }
    pollfd pfd = { fd, POLLIN, 0 };
    int ret;
    do {
        ret = poll(&pfd, 1, -1);
    } while (ret < 0 && errno == EINTR);
    if (ret != 1 || pfd.revents != POLLIN) {
        throw std::runtime_error("Rocket job fence failed");
    }
    sync_file_info info = {};
    rocket_ioctl(fd, SYNC_IOC_FILE_INFO, &info);
    if (info.status != 1) {
        throw std::runtime_error("Rocket job completed with an error");
    }
}

// A buffer object imported into each NPU core's address space. Shared buffers
// are dma-buf allocations, so the helper backend can import the same memory.
struct rocket_bo {
    ggml_backend_buffer_t buffer = nullptr;
    void * data = nullptr;
    size_t size = 0;
    std::array<int, ROCKET_CORES> fds;
    std::array<uint32_t, ROCKET_CORES> handles = {};
    std::array<uint32_t, ROCKET_CORES> addresses = {};

    rocket_bo(const std::array<int, ROCKET_CORES> & fds, size_t bytes, bool shared) : fds(fds) {
        if (bytes > UINT32_MAX - 4095) {
            throw std::bad_alloc();
        }
        size = GGML_PAD(bytes, 4096);
        try {
            if (shared) {
                buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dmabuf_buffer_type(), size);
                if (!buffer) {
                    throw std::bad_alloc();
                }
                size = ggml_backend_buffer_get_size(buffer);
                data = ggml_backend_buffer_get_base(buffer);
                for (int c = 0; c < ROCKET_CORES; ++c) {
                    if (fds[c] < 0) {
                        continue;
                    }
                    drm_rocket_import_bo arg = {};
                    arg.fd = ggml_backend_dmabuf_buffer_fd(buffer);
                    rocket_ioctl(fds[c], DRM_IOCTL_ROCKET_IMPORT_BO, &arg);
                    handles[c] = arg.handle;
                    if (arg.dma_address + size > UINT32_MAX) {
                        throw std::bad_alloc();
                    }
                    addresses[c] = arg.dma_address;
                }
            } else {
                // Register buffers have a private mapping in one DRM address space.
                drm_rocket_create_bo arg = {};
                arg.size = size;
                rocket_ioctl(fds[0], DRM_IOCTL_ROCKET_CREATE_BO, &arg);
                handles[0] = arg.handle;
                if (arg.dma_address + size > UINT32_MAX) {
                    throw std::bad_alloc();
                }
                addresses[0] = arg.dma_address;
                data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], arg.offset);
                if (data == MAP_FAILED) {
                    data = nullptr;
                    throw std::runtime_error("Rocket register buffer mmap failed");
                }
            }
        } catch (...) {
            release();
            throw;
        }
    }

    void flush() {
        drm_rocket_fini_bo arg = { handles[0], 0 };
        rocket_ioctl(fds[0], DRM_IOCTL_ROCKET_FINI_BO, &arg);
    }

    void release() {
        for (int c = 0; c < ROCKET_CORES; ++c) {
            if (handles[c]) {
                drm_gem_close arg = { handles[c], 0 };
                ioctl(fds[c], DRM_IOCTL_GEM_CLOSE, &arg);
                handles[c] = 0;
            }
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        } else if (data) {
            munmap(data, size);
        }
        data = nullptr;
    }

    ~rocket_bo() { release(); }
    rocket_bo(const rocket_bo &) = delete;
    rocket_bo & operator=(const rocket_bo &) = delete;
};

// Keeps a conversion node alive; the helper tracks tensor pointers until synchronize.
struct rocket_copy {
    ggml_tensor src = {};
    ggml_tensor dst = {};
};

// Per (m, k, n) tiling and register plan. persistent for the backend lifetime.
struct rocket_plan {
    int h = 0;
    int padded_m = 0;
    int k_tile = 0;
    int parts = 0;
    std::shared_ptr<rocket_bo> packed;
    std::shared_ptr<rocket_bo> input;
    std::array<std::shared_ptr<rocket_bo>, ROCKET_CORES> output;
    std::array<std::shared_ptr<rocket_bo>, ROCKET_CORES> partial;
    std::array<int, ROCKET_CORES> n_begin = {}, n_count = {};
    std::array<std::shared_ptr<rocket_bo>, ROCKET_CORES> regs;
    std::array<std::vector<drm_rocket_task>, ROCKET_CORES> tasks;
};

struct rocket_event {
    ggml_backend_event_t helper = nullptr;
    ggml_backend_fence_t fence = nullptr;
};

struct rocket_context {
    int min_batch = rocket_min_batch();
    ggml_backend_t helper = nullptr;
    ggml_backend_event_t prepare_event = nullptr;
    std::array<int, ROCKET_CORES> fds = {{ -1, -1, -1 }};
    std::array<std::shared_ptr<rocket_bo>, ROCKET_CORES> guard;
    std::shared_ptr<rocket_bo> bias;
    // One reusable BF16 buffer holds the packed weights of the largest matrix seen.
    std::shared_ptr<rocket_bo> packed;
    // Bounded scratch for weights the helper cannot copy to the BF16 NPU layout directly.
    std::shared_ptr<rocket_bo> expand;
    std::map<std::tuple<int, int, int>, std::shared_ptr<rocket_plan>> plans;
    std::vector<std::unique_ptr<rocket_copy>> copies;
    std::vector<ggml_tensor *> conversions;
    std::vector<int> job_fences;

    void retire_fences() {
        size_t completed = 0;
        for (int & fd : job_fences) {
            if (fd >= 0) {
                pollfd pfd = { fd, POLLIN, 0 };
                int ret;
                do {
                    ret = poll(&pfd, 1, 0);
                } while (ret < 0 && errno == EINTR);
                if (ret < 0) {
                    throw std::runtime_error("Rocket cannot poll completion fence");
                }
                if (ret == 0) {
                    break;
                }
                rocket_wait(fd);
                close(fd);
                fd = -1;
            }
            ++completed;
        }
        job_fences.erase(job_fences.begin(), job_fences.begin() + completed);
    }

    void synchronize() {
        ggml_backend_synchronize(helper);
        ggml_backend_event_synchronize(prepare_event);
        for (int fd : job_fences) {
            rocket_wait(fd);
            close(fd);
        }
        job_fences.clear();
        conversions.clear();
        copies.clear();
    }
};

// Helper backend that packs weights and converts layouts. GGML_NPU_HELPER
// selects it by device name (for example "Vulkan0"); the default is Vulkan0.
static ggml_backend_dev_t rocket_helper_dev() {
    static ggml_backend_dev_t dev = [] {
        const char * name = getenv("GGML_NPU_HELPER");
        if (name == nullptr) {
            name = "Vulkan0";
        }
        ggml_backend_dev_t found = ggml_backend_dev_by_name(name);
        if (found == nullptr && strcmp(name, "Vulkan0") != 0) {
            GGML_LOG_WARN("Rocket: GGML_NPU_HELPER '%s' is not a valid device, using Vulkan0\n", name);
            found = ggml_backend_dev_by_name("Vulkan0");
        }
        return found;
    }();
    return dev;
}

// True when the helper can copy this quantized type to BF16 (32-value blocks) or F32.
static bool rocket_helper_quant(ggml_type type) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    static const auto enabled = [] {
        std::array<bool, GGML_TYPE_COUNT> result = {};
        ggml_backend_dev_t helper = rocket_helper_dev();
        if (!helper) {
            return result;
        }
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const auto candidate = (ggml_type) i;
            if (!ggml_is_quantized(candidate)) {
                continue;
            }
            // The float-4 formats lose too much precision when rounded to BF16.
            if (candidate == GGML_TYPE_MXFP4 || candidate == GGML_TYPE_NVFP4) {
                continue;
            }
            const ggml_type to = ggml_blck_size(candidate) == 32 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
            ggml_context * ctx = ggml_init({ 3*ggml_tensor_overhead(), nullptr, true });
            GGML_ASSERT(ctx);
            ggml_tensor * src = ggml_new_tensor_2d(ctx, candidate, ggml_blck_size(candidate), 1);
            ggml_tensor * copy = ggml_cast(ctx, src, to);
            result[i] = ggml_backend_dev_supports_op(helper, copy);
            ggml_free(ctx);
        }
        return result;
    }();
    return enabled[type];
}

static bool rocket_noop(const ggml_tensor * op) {
    return op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE;
}

static bool rocket_matmul(const ggml_tensor * op, int min_batch) {
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * a = op->src[1];
    const int32_t acc = ggml_get_op_params_i32(op, 0);
    const int32_t src = ggml_get_op_params_i32(op, 3);
    // The NPU accumulates in BF16 and cannot honor explicit F32 requests.
    if ((acc != GGML_PREC_UNDEFINED && acc < GGML_PREC_BF16) || (src != GGML_PREC_UNDEFINED && src < GGML_PREC_BF16)) {
        return false;
    }
    return (w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F16 || rocket_helper_quant(w->type)) && a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        w->ne[0] >= 32 && w->ne[0] <= INT32_MAX && w->ne[0] % 32 == 0 &&
        w->ne[1] > 0 && w->ne[1] <= INT32_MAX && w->ne[1] % 16 == 0 &&
        a->ne[1] >= min_batch && ggml_nrows(a) <= 4096 && w->ne[2] == 1 && w->ne[3] == 1 &&
        (ggml_is_contiguous(w) || (w->type == GGML_TYPE_F16 && w->nb[0] == 2 && w->nb[1] % 2 == 0 && w->nb[1] >= size_t(w->ne[0])*2)) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(op);
}

static ggml_tensor rocket_tensor(ggml_type type, ggml_backend_buffer_t buffer, void * data,
                                 std::array<int64_t, 4> ne, std::array<size_t, 4> nb) {
    ggml_tensor t = {};
    t.type = type;
    t.buffer = buffer;
    t.data = data;
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = ne[i];
        t.nb[i] = nb[i];
    }
    return t;
}

// Queue one conversion node for the helper backend.
static void rocket_convert(rocket_context * ctx, ggml_tensor src, ggml_tensor dst, ggml_op op = GGML_OP_CPY) {
    std::unique_ptr<rocket_copy> copy(new rocket_copy);
    copy->src = src;
    copy->dst = dst;
    copy->dst.op = op;
    copy->dst.flags = GGML_TENSOR_FLAG_COMPUTE;
    copy->dst.src[0] = &copy->src;
    copy->dst.src[1] = op == GGML_OP_CPY ? &copy->dst : nullptr;
    ggml_set_name(&copy->dst, "rocket_convert");
    ctx->copies.push_back(std::move(copy));
    ctx->conversions.push_back(&ctx->copies.back()->dst);
}

static void rocket_flush_conversions(rocket_context * ctx) {
    if (ctx->conversions.empty()) {
        return;
    }
    ggml_cgraph graph = {};
    graph.size = graph.n_nodes = ctx->conversions.size();
    graph.nodes = ctx->conversions.data();
    if (ggml_backend_graph_compute_async(ctx->helper, &graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Rocket helper conversion failed");
    }
    ctx->conversions.clear();
}

static std::shared_ptr<rocket_plan> rocket_get_plan(rocket_context * ctx, ggml_tensor * op) {
    const int k = op->src[0]->ne[0];
    const int n = op->ne[0];
    const int m = ggml_nrows(op);
    const size_t bytes = size_t(k)*n*2;
    if (!ctx->packed || ctx->packed->size < bytes) {
        // Retire in-flight work before the old packed buffer is freed.
        ctx->synchronize();
        ctx->plans.clear();
        ctx->packed = std::make_shared<rocket_bo>(ctx->fds, bytes, true);
        GGML_LOG_DEBUG("Rocket: BF16 weight scratch %.2f MiB\n", ctx->packed->size/(1024.0*1024.0));
    }
    auto key = std::make_tuple(m, k, n);
    auto found = ctx->plans.find(key);
    if (found != ctx->plans.end()) {
        return found->second;
    }

    std::shared_ptr<rocket_plan> plan(new rocket_plan);
    plan->packed = ctx->packed;
    const int k_tile = std::min(k, ROCKET_K_TILE);
    plan->k_tile = k_tile;
    plan->parts = (k - 1)/k_tile + 1;
    // Choose the largest row tile that fits the CBUF alongside the weight banks.
    const int weight_banks = (k_tile*16*2 + 32767)/32768 + 1;
    const int cbuf_h = std::min(512, (12 - weight_banks)*32768/(k_tile*2))/4*4;
    const int row_alignment = m % 8 == 0 && cbuf_h >= 8 ? 8 : 4;
    const int max_h = cbuf_h/row_alignment*row_alignment;
    const int row_tiles = (m + max_h - 1)/max_h;
    const int h = ((m + row_tiles - 1)/row_tiles + row_alignment - 1)/row_alignment*row_alignment;
    const int stride = (m + h - 1)/h*h;
    plan->h = h;
    plan->padded_m = stride;
    plan->input = std::make_shared<rocket_bo>(ctx->fds, size_t(stride)*k*2, true);
    ggml_backend_buffer_clear(plan->input->buffer, 0);
    int begin = 0;
    for (int c = 0; c < ROCKET_CORES; ++c) {
        // Split output channels evenly, rounded to the 16-channel atom.
        int count = ((n - begin)/16 + ROCKET_CORES - c - 1)/(ROCKET_CORES - c)*16;
        plan->n_begin[c] = begin;
        plan->n_count[c] = count;
        if (count) {
            plan->output[c] = std::make_shared<rocket_bo>(ctx->fds, size_t(stride)*count*2*plan->parts, true);
            if (plan->parts > 1) {
                plan->partial[c] = std::make_shared<rocket_bo>(ctx->fds, size_t(plan->parts)*count*m*4, true);
            }
        }
        begin += count;
    }

    for (int c = 0; c < ROCKET_CORES; ++c) {
        if (plan->n_count[c] == 0) {
            continue;
        }
        std::vector<uint64_t> commands;
        std::vector<uint32_t> offsets;
        for (int ki = 0, part = 0; ki < k; ki += k_tile, ++part) {
            const int kc = std::min(k_tile, k - ki);
            for (int ni = 0; ni < plan->n_count[c]; ni += ROCKET_N_TILE) {
                const int nc = std::min(ROCKET_N_TILE, plan->n_count[c] - ni);
                for (int mi = 0; mi < stride; mi += h) {
                    offsets.push_back(commands.size()*8);
                    auto regs = rocket_matmul_regs(h, kc, nc, stride,
                        plan->input->addresses[c] + size_t(stride)*ki*2 + size_t(mi)*kc*2,
                        plan->packed->addresses[c] + size_t(n)*ki*2 + size_t(plan->n_begin[c] + ni)*kc*2,
                        plan->output[c]->addresses[c] + (size_t(part)*plan->n_count[c] + ni)*stride*2 + size_t(mi)*16,
                        ctx->bias->addresses[c]);
                    commands.insert(commands.end(), regs.begin(), regs.end());
                    // The PC fetches pairs of 64-bit commands.
                    if (commands.size() % 2) {
                        commands.push_back(0);
                    }
                    plan->tasks[c].push_back({ 0, uint32_t(regs.size()) });
                }
            }
        }
        std::array<int, ROCKET_CORES> fd = {{ ctx->fds[c], -1, -1 }};
        plan->regs[c] = std::make_shared<rocket_bo>(fd, commands.size()*8, false);
        memcpy(plan->regs[c]->data, commands.data(), commands.size()*8);
        plan->regs[c]->flush();
        for (size_t i = 0; i < offsets.size(); ++i) {
            plan->tasks[c][i].regcmd = plan->regs[c]->addresses[0] + offsets[i];
        }
    }
    ctx->plans.emplace(key, plan);
    return plan;
}

// Pack the weights with the helper's OP_CPY into the reusable BF16 buffer in NPU layout,
// then run one job per core and convert the BF16 output back to F32.
static void rocket_compute_matmul(rocket_context * ctx, ggml_tensor * op) {
    const int k = op->src[0]->ne[0];
    const int n = op->ne[0];
    const int m = ggml_nrows(op);
    auto plan = rocket_get_plan(ctx, op);
    const int h = plan->h, stride = plan->padded_m, k_tile = plan->k_tile;
    auto & packed = *plan->packed;
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * a = op->src[1];
    ggml_backend_buffer_t w_buffer = w->buffer ? w->buffer : w->view_src->buffer;
    ggml_backend_buffer_t a_buffer = a->buffer ? a->buffer : a->view_src->buffer;
    const bool quantized = ggml_is_quantized(w->type);
    const bool direct = quantized && ggml_blck_size(w->type) == 32;
    const bool expand = (quantized && !direct) || w->type == GGML_TYPE_F16;
    // F16 has no copy to BF16, so it expands through F32. Quantized types
    // expand through BF16, which halves the scratch.
    const ggml_type expand_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F32 : GGML_TYPE_BF16;
    const size_t expand_size = ggml_type_size(expand_type);
    // Keep conversion tiles aligned to the quantization block and the NPU K tile.
    const int alignment = std::lcm(k_tile, int(ggml_blck_size(w->type)));
    const int columns = expand ? std::min(k, (65536/alignment)*alignment) : k;
    const int rows = expand ? std::min(n, int(ROCKET_EXPAND_ELEMS/columns))/16*16 : n;

    if (expand) {
        const size_t bytes = size_t(columns)*rows*expand_size;
        if (!ctx->expand || ctx->expand->size < bytes) {
            ctx->synchronize();
            const std::array<int, ROCKET_CORES> gpu_only = {{ -1, -1, -1 }};
            ctx->expand = std::make_shared<rocket_bo>(gpu_only, bytes, true);
            GGML_LOG_DEBUG("Rocket: %s conversion scratch %.2f MiB\n", ggml_type_name(expand_type), ctx->expand->size/(1024.0*1024.0));
        }
    }

    for (int kb = 0; kb < k; ) {
        const int width = std::min(columns, k - kb);
        for (int ni = 0; ni < n; ni += rows) {
            const int nc = std::min(rows, n - ni);
            ggml_type type = w->type;
            void * data = (char *) w->data + size_t(ni)*w->nb[1] + size_t(kb/ggml_blck_size(w->type))*w->nb[0];
            size_t row_stride = w->nb[1];
            ggml_backend_buffer_t source = w_buffer;
            if (expand) {
                type = expand_type;
                row_stride = size_t(width)*expand_size;
                rocket_convert(ctx,
                    rocket_tensor(w->type, w_buffer, data,
                        {{ width, nc, 1, 1 }}, {{ w->nb[0], w->nb[1], size_t(nc)*w->nb[1], size_t(nc)*w->nb[1] }}),
                    rocket_tensor(expand_type, ctx->expand->buffer, ctx->expand->data,
                        {{ width, nc, 1, 1 }}, {{ expand_size, row_stride, row_stride*nc, row_stride*nc }}));
                data = ctx->expand->data;
                source = ctx->expand->buffer;
            }
            const size_t element = ggml_type_size(type);
            const size_t block = ggml_blck_size(type);
            for (int ki = kb; ki < kb + width; ) {
                const int kc = std::min(k_tile, kb + width - ki);
                const std::array<int64_t, 4> ne = {{ 32, 16, kc/32, nc/16 }};
                rocket_convert(ctx,
                    rocket_tensor(type, source, (char *) data + size_t(ki - kb)/block*element, ne,
                        {{ element, row_stride, 32/block*element, row_stride*16 }}),
                    rocket_tensor(GGML_TYPE_BF16, packed.buffer, (char *) packed.data + (size_t(n)*ki + size_t(ni)*kc)*2,
                        ne, {{ 2, 64, 1024, size_t(kc)*32 }}));
                ki += kc;
            }
        }
        kb += width;
    }

    for (int ki = 0; ki < k; ki += k_tile) {
        const int kc = std::min(k_tile, k - ki);
        for (int mi = 0; mi < m; ) {
            const int tile_rows = std::min(h, m - mi);
            const int tiles = tile_rows == h ? (m - mi)/h : 1;
            const std::array<int64_t, 4> ne = {{ 8, kc/8, tile_rows, tiles }};
            rocket_convert(ctx,
                rocket_tensor(GGML_TYPE_F32, a_buffer, (char *) a->data + (size_t(mi)*k + ki)*4, ne,
                    {{ 4, 32, size_t(k)*4, size_t(h)*k*4 }}),
                rocket_tensor(GGML_TYPE_BF16, plan->input->buffer, (char *) plan->input->data + (size_t(stride)*ki + size_t(mi)*kc)*2, ne,
                    {{ 2, size_t(h)*16, 16, size_t(h)*kc*2 }}));
            mi += tile_rows*tiles;
        }
    }
    rocket_flush_conversions(ctx);

    // The helper owns the packs; hand the NPU a fence for them.
    ggml_backend_event_record(ctx->prepare_event, ctx->helper);
    rocket_fence prepare(ggml_backend_event_export_fence(ctx->helper, ctx->prepare_event), ggml_backend_fence_free);
    if (!prepare) {
        throw std::runtime_error("Rocket requires helper sync_file export");
    }
    std::array<int, ROCKET_CORES> done = {{ -1, -1, -1 }};
    for (int c = 0; c < ROCKET_CORES; ++c) {
        if (plan->n_count[c] == 0) {
            continue;
        }
        uint32_t inputs[] = { plan->input->handles[c], packed.handles[c], ctx->bias->handles[c], plan->regs[c]->handles[0] };
        uint32_t output = plan->output[c]->handles[c];
        ctx->job_fences.reserve(ctx->job_fences.size() + (plan->tasks[c].size() + ROCKET_MAX_TASKS - 1)/ROCKET_MAX_TASKS);
        for (size_t first = 0; first < plan->tasks[c].size(); first += ROCKET_MAX_TASKS) {
            drm_rocket_job job = {};
            job.tasks = (uintptr_t) (plan->tasks[c].data() + first);
            job.task_count = std::min(ROCKET_MAX_TASKS, plan->tasks[c].size() - first);
            job.task_struct_size = sizeof(drm_rocket_task);
            job.in_bo_handles = (uintptr_t) inputs;
            job.in_bo_handle_count = 4;
            job.out_bo_handles = (uintptr_t) &output;
            job.out_bo_handle_count = 1;
            job.flags = DRM_ROCKET_JOB_IN_FENCE_FD | DRM_ROCKET_JOB_OUT_FENCE_FD;
            job.in_fence_fd = ggml_backend_fence_fd(prepare.get());
            job.out_fence_fd = -1;
            drm_rocket_submit submit = {};
            submit.jobs = (uintptr_t) &job;
            submit.job_count = 1;
            submit.job_struct_size = sizeof(job);
            rocket_ioctl(ctx->fds[c], DRM_IOCTL_ROCKET_SUBMIT, &submit);
            if (job.out_fence_fd < 0) {
                throw std::runtime_error("Rocket requires SUBMIT output fences");
            }
            done[c] = job.out_fence_fd;
            ctx->job_fences.push_back(done[c]);
        }
    }

    // Make the helper wait for each core, then convert/reduce the BF16 results.
    for (int c = 0; c < ROCKET_CORES; ++c) {
        if (done[c] < 0) {
            continue;
        }
        const int wait_fd = fcntl(done[c], F_DUPFD_CLOEXEC, 0);
        if (wait_fd < 0) {
            throw std::runtime_error("Rocket cannot duplicate completion fence");
        }
        rocket_fence fence(ggml_backend_fence_init(wait_fd), ggml_backend_fence_free);
        if (!ggml_backend_fence_wait(ctx->helper, fence.get())) {
            throw std::runtime_error("Rocket requires helper sync_file import");
        }
        const int nc = plan->n_count[c];
        if (plan->parts == 1) {
            rocket_convert(ctx,
                rocket_tensor(GGML_TYPE_BF16, plan->output[c]->buffer, plan->output[c]->data, {{ 1, 8, nc/8, m }},
                    {{ size_t(stride)*nc*2, 2, size_t(stride)*16, 16 }}),
                rocket_tensor(GGML_TYPE_F32, op->buffer, (char *) op->data + size_t(plan->n_begin[c])*4, {{ 1, 8, nc/8, m }},
                    {{ 4, 4, 32, size_t(n)*4 }}));
        } else {
            // Each K tile wrote a separate partial; gather them and sum on the GPU.
            const int parts = plan->parts;
            for (int p = 0; p < parts; ++p) {
                rocket_convert(ctx,
                    rocket_tensor(GGML_TYPE_BF16, plan->output[c]->buffer, (char *) plan->output[c]->data + size_t(p)*nc*stride*2,
                        {{ 1, 8, nc/8, m }}, {{ size_t(stride)*nc*2, 2, size_t(stride)*16, 16 }}),
                    rocket_tensor(GGML_TYPE_F32, plan->partial[c]->buffer, (char *) plan->partial[c]->data + size_t(p)*4,
                        {{ 1, 8, nc/8, m }}, {{ 4, size_t(4)*parts, size_t(32)*parts, size_t(nc)*4*parts }}));
            }
            rocket_convert(ctx,
                rocket_tensor(GGML_TYPE_F32, plan->partial[c]->buffer, plan->partial[c]->data, {{ parts, 8, nc/8, m }},
                    {{ 4, size_t(4)*parts, size_t(32)*parts, size_t(nc)*4*parts }}),
                rocket_tensor(GGML_TYPE_F32, op->buffer, (char *) op->data + size_t(plan->n_begin[c])*4, {{ 1, 8, nc/8, m }},
                    {{ 4, 4, 32, size_t(n)*4 }}), GGML_OP_SUM_ROWS);
        }
    }
    rocket_flush_conversions(ctx);
}

static const char * rocket_name(ggml_backend_t) { return "Rocket"; }

static void rocket_synchronize(ggml_backend_t backend) {
    try {
        ((rocket_context *) backend->context)->synchronize();
    } catch (const std::exception & e) {
        GGML_ABORT("%s", e.what());
    }
}

static void rocket_free(ggml_backend_t backend) {
    auto * ctx = (rocket_context *) backend->context;
    rocket_synchronize(backend);
    ctx->plans.clear();
    ctx->packed.reset();
    ctx->expand.reset();
    ctx->bias.reset();
    ctx->guard = {};
    ggml_backend_event_free(ctx->prepare_event);
    ggml_backend_free(ctx->helper);
    for (int fd : ctx->fds) {
        close(fd);
    }
    delete ctx;
    delete backend;
}

static ggml_status rocket_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = (rocket_context *) backend->context;
    try {
        ctx->retire_fences();
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * node = graph->nodes[i];
            if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE) || rocket_noop(node)) {
                continue;
            }
            if (!rocket_matmul(node, ctx->min_batch)) {
                GGML_LOG_ERROR("%s: unsupported operation %s (%s)\n", __func__, ggml_op_name(node->op), node->name);
                return GGML_STATUS_FAILED;
            }
            rocket_compute_matmul(ctx, node);
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        ctx->conversions.clear();
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::exception & e) {
        ctx->conversions.clear();
        GGML_LOG_ERROR("%s: %s\n", __func__, e.what());
        return GGML_STATUS_FAILED;
    }
}

static void rocket_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (rocket_context *) backend->context;
    auto * ev = (rocket_event *) event->context;
    ggml_backend_fence_free(ev->fence);
    ggml_backend_event_record(ev->helper, ctx->helper);
    ev->fence = ggml_backend_event_export_fence(ctx->helper, ev->helper);
}

static bool rocket_fence_wait(ggml_backend_t backend, ggml_backend_fence_t fence) {
    return ggml_backend_fence_wait(((rocket_context *) backend->context)->helper, fence);
}

static void rocket_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ev = (rocket_event *) event->context;
    GGML_ASSERT(ev->fence && rocket_fence_wait(backend, ev->fence));
}

static ggml_backend_fence_t rocket_event_export(ggml_backend_t, ggml_backend_event_t event) {
    return ggml_backend_fence_dup(((rocket_event *) event->context)->fence);
}

static const ggml_backend_i rocket_iface = {
    /* .get_name                = */ rocket_name,
    /* .free                    = */ rocket_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ rocket_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ rocket_graph_compute,
    /* .event_record            = */ rocket_event_record,
    /* .event_wait              = */ rocket_event_wait,
    /* .graph_optimize          = */ nullptr,
    /* .event_export_fence      = */ rocket_event_export,
    /* .fence_wait              = */ rocket_fence_wait,
};

static const char * rocket_device_name(ggml_backend_dev_t) { return "Rocket"; }
static const char * rocket_description(ggml_backend_dev_t) { return "RK3588 NPU (dma-buf)"; }
static enum ggml_backend_dev_type rocket_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void rocket_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
    ggml_backend_dev_t helper = rocket_helper_dev();
    if (helper) {
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(helper, &props);
        *free = props.memory_free;
        *total = props.memory_total;
    }
}

static void rocket_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = rocket_device_name(dev);
    props->description = rocket_description(dev);
    props->type = rocket_type(dev);
    props->device_id = nullptr;
    rocket_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { true, false, false, true, false };
}

static ggml_backend_buffer_type_t rocket_buft(ggml_backend_dev_t) {
    // Initialize the helper's dma-buf importer before any model buffers are allocated.
    ggml_backend_dev_t helper = rocket_helper_dev();
    if (helper) {
        ggml_backend_dev_buffer_type(helper);
    }
    return ggml_backend_dmabuf_buffer_type();
}

static bool rocket_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return rocket_noop(op) || rocket_matmul(op, rocket_min_batch());
}

static bool rocket_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_dmabuf_buffer_type();
}

static ggml_backend_t rocket_init(ggml_backend_dev_t dev, const char *) {
    std::unique_ptr<rocket_context> ctx(new rocket_context);
    try {
        ggml_backend_dev_t helper = rocket_helper_dev();
        if (!helper) {
            throw std::runtime_error("no helper backend for Rocket");
        }
        ctx->helper = ggml_backend_dev_init(helper, nullptr);
        if (!ctx->helper) {
            return nullptr;
        }
        ctx->prepare_event = ggml_backend_event_new(helper);
        for (int & fd : ctx->fds) {
            fd = open(rocket_path(), O_RDWR | O_CLOEXEC);
            if (fd < 0) {
                throw std::runtime_error(std::string("open ") + rocket_path() + ": " + strerror(errno));
            }
        }
        for (int c = 0; c < ROCKET_CORES; ++c) {
            std::array<int, ROCKET_CORES> fd = {{ ctx->fds[c], -1, -1 }};
            ctx->guard[c] = std::make_shared<rocket_bo>(fd, 4096, false);
        }
        ctx->bias = std::make_shared<rocket_bo>(ctx->fds, size_t(ROCKET_N_TILE)*4 + 4096, true);
        ggml_backend_buffer_clear(ctx->bias->buffer, 0);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: %s\n", __func__, e.what());
        ctx->bias.reset();
        ctx->guard = {};
        ggml_backend_event_free(ctx->prepare_event);
        if (ctx->helper) {
            ggml_backend_free(ctx->helper);
        }
        for (int fd : ctx->fds) {
            if (fd >= 0) {
                close(fd);
            }
        }
        return nullptr;
    }
    static ggml_guid guid = { 0x68, 0x4f, 0xc4, 0xe3, 0x4a, 0x77, 0x4c, 0x9f, 0x88, 0x33, 0x32, 0x6c, 0x1d, 0x82, 0x00, 0x01 };
    return new ggml_backend { &guid, rocket_iface, dev, ctx.release() };
}

static ggml_backend_event_t rocket_event_new(ggml_backend_dev_t dev) {
    auto * ev = new rocket_event;
    ev->helper = ggml_backend_event_new(rocket_helper_dev());
    return new ggml_backend_event { dev, ev };
}

static void rocket_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    auto * ev = (rocket_event *) event->context;
    ggml_backend_fence_free(ev->fence);
    ggml_backend_event_free(ev->helper);
    delete ev;
    delete event;
}

static void rocket_event_sync(ggml_backend_dev_t, ggml_backend_event_t event) {
    ggml_backend_event_synchronize(((rocket_event *) event->context)->helper);
}

static const ggml_backend_device_i rocket_device_iface = {
    /* .get_name             = */ rocket_device_name,
    /* .get_description      = */ rocket_description,
    /* .get_memory           = */ rocket_memory,
    /* .get_type             = */ rocket_type,
    /* .get_props            = */ rocket_props,
    /* .init_backend         = */ rocket_init,
    /* .get_buffer_type      = */ rocket_buft,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ rocket_supports_op,
    /* .supports_buft        = */ rocket_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ rocket_event_new,
    /* .event_free           = */ rocket_event_free,
    /* .event_synchronize    = */ rocket_event_sync,
};

static const char * rocket_reg_name(ggml_backend_reg_t) { return "Rocket"; }

static size_t rocket_device_count(ggml_backend_reg_t) {
    const int fd = open(rocket_path(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    char name[16] = {};
    drm_version version = {};
    version.name = name;
    version.name_len = sizeof(name) - 1;
    const bool is_rocket = ioctl(fd, DRM_IOCTL_VERSION, &version) == 0 && strcmp(name, "rocket") == 0;
    close(fd);
    // The helper device cannot be resolved here: this runs while the backend
    // registry is still being built. rocket_init() resolves it later.
    return is_rocket ? 1 : 0;
}

static ggml_backend_dev_t rocket_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { rocket_device_iface, nullptr, nullptr };
    dev.reg = reg;
    return &dev;
}

ggml_backend_reg_t ggml_backend_rocket_reg(void) {
    static ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        { rocket_reg_name, rocket_device_count, rocket_device, nullptr },
        nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rocket_reg)
