// Local NPU1 experiment. Included after the Vulkan graph implementation.

static bool ggml_vk_npu_matches(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        node->type != GGML_TYPE_F32 || node->ne[0] != 10240 || node->ne[1] != 512 ||
        node->ne[2] != 1 || node->ne[3] != 1 || !ggml_is_contiguous(node)) {
        return false;
    }
    const auto * w = node->src[0];
    const auto * a = node->src[1];
    const int32_t acc = ggml_get_op_params_i32(node, 0);
    if ((acc != GGML_PREC_DEFAULT && acc != GGML_PREC_BF16) || ggml_get_op_params_i32(node, 3) == GGML_PREC_F32) {
        return false;
    }
    if (!w || !a || w->type != GGML_TYPE_Q5_K || w->op != GGML_OP_NONE ||
        w->ne[0] != 2560 || w->ne[1] != 10240 || w->ne[2] != 1 || w->ne[3] != 1 ||
        a->type != GGML_TYPE_F32 || a->ne[0] != 2560 || a->ne[1] != 512 ||
        a->ne[2] != 1 || a->ne[3] != 1 || !ggml_is_contiguous(w) || !ggml_is_contiguous(a)) {
        return false;
    }
    for (const auto * t : {node, w, a}) {
        if (!t->buffer || !ggml_backend_buffer_is_vk(t->buffer)) {
            return false;
        }
    }
    return static_cast<ggml_backend_vk_buffer_context *>(w->buffer->context)->dev_buffer->npu_exportable;
}

static xrt::bo & ggml_vk_npu_import(const vk_buffer & buffer, xrt::device & device) {
    std::lock_guard<std::mutex> lock(buffer->npu_mutex);
    if (!buffer->npu_exportable) {
        throw std::runtime_error("Q5_K NPU: buffer is not exportable");
    }
    if (!buffer->npu_bo) {
        const int fd = buffer->device->device.getMemoryFdKHR({buffer->device_memory, vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT});
        try {
            buffer->npu_bo = std::make_unique<xrt::bo>(device, fd);
        } catch (...) {
            close(fd);
            throw;
        }
        buffer->npu_dma_fd = fd;
    }
    return *buffer->npu_bo;
}

static void ggml_vk_npu_handoff(ggml_backend_vk_context * ctx, const std::array<vk_subbuffer, 3> & ranges, bool release, bool uploaded = false) {
    auto subctx = ggml_vk_get_compute_ctx(ctx);
    // The NPU reads weights and A, and overwrites C.
    const std::array<vk::AccessFlags, 3> writes = {
        uploaded ? vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite : vk::AccessFlags{},
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlags{}};
    const std::array<vk::AccessFlags, 3> reads = {vk::AccessFlagBits::eShaderRead, vk::AccessFlags{}, vk::AccessFlagBits::eShaderRead};
    vk::ExternalMemoryAcquireUnmodifiedEXT unmodified{true};
    std::array<vk::BufferMemoryBarrier, 3> barriers;
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto & range = ranges[i];
        auto & barrier = barriers[i];
        barrier.srcAccessMask = release ? writes[i] : vk::AccessFlags{};
        barrier.dstAccessMask = release ? vk::AccessFlags{} : reads[i];
        barrier.srcQueueFamilyIndex = release ? ctx->device->compute_queue->queue_family_index : VK_QUEUE_FAMILY_FOREIGN_EXT;
        barrier.dstQueueFamilyIndex = release ? VK_QUEUE_FAMILY_FOREIGN_EXT : ctx->device->compute_queue->queue_family_index;
        barrier.buffer = range.buffer->buffer;
        barrier.offset = range.offset;
        barrier.size = range.size;
        if (!release && i < 2 && ctx->device->external_memory_acquire_unmodified) {
            barrier.pNext = &unmodified;
        }
    }
    subctx->s->buffer->buf.pipelineBarrier(
        release ? vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eHost : vk::PipelineStageFlagBits::eTopOfPipe,
        release ? vk::PipelineStageFlagBits::eBottomOfPipe : vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, barriers, {});
}

struct ggml_vk_npu {
    static constexpr int M = 512;
    static constexpr int K = 2560;
    static constexpr int N = 10240;
    int npu_n = 0;
    xrt::device device{0};
    xrt::hw_context hw_context;
    xrt::kernel kernel;
    xrt::bo instructions;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> scratch{nullptr, ggml_backend_buffer_free};
    ggml_tensor * a = nullptr;
    ggml_tensor * c = nullptr;
    xrt::bo a_bo, c_bo;
    uint32_t instruction_bytes = 0;
    struct dispatch {
        ggml_tensor cast{}, gpu_w{}, gpu_c{}, merge{};
        vk_buffer weights;
        xrt::bo b_bo;
        xrt::run run;
        vk::UniqueSemaphore ready, completion;
        vk::UniqueQueryPool queries;
        bool running = false;
        bool profile_pending = false;
        bool gpu_started = false, gpu_done = false;
    };
    // Keep commands and tensor views alive until the backend synchronizes.
    std::deque<dispatch> pending;
    uint64_t calls = 0;
    double total_ms = 0, prepare_ms = 0, npu_join_ms = 0, finish_ms = 0;
    const bool profile = getenv("GGML_VK_NPU_PROFILE") != nullptr;
    const bool verify = getenv("GGML_VK_NPU_VERIFY") != nullptr;
    nlohmann::json scale_table;
    vk_device profile_device;

    ggml_vk_npu(ggml_backend_t backend, const std::string & directory) {
        std::ifstream manifest_file(directory + "/q5_k_mm.json");
        if (!manifest_file) throw std::runtime_error("Q5_K NPU: missing q5_k_mm.json");
        const auto manifest = nlohmann::json::parse(manifest_file);
        const auto & params = manifest.at("parameters");
        npu_n = params.at("N").get<int>();
        if (manifest.at("schema_version") != 1 || manifest.at("target") != "npu1" ||
            manifest.at("kernel_name") != "MLIR_AIE" || params.at("M") != M || params.at("K") != K ||
            npu_n <= 0 || npu_n >= N || npu_n % 256 || params.at("math") != "bf16" ||
            params.at("profile") != "full" || params.at("c_bf16") != true ||
            params.at("bounded_scales") != true || params.at("trace_size") != 0 ||
            manifest.at("xrt").at("opcode") != 3 || manifest.at("xrt").at("num_host_bos") != 3 ||
            manifest.at("xrt").at("instruction_size_unit") != "bytes") {
            throw std::runtime_error("Q5_K NPU: expected a bounded BF16 512 x 2560 kernel with 0 < N < 10240");
        }
        const size_t a_bytes = size_t(M) * K * 2;
        const size_t b_bytes = size_t(npu_n) * (K / 256) * 176;
        const size_t c_bytes = size_t(M) * npu_n * 2;
        if (const char * path = getenv("GGML_VK_NPU_SCALE_TABLE")) {
            std::ifstream input(path);
            scale_table = nlohmann::json::parse(input);
            if (scale_table.at("schema_version") != 1 || scale_table.at("K") != K ||
                scale_table.at("N") != N || scale_table.at("npu_n") != npu_n) {
                throw std::runtime_error("Q5_K NPU: scale table does not match the split");
            }
        }
        const auto & buffers = manifest.at("buffers");
        if (buffers.size() != 3 || buffers[0].at("bytes") != a_bytes || buffers[1].at("bytes") != b_bytes ||
            buffers[2].at("bytes") != c_bytes || buffers[0].at("dtype") != "bf16" ||
            buffers[1].at("dtype") != "uint8" || buffers[2].at("dtype") != "bf16" ||
            buffers[0].at("argument") != 3 || buffers[1].at("argument") != 4 || buffers[2].at("argument") != 5) {
            throw std::runtime_error("Q5_K NPU: incompatible buffer ABI");
        }

        xrt::xclbin binary(directory + "/q5_k_mm.xclbin");
        device.register_xclbin(binary);
        hw_context = xrt::hw_context(device, binary.get_uuid());
        kernel = xrt::kernel(hw_context, "MLIR_AIE");
        std::ifstream stream(directory + "/q5_k_mm.bin", std::ios::binary | std::ios::ate);
        if (!stream || stream.tellg() <= 0) throw std::runtime_error("Q5_K NPU: missing instructions");
        const size_t bytes = size_t(stream.tellg());
        if (bytes % 4 || bytes > UINT32_MAX || manifest.at("artifacts").at("bin").at("bytes") != bytes) {
            throw std::runtime_error("Q5_K NPU: invalid instruction size");
        }
        instructions = xrt::bo(device, bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
        stream.seekg(0);
        if (!stream.read(instructions.map<char *>(), bytes)) throw std::runtime_error("Q5_K NPU: cannot read instructions");
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        tensors.reset(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
        if (!tensors) throw std::bad_alloc();
        a = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_BF16, K, M);
        c = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_BF16, npu_n, M);
        ggml_set_name(a, "npu_q5_input_bf16");
        ggml_set_name(c, "npu_q5_output_bf16");
        scratch.reset(ggml_backend_alloc_ctx_tensors(tensors.get(), backend));
        if (!scratch) throw std::bad_alloc();
        auto buffer = static_cast<ggml_backend_vk_buffer_context *>(scratch->context)->dev_buffer;
        auto & imported = ggml_vk_npu_import(buffer, device);
        a_bo = xrt::bo(imported, a_bytes, vk_tensor_offset(a));
        c_bo = xrt::bo(imported, c_bytes, vk_tensor_offset(c));
        instruction_bytes = uint32_t(bytes);
        profile_device = static_cast<ggml_backend_vk_context *>(backend->context)->device;
        GGML_LOG_INFO("ggml_vulkan: Q5_K split M=%d K=%d N=%d: NPU=%d GPU=%d, original weights via dma-buf, %s\n",
                      M, K, N, npu_n, N - npu_n, getenv("GGML_VK_NPU_CONTROL") ? "import only control" : "dma-buf fences in both directions");
    }

    ~ggml_vk_npu() {
        if (calls) {
            GGML_LOG_INFO("ggml_vulkan: Q5_K split %llu calls, mean CPU call %.3f ms (prepare %.3f, NPU dispatch %.3f, merge submit %.3f)\n",
                          (unsigned long long) calls, total_ms / calls, prepare_ms / calls, npu_join_ms / calls, finish_ms / calls);
        }
    }

    void finish() {
        for (auto & job : pending) {
            if (job.running) {
                const auto state = job.run.state();
                if (state != ERT_CMD_STATE_COMPLETED) {
                    throw std::runtime_error("Q5_K NPU: GPU completed but XRT state=" + std::to_string(state));
                }
                job.running = false;
            }
            collect_profile(job);
        }
        pending.clear();
    }

    void export_ready(ggml_backend_vk_context * ctx, dispatch & job, const vk_buffer & storage) {
        auto subctx = ggml_vk_get_compute_ctx(ctx);
        subctx->s->signal_semaphores.push_back({*job.ready, 0});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ctx->compute_ctx.reset();

        const int fd = profile_device->device.getSemaphoreFdKHR({*job.ready, vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd});
        // A SYNC_FD of -1 is already signaled.
        if (fd == -1) return;
        const dma_buf_import_sync_file fence{DMA_BUF_SYNC_WRITE, fd};
        const int result = ioctl(storage->npu_dma_fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &fence);
        const int error = errno;
        close(fd);
        if (result < 0) {
            throw std::system_error(error, std::generic_category(), "Q5_K NPU: import Vulkan release fence");
        }
    }

    void import_completion(dispatch & job, const vk_buffer & storage) {
        dma_buf_export_sync_file fence{};
        fence.flags = DMA_BUF_SYNC_READ;
        fence.fd = -1;
        if (ioctl(storage->npu_dma_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &fence) < 0) {
            throw std::system_error(errno, std::generic_category(), "Q5_K NPU: export completion fence");
        }
        try {
            profile_device->device.importSemaphoreFdKHR({*job.completion, vk::SemaphoreImportFlagBits::eTemporary, vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd, fence.fd});
        } catch (...) {
            close(fence.fd);
            throw;
        }
    }

    void collect_profile(dispatch & job) {
        if (!job.profile_pending) return;
        std::array<uint64_t, 6> ticks{};
        const auto result = profile_device->device.getQueryPoolResults(*job.queries, 0, ticks.size(), sizeof(ticks), ticks.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eSuccess) {
            const double ms = profile_device->properties.limits.timestampPeriod / 1e6;
            GGML_LOG_INFO("ggml_vulkan: Q5_K device %s cast=%.3f gpu_mm=%.3f merge=%.3f cast_to_mm=%.3f mm_to_merge=%.3f gpu_started_at_dispatch=%d gpu_done_at_dispatch=%d ms\n",
                          job.gpu_w.name, (ticks[1] - ticks[0]) * ms, (ticks[3] - ticks[2]) * ms, (ticks[5] - ticks[4]) * ms,
                          (ticks[2] - ticks[1]) * ms, (ticks[4] - ticks[3]) * ms, job.gpu_started, job.gpu_done);
        } else {
            GGML_LOG_WARN("ggml_vulkan: Q5_K device timestamps unavailable: %s\n", vk::to_string(result).c_str());
        }
        job.profile_pending = false;
    }

    bool check_scales(vk_buffer & buffer, size_t offset, const char * name) {
        const size_t bytes = size_t(npu_n) * (K / 256) * 176;
        std::lock_guard<std::mutex> lock(buffer->npu_mutex);
        const auto key = std::make_pair(offset, bytes);
        const auto cached = buffer->npu_scales.find(key);
        if (cached != buffer->npu_scales.end()) return cached->second.valid;
        if (!scale_table.is_null()) {
            const auto & entries = scale_table.at("tensors");
            const bool valid = entries.contains(name) && entries.at(name).at("eligible").get<bool>();
            buffer->npu_scales.emplace(key, vk_buffer_struct::npu_weight_state{valid, true});
            if (!valid) GGML_LOG_WARN("ggml_vulkan: Q5_K NPU scale table excludes %s; using Vulkan\n", name);
            return valid;
        }
        std::vector<uint8_t> copy;
        const uint8_t * data;
        if (buffer->ptr && (buffer->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            data = static_cast<const uint8_t *>(buffer->ptr) + offset;
        } else {
            copy.resize(bytes);
            ggml_vk_buffer_read(buffer, offset, copy.data(), bytes);
            data = copy.data();
        }
        bool valid = true;
        for (size_t i = 0; i < bytes; i += 176) {
            uint16_t scales[2];
            memcpy(scales, data + i, sizeof(scales));
            const int ed = std::max(1, int((scales[0] >> 10) & 31));
            const int em = std::max(1, int((scales[1] >> 10) & 31));
            if ((scales[0] | scales[1]) & 0x8000 || ed == 31 || em == 31 || em < ed || em > ed + 5) {
                valid = false;
                break;
            }
        }
        buffer->npu_scales.emplace(key, vk_buffer_struct::npu_weight_state{valid});
        if (!valid) GGML_LOG_WARN("ggml_vulkan: Q5_K NPU scale limits exceeded for %s; using Vulkan\n", name);
        return valid;
    }

    static double elapsed(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    bool execute(ggml_backend_t backend, const ggml_cgraph * graph, ggml_tensor * node) {
        auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
        const auto begin = std::chrono::steady_clock::now();
        auto * w = node->src[0];
        auto weights = static_cast<ggml_backend_vk_buffer_context *>(w->buffer->context)->dev_buffer;
        const size_t weight_offset = ggml_vk_tensor_buffer_offset(ctx, w);
        const size_t weight_bytes = size_t(npu_n) * w->nb[1];
        if (weights->device != ctx->device) return false;
        if (weight_offset > weights->size || weight_bytes > weights->size - weight_offset) {
            throw std::runtime_error("Q5_K NPU: weight slice exceeds the allocation");
        }
        if (!check_scales(weights, weight_offset, w->name)) return false;
        const auto checked = std::chrono::steady_clock::now();
        pending.emplace_back();
        auto & job = pending.back();
        job.weights = weights;
        auto & imported = ggml_vk_npu_import(weights, device);
        job.b_bo = xrt::bo(imported, weight_bytes, weight_offset);
        job.run = xrt::run(kernel);
        job.run.set_arg(0, uint32_t(3));
        job.run.set_arg(1, instructions);
        job.run.set_arg(2, instruction_bytes);
        job.run.set_arg(3, a_bo);
        job.run.set_arg(4, job.b_bo);
        job.run.set_arg(5, c_bo);
        vk::ExportSemaphoreCreateInfo export_info{vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd};
        vk::SemaphoreCreateInfo semaphore_info{};
        semaphore_info.setPNext(&export_info);
        job.ready = profile_device->device.createSemaphoreUnique(semaphore_info);
        job.completion = profile_device->device.createSemaphoreUnique({});
        const auto bound = std::chrono::steady_clock::now();
        if (profile) {
            job.queries = profile_device->device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 6});
            profile_device->device.resetQueryPool(*job.queries, 0, 6);
            ctx->npu_profile_pool = *job.queries;
        }

        auto gpu_node = [&](ggml_tensor * tensor, int query = -1) {
            const uint64_t saved_flops = ctx->last_total_flops;
            ggml_cgraph part = *graph;
            part.nodes = &tensor;
            part.n_nodes = 1;
            ctx->npu_profile_query = profile ? query : -1;
            const auto status = ggml_backend_vk_graph_compute_gpu(backend, &part);
            ctx->npu_profile_query = -1;
            ctx->last_total_flops = saved_flops;
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Q5_K split: Vulkan graph failed");
        };

        auto & cast = job.cast;
        cast = *a;
        cast.op = GGML_OP_CPY;
        cast.flags = GGML_TENSOR_FLAG_COMPUTE;
        cast.src[0] = node->src[1];
        gpu_node(&cast, 0);
        const auto cast_submitted = std::chrono::steady_clock::now();
        auto storage = static_cast<ggml_backend_vk_buffer_context *>(scratch->context)->dev_buffer;
        const std::array<vk_subbuffer, 3> shared = {{{weights, weight_offset, weight_bytes},
            {storage, vk_tensor_offset(a), ggml_nbytes(a)}, {storage, vk_tensor_offset(c), ggml_nbytes(c)}}};
        {
            std::lock_guard<std::mutex> lock(weights->npu_mutex);
            auto & state = weights->npu_scales.at({weight_offset, weight_bytes});
            ggml_vk_npu_handoff(ctx, shared, true, state.uploaded);
            state.uploaded = false;
        }
        export_ready(ctx, job, storage);
        const auto released = std::chrono::steady_clock::now();

        auto & gpu_w = job.gpu_w;
        auto & gpu_c = job.gpu_c;
        gpu_w = *w;
        gpu_w.view_src = w->view_src ? w->view_src : w;
        gpu_w.view_offs += weight_bytes;
        gpu_w.data = static_cast<char *>(w->data) + weight_bytes;
        gpu_w.ne[1] = N - npu_n;
        gpu_w.nb[2] = gpu_w.nb[3] = gpu_w.ne[1] * gpu_w.nb[1];
        gpu_c = *node;
        gpu_c.view_src = node->view_src ? node->view_src : node;
        gpu_c.view_offs += npu_n * sizeof(float);
        gpu_c.data = static_cast<char *>(node->data) + npu_n * sizeof(float);
        gpu_c.ne[0] = N - npu_n;
        gpu_c.src[0] = &gpu_w;
        ggml_format_name(&gpu_c, "%s_gpu", node->name);

        const auto launch = std::chrono::steady_clock::now();
        double gpu_ms = 0;
        double start_ms = 0;
        try {
            job.run.start();
            const auto started = std::chrono::steady_clock::now();
            start_ms = elapsed(launch, started);
            job.running = true;
            import_completion(job, storage);
            const auto gpu_begin = std::chrono::steady_clock::now();
            gpu_node(&gpu_c, 2);
            gpu_ms = elapsed(gpu_begin, std::chrono::steady_clock::now());
        } catch (...) {
            if (job.running) job.run.abort();
            job.running = false;
            ggml_vk_synchronize(ctx);
            ggml_vk_graph_cleanup(ctx);
            ggml_vk_npu_handoff(ctx, shared, false);
            ggml_vk_synchronize(ctx);
            ggml_vk_graph_cleanup(ctx);
            throw;
        }
        const auto joined = std::chrono::steady_clock::now();
        if (profile) {
            std::array<uint64_t, 4> probe{};
            const auto result = profile_device->device.getQueryPoolResults(*job.queries, 2, 2, sizeof(probe), probe.data(), sizeof(uint64_t) * 2,
                                                                          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
            job.gpu_started = (result == vk::Result::eSuccess || result == vk::Result::eNotReady) && probe[1] != 0;
            job.gpu_done = result == vk::Result::eSuccess && probe[3] != 0;
        }
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_get_compute_ctx(ctx)->s->wait_semaphores.push_back({*job.completion, 0, vk::PipelineStageFlagBits::eAllCommands});
        ggml_vk_npu_handoff(ctx, shared, false);
        const auto acquired = std::chrono::steady_clock::now();
        auto & merge = job.merge;
        merge = *node;
        merge.op = GGML_OP_CPY;
        merge.ne[0] = npu_n;
        std::fill(std::begin(merge.src), std::end(merge.src), nullptr);
        merge.src[0] = c;
        ggml_format_name(&merge, "%s_npu", node->name);
        gpu_node(&merge, 4);
        if (profile) {
            job.profile_pending = true;
        }
        const auto end = std::chrono::steady_clock::now();
        ++calls;
        total_ms += elapsed(begin, end);
        prepare_ms += elapsed(begin, launch);
        npu_join_ms += elapsed(launch, joined);
        finish_ms += elapsed(joined, end);
        if (profile) {
            GGML_LOG_INFO("ggml_vulkan: Q5_K split %s prepare=%.3f npu_dispatch=%.3f gpu_submit=%.3f merge_submit=%.3f cpu_total=%.3f ms\n",
                          w->name, elapsed(begin, launch), elapsed(launch, joined), gpu_ms, elapsed(joined, end), elapsed(begin, end));
            GGML_LOG_INFO("ggml_vulkan: Q5_K stages %s check=%.3f bind=%.3f cast_submit=%.3f release_submit=%.3f views=%.3f xrt_start=%.3f acquire=%.3f merge=%.3f ms\n",
                          w->name, elapsed(begin, checked), elapsed(checked, bound), elapsed(bound, cast_submitted),
                          elapsed(cast_submitted, released), elapsed(released, launch), start_ms,
                          elapsed(joined, acquired), elapsed(acquired, end));
        }
        if (verify) {
            std::vector<float> actual(M * N), reference(M * N);
            ggml_backend_tensor_get(node, actual.data(), 0, ggml_nbytes(node));
            gpu_node(node);
            ggml_backend_tensor_get(node, reference.data(), 0, ggml_nbytes(node));
            double error = 0, norm = 0;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < npu_n; ++n) {
                    const size_t i = size_t(m) * N + n;
                    const double delta = double(actual[i]) - reference[i];
                    error += delta * delta;
                    norm += double(reference[i]) * reference[i];
                }
            }
            const double relative = std::sqrt(error / std::max(norm, 1e-30));
            GGML_LOG_INFO("ggml_vulkan: Q5_K verify %s offset=%zu relative_l2=%.6g\n", w->name, weight_offset, relative);
            if (!std::isfinite(relative) || relative > .02) throw std::runtime_error("Q5_K NPU: model projection verification failed");
        }
        return true;
    }
};

static void ggml_vk_npu_finish(ggml_backend_vk_context * ctx) {
    if (ctx->npu) ctx->npu->finish();
}

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    const char * directory = getenv("GGML_VK_NPU_Q5_K_DIR");
    if (!directory) return ggml_backend_vk_graph_compute_gpu(backend, cgraph);
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    const char * control = getenv("GGML_VK_NPU_CONTROL");
    const bool asynchronous = control && strcmp(control, "segmented-async") == 0;
    const bool sleeping = control && strcmp(control, "segmented-sleep") == 0;
    const bool segmented = control && (strcmp(control, "segmented") == 0 || asynchronous || sleeping);
    const bool profile = getenv("GGML_VK_NPU_PROFILE") != nullptr;
    if (control && !segmented) {
        if (strcmp(control, "import") == 0) {
            try {
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    if (!ggml_vk_npu_matches(cgraph->nodes[i])) continue;
                    if (!ctx->npu) ctx->npu = std::make_shared<ggml_vk_npu>(backend, directory);
                    auto buffer = static_cast<ggml_backend_vk_buffer_context *>(cgraph->nodes[i]->src[0]->buffer->context)->dev_buffer;
                    ggml_vk_npu_import(buffer, ctx->npu->device);
                }
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml_vulkan: Q5_K import control failed: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
        } else if (strcmp(control, "export") != 0) {
            GGML_LOG_ERROR("ggml_vulkan: GGML_VK_NPU_CONTROL must be export, import, segmented, segmented-async, or segmented-sleep\n");
            return GGML_STATUS_FAILED;
        }
        const auto begin = std::chrono::steady_clock::now();
        const auto status = ggml_backend_vk_graph_compute_gpu(backend, cgraph);
        if (profile) {
            GGML_LOG_INFO("ggml_vulkan: Q5_K graph mode=%s cuts=0 cpu_total=%.3f ms\n",
                          control, ggml_vk_npu::elapsed(begin, std::chrono::steady_clock::now()));
        }
        return status;
    }
    const auto graph_begin = std::chrono::steady_clock::now();
    double submit_ms = 0, drain_ms = 0, execute_ms = 0;
    int cuts = 0;
    const uint64_t saved_flops = ctx->last_total_flops;
    int start = 0;
    auto gpu_range = [&](int end) {
        if (start == end) return GGML_STATUS_SUCCESS;
        ctx->last_total_flops = saved_flops;
        ggml_cgraph part = *cgraph;
        part.nodes = cgraph->nodes + start;
        part.n_nodes = end - start;
        const auto begin = std::chrono::steady_clock::now();
        const auto status = ggml_backend_vk_graph_compute_gpu(backend, &part);
        submit_ms += ggml_vk_npu::elapsed(begin, std::chrono::steady_clock::now());
        return status;
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * node = cgraph->nodes[i];
        if (profile && node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_Q5_K &&
            node->src[0]->ne[0] == 2560 && node->src[0]->ne[1] == 10240 && !ggml_vk_npu_matches(node)) {
            auto * w = node->src[0];
            auto * a = node->src[1];
            GGML_LOG_INFO("ggml_vulkan: Q5_K split skip %s dst=[%lld,%lld,%lld,%lld] flags=%d acc=%d srcprec=%d wop=%s buffers=%s/%s/%s a=[%lld,%lld,%lld,%lld]\n",
                          w->name, (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3],
                          node->flags, ggml_get_op_params_i32(node, 0), ggml_get_op_params_i32(node, 3), ggml_op_name(w->op),
                          w->buffer ? ggml_backend_buffer_name(w->buffer) : "none", a->buffer ? ggml_backend_buffer_name(a->buffer) : "none",
                          node->buffer ? ggml_backend_buffer_name(node->buffer) : "none",
                          (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3]);
        }
        if (!ggml_vk_npu_matches(node)) continue;
        const auto range_begin = std::chrono::steady_clock::now();
        const auto status = gpu_range(i);
        if (status != GGML_STATUS_SUCCESS) return status;
        const auto submitted = std::chrono::steady_clock::now();
        if (segmented && !asynchronous) {
            ctx->npu_blocking_wait = sleeping;
            ggml_vk_synchronize(ctx);
            ctx->npu_blocking_wait = false;
            ggml_vk_graph_cleanup(ctx);
        }
        const auto drained = std::chrono::steady_clock::now();
        drain_ms += ggml_vk_npu::elapsed(submitted, drained);
        ++cuts;
        try {
            if (segmented) {
                start = i;
                const auto gpu_status = gpu_range(i + 1);
                if (gpu_status != GGML_STATUS_SUCCESS) return gpu_status;
                start = i + 1;
            } else {
                if (!ctx->npu) ctx->npu = std::make_shared<ggml_vk_npu>(backend, directory);
                start = ctx->npu->execute(backend, cgraph, node) ? i + 1 : i;
            }
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml_vulkan: Q5_K split failed: %s\n", e.what());
            return GGML_STATUS_FAILED;
        }
        const auto end = std::chrono::steady_clock::now();
        if (!segmented) execute_ms += ggml_vk_npu::elapsed(drained, end);
        if (profile) {
            GGML_LOG_INFO("ggml_vulkan: Q5_K boundary %s mode=%s range_submit=%.3f range_drain=%.3f dispatch=%.3f ms\n",
                          node->src[0]->name, segmented ? control : "hybrid", ggml_vk_npu::elapsed(range_begin, submitted),
                          ggml_vk_npu::elapsed(submitted, drained), ggml_vk_npu::elapsed(drained, end));
        }
    }
    const auto status = gpu_range(cgraph->n_nodes);
    if (start != 0) {
        ctx->last_total_flops = 0;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ctx->last_total_flops += ggml_vk_get_node_flops(cgraph->nodes[i]);
        }
    }
    if (profile && cuts) {
        GGML_LOG_INFO("ggml_vulkan: Q5_K graph mode=%s cuts=%d range_submit=%.3f range_drain=%.3f execute=%.3f cpu_total=%.3f ms\n",
                      segmented ? control : "hybrid", cuts, submit_ms, drain_ms, execute_ms,
                      ggml_vk_npu::elapsed(graph_begin, std::chrono::steady_clock::now()));
    }
    return status;
}
