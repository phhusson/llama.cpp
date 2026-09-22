// Local NPU1 experiment. Included after the Vulkan graph implementation.

static bool ggml_vk_npu_matches(const ggml_tensor * node, int width = 10240, int inner = 2560, ggml_type type = GGML_TYPE_Q5_K) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        node->type != GGML_TYPE_F32 || node->ne[0] != width || node->ne[1] < 512 || node->ne[1] > 4096 || node->ne[1] % 512 ||
        node->ne[2] != 1 || node->ne[3] != 1 || !ggml_is_contiguous(node)) {
        return false;
    }
    const auto * w = node->src[0];
    const auto * a = node->src[1];
    const int32_t acc = ggml_get_op_params_i32(node, 0);
    if ((acc != GGML_PREC_DEFAULT && acc != GGML_PREC_BF16) || ggml_get_op_params_i32(node, 3) == GGML_PREC_F32) {
        return false;
    }
    if (!w || !a || w->type != type || w->op != GGML_OP_NONE ||
        w->ne[0] != inner || w->ne[1] != width || w->ne[2] != 1 || w->ne[3] != 1 ||
        a->type != GGML_TYPE_F32 || a->ne[0] != inner || a->ne[1] != node->ne[1] ||
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

static bool ggml_vk_npu_can_fuse_swiglu(ggml_backend_vk_context * ctx, const ggml_cgraph * graph, int projection_idx, int join_idx, const ggml_tensor * partner) {
    if (!getenv("GGML_VK_NPU_FFN_JOIN") || !ctx->device->pipeline_npu_swiglu || join_idx >= graph->n_nodes) return false;
    const auto * projection = graph->nodes[projection_idx];
    const auto * join = graph->nodes[join_idx];
    if (join->op != GGML_OP_GLU || ggml_get_glu_op(join) != GGML_GLU_OP_SWIGLU || join->src[2] ||
        !((join->src[0] == projection && join->src[1] == partner) || (join->src[1] == projection && join->src[0] == partner))) return false;
    const int indices[] = {projection_idx, join_idx};
    const ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_GLU};
    if (!ggml_can_fuse_subgraph_ext(graph, indices, 2, ops, &join_idx, 1)) return false;
    for (const auto * tensor : {projection, partner, join}) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_are_same_shape(tensor, projection) || !ggml_is_contiguous(tensor) ||
            !tensor->buffer || !ggml_backend_buffer_is_vk(tensor->buffer) || get_misalign_bytes(ctx, tensor)) return false;
        const auto buffer = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context)->dev_buffer;
        if (buffer->device != ctx->device) return false;
    }
    return !ggml_vk_tensors_overlap(projection, join, true) && !ggml_vk_tensors_overlap(partner, join, true);
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
    int M = 0;
    const ggml_type weight_type;
    const bool iq;
    const int K;
    const bool gate;
    const bool q6;
    const int N;
    int npu_n = 0;
    bool packed_i8 = false;
    bool command_stream = false;
    std::string hardware_program;
    xrt::device device{0};
    xrt::hw_context hw_context;
    xrt::kernel kernel;
    xrt::bo instructions;
    xrt::bo commands;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> scratch{nullptr, ggml_backend_buffer_free};
    ggml_tensor * a = nullptr;
    ggml_tensor * c = nullptr;
    ggml_tensor * a_verify = nullptr;
    xrt::bo a_bo, c_bo;
    uint32_t instruction_bytes = 0;
    struct dispatch {
        ggml_tensor cast{}, saved_input{}, gpu_w{}, gpu_c{}, merge{}, reference_op{};
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
    uint64_t calls = 0, packed_calls = 0;
    double total_ms = 0, prepare_ms = 0, npu_join_ms = 0, finish_ms = 0;
    const bool profile = getenv("GGML_VK_NPU_PROFILE") != nullptr;
    const bool verify = getenv("GGML_VK_NPU_VERIFY") != nullptr;
    nlohmann::json scale_table;
    vk_device profile_device;

    ggml_vk_npu(ggml_backend_t backend, const std::string & directory, bool is_gate = getenv("GGML_VK_NPU_GATE") != nullptr,
                const char * scale_path = getenv("GGML_VK_NPU_SCALE_TABLE"), ggml_type type = GGML_TYPE_Q5_K, int width = 0, int inner = 0) :
                weight_type(type), iq(type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_S), K(inner ? inner : iq ? 5120 : type == GGML_TYPE_Q6_K ? 6144 : 2560), gate(is_gate), q6(type == GGML_TYPE_Q6_K), N(width ? width : q6 ? 2560 : gate ? 6144 : 10240) {
        const std::string basename = weight_type == GGML_TYPE_IQ2_S ? "/iq2_s_mm" : weight_type == GGML_TYPE_IQ3_XXS ? "/iq3_xxs_mm" : weight_type == GGML_TYPE_IQ3_S ? "/iq3_s_mm" : weight_type == GGML_TYPE_IQ4_XS ? "/iq4_xs_mm" : q6 ? "/q6_k_mm" : "/q5_k_mm";
        std::ifstream manifest_file(directory + basename + ".json");
        if (!manifest_file) throw std::runtime_error("Dense NPU: missing manifest");
        const auto manifest = nlohmann::json::parse(manifest_file);
        const auto & params = manifest.at("parameters");
        M = params.at("M").get<int>();
        npu_n = params.at("N").get<int>();
        packed_i8 = params.at("math") == "i8-packed";
        command_stream = params.value("command_stream", false);
        if (manifest.at("schema_version") != 1 || manifest.at("target") != "npu1" ||
            manifest.at("kernel_name") != "MLIR_AIE" || M < 512 || M > 4096 || M % 512 || params.at("K") != K ||
            npu_n <= 0 || npu_n > N || (gate && npu_n != N) || (!gate && (!iq || K == 17408) && npu_n == N) || npu_n % 256 ||
            (!packed_i8 && params.at("math") != "bf16") || (q6 && (gate || !packed_i8)) || (iq && !packed_i8) ||
            params.at("profile") != "full" || params.at("c_bf16") != true ||
            (weight_type == GGML_TYPE_Q5_K && params.at("bounded_scales") != true) || (weight_type != GGML_TYPE_Q5_K && manifest.at("scale_domain").at("native_fp16_finite") != true) || params.at("trace_size") != 0 ||
            (command_stream && !iq) || manifest.at("xrt").at("opcode") != 3 || manifest.at("xrt").at("num_host_bos") != 3 + int(command_stream) ||
            manifest.at("xrt").at("instruction_size_unit") != "bytes") {
            throw std::runtime_error("Q5_K NPU: kernel does not match the selected split or gate shape");
        }
        const size_t a_bytes = packed_i8 ? size_t(M) * K * 33 / 32 : size_t(M) * K * 2;
        const size_t b_bytes = size_t(npu_n) * ggml_row_size(weight_type, K);
        const size_t c_bytes = size_t(M) * npu_n * 2;
        if (scale_path) {
            std::ifstream input(scale_path);
            scale_table = nlohmann::json::parse(input);
            if (scale_table.at("schema_version") != 1 || scale_table.at("K") != K ||
                scale_table.at("N") != N || scale_table.at("npu_n") != npu_n) {
                throw std::runtime_error("Q5_K NPU: scale table does not match the split");
            }
        }
        const auto & buffers = manifest.at("buffers");
        if (buffers.size() != 3 + size_t(command_stream) || buffers[0].at("bytes") != a_bytes || buffers[1].at("bytes") != b_bytes ||
            buffers[2].at("bytes") != c_bytes || buffers[0].at("dtype") != (packed_i8 ? "int8" : "bf16") ||
            buffers[1].at("dtype") != "uint8" || buffers[2].at("dtype") != "bf16" ||
            buffers[0].at("argument") != 3 || buffers[1].at("argument") != 4 || buffers[2].at("argument") != 5) {
            throw std::runtime_error("Q5_K NPU: incompatible buffer ABI");
        }
        if (packed_i8 && (buffers[0].at("layout") != "k256-m8-i8-dyadic-v1" || manifest.at("scale_domain").at("combined_int8_scale_exponent") != nlohmann::json::array({-126, 104}))) {
            throw std::runtime_error("Q5_K NPU: incompatible packed activation layout");
        }

        bool shared = false;
        if (command_stream) {
            hardware_program = manifest.at("hardware_program_sha256").get<std::string>();
            if (hardware_program.size() != 64) throw std::runtime_error("Dense NPU: invalid hardware program identifier");
            auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
            for (const auto & peer : {ctx->npu_iq4_xs, ctx->npu_iq4_xs_ffn, ctx->npu_iq4_xs_down, ctx->npu_iq3_s, ctx->npu_iq3_s_ffn, ctx->npu_iq3_s_down, ctx->npu_iq3_xxs, ctx->npu_iq3_xxs_ffn, ctx->npu_iq3_xxs_down, ctx->npu_iq4_xs_qkv, ctx->npu_iq3_s_qkv, ctx->npu_iq3_xxs_qkv, ctx->npu_iq4_xs_out, ctx->npu_iq3_s_out, ctx->npu_iq3_xxs_out, ctx->npu_iq2_s, ctx->npu_iq2_s_ffn, ctx->npu_iq2_s_down}) {
                if (peer && peer->weight_type == weight_type && peer->hardware_program == hardware_program) {
                    hw_context = peer->hw_context;
                    kernel = peer->kernel;
                    shared = true;
                    break;
                }
            }
        }
        if (!shared) {
            xrt::xclbin binary(directory + basename + ".xclbin");
            device.register_xclbin(binary);
            hw_context = xrt::hw_context(device, binary.get_uuid());
            kernel = xrt::kernel(hw_context, "MLIR_AIE");
        }
        if (command_stream) {
            const int cols = params.at("cols"), rows = params.at("rows"), m = params.at("m"), n = params.at("n");
            if (cols <= 0 || cols > 16 || rows <= 0 || rows > 16 / cols || m <= 0 || m > M || n <= 0 || n > npu_n ||
                M % (m * rows) || npu_n % (n * cols) || K % 512) throw std::runtime_error("Dense NPU: invalid command geometry");
            const size_t stride = size_t(n) * 2 * ggml_type_size(weight_type);
            const size_t bytes = cols * stride;
            const int repeats = (M / (m * rows)) * (npu_n / (n * cols));
            if (repeats <= 0 || repeats > UINT16_MAX || K / 512 > UINT16_MAX || buffers[3].at("bytes") != bytes ||
                buffers[3].at("dtype") != "uint8" || buffers[3].at("argument") != 6 || buffers[3].at("layout") != "b-stream-u16-repeat-k512-v1") {
                throw std::runtime_error("Dense NPU: invalid command buffer ABI");
            }
            commands = xrt::bo(device, bytes, xrt::bo::flags::host_only, kernel.group_id(6));
            auto * data = commands.map<uint8_t *>();
            memset(data, 0, bytes);
            const uint16_t header[2] = {uint16_t(repeats), uint16_t(K / 512)};
            for (size_t offset = 0; offset < bytes; offset += stride) memcpy(data + offset, header, sizeof(header));
            commands.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            GGML_LOG_INFO("ggml_vulkan: %s %s hardware context for command-stream kernel\n", label(), shared ? "reuses" : "creates");
        }
        std::ifstream stream(directory + basename + ".bin", std::ios::binary | std::ios::ate);
        if (!stream || stream.tellg() <= 0) throw std::runtime_error("Q5_K NPU: missing instructions");
        const size_t bytes = size_t(stream.tellg());
        if (bytes % 4 || bytes > UINT32_MAX || manifest.at("artifacts").at("bin").at("bytes") != bytes) {
            throw std::runtime_error("Q5_K NPU: invalid instruction size");
        }
        instructions = xrt::bo(device, bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
        stream.seekg(0);
        if (!stream.read(instructions.map<char *>(), bytes)) throw std::runtime_error("Q5_K NPU: cannot read instructions");
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        tensors.reset(ggml_init({ggml_tensor_overhead() * (verify ? 3 : 2), nullptr, true}));
        if (!tensors) throw std::bad_alloc();
        a = packed_i8 ? ggml_new_tensor_1d(tensors.get(), GGML_TYPE_I8, a_bytes) : ggml_new_tensor_2d(tensors.get(), GGML_TYPE_BF16, K, M);
        c = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_BF16, npu_n, M);
        if (verify) a_verify = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, K, M);
        ggml_set_name(a, packed_i8 ? "npu_q5_input_i8" : "npu_q5_input_bf16");
        ggml_set_name(c, "npu_q5_output_bf16");
        scratch.reset(ggml_backend_alloc_ctx_tensors(tensors.get(), backend));
        if (!scratch) throw std::bad_alloc();
        auto buffer = static_cast<ggml_backend_vk_buffer_context *>(scratch->context)->dev_buffer;
        auto & imported = ggml_vk_npu_import(buffer, device);
        a_bo = xrt::bo(imported, a_bytes, vk_tensor_offset(a));
        c_bo = xrt::bo(imported, c_bytes, vk_tensor_offset(c));
        instruction_bytes = uint32_t(bytes);
        profile_device = static_cast<ggml_backend_vk_context *>(backend->context)->device;
        if (packed_i8 && !profile_device->pipeline_npu_q5_pack) throw std::runtime_error("Q5_K NPU: missing activation packing pipeline");
        GGML_LOG_INFO("ggml_vulkan: %s split M=%d K=%d N=%d: NPU=%d GPU=%d, original weights via dma-buf, %s\n",
                      label(), M, K, N, npu_n, N - npu_n, getenv("GGML_VK_NPU_CONTROL") ? "import only control" : "dma-buf fences in both directions");
    }

    const char * label() const {
        return weight_type == GGML_TYPE_IQ2_S ? "IQ2_S" : weight_type == GGML_TYPE_IQ3_XXS ? "IQ3_XXS" : weight_type == GGML_TYPE_IQ3_S ? "IQ3_S" : weight_type == GGML_TYPE_IQ4_XS ? "IQ4_XS" : q6 ? "Q6_K" : "Q5_K";
    }

    ~ggml_vk_npu() {
        if (calls) {
            GGML_LOG_INFO("ggml_vulkan: %s split %llu calls, mean CPU call %.3f ms (prepare %.3f, NPU dispatch %.3f, merge submit %.3f)\n",
                          label(), (unsigned long long) calls, total_ms / calls, prepare_ms / calls, npu_join_ms / calls, finish_ms / calls);
        }
        if (packed_calls) GGML_LOG_INFO("ggml_vulkan: %s consumes %llu prepacked inputs\n", label(), (unsigned long long) packed_calls);
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
            GGML_LOG_INFO("ggml_vulkan: %s device %s cast=%.3f gpu_work=%.3f merge=%.3f cast_to_work=%.3f work_to_merge=%.3f gpu_started_at_dispatch=%d gpu_done_at_dispatch=%d ms\n",
                          label(), job.gpu_w.name, (ticks[1] - ticks[0]) * ms, (ticks[3] - ticks[2]) * ms, (ticks[5] - ticks[4]) * ms,
                          (ticks[2] - ticks[1]) * ms, (ticks[4] - ticks[3]) * ms, job.gpu_started, job.gpu_done);
        } else {
            GGML_LOG_WARN("ggml_vulkan: Q5_K device timestamps unavailable: %s\n", vk::to_string(result).c_str());
        }
        job.profile_pending = false;
    }

    bool check_scales(vk_buffer & buffer, size_t offset, const char * name) {
        const size_t bytes = size_t(npu_n) * ggml_row_size(weight_type, K);
        std::lock_guard<std::mutex> lock(buffer->npu_mutex);
        const auto key = std::make_pair(offset, bytes);
        const auto cached = buffer->npu_scales.find(key);
        if (cached != buffer->npu_scales.end()) return cached->second.valid;
        if (!scale_table.is_null()) {
            const auto & entries = scale_table.at("tensors");
            const bool valid = entries.contains(name) && entries.at(name).at("eligible").get<bool>();
            buffer->npu_scales.emplace(key, vk_buffer_struct::npu_weight_state{valid, true});
            if (!valid) GGML_LOG_WARN("ggml_vulkan: %s NPU scale table excludes %s; using Vulkan\n", label(), name);
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
        for (size_t i = 0; i < bytes; i += ggml_type_size(weight_type)) {
            if (weight_type != GGML_TYPE_Q5_K) {
                uint16_t scale;
                memcpy(&scale, data + i + (q6 ? 208 : 0), sizeof(scale));
                if ((scale & 0x7c00) == 0x7c00) {
                    valid = false;
                    break;
                }
                continue;
            }
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
        if (!valid) GGML_LOG_WARN("ggml_vulkan: %s NPU scale limits exceeded for %s; using Vulkan\n", label(), name);
        return valid;
    }

    static double elapsed(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    void verify_iq(const ggml_tensor * weights, const std::vector<float> & actual, const std::vector<uint8_t> & packed) const {
        const std::array<int, 16> columns = {0, 1, 15, 16, 31, 32, 63, 64, npu_n / 2 - 1, npu_n / 2, npu_n - 65, npu_n - 64, npu_n - 33, npu_n - 32, npu_n - 2, npu_n - 1};
        const std::array<int, 8> rows = {0, 1, 7, 8, M / 2 - 1, M / 2, M - 2, M - 1};
        std::vector<uint8_t> raw(weights->nb[1]);
        std::vector<float> native(K), scales(K / 256);
        std::vector<int8_t> codes(K);
        for (int column : columns) {
            ggml_backend_tensor_get(weights, raw.data(), size_t(column) * weights->nb[1], raw.size());
            ggml_get_type_traits(weight_type)->to_float(raw.data(), native.data(), K);
            for (int block = 0; block < K / 256; ++block) {
                const auto * q = raw.data() + block * ggml_type_size(weight_type);
                uint16_t hd, high;
                memcpy(&hd, q, sizeof(hd));
                memcpy(&high, q + 2, sizeof(high));
                int peak = 0;
                for (int group = 0; group < (weight_type == GGML_TYPE_IQ2_S ? 16 : 8); ++group) {
                    int scale;
                    if (weight_type == GGML_TYPE_IQ2_S) {
                        scale = 1 + 2 * ((q[74 + group / 2] >> (4 * (group % 2))) & 15);
                    } else if (weight_type == GGML_TYPE_IQ3_XXS) {
                        uint32_t word;
                        memcpy(&word, q + 66 + group * 4, sizeof(word));
                        scale = 1 + 2 * (word >> 28);
                    } else {
                        scale = weight_type == GGML_TYPE_IQ3_S ? 1 + 2 * ((q[106 + group / 2] >> (4 * (group % 2))) & 15) :
                            int(((q[4 + group / 2] >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4)) - 32;
                    }
                    peak = std::max(peak, std::abs(scale));
                }
                const double grid_scale = weight_type == GGML_TYPE_IQ2_S ? 43.0 / (8 * 127.0) : weight_type == GGML_TYPE_IQ3_XXS ? 15.5 / 127.0 : weight_type == GGML_TYPE_IQ3_S ? 15.0 / 127.0 : 1.0;
                const double magnitude = std::abs(double(ggml_fp16_to_fp32(hd))) * peak * grid_scale;
                const int minimum = std::max(1, int((hd >> 10) & 31)) - (weight_type == GGML_TYPE_IQ2_S ? 28 : weight_type == GGML_TYPE_IQ3_XXS ? 27 : 25);
                const int exponent = std::max(minimum, int(std::ceil(std::log2(std::max(magnitude, std::ldexp(1.0, -100))))));
                scales[block] = std::ldexp(1.0f, exponent);
                for (int k = 0; k < 256; ++k) codes[block * 256 + k] = int8_t(std::clamp(int(std::nearbyint(double(native[block * 256 + k]) / scales[block])), -127, 127));
            }
            for (int row : rows) {
                float value = 0;
                for (int block = 0; block < K / 256; ++block) {
                    const size_t base = (size_t(block) * (M / 8) + row / 8) * 2112;
                    float scale;
                    memcpy(&scale, packed.data() + base + 2048 + (row % 8) * 4, sizeof(scale));
                    int32_t dot = 0;
                    for (int k = 0; k < 256; ++k) dot += int8_t(packed[base + k * 8 + row % 8]) * int32_t(codes[block * 256 + k]);
                    const float term = (float(dot) * scales[block]) * scale;
                    value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value + term));
                }
                if (actual[size_t(row) * N + column] != value) throw std::runtime_error("Dense IQ NPU: sampled output differs from the INT8 reference");
            }
        }
        GGML_LOG_INFO("ggml_vulkan: %s 128 sampled outputs match the CPU INT8 reference exactly\n", label());
    }

    bool execute(ggml_backend_t backend, const ggml_cgraph * graph, ggml_tensor * node, ggml_cgraph * parallel = nullptr,
                 ggml_vk_npu * split = nullptr, const std::function<void()> & on_started = {}, ggml_tensor * swiglu = nullptr,
                 ggml_tensor * packed_output = nullptr, bool input_packed = false) {
        if (node->ne[1] != M) return false;
        auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
        const bool timestamps = profile;
        GGML_ASSERT(node->ne[0] == N && npu_n <= N && (parallel || npu_n < N));
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
        const bool output_half = q6 && !parallel && profile_device->coopmat_support && !profile_device->coopmat2 && profile_device->pipeline_npu_q5_pack_half;
        const uint64_t half_size = uint64_t(M) * K * sizeof(ggml_fp16_t);
        if (output_half && ctx->prealloc_size_y < half_size) {
            ctx->prealloc_size_y = half_size;
            auto subctx = ggml_vk_get_compute_ctx(ctx);
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
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
        if (command_stream) job.run.set_arg(6, commands);
        vk::ExportSemaphoreCreateInfo export_info{vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd};
        vk::SemaphoreCreateInfo semaphore_info{};
        semaphore_info.setPNext(&export_info);
        job.ready = profile_device->device.createSemaphoreUnique(semaphore_info);
        job.completion = profile_device->device.createSemaphoreUnique({});
        const auto bound = std::chrono::steady_clock::now();
        if (timestamps) {
            job.queries = profile_device->device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 6});
            profile_device->device.resetQueryPool(*job.queries, 0, 6);
            ctx->npu_profile_pool = *job.queries;
        }

        auto gpu_node = [&](ggml_tensor * tensor, int query = -1, const ggml_tensor * half_input = nullptr) {
            const uint64_t saved_flops = ctx->last_total_flops;
            ggml_cgraph part = *graph;
            part.nodes = &tensor;
            part.n_nodes = 1;
            ctx->npu_profile_query = timestamps ? query : -1;
            const auto status = ggml_backend_vk_graph_compute_gpu(backend, &part, half_input);
            ctx->npu_profile_query = -1;
            ctx->last_total_flops = saved_flops;
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Q5_K split: Vulkan graph failed");
        };

        auto & cast = job.cast;
        cast = *a;
        cast.op = GGML_OP_CPY;
        cast.flags = GGML_TENSOR_FLAG_COMPUTE;
        cast.src[0] = node->src[1];
        vk_buffer packed_half;
        if (input_packed) {
            GGML_ASSERT(packed_i8 && !verify && !output_half);
            if (timestamps) {
                auto subctx = ggml_vk_get_compute_ctx(ctx);
                subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 0);
                subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 1);
            }
        } else if (packed_i8) {
            auto subctx = ggml_vk_get_compute_ctx(ctx);
            ggml_vk_sync_buffers(ctx, subctx);
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            if (timestamps) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 0);
            auto pipeline = output_half ? profile_device->pipeline_npu_q5_pack_half : profile_device->pipeline_npu_q5_pack;
            ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
            const std::array<uint32_t, 2> pc = {uint32_t(M), uint32_t(K)};
            if (output_half) {
                packed_half = ctx->prealloc_y;
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                    {ggml_vk_tensor_subbuffer(ctx, node->src[1]), ggml_vk_tensor_subbuffer(ctx, a), vk_subbuffer{packed_half, 0, half_size}}, pc, {uint32_t(M / 8 * 256), uint32_t(K / 256), 1});
                ggml_vk_sync_buffers(ctx, subctx);
            } else {
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                    {ggml_vk_tensor_subbuffer(ctx, node->src[1]), ggml_vk_tensor_subbuffer(ctx, a)}, pc, {uint32_t(M / 8 * 256), uint32_t(K / 256), 1});
            }
            ctx->unsynced_nodes_read.push_back(node->src[1]);
            ctx->unsynced_nodes_written.push_back(&cast);
            if (timestamps) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 1);
        } else {
            gpu_node(&cast, 0);
        }
        if (verify) {
            // Parallel work can reuse the original activation buffer.
            auto & saved_input = job.saved_input;
            saved_input = *a_verify;
            saved_input.op = GGML_OP_CPY;
            saved_input.flags = GGML_TENSOR_FLAG_COMPUTE;
            saved_input.src[0] = node->src[1];
            gpu_node(&saved_input);
        }
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

        if (npu_n < N) {
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
        }

        const auto launch = std::chrono::steady_clock::now();
        double gpu_ms = 0;
        double start_ms = 0;
        try {
            bool launched = false;
            auto start_job = [&] {
                const auto before_start = std::chrono::steady_clock::now();
                job.run.start();
                start_ms = elapsed(before_start, std::chrono::steady_clock::now());
                job.running = true;
                import_completion(job, storage);
                launched = true;
                if (on_started) on_started();
            };
            ggml_cgraph remaining{};
            std::vector<ggml_tensor *> remaining_nodes;
            if (parallel) {
                remaining = *parallel;
                for (int i = 0; i < parallel->n_nodes; ++i) {
                    auto * tensor = parallel->nodes[i];
                    if (tensor != node) remaining_nodes.push_back(tensor);
                    else if (npu_n < N) remaining_nodes.push_back(&job.gpu_c);
                }
                remaining.nodes = remaining_nodes.data();
                remaining.n_nodes = remaining_nodes.size();
            }
            if (split && remaining.n_nodes && ggml_vk_npu_matches(remaining.nodes[0], split->N, split->K, split->weight_type) &&
                split->execute(backend, graph, remaining.nodes[0], nullptr, nullptr, start_job)) {
                // Submit QKV before the gate; consume its fence before the recurrent branch.
                ++remaining.nodes;
                --remaining.n_nodes;
                GGML_ASSERT(launched);
            }
            if (!launched) start_job();
            if (timestamps) ctx->npu_profile_pool = *job.queries;
            const auto gpu_begin = std::chrono::steady_clock::now();
            if (parallel) {
                if (timestamps) {
                    ggml_format_name(&job.gpu_w, "%s", w->name);
                    ctx->npu_profile_query = 2;
                    ctx->npu_profile_span = true;
                    ctx->npu_profile_first = true;
                }
                const auto status = ggml_backend_vk_graph_compute_gpu(backend, &remaining);
                ctx->npu_profile_query = -1;
                ctx->npu_profile_span = false;
                if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Q5_K gate: parallel Vulkan graph failed");
            } else {
                gpu_node(&job.gpu_c, 2, packed_half && packed_half == ctx->prealloc_y ? node->src[1] : nullptr);
            }
            gpu_ms = elapsed(gpu_begin, std::chrono::steady_clock::now());
        } catch (...) {
            ctx->npu_profile_query = -1;
            ctx->npu_profile_span = false;
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
        if (timestamps) {
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
        if (swiglu) {
            GGML_ASSERT(!verify && c->type == GGML_TYPE_BF16);
            merge = *swiglu;
            // The local GLU clone reads the NPU prefix as its third input.
            merge.src[2] = c;
            ggml_set_op_params_i32(&merge, 4, swiglu->src[0] == node);
            ggml_format_name(&merge, "%s_npu_join", swiglu->name);
        } else {
            merge = *node;
            merge.op = GGML_OP_CPY;
            merge.ne[0] = npu_n;
            std::fill(std::begin(merge.src), std::end(merge.src), nullptr);
            merge.src[0] = c;
            ggml_format_name(&merge, "%s_npu", node->name);
        }
        if (packed_output) {
            GGML_ASSERT(swiglu && !verify && packed_output->type == GGML_TYPE_I8 && ggml_nbytes(packed_output) == size_t(M) * N * 33 / 32);
            auto subctx = ggml_vk_get_compute_ctx(ctx);
            ggml_vk_sync_buffers(ctx, subctx);
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            if (timestamps) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 4);
            auto * partner = swiglu->src[swiglu->src[0] == node ? 1 : 0];
            const std::array<uint32_t, 4> pc = {uint32_t(M), uint32_t(N), uint32_t(npu_n), uint32_t(swiglu->src[0] == node)};
            auto pipeline = profile_device->pipeline_npu_swiglu_pack;
            ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {ggml_vk_tensor_subbuffer(ctx, node), ggml_vk_tensor_subbuffer(ctx, partner), ggml_vk_tensor_subbuffer(ctx, c),
                 ggml_vk_tensor_subbuffer(ctx, &merge), ggml_vk_tensor_subbuffer(ctx, packed_output)}, pc, {uint32_t(N), uint32_t(M / 8), 1});
            ctx->unsynced_nodes_read.insert(ctx->unsynced_nodes_read.end(), {node, partner, c});
            ctx->unsynced_nodes_written.insert(ctx->unsynced_nodes_written.end(), {&merge, packed_output});
            if (timestamps) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 5);
        } else {
            gpu_node(&merge, 4);
        }
        if (timestamps) {
            job.profile_pending = true;
        }
        const auto end = std::chrono::steady_clock::now();
        ++calls;
        if (input_packed) ++packed_calls;
        total_ms += elapsed(begin, end);
        prepare_ms += elapsed(begin, launch);
        npu_join_ms += elapsed(launch, joined);
        finish_ms += elapsed(joined, end);
        if (profile) {
            GGML_LOG_INFO("ggml_vulkan: %s split %s prepare=%.3f npu_dispatch=%.3f gpu_submit=%.3f merge_submit=%.3f cpu_total=%.3f ms\n",
                          label(), w->name, elapsed(begin, launch), elapsed(launch, joined), gpu_ms, elapsed(joined, end), elapsed(begin, end));
            GGML_LOG_INFO("ggml_vulkan: %s stages %s check=%.3f bind=%.3f cast_submit=%.3f release_submit=%.3f views=%.3f xrt_start=%.3f acquire=%.3f merge=%.3f ms\n",
                          label(), w->name, elapsed(begin, checked), elapsed(checked, bound), elapsed(bound, cast_submitted),
                          elapsed(cast_submitted, released), elapsed(released, launch), start_ms,
                          elapsed(joined, acquired), elapsed(acquired, end));
        }
        if (verify) {
            std::vector<float> actual(M * N), reference(M * N);
            ggml_backend_tensor_get(node, actual.data(), 0, ggml_nbytes(node));
            if (packed_i8) {
                std::vector<float> input(size_t(M) * K);
                std::vector<uint8_t> packed(ggml_nbytes(a)), expected(packed.size(), 0);
                ggml_backend_tensor_get(a_verify, input.data(), 0, ggml_nbytes(a_verify));
                ggml_backend_tensor_get(a, packed.data(), 0, packed.size());
                for (int row = 0; row < M; ++row) for (int block = 0; block < K / 256; ++block) {
                    float values[256];
                    double peak = std::ldexp(1.0, -60);
                    for (int k = 0; k < 256; ++k) {
                        values[k] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(input[size_t(row) * K + block * 256 + k]));
                        if (!std::isfinite(values[k])) throw std::runtime_error("Q5_K NPU: nonfinite activation");
                        peak = std::max(peak, double(std::abs(values[k])));
                    }
                    const int exponent = int(std::ceil(std::log2(peak / 127.0)));
                    if (exponent > (iq && weight_type != GGML_TYPE_IQ4_XS ? 86 : weight_type == GGML_TYPE_IQ4_XS ? 83 : q6 ? 82 : 84)) throw std::runtime_error("Dense NPU: activation scale exceeds validated domain");
                    const float scale = std::ldexp(1.0f, exponent);
                    const size_t base = (size_t(block) * (M / 8) + row / 8) * 2112;
                    memcpy(expected.data() + base + 2048 + (row % 8) * 4, &scale, sizeof(scale));
                    for (int k = 0; k < 256; ++k) {
                        const int value = int(std::nearbyint(double(values[k]) / scale));
                        expected[base + k * 8 + row % 8] = uint8_t(std::max(-127, std::min(127, value)));
                    }
                }
                if (packed != expected) throw std::runtime_error("Q5_K NPU: packed activation bytes differ from CPU reference");
                GGML_LOG_INFO("ggml_vulkan: %s packed activation bytes match CPU reference\n", label());
                if (iq) verify_iq(w, actual, expected);
            }
            job.reference_op = *node;
            job.reference_op.src[1] = a_verify;
            gpu_node(&job.reference_op);
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
            if (iq) {
                for (int m = 0; m < M; ++m) for (int n = npu_n; n < N; ++n) {
                    const size_t i = size_t(m) * N + n;
                    if (actual[i] != reference[i]) throw std::runtime_error("Dense IQ NPU: Vulkan remainder differs from the full projection");
                }
            }
            GGML_LOG_INFO("ggml_vulkan: %s verify %s offset=%zu relative_l2=%.6g\n", label(), w->name, weight_offset, relative);
            if (!std::isfinite(relative) || (!iq && relative > (q6 ? .03 : .02))) throw std::runtime_error("Dense NPU: model projection verification failed");
        }
        return true;
    }
};

static uint32_t ggml_vk_npu_iq1_slots() {
    static const uint32_t slots = [] {
        const char * value = getenv("GGML_VK_NPU_IQ1_S_SLOTS");
        const int count = value ? std::stoi(value) : 1;
        if (count < 1 || count > 24) throw std::runtime_error("IQ1_S NPU: slots must be between 1 and 24");
        return uint32_t(count);
    }();
    return slots;
}

static bool ggml_vk_npu_iq1_can_run(ggml_backend_vk_context * ctx, const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids, const ggml_tensor * output, const ggml_tensor * up) {
    if (!ctx->device->pipeline_moe_npu_pair_buckets[0] || !up || weights->type != GGML_TYPE_IQ1_S || up->type != GGML_TYPE_IQ1_S ||
        !ggml_are_same_shape(weights, up) || weights->ne[0] != 2560 || weights->ne[1] != 640 || weights->ne[2] < ggml_vk_npu_iq1_slots() || weights->ne[2] > 512 || weights->ne[3] != 1 ||
        input->type != GGML_TYPE_F32 || input->ne[0] != 2560 || input->ne[1] != 1 || input->ne[2] != ids->ne[1] || input->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->ne[1] < 512 || ids->ne[1] > 4096 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        output->type != GGML_TYPE_F32 || output->ne[0] != 640 || output->ne[1] != ids->ne[0] || output->ne[2] != ids->ne[1] || output->ne[3] != 1) return false;
    for (const auto * tensor : {weights, input, output, up}) {
        if (!ggml_is_contiguous(tensor) || !tensor->buffer || !ggml_backend_buffer_is_vk(tensor->buffer) || get_misalign_bytes(ctx, tensor)) return false;
        const auto buffer = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context)->dev_buffer;
        if (buffer->device != ctx->device || !buffer->npu_exportable) return false;
    }
    return true;
}

static uint32_t ggml_vk_npu_iq4_slots() {
    static const uint32_t slots = [] {
        const char * value = getenv("GGML_VK_NPU_IQ4_NL_SLOTS");
        const int count = value ? std::stoi(value) : int(ggml_vk_npu_iq1_slots());
        if (count < 1 || count > 24) throw std::runtime_error("IQ4_NL NPU: slots must be between 1 and 24");
        return uint32_t(count);
    }();
    return slots;
}

static bool ggml_vk_npu_iq4_can_run(ggml_backend_vk_context * ctx, const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids, const ggml_tensor * output) {
    if (!ctx->device->pipeline_moe_npu_down_buckets[0] || weights->type != GGML_TYPE_IQ4_NL ||
        weights->ne[0] != 640 || weights->ne[1] != 2560 || weights->ne[2] != 512 || weights->ne[3] != 1 ||
        input->type != GGML_TYPE_F32 || input->ne[0] != 640 || input->ne[1] != ids->ne[0] || input->ne[2] != ids->ne[1] || input->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->ne[1] < 512 || ids->ne[1] > 4096 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        output->type != GGML_TYPE_F32 || output->ne[0] != 2560 || output->ne[1] != ids->ne[0] || output->ne[2] != ids->ne[1] || output->ne[3] != 1) return false;
    for (const auto * tensor : {weights, input, output}) {
        if (!ggml_is_contiguous(tensor) || !tensor->buffer || !ggml_backend_buffer_is_vk(tensor->buffer) || get_misalign_bytes(ctx, tensor)) return false;
        const auto buffer = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context)->dev_buffer;
        if (buffer->device != ctx->device || !buffer->npu_exportable) return false;
    }
    return true;
}

struct ggml_vk_npu_iq1 {
    const bool down;
    const uint32_t K, N, expert_bytes;
    xrt::device device{0};
    xrt::hw_context hw_context;
    xrt::kernel kernel;
    vk_device gpu;
    vk_buffer storage;
    uint32_t M = 0, slots, instruction_bytes = 0;
    uint32_t a_prefix_bytes = 0;
    bool core_dma = false, a_int8 = false, c_bf16 = false, interleaved_rows = false;
    size_t gate_offset = 0, up_offset = 0, a_bytes = 0, c_bytes = 0;
    xrt::bo instructions;
    std::vector<xrt::bo> a, gate, up;
    nlohmann::json expert_table;
    const bool profile;
    struct dispatch {
        std::array<vk_buffer, 2> weights;
        std::vector<xrt::bo> b;
        std::vector<xrt::run> runs;
        vk::UniqueSemaphore ready, completion;
        vk::UniqueQueryPool queries;
        std::string name;
        uint32_t n_experts = 0;
    };
    std::deque<dispatch> pending;
    uint64_t calls = 0;

    ggml_vk_npu_iq1(ggml_backend_vk_context * ctx, const std::string & directory, bool is_down = false) : down(is_down), K(down ? 640 : 2560), N(down ? 2560 : 640), expert_bytes(down ? 921600 : 320000), gpu(ctx->device), slots(down ? ggml_vk_npu_iq4_slots() : ggml_vk_npu_iq1_slots()), profile(getenv(down ? "GGML_VK_NPU_IQ4_NL_PROFILE" : "GGML_VK_NPU_IQ1_S_PROFILE") != nullptr) {
        const std::string basename = down ? "/iq4_nl_mm" : "/iq1_s_mm";
        std::ifstream file(directory + basename + ".json");
        if (!file) throw std::runtime_error("IQ1_S NPU: missing manifest");
        const auto manifest = nlohmann::json::parse(file);
        const auto & params = manifest.at("parameters");
        M = params.at("M").get<uint32_t>();
        core_dma = params.value("core_dma", false);
        a_int8 = params.value("a_int8", false);
        c_bf16 = params.at("c_bf16").get<bool>();
        interleaved_rows = params.value("interleaved_rows", false);
        a_prefix_bytes = params.value("a_prefix_bytes", 0u);
        const bool decoder_multicast = params.value("decoder_multicast", false);
        if (interleaved_rows && (down || !decoder_multicast || !core_dma || !a_int8 || M != 576)) throw std::runtime_error("IQ1_S NPU: incompatible interleaved rows");
        const bool valid_rows = down || decoder_multicast ? core_dma && a_int8 && (M == 576 || (down && (M == 768 || M == 1024))) : M >= 512 && M <= 4096 && M % 512 == 0;
        if (manifest.at("schema_version") != 1 || manifest.at("target") != "npu1" || manifest.at("kernel_name") != "MLIR_AIE" ||
            params.at("K") != K || params.at("N") != N || (c_bf16 && !down) || (!down && params.value("fused", false) != true && !decoder_multicast) ||
            !valid_rows || manifest.at("xrt").at("opcode") != 3 ||
            manifest.at("xrt").at("num_host_bos") != 3 || manifest.at("xrt").at("instruction_size_unit") != "bytes") {
            throw std::runtime_error("Routed NPU: incompatible native kernel geometry or output type");
        }
        const uint32_t expected_prefix = down && M == 1024 ? 8192 : 4096;
        if ((core_dma && ((!down && M != (decoder_multicast ? 576u : 512u)) || a_prefix_bytes != expected_prefix)) || (!core_dma && (a_prefix_bytes != 0 || a_int8))) throw std::runtime_error("IQ1_S NPU: incompatible DMA metadata layout");
        a_bytes = a_prefix_bytes + size_t(M) * K * (a_int8 ? 1 : 2);
        c_bytes = size_t(M) * N * (c_bf16 ? 2 : 4);
        const auto & buffers = manifest.at("buffers");
        if (buffers.size() != 3 || buffers[0].at("bytes") != a_bytes || buffers[1].at("bytes") != expert_bytes * (core_dma ? 512u : 1u) || buffers[2].at("bytes") != c_bytes ||
            buffers[0].at("dtype") != (core_dma ? "uint8" : "bf16") || buffers[1].at("dtype") != "uint8" || buffers[2].at("dtype") != (c_bf16 ? "bf16" : "float32") ||
            buffers[0].at("argument") != 3 || buffers[1].at("argument") != 4 || buffers[2].at("argument") != 5) throw std::runtime_error("IQ1_S NPU: incompatible buffers");

        std::ifstream stream(directory + basename + ".bin", std::ios::binary | std::ios::ate);
        if (!stream || stream.tellg() < 16 || stream.tellg() > 1024 * 1024 || size_t(stream.tellg()) % 4) throw std::runtime_error("IQ1_S NPU: invalid instruction size");
        instruction_bytes = uint32_t(stream.tellg());
        xrt::xclbin binary(directory + basename + ".xclbin");
        device.register_xclbin(binary);
        hw_context = xrt::hw_context(device, binary.get_uuid());
        kernel = xrt::kernel(hw_context, "MLIR_AIE");
        instructions = xrt::bo(device, instruction_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
        stream.seekg(0);
        if (!stream.read(instructions.map<char *>(), instruction_bytes)) throw std::runtime_error("IQ1_S NPU: cannot read instructions");
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (const char * path = getenv(down ? "GGML_VK_NPU_IQ4_NL_EXPERTS" : "GGML_VK_NPU_IQ1_S_EXPERTS")) {
            std::ifstream table(path);
            expert_table = nlohmann::json::parse(table);
        }
        gate_offset = slots * a_bytes;
        up_offset = gate_offset + slots * c_bytes;
        storage = ggml_vk_create_buffer_device(gpu, up_offset + (down ? 0 : slots * c_bytes));

        auto & imported = ggml_vk_npu_import(storage, device);
        for (uint32_t slot = 0; slot < slots; ++slot) {
            a.emplace_back(imported, a_bytes, slot * a_bytes);
            gate.emplace_back(imported, c_bytes, gate_offset + slot * c_bytes);
            if (!down) up.emplace_back(imported, c_bytes, up_offset + slot * c_bytes);
        }
        GGML_LOG_INFO("ggml_vulkan: %s routed NPU slots=%u rows=%u, %s activations, %s outputs, %s, native weights via dma-buf\n", down ? "IQ4_NL" : "IQ1_S", slots, M, a_int8 ? "INT8" : "BF16", c_bf16 ? "BF16" : "FP32", core_dma ? "core DMA with GPU expert selection" : "fixed expert partition");
    }

    ~ggml_vk_npu_iq1() {
        GGML_LOG_INFO("ggml_vulkan: %s routed NPU %llu FFNs, %llu %s jobs\n", down ? "IQ4_NL" : "IQ1_S", (unsigned long long) calls, (unsigned long long) (calls * slots * (down ? 1 : 2)), down ? "down" : "gate/up");
    }

    void finish() {
        for (auto & job : pending) {
            for (auto & run : job.runs) {
                const auto state = run.state();
                if (state != ERT_CMD_STATE_COMPLETED) throw std::runtime_error("IQ1_S NPU: XRT state after Vulkan consumer: " + std::to_string(state));
            }
            if (job.queries) {
                std::array<uint64_t, 6> ticks{};
                const auto result = gpu->device.getQueryPoolResults(*job.queries, 0, ticks.size(), sizeof(ticks), ticks.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
                if (result != vk::Result::eSuccess) throw std::runtime_error("IQ1_S NPU: timestamps unavailable");
                const double ms = gpu->properties.limits.timestampPeriod / 1e6;
                GGML_LOG_INFO("ggml_vulkan: %s device %s gather=%.3f dispatch_gap=%.3f gpu_rest=%.3f wait_gap=%.3f merge=%.3f ms\n",
                    down ? "IQ4_NL" : "IQ1_S", job.name.c_str(), (ticks[1] - ticks[0]) * ms, (ticks[2] - ticks[1]) * ms, (ticks[3] - ticks[2]) * ms,
                    (ticks[4] - ticks[3]) * ms, (ticks[5] - ticks[4]) * ms);
            }
        }
        pending.clear();
    }

    void handoff(vk_context & subctx, bool release) {
        vk::BufferMemoryBarrier barrier;
        barrier.buffer = storage->buffer;
        barrier.size = storage->size;
        barrier.srcAccessMask = release ? vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite : vk::AccessFlags{};
        barrier.dstAccessMask = release ? vk::AccessFlags{} : vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        barrier.srcQueueFamilyIndex = release ? gpu->compute_queue->queue_family_index : VK_QUEUE_FAMILY_FOREIGN_EXT;
        barrier.dstQueueFamilyIndex = release ? VK_QUEUE_FAMILY_FOREIGN_EXT : gpu->compute_queue->queue_family_index;
        subctx->s->buffer->buf.pipelineBarrier(release ? vk::PipelineStageFlagBits::eAllCommands : vk::PipelineStageFlagBits::eTopOfPipe,
            release ? vk::PipelineStageFlagBits::eBottomOfPipe : vk::PipelineStageFlagBits::eComputeShader, {}, {}, {barrier}, {});
    }

    void submit(ggml_backend_vk_context * ctx, vk_context & subctx) {
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_ctx_begin(gpu, subctx);
    }

    void begin(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids, const ggml_tensor * up_weights, const vk_subbuffer & experts) {
        pending.emplace_back();
        auto & job = pending.back();
        job.n_experts = weights->ne[2];
        std::vector<uint32_t> selected;
        if (expert_table.contains(weights->name)) selected = expert_table.at(weights->name).get<std::vector<uint32_t>>();
        else for (uint32_t slot = 0; slot < slots; ++slot) selected.push_back(core_dma ? UINT32_MAX : slot);
        if (selected.size() < slots) throw std::runtime_error("IQ1_S NPU: expert table has too few entries");
        selected.resize(slots);
        for (uint32_t slot = 0; slot < slots; ++slot) {
            if (core_dma && !expert_table.contains(weights->name)) continue;
            if (selected[slot] >= weights->ne[2] || std::find(selected.begin(), selected.begin() + slot, selected[slot]) != selected.begin() + slot) throw std::runtime_error("IQ1_S NPU: invalid expert partition");
        }
        const ggml_tensor * tensors[] = {weights, up_weights};
        for (uint32_t projection = 0; projection < (down ? 1u : 2u); ++projection) {
            auto range = ggml_vk_tensor_subbuffer(ctx, tensors[projection]);
            job.weights[projection] = range.buffer;
            auto & imported = ggml_vk_npu_import(range.buffer, device);
            if (core_dma) job.b.emplace_back(imported, size_t(weights->ne[2]) * expert_bytes, range.offset);
            else for (uint32_t slot = 0; slot < slots; ++slot) job.b.emplace_back(imported, expert_bytes, range.offset + size_t(selected[slot]) * expert_bytes);
        }
        for (uint32_t slot = 0; slot < slots; ++slot) for (uint32_t projection = 0; projection < (down ? 1u : 2u); ++projection) {
            job.runs.emplace_back(kernel);
            auto & run = job.runs.back();
            run.set_arg(0, uint32_t(3));
            run.set_arg(1, instructions);
            run.set_arg(2, instruction_bytes);
            run.set_arg(3, a[slot]);
            run.set_arg(4, job.b[core_dma ? projection : projection * slots + slot]);
            run.set_arg(5, projection ? up[slot] : gate[slot]);
        }
        vk::ExportSemaphoreCreateInfo export_info{vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd};
        vk::SemaphoreCreateInfo semaphore_info{};
        semaphore_info.setPNext(&export_info);
        job.ready = gpu->device.createSemaphoreUnique(semaphore_info);
        job.completion = gpu->device.createSemaphoreUnique({});
        if (profile) {
            job.name = weights->name;
            job.queries = gpu->device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 6});
            gpu->device.resetQueryPool(*job.queries, 0, 6);
            subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 0);
        }
        std::vector<uint32_t> pc = {uint32_t(weights->ne[2]), uint32_t(experts.size / 4 - slots * 4), slots, M, uint32_t(ids->ne[0])};
        pc.insert(pc.end(), selected.begin(), selected.end());
        pc.resize(32);
        pc[29] = a_prefix_bytes / 4;
        pc[30] = a_int8;
        pc[31] = interleaved_rows;
        auto & gather_pipeline = down ? gpu->pipeline_npu_iq4_gather : a_int8 ? gpu->pipeline_npu_iq1_gather_i8 : gpu->pipeline_npu_iq1_gather;
        ggml_pipeline_request_descriptor_sets(ctx, gpu->pipeline_npu_iq1_select, 1);
        ggml_pipeline_request_descriptor_sets(ctx, gather_pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, gpu->pipeline_npu_iq1_select, {experts}, pc, {slots, 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, gather_pipeline, {ggml_vk_tensor_subbuffer(ctx, input), experts, vk_subbuffer{storage, 0, slots * a_bytes}}, pc, {a_int8 ? M * 256 : M * K / 2, slots, 1});
        handoff(subctx, true);
        if (profile) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 1);
        subctx->s->signal_semaphores.push_back({*job.ready, 0});
        submit(ctx, subctx);
        const int ready_fd = gpu->device.getSemaphoreFdKHR({*job.ready, vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd});
        if (ready_fd != -1) {
            const dma_buf_import_sync_file fence{DMA_BUF_SYNC_WRITE, ready_fd};
            const int result = ioctl(storage->npu_dma_fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &fence);
            const int error = errno;
            close(ready_fd);
            if (result < 0) throw std::system_error(error, std::generic_category(), "IQ1_S NPU: import Vulkan release fence");
        }
        for (auto & run : job.runs) run.start();
        dma_buf_export_sync_file fence{};
        fence.flags = DMA_BUF_SYNC_READ;
        fence.fd = -1;
        if (ioctl(storage->npu_dma_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &fence) < 0) throw std::system_error(errno, std::generic_category(), "IQ1_S NPU: export completion fence");
        try {
            gpu->device.importSemaphoreFdKHR({*job.completion, vk::SemaphoreImportFlagBits::eTemporary, vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd, fence.fd});
        } catch (...) {
            close(fence.fd);
            throw;
        }
        if (profile) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *job.queries, 2);
        ++calls;
    }

    void end(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * ids, const ggml_tensor * output, const vk_subbuffer & experts) {
        if (profile) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *pending.back().queries, 3);
        submit(ctx, subctx);
        subctx->s->wait_semaphores.push_back({*pending.back().completion, 0, vk::PipelineStageFlagBits::eAllCommands});
        handoff(subctx, false);
        if (profile) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *pending.back().queries, 4);
        const uint32_t n_experts = pending.back().n_experts;
        std::vector<uint32_t> pc = {n_experts, uint32_t(experts.size / 4 - slots * 4), slots, M, uint32_t(ids->ne[0])};
        pc.resize(32);
        pc[29] = a_prefix_bytes / 4;
        pc[30] = a_int8;
        pc[31] = interleaved_rows;
        auto & merge_pipeline = down ? (c_bf16 ? gpu->pipeline_npu_iq4_merge_bf16 : gpu->pipeline_npu_iq4_merge) : gpu->pipeline_npu_iq1_merge;
        ggml_pipeline_request_descriptor_sets(ctx, merge_pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, merge_pipeline,
            {vk_subbuffer{storage, gate_offset, slots * c_bytes}, vk_subbuffer{storage, down ? gate_offset : up_offset, slots * c_bytes}, experts, ggml_vk_tensor_subbuffer(ctx, output), vk_subbuffer{storage, 0, slots * a_bytes}}, pc, {M * N, slots, 1});
        if (profile) subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *pending.back().queries, 5);
    }
};

static void ggml_vk_npu_iq1_begin(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids, const ggml_tensor * up, const vk_subbuffer & experts) {
    if (!ctx->npu_iq1) ctx->npu_iq1 = std::make_shared<ggml_vk_npu_iq1>(ctx, getenv("GGML_VK_NPU_IQ1_S_DIR"));
    ctx->npu_iq1->begin(ctx, subctx, weights, input, ids, up, experts);
}

static void ggml_vk_npu_iq1_end(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * ids, const ggml_tensor * output, const vk_subbuffer & experts) {
    ctx->npu_iq1->end(ctx, subctx, ids, output, experts);
}

static void ggml_vk_npu_iq4_begin(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids, const vk_subbuffer & experts) {
    if (!ctx->npu_iq4) ctx->npu_iq4 = std::make_shared<ggml_vk_npu_iq1>(ctx, getenv("GGML_VK_NPU_IQ4_NL_DIR"), true);
    ctx->npu_iq4->begin(ctx, subctx, weights, input, ids, nullptr, experts);
}

static void ggml_vk_npu_iq4_end(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * ids, const ggml_tensor * output, const vk_subbuffer & experts) {
    ctx->npu_iq4->end(ctx, subctx, ids, output, experts);
}

static void ggml_vk_npu_finish(ggml_backend_vk_context * ctx) {
    if (ctx->npu) ctx->npu->finish();
    if (ctx->npu_qkv) ctx->npu_qkv->finish();
    if (ctx->npu_q6) ctx->npu_q6->finish();
    if (ctx->npu_iq4_xs) ctx->npu_iq4_xs->finish();
    if (ctx->npu_iq4_xs_ffn) ctx->npu_iq4_xs_ffn->finish();
    if (ctx->npu_iq3_s) ctx->npu_iq3_s->finish();
    if (ctx->npu_iq3_s_ffn) ctx->npu_iq3_s_ffn->finish();
    if (ctx->npu_iq4_xs_down) ctx->npu_iq4_xs_down->finish();
    if (ctx->npu_iq3_s_down) ctx->npu_iq3_s_down->finish();
    if (ctx->npu_iq3_xxs) ctx->npu_iq3_xxs->finish();
    if (ctx->npu_iq3_xxs_ffn) ctx->npu_iq3_xxs_ffn->finish();
    if (ctx->npu_iq3_xxs_down) ctx->npu_iq3_xxs_down->finish();
    if (ctx->npu_iq2_s) ctx->npu_iq2_s->finish();
    if (ctx->npu_iq2_s_ffn) ctx->npu_iq2_s_ffn->finish();
    if (ctx->npu_iq2_s_down) ctx->npu_iq2_s_down->finish();
    if (ctx->npu_iq4_xs_qkv) ctx->npu_iq4_xs_qkv->finish();
    if (ctx->npu_iq3_s_qkv) ctx->npu_iq3_s_qkv->finish();
    if (ctx->npu_iq3_xxs_qkv) ctx->npu_iq3_xxs_qkv->finish();
    if (ctx->npu_iq4_xs_out) ctx->npu_iq4_xs_out->finish();
    if (ctx->npu_iq3_s_out) ctx->npu_iq3_s_out->finish();
    if (ctx->npu_iq3_xxs_out) ctx->npu_iq3_xxs_out->finish();
    if (ctx->npu_iq1) ctx->npu_iq1->finish();
    if (ctx->npu_iq4) ctx->npu_iq4->finish();
}

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    ggml_vk_prepare_hc_compact_norm(ctx, cgraph);
    const char * directory = getenv("GGML_VK_NPU_Q5_K_DIR");
    const char * q6_directory = getenv("GGML_VK_NPU_Q6_K_DIR");
    const char * iq4_xs_directory = getenv("GGML_VK_NPU_IQ4_XS_DIR");
    const char * iq4_xs_ffn_directory = getenv("GGML_VK_NPU_IQ4_XS_FFN_DIR");
    const char * iq3_s_directory = getenv("GGML_VK_NPU_IQ3_S_DIR");
    const char * iq3_s_ffn_directory = getenv("GGML_VK_NPU_IQ3_S_FFN_DIR");
    const char * iq4_xs_down_directory = getenv("GGML_VK_NPU_IQ4_XS_DOWN_DIR");
    const char * iq3_s_down_directory = getenv("GGML_VK_NPU_IQ3_S_DOWN_DIR");
    const char * iq3_xxs_directory = getenv("GGML_VK_NPU_IQ3_XXS_DIR");
    const char * iq3_xxs_ffn_directory = getenv("GGML_VK_NPU_IQ3_XXS_FFN_DIR");
    const char * iq3_xxs_down_directory = getenv("GGML_VK_NPU_IQ3_XXS_DOWN_DIR");
    const char * iq2_s_directory = getenv("GGML_VK_NPU_IQ2_S_DIR");
    const char * iq2_s_ffn_directory = getenv("GGML_VK_NPU_IQ2_S_FFN_DIR");
    const char * iq2_s_down_directory = getenv("GGML_VK_NPU_IQ2_S_DOWN_DIR");
    const char * iq4_xs_out_directory = getenv("GGML_VK_NPU_IQ4_XS_OUT_DIR");
    const char * iq3_s_out_directory = getenv("GGML_VK_NPU_IQ3_S_OUT_DIR");
    const char * iq3_xxs_out_directory = getenv("GGML_VK_NPU_IQ3_XXS_OUT_DIR");
    if (!directory && !q6_directory && !iq4_xs_directory && !iq4_xs_ffn_directory && !iq3_s_directory && !iq3_s_ffn_directory && !iq4_xs_down_directory && !iq3_s_down_directory && !iq3_xxs_directory && !iq3_xxs_ffn_directory && !iq3_xxs_down_directory && !iq2_s_directory && !iq2_s_ffn_directory && !iq2_s_down_directory && !iq4_xs_out_directory && !iq3_s_out_directory && !iq3_xxs_out_directory) return ggml_backend_vk_graph_compute_gpu(backend, cgraph);
    const char * control = getenv("GGML_VK_NPU_CONTROL");
    const bool asynchronous = control && strcmp(control, "segmented-async") == 0;
    const bool sleeping = control && strcmp(control, "segmented-sleep") == 0;
    const bool segmented = control && (strcmp(control, "segmented") == 0 || asynchronous || sleeping);
    const bool profile = getenv("GGML_VK_NPU_PROFILE") != nullptr;
    const bool gate = getenv("GGML_VK_NPU_GATE") != nullptr;
    if (control && !segmented) {
        if (strcmp(control, "import") == 0) {
            try {
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    if (!directory || !ggml_vk_npu_matches(cgraph->nodes[i], gate ? 6144 : 10240)) continue;
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
    auto gpu_range = [&](int end, ggml_vk_npu * pack = nullptr, const ggml_tensor * input = nullptr) {
        ctx->npu_rms_pack_written = false;
        if (start == end) return GGML_STATUS_SUCCESS;
        struct reset_pack {
            ggml_backend_vk_context * ctx;
            ~reset_pack() { ctx->npu_rms_pack_input = nullptr; ctx->npu_rms_pack_output = nullptr; }
        } reset{ctx};
        if (pack && input && getenv("GGML_VK_NPU_RMS_PACK") && pack->packed_i8 && !pack->verify &&
            pack->K == 5120 && input->ne[0] == 5120 && input->ne[1] == pack->M && pack->M % 512 == 0) {
            ctx->npu_rms_pack_input = input;
            ctx->npu_rms_pack_output = pack->a;
        }
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
        if ((iq4_xs_ffn_directory || iq3_s_ffn_directory || iq3_xxs_ffn_directory || iq2_s_ffn_directory) && !segmented && node->op == GGML_OP_MUL_MAT &&
            ggml_vk_npu_matches(node, 17408, 5120, node->src[0]->type)) {
            int partner = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                auto * candidate = cgraph->nodes[j];
                if (candidate->op == GGML_OP_MUL_MAT && candidate->src[1] == node->src[1] &&
                    ggml_vk_npu_matches(candidate, 17408, 5120, candidate->src[0]->type)) {
                    partner = j;
                    break;
                }
            }
            if (partner < 0) continue;
            int target = -1;
            for (auto type : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S}) {
                if (!(type == GGML_TYPE_IQ4_XS ? iq4_xs_ffn_directory : type == GGML_TYPE_IQ3_S ? iq3_s_ffn_directory : type == GGML_TYPE_IQ3_XXS ? iq3_xxs_ffn_directory : iq2_s_ffn_directory)) continue;
                if (node->src[0]->type == type) target = i;
                else if (cgraph->nodes[partner]->src[0]->type == type) target = partner;
                if (target >= 0) break;
            }
            if (target < 0) continue;
            auto * projection = cgraph->nodes[target];
            const auto type = projection->src[0]->type;
            auto & npu = type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs_ffn : type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s_ffn : type == GGML_TYPE_IQ3_XXS ? ctx->npu_iq3_xxs_ffn : ctx->npu_iq2_s_ffn;
            if (npu && node->ne[1] != npu->M) continue;
            std::vector<const ggml_tensor *> views{projection};
            int join = target + 1;
            for (; join < cgraph->n_nodes; ++join) {
                auto * consumer = cgraph->nodes[join];
                bool dependent = false;
                for (auto * source : consumer->src) dependent |= std::find(views.begin(), views.end(), source) != views.end();
                if (!dependent) continue;
                if (!ggml_vk_is_empty(consumer)) break;
                views.push_back(consumer);
            }
            if (join <= partner) continue;
            try {
                if (!npu) npu = std::make_shared<ggml_vk_npu>(backend, type == GGML_TYPE_IQ4_XS ? iq4_xs_ffn_directory : type == GGML_TYPE_IQ3_S ? iq3_s_ffn_directory : type == GGML_TYPE_IQ3_XXS ? iq3_xxs_ffn_directory : iq2_s_ffn_directory, false,
                    getenv(type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_FFN_SCALE_TABLE" : type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_FFN_SCALE_TABLE" : type == GGML_TYPE_IQ3_XXS ? "GGML_VK_NPU_IQ3_XXS_FFN_SCALE_TABLE" : "GGML_VK_NPU_IQ2_S_FFN_SCALE_TABLE"), type, 17408);
                const auto status = gpu_range(i, npu.get(), projection->src[1]);
                if (status != GGML_STATUS_SUCCESS) return status;
                const bool input_packed = ctx->npu_rms_pack_written;
                ggml_cgraph parallel = *cgraph;
                parallel.nodes = cgraph->nodes + i;
                parallel.n_nodes = join - i;
                ggml_tensor * swiglu = !npu->verify && npu->c->type == GGML_TYPE_BF16 &&
                    ggml_vk_npu_can_fuse_swiglu(ctx, cgraph, target, join, cgraph->nodes[target == i ? partner : i]) ? cgraph->nodes[join] : nullptr;
                ggml_vk_npu * packed_down = nullptr;
                int down_idx = join + 1;
                while (swiglu && down_idx < cgraph->n_nodes && ggml_vk_is_empty(cgraph->nodes[down_idx])) ++down_idx;
                ggml_tensor * down_node = swiglu && down_idx < cgraph->n_nodes ? cgraph->nodes[down_idx] : nullptr;
                if (down_node && down_node->op == GGML_OP_MUL_MAT && down_node->src[1] == swiglu && getenv("GGML_VK_NPU_FFN_PACK")) {
                    const auto down_type = down_node->src[0]->type;
                    const char * down_directory = down_type == GGML_TYPE_IQ4_XS ? iq4_xs_down_directory : down_type == GGML_TYPE_IQ3_S ? iq3_s_down_directory : down_type == GGML_TYPE_IQ3_XXS ? iq3_xxs_down_directory : down_type == GGML_TYPE_IQ2_S ? iq2_s_down_directory : nullptr;
                    if (down_directory && ggml_vk_npu_matches(down_node, 5120, 17408, down_type)) {
                        auto & down = down_type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs_down : down_type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s_down : down_type == GGML_TYPE_IQ3_XXS ? ctx->npu_iq3_xxs_down : ctx->npu_iq2_s_down;
                        if (!down) down = std::make_shared<ggml_vk_npu>(backend, down_directory, false,
                            getenv(down_type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_DOWN_SCALE_TABLE" : down_type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_DOWN_SCALE_TABLE" : down_type == GGML_TYPE_IQ3_XXS ? "GGML_VK_NPU_IQ3_XXS_DOWN_SCALE_TABLE" : "GGML_VK_NPU_IQ2_S_DOWN_SCALE_TABLE"), down_type, 5120, 17408);
                        if (down->M == npu->M && down->packed_i8 && !down->verify) packed_down = down.get();
                    }
                }
                if (profile) GGML_LOG_INFO("ggml_vulkan: %s FFN overlap %s nodes=%d\n", npu->label(), projection->src[0]->name, parallel.n_nodes);
                if (profile && swiglu) GGML_LOG_INFO("ggml_vulkan: %s FFN fused SwiGLU %s\n", npu->label(), projection->src[0]->name);
                if (profile && swiglu && getenv("GGML_VK_NPU_FFN_PACK")) GGML_LOG_INFO("ggml_vulkan: FFN pack %s next=%s/%s direct=%d selected=%d\n", projection->src[0]->name,
                    down_node ? down_node->name : "end", down_node ? ggml_op_name(down_node->op) : "end", down_node && down_node->src[1] == swiglu, packed_down != nullptr);
                const auto begin = std::chrono::steady_clock::now();
                if (npu->execute(backend, cgraph, projection, &parallel, nullptr, {}, swiglu, packed_down ? packed_down->a : nullptr, input_packed)) {
                    ++cuts;
                    start = join + (swiglu ? 1 : 0);
                    if (packed_down && packed_down->execute(backend, cgraph, down_node, nullptr, nullptr, {}, nullptr, nullptr, true)) {
                        start = down_idx + 1;
                        ++cuts;
                    }
                    i = start - 1;
                } else {
                    start = i;
                }
                execute_ms += ggml_vk_npu::elapsed(begin, std::chrono::steady_clock::now());
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml_vulkan: dense IQ FFN failed: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        if ((iq4_xs_directory || iq3_s_directory || iq3_xxs_directory || iq2_s_directory) && !segmented && node->op == GGML_OP_MUL_MAT &&
            ggml_vk_npu_matches(node, 10240, 5120, node->src[0]->type)) {
            int target = -1;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                auto * candidate = cgraph->nodes[j];
                if (candidate->op != GGML_OP_MUL_MAT) continue;
                const auto type = candidate->src[0]->type;
                const char * path = type == GGML_TYPE_IQ4_XS ? iq4_xs_directory : type == GGML_TYPE_IQ3_S ? iq3_s_directory : type == GGML_TYPE_IQ3_XXS ? iq3_xxs_directory : type == GGML_TYPE_IQ2_S ? iq2_s_directory : nullptr;
                if (path && candidate->src[1] == node->src[1] && ggml_vk_npu_matches(candidate, 6144, 5120, type)) {
                    target = j;
                    break;
                }
            }
            if (target < 0) continue;
            auto * projection = cgraph->nodes[target];
            const auto type = projection->src[0]->type;
            auto & npu = type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs : type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s : type == GGML_TYPE_IQ3_XXS ? ctx->npu_iq3_xxs : ctx->npu_iq2_s;
            if (npu && node->ne[1] != npu->M) continue;
            std::vector<const ggml_tensor *> views{projection};
            int join = target + 1;
            for (; join < cgraph->n_nodes; ++join) {
                auto * consumer = cgraph->nodes[join];
                bool dependent = false;
                for (auto * source : consumer->src) dependent |= std::find(views.begin(), views.end(), source) != views.end();
                if (!dependent) continue;
                if (!ggml_vk_is_empty(consumer)) break;
                views.push_back(consumer);
            }
            try {
                if (!npu) npu = std::make_shared<ggml_vk_npu>(backend, type == GGML_TYPE_IQ4_XS ? iq4_xs_directory : type == GGML_TYPE_IQ3_S ? iq3_s_directory : type == GGML_TYPE_IQ3_XXS ? iq3_xxs_directory : iq2_s_directory, true,
                    getenv(type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_SCALE_TABLE" : type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_SCALE_TABLE" : type == GGML_TYPE_IQ3_XXS ? "GGML_VK_NPU_IQ3_XXS_SCALE_TABLE" : "GGML_VK_NPU_IQ2_S_SCALE_TABLE"), type);
                const auto status = gpu_range(i, npu.get(), projection->src[1]);
                if (status != GGML_STATUS_SUCCESS) return status;
                const bool input_packed = ctx->npu_rms_pack_written;
                std::vector<ggml_tensor *> parallel_nodes(cgraph->nodes + i, cgraph->nodes + target);
                parallel_nodes.insert(parallel_nodes.end(), cgraph->nodes + target + 1, cgraph->nodes + join);
                ggml_cgraph parallel = *cgraph;
                parallel.nodes = parallel_nodes.data();
                parallel.n_nodes = parallel_nodes.size();
                if (profile) GGML_LOG_INFO("ggml_vulkan: %s gate overlap %s nodes=%d first=%s last=%s\n", npu->label(), projection->src[0]->name, parallel.n_nodes, parallel.nodes[0]->name, parallel.nodes[parallel.n_nodes - 1]->name);
                const auto qkv_type = node->src[0]->type;
                const char * qkv_key = qkv_type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_QKV_DIR" : qkv_type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_QKV_DIR" : qkv_type == GGML_TYPE_IQ3_XXS ? "GGML_VK_NPU_IQ3_XXS_QKV_DIR" : nullptr;
                const char * qkv_directory = qkv_key ? getenv(qkv_key) : nullptr;
                ggml_vk_npu * split = nullptr;
                if (qkv_directory) {
                    auto & qkv = qkv_type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs_qkv : qkv_type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s_qkv : ctx->npu_iq3_xxs_qkv;
                    if (!qkv) qkv = std::make_shared<ggml_vk_npu>(backend, qkv_directory, false,
                        getenv(qkv_type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_QKV_SCALE_TABLE" : qkv_type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_QKV_SCALE_TABLE" : "GGML_VK_NPU_IQ3_XXS_QKV_SCALE_TABLE"), qkv_type, 10240);
                    if (qkv->M == npu->M) split = qkv.get();
                    if (profile && split) GGML_LOG_INFO("ggml_vulkan: %s QKV overlap %s before %s\n", qkv->label(), node->src[0]->name, projection->src[0]->name);
                }
                const auto begin = std::chrono::steady_clock::now();
                if (npu->execute(backend, cgraph, projection, &parallel, split, {}, nullptr, nullptr, input_packed)) {
                    ++cuts;
                    start = join;
                    i = join - 1;
                } else {
                    start = i;
                }
                execute_ms += ggml_vk_npu::elapsed(begin, std::chrono::steady_clock::now());
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml_vulkan: dense IQ gate failed: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        const auto output_type = node->op == GGML_OP_MUL_MAT ? node->src[0]->type : GGML_TYPE_COUNT;
        const bool ssm_out = node->op == GGML_OP_MUL_MAT && node->src[0]->ne[0] == 6144;
        const int output_inner = ssm_out ? 6144 : 17408;
        const char * output_directory = ssm_out ? (output_type == GGML_TYPE_IQ4_XS ? iq4_xs_out_directory : output_type == GGML_TYPE_IQ3_S ? iq3_s_out_directory : output_type == GGML_TYPE_IQ3_XXS ? iq3_xxs_out_directory : nullptr) :
            (output_type == GGML_TYPE_IQ4_XS ? iq4_xs_down_directory : output_type == GGML_TYPE_IQ3_S ? iq3_s_down_directory : output_type == GGML_TYPE_IQ3_XXS ? iq3_xxs_down_directory : output_type == GGML_TYPE_IQ2_S ? iq2_s_down_directory : nullptr);
        if (output_directory && !segmented && ggml_vk_npu_matches(node, 5120, output_inner, output_type)) {
            auto & npu = ssm_out ? (output_type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs_out : output_type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s_out : ctx->npu_iq3_xxs_out) :
                (output_type == GGML_TYPE_IQ4_XS ? ctx->npu_iq4_xs_down : output_type == GGML_TYPE_IQ3_S ? ctx->npu_iq3_s_down : output_type == GGML_TYPE_IQ3_XXS ? ctx->npu_iq3_xxs_down : ctx->npu_iq2_s_down);
            if (npu && node->ne[1] != npu->M) continue;
            const auto status = gpu_range(i);
            if (status != GGML_STATUS_SUCCESS) return status;
            try {
                if (!npu) {
                    const char * scale_key = ssm_out ? (output_type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_OUT_SCALE_TABLE" : output_type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_OUT_SCALE_TABLE" : "GGML_VK_NPU_IQ3_XXS_OUT_SCALE_TABLE") :
                        (output_type == GGML_TYPE_IQ4_XS ? "GGML_VK_NPU_IQ4_XS_DOWN_SCALE_TABLE" : output_type == GGML_TYPE_IQ3_S ? "GGML_VK_NPU_IQ3_S_DOWN_SCALE_TABLE" : output_type == GGML_TYPE_IQ3_XXS ? "GGML_VK_NPU_IQ3_XXS_DOWN_SCALE_TABLE" : "GGML_VK_NPU_IQ2_S_DOWN_SCALE_TABLE");
                    npu = std::make_shared<ggml_vk_npu>(backend, output_directory, false, getenv(scale_key), output_type, 5120, output_inner);
                }
                const auto begin = std::chrono::steady_clock::now();
                start = npu->execute(backend, cgraph, node) ? i + 1 : i;
                if (start == i + 1) ++cuts;
                execute_ms += ggml_vk_npu::elapsed(begin, std::chrono::steady_clock::now());
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml_vulkan: dense IQ output split failed: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        if (q6_directory && !segmented && ggml_vk_npu_matches(node, 2560, 6144, GGML_TYPE_Q6_K) && (!ctx->npu_q6 || node->ne[1] == ctx->npu_q6->M)) {
            const auto status = gpu_range(i);
            if (status != GGML_STATUS_SUCCESS) return status;
            try {
                if (!ctx->npu_q6) ctx->npu_q6 = std::make_shared<ggml_vk_npu>(backend, q6_directory, false, getenv("GGML_VK_NPU_Q6_SCALE_TABLE"), GGML_TYPE_Q6_K);
                start = ctx->npu_q6->execute(backend, cgraph, node) ? i + 1 : i;
                if (start == i + 1) ++cuts;
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml_vulkan: Q6_K split failed: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        if (!directory) continue;
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
        if (ctx->npu && node->ne[1] != ctx->npu->M) continue;
        int target = i;
        int join = i + 1;
        if (gate && !segmented) {
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                auto * candidate = cgraph->nodes[j];
                if (candidate->src[1] == node->src[1] && ggml_vk_npu_matches(candidate, 6144)) {
                    target = j;
                    break;
                }
            }
            if (target == i) continue;
            node = cgraph->nodes[target];
            std::vector<const ggml_tensor *> views{node};
            for (join = target + 1; join < cgraph->n_nodes; ++join) {
                auto * consumer = cgraph->nodes[join];
                bool dependent = false;
                for (auto * source : consumer->src) {
                    dependent |= std::find(views.begin(), views.end(), source) != views.end();
                }
                if (!dependent) continue;
                if (!ggml_vk_is_empty(consumer)) break;
                views.push_back(consumer);
            }
            if (join >= target + 3 && ggml_vk_can_fuse_rms_norm_gate(ctx, cgraph, join - 2)) join -= 2;
        }
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
                if (gate && !ctx->npu_qkv) {
                    if (const char * qkv_directory = getenv("GGML_VK_NPU_QKV_DIR")) {
                        ctx->npu_qkv = std::make_shared<ggml_vk_npu>(backend, qkv_directory, false, getenv("GGML_VK_NPU_QKV_SCALE_TABLE"));
                    }
                }
                ggml_cgraph parallel = *cgraph;
                std::vector<ggml_tensor *> parallel_nodes;
                if (gate) {
                    parallel_nodes.insert(parallel_nodes.end(), cgraph->nodes + i, cgraph->nodes + target);
                    parallel_nodes.insert(parallel_nodes.end(), cgraph->nodes + target + 1, cgraph->nodes + join);
                    parallel.nodes = parallel_nodes.data();
                    parallel.n_nodes = parallel_nodes.size();
                }
                if (gate && profile) {
                    GGML_LOG_INFO("ggml_vulkan: Q5_K gate overlap %s nodes=%d first=%s last=%s\n", node->src[0]->name, parallel.n_nodes,
                                  parallel.nodes[0]->name, parallel.nodes[parallel.n_nodes - 1]->name);
                }
                if (ctx->npu->execute(backend, cgraph, node, gate ? &parallel : nullptr, ctx->npu_qkv.get())) {
                    start = join;
                    i = join - 1;
                } else {
                    start = i;
                }
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
