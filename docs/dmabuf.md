# dma-buf shared memory buffers

`ggml-dmabuf` provides a `ggml` buffer type backed by a Linux dma-buf
allocation. Tensors placed in this buffer type can be shared between backends
(for example a GPU and an NPU) without copies, as long as the consuming
backends support importing the dma-buf.

DMA-BUF is an allocation buffer type exposed by compute backends through their extra buffer types.

## Build

Linux only, off by default:

```sh
cmake -B build -DGGML_DMABUF=ON
```

This builds DMA-BUF allocation support into `ggml-base`. Backends that import DMA-BUF expose the `DMA_BUF` buffer type through `ggml_backend_dev_get_extra_bufts`.

## Usage

The buffer type is not selected automatically. It is exposed as an extra buffer
type and can be requested with tensor buffer overrides:

```sh
llama-cli -m model.gguf -dev Vulkan0 -ot '.*=DMA_BUF'
```

The override patterns are regexes over tensor names. Scope them to the tensors
you actually want shared, for example `blk\..*=DMA_BUF` to exclude the token
embedding and other host-only tensors.

## How it works

- `ggml-dmabuf` allocates a dma-buf (see "Allocation source"), mmaps it, and
  exposes it as a CPU-accessible host buffer type (`is_host = true`).
- Backends that want zero-copy access register an importer through `ggml-dmabuf.h`.
  The importer is notified on buffer allocation and free, imports the dma-buf
  into its own API, and releases it on free.
- The Vulkan backend imports via `VK_EXT_external_memory_dma_buf` +
  `VK_KHR_external_memory_fd`, and resolves tensors through the same pointer
  lookup used for host-mapped tensors, on both UMA and non-UMA devices.
- The scheduler only inserts a copy between two backends when the target
  backend does not support the source buffer type, so a shared dma-buf tensor
  is used in place.

## Caveats

### Import failure is a hard error

When a Vulkan device advertises `VK_KHR_external_memory_fd` and
`VK_EXT_external_memory_dma_buf`, a failure to import an allocated dma-buf
aborts the process instead of falling back. This is deliberate: the scheduler
decides placement from `supports_buft`, which only knows that the extensions
exist, not whether a specific file descriptor imports successfully. A silent
failure would leave an op assigned to a backend that cannot resolve the tensor.
If a device with the extensions is present but unused, a failed import still
aborts. Treat a dma-buf import failure as a configuration/driver problem.

### Shared tensors are assumed read-only

The Vulkan graph sync and fusion overlap checks treat buffers that are not
Vulkan-owned as non-overlapping. This is correct for tensors that are only read
during graph execution (weights), which is the intended use. It is not correct
for dma-buf tensors that are written as part of the graph (activations). Do not
use the dma-buf buffer type for graph-computed tensors until the overlap/sync
logic understands cross-backend buffers.

### `-ot` bypasses the load-time support check

Tensor buffer overrides force the requested buffer type without checking that a
backend can compute the op. A tensor that no backend wants in the dma-buf (for
example a quantized `GET_ROWS` embedding) is still moved there. This can waste
memory and, on a device-local dma-buf, exhaust VRAM or force slow CPU access to
device memory. Prefer scoped patterns.

### `is_host = true`

The buffer type reports as host memory because the current allocation is
CPU-mappable, so the CPU backend can run ops on it and act as a fallback.
Non-UMA device-local dma-bufs (for example an nvkms GEM buffer) are still
reported as host even though CPU access is slow. If a future allocation is not
CPU-mappable, `is_host` must be set to false. Note that this alone is not
enough: with `is_host = false` the CPU backend rejects the buffer type, and the
`-ot` path still forces tensors into it. Selection must be routed through the
buffer type lists (or the override must validate support) for a graceful
fallback.

### Non-UMA performance

A non-UMA GPU can import a system-heap dma-buf and read it over PCIe. Prompt
processing (compute bound) is only partially affected, token generation
(memory bound) is limited by the link bandwidth. For a discrete GPU, prefer a
device-local allocation. The current default (`/dev/dma_heap/system`) is
suitable for UMA.

### Permissions

`/dev/dma_heap/system` is usually root-only. Grant access with a udev rule or
run with the required permissions.

## Allocation source

The allocation source is selected with environment variables. It is the only
platform/vendor-specific part of this buffer type.

- `GGML_DMABUF_HEAP` (default `system`): name of a dma-heap. The allocator opens
  `/dev/dma_heap/$GGML_DMABUF_HEAP`. The special value `nvidia` selects the
  NVIDIA allocator instead.
- `GGML_DMABUF_NVIDIA_DEVICE`: DRM render node used by the NVIDIA allocator,
  for example `/dev/dri/renderD129`. Required when `GGML_DMABUF_HEAP=nvidia`.
- `GGML_DMABUF_HEAP=amdgpu`: allocate an AMDGPU GTT buffer and export it as a DMA-BUF. Requires `drm/amdgpu_drm.h` at build time.
- `GGML_DMABUF_AMDGPU_DEVICE`: AMDGPU DRM render node, for example `/dev/dri/renderD128`. Required when `GGML_DMABUF_HEAP=amdgpu`.

Examples:

```sh
# system heap (default)
llama-cli -m model.gguf -dev Vulkan0 -ot 'blk\..*=DMA_BUF'

# device-local memory on an NVIDIA GPU
GGML_DMABUF_HEAP=nvidia GGML_DMABUF_NVIDIA_DEVICE=/dev/dri/renderD129 \
    llama-cli -m model.gguf -dev Vulkan0 -ot 'blk\..*=DMA_BUF'
```

The NVIDIA allocator uses `DRM_IOCTL_NVIDIA_GEM_ALLOC_NVKMS_MEMORY` followed by
`DRM_IOCTL_PRIME_HANDLE_TO_FD`. This is an undocumented, unstable ioctl and must
not be treated as a stable API. It can change or disappear without notice. It is
useful to allocate device-local memory on an NVIDIA GPU, but it is opt-in only
and is never enabled by default. Vendor allocators are expected to be kept out
of the mainline buffer implementation.
