# Apple Neural Engine (experimental)

Build on Apple Silicon macOS with `-DGGML_ANE=ON`. Metal is enabled by default. The backend uses private AppleNeuralEngine and IOSurface APIs; it has been tested on an M4 with macOS 26.6.2. Availability on other macOS versions is not established.

```sh
cmake -S . -B build-ane -DGGML_ANE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-ane --target llama-bench test-backend-ops -j8
build-ane/bin/llama-bench --list-devices
build-ane/bin/llama-bench -m model-f16.gguf -dev ANE/MTL0 -sm layer -ts 1/0 -ngl 99 -p 128 -n 32
```

`llama-bench` uses `/` for devices within one configuration; `ANE,MTL0` runs two separate configurations. For tools using the common argument parser, use `--device ANE,MTL0 --split-mode layer`. Keep ANE first to give it scheduling priority for supported operations. Split mode `none` removes the other device, so it does not provide Metal fallback. Layer placement still controls which weights are available to ANE; this is not an automatic performance policy for every model. With `-ts 1/0`, offloaded layers prefer ANE. If the layer device does not support a weight's operation, the loader tries selected peer devices that support its buffer types before falling back to CPU. These fallback weights use the peer's native buffer type; ANE matmul weights and shared activations retain the ANE buffer type where needed. The input embedding remains on CPU by the normal loader policy. Offloading a tied output projection can require a separate copy of the embedding weights, as with Metal-only offload.

Supported operations are contiguous, two-dimensional matmuls with F16 weights, F16 or F32 activations, and F32 output. For `weights[K,M] * activations[K,N]`, K and M must be multiples of 32 between 32 and 16384; N must be a multiple of 32 between 32 and 4096. Each supported matmul runs entirely on ANE. Unsupported shapes, batching, types, and operations remain on another backend. Single-token decoding is therefore not offloaded. Explicit F32/BF16 accumulation or activation precision requirements also prevent offload.

Metal scales each activation token by a power of two chosen from its maximum absolute value, then converts it to F16. This keeps small activations out of the range where ANE loses F16 subnormals. The output shader removes the scale. ANE computes `W * A^T` in F16, then a tiled Metal shader transposes, rescales, and converts the result to ggml's F32 output layout. The ANE program splits K into reductions of at most 2048 elements and adds the partial results in F16 within the same request. This reduces the measured cost of large down-projections. This can round or overflow values that a full F32 path would retain. Model compilation is lazy and cached by shape for the backend lifetime; compiler failures are reported as execution failures, rather than retried on Metal. First-use timings include compilation. Compiled artifacts use the system temporary directory and the ANE model identifier.

The ANE buffer type owns an IOSurface allocation and a Metal buffer over its storage. Allocations can use the Metal device's full buffer-size limit. Tensors are page aligned and keep their normal ggml layout. Metal advertises support for this buffer type and accesses the same allocation with a byte offset. ANE receives a size-bounded IOSurface created with `IOSurfaceAddress` over the tensor's userspace pointer, with the view size rounded up to a page. The backing allocation outlives all cached wrappers, and GPU work must finish before buffers are reset or freed. F16 weights and F16 activation views with unaligned offsets fall back to another backend; F32 activation views can use the Metal conversion path.

Child IOSurfaces are not used: experiments found that ANE wraps child offsets above 4 GiB modulo 4 GiB and rejects an exactly 4 GiB parent. Bounded pointer wrappers passed the corresponding read, write, and guard checks.

Metal shared-event fences order producers and consumers without a CPU wait at each backend boundary. The ANE completion fence is recorded after output conversion. Pending requests are bounded to limit temporary storage. There is no separate `ggml-iosurface` backend, no split matmul, and no PQ2 or planar quantized-weight repacking in this initial F16 implementation.

The existing backend test executable includes CPU-reference F16 cases and an ANE/Metal chain test:

```sh
build-ane/bin/test-backend-ops -b ANE -o MUL_MAT \
  -p 'type_a=f16,type_b=f(16|32),m=(64|256),n=(32|128),k=(128|512),bs=\[1,1\],nr=\[1,1\]'
```

The chain alternates Metal scale operations and ANE matmuls through shared activation buffers, exercises nonzero tensor offsets and repeated execution, checks precision rejection and single-token fallback, and duplicates and waits on native fences from both backends. A tiny synthetic F16 Llama model has also completed prefill and decoding with both devices enabled. These checks establish functional operation on the tested machine. The chain also covers activation tokens with tiny, large, and zero values to check per-token scaling.

## Measured prefill performance

On Apple M4 with `LFM2.5-2.6B-F16.gguf`, five warmed pp4096 runs at batch 4096 and microbatch 2048 measured 788.52 +/- 64.11 tokens/s for ANE/Metal and 653.84 +/- 0.39 for Metal alone, about 21% higher mean throughput. The default microbatch 512 did not show a mixed-backend advantage in the earlier sweep. Mixed timings vary more than Metal timings.

```
./build/bin/llama-bench -m ~/LFM2.5-2.6B-F16.gguf -dev ANE/MTL0,MTL0 -sm layer -ts 1/0 -ngl 99 -p 4096 -b 4096 -ub 2048 -n 0 -r 5
```

A 4096-token passage from this repository's README gave mean KL divergence 0.000053 and 100% top-token agreement against Metal. This is a limited numerical check, not a general model-quality evaluation. Without activation scaling, mean KL divergence was about 0.406, despite the basic matmul tests passing.
