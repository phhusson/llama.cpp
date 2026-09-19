# Apple Metal / ANE PQ2_0 experiment

This experiment runs Metal and the Apple Neural Engine (ANE) together during prompt processing in bonsai-llama.cpp. It targets `~/ML/Ternary-Bonsai-2-27B-PQ2_0.gguf`. Single-token generation runs entirely on Metal.

## Before/after benchmark

Test machine: Apple M4 Mac mini, 16 GB, macOS 26.6.2 (25G83).

Same model as above, pp4096/tg32, batch 4096, microbatch 2048, flash attention, and all layers offloaded. Each result is the mean of three timed repetitions after a full warmup; model loading and weight conversion are excluded from the timed results.

| Configuration | pp4096 tok/s | tg32 tok/s | Timed runs |
| --- | ---: | ---: | ---: |
| [Before: original Metal, clean `1a07bfa5f`](ane-kernel/results/metal-original.json) | 58.73 | 10.98 | 3 |
| [MPS ON / ANE OFF, planar weights](ane-kernel/results/mps-recheck-on-f000.json) | 59.93 | 10.54 | 3 |
| [MPS ON / ANE ON, planar weights, fraction 0.65](ane-kernel/results/fraction-065-repeat.json) | 115.68 | 10.58 | 3 |

The MPS-only configuration measured 2.0% higher prefill throughput than original Metal. Adding the 0.65 ANE split measured 93.0% higher throughput than MPS alone. The combined result includes changes to the Metal path and weight layout as well as ANE execution. Generation at 0.65 measured 3.6% below original Metal. Configurations were measured separately: the first two rows are earlier measurements, and the 0.65 row was rerun with the CMake-embedded programs. See the [benchmark history and validation](ane-kernel/HISTORY.md) for samples, provenance, and earlier comparisons.

For the MPS ON / ANE OFF row, `GGML_METAL_ANE_FRACTION=0` keeps planar storage and runs full-row MPS matmuls with zero ANE requests.

The ANE executes fused packed PQ2_0 decoding, scaling, and matmul. CMake uses the `coremltools` generator in `ane-kernel/` and embeds the programs in the Metal backend. Inference submits them through the private AppleNeuralEngine API. The synchronous public Core ML prediction path has been removed from ggml-metal.

## Build and run

```sh
cmake -S . -B build
cmake --build build --target llama-bench -j 8
build/bin/llama-bench \
    -m ~/ML/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -p 4096 -n 32 -b 4096 -ub 2048 -ngl 99 -fa on -r 3
```

Native Apple Silicon macOS builds enable `GGML_METAL_ANE` by default. Python 3.12 must be installed on the build machine. CMake creates a private Python environment in the build directory, installs `ane-kernel/requirements.txt`, generates and checks the six programs, and embeds their MIL text and constant blobs in `libggml-metal`. Python and coremltools are only build-time dependencies. The first build needs access to the Python packages; subsequent unchanged builds reuse the generated data. A missing dependency or failed generation stops the build.

No environment variable, model-directory path, wrapper, or separate generation command is needed at runtime. The executable and its libraries can be used from any working directory. `llama-cli` and other frontends use the same embedded programs. Use `-b 4096 -ub 2048 -ngl 99 -fa on`. The supplied programs cover 2048-token microbatches and six matrix shapes. Other prompt shapes use Metal's planar kernels. Token generation uses the planar Metal matvec kernel without ANE requests.

To build without ANE support or its Python dependencies:

```sh
cmake -S . -B build -DGGML_METAL_ANE=OFF
```

Generated packages, compute plans, and the embedded header live under `build/ggml/src/ggml-metal/ane-kernel/`. They contain no model weight values. Generation checks that every operation reported by the Core ML compute plan prefers ANE. At runtime, `_ANEInMemoryModel` compiles and loads the embedded MIL on first use; its temporary files are managed internally.

For a different nonzero split, configure `-DGGML_METAL_ANE_FRACTION=0.55` and rebuild. CMake regenerates the matching shapes and embeds that fraction as the runtime default. `ane-kernel/generate.py` and `build.sh` remain available for kernel development, but their source-tree output directory is not used by inference.

## One load-time weight conversion

On disk, PQ2_0 remains unchanged: each 128-weight block contains an FP16 scale and 32 packed bytes. When ANE is enabled, the loader converts the 368 eligible matmul tensors into shared IOSurfaces once, during model loading:

- `Q`: 32 packed bytes per block, with each byte holding the codes at positions `i`, `i+32`, `i+64`, and `i+96`.
- `D`: the original FP16 block scales, in a separate plane. Each row's scale count is padded to a multiple of 32.

For a two-bit code `q[j]`, the stored byte is:

```text
Q[row, block, i] = q[i] | (q[i+32] << 2) | (q[i+64] << 4) | (q[i+96] << 6)
weight[j] = D[row, block] * (q[j] - 1)
```

Metal imports the same IOSurface allocations with `newBufferWithBytesNoCopy`. ANE binds the leading rows of those surfaces directly. There is no second persistent PQ2 weight copy for ANE, and no per-matmul weight repack or scale copy. Metal reads these planes during both prefill and generation. The JIT planar repack path has been removed.

The model loader uses a dedicated weight buffer type and a small optional backend hook to select it. Embeddings and other tensors keep their existing formats. Generated F32 Hadamard rotations use normal device storage. Metal's direct mapping of the GGUF is disabled in planar mode so unused original PQ2 pages are not pinned alongside the converted weights. The on-disk model is never modified.

The default Metal prefill partition expands its planar rows into reusable FP16 scratch and invokes MPS matmul. That temporary expansion still happens for each matmul. The ordinary ggml Metal matmul also supports inline planar decoding; see the experimental switches below.

## Row split and shared-event synchronization

The default ANE row count is `floor(0.65 * output_rows / 256) * 256`:

| Projection | Reduction width | ANE rows | Metal rows |
| --- | ---: | ---: | ---: |
| FFN gate/up | 5120 | 11264 | 6144 |
| FFN down | 17408 | 3328 | 1792 |
| Linear attention QKV | 5120 | 6656 | 3584 |
| Linear attention gate | 5120 | 3840 | 2304 |
| Full attention Q + gate | 5120 | 7936 | 4352 |
| Attention output | 6144 | 3328 | 1792 |

ANE takes the leading rows; Metal takes the trailing rows. This permits ANE bindings at surface offset zero. The attempted nonzero-offset binding failed numerical validation on this OS.

```text
Metal: preceding ops + activation FP16 cast -> signal input event
ANE:   wait input event -> PQ2 decode + matmul -> signal output event
Metal: own row partition -> wait output event -> join -> dependent ops
```

`MTLSharedEvent` and private `_ANESharedWaitEvent` / `_ANESharedSignalEvent` provide cross-device fences. The success path signals both events from device work. CPU completion callbacks observe errors and lifetimes. There is no per-matmul `waitUntilCompleted`; the host can queue eight requests, waits for queue capacity when necessary, and synchronizes at the graph boundary.

The ANE program multiplies scales by 32 and Metal divides the output by 32 during the join. This avoids the small-result numerical issue found during the earlier Core ML experiment. Activation conversion and output joining remain GPU work per matmul.

## Experimental switches

- Default: run the frontend directly to use persistent planar weights and MPS for the Metal prefill partition.
- `GGML_METAL_ANE=0` disables ANE and its automatic planar weight conversion. `GGML_METAL_ANE` is now a boolean, not a path; the old directory setting produces an explicit error.
- `GGML_METAL_ANE_MPS_DISABLE=1` uses ggml's fused planar decode/matmul for the Metal partition instead of temporary expansion plus MPS.
- `GGML_METAL_ANE_FRACTION=0` uses MPS for all rows of compatible prefill matmuls, with no ANE requests. Adding `GGML_METAL_ANE_MPS_DISABLE=1` selects the ordinary planar Metal kernels instead. Single-token generation uses the same Metal matvec in each case.
- `GGML_METAL_ANE_PROFILE=1` logs conversions, ANE completions, and graph times. Add `-v` to `llama-bench` to retain these logs.
- `GGML_METAL_ANE_VALIDATE=1` checks 64 scalar outputs per partition per matmul and adds host waits. Disable it for performance measurements.

The optional `ane-kernel/run.sh` wrapper only sets `GGML_METAL_ANE=1`. `GGML_METAL_ANE=0 GGML_METAL_ANE_PLANAR=1` allows testing planar storage with Metal alone. Runtime fraction overrides require matching embedded shapes; unsupported shapes use Metal. The private API has only been checked on the machine and OS above.

## Validation

The standalone checks cover lossless planar weight readback, Metal matvec/matmul, and both sides of the ANE/Metal split. The [benchmark history and validation](ane-kernel/HISTORY.md#startup-and-correctness) contains the commands and real-model numerical comparisons.
