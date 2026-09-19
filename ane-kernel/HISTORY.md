# Apple Metal / ANE PQ2_0 benchmark history

See [README-Apple.md](../README-Apple.md) for the current comparison, build instructions, and implementation overview. Measurements below used the Apple M4 Mac mini and Ternary-Bonsai-2-27B-PQ2_0 model described there. Run the validation commands from the repository root.

## Benchmarks and validation

Historical measurements used pp4096, batch 4096, microbatch 2048, flash attention, all layers offloaded, and three timed runs after warmup:

| Configuration | pp4096 tok/s | tg32 tok/s |
| --- | ---: | ---: |
| Original Metal, clean commit `1a07bfa5f` | 58.73 | 10.98 |
| Previous event split, per-matmul repack | 106.32 | 10.23 |

The previous event split improved prefill by 81% over original Metal. Those results include per-matmul weight repacking; they are not measurements of load-time planar storage. The original source was rebuilt in an isolated clean worktree after the experiment. Exact samples are in `ane-kernel/results/metal-original.json` and `events-jit-before.json`.

Before removing the JIT repack path, both layouts were measured with ANE taking the leading rows. With the same warmup and three-run settings:

| Measured configuration | pp4096 tok/s | tg32 tok/s |
| --- | ---: | ---: |
| JIT weight repack + Metal MPS + ANE (removed) | 105.25 | 10.25 |
| Load-time planar weights + Metal MPS + ANE | 103.37 | 10.59 |
| Load-time planar weights + fused Metal matmul + ANE | 84.10 | 10.53 |

Planar/MPS showed no prefill improvement in these runs: 1.8% lower pp throughput, with 3.3% higher generation throughput. The measurements were sequential, not interleaved; the small difference should not be treated as a definitive hardware limit. Exact samples are in `ane-kernel/results/jit-mps-pp4096.json` and `planar-mps-pp4096.json`.

The fused planar Metal matmul reached 84.10 pp tok/s, 20.1% below JIT/MPS. It passed the split-boundary numerical checks, but MPS remains faster for the Metal partition on this M4. All three benchmarks completed 2,944 successful ANE requests across warmup and three timed passes, with zero execution errors. See `ane-kernel/results/benchmark-summary.json` and `planar-fused-pp4096.json`.

### MPS and ANE fraction recheck

After removing JIT repacking, all three configurations below were measured with the same planar weights, pp4096/tg32, batch 4096, microbatch 2048, flash attention, and three timed repetitions after warmup. The runs were sequential in table order.

| MPS | ANE fraction | pp4096 tok/s | tg32 tok/s |
| --- | ---: | ---: | ---: |
| On | 0.55 | 104.05 | 10.58 |
| Off | 0.55 | 84.18 | 10.59 |
| On | 0 | 59.93 | 10.54 |

MPS increased prefill throughput by 23.6% at fraction 0.55. With MPS enabled, adding the 0.55 ANE split increased prefill throughput by 73.6% over fraction zero. Generation used the same Metal matvec in every configuration; the measured spread was below 0.5%.

Fraction zero previously bypassed MPS. It now explicitly runs full-row MPS matmuls for compatible prefill operations. The scalar check passed with zero error. Each split benchmark completed 2,944 successful ANE requests; fraction zero completed 2,944 Metal-only MPS matmuls and zero ANE requests. Raw samples are in `ane-kernel/results/mps-recheck-{on-f055,off-f055,on-f000}.json`; the summary and command/source provenance are in `mps-recheck-summary.json` and `mps-recheck-provenance.json` in the same directory.

### ANE fraction tuning

With MPS enabled, fraction 0.65 reached 115.25 pp4096 tok/s versus 103.98 at 0.55, a 10.8% gain. Each sweep result used one timed pass after a full warmup with the same batch and microbatch settings. Fraction 0.65 was the fastest completed setting and was accepted as the new default; the finer sweep was stopped. These are measured candidates, not a claim of an exhaustive optimum.

| ANE fraction | pp4096 tok/s |
| ---: | ---: |
| 0.35 | 81.96 |
| 0.45 | 92.41 |
| 0.55 | 103.98 |
| 0.60 | 107.98 |
| 0.625 | 109.32 |
| 0.64 | 110.36 |
| 0.65 | 115.25 |
| 0.664 | 113.61 |
| 0.675 | 112.64 |
| 0.70 | 111.88 |
| 0.75 | 102.18 |
| 0.85 | 96.02 |
| 1.0 | 85.41 |

All completed sweep runs executed the expected 1,472 ANE requests across warmup and timing, with no missing-kernel fallback. CMake now generates and embeds the six selected shapes for 0.65 automatically. Raw results and command/source provenance are under `ane-kernel/results/fraction-{coarse,refine,plateaus}-*`.

### Fraction 0.65 repeat measurement

The CMake-embedded build was then measured with MPS enabled, fraction 0.65, and the same pp4096/tg32 settings, using a full warmup and three timed repetitions. Prefill reached 115.68 +/- 0.42 tok/s and generation reached 10.581 +/- 0.012 tok/s (mean +/- sample standard deviation). The prefill samples were 115.30, 115.60, and 116.13 tok/s; generation samples were 10.5859, 10.5681, and 10.5896 tok/s.

All six embedded programs loaded, and all 2,944 ANE requests across warmup and timing succeeded, with zero missing-kernel fallbacks. Generation used Metal. Source files and binaries were unchanged throughout the run. See the [raw results](results/fraction-065-repeat.json), [request summary](results/fraction-065-repeat-summary.json), and [command and source provenance](results/fraction-065-repeat-provenance.json).

### Startup and correctness

The embedded build was checked from `/tmp` with `GGML_METAL_ANE` and the fraction override unset. The split numerical check passed with maximum sampled scalar error 0.000122070312. A real-model pp2048 smoke run loaded all six programs and completed 368 successful ANE requests with no missing-kernel fallback. An unchanged configure/build reused the generated data, and an ANE-disabled build succeeded without the generator dependencies.

Startup conversion is excluded from timed prompt throughput. Converting 368 matrices (6120 MiB of original packed weights) took 5.7-8.9 seconds across the validation and benchmark runs. Their planar allocations total 6292 MiB, including padded scales and surface alignment.

`ane-kernel/check-planar.cpp` verifies lossless weight load/readback and 20 Metal matvec/matmul cases, including token counts 1, 2, 8, 32, and 129 and widths 128, 5120, 6144, and 17408:

```sh
xcrun clang++ -std=c++17 -O2 -I ggml/include ane-kernel/check-planar.cpp \
    -L build/bin -lggml -lggml-base -Wl,-rpath,"$PWD/build/bin" \
    -o /tmp/pq2-check-planar
/tmp/pq2-check-planar
```

The earlier experiments, logs, and comparison harness remain under `experiments/coreml-pq2/` as historical material. The current generator and launch scripts are under `ane-kernel/`.

The real-model check uses the first 4096 tokens of `docs/build.md`, followed by eight fixed continuation tokens. All 736 split matmuls passed 94,208 scalar checks; maximum relative RMS error was 0.0013304 for ANE and 0.0000027293 for Metal. All nine top tokens matched the original Metal reference, and maximum KL divergence was 1.01e-6. The queued run with regenerated kernels produced bit-for-bit identical logits to the serial validation run. See `ane-kernel/results/verify-planar-*.json`.

To check the split with the ordinary planar Metal matmul, including both sides of the row boundary:

```sh
GGML_METAL_ANE_MPS_DISABLE=1 GGML_METAL_ANE_VALIDATE=1 \
    ane-kernel/run.sh /tmp/pq2-check-planar --ane
```

The initial planar implementation changed the existing Metal shaders by +72/-17 lines: the shared matmul gains a planar reader, and the existing PQ2 matvec is parameterized for planar addressing. Small batches reuse that matvec. The main additional component is the approximately 200-line planar buffer implementation in `ggml-metal-pq2.m`.
