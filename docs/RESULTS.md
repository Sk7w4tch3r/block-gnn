# Empirical report

The [visual walkthrough](figures/README.md) plots these saved measurements and
explains the operator transformation and CUDA execution. All measured shapes,
including slower cases, appear in the CPU and GPU crossover figures.

## Claim

Compiling a fixed convolution into CSR can reduce CPU inference latency for
batched, low-channel workloads. It is not consistently faster than `Conv2d`.
The best measured CSR result in the committed suite is **3.22×**, while
channel-rich cases are **2.0–2.3× slower** and require much more memory.

On CUDA, avoiding the general-library setup path with a specialized direct
stencil reaches **2.29× over cuDNN** in the measured low-work regime, while the
tiled specialization reaches **2.30×**. cuDNN wins for larger batches with
multiple output channels, and expanded CSR/cuSPARSE
loses every measured GPU case.

## CPU methodology

- CPU: Intel Core i7-12700K
- OS: Linux 6.8, x86-64
- Python: 3.11.4
- PyTorch: 2.11.0+cu128, CPU execution
- dtype: float32
- intra-op threads: 1; inter-op threads: 1
- oneDNN primitive cache: 1024 entries
- MKL dynamic threading: disabled
- warm-up: 25 calls per implementation
- samples: nine independently timed groups per implementation
- statistic: median latency; IQR retained in JSON
- isolation: every shape runs in a fresh subprocess
- correctness: `torch.allclose(atol=3e-5, rtol=1e-5)` before timing

Compilation is excluded from hot-path latency and reported separately in the
JSON. Dense and sparse calls both include their complete module invocation,
layout handling, output reshape, and bias.

## CPU results

| Case | Shape | Dense | CSR | CSR speedup | CSR memory | C++ direct | C++ speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| tiny-single | `B1, 1→1, 8×8` | 5.9 µs | 11.6 µs | 0.51× | <0.01 MiB | 3.4 µs | **1.74×** |
| tiny-batch | `B32, 1→1, 8×8` | 44.2 µs | 13.7 µs | **3.22×** | <0.01 MiB | 29.9 µs | **1.48×** |
| filter-28 | `B32, 1→1, 28×28` | 103.5 µs | 37.7 µs | **2.75×** | 0.05 MiB | 382.7 µs | 0.27× |
| filter-64 | `B32, 1→1, 64×64` | 338.5 µs | 164.3 µs | **2.06×** | 0.29 MiB | 2008.7 µs | 0.17× |
| mnist-small | `B32, 1→32, 8×8` | 57.8 µs | 47.9 µs | **1.21×** | 0.13 MiB | 874.2 µs | 0.07× |
| mnist-single | `B1, 1→32, 28×28` | 12.1 µs | 146.0 µs | 0.08× | 1.74 MiB | 386.1 µs | 0.03× |
| mnist-batch | `B32, 1→32, 28×28` | 1505.5 µs | 1680.7 µs | 0.90× | 1.74 MiB | 12985.5 µs | 0.12× |
| rgb-stem | `B32, 3→32, 32×32` | 608.2 µs | 1342.5 µs | 0.45× | 6.60 MiB | 45858.8 µs | 0.01× |
| feature-map | `B32, 16→32, 16×16` | 626.9 µs | 1414.2 µs | 0.44× | 8.30 MiB | 59486.9 µs | 0.01× |

Maximum absolute error across the CSR cases was `1.43e-6`. Complete precision,
IQR, call-count, nonzero-count, compilation-time, and storage data are in
[`i7-12700k-pytorch-2.11-threads1.json`](../benchmarks/results/i7-12700k-pytorch-2.11-threads1.json).

## CPU interpretation

1. **Batching is decisive.** Sparse lowering loses for one tiny sample but
   amortizes well when 32 samples become the dense right-hand side of one SpMM.
2. **The original single-filter hypothesis is supported.** At batch 32, CSR is
   2.06–3.22× faster for tested `1→1` filters from 8×8 through 64×64.
3. **A complete layer changes the result.** The full `1→32, 28×28` layer is
   close but still 10% slower than `Conv2d`; RGB and feature-map cases lose.
4. **Explicit CSR is memory-inefficient.** It duplicates shared weights per
   spatial position. This motivates the repeated-relation/block representation.
5. **The first custom C++ loop is not the final kernel.** It wins only where
   dispatch overhead dominates, then falls behind optimized dense convolution.

## CUDA methodology

- GPU: NVIDIA GeForce RTX 3060, 12 GiB, compute capability 8.6
- driver: 560.35.03
- PyTorch: 2.11.0+cu128
- CUDA toolkit used for extension compilation: 12.3
- cuDNN: 9.19
- dtype: float32
- cuDNN benchmark/autotuning: enabled
- cuDNN TF32: enabled; matmul TF32: disabled
- warm-up: 50 calls per implementation
- timing: CUDA events on the current stream
- samples: nine independently timed groups per implementation
- isolation: every shape runs in a fresh subprocess
- correctness: `torch.allclose(atol=3e-5, rtol=1e-5)` before timing

CUDA timing uses events around repeated eager operator calls. The measured
stream interval can include gaps while the host enqueues work; it is not an
isolated measurement of kernel execution or end-to-end application latency.
The calls allocate their outputs. The benchmark uses seeded synthetic inputs
and weights rather than a trained checkpoint or a full model accuracy study.
Extension compilation and CSR construction/device transfer are outside the hot
path. The direct kernel executes one output element per CUDA thread. The tiled
kernel loads an input tile plus halo into shared memory and currently supports
only `1→1` convolutions.

## CUDA results

| Case | Shape | cuDNN | cuSPARSE CSR | CSR speedup | CUDA direct | Direct speedup | CUDA tiled | Tiled speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny-single | `B1, 1→1, 8×8` | 10.5 µs | 20.8 µs | 0.51× | 4.8 µs | **2.20×** | 4.8 µs | **2.20×** |
| tiny-batch | `B32, 1→1, 8×8` | 10.0 µs | 24.3 µs | 0.41× | 4.6 µs | **2.16×** | 4.7 µs | **2.16×** |
| filter-28 | `B32, 1→1, 28×28` | 10.5 µs | 24.8 µs | 0.43× | 4.7 µs | **2.27×** | 4.6 µs | **2.30×** |
| filter-64 | `B32, 1→1, 64×64` | 10.6 µs | 28.4 µs | 0.37× | 11.7 µs | 0.90× | 9.2 µs | **1.16×** |
| mnist-small | `B32, 1→32, 8×8` | 10.7 µs | 24.9 µs | 0.43× | 7.0 µs | **1.53×** | — | — |
| mnist-single | `B1, 1→32, 28×28` | 10.8 µs | 21.0 µs | 0.51× | 4.7 µs | **2.29×** | — | — |
| mnist-batch | `B32, 1→32, 28×28` | 48.1 µs | 109.7 µs | 0.44× | 61.3 µs | 0.78× | — | — |
| rgb-stem | `B32, 3→32, 32×32` | 59.1 µs | 274.4 µs | 0.22× | 175.5 µs | 0.34× | — | — |
| feature-map | `B32, 16→32, 16×16` | 27.9 µs | 294.8 µs | 0.09× | 211.5 µs | 0.13× | — | — |

Maximum absolute error across the CUDA suite was `1.67e-6`. The extension test
suite also validates float16 direct-kernel output. Complete IQR, call counts,
operator memory, compilation time, and per-backend error are in
[`rtx3060-pytorch-2.11-cuda.json`](../benchmarks/results/rtx3060-pytorch-2.11-cuda.json).

## CUDA interpretation

1. **Launch/setup overhead defines the winning regime.** One direct stencil
   launch takes about 4.6–4.8 µs, versus roughly 10–11 µs for cuDNN on the small
   shapes. This produces a real 2× crossover without reducing arithmetic.
2. **Shared memory extends the single-filter regime.** At `B32, 64×64`, the
   direct kernel loses, while input-tile reuse keeps the tiled kernel 1.16× ahead.
3. **cuDNN wins once channel reuse matters.** The simple direct loop does not
   tile over channels or exploit tensor cores, so it loses on the full batched
   `1→32, 28×28`, RGB, and feature-map cases.
4. **Expanded CSR is a CPU-specific win here.** cuSPARSE launch and metadata
   overhead dominate every tested GPU workload; its best result is only 0.51×.
5. **Dispatch must be shape-aware.** A production wrapper should select the
   custom direct/tiled kernel only inside a measured device/dtype/shape envelope.

## Reproduction

```bash
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-ext \
OMP_NUM_THREADS=1 \
python benchmarks/benchmark_inference.py \
  --threads 1 \
  --min-run-time 0.75 \
  --include-cpp \
  --output benchmarks/results/local.json

CUDA_HOME=/usr/local/cuda-12.3 \
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-cuda-ext \
python benchmarks/benchmark_cuda.py \
  --min-run-time 0.75 \
  --output benchmarks/results/local-cuda.json
```

Results are hardware- and backend-specific. Run the benchmark on the intended
serving machine before selecting this backend.

## What is not measured

The report makes no claim about backward performance, training throughput,
multi-GPU behavior, distributed scaling, or GPUs other than the RTX 3060.
Those require separate kernels and experiments.
