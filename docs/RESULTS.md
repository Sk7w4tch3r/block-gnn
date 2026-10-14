# Empirical report

## Claim

Compiling a fixed convolution into CSR can reduce CPU inference latency for
batched, low-channel workloads. It is not consistently faster than `Conv2d`.
The best measured CSR result in the committed suite is **3.22×**, while
channel-rich cases are **2.0–2.3× slower** and require much more memory.

## Methodology

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

## Results

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

## Interpretation

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

## Reproduction

```bash
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-ext \
OMP_NUM_THREADS=1 \
python benchmarks/benchmark_inference.py \
  --threads 1 \
  --min-run-time 0.75 \
  --include-cpp \
  --output benchmarks/results/local.json
```

Results are hardware- and backend-specific. Run the benchmark on the intended
serving machine before selecting this backend.

## What is not measured

There was no available CUDA device in the test environment. The report makes no
claim about GPU latency, GPU memory, multi-GPU behavior, backward performance,
or training throughput. Those require separate kernels and experiments.
