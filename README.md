# Block GNN

[![CI](https://github.com/Sk7w4tch3r/block-gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/Sk7w4tch3r/block-gnn/actions/workflows/ci.yml)

Block GNN is an inference-systems experiment: compile a **trained**
`torch.nn.Conv2d` into a fixed sparse linear operator, then execute it with
sparse-dense matrix multiplication. It also includes an inspectable C++/LibTorch
CPU operator and two CUDA kernels registered through the PyTorch dispatcher.

The project asks a deliberately empirical question:

> For which shapes is ahead-of-time sparse lowering faster than an optimized
> dense convolution, and what memory cost does that speedup require?

It is not a trainable replacement for `Conv2d`, and it does not claim a
universal speedup.

## Result at a glance

On an Intel i7-12700K, PyTorch 2.11, float32, one CPU thread, and batch size 32:

| Layer and input | Dense `Conv2d` | Compiled CSR | Speedup |
|---|---:|---:|---:|
| `1→1`, 8×8 | 44.2 µs | 13.7 µs | **3.22×** |
| `1→1`, 28×28 | 103.5 µs | 37.7 µs | **2.75×** |
| `1→1`, 64×64 | 338.5 µs | 164.3 µs | **2.06×** |
| `1→32`, 8×8 | 57.8 µs | 47.9 µs | **1.21×** |
| `1→32`, 28×28 | 1505.5 µs | 1680.7 µs | 0.90× |
| `3→32`, 32×32 | 608.2 µs | 1342.5 µs | 0.45× |

These are medians from nine samples after warm-up. Each case runs in an
isolated process with a fixed oneDNN primitive cache and MKL dynamic threading
disabled. The complete machine-readable result, including IQR, compilation
time, numerical error, memory, and custom-kernel measurements, is committed in
[`benchmarks/results`](benchmarks/results/i7-12700k-pytorch-2.11-threads1.json).

The conclusion is as important as the speedup: explicit CSR works well for
batched, low-channel filters, but repeated values make it memory-heavy and it
loses as channel count grows. See [the full results and interpretation](docs/RESULTS.md).

On an NVIDIA RTX 3060, PyTorch 2.11/CUDA 12.8, float32:

| Layer and input | cuDNN | CUDA direct | CUDA tiled | Best speedup |
|---|---:|---:|---:|---:|
| `B1, 1→1`, 8×8 | 10.5 µs | 4.8 µs | 4.8 µs | **2.20×** |
| `B32, 1→1`, 28×28 | 10.5 µs | 4.7 µs | 4.6 µs | **2.30×** |
| `B32, 1→1`, 64×64 | 10.6 µs | 11.7 µs | 9.2 µs | **1.16×** |
| `B32, 1→32`, 8×8 | 10.7 µs | 7.0 µs | — | **1.53×** |
| `B1, 1→32`, 28×28 | 10.8 µs | 4.7 µs | — | **2.29×** |
| `B32, 1→32`, 28×28 | 48.1 µs | 61.3 µs | — | 0.78× |

The custom CUDA path wins when a single lightweight kernel avoids cuDNN setup
overhead; cuDNN retakes the lead as arithmetic intensity and channel count grow.
The CUDA CSR/cuSPARSE path lost every measured GPU case. Full data are in
[`rtx3060-pytorch-2.11-cuda.json`](benchmarks/results/rtx3060-pytorch-2.11-cuda.json).

## Use it

```bash
python -m pip install -e .
```

Compile a trained layer once, outside the serving hot path:

```python
import torch
from torch import nn

from block_gnn import SparseConv2d

trained_conv = nn.Conv2d(1, 32, 3, padding=1).eval()
# In a real application, load trained_conv's checkpoint first.

sparse_conv = SparseConv2d.from_conv2d(
    trained_conv,
    input_size=(28, 28),
).eval()

images = torch.randn(32, 1, 28, 28)
with torch.inference_mode():
    expected = trained_conv(images)
    actual = sparse_conv(images)

torch.testing.assert_close(actual, expected)
print(sparse_conv)
```

The compiled module:

- supports bias, rectangular kernels and inputs, stride, dilation, and groups;
- stores its CSR operator and bias as registered buffers;
- participates in `state_dict` serialization and `.to(...)` device/dtype moves;
- validates its fixed input shape, device, and dtype at runtime;
- has no trainable parameters by design.

## Custom PyTorch/LibTorch operators

[`stencil_conv2d.cpp`](src/block_gnn/csrc/stencil_conv2d.cpp) implements a
direct CPU stencil kernel with `AT_DISPATCH_FLOATING_TYPES`,
`at::parallel_for`, `TORCH_LIBRARY`, and `TORCH_LIBRARY_IMPL`. It is compiled
lazily through `torch.utils.cpp_extension`:

```python
from block_gnn.cpp_extension import stencil_conv2d

output = stencil_conv2d(
    images,
    trained_conv.weight,
    trained_conv.bias,
    padding=1,
)
```

[`stencil_conv2d_cuda.cu`](src/block_gnn/csrc/stencil_conv2d_cuda.cu) adds:

- a general grid-stride direct CUDA kernel for ungrouped, stride-one convolution;
- a shared-memory tiled specialization for single-input/single-output filters;
- float32, float64, and float16 accumulation dispatch;
- CUDA stream/device guards, launch validation, and PyTorch CUDA dispatch.

`stencil_conv2d` dispatches to CPU or CUDA based on its input. The specialized
CUDA path is available separately:

```python
from block_gnn.cpp_extension import stencil_conv2d_tiled

tiled_output = stencil_conv2d_tiled(
    images.cuda(),
    trained_conv.weight.cuda(),
    trained_conv.bias.cuda(),
    padding=1,
)
```

The kernels are specialized inference backends, not universal replacements.
Keeping their losing shapes visible is intentional: backend selection should
follow measurements, not the existence of custom code.

## Reproduce the work

```bash
python -m pip install -e '.[dev]'
python -m pytest

# Also compile and test the optional C++ extension.
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-ext \
BLOCK_GNN_TEST_CPP=1 python -m pytest -m cpp

# Compile and test CUDA direct/tiled kernels. CUDA_HOME must match PyTorch's
# CUDA major version; this machine uses the installed CUDA 12.3 toolkit.
CUDA_HOME=/usr/local/cuda-12.3 \
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-cuda-ext \
BLOCK_GNN_TEST_CUDA=1 python -m pytest -m cuda

# Full isolated-process CPU benchmark, including the C++ operator.
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-ext \
python benchmarks/benchmark_inference.py \
  --threads 1 \
  --min-run-time 0.75 \
  --include-cpp \
  --output benchmarks/results/local.json

# Full isolated-process CUDA benchmark against cuDNN and cuSPARSE.
CUDA_HOME=/usr/local/cuda-12.3 \
TORCH_EXTENSIONS_DIR=/tmp/block-gnn-cuda-ext \
python benchmarks/benchmark_cuda.py \
  --min-run-time 0.75 \
  --output benchmarks/results/local-cuda.json
```

This repository makes no multi-GPU, training-throughput, backward-kernel, or
distributed-scaling claim.

## Design

A convolution at a fixed input shape is a linear map. Compilation constructs
the CSR matrix

\[
M \in \mathbb{R}^{(C_{out}H_{out}W_{out}) \times
(C_{in}H_{in}W_{in})}
\]

whose nonzero entries are the trained convolution weights repeated at their
valid spatial locations. Batched inference becomes

\[
Y_{flat} = M X_{flat} + b.
\]

This exact lowering is the baseline. The longer-term systems direction is a
relation-typed/block representation that stores repeated topology and weights
once rather than expanding them into every CSR entry. The trade-offs and
proposed execution model are detailed in [the design document](docs/DESIGN.md).

## Repository map

```text
src/block_gnn/sparse_conv.py       Conv2d → CSR compiler and module
src/block_gnn/csrc/                C++ and CUDA dispatcher implementations
benchmarks/                        isolated-process latency harness and JSON data
tests/                             numerical, serialization, validation, C++ tests
docs/                              design and empirical report
research/gnn_mnist/                preserved future GNN classification direction
research/legacy_notebooks/         cleaned historical experiments
```

The MNIST GNN work is intentionally separated from the supported package. It is
a future research direction, not evidence for the current inference claim.

## Current limitations

- CSR is compiled for one spatial input shape.
- Compilation is offline and currently implemented in Python.
- The CSR backend currently supports float32 and float64.
- Explicit CSR duplicates shared convolution weights at every spatial location.
- The measured crossovers are hardware-, batch-, and shape-dependent.
- The C++ reference kernel currently supports ungrouped, stride-one,
  dilation-one cross-correlation.
- The CUDA direct kernel has the same stride/dilation/group limitations; the
  tiled kernel additionally requires `1→1` channels.
- Backward, grouped CUDA, and compressed relation-typed kernels remain future work.

This repository began as exploratory research in 2024 and was rebuilt into a
tested, reproducible inference prototype in 2026.
