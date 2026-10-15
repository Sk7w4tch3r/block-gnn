# Design: convolution as a compiled sparse operator

## Contract

`SparseConv2d` is an ahead-of-time inference transformation. Its source is a
trained `nn.Conv2d`; its output is a frozen module specialized to one input
height and width. Training happens before compilation.

This contract makes three properties explicit:

1. The topology is static and can be constructed once.
2. The trained weights are immutable and can be copied into an inference layout.
3. Conversion latency can be amortized over repeated inference calls.

## Exact lowering

For one sample, flatten an NCHW tensor into a vector `x`. Every output element
of `Conv2d` is a weighted sum of selected input elements. Therefore there is a
sparse matrix `M` such that:

```text
y = M @ x + bias
```

For a batch, samples are columns of a dense matrix:

```text
X: [C_in * H_in * W_in, batch]
M: [C_out * H_out * W_out, C_in * H_in * W_in]
Y = M @ X
```

The compiler emits CSR rows in contiguous output-channel/row/column order. For
each output position it walks the valid input channels and kernel coordinates,
accounts for padding, stride, dilation, and groups, and inserts the corresponding
trained weight. Boundary conditions are represented by omitted entries rather
than explicit zeros.

## Why int32 CSR indices

Index memory and bandwidth are significant for a matrix whose values are all
float32. PyTorch accepts int32 CSR indices, which halves index storage compared
with the default int64 representation. The compiler rejects an operator that
would exceed the int32 range; such an operator would already be impractical to
materialize on the intended hardware.

## The expansion problem

Ignoring boundaries, explicit CSR stores approximately

```text
nnz = C_out * C_in/groups * K_h * K_w * H_out * W_out
```

entries. A conventional convolution stores each kernel weight once; CSR repeats
it at every output position. In the committed benchmark, the `1→32, 28×28`
layer expands 1.25 KiB of weights and bias into a 1.74 MiB operator. The
`3→32, 32×32` operator occupies 6.60 MiB.

This is why CSR is a useful correctness/performance baseline rather than the
final representation.

## Repeated-relation representation

A non-expanded formulation associates every edge with a small relation ID:

```text
(source_node, destination_node, relation_id)
weight_table[relation_id, output_channel, input_channel]
```

For a regular 3×3 grid, the relation vocabulary is the nine kernel offsets. A
semi-regular graph can use a small finite vocabulary of oriented local
relationships. Its computation is:

\[
Y = \sum_{r \in R} A_r X W_r + b,
\]

where `A_r` is topology-only and `W_r` stores learned channel mixing. This is
also the clean connection to relation-typed graph message passing.

A useful custom kernel would fuse edge traversal, relation lookup, channel
mixing, and reduction without materializing the expanded CSR values. That is
the main future optimization suggested by the negative high-channel CSR result.

## C++/LibTorch and CUDA kernels

The bundled C++ operator demonstrates the mechanics of extending PyTorch:

- schema definition with `TORCH_LIBRARY`;
- CPU dispatch with `TORCH_LIBRARY_IMPL`;
- dtype dispatch with `AT_DISPATCH_FLOATING_TYPES`;
- work partitioning with `at::parallel_for`;
- validation with `TORCH_CHECK`;
- lazy compilation and registration through `torch.utils.cpp_extension`.

It directly evaluates the repeated stencil and is numerically exact against
`Conv2d` within floating-point tolerance. Profiling shows that its simple loop
nest only wins for very small single-filter CPU cases.

The CUDA library registers two implementations under the same dispatcher
namespace:

1. **Direct grid-stride kernel.** Each CUDA thread owns one NCHW output element
   and loops over input channels and kernel coordinates. It avoids expanded CSR
   storage and minimizes setup for low-work inference.
2. **Shared-memory tiled kernel.** A 16×16 output tile cooperatively loads its
   input region and halo, then reuses those values across neighboring outputs.
   The current specialization handles `1→1`, stride-one, dilation-one filters.

Both kernels use the current PyTorch CUDA stream, `CUDAGuard`, checked launches,
and accumulation types appropriate to float32, float64, and float16. The direct
kernel reaches 2.29× over cuDNN in the measured low-work regime; it loses once
channel reuse and tensor-core-friendly computation dominate. The tiled kernel
extends the single-filter crossover through the tested 64×64 input.

## Backend-selection direction

The results suggest a shape-aware policy rather than a replacement operator:

- native `Conv2d` for single samples and channel-rich layers;
- CSR for batched, very low-channel filters when measured to win;
- direct CUDA for launch-bound, low-work layers;
- tiled CUDA for measured single-filter shapes;
- a future channel-tiled/relation-typed kernel where CSR expansion dominates;
- conversion refused when expected storage exceeds a configured budget.

A production implementation would benchmark/cache the choice by device, dtype,
shape, batch range, and PyTorch/backend version.
