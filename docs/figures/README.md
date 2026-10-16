# Block GNN figures

Five explanatory figures, provided as 1600×1600 PNG images, editable SVGs,
and a five-page [PDF carousel](block_gnn_carousel.pdf). The text for the
accompanying LinkedIn post is in [linkedin_post.txt](../linkedin_post.txt).

Generate everything from the committed benchmark records:

```bash
python -m pip install matplotlib numpy
MPLBACKEND=Agg python benchmarks/render_figures.py
```

The charts use all nine measured CPU/GPU cases. Ratios are calculated from
the median latencies, without filtering out slower cases. The memory chart
reports persistent tensor storage, not peak process or GPU memory. These
figures do not introduce new benchmark measurements. Source JSON hashes are
recorded in [manifest.json](manifest.json).

## Figure descriptions and accessible text

1. **Frozen operator.** A trained convolution is compiled offline into CSR
   indices and values, then applied to batches with sparse multiplication and
   bias. A color-coded 3×3 kernel maps into a 16×16 operator for a 4×4 image;
   matching colors identify repeated trained weights. Boundary entries are
   omitted for zero padding.
2. **CUDA execution.** The direct kernel gives each thread one output and
   accumulates its neighborhood. The tiled kernel cooperatively loads an
   18×18 input patch, including halo, for a 16×16 output tile with a 3×3 kernel.
   The tiled implementation is limited to one input/output channel; both
   implementations use stride one, dilation one, and no groups.
3. **CPU crossover.** All nine CSR speedup ratios against eager Conv2d on
   the i7-12700K, float32, one thread. Speedup peaks at 3.22× for batch 32,
   one input/output channel, 8×8. Larger channel counts and batch-one cases
   often lose. The vertical line marks equal latency; left of it is slower.
4. **CUDA crossover.** All nine GPU cases on the RTX 3060. Direct and tiled
   kernels win on small workloads, peaking at 2.29× and 2.30× respectively.
   cuDNN wins on larger multi-channel batches. CSR loses every GPU case.
   Tiled points are absent for unsupported multi-channel configurations.
5. **Memory trade-off.** A logarithmic storage chart compares compact weights
   plus bias with expanded CSR plus bias for five shapes. The 1→32, 28×28
   case grows from 1.25 KiB to 1.74 MiB. Direct CUDA retains the compact weights;
   tiled CUDA uses temporary shared memory for an input patch.

## Timing interpretation

GPU results use CUDA events around a loop of eager calls. They measure a
stream interval that may contain host launch gaps, not exclusively kernel
execution time. The available data do not isolate dispatch, allocation,
memory traffic, or cuDNN setup as causal explanations. Neither the figures
nor the accompanying post claim training or end-to-end model acceleration.
