"""Lazy loaders for the optional LibTorch CPU and CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

_LOADED = False
_CUDA_LOADED = False


def load_cpp_extension(*, verbose: bool = False) -> None:
    """Compile and register ``torch.ops.block_gnn.stencil_conv2d`` once.

    PyTorch caches the shared library in ``TORCH_EXTENSIONS_DIR`` (or its
    standard user cache). Set that environment variable to control build output.
    """
    global _LOADED
    if _LOADED:
        return
    source = Path(__file__).with_name("csrc") / "stencil_conv2d.cpp"
    if not source.exists():
        raise FileNotFoundError(f"C++ source is missing: {source}")
    load(
        name="block_gnn_cpp",
        sources=[str(source)],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=verbose,
    )
    _LOADED = True


def load_cuda_extension(*, verbose: bool = False) -> None:
    """Compile and register the CUDA implementations once.

    ``CUDA_HOME`` must point to a toolkit with the same major CUDA version as
    the installed PyTorch build. The CPU extension is loaded first because it
    owns the operator schemas; the CUDA library adds dispatcher implementations.
    """
    global _CUDA_LOADED
    if _CUDA_LOADED:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch process")
    load_cpp_extension(verbose=verbose)
    source = Path(__file__).with_name("csrc") / "stencil_conv2d_cuda.cu"
    if not source.exists():
        raise FileNotFoundError(f"CUDA source is missing: {source}")
    load(
        name="block_gnn_cuda",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--threads=2"],
        is_python_module=False,
        verbose=verbose,
    )
    _CUDA_LOADED = True


def stencil_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Run the custom direct CPU or CUDA kernel selected by input device.

    The kernel currently implements stride-one, dilation-one, ungrouped 2-D
    cross-correlation. It exists as an inspectable LibTorch/dispatcher example
    and a measured low-work inference specialization.
    """
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    verbose = os.environ.get("BLOCK_GNN_VERBOSE_BUILD") == "1"
    if input.is_cuda:
        load_cuda_extension(verbose=verbose)
    else:
        load_cpp_extension(verbose=verbose)
    return torch.ops.block_gnn.stencil_conv2d(
        input, weight, bias, padding_h, padding_w
    )


def stencil_conv2d_tiled(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Run the shared-memory CUDA specialization for ``1→1`` convolution."""
    if not input.is_cuda:
        raise ValueError("the tiled implementation is CUDA-only")
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    load_cuda_extension(verbose=os.environ.get("BLOCK_GNN_VERBOSE_BUILD") == "1")
    return torch.ops.block_gnn.stencil_conv2d_tiled(
        input, weight, bias, padding_h, padding_w
    )
