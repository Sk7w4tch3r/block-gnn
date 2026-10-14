"""Lazy loader for the optional LibTorch CPU reference kernel."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

_LOADED = False


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


def stencil_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Run the custom CPU reference kernel.

    The kernel currently implements stride-one, dilation-one, ungrouped 2-D
    cross-correlation. It exists as an inspectable LibTorch/dispatcher example;
    the CSR backend is the optimized path demonstrated by this repository.
    """
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    load_cpp_extension(verbose=os.environ.get("BLOCK_GNN_VERBOSE_BUILD") == "1")
    return torch.ops.block_gnn.stencil_conv2d(
        input, weight, bias, padding_h, padding_w
    )
