from __future__ import annotations

import os

import pytest
import torch
from torch import nn

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        os.environ.get("BLOCK_GNN_TEST_CUDA") != "1",
        reason="set BLOCK_GNN_TEST_CUDA=1 to build and test the CUDA extension",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_direct_matches_conv2d(dtype: torch.dtype) -> None:
    from block_gnn.cpp_extension import stencil_conv2d

    torch.manual_seed(17)
    conv = nn.Conv2d(3, 5, 3, padding=1, dtype=dtype, device="cuda").eval()
    input = torch.randn(4, 3, 11, 9, dtype=dtype, device="cuda")
    with torch.inference_mode():
        actual = stencil_conv2d(input, conv.weight, conv.bias, padding=1)
        expected = conv(input)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("size", [8, 28, 64])
def test_cuda_tiled_matches_conv2d(size: int) -> None:
    from block_gnn.cpp_extension import stencil_conv2d_tiled

    torch.manual_seed(23)
    conv = nn.Conv2d(1, 1, 3, padding=1, device="cuda").eval()
    input = torch.randn(7, 1, size, size, device="cuda")
    with torch.inference_mode():
        actual = stencil_conv2d_tiled(input, conv.weight, conv.bias, padding=1)
        expected = conv(input)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def test_cuda_tiled_rejects_multiple_channels() -> None:
    from block_gnn.cpp_extension import stencil_conv2d_tiled

    conv = nn.Conv2d(2, 3, 3, padding=1, device="cuda").eval()
    input = torch.randn(1, 2, 8, 8, device="cuda")
    with pytest.raises(RuntimeError, match="one input and one output channel"):
        stencil_conv2d_tiled(input, conv.weight, conv.bias, padding=1)
