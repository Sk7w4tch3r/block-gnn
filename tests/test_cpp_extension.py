from __future__ import annotations

import os

import pytest
import torch
from torch import nn


@pytest.mark.cpp
@pytest.mark.skipif(
    os.environ.get("BLOCK_GNN_TEST_CPP") != "1",
    reason="set BLOCK_GNN_TEST_CPP=1 to build the optional C++ extension",
)
def test_cpp_stencil_matches_conv2d() -> None:
    from block_gnn.cpp_extension import stencil_conv2d

    torch.manual_seed(11)
    for dtype in (torch.float32, torch.float64):
        conv = nn.Conv2d(2, 3, 3, padding=1, dtype=dtype).eval()
        input = torch.randn(4, 2, 7, 5, dtype=dtype)
        actual = stencil_conv2d(input, conv.weight, conv.bias, padding=1)
        torch.testing.assert_close(actual, conv(input))
