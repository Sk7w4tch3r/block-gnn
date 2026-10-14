from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from block_gnn import SparseConv2d, compile_conv2d


@pytest.mark.parametrize(
    ("conv", "shape", "batch"),
    [
        (nn.Conv2d(1, 1, 3, padding=1), (5, 7), 1),
        (nn.Conv2d(2, 3, (3, 2), stride=(2, 1), padding=(1, 0)), (8, 9), 4),
        (nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2, bias=False), (9, 8), 2),
        (nn.Conv2d(4, 4, 3, padding=1, groups=4), (6, 6), 3),
    ],
)
def test_matches_conv2d(conv: nn.Conv2d, shape: tuple[int, int], batch: int) -> None:
    torch.manual_seed(7)
    conv.eval()
    sparse = SparseConv2d.from_conv2d(conv, shape).eval()
    input = torch.randn(batch, conv.in_channels, *shape)

    expected = conv(input)
    actual = sparse(input)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert list(sparse.parameters()) == []


def test_compilation_is_a_frozen_snapshot() -> None:
    conv = nn.Conv2d(1, 2, 3, padding=1)
    sparse = SparseConv2d.from_conv2d(conv, (6, 6))
    input = torch.randn(2, 1, 6, 6)
    expected = sparse(input).clone()

    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()

    torch.testing.assert_close(sparse(input), expected)
    assert not torch.allclose(conv(input), expected)


def test_state_dict_round_trip() -> None:
    conv = nn.Conv2d(2, 3, 3, padding=1)
    first = SparseConv2d.from_conv2d(conv, (4, 5))
    second = SparseConv2d.from_conv2d(nn.Conv2d(2, 3, 3, padding=1), (4, 5))
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second.load_state_dict(torch.load(buffer, weights_only=True))
    input = torch.randn(2, 2, 4, 5)
    torch.testing.assert_close(first(input), second(input))


def test_dtype_conversion() -> None:
    conv = nn.Conv2d(1, 2, 3, padding=1)
    sparse = SparseConv2d.from_conv2d(conv, (4, 4)).double()
    conv = conv.double()
    input = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(sparse(input), conv(input), atol=1e-12, rtol=1e-10)


def test_metadata_and_storage_accounting() -> None:
    compiled = compile_conv2d(nn.Conv2d(1, 2, 3, padding=1), (4, 4))
    assert compiled.input_shape == (1, 4, 4)
    assert compiled.output_shape == (2, 4, 4)
    assert compiled.nnz > 0
    assert compiled.storage_bytes > compiled.operator.values().numel() * 4


@pytest.mark.parametrize(
    "input",
    [torch.randn(1, 1, 5, 4), torch.randn(1, 2, 4, 4), torch.randn(1, 1, 4)],
)
def test_rejects_incompatible_input(input: torch.Tensor) -> None:
    sparse = SparseConv2d.from_conv2d(nn.Conv2d(1, 2, 3, padding=1), (4, 4))
    with pytest.raises(ValueError):
        sparse(input)


def test_rejects_nonzero_padding_mode() -> None:
    conv = nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect")
    with pytest.raises(ValueError, match="padding_mode"):
        compile_conv2d(conv, (8, 8))


def test_rejects_unsupported_dtype() -> None:
    conv = nn.Conv2d(1, 1, 3, padding=1).half()
    with pytest.raises(TypeError, match="float32 and float64"):
        compile_conv2d(conv, (8, 8))
