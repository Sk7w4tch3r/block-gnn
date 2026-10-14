"""Ahead-of-time lowering of a frozen ``nn.Conv2d`` to sparse matrix multiply."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn


class SpatialShape(NamedTuple):
    height: int
    width: int


@dataclass(frozen=True)
class CompiledConv2d:
    """Topology and values produced by :func:`compile_conv2d`."""

    operator: Tensor
    bias: Tensor | None
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]

    @property
    def nnz(self) -> int:
        return self.operator._nnz()

    @property
    def storage_bytes(self) -> int:
        """Bytes occupied by CSR values and index arrays."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.operator.values(),
                self.operator.col_indices(),
                self.operator.crow_indices(),
            )
        )


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return value, value


def _output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def compile_conv2d(
    conv: nn.Conv2d,
    input_size: tuple[int, int] | SpatialShape,
    *,
    index_device: torch.device | str = "cpu",
) -> CompiledConv2d:
    """Compile a frozen ``Conv2d`` into an exact CSR linear operator.

    The resulting operator has shape ``(C_out*H_out*W_out, C_in*H_in*W_in)``.
    Its values are copied from ``conv`` and are intentionally detached: this is
    an inference transformation, not a trainable replacement layer.

    Args:
        conv: Source layer. Groups, stride, dilation, padding, and optional bias
            are supported. Only zero padding is representable as a sparse linear
            operator without a separate preprocessing step.
        input_size: Fixed ``(height, width)`` accepted by the compiled operator.
        index_device: Device on which to construct and store the CSR operator.

    Returns:
        A :class:`CompiledConv2d` containing the CSR operator and shape metadata.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"expected nn.Conv2d, got {type(conv).__name__}")
    if conv.padding_mode != "zeros":
        raise ValueError("only padding_mode='zeros' is supported")
    if conv.weight.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "the CSR backend currently supports float32 and float64 weights, "
            f"got {conv.weight.dtype}"
        )

    height, width = map(int, input_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"input_size must be positive, got {(height, width)}")

    stride_h, stride_w = _pair(conv.stride)
    pad_h, pad_w = _pair(conv.padding)
    dilation_h, dilation_w = _pair(conv.dilation)
    out_channels, channels_per_group, kernel_h, kernel_w = conv.weight.shape
    out_h = _output_size(height, kernel_h, stride_h, pad_h, dilation_h)
    out_w = _output_size(width, kernel_w, stride_w, pad_w, dilation_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"convolution produces invalid output size {(out_h, out_w)}")

    weight = conv.weight.detach().to(device="cpu")
    out_channels_per_group = conv.out_channels // conv.groups

    crow_indices = [0]
    col_indices: list[int] = []
    values: list[float] = []

    # CSR rows follow the contiguous NCHW order of one output sample.
    for out_channel in range(out_channels):
        group = out_channel // out_channels_per_group
        input_channel_start = group * channels_per_group
        for out_y in range(out_h):
            for out_x in range(out_w):
                for in_group_channel in range(channels_per_group):
                    in_channel = input_channel_start + in_group_channel
                    for kernel_y in range(kernel_h):
                        in_y = out_y * stride_h - pad_h + kernel_y * dilation_h
                        if in_y < 0 or in_y >= height:
                            continue
                        for kernel_x in range(kernel_w):
                            in_x = out_x * stride_w - pad_w + kernel_x * dilation_w
                            if in_x < 0 or in_x >= width:
                                continue
                            col_indices.append((in_channel * height + in_y) * width + in_x)
                            values.append(
                                weight[
                                    out_channel,
                                    in_group_channel,
                                    kernel_y,
                                    kernel_x,
                                ].item()
                            )
                crow_indices.append(len(col_indices))

    device = torch.device(index_device)
    value_tensor = torch.tensor(values, dtype=weight.dtype, device=device)
    # int32 is sufficient for operators that fit practical host/device memory
    # and halves index traffic relative to PyTorch's default int64 indices.
    max_index = max(len(values), conv.in_channels * height * width)
    if max_index >= torch.iinfo(torch.int32).max:
        raise OverflowError("compiled operator exceeds the int32 CSR index range")
    col_tensor = torch.tensor(col_indices, dtype=torch.int32, device=device)
    crow_tensor = torch.tensor(crow_indices, dtype=torch.int32, device=device)
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        operator = torch.sparse_csr_tensor(
            crow_tensor,
            col_tensor,
            value_tensor,
            size=(conv.out_channels * out_h * out_w, conv.in_channels * height * width),
            device=device,
            dtype=weight.dtype,
        )
    bias = None if conv.bias is None else conv.bias.detach().to(device=device).clone()
    return CompiledConv2d(
        operator=operator,
        bias=bias,
        input_shape=(conv.in_channels, height, width),
        output_shape=(conv.out_channels, out_h, out_w),
    )


class SparseConv2d(nn.Module):
    """Frozen, fixed-shape ``Conv2d`` executed as CSR sparse-dense matmul.

    Use :meth:`from_conv2d` after training. The module has no trainable
    parameters; its CSR operator and bias are registered buffers, so they are
    included in ``state_dict`` and follow normal ``Module.to(...)`` semantics.
    """

    def __init__(self, compiled: CompiledConv2d) -> None:
        super().__init__()
        self.input_shape = compiled.input_shape
        self.output_shape = compiled.output_shape
        self.register_buffer("operator", compiled.operator)
        self.register_buffer("bias", compiled.bias)

    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        input_size: tuple[int, int] | SpatialShape,
        *,
        device: torch.device | str | None = None,
    ) -> SparseConv2d:
        """Compile ``conv`` once and return an inference-only module."""
        target = conv.weight.device if device is None else torch.device(device)
        return cls(compile_conv2d(conv, input_size, index_device=target))

    @property
    def nnz(self) -> int:
        return self.operator._nnz()

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.operator.values(),
            self.operator.col_indices(),
            self.operator.crow_indices(),
        )
        total = sum(t.numel() * t.element_size() for t in tensors)
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(input.shape)}")
        if tuple(input.shape[1:]) != self.input_shape:
            raise ValueError(
                f"compiled for (*, {self.input_shape}), got input shape {tuple(input.shape)}"
            )
        if input.device != self.operator.device:
            raise ValueError(
                f"input is on {input.device}, but compiled operator is on {self.operator.device}"
            )
        if input.dtype != self.operator.dtype:
            raise ValueError(
                f"input has dtype {input.dtype}, but compiled operator has {self.operator.dtype}"
            )

        batch = input.shape[0]
        flattened = input.reshape(batch, -1).transpose(0, 1).contiguous()
        output = torch.sparse.mm(self.operator, flattened).transpose(0, 1)
        output = output.reshape(batch, *self.output_shape)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def extra_repr(self) -> str:
        mib = self.storage_bytes / (1024**2)
        return (
            f"input_shape={self.input_shape}, output_shape={self.output_shape}, "
            f"nnz={self.nnz:,}, storage={mib:.2f} MiB"
        )
