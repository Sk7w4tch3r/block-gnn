#!/usr/bin/env python3
"""CUDA inference benchmark for cuDNN, cuSPARSE CSR, and custom kernels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from block_gnn import SparseConv2d
from block_gnn.cpp_extension import load_cuda_extension


@dataclass(frozen=True)
class Case:
    name: str
    in_channels: int
    out_channels: int
    size: int
    batch: int
    kernel: int = 3


CASES = [
    Case("tiny-single", 1, 1, 8, 1),
    Case("tiny-batch", 1, 1, 8, 32),
    Case("filter-28", 1, 1, 28, 32),
    Case("filter-64", 1, 1, 64, 32),
    Case("mnist-small", 1, 32, 8, 32),
    Case("mnist-single", 1, 32, 28, 1),
    Case("mnist-batch", 1, 32, 28, 32),
    Case("rgb-stem", 3, 32, 32, 32),
    Case("feature-map", 16, 32, 16, 32),
]


def measure_cuda(function: Callable[[], torch.Tensor], seconds: float) -> dict:
    """Measure asynchronous CUDA work with events on the current stream."""
    for _ in range(50):
        function()
    torch.cuda.synchronize()

    calibration_start = torch.cuda.Event(enable_timing=True)
    calibration_end = torch.cuda.Event(enable_timing=True)
    calibration_start.record()
    for _ in range(20):
        function()
    calibration_end.record()
    calibration_end.synchronize()
    estimate_seconds = calibration_start.elapsed_time(calibration_end) / 20 / 1e3

    repeats = 9
    calls = max(1, min(100_000, int(seconds / repeats / max(estimate_seconds, 1e-9))))
    samples_us = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            function()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3 / calls)
    quartiles = statistics.quantiles(samples_us, n=4)
    return {
        "median_us": statistics.median(samples_us),
        "iqr_us": quartiles[2] - quartiles[0],
        "measurements": repeats,
        "calls_per_measurement": calls,
    }


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> float:
    max_error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=3e-5, rtol=1e-5):
        raise AssertionError(f"{name} mismatch: max absolute error {max_error}")
    return max_error


def run_case(case: Case, device: torch.device, seconds: float) -> dict:
    torch.manual_seed(0)
    cpu_conv = nn.Conv2d(
        case.in_channels,
        case.out_channels,
        case.kernel,
        padding=case.kernel // 2,
        bias=True,
    ).eval()
    start = time.perf_counter()
    sparse = SparseConv2d.from_conv2d(cpu_conv, (case.size, case.size)).to(device).eval()
    compile_ms = (time.perf_counter() - start) * 1e3
    conv = cpu_conv.to(device).eval()
    input = torch.randn(
        case.batch,
        case.in_channels,
        case.size,
        case.size,
        device=device,
    )
    padding = case.kernel // 2

    with torch.inference_mode():
        expected = conv(input)
        csr_output = sparse(input)
        direct_output = torch.ops.block_gnn.stencil_conv2d(
            input, conv.weight, conv.bias, padding, padding
        )
        csr_error = compare(csr_output, expected, f"{case.name}/CSR")
        direct_error = compare(direct_output, expected, f"{case.name}/direct")
        dense = measure_cuda(lambda: conv(input), seconds)
        csr = measure_cuda(lambda: sparse(input), seconds)
        direct = measure_cuda(
            lambda: torch.ops.block_gnn.stencil_conv2d(
                input, conv.weight, conv.bias, padding, padding
            ),
            seconds,
        )

        result = {
            **asdict(case),
            "compile_ms": compile_ms,
            "weight_bytes": sum(tensor_bytes(parameter) for parameter in conv.parameters()),
            "csr_bytes": sparse.storage_bytes,
            "nnz": sparse.nnz,
            "dense": dense,
            "csr": csr,
            "direct": direct,
            "csr_speedup": dense["median_us"] / csr["median_us"],
            "direct_speedup": dense["median_us"] / direct["median_us"],
            "csr_max_abs_error": csr_error,
            "direct_max_abs_error": direct_error,
        }

        if case.in_channels == 1 and case.out_channels == 1:
            tiled_output = torch.ops.block_gnn.stencil_conv2d_tiled(
                input, conv.weight, conv.bias, padding, padding
            )
            tiled_error = compare(tiled_output, expected, f"{case.name}/tiled")
            tiled = measure_cuda(
                lambda: torch.ops.block_gnn.stencil_conv2d_tiled(
                    input, conv.weight, conv.bias, padding, padding
                ),
                seconds,
            )
            result.update(
                {
                    "tiled": tiled,
                    "tiled_speedup": dense["median_us"] / tiled["median_us"],
                    "tiled_max_abs_error": tiled_error,
                }
            )
    return result


def gpu_environment(device: torch.device, command: str) -> dict:
    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device.index or 0}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": os.environ.get("CUDA_HOME", "auto-detected"),
        "driver": driver,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "device": str(device),
        "dtype": "float32",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "git_commit": os.environ.get("GIT_COMMIT", "working-tree"),
        "command": command,
    }


def print_table(results: list[dict]) -> None:
    header = (
        f"{'case':<16} {'shape':<20} {'cuDNN us':>10} {'CSR us':>10} {'CSR x':>7} "
        f"{'direct us':>10} {'dir x':>7} {'tiled us':>10} {'tile x':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        shape = (
            f"{result['batch']}x{result['in_channels']}->{result['out_channels']} "
            f"@{result['size']}"
        )
        tiled_us = result.get("tiled", {}).get("median_us")
        tiled_speedup = result.get("tiled_speedup")
        tiled_us_text = "-" if tiled_us is None else f"{tiled_us:.1f}"
        tiled_x_text = "-" if tiled_speedup is None else f"{tiled_speedup:.2f}"
        print(
            f"{result['name']:<16} {shape:<20} "
            f"{result['dense']['median_us']:>10.1f} {result['csr']['median_us']:>10.1f} "
            f"{result['csr_speedup']:>7.2f} {result['direct']['median_us']:>10.1f} "
            f"{result['direct_speedup']:>7.2f} {tiled_us_text:>10} {tiled_x_text:>7}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true", help="run only low-channel cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker-case",
        choices=[case.name for case in CASES],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    load_cuda_extension(verbose=os.environ.get("BLOCK_GNN_VERBOSE_BUILD") == "1")
    case = next(case for case in CASES if case.name == args.worker_case)
    with torch.inference_mode():
        result = run_case(case, device, args.min_run_time)
    return {
        "schema_version": 1,
        "environment": gpu_environment(device, " ".join(sys.argv)),
        "results": [result],
    }


def run_isolated(args: argparse.Namespace) -> dict:
    cases = CASES[:4] if args.quick else CASES
    results = []
    environment = None
    with tempfile.TemporaryDirectory(prefix="block-gnn-cuda-benchmark-") as directory:
        for case in cases:
            output = Path(directory) / f"{case.name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--device",
                args.device,
                "--min-run-time",
                str(args.min_run_time),
                "--worker-case",
                case.name,
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                print(completed.stdout, end="", file=sys.stderr)
                print(completed.stderr, end="", file=sys.stderr)
                raise SystemExit(completed.returncode)
            worker = json.loads(output.read_text())
            environment = worker["environment"]
            results.extend(worker["results"])
    assert environment is not None
    environment["command"] = " ".join(sys.argv)
    return {"schema_version": 1, "environment": environment, "results": results}


def main() -> None:
    args = parse_args()
    document = run_worker(args) if args.worker_case else run_isolated(args)
    if not args.worker_case:
        print_table(document["results"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        if not args.worker_case:
            print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
