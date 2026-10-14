#!/usr/bin/env python3
"""Reproducible CPU inference benchmark for dense, CSR, and optional C++ paths."""

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

# Make backend selection independent of case ordering. These defaults must be
# applied before importing torch; callers can override either variable.
# oneDNN defaults to a 1024-entry cache in current PyTorch builds. Make that
# choice explicit so subprocesses use the same steady-state inference path.
os.environ.setdefault("DNNL_PRIMITIVE_CACHE_CAPACITY", "1024")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import torch
from torch import nn

from block_gnn import SparseConv2d


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


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def measure(function: Callable[[], torch.Tensor], seconds: float) -> dict:
    """Measure repeated CPU calls without changing thread-pool state.

    ``torch.utils.benchmark.Timer`` normally provides this machinery, but it
    temporarily changes PyTorch's thread count. On oneDNN that can change the
    selected primitive and make the result depend on benchmark order. This
    harness fixes the thread pool once at process start and reports the median
    and interquartile range over nine independent samples.
    """
    for _ in range(25):
        function()
    calibration_start = time.perf_counter_ns()
    for _ in range(5):
        function()
    estimate_seconds = (time.perf_counter_ns() - calibration_start) / 5e9
    repeats = 9
    calls = max(1, int(seconds / repeats / max(estimate_seconds, 1e-9)))
    samples_us = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            function()
        samples_us.append((time.perf_counter_ns() - start) / calls / 1e3)
    quartiles = statistics.quantiles(samples_us, n=4)
    return {
        "median_us": statistics.median(samples_us),
        "iqr_us": quartiles[2] - quartiles[0],
        "measurements": repeats,
        "calls_per_measurement": calls,
    }


def run_case(case: Case, threads: int, seconds: float, include_cpp: bool) -> dict:
    torch.manual_seed(0)
    conv = nn.Conv2d(
        case.in_channels,
        case.out_channels,
        case.kernel,
        padding=case.kernel // 2,
        bias=True,
    ).eval()
    input = torch.randn(case.batch, case.in_channels, case.size, case.size)

    start = time.perf_counter()
    sparse = SparseConv2d.from_conv2d(conv, (case.size, case.size)).eval()
    compile_ms = (time.perf_counter() - start) * 1e3

    with torch.inference_mode():
        dense_output = conv(input)
        sparse_output = sparse(input)
        for _ in range(25):
            conv(input)
            sparse(input)
    max_error = (dense_output - sparse_output).abs().max().item()
    if not torch.allclose(dense_output, sparse_output, atol=3e-5, rtol=1e-5):
        raise AssertionError(f"{case.name}: dense/CSR mismatch, max error {max_error}")

    dense = measure(lambda: conv(input), seconds)
    csr = measure(lambda: sparse(input), seconds)
    result = {
        **asdict(case),
        "compile_ms": compile_ms,
        "max_abs_error": max_error,
        "weight_bytes": sum(p.numel() * p.element_size() for p in conv.parameters()),
        "csr_bytes": sparse.storage_bytes,
        "nnz": sparse.nnz,
        "dense": dense,
        "csr": csr,
        "csr_speedup": dense["median_us"] / csr["median_us"],
    }

    if include_cpp:
        from block_gnn.cpp_extension import load_cpp_extension

        load_cpp_extension()
        padding = case.kernel // 2
        with torch.inference_mode():
            cpp_output = torch.ops.block_gnn.stencil_conv2d(
                input, conv.weight, conv.bias, padding, padding
            )
        cpp_error = (dense_output - cpp_output).abs().max().item()
        if not torch.allclose(dense_output, cpp_output, atol=3e-5, rtol=1e-5):
            raise AssertionError(f"{case.name}: dense/C++ mismatch, max error {cpp_error}")
        cpp = measure(
            lambda: torch.ops.block_gnn.stencil_conv2d(
                input, conv.weight, conv.bias, padding, padding
            ),
            seconds,
        )
        result.update(
            {
                "cpp_max_abs_error": cpp_error,
                "cpp": cpp,
                "cpp_speedup": dense["median_us"] / cpp["median_us"],
            }
        )
    return result


def print_table(results: list[dict], include_cpp: bool) -> None:
    header = (
        f"{'case':<16} {'shape':<20} {'dense us':>10} {'CSR us':>10} "
        f"{'CSR x':>8} {'CSR MiB':>9}"
    )
    if include_cpp:
        header += f" {'C++ us':>10} {'C++ x':>8}"
    print(header)
    print("-" * len(header))
    for result in results:
        shape = (
            f"{result['batch']}x{result['in_channels']}->{result['out_channels']} "
            f"@{result['size']}"
        )
        row = (
            f"{result['name']:<16} {shape:<20} "
            f"{result['dense']['median_us']:>10.1f} {result['csr']['median_us']:>10.1f} "
            f"{result['csr_speedup']:>8.2f} {result['csr_bytes'] / 1024**2:>9.2f}"
        )
        if include_cpp:
            row += (
                f" {result['cpp']['median_us']:>10.1f} "
                f"{result['cpp_speedup']:>8.2f}"
            )
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--include-cpp", action="store_true")
    parser.add_argument("--quick", action="store_true", help="run only the first five cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker-case",
        choices=[case.name for case in CASES],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def environment(args: argparse.Namespace) -> dict:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "threads": args.threads,
        "interop_threads": 1,
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "dnnl_primitive_cache_capacity": os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"],
        "mkl_dynamic": os.environ["MKL_DYNAMIC"],
        "git_commit": os.environ.get("GIT_COMMIT", "working-tree"),
        "command": " ".join(sys.argv),
    }


def run_worker(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    case = next(case for case in CASES if case.name == args.worker_case)
    with torch.inference_mode():
        result = run_case(case, args.threads, args.min_run_time, args.include_cpp)
    return {
        "schema_version": 1,
        "environment": environment(args),
        "results": [result],
    }


def run_isolated(args: argparse.Namespace) -> dict:
    cases = CASES[:5] if args.quick else CASES
    results = []
    with tempfile.TemporaryDirectory(prefix="block-gnn-benchmark-") as directory:
        for case in cases:
            output = Path(directory) / f"{case.name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--threads",
                str(args.threads),
                "--min-run-time",
                str(args.min_run_time),
                "--worker-case",
                case.name,
                "--output",
                str(output),
            ]
            if args.include_cpp:
                command.append("--include-cpp")
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                print(completed.stdout, end="", file=sys.stderr)
                print(completed.stderr, end="", file=sys.stderr)
                raise SystemExit(completed.returncode)
            results.extend(json.loads(output.read_text())["results"])
    return {"schema_version": 1, "environment": environment(args), "results": results}


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    document = run_worker(args) if args.worker_case else run_isolated(args)
    if not args.worker_case:
        print_table(document["results"], args.include_cpp)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        if not args.worker_case:
            print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
