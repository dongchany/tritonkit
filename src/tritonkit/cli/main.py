"""Command-line entrypoints for TritonKit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import torch
import triton

def _hardware_fingerprint() -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }

    if hasattr(torch, "xpu"):
        try:
            fingerprint["xpu_available"] = torch.xpu.is_available()
        except Exception:
            fingerprint["xpu_available"] = False

    if hasattr(torch.version, "hip"):
        fingerprint["rocm_available"] = torch.version.hip is not None

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        fingerprint.update(
            {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_gb": round(props.total_memory / 1e9, 2),
                "sm_count": props.multi_processor_count,
            }
        )

        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(device_index),
                    "--query-gpu=driver_version,clocks.sm,clocks.mem,pcie.link.gen.current",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if smi.returncode == 0:
                first_line = smi.stdout.strip().splitlines()[0]
                parts = first_line.split(", ")
                if len(parts) >= 4:
                    fingerprint["driver_version"] = parts[0].strip()
                    fingerprint["sm_clock_mhz"] = int(parts[1].strip())
                    fingerprint["mem_clock_mhz"] = int(parts[2].strip())
                    fingerprint["pcie_gen"] = parts[3].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return fingerprint


def _print_fingerprint(as_json: bool) -> None:
    fingerprint = _hardware_fingerprint()
    if as_json:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))
        return

    for key, value in fingerprint.items():
        print(f"{key}: {value}")


def _cmd_bench(args: argparse.Namespace) -> int:
    _print_fingerprint(args.json)
    if not args.json:
        print("Benchmark execution is library-driven; use tritonkit.bench APIs for specific kernels.")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest", *args.pytest_args]
    return subprocess.run(cmd, check=False).returncode


def _cmd_info(args: argparse.Namespace) -> int:
    _print_fingerprint(args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tritonkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("bench", help="show benchmark environment details")
    bench_parser.add_argument("--json", action="store_true", help="print hardware info as JSON")
    bench_parser.set_defaults(handler=_cmd_bench)

    test_parser = subparsers.add_parser("test", help="run the test suite")
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to pytest",
    )
    test_parser.set_defaults(handler=_cmd_test)

    info_parser = subparsers.add_parser("info", help="print the hardware fingerprint")
    info_parser.add_argument("--json", action="store_true", help="print hardware info as JSON")
    info_parser.set_defaults(handler=_cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
