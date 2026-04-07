from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from tritonkit.bench import compare
from tritonkit.examples import int8_gemm

RESULTS_DIR = ROOT / "benchmarks" / "results"
GEMM_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def _symmetric_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax()
    qmax = 127.0
    scale = torch.where(scale > 0, scale / qmax, torch.ones_like(scale))
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float16)


def tritonkit_int8_gemm(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    a_fp16: torch.Tensor,
    b_fp16: torch.Tensor,
) -> torch.Tensor:
    del a_fp16, b_fp16
    return int8_gemm(a_int8, b_int8, scale_a, scale_b)


def torch_mm_fp16(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    a_fp16: torch.Tensor,
    b_fp16: torch.Tensor,
) -> torch.Tensor:
    del a_int8, b_int8, scale_a, scale_b
    return torch.mm(a_fp16, b_fp16)


def input_generator(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del dtype
    m, n, k = shape
    a_fp16 = torch.randn((m, k), dtype=torch.float16, device=device)
    b_fp16 = torch.randn((k, n), dtype=torch.float16, device=device)
    a_int8, scale_a = _symmetric_quantize(a_fp16)
    b_int8, scale_b = _symmetric_quantize(b_fp16)
    return a_int8, b_int8, scale_a, scale_b, a_fp16, b_fp16


def flop_counter(shape: tuple[int, int, int], dtype: torch.dtype) -> int:
    del dtype
    m, n, k = shape
    return 2 * m * n * k


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"quantize_{slug}.json"


def _speedup_rows(result) -> list[str]:
    grouped: dict[tuple[int, ...], dict[str, float]] = {}
    for entry in result.results:
        grouped.setdefault(entry.shape, {})[entry.name] = entry.median_us

    rows: list[str] = []
    for shape in sorted(grouped):
        timings = grouped[shape]
        if "tritonkit_int8_gemm" in timings and "torch_mm_fp16" in timings:
            speedup = timings["torch_mm_fp16"] / timings["tritonkit_int8_gemm"]
            rows.append(f"{shape}: {speedup:.2f}x vs torch_mm_fp16")
    return rows


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
    if int8_gemm is None:
        raise RuntimeError("tritonkit.examples.int8_gemm is unavailable.")

    result = compare(
        candidates={
            "tritonkit_int8_gemm": tritonkit_int8_gemm,
            "torch_mm_fp16": torch_mm_fp16,
        },
        shapes=GEMM_SHAPES,
        dtypes=[torch.float16],
        kernel_name="quantize",
        input_generator=input_generator,
        flop_counter=flop_counter,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _result_path(result)
    result.export_json(str(output_path))
    return result, output_path


def main() -> None:
    result, output_path = run_suite()
    result.print_table(sort_by="tflops")
    for row in _speedup_rows(result):
        print(f"Speedup {row}")
    print(f"Exported JSON: {output_path}")


if __name__ == "__main__":
    main()
