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
from tritonkit.examples import gemm_fp16

RESULTS_DIR = ROOT / "benchmarks" / "results"
SQUARE_SIZES = [256, 512, 1024, 2048, 4096, 8192]
GEMM_SHAPES = [(size, size, size) for size in SQUARE_SIZES] + [
    (1, 11008, 4096),
    (32, 11008, 4096),
    (128, 11008, 4096),
    (4096, 11008, 4096),
    (4096, 4096, 11008),
    (4096, 14336, 4096),
]


def input_generator(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = shape
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    return a, b


def flop_counter(shape: tuple[int, int, int], dtype: torch.dtype) -> int:
    del dtype
    m, n, k = shape
    return 2 * m * n * k


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"{result.kernel}_{slug}.json"


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
    if gemm_fp16 is None:
        raise RuntimeError("tritonkit.examples.gemm_fp16 is unavailable.")

    result = compare(
        candidates={
            "tritonkit_gemm_fp16": gemm_fp16,
            "torch_mm": torch.mm,
        },
        shapes=GEMM_SHAPES,
        dtypes=[torch.float16],
        kernel_name="gemm",
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
    print(f"Exported JSON: {output_path}")


if __name__ == "__main__":
    main()

