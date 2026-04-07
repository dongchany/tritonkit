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
from tritonkit.bench.baselines import BaselineRegistry
from tritonkit.examples import rmsnorm_fused
from tritonkit.testing import LLM_SHAPES

EPS = 1e-6
RESULTS_DIR = ROOT / "benchmarks" / "results"


def input_generator(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)
    return x, weight


def byte_counter(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    m, n = shape
    return (m * n + n + m * n) * element_size


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"{result.kernel}_{slug}.json"


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
    if rmsnorm_fused is None:
        raise RuntimeError("tritonkit.examples.rmsnorm_fused is unavailable.")

    candidates = {"tritonkit_rmsnorm_fused": lambda x, weight: rmsnorm_fused(x, weight, eps=EPS)}
    candidates.update(BaselineRegistry.get("rmsnorm"))

    result = compare(
        candidates=candidates,
        shapes=LLM_SHAPES,
        dtypes=[torch.float16],
        kernel_name="rmsnorm",
        input_generator=input_generator,
        byte_counter=byte_counter,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _result_path(result)
    result.export_json(str(output_path))
    return result, output_path


def main() -> None:
    result, output_path = run_suite()
    result.print_table()
    print(f"Exported JSON: {output_path}")


if __name__ == "__main__":
    main()
