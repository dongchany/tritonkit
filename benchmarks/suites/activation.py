from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from tritonkit.bench import compare
from tritonkit.examples import swiglu_fused
from tritonkit.testing import LLM_SHAPES

RESULTS_DIR = ROOT / "benchmarks" / "results"


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def input_generator(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = torch.randn(shape, dtype=dtype, device=device)
    up = torch.randn(shape, dtype=dtype, device=device)
    return gate, up


def byte_counter(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = 1
    for dim in shape:
        elements *= dim
    return (elements * 3) * element_size


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"{result.kernel}_{slug}.json"


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
    if swiglu_fused is None:
        raise RuntimeError("tritonkit.examples.swiglu_fused is unavailable.")

    result = compare(
        candidates={
            "tritonkit_swiglu_fused": swiglu_fused,
            "pytorch_swiglu": pytorch_swiglu,
        },
        shapes=LLM_SHAPES,
        dtypes=[torch.float16],
        kernel_name="swiglu",
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

