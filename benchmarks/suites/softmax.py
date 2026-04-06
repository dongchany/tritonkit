from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import triton
import triton.language as tl

from tritonkit.bench import compare

RESULTS_DIR = ROOT / "benchmarks" / "results"
ATTENTION_SCORE_SHAPES = [
    (1, 32, 128, 128),
    (1, 32, 512, 512),
    (4, 32, 128, 128),
    (4, 32, 512, 512),
    (8, 32, 128, 128),
    (1, 32, 2048, 2048),
]


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = input_ptr + row_idx * input_row_stride
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row = row - tl.max(row, axis=0)
    numerators = tl.exp(row)
    denominator = tl.sum(numerators, axis=0)
    softmax_row = numerators / denominator

    output_row_start = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start + col_offsets, softmax_row, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    rows, cols = x_2d.shape
    y = torch.empty_like(x_2d)
    block_size = triton.next_power_of_2(cols)
    num_warps = min(max(block_size // 256, 1), 8)

    _softmax_kernel[(rows,)](
        x_2d,
        y,
        x_2d.stride(0),
        y.stride(0),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y.reshape(original_shape)


def input_generator(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor]:
    return (torch.randn(shape, dtype=dtype, device=device),)


def byte_counter(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = 1
    for dim in shape:
        elements *= dim
    return (elements * 2) * element_size


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"{result.kernel}_{slug}.json"


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")

    result = compare(
        candidates={
            "torch_softmax": lambda x: torch.softmax(x, dim=-1),
            "triton_softmax": triton_softmax,
        },
        shapes=ATTENTION_SCORE_SHAPES,
        dtypes=[torch.float16],
        kernel_name="softmax",
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

