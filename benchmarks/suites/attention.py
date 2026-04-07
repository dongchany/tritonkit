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
from tritonkit.examples import flash_attention

RESULTS_DIR = ROOT / "benchmarks" / "results"
ATTENTION_SHAPES = [
    (2, 8, 512, 64),
    (2, 8, 1024, 64),
    (2, 8, 2048, 64),
    (4, 16, 1024, 128),
]
ATTENTION_CASES = [shape + (int(causal),) for shape in ATTENTION_SHAPES for causal in (False, True)]


def tritonkit_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_flag: torch.Tensor,
) -> torch.Tensor:
    return flash_attention(q, k, v, causal=bool(causal_flag.item()))


def pytorch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_flag: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal_flag.item()))


def input_generator(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim, causal = shape
    q = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
    causal_flag = torch.tensor(bool(causal), device=device, dtype=torch.bool)
    return q, k, v, causal_flag


def flop_counter(shape: tuple[int, int, int, int, int], dtype: torch.dtype) -> int:
    del dtype
    batch, heads, seq_len, head_dim, causal = shape
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    return flops // 2 if causal else flops


def _result_path(result) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", result.hardware.gpu_name.lower()).strip("_")
    return RESULTS_DIR / f"attention_{slug}.json"


def run_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
    if flash_attention is None:
        raise RuntimeError("tritonkit.examples.flash_attention is unavailable.")

    result = compare(
        candidates={
            "tritonkit_flash_attention": tritonkit_flash_attention,
            "torch_sdpa": pytorch_sdpa,
        },
        shapes=ATTENTION_CASES,
        dtypes=[torch.float16],
        kernel_name="attention",
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
