"""Numerical stability checks for Triton kernels."""

from __future__ import annotations

from collections.abc import Callable

import torch

from tritonkit.testing.shapes import ShapePreset, STANDARD_SHAPES


def check_numerical_stability(
    fn: Callable,
    shapes: list[tuple[int, ...]] | ShapePreset | None = None,
    dtypes: list[torch.dtype] | None = None,
    input_range: tuple[float, float] = (-100.0, 100.0),
    check_nan: bool = True,
    check_inf: bool = True,
    device: str = "cuda",
) -> None:
    """Check that fn does not produce NaN or Inf for inputs in input_range.

    Generates random inputs uniformly in [input_range[0], input_range[1]],
    calls fn, and asserts no NaN/Inf in output.

    Raises:
        AssertionError: If NaN or Inf found.
    """
    if shapes is None:
        shapes = STANDARD_SHAPES
    if dtypes is None:
        dtypes = [torch.float16, torch.bfloat16]

    lo, hi = input_range

    for shape in shapes:
        for dtype in dtypes:
            x = torch.empty(shape, dtype=dtype, device=device).uniform_(lo, hi)
            out = fn(x)

            if check_nan and torch.isnan(out).any():
                nan_count = torch.isnan(out).sum().item()
                raise AssertionError(
                    f"NaN detected: {nan_count}/{out.numel()} elements, "
                    f"shape={shape}, dtype={dtype}, input_range={input_range}"
                )

            if check_inf and torch.isinf(out).any():
                inf_count = torch.isinf(out).sum().item()
                raise AssertionError(
                    f"Inf detected: {inf_count}/{out.numel()} elements, "
                    f"shape={shape}, dtype={dtype}, input_range={input_range}"
                )
