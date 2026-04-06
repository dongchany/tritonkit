"""Correctness testing: compare Triton kernels against PyTorch references."""

from __future__ import annotations

from collections.abc import Callable

import torch

from tritonkit.testing.shapes import ShapePreset


def assert_matches(
    triton_fn: Callable,
    reference_fn: Callable,
    shapes: list[tuple[int, ...]] | ShapePreset,
    dtypes: list[torch.dtype] | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    input_generator: Callable | None = None,
    device: str = "cuda",
) -> None:
    """Test that triton_fn produces the same output as reference_fn.

    Iterates over all (shape, dtype) combinations, generates inputs, and
    compares outputs with torch.testing.assert_close.

    Raises:
        AssertionError: With detailed mismatch report.
    """
    if dtypes is None:
        dtypes = [torch.float16, torch.bfloat16]

    for shape in shapes:
        for dtype in dtypes:
            if input_generator is not None:
                inputs = input_generator(shape, dtype, device)
            else:
                inputs = (torch.randn(shape, dtype=dtype, device=device),)

            ref_out = reference_fn(*inputs)
            tri_out = triton_fn(*inputs)

            try:
                torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=rtol)
            except AssertionError as e:
                diff = (tri_out - ref_out).abs()
                rel_diff = diff / (ref_out.abs() + 1e-8)
                msg = (
                    f"\nShape: {shape}, dtype: {dtype}\n"
                    f"Max absolute error: {diff.max().item():.6f}\n"
                    f"Max relative error: {rel_diff.max().item():.6f}\n"
                    f"Mismatched elements: {(diff > atol).sum().item()} / {diff.numel()} "
                    f"({(diff > atol).sum().item() / diff.numel() * 100:.1f}%)\n"
                    f"Worst index: {diff.argmax().item()}\n"
                    f"Original error: {e}"
                )
                raise AssertionError(msg) from None


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Thin wrapper around torch.testing.assert_close with better error messages."""
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        diff = (actual - expected).abs()
        rel_diff = diff / (expected.abs() + 1e-8)
        msg = (
            f"\nMax absolute error: {diff.max().item():.6f}\n"
            f"Max relative error: {rel_diff.max().item():.6f}\n"
            f"Mismatched: {(diff > atol).sum().item()} / {diff.numel()}\n"
            f"Original: {e}"
        )
        raise AssertionError(msg) from None
