import pytest
import torch

import tritonkit as tk


def test_assert_matches_passes_for_identical(device: str) -> None:
    def reference_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    tk.testing.assert_matches(
        triton_fn=reference_fn,
        reference_fn=reference_fn,
        shapes=[(32,)],
        dtypes=[torch.float16],
        device=device,
    )


def test_assert_matches_fails_for_wrong(device: str) -> None:
    def triton_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def reference_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(AssertionError):
        tk.testing.assert_matches(
            triton_fn=triton_fn,
            reference_fn=reference_fn,
            shapes=[(32,)],
            dtypes=[torch.float16],
            device=device,
        )


def test_numerical_stability_catches_nan(device: str) -> None:
    def unstable_fn(x: torch.Tensor) -> torch.Tensor:
        zero = x - x
        return zero / zero

    with pytest.raises(AssertionError, match="NaN detected"):
        tk.testing.check_numerical_stability(
            fn=unstable_fn,
            shapes=[(32,)],
            dtypes=[torch.float16],
            device=device,
        )


def test_shape_presets_nonempty() -> None:
    assert isinstance(tk.testing.STANDARD_SHAPES, list)
    assert isinstance(tk.testing.EDGE_SHAPES, list)
    assert isinstance(tk.testing.LLM_SHAPES, list)
    assert tk.testing.STANDARD_SHAPES
    assert tk.testing.EDGE_SHAPES
    assert tk.testing.LLM_SHAPES
