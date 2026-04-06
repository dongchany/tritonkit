"""Unified correctness testing for Triton kernels."""

from tritonkit.testing.shapes import (
    STANDARD_SHAPES,
    EDGE_SHAPES,
    LLM_SHAPES,
    ShapePreset,
)
from tritonkit.testing.correctness import assert_matches, assert_close
from tritonkit.testing.numerical import check_numerical_stability
from tritonkit.testing.hardware import run_on_available_backends

__all__ = [
    "assert_matches",
    "assert_close",
    "check_numerical_stability",
    "run_on_available_backends",
    "STANDARD_SHAPES",
    "EDGE_SHAPES",
    "LLM_SHAPES",
    "ShapePreset",
]
