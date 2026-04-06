"""Cross-hardware backend validation."""

from __future__ import annotations

from collections.abc import Callable

import torch


def run_on_available_backends(
    test_fn: Callable[[], None],
) -> dict[str, bool | str]:
    """Run test_fn on all available GPU backends.

    Detects CUDA (NVIDIA), ROCm (AMD), and XPU (Intel), runs test_fn on each.

    Returns:
        Dict mapping backend name to True (passed) or error string (failed).
    """
    results: dict[str, bool | str] = {}

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            test_fn()
            results["cuda"] = True
        except Exception as e:
            results["cuda"] = str(e)

    # ROCm (AMD) — also uses torch.cuda but with HIP backend
    # Detected via torch.version.hip
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        results.setdefault("rocm", results.get("cuda", "not tested"))

    # XPU (Intel)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            test_fn()
            results["xpu"] = True
        except Exception as e:
            results["xpu"] = str(e)

    if not results:
        results["none"] = "No GPU backend available"

    return results
