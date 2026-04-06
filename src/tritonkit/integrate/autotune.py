"""Utilities for exporting Triton autotune selections."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any


def _unwrap_autotuner(kernel: Any) -> Any:
    candidate = kernel
    seen: set[int] = set()

    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        if any(hasattr(candidate, attr) for attr in ("best_config", "cache", "configs")):
            return candidate
        candidate = getattr(candidate, "fn", None)

    return kernel


def _serialize_config(config: Any) -> dict[str, Any]:
    return {
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", None),
        "maxnreg": getattr(config, "maxnreg", None),
        "ir_override": getattr(config, "ir_override", None),
    }


def export_autotune_config(kernel: Any, path: str | PathLike[str]) -> Path:
    """Export the selected Triton autotune config to a JSON file."""
    autotuner = _unwrap_autotuner(kernel)
    best_config = getattr(autotuner, "best_config", None)

    if best_config is None:
        cache = getattr(autotuner, "cache", {})
        if cache:
            best_config = next(iter(cache.values()))

    if best_config is None:
        raise ValueError("No autotune result available. Run the kernel before exporting.")

    payload = {
        "kernel": getattr(
            getattr(autotuner, "base_fn", None),
            "__name__",
            getattr(kernel, "__name__", kernel.__class__.__name__),
        ),
        "best_config": _serialize_config(best_config),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
