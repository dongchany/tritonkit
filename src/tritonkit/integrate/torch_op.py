"""Helpers for registering Python-backed torch.library operators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_LIBRARIES: dict[str, torch.library.Library] = {}
_SCHEMAS: dict[tuple[str, str], str] = {}


def register_torch_op(
    name: str,
    fn: Callable[..., Any],
    schema: str,
    namespace: str = "tritonkit",
):
    """Register a Python op under torch.ops.<namespace>.<name>."""
    if not name:
        raise ValueError("name must be non-empty")

    op_schema = schema.strip()
    if not op_schema.startswith(f"{name}("):
        op_schema = f"{name}{op_schema}"

    library = _LIBRARIES.get(namespace)
    if library is None:
        library = torch.library.Library(namespace, "FRAGMENT")
        _LIBRARIES[namespace] = library

    op_key = (namespace, name)
    if op_key not in _SCHEMAS:
        try:
            library.define(op_schema)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "already" not in message and "multiple times" not in message:
                raise
        _SCHEMAS[op_key] = op_schema

    library.impl(
        name,
        fn,
        dispatch_key="CompositeExplicitAutograd",
        allow_override=True,
    )
    return getattr(getattr(torch.ops, namespace), name)
