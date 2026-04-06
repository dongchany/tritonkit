"""Helpers for making Python wrappers friendly to torch.compile."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import torch

F = TypeVar("F", bound=Callable[..., Any])


def _allow_in_graph() -> Callable[[F], F]:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "allow_in_graph"):
        return cast(Callable[[F], F], torch.compiler.allow_in_graph)
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "allow_in_graph"):
        return cast(Callable[[F], F], torch._dynamo.allow_in_graph)
    return lambda fn: fn


def make_compilable(fn: F) -> F:
    """Wrap fn as a leaf callable that torch.compile can keep in-graph."""

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any):
        return fn(*args, **kwargs)

    compilable = _allow_in_graph()(wrapped)
    setattr(compilable, "__tritonkit_original__", fn)
    return cast(F, compilable)
