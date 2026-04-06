"""Built-in baseline implementations for benchmark comparison."""

from __future__ import annotations

from collections.abc import Callable

import torch


class BaselineRegistry:
    """Registry of known baseline implementations."""

    _baselines: dict[str, dict[str, Callable]] = {}

    @classmethod
    def register(cls, operation: str, name: str, fn: Callable) -> None:
        cls._baselines.setdefault(operation, {})[name] = fn

    @classmethod
    def get(cls, operation: str) -> dict[str, Callable]:
        return dict(cls._baselines.get(operation, {}))

    @classmethod
    def list_operations(cls) -> list[str]:
        return list(cls._baselines.keys())


def _register_builtins() -> None:
    """Register built-in baselines. Called on module import."""

    # PyTorch (cuBLAS) GEMM
    BaselineRegistry.register("gemm", "pytorch_cublas", lambda a, b: torch.mm(a, b))

    # PyTorch RMSNorm
    def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + eps)) * weight

    BaselineRegistry.register("rmsnorm", "pytorch", pytorch_rmsnorm)

    # PyTorch SDPA
    BaselineRegistry.register(
        "attention",
        "pytorch_sdpa",
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
    )

    # PyTorch SwiGLU
    def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up

    BaselineRegistry.register("swiglu", "pytorch", pytorch_swiglu)

    # Optional: Liger-Kernel
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        BaselineRegistry.register("rmsnorm", "liger", LigerRMSNormFunction.apply)
    except ImportError:
        pass

    # Optional: FlashAttention
    try:
        from flash_attn import flash_attn_func

        BaselineRegistry.register("attention", "flash_attn", flash_attn_func)
    except ImportError:
        pass


_register_builtins()
