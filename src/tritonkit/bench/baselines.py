"""Built-in baseline implementations for benchmark comparison."""

from __future__ import annotations

import importlib
import sys
import types
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


def _ensure_dtensor_stub() -> None:
    """Install a minimal DTensor stub for optional integrations on older Torch builds."""
    distributed = getattr(torch, "distributed", None)
    if distributed is None or hasattr(distributed, "tensor"):
        return

    module_name = "torch.distributed.tensor"
    tensor_module = types.ModuleType(module_name)

    class DTensor:  # pragma: no cover - compatibility shim
        @classmethod
        def from_local(cls, tensor, *args, **kwargs):
            del args, kwargs
            return tensor

    class DeviceMesh:  # pragma: no cover - compatibility shim
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.ndim = 1

    class Replicate:  # pragma: no cover - compatibility shim
        pass

    class Shard:  # pragma: no cover - compatibility shim
        def __init__(self, dim: int = 0):
            self.dim = dim

    def distribute_tensor(tensor, *args, **kwargs):  # pragma: no cover - compatibility shim
        del args, kwargs
        return tensor

    tensor_module.DTensor = DTensor
    tensor_module.DeviceMesh = DeviceMesh
    tensor_module.Replicate = Replicate
    tensor_module.Shard = Shard
    tensor_module.distribute_tensor = distribute_tensor
    tensor_module.__all__ = ["DTensor", "DeviceMesh", "Replicate", "Shard", "distribute_tensor"]
    sys.modules.setdefault(module_name, tensor_module)
    setattr(distributed, "tensor", tensor_module)


def _register_optional(
    operation: str,
    name: str,
    factory: Callable[[], Callable],
) -> None:
    try:
        BaselineRegistry.register(operation, name, factory())
    except Exception:
        pass


def _resolve_attr(module_names: list[str], attr_names: list[str]) -> Callable | None:
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name in attr_names:
            value = getattr(module, attr_name, None)
            if callable(value):
                return value
    return None


def _causal_flag(value: torch.Tensor | bool | None) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _gemlite_cache_key(a: torch.Tensor, b: torch.Tensor) -> tuple[int, tuple[int, ...], torch.dtype, str]:
    return (b.data_ptr(), tuple(b.shape), a.dtype, str(b.device))


def _make_flag_gems_rmsnorm() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["rms_norm", "rmsnorm"],
    )
    if fn is None:
        raise ImportError("flag_gems rmsnorm is unavailable")

    def flag_gems_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return fn(x, (x.shape[-1],), weight, eps=eps)

    return flag_gems_rmsnorm


def _make_flag_gems_layernorm() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["layer_norm", "layernorm"],
    )
    if fn is None:
        raise ImportError("flag_gems layernorm is unavailable")

    def flag_gems_layernorm(
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return fn(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)

    return flag_gems_layernorm


def _make_flag_gems_swiglu() -> Callable:
    silu_and_mul = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["silu_and_mul"],
    )
    if silu_and_mul is not None:
        return silu_and_mul

    swiglu = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["swiglu"],
    )
    if swiglu is None:
        raise ImportError("flag_gems swiglu is unavailable")

    def flag_gems_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return swiglu(torch.cat((gate, up), dim=-1))

    return flag_gems_swiglu


def _make_flag_gems_gemm() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["mm"],
    )
    if fn is None:
        raise ImportError("flag_gems mm is unavailable")
    return fn


def _make_flag_gems_softmax() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["softmax"],
    )
    if fn is None:
        raise ImportError("flag_gems softmax is unavailable")

    def flag_gems_softmax(x: torch.Tensor) -> torch.Tensor:
        return fn(x, dim=-1)

    return flag_gems_softmax


def _make_flag_gems_attention() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["scaled_dot_product_attention"],
    )
    if fn is None:
        raise ImportError("flag_gems attention is unavailable")

    def flag_gems_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_flag: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        return fn(q, k, v, is_causal=_causal_flag(causal_flag))

    return flag_gems_attention


def _make_flag_gems_rope() -> Callable:
    fn = _resolve_attr(
        ["flag_gems.ops", "flag_gems.fused", "flag_gems"],
        ["apply_rotary_pos_emb", "rope"],
    )
    if fn is None:
        raise ImportError("flag_gems rope is unavailable")
    return fn


def _make_xformers_attention() -> Callable:
    xops = importlib.import_module("xformers.ops")
    attention_fn = getattr(xops, "memory_efficient_attention", None)
    if attention_fn is None:
        raise ImportError("xformers attention is unavailable")
    lower_triangular_mask = getattr(xops, "LowerTriangularMask", None)

    def xformers_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_flag: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        attn_bias = lower_triangular_mask() if _causal_flag(causal_flag) and lower_triangular_mask else None
        return attention_fn(q, k, v, attn_bias=attn_bias)

    return xformers_attention


def _make_flash_attn_attention() -> Callable:
    flash_attn_func = getattr(importlib.import_module("flash_attn"), "flash_attn_func", None)
    if flash_attn_func is None:
        raise ImportError("flash_attn_func is unavailable")

    def flash_attn_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_flag: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        q_flash, k_flash, v_flash = [tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)]
        out = flash_attn_func(q_flash, k_flash, v_flash, causal=_causal_flag(causal_flag))
        return out.transpose(1, 2).contiguous()

    return flash_attn_attention


def _make_gemlite_a16w8() -> Callable:
    helper = importlib.import_module("gemlite.helper")
    processor_cls = getattr(helper, "A16W8_INT8", None) or getattr(helper, "A16W8", None)
    if processor_cls is None:
        raise ImportError("gemlite A16W8 is unavailable")

    cache: dict[tuple[int, tuple[int, ...], torch.dtype, str], Callable] = {}

    def gemlite_a16w8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        key = _gemlite_cache_key(a, b)
        module = cache.get(key)
        if module is None:
            processor = processor_cls(device=str(b.device), dtype=a.dtype)
            module = processor.from_weights(b.t().contiguous())
            cache[key] = module
        return module(a)

    return gemlite_a16w8


def _make_gemlite_a16w4() -> Callable:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (8, 9):
            raise RuntimeError("gemlite A16W4 requires sm89+")

    helper = importlib.import_module("gemlite.helper")
    processor_cls = getattr(helper, "A16W4_MXFP", None)
    if processor_cls is None:
        raise ImportError("gemlite A16W4 is unavailable")

    cache: dict[tuple[int, tuple[int, ...], torch.dtype, str], Callable] = {}

    def gemlite_a16w4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        key = _gemlite_cache_key(a, b)
        module = cache.get(key)
        if module is None:
            linear = torch.nn.Linear(
                in_features=b.shape[0],
                out_features=b.shape[1],
                bias=False,
                device=b.device,
                dtype=b.dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(b.t().contiguous())
            processor = processor_cls(device=str(b.device), dtype=a.dtype)
            module = processor.from_linear(linear, del_orig=False)
            cache[key] = module
        return module(a)

    return gemlite_a16w4


def _make_bitsandbytes_matmul_4bit() -> Callable:
    bnb = importlib.import_module("bitsandbytes")
    bnbf = importlib.import_module("bitsandbytes.functional")
    quantize_4bit = getattr(bnbf, "quantize_4bit", None)
    matmul_4bit = getattr(bnb, "matmul_4bit", None)
    if quantize_4bit is None or matmul_4bit is None:
        raise ImportError("bitsandbytes 4-bit matmul is unavailable")

    cache: dict[tuple[int, tuple[int, ...], torch.dtype, str], tuple[torch.Tensor, object]] = {}

    def bitsandbytes_matmul_4bit(
        a_int8: torch.Tensor,
        b_int8: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        a_fp16: torch.Tensor,
        b_fp16: torch.Tensor,
    ) -> torch.Tensor:
        del a_int8, b_int8, scale_a, scale_b
        key = _gemlite_cache_key(a_fp16, b_fp16)
        quantized = cache.get(key)
        if quantized is None:
            quantized = quantize_4bit(b_fp16.contiguous())
            cache[key] = quantized
        qweight, quant_state = quantized
        return matmul_4bit(a_fp16, qweight, quant_state)

    return bitsandbytes_matmul_4bit


def _register_builtins() -> None:
    """Register built-in baselines. Called on module import."""

    # PyTorch (cuBLAS) GEMM
    BaselineRegistry.register("gemm", "pytorch_cublas", lambda a, b: torch.mm(a, b))

    # PyTorch RMSNorm
    def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + eps)) * weight

    BaselineRegistry.register("rmsnorm", "pytorch", pytorch_rmsnorm)
    if hasattr(torch.nn.functional, "rms_norm"):
        BaselineRegistry.register(
            "rmsnorm",
            "pytorch_functional",
            lambda x, weight: torch.nn.functional.rms_norm(
                x,
                (x.shape[-1],),
                weight=weight,
                eps=1e-6,
            ),
        )

    BaselineRegistry.register(
        "layernorm",
        "pytorch",
        lambda x, weight=None, bias=None, eps=1e-5: torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        ),
    )

    # PyTorch SDPA
    BaselineRegistry.register(
        "attention",
        "pytorch_sdpa",
        lambda q, k, v, causal_flag=None: torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=_causal_flag(causal_flag),
        ),
    )

    # PyTorch SwiGLU
    def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up

    BaselineRegistry.register("swiglu", "pytorch", pytorch_swiglu)
    BaselineRegistry.register("softmax", "pytorch", lambda x: torch.softmax(x, dim=-1))
    BaselineRegistry.register(
        "quantize_gemm",
        "torch_mm_fp16",
        lambda a_int8, b_int8, scale_a, scale_b, a_fp16, b_fp16: torch.mm(a_fp16, b_fp16),
    )

    # Optional: Liger-Kernel
    try:
        _ensure_dtensor_stub()
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        def liger_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            eps: float = 1e-6,
        ) -> torch.Tensor:
            try:
                return LigerRMSNormFunction.apply(x, weight, eps)
            except AttributeError as exc:
                raise RuntimeError(
                    "Liger RMSNorm is incompatible with this torch.distributed build"
                ) from exc

        BaselineRegistry.register("rmsnorm", "liger", liger_rmsnorm)
    except Exception:
        pass

    _register_optional("rmsnorm", "flag_gems", _make_flag_gems_rmsnorm)
    _register_optional("layernorm", "flag_gems", _make_flag_gems_layernorm)
    _register_optional("swiglu", "flag_gems", _make_flag_gems_swiglu)
    _register_optional("gemm", "flag_gems", _make_flag_gems_gemm)
    _register_optional("softmax", "flag_gems", _make_flag_gems_softmax)
    _register_optional("attention", "flag_gems", _make_flag_gems_attention)
    _register_optional("rope", "flag_gems", _make_flag_gems_rope)

    _register_optional("gemm", "gemlite_a16w8", _make_gemlite_a16w8)
    _register_optional("gemm", "gemlite_a16w4", _make_gemlite_a16w4)
    _register_optional("attention", "xformers", _make_xformers_attention)
    _register_optional("attention", "flash_attn", _make_flash_attn_attention)
    _register_optional("quantize_gemm", "bitsandbytes_matmul_4bit", _make_bitsandbytes_matmul_4bit)


_register_builtins()
