import pytest
import torch
import torch.nn.functional as F

from tritonkit.examples import gemm_fp16, rmsnorm_fused, swiglu_fused


def test_rmsnorm_fused(device: str) -> None:
    if rmsnorm_fused is None:
        pytest.skip("rmsnorm_fused is unavailable")

    x = torch.randn((128, 4096), device=device, dtype=torch.float16)
    weight = torch.randn((4096,), device=device, dtype=torch.float16)

    x_float = x.to(torch.float32)
    weight_float = weight.to(torch.float32)
    rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = (x_float * rms * weight_float).to(torch.float16)

    actual = rmsnorm_fused(x, weight)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_swiglu_fused(device: str) -> None:
    if swiglu_fused is None:
        pytest.skip("swiglu_fused is unavailable")

    gate = torch.randn((128, 4096), device=device, dtype=torch.float16)
    up = torch.randn((128, 4096), device=device, dtype=torch.float16)

    expected = F.silu(gate) * up
    actual = swiglu_fused(gate, up)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_gemm_fp16(device: str) -> None:
    if gemm_fp16 is None:
        pytest.skip("gemm_fp16 is unavailable")

    a = torch.randn((512, 512), device=device, dtype=torch.float16)
    b = torch.randn((512, 512), device=device, dtype=torch.float16)

    expected = torch.mm(a, b)
    actual = gemm_fp16(a, b)

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
