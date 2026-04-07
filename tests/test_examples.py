import pytest
import torch
import torch.nn.functional as F

from tritonkit.examples import gemm_fp16, int8_gemm, rmsnorm_fused, swiglu_fused, w4a16_gemm


def _quantize_pack_int4(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k, n = weight.shape
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")

    qmax = 7.0
    num_groups = k // group_size

    weight_groups = weight.to(torch.float32).reshape(num_groups, group_size, n)
    max_abs = weight_groups.abs().amax(dim=1)
    scales = torch.where(
        max_abs > 0,
        max_abs / qmax,
        torch.ones_like(max_abs),
    ).to(torch.float16)

    quantized = torch.clamp(
        torch.round(weight_groups / scales[:, None, :].to(torch.float32)),
        -8,
        7,
    ).to(torch.int32)
    dequantized = (quantized.to(torch.float16) * scales[:, None, :]).reshape(k, n)

    packed = torch.where(quantized < 0, quantized + 16, quantized).reshape(k, n)
    packed = packed.reshape(k // 8, 8, n)

    qweight = torch.zeros((k // 8, n), device=weight.device, dtype=torch.int32)
    for idx in range(8):
        qweight |= packed[:, idx, :] << (idx * 4)

    return qweight, scales.contiguous(), dequantized.contiguous()


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


def test_int8_gemm(device: str) -> None:
    if int8_gemm is None:
        pytest.skip("int8_gemm is unavailable")

    a = torch.randn((256, 192), device=device, dtype=torch.float16)
    b = torch.randn((192, 320), device=device, dtype=torch.float16)

    qmax = 127.0
    scale_a = torch.where(a.abs().amax() > 0, a.abs().amax() / qmax, torch.ones((), device=device, dtype=a.dtype))
    scale_b = torch.where(b.abs().amax() > 0, b.abs().amax() / qmax, torch.ones((), device=device, dtype=b.dtype))
    a_int8 = torch.clamp(torch.round(a / scale_a), -qmax, qmax).to(torch.int8)
    b_int8 = torch.clamp(torch.round(b / scale_b), -qmax, qmax).to(torch.int8)

    expected = torch.mm(a_int8.to(torch.float16) * scale_a, b_int8.to(torch.float16) * scale_b)
    actual = int8_gemm(a_int8, b_int8, scale_a.to(torch.float16), scale_b.to(torch.float16))

    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.1)


def test_w4a16_gemm(device: str) -> None:
    if w4a16_gemm is None:
        pytest.skip("w4a16_gemm is unavailable")

    torch.manual_seed(0)

    m, k, n = 96, 256, 160
    group_size = 128

    a = torch.randn((m, k), device=device, dtype=torch.float16)
    weight = torch.randn((k, n), device=device, dtype=torch.float16)

    qweight, scales, dequantized = _quantize_pack_int4(weight, group_size=group_size)

    expected = torch.mm(a, dequantized)
    actual = w4a16_gemm(a, qweight, scales, group_size=group_size)

    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.1)
