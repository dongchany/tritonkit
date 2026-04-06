import pytest
import torch
import torch.nn.functional as F

from tritonkit.examples import flash_attention


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention(device: str, causal: bool) -> None:
    if flash_attention is None:
        pytest.skip("flash_attention is unavailable")

    q = torch.randn((2, 8, 128, 64), device=device, dtype=torch.float16)
    k = torch.randn((2, 8, 128, 64), device=device, dtype=torch.float16)
    v = torch.randn((2, 8, 128, 64), device=device, dtype=torch.float16)

    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    actual = flash_attention(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
