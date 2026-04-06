import pytest
import torch


@pytest.fixture
def requires_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")


@pytest.fixture
def device(requires_cuda: None) -> str:
    return "cuda"
