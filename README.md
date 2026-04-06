# TritonKit

Build test and benchmark Triton kernels

## Install

```bash
pip install -e .
```

## Quick Example

```python
import torch
import tritonkit as tk


def triton_kernel(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def reference(x: torch.Tensor) -> torch.Tensor:
    return x * 2


tk.testing.assert_matches(
    triton_fn=triton_kernel,
    reference_fn=reference,
    shapes=[(128, 128)],
    dtypes=[torch.float16],
)
```
