# TritonKit Cross-Hardware Benchmark Report

> **Date**: 2026-04-08
> **Total cost**: ~$0.30 across 7 vast.ai instances + local RTX 3080 Ti
> **GPUs covered**: 7 (1 local + 6 vast.ai rentals)

---

## Hardware Coverage

| GPU | Source | Cost | Status |
|-----|--------|------|--------|
| RTX 3080 Ti 12GB | Local (Ampere SM 8.6) | $0 | ✅ |
| RTX 4080 16GB | vast.ai | ~$0.02 | ✅ |
| RTX 4090 24GB | vast.ai | ~$0.02 | ✅ |
| L40S 48GB | vast.ai | ~$0.05 | ✅ |
| A100 SXM 40GB | vast.ai | ~$0.06 | ✅ |
| H100 SXM 80GB | vast.ai | ~$0.10 | ✅ |
| H200 141GB | vast.ai | ~$0.13 | ✅ |

**Total: 7 GPU types covered for ~$0.40.** This already exceeds triton-bench.ai's
public coverage (3 GPUs: H100, B200, MI350) on the consumer GPU dimension.

---

## Key Findings

### GEMM 4096×4096 FP16 (TFLOPS)

| GPU | tritonkit_gemm_fp16 | Notes |
|-----|---------------------|-------|
| RTX 4080 | 82.4 | Consumer Ada |
| RTX 4090 | 126.3 | Consumer Ada |
| L40S | 135.5 | Datacenter Ada |
| A100 SXM 40G | 162.4 | Datacenter Ampere |
| H100 SXM 80G | **401.7** | Datacenter Hopper |
| H200 | **419.0** | Latest Hopper |

**Speedup ladder**: 4080 → 4090 (1.5x) → L40S (1.1x) → A100 (1.2x) → H100 (2.5x) → H200 (1.04x)

The biggest jump is **A100 → H100 (2.5x)** — Hopper's tensor cores deliver
dramatic compute density improvement. H200 is essentially the same compute as
H100 but with much more memory.

### RMSNorm 4096×4096 (median latency in us)

| GPU | tritonkit | Bandwidth-relative |
|-----|-----------|-------------------|
| RTX 4080 | 516 us | Worst |
| RTX 3080 Ti | 395 us | |
| RTX 4090 | 357 us | |
| L40S | 474 us | |
| A100 SXM 40G | 231 us | |
| **H100 SXM 80G** | **112 us** | |
| **H200** | **105 us** | Best |

RMSNorm is **memory bandwidth bound**, so this directly tracks HBM
bandwidth. H200's HBM3e gives it a slight edge over H100's HBM3.

### INT8 GEMM 4096×4096 (Ampere INT8 tensor cores)

| GPU | tritonkit_int8_gemm |
|-----|---------------------|
| RTX 3080 Ti | 1277 us |
| RTX 4080 | 1272 us |
| L40S | 959 us |
| A100 SXM 40G | 980 us |
| RTX 4090 | 824 us |
| H100 SXM 80G | **587 us** |
| H200 | **588 us** |

INT8 GEMM scales similarly. Note that H100/H200 with FP8 tensor cores
would be even faster — that's a future kernel to add.

---

## Methodology Notes

- **Each GPU was rented from vast.ai for 2-3 minutes** using the safe runner
  script (`scripts/vastai_bench.sh`) which guarantees instance destruction
  via shell trap on exit (success/failure/interrupt).
- **All benchmarks used the same Triton 3.1 / PyTorch 2.5.1 environment**
  from the `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` Docker image.
- **L2 cache flush** before each iteration via 256 MB buffer zeroing.
- **CUDA Events timing** with median + p20/p80 reported.
- **Warmup**: 25ms, **rep**: 100ms time budgets per measurement.

## Coverage vs Competition

| Metric | triton-bench.ai | TritonKit |
|--------|----------------|-----------|
| Public GPU coverage | 3 (H100, B200, MI350) | **7** |
| Consumer GPU coverage | 0 | **3** (RTX 4080, 4090, 3080 Ti) |
| Cross-framework comparison | None | **5+ libraries** |
| Public methodology | Undocumented | Open-source script |
| Reproducible | Login wall | Public command |
| Cost per GPU type | N/A | **~$0.05** |

---

## Reproduce Locally

```bash
# Clone the repo
git clone https://github.com/dongchany/tritonkit
cd tritonkit

# Install
uv venv && uv pip install -e ".[dev]"

# Run all benchmarks on YOUR GPU
.venv/bin/python benchmarks/run_all.py
```

Results will be exported to `benchmarks/results/<kernel>_<gpu>.json`
in the same format as the published data.
