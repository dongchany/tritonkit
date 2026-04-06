# TritonKit — Product Manual

> **Status**: Draft v0.3 (Decisions finalized, competitive analysis added)
> **Author**: Dongcheng Ye
> **Date**: 2026-04-07

---

## 1. Product Vision

### One-Line Pitch

A Triton kernel development toolkit and cross-hardware benchmark platform —
making it easy to write, test, benchmark, and compare GPU kernels across
implementations, frameworks, and hardware.

### What Changed (v0.1 → v0.2)

v0.1 proposed "yet another Triton kernel library." Market analysis revealed:
- 50+ Triton kernel projects already exist (FlagGems, Liger-Kernel, gemlite,
  FLASHNN, conch, attorch, etc.)
- Standard GEMM in Triton achieves 85-100% of cuBLAS — not a compelling
  reason to switch
- End-to-end LLM inference gains from custom kernels are 10-30%, not multi-fold
- The real pain is not "we need more kernels" but "writing, testing, and
  comparing kernels is fragmented and painful"

v0.2 pivots to: **solve the kernel development experience problem, and
provide the ecosystem's missing benchmark visibility layer.**

### The Problem

Writing a Triton kernel today requires:

1. Start from scratch (or copy-paste from tutorials)
2. Write your own correctness tests (vs PyTorch reference)
3. Write your own benchmarks (vs cuBLAS / other implementations)
4. Do your own autotuning (redo for each hardware)
5. Write boilerplate for PyTorch integration (custom op registration)
6. Debug with only `tl.device_print` (no real debugger)
7. Discover later that someone already wrote a similar kernel
8. No way to know: "Is my kernel actually good compared to alternatives?"

There is no unified infrastructure for any of this.

Meanwhile, the ecosystem lacks visibility:
- "Which flash attention implementation is fastest on my RTX 3090?"
- "How does FlagGems' RMSNorm compare to Liger-Kernel's on A100?"
- "Is vLLM's Triton attention backend faster than FlashInfer on AMD?"

Nobody can answer these questions today. Benchmarks are scattered across
individual repos, run on different hardware, with different methodologies.

### The Solution

**TritonKit**: Two complementary products:

1. **tritonkit** (Python library): Composable primitives, unified testing,
   benchmarking framework, and integration tools for Triton kernel
   development.

2. **tritonkit.dev** (Website): Public benchmark dashboard showing kernel
   performance across implementations × hardware × frameworks. The
   "Can I Use" for GPU kernels.

---

## 2. Target Users

### Primary: Triton Kernel Developers

**Who**: Engineers writing custom Triton kernels for PyTorch, SGLang, vLLM,
or their own projects. Includes both experienced GPU programmers and ML
engineers learning Triton.

**Their need**: Faster development cycle. Don't reinvent testing, benchmarking,
and integration infrastructure for every kernel.

**How we serve them**:
- Composable tile-level primitives (masked load, online softmax, split-K, etc.)
- One-line correctness testing against PyTorch references
- One-line benchmarking against cuBLAS/FlashAttention/etc.
- One-line PyTorch custom op registration
- Pre-tuned autotuning configs per hardware

**Success metric**: Kernel development time reduced from days to hours.

### Secondary: Framework Developers (SGLang, vLLM, etc.)

**Who**: Teams building LLM serving/training frameworks who need to choose
and maintain kernel backends.

**Their need**: Data-driven kernel selection. "Which attention implementation
should we default to on Ampere?" Currently answered by gut feel or
one-off benchmarks.

**How we serve them**:
- Public benchmark data across all major kernel implementations
- Hardware-specific recommendations backed by reproducible numbers
- Drop-in kernel integration tools

**Success metric**: Framework teams cite our benchmark data in kernel
selection decisions.

### Tertiary: The Broader ML Community

**Who**: ML engineers, researchers, students who want to understand GPU
kernel performance but don't write kernels themselves.

**Their need**: "Should I use A100 or H100 for my workload?" "Is FP8
actually faster on my hardware?" "Which framework is fastest?"

**How we serve them**: The benchmark website provides clear, visual answers.

**Success metric**: The website becomes a reference people link to in
discussions.

---

## 3. Product Scope

### Product 1: `tritonkit` Python Library

#### 3.1 Composable Primitives (`tritonkit.primitives`)

Reusable building blocks for constructing kernels, analogous to what
CUTLASS provides for CUDA:

| Primitive | What It Does | Used In |
|-----------|-------------|---------|
| `masked_load` / `masked_store` | Boundary-safe tile I/O with padding | Everything |
| `online_softmax` | Numerically stable streaming softmax | Attention |
| `online_mean_var` | Single-pass mean + variance | Normalization |
| `tiled_reduce` | Configurable reduction (sum/max/min) | Normalization, pooling |
| `split_k_accumulate` | Split-K GEMM accumulation | GEMM, MoE |
| `block_quantize` | Per-block INT8/FP8 quantization | Quantized GEMM |
| `swizzle_tile` | L2-cache-friendly tile ordering | GEMM |
| `warp_reduce` | Warp-level reduction primitives | Various |

These are NOT complete kernels — they are composable pieces that
kernel authors combine.

#### 3.2 Testing Framework (`tritonkit.testing`)

```python
import tritonkit as tk

# Correctness: auto-sweep shapes, dtypes, edge cases
tk.testing.assert_matches(
    triton_fn=my_rmsnorm_kernel,
    reference_fn=torch.nn.functional.rms_norm,
    shapes=tk.testing.STANDARD_SHAPES,  # includes edge cases
    dtypes=[torch.float16, torch.bfloat16, torch.float32],
    atol=1e-2, rtol=1e-2
)

# Numerical stability
tk.testing.check_numerical_stability(
    my_kernel, input_range=(-100, 100), check_nan=True, check_inf=True
)

# Hardware portability
tk.testing.run_on_available_backends()  # auto-detect NVIDIA/AMD
```

Built-in shape presets:
- `STANDARD_SHAPES`: Common LLM dimensions (128, 256, 512, ..., 8192)
- `EDGE_SHAPES`: Non-power-of-2, prime numbers, 1-element, very large
- `LLM_SHAPES`: Typical Llama/GPT shapes (4096×4096, 4096×11008, etc.)

#### 3.3 Benchmark Framework (`tritonkit.bench`)

```python
import tritonkit as tk

# Compare your kernel against multiple baselines
results = tk.bench.compare(
    candidates={
        "my_kernel": my_rmsnorm_kernel,
        "liger": liger_kernel.rmsnorm,
        "flaggems": flag_gems.rmsnorm,
        "pytorch": torch.nn.functional.rms_norm,
    },
    shapes=tk.bench.LLM_SHAPES,
    dtypes=[torch.float16, torch.bfloat16],
    metrics=["throughput_gbps", "latency_us", "tflops"],
)

# Auto-generate comparison table
results.print_table()

# Auto-generate roofline plot
results.plot_roofline(save="rmsnorm_roofline.png")

# Export for website
results.export_json("rmsnorm_results.json")
```

Built-in baselines:
- cuBLAS (via PyTorch)
- FlashAttention (if installed)
- Liger-Kernel (if installed)
- FlagGems (if installed)
- PyTorch eager / torch.compile

#### 3.4 Integration Tools (`tritonkit.integrate`)

```python
import tritonkit as tk

# Register as PyTorch custom op (one line)
tk.integrate.register_torch_op(
    "tritonkit::rmsnorm", my_rmsnorm_kernel,
    schema="(Tensor x, Tensor weight, float eps) -> Tensor"
)

# Generate torch.compile-compatible wrapper
tk.integrate.make_compilable(my_rmsnorm_kernel)

# Export autotune configs for current hardware
tk.integrate.export_autotune_config(my_kernel, "configs/ampere_sm86.json")
```

#### 3.5 Example Kernels (`tritonkit.examples`)

Reference kernels built WITH the toolkit, demonstrating best practices:

| Example | Purpose | Uses Primitives |
|---------|---------|----------------|
| `gemm_fp16` | Tiled GEMM with software pipelining | swizzle_tile, split_k |
| `rmsnorm_fused` | Fused RMSNorm + residual | online_mean_var, masked_load |
| `flash_attention` | Flash Attention v2 | online_softmax, masked_load |
| `swiglu_fused` | Fused SwiGLU activation | masked_load |
| `int8_gemm` | INT8 quantized GEMM | block_quantize |

These serve dual purpose: (1) useful kernels people can use directly,
(2) templates showing how to use the toolkit.

### Product 2: `tritonkit.dev` Benchmark Website

#### 3.6 Website Features

**Kernel Performance Matrix**

A searchable table: Operation × Implementation × Hardware → Performance.

```
┌─────────────────────────────────────────────────────────────────────┐
│  tritonkit.dev — GPU Kernel Benchmark Dashboard                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Operation: [Flash Attention ▼]   Dtype: [FP16 ▼]                  │
│  Seq Length: 4096   Head Dim: 128   Causal: Yes                    │
│                                                                     │
│  ┌──────────────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ Implementation   │ RTX 3090 │ A100     │ H100     │ MI300X   │  │
│  ├──────────────────┼──────────┼──────────┼──────────┼──────────┤  │
│  │ FlashAttention-3 │ N/A      │ 245 TF/s │ 512 TF/s │ N/A      │  │
│  │ FlashInfer       │ 198 TF/s │ 238 TF/s │ 505 TF/s │ N/A      │  │
│  │ FlagGems         │ 182 TF/s │ 220 TF/s │ 480 TF/s │ 195 TF/s │  │
│  │ vLLM Triton      │ 175 TF/s │ 215 TF/s │ 515 TF/s │ 188 TF/s │  │
│  │ Triton tutorial  │ 155 TF/s │ 190 TF/s │ 410 TF/s │ 170 TF/s │  │
│  │ PyTorch SDPA     │ 160 TF/s │ 210 TF/s │ 490 TF/s │ 180 TF/s │  │
│  └──────────────────┴──────────┴──────────┴──────────┴──────────┘  │
│                                                                     │
│  [View Raw Data]  [Download CSV]  [Reproduce Locally]               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hardware Comparison View**

Same kernel, same implementation, across GPUs:
- Memory bandwidth utilization (% of peak)
- Compute utilization (% of peak TFLOPS)
- Roofline model overlay

**Framework Kernel Inventory**

Which kernels does each framework use, and are they Triton or CUDA?

```
┌─────────────────┬───────────┬───────────┬───────────┬──────────────┐
│ Operation       │ SGLang    │ vLLM      │ FlagGems  │ Liger-Kernel │
├─────────────────┼───────────┼───────────┼───────────┼──────────────┤
│ Attention       │ FlashInfer│ FA3/Triton│ Triton    │ -            │
│ RMSNorm         │ CUDA      │ CUDA      │ Triton    │ Triton       │
│ SwiGLU          │ CUDA      │ CUDA      │ Triton    │ Triton       │
│ GEMM            │ cuBLAS    │ cuBLAS    │ Triton    │ -            │
│ MoE             │ Triton    │ Triton    │ Triton    │ -            │
│ Quantized GEMM  │ CUTLASS   │ CUTLASS   │ Triton    │ -            │
│ Sampling        │ CUDA      │ CUDA      │ -         │ -            │
│ RoPE            │ CUDA      │ CUDA      │ Triton    │ Triton       │
└─────────────────┴───────────┴───────────┴───────────┴──────────────┘
```

**Community Submissions**

Users can run benchmarks on their own hardware and submit results:

```bash
# Run the benchmark suite locally
pip install tritonkit
tritonkit bench --suite=attention --submit

# Results uploaded to tritonkit.dev with hardware fingerprint
```

### What We Don't Build (Out of Scope)

| Out of Scope | Reason |
|-------------|--------|
| Complete kernel library (competing with FlagGems) | Not our game; we provide tools to build and compare kernels |
| Serving framework | That's SGLang/vLLM's job |
| Training framework | That's PyTorch/Lightning's job |
| Hardware-specific CUDA kernels | Defeats Triton portability |
| Autotuning search algorithms | Triton upstream handles this |

---

## 4. Competitive Positioning

### Why This Doesn't Exist Yet

| Existing Platform | What It Does | What It Misses |
|------------------|-------------|----------------|
| triton-bench.ai | Triton operator benchmarks | No cross-framework comparison; no multi-hardware |
| hud.pytorch.org | PyTorch CI regression tracking | Internal, not cross-framework |
| GPU MODE Leaderboard | Competitive kernel programming | Problem-based, not kernel library-based |
| MLPerf | System-level benchmarks | Not kernel-level; not Triton-specific |
| vLLM Dashboard | vLLM throughput tracking | vLLM-only |

**Our unique position**: The **neutral, cross-framework, cross-hardware**
kernel benchmark platform. We don't sell a framework or a kernel library.
We provide truth.

### Positioning Statement

```
For Triton kernel developers who need to build, test, and compare GPU kernels,
TritonKit is a development toolkit and benchmark platform that provides
composable primitives, unified testing, and cross-hardware performance
visibility. Unlike individual kernel libraries (FlagGems, Liger-Kernel) or
framework-specific dashboards (triton-bench.ai, vLLM dashboard), TritonKit
is framework-neutral and hardware-neutral, showing which kernel is best for
which hardware with reproducible data.
```

---

## 5. Architecture Overview

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    tritonkit.dev (Website)                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Dashboard: Kernel × Hardware × Framework Matrix     │    │
│  │  Roofline plots, comparison charts, CSV export       │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │ REST API                         │
│  ┌────────────────────────▼─────────────────────────────┐    │
│  │  Results Database (benchmark results, hardware info)  │    │
│  └────────────────────────▲─────────────────────────────┘    │
└───────────────────────────┼──────────────────────────────────┘
                            │ Upload
┌───────────────────────────┼──────────────────────────────────┐
│                  tritonkit (Python Library)                    │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ primitives/ │  │  testing/   │  │     bench/           │  │
│  │ Tile-level  │  │ Correctness │  │ Perf measurement     │  │
│  │ building    │  │ Numerical   │  │ Baseline comparison  │  │
│  │ blocks      │  │ Hardware    │  │ Result export        │──┼── tritonkit bench --submit
│  └─────────────┘  └─────────────┘  └──────────────────────┘  │
│                                                               │
│  ┌─────────────┐  ┌──────────────────────────────────────┐   │
│  │ integrate/  │  │           examples/                   │   │
│  │ PyTorch op  │  │ Reference kernels built with toolkit  │   │
│  │ torch.compile│ │ (GEMM, attention, norm, MoE, ...)    │   │
│  │ SGLang/vLLM │  │                                      │   │
│  └─────────────┘  └──────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Python library | Python + Triton | Core user base is Python |
| Benchmark runner | triton.testing.do_bench + custom | Consistent methodology |
| Website frontend | Static site (Next.js or Astro) | Simple, fast, deployable on Vercel/Cloudflare |
| Website backend | Simple API (FastAPI or serverless) | Receives benchmark submissions |
| Database | SQLite or PostgreSQL | Structured benchmark results |
| CI benchmarks | GitHub Actions + self-hosted GPU runner | Reproducible, automated |

### Key Design Principles

1. **Neutral ground.** We benchmark all implementations fairly.
   Same methodology, same shapes, same hardware, same measurement.
   No cherry-picking.

2. **Reproducible.** Every number on the website has a command to reproduce
   it locally. `tritonkit bench --reproduce <result-id>`.

3. **Composable, not monolithic.** Users can use primitives without the
   benchmark framework, or the benchmark framework without primitives.
   No all-or-nothing adoption.

4. **Community-driven data.** We run benchmarks on our own hardware
   (Ampere initially), but the community can submit results from any
   hardware. All results include hardware fingerprint and Triton version.

---

## 6. Delivery Roadmap

> **Principle**: Library first, website later. The product must produce
> real results on the local machine before any web UI is built. The website
> is a display layer, not the core product.

### Phase 1: Core Toolkit (Weeks 1–6)

**Goal**: Ship `pip install tritonkit` with testing + benchmarking +
primitives. Everything runs locally on RTX 3080 Ti.

| Week | Deliverable | Acceptance Criteria |
|------|------------|-------------------|
| 1-2 | `tritonkit.testing` — correctness framework | Can validate a kernel against PyTorch reference with one function call |
| 3-4 | `tritonkit.bench` — benchmark framework | Can compare N implementations, generate table + roofline plot in terminal |
| 4-5 | `tritonkit.primitives` — first 4 primitives | masked_load/store, online_softmax, online_mean_var, tiled_reduce |
| 5-6 | `tritonkit.examples` — 3 reference kernels | GEMM, RMSNorm, Flash Attention, all built with the toolkit |

**Milestone**: v0.1 release. `pip install tritonkit` works. Developers can
test and benchmark their kernels in minutes instead of hours.

### Phase 2: Benchmark Data + Expand (Weeks 7–14)

**Goal**: Produce comprehensive benchmark data locally. No website yet.

| Week | Deliverable | Acceptance Criteria |
|------|------------|-------------------|
| 7-8 | Benchmark suite — 6 ops × 5+ implementations | GEMM, RMSNorm, SwiGLU, Softmax, Flash Attention, MoE |
| 9-10 | Run full suite on own hardware (2× RTX 3080 Ti) | Reproducible JSON results with hardware fingerprint |
| 11-12 | Add quantized GEMM benchmarks (INT8, W4A16, FP8) | gemlite, GPTQ-triton, FlagGems compared |
| 13-14 | `tritonkit.integrate` — PyTorch custom op + torch.compile | One-line integration works |

**Milestone**: v0.2. Comprehensive Ampere benchmark data in JSON/CSV.
Framework kernel inventory documented.

### Phase 3: Multi-Hardware + Website (Weeks 15–22)

**Goal**: Expand to more GPUs via vast.ai. Build website to display results.

| Week | Deliverable |
|------|------------|
| 15-16 | Rent H100 via vast.ai — run benchmarks on Hopper |
| 17-18 | Rent AMD MI300 — run benchmarks on AMD |
| 19-20 | Website — self-hosted dashboard with comparison tables |
| 21-22 | Community submission — `tritonkit bench --submit` via PR |

**Milestone**: v0.3. Multi-hardware benchmark data. Website live.

### Phase 4: Growth (Ongoing)

| Activity | Trigger |
|----------|---------|
| Automated nightly benchmarks | Self-hosted GPU runner stable |
| More vast.ai GPU types (L40S, RTX 4090, B200) | Budget allows |
| 2:4 structured sparsity primitives | When ready for deep dive |
| Contribute to triton-lang/kernels | When primitives + examples are mature |
| Surpass triton-bench.ai coverage | When vast.ai benchmarks are automated |

---

## 7. Success Metrics

### Toolkit Metrics

| Metric | Target (3 months) | Target (6 months) |
|--------|-------------------|-------------------|
| PyPI installs / month | 200+ | 2000+ |
| GitHub stars | 100+ | 500+ |
| External contributors | 3+ | 10+ |
| Primitives available | 8+ | 15+ |
| Example kernels | 5+ | 10+ |

### Website Metrics

| Metric | Target (3 months) | Target (6 months) |
|--------|-------------------|-------------------|
| Monthly unique visitors | 500+ | 5000+ |
| Benchmark data points | 500+ | 5000+ |
| Hardware types represented | 2+ (Ampere, community) | 5+ (Ampere, Hopper, AMD, ...) |
| Implementations benchmarked | 5+ | 10+ |
| Community submissions | 10+ | 100+ |

### Influence Metrics

| Metric | Target (12 months) |
|--------|-------------------|
| Cited by framework teams in kernel decisions | 2+ instances |
| Referenced in Triton community discussions | Regular |
| Linked from kernel project READMEs | 3+ projects |

---

## 8. Benchmark Methodology

> No ISO/IEEE standard exists for GPU kernel benchmarking. The following
> methodology is based on MLPerf rules, Triton's `do_bench` implementation,
> and CUTLASS benchmarking practices.

### 8.1 Measurement Protocol

| Step | What | Why |
|------|------|-----|
| 1. JIT warm-up | Call kernel once outside timed region | Exclude compilation time |
| 2. Clock locking | `nvidia-smi --lock-gpu-clocks` | Eliminate boost clock variance (10-20% swing) |
| 3. L2 cache flush | 256 MB `cache.zero_()` before each iteration | Prevent cache hits inflating bandwidth numbers |
| 4. CUDA Events timing | `start_event.elapsed_time(end_event)` | GPU-side timing, excludes Python/CPU overhead |
| 5. Auto-scaled iterations | Time budget based (warmup=25ms, rep=100ms) | Fast kernels get more samples automatically |
| 6. Report median + p20/p80 | `quantiles=[0.5, 0.2, 0.8]` | Robust to outliers; shows variance |

For published/PR-quality results: `warmup=100ms, rep=1000ms` for stability.

### 8.2 Dual-Mode Fairness

| Mode | How | Answers |
|------|-----|---------|
| **Default** | All implementations use their shipped default configs | "Out of the box, which is fastest?" |
| **Best** | Each implementation runs its own autotuning first | "Fully tuned, which is fastest?" |

Both modes are always reported. The autotuning config used in Best mode is
recorded and published alongside results.

### 8.3 Reproducibility Requirements

Every benchmark result includes:

```
{
  "kernel": "flash_attention",
  "implementation": "flaggems",
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3080 Ti",
    "compute_capability": "8.6",
    "driver": "570.86.16",
    "memory_gb": 12,
    "sm_clock_mhz": 1665,
    "mem_clock_mhz": 9501
  },
  "software": {
    "triton": "3.6.0",
    "cuda": "13.1",
    "pytorch": "2.12.0",
    "python": "3.12"
  },
  "params": {
    "dtype": "float16",
    "seq_len": 4096,
    "head_dim": 128,
    "causal": true
  },
  "results": {
    "median_us": 245.3,
    "p20_us": 240.1,
    "p80_us": 251.7,
    "tflops": 182.4,
    "bandwidth_gbps": 1250.3,
    "roofline_efficiency": 0.81
  },
  "reproduce": "tritonkit bench --kernel flash_attention --impl flaggems --dtype fp16 --seq-len 4096"
}
```

### 8.4 Roofline Model

Hardware ceilings are **measured**, not from spec sheets:
- **Bandwidth ceiling**: `cuMemsetAsync` on 4 GB buffer → actual DRAM BW
- **Compute ceiling**: cuBLAS matmul on 8192×8192×8192 → actual TFLOPS
- **Efficiency** = measured / min(arithmetic_intensity × max_BW, max_TFLOPS)

---

## 9. Competitive Analysis: triton-bench.ai

### 9.1 What They Are

triton-bench.ai is operated by **Kernelize Inc.** (founded by former AMD
Triton lead Simon Waters). The underlying engine is Meta's open-source
`pytorch-labs/tritonbench` (339 stars, 77 forks). It benchmarks ~52 Triton
operator types with nightly CI on H100, B200, MI350.

### 9.2 Their Weaknesses

| Weakness | Impact |
|----------|--------|
| No cross-framework comparison (vLLM vs SGLang vs FlagGems) | Cannot answer "which framework's kernel is fastest" |
| No consumer/Ampere GPU coverage | Irrelevant to ~60% of deployed GPUs |
| Bare data table UX, no interactive visualization | Hard to explore and share |
| Login wall on detailed data | Limits public trust and adoption |
| No community self-submission | Closed ecosystem |
| No public methodology documentation | Results not independently verifiable |
| 37/184 kernels fail on AMD (academic paper data) | AMD coverage unreliable |
| No end-to-end model benchmarks | Only isolated kernel metrics |
| No power/energy efficiency metrics | Missing for cost-conscious users |
| 339 GitHub stars, no detectable web traffic | Low adoption |

### 9.3 Our Competitive Strategy

**Do NOT compete on kernel micro-benchmark depth** — Meta has H100/B200
clusters and a full-time team. We cannot out-resource them.

**Compete on user value dimensions they ignore:**

| Dimension | triton-bench.ai | TritonKit target |
|-----------|-----------------|------------------|
| Framework comparison | None | vLLM vs SGLang vs FlagGems vs cuBLAS |
| Consumer GPU coverage | None | RTX 3080 Ti, RTX 4090, A10G (via vast.ai) |
| Visualization | Bare table | Interactive charts, roofline, trends |
| Public access | Login wall | Fully public, shareable permalinks |
| Community submission | Closed | `tritonkit bench --submit` via PR |
| Methodology | Undocumented | Fully transparent, every result reproducible |
| Hardware breadth | 3 datacenter GPUs | 10+ GPUs via vast.ai + community |

**We are a strict superset of triton-bench.ai.** We cover everything they
do (kernel-level benchmarks) plus cross-framework comparison, consumer GPU
coverage, interactive visualization, open methodology, development toolkit,
and community submissions. Their only advantage is Meta's nightly CI
infrastructure — which vast.ai neutralizes at a few dollars per hour.
We compete directly from day one.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Benchmark methodology questioned | High | High | Open-source all code; document methodology (Section 8); invite review |
| Library maintainers object to being benchmarked | Medium | Medium | Be transparent; share results before publishing; frame as ecosystem service |
| triton-bench.ai expands to cover our scope | Medium | Medium | Our cross-framework + consumer GPU angle is structurally different |
| Community doesn't submit benchmark results | Medium | Medium | Seed with own data + vast.ai data; make submission frictionless |
| Primitives API unstable, breaks user code | Medium | Medium | Semantic versioning; deprecation warnings; stability guarantees after v1.0 |
| Scope creep into "full kernel library" | High | Medium | Strict scope: tools + benchmarks, not competing kernel implementations |

---

## 11. Naming and Branding

### Name: `tritonkit`

| Aspect | Decision |
|--------|---------|
| PyPI package | `tritonkit` |
| Import | `import tritonkit as tk` |
| Website | `tritonkit.dev` (self-hosted, Phase 3) |
| GitHub | `tritonkit/tritonkit` |
| CLI | `tritonkit bench`, `tritonkit test` |

**Tagline**: "Build, test, and benchmark Triton kernels."

---

## 12. Decisions Log

Previously open questions, now resolved:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | License | **Apache 2.0** | Matches Triton, PyTorch, and broader ecosystem |
| 2 | Benchmark fairness | **Dual-mode: Default + Best** | Both out-of-box and fully-tuned results; more honest |
| 3 | Website hosting | **Self-hosted** (Phase 3, not urgent) | Full control; library ships first, website is display layer |
| 4 | Community governance | **PR-based submissions** | Simple, auditable, Git-native |
| 5 | triton-bench.ai relationship | **Direct competition** | We are a strict superset; vast.ai neutralizes their hardware advantage |
| 6 | GPU CI | **Local first → vast.ai** | Start on RTX 3080 Ti; expand to vast.ai for H100/MI300/4090 |

### Infrastructure Accounts

GPU expansion via vast.ai when ready for Phase 3. VPS via Vultr for
self-hosted website when ready. Credentials managed separately (not in repo).
