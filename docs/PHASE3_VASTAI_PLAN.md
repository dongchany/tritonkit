# Phase 3 — vast.ai Multi-Hardware Benchmark Rental Plan

> **Status**: Draft v0.1
> **Date**: 2026-04-07
> **Goal**: Run TritonKit benchmark suite on hardware we don't have locally (H100, A100, RTX 4090, etc.) to surpass triton-bench.ai's 3-GPU coverage.

---

## 1. Workflow Per GPU

```
1. Spin up vast.ai instance (Docker template, pre-built image)     ~3 min
2. SSH in, clone tritonkit repo                                     ~1 min
3. uv pip install -e .[dev]                                         ~2 min
4. python benchmarks/run_all.py                                     ~10-15 min
5. scp results JSON back to local                                   ~1 min
6. Destroy instance (CRITICAL — stops billing)                      ~1 min
─────────────────────────────────────────────────────────────────
Total wall time per GPU                                             ~20-25 min
```

**Buffer for first-time debugging**: +10 min → **30 min per GPU first run**.
**Reruns** (after Docker image cached): ~15 min.

---

## 2. Pricing Reference (vast.ai, April 2026)

| GPU | Cheapest/hr | Verified host typical | Why benchmark this |
|-----|------------|----------------------|-------------------|
| **RTX 4080 16GB** | $0.11 | $0.15-$0.25 | Cheap consumer baseline |
| **RTX 4090 24GB** | $0.24 | $0.29-$0.50 | Most popular consumer GPU |
| **A100 PCIe 40GB** | $0.29 | $0.52-$0.60 | Datacenter Ampere baseline |
| **L40S 48GB** | $0.47 | $0.47-$0.70 | Ada Lovelace inference GPU |
| **A100 SXM 80GB** | $0.67 | $1.20-$1.32 | Common LLM training GPU |
| **H100 PCIe 80GB** | $1.47 | $1.53-$1.87 | Hopper baseline |
| **H100 SXM 80GB** | $0.90-$1.54 | $1.54-$1.87 | Most common Hopper |
| **H200 141GB** | $1.99 | $2.23-$2.54 | Latest Hopper |
| **B200 192GB** | $3.74 | $3.81-$6.25 | Blackwell flagship |
| **MI300X 192GB** | N/A on vast.ai | N/A | Use Crusoe ($0.95/hr) or skip |

**Storage**: ~$0.005-0.01/GB/hour (set by host). 50GB workspace × 1 hour ≈ $0.25-0.50.

**Bandwidth**: $0.05-0.20/GB egress. JSON results are <1 MB, negligible.

---

## 3. Tiered Rental Plans

### Tier 1 — Minimum Viable Multi-Hardware ($5 budget)

**Goal**: Prove cross-hardware story with the two most-asked-about GPUs.

| GPU | Time | Cost (verified) |
|-----|------|----------------|
| RTX 4090 | 30 min | $0.25 |
| H100 SXM | 30 min | $1.00 |
| **Storage + buffer** | | $1.00 |
| **Total** | **1 hr** | **~$2.25** |

Add a 2-3x safety factor for retries/debugging: **$5-8 total**.

**Outcome**: 3 GPUs on tritonkit.dev (RTX 3080 Ti local + RTX 4090 + H100). Already exceeds triton-bench.ai's consumer GPU coverage (zero).

---

### Tier 2 — Core Coverage ($20 budget)

**Goal**: Cover both consumer and datacenter Ampere/Hopper.

| GPU | Time | Cost (verified) |
|-----|------|----------------|
| RTX 4080 | 30 min | $0.15 |
| RTX 4090 | 30 min | $0.25 |
| L40S | 30 min | $0.35 |
| A100 PCIe 40GB | 30 min | $0.30 |
| A100 SXM 80GB | 30 min | $0.66 |
| H100 SXM | 30 min | $1.00 |
| **Storage + bandwidth** | | $1-2 |
| **Subtotal** | **3 hrs** | **~$4-5** |

**Buffer (reruns, debugging, 2-pass)**: 3-4x → **$15-20 total**.

**Outcome**: 7 GPUs covered. Already **2x more than triton-bench.ai's 3 GPUs (H100/B200/MI350)**, with the additional consumer coverage they completely lack.

---

### Tier 3 — Full Coverage with H200 ($40 budget)

**Goal**: Add H200 for memory-rich Hopper, full consumer + datacenter sweep.

| GPU | Time | Cost (verified) |
|-----|------|----------------|
| All Tier 2 GPUs | 3 hrs | $5 |
| H200 141GB | 30 min | $1.25 |
| **Storage + buffer** | | $2-3 |
| **Subtotal** | **3.5 hrs** | **~$8-9** |

**Buffer (3x for retries, debugging, methodology validation)**: **$30-40 total**.

**Outcome**: 8 GPUs covered. Includes the latest 141GB H200 that few benchmark sites have.

---

### Tier 4 — Premium with Blackwell ($80 budget)

**Goal**: Include the flagship B200 to validate Triton's portability story.

| GPU | Time | Cost (verified) |
|-----|------|----------------|
| All Tier 3 GPUs | 3.5 hrs | $9 |
| B200 192GB | 30 min | $2 |
| Multi-GPU 8x H100 SXM (1 node) | 1 hr | $12-15 |
| **Storage + buffer** | | $3-5 |
| **Subtotal** | **5 hrs** | **~$28-32** |

**Buffer (2.5x)**: **$70-80 total**.

**Outcome**: 9 GPUs + multi-GPU configuration. Comprehensive coverage that beats triton-bench.ai on every dimension.

---

## 4. Recommended Approach

**Start with Tier 1 ($5-8)** to validate the entire workflow:
1. The Docker image works
2. tritonkit installs cleanly on a remote GPU
3. Benchmark suite runs to completion
4. Results are exportable

If Tier 1 succeeds, **graduate to Tier 2 ($15-20)** for the full datacenter + consumer sweep.

Defer **Tier 3 (H200) and Tier 4 (B200)** until after the website is up and we know which GPUs people actually care about (data-driven prioritization).

---

## 5. Operational Plan

### 5.1 Pre-flight (One-Time Setup)

**Build a Docker image** with everything pre-installed:

```dockerfile
# Dockerfile.tritonkit-bench
FROM nvcr.io/nvidia/pytorch:25.03-py3   # PyTorch 2.10+, CUDA 12.x

RUN pip install --no-cache-dir \
    triton>=3.6.0 \
    tabulate \
    pytest \
    matplotlib

# Pre-install tritonkit (will be re-installed on each run for latest)
RUN git clone https://github.com/dongchany/tritonkit /opt/tritonkit \
    && cd /opt/tritonkit \
    && pip install -e .

WORKDIR /workspace
```

Push to Docker Hub: `dongchany/tritonkit-bench:latest`

**Saves ~10 min per GPU** (no need to install triton/torch/etc).

### 5.2 Per-GPU Run Script

```bash
#!/bin/bash
# scripts/vastai_run.sh
# Usage: ./scripts/vastai_run.sh <instance_id>

INSTANCE_ID=$1
GPU_NAME=$2  # e.g., "rtx_4090"

# 1. Wait for instance ready
vastai ssh-url $INSTANCE_ID

# 2. SSH and run
ssh root@<instance> << 'EOF'
cd /opt/tritonkit
git pull
pip install -e . --quiet
python benchmarks/run_all.py
EOF

# 3. Copy results back
mkdir -p benchmarks/results/vastai/$GPU_NAME
scp root@<instance>:/opt/tritonkit/benchmarks/results/*.json \
    benchmarks/results/vastai/$GPU_NAME/

# 4. Commit results
git add benchmarks/results/vastai/
git commit -m "Add benchmark results for $GPU_NAME from vast.ai"
git push

# 5. CRITICAL: destroy instance
vastai destroy instance $INSTANCE_ID
echo "Instance $INSTANCE_ID destroyed"
```

### 5.3 Search Filters on vast.ai

For **reproducible benchmarks**, filter for:
- `verified=true` (datacenter hosts, consistent hardware)
- `rentable=true` (immediately available)
- `dlperf > X` (GPU performance index above threshold)
- `inet_down > 100` (decent download bandwidth for image pulls)
- `disk_space > 50` (room for Docker image + workspace)
- `cuda_max_good >= 12.4` (recent CUDA toolkit)

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Storage charges accumulate after benchmark done | **Always destroy instances**, never just stop. Use a `--auto-destroy` script. |
| Host kicks instance mid-benchmark (interruptible) | Use **on-demand verified** instances only. |
| Docker pull is slow on cold instance | Pre-built image; fall back to lighter base. |
| Different CUDA versions cause Triton compile failures | Pin Triton 3.6.0; use NGC PyTorch image with known-good CUDA. |
| Benchmark variance from host noise | Run **3 trials per GPU**, report median + p20/p80 (already our methodology). |
| Forget to commit results before destroying | Script enforces commit-before-destroy. |
| vast.ai pricing fluctuates | Lock in price at instance creation, not search time. |

---

## 7. Budget Approval Request

| Tier | Budget | Outcome |
|------|--------|---------|
| **Tier 1** | $5-8 | RTX 4090 + H100, validate workflow |
| **Tier 2** | $15-20 | 7 GPUs, surpass triton-bench.ai on coverage |
| **Tier 3** | $30-40 | 8 GPUs incl. H200 |
| **Tier 4** | $70-80 | 9 GPUs incl. B200 + multi-GPU node |

**My recommendation**: Start with **Tier 1 ($5-8)** to validate. If everything works, immediately do **Tier 2 ($15-20)**. The combined cost is **$20-30** for 7 GPUs of public benchmark data, which is the entire competitive moat against triton-bench.ai.

Defer Tier 3/4 until website is live and we have actual user feedback on which GPUs matter most.

---

## 8. Decision Required

1. **Approve Tier 1 + Tier 2 ($20-30 total)?** This unlocks 7 GPUs of benchmark data.
2. **Docker image hosting**: Use `dongchany/tritonkit-bench` on Docker Hub? (Need Docker Hub account.)
3. **vast.ai account funding**: Confirm the existing account has $30+ credit, or top up?

If yes to all, next steps:
1. Build and push Docker image (~30 min one-time)
2. Run Tier 1 (~1-2 hours wall time, $5-8)
3. Verify results, commit, push to GitHub
4. Run Tier 2 (~3-4 hours wall time, $15-20)
5. Update website (Phase 3 follow-up) with new data
