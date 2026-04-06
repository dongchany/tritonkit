# TritonKit — 产品手册

> **状态**: 草案 v0.3（决策已定，竞争分析已加入）
> **作者**: 叶东成
> **日期**: 2026-04-07

---

## 1. 产品愿景

### 一句话定位

一个 Triton kernel 开发工具链和跨硬件 benchmark 平台 ——
让写 kernel、测 kernel、比 kernel 变简单，覆盖所有实现、框架和硬件。

### 为什么从 v0.1 转型

v0.1 想做"又一个 Triton kernel 库"。市场分析发现：
- 已有 50+ 个 Triton kernel 项目（FlagGems、Liger-Kernel、gemlite、
  FLASHNN、conch、attorch 等）
- 标准 GEMM 用 Triton 写只能达到 cuBLAS 的 85-100% —— 没有说服力
- 端到端 LLM 推理用自定义 kernel 的真实提升是 10-30%，不是好几倍
- 真正的痛点不是"我们需要更多 kernel"，而是"写 kernel、测 kernel、
  比较 kernel 的过程是割裂和痛苦的"

v0.2 转型为：**解决 kernel 开发体验问题，提供生态缺失的 benchmark 可见层。**

### 要解决的问题

今天写一个 Triton kernel 需要：

1. 从零开始写（或从 tutorial 复制粘贴）
2. 自己写正确性测试（对照 PyTorch 参考）
3. 自己写 benchmark（对照 cuBLAS 等）
4. 自己做 autotuning（换硬件就得重来）
5. 自己写 PyTorch 集成的样板代码（custom op 注册）
6. 只能用 `tl.device_print` 调试（没有真正的 debugger）
7. 写完才发现别人已经写过类似的
8. 没有办法知道："我的 kernel 和其他方案比到底怎么样？"

这些环节没有统一的基础设施。

与此同时，整个生态缺少可见性：
- "哪个 flash attention 实现在我的 RTX 3090 上最快？"
- "FlagGems 的 RMSNorm 和 Liger-Kernel 的在 A100 上谁更快？"
- "vLLM 的 Triton attention 后端在 AMD 上比 FlashInfer 快吗？"

今天没有人能回答这些问题。Benchmark 分散在各个 repo 里，
在不同硬件上跑，用不同的方法论。

### 解决方案

**TritonKit**：两个互补的产品：

1. **tritonkit**（Python 库）：可组合的原语、统一的测试框架、
   benchmark 框架、集成工具，用于 Triton kernel 开发。

2. **tritonkit.dev**（网站）：公开的 benchmark 仪表板，展示
   kernel 性能在 实现 × 硬件 × 框架 三个维度的对比。
   GPU kernel 界的 "Can I Use"。

---

## 2. 目标用户

### 第一优先级：Triton Kernel 开发者

**用户画像**：为 PyTorch、SGLang、vLLM 或自有项目编写自定义 Triton kernel 的
工程师。包括有经验的 GPU 程序员和正在学习 Triton 的 ML 工程师。

**核心需求**：更快的开发周期。不要为每个 kernel 重新发明测试、benchmark 和
集成基础设施。

**我们如何服务**：
- 可组合的 tile 级原语（masked load、online softmax、split-K 等）
- 一行代码对照 PyTorch 参考进行正确性测试
- 一行代码对照 cuBLAS/FlashAttention 等进行 benchmark
- 一行代码注册 PyTorch custom op
- 按硬件预调优的 autotuning 配置

**成功指标**：kernel 开发时间从数天缩短到数小时。

### 第二优先级：框架开发者（SGLang、vLLM 等）

**用户画像**：构建 LLM 推理/训练框架、需要选择和维护 kernel 后端的团队。

**核心需求**：数据驱动的 kernel 选型。"Ampere 上应该默认用哪个 attention 实现？"
目前靠直觉或一次性 benchmark 来回答。

**我们如何服务**：
- 覆盖所有主要 kernel 实现的公开 benchmark 数据
- 有可复现数据支撑的硬件专属推荐
- 即插即用的 kernel 集成工具

**成功指标**：框架团队在 kernel 选型时引用我们的 benchmark 数据。

### 第三优先级：广泛 ML 社区

**用户画像**：想了解 GPU kernel 性能但不自己写 kernel 的 ML 工程师、研究者、学生。

**核心需求**："我的任务应该用 A100 还是 H100？" "FP8 在我的硬件上真的更快吗？"
"哪个框架最快？"

**我们如何服务**：benchmark 网站提供清晰、可视化的答案。

**成功指标**：网站成为人们在讨论中链接的参考来源。

---

## 3. 产品范围

### 产品 1：`tritonkit` Python 库

#### 3.1 可组合原语（`tritonkit.primitives`）

用于构建 kernel 的可复用构建块，类似于 CUTLASS 之于 CUDA：

| 原语 | 功能 | 适用场景 |
|------|------|---------|
| `masked_load` / `masked_store` | 边界安全的 tile I/O + padding | 所有场景 |
| `online_softmax` | 数值稳定的流式 softmax | Attention |
| `online_mean_var` | 单遍 mean + variance | Normalization |
| `tiled_reduce` | 可配置的归约（sum/max/min） | Normalization、pooling |
| `split_k_accumulate` | Split-K GEMM 累加 | GEMM、MoE |
| `block_quantize` | 按块 INT8/FP8 量化 | 量化 GEMM |
| `swizzle_tile` | L2 cache 友好的 tile 排列 | GEMM |
| `warp_reduce` | Warp 级归约原语 | 各种场景 |

这些**不是**完整 kernel —— 它们是可组合的部件，kernel 开发者可以自由组合。

#### 3.2 测试框架（`tritonkit.testing`）

```python
import tritonkit as tk

# 正确性：自动扫描 shape、dtype、边界情况
tk.testing.assert_matches(
    triton_fn=my_rmsnorm_kernel,
    reference_fn=torch.nn.functional.rms_norm,
    shapes=tk.testing.STANDARD_SHAPES,  # 包含边界情况
    dtypes=[torch.float16, torch.bfloat16, torch.float32],
    atol=1e-2, rtol=1e-2
)

# 数值稳定性
tk.testing.check_numerical_stability(
    my_kernel, input_range=(-100, 100), check_nan=True, check_inf=True
)

# 硬件可移植性
tk.testing.run_on_available_backends()  # 自动检测 NVIDIA/AMD
```

内置 shape 预设：
- `STANDARD_SHAPES`：常见 LLM 维度（128, 256, 512, ..., 8192）
- `EDGE_SHAPES`：非 2 的幂、素数、单元素、超大尺寸
- `LLM_SHAPES`：典型 Llama/GPT shape（4096×4096, 4096×11008 等）

#### 3.3 Benchmark 框架（`tritonkit.bench`）

```python
import tritonkit as tk

# 将你的 kernel 与多个基线对比
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

# 自动生成对比表格
results.print_table()

# 自动生成 roofline 图
results.plot_roofline(save="rmsnorm_roofline.png")

# 导出到网站
results.export_json("rmsnorm_results.json")
```

内置基线：
- cuBLAS（通过 PyTorch）
- FlashAttention（如已安装）
- Liger-Kernel（如已安装）
- FlagGems（如已安装）
- PyTorch eager / torch.compile

#### 3.4 集成工具（`tritonkit.integrate`）

```python
import tritonkit as tk

# 注册为 PyTorch custom op（一行代码）
tk.integrate.register_torch_op(
    "tritonkit::rmsnorm", my_rmsnorm_kernel,
    schema="(Tensor x, Tensor weight, float eps) -> Tensor"
)

# 生成 torch.compile 兼容包装器
tk.integrate.make_compilable(my_rmsnorm_kernel)

# 导出当前硬件的 autotune 配置
tk.integrate.export_autotune_config(my_kernel, "configs/ampere_sm86.json")
```

#### 3.5 示例 Kernel（`tritonkit.examples`）

用本工具链构建的参考 kernel，展示最佳实践：

| 示例 | 目的 | 使用的原语 |
|------|------|-----------|
| `gemm_fp16` | 带软件流水线的分块 GEMM | swizzle_tile, split_k |
| `rmsnorm_fused` | 融合 RMSNorm + residual | online_mean_var, masked_load |
| `flash_attention` | Flash Attention v2 | online_softmax, masked_load |
| `swiglu_fused` | 融合 SwiGLU 激活 | masked_load |
| `int8_gemm` | INT8 量化 GEMM | block_quantize |

双重用途：(1) 可直接使用的实用 kernel，(2) 展示工具链用法的模板。

### 产品 2：`tritonkit.dev` Benchmark 网站

#### 3.6 网站功能

**Kernel 性能矩阵**

可搜索的表格：操作 × 实现 × 硬件 → 性能。

```
┌─────────────────────────────────────────────────────────────────────┐
│  tritonkit.dev — GPU Kernel Benchmark 仪表板                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  操作: [Flash Attention ▼]   数据类型: [FP16 ▼]                    │
│  序列长度: 4096   Head维度: 128   因果: 是                          │
│                                                                     │
│  ┌──────────────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ 实现             │ RTX 3090 │ A100     │ H100     │ MI300X   │  │
│  ├──────────────────┼──────────┼──────────┼──────────┼──────────┤  │
│  │ FlashAttention-3 │ N/A      │ 245 TF/s │ 512 TF/s │ N/A      │  │
│  │ FlashInfer       │ 198 TF/s │ 238 TF/s │ 505 TF/s │ N/A      │  │
│  │ FlagGems         │ 182 TF/s │ 220 TF/s │ 480 TF/s │ 195 TF/s │  │
│  │ vLLM Triton      │ 175 TF/s │ 215 TF/s │ 515 TF/s │ 188 TF/s │  │
│  │ Triton tutorial  │ 155 TF/s │ 190 TF/s │ 410 TF/s │ 170 TF/s │  │
│  │ PyTorch SDPA     │ 160 TF/s │ 210 TF/s │ 490 TF/s │ 180 TF/s │  │
│  └──────────────────┴──────────┴──────────┴──────────┴──────────┘  │
│                                                                     │
│  [查看原始数据]  [下载 CSV]  [本地复现]                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**硬件对比视图**

同一 kernel，同一实现，跨 GPU 对比：
- 内存带宽利用率（占峰值 %）
- 计算利用率（占峰值 TFLOPS %）
- Roofline 模型叠加

**框架 Kernel 清单**

每个框架使用哪些 kernel，是 Triton 还是 CUDA？

```
┌─────────────────┬───────────┬───────────┬───────────┬──────────────┐
│ 操作            │ SGLang    │ vLLM      │ FlagGems  │ Liger-Kernel │
├─────────────────┼───────────┼───────────┼───────────┼──────────────┤
│ Attention       │ FlashInfer│ FA3/Triton│ Triton    │ -            │
│ RMSNorm         │ CUDA      │ CUDA      │ Triton    │ Triton       │
│ SwiGLU          │ CUDA      │ CUDA      │ Triton    │ Triton       │
│ GEMM            │ cuBLAS    │ cuBLAS    │ Triton    │ -            │
│ MoE             │ Triton    │ Triton    │ Triton    │ -            │
│ 量化 GEMM       │ CUTLASS   │ CUTLASS   │ Triton    │ -            │
│ Sampling        │ CUDA      │ CUDA      │ -         │ -            │
│ RoPE            │ CUDA      │ CUDA      │ Triton    │ Triton       │
└─────────────────┴───────────┴───────────┴───────────┴──────────────┘
```

**社区提交**

用户可以在自己的硬件上运行 benchmark 并提交结果：

```bash
# 在本地运行 benchmark 套件
pip install tritonkit
tritonkit bench --suite=attention --submit

# 结果连同硬件指纹上传到 tritonkit.dev
```

### 不构建的内容（范围外）

| 范围外 | 原因 |
|-------|------|
| 完整 kernel 库（与 FlagGems 竞争） | 不是我们的赛道；我们提供构建和比较 kernel 的工具 |
| 推理框架 | 那是 SGLang/vLLM 的工作 |
| 训练框架 | 那是 PyTorch/Lightning 的工作 |
| 硬件专属 CUDA kernel | 违背 Triton 可移植性 |
| Autotuning 搜索算法 | Triton 上游负责 |

---

## 4. 竞争定位

### 为什么这个还不存在

| 现有平台 | 它做什么 | 它缺什么 |
|---------|---------|---------|
| triton-bench.ai | Triton 算子 benchmark | 无跨框架对比；无多硬件 |
| hud.pytorch.org | PyTorch CI 回归追踪 | 内部使用，非跨框架 |
| GPU MODE Leaderboard | 竞赛式 kernel 编程 | 基于题目，非 kernel 库级别 |
| MLPerf | 系统级 benchmark | 非 kernel 级；非 Triton 专属 |
| vLLM Dashboard | vLLM 吞吐追踪 | 仅 vLLM |

**我们的独特定位**：**中立的、跨框架的、跨硬件的** kernel benchmark 平台。
我们不卖框架也不卖 kernel 库。我们提供真相。

### 定位声明

```
对于需要构建、测试和比较 GPU kernel 的 Triton kernel 开发者，
TritonKit 是一个开发工具链和 benchmark 平台，提供可组合原语、
统一测试和跨硬件性能可见性。与具体的 kernel 库（FlagGems、
Liger-Kernel）或框架专属仪表板（triton-bench.ai、vLLM dashboard）
不同，TritonKit 是框架中立和硬件中立的，用可复现的数据告诉你
哪个 kernel 在哪个硬件上最好。
```

---

## 5. 架构概览

### 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    tritonkit.dev（网站）                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  仪表板：Kernel × 硬件 × 框架 矩阵                    │    │
│  │  Roofline 图、对比图表、CSV 导出                       │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │ REST API                         │
│  ┌────────────────────────▼─────────────────────────────┐    │
│  │  结果数据库（benchmark 结果、硬件信息）                  │    │
│  └────────────────────────▲─────────────────────────────┘    │
└───────────────────────────┼──────────────────────────────────┘
                            │ 上传
┌───────────────────────────┼──────────────────────────────────┐
│                  tritonkit（Python 库）                        │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ primitives/ │  │  testing/   │  │     bench/           │  │
│  │ Tile 级     │  │ 正确性      │  │ 性能测量             │  │
│  │ 构建块      │  │ 数值稳定性  │  │ 基线对比             │  │
│  │             │  │ 硬件兼容    │  │ 结果导出             │──┼── tritonkit bench --submit
│  └─────────────┘  └─────────────┘  └──────────────────────┘  │
│                                                               │
│  ┌─────────────┐  ┌──────────────────────────────────────┐   │
│  │ integrate/  │  │           examples/                   │   │
│  │ PyTorch op  │  │ 用工具链构建的参考 kernel              │   │
│  │ torch.compile│ │ (GEMM, attention, norm, MoE, ...)    │   │
│  │ SGLang/vLLM │  │                                      │   │
│  └─────────────┘  └──────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### 技术栈

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| Python 库 | Python + Triton | 核心用户群是 Python |
| Benchmark 运行器 | triton.testing.do_bench + 自定义 | 一致的方法论 |
| 网站前端 | 静态站点（Next.js 或 Astro） | 简单、快速、可部署到 Vercel/Cloudflare |
| 网站后端 | 简单 API（FastAPI 或 serverless） | 接收 benchmark 提交 |
| 数据库 | SQLite 或 PostgreSQL | 结构化 benchmark 结果 |
| CI benchmark | GitHub Actions + 自托管 GPU runner | 可复现、自动化 |

### 核心设计原则

1. **中立立场。** 公平 benchmark 所有实现。
   相同的方法论、shape、硬件、测量方式。不挑选数据。

2. **可复现。** 网站上的每个数字都有本地复现的命令。
   `tritonkit bench --reproduce <result-id>`。

3. **可组合而非整体。** 用户可以只用原语而不用 benchmark 框架，
   或只用 benchmark 框架而不用原语。不存在"全部采纳或全不采纳"。

4. **社区驱动的数据。** 我们在自己的硬件（初期是 Ampere）上运行 benchmark，
   但社区可以从任何硬件提交结果。所有结果包含硬件指纹和 Triton 版本。

---

## 6. 交付路线图

> **原则**：先做库，后做网站。产品必须先在本机跑出真实结果，
> 才去做任何 Web UI。网站只是展示层，不是核心产品。

### Phase 1：核心工具链（第 1–6 周）

**目标**：发布 `pip install tritonkit`，包含测试 + benchmark + 原语。
一切在本机 RTX 3080 Ti 上运行。

| 周次 | 交付物 | 验收标准 |
|------|--------|---------|
| 1-2 | `tritonkit.testing` — 正确性框架 | 一个函数调用即可验证 kernel 对照 PyTorch 参考 |
| 3-4 | `tritonkit.bench` — benchmark 框架 | 可对比 N 个实现，在终端生成表格 + roofline 图 |
| 4-5 | `tritonkit.primitives` — 首批 4 个原语 | masked_load/store, online_softmax, online_mean_var, tiled_reduce |
| 5-6 | `tritonkit.examples` — 3 个参考 kernel | GEMM、RMSNorm、Flash Attention，全部用工具链构建 |

**里程碑**：v0.1 发布。`pip install tritonkit` 可用。
开发者可以在几分钟而非几小时内测试和 benchmark 自己的 kernel。

### Phase 2：Benchmark 数据 + 扩展（第 7–14 周）

**目标**：在本机产出全面 benchmark 数据。暂不做网站。

| 周次 | 交付物 | 验收标准 |
|------|--------|---------|
| 7-8 | Benchmark 套件 — 6 种操作 × 5+ 种实现 | GEMM、RMSNorm、SwiGLU、Softmax、Flash Attention、MoE |
| 9-10 | 在自有硬件（2× RTX 3080 Ti）上运行完整套件 | 可复现 JSON 结果 + 硬件指纹 |
| 11-12 | 增加量化 GEMM benchmark（INT8、W4A16、FP8） | gemlite、GPTQ-triton、FlagGems 对比 |
| 13-14 | `tritonkit.integrate` — PyTorch custom op + torch.compile | 一行代码集成可用 |

**里程碑**：v0.2。全面的 Ampere benchmark 数据（JSON/CSV）。
框架 kernel 清单已文档化。

### Phase 3：多硬件 + 网站（第 15–22 周）

**目标**：通过 vast.ai 扩展到更多 GPU。搭建网站展示结果。

| 周次 | 交付物 |
|------|--------|
| 15-16 | 通过 vast.ai 租用 H100 — 在 Hopper 上运行 benchmark |
| 17-18 | 租用 AMD MI300 — 在 AMD 上运行 benchmark |
| 19-20 | 网站 — 自托管仪表板，包含对比表格 |
| 21-22 | 社区提交 — `tritonkit bench --submit`（通过 PR） |

**里程碑**：v0.3。多硬件 benchmark 数据。网站上线。

### Phase 4：增长（持续进行）

| 活动 | 触发条件 |
|------|---------|
| 自动化每夜 benchmark | 自托管 GPU runner 稳定后 |
| 更多 vast.ai GPU 类型（L40S、RTX 4090、B200） | 预算允许时 |
| 2:4 结构化稀疏原语 | 时机成熟时深入 |
| 向 triton-lang/kernels 贡献 | 当原语和示例成熟时 |
| 超越 triton-bench.ai 覆盖 | 当 vast.ai benchmark 自动化后 |

---

## 7. 成功指标

### 工具链指标

| 指标 | 目标（3 个月） | 目标（6 个月） |
|------|---------------|---------------|
| PyPI 安装量/月 | 200+ | 2000+ |
| GitHub stars | 100+ | 500+ |
| 外部贡献者 | 3+ | 10+ |
| 可用原语数 | 8+ | 15+ |
| 示例 kernel 数 | 5+ | 10+ |

### 网站指标

| 指标 | 目标（3 个月） | 目标（6 个月） |
|------|---------------|---------------|
| 月独立访客 | 500+ | 5000+ |
| Benchmark 数据点 | 500+ | 5000+ |
| 涵盖硬件类型 | 2+（Ampere + 社区） | 5+（Ampere、Hopper、AMD...） |
| 已 benchmark 的实现 | 5+ | 10+ |
| 社区提交 | 10+ | 100+ |

### 影响力指标

| 指标 | 目标（12 个月） |
|------|---------------|
| 被框架团队在 kernel 选型中引用 | 2+ 次 |
| 在 Triton 社区讨论中被引用 | 经常 |
| 被 kernel 项目 README 链接 | 3+ 个项目 |

---

## 8. Benchmark 方法论

> GPU kernel benchmarking 没有 ISO/IEEE 正式标准。以下方法论基于 MLPerf 规则、
> Triton `do_bench` 实现和 CUTLASS benchmarking 实践。

### 8.1 测量协议

| 步骤 | 做什么 | 为什么 |
|------|--------|--------|
| 1. JIT 预热 | 在计时区域外调用 kernel 一次 | 排除编译时间 |
| 2. 锁定时钟 | `nvidia-smi --lock-gpu-clocks` | 消除 boost clock 波动（10-20% 摆幅） |
| 3. L2 cache 刷新 | 每次迭代前 256 MB `cache.zero_()` | 防止缓存命中虚高带宽数字 |
| 4. CUDA Events 计时 | `start_event.elapsed_time(end_event)` | GPU 侧计时，排除 Python/CPU 开销 |
| 5. 自动缩放迭代数 | 基于时间预算（warmup=25ms, rep=100ms） | 快 kernel 自动获得更多采样 |
| 6. 报告 median + p20/p80 | `quantiles=[0.5, 0.2, 0.8]` | 抗异常值；显示方差 |

发布/PR 级结果使用：`warmup=100ms, rep=1000ms` 以确保稳定性。

### 8.2 双模式公平性

| 模式 | 做法 | 回答的问题 |
|------|------|-----------|
| **Default** | 所有实现使用默认出厂配置 | "开箱即用谁最快？" |
| **Best** | 每个实现先运行自己的 autotuning | "充分调优后谁最快？" |

两种模式始终同时报告。Best 模式中使用的 autotuning 配置会随结果一起记录和公开。

### 8.3 可复现性要求

每条 benchmark 结果包含：

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

### 8.4 Roofline 模型

硬件天花板使用**实测值**，不用规格书数据：
- **带宽天花板**：`cuMemsetAsync` 在 4 GB buffer 上 → 实际 DRAM 带宽
- **计算天花板**：cuBLAS matmul 在 8192×8192×8192 上 → 实际 TFLOPS
- **效率** = 实测值 / min(算术强度 × 最大带宽, 最大 TFLOPS)

---

## 9. 竞争分析：triton-bench.ai

### 9.1 他们是谁

triton-bench.ai 由 **Kernelize Inc.** 运营（创始人 Simon Waters，前 AMD Triton
团队负责人）。底层引擎是 Meta 开源的 `pytorch-labs/tritonbench`（339 stars，
77 forks）。Benchmark 约 52 种 Triton 算子，H100/B200/MI350 每夜 CI。

### 9.2 他们的弱点

| 弱点 | 影响 |
|------|------|
| 不做跨框架对比（vLLM vs SGLang vs FlagGems） | 无法回答"哪个框架的 kernel 最快" |
| 不覆盖消费级/Ampere GPU | 与约 60% 已部署 GPU 无关 |
| 裸数据表格 UX，无交互可视化 | 难以探索和分享 |
| 详细数据有登录墙 | 限制公众信任和采纳 |
| 无社区自助提交 | 封闭生态 |
| 无公开方法论文档 | 结果无法独立验证 |
| 37/184 kernel 在 AMD 上失败（学术论文数据） | AMD 覆盖不可靠 |
| 无端到端模型 benchmark | 仅孤立 kernel 指标 |
| 无功耗效率指标 | 缺少成本敏感用户关心的数据 |
| 339 GitHub stars，无可检测的网站流量 | 低采纳率 |

### 9.3 我们的竞争策略

**不在 kernel 微 benchmark 深度上正面竞争** — Meta 有 H100/B200 集群和全职团队。
我们无法在资源上超过他们。

**在他们忽视的用户价值维度上竞争：**

| 维度 | triton-bench.ai | TritonKit 目标 |
|------|-----------------|---------------|
| 框架对比 | 无 | vLLM vs SGLang vs FlagGems vs cuBLAS |
| 消费级 GPU | 无 | RTX 3080 Ti、RTX 4090、A10G（via vast.ai） |
| 可视化 | 裸表格 | 交互图表、roofline、趋势线 |
| 公开访问 | 登录墙 | 完全公开，可分享永久链接 |
| 社区提交 | 封闭 | `tritonkit bench --submit`（通过 PR） |
| 方法论 | 未文档化 | 完全透明，每条结果可复现 |
| 硬件广度 | 3 种数据中心 GPU | 10+ 种 GPU（via vast.ai + 社区） |

**我们是 triton-bench.ai 的严格超集。** 我们覆盖他们做的一切（kernel 级
benchmark），还有跨框架对比、消费级 GPU 覆盖、交互可视化、公开方法论、
开发工具链和社区提交。他们唯一的优势是 Meta 的 nightly CI 基础设施 ——
vast.ai 几美元/小时就能中和这个优势。我们从第一天起直接竞争。

---

## 10. 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|-------|------|---------|
| Benchmark 方法论被质疑 | 高 | 高 | 开源所有代码；文档化方法论（第 8 章）；邀请评审 |
| 库维护者反对被 benchmark | 中 | 中 | 透明操作；发布前分享结果；定位为生态服务 |
| triton-bench.ai 扩展到覆盖我们的范围 | 中 | 中 | 我们的跨框架 + 消费级 GPU 角度是结构性差异 |
| 社区不提交 benchmark 结果 | 中 | 中 | 用自有数据 + vast.ai 数据播种；让提交零摩擦 |
| 原语 API 不稳定，破坏用户代码 | 中 | 中 | 语义化版本；弃用警告；v1.0 后稳定性保证 |
| 范围蔓延，变成"完整 kernel 库" | 高 | 中 | 严守范围：工具 + benchmark，不是竞争的 kernel 实现 |

---

## 11. 命名与品牌

### 名称：`tritonkit`

| 方面 | 决定 |
|------|------|
| PyPI 包名 | `tritonkit` |
| 导入名 | `import tritonkit as tk` |
| 网站 | `tritonkit.dev`（自托管，Phase 3） |
| GitHub | `tritonkit/tritonkit` |
| CLI | `tritonkit bench`, `tritonkit test` |

**标语**："构建、测试、对比 Triton kernel。"

---

## 12. 决策记录

此前的待决事项，现已确定：

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| 1 | 开源协议 | **Apache 2.0** | 与 Triton、PyTorch 及更广泛生态一致 |
| 2 | Benchmark 公平性 | **双模式：Default + Best** | 同时报告开箱即用和充分调优结果；更诚实 |
| 3 | 网站托管 | **自托管**（Phase 3，不紧急） | 完全控制权；先发库，网站是展示层 |
| 4 | 社区治理 | **PR 方式提交** | 简单、可审计、Git 原生 |
| 5 | 与 triton-bench.ai 关系 | **直接竞争** | 我们是严格超集；vast.ai 中和他们的硬件优势 |
| 6 | GPU CI | **先本机 → 后 vast.ai** | 从 RTX 3080 Ti 起步；扩展到 vast.ai 覆盖 H100/MI300/4090 |

### 基础设施帐号

GPU 扩展（Phase 3）通过 vast.ai。网站 VPS 通过 Vultr。
凭据单独管理（不放在 repo 中）。
