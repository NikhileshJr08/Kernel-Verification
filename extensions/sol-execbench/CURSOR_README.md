# Multi-Agent CUDA Kernel Optimizations

Solutions and metrics from Cursor's multi-agent system that autonomously optimized 235 CUDA kernels for NVIDIA Blackwell B200 GPUs, achieving a **38% geomean speedup** over baselines.

Read the full writeup: [Speeding up GPU kernels by 38% with a multi-agent system](https://cursor.com/blog/multi-agent-kernels)

## Repository structure

- **`L1/`** — 94 single-operator kernel problems (e.g. attention, RoPE, RMSNorm)
- **`L2/`** — 82 multi-operator fused kernel problems (e.g. full decoder layers, MoE routing)
- **`Quant/`** — 33 quantized kernel problems (FP8, NVFP4)
- **`FlashInfer-Bench/`** — 26 problems benchmarked against FlashInfer (GEMM, GQA, MoE, fused ops)
- **`combined_metrics.csv`** — Per-workload results: baseline latency, SOL latency, selected latency, and SOL score
- **`problem_level_metrics.csv`** — Per-problem aggregate results: SOL score and speedup vs. baseline

Each problem directory contains `src/` (the kernel solution), `solution.json`, and `traces.jsonl`.
