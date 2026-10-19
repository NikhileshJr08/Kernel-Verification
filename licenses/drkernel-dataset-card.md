---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- triton
- code-generation
- reinforcement-learning
---
---

# DR.Kernel Cold-Start Dataset

[**Paper**](https://huggingface.co/papers/2602.05885) | [**Code**](https://github.com/hkust-nlp/KernelGYM)

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-hkust--nlp/drkernel--coldstart--8k-yellow)](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k)

This directory documents the format of `hkust-nlp/drkernel-coldstart-8k`.

The cold-start set is used for supervised fine-tuning (SFT) before RL in DR.Kernel. As described in the paper, it is built from **5-turn multi-turn trajectories** collected with KernelGYM feedback.

## Overview

- Purpose: initialize kernel-generation ability (Triton coding + iterative optimization) before TRLOO/MRS/PR/PRS RL.
- Data form: one row per full multi-turn trajectory.
- Current local Parquet (`drkernel-coldstart-8k.parquet`) contains **8,920 trajectories**.

## Dataset Structure

The file is a Parquet table with these columns:

| Field | Type | Description |
|---|---|---|
| `messages` | `list<struct<role: string, content: string>>` | Complete multi-turn chat history for one trajectory |
| `uuid` | `int64` | Sample id (0..N-1) |
| `entry_point` | `string` | Entry class/function name (`Model`) |
| `repo_name` | `string` | Reserved metadata |
| `module_name` | `string` | Reserved metadata |
| `final_speedup` | `double` | Final speedup from the trajectory |
| `num_rounds` | `int64` | Number of user-assistant rounds (fixed to 5 in this release) |
| `original_python_code` | `string` | Original PyTorch architecture |
| `best_round` | `int64` | Best-performing round index (1..5) |
| `timestamp` | `double` | Collection timestamp |
| `conversion_mode` | `string` | Conversion tag (`full_conversation_enhanced`) |
| `enable_thinking` | `bool` | Thinking flag (all `false` in this release) |

### Conversation Format

Each row in `messages` is a full 5-turn trajectory with **10 messages** and fixed role order:

```text
user, assistant, user, assistant, user, assistant, user, assistant, user, assistant
```

Pattern by turn:

1. Turn 1 user message:
   - Task template for Triton kernel optimization
   - A reference inline Triton example
   - The target PyTorch `Model` code (`get_inputs`, `get_init_inputs`)
   - Instruction to return `ModelNew`
2. Turn 1 assistant message:
   - Analysis/plan + first implementation
3. Turn 2-5 user messages:
   - KernelGYM server feedback blob (compile/correctness/speedup/profiling/runtime error)
   - Request to improve the previous implementation
4. Turn 2-5 assistant messages:
   - Refined implementation based on feedback

Example (simplified):

```json
{
  "messages": [
    {"role": "user", "content": "You write custom Triton kernels ... Optimize `Model` and output `ModelNew` ..."},
    {"role": "assistant", "content": "Analysis ... ```python\
# ModelNew v1\
...\
```"},
    {"role": "user", "content": "Server feedback: {\"compiled\": true, \"correctness\": false, \"speedup\": 0.0, ...}"},
    {"role": "assistant", "content": "Diagnosis ... ```python\
# ModelNew v2\
...\
```"},
    {"role": "user", "content": "Server feedback: {...}"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Server feedback: {...}"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Server feedback: {...}"},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Usage

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
print(ds.column_names)
# ['messages', 'uuid', 'entry_point', 'repo_name', 'module_name', 'final_speedup',
#  'num_rounds', 'original_python_code', 'best_round', 'timestamp',
#  'conversion_mode', 'enable_thinking']
```

### SFT Training (Multi-Turn)

Use the multi-turn dataset path in VERL/DR.Kernel:

```bash
cd drkernel/kernel/scripts/sft
bash 8b-coldstart.sh
# or
bash 14b-coldstart.sh
```

Recommended data config:

```bash
data.multiturn.enable=True
data.multiturn.messages_key=messages
data.multiturn.enable_thinking_key=enable_thinking
data.max_length=18432
data.truncation=right
```

## Data Collection Notes

Consistent with the DR.Kernel paper (Section 4.1 and Appendix prompt templates):

- Cold-start data is distilled from strong proprietary teachers through multi-turn interaction.
- Each turn appends execution feedback to prompt iterative refinement.
- This dataset is the SFT warm-up stage before multi-turn RL.

## Query Source and Attribution

- The optimization queries/tasks used to build these trajectories are sourced from:
  - [ByteDance-Seed/cudaLLM-data](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data)
- We respect and acknowledge the original dataset authors and contributors.
- `hkust-nlp/drkernel-coldstart-8k` mainly contributes multi-turn trajectory construction (iterative feedback + refinement) on top of those query/task sources.
- If you use this dataset, please credit both DR.Kernel and the original query source (`ByteDance-Seed/cudaLLM-data`).

## Related Resources

| Resource | Link |
|---|---|
| DR.Kernel Paper | [arXiv:2602.05885](https://arxiv.org/abs/2602.05885) |
| KernelGYM Repo | [GitHub](https://github.com/hkust-nlp/KernelGYM) |
| DR.Kernel Training README | [`drkernel/README.md`](../../drkernel/README.md) |
| KernelGYM Root README | [`README.md`](../../README.md) |
| Query Source Dataset | [ByteDance-Seed/cudaLLM-data](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data) |
| RL Training Data | [hkust-nlp/drkernel-rl-data](https://huggingface.co/datasets/hkust-nlp/drkernel-rl-data) |
| Validation Data | [hkust-nlp/drkernel-validation-data](https://huggingface.co/datasets/hkust-nlp/drkernel-validation-data) |

## Citation

```bibtex
@article{liuetal2026,
  title={Dr.Kernel: Reinforcement Learning Done Right for Triton Kernel Generations},
  author={Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He},
  journal={arXiv:2602.05885},
  year={2026}
}
```

## License

MIT License