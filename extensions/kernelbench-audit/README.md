# ABC audit / Sakana archive

The acquired ABC revision includes the candidate CUDA files directly. Its preprocessing script assigns each row its positional dataframe index before selecting `Correct=True`. We used that exact rule against the pinned Sakana Level 1 parquet.

For all 129 files in `tests_verified/level_1`, the candidate equals the Sakana `CUDA_Code` string byte-for-byte, and the same row has `Correct=True`. Each case includes both original PyTorch reference forms, numerical-result metadata, and the verified adversarial test. Where present, the earlier model-generated test is also retained.

**The verified-test directory is not a per-case failure report.** Some tests explicitly skip unsupported layouts; others may pass. No per-candidate replay logs were found in the acquired repository. Do not call every recovered case a numerical-test escape until replay establishes a valid failure.

The original environment instructions specify Python 3.11.11 and CUDA 12.4. Dependencies are retained in `upstream/requirements.txt`.

```bash
python scripts/replay_kernelbench_audit.py 3492 --out results/A03492-local.json
```

This wrapper stages the unmodified CUDA source, test, and original build helper in the directory layout expected by ABC. It requires PyTorch, a CUDA GPU, nvcc, pytest-timeout and pytest-json-report. Inspect failure and skip counts; a skipped test is not a correctness pass.

Source: [ABC repository](https://github.com/uiuc-kang-lab/agentic-benchmarks/tree/main/benchmarks/kernel-bench), [Sakana archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive). Exact revisions are in `../upstream-lock.json`. Sakana data is CC BY 4.0; retain the attribution and citation in `DATASET_CARD.md`.
