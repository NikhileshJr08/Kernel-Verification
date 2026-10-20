# Specifically requested collections

This extension preserves the original twenty-case release and adds the six requested research tracks. `manifest.json` indexes **162 records**, including recoverable code, test-only evidence, and unresolved leads. This is not a count of 162 confirmed bugs.

| Requested collection | Recovered | Remaining gap |
|---|---|---|
| Gimlet's three examples | Public KernelBench references for clamp/scale and HardSigmoid; existing scalar reconstructions retained | All three complete original generated candidates, wrappers, and original test configurations remain unidentified |
| KernelBench / ABC audit | 129 CUDA candidates, exact Sakana reference pairs, original `Correct=True` metadata, and manually verified tests; all candidate bytes match ABC | No individual adversarial-run reports shipped by ABC; GPU replay pending |
| Megatron LayerNorm and GradInput | Exact parent/fixing-commit CUDA and C++ binding files, patches, specialized CUDA/PTX verification artifacts, published formal outputs, and launch configuration source | Original upstream compiler environment not pinned; artifacts are specializations rather than identical raw upstream files |
| doubleAI / SOL-ExecBench | All 15 named failures, the embedding-gradient mapping, and three additional illustrative cases: 19 Cursor source bundles with original passing-evaluation records | NVIDIA references/workloads require direct local download under their evaluation license; doubleAI did not publish failed workload indices or its exact harness revision |
| VOLTA bounds illustration | Illustration identified and artifact searched | Complete original faulty kernel remains unmapped; unrelated Conv2D code is not substituted |
| Historical Triton FlashAttention | Seven original parent/fix source-and-test pairs; one separate FP32 issue reproducer with its reported compiler version | Exact compiler builds for the seven 2022 fixes remain unpinned; GPU reproduction pending |

Begin with [SOL case mappings](sol-execbench/INDEX.md), [FlashAttention history](flashattention/INDEX.md), [Megatron notes](megatron/README.md), or the [129 audit pair index](kernelbench-audit/INDEX.md).

All code acquisition revisions and SHA-256 values are recorded. New integrity checks are run with `python scripts/check_extensions.py`. The original core manifest and its validation scripts remain usable.

## NVIDIA dataset handling

The NVIDIA Evaluation Dataset License, sections 3.1 and 3.2, excludes training use and redistribution. Accordingly, this repository does not embed NVIDIA reference programs, workload records, or parquet data. It supplies names, hashes, pins, and a downloader. The original passing records extracted from Cursor omit their workload field. Read `sol-execbench/NVIDIA_EVALUATION_LICENSE.txt` before downloading.

```bash
python -m pip install pyarrow
python scripts/fetch_nvidia_references.py --case 087_embedding_with_initial_layernorm_backward --accept-nvidia-evaluation-license
```

Downloads go to ignored `local-data/`. Do not redistribute that directory or use that dataset to train a model. Benchmarking verifier execution and using source material as model-training data are different uses.

No new GPU correctness or performance results are claimed. A recovered test, a historical bug-fixing commit, a published numerical pass, and an independently reproduced failure are distinct evidence states.
