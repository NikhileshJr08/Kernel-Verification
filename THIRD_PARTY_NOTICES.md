# Attribution and licensing

- `vendor/volta/`: willtunnels/volta, MIT. Revision in `evidence/upstream-lock.json`; original LICENSE retained and copied into `licenses/volta.txt`.
- `vendor/volta-benchmarks/` and copied V-case CUDA/PTX: authors of *Equivalence Checking of ML GPU Kernels*, original Zenodo artifact, MIT. See `licenses/volta-artifact.txt`. Its kernels include adaptations from NVIDIA tutorials, OpenMM, Megatron-LM, and other sources identified in the original code. Preserve those embedded attribution comments.
- `vendor/lethe/` and copied audit result records: Rishi Shah / Rishav Shrestha, Lethe, MIT. Original license retained and copied into `licenses/lethe.txt`.
- D-case reference/candidate code and trajectories: `hkust-nlp/drkernel-coldstart-8k`, HKUST NLP, declared MIT in the pinned dataset card (`licenses/drkernel-dataset-card.md`). The card also documents upstream query provenance. The dataset contains generated candidate code and reference programs; no claim is made that this curation changes their licensing or ownership.
- Supplemental Gimlet examples are newly written scalar reconstructions with source attribution, not redistributed full original kernels. Article prose is not vendored.

The repository-level MIT license applies to newly authored wrappers, metadata, analysis, and documentation. It does not replace upstream notices.

## Requested-source expansion

- `extensions/kernelbench-audit/cases/`: CUDA and reference programs from the SakanaAI AI-CUDA-Engineer Archive, CC BY 4.0. Its full dataset-card attribution and citation are retained. ABC's tests and helpers are copied from the pinned uiuc-kang-lab/agentic-benchmarks revision; no replacement license is asserted for those upstream files.
- `extensions/flashattention/`: original Dao-AILab FlashAttention source/test snapshots and patches, BSD 3-Clause; see `licenses/flashattention.txt`. The separate issue #674 reproducer is attributed to its original GitHub issue author and repository.
- `extensions/megatron/`: original Megatron-LM source and patch snapshots; see `licenses/megatron.txt`. Verification artifacts retain the earlier VOLTA notices.
- `extensions/sol-execbench/cases/`: Cursor's published source bundles and evaluation-result extracts from the pinned anysphere/kernel-optimization-results repository. The new wrapper MIT license does not relicense this upstream code. NVIDIA reference/workload data is not bundled; its separate evaluation license is retained with a local downloader.
- `extensions/gimlet/`: reference tasks from the pinned Gimlet KernelBench fork, MIT; see `licenses/gimlet-kernelbench.txt`. Complete private generated candidates were not recovered.
