# Attribution and licensing

- `vendor/volta/`: willtunnels/volta, MIT. Revision in `evidence/upstream-lock.json`; original LICENSE retained and copied into `licenses/volta.txt`.
- `vendor/volta-benchmarks/` and copied V-case CUDA/PTX: authors of *Equivalence Checking of ML GPU Kernels*, original Zenodo artifact, MIT. See `licenses/volta-artifact.txt`. Its kernels include adaptations from NVIDIA tutorials, OpenMM, Megatron-LM, and other sources identified in the original code. Preserve those embedded attribution comments.
- `vendor/lethe/` and copied audit result records: Rishi Shah / Rishav Shrestha, Lethe, MIT. Original license retained and copied into `licenses/lethe.txt`.
- D-case reference/candidate code and trajectories: `hkust-nlp/drkernel-coldstart-8k`, HKUST NLP, declared MIT in the pinned dataset card (`licenses/drkernel-dataset-card.md`). The card also documents upstream query provenance. The dataset contains generated candidate code and reference programs; no claim is made that this curation changes their licensing or ownership.
- Supplemental Gimlet examples are newly written scalar reconstructions with source attribution, not redistributed full original kernels. Article prose is not vendored.

The repository-level MIT license applies to newly authored wrappers, metadata, analysis, and documentation. It does not replace upstream notices.
