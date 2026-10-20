# Harris reduction Red-5

Evidence: `formal_race_report`. Contract: `data_race_freedom`.

Unsynchronized shared-memory accesses in the warp-reduction tail.

Original Zenodo PTX/CUDA use BLOCKSIZE=8. Current Rust benchmark expects later BLOCKSIZE=128 PTX, so replay uses direct CLI at eight threads. Published reports are historical bug-family evidence, without per-file hash binding. Historical timings from the 128-thread recompilation must not be attached to these bytes. Integer reduction, not an LLM-generated floating-point ML kernel. Uninitialized shared-memory reads may coexist with the race.

Reference: `reference.ptx`. Candidate: `candidate.ptx`.

Numerical-pass evidence: Tutorial provenance; no individual historical numerical-test log recovered.

Failure evidence: `cases/V01_reduction_5/formal_report.json`



## Primary sources

- https://arxiv.org/html/2511.12638v3#S6.SS1.SSS1
- https://doi.org/10.5281/zenodo.21529230
