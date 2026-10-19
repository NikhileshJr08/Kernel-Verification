# Megatron-LM LayerNorm

Evidence: `formal_race_report`. Contract: `data_race_freedom`.

Missing barrier before shared mean/inverse-standard-deviation broadcast.

Specialized artifact; use one warp, not the nominal geometry in the CUDA header.

Reference: `reference.ptx`. Candidate: `candidate.ptx`.

Numerical-pass evidence: Real upstream bug-fix provenance and published formal race detection; no individual historical numerical-test pass log recovered.

Failure evidence: `cases/V07_layernorm/formal_report.json`



## Primary sources

- https://arxiv.org/html/2511.12638v3#S6.SS4
- https://doi.org/10.5281/zenodo.21529230
- https://github.com/NVIDIA/Megatron-LM/commit/4831071c9
