# Megatron-LM GradInput

Evidence: `formal_race_report`. Contract: `data_race_freedom`.

Shared reduction indices collide across threadIdx.y.

The fixed artifact also changes launch geometry from 32x4 to 32x1; compare kernel plus launch configuration.

Reference: `reference.ptx`. Candidate: `candidate.ptx`.

Numerical-pass evidence: Real upstream bug-fix provenance and published formal race detection; no individual historical numerical-test pass log recovered.

Failure evidence: `cases/V08_gradinput/formal_report.json`



## Primary sources

- https://arxiv.org/html/2511.12638v3#S6.SS4
- https://doi.org/10.5281/zenodo.21529230
- https://github.com/NVIDIA/Megatron-LM/commit/f1295380c
