# OpenMM ComputeRange

Evidence: `formal_race_report`. Contract: `data_race_freedom`.

Missing barrier between min-result reads and max-phase writes.

Non-ML molecular simulation; useful concurrency control.

Reference: `reference.ptx`. Candidate: `candidate.ptx`.

Numerical-pass evidence: Real upstream bug-fix provenance and published formal race detection; no individual historical numerical-test pass log recovered.

Failure evidence: `cases/V05_computerange/formal_report.json`



## Primary sources

- https://arxiv.org/html/2511.12638v3#S6.SS4
- https://doi.org/10.5281/zenodo.21529230
- https://github.com/openmm/openmm/commit/65caf22b4
