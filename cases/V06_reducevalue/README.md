# OpenMM ReduceValue

Evidence: `formal_race_report`. Contract: `data_race_freedom`.

Shared reduction buffer reused before all readers finish.

Non-ML molecular simulation; useful concurrency control.

Reference: `reference.ptx`. Candidate: `candidate.ptx`.

Numerical-pass evidence: Real upstream bug-fix provenance and published formal race detection; no individual historical numerical-test pass log recovered.

Failure evidence: `cases/V06_reducevalue/formal_report.json`



## Primary sources

- https://arxiv.org/html/2511.12638v3#S6.SS4
- https://doi.org/10.5281/zenodo.21529230
- https://github.com/openmm/openmm/commit/de666e305
