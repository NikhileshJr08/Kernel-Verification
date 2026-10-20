# Local validation

Prepared October 19, 2026.

| Check | Result | Scope |
|---|---|---|
| Corpus integrity | PASS: 20 cases, 40 reference/candidate files | Source hashes, Python syntax, PTX entry presence, evidence joins, unmodified trajectory extraction |
| Published formal evidence | PASS: eight RACE findings; five also have post-fix OK records | Author-published records; historical code/hash association limitations remain documented |
| Published numerical/adversarial joins | PASS: twelve numerical passes and twelve gate-failure records | Corpus ID joins; every selected final/best round is 5 |
| Scalar witnesses | PASS: D1088 plus G01/G02 | Source-derived formulas; no compiled GPU execution |
| Z3 checks | PASS: three obligations | D1088 equality UNSAT; G02 inequality SAT; G03 real-coefficient inequality SAT |
| VOLTA command construction | PASS: race-harness and direct eight-thread reduction commands inspected | Does not establish runtime CLI acceptance or verifier results |
| Original GPU kernels | NOT RUN | No CUDA device, PyTorch, or Triton available here |
| VOLTA locally | NOT RUN | Rust/Cargo unavailable here |

`environment-preflight.json` records the expected missing-PyTorch environment error. It is not a kernel failure. No new GPU or full-kernel formal timings are claimed.

The reference/candidate files are preserved as retrieved. In particular, the original eight-thread reduction artifact has not been replaced by a newly generated 128-thread kernel.
