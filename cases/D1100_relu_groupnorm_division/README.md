# ReLU, GroupNorm, division

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `floating_point_error_and_special_values`.

Audit reports large-input normalization error and non-finite-mask disagreement.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D1100_relu_groupnorm_division/numerical_pass.json

Failure evidence: `cases/D1100_relu_groupnorm_division/audit.json`

- CMP-01: large_1e6: max_err=4.343e+00 > atol=1.296e-03 (scaled)
- EXC-01: scattered_nan: NaN mask mismatch; scattered_pos_inf: NaN mask mismatch; scattered_pos_inf: -Inf mask mismatch (sign flip?)

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
