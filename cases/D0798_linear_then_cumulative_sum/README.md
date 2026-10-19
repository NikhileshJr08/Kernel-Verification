# Linear then cumulative sum

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `floating_point_error_and_special_values`.

Audit reports finite errors above its derived tolerance and non-finite-mask disagreements.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D0798_linear_then_cumulative_sum/numerical_pass.json

Failure evidence: `cases/D0798_linear_then_cumulative_sum/audit.json`

- CMP-01: random[0]: max_err=1.477e-02 > atol=3.419e-04 (scaled); random[1]: max_err=1.218e-02 > atol=2.934e-04 (scaled); random[2]: max_err=1.172e-02 > atol=2.469e-04 (scaled)
- CMP-03: shape=(8, 128): max_err=9.561e-03 > atol=1.814e-04; shape=(32, 128): max_err=1.110e-02 > atol=2.371e-04
- EXC-01: scattered_pos_inf: NaN mask mismatch; scattered_pos_inf: -Inf mask mismatch (sign flip?); scattered_neg_inf: NaN mask mismatch
- EXC-02: subnormal non-zero values disagree: max_err=1.768e-04
- ORD-01: max_err=9.540e-03 > atol=2.178e-04

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
