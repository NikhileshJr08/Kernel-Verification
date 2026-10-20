# Triplet-margin loss then mean

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `floating_point_error_and_special_values`.

Audit reports large-input error and NaN/Inf-mask disagreements.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D0883_triplet_margin_loss_then_mean/numerical_pass.json

Failure evidence: `cases/D0883_triplet_margin_loss_then_mean/audit.json`

- CMP-01: large_1e6: max_err=1.562e-02 > atol=1.078e-05 (scaled)
- EXC-01: scattered_nan: NaN mask mismatch; scattered_pos_inf: NaN mask mismatch; scattered_neg_inf: NaN mask mismatch

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
