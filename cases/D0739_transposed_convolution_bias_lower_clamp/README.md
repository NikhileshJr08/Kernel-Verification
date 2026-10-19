# Transposed convolution, bias, lower clamp

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `ieee_special_values`.

Fused clamp uses tl.maximum; audit reports a NaN-mask disagreement.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D0739_transposed_convolution_bias_lower_clamp/numerical_pass.json

Failure evidence: `cases/D0739_transposed_convolution_bias_lower_clamp/audit.json`

- EXC-01: scattered_nan: NaN mask mismatch

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
