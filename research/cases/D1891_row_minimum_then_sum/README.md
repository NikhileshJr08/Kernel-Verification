# Row minimum then sum

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `determinism_and_special_values`.

Unordered FP32 atomics in final reduction; audit also reports a NaN-mask disagreement.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D1891_row_minimum_then_sum/numerical_pass.json

Failure evidence: `cases/D1891_row_minimum_then_sum/audit.json`

- EXC-01: scattered_nan: NaN mask mismatch
- EXC-02: flush-to-zero behaviour disagrees with reference
- ORD-02: run 0 and run 4 differ (non-deterministic)

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
