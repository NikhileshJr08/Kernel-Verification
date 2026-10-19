# Softmax, logsumexp, tanh

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `real_arithmetic_equivalence`.

Masked softmax lanes become zero probabilities; exp(0)=1 then contributes to the next sum. Candidate omits masking that sum.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D1088_softmax_logsumexp_tanh/numerical_pass.json

Failure evidence: `cases/D1088_softmax_logsumexp_tanh/audit.json`

- CMP-01: random[0]: max_err=1.441e-02 > atol=2.725e-04 (scaled); random[1]: max_err=1.441e-02 > atol=2.725e-04 (scaled); random[2]: max_err=1.441e-02 > atol=2.725e-04 (scaled)
- CMP-03: shape=(16, 10, 32, 32): max_err=1.441e-02 > atol=2.725e-04; shape=(64, 10, 32, 32): max_err=1.441e-02 > atol=2.725e-04; shape=(32, 10, 32, 64): max_err=1.441e-02 > atol=2.725e-04
- EXC-02: subnormal non-zero values disagree: max_err=1.441e-02
- ORD-01: max_err=1.440e-02 > atol=2.725e-04
- ORD-03: not within atol=1.0e-03/rtol=1.0e-03: max_err=1.441e-02 (reduction order differs)
- PRC-01: torch.float16: max_err=1.416e-02 > atol=1.000e-03

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
