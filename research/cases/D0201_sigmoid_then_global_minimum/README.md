# Sigmoid then global minimum

Evidence: `numerical_pass_then_adversarial_failure`. Contract: `ieee_special_values`.

Min-reduction handling of NaNs differs from the reference.

Published adversarial testing result, not an SMT proof or a documented production incident. Final and best round are both 5, avoiding a last-vs-best extraction ambiguity. Upstream ID join supplies the pass/fail provenance; locally extracted candidate hashes are recorded, but the audit does not publish per-row candidate hashes.

Reference: `reference.py`. Candidate: `candidate.py`.

Numerical-pass evidence: cases/D0201_sigmoid_then_global_minimum/numerical_pass.json

Failure evidence: `cases/D0201_sigmoid_then_global_minimum/audit.json`

- EXC-01: scattered_nan: NaN mask mismatch
- PRC-01: torch.float16: exception — at 9:8:
def _sigmoid_min_stage1_kernel(x_ptr, n_elements, partial_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input; masked OOB lanes get 0.0
    x = tl.load(x_pt...
- PRC-02: exception during execution: at 9:8:
def _sigmoid_min_stage1_kernel(x_ptr, n_elements, partial_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input; masked OOB lanes get 0.0
    x = tl.load(x_p...

## Primary sources

- https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/audit_rows.jsonl.gz
- https://github.com/RishiShah99/lethe/blob/92330ed8130f00f6a1c66d7cc213c8ff7e4f5492/results/kernelbench_actual_rows.jsonl
- https://arxiv.org/html/2608.12700v2
