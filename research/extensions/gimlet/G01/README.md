# Runnable attention-clamp reconstruction

Based on the reference/candidate semantics in [Gimlet's blog](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels#the-limits-of-numerical-testing).
These are newly written PyTorch implementations and a demonstration harness, not
the recovered original generated kernel, original numerical tests, or verifier.
They are excluded from the original-case counts and source provenance manifests.

- `reference.py`: scaled attention with score clamp before softmax and output clamp.
- `candidate.py`: intentionally faulty SDPA replacement with only the output clamp.
- `run.py`: seeded random comparison and a deterministic counterexample.

Install PyTorch if needed (`python3 -m pip install torch`), then from the repository root:

```bash
python3 extensions/gimlet/G01/run.py
python3 extensions/gimlet/G01/run.py --device cuda:0
```

The CPU path requires no CUDA or Triton. CUDA execution requires a compatible
PyTorch installation and NVIDIA GPU; an unavailable device raises an error.
The example uses eager PyTorch and its SDPA operator, with backend selection left
to PyTorch. It makes no performance claim or claim of reproducing a Triton kernel.

For small random Q/K values the clamp is inactive and outputs agree within
floating-point tolerance. This is a deliberately limited input distribution,
not a claim that all random inputs pass.
For Q=[1], K=[-6,-7], V=[0,1], the reference clamps scores to [-5,-5] and returns
0.5. The candidate uses unequal softmax weights and returns approximately
0.26894142. Both results lie inside the final output clamp, which cannot hide
this discrepancy. Successful execution asserts both behaviors and exits zero.

The harness also checks batched multi-head attention with Q/K/V shape
`(2, 4, 128, 64)`, exhibiting the same missing-clamp discrepancy.
