# Gimlet scalar reconstructions — excluded from the 20 original cases

Source: [Formally Verifying AI-Generated GPU Kernels](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).

These are reduced models written for this repository, not the complete original generated kernels. The article reports numerical-test escapes for these examples. Its abbreviated code is insufficient to recover full original sources and launch configurations.

| ID | Reference | Candidate | Useful witness / limit |
|---|---|---|---|
| G01 | Attention using clamped scores | Clamp omitted | Scores -6 and -7; clamp to -5 makes equal softmax weights |
| G02 | Clamp, scale, clamp, unscale | Folded bound assumes scale 2 | Positive scale 1, pre-clamp value 0.75 |
| G03 | HardSigmoid using division by 6 | Multiplication by rounded FP32 reciprocal | Real coefficient mismatch; acceptance depends on floating-point error contract |

`scripts/scalar_witnesses.py` implements G01/G02 formulas; `scripts/smt_smoke.py` includes G02/G03 obligations. G03's witness belongs to the real-coefficient model and does not establish different rounded FP32 outputs at that same input.
