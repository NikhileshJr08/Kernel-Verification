# Findings and limits

## Best starting point for exact-real equivalence

**D1088: softmax → logsumexp → tanh.** In `candidate.py`, `mask = offs < C` protects the first load, but `sum_exp_probs = tl.sum(tl.exp(scaled), axis=0)` includes all padded lanes. Their first softmax probability is zero, so they each contribute one to this second reduction. With ten channels and a tile of sixteen, six extra terms enter the sum. Larger autotune tiles add even more.

For zero logits, the correct intermediate sum is `10*exp(1/10)`; the candidate computes that sum plus six for the smallest tile. The subsequent rational expression for tanh(log(S)) differs. `scripts/scalar_witnesses.py` evaluates this with 70-digit decimal arithmetic. `scripts/smt_smoke.py` checks the corresponding equality obligation using `E=exp(1/10)>1`. The equality query is UNSAT, establishing a mismatch in this source-derived scalar model, not verifying that the model faithfully captures all compiler behavior.

This is an original generated kernel, not a newly seeded mutation. Its recorded numerical pass and later audit failure are retained beside the code. The source-level explanation is our analysis of that code.

## Cases for other contracts

Normalization cases D0416 and D0463 compute variance through FP32 second moments. Cancellation can break them despite real-arithmetic equivalence of the variance formulas. D1100, D0278, D0798, and D0883 have additional large-input or special-value findings. Do not count an exact-real verifier's acceptance of an algebraically valid optimization as a false negative for a floating-point error contract.

D0064, D0201, and D0739 are useful for non-finite propagation. The primary records describe differences in NaN/Inf masks. These require IEEE-aware semantics and precise specifications of primitives such as min/max. NaNs and infinities are not real-number inputs.

D0967 and D1891 contain atomic accumulation and have recorded run-to-run differences. Floating-point non-associativity can make an otherwise acceptable parallel reduction nondeterministic. Bitwise determinism must be a stated requirement before calling that observation a correctness violation. D1891 additionally has special-value findings.

The eight VOLTA cases exercise shared-memory synchronization. Five preserve actual before/after-fix artifacts; three are tutorial reductions. They establish race findings, not a scalar mathematical mismatch. OpenMM's three cases are explicitly non-ML controls. Megatron-LM's two cases concern LayerNorm and its input gradient. CUDA artifact headers can contain nominal geometry that is inappropriate for specialized PTX. The archive's three racy reductions specialize to eight threads, whereas the current Rust benchmark expects later 128-thread PTX files; our wrapper uses direct CLI analysis at eight threads for those originals. The published reports cover both historical runs, but do not provide file hashes that prove an exact byte-for-byte association. Do not reuse their 128-thread timings for the eight-thread artifacts.

## Gimlet examples

The requested [Gimlet article](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) supplies abbreviated examples, not complete downloadable original pairs for every example. We therefore keep three small reconstructions under `supplemental/`, outside the original-case count. They cover an omitted attention clamp, a fixed scaling constant, and a rounded reciprocal in HardSigmoid.

The HardSigmoid example is particularly sensitive to the equivalence definition: inequality between exact-real expressions with a recovered FP32 coefficient is not automatically an unacceptable FP32 error. The supplemental SMT script labels that distinction explicitly. No end-to-end Gimlet performance comparison is claimed.

## Evidence quality

- **Formal reports:** original artifact code, before/after fixes where available, and published verifier outputs. Historical per-kernel numerical-pass logs are missing for these entries.
- **Numerical escapes:** original generated code, source trajectory, numerical checker result, and adversarial gate result joined by ID. These records are author-published evidence; they were not independently replayed on a GPU here.
- **Local scalar checks:** reduced formulas derived from source. Their results are independently reproducible on a CPU; they do not certify compiler, memory, or concurrency semantics.

We excluded a tempting Triton matmul issue (#9839) from the corpus: its attribution of valid-output error to wrapped out-of-range rows is not established by the issue's code, and its posted output indexing contains a separate stride problem. A reported issue alone is insufficient ground truth. Synthetic mutation corpora and Lethe's deliberately cheating kernels were also excluded from the twenty original cases.

## What this collection does not establish

It does not establish twenty production failures, twenty exact-real inequivalences, an independent reproduction of every author finding, or a speed advantage for any new verifier. It is a curated starting set with explicit evidence tiers. A publication requiring fifteen to twenty independently validated, exact-real, test-escaping ML kernels needs further collection and GPU replay; the mixed cases here must not be relabeled to satisfy that denominator.

Primary sources: [VOLTA paper](https://arxiv.org/html/2511.12638v3), [original artifact](https://doi.org/10.5281/zenodo.21529230), [VOLTA implementation](https://github.com/willtunnels/volta), [Lethe audit paper](https://arxiv.org/html/2608.12700v2), [Lethe source and records](https://github.com/RishiShah99/lethe), [pinned DR.Kernel dataset](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k/tree/cba0ef0). Exact acquisition revisions are recorded in `evidence/upstream-lock.json`.
