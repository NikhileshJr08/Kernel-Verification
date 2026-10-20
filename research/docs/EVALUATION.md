# Comparing a new verification approach

## Freeze the contract first

Choose one contract per experiment. Possible contracts include exact-real equality, bounded floating-point error, non-finite propagation, race freedom, memory safety, and deterministic execution. Record the allowed shapes, layouts, dtypes, scalar parameter ranges, aliasing assumptions, launch geometry, and output footprint. Allowing a model to specialize to a declared fixed shape is different from silently assuming that shape while claiming shape polymorphism.

Floating-point inputs form a finite set for a fixed shape, but exhaustive testing is still impractical. Exact-real equality, IEEE-754 equality, and tolerance-based agreement are different properties. Formal guarantees depend on the chosen semantics and supported input domain.

## Suggested experiment order

1. **Semantic mismatch:** analyze D1088. Check both the original candidate and a separately labeled corrected control that masks the second reduction. Do not overwrite the original file.
2. **Concurrency:** run the eight V-cases with the supplied launch configurations. For the five pre/post pairs, require rejection before the fix and race freedom after it. Only count equivalence results if equivalence was actually checked.
3. **Floating-point robustness:** use the D-cases with finite large-input errors, then add special values and determinism if supported. Report unsupported contracts separately.
4. **Correct controls:** use the vendored VOLTA correct reduction, matmul, or attention pairs, plus reference-versus-itself checks under the same contracts. Avoid measuring rejection alone.

No generated corrected control has been substituted for an upstream reference in this release.

## Fair timing

For each approach, separately measure parsing/lowering, symbolic execution or decomposition, VC construction, solving, and total wall time. Record peak memory, thread count, hardware, solver/compiler versions, timeout, warm/cold cache status, and number of output elements checked. Repeat timed runs on the same machine and report median plus spread. Exclude build/install time or report it separately for every system.

VOLTA's `generate` and `solve` commands expose phase boundaries. Its VC dump/replay mode can compare solving without repeating symbolic execution. Do not compare your solve-only time to its full pipeline. A race rejection during generation has no subsequent equivalence-solver phase to accelerate.

Gimlet's article does not provide a public end-to-end verifier artifact in the source inspected here. Source-derived SMT models can be useful microbenchmarks, but they cannot support a claim that an implementation outperforms Gimlet's full system.

## Required result labels

Use `equivalent`, `counterexample`, `race`, `memory_error`, `unknown`, `timeout`, `unsupported`, and `tool_error` as distinct outcomes. An unproved candidate is not necessarily incorrect. Never aggregate UNKNOWN or unsupported cases as detected bugs.

For each result, store a counterexample or race trace when available. Check numeric counterexamples against the original compiled GPU programs under the declared arithmetic semantics. For a concurrency bug, preserve the conflicting accesses, launch geometry, and relevant schedule assumptions. Scalar witnesses alone do not demonstrate GPU execution behavior.

An example per-run record is provided in `result.schema.json`. Use the stable case ID and source hashes to associate it with the corpus. Custom methods can consume the original `.py` or `.ptx` paths from `manifest.json`; no custom-verifier adapter has been invented without knowing your API.

## Reporting bug recall and speed

Compute bug recall only over applicable cases with independently confirmed ground truth for the chosen contract. Report false positives on correct controls and coverage on all attempted cases. Keep the source-derived D1088 proof, published race reports, and adversarial testing observations distinguishable.

A training-speed claim additionally needs measurement inside the generation/training loop: verifier cost per candidate, rejected-candidate rate, unsupported fraction, GPU execution time avoided, and total training wall time. Faster isolated solving alone does not establish faster training.
