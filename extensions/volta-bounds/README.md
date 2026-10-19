# VOLTA bounds illustration: original mapping unresolved

The paper's example reads shared `A[48]` with 64 threads before guarding use with `tid < 48`. It is an illustration of an invalid read that can be tolerated at runtime. That does not identify a full standalone kernel in the public artifact.

The original artifact and verifier corpus were inspected, including Conv2D sources. No unique complete faulty-kernel/repair pair corresponding to this illustration was established. The shipped Conv2D artifact is therefore not relabeled as that missing candidate, and a newly written 48-element reproducer is not passed off as the original.

`case.json` preserves the unresolved status and source coordinates. This lead is not counted as an additional recovered original pair.

Source: [VOLTA paper](https://theory.stanford.edu/~aiken/publications/papers/oopsla26c.pdf), [original benchmark artifact](https://doi.org/10.5281/zenodo.21529230).
