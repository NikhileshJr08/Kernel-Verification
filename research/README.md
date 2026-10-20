# Kernel Correctness Corpus

This folder preserves the detailed research collection. For the simplified
reference/candidate layout, open [../kernels/](../kernels/README.md).
Run the relative commands below from this directory (`cd research` from the
repository root).

**Requested-source expansion:** see [extensions/README.md](extensions/README.md) for the ABC/Sakana audit, exact Megatron fixes, doubleAI/SOL case mappings, historical FlashAttention revisions, and the remaining Gimlet/bounds gaps. The original twenty-case manifest below is preserved; the extension has a separate evidence index.

A source-backed research starter repository for comparing GPU-kernel verification approaches. It contains **20 original reference/candidate cases**, their provenance, published failure records, and replay tools. Retrieved October 19, 2026.

**The evidence is mixed, deliberately labeled:**

| Collection | Count | What is established |
|---|---:|---|
| VOLTA artifact | 8 | Published formal race reports, with original CUDA/PTX and canonical launch configurations. Five have pre/post-fix versions. Individual historical numerical-pass logs were not recovered. |
| DR.Kernel / Lethe audit | 12 | A recorded numerical-check pass and a later adversarial-contract failure, joined by corpus ID; original generated Triton and reference PyTorch are included. These are testing results, not upstream SMT proofs. |

This is **not 20 confirmed production incidents or 20 formally disproved ML equivalence pairs**. No selected entry has evidence establishing a production incident. Three OpenMM cases are non-ML concurrency controls; three Harris cases are integer reductions. The remaining fourteen cases are ML operations. These boundaries matter when reporting results.

Start with [the catalog](docs/CATALOG.md), [research findings](docs/RESEARCH_NOTES.md), and [evaluation protocol](docs/EVALUATION.md).

## Quick start

Python 3.10+ is sufficient for offline inspection:

```bash
python scripts/check_integrity.py
python scripts/scalar_witnesses.py
```

The latter demonstrates the source-derived D1088 padding defect and two supplemental scalar examples. To check the small real-arithmetic obligations with Z3:

```bash
python -m pip install -r requirements-symbolic.txt
python scripts/smt_smoke.py
```

These scalar checks do not verify GPU lowering or parallel execution.

## Replay original adversarial tests on a GPU

Use an NVIDIA CUDA environment with PyTorch and Triton installed. The published Lethe audit used a B200 with torch 2.12 / Triton 3.7; different toolchains may produce different failures. See `vendor/lethe/pyproject.toml` for upstream dependencies. No universal environment lock is claimed.

```bash
python scripts/replay_audit.py 1088 --device cuda:0 --out results/D1088-local.json
python scripts/replay_audit.py 416 --device cuda:0 --out results/D0416-local.json
```

The wrapper runs the vendored upstream audit in a fresh subprocess, sets a timeout, records the GPU/toolchain, and refuses CPU fallback. It **does not** rerun the original KernelGYM or KernelBench numerical-pass harness; those passes are supplied as upstream records. The wrapper's exit code distinguishes completed auditing from an execution failure; inspect gate verdicts to determine which contracts failed.

Candidate code is preserved verbatim. Do not repair it before measuring bug detection. Audit findings and inferred causes are distinct: the audit supplies observations; case READMEs identify source-level explanations where supported.

## Replay VOLTA on a CPU

Rust/Cargo is required. No CUDA device or nvcc is needed to symbolically analyze the supplied PTX. The first build may need network access for Cargo dependencies.

```bash
python scripts/run_volta.py V07_layernorm --print-command
python scripts/run_volta.py V07_layernorm
python scripts/run_volta.py all-races
python scripts/run_volta.py all-reductions
```

`generate` performs symbolic execution and detects races; no equivalence solve is needed for a race rejection. The full original benchmark corpus is included under `vendor/volta-benchmarks`, providing additional correct equivalence pairs for controls. It is not counted as additional negative cases.

**Use PTX as the canonical artifact.** The Zenodo racy reductions use an eight-thread specialization; the current Rust benchmark configs expect a later 128-thread recompilation absent from that archive. Our wrapper analyzes those three original files directly at eight threads. Published 128-thread timings are not timings for these supplied eight-thread files. Other launch configurations and array declarations live in `vendor/volta/crates/volta_bench/src/benchmarks/`.

## Layout

```text
cases/<id>/       original reference/candidate, metadata, evidence, explanation
manifest.json     machine-readable case index and reference/candidate hashes
docs/             catalog, research findings, comparison protocol
scripts/          integrity, scalar/SMT checks, GPU audit and VOLTA replay
supplemental/     Gimlet scalar reconstructions, excluded from the 20-case count
evidence/         acquisition pins and artifact checksums
vendor/           pinned VOLTA, its benchmark artifact, and Lethe sources
licenses/         upstream attribution and license notices
results/          local validation output; upstream results remain in vendor/
```

The dataset was pinned to `cba0ef0`; full acquired-file SHA-256 values and verifier commit IDs are in `evidence/upstream-lock.json`. Each generated case retains the full source trajectory. All twelve selected trajectories have final round = best round = 5, avoiding ambiguity between extracting the final and the best implementation. Published audit rows identify candidates by dataset ID rather than per-candidate source hash; that remaining provenance limitation is retained in metadata.

## Local verification status

See `results/local-validation.md`. Code integrity, evidence joins, scalar witnesses, and three small Z3 obligations were checked here. GPU execution and Rust/VOLTA reruns were unavailable in the preparation environment. Published reports are kept separate from locally generated results.

New curation and wrapper code is MIT licensed. Upstream files retain their licenses; see [third-party attribution](THIRD_PARTY_NOTICES.md).
