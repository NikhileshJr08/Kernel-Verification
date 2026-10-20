# Kernel Correctness Corpus

**Start in [kernels/](kernels/README.md).** It contains the reference and candidate
implementations, organized by collection and case.

```text
kernels/
  gimlet/
    G01/
      reference.py
      candidate.py
      run.py
      README.md
      evidence/       original notes and metadata
  drkernel/           12 PyTorch / generated Triton pairs
  kernelbench-audit/  129 PyTorch / generated CUDA pairs
  sol-execbench/      19 generated source bundles
  volta/             8 CUDA reference/candidate pairs
  megatron/          2 historical faulty/fixed pairs
  flashattention/    7 historical pairs and 1 combined reproducer

research/            evidence, upstream snapshots, replay tools, and detailed docs
```

Each case uses **reference** for the baseline and **candidate** for the implementation
being compared. Historical faulty/fixed pairs are not necessarily LLM-generated.
SOL candidates need several files, so their code stays together in `candidate/`.
Missing implementations are described in the case README rather than replaced
with invented code. Gimlet G01 is explicitly labeled as a reconstruction; G02/G03
have only public reference tasks.

Source files in `kernels/` are relative symbolic links to the preserved originals
in `research/`. Open and edit them normally; both paths refer to the same file.
Keep both folders together. No source content was changed by the reorganization.
Some cases need their original build/test harness; complete source is not a
promise that a file runs by itself.

## Run the Gimlet attention example

```bash
python3 kernels/gimlet/G01/run.py
```

## Research tools

```bash
python3 research/scripts/check_integrity.py
python3 research/scripts/check_extensions.py
```

For the original replay commands and methodology, see
[research/README.md](research/README.md). Run its relative commands from inside
`research/` (`cd research` first). Published findings, local reproductions, and
unresolved leads remain distinct evidence states.

[Licensing and attribution](research/THIRD_PARTY_NOTICES.md).
To rebuild the browsing layout: `python3 research/scripts/build_kernel_view.py`.
