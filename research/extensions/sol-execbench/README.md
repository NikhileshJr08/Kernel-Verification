# SOL-ExecBench source collection

The collection maps the article's eight reseeding names and seven updated-benchmark names to their full Cursor submission directories. It also includes the embedding-gradient case and three illustrative specialization cases. See [INDEX.md](INDEX.md).

Each folder preserves `solution.json`, its embedded canonical sources under `candidate/`, the separately published `src/` tree under `published_src/`, and evaluation-only extracts from Cursor's traces. The retained source entrypoint and compilation options are in `solution.json`. The source names in every retained evaluation agree with that solution's name, and all retained evaluations have `status=PASSED`.

Cursor's original environments are recorded in the evaluation data, including B200 hardware and torch/Triton/CUDA versions. Later failure labels are author-reported observations from doubleAI, not local reruns. The exact failed workload IDs and doubleAI harness commit are not supplied by the article.

## Embedding mapping

The training example matches **L1/087_embedding_with_initial_layernorm_backward**. The submission has V=65,536 and H=4,096 and performs BF16 packed atomics into the embedding gradient; the reference hash corresponds to a program with FP32 scratch accumulation. The problem ID is inferred from those matching details and its unique operation name, rather than explicitly stated in the article. `embedding-mapping-evidence.json` records relevant source lines.

`python scripts/bf16_accumulation_witness.py` independently shows the rounding mechanism on a reduced sum. This is not a rerun of the transformer experiment or the full compiled kernel.

## References and workloads

NVIDIA's separate evaluation license excludes redistribution and training use. Consequently those files are not included. The downloader supports the last dataset revision preceding Cursor's April publication (`cursor-era`, March 20) and the April 15 update (`current`, still the current dataset at acquisition). The reference hashes and whether those versions differ are recorded per case. Exact per-workload changes can be inspected locally after downloading both.

```bash
python -m pip install pyarrow
python scripts/fetch_nvidia_references.py --case 087_embedding_with_initial_layernorm_backward --revision current --accept-nvidia-evaluation-license
```

The downloader creates reference, definition, workloads and candidate solution files in ignored `local-data/`. It verifies both the parquet and reference hashes.

For replay, install the SOL-ExecBench harness at the exact `sol_harness` commit in `extensions/upstream-lock.json`, following its own environment setup. The wrapper below uses its CLI configuration interface to set the seed:

```bash
python scripts/replay_sol_case.py local-data/sol-execbench/<dataset-revision>/<subset>/<problem> --seed 200
python scripts/replay_sol_case.py local-data/sol-execbench/<dataset-revision>/<subset>/<problem> --seed 1337
```

Repeat with 13337 and 133337 for the other reported reseeds. `--print-command` constructs the command without launching the kernel. The acquired harness is pinned for reproducibility; it is not claimed to be doubleAI's unpublished evaluation commit. A disagreement on another harness or driver must be investigated rather than assumed to reproduce the historical result.

Sources: [Cursor release](https://github.com/anysphere/kernel-optimization-results), [doubleAI report](https://www.doubleai.com/research/maxwell-approaches-speed-of-light-on-blackwell), [NVIDIA harness](https://github.com/NVIDIA/SOL-ExecBench), [NVIDIA reference dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench).
