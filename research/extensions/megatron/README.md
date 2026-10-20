# Exact Megatron fixes and verification artifacts

| Kernel | Parent (faulty upstream source) | Fix |
|---|---|---|
| LayerNorm | `cdc614cfd4154d791b8b8d4756c2bcc1d8613a4d` | `4831071c900df5901570984c28b9f907892ff632` |
| GradInput | `1ec6b0e941b4e324602d3fd6a955f28cd334a383` | `f1295380cf06b0475165fbcbd89fbd88e9fee84c` |

Each folder contains the complete upstream `layer_norm_cuda_kernel.cu` and `layer_norm_cuda.cpp` at both revisions, GitHub commit metadata, and the actual patch. Both fixes add a block barrier at the end of a row iteration; the GradInput patch explicitly describes preventing buffer overwrite before reads finish.

`verification-artifact/` separately retains the specialized pre/post CUDA and PTX from VOLTA's original artifact, its published race/race-free outputs, and the canonical launch-declaration source `races.rs`. These specializations bake dimensions into code and are not asserted to equal the raw upstream source byte-for-byte.

The artifact also contains deliberate reproduction edits: its LayerNorm pre-version adds volatile shared-memory reads and removes a broadcast barrier, and its GradInput configuration changes block geometry. Thus, the published formal outputs establish findings on those modified artifact programs. They are not a fresh verification of the untouched raw upstream parent/fix files collected here.

The evaluated GradInput configuration uses block `(32,4,1)` before the fix and `(32,1,1)` after; LayerNorm uses one warp `(32,1,1)` on both sides. Preserve this change in any reproduction. The raw original GradInput source fix is a barrier change; the specialized artifact's geometry change is an additional evaluation detail, not a replacement description of that upstream patch.

The original core replay wrapper can analyze the artifacts:

```bash
python scripts/run_volta.py V07_layernorm
python scripts/run_volta.py V08_gradinput
python scripts/run_volta.py all-races
```

Only the pre-fix case is selected by the individual commands; `all-races` includes the fixed controls as well. Formal reports are upstream evidence, not fresh local verifier executions.
