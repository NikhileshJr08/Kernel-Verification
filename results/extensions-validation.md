# Requested-source validation

Prepared October 19, 2026. No CUDA GPU execution is claimed.

| Check | Result |
|---|---|
| ABC/Sakana positional joins | 129/129 Correct=True, candidate bytes equal the pinned archive and ABC files |
| Extension source integrity | PASS: 162 records, 320 hashed source files checked |
| Parent/fix identity | PASS: two Megatron and seven FlashAttention parent IDs match commit metadata |
| Historical source/test syntax | PASS: recovered Python candidates/references/tests parse |
| SOL candidate source extraction | PASS: solution.json embedded sources equal extracted files |
| SOL published passes | PASS: retained records all PASSED and match the published solution names |
| NVIDIA dataset exclusion | PASS: no reference/workload/parquet bytes distributed in the SOL extension |
| Reduced BF16 rounding witness | PASS: 512 adds of one round to 256 when rounded after each BF16 add, versus 512 with FP32 accumulation and one BF16 rounding |
| GPU audit replay | NOT RUN: PyTorch/Triton/CUDA device unavailable |
| Exact 2022 FlashAttention compiler reproduction | NOT RUN: exact compiler builds for seven fixes remain unpinned |
| Training-loss experiment | NOT RUN |

The BF16 witness models the accumulation mechanism only. The 129 verified audit tests are available tests, not 129 newly observed failures. Gimlet originals and the full VOLTA bounds kernel remain unresolved. Historical published race reports apply to modified/specialized artifact programs, not freshly verified raw upstream originals.
