# Historical FlashAttention fixes

Seven entries contain complete parent/fix implementations, tests and patches. A separate issue reproducer has an exact reported compiler version.

| Fix / issue | Change | Faulty parent | Compiler pin |
|---|---|---|---|
| [470010f59b00](F_470010f59b00/case.json) | Fix race condition for Triton bwd for headdim 48 and 96 | aacc10fbab8c | unresolved |
| [62025e1aff6d](F_62025e1aff6d/case.json) | Fix more race condition in Triton bwd when there's bias | ff78ea4123a2 | unresolved |
| [731f154de328](F_731f154de328/case.json) | Fix race conditions in the Triton bwd for headdim=64 | 9b0bc978729d | unresolved |
| [7479757191c0](F_7479757191c0/case.json) | Fix pipelining bug in Triton bwd with bias_type=matrix | 557781933dbc | unresolved |
| [9b0bc978729d](F_9b0bc978729d/case.json) | Fix race condition in Triton fwd | 215930bce365 | unresolved |
| [d11341fd1a69](F_d11341fd1a69/case.json) | Fix Triton fwd to support seqlen not multiples of 128 | b0c0db81f64c | unresolved |
| [ff78ea4123a2](F_ff78ea4123a2/case.json) | Fix race condition in Triton bwd when there's bias | 86862cfd7bab | unresolved |
| [Triton issue #674](T674_fp32_issue_reproducer/case.json) | FP16/BF16 pass, FP32 fails | issue attach | 2.0.0.dev20220830 |

The later 2023 source note names Triton `2.0.0.dev20221202`. That note is retained in `later-version-note.py` and is **not** asserted to pin the compiler that triggered each 2022 bug. Historical code uses APIs removed from modern Triton; installing the latest compiler changes the experiment.

Full source checkouts can be obtained from the recorded parent/fix commit IDs. Use the corresponding `tests/test_flash_attn.py`, not a test imported from a modern checkout. The supplied source/test snapshots preserve the original contexts; no GPU results are claimed.
