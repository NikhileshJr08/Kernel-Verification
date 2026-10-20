# Requested-source expansion

The six requested tracks are indexed in [extensions/README.md](../extensions/README.md).

This release adds 129 exact ABC/Sakana source joins, 19 SOL candidate/evaluation bundles, seven genuine FlashAttention parent/fix pairs plus a compiler-pinned FP32 issue reproducer, and the two exact Megatron parent/fix source snapshots with specialized verification artifacts. It also records the unresolved Gimlet original candidates and VOLTA bounds-kernel mapping.

The additional manifest has 162 records, including tests and unresolved leads, not 162 confirmed bugs. It supplements the original 20-case corpus without changing that denominator.

NVIDIA reference/workload data is intentionally absent under its separate evaluation license. The downloader uses pinned revisions and checks hashes, and writes only to ignored local-data/.

Source integrity checks and a reduced BF16 rounding witness passed locally. GPU executions, historical compiler reproduction, and training-loss reproduction were not performed.
