# Gimlet original-candidate search

The public Gimlet KernelBench fork identifies plausible complete reference tasks:

- G02: Level 2 problem 2, `ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide`.
- G03: Level 1 problem 28, `HardSigmoid`.

The operation sequences agree with the article, but these public tasks are not asserted to be byte-identical copies of Gimlet's private verifier inputs. Their complete reference files, including constructors and input-generation functions, are included here.

G01 includes a [runnable PyTorch reconstruction](G01/README.md) of the article's adapted attention example, with separate reference and intentionally faulty candidate files. Run `python3 extensions/gimlet/G01/run.py` from the repository root. These are newly written reconstructions, not recovered original artifacts. The full original generated candidates, wrappers, launch choices, numerical test configurations, and verifier artifacts for all three examples were not identified in the article or the public repositories inspected. The public organization/repository search does not prove they are unavailable elsewhere.

No reconstructed candidate is labeled as recovered original code. The existing scalar reconstructions remain under `supplemental/` and are excluded from original-kernel counts.

Source: [Gimlet article](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels), [public Gimlet KernelBench fork](https://github.com/gimletlabs/KernelBench). Exact reference-file revision and hashes are in each case record.
