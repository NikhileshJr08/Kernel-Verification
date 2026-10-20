# SOL-ExecBench problem mappings

This collection contains 19 recovered Cursor submissions. All include the original passing-evaluation extracts. Reference programs/workloads are available through the pinned local downloader, not embedded.

| Subset | Problem | Follow-up category | Published pass records |
|---|---|---|---|
| Quant | [011_fp8_moe_gate_routing](cases/Quant/011_fp8_moe_gate_routing/case.json) | reseeding_failure | 16 |
| Quant | [018_nvfp4_attention_output_projection_with_residual](cases/Quant/018_nvfp4_attention_output_projection_with_residual/case.json) | reseeding_failure | 16 |
| L1 | [024_vision_rotary_position_embedding_generation_backward](cases/L1/024_vision_rotary_position_embedding_generation_backward/case.json) | reseeding_failure | 16 |
| Quant | [030_nvfp4_fused_decoder_layer_pre_norm_attention](cases/Quant/030_nvfp4_fused_decoder_layer_pre_norm_attention/case.json) | reseeding_failure | 16 |
| L2 | [032_dual_stream_attention_with_conditional_cross_attention](cases/L2/032_dual_stream_attention_with_conditional_cross_attention/case.json) | reseeding_failure | 16 |
| Quant | [033_nvfp4_moe_routing_with_topk_selection](cases/Quant/033_nvfp4_moe_routing_with_topk_selection/case.json) | reseeding_failure | 16 |
| L2 | [038_audio_relative_position_attention](cases/L2/038_audio_relative_position_attention/case.json) | reseeding_failure | 14 |
| L2 | [049_group_limited_topk_routing](cases/L2/049_group_limited_topk_routing/case.json) | reseeding_failure | 16 |
| L2 | [004_fused_residual_rms_mlp](cases/L2/004_fused_residual_rms_mlp/case.json) | updated_benchmark_failure | 16 |
| Quant | [008_nvfp4_multimodal_embedding_projection](cases/Quant/008_nvfp4_multimodal_embedding_projection/case.json) | updated_benchmark_failure | 16 |
| L2 | [018_cu_seqlens_variable_length_vision_attention](cases/L2/018_cu_seqlens_variable_length_vision_attention/case.json) | updated_benchmark_failure | 16 |
| L2 | [031_flux_timestep_guidance_projection_embedding](cases/L2/031_flux_timestep_guidance_projection_embedding/case.json) | updated_benchmark_failure | 16 |
| L1 | [052_altup_hidden_state_collapse_with_magnitude_normalization](cases/L1/052_altup_hidden_state_collapse_with_magnitude_normalization/case.json) | updated_benchmark_failure | 16 |
| L2 | [052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection](cases/L2/052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection/case.json) | updated_benchmark_failure | 16 |
| L2 | [057_residual_coupling_flow_block](cases/L2/057_residual_coupling_flow_block/case.json) | updated_benchmark_failure | 16 |
| L1 | [087_embedding_with_initial_layernorm_backward](cases/L1/087_embedding_with_initial_layernorm_backward/case.json) | embedding_training_failure_mapping | 16 |
| L1 | [046_attention_softmax_with_softcapping_and_dropout](cases/L1/046_attention_softmax_with_softcapping_and_dropout/case.json) | illustrative_specialization_case | 16 |
| L1 | [016_rope_inverse_frequency_computation](cases/L1/016_rope_inverse_frequency_computation/case.json) | illustrative_specialization_case | 16 |
| FlashInfer-Bench | [001_fused_add_rmsnorm_h2048](cases/FlashInfer-Bench/001_fused_add_rmsnorm_h2048/case.json) | illustrative_specialization_case | 7 |

The first eight names are the article's reseeding set; the next seven are its updated-benchmark set. Seeds: default 200; reseeds 1337, 13337, 133337. Specific failed workload indices are not published. The training example maps to **L1/087_embedding_with_initial_layernorm_backward** through its unique name, V=65,536/H=4,096 constants, and BF16 atomic accumulation. The article does not itself print that numeric ID.

The three remaining submissions correspond to the article's softcap, theta-specialization, and batch-specialization illustrations; they are not added to the fifteen reseeding/update failures.
