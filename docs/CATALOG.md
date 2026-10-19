# Corpus catalog

| ID | Kernel | Evidence | Contract |
|---|---|---|---|
| [V01_reduction_5](../cases/V01_reduction_5/README.md) | Harris reduction Red-5 | formal_race_report | data_race_freedom |
| [V02_reduction_6](../cases/V02_reduction_6/README.md) | Harris reduction Red-6 | formal_race_report | data_race_freedom |
| [V03_reduction_7](../cases/V03_reduction_7/README.md) | Harris reduction Red-7 | formal_race_report | data_race_freedom |
| [V04_bucketpositions](../cases/V04_bucketpositions/README.md) | OpenMM BucketPositions | formal_race_report | data_race_freedom |
| [V05_computerange](../cases/V05_computerange/README.md) | OpenMM ComputeRange | formal_race_report | data_race_freedom |
| [V06_reducevalue](../cases/V06_reducevalue/README.md) | OpenMM ReduceValue | formal_race_report | data_race_freedom |
| [V07_layernorm](../cases/V07_layernorm/README.md) | Megatron-LM LayerNorm | formal_race_report | data_race_freedom |
| [V08_gradinput](../cases/V08_gradinput/README.md) | Megatron-LM GradInput | formal_race_report | data_race_freedom |
| [D0064_relu_then_minimum](../cases/D0064_relu_then_minimum/README.md) | ReLU then minimum | numerical_pass_then_adversarial_failure | ieee_special_values |
| [D0201_sigmoid_then_global_minimum](../cases/D0201_sigmoid_then_global_minimum/README.md) | Sigmoid then global minimum | numerical_pass_then_adversarial_failure | ieee_special_values |
| [D0278_mean_then_log_softmax](../cases/D0278_mean_then_log_softmax/README.md) | Mean then log-softmax | numerical_pass_then_adversarial_failure | floating_point_error_and_special_values |
| [D0416_layernorm_then_tanh](../cases/D0416_layernorm_then_tanh/README.md) | LayerNorm then tanh | numerical_pass_then_adversarial_failure | floating_point_error |
| [D0463_groupnorm_then_subtract_0_5](../cases/D0463_groupnorm_then_subtract_0_5/README.md) | GroupNorm then subtract 0.5 | numerical_pass_then_adversarial_failure | floating_point_error |
| [D0739_transposed_convolution_bias_lower_clamp](../cases/D0739_transposed_convolution_bias_lower_clamp/README.md) | Transposed convolution, bias, lower clamp | numerical_pass_then_adversarial_failure | ieee_special_values |
| [D0798_linear_then_cumulative_sum](../cases/D0798_linear_then_cumulative_sum/README.md) | Linear then cumulative sum | numerical_pass_then_adversarial_failure | floating_point_error_and_special_values |
| [D0883_triplet_margin_loss_then_mean](../cases/D0883_triplet_margin_loss_then_mean/README.md) | Triplet-margin loss then mean | numerical_pass_then_adversarial_failure | floating_point_error_and_special_values |
| [D0967_adaptive_average_pool_max_pool_division](../cases/D0967_adaptive_average_pool_max_pool_division/README.md) | Adaptive average pool, max pool, division | numerical_pass_then_adversarial_failure | determinism |
| [D1088_softmax_logsumexp_tanh](../cases/D1088_softmax_logsumexp_tanh/README.md) | Softmax, logsumexp, tanh | numerical_pass_then_adversarial_failure | real_arithmetic_equivalence |
| [D1100_relu_groupnorm_division](../cases/D1100_relu_groupnorm_division/README.md) | ReLU, GroupNorm, division | numerical_pass_then_adversarial_failure | floating_point_error_and_special_values |
| [D1891_row_minimum_then_sum](../cases/D1891_row_minimum_then_sum/README.md) | Row minimum then sum | numerical_pass_then_adversarial_failure | determinism_and_special_values |
