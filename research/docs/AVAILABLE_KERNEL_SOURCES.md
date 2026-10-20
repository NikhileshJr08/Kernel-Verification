# Available kernel source index

All indexed cases with a local reference, candidate, or reproducer source. Source availability does not establish a reproduced bug or a working build environment. VOLTA, Megatron, and FlashAttention sources are not generally LLM-generated. Vendored auxiliary benchmark/control programs are outside this case index.

## Core: VOLTA (8 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| V01_reduction_5 | [reference.ptx](../cases/V01_reduction_5/reference.ptx)<br>[reference.cu](../cases/V01_reduction_5/reference.cu) | [candidate.ptx](../cases/V01_reduction_5/candidate.ptx)<br>[candidate.cu](../cases/V01_reduction_5/candidate.cu) |
| V02_reduction_6 | [reference.ptx](../cases/V02_reduction_6/reference.ptx)<br>[reference.cu](../cases/V02_reduction_6/reference.cu) | [candidate.ptx](../cases/V02_reduction_6/candidate.ptx)<br>[candidate.cu](../cases/V02_reduction_6/candidate.cu) |
| V03_reduction_7 | [reference.ptx](../cases/V03_reduction_7/reference.ptx)<br>[reference.cu](../cases/V03_reduction_7/reference.cu) | [candidate.ptx](../cases/V03_reduction_7/candidate.ptx)<br>[candidate.cu](../cases/V03_reduction_7/candidate.cu) |
| V04_bucketpositions | [reference.ptx](../cases/V04_bucketpositions/reference.ptx)<br>[reference.cu](../cases/V04_bucketpositions/reference.cu) | [candidate.ptx](../cases/V04_bucketpositions/candidate.ptx)<br>[candidate.cu](../cases/V04_bucketpositions/candidate.cu) |
| V05_computerange | [reference.ptx](../cases/V05_computerange/reference.ptx)<br>[reference.cu](../cases/V05_computerange/reference.cu) | [candidate.ptx](../cases/V05_computerange/candidate.ptx)<br>[candidate.cu](../cases/V05_computerange/candidate.cu) |
| V06_reducevalue | [reference.ptx](../cases/V06_reducevalue/reference.ptx)<br>[reference.cu](../cases/V06_reducevalue/reference.cu) | [candidate.ptx](../cases/V06_reducevalue/candidate.ptx)<br>[candidate.cu](../cases/V06_reducevalue/candidate.cu) |
| V07_layernorm | [reference.ptx](../cases/V07_layernorm/reference.ptx)<br>[reference.cu](../cases/V07_layernorm/reference.cu) | [candidate.ptx](../cases/V07_layernorm/candidate.ptx)<br>[candidate.cu](../cases/V07_layernorm/candidate.cu) |
| V08_gradinput | [reference.ptx](../cases/V08_gradinput/reference.ptx)<br>[reference.cu](../cases/V08_gradinput/reference.cu) | [candidate.ptx](../cases/V08_gradinput/candidate.ptx)<br>[candidate.cu](../cases/V08_gradinput/candidate.cu) |

## Core: DR.Kernel / Lethe (12 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| D0064_relu_then_minimum | [reference.py](../cases/D0064_relu_then_minimum/reference.py) | [candidate.py](../cases/D0064_relu_then_minimum/candidate.py) |
| D0201_sigmoid_then_global_minimum | [reference.py](../cases/D0201_sigmoid_then_global_minimum/reference.py) | [candidate.py](../cases/D0201_sigmoid_then_global_minimum/candidate.py) |
| D0278_mean_then_log_softmax | [reference.py](../cases/D0278_mean_then_log_softmax/reference.py) | [candidate.py](../cases/D0278_mean_then_log_softmax/candidate.py) |
| D0416_layernorm_then_tanh | [reference.py](../cases/D0416_layernorm_then_tanh/reference.py) | [candidate.py](../cases/D0416_layernorm_then_tanh/candidate.py) |
| D0463_groupnorm_then_subtract_0_5 | [reference.py](../cases/D0463_groupnorm_then_subtract_0_5/reference.py) | [candidate.py](../cases/D0463_groupnorm_then_subtract_0_5/candidate.py) |
| D0739_transposed_convolution_bias_lower_clamp | [reference.py](../cases/D0739_transposed_convolution_bias_lower_clamp/reference.py) | [candidate.py](../cases/D0739_transposed_convolution_bias_lower_clamp/candidate.py) |
| D0798_linear_then_cumulative_sum | [reference.py](../cases/D0798_linear_then_cumulative_sum/reference.py) | [candidate.py](../cases/D0798_linear_then_cumulative_sum/candidate.py) |
| D0883_triplet_margin_loss_then_mean | [reference.py](../cases/D0883_triplet_margin_loss_then_mean/reference.py) | [candidate.py](../cases/D0883_triplet_margin_loss_then_mean/candidate.py) |
| D0967_adaptive_average_pool_max_pool_division | [reference.py](../cases/D0967_adaptive_average_pool_max_pool_division/reference.py) | [candidate.py](../cases/D0967_adaptive_average_pool_max_pool_division/candidate.py) |
| D1088_softmax_logsumexp_tanh | [reference.py](../cases/D1088_softmax_logsumexp_tanh/reference.py) | [candidate.py](../cases/D1088_softmax_logsumexp_tanh/candidate.py) |
| D1100_relu_groupnorm_division | [reference.py](../cases/D1100_relu_groupnorm_division/reference.py) | [candidate.py](../cases/D1100_relu_groupnorm_division/candidate.py) |
| D1891_row_minimum_then_sum | [reference.py](../cases/D1891_row_minimum_then_sum/reference.py) | [candidate.py](../cases/D1891_row_minimum_then_sum/candidate.py) |

## kernelbench-audit (129 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| A12056_100_HingeLoss | [reference.py](../extensions/kernelbench-audit/cases/A12056_100_HingeLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A12056_100_HingeLoss/candidate.cu) |
| A01111_10_3D_tensor_matrix_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A01111_10_3D_tensor_matrix_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A01111_10_3D_tensor_matrix_multiplication/candidate.cu) |
| A01142_10_3D_tensor_matrix_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A01142_10_3D_tensor_matrix_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A01142_10_3D_tensor_matrix_multiplication/candidate.cu) |
| A01330_12_Matmul_with_diagonal_matrices_ | [reference.py](../extensions/kernelbench-audit/cases/A01330_12_Matmul_with_diagonal_matrices_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A01330_12_Matmul_with_diagonal_matrices_/candidate.cu) |
| A01404_12_Matmul_with_diagonal_matrices_ | [reference.py](../extensions/kernelbench-audit/cases/A01404_12_Matmul_with_diagonal_matrices_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A01404_12_Matmul_with_diagonal_matrices_/candidate.cu) |
| A01601_14_Matmul_for_upper_triangular_matrices | [reference.py](../extensions/kernelbench-audit/cases/A01601_14_Matmul_for_upper_triangular_matrices/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A01601_14_Matmul_for_upper_triangular_matrices/candidate.cu) |
| A02027_15_Matmul_for_lower_triangular_matrices | [reference.py](../extensions/kernelbench-audit/cases/A02027_15_Matmul_for_lower_triangular_matrices/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02027_15_Matmul_for_lower_triangular_matrices/candidate.cu) |
| A02185_16_Matmul_with_transposed_A | [reference.py](../extensions/kernelbench-audit/cases/A02185_16_Matmul_with_transposed_A/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02185_16_Matmul_with_transposed_A/candidate.cu) |
| A02203_16_Matmul_with_transposed_A | [reference.py](../extensions/kernelbench-audit/cases/A02203_16_Matmul_with_transposed_A/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02203_16_Matmul_with_transposed_A/candidate.cu) |
| A02350_17_Matmul_with_transposed_B | [reference.py](../extensions/kernelbench-audit/cases/A02350_17_Matmul_with_transposed_B/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02350_17_Matmul_with_transposed_B/candidate.cu) |
| A02399_17_Matmul_with_transposed_B | [reference.py](../extensions/kernelbench-audit/cases/A02399_17_Matmul_with_transposed_B/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02399_17_Matmul_with_transposed_B/candidate.cu) |
| A02408_17_Matmul_with_transposed_B | [reference.py](../extensions/kernelbench-audit/cases/A02408_17_Matmul_with_transposed_B/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02408_17_Matmul_with_transposed_B/candidate.cu) |
| A02452_18_Matmul_with_transposed_both | [reference.py](../extensions/kernelbench-audit/cases/A02452_18_Matmul_with_transposed_both/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02452_18_Matmul_with_transposed_both/candidate.cu) |
| A02472_18_Matmul_with_transposed_both | [reference.py](../extensions/kernelbench-audit/cases/A02472_18_Matmul_with_transposed_both/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02472_18_Matmul_with_transposed_both/candidate.cu) |
| A02506_18_Matmul_with_transposed_both | [reference.py](../extensions/kernelbench-audit/cases/A02506_18_Matmul_with_transposed_both/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02506_18_Matmul_with_transposed_both/candidate.cu) |
| A02569_19_ReLU | [reference.py](../extensions/kernelbench-audit/cases/A02569_19_ReLU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02569_19_ReLU/candidate.cu) |
| A02645_19_ReLU | [reference.py](../extensions/kernelbench-audit/cases/A02645_19_ReLU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02645_19_ReLU/candidate.cu) |
| A02671_20_LeakyReLU | [reference.py](../extensions/kernelbench-audit/cases/A02671_20_LeakyReLU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02671_20_LeakyReLU/candidate.cu) |
| A02681_20_LeakyReLU | [reference.py](../extensions/kernelbench-audit/cases/A02681_20_LeakyReLU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02681_20_LeakyReLU/candidate.cu) |
| A02710_20_LeakyReLU | [reference.py](../extensions/kernelbench-audit/cases/A02710_20_LeakyReLU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02710_20_LeakyReLU/candidate.cu) |
| A02834_21_Sigmoid | [reference.py](../extensions/kernelbench-audit/cases/A02834_21_Sigmoid/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02834_21_Sigmoid/candidate.cu) |
| A02950_22_Tanh | [reference.py](../extensions/kernelbench-audit/cases/A02950_22_Tanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A02950_22_Tanh/candidate.cu) |
| A03001_22_Tanh | [reference.py](../extensions/kernelbench-audit/cases/A03001_22_Tanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03001_22_Tanh/candidate.cu) |
| A03011_22_Tanh | [reference.py](../extensions/kernelbench-audit/cases/A03011_22_Tanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03011_22_Tanh/candidate.cu) |
| A03040_23_Softmax | [reference.py](../extensions/kernelbench-audit/cases/A03040_23_Softmax/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03040_23_Softmax/candidate.cu) |
| A03057_23_Softmax | [reference.py](../extensions/kernelbench-audit/cases/A03057_23_Softmax/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03057_23_Softmax/candidate.cu) |
| A03058_23_Softmax | [reference.py](../extensions/kernelbench-audit/cases/A03058_23_Softmax/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03058_23_Softmax/candidate.cu) |
| A03153_24_LogSoftmax | [reference.py](../extensions/kernelbench-audit/cases/A03153_24_LogSoftmax/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03153_24_LogSoftmax/candidate.cu) |
| A03249_24_LogSoftmax | [reference.py](../extensions/kernelbench-audit/cases/A03249_24_LogSoftmax/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03249_24_LogSoftmax/candidate.cu) |
| A03366_25_Swish | [reference.py](../extensions/kernelbench-audit/cases/A03366_25_Swish/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03366_25_Swish/candidate.cu) |
| A03370_25_Swish | [reference.py](../extensions/kernelbench-audit/cases/A03370_25_Swish/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03370_25_Swish/candidate.cu) |
| A03389_26_GELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03389_26_GELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03389_26_GELU_/candidate.cu) |
| A03446_26_GELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03446_26_GELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03446_26_GELU_/candidate.cu) |
| A03492_26_GELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03492_26_GELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03492_26_GELU_/candidate.cu) |
| A03514_27_SELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03514_27_SELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03514_27_SELU_/candidate.cu) |
| A03532_27_SELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03532_27_SELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03532_27_SELU_/candidate.cu) |
| A03604_27_SELU_ | [reference.py](../extensions/kernelbench-audit/cases/A03604_27_SELU_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03604_27_SELU_/candidate.cu) |
| A03647_28_HardSigmoid | [reference.py](../extensions/kernelbench-audit/cases/A03647_28_HardSigmoid/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03647_28_HardSigmoid/candidate.cu) |
| A03711_28_HardSigmoid | [reference.py](../extensions/kernelbench-audit/cases/A03711_28_HardSigmoid/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03711_28_HardSigmoid/candidate.cu) |
| A03778_29_Softplus | [reference.py](../extensions/kernelbench-audit/cases/A03778_29_Softplus/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03778_29_Softplus/candidate.cu) |
| A03847_29_Softplus | [reference.py](../extensions/kernelbench-audit/cases/A03847_29_Softplus/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03847_29_Softplus/candidate.cu) |
| A00171_2_Standard_matrix_multiplication_ | [reference.py](../extensions/kernelbench-audit/cases/A00171_2_Standard_matrix_multiplication_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00171_2_Standard_matrix_multiplication_/candidate.cu) |
| A03917_30_Softsign | [reference.py](../extensions/kernelbench-audit/cases/A03917_30_Softsign/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03917_30_Softsign/candidate.cu) |
| A03920_30_Softsign | [reference.py](../extensions/kernelbench-audit/cases/A03920_30_Softsign/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03920_30_Softsign/candidate.cu) |
| A03928_30_Softsign | [reference.py](../extensions/kernelbench-audit/cases/A03928_30_Softsign/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A03928_30_Softsign/candidate.cu) |
| A04049_31_ELU | [reference.py](../extensions/kernelbench-audit/cases/A04049_31_ELU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04049_31_ELU/candidate.cu) |
| A04090_31_ELU | [reference.py](../extensions/kernelbench-audit/cases/A04090_31_ELU/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04090_31_ELU/candidate.cu) |
| A04137_32_HardTanh | [reference.py](../extensions/kernelbench-audit/cases/A04137_32_HardTanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04137_32_HardTanh/candidate.cu) |
| A04149_32_HardTanh | [reference.py](../extensions/kernelbench-audit/cases/A04149_32_HardTanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04149_32_HardTanh/candidate.cu) |
| A04224_32_HardTanh | [reference.py](../extensions/kernelbench-audit/cases/A04224_32_HardTanh/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04224_32_HardTanh/candidate.cu) |
| A04259_33_BatchNorm | [reference.py](../extensions/kernelbench-audit/cases/A04259_33_BatchNorm/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04259_33_BatchNorm/candidate.cu) |
| A04331_33_BatchNorm | [reference.py](../extensions/kernelbench-audit/cases/A04331_33_BatchNorm/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04331_33_BatchNorm/candidate.cu) |
| A04473_35_GroupNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04473_35_GroupNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04473_35_GroupNorm_/candidate.cu) |
| A04476_35_GroupNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04476_35_GroupNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04476_35_GroupNorm_/candidate.cu) |
| A04586_36_RMSNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04586_36_RMSNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04586_36_RMSNorm_/candidate.cu) |
| A04670_36_RMSNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04670_36_RMSNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04670_36_RMSNorm_/candidate.cu) |
| A04714_37_FrobeniusNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04714_37_FrobeniusNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04714_37_FrobeniusNorm_/candidate.cu) |
| A04782_37_FrobeniusNorm_ | [reference.py](../extensions/kernelbench-audit/cases/A04782_37_FrobeniusNorm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04782_37_FrobeniusNorm_/candidate.cu) |
| A04829_38_L1Norm_ | [reference.py](../extensions/kernelbench-audit/cases/A04829_38_L1Norm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04829_38_L1Norm_/candidate.cu) |
| A04972_39_L2Norm_ | [reference.py](../extensions/kernelbench-audit/cases/A04972_39_L2Norm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A04972_39_L2Norm_/candidate.cu) |
| A05024_39_L2Norm_ | [reference.py](../extensions/kernelbench-audit/cases/A05024_39_L2Norm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05024_39_L2Norm_/candidate.cu) |
| A05043_39_L2Norm_ | [reference.py](../extensions/kernelbench-audit/cases/A05043_39_L2Norm_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05043_39_L2Norm_/candidate.cu) |
| A00234_3_Batched_matrix_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A00234_3_Batched_matrix_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00234_3_Batched_matrix_multiplication/candidate.cu) |
| A00256_3_Batched_matrix_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A00256_3_Batched_matrix_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00256_3_Batched_matrix_multiplication/candidate.cu) |
| A00265_3_Batched_matrix_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A00265_3_Batched_matrix_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00265_3_Batched_matrix_multiplication/candidate.cu) |
| A05073_40_LayerNorm | [reference.py](../extensions/kernelbench-audit/cases/A05073_40_LayerNorm/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05073_40_LayerNorm/candidate.cu) |
| A05089_40_LayerNorm | [reference.py](../extensions/kernelbench-audit/cases/A05089_40_LayerNorm/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05089_40_LayerNorm/candidate.cu) |
| A05148_40_LayerNorm | [reference.py](../extensions/kernelbench-audit/cases/A05148_40_LayerNorm/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05148_40_LayerNorm/candidate.cu) |
| A05193_41_Max_Pooling_1D | [reference.py](../extensions/kernelbench-audit/cases/A05193_41_Max_Pooling_1D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05193_41_Max_Pooling_1D/candidate.cu) |
| A05261_41_Max_Pooling_1D | [reference.py](../extensions/kernelbench-audit/cases/A05261_41_Max_Pooling_1D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05261_41_Max_Pooling_1D/candidate.cu) |
| A05728_42_Max_Pooling_2D | [reference.py](../extensions/kernelbench-audit/cases/A05728_42_Max_Pooling_2D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05728_42_Max_Pooling_2D/candidate.cu) |
| A05765_42_Max_Pooling_2D | [reference.py](../extensions/kernelbench-audit/cases/A05765_42_Max_Pooling_2D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05765_42_Max_Pooling_2D/candidate.cu) |
| A05898_43_Max_Pooling_3D | [reference.py](../extensions/kernelbench-audit/cases/A05898_43_Max_Pooling_3D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05898_43_Max_Pooling_3D/candidate.cu) |
| A05917_43_Max_Pooling_3D | [reference.py](../extensions/kernelbench-audit/cases/A05917_43_Max_Pooling_3D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05917_43_Max_Pooling_3D/candidate.cu) |
| A05960_44_Average_Pooling_1D | [reference.py](../extensions/kernelbench-audit/cases/A05960_44_Average_Pooling_1D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05960_44_Average_Pooling_1D/candidate.cu) |
| A05976_44_Average_Pooling_1D | [reference.py](../extensions/kernelbench-audit/cases/A05976_44_Average_Pooling_1D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A05976_44_Average_Pooling_1D/candidate.cu) |
| A06070_45_Average_Pooling_2D | [reference.py](../extensions/kernelbench-audit/cases/A06070_45_Average_Pooling_2D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06070_45_Average_Pooling_2D/candidate.cu) |
| A06099_45_Average_Pooling_2D | [reference.py](../extensions/kernelbench-audit/cases/A06099_45_Average_Pooling_2D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06099_45_Average_Pooling_2D/candidate.cu) |
| A06243_46_Average_Pooling_3D | [reference.py](../extensions/kernelbench-audit/cases/A06243_46_Average_Pooling_3D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06243_46_Average_Pooling_3D/candidate.cu) |
| A06255_46_Average_Pooling_3D | [reference.py](../extensions/kernelbench-audit/cases/A06255_46_Average_Pooling_3D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06255_46_Average_Pooling_3D/candidate.cu) |
| A06329_47_Sum_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06329_47_Sum_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06329_47_Sum_reduction_over_a_dimension/candidate.cu) |
| A06331_47_Sum_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06331_47_Sum_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06331_47_Sum_reduction_over_a_dimension/candidate.cu) |
| A06418_47_Sum_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06418_47_Sum_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06418_47_Sum_reduction_over_a_dimension/candidate.cu) |
| A06456_48_Mean_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06456_48_Mean_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06456_48_Mean_reduction_over_a_dimension/candidate.cu) |
| A06560_49_Max_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06560_49_Max_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06560_49_Max_reduction_over_a_dimension/candidate.cu) |
| A00457_4_Matrix_vector_multiplication_ | [reference.py](../extensions/kernelbench-audit/cases/A00457_4_Matrix_vector_multiplication_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00457_4_Matrix_vector_multiplication_/candidate.cu) |
| A06701_50_Product_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06701_50_Product_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06701_50_Product_reduction_over_a_dimension/candidate.cu) |
| A06823_51_Argmax_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06823_51_Argmax_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06823_51_Argmax_over_a_dimension/candidate.cu) |
| A06921_52_Argmin_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A06921_52_Argmin_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A06921_52_Argmin_over_a_dimension/candidate.cu) |
| A07111_53_Min_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A07111_53_Min_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07111_53_Min_reduction_over_a_dimension/candidate.cu) |
| A07124_53_Min_reduction_over_a_dimension | [reference.py](../extensions/kernelbench-audit/cases/A07124_53_Min_reduction_over_a_dimension/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07124_53_Min_reduction_over_a_dimension/candidate.cu) |
| A07184_55_conv_standard_2D__asymmetric_input__square_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07184_55_conv_standard_2D__asymmetric_input__square_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07184_55_conv_standard_2D__asymmetric_input__square_kernel/candidate.cu) |
| A07256_55_conv_standard_2D__asymmetric_input__square_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07256_55_conv_standard_2D__asymmetric_input__square_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07256_55_conv_standard_2D__asymmetric_input__square_kernel/candidate.cu) |
| A07286_56_conv_standard_2D__asymmetric_input__asymmetric_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07286_56_conv_standard_2D__asymmetric_input__asymmetric_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07286_56_conv_standard_2D__asymmetric_input__asymmetric_kernel/candidate.cu) |
| A07451_57_conv_transposed_2D__square_input__square_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07451_57_conv_transposed_2D__square_input__square_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07451_57_conv_transposed_2D__square_input__square_kernel/candidate.cu) |
| A07546_58_conv_transposed_3D__asymmetric_input__asymmetric_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07546_58_conv_transposed_3D__asymmetric_input__asymmetric_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07546_58_conv_transposed_3D__asymmetric_input__asymmetric_kernel/candidate.cu) |
| A00552_5_Matrix_scalar_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A00552_5_Matrix_scalar_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00552_5_Matrix_scalar_multiplication/candidate.cu) |
| A00557_5_Matrix_scalar_multiplication | [reference.py](../extensions/kernelbench-audit/cases/A00557_5_Matrix_scalar_multiplication/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00557_5_Matrix_scalar_multiplication/candidate.cu) |
| A07888_63_conv_standard_2D__square_input__square_kernel | [reference.py](../extensions/kernelbench-audit/cases/A07888_63_conv_standard_2D__square_input__square_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A07888_63_conv_standard_2D__square_input__square_kernel/candidate.cu) |
| A08258_67_conv_standard_1D | [reference.py](../extensions/kernelbench-audit/cases/A08258_67_conv_standard_1D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08258_67_conv_standard_1D/candidate.cu) |
| A08440_69_conv_transposed_2D__asymmetric_input__asymmetric_kernel | [reference.py](../extensions/kernelbench-audit/cases/A08440_69_conv_transposed_2D__asymmetric_input__asymmetric_kernel/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08440_69_conv_transposed_2D__asymmetric_input__asymmetric_kernel/candidate.cu) |
| A00592_6_Matmul_with_large_K_dimension_ | [reference.py](../extensions/kernelbench-audit/cases/A00592_6_Matmul_with_large_K_dimension_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00592_6_Matmul_with_large_K_dimension_/candidate.cu) |
| A00609_6_Matmul_with_large_K_dimension_ | [reference.py](../extensions/kernelbench-audit/cases/A00609_6_Matmul_with_large_K_dimension_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00609_6_Matmul_with_large_K_dimension_/candidate.cu) |
| A08670_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | [reference.py](../extensions/kernelbench-audit/cases/A08670_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08670_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/candidate.cu) |
| A08676_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | [reference.py](../extensions/kernelbench-audit/cases/A08676_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08676_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/candidate.cu) |
| A08708_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | [reference.py](../extensions/kernelbench-audit/cases/A08708_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08708_72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_/candidate.cu) |
| A08993_76_conv_standard_1D_dilated_strided__ | [reference.py](../extensions/kernelbench-audit/cases/A08993_76_conv_standard_1D_dilated_strided__/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A08993_76_conv_standard_1D_dilated_strided__/candidate.cu) |
| A09106_78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__ | [reference.py](../extensions/kernelbench-audit/cases/A09106_78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A09106_78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__/candidate.cu) |
| A00710_7_Matmul_with_small_K_dimension_ | [reference.py](../extensions/kernelbench-audit/cases/A00710_7_Matmul_with_small_K_dimension_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00710_7_Matmul_with_small_K_dimension_/candidate.cu) |
| A00741_7_Matmul_with_small_K_dimension_ | [reference.py](../extensions/kernelbench-audit/cases/A00741_7_Matmul_with_small_K_dimension_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00741_7_Matmul_with_small_K_dimension_/candidate.cu) |
| A00818_7_Matmul_with_small_K_dimension_ | [reference.py](../extensions/kernelbench-audit/cases/A00818_7_Matmul_with_small_K_dimension_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00818_7_Matmul_with_small_K_dimension_/candidate.cu) |
| A10155_86_conv_depthwise_separable_2D | [reference.py](../extensions/kernelbench-audit/cases/A10155_86_conv_depthwise_separable_2D/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10155_86_conv_depthwise_separable_2D/candidate.cu) |
| A10321_88_MinGPTNewGelu | [reference.py](../extensions/kernelbench-audit/cases/A10321_88_MinGPTNewGelu/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10321_88_MinGPTNewGelu/candidate.cu) |
| A10429_89_cumsum | [reference.py](../extensions/kernelbench-audit/cases/A10429_89_cumsum/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10429_89_cumsum/candidate.cu) |
| A10463_89_cumsum | [reference.py](../extensions/kernelbench-audit/cases/A10463_89_cumsum/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10463_89_cumsum/candidate.cu) |
| A00886_8_Matmul_with_irregular_shapes_ | [reference.py](../extensions/kernelbench-audit/cases/A00886_8_Matmul_with_irregular_shapes_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00886_8_Matmul_with_irregular_shapes_/candidate.cu) |
| A10553_90_cumprod | [reference.py](../extensions/kernelbench-audit/cases/A10553_90_cumprod/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10553_90_cumprod/candidate.cu) |
| A10634_90_cumprod | [reference.py](../extensions/kernelbench-audit/cases/A10634_90_cumprod/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10634_90_cumprod/candidate.cu) |
| A10695_91_cumsum_reverse | [reference.py](../extensions/kernelbench-audit/cases/A10695_91_cumsum_reverse/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10695_91_cumsum_reverse/candidate.cu) |
| A10862_93_masked_cumsum | [reference.py](../extensions/kernelbench-audit/cases/A10862_93_masked_cumsum/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10862_93_masked_cumsum/candidate.cu) |
| A10961_94_MSELoss | [reference.py](../extensions/kernelbench-audit/cases/A10961_94_MSELoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10961_94_MSELoss/candidate.cu) |
| A10996_94_MSELoss | [reference.py](../extensions/kernelbench-audit/cases/A10996_94_MSELoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A10996_94_MSELoss/candidate.cu) |
| A11038_95_CrossEntropyLoss | [reference.py](../extensions/kernelbench-audit/cases/A11038_95_CrossEntropyLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A11038_95_CrossEntropyLoss/candidate.cu) |
| A11171_96_HuberLoss | [reference.py](../extensions/kernelbench-audit/cases/A11171_96_HuberLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A11171_96_HuberLoss/candidate.cu) |
| A11180_96_HuberLoss | [reference.py](../extensions/kernelbench-audit/cases/A11180_96_HuberLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A11180_96_HuberLoss/candidate.cu) |
| A11289_97_CosineSimilarityLoss | [reference.py](../extensions/kernelbench-audit/cases/A11289_97_CosineSimilarityLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A11289_97_CosineSimilarityLoss/candidate.cu) |
| A11997_99_TripletMarginLoss | [reference.py](../extensions/kernelbench-audit/cases/A11997_99_TripletMarginLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A11997_99_TripletMarginLoss/candidate.cu) |
| A12011_99_TripletMarginLoss | [reference.py](../extensions/kernelbench-audit/cases/A12011_99_TripletMarginLoss/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A12011_99_TripletMarginLoss/candidate.cu) |
| A00958_9_Tall_skinny_matrix_multiplication_ | [reference.py](../extensions/kernelbench-audit/cases/A00958_9_Tall_skinny_matrix_multiplication_/reference.py) | [candidate.cu](../extensions/kernelbench-audit/cases/A00958_9_Tall_skinny_matrix_multiplication_/candidate.cu) |

## megatron (2 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| M01_layernorm | [layer_norm_cuda_kernel.cu](../extensions/megatron/M01_layernorm/post/layer_norm_cuda_kernel.cu) | [layer_norm_cuda_kernel.cu](../extensions/megatron/M01_layernorm/pre/layer_norm_cuda_kernel.cu) |
| M02_gradinput | [layer_norm_cuda_kernel.cu](../extensions/megatron/M02_gradinput/post/layer_norm_cuda_kernel.cu) | [layer_norm_cuda_kernel.cu](../extensions/megatron/M02_gradinput/pre/layer_norm_cuda_kernel.cu) |

## flashattention (8 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| F_470010f59b00 | [flash_attn_triton.py](../extensions/flashattention/F_470010f59b00/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_470010f59b00/pre/flash_attn/flash_attn_triton.py) |
| F_62025e1aff6d | [flash_attn_triton.py](../extensions/flashattention/F_62025e1aff6d/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_62025e1aff6d/pre/flash_attn/flash_attn_triton.py) |
| F_731f154de328 | [flash_attn_triton.py](../extensions/flashattention/F_731f154de328/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_731f154de328/pre/flash_attn/flash_attn_triton.py) |
| F_7479757191c0 | [flash_attn_triton.py](../extensions/flashattention/F_7479757191c0/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_7479757191c0/pre/flash_attn/flash_attn_triton.py) |
| F_9b0bc978729d | [flash_attn_triton.py](../extensions/flashattention/F_9b0bc978729d/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_9b0bc978729d/pre/flash_attn/flash_attn_triton.py) |
| F_d11341fd1a69 | [flash_attn_triton.py](../extensions/flashattention/F_d11341fd1a69/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_d11341fd1a69/pre/flash_attn/flash_attn_triton.py) |
| F_ff78ea4123a2 | [flash_attn_triton.py](../extensions/flashattention/F_ff78ea4123a2/post/flash_attn/flash_attn_triton.py) | [flash_attn_triton.py](../extensions/flashattention/F_ff78ea4123a2/pre/flash_attn/flash_attn_triton.py) |
| T674_fp32_issue_reproducer | Not included | [candidate_and_reference_test.py](../extensions/flashattention/T674_fp32_issue_reproducer/candidate_and_reference_test.py) |

## sol-execbench (19 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| S_Quant_011_fp8_moe_gate_routing | Not included | [kernel.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel.py)<br>[kernel_mid_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_mid_family.py)<br>[kernel_mid_plain_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_mid_plain_family.py)<br>[kernel_rawptr_all_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_rawptr_all_family.py)<br>[kernel_qdq12_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_qdq12_family.py)<br>[kernel_qdq_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_qdq_family.py)<br>[kernel_v20_family.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/kernel_v20_family.py)<br>[routing_workspace_common.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/routing_workspace_common.py)<br>[blockwise_gemm_impl.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/blockwise_gemm_impl.py)<br>[fp8_blockwise_common_dynamic.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/fp8_blockwise_common_dynamic.py)<br>[fp8_blockwise_common_rawptr.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/fp8_blockwise_common_rawptr.py)<br>[fp8_blockwise_common.py](../extensions/sol-execbench/cases/Quant/011_fp8_moe_gate_routing/candidate/fp8_blockwise_common.py) |
| S_Quant_018_nvfp4_attention_output_projection_with_residual | Not included | [kernel.cu](../extensions/sol-execbench/cases/Quant/018_nvfp4_attention_output_projection_with_residual/candidate/kernel.cu) |
| S_L1_024_vision_rotary_position_embedding_generation_backward | Not included | [kernel.cu](../extensions/sol-execbench/cases/L1/024_vision_rotary_position_embedding_generation_backward/candidate/kernel.cu) |
| S_Quant_030_nvfp4_fused_decoder_layer_pre_norm_attention | Not included | [kernel.py](../extensions/sol-execbench/cases/Quant/030_nvfp4_fused_decoder_layer_pre_norm_attention/candidate/kernel.py)<br>[common_nvfp4_gemm.py](../extensions/sol-execbench/cases/Quant/030_nvfp4_fused_decoder_layer_pre_norm_attention/candidate/common_nvfp4_gemm.py) |
| S_L2_032_dual_stream_attention_with_conditional_cross_attention | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/032_dual_stream_attention_with_conditional_cross_attention/candidate/kernel.py)<br>[rope_half_stacked_qk_rows8x32.py](../extensions/sol-execbench/cases/L2/032_dual_stream_attention_with_conditional_cross_attention/candidate/rope_half_stacked_qk_rows8x32.py) |
| S_Quant_033_nvfp4_moe_routing_with_topk_selection | Not included | [kernel.cu](../extensions/sol-execbench/cases/Quant/033_nvfp4_moe_routing_with_topk_selection/candidate/kernel.cu) |
| S_L2_038_audio_relative_position_attention | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/038_audio_relative_position_attention/candidate/kernel.py)<br>[exact_m_dense_gemm.py](../extensions/sol-execbench/cases/L2/038_audio_relative_position_attention/candidate/exact_m_dense_gemm.py)<br>[dense_gemm_kernel.py](../extensions/sol-execbench/cases/L2/038_audio_relative_position_attention/candidate/dense_gemm_kernel.py) |
| S_L2_049_group_limited_topk_routing | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/049_group_limited_topk_routing/candidate/kernel.py) |
| S_L2_004_fused_residual_rms_mlp | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/004_fused_residual_rms_mlp/candidate/kernel.py)<br>[dynamic_persistent_gemm_impl.py](../extensions/sol-execbench/cases/L2/004_fused_residual_rms_mlp/candidate/dynamic_persistent_gemm_impl.py) |
| S_Quant_008_nvfp4_multimodal_embedding_projection | Not included | [kernel.py](../extensions/sol-execbench/cases/Quant/008_nvfp4_multimodal_embedding_projection/candidate/kernel.py)<br>[nvfp4_gemm.py](../extensions/sol-execbench/cases/Quant/008_nvfp4_multimodal_embedding_projection/candidate/nvfp4_gemm.py)<br>[nvfp4_common.py](../extensions/sol-execbench/cases/Quant/008_nvfp4_multimodal_embedding_projection/candidate/nvfp4_common.py) |
| S_L2_018_cu_seqlens_variable_length_vision_attention | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/018_cu_seqlens_variable_length_vision_attention/candidate/kernel.py)<br>[gemm.py](../extensions/sol-execbench/cases/L2/018_cu_seqlens_variable_length_vision_attention/candidate/gemm.py)<br>[dense_gemm_helper.py](../extensions/sol-execbench/cases/L2/018_cu_seqlens_variable_length_vision_attention/candidate/dense_gemm_helper.py) |
| S_L2_031_flux_timestep_guidance_projection_embedding | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/031_flux_timestep_guidance_projection_embedding/candidate/kernel.py)<br>[dense_gemm_kernel.py](../extensions/sol-execbench/cases/L2/031_flux_timestep_guidance_projection_embedding/candidate/dense_gemm_kernel.py) |
| S_L1_052_altup_hidden_state_collapse_with_magnitude_normalization | Not included | [kernel.py](../extensions/sol-execbench/cases/L1/052_altup_hidden_state_collapse_with_magnitude_normalization/candidate/kernel.py)<br>[gemm_impl.py](../extensions/sol-execbench/cases/L1/052_altup_hidden_state_collapse_with_magnitude_normalization/candidate/gemm_impl.py) |
| S_L2_052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection/candidate/kernel.py) |
| S_L2_057_residual_coupling_flow_block | Not included | [kernel.py](../extensions/sol-execbench/cases/L2/057_residual_coupling_flow_block/candidate/kernel.py) |
| S_L1_087_embedding_with_initial_layernorm_backward | Not included | [kernel.cu](../extensions/sol-execbench/cases/L1/087_embedding_with_initial_layernorm_backward/candidate/kernel.cu) |
| S_L1_046_attention_softmax_with_softcapping_and_dropout | Not included | [a.cu](../extensions/sol-execbench/cases/L1/046_attention_softmax_with_softcapping_and_dropout/candidate/a.cu) |
| S_L1_016_rope_inverse_frequency_computation | Not included | [kernel.py](../extensions/sol-execbench/cases/L1/016_rope_inverse_frequency_computation/candidate/kernel.py) |
| S_FlashInfer-Bench_001_fused_add_rmsnorm_h2048 | Not included | [kernel.py](../extensions/sol-execbench/cases/FlashInfer-Bench/001_fused_add_rmsnorm_h2048/candidate/kernel.py) |

## gimlet (2 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|
| G02 | [reference.py](../extensions/gimlet/G02/reference.py) | Not included |
| G03 | [reference.py](../extensions/gimlet/G03/reference.py) | Not included |

## volta-bounds (0 records)

| Case | Reference / fixed source | Candidate / pre-fix source |
|---|---|---|

## Gimlet G01 reconstruction (separate from recovered sources)

Newly written runnable files: [reference](../extensions/gimlet/G01/reference.py), [candidate](../extensions/gimlet/G01/candidate.py), [runner](../extensions/gimlet/G01/run.py). See the [reconstruction notes](../extensions/gimlet/G01/README.md). These do not increase the recovered-source count.
