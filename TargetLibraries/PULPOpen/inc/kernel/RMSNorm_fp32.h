/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RMSNORM_FP32_KERNEL_HEADER_
#define __DEEPLOY_MATH_RMSNORM_FP32_KERNEL_HEADER_

#include "DeeployPULPMath.h"

/* Row-wise RMSNorm over the last dimension, rows split over the cluster cores (same chunking as
 * PULP_Layernorm_fp32_fp32): out = in / sqrt(mean(in^2) + eps) * weight. */
void PULP_RMSNorm_fp32_fp32(float32_t *data_in, float32_t *weight, float32_t *data_out, uint32_t size,
                            uint32_t lastDimLength, float32_t eps);

#endif // __DEEPLOY_MATH_RMSNORM_FP32_KERNEL_HEADER_
