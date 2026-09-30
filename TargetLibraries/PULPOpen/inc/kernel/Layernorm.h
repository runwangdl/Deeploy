/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias, uint32_t size,
                              uint32_t lastDimLength, float32_t epsilon);

void PULP_Layernorm_s8_s8(int8_t *data_in, int8_t *data_out,
                          int32_t *weight, int64_t *bias, int32_t size,
                          int32_t lastDimLength);

#endif // __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER__