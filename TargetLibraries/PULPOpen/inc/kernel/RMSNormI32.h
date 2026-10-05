/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __DEEPLOY_MATH_RMSNORMI32_KERNEL_HEADER_
#define __DEEPLOY_MATH_RMSNORMI32_KERNEL_HEADER_
#include "DeeployPULPMath.h"
void PULP_RMSNormI32_s32_s8(const int32_t *data_in, const int32_t *weight, int8_t *data_out, uint32_t size,
                            uint32_t lastDimLength, int32_t out_shift);
void PULP_RequantShift_s32_s8_NHWC(const int32_t *data_in, uint32_t size, const int32_t *mul, const int32_t *add,
                                   int8_t *data_out, int32_t log2D, uint32_t channels, int32_t output_min,
                                   int32_t output_max);
#endif
