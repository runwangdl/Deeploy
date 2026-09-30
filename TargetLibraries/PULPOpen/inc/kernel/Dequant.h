/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_DEQUANT_KERNEL_HEADER_
#define __DEEPLOY_MATH_DEQUANT_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Dequant_s8_f32(int8_t *data_in, int32_t size, float32_t scale,
                          float32_t zero_point, float32_t *data_out);

void PULP_Dequant_s32_f32(int32_t *data_in, int32_t size, float32_t scale,
                           float32_t zero_point, float32_t *data_out);

#endif // __DEEPLOY_MATH_DEQUANT_KERNEL_HEADER_
