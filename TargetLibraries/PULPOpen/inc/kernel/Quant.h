/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_QUANT_KERNEL_HEADER_
#define __DEEPLOY_MATH_QUANT_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Quant_f32_s8(float32_t *data_in, int32_t size, float32_t scale,
                        float32_t zero_point, int32_t min_val, int32_t max_val,
                        int8_t *data_out);

#endif // __DEEPLOY_MATH_QUANT_KERNEL_HEADER_
