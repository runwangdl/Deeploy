/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Prototypes only; softplus LUTs live in SoftplusLUT.h (included by Softplus.c).

#ifndef __DEEPLOY_MATH_SOFTPLUS_KERNEL_HEADER_
#define __DEEPLOY_MATH_SOFTPLUS_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Softplus_s8_s8(const int8_t *data_in, int8_t *data_out, uint32_t size,
                         int32_t input_offset);

void PULP_Softplus_s32_s16(const int32_t *data_in, int16_t *data_out,
                           uint32_t size, int32_t input_offset);

#endif // __DEEPLOY_MATH_SOFTPLUS_KERNEL_HEADER_
