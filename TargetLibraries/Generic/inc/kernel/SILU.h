/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Prototypes only; SILU LUTs live in SILULUT.h (included by the .c files).

#ifndef __DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void SILU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize,
                 int32_t input_offset);

void SILU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                int32_t input_offset, int8_t *lut);

#endif //__DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_
