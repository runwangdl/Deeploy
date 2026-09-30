/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// SILU prototypes; LUTs live in Generic/inc/kernel/SILULUT.h. Guard must differ from that header's.

#ifndef __DEEPLOY_MATH_SILU_KERNEL_HEADER_
#define __DEEPLOY_MATH_SILU_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_SILU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize,
                      int32_t input_offset);

void PULP_SILU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                     int32_t input_offset, int8_t *lut);

#endif // __DEEPLOY_MATH_SILU_KERNEL_HEADER_
