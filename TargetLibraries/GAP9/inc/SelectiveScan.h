/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// GAP9 SelectiveScan prototype; exp LUT in SelectiveScanLUT.h. Guard distinct from PULP header.

#ifndef __DEEPLOY_MATH_GAP9_SELECTIVESCAN_KERNEL_HEADER_
#define __DEEPLOY_MATH_GAP9_SELECTIVESCAN_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void GAP9_SelectiveScan_i8_i8(
    const int8_t *__restrict__ x,
    const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt,
    const int32_t *__restrict__ B,
    const int32_t *__restrict__ C,
    const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip,
    int8_t *__restrict__ y,
    int32_t *__restrict__ h_buffer,
    const int32_t *__restrict__ gate_lut,
    uint32_t B_size, uint32_t L, uint32_t D_inner, uint32_t N,
    int32_t output_requant_mul_q40,
    uint32_t is_first_L_tile);

#endif // __DEEPLOY_MATH_GAP9_SELECTIVESCAN_KERNEL_HEADER_
