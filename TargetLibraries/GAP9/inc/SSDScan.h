/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef __DEEPLOY_MATH_GAP9_SSDSCAN_KERNEL_HEADER_
#define __DEEPLOY_MATH_GAP9_SSDSCAN_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// Q15 exp LUT (exp(x), x in [-20,0], 128 steps/unit, 2561 entries) living in L1; defined in SSDScan.c
extern const int16_t *SSDScan_exp_lut_ptr;

// gate_lut: 256-entry Q13 SiLU-gate LUT, indexed by z + 128.
// output_requant_mul_q40: INT8 output requant multiplier (round_shift by 40).
// h_state: [B, N_heads, Head_dim, N] recurrent state; persists across L-tiles.
// init_state: 1 zeroes state at sequence start, 0 continues a previous tile.
void GAP9_SSDScan_i8_i8(
    const int8_t *__restrict__ x,
    const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt,
    const int32_t *__restrict__ B,
    const int32_t *__restrict__ C,
    const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip,
    int8_t *__restrict__ y,
    int32_t *__restrict__ h_state,
    const int32_t *__restrict__ gate_lut,
    uint32_t B_size, uint32_t Chunk_size, uint32_t N,
    uint32_t Head_dim, uint32_t Group_dim, uint32_t N_heads,
    uint32_t L, int32_t output_requant_mul_q40,
    uint32_t init_state);

#endif // __DEEPLOY_MATH_GAP9_SSDSCAN_KERNEL_HEADER_
