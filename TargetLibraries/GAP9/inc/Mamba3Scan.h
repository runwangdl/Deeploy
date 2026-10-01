/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GAP9_MAMBA3SCAN_KERNEL_HEADER_
#define __DEEPLOY_MATH_GAP9_MAMBA3SCAN_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// Mamba-3 (arXiv:2603.15569) chunked scan in the SSDScan Q15 domain, with the three
// Mamba-3 axes each controlled by its own input so they can be ablated one at a time:
//
//   gamma [B, L, N_heads] int16 Q8.8: lam_t * dt_t, the weight of v_t on the diagonal.
//   w     [B, L, N_heads] int16 Q8.8: gamma_t + (1 - lam_{t+1}) dt_{t+1}, the weight of
//         v_t everywhere else (the alpha_{t+1} in beta_{t+1} cancels one decay step, so
//         both parts of v_t's contribution share one decay - see mamba3_q.py).
//         The reference derives both from lam; they are precomputed outside the scan so
//         the kernel never has to look one timestep ahead, which an L-tile boundary would
//         truncate. gamma == w == dt is Euler, i.e. bit-exact SSDScan.
//   theta [B, L, N_heads] int16, 1/65536 turn per unit: data-dependent RoPE angle
//         increment. 0 everywhere disables the rotation.
//   R     mimo rank. x/z/y carry [.., Head_dim, R], B/C carry [.., N, R]; the state
//         stays [Head_dim, N] because the rank is summed over on the way in (Eq. 13).
//
// theta_state: [B, N_heads] int32 accumulated angle, carried across L-tiles like
// h_state; zeroed when init_state == 1.
//
// Everything else - gate_lut, output_requant_mul_q40, h_state, init_state - is as in
// GAP9_SSDScan_i8_i8. With gamma == w == dt, theta == 0 and R == 1 the output is bit-identical
// to SSDScan on the same inputs.
void GAP9_Mamba3Scan_i8_i8(
    const int8_t *__restrict__ x,
    const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt,
    const int32_t *__restrict__ B,
    const int32_t *__restrict__ C,
    const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip,
    const int16_t *__restrict__ gamma,
    const int16_t *__restrict__ w,
    const int16_t *__restrict__ theta,
    int8_t *__restrict__ y,
    int32_t *__restrict__ h_state,
    int32_t *__restrict__ theta_state,
    const int32_t *__restrict__ gate_lut,
    uint32_t B_size, uint32_t Chunk_size, uint32_t N,
    uint32_t Head_dim, uint32_t Group_dim, uint32_t N_heads,
    uint32_t L, uint32_t R, int32_t output_requant_mul_q40,
    uint32_t init_state);

#endif // __DEEPLOY_MATH_GAP9_MAMBA3SCAN_KERNEL_HEADER_
