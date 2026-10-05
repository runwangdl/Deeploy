/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GAP9_SSDSCAN_NE16_KERNEL_HEADER_
#define __DEEPLOY_MATH_GAP9_SSDSCAN_NE16_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// SSD_Scan_NE16: Mamba-2 chunked scan whose three per-head chunk products (Y_intra, S_local, Y_inter)
// run on the NE16 as 1x1 convolutions with the head's channels as pixels (GH x GW grid, P real).
// Inputs as GAP9_SSDScan_i8_i8 (tile layouts [B][L][NHt*P], dt [B][L][NHt], B/C [B][L][N], A/D [NHt]).
// h_state: [B][NHt][P][N] int32, persists across L tiles of the same head tile (init_state zeroes it).
// scratch: L1 lump of ssdscan_ne16_scratch_bytes(Q, N, P, NHt, GH, GW) bytes (see the template).
// epilogue_version: 1 = SSD_Scan's int64 gate+Q40 requant, 2 = folded 16-bit gate*mul LUT (one 32-bit multiply).
// out_bits: 8 (default, int8 y) or 32 (int32 y = sat32(rs(y_g, out_shift)), epilogue 1, feeds RMSNormI32).
// decay_mode 1 (two-scale decay, int32 output only): shared weights per chunk from Lam = cumsum asr(dta*A[0], 8),
// per-head data scaling from rho = cumsum(-asr(R*resid_mul, 8)); dta/R may be NULL for decay_mode 0.
// mamba3 1 (Mamba-3 rank 1): per-key weights m3_w (off-diagonal, state) / m3_gamma (diagonal) instead of dt, and
// m3_theta (may be NULL) rotates B and C per head (RoPE, angle carried across tiles); see ssd_ne16_ref.
// Only built on the GAP9_w_NE16 platform (DEEPLOY_USE_NE16).
void GAP9_SSDScanNE16_i8_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z,
                            const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
                            const int32_t *__restrict__ C, const int32_t *__restrict__ A,
                            const int32_t *__restrict__ D_skip, void *__restrict__ y, int32_t *__restrict__ h_state,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t B_size,
                            uint32_t Q, uint32_t N, uint32_t P, uint32_t NHt, uint32_t L, uint32_t GH, uint32_t GW,
                            int32_t output_requant_mul_q40, uint32_t init_state, uint32_t epilogue_version,
                            uint32_t out_bits, uint32_t out_shift, const int16_t *__restrict__ dta,
                            const int8_t *__restrict__ R, uint32_t decay_mode, int32_t resid_mul,
                            const int16_t *__restrict__ m3_gamma, const int16_t *__restrict__ m3_w,
                            const int16_t *__restrict__ m3_theta, uint32_t mamba3, uint32_t mimo_rank,
                            const int32_t *__restrict__ bc_Bw, const int32_t *__restrict__ bc_Cw,
                            const int32_t *__restrict__ bc_Bb, const int32_t *__restrict__ bc_Cb, uint32_t bc_norm,
                            uint32_t core_mm);

// StaticScan_NE16 (constant-parameter Mamba-2 block as per-head NE16 1x1 jobs over the whole window), see SSDScanNE16.c
void GAP9_StaticScanNE16_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z, const uint8_t *__restrict__ wenc,
                            const int32_t *__restrict__ comp, const int32_t *__restrict__ M,
                            const int32_t *__restrict__ Dq, int32_t *__restrict__ y,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t L, uint32_t P,
                            uint32_t NHt, int32_t out_shift);

#endif // __DEEPLOY_MATH_GAP9_SSDSCAN_NE16_KERNEL_HEADER_
