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
// Only built on the GAP9_w_NE16 platform (DEEPLOY_USE_NE16).
void GAP9_SSDScanNE16_i8_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z,
                            const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
                            const int32_t *__restrict__ C, const int32_t *__restrict__ A,
                            const int32_t *__restrict__ D_skip, void *__restrict__ y, int32_t *__restrict__ h_state,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t B_size,
                            uint32_t Q, uint32_t N, uint32_t P, uint32_t NHt, uint32_t L, uint32_t GH, uint32_t GW,
                            int32_t output_requant_mul_q40, uint32_t init_state, uint32_t epilogue_version,
                            uint32_t out_bits, uint32_t out_shift);

#endif // __DEEPLOY_MATH_GAP9_SSDSCAN_NE16_KERNEL_HEADER_
