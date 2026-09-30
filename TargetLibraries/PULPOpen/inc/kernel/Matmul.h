/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O);

void PULP_MatMul_s32_s8_s32(const int32_t *__restrict__ pSrcA,
                             const int8_t *__restrict__ pSrcB,
                             int32_t *__restrict__ pDstC,
                             uint32_t M, uint32_t N, uint32_t O,
                             int32_t A_offset, int32_t B_offset,
                             int32_t C_offset);

void PULP_MatMul_s8_s8_s32(const int8_t *__restrict__ pSrcA,
                            const int8_t *__restrict__ pSrcB,
                            int32_t *__restrict__ pDstC,
                            uint32_t M, uint32_t N, uint32_t O,
                            int32_t A_offset, int32_t B_offset,
                            int32_t C_offset);

#endif // __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
