/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  const float32_t *local_pSrcA = pSrcA + M_start * N;
  float32_t *local_pDstY = pDstY + M_start * O;

  uint32_t O_block = O - (O % 7);

  for (uint32_t i = 0; i < M_size; i++) {

    for (uint32_t j = 0; j < O_block; j += 7) {
      float32_t sum0 = 0.0f;
      float32_t sum1 = 0.0f;
      float32_t sum2 = 0.0f;
      float32_t sum3 = 0.0f;
      float32_t sum4 = 0.0f;
      float32_t sum5 = 0.0f;
      float32_t sum6 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a0 = local_pSrcA[i * N + k];

        float32_t b0 = pSrcB[k * O + (j + 0)];
        float32_t b1 = pSrcB[k * O + (j + 1)];
        float32_t b2 = pSrcB[k * O + (j + 2)];
        float32_t b3 = pSrcB[k * O + (j + 3)];
        float32_t b4 = pSrcB[k * O + (j + 4)];
        float32_t b5 = pSrcB[k * O + (j + 5)];
        float32_t b6 = pSrcB[k * O + (j + 6)];

        sum0 += a0 * b0;
        sum1 += a0 * b1;
        sum2 += a0 * b2;
        sum3 += a0 * b3;
        sum4 += a0 * b4;
        sum5 += a0 * b5;
        sum6 += a0 * b6;
      }

      local_pDstY[i * O + (j + 0)] = sum0;
      local_pDstY[i * O + (j + 1)] = sum1;
      local_pDstY[i * O + (j + 2)] = sum2;
      local_pDstY[i * O + (j + 3)] = sum3;
      local_pDstY[i * O + (j + 4)] = sum4;
      local_pDstY[i * O + (j + 5)] = sum5;
      local_pDstY[i * O + (j + 6)] = sum6;
    }

    for (uint32_t j = O_block; j < O; j++) {
      float32_t sum = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a_val = local_pSrcA[i * N + k];
        float32_t b_val = pSrcB[k * O + j];
        sum += a_val * b_val;
      }

      local_pDstY[i * O + j] = sum;
    }
  }
}

void PULP_MatMul_s32_s8_s32(const int32_t *__restrict__ pSrcA,
                             const int8_t *__restrict__ pSrcB,
                             int32_t *__restrict__ pDstC,
                             uint32_t M, uint32_t N, uint32_t O,
                             int32_t A_offset, int32_t B_offset,
                             int32_t C_offset) {
  // 4x4 register blocking (8 loads per 16 MACs); exact integer sums, same as the 2x2 version.
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);
  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN((uint32_t)core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  for (uint32_t i = M_start; i < M_end; i += 4) {
    const uint32_t ni = (M_end - i < 4) ? M_end - i : 4;
    const int32_t *a0 = pSrcA + i * N;
    const int32_t *a1 = a0 + ((ni > 1) ? N : 0);
    const int32_t *a2 = a0 + ((ni > 2) ? 2 * N : 0);
    const int32_t *a3 = a0 + ((ni > 3) ? 3 * N : 0);
    uint32_t k = 0;
    for (; k + 4 <= O; k += 4) {
      int32_t s00 = C_offset, s01 = C_offset, s02 = C_offset, s03 = C_offset;
      int32_t s10 = C_offset, s11 = C_offset, s12 = C_offset, s13 = C_offset;
      int32_t s20 = C_offset, s21 = C_offset, s22 = C_offset, s23 = C_offset;
      int32_t s30 = C_offset, s31 = C_offset, s32 = C_offset, s33 = C_offset;
      const int8_t *bp = pSrcB + k;
      for (uint32_t j = 0; j < N; j++) {
        const int32_t x0 = a0[j] + A_offset, x1 = a1[j] + A_offset, x2 = a2[j] + A_offset, x3 = a3[j] + A_offset;
        const int32_t b0 = bp[0] + B_offset, b1 = bp[1] + B_offset, b2 = bp[2] + B_offset, b3 = bp[3] + B_offset;
        bp += O;
        s00 += x0 * b0; s01 += x0 * b1; s02 += x0 * b2; s03 += x0 * b3;
        s10 += x1 * b0; s11 += x1 * b1; s12 += x1 * b2; s13 += x1 * b3;
        s20 += x2 * b0; s21 += x2 * b1; s22 += x2 * b2; s23 += x2 * b3;
        s30 += x3 * b0; s31 += x3 * b1; s32 += x3 * b2; s33 += x3 * b3;
      }
      int32_t *c = pDstC + i * O + k;
      c[0] = s00; c[1] = s01; c[2] = s02; c[3] = s03;
      if (ni > 1) { c += O; c[0] = s10; c[1] = s11; c[2] = s12; c[3] = s13; }
      if (ni > 2) { c += O; c[0] = s20; c[1] = s21; c[2] = s22; c[3] = s23; }
      if (ni > 3) { c += O; c[0] = s30; c[1] = s31; c[2] = s32; c[3] = s33; }
    }
    for (; k < O; k++) {
      for (uint32_t r = 0; r < ni; r++) {
        const int32_t *ar = pSrcA + (i + r) * N;
        int32_t sum = C_offset;
        for (uint32_t j = 0; j < N; j++) sum += (ar[j] + A_offset) * (pSrcB[j * O + k] + B_offset);
        pDstC[(i + r) * O + k] = sum;
      }
    }
  }
}

void PULP_MatMul_s8_s8_s32(const int8_t *__restrict__ pSrcA,
                            const int8_t *__restrict__ pSrcB,
                            int32_t *__restrict__ pDstC,
                            uint32_t M, uint32_t N, uint32_t O,
                            int32_t A_offset, int32_t B_offset,
                            int32_t C_offset) {
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN((uint32_t)core_id * M_chunk, M);
  uint32_t M_end   = MIN(M_start + M_chunk, M);

  if (M_start >= M_end) {
    return;
  }

  const int8_t *local_pSrcA = pSrcA + M_start * N;
  int32_t *local_pDstC = pDstC + M_start * O;
  uint32_t M_size = M_end - M_start;

  for (uint32_t i = 0; i < M_size / 2; i++) {
    for (uint32_t k = 0; k < O / 2; k++) {
      int32_t sum00 = C_offset;
      int32_t sum01 = C_offset;
      int32_t sum10 = C_offset;
      int32_t sum11 = C_offset;

      for (uint32_t j = 0; j < N; j++) {
        int32_t a0 = (int32_t)local_pSrcA[(i * 2)     * N + j] + A_offset;
        int32_t a1 = (int32_t)local_pSrcA[(i * 2 + 1) * N + j] + A_offset;
        int32_t b0 = (int32_t)pSrcB[j * O + (k * 2)]     + B_offset;
        int32_t b1 = (int32_t)pSrcB[j * O + (k * 2 + 1)] + B_offset;

        sum00 += a0 * b0;
        sum01 += a0 * b1;
        sum10 += a1 * b0;
        sum11 += a1 * b1;
      }

      local_pDstC[(i * 2)     * O + (k * 2)]     = sum00;
      local_pDstC[(i * 2)     * O + (k * 2 + 1)] = sum01;
      local_pDstC[(i * 2 + 1) * O + (k * 2)]     = sum10;
      local_pDstC[(i * 2 + 1) * O + (k * 2 + 1)] = sum11;
    }

    // clean up for odd O
    for (uint32_t k = (O / 2) * 2; k < O; k++) {
      int32_t sum0 = C_offset;
      int32_t sum1 = C_offset;
      for (uint32_t j = 0; j < N; j++) {
        int32_t b = (int32_t)pSrcB[j * O + k] + B_offset;
        sum0 += ((int32_t)local_pSrcA[(i * 2)     * N + j] + A_offset) * b;
        sum1 += ((int32_t)local_pSrcA[(i * 2 + 1) * N + j] + A_offset) * b;
      }
      local_pDstC[(i * 2)     * O + k] = sum0;
      local_pDstC[(i * 2 + 1) * O + k] = sum1;
    }
  }

  // clean up for odd M_size
  for (uint32_t i = (M_size / 2) * 2; i < M_size; i++) {
    for (uint32_t k = 0; k < O; k++) {
      int32_t sum = C_offset;
      for (uint32_t j = 0; j < N; j++) {
        sum += ((int32_t)local_pSrcA[i * N + j] + A_offset) * ((int32_t)pSrcB[j * O + k] + B_offset);
      }
      local_pDstC[i * O + k] = sum;
    }
  }
}
