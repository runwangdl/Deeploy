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
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN((uint32_t)core_id * M_chunk, M);
  uint32_t M_end   = MIN(M_start + M_chunk, M);

  if (M_start >= M_end) {
    return;
  }

  const int32_t *local_pSrcA = pSrcA + M_start * N;
  int32_t *local_pDstC = pDstC + M_start * O;
  uint32_t M_size = M_end - M_start;

  for (uint32_t i = 0; i < M_size / 2; i++) {
    for (uint32_t k = 0; k < O / 2; k++) {
      int32_t sum00 = C_offset;
      int32_t sum01 = C_offset;
      int32_t sum10 = C_offset;
      int32_t sum11 = C_offset;

      for (uint32_t j = 0; j < N; j++) {
        int32_t a0 = local_pSrcA[(i * 2)     * N + j] + A_offset;
        int32_t a1 = local_pSrcA[(i * 2 + 1) * N + j] + A_offset;
        int32_t b0 = pSrcB[j * O + (k * 2)]     + B_offset;
        int32_t b1 = pSrcB[j * O + (k * 2 + 1)] + B_offset;

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
        int32_t b = pSrcB[j * O + k] + B_offset;
        sum0 += (local_pSrcA[(i * 2)     * N + j] + A_offset) * b;
        sum1 += (local_pSrcA[(i * 2 + 1) * N + j] + A_offset) * b;
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
        sum += (local_pSrcA[i * N + j] + A_offset) * (pSrcB[j * O + k] + B_offset);
      }
      local_pDstC[i * O + k] = sum;
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
