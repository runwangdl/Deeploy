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
  // 2x4 register blocking: 6 loads per 8 MACs, 8 accumulators (4x4 spilled on RV32: slower
  // than the original 2x2). Exact integer sums.
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);
  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN((uint32_t)core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  for (uint32_t i = M_start; i < M_end; i += 2) {
    const uint32_t two = (i + 1 < M_end);
    const int32_t *a0 = pSrcA + i * N;
    const int32_t *a1 = a0 + (two ? N : 0);
    uint32_t k = 0;
    for (; k + 4 <= O; k += 4) {
      int32_t s00 = C_offset, s01 = C_offset, s02 = C_offset, s03 = C_offset;
      int32_t s10 = C_offset, s11 = C_offset, s12 = C_offset, s13 = C_offset;
      const int8_t *bp = pSrcB + k;
      for (uint32_t j = 0; j < N; j++) {
        const int32_t x0 = a0[j] + A_offset, x1 = a1[j] + A_offset;
        const int32_t b0 = bp[0] + B_offset, b1 = bp[1] + B_offset, b2 = bp[2] + B_offset, b3 = bp[3] + B_offset;
        bp += O;
        s00 += x0 * b0; s01 += x0 * b1; s02 += x0 * b2; s03 += x0 * b3;
        s10 += x1 * b0; s11 += x1 * b1; s12 += x1 * b2; s13 += x1 * b3;
      }
      int32_t *c = pDstC + i * O + k;
      c[0] = s00; c[1] = s01; c[2] = s02; c[3] = s03;
      if (two) { c += O; c[0] = s10; c[1] = s11; c[2] = s12; c[3] = s13; }
    }
    for (; k < O; k++) {
      for (uint32_t r = 0; r < 1 + two; r++) {
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

// int16 A x int8 B -> int32 (FEMBA Mamba-2 classifier out_proj: A = wide SelectiveScanI16 output).
// A [M][N] int16 (rows 4-byte aligned when N is even), B [N][O] int8, C [M][O]. 4x2 output blocks (8 accumulators:
// a 4x4 block spills on RV32), k in pairs with pv.sdotsp.h: per k-pair 4 A words + 4 B bytes packed to 2 int16x2
// + 8 dot products. Work items = (row block, col block), round-robin over the cores; edge blocks clamp their
// row/column indices (duplicate work, masked stores), so the inner loop has no branches. Exact integer sums.
void PULP_MatMul_s16_s8_s32(const int16_t *__restrict__ pSrcA, const int8_t *__restrict__ pSrcB,
                            int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t O, int32_t C_offset) {
  const uint32_t core_id = pi_core_id();
  const uint32_t RB = (M + 3) >> 2, CB = (O + 1) >> 1;
  const uint32_t Np = N & ~1u;
  const uint32_t O2 = 2 * O;
  for (uint32_t w = core_id; w < RB * CB; w += NUM_CORES) {
    const uint32_t i = (w / CB) << 2, k = (w % CB) << 1;
    const uint32_t r1 = i + 1 < M ? i + 1 : M - 1, r2 = i + 2 < M ? i + 2 : M - 1, r3 = i + 3 < M ? i + 3 : M - 1;
    const uint32_t k1 = k + 1 < O ? k + 1 : O - 1;
    const int16_t *a0 = pSrcA + i * N, *a1 = pSrcA + r1 * N, *a2 = pSrcA + r2 * N, *a3 = pSrcA + r3 * N;
    const int8_t *b0 = pSrcB + k, *b1 = pSrcB + k1;
    int32_t s00 = C_offset, s01 = C_offset, s10 = C_offset, s11 = C_offset;
    int32_t s20 = C_offset, s21 = C_offset, s30 = C_offset, s31 = C_offset;
    for (uint32_t j = 0; j < Np; j += 2) {
      const v2s x0 = *(const v2s *)(a0 + j), x1 = *(const v2s *)(a1 + j);
      const v2s x2 = *(const v2s *)(a2 + j), x3 = *(const v2s *)(a3 + j);
      const v2s w0 = __builtin_pulp_pack2((int16_t)b0[0], (int16_t)b0[O]);
      const v2s w1 = __builtin_pulp_pack2((int16_t)b1[0], (int16_t)b1[O]);
      b0 += O2;
      b1 += O2;
      s00 = __builtin_pulp_sdotsp2(x0, w0, s00); s01 = __builtin_pulp_sdotsp2(x0, w1, s01);
      s10 = __builtin_pulp_sdotsp2(x1, w0, s10); s11 = __builtin_pulp_sdotsp2(x1, w1, s11);
      s20 = __builtin_pulp_sdotsp2(x2, w0, s20); s21 = __builtin_pulp_sdotsp2(x2, w1, s21);
      s30 = __builtin_pulp_sdotsp2(x3, w0, s30); s31 = __builtin_pulp_sdotsp2(x3, w1, s31);
    }
    if (N & 1) {
      const int32_t v0 = b0[0], v1 = b1[0];
      const uint32_t j = N - 1;
      s00 += a0[j] * v0; s01 += a0[j] * v1; s10 += a1[j] * v0; s11 += a1[j] * v1;
      s20 += a2[j] * v0; s21 += a2[j] * v1; s30 += a3[j] * v0; s31 += a3[j] * v1;
    }
    const int ok1 = k + 1 < O;
    int32_t *c = pDstC + i * O + k;
    c[0] = s00; if (ok1) c[1] = s01;
    if (i + 1 < M) { c += O; c[0] = s10; if (ok1) c[1] = s11; }
    if (i + 2 < M) { c += O; c[0] = s20; if (ok1) c[1] = s21; }
    if (i + 3 < M) { c += O; c[0] = s30; if (ok1) c[1] = s31; }
  }
}
