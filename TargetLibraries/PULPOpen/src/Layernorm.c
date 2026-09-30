/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

#include <math.h>

void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias, uint32_t size,
                              uint32_t lastDimLength, float32_t epsilon) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t seq_length = size / lastDimLength;
  int32_t chunk =
      (seq_length >> log2Core) + ((seq_length & (NUM_CORES - 1)) != 0);
  int32_t start_seq = MIN(chunk * core_id, seq_length);
  int32_t end_seq = MIN(start_seq + chunk, seq_length);

  int32_t elem_start = start_seq * lastDimLength;
  int32_t elem_end = end_seq * lastDimLength;

  float32_t *local_data_in = data_in + elem_start;
  float32_t *local_data_out = data_out + elem_start;
  int32_t local_size = elem_end - elem_start;

  float32_t mean;
  float32_t sum;
  float32_t std;
  float32_t temp;

  int32_t local_seq_count = local_size / lastDimLength;

  for (int32_t i = 0; i < local_seq_count; i++) {

    sum = 0.0f;
    mean = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      mean += local_data_in[j + i * lastDimLength];
    }
    mean = mean / (float32_t)lastDimLength;

    sum = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      temp = local_data_in[j + i * lastDimLength] - mean;
      sum += temp * temp;
    }
    sum = sum / (float32_t)lastDimLength;
    sum += epsilon;
    std = sqrtf(sum);

    for (int32_t j = 0; j < lastDimLength; j++) {
      local_data_out[j + i * lastDimLength] =
          ((local_data_in[j + i * lastDimLength] - mean) / std) * scale[j] +
          bias[j];
    }
  }
}

#define SumDotpSS(a, b, c) __builtin_pulp_sdotsp4(a, b, c)

// floor(sqrt(x)) for x in [0, 2^30] via binary search
static inline int32_t _i_sqrt30(int32_t x) {
  if (x <= 0)
    return 0;
  int32_t r = 0;
  for (int bit = 15; bit >= 0; bit--) {
    const int32_t t = r + (1 << bit);
    if ((int64_t)t * t <= (int64_t)x)
      r = t;
  }
  return r;
}

static inline int32_t layernorm_isqrt_q15_from_variance(int64_t variance) {
  const int64_t v = variance + 1;
  return (v > 0) ? _i_sqrt30((int32_t)(((int64_t)1 << 30) / v)) : 0;
}

static inline int32_t layernorm_shift_round_away(int64_t x, int shift) {
  const int64_t half = (int64_t)1 << (shift - 1);
  const int64_t rounded = (x >= 0) ? (x + half) : (x - half);
  return (int32_t)(rounded >> shift);
}

void PULP_Layernorm_s8_s8(int8_t *data_in, int8_t *data_out,
              int32_t *gamma_mul_i32, int64_t *beta_add_i64,
              int32_t size, int32_t lastDimLength) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  const uint32_t D = (uint32_t)lastDimLength;
  const uint32_t num_vectors = (uint32_t)(size / lastDimLength);
  uint32_t chunk = (num_vectors >> log2Core) + ((num_vectors & (NUM_CORES - 1)) != 0);
  uint32_t chunk_start = MIN(chunk * core_id, num_vectors);
  uint32_t chunk_stop = MIN(chunk_start + chunk, num_vectors);

  for (uint32_t v = chunk_start; v < chunk_stop; v++) {
    const int8_t *vec_in = data_in + (v * D);
    int8_t *vec_out = data_out + (v * D);

    // pass 1: accumulate sum(x)
    int32_t sum_val = 0;
    if ((D & 3U) == 0U) {
      const v4s ones = (v4s){1, 1, 1, 1};
      const v4s *p = (const v4s *)vec_in;
      const uint32_t simd_count = D >> 2;
      for (uint32_t i = 0; i < simd_count; i++) {
        sum_val = SumDotpSS(p[i], ones, sum_val);
      }
    } else {
      for (uint32_t i = 0; i < D; i++) {
        sum_val += (int32_t)vec_in[i];
      }
    }

    const int64_t mean = (int64_t)sum_val / (int64_t)D;

    // pass 1.5: accumulate sum((x - mean)^2)
    int64_t var_sum = 0;
    for (uint32_t i = 0; i < D; i++) {
      const int64_t x_c = (int64_t)vec_in[i] - mean;
      var_sum += x_c * x_c;
    }
    const int64_t variance = var_sum / (int64_t)D;
    const int32_t isqrt_q15 = layernorm_isqrt_q15_from_variance(variance);

    // pass 2: y = round_half_away((x - mean)/std * gamma + beta, 39), clamp to int8
    for (uint32_t i = 0; i < D; i++) {
      const int32_t x_c = (int32_t)vec_in[i] - (int32_t)mean;
      const int32_t n_q15 = x_c * isqrt_q15;
      const int64_t total = (int64_t)n_q15 * (int64_t)gamma_mul_i32[i] + (int64_t)beta_add_i64[i];
      int32_t y = layernorm_shift_round_away(total, 39);
      vec_out[i] = (int8_t)CLAMP(y, -128, 127);
    }
  }
}
