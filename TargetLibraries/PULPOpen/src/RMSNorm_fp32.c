/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include <math.h>

void PULP_RMSNorm_fp32_fp32(float32_t *data_in, float32_t *weight, float32_t *data_out, uint32_t size,
                            uint32_t lastDimLength, float32_t eps) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t seq_length = size / lastDimLength;
  int32_t chunk = (seq_length >> log2Core) + ((seq_length & (NUM_CORES - 1)) != 0);
  int32_t start_seq = MIN(chunk * core_id, seq_length);
  int32_t end_seq = MIN(start_seq + chunk, seq_length);

  for (int32_t i = start_seq; i < end_seq; i++) {
    float32_t *in = data_in + i * lastDimLength;
    float32_t *out = data_out + i * lastDimLength;

    float32_t sum = 0.0f;
    for (uint32_t j = 0; j < lastDimLength; j++) {
      sum += in[j] * in[j];
    }
    float32_t inv = 1.0f / sqrtf(sum / (float32_t)lastDimLength + eps);

    for (uint32_t j = 0; j < lastDimLength; j++) {
      out[j] = in[j] * inv * weight[j];
    }
  }
}
