/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

// PULP-parallel dequantization (sN -> float32).
void PULP_Dequant_s8_f32(int8_t *data_in, int32_t size, float32_t scale,
                          float32_t zero_point, float32_t *data_out) {
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);

  uint32_t chunk = ((uint32_t)size >> log2Core) +
                    (((uint32_t)size & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN((uint32_t)core_id * chunk, (uint32_t)size);
  uint32_t end = MIN(start + chunk, (uint32_t)size);

  for (uint32_t i = start; i < end; i++) {
    data_out[i] = ((float32_t)data_in[i] - zero_point) * scale;
  }
}

void PULP_Dequant_s32_f32(int32_t *data_in, int32_t size, float32_t scale,
                           float32_t zero_point, float32_t *data_out) {
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);

  uint32_t chunk = ((uint32_t)size >> log2Core) +
                    (((uint32_t)size & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN((uint32_t)core_id * chunk, (uint32_t)size);
  uint32_t end = MIN(start + chunk, (uint32_t)size);

  for (uint32_t i = start; i < end; i++) {
    data_out[i] = ((float32_t)data_in[i] - zero_point) * scale;
  }
}
