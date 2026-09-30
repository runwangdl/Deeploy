/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

// PULP-parallel quantization (float32 -> sN): round-half-away-from-zero, then clamp.
void PULP_Quant_f32_s8(float32_t *data_in, int32_t size, float32_t scale,
                        float32_t zero_point, int32_t min_val, int32_t max_val,
                        int8_t *data_out) {
  int32_t core_id = pi_core_id();
  int32_t log2Core = LOG2(NUM_CORES);

  uint32_t chunk = ((uint32_t)size >> log2Core) +
                    (((uint32_t)size & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN((uint32_t)core_id * chunk, (uint32_t)size);
  uint32_t end = MIN(start + chunk, (uint32_t)size);

  for (uint32_t i = start; i < end; i++) {
    float32_t shifted_val = data_in[i] * scale + zero_point;
    int32_t quantized =
        (int32_t)(shifted_val + 0.5f * (shifted_val >= 0.0f ? 1.0f : -1.0f));
    data_out[i] = (int8_t)CLAMP(quantized, min_val, max_val);
  }
}
