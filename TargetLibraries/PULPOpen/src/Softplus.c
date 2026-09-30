/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

#include "kernel/SoftplusLUT.h"

void PULP_Softplus_s8_s8(const int8_t *data_in, int8_t *data_out, uint32_t size,
                         int32_t input_offset) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  uint32_t chunk_start = MIN(chunk * core_id, size);
  uint32_t chunk_stop = MIN(chunk_start + chunk, size);

  for (uint32_t i = chunk_start; i < chunk_stop; i++) {
    uint8_t idx = (uint8_t)((int32_t)data_in[i] + 128 - input_offset);
    data_out[i] = softplus_lut[idx];
  }
}

void PULP_Softplus_s32_s16(const int32_t *data_in, int16_t *data_out,
                           uint32_t size, int32_t input_offset) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  uint32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  uint32_t chunk_start = MIN(chunk * core_id, size);
  uint32_t chunk_stop = MIN(chunk_start + chunk, size);

  for (uint32_t i = chunk_start; i < chunk_stop; i++) {
    // Q20 -> index: floor(x_float / 0.1) = (q20 * 10) >> 20
    int32_t idx = (int32_t)(((int64_t)data_in[i] * 10) >> 20);
    idx = CLAMP(idx, -128, 127);
    data_out[i] = softplus_q88_lut[(uint8_t)(idx + 128 - input_offset)];
  }
}
