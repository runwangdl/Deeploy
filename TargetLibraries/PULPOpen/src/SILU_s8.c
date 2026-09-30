/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

#include "kernel/SILULUT.h"

void PULP_SILU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize,
                      int32_t input_offset) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  uint32_t chunk = ((uint32_t)dataSize >> log2Core) +
                   (((uint32_t)dataSize & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN(chunk * core_id, (uint32_t)dataSize);
  uint32_t stop = MIN(start + chunk, (uint32_t)dataSize);
  for (uint32_t i = start; i < stop; i++) {
    int32_t x = data_in[i] + 128 - input_offset;
    data_out[i] = SILU_lut_s8_s32[x];
  }
}

void PULP_SILU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                     int32_t input_offset, int8_t *lut) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  uint32_t chunk = ((uint32_t)dataSize >> log2Core) +
                   (((uint32_t)dataSize & (NUM_CORES - 1)) != 0);
  uint32_t start = MIN(chunk * core_id, (uint32_t)dataSize);
  uint32_t stop = MIN(start + chunk, (uint32_t)dataSize);
  for (uint32_t i = start; i < stop; i++) {
    int32_t x = data_in[i] + 128 - input_offset;
    data_out[i] = lut[x];
  }
}
