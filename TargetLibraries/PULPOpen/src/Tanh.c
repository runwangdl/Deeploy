/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>

void PULP_Tanh_fp32_fp32(float32_t *input, float32_t *output, uint32_t size) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t start = MIN(chunk * core_id, size);
  int32_t end = MIN(start + chunk, size);
  int32_t local_size = end - start;

  float32_t *local_input = input + start;
  float32_t *local_output = output + start;

  for (int32_t i = 0; i < local_size; i++) {
    local_output[i] = tanhf(local_input[i]);
  }
}
