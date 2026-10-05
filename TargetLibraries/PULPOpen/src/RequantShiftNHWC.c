/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

// Per-channel RequantShift, int32 -> int8, channels last, parallel over rows (or over the flat range when there
// are fewer rows than cores). Same arithmetic as the Generic RequantShift_s32_s8_NHWC (64-bit product, rounding),
// without its per-element modulo and without every core redoing the whole tensor.
void PULP_RequantShift_s32_s8_NHWC(const int32_t *data_in, uint32_t size, const int32_t *mul, const int32_t *add,
                                   int8_t *data_out, int32_t log2D, uint32_t channels, int32_t output_min,
                                   int32_t output_max) {
  const uint32_t core_id = pi_core_id();
  const uint32_t rows = size / channels;
  const int64_t rnd = log2D > 0 ? ((int64_t)1 << (log2D - 1)) : 0;
  uint32_t r0, r1, c0, c1;
  if (rows >= NUM_CORES) {
    const uint32_t chunk = (rows + NUM_CORES - 1) / NUM_CORES;
    r0 = core_id * chunk < rows ? core_id * chunk : rows;
    r1 = r0 + chunk < rows ? r0 + chunk : rows;
    c0 = 0;
    c1 = channels;
  } else {  // few rows: split the channels instead
    const uint32_t chunk = (channels + NUM_CORES - 1) / NUM_CORES;
    r0 = 0;
    r1 = rows;
    c0 = core_id * chunk < channels ? core_id * chunk : channels;
    c1 = c0 + chunk < channels ? c0 + chunk : channels;
  }
  for (uint32_t r = r0; r < r1; r++) {
    const int32_t *in = data_in + r * channels;
    int8_t *out = data_out + r * channels;
    for (uint32_t c = c0; c < c1; c++) {
      int64_t v = (((int64_t)in[c] * mul[c] + add[c] + rnd) >> log2D);
      v = v < output_min ? output_min : (v > output_max ? output_max : v);
      out[c] = (int8_t)v;
    }
  }
}
