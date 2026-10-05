/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

// RMSNorm on an int32 tensor -> int8, row-wise over the last dimension (rows split over the cores).
// Scale-invariant in the input, so the int32 input may be in any scale (FEMBA Mamba-2: SSD gated output).
// per row: sh = max(0, bitlen(max|y|) - 15); ys = y >> sh; ms = sum(ys^2) / D; r = max(isqrt(ms), 1)
//          inv = 2^30 / r; n = (ys * inv) >> 15; out = sat8(round_half_away(n * w, out_shift))
// Bit-exact to RMSNormI32 in AI_AGENT/Mamba/scripts/int_onnx_sim.py and m2_export/m2_graph.py.

static inline uint32_t rmsi32_isqrt64(uint64_t x) {
  uint64_t r = 0, bit = (uint64_t)1 << 62;
  while (bit > x)
    bit >>= 2;
  while (bit) {
    if (x >= r + bit) {
      x -= r + bit;
      r = (r >> 1) + bit;
    } else {
      r >>= 1;
    }
    bit >>= 2;
  }
  return (uint32_t)r;
}

void PULP_RMSNormI32_s32_s8(const int32_t *data_in, const int32_t *weight, int8_t *data_out, uint32_t size,
                            uint32_t lastDimLength, int32_t out_shift) {
  const uint32_t core_id = pi_core_id();
  const uint32_t D = lastDimLength;
  const uint32_t rows = size / D;
  const uint32_t chunk = (rows + NUM_CORES - 1) / NUM_CORES;
  const uint32_t r0 = core_id * chunk < rows ? core_id * chunk : rows;
  const uint32_t r1 = r0 + chunk < rows ? r0 + chunk : rows;
  const int64_t half = (int64_t)1 << (out_shift - 1);
  for (uint32_t r = r0; r < r1; r++) {
    const int32_t *v = data_in + r * D;
    int8_t *o = data_out + r * D;
    uint32_t m = 0;
    for (uint32_t i = 0; i < D; i++) {
      const uint32_t a = (uint32_t)(v[i] < 0 ? -(int64_t)v[i] : v[i]);
      m = a > m ? a : m;
    }
    const int32_t bl = m ? 32 - __builtin_clz(m) : 0;
    const int32_t sh = bl > 15 ? bl - 15 : 0;
    int64_t ss = 0;
    for (uint32_t i = 0; i < D; i++) {
      const int32_t ys = v[i] >> sh;
      ss += (int64_t)ys * ys;
    }
    uint32_t rt = rmsi32_isqrt64((uint64_t)ss / D);
    if (rt < 1)
      rt = 1;
    const int64_t inv = ((int64_t)1 << 30) / rt;
    for (uint32_t i = 0; i < D; i++) {
      const int64_t n = ((int64_t)(v[i] >> sh) * inv) >> 15;
      const int64_t t = n * (int64_t)weight[i];
      const int64_t q = t >= 0 ? (t + half) >> out_shift : -(((-t) + half) >> out_shift);
      o[i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
  }
}
