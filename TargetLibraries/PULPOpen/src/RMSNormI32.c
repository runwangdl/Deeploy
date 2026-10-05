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
  const uint32_t half = out_shift > 0 ? (uint32_t)1 << (out_shift - 1) : 0;
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
    // |ys| < 2^15 -> ys^2 < 2^30: four squares fit a uint32, then into the 64-bit total
    uint64_t ss = 0;
    uint32_t i = 0;
    for (; i + 4 <= D; i += 4) {
      const int32_t y0 = v[i] >> sh, y1 = v[i + 1] >> sh, y2 = v[i + 2] >> sh, y3 = v[i + 3] >> sh;
      ss += (uint32_t)(y0 * y0) + (uint32_t)(y1 * y1) + (uint32_t)(y2 * y2) + (uint32_t)(y3 * y3);
    }
    for (; i < D; i++) {
      const int32_t y0 = v[i] >> sh;
      ss += (uint32_t)(y0 * y0);
    }
    uint32_t rt = rmsi32_isqrt64(ss / D);
    if (rt < 1)
      rt = 1;
    // inv = 2^30 / rt <= 2^30 = ih * 2^15 + il: floor(ys * inv / 2^15) = ys * ih + floor(ys * il / 2^15), all 32-bit
    const int32_t inv = (int32_t)(((uint32_t)1 << 30) / rt);
    const int32_t ih = inv >> 15, il = inv & 0x7fff;
    for (uint32_t i = 0; i < D; i++) {
      const int32_t ys = v[i] >> sh;
      const int32_t n = ys * ih + ((ys * il) >> 15);
      const int32_t w = weight[i];
      const int32_t lo = n * w;
      int32_t q;
      const int64_t t = (int64_t)n * w;
      if (t != (int64_t)lo) {
        q = t < 0 ? -128 : 127;
      } else {
        const uint32_t at = lo < 0 ? (uint32_t)0 - (uint32_t)lo : (uint32_t)lo;
        const int32_t qa = (int32_t)((at + half) >> out_shift);
        q = lo < 0 ? -qa : qa;
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
      }
      o[i] = (int8_t)q;
    }
  }
}
