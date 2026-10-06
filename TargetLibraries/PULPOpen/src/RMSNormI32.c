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

// per-row reductions shared by the cores (two slots, alternating by row: one barrier between a slot's write and its
// reads, and a slot is rewritten only two rows later, after every core has passed the next row's first barrier)
static PI_L1 uint32_t rmsi32_pmax[2][NUM_CORES];
static PI_L1 uint64_t rmsi32_pss[2][NUM_CORES];

// Every row is split over the cores by columns (a tile holds ~10 rows, too few to split rows over 8 cores evenly).
void PULP_RMSNormI32_s32_s8(const int32_t *data_in, const int32_t *weight, int8_t *data_out, uint32_t size,
                            uint32_t lastDimLength, int32_t out_shift) {
  const uint32_t core_id = pi_core_id();
  const uint32_t D = lastDimLength;
  const uint32_t rows = size / D;
  const uint32_t cch = ((D + NUM_CORES - 1) / NUM_CORES + 3) & ~3u;     // column chunk, a multiple of 4
  const uint32_t c0 = core_id * cch < D ? core_id * cch : D;
  const uint32_t c1 = c0 + cch < D ? c0 + cch : D;
  const uint32_t half = out_shift > 0 ? (uint32_t)1 << (out_shift - 1) : 0;
  for (uint32_t r = 0; r < rows; r++) {
    const int32_t *v = data_in + r * D;
    int8_t *o = data_out + r * D;
    const uint32_t slot = r & 1;
    // pass 1: max |y| over the slice (branch-free abs; INT_MIN -> 2^31 as uint32)
    uint32_t m = 0;
    for (uint32_t i = c0; i < c1; i++) {
      const uint32_t x = (uint32_t)v[i], sg = (uint32_t)(v[i] >> 31);
      const uint32_t a = (x ^ sg) - sg;
      m = a > m ? a : m;
    }
    rmsi32_pmax[slot][core_id] = m;
    pi_cl_team_barrier();
    m = 0;
    for (uint32_t k = 0; k < NUM_CORES; k++)
      m = rmsi32_pmax[slot][k] > m ? rmsi32_pmax[slot][k] : m;
    const int32_t bl = m ? 32 - __builtin_clz(m) : 0;
    const int32_t sh = bl > 15 ? bl - 15 : 0;
    // pass 2: |ys| < 2^15 -> ys^2 < 2^30: four squares fit a uint32, then into the 64-bit total
    uint64_t ss = 0;
    uint32_t i = c0;
    for (; i + 4 <= c1; i += 4) {
      const int32_t y0 = v[i] >> sh, y1 = v[i + 1] >> sh, y2 = v[i + 2] >> sh, y3 = v[i + 3] >> sh;
      ss += (uint32_t)(y0 * y0) + (uint32_t)(y1 * y1) + (uint32_t)(y2 * y2) + (uint32_t)(y3 * y3);
    }
    for (; i < c1; i++) {
      const int32_t y0 = v[i] >> sh;
      ss += (uint32_t)(y0 * y0);
    }
    rmsi32_pss[slot][core_id] = ss;
    pi_cl_team_barrier();
    ss = 0;
    for (uint32_t k = 0; k < NUM_CORES; k++)
      ss += rmsi32_pss[slot][k];
    uint32_t rt = rmsi32_isqrt64(ss / D);
    if (rt < 1)
      rt = 1;
    // inv = 2^30 / rt <= 2^30 = ih * 2^15 + il: floor(ys * inv / 2^15) = ys * ih + floor(ys * il / 2^15), all 32-bit
    const int32_t inv = (int32_t)(((uint32_t)1 << 30) / rt);
    const int32_t ih = inv >> 15, il = inv & 0x7fff;
    // pass 3: q = sat8(round_half_away(n * w, out_shift)); a product outside int32 saturates (sign of the product).
    // (a branch-free variant with masks + p.clip was slower: 0.80 M vs 0.75 M per norm -- the overflow branch is
    // never taken, so the branchy loop costs less than the extra mask arithmetic)
    for (i = c0; i < c1; i++) {
      const int32_t ys = v[i] >> sh;
      const int32_t n = ys * ih + ((ys * il) >> 15);
      const int64_t t = (int64_t)n * (int64_t)weight[i];
      const int32_t lo = (int32_t)t, hi = (int32_t)(t >> 32);
      const uint32_t sg = (uint32_t)(lo >> 31);
      const uint32_t at = ((uint32_t)lo ^ sg) - sg;
      const uint32_t u = (at + half) >> out_shift;          // unsigned: 2^31 (lo = INT_MIN, out_shift 0) stays large
      const int32_t qa = (int32_t)(u < 128u ? u : 128u);    // |q| <= 128 before the final clip
      int32_t q = (int32_t)(((uint32_t)qa ^ sg) - sg);
      q = q > 127 ? 127 : q;
      if (hi != (lo >> 31))
        q = hi < 0 ? -128 : 127;
      o[i] = (int8_t)q;
    }
  }
}
