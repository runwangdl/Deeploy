/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployGAP9Math.h"

// Mamba-3 gates (AI_AGENT m3_ref.m3_gates), [B][L][H] int16 outputs, untiled along L (w looks one token ahead):
//   which 0: gamma = (lam * dt) >> 15,  lam = lut[raw + 128] (Q15 sigmoid)
//   which 1: w[t] = gamma[t] + ((32768 - lam[t+1]) * dt[t+1]) >> 15  (0 for the last token)
//   which 2: theta = lut[raw + 128]  (RoPE angle step, 1/65536 turn)
void GAP9_M3Gates(const int16_t *dt, const int8_t *raw, const int16_t *lut, int16_t *out, uint32_t B, uint32_t L,
                  uint32_t H, uint32_t which) {
  const uint32_t core = pi_core_id(), tot = B * L * H;
  const uint32_t ch = (tot + NUM_CORES - 1) / NUM_CORES;
  const uint32_t i0 = core * ch < tot ? core * ch : tot, i1 = i0 + ch < tot ? i0 + ch : tot;
  for (uint32_t i = i0; i < i1; i++) {
    const int32_t l = lut[(int32_t)raw[i] + 128];
    if (which == 2) {
      out[i] = (int16_t)l;
      continue;
    }
    int32_t v = (l * (int32_t)dt[i]) >> 15;
    if (which == 1) {
      const uint32_t t = (i / H) % L;
      if (t + 1 < L) {
        const int32_t ln = lut[(int32_t)raw[i + H] + 128];
        v += ((32768 - ln) * (int32_t)dt[i + H]) >> 15;
      }
    }
    out[i] = (int16_t)v;
  }
  pi_cl_team_barrier();
}
