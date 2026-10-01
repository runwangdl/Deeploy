#include "DeeployPULPMath.h"
#include "pmsis.h"

// Depthwise 1-D convolution with fused per-channel requantization, int8 -> int8.
// in: [C][L_in] (channels-first, as the conv1d node feeds it), w: [C][K], out: [L_out][C]
// (the layout the fused pulp-nn depthwise kernel produced, so the rest of the graph is unchanged).
// requant: (acc * mul[c] + add[c]) >> log2D, truncating; `add` carries the rounding half (merge pass).
// Replaces pulp_nn_depthwise_i8_i8_i8 for FEMBA's conv1d: no im2col, exact at tile edges,
// ~5x fewer cycles. Parallel over channels.
void PULP_DWConv1D_s8_s8_rq(const int8_t *__restrict__ in, const int8_t *__restrict__ w,
                            const int32_t *__restrict__ mul, const int32_t *__restrict__ add,
                            int8_t *__restrict__ out, uint32_t C, uint32_t L_in, uint32_t L_out,
                            uint32_t K, uint32_t pad_top, uint32_t stride, uint32_t log2D) {
  const uint32_t core_id = pi_core_id();
  const uint32_t chunk = (C + NUM_CORES - 1) / NUM_CORES;
  const uint32_t c0 = core_id * chunk;
  const uint32_t c1 = (c0 + chunk < C) ? c0 + chunk : C;
  for (uint32_t c = c0; c < c1; c++) {
    const int8_t *x = in + c * L_in;
    const int8_t *wc = w + c * K;
    const int64_t m = mul[c];
    const int64_t a = add[c];
    int8_t *o = out + c;
    for (uint32_t t = 0; t < L_out; t++) {
      int32_t acc = 0;
      const int32_t base = (int32_t)(t * stride) - (int32_t)pad_top;
      if (base >= 0 && base + (int32_t)K <= (int32_t)L_in) {
        const int8_t *xp = x + base;
        for (uint32_t r = 0; r < K; r++) acc += (int32_t)xp[r] * (int32_t)wc[r];
      } else {
        for (uint32_t r = 0; r < K; r++) {
          const int32_t ti = base + (int32_t)r;
          if (ti >= 0 && ti < (int32_t)L_in) acc += (int32_t)x[ti] * (int32_t)wc[r];
        }
      }
      int64_t v = ((int64_t)acc * m + a) >> log2D;
      o[t * C] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
  }
}
