/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// SSD (Mamba-2 chunked scan) with the three per-head chunk products executed on the NE16 as 1x1
// convolutions (regime R2: exact decay mask inside int8 weights, per-(head, chunk) power-of-two
// scales). Layout trick: the head's channels are the pixels (GH x GW grid, P real), tokens are the
// input/output channels. Per head and chunk:
//   J1  Y_intra : in  x~[c][s]  (uint8 = x+128)       W1[t][s] = q8(G_h . L)      acc1[c][t]
//   J2  S_local : in  x~[c][s]                        W2[n][s] = q8(dB e^{lam_Q-lam_s})  acc2[c][n]
//   J3  Y_inter : in  S8[c][n]  (uint8 = S8+128)      W3[t][n] = q8(C e^{lam_t})   acc3[c][t]
// Cores: lambda / exp tables / weights / repacking / x transpose / state quantisation (phase A, heads
// in parallel), then per head: core 0 dispatches the three jobs while cores 1..7 combine the previous
// head (y epilogue identical to GAP9_SSDScan_i8_i8) and fold its S_local into the int32 state.
// Numerics are mirrored bit for bit by AI_AGENT/Mamba/scripts/ssd_ne16_ref.py.

#ifdef DEEPLOY_USE_NE16

#include "DeeployPULPMath.h"
#include "SSDScan.h"
#include "SSDScanNE16.h"

#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"

#define SSDN_ONE_Q15 32767
#define SSDN_Q20 (1 << 20)
#define SSDN_EXP_STEP (SSDN_Q20 / 128)
#define SSDN_EXP_RANGE_Q20 ((int64_t)20 * SSDN_Q20)
#define SSDN_EXP_LUT_LEN 2561
#define SSDN_OUT_SHIFT 40
#define SSDN_SLOTS 2
#define SSDN_META_HDR 8
#ifndef SSDN_PROFILE
#define SSDN_PROFILE 1
#endif
#if SSDN_PROFILE
#define SSDN_T(var) uint32_t var = pi_perf_read(PI_PERF_CYCLES)
#else
#define SSDN_T(var)
#endif

static ne16_task_t ssdn_tasks[SSDN_SLOTS][3];

static inline int32_t ssdn_expq(const int16_t *lut, int32_t z_q15) {
  int64_t arg = (int64_t)z_q15 * 32;
  if (arg >= 0)
    return SSDN_ONE_Q15;
  if (arg < -SSDN_EXP_RANGE_Q20)
    arg = -SSDN_EXP_RANGE_Q20;
  int32_t idx = (int32_t)((arg + SSDN_EXP_RANGE_Q20 + (SSDN_EXP_STEP / 2)) / SSDN_EXP_STEP);
  if (idx > SSDN_EXP_LUT_LEN - 1)
    idx = SSDN_EXP_LUT_LEN - 1;
  return lut[idx];
}

// smallest shift whose rounded result fits 7 bits (negative = left shift, at most 8)
static inline int ssdn_pow2_exp(uint32_t m) {
  if (m == 0)
    return 0;
  int sh = (32 - __builtin_clz(m)) - 7;
  if (sh > 0 && (((m + (1u << (sh - 1))) >> sh) > 127))
    sh++;
  if (sh < -8)
    sh = -8;
  return sh;
}

// round-half-up shift without the (v + half) overflow: floor(v / 2^sh) + bit (sh-1) of v
static inline int32_t ssdn_qshift(int32_t v, int sh) {
  return sh > 0 ? ((v >> sh) + ((v >> (sh - 1)) & 1)) : (v << (-sh));
}

static inline int64_t ssdn_mul_pow2(int64_t v, int e) { return e >= 0 ? (v << e) : (v >> (-e)); }

static inline int64_t ssdn_round_shift(int64_t x, int s) {
  int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

static inline int32_t ssdn_sat_i32(int64_t x) {
  if (x > (int64_t)2147483647)
    return (int32_t)2147483647;
  if (x < -(int64_t)2147483648)
    return (int32_t)(-2147483648);
  return (int32_t)x;
}

static inline int8_t ssdn_sat_i8(int64_t x) { return (int8_t)(x > 127 ? 127 : (x < -128 ? -128 : x)); }

static inline uint32_t ssdn_absmax(uint32_t m, int32_t v) {
  uint32_t a = (uint32_t)(v < 0 ? -v : v);
  return a > m ? a : m;
}

// 8x8 bit-matrix transpose (Hacker's Delight 7-3, 32-bit); see ssdn_repack for the row/plane order
static inline void ssdn_transpose8(uint32_t x, uint32_t y, uint8_t *b, uint32_t stride) {
  uint32_t t;
  t = (x ^ (x >> 7)) & 0x00AA00AAu; x = x ^ t ^ (t << 7);
  t = (y ^ (y >> 7)) & 0x00AA00AAu; y = y ^ t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCCu; x = x ^ t ^ (t << 14);
  t = (y ^ (y >> 14)) & 0x0000CCCCu; y = y ^ t ^ (t << 14);
  t = (x & 0xF0F0F0F0u) | ((y >> 4) & 0x0F0F0F0Fu);
  y = ((x << 4) & 0xF0F0F0F0u) | (y & 0x0F0F0F0Fu);
  x = t;
  // HD convention is MSB-first on both axes; with rows fed in reverse order, B[j] is bit-plane 7-j
  b[7 * stride] = (uint8_t)(x >> 24); b[6 * stride] = (uint8_t)(x >> 16);
  b[5 * stride] = (uint8_t)(x >> 8);  b[4 * stride] = (uint8_t)x;
  b[3 * stride] = (uint8_t)(y >> 24); b[2 * stride] = (uint8_t)(y >> 16);
  b[1 * stride] = (uint8_t)(y >> 8);  b[0 * stride] = (uint8_t)y;
}

// int8 [KO][KI] -> NE16 1x1 8-bit weight layout [KO][KI/16][8 bit-planes][2 bytes], stored as w+128
static void ssdn_repack(const int8_t *w, uint8_t *enc, uint32_t KO, uint32_t KI) {
  const uint32_t kim_n = KI / 16;
  for (uint32_t ko = 0; ko < KO; ko++) {
    for (uint32_t kim = 0; kim < kim_n; kim++) {
      const uint8_t *r = (const uint8_t *)w + ko * KI + kim * 16;
      uint8_t *e = enc + (ko * kim_n + kim) * 16;
      for (uint32_t half = 0; half < 2; half++) {
        // little-endian word loads put ki 8h+7 in the MSB: HD's transpose then yields plane (7-j) in B[j]
        const uint32_t *q = (const uint32_t *)(r + half * 8);
        ssdn_transpose8(q[1] ^ 0x80808080u, q[0] ^ 0x80808080u, e + half, 2);
      }
    }
  }
}

static void ssdn_task_setup(ne16_task_t *tk, uint32_t KI, uint32_t KO, uint32_t GH, uint32_t GW) {
  ne16_task_init(tk);
  ne16_task_set_op_to_conv(tk, 1, 0, 1);
  ne16_task_set_bits(tk, 8, 32, 8);
  ne16_task_set_weight_offset(tk, weightOffsetModeLayerWise, -128);
  ne16_task_set_strides(tk, KI, GW * KI, KI, GW * KO * 4, KO * 4);
  ne16_task_set_counters(tk, KI, GH, GW, KO, 0, 0);
  ne16_task_set_padding(tk, 0, 0, 0, 0, 0);
}

static inline void ssdn_dispatch(const ne16_dev_t *dev, ne16_task_t *tk, const uint8_t *in, const uint8_t *wenc,
                                 int32_t *out) {
  tk->data.infeat_addr = (uint32_t)in;
  tk->data.weights_addr = (uint32_t)wenc;
  tk->data.outfeat_addr = (uint32_t)out;
  ne16_nnx_dispatch_wait(dev);
  ne16_nnx_dispatch(dev, tk);
}

void GAP9_SSDScanNE16_i8_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z,
                            const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
                            const int32_t *__restrict__ C, const int32_t *__restrict__ A,
                            const int32_t *__restrict__ D_skip, int8_t *__restrict__ y, int32_t *__restrict__ h_state,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t B_size,
                            uint32_t Q, uint32_t N, uint32_t P, uint32_t NHt, uint32_t L, uint32_t GH, uint32_t GW,
                            int32_t output_requant_mul_q40, uint32_t init_state) {
  const int core = (int)pi_core_id();
  const uint32_t PXA = GH * GW;
  const uint32_t d_inner = NHt * P;
  const uint32_t WSZ = Q * Q + N * Q + Q * N;     // int8 per head: W1 | W2 | W3
  const uint32_t ASZ = PXA * (2 * Q + N);         // int32 per slot: acc1 | acc2 | acc3
  const uint32_t MSZ = SSDN_META_HDR + 2 * Q + N; // int32 per head: sh1 sh2 sh3 shs eQ0 - - - comp1[Q] comp2[N] comp3[Q]
  const int16_t *exp_lut = SSDScan_exp_lut_ptr;

  uint8_t *xt = scratch;                                    // [NHt][PXA][Q]
  uint8_t *su = xt + NHt * PXA * Q;                         // [NHt][PXA][N]
  int8_t *w8 = (int8_t *)(su + NHt * PXA * N);              // [NHt][WSZ]
  uint8_t *wenc = (uint8_t *)w8 + NHt * WSZ;                // [NHt][WSZ]
  int32_t *acc = (int32_t *)(wenc + NHt * WSZ);             // [SLOTS][ASZ]
  int32_t *meta = acc + SSDN_SLOTS * ASZ;                   // [NHt][MSZ]
  int32_t *G = meta + NHt * MSZ;                            // [Q][Q]

  const ne16_dev_t *dev = ne16_pulp_get_dev();
  if (core == 0) {
    for (uint32_t s = 0; s < SSDN_SLOTS; s++) {
      ssdn_task_setup(&ssdn_tasks[s][0], Q, Q, GH, GW);
      ssdn_task_setup(&ssdn_tasks[s][1], Q, N, GH, GW);
      ssdn_task_setup(&ssdn_tasks[s][2], N, Q, GH, GW);
    }
  }

  const int number_of_chunks = (int)(L / Q);
#if SSDN_PROFILE
  uint32_t prof_G = 0, prof_A = 0, prof_B = 0;
  if (core == 0) { pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start(); }
  SSDN_T(prof_t0);
#endif

  for (uint32_t b = 0; b < B_size; b++) {
    if (init_state) {
      int32_t *hb = h_state + (uint32_t)b * NHt * P * N;
      const uint32_t n_words = NHt * P * N;
      for (uint32_t i = (uint32_t)core; i < n_words; i += NUM_CORES)
        hb[i] = 0;
    }
    pi_cl_team_barrier();

    for (int chunk = 0; chunk < number_of_chunks; chunk++) {
      const uint32_t t0 = (uint32_t)chunk * Q;
      const int32_t *Bk = B + ((uint32_t)b * L + t0) * N; // [Q][N]
      const int32_t *Ck = C + ((uint32_t)b * L + t0) * N;

      SSDN_T(prof_g0);
      // ---- shared G[t][s] = (C_t . B_s) >> 15, rows over cores ----
      for (uint32_t t = (uint32_t)core; t < Q; t += NUM_CORES) {
        for (uint32_t s = 0; s < Q; s++) {
          int64_t d = 0;
          for (uint32_t n = 0; n < N; n++)
            d += (int64_t)Ck[t * N + n] * (int64_t)Bk[s * N + n];
          G[t * Q + s] = (int32_t)(d >> 15);
        }
      }
      pi_cl_team_barrier();
      SSDN_T(prof_a0);

      // ---- phase A: per head (heads over cores) ----
      for (uint32_t hh = (uint32_t)core; hh < NHt; hh += NUM_CORES) {
        int32_t lam[32], e_t0[32], e_Qs[32];
        int32_t *mt = meta + hh * MSZ;
        int32_t *comp1 = mt + SSDN_META_HDR, *comp2 = comp1 + Q, *comp3 = comp2 + N;
        int8_t *W1 = w8 + hh * WSZ, *W2 = W1 + Q * Q, *W3 = W2 + N * Q;
        const int64_t A_h = (int64_t)A[hh];
        int32_t running = 0;
        for (uint32_t t = 0; t < Q; t++) {
          const int64_t dtv = (int64_t)dt[((uint32_t)b * L + t0 + t) * NHt + hh];
          running += (int32_t)((dtv * A_h) >> 8);
          lam[t] = running;
        }
        for (uint32_t t = 0; t < Q; t++)
          e_t0[t] = ssdn_expq(exp_lut, lam[t]);
        for (uint32_t s = 0; s + 1 < Q; s++)
          e_Qs[s] = ssdn_expq(exp_lut, lam[Q - 1] - lam[s]);
        e_Qs[Q - 1] = SSDN_ONE_Q15;

        // W1[t][s] = q(M1), M1 = ((G[t][s]*dt_s) >> 8) * e_ts >> 15, s <= t
        uint32_t mx = 0;
        for (uint32_t t = 0; t < Q; t++) {
          for (uint32_t s = 0; s <= t; s++) {
            const int32_t e = (s == t) ? SSDN_ONE_Q15 : ssdn_expq(exp_lut, lam[t] - lam[s]);
            const int64_t dts = (int64_t)dt[((uint32_t)b * L + t0 + s) * NHt + hh];
            const int32_t Gh = (int32_t)(((int64_t)G[t * Q + s] * dts) >> 8);
            mx = ssdn_absmax(mx, (int32_t)(((int64_t)Gh * e) >> 15));
          }
        }
        const int sh1 = ssdn_pow2_exp(mx);
        for (uint32_t t = 0; t < Q; t++) {
          int32_t comp = 0;
          for (uint32_t s = 0; s < Q; s++) {
            int32_t w = 0;
            if (s <= t) {
              const int32_t e = (s == t) ? SSDN_ONE_Q15 : ssdn_expq(exp_lut, lam[t] - lam[s]);
              const int64_t dts = (int64_t)dt[((uint32_t)b * L + t0 + s) * NHt + hh];
              const int32_t Gh = (int32_t)(((int64_t)G[t * Q + s] * dts) >> 8);
              w = ssdn_qshift((int32_t)(((int64_t)Gh * e) >> 15), sh1);
            }
            W1[t * Q + s] = (int8_t)w;
            comp += w;
          }
          comp1[t] = 128 * comp;
        }

        // W2[n][s] = q(dBd), dBd = ((dt_s*B[s][n]) >> 8) * e_Qs[s] >> 15
        mx = 0;
        for (uint32_t s = 0; s < Q; s++) {
          const int64_t dts = (int64_t)dt[((uint32_t)b * L + t0 + s) * NHt + hh];
          for (uint32_t n = 0; n < N; n++) {
            const int32_t dB = (int32_t)((dts * (int64_t)Bk[s * N + n]) >> 8);
            mx = ssdn_absmax(mx, (int32_t)(((int64_t)dB * e_Qs[s]) >> 15));
          }
        }
        const int sh2 = ssdn_pow2_exp(mx);
        for (uint32_t n = 0; n < N; n++)
          comp2[n] = 0;
        for (uint32_t s = 0; s < Q; s++) {
          const int64_t dts = (int64_t)dt[((uint32_t)b * L + t0 + s) * NHt + hh];
          for (uint32_t n = 0; n < N; n++) {
            const int32_t dB = (int32_t)((dts * (int64_t)Bk[s * N + n]) >> 8);
            const int32_t w = ssdn_qshift((int32_t)(((int64_t)dB * e_Qs[s]) >> 15), sh2);
            W2[n * Q + s] = (int8_t)w;
            comp2[n] += w;
          }
        }
        for (uint32_t n = 0; n < N; n++)
          comp2[n] *= 128;

        // W3[t][n] = q(M3), M3 = (C[t][n]*e_t0[t]) >> 15
        mx = 0;
        for (uint32_t t = 0; t < Q; t++)
          for (uint32_t n = 0; n < N; n++)
            mx = ssdn_absmax(mx, (int32_t)(((int64_t)Ck[t * N + n] * e_t0[t]) >> 15));
        const int sh3 = ssdn_pow2_exp(mx);
        for (uint32_t t = 0; t < Q; t++) {
          int32_t comp = 0;
          for (uint32_t n = 0; n < N; n++) {
            const int32_t w = ssdn_qshift((int32_t)(((int64_t)Ck[t * N + n] * e_t0[t]) >> 15), sh3);
            W3[t * N + n] = (int8_t)w;
            comp += w;
          }
          comp3[t] = 128 * comp;
        }

        // state -> S8 (+128) into su[hh][c][n]
        const int32_t *hs = h_state + ((uint32_t)b * NHt + hh) * P * N;
        mx = 0;
        for (uint32_t i = 0; i < P * N; i++)
          mx = ssdn_absmax(mx, hs[i]);
        const int shs = ssdn_pow2_exp(mx);
        uint8_t *su_h = su + hh * PXA * N;
        for (uint32_t i = 0; i < P * N; i++)
          su_h[i] = (uint8_t)(ssdn_qshift(hs[i], shs) ^ 0x80);

        // x transpose -> xt[hh][c][s] (+128)
        uint8_t *xt_h = xt + hh * PXA * Q;
        for (uint32_t s = 0; s < Q; s++) {
          const uint8_t *xrow = (const uint8_t *)x + ((uint32_t)b * L + t0 + s) * d_inner + hh * P;
          for (uint32_t c = 0; c < P; c++)
            xt_h[c * Q + s] = xrow[c] ^ 0x80;
        }

        // repack the three weights
        uint8_t *we = wenc + hh * WSZ;
        ssdn_repack(W1, we, Q, Q);
        ssdn_repack(W2, we + Q * Q, N, Q);
        ssdn_repack(W3, we + Q * Q + N * Q, Q, N);

        mt[0] = sh1; mt[1] = sh2; mt[2] = sh3; mt[3] = shs; mt[4] = e_t0[Q - 1];
      }
      pi_cl_team_barrier();
      SSDN_T(prof_b0);

      // ---- phase B: core 0 feeds the NE16 with head hh, cores 1..7 finish head hh-1 ----
      for (uint32_t hh = 0; hh <= NHt; hh++) {
        if (core == 0) {
          if (hh < NHt) {
            const uint32_t sl = hh & 1;
            const uint8_t *we = wenc + hh * WSZ;
            int32_t *a = acc + sl * ASZ;
            ssdn_dispatch(dev, &ssdn_tasks[sl][0], xt + hh * PXA * Q, we, a);
            ssdn_dispatch(dev, &ssdn_tasks[sl][1], xt + hh * PXA * Q, we + Q * Q, a + PXA * Q);
            ssdn_dispatch(dev, &ssdn_tasks[sl][2], su + hh * PXA * N, we + Q * Q + N * Q, a + PXA * Q + PXA * N);
            ne16_nnx_resolve_wait(dev, &ssdn_tasks[sl][2]);
          }
        } else if (hh > 0) {
          const uint32_t hp = hh - 1;
          const uint32_t sl = hp & 1;
          const int32_t *a1 = acc + sl * ASZ, *a2 = a1 + PXA * Q, *a3 = a2 + PXA * N;
          const int32_t *mt = meta + hp * MSZ;
          const int32_t *comp1 = mt + SSDN_META_HDR, *comp2 = comp1 + Q, *comp3 = comp2 + N;
          const int sh1 = mt[0], sh2 = mt[1], e3 = mt[2] + mt[3] - 15;
          const int64_t eQ0 = (int64_t)mt[4];
          const int64_t D_h = (int64_t)D_skip[hp];
          const int wcore = core - 1, wcores = NUM_CORES - 1;
          // y epilogue, rows t over the worker cores
          for (uint32_t t = (uint32_t)wcore; t < Q; t += (uint32_t)wcores) {
            const uint32_t row = ((uint32_t)b * L + t0 + t) * d_inner + hp * P;
            const int8_t *xr = x + row, *zr = z + row;
            int8_t *yr = y + row;
            const int64_t c1 = comp1[t], c3 = comp3[t];
            for (uint32_t c = 0; c < P; c++) {
              const int64_t v1 = (int64_t)a1[c * Q + t] - c1;
              const int64_t v3 = (int64_t)a3[c * Q + t] - c3;
              const int64_t y_acc = ssdn_mul_pow2(v1, sh1) + ssdn_mul_pow2(v3, e3) + D_h * (int64_t)xr[c];
              const int64_t gate = (int64_t)gate_lut[(int32_t)zr[c] + 128];
              const int64_t y_g = ssdn_round_shift(y_acc * gate, 13);
              yr[c] = ssdn_sat_i8(ssdn_round_shift(y_g * (int64_t)output_requant_mul_q40, SSDN_OUT_SHIFT));
            }
          }
          // state update, channels over the worker cores
          int32_t *hs = h_state + ((uint32_t)b * NHt + hp) * P * N;
          for (uint32_t c = (uint32_t)wcore; c < P; c += (uint32_t)wcores) {
            for (uint32_t n = 0; n < N; n++) {
              const int64_t dec = (eQ0 * (int64_t)hs[c * N + n]) >> 15;
              const int64_t sl_ = ssdn_mul_pow2((int64_t)a2[c * N + n] - comp2[n], sh2);
              hs[c * N + n] = ssdn_sat_i32(dec + sl_);
            }
          }
        }
        pi_cl_team_barrier();
      }
#if SSDN_PROFILE
      if (core == 0) { uint32_t t = pi_perf_read(PI_PERF_CYCLES); prof_G += prof_a0 - prof_g0; prof_A += prof_b0 - prof_a0; prof_B += t - prof_b0; }
#endif
    }
  }
  pi_cl_team_barrier();
#if SSDN_PROFILE
  if (core == 0) printf("SSDN_PROF NHt=%u L=%u G=%u A=%u B=%u total=%u\n", NHt, L, prof_G, prof_A, prof_B, pi_perf_read(PI_PERF_CYCLES) - prof_t0);
#endif
}

#endif // DEEPLOY_USE_NE16
