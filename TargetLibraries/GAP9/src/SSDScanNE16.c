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
// Phase A (per chunk): 4 work items per head over the 8 cores -- W1 / W2 / W3 (build, quantise,
// bit-plane repack) and the data (x transpose, state quantisation). Phase B: core 0 queues the three
// jobs of head h, then all cores run the y epilogue (identical to GAP9_SSDScan_i8_i8) and the state
// update of head h-1 while the NE16 works. Numerics mirrored bit for bit by
// AI_AGENT/Mamba/scripts/ssd_ne16_ref.py.

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
#define SSDN_PROFILE 0
#endif
#if SSDN_PROFILE
static PI_L1 uint32_t ssdn_p1[3];
#define SSDN_T(var) uint32_t var = pi_perf_read(PI_PERF_CYCLES)
#else
#define SSDN_T(var)
#endif

static ne16_task_t ssdn_tasks[SSDN_SLOTS][3];
#if SSDN_PROFILE
static uint32_t ssdn_p_epi, ssdn_p_upd;  // core 1 only
#endif

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

static inline int64_t ssdn_round_shift(int64_t x, int s) {
  int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

// (int64_t)v << l for an int32 v and 1 <= l < 32, built from two 32-bit shifts (RV32 has no 64-bit shifter)
static inline __attribute__((always_inline)) int64_t ssdn_shl32(int32_t v, int l) {
  const uint32_t lo = (uint32_t)v << l;
  const int32_t hi = v >> (32 - l);
  return (int64_t)(((uint64_t)(uint32_t)hi << 32) | lo);
}

// branchless symmetric rounding shift by 13 (the SiLU gate), int64
static inline __attribute__((always_inline)) int64_t ssdn_rs13(int64_t x) {
  const int64_t m = x >> 63;
  const int64_t a = ((x ^ m) - m + (1 << 12)) >> 13;
  return (a ^ m) - m;
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

// quantise an int32 matrix held in scr (rows x cols, row stride cols) to int8 with a pow2 scale;
// W gets it transposed when `transposed` ([cols][rows]); comp[ko] = 128 * sum of row ko of W
static inline int ssdn_quantise(const int32_t *scr, uint32_t rows, uint32_t cols, uint32_t mx, int8_t *W,
                                int32_t *comp, int transposed) {
  const int sh = ssdn_pow2_exp(mx);
  if (!transposed) {
    for (uint32_t r = 0; r < rows; r++) {
      int32_t c = 0;
      for (uint32_t k = 0; k < cols; k++) {
        const int32_t w = ssdn_qshift(scr[r * cols + k], sh);
        W[r * cols + k] = (int8_t)w;
        c += w;
      }
      comp[r] = 128 * c;
    }
  } else {
    for (uint32_t k = 0; k < cols; k++) {
      int32_t c = 0;
      for (uint32_t r = 0; r < rows; r++) {
        const int32_t w = ssdn_qshift(scr[r * cols + k], sh);
        W[k * rows + r] = (int8_t)w;
        c += w;
      }
      comp[k] = 128 * c;
    }
  }
  return sh;
}


typedef struct {
  const int8_t *x, *z; void *y; const int16_t *dt; const int32_t *A, *D_skip, *gate_lut, *Bk, *Ck, *G;
  int32_t *h_state, *hmax, *acc, *meta; uint8_t *xt, *su, *wenc; int8_t *w8; const int16_t *exp_lut, *gm16;
  int32_t ep2_SG, mul_q40; uint32_t b, L, t0, NHt, P, N, Q, PXA, WSZ, ASZ, MSZ, d_inner, epilogue_version, out_bits, out_shift;
  const int16_t *dta; const int8_t *R; uint32_t decay_mode; int32_t resid_mul;  // two-scale decay (decay_mode 1)
  const int16_t *m3g, *m3w, *m3th; int32_t *rot, *thstate; uint32_t m3;          // Mamba-3 rank 1 (gamma, w, theta)
  uint32_t Rk;                                                                    // Mamba-3 MIMO rank (1 otherwise)
  int32_t *mtab;   // MIMO: per head [Q*Q e_ts | Q e_Qs | Q e_t0 | QR + N + QR row maxima]
  // trained Mamba-3 (bc_norm): int8 B/C projections, RMS-normalised per (token, rank) with Bw/Cw, + per-head Bb/Cb,
  // rotated per head; per-head rows (rotm [NHt][2][Q][N*R]) and per-head G (Gph [NHt][QR][QR])
  const int8_t *Bp8, *Cp8; const int32_t *Bw, *Cw, *Bb, *Cb; int32_t *bn, *rotm, *Gph; uint32_t bcn;
} ssdn_ctx_t;
// one shared context in L1 (same values on every core; core 0 writes it, the following barriers order the reads)
static PI_L1 ssdn_ctx_t ssdn_cx;

// Mamba-3 RoPE (mamba3_q._apply_rope): cos/sin over one turn in 256 steps, Q15, clamped to +-32767
static int16_t ssdn_rot_cos[256], ssdn_rot_sin[256];
static const int16_t ssdn_rot_q[65] = {  // = ssd_ne16_ref.ROT_SIN[0..64]
    0, 804, 1608, 2411, 3212, 4011, 4808, 5602, 6393, 7180, 7962, 8740, 9512, 10279, 11039, 11793, 12540, 13279,
    14010, 14733, 15447, 16151, 16846, 17531, 18205, 18868, 19520, 20160, 20788, 21403, 22006, 22595, 23170, 23732,
    24279, 24812, 25330, 25833, 26320, 26791, 27246, 27684, 28106, 28511, 28899, 29269, 29622, 29957, 30274, 30572,
    30853, 31114, 31357, 31581, 31786, 31972, 32138, 32286, 32413, 32522, 32610, 32679, 32729, 32758, 32767};
static void ssdn_rot_init(void) {  // sin over [0, 2pi) from the quarter-wave table (exact copies of the reference values)
  static uint8_t done;
  if (done)
    return;
  done = 1;
  for (int i = 0; i < 256; i++) {
    const int q = i >> 6, r = i & 63;
    const int sv = (q == 0) ? ssdn_rot_q[r] : (q == 1) ? ssdn_rot_q[64 - r] : (q == 2) ? -ssdn_rot_q[r] : -ssdn_rot_q[64 - r];
    const int j = (i + 64) & 255, qc = j >> 6, rc = j & 63;
    const int cv = (qc == 0) ? ssdn_rot_q[rc] : (qc == 1) ? ssdn_rot_q[64 - rc] : (qc == 2) ? -ssdn_rot_q[rc] : -ssdn_rot_q[64 - rc];
    ssdn_rot_sin[i] = (int16_t)sv;
    ssdn_rot_cos[i] = (int16_t)cv;
  }
}

// Mamba-3 step R, head hh: rotate this chunk's B and C rows by the accumulated angle (carried per head across chunks
// and L tiles in thstate) into rot[hh][0|1][t][n]
static __attribute__((noinline)) void ssdn_m3_rope(const ssdn_ctx_t *cx, uint32_t hh) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, N = cx->N, Q = cx->Q;
  int32_t *rb = cx->rot + hh * 2 * Q * N, *rc = rb + Q * N;
  int32_t th = cx->thstate[b * NHt + hh];
  for (uint32_t t = 0; t < Q; t++) {
    th += (int32_t)cx->m3th[(b * L + t0 + t) * NHt + hh];
    const uint32_t idx = ((uint32_t)th >> 8) & 255u;
    const int64_t c = ssdn_rot_cos[idx], sn = ssdn_rot_sin[idx];
    const int32_t *Bt = cx->Bk + t * N, *Ct = cx->Ck + t * N;
    for (uint32_t n = 0; n + 1 < N; n += 2) {
      const int64_t be = Bt[n], bo = Bt[n + 1], ce = Ct[n], co = Ct[n + 1];
      rb[t * N + n] = (int32_t)((c * be + sn * bo) >> 15);
      rb[t * N + n + 1] = (int32_t)((-sn * be + c * bo) >> 15);
      rc[t * N + n] = (int32_t)((c * ce + sn * co) >> 15);
      rc[t * N + n + 1] = (int32_t)((-sn * ce + c * co) >> 15);
    }
    if (N & 1) { rb[t * N + N - 1] = Bt[N - 1]; rc[t * N + N - 1] = Ct[N - 1]; }
  }
  cx->thstate[b * NHt + hh] = th;
}

// one phase-A work item (head hh, part 0..3): own frame, so it does not add to the caller's stack
static __attribute__((noinline)) void ssdn_phaseA_item(const ssdn_ctx_t *cx, uint32_t it, int32_t *cscr) {
  const int8_t *x = cx->x; const int16_t *dt = cx->dt; const int32_t *A = cx->A;
  const int32_t *h_state = cx->h_state; const int32_t *hmax = cx->hmax; const int32_t *G = cx->G;
  const int32_t *Bk = cx->Bk, *Ck = cx->Ck; const int16_t *exp_lut = cx->exp_lut;
  uint8_t *xt = cx->xt, *su = cx->su, *wenc = cx->wenc; int8_t *w8 = cx->w8; int32_t *meta = cx->meta;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, P = cx->P, N = cx->N, Q = cx->Q;
  const uint32_t PXA = cx->PXA, WSZ = cx->WSZ, MSZ = cx->MSZ, d_inner = cx->d_inner;
  int32_t *lam = cscr + Q * (Q > N ? Q : N); // [Q] per-core, after the matrix scratch
  const uint32_t hh = it >> 2, part = it & 3;
  int32_t *mt = meta + hh * MSZ;
  int32_t *comp1 = mt + SSDN_META_HDR, *comp2 = comp1 + Q, *comp3 = comp2 + N;
  int8_t *W1 = w8 + hh * WSZ, *W2 = W1 + Q * Q, *W3 = W2 + N * Q;
  uint8_t *we = wenc + hh * WSZ;
  const int16_t *dth = dt + (b * L + t0) * NHt + hh; // stride NHt
  const uint32_t ts = cx->decay_mode == 1;
  const uint32_t m3 = cx->m3, m3rot = cx->m3 && cx->m3th != 0;
  const int16_t *m3g = m3 ? cx->m3g + (b * L + t0) * NHt + hh : 0, *m3w = m3 ? cx->m3w + (b * L + t0) * NHt + hh : 0;
  const int32_t *rB = m3rot ? cx->rot + hh * 2 * Q * N : 0, *rC = m3rot ? rB + Q * N : 0;
  const int32_t *Bs = m3rot ? rB : Bk, *Cs = m3rot ? rC : Ck;  // B, C rows used by W2 / W3

  if (ts && part == 3) {
    // two-scale step 1, head hh: rho = cumsum(-asr(R*rmul, 8)); gx[s] = e(rho_Q - rho_s); f1 = 2^30 / gx; f3 = e(rho_t);
    // dec = e(Lam_Q + rho_Q); shs from the per-core state maxima. x~ and S8 are steps 2/3 (spread over the cores).
    int32_t *f1 = comp3 + Q, *f3 = f1 + Q, *gx = f3 + Q;
    const int8_t *Rh = cx->R + (b * L + t0) * NHt + hh;
    int32_t rho = 0, lamQ = 0;
    const int64_t A0 = (int64_t)A[0];
    const int16_t *dta = cx->dta + (b * L + t0);
    for (uint32_t t = 0; t < Q; t++) {
      rho -= (int32_t)(((int32_t)Rh[t * NHt] * cx->resid_mul) >> 8);
      f3[t] = ssdn_expq(exp_lut, rho);
      gx[t] = rho;  // rho_t for now
      lamQ += (int32_t)(((int64_t)dta[t] * A0) >> 8);
    }
    const int32_t rhoQ = gx[Q - 1];
    for (uint32_t t = 0; t < Q; t++) {
      const int32_t g = (t + 1 < Q) ? ssdn_expq(exp_lut, rhoQ - gx[t]) : SSDN_ONE_Q15;
      gx[t] = g;
      f1[t] = (int32_t)(((uint32_t)1 << 30) / (uint32_t)g);
    }
    mt[4] = ssdn_expq(exp_lut, lamQ + rhoQ);
    uint32_t mx = 0;
    for (uint32_t i = 0; i < NUM_CORES; i++) {
      const uint32_t m = (uint32_t)hmax[(b * NHt + hh) * NUM_CORES + i];
      mx = m > mx ? m : mx;
    }
    mt[3] = ssdn_pow2_exp(mx);
    (void)xt; (void)su; (void)h_state;
    return;
  }

  if (part == 3) {
    // state -> S8 (+128): the max |state| was collected by the previous update
    uint32_t mx = 0;
    for (uint32_t i = 0; i < NUM_CORES; i++) {
      const uint32_t m = (uint32_t)hmax[(b * NHt + hh) * NUM_CORES + i];
      mx = m > mx ? m : mx;
    }
    const int shs = ssdn_pow2_exp(mx);
    const int32_t *hs = h_state + (b * NHt + hh) * P * N;
    uint8_t *su_h = su + hh * PXA * N;
    for (uint32_t i = 0; i < P * N; i++)
      su_h[i] = (uint8_t)(ssdn_qshift(hs[i], shs) ^ 0x80);
    mt[3] = shs;
    // x transpose -> xt[hh][c][s] (+128)
    uint8_t *xt_h = xt + hh * PXA * Q;
    for (uint32_t s = 0; s < Q; s++) {
      const uint8_t *xrow = (const uint8_t *)x + (b * L + t0 + s) * d_inner + hh * P;
      uint8_t *dst = xt_h + s;
      for (uint32_t c = 0; c < P; c++)
        dst[c * Q] = xrow[c] ^ 0x80;
    }
    return;
  }

  if (ts) {   // shared log-decay Lam[t] = cumsum asr(dta*A, 8)
    const int64_t A0 = (int64_t)A[0];
    const int16_t *dta = cx->dta + (b * L + t0);
    int32_t running = 0;
    for (uint32_t t = 0; t < Q; t++) {
      running += (int32_t)(((int64_t)dta[t] * A0) >> 8);
      lam[t] = running;
    }
  } else {
    const int64_t A_h = (int64_t)A[hh];
    int32_t running = 0;
    for (uint32_t t = 0; t < Q; t++) {
      running += (int32_t)(((int64_t)dth[t * NHt] * A_h) >> 8);
      lam[t] = running;
    }
  }
  uint32_t mx = 0;
  if (part == 0) {
    // M1[t][s] = ((G[t][s]*dt_s) >> 8) * e_ts >> 15, s <= t
    for (uint32_t t = 0; t < Q; t++) {
      for (uint32_t s = 0; s <= t; s++) {
        const int32_t e = (s == t) ? SSDN_ONE_Q15 : ssdn_expq(exp_lut, lam[t] - lam[s]);
        int64_t Gv = G[t * Q + s];
        if (m3rot) {   // Mamba-3 RoPE: per-head G from the rotated rows
          int64_t d = 0;
          for (uint32_t n = 0; n < N; n++)
            d += (int64_t)rC[t * N + n] * (int64_t)rB[s * N + n];
          Gv = (int32_t)(d >> 15);
        }
        // weight per key: dt (Mamba-2) | 256 (two-scale, exact) | Mamba-3: w off the diagonal, gamma on it
        const int64_t kw = m3 ? (int64_t)(s == t ? m3g[s * NHt] : m3w[s * NHt]) : (ts ? (int64_t)256 : (int64_t)dth[s * NHt]);
        const int32_t Gh = (int32_t)((Gv * kw) >> 8);
        const int32_t m = (int32_t)(((int64_t)Gh * e) >> 15);
        cscr[t * Q + s] = m;
        mx = ssdn_absmax(mx, m);
      }
      for (uint32_t s = t + 1; s < Q; s++)
        cscr[t * Q + s] = 0;
    }
    mt[0] = ssdn_quantise(cscr, Q, Q, mx, W1, comp1, 0);
    ssdn_repack(W1, we, Q, Q);
  } else if (part == 1) {
    // dBd[s][n] = ((dt_s*B[s][n]) >> 8) * e_Qs[s] >> 15  ->  W2[n][s]
    for (uint32_t s = 0; s < Q; s++) {
      const int32_t eQ = (s + 1 < Q) ? ssdn_expq(exp_lut, lam[Q - 1] - lam[s]) : SSDN_ONE_Q15;
      // two-scale: no dt in the weights (x256 >> 8 exact); Mamba-3: the per-key weight w
      const int64_t dts = m3 ? (int64_t)m3w[s * NHt] : (ts ? (int64_t)256 : (int64_t)dth[s * NHt]);
      for (uint32_t n = 0; n < N; n++) {
        const int32_t dB = (int32_t)((dts * (int64_t)Bs[s * N + n]) >> 8);
        const int32_t m = (int32_t)(((int64_t)dB * eQ) >> 15);
        cscr[s * N + n] = m;
        mx = ssdn_absmax(mx, m);
      }
    }
    mt[1] = ssdn_quantise(cscr, Q, N, mx, W2, comp2, 1);
    ssdn_repack(W2, we + Q * Q, N, Q);
  } else {
    // M3[t][n] = (C[t][n]*e_t0[t]) >> 15
    int32_t eQ0 = SSDN_ONE_Q15;
    for (uint32_t t = 0; t < Q; t++) {
      const int32_t e0 = ssdn_expq(exp_lut, lam[t]);
      eQ0 = e0;
      for (uint32_t n = 0; n < N; n++) {
        const int32_t m = (int32_t)(((int64_t)Cs[t * N + n] * e0) >> 15);
        cscr[t * N + n] = m;
        mx = ssdn_absmax(mx, m);
      }
    }
    mt[2] = ssdn_quantise(cscr, Q, N, mx, W3, comp3, 0);
    ssdn_repack(W3, we + Q * Q + N * Q, Q, N);
    if (!ts)
      mt[4] = eQ0;  // two-scale: the per-head data item writes the state decay
    (void)eQ0;
  }

}

// two-scale epilogue (int32 output), own frame so that ssdn_finish_head stays small on the 1 KB slave stacks
static __attribute__((noinline)) void ssdn_epilogue_ts(const ssdn_ctx_t *cx, uint32_t hp, const int32_t *mt,
                                                       const int32_t *a1, const int32_t *a3, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int32_t *y32 = (int32_t *)cx->y; const int32_t *meta = cx->meta;
  const int32_t *gate_lut = cx->gate_lut;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, d_inner = cx->d_inner;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int32_t *comp3 = mt + SSDN_META_HDR + Q + N;
    // two-scale (int32 output): y1 = mul_pow2(asr(acc1*f1, 15), sh1+shx-8), y3 = mul_pow2(asr(acc3*f3, 15), sh3+shs-15)
    const int32_t *m0 = meta;  // shared weights: shifts and +128 compensation live in head 0's metadata
    const int32_t *c1v = m0 + SSDN_META_HDR, *c3v = c1v + Q + N;
    const int32_t *f1 = comp3 + Q, *f3 = f1 + Q;
    const int ex1 = m0[0] + mt[5] - 8, ex3 = m0[2] + mt[3] - 15;
    const int osh = (int)cx->out_shift;
    for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
      const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
      const uint32_t row = (b * L + t0 + t) * d_inner + hp * P;
      const int8_t *xr = x + row, *zr = z + row;
      int32_t *yr = y32 + row;
      const int32_t c1 = c1v[t], c3 = c3v[t];
      const int64_t g1 = f1[t], g3 = f3[t];
      const int32_t *p1 = a1 + t + cb * (Q + N), *p3 = a3 + t + cb * Q;
      for (uint32_t c = cb; c < ce; c++, p1 += Q + N, p3 += Q) {
        const int64_t u1 = ((int64_t)(*p1 - c1) * g1) >> 15;
        const int64_t u3 = ((int64_t)(*p3 - c3) * g3) >> 15;
        const int64_t y_acc = (ex1 >= 0 ? (u1 << ex1) : (u1 >> (-ex1))) + (ex3 >= 0 ? (u3 << ex3) : (u3 >> (-ex3))) +
                              D_h * (int64_t)xr[c];
        const int64_t y_g = ssdn_round_shift(y_acc * (int64_t)gate_lut[(int32_t)zr[c] + 128], 13);
        yr[c] = ssdn_sat_i32(osh > 0 ? ssdn_round_shift(y_g, osh) : y_g);
      }
    }
}

// two-scale step 2 (core `core` of NUM_CORES): per-(head, token) row maxima of v = floor(x*dt*gx / 2^15) into the
// core's own scratch (pm[hh]), and the state -> S8 (+128) quantisation as contiguous ranges
static __attribute__((noinline)) void ssdn_ts_step2(const ssdn_ctx_t *cx, uint32_t core, int32_t *pm) {
  const int8_t *x = cx->x; const int16_t *dt = cx->dt; const int32_t *h_state = cx->h_state, *meta = cx->meta;
  uint8_t *su = cx->su;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, P = cx->P, N = cx->N, Q = cx->Q;
  const uint32_t PXA = cx->PXA, MSZ = cx->MSZ, d_inner = cx->d_inner;
  for (uint32_t hh = 0; hh < NHt; hh++)
    pm[hh] = 0;
  for (uint32_t r = core; r < NHt * Q; r += NUM_CORES) {
    const uint32_t hh = r / Q, s_ = r % Q;
    const int32_t *gx = meta + hh * MSZ + SSDN_META_HDR + 4 * Q + N;
    const int8_t *xrow = x + (b * L + t0 + s_) * d_inner + hh * P;
    // m = dt * gx < 2^12 * 2^15 (Softplus Q8.8 LUT max 3251): floor(x*m / 2^15) = high word of (x << 17) * m
    const int32_t m = (int32_t)dt[(b * L + t0 + s_) * NHt + hh] * gx[s_];
    uint32_t vmx = (uint32_t)pm[hh];
    for (uint32_t c = 0; c < P; c++)
      vmx = ssdn_absmax(vmx, (int32_t)(((int64_t)((int32_t)xrow[c] << 17) * m) >> 32));
    pm[hh] = (int32_t)vmx;
  }
  const uint32_t PN = P * N, tot = NHt * PN, ch = (tot + NUM_CORES - 1) / NUM_CORES;
  const uint32_t i0 = core * ch < tot ? core * ch : tot, i1 = i0 + ch < tot ? i0 + ch : tot;
  for (uint32_t i = i0; i < i1;) {
    const uint32_t hh = i / PN, k0 = i - hh * PN, k1 = (hh + 1) * PN < i1 ? PN : i1 - hh * PN;
    const int shs = meta[hh * MSZ + 3];
    const int32_t *hs = h_state + (b * NHt + hh) * PN;
    uint8_t *su_h = su + hh * PXA * N;
    for (uint32_t k = k0; k < k1; k++)
      su_h[k] = (uint8_t)(ssdn_qshift(hs[k], shs) ^ 0x80);
    i = hh * PN + k1;
  }
}

// two-scale step 3: shx per head from the cores' partial maxima (scratch stride `cstride`), x~ rows over the cores
static __attribute__((noinline)) void ssdn_ts_step3(const ssdn_ctx_t *cx, uint32_t core, const int32_t *pm0,
                                                    uint32_t cstride) {
  const int8_t *x = cx->x; const int16_t *dt = cx->dt; int32_t *meta = cx->meta; uint8_t *xt = cx->xt;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, P = cx->P, N = cx->N, Q = cx->Q;
  const uint32_t PXA = cx->PXA, MSZ = cx->MSZ, d_inner = cx->d_inner;
  for (uint32_t r = core; r < NHt * Q; r += NUM_CORES) {
    const uint32_t hh = r / Q, s_ = r % Q;
    uint32_t vmx = 0;
    for (uint32_t c = 0; c < NUM_CORES; c++)
      vmx = (uint32_t)pm0[c * cstride + hh] > vmx ? (uint32_t)pm0[c * cstride + hh] : vmx;
    const int shx = ssdn_pow2_exp(vmx);
    int32_t *mt = meta + hh * MSZ;
    if (s_ == 0)
      mt[5] = shx;
    const int32_t *gx = mt + SSDN_META_HDR + 4 * Q + N;
    const int8_t *xrow = x + (b * L + t0 + s_) * d_inner + hh * P;
    const int32_t m = (int32_t)dt[(b * L + t0 + s_) * NHt + hh] * gx[s_];
    uint8_t *dst = xt + hh * PXA * Q + s_;
    for (uint32_t c = 0; c < P; c++)
      dst[c * Q] = (uint8_t)(ssdn_qshift((int32_t)(((int64_t)((int32_t)xrow[c] << 17) * m) >> 32), shx) ^ 0x80);
  }
}

// ---- Mamba-3 MIMO (rank R > 1, no RoPE): token axis -> (token, rank) pairs, index t*R + q. W1 [QR][QR], W2 [N][QR],
// W3 [QR][N]; B, C rows [N*R] ordered (n, r); x, z, y columns of head h: h*P*R + c*R + q (ssd_ne16_ref mimo).
// The matrices are quantised in two passes (max, then recompute) instead of through the per-core scratch.
static __attribute__((noinline)) void ssdn_mimo_item(const ssdn_ctx_t *cx, uint32_t it, int32_t *tab, int32_t *lam) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, P = cx->P, N = cx->N, Q = cx->Q, R = cx->Rk;
  const uint32_t QR = Q * R, NR = N * R, PXA = cx->PXA, WSZ = cx->WSZ, MSZ = cx->MSZ, d_inner = cx->d_inner;
  const int16_t *exp_lut = cx->exp_lut;
  const uint32_t hh = it >> 2, part = it & 3;
  int32_t *mt = cx->meta + hh * MSZ;
  int32_t *comp1 = mt + SSDN_META_HDR, *comp2 = comp1 + QR, *comp3 = comp2 + N;
  int8_t *W1 = cx->w8 + hh * WSZ, *W2 = W1 + QR * QR, *W3 = W2 + N * QR;
  uint8_t *we = cx->wenc + hh * WSZ;
  const int16_t *dth = cx->dt + (b * L + t0) * NHt + hh;
  const int16_t *kw = cx->m3 ? cx->m3w + (b * L + t0) * NHt + hh : dth;   // off-diagonal / state weight
  const int16_t *kg = cx->m3 ? cx->m3g + (b * L + t0) * NHt + hh : dth;   // diagonal weight
  if (part == 3) {
    uint32_t mx = 0;
    for (uint32_t i = 0; i < NUM_CORES; i++) {
      const uint32_t m = (uint32_t)cx->hmax[(b * NHt + hh) * NUM_CORES + i];
      mx = m > mx ? m : mx;
    }
    const int shs = ssdn_pow2_exp(mx);
    const int32_t *hs = cx->h_state + (b * NHt + hh) * P * N;
    uint8_t *su_h = cx->su + hh * PXA * N;
    for (uint32_t i = 0; i < P * N; i++)
      su_h[i] = (uint8_t)(ssdn_qshift(hs[i], shs) ^ 0x80);
    mt[3] = shs;
    uint8_t *xt_h = cx->xt + hh * PXA * QR;          // xt[c][(s,r)] = x[s][h*P*R + c*R + r] (+128)
    for (uint32_t s_ = 0; s_ < Q; s_++) {
      const uint8_t *xrow = (const uint8_t *)cx->x + (b * L + t0 + s_) * d_inner + hh * P * R;
      for (uint32_t c = 0; c < P; c++)
        for (uint32_t r = 0; r < R; r++)
          xt_h[c * QR + s_ * R + r] = xrow[c * R + r] ^ 0x80;
    }
    return;
  }
  {
    const int64_t A_h = (int64_t)cx->A[hh];
    int32_t running = 0;
    for (uint32_t t = 0; t < Q; t++) {
      running += (int32_t)(((int64_t)dth[t * NHt] * A_h) >> 8);
      lam[t] = running;
    }
  }
  // decay tables in the core's scratch (tab[0..Q*Q): e_ts (ONE on the diagonal, 0 above); tab[Q*Q..+Q): e_Qs;
  // tab[Q*Q+Q..+Q): e_t0), then two passes: max -> pow2 shift; int8 weights row-major [KO][KI], comp = 128 * row sum
  // the tables (Q*Q + 2Q words) run past the matrix scratch into lam[]: work from a local copy of lam (Q <= 32)
  int32_t lamv[32];
  for (uint32_t t = 0; t < Q; t++)
    lamv[t] = lam[t];
  int32_t *ets = tab, *eQs = tab + Q * Q, *et0 = eQs + Q;
  for (uint32_t t = 0; t < Q; t++) {
    for (uint32_t s_ = 0; s_ < Q; s_++)
      ets[t * Q + s_] = s_ > t ? 0 : (s_ == t ? SSDN_ONE_Q15 : ssdn_expq(exp_lut, lamv[t] - lamv[s_]));
    eQs[t] = (t + 1 < Q) ? ssdn_expq(exp_lut, lamv[Q - 1] - lamv[t]) : SSDN_ONE_Q15;
    et0[t] = ssdn_expq(exp_lut, lamv[t]);
  }
  uint32_t KO, KI;
  int8_t *W; int32_t *comp; uint8_t *wenc_p;
  if (part == 0) { KO = QR; KI = QR; W = W1; comp = comp1; wenc_p = we; }
  else if (part == 1) { KO = N; KI = QR; W = W2; comp = comp2; wenc_p = we + QR * QR; }
  else { KO = QR; KI = N; W = W3; comp = comp3; wenc_p = we + QR * QR + N * QR; }
  uint32_t mx = 0;
  for (int pass = 0; pass < 2; pass++) {
    const int sh = pass ? ssdn_pow2_exp(mx) : 0;
#define SSDN_MIMO_EMIT(v)                                    \
  do {                                                       \
    const int32_t v_ = (v);                                  \
    if (!pass) {                                             \
      mx = ssdn_absmax(mx, v_);                              \
    } else {                                                 \
      const int8_t w8_ = (int8_t)ssdn_qshift(v_, sh);        \
      *wp++ = w8_;                                           \
      csum += w8_;                                           \
    }                                                        \
  } while (0)
    int8_t *wp = W;
    if (part == 0) {        // W1[(t,q)][(s,r)] = asr(asr(G * k, 8) * e_ts, 15), k = w[s] (s < t) | gamma[t] (s == t)
      for (uint32_t t = 0; t < Q; t++)
        for (uint32_t q = 0; q < R; q++) {
          int32_t csum = 0;
          const int32_t *Grow = cx->G + (t * R + q) * QR;
          for (uint32_t s_ = 0; s_ < Q; s_++) {
            const int32_t e = ets[t * Q + s_];
            const int64_t k = (s_ == t) ? (int64_t)kg[t * NHt] : (int64_t)kw[s_ * NHt];
            for (uint32_t r = 0; r < R; r++) {
              const int32_t Gh = (int32_t)(((int64_t)Grow[s_ * R + r] * k) >> 8);
              SSDN_MIMO_EMIT(e ? (int32_t)(((int64_t)Gh * e) >> 15) : 0);
            }
          }
          if (pass) comp[t * R + q] = 128 * csum;
        }
    } else if (part == 1) { // W2[n][(s,r)] = asr(asr(w_s B[s][n,r], 8) e_Qs[s], 15)
      for (uint32_t n = 0; n < N; n++) {
        int32_t csum = 0;
        for (uint32_t s_ = 0; s_ < Q; s_++) {
          const int64_t k = (int64_t)kw[s_ * NHt];
          const int32_t *Br = cx->Bk + s_ * NR + n * R;
          for (uint32_t r = 0; r < R; r++) {
            const int32_t dB = (int32_t)((k * (int64_t)Br[r]) >> 8);
            SSDN_MIMO_EMIT((int32_t)(((int64_t)dB * eQs[s_]) >> 15));
          }
        }
        if (pass) comp[n] = 128 * csum;
      }
    } else {                // W3[(t,q)][n] = asr(C[t][n,q] e(lam_t), 15)
      for (uint32_t t = 0; t < Q; t++)
        for (uint32_t q = 0; q < R; q++) {
          int32_t csum = 0;
          const int32_t *Cr = cx->Ck + t * NR + q;
          for (uint32_t n = 0; n < N; n++)
            SSDN_MIMO_EMIT((int32_t)(((int64_t)Cr[n * R] * et0[t]) >> 15));
          if (pass) comp[t * R + q] = 128 * csum;
        }
    }
#undef SSDN_MIMO_EMIT
    if (pass)
      mt[part] = sh;
  }
  if (part == 2)
    mt[4] = et0[Q - 1];
  ssdn_repack(W, wenc_p, KO, KI);
}

// ---- MIMO phase A over all cores: step 0 decay tables per head; step 1 row maxima (W1 rows, W2 rows, W3 rows) and the
// data (S8, xt); step 2 per-matrix pow2 shift, quantise + repack each row. Same values as ssdn_mimo_item.
static inline uint32_t ssdn_mimo_tabsz(uint32_t Q, uint32_t QR, uint32_t N) { return Q * Q + 2 * Q + 2 * QR + N; }

static __attribute__((noinline)) void ssdn_mimo_tables(const ssdn_ctx_t *cx, uint32_t hh) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, N = cx->N, Q = cx->Q, QR = Q * cx->Rk;
  int32_t *tab = cx->mtab + hh * ssdn_mimo_tabsz(Q, QR, N);
  int32_t *ets = tab, *eQs = tab + Q * Q, *et0 = eQs + Q;
  const int16_t *dth = cx->dt + (b * L + t0) * NHt + hh;
  const int64_t A_h = (int64_t)cx->A[hh];
  int32_t lamv[32], running = 0;
  for (uint32_t t = 0; t < Q; t++) {
    running += (int32_t)(((int64_t)dth[t * NHt] * A_h) >> 8);
    lamv[t] = running;
  }
  for (uint32_t t = 0; t < Q; t++) {
    for (uint32_t s_ = 0; s_ < Q; s_++)
      ets[t * Q + s_] = s_ > t ? 0 : (s_ == t ? SSDN_ONE_Q15 : ssdn_expq(cx->exp_lut, lamv[t] - lamv[s_]));
    eQs[t] = (t + 1 < Q) ? ssdn_expq(cx->exp_lut, lamv[Q - 1] - lamv[t]) : SSDN_ONE_Q15;
    et0[t] = ssdn_expq(cx->exp_lut, lamv[t]);
  }
}

// values of one row: mat 0 = W1 row (t,q) [QR], 1 = W2 row n [QR], 2 = W3 row (t,q) [N]
static inline uint32_t ssdn_mimo_rowvals(const ssdn_ctx_t *cx, uint32_t hh, uint32_t mat, uint32_t ko, int32_t *v) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, N = cx->N, Q = cx->Q, R = cx->Rk;
  const uint32_t QR = Q * R, NR = N * R;
  const int32_t *tab = cx->mtab + hh * ssdn_mimo_tabsz(Q, QR, N);
  const int32_t *ets = tab, *eQs = tab + Q * Q, *et0 = eQs + Q;
  const int16_t *dth = cx->dt + (b * L + t0) * NHt + hh;
  const int16_t *kw = cx->m3 ? cx->m3w + (b * L + t0) * NHt + hh : dth;
  const int16_t *kg = cx->m3 ? cx->m3g + (b * L + t0) * NHt + hh : dth;
  uint32_t i = 0;
  if (mat == 0) {
    const uint32_t t = ko / R;
    const int32_t *Grow = (cx->Gph ? cx->Gph + hh * QR * QR : cx->G) + ko * QR;
    for (uint32_t s_ = 0; s_ < Q; s_++) {
      const int32_t e = ets[t * Q + s_];
      const int64_t k = (s_ == t) ? (int64_t)kg[t * NHt] : (int64_t)kw[s_ * NHt];
      for (uint32_t r = 0; r < R; r++, i++)
        v[i] = e ? (int32_t)(((int64_t)(int32_t)(((int64_t)Grow[s_ * R + r] * k) >> 8) * e) >> 15) : 0;
    }
  } else if (mat == 1) {
    for (uint32_t s_ = 0; s_ < Q; s_++) {
      const int64_t k = (int64_t)kw[s_ * NHt];
      const int32_t *Br = (cx->rotm ? cx->rotm + hh * 2 * Q * NR : cx->Bk) + s_ * NR + ko * R;
      for (uint32_t r = 0; r < R; r++, i++)
        v[i] = (int32_t)(((int64_t)(int32_t)((k * (int64_t)Br[r]) >> 8) * eQs[s_]) >> 15);
    }
  } else {
    const uint32_t t = ko / R, q = ko - t * R;
    const int32_t *Cr = (cx->rotm ? cx->rotm + hh * 2 * Q * NR + Q * NR : cx->Ck) + t * NR + q;
    for (uint32_t n = 0; n < N; n++, i++)
      v[i] = (int32_t)(((int64_t)Cr[n * R] * et0[t]) >> 15);
  }
  return i;
}

// step 1 unit u of [0, NHt * (2QR + N)) rows, then the data units
static __attribute__((noinline)) void ssdn_mimo_step1(const ssdn_ctx_t *cx, uint32_t core) {
  const uint32_t NHt = cx->NHt, N = cx->N, Q = cx->Q, R = cx->Rk, QR = Q * R, P = cx->P, b = cx->b, L = cx->L, t0 = cx->t0;
  const uint32_t rows = 2 * QR + N, TS = ssdn_mimo_tabsz(Q, QR, N);
  int32_t v[64];
  for (uint32_t u = core; u < NHt * rows; u += NUM_CORES) {
    const uint32_t hh = u / rows, rr = u - hh * rows;
    const uint32_t mat = rr < QR ? 0 : (rr < QR + N ? 1 : 2), ko = mat == 0 ? rr : (mat == 1 ? rr - QR : rr - QR - N);
    const uint32_t n = ssdn_mimo_rowvals(cx, hh, mat, ko, v);
    uint32_t mx = 0;
    for (uint32_t i = 0; i < n; i++)
      mx = ssdn_absmax(mx, v[i]);
    cx->mtab[hh * TS + Q * Q + 2 * Q + rr] = (int32_t)mx;
  }
  // data: S8 rows (channels) and xt rows (channels) per head, channels over the cores
  for (uint32_t u = core; u < NHt * P; u += NUM_CORES) {
    const uint32_t hh = u / P, c = u - hh * P;
    uint32_t mx = 0;
    for (uint32_t i = 0; i < NUM_CORES; i++) {
      const uint32_t m = (uint32_t)cx->hmax[(b * NHt + hh) * NUM_CORES + i];
      mx = m > mx ? m : mx;
    }
    const int shs = ssdn_pow2_exp(mx);
    if (c == 0)
      cx->meta[hh * cx->MSZ + 3] = shs;
    const int32_t *hs = cx->h_state + (b * NHt + hh) * P * N + c * N;
    uint8_t *su_r = cx->su + hh * cx->PXA * N + c * N;
    for (uint32_t k = 0; k < N; k++)
      su_r[k] = (uint8_t)(ssdn_qshift(hs[k], shs) ^ 0x80);
    uint8_t *xt_r = cx->xt + hh * cx->PXA * QR + c * QR;
    const uint8_t *xs = (const uint8_t *)cx->x + (b * L + t0) * cx->d_inner + hh * P * R + c * R;
    for (uint32_t s_ = 0; s_ < Q; s_++)
      for (uint32_t r = 0; r < R; r++)
        xt_r[s_ * R + r] = xs[s_ * cx->d_inner + r] ^ 0x80;
  }
}

// step 2: shifts from the row maxima; quantise + comp + repack each row
static __attribute__((noinline)) void ssdn_mimo_step2(const ssdn_ctx_t *cx, uint32_t core) {
  const uint32_t NHt = cx->NHt, N = cx->N, Q = cx->Q, R = cx->Rk, QR = Q * R, WSZ = cx->WSZ, MSZ = cx->MSZ;
  const uint32_t rows = 2 * QR + N, TS = ssdn_mimo_tabsz(Q, QR, N);
  int32_t v[64];
  for (uint32_t u = core; u < NHt * rows; u += NUM_CORES) {
    const uint32_t hh = u / rows, rr = u - hh * rows;
    const uint32_t mat = rr < QR ? 0 : (rr < QR + N ? 1 : 2), ko = mat == 0 ? rr : (mat == 1 ? rr - QR : rr - QR - N);
    const int32_t *rmx = cx->mtab + hh * TS + Q * Q + 2 * Q;
    const uint32_t r0 = mat == 0 ? 0 : (mat == 1 ? QR : QR + N), r1 = mat == 0 ? QR : (mat == 1 ? QR + N : rows);
    uint32_t mx = 0;
    for (uint32_t i = r0; i < r1; i++)
      mx = (uint32_t)rmx[i] > mx ? (uint32_t)rmx[i] : mx;
    const int sh = ssdn_pow2_exp(mx);
    int32_t *mt = cx->meta + hh * MSZ;
    if (ko == 0)
      mt[mat] = sh;
    const uint32_t KI = mat == 2 ? N : QR;
    int8_t *W = cx->w8 + hh * WSZ + (mat == 0 ? 0 : (mat == 1 ? QR * QR : QR * QR + N * QR)) + ko * KI;
    int32_t *comp = mt + SSDN_META_HDR + (mat == 0 ? 0 : (mat == 1 ? QR : QR + N));
    const uint32_t n = ssdn_mimo_rowvals(cx, hh, mat, ko, v);
    int32_t csum = 0;
    for (uint32_t i = 0; i < n; i++) {
      const int8_t w8 = (int8_t)ssdn_qshift(v[i], sh);
      W[i] = w8;
      csum += w8;
    }
    comp[ko] = 128 * csum;
    ssdn_repack(W, cx->wenc + hh * WSZ + (mat == 0 ? 0 : (mat == 1 ? QR * QR : QR * QR + N * QR)) + ko * (KI / 16) * 16, 1, KI);
    if (mat == 2 && ko == 0) {
      const int32_t *et0 = cx->mtab + hh * TS + Q * Q + Q;
      mt[4] = et0[Q - 1];
    }
  }
}

// ---- trained Mamba-3 (bc_norm) preparation, per chunk, over all cores (m3_ref.bc_norm + per-head bias + RoPE):
// step a: Bn/Cn rows (token, rank): rt = isqrt((sum_n v^2 << 16) / N) (rms * 2^8), inv = 2^38 / rt,
//         out = rs(v * W[n,r] * inv, 30) (2^-10 units);  step b: per (head, token) rows + bias, rotated by the head's
//         accumulated angle;  step c: per-head G rows (t,q) and the carried angle.
static inline uint32_t ssdn_isqrt64(uint64_t x) {
  uint64_t r = 0, bit = (uint64_t)1 << 62;
  while (bit > x)
    bit >>= 2;
  while (bit) {
    if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else { r >>= 1; }
    bit >>= 2;
  }
  return (uint32_t)r;
}

static inline uint32_t ssdn_isqrt32(uint32_t x) {
  uint32_t r = 0, bit = 1u << 30;
  while (bit > x)
    bit >>= 2;
  while (bit) {
    if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else { r >>= 1; }
    bit >>= 2;
  }
  return r;
}

static __attribute__((noinline)) void ssdn_bcn_stepa(const ssdn_ctx_t *cx, uint32_t core) {
  const uint32_t Q = cx->Q, N = cx->N, R = cx->Rk, NR = N * R;
  for (uint32_t u = core; u < 2 * Q * R; u += NUM_CORES) {
    const uint32_t which = u / (Q * R), tr = u - which * Q * R, t = tr / R, r = tr - t * R;
    const int8_t *v = (which ? cx->Cp8 : cx->Bp8) + t * NR + r;
    const int32_t *W = (which ? cx->Cw : cx->Bw) + r;
    int32_t *o = cx->bn + which * Q * NR + t * NR + r;
    uint32_t ss = 0;   // <= N * 2^14
    for (uint32_t n = 0; n < N; n++)
      ss += (uint32_t)((int32_t)v[n * R] * (int32_t)v[n * R]);
    // (ss << 16) / N <= 2^30: a 32-bit shift for power-of-two N
    uint32_t rt = (N & (N - 1)) == 0 ? ssdn_isqrt32(ss << (16 - __builtin_ctz(N))) : ssdn_isqrt64(((uint64_t)ss << 16) / N);
    if (rt < 1)
      rt = 1;
    const int64_t inv = ((int64_t)1 << 38) / rt;
    for (uint32_t n = 0; n < N; n++) {
      const int64_t tv = (int64_t)((int32_t)v[n * R] * W[n * R]) * inv;   // |v * W| < 2^25
      o[n * R] = (int32_t)(tv >= 0 ? (tv + ((int64_t)1 << 29)) >> 30 : -(((-tv) + ((int64_t)1 << 29)) >> 30));
    }
  }
}

static __attribute__((noinline)) void ssdn_bcn_stepb(const ssdn_ctx_t *cx, uint32_t core) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, Q = cx->Q, N = cx->N, R = cx->Rk, NR = N * R;
  for (uint32_t u = core; u < NHt * Q; u += NUM_CORES) {
    const uint32_t hh = u / Q, t = u - hh * Q;
    int32_t th = cx->thstate[b * NHt + hh];
    if (cx->m3th)
      for (uint32_t k = 0; k <= t; k++)
        th += (int32_t)cx->m3th[(b * L + t0 + k) * NHt + hh];
    const uint32_t idx = ((uint32_t)th >> 8) & 255u;
    const int64_t c = cx->m3th ? ssdn_rot_cos[idx] : 32767, sn = cx->m3th ? ssdn_rot_sin[idx] : 0;
    for (uint32_t which = 0; which < 2; which++) {
      const int32_t *src = cx->bn + which * Q * NR + t * NR, *bias = (which ? cx->Cb : cx->Bb) + hh * NR;
      int32_t *dst = cx->rotm + hh * 2 * Q * NR + which * Q * NR + t * NR;
      if (!cx->m3th) {
        for (uint32_t k = 0; k < NR; k++)
          dst[k] = src[k] + bias[k];
        continue;
      }
      for (uint32_t r = 0; r < R; r++)
        for (uint32_t n = 0; n + 1 < N; n += 2) {
          const int64_t e = src[n * R + r] + bias[n * R + r], o = src[(n + 1) * R + r] + bias[(n + 1) * R + r];
          dst[n * R + r] = (int32_t)((c * e + sn * o) >> 15);
          dst[(n + 1) * R + r] = (int32_t)((-sn * e + c * o) >> 15);
        }
    }
  }
}

static __attribute__((noinline)) void ssdn_bcn_stepc(const ssdn_ctx_t *cx, uint32_t core) {
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, NHt = cx->NHt, Q = cx->Q, N = cx->N, R = cx->Rk;
  const uint32_t NR = N * R, QR = Q * R;
  for (uint32_t u = core; u < NHt * QR; u += NUM_CORES) {
    const uint32_t hh = u / QR, row = u - hh * QR, t = row / R, q = row - t * R;
    const int32_t *Cr = cx->rotm + hh * 2 * Q * NR + Q * NR + t * NR + q;
    int32_t *Gr = cx->Gph + hh * QR * QR + row * QR;
    for (uint32_t s_ = 0; s_ <= t; s_++)
      for (uint32_t r = 0; r < R; r++) {
        const int32_t *Br = cx->rotm + hh * 2 * Q * NR + s_ * NR + r;
        int64_t d = 0;
        for (uint32_t n = 0; n < N; n++)
          d += (int64_t)Cr[n * R] * (int64_t)Br[n * R];
        Gr[s_ * R + r] = (int32_t)(d >> 15);
      }
  }
  if (core == 0 && cx->m3th)   // carry the angle (everybody has read it in step b)
    for (uint32_t hh = 0; hh < NHt; hh++) {
      int32_t th = cx->thstate[b * NHt + hh];
      for (uint32_t k = 0; k < Q; k++)
        th += (int32_t)cx->m3th[(b * L + t0 + k) * NHt + hh];
      cx->thstate[b * NHt + hh] = th;
    }
}

// MIMO epilogue, int32 output (trained Mamba-3 -> RMSNormI32): y = sat32(rs(rs(y_acc * gate, 13), out_shift))
static __attribute__((noinline)) void ssdn_epi_mimo_out32(const ssdn_ctx_t *cx, uint32_t hp, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int32_t *y32 = (int32_t *)cx->y; const int32_t *gate_lut = cx->gate_lut;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, R = cx->Rk, d_inner = cx->d_inner;
  const uint32_t QR = Q * R;
  const int32_t *mt = cx->meta + hp * cx->MSZ;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a3 = a1 + cx->PXA * (QR + N);
  const int32_t *comp1 = mt + SSDN_META_HDR, *comp3 = comp1 + QR + N;
  const int sh1 = mt[0], e3 = mt[2] + mt[3] - 15;
  const int l1 = sh1 > 0 ? sh1 : 0, r1 = sh1 < 0 ? -sh1 : 0;
  const int l3 = e3 > 0 ? e3 : 0, r3 = e3 < 0 ? -e3 : 0;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int osh = (int)cx->out_shift;
  // (32-bit fast paths with 64-bit fallbacks were not faster: 6.36 M vs 6.27 M on b0f.ssd)
  for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
    const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
    const uint32_t row = (b * L + t0 + t) * d_inner + hp * P * R;
    for (uint32_t c = cb; c < ce; c++) {
      const int32_t *p1 = a1 + c * (QR + N) + t * R, *p3 = a3 + c * QR + t * R;
      for (uint32_t q = 0; q < R; q++) {
        const uint32_t col = row + c * R + q;
        const int64_t v1 = (int64_t)(p1[q] - comp1[t * R + q]);
        const int64_t v3 = (int64_t)(p3[q] - comp3[t * R + q]);
        const int64_t y_acc = ((v1 << l1) >> r1) + ((v3 << l3) >> r3) + D_h * (int64_t)x[col];
        const int64_t y_g = ssdn_round_shift(y_acc * (int64_t)gate_lut[(int32_t)z[col] + 128], 13);
        y32[col] = ssdn_sat_i32(osh > 0 ? ssdn_round_shift(y_g, osh) : y_g);
      }
    }
  }
}

// MIMO epilogue v2 (int8 out): ssd_ne16_ref mimo; flattened (t, c) range, inner rank q
static __attribute__((noinline)) void ssdn_epi_mimo_v2(const ssdn_ctx_t *cx, uint32_t hp, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int8_t *y = (int8_t *)cx->y;
  const int16_t *gm16 = cx->gm16; const int32_t ep2_SG = cx->ep2_SG;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, R = cx->Rk, d_inner = cx->d_inner;
  const uint32_t QR = Q * R;
  const int32_t *mt = cx->meta + hp * cx->MSZ;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a3 = a1 + cx->PXA * (QR + N);
  const int32_t *comp1 = mt + SSDN_META_HDR, *comp3 = comp1 + QR + N;
  const int sh1 = mt[0], e3 = mt[2] + mt[3] - 15;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int64_t b1a = ((int64_t)1 << 18) * (QR > 16 ? QR : 16) / 16;
  const int64_t b1 = sh1 >= 0 ? b1a << sh1 : b1a >> (-sh1);
  const int64_t bN = ((int64_t)1 << 18) * (N > 16 ? N : 16) / 16;
  const int64_t b3 = e3 >= 0 ? bN << e3 : bN >> (-e3);
  const int64_t bound = b1 + b3 + (D_h < 0 ? -D_h : D_h) * 128;
  int bl = 0;
  while (bl < 64 && (bound >> bl) != 0)
    bl++;
  const int ysh = bl > 15 ? bl - 15 : 0;
  const int S = 53 - ep2_SG - ysh;
  const int sr = S > 0 ? S : 0, sl = S < 0 ? -S : 0;
  const int32_t half = sr > 0 ? (1 << (sr - 1)) : 0;
  const int s1 = ysh - sh1, s3 = ysh - e3;
  const int32_t h1 = s1 > 0 ? (1 << (s1 - 1)) : 0, h3 = s3 > 0 ? (1 << (s3 - 1)) : 0;
  const int32_t hd = ysh > 0 ? (1 << (ysh - 1)) : 0;
  const int32_t D32 = (int32_t)D_h;
  for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
    const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
    const uint32_t row = (b * L + t0 + t) * d_inner + hp * P * R;
    for (uint32_t c = cb; c < ce; c++) {
      const int32_t *p1 = a1 + c * (QR + N) + t * R, *p3 = a3 + c * QR + t * R;
      for (uint32_t q = 0; q < R; q++) {
        const uint32_t col = row + c * R + q;
        const int32_t v1 = p1[q] - comp1[t * R + q];
        const int32_t v3 = p3[q] - comp3[t * R + q];
        const int32_t dx = D32 * (int32_t)x[col];
        const int32_t t1 = s1 > 0 ? (v1 >= 0 ? ((v1 + h1) >> s1) : -(((-v1) + h1) >> s1)) : v1;
        const int32_t t3 = s3 > 0 ? (v3 >= 0 ? ((v3 + h3) >> s3) : -(((-v3) + h3) >> s3)) : v3;
        const int32_t td = ysh > 0 ? (dx >= 0 ? ((dx + hd) >> ysh) : -(((-dx) + hd) >> ysh)) : dx;
        const int32_t prod = (t1 + t3 + td) * (int32_t)gm16[(int32_t)z[col] + 128];
        int32_t yv;
        if (sr > 0)
          yv = prod >= 0 ? ((prod + half) >> sr) : -(((-prod) + half) >> sr);
        else
          yv = prod << sl;
        y[col] = (int8_t)(yv > 127 ? 127 : (yv < -128 ? -128 : yv));
      }
    }
  }
}

// y epilogue variants (int8 v2 / int32 / int8 v1), rows t of head hp in [e0, e1) of the flattened (t, c) range
static __attribute__((noinline)) void ssdn_epi_v2(const ssdn_ctx_t *cx, uint32_t hp, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int8_t *y = (int8_t *)cx->y; int32_t *y32 = (int32_t *)cx->y;
  const int32_t *gate_lut = cx->gate_lut; const int16_t *gm16 = cx->gm16; const int32_t ep2_SG = cx->ep2_SG;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, d_inner = cx->d_inner;
  const int32_t *mt = cx->meta + hp * cx->MSZ;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a3 = a1 + cx->PXA * (Q + N);
  const int32_t *comp1 = mt + SSDN_META_HDR, *comp3 = comp1 + Q + N;
  const int sh1 = mt[0], e3 = mt[2] + mt[3] - 15;
  const int l1 = sh1 > 0 ? sh1 : 0, r1 = sh1 < 0 ? -sh1 : 0;
  const int l3 = e3 > 0 ? e3 : 0, r3 = e3 < 0 ? -e3 : 0;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int64_t mul = (int64_t)cx->mul_q40;
  (void)y; (void)y32; (void)gm16; (void)ep2_SG; (void)l1; (void)r1; (void)l3; (void)r3; (void)mul; (void)gate_lut;
    // v2: y = sat8(rs(rs(y_acc, ysh) * gm16[z], 53 - SG - ysh)), ysh from a static bound on |y_acc|
    const int64_t b1 = sh1 >= 0 ? ((int64_t)1 << 18) << sh1 : ((int64_t)1 << 18) >> (-sh1);
    const int64_t bN = ((int64_t)1 << 18) * (N > 16 ? N : 16) / 16;
    const int64_t b3 = e3 >= 0 ? bN << e3 : bN >> (-e3);
    const int64_t bound = b1 + b3 + (D_h < 0 ? -D_h : D_h) * 128;
    int bl = 0;
    while (bl < 64 && (bound >> bl) != 0)
      bl++;
    const int ysh = bl > 15 ? bl - 15 : 0;
    const int S = 53 - ep2_SG - ysh;
    const int sr = S > 0 ? S : 0, sl = S < 0 ? -S : 0;
    const int32_t half = sr > 0 ? (1 << (sr - 1)) : 0;
    // the three y_acc terms are rounded separately (ysh >= sh1, e3 by construction): all 32-bit
    const int s1 = ysh - sh1, s3 = ysh - e3;
    const int32_t h1 = s1 > 0 ? (1 << (s1 - 1)) : 0, h3 = s3 > 0 ? (1 << (s3 - 1)) : 0;
    const int32_t hd = ysh > 0 ? (1 << (ysh - 1)) : 0;
    const int32_t D32 = (int32_t)D_h;
    for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
      const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
      const uint32_t row = (b * L + t0 + t) * d_inner + hp * P;
      const int8_t *xr = x + row, *zr = z + row;
      int8_t *yr = y + row;
      const int32_t c1 = comp1[t], c3 = comp3[t];
      const int32_t *p1 = a1 + t + cb * (Q + N), *p3 = a3 + t + cb * Q;
      for (uint32_t c = cb; c < ce; c++, p1 += Q + N, p3 += Q) {
        const int32_t v1 = *p1 - c1;
        const int32_t v3 = *p3 - c3;
        const int32_t dx = D32 * (int32_t)xr[c];
        const int32_t t1 = s1 > 0 ? (v1 >= 0 ? ((v1 + h1) >> s1) : -(((-v1) + h1) >> s1)) : v1;
        const int32_t t3 = s3 > 0 ? (v3 >= 0 ? ((v3 + h3) >> s3) : -(((-v3) + h3) >> s3)) : v3;
        const int32_t td = ysh > 0 ? (dx >= 0 ? ((dx + hd) >> ysh) : -(((-dx) + hd) >> ysh)) : dx;
        const int32_t y16 = t1 + t3 + td;
        const int32_t prod = y16 * (int32_t)gm16[(int32_t)zr[c] + 128];
        int32_t yv;
        if (sr > 0)
          yv = prod >= 0 ? ((prod + half) >> sr) : -(((-prod) + half) >> sr);
        else
          yv = prod << sl;
        yr[c] = (int8_t)(yv > 127 ? 127 : (yv < -128 ? -128 : yv));
      }
    }
}

static __attribute__((noinline)) void ssdn_epi_out32(const ssdn_ctx_t *cx, uint32_t hp, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int8_t *y = (int8_t *)cx->y; int32_t *y32 = (int32_t *)cx->y;
  const int32_t *gate_lut = cx->gate_lut; const int16_t *gm16 = cx->gm16; const int32_t ep2_SG = cx->ep2_SG;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, d_inner = cx->d_inner;
  const int32_t *mt = cx->meta + hp * cx->MSZ;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a3 = a1 + cx->PXA * (Q + N);
  const int32_t *comp1 = mt + SSDN_META_HDR, *comp3 = comp1 + Q + N;
  const int sh1 = mt[0], e3 = mt[2] + mt[3] - 15;
  const int l1 = sh1 > 0 ? sh1 : 0, r1 = sh1 < 0 ? -sh1 : 0;
  const int l3 = e3 > 0 ? e3 : 0, r3 = e3 < 0 ? -e3 : 0;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int64_t mul = (int64_t)cx->mul_q40;
  (void)y; (void)y32; (void)gm16; (void)ep2_SG; (void)l1; (void)r1; (void)l3; (void)r3; (void)mul; (void)gate_lut;
    // wide output for a following RMSNormI32: y32 = sat32(rs(y_g, out_shift)), no Q40 requant
    // (a variant with the shifts built from 32-bit pieces was not faster: 7.24 M vs 7.19 M on b0f.ssd)
    const int osh = (int)cx->out_shift;
    for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
      const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
      const uint32_t row = (b * L + t0 + t) * d_inner + hp * P;
      const int8_t *xr = x + row, *zr = z + row;
      int32_t *yr = y32 + row;
      const int32_t c1 = comp1[t], c3 = comp3[t];
      const int32_t *p1 = a1 + t + cb * (Q + N), *p3 = a3 + t + cb * Q;
      for (uint32_t c = cb; c < ce; c++, p1 += Q + N, p3 += Q) {
        const int64_t v1 = (int64_t)(*p1 - c1);
        const int64_t v3 = (int64_t)(*p3 - c3);
        const int64_t y_acc = ((v1 << l1) >> r1) + ((v3 << l3) >> r3) + D_h * (int64_t)xr[c];
        const int64_t y_g = ssdn_round_shift(y_acc * (int64_t)gate_lut[(int32_t)zr[c] + 128], 13);
        yr[c] = ssdn_sat_i32(osh > 0 ? ssdn_round_shift(y_g, osh) : y_g);
      }
    }
}

static __attribute__((noinline)) void ssdn_epi_v1(const ssdn_ctx_t *cx, uint32_t hp, uint32_t e0, uint32_t e1) {
  const int8_t *x = cx->x, *z = cx->z; int8_t *y = (int8_t *)cx->y; int32_t *y32 = (int32_t *)cx->y;
  const int32_t *gate_lut = cx->gate_lut; const int16_t *gm16 = cx->gm16; const int32_t ep2_SG = cx->ep2_SG;
  const uint32_t b = cx->b, L = cx->L, t0 = cx->t0, P = cx->P, N = cx->N, Q = cx->Q, d_inner = cx->d_inner;
  const int32_t *mt = cx->meta + hp * cx->MSZ;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a3 = a1 + cx->PXA * (Q + N);
  const int32_t *comp1 = mt + SSDN_META_HDR, *comp3 = comp1 + Q + N;
  const int sh1 = mt[0], e3 = mt[2] + mt[3] - 15;
  const int l1 = sh1 > 0 ? sh1 : 0, r1 = sh1 < 0 ? -sh1 : 0;
  const int l3 = e3 > 0 ? e3 : 0, r3 = e3 < 0 ? -e3 : 0;
  const int64_t D_h = (int64_t)cx->D_skip[hp];
  const int64_t mul = (int64_t)cx->mul_q40;
  (void)y; (void)y32; (void)gm16; (void)ep2_SG; (void)l1; (void)r1; (void)l3; (void)r3; (void)mul; (void)gate_lut;
  // y epilogue (identical to GAP9_SSDScan_i8_i8), rows t over the cores. y = sat8(rs(y_g*mul, 40)) is
  // exactly 0 for |y_g| <= zth, which skips the second 64-bit multiply (SSD outputs are mostly zero).
  const int64_t zth = (((int64_t)1 << 39) - 1) / mul;
  for (uint32_t t = e0 / P; t < Q && t * P < e1; t++) {
      const uint32_t cb = t * P < e0 ? e0 - t * P : 0, ce = (t + 1) * P > e1 ? e1 - t * P : P;
    const uint32_t row = (b * L + t0 + t) * d_inner + hp * P;
    const int8_t *xr = x + row, *zr = z + row;
    int8_t *yr = y + row;
    const int32_t c1 = comp1[t], c3 = comp3[t];
    const int32_t *p1 = a1 + t + cb * (Q + N), *p3 = a3 + t + cb * Q;
    for (uint32_t c = cb; c < ce; c++, p1 += Q + N, p3 += Q) {
      const int64_t v1 = (int64_t)(*p1 - c1);
      const int64_t v3 = (int64_t)(*p3 - c3);
      const int64_t y_acc = ((v1 << l1) >> r1) + ((v3 << l3) >> r3) + D_h * (int64_t)xr[c];
      const int64_t gate = (int64_t)gate_lut[(int32_t)zr[c] + 128];
      const int64_t y_g = ssdn_round_shift(y_acc * gate, 13);
      const int64_t ag = y_g < 0 ? -y_g : y_g;
      yr[c] = (ag <= zth) ? (int8_t)0 : ssdn_sat_i8(ssdn_round_shift(y_g * mul, SSDN_OUT_SHIFT));
    }
  }
}

// finish head hh-1: y epilogue + state update (own frame)
// work split: worker wi of nw (core 0 sits out while it dispatches the next head's NE16 jobs); outputs and state
// elements are split as contiguous ranges of the flattened (row, col) / (channel, state) index
static __attribute__((noinline)) void ssdn_finish_head(const ssdn_ctx_t *cx, uint32_t hh, uint32_t wi, uint32_t nw) {
  int32_t *h_state = cx->h_state; int32_t *hmax = cx->hmax; const int32_t *meta = cx->meta;
  const uint32_t b = cx->b, NHt = cx->NHt, P = cx->P, N = cx->N, Q = cx->Q;
  const uint32_t epilogue_version = cx->epilogue_version;
  const uint32_t hp = hh - 1;
  const uint32_t QP = Q * P, ech = (QP + nw - 1) / nw;
  const uint32_t e0 = wi * ech < QP ? wi * ech : QP, e1 = e0 + ech < QP ? e0 + ech : QP;
  const uint32_t PN = P * N, sch = (PN + nw - 1) / nw;
#if SSDN_PROFILE
  const uint32_t pf0 = pi_perf_read(PI_PERF_CYCLES);
#endif
  const uint32_t s0 = wi * sch < PN ? wi * sch : PN, s1 = s0 + sch < PN ? s0 + sch : PN;
  // acc slot: [px][QR + N] (acc1 | acc2 interleaved per pixel, one NE16 job) then acc3 [px][QR]; QR = Q * rank
  const uint32_t QR = Q * cx->Rk;
  const int32_t *a1 = cx->acc + (hp & 1) * cx->ASZ, *a2 = a1 + QR, *a3 = a1 + cx->PXA * (QR + N);
  const int32_t *mt = meta + hp * cx->MSZ;
  const int ts = cx->decay_mode == 1;
  // two-scale: W2's shift and +128 compensation are shared (head 0); acc2 is in units of the head's data scale
  const int sh2 = ts ? meta[1] + mt[5] - 8 : mt[1];
  const int32_t *comp2s = ts ? meta + SSDN_META_HDR + Q : mt + SSDN_META_HDR + QR;
  const int l2 = sh2 > 0 ? sh2 : 0, r2 = sh2 < 0 ? -sh2 : 0;
  const int64_t eQ0 = (int64_t)mt[4];
  // y epilogue: one variant per call, each in its own frame (the 1 KB slave stacks)
  if (cx->Rk > 1 && cx->out_bits == 32)
    ssdn_epi_mimo_out32(cx, hp, e0, e1);
  else if (cx->Rk > 1)
    ssdn_epi_mimo_v2(cx, hp, e0, e1);
  else if (epilogue_version == 2)
    ssdn_epi_v2(cx, hp, e0, e1);
  else if (cx->decay_mode == 1)
    ssdn_epilogue_ts(cx, hp, mt, a1, a3, e0, e1);
  else if (cx->out_bits == 32)
    ssdn_epi_out32(cx, hp, e0, e1);
  else
    ssdn_epi_v1(cx, hp, e0, e1);
#if SSDN_PROFILE
  const uint32_t pf1 = pi_perf_read(PI_PERF_CYCLES);
  if (pi_core_id() == 1) ssdn_p_epi += pf1 - pf0;
#endif
  // state update, channels over the cores; per-core |state| max for the next quantisation
  int32_t *hs = h_state + (b * NHt + hp) * P * N;
  uint32_t mxs = 0;
  for (uint32_t c = s0 / N; c < P && c * N < s1; c++) {
    int32_t *hr = hs + c * N;
    const int32_t *ar = a2 + c * (QR + N);
    const uint32_t nb = c * N < s0 ? s0 - c * N : 0, ne = (c + 1) * N > s1 ? s1 - c * N : N;
    for (uint32_t n = nb; n < ne; n++) {
      const int64_t dec = (eQ0 * (int64_t)hr[n]) >> 15;
      const int64_t sl_ = (((int64_t)(ar[n] - comp2s[n])) << l2) >> r2;
      const int32_t hv = ssdn_sat_i32(dec + sl_);
      hr[n] = hv;
      mxs = ssdn_absmax(mxs, hv);
    }
  }
  hmax[(b * NHt + hp) * NUM_CORES + wi] = (int32_t)mxs;
  if (wi == 0)
    for (uint32_t i = nw; i < NUM_CORES; i++)
      hmax[(b * NHt + hp) * NUM_CORES + i] = 0;
#if SSDN_PROFILE
  if (pi_core_id() == 1) ssdn_p_upd += pi_perf_read(PI_PERF_CYCLES) - pf1;
#endif

}

void GAP9_SSDScanNE16_i8_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z,
                            const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
                            const int32_t *__restrict__ C, const int32_t *__restrict__ A,
                            const int32_t *__restrict__ D_skip, void *__restrict__ y, int32_t *__restrict__ h_state,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t B_size,
                            uint32_t Q, uint32_t N, uint32_t P, uint32_t NHt, uint32_t L, uint32_t GH, uint32_t GW,
                            int32_t output_requant_mul_q40, uint32_t init_state, uint32_t epilogue_version,
                            uint32_t out_bits, uint32_t out_shift, const int16_t *__restrict__ dta,
                            const int8_t *__restrict__ R, uint32_t decay_mode, int32_t resid_mul,
                            const int16_t *__restrict__ m3_gamma, const int16_t *__restrict__ m3_w,
                            const int16_t *__restrict__ m3_theta, uint32_t mamba3, uint32_t mimo_rank,
                            const int32_t *__restrict__ bc_Bw, const int32_t *__restrict__ bc_Cw,
                            const int32_t *__restrict__ bc_Bb, const int32_t *__restrict__ bc_Cb, uint32_t bc_norm) {
  const int core = (int)pi_core_id();
  pi_cl_team_barrier();   // the template's gate LUT copy (split over the cores) is complete
  const uint32_t Rk = mimo_rank ? mimo_rank : 1;   // Mamba-3 MIMO rank: the token axis becomes (token, rank) pairs
  const uint32_t QR = Q * Rk;
  const uint32_t PXA = GH * GW;
  const uint32_t d_inner = NHt * P * Rk;
  const uint32_t WSZ = QR * QR + N * QR + QR * N; // int8 per head: W1 | W2 | W3
  const uint32_t ASZ = PXA * (2 * QR + N);        // int32 per slot: acc1 | acc2 | acc3
  // int32 per head: sh1 sh2 sh3 shs eQ0 shx - - comp1[QR] comp2[N] comp3[QR] (+ two-scale: f1[Q] f3[Q] gx[Q])
  const uint32_t MSZ = SSDN_META_HDR + 2 * QR + N + (decay_mode == 1 ? 3 * Q : 0);
  const uint32_t QM = Q * (Q > N ? Q : N);        // int32 per-core scratch (rank 1; MIMO quantises in two passes)
  const int16_t *exp_lut = SSDScan_exp_lut_ptr;

  uint8_t *xt = scratch;                                    // [NHt][PXA][QR]
  uint8_t *su = xt + NHt * PXA * QR;                        // [NHt][PXA][N]
  int8_t *w8 = (int8_t *)(su + NHt * PXA * N);              // [NHt][WSZ]
  uint8_t *wenc = (uint8_t *)w8 + NHt * WSZ;                // [NHt][WSZ]
  int32_t *acc = (int32_t *)(wenc + NHt * WSZ);             // [SLOTS][ASZ]
  int32_t *meta = acc + (NHt > 1 ? SSDN_SLOTS : 1) * ASZ;  // [NHt][MSZ] (one head per tile: one slot)
  int32_t *G = meta + NHt * MSZ;                            // [QR][QR] (bc_norm: [NHt][QR][QR] per-head G)
  const uint32_t GSZ = QR * QR * (bc_norm ? NHt : 1);
  int32_t *cscr = G + GSZ + (uint32_t)core * (QM + Q);      // [NUM_CORES][QM + Q]: matrix scratch + lam[Q]
  int16_t *gm16 = (int16_t *)(G + GSZ + NUM_CORES * (QM + Q)); // [256] epilogue v2 LUT (gate * mul folded)
  int32_t ep2_SG = 0;
  int32_t *hmax = h_state + B_size * NHt * P * N;           // [B][NHt][NUM_CORES] per-core |state| maxima
  int32_t *thstate = hmax + B_size * NHt * NUM_CORES;       // Mamba-3: [B][NHt] accumulated RoPE angle
  int32_t *rot = (int32_t *)(gm16 + 256);                   // Mamba-3 RoPE: [NHt][2][Q][N] rotated B, C rows
  int32_t *mtab = rot + (mamba3 && m3_theta && Rk == 1 ? NHt * 2 * Q * N : 0);   // MIMO: [NHt][ssdn_mimo_tabsz]
  // bc_norm: normalised rows [2][Q][N*R] live in the per-core scratch (idle before phase A), per-head G in G
  int32_t *bnb = G + GSZ;
  int32_t *rotm = mtab + (Rk > 1 ? NHt * (Q * Q + 2 * Q + 2 * QR + N) : 0);     // bc_norm: [NHt][2][Q][N*R]
  int32_t *Gph = G;

  const ne16_dev_t *dev = ne16_pulp_get_dev();
  static uint32_t ssdn_task_key;   // the task descriptors depend only on (QR, N, GH, GW): set up once per shape
  const uint32_t tkey = (QR << 24) | (N << 16) | (GH << 8) | GW;
  if (core == 0 && ssdn_task_key != tkey) {
    for (uint32_t s = 0; s < SSDN_SLOTS; s++) {
      ssdn_task_setup(&ssdn_tasks[s][0], QR, QR + N, GH, GW);  // J1 | J2: same input, W1 and W2 stacked on Ko
      ssdn_task_setup(&ssdn_tasks[s][2], N, QR, GH, GW);
    }
    ssdn_task_key = tkey;
  }

  if (epilogue_version == 2) {
    // gm16[i] = rs(gate_lut[i] * mul, SG) with SG such that |gm16| < 2^15 (same on every core, no sync needed)
    uint64_t mx = 0;
    for (uint32_t i = 0; i < 256; i++) {
      const int64_t v = (int64_t)gate_lut[i] * (int64_t)output_requant_mul_q40;
      const uint64_t a = (uint64_t)(v < 0 ? -v : v);
      mx = a > mx ? a : mx;
    }
    int bl = 0;
    while (bl < 64 && (mx >> bl) != 0)
      bl++;
    ep2_SG = bl > 15 ? bl - 15 : 0;
    for (uint32_t i = (uint32_t)core; i < 256; i += NUM_CORES) {
      const int64_t v = (int64_t)gate_lut[i] * (int64_t)output_requant_mul_q40;
      gm16[i] = (int16_t)(ep2_SG > 0 ? ssdn_round_shift(v, ep2_SG) : v);
    }
  }
  pi_cl_team_barrier();

  ssdn_ctx_t *const cxp = &ssdn_cx;  // shared, in L1 (not on the 1 KB slave stacks)
  if (core == 0) {
    cxp->x = x;
    cxp->z = z;
    cxp->y = y;
    cxp->dt = dt;
    cxp->A = A;
    cxp->D_skip = D_skip;
    cxp->gate_lut = gate_lut;
    cxp->G = G;
    cxp->h_state = h_state;
    cxp->hmax = hmax;
    cxp->acc = acc;
    cxp->meta = meta;
    cxp->xt = xt;
    cxp->su = su;
    cxp->wenc = wenc;
    cxp->w8 = w8;
    cxp->exp_lut = exp_lut;
    cxp->gm16 = gm16;
    cxp->ep2_SG = ep2_SG;
    cxp->mul_q40 = output_requant_mul_q40;
    cxp->L = L;
    cxp->NHt = NHt;
    cxp->P = P;
    cxp->N = N;
    cxp->Q = Q;
    cxp->PXA = PXA;
    cxp->WSZ = WSZ;
    cxp->ASZ = ASZ;
    cxp->MSZ = MSZ;
    cxp->d_inner = d_inner;
    cxp->epilogue_version = epilogue_version;
    cxp->out_bits = out_bits;
    cxp->out_shift = out_shift;
    cxp->dta = dta;
    cxp->R = R;
    cxp->decay_mode = decay_mode;
    cxp->resid_mul = resid_mul;
    cxp->m3 = mamba3;
    cxp->Rk = Rk;
    cxp->mtab = mtab;
    cxp->bcn = bc_norm;
    cxp->Bw = bc_Bw; cxp->Cw = bc_Cw; cxp->Bb = bc_Bb; cxp->Cb = bc_Cb;
    cxp->bn = bc_norm ? bnb : 0;
    cxp->rotm = bc_norm ? rotm : 0;
    cxp->Gph = bc_norm ? Gph : 0;
    cxp->m3g = m3_gamma;
    cxp->m3w = m3_w;
    cxp->m3th = (mamba3 && m3_theta) ? m3_theta : 0;
    cxp->rot = rot;
    cxp->thstate = thstate;
    if (mamba3 && m3_theta)
      ssdn_rot_init();
  }

  const int number_of_chunks = (int)(L / Q);
#if SSDN_PROFILE
  uint32_t prof_N = 0, prof_G = 0, prof_A = 0, prof_B = 0, pb_disp = 0, pb_res = 0, pb_epi = 0, pb_upd = 0, pb_bar = 0;
  if (core == 0 || core == 1) { pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start(); }
  SSDN_T(prof_t0);
#endif

  for (uint32_t b = 0; b < B_size; b++) {
    if (init_state) {
      int32_t *hb = h_state + b * NHt * P * N;
      const uint32_t n_words = NHt * P * N;
      for (uint32_t i = (uint32_t)core; i < n_words; i += NUM_CORES)
        hb[i] = 0;
      for (uint32_t i = (uint32_t)core; i < NHt * NUM_CORES; i += NUM_CORES)
        hmax[b * NHt * NUM_CORES + i] = 0;
      if (mamba3)
        for (uint32_t i = (uint32_t)core; i < NHt; i += NUM_CORES)
          thstate[b * NHt + i] = 0;
    }
    pi_cl_team_barrier();

    for (int chunk = 0; chunk < number_of_chunks; chunk++) {
      const uint32_t t0 = (uint32_t)chunk * Q;
      const int32_t *Bk = B + (b * L + t0) * N * Rk; // [Q][N*R]
      const int32_t *Ck = C + (b * L + t0) * N * Rk;
      if (core == 0) {   // read after the barrier below
        cxp->b = b; cxp->t0 = t0; cxp->Bk = Bk; cxp->Ck = Ck;
        cxp->Bp8 = (const int8_t *)B + (b * L + t0) * N * Rk;
        cxp->Cp8 = (const int8_t *)C + (b * L + t0) * N * Rk;
      }
      SSDN_T(prof_g0);

      // ---- shared G[t][s] = (C_t . B_s) >> 15, rows over cores (MIMO: G[(t,q)][(s,r)] over the rank pairs) ----
      if (bc_norm) {
        // trained Mamba-3: per-head rows and G (steps a-c below), no shared G
      } else if (Rk > 1) {
        for (uint32_t row = (uint32_t)core; row < QR; row += NUM_CORES) {
          const uint32_t t = row / Rk, q = row - t * Rk;
          const int32_t *Cr = Ck + t * N * Rk + q;
          int32_t *Gr = G + row * QR;
          for (uint32_t s_ = 0; s_ <= t; s_++)            // only s <= t is used (causal)
            for (uint32_t r = 0; r < Rk; r++) {
              const int32_t *Br = Bk + s_ * N * Rk + r;
              int64_t d = 0;
              for (uint32_t n = 0; n < N; n++)
                d += (int64_t)Cr[n * Rk] * (int64_t)Br[n * Rk];
              Gr[s_ * Rk + r] = (int32_t)(d >> 15);
            }
        }
      } else
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

      // ---- phase A: 4 items per head over the cores ----
      // part-major, heaviest part first (3: state quant + x transpose, 1: W2, 2: W3, 0: W1), round-robin over the
      // cores: head-major order gave every core the same part, so cores 3 and 7 did all the heavy items
#if SSDN_PROFILE
      const uint32_t pn0 = pi_perf_read(PI_PERF_CYCLES);
#endif
      if (bc_norm) {   // B/C norm, per-head bias + RoPE rows, per-head G
        ssdn_bcn_stepa(cxp, (uint32_t)core);
        pi_cl_team_barrier();
        ssdn_bcn_stepb(cxp, (uint32_t)core);
        pi_cl_team_barrier();
        ssdn_bcn_stepc(cxp, (uint32_t)core);
        pi_cl_team_barrier();
      }
#if SSDN_PROFILE
      prof_N += pi_perf_read(PI_PERF_CYCLES) - pn0;
#endif
      if (Rk > 1) {
        // MIMO: tables per head; row maxima + data over all cores; quantise + repack rows over all cores
        for (uint32_t hh = (uint32_t)core; hh < NHt; hh += NUM_CORES)
          ssdn_mimo_tables(cxp, hh);
        pi_cl_team_barrier();
        ssdn_mimo_step1(cxp, (uint32_t)core);
        pi_cl_team_barrier();
        ssdn_mimo_step2(cxp, (uint32_t)core);
      } else if (decay_mode == 1) {
        // two-scale: (1) the three shared weights once per chunk (head 0's slots) + per-head factors;
        // (2) x~ row maxima and S8 over all cores; (3) x~ quantisation over all cores
        for (uint32_t k = (uint32_t)core; k < 3 + NHt; k += NUM_CORES) {
          static const uint8_t ssdn_ts_order[3] = {1, 2, 0};
          ssdn_phaseA_item(cxp, k < 3 ? ssdn_ts_order[k] : (k - 3) * 4 + 3, cscr);
        }
        pi_cl_team_barrier();
        ssdn_ts_step2(cxp, (uint32_t)core, cscr);
        pi_cl_team_barrier();
        ssdn_ts_step3(cxp, (uint32_t)core, G + Q * Q, QM + Q);
      } else {
        if (mamba3 && m3_theta) {   // Mamba-3 RoPE: rotated B, C rows per head before the weight items
          for (uint32_t hh = (uint32_t)core; hh < NHt; hh += NUM_CORES)
            ssdn_m3_rope(cxp, hh);
          pi_cl_team_barrier();
        }
        for (uint32_t k = (uint32_t)core; k < 4 * NHt; k += NUM_CORES) {
          static const uint8_t ssdn_part_order[4] = {3, 1, 2, 0};
          ssdn_phaseA_item(cxp, (k % NHt) * 4 + ssdn_part_order[k / NHt], cscr);
        }
      }
      pi_cl_team_barrier();
      SSDN_T(prof_b0);

      // ---- phase B: core 0 queues head hh, everybody finishes head hh-1 while the NE16 runs ----
      for (uint32_t hh = 0; hh <= NHt; hh++) {
        SSDN_T(pb_t0);
        if (core == 0 && hh < NHt) {
          const uint32_t sl = hh & 1;
          const uint8_t *we = wenc + (decay_mode == 1 ? 0 : hh * WSZ);
          int32_t *a = acc + sl * ASZ;
          ssdn_dispatch(dev, &ssdn_tasks[sl][0], xt + hh * PXA * QR, we, a);   // encoded W1 rows then W2 rows
          ssdn_dispatch(dev, &ssdn_tasks[sl][2], su + hh * PXA * N, we + QR * QR + N * QR, a + PXA * (QR + N));
        }
        SSDN_T(pb_t1);
#if SSDN_PROFILE
        if (core == 0) pb_disp += pb_t1 - pb_t0;
#endif
        if (hh > 0) {
          if (hh < NHt) {  // core 0 dispatched head hh: cores 1..7 finish head hh-1
            if (core > 0)
              ssdn_finish_head(cxp, hh, (uint32_t)core - 1, NUM_CORES - 1);
          } else {
            ssdn_finish_head(cxp, hh, (uint32_t)core, NUM_CORES);
          }
        }
        SSDN_T(pb_t3);
        pi_cl_team_barrier();
        SSDN_T(pb_t4);
        if (core == 0 && hh < NHt)
          ne16_nnx_resolve_wait(dev, &ssdn_tasks[hh & 1][2]);
        SSDN_T(pb_t5);
        pi_cl_team_barrier();
#if SSDN_PROFILE
        if (core == 0) pb_res += pb_t5 - pb_t4;
        if (core == 1) pb_bar += (pb_t4 - pb_t3) + (pi_perf_read(PI_PERF_CYCLES) - pb_t5);
#endif
      }
#if SSDN_PROFILE
      if (core == 0) { uint32_t t = pi_perf_read(PI_PERF_CYCLES); prof_G += prof_a0 - prof_g0; prof_A += prof_b0 - prof_a0; prof_B += t - prof_b0; }
#endif
    }
  }
  pi_cl_team_barrier();
#if SSDN_PROFILE
  static PI_L1 uint32_t ssdn_ncall;
  if (core == 0) ssdn_ncall++;
  pi_cl_team_barrier();
  if (ssdn_ncall != 25) return;
  if (core == 0) printf("SSDN_PROF N=%u NHt=%u L=%u G=%u A=%u B=%u total=%u entry=%u core0: dispatch=%u resolve_wait=%u\n", prof_N, NHt, L, prof_G, prof_A, prof_B, pi_perf_read(PI_PERF_CYCLES) - prof_t0, prof_t0, pb_disp, pb_res);
  pi_cl_team_barrier();
  if (core == 1) { (void)pb_epi; (void)pb_upd; }
  if (core == 1) { ssdn_p1[0] = ssdn_p_epi; ssdn_p1[1] = ssdn_p_upd; ssdn_p1[2] = pb_bar; }
  pi_cl_team_barrier();
  if (core == 0) printf("SSDN_PROF core1: epilogue=%u state_update=%u barrier_wait=%u\n", ssdn_p1[0], ssdn_p1[1], ssdn_p1[2]);
#endif
}

// ----------------------------------------------------------------------------------------------------------------------
// StaticScan_NE16: a Mamba-2 block whose decay, Delta, B, C are constants is a per-head causal LTI filter over the window.
// One NE16 1x1 job per (head, pixel chunk): pixels = the head's channels (SSTAT_PXC per chunk, a 3 x 9 grid), Ki = s,
// Ko = t (whole window, L a multiple of 16), constant int8 weights already in the NE16 layout (wenc, + comp = 128 sum W).
// Epilogue on the cores (m2_export/static_scan.py):
//   y_acc = (acc - comp[t]) * M[t] + Dq * x[t][c];  y = sat32(rs(rs(y_acc * gate_lut[z + 128], 13), out_shift))
#define SSTAT_GH 3
#define SSTAT_GW 9
#define SSTAT_PXC (SSTAT_GH * SSTAT_GW)
static ne16_task_t sstat_task;

static __attribute__((noinline)) void sstat_epilogue(const int32_t *acc, const int8_t *x, const int8_t *z, int32_t *y,
                                                     const int32_t *comp, const int32_t *M, int32_t Dq,
                                                     const int32_t *gate_lut, uint32_t L, uint32_t d_inner,
                                                     uint32_t col0, uint32_t npx, int osh, uint32_t core) {
  const uint32_t tot = npx * L, ch = (tot + NUM_CORES - 1) / NUM_CORES;
  const uint32_t i0 = core * ch < tot ? core * ch : tot, i1 = i0 + ch < tot ? i0 + ch : tot;
  const int64_t D64 = (int64_t)Dq;
  if (i0 >= i1)
    return;
  // walk (pixel cl, token t) with counters: no per-element division
  uint32_t cl = i0 / L, t = i0 - cl * L;
  uint32_t idx = t * d_inner + col0 + cl;
  for (uint32_t i = i0; i < i1; i++) {
    const int64_t y_acc = (int64_t)(acc[i] - comp[t]) * (int64_t)M[t] + D64 * (int64_t)x[idx];
    const int64_t y_g = ssdn_round_shift(y_acc * (int64_t)gate_lut[(int32_t)z[idx] + 128], 13);
    y[idx] = ssdn_sat_i32(osh > 0 ? ssdn_round_shift(y_g, osh) : y_g);
    if (++t == L) {
      t = 0;
      cl++;
      idx = col0 + cl;
    } else {
      idx += d_inner;
    }
  }
}

void GAP9_StaticScanNE16_i8(const int8_t *__restrict__ x, const int8_t *__restrict__ z, const uint8_t *__restrict__ wenc,
                            const int32_t *__restrict__ comp, const int32_t *__restrict__ M,
                            const int32_t *__restrict__ Dq, int32_t *__restrict__ y,
                            const int32_t *__restrict__ gate_lut, uint8_t *__restrict__ scratch, uint32_t L, uint32_t P,
                            uint32_t NHt, int32_t out_shift) {
  const uint32_t core = pi_core_id();
  const uint32_t d_inner = NHt * P;
  const uint32_t nchunk = (P + SSTAT_PXC - 1) / SSTAT_PXC;
  uint8_t *xt = scratch;                                         // [nchunk * PXC][L] (+128), padded pixels unused
  int32_t *acc = (int32_t *)(xt + nchunk * SSTAT_PXC * L);       // [PXC][L]
  const ne16_dev_t *dev = ne16_pulp_get_dev();
  if (core == 0)
    ssdn_task_setup(&sstat_task, L, L, SSTAT_GH, SSTAT_GW);
  for (uint32_t hh = 0; hh < NHt; hh++) {
    // x columns of head hh -> xt[c][s] (+128), channels over the cores
    for (uint32_t c = core; c < P; c += NUM_CORES) {
      const int8_t *src = x + hh * P + c;
      uint8_t *dst = xt + c * L;
      for (uint32_t s_ = 0; s_ < L; s_++)
        dst[s_] = (uint8_t)src[s_ * d_inner] ^ 0x80;
    }
    pi_cl_team_barrier();
    const uint8_t *wh = wenc + hh * L * L;
    for (uint32_t k = 0; k < nchunk; k++) {
      if (core == 0) {
        ssdn_dispatch(dev, &sstat_task, xt + k * SSTAT_PXC * L, wh, acc);
        ne16_nnx_resolve_wait(dev, &sstat_task);
      }
      pi_cl_team_barrier();
      const uint32_t npx = (k + 1) * SSTAT_PXC <= P ? SSTAT_PXC : P - k * SSTAT_PXC;
      sstat_epilogue(acc, x, z, y, comp + hh * L, M + hh * L, Dq[hh], gate_lut, L, d_inner, hh * P + k * SSTAT_PXC, npx,
                     (int)out_shift, core);
      pi_cl_team_barrier();
    }
  }
}

#endif // DEEPLOY_USE_NE16
