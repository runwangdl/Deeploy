/*
 * SelectiveScan with an int16 recurrent state (FEMBA / Mamba-1), GAP9 cluster, 8 cores.
 *
 * Numerics == AI_AGENT/Mamba/scripts/femba_state_bits.py, SSM_STATE_MODE=i16hw (the model whose
 * AUROC was measured), == ssm_i16_bench/gen_bench.py except for the gate/output rounding, which
 * here is the symmetric rounding of the original kernel (bit-exact with the ARES TQ reference):
 *   B16/C16 = sat16(rs(B, bc)), rs = symmetric rounding shift        (done here, per tile)
 *   v   = (dt * A16[d][n]) >> (8 - sA[d])   (left shift if sA > 8)
 *   dA  = exp_lut[max((v + RANGE + HALF) >> 8, 0)]
 *   h   = clip16(((dA * h16 + 2^14) >> 15) + (((dt * B16 + 2^(NB-1)) >> NB) * x) >> sH[d])
 *   acc = sum_n h16 * C16                     (int32)
 *   y   = ((acc << (sH[d] + bc)) >> 15) + D[d] * x ;  gate (rs 13) ; requant (rs 40) ; clip8
 * Layout: x, z, dt, y [L][D_inner]; B, C [L][N] int32; A16 [D_inner][N]; shA = 8 - sA (int8,
 * may be negative), sH, ysh = sH + bc (uint8) per channel; h_buffer [D_inner][N] int16 persists
 * across L tiles (is_first_L_tile clears it).
 */
#include "DeeployGAP9Math.h"
#include "pmsis.h"

#define I16_RANGE_Q15 (20 << 15)
#define I16_HALF_Q15 128
#define I16_LUT_MAX 2560

static inline int64_t _i16_round_shift_i64(int64_t x, int s) {
  if (s <= 0)
    return x;
  const int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

static inline int32_t _i16_rs32(int32_t x, int s) {
  if (s <= 0)
    return x;
  const int32_t half = 1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

static inline int32_t _i16_clip16(int32_t v) {
  return v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
}

void GAP9_SelectiveScanI16_i8_i8(
    const int8_t *__restrict__ x, const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
    const int32_t *__restrict__ C, const int16_t *__restrict__ A16,
    const int32_t *__restrict__ D_skip, const int8_t *__restrict__ shA,
    const uint8_t *__restrict__ sH, const uint8_t *__restrict__ ysh,
    int8_t *__restrict__ y, int16_t *__restrict__ h_buffer,
    int16_t *__restrict__ BC16, /* scratch [2][L][N] */
    const int32_t *__restrict__ gate_lut, const int16_t *__restrict__ exp_lut,
    uint32_t L, uint32_t D_inner, uint32_t N, uint32_t bc_shift,
    int32_t output_requant_mul_q40, uint32_t is_first_L_tile) {
  const uint32_t core_id = pi_core_id();
  const uint32_t D_chunk = (D_inner >> 3) + ((D_inner & 7) != 0);
  const uint32_t D_start = (core_id * D_chunk < D_inner) ? core_id * D_chunk : D_inner;
  const uint32_t D_end = (D_start + D_chunk < D_inner) ? D_start + D_chunk : D_inner;
  int16_t *B16 = BC16;
  int16_t *C16 = BC16 + L * N;
  // B/C -> int16 (shared by all channels): split L*N over the cores
  {
    const uint32_t tot = L * N;
    const uint32_t ch = (tot >> 3) + ((tot & 7) != 0);
    const uint32_t s0 = (core_id * ch < tot) ? core_id * ch : tot;
    const uint32_t s1 = (s0 + ch < tot) ? s0 + ch : tot;
    for (uint32_t i = s0; i < s1; i++) {
      B16[i] = (int16_t)_i16_clip16(_i16_rs32(B[i], (int)bc_shift));
      C16[i] = (int16_t)_i16_clip16(_i16_rs32(C[i], (int)bc_shift));
    }
  }
  if (is_first_L_tile) {
    for (uint32_t d = D_start; d < D_end; d++)
      for (uint32_t n = 0; n < N; n++)
        h_buffer[d * N + n] = 0;
  }
  pi_cl_team_barrier();
  const int32_t NB = 8 - (int32_t)bc_shift;
  const int32_t dB_half = (NB > 0) ? (1 << (NB - 1)) : 0;
  const int8_t *lut8 = (const int8_t *)exp_lut;
  const int32_t exp_off = I16_RANGE_Q15 + I16_HALF_Q15;
  for (uint32_t d = D_start; d < D_end; d++) {
    int32_t h_l[16];
    const int16_t *A_row = A16 + d * N;
    int16_t *const h_row = h_buffer + d * N;
    for (uint32_t n = 0; n < N; n++)
      h_l[n] = h_row[n];
    const int32_t d_skip_val = D_skip[d];
    const int32_t sh_a = shA[d];       /* 8 - sA, may be negative */
    const int32_t sh_h = sH[d];
    const int32_t sh_y = ysh[d];
    int32_t A_l[16];
    for (uint32_t n = 0; n < N; n++)
      A_l[n] = A_row[n];
    for (uint32_t t = 0; t < L; t++) {
      const int16_t *B_row = B16 + t * N;
      const int16_t *C_row = C16 + t * N;
      const int32_t x_val = (int32_t)x[t * D_inner + d];
      const int32_t z_val = (int32_t)z[t * D_inner + d];
      const int32_t dt_val = (int32_t)dt[t * D_inner + d];
      int32_t acc = 0;
      /* dt >= 0 (softplus) and A16 <= 0  =>  dt*A16 <= 0  =>  idx <= 2560: only the lower clip is needed */
#define I16_LANE(n, VSHIFT, DBP)                                                              \
      do {                                                                                    \
        const int32_t v_ = VSHIFT(dt_val * A_l[n]);                                           \
        const int32_t idx_ = __builtin_pulp_maxsi((v_ + exp_off) >> 8, 0);                    \
        int32_t dA_;                                                                          \
        __asm__("p.lh %0,%1(%2)" : "=r"(dA_) : "r"(idx_ << 1), "r"(lut8));                     \
        int32_t hn_;                                                                          \
        __asm__("p.mulsRN %0,%1,%2,15" : "=r"(hn_) : "r"(dA_), "r"(h_l[n]));                  \
        const int32_t dBp_ = DBP((int32_t)B_row[n]);                                          \
        hn_ = __builtin_pulp_clip(hn_ + ((dBp_ * x_val) >> sh_h), -32768, 32767);             \
        h_l[n] = hn_;                                                                         \
        acc += hn_ * (int32_t)C_row[n];                                                       \
      } while (0)
#define I16_VR(p) ((p) >> sh_a)
#define I16_VL(p) ((p) << sh_al)
#define I16_DBP2(b) ({ int32_t r_; __asm__("p.mulsRN %0,%1,%2,2" : "=r"(r_) : "r"(dt_val), "r"(b)); r_; })
#define I16_ALL(VS, DB) I16_LANE(0,VS,DB); I16_LANE(1,VS,DB); I16_LANE(2,VS,DB); I16_LANE(3,VS,DB); \
                        I16_LANE(4,VS,DB); I16_LANE(5,VS,DB); I16_LANE(6,VS,DB); I16_LANE(7,VS,DB); \
                        I16_LANE(8,VS,DB); I16_LANE(9,VS,DB); I16_LANE(10,VS,DB); I16_LANE(11,VS,DB); \
                        I16_LANE(12,VS,DB); I16_LANE(13,VS,DB); I16_LANE(14,VS,DB); I16_LANE(15,VS,DB)
      if (N == 16 && NB == 2) {
        if (sh_a >= 0) { I16_ALL(I16_VR, I16_DBP2); }
        else { const int32_t sh_al = -sh_a; I16_ALL(I16_VL, I16_DBP2); }
      } else {
        for (uint32_t n = 0; n < N; n++) {
          const int32_t p = dt_val * A_l[n];
          const int32_t v = (sh_a >= 0) ? (p >> sh_a) : (p << (-sh_a));
          const int32_t idx = __builtin_pulp_maxsi((v + exp_off) >> 8, 0);
          const int32_t dA = exp_lut[idx];
          int32_t hn = (int32_t)(((int64_t)dA * h_l[n] + (1 << 14)) >> 15);
          const int32_t dBp = (dt_val * (int32_t)B_row[n] + dB_half) >> NB;
          hn = _i16_clip16(hn + ((dBp * x_val) >> sh_h));
          h_l[n] = hn;
          acc += hn * (int32_t)C_row[n];
        }
      }
      int64_t y_acc = (((int64_t)acc) << sh_y) >> 15;
      y_acc += (int64_t)d_skip_val * (int64_t)x_val;
      const int64_t y_gated = _i16_round_shift_i64(y_acc * (int64_t)gate_lut[z_val + 128], 13);
      int64_t y_out = _i16_round_shift_i64(y_gated * (int64_t)output_requant_mul_q40, 40);
      y_out = y_out > 127 ? 127 : (y_out < -128 ? -128 : y_out);
      y[t * D_inner + d] = (int8_t)y_out;
    }
    for (uint32_t n = 0; n < N; n++)
      h_row[n] = (int16_t)h_l[n];
  }
  pi_cl_team_barrier();
}
