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


// store one output: out_bits 8 -> int8 Q40 requant (default); otherwise int32 tensor holding
// sat_{out_bits}(rs(y_gated, out_shift)) (wide output for an int32 x int8 out_proj, FEMBA Mamba-2 classifier head)
static inline void _i16_store(void *y, uint32_t i, int64_t y_gated, int32_t mul_q40, uint32_t out_bits, uint32_t out_shift) {
  if (out_bits == 8) {
    int64_t v = _i16_round_shift_i64(y_gated * (int64_t)mul_q40, 40);
    ((int8_t *)y)[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
  } else {
    const int64_t lim = ((int64_t)1 << (out_bits - 1)) - 1;
    int64_t v = out_shift ? _i16_round_shift_i64(y_gated, (int)out_shift) : y_gated;
    ((int32_t *)y)[i] = (int32_t)(v > lim ? lim : (v < -lim - 1 ? -lim - 1 : v));
  }
}

static int8_t __attribute__((unused)) _i16_output(int32_t acc, int32_t sh_y, int32_t d_skip_val, int32_t x_val,
                                                    int32_t gate, int32_t output_requant_mul_q40) {
  int64_t y_acc = (((int64_t)acc) << sh_y) >> 15;
  y_acc += (int64_t)d_skip_val * (int64_t)x_val;
  const int64_t y_gated = _i16_round_shift_i64(y_acc * (int64_t)gate, 13);
  int64_t y_out = _i16_round_shift_i64(y_gated * (int64_t)output_requant_mul_q40, 40);
  return (int8_t)(y_out > 127 ? 127 : (y_out < -128 ? -128 : y_out));
}

static void __attribute__((noinline)) _i16_scan_generic(
    const int8_t *x, const int8_t *z, const int16_t *dt, const int16_t *B16, const int16_t *C16,
    const int16_t *A16, const int32_t *D_skip, const int8_t *shA, const uint8_t *sH, const uint8_t *ysh,
    void *y, int16_t *h_buffer, const int32_t *gate_lut, const int16_t *exp_lut, uint32_t L,
    uint32_t D_inner, uint32_t N, int32_t NB, int32_t output_requant_mul_q40, uint32_t D_start, uint32_t D_end,
    uint32_t out_bits, uint32_t out_shift) {
  const int32_t dB_half = (NB > 0) ? (1 << (NB - 1)) : 0;
  const int32_t exp_off = I16_RANGE_Q15 + I16_HALF_Q15;
  for (uint32_t d = D_start; d < D_end; d++) {
    int32_t h_l[32];
    const int16_t *A_row = A16 + d * N;
    int16_t *const h_row = h_buffer + d * N;
    for (uint32_t n = 0; n < N; n++)
      h_l[n] = h_row[n];
    const int32_t sh_a = shA[d], sh_h = sH[d], sh_y = ysh[d], d_skip_val = D_skip[d];
    for (uint32_t t = 0; t < L; t++) {
      const int16_t *B_row = B16 + t * N;
      const int16_t *C_row = C16 + t * N;
      const int32_t x_val = x[t * D_inner + d], z_val = z[t * D_inner + d], dt_val = dt[t * D_inner + d];
      int32_t acc = 0;
      for (uint32_t n = 0; n < N; n++) {
        const int32_t p = dt_val * (int32_t)A_row[n];
        const int32_t v = (sh_a >= 0) ? (p >> sh_a) : (p << (-sh_a));
        int32_t idx = (v + exp_off) >> 8;
        idx = idx < 0 ? 0 : (idx > I16_LUT_MAX ? I16_LUT_MAX : idx);
        const int32_t dA = exp_lut[idx];
        int32_t hn = (int32_t)(((int64_t)dA * h_l[n] + (1 << 14)) >> 15);
        const int32_t dBp = (dt_val * (int32_t)B_row[n] + dB_half) >> NB;
        hn = _i16_clip16(hn + ((dBp * x_val) >> sh_h));
        h_l[n] = hn;
        acc += hn * (int32_t)C_row[n];
      }
      {
        int64_t y_acc = ((((int64_t)acc) << sh_y) >> 15) + (int64_t)d_skip_val * (int64_t)x_val;
        _i16_store(y, t * D_inner + d, _i16_round_shift_i64(y_acc * (int64_t)gate_lut[z_val + 128], 13),
                   output_requant_mul_q40, out_bits, out_shift);
      }
    }
    for (uint32_t n = 0; n < N; n++)
      h_row[n] = (int16_t)h_l[n];
  }
}

void GAP9_SelectiveScanI16_i8_i8(
    const int8_t *__restrict__ x, const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
    const int32_t *__restrict__ C, const int16_t *__restrict__ A16,
    const int32_t *__restrict__ D_skip, const int8_t *__restrict__ shA,
    const uint8_t *__restrict__ sH, const uint8_t *__restrict__ ysh,
    void *__restrict__ y, int16_t *__restrict__ h_buffer,
    int16_t *__restrict__ BC16, /* scratch [2][L][N] */
    const int32_t *__restrict__ gate_lut, const int16_t *__restrict__ exp_lut,
    uint32_t L, uint32_t D_inner, uint32_t N, uint32_t bc_shift,
    int32_t output_requant_mul_q40, uint32_t is_first_L_tile, uint32_t out_bits, uint32_t out_shift) {
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
  if (N != 16 || NB != 2) {
    _i16_scan_generic(x, z, dt, B16, C16, A16, D_skip, shA, sH, ysh, y, h_buffer, gate_lut, exp_lut, L, D_inner, N, NB,
                      output_requant_mul_q40, D_start, D_end, out_bits, out_shift);
    pi_cl_team_barrier();
    return;
  }
  const int8_t *lut8 = (const int8_t *)exp_lut;
  /* Hot path == ssm_i16_bench v5 (844 B of code, 49.7 cyc/step on gvsoc); channels whose A needs a
     left shift (sA > 8, rare) take the generic path so the hot loop keeps a single shift. */
  for (uint32_t d = D_start; d < D_end; d++) {
    const int32_t sh_a = shA[d];
    if (sh_a < 0) {
      _i16_scan_generic(x, z, dt, B16, C16, A16, D_skip, shA, sH, ysh, y, h_buffer, gate_lut, exp_lut, L, D_inner, 16, 2,
                        output_requant_mul_q40, d, d + 1, out_bits, out_shift);
      continue;
    }
    int32_t h_l[16];
    const int16_t *A_row = A16 + d * 16;
    int16_t *const h_row = h_buffer + d * 16;
    for (uint32_t n = 0; n < 16; n++)
      h_l[n] = h_row[n];
    const int32_t d_skip_val = D_skip[d];
    const int32_t sh_h = sH[d];
    const int32_t sh_y = ysh[d];
    const int32_t exp_add = (I16_RANGE_Q15 + I16_HALF_Q15) << sh_a;
    const int32_t exp_sh = sh_a + 8;
    for (uint32_t t = 0; t < L; t++) {
      const int16_t *B_row = B16 + t * 16;
      const int16_t *C_row = C16 + t * 16;
      const int32_t x_val = (int32_t)x[t * D_inner + d];
      const int32_t z_val = (int32_t)z[t * D_inner + d];
      const int32_t dt_val = (int32_t)dt[t * D_inner + d];
      int32_t acc = 0;
#define I16_LANE(n) do { \
        int32_t v = __builtin_pulp_maxsi((dt_val * (int32_t)A_row[n] + exp_add) >> exp_sh, 0); \
        int32_t dA; __asm__("p.lh %0,%1(%2)" : "=r"(dA) : "r"(v << 1), "r"(lut8)); \
        int32_t hn; __asm__("p.mulsRN %0,%1,%2,15" : "=r"(hn) : "r"(dA), "r"(h_l[n])); \
        int32_t dBp; __asm__("p.mulsRN %0,%1,%2,2" : "=r"(dBp) : "r"(dt_val), "r"((int32_t)B_row[n])); \
        hn = __builtin_pulp_clip(hn + ((dBp * x_val) >> sh_h), -32768, 32767); h_l[n] = hn; \
        acc += hn * (int32_t)C_row[n]; \
      } while (0);
      for (uint32_t n = 0; n < 16; n++) { I16_LANE(n) }
#undef I16_LANE
      int64_t y_acc = (((int64_t)acc) << sh_y) >> 15;
      y_acc += (int64_t)d_skip_val * (int64_t)x_val;
      const int64_t y_gated = _i16_round_shift_i64(y_acc * (int64_t)gate_lut[z_val + 128], 13);
      _i16_store(y, t * D_inner + d, y_gated, output_requant_mul_q40, out_bits, out_shift);
    }
    for (uint32_t n = 0; n < 16; n++)
      h_row[n] = (int16_t)h_l[n];
  }
  pi_cl_team_barrier();
}
