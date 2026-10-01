/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Mamba-3 chunked scan. Structure and fixed-point conventions follow SSDScan.c, which this
// is a strict superset of: lam == 32767, theta == 0, R == 1 reproduces it bit for bit.
//
// What the three Mamba-3 axes cost here, and why:
//
//  * Trapezoidal (gamma, w): the recurrence h_t = a_t h_{t-1} + beta_t v_{t-1} + gamma_t v_t
//    has beta_t = (1-lam_t) dt_t a_t, and that a_t cancels the one extra decay step v_{t-1}
//    would otherwise have taken, so every input v_k reaches h_t (t > k) through the same
//    decay as in SSD with weight w_k = gamma_k + (1-lam_{k+1}) dt_{k+1}. Phase 1 therefore
//    builds dB from w_k instead of dt_k, and only the i == j entry of the intra-chunk
//    matrix (where v_k has not decayed yet) still uses gamma_k alone, which phase 3
//    recomputes into an [N*R] temp. No extra multiply in any hot loop.
//    gamma and w arrive precomputed (an O(L*H) elementwise op upstream): w_k needs
//    dt_{k+1}, and computing it here would be truncated at every L-tile boundary - that
//    showed up as 5 errors at L_tile=16 and 9 at L_tile=8, zero untiled, before this.
//
//  * RoPE (theta): Prop. 3 of the paper lets the per-step 2x2 state rotation be moved onto
//    B and C. Each is an [N*R] row per step, against the [Head_dim*N] state, so the cost is
//    one pair rotation per two state slots per step, done once in phase 1 (B) and once in
//    phase 3 (C). The accumulated angle is per (batch, head) and is carried across L-tiles
//    in theta_state exactly as h_state is.
//
//  * MIMO (R): every dot product over n becomes a dot product over (n, r), and the x and y
//    indexing gains the rank. This is the only axis that adds arithmetic - R x the MACs of
//    the SSD kernel on the same [Head_dim, N] state - which is its purpose.
//
// Stack: dB_chunk is Chunk_size*N*R int32, intra_chunk_scores is Chunk_size*R*R int32, plus
// five Chunk_size arrays and two N*R rows, so SLAVESTACKSIZE must cover roughly
// 4*Chunk_size*(N*R + R*R) + 8*N*R + 20*Chunk_size + frame. At Chunk_size=8, N=16 that is
// ~0.9 KB for R=1, ~1.6 KB for R=2 and ~3.1 KB for R=4; use SLAVESTACKSIZE=4096.

#include "DeeployPULPMath.h"
#include "Mamba3Scan.h"

#include "Mamba3ScanLUT.h"

// Define M3_SKIP_ROPE to compile out the three m3_rope_row calls. The rotation runs
// regardless of theta (there is no data-dependent skip), so a theta == 0 fixture already pays
// for it; compiling it out is the only way to measure what it costs. Measured on the euler
// fixture: 7,147,452 with rotation, 6,962,126 without -> 2.6% of the kernel, against the
// 4N/(N*P) = 5% back-of-envelope. Leave undefined for production.
#define M3_WIDE_FRAC_BITS 15
#define M3_Q20 (1 << 20)
#define M3_EXP_STEP (M3_Q20 / 128)
#define M3_EXP_RANGE_Q20 ((int64_t)20 * M3_Q20)
#define M3_ONE_Q15 ((1 << M3_WIDE_FRAC_BITS) - 1)
#define M3_LAM_ONE 32767
#define M3_OUTPUT_REQUANT_SHIFT 40

static inline int64_t m3_asr(int64_t x, int s) {
  if (s <= 0)
    return x;
  return x >> s;
}

static inline int64_t m3_round_shift(int64_t x, int s) {
  if (s <= 0)
    return x;
  int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

static inline int32_t m3_sat_i32(int64_t x) {
  if (x > (int64_t)2147483647)
    return (int32_t)2147483647;
  if (x < -(int64_t)2147483648)
    return (int32_t)(-2147483648);
  return (int32_t)x;
}

static inline int8_t m3_sat_i8(int64_t x) {
  if (x > 127)
    return (int8_t)127;
  if (x < -128)
    return (int8_t)(-128);
  return (int8_t)x;
}

__attribute__((always_inline)) static inline int32_t m3_expq(int64_t z_q15) {
  int64_t arg_q20 = z_q15 * (int64_t)32;
  if (arg_q20 >= 0)
    return M3_ONE_Q15;
  if (arg_q20 < -M3_EXP_RANGE_Q20)
    arg_q20 = -M3_EXP_RANGE_Q20;
  int64_t idx = (arg_q20 + M3_EXP_RANGE_Q20 + (M3_EXP_STEP / 2)) / M3_EXP_STEP;
  if (idx < 0)
    idx = 0;
  if (idx > M3_EXP_LUT_LEN - 1)
    idx = M3_EXP_LUT_LEN - 1;
  return M3_EXP_LUT[(int)idx];
}

// R(theta)^T on adjacent (2k, 2k+1) slots of an [N*R] row laid out [n][r]: the pair is
// (n, n+1) at fixed r. Matches mamba3_q._apply_rope: even' = (c e + s o) >> 15,
// odd' = (-s e + c o) >> 15. An odd N leaves the last slot untouched.
static inline void m3_rope_row(int32_t *__restrict__ row, uint32_t N, uint32_t R, int32_t theta_cum) {
  const int idx = (int)((theta_cum >> 8) & (M3_ROT_LUT_LEN - 1));
  const int64_t c = M3_ROT_COS[idx];
  const int64_t s = M3_ROT_SIN[idx];
  const uint32_t n_pairs = N / 2;
  for (uint32_t pair = 0; pair < n_pairs; ++pair) {
    int32_t *even = row + (2 * pair) * R;
    int32_t *odd = even + R;
    for (uint32_t r = 0; r < R; ++r) {
      const int64_t e = even[r];
      const int64_t o = odd[r];
      even[r] = (int32_t)m3_asr(c * e + s * o, M3_WIDE_FRAC_BITS);
      odd[r] = (int32_t)m3_asr(-s * e + c * o, M3_WIDE_FRAC_BITS);
    }
  }
}

void GAP9_Mamba3Scan_i8_i8(
    const int8_t *__restrict__ x,
    const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt,
    const int32_t *__restrict__ B,
    const int32_t *__restrict__ C,
    const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip,
    const int16_t *__restrict__ gamma,
    const int16_t *__restrict__ w,
    const int16_t *__restrict__ theta,
    int8_t *__restrict__ y,
    int32_t *__restrict__ h_state,
    int32_t *__restrict__ theta_state,
    const int32_t *__restrict__ gate_lut,
    uint32_t B_size, uint32_t Chunk_size, uint32_t N,
    uint32_t Head_dim, uint32_t Group_dim, uint32_t N_heads,
    uint32_t L, uint32_t R, int32_t output_requant_mul_q40,
    uint32_t init_state)
{
  const int number_of_chunks = L / Chunk_size;
  const int heads_per_group = N_heads / Group_dim;
  const int NR = N * R;                       // one B/C row, rank-major inside each n
  const int d_inner = N_heads * Head_dim * R; // x/z/y last dim
  const int group_state = Group_dim * NR;     // B/C last dim
  const int head_state_size = Head_dim * N;   // state is rank-free

  const int core_id = (int)pi_core_id();

  // Same (head teams) x (Head_dim lanes) partition as SSDScan; the model gains the R factor
  // on the parallel part only, which is where the rank's MACs go.
  int64_t best_modelled_cost = -1;
  int cores_per_head = 1;
  {
    const int64_t chunk_triangular = (int64_t)Chunk_size * (Chunk_size + 1) / 2;
    const int64_t parallel_work = (int64_t)Head_dim * R * (chunk_triangular + 2 * (int64_t)Chunk_size * NR);
    const int64_t spine_work = (int64_t)NR * (3 * (int64_t)Chunk_size + chunk_triangular);
    for (int cand = 1; cand <= NUM_CORES; ++cand) {
      if (NUM_CORES % cand != 0) continue;
      if (cand > (int)Head_dim) continue;
      const int teams = NUM_CORES / cand;
      const int64_t heads_per_team = ((int64_t)N_heads + teams - 1) / teams;
      const int64_t cost = heads_per_team * (spine_work + parallel_work / cand);
      if (best_modelled_cost < 0 || cost < best_modelled_cost) {
        best_modelled_cost = cost;
        cores_per_head = cand;
      }
    }
  }
  const int head_teams = NUM_CORES / cores_per_head;
  const int my_head_team = core_id / cores_per_head;
  const int my_lane = core_id % cores_per_head;

  for (int head = my_head_team; head < (int)N_heads; head += head_teams) {

    const int group = head / heads_per_group;
    const int64_t A_head = (int64_t)A[head];
    const int64_t D_head = (int64_t)D_skip[head];

    const int feature_lo = (int)(((int64_t)my_lane * Head_dim) / cores_per_head);
    const int feature_hi = (int)((((int64_t)my_lane + 1) * Head_dim) / cores_per_head);

    // per-head scratch (see the stack note at the top)
    int32_t dB_chunk[Chunk_size * NR];             // w_k-weighted, rotated B rows
    int32_t dB_diag[NR];                           // gamma_q-weighted rotated B_q (phase 3)
    int32_t cumulative_log_decay[Chunk_size];
    int32_t decay_from_chunk_start[Chunk_size];
    int32_t decay_to_chunk_end[Chunk_size];
    int32_t decay_matrix_row[Chunk_size];
    // [key][q][r]: q = readout rank (which C^(q) reads out), r = input rank (which x^(r) went
    // in). They are different indices and must not be contracted together - the R=1 scalar
    // score would be wrong for every R > 1. Chunk_size*R*R int32 (512 B at Chunk=8, R=4).
    int32_t intra_chunk_scores[Chunk_size * R * R];
    int32_t theta_chunk[Chunk_size];               // accumulated angle at each key
    int32_t C_query_cache[NR];

    for (int batch = 0; batch < (int)B_size; ++batch) {

      int32_t *h_state_head = h_state + (int64_t)(batch * N_heads + head) * head_state_size;
      int32_t *theta_cum_ptr = theta_state + (batch * N_heads + head);

      if (init_state) {
        for (int feature_idx = feature_lo; feature_idx < feature_hi; ++feature_idx) {
          int32_t *h_row = &h_state_head[feature_idx * N];
          for (int state_idx = 0; state_idx < (int)N; ++state_idx)
            h_row[state_idx] = 0;
        }
      }
      // The angle is shared by every lane of a head team. On a fresh sequence each lane
      // starts from 0 locally rather than reading a value lane 0 is in the middle of
      // zeroing - a team barrier here would deadlock, since teams own different numbers of
      // heads. On an L-tile continuation all lanes read the value lane 0 stored at the end
      // of the previous tile, before anyone could write this one. Only lane 0 writes back.
      int32_t theta_cum = init_state ? 0 : *theta_cum_ptr;

      for (int chunk = 0; chunk < number_of_chunks; ++chunk) {
        const int chunk_start = chunk * Chunk_size;

        // Phase 1: per key: angle, log-decay, and the w_k-weighted (rotated) B row.
        int32_t running_cumulative_log_decay = 0;
        for (int pos_in_chunk = 0; pos_in_chunk < (int)Chunk_size; ++pos_in_chunk) {
          const int time_step = chunk_start + pos_in_chunk;
          const int t_idx = (batch * L + time_step) * N_heads + head;
          const int64_t dt_value = (int64_t)dt[t_idx];

          theta_cum += (int32_t)theta[t_idx];
          theta_chunk[pos_in_chunk] = theta_cum;

          running_cumulative_log_decay += (int32_t)m3_asr(dt_value * A_head, 8);
          cumulative_log_decay[pos_in_chunk] = running_cumulative_log_decay;

          const int64_t w_key = (int64_t)w[t_idx];

          const int32_t *B_row = &B[(batch * L + time_step) * group_state + group * NR];
          int32_t *dB_row = &dB_chunk[pos_in_chunk * NR];
          for (int j = 0; j < NR; ++j)
            dB_row[j] = B_row[j];
#ifndef M3_SKIP_ROPE
          m3_rope_row(dB_row, N, R, theta_cum);
#endif
          for (int j = 0; j < NR; ++j)
            dB_row[j] = (int32_t)m3_asr(w_key * (int64_t)dB_row[j], 8);
        }

        // Phase 2: exp tables
        for (int pos_in_chunk = 0; pos_in_chunk < (int)Chunk_size; ++pos_in_chunk)
          decay_from_chunk_start[pos_in_chunk] = m3_expq((int64_t)cumulative_log_decay[pos_in_chunk]);
        decay_to_chunk_end[Chunk_size - 1] = M3_ONE_Q15;
        for (int pos_in_chunk = 0; pos_in_chunk < (int)Chunk_size - 1; ++pos_in_chunk)
          decay_to_chunk_end[pos_in_chunk] = m3_expq((int64_t)(cumulative_log_decay[Chunk_size - 1] - cumulative_log_decay[pos_in_chunk]));
        const int32_t decay_full_chunk = decay_from_chunk_start[Chunk_size - 1];

        // Phase 3: outputs for every query of the chunk
        for (int query_idx = 0; query_idx < (int)Chunk_size; ++query_idx) {
          const int time_query = chunk_start + query_idx;
          const int tq_idx = (batch * L + time_query) * N_heads + head;
          const int32_t *C_query = &C[(batch * L + time_query) * group_state + group * NR];
          const int8_t *x_query = &x[(batch * L + time_query) * d_inner + head * Head_dim * R];
          const int8_t *z_query = &z[(batch * L + time_query) * d_inner + head * Head_dim * R];

          for (int j = 0; j < NR; ++j)
            C_query_cache[j] = C_query[j];
#ifndef M3_SKIP_ROPE
          m3_rope_row(C_query_cache, N, R, theta_chunk[query_idx]);
#endif

          // diagonal weight gamma_q on the rotated, un-w-weighted B_q
          {
            const int64_t gamma_q = (int64_t)gamma[tq_idx];
            const int32_t *B_q = &B[(batch * L + time_query) * group_state + group * NR];
            for (int j = 0; j < NR; ++j)
              dB_diag[j] = B_q[j];
#ifndef M3_SKIP_ROPE
            m3_rope_row(dB_diag, N, R, theta_chunk[query_idx]);
#endif
            for (int j = 0; j < NR; ++j)
              dB_diag[j] = (int32_t)m3_asr(gamma_q * (int64_t)dB_diag[j], 8);
          }

          for (int key_idx = 0; key_idx < query_idx; ++key_idx)
            decay_matrix_row[key_idx] = m3_expq((int64_t)(cumulative_log_decay[query_idx] - cumulative_log_decay[key_idx]));
          decay_matrix_row[query_idx] = M3_ONE_Q15;

          // Pass 3a: scores. k < q uses dB_chunk (w-weighted), k == q uses dB_diag (gamma-weighted).
          // One R x R block per key: score[k][q][r] = sum_n C[n][q] * dB_k[n][r].
          for (int key_idx = 0; key_idx <= query_idx; ++key_idx) {
            const int32_t *dB_row = (key_idx == query_idx) ? dB_diag : &dB_chunk[key_idx * NR];
            const int32_t dmr = decay_matrix_row[key_idx];
            int32_t *score_blk = &intra_chunk_scores[key_idx * R * R];
            for (int rq = 0; rq < (int)R; ++rq) {
              for (int rr = 0; rr < (int)R; ++rr) {
                int64_t cb_dot_product = 0;
                for (int state_idx = 0; state_idx < (int)N; ++state_idx)
                  cb_dot_product += (int64_t)C_query_cache[state_idx * R + rq] * (int64_t)dB_row[state_idx * R + rr];
                cb_dot_product = m3_asr(cb_dot_product, M3_WIDE_FRAC_BITS);
                score_blk[rq * R + rr] = (int32_t)m3_asr((int64_t)cb_dot_product * dmr, M3_WIDE_FRAC_BITS);
              }
            }
          }

          const int32_t query_decay_from_start = decay_from_chunk_start[query_idx];
          for (int feature_idx = feature_lo; feature_idx < feature_hi; ++feature_idx) {
            const int32_t *h_row = &h_state_head[feature_idx * N];
            for (int r = 0; r < (int)R; ++r) {
              const int xr = feature_idx * R + r;   // r is the readout rank of this output

              // y_diag^(r) = sum_k sum_rr score[k][r][rr] * x_k^(rr)
              int64_t y_diag_p = 0;
              for (int key_idx = 0; key_idx <= query_idx; ++key_idx) {
                const int32_t *score_row = &intra_chunk_scores[(key_idx * R + r) * R];
                const int8_t *x_key = &x[(batch * L + chunk_start + key_idx) * d_inner + head * Head_dim * R + feature_idx * R];
                for (int rr = 0; rr < (int)R; ++rr)
                  y_diag_p += (int64_t)score_row[rr] * (int64_t)x_key[rr];
              }

              int64_t state_readout = 0;
              for (int state_idx = 0; state_idx < (int)N; ++state_idx)
                state_readout += (int64_t)C_query_cache[state_idx * R + r] * (int64_t)h_row[state_idx];
              state_readout = m3_asr(state_readout, M3_WIDE_FRAC_BITS);
              const int64_t y_offdiagonal = m3_asr((int64_t)state_readout * query_decay_from_start, M3_WIDE_FRAC_BITS);

              const int64_t y_accumulator = y_diag_p + y_offdiagonal + D_head * (int64_t)x_query[xr];
              const int64_t gate_q13 = (int64_t)gate_lut[(int32_t)z_query[xr] + 128];
              const int64_t y_gated = m3_round_shift(y_accumulator * gate_q13, 13);
              y[(batch * L + time_query) * d_inner + head * Head_dim * R + xr] =
                  m3_sat_i8(m3_round_shift(y_gated * (int64_t)output_requant_mul_q40, M3_OUTPUT_REQUANT_SHIFT));
            }
          }
        }

        // Phase 4: scale dB_chunk rows by decay_to_chunk_end, then fold into the state.
        for (int key_idx = 0; key_idx < (int)Chunk_size; ++key_idx) {
          const int32_t decay_to_end = decay_to_chunk_end[key_idx];
          int32_t *dB_row = &dB_chunk[key_idx * NR];
          for (int j = 0; j < NR; ++j)
            dB_row[j] = (int32_t)m3_asr((int64_t)decay_to_end * dB_row[j], M3_WIDE_FRAC_BITS);
        }

        // h[p][n] = sat(decay_full*h[p][n]>>15 + sum_k sum_r dB[k][n][r]*x[k][p][r]).
        // 2x2 register block over (p, n); the (k, r) inner loop is the rank-extended SSD loop.
        const int8_t *x_chunk_base = &x[(batch * L + chunk_start) * d_inner + head * Head_dim * R];
        int feature_idx = feature_lo;
        for (; feature_idx + 1 < feature_hi; feature_idx += 2) {
          int32_t *h_row0 = &h_state_head[feature_idx * N];
          int32_t *h_row1 = h_row0 + N;
          const int8_t *x_base0 = x_chunk_base + feature_idx * R;
          const int8_t *x_base1 = x_base0 + R;
          int state_idx = 0;
          for (; state_idx + 1 < (int)N; state_idx += 2) {
            int64_t acc00 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx], M3_WIDE_FRAC_BITS);
            int64_t acc01 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx + 1], M3_WIDE_FRAC_BITS);
            int64_t acc10 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx], M3_WIDE_FRAC_BITS);
            int64_t acc11 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx + 1], M3_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx * R];
            const int8_t *xk0 = x_base0;
            const int8_t *xk1 = x_base1;
            for (int key_idx = 0; key_idx < (int)Chunk_size; ++key_idx) {
              for (int r = 0; r < (int)R; ++r) {
                const int64_t dB0 = dB_col[r];
                const int64_t dB1 = dB_col[R + r];
                const int64_t xv0 = xk0[r];
                const int64_t xv1 = xk1[r];
                acc00 += dB0 * xv0;
                acc01 += dB1 * xv0;
                acc10 += dB0 * xv1;
                acc11 += dB1 * xv1;
              }
              dB_col += NR;
              xk0 += d_inner;
              xk1 += d_inner;
            }
            h_row0[state_idx] = m3_sat_i32(acc00);
            h_row0[state_idx + 1] = m3_sat_i32(acc01);
            h_row1[state_idx] = m3_sat_i32(acc10);
            h_row1[state_idx + 1] = m3_sat_i32(acc11);
          }
          for (; state_idx < (int)N; ++state_idx) {
            int64_t acc0 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx], M3_WIDE_FRAC_BITS);
            int64_t acc1 = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx], M3_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx * R];
            const int8_t *xk0 = x_base0;
            const int8_t *xk1 = x_base1;
            for (int key_idx = 0; key_idx < (int)Chunk_size; ++key_idx) {
              for (int r = 0; r < (int)R; ++r) {
                acc0 += (int64_t)dB_col[r] * (int64_t)xk0[r];
                acc1 += (int64_t)dB_col[r] * (int64_t)xk1[r];
              }
              dB_col += NR;
              xk0 += d_inner;
              xk1 += d_inner;
            }
            h_row0[state_idx] = m3_sat_i32(acc0);
            h_row1[state_idx] = m3_sat_i32(acc1);
          }
        }
        for (; feature_idx < feature_hi; ++feature_idx) {
          int32_t *h_row = &h_state_head[feature_idx * N];
          const int8_t *x_base = x_chunk_base + feature_idx * R;
          for (int state_idx = 0; state_idx < (int)N; ++state_idx) {
            int64_t acc = m3_asr((int64_t)decay_full_chunk * (int64_t)h_row[state_idx], M3_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx * R];
            const int8_t *xk = x_base;
            for (int key_idx = 0; key_idx < (int)Chunk_size; ++key_idx) {
              for (int r = 0; r < (int)R; ++r)
                acc += (int64_t)dB_col[r] * (int64_t)xk[r];
              dB_col += NR;
              xk += d_inner;
            }
            h_row[state_idx] = m3_sat_i32(acc);
          }
        }
      }

      if (my_lane == 0)
        *theta_cum_ptr = theta_cum;
    }
  }

  pi_cl_team_barrier();
}
