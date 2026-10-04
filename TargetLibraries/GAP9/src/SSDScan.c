/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "SSDScan.h"

#include "SSDScanLUT.h"

// Exported for SSDScanNE16.c (the table itself stays private to this translation unit).
const int16_t *SSDScan_exp_lut_ptr = SSD_EXP_LUT;

// Q15 fixed-point domain (matches ssd_q.py)
#define SSD_WIDE_FRAC_BITS 15
#define SSD_Q20 (1 << 20)
#define SSD_EXP_STEP (SSD_Q20 / 128)
#define SSD_EXP_RANGE_Q20 ((int64_t)20 * SSD_Q20)
#define SSD_ONE_Q15 ((1 << SSD_WIDE_FRAC_BITS) - 1)
#define SSD_OUTPUT_REQUANT_SHIFT 40

// Arithmetic (floor) right shift
static inline int64_t ssd_asr(int64_t x, int s) {
  if (s <= 0)
    return x;
  return x >> s;
}

// Symmetric rounding right shift (final output requant)
static inline int64_t ssd_round_shift(int64_t x, int s) {
  if (s <= 0)
    return x;
  int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

// Saturate to int32 (state kept as int32 across chunks)
static inline int32_t ssd_sat_i32(int64_t x) {
  if (x > (int64_t)2147483647)
    return (int32_t)2147483647;
  if (x < -(int64_t)2147483648)
    return (int32_t)(-2147483648);
  return (int32_t)x;
}

// Saturate to int8
static inline int8_t ssd_sat_i8(int64_t x) {
  if (x > 127)
    return (int8_t)127;
  if (x < -128)
    return (int8_t)(-128);
  return (int8_t)x;
}

// exp of a Q15 log-decay arg (<= 0) -> Q15 in (0,1], via SSD_EXP_LUT
__attribute__((always_inline)) static inline int32_t ssd_expq(int64_t z_q15) {
  int64_t arg_q20 = z_q15 * (int64_t)32; // z << (20 - 15)
  if (arg_q20 >= 0)
    return SSD_ONE_Q15;
  if (arg_q20 < -SSD_EXP_RANGE_Q20)
    arg_q20 = -SSD_EXP_RANGE_Q20;
  int64_t idx = (arg_q20 + SSD_EXP_RANGE_Q20 + (SSD_EXP_STEP / 2)) / SSD_EXP_STEP;
  if (idx < 0)
    idx = 0;
  if (idx > SSD_EXP_LUT_LEN - 1)
    idx = SSD_EXP_LUT_LEN - 1;
  return SSD_EXP_LUT[(int)idx];
}

enum { SSD_UNROLL = 1 };

void GAP9_SSDScan_i8_i8(
    const int8_t *__restrict__ x,
    const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt,
    const int32_t *__restrict__ B,
    const int32_t *__restrict__ C,
    const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip,
    int8_t *__restrict__ y,
    int32_t *__restrict__ h_state,
    const int32_t *__restrict__ gate_lut,
    uint32_t B_size,
    uint32_t Chunk_size,
    uint32_t N,
    uint32_t Head_dim,
    uint32_t Group_dim,
    uint32_t N_heads,
    uint32_t L,
    int32_t output_requant_mul_q40,
    uint32_t init_state)
{
  // M is per-head and L x L, split into L/Chunk_size chunks of Chunk_size x Chunk_size.
  const int number_of_chunks = L / Chunk_size;

  // A group shares B and C; decay is computed once per chunk and reused every timestep.
  const int heads_per_group = N_heads / Group_dim;
  const int d_inner = N_heads * Head_dim;
  const int head_state_size = Head_dim * N;

  const int core_id = (int)pi_core_id();

  // Adaptive partition: (head teams) x (Head_dim lanes). cores_per_head (C_h) minimises
  // the modelled critical path; a full tile picks C_h=1, a tail tile picks C_h>1.
  int64_t best_modelled_cost = -1;
  int cores_per_head = 1;
  {
    const int64_t chunk_triangular = (int64_t)Chunk_size * (Chunk_size + 1) / 2;
    const int64_t parallel_work = (int64_t)Head_dim * (chunk_triangular + 2 * (int64_t)Chunk_size * N);
    const int64_t spine_work = (int64_t)N * (3 * (int64_t)Chunk_size + chunk_triangular);
    for (int candidate_cores_per_head = 1; candidate_cores_per_head <= NUM_CORES; ++candidate_cores_per_head) {
      if (NUM_CORES % candidate_cores_per_head != 0) continue;
      if (candidate_cores_per_head > Head_dim) continue;
      const int candidate_head_teams = NUM_CORES / candidate_cores_per_head;
      const int64_t heads_per_team = ((int64_t)N_heads + candidate_head_teams - 1) / candidate_head_teams;
      const int64_t modelled_cost = heads_per_team * (spine_work + parallel_work / candidate_cores_per_head);
      if (best_modelled_cost < 0 || modelled_cost < best_modelled_cost) {
        best_modelled_cost = modelled_cost;
        cores_per_head = candidate_cores_per_head;
      }
    }
  }
  const int head_teams = NUM_CORES / cores_per_head;
  const int my_head_team = core_id / cores_per_head;
  const int my_lane = core_id % cores_per_head;

  // heads are independent: round-robin
  for (int head = my_head_team; head < N_heads; head += head_teams) {

    const int group = head / heads_per_group;
    const int64_t A_head = (int64_t)A[head];
    const int64_t D_head = (int64_t)D_skip[head];

    // contiguous Head_dim slice for this lane
    const int feature_lo = (int)(((int64_t)my_lane * Head_dim) / cores_per_head);
    const int feature_hi = (int)((((int64_t)my_lane + 1) * Head_dim) / cores_per_head);

    // per-head scratch, reused across batch and chunk
    int32_t dB_chunk[Chunk_size * N];                  // max |dB| ~ 4.2M, fits int32
    int32_t cumulative_log_decay[Chunk_size];          // Q15
    int32_t decay_from_chunk_start[Chunk_size];        // exp(cls[k])
    int32_t decay_to_chunk_end[Chunk_size];            // exp(cls[end]-cls[k])
    // one row of exp(cls[i]-cls[j]), rebuilt per query_idx to avoid a Chunk^2 matrix
    int32_t decay_matrix_row[Chunk_size];
    int32_t intra_chunk_scores[Chunk_size];            // score for (qi, ki)
    int32_t C_query_cache[N];                          // cached, reused across ki and n

    for (int batch = 0; batch < B_size; ++batch) {

      // per-(batch, head) state, [B, N_heads, Head_dim, N], carried across L-tiles
      int32_t *h_state_head = h_state + (int64_t)(batch * N_heads + head) * head_state_size;

      // init_state=1 zeroes state (new sequence); =0 continues a previous tile
      if (init_state) {
        for (int feature_idx = feature_lo; feature_idx < feature_hi; ++feature_idx) {
          int32_t *h_row = &h_state_head[feature_idx * N];
          for (int state_idx = 0; state_idx < N; ++state_idx)
            h_row[state_idx] = 0;
        }
      }

      for (int chunk = 0; chunk < number_of_chunks; ++chunk) {
        const int chunk_start = chunk * Chunk_size;

        // Phase 1: discretise B, build cumulative log-decay
        int32_t running_cumulative_log_decay = 0;
        for (int pos_in_chunk = 0; pos_in_chunk < Chunk_size; ++pos_in_chunk) {
          const int time_step = chunk_start + pos_in_chunk;
          const int64_t dt_value = (int64_t)dt[(batch * L + time_step) * N_heads + head];

          running_cumulative_log_decay += (int32_t)ssd_asr(dt_value * A_head, 8);
          cumulative_log_decay[pos_in_chunk] = running_cumulative_log_decay;

          const int32_t *B_row = &B[((batch * L + time_step) * Group_dim + group) * N];
          int32_t *dB_row = &dB_chunk[pos_in_chunk * N];
          int state_idx = 0;
          for (; state_idx + SSD_UNROLL <= N; state_idx += SSD_UNROLL) {
            for (int unroll_offset = 0; unroll_offset < SSD_UNROLL; ++unroll_offset)
              dB_row[state_idx + unroll_offset] = (int32_t)ssd_asr(dt_value * (int64_t)B_row[state_idx + unroll_offset], 8);
          }
          for (; state_idx < N; ++state_idx)
            dB_row[state_idx] = (int32_t)ssd_asr(dt_value * (int64_t)B_row[state_idx], 8);
        }

        // Phase 2: exp tables for phases 3 and 4 (decay_matrix_row is rebuilt per row in phase 3)
        for (int pos_in_chunk = 0; pos_in_chunk < Chunk_size; ++pos_in_chunk)
          decay_from_chunk_start[pos_in_chunk] = ssd_expq((int64_t)cumulative_log_decay[pos_in_chunk]);

        decay_to_chunk_end[Chunk_size - 1] = SSD_ONE_Q15;
        for (int pos_in_chunk = 0; pos_in_chunk < Chunk_size - 1; ++pos_in_chunk)
          decay_to_chunk_end[pos_in_chunk] = ssd_expq((int64_t)(cumulative_log_decay[Chunk_size - 1] - cumulative_log_decay[pos_in_chunk]));

        const int32_t decay_full_chunk = decay_from_chunk_start[Chunk_size - 1];

        // Phase 3: Y[i] = Y_diag[i] + Y_off[i] + D*x, requantised to int8
        for (int query_idx = 0; query_idx < Chunk_size; ++query_idx) {
          const int time_query = chunk_start + query_idx;
          const int32_t *C_query = &C[((batch * L + time_query) * Group_dim + group) * N];
          const int8_t *x_query = &x[(batch * L + time_query) * d_inner + head * Head_dim];
          const int8_t *z_query = &z[(batch * L + time_query) * d_inner + head * Head_dim];

          for (int state_idx = 0; state_idx < N; ++state_idx)
            C_query_cache[state_idx] = C_query[state_idx];

          // decay_matrix_row[ki] = exp(cls[qi]-cls[ki]) for ki<qi, ONE_Q15 at ki==qi
          for (int key_idx = 0; key_idx < query_idx; ++key_idx)
            decay_matrix_row[key_idx] = ssd_expq((int64_t)(cumulative_log_decay[query_idx] - cumulative_log_decay[key_idx]));
          decay_matrix_row[query_idx] = SSD_ONE_Q15;

          // Pass 3a: intra_chunk_scores[ki] = (C[qi].dB[ki] >> 15) * decay_matrix_row[ki] >> 15
          for (int key_idx = 0; key_idx <= query_idx; ++key_idx) {
            int64_t cb_dot_product = 0;
            const int32_t *dB_row = &dB_chunk[key_idx * N];
            int state_idx = 0;
            for (; state_idx + SSD_UNROLL <= N; state_idx += SSD_UNROLL) {
              for (int unroll_offset = 0; unroll_offset < SSD_UNROLL; ++unroll_offset)
                cb_dot_product += (int64_t)C_query_cache[state_idx + unroll_offset] * (int64_t)dB_row[state_idx + unroll_offset];
            }
            for (; state_idx < N; ++state_idx)
              cb_dot_product += (int64_t)C_query_cache[state_idx] * (int64_t)dB_row[state_idx];
            cb_dot_product = ssd_asr(cb_dot_product, SSD_WIDE_FRAC_BITS);
            intra_chunk_scores[key_idx] = (int32_t)ssd_asr((int64_t)cb_dot_product * decay_matrix_row[key_idx], SSD_WIDE_FRAC_BITS);
          }

          const int32_t query_decay_from_start = decay_from_chunk_start[query_idx];
          for (int feature_idx = feature_lo; feature_idx < feature_hi; ++feature_idx) {
            int64_t y_diag_p = 0;
            for (int key_idx = 0; key_idx <= query_idx; ++key_idx)
              y_diag_p += (int64_t)intra_chunk_scores[key_idx] *
                  (int64_t)x[(batch * L + chunk_start + key_idx) * d_inner + head * Head_dim + feature_idx];

            int64_t state_readout = 0;
            const int32_t *h_row = &h_state_head[feature_idx * N];
            {
              int state_idx = 0;
              for (; state_idx + SSD_UNROLL <= N; state_idx += SSD_UNROLL) {
                for (int unroll_offset = 0; unroll_offset < SSD_UNROLL; ++unroll_offset)
                  state_readout += (int64_t)C_query_cache[state_idx + unroll_offset] * (int64_t)h_row[state_idx + unroll_offset];
              }
              for (; state_idx < N; ++state_idx)
                state_readout += (int64_t)C_query_cache[state_idx] * (int64_t)h_row[state_idx];
            }
            state_readout = ssd_asr(state_readout, SSD_WIDE_FRAC_BITS);
            const int64_t y_offdiagonal = ssd_asr((int64_t)state_readout * query_decay_from_start, SSD_WIDE_FRAC_BITS);
            const int64_t y_accumulator = y_diag_p + y_offdiagonal + D_head * (int64_t)x_query[feature_idx];
            const int64_t gate_q13 = (int64_t)gate_lut[(int32_t)z_query[feature_idx] + 128];
            const int64_t y_gated = ssd_round_shift(y_accumulator * gate_q13, 13);
            y[(batch * L + time_query) * d_inner + head * Head_dim + feature_idx] = ssd_sat_i8(ssd_round_shift(y_gated * (int64_t)output_requant_mul_q40, SSD_OUTPUT_REQUANT_SHIFT));
          }
        }

        // Phase 4: scale dB_chunk[ki,:] by decay_to_chunk_end[ki] in place, then fold into h_state
        for (int key_idx = 0; key_idx < Chunk_size; ++key_idx) {
          const int32_t decay_to_end = decay_to_chunk_end[key_idx];
          int32_t *dB_row = &dB_chunk[key_idx * N];
          int state_idx = 0;
          for (; state_idx + SSD_UNROLL <= N; state_idx += SSD_UNROLL) {
            for (int unroll_offset = 0; unroll_offset < SSD_UNROLL; ++unroll_offset)
              dB_row[state_idx + unroll_offset] = (int32_t)ssd_asr((int64_t)decay_to_end * dB_row[state_idx + unroll_offset], SSD_WIDE_FRAC_BITS);
          }
          for (; state_idx < N; ++state_idx)
            dB_row[state_idx] = (int32_t)ssd_asr((int64_t)decay_to_end * dB_row[state_idx], SSD_WIDE_FRAC_BITS);
        }

        // h_state update: hottest loop, only this lane's features. Outer product
        // h[p][n] += sum_ki x[ki][p]*dB[ki][n] with a 2x2 register block; same summation
        // order as the plain loop and one ssd_sat_i32 after the full sum, so it stays bit-exact.
        const int8_t *x_chunk_base = &x[(batch * L + chunk_start) * d_inner + head * Head_dim];
        int feature_idx = feature_lo;
        for (; feature_idx + 1 < feature_hi; feature_idx += 2) {
          int32_t *h_row0 = &h_state_head[feature_idx * N];
          int32_t *h_row1 = h_row0 + N;
          const int8_t *x_base = x_chunk_base + feature_idx;
          int state_idx = 0;
          for (; state_idx + 1 < N; state_idx += 2) {
            int64_t acc00 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx], SSD_WIDE_FRAC_BITS);
            int64_t acc01 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx + 1], SSD_WIDE_FRAC_BITS);
            int64_t acc10 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx], SSD_WIDE_FRAC_BITS);
            int64_t acc11 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx + 1], SSD_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx];
            const int8_t *x_key = x_base;
            for (int key_idx = 0; key_idx < Chunk_size; ++key_idx) {
              const int32_t dB0 = dB_col[0];
              const int32_t dB1 = dB_col[1];
              const int32_t xv0 = (int32_t)x_key[0];
              const int32_t xv1 = (int32_t)x_key[1];
              acc00 += (int64_t)dB0 * (int64_t)xv0;
              acc01 += (int64_t)dB1 * (int64_t)xv0;
              acc10 += (int64_t)dB0 * (int64_t)xv1;
              acc11 += (int64_t)dB1 * (int64_t)xv1;
              dB_col += N;
              x_key += d_inner;
            }
            h_row0[state_idx] = ssd_sat_i32(acc00);
            h_row0[state_idx + 1] = ssd_sat_i32(acc01);
            h_row1[state_idx] = ssd_sat_i32(acc10);
            h_row1[state_idx + 1] = ssd_sat_i32(acc11);
          }
          for (; state_idx < N; ++state_idx) {
            int64_t acc0 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row0[state_idx], SSD_WIDE_FRAC_BITS);
            int64_t acc1 = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row1[state_idx], SSD_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx];
            const int8_t *x_key = x_base;
            for (int key_idx = 0; key_idx < Chunk_size; ++key_idx) {
              const int32_t dB0 = *dB_col;
              acc0 += (int64_t)dB0 * (int64_t)x_key[0];
              acc1 += (int64_t)dB0 * (int64_t)x_key[1];
              dB_col += N;
              x_key += d_inner;
            }
            h_row0[state_idx] = ssd_sat_i32(acc0);
            h_row1[state_idx] = ssd_sat_i32(acc1);
          }
        }
        for (; feature_idx < feature_hi; ++feature_idx) {
          int32_t *h_row = &h_state_head[feature_idx * N];
          const int8_t *x_base = x_chunk_base + feature_idx;
          for (int state_idx = 0; state_idx < N; ++state_idx) {
            int64_t state_accumulator = ssd_asr((int64_t)decay_full_chunk * (int64_t)h_row[state_idx], SSD_WIDE_FRAC_BITS);
            const int32_t *dB_col = &dB_chunk[state_idx];
            const int8_t *x_key = x_base;
            for (int key_idx = 0; key_idx < Chunk_size; ++key_idx) {
              state_accumulator += (int64_t)(*dB_col) * (int64_t)(*x_key);
              dB_col += N;
              x_key += d_inner;
            }
            h_row[state_idx] = ssd_sat_i32(state_accumulator);
          }
        }
      }
    }
  }

  pi_cl_team_barrier();
}
