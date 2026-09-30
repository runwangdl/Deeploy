# `GAP9_SelectiveScan_i8_i8` — Instrumented backup (GAP9)

This is the **instrumented backup** of the GAP9 selective-scan kernel: the
profiling is **baked in and active** — no `#if` guards, nothing commented out —
so this version builds and runs directly to profile. The de-instrumented
production kernel lives in [SelectiveScan.c](SelectiveScan.c).

It uses **`perf_utils_gap9.h`**, a dedicated GAP9 instrumentation layer
(`perf_bench_*`, `pi_perf_*`, `perf_stats_t`, `PI_PERF_*`), distinct from the
PULPOpen `perf_utils.h`.

## What the instrumentation measures

The **temporal-recurrence hot-loop profiler**. It measures the **t-loop only** —
the loop-carried scan `h[t] = dA·h[t-1] + dB·x ; y[t] = C·h[t]` — and **excludes**
the per-`d` A/h load and write-back, the `h_buffer` zeroing, and the cluster
barriers. Counter snapshots (`_ssm_hot_snap`) are taken **outside** the t-loop
body so the hot loop's own instruction stream / I-cache footprint is unperturbed;
deltas accumulate across every `(b,d)` iteration and every tile invocation. The
reporter (`_ssm_hotloop_report`, `noinline` to stay out of the hot I-cache
window, core 0 only) prints, for the whole scan: cumulative cycles and
instructions (summed over cores), **per-core IPC** as `Sum(instr)/Sum(cycles)`
(ceiling ≈ 1.0 on single-issue RI5CY), loads/stores, branches/RVC, and
load-stall / jump-stall / I-cache-miss rates per 1k instructions. The measured
math is bit-identical to the production kernel.

## Instrumented kernel

```c
/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "perf_utils_gap9.h"

// Q15 fixed-point domain for dA, dB, h
#define SSM_WIDE_FRAC_BITS 15
#define SSM_EXP_RANGE_Q15 ((int64_t)20 << SSM_WIDE_FRAC_BITS)
#define SSM_EXP_HALF_Q15 ((int64_t)128)
#define SSM_EXP_STEP_LOG2_Q15 8

// Loop unrolling factor for the inner n-loop.
enum { SSM_N_UNROLL = 1 };

// Saturating narrow of the int64 state accumulator to int32
static inline int32_t _ssm_sat_i32_i64(int64_t x) {
  const int32_t lo = (int32_t)x;
  const int32_t hi = (int32_t)(x >> 32);
  if (__builtin_expect(hi == (lo >> 31), 1))
    return lo;
  return (hi >= 0) ? INT32_MAX : INT32_MIN;
}

// Symmetric rounding right-shift
static inline int64_t _ssm_round_shift_i64(int64_t x, int s) {
  if (s <= 0)
    return x;
  const int64_t half = (int64_t)1 << (s - 1);
  return (x >= 0) ? ((x + half) >> s) : -(((-x) + half) >> s);
}

// dB is computed before the LUT load so the multiplies overlap the LUT load
// latency.
static inline int32_t _ssm_h_update(int32_t A_n, int32_t B_n, int32_t h_n,
                                    int32_t dt_val, int32_t x_val) {
  const int64_t dB_q15 = ((int64_t)dt_val * B_n) >> 8;
  const int64_t dB_x_q15 = dB_q15 * (int64_t)x_val;
  const int64_t prod_dtA = (int64_t)dt_val * A_n;
  const int64_t dt_A_q15 = prod_dtA >> 8;
  const int64_t dt_A_clipped =
      dt_A_q15 < -(int64_t)SSM_EXP_RANGE_Q15
          ? -(int64_t)SSM_EXP_RANGE_Q15
          : (dt_A_q15 > (int64_t)0 ? (int64_t)0 : dt_A_q15);
  const uint32_t exp_idx =
      (uint32_t)((int32_t)dt_A_clipped +
                 (int32_t)(SSM_EXP_RANGE_Q15 + SSM_EXP_HALF_Q15)) >>
      SSM_EXP_STEP_LOG2_Q15;
  const int32_t dA_q15 = SelectiveScan_exp_lut_qwide[exp_idx];
  int64_t h_acc = ((int64_t)dA_q15 * h_n) >> SSM_WIDE_FRAC_BITS;
  h_acc += (int64_t)dB_x_q15;
  return _ssm_sat_i32_i64(h_acc);
}

static inline int64_t _ssm_y_contrib(int32_t h_new, int32_t C_n) {
  return ((int64_t)h_new * (int64_t)C_n) >> SSM_WIDE_FRAC_BITS;
}

// Per-core accumulator for the t-loop region, summed over every (b,d) and every
// tile. Lives in L1.
PI_L1 static perf_stats_t _ssm_hot_slots[NUM_CORES];

// Cumulative snapshot of the counters for the hot-loop IPC. PI_PERF_CYCLES is a
// free-running timer and the PCCR events accumulate since pi_perf_start(), so
// interval counts are end-start deltas. CYCLES is read last so the cheap reads
// above are not billed to the interval.
static inline void _ssm_hot_snap(perf_stats_t *s) {
  s->instr = pi_perf_read(PI_PERF_INSTR);
  s->ld = pi_perf_read(PI_PERF_LD);
  s->st = pi_perf_read(PI_PERF_ST);
  s->ld_stall = pi_perf_read(PI_PERF_LD_STALL);
  s->jmp_stall = pi_perf_read(PI_PERF_JR_STALL);
  s->imiss = pi_perf_read(PI_PERF_IMISS);
  s->branch = pi_perf_read(PI_PERF_BRANCH);
  s->rvc = pi_perf_read(PI_PERF_RVC);
  s->cycles = pi_perf_read(PI_PERF_CYCLES);
}

// Cold reporter (noinline). Prints the cumulative t-loop stats across all tile
// calls so far.
__attribute__((noinline)) static void
_ssm_hotloop_report(uint32_t B_size, uint32_t L, uint32_t D_inner, uint32_t N) {
  perf_stats_t agg = {0};
  for (int i = 0; i < NUM_CORES; i++) {
    agg.cycles += _ssm_hot_slots[i].cycles;
    agg.instr += _ssm_hot_slots[i].instr;
    agg.ld += _ssm_hot_slots[i].ld;
    agg.st += _ssm_hot_slots[i].st;
    agg.ld_stall += _ssm_hot_slots[i].ld_stall;
    agg.jmp_stall += _ssm_hot_slots[i].jmp_stall;
    agg.imiss += _ssm_hot_slots[i].imiss;
    agg.branch += _ssm_hot_slots[i].branch;
    agg.rvc += _ssm_hot_slots[i].rvc;
  }
  const uint32_t ipc_milli =
      agg.cycles ? (uint32_t)((uint64_t)agg.instr * 1000u / agg.cycles) : 0u;
  const uint32_t ld_pki =
      agg.instr ? (uint32_t)((uint64_t)agg.ld_stall * 1000u / agg.instr) : 0u;
  const uint32_t jmp_pki =
      agg.instr ? (uint32_t)((uint64_t)agg.jmp_stall * 1000u / agg.instr) : 0u;
  const uint32_t ims_pki =
      agg.instr ? (uint32_t)((uint64_t)agg.imiss * 1000u / agg.instr) : 0u;
  printf("\n=== SSM temporal-recurrence hot-loop (t-loop only, cumulative) "
         "B=%u L=%u D=%u N=%u ===\n",
         (unsigned)B_size, (unsigned)L, (unsigned)D_inner, (unsigned)N);
  printf("  Hot cycles (sum cores):  %10u\n", (unsigned)agg.cycles);
  printf("  Hot instr  (sum cores):  %10u\n", (unsigned)agg.instr);
  printf("  IPC per-core (sum/sum):  %7u.%03u\n", ipc_milli / 1000u,
         ipc_milli % 1000u);
  printf("  Loads/Stores:            %10u / %u\n", (unsigned)agg.ld,
         (unsigned)agg.st);
  printf("  Branches / RVC:          %10u / %u\n", (unsigned)agg.branch,
         (unsigned)agg.rvc);
  printf("  Stalls /1k instr:        ld=%u jr=%u imiss=%u\n", (unsigned)ld_pki,
         (unsigned)jmp_pki, (unsigned)ims_pki);
  printf("================================================================\n");
}

void GAP9_SelectiveScan_i8_i8(
    const int8_t *__restrict__ x, const int8_t *__restrict__ z,
    const int16_t *__restrict__ dt, const int32_t *__restrict__ B,
    const int32_t *__restrict__ C, const int32_t *__restrict__ A,
    const int32_t *__restrict__ D_skip, int8_t *__restrict__ y,
    int32_t *__restrict__ h_buffer, const int32_t *__restrict__ gate_lut,
    uint32_t B_size, uint32_t L, uint32_t D_inner, uint32_t N,
    int32_t output_requant_mul_q40, uint32_t is_first_L_tile) {
  const int8_t core_id = pi_core_id();
  const int8_t log2Core = LOG2(NUM_CORES);

  const uint32_t D_chunk =
      (D_inner >> log2Core) + ((D_inner & (NUM_CORES - 1)) != 0);
  const uint32_t D_start = MIN(core_id * D_chunk, D_inner);
  const uint32_t D_end = MIN(D_start + D_chunk, D_inner);

  // Zero h_buffer at the start of each new L sequence
  if (is_first_L_tile) {
    for (uint32_t b = 0; b < B_size; b++) {
      for (uint32_t d = D_start; d < D_end; d++) {
        int32_t *h_row = h_buffer + (b * D_inner + d) * N;
        for (uint32_t n = 0; n < N; n++)
          h_row[n] = 0;
      }
    }
    pi_cl_team_barrier();
  }

  if (D_start >= D_end) {
    pi_cl_team_barrier();
    return;
  }

  // Configure and start the performance counters. No pi_perf_reset(): the
  // hot-loop figures are end-start deltas around the t-loop, so no reset is
  // needed.
  perf_bench_init();
  pi_perf_start();
  perf_stats_t _hot = {0};
  perf_stats_t _s0, _s1;

  // Main scan: loop order (b, d, t, n).
  for (uint32_t b = 0; b < B_size; b++) {
    for (uint32_t d = D_start; d < D_end; d++) {

      // A[d,:] is t-invariant: hoist to a stack array reused for all L steps.
      int32_t A_local[16];
      {
        const int32_t *A_row = A + d * N;
        for (uint32_t n = 0; n < N; n++)
          A_local[n] = A_row[n];
      }

      // Load persistent state into a stack array; written back after the
      // t-loop.
      int32_t h_local[16];
      int32_t *const h_row_ptr = h_buffer + (b * D_inner + d) * N;
      for (uint32_t n = 0; n < N; n++)
        h_local[n] = h_row_ptr[n];

      // D_skip[d] is t-invariant: hoist outside the t-loop to save one L1 load
      // per step.
      const int32_t d_skip_val = D_skip[d];

      _ssm_hot_snap(&_s0); // snapshot just before the temporal recurrence
      for (uint32_t t = 0; t < L; t++) {

        const int32_t *B_row = B + (b * L + t) * N;
        const int32_t *C_row = C + (b * L + t) * N;
        const int32_t x_val = (int32_t)x[(b * L + t) * D_inner + d];
        const int32_t z_val = (int32_t)z[(b * L + t) * D_inner + d];
        const int32_t dt_val = (int32_t)dt[(b * L + t) * D_inner + d];
        int64_t y_acc = 0;

        uint32_t n = 0;
        for (; n + (uint32_t)SSM_N_UNROLL <= N; n += (uint32_t)SSM_N_UNROLL) {
          for (uint32_t u = 0; u < (uint32_t)SSM_N_UNROLL; u++) {
            h_local[n + u] = _ssm_h_update(A_local[n + u], B_row[n + u],
                                           h_local[n + u], dt_val, x_val);
            y_acc += _ssm_y_contrib(h_local[n + u], C_row[n + u]);
          }
        }

        y_acc += (int64_t)d_skip_val * (int64_t)x_val;
        const int64_t gate_q13 = (int64_t)gate_lut[z_val + 128];
        const int64_t y_gated = _ssm_round_shift_i64(y_acc * gate_q13, 13);
        int64_t y_out =
            _ssm_round_shift_i64(y_gated * (int64_t)output_requant_mul_q40, 40);
        if (y_out > 127)
          y_out = 127;
        if (y_out < -128)
          y_out = -128;
        y[(b * L + t) * D_inner + d] = (int8_t)y_out;
      }
      _ssm_hot_snap(&_s1); // snapshot just after the temporal recurrence
      _hot.cycles += _s1.cycles - _s0.cycles;
      _hot.instr += _s1.instr - _s0.instr;
      _hot.ld += _s1.ld - _s0.ld;
      _hot.st += _s1.st - _s0.st;
      _hot.ld_stall += _s1.ld_stall - _s0.ld_stall;
      _hot.jmp_stall += _s1.jmp_stall - _s0.jmp_stall;
      _hot.imiss += _s1.imiss - _s0.imiss;
      _hot.branch += _s1.branch - _s0.branch;
      _hot.rvc += _s1.rvc - _s0.rvc;

      // Persist h state for the next L-tile
      for (uint32_t n = 0; n < N; n++)
        h_row_ptr[n] = h_local[n];
    }
  }

  // Fold this tile's per-(b,d) hot-loop totals into the cross-tile per-core
  // slot.
  _ssm_hot_slots[core_id].cycles += _hot.cycles;
  _ssm_hot_slots[core_id].instr += _hot.instr;
  _ssm_hot_slots[core_id].ld += _hot.ld;
  _ssm_hot_slots[core_id].st += _hot.st;
  _ssm_hot_slots[core_id].ld_stall += _hot.ld_stall;
  _ssm_hot_slots[core_id].jmp_stall += _hot.jmp_stall;
  _ssm_hot_slots[core_id].imiss += _hot.imiss;
  _ssm_hot_slots[core_id].branch += _hot.branch;
  _ssm_hot_slots[core_id].rvc += _hot.rvc;

  pi_cl_team_barrier();

  if (core_id == 0)
    _ssm_hotloop_report(B_size, L, D_inner, N);
}
```
