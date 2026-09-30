# `PULP_SelectiveScan_i8_i8` — Instrumented backup (PULPOpen)

This is the **instrumented backup** of the PULPOpen (Siracusa) selective-scan
kernel: the profiling is **baked in and active** — nothing is commented out — so
this version builds and runs directly to profile. The de-instrumented production
kernel lives in [SelectiveScan.c](SelectiveScan.c).

It uses **`perf_utils.h`** (`perf_bench_*`, `perf_stats_t`).

## What the instrumentation measures

A **whole-kernel** `perf_bench_*` profiler. Each core records its own slot
between `perf_bench_start()` (placed after the `h_buffer` zeroing, so
initialisation is excluded) and `perf_bench_stop()` (placed after the h-state
write-back, so real cost is included). `_ssm_perf_report` (core 0 only, `noinline`
to keep the cold printf/aggregation code out of the hot inner-loop's 4-KB
I-cache) then aggregates across the cluster — wall-clock `cycles` = **max** over
cores, every event field **summed** then divided by `NUM_CORES` for the per-core
average — and prints per-core cycles, instructions, loads/stores, load/jump
stalls, I-cache misses and TCDM contention, plus **cycles / D-step** and stall
rates per 1k instructions. The measured math is bit-identical to the production
kernel.

## Instrumented kernel

```c
/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "perf_utils.h"

// Q15 fixed-point domain for dA, dB, h
#define SSM_WIDE_FRAC_BITS 15
#define SSM_EXP_RANGE_Q15 ((int64_t)20 << SSM_WIDE_FRAC_BITS)
#define SSM_EXP_HALF_Q15 ((int64_t)128)
#define SSM_EXP_STEP_LOG2_Q15 8

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

// Per-core profiling slots: each core writes its own slot to avoid write
// conflicts.
PI_L1 static perf_stats_t _ssm_perf_slots[NUM_CORES];

// Aggregate and report cluster-wide performance statistics. noinline keeps the
// cold printf/aggregation code from aliasing the hot inner-loop lines in the
// shared 4-KB I-cache.
__attribute__((noinline)) static void
_ssm_perf_report(uint32_t B_size, uint32_t L, uint32_t D_inner, uint32_t N) {
  perf_stats_t agg = {0};
  for (int _i = 0; _i < NUM_CORES; _i++) {
    if (_ssm_perf_slots[_i].cycles > agg.cycles)
      agg.cycles = _ssm_perf_slots[_i].cycles;
    agg.instr += _ssm_perf_slots[_i].instr;
    agg.ld += _ssm_perf_slots[_i].ld;
    agg.st += _ssm_perf_slots[_i].st;
    agg.ld_stall += _ssm_perf_slots[_i].ld_stall;
    agg.jmp_stall += _ssm_perf_slots[_i].jmp_stall;
    agg.imiss += _ssm_perf_slots[_i].imiss;
    agg.branch += _ssm_perf_slots[_i].branch;
    agg.taken_branch += _ssm_perf_slots[_i].taken_branch;
    agg.rvc += _ssm_perf_slots[_i].rvc;
    agg.ld_ext += _ssm_perf_slots[_i].ld_ext;
    agg.st_ext += _ssm_perf_slots[_i].st_ext;
    agg.ld_ext_cyc += _ssm_perf_slots[_i].ld_ext_cyc;
    agg.st_ext_cyc += _ssm_perf_slots[_i].st_ext_cyc;
    agg.tcdm_cont += _ssm_perf_slots[_i].tcdm_cont;
  }
  // perf_bench_print() computes IPC = instr/cycles for single-core data; we
  // hold agg.cycles = MAX(cores) and agg.instr = SUM(cores), so divide every
  // SUM field by NUM_CORES first for a per-core average.
  perf_stats_t per_core = agg;
  per_core.instr /= NUM_CORES;
  per_core.ld /= NUM_CORES;
  per_core.st /= NUM_CORES;
  per_core.ld_stall /= NUM_CORES;
  per_core.jmp_stall /= NUM_CORES;
  per_core.imiss /= NUM_CORES;
  per_core.branch /= NUM_CORES;
  per_core.taken_branch /= NUM_CORES;
  per_core.rvc /= NUM_CORES;
  per_core.ld_ext /= NUM_CORES;
  per_core.st_ext /= NUM_CORES;
  per_core.ld_ext_cyc /= NUM_CORES;
  per_core.st_ext_cyc /= NUM_CORES;
  per_core.tcdm_cont /= NUM_CORES;
  perf_bench_print("PULP_SelectiveScan (per-core avg)", &per_core);

  const uint32_t total_dsteps =
      (uint32_t)B_size * (uint32_t)L * (uint32_t)D_inner;
  const uint32_t cpd_milli =
      (total_dsteps > 0)
          ? (uint32_t)((uint64_t)agg.cycles * 1000u / total_dsteps)
          : 0u;
  const uint32_t ld_pki =
      (agg.instr > 0) ? (uint32_t)((uint64_t)agg.ld_stall * 1000u / agg.instr)
                      : 0u;
  const uint32_t jmp_pki =
      (agg.instr > 0) ? (uint32_t)((uint64_t)agg.jmp_stall * 1000u / agg.instr)
                      : 0u;
  const uint32_t ims_pki =
      (agg.instr > 0) ? (uint32_t)((uint64_t)agg.imiss * 1000u / agg.instr)
                      : 0u;
  const uint32_t tcd_pki =
      (agg.instr > 0) ? (uint32_t)((uint64_t)agg.tcdm_cont * 1000u / agg.instr)
                      : 0u;
  printf("\n=== SSM Bottleneck Analysis (B=%u L=%u D=%u N=%u) ===\n",
         (unsigned)B_size, (unsigned)L, (unsigned)D_inner, (unsigned)N);
  printf("Cycles (wall-clock):     %10u\n", (unsigned)agg.cycles);
  printf("Cycles / D-step:         %7u.%03u\n", cpd_milli / 1000u,
         cpd_milli % 1000u);
  printf("Stall rates (/1k instr, cluster SUM / cluster SUM):\n");
  printf("  Load stalls:           %5u\n", (unsigned)ld_pki);
  printf("  Jump stalls:           %5u\n", (unsigned)jmp_pki);
  printf("  I-cache misses:        %5u\n", (unsigned)ims_pki);
  printf("  TCDM contention:       %5u\n", (unsigned)tcd_pki);
  printf("================================================\n");
}

// noinline: the jalr it emits inside the n-loop blocks the XpulpV2 hw-loop
// pass, whose lp.setup software-pipelining silently drops h-state stores.
__attribute__((noinline)) static int32_t _ssm_exp_lut_lookup(uint32_t idx) {
  return (int32_t)SelectiveScan_exp_lut_qwide[idx];
}

void PULP_SelectiveScan_i8_i8(
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

  // Configure hardware performance counters on this core.
  perf_bench_init();

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

  // Start measurement after the h_buffer zeroing so initialisation overhead is
  // excluded.
  perf_bench_start();

  if (D_start >= D_end) {
    perf_bench_stop();
    perf_bench_read(&_ssm_perf_slots[core_id]);
    pi_cl_team_barrier();
    return;
  }

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

      for (uint32_t t = 0; t < L; t++) {

        const int32_t *B_row = B + (b * L + t) * N;
        const int32_t *C_row = C + (b * L + t) * N;
        const int32_t x_val = (int32_t)x[(b * L + t) * D_inner + d];
        const int32_t z_val = (int32_t)z[(b * L + t) * D_inner + d];
        const int32_t dt_val = (int32_t)dt[(b * L + t) * D_inner + d];
        int64_t y_acc = 0;

        for (uint32_t n = 0; n < N; n++) {
          // dB path: independent of the LUT, computed first to overlap with the
          // LUT-call latency.
          const int64_t dB_q15 = ((int64_t)dt_val * B_row[n]) >> 8;
          const int64_t dB_x_q15 = dB_q15 * (int64_t)x_val;
          // dA path: exp LUT lookup.
          const int64_t prod_dtA = (int64_t)dt_val * A_local[n];
          const int64_t dt_A_q15 = prod_dtA >> 8;
          const int64_t dt_A_clipped =
              dt_A_q15 < -(int64_t)SSM_EXP_RANGE_Q15
                  ? -(int64_t)SSM_EXP_RANGE_Q15
                  : (dt_A_q15 > (int64_t)0 ? (int64_t)0 : dt_A_q15);
          const uint32_t exp_idx =
              (uint32_t)((int32_t)dt_A_clipped +
                         (int32_t)(SSM_EXP_RANGE_Q15 + SSM_EXP_HALF_Q15)) >>
              SSM_EXP_STEP_LOG2_Q15;
          const int32_t dA_q15 = _ssm_exp_lut_lookup(exp_idx);
          int64_t h_acc = ((int64_t)dA_q15 * h_local[n]) >> SSM_WIDE_FRAC_BITS;
          h_acc += dB_x_q15;
          const int32_t h_new = _ssm_sat_i32_i64(h_acc);
          y_acc += ((int64_t)h_new * (int64_t)C_row[n]) >> SSM_WIDE_FRAC_BITS;
          h_local[n] = h_new;
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

      // Persist h state for the next L-tile
      for (uint32_t n = 0; n < N; n++)
        h_row_ptr[n] = h_local[n];
    }
  }

  // Stop measurement (includes the h-state write-back, part of the kernel's
  // real cost).
  perf_bench_stop();
  perf_bench_read(&_ssm_perf_slots[core_id]);

  // Mandatory Deeploy inter-kernel sync point (also guarantees every core's
  // slot is committed).
  pi_cl_team_barrier();

  if (core_id == 0)
    _ssm_perf_report(B_size, L, D_inner, N);
}
```
