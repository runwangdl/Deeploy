/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Performance Counter Utilities for GAP9 Benchmarking
 */

#ifndef __PERF_UTILS_GAP9_H__
#define __PERF_UTILS_GAP9_H__

#include "pmsis.h"

// PI_PERF_* are already defined by the GAP9 SDK pmsis.h - do NOT redefine them here

// Benchmark statistics structure
typedef struct {
  unsigned int cycles;
  unsigned int instr;
  unsigned int ld;
  unsigned int st;
  unsigned int ld_stall;
  unsigned int jmp_stall;
  unsigned int imiss;
  unsigned int branch;
  unsigned int rvc;
  unsigned int ld_ext;
  unsigned int st_ext;
  unsigned int ld_ext_cyc;
  // st_ext_cyc:    not available on GAP9
  // tcdm_cont:     not available on GAP9
  // taken_branch:  not available on GAP9 (use branch only)
} perf_stats_t;

// Initialize performance counters for comprehensive benchmarking
static inline void perf_bench_init() {
  pi_perf_conf(
      (1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR) | (1 << PI_PERF_LD_STALL) |
      (1 << PI_PERF_JR_STALL) | (1 << PI_PERF_IMISS) | (1 << PI_PERF_LD) |
      (1 << PI_PERF_ST) | (1 << PI_PERF_JUMP) | (1 << PI_PERF_BRANCH) |
      (1 << PI_PERF_RVC) | (1 << PI_PERF_LD_EXT) |
      (1 << PI_PERF_ST_EXT) | (1 << PI_PERF_LD_EXT_CYC));
}

/* Per-core cycle baseline: GVSOC doesn't reset the counter on pi_perf_reset(); use a delta. */
PI_L1 static unsigned int _perf_bench_cycle_start[NUM_CORES];

// Start performance monitoring
static inline void perf_bench_start() {
  pi_perf_reset();
  pi_perf_start();
  _perf_bench_cycle_start[pi_core_id()] = pi_perf_read(PI_PERF_CYCLES);
}

// Stop performance monitoring
static inline void perf_bench_stop() { pi_perf_stop(); }

// Read all performance counters into structure
static inline void perf_bench_read(perf_stats_t *stats) {
  /* Elapsed cycles = current - baseline (correct on GVSOC and silicon). */
  stats->cycles     = pi_perf_read(PI_PERF_CYCLES) - _perf_bench_cycle_start[pi_core_id()];
  stats->instr        = pi_perf_read(PI_PERF_INSTR);
  stats->ld           = pi_perf_read(PI_PERF_LD);
  stats->st           = pi_perf_read(PI_PERF_ST);
  stats->ld_stall     = pi_perf_read(PI_PERF_LD_STALL);
  stats->jmp_stall    = pi_perf_read(PI_PERF_JR_STALL);
  stats->imiss        = pi_perf_read(PI_PERF_IMISS);
  stats->branch       = pi_perf_read(PI_PERF_BRANCH);
  stats->rvc          = pi_perf_read(PI_PERF_RVC);
  stats->ld_ext       = pi_perf_read(PI_PERF_LD_EXT);
  stats->st_ext       = pi_perf_read(PI_PERF_ST_EXT);
  stats->ld_ext_cyc   = pi_perf_read(PI_PERF_LD_EXT_CYC);
}

// core 0 only, to avoid clutter
static inline void perf_bench_print(const char *label, perf_stats_t *stats) {
  if (pi_core_id() == 0) {
    printf("\n=== Performance Statistics: %s ===\n", label);
    printf("Cycles:              %10u\n", stats->cycles);
    printf("Instructions:        %10u\n", stats->instr);
    printf("IPC:                 %10.3f\n",
           stats->cycles > 0 ? (float)stats->instr / stats->cycles : 0.0f);
    printf("\n--- Instruction Mix ---\n");
    printf("Loads:               %10u (%.2f%%)\n", stats->ld,
           stats->instr > 0 ? 100.0f * stats->ld / stats->instr : 0.0f);
    printf("Stores:              %10u (%.2f%%)\n", stats->st,
           stats->instr > 0 ? 100.0f * stats->st / stats->instr : 0.0f);
    printf("Branches:            %10u (%.2f%%)\n", stats->branch,
           stats->instr > 0 ? 100.0f * stats->branch / stats->instr : 0.0f);
    printf("Compressed (RVC):    %10u (%.2f%%)\n", stats->rvc,
           stats->instr > 0 ? 100.0f * stats->rvc / stats->instr : 0.0f);
    printf("\n--- Stalls & Hazards ---\n");
    printf("Load Stalls:         %10u\n", stats->ld_stall);
    printf("Jump Stalls:         %10u\n", stats->jmp_stall);
    printf("I-cache Misses:      %10u\n", stats->imiss);
    printf("\n--- Memory Hierarchy ---\n");
    printf("External Loads:      %10u (%.2f%%)\n", stats->ld_ext,
           stats->ld > 0 ? 100.0f * stats->ld_ext / stats->ld : 0.0f);
    printf("External Stores:     %10u (%.2f%%)\n", stats->st_ext,
           stats->st > 0 ? 100.0f * stats->st_ext / stats->st : 0.0f);
    printf("Ext Load Cycles:     %10u (avg: %.2f)\n", stats->ld_ext_cyc,
           stats->ld_ext > 0 ? (float)stats->ld_ext_cyc / stats->ld_ext : 0.0f);
    printf("========================================\n\n");
  }
}

// Compute difference between two stats (for analyzing specific code sections)
static inline void perf_bench_diff(perf_stats_t *result, perf_stats_t *end,
                                   perf_stats_t *start) {
  result->cycles       = end->cycles       - start->cycles;
  result->instr        = end->instr        - start->instr;
  result->ld           = end->ld           - start->ld;
  result->st           = end->st           - start->st;
  result->ld_stall     = end->ld_stall     - start->ld_stall;
  result->jmp_stall    = end->jmp_stall    - start->jmp_stall;
  result->imiss        = end->imiss        - start->imiss;
  result->branch       = end->branch       - start->branch;
  result->rvc          = end->rvc          - start->rvc;
  result->ld_ext       = end->ld_ext       - start->ld_ext;
  result->st_ext       = end->st_ext       - start->st_ext;
  result->ld_ext_cyc   = end->ld_ext_cyc   - start->ld_ext_cyc;
}

#endif // __PERF_UTILS_GAP9_H__
