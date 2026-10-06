/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

// RW: Remove MAINSTACKSIZE because gap9-sdk does not use it
// Slave stack size. Filippo Cordella measured 288M -> 180M cycles on the full
// Femba model by moving these stacks into L1 TCDM and shrinking them from
// 8192 B to 768 B/core: SelectiveScan hits the stack hard, so stack locality
// dominates. The L3 tiling closures run on the controller core (cc_stack), not
// on these, so 768 B is enough.
// Overridable via -DSLAVESTACKSIZE=<n> from CMake (the #ifndef keeps a
// command-line define from tripping "redefined" under -Werror).
// 512 B/core is what my own GAP9 training runs used successfully
// (result_check.md: "CC 4096 / slave 512", --l1 122000). 9 cores x 512 = 4608 B,
// leaving 131072 - 4608 - 4096 = 122368 B of TCDM for the tiler, hence --l1 122000.
#ifndef SLAVESTACKSIZE
#define SLAVESTACKSIZE 768
#endif

/* On-chip memory windows. HyperRAM/L3 (cl_ram_malloc) is NOT CPU-addressable, so a raw memcpy / CPU-deref of
 * an L3 pointer faults on real silicon -- GVSoC models HyperRAM as flat RAM and hides it. The previous
 * `>= 0x10000000` / `< 0x10000000` tests also matched HyperRAM. (Same as TrainDeeploy's harness.) */
#define IS_L1(ptr) ((uint32_t)(ptr) >= 0x10000000u && (uint32_t)(ptr) < 0x10040000u)
#define IS_L2(ptr) (((uint32_t)(ptr) >= 0x1C000000u && (uint32_t)(ptr) < 0x1C200000u) || IS_L1(ptr))

// Run gvsoc at the same operating point as the EVK. The SDK default (cluster 50 MHz) makes every PSRAM wait
// ~7x cheaper in cycles: m2_anna_s0_b0static 65.0 M on gvsoc@default vs 107.8 M on gvsoc@370 MHz vs 108.8 M
// on the EVK. -DGVSOC_DEFAULT_FREQ restores the old behaviour.
#ifndef GVSOC_DEFAULT_FREQ
#define GVSOC_SET_FREQ
#endif
/* Board operating point (the SDK boots at a low safe clock). 370 MHz needs 0.8 V. Override with -DFREQ_FC=... */
#ifndef FREQ_FC
#define FREQ_FC 370
#endif
#ifndef FREQ_CL
#define FREQ_CL 370
#endif
#ifndef FREQ_PE
#define FREQ_PE 370
#endif
#ifndef VOLTAGE
#define VOLTAGE 800
#endif

#define CLUSTER_MAX_CORES 9
PI_L1 uint8_t cluster_slave_stacks[SLAVESTACKSIZE * CLUSTER_MAX_CORES]
    __attribute__((aligned(16)));
#define SET_SLAVE_STACK(t) \
  do { (t).slave_stack_size = SLAVESTACKSIZE; (t).stacks = cluster_slave_stacks; } while(0)

#ifdef POWER_MEASUREMENT
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs, x)
#endif

struct pi_device cluster_dev;
uint32_t total_cycles = 0;

typedef struct {
  void *expected;
  void *actual;
  int num_elements;
  int output_buf_index;
  int *err_count;
} FloatCompareArgs;

void CompareFloatOnCluster(void *args) {

  if (pi_core_id() == 0) {
    FloatCompareArgs *compare_args = (FloatCompareArgs *)args;
    float *expected = (float *)compare_args->expected;
    float *actual = (float *)compare_args->actual;
    int num_elements = compare_args->num_elements;
    int output_buf_index = compare_args->output_buf_index;
    int *err_count = compare_args->err_count;

    int local_err_count = 0;

    for (int i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;

      if ((diff < -1e-4) || (diff > 1e-4) || isnan(diff)) {
        local_err_count += 1;

        printf("Expected: %10.6f  ", expected_val);
        printf("Actual: %10.6f  ", actual_val);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", diff, i,
               output_buf_index);
      }
    }

    *err_count = local_err_count;
  }
}

void CL_CompareFloat(void *arg) {
  pi_cl_team_fork(NUM_CORES, CompareFloatOnCluster, arg);
}

void InitNetworkWrapper(void *args) {
  (void)args;
  InitNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunNetworkWrapper(void *args) {
  (void)args;
  // Initialize performance counter in cluster context
  ResetTimer();
  StartTimer();
  RunNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
  total_cycles = getCycles();
  StopTimer();
}

int main(void) {

#ifdef POWER_MEASUREMENT
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
#endif

#ifndef CI
  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
  printf("[%d %d] Hello World!\n", cluster_id, core_id);
#endif
  struct pi_cluster_conf conf;

  pi_cluster_conf_init(&conf);
  conf.id = 0;
#ifndef CC_STACK_SIZE
#define CC_STACK_SIZE 4096
#endif
  conf.cc_stack_size = CC_STACK_SIZE;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

#if defined(__PLATFORM_BOARD__) || defined(GVSOC_SET_FREQ)
  pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE * 1000 * 1000);
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
  printf("[FREQ] FC=%d CL=%d PE=%d Hz, %d mV\r\n", pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL),
         pi_freq_get(PI_FREQ_DOMAIN_PERIPH), VOLTAGE);
#endif

  mem_init();
#ifndef NOFLASH
  open_fs();
#endif

  printf("Intializing\r\n");

  struct pi_cluster_task cluster_task;

  pi_cluster_task(&cluster_task, InitNetworkWrapper, NULL);
  SET_SLAVE_STACK(cluster_task);
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifndef CI
  printf("Initialized\r\n");
#endif
  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    if (IS_L2(DeeployNetwork_inputs[buf])) {
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
             DeeployNetwork_inputs_bytes[buf]);
    }
  }

#ifndef CI
  printf("Input copied\r\n");
#endif

  pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
  SET_SLAVE_STACK(cluster_task);

#ifdef POWER_MEASUREMENT
  WRITE_GPIO(1);
#endif

  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifdef POWER_MEASUREMENT
  WRITE_GPIO(0);
#endif

#ifndef CI
  printf("Output:\r\n");
#endif

  uint32_t tot_err, tot_tested;
  tot_err = 0;
  tot_tested = 0;
  void *compbuf;
  FloatCompareArgs float_compare_args;
  uint32_t float_error_count = 0;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot_tested += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);

    if (!IS_L2(DeeployNetwork_outputs[buf])) {
      compbuf = pi_l2_malloc(DeeployNetwork_outputs_bytes[buf]);
      ram_read(compbuf, DeeployNetwork_outputs[buf],
               DeeployNetwork_outputs_bytes[buf]);
    } else {
      compbuf = DeeployNetwork_outputs[buf];
    }

    if (ISOUTPUTFLOAT) {
      float_error_count = 0;
      float_compare_args.expected = testOutputVector[buf];
      float_compare_args.actual = compbuf;
      float_compare_args.num_elements =
          DeeployNetwork_outputs_bytes[buf] / sizeof(float);
      float_compare_args.output_buf_index = buf;
      float_compare_args.err_count = (int *)&float_error_count;

      pi_cluster_task(&cluster_task, CL_CompareFloat, &float_compare_args);
      SET_SLAVE_STACK(cluster_task);
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      tot_err += float_error_count;
    } else {

      for (uint32_t i = 0;
           i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
        OUTPUTTYPE expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
        OUTPUTTYPE actual = ((OUTPUTTYPE *)compbuf)[i];
        OUTPUTTYPE diff = expected - actual;

        if (diff) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if (!IS_L2(DeeployNetwork_outputs[buf])) {
      pi_l2_free(compbuf, DeeployNetwork_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", total_cycles);
  { // per-row checksums of named tensors (filled only by a patched Network.c, see patch_ck_named.py)
    extern int32_t sdbg_ck_r[8][80] __attribute__((weak)); extern int32_t sdbg_ck_n __attribute__((weak));
    if (&sdbg_ck_n) for (int k = 0; k < sdbg_ck_n; k++) {
      printf("SDBGCKR%d", k); for (int r = 0; r < 80; r++) printf(" %ld", (long)sdbg_ck_r[k][r]); printf("\r\n");
    }
  }
  { // per-node timestamps (filled only by a patched Network.c)
    extern uint32_t sdbg_nt[] __attribute__((weak)); extern int32_t sdbg_nt_n __attribute__((weak));
    extern const char *sdbg_nt_name[] __attribute__((weak));
    if (&sdbg_nt_n) for (int k = 0; k < sdbg_nt_n; k++) printf("NT %d %s %u\r\n", k, sdbg_nt_name[k], sdbg_nt[k]);
  }
  printf("Errors: %u out of %u \r\n", tot_err, tot_tested);

  return 0;
}