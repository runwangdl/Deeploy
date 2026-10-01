/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void UniformRequantShift_s8_s8(int8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t chunk_start = MIN(chunk * core_id, size);
  int32_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  int8_t reg_data_in_A;
  int8_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

/* int8 -> uint8 (NE16 reads activations as unsigned bytes; see NE16UnsignedInputPass) */
void UniformRequantShift_s8_u8(int8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, uint8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, uint8_t output_min,
                               uint8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t chunk_start = MIN(chunk * core_id, size);
  int32_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  uint8_t out;
  int8_t reg_data_in_A;
  int8_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_u8_s8(uint8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t chunk_start = MIN(chunk * core_id, size);
  int32_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  uint8_t reg_data_in_A;
  uint8_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_s16_s8(int16_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t chunk_start = MIN(chunk * core_id, size);
  int32_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  int16_t reg_data_in_A;
  int16_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? (1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_s32_s32(int32_t *data_in, int32_t size, int32_t mul,
                                 int32_t add, int32_t *data_out, int32_t log2D,
                                 int32_t HW, int32_t input_offset,
                                 int32_t output_offset, int32_t output_min,
                                 int32_t output_max, bool rounding) {
  // int64 intermediate (int32 acc * mul overflows int32 for FEMBA scales); the rounding term
  // is hoisted and the loop is a plain 4x-unrolled stream (the old two-stream version spent
  // ~55 core-cycles per element on variable 64-bit shifts and clamps).
  const int32_t core_id = pi_core_id();
  const int32_t log2Core = LOG2(NUM_CORES);
  const int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  const int32_t start = MIN(chunk * core_id, size);
  const int32_t stop = MIN(start + chunk, size);
  const int64_t addr = (int64_t)add + ((rounding && log2D > 0) ? ((int64_t)1 << (log2D - 1)) : 0);
  const int64_t lo = output_min, hi = output_max;
  int32_t i = start;
  for (; i + 4 <= stop; i += 4) {
    int64_t v0 = (((int64_t)(data_in[i] + input_offset)) * mul + addr) >> log2D;
    int64_t v1 = (((int64_t)(data_in[i + 1] + input_offset)) * mul + addr) >> log2D;
    int64_t v2 = (((int64_t)(data_in[i + 2] + input_offset)) * mul + addr) >> log2D;
    int64_t v3 = (((int64_t)(data_in[i + 3] + input_offset)) * mul + addr) >> log2D;
    v0 = (v0 < lo) ? lo : ((v0 > hi) ? hi : v0);
    v1 = (v1 < lo) ? lo : ((v1 > hi) ? hi : v1);
    v2 = (v2 < lo) ? lo : ((v2 > hi) ? hi : v2);
    v3 = (v3 < lo) ? lo : ((v3 > hi) ? hi : v3);
    data_out[i] = (int32_t)v0 + output_offset;
    data_out[i + 1] = (int32_t)v1 + output_offset;
    data_out[i + 2] = (int32_t)v2 + output_offset;
    data_out[i + 3] = (int32_t)v3 + output_offset;
  }
  for (; i < stop; i++) {
    int64_t v = (((int64_t)(data_in[i] + input_offset)) * mul + addr) >> log2D;
    v = (v < lo) ? lo : ((v > hi) ? hi : v);
    data_out[i] = (int32_t)v + output_offset;
  }
}

void UniformRequantShift_s32_s8(int32_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t chunk_start = MIN(chunk * core_id, size);
  int32_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int64_t intermediate;  // 64-bit: int32 acc * mul overflows for FEMBA-scale requants (mul ~ 7e4)
  int8_t out;
  int32_t reg_data_in_A;
  int32_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = ((int64_t)reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? ((int64_t)1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = ((int64_t)reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? ((int64_t)1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = ((int64_t)reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? ((int64_t)1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = ((int64_t)reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((log2D > 0 ? ((int64_t)1 << (log2D - 1)) : 0)) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}
// x -> x + 128 as uint8 == x ^ 0x80, 4 bytes per op. Used in front of NE16 (NE16UnsignedInputPass
// emits RequantShift(mul 2, add 256, div 2) for it; the template routes that case here).
void Xor128_s8_u8(int8_t *data_in, int32_t size, uint8_t *data_out) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int32_t words = size >> 2;
  int32_t chunk = (words >> log2Core) + ((words & (NUM_CORES - 1)) != 0);
  int32_t start = MIN(chunk * core_id, words);
  int32_t stop = MIN(start + chunk, words);
  const uint32_t *in = (const uint32_t *)data_in;
  uint32_t *out = (uint32_t *)data_out;
  for (int32_t i = start; i < stop; i++) {
    out[i] = in[i] ^ 0x80808080u;
  }
  if (core_id == 0) {
    for (int32_t i = words << 2; i < size; i++) {
      data_out[i] = (uint8_t)data_in[i] ^ 0x80;
    }
  }
}
