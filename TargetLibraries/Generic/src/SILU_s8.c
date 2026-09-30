/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

#include "kernel/SILULUT.h"

void SILU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize,
                 int32_t input_offset) {
  for (int i = 0; i < dataSize; i++) {
    int32_t x = data_in[i] + 128 - input_offset;
    data_out[i] = SILU_lut_s8_s32[x];
  }
}

void SILU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                int32_t input_offset, int8_t *lut) {
  for (int i = 0; i < dataSize; i++) {
    int32_t x = data_in[i] + 128 - input_offset;
    data_out[i] = lut[x];
  }
}
