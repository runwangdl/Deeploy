/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __DEEPLOY_MATH_GAP9_M3GATES_HEADER_
#define __DEEPLOY_MATH_GAP9_M3GATES_HEADER_
#include <stdint.h>
void GAP9_M3Gates(const int16_t *dt, const int8_t *raw, const int16_t *lut, int16_t *out, uint32_t B, uint32_t L,
                  uint32_t H, uint32_t which);
#endif
