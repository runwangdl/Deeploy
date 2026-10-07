/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_TANH_KERNEL_HEADER_
#define __DEEPLOY_MATH_TANH_KERNEL_HEADER_

#include "DeeployPULPMath.h"

/* Elementwise Tanh, parallelised over the cluster cores (same chunking as PULP_Relu). */
void PULP_Tanh_fp32_fp32(float32_t *input, float32_t *output, uint32_t size);

#endif // __DEEPLOY_MATH_TANH_KERNEL_HEADER_
