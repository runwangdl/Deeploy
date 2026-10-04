/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Guard must differ from DeeployPULPMath.h's __DEEPLOY_MATH_HEADER_: a shared
// guard would no-op the include below, leaving PULP_* prototypes undeclared.
#ifndef __DEEPLOY_GAP9_MATH_HEADER_
#define __DEEPLOY_GAP9_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Full PULP kernel prototypes; its quoted includes resolve to PULPOpen/inc, so
// PULP_* win over the Generic siblings that -I ordering would otherwise shadow.
#include "DeeployPULPMath.h"

// GAP9 gates single-core sections on core 8 as well as core 0; override PULP's
// core-0-only definition here. #undef first so the redefinition is warning-clean.
#undef BEGIN_SINGLE_CORE
#undef END_SINGLE_CORE
#undef SINGLE_CORE
#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

#include "dory_dma.h"
#include "dory_mem.h"

// GAP9-specific kernel prototypes.
#include "SelectiveScan.h"
#include "SSDScan.h"
#include "SSDScanNE16.h"

#endif // __DEEPLOY_GAP9_MATH_HEADER_
