# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// M3Gates (Name: ${nodeName}, Op: ${nodeOp}, which ${which})
GAP9_M3Gates((const int16_t *) ${dt}, (const int8_t *) ${raw}, (const int16_t *) ${lut}, (int16_t *) ${out}, ${B}, ${L}, ${H}, ${which});
""")
