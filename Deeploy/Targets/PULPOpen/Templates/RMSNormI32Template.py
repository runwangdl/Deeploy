# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// RMSNormI32 (Name: ${nodeName}, Op: ${nodeOp})
PULP_RMSNormI32_s32_s8((const int32_t *)${data_in}, (const int32_t *)${weight}, (int8_t *)${data_out}, ${inputSize}, ${lastDimLength}, ${out_shift});
""")
