# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float RMSNorm (Name: ${nodeName}, Op: ${nodeOp})
PULP_RMSNorm_fp32_fp32(${data_in}, ${weight}, ${data_out}, ${inputSize}, ${lastDimLength}, ${eps});
""")
