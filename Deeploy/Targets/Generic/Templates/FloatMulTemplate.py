# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Mul; scalar B when sizeB == 1, else elementwise (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
% if sizeB == 1:
    for (uint32_t i=0;i<${size};i++){
        ${C}[i] = ${A}[i] * ${B}[0];
    }
% else:
    for (uint32_t i=0;i<${size};i++){
        ${C}[i] = ${A}[i] * ${B}[i];
    }
% endif
END_SINGLE_CORE
""")
