# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _DequantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _DequantTemplate("""
// Dequantization (Name: ${nodeName}, Op: ${nodeOp})
% if data_in_type.referencedType.typeWidth == 32:
PULP_Dequant_s32_f32(${data_in}, ${size}, ${scale}, ${zero_point}, ${data_out});
% else:
PULP_Dequant_s8_f32(${data_in}, ${size}, ${scale}, ${zero_point}, ${data_out});
% endif
""")
