# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _QuantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _QuantTemplate("""
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
PULP_Quant_f32_s8(${data_in}, ${size}, ${scale}, ${zero_point}, ${min_val}, ${max_val}, ${data_out});
""")
