# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _iLayerNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _iLayerNormTemplate("""
// iLayernorm (Name: ${nodeName}, Op: ${nodeOp})
PULP_Layernorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weight}, ${bias}, ${size}, ${lastDimLength});
""")