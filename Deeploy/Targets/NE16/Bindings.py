# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.Targets.GAP9.Bindings import GAP9ClusterTransformer as ClusterTransformer
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE16DenseConv2D_Template, NE16DWConv2D_Template, \
    NE16PWConv2D_Template, NE16RqntDenseConv2D_Template, NE16RqntDWConv2D_Template, NE16RqntPWConv2D_Template
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker

class NE16RQSConvChecker(PULPConvChecker):
    """PULPConvChecker that also matches the binding's output width against the node's
    n_levels: the RQS pointwise bindings come in 8-bit and 32-bit output flavours and the
    first binding whose *inputs* type-check would otherwise always win, typing a 32-bit
    (n_levels 2^32) NE16 output as int8."""

    def checkOutputType(self, inputs, operatorRepresentation) -> bool:
        outType = self.output_types[0].referencedType
        nLevels = int(operatorRepresentation['n_levels'])
        signed = bool(operatorRepresentation['signed'])
        return nLevels <= outType.nLevels and (outType.typeMin < 0) == signed


NE16RQSPWConv2DBindings = [
    NodeBinding(
        NE16RQSConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t, int32_t]
    for weight_type in [uint8_t, int8_t]
]
NE16PWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16PWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NE16RQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NE16DWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16DWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NE16RQSDenseConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NE16RqntDenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NE16DenseConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NE16DenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
