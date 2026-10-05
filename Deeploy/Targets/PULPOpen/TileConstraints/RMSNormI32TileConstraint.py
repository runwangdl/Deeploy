# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Generic.TileConstraints.iRMSNormTileConstraint import iRMSNormTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RMSNormI32TileConstraint(iRMSNormTileConstraint):
    """rows tiled, last dimension whole (same geometry as iRMSNorm); tile size as uint32 (int32 rows of 1408)"""

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                                  ['data_in', 'weight', 'data_out'])
        replacements = {"inputSize": [int(np.prod(c.dims)) for c in outputCubes]}
        inputLoadSchedule = [{"data_in": c, "weight": HyperRectangle((c.offset[-1],), (c.dims[-1],))} for c in outputCubes]
        outputLoadSchedule = [{"data_out": c} for c in outputCubes]
        return VariableReplacementScheme(replacements, {"inputSize": PointerClass(uint32_t)}), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
