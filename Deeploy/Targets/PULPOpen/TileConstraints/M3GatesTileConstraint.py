# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class M3GatesTileConstraint(TileConstraint):
    """untiled: w at token t reads token t+1 (the tensors are a few hundred bytes)"""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for k in ('dt', 'raw', 'lut', 'out'):
            buf = ctxt.lookup(parseDict[k])
            tilerModel.addTensorDimToModel(ctxt, buf.name)
            for d, sz in enumerate(buf.shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = d) == sz)
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        inBase, outBase = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation, ['dt', 'raw', 'lut', 'out'])
        lut = ctxt.lookup(operatorRepresentation['lut'])
        ins, outs = [], []
        for c in absoluteOutputCubes:
            cube = c.rectangle
            ins.append({"dt": cube, "raw": cube, "lut": HyperRectangle((0,), (lut.shape[0],))})
            outs.append({"out": cube})
        return VariableReplacementScheme({}, {}), TilingSchedule(inBase, outBase, ins, outs)
