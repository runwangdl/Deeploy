# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class StaticScanNE16TileConstraint(TileConstraint):
    """whole heads per tile (the per-head constant weights go with them), the window (L) and batch untiled"""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        names = [parseDict[k] for k in ('x', 'z', 'wenc', 'comp', 'M', 'Dq', 'y')]
        for n in names:
            tilerModel.addTensorDimToModel(ctxt, n)
        x, z, w, comp, M, Dq, y = [ctxt.lookup(n) for n in names]
        P = parseDict['head_dim']
        for d in range(3):
            xv = tilerModel.getTensorDimVar(tensorName = x.name, dimIdx = d)
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = z.name, dimIdx = d) == xv)
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = y.name, dimIdx = d) == xv)
        hv = tilerModel.getTensorDimVar(tensorName = w.name, dimIdx = 0)
        for buf in (comp, M, Dq):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = 0) == hv)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = x.name, dimIdx = 2) == hv * P)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = x.name, dimIdx = 0) == x.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = x.name, dimIdx = 1) == x.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = w.name, dimIdx = 1) == w.shape[1])
        for buf in (comp, M):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = 1) == buf.shape[1])
        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        w = ctxt.lookup(parseDict['wenc'])
        hv = tilerModel.getTensorDimVar(tensorName = w.name, dimIdx = 0)
        tilerModel.addConstraint(hv >= w.shape[0], strategy = PerformanceHint(priority = 1))
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> Dict:
        rep = parseDict.copy()
        rep['n_heads'] = tilerModel.getTensorDimVar(ctxt.lookup(parseDict['wenc']).name, 0)
        return rep

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [c.rectangle for c in absoluteOutputCubes]
        inBase, outBase = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                              ['x', 'z', 'wenc', 'comp', 'M', 'Dq', 'y'])
        P, L = operatorRepresentation['head_dim'], operatorRepresentation['seq_len']
        rep = {"n_heads": []}
        ins, outs = [], []
        for cube in outputCubes:
            (bo, lo, do), (bs, ls, ds) = cube.offset, cube.dims
            h0, nh = do // P, ds // P
            rep["n_heads"].append(nh)
            ins.append({
                "x": cube, "z": cube,
                "wenc": HyperRectangle((h0, 0), (nh, L * L)),
                "comp": HyperRectangle((h0, 0), (nh, L)),
                "M": HyperRectangle((h0, 0), (nh, L)),
                "Dq": HyperRectangle((h0,), (nh,)),
            })
            outs.append({"y": cube})
        return VariableReplacementScheme(rep, {"n_heads": PointerClass(uint16_t)}), \
            TilingSchedule(inBase, outBase, ins, outs)
