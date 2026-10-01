# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class SelectiveScanTileConstraint(TileConstraint):

    # Relationship between input and output tile shapes.
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xBuffer = ctxt.lookup(name = parseDict["x"])
        zBuffer = ctxt.lookup(name = parseDict["z"])
        dtBuffer = ctxt.lookup(name = parseDict["dt"])
        BBuffer = ctxt.lookup(name = parseDict["B"])
        CBuffer = ctxt.lookup(name = parseDict["C"])
        ABuffer = ctxt.lookup(name = parseDict["A"])
        DSkipBuffer = ctxt.lookup(name = parseDict["D_skip"])
        YBuffer = ctxt.lookup(name = parseDict["y"])

        for bufferName in [xBuffer.name, zBuffer.name, dtBuffer.name, BBuffer.name, CBuffer.name, ABuffer.name, DSkipBuffer.name, YBuffer.name]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # x, z, dt, y all share shape [batch, L, d_inner]: constrain every dim to match x
        for dim in range(len(xBuffer.shape)):
            xDimVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = dim)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = zBuffer.name, dimIdx = dim) == xDimVar)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = dtBuffer.name, dimIdx = dim) == xDimVar)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = YBuffer.name, dimIdx = dim) == xDimVar)

        # B and C have shape [batch, L, d_state]: only batch (dim 0) and L (dim 1) match x
        for dim in range(len(xBuffer.shape) - 1):
            xDimVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = dim)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = dim) == xDimVar)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = CBuffer.name, dimIdx = dim) == xDimVar)

        # B and C must have the same d_state tile size
        bStateDimVar = tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = len(BBuffer.shape) - 1)
        cStateDimVar = tilerModel.getTensorDimVar(tensorName = CBuffer.name, dimIdx = len(CBuffer.shape) - 1)
        tilerModel.addConstraint(bStateDimVar == cStateDimVar)

        #A is tiled along d_inner (dim 0) and pinned along d_state (dim 1): A.dim_0 == x.dim_2, A.dim_1 == B.dim_-1
        xDInnerDimVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 2)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = ABuffer.name, dimIdx = 0) == xDInnerDimVar)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = ABuffer.name, dimIdx = 1) == bStateDimVar)

        # D_skip is tiled along d_inner (dim 0): D_skip.dim_0 == x.dim_2
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = DSkipBuffer.name, dimIdx = 0) == xDInnerDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # d_state must not be tiled; set on B, propagated to C and A geometrically.
        BBuffer = ctxt.lookup(name = parseDict["B"])

        # B, C have shape [batch, L, d_state]: pin d_state at dim -1
        dStateVarB = tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = len(BBuffer.shape) - 1)
        tilerModel.addConstraint(dStateVarB == parseDict['d_state'])

        # 2D (d_inner, L) tiling: solver only maximises L1, so shape it with the two hints below.
        xBuffer = ctxt.lookup(name = parseDict["x"])
        LFull = xBuffer.shape[1]
        DInnerFull = xBuffer.shape[2]
        LVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 1)
        DInnerVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 2)

        # d_inner: no cap; L1 already forces tiling, so let it pick the largest DT that fits.
        tilerModel.addConstraint(DInnerVar <= DInnerFull, strategy = PerformanceHint(priority = 1))

        # L: prefer ~2 L-tiles (LT<=40, measured optimum).
        tilerModel.addConstraint(LVar <= 40, strategy = PerformanceHint(priority = 2))

        return tilerModel

    # Bridges the TileConstraint to transient-buffer sizing (h_buffer = batch * d_inner * d_state).
    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        xBuffer = ctxt.lookup(name = parseDict['x'])
        BBuffer = ctxt.lookup(name = parseDict['B'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['batch_size'] = tilerModel.getTensorDimVar(xBuffer.name, 0)
        symbolicParseDict['seq_len'] = tilerModel.getTensorDimVar(xBuffer.name, 1)
        symbolicParseDict['d_inner'] = tilerModel.getTensorDimVar(xBuffer.name, 2)
        symbolicParseDict['d_state'] = tilerModel.getTensorDimVar(BBuffer.name, len(BBuffer.shape) - 1)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        # Absolute L offsets: needed for is_first_L_tile at inner tiling stages where rectangle.offset is relative (always 0).
        absoluteLOffsets = [cube.absoluteOffset[1] for cube in absoluteOutputCubes]

        addrNames = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip', 'y']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation, addrNames)

        # Only d_state is not tiled.
        NSize = ctxt.lookup(operatorRepresentation['A']).shape[-1]
        NOffset = 0

        inputXCubes = []
        inputresCubes = []
        inputdtCubes = []
        inputBCubes = []
        inputCCubes = []
        inputACubes = []
        inputDSkipCubes = []

        replacements = {"batch_size": [], "seq_len": [], "d_inner": [], "d_state": [], "is_first_L_tile": []}

        for cube, absLOffset in zip(outputCubes, absoluteLOffsets):
            BatchOffset, LOffset, DInnerOffset = cube.offset
            BatchSize, LSize, DInnerSize = cube.dims

            replacements["batch_size"].append(BatchSize)
            replacements["seq_len"].append(LSize)
            replacements["d_inner"].append(DInnerSize)
            replacements["d_state"].append(NSize)
            replacements["is_first_L_tile"].append(1 if absLOffset == 0 else 0)

            XresYTensorOffsets = (BatchOffset, LOffset, DInnerOffset)
            XresYTensorDims = (BatchSize, LSize, DInnerSize)

            XCUbe = HyperRectangle(XresYTensorOffsets, XresYTensorDims)
            inputXCubes.append(XCUbe)

            resCube = HyperRectangle(XresYTensorOffsets, XresYTensorDims)
            inputresCubes.append(resCube)

            dtCube = HyperRectangle(XresYTensorOffsets, XresYTensorDims)
            inputdtCubes.append(dtCube)

            BCTesnorOffsets = (BatchOffset, LOffset, NOffset)
            BCTensorDims = (BatchSize, LSize, NSize)

            BCube = HyperRectangle(BCTesnorOffsets, BCTensorDims)
            inputBCubes.append(BCube)

            CCube = HyperRectangle(BCTesnorOffsets, BCTensorDims)
            inputCCubes.append(CCube)

            # A has shape [d_inner, d_state]: tile d_inner like x.dim_2.
            ACube = HyperRectangle((DInnerOffset, NOffset), (DInnerSize, NSize))
            inputACubes.append(ACube)

            # D_skip has shape [d_inner]: tile d_inner like x.dim_2.
            DSkipCube = HyperRectangle((DInnerOffset,), (DInnerSize,))
            inputDSkipCubes.append(DSkipCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        # TODO: B/C are reloaded per D-tile though they are independent of d_inner.
        for x, z, dt, b, c, a, dskip in zip(inputXCubes, inputresCubes, inputdtCubes, inputBCubes, inputCCubes, inputACubes, inputDSkipCubes):
            inputLoadSchedule.append({"x": x, "z": z, "dt": dt, "B": b, "C": c, "A": a, "D_skip": dskip})

        for out in outputCubes:
            outputLoadSchedule.append({"y": out})

        replacementTypes = {
            "batch_size":     PointerClass(uint8_t),
            "seq_len":        PointerClass(uint16_t),
            "d_inner":        PointerClass(uint16_t),
            "d_state":        PointerClass(uint16_t),
            "is_first_L_tile": PointerClass(uint8_t),
        }

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule


class SelectiveScanI16TileConstraint(SelectiveScanTileConstraint):
    """SelectiveScanI16: same tiling as SelectiveScan plus the per-channel constants shA, sH, ysh ([d_inner],
    tiled like D_skip)."""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = SelectiveScanTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        xBuffer = ctxt.lookup(name = parseDict["x"])
        xDInnerDimVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 2)
        for key in ("shA", "sH", "ysh"):
            buf = ctxt.lookup(name = parseDict[key])
            tilerModel.addTensorDimToModel(ctxt, buf.name)
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buf.name, dimIdx = 0) == xDInnerDimVar)
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        absoluteLOffsets = [cube.absoluteOffset[1] for cube in absoluteOutputCubes]
        addrNames = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip', 'shA', 'sH', 'ysh', 'y']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation, addrNames)
        NSize = ctxt.lookup(operatorRepresentation['A']).shape[-1]
        replacements = {"batch_size": [], "seq_len": [], "d_inner": [], "d_state": [], "is_first_L_tile": []}
        inputLoadSchedule, outputLoadSchedule = [], []
        for cube, absLOffset in zip(outputCubes, absoluteLOffsets):
            BatchOffset, LOffset, DInnerOffset = cube.offset
            BatchSize, LSize, DInnerSize = cube.dims
            replacements["batch_size"].append(BatchSize)
            replacements["seq_len"].append(LSize)
            replacements["d_inner"].append(DInnerSize)
            replacements["d_state"].append(NSize)
            replacements["is_first_L_tile"].append(1 if absLOffset == 0 else 0)
            xyz = HyperRectangle((BatchOffset, LOffset, DInnerOffset), (BatchSize, LSize, DInnerSize))
            bc = HyperRectangle((BatchOffset, LOffset, 0), (BatchSize, LSize, NSize))
            a = HyperRectangle((DInnerOffset, 0), (DInnerSize, NSize))
            d = HyperRectangle((DInnerOffset,), (DInnerSize,))
            inputLoadSchedule.append({"x": xyz, "z": xyz, "dt": xyz, "B": bc, "C": bc, "A": a, "D_skip": d, "shA": d, "sH": d, "ysh": d})
            outputLoadSchedule.append({"y": xyz})
        replacementTypes = {
            "batch_size": PointerClass(uint8_t),
            "seq_len": PointerClass(uint16_t),
            "d_inner": PointerClass(uint16_t),
            "d_state": PointerClass(uint16_t),
            "is_first_L_tile": PointerClass(uint8_t),
        }
        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        return VariableReplacementScheme(replacements, replacementTypes), schedule
