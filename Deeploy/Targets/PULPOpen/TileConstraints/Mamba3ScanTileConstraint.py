# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class Mamba3ScanTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xBuffer = ctxt.lookup(name = parseDict["x"])
        zBuffer = ctxt.lookup(name = parseDict["z"])
        dtBuffer = ctxt.lookup(name = parseDict["dt"])
        BBuffer = ctxt.lookup(name = parseDict["B"])
        CBuffer = ctxt.lookup(name = parseDict["C"])
        ABuffer = ctxt.lookup(name = parseDict["A"])
        DSkipBuffer = ctxt.lookup(name = parseDict["D_skip"])
        gammaBuffer = ctxt.lookup(name = parseDict["gamma"])
        wBuffer = ctxt.lookup(name = parseDict["w"])
        thetaBuffer = ctxt.lookup(name = parseDict["theta"])
        YBuffer = ctxt.lookup(name = parseDict["y"])

        for bufferName in [
                xBuffer.name, zBuffer.name, dtBuffer.name, BBuffer.name, CBuffer.name, ABuffer.name, DSkipBuffer.name,
                gammaBuffer.name, wBuffer.name, thetaBuffer.name, YBuffer.name
        ]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        headDim = parseDict['head_dim']
        rank = parseDict.get('mimo_rank', 1)

        xBatchVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 0)
        xLVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 1)
        xDInnerVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 2)

        # y and z share x's shape entirely
        for dim in range(len(xBuffer.shape)):
            xDimVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = dim)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = YBuffer.name, dimIdx = dim) == xDimVar)
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = zBuffer.name, dimIdx = dim) == xDimVar)

        # dt, gamma, w, theta, B, C share batch (dim 0) and L (dim 1) with x
        for buffer in [dtBuffer, gammaBuffer, wBuffer, thetaBuffer, BBuffer, CBuffer]:
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buffer.name, dimIdx = 0) == xBatchVar)
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buffer.name, dimIdx = 1) == xLVar)

        # B and C must have the same d_state tile size
        bStateVar = tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = len(BBuffer.shape) - 1)
        cStateVar = tilerModel.getTensorDimVar(tensorName = CBuffer.name, dimIdx = len(CBuffer.shape) - 1)
        tilerModel.addConstraint(bStateVar == cStateVar)

        # Head tiling: dt (dim 2), A and D_skip (dim 0) share the head count; d_inner = heads * head_dim
        aHeadVar = tilerModel.getTensorDimVar(tensorName = ABuffer.name, dimIdx = 0)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = DSkipBuffer.name, dimIdx = 0) == aHeadVar)
        for buffer in [dtBuffer, gammaBuffer, wBuffer, thetaBuffer]:
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = buffer.name, dimIdx = 2) == aHeadVar)
        tilerModel.addConstraint(xDInnerVar == aHeadVar * (headDim * rank))

        # L tiled in whole chunks only; without chunk_size the boundary is unknown, so L stays untiled.
        if 'chunk_size' in parseDict:
            if 'seq_len' not in parseDict:
                parseDict['seq_len'] = xBuffer.shape[1]
            tilerModel.addTileSizeDivisibleConstraint(parseDict, 'seq_len', xLVar, parseDict['chunk_size'])
        else:
            tilerModel.addConstraint(xLVar == xBuffer.shape[1])

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xBuffer = ctxt.lookup(name = parseDict["x"])
        BBuffer = ctxt.lookup(name = parseDict["B"])
        ABuffer = ctxt.lookup(name = parseDict["A"])

        # d_state (times the rank, which rides along in B/C's last dim) is never tiled.
        dStateVar = tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = len(BBuffer.shape) - 1)
        tilerModel.addConstraint(dStateVar == parseDict['d_state'] * parseDict.get('mimo_rank', 1))

        # batch is not tiled.
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 0) == xBuffer.shape[0])

        # Pin HEAD_TILE(=8) heads/tile (one head per core); fewer heads -> soft hint.
        HEAD_TILE = 8
        aHeadVar = tilerModel.getTensorDimVar(tensorName = ABuffer.name, dimIdx = 0)
        if ABuffer.shape[0] >= HEAD_TILE:
            tilerModel.addConstraint(aHeadVar == HEAD_TILE)
        else:
            tilerModel.addConstraint(aHeadVar <= ABuffer.shape[0] - 1, strategy = PerformanceHint(priority = 2))

        xLVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 1)
        tilerModel.addConstraint(xLVar <= xBuffer.shape[1] - 1, strategy = PerformanceHint(priority = 1))

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        xBuffer = ctxt.lookup(name = parseDict['x'])
        BBuffer = ctxt.lookup(name = parseDict['B'])
        ABuffer = ctxt.lookup(name = parseDict['A'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['batch_size'] = tilerModel.getTensorDimVar(xBuffer.name, 0)
        symbolicParseDict['seq_len'] = tilerModel.getTensorDimVar(xBuffer.name, 1)
        # B's last dim is N*R; the kernel wants N and R is a constant, so d_state stays as parsed.
        symbolicParseDict['n_heads'] = tilerModel.getTensorDimVar(ABuffer.name, 0)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        # init_state=1 when the tile's absolute L offset is 0 (start of sequence).
        absoluteLOffsets = [cube.absoluteOffset[1] for cube in absoluteOutputCubes]

        addrNames = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip', 'gamma', 'w', 'theta', 'y']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        headDim = operatorRepresentation['head_dim']
        rank = operatorRepresentation.get('mimo_rank', 1)
        NRSize = ctxt.lookup(operatorRepresentation['B']).shape[-1]      # N * R
        NSize = NRSize // rank

        replacements = {"batch_size": [], "seq_len": [], "d_state": [], "n_heads": [], "init_state": []}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube, absLOffset in zip(outputCubes, absoluteLOffsets):
            BatchOffset, LOffset, DInnerOffset = cube.offset
            BatchSize, LSize, DInnerSize = cube.dims

            HeadOffset = DInnerOffset // (headDim * rank)
            HeadSize = DInnerSize // (headDim * rank)

            replacements["batch_size"].append(BatchSize)
            replacements["seq_len"].append(LSize)
            replacements["d_state"].append(NSize)
            replacements["n_heads"].append(HeadSize)
            replacements["init_state"].append(1 if absLOffset == 0 else 0)

            xCube = HyperRectangle((BatchOffset, LOffset, DInnerOffset), (BatchSize, LSize, DInnerSize))
            zCube = HyperRectangle((BatchOffset, LOffset, DInnerOffset), (BatchSize, LSize, DInnerSize))
            dtCube = HyperRectangle((BatchOffset, LOffset, HeadOffset), (BatchSize, LSize, HeadSize))
            # B and C are per group (n_groups == 1): full on d_state, sliced on L like x.
            BCube = HyperRectangle((BatchOffset, LOffset, 0), (BatchSize, LSize, NRSize))
            CCube = HyperRectangle((BatchOffset, LOffset, 0), (BatchSize, LSize, NRSize))
            ACube = HyperRectangle((HeadOffset,), (HeadSize,))
            DSkipCube = HyperRectangle((HeadOffset,), (HeadSize,))
            gammaCube = HyperRectangle((BatchOffset, LOffset, HeadOffset), (BatchSize, LSize, HeadSize))
            wCube = HyperRectangle((BatchOffset, LOffset, HeadOffset), (BatchSize, LSize, HeadSize))
            thetaCube = HyperRectangle((BatchOffset, LOffset, HeadOffset), (BatchSize, LSize, HeadSize))

            inputLoadSchedule.append({
                "x": xCube, "z": zCube, "dt": dtCube, "B": BCube, "C": CCube, "A": ACube, "D_skip": DSkipCube,
                "gamma": gammaCube, "w": wCube, "theta": thetaCube
            })
            outputLoadSchedule.append({"y": cube})

        replacementTypes = {
            "batch_size": PointerClass(uint8_t),
            "seq_len": PointerClass(uint16_t),
            "d_state": PointerClass(uint16_t),
            "n_heads": PointerClass(uint16_t),
            "init_state": PointerClass(uint8_t),
        }

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
