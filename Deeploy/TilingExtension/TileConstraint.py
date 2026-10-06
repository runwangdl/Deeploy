# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, MemoryTransfer, \
    TilingSchedule, VariableReplacementScheme, computeTileHyperRectangles


class TileConstraint():

    # Override this
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        Override this function to add your geometric constraints.
        Each dimension of the output tensors should be determinable through a linear equation that utilizes the dimensions of the input tensors and the attributes of the nodes.
        '''
        return tilerModel

    # Override this
    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        Override this function to add your custom constraints to your node.
        '''
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        return {}

    @staticmethod
    def getBaseAddr(tilingSolution, targetMemLevel, name) -> List[Optional[int]]:
        mc = tilingSolution.tensorMemoryConstraints[name].memoryConstraints[targetMemLevel]

        if mc.addrSpace is None:
            return [None]

        start, end = mc.addrSpace
        bufferSize = (end - start) // mc.multiBufferCoefficient

        return [start + bufferSize * i for i in range(mc.multiBufferCoefficient)]

    @staticmethod
    def extractBaseAddr(tilingSolution: NodeMemoryConstraint, targetMemLevel: str,
                        operatorRepresentation: OperatorRepresentation,
                        addrNames: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:

        varList = list(map(lambda x: operatorRepresentation[x], addrNames))
        addrList = list(map(lambda x: TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, x), varList))

        inputBaseOffsets = {}
        outputBaseOffsets = {}

        for addr, addrName, varName in zip(addrList, addrNames, varList):
            if varName in tilingSolution.outputTensorMemoryConstraints.keys():
                outputBaseOffsets[addrName] = addr
            elif varName in tilingSolution.inputTensorMemoryConstraints.keys():
                inputBaseOffsets[addrName] = addr
            else:
                raise Exception(f"{addrName} not in input or output!")

        return inputBaseOffsets, outputBaseOffsets

    @staticmethod
    def sanitizeTilingSchedule(tilingSchedule: TilingSchedule) -> TilingSchedule:
        # Inputs without a buffer at this level are already resident in the local memory (e.g. an activation
        # annotated to L2 under an L3 tiling loop). Keep their per-tile rectangles so the code generation can
        # still advance their pointer to each tile's position in the full tensor.
        residentInputRects = getattr(tilingSchedule, "residentInputRects", {})
        for baseOffsetName, baseOffsetValue in tilingSchedule.inputBaseOffsets.copy().items():
            if baseOffsetValue == [None]:
                residentInputRects[baseOffsetName] = [step[baseOffsetName] for step in tilingSchedule.inputLoadSchedule]
                for step in tilingSchedule.inputLoadSchedule:
                    del step[baseOffsetName]
                del tilingSchedule.inputBaseOffsets[baseOffsetName]

        for baseOffsetName, baseOffsetValue in tilingSchedule.outputBaseOffsets.copy().items():
            if baseOffsetValue == [None]:
                residentInputRects[baseOffsetName] = [step[baseOffsetName] for step in tilingSchedule.outputLoadSchedule]
                for step in tilingSchedule.outputLoadSchedule:
                    del step[baseOffsetName]
                del tilingSchedule.outputBaseOffsets[baseOffsetName]

        tilingSchedule.residentInputRects = residentInputRects  # inputs and outputs resident in the local memory
        return tilingSchedule

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        def getMemoryTransfer(tensorConstraint: TensorMemoryConstraint, sourceCube: HyperRectangle,
                              sourceMemoryLevel: str, targetMemoryLevel: str) -> MemoryTransfer:

            size = np.prod(sourceCube.dims)
            sourceConstraint = MemoryConstraint(sourceMemoryLevel, size)
            sourceConstraint.shape = sourceCube.dims

            destConstraint = copy.copy(tensorConstraint.memoryConstraints[targetMemoryLevel])

            if any(dim1 > dim2 for dim1, dim2 in zip(destConstraint.shape, sourceConstraint.shape)):
                destConstraint.shape = sourceConstraint.shape

            return MemoryTransfer(sourceConstraint, destConstraint)

        def _offsetAdd(offsetA: Tuple[int, ...], offsetB: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(dimA + dimB for dimA, dimB in zip(offsetA, offsetB))

        def getCubeTransfers(tensorConstraint: TensorMemoryConstraint, sourceCubes: List[AbsoluteHyperRectangle],
                             sourceMemoryLevel: str,
                             targetMemoryLevel: str) -> Tuple[List[AbsoluteHyperRectangle], List[int]]:
            solution = []
            solutionLengths = []

            for sourceCube in sourceCubes:
                memTransfer = getMemoryTransfer(tensorConstraint, sourceCube.rectangle, sourceMemoryLevel,
                                                targetMemoryLevel)
                solutionCubes = computeTileHyperRectangles(memTransfer)
                solutionAbsoluteCubes = [
                    AbsoluteHyperRectangle(rectangle = cube,
                                           absoluteOffset = _offsetAdd(sourceCube.absoluteOffset, cube.offset))
                    for cube in solutionCubes
                ]
                solution += solutionAbsoluteCubes
                solutionLengths.append(len(solutionAbsoluteCubes))

            return solution, solutionLengths

        assert len(tilingSolution.outputTensorMemoryConstraints) == 1, "Expected node to have only one output!"

        outVar, outTensorConstraint = next(iter(tilingSolution.outputTensorMemoryConstraints.items()))
        memoryPath = list(outTensorConstraint.memoryConstraints.keys())

        assert targetMemLevel in memoryPath, \
            f"Target memory level {targetMemLevel} does not exist in the memory path {memoryPath}"

        targetIdx = memoryPath.index(targetMemLevel)

        forcedTargetIdx = False
        if targetIdx == 0:
            # SCHEREMO: Watch out - this happens if inputs are in L(N+1) but outputs only in L(N)
            targetIdx = 1
            forcedTargetIdx = True

        # Output resident in an inner level (path starts below the outermost input level): the outer loop then
        # iterates the innermost output cubes one by one, so the inner loop must take exactly one cube per call.
        inputOuterLevels = {
            next(iter(tc.memoryConstraints.keys())) for tc in tilingSolution.inputTensorMemoryConstraints.values()
        }
        residentOutputInner = (not forcedTargetIdx) and len(memoryPath) >= 2 and \
            any(level != memoryPath[0] for level in inputOuterLevels) and \
            all(level not in memoryPath or level == memoryPath[0] for level in inputOuterLevels)

        fullShape = ctxt.lookup(outVar).shape
        initialOffset = (0,) * len(fullShape)
        outputCubes = [
            AbsoluteHyperRectangle(rectangle = HyperRectangle(offset = initialOffset, dims = tuple(fullShape)),
                                   absoluteOffset = initialOffset)
        ]

        for source, target in zip(memoryPath[:targetIdx], memoryPath[1:targetIdx + 1]):
            outputCubes, solutionLengths = getCubeTransfers(outTensorConstraint, outputCubes, source, target)

        arrayOfCubes = []
        _idx = 0
        for idxLen in solutionLengths:
            arrayOfCubes += [outputCubes[_idx:_idx + idxLen]]
            _idx += idxLen
        if residentOutputInner and os.environ.get("DEEPLOY_MINIMALLOC_MULTILEVEL") == "1":
            arrayOfCubes = [[cube] for cube in outputCubes]

        varReplacements = []
        tilingSchedules = []

        for _outputCubes in arrayOfCubes:

            varReplacement, tilingSchedule = cls.serializeTilingSolution(tilingSolution, _outputCubes, targetMemLevel,
                                                                         ctxt, operatorRepresentation)
            sanitizedTilingSchedule = cls.sanitizeTilingSchedule(tilingSchedule)

            varReplacements.append(varReplacement)
            tilingSchedules.append(sanitizedTilingSchedule)

        flatReplacement = varReplacements[0]
        for replacement in varReplacements[1:]:
            flatReplacement += replacement

        return flatReplacement, tilingSchedules

    @classmethod
    @abstractmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """Compute the required input tiles as a sequence of HyperRectangles

        Parameters
        ----------
        tilingSolution : NodeMemoryConstraint
            The final tiling solution computed in the midend
        absoluteOutputCubes : List[AbsoluteHyperRectangle]
            A list of HyperRectangles that represent tiles of the
            operator's outputs with absolute offsets
        targetMemLevel : str
            The name of the MemoryLevel registered within the
            Platform's MemoryHierarchy where tiles should be
            transferred into (e.g.: L2, L1,... )
        ctxt : NetworkContext
            The current NetworkContext
        operatorRepresentation : Dict
            The operator's node representation dictionary

        Returns
        -------
        Tuple[VariableReplacementScheme, TilingSchedule]
            Return a VariableReplacementScheme to express which
            expressions within the target template might have to be
            replaced due to tiling. Also return a TilingSchedule to
            define one input HyperRectangle tuple for each output tile

        Raises
        ------
        Exception
            Raises an exception unless overridden in the calling class

        """

        raise Exception(f"serializeTilingSolution not implemented for class {cls.__name__}!")
