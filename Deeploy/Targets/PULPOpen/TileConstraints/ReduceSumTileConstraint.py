# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class ReduceSumTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        # The kernel accumulates over the whole reduction axes, and serializeTilingSolution below
        # hands it one cube spanning the complete tensors. Pinning every dimension to its full
        # extent keeps the buffers the tiler sizes consistent with those cubes, while still
        # allowing the L3 -> L2 -> L1 transfer chain that UntiledTileConstraint cannot express.
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

            for idx, shapeDim in enumerate(ctxt.lookup(bufferName).shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = idx) == shapeDim)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # No constraints - let the tiler handle dimensions normally
        # We'll handle the actual ReduceSum logic in serializeTilingSolution
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)

        symbolicParseDict = parseDict.copy()

        # Since we force all dimensions to be full size, we can use the actual shape
        # This ensures the template gets the correct dimensions for the single cube
        symbolicParseDict['data_in_shape'] = list(inputBuffer.shape)

        # Add axes information (normalized)
        if 'axis' in parseDict:
            axis = parseDict['axis']
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)

            # Handle negative axis indexing
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    ax = len(inputBuffer.shape) + ax
                normalized_axes.append(ax)

            symbolicParseDict['axes'] = normalized_axes
        else:
            # Global reduction - all axes
            symbolicParseDict['axes'] = list(range(len(inputBuffer.shape)))

        # Add keepdims information
        symbolicParseDict['keepdims'] = parseDict.get('keepdims', True)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Get original tensor shapes from context
        inputBufferName = operatorRepresentation['data_in']
        outputBufferName = operatorRepresentation['data_out']
        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        # Use original dimensions for ReduceSum computation
        originalInputShape = list(inputBuffer.shape)
        originalOutputShape = list(outputBuffer.shape)

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"data_in_shape": [], "axes": [], "keepdims": [], "reduceLength": []}
        replacementTypes = {
            "data_in_shape": PointerClass(uint32_t),
            "axes": PointerClass(uint32_t),
            "keepdims": PointerClass(uint32_t),
            "reduceLength": PointerClass(uint32_t)
        }

        # Get axis and keepdims information from operator representation
        # Note: the key might be 'axes' (plural) instead of 'axis' (singular)
        axis = operatorRepresentation.get('axis', operatorRepresentation.get('axes', None))
        keepdims = operatorRepresentation.get('keepdims', True)

        # Calculate axes (normalize negative indices)
        if axis is not None:
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)

            # Handle negative axis indexing
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    ax = len(originalInputShape) + ax
                normalized_axes.append(ax)
            axes = normalized_axes
        else:
            # Global reduction - all axes
            axes = list(range(len(originalInputShape)))

        # Calculate reduceLength (product of dimensions being reduced)
        reduceLength = 1
        for ax in axes:
            reduceLength *= originalInputShape[ax]

        # For ReduceSum, we always use the original tensor dimensions
        # regardless of how the tiler decides to split them
        replacements['data_in_shape'].append(tuple(originalInputShape))
        replacements['axes'].append(tuple(axes))
        replacements['keepdims'].append(1 if keepdims else 0)
        replacements['reduceLength'].append(reduceLength)

        # Create scheduling based on original dimensions
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Create HyperRectangles with original dimensions
        from Deeploy.TilingExtension.TilingCodegen import HyperRectangle

        inputCube = HyperRectangle(dims = originalInputShape, offset = [0] * len(originalInputShape))

        outputCube = HyperRectangle(dims = originalOutputShape, offset = [0] * len(originalOutputShape))

        inputLoadSchedule.append({"data_in": inputCube})
        outputLoadSchedule.append({"data_out": outputCube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
