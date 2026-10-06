# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import NetworkContext, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy


class AnnotateDefaultMemoryLevel(SequentialPass):

    def __init__(self, memoryHierarchy: MemoryHierarchy):
        super().__init__()
        self.memoryHierarchy = memoryHierarchy

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _buffer in {**ctxt.localObjects, **ctxt.globalObjects}.values():
            if not hasattr(_buffer, "_memoryLevel"):
                _buffer._memoryLevel = self.memoryHierarchy.getDefaultMemoryLevel().name
        return ctxt, graph


class AnnotateIOMemoryLevel(SequentialPass):

    def __init__(self, ioLevel: str):
        super().__init__()
        self.ioLevel = ioLevel

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        buffers = []

        def globalBuffers(tensors: List[gs.Tensor]) -> List[VariableBuffer]:
            return [ctxt.globalObjects[tensor.name] for tensor in tensors if tensor.name in ctxt.globalObjects.keys()]

        inputBuffers = globalBuffers(graph.inputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer) and len(_buffer._users) > 0,
                          inputBuffers)

        outputBuffers = globalBuffers(graph.outputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer), outputBuffers)

        for _buffer in buffers:
            _buffer._memoryLevel = self.ioLevel

        return ctxt, graph

class AnnotateActivationMemoryLevel(SequentialPass):
    """Put intermediate activations (non-constant, non-IO buffers) of at most maxBytes into `level`.

    Used with defaultMemLevel=L3: weights stay in L3 and are streamed by the L3 tiler, while activations
    that fit are kept resident in L2, so the producer/consumer pair no longer round-trips through L3.
    Must run before AnnotateDefaultMemoryLevel (which only fills buffers without a level).
    """

    def __init__(self, level: str, maxBytes: int, nameRegex: str = ""):
        super().__init__()
        self.level = level
        self.maxBytes = maxBytes
        self.nameRegex = re.compile(nameRegex) if nameRegex else None

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        ioNames = {t.name for t in graph.inputs} | {t.name for t in graph.outputs}
        for _buffer in ctxt.localObjects.values():
            if not isinstance(_buffer, ctxt.VariableBuffer) or isinstance(_buffer, (ctxt.ConstantBuffer,
                                                                                   ctxt.TransientBuffer)):
                continue
            if _buffer.name in ioNames or hasattr(_buffer, "_memoryLevel"):
                continue
            if self.nameRegex is not None and not self.nameRegex.search(_buffer.name):
                continue
            if _buffer.sizeInBytes <= self.maxBytes:
                _buffer._memoryLevel = self.level
        return ctxt, graph


class AnnotateConstantMemoryLevel(SequentialPass):
    """Put constant buffers (weights, requant tables) of at least minBytes into `level`.

    Used with defaultMemLevel=L2: activations stay in L2 (MiniMalloc only requires the *local* buffers to
    share the default level), while the large weights live in L3 and are streamed by the tiler.
    """

    def __init__(self, level: str, minBytes: int):
        super().__init__()
        self.level = level
        self.minBytes = minBytes

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _buffer in ctxt.globalObjects.values():
            if isinstance(_buffer, ctxt.ConstantBuffer) and not hasattr(_buffer, "_memoryLevel") \
                    and _buffer.sizeInBytes >= self.minBytes:
                _buffer._memoryLevel = self.level
        return ctxt, graph
