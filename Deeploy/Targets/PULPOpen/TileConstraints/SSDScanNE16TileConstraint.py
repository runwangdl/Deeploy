# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme

from .SSDScanTileConstraint import SSDScanTileConstraint


class SSDScanNE16TileConstraint(SSDScanTileConstraint):
    """SSD_Scan_NE16 tiling: L in whole chunks (one chunk per tile preferred), heads in tiles of at most
    HEAD_TILE_MAX. The per-tile state buffer is sized for the head tile, so the tile order is rewritten to
    heads-outer / L-inner: every head tile runs all its L tiles back to back and its state never leaves L1."""

    HEAD_TILE_MAX = 8

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xBuffer = ctxt.lookup(name = parseDict["x"])
        BBuffer = ctxt.lookup(name = parseDict["B"])
        ABuffer = ctxt.lookup(name = parseDict["A"])

        dStateVar = tilerModel.getTensorDimVar(tensorName = BBuffer.name, dimIdx = len(BBuffer.shape) - 1)
        tilerModel.addConstraint(dStateVar == parseDict['d_state'])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 0) == xBuffer.shape[0])

        nHeads = ABuffer.shape[0]
        aHeadVar = tilerModel.getTensorDimVar(tensorName = ABuffer.name, dimIdx = 0)
        headTile = min(nHeads, SSDScanNE16TileConstraint.HEAD_TILE_MAX)
        tilerModel.addConstraint(aHeadVar <= headTile)
        tilerModel.addConstraint(aHeadVar >= headTile, strategy = PerformanceHint(priority = 2))

        # one chunk per tile (the NE16 jobs are per chunk anyway; smaller x/z/y tiles leave L1 to the state)
        xLVar = tilerModel.getTensorDimVar(tensorName = xBuffer.name, dimIdx = 1)
        chunk = parseDict['chunk_size']
        if xBuffer.shape[1] > chunk:
            tilerModel.addConstraint(xLVar <= chunk, strategy = PerformanceHint(priority = 1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # heads-outer / L-inner: sort by (d_inner offset, batch offset, L offset)
        order = sorted(range(len(absoluteOutputCubes)),
                       key = lambda i: (absoluteOutputCubes[i].rectangle.offset[2], absoluteOutputCubes[i].rectangle.
                                        offset[0], absoluteOutputCubes[i].rectangle.offset[1]))
        reordered = [absoluteOutputCubes[i] for i in order]
        return super(SSDScanNE16TileConstraint, cls).serializeTilingSolution(tilingSolution, reordered, targetMemLevel,
                                                                               ctxt, operatorRepresentation)
