# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BlockingDmaFromAsyncDmaAdapter, DmaDirection, \
    Future, PerTensorWaitingStrategy


# In-flight pi_cl_ram_copy requests per tensor/direction; a transfer loop longer than this
# flushes before continuing. Measured on gvsoc (XProj128_i32, DB): 1 slot 231.7k cycles, 2 slots
# 186.8k, 3 slots 175.9k, 4+ slots hang in the hyperbus path. The C macro of the same name in
# TargetLibraries/GAP9/inc/dory_mem.h must match.
GAP9_L3_REQ_SLOTS = 3


class GAP9L3DmaFuture(Future):

    # One request slot per in-flight copy: a tensor whose tile is a 3-D rectangle is
    # expanded into shape[0] separate pi_cl_ram_copy_2d calls before the single wait
    # (NE16 int32 outputs: up to 15 per tile). Sharing one pi_cl_ram_req_t between
    # them corrupts the PMSIS request list and silently drops copies (FEMBA NE16 DB).
    _initTemplate = NodeTemplate("static pi_cl_ram_req_t ${name}[GAP9_L3_REQ_SLOTS]; uint32_t ${name}_n = 0;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate("""
    for (uint32_t _r = 0; _r < ${name}_n; _r++) pi_cl_ram_copy_wait(&${name}[_r]);
    ${name}_n = 0;""")


class GAP9L3Dma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate(
                """if (${future}_n == GAP9_L3_REQ_SLOTS) { for (uint32_t _r = 0; _r < ${future}_n; _r++) pi_cl_ram_copy_wait(&${future}[_r]); ${future}_n = 0; }
${future}_n += gap9_ram_copy_2d((uint32_t)${ext}, (void *)${loc}, (uint32_t)${transfer_size}, (uint32_t)${stride}, (uint32_t)${length}, ${ext2loc}, &${future}[${future}_n]);"""
            )
    }
    _waitingStrategy = PerTensorWaitingStrategy(GAP9L3DmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "GAP9 RAM API requires contiguous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"GAP9 RAM API requires contiguous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)
        operatorRepresentation.update({
            "ext2loc": 1 if direction == "ExternalToLocal" else 0,
            "transfer_size": math.prod(shape),
            "length": shape[1],
            "stride": strideExt[0],
        })
        return operatorRepresentation


# gap9_ram_copy_2d (TargetLibraries/GAP9/src/dory_mem.c) wraps pi_cl_ram_copy_2d: it splits the rare
# rows that would hit the EVK PSRAM page-wrap bug and returns 0 (copied synchronously) or 1 (in flight).

# Blocking adapter for L3 DMA. Used as the single-buffer path in PULPL3Tiling:
# SB blocks on each transfer, DB uses the async GAP9L3Dma above so the L3<->L2
# prefetch overlaps the previous tile's compute. Filippo Cordella measured the
# DMA overlap as the whole of the 127 M-cycle Siracusa/GAP9 gap on full FEMBA.
gap9L3DmaHack = BlockingDmaFromAsyncDmaAdapter(GAP9L3Dma())
