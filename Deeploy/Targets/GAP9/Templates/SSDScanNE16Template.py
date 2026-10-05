# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

NE16_SLOTS = 2
META_HDR = 8
NUM_CORES = 8


def pixelGrid(P: int) -> Tuple[int, int]:
    """GH x GW pixel grid for a head of P channels: GH a multiple of 3 (NE16 3x3 spatial subtiles),
    fewest subtiles first, then least padding."""
    best = None
    GH = 3
    while GH <= 3 * ((P + 2) // 3) + 3:
        GW = -(-P // GH)
        key = ((GH // 3) * (-(-GW // 3)), GH * GW - P, GH)
        if best is None or key < best[0]:
            best = (key, GH, GW)
        GH += 3
    return best[1], best[2]


def scratchBytes(Q, N, P, NHt, GH, GW, decay_mode = 0, mamba3 = 0, R = 1):
    """Mirrors the carve-up in GAP9_SSDScanNE16_i8_i8 (NHt may be a solver variable); R = Mamba-3 MIMO rank."""
    PXA = GH * GW
    QR = Q * R
    WSZ = QR * (QR + 2 * N)
    ASZ = PXA * (2 * QR + N)
    MSZ = META_HDR + 2 * QR + N + (3 * Q if decay_mode == 1 else 0)
    QM = Q * max(Q, N)
    rot = NHt * 2 * Q * N * 4 if mamba3 else 0   # Mamba-3 RoPE: rotated B, C rows per head
    rot += NHt * (Q * Q + 2 * Q + 2 * QR + N) * 4 if R > 1 else 0   # MIMO: per-head decay tables + row maxima
    return NHt * (PXA * (QR + N) + 2 * WSZ + 4 * MSZ) + 4 * NE16_SLOTS * ASZ + 4 * QR * QR + 4 * NUM_CORES * (QM + Q) + 512 + rot


class GAP9SSDScanNE16Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        P = operatorRepresentation['head_dim']
        N = operatorRepresentation['d_state']
        Q = operatorRepresentation['chunk_size']
        NHt = operatorRepresentation['n_heads']
        batchSize = operatorRepresentation['batch_size']
        GH, GW = pixelGrid(P)
        name = operatorRepresentation['nodeName']
        # state [B][NHt][P][N] int32 followed by the per-core |state| maxima [B][NHt][NUM_CORES]
        m3 = int(operatorRepresentation.get('mamba3', 0))
        return [(name + "_h_state", batchSize * NHt * (P * N + NUM_CORES + (1 if m3 else 0)) * 4), (name + "_gate_lut_l1", 256 * 4),
                (name + "_ne16_scratch",
                 scratchBytes(Q, N, P, NHt, GH, GW, int(operatorRepresentation.get('decay_mode', 0)), m3,
                              int(operatorRepresentation.get('mimo_rank', 1))))]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = GAP9SSDScanNE16Template.computeTransientBuffersSize(ctxt, operatorRepresentation)
        names = []
        for (name, size), key in zip(buffers, ('h_state', 'gate_lut_l1', 'ne16_scratch')):
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation[key] = name
            names.append(name)
        return ctxt, operatorRepresentation, names

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Untiled default; overwritten per tile by SSDScanNE16TileConstraint.serializeTilingSolution.
        operatorRepresentation['init_state'] = 1
        GH, GW = pixelGrid(operatorRepresentation['head_dim'])
        operatorRepresentation['grid_h'] = GH
        operatorRepresentation['grid_w'] = GW
        return ctxt, operatorRepresentation, []


referenceTemplate = GAP9SSDScanNE16Template("""
// SSDScanNE16 (Name: ${nodeName}, Op: ${nodeOp}) -- per-head chunk products on the NE16
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
GAP9_SSDScanNE16_i8_i8(
    (const int8_t *) ${x},
    (const int8_t *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int32_t *) ${A},
    (const int32_t *) ${D_skip},
    (void *) ${y},
    (int32_t *) ${h_state},
    (const int32_t *) ${gate_lut_l1},
    (uint8_t *) ${ne16_scratch},
    ${batch_size},
    ${chunk_size},
    ${d_state},
    ${head_dim},
    ${n_heads},
    ${seq_len},
    ${grid_h},
    ${grid_w},
    (int32_t) ${output_requant_mul_q40},
    (uint32_t) ${init_state},
    ${epilogue_version},
    ${out_bits},
    ${out_shift},
    (const int16_t *) ${dta},
    (const int8_t *) ${R},
    ${decay_mode},
    ${resid_mul},
    (const int16_t *) ${m3_gamma},
    (const int16_t *) ${m3_w},
    (const int16_t *) ${m3_theta},
    ${mamba3},
    ${mimo_rank}
);
""")
