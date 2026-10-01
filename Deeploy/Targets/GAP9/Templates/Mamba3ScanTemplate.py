# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class GAP9Mamba3ScanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        headDim = operatorRepresentation['head_dim']
        N = operatorRepresentation['d_state']
        nHeads = operatorRepresentation['n_heads']
        batchSize = operatorRepresentation['batch_size']
        name = operatorRepresentation['nodeName']
        # The state is rank-free: [B, H, P, N] regardless of mimo_rank (Eq. 13 sums the rank in).
        h_state_size = batchSize * nHeads * headDim * N * 4
        # One accumulated RoPE angle per (batch, head), carried across L-tiles like h_state.
        theta_state_size = batchSize * nHeads * 4
        return [(name + "_h_state", h_state_size), (name + "_theta_state", theta_state_size),
                (name + "_gate_lut_l1", 256 * 4)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = GAP9Mamba3ScanTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)
        names = []
        for key, (bufName, size) in zip(('h_state', 'theta_state', 'gate_lut_l1'), buffers):
            ctxt.hoistTransientBuffer(bufName, size)
            operatorRepresentation[key] = bufName
            names.append(bufName)
        return ctxt, operatorRepresentation, names

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['init_state'] = 1
        return ctxt, operatorRepresentation, []


referenceTemplate = GAP9Mamba3ScanTemplate("""
// Mamba3Scan (Name: ${nodeName}, Op: ${nodeOp})
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
GAP9_Mamba3Scan_i8_i8(
    (const int8_t *) ${x},
    (const int8_t *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int32_t *) ${A},
    (const int32_t *) ${D_skip},
    (const int16_t *) ${gamma},
    (const int16_t *) ${w},
    (const int16_t *) ${theta},
    (int8_t*) ${y},
    (int32_t *) ${h_state},
    (int32_t *) ${theta_state},
    (const int32_t *) ${gate_lut_l1},
    ${batch_size},
    ${chunk_size},
    ${d_state},
    ${head_dim},
    ${n_groups},
    ${n_heads},
    ${seq_len},
    ${mimo_rank},
    (int32_t) ${output_requant_mul_q40},
    (uint32_t) ${init_state}
);
""")
