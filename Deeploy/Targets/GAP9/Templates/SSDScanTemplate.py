# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class GAP9SSDScanTemplate(NodeTemplate):

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
        h_state_size = batchSize * nHeads * headDim * N * 4
        h_state_name = operatorRepresentation['nodeName'] + "_h_state"
        gate_lut_l1_name = operatorRepresentation['nodeName'] + "_gate_lut_l1"
        return [(h_state_name, h_state_size), (gate_lut_l1_name, 256 * 4)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = GAP9SSDScanTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)
        h_state_name, h_state_size = buffers[0]
        gate_lut_l1_name, gate_lut_l1_size = buffers[1]

        ctxt.hoistTransientBuffer(h_state_name, h_state_size)
        operatorRepresentation['h_state'] = h_state_name

        ctxt.hoistTransientBuffer(gate_lut_l1_name, gate_lut_l1_size)
        operatorRepresentation['gate_lut_l1'] = gate_lut_l1_name

        return ctxt, operatorRepresentation, [h_state_name, gate_lut_l1_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Untiled default; overwritten per-tile by SSDScanTileConstraint.serializeTilingSolution.
        operatorRepresentation['init_state'] = 1
        return ctxt, operatorRepresentation, []


referenceTemplate = GAP9SSDScanTemplate("""
// SSDScan (Name: ${nodeName}, Op: ${nodeOp})
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
GAP9_SSDScan_i8_i8(
    (const int8_t *) ${x},
    (const int8_t *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int32_t *) ${A},
    (const int32_t *) ${D_skip},
    (int8_t*) ${y},
    (int32_t *) ${h_state},
    (const int32_t *) ${gate_lut_l1},
    ${batch_size},
    ${chunk_size},
    ${d_state},
    ${head_dim},
    ${n_groups},
    ${n_heads},
    ${seq_len},
    (int32_t) ${output_requant_mul_q40},
    (uint32_t) ${init_state}
);
""")
