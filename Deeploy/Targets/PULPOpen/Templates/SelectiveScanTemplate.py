# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULPSelectiveScanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        
        N = operatorRepresentation['d_state']
        h_buffer_size = (operatorRepresentation['batch_size'] * operatorRepresentation['d_inner'] * N * 4)
        h_buffer_name = operatorRepresentation['nodeName'] + "_h_buffer"
        gate_lut_l1_name = operatorRepresentation['nodeName'] + "_gate_lut_l1"
        return [(h_buffer_name, h_buffer_size), (gate_lut_l1_name, 256 * 4)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = PULPSelectiveScanTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)
        h_buffer_name, h_buffer_dim = buffers[0]
        gate_lut_l1_name, gate_lut_l1_size = buffers[1]

        ctxt.hoistTransientBuffer(h_buffer_name, h_buffer_dim)
        operatorRepresentation['h_buffer'] = h_buffer_name

        ctxt.hoistTransientBuffer(gate_lut_l1_name, gate_lut_l1_size)
        operatorRepresentation['gate_lut_l1'] = gate_lut_l1_name

        return ctxt, operatorRepresentation, [h_buffer_name, gate_lut_l1_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Untiled default; overwritten per-tile by SelectiveScanTileConstraint.serializeTilingSolution.
        operatorRepresentation['is_first_L_tile'] = 1
        return ctxt, operatorRepresentation, []


referenceTemplate = PULPSelectiveScanTemplate("""
// SelectiveScan (Name: ${nodeName}, Op: ${nodeOp})
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
PULP_SelectiveScan_i8_i8(
    (const int8_t  *) ${x},
    (const int8_t  *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int32_t *) ${A},
    (const int32_t *) ${D_skip},
    (int8_t        *) ${y},
    (int32_t       *) ${h_buffer},
    (const int32_t *) ${gate_lut_l1},
    ${batch_size},
    ${seq_len},
    ${d_inner},
    ${d_state},
    (int32_t) ${output_requant_mul_q40},
    (uint32_t) ${is_first_L_tile}
);
""")
