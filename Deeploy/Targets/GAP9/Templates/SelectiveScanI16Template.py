# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class GAP9SelectiveScanI16Template(NodeTemplate):
    """SelectiveScan with an int16 state: h_buffer is int16 [batch, d_inner, d_state] (persists across
    L tiles), BC16 is a per-tile scratch [2, L, d_state] int16 (B/C narrowed on the cluster), plus the
    gate LUT copy in L1. The exp LUT (SelectiveScan_exp_lut_qwide) already lives in L1."""

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        N = operatorRepresentation['d_state']
        h_buffer_size = operatorRepresentation['batch_size'] * operatorRepresentation['d_inner'] * N * 2
        bc16_size = 2 * operatorRepresentation['seq_len'] * N * 2
        name = operatorRepresentation['nodeName']
        return [(name + "_h_buffer", h_buffer_size), (name + "_gate_lut_l1", 256 * 4), (name + "_bc16", bc16_size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = GAP9SelectiveScanI16Template.computeTransientBuffersSize(ctxt, operatorRepresentation)
        names = []
        for (name, size), key in zip(buffers, ('h_buffer', 'gate_lut_l1', 'bc16')):
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation[key] = name
            names.append(name)
        return ctxt, operatorRepresentation, names

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['is_first_L_tile'] = 1
        return ctxt, operatorRepresentation, []


referenceTemplate = GAP9SelectiveScanI16Template("""
// SelectiveScanI16 (Name: ${nodeName}, Op: ${nodeOp}) -- int16 state, per-channel shifts
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
GAP9_SelectiveScanI16_i8_i8(
    (const int8_t  *) ${x},
    (const int8_t  *) ${z},
    (const int16_t *) ${dt},
    (const int32_t *) ${B},
    (const int32_t *) ${C},
    (const int16_t *) ${A},
    (const int32_t *) ${D_skip},
    (const int8_t  *) ${shA},
    (const uint8_t *) ${sH},
    (const uint8_t *) ${ysh},
    (int8_t        *) ${y},
    (int16_t       *) ${h_buffer},
    (int16_t       *) ${bc16},
    (const int32_t *) ${gate_lut_l1},
    (const int16_t *) SelectiveScan_exp_lut_ptr,
    ${seq_len},
    ${d_inner},
    ${d_state},
    ${bc_shift},
    (int32_t) ${output_requant_mul_q40},
    (uint32_t) ${is_first_L_tile}
);
""")
