# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

PXC = 27  # pixels per NE16 job (3 x 9 grid), SSTAT_PXC in SSDScanNE16.c


class GAP9StaticScanNE16Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        L, P = operatorRepresentation['seq_len'], operatorRepresentation['head_dim']
        nchunk = (P + PXC - 1) // PXC
        name = operatorRepresentation['nodeName']
        return [(name + "_gate_lut_l1", 256 * 4), (name + "_sstat_scratch", nchunk * PXC * L + PXC * L * 4 + 64)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        names = []
        for (name, size), key in zip(GAP9StaticScanNE16Template.computeTransientBuffersSize(ctxt, operatorRepresentation),
                                     ('gate_lut_l1', 'sstat_scratch')):
            ctxt.hoistTransientBuffer(name, size)
            operatorRepresentation[key] = name
            names.append(name)
        return ctxt, operatorRepresentation, names


referenceTemplate = GAP9StaticScanNE16Template("""
// StaticScanNE16 (Name: ${nodeName}, Op: ${nodeOp}) -- constant-parameter Mamba-2 block, per-head NE16 1x1 jobs
memcpy(${gate_lut_l1}, ${gate_lut}, 256 * sizeof(int32_t));
GAP9_StaticScanNE16_i8((const int8_t *) ${x}, (const int8_t *) ${z}, (const uint8_t *) ${wenc}, (const int32_t *) ${comp},
                       (const int32_t *) ${M}, (const int32_t *) ${Dq}, (int32_t *) ${y}, (const int32_t *) ${gate_lut_l1},
                       (uint8_t *) ${sstat_scratch}, ${seq_len}, ${head_dim}, ${n_heads}, ${out_shift});
""")
