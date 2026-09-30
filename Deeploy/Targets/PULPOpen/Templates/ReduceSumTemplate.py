# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ReduceSumTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        return ctxt, operatorRepresentation, []


# Parallel integer ReduceSum.
referenceTemplate = _ReduceSumTemplate("""
// Integer ReduceSum (Name: ${nodeName}, Op: ${nodeOp})
<%
reduceLength = 1
for i, axis in enumerate(axes):
    if axis < 0:
        axes[i] += len(data_in_shape)
    reduceLength = reduceLength * data_in_shape[axes[i]]

restDims = sorted(set(range(len(data_in_shape))).difference(set(axes)))
parallelDim = restDims[-1]
parallelSize = data_in_shape[parallelDim]

shapeStr = ''
for i in data_in_shape[1:]:
    shapeStr += '['+str(i)+']'

accessStr = ''
for j in range(len(data_in_shape)):
    accessStr += '[i_'+str(j)+']'

outIdxStr = ''
outStride = 1
for k in reversed(restDims):
    term = 'i_'+str(k) if k == parallelDim else 'i_'+str(k)+'*'+str(outStride)
    outIdxStr = term if not outIdxStr else term + ' + ' + outIdxStr
    outStride *= data_in_shape[k]
%>
uint32_t core_id = pi_core_id();
uint32_t log2Core = (uint32_t) LOG2(NUM_CORES);
uint32_t chunk = (${parallelSize}U >> log2Core) + ((${parallelSize}U & (NUM_CORES - 1)) != 0);
uint32_t chunk_start = MIN(chunk * core_id, ${parallelSize}U);
uint32_t chunk_stop = MIN(chunk_start + chunk, ${parallelSize}U);

% for i in restDims[:-1]:
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
for(uint32_t i_${parallelDim} = chunk_start; i_${parallelDim}<chunk_stop; i_${parallelDim}++){
int32_t ${data_out}_accumulator = ${input_offset}*${reduceLength};
% for i in axes:
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};
% for i in range(len(axes)):
}
% endfor
${data_out}[${outIdxStr}] = (${data_out_type.referencedType.typeName}) (${data_out}_accumulator + ${output_offset});
}
% for i in range(len(restDims)-1):
}
% endfor
""")
