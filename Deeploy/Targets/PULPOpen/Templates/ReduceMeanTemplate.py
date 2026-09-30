# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ReduceMeanTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = 0  #-(data_out._signed==0) * int(data_in.nLevels/2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _ReduceMeanTemplate("""
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
int32_t ${data_out}_accumulator = 0;
<%
reduceLength = 1
for i in axes:
    reduceLength = reduceLength * data_in_shape[i]
%>
<%
    shapeStr = ''
    accessStr = ''
%>
% for idx, i in enumerate(data_in_shape[1:]):
<%
    shapeStr += '['+str(i)+']'
%>
% endfor
% for j in range(len(data_in_shape)):
<%
    accessStr += '[i_'+str(j)+']'
%>
% endfor
${data_out_type.typeName} dummy_${data_out} = ${data_out};

<%
restDims = set(list(range(len(data_in_shape)))).difference(set(axes))
%>
% for i in list(restDims):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator = ${input_offset}*${reduceLength};
% for i in list(axes):
for(uint32_t i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator += ((${data_in_type.referencedType.typeName} (*)${shapeStr})${data_in})${accessStr};

% for i in range(len(axes)):
}
% endfor
% if keepdims:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) ((${data_out}_accumulator + ${data_out}_sgn*(${reduceLength}>>1)) / ${reduceLength} + ${output_offset});
% else:
<%

import numpy as np

shift = None
if (np.log2(reduceLength) - int(np.log2(reduceLength))) == 0:
    shift = int(np.log2(reduceLength))
%>
% if shift is not None:
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) (((${data_out}_accumulator + (1<<(${shift}-1))) >> ${shift}) + ${output_offset});
% else:
int8_t ${data_out}_sgn = 0;
${data_out}_sgn = -(${data_out}_accumulator<0) + (${data_out}_accumulator >= 0);
*dummy_${data_out}++ = (${data_out_type.referencedType.typeName}) ((${data_out}_accumulator + ${data_out}_sgn*(${reduceLength}>>1)) / ${reduceLength} + ${output_offset});
% endif
% endif
% for i in range(len(restDims)):
}
% endfor
END_SINGLE_CORE
""")

S8S32ParallelTemplate = _ReduceMeanTemplate("""
// ReduceMean (Name: ${nodeName}, Op: ${nodeOp})
<%
reduceLength = 1
for i in axes:
    reduceLength = reduceLength * data_in_shape[i]

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

import numpy as np
shift = None
if (np.log2(reduceLength) - int(np.log2(reduceLength))) == 0:
    shift = int(np.log2(reduceLength))
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
% if shift is not None:
${data_out}[${outIdxStr}] = (${data_out}_accumulator + (1<<(${shift}-1))) >> ${shift};
% else:
int8_t ${data_out}_sgn = -(${data_out}_accumulator<0) + (${data_out}_accumulator >= 0);
${data_out}[${outIdxStr}] = (${data_out}_accumulator + ${data_out}_sgn*(${reduceLength}>>1)) / ${reduceLength};
% endif
}
% for i in range(len(restDims)-1):
}
% endfor
""")
