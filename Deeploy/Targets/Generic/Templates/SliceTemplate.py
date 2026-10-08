# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _SliceTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Immediate-ify start
        startsBuffer = ctxt.lookup(operatorRepresentation['starts'])
        axesBuffer = ctxt.lookup(operatorRepresentation['axes'])
        endsBuffer = ctxt.lookup(operatorRepresentation['ends'])
        stepsBuffer = ctxt.lookup(operatorRepresentation['steps'])

        startsBuffer._deploy = False
        axesBuffer._deploy = False
        endsBuffer._deploy = False
        stepsBuffer._deploy = False

        axes = [int(x) for x in axesBuffer.values]
        starts = [int(x) for x in startsBuffer.values]
        ends = [int(x) for x in endsBuffer.values]
        steps = [int(x) for x in stepsBuffer.values]

        shape = operatorRepresentation['data_in_shape']
        dims = len(shape)

        # Normalize negative axes
        axes = [ax + dims if ax < 0 else ax for ax in axes]

        # Normalize negative start/end and clamp to valid range (ONNX spec).
        for i, (ax, start, end, step) in enumerate(zip(axes, starts, ends, steps)):
            dim = shape[ax]
            if step > 0:
                start = max(0, min(dim, start + dim if start < 0 else start))
                end = max(0, min(dim, end + dim if end < 0 else end))
            else:
                start = max(0, min(dim - 1, start + dim if start < 0 else start))
                if end < 0:
                    end = end + dim
                    if end < 0:
                        end = -1  # sentinel: include element 0
                else:
                    end = min(dim - 1, end)
            starts[i] = start
            ends[i] = end

        operatorRepresentation['axes'] = axes
        operatorRepresentation['starts'] = starts
        operatorRepresentation['ends'] = ends
        operatorRepresentation['steps'] = steps
        operatorRepresentation['data_in_size'] = np.prod(shape)

        return ctxt, operatorRepresentation, []


referenceTemplate = _SliceTemplate("""
// Slice (Name: ${nodeName}, Op: ${nodeOp})
<%
dimSteps = [1] * len(data_in_shape)
for i in range(len(data_in_shape) - 2, -1, -1):
    dimSteps[i] = dimSteps[i + 1] * int(data_in_shape[i + 1])

elemBytes = data_out_type.referencedType.typeWidth // 8

# Tiled flow: data_in_size becomes a str pointer-ref; L1 tile is contiguous so a single bulk memcpy is correct (#134).
tiledFastPath = (isinstance(data_in_size, str) or int(context.get('tiled_slice', 0)) == 1) and all(int(s) == 1 for s in steps)

lastAx = int(axes[-1])
# Un-tiled fast path: collapse innermost step==1 axis into one memcpy.
collapseInner = (int(steps[-1]) == 1)
if collapseInner:
    transferSize = (int(ends[-1]) - int(starts[-1])) * dimSteps[lastAx]
    innerConstOffset = int(starts[-1]) * dimSteps[lastAx]
    loopAxes = list(axes[:-1])
    loopStarts = list(starts[:-1])
    loopEnds = list(ends[:-1])
    loopSteps = list(steps[:-1])
else:
    transferSize = dimSteps[lastAx]
    innerConstOffset = 0
    loopAxes = list(axes)
    loopStarts = list(starts)
    loopEnds = list(ends)
    loopSteps = list(steps)

if int(axes[0]) > 0:
    preAxes = list(range(int(axes[0])))
else:
    preAxes = []

allLoopAxes = list(preAxes) + list(loopAxes)
offsetAxis = allLoopAxes[-1] if allLoopAxes else None
memcpyBytes = transferSize * elemBytes
%>

% if tiledFastPath:
memcpy(${data_out}, ${data_in}, ${data_in_size} * ${elemBytes});
% else:
${data_out_type.referencedType.typeName}* ref_${data_out} = ${data_out};
% for axis in allLoopAxes:
uint32_t ${data_out}_offset_${axis} = 0;
% endfor

% for axis, axisLen in zip(preAxes, list(data_in_shape)):
for(uint32_t i_${axis} = 0; i_${axis} < ${axisLen}; i_${axis}++){
% if axis == 0:
${data_out}_offset_0 = ${dimSteps[axis]} * i_${axis};
% else:
${data_out}_offset_${axis} = ${data_out}_offset_${axis-1} + ${dimSteps[axis]} * i_${axis};
% endif
% endfor
% for axis, start, end, step in zip(loopAxes, loopStarts, loopEnds, loopSteps):
% if int(step) > 0:
for(uint32_t i_${axis} = ${start}; i_${axis} < ${end}; i_${axis} += ${step}){
% else:
for(int32_t i_${axis} = ${start}; i_${axis} > ${end}; i_${axis} += ${step}){
% endif
% if axis == 0:
${data_out}_offset_0 = ${dimSteps[axis]} * i_${axis};
% else:
${data_out}_offset_${axis} = ${data_out}_offset_${axis-1} + ${dimSteps[axis]} * i_${axis};
% endif
% endfor
% if offsetAxis is None:
memcpy(ref_${data_out}, ${data_in} + ${innerConstOffset}, ${memcpyBytes});
% elif innerConstOffset == 0:
memcpy(ref_${data_out}, ${data_in} + ${data_out}_offset_${offsetAxis}, ${memcpyBytes});
% else:
memcpy(ref_${data_out}, ${data_in} + ${data_out}_offset_${offsetAxis} + ${innerConstOffset}, ${memcpyBytes});
% endif
ref_${data_out} += ${transferSize};
% for axis in allLoopAxes:
}
% endfor
% endif
""")
