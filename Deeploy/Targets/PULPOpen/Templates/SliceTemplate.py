# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.Generic.Templates.SliceTemplate import _SliceTemplate

# PULP Slice: tiled step +1/-1 use a parallel memcpy fast path; everything else uses the sequential loop.

referenceTemplate = _SliceTemplate("""
// Slice (Name: ${nodeName}, Op: ${nodeOp})
<%
dimSteps = [1] * len(data_in_shape)
for i in range(len(data_in_shape) - 2, -1, -1):
    dimSteps[i] = dimSteps[i + 1] * int(data_in_shape[i + 1])

elemBytes = data_out_type.referencedType.typeWidth // 8

lastAx = int(axes[-1])
rowElems = dimSteps[lastAx]
rowBytes = rowElems * elemBytes

# Parallel fast paths: forward only for tiled slices (data_in_size is a per-tile ref); the row-reversal path
# also for untiled slices (FEMBA's sequence flips stay untiled and ran the sequential loop on one core, ~9 cycles/B).
# Tiled flow: the input tile is anchored at the slice start, so the slice is a plain copy of
# data_in_size elements. data_in_size is a per-tile symbol when tiles differ, or a folded constant when
# there is a single tile; input_cube_anchored (1 in the tiled flow, 0 untiled) covers the latter case,
# where the old `isinstance(data_in_size, str)` test wrongly fell back to the offset path and added
# `starts` a second time (RSSM GRU gate slices [128:256] and [256:384] came out garbage).
forwardPath = (isinstance(data_in_size, str) or int(input_cube_anchored) == 1) and all(int(s) == 1 for s in steps)
reversePath = all(int(s) == -1 for s in steps) and len(axes) == 1 and int(axes[0]) == 1 and \
    int(starts[0]) in (-1, int(data_in_shape[1]) - 1) and (int(ends[0]) == -1 or int(ends[0]) <= -int(data_in_shape[1]) - 1)

# Sequential-fallback setup (used only when neither fast path applies).
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

% if forwardPath:
{
    const uint32_t _total = ${data_in_size};
    const int32_t _core_id = pi_core_id();
    const int32_t _log2Core = LOG2(NUM_CORES);
    const uint32_t _chunk = (_total >> _log2Core) + ((_total & (NUM_CORES - 1)) != 0);
    const uint32_t _elem_start = MIN((uint32_t)_core_id * _chunk, _total);
    const uint32_t _elem_end   = MIN(_elem_start + _chunk, _total);
    memcpy((char*)${data_out} + _elem_start * ${elemBytes},
           (char*)${data_in}  + _elem_start * ${elemBytes},
           (_elem_end - _elem_start) * ${elemBytes});
}
% elif reversePath:
{
    const uint32_t _rowBytes = ${rowBytes};
    const uint32_t _nRows = (${data_in_size}) / ${rowElems};
    const int32_t _core_id = pi_core_id();
    const int32_t _log2Core = LOG2(NUM_CORES);
    const uint32_t _chunk = (_nRows >> _log2Core) + ((_nRows & (NUM_CORES - 1)) != 0);
    const uint32_t _row_start = MIN((uint32_t)_core_id * _chunk, _nRows);
    const uint32_t _row_end   = MIN(_row_start + _chunk, _nRows);
    for (uint32_t _i = _row_start; _i < _row_end; _i++) {
        memcpy((char*)${data_out} + _i * _rowBytes,
               (char*)${data_in}  + (_nRows - 1 - _i) * _rowBytes,
               _rowBytes);
    }
}
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
