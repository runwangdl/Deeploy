# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import math
from functools import partial
from typing import Generator, List, Tuple

import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass, \
    contextagnostic
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveGlobalOutputReshapePass, _appendTranspose, _createReshape, _transformLayoutPermutation
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import \
    EngineDiscolorationPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, ReshapeMergePass


def _weightEncode(weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False) -> npt.NDArray[np.uint8]:
    """NE16 weight encoder, ported from pulp-nnx/test/Ne16Weight.py.

    Expected weight shape: (cout, cin, H, W).
    Output layout: (cout, cinMajor, Bits, H*W, cinMinorBytes) where
    CIN_SUBTILE = 16 (single mode, no 1x1 vs 3x3 split like Neureka).
    """
    _NE16_CIN_SUBTILE = 16

    if depthwise:
        weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

    cout, cin, height, width = weight.shape

    # Pad cin to be divisible with CIN_SUBTILE
    if cin % _NE16_CIN_SUBTILE != 0:
        cinPad = _NE16_CIN_SUBTILE - cin % _NE16_CIN_SUBTILE
        weight = np.pad(
            weight,
            ((0, 0), (0, cinPad), (0, 0), (0, 0)),
            "constant",
            constant_values = 0,
        )
        cin = cin + cinPad

    cinMajor = cin // _NE16_CIN_SUBTILE
    cinMinor = _NE16_CIN_SUBTILE

    # (cout, cinMajor, cinMinor, H*W, 1)
    weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)
    # (cout, cinMajor, cinMinor, H*W, Bits)
    weight = np.unpackbits(weight, axis = -1, count = bits, bitorder = "little")
    # (cout, cinMajor, Bits, H*W, cinMinor)
    weight = weight.transpose(0, 1, 4, 3, 2)
    # Pack cinMinor bits into bytes — 16 bits = 2 bytes
    weight = weight.reshape(-1, 8)
    weight = np.packbits(weight, axis = -1, bitorder = "little")
    cinMinorBytes = cinMinor // 8
    # Layout rank varies by conv kind:
    #   - Dense 3x3 (!depthwise, kernel 3x3): rank 4
    #       (cout, cinMajor, Bits, H*W*cinMinorBytes)
    #     — NE16DenseConstraint tiles over weight.shape[3].
    #   - PW 1x1 and DW 3x3: rank 3
    #       (cout, cinMajor, Bits*H*W*cinMinorBytes)
    #     — NE16{Pointwise,Depthwise}Constraint don't need a bits dim.
    if not depthwise and height == 3 and width == 3:
        return weight.reshape(cout, cinMajor, bits, height * width * cinMinorBytes)
    return weight.reshape(cout, cinMajor, bits * height * width * cinMinorBytes)


def _ne16_adjust_weight_memory_layout_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                          ne16EngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == ne16EngineName):
        return graph

    weightTensor = node.inputs[1]

    if not isinstance(weightTensor, gs.Constant):
        return graph

    # Adjust N-EUREKA's weights
    values = weightTensor.values

    if "channels_first" in node.attrs:
        channels_first = node.attrs["channels_first"]
    else:
        channels_first = default_channels_first

    # Pad the input-channel axis to a multiple of NE16's 16-channel subtile BEFORE the
    # offset shift below. _weightEncode pads with 0, but after the shift 0 encodes
    # W = weight_offset (e.g. -127), not W = 0, so for Ki % 16 != 0 the last subtile
    # multiplied whatever bytes follow the pixel's channels in L1 by weight_offset.
    # FEMBA's projections (Ki = 385 and 1540) came out sign-flipped because of this;
    # every CI network has Ki % 16 == 0 and never hit it.
    if node.attrs['group'] == 1:
        cinAxis = 3 if not channels_first else 1
        cin = values.shape[cinAxis]
        if cin % 16 != 0:
            padWidth = [(0, 0)] * values.ndim
            padWidth[cinAxis] = (0, 16 - cin % 16)
            values = np.pad(values, padWidth, "constant", constant_values = 0)

    # Extract weight offset and translate weights by the offset
    weight_offset = values.min()
    values = values - weight_offset
    node.attrs["weight_offset"] = weight_offset

    # Weight encode expects channels-first (cout, cin_per_group, H, W)
    if not channels_first:
        values = values.transpose(0, 3, 1, 2)

    bits = 8  # Support only 8 bit weights for now
    if node.attrs['group'] == 1:
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = False)
    else:
        # Depthwise: Deeploy's NHWC pass leaves weight as
        # (cin_per_group=1, cout=group, H, W) after the transpose above;
        # Ne16Weight.py's encode expects standard (cout, cin_per_group, H, W)
        # — swap axes 0/1 before encoding so the result is a single packed
        # (1, 1, packed_bytes) block across up to NE16_SUBTILE_INPUT_CHANNEL=16
        # parallel output channels.
        values = values.transpose(1, 0, 2, 3)
        weightTensor.values = _weightEncode(values.astype(np.uint8), bits, depthwise = True)
    weightTensor.name = f"{name}_{weightTensor.name}"

    return graph


@contextagnostic
class NE16AdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_ne16_adjust_weight_memory_layout_fun,
                    default_channels_first = default_channels_first,
                    ne16EngineName = ne16EngineName), "_NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS",
            NonBranchingMatcher(regex_op = True))


def _findAllMultiplicands(x: int) -> List[int]:
    multiplicands = []
    tmpX = x
    for i in range(2, int(math.sqrt(x)) + 1):  # sqrt(x) itself must be tried: 9 = 3*3
        while tmpX % i == 0:
            multiplicands.append(i)
            tmpX = tmpX / i

    if x // math.prod(multiplicands) > 1:
        multiplicands.append(x // math.prod(multiplicands))

    return multiplicands


def _findAllReshapeOptions(dim: int) -> Generator[Tuple[int, int], None, None]:
    multiplicands = _findAllMultiplicands(dim)
    for combLen in range(1, 1 + (len(multiplicands) // 2)):
        for comb in itertools.combinations(multiplicands, combLen):
            a = math.prod(comb)
            b = dim // a
            yield a, b


# NE16 retires a 3x3 output window per subtile. The 6 this used to divide by is
# N-EUREKA's window, which this file was written against.
NE16_SPATIAL_SUBTILE = 3


def _nSubtiles(dims: Tuple[int, int]):
    return math.ceil(dims[0] / NE16_SPATIAL_SUBTILE) * math.ceil(dims[1] / NE16_SPATIAL_SUBTILE)


def _findLowestNumberOfSubtilesReshapeOptions(dim: int) -> List[Tuple[int, int]]:
    lowestNumberOfSubtiles = dim
    bestOptions: List[Tuple[int, int]] = [(dim, 1)]
    for option in _findAllReshapeOptions(dim):
        nSubtiles = _nSubtiles(option)
        if nSubtiles < lowestNumberOfSubtiles:
            lowestNumberOfSubtiles = nSubtiles
            bestOptions = [option]
        elif nSubtiles == lowestNumberOfSubtiles:
            bestOptions.append(option)
    return bestOptions


def _bestReshapeOption(dim: int) -> Tuple[int, int]:
    smallestDim = dim
    biggestDim = 1
    for option in _findLowestNumberOfSubtilesReshapeOptions(dim):
        if option[0] < smallestDim:
            smallestDim = option[0]
            biggestDim = option[1]
        elif option[1] < smallestDim:
            smallestDim = option[1]
            biggestDim = option[0]
    return biggestDim, smallestDim


def _ne16_reshape_pointwise_convolution_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                            ne16EngineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == ne16EngineName):
        return graph

    if not (node.attrs["kernel_shape"] == [1, 1]):
        return graph

    if "channels_first" in node.attrs:
        channels_first = node.attrs["channels_first"]
    else:
        channels_first = default_channels_first

    def extractSpatialDims(shape: List[int]) -> List[int]:
        if channels_first:
            return shape[-2:]
        else:
            return shape[-3:-1]

    def replaceSpatialDims(shape: List[int], newSpatialDims: Tuple[int, int]) -> List[int]:
        if channels_first:
            return shape[:-2] + list(newSpatialDims)
        else:
            return shape[:-3] + list(newSpatialDims) + shape[-1:]

    _input = node.inputs[0]
    spatialDims = extractSpatialDims(_input.shape)
    newSpatialDims = _bestReshapeOption(math.prod(spatialDims))
    newInputShape = replaceSpatialDims(_input.shape, newSpatialDims)

    inputReshapeNode, reshapedInput = _createReshape(_input, name, newInputShape)
    graph.nodes.append(inputReshapeNode)
    node.inputs[0] = reshapedInput

    output = node.outputs[0]
    newOutputShape = replaceSpatialDims(output.shape, newSpatialDims)
    reshapedOutput = gs.Variable(output.name + "_Reshaped", dtype = output.dtype, shape = newOutputShape)
    outputReshapeNode, _ = _createReshape(reshapedOutput, name, output.shape, output)
    graph.nodes.append(outputReshapeNode)
    node.outputs[0] = reshapedOutput

    return graph


@contextagnostic
class NE16ReshapePointwiseConvolutionPass(ReplaceSequentialPatternPass):
    """Reshape pointwise convolution's spatial dimensions so that they work better for N-EUREKA's hardware tiling"""

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_ne16_reshape_pointwise_convolution_fun,
                    default_channels_first = default_channels_first,
                    ne16EngineName = ne16EngineName), "_NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS",
            NonBranchingMatcher(regex_op = True))


_SIGNED_PRODUCERS = ("SILU", "SelectiveScan", "SSD_Scan", "Mamba3_Scan", "Softplus", "Add", "MatMul", "Gemm",
                     "iLayerNorm", "LayerNormalization", "iRMSNorm", "Sub", "Mul")
_SHAPE_OPS = ("Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Slice", "Split")


def _inputIsSigned(tensor: gs.Variable) -> bool:
    """Best-effort signedness of an activation before type inference has run: follow the
    producer through shape-only ops; requantizing producers carry a 'signed' attribute."""
    seen = 0
    while len(tensor.inputs) == 1 and tensor.inputs[0].op in _SHAPE_OPS and seen < 8:
        tensor = tensor.inputs[0].inputs[0]
        seen += 1
    if len(tensor.inputs) != 1:
        return True  # graph input: assume signed (FEMBA); CI's uint8 nets never reach here
    producer = tensor.inputs[0]
    if "signed" in producer.attrs:
        v = producer.attrs["signed"]
        v = v.values if isinstance(v, gs.Constant) else v
        return bool(int(np.asarray(v).reshape(-1)[0]))
    if producer.op in ("Relu", "MaxPool", "AveragePool"):
        return False
    return True


def _ne16_unsigned_input_fun(graph: gs.Graph, match: Match, name: str, ne16EngineName: str):
    """NE16 has no signed-activation mode (config bit 26 is undefined in the gvsoc model and in
    pulp-nnx): it multiplies the raw bytes as uint8. For a signed int8 feature map x this pass
    feeds NE16 x_u = x + 128 (a RequantShift with mul 1 / add 128 / div 1 to uint8) and removes
    the 128 * sum_k W[k, ko] that adds to every output through NE16's own per-channel bias, so
    the accumulator is exact. The node stays a RequantizedConv with scale 1, shift 0 and a
    32-bit output; the original requantization (mul, add, div) is then done by the cluster's
    RequantShift kernel, because NE16's norm/quant stage truncates acc * scale to 32 bit and
    FEMBA's scales (mul ~ 7e4 over 2^24) overflow it."""
    node = list(match.nodes_map.values())[0]
    if os.environ.get("DEEPLOY_NE16_UNSIGNED_PASS", "1") == "0":  # debugging kill switch
        return graph
    if node.attrs.get("engine") != ne16EngineName:
        return graph
    if node.attrs.get("kernel_shape") != [1, 1] or node.attrs.get("group", 1) != 1:
        return graph
    if node.attrs.get("ne16_unsigned_input_done"):
        return graph
    x = node.inputs[0]
    weight = node.inputs[1]
    if not isinstance(weight, gs.Constant) or not isinstance(x, gs.Variable):
        return graph
    if not _inputIsSigned(x):
        return graph

    W = weight.values.astype(np.int64)
    Ko = W.shape[0]
    comp = (-128 * W.reshape(Ko, -1).sum(axis = 1)).astype(np.int32)

    def _attr(v):
        return gs.Constant(f"{name}_attr_{np.random.randint(1 << 30)}", np.array([v]))

    # 1) x -> x + 128 as uint8
    x_u = gs.Variable(name = f"{name}_{x.name}_u8", dtype = np.float32, shape = x.shape)
    graph.nodes.append(
        gs.Node(op = "RequantShift",
                name = f"{name}_to_u8",
                inputs = [
                    x,
                    # scalar mul/add: the uniform kernel is the 8-core one (the per-channel
                    # RequantShift_s8_u8_NHWC is a single-core loop with a modulo per element,
                    # 5.2 M cycles per FEMBA projection input)
                    # (2x + 256 + 1) >> 1 == x + 128 exactly; div 2 instead of 1 because the
                    # RequantShift kernels' rounding term is 1 << (log2D - 1), undefined at log2D 0
                    gs.Constant(f"{name}_u8_mul", np.array([2], dtype = np.int32)),
                    gs.Constant(f"{name}_u8_add", np.array([256], dtype = np.int32))
                ],
                outputs = [x_u],
                attrs = {
                    "div": _attr(2),
                    "n_levels_out": np.array([256.0]),
                    "signed": np.array([0.0])
                }))
    node.inputs[0] = x_u

    # 2) the conv: scale 1, bias -128*sum(W), shift 0, 32-bit output
    y = node.outputs[0]
    orig = None
    if node.op == "RequantizedConv":
        def _arr(v):  # attributes may arrive as python scalars, arrays or Constants; parsers want arrays/Constants
            if isinstance(v, gs.Constant):
                return v
            return np.asarray(v, dtype = np.float64).reshape(-1)
        orig = (node.inputs[2], node.inputs[3], _arr(node.attrs["div"]),
                _arr(node.attrs.get("n_levels_out", node.attrs.get("n_levels"))), _arr(node.attrs.get("signed", 1.0)))
        node.inputs[2] = gs.Constant(f"{name}_scale1", np.ones(Ko, dtype = np.int32))
        node.inputs[3] = gs.Constant(f"{name}_comp", comp)
    else:
        node.op = "RequantizedConv"
        node.inputs.append(gs.Constant(f"{name}_scale1", np.ones(Ko, dtype = np.int32)))
        node.inputs.append(gs.Constant(f"{name}_comp", comp))
    node.attrs["div"] = _attr(1)
    node.attrs["n_levels_out"] = np.array([float(2**32)])
    node.attrs["signed"] = np.array([1.0])
    node.attrs["shift"] = _attr(0)
    node.attrs["ne16_unsigned_input_done"] = 1

    # 3) the original requantization, on the cluster
    if orig is not None:
        mul0, add0, div0, nlev0, signed0 = orig
        # The Conv/GEMM + RequantShift merge passes fold the rounding half 2^(shift-1) into
        # `add` because the fused pulp-nn kernels truncate; the standalone RequantShift
        # kernel rounds itself, so take it out again or every output rounds twice.
        shift = int(np.log2(float(np.asarray(div0.values if isinstance(div0, gs.Constant) else div0).reshape(-1)[0])))
        addVals = np.asarray(add0.values, dtype = np.int64) - ((1 << (shift - 1)) if shift > 0 else 0)
        add1 = gs.Constant(f"{name}_requant_add", addVals.astype(np.int32))
        y32 = gs.Variable(name = f"{name}_{y.name}_i32", dtype = np.float32, shape = y.shape)
        node.outputs[0] = y32
        graph.nodes.append(
            gs.Node(op = "RequantShift",
                    name = f"{name}_requant",
                    inputs = [y32, mul0, add1],
                    outputs = [y],
                    attrs = {
                        "div": div0,
                        "n_levels_out": nlev0,
                        "signed": signed0
                    }))
    return graph


@contextagnostic
class NE16UnsignedInputPass(ReplaceSequentialPatternPass):

    def __init__(self, ne16EngineName: str = "NE16"):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)
        super().__init__(graph, partial(_ne16_unsigned_input_fun, ne16EngineName = ne16EngineName),
                         "_NE16_UNSIGNED_INPUT_PASS", NonBranchingMatcher(regex_op = True))


class ConvEngineDiscolorationPass(EngineDiscolorationPass):

    def __init__(self):
        pattern = gs.Graph()
        _input = gs.Variable(name = 'input')
        output = pattern.layer(inputs = [_input], outputs = ['output'], op = 'RequantizedConv|Conv', name = 'conv')
        pattern.outputs.append(output)
        pattern.inputs.append(_input)
        super().__init__(pattern, "_CONV_ENGINE_DISCOLORATION_PASS", matcher = NonBranchingMatcher(regex_op = True))


def _ne16_dw_layout_fixup_fun(graph: gs.Graph, match: Match, name: str, ne16EngineName: str):
    """Convert NE16-colored DW conv from PULP NHWC layout to NE16 NHWC layout.

    After PULPNCHWtoNHWCPass runs, every DW conv has:
      - weight in PULP NHWC layout (cout, H, W, cin/g)
      - input NOT transposed (PULP DW kernel convention)

    NE16 DW expects:
      - weight in NE16 NHWC layout (cin/g=1, H, W, cout)
      - input in NHWC layout

    For NE16-colored DW convs we do both adjustments here. Cluster-colored
    DW convs (e.g. stride-2 fallbacks when --enable-3x3 is on) are left
    untouched, so the PULP cluster path still works.
    """
    node = list(match.nodes_map.values())[0]
    if node.op not in ("Conv", "RequantizedConv"):
        return graph
    if node.attrs.get("group", 1) == 1:
        return graph
    if node.attrs.get("engine") != ne16EngineName:
        return graph
    if len(node.inputs) < 2 or not isinstance(node.inputs[1], gs.Constant):
        return graph

    weightTensor = node.inputs[1]
    if weightTensor.values.ndim != 4:
        return graph

    # Weight: (cout, H, W, cin/g=1) -> (cin/g=1, H, W, cout)
    weightTensor.values = weightTensor.values.transpose(3, 1, 2, 0)

    # PULP DW NHWC doesn't insert an input transpose; NE16 DW needs NHWC input.
    tensorIn = node.inputs[0]
    spatialDims = 2
    permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, False)
    graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))

    return graph


@contextagnostic
class NE16DwLayoutFixupPass(ReplaceSequentialPatternPass):

    def __init__(self, ne16EngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(graph, partial(_ne16_dw_layout_fixup_fun, ne16EngineName = ne16EngineName),
                         "_NE16_DW_LAYOUT_FIXUP_PASS", NonBranchingMatcher(regex_op = True))


@contextagnostic
class NE16OptimizationPass(SequentialPass):

    def __init__(self, default_channels_first: bool, ne16EngineName: str):
        super().__init__(NE16DwLayoutFixupPass(ne16EngineName),
                         NE16AdjustWeightMemoryLayoutPass(default_channels_first, ne16EngineName),
                         NE16ReshapePointwiseConvolutionPass(default_channels_first, ne16EngineName),
                         ReshapeMergePass(),
                         ReshapeConstOptPass(),
                         RemoveGlobalOutputReshapePass(),
                         name_prefix = '')
