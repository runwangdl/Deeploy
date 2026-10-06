# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation


def _getNumTiles(fullDim: int, tileDim: int) -> int:
    return int(np.ceil(fullDim / tileDim))


def _getBorderTileSize(fullDim: int, tileDim: int) -> int:
    return fullDim % tileDim if fullDim % tileDim > 0 else tileDim


def ioStridesFromDimensions(width: int, channel: int, bits: int) -> Tuple[int, int]:
    """stridesFromDimensions
    Returns strides in bytes.
    """
    width_stride = channel * bits // 8
    height_stride = width * width_stride
    return height_stride, width_stride


def getNormQuantConf0(use_relu: bool, layerwise_output_shift: int, scale_bits: int, use_bias: bool,
                      use_shift: bool) -> int:
    conf0 = 0
    conf0 |= 1 << 4  # Use Normalization and quantization
    if scale_bits == 32:
        conf0 |= 2 << 12
    conf0 |= layerwise_output_shift << 16
    if not use_relu:
        conf0 |= 1 << 23
    if use_shift:
        conf0 |= 1 << 24
    if use_bias:
        conf0 |= 1 << 25
    return conf0


def getInputAddrOffset(width_in: int, width_in_stride: int, padding_top: int, padding_left: int) -> int:
    return (padding_top * width_in + padding_left) * width_in_stride


class NE16ConvTemplate(NodeTemplate):

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    @abstractmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        pass

    @classmethod
    @abstractmethod
    def getWeightStrides(cls, channel_in: int, weight_bits: int = 8) -> Tuple[int, int, int]:
        pass

    @classmethod
    @abstractmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        pass

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in: ConstantBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        data_out: ConstantBuffer = ctxt.lookup(operatorRepresentation['data_out'])
        weight: ConstantBuffer = ctxt.lookup(operatorRepresentation['weight'])

        operatorRepresentation['input_signed'] = data_in._type.referencedType.typeMin < 0
        operatorRepresentation['use_relu'] = data_out._type.referencedType.typeMin >= 0

        operatorRepresentation['input_bits'] = data_in._type.referencedType.typeWidth
        operatorRepresentation['output_bits'] = data_out._type.referencedType.typeWidth
        # bit-serial weights: the pass may have packed fewer planes than the storage type's width
        operatorRepresentation['weight_bits'] = int(operatorRepresentation.get('weight_bits', weight._type.referencedType.typeWidth))

        operatorRepresentation["input_typeWidth_bytes"] = int(np.ceil(data_in._type.referencedType.typeWidth / 8))
        operatorRepresentation["output_typeWidth_bytes"] = int(np.ceil(data_out._type.referencedType.typeWidth / 8))

        operatorRepresentation["weight_addr_offset"] = 0

        operatorRepresentation["use_wmem"] = hasattr(weight,
                                                     "_memoryLevel") and weight._memoryLevel == "WeightMemory_SRAM"

        dim_im_in_x_stride, dim_im_in_y_stride = ioStridesFromDimensions(operatorRepresentation["dim_im_in_y"],
                                                                         operatorRepresentation["ch_im_in"],
                                                                         operatorRepresentation["input_bits"])
        operatorRepresentation["dim_im_in_y_stride"] = dim_im_in_y_stride
        operatorRepresentation["dim_im_in_x_stride"] = dim_im_in_x_stride

        dim_im_out_x_stride, dim_im_out_y_stride = ioStridesFromDimensions(operatorRepresentation["dim_im_out_y"],
                                                                           operatorRepresentation["ch_im_out"],
                                                                           operatorRepresentation["output_bits"])
        operatorRepresentation["dim_im_out_y_stride"] = dim_im_out_y_stride
        operatorRepresentation["dim_im_out_x_stride"] = dim_im_out_x_stride

        operatorRepresentation["input_addr_offset"] = getInputAddrOffset(operatorRepresentation["dim_im_in_y"],
                                                                         operatorRepresentation["dim_im_in_y_stride"],
                                                                         operatorRepresentation["padding_y_top"],
                                                                         operatorRepresentation["padding_x_left"])

        nKo, nKi, nHo, nWo, bKo, bKi, bHo, bWo, bHi, bWi = self.getCounters(
            operatorRepresentation["ch_im_in"], operatorRepresentation["dim_im_out_x"],
            operatorRepresentation["dim_im_out_y"], operatorRepresentation["ch_im_out"],
            operatorRepresentation["padding_y_bottom"], operatorRepresentation["padding_x_right"],
            operatorRepresentation)

        operatorRepresentation["nKo"] = nKo
        operatorRepresentation["nKi"] = nKi
        operatorRepresentation["nHo"] = nHo
        operatorRepresentation["nWo"] = nWo
        operatorRepresentation["bKo"] = bKo
        operatorRepresentation["bKi"] = bKi
        operatorRepresentation["bHo"] = bHo
        operatorRepresentation["bWo"] = bWo
        operatorRepresentation["bHi"] = bHi
        operatorRepresentation["bWi"] = bWi

        weightStrideD0, weightStrideD1, weightStrideD2 = self.getWeightStrides(operatorRepresentation["ch_im_in"],
                                                                                operatorRepresentation["weight_bits"])

        operatorRepresentation["weightStrideD0"] = weightStrideD0
        operatorRepresentation["weightStrideD1"] = weightStrideD1
        operatorRepresentation["weightStrideD2"] = weightStrideD2

        operatorRepresentation["conf0"] = self.getConf0(operatorRepresentation["output_bits"],
                                                        operatorRepresentation["weight_bits"],
                                                        operatorRepresentation["input_signed"],
                                                        operatorRepresentation["use_wmem"])

        operatorRepresentation["wmem_addr_offset"] = 0x10400000 if operatorRepresentation["use_wmem"] else 0

        operatorRepresentation["ne16_kernel_shape"] = self.NE16_KERNEL_SHAPE
        operatorRepresentation["ne16_depthwise"] = self.NE16_IS_DEPTHWISE
        operatorRepresentation["ne16_subtile_output_channel"] = self.NE16_SUBTILE_OUTPUT_CHANNEL

        # If requantized
        if operatorRepresentation["mul"] != "NULL":
            mulBuff = ctxt.lookup(operatorRepresentation["mul"])
            mulBits = mulBuff._type.referencedType.typeWidth
            operatorRepresentation["conf0"] |= getNormQuantConf0(operatorRepresentation["use_relu"],
                                                                 operatorRepresentation["log2D"], mulBits, "add"
                                                                 in operatorRepresentation, False)
        return ctxt, operatorRepresentation, []


class NE162DPWConvTemplate(NE16ConvTemplate):

    NE16_KERNEL_SHAPE = 1
    NE16_IS_DEPTHWISE = 0
    NE16_SUBTILE_OUTPUT_CHANNEL = 32

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt, operatorRepresentation, names = super().alignToContext(ctxt, operatorRepresentation)
        # Flat pixel column (W == 1, stride 1, no padding): the tile's pixels are contiguous, so the execution
        # template re-views them as silicon-legal jobs (see NE16PWTaskExecutionTemplateStr).
        operatorRepresentation["pw_flat"] = (operatorRepresentation["dim_im_out_y"] == 1
                                             and operatorRepresentation["dim_im_in_y"] == 1
                                             and all(int(p) == 0 for p in operatorRepresentation.get("pads", [0]))
                                             and all(int(st) == 1 for st in operatorRepresentation.get("strides", [1])))
        operatorRepresentation["pw_npix"] = operatorRepresentation["dim_im_out_x"] * operatorRepresentation["dim_im_out_y"]
        return ctxt, operatorRepresentation, names

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        # NE16 subtiles: INPUT_CHANNEL=16, OUTPUT_HxW=3x3, OUTPUT_CHANNEL=32
        n_channel_out_subtiles = _getNumTiles(channel_out, 32)
        n_channel_in_subtiles = _getNumTiles(channel_in, 16)
        n_height_out_subtiles = _getNumTiles(height_out, 3)
        n_width_out_subtiles = _getNumTiles(width_out, 3)

        channel_out_border = _getBorderTileSize(channel_out, 32)
        channel_in_border = _getBorderTileSize(channel_in, 16)
        height_out_border = _getBorderTileSize(height_out, 3)
        width_out_border = _getBorderTileSize(width_out, 3)
        height_in_border = height_out_border - padding_bottom
        width_in_border = width_out_border - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int, weight_bits: int = 8) -> Tuple[int, int, int]:
        # NE16 PW 1x1: per (cout, cinMajor) block = bits * H*W * cinMinorBytes
        # = 8 * 1 * 2 = 16 bytes for 8-bit weights with CIN_SUBTILE=16
        n_channel_in = _getNumTiles(channel_in, 16)
        _NE16_PW_WEIGHT_BYTES = weight_bits * 2  # bits * HW * cinMinorBytes = bits*1*2
        return _NE16_PW_WEIGHT_BYTES, _NE16_PW_WEIGHT_BYTES * n_channel_in, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        conf0 |= 2 << 5  # PW MODE
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


class NE162DDWConvTemplate(NE16ConvTemplate):

    NE16_KERNEL_SHAPE = 3
    NE16_IS_DEPTHWISE = 1
    # For DW, hardware replicates input channels as output channels, so the
    # output-channel subtile size equals the input-channel subtile (16).
    NE16_SUBTILE_OUTPUT_CHANNEL = 16

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        _ = operatorRepresentation  # operatorRepresentation not accessed for now because it's just for pointwise kernels

        # NE16 DW 3x3: CIN_SUBTILE=16 single mode, output 3x3
        n_channel_out_subtiles = _getNumTiles(channel_out, 16)
        n_channel_in_subtiles = n_channel_out_subtiles
        n_height_out_subtiles = _getNumTiles(height_out, 3)
        n_width_out_subtiles = _getNumTiles(width_out, 3)

        channel_out_border = _getBorderTileSize(channel_out, 16)
        channel_in_border = channel_out_border
        height_out_border = _getBorderTileSize(height_out, 3)
        width_out_border = _getBorderTileSize(width_out, 3)
        height_in_border = height_out_border + 2 - padding_bottom
        width_in_border = width_out_border + 2 - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int, weight_bits: int = 8) -> Tuple[int, int, int]:
        # Match ne16_task_set_strides for depthwise 3x3:
        #   d0 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride
        #      = 3 * 3 * 2 = 18
        #   d1 = 0 (DW has no cin-major striding from the HW's perspective).
        _NE16_FILTER_SIZE = 3
        _NE16_WEIGHT_D0_STRIDE_MODE8 = 2
        d0 = _NE16_FILTER_SIZE * _NE16_FILTER_SIZE * _NE16_WEIGHT_D0_STRIDE_MODE8
        return d0, 0, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        conf0 |= 1 << 5  # DW MODE
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


class NE162DDenseConvTemplate(NE16ConvTemplate):

    NE16_KERNEL_SHAPE = 3
    NE16_IS_DEPTHWISE = 0
    NE16_SUBTILE_OUTPUT_CHANNEL = 32

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        _ = operatorRepresentation  # operatorRepresentation not accessed for now because it's just for pointwise kernels

        # NE16 Dense 3x3: CIN_SUBTILE=16, OUTPUT 3x3x32
        n_channel_out_subtiles = _getNumTiles(channel_out, 32)
        n_channel_in_subtiles = _getNumTiles(channel_in, 16)
        n_height_out_subtiles = _getNumTiles(height_out, 3)
        n_width_out_subtiles = _getNumTiles(width_out, 3)

        channel_out_border = _getBorderTileSize(channel_out, 32)
        channel_in_border = _getBorderTileSize(channel_in, 16)
        height_out_border = _getBorderTileSize(height_out, 3)
        width_out_border = _getBorderTileSize(width_out, 3)
        height_in_border = height_out_border + 2 - padding_bottom
        width_in_border = width_out_border + 2 - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int, weight_bits: int = 8) -> Tuple[int, int, int]:
        # Match ne16_task_set_strides for dense 3x3 (non-DW):
        #   d0 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride = 18
        #   d1 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride * qw * num_k_in
        #      = 18 * 8 * num_k_in
        _NE16_FILTER_SIZE = 3
        _NE16_WEIGHT_D0_STRIDE_MODE8 = 2
        _QW = 8
        n_channel_in = _getNumTiles(channel_in, 16)
        d0 = _NE16_FILTER_SIZE * _NE16_FILTER_SIZE * _NE16_WEIGHT_D0_STRIDE_MODE8
        d1 = d0 * _QW * n_channel_in
        return d0, d1, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


NE16TaskInitTemplateStr = """
// N-EUREKA Task Init
ne16_task_t task = {
    .data = (ne16_task_data_t) {
        .weights_addr = (uint32_t)${weight} - ${wmem_addr_offset} + ${weight_addr_offset},
        .infeat_addr = (uint32_t)${data_in} - ${input_addr_offset},
        .outfeat_addr = (uint32_t)${data_out},
        .scale_addr = (uint32_t)${mul},
        .scale_shift_addr = (uint32_t)${shift},
        .scale_bias_addr = (uint32_t)${add},
        .cfg = (ne16_cfg_t) {
            .input_stride = (ne16_stride_t) {
                .d0 = ${dim_im_in_y_stride},
                .d1 = ${dim_im_in_x_stride},
                .d2 = 0
            },
            .output_stride = (ne16_stride_t) {
                .d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                .d1 = ${dim_im_out_y_stride},
                .d2 = ${dim_im_out_x_stride}
            },
            task.data.cfg.weights_stride = (ne16_stride_t) {
                .d0 = ${weightStrideD0},
                .d1 = ${weightStrideD1},
                .d2 = ${weightStrideD2}
            },
            .subtile = (ne16_subtile_t) {
                .number = {
                    .KoKi = nnx_concat_half(${nKo}, ${nKi}),
                    .HoWo = nnx_concat_half(${nHo}, ${nWo})
                },
                .remainder = {
                    .KoKi = nnx_concat_half(${bKo}, ${bKi}),
                    .HoWo = nnx_concat_half(${bHo}, ${bWo}),
                    .HiWi = nnx_concat_half(${bHi}, ${bWi})
                }
            },
            .padding = (${padding_y_top} << 28) + (${padding_x_right} << 24) + (${padding_y_bottom} << 20) + (${padding_x_left} << 16),
            .weight_offset_factor = ${weight_offset},
            .filter_mask = 0,
            .conf0 = ${conf0},
        }
    }
};
// NE16 top-level task struct fields (required by HAL helpers and NE16 HW for
// non-1x1 paths). Kept consistent with ne16_task_set_op_to_conv/_set_bits.
task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
task.qw = ${weight_bits};
task.subtile_output_channel = ${ne16_subtile_output_channel};
task.kernel_shape = ${ne16_kernel_shape};
task.depthwise = ${ne16_depthwise};
"""

NE16TaskExecutionTemplateStr = """
// N-EUREKA Task Execution
ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
"""

# Pointwise execution. On the GAP9 EVK a 1x1 job whose H or W spans several 3-wide subtiles with a partial last one
# corrupts outputs (gvsoc does not model it), so every job must have H and W each a multiple of 3 or < 3. For a flat
# pixel column the npix contiguous pixels of the tile are re-viewed as jobs of that shape:
#   npix >= 9 : (3*(npix/9), 3) at pixel 0, plus for the npix%9 tail one (ceil(tail/3), 3) job ending at pixel npix
#               (it may overlap the first job; overlapped pixels are recomputed with identical inputs, so identical
#               outputs are written twice);
#   3..8      : (npix/3, 3) at pixel 0, plus a (1, 3) job ending at npix if npix%3;
#   < 3       : (npix, 1).
# That is ceil(npix/9) 3x3 subtiles per Ko/Ki step (optimal) for npix >= 9, independently of how M factors.
NE16PWTaskExecutionTemplateStr = """
% if pw_flat:
// N-EUREKA Task Execution (flat pixel column -> silicon-legal jobs)
{
    const uint32_t _npix = ${pw_npix};
    const uint32_t _in_px = ${dim_im_in_y_stride};
    const uint32_t _out_px = ${dim_im_out_y_stride};
    uint32_t _h0, _w0, _h1 = 0, _off1 = 0;
    if (_npix >= 9) {
        const uint32_t _tail = _npix % 9;
        _h0 = 3 * (_npix / 9); _w0 = 3;
        if (_tail) { _h1 = (_tail + 2) / 3; _off1 = _npix - 3 * _h1; }
    } else if (_npix >= 3) {
        _h0 = _npix / 3; _w0 = 3;
        if (_npix % 3) { _h1 = 1; _off1 = _npix - 3; }
    } else {
        _h0 = _npix; _w0 = 1;
    }
    {
        const uint16_t _nh = (_h0 + 2) / 3, _bh = _h0 - 3 * (_nh - 1);
        const uint16_t _nw = (_w0 + 2) / 3, _bw = _w0 - 3 * (_nw - 1);
        task.data.cfg.subtile.number.HoWo = nnx_concat_half(_nh, _nw);
        task.data.cfg.subtile.remainder.HoWo = nnx_concat_half(_bh, _bw);
        task.data.cfg.subtile.remainder.HiWi = nnx_concat_half(_bh, _bw);
        task.data.cfg.input_stride.d1 = _w0 * _in_px;
        task.data.cfg.output_stride.d2 = _w0 * _out_px;
    }
    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
    if (_h1) {
        // tail job (_h1 <= 3, W = 3)
        task.data.infeat_addr += _off1 * _in_px;
        task.data.outfeat_addr += _off1 * _out_px;
        task.data.cfg.subtile.number.HoWo = nnx_concat_half(1, 1);
        task.data.cfg.subtile.remainder.HoWo = nnx_concat_half(_h1, 3);
        task.data.cfg.subtile.remainder.HiWi = nnx_concat_half(_h1, 3);
        task.data.cfg.input_stride.d1 = 3 * _in_px;
        task.data.cfg.output_stride.d2 = 3 * _out_px;
        ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
        ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
    }
    ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
}
% else:
""" + NE16TaskExecutionTemplateStr + """
% endif
"""

NE16RqntPWConv2D_Template = NE162DPWConvTemplate(NE16TaskInitTemplateStr + NE16PWTaskExecutionTemplateStr)
NE16PWConv2D_Template = NE162DPWConvTemplate(NE16TaskInitTemplateStr + NE16PWTaskExecutionTemplateStr)

NE16RqntDWConv2D_Template = NE162DDWConvTemplate(NE16TaskInitTemplateStr + NE16TaskExecutionTemplateStr)
NE16DWConv2D_Template = NE162DDWConvTemplate(NE16TaskInitTemplateStr + NE16TaskExecutionTemplateStr)

NE16RqntDenseConv2D_Template = NE162DDenseConvTemplate(NE16TaskInitTemplateStr + NE16TaskExecutionTemplateStr)
NE16DenseConv2D_Template = NE162DDenseConvTemplate(NE16TaskInitTemplateStr + NE16TaskExecutionTemplateStr)
