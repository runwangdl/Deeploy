# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int32_t
from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import Conv2DParser, GEMMParser, ReduceMeanParser, RQSConv1DParser, \
    RQSConv2DParser, RQSParserInterface


class PULPConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                len(node.inputs) == 4,
                'shift' in node.attrs,
            ])

            self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
            self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
            self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Current PULP kernel only supports grouping of 1
                self.operatorRepresentation['group'] == 1,

                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Set inputs names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPFPDWConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        # Parse root conv 2D information
        wellFormed = super().parseNode(node)

        if wellFormed:
            # Check if the node is a depthwise convolution
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],

                # Check number of inputs
                # 2 inputs if no bias, 3 if layer has bias
                len(node.inputs) in [2, 3],
            ])

            # Extract additional attributes
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
            self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # Parse node context for 2D conv
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Define input names
            inputs = ['data_in', 'weight']

            # Handle bias, if present
            if len(node.inputs) == 2:
                self.operatorRepresentation["has_bias"] = "false"
                self.operatorRepresentation["bias"] = "NULL"
            else:
                inputs.append("bias")
                self.operatorRepresentation["has_bias"] = "true"

            # Map input nodes to operator representation
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            # Check if DW
            if self.operatorRepresentation['group'] == self.operatorRepresentation['ch_im_in']:
                return newCtxt, True

        return ctxt, False


class PULPDWConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                #self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            if ret:

                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)

                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            # Input stays NCHW (only output is transposed); override base-parser dim reads accordingly.
            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
            self.operatorRepresentation['dim_im_in_y'] = data_in.shape[2]
            if channels_first:
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]
            else:
                self.operatorRepresentation['ch_im_out'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[1]

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            return newCtxt, True

        return ctxt, False


class PULPDWConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
                self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
                'shift' in node.attrs,
                any(['n_levels' in node.attrs, 'n_levels_out' in node.attrs]),
                'signed' in node.attrs
            ])

            if ret:
                self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
                self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
                self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
                self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
                self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
                self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][1])
                self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][2])
                self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][3])
                self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
                self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])

                if 'n_levels' in node.attrs:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.operatorRepresentation['n_levels'] = int(node.attrs['n_levels_out'].values)
                self.operatorRepresentation['signed'] = int(node.attrs['signed'].values)
                self.operatorRepresentation['log2D'] = int(math.log2(node.attrs['div'].values))

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.operatorRepresentation['group'] == newCtxt.lookup(
                    self.operatorRepresentation['weight']).shape[0]:
                return ctxt, False

            data_in = newCtxt.lookup(self.operatorRepresentation['data_in'])
            data_out = newCtxt.lookup(self.operatorRepresentation['data_out'])
            _ = newCtxt.lookup(self.operatorRepresentation['weight'])

            # if not newCtxt.is_global(self.operatorRepresentation['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.operatorRepresentation['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            if channels_first:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[2]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[3]
            else:
                self.operatorRepresentation['ch_im_in'] = data_in.shape[1]
                self.operatorRepresentation['dim_im_in_x'] = data_in.shape[2]
                self.operatorRepresentation['dim_im_in_y'] = data_in.shape[3]
                self.operatorRepresentation['ch_im_out'] = data_out.shape[3]
                self.operatorRepresentation['dim_im_out_x'] = data_out.shape[1]
                self.operatorRepresentation['dim_im_out_y'] = data_out.shape[2]

            return newCtxt, True

        return ctxt, False


class PULPConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.operatorRepresentation['group'] == 1,
                self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
                #self.operatorRepresentation['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
                len(node.inputs) == 4,
            ])

            self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
            self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
            self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][0])
            self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][1])
            self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPGEMMParser(GEMMParser, RQSParserInterface):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> (bool):

        ret_rqs = RQSParserInterface.parseNode(self, node)
        ret_matmul = GEMMParser.parseNode(self, node)

        ret = all([
            ret_rqs == True,
            ret_matmul == True,
            'shift' in node.attrs,
            len(node.inputs) == 4,
        ])

        if ret:
            self.operatorRepresentation['shift'] = int(node.attrs['shift'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B', 'C', 'mul']
            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            return newCtxt, True

        else:
            return ctxt, False


class PULPMatrixVecParser(PULPGEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not (self.operatorRepresentation['M'] == 1 and self.operatorRepresentation['batch'] >= 8):
            return ctxt, False

        return newCtxt, True


class PULPTallGEMMParser(PULPGEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        ret = all([
            self.operatorRepresentation['batch'] < 8,
            self.operatorRepresentation['M'] >= 8,
            self.operatorRepresentation['M'] % 8 < self.operatorRepresentation['O'] % 8,
        ])

        if not ret:
            return ctxt, False

        return newCtxt, True


class PULPReduceMeanParser(ReduceMeanParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # Inherit the generic ReduceMean parsing
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # Add to operator representation the non-reduced dimensions for tiling purposes
            originalInputShape = newCtxt.lookup(self.operatorRepresentation['data_in']).shape
            reducedAxes = self.operatorRepresentation['axes']

            for ax in range(len(originalInputShape)):
                if ax not in reducedAxes:
                    self.operatorRepresentation['dim_in_' + str(ax)] = originalInputShape[ax]

            return newCtxt, True
        else:
            return ctxt, False


class PULPSoftplusParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            node.op == 'Softplus',
            len(node.inputs) == 1,
            len(node.outputs) == 1,
        ])
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))

        return ctxt, True


class PULPSelectiveScanParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            node.op == 'SelectiveScan',
            len(node.inputs) == 7,
            len(node.outputs) == 1,
        ])
        if ret:
            self.operatorRepresentation['batch_size'] = int(node.attrs['batch'])
            if 'seq_len' in node.attrs:
                self.operatorRepresentation['seq_len'] = int(node.attrs['seq_len'])
            if 'd_inner' in node.attrs:
                self.operatorRepresentation['d_inner'] = int(node.attrs['d_inner'])
            if 'd_state' in node.attrs:
                self.operatorRepresentation['d_state'] = int(node.attrs['d_state'])
            if 'output_requant_mul_q40' in node.attrs:
                self.operatorRepresentation['output_requant_mul_q40'] = int(node.attrs['output_requant_mul_q40'])
            if 'gate_z_scale' in node.attrs:
                self.operatorRepresentation['gate_z_scale'] = float(node.attrs['gate_z_scale'])

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # ONNX input order matches [x, z, dt, B, C, A, D_skip]
            inputs = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip']
            outputs = ['y']

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            # Precompute 256-entry SiLU gate LUT (int32 Q13) at compile time; stored as L2 ConstantBuffer.
            gate_z_scale = self.operatorRepresentation['gate_z_scale']
            lut_name = node.name + '_gate_lut'

            if lut_name not in newCtxt.globalObjects:
                indices = np.arange(256, dtype=np.float64)
                z_d     = (indices - 128.0) * gate_z_scale
                z_clip  = np.clip(z_d, -20.0, 20.0)
                sig     = 1.0 / (1.0 + np.exp(-z_clip))
                q20     = np.round(z_d * sig * float(1 << 20)).astype(np.int64)
                q13     = np.where(q20 >= 0, (q20 + 64) >> 7, -((-q20 + 64) >> 7))
                lut_values = q13.astype(np.int32)

                gate_lut_buf = newCtxt.ConstantBuffer(lut_name, [256], lut_values)
                newCtxt.add(gate_lut_buf, ctxt='global')
                newCtxt.annotateType(lut_name, PointerClass(int32_t))
                gate_lut_buf._memoryLevel = "L2"

            self.operatorRepresentation['gate_lut'] = lut_name

            return newCtxt, True
        else:
            return ctxt, False


class PULPSSDScanParser(NodeParser):

    OP_NAME = 'SSD_Scan'

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            node.op == self.OP_NAME,
            len(node.inputs) == 7,
            len(node.outputs) == 1,
            'n_groups' in node.attrs and int(node.attrs['n_groups']) == 1,
        ])
        if ret:
            self.operatorRepresentation['batch_size'] = int(node.attrs['batch_size'])
            if 'seq_len' in node.attrs:
                self.operatorRepresentation['seq_len'] = int(node.attrs['seq_len'])
            if 'chunk_size' in node.attrs:
                self.operatorRepresentation['chunk_size'] = int(node.attrs['chunk_size'])
            if 'd_state' in node.attrs:
                self.operatorRepresentation['d_state'] = int(node.attrs['d_state'])
            if 'head_dim' in node.attrs:
                self.operatorRepresentation['head_dim'] = int(node.attrs['head_dim'])
            if 'n_groups' in node.attrs:
                self.operatorRepresentation['n_groups'] = int(node.attrs['n_groups'])
            if 'n_heads' in node.attrs:
                self.operatorRepresentation['n_heads'] = int(node.attrs['n_heads'])

            # d_inner = n_heads * head_dim (matches x/y's last dimension)
            if 'head_dim' in self.operatorRepresentation and 'n_heads' in self.operatorRepresentation:
                self.operatorRepresentation['d_inner'] = (self.operatorRepresentation['head_dim'] *
                                                          self.operatorRepresentation['n_heads'])

            if 'gate_z_scale' in node.attrs:
                self.operatorRepresentation['gate_z_scale'] = float(node.attrs['gate_z_scale'])
            self.operatorRepresentation['output_requant_mul_q40'] = int(node.attrs['output_requant_mul_q40'])
            self.operatorRepresentation['epilogue_version'] = int(node.attrs.get('epilogue_version', 1))
            self.operatorRepresentation['out_bits'] = int(node.attrs.get('out_bits', 8))
            self.operatorRepresentation['out_shift'] = int(node.attrs.get('out_shift', 0))

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            # ONNX input order matches [x, z, dt, B, C, A, D_skip]
            inputs = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip']
            outputs = ['y']

            for idx, inputNode in enumerate(node.inputs):
                self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            # Precompute 256-entry SiLU gate LUT (int32 Q13) at compile time; stored as L2 ConstantBuffer.
            gate_z_scale = self.operatorRepresentation['gate_z_scale']
            lut_name = node.name + '_gate_lut'

            if lut_name not in newCtxt.globalObjects:
                indices = np.arange(256, dtype=np.float64)
                z_d = (indices - 128.0) * gate_z_scale
                z_clip = np.clip(z_d, -20.0, 20.0)
                sig = 1.0 / (1.0 + np.exp(-z_clip))
                q20 = np.round(z_d * sig * float(1 << 20)).astype(np.int64)
                q13 = np.where(q20 >= 0, (q20 + 64) >> 7, -((-q20 + 64) >> 7))
                lut_values = q13.astype(np.int32)

                gate_lut_buf = newCtxt.ConstantBuffer(lut_name, [256], lut_values)
                newCtxt.add(gate_lut_buf, ctxt='global')
                newCtxt.annotateType(lut_name, PointerClass(int32_t))
                gate_lut_buf._memoryLevel = "L2"

            self.operatorRepresentation['gate_lut'] = lut_name

            return newCtxt, True
        else:
            return ctxt, False


class PULPSSDScanNE16Parser(PULPSSDScanParser):
    """SSD_Scan_NE16: same inputs/attributes as SSD_Scan; the three per-head chunk products run on the
    NE16 (per-head int8 weights, power-of-two scales), see TargetLibraries/GAP9/src/SSDScanNE16.c."""
    OP_NAME = 'SSD_Scan_NE16'


class PULPMamba3ScanParser(PULPSSDScanParser):
    """Mamba3_Scan: SSD_Scan's seven inputs plus gamma, w, theta (int16), and a mimo_rank
    attribute. gamma/w are the trapezoid per-key weights precomputed upstream (w needs
    dt_{t+1}, which an L-tiled kernel cannot see). d_inner = n_heads * head_dim * mimo_rank."""

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            node.op == 'Mamba3_Scan',
            len(node.inputs) == 10,
            len(node.outputs) == 1,
            'n_groups' in node.attrs and int(node.attrs['n_groups']) == 1,
        ])
        if not ret:
            return False
        for key in ('batch_size', 'seq_len', 'chunk_size', 'd_state', 'head_dim', 'n_groups', 'n_heads'):
            if key in node.attrs:
                self.operatorRepresentation[key] = int(node.attrs[key])
        self.operatorRepresentation['mimo_rank'] = int(node.attrs.get('mimo_rank', 1))
        self.operatorRepresentation['d_inner'] = (self.operatorRepresentation['head_dim'] *
                                                  self.operatorRepresentation['n_heads'] *
                                                  self.operatorRepresentation['mimo_rank'])
        if 'gate_z_scale' in node.attrs:
            self.operatorRepresentation['gate_z_scale'] = float(node.attrs['gate_z_scale'])
        self.operatorRepresentation['output_requant_mul_q40'] = int(node.attrs['output_requant_mul_q40'])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        # NodeParser.parseNodeCtxt, not the SSD one: it would try to map the first 7 inputs only.
        newCtxt, ret = NodeParser.parseNodeCtxt(self, ctxt, node, channels_first)
        if not ret:
            return ctxt, False
        inputs = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip', 'gamma', 'w', 'theta']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
        self.operatorRepresentation['y'] = newCtxt.lookup(node.outputs[0].name).name

        gate_z_scale = self.operatorRepresentation['gate_z_scale']
        lut_name = node.name + '_gate_lut'
        if lut_name not in newCtxt.globalObjects:
            indices = np.arange(256, dtype = np.float64)
            z_d = (indices - 128.0) * gate_z_scale
            z_clip = np.clip(z_d, -20.0, 20.0)
            sig = 1.0 / (1.0 + np.exp(-z_clip))
            q20 = np.round(z_d * sig * float(1 << 20)).astype(np.int64)
            q13 = np.where(q20 >= 0, (q20 + 64) >> 7, -((-q20 + 64) >> 7))
            gate_lut_buf = newCtxt.ConstantBuffer(lut_name, [256], q13.astype(np.int32))
            newCtxt.add(gate_lut_buf, ctxt = 'global')
            newCtxt.annotateType(lut_name, PointerClass(int32_t))
            gate_lut_buf._memoryLevel = "L2"
        self.operatorRepresentation['gate_lut'] = lut_name
        return newCtxt, True


class PULPSelectiveScanI16Parser(PULPSelectiveScanParser):
    """SelectiveScanI16: int16 recurrent state with per-channel power-of-two shifts.
    inputs: x, z, dt, B, C, A16 (int16 [D,N]), D_skip, shA (int8 [D] = 8 - sA), sH (uint8 [D]), ysh (uint8 [D] = sH + bc_shift)
    attrs:  batch, seq_len, d_inner, d_state, output_requant_mul_q40, gate_z_scale, bc_shift"""

    def parseNode(self, node: gs.Node) -> bool:
        ret = all([
            node.op == 'SelectiveScanI16',
            len(node.inputs) == 10,
            len(node.outputs) == 1,
        ])
        if ret:
            self.operatorRepresentation['batch_size'] = int(node.attrs['batch'])
            for k in ('seq_len', 'd_inner', 'd_state', 'output_requant_mul_q40', 'bc_shift'):
                if k in node.attrs:
                    self.operatorRepresentation[k] = int(node.attrs[k])
            self.operatorRepresentation.setdefault('bc_shift', 6)
            if 'gate_z_scale' in node.attrs:
                self.operatorRepresentation['gate_z_scale'] = float(node.attrs['gate_z_scale'])
            self.operatorRepresentation['out_bits'] = int(node.attrs.get('out_bits', 8))
            self.operatorRepresentation['out_shift'] = int(node.attrs.get('out_shift', 0))
        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = NodeParser.parseNodeCtxt(self, ctxt, node, channels_first)
        if not ret:
            return ctxt, False
        inputs = ['x', 'z', 'dt', 'B', 'C', 'A', 'D_skip', 'shA', 'sH', 'ysh']
        for idx, inputNode in enumerate(node.inputs):
            self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
        self.operatorRepresentation['y'] = newCtxt.lookup(node.outputs[0].name).name
        gate_z_scale = self.operatorRepresentation['gate_z_scale']
        lut_name = node.name + '_gate_lut'
        if lut_name not in newCtxt.globalObjects:
            indices = np.arange(256, dtype = np.float64)
            z_d = (indices - 128.0) * gate_z_scale
            z_clip = np.clip(z_d, -20.0, 20.0)
            sig = 1.0 / (1.0 + np.exp(-z_clip))
            q20 = np.round(z_d * sig * float(1 << 20)).astype(np.int64)
            q13 = np.where(q20 >= 0, (q20 + 64) >> 7, -((-q20 + 64) >> 7))
            gate_lut_buf = newCtxt.ConstantBuffer(lut_name, [256], q13.astype(np.int32))
            newCtxt.add(gate_lut_buf, ctxt = 'global')
            newCtxt.annotateType(lut_name, PointerClass(int32_t))
            gate_lut_buf._memoryLevel = "L2"
        self.operatorRepresentation['gate_lut'] = lut_name
        return newCtxt, True


class PULPRMSNormI32Parser(NodeParser):
    """RMSNormI32: int32 in (any scale), int32 per-channel weight, int8 out; attr out_shift. Row-wise over the
    last dimension. Kernel PULP_RMSNormI32_s32_s8 (TargetLibraries/PULPOpen/src/RMSNormI32.c)."""

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        ret = node.op == 'RMSNormI32' and len(node.inputs) == 2 and len(node.outputs) == 1 and 'out_shift' in node.attrs
        if ret:
            self.operatorRepresentation['out_shift'] = int(node.attrs['out_shift'])
        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        for idx, name in enumerate(['data_in', 'weight']):
            self.operatorRepresentation[name] = ctxt.lookup(node.inputs[idx].name).name
        self.operatorRepresentation['data_out'] = ctxt.lookup(node.outputs[0].name).name
        shape = list(ctxt.lookup(node.inputs[0].name).shape)
        self.operatorRepresentation['inputSize'] = int(np.prod(shape))
        self.operatorRepresentation['lastDimLength'] = int(shape[-1])
        return ctxt, True
