# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NodeMapper, ONNXLayer, Shape
from Deeploy.Targets.Generic.Layers import RQGEMMLayer, RQSConvLayer


class PULPRQSConvLayer(RQSConvLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if channels_first:
            inputShapes[2] = [outputShapes[0][1]]  # Channels out dimension of Kernel
            inputShapes[3] = [outputShapes[0][1]]  # Channels out dimension of Kernel
        else:
            inputShapes[2] = [outputShapes[0][-1]]  # Channels out dimension of Kernel
            inputShapes[3] = [outputShapes[0][-1]]  # Channels out dimension of Kernel
        return (inputShapes, outputShapes)


class PULPRQSGEMMLayer(RQGEMMLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:

        if operatorRepresentation['transB']:
            channelDim = -2
        else:
            channelDim = -1

        inputShapes[2] = [inputShapes[1][channelDim]]  # Channels out dimension of Kernel
        inputShapes[3] = [inputShapes[1][channelDim]]  # Channels out dimension of Kernel

        return (inputShapes, outputShapes)


class PULPSoftplusLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        # Elementwise: output shape equals input shape
        return (inputShapes, outputShapes)

    def computeOps(self):
        # LUT-based: only memory loads/stores, no arithmetic ops
        return 0


class PULPSelectiveScanLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        B = self.mapper.parser.operatorRepresentation['batch_size']
        L = self.mapper.parser.operatorRepresentation['seq_len']
        D = self.mapper.parser.operatorRepresentation['d_inner']
        N = self.mapper.parser.operatorRepresentation['d_state']

        # Per-N ops (b,t,d): counts tagged inline.
        ops_mul_per_n = 5  # dt*A, dt*B, dA*h, dB*x, h*C
        ops_add_per_n = 3  # exp_idx rounding, h += dB*x, y_acc += ...
        ops_shift_per_n = 5  # >>8, >>LOG2, >>8, >>15, >>15
        ops_per_n = ops_mul_per_n + ops_add_per_n + ops_shift_per_n  # 13

        # Post-N ops (b,t,d): D_skip, gate, requant.
        ops_per_td_post = 3 + 3 + 2

        return B * L * D * (ops_per_n * N + ops_per_td_post)


class PULPSelectiveScanI16Layer(PULPSelectiveScanLayer):
    pass


class PULPSSDScanLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        B = self.mapper.parser.operatorRepresentation['batch_size']
        L = self.mapper.parser.operatorRepresentation['seq_len']
        H = self.mapper.parser.operatorRepresentation['n_heads']
        P = self.mapper.parser.operatorRepresentation['head_dim']
        N = self.mapper.parser.operatorRepresentation['d_state']
        Chunk = self.mapper.parser.operatorRepresentation['chunk_size']

        numChunks = L // Chunk
        # Triangular count of within-chunk causal pairs (Y_diag loop).
        triangular = Chunk * (Chunk + 1) // 2

        # (1) Discretization loop over the chunk
        mul_1 = Chunk * (1 + N)
        add_1 = Chunk
        shift_1 = Chunk * (1 + N)

        # (2a) Y_diag double loop
        mul_2a = triangular * (N + 1 + P)
        add_2a = triangular * (N + P)
        shift_2a = triangular * 2

        # (2b) Y_off + output requant, once per (query, head-dim) pair:
        mul_2b = Chunk * P * (N + 3)
        add_2b = Chunk * P * (N + 3)
        shift_2b = Chunk * P * 3

        # (3a) Chunk-state decay rescale, over Chunk key steps x N state dims:
        mul_3a = Chunk * N
        shift_3a = Chunk * N

        # (3b) Chunk-state recurrence, over P head-dims x N state dims:
        mul_3b = P * N * (1 + Chunk)
        add_3b = P * N * Chunk
        shift_3b = P * N

        opsPerChunk = (mul_1 + add_1 + shift_1 + mul_2a + add_2a + shift_2a + mul_2b + add_2b + shift_2b + mul_3a +
                      shift_3a + mul_3b + add_3b + shift_3b)

        return B * H * numChunks * opsPerChunk


class PULPSSDScanNE16Layer(PULPSSDScanLayer):
    pass


class PULPMamba3ScanLayer(PULPSSDScanLayer):

    def computeOps(self):
        # SSD's count with the rank folded in: every dot product over N becomes one over N*R
        # and the per-(query, feature) work is repeated per rank. RoPE adds 4 multiplies per
        # state pair for B and for C per step; trapezoid adds one multiply-add per key.
        rep = self.mapper.parser.operatorRepresentation
        R = rep.get('mimo_rank', 1)
        B, L, H, P, N, Chunk = (rep['batch_size'], rep['seq_len'], rep['n_heads'], rep['head_dim'], rep['d_state'],
                                rep['chunk_size'])
        numChunks = L // Chunk
        tri = Chunk * (Chunk + 1) // 2
        NR = N * R
        per_chunk = (Chunk * (2 + NR)                      # phase 1: w_k, rotate, weight
                     + Chunk * 2 * N                       # rope on B and C rows (4 mul / pair)
                     + tri * (NR + 1)                      # scores
                     + Chunk * P * R * (tri // Chunk + N + 3)   # y_diag, readout, gate, requant
                     + Chunk * NR                          # phase 4 scaling
                     + P * N * Chunk * R)                  # state fold
        return B * H * numChunks * per_chunk


class PULPStaticScanLayer(ONNXLayer):
    """StaticScan_NE16: per-head causal LTI filter over the window (lower-triangular L x L weights per head)"""

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        rep = self.mapper.parser.operatorRepresentation
        L, P, H = rep['seq_len'], rep['head_dim'], rep['n_heads']
        return 2 * H * P * (L * (L + 1) // 2) + 6 * H * P * L


class PULPM3GatesLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)
