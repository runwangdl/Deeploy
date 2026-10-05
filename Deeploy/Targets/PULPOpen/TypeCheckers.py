# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, Type

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import OperatorRepresentation, VariableBuffer


class PULPDMASliceChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class PULPRQAddChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['rqsOut_n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["rqsOut_signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if operatorRepresentation['rqsOut_signed'] and outputTypeSigned:
            return True
        if (not operatorRepresentation['rqsOut_signed']) and (not outputTypeSigned):
            return True
        return False


class PULPRequantShiftChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [operatorRepresentation["signed"]]

    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if operatorRepresentation['signed'] and outputTypeSigned:
            return True
        if (not operatorRepresentation['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPConvChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if operatorRepresentation['signed'] and outputTypeSigned:
            return True
        if (not operatorRepresentation['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPLinearChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if operatorRepresentation['signed'] and outputTypeSigned:
            return True
        if (not operatorRepresentation['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPMaxPoolChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        return True


class PULPSoftplusChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs, operatorRepresentation):
        # Compact256 softplus: int32 Q20 -> int16 Q8.8, range [0, 32767]
        if inputs[0].nLevels > 65536:
            return [32768]
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs, operatorRepresentation):
        # Softplus output is always non-negative
        if inputs[0].nLevels > 65536:
            return [False]
        return [inputs[0]._signed]

    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        return True


class PULPSelectiveScanChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:

        return True


class _WideOutMixin:
    """Scan ops with an optional wide output (attr out_bits = 16/32 -> int32 tensor): the binding whose output
    width matches out_bits is the one that type-checks, and the int32 output gets 2^32 levels."""

    def _wantWidth(self, operatorRepresentation):
        b = int(operatorRepresentation.get('out_bits', 8))
        return 8 if b <= 8 else (16 if b <= 16 else 32)

    def _inferNumLevels(self, inputs, operatorRepresentation):
        w = self._wantWidth(operatorRepresentation)
        return [inputs[0].nLevels] if w == 8 else [2**w]

    def checkOutputType(self, inputs, operatorRepresentation) -> bool:
        return self.output_types[0].referencedType.typeWidth == self._wantWidth(operatorRepresentation)


class PULPSelectiveScanI16Checker(_WideOutMixin, PULPSelectiveScanChecker):
    pass


class PULPSSDScanChecker(_WideOutMixin, SignPropTypeChecker):

    def typeCheck(self, ctxt, node, operatorRepresentation):
        # SSD_Scan_NE16 has a 7-input and a 9-input (two-scale) form: a binding only matches its own arity
        # (the input check zips inputs with input_types and would otherwise ignore the extra inputs)
        if len(node.inputs) != len(self.input_types):
            return ctxt, False
        return super().typeCheck(ctxt, node, operatorRepresentation)


    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class PULPMamba3ScanChecker(PULPSSDScanChecker):
    # Same contract as SSDScan: int8 in, int8 out, signedness from x. The two extra int16
    # inputs (lam, theta) do not affect the output type.
    pass


class PULPRMSNormI32Checker(SignPropTypeChecker):
    """RMSNormI32: int32 in (any scale), int8 out with 256 signed levels"""

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [256]

    def _inferSignedness(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class PULPStaticScanChecker(SignPropTypeChecker):
    """StaticScan_NE16: int32 output (consumed by RMSNormI32), 2^32 signed levels"""

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**32]

    def _inferSignedness(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]
