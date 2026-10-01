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


class PULPSelectiveScanI16Checker(PULPSelectiveScanChecker):
    pass


class PULPSSDScanChecker(SignPropTypeChecker):

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


class PULPMamba3ScanChecker(PULPSSDScanChecker):
    # Same contract as SSDScan: int8 in, int8 out, signedness from x. The two extra int16
    # inputs (lam, theta) do not affect the output type.
    pass
