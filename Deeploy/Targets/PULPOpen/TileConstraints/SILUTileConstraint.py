# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class SILUTileConstraint(UnaryTileConstraint):

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> Dict:
        return parseDict.copy()

