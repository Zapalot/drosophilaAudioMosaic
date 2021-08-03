"""
    Name: artificialMixtureMatrix
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

import numpy as np

from NMFtoolbox.utils import EPS


def artificialMixtureMatrix(numComp = 3, templateDim=3, maxOccurence=160):
    """
    Summary goes here? TODO
    :return: V
             TrueW
    """

    trueW = np.array([
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    ])

    V = np.zeros((templateDim * numComp, templateDim * maxOccurence))
    lastOnOrOff = 0

    for h in range(numComp):
        for k in range(maxOccurence):
            onOrOff = (np.random.rand() > 0.5) * 1

            if k == 1 or k == maxOccurence:
                onOrOff = 0

            if onOrOff == 1 and lastOnOrOff == 1:
                onOrOff = 0

            lastOnOrOff = onOrOff
            V[h * templateDim: (h+1) * templateDim,
              k * templateDim: (k+1) * templateDim] = onOrOff * trueW[h]

    V += EPS

    return V, trueW