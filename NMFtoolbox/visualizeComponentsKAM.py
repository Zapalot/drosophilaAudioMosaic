"""
    Name: visualizeComponentsNMF
    Date: Jul 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    []  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
import matplotlib.pyplot as plt

from NMFtoolbox.logFreqLogMag import logFreqLogMag
from NMFtoolbox.utils import EPS
from NMFtoolbox.coloredComponents import coloredComponents


def visualizeComponentsKAM(compA, parameter=None):
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components,
    this function provides a visualization.

    Parameters
    ----------
    compA: array-like
        List with R individual component magnitude spectrograms

    parameter: dict
        deltaT        Temporal resolution
        deltaF        Spectral resolution
        startSec      Where to zoom in on the time axis
        endeSec       Where to zoom in on the time axis

    Returns
    -------
    fh: The figure handle
    """
    # check parameters

    # get spectrogram dimensions
    numLinBins, numFrames = compA[0].shape

    parameter = init_params(parameter, numFrames)

    # plot MMF / NMFD components
    # map template spectrograms to a logarithmically - spaced frequency
    # and logarithmic magnitude compression

    logFreqLogMagCompA, logFreqAxis = logFreqLogMag(compA, parameter['deltaF'])
    numLogBins = len(logFreqAxis)

    timeAxis = np.arange(numFrames) * parameter['deltaT']
    freqAxis = np.arange(numLogBins)

    # subsample freq axis
    subSampFreqAxis = logFreqAxis[np.mod(logFreqAxis, 55) < 0.001]

    # further plot params
    plotHeight = 0.4  # TODO: Ask Christian
    setFontSize = 11 if 'fontSize' not in parameter else parameter['fontSize']  # TODO: Ask Christian whether it isn't too big

    startSec = parameter['startSec']
    endeSec = parameter['endeSec']

    # make new  figure
    fh, ax = plt.subplots(figsize=(15, 10))

    # plot the component spectrogram matrix
    ax.imshow(coloredComponents(logFreqLogMagCompA),
              origin='lower', aspect='auto', cmap='gray_r',
              extent=[timeAxis[0], timeAxis[-1], freqAxis[0], freqAxis[-1]])

    ax.set_title('A = A_p + A_h')
    ax.set_xlabel('Time in seconds')

    return fh


def init_params(parameter, numFrames):
    """Auxiliary function to set the parameter dictionary

       Parameters
       ----------
       parameter: dict
           See the above function visualizeComponentsKAM for further information

       Returns
       -------
       parameter: dict
    """
    parameter['startSec'] = 1 * parameter['deltaT'] if 'startSec' not in parameter else parameter['startSec']
    parameter['endeSec'] = numFrames * parameter['deltaT'] if 'endeSec' not in parameter else parameter['endeSec']

    return parameter
