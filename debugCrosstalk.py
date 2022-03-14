#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:02:03 2022

@author: melanogaster
"""

import os
import math
import time as tm
import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import scipy.signal as sig

from numpy.fft import fft, ifft
import matplotlib
import matplotlib.pyplot as plt

with open('test.npy', 'rb') as f:
    playedDisturbance = np.load(f)
    recordedSignal = np.load(f)
    denoisedSignalComplete = np.load(f)
    simulatedSignalComplete = np.load(f)
    impulseResponse = np.load(f)
	
fig, ax = plt.subplots()
ax.plot(playedDisturbance,color="pink")
ax.plot(recordedSignal[:,0],color="black")
ax.plot(recordedSignal[:,1],color="red")
ax.plot(simulatedSignalComplete[:,1],color="blue")
ax.plot(denoisedSignalComplete[:,1],color="green")