#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:55:30 2021

@author: fbo
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

import sounddevice as sd
import threading
from CrosstalkSuppression import CrosstalkSuppressor


subsampling=1 
input_device="Steinberg UR44"
output_device="Steinberg UR44"

samplerate=48000
soundIoBlockSize=512*subsampling
dtype='float32'
latency="high"
nTotalChannels=2		

#settings for crosstalk elemination
noiseSendChannel=0
noiseReceiveChannels=[0]
noiseImpulseLength=soundIoBlockSize*8

headphoneChannels=1


antiCrosstalk=CrosstalkSuppressor(soundIoBlockSize,noiseImpulseLength,len(noiseReceiveChannels))

# here some things for debugging
testSigLength=soundIoBlockSize
t = np.linspace(0, 1, testSigLength)
disturbanceSignal= np.sin(2 * np.pi * 10 * t)*0.2#
disturbanceSignal=sig.sawtooth(2 * np.pi * 3 * t,0.5)*0.2
targetSignal=np.sin(2 * np.pi * 10 * t)*0.2
nTestSignalBlocksLeft=200


playedDisturbance=np.zeros(0)
recordedSignal=np.zeros(0)
denoisedSignalComplete=np.zeros(0)
simulatedSignalComplete=np.zeros(0)


calibrationFinished=threading.Event()

#this callback is passed by the sounddeivce library
def callback(indata, outdata, frames, time, status):
	global playedDisturbance
	global recordedSignal
	global nTestSignalBlocksLeft
	global denoisedSignalComplete
	global simulatedSignalComplete
	if status:
		print(status)
	oldTime=tm.perf_counter() 
	speakerSignal,denoisedSignal,simulatedSignal=antiCrosstalk.process(disturbanceSignal,np.transpose( indata[:,noiseReceiveChannels]))

	outdata[:,noiseSendChannel] = speakerSignal
	outdata[:,headphoneChannels] = 0
	
	#debugging
	playedDisturbance=np.append(playedDisturbance,outdata[:,noiseSendChannel])
	recordedSignal=np.append(recordedSignal,indata[:,noiseReceiveChannels])
	denoisedSignalComplete=np.append(denoisedSignalComplete,denoisedSignal[noiseReceiveChannels,:])
	simulatedSignalComplete=np.append(simulatedSignalComplete,simulatedSignal[noiseReceiveChannels,:])
	
	if antiCrosstalk.isInitialized:
		if nTestSignalBlocksLeft>0:
			nTestSignalBlocksLeft-=1
		else:
			calibrationFinished.set()
			raise sd.CallbackStop

nImpulseMeas=1
savedImpulseLength=20000
impulses=np.zeros([nImpulseMeas,savedImpulseLength])
for i in range(0,nImpulseMeas):
	calibrationFinished.clear()
	antiCrosstalk.reset()
	nTestSignalBlocksLeft=200
	#run callback based loop until someone hits a key
	try:
		with sd.Stream(device=(input_device, output_device),
					   samplerate=samplerate, blocksize=soundIoBlockSize,
					   dtype=dtype, latency=latency,
					   channels=nTotalChannels, callback=callback):
			calibrationFinished.wait()
	except KeyboardInterrupt:
		print ('calibration aborted')	
	impulses[i,]=antiCrosstalk.impulseMeasurement.impulseResponse[0,0:savedImpulseLength]
	print( np.argmax(antiCrosstalk.impulseMeasurement.impulseResponse))

fig, ax = plt.subplots()
for i in range(0,nImpulseMeas):
	ax.plot(impulses[i,],color="blue")

		
############some result analysis
fig, ax = plt.subplots()
ax.plot(antiCrosstalk.impulseMeasurement.impulseResponse[0,],color="red")
#ax.plot(antiCrosstalk.impulseMeasurement.impulseResponse[1,],color="blue")
	
fig, ax = plt.subplots()
ax.plot(disturbanceSignal,color="blue")
ax.plot(targetSignal,color="blue")


processedSciPi=sig.convolve(antiCrosstalk.impulseMeasurement.impulseResponse[0,],playedDisturbance) 

sawtoothStart=antiCrosstalk.impulseMeasurement.impulseResponse[0,].shape[0]

displayRange=range(denoisedSignalComplete.shape[0]-1000,denoisedSignalComplete.shape[0])
#displayRange=range(0,20000)
fig, ax = plt.subplots()

#ax.plot(antiCrosstalk.impulseMeasurement.recordedSignal[0,displayRange],color="green")
ax.plot(recordedSignal[displayRange],color="blue")
#ax.plot(processedSciPi[displayRange],color="pink")
ax.plot(simulatedSignalComplete[displayRange],color="green")
#ax.plot(playedDisturbance[displayRange],color="green")
ax.plot(denoisedSignalComplete[displayRange],color="red")
#ax.plot(recordedSignal[displayRange]-processedSciPi[displayRange],color="green")

##Todo:
"""
- callback-Schleife unterbrechen können
- ggf Prozessierung in eigenen Thread?
- mal ein paar Testläufe und maximum der Impusantwort bestimmen

"""