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

########mosaicing
from realtimeMosaicing_Mel import *

import librosa 

#Parameters of Mosaicing
paramSTFT = dict()
paramSTFT['blockSize'] = 1024
paramSTFT['hopSize'] = 512
paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
paramSTFT['reconstMirror'] = True
paramSTFT['appendFrame'] = False
paramSTFT['numIterGriffinLim'] =3
paramSTFT['analyticSig'] = False
# parameters taken from Jonathan Driedger's toolbox
paramNMFdiag = dict()
paramNMFdiag['fixW'] = True
paramNMFdiag['numOfIter'] = 3
paramNMFdiag['continuity'] = dict()
paramNMFdiag['continuity']['polyphony'] = 1
paramNMFdiag['continuity']['length'] = 7
paramNMFdiag['continuity']['grid'] = 1
paramNMFdiag['continuity']['sparsen'] = [1, 7]

#!!! adjust volume here!!!
speakerOutputLoudnessMultiplier=0.15 #all output is multiplied by this

subsampling=2           # use only every N'th sample to save processing power
maxFlyTemplateSize=2048  # use only the first n mosaic grains to speed up NMFdiag
reverbFactorFFT=0.95;   # fraction of old loudness retained from last chunk --- 0: no reverb ... 0.9999: almost inifinite reverb

mosaicLoudnessBoost=100;			# loudness multiplier to compensate for losses due to reverb normalization
#initialize synthesizer with fly waveform
filenameFly = 'template/ZOOM0006_Tr12_excerpt.WAV'
#filenameFly = 'template/Exzerpt.wav'
melMatrixVoice= librosa.filters.mel(48000, paramSTFT['blockSize'],n_mels=32,fmax=10000)
melMatrixFly= librosa.filters.mel(48000, paramSTFT['blockSize'],n_mels=32,fmax=10000)
synthesizer=MosaicSynthesizer(paramSTFT, paramNMFdiag,melMatrixVoice,melMatrixFly,reverbFactorFFT,subsampling)
synthesizer.prepareFlyInformation(filenameFly,maxFlyTemplateSize)

reverbBuffer = np.zeros(paramSTFT['hopSize'])

########## parameters of sound i/o + crosstalk suppression
input_device="Steinberg UR44"
output_device="Steinberg UR44"

samplerate=48000
soundIoBlockSize=512*subsampling
dtype='float32'
latency="low"
nTotalChannels=3	

#settings for crosstalk elemination
drosophilaSpeakerChannel=2
drosophilaMikeChannels=[0,1]
drosophilaImpulseLength=soundIoBlockSize*2

headphoneSendChannels=[0,1]
humanVoiceMikeChannel=2
##########initialize crosstalk suppression
antiCrosstalk=CrosstalkSuppressor(soundIoBlockSize,drosophilaImpulseLength,len(drosophilaMikeChannels))
########## OSC spectrum transmission
from pythonosc import udp_client

import threading

oscTargetIp="192.168.5.100"
oscTargetPort=8000
oscFFTWindow=np.hamming(soundIoBlockSize)
oscClient=udp_client.SimpleUDPClient(oscTargetIp, oscTargetPort)
#self.impulseResponseReadyEvent=threading.Event()
			
def sendOscSpectra(speakerWaveForm, drosophilaWaveForm):
	drosophilaWaveForm*=oscFFTWindow
	drosophilaFFT=np.abs(np.fft.rfft(drosophilaWaveForm))
	
	speakerWaveForm*=oscFFTWindow
	speakerFFT=np.abs(np.fft.rfft(speakerWaveForm))
	oscClient.send_message("/fft/drosophila", list(drosophilaFFT))
	oscClient.send_message("/fft/human", list(speakerFFT))
	
#def sendOscSpectraThreadFun(speakerWaveForm, drosophilaWaveForm):

#all time records
playedDisturbance=np.zeros(0)
recordedSignal=np.zeros(0)
denoisedSignalComplete=np.zeros(0)
simulatedSignalComplete=np.zeros(0)


calibrationFinished=threading.Event()

#debugging
inCopy=0
outCopy=0
mosaicCopy=0
denoisedCopy=0
#give synthesizer data to initialize it self so it wont cause a buffer underrun when it does in the callback
for i in range(1,50):
	drosophilaSpeakerSignal=synthesizer.processAudioChunk( np.zeros(soundIoBlockSize))
	
#this callback is passed by the sounddeivce library
def callback(indata, outdata, frames, time, status):
	if status:
		print(status)
	startTime=tm.perf_counter() 
	if antiCrosstalk.isInitialized:
		drosophilaSpeakerSignal=synthesizer.processAudioChunk( indata[:,humanVoiceMikeChannel])	*mosaicLoudnessBoost
	else:
		drosophilaSpeakerSignal=indata[:,humanVoiceMikeChannel]
	synthTime=tm.perf_counter() 

	speakerSignal,denoisedSignal,simulatedSignal=antiCrosstalk.process(drosophilaSpeakerSignal,np.transpose( indata[:,drosophilaMikeChannels]))
	denoiseTime=tm.perf_counter() 
	#fft sending
	if antiCrosstalk.isInitialized:
		denoisedSum=np.sum(denoisedSignal,axis=0)
		try:
			sendOscSpectra( indata[:,humanVoiceMikeChannel],denoisedSum)
		except:
			pass
	oscTime=tm.perf_counter() 
	# sound output
	global outCopy
	global inCopy
	global mosaicCopy
	global denoisedCopy
	
	outCopy=outdata
	inCopy=indata
	denoisedCopy=denoisedSignal
	mosaicCopy=drosophilaSpeakerSignal

	outdata[:,drosophilaSpeakerChannel] = speakerSignal*speakerOutputLoudnessMultiplier #this might either be a test signal or the signal from mosaiicing
	outdata[:,headphoneSendChannels] = np.transpose(denoisedSignal)

	# performance report
	timeAvailableForOneBlock=soundIoBlockSize/samplerate
	timeSpentOnMosaic=synthTime-startTime
	timeSpentOnCrosstalk=denoiseTime-synthTime
	timeSpentOnOsc=oscTime-denoiseTime
	print ("mosaic:"+str(timeSpentOnMosaic/timeAvailableForOneBlock)+" crosstalk:"+str(timeSpentOnCrosstalk/timeAvailableForOneBlock)+" osc:"+str(timeSpentOnOsc/timeAvailableForOneBlock))   # how many times realtime performance do we achieve?
			


	

	#debug recording
	#global playedDisturbance
	#global recordedSignal
	#global nTestSignalBlocksLeft
	#global denoisedSignalComplete
	#global simulatedSignalComplete
	#playedDisturbance=np.append(playedDisturbance,outdata[:,drosophilaSpeakerChannel])
	#recordedSignal=np.append(recordedSignal,indata[:,drosophilaMikeChannels])
	#denoisedSignalComplete=np.append(denoisedSignalComplete,denoisedSignal[drosophilaMikeChannels,:])
	#simulatedSignalComplete=np.append(simulatedSignalComplete,simulatedSignal[drosophilaMikeChannels,:])
	# a means of stopping that callback
	#if antiCrosstalk.isInitialized:
	#	if nTestSignalBlocksLeft>0:
	#		nTestSignalBlocksLeft-=1
	#	else:
	#		calibrationFinished.set()
	#		raise sd.CallbackStop

#fire the whole thing up
calibrationFinished.clear()
antiCrosstalk.reset()
#run callback based loop until someone hits a key
try:
	with sd.Stream(device=(input_device, output_device),
				   samplerate=samplerate, blocksize=soundIoBlockSize,
				   dtype=dtype, latency=latency,
				   channels=nTotalChannels, callback=callback):
		calibrationFinished.wait()
except KeyboardInterrupt:
	print ('stopped by user')	



		
############some result analysis

"""
- callback-Schleife unterbrechen können
- ggf Prozessierung in eigenen Thread?
- mal ein paar Testläufe und maximum der Impusantwort bestimmen

"""
