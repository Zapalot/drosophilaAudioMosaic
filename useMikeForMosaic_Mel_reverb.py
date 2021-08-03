#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:04:26 2020

@author: fbo
"""
import os
import time as tm
import numpy as np
import scipy.io.wavfile as wav

import sounddevice as sd
import matplotlib.pyplot as plt
from realtimeMosaicing_Mel import *

import librosa 

######create some test data
#Parameters of STFT
paramSTFT = dict()
paramSTFT['blockSize'] = 2048
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


subsampling=1           # use only every N'th sample to save processing power
maxFlyTemplateSize=256  # use only the first n mosaic grains to speed up NMFdiag
useLiveAudio=True      # use audio input/outpur of sounddevice instead of files
reverbFactorFFT=0.98;   # fraction of old loudness retained from last chunk --- 0: no reverb ... 0.9999: almost inifinite reverb
reverbFactorWave=0.00;
loudnessBoost=3;			# loudness multiplier to compensate for losses due to reverb normalization
#initialize synthesizer with fly waveform
filenameFly = 'template/ZOOM0006_Tr12_excerpt.WAV'

melMatrixVoice= librosa.filters.mel(48000, paramSTFT['blockSize'],n_mels=32,fmax=10000)
melMatrixFly= librosa.filters.mel(48000, paramSTFT['blockSize'],n_mels=32,fmax=10000)
synthesizer=MosaicSynthesizer(paramSTFT, paramNMFdiag,melMatrixVoice,melMatrixFly,reverbFactorFFT,1)
synthesizer.prepareFlyInformation(filenameFly,maxFlyTemplateSize)

reverbBuffer = np.zeros(paramSTFT['hopSize'])

if  useLiveAudio:
	#use live audio

	input_device=0
	output_device=1
	samplerate=48000
	blocksize=512*subsampling
	dtype='float32'
	latency="high"
	channels=1
				
	#this callback is passed by the sounddeivce library
	def callback(indata, outdata, frames, time, status):
		global reverbBuffer
		if status:
			print(status)
		oldTime=tm.perf_counter() 
		reducedIndata=indata[0::subsampling,0]  # use only every N'th sample to save processing power
		reverbBuffer=reverbBuffer*(reverbFactorWave)+reducedIndata
		newMosaicWaveform=synthesizer.processInputChunk(reverbBuffer)
		extractedChunk= synthesizer.extractCentralChunk(newMosaicWaveform)*loudnessBoost
		newTime=tm.perf_counter() 
		print ((512/samplerate)/(newTime-oldTime))   # how many times realtime performance do we achieve?
		outdata[:,0] = np.repeat(extractedChunk.squeeze(),subsampling) # repeat samples to match input sample rate and pass to output
	
	#run callback based loop until someone hits a key
	try:
		with sd.Stream(device=(input_device, output_device),
					   samplerate=samplerate, blocksize=blocksize,
					   dtype=dtype, latency=latency,
					   channels=channels, callback=callback):
			print('#' * 80)
			print('press Return to quit')
			print('#' * 80)
			input()
	except KeyboardInterrupt:
		print ('done')
else:
	#read and write audio data from files
	
	filenameVoice = '../data/text_mosaicing_mixdown_ursula.wav'
	fs, voiceWave = wav.read(filenameVoice)
	make_monaural(voiceWave)
	voiceWave = pcmInt16ToFloat32Numpy(voiceWave)
	

	
	nextInputIndex=0
	outputWave=np.zeros(0)
	shift=0
	oldTime=tm.perf_counter() 
	#process onyl some arbitrary number of windows for comparison purposes
	for i in range(0,round(1000/subsampling)):
		nextVoiceChunk=voiceWave[nextInputIndex:nextInputIndex+paramSTFT['hopSize']*subsampling]
		nextInputIndex=nextInputIndex+paramSTFT['hopSize']*subsampling
		reducedIndata=nextVoiceChunk[0::subsampling]
		reverbBuffer=reverbBuffer*(reverbFactorWave)+reducedIndata
		newMosaicWaveform=synthesizer.processInputChunk(reverbBuffer)*loudnessBoost
		if(len(newMosaicWaveform)>0):
			outputWave=np.append(outputWave,np.repeat(synthesizer.extractCentralChunk(newMosaicWaveform),subsampling))
		#ax.plot(range(nextInputIndex-shift,nextInputIndex+len(newMosaicWaveform)-shift),newMosaicWaveform)
	newTime=tm.perf_counter() 

	#plot output
	import matplotlib
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	plotStartIndex=round(paramSTFT['hopSize']*8.5)
	ax.plot(range(plotStartIndex,plotStartIndex+len(outputWave)),outputWave,color="red")
	
	filenameOut="test_out"+\
	"--signal_extend"+str(synthesizer.signalExtensionFrames)+\
	"--subsampling"+str(subsampling)+\
	"--maxFlyTemplateSize"+str(maxFlyTemplateSize)+\
	"--numIterGriffinLim_"+str(paramSTFT['numIterGriffinLim'])+\
	"--paramNMFdiag_iter"+str(paramNMFdiag['numOfIter'])+\
	"--paramNMFdiag_continuity_length"+str(paramNMFdiag['continuity']['length'])+\
	"--time_needed"+str(round(newTime-oldTime,2))+\
	".wav"
	
	wav.write(filename=filenameOut,
	          rate=fs,
	          data=outputWave)
