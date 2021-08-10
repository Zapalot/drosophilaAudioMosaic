#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:40:45 2020

@author: fbo
"""
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy.fftpack import ifft

from NMFtoolbox.utils import EPS, MAX_WAV_VALUE, make_monaural, pcmInt16ToFloat32Numpy
from NMFtoolbox.forwardSTFT import forwardSTFT
from copy import deepcopy
from NMFtoolbox.inverseSTFT import inverseSTFT
from NMFtoolbox.NMFdiag import NMFdiag
from NMFtoolbox.visualizeComponentsNMF import visualizeComponentsNMF
from NMFtoolbox.LSEE_MSTFTM_GriffinLim import LSEE_MSTFTM_GriffinLim
from NMFtoolbox.LSEE_MSTFTM_GriffinLim import init_parameters

from HopWiseInverseSTFT import HopWiseInverseSTFT



# a class for realtime Audio Moscaicing based on the NFM Toolbox

class MosaicSynthesizer:
		def __init__(self, paramSTFT,paramNMFdiag,melMatrixVoice,melMatrixFly,reverbFactor=0.0, subsampling=1):
			self.reverbFactor=reverbFactor;
			self.subsampling=subsampling;
			self.melMatrixVoice=melMatrixVoice
			self.melMatrixFly=melMatrixFly
			self.paramNMFdiag=paramNMFdiag
			self.paramSTFT=paramSTFT
			self.voiceWaveform=np.zeros([1, ])
			


			self.blockSize=paramSTFT['blockSize']
			self.hopSize=paramSTFT['hopSize']
			self.signalExtensionFrames=4                         # how many hops shall the analyszed signal continue to the left and right?
			self.numVoiceFrames=self.signalExtensionFrames*2+1   # how many frames in total is the analyzed waveform?
			
			#derived lengths of waveform for convenience
			self.signalExtensionSamples=self.signalExtensionFrames*self.hopSize
			self.analyzedChunkLength=self.blockSize+2*self.signalExtensionFrames*self.hopSize+1 #ToDo: The "+1" at the end is an ugly hack to make up for the wrong counting of Frames in the forwardSTFT-Method
		
			# an overlap-Add-iFFT  to turn the mosaic spectrum back into audio
			self.ifftProcessor=HopWiseInverseSTFT(paramSTFT,paramSTFT['blockSize']/2+1)
			
			self.lastFFTAmplitudes=np.zeros(round(self.blockSize / 2) + 1) #used for external visualization
			
		# prepare some STFT information of the mosaic grains
		def  prepareFlyInformation(self,templateFileName, maxFlyTemplateSize=512):
			fs, flyWave = wav.read( templateFileName)
			make_monaural(flyWave)
			flyWave = pcmInt16ToFloat32Numpy(flyWave)
			flyWave=flyWave[0::self.subsampling]
			self.paramSTFT['numSamples'] = len(flyWave)
			flySTFT, flyAmplitudes, _ = forwardSTFT(flyWave, self.paramSTFT)
			#Do a Mel-Transform of the Spectrum for faster Mosaicing
			flyAmplitudesMel=np.matmul(self.melMatrixFly,flyAmplitudes)
			
			#cut the number of templates
			
			flySTFT=flySTFT[:,:maxFlyTemplateSize]
			flyAmplitudes=flyAmplitudes[:,:maxFlyTemplateSize]
			
			flyAmplitudesMel=flyAmplitudesMel[:,:maxFlyTemplateSize]
			
			#build normalized Versions for mosaicing
			self.flyAmplitudesNormalized = flyAmplitudes * 1./ (EPS + np.sum(flyAmplitudes, axis=0)) # init templates by source frames
			self.flySTFTNormalized = flySTFT * 1./ (EPS + np.sum(flyAmplitudes, axis=0))
			
			self.flyAmplitudesNormalizedMel = flyAmplitudesMel * 1./ (EPS + np.sum(flyAmplitudesMel, axis=0)) # init templates by source frames
			
			self.numBins, self.numFlyFrames = flySTFT.shape
			self.numBinsMel,_=flyAmplitudesMel.shape
	
		# process one "hop" of additional audio wave widthout any additional pre/postprocessing/subsampling/cutting
		def processInputChunk(self,newWaveform):
			if(len(self.voiceWaveform)<self.analyzedChunkLength):
				self.voiceWaveform=np.append(self.voiceWaveform,newWaveform)
				if(len(self.voiceWaveform)==self.analyzedChunkLength):
					return self.initializeVoice()
				else:
					#return np.zeros(0)
					return np.zeros(self.hopSize*4)
			else:
				return self.updateMosaic(newWaveform)
		def processAudioChunk(self,newAudio):
			reducedIndata=newAudio[0::self.subsampling]  # use only every N'th sample to save processing power
			newMosaicWaveform=self.processInputChunk(reducedIndata)
			return np.repeat(self.extractCentralChunk(newMosaicWaveform).squeeze(),self.subsampling)
			 		
		#once we have gathered enough voice frames, we can fill the intital set of buffers and matrices:
		def initializeVoice(self):
			#fill voice STFT matrix
			self.paramSTFT['numSamples'] = len(self.voiceWaveform) 
			self.voiceSTFT, self.voiceAmplitudes, _ = forwardSTFT(self.voiceWaveform, self.paramSTFT)
			 
			#calculate mel Spectra
			self.voiceAmplitudesMel=np.matmul(self.melMatrixVoice,self.voiceAmplitudes)
			
			#start the algorithm with a random matrix as first mixture guess
			self.mixMatrix = np.random.rand(self.numFlyFrames, self.numVoiceFrames)
			# call the reference implementation as provided by Jonathan Driedger --- this is the thing that actually does the magic
			# the resulting "H" turns flyAmplitudesNormalized(Which is a normalized voiceAmplitudes) into flyAmplitudes  (flyAmplitudes=nmfdiagW*nmfdiagH, where nmfdiagW==flyAmplitudesNormalized because of paramNMFdiag['fixW'] = True ) 

			_, self.mixMatrix = NMFdiag(self.voiceAmplitudesMel, self.flyAmplitudesNormalizedMel, self.mixMatrix, self.paramNMFdiag)
			self.mosaicSTFT = np.matmul(self.flySTFTNormalized, self.mixMatrix)
			
			# resynthesize using Griffin-Lim
			#self.mosaicSTFT, _, mosaicWaveform = LSEE_MSTFTM_GriffinLim(self.mosaicSTFT,self.paramSTFT)
			#initialize overlap-add
			for  i in range(0,self.mosaicSTFT.shape[1]):
				self.ifftProcessor.addChunkStft(self.mosaicSTFT[:,i])
			
			return self.ifftProcessor.getNewestHop()#[self.blockSize:self.blockSize+self.hopSize]
		
		#with each additional frame of audio, we drop one frame/column of the old data and run the mosaicing algorithm
		def updateMosaic(self,newWaveform):
			##### update waveform and spectral data with additional chunk of input data
			
			# append  additional chunk of input data to waveform buffer
			self.voiceWaveform=np.append(self.voiceWaveform,newWaveform)[-self.analyzedChunkLength:]
			
			#calculate FFT of last Block of WaveForm
			lastVoiceBlock=self.voiceWaveform[-self.blockSize:]
			windowedSignal =self.paramSTFT['winFunc']*lastVoiceBlock 			# apply windowing
			voiceSTFTNew = fft(windowedSignal, axis=0)[:round(self.blockSize / 2) + 1] #  drop mirror spectrum of fft
			voiceAmplitudesNew=np.abs(voiceSTFTNew)
			self.lastFFTAmplitudes=	voiceAmplitudesNew
			#reverb
			voiceAmplitudesNew=voiceAmplitudesNew*(1.0-self.reverbFactor)+self.voiceAmplitudes[:,-1]*(self.reverbFactor)
			  
			#replace oldest Data in Voice STFTS by new result of FFT 
			self.voiceSTFT=np.roll(self.voiceSTFT,-1,axis=1);
			self.voiceSTFT[:,-1]=voiceSTFTNew
			
			self.voiceAmplitudes=np.roll(self.voiceAmplitudes,-1,axis=1);
			self.voiceAmplitudes[:,-1]=voiceAmplitudesNew
			
			#do the same with mel spectra
			voiceAmplitudesNewMel=np.matmul(self.melMatrixVoice,voiceAmplitudesNew)
			self.voiceAmplitudesMel=np.roll(self.voiceAmplitudesMel,-1,axis=1);
			self.voiceAmplitudesMel[:,-1]=voiceAmplitudesNewMel
			
			
			##### discard oldest column in mix Matrix and add random intialization for new Block
			self.mixMatrix=np.roll(self.mixMatrix,-1,axis=1);
			self.mixMatrix[:,-1]=np.random.rand(self.numFlyFrames)
			
			##### run the mosaicing step with one additional column of data. ToDo: Maybe this can be sped up somehow?
			_, self.mixMatrix = NMFdiag(self.voiceAmplitudesMel, self.flyAmplitudesNormalizedMel, self.mixMatrix, self.paramNMFdiag)
			
			##### push STFT of added chunk into the matrixs
			self.mosaicSTFT=np.roll(self.mosaicSTFT,-1,axis=1);
			self.mosaicSTFT[:,-1]=np.matmul(self.flySTFTNormalized,self.mixMatrix[:,-1])
			
			##### use GriffinLim to adjust phase of last Chunk and synthesize Waveform of Mosaik
			self.paramSTFT['numSamples'] = len(self.voiceWaveform)
			#self.mosaicSTFT,  mosaicWaveform = LSEE_MSTFTM_GriffinLim_add_chunk(self.mosaicSTFT, self.paramSTFT)
			self.ifftProcessor.addChunkStft(self.mosaicSTFT[:,0])
			return self.ifftProcessor.getNewestHop()#[self.blockSize:self.blockSize+self.hopSize]
		
		#we use the central piece of the mosaicing output to build up the output waveform
		def extractCentralChunk(self, mosaicWaveform):
			if(len(mosaicWaveform)<=self.hopSize):
				return mosaicWaveform
			else:
				startIndex=round(len(mosaicWaveform)/2-self.hopSize/2)
				return mosaicWaveform[startIndex:startIndex+self.hopSize]
		
			
			
			
#a modified version of GriffinLim that only updates the last window of the STFT

def LSEE_MSTFTM_GriffinLim_add_chunk(X, parameter=None):
    """Performs phase reconstruction as described in [2], leaving phase in all but the last hop unaltered

    References
    ----------
    [2] Daniel W. Griffin and Jae S. Lim, Signal estimation
    from modified short-time fourier transform, IEEE
    Transactions on Acoustics, Speech and Signal Processing,
    vol. 32, no. 2, pp. 236-243, Apr 1984.

    The operation performs an iSTFT (LSEE-MSTFT) followed by STFT on the
    resynthesized signal.

    Parameters
    ----------
    X: array-like
        The STFT spectrogram to iterate upon

    parameter: dict
        blockSize:       The blocksize to use during analysis
        hopSize:         The used hopsize (denoted as S in [1])
        anaWinFunc:      The window used for analysis (denoted w in [1])
        synWinFunc:      The window used for synthesis (denoted w in [1])
        reconstMirror:   If this is enabled, we have to generate the
                         mirror spectrum by means of conjugation and flipping
        appendFrames:    If this is enabled, safety spaces have to be removed
                         after the iSTFT
        targetEnv:       If desired, we can define a time-signal mask from the
                         outside for better restoration of transients

    Returns
    -------
    Xout: array-like
        The spectrogram after iSTFT->STFT processing

    Pout: array-like
        The phase spectrogram after iSTFT->STFT processing

    res: array-like
        Reconstructed time-domain signal obtained via iSTFT
    """

    numBins, _ = X.shape
    parameter = init_parameters(parameter, numBins)

    Xout = deepcopy(X)

    A = abs(Xout)

    for k in range(parameter['numIterGriffinLim']):
        # perform inverse STFT
        res, _ = inverseSTFT(Xout, parameter) #TODO: we could run this only on the last window to gain performance

        # perform forward FFT only on last chunk
        windowedSignal =parameter['winFunc']*res[1,-parameter['blockSize']:]
        chunkFFT = fft(windowedSignal, axis=0)[:round(parameter['blockSize'] / 2) + 1] # drop mirror spectrum of fft
        newPhases = np.angle(chunkFFT)
	
		# replace old part of the phases
        Xout[:,-1]=(A[:,-1]).squeeze()* np.exp(1j * newPhases)
    return Xout,  res



