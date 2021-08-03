#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:10:50 2021

@author: fbo

"""
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft

class HopWiseInverseSTFT:
	def __init__(self,paramSTFT,numBins):
		self.numBins=numBins
		self.reconstMirror=paramSTFT['reconstMirror']
		self.analyticSig=paramSTFT['analyticSig']
		self.blockSize=paramSTFT['blockSize']
		
		# initialize parameters
		
		self.reconstMirror = paramSTFT['reconstMirror']
		self.appendFrame = paramSTFT['appendFrame']
		self.analyticSig = paramSTFT['analyticSig']
		self.blockSize = paramSTFT['blockSize']
		self.hopSize = paramSTFT['hopSize']
		self.numPadBins = self.blockSize - self.numBins
		self.analysisWinFunc = paramSTFT['winFunc']
		self.synthesisWinFunc = paramSTFT['winFunc']
		
		
		self.numSamples=self.blockSize+self.hopSize
		#construct buffer for overlap/Add 
		self.overLapAddBuffer = np.zeros(self.numSamples, dtype=np.complex32) if self.analyticSig else np.zeros(self.numSamples, dtype=np.float32)
		
		# construct normalization function for the synthesis window
		# that equals the denominator in eq. (6) in [1]
		    # we need to change the signal scaling in case of the analytic signal
		scale = 2.0 if self.analyticSig else 1.0
		winFuncProd = self.analysisWinFunc * self.synthesisWinFunc
		redundancy = round(self.blockSize / self.hopSize)
	
		# construct hopSize-periodic normalization function that will be
		# applied to the synthesis window
		nrmFunc = np.zeros(self.blockSize)
	
		# begin with construction outside the support of the window
		for k in range(-redundancy + 1, redundancy):
			nrmFuncInd = self.hopSize * k
			winFuncInd = np.arange(0, self.blockSize)
			nrmFuncInd += winFuncInd
	
			# check which indices are inside the defined support of the window
			validIndex = np.where((nrmFuncInd >= 0) & (nrmFuncInd < self.blockSize))
			nrmFuncInd = nrmFuncInd[validIndex]
			winFuncInd = winFuncInd[validIndex]
	
			# accumulate product of analysis and synthesis window
			nrmFunc[nrmFuncInd] += winFuncProd[winFuncInd]
	
		# apply normalization function
		self.synthesisWinFunc /= nrmFunc
		self.synthesisWinFunc*=scale
		
	def addChunkStft(self, currSpec):
		numPadBins = self.blockSize - self.numBins
		# if desired, construct artificial mirror spectrum
		if self.reconstMirror:
			#if the analytic signal is wanted, put zeros instead
			padMirrorSpec = np.zeros(int(numPadBins))
			if not self.analyticSig:
				padMirrorSpec = np.conjugate(np.flip(currSpec[1:int(self.numBins) - 1], axis=0))
			# concatenate mirror spectrum to base spectrum
			currSpec = np.concatenate((currSpec, padMirrorSpec), axis=0)
		
        # transform to time-domain
		snip = ifft(currSpec)
		snip = np.real(snip)
		signalWindowed=snip*self.synthesisWinFunc
		
		# update overlap-add buffer
		#always write to the last block of the buffer:
			
		overlapAddInd = range(self.hopSize,self.hopSize+self.blockSize)

		# and do the overlap add, with synthesis window and scaling factor included
		self.overLapAddBuffer[overlapAddInd] += signalWindowed
		 
		#remove oldest data from buffer and place zeros in the front
		self.overLapAddBuffer=np.roll(self.overLapAddBuffer,-self.hopSize)
		#self.overLapAddBuffer[0:self.blockSize]=self.overLapAddBuffer[self.hopSize:(self.blockSize+self.hopSize)]
		self.overLapAddBuffer[self.blockSize:self.blockSize+self.hopSize]=0
	def getNewestHop(self):
		return self.overLapAddBuffer[0:self.hopSize]