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

import threading

#from HopWiseInverseSTFT import HopWiseInverseSTFT
from pyOLS import OLS

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x).conj() * fft(y)).real

class ImpulseResponseMeasurement:
		def __init__(self,nBits,nChannels):
			self.testSignalTemplate=(sig.max_len_seq(nBits)[0]-0.5)*0.5
			
			fsample = 44100
			T = nBits
			t = np.arange(0, int(T*fsample)) / fsample
			
		#	self.testSignalTemplate=sig.chirp(t,f0=fsample/2,f1=20,t1=T,method='logarithmic')*0.2
			
			self.testSignalTemplateLength=self.testSignalTemplate.shape[0]		
			self.testSignalTemplate=np.append(self.testSignalTemplate,self.testSignalTemplate) # double the signal so we can do convenient warp around slicing when requesting blocks
			self.testSignalTemplatePos=0
			#place holders for data to be analyzed
			self.recordedSignal=np.zeros([nChannels,self.testSignalTemplateLength*2]) # we reserve some more space so the dont run into trouble with getting blocks of data
			self.recordedSignalPos=0
			#information for state control
			self.needMoreData=True
			self.impulseResponseReadyEvent=threading.Event() # used for signalling that the thread that calculated the IR has finished Processing
		def getNextTestSignalChunk(self,nSamples):
			endIndex=self.testSignalTemplatePos+nSamples
			returnedRange=range(self.testSignalTemplatePos,endIndex)
			self.testSignalTemplatePos=endIndex%self.testSignalTemplateLength
			return self.testSignalTemplate[returnedRange]
		def processSignal(self, recordedSignalChunk):
			nSamples=recordedSignalChunk.shape[1]
			endIndex=self.recordedSignalPos+nSamples
			returnedRange=range(self.recordedSignalPos,endIndex)
			
			self.recordedSignal[:,returnedRange]=recordedSignalChunk
			self.recordedSignalPos+=nSamples
			if self.recordedSignalPos>=self.testSignalTemplateLength:
				self.needMoreData=False
				self.workThread=(threading.Thread(target = self.calculateImpulseResponse))
				self.workThread.start()
				return True
			else: 
				return False

		def calculateImpulseResponse(self):
			cutTemplate=np.tile(self.testSignalTemplate[0:self.testSignalTemplateLength],(self.recordedSignal.shape[0],1))
			cutResponse=self.recordedSignal[:,0:self.testSignalTemplateLength]

					   
			templateFFT=fft(cutTemplate,axis=1)
			responseFFT=fft(cutResponse,axis=1)
			
			templateFFTConj=templateFFT.conj()
			templateFFTSquare=(templateFFT*templateFFTConj).real+np.finfo(float).eps # avoid divisions by zero
			

			self.impulseResponse=ifft (responseFFT*templateFFT.conj()/(templateFFTSquare),axis=1).real
			self.recordedSignalPos=0
			self.testSignalTemplatePos=0
			self.impulseResponseReadyEvent.set()
		def getBestImpulseResponseCut(self,
						  outSize, # length of output
						  hopSize,  # cut signal only at multiples of this, to facilitate integration to audio callbacks/buffering
						  inSignal
						  ):
			#find Part of impulse Response that fits into outSize and contains the highest possible fraction of the impulse Energy
			impulseEnergies=inSignal*inSignal
			cumEnergy=np.cumsum(impulseEnergies)
			maxStartHop=math.floor((inSignal.shape[0]-outSize)/hopSize)
			containedEnergies=[cumEnergy[i*hopSize+outSize]-cumEnergy[i*hopSize] for i in range(0,maxStartHop)]
			bestStartHop= np.argmax(containedEnergies)
			startBin=bestStartHop*hopSize
			endBin=bestStartHop*hopSize+outSize
			return bestStartHop, startBin,endBin 

# audio arrives and goes out in blocks of nIn
# processing works in Blocks of nProcess
# processingBlockSize has to be >ioBlockSize
# can add latency on demand
class BlockSizeAdapter:
	def __init__(self,ioBlockSize, processingBlockSize,nChannels, targetLatencyBlocks, processor):
		self.processingBlockSize=processingBlockSize
		self.ioBlockSize=ioBlockSize

		
		self.inDataCollection=np.zeros([nChannels,processingBlockSize])
		self.outDataCollection=np.zeros([nChannels,processingBlockSize*2])
		self.inputChunkStartIndex=0

		#self.outputChunkStartIndexOffset=processingBlockSize # can be between ioBlockSize and processingBlockSize to adjust additional latency
		#self.outputChunkStartIndexOffset=targetLatency 
		
		ioBlocksPerProcessingBlock=int(self.processingBlockSize/self.ioBlockSize)
		
		if targetLatencyBlocks<(ioBlocksPerProcessingBlock-1) or targetLatencyBlocks>(ioBlocksPerProcessingBlock*2-1):
			raise ValueError('targetLatencyBlocks is ' + str(targetLatencyBlocks)+'but must be between '+ str(ioBlocksPerProcessingBlock-1 )+ "and "+ str(ioBlocksPerProcessingBlock*2-1) )
		additionalLatencyBlocks=targetLatencyBlocks-(ioBlocksPerProcessingBlock-1)
		self.outputChunkStartIndex=self.processingBlockSize+(1-additionalLatencyBlocks)*self.ioBlockSize # can be between ioBlockSize and processingBlockSize to adjust additional latency
		self.processor=processor

	def process(self,inData):
		# first index is for channels, second index is for samples
		inputChunkEndIndex=self.inputChunkStartIndex+self.ioBlockSize
		inputRange=range(self.inputChunkStartIndex,inputChunkEndIndex)
	#	print("in range: "+str(inputRange))
		self.inDataCollection[:,inputRange]=inData
		self.inputChunkStartIndex=inputChunkEndIndex
		#if we have finished a whole block, process it
		if inputChunkEndIndex==(self.processingBlockSize):
			#print("update!")
			self.inputChunkStartIndex=0
			self.outputChunkStartIndex-=self.processingBlockSize
			
			newOutData=self.processor.process(self.inDataCollection)
			
			# shift rolling buffer for output by one processing block
			self.outDataCollection[:,:-self.processingBlockSize]= self.outDataCollection[:,self.processingBlockSize:]
			
			
			self.outDataCollection[:,-self.processingBlockSize:]=newOutData
		# return a chunk of the processed data

		outputChunkEndIndex = self.outputChunkStartIndex%(self.processingBlockSize*2)+self.ioBlockSize
		outputRange=range( self.outputChunkStartIndex%(self.processingBlockSize*2),outputChunkEndIndex)
	#	print("out range: "+str(outputRange))
		self.outputChunkStartIndex=outputChunkEndIndex
		return self.outDataCollection[:,outputRange]
	
class DummyProcessor:
	def process(self,inChunk):
		return inChunk

class BlockDelay:
	def __init__(self,blockSize,nChannels,delay):
		self.nBlocks=delay
		self.nChannels=nChannels
		self.blockSize=blockSize
		self.curWriteBlock=delay
		self.curReadBlock=0
		self.buffer=np.zeros([self.nBlocks,nChannels,blockSize])
	def process(self,data):
		self.curReadBlock=(self.curReadBlock+1)%self.nBlocks
		self.curWriteBlock=(self.curWriteBlock+1)%self.nBlocks
		self.buffer[self.curWriteBlock,:,:]=data
		return(self.buffer[self.curReadBlock,:,:])
		
class CrosstalkSuppressor:
	def __init__(self,ioBlockSize,convolutionSize,nChannels):
		self.ioBlockSize =ioBlockSize
		self.convolutionSize=convolutionSize
		self.nChannels=nChannels
		self.impulseMeasurement=ImpulseResponseMeasurement(19,self.nChannels)
		self.isInitialized=False
	def reset(self):
		self.impulseMeasurement=ImpulseResponseMeasurement(19,self.nChannels)
		self.isInitialized=False
	def _initialize(self):
		self.bestStartHop, startBin,endBin =self.getBestImpulseResponseCut(self.convolutionSize, self.ioBlockSize,self.impulseMeasurement.impulseResponse)
		self.cutResponses=np.reshape(self.impulseMeasurement.impulseResponse[:,startBin:endBin],[1,self.nChannels,self.convolutionSize])
		self.ols=OLS(self.cutResponses)
		self.bsa=BlockSizeAdapter(self.ioBlockSize,self.convolutionSize,self.nChannels,self.bestStartHop,self.ols)
		self.isInitialized=True
	def process(self,
			 crosstalkPlayedData, # what is supposed to be played on the disturbing loudspeaker 
			  crosstalkReceivedData, 		  # what is coming from the microphones that are contaminated by crosstalk
			  crosstalkEleminationFraction # how much of the  crosstalk is supposed to be eleminated (fraction [0,1]) 
			  ):
		
		# we are ready to process external audio
		if self.isInitialized:
			simulatedNoiseAtMike=self.bsa.process(crosstalkPlayedData)
			simulatedNoiseFreeSignal=crosstalkReceivedData-simulatedNoiseAtMike*crosstalkEleminationFraction
			return crosstalkPlayedData,simulatedNoiseFreeSignal,simulatedNoiseAtMike
		#the impulse response calculation has just finished
		if self.impulseMeasurement.impulseResponseReadyEvent.isSet():
			self._initialize()
			return crosstalkPlayedData,np.zeros([self.nChannels,self.ioBlockSize]),np.zeros([self.nChannels,self.ioBlockSize])
		#the impulse response measurement needs more data
		if self.impulseMeasurement.needMoreData:
			self.impulseMeasurement.processSignal(crosstalkReceivedData)
			return self.impulseMeasurement.getNextTestSignalChunk(self.ioBlockSize),np.zeros([self.nChannels,self.ioBlockSize]),np.zeros([self.nChannels,self.ioBlockSize])
		# we are probably just waiting for the impulse response calculation to finish:
		return crosstalkPlayedData,np.zeros([self.nChannels,self.ioBlockSize]),np.zeros([self.nChannels,self.ioBlockSize])
		################### old version
		"""
		if not self.impulseMeasurement.getIsReady():
			finished=self.impulseMeasurement.processSignal(crosstalkReceivedData)
			if  not finished:
				return self.impulseMeasurement.getNextTestSignalChunk(self.ioBlockSize),np.zeros([self.nChannels,self.ioBlockSize]),np.zeros([self.nChannels,self.ioBlockSize])
			else:
				self._initialize()
			return crosstalkPlayedData,np.zeros([self.nChannels,self.ioBlockSize]),np.zeros([self.nChannels,self.ioBlockSize])
		else:
			simulatedNoiseAtMike=self.bsa.process(crosstalkPlayedData)
			simulatedNoiseFreeSignal=crosstalkReceivedData-simulatedNoiseAtMike
			return crosstalkPlayedData,simulatedNoiseFreeSignal,simulatedNoiseAtMike
		"""

		
	def getBestImpulseResponseCut(self,
					  outSize, # length of output
					  hopSize,  # cut signal only at multiples of this, to facilitate integration to audio callbacks/buffering
					  inSignal
					  ):
		#find Part of impulse Response that fits into outSize and contains the highest possible fraction of the impulse Energy
		impulseEnergies=inSignal*inSignal
		impulseEnergies=np.sum(impulseEnergies,axis=0) #sum energies of both responses
		cumEnergy=np.cumsum(impulseEnergies)
		maxStartHop=math.floor((inSignal.shape[1]-outSize)/hopSize)
		containedEnergies=[cumEnergy[i*hopSize+outSize]-cumEnergy[i*hopSize] for i in range(0,maxStartHop)]
		bestStartHop= np.argmax(containedEnergies)
		startBin=bestStartHop*hopSize
		endBin=bestStartHop*hopSize+outSize
		return bestStartHop, startBin,endBin 
