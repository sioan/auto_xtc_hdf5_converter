#!/reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.3.58/bin/python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: README 2017-08-06 18:54:12Z sioan@SLAC.STANFORD.EDU $
#
# Description:
#  README file for data analysis
#------------------------------------------------------------------------

#Package author: Sioan Zohar

#Brief description:
#==================
#Abstracts away details of psana, small data, and allows users to focus on developing scientific analysis code.
#
#==================

#to run interactively 
#ipython -i ../../myAnalysisTools/psanaXtcDataExtractor.py -- -e sxri0414 -r 79 -td TSS_OPAL -tc 162 -f 150 -s 4 -t

#to run on batch nodes
#bsub -o %J.log -q psnehprioq -n 48 mpirun --mca btl ^openib psanaXtcDataExtractor.py -e sxri0414 -r 60 -td TSS_OPAL -tc 162 -f 20000

#mpi for specified hosts
#mpirun -n 40 --host daq-amo-mon02,daq-amo-mon03,daq-amo-mon04,daq-amo-mon05,daq-amo-mon06 amon0816.sh

#==================
from mpi4py import MPI
myComm = MPI.COMM_WORLD
myRank = myComm.Get_rank()
size = myComm.Get_size()

import argparse
import os
import sys
sys.path.append(os.curdir)
#sys.path.append(os.getcwd())  
from config import analysisFunctions
from pylab import *
import psana
import subprocess
import time
import h5py
import TimeTool
import pickle
import IPython


#this small data wrapper is to swap out small data implementations 
#depending whether MPIDataSouce or plain data source is used
ttAnalyze = None

def makeDataSourceAndSmallData(experimentNameAndRun,h5FileName,ttDevice,ttCode):
	global ttAnalyze
	smldata = "None"
	small_hdf5_dir = "hdf5"
	os.system("mkdir "+small_hdf5_dir)
	if(ttDevice is not None ):

		print("setting up time tool.") 
		print("Device = "+ttDevice)
		print("bykick code = "+str(ttCode)) 
		ttOptions = TimeTool.AnalyzeOptions(get_key=ttDevice,eventcode_nobeam = ttCode)
		ttAnalyze = TimeTool.PyAnalyze(ttOptions)
	
		print("loading experiment data using standard small data")
		myDataSource = psana.MPIDataSource(experimentNameAndRun,module=ttAnalyze)	

		

		print("defining small data")
		if(h5FileName!="None"):
			smldata = myDataSource.small_data(small_hdf5_dir+"/"+h5FileName, gather_interval=10)
	else:
		print("loading mpi data source")
		myDataSource = psana.MPIDataSource(experimentNameAndRun)

		print("defining small data. hook in place ")
		if(h5FileName!="None"):
			smldata = myDataSource.small_data(small_hdf5_dir+"/"+h5FileName, gather_interval=10)


	return (myDataSource,smldata)

def generateDetectorDictionary(configFileName):
	global ttAnalyze

	myWorkingDirectory = subprocess.check_output("pwd")[:-1]
	print("working directory = "+str(myWorkingDirectory))

	print("reading config file")
	f = open(myWorkingDirectory+"/"+configFileName,'r')
	myDetectorObjectDictionary = {}
	myDetectorObjectDictionary['analyzer'] = {}
	myDetectorObjectDictionary['summarizer'] = {}				
	print("Generating analyzer summarizer and detector objects")
	#print(str(analysisFunctions.__dict__.keys()))	

	for thisDetectorConfig in f:
		if('#'  not in thisDetectorConfig):
			myParsedString = thisDetectorConfig.split(',')
			print(thisDetectorConfig)
			print("found detector object named "+myParsedString[3])
			try:
				test = psana.Detector(myParsedString[0])
			except KeyError:
				print("detector named " + myParsedString[0] + " does not exist in this run")
				continue
			
			myDetectorObjectDictionary[myParsedString[3]] = psana.Detector(myParsedString[0])
			if(myParsedString[4]!='None'):
				myDetectorObjectDictionary['analyzer'][myParsedString[3]] = analysisFunctions.__dict__[myParsedString[4]]
			if(myParsedString[5]!='None'):		
				myDetectorObjectDictionary['summarizer'][myParsedString[3]] = analysisFunctions.__dict__[myParsedString[5]]

		else:
			pass

		if(ttAnalyze is not None):
			print("casting time tool as detector object")
			myDetectorObjectDictionary['TSS_OPAL'] = ttAnalyze
			myDetectorObjectDictionary['analyzer']['TSS_OPAL'] = analysisFunctions.__dict__['getTimeToolData']
			#myDetectorObjectDictionary['summarizer']['TSS_OPAL'] = None
		
	return myDetectorObjectDictionary

def initializeDataDictionaries(myDetectorObjectDictionary):

	myDataDictionary = {}
	summaryDataDictionary={}
	for i in myDetectorObjectDictionary['summarizer']:
		#summaryDataDictionary[i] = 0	#I want to change this to  dict
		summaryDataDictionary[i] = {}	

	return[myDataDictionary,summaryDataDictionary]

def renameSummaryKeys(myDict):
	tempKeys = myDict.keys()
	for i in tempKeys:
		#myDict[i+'Summarized'] = myDict.pop(i)
		myDict[i] = myDict.pop(i)

##################################################################
################### main##########################################

def main(myExp, myRun, configFileName,h5FileName,testSample,ttDevice,ttCode,startEvent,finalEvent):
	global smldata,	summaryDataDictionary,myDataDictionary,myEnumeratedEvents,eventNumber,thisEvent,myDetectorObjectDictionary,mergedGatheredSummary,myRank,myComm

	#global myExp,myRun
	print ("exp = "+str(myExp))
	print ("run = "+str(myRun))
	startTime = time.time()
	print("entering main function")
	
	experimentNameAndRun = "exp=%s:run=%d:smd"%(myExp, myRun)
	#print("loading experiment")
	#myDataSource = psana.MPIDataSource(experimentNameAndRun+":smd")	#this needs to be merged

	#print("defining small data")
	#smldata = myDataSource.small_data(h5FileName)
	
	if(h5FileName!="None"):
		h5FileName = myExp+'run'+str(myRun)+str(h5FileName)+'.h5'
		#print("removing existing file")
		#os.system("rm "+h5FileName)
		#if(myRank==0):		#may still not work if race conditions
			#os.system("rm "+h5FileName)

	myDataSource, smldata = makeDataSourceAndSmallData(experimentNameAndRun,h5FileName,ttDevice,ttCode)

	print("loading detector object dictionary")
	myDetectorObjectDictionary = generateDetectorDictionary(configFileName)
	print("detector object dictionary loaded")
	myDetectorObjectDictionary['rank'] = myRank
	myDetectorObjectDictionary['data_source'] = myDataSource
	myDetectorObjectDictionary['h5FileName'] = h5FileName
	print("supplemental data added to detector object dictionary ")
	
	print("initializing data dictionaries")
	myDataDictionary,summaryDataDictionary = initializeDataDictionaries(myDetectorObjectDictionary)

	messageFeedBackRate = 20+980*int(testSample==False)
	
	print("iterating over enumerated events")
	if(myRank==0):
		print("processing started at "+ str(time.asctime()))
	
	myEnumeratedEvents = enumerate(myDataSource.events())
	for eventNumber,thisEvent in myEnumeratedEvents:
		if(eventNumber %messageFeedBackRate == 1):
			print("iterating over enumerated events. Rank = "+str(myRank)+" Event number = "+str(myRank+eventNumber)+" Elapsed Time (s) = "+str(time.time()-startTime))
			
		if(eventNumber<startEvent):
			continue
		if(testSample and (eventNumber > 200)):
			break
		if(eventNumber > finalEvent):
			break
		
		for i in myDetectorObjectDictionary['analyzer']:
			myDetectorObjectDictionary['self_name'] = i
			myDataDictionary[i] = myDetectorObjectDictionary['analyzer'][i](myDetectorObjectDictionary,thisEvent)
			del myDetectorObjectDictionary['self_name']
		
		for i in myDetectorObjectDictionary['summarizer']:
				myDetectorObjectDictionary['self_name'] = i
				summaryDataDictionary[i] = myDetectorObjectDictionary['summarizer'][i](myDetectorObjectDictionary,thisEvent,summaryDataDictionary[i])

		#for i in myDataDictionary:
		if any([None is myDataDictionary[k] for k in myDataDictionary]):
			continue
		#	if (None is myDataDictionary[i]):
		#		del myDataDictionary[i]

		if(h5FileName!="None"):
			#print myDataDictionary[myDataDictionary.keys()[0]]
			smldata.event(myDataDictionary)


	print("finished looping over events")
	if(h5FileName!="None"):
		print("saving small data")
	
		renameSummaryKeys(summaryDataDictionary)
		#print("gathering dictionaries. rank = "+str(myRank))
		gatheredSummary = myComm.gather(summaryDataDictionary,root=0)
		if myRank==0:
			#print("merging dictionary. rank = " + str(myRank)+". gathered summary "+str(gatheredSummary)+" end of gathered summary.")
			mergedGatheredSummary = merge_dicts(gatheredSummary)
			#print("Here's the merged dictionary")
			#print mergedGatheredSummary
			print("end of merged dictionary")
			smldata.save(mergedGatheredSummary)
			#smldata.save()
			#pickle.dump(gatheredSummary, open( "dictifiedData.pkl", "wb" ))	#for testing.  use code below to read it in.
			#myData = pickle.load(open("dictifiedData.pkl","rb"))
			

			print("small data file saved")
			#smldata.close()
			print("small data file closed")

			
		else:
			smldata.save()


	return

#resulting dictionary tree works well enough. need to streamline.
def merge_dicts(dict_list):
	"""
	Given any number of dicts, shallow copy and merge into a new dict,
	precedence goes to key value pairs in latter dicts.
	"""
	#print ("merging dictionary")
	result = {}
	result['summary'] = {}
	#print("merging the dictionary list.")
	#print(dict_list)
	myCounter = 0
	for dictionary in dict_list:
		result['summary']["nonMeaningfulCoreNumber"+str(myCounter)]={}
		myCounter += 1

	myCounter = 0
	for dictionary in dict_list:
		result['summary']["nonMeaningfulCoreNumber"+str(myCounter)].update(dictionary)
		myCounter += 1
		#print dictionary
		#result.update(dictionary)
	return result


if __name__ == '__main__':
	
	print("parsing arguments")
	myParser = argparse.ArgumentParser(description='Abstracts data analysis into user functions')
		
	myParser.add_argument('-e','--exp', help='the experiment name')
	myParser.add_argument('-r','--run',type=int,help='the run number to use when running offline')
	myParser.add_argument('-c','--configFile',help='the config file to read from',default='config/analysis.cfg')
	myParser.add_argument('-hd5','--hd5File',help='extension of the small data file to write to. typically a,b or c',default="")
	myParser.add_argument('-t','--testSample',action='store_true',help='only take a small set of data for testing')
	myParser.add_argument('-td','--ttDevice',type=str,help='device to use for getting time tool ', default=None)
	myParser.add_argument('-tc','--ttCode',type=int,help='event code to identify by kick', default=None)
	
	myParser.add_argument('-s','--start',type=int,help='skips until starting event reached', default=-1)
	myParser.add_argument('-f','--final',type=int,help='up to final event', default=1e12)
	

	myArguments = myParser.parse_args()
	print("arguments parsed")

	main(
		myArguments.exp,
		myArguments.run,
		myArguments.configFile,
		myArguments.hd5File,
		myArguments.testSample,
		myArguments.ttDevice,
		myArguments.ttCode,
		myArguments.start,
		myArguments.final)

	print("finished collecting data")
	if(myArguments.testSample):
		print("interactive usage instructions.")
		print("The event number is stored in variable 'eventNumber'.")
		print("The this current event is stored in object 'thisEvent'.")
		print("detector objects are pre-initialized and stored in 'myDetectorObjectDictionary'")
		print("thisEvent can be incremented with this snippet of code:")
		print("									eventNumber,thisEvent = next(myEnumeratedEvents")
		print("the current event data to be saved to 'small data' is contained in 'myDataDictionary'")
		print("the summarized result to be saved at after all events are finished is contained in 'summaryDataDictionary'")

		
	
