from pylab import *
import psana
import IPython
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import h5py
import numpy as np
from psmon.plots import MultiPlot,Image,XYPlot
from psmon import publish
import time
from mpi4py import MPI

######################################################
#######Using the eigen traces#########################
######################################################

#try:
	#eigen_traces_h5py=h5py.File("eigen_traces.h5")
	#eigen_traces = f['summary/nonMeaningfulCoreNumber0/Acq01/ch1/eigen_wave_forms']
#except:
#	print("eigen_traces.h5 not found")
#	pass


def use_acq_svd_basis(detectorObject, thisEvent):
	global eigen_traces_h5py
	try:
		temp = eigen_traces_h5py.keys()
	except:
		eigen_file = "hdf5/"+detectorObject['h5FileName'][:-3]+"_eigen_basis.h5"
		eigen_traces_h5py=h5py.File(eigen_file)
		print("loaded "+eigen_file)

	selfName = detectorObject['self_name']
	my_results = {}
	config_parameters = {"thresh_hold":0.05,"waveform_mask":arange(1100,1700),"eigen_basis_size":25,"offset_mask":arange(300)}

	if(None is detectorObject[selfName](thisEvent)):
		#fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return None

	if(len(detectorObject[selfName](thisEvent))==4):		#this is temp cludge that will break when hsd is set to two channels
		the_wave_forms = detectorObject[selfName](thisEvent)
	else:
		the_wave_forms = detectorObject[selfName](thisEvent)[0]

	#x = detectorObject[selfName](thisEvent)[1][0]
	for i in arange(len(the_wave_forms)):
		eigen_traces = eigen_traces_h5py["summary/nonMeaningfulCoreNumber0/"+selfName+"/ch"+str(i)+"/norm_eigen_wave_forms"]
		#eigen_traces = eigen_traces_h5py["summary/nonMeaningfulCoreNumber0/"+selfName+"/ch"+str(i)+"/eigen_wave_forms"]
		#eigen_traces = array([eigen_traces[j]/sum(eigen_traces[j]**2)**0.5 for j in arange(len(eigen_traces))])	#not efficient. constantly renormalizing. will optimize later

		y = 1.0*the_wave_forms[i]
		y -= mean(y[config_parameters['offset_mask']])
		weightings = dot(eigen_traces,y)
		residuals = y-dot(weightings,eigen_traces)
		variance = sum(residuals**2)/len(y)
		#variance = dot(eigen_traces,residuals)**2
		#approximation in line above comes from eigen_traces is orthogonal matrix,
		#so dot (eigen_traces.transpose(),eigen_traces) is diagonal.
		#missing something with number of points and degrees of freedom
		my_results["ch"+str(i)] = {"weightings":weightings,"variance":variance}

	return my_results

######################################################
#######Creating the acqiris eigen basis###############
######################################################
def svd_update(eigen_system,new_vector,config_parameters):
	roi_mask_start = 1000
	roi_mask_end = 2500


	try:
		reconstructed_system = dot(eigen_system['eigen_weightings'], eigen_system['eigen_wave_forms'])
		reconstructed_system = vstack([reconstructed_system,new_vector])

		roi_mask = zeros(reconstructed_system.shape).astype(bool)
		roi_mask[:,roi_mask_start:roi_mask_end]=True
		masked_reconstructed_system = reconstructed_system[roi_mask]
		masked_reconstructed_system.shape = (len(roi_mask),roi_mask_start-roi_mask_end)

		#singular_values,svd_lsv = eig(dot(reconstructed_system,reconstructed_system.transpose()))
		singular_values,svd_lsv = eig(dot(masked_reconstructed_system,masked_reconstructed_system.transpose()))

		new_weightings = dot(svd_lsv,diag(singular_values))
		new_eigen_vectors = dot(pinv(new_weightings),reconstructed_system)[:config_parameters["eigen_basis_size"]]

		norm_eigen_vectors = real(array([new_eigen_vectors[i]/sum(new_eigen_vectors[i]**2)**0.5 for i in arange(len(new_eigen_vectors))]))

		eigen_system = {'eigen_weightings':new_weightings[:config_parameters["eigen_basis_size"],:config_parameters["eigen_basis_size"]] ,'eigen_wave_forms':new_eigen_vectors,'norm_eigen_wave_forms':norm_eigen_vectors}


	except TypeError:
		if ((None is new_vector) and (len(eigen_system['eigen_weightings'])>1)):
			pass
		else:
			eigen_system['eigen_weightings'] = [1]
			eigen_system['eigen_wave_forms'] = new_vector
			eigen_system['norm_eigen_wave_forms'] = new_vector

	except ValueError:
		if (1==len(eigen_system['eigen_weightings'])):
			eigen_system['eigen_weightings'] = array([[1,0],[0,1]])
			eigen_system['eigen_wave_forms'] = vstack([eigen_system['eigen_wave_forms'],new_vector])
			eigen_system['norm_eigen_wave_forms'] = eigen_system['eigen_wave_forms']

	return eigen_system

def make_acq_svd_basis(detectorObject,thisEvent,previousProcessing):
	selfName = detectorObject['self_name']
	config_parameters = {"thresh_hold":0.05,"waveform_mask":arange(1200,1230),"eigen_basis_size":25,"offset_mask":arange(300)}

	eigen_system = {}

	##############################
	#### initializing arrays #####
	##############################
	if None is detectorObject[selfName](thisEvent):
		return None

	if(len(detectorObject[selfName](thisEvent))==4):
		the_wave_forms = detectorObject[selfName](thisEvent)
	else:
		the_wave_forms = detectorObject[selfName](thisEvent)[0]

	for i in arange(len(the_wave_forms)):

		try:
			eigen_system["ch"+str(i)] = previousProcessing["ch"+str(i)]
		except (KeyError,TypeError) as e:
			try:
				y =  1.0*the_wave_forms[i]
				y -= mean(y[config_parameters['offset_mask']])
				eigen_system["ch"+str(i)]= {'eigen_wave_forms':y,'eigen_weightings':[1],'norm_eigen_wave_forms':[1]}
			except (KeyError,TypeError) as e:
				eigen_system["ch"+str(i)] = {'eigen_wave_forms':None,'eigen_weightings':None,'norm_eigen_wave_forms':None}

	##############################
	###main part of calculation###
	##############################
	new_eigen_system = {}
	for i in arange(len(the_wave_forms)):

		y = 1.0 *the_wave_forms[i]
		y -= mean(y[config_parameters['offset_mask']])
		start_time = time.time()
		new_eigen_system["ch"+str(i)] = svd_update(eigen_system["ch"+str(i)],y,config_parameters)

	#print(time.time() - start_time)
	#for j in eigen_system["ch"+str(i)]:
	#	print(str(j)+", "+str(eigen_system["ch"+str(i)][j].shape))

	########################################################
	########plotting for real time SVD debugging############
	########################################################
	publish.local=True
	try:
		wave_to_plot = new_eigen_system["ch2"]['norm_eigen_wave_forms']
		to_plot = XYPlot(time.time(),"eigen_system",[arange(len(wave_to_plot[0])),arange(len(wave_to_plot[0]))],[wave_to_plot[0],wave_to_plot[1]])
		publish.send('eigen_system_'+selfName,to_plot)
		#psplot -s hostname -p 12303 eigen_system
	except:
		pass

	########################################################
	########end of  SVD debugging###########################
	########################################################

	return new_eigen_system

######################################################
#######End of eigen basis generation##################
######################################################

def genericReturn(detectorObject,thisEvent):
	selfName = detectorObject['self_name']
	return detectorObject[selfName](thisEvent)


def get_projection(detectorObject,thisEvent):

	selfName = detectorObject['self_name']
	myImage = detectorObject[selfName].raw(thisEvent)



	if None == myImage:
		return (zeros(1024))
	else:
		return sum(myImage[370:],axis=0)


def genericSummaryZero(detectorObject,thisEvent,previousProcessing):
	return 0

def myZeroReturn(detectorObject,thisEvent,previousProcessing):
	return 0

def getTimeToolData(detectorObject,thisEvent):
	selfName = detectorObject['self_name']
	ttData = detectorObject[selfName].process(thisEvent)
	myDict = {}
	if(ttData is None):

		myDict['amplitude'] = -99999.0
		myDict['pixelTime'] = -99999.0
		myDict['positionFWHM'] = -99999.0


	else:

		myDict['amplitude'] = ttData.amplitude()
		myDict['pixel_position'] = ttData.position_pixel()
		myDict['pixelTime'] = ttData.position_time()
		myDict['positionFWHM'] = ttData.position_fwhm()

	return myDict

def getPeak(detectorObject,thisEvent):
	selfName = detectorObject['self_name']

	if(None is detectorObject[selfName](thisEvent)):
		fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return fit_results


	myWaveForm = -detectorObject[selfName](thisEvent)[0][0]

	myWaveForm -= mean(myWaveForm[:2500])

	x = arange(len(myWaveForm))[7500:10000]-8406
	#myFit = polyfit(x, myWaveForm[7500:10000],3)
	#p = poly1d(myFit)
	#myMax = max(p(x))
	#return myMax


	try:
		popt,pcov = curve_fit(peakFunction,x,myWaveForm[7500:10000])

		fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}

	except RuntimeError:
		fit_results = {'amplitude':-9999.0,'uncertainty_cov':99999.0}


	return fit_results

def accumulateAverageWave(detectorObject,thisEvent,previousProcessing):
	selfName = detectorObject['self_name']

	myWaveForm = -detectorObject[selfName](thisEvent)[0][0]
	myWaveForm -= mean(myWaveForm[:2500])

	return (previousProcessing+myWaveForm)

def getWaveForm(detectorObject,thisEvent):
	selfName = detectorObject['self_name']

	if (None not in [detectorObject[selfName](thisEvent)[0][0]]):
		return detectorObject[selfName](thisEvent)[0][0]
	else:
		return 0

def get(detectorObject,thisEvent):
	selfName = detectorObject['self_name']

	if (None not in [detectorObject[selfName](thisEvent)]):
		return detectorObject[selfName](thisEvent)
	else:
		return 0

def getRaw(detectorObject,thisEvent):
	selfName = detectorObject['self_name']

	if (None not in [detectorObject[selfName](thisEvent)]):
		return detectorObject[selfName](thisEvent)
	else:
		return 0

def getGMD(detectorObject,thisEvent):
	selfName = detectorObject['self_name']

	temp = detectorObject[selfName].get(thisEvent)

	my_dict = {"milliJoulesPerPulse":-99999.0,"milliJoulesAverage":-99999.0,"relativeEnergyPerPulse":999999.0}

	if (None not in [temp]):
		my_dict["milliJoulesPerPulse"]=temp.milliJoulesPerPulse()
		my_dict["milliJoulesAverage"]=temp.milliJoulesAverage()
		my_dict["relativeEnergyPerPulse"]=temp.relativeEnergyPerPulse()

	return my_dict


#for slow cameras that would crash psana if written every event cause of back filling with zeros
def slowCameraImageSummarizer(detectorObject,thisEvent,previousProcessing):

	selfName = detectorObject['self_name']
	#return detectorObject.image(thisEvent)
	tempImage = detectorObject[selfName].image(thisEvent)
	myDict= {}

	try:
		if(type(previousProcessing) != dict):
			previousProcessing = {}
	except NameError:
		previousProcessing = {}

	if(tempImage is not None):
		print("got image")
		myEventId = thisEvent.get(psana.EventId)
		myTime = myEventId.time()[0]
		myDict["sec"+str(myTime)] = tempImage

		previousProcessing.update(myDict)

	return previousProcessing

def getDLS(detectorObject, thisEvent):
        selfName = detectorObject['self_name']

        if detectorObject[selfName].values(thisEvent) is None:
                myDictionary = {'DLS_PS': -999.0}
                return myDictionary

        DLS_PS = detectorObject[selfName].values(thisEvent)[0]
        return {'DLS_PS': DLS_PS}

def peakFunction(x,a,x0,offset):
	return a*(x-x0)**2+offset

def generic_acqiris_analyzer(detectorObject,thisEvent):

	peak_width = 12	############################## tunable parameter. needs to be different for each trace.
	selfName = detectorObject['self_name']
	fit_results = {}

	if(None is detectorObject[selfName](thisEvent)):
		#fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return None

	x = detectorObject[selfName](thisEvent)[1][0]
	for i in arange(len(detectorObject[selfName](thisEvent)[0])):

		y = abs(detectorObject[selfName](thisEvent)[0][i])


		smoothed_wave = convolve(y,[1,1,1,1,1,1],mode='same')
		initial_peak = argmax(smoothed_wave)	#how to hardcode
		initial_height = smoothed_wave[initial_peak]

		y_small = y[initial_peak-peak_width:initial_peak+peak_width] - mean(y[:])
		x_small = x[initial_peak-peak_width:initial_peak+peak_width]

		fit_results['ch'+str(i)] = {"position":-999999.0,'area':-999999.0,'position_var':-999999.0,'amplitude_var':-999999.0}

		try:
			fit_results['ch'+str(i)]['area']=sum(y_small)

			popt,pcov = curve_fit(peakFunction,x_small,y_small,p0=[0.0,initial_peak,initial_height])

			fit_results['ch'+str(i)]["position"]      = popt[1]
			fit_results['ch'+str(i)]['amplitude']     = popt[2]
			fit_results['ch'+str(i)]['position_var']  = pcov[1,1]
			fit_results['ch'+str(i)]['amplitude_var'] = pcov[2,2]

		except (RuntimeError,TypeError) as e:
			print("fitting failed")
			pass

	return fit_results

def get_raw_acq(detectorObject,thisEvent):
	

	mask_dict = {}	
	mask_dict['hsd',1] = [2000,2750]
	mask_dict['hsd',2] = [2000,2750]
	mask_dict['hsd',3] = [2000,2750]
	mask_dict['hsd',4] = [2000,2750]

	mask_dict['acq02',1] = [750,2000]
	mask_dict['acq02',2] = [750,2000]
	mask_dict['acq02',3] = [750,2000]
	mask_dict['acq02',4] = [750,2000]

	selfName = detectorObject['self_name']
	my_dict = {}

	if(None is detectorObject[selfName](thisEvent)):
		#fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return None

	if("hsd" in selfName):		#this is temp cludge that will break when hsd is set to two channels
			the_wave_forms = detectorObject[selfName](thisEvent)
	else:
			the_wave_forms = detectorObject[selfName](thisEvent)[0]
	
	
	for i in arange(0,len(the_wave_forms)):

		y = the_wave_forms[i]

		my_dict['ch'+str(i+1)] = y[mask_dict[selfName,i+1][0]:mask_dict[selfName,i+1][1] ]

	return my_dict

def getAndorFVBImage(detectorObject,thisEvent):
	'''
	get the andor profile, if camera was used in Full Vertical Binning mode.
	'''
	selfName = detectorObject['self_name']
	myImage = detectorObject[selfName].raw(thisEvent)
	my_dict = {}

	if None == myImage:
		my_dict['image'] = zeros(2048)
		#print("None")
	else:
		my_dict['image'] = np.sum(myImage, axis=0)
		#print(myImage.shape)
	return my_dict


def getMonoEncoderValues(detectorObject,thisEvent):
	selfName = detectorObject['self_name']
	to_return = [0.0,0.0,0.0,0.0]
	if(None != detectorObject[selfName].values(thisEvent)):
		to_return = detectorObject[selfName].values(thisEvent)
	return to_return


def miniTOF(detobj, thisEvent):
	selfName = detobj['self_name']
	minitof_volts_raw = detobj[selfName].waveform(thisEvent)
	if minitof_volts_raw is None:
		return None
	minitof_volts = minitof_volts_raw[2]  # channel 3
	#iTOF_yield = np.mean(minitof_volts[6700:8000]) - np.mean(minitof_volts[-1000:])
	bg = np.mean(minitof_volts[-1000:])
	return dict(volts=minitof_volts-bg, bg=bg)


def pnccd_image(detobj, thisEvent):
	selfName = detobj['self_name']
	image = detobj[selfName].image(thisEvent)
	if image is None:
		return None
	return dict(image=image)


def opal_image(detobj, thisEvent):
	selfName = detobj['self_name']
	image = detobj[selfName].raw(thisEvent)
	if image is None:
		return None
	return dict(image=image)


roidict1 = dict()
roidict1[6] = slice(480, 580)
roidict1[3] = slice(550, 650)  # stability test
roidict2 = dict()
roidict2[6] = slice(630, 730)
for i in range(1, 200):
	roidict1[i] = roidict1[6]
	roidict2[i] = roidict2[6]
roidict1[11] = slice(330, 430)
roidict2[11] = slice(470, 570)
for i in range(11, 200):
	roidict1[i] = roidict1[11]
	roidict2[i] = roidict2[11]
roidict1[32] = slice(470, 570) 
roidict2[32] = slice(620, 720)
for i in range(32, 200):
	roidict1[i] = roidict1[32]
	roidict2[i] = roidict2[32]


def opal_roi1(detobj, thisEvent):
	return _opal_roi(detobj, thisEvent, roidict1)

def opal_profile1(detobj, thisEvent):
	img = opal_roi1(detobj, thisEvent)
	if img is None:
		return None
	return np.mean(img, axis=0)

def opal_profile2(detobj, thisEvent):
	img = opal_roi2(detobj, thisEvent)
	if img is None:
		return None
	return np.mean(img, axis=0)

def opal_roi2(detobj, thisEvent):
	return _opal_roi(detobj, thisEvent, roidict2)

def _opal_roi(detobj, thisEvent, roidict):
	'''
	Returns only an ROI for the specified image.
	Cannot be used as analysis function.
	Use opal_roi1 and opal_roi2 instead.
	'''
	selfName = detobj['self_name']
	image = detobj[selfName].raw(thisEvent)
	if image is None:
		return None
	run = thisEvent.run()
	if run not in roidict:
		return None
	roi = roidict[run]
	return np.asarray(image)[roi]



def plot_acqiris(detectorObject,thisEvent):
	selfName = detectorObject['self_name']
	y = detectorObject[selfName](thisEvent)[0][1]

	if(None != y):
		to_plot = XYPlot(time.time(),"x vs y", arange(len(y)),y)
		publish.send('my_plot',to_plot)


def plot_acqiris_mpi(detectorObject,thisEvent):

	nevent = detectorObject['event_number']
	comm   = detectorObject['myComm']
	rank   = detectorObject['rank']

	selfName = detectorObject['self_name']
	y = detectorObject[selfName](thisEvent)[0][1]


	all_traces = {}
	all_traces[rank] = y

	if(sum(y)>-430):
		gatheredSummary = comm.gather(y,root=0)
	else:
		gatheredSummary = comm.gather(zeros([15000]),root=0)

	if rank==0:
		for i in gatheredSummary:
			try:
				to_plot = XYPlot(time.time(),"x vs y", arange(len(i)),i)
				publish.send('my_plot',to_plot)
				print(sum(i))
			except:
				pass

		print(len(gatheredSummary))


def getAndorFVB_detCount(detectorObject,thisEvent):
	#IPython.embed()
	selfName = detectorObject['self_name']
	myImage = detectorObject[selfName].raw(thisEvent)
	my_dict = {}

	#IPython.embed()
	if None == myImage:
		my_dict['image'] = zeros(2048)*1.0
		my_dict['photon_count'] = zeros(2048)*1.0
		#print("None")
	else:
		my_dict['image'] = myImage[0]
		#print(myImage.shape)
		temp_image = vstack([zeros(2048),vstack([myImage-median(myImage),zeros(2048)])])
		my_dict['photon_count'] = detectorObject[selfName].photons(thisEvent,nda_calib=temp_image,adu_per_photon=(350-median(myImage)))[1]

	return my_dict
