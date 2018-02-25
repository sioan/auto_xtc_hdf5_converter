from pylab import *
import psana
import IPython
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


######################################################
#######Using the eigen traces#########################
######################################################

try:
	eigen_traces_h5py=h5py.File("eigen_traces_run192.h5")
	#eigen_traces = f['summary/nonMeaningfulCoreNumber0/Acq01/ch1/eigen_wave_forms']
except:
	print("eigen_traces_run192.h5")
	pass


def use_acq_svd_basis(detectorObject, thisEvent):
	selfName = detectorObject['self_name']
	my_results = {}
	config_parameters = {"thresh_hold":0.05,"waveform_mask":arange(1200,1230),"eigen_basis_size":25,"offset_mask":arange(300)}

	if(None is detectorObject[selfName](thisEvent)):
		#fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return None
		
	#x = detectorObject[selfName](thisEvent)[1][0]
	for i in arange(len(detectorObject[selfName](thisEvent)[0])):
		eigen_traces = eigen_traces_h5py["summary/nonMeaningfulCoreNumber0/"+selfName+"/ch"+str(i)+"/norm_eigen_wave_forms"]
		#eigen_traces = eigen_traces_h5py["summary/nonMeaningfulCoreNumber0/"+selfName+"/ch"+str(i)+"/eigen_wave_forms"]
		#eigen_traces = array([eigen_traces[j]/sum(eigen_traces[j]**2)**0.5 for j in arange(len(eigen_traces))])	#not efficient. constantly renormalizing. will optimize later
		#IPython.embed()
		y = detectorObject[selfName](thisEvent)[0][i]
		y -= mean(y[config_parameters['offset_mask']])
		weightings = dot(eigen_traces,y)
		residuals = y-dot(weightings,eigen_traces)
		variance = dot(eigen_traces,residuals)**2
		#approximation in line above comes from eigen_traces is orthogonal matrix, 
		#so dot (eigen_traces.transpose(),eigen_traces) is diagonal. 
		#missing something with number of points and degrees of freedom
		my_results["ch"+str(i)] = {"weightings":weightings,"variance":variance}

	return my_results

######################################################
#######Creating the acqiris eigen basis###############
######################################################
def svd_update(eigen_system,new_vector,config_parameters):
	
	try:
		reconstructed_system = dot(eigen_system['eigen_weightings'], eigen_system['eigen_wave_forms'])
		reconstructed_system = vstack([reconstructed_system,new_vector])

		
		singular_values,svd_lsv = eig(dot(reconstructed_system,reconstructed_system.transpose()))
		
		new_weightings = dot(svd_lsv,diag(singular_values))
		new_eigen_vectors = dot(pinv(new_weightings),reconstructed_system)[:config_parameters["eigen_basis_size"]]

		norm_eigen_vectors = real(array([new_eigen_vectors[i]/sum(new_eigen_vectors[i]**2)**0.5 for i in arange(len(new_eigen_vectors))]))

		eigen_system = {'eigen_weightings':new_weightings,'eigen_wave_forms':new_eigen_vectors,'norm_eigen_wave_forms':norm_eigen_vectors}
	
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

	for i in arange(len(detectorObject[selfName](thisEvent)[0])):

		try:
			eigen_system["ch"+str(i)] = previousProcessing["ch"+str(i)]
		except (KeyError,TypeError) as e:
			try:
				y =  detectorObject[selfName](thisEvent)[0][i]			
				y -= mean(y[config_parameters['offset_mask']])			
				eigen_system["ch"+str(i)]= {'eigen_wave_forms':y,'eigen_weightings':[1],'norm_eigen_wave_forms':[1]}
			except (KeyError,TypeError) as e:
				eigen_system["ch"+str(i)] = {'eigen_wave_forms':None,'eigen_weightings':None,'norm_eigen_wave_forms':None}

	##############################
	###main part of calculation###
	##############################
	new_eigen_system = {}
	for i in arange(len(detectorObject[selfName](thisEvent)[0])):

		
		new_eigen_system["ch"+str(i)] = svd_update(eigen_system["ch"+str(i)],detectorObject[selfName](thisEvent)[0][i],config_parameters)
		

	return new_eigen_system

######################################################
#######End of eigen basis generation##################
######################################################

def generic_acqiris_analyzer(detectorObject,thisEvent):
	selfName = detectorObject['self_name']
	fit_results = {}

	if(None is detectorObject[selfName](thisEvent)):
		#fit_results = {'amplitude':popt[2],'uncertainty_cov':pcov[2,2]}
		return None
		
	x = detectorObject[selfName](thisEvent)[1][0]
	for i in arange(len(detectorObject[selfName](thisEvent)[0])):

		y = detectorObject[selfName](thisEvent)[0][i]


		smoothed_wave = convolve(y,[1,1,1,1,1,1],mode='same')
		initial_peak = argmax(smoothed_wave)	#how to hardcode
		initial_height = smoothed_wave[initial_peak]
		peak_width = 4	############################## tunable parameter

		y_small = y[initial_peak-peak_width:initial_peak+peak_width] - mean(y[:])
		x_small = x[initial_peak-peak_width:initial_peak+peak_width]

		#IPython.embed()
		try:
			popt,pcov = curve_fit(peakFunction,x_small,y_small,p0=[0.0,initial_peak,initial_height])
		
			fit_results['ch'+str(i)] = {"position":popt[1],'amplitude':popt[2],'position_var':pcov[1,1],'amplitude_var':pcov[2,2]}

		except (RuntimeError,TypeError) as e:
			fit_results = None

	return fit_results
