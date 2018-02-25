%matplotlib
my_eigen_basis = h5py.File("eigen_traces_run192.h5")
eig_bas = array(my_eigen_basis['summary/nonMeaningfulCoreNumber0/Acq01/ch1/norm_eigen_wave_forms'])
#eig_bas = array(my_eigen_basis['summary/nonMeaningfulCoreNumber0/Acq01/ch1/eigen_wave_forms'])
#eig_bas = array([eig_bas[i]/sum(eig_bas[i]**2)**0.5 for i in arange(len(eig_bas))])
import psana
my_data_source = psana.MPIDataSource("exp=sxrlq8415:run=192:smd")
acq_det_obj = psana.Detector("Acq01")
my_enum_events = enumerate(my_data_source.events())
for evt_num,this_event in my_enum_events:
    if evt_num>100:
        break
my_index=argmax((this_event.get(psana.EventId).fiducials() == my_dict['fiducials']).astype(int))
plot(acq_det_obj(this_event)[0][1]-mean(acq_det_obj(this_event)[0][1][:300]))

plot(acq_det_obj(this_event)[0][1]-mean(acq_det_obj(this_event)[0][1][:300]))
plot(dot(my_dict['Acq01/ch1/weightings'][my_index], eig_bas),'r.')
