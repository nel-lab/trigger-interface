#%%
import caiman as cm
import pylab as pl
from glob import glob
from numpy import loadtxt
import numpy as np

import os

import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import datetime as dt
#%%
folders = glob('/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/TEST_ROSS')
folders.sort()
folders = folders[:]
for ff in folders:
    print(ff)
time_cs = 4.0
#%%
def find_CS_onset_mask(base_folder, t_cs=4.0):
    m = cm.load(os.path.join(base_folder, 'trial1_0.avi'))
    frame = m[0]
    time = loadtxt(os.path.join(base_folder, 'trial1_0_time.txt'))
    pl.imshow(frame, cmap=mpl_cm.Greys_r)
    pts = []
    while not len(pts):
        pts = pl.ginput(0)
        pl.close()
        path = mpl_path.Path(pts)
        mask = np.ones(np.shape(frame), dtype=bool)
        for ridx, row in enumerate(mask):
            for cidx, pt in enumerate(row):
                if path.contains_point([cidx, ridx]):
                    mask[ridx, cidx] = False

    pl.plot(time - time[0], ((m * (1.0 - mask)).mean(axis=(1, 2))))
    return mask
#%% select masks for aligning triggers (dirty ad hoc solution)
if False:
    for base_folder in folders:
        pl.figure()
        mask_trig = find_CS_onset_mask(base_folder, t_cs=time_cs)
        pl.pause(.1)
        np.save(os.path.join(base_folder, 'mask_trigger.npy'), mask_trig)

#%% Extract time traces from eyelid and whiskers selections
kernel = np.ones((5,5),np.uint8)
for base_folder in folders:
    #mask_trig = np.load(os.path.join(base_folder, 'mask_trigger.npy'))
    masks_file = os.path.join(base_folder, 'masks.npy')
    times = []
    tm0s = []
    time_orig = []
    trig_type = []
    trace_red_light = []
    fls = glob(os.path.join(base_folder, '*.avi'))

    for nfl in range(len(fls)):
        fl = os.path.join(base_folder, 'trial' + str(nfl + 1) + '_0.avi')
        fl_trig = os.path.join(base_folder, 'trial' + str(nfl + 1) + '.npz')
        m = cm.load(fl)
        print('Processing:' + fl + ' and ' + fl[:-4] + '_time.txt')
        if os.path.exists(fl_trig):
           with np.load(fl_trig, allow_pickle=True) as ld:
               trig_type.append(ld['trigger_type'][()]['name'])
        else:
           continue

        time = loadtxt(fl[:-4] + '_time.txt')
        time_0 = time - time[0]
        time_orig.append(time_0)
        if trig_type[-1] == 'US':
            time_trig = time[0]
            times.append(time - time_trig - time_cs)
        else:
            trace_trig = np.mean((1 - mask_trig) * (m), axis=(1, 2))
            trace_red_light.append(trace_trig)
            idx_trig = np.argmax(np.diff(trace_trig[(time_0 > time_cs - 1) & (time_0 < time_cs + 1)]))
            time_trig = time[(time_0 > time_cs - 1) & (time_0 < time_cs + 1)][idx_trig]
            times.append(time - time_trig)
            # time -= time[0] + np.mean(np.diff(time))
            # pl.plot(time - time_trig, trace)
            pl.pause(0.01)



    pl.cla()
    [pl.plot(ttime, trace) for ttime, trace in zip(time_orig, trace_red_light)]
#%%
pl.subplot(2,1,1)
[pl.plot(ttime,trace_trig,'b') for ttime, trace_trig in zip(time_orig ,trace_red_light)]
pl.subplot(2,1,2)
[pl.plot(trace_trig,'r') for ttime, trace_trig in zip(time_orig ,trace_red_light)]

#%%
is_extinction = True
fls = [os.path.join(fl,'results_analysis_binary.npz') for fl in folders]
# fls = [
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190502_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190503_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190504_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190505_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190506_SESS_01/results_analysis_binary.npz',
#     #   # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190506_SESS_02/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190507_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190508_SESS_01/results_analysis_binary.npz',
#     #   '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190510_SESS_01/results_analysis_binary.npz',
#     #     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190511_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190512_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190513_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190514_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190515_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190516_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190517_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190518_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190519_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190520_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190521_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190522_SESS_01/results_analysis_binary.npz',
#     # '/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190523_SESS_01/results_analysis_binary.npz',
#     '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_EXTINCTION_20190525_SESS_01/results_analysis_binary.npz',
# ]

import scipy
num_samples = np.int(150 * 7)
time_vec = np.linspace(-3.5,3.5, num_samples)
day_name = []
ampl_CRs = []
ampl_USs = []
pl.figure()
from cycler import cycler
pl.rcParams['axes.prop_cycle'] = cycler(color=[[i,i,i] for i in np.arange(0.9,0.05,-0.02)])
# pl.rcParams['axes.prop_cycle'] = cycler( color='brgycmk')

for idx,fl in enumerate(fls):
    trs_whisk = []
    trs_eye = []
    day_name.append(fl.split('/')[-2])
    ampls_sess = []
    with np.load(fl, allow_pickle=True) as ld:
        locals().update(ld)
        counter = 0
        timeseries = dict()  # timeseries are collected in a dictionary
        timeseries_whisk = dict()
        us_size = []
        for tr, tr_whisk,  tm, ttype in zip(np.copy(traces), np.copy(traces_whisk), np.copy(times), trig_type):
            time_ = tm
            if ttype == 'CSUS' or ttype == 'CS':
                try:
                    tr = tr[np.where((time_>-3.5) & (time_<3.5))]
                    tr_eye = scipy.signal.resample(tr, num_samples)
                    tr_eye -= np.median(tr_eye[(time_vec>-1) & (time_vec<0)])
                    if ttype == 'CSUS' or ttype == 'US':
                        if not is_extinction:
                            ampl_USs.append(np.nanmax(tr_eye[(time_vec>=0.42) & (time_vec<0.7)]))
                        else:
                            ampl_USs.append(np.nanmax(tr_eye[(time_vec >= 0.42)]))

                    full_closure = np.nanmedian(ampl_USs[-5:])
                    tr_eye = tr_eye/full_closure

#                    tr_eye /= np.max(tr_eye)
#                    tr_eye = tr_eye/np.max(tr_eye[(time_vec>=0) & (time_vec<0.2)])                    
                    tr_whisk = tr_whisk[np.where((time_>-3.5) & (time_<3.5))]
                    tr_whisk = scipy.signal.resample(tr_whisk, num_samples)
                    tr_whisk /= np.max(tr_whisk)
                    if np.max(tr_whisk[(time_vec>-1) & (time_vec<0)])<1 and np.max(tr_eye)<3:
                        trs_whisk.append(tr_whisk)
                        trs_eye.append(tr_eye)
                        ampls_sess.append(np.max(tr_eye[(time_vec >= 0.35) & (time_vec < 0.42)]))
                
                    counter += 1
                    # if counter>50:
                    #     continue

                except Exception as e:
                    print('Failed: ' + str(e))

            
            
            
        ampl_CRs.append(ampls_sess)
        avg_eye = np.median(np.array(trs_eye),0)

        print(np.array(trs_eye).shape)
        pl.plot(time_vec,avg_eye)
        pl.pause(1)
        

pl.axvspan(0, 0.8, alpha=0.1, color='green')
pl.axvspan(0.4, 0.44, alpha=0.5, color='red')
pl.xlim([-.3,0.9])
# pl.legend([d[6:14] for d in day_name])
pl.legend(['session ' + str(d+1) for d in range(len(day_name))])

pl.xlabel('Time from CS onset (s)')
pl.ylabel('Eyelid Closure')
 

pl.show()
# pl.rcParams['axes.prop_cycle'] = cycler( color='brgycmk')



#%%
thr = 0.3
# thr = 0.1
pl.subplot(2,1,1)

pl.plot(np.arange(len(ampl_CRs))+1,[np.median(amp) for amp in ampl_CRs],'ko-')
pl.ylabel('Ampl CRs')
pl.subplot(2,1,2)
pl.plot(np.arange(len(ampl_CRs))+1,[np.mean(np.array(amp)>thr) for amp in ampl_CRs],'ko-')
pl.ylabel('% CRs')
pl.xlabel('Session')

#%%
pl.plot(time_vec,np.array(trs_eye).T,color=[0.8,0.8,0.8])
pl.plot(time_vec,avg_eye,'k')
pl.axvspan(0, 0.8, alpha=0.1, color='green')
pl.axvspan(0.4, 0.44, alpha=0.5, color='red')
pl.xlim([-.3,0.9])
pl.ylim([-.1,1.2])

pl.xlabel('Time from CS onset (s)')
pl.ylabel('Eyelid Closure')
#%% trace for example
with  np.load('/home/andrea/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/WHISK_CS/30042019_SESS02/results_analysis_binary.npz') as ld:
    locals().update(ld)
    tr_eye = traces[7]/np.max(traces[7][(times[7]>=0) & (times[7]<0.2)])
    pl.plot(times[7],tr_eye) 
    pl.axvspan(0, 0.8, alpha=0.1, color='green')
    pl.axvspan(0.4, 0.44, alpha=0.05, color='red')
    pl.xlim([-.3,0.9])
    pl.xlabel('Time from CS onset (s)')
    pl.ylabel('Eyelid Closure')
#%%
pl.plot(time_vec,np.array(trs_eye).T,'c')
#%%
data_day = dict()
for base_folder in folders:
    day_name = base_folder.split('/')[-1]
    with np.load(os.path.join(base_folder, 'results_analysis.npz')) as ld:
        data_day[day_name] = dict(ld)

np.save(os.path.join('/'.join,base_folder.split('/')[:-2]),'all_days_analysis.npz')
#%%
for base_folder in folders:
    with np.load(os.path.join('/'.join,base_folder.split('/')[:-2]),'all_days_analysis.npz') as ld:
        data_day = ld['data_day'][()]
        
    start = dt.datetime(year=2000,month=1,day=1)
    counter = 0
    timeseries = dict() #timeseries are collected in a dictionary
    
    for tr, tm, ttype in zip(np.copy(traces), np.copy(times),trig_type):
        time_ = tm
        floatseconds = map(float,time_) #str->float
        datetimes = map(lambda x:dt.timedelta(seconds=x)+start,floatseconds)
        data = tr-np.median(tr[400:600])
        t_s = pd.Series(data,index=datetimes,name='trial_'+str(counter))
        if ttype == 'CSUS':
            timeseries['trial_'+str(counter)] = t_s
            counter += 1
        #convert timeseries dict to dataframe
    dataframe = pd.DataFrame(timeseries)
    
    combined = dataframe.apply(
        pd.Series.interpolate,
        args=('time',)
    )




