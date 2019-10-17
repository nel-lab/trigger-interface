#%%
import caiman as cm
import pylab as pl
from glob import glob
from numpy import loadtxt
import numpy as np
import os

import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import pandas as pd
import datetime as dt
#%%
def find_CS_onset_mask(base_folder, t_cs=4.0):
    m = cm.load(os.path.join(base_folder,'trial1_0.avi'))
    frame = m[0]
    time = loadtxt(os.path.join(base_folder,'trial1_0_time.txt'))
    pl.imshow(frame, cmap=mpl_cm.Greys_r)
    pts = []
    while not len(pts):
        pts = pl.ginput(0)
        pl.close()
        path = mpl_path.Path(pts)
        mask = np.ones(np.shape(frame), dtype=bool)
        for ridx,row in enumerate(mask):
            for cidx,pt in enumerate(row):
                if path.contains_point([cidx, ridx]):
                    mask[ridx,cidx] = False
            
    pl.plot(time-time[0],((m*(1.0-mask)).mean(axis=(1,2))))
    return mask
#%%
time_cs = 4.0
base_folder = 'FERRETANGEL_TEST/01052019_SESS02/'
mask_trig = find_CS_onset_mask(base_folder,t_cs=time_cs) 
#%%
fls = glob(os.path.join(base_folder,'*.avi'))
masks_file = os.path.join(base_folder,'masks.npy')
masks = np.load(masks_file)
mask_eye = masks[0]['EYE']
traces = []
times = []
tm0s = []
time_orig = []
trig_type = [] 
for nfl in range(len(fls)):
    fl = os.path.join(base_folder,'trial' + str(nfl+1) + '_0.avi')
    fl_trig = os.path.join(base_folder,'trial' + str(nfl+1) + '.npz')
    m = cm.load(fl)
    print('Processing:' + fl + ' and ' + fl[:-4]+'_time.txt')
    with np.load(fl_trig) as ld:    
        trig_type.append(ld['trigger_type'][()]['name'])

#            print(str(ld['time'][0]) + ',' + ld['trigger_type'][()]['name'])
#            tm0s.append(ld['time'][0])
    # mtemp = (1-mask_eye)*m[:50]
    # mtemp = mtemp[mtemp>0]
    # iqrs = np.percentile(mtemp, (25,75))
    # thresh = np.median(mtemp)
    trace = np.mean((1-mask_eye)*(m),axis=(1,2))
    
    
    #trace -= trace[100].mean()
    traces.append(trace)
    time = loadtxt(fl[:-4]+'_time.txt')
    time_0 = time-time[0]
    time_orig.append(time_0)
    if trig_type[-1] == 'US':
        time_trig = time[0]
        times.append(time-time_trig-time_cs)
    else:
        trace_trig = np.mean((1-mask_trig)*(m),axis=(1,2))
        idx_trig = np.argmax(np.diff(trace_trig[(time_0>time_cs-1) & (time_0<time_cs+1)]))
        time_trig = time[(time_0>time_cs-1) & (time_0<time_cs+1)][idx_trig]
        times.append(time-time_trig)
        #time -= time[0] + np.mean(np.diff(time))
        pl.plot(time-time_trig,trace_trig)

#%%
np.savez(os.path.join(base_folder,'results_analysis.npz'),traces = traces, times = times, tm0s =  tm0s, time_orig = time_orig, trig_type = trig_type)
#%%
for tr, tm in zip(np.copy(traces), np.copy(time_orig)):
#    pl.plot(np.diff(tm))
    pl.plot(tm, tr-np.median(tr[:100]))
#%%

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
#%%
#%%
pl.plot((combined.values),'c')
#%%
pl.plot(np.mean(combined.values[:,:10],1),'r')
pl.plot(np.mean(combined.values[:,-10:],1),'b')
pl.plot(np.mean(combined.values[:,:],1),'g')

#%%
fls = [
       #'/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/24042019_SESS02_2/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/25042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/26042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/27042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/28042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/28042019_SESS02/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/28042019_SESS03/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/29042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/29042019_SESS02/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/30042019_SESS01/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/30042019_SESS02/results_analysis.npz',
#       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/05012019_SESS01/results_analysis.npz',
       '/Users/agiovann/Dropbox/NEL/Experiments/Ferret/trigger-interface/FERRETANGEL_TEST/SOUND_20190502_SESS_01/results_analysis.npz']


#%%
start = dt.datetime(year=2000,month=1,day=1)
for idx,fl in enumerate(fls):
    pl.subplot(3,1,idx+1)
    with np.load(fl) as ld:
        locals().update(ld)
        counter = 0
        timeseries = dict()  # timeseries are collected in a dictionary
        timeseries_whisk = dict()
        for tr, tr_whisk,  tm, ttype in zip(np.copy(traces), np.copy(traces_whisk), np.copy(times), trig_type):
            time_ = tm
            floatseconds = list(map(float, time_))  # str->float
            datetimes = list(map(lambda x: dt.timedelta(seconds=x) + start, floatseconds))
            data = tr - np.median(tr[400:600])
            data_whisk = tr_whisk - np.median(tr_whisk[400:600])
            data_whisk = np.hstack([data_whisk[0], data_whisk])
            t_s = pd.Series(data, index=datetimes, name='trial_' + str(counter))
            t_s_w = pd.Series(data_whisk, index=datetimes, name='trial_' + str(counter))

            if ttype == 'CSUS':
                timeseries['trial_' + str(counter)] = t_s
                timeseries_whisk ['trial_' + str(counter)] = t_s_w
                counter += 1
            # convert timeseries dict to dataframe
        dataframe = pd.DataFrame(timeseries)
        dataframe_whisk = pd.DataFrame(timeseries_whisk)
        
        combined = dataframe.apply(
            pd.Series.interpolate,
            args=('time',)
        )
        combined_whisk = dataframe_whisk.apply(
            pd.Series.interpolate,
            args=('time',)
        )
        
        pl.plot(np.mean(combined.values[:, :], 1))
        pl.plot(np.mean(combined_whisk.values[:, :], 1))
#%%
import scipy
num_samples = np.int(150 * 7)
time_vec = np.linspace(-3.5,3.5, num_samples)
day_name = []

for idx,fl in enumerate(fls[:]):
    trs_whisk = []
    trs_eye = []
    day_name.append(fl.split('/')[-2])
    with np.load(fl) as ld:
        locals().update(ld)
        counter = 0
        timeseries = dict()  # timeseries are collected in a dictionary
        timeseries_whisk = dict()
        for tr, tr_whisk,  tm, ttype in zip(np.copy(traces), np.copy(traces_whisk), np.copy(times), trig_type):
            time_ = tm
            if ttype == 'CSUS':
                tr = tr[np.where((time_>-3.5) & (time_<3.5))]
                tr_eye = scipy.signal.resample(tr, num_samples)
                tr_eye -= np.mean(tr_eye[(time_vec>-1) & (time_vec<0)])
                tr_whisk = tr_whisk[np.where((time_>-3.5) & (time_<3.5))]
                tr_whisk = scipy.signal.resample(tr_whisk, num_samples)
                tr_whisk /= np.max(tr_whisk)
                if np.max(tr_whisk[(time_vec>-1) & (time_vec<0)])<0.5:  
                    trs_whisk.append(tr_whisk)
                    trs_eye.append(tr_eye)
            
                counter+=1
                if counter>50:
                    continue
                
            
            
            
            
        avg_eye = np.median(np.array(trs_eye),0)
        pl.plot(time_vec,avg_eye/np.max(avg_eye))
            
pl.legend(day_name)        
#%%
pl.imshow(np.array(trs_eye), aspect='auto')
#%%
for tr, tm, ttype in zip(np.copy(traces), np.copy(times),trig_type):
#    pl.plot(np.diff(tm))
    print(ttype)
    if ttype == 'CSUS':
        pl.subplot(3,1,1)
        pl.plot(tm, tr-np.median(tr[(tm>-0.2) &  (tm<0)]),'c')
        pl.xlim([-0.1, .6])
    elif ttype == 'CS':
        pl.subplot(3,1,2)
        pl.plot(tm, tr-np.median(tr[(tm>-0.2) &  (tm<0)]),'c')
        pl.xlim([-0.1, .6])
    elif ttype == 'US':
        pl.subplot(3,1,3)
        pl.plot(tm, tr-np.median(tr[(tm>-0.2) &  (tm<0)]),'c')
        pl.xlim([-0.1, .6])

#

#%%
frame = m[0]
pl.imshow(frame, cmap=mpl_cm.Greys_r)
pts = []
while not len(pts):
    pts = pl.ginput(0)
pl.close()
path = mpl_path.Path(pts)
mask = np.ones(np.shape(frame), dtype=bool)
for ridx,row in enumerate(mask):
    for cidx,pt in enumerate(row):
        if path.contains_point([cidx, ridx]):
            mask[ridx,cidx] = False
            
pl.plot(tm[1:]-tm[0],((m*(1.0-mask)).mean(axis=(1,2))))