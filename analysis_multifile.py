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
#%% Defin function to extract masks
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
#%% fix multiple runs

if False:
    import shutil
    base_fol_num = 26
    num_file_other_folder = 33
    for i in range(num_file_other_folder):
        fls_time = glob('trial'+str(i+1)+'_0_time.txt') 
        fls_avi = glob('trial'+str(i+1)+'_0.avi')
        fls_npz = glob('trial'+str(i+1)+'.npz')
        print([fls_time, fls_avi,fls_npz]) 
        shutil.copyfile(fls_time[0], '../trial'+str(i+1+base_fol_num)+'_0_time.txt')  
        shutil.copyfile(fls_avi[0], '../trial'+str(i+1+base_fol_num)+'_0.avi') 
        shutil.copyfile(fls_npz[0], '../trial'+str(i+1+base_fol_num)+'.npz')



#%%
all_folders = []
#folders = glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/ExperimentArchive/SOUND_CS/Ferret_172/SOUND_E*')
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/ExperimentArchive/WHISK_CS/2019*'))
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/ExperimentArchive/SOUND_CS/Ferret_168/EB_*/'))
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_167/LEARNING_*/'))
all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Mouse/EBC/Mouse_5/2019101*/'))
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Mouse/EBC/testing100'))
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_160/LEARNING_*/'))
# folders = glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_167/LEARNING_*/')
#folders = glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/EXTINCTION_*/')
#folders = glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/RELEARNING_*/')
#all_folders.append(glob('/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_167/RELEARNING_*/'))
for folders in all_folders:
    folders.sort()
    for ff in folders:
        print(ff)
time_cs = 4.0
#%% select masks for aligning triggers (dirty ad hoc solution)
use_light_sync = False
if False:
    if use_light_sync == True:
        for base_folder in folders:
            pl.figure()
            mask_trig = find_CS_onset_mask(base_folder, t_cs=time_cs)
            pl.pause(.1)
            np.save(os.path.join(base_folder, 'mask_trigger.npy'), mask_trig)
    for base_folder in folders:
        pl.figure()
        mask_whisk = find_CS_onset_mask(base_folder, t_cs=time_cs)
        np.save(os.path.join(base_folder, 'mask_whisk.npy'), mask_whisk)
#%%
if False:
    for base_folder in folders:
        print(base_folder)
        mask_eye = find_CS_onset_mask(base_folder, t_cs=time_cs)
        pl.cla()
        np.save(os.path.join(base_folder, 'mask_eye.npy'), mask_eye)

#%% Extract time traces from eyelid and whiskers selections
kernel = np.ones((5,5),np.uint8)
for base_folder in folders:
    if use_light_sync == True:
        mask_trig = np.load(os.path.join(base_folder, 'mask_trigger.npy'))
    masks_file = os.path.join(base_folder, 'masks.npy')
    mask_eye = np.load(os.path.join(base_folder, 'mask_eye.npy'))
    mask_whisk = np.load(os.path.join(base_folder, 'mask_whisk.npy'))
    traces = []
    traces_whisk = []
    times = []
    tm0s = []
    time_orig = []
    trig_type = []
    if use_light_sync == True:
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
        m_eye = (1 - mask_eye) * m
        iqrs = np.percentile(m_eye[m_eye>0], (25,75))
        thresh = np.median(m_eye[m_eye>0]) + 0.1*(iqrs[1]-iqrs[0])

        trace = np.sum(m_eye>thresh, axis=(1, 2))/np.sum(mask_eye==0)
        trace_whisk = np.mean(np.diff((1 - mask_whisk) * (m)**2, axis=0), axis=(1, 2))

        # trace -= trace[100].mean()
        traces.append(trace)
        traces_whisk.append(trace_whisk)
        time = loadtxt(fl[:-4] + '_time.txt')
        time_0 = time - time[0]
        time_orig.append(time_0)
        #compensating camera problems
        if use_light_sync == True:
            if trig_type[-1] == 'US':
                time_trig = time[0]
                times.append(time - time_trig - time_cs)
            else:

                trace_trig = np.mean((1 - mask_trig) * (m), axis=(1, 2))
                trace_red_light.append(trace_trig)
                idx_trig = np.argmax(np.diff(trace_trig[(time_0 > time_cs - 1) & (time_0 < time_cs + 1)]))
                time_trig = time[(time_0 > time_cs - 1) & (time_0 < time_cs + 1)][idx_trig]
                times.append(time - time_trig)
                pl.plot(time - time_trig, trace)

                # time -= time[0] + np.mean(np.diff(time))
        else:
            times.append(time_0 - time_cs)
            pl.plot(time_0 - time_cs, trace)

        pl.pause(0.01)
        


    np.savez(os.path.join(base_folder, 'results_analysis_binary.npz'), traces_whisk = traces_whisk, traces=traces, time=time, times=times, tm0s=tm0s,
             time_orig=time_orig, trig_type=trig_type)
    pl.cla()
#%%
for folders in all_folders:
    pl.close()
    is_extinction = False
    fls = [os.path.join(fl,'results_analysis_binary.npz') for fl in folders]
    #fls = [
          #'/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/LEARNING_20190612/results_analysis_binary.npz',
          #'/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/LEARNING_20190618_1/results_analysis_binary.npz',
         # '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/LEARNING_20190624/results_analysis_binary.npz'
           # '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/ExperimentArchive/WHISK_CS/20190424_SESS02_2',
            #'/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/ExperimentArchive/WHISK_CS/20190424_SESS02_2','

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
     #]

    import scipy

    day_name = []
    ampl_CRs = []
    ampl_USs = []
    avgs_eye = []
    pl.figure()
    from cycler import cycler
    pl.rcParams['axes.prop_cycle'] = cycler(color=[[i,i,i] for i in np.arange(0.9,0.05,-0.02)])  #grayscale lines
    #pl.rcParams['axes.prop_cycle'] = cycler( color='brgycmk')  #colored lines

    for idx,fl in enumerate(fls):

        trs_whisk = []
        trs_eye = []
        type_trs = []
        day_name.append(fl.split('/')[-2])
        ampls_sess = []
        with np.load(fl, allow_pickle=True) as ld:

            locals().update(ld)
            num_samples = np.round(7 / np.median([np.median(np.diff(tr_)) for tr_ in time_orig]) * 1.2).astype(np.int)
            time_vec = np.linspace(-3.5, 3.5, num_samples)
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
                        tr_eye = tr_eye - np.median(tr_eye[(time_vec>-1) & (time_vec<0)])
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
                            type_trs.append(ttype)
                            ampls_sess.append(np.max(tr_eye[(time_vec >= 0.35) & (time_vec < 0.40)]))

                        counter += 1


                    except Exception as e:
                        print('Failed: ' + str(e))




            ampl_CRs.append(ampls_sess)
            avg_eye = np.mean(np.array(trs_eye),0)
            avgs_eye.append(avg_eye)
            print(np.array(trs_eye).shape)
            pl.plot(time_vec,avg_eye)
            pl.pause(1)


    pl.axvspan(0, 0.8, alpha=0.1, color='green')
    pl.axvspan(0.4, 0.44, alpha=0.5, color='red')
    pl.xlim([-.3,0.9])
    # pl.legend([d[6:14] for d in day_name])
    pl.legend(['session ' + str(d+1) for d in range(len(day_name))])
    #pl.legend(['Start', 'Intermediate', 'End'])  #For naming specific lines

    pl.xlabel('Time from CS onset (s)')
    pl.ylabel('Eyelid Closure')


    pl.show()
    pl.rcParams['axes.prop_cycle'] = cycler( color='brgycmk')
    np.savez(os.path.sep.join(list(folders[0].split(os.path.sep))[:-2] + ['results_analysis_summary.npz']),
             time_vec=np.array(time_vec), trs_eye=np.array(trs_eye),
              avg_eye=np.array(avg_eye), avgs_eye=np.array(avgs_eye),ampl_CRs=np.array(ampl_CRs), type_trs=type_trs)


#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
avg_CRs = []
avg_AMPL = []
name_animal = []
for folders in all_folders:
    with np.load(os.path.sep.join(list(folders[0].split(os.path.sep))[:-2] + ['results_analysis_summary.npz'])) as ld:
        locals().update(ld)
    thr = 0.2
    ampl_CRs = ampl_CRs[:13]
    #thr = 0.1
    ax1 = pl.subplot(2,1,1)
    ax1.plot(np.arange(len(ampl_CRs))+1,[np.median(amp) for amp in ampl_CRs],'.--')
    avg_CRs.append([np.median(amp) for amp in ampl_CRs])
    pl.ylabel('Ampl CRs')
    ax2 = pl.subplot(2,1,2)
    ax2.plot(np.arange(len(ampl_CRs))+1,[np.mean(np.array(amp)>thr) for amp in ampl_CRs],'.--')
    avg_AMPL.append(np.array([np.mean(np.array(amp)>thr) for amp in ampl_CRs]))
    pl.ylabel('% CRs')
    pl.xlabel('Session')
    name_animal.append(folders[0].split(os.path.sep)[-3])

name_animal.append('avg')
ax1.plot(np.arange(len(ampl_CRs))+1,np.median(np.array(avg_CRs),axis=0),'ko-')
ax2.plot(np.arange(len(ampl_CRs))+1,np.median(np.array(avg_AMPL),axis=0),'ko-')
pl.legend(name_animal)
#%%
with np.load(os.path.sep.join(list(all_folders[0][0].split(os.path.sep))[:-2] + ['results_analysis_summary.npz'])) as ld:
    locals().update(ld)
    pl.rcParams['axes.prop_cycle'] = cycler(color=[[i,i,i] for i in np.arange(0.9,0.05,-0.02)])  #grayscale lines
    count = 0
    for av in avgs_eye:
        count += 1
        pl.plot(time_vec[:len(av)], av)

    pl.axvspan(0, 0.8, alpha=0.1, color='green')
    pl.axvspan(0.4, 0.44, alpha=0.5, color='red')
    pl.xlim([-.3, 0.9])
    pl.ylim([-.5, 1.25])
    pl.legend(['sess ' + str(i+1) for i in range(count)])
#%%
pl.plot(time_vec,np.array(np.array(trs_eye)[np.array([ttt=='CSUS' for ttt in type_trs])]).T,color=[0.8,0.8,0.8])
pl.plot(time_vec,avg_eye,'k')
pl.axvspan(0, 0.8, alpha=0.1, color='green')
pl.axvspan(0.4, 0.44, alpha=0.5, color='red')
pl.xlim([-.3,1.2])
pl.ylim([-.3,1.95])

pl.xlabel('Time from CS onset (s)')
pl.ylabel('Eyelid Closure')
#%%


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




#%%
#%%
thr = 0.2
#thr = 0.1


pl.plot(np.arange(len(ampl_CRs))+1,[np.median(amp) for amp in ampl_CRs],'ko-')
pl.ylabel('Ampl CRs')
pl.xlabel('Session')