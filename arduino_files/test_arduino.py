#%%
import caiman as cm
import pylab as pl
from glob import glob
from numpy import loadtxt
import numpy as np

#%%
fls = glob('FERRETANGEL_TEST/04172019/_1/*.avi')
fls.sort()
for fl in fls:
    m = cm.load(fl)
    time = loadtxt(fl[:-4]+'_time.txt')
    time -= time[0]
    pl.plot(time,m.mean(axis=(1,2)))



