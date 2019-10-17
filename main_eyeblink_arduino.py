#%%
from pseyepy import Camera, Display
from core.daq import ArduinoTrigger
from interface_eyeblink_arduino import Experiment, TriggerCycle
#%%
#name = '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Mouse/EBC/testing70'
name = '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Mouse/EBC/Mouse_5/20191017'
#name = '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_158/LEARNING_20190625'
#name = '/Users/agiovann/NEL-LAB Dropbox/NEL/Experiments/Ferret/Ferret_167/RELEARNING_20190721'

cam = Camera(ids=[0], resolution=Camera.RES_SMALL, fps=120, colour=False, gain=10, exposure=25)
#change fps to 90 or 100 to see if it gets better 120 is normal
CS_start = 4000 # in msec
CS_end = 4800
US_start = 4400
US_end = 4430
T_end = 8000        
duration = (T_end*1.05)/1000 # in seconds
min_time_after_trial_end = 32.0 #in seconds

is_extinction = False
CS = ArduinoTrigger(msg='C',duration=duration, name='CS')
US = ArduinoTrigger(msg='U', duration=duration, name='US')
CSUS = ArduinoTrigger(msg='D', duration=duration, name='CSUS')

if is_extinction:
    trigger_cycle = TriggerCycle(triggers=[CS, CS, CS, CS, CS, CS, CS, CS, CS, US,
                                           CS, CS, CS, CS, CS, US, CS, CS, CS, CS,
                                           CS, CS, CS, CS, US, CS, CS, CS, CS, US])
else:
    trigger_cycle = TriggerCycle(triggers=[CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CS, CSUS, US,
                                       CSUS, CSUS, CSUS, CSUS, CSUS, CS, CSUS, CSUS, CSUS, CSUS,
                                       CSUS, US, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CS, CSUS])


exp = Experiment(name=name, camera=cam, trigger_cycle=trigger_cycle, n_trials=-1, inter_trial_min=min_time_after_trial_end,
                 CS_start=CS_start, CS_end=CS_end, US_start=US_start, US_end=US_end, T_end=T_end, pseudorandom=is_extinction)

exp.run() #'q' can always be used to end the run early. don't kill the process
cam.end()

"""
# Important parameters for Experiment object:

camera: a Camera object, see examples
trigger_cycle: the triggers of the experiment, see examples

movement_query_frames: the number of frames in the buffer that calculates wheel movement
inter_trial_min: minimum number of seconds between triggers
n_trials: number of triggers (give -1 for limitless)
resample: look at every n'th frame when performing wheel and eyelid calculations
monitor_vals_display: number of values (eyelid mean and wheel std) to show in window
"""
