#natives
import json
import os
import time as pytime

#numpy, scipy, matplotlib
import numpy as np 
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import caiman as cm
from numpy import loadtxt

#opencv
import cv2
#cv = cv2.cv

#custom
from core.daq import ArduinoSerial, ArduinoTrigger
from pseyepy import Camera, Stream

class Experiment(object):
    def __init__(self, name=None, camera=None, daq=None, mask_names=('WHEEL','EYE'), movement_query_frames=10,
                 movement_std_thresh=10, eyelid_thresh=0, trigger_cycle=None, inter_trial_min=5.0, n_trials=-1,
                 resample=1, monitor_vals_display=100,
                 CS_start = 4000, CS_end = 4500, US_start = 4250, US_end = 4280, T_end = 10000, pseudorandom=False):

        self.name = name
        self.pseudorandom = pseudorandom
        self.CS_start = CS_start
        self.CS_end = CS_end
        self.US_start = US_start
        self.US_end = US_end
        self.T_end = T_end
        if type(camera) == Camera:
            self.camera = camera
            self.camera.read()
        else:
            raise Exception('No valid camera supplied.')

        if type(daq) == ArduinoTrigger:
            self.daq = daq
        elif daq == None:
            self.daq = ArduinoSerial()
            self.daq.set_parameters(CS_start=CS_start, CS_end=CS_end, US_start=US_start, US_end=US_end, T_end=T_end)
        else:
            raise Exception('No valid Arduino supplied.')

        # Set static parameters
        self.trigger_cycle = trigger_cycle
        self.mask_names = mask_names
        self.resample = resample
        self.movement_query_frames = movement_query_frames
        self.monitor_vals_display = monitor_vals_display

        # Set variable parameters
        self.param_names = ['movement_std_threshold', 'eyelid_threshold',
                            'inter_trial_min', 'wheel_translation','wheel_stretch',
                            'eye_translation','eye_stretch']
        self.params = {}
        self.params['movement_std_threshold'] = movement_std_thresh
        self.params['eyelid_threshold'] = eyelid_thresh
        self.params['inter_trial_min'] = inter_trial_min
        self.params['wheel_translation'] = 50
        self.params['wheel_stretch'] = 200
        self.params['eye_translation'] = 0
        self.params['eye_stretch'] = 50
        self.params['CS_start'] = CS_start
        self.params['CS_end'] = CS_end
        self.params['US_start'] = US_start
        self.params['US_end'] = US_end
        self.params['T_end'] = T_end


        # Setup interface
        pl.ion()
        self.fig = pl.figure()
        self.ax = self.fig.add_subplot(121)
        self.ax.set_ylim([-1, 255])
        self.plotdata = {m:self.ax.plot(np.arange(self.monitor_vals_display),np.zeros(self.monitor_vals_display), c)[0] for m,c in zip(self.mask_names,['r-','b-'])}
        self.plotline, = self.ax.plot(np.arange(self.monitor_vals_display), np.repeat(self.params['movement_std_threshold'], self.monitor_vals_display), 'r--')
        self.plotline2, = self.ax.plot(np.arange(self.monitor_vals_display), np.repeat(self.params['eyelid_threshold'], self.monitor_vals_display), 'b--')
        self.window = 'Camera'
        self.control = 'Status'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window, 0, 0)
        cv2.moveWindow(self.control, 600, 0)
        self.controls = {'Pause':'p', 'Go':'g', 'Redo':'r', 'Quit':'q', 'Manual Trigger':'t'}
        for pn in self.param_names:
            cv2.createTrackbar(pn,self.control,int(self.params[pn]), 200, self.update_trackbar_params)
        self.update_trackbar_params(self)

        # Set initial variables
        self.masks = {}
        self.mask_idxs = {}
        self.mask_pts = {}

        # Run interactive init
        self.init(trials=n_trials)

    def update_trackbar_params(self, _):
        for param in self.param_names:
            self.params[param] = cv2.getTrackbarPos(param,self.control)
        self.params['wheel_translation'] -= 50
        self.params['eye_translation'] -= 100
        self.params['wheel_stretch'] /= 25.
        self.params['eye_stretch'] /= 25.
        self.plotline.set_ydata(np.repeat(self.params['movement_std_threshold'], self.monitor_vals_display))
        self.plotline2.set_ydata(np.repeat(self.params['eyelid_threshold'], self.monitor_vals_display))

    def update_status(self):
        order = ['Controls','Pause','Go','Redo','Manual Trigger','Quit','Status','Paused','Trials done','Since last',
                 'Last trigger','Eyelid Value','Frame Rate']
        lab_origin = 10
        val_origin = 120
        textsize = 0.4
        textheight = 25
        self.status_img = np.ones((round(textheight*len(order)*1.1),300))*255

        items = self.controls
        items['Controls'] = ''
        items['Status'] = ''
        items['Since last'] = round(pytime.time()-self.last_trial_off, 3)
        items['Trials done'] = self.trial_count
        items['Paused'] = self.TRIAL_PAUSE
        items['Last trigger'] = self.trigger_cycle.current.name
        if len(self.monitor_vals['EYE']):
            if self.monitor_vals['EYE'][-1] > 255. or self.monitor_vals['EYE'][-1] <0.:
                items['Eyelid Value'] = 'ERROR'
            else:
                items['Eyelid Value'] = round(self.monitor_vals['EYE'][-1],2)
        else:
            items['Eyelid Value'] = '(none yet)'
        items['Frame Rate'] = round(self.inst_frame_rate)

        for item in items:
            items[item] = str(items[item])

        for idx,item in enumerate(order):
            cv2.putText(self.status_img,item+':', (lab_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0))
            cv2.putText(self.status_img,items[item], (val_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0))

        cv2.imshow(self.control, self.status_img)

    def init(self, trials):
        if self.name == None:
            self.name = pytime.strftime("%Y%m%d_%H%M%S")
        if os.path.isdir(self.name):
            i = 1
            while os.path.isdir(self.name+'_%i'%i):
                i += 1
            self.name = self.name+'_%i'%i
        os.mkdir(self.name)

        # set up frame rate details
        self.last_timestamp = pytime.time()
        self.inst_frame_rate = 0

        # set up trial count
        self.trials_total = trials
        if trials == -1:
            self.trials_total = 10**3
        self.trial_count = 0

        # ask user for masks and set them
        if len(self.masks)==0:
            self.set_masks()
        self.save_masks()

        # setup containers for acquired data
        self.monitor_img_set = np.empty((self.camera.resolution[0][1],self.camera.resolution[0][0],self.movement_query_frames))
        self.monitor_img_set[:] = None
        self.monitor_vals = {m:np.empty(self.monitor_vals_display) for m in self.mask_names}
        for m in self.monitor_vals:
            self.monitor_vals[m][:] = None

        self.TRIAL_ON = False
        self.TRIAL_PAUSE = False
        self.last_trial_off = pytime.time()
        self.frame_count = 0

        self.update_status()
        # run some initial frames
        for _ in range(self.movement_query_frames):
            self.next_frame()

    def update_framerate(self, timestamp):
        fr = 1/(timestamp - self.last_timestamp)
        self.inst_frame_rate = fr
        self.last_timestamp = timestamp

    def save_masks(self):
        np.save(os.path.join(self.name,'masks'), np.atleast_1d([self.masks]))

    def set_masks(self):
        for m in self.mask_names:
            frame, timestamp = self.camera.read()
            pl.figure()
            pl.title("Select mask: %s."%m)
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
            self.mask_pts[m] = np.array(pts, dtype=np.int32)
            self.masks[m] = mask
            self.mask_idxs[m] = np.where(mask==False)

    def end(self):
        try:
            self.camera.release()
            self.daq.release()
            cv2.destroyAllWindows()
            pl.close(self.fig)
        except:
            pass

    def query_for_trigger(self):
        if pytime.time()-self.last_trial_off < self.params['inter_trial_min']:
            return False
        return (self.monitor_vals['WHEEL'][-1] < self.params['movement_std_threshold']) and (self.monitor_vals['EYE'][-1] < self.params['eyelid_threshold'])

    def monitor_frame(self, frame, masks=('WHEEL', 'EYE'), show=True):
        if 'WHEEL' in masks:
            if None in self.monitor_img_set:
                return
            self.monitor_img_set = np.roll(self.monitor_img_set, 1, axis=2)
            self.monitor_img_set[:,:,0] = frame
            pts = self.monitor_img_set[self.mask_idxs['WHEEL'][0],self.mask_idxs['WHEEL'][1],:]
            std_pts = np.std(pts, axis=1)
            wval = np.mean(std_pts) * self.params['wheel_stretch'] + self.params['wheel_translation']
            self.monitor_vals['WHEEL'] = np.roll(self.monitor_vals['WHEEL'], -1)
            self.monitor_vals['WHEEL'][-1] = wval
        if 'EYE' in masks:
            pts = frame[self.mask_idxs['EYE'][0],self.mask_idxs['EYE'][1]]
            eyval = np.mean(pts) * self.params['eye_stretch'] + self.params['eye_translation']
            self.monitor_vals['EYE'] = np.roll(self.monitor_vals['EYE'], -1)
            self.monitor_vals['EYE'][-1] = eyval

        if show:
            self.update_plots()




    def normalize(self, val, oldmin, oldmax, newmin, newmax):
        return ((val-oldmin)/oldmax) * (newmax-newmin) + newmin

    def update_plots(self):
        toshow_w = np.array(self.monitor_vals['WHEEL'])
        toshow_e = np.array(self.monitor_vals['EYE'])
        if len(toshow_w) != self.monitor_vals_display:
            toshow_w = np.append(toshow_w, np.repeat(None, self.monitor_vals_display-len(toshow_w)))
            toshow_e = np.append(toshow_e, np.repeat(None, self.monitor_vals_display-len(toshow_e)))
        self.plotdata['WHEEL'].set_ydata(toshow_w)
        self.plotdata['EYE'].set_ydata(toshow_e)
        self.fig.canvas.draw()

    def next_frame(self):

        if self.TRIAL_ON:
            # frames, timestamps = self.queue.get()
            # timestamp = timestamps[-1]
            # frame = frames[-1]
            qq=pytime.time()
            # self.writer.write(frame)
            
            # self.monitor_frame(frame, masks=('EYE'), show=False)
        else:
            frame, timestamp = self.camera.read()

            self.update_framerate(timestamp)
            self.frame_count += 1

            if not self.frame_count % self.resample:
                if not self.TRIAL_ON and not self.TRIAL_PAUSE:
                    self.monitor_frame(frame, masks=('WHEEL','EYE'))
                frame_show = np.copy(frame)
                cv2.polylines(frame_show, [self.mask_pts[m] for m in self.mask_names], 1, (255,255,255), thickness=1)
                cv2.imshow(self.window, frame_show)

    def send_trigger(self):
        self.daq.trigger(self.trigger_cycle.next)
        print(("Sent trigger #%i"%(self.trial_count+1)))

    def start_trial(self):
        self.TRIAL_ON = pytime.time()
        self.trial_count += 1
        
        
        self.filename = os.path.join(self.name,'trial%i'%(self.trial_count))
        if os.path.isfile(self.filename):
            i = 1
            while os.path.isfile(os.path.join(self.name,'trial%i_redo%i'%(self.trial_count,i))):
                i += 1
            self.filename = os.path.join(self.name,'trial%i_redo%i.npz'%(self.trial_count,i))



        # self.writer = cv2.VideoWriter(self.filename+'.avi',0,self.inst_frame_rate,frameSize=self.camera.resolution,isColor=False)
        self.writer = Stream(self.camera,file_name=self.filename+'.avi', codec='png', display=False)
        self.queue = self.writer.ques['file']
        
        self.monitor_img_set = np.empty((self.camera.resolution[0][1],self.camera.resolution[0][0],self.movement_query_frames))
        self.monitor_img_set[:] = None
        self.monitor_vals = {m:np.empty(self.monitor_vals_display) for m in self.mask_names}
        for m in self.monitor_vals:
            self.monitor_vals[m][:] = None

    def end_trial(self):
        self.TRIAL_ON = False
        self.last_trial_off = pytime.time()
        self.time.append(self.last_trial_off)
        self.writer.end()
        if self.pseudorandom:
            puff_dur = self.US_end - self.US_start
            US_start = np.int(self.US_start + (self.T_end - self.US_start - puff_dur)*np.random.random())
            US_end = US_start + puff_dur
            self.daq.set_parameters(CS_start=self.CS_start, CS_end=self.CS_end, US_start=US_start, US_end=US_end,
                                    T_end=self.T_end)

        pytime.sleep(3)
        np.savez_compressed(self.filename + '.npz', time=self.time, trigger_type=self.trigger_cycle.current.metadata())
        print(self.filename+'_0.avi')
        m = cm.load(self.filename+'_0.avi')
        trace = np.mean((1 - self.masks[self.mask_names[1]]) * (m), axis=(1, 2))
        time = loadtxt(self.filename+'_0_time.txt')
        time = time - time[0] - self.params['US_start']/1000
        self.fig.add_subplot(122)
        pl.cla()
        pl.plot(time, trace)
        pl.plot(-(self.params['US_start']-self.params['CS_start'])/1000, 1 ,'r*')
        pl.plot(0, 1, 'g*')
        pl.xlim([-0.8, 0.8])
        del m
        self.filename = None

    def step(self):
        self.next_frame()
        c = cv2.waitKey(1)

        if self.TRIAL_ON:
            if pytime.time()-self.TRIAL_ON >= self.trigger_cycle.current.duration:
                self.end_trial()

        if not self.TRIAL_ON:

            if c == ord('p'):
                self.TRIAL_PAUSE = True
                self.update_status()

            if c == ord('g'):
                self.TRIAL_PAUSE = False

            if c == ord('r'):
                self.trigger_cycle.redo()
                self.trial_count -= 1

            if c == ord('q') or (self.trial_count==self.trials_total):
                return False

            if not self.TRIAL_PAUSE:
                if self.query_for_trigger() or c==ord('t'):
                    self.time = [pytime.time()]
                    self.send_trigger()
                    self.start_trial()
                self.update_status()

        return True

    def run(self):
        cont = True
        while cont:
            cont = self.step()
        self.end()
        print("Experiment ended.")

class TriggerCycle(object):
    def __init__(self, triggers=[]):
        self.triggers = np.array(triggers)
        self.current = ArduinoTrigger(msg='Z',  name='(no trigger yet)')

    @property
    def next(self):
        n = self.triggers[0]
        self.current = n
        self.triggers = np.roll(self.triggers, -1)
        return n

    def redo(self):
        self.triggers = np.roll(self.triggers, 1)
        self.current = self.triggers[-1]

    def metadata(self):
        md = {}
        md['triggers'] = [t.metadata() for t in self.triggers]
        return md

if __name__=='__main__':
    pass
