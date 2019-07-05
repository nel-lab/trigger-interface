try:
    import PyDAQmx as pydaq
except:
    import csv as pydaq
import numpy as np
from serial import Serial

class ArduinoTrigger(object):
    def __init__(self, msg=[], duration=None, name='noname'):
        self._msg = None
        self.msg = msg
        self.name = name
        self.duration = duration

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, msg):
        self._msg = msg


    def metadata(self):
        md = {}
        md['duration'] = self.duration
        md['msg'] = str(self.msg)
        md['name'] = self.name
        return md



class Trigger(object):
    def __init__(self, msg=[], duration=None, name='noname'):
        self.duration = duration
        self._msg = None
        self.msg = msg
        self.name = name

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, msg):
        if type(msg) == list or type(msg) == np.array:
            self._msg = np.array(msg).astype(np.uint8)
        else:
            self._msg = np.array(msg).astype(np.float64)

    def metadata(self):
        md = {}
        md['duration'] = self.duration
        md['msg'] = str(self.msg)
        md['name'] = self.name
        return md


class DAQ(object):
    ANALOG = 0
    DIGITAL = 1

    def __init__(self, mode, port_digital="Dev1/Port1/Line0:3", port_analog="Dev1/ao0"):
        self.mode = mode
        if self.mode == self.ANALOG:
            self.port = port_analog
            self.minn = 0.0
            self.maxx = 10.0
        elif self.mode == self.DIGITAL:
            self.port = port_digital
        self.clear_trig = Trigger(msg=[0, 0, 0, 0])
        try:
            self.task = pydaq.TaskHandle()
            pydaq.DAQmxCreateTask("", pydaq.byref(self.task))
            if self.mode == self.DIGITAL:
                pydaq.DAQmxCreateDOChan(self.task, self.port, "OutputOnly", pydaq.DAQmx_Val_ChanForAllLines)
            elif self.mode == self.ANALOG:
                pydaq.DAQmxCreateAOVoltageChan(self.task, self.port, "", self.minn, self.maxx, pydaq.DAQmx_Val_Volts,
                                               None)

            pydaq.DAQmxStartTask(self.task)
        except:
            self.task = None
            print("DAQ task did not successfully initialize.")



    def trigger(self, trig):
        if self.task:
            if self.mode == self.DIGITAL:
                pydaq.DAQmxWriteDigitalLines(self.task, 1, 1, 10.0, pydaq.DAQmx_Val_GroupByChannel, trig.msg, None,
                                             None)
                pydaq.DAQmxWriteDigitalLines(self.task, 1, 1, 10.0, pydaq.DAQmx_Val_GroupByChannel, self.clear_trig.msg,
                                             None, None)
            elif self.mode == self.ANALOG:
                pydaq.DAQmxWriteAnalogF64(self.task, 1, 1, 10.0, pydaq.DAQmx_Val_GroupByChannel, trig.msg, None, None)
        else:
            print(("DAQ task not functional. Attempted to write %s." % str(trig.msg)))

    def release(self):
        if self.task:
            pydaq.DAQmxStopTask(self.task)
            pydaq.DAQmxClearTask(self.task)

    def metadata(self):
        md = {}
        md['port'] = self.port

        return md


class ArduinoSerial(object):
    def __init__(self, port='/dev/tty.usbserial-AK06VDQI', baudrate=115200):
        self.clear_trig = Trigger(msg=[])
        #initialize Arduino for task
        try:
            self.serial = Serial(port, baudrate=baudrate)
            self.write('A') # establish connection
            string = self.read()
            if string != 'B':
                print('Invalid Character returned!:' + string)
                raise Exception('Invalid Character returned!')
            else:
                print('Arduino correctly Initialized')

        except:
            self.serial = None
            print("Arduino did not successfully initialize.")

    def write(self, string):
        self.serial.write(string.encode('ASCII'))

    def read(self):
        string = self.serial.read().decode('ASCII')
        return string

    def set_parameters(self, CS_start = 4000, CS_end = 4500, US_start = 4250, US_end = 4280, T_end = 8000):
        string = 'S' + "{:05d}".format(CS_start)
        string += '-' + "{:05d}".format(CS_end)
        string += '-' + "{:05d}".format(US_start)
        string += '-' + "{:05d}".format(US_end)
        string += '-' + "{:05d}".format(T_end)
        self.write(string)
        print('Setting Parameters: ' + string)
        print('Parameters set to:' + self.read())


    def trigger(self, trig):
        if self.serial:
            self.write(trig.msg)
            print(("Sending Via Serial %s." % str(trig.msg)))

        else:
            print(("Serial not functional. Attempted to write %s." % str(trig.msg)))

    def release(self):
        # self.write('R')
        self.serial.close()

