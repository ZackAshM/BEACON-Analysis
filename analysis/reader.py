#! /usr/bin/env python3 

# -------------------------
# Be warned, not all of the information in the comments is necessarily correct.
# This class was originally written by Andrew Zeolla, Dan Southall, and Cosmin Deaconu (This is my guess, at least)
# This class has been revised on Sept 19, 2024 by Zachary Martin (me)
# The changes are mainly minor fixes that reflect the new status of the BEACON instrument, and some code clean up
# -------------------------

import ROOT
import numpy 
import sys
import os
import inspect
import cppyy

# You need to load the libbeaconroot library
# For the following line to work, the shared library must be in 
# your dynamic loader path (LD_LIBRARY_PATH or DYLD_LIBRARY_PATH, accordingly) 
ROOT.gSystem.Load("libbeaconroot.so");

# A class to read in data from a run
#  
#   e.g. d = Reader("/path/to/runs",run_num) 
#
#   To select an entry in the run
#     d.setEntry(65079); 
#   Or, you can select by event number instead 
#     d.setEvent(360000065079); 
#  
#   You can then obtain the corresponding header, event or status objects
#     d.header()
#     d.event() 
#     d.status()  (this gets the status with the closest readout time to the event) 
#    
#   For now, you can look at the c++ doc to see what is in each
#  
#  You can also get numpy arrays of the waveform values and time using 
#    d.wf(channel) 
#    d.t(). (same for all channels)
class Reader:
  '''
  This is the python reader wrapper that allows for pulling the event data for BEACON.
  
  Parameters
  ----------
  base_dir : str
      The directory in which you have BEACON data files stored.
  run : int
      The run number for which you want to access data.
  '''
  def __init__(self,base_dir, run):
    try:
      self.failed_setup = False
      self.run = run; 
      self.base_dir = base_dir

      self.event_file = ROOT.TFile.Open(os.path.join(self.base_dir, "run%d/event.root"%run))
      self.event_tree = self.event_file.Get("event") 
      # self.evt = ROOT.beacon.Event() 
      self.event_entry = -1; 
      # self.event_tree.SetBranchAddress("event",ROOT.addressof(self.evt))

      self.head_file = ROOT.TFile.Open(os.path.join(self.base_dir, "run%d/header.root"%run))
      self.head_tree = self.head_file.Get("header") 
      # self.head = ROOT.beacon.Header(); 
      self.head_entry = -1
      # self.head_tree.SetBranchAddress("header",ROOT.addressof(self.head))
      self.head_tree.BuildIndex("header.event_number") 

      self.status_file = ROOT.TFile.Open(os.path.join(self.base_dir, "run%d/status.root"%run))
      self.status_tree = self.status_file.Get("status") 
      # self.stat= ROOT.beacon.Status(); 
      # self.status_tree.SetBranchAddress("status",self.stat) 
      self.status_tree.BuildIndex("status.readout_time","status.readout_time_ns"); 
      self.status_entry =-1; 

      self.current_entry = 0; 
      
      self.times_ns = numpy.linspace(0, (self.event().getBufferLength() - 1)*4, self.event().getBufferLength()) #Should have spacing of 4 ns
    except Exception as e:
      self.failed_setup = True
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def setEntry(self,i): 
    '''
    Sets the current entry to i.  Other functions such as self.wf and self.status will then pull information for the set event.
    
    Parameters
    ----------
    i : int
        The enetry/eventid to set the state of the reader.
    '''
    try:
      if (i < 0 or i >= self.head_tree.GetEntries()):
        sys.stderr.write("Entry out of bounds!") 
      else: 
        self.current_entry = i; 
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def setEvent(self,i):
    '''
    Sets the current entry in the root tree to i.  Other functions such as self.wf and self.status will then pull information for the set event.
    
    Parameters
    ----------
    i : int
        The enetry/eventid to set the state of the reader.
    '''
    try:
      setEntry(self.head_tree.GetEntryNumberWithIndex(i)) 
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)


  def event(self,force_reload = False):
    '''
    Does the required preparations for the event to be loaded.  By default this does nothing if the event is already properly set.
    
    Parameters
    ----------
    force_reload : bool
        Will force this to reset entry info.
    '''
    try:
      if (self.event_entry != self.current_entry or force_reload):
        self.event_tree.GetEntry(self.current_entry)
        self.event_entry = self.current_entry 
        self.evt = getattr(self.event_tree,"event")
      return self.evt 
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)


  def wf(self,ch = 0, bd = 0):  
    '''
    This will pull the waveform data (returned in adu) for the requested channel.  The channel mapping for the
    2019 prototype is: 
    0: NE, Hpol
    1: NE, Vpol
    2: NW, Hpol
    3: NW, Vpol
    4: SE, Hpol
    5: SE, Vpol
    6: SW, Hpol
    7: SW, Vpol
    
    This is subject to change so always cross reference the run with with those in the know to be sure.
    
    Parameters
    ----------
    ch : int
      The channel you specifically want to read out a signal for.
    '''
    ## stupid hack because for some reason it doesn't always report the right buffer length 
    try:
      ev = self.event() 
      v = numpy.copy(numpy.frombuffer(ev.getData(int(ch),bd), numpy.dtype('float64'), ev.getBufferLength()))
      return v
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def t(self):
    '''
    This will return the timing info for a typical run.  This is predefined to assuming 2ns timing between samples, and is
    calculated rather than measured. 
    '''
    try:
      return self.times_ns
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def header(self,force_reload = False): 
    '''
    This will print out the header info from the root tree.
    
    Parameters
    ----------
    force_reload : bool
      If True, this will ensure the event information is properly set before running the function.
    '''
    try:
      if (self.head_entry != self.current_entry or force_reload): 
        self.head_tree.GetEntry(self.current_entry); 
        self.head_entry = self.current_entry 
        self.head = getattr(self.head_tree,"header")
      return self.head 
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def status(self,force_reload = False): 
    '''
    This will print out the status info from the root tree.
    
    Parameters
    ----------
    force_reload : bool
      If True, this will ensure the event information is properly set before running the function.
    '''
    try:
      if (self.status_entry != self.current_entry or force_reload): 
        self.status_tree.GetEntry(self.status_tree.GetEntryNumberWithBestIndex(self.header().readout_time, self.header().readout_time_ns))
        self.status_entry = self.current_entry
        self.stat = getattr(self.status_tree,"status")

      return self.stat
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)


  def N(self):
    '''
    This will return the number of entries (eventids) in a given run.
    '''
    try:
      return self.head_tree.GetEntries() 
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)
  
  def global_scalers(self):
    ''' 
    Returns the global scalers for a given entry in the status tree
    This version fails for some users working on midway returning:
    module 'cppyy' has no attribute 'll'
    '''
    # uses a workaround to access the array from cppyy.ll
    try:
      return numpy.frombuffer(cppyy.ll.cast['uint16_t*'](stat.global_scalers), dtype=numpy.uint16, count=3)
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def beam_scalers(self):
    ''' 
    Returns the beam scalers for a given entry in the status tree
    This version fails for some users working on midway returning:
    module 'cppyy' has no attribute 'll'
    '''
    try:
      return numpy.frombuffer(cppyy.ll.cast['uint16_t*'](self.stat.beam_scalers), dtype=numpy.uint16, count=60).reshape(3,20)
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)


  def returnTriggerThresholds(self, expected_max_beam=19, plot=False):
    '''
    Given a reader object this will extract the trigger thresholds for each beam.

    If expected_max_beam = None then this will attempt to access beams until the reader returns an error.
    '''
    try:
      failed = False
      beam_index = 0
      while failed == False:
        try:
          if beam_index == 0:
            N = self.status_tree.Draw("trigger_thresholds[%i]:Entry$"%beam_index,"","goff")
            thresholds = numpy.zeros((expected_max_beam + 1, N))
            eventids = numpy.frombuffer(self.status_tree.GetV2(), numpy.dtype('float64'), N).astype(int)
          else:
            N = self.status_tree.Draw("trigger_thresholds[%i]"%beam_index,"","goff")
          
          thresholds[beam_index] = numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N)#numpy.vstack((thresholds,numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N)))
          
          if beam_index is not None:
            if beam_index == expected_max_beam:
              failed=True
          beam_index += 1
        except:
          failed = True

      if plot:
        import matplotlib.pyplot as plt # Can't be imported at the top of the script, it causes problems with other programs that import this class for some reason.
        plt.figure()
        plt.title('Trigger Thresholds')
        for beam_index, t in enumerate(thresholds):
          plt.plot(eventids, t, label='Beam %i'%beam_index)
        plt.xlabel('EntryId / eventid')
        plt.ylabel('Power Sum (arb)')

      return thresholds
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def returnBeamScalers(self, expected_max_beam=19, plot=False):
    '''
    Given a reader object this will extract the beam scalers for each beam as they are presented on monutau.

    If expected_max_beam = None then this will attempt to access beams until the reader returns an error.
    '''
    try:
      failed = False
      beam_index = 0
      while failed == False:
        try:
          if beam_index == 0:
            N = self.status_tree.Draw("readout_time","","goff")
            readout_time = numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
            beam_scalers = numpy.zeros((expected_max_beam + 1, N))
            trigger_thresholds = numpy.zeros((expected_max_beam + 1, N))

          N = self.status_tree.Draw("beam_scalers[0][%i]/10.:trigger_thresholds[%i]"%(beam_index,beam_index),"","goff")
          beam_scalers[beam_index]        = numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N)
          trigger_thresholds[beam_index]  = numpy.frombuffer(self.status_tree.GetV2(), numpy.dtype('float64'), N)

          if beam_index is not None:
            if beam_index == expected_max_beam:
              failed=True
          beam_index += 1
        except:
          failed = True

      if plot:
        import matplotlib.pyplot as plt # Can't be imported at the top of the script, it causes problems with other programs that import this class for some reason.
        plt.figure()
        plt.title('Beam Scalers')
        for beam_index, t in enumerate(beam_scalers):
          plt.plot(readout_time, t, label='Beam %i'%beam_index)
        plt.xlabel('EntryId / eventid')
        plt.ylabel('Hz')

      return beam_scalers, trigger_thresholds, readout_time
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)


  def returnGlobalScalers(self, plot=False):
    '''
    This will return global scalers global_scalers[0], global_scalers[1]/10, global_scalers[2]/10 as they are
    read in and presented on monutau.
    This is currently bugged and doesn't make sense because of the beam index bit.

    global_scalers[2]/10 seems to be indentical to the readout_time and to global_scalers[0]
    and global_scalers[1]/10. seems to just be the index of those.

    '''
    try:
      N = self.status_tree.Draw("global_scalers[0]:global_scalers[1]/10.:global_scalers[2]/10.","","goff")

      global_scalers_0 = numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N)
      global_scalers_1 = numpy.frombuffer(self.status_tree.GetV2(), numpy.dtype('float64'), N) 
      global_scalers_2 = numpy.frombuffer(self.status_tree.GetV3(), numpy.dtype('float64'), N)
      readout_time = numpy.frombuffer(self.status_tree.GetV4(), numpy.dtype('float64'), N).astype(int)

      if plot:
        import matplotlib.pyplot as plt # Can't be imported at the top of the script, it causes problems with other programs that import this class for some reason.
        plt.figure()
        plt.plot(global_scalers_0, label = 'Fast') # Labels taken from Monutau.
        plt.plot(global_scalers_1, label = 'Slow Gated') # Labels taken from Monutau.
        plt.plot(global_scalers_2, label = 'Slow') # Labels taken from Monutau.
        plt.ylabel('Hz')
        plt.xlabel('EntryId / eventid')
        plt.legend()

      return global_scalers_0, global_scalers_1, global_scalers_2
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def returnTriggerInfo(self):
    '''
    This will return global scalers global_scalers[0], global_scalers[1]/10, global_scalers[2]/10 as they are
    read in and presented on monutau.
    '''
    try:

      # The below attempts return nan or inf.  Unsure why. Using a workaround loop below.  
      # triggered_beams = numpy.log2(numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N).astype(int))
      # beam_power = numpy.frombuffer(self.status_tree.GetV2(), numpy.dtype('float64'), N).astype(int)
      #N = self.head_tree.Draw("triggered_beams:beam_power:Entry$","","goff")
      # eventids = numpy.frombuffer(self.status_tree.GetV3(), numpy.dtype('float64'), N).astype(int)
      # N = self.head_tree.Draw("readout_time:readout_time_ns","","goff")
      # readout_time = numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N) + numpy.frombuffer(self.status_tree.GetV1(), numpy.dtype('float64'), N)/1e9
      
      triggered_beams = numpy.zeros(self.N())
      beam_power = numpy.zeros(self.N())
      for eventid in range(self.N()):
        self.setEntry(eventid)
        triggered_beams[eventid] = int(self.header().triggered_beams)
        beam_power[eventid] = int(self.header().beam_power)

      return numpy.log2(triggered_beams).astype(int), beam_power.astype(int)
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)

  def loadTriggerTypes(self):
    try:
      N = self.head_tree.Draw("trigger_type","","goff") 
      trigger_type = numpy.frombuffer(self.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
    except Exception as e:
      print('\nError in %s'%inspect.stack()[0][3])
      print('Error while trying to copy header elements to attrs.')
      print(e)
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)
      raise e
    return trigger_type

  def getTriggerType(self, triggerInt):
    '''
    Return all event entries for the given trigger integer type.
    As of 20240927, trigger types are
        1 = Software (SW)
        4 = Three or More Coincident Scintillators (COIN)
    '''
    return numpy.where(self.loadTriggerTypes()==int(triggerInt))[0]
      