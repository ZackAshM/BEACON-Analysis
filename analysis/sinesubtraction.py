import os
import numpy as np
import ROOT
ROOT.gInterpreter.ProcessLine('#include "%s"'%(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'include/FFTtools.h'))
ROOT.gSystem.Load(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'build/libRootFftwWrapper.so.3')
from ROOT import FFTtools

class SineSubtract:
    '''
    This is a wrapper class written by KA Hughes for the Sine Subtraction ROOT code written by C Deaconu.  This class
    will attempt time domain subtraction of sine waves from signals, with the goal of removing CW from signals.  It
    is done in the time domain to account for the fact the a time windowed sine wave does not result in a single freq
    in the frequency domain.  
    This code was originally used for ARA phased array analysis, for which reasonable values seemed to be:
    min_power_ratio = 0.1 (originally 0.05)
    min_freq = 0.2 (in GHz)
    max_freq = 0.7 (in GHz)
    Parameters
    ----------
    min_freq : float
        The minium frequency to be considered part of the same known CW source.  This should be given in GHz.  
    max_freq : float
        The maximum frequency to be considered part of the same known CW source.  This should be given in GHz.  
    min_power_ratio : float
        This is the threshold for a prominence to be considered CW.  If the power in the band defined by min_freq and
        max_freq contains more than min_power_ratio percent (where min_power_ratio <= 1.0) of the total signal power,
        then it is considered a CW source, and will be removed. 
    max_failed_iterations : int
        This sets a limiter on the number of attempts to make when removing signals, before exiting.
    '''
    def __init__(self, min_freq, max_freq, min_power_ratio, max_failed_iterations=5, verbose=False):
        self.sine_subtract = FFTtools.SineSubtract(max_failed_iterations, min_power_ratio,False)
        self.sine_subtract.setVerbose(False) #Don't print a bunch to the screen
        self.sine_subtract.setFreqLimits(min_freq, max_freq)

    def CWFilter(self, time, waveform):
        '''        
        '''
        #create time array and calculate dt
        dt = (time[1]-time[0])
        
        #load in voltages and create an output array
        v_out = np.zeros(len(waveform),dtype=np.double)

        #Do the sine subtraction
        self.sine_subtract.subtractCW(len(waveform),waveform,dt,v_out)

        #Check how many solutions were found
        n_fit = self.sine_subtract.getNSines()

        return n_fit, v_out