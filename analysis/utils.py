'''Stand-alone helper functions'''

from cfg import config
import numpy as np
from scipy.signal import find_peaks, hilbert#, savgol_filter


def rfsnr(wfm):
    '''
    Calculate snr from an RF waveform. This function uses the following definition:
        (wfm.max - wfm.min) / 2*RMS
    RMS = min(RMS_i); RMS_i = sqrt(sum(wfm_i**2) / N)
    This RMS is calculated for subwindows of the waveform; the number of windows is determined in the config.
    This is done twice, where the second time is over the waveform shifted by half of the subwindow size in order
    to discount biases towards windows that perfectly overlap a pulse.
    '''

    calcRMS = lambda wf: np.std(wf - np.mean(wf))
    
    # the window size over which each rms sample is calculated
    rmsLength = len(wfm) // config.RF_RMS_SPLIT
    
    rolledWfm = np.roll(wfm, int(0.5*rmsLength))

    # get rms samples in each window of the shifted and nonshifted waveform
    rmsSamples = []
    for i in range(config.RF_RMS_SPLIT):
        rmsSamples.append(calcRMS(wfm[i*rmsLength:(i+1)*rmsLength]))
        rmsSamples.append(calcRMS(rolledWfm[i*rmsLength:(i+1)*rmsLength]))

    # now decide the RMS value to represent the noise
    noiseRMS = np.median(rmsSamples)
    
    return (np.max(wfm) - np.min(wfm)) / (2*noiseRMS)


def rfImpulsivity(wfm):
    '''
    Returns the Hilbert envelope, the peak value, the peak location, and the average value of the cumulative power from the peak
    and up to 200ns after. The latter value is scaled to roughly be in the range [0,1] (such that 0 is noise, 1 is impulsive, 
    and negative indicates another pulse(s) later).
    The analytic signal, wfm + i*Hilbert(wfm), is obtained using scipy.signal.hilbert. The envelope is the magnitude.
    '''

    # sample length for 200ns
    t200idx = int( (200 / (1e9/config.SAMPLE_RATE)) * (len(wfm) / config.SAMPLE_SIZE))

    # hilbert envelope
    envelope = np.abs(hilbert(wfm))[1:-1] # exclude edges
    envelope_peak = np.max(envelope)
    envelope_peak_idx = np.argmax(envelope)

    # cumulative power ratio
    total_power = np.sum(envelope[envelope_peak_idx:envelope_peak_idx+t200idx]**2)
    cum_power = np.cumsum(envelope[envelope_peak_idx:envelope_peak_idx+t200idx]**2) / total_power
    impulsivity = (np.mean(cum_power) - 0.5) / 0.5

    return (envelope, envelope_peak, envelope_peak_idx, impulsivity)


def sipmPulse(t, amp=int(0.15*config.ADC_RANGE), t0=int(0.07*config.SAMPLE_SIZE*1e9/config.SAMPLE_RATE), tdecay=3.2, trise=8.9):
    '''Return an approximate SiPM pulse within the time range given.'''
    pulse = np.zeros(len(t))
    pulse[t > t0] = np.exp(-(t[t>t0]-t0) / (trise+tdecay)) - np.exp(-(t[t>t0]-t0) / trise)
    norm_pulse = pulse / pulse.max()
    return amp * norm_pulse


def getPulseWidths(wfm, t=None):
    '''
    Returns an array of delta time widths of each pulse in the given waveform. The width is determined
    by the 0's on either side of the pulse max.

    Parameters
    ----------
    wfm : array
        The voltage array in ADU.
    t : array, optional
        The time array. Uniformed spacing assumed.

    Returns
    -------
    widths : array
        The array of pulse widths. If t is given, it is in the same units. Otherwise, it is a 
        width in index.

    See Also
    --------
    analysis.utils.getPeaks
        For how each pulse is identified.
    '''    

    widths = []
    posWfm = wfm.copy()
    posWfm[posWfm < 0] = 0 # positive only
    peakIdx, peakAmp = getPeaks(wfm)

    for idx in peakIdx:
        leftIdx, rightIdx = findNearestIndices(posWfm, idx, 0) # get indices of 0 nearest to idx in the wfm
        deltaIdx = rightIdx - leftIdx
        if t is None:
            widths.append(deltaIdx)
        else:
            widths.append(deltaIdx * (t[1]-t[0])) # uniform time spacing assumed

    return np.array(widths)


def getRiseDurations(wfm, t=None):
    '''
    Returns an array of time durations of each pulse's rising edge in the given waveform. The duration is defined as
    the time from the peak to the first 0 before it.

    Parameters
    ----------
    wfm : array
        The voltage array in ADU.
    t : array, optional
        The time array. Uniformed spacing assumed.

    Returns
    -------
    durations : array
        The array of rising edge durations. If t is given, it is in the same units. Otherwise, it is a 
        duration in index.

    See Also
    --------
    analysis.utils.getPeaks
        For how each peak is identified.
    '''    

    durations = []
    posWfm = wfm.copy()
    posWfm[posWfm < 0] = 0 # positive only
    peakIdx, peakAmp = getPeaks(wfm)

    for idx in peakIdx:
        leftIdx = findNearestIndices(posWfm, idx, 0)[0] # get indices of 0 nearest to idx in the wfm
        deltaIdx = idx - leftIdx
        if t is None:
            durations.append(deltaIdx)
        else:
            durations.append(deltaIdx * (t[1]-t[0])) # uniform time spacing assumed

    return np.array(durations)


def getPeaks(wfm, width=24):
    '''
    Returns an array of amplitudes and indices associated with peaks in the given waveform.
    Current implementation uses scipy.signal.find_peaks().

    Parameters
    ----------
    wfm : array
        The voltage array in ADU.
    width : number
        The necessary width in ns to resolve peaks. Defaults is 24ns.

    Returns
    -------
    Indices : array
    Amplitudes : array
    '''

    # find peaks at least 24ns apart and greater than 2% of the ADC range (= ~ 2 ADU)
    peaks = find_peaks(wfm, width=int(width*config.SAMPLE_RATE/1e9), height=config.SCINT_EXCLUSION_THRESHOLD)

    peakIndices = peaks[0]
    peakAmplitudes = peaks[1]['peak_heights']

    return (peakIndices, peakAmplitudes)


def getRisetimes(t, wfm, cfd=0.5):
    '''
    Returns an array of risetimes and uncertainties for each peak in the given waveform, determined by a
    constant fraction discriminator. Peaks are determined by analysis.utils.getPeaks().
    Risetimes are calculated by fitting a line through the 10% and 90% amplitude points on the
    rising edge, and calculating the point that passes the cfd*amplitude height.
    If pulses are overlapping, a line fit will be attempted for the 90% point and the local minimum between
    peaks.
    Invalid calculations will return NaN elements.

    Parameters
    ----------
    t : array
        The time array in ns.
    wfm : array
        The voltage array in any units.
    cfd : Real number
        The Constant Fraction Discriminator value.

    Returns
    -------
    risetimes : array
    uncertainties : array
    '''

    risetimes = []
    uncertainties = []
    inds, amps = getPeaks(wfm)
    for j in range(len(inds)):
        i = inds[j]
        a = amps[j]
        try:
            if j == 0: # start looking 100ns before pulse
                i0 = i-int(100*config.SAMPLE_RATE/1e9)
            else: # determine starting point if overlapping with previous pulse
                overlap_below10percent = np.where(wfm[inds[j-1]:i] <= 0.1*a)[0]
                if overlap_below10percent.size == 0: # start at local minimum between pulses
                    i0 = np.where(wfm[inds[j-1]:i] == np.min(wfm[inds[j-1]:i]))[0][-1] + inds[j-1]
                else: # start at the end of the overlap below 10%
                    i0 = overlap_below10percent[-1] + inds[j-1]
    
            # get indices of two points around 10% and 90% of the amplitude along the rising edge (~ within 100ns before peak)
            # if overlapping, just the first point above 10%
            i1 = np.where(wfm[i0:i+1]>=0.1*a)[0][0]+i0
            i2 = np.where(wfm[i0:i+1]>=0.9*a)[0][0]+i0
    
            # handle possible problems that led to picking the same point
            if i1 == i2:
                if j == 0: # something went wrong
                    risetimes.append(np.nan)
                    uncertainties.append(np.nan)
                    continue
                else: # overlap is too extreme, give it another chance
                    if i1 == i: # something wrong
                        risetimes.append(np.nan)
                        uncertainties.append(np.nan)
                        continue
                    else: # manually put small distance between points
                        i2 = i1 + 1
                    
    
            # get risetime from linear fit
            t1, t2 = t[i1], t[i2]
            v1, v2 = wfm[i1], wfm[i2]
            slope = (v2-v1)/(t2-t1)
            if slope == 0:
                risetimes.append(np.nan)
                uncertainties.append(np.nan)
                continue
            risetime = (cfd*a - v2)/slope + t2
            
            
            # calculate uncertainties
    
            # linear fit uncertainties
            d_amp = 1
            d_slope = d_amp / (t2-t1)
            dt_dslope = -(cfd*a - v2) / slope**2
            dt_damp = cfd / slope
    
            # linear approximation error
            lin_a = lambda t_input: slope*(t_input - t2) + v2
            N = len(wfm[i1:i2])
            d_linear = np.abs(t2-t1) * np.sum([0 if wfm[i1+it] < 1 else np.abs(lin_a(t[i1+it]) - wfm[i1+it])/wfm[i1+it] 
                                               for it in range(N)]) / N
    
            # sampling uncertainty
            dT = (1e9 / config.SAMPLE_RATE) / 2
    
            # empirical overlap uncertainty
            if (j == 0) or (overlap_below10percent.size > 0):
                d_overlap = 0
            else:
                overlapMinima = np.where(wfm[inds[j-1]:i] == np.min(wfm[inds[j-1]:i]))[0]
                i_min = overlapMinima[int(len(overlapMinima)/2)] + inds[j-1]
                a_min, t_min = wfm[i_min], t[i_min]
                # average time between min and peaks, scale by ratio of min to pulse amplitude
                d_overlap = (a_min / a) * (np.abs(t_min - t[i]) - np.abs(t_min - t[inds[j-1]])) / 2
    
            fullUncertainty = np.sqrt((dt_dslope*d_slope)**2 + (dt_damp*d_amp)**2 + dT**2 + d_linear**2 + d_overlap**2)
    
            
            risetimes.append(risetime)
            uncertainties.append(fullUncertainty)

        except: 
            # IndexError if somehow doesn't find indices within the range above those amp thresholds
            risetimes.append(np.nan)
            uncertainties.append(np.nan)
            
    assert len(risetimes) == len(uncertainties)
    return (np.array(risetimes), np.array(uncertainties))

    # # OLD
    # znorm_wfm = wfm.copy()#(waveform - np.mean(waveform))/np.std(waveform) # Z-normalization
    # window_size = int(0.2*len(znorm_wfm)) if int(0.2*len(znorm_wfm))%2 == 1 else int(0.2*len(znorm_wfm) + 1)
    # smooth_wfm = savgol_filter(znorm_wfm, window_size, 2)
    # # wfm_deriv = np.gradient(znorm_wfm, t)

    # znorm_wfm[znorm_wfm<0] = 0
    # smooth_wfm = savgol_filter(znorm_wfm, int(0.05*len(znorm_wfm)), 2)
    
    # peakV = smooth_wfm.max()
    # t50indp = np.where(smooth_wfm > peakV/2)[0][0]
    # t50indn = t50indp - 1
    # tp = t[t50indp]
    # tn = t[t50indn]
    # vp = smooth_wfm[t50indp]
    # vn = smooth_wfm[t50indn]
    # t50 = (tp-tn)*(peakV/2 - vn)/(vp - vn) + tn
    # dt50 = 2

    # return (np.array([t50]), np.array([dt50]))


def integratePulses(t, wfm):
    '''
    Returns the integration over each pulse, over the whole waveform, and over only positive values of the waveform. 
    A pulse is identified as a 400ns window around peaks.
    The subsets of indices windowing each pulse are also returned. The integration is NOT corrected for time and gain.
    
    Parameters
    ----------
    t : array
        The time array in ns.
    wfm : array
        The voltage array in ADU.

    Returns
    -------
    pulseIntegrations : array
    pulseIndices : array
    fullIntegration: number
    positiveIntegration : number
    '''

    pulseIntegrations = []
    pulseIndices = []
    for i in getPeaks(wfm)[0]:

        # find positive values in the range 100ns before peak and 300ns after peak
        pulseInds = np.where(wfm[i-int(100*config.SAMPLE_RATE/1e9):i+int(300*config.SAMPLE_RATE/1e9)]>=0)[0]+(i-int(100*config.SAMPLE_RATE/1e9))
        pulseIndices.append(pulseInds)

        # left edge integration
        integration = np.sum([(t[j+1]-t[j])*wfm[j] for j in range(len(pulseInds)-1)])
        pulseIntegrations.append(integration)

    fullIntegration = (t[1]-t[0])*np.sum(wfm)
    posWfm = wfm.copy()
    posWfm[posWfm < 0] = 0
    posIntegration = (t[1]-t[0])*np.sum(posWfm)
    
    return (pulseIntegrations, pulseIndices, fullIntegration, posIntegration)


def findNearestIndices(arr, index, value, margin=1e-5):
    '''Get nearest left and right indices to index corresponding to value in arr within margin'''
    leftIdxs = np.where(np.abs(np.array(arr[:index]) - value) <= margin)[0]
    rightIdxs = np.where(np.abs(np.array(arr[index+1:]) - value) <= margin)[0]
    leftIdx = leftIdxs[-1] if len(leftIdxs) > 0 else index
    rightIdx = rightIdxs[0]+index+1 if len(rightIdxs) > 0 else index
    return (leftIdx, rightIdx)


def skyDist(AzZen1, AzZen2):
    '''
    Returns angular distance in deg from 2 points.
    AzZen = (az, zen) in degrees, where az and zen are either scalars, ndarrays, dask array, pandas Series, etc.
    The returned object will match the type of az and zen.
    '''

    azDeg1, zen1 = AzZen1[0], AzZen1[1]
    azDeg2, zen2 = AzZen2[0], AzZen2[1]
    same_points = (azDeg1 == azDeg2) & (zen1 == zen2)
    
    el1 = np.deg2rad(-1*zen1 + 90) # to elevation in rads
    el2 = np.deg2rad(-1*zen2 + 90)
    az1 = np.deg2rad(azDeg1) # az in rads
    az2 = np.deg2rad(azDeg2)
    
    # https://en.wikipedia.org/wiki/Angular_distance
    cos_dist = np.cos(el1)*np.cos(el2)*np.cos(az1 - az2) + np.sin(el1)*np.sin(el2)
    dist = np.rad2deg(np.arccos(cos_dist))
    if len([dist]) > 1:
        dist[same_points] = 0 # change floating point error NaNs to 0 aka when the points are equal
    
    return dist


def getHorizon(azimuth, slope=config.HORIZON_SLOPE, behind=True):
    '''
    Return the horizon zenith value for a given azimuth in degrees and observation slope.
    By default, it is assumed the horizon behind (i.e. 90 < azimuth < 270) is non-sloped (and thus at 90deg).
    '''
    if behind: # behind is visible / non-sloped
        return np.clip(90 + np.rad2deg(np.arcsin(np.sin(np.deg2rad(slope)) * np.cos(np.deg2rad(azimuth)))), a_min=90, a_max=None)
    else:
        return 90 + np.rad2deg(np.arcsin(np.sin(np.deg2rad(slope)) * np.cos(np.deg2rad(azimuth))))
        

# # No longer used
# def getSimpleContour(image, thresh=0.5):
#     '''Return the contour indices for an image where the values slice the "thresh" percentage of the amplitude'''

#     # convert image to 0 or 1 values based on threshold
#     image = np.array(image, dtype=np.float64)
#     simpleImage = image.copy()
#     simpleImage[simpleImage >= thresh*np.max(simpleImage)] = 1
#     simpleImage[simpleImage < thresh*np.max(simpleImage)] = 0

#     # convolve with simple edge detection kernel
#     edgeDetect = np.convolve(simpleImage.ravel(), np.array([1,-1]), mode='same').reshape(simpleImage.shape)
#     edgesUnsorted = np.array(np.where(edgeDetect != 0))

#     # sort clockwise
#     edge_x, edge_y = edgesUnsorted
#     xhat, yhat = edge_x-np.mean(edge_x), edge_y-np.mean(edge_y)
#     angles = np.arctan2(xhat, yhat)
#     edgesSorted = np.array([[edge_x[i], edge_y[i]] for i in np.argsort(angles)])

#     return edgesSorted

# # old version of snr calc by defining a noise window
# def rfsnr2(wfm, t=None, window=200):
#     '''
#     Calculate snr from an RF waveform. This function uses the following definition:
#         (wfm.max - wfm.min) / 2*RMS
#     RMS = sqrt(sum(wfm**2) / N) 
#     This RMS is calculated for values outside the vicinity of the maximum (assuming a pulse) -30% and +70% 
#     of the window.
#     If not given, t defaults to the index array.
#     '''
#     t = np.arange(len(wfm)) if t is None else t
#     peakInd = np.argmax(np.abs(wfm))
#     noiseWindow = np.where(np.logical_or(t < t[peakInd] - 0.3*window, t > t[peakInd] + 0.7*window))[0]
#     noiseRMS = np.std(np.array(wfm[noiseWindow]) - np.mean(wfm[noiseWindow]))
#     return (np.max(wfm) - np.min(wfm)) / (2*noiseRMS)