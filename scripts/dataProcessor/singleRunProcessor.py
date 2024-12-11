"""
This script processes BEACON data for a single run.

Note: Invalid values will be entered as NaN.
"""

# ---------------------------------------------------------
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))

# config imports
from cfg import config

# analysis imports
from analysis import reconstruction, utils
from analysis.reader import Reader
from analysis.sinesubtraction import SineSubtract

# standard imports
import numpy as np
np.seterr(divide='raise', invalid='raise')
import pandas as pd
import datetime

# commandline arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('run') 
parser.add_argument('fileNameAppend') 
run = int(parser.parse_args().run)
fileNameAppend = str(parser.parse_args().fileNameAppend)

# ---------------------------------------------------------

print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] PROCESSING RUN {run}...')

# ---------------------------------------------------------
# ---------------------------------------------------------

# Define Data Variables --> eventually split into 1. Cut Metrics and 2. Descriptive Metrics

# -- General
_RUNS = []
_ENTRIES = []
_TRIG_TYPES = []
_TIMES = []

# -- Scint
_SC_SCORES = []
_SC_ZENITHS = []
_SC_AZIMUTHS = []
_SC_ZEN_SPREADS = []
_SC_AZ_SPREADS = []
_SC_AMPLITUDE_RANGES = []
_SC_AMPLITUDE_MEANS = []
_SC_PEAKS_MINS = [] # number of peaks
_SC_PEAKS_MAXS = []
_SC_PEAKS_MEANS = []
_SC_ARRIVALTIME_MEANS = [] # average arrival time of each channel's best pulse
_SC_RISETIMES_RANGES = [] # over all risetimes, including multi-peak events
_SC_RISETIMES_MEANS = []
_SC_INTEGRATION_RANGES = [] # full wfm positive integrations
_SC_INTEGRATION_MEANS = []

# -- RF Hpol
_RFH_SCORES = []
_RFH_ZENITHS = []
_RFH_AZIMUTHS = []
_RFH_ZEN_SPREADS = []
_RFH_AZ_SPREADS = []
_RFH_SNR_MINS = []
_RFH_SNR_MAXS = []
_RFH_SNR_MEANS = []
_RFH_COHERENT_SNRS = []
_RFH_TIMEDELAYS_RANGES = []
_RFH_TIMEDELAYS_MEANS = []
_RFH_HILBERT_PEAK_RANGES = []
_RFH_HILBERT_PEAK_MEANS = []
_RFH_HILBERT_PEAK_LOCATION_MINS = [] # ratio of peak index to size of wfm (i.e. penalize when closer to 1)
_RFH_HILBERT_PEAK_LOCATION_MAXS = []
_RFH_HILBERT_PEAK_LOCATION_MEANS = []
_RFH_COHERENT_HILBERT_PEAK_LOCATIONS = []
_RFH_IMPULSIVITY_MINS = []
_RFH_IMPULSIVITY_MAXS = []
_RFH_IMPULSIVITY_MEANS = []
_RFH_COHERENT_IMPULSIVITYS = []

# -- RF Vpol
# _RFV_SCORES = []
# _RFV_ZENITHS = []
# _RFV_AZIMUTHS = []
# _RFV_ZEN_SPREADS = []
# _RFV_AZ_SPREADS = []
_RFV_SNR_MINS = []
_RFV_SNR_MAXS = []
_RFV_SNR_MEANS = []
# _RFV_COHERENT_SNRS = []
# _RFV_TIMEDELAYS_RANGES = []
# _RFV_TIMEDELAYS_MEANS = []
_RFV_HILBERT_PEAK_RANGES = []
_RFV_HILBERT_PEAK_MEANS = []
_RFV_HILBERT_PEAK_LOCATION_MINS = []
_RFV_HILBERT_PEAK_LOCATION_MAXS = []
_RFV_HILBERT_PEAK_LOCATION_MEANS = []
# _RFH_COHERENT_HILBERT_PEAK_LOCATIONS = []
_RFV_IMPULSIVITY_MINS = []
_RFV_IMPULSIVITY_MAXS = []
_RFV_IMPULSIVITY_MEANS = []
# _RFV_COHERENT_IMPULSIVITYS = []


# Set up analysis macro-variables

# -- reconstruction map settings
sourceDistance = config.FARFIELD_SOURCE_DISTANCE
resolution = config.MAP_RESOLUTION
phi = np.arange(-180, 180+resolution/2, resolution)
theta = np.arange(0, 180+resolution/2, resolution)

# -- sine subtraction between 0 to 125 MHz
CWsub_broad = SineSubtract(0, 0.125, 0.03)

# -- progress trackers
maxNum = 100000
iter = 1
t0 = datetime.datetime.now()
current = t0

# Begin processing and recording data
try:
    reader = Reader(config.BEACON_DATAPATH, run)
    triggerTypes = reader.loadTriggerTypes()
except:
    print(f'Run {run}: Error reading this run :(')
    del reader
    sys.exit()

t = None
totalEntries = len(triggerTypes)
for entry in range(totalEntries):
    try:

        # status trackers
        # if 100*(iter/maxNum)%10==0:
        #     now = datetime.datetime.now()
        #     dt = now - current
        #     print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} [{(dt.seconds + 1e-6*dt.microseconds):0.3} s]: {100*(iter/maxNum)}%, run {run}, entry {entry}')
        #     current = now
        if iter > maxNum:
            # maxRun = run
            # maxEntry = entry
            break
    
        # set entry
        reader.setEntry(entry)
        if t is None: # same t for all
            t = reader.t()
    
        # General Variables
        hdr = reader.header()
        evt_time = hdr.readout_time
    
        # Reconstruction Variables
        # -- scints
        sc_map, sc_zen, sc_az, sc_arrTimes = reconstruction.mapScint(reader, entry, source_distance=sourceDistance, resolution=resolution, 
                                                                     norm=True, plot=False, plot_contours=False, verbose=0)
        if ~np.isnan(sc_map).any():
            sc_phiMap = np.sum(sc_map, axis=0)
            sc_thetaMap = np.sum(sc_map, axis=1)
            sc_azSpread = len(sc_phiMap[sc_phiMap >= 0.5*np.max(sc_phiMap)])*resolution
            sc_zenSpread = len(sc_thetaMap[sc_thetaMap >= 0.5*np.max(sc_thetaMap)])*resolution
        else:
            sc_phiMap, sc_thetaMap, sc_azSpread, sc_zenSpread = np.nan, np.nan, np.nan, np.nan
            
        
        # -- RF Hpol
        rfH_map, rfH_zen, rfH_az, rfH_timeDelays, _, rfH_coherentwfm = reconstruction.mapRF(reader, entry, "hpol", CWsub=CWsub_broad, mask=None, 
                                                                             source_distance=sourceDistance, resolution=resolution, 
                                                                             upsample_factor=config.RF_UPSAMPLE_FACTOR, plot=False, verbose=0)
        if ~np.isnan(rfH_map).any():
            rfH_phiMap = np.sum(rfH_map, axis=0)
            rfH_thetaMap = np.sum(rfH_map, axis=1)
            rfH_azSpread = len(rfH_phiMap[rfH_phiMap >= 0.5*np.max(rfH_phiMap)])*resolution
            rfH_zenSpread = len(rfH_thetaMap[rfH_thetaMap >= 0.5*np.max(rfH_thetaMap)])*resolution
        else:
            rfH_phiMap, rfH_thetaMap, rfH_azSpread, rfH_zenSpread = np.nan, np.nan, np.nan, np.nan
        if ~np.isnan(rfH_coherentwfm).any():
            rfH_coherentPeakIdx, rfH_coherentImp = utils.rfImpulsivity(rfH_coherentwfm)[2:]
            rfH_coherentHPeakLoc = rfH_coherentPeakIdx / (len(rfH_coherentwfm)-2)
        else:
            rfH_coherentPeakIdx, rfH_coherentImp, rfH_coherentHPeakLoc = np.nan, np.nan, np.nan
    
        # # -- RF Vpol (currently not eligible for a true reconstruction due to no calibration)
        # rfV_map, rfV_zen, rfV_az, rfV_timeDelays, _, rfV_coherentwfm = reconstruction.mapRF(reader, entry, "vpol", CWsub=CWsub_broad, mask=None, 
        #                                                                      source_distance=sourceDistance, resolution=resolution, 
        #                                                                      upsample_factor=config.RF_UPSAMPLE_FACTOR, plot=False, verbose=0)
        # if ~np.isnan(rfV_map).any():
        #     rfV_phiMap = np.sum(rfV_map, axis=0)
        #     rfV_thetaMap = np.sum(rfV_map, axis=1)
        #     rfV_azSpread = len(rfV_phiMap[rfV_phiMap >= 0.5*np.max(rfV_phiMap)])*resolution
        #     rfV_zenSpread = len(rfV_thetaMap[rfV_thetaMap >= 0.5*np.max(rfV_thetaMap)])*resolution
        # else:
        #     rfV_phiMap, rfV_thetaMap, rfV_azSpread, rfV_zenSpread = np.nan, np.nan, np.nan, np.nan
        # if ~np.isnan(rfV_coherentwfm).any():
        #     rfV_coherentPeakIdx, rfV_coherentImp = utils.rfImpulsivity(rfV_coherentwfm)[2:]
        #     rfV_coherentHPeakLoc = rfV_coherentPeakIdx / (len(rfV_coherentwfm)-2)
        # else:
        #     rfV_coherentPeakIdx, rfV_coherentImp, rfV_coherentHPeakLoc = np.nan, np.nan, np.nan
    
        
        # Waveform Variables
        # -- scints
        sc_amps = []
        sc_peaks = []
        sc_risetimes = []
        sc_integrations = []
        sc_channels = np.fromiter(config.SCINT_MAP.values(), dtype=int)
        for sc_ch in sc_channels:
            sc_wfm = reader.wf(sc_ch)
            sc_peakAmps = utils.getPeaks(sc_wfm)[1]
            sc_amps.extend(sc_peakAmps) # amplitudes of all pulses
            sc_peaks.append(len(sc_peakAmps)) # number of peaks
            sc_risetimes.extend(utils.getRisetimes(t, sc_wfm)[0]) # risetimes of all pulses
            sc_integrations.append(utils.integratePulses(t, sc_wfm)[3]) # full wfm positive integration
        # ---- handle empty arrays
        if len(sc_amps) == 0: sc_amps = np.nan
        if len(sc_peaks) == 0: sc_peaks = np.nan
        if len(sc_risetimes) == 0: 
            sc_risetimes = np.nan
        else:
            sc_risetimes = np.array(sc_risetimes)[~np.isnan(sc_risetimes)] # may contain nan even if not empty list
            if len(sc_risetimes) == 0: sc_risetimes = np.nan # if it's empty after removing nans
        
    
        # -- RF Hpol
        rfH_snrs = []
        rfH_peaks = []
        rfH_peakIdxRatios = [] # peak index / sample size
        rfH_imps = []
        for rfH_ch in np.fromiter(config.HPOL_MAP.values(), dtype=int):
            rfH_rawWfm = reader.wf(rfH_ch)
            rfH_wfm = CWsub_broad.CWFilter(t, rfH_rawWfm)[1] # *may not be consistent with sine-subtracted wfm in reconstruction due to randomness in algorithm
            rfH_snrs.append(utils.rfsnr(rfH_wfm))
            rfH_peak, rfH_peakIdx, rfH_imp = utils.rfImpulsivity(rfH_wfm)[1:]
            rfH_peaks.append(rfH_peak)
            rfH_peakIdxRatios.append(rfH_peakIdx / config.SAMPLE_SIZE)
            rfH_imps.append(rfH_imp)
    
        # -- RF Vpol
        rfV_snrs = []
        rfV_peaks = []
        rfV_peakIdxRatios = [] # peak index / sample size
        rfV_imps = []
        for rfV_ch in np.fromiter(config.VPOL_MAP.values(), dtype=int):
            rfV_rawWfm = reader.wf(rfV_ch)
            rfV_wfm = CWsub_broad.CWFilter(t, rfV_rawWfm)[1]
            rfV_snrs.append(utils.rfsnr(rfV_wfm))
            rfV_peak, rfV_peakIdx, rfV_imp = utils.rfImpulsivity(rfV_wfm)[1:]
            rfV_peaks.append(rfV_peak)
            rfV_peakIdxRatios.append(rfV_peakIdx / config.SAMPLE_SIZE)
            rfV_imps.append(rfV_imp)
    
        
        # Data Appends
        # -- General
        _RUNS.append(run)
        _ENTRIES.append(entry)
        _TRIG_TYPES.append(triggerTypes[entry])
        _TIMES.append(evt_time)
        
        # -- Scints
        _SC_SCORES.append( np.max(sc_map) )
        _SC_ZENITHS.append( sc_zen )
        _SC_AZIMUTHS.append( sc_az )
        _SC_ZEN_SPREADS.append( sc_zenSpread )
        _SC_AZ_SPREADS.append( sc_azSpread )
        _SC_AMPLITUDE_RANGES.append( np.max(sc_amps) - np.min(sc_amps) )
        _SC_AMPLITUDE_MEANS.append( np.mean(sc_amps) )
        _SC_PEAKS_MINS.append( np.min(sc_peaks) )
        _SC_PEAKS_MAXS.append( np.max(sc_peaks) )
        _SC_PEAKS_MEANS.append( np.mean(sc_peaks) )
        _SC_ARRIVALTIME_MEANS.append( np.mean(sc_arrTimes) )
        _SC_RISETIMES_RANGES.append( np.max(sc_risetimes) - np.min(sc_risetimes) )
        _SC_RISETIMES_MEANS.append( np.mean(sc_risetimes) )
        _SC_INTEGRATION_RANGES.append( np.max(sc_integrations) - np.min(sc_integrations) )
        _SC_INTEGRATION_MEANS.append( np.mean(sc_integrations) )
        
        # -- RF Hpol
        _RFH_SCORES.append( np.max(rfH_map) )
        _RFH_ZENITHS.append( rfH_zen )
        _RFH_AZIMUTHS.append( rfH_az )
        _RFH_ZEN_SPREADS.append( rfH_zenSpread )
        _RFH_AZ_SPREADS.append( rfH_azSpread )
        _RFH_SNR_MINS.append( np.min(rfH_snrs) )
        _RFH_SNR_MAXS.append( np.max(rfH_snrs) )
        _RFH_SNR_MEANS.append( np.mean(rfH_snrs) )
        _RFH_COHERENT_SNRS.append( utils.rfsnr(rfH_coherentwfm) )
        _RFH_TIMEDELAYS_RANGES.append( np.max(rfH_timeDelays) - np.min(rfH_timeDelays) )
        _RFH_TIMEDELAYS_MEANS.append( np.mean(rfH_timeDelays) )
        _RFH_HILBERT_PEAK_RANGES.append( np.max(rfH_peaks) - np.min(rfH_peaks) )
        _RFH_HILBERT_PEAK_MEANS.append( np.mean(rfH_peaks) )
        _RFH_HILBERT_PEAK_LOCATION_MINS.append( np.min(rfH_peakIdxRatios) )
        _RFH_HILBERT_PEAK_LOCATION_MAXS.append( np.max(rfH_peakIdxRatios) )
        _RFH_HILBERT_PEAK_LOCATION_MEANS.append( np.mean(rfH_peakIdxRatios) )
        _RFH_COHERENT_HILBERT_PEAK_LOCATIONS.append( rfH_coherentHPeakLoc )
        _RFH_IMPULSIVITY_MINS.append( np.min(rfH_imps) )
        _RFH_IMPULSIVITY_MAXS.append( np.max(rfH_imps) )
        _RFH_IMPULSIVITY_MEANS.append( np.mean(rfH_imps) )
        _RFH_COHERENT_IMPULSIVITYS.append( rfH_coherentImp )
        
        # -- RF Vpol
        # _RFV_SCORES.append( np.max(rfV_map) )
        # _RFV_ZENITHS.append( rfV_zen )
        # _RFV_AZIMUTHS.append( rfV_az )
        # _RFV_ZEN_SPREADS.append( rfV_zenSpread )
        # _RFV_AZ_SPREADS.append( rfV_azSpread )
        _RFV_SNR_MINS.append( np.min(rfV_snrs) )
        _RFV_SNR_MAXS.append( np.max(rfV_snrs) )
        _RFV_SNR_MEANS.append( np.mean(rfV_snrs) )
        # _RFV_COHERENT_SNRS.append( utils.rfsnr(rfV_coherentwfm) )
        # _RFV_TIMEDELAYS_RANGES.append( np.max(rfV_timeDelays) - np.min(rfV_timeDelays) )
        # _RFV_TIMEDELAYS_MEANS.append( np.mean(rfV_timeDelays) )
        _RFV_HILBERT_PEAK_RANGES.append( np.max(rfV_peaks) - np.min(rfV_peaks) )
        _RFV_HILBERT_PEAK_MEANS.append( np.mean(rfV_peaks) )
        _RFV_HILBERT_PEAK_LOCATION_MINS.append( np.min(rfV_peakIdxRatios) )
        _RFV_HILBERT_PEAK_LOCATION_MAXS.append( np.max(rfV_peakIdxRatios) )
        _RFV_HILBERT_PEAK_LOCATION_MEANS.append( np.mean(rfV_peakIdxRatios) )
        # _RFV_COHERENT_HILBERT_PEAK_LOCATIONS.append( rfV_coherentHPeakLoc )
        _RFV_IMPULSIVITY_MINS.append( np.min(rfV_imps) )
        _RFV_IMPULSIVITY_MAXS.append( np.max(rfV_imps) )
        _RFV_IMPULSIVITY_MEANS.append( np.mean(rfV_imps) )
        # _RFV_COHERENT_IMPULSIVITYS.append( utils.rfImpulsivity(rfV_coherentwfm)[3] )
    
        
        # delete variables
        del hdr, evt_time
        del sc_map, sc_zen, sc_az, sc_arrTimes, sc_phiMap, sc_thetaMap, sc_azSpread, sc_zenSpread
        del rfH_map, rfH_zen, rfH_az, rfH_timeDelays, rfH_coherentwfm, rfH_phiMap, rfH_thetaMap, rfH_azSpread, rfH_zenSpread, rfH_coherentHPeakLoc, rfH_coherentPeakIdx, rfH_coherentImp
        # del rfV_map, rfV_zen, rfV_az, rfV_timeDelays, rfV_coherentwfm, rfV_phiMap, rfV_thetaMap, rfV_azSpread, rfV_zenSpread, rfV_coherentHPeakLoc, rfV_coherentPeakIdx, rfV_coherentImp
        del sc_amps, sc_peaks, sc_risetimes, sc_integrations, sc_channels, sc_wfm, sc_peakAmps
        del rfH_snrs, rfH_peaks, rfH_peakIdxRatios, rfH_imps, rfH_rawWfm, rfH_wfm, rfH_peak, rfH_peakIdx, rfH_imp
        del rfV_snrs, rfV_peaks, rfV_peakIdxRatios, rfV_imps, rfV_rawWfm, rfV_wfm, rfV_peak, rfV_peakIdx, rfV_imp
        del _
        
        iter += 1
    except Exception as err:
        print(f'Error on Run {run} Entry {entry}:\n{err}')
        continue
        

# delete variables
del reader
del triggerTypes

# Record and save data
BATCH_DATA = pd.DataFrame({
    'run' : _RUNS,
    'entry' : _ENTRIES,
    'type' : _TRIG_TYPES,
    'time' : _TIMES,
    
    'sc score' : _SC_SCORES,
    'sc zen' : _SC_ZENITHS,
    'sc az' : _SC_AZIMUTHS,
    'sc zen spread' : _SC_ZEN_SPREADS,
    'sc az spread' : _SC_AZ_SPREADS,
    'sc amp range' : _SC_AMPLITUDE_RANGES,
    'sc amp mean' : _SC_AMPLITUDE_MEANS,
    'sc peaks min' : _SC_PEAKS_MINS,
    'sc peaks max' : _SC_PEAKS_MAXS,
    'sc peaks mean' : _SC_PEAKS_MEANS,
    'sc arrival time mean' : _SC_ARRIVALTIME_MEANS,
    'sc risetime range' : _SC_RISETIMES_RANGES,
    'sc risetime mean' : _SC_RISETIMES_MEANS,
    'sc integration range' : _SC_INTEGRATION_RANGES,
    'sc integration mean' : _SC_INTEGRATION_MEANS,
    
    'rfH score' : _RFH_SCORES,
    'rfH zen' : _RFH_ZENITHS,
    'rfH az' : _RFH_AZIMUTHS,
    'rfH zen spread' : _RFH_ZEN_SPREADS,
    'rfH az spread' : _RFH_AZ_SPREADS,
    'rfH snr min' : _RFH_SNR_MINS,
    'rfH snr max' : _RFH_SNR_MAXS,
    'rfH snr mean' : _RFH_SNR_MEANS,
    'rfH coherent snr' : _RFH_COHERENT_SNRS,
    'rfH time delay range' : _RFH_TIMEDELAYS_RANGES,
    'rfH time delay mean' : _RFH_TIMEDELAYS_MEANS,
    'rfH h peak range' : _RFH_HILBERT_PEAK_RANGES,
    'rfH h peak mean' : _RFH_HILBERT_PEAK_MEANS,
    'rfH h peak loc min' : _RFH_HILBERT_PEAK_LOCATION_MINS,
    'rfH h peak loc max' : _RFH_HILBERT_PEAK_LOCATION_MAXS,
    'rfH h peak loc mean' : _RFH_HILBERT_PEAK_LOCATION_MEANS,
    'rfH coherent h peak loc' : _RFH_COHERENT_HILBERT_PEAK_LOCATIONS,
    'rfH impulsitivy min' : _RFH_IMPULSIVITY_MINS,
    'rfH impulsivity max' : _RFH_IMPULSIVITY_MAXS,
    'rfH impulsivity mean' : _RFH_IMPULSIVITY_MEANS,
    'rfH coherent impulsivity' : _RFH_COHERENT_IMPULSIVITYS,
    
    # 'rfV score' : _RFV_SCORES,
    # 'rfV zen' : _RFV_ZENITHS,
    # 'rfV az' : _RFV_AZIMUTHS,
    # 'rfV zen spread' : _RFV_ZEN_SPREADS,
    # 'rfV az spread' : _RFV_AZ_SPREADS,
    'rfV snr min' : _RFV_SNR_MINS,
    'rfV snr max' : _RFV_SNR_MAXS,
    'rfV snr mean' : _RFV_SNR_MEANS,
    # 'rfV coherent snr' : _RFV_COHERENT_SNRS,
    # 'rfV time delay range' : _RFV_TIMEDELAYS_RANGES,
    # 'rfV time delay mean' : _RFV_TIMEDELAYS_MEANS,
    'rfV h peak range' : _RFV_HILBERT_PEAK_RANGES,
    'rfV h peak mean' : _RFV_HILBERT_PEAK_MEANS,
    'rfV h peak loc min' : _RFV_HILBERT_PEAK_LOCATION_MINS,
    'rfV h peak loc max' : _RFV_HILBERT_PEAK_LOCATION_MAXS,
    'rfV h peak loc mean' : _RFV_HILBERT_PEAK_LOCATION_MEANS,
    # 'rfV coherent h peak loc' : _RFV_COHERENT_HILBERT_PEAK_LOCATIONS,
    'rfV impulsitivy min' : _RFV_IMPULSIVITY_MINS,
    'rfV impulsivity max' : _RFV_IMPULSIVITY_MAXS,
    'rfV impulsivity mean' : _RFV_IMPULSIVITY_MEANS,
    # 'rfV coherent impulsivity' : _RFV_COHERENT_IMPULSIVITYS,
})

BATCH_DATA.to_pickle(config.BEACON_PROCESSED_DATAPATH/f'processedRun{run}_{fileNameAppend}.pkl')
del BATCH_DATA


tf = datetime.datetime.now() - t0
print(f'---- [{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] RUN {run} FINISHED ----\nTotal Elapsed Time: {tf} | Total Entries: {iter-1} out of {totalEntries}\n')
