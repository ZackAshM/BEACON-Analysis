from pathlib import Path
import os
HERE = Path(__file__).parent
from glob import glob
import numpy as np

# data settings
BEACON_DATAPATH = "/SET/YOUR/DATA/PATH/"
RUNS = [int(runpath.split('/')[-1][3:]) for runpath in glob(BEACON_DATAPATH+'run*') if os.path.exists(runpath+'/header.root')]
RUNS.sort()
BEACON_PROCESSED_DATAPATH = "/PATH/FOR/PROCESSED/DATA/"

# DAQ settings
SAMPLE_RATE = 250e6
ADC_RANGE = 127
SAMPLE_SIZE = 512

# channel mappings - {name label : channel label}
SCINT_MAP = {
    1:0,
    2:1,
    3:2,
    4:3,
}
HPOL_MAP = {
    1:12,
    2:13,
    3:14,
    4:15,
    5:8,
    6:9,
}
VPOL_MAP = {
    1:4,
    2:5,
    3:6,
    4:7,
    5:10,
    6:11,
}

# antenna calibrations
RF_HPOL_POS = np.load(HERE / 'calibration/rfhpol.npz')['pos'] # relative to antenna 1
RF_HPOL_DELAYS = np.load(HERE / 'calibration/rfhpol.npz')['cables']
RF_VPOL_POS = np.load(HERE / 'calibration/rfhpol.npz')['pos'] # relative to antenna 1; approx the same as HPOL
RF_VPOL_DELAYS = np.load(HERE / 'calibration/rfhpol.npz')['cables']
SCINT_POS_CORNERS = np.load(HERE / 'calibration/scint.npz')['pos_corners']
SCINT_POS = np.load(HERE / 'calibration/scint.npz')['pos_centers']
SCINT_DELAYS = np.load(HERE / 'calibration/scint.npz')['cables']
SCINT_TILTS_NS = np.load(HERE / 'calibration/scint.npz')['tilts'].T[0]
SCINT_TILTS_EW = np.load(HERE / 'calibration/scint.npz')['tilts'].T[1]
SCINT_DIMENSIONS = np.load(HERE / 'calibration/scint.npz')['dimensions'] # sensitive area, not including enclosure
SCINT_FACE_NORMALS = np.load(HERE / 'calibration/scint.npz')['face_normals']

# analysis parameters
MAP_RESOLUTION = 0.25 # degree bin size of reconstruction maps
FARFIELD_SOURCE_DISTANCE = 1e6 # distance from source assumed during reconstruction
RF_RECON_REQUISITE = 4 # how many RF channels are required for a reconstruction attempt
RF_RMS_SPLIT = 8 # when calculating SNR, the RMS used will be the minimum RMS of those calculated in this many subwindows of the waveform
RF_UPSAMPLE_FACTOR = 100 # how much to upsample RF waveforms during reconstruction
SCINT_RECON_REQUISITE = 3 # how many scint channels are required for a reconstruction attempt
SCINT_EXCLUSION_THRESHOLD = int(0.02*ADC_RANGE) # if a scint amplitude is lower than this, it is excluded from reconstruction
