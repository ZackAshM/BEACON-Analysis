# BEACON Analysis

This package is an ensemble of analysis tools for the Beamforming Elevated Array for COsmic Neutrinos (BEACON). 

## Contents

### analysis
This is the main module for analysis tools.
1. **plotter.py**
    * Tools related to plotting for convenience. Currently empty.
2. **reader.py**
    * Tools for reading raw BEACON data.
3. **reconstruction.py**
    * Tools to perform source direction reconstruction on BEACON scintillator and RF data.
4. **sinesubtraction.py**
    * Tools for continuous-wave (CW) subtraction, or sine subtraction in RF waveforms.
5. **utils.py**
    * Tools for performing particular analysis calculations, such as RF SNR, RF impulsivity, and Scintillator pulse risetimes.

### cfg
This module defines calibration and configuration settings.
1. **calibration/**
    * Contains calibration and information for the scintillator array and RF antenna array.
2. **config.py**
    * Header module containing global variables defining a configuration for the analysis.
  
### scripts
Contains macros for using the analysis package.
1. **dataProcessor**
    * Contains scripts for processing BEACON data.
        * `singleRunProcessor.py` will calculate analysis variables for all events in a single data run. The process can be parallelized among other runs using SLURM, and job submissions are handled by `runAll.sh`.
     
### notebooks
Contains Jupyter Notebooks to provide example code.
1. **analysis.ipynb**
    * Example code for loading and viewing processed data.
2. **examples.ipynb**
    * Example code for using some tools in the analysis package.


## Notable History
* 2024/12/09 - Initial Commit