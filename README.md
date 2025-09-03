# BEACON Analysis

This package is an ensemble of analysis tools for the Beamforming Elevated Array for COsmic Neutrinos (BEACON). 

## Getting Started
To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/zackashm/BEACON-Analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
    **Note (ROOT):** You must have ROOT installed. The version I used was ROOT 6.28/06 with c++ (GCC) 13.2.0. Instructions for installations can be found on the ROOT website.

    **Note (Jupyter Notebook):** Most versions of Jupyter Notebook will work. The notebooks can also be used in other cell-based apps. 
    * To install packages used only in the notebooks, run:
        ```bash
        cd notebooks/
        pip install -r requirements.txt
        ```

## Contents

### analysis
This is the main module for analysis tools.
1. **plotter.py**
    * Tools related to plotting for convenience.
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
    * NOTE!!! This file may contain outdated variables, or may be missing variables that previous files refer to. It is up to the user to update this file as needed.
  
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
* 2025/09/03 - I forgot to add comments on previous updates. But as of this date, there are a suite of tools in plotting and utils, as well as updated reconstuction functions. I have decided to keep notebooks, scripts and cfg/config.py as user-dependent files, meaning they won't be updated in this repo, and should instead be updated or written as the user desires. The current script is merely an example of using the tools in this package.