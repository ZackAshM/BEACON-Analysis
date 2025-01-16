'''Functions for plotting'''

from cfg import config
from analysis.reader import Reader

import numpy as np
from matplotlib import pyplot as plt
import datetime
from collections.abc import Iterable

COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

def plotEventWaveforms(runOrReader, entry, ylims=[None, None, None], monutau=False, CWSubtracter=None, subplotsObject=None, save=''):
    '''
    Plot BEACON event waveforms.

    Parameters
    ----------
    runOrReader : int or analysis.reader.Reader
        Either the integer run number, or the BEACON data Reader object.
    entry : int
        The entry or event number in the given run.
    ylims : array((min,max), 'common', 'full', None)
        The plot y-limits in the order [scint, hpol RF, vpol RF]. In each slot, the following can be set: 
            - If (min, max), then each channel will be set to this y-limit. 
            - If 'common', the ylim for each channel is set to the min,max across all channels (i.e. scint='common' will 
            use the ymin and ymax across all scint channels).
            - If None (default), each channel will be set by its default ylim assumed by matplotlib.pyplot.
    monutau : bool, default False
        If True, plot in same arrangement as monutau (channel oriented), otherwise organize by scint then antenna 
        number (antenna oriented).
    CWSubtracter : analysis.sinesubtraction.SineSubtract, optional
        If given, sine subtract RF waveforms using this SineSubtract object.
    subplotsObject : (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        The Figure and Axes returned from matplotlib.pyplot.subplots. If not given, one will be created.
    save : str
        If given, save the plot with this name.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The used Figure and Axes created by matplotlib.pyplot.subplots.
    '''

    # determine reader and set entry
    reader = runOrReader if isinstance(runOrReader, Reader) else Reader(config.BEACON_DATAPATH, runOrReader)
    reader.setEntry(entry)

    # extract some metadata
    run = reader.run
    hdr = reader.header(force_reload=True)
    evtnum = hdr.event_number
    utcdate = datetime.datetime.fromtimestamp(hdr.readout_time).strftime('%Y-%m-%d %H:%M:%S')

    # labels for each detector
    scintLabels = [f'Scint {scNum}' for scNum in config.SCINT_MAP.keys()]
    hpolLabels = [f'Hpol {hNum}' for hNum in config.HPOL_MAP.keys()]
    vpolLabels = [f'Vpol {vNum}' for vNum in config.VPOL_MAP.keys()]

    # axes subplot index for each detector
    scintIdx = list(config.SCINT_MAP.values())
    hpolIdx = list(config.HPOL_MAP.values()) if monutau else np.arange(1, 2*len(hpolLabels), 2) + scintIdx[-1]
    vpolIdx = list(config.VPOL_MAP.values()) if monutau else np.arange(2, 2*len(vpolLabels)+1, 2) + scintIdx[-1]

    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=[20,12]) if (subplotsObject is None) else subplotsObject
    axesFlat = axes.flatten()

    time = reader.t()

    # def plotting
    def plot(reader, t, subtractCW, channels, axIndices, labels, flatAxes, ylims, color):
        vlims = [0,0] # track waveform min and max across all channels
        for ch, idx, lab in zip(channels, axIndices, labels):
            
            # get waveform
            if subtractCW and not (CWSubtracter is None):
                wfm = CWSubtracter.CWFilter(time, reader.wf(ch))[1]
            else:
                wfm = reader.wf(ch)
    
            # update wfm limits
            vlims[0] = min((min(wfm), vlims[0])) 
            vlims[1] = max((max(wfm), vlims[1]))
    
            # plot
            flatAxes[idx].plot(t, wfm, c=color, lw=2)
            flatAxes[idx].set(title=f'{lab}, Ch {ch}')

            # ylim handling: (min, max)
            if (isinstance(ylims, Iterable)) and (not isinstance(ylims, str)):
                flatAxes[idx].set(ylim=[ylims[0],ylims[1]])
        # ylim handling: max across all channels
        if ylims == 'common':
            for idx in axIndices:
                flatAxes[idx].set(ylim=(vlims[0]*1.02-1, 1.02*vlims[1]+1)) # 2% padding

    # plot
    plot(reader, time, False, config.SCINT_MAP.values(), scintIdx, scintLabels, axesFlat, ylims[0], COLORS[1]) # scint
    plot(reader, time, True, config.HPOL_MAP.values(), hpolIdx, hpolLabels, axesFlat, ylims[1], COLORS[0]) # hpol
    plot(reader, time, True, config.VPOL_MAP.values(), vpolIdx, vpolLabels, axesFlat, ylims[2], COLORS[2]) # vpol

    # all axes options
    for ax in axesFlat:
        ax.set(xlabel='Time [ns]', ylabel='Voltage [ADU]')
        ax.grid()

    fig.suptitle(f'{evtnum}; {utcdate}')
    fig.tight_layout()

    if save:
        plt.savefig(save, dpi=300)

    return (fig, axes)