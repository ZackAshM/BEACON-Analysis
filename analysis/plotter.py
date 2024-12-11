# Functions for plotting

from cfg import config

import numpy as np
from matplotlib import pyplot as plt

def plotEvent(reader, entry, ylims=[None, None, None], monutau=True, CWSubtract=None, save=''):
    '''
    Plot all waveforms from a given BEACON event.

    Parameters
    ----------
    reader : analysis.reader.Reader
        BEACON data Reader object.
    entry : int
    ylims : array((min,max), 'common', 'full', None)
        The plot y-limits in the order [scint, hpol RF, vpol RF]. In each slot, the following can be set: If (min, max), then
        each channel will be set to this y-limit. If 'common', the ylim for each channel is set to the min,max across all channels 
        (i.e. scint='max' will use the ymin and ymax across all scint channels). If 'full', the waveform min,max is assumed
        to be -127,127 for the scints and -64,64 for RF. If None (default), each channel will be set by its default ylim assumed
        by matplotlib.pyplot.
    monutau : bool, default True
        If True, plot in same arrangement as monutau, otherwise organize by scint then antenna number
    CWSubtract : analysis.sinesubtraction.SineSubtract, optional
        If given, sine subtract RF waveforms
    save : str
        If given, save the plot with this name.
    '''
    
    # scintillators 1-4 are channels 0-3
    # hpols in order are channels 12,13,14,15,8,9
    # vpols in order are channels 4,5,6,7,10,11

    t = reader.t()
    run = reader.run
    
    scintCh = [(ax_ind, label) for ax_ind, label in ] # index, label
    hpolCh = [(4,0,12),(6,1,13),(8,2,14),(10,3,15),(12,4,8),(14,5,9)] # ax index, label, channel index
    vpolCh = [(5,0,4),(7,1,5),(9,2,6),(11,3,7),(13,4,10),(15,5,11)]

    fig, ax = plt.subplots(nrows=4,ncols=4,figsize=[20, 15])
    axflat = ax.flatten()

    sVmax = [0,0]
    for ch, label in scintCh:
        wfm = reader.wf(ch)
        sVmax[0] = min((min(wfm), sVmax[0])) 
        sVmax[1] = max((max(wfm), sVmax[1]))

        axflat[ch].plot(t, wfm, c='red')
        axflat[ch].set(title=f'Scint {label}, Ch {ch}, run {run}, event {event+1}')

        if (isinstance(sy, Iterable)) and (not isinstance(sy, str)):
            axflat[ch].set(ylim=[sy[0],sy[1]])
    if sy == 'max':
        for ch, label in scintCh:
            axflat[ch].set(ylim=(sVmax[0]-2, sVmax[1]+2))
    
    hVmax = [0,0]
    for axs, label, ch in hpolCh:
        if monutau == True:
            axs = ch
        if CWSubtract is None:
            wfm = reader.wf(ch)
        else:
            __, wfm = CWSubtract.CWFilter(t, reader.wf(ch))
        hVmax[0] = min((min(wfm), hVmax[0]))
        hVmax[1] = max((max(wfm), hVmax[1]))

        axflat[axs].plot(t, wfm, c='darkgreen')
        axflat[axs].set(title=f'Hpol {label+1}, Ch {ch}, run {run}, event {event+1}')
        
        if (isinstance(hy, Iterable)) and (not isinstance(hy, str)):
            axflat[axs].set(ylim=[hy[0],hy[1]])
    if hy == 'max':
        for axs, label, ch in hpolCh:
            if monutau == True:
                axs = ch
            axflat[axs].set(ylim=(hVmax[0]-5, hVmax[1]+5))

    vVmax = [0,0]
    for axs, label, ch in vpolCh:
        if monutau == True:
            axs = ch
        if CWSubtract is None:
            wfm = reader.wf(ch)
        else:
            __, wfm = CWSubtract.CWFilter(t, reader.wf(ch))
        vVmax[0] = min((min(wfm), vVmax[0]))
        vVmax[1] = max((max(wfm), vVmax[1]))

        axflat[axs].plot(t, wfm, c='blue')
        axflat[axs].set(title=f'Vpol {label+1}, Ch {ch}, run {run}, event {event+1}')

        if (isinstance(vy, Iterable)) and (not isinstance(vy, str)):
            axflat[axs].set(ylim=[vy[0],vy[1]])
    if vy == 'max':
        for axs, label, ch in vpolCh:
            if monutau == True:
                axs = ch
            axflat[axs].set(ylim=(vVmax[0]-5, vVmax[1]+5))

    if rfy==True:
        ymin = min((hVmax[0], vVmax[0]))
        ymax = max((hVmax[1], vVmax[1]))
        for axs, label, ch in hpolCh:
            if monutau == True:
                axs = ch
            axflat[axs].set(ylim=(ymin-5, ymax+5))
        for axs, label, ch in vpolCh:
            if monutau == True:
                axs = ch
            axflat[axs].set(ylim=(ymin-5, ymax+5))

    # all axes options
    for axs in axflat:
        axs.set(xlabel='time [ns]', ylabel='voltage [adu]')
        axs.grid()

    hdr = reader.header(force_reload=True)
    evtnum = hdr.event_number
    utcdate = datetime.datetime.fromtimestamp(hdr.readout_time).strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f'{evtnum}; {utcdate}')
    
    fig.set_layout_engine(layout='tight')

    if save:
        plt.savefig(save, dpi=100)