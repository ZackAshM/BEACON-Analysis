'''Functions for plotting'''

from cfg import config
from analysis.reader import Reader

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dask.dataframe as dd
import dask.array as da
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


def plotDistribution2D(df, column1, column2, bins1, bins2, hist=True, logHist=True, logc1=False, logc2=False, ax=None, markersize=4, **kwargs):
    """Generates a histogram or scatter plot based on cuts and configuration."""
    
    # Detect if df is a Dask or Pandas DataFrame
    is_dask = isinstance(df, dd.DataFrame)

    # Copy DataFrame
    cutData = df.copy()
    
    if len(cutData) == 0:
        raise ValueError("No events left.")

    # Compute min/max stats
    if is_dask:
        cutStats = pd.DataFrame({
            'min': cutData.min(skipna=True).compute(),
            'max': cutData.max(skipna=True).compute()
        })
    else:
        cutStats = pd.DataFrame({
            'min': cutData.min(skipna=True),
            'max': cutData.max(skipna=True)
        })

    cutData1 = cutData[column1]
    cutData2 = cutData[column2]

    # Set axis limits
    col1lim = (cutStats.loc[column1, 'min'], cutStats.loc[column1, 'max'])
    col2lim = (cutStats.loc[column2, 'min'], cutStats.loc[column2, 'max'])

    if logc1:
        col1lim = (np.log10(col1lim[0]), np.log10(col1lim[1]))
    if logc2:
        col2lim = (np.log10(col2lim[0]), np.log10(col2lim[1]))

    limits = (col1lim, col2lim)

    if hist:
        # Prepare data for histogram
        histData1 = np.log10(cutData1) if logc1 else cutData1
        histData2 = np.log10(cutData2) if logc2 else cutData2

        # Convert to NumPy/Dask arrays based on DataFrame type
        if is_dask:
            histData1 = histData1.to_dask_array()
            histData2 = histData2.to_dask_array()
            
            # Generate 2D histogram
            histDelayed, col1edges, col2edges = da.histogram2d(
            histData1, histData2,
            bins=(bins1, bins2), range=limits
            )
        
            # Compute histogram if using Dask
            histVals = np.log10(histDelayed.T.compute()) if logHist else histDelayed.T.compute()
            
        else:
            histData1 = histData1.values
            histData2 = histData2.values

            histDelayed, col1edges, col2edges = np.histogram2d(
            histData1, histData2,
            bins=(bins1, bins2), range=limits
            )

            histVals = np.log10(histDelayed.T) if logHist else histDelayed.T

        # Plot histogram
        pc = ax.pcolormesh(col1edges, col2edges, histVals)
        if 'zen' in column2:
            ax.get_yaxis().set_inverted(True)
        plt.colorbar(pc, ax=ax, label='Log(Counts)' if logHist else 'Counts')

        # Label axes
        ax.set(xlabel=column1, ylabel=column2)

        return (col1edges, col2edges, histVals)

    else:
        # Scatter plot
        if is_dask:
            sc = ax.scatter(cutData1.values.compute(), cutData2.values.compute(), s=markersize, **kwargs)
        else:
            sc = ax.scatter(cutData1.values, cutData2.values, s=markersize, **kwargs)

        # Label axes
        ax.set(xlabel=column1, ylabel=column2)

        return sc


# to plot overlays of specific events, particularly candidate CR events
_candidates1 = [(189, 26165), (254, 9824), (459, 13620), (500, 27996), (509, 9596), 
              (522, 22089), (538, 27140), (627, 17475), (717, 21811), (766, 6785)]
def plotOverlaidEvents(df, colx, coly=None, events=_candidates1, ax=None, legend=False, color=None, logx=False, logy=False, **kwargs):
    """
    Plots vlines onto 1D distribution plots (typically 1D histograms), and scatter points onto 2D distribution plots.
    Plotting onto 2D only available if coly is given.
    A color can be given for all overlaid data to follow, otherwise they will plot with different colors (pyplot Paired cmap).
    Kwargs are fed into axvline for 1D, or scatter for 2D.
    If ax not given, a (fig, ax) will be created and returned with a dataframe containing only the overlaid events. Otherwise,
    (fig, ax) = (None, None) when returned.

    Be aware that the events should be contained in df with columns 'run' and 'entry', as well as the desired colx and coly.
    """

    # get specific event data
    events_simpledf = pd.DataFrame(events, columns=['run', 'entry'])
    ### addColx = [] if colx in ['run', 'entry'] else [colx]
    ### addColy = [] if coly in ['run', 'entry', colx, None] else [coly]
    ### if data_df is None:
    ###     fullData = dd.read_parquet(DATAFILES, columns=['run', 'entry'] + addColx + addColy, 
    ###                                filters=cuts, assume_sorted=True)
    ### else:
    ###     fullData = data_df.copy()
    data_df = df.copy()
    overlaid_df = data_df.merge(events_simpledf, on=['run', 'entry'], how="inner")
    if isinstance(overlaid_df, dd.DataFrame):
        overlaid_df = overlaid_df.compute()

    if logx: overlaid_df[colx] = overlaid_df[colx].apply(np.log10)
    if logy: overlaid_df[coly] = overlaid_df[coly].apply(np.log10)

    fig = None
    if ax is None: 
        fig, ax = plt.subplots(figsize=[10,7])
        ax.set(xlabel=colx, ylabel=coly)

    if coly is None: # plot vlines
        plotData = overlaid_df[colx].values
        for line_i, line in enumerate(plotData):
            colour = [plt.cm.Paired(each) for each in np.linspace(0, 1, len(plotData))][line_i] if color is None else color
            lineRun, lineEntry = overlaid_df.loc[:, ['run', 'entry']].values[line_i]
            ax.axvline(line, ymin=0, ymax=1, label=f'{lineRun}, {lineEntry}', color=colour, **kwargs)
    else: # scatter
        plotData_colx = overlaid_df[colx].values
        plotData_coly = overlaid_df[coly].values
        for point_i, point in enumerate(zip(plotData_colx, plotData_coly)):
            colour = [plt.cm.Paired(each) for each in np.linspace(0, 1, len(plotData_colx))][point_i] if color is None else color
            pointRun, pointEntry = overlaid_df.loc[:, ['run', 'entry']].values[point_i]
            ax.scatter(point[0], point[1], label=f'{pointRun}, {pointEntry}', color=colour, **kwargs)

    if legend: ax.legend(loc='best')

    return (fig, ax, overlaid_df)