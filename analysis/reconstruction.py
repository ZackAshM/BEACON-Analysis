'''functions for beacon event reconstruction, split up by RF and Scint'''
from analysis.reader import Reader
from analysis.sinesubtraction import SineSubtract
from analysis import utils
from cfg import config

import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import itertools


def mapRF(reader, entry, pol, CWsub=None, mask=None, source_distance=1e6, resolution=1, 
          upsample_factor=100, plot=True, verbose=1):
    '''
    Correlate RF waveforms to reconstruct a source direction map, and return the reconstruction parameters.

    Parameters
    ----------
    reader : analysis.reader.Reader
    entry : int
    pol : ['h', 'v']
    CWsub : analysis.sinesubtraction.SineSubtract, optional
        Default is None.
    source_distance : integer or real, optional
        The distance in meters calculated for plane wave approximation.
    resolution : integer or real, optional
        Angular resolution, or angular step size in degrees for both zenith and azimuth.
    mask : number, optional
        If given, exclude waveforms from the correlation whose snr is below this value. 
        If more than a certain number of waveforms are excluded, then the return array elements are all -1.
        See cfg.config for setting threshold parameters.
    upsample_factor : int or None, optional
        Upsample by this factor via scipy.resample_poly(). Default is 100.
    plot : bool, optional
        If True, the map and coherent waveforms are plotted. Default is True.
    verbose : [0,1,2], optional
        Level of verbosity. If 0, no statements are printed. If 1, info about the reconstruction are printed.
        If 2, warnings and errors are also printed. Default is 1.

    Returns
    -------
    Correlation Map : ndarray
        The correlation map of the reconstructed direction.
    Best Zenith Angle : real number
        Zenith angle in degrees corresponding to the best reconstructed direction.
    Best Azimuth Angle : real number
        Azimuth angle in degrees corresponding to the best reconstructed direction.
    Best Time Delays : list of real numbers
        Time delays in ns relative to the first non-masked channel corresponding to the best reconstructed direction.
    Coherent Summed Time and Waveform : ndarray
        The coherently summed waveform time and ADU arrays using the best reconstructed direction time delays.
    (optional) Mask Indices : list of integers
        If mask is given, this is the list of indices to apply for masking.
    '''

    # handle args
    if verbose > 1:
        if not isinstance(reader, Reader):
            print('Warning: "reader" is not recognized as analysis.reader.Reader object')
        if not isinstance(entry, int):
            print('Warning: "entry" is not recognized as int')
        if not isinstance(CWsub, SineSubtract):
            print('Warning: "CWsub" is not recognized as analysis.sinesubtraction.SineSubtract object')
    if pol.lower() not in ['h','v', 'hpol','vpol']:
        raise ValueError('"pol" must be "h", "v", "hpol", or "vpol"')
    if upsample_factor is None: upsample_factor = 1
    
    reader.setEntry(entry)
    t = reader.t() # time in ns
    dt = t[1] - t[0]
    
    # get channel indices and cable delays
    allChannels = np.fromiter(config.HPOL_MAP.values(), dtype=int) if ('h' in pol.lower()) else np.fromiter(config.VPOL_MAP.values(), dtype=int)
    allAntsPos = config.RF_HPOL_POS if ('h' in pol.lower()) else config.RF_VPOL_POS
    allCableDelays = config.RF_HPOL_DELAYS if ('h' in pol.lower()) else config.RF_VPOL_DELAYS
    
    # get waveforms
    raw_wfms = []
    clean_wfms = []
    z_wfms = []
    for ch in allChannels:
        wfm = reader.wf(ch)
        raw_wfms.append(wfm)
        filt_wfm = wfm if (CWsub is None) else CWsub.CWFilter(t, wfm)[1] # sine subtraction
        upsampled_wfm = filt_wfm if (upsample_factor == 1) else signal.resample_poly(filt_wfm, upsample_factor, 1) # upsampling; don't waste time if not upsampling
        clean_wfms.append(upsampled_wfm)
        z_wfm = (upsampled_wfm - np.mean(upsampled_wfm))/np.std(upsampled_wfm) # Z-normalization
        z_wfms.append(z_wfm)
    z_wfms = np.array(z_wfms)
    clean_wfms = np.array(clean_wfms)
    raw_wfms = np.array(raw_wfms)
    upsampled_t = np.arange(len(z_wfms[0]))*dt/upsample_factor
    
    # grid definition
    phi = np.arange(-180, 180+resolution/2, resolution)
    theta = np.arange(0, 180+resolution/2, resolution)
    azimuth, zenith = np.meshgrid(np.deg2rad(phi), np.deg2rad(theta))

    # exclude low snr waveforms from correlation
    maskArray = np.array([0 if utils.rfsnr(z_wfms[i]) < mask else 1 for i in range(len(allChannels))]) if mask is not None else np.ones(len(allChannels))
    if np.sum(maskArray) < config.RF_RECON_REQUISITE:
        if verbose > 0:
            print('This low SNR event is ineligible for reconstruction with masking. Accept ' +
            'this or use "mask=False" argument')
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    maskIndices = np.where(maskArray == 1)[0]

    if verbose > 0:
        print('Running reconstruction...')
        
    channels = allChannels[maskIndices]
    ants = allAntsPos[maskIndices]
    cable_delays = allCableDelays[maskIndices]
    wfms = z_wfms[maskIndices]
    cleanWfms = clean_wfms[maskIndices]

    # calculate source position and expected plane wave travel times to each antenna
    source_position = source_distance * np.stack([np.cos(azimuth) * np.sin(zenith),
                                                  np.sin(azimuth) * np.sin(zenith),
                                                  np.cos(zenith)], axis=-1)
    travel_times = np.array([np.linalg.norm(source_position - ants[i], axis=2) / c * 1e9 + cable_delays[i] for i in range(len(channels))])

    # calculate pairwise parameters: expected time differences, waveform correlations
    upperTrianglePairIndices = np.triu_indices(len(channels), k=1) # upper triangle indices
    expected_time_diffs = np.array([travel_times[j] - travel_times[i] for i,j in zip(*upperTrianglePairIndices)]) # pairwise differences ie 1-0, 2-0, ..., 2-1,3-1,... etc. 
    normalizedCorrelation = lambda x, y: signal.correlate(x, y) / np.sqrt(np.sum(x**2)*np.sum(y**2))
    wfm_corr = np.array([normalizedCorrelation(wfms[i], wfms[j]) for i,j in zip(*upperTrianglePairIndices)])

    # extract correlation values at expected time differences for each grid bin
    center = wfms.shape[1]
    corr_vals = np.array([wfm_corr[i][np.rint(-upsample_factor*expected_time_diffs[i]/dt+(center-1)).astype(int)] for i in range(len(wfm_corr))])

    # weight and normalize correlation map
    baselines = np.linalg.norm(np.array([ants[j]-ants[i] for i,j in zip(*upperTrianglePairIndices)]), axis=1)
    corr_map = np.average(corr_vals, axis=0, weights=baselines)

    # find point on the sky where the correlation is greatest = best reconstruction
    best_theta_ind, best_phi_ind = np.unravel_index(corr_map.argmax(), corr_map.shape)
    theta_best, phi_best = theta[best_theta_ind], phi[best_phi_ind]

    # coherently sum
    best_t_diffs = np.array([0]+[expected_time_diffs[i][best_theta_ind, best_phi_ind] for i in range(len(channels)-1)])
    tmin = np.max([np.min(upsampled_t - best_t_diffs[i]) for i in range(len(channels))])
    tmax = np.min([np.max(upsampled_t - best_t_diffs[i]) for i in range(len(channels))])
    teval = np.arange(tmin+dt/upsample_factor, tmax, dt/upsample_factor)
    coherent_wfm = np.mean([interp1d(upsampled_t - best_t_diffs[i], cleanWfms[i], fill_value='extrapolate')(teval) for i in range(len(channels))], axis=0)
    wfmsnr = utils.rfsnr(coherent_wfm)

    if plot:
        fig0, ax0 = plt.subplots(ncols=len(allChannels), figsize=(30,3))
        for i in range(len(allChannels)):
            alpha = 0.3 if maskArray[i] == 0 else 1
            ax0[i].plot(t, raw_wfms[i], alpha=alpha)
            ax0[i].plot(upsampled_t, clean_wfms[i], alpha=alpha)
            ax0[i].add_artist(AnchoredText(f"SNR: {utils.rfsnr(clean_wfms[i]):0.2f}", loc="upper left", frameon=True))
            
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.pcolormesh(phi, theta, corr_map, shading='nearest',cmap=plt.cm.coolwarm)
        ax.hlines(theta_best, -180, 180, ls="--", colors="fuchsia")
        ax.vlines(phi_best, 0, 180, ls="--", colors="fuchsia")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Correlation Value", rotation=270, labelpad=15)
        ax.set_xlabel('Azimuth Angle (Degrees)')
        ax.set_ylabel('Zenith Angle (Degrees)')
        ax.get_yaxis().set_inverted(True)
        ax.add_artist(AnchoredText(f"Reconstruction (zen, az): {theta_best:0.2f}$\degree$, {phi_best:0.2f}$\degree$", 
                                loc="lower right", frameon=True))

        fig2, ax2 = plt.subplots(figsize=(10,3))
        getKey = lambda d, val: list(d.keys())[list(d.values()).index(val)]
        for i, ch in enumerate(channels):
            antNum = getKey(config.HPOL_MAP, ch) if ('h' in pol.lower()) else getKey(config.VPOL_MAP, ch)
            ax2.plot(upsampled_t - best_t_diffs[i], cleanWfms[i], label=f"Antenna {antNum}")
        ax2.plot(teval, coherent_wfm, ls='--', c='black', label=f"Coherently Summed, snr={wfmsnr:0.1f}")
        ax2.set(xlabel="Time [ns]", ylabel="Voltage [ADU]", title="Aligned Pulses from Best Correlator Bin")
        ax2.legend(loc='upper right', ncols=2, fontsize=8)
        ax2.grid()

    # summary statements
    if verbose > 0:
        print("From the correlation plot:")
        print("Best zenith angle:",theta_best)
        print("Best azimuth angle:",phi_best)
        for i,j in zip(upperTrianglePairIndices[0][:len(channels)-1], upperTrianglePairIndices[1][:len(channels)-1]):
            exp_time_diff = -(np.argmax(wfm_corr[j-1])-(center-1))*dt/upsample_factor
            print(f'Predicted (Expected) time delays between A{i+1} and A{j+1}: {best_t_diffs[j]:0.2f} ({exp_time_diff:0.2f}) ns')
        print(f"Coherent Waveform SNR = {wfmsnr}")

    if mask is None:
        return (corr_map, theta_best, phi_best, best_t_diffs, teval, coherent_wfm)
    else:
        return (corr_map, theta_best, phi_best, best_t_diffs, teval, coherent_wfm, maskIndices)

def mapScint(reader, entry, source_distance=1e6, resolution=1, norm=True, plot=True, plot_contours=True, verbose=1):
    '''
    Fit scintillator waveforms to a plane wave to reconstruct a source direction map, and return the reconstruction parameters.
    The scintillator arrival times are determined by a 50% Constant Fraction Discriminator.

    When more than a certain number of waveforms are low, or the waveform(s) is abnormal, then returned are arrays and numbers
    with elements filled with -1's. See cfg.config for setting threshold parameters and other related settings.

    Parameters
    ----------
    reader : analysis.reader.Reader
    entry : int
    source_distance : integer or real, optional
        The distance in meters calculated for plane wave approximation.
    resolution : integer or real, optional
        Angular resolution, or angular step size in degrees for both zenith and azimuth.
    norm : bool, optional
        If True, the correlation map is normalized to sum to 1.
    plot : bool, optional
        If True, the sky map is plotted. Default is True.
    plot_contours : bool, [bool, bool, bool], optional
        Plot the [68%, 95%, 99.7%] probability contours.
    verbose : [0,1,2], optional
        Level of verbosity. If 0, no statements are printed. If 1, info about the reconstruction are printed.
        If 2, warnings and errors are also printed. Default is 1.

    Returns
    -------
    Correlation Map : ndarray
        The correlation map of the reconstructed direction.
    Best Zenith Angle : real number
        Zenith angle in degrees corresponding to the best reconstructed direction.
    Best Azimuth Angle : real number
        Azimuth angle in degrees corresponding to the best reconstructed direction.
    Best Arrival Times : list of real numbers
        The arrival times of each non-masked channels' pulse that correspond to the best reconstructed direction.
    '''

    # handle args
    if verbose > 1:
        if not isinstance(reader, Reader):
            print('Warning: "reader" is not recognized as analysis.reader.Reader object')
        if not isinstance(entry, int):
            print('Warning: "entry" is not recognized as int')

    reader.setEntry(entry)
    t = reader.t() # time in ns
    dt = t[1] - t[0]
    
    # get channel indices and cable delays
    allChannels = np.fromiter(config.SCINT_MAP.values(), dtype=int)
    allCableDelays = config.SCINT_DELAYS

    # get waveforms
    all_wfms = np.array([reader.wf(ch) for ch in allChannels])

    # the resolution
    phi = np.arange(-180, 180+resolution/2, resolution)
    theta = np.arange(0, 180+resolution/2, resolution)
    azimuth, zenith = np.meshgrid(np.deg2rad(phi), np.deg2rad(theta))

    # return this when something goes wrong or is not an eligible event for reconstruction
    fail = (np.nan, np.nan, np.nan, np.nan)
    
    # exclude low scint and empty risetimes from reconstruction
    maskCondition1 = lambda wfm: (np.max(wfm) < config.SCINT_EXCLUSION_THRESHOLD)
    def maskCondition2(wfm):
        tempRiseTimes = np.array(utils.getRisetimes(t, wfm)[0])
        return (len(tempRiseTimes[~np.isnan(tempRiseTimes)]) == 0)
    maskArray = np.array([0 if np.logical_or(maskCondition1(all_wfms[ch]), maskCondition2(all_wfms[ch])) else 1 for ch in allChannels])
    if np.sum(maskArray) < config.SCINT_RECON_REQUISITE: 
        if verbose > 0: print('Too many masked scint channels. See "SCINT_RECON_REQUISITE" and "SCINT_EXCLUSION_THRESHOLD" in cfg.config.')
        return fail
    maskIndices = np.where(maskArray == 1)[0]
        
    channels = allChannels[maskIndices]
    ants = config.SCINT_POS[maskIndices]
    cable_delays = allCableDelays[maskIndices]
    wfms = all_wfms[maskIndices]

    if verbose > 0:
        print('Running reconstruction...')
    
    # get risetimes --> measured arrival times
    allArrivalTimes = []
    allUncertainties = []
    for i in range(len(wfms)):
        risetimes, d_risetimes = utils.getRisetimes(t, wfms[i])
        risetimes = risetimes[~np.isnan(risetimes)]
        d_risetimes = d_risetimes[~np.isnan(d_risetimes)]
        allArrivalTimes.append(risetimes - cable_delays[i]) # Correct for cable delays in either measured or expected, I chose measured
        allUncertainties.append(d_risetimes)
    
    # for situations with multiple pulses, get the group of pulses most likely to correspond to the same source via minimizing time differences
    arrivalTimeGroups = list(itertools.product(*allArrivalTimes))
    d_arrivalTimeGroups = list(itertools.product(*allUncertainties))
    d_group = [np.sqrt(np.sum([d_time**2 for d_time in dg])) for dg in d_arrivalTimeGroups]
    scorePulseGroups = lambda group: np.sum([(element-np.mean(group))**2/dg for element, dg in zip(group, d_group)])
    scores = [scorePulseGroups(g) for g in arrivalTimeGroups]
    bestGroupIndex = np.where(scores == np.min(scores))[0][0]
    
    arrivalTimes = arrivalTimeGroups[bestGroupIndex]
    d_arrivalTimes = d_arrivalTimeGroups[bestGroupIndex]

    # calculate source position and expected plane wave travel times to each antenna
    sourcePosition = source_distance * np.stack([np.cos(azimuth) * np.sin(zenith),
                                                 np.sin(azimuth) * np.sin(zenith),
                                                 np.cos(zenith)], axis=-1)  # ENU
    travelTimes = np.array([np.linalg.norm(sourcePosition - ants[i], axis=2) / c * 1e9 for i in range(len(channels))])

    # calculate pairwise time differences
    upperTrianglePairIndices = np.triu_indices(len(channels), k=1) # upper triangle indices
    expectedTimeDiffs = np.array([travelTimes[j] - travelTimes[i] for i,j in zip(*upperTrianglePairIndices)]) # pairwise differences ie 1-0, 2-0, ..., 2-1,3-1,... etc. 
    measuredTimeDiffs = np.array([arrivalTimes[j] - arrivalTimes[i] for i,j in zip(*upperTrianglePairIndices)]) 
    d_measuredTimeDiffs = np.array([np.sqrt(d_arrivalTimes[j]**2 + d_arrivalTimes[i]**2) for i,j in zip(*upperTrianglePairIndices)])

    # calc chi sqs
    chiSq = np.array([(t_meas - t_exp)**2 / d_t_meas**2 for t_exp, t_meas, d_t_meas in zip(expectedTimeDiffs, measuredTimeDiffs, d_measuredTimeDiffs)])

    # weigh by baselines
    baselines = np.array([ants[j] - ants[i] for i,j in zip(*upperTrianglePairIndices)])
    baselinesNorm = np.linalg.norm(baselines, axis=1)
    corr_map = np.average(chiSq, axis=0, weights=baselinesNorm)

    likelihood_map = np.exp(- corr_map / 2) # note: exponent already squared (it's chi squared)

    # correct for spherical area
    likelihood_map *= np.sin(zenith) 
    
    # correct for effective detector area
    scintFaceNormals = config.SCINT_FACE_NORMALS[maskIndices]
    scintDims = config.SCINT_DIMENSIONS[maskIndices]
    sourceNormalVector = sourcePosition / source_distance
    scintFaceWeights = [] # get weights for each normal vector dot product based on face areas
    for scintDim in scintDims:
        scintAreaNorm = np.sqrt((scintDim[0]*scintDim[1])**2 + (scintDim[0]*scintDim[2])**2 + (scintDim[1]*scintDim[2])**2) # max projected area
        scintFaceAreas = np.array([scintDim[0]*scintDim[1], scintDim[0]*scintDim[2], scintDim[1]*scintDim[2]]) / scintAreaNorm
        scintFaceWeights.append(scintFaceAreas)
    sourceDotFaces = np.tensordot(sourcePosition, scintFaceNormals, axes=([2], [2]))
    sourceDotFaces[sourceDotFaces < 0] = 0 # disallow signals from beneath / enforce very unlikely
    effectiveArea = np.sum(np.sum(np.array(scintFaceWeights)[None,None,:,:] * sourceDotFaces, axis=-1), axis=-1) # sum over faces and over scints
    effectiveArea = effectiveArea / effectiveArea.max() # normalize
    likelihood_map *= effectiveArea

    # convert to probability distribution
    if np.sum(likelihood_map) == 0: return fail # caught weirdly impossible event probably (eg. run250 evt127)
    likelihood_map = likelihood_map / np.sum(likelihood_map)

    # find point on the sky where the correlation is greatest = best reconstruction
    best_theta_ind, best_phi_ind = np.unravel_index(likelihood_map.argmax(), likelihood_map.shape)
    theta_best, phi_best = theta[best_theta_ind], phi[best_phi_ind]

    if plot:
        # plot waveforms
        fig, axes = plt.subplots(nrows=len(allChannels), figsize=[30,15])
        fig.suptitle(f'Run {reader.run} Event {reader.event_entry}', fontsize=20)
        fig.supxlabel('Time [ns]', fontsize=20)
        fig.supylabel('Voltage [ADU]', fontsize=20)
        for j in range(len(allChannels)):
            wfm = all_wfms[j]
            ax = axes[j]

            # plot raw waveforms, less alpha if not included in reconstruction
            alpha = 0.3 if maskArray[j] == 0 else 1
            ax.plot(t, wfm, alpha=alpha)

            # get peaks, risetimes, and integrations
            inds, amps = utils.getPeaks(wfm)
            rt, drt = utils.getRisetimes(t, wfm)
            pulseInts, pulseInds, totalInt, posInt = utils.integratePulses(t, wfm)

            # plot peaks, risetimes, and integrations
            for num in range(len(inds)):

                # peaks
                i = inds[num]
                ax.scatter(t[inds[num]], amps[num], alpha=alpha)

                # integrations
                ax.plot(t[pulseInds[num]], wfm[pulseInds[num]], c='red', alpha=alpha)
                ax.fill_between(t[pulseInds[num]], wfm[pulseInds[num]], color='red', alpha=0.1*alpha)
                ax.add_artist(AnchoredText(f'Positive Integration: {posInt:0.0f}', bbox_to_anchor=(0, 0.2), bbox_transform=ax.transAxes,
                                           loc="lower left", frameon=True, prop=dict(fontsize=20)))

                # risetimes
                col = 'blue' if rt[num]-allCableDelays[j] in arrivalTimeGroups[bestGroupIndex] else 'black'
                if ~np.isnan(rt[num]):
                    ax.errorbar(rt[num], amps[num]/2, xerr=drt[num], capsize=3, color=col)
                ymin, ymax = ax.get_ylim()
                yrange = ymax - ymin
                v = lambda y: (y-ymin)/(ymax-ymin)
                vmin = v(amps[num]/2-0.1*yrange)
                vmax = v(amps[num]/2+0.1*yrange)
                ax.axvline(rt[num], ymin=vmin, ymax=vmax, color=col, alpha=alpha, ls='--')
                ax.add_artist(AnchoredText(f'Risetimes: {", ".join(["{0:0.2f}+/-{1:0.2f}".format(r, dr) for r, dr in zip(rt,drt)])}', loc="lower left", frameon=True, prop=dict(fontsize=20)))
                
        fig.tight_layout(pad=2)

        # plot reconstruction map
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.pcolormesh(phi, theta, likelihood_map, shading='nearest',cmap=plt.cm.coolwarm)
        ax.hlines(theta_best, -180, 180, ls="--", colors="fuchsia")
        ax.vlines(phi_best, 0, 180, ls="--", colors="fuchsia")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Probability Density", rotation=270, labelpad=15)
        ax.set(xlabel='South <--- Azimuth Angle (Degrees From East) ---> North', ylabel='Zenith Angle (Degrees)',
              title=f'Run {reader.run} Event {reader.event_entry}')
        ax.get_yaxis().set_inverted(True)
        ax.add_artist(AnchoredText(f"Reconstruction (zen, az): {theta_best:0.2f}$\degree$, {phi_best:0.2f}$\degree$", 
                                loc="lower right", frameon=True))

        # plot probability contours
        if (len(np.shape(plot_contours)) == 1) and (len(plot_contours) != 3):
            if verbose > 1: print('Warning for "plot_contours": Invalid argument. Defaulting to True.')
            plot_contours = True
        plot_contours = 3*[plot_contours] if len(np.shape(plot_contours)) == 0 else plot_contours
            
        image = likelihood_map.copy()
        valOrder = np.sort(np.unique(image), axis=None, kind='stable')[::-1]
        contourColors = np.array(['black', 'gray', 'white'])[np.where(plot_contours)[0]]
        targetSums = np.array([0.997, 0.95, 0.68])[np.where(plot_contours)[0]]  # Target sum for the area
        levels = []
        # use variable stepsize algorithm to find first occurence when image sum exceeds targets
        for j, targetSum in enumerate(targetSums):
            i, imax, di, imSum = 0, np.inf, int(len(valOrder)/2), 0 # iteration variables, stepsize starts large then halve
            while abs(i - imax) > 2:
                imSelect = image >= valOrder[i]
                imSum = image[imSelect].sum() # should approximate cumulative probability of peaks
                i = i+di if imSum < targetSum else i-di # move forward or backwards depending on exceeding or lowballing target
                if imSum > targetSum: imax = i+di # if exceeded target, consider this closest attempt. If current index <2 away from this, we're at the ~closest possible
                di = 1 if int(di/2) == 0 else int(di/2) # change stepsize by half
            levels.append(valOrder[i])
        CS = ax.contour(phi, theta, image, levels=levels, linestyles='--', colors=contourColors, linewidths=1.5)
        ax.clabel(CS, CS.levels, inline=True, fmt={l:f'{ts*100:.1f}%' for l, ts in zip(CS.levels, targetSums)}, fontsize=10)

    if verbose > 0:
        print(f"Best Zenith, Azimuth: {theta_best:0.3f}, {phi_best:0.3f}")

        timeDiffErrors = expectedTimeDiffs[:,best_theta_ind,best_phi_ind] - measuredTimeDiffs
        avgTimeDiffError = np.mean(np.abs(timeDiffErrors/expectedTimeDiffs[:,best_theta_ind,best_phi_ind]))
        stdTimeDiffError = np.std(np.abs(timeDiffErrors/expectedTimeDiffs[:,best_theta_ind,best_phi_ind]))
        print(f'Expected Time Delays: \n{", ".join(["{0:0.2f}".format(exp_dt) for exp_dt in expectedTimeDiffs[:,best_theta_ind,best_phi_ind]])} ns')
        print(f'Measured Time Delays: \n{", ".join(["{0:0.2f}+/-{1:0.2f}".format(mdt, dmdt) for mdt, dmdt in zip(measuredTimeDiffs,d_measuredTimeDiffs)])} ns')
        print(f'Percent Differences: \n{", ".join(["{0:0.2f}".format(ddt) for ddt in timeDiffErrors])} ns')
        print(f'Average Difference = {avgTimeDiffError:0.03f} +/- {stdTimeDiffError:0.03f} ns')

    return (likelihood_map, theta_best, phi_best, arrivalTimes)