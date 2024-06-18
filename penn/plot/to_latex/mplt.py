import penn

import matplotlib.pyplot as plt
import numpy as np

from typing import List


def plot_logits(axes : plt.Axes,
                logits : np.ndarray, 
                hop_length_seconds=penn.HOPSIZE_SECONDS,
                time_offset=0):
    """
    Add a plot of logits

    Parameters:
        axis - matplotlib pyplot figure axis to have the STFT plot
        logits - a numpy array representing the model output layer
    """
    logits = logits.squeeze()

    # divide logits into strings
    logits_chunks = np.split(logits, logits.shape[1], axis=1)

    freqs = penn.convert.bins_to_frequency(np.arange(logits.shape[-1]))
    times = np.arange(logits.shape[0]) * hop_length_seconds + time_offset

    for axis, logits_chunk in zip(axes, logits_chunks):
        logits_chunk = logits_chunk.squeeze(axis=1).T
        axis.pcolormesh(times, freqs, logits_chunk)


def plot_stft(axes : plt.Axes,
              audio,
              sr=penn.SAMPLE_RATE,
              window_length=2048*4,
              hop_length=penn.data.preprocess.GSET_HOPSIZE,
              time_offset=0):
    """
    Add a plot of STFT to given audio.

    Parameters:
        axis - matplotlib pyplot figure axis to have the STFT plot
        audio - source data
        sr - sampling rate
        window_length - length of the moving STFT window
        hop_length - hop step of the moving window in samples
        time_offset - increase the values of the x axis by given time offset
    """

    stft, freqs, times = penn.plot.raw_data.extract_spectrogram(audio,
                                             sr=sr,
                                             window_length=window_length,
                                             hop_length=hop_length)
    times += time_offset

    for axis in axes:
        axis.pcolormesh(times, freqs, np.abs(stft), cmap='grey')
        axis.set_ylim([50, 300])
        axis.set_xlim([times[0], times[-1]])

    # take inspiration from this post: https://dsp.stackexchange.com/a/70136


def plot_pitch(axis : plt.Axes,
               pitch,
               times,
               set_pitch_lims=True,
               plot_red=False,
               linewidth=1,
               periodicity=None,
               threshold=0.5,
               ylim=[0, 500]):
    """
    Add a plot of pitch. Optionally, set the frequency limits based
    on the max i min values of the provided pitch.

    Parameters:
        axis - matplotlib pyplot figure axis to have the STFT plot
        pitch - pitch array
        times - times array
        set_pitch_lims - flag indicating if freqnecy limits are to be adjusted
        plot_red - set true to plot all lines in a red color
        linewidth - set the matplotlib plot line width
        periodicity - 
        threshold - threshold for the periodicity plot
    """
    max_pitch = []
    min_pitch = []

    mask_for_pitch = pitch!=0

    if periodicity is not None: 
        periodicity_for_mask = periodicity.squeeze()
        periodicity_mask = periodicity_for_mask >= threshold
        mask_for_pitch = np.logical_and(mask_for_pitch, periodicity_mask)

    # pitch_masked = np.ma.MaskedArray(pitch, mask_for_pitch)
    pitch_split = np.split(pitch, pitch.shape[0])
    mask_split = np.split(mask_for_pitch, mask_for_pitch.shape[0])

    for no_slice, (pitch_slice, mask_slice) in enumerate(zip(pitch_split, mask_split)):
        pitch_slice = pitch_slice.reshape(-1)
        y = np.ma.MaskedArray(pitch_slice, np.logical_not(mask_slice))
        x = times

        # axis.scatter(x, y, label=f"String {no_slice}")
        if plot_red:
            axis.plot(x, y, 'r-', linewidth=linewidth, label=f"String {no_slice}")
        else:
            axis.scatter(x, y, linewidth=linewidth, label=f"String {no_slice}")


        if pitch_slice.size > 0:
            max_pitch.append(pitch_slice.max())
            min_pitch.append(pitch_slice.min())

    if ylim is None:

        ymax = max(max_pitch)
        ymin = min(min_pitch)

        offset = (ymax - ymin) * 0.1
        ymax += offset
        ymin -= offset

        ylim = [ymin, ymax]

    if set_pitch_lims:
        axis.set_ylim(ylim)

    axis.set_ylabel('Frequency [Hz]')
    axis.set_xlabel('Time [s]')


def plot_multipitch(axes : List[plt.Axes],
                    pitch : np.ndarray,
                    times : np.ndarray,
                    set_pitch_lims=True,
                    plot_red=False,
                    linewidth=1,
                    periodicity : np.ndarray=None,
                    threshold=0.5):

    pitch_split = np.split(pitch, pitch.shape[0])

    if periodicity is not None: 
        periodicity = periodicity.squeeze()
        periodicity = np.split(periodicity, periodicity.shape[0])
    else:
        periodicity = [None] * len(pitch_split)

    for axis, pitch_slice, periodicity_slice in zip(axes, pitch_split, periodicity):
        plot_pitch(axis=axis,
                   pitch=pitch_slice,
                   times=times,
                   set_pitch_lims=set_pitch_lims,
                   plot_red=plot_red,
                   linewidth=linewidth,
                   periodicity=periodicity_slice,
                   threshold=threshold)


def plot_periodicity(axis : plt.Axes,
                     periodicity : np.ndarray,
                     times : np.ndarray,
                     threshold : float=0.05):
    """
    Plot the periodicity plot with or without threshold.
    """

    periodicity_for_plot = periodicity.squeeze().T

    # offset = np.arange(0, penn.PITCH_CATS) * int(not penn.LOSS_MULTI_HOT)
    # periodicity_for_plot += offset

    twin = axis.twinx()
    twin.set_ylim(ymin=0, ymax=1)
    twin.margins(y=0)

    # this https://stackoverflow.com/a/27198519/11287083
    # should help with removing the whitespace at the bottom of the plot

    twin.plot(times, periodicity_for_plot, 'g:', linewidth=2)

    if threshold is not None:
        periodicity_mask = periodicity_for_plot >= threshold
        # mask periodicity under the threshold 
        periodicity_masked = np.ma.MaskedArray(periodicity_for_plot, np.logical_not(periodicity_mask))

        twin.plot(times, periodicity_masked, 'm:', linewidth=2)


def plot_multiperiodicity(axes : List[plt.Axes],
                          periodicity : np.ndarray,
                          times : np.ndarray,
                          threshold : float=0.05):
    periodicity = periodicity.squeeze()
    periodicity_list = np.split(periodicity, periodicity.shape[0])

    for axis, periodicity_slice in zip(axes, periodicity_list):
        plot_periodicity(axis, periodicity_slice, times, threshold)


def plot_with_matplotlib(
        audio,
        sr=penn.SAMPLE_RATE,
        pred_pitch=None,
        pred_times=None,
        gt_pitch=None,
        gt_times=None,
        periodicity=None,
        threshold=0.05,
        time_offset=0,
        mutlipitch=False,
        logits=None):
    """
    Plot stft to the given audio. Optionally put raw pitch data
    or even thresholded periodicity data on top of it.
    """

    if mutlipitch:
        # Create plot
        figure, axes = plt.subplots(nrows=penn.PITCH_CATS,
                                    ncols=1)
    else:
        # Create plot
        figure, axis = plt.subplots(figsize=(7, 3))
        axes = [axis]

    # Make pretty
    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)

    if logits is not None:
        plot_logits(axes, logits, time_offset=time_offset)
    else:
        plot_stft(axes, audio, sr, time_offset=time_offset)

    # if pred_pitch is not None and pred_times is not None:
    #     if mutlipitch:
    #         plot_multipitch(
    #                 axes, pred_pitch, pred_times,
    #                 linewidth=1.5,
    #                 periodicity=periodicity,
    #                 threshold=threshold)
    #     else:
    #         plot_pitch(
    #                 axes[0], pred_pitch, pred_times,
    #                 linewidth=1.5,
    #                 periodicity=periodicity,
    #                 threshold=threshold)

    if gt_pitch is not None and gt_times is not None:
        if mutlipitch:
            plot_multipitch(
                    axes, gt_pitch, gt_times,
                    plot_red=True)
        else:
            plot_pitch(
                    axes[0], gt_pitch, gt_times,
                    plot_red=True)
    #
    # if periodicity is not None:
    #     if mutlipitch:
    #         plot_multiperiodicity(axes, periodicity, pred_times, threshold)
    #     else:
    #         plot_periodicity(axes[0], periodicity, pred_times, threshold)

    figure.suptitle(f"Pitch thresholded with periodicity above {threshold}")

    # figure.show()
    plt.show()
