import penn

import matplotlib.pyplot as plt
import numpy as np


def plot_stft(axis : plt.Axes,
              audio,
              sr=penn.SAMPLE_RATE,
              window_length=2048*4,
              hop_length=penn.data.preprocess.GSET_HOPSIZE):
    """
    Add a plot of STFT to given audio.

    Parameters:
        axis - matplotlib pyplot figure axis to have the STFT plot
        audio - source data
        sr - sampling rate
        window_length - length of the moving STFT window
        hop_length - hop step of the moving window in samples
    """

    stft, freqs, times = penn.plot.raw_data.extract_spectrogram(audio,
                                             sr=sr,
                                             window_length=window_length,
                                             hop_length=hop_length)

    axis.pcolormesh(times, freqs, np.abs(stft), cmap='grey')
    axis.set_ylim([50, 300])
    axis.set_xlim([times[0], times[-1]])

    # take inspiration from this post: https://dsp.stackexchange.com/a/70136


def plot_pitch(axis : plt.Axes,
               pitch,
               times,
               set_pitch_lims=True,
               plot_red=False,
               linewidth=1):
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
    """
    max_pitch = []
    min_pitch = []

    pitch_masked = np.ma.MaskedArray(pitch, pitch==0)

    for no_slice, pitch_slice in enumerate(pitch_masked):
        y = pitch_slice.reshape(-1)
        x = times

        # axis.scatter(x, y, label=f"String {no_slice}")
        if plot_red:
            axis.plot(x, y, 'r-', linewidth=linewidth, label=f"String {no_slice}")
        else:
            axis.plot(x, y, '-', linewidth=linewidth, label=f"String {no_slice}")


        if pitch_slice.size > 0:
            max_pitch.append(pitch_slice.max())
            min_pitch.append(pitch_slice.min())

    ymax = max(max_pitch)
    ymin = min(min_pitch)

    offset = (ymax - ymin) * 0.1
    ymax += offset
    ymin -= offset

    if set_pitch_lims:
        axis.set_ylim([ymin, ymax])

    axis.set_ylabel('Frequency [Hz]')
    axis.set_xlabel('Time [s]')


def plot_periodicity(axis : plt.Axes, periodicity, threshold=None):
    """
    Plot the periodicity plot with or without threshold.
    """
    if threshold is None:
        pass


def plot_with_matplotlib(audio, sr=penn.SAMPLE_RATE, pred_pitch=None, pred_times=None, gt_pitch=None, gt_times=None, periodicity=None, threshold=None):
    """
    Plot stft to the given audio. Optionally put raw pitch data
    or even thresholded periodicity data on top of it.
    """

    # Create plot
    figure, axis = plt.subplots(figsize=(7, 3))

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plot_stft(axis, audio, sr)

    if pred_pitch is not None and pred_times is not None:
        plot_pitch(axis, pred_pitch, pred_times, linewidth=2)

    if gt_pitch is not None and gt_times is not None:
        plot_pitch(axis, gt_pitch, gt_times, plot_red=True)

    if periodicity is not None:
        plot_periodicity(axis, periodicity, threshold)

    # figure.show()
    plt.show()
