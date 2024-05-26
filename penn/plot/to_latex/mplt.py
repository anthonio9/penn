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

    axis.pcolormesh(times, freqs, np.abs(stft), )
    axis.set_ylim([50, 300])
    axis.set_xlim([times[0], times[-1]])

    # take inspiration from this post: https://dsp.stackexchange.com/a/70136


def plot_with_matplotlib(audio, sr=penn.SAMPLE_RATE, pitch_pred=None, pred_times=None, ground_truth=None, periodicity=None, threshold=None):
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

    # figure.show()
    plt.show()
