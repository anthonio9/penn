import torch
import numpy as np

import penn


###############################################################################
# Pitch conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    return penn.CENTS_PER_BIN * bins


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = quantize_fn(cents / penn.CENTS_PER_BIN)

    if type(bins) is np.ndarray:
        bins = bins.astype(np.int64)
    else:
        bins = bins.long()

    bins[bins < 0] = 0
    bins[bins >= penn.PITCH_BINS] = penn.PITCH_BINS - 1
    return bins


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return penn.FMIN * 2 ** (cents / penn.OCTAVE)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    if type(frequency) is np.ndarray:
        return penn.OCTAVE * np.log2(frequency / penn.FMIN)
    else:
        return penn.OCTAVE * torch.log2(frequency / penn.FMIN)


def frequency_to_samples(frequency, sample_rate=penn.SAMPLE_RATE):
    """Convert frequency in Hz to number of samples per period"""
    return sample_rate / frequency


def midi_to_frequency(midi):
    """Convert midi notes to frequency in Hz"""
    return 440 * 2 ** ((midi - 69) / 12)


def midi_to_cents(midi):
    """Convert midi notes to cents"""
    return frequency_to_cents(midi_to_frequency(midi))


def midi_to_organ_key(midi):
    """Convert midi note number to an organ key number. 
    The difference between midi note numbers and organs is that
    organs only have 61 keys. Midi note number 36 translates to
    an organ key number 0"""
    return midi - 36


###############################################################################
# Time conversions
###############################################################################


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * penn.HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * penn.HOPSIZE_SECONDS


def seconds_to_frames(seconds):
    """Convert seconds to number of frames"""
    return samples_to_frames(seconds_to_samples(seconds))


def seconds_to_samples(seconds, sample_rate=penn.SAMPLE_RATE):
    """Convert seconds to number of samples"""
    return seconds * sample_rate


def samples_to_frames(samples):
    """Convert samples to number of frames"""
    return samples // penn.HOPSIZE


def samples_to_seconds(samples, sample_rate=penn.SAMPLE_RATE):
    """Convert number of samples to seconds"""
    return samples / sample_rate


###############################################################################
# Larger conversions
###############################################################################

def merge_multi_string(logits, bins):
    pass
