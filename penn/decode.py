import numpy as np
import torch

import penn


###############################################################################
# Decode pitch contour from logits of pitch posteriorgram
###############################################################################


def argmax(logits):
    """Decode pitch using argmax"""
    # Get pitch bins
    if penn.FCN:
        bins = logits.argmax(dim=-1)
    else:
        bins = logits.argmax(dim=-2)

    # Convert to hz
    pitch = penn.convert.bins_to_frequency(bins)

    return bins, pitch


def viterbi(logits):
    """Decode pitch using viterbi decoding (from librosa)"""
    import librosa

    # Normalize and convert to numpy
    if penn.METHOD == 'pyin':
        periodicity = penn.periodicity.sum(logits).T
        unvoiced = (
            (1 - periodicity) / penn.PITCH_BINS).repeat(penn.PITCH_BINS, 1)
        distributions = torch.cat(
            (torch.exp(logits.permute(2, 1, 0)), unvoiced[None]),
            dim=1).numpy()
    else:

        # Viterbi REQUIRES a categorical distribution, even if the loss was BCE
        distributions = torch.nn.functional.softmax(logits, dim=1)
        distributions = distributions.permute(2, 1, 0)
        distributions = distributions.to(
            device=torch.device('cpu'),
            dtype=torch.float32
        ).numpy()

    # Cache viterbi probabilities
    if not hasattr(viterbi, 'transition'):
        # Get number of bins per frame
        bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
        max_octaves_per_frame = \
            penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
        max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1

        # Construct the within voicing transition probabilities
        viterbi.transition = librosa.sequence.transition_local(
            penn.PITCH_BINS,
            max_bins_per_frame,
            window='triangle',
            wrap=False)

        if penn.METHOD == 'pyin':

            # Add unvoiced probabilities
            viterbi.transition = np.kron(
                librosa.sequence.transition_loop(2, .99),
                viterbi.transition)

            # Uniform initial probabilities
            viterbi.initial = np.zeros(2 * penn.PITCH_BINS)
            viterbi.initial[penn.PITCH_BINS:] = 1 / penn.PITCH_BINS

        else:

            # Uniform initial probabilities
            viterbi.initial = np.full(penn.PITCH_BINS, 1 / penn.PITCH_BINS)

    # Viterbi decoding
    bins = librosa.sequence.viterbi(
        distributions,
        viterbi.transition,
        p_init=viterbi.initial)
    bins = torch.from_numpy(bins.astype(np.int32))

    # Convert to frequency in Hz
    if penn.DECODER.endswith('normal'):

        # Decode using an assumption of normality around to the viterbi path
        pitch = local_expected_value_from_bins(
            bins.T.to(logits.device),
            logits).T

    else:

        # Argmax decoding
        pitch = penn.convert.bins_to_frequency(bins)

    if penn.METHOD == 'pyin':

        # Linearly interpolate unvoiced regions
        pitch[bins >= penn.PITCH_BINS] = 0
        pitch = penn.data.preprocess.interpolate_unvoiced(pitch.numpy())[0]
        pitch = torch.from_numpy(pitch)

    return bins.T, pitch.T


def local_expected_value(logits, window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using a normal assumption around the argmax"""
    # Get center bins
    # dim -2 => [128, 1440, 1] (1440), [128, 6, 1440, 1] (1440)
    bins = logits.argmax(dim=-2)

    return bins, local_expected_value_from_bins(bins, logits, window)


###############################################################################
# Utilities
###############################################################################


def expected_value(logits, cents):
    """Expected value computation from logits"""
    # Get local distributions
    if penn.LOSS == 'categorical_cross_entropy':
        distributions = torch.nn.functional.softmax(logits, dim=-1)
    elif penn.LOSS == 'binary_cross_entropy':
        distributions = torch.sigmoid(logits)
    else:
        raise ValueError(f'Loss {penn.LOSS} is not defined')

    # Pitch is expected value in cents
    pitch = (distributions * cents).sum(dim=-1, keepdims=True)

    # BCE requires normalization
    if penn.LOSS == 'binary_cross_entropy':
        pitch = pitch / distributions.sum(dim=-1)

    # Convert to hz
    return penn.convert.cents_to_frequency(pitch)


def local_expected_value_from_bins(
    bins,
    logits,
    window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using normal assumption around argmax from bin indices

    Returns
        freq - pitch vector in Hz
    """
    # Pad
    padded = torch.nn.functional.pad(
        logits.squeeze(-1),
        (window // 2, window // 2),
        value=-float('inf'))

    # regular case of logits.shape == [128, 1440, 1] and bins [128, 1]
    repeat_window = [1, window]

    # make the algorithm work with logits.shape == [128, 6, 1440, 1] => the poly pitch net
    if len(bins.shape) == 3:
        repeat_window = [1, 1, window]
        window_indxs = torch.arange(window, device=bins.device)[None][None]

    # Get indices
    breakpoint()
    indices = \
        bins.repeat(repeat_window) + torch.arange(window, device=bins.device)[None]

    # Get values in cents
    cents = penn.convert.bins_to_cents(torch.clip(indices - window // 2, 0))

    # Decode using local expected value
    return expected_value(torch.gather(padded, -1, indices), cents)
