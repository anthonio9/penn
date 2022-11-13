import torch

import penne


###############################################################################
# Methods for extracting a periodicity estimate from pitch posteriorgram logits
###############################################################################


def average(logits):
    """Periodicity as the average of the logits over pitch bins"""
    averages = logits.mean(dim=1)
    if penne.LOSS == 'binary_cross_entropy':
        return torch.sigmoid(averages)
    elif penne.LOSS == 'categorical_cross_entropy':
        return torch.exp(averages) / torch.exp(logits).sum(dim=1)
    raise ValueError(f'Loss function {penne.LOSS} is not implemented')


def entropy(logits):
    """Entropy-based periodicity"""
    distribution = torch.nn.functional.softmax(logits, dim=1)
    return (
        1 - (1 / torch.log(penne.PITCH_BINS)) * \
        ((distribution * torch.log(distribution)).sum(dim=1)))


def max(logits):
    """Periodicity as the maximum confidence"""
    if penne.LOSS == 'binary_cross_entropy':
        return torch.sigmoid(logits).max(dim=1).values
    elif penne.LOSS == 'categorical_cross_entropy':
        return torch.nn.functional.softmax(
            logits, dim=1).max(dim=1).values
    raise ValueError(f'Loss function {penne.LOSS} is not implemented')


def sum(logits):
    """Periodicity as the sum of the distribution

    This is really just for PYIN, which performs a masking of the distribution
    probabilities so that it does not always add to one.
    """
    return torch.clip(torch.exp(logits).sum(dim=1), 0, 1)