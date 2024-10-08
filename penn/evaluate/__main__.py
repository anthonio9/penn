import argparse
from pathlib import Path

import penn


###############################################################################
# Evaluate pitch and periodicity estimation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=penn.EVALUATION_DATASETS,
        help='The datasets to evaluate on')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')
    parser.add_argument(
        '-b',
        '--benchmark',
        action='store_true',
        help='Enable benchmaring the model, get times of inference on CPU and GPU')
    parser.add_argument(
        '-ep',
        '--evaluate-periodicity',
        action='store_true',
        help='Enable periodicity evaluation')
    parser.add_argument(
        '-i',
        '--iterations',
        type=int,
        help='number of tested data pieces')

    return parser.parse_known_args()[0]


penn.evaluate.datasets(**vars(parse_args()))
