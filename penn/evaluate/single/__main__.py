import argparse
from pathlib import Path

import penn


###############################################################################
# Create figure
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create logits figure')
    parser.add_argument(
        '--audio_file',
        type=Path,
        help='The audio file to plot the logits of')
    parser.add_argument(
        '--ground_truth_file',
        type=Path,
        help='The ground truth file')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file to use for inference')
    parser.add_argument(
        '--output_file',
        type=Path,
        help='The jpg file to save the plot')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference')
    parser.add_argument(
        '--start',
        type=float,
        default=0,
        help='Start timestamp of the audio file in seconds')
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration of the audio excerpt in seconds')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.5,
        help='Periodicity threshold value')
    parser.add_argument(
        '-s',
        '--silence',
        action='store_true',
        help='Set to replace periodicity estimation with voicing head predictions')
    return parser.parse_known_args()[0]


penn.evaluate.single.from_file_to_file(**vars(parse_args()))
