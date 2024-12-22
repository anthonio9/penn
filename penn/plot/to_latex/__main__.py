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
        '-m',
        '--multipitch',
        action='store_true',
        help='Set to make a separate plot for every string')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.5,
        help='Periodicity threshold value')
    parser.add_argument(
        '-l', 
        '--plot_logits',
        action='store_true',
        help='Set to plot model output logits instead of STFT')
    parser.add_argument(
        '-np',
        '--no_pred',
        action='store_true',
        help='Set to omit the plots of the predicted values')
    parser.add_argument(
        '-s',
        '--silence',
        action='store_true',
        help='Set to replace periodicity estimation with silence head predictions')
    parser.add_argument(
        '-lw',
        '--linewidth',
        type=float,
        default=0.5,
        help='Set the line width of the predicted pitch')
    parser.add_argument(
        '-lwgt',
        '--linewidth_gt',
        type=float,
        default=1,
        help='Set the line width of the ground truth pitch')
    parser.add_argument(
        '-fs',
        '--fontsize',
        type=float,
        default=1,
        help='Set the line width of the ground truth pitch')
    parser.add_argument(
        '-noleg',
        '--no_legend',
        action='store_true',
        help='Set in order to omit legend plotting')
    return parser.parse_known_args()[0]


penn.plot.to_latex.from_file_to_file(**vars(parse_args()))

