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
        '--data_dir',
        type=Path,
        required=True,
        help='The audio file to plot the logits of')
    parser.add_argument(
        '--file_stem',
        type=str,
        required=True,
        help='The jpg file to save the plot')
    return parser.parse_known_args()[0]


penn.plot.raw_data.from_data(**vars(parse_args()))

