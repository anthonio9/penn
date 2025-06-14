import argparse

import penn


###############################################################################
# Partition datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=penn.DATASETS,
        help='The datasets to partition')
    return parser.parse_known_args()[0]


penn.partition.datasets(**vars(parse_args()))
