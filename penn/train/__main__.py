import argparse
import shutil
from pathlib import Path

import torchutil

import penn


###############################################################################
# Entry point
###############################################################################


def main(config, datasets, gpu, use_wandb):
    # Create output directory
    directory = penn.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    penn.train(datasets, directory, gpu, use_wandb)

    # Get latest checkpoint
    checkpoint = torchutil.checkpoint.latest_path(directory)

    # Evaluate
    penn.evaluate.datasets(penn.EVALUATION_DATASETS, checkpoint, gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=penn.DATASETS,
        help='The datasets to train on')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    parser.add_argument(
        '-w', 
        '--use_wandb',
        action="store_true",
        help='To use [True] or not [False] to use Weights & Biases logging')
    return parser.parse_args()


main(**vars(parse_args()))
