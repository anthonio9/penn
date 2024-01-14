import json
import random
import numpy as np

import penn


###############################################################################
# Dataset-specific
###############################################################################


def datasets(datasets):
    """Partition datasets"""
    for name in datasets:
        dataset(name)


def dataset(name):
    """Partition dataset"""
    # Get dataset stems
    stems = [file.stem[:-6].split('-')[0] for file in
             (penn.CACHE_DIR / name).glob('*-audio.npy')]

    # Get the unique stems
    stems = list(set(stems))

    # Finally sort them
    stems = sorted(stems)

    if 'gset' in name and penn.GSET_SPLIT_PLAYERS:
        left, right = int((0.7 * 360 // 60) * 60), int((0.85 * 360 // 60) * 60)
        print(f"GSET_SPLIT_PLAYERS, left: {left}, right: {right}")
    else:
        random.seed(penn.RANDOM_SEED)
        random.shuffle(stems)

        # Get split points
        left, right = int(.70 * len(stems)), int(.85 * len(stems))

    # Perform partition
    partition = {
        'train': sorted(stems[:left]),
        'valid': sorted(stems[left:right]),
        'test': sorted(stems[right:])}

    if 'gset' in name and penn.GSET_AUGUMENT is not None:
        # append only to the training partition
        train_part = partition['train']
        noise_stems = [f'snr{snr:02d}' for snr in penn.GSET_AUGUMENT]

        # list for newly created noise file stems
        noise_part = []

        for stem in train_part:
            noise_part.extend(
                    [f"{stem}-{noise_stem}" for noise_stem in noise_stems])

        train_part.extend(noise_part)
        partition['train'] = sorted(train_part)

    # Write partition file
    with open(penn.PARTITION_DIR / f'{name}.json', 'w') as file:
        json.dump(partition, file, indent=4)

