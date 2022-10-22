import itertools

import numpy as np
import torch

import penne


###############################################################################
# Constants
###############################################################################


# MDB analysis parameters
MDB_HOPSIZE = 128 / 44100  # seconds

# PTDB analysis parameters
PTDB_HOPSIZE = .01  # seconds
PTDB_WINDOW_SIZE = .032  # seconds


###############################################################################
# Preprocess datasets
###############################################################################


def datasets(datasets):
    """Preprocess datasets"""
    if 'mdb' in datasets:
        mdb()

    if 'ptdb' in datasets:
        ptdb()


###############################################################################
# Individual datasets
###############################################################################


def mdb():
    """Preprocess mdb dataset"""
    # Get audio files
    audio_files = (penne.DATA_DIR / 'mdb'/ 'audio_stems').glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotation_stems' /
        file.with_suffix('.csv').name
        for file in audio_files]

    # Create cache
    output_directory = penne.CACHE_DIR / 'mdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penne.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing mdb',
        len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penne.load.audio(audio_file)

        # Pad half windows on end of each file for precise alignment
        audio = torch.nn.functional.pad(
            audio,
            (penne.WINDOW_SIZE // 2, penne.WINDOW_SIZE // 2))

        # Save to cache
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Load pitch
        annotations = np.loadtxt(open(pitch_file), delimiter=',')
        times, pitch = annotations[:,0], annotations[:,1]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penne.convert.seconds_to_frames(
            audio.shape[-1] / penne.SAMPLE_RATE)

        # Linearly interpolate to target number of frames
        new_times = (penne.HOPSIZE / penne.SAMPLE_RATE) * np.arange(0, frames)
        pitch = 2 ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    directory = penne.DATA_DIR / 'ptdb' / 'SPEECH DATA'
    male = (directory / 'MALE' / 'MIC').rglob('*.wav')
    female = (directory / 'FEMALE' / 'MIC').rglob('*.wav')
    audio_files = sorted(itertools.chain(male, female))

    # Get pitch files
    pitch_files = [
        file.parent.parent.parent /
        'REF' /
        file.parent.name /
        file.with_suffix('.f0').name.replace('mic', 'ref')
        for file in audio_files]

    # Create cache
    output_directory = penne.CACHE_DIR / 'ptdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    iterator = penne.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing ptdb',
        len(audio_files))
    for i, (audio_file, pitch_file) in iterator:
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penne.load.audio(audio_file)

        # Save to cache
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Simluate the common padding error
        audio = torch.nn.functional.pad(
            audio,
            (penne.WINDOW_SIZE // 2, penne.WINDOW_SIZE // 2))
        np.save(
            output_directory / f'{stem}-misalign.npy',
            audio[:, :-penne.WINDOW_SIZE])

        # Load pitch
        pitch = np.loadtxt(open(pitch_file), delimiter=' ')[:,0]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


###############################################################################
# Utilities
###############################################################################


def interpolate_unvoiced(pitch):
    """Fill unvoiced regions via linear interpolation"""
    voiced = pitch != 0
    pitch = np.log2(pitch)
    pitch[~voiced] = 2 ** np.interp(
        np.where(~voiced)[0],
        np.where(voiced)[0],
        pitch[voiced])
    return pitch, voiced
