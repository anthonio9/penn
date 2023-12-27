import itertools
import warnings

from amt_tools import tools
import jams
import numpy as np
import torchaudio
import torchutil
import typing

import penn


###############################################################################
# Constants
###############################################################################


# MDB analysis parameters
MDB_HOPSIZE = 128  # samples
MDB_SAMPLE_RATE = 44100  # samples per second

# PTDB analysis parameters
PTDB_HOPSIZE = 160  # samples
PTDB_SAMPLE_RATE = 16000  # samples per second
PTDB_WINDOW_SIZE = 512  # samples
PTDB_HOPSIZE_SECONDS = PTDB_HOPSIZE / PTDB_SAMPLE_RATE

# GSET parameters
GSET_HOPSIZE = 256
GSET_SAMPLE_RATE = 44100
GSET_HOPSIZE_SECONDS = GSET_HOPSIZE / GSET_SAMPLE_RATE 

##################################################
# JAMS ATTRIBUTES                                #
##################################################

JAMS_NOTE_MIDI = 'note_midi'
JAMS_PITCH_HZ = 'pitch_contour'
JAMS_STRING_IDX = 'data_source'
JAMS_METADATA = 'file_metadata'

###############################################################################
# Preprocess datasets
###############################################################################


@torchutil.notify('preprocess')
def datasets(datasets):
    """Preprocess datasets"""
    if 'mdb' in datasets:
        mdb()

    if 'ptdb' in datasets:
        ptdb()

    if 'gset' in datasets:
        gset()


###############################################################################
# Individual datasets
###############################################################################


def mdb():
    """Preprocess mdb dataset"""
    # Get audio files
    audio_files = (penn.DATA_DIR / 'mdb'/ 'audio_stems').glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotation_stems' /
        file.with_suffix('.csv').name
        for file in audio_files]

    # Create cache
    output_directory = penn.CACHE_DIR / 'mdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    for i, (audio_file, pitch_file) in torchutil.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing mdb',
        total=len(audio_files)
    ):
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penn.load.audio(audio_file)

        # Save as numpy array for fast memory-mapped reads
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penn.SAMPLE_RATE)

        # Load pitch
        annotations = np.loadtxt(open(pitch_file), delimiter=',')
        times, pitch = annotations[:, 0], annotations[:, 1]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penn.convert.samples_to_frames(audio.shape[-1])

        # Linearly interpolate to target number of frames
        new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penn.HOPSIZE_SECONDS / 2.
        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penn.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    directory = penn.DATA_DIR / 'ptdb' / 'SPEECH DATA'
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
    output_directory = penn.CACHE_DIR / 'ptdb'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    for i, (audio_file, pitch_file) in torchutil.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing ptdb',
        total=len(audio_files)
    ):
        stem = f'{i:06d}'

        # Load and resample to PTDB sample rate
        audio, sample_rate = torchaudio.load(audio_file)
        audio = penn.resample(audio, sample_rate, PTDB_SAMPLE_RATE)

        # Remove padding
        offset = PTDB_WINDOW_SIZE - PTDB_HOPSIZE // 2
        if (audio.shape[-1] - 2 * offset) % PTDB_HOPSIZE == 0:
            offset += PTDB_HOPSIZE // 2
        audio = audio[:, offset:-offset]

        # Resample to pitch estimation sample rate
        audio = penn.resample(audio, PTDB_SAMPLE_RATE)

        # Save as numpy array for fast memory-mapped read
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penn.SAMPLE_RATE)

        # Load pitch
        pitch = np.loadtxt(open(pitch_file), delimiter=' ')[:, 0]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penn.convert.samples_to_frames(audio.shape[-1])

        # Get original times
        times = PTDB_HOPSIZE_SECONDS * np.arange(0, len(pitch))
        times += PTDB_HOPSIZE_SECONDS / 2

        # Linearly interpolate to target number of frames
        new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penn.HOPSIZE_SECONDS / 2.

        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penn.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


def gset():
    """Preprocess GuitarSet"""
    # Get audio files
    audio_files = (penn.DATA_DIR / 'gset'/ 'audio-mono-mic').glob('*.wav')
    audio_files = sorted(audio_files)

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotations' /
        file.with_suffix('.jams').name
        for file in audio_files]

    # Create cache
    output_directory = penn.CACHE_DIR / 'gset'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write audio and pitch to cache
    for i, (audio_file, pitch_file) in torchutil.iterator(
        enumerate(zip(audio_files, pitch_files)),
        'Preprocessing gset',
        total=len(audio_files)
    ):
        stem = f'{i:06d}'

        # Load and resample audio
        audio = penn.load.audio(audio_file)

        # Save as numpy array for fast memory-mapped reads
        np.save(
            output_directory / f'{stem}-audio.npy',
            audio.numpy().squeeze())

        # Save audio for listening and evaluation
        torchaudio.save(
            output_directory / f'{stem}.wav',
            audio,
            penn.SAMPLE_RATE)

        # Load pitch
        pitch_file = pitch_file.with_stem(pitch_file.stem.replace("_mic", ""))
        jams_track = jams.load(str(pitch_file))
        pitch, times = extract_pitch_array_jams(
                jams_track, audio_file.stem, uniform=True)
        pitch = pitch[penn.STRING_INDEX, :]

        # Fill unvoiced regions via linear interpolation
        pitch, voiced = interpolate_unvoiced(pitch)

        # Get target number of frames
        frames = penn.convert.samples_to_frames(audio.shape[-1])

        # Linearly interpolate to target number of frames
        new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
        new_times += penn.HOPSIZE_SECONDS / 2.
        pitch = 2. ** np.interp(new_times, times, np.log2(pitch))

        # Linearly interpolate voiced/unvoiced tokens
        voiced = np.interp(new_times, times, voiced) > .5

        # Check shapes
        assert (
            penn.convert.samples_to_frames(audio.shape[-1]) ==
            pitch.shape[-1] ==
            voiced.shape[-1])

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)


###############################################################################
# Utilities
###############################################################################


def interpolate_unvoiced(pitch):
    """Fill unvoiced regions via linear interpolation"""
    unvoiced = pitch == 0

    # Ignore warning of log setting unvoiced regions (zeros) to nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Pitch is linear in base-2 log-space
        pitch = np.log2(pitch)

    try:

        # Interpolate
        pitch[unvoiced] = np.interp(
            np.where(unvoiced)[0],
            np.where(~unvoiced)[0],
            pitch[~unvoiced])

    except ValueError:

        # Allow all unvoiced
        pass

    return 2 ** pitch, ~unvoiced


def extract_pitch_array_jams(jam, track, uniform=True):
    """
    Extract pitch lists spread across slices (e.g. guitar strings) from JAMS annotations into a dictionary.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    uniform : bool
      Whether to place annotations on a uniform time grid

    Returns
    ----------
    pitch_dict : dict
      Dictionary containing pitch_array with pitch values in Hz and time steps array
      pitch_array shape is (S, T), 
      time_steps array is of shape (T, )
      S - number of strings, T - number of time steps
    """
    # Extract all of the pitch annotations
    pitch_data_slices = jam.annotations[tools.constants.JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()
    slice_names = []

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the pitch list pertaining to this slice
        slice_pitches = pitch_data_slices[slc]

        # Extract the string label for this slice
        string = slice_pitches.annotation_metadata[tools.constants.JAMS_STRING_IDX]
        slice_names.append(string)

        # Initialize an array/list to hold the times/frequencies associated with each observation
        entry_times, slice_pitch_list = np.empty(0), list()

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero or unvoiced frequencies
            if np.sum(freq) == 0 or not pitch.value['voiced']:
                freq = np.empty(0)

            # Append the observation time
            entry_times = np.append(entry_times, pitch.time)
            # Append the frequency
            slice_pitch_list.append(freq)

        # Sort the pitch list before resampling just in case it is not already sorted
        entry_times, slice_pitch_list = tools.utils.sort_pitch_list(entry_times, slice_pitch_list)

        if uniform:
            # Align the pitch list with a uniform time grid
            entry_times, slice_pitch_list = tools.utils.time_series_to_uniform(
                    times=entry_times,
                    values=slice_pitch_list,
                    hop_length=GSET_HOPSIZE_SECONDS,
                    duration=jam.file_metadata.duration)

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(tools.utils.pitch_list_to_stacked_pitch_list(entry_times, slice_pitch_list, string))

    # Determine the total number of observations in the uniform time series
    num_entries = int(np.ceil(jam.file_metadata.duration / GSET_HOPSIZE_SECONDS)) + 1
    time_steps_array = GSET_HOPSIZE_SECONDS * np.arange(num_entries)

    pitch_array_slices_list = []

    # for idx, slc in enumerate(slice_names):
    for slc in slice_names:
        # get the list of pitches in hz for slc string
        pitch_list = stacked_pitch_list[slc][1]

        # fill the empty numpy arrays in the pitch_list with zeros
        pitch_list = [np.zeros(1) if pitch.size == 0 else pitch for pitch in pitch_list]

        try: 
            # concatenate the whole thing into a numpy array
            pitch_list = np.concatenate(pitch_list)
        except ValueError:
            print(f"Empty array, track: {track}")
            print(f"Replacing with np.zeros({len(time_steps_array)})")
            pitch_list = np.zeros(len(time_steps_array))

        # append the slice to a list of all slices 
        pitch_array_slices_list.append(pitch_list)

    try: 
        pitch_array = np.vstack(pitch_array_slices_list)
    except ValueError as err:
        print(f"{err}, track: {track}, slice lengths: {[len(slice) for slice in pitch_array_slices_list]}")

    assert pitch_array.shape == (stack_size, time_steps_array.size)

    return pitch_array, time_steps_array


def extract_pitch_array_jams2(jam: jams.JAMS, track, uniform=True):
    """
    Extract pitch lists spread across slices (e.g. guitar strings) from JAMS annotations into a dictionary.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    uniform : bool
      Whether to place annotations on a uniform time grid

    Returns
    ----------
    pitch_dict : dict
      Dictionary containing pitch_array with pitch values in Hz and time steps array
      pitch_array shape is (S, T), 
      time_steps array is of shape (T, )
      S - number of strings, T - number of time steps
    """
    pitch_array = []
    time_steps_array = []

    # Extract all of the pitch annotations
    pitch_data_slices = jam.annotations[JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()
    slice_names = []

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the pitch list pertaining to this slice
        slice_pitches = pitch_data_slices[slc]

        # Extract the string label for this slice
        string = slice_pitches.annotation_metadata[JAMS_STRING_IDX]
        slice_names.append(string)

        # Initialize an array/list to hold the times/frequencies associated with each observation
        entry_times, slice_pitch_list = np.empty(0), list()

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero or unvoiced frequencies
            if np.sum(freq) == 0 or not pitch.value['voiced']:
                freq = np.empty(0)

            # Append the observation time
            entry_times = np.append(entry_times, pitch.time)
            # Append the frequency
            slice_pitch_list.append(freq)

    return pitch_array, time_steps_array
