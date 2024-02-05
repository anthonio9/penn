import itertools
import warnings

import jams
import numpy as np
import torchaudio
import torchutil
from typing import Tuple

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

JAMS_FREQ = "frequency"
JAMS_INDEX = "index"
JAMS_TIMES = "times"

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

        if penn.STRING_INDEX is not None:
            pitch = pitch[penn.STRING_INDEX, :]
            pitch = pitch[None, :]

        if penn.INTERPOLATE_UNVOICED:
            # Fill unvoiced regions via linear interpolation
            pitch, voiced = interpolate_unvoiced(pitch)
        else:
            unvoiced = pitch == 0
            voiced = ~unvoiced

        # FOR sampling rates like 11025, 22050, 44100, resampling isn't necessary
        if GSET_SAMPLE_RATE / penn.SAMPLE_RATE % 1 != 0:
            print("Resampling to penn.SAMPLE_RATE")

            pitch_list = np.vsplit(pitch, pitch.shape[0])
            pitch_list_final = []

            voiced_list = np.vsplit(voiced, voiced.shape[0])
            voiced_list_final = []

            for pitch_arr, voiced_arr in zip(pitch_list, voiced_list):
                # Get target number of frames
                frames = penn.convert.samples_to_frames(audio.shape[-1])

                pitch_arr = pitch_arr[0, :]
                voiced_arr = voiced_arr[0, :]

                # Linearly interpolate to target number of frames
                new_times = penn.HOPSIZE_SECONDS * np.arange(0, frames)
                new_times += penn.HOPSIZE_SECONDS / 2.
                pitch_arr = 2. ** np.interp(new_times, times, np.log2(pitch_arr))

                # Linearly interpolate voiced_arr/unvoiced_arr tokens
                voiced_arr = np.interp(new_times, times, voiced_arr) > .5

                # Check shapes
                assert (
                    penn.convert.samples_to_frames(audio.shape[-1]) ==
                    pitch_arr.shape[-1] ==
                    voiced_arr.shape[-1])

                assert np.logical_not(pitch_arr[voiced_arr] == 0).all()

                pitch_list_final.append(pitch_arr)
                voiced_list_final.append(voiced_arr)

            pitch = np.vstack(pitch_list_final)
            voiced = np.vstack(voiced_list_final)

            if pitch.shape[0] == 1:
                pitch = pitch[0, :]

            if voiced.shape[0] == 1:
                voiced = voiced[0, :]
        else:
            overload = np.abs(audio.shape[-1] // penn.HOPSIZE - pitch.shape[-1])
            # this is a bad, ugly hack, but well, it is what it is, has to be enabled if resampling isn't enabled
            pitch = pitch[..., :-overload]
            voiced = voiced[..., :-overload]

        assert pitch.shape[-1] == audio.shape[-1] // penn.HOPSIZE

        # Save to cache
        np.save(output_directory / f'{stem}-pitch.npy', pitch)
        np.save(output_directory / f'{stem}-voiced.npy', voiced)

        if penn.GSET_AUGUMENT is not None:
            audio_numpy = audio.numpy()
            for snr in penn.GSET_AUGUMENT:
                audio_awgn = awgn_snr(audio_numpy, snr=snr)
                noise_stem = f'snr{snr:02d}'

                # Save as numpy array for fast memory-mapped reads
                np.save(
                    output_directory / f'{stem}-{noise_stem}-audio.npy',
                    audio_awgn.squeeze())

                # Save to cache
                np.save(output_directory / f'{stem}-{noise_stem}-pitch.npy', pitch)
                np.save(output_directory / f'{stem}-{noise_stem}-voiced.npy', voiced)


###############################################################################
# Utilities
###############################################################################


def awgn_snr(x: np.ndarray, snr: int):
    """Add AWGN based on the given SNR value

    Parameters
    ----------
    x : ndarray (N)
        Signal in the time domain, 1-D numpy array of length N.
    snr : int 
        Target signal to noise raitio in dB.

    Returns
    -------
    y : ndarray (N)
        Input signal with the noise added.
    """
    # Calculate signal power and convert to dB 
    x_avg_db = 10 * np.log10(np.mean(x ** 2))

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = x_avg_db - snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x))

    # Noise up the original signal
    y = x + noise.astype(x.dtype)

    return y

def awgn_pwr(x: np.ndarray, pwr: int):
    """Add AWGN of given power to the input signal x

    Parameters
    ----------
    x : ndarray (N)
        Signal in the time domain, 1-D numpy array of length N.
    pwr : int 
        Power of the noise given in dB.

    Returns
    -------
    y : ndarray (N)
        Input signal with the noise added.
    """
    # Convert to linear Watt units
    target_noise_watts = 10 ** (pwd / 10)

    # Generate noise samples
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(x))

    # Noise up the original signal (again) and plot
    y = x + noise.astype(x.dtype)

    return y


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


def sort_pitch_list(times: np.ndarray, pitch_list: list) -> Tuple[np.ndarray, list]:
    """
    Sort a pitch list by frame time.

    Parameters
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames)

    Returns
    ----------
    times : ndarray (N)
      Time in seconds of beginning of each frame, sorted by time
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches corresponding to notes
      N - number of pitch observations (frames), sorted by time
    """

    # Obtain the indices corresponding to the sorted times
    sort_order = list(np.argsort(times))

    # Sort the times
    times = np.sort(times)

    # Sort the pitch list
    pitch_list = [pitch_list[i] for i in sort_order]

    return times, pitch_list


def time_series_to_uniform(times: np.ndarray, values: list, hop_length: float, duration=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a semi-regular time series with gaps into a uniform time series.

    Adapted from mir_eval pull request #336.

    Parameters
    ----------
    times : ndarray
      Array of times corresponding to a time series
    values : list of ndarray
      Observations made at times
    hop_length : number (optional)
      Time interval (seconds) between each observation in the uniform series
    duration : number or None (optional)
      Total length (seconds) of times series
      If specified, should be greater than all observation times

    Returns
    -------
    times : ndarray
      Uniform time array
    values : ndarray
      Observations corresponding to uniform times
    """

    if duration is None:
        # Default the duration to the last reported time in the series
        duration = times[-1]

    # Determine the total number of observations in the uniform time series
    num_entries = int(np.ceil(duration / hop_length)) + 1

    # Attempt to fill in blank frames with the appropriate value
    empty_fill = np.zeros(1)
    new_values = [empty_fill] * num_entries
    new_times = hop_length * np.arange(num_entries)

    if not len(times) or not len(values):
        return new_times, np.array(new_values)

    # Determine which indices the provided observations fall under
    idcs = np.round(times / hop_length).astype(int)

    # Fill the observed values into their respective locations in the uniform series
    for i in range(len(idcs)):
        if times[i] <= duration:
            new_values[idcs[i]] = values[i]

    return new_times, np.array(new_values)


def extract_pitch_array_jams(jam: jams.JAMS, track, uniform=True) -> Tuple[np.ndarray, np.ndarray]:
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
    pitch_data_slices = jam.annotations[JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    slice_names = []
    pitch_list = []
    times_list = []

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
            if np.sum(freq) != 0 and pitch.value['voiced']:
                # Append the observation time
                entry_times = np.append(entry_times, pitch.time)
                # Append the frequency
                slice_pitch_list.append(freq)

        # Sort the pitch list before resampling just in case it is not already sorted
        entry_times, slice_pitch_array = sort_pitch_list(entry_times, slice_pitch_list)

        # Align the pitch list with a uniform time grid
        entry_times, slice_pitch_array = time_series_to_uniform(
                times=entry_times,
                values=slice_pitch_array,
                hop_length=penn.data.preprocess.GSET_HOPSIZE_SECONDS,
                duration=jam.file_metadata.duration)

        times_list.append(entry_times)
        pitch_list.append(slice_pitch_array.T)

    # assert all entry times arrays are of the same lenght
    time_lenghts = [len(times) for times in times_list]
    assert time_lenghts[0] == sum(time_lenghts) / len(time_lenghts)

    time_steps_array = times_list[0]
    pitch_array = np.vstack(pitch_list)

    return pitch_array, time_steps_array


def jams_to_notes(jam: jams.JObject):
    """
    Parameters:
        jams object 
            jams object containing all the information about a track
    Returns:
        notes dict 
            dictionary of notes and their timestamps segregated into strings
            dict {list [list]}
    """
    notes = {}

    # Extract all of the pitch annotations
    pitch_data_slices = jam.annotations[JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the pitch list pertaining to this slice
        slice_pitches = pitch_data_slices[slc]

        # Extract the string label for this slice
        string = slice_pitches.annotation_metadata[JAMS_STRING_IDX]
        
        try: 
            last_index = slice_pitches.data[-1].value[JAMS_INDEX] + 1
            # prepare empty lists for the notes
            note_list = [[] for i in range(last_index)]
            notes_times_list = [[] for i in range(last_index)]

        except IndexError:
            note_list = []
            notes_times_list = []

        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero or unvoiced frequencies
            if np.sum(freq) != 0 and pitch.value['voiced']:
                note_list[pitch.value[JAMS_INDEX]].append(pitch.value[JAMS_FREQ])
                notes_times_list[pitch.value[JAMS_INDEX]].append(pitch.time)

        notes[int(string)] = {
                JAMS_FREQ : note_list,
                JAMS_TIMES : notes_times_list}

    return notes


def notes_to_pitch_array():
    pass


def remove_overhangs(notes_dict: dict):
    pass
