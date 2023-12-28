import penn
import numpy as np
import jams
from amt_tools import tools


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
                    hop_length=penn.data.preprocess.GSET_HOPSIZE_SECONDS,
                    duration=jam.file_metadata.duration)

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(tools.utils.pitch_list_to_stacked_pitch_list(entry_times, slice_pitch_list, string))

    # Determine the total number of observations in the uniform time series
    num_entries = int(np.ceil(jam.file_metadata.duration / penn.data.preprocess.GSET_HOPSIZE_SECONDS)) + 1
    time_steps_array = penn.data.preprocess.GSET_HOPSIZE_SECONDS * np.arange(num_entries)

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


def test_extract_pitch_array_jams2():
    audio_files = (penn.DATA_DIR / 'gset'/ 'audio-mono-mic').glob('*.wav')
    audio_files = sorted(audio_files)

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotations' /
        file.with_suffix('.jams').name.replace('_mic', '')
        for file in audio_files]

    audio_file = audio_files[0]
    pitch_file = pitch_files[0]

    print(f"audio_file: {audio_file}, pitch_file: {pitch_file}")

    jams_track = jams.load(str(pitch_file))
    pitch_array_rtn, time_steps_array_rtn = penn.data.preprocess.extract_pitch_array_jams(jams_track, audio_file)
    
    pitch_array = []

    # Extract all of the pitch annotations
    pitch_data_slices = jams_track.annotations[penn.data.preprocess.JAMS_PITCH_HZ]

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
        string = slice_pitches.annotation_metadata[penn.data.preprocess.JAMS_STRING_IDX]
        slice_names.append(string)

        # Initialize an array/list to hold the times/frequencies associated with each observation
        entry_times, slice_pitch_list = np.empty(0), list()

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero or unvoiced frequencies
            if np.sum(freq) == 0 or not pitch.value['voiced']:
                freq = np.zeros(1)

            # Append the observation time
            entry_times = np.append(entry_times, pitch.time)
            # Append the frequency
            slice_pitch_list.append(freq)

        # Sort the pitch list before resampling just in case it is not already sorted
        entry_times, slice_pitch_list = penn.data.preprocess.sort_pitch_list(entry_times, slice_pitch_list)

        # Align the pitch list with a uniform time grid
        entry_times, slice_pitch_list = penn.data.preprocess.time_series_to_uniform(
                times=entry_times,
                values=slice_pitch_list,
                hop_length=penn.data.preprocess.GSET_HOPSIZE_SECONDS,
                duration=jams_track.file_metadata.duration)

        stacked_pitch_list[slc] = (entry_times, slice_pitch_list)

    pitch_list = [val[1].T for val in stacked_pitch_list.values()] 
    pitch_array = np.vstack(pitch_list)

    assert (pitch_array == pitch_array_rtn).all()


def test_extract_pitch_array_jams22():
    audio_files = (penn.DATA_DIR / 'gset'/ 'audio-mono-mic').glob('*.wav')
    audio_files = sorted(audio_files)

    # Get pitch files
    pitch_files = [
        file.parent.parent /
        'annotations' /
        file.with_suffix('.jams').name.replace('_mic', '')
        for file in audio_files]

    audio_file = audio_files[0]
    pitch_file = pitch_files[0]

    print(f"audio_file: {audio_file}, pitch_file: {pitch_file}")

    jams_track = jams.load(str(pitch_file))
    pitch_array_rtn, time_steps_array_rtn = penn.data.preprocess.extract_pitch_array_jams(jams_track, audio_file)
    pitch_array_test, times_steps_array_test = extract_pitch_array_jams(jams_track, audio_file, True)
    
    assert (pitch_array_test == pitch_array_rtn).all()
