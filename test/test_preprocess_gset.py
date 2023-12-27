import penn
import numpy as np
import jams

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
    pitch_array_rtn, time_steps_array_rtn = penn.data.preprocess.extract_pitch_array_jams2(jams_track, audio_file)
    
    pitch_array = []
    time_steps_array = []

    # Extract all of the pitch annotations
    pitch_data_slices = jams_track.annotations[penn.data.preprocess.JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()
    slice_names = []

    max_times = []

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

        try:
            max_times.append(entry_times[-1])
        except IndexError:
            print (f"No entries on string {slc}")

        entry_times = np.array(entry_times)
        slice_pitch_list = np.array(slice_pitch_list)
        stacked_pitch_list[slc] = (entry_times, slice_pitch_list)

    times_full = np.arange(
            start=0, 
            stop=max(max_times), 
            step=penn.data.preprocess.GSET_HOPSIZE_SECONDS)

    print(f"GSET_HOPSIZE_SECONDS: {penn.data.preprocess.GSET_HOPSIZE_SECONDS}")
    print(f"times max: {max(max_times)}")

    pitch_full = np.zeros(shape=(stack_size, len(times_full))) 

    for slc in range(stack_size):
        times, pitch_array = stacked_pitch_list[slc]
        assert len(times) == len(pitch_array)

        breakpoint()
        indxs_full = times_full // penn.data.preprocess.GSET_HOPSIZE_SECONDS
        indxs_slice = times // penn.data.preprocess.GSET_HOPSIZE_SECONDS

        indxs_full = indxs_full.astype(np.int64)
        indxs_slice = indxs_slice.astype(np.int64)

        indxs = np.in1d(indxs_full, indxs_slice, assume_unique=True)
        pitch_full[slc, indxs] = pitch_array.squeeze(axis=-1)

