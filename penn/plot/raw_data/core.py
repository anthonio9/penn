import penn
import jams
import scipy
import numpy as np
from os.path import isfile

import plotly.express as px
import plotly.graph_objects as go


def extract_pitch(jams_track):
    # Extract all of the pitch annotations
    pitch_data_slices = jams_track.annotations[penn.data.preprocess.JAMS_PITCH_HZ]

    pitch_dict = {}

    for slice in pitch_data_slices:
        string_indx = slice.annotation_metadata[penn.data.preprocess.JAMS_STRING_IDX]
        pitch_slice_dict = pitch_dict[int(string_indx)] = {}

        freq_list = []
        time_list = []

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice:
            # Extract the pitch
            freq = pitch.value['frequency']
            time = pitch.time

            if np.sum(freq) != 0 and pitch.value['voiced']:
                freq_list.append(freq)
                time_list.append(time)

        pitch_slice_dict["pitch"] = np.array(freq_list)
        pitch_slice_dict["time"] = np.array(time_list)

    return pitch_dict

def pitch_with_plotly(pitch_dict):
    # fig = px.scatter(y=pitch_dict[0]["pitch"], x=pitch_dict[0]["time"])

    fig = go.Figure()

    for no_slice, pitch_slice in pitch_dict.items():
        fig = fig.add_trace(go.Scatter(
            name=f"String {no_slice}",
            x = pitch_slice["time"],
            y = pitch_slice["pitch"],
            mode="markers",
            marker=dict (
                size=5
                )
            ))
    fig.show()



def from_data(data_dir, file_stem):
    file_stem = file_stem.split('.')[0].split('mic')[0]

    pitch_file = data_dir / 'annotations' / f"{file_stem}.jams"
    audio_file = data_dir / 'audio-mono-mic' / f"{file_stem}_mic.wav"

    print(audio_file)
    print(pitch_file)

    assert isfile(audio_file)
    assert isfile(pitch_file)

    jams_track = jams.load(str(pitch_file))
    pitch_dict = extract_pitch(jams_track)
    pitch_with_plotly(pitch_dict)
