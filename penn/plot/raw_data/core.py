import penn
import jams
import librosa
import numpy as np
from os.path import isfile
import pandas as pd
import torchaudio

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


def extract_notes(jams_track):
    pass


def extract_spectrogram(audio, sr, window_length, hop_length):
    audio_stft = librosa.stft(
            y=audio,
            n_fft=window_length,
            hop_length=hop_length)

    audio_stft = np.log10(np.abs(audio_stft.squeeze(0))**2)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_length)
    times = np.arange(audio_stft.shape[-1]) * hop_length / sr;

    return audio_stft, freqs, times


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


def pitch_stft_with_plotly(pitch_dict, audio_file):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.cpu().numpy()
    stft, freqs, times = extract_spectrogram(audio,
                                             sr=sr,
                                             window_length=2048*4,
                                             hop_length=penn.data.preprocess.GSET_HOPSIZE)

    fig = px.imshow(
            stft, 
            color_continuous_scale=px.colors.continuous.Cividis_r,
            x=times,
            y=freqs,
            aspect='auto',
            origin='lower')

    # fig.update_yaxes(tickvals=freqs)
    fig.update_yaxes(ticks="outside")

    max_pitch = []
    min_pitch = []

    for no_slice, pitch_slice in pitch_dict.items():
        fig = fig.add_trace(go.Scatter(
            name=f"String {no_slice}",
            x = pitch_slice["time"],
            y = pitch_slice["pitch"],
            mode="markers",
            marker=dict (size=5)))

        if pitch_slice["pitch"].size > 0:
            max_pitch.append(pitch_slice["pitch"].max())
            min_pitch.append(pitch_slice["pitch"].min())

    ymax = max(max_pitch)
    ymin = min(min_pitch)

    offset = (ymax - ymin) * 0.1
    ymax += offset
    ymin -= offset

    fig.update_yaxes(range=[ymin, ymax], autorange=False)
        
    fig.show()


def edit_with_plotly(pitch_dict):
    fig = go.FigureWidget()

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

    # create our callback function
    def update_point(trace, points, selector):
        print(points)
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)
        for i in points.point_inds:
            c[i] = '#bae2be'
            s[i] = 20
            with f.batch_update():
                scatter.marker.color = c
                scatter.marker.size = s

    for data in fig.data:
        data.on_click(update_point)

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
    df = pd.read_csv('https://raw.githubusercontent.com/jonmmease/plotly_ipywidget_notebooks/master/notebooks/data/cars/cars.csv')

    # pitch_with_plotly(pitch_dict)
    # edit_with_plotly(pitch_dict)
    pitch_stft_with_plotly(pitch_dict, audio_file)
