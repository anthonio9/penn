import penn
import jams
import librosa
import numpy as np
from os.path import isfile
import pandas as pd
import torchaudio

import plotly.express as px
import plotly.graph_objects as go


MIDI_TIME = 0
MIDI_DURATION = 1
MIDI_VALUE = 2


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


def extract_midi(jams_track):
    midi_data_slices = jams_track.annotations[penn.data.preprocess.JAMS_NOTE_MIDI]

    midi_dict = {}

    for slice in midi_data_slices:
        string_indx = slice.annotation_metadata[penn.data.preprocess.JAMS_STRING_IDX]
        midi_slice_dict = midi_dict[string_indx] = {}

        midi_list = []
        time_list = []

        for midi_note in slice.data:
            midi = midi_note[MIDI_VALUE] 
            time = midi_note[MIDI_TIME]
            duration = midi_note[MIDI_DURATION]

            # put None in between the notes
            if len(midi_list) != 0:
                time_none = (time_list[-1] + time) / 2
                midi_list.append(None)
                time_list.append(time_none)

            # starting timestamp
            midi_list.append(penn.convert.midi_to_frequency(midi))
            time_list.append(time)

            # ending timestamp
            midi_list.append(penn.convert.midi_to_frequency(midi))
            time_list.append(time + duration)

        midi_slice_dict["pitch"] = np.array(midi_list)
        midi_slice_dict["time"] = np.array(time_list)

    return midi_dict


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


def pitch_stft_with_plotly(pitch_dict, audio_file, return_fig=False):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.cpu().numpy()
    stft, freqs, times = extract_spectrogram(audio,
                                             sr=sr,
                                             window_length=2048*4,
                                             hop_length=penn.data.preprocess.GSET_HOPSIZE)

    fig = px.imshow(
            stft, 
            color_continuous_scale="aggrnyl",
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

    if return_fig:
        return fig
        
    fig.show()
    

def pitch_midi_stft_with_plotly(pitch_dict, midi_dict, audio_file):
    fig = pitch_stft_with_plotly(pitch_dict, audio_file, True)

    for no_slice, midi_slice in midi_dict.items():
        fig = fig.add_trace(go.Scatter(
            x = midi_slice["time"],
            y = midi_slice["pitch"],
            mode="lines+markers",
            marker=dict (size=3)))

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
    midi_dict = extract_midi(jams_track)

    notes_dict = penn.data.preprocess.jams_to_notes(jams_track)
    removed_overhangs = penn.data.preprocess.remove_overhangs(notes_dict)
    pitch_dict = penn.data.preprocess.notes_dict_to_pitch_dict(removed_overhangs)
    # pitch_with_plotly(pitch_dict)
    # edit_with_plotly(pitch_dict)
    pitch_stft_with_plotly(pitch_dict, audio_file)
    pitch_midi_stft_with_plotly(pitch_dict, midi_dict, audio_file)
