import numpy as np
import torch
from os.path import isfile

import penn

import torchaudio
import torchutil
import jams
import plotly.express as px
import plotly.graph_objects as go

# this funciton is supposed to plot any audio track with pitch preditions over the given ground truth file, 
def from_audio(
    audio,
    sample_rate,
    checkpoint=None,
    gpu=None):
    """Plot logits with pitch overlay"""
    logits = []

    # Preprocess audio
    for frames in penn.preprocess(
        audio,
        sample_rate,
        batch_size=penn.EVALUATION_BATCH_SIZE,
        center='half-hop'
    ):

        # Copy to device
        frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits.append(penn.infer(frames, checkpoint=checkpoint).detach())

    # Concatenate results
    logits = torch.cat(logits)
    pitch = None
    times = None

    with torchutil.time.context('decode'):
        # pitch is in Hz
        predicted, pitch, periodicity = penn.postprocess(logits)
        pitch = pitch.detach().numpy()[0, ...]
        pitch = np.split(pitch, pitch.shape[0])
        times = penn.HOPSIZE_SECONDS * np.arange(pitch[0].shape[-1])

    return pitch, times


def get_ground_truth(ground_truth_file):
    assert isfile(ground_truth_file)

    jams_track = jams.load(str(ground_truth_file))
    notes_dict = penn.data.preprocess.jams_to_notes(jams_track)
    pitch_dict = penn.data.preprocess.notes_dict_to_pitch_dict(notes_dict)
    return pitch_dict


def plot_over_gt_with_plotly(audio, sr, pred_freq, pred_times, gt, return_fig=False):
    stft, freqs, times = penn.plot.raw_data.extract_spectrogram(audio,
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

    max_pitch = []
    min_pitch = []
    #
    # for no_slice, pitch_slice in gt.items():
    #     fig = fig.add_trace(go.Scatter(
    #         name=f"String {no_slice}",
    #         x = pitch_slice["times"],
    #         y = pitch_slice["frequency"],
    #         mode="markers",
    #         marker=dict (size=5)))
    #
    #     if pitch_slice["frequency"].size > 0:
    #         max_pitch.append(pitch_slice["frequency"].max())
    #         min_pitch.append(pitch_slice["frequency"].min())
    #
    for no_slice, pitch_slice in enumerate(pred_freq):
        fig = fig.add_trace(go.Scatter(
            name=f"String {no_slice}",
            x = pred_times,
            y = pitch_slice.reshape(-1),
            mode="markers",
            marker=dict (size=5)))

        if pitch_slice.size > 0:
            max_pitch.append(pitch_slice.max())
            min_pitch.append(pitch_slice.min())

    ymax = max(max_pitch)
    ymin = min(min_pitch)

    offset = (ymax - ymin) * 0.1
    ymax += offset
    ymin -= offset

    fig.update_yaxes(range=[ymin, ymax], autorange=False)
    if return_fig:
        return fig
        
    fig.show()
    

def from_file_to_file(audio_file, ground_truth_file, checkpoint, output_file=None, gpu=None):
    # Load audio
    audio = penn.load.audio(audio_file)

    if checkpoint is None:
        return 

    # get logits
    pred_freq, pred_times = from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)

    # get the ground truth
    gt = get_ground_truth(ground_truth_file)

    # get the stft of the audio
    audio, sr = torchaudio.load(audio_file)
    audio = audio.cpu().numpy()

    # now that we have both ground truth, STFT and the preditcted pitch, plot all with matplotlib and plotly
    # well, do we have predicted pitch?
    # plot_over_gt_with_plotly(audio, sr, pred_freq, pred_times, gt)
    
    penn.plot.to_latex.mplt.plot_with_matplotlib(audio, sr, pred_freq, pred_times, gt)
