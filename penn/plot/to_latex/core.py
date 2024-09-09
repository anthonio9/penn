import numpy as np
import torch
from os import path

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

        if penn.FCN:
            chunks = frames.shape[0]
            frames_chunks = frames.chunk(chunks, dim=0)
            frames = torch.cat(frames_chunks, dim=-1)

        # Infer
        logits.append(penn.infer(frames, checkpoint=checkpoint).detach())

    # Concatenate results
    if penn.FCN:
        logits = torch.cat(logits, dim=-2)
        logits = logits.permute(3, 1, 2, 0)
    else:
        logits = torch.cat(logits)
    pitch = None
    times = None
    periodicity = None

    with torchutil.time.context('decode'):
        # pitch is in Hz
        predicted, pitch, periodicity = penn.postprocess(logits)
        pitch = pitch.detach().numpy()[0, ...]
        # pitch = np.split(pitch, pitch.shape[0])
        times = penn.HOPSIZE_SECONDS * np.arange(pitch[0].shape[-1])
        periodicity = periodicity.detach().numpy()

    logits = torch.nan_to_num(
            logits,
            neginf=torch.min(logits[torch.logical_not(torch.isneginf(logits))]),
            posinf=torch.max(logits[torch.logical_not(torch.isposinf(logits))])
            )
    logits = torch.softmax(logits, dim=2)

    return pitch, times, periodicity, logits


def get_ground_truth(ground_truth_file,
                     start : float=0,
                     duration : float=None,
                     hop_length_seconds : float=penn.HOPSIZE_SECONDS):
    assert path.isfile(ground_truth_file)

    filename, file_extension = path.splitext(ground_truth_file)

    if file_extension == '.npy' and hop_length_seconds is not None:
        pitch_array = np.load(ground_truth_file)
        times_array = np.arange(pitch_array.shape[-1]) * hop_length_seconds
        voiced_path = str(ground_truth_file).replace("pitch", "voiced")

        if path.isfile(voiced_path):
            voiced_array = np.load(voiced_path)
            pitch_array[np.logical_not(voiced_array)] = 0

    elif file_extension == '.jams':
        jams_track = jams.load(str(ground_truth_file))
        duration = jams_track.file_metadata.duration    
        notes_dict = penn.data.preprocess.jams_to_notes(jams_track)
        pitch_array, times_array = penn.data.preprocess.notes_dict_to_pitch_array(notes_dict, duration)
    else:
        raise ValueError("File extension is not supported")

    start_frame = 0
    end_frame = -1

    start_frame = np.argmin(np.abs(times_array - start))

    if duration is not None:
        end_frame = np.argmin(np.abs(times_array - start - duration))

    return pitch_array[..., start_frame:end_frame], times_array[..., start_frame:end_frame]


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
    

def from_file_to_file(audio_file,
                      ground_truth_file,
                      checkpoint,
                      output_file=None,
                      gpu=None,
                      start : float=0.0,
                      duration : float=None,
                      multipitch=False,
                      threshold=0.5,
                      plot_logits=False,
                      no_pred=True):
    # Load audio
    audio = penn.load.audio(audio_file)

    file_stem = path.basename(audio_file)

    if checkpoint is None:
        return 

    # get the timestamps in frame numbers
    start_frame = round(start * penn.SAMPLE_RATE)

    end_frame = -1
    if duration is not None:
        end_frame = round((start + duration)* penn.SAMPLE_RATE)

    audio = audio[..., start_frame : end_frame]

    # get logits
    pred_freq, pred_times, periodicity, logits = from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)
    pred_times += start

    if not plot_logits:
        logits = None

    # get the ground truth
    gt_pitch, gt_times = get_ground_truth(ground_truth_file, start, duration)

    # get the stft of the audio
    audio, sr = torchaudio.load(audio_file)
    audio = audio.cpu().numpy()

    # get the timestamps in frame numbers
    start_frame = round(start * sr)

    end_frame = -1
    if duration is not None:
        end_frame = round((start + duration)* sr)

    audio = audio[..., start_frame:end_frame]

    # now that we have both ground truth, STFT and the preditcted pitch, plot all with matplotlib and plotly
    # well, do we have predicted pitch?
    # plot_over_gt_with_plotly(audio, sr, pred_freq, pred_times, gt)

    if no_pred:
        pred_freq   = None
        pred_times  = None
        periodicity = None
    
    penn.plot.to_latex.mplt.plot_with_matplotlib(
            title=file_stem,
            audio=audio,
            sr=sr,
            pred_pitch=pred_freq, 
            pred_times=pred_times,
            gt_pitch=gt_pitch,
            gt_times=gt_times,
            periodicity=periodicity, 
            time_offset=start,
            mutlipitch=multipitch,
            threshold=threshold,
            logits=logits)
