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
                      no_pred=True,
                      silence=False,
                      linewidth=0.5,
                      linewidth_gt=1.0,
                      fontsize=3,
                      no_legend=False,
                      no_title=False,
                      min_frequency=None,
                      max_frequency=None):

    audio, pred_freq, pred_times, gt_pitch, gt_times, periodicity, logits = \
            penn.common_utils.from_path(
            audio_file,
            ground_truth_file,
            checkpoint,
            silence=silence,
            gpu=gpu,
            start=start,
            duration=duration)

    file_stem = path.basename(audio_file)

    ylim = None
    if min_frequency is not None and max_frequency is not None:
        ylim = [min_frequency, max_frequency]

    if not plot_logits:
        logits = None

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
            logits=logits, 
            fontsize=fontsize,
            linewidth=linewidth,
            linewidth_gt=linewidth_gt,
            legend=not no_legend,
            show_title=not no_title,
            ylim=ylim)
