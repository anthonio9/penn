import numpy as np
import torch
from os.path import isfile

import penn

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

    return logits


def get_ground_truth(ground_truth_file):
    assert isfile(ground_truth_file)

    pitch_dict = extract_pitch(jams_track)
    return pitch_dict


def plot_over_gt(stft, logits, gt):
    pass
    


def from_file_to_file(audio_file, ground_truth_file, checkpoint, output_file=None, gpu=None):
    # Load audio
    audio = penn.load.audio(audio_file)

    if checkpoint is None:
        return 

    checkpoint = torch.load(checkpoint, map_location='cpu')

    # Initialize model
    model = penn.Model()

    # Load from disk
    model.load_state_dict(checkpoint['model'])

    # get logits
    logits = from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)

    # get the ground truth
    gt = get_ground_truth(ground_truth_file)

    # get the stft of the audio
    audio, sr = torchaudio.load(audio_file)
    audio = audio.cpu().numpy()
    stft, freqs, times = extract_spectrogram(audio,
                                             sr=sr,
                                             window_length=2048*4,
                                             hop_length=penn.data.preprocess.GSET_HOPSIZE)
