import numpy as np
import torch
import torchutil
from os import path
import jams

import penn


def logits_from_audio(
    audio,
    sample_rate,
    checkpoint=None,
    gpu=None,
    silence=False):

    logits = []
    logits_silence = []

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
        logits_dict = penn.infer(frames, checkpoint=checkpoint)
        logits.append(logits_dict[penn.model.KEY_LOGITS].detach())

        try:
            logits_silence.append(
                    torch.sigmoid(
                        logits_dict[penn.model.KEY_SILENCE].detach()))
        except KeyError as e:
            print(f"from_audio KeyError: {e}")

    # Concatenate results
    if penn.FCN:
        logits = torch.cat(logits, dim=-2)
        logits = logits.permute(3, 1, 2, 0)

        try:
            logits_silence = torch.cat(logits_silence, dim=-2)
            # logits_silence = logits_silence.permute(2, 1, 0)
        except (RuntimeError, ValueError) as e:
            logits_silence = []
            print(f"from_audio exception: {e}")
    else:
        logits = torch.cat(logits)

    logits_dict = {}
    logits_dict[penn.model.KEY_LOGITS] = logits

    if len(logits_silence) > 0:
        logits_dict[penn.model.KEY_SILENCE] = logits_silence

    return logits_dict


def process_logits(
        logits_dict,
        silence=False,
        as_numpy=False):

    pitch = None
    times = None

    if not silence:
        logits_dict.pop(penn.model.KEY_SILENCE, None)

    # pitch is in Hz
    predicted_bins, pitch, periodicity = penn.postprocess(logits_dict)

    times = penn.HOPSIZE_SECONDS * torch.arange(pitch[0].shape[-1])

    logits = logits_dict[penn.model.KEY_LOGITS]

    logits = torch.nan_to_num(
            logits,
            neginf=torch.min(logits[torch.logical_not(torch.isneginf(logits))]),
            posinf=torch.max(logits[torch.logical_not(torch.isposinf(logits))])
            )
    logits = torch.softmax(logits, dim=2)

    if silence and len(logits_dict[penn.model.KEY_SILENCE]) > 0:
        periodicity = logits_dict[penn.model.KEY_SILENCE]

    if as_numpy:
        pitch = pitch.detach().numpy()[0, ...]
        # pitch = np.split(pitch, pitch.shape[0])
        times = times.detach().numpy()
        periodicity = periodicity.detach().numpy()
        logits = logits.detach().numpy()

    return pitch, times, periodicity, logits


def from_audio(
    audio,
    sample_rate,
    checkpoint=None,
    gpu=None,
    silence=False,
    as_numpy=False):
    """Plot logits with pitch overlay"""

    logits_dict = logits_from_audio(
            audio=audio,
            sample_rate=sample_rate,
            checkpoint=checkpoint,
            gpu=gpu,
            silence=silence)

    return process_logits(logits_dict, as_numpy=as_numpy, silence=silence)


def load_audio(audio_file : str):
    assert path.isfile(audio_file)

    filename, file_extension = path.splitext(audio_file)

    if file_extension == '.npy':
        audio = np.load(audio_file)
        audio = torch.from_numpy(audio)
        audio = penn.resample(audio, penn.SAMPLE_RATE)
        audio = torch.unsqueeze(audio, dim=0)
    else:
        audio = penn.load.audio(audio_file)

    return audio


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


def from_path(
        audio_file,
        ground_truth_file,
        checkpoint,
        silence=False,
        gpu=None,
        start : float=0.0,
        duration : float=None,
        as_numpy : bool=False):
    """
    Load audio file and the groundtruth and extract pitch predictions
    """

    # Load audio
    audio = load_audio(audio_file)

    if checkpoint is None:
        raise ValueError("checkpoint parameter not provided! Can't proceed, stopping!")

    # get the timestamps in frame numbers
    start_frame = round(start * penn.SAMPLE_RATE)

    end_frame = -1
    if duration is not None:
        end_frame = round((start + duration)* penn.SAMPLE_RATE)

    audio = audio[..., start_frame : end_frame]

    # get logits
    pred_freq, pred_times, periodicity, logits =\
            penn.common_utils.from_audio(
                    audio,
                    penn.SAMPLE_RATE,
                    checkpoint,
                    gpu,
                    silence=silence,
                    as_numpy=as_numpy)
    pred_times += start

    # get the ground truth
    gt_pitch, gt_times = penn.common_utils.get_ground_truth(ground_truth_file, start, duration)

    # convert the ground truth to torch tensors to be compatible with metrics class
    if not as_numpy:
        gt_pitch = torch.from_numpy(gt_pitch)
        gt_times = torch.from_numpy(gt_times)

        if len(gt_pitch.shape) == 2:
            gt_pitch = gt_pitch.unsqueeze(dim=0)

    return audio, pred_freq, pred_times, gt_pitch, gt_times, periodicity, logits
