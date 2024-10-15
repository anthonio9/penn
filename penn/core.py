import contextlib
import functools
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import huggingface_hub
import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
import torchutil

import penn


###############################################################################
# Pitch and periodicity estimation
###############################################################################


def from_audio(
        audio: torch.Tensor,
        sample_rate: int = penn.SAMPLE_RATE,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation

    Args:
        audio: The audio to extract pitch and periodicity from
        sample_rate: The audio sample rate
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
        periodicity: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    """
    pitch, periodicity = [], []

    # Preprocess audio
    for frames in preprocess(
        audio,
        sample_rate,
        hopsize,
        batch_size,
        center
    ):

        # Copy to device
        with torchutil.time.context('copy-to'):
            frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

        # Infer
        logits = infer(frames, checkpoint).detach()

        # Postprocess
        with torchutil.time.context('postprocess'):
            result = postprocess(logits, fmin, fmax)
            pitch.append(result[1])
            periodicity.append(result[2])

    # Concatenate results
    pitch, periodicity = torch.cat(pitch, 1), torch.cat(periodicity, 1)

    # Maybe interpolate unvoiced regions
    if interp_unvoiced_at is not None:
        pitch = penn.voicing.interpolate(
            pitch,
            periodicity,
            interp_unvoiced_at)

    return pitch, periodicity


def from_file(
        file: Path,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation from audio on disk

    Args:
        file: The audio file
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Load audio
    with torchutil.time.context('load'):
        audio, sample_rate = torchaudio.load(file)

    # Inference
    return from_audio(
        audio,
        sample_rate,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        interp_unvoiced_at,
        gpu)


def from_file_to_file(
        file: Path,
        output_prefix: Optional[Path] = None,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> None:
    """Perform pitch and periodicity estimation from audio on disk and save

    Args:
        file: The audio file
        output_prefix: The file to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on
    """
    # Inference
    pitch, periodicity = from_file(
        file,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        interp_unvoiced_at,
        gpu)

    # Move to cpu
    with torchutil.time.context('copy-from'):
        pitch, periodicity = pitch.cpu(), periodicity.cpu()

    # Save to disk
    with torchutil.time.context('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files: List[Path],
    output_prefixes: Optional[List[Path]] = None,
    hopsize: float = penn.HOPSIZE_SECONDS,
    fmin: float = penn.FMIN,
    fmax: float = penn.FMAX,
    checkpoint: Optional[Path] = None,
    batch_size: Optional[int] = None,
    center: str = 'half-window',
    interp_unvoiced_at: Optional[float] = None,
    num_workers: int = penn.NUM_WORKERS,
    gpu: Optional[int] = None
) -> None:
    """Perform pitch and periodicity estimation from files on disk and save

    Args:
        files: The audio files
        output_prefixes: Files to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        num_workers: Number of CPU threads for async data I/O
        gpu: The index of the gpu to run inference on
    """
    # Maybe use default output filenames
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in files]

    # Single-threaded
    if num_workers == 0:

        # Iterate over files
        for i, (file, output_prefix) in torchutil.iterator(
            enumerate(zip(files, output_prefixes)),
            f'{penn.CONFIG}',
            total=len(files)
        ):

            # Infer
            from_file_to_file(
                file,
                output_prefix,
                hopsize,
                fmin,
                fmax,
                checkpoint,
                batch_size,
                center,
                interp_unvoiced_at,
                gpu)

    # Multi-threaded
    else:

        # Initialize multi-threaded dataloader
        loader = inference_loader(
            files,
            hopsize,
            batch_size,
            center,
            int(math.ceil(num_workers / 2)))

        # Maintain file correspondence
        output_prefixes = {
            file: output_prefix
            for file, output_prefix in zip(files, output_prefixes)}

        # Setup multiprocessing
        futures = []
        pool = mp.get_context('spawn').Pool(max(1, num_workers // 2))

        # Setup progress bar
        progress = torchutil.iterator(
            range(len(files)),
            penn.CONFIG,
            total=len(files))

        try:

            # Track residual to fill batch
            residual_files = []
            residual_frames = torch.zeros((0, 1, 1024))
            residual_lengths = torch.zeros((0,), dtype=torch.long)

            # Iterate over data
            pitch, periodicity = torch.zeros((1, 0)), torch.zeros((1, 0))

            for frames, lengths, input_files in loader:

                # Prepend residual
                if residual_files:
                    frames = torch.cat((residual_frames, frames), dim=0)
                    lengths = torch.cat((residual_lengths, lengths))
                    input_files = residual_files + input_files

                i = 0
                while batch_size is None or i + batch_size <= len(frames):

                    # Copy to device
                    size = len(frames) if batch_size is None else batch_size
                    batch_frames = frames[i:i + size].to(
                        'cpu' if gpu is None else f'cuda:{gpu}')

                    # Infer
                    logits = infer(batch_frames, checkpoint).detach()

                    # Postprocess
                    results = postprocess(logits, fmin, fmax)

                    # Append to residual
                    pitch = torch.cat((pitch, results[1].cpu()), dim=1)
                    periodicity = torch.cat(
                        (periodicity, results[2].cpu()),
                        dim=1)

                    i += len(batch_frames)

                # Save to disk
                j, k = 0, 0
                for length, file in zip(lengths, input_files):

                    # Slice and save in another process
                    if j + length <= pitch.shape[-1]:
                        futures.append(
                            pool.apply_async(
                                save_worker,
                                args=(
                                    output_prefixes[file],
                                    pitch[:, j:j + length],
                                    periodicity[:, j:j + length],
                                    interp_unvoiced_at)))
                        while len(futures) > 100:
                            futures = [f for f in futures if not f.ready()]
                            time.sleep(.1)
                        j += length
                        k += 1
                        progress.update()
                    else:
                        break

                # Setup residual for next iteration
                pitch = pitch[:, j:]
                periodicity = periodicity[:, j:]
                residual_files = input_files[k:]
                residual_lengths = lengths[k:]
                residual_frames = frames[i:]

            # Handle final files
            if residual_frames.numel():

                # Copy to device
                batch_frames = residual_frames.to(
                    'cpu' if gpu is None else f'cuda:{gpu}')

                # Infer
                logits = infer(batch_frames, checkpoint).detach()

                # Postprocess
                results = postprocess(logits, fmin, fmax)

                # Append to residual
                pitch = torch.cat((pitch, results[1].cpu()), dim=1)
                periodicity = torch.cat((periodicity, results[2].cpu()), dim=1)

                # Save
                i = 0
                for length, file in zip(residual_lengths, residual_files):

                    # Slice and save in another process
                    if i + length <= pitch.shape[-1]:
                        futures.append(
                            pool.apply_async(
                                save_worker,
                                args=(
                                    output_prefixes[file],
                                    pitch[:, i:i + length],
                                    periodicity[:, i:i + length],
                                    interp_unvoiced_at)))
                        while len(futures) > 100:
                            futures = [f for f in futures if not f.ready()]
                            time.sleep(.1)
                        i += length
                        progress.update()

            # Wait
            for future in futures:
                future.wait()

        finally:

            # Shutdown multiprocessing
            pool.close()
            pool.join()

            # Close progress bar
            progress.close()


###############################################################################
# Inference pipeline stages
###############################################################################


def infer(frames, checkpoint=None):
    """Forward pass through the model"""
    # Time model loading
    with torchutil.time.context('model'):

        # Load and cache model
        if (
            not hasattr(infer, 'model') or
            infer.checkpoint != checkpoint or
            infer.device != frames.device
        ):

            # Initialize model
            model = penn.Model()

            # Maybe download from HuggingFace
            if checkpoint is None:
                checkpoint = huggingface_hub.hf_hub_download(
                    'maxrmorrison/fcnf0-plus-plus',
                    'fcnf0++.pt')
                infer.checkpoint = None
            else:
                infer.checkpoint = checkpoint
            checkpoint = torch.load(checkpoint, map_location='cpu')

            # Load from disk
            model.load_state_dict(checkpoint['model'])
            infer.device = frames.device

            # Move model to correct device (no-op if devices are the same)
            infer.model = model.to(frames.device)

    # Time inference
    with torchutil.time.context('infer'):

        # Prepare model for inference
        with inference_context(infer.model):

            # Infer
            logits_dict = infer.model(frames)

        # If we're benchmarking, make sure inference finishes within timer
        if penn.BENCHMARK and logits_dict[penn.model.KEY_LOGITS].device.type == 'cuda':
            torch.cuda.synchronize(logits_dict[penn.model.KEY_LOGITS].device)

        return logits_dict


def postprocess(logits_dict : dict[str, torch.Tensor],
                fmin : float=penn.FMIN,
                fmax : float=penn.FMAX) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert model output to pitch and periodicity.

    Parameters
    ----------
    logits_dict : dict of torch.Tensors
        Dictionary of model outputs with penn.model.KEY_LOGITS and optionally penn.model.KEY_SILENCE keys.
        If penn.model.KEY_SILENCE is present, then periodicity is set to the sigmoid of the silence logits.
    fmin : float
        The frequency of the smallest bin.
    fmax : float
        The frequency of the largest bin.

    Returns
    -------
    tuple of torch.Tensors [pitch, periodicity]
        Predicted pitch represented in both raw bins and frequency tensors followed by the periodicity.
    """
    # Turn off gradients
    with torch.inference_mode():

        # Convert frequency range to pitch bin range
        minidx = penn.convert.frequency_to_bins(torch.tensor(fmin))
        maxidx = penn.convert.frequency_to_bins(
            torch.tensor(fmax),
            torch.ceil)

        # Remove frequencies outside of allowable range
        # below works with shapes [128, 1440, 1] and [128, 6, 1440, 1]
        logits = logits_dict[penn.model.KEY_LOGITS]
        logits[..., :minidx, :] = -float('inf')
        logits[..., maxidx:, :] = -float('inf')

        # Decode pitch from logits
        if penn.DECODER == 'argmax':
            bins, pitch = penn.decode.argmax(logits)
        elif penn.DECODER.startswith('viterbi'):
            bins, pitch = penn.decode.viterbi(logits)
        elif penn.DECODER == 'local_expected_value':
            bins, pitch = penn.decode.local_expected_value(logits)
        else:
            raise ValueError(f'Decoder method {penn.DECODER} is not defined')

        # Decode periodicity from logits
        if penn.model.KEY_SILENCE in logits_dict:
            periodicity = torch.sigmoid(logits_dict[penn.model.KEY_SILENCE])
        elif penn.PERIODICITY == 'entropy':
            periodicity = penn.periodicity.entropy(logits)
        elif penn.PERIODICITY == 'max':
            periodicity = penn.periodicity.max(logits)
        elif penn.PERIODICITY == 'sum':
            periodicity = penn.periodicity.sum(logits)
        else:
            raise ValueError(
                f'Periodicity method {penn.PERIODICITY} is not defined')

        binsT = bins.permute(*torch.arange(bins.ndim - 1, -1, -1))
        pitchT = pitch.permute(*torch.arange(pitch.ndim - 1, -1, -1))
        periodicityT = periodicity.permute(*torch.arange(periodicity.ndim - 1, -1, -1))

        return binsT, pitchT, periodicityT


def postprocess_with_periodicity(logits, pthreshold, fmin=penn.FMIN, fmax=penn.FMAX):
    """Convert model output to pitch based on periodicity

    Paramters:
        logits - numpy array
            [N, C, B] size array with pitch probabilities BEFORE softmax
        pthreshold - float
            value for thresholding the periodicity

    Returns: 
    """
    with torch.inference_mode():
        bins, pitch, periodicity = postprocess(logits, fmin, fmax)
        # shape is [1, 6, N] for all bins, pitch and periodicity
        pitch[periodicity < pthreshold] = 0

        return bins, pitch, periodicity >= pthreshold

    return None


def remove_empty_timestamps(pitch, voiced):
    """Remove timestamps where the voiced array indicated silence on all channels.

    The function will checks the pitch at a timestamp given by the voiced array,
    will return unique values from the pitch array if voiced and empty if not.
    The timestamps where all the strings are unvoiced are deleted from the array.

    Paramters:
        pitch - numpy array
            [1, C, N] size array returned by postprocess or postprocess_with_periodicity

        voiced - numpy array indicating the voiced parts
            [1, C, N] size

    Returns:
        pitch_array - numpy array
            [1, C, M] size where M - number of voiced time steps

        voiced - numpy array of size [1, C, M]
            array indicating the voiced pitch over all pitch categories.
            Every category has the same number voiced array.
    """
    # tell if voiced at a given timestamp, shape: [N]
    voiced_summed = voiced.sum(dim=1).squeeze().bool()

    return pitch[..., voiced_summed], voiced[..., voiced_summed]


def postprocess_pitch_and_sort(pitch, voiced):
    """Convert an array of pitch to a list of sorted pitches

    The function will checks the pitch at a timestamp given by the voiced array,
    will return unique values from the pitch array if voiced and empty if not.
    The timestamps where all the strings are unvoiced are deleted from the array.

    Paramters:
        pitch - numpy array
            [1, C, N] size array returned by postprocess or postprocess_with_periodicity

        voiced - numpy array indicating the voiced parts
            [1, C, N] size

    Returns:
        pitch_array - numpy array
            [1, C, M] size where M - number of voiced time steps

        voiced - numpy array of size [1, C, M]
            array indicating the voiced pitch over all pitch categories.
            Every category has the same number voiced array.
            
    """
    # tell if voiced at a given timestamp, shape: [N]
    voiced_summed = voiced.sum(dim=1).squeeze().bool()

    breakpoint()

    pitch_voiced = pitch[..., voiced_summed]
    n = pitch_voiced.shape[-1]

    pitch_voiced_chunks = pitch_voiced.chunk(n, dim=-1)

    pitch_list = []
    sort_indx_list = []

    for chunk in pitch_voiced_chunks:
        pitch_sorted, sort_indx = chunk.sort(dim=1, descending=True)
        pitch_list.append(pitch_sorted)
        sort_indx_list.append(sort_indx)

    pitch_array = torch.vstack(pitch_list).permute(2, 1, 0)
    voiced_array = torch.vstack(sort_indx_list).permute(2, 1, 0)

    # return pitch_array, voiced.expand(1, pitch_array.shape[1], -1)
    return pitch_array, torch.gather(voiced[..., voiced_summed], 1, voiced_array)


def preprocess(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    batch_size=None,
    center='half-window'):
    """Convert audio to model input"""
    # Get number of frames
    total_frames = expected_frames(
        audio.shape[-1],
        sample_rate,
        hopsize,
        center)

    # Maybe resample
    if sample_rate != penn.SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    # Maybe pad audio
    hopsize = penn.convert.seconds_to_samples(hopsize)
    if center in ['half-hop', 'zero']:
        if center == 'half-hop':
            padding = int((penn.WINDOW_SIZE - hopsize) / 2)
        else:
            padding = int(penn.WINDOW_SIZE / 2)
        audio = torch.nn.functional.pad(
            audio,
            (padding, padding),
            mode='reflect')

    # Integer hopsizes permit a speedup using torch.unfold
    if isinstance(hopsize, int) or hopsize.is_integer():
        hopsize = int(round(hopsize))
        start_idxs = None

    else:

        # Find start indices
        start_idxs = torch.round(
            torch.tensor([hopsize * i for i in range(total_frames + 1)])
        ).int()

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Size of this batch
        batch = min(total_frames - i, batch_size)

        # Fast implementation for integer hopsizes
        if start_idxs is None:

            # Batch indices
            start = i * hopsize
            end = start + int((batch - 1) * hopsize) + penn.WINDOW_SIZE
            end = min(end, audio.shape[-1])
            batch_audio = audio[:, start:end]

            # Maybe pad to a single frame
            if end - start < penn.WINDOW_SIZE:
                padding = penn.WINDOW_SIZE - (end - start)

                # Handle multiple of hopsize
                remainder = (end - start) % hopsize
                if remainder:
                    padding += end - start - hopsize

                # Pad
                batch_audio = torch.nn.functional.pad(
                    batch_audio,
                    (0, padding))

            # Slice and chunk audio
            frames = torch.nn.functional.unfold(
                batch_audio[:, None, None],
                kernel_size=(1, penn.WINDOW_SIZE),
                stride=(1, hopsize)).permute(2, 0, 1)

        # Slow implementation for floating-point hopsizes
        else:

            # Allocate frames
            frames = torch.zeros(batch, 1, penn.WINDOW_SIZE)

            # Fill each frame with a window starting at the start index
            for j in range(batch):
                start = start_idxs[i + j]
                end = min(start + penn.WINDOW_SIZE, audio.shape[-1])
                frames[j, :, : end - start] = audio[:, start:end]

        yield frames


###############################################################################
# Inference acceleration
###############################################################################


def inference_collate(batch):
    frames, lengths, files = zip(*batch)
    return (
        torch.cat(frames, dim=0),
        torch.tensor(lengths, dtype=torch.long),
        files)


def inference_loader(
    files,
    hopsize=penn.HOPSIZE_SECONDS,
    batch_size=None,
    center='half-window',
    num_workers=penn.NUM_WORKERS // 2
):
    dataset = InferenceDataset(files, hopsize, batch_size, center)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=InferenceSampler(dataset),
        num_workers=num_workers,
        collate_fn=inference_collate)


def save_worker(prefix, pitch, periodicity, interp_unvoiced_at=None):
    """Save pitch and periodicity to disk"""
    # Maybe interpolate unvoiced regions
    if interp_unvoiced_at is not None:
        pitch = penn.voicing.interpolate(
            pitch,
            periodicity,
            interp_unvoiced_at)

    # Save
    Path(prefix).parent.mkdir(exist_ok=True, parents=True)
    torch.save(pitch, f'{prefix}-pitch.pt')
    torch.save(periodicity, f'{prefix}-periodicity.pt')

    # Clean-up
    del pitch
    del periodicity


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        files,
        hopsize=penn.HOPSIZE_SECONDS,
        batch_size=None,
        center='half-window'):
        self.files = files
        self.batch_size = batch_size
        self.lengths = []
        for file in files:
            info = torchaudio.info(file)
            self.lengths.append(
                expected_frames(
                    info.num_frames,
                    info.sample_rate,
                    hopsize,
                    center))
        self.preprocess_fn = functools.partial(
            preprocess,
            hopsize=hopsize,
            batch_size=batch_size,
            center=center)

    def __getitem__(self, index):
        frames = torch.cat(
            [
                frame for frame in
                self.preprocess_fn(*torchaudio.load(self.files[index]))
            ],
            dim=0)
        return frames, self.lengths[index], self.files[index]

    def __len__(self):
        return len(self.files)


class InferenceSampler(torch.utils.data.Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    @functools.cached_property
    def batch(self):
        count = 0
        batch, batches = [], []
        for i, length in enumerate(self.dataset.lengths):
            batch.append(i)
            if self.dataset.batch_size is None:
                batches.append(batch)
                batch = []
            else:
                count += length
                if count >= self.dataset.batch_size:
                    batches.append(batch)
                    batch = []
                    count = 0
        if batch:
            batches.append(batch)
        return batches


###############################################################################
# Utilities
###############################################################################


def cents(a, b):
    """Compute pitch difference in cents"""
    if type(a) is torch.Tensor or type(a) is torch.masked.maskedtensor.core.MaskedTensor:
        return penn.OCTAVE * torch.log2(a / b)
    elif type(a) is np.ndarray or type(b) is np.ndarray:
        return penn.OCTAVE * np.log2(a / b)
    else:
        raise TypeError(f"Type {type(a)} not supported")


def expected_frames(samples, sample_rate, hopsize, center):
    """Compute expected number of output frames"""
    # Calculate expected number of frames
    hopsize_resampled = penn.convert.seconds_to_samples(
        hopsize,
        sample_rate)
    if center == 'half-window':
        window_size_resampled = \
            penn.WINDOW_SIZE / penn.SAMPLE_RATE * sample_rate
        samples = samples - (window_size_resampled - hopsize_resampled)
    elif center == 'half-hop':
        samples = samples
    elif center == 'zero':
        samples = samples + hopsize_resampled
    else:
        raise ValueError(f'Unknown center sample {center}')
    return max(1, int(samples / hopsize_resampled))


@contextlib.contextmanager
def inference_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.inference_mode():

        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield
        else:
            yield

    # Prepare model for training
    model.train()


def interpolate(x, xp, fp):
    """1D linear interpolation for monotonically increasing sample points"""
    # Handle edge cases
    if xp.shape[-1] == 0:
        return x
    if xp.shape[-1] == 1:
        return torch.full(
            x.shape,
            fp.squeeze(),
            device=fp.device,
            dtype=fp.dtype)

    # Get slope and intercept using right-side first-differences
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # Get indices to sample slope and intercept
    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)
    line_idx = torch.linspace(
        0,
        indicies.shape[0],
        1,
        device=indicies.device).to(torch.long).expand(indicies.shape)

    # Interpolate
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]


def normalize(frames):
    """Normalize audio frames to have mean zero and std dev one"""
    # Mean-center
    frames -= frames.mean(dim=2, keepdim=True)

    # Scale
    frames /= torch.max(
        torch.tensor(1e-10, device=frames.device),
        frames.std(dim=2, keepdim=True))

    return frames


def resample(audio, sample_rate, target_rate=penn.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)


def peak_notes_v1(logits):
    """Find the notes with highest values / priority

    Inspired by: https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404"""

    # mask all the point that are NOT larger than their direct neighbours
    peak_mask_core = logits[..., :-2, :] < logits[..., 1:-1, :]
    peak_mask_core = peak_mask_core & (logits[..., 2:, :] < logits[..., 1:-1, :])

    # pad the 2nd to last dimention with zeros
    peak_mask = torch.nn.functional.pad(peak_mask_core, (0, 0, 1, 1))
    assert peak_mask.shape == logits.shape

    breakpoint()


def peak_notes_v2(logits, width=101, threshold_type=None, peak_window_size=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Find the notes with highest values / priority

    Inspired by: https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404"""
    padding = width // 2
    width = padding * 2 + 1

    logits_modified = logits.permute(0, 2, 1)
    # deduct the average from the logits
    # logits_modified -= logits_modified.mean(dim=-1).expand(logits.shape[0], penn.PITCH_BINS)[:, None, :]

    # set below average to inf
    if threshold_type == 'average':
        average = logits_modified.mean(dim=-1).expand(logits.shape[0], penn.PITCH_BINS)[:, None, :]
        logits_modified[logits_modified <= average] = -float('inf')
    elif threshold_type == 'sorted' or threshold_type is None:
        percentile = 0.8
        percentile_indx = int(penn.PITCH_BINS * percentile)

        logits_sorted, _ = torch.sort(logits_modified, dim=-1)
        threshold = logits_sorted[..., percentile_indx].expand(logits.shape[0], penn.PITCH_BINS)[:, None, :]
        logits_modified[logits_modified <= threshold] = -float('inf')

    # find the window maxima on data of shape [N, 1, B], f. eg. [N, 1, 1440]
    window_maxima = torch.nn.functional.max_pool1d_with_indices(
            logits_modified, 
            kernel_size=width,
            stride=width // 2,
            padding=padding)[1]

    # split into chunks of shape [1, 1, 1440]
    maxima_chunks = window_maxima.chunk(window_maxima.shape[0], dim=0)
    logits_chunks = logits.chunk(logits_modified.shape[0], dim=0)
    logits_modified_chunks = logits_modified.chunk(logits_modified.shape[0], dim=0)

    nice_peaks_list = []
    peak_list = []

    for mchunk, lchunk, lmchunk in zip(maxima_chunks, logits_chunks, logits_modified_chunks):
        # get unique values from the chunk of indexes
        candidates = mchunk.unique()
        flat_mchunk = mchunk.squeeze()

        # # get the boolean table for where the peaks are
        # nice_peaks = candidates[flat_mchunk[candidates] == candidates]
        #
        # flat_lmchunk = lmchunk.squeeze()
        # nice_peaks = nice_peaks[flat_lmchunk[nice_peaks].isfinite()]
        #
        # nice_peaks_list.append(nice_peaks)

        nice_peaks = candidates[lmchunk.squeeze()[candidates].isfinite()]

        lchunk_peak_list = []

        for peak in nice_peaks:
            minidx = 0 if peak - peak_window_size <= 0 else peak - peak_window_size
            maxidx = penn.PITCH_BINS if peak + peak_window_size >- penn.PITCH_BINS else peak + peak_window_size

            lchunk_with_center = lchunk.clone()

            # Remove frequencies outside of allowable range
            # below works with shapes [128, 1440, 1] and [128, 6, 1440, 1]
            lchunk_with_center[..., :minidx, :] = -float('inf')
            lchunk_with_center[..., maxidx:, :] = -float('inf')

            bins, pitch = penn.decode.local_expected_value(
                    lchunk_with_center, window=peak_window_size)

            lchunk_peak_list.append(bins)

        peak_list.append(lchunk_peak_list)

    return peak_list


def peak_notes_to_peak_array(peak_list):
    """Convert the list of top peaks to an array with a fixed width"""
    max_len = len(max(peak_list, key=len))

    # create a zeros array of size N x M, 
    # where N is the number of timesteps, 
    # M the maximum number of pitches in a timestep
    peak_array = np.zeros((len(peak_list), max_len))
    peak_array -= 1

    for idx, peak_slice_list in enumerate(peak_list):
        np_slice_list = [pk.numpy() if type(pk) is torch.Tensor else pk for pk in peak_slice_list]
        np_slice_array = np.unique(np.array(np_slice_list))
        peak_array[idx, :np_slice_array.size] = np_slice_array.flatten()

    peak_array_masked = np.ma.MaskedArray(peak_array, peak_array == -1)
    return peak_array


def logits_dict_to_device(logits_dict, device):
    new_logits_dict = {}
    for key, value in logits_dict.items():
        new_logits_dict[key] = value.to(device)

    return new_logits_dict


def logits_dict_detach(logits_dict):
    new_logits_dict = {}
    for key, value in logits_dict.items():
        new_logits_dict[key] = value.detach()

    return new_logits_dict
