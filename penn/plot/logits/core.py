import numpy as np
import torch

import penn

import plotly.express as px
import plotly.graph_objects as go


###############################################################################
# Create figure with plotly (compatible with wandb)
###############################################################################


def logits_plotly(
        logits,
        ground_truth_logits=None,
        ):

    # convert to numpy
    logits = logits.cpu().numpy()
    logits_and_gnd = logits

    if ground_truth_logits is not None:
        ground_truth_logits = ground_truth_logits.cpu().numpy()
        logits_and_gnd = np.concatenate((logits, ground_truth_logits), axis=0)

    lh = logits.shape[0]
    lw = logits.shape[0]


    fig = px.imshow(
            logits_and_gnd, 
            color_continuous_scale=px.colors.sequential.Cividis_r,
            height=lh,
            width= lh if lw <= lh else lw)

    return fig


###############################################################################
# Create figure
###############################################################################


def process_logits(logits: torch.Tensor):
    # Convert to distribution
    # NOTE - We use softmax even if the loss is BCE for more comparable
    #        visualization. Otherwise, the variance of models trained with
    #        BCE looks erroneously lower.
    distributions = torch.nn.functional.softmax(logits, dim=1)

    # Take the log again for display
    distributions = torch.log(distributions)
    distributions[torch.isinf(distributions)] = \
        distributions[~torch.isinf(distributions)].min()

    # Prepare for plotting
    distributions = distributions.cpu().squeeze(2).T
    new_distributions = distributions

    figsize=(18, 2)

    # Prepare the ptich posteriorgram in case it's multipitch
    if len(distributions.shape) == 4:
        distr_chunk = torch.chunk(distributions, distributions.shape[-2], -2)
        distr_chunk = [distr.squeeze(dim=-2).squeeze(dim=0) for distr in distr_chunk]
        new_distributions = torch.vstack(distr_chunk)
        figsize = (18, 10)

    return new_distributions, figsize


def logits_matplotlib(logits, bins=None, voiced=None):
    import matplotlib
    import matplotlib.pyplot as plt

    distributions, figsize = process_logits(logits)

    predicted_bins, pitch, periodicity = penn.postprocess(logits)

    # Change font size
    matplotlib.rcParams.update({'font.size': 5})

    # Setup figure
    figure, axis = plt.subplots(figsize=figsize)

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    xticks = torch.arange(0, len(distributions), int(penn.SAMPLE_RATE / penn.HOPSIZE))
    xlabels = torch.round(xticks * (penn.HOPSIZE / penn.SAMPLE_RATE)).int()
    axis.get_xaxis().set_ticks(xticks.tolist(), xlabels.tolist())

    yticks = torch.linspace(0, penn.PITCH_BINS - 1, 5)
    ylabels = penn.convert.bins_to_frequency(yticks)
    ylabels_chunk = [ylabels for _ in range(penn.PITCH_CATS)]
    ylabels = torch.cat(ylabels_chunk)
    yticks = torch.linspace(0, 
                            penn.PITCH_BINS * penn.PITCH_CATS - 1,
                            5 * penn.PITCH_CATS)

    ylabels = ylabels.round().int().tolist()
    axis.get_yaxis().set_ticks(yticks, ylabels)
    axis.set_xlabel('Time (seconds)')
    axis.set_ylabel('Frequency (Hz)')

    if bins is not None and voiced is not None:
        nbins = bins.detach().cpu().numpy()
        nvoiced = voiced.detach().cpu().numpy()

        npredicted_bins = predicted_bins.detach().cpu().numpy()

        nbins = nbins.squeeze().T
        npredicted_bins = npredicted_bins.squeeze().T
        nvoiced = nvoiced.squeeze().T

        offset = np.arange(0, penn.PITCH_CATS)*penn.PITCH_BINS
        nbins += offset
        npredicted_bins += offset

        nbins_masked = np.ma.MaskedArray(nbins, np.logical_not(nvoiced))
        npredicted_bins_masked = np.ma.MaskedArray(npredicted_bins, np.logical_not(nvoiced))

        axis.plot(nbins_masked, 'r--', linewidth=0.5)
        axis.plot(npredicted_bins_masked, 'b:', linewidth=0.5)

    # Plot pitch posteriorgram
    # if len(distributions.shape) == 4:
    #     axis.imshow(new_distributions, extent=[0,100,0,1], aspect=80, origin='lower')
    # else:
    #     axis.imshow(new_distributions, aspect='auto', origin='lower')
    axis.imshow(distributions, aspect='auto', origin='lower')

    return figure


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

    return logits_matplotlib(logits)



###############################################################################
# Plotting the logits from a testset
###############################################################################


def from_model_and_testset(model, loader, gpu=None):
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Prepare model for inference
    with penn.inference_context(model):

        # Iterate over test set
        audio, bins, _, voiced, stem = next(loader)

        # Accumulate logits
        logits = []

        # Preprocess audio
        batch_size = \
            None if gpu is None else penn.EVALUATION_BATCH_SIZE
        for frames in penn.preprocess(
            audio[0],
            penn.SAMPLE_RATE,
            batch_size=batch_size,
            center='half-hop'
        ):

            # Copy to device
            frames = frames.to(device)

            # Infer
            batch_logits = model(frames)

            # Accumulate logits
            logits.append(batch_logits)

        logits = torch.cat(logits)

        return logits_matplotlib(logits, bins, voiced)


def from_testset(checkpoint=None, gpu=None):
    # Initialize model
    model = penn.Model()

    # Maybe download from HuggingFace
    if checkpoint is None:
        return 

    checkpoint = torch.load(checkpoint, map_location='cpu')

    # Load from disk
    model.load_state_dict(checkpoint['model'])

    loader = penn.data.loader(penn.EVALUATION_DATASETS, 'test')
    loader = iter(loader)

    return from_model_and_testset(model, loader)


def from_file(audio_file, checkpoint=None, gpu=None):
    """Plot logits and optional pitch"""
    # Load audio
    audio = penn.load.audio(audio_file)

    # Plot
    return from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)


def from_file_to_file(audio_file=None, output_file=None, checkpoint=None, gpu=None):
    """Plot pitch and periodicity and save to disk"""
    # Plot
    if audio_file is not None:
        figure = from_file(audio_file, checkpoint, gpu)
    else:
        figure = from_testset(checkpoint, gpu)

    # Save to disk
    if output_file is not None:
        figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=900)
