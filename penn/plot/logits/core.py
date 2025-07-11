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

    if penn.LOSS_MULTI_HOT:
        distributions = torch.nn.functional.sigmoid(logits)
    else:
        distributions = torch.nn.functional.softmax(logits, dim=1)

    # Take the log again for display
    distributions = torch.log(distributions)
    distributions[torch.isinf(distributions)] = \
        distributions[~torch.isinf(distributions)].min()

    # Prepare for plotting
    distributions = distributions.cpu().squeeze(2).T
    new_distributions = distributions

    figsize=(18, 10)

    # Prepare the ptich posteriorgram in case it's multipitch
    if len(distributions.shape) == 4:
        distr_chunk = torch.chunk(distributions, distributions.shape[-2], -2)
        distr_chunk = [distr.squeeze(dim=-2).squeeze(dim=0) for distr in distr_chunk]
        new_distributions = torch.vstack(distr_chunk)

    return new_distributions, figsize


def logits_matplotlib(logits, true_pitch=None, bins=None, voiced=None, stem=None):
    import matplotlib
    import matplotlib.pyplot as plt

    chunk_start = int(4.8 * penn.SAMPLE_RATE // penn.HOPSIZE)
    chunk_end = int(5.4 * penn.SAMPLE_RATE // penn.HOPSIZE)

    logits = logits[chunk_start:chunk_end, ...]
    true_pitch = true_pitch[..., chunk_start:chunk_end]
    bins = bins[..., chunk_start:chunk_end]
    voiced = voiced[..., chunk_start:chunk_end]

    distributions, figsize = process_logits(logits)

    peak_array = None

    if penn.LOSS_MULTI_HOT:
        peak_logits = penn.core.peak_notes_v2(logits)
        peak_array = penn.core.peak_notes_to_peak_array(peak_logits)
    else:
        predicted_bins, pitch, periodicity = penn.postprocess(logits)

        # pitch = pitch[..., chunk_start:chunk_end]
        # periodicity = periodicity[..., chunk_start:chunk_end]

        # predicted_bins, pitch_predicted, voiced_predicted = penn.postprocess_with_periodicity(logits, 0.5)
        # penn.core.postprocess_pitch_and_sort(pitch_predicted, voiced)

    # distributions = distributions[..., chunk_start:chunk_end]

    # Change font size
    matplotlib.rcParams.update({'font.size': 10})

    # Setup figure
    figure, axis = plt.subplots(figsize=figsize)

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    xticks = torch.arange(0, distributions.shape[-1], int(penn.SAMPLE_RATE / penn.HOPSIZE))
    xlabels = torch.round(xticks * (penn.HOPSIZE / penn.SAMPLE_RATE)).int()
    axis.get_xaxis().set_ticks(xticks.tolist(), xlabels.tolist())

    yticks = torch.linspace(0, penn.PITCH_BINS - 1, 5)
    ylabels = penn.convert.bins_to_frequency(yticks)
    # ylabels_chunk = [ylabels for _ in range(penn.PITCH_CATS)]
    # ylabels = torch.cat(ylabels_chunk)
    # yticks = torch.linspace(0, 
    #                         penn.PITCH_BINS * penn.PITCH_CATS - 1,
    #                         5 * penn.PITCH_CATS)

    # ylabels = ylabels.round().int().tolist()
    # axis.get_yaxis().set_ticks(yticks, ylabels)
    axis.set_xlabel('Time (seconds)')
    axis.set_ylabel('Frequency (Hz)')
    axis.set_title(f"track: {stem}")

    if bins is not None and voiced is not None:
        nbins = bins.clone().detach().cpu().numpy()
        nvoiced = voiced.clone().detach().cpu().numpy()

        nbins = nbins.squeeze().T
        nvoiced = nvoiced.squeeze().T

        offset = np.arange(0, penn.PITCH_CATS)*penn.PITCH_BINS * int(not penn.LOSS_MULTI_HOT)

        nbins += offset

        nbins_masked = np.ma.MaskedArray(nbins, np.logical_not(nvoiced))

        for nbins_row in range(penn.PITCH_CATS):
            axis.plot(nbins_masked[:, nbins_row], 'r--', marker='o', linewidth=.5, markersize=1.5, zorder=1)

        if predicted_bins is not None:
            # metrics = penn.evaluate.MutliPitchMetrics(thresholds=[0.5])
            # metrics = penn.evaluate.MutliPitchMetrics()
            # metrics.reset()
            # metrics.update(
            #         torch.tensor(pitch), 
            #         torch.tensor(periodicity),
            #         torch.tensor(true_pitch),
            #         voiced.clone().detach().cpu())
            # metrics_dict = metrics()

            npredicted_bins = predicted_bins.detach().cpu().numpy()
            npredicted_bins = npredicted_bins.squeeze().T

            periodicity_for_mask = periodicity.detach().cpu().numpy()
            periodicity_for_mask = periodicity_for_mask.squeeze().T
            periodicity_mask = periodicity_for_mask >= 0.05

            # nvoiced_predicted = voiced_predicted.detach().cpu().numpy()

            npredicted_bins += offset
            # npredicted_bins_masked = np.ma.MaskedArray(npredicted_bins, np.logical_not(nvoiced_predicted))
            npredicted_bins_masked = np.ma.MaskedArray(npredicted_bins, np.logical_not(periodicity_mask))

            axis.plot(npredicted_bins_masked, 'b:', linewidth=2)

        if periodicity is not None:
            periodicity_for_plot = periodicity.detach().cpu().numpy()
            periodicity_for_plot = periodicity_for_plot.squeeze().T
            periodicity_mask = periodicity_for_plot >= 0.05

            offset = np.arange(0, penn.PITCH_CATS) * int(not penn.LOSS_MULTI_HOT)
            periodicity_for_plot += offset

            # mask periodicity under the threshold 
            periodicity_masked = np.ma.MaskedArray(periodicity_for_plot, np.logical_not(periodicity_mask))

            twin = axis.twinx()
            twin.set_ylim(ymin=0, ymax=penn.PITCH_CATS)

            twin.plot(periodicity_for_plot, 'g:', linewidth=2)
            twin.plot(periodicity_masked, 'm:', linewidth=2)

    if peak_array is not None:
        slices = peak_array.shape[-1]

        x = np.arange(peak_array.shape[0])
        for idx in range(slices):
            # axis.plot(peak_array[:, idx], 'b', marker='o', linewidth=.5)
            axis.scatter(x, peak_array[:, idx], s=4, marker='x', zorder=2)

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


def from_model_and_testset(model, loader, gpu=None, iters=0):
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Prepare model for inference
    with penn.inference_context(model):

        for iter in range(iters):
            next(loader)

        # Iterate over test set
        audio, bins, pitch, voiced, stem = next(loader)

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

        return logits_matplotlib(logits, pitch, bins, voiced, stem)


def from_testset(checkpoint=None, gpu=None, iters=0):
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

    return from_model_and_testset(model, loader, iters=iters)


def from_file(audio_file, checkpoint=None, gpu=None):
    """Plot logits and optional pitch"""
    # Load audio
    audio = penn.load.audio(audio_file)

    # Plot
    return from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)


def from_file_to_file(audio_file=None, output_file=None, checkpoint=None, gpu=None, iters=0):
    """Plot pitch and periodicity and save to disk"""
    # Plot
    if audio_file is not None:
        figure = from_file(audio_file, checkpoint, gpu)
    else:
        figure = from_testset(checkpoint, gpu, iters=iters)

    breakpoint()

    # Save to disk
    if output_file is not None:
        figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=900)
