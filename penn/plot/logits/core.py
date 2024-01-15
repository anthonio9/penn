import torch

import penn


###############################################################################
# Create figure
###############################################################################


def from_audio(
    audio,
    sample_rate,
    checkpoint=None,
    gpu=None):
    """Plot logits with pitch overlay"""
    import matplotlib
    import matplotlib.pyplot as plt

    logits = []

    # Change font size
    matplotlib.rcParams.update({'font.size': 5})

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

    # Setup figure
    figure, axis = plt.subplots(figsize=figsize)

    # Make pretty
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    xticks = torch.arange(0, len(logits), int(penn.SAMPLE_RATE / penn.HOPSIZE))
    xlabels = torch.round(xticks * (penn.HOPSIZE / penn.SAMPLE_RATE)).int()
    axis.get_xaxis().set_ticks(xticks.tolist(), xlabels.tolist())

    yticks = torch.linspace(0, penn.PITCH_BINS - 1, 5)
    ylabels = penn.convert.bins_to_frequency(yticks)

    if len(distributions.shape) == 4:
        no_poly_cats = distributions.shape[-2]
        ylabels = penn.convert.bins_to_frequency(yticks)
        ylabels_chunk = [ylabels for _ in range(no_poly_cats)]
        ylabels = torch.cat(ylabels_chunk)
        yticks = torch.linspace(0, 
                                penn.PITCH_BINS * no_poly_cats - 1,
                                5 * no_poly_cats)

    ylabels = ylabels.round().int().tolist()
    axis.get_yaxis().set_ticks(yticks, ylabels)
    axis.set_xlabel('Time (seconds)')
    axis.set_ylabel('Frequency (Hz)')

    # Plot pitch posteriorgram
    # if len(distributions.shape) == 4:
    #     axis.imshow(new_distributions, extent=[0,100,0,1], aspect=80, origin='lower')
    # else:
    #     axis.imshow(new_distributions, aspect='auto', origin='lower')
    axis.imshow(new_distributions, aspect='auto', origin='lower')

    return figure


def from_file(audio_file, checkpoint=None, gpu=None):
    """Plot logits and optional pitch"""
    # Load audio
    audio = penn.load.audio(audio_file)

    # Plot
    return from_audio(audio, penn.SAMPLE_RATE, checkpoint, gpu)


def from_file_to_file(audio_file, output_file, checkpoint=None, gpu=None):
    """Plot pitch and periodicity and save to disk"""
    # Plot
    figure = from_file(audio_file, checkpoint, gpu)

    # Save to disk
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=900)
