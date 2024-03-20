import functools

import torch
import torchutil

import penn
import wandb
from socket import gethostname


###############################################################################
# Training
###############################################################################


@torchutil.notify('train')
def train(datasets, directory, gpu=None, use_wand=False):
    """Train a model"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    model = penn.model.Model().to(device)

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(penn.RANDOM_SEED)
    train_loader = penn.data.loader(datasets, 'train')
    train_loader_for_plots = penn.data.loader(datasets, 'train')
    valid_loader = penn.data.loader(datasets, 'valid')
    test_loader = penn.data.loader(datasets, 'test')
    test_loader = iter(test_loader)

    ####################
    # Create optimizer #
    ####################

    if penn.WEIGHT_DECAY is not None:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=penn.LEARNING_RATE,
                                     weight_decay=penn.WEIGHT_DECAY)
        print(f"WEIGHT_DECAY: {penn.WEIGHT_DECAY}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=penn.LEARNING_RATE)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = torchutil.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, state = torchutil.checkpoint.load(
            path,
            model,
            optimizer)
        step, epoch = state['step'], state['epoch']

    else:

        # Train from scratch
        step, epoch = 0, 0

    ##############################
    # Maybe setup early stopping #
    ##############################

    if penn.EARLY_STOPPING:
        counter = penn.EARLY_STOPPING_STEPS
        best_accuracy = 0.
        stop = False

    ####################
    # Weights & Biases #
    ####################

    log_wandb = None

    if use_wand:
        wandb.login()
        log_wandb = wandb.init(
            # dir=tempfile.gettempdir(),
            # Set the project where this run will be logged
            project="PENNbyAntoni",
            name=f"{penn.CONFIG}",

            # Track hyperparameters and run metadata
            config={
                "learning_rate": penn.LEARNING_RATE,
                "epochs": penn.STEPS,
                "loss" : penn.LOSS,
                "pitch_bins" : penn.PITCH_BINS,
                "config" : penn.CONFIG,
                "datasets" : penn.DATASETS,
                "hostname" : gethostname(),
                "weight_decay" : penn.WEIGHT_DECAY,
                "dropout" : penn.DROPOUT,
            })


    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, penn.STEPS),
        f'Training {penn.CONFIG}',
        step,
        penn.STEPS)
    while step < penn.STEPS and (not penn.EARLY_STOPPING or not stop):
        print(f"epoch: {epoch}")

        # Seed sampler
        train_loader.sampler.set_epoch(epoch)
        train_loss_list = []
        train_accuracy_list = []
        valid_accuracy_list = []

        for batch in train_loader:

            # Unpack batch
            audio, bins, *_ = batch

            with torch.autocast(device.type):

                # Forward pass
                logits = model(audio.to(device))

                # Compute losses
                losses = loss(logits, bins.to(device))

                train_loss_list.append(losses.item())

            ##################
            # Optimize model #
            ##################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ##############
            # Evaluation #
            ##############

            # Save checkpoint
            if step and step % penn.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch)

                fig = penn.plot.logits.from_model_and_testset(
                        model=model,
                        loader=test_loader,
                        gpu=gpu)

                fig2 = penn.plot.logits.from_model_and_testset(
                        model=model,
                        loader=train_loader_for_plots,
                        gpu=gpu)

                if use_wand:
                    log_wandb.log({"test_logits": wandb.Image(fig)})
                    log_wandb.log({"train_logits": wandb.Image(fig2)})

            # Evaluate
            if step % penn.LOG_INTERVAL == 0:
                evaluate_fn = functools.partial(
                    evaluate,
                    directory,
                    step,
                    model,
                    gpu)

                train_accuracy = evaluate_fn('train', train_loader, log_wandb)
                train_accuracy_list.append(train_accuracy)
                valid_accuracy = evaluate_fn('valid', valid_loader, log_wandb)
                valid_accuracy_list.append(valid_accuracy) 

                if use_wand:
                    metrics = {}
                    metrics[f"train_loss_avg_{penn.LOG_INTERVAL}"] = sum(train_loss_list[-penn.LOG_INTERVAL:]) \
                                / len(train_loss_list[-penn.LOG_INTERVAL:])

                    if len(train_accuracy_list) > 0:
                        metrics[f"train_accuracy_{penn.LOG_INTERVAL}"] = train_accuracy

                    if len(valid_accuracy_list) > 0:
                        metrics[f"valid_accuracy_{penn.LOG_INTERVAL}"] = valid_accuracy

                    log_wandb.log(data=metrics,
                                  step=step)

                # Maybe stop training
                if penn.EARLY_STOPPING:
                    counter -= 1

                    # Update best validation loss
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        counter = penn.EARLY_STOPPING_STEPS

                    # Stop training
                    elif counter == 0:
                        stop = True

            # Update training step count
            if step >= penn.STEPS or (penn.EARLY_STOPPING and stop):
                break
            step += 1

            # Update progress bar
            progress.update()

        if use_wand:
            metrics = {}
            metrics["train_loss"] = sum(train_loss_list) / len(train_loss_list)

            if len(train_accuracy_list) > 0:
                metrics["train_accuracy_per_epoch"] = sum(train_accuracy_list) / len(train_accuracy_list)

            if len(valid_accuracy_list) > 0:
                metrics["valid_accuracy_per_epoch"] = sum(valid_accuracy_list) / len(valid_accuracy_list)

            log_wandb.log(data=metrics,
                          step=step)
        # Update epoch
        epoch += 1

    # Close progress bar
    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        directory / f'{step:08d}.pt',
        model,
        optimizer,
        step=step,
        epoch=epoch)


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader, log_wandb):
    """Perform model evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup evaluation metrics
    metrics = penn.evaluate.Metrics()

    # Prepare model for inference
    with penn.inference_context(model):

        # Unpack batch
        for i, (audio, bins, pitch, voiced, *_) in enumerate(loader):

            # Forward pass
            logits = model(audio.to(device))

            if penn.MIDI60:
                bins = bins[:, :penn.PITCH_CATS, :]
                voiced = voiced[:, :penn.PITCH_CATS, :]
                pitch = pitch[:, :penn.PITCH_CATS, :]

            if len(logits.shape) == 4 or penn.LOSS_MULTI_HOT:
                binsT = bins.permute(*torch.arange(bins.ndim - 1, -1, -1))
                pitchT = pitch.permute(*torch.arange(pitch.ndim - 1, -1, -1))
                voicedT = voiced.permute(*torch.arange(voiced.ndim - 1, -1, -1))

                # Update metrics
                metrics.update(
                    logits.to(device),
                    binsT.to(device),
                    pitchT.to(device),
                    voicedT.to(device))

            else:
                # Update metrics
                metrics.update(
                    logits.to(device),
                    bins.T.to(device),
                    pitch.T.to(device),
                    voiced.T.to(device))

            # Stop when we exceed some number of batches
            if i + 1 == penn.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    if log_wandb is not None:
        log_wandb.log(data=scalars,
                      step=step)

    # Write to tensorboard
    torchutil.tensorboard.update(directory, step, scalars=scalars)

    return scalars[f'loss/{condition}']


###############################################################################
# Loss function
###############################################################################


def loss(logits, bins):
    """Compute loss function"""
    # Reshape inputs
    if len(logits.shape) == 4:
        logits = logits.permute(0, 1, 3, 2)

        if penn.LOSS_CHUNKED:
            logits = logits.squeeze(-2)
            logits_chunks = logits.chunk(penn.PITCH_CATS, dim=1)
            logits_chunks = [chunk.squeeze(1) for chunk in logits_chunks]
            logits = torch.stack(logits_chunks, dim=0)
        elif penn.MIDI60:
            # fix this
            logits = logits.squeeze(-2)
            logits_chunks = logits.chunk(penn.PITCH_CATS, dim=1)
            logits_chunks = [chunk.squeeze(dim=1) for chunk in logits_chunks]

            # logits_chunks = logits.chunk(logits.shape[0], dim=0)
            # logits_chunks = [chunk.flatten() for chunk in logits_chunks]
            logits = torch.vstack(logits_chunks)
        else:
            logits = logits.reshape(-1, penn.PITCH_BINS)
    else:
        logits = logits.permute(0, 2, 1).reshape(-1, penn.PITCH_BINS)

    def get_bins(bins):

        # Maybe blur target
        if penn.GAUSSIAN_BLUR:
            bins = bins.flatten()

            # Cache cents values to evaluate distributions at
            if not hasattr(loss, 'cents'):
                loss.cents = penn.convert.bins_to_cents(
                    torch.arange(penn.PITCH_BINS))[:, None]

            # Ensure values are on correct device (no-op if devices are the same)
            loss.cents = loss.cents.to(bins.device)

            # Create normal distributions
            distributions = torch.distributions.Normal(
                penn.convert.bins_to_cents(bins),
                25)

            # Sample normal distributions
            bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

            # Normalize
            bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)

        else:
            if penn.MIDI60:
                # bins_chunk = bins.chunk(bins.shape[0], dim=0)
                # bins_chunk_1hot = [
                #         torch.nn.functional.one_hot(chunk.flatten(), penn.PITCH_BINS).float().flatten()
                #         for chunk in bins_chunk]
                # bins = torch.stack(bins_chunk_1hot, dim=0)

                bins = bins.flatten()

                # One-hot encoding
                bins = torch.nn.functional.one_hot(bins, penn.PITCH_BINS).float()
            else:
                bins = bins.flatten()

                # One-hot encoding
                bins = torch.nn.functional.one_hot(bins, penn.PITCH_BINS).float()

        return bins

    if penn.LOSS_CHUNKED:
        bins_chunks = bins.chunk(penn.PITCH_CATS, dim=1)
        bins_chunks = [get_bins(chunk) for chunk in bins_chunks]
        bins = torch.stack(bins_chunks)

    elif penn.LOSS_MULTI_HOT:
        bins_chunks = bins.chunk(penn.PITCH_CATS, dim=1)
        bins_chunks = [get_bins(chunk) for chunk in bins_chunks]
        bins = torch.stack(bins_chunks)

        # combine all one-hot vectors into a single multi-hot vector
        bins = torch.sum(bins, dim=0)

        # it may happen that two strings are playing the same note, in that case interpret it as a single note
        bins[bins > 1] = 1

    elif penn.MIDI60:
        midi_array = penn.convert.midi_to_organ_key(bins[:, :penn.PITCH_CATS, :])

        midi_bins = get_bins(midi_array.long())

        if penn.MODEL == 'ppnmidi60x2':
            offset_array = bins[:, penn.PITCH_CATS:, :]

            # Set unvoiced bins to random values
            offset_array = torch.where(
                offset_array == 0,
                torch.randint(
                    -penn.MIDI_OFFSET_LIMIT,
                    penn.MIDI_OFFSET_LIMIT,
                    offset_array.shape,
                    dtype=torch.long)
                .to(offset_array.device),
                offset_array)

            # this assumes that the deviation is in range [-100, +100] cents
            resolution = penn.MIDI_OFFSET_LIMIT*2 / 60
            offset_array += penn.MIDI_OFFSET_LIMIT
            offset_array[offset_array < 0] = 0
            offset_array[offset_array >= penn.MIDI_OFFSET_LIMIT] = penn.MIDI_OFFSET_LIMIT - 1
            offset_array = torch.round(offset_array / resolution)
            offset_bins = get_bins(offset_array.long())

            bins = torch.vstack([midi_bins, offset_bins])
        else:
            bins = midi_bins
        
    else: 
        bins = get_bins(bins)

    if penn.LOSS == 'binary_cross_entropy':

        # Compute binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            bins)

    elif penn.LOSS == 'categorical_cross_entropy':

        # Compute categorical cross-entropy loss
        return torch.nn.functional.cross_entropy(logits, bins)

    else:

        raise ValueError(f'Loss {penn.LOSS} is not implemented')
