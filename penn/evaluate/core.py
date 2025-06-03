import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchutil

import penn


###############################################################################
# Evaluate
###############################################################################


@torchutil.notify('evaluate')
def datasets(
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=None,
    gpu=None,
    benchmark=False,
    evaluate_periodicity=False,
    iterations=None,
    silence=False,
    inference_only=False):
    """Perform evaluation"""
    # Make output directory
    directory = penn.EVAL_DIR / penn.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Evaluate pitch estimation quality and save logits
    pitch_quality(directory,
                  datasets,
                  checkpoint,
                  gpu,
                  iterations=iterations,
                  silence=silence,
                  inference_only=inference_only)

    # Perform benchmarking on CPU
    if benchmark:
        benchmark_results = {'cpu': benchmark(datasets, checkpoint)}

        # PYIN and DIO do not have GPU support
        if penn.METHOD not in ['dio', 'pyin']:
            benchmark_results ['gpu'] = benchmark(datasets, checkpoint, gpu)

        # Write benchmarking information
        with open(penn.EVAL_DIR / penn.CONFIG / 'time.json', 'w') as file:
            json.dump(benchmark_results, file, indent=4)

    if not evaluate_periodicity:
        return

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Get periodicity methods
        if penn.METHOD == 'dio':
            periodicity_fns = {}
        elif penn.METHOD == 'pyin':
            periodicity_fns = {'sum': penn.periodicity.sum}
        else:
            periodicity_fns = {
                'entropy': penn.periodicity.entropy,
                'max': penn.periodicity.max}

        # Evaluate periodicity
        periodicity_results = {}
        for key, val in periodicity_fns.items():
            periodicity_results[key] = periodicity_quality(
                directory,
                val,
                datasets,
                checkpoint=checkpoint,
                gpu=gpu)

        # Write periodicity results
        file = penn.EVAL_DIR / penn.CONFIG / 'periodicity.json'
        with open(file, 'w') as file:
            json.dump(periodicity_results, file, indent=4)



###############################################################################
# Individual evaluations
###############################################################################


def benchmark(
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=None,
    gpu=None):
    """Perform benchmarking"""
    # Get audio files
    dataset_stems = {
        dataset: penn.load.partition(dataset)['test'] for dataset in datasets}
    files = [
        penn.CACHE_DIR / dataset / f'{stem}.wav'
        for dataset, stems in dataset_stems.items()
        for stem in stems]

    # Setup temporary directory
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Create output directories
        for dataset in datasets:
            (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Get output prefixes
        output_prefixes = [
            directory / file.parent.name / file.stem for file in files]

        # Start benchmarking
        penn.BENCHMARK = True
        torchutil.time.reset()
        start_time = time.time()

        # Infer to temporary storage
        if penn.METHOD == 'penn':
            penn.from_files_to_files(
                files,
                output_prefixes,
                checkpoint=checkpoint,
                batch_size=penn.EVALUATION_BATCH_SIZE,
                center='half-hop',
                gpu=gpu)

        elif penn.METHOD == 'torchcrepe':

            import torchcrepe

            # Get output file paths
            pitch_files = [
                file.parent / f'{file.stem}-pitch.pt'
                for file in output_prefixes]
            periodicity_files = [
                file.parent / f'{file.stem}-periodicity.pt'
                for file in output_prefixes]

            # Infer
            # Note - this does not perform the correct padding, but suffices
            #        for benchmarking purposes
            torchcrepe.predict_from_files_to_files(
                files,
                pitch_files,
                output_periodicity_files=periodicity_files,
                hop_length=penn.HOPSIZE,
                decoder=torchcrepe.decode.argmax,
                batch_size=penn.EVALUATION_BATCH_SIZE,
                device='cpu' if gpu is None else f'cuda:{gpu}',
                pad=False)
        elif penn.METHOD == 'dio':
            penn.dsp.dio.from_files_to_files(files, output_prefixes)
        elif penn.METHOD == 'pyin':
            penn.dsp.pyin.from_files_to_files(files, output_prefixes)

        # Turn off benchmarking
        penn.BENCHMARK = False

        # Get benchmarking information
        benchmark = torchutil.time.results()
        benchmark['total'] = time.time() - start_time

    # Get total number of samples and seconds in test data
    samples = sum([
        len(np.load(file.parent / f'{file.stem}-audio.npy', mmap_mode='r'))
        for file in files])
    seconds = penn.convert.samples_to_seconds(samples)

    # Format benchmarking results
    return {
        key: {
            'real-time-factor': value / seconds,
            'samples': samples,
            'samples-per-second': samples / value,
            'seconds': value
        } for key, value in benchmark.items()}


def periodicity_quality(
        directory,
        periodicity_fn,
        datasets=penn.EVALUATION_DATASETS,
        steps=8,
        checkpoint=None,
        gpu=None):
    """Fine-grained periodicity estimation quality evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Evaluate each dataset
    for dataset in datasets:

        # Create output directory
        (directory / dataset).mkdir(exist_ok=True, parents=True)

        # Iterate over validation set
        for audio, _, _, voiced, stem in torchutil.iterator(
            penn.data.loader([dataset], 'valid', True),
            f'Evaluating {penn.CONFIG} periodicity quality on {dataset}'
        ):

            if penn.METHOD == 'penn':

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
                    batch_logits = penn.infer(frames, checkpoint).detach()

                    # Accumulate logits
                    logits.append(batch_logits)

                logits = torch.cat(logits)

            elif penn.METHOD == 'torchcrepe':

                import torchcrepe

                # Accumulate logits
                logits = []

                # Postprocessing breaks gradients, so just don't compute them
                with torch.no_grad():

                    # Preprocess Xh62H3!QkeGCVp3OXh62H3!QkeGCVp3Oaudio
                    batch_size = \
                        None if gpu is None else penn.EVALUATION_BATCH_SIZE
                    pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                    generator = torchcrepe.preprocess(
                        torch.nn.functional.pad(audio, (pad, pad))[0],
                        penn.SAMPLE_RATE,
                        penn.HOPSIZE,
                        batch_size,
                        device,
                        False)
                    for frames in generator:

                        # Infer independent probabilities for each pitch bin
                        batch_logits = torchcrepe.infer(
                            frames.to(device))[:, :, None]

                        # Accumulate logits
                        logits.append(batch_logits)
                    logits = torch.cat(logits)

            elif penn.METHOD == 'pyin':

                # Pad
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                audio = torch.nn.functional.pad(audio, (pad, pad))

                # Infer
                logits = penn.dsp.pyin.infer(audio[0])

            # Save to temporary storage
            file = directory / dataset / f'{stem[0]}-logits.pt'
            torch.save(logits, file)

    # Default values
    best_threshold = .5
    best_value = 0.
    stepsize = .05

    # Setup metrics
    metrics = penn.evaluate.metrics.F1()

    step = 0
    while step < steps:

        for dataset in datasets:

            # Setup loader
            loader = penn.data.loader([dataset], 'valid', True)

            # Iterate over validation set
            for _, _, _, voiced, stem in loader:

                # Load logits
                logits = torch.load(directory / dataset / f'{stem[0]}-logits.pt')

                # Decode periodicity
                periodicity = periodicity_fn(logits.to(device)).T

                # Update metrics
                metrics.update(periodicity, voiced.to(device))

        # Get best performing threshold
        results = {
            key: val for key, val in metrics().items() if key.startswith('f1')
            and not math.isnan(val)}
        key = max(results, key=results.get)
        threshold = float(key[3:])
        value = results[key]
        if value > best_value:
            best_value = value
            best_threshold = threshold

        # Reinitialize metrics with new thresholds
        metrics = penn.evaluate.metrics.F1(
            [best_threshold - stepsize, best_threshold + stepsize])

        # Binary search for optimal threshold
        stepsize /= 2
        step += 1

    # Setup metrics with optimal threshold
    metrics = penn.evaluate.metrics.F1([best_threshold])

    # Setup test loader
    loader = penn.data.loader(datasets, 'test')

    # Iterate over test set
    for audio, _, _, voiced, stem in loader:

        if penn.METHOD == 'penn':

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
                batch_logits = penn.infer(frames, checkpoint).detach()

                # Accumulate logits
                logits.append(batch_logits)

            logits = torch.cat(logits)

        elif penn.METHOD == 'torchcrepe':

            import torchcrepe

            # Accumulate logits
            logits = []

            # Postprocessing breaks gradients, so just don't compute them
            with torch.no_grad():

                # Preprocess audio
                batch_size = \
                    None if gpu is None else penn.EVALUATION_BATCH_SIZE
                pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
                generator = torchcrepe.preprocess(
                    torch.nn.functional.pad(audio, (pad, pad))[0],
                    penn.SAMPLE_RATE,
                    penn.HOPSIZE,
                    batch_size,
                    device,
                    False)
                for frames in generator:

                    # Infer independent probabilities for each pitch bin
                    batch_logits = torchcrepe.infer(
                        frames.to(device))[:, :, None]

                    # Accumulate logits
                    logits.append(batch_logits)
                logits = torch.cat(logits)

        elif penn.METHOD == 'pyin':

            # Pad
            pad = (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
            audio = torch.nn.functional.pad(audio, (pad, pad))

            # Infer
            logits = penn.dsp.pyin.infer(audio[0]).to(device)

        # Decode periodicity
        periodicity = periodicity_fn(logits).T

        # Update metrics
        metrics.update(periodicity, voiced.to(device))

    # Get F1 score on test set
    score = metrics()[f'f1-{best_threshold:.6f}']

    return {'threshold': best_threshold, 'f1': score}


def pitch_quality(
    directory,
    datasets=penn.EVALUATION_DATASETS,
    checkpoint=None,
    gpu=None,
    iterations=None,
    silence=False,
    inference_only=False):
    """Evaluate pitch estimation quality"""

    # only made for penn, as this is a polyphonic pitch recognition project
    if penn.METHOD != 'penn':
        return

    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    print(f"Iterations: {iterations}")

    # Get metric class
    metric_fn = penn.evaluate.MutliPitchMetrics
    metric_fn_pitch = penn.evaluate.PitchMetrics

    # Per-file metrics
    file_metrics = metric_fn()
    file_metrics_pitch = metric_fn_pitch()

    # Per-dataset metrics
    dataset_metrics = metric_fn()
    dataset_metrics_pitch = metric_fn_pitch()

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn()
    aggregate_metrics_pitch = metric_fn_pitch()


    # Evaluate each dataset
    for dataset in datasets:

        dataset_iterator = 0

        # Reset dataset metrics
        dataset_metrics.reset()
        dataset_metrics_pitch.reset()

        # Iterate over test set
        for audio, bins, gt_pitch, voiced, stem in torchutil.iterator(
            penn.data.loader([dataset], 'test'),
            f'Evaluating {penn.CONFIG} pitch quality on {dataset}'
        ):
            dataset_iterator += 1

            if iterations is not None and dataset_iterator > iterations:
                break


            # Accumulate logits
            logits = []

            # if penn.FCN:
            #     audio = audio.squeeze(dim=0)
            audio = audio.squeeze(dim=0)

            pred_pitch, pred_times, periodicity, logits = \
                    penn.common_utils.from_audio(
                            audio,
                            penn.SAMPLE_RATE,
                            checkpoint=checkpoint,
                            gpu=gpu,
                            silence=silence,
                            as_numpy=False)


            if inference_only:
                continue

            # Reset file metrics
            file_metrics.reset()
            file_metrics_pitch.reset()

            max_len = min(pred_pitch.shape[-1], gt_pitch.shape[-1])
            pred_pitch = pred_pitch[..., :max_len]
            gt_pitch = gt_pitch[..., :max_len].to(pred_pitch.device)
            voiced = voiced[..., :max_len]
            periodicity = periodicity[..., :max_len]

            # set non-voiced to 0 Hz
            gt_pitch[torch.logical_not(voiced)] = 0

            eval_args = (pred_pitch, periodicity, gt_pitch, gt_pitch != 0)
            eval_args_pitch = (pred_pitch, gt_pitch, gt_pitch != 0)

            file_metrics.update(*eval_args)
            dataset_metrics.update(*eval_args)
            aggregate_metrics.update(*eval_args)

            file_metrics_pitch.update(*eval_args_pitch)
            dataset_metrics_pitch.update(*eval_args_pitch)
            aggregate_metrics_pitch.update(*eval_args_pitch)

            # Copy results
            granular_dict = file_metrics()
            granular_dict.update(file_metrics_pitch())
            granular[f'{dataset}/{stem[0]}'] = granular_dict


        if inference_only:
            continue

        overall_dict = dataset_metrics()
        overall_dict.update(dataset_metrics_pitch())
        overall[dataset] = overall_dict

    if inference_only:
        return

    aggregate_dict = aggregate_metrics()
    aggregate_dict.update(aggregate_metrics_pitch())
    overall['aggregate'] = aggregate_dict

    # Write to json files
    directory = penn.EVAL_DIR / penn.CONFIG
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
