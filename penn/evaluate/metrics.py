import torch
import torchutil

import penn


###############################################################################
# Constants
###############################################################################


# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.accuracy = torchutil.metrics.Accuracy()
        self.f1 = F1()
        self.loss = Loss()
        self.pitch_metrics = PitchMetrics()
        self.multi_pitch_metrics = MutliPitchMetrics()
        self.f1_silence = F1(prefix='silence-')

    def __call__(self):
        if penn.LOSS_MULTI_HOT:
            return (
                {
                    'loss': self.loss()
                })
        else:
            metrics_dict = {
                    'accuracy': self.accuracy(),
                    'loss': self.loss()
                    }
            metrics_dict |= self.f1() 
            metrics_dict |= self.pitch_metrics()
            metrics_dict |= self.f1_silence() 

            if penn.PITCH_CATS > 1:
               metrics_dict |= self.multi_pitch_metrics() 

            return metrics_dict

    def update(self, logits, bins, target, voiced, logits_silence=None):
        # Detach from graph
        logits = logits.detach()

        # Update loss
        binsT = bins.permute(*torch.arange(bins.ndim - 1, -1, -1))
        self.loss.update(logits[:, :penn.PITCH_BINS], binsT)

        with torchutil.time.context('decode'):
            predicted, pitch, periodicity = penn.postprocess(logits)

            if not penn.LOSS_MULTI_HOT:
                # Decode bins, pitch, and periodicity

                # Update bin accuracy
                self.accuracy.update(predicted[voiced], bins[voiced])

                # Update pitch metrics
                self.pitch_metrics.update(pitch, target, voiced)

                # Update periodicity metrics
                self.f1.update(periodicity, voiced)

                if logits_silence is not None:
                    logits_silence = logits_silence.detach()
                    self.f1_silence.update(logits_silence, voiced)

                if penn.PITCH_CATS > 1:
                   self.multi_pitch_metrics.update(pitch, periodicity, target, voiced)

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
        self.loss.reset()
        self.pitch_metrics.reset()
        self.multi_pitch_metrics.reset()
        self.f1_silence.reset()


class PitchMetrics:
    def __init__(self):
        self.l1 = L1()
        self.rca = RCA()
        self.rmse = RMSE()
        self.rpa = RPA()

    def __call__(self):
        return {
            'l1': self.l1(),
            'rca': self.rca(),
            'rmse': self.rmse(),
            'rpa': self.rpa()}

    def update(self, pitch, target, voiced):
        # Mask unvoiced
        pitch, target = pitch[voiced], target[voiced]

        # Update metrics
        self.l1.update(pitch, target)
        self.rca.update(pitch, target)
        self.rmse.update(pitch, target)
        self.rpa.update(pitch, target)

    def reset(self):
        self.l1.reset()
        self.rca.reset()
        self.rmse.reset()
        self.rpa.reset()


class MutliPitchMetrics:

    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))

        self.thresholds = thresholds
        self.frca = [FRCA() for _ in range(len(thresholds))]
        self.frca2 = [FRCA2() for _ in range(len(thresholds))]
        self.frmse = [RMSE() for _ in range(len(thresholds))]
        self.frmse2 = [FRMSE2() for _ in range(len(thresholds))]
        self.frpa = [RPA() for _ in range(len(thresholds))]

    def __call__(self):
        result = {}
        for frca, frca2, frmse, frmse2, frpa, threshold in zip(
            self.frca,
            self.frca2,
            self.frmse,
            self.frmse2,
            self.frpa,
            self.thresholds
        ):
            result |= {
                f'frca-{threshold:.6f}': frca(),
                f'frca2-{threshold:.6f}': frca2(),
                f'frmse-{threshold:.6f}': frmse(),
                f'frmse2-{threshold:.6f}': frmse2(),
                f'frpa-{threshold:.6f}': frpa()}

        return result

    def update(self, pitch, periodicity, target, target_voiced):
        for frca, frca2, frmse, frmse2, frpa, threshold in zip(
            self.frca,
            self.frca2,
            self.frmse,
            self.frmse2,
            self.frpa,
            self.thresholds
        ):
            pitch_with_periodicity = pitch.clone().detach()
            pitch_with_periodicity[periodicity < threshold] = 0
            target[torch.logical_not(target_voiced)] = 0

            BS = pitch_with_periodicity.shape[0]

            pitch_with_periodicity_chunks = pitch_with_periodicity.chunk(dim=0, chunks=BS)
            target_chunks = target.chunk(dim=0, chunks=BS)
            target_voiced_chunks = target_voiced.chunk(dim=0, chunks=BS)

            for pitch_with_periodicity, target, target_voiced in zip(pitch_with_periodicity_chunks, target_chunks, target_voiced_chunks):
                pitch_array, _ = penn.core.postprocess_pitch_and_sort(
                        pitch_with_periodicity,
                        target_voiced)

                target_array, target_voiced_compressed = penn.core.postprocess_pitch_and_sort(
                        target, target_voiced)

                # add a very small number to get rid off possible log errors
                pitch_array[pitch_array == 0] = penn.FMIN + 10e-5
                target_array[target_array == 0] = penn.FMIN + 10e-5
                # target_array[torch.logical_not(target_voiced_compressed)] = penn.FMIN + 10e-5

                pitch_cents = penn.convert.frequency_to_cents(pitch_array)
                target_cents = penn.convert.frequency_to_cents(target_array)

                # Update metrics
                frca.update(pitch_cents, target_cents, target_voiced_compressed)
                frca2.update(pitch_cents, target_cents, target_voiced_compressed)
                frmse.update(pitch_cents, target_cents)
                frmse2.update(pitch_cents, target_cents, target_voiced_compressed)
                frpa.update(pitch_cents, target_cents)

    def reset(self):
        for frca, frca2, frmse, frmse2, frpa in zip(
            self.frca,
            self.frca2,
            self.frmse,
            self.frmse2,
            self.frpa
        ):
            # reset metrics
            frca.reset()
            frca2.reset()
            frmse.reset()
            frmse2.reset()
            frpa.reset()

###############################################################################
# Individual metrics
###############################################################################

class FRMSE2(torchutil.metrics.RMSE):
    """Raw chroma accuracy"""
    def update(self, predicted, target, target_voiced):
        target_tmp = target.clone()

        # subtrack each row of predicted from the target
        for ind in range(penn.PITCH_CATS):
            # evaluate on per-string basis, row by row of voiced target
            voiced_row = target_voiced[..., ind, :].squeeze()
            target_row = target_tmp[..., ind, voiced_row]

            # evaluate predicted only where target is voiced
            predicted_tmp = predicted[..., voiced_row]

            # Compute pitch difference in cents row by row, cause it seems that other way does not work well
            difference = penn.cents(predicted_tmp, target_row)

            # find one minimum in each timestamp - find the string with the note
            difference_min, indxs_min = difference.abs().min(dim=1, keepdim=True)
            predicted_row = torch.gather(predicted_tmp, dim=1, index=indxs_min)

            super().update(
                penn.OCTAVE * torch.log2(predicted_row),
                penn.OCTAVE * torch.log2(target_row))


class FRCA2(torchutil.metrics.Average):
    """Very simple metric checking if ground truth is present
    in any of the predicted values, no mask"""
    def update(self, predicted, target, target_voiced):
        target_tmp = target.clone()

        # subtrack each row of predicted from the target
        for ind in range(penn.PITCH_CATS):
            # evaluate on per-string basis, row by row of voiced target
            voiced_row = target_voiced[..., ind, :].squeeze()
            target_row = target_tmp[..., ind, voiced_row]

            # evaluate predicted only where target is voiced
            predicted_tmp = predicted[..., voiced_row]

            # Compute pitch difference in cents
            difference = penn.cents(predicted_tmp, target_row)

            # Forgive octave errors
            difference[difference > (penn.OCTAVE - THRESHOLD)] -= penn.OCTAVE
            difference[difference < -(penn.OCTAVE - THRESHOLD)] += penn.OCTAVE

            difference_under_threshold = torch.abs(difference) < THRESHOLD
            difference_under_threshold_sum = difference_under_threshold.sum(dim=1)
            difference_under_threshold_sum = difference_under_threshold_sum.bool()

            # Count predictions that are within 50 cents of target
            super().update(
                difference_under_threshold_sum.sum(),
                target_row.numel())

class FRCA(torchutil.metrics.Average):
    """Raw chroma accuracy"""
    def update(self, predicted, target, target_voiced):
        target_tmp = target.clone()
        # store masked values for all the iterations
        target_masked = torch.zeros(predicted.shape).to(target.device)

        # subtrack each row of predicted from the target
        for ind in range(penn.PITCH_CATS):
            # evaluate on per-string basis
            row = predicted[..., ind, :]

            # Compute pitch difference in cents
            difference = penn.cents(target_tmp, row)

            try:
                # Forgive octave errors
                mask_saver = difference.get_mask()
                data_saver = difference.get_data()
                choser_mask = data_saver > (penn.OCTAVE - THRESHOLD)
                data_saver[choser_mask] -= penn.OCTAVE

                choser_mask = data_saver < -(penn.OCTAVE - THRESHOLD)
                data_saver[choser_mask] += penn.OCTAVE

                difference = torch.masked.masked_tensor(
                        data_saver,
                        mask_saver)

            except AttributeError:
                # Forgive octave errors
                difference[difference > (penn.OCTAVE - THRESHOLD)] -= penn.OCTAVE
                difference[difference < -(penn.OCTAVE - THRESHOLD)] += penn.OCTAVE

            # find minimum value from each 6 strings
            min_diff_ind = torch.argmin(torch.abs(difference), dim=1)

            try: 
                min_diff_ind = min_diff_ind.get_data().long()
            except AttributeError:
                pass

            min_diff_1hot = torch.nn.functional.one_hot(
                    min_diff_ind, num_classes=penn.PITCH_CATS).bool()
            min_diff_1hot = min_diff_1hot.permute(0, 2, 1)

            target_potential = torch.abs(difference) < THRESHOLD

            try: 
                target_potential = target_potential.get_data()
            except AttributeError:
                pass

            # merge minimal diff with those below the treshold
            target_potential = torch.logical_and(
                    target_potential, min_diff_1hot)

            # take 0s into account
            target_validated_zeros = torch.logical_and(
                    target_potential,
                    torch.logical_not(target_voiced))

            # remove values that were already used
            target_validated = torch.logical_and(
                    target_potential, torch.logical_not(target_masked))

            target_validated = torch.logical_or(
                    target_validated,
                    target_validated_zeros)

            # add freshly found differences to the masked target
            target_masked = torch.logical_or(target_masked, target_validated)

            try:
                target_tmp = target_tmp.get_data()
            except AttributeError:
                pass

            target_tmp = torch.masked.masked_tensor(target_tmp, torch.logical_not(target_masked))
            # target_tmp[target_masked] = penn.convert.frequency_to_cents(torch.tensor(penn.FMIN + 10e-5))

            # Count predictions that are within 50 cents of target
            super().update(
                target_validated.sum(),
                row.numel())

class F1:

    def __init__(self, thresholds=None, prefix=''):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))
        self.thresholds = thresholds
        self.precision = [
            torchutil.metrics.Precision() for _ in range(len(thresholds))]
        self.recall = [
            torchutil.metrics.Recall() for _ in range(len(thresholds))]
        self.prefix = prefix

    def __call__(self):
        result = {}
        for threshold, precision, recall in zip(
            self.thresholds,
            self.precision,
            self.recall
        ):
            precision = precision()
            recall = recall()
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.
            result |= {
                f'{self.prefix}f1-{threshold:.6f}': f1,
                f'{self.prefix}precision-{threshold:.6f}': precision,
                f'{self.prefix}recall-{threshold:.6f}': recall}
        return result

    def update(self, periodicity, voiced):
        for threshold, precision, recall in zip(
            self.thresholds,
            self.precision,
            self.recall
        ):
            predicted = penn.voicing.threshold(periodicity, threshold)
            precision.update(predicted, voiced)
            recall.update(predicted, voiced)

    def reset(self):
        """Reset the F1 score"""
        for precision, recall in zip(self.precision, self.recall):
            precision.reset()
            recall.reset()


class L1(torchutil.metrics.L1):
    """L1 pitch distance in cents"""
    def update(self, predicted, target):
        super().update(
            penn.OCTAVE * torch.log2(predicted),
            penn.OCTAVE * torch.log2(target))


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, logits, bins):
        super().update(penn.loss(logits, bins), bins.shape[0])


class RCA(torchutil.metrics.Average):
    """Raw chroma accuracy"""
    def update(self, predicted, target):
        # Compute pitch difference in cents
        difference = penn.cents(predicted, target)

        # Forgive octave errors
        difference[difference > (penn.OCTAVE - THRESHOLD)] -= penn.OCTAVE
        difference[difference < -(penn.OCTAVE - THRESHOLD)] += penn.OCTAVE

        # Count predictions that are within 50 cents of target
        super().update(
            (torch.abs(difference) < THRESHOLD).sum(),
            predicted.numel())


class RMSE(torchutil.metrics.RMSE):
    """Root mean square error of pitch distance in cents"""
    def update(self, predicted, target):
        super().update(
            penn.OCTAVE * torch.log2(predicted),
            penn.OCTAVE * torch.log2(target))


class RPA(torchutil.metrics.Average):
    """Raw prediction accuracy"""
    def update(self, predicted, target):
        difference = penn.cents(predicted, target)
        super().update(
            (torch.abs(difference) < THRESHOLD).sum(),
            predicted.numel())

