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

            if penn.PITCH_CATS > 1:
               metrics_dict |= self.multi_pitch_metrics() 

            return metrics_dict

    def update(self, 
               logits_dict : dict[str, torch.tensor],
               bins : torch.Tensor,
               target : torch.Tensor,
               voiced : torch.Tensor):
        # Detach from graph
        logits = logits_dict[penn.model.KEY_LOGITS]

        # Update loss
        binsT = bins.permute(*torch.arange(bins.ndim - 1, -1, -1))
        self.loss.update(logits[:, :penn.PITCH_BINS], binsT)

        with torchutil.time.context('decode'):
            predicted, pitch, periodicity = penn.postprocess(logits_dict)

            if not penn.LOSS_MULTI_HOT:
                # Decode bins, pitch, and periodicity

                # Update bin accuracy
                self.accuracy.update(predicted[voiced], bins[voiced])

                # Update pitch metrics
                self.pitch_metrics.update(pitch, target, voiced)

                # Update periodicity metrics
                self.f1.update(periodicity, voiced)

                if penn.PITCH_CATS > 1:
                   self.multi_pitch_metrics.update(pitch, periodicity, target, voiced)

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
        self.loss.reset()
        self.pitch_metrics.reset()
        self.multi_pitch_metrics.reset()


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

    def __init__(self, thresholds=None, pitch_cats=penn.PITCH_CATS):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))

        self.thresholds = thresholds
        self.pitch_cats = pitch_cats
        self.mrca = [FRCA2() for _ in range(len(thresholds))]
        self.mrmse = [FRMSE2() for _ in range(len(thresholds))]
        self.mrpa = [FRPA2() for _ in range(len(thresholds))]
        self.mrca2 = [FRCA2() for _ in range(len(thresholds))]
        self.mrmse2 = [FRMSE2() for _ in range(len(thresholds))]
        self.mrpa2 = [FRPA2() for _ in range(len(thresholds))]

    def __call__(self):
        result = {}
        for mrca, mrmse, mrpa, mrca2, mrmse2, mrpa2, threshold in zip(
            self.mrca,
            self.mrmse,
            self.mrpa,
            self.mrca2,
            self.mrmse2,
            self.mrpa2,
            self.thresholds
        ):
            try:
                result |= {
                    f'mrca-{threshold:.6f}': mrca(),
                    f'mrmse-{threshold:.6f}': mrmse(),
                    f'mrpa-{threshold:.6f}': mrpa(),
                    f'mrca2-{threshold:.6f}': mrca2(),
                    f'mrmse2-{threshold:.6f}': mrmse2(),
                    f'mrpa2-{threshold:.6f}': mrpa2()}
            except ZeroDivisionError as er:
                result |= {}
                print(f"MultiPitchMetrics error for threshold {threshold}: {er}")

        return result

    def update(self, pitch, periodicity, target, target_voiced):
        for mrca, mrmse, mrpa, mrca2, mrmse2, mrpa2, threshold in zip(
            self.mrca,
            self.mrmse,
            self.mrpa,
            self.mrca2,
            self.mrmse2,
            self.mrpa2,
            self.thresholds
        ):
            pitch_with_periodicity = pitch.clone().detach()
            pitch_with_periodicity[periodicity < threshold] = 0
            target[torch.logical_not(target_voiced)] = 0

            BS = pitch_with_periodicity.shape[0]

            pitch_with_periodicity_chunks = pitch_with_periodicity.chunk(dim=0, chunks=BS)
            target_chunks = target.chunk(dim=0, chunks=BS)
            target_voiced_chunks = target_voiced.chunk(dim=0, chunks=BS)

            pitch_restored = torch.cat(pitch_with_periodicity_chunks, dim=-1)
            target_restored = torch.cat(target_chunks, dim=-1)
            target_voiced_restored = torch.cat(target_voiced_chunks, dim=-1)

            # remember for the "Special case" section
            predicted_voiced = torch.logical_not(pitch_restored == 0)

            # for RMSE
            pitch_non_empty, _ = penn.core.remove_empty_timestamps(pitch_restored, target_voiced_restored)
            target_non_empty, target_non_empty_voicing = penn.core.remove_empty_timestamps(target_restored, target_voiced_restored)

            # add a very small number to get rid off possible log errors
            pitch_non_empty[pitch_non_empty == 0] += 10e-5
            target_non_empty[target_non_empty == 0] += 10e-5

            # add a very small number to get rid off possible log errors
            pitch_restored[pitch_restored == 0] = penn.FMIN + 10e-5
            target_restored[target_restored == 0] = penn.FMIN + 10e-5

            pitch_cents = penn.convert.frequency_to_cents(pitch_restored)
            target_cents = penn.convert.frequency_to_cents(target_restored)

            # for pitch_with_periodicity, target, target_voiced in zip(pitch_with_periodicity_chunks, target_chunks, target_voiced_chunks):
            pitch_cents_not_empty, _ = penn.core.remove_empty_timestamps(
                    pitch_cents,
                    target_voiced_restored)

            target_cents_not_empty, target_voiced_compressed = penn.core.remove_empty_timestamps(
                    target_cents, target_voiced_restored)

            # Update metrics
            mrca.update(pitch_cents_not_empty, target_cents_not_empty, target_voiced_compressed, pitch_cats=self.pitch_cats)
            mrpa.update(pitch_cents_not_empty, target_cents_not_empty, target_voiced_compressed, pitch_cats=self.pitch_cats)
            # mrmse.update(pitch_non_empty, target_non_empty, target_non_empty_voicing, pitch_cats=self.pitch_cats)
            mrmse.update(pitch_cents_not_empty, target_cents_not_empty, target_voiced_compressed, pitch_cats=self.pitch_cats)

            # Special case with deleted empty timestamps in the pitch array
            # get common timestamps
            pred_voiced_summed = predicted_voiced.sum(dim=1).squeeze().bool()
            tar_voiced_summed = target_voiced_restored.sum(dim=1).squeeze().bool()
            common_voiced = torch.logical_and(pred_voiced_summed, tar_voiced_summed)

            target_voiced_compressed = target_voiced_restored[..., common_voiced]
            pitch_cents = pitch_cents[..., common_voiced]
            target_cents = target_cents[..., common_voiced]

            try: 
                mrca2.update(pitch_cents, target_cents, target_voiced_compressed, pitch_cats=self.pitch_cats)
                mrpa2.update(pitch_cents, target_cents, target_voiced_compressed, pitch_cats=self.pitch_cats)
                mrmse2.update(pitch_cents, target_cents, target_voiced_compressed, pitch_cats=self.pitch_cats)

            except IndexError as ex:
                print(f"MultiPitch Metrics: {ex}")

    def reset(self):
        for mrca, mrmse, mrpa, mrca2, mrmse2, mrpa2 in zip(
            self.mrca,
            self.mrmse,
            self.mrpa,
            self.mrca2,
            self.mrmse2,
            self.mrpa2
        ):
            # reset metrics
            mrca.reset()
            mrmse.reset()
            mrpa.reset()
            mrca2.reset()
            mrmse2.reset()
            mrpa2.reset()

###############################################################################
# Individual metrics
###############################################################################

class FRMSE2(torchutil.metrics.RMSE):
    """Raw chroma accuracy"""
    def update(self, predicted, target, target_voiced, pitch_cats=penn.PITCH_CATS):
        # subtrack each row of predicted from the target
        for ind in range(pitch_cats):
            # breakpoint()
            # evaluate on per-string basis, row by row of voiced target
            voiced_row = target_voiced[..., ind, :].squeeze()
            target_row = target[..., ind, voiced_row]

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
    def update(self, predicted, target, target_voiced, pitch_cats=penn.PITCH_CATS):
        # subtract each row of predicted from the target
        for ind in range(pitch_cats):
            # evaluate on per-string basis, row by row of voiced target
            voiced_row = target_voiced[..., ind, :].squeeze()
            target_row = target[..., ind, voiced_row]

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


class FRPA2(torchutil.metrics.Average):
    """Very simple metric checking if ground truth is present
    in any of the predicted values, no mask"""
    def update(self, predicted, target, target_voiced, pitch_cats=penn.PITCH_CATS):
        # subtract each row of predicted from the target
        for ind in range(pitch_cats):
            # evaluate on per-string basis, row by row of voiced target
            voiced_row = target_voiced[..., ind, :].squeeze()
            target_row = target[..., ind, voiced_row]

            # evaluate predicted only where target is voiced
            predicted_tmp = predicted[..., voiced_row]

            # Compute pitch difference in cents
            difference = penn.cents(predicted_tmp, target_row)

            difference_under_threshold = torch.abs(difference) < THRESHOLD
            difference_under_threshold_sum = difference_under_threshold.sum(dim=-2)
            difference_under_threshold_sum = difference_under_threshold_sum.bool()

            # Count predictions that are within 50 cents of target
            super().update(
                difference_under_threshold_sum.sum(),
                target_row.numel())


class F1:

    def __init__(self, thresholds=None, prefix=''):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))
        self.thresholds = thresholds
        self.accuracy = [
            torchutil.metrics.Accuracy() for _ in range(len(thresholds))]
        self.precision = [
            torchutil.metrics.Precision() for _ in range(len(thresholds))]
        self.recall = [
            torchutil.metrics.Recall() for _ in range(len(thresholds))]
        self.prefix = prefix

    def __call__(self):
        result = {}
        for threshold, accuracy, precision, recall in zip(
            self.thresholds,
            self.accuracy,
            self.precision,
            self.recall
        ):
            accuracy = accuracy()
            precision = precision()
            recall = recall()
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.
            result |= {
                f'{self.prefix}f1-{threshold:.6f}': f1,
                f'{self.prefix}accuracy-{threshold:.6f}': accuracy,
                f'{self.prefix}precision-{threshold:.6f}': precision,
                f'{self.prefix}recall-{threshold:.6f}': recall}
        return result

    def update(self, periodicity, voiced):
        for threshold, accuracy, precision, recall in zip(
            self.thresholds,
            self.accuracy,
            self.precision,
            self.recall
        ):
            predicted = penn.voicing.threshold(periodicity, threshold)
            accuracy.update(predicted, voiced)
            precision.update(predicted, voiced)
            recall.update(predicted, voiced)

    def reset(self):
        """Reset the F1 score"""
        for accuracy, precision, recall in zip(self.accuracy, self.precision, self.recall):
            accuracy.reset()
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

