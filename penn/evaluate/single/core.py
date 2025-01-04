import penn
import pprint
import torch

def from_file_to_file(audio_file,
                      ground_truth_file,
                      checkpoint,
                      output_file=None,
                      gpu=None,
                      start : float=0.0,
                      duration : float=None,
                      multipitch=False,
                      threshold=0.5,
                      plot_logits=False,
                      no_pred=True,
                      silence=False,
                      linewidth=0.5,
                      linewidth_gt=1.0,
                      fontsize=3,
                      no_legend=False,
                      no_title=False,
                      min_frequency=None,
                      max_frequency=None):

    audio, pred_pitch, pred_times, gt_pitch, gt_times, periodicity, logits = \
            penn.common_utils.from_path(
            audio_file,
            ground_truth_file,
            checkpoint,
            silence=silence,
            gpu=gpu,
            start=start,
            duration=duration)

    pred_pitch = torch.from_numpy(pred_pitch)
    periodicity = torch.from_numpy(periodicity)
    gt_pitch = torch.from_numpy(gt_pitch)

    if len(pred_pitch.shape) == 2:
        pred_pitch = pred_pitch.unsqueeze(dim=0)

    if len(periodicity.shape) == 2:
        periodicity = periodicity.unsqueeze(dim=0)

    if len(gt_pitch.shape) == 2:
        gt_pitch = gt_pitch.unsqueeze(dim=0)

    # Get metric class
    metrics = penn.evaluate.MutliPitchMetrics()
    metrics.update(pred_pitch, periodicity, gt_pitch, gt_pitch != 0)

    pprint.pp(f"metrics: {metrics()}")

