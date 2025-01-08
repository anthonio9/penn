import penn
import numpy as np
import torch 
import yapecs
import importlib
from pathlib import Path
import pytest


class TestRPA:
    pitch_pred = torch.Tensor(
                [
                    [30, 40, 50],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 21, 1],
                    [1, 1, 1],
                    [11, 1, 1],
                ]
            )


    pitch_gt = torch.Tensor(
                [
                    [30, 40, 50],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 21, 1],
                    [1, 1, 1],
                    [11, 1, 1],
                ]
            )

    pitch_pred2 = torch.Tensor(
                [
                    [30, 1, 1],
                    [1, 1, 50],
                    [1, 40, 1],
                    [1, 1, 1],
                    [1, 21, 1],
                    [1, 1, 1],
                ]
            )


    pitch_gt2 = torch.Tensor(
                [
                    [
                        [30, 40, 50],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 21, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                ]
            )

    voiced_gt2 = torch.LongTensor(
                [
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )

    pitch_pred3 = torch.Tensor(
                [
                    [300,	0,		0,		0,		0],
                    [0,		0,		500,	82,		0],
                    [0,		400,	0,		0,		0],
                    [0,		0,		0,		114,	0],
                    [0,		200,	0,		0,		0],
                    [0,		0,		0,		0,		0],
                ],
            )

    pitch_gt3 = torch.Tensor(
                [
                    [300,   0,      0,      82,     0],
                    [0,     400,    500,    0,      0],
                    [0,     0,      0,      0,      0],
                    [335,   0,      0,      114,    0],
                    [0,     200,    0,      0,      0],
                    [0,     0,      0,      0,      0],
                ]
            )

    @classmethod
    def setup_class(cls):
        cls.pitch_pred3[cls.pitch_gt3 != 0] + 1e-10
        cls.pitch_gt3[cls.pitch_gt3 != 0] + 1e-10


    def test_rpas(self):
        RPA1 = penn.evaluate.metrics.RPA()
        RPA1.update(self.pitch_pred, self.pitch_gt)
        assert RPA1() == 1.0

        RPA1.reset()
        RPA1.update(self.pitch_pred3, 
                    self.pitch_gt3)
        assert RPA1() == pytest.approx(4/7) 

    def test_frpa2(self):
        RPA = penn.evaluate.metrics.FRPA2()
        RPA.update(self.pitch_pred, self.pitch_gt, self.pitch_gt != 1, pitch_cats=6)

        assert RPA() == 1.0

        RPA.reset()
        RPA.update(self.pitch_pred2, self.pitch_gt2, self.voiced_gt2, pitch_cats=6)

        assert RPA() == 1.0

        RPA.reset()
        voiced = self.pitch_gt3 != 0

        RPA.update(self.pitch_pred3, self.pitch_gt3, voiced, pitch_cats=6)

        assert RPA() == pytest.approx(6/7)        

    def test_rmse(self):

        RMSE = penn.evaluate.metrics.FRMSE2()

        RMSE.reset()
        predicted = torch.tensor([[[177.0]]])
        target = torch.tensor([[[320]]])
        voiced = torch.tensor([[[True]]])
        RMSE.update(predicted,
                    target,
                    voiced, pitch_cats=1)

        assert RMSE() == pytest.approx(abs(penn.cents(predicted, target)))

        RMSE.reset()
        predicted = penn.convert.frequency_to_cents(torch.tensor([[[177.0]]]))
        target = penn.convert.frequency_to_cents(torch.tensor([[[320]]]))
        voiced = torch.tensor([[[True]]])
        RMSE.update(predicted,
                    target,
                    voiced, pitch_cats=1)

        assert RMSE() == pytest.approx(abs(penn.cents(predicted, target)))


    def test_multimetrics(self):
        multi = penn.evaluate.metrics.MutliPitchMetrics([0.5], pitch_cats=6)

        self.pitch_pred3 = torch.unsqueeze(self.pitch_pred3, 0)
        self.pitch_gt3 = torch.unsqueeze(self.pitch_gt3, 0)
        multi.update(self.pitch_pred3, (self.pitch_pred3 != 0).double(), self.pitch_gt3, self.pitch_gt3 != 0)
        print(multi())

        multi.reset()

        pred = self.pitch_pred3.clone()
        pred[..., 0, :] = 0
        
        multi.update(pred, (pred != 0).double(), self.pitch_gt3, self.pitch_gt3 != 0)
        print(multi())
