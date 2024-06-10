import torch

import penn

class PolyPitchNet(torch.nn.Sequential):
    def __init__(self, layers):
        super().__init__(*layers)

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        logits = super().forward(frames[:, :, 16:-15])
        
        # [128, 8640, 1] => [128, 1440, 1] * 6
        logits_chunks = logits.chunk(penn.PITCH_CATS, dim=-2)

        # shape [128, 6, 1440, 1]
        logits = torch.stack(logits_chunks, dim=1)

        return logits


class PolyPitchNet1(PolyPitchNet):

    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        layers += (
            Block(1, 256, 481, (2, 2)),
            Block(256, 32, 225, (2, 2)),
            Block(32, 32, 97, (2, 2)),
            Block(32, 128, 66),
            Block(128, 256, 35),
            Block(256, 512, 4),
            torch.nn.Conv1d(512, penn.PITCH_BINS * penn.PITCH_CATS, 4))
        super().__init__(layers)


class PolyPitchNet60x2(PolyPitchNet):

    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        layers += (
            Block(1, 256, 481, (2, 2)),
            Block(256, 32, 225, (2, 2)),
            Block(32, 32, 97, (2, 2)),
            Block(32, 128, 66),
            Block(128, 256, 35),
            Block(256, 512, 4),
            torch.nn.Conv1d(512, penn.PITCH_CATS * penn.PITCH_BINS * 2, 4))
        super().__init__(layers)

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        logits = torch.nn.Sequential.forward(self, frames[:, :, 16:-15])

        # [128, 1440, 6] => [128, 6, 1440]
        logits = logits.permute(0, 2, 1)
        
        # [128, 1440, 6] => [128, 6, 1440, 1]
        logits = logits[..., None]

        return logits

class Block(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        pooling=None,
        kernel_size=32,
        padding=0):
        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding), )

        if penn.RELU == 'leaky':
            layers += (torch.nn.LeakyReLU(),)
        else: 
            layers += (torch.nn.ReLU(),)

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add normalization
        if penn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        # Maybe add dropout
        if penn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)
