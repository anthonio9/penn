import torch

import penn

class PolyPitchNet(torch.nn.Sequential):
    def __init__(self, layers):
        super().__init__(*layers)

    def forward(self, frames : torch.Tensor):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        logits = super().forward(frames[:, :, 16:-15])
        
        # [128, 8640, 1] => [128, 1440, 1] * 6
        logits_chunks = logits.chunk(penn.PITCH_CATS, dim=-2)

        # shape [128, 6, 1440, 1]
        logits = torch.stack(logits_chunks, dim=1)

        logits_dict = { penn.model.KEY_LOGITS   : logits }
        return logits_dict


class PolyPENNFCNBase(PolyPitchNet):
    def forward(self, frames : torch.Tensor):
        frames = frames.squeeze(dim=1)

        frames_list = torch.chunk(frames, chunks=frames.shape[-1] // penn.HOPSIZE, dim=-1)
        # frames shape [BATCH_SIZE, FRAMES, FRAME_LENGTH]
        # FRAMES dimention is made out of initial frames and the actual frames which are present in the ground truth. Meaning that the first few frames made out of penn.WINDOW_SIZE do not translate into ground truth at all and should be randomize on the ground truth side.
        frames = torch.stack(frames_list, dim=1)
        # [BS, T, HOPSIZE] => [BS, HOPSIZE, T]
        frames = torch.permute(frames, (0, 2, 1))

        logits = torch.nn.Sequential.forward(self, frames)

        # [128, 8640, *] => [128, 1440, *] * 6
        logits_chunks = logits.chunk(penn.PITCH_CATS, dim=-2)

        # shape [128, 6, 1440, *]
        logits = torch.stack(logits_chunks, dim=1)


        # [BS, PITCH_CATS, PITCH_BINS, T] => [BS, PITCH_CATS, T, PITCH_BINS]
        logits_dict = { penn.model.KEY_LOGITS   : logits }
        return logits_dict


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


class PolyPENNFCN(PolyPitchNet):
    def forward(self, frames : torch.Tensor):
        frames = frames.squeeze(dim=1)

        frames_list = torch.chunk(frames, chunks=frames.shape[-1] // penn.HOPSIZE, dim=-1)
        # frames shape [BATCH_SIZE, FRAMES, FRAME_LENGTH]
        # FRAMES dimention is made out of initial frames and the actual frames which are present in the ground truth. Meaning that the first few frames made out of penn.WINDOW_SIZE do not translate into ground truth at all and should be randomize on the ground truth side.
        frames = torch.stack(frames_list, dim=1)
        # [BS, T, HOPSIZE] => [BS, HOPSIZE, T]
        frames = torch.permute(frames, (0, 2, 1))

        logits = torch.nn.Sequential.forward(self, frames)

        # [128, 8640, *] => [128, 1440, *] * 6
        logits_chunks = logits.chunk(penn.PITCH_CATS, dim=-2)

        # shape [128, 6, 1440, *]
        logits = torch.stack(logits_chunks, dim=1)


        # [BS, PITCH_CATS, PITCH_BINS, T] => [BS, PITCH_CATS, T, PITCH_BINS]
        # logits = logits.permute(0, 1, 3, 2)
        logits_dict = { penn.model.KEY_LOGITS   : logits }
        return logits_dict

    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()

        layers += (
            Block(penn.HOPSIZE, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 128, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(128, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 512, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            torch.nn.Conv1d(512, penn.PITCH_BINS * penn.PITCH_CATS, 1))
        super().__init__(layers)


class PolyPENNFCN2(torch.nn.Module):
    def forward(self, frames : torch.Tensor):

        frames = frames.squeeze(dim=1)

        frames_list = torch.chunk(frames, chunks=frames.shape[-1] // penn.HOPSIZE, dim=-1)
        # frames shape [BATCH_SIZE, FRAMES, FRAME_LENGTH]
        # FRAMES dimention is made out of initial frames and the actual frames which are present in the ground truth. Meaning that the first few frames made out of penn.WINDOW_SIZE do not translate into ground truth at all and should be randomize on the ground truth side.
        frames = torch.stack(frames_list, dim=1)
        # [BS, T, HOPSIZE] => [BS, HOPSIZE, T]
        frames = torch.permute(frames, (0, 2, 1))

        main_logits = self.main_sequence(frames)
        logits = self.pitch_head(main_logits)
        silence_logits = self.silence_head(main_logits)

        # [128, 8640, *] => [128, 1440, *] * 6
        logits_chunks = logits.chunk(penn.PITCH_CATS, dim=-2)

        # shape [128, 6, 1440, *]
        logits = torch.stack(logits_chunks, dim=1)

        # [BS, PITCH_CATS, PITCH_BINS, T] => [BS, PITCH_CATS, T, PITCH_BINS]
        # logits = logits.permute(0, 1, 3, 2)
        logits_dict = {
                penn.model.KEY_LOGITS   : logits,
                penn.model.KEY_SILENCE  : silence_logits,
                       }
        return logits_dict

    def __init__(self):
        super().__init__()
        # self.silence_head = Block(512, 1024, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE)
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()

        layers += (
            Block(penn.HOPSIZE, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 128, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(128, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 512, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE))
            # torch.nn.Conv1d(512, penn.PITCH_BINS * penn.PITCH_CATS, 1))

        self.main_sequence = torch.nn.Sequential(*layers)
        self.pitch_head = torch.nn.Conv1d(512, penn.PITCH_BINS * penn.PITCH_CATS, 1)

        # prepare the SILENCE HEAD
        silence_layers = (
            Block(512, 1024, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            torch.nn.Conv1d(1024, penn.PITCH_CATS, 1))
        self.silence_head = torch.nn.Sequential(*silence_layers)


class Block(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        pooling=None,
        kernel_size=32,
        padding=0,
        dilation=1
        ):
        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            torch.nn.ReLU())

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
