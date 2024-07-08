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
        # logits = logits.permute(0, 1, 3, 2)
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
        return logits

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


class PolyPENNDFCN(PolyPENNFCNBase):
    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()

        if type(penn.KERNEL_SIZE) is not list:
            raise ValueError("penn.KERNEL_SIZE should be a list of integers")

        for chann_in, chann_out, ks, pds, dil in zip(
                penn.CHANN_IN, penn.CHANN_OUT, penn.KERNEL_SIZE, penn.PADDING_SIZE, penn.DILATION):
            layers += (
                Block(
                    in_channels=chann_in,
                    out_channels=chann_out,
                    kernel_size=ks,
                    padding=pds,
                    dilation=dil), )

        layers += (
            torch.nn.Conv1d(penn.CHANN_OUT[-1], penn.PITCH_BINS * penn.PITCH_CATS, 1), )
        super().__init__(layers)



class PolyPENNHFCN(PolyPENNFCNBase):
    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        layers += (
            Block(penn.HOPSIZE, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 128, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(128, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 512, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(512, penn.PITCH_BINS, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            HierarchicalBlock(penn.PITCH_BINS, 6))
        super().__init__(layers)


class PolyPENNRFCN(PolyPENNFCNBase):
    def __init__(self):
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        layers += (
            Block(penn.HOPSIZE, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 32, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(32, 128, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(128, 256, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(256, 512, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            Block(512, penn.PITCH_BINS * penn.PITCH_CATS, kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
            ReccurentBlock(),
            torch.nn.Conv1d(penn.PITCH_BINS * penn.PITCH_CATS, penn.PITCH_BINS * penn.PITCH_CATS, 1))
        super().__init__(layers)


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
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), )

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


class HierarchicalBlock(torch.nn.Module):
    def __init__(
        self,
        pitch_bins,
        no_strings):

        super().__init__()

        self.pitch_bins = pitch_bins
        self.no_strings = no_strings

        self.string_blocks = []
        self.head_blocks = []

        for string in range(no_strings):
            string_block = torch.nn.Sequential(
                Block((1 + string)*self.pitch_bins, (1+string)*2048,
                    kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
                Block((1 + string)*2048, self.pitch_bins,
                    kernel_size=penn.KERNEL_SIZE, padding=penn.PADDING_SIZE),
                )

            self.string_blocks.append(string_block)

            head_block = torch.nn.Conv1d(penn.PITCH_BINS, penn.PITCH_BINS, 1)
            self.head_blocks.append(head_block)

    def forward(self, embeddings):
        embeddings = embeddings.to(torch.float)
        embeddings_list = [embeddings]
        logits_list = []

        for string in range(self.no_strings):
            stacked_embeddings = torch.cat(embeddings_list, dim=1)

            embeddings_out = self.string_blocks[string](stacked_embeddings)
            embeddings_list.append(logits)

            logits_out = self.head_blocks[string](embeddings_out)
            logits_list.append(logits_out)

        logits = torch.cat(logits_list, dim=1)


class ReccurentBlock(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, embeddings):

        embeddings2 = torch.clone(embeddings) 
        embeddings2[..., 1:] = embeddings2[..., 0:-1]
        embeddings2[..., 0] = 0

        return embeddings + embeddings2
