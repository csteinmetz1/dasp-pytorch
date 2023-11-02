import os
import glob
import torch
import auraloss
import torchaudio
import numpy as np

from tqdm import tqdm
from typing import List
from dasp_pytorch import ParametricEQ, Compressor, NoiseShapedReverb


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=2,
        )
        self.relu1 = torch.nn.PReLU(out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=1,
        )
        self.relu2 = torch.nn.PReLU(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        return x


class Encoder(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # we will use a simple TCN to estimate a single conditioning parameter
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(TCNBlock(1, 16, 7, dilation=1))
        self.blocks.append(TCNBlock(16, 32, 7, dilation=2))
        self.blocks.append(TCNBlock(32, 64, 7, dilation=4))
        self.blocks.append(TCNBlock(64, 128, 7, dilation=8))
        self.blocks.append(TCNBlock(128, 128, 7, dilation=16))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)  # aggregate over time
        return self.mlp(x)  # map to latent


class ParameterProjector(torch.nn.Module):
    def __init__(self, embed_dim: int, num_control_params: int, num_hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hidden = num_hidden
        self.num_control_params = num_control_params

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_control_params),
        )

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.layers(x))


class StyleTransferModel(torch.nn.Module):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()

        # create efffects
        self.equalizer = ParametricEQ(sample_rate)
        self.compressor = Compressor(sample_rate)
        self.reverb = NoiseShapedReverb(sample_rate)

        self.equalizer_parameters = torch.nn.Parameter(
            torch.randn(self.equalizer.num_params)
        )
        self.compressor_parameters = torch.nn.Parameter(
            torch.randn(self.compressor.num_params)
        )
        self.reverb_parameters = torch.nn.Parameter(torch.randn(self.reverb.num_params))

        # create networks
        self.encoder = Encoder(512)
        self.equalizer_projector = ParameterProjector(
            self.encoder.embed_dim * 2, self.equalizer.num_params
        )
        self.compressor_projector = ParameterProjector(
            self.encoder.embed_dim * 2, self.compressor.num_params
        )
        self.reverb_projector = ParameterProjector(
            self.encoder.embed_dim * 2, self.reverb.num_params
        )

    def forward(self, input: torch.Tensor, ref: torch.Tensor):
        # process the input and reference with encoder
        z_input = self.encoder(input)
        z_ref = self.encoder(ref)

        # combine the input and reference embeddings
        z = torch.cat((z_input, z_ref), dim=-1)

        # estimate parameters for each effect
        equalizer_params = self.equalizer_projector(z)
        compressor_params = self.compressor_projector(z)
        reverb_params = self.reverb_projector(z)

        # process audio with estimated parameters
        y = self.equalizer.process_normalized(input, equalizer_params)
        y = self.compressor.process_normalized(y, compressor_params)
        y = self.reverb.process_normalized(y, reverb_params)

        return y


class AudioFileDataset(torch.nn.Module):
    def __init__(
        self,
        filepaths: List[str],
        length: int = 131072,
    ) -> None:
        super().__init__()
        self.length = length

        assert len(filepaths) > 0, "No files found."

        self.examples = []
        # create example of length `length` from each file
        print("Creating dataset...")
        for filepath in tqdm(filepaths):
            md = torchaudio.info(filepath)
            if md.num_frames < length:
                continue
            num_examples = md.num_frames // length
            for n in range(num_examples):
                frame_offset = n * length

                frame, sr = torchaudio.load(
                    filepath,
                    frame_offset=frame_offset,
                    num_frames=length,
                    backend="soundfile",
                )

                # check for silence
                if torch.max(torch.abs(frame)) < 1e-4:
                    continue

                self.examples.append((filepath, frame_offset))

        self.examples = self.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        filepath, frame_offset = self.examples[idx]

        # read segment of audio from file
        x, sr = torchaudio.load(
            filepath,
            frame_offset=frame_offset,
            num_frames=self.length,
            backend="soundfile",
        )

        # clamp to [-1,1] to ensure within range
        x = torch.clamp(x, -1, 1)

        return x


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 1e-3,
    epochs: int = 100,
    use_gpu: bool = False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss()

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    epoch_loss_history = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        loss_history = []
        for batch in pbar:
            input = batch

            if use_gpu:
                input = input.cuda()

            # generate reference by randomly processing input
            rand_equalizer_params = torch.rand(
                input.shape[0], model.equalizer.num_params
            ).type_as(input)
            rand_compressor_params = torch.rand(
                input.shape[0], model.compressor.num_params
            ).type_as(input)
            rand_reverb_params = torch.rand(
                input.shape[0], model.reverb.num_params
            ).type_as(input)

            ref = model.equalizer.process_normalized(input, rand_equalizer_params)
            ref = model.compressor.process_normalized(ref, rand_compressor_params)
            ref = model.reverb.process_normalized(ref, rand_reverb_params)

            # split into A and B sections
            input_a, input_b = torch.chunk(input, 2, dim=-1)
            ref_a, ref_b = torch.chunk(ref, 2, dim=-1)

            # forward pass
            output_a = model(input_a, torch.mean(ref_b, dim=1, keepdim=True))

            # compute loss on A section
            loss = loss_fn(output_a, ref_a)
            loss_history.append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss: {np.mean(loss_history):.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_history.append(np.mean(loss_history))


if __name__ == "__main__":
    sample_rate = 44100
    model = StyleTransferModel(sample_rate)

    filepaths = glob.glob(
        "/import/c4dm-datasets/VocalSet1-2/data_by_singer/**/*.wav",
        recursive=True,
    )
    train_filepaths = filepaths[: int(len(filepaths) * 0.8)]
    val_filepaths = filepaths[int(len(filepaths) * 0.8) :]
    dataset = AudioFileDataset(train_filepaths, length=262144)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=16
    )

    train(model, dataloader, use_gpu=True)
