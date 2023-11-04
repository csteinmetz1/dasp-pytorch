import os
import glob
import torch
import auraloss
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Optional
from dasp_pytorch import ParametricEQ, Compressor, NoiseShapedReverb, Gain


def plot_loss(log_dir, loss_history: List[float]):
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.grid(c="lightgray")
    outfilepath = os.path.join(log_dir, "loss.png")
    plt.savefig(outfilepath, dpi=300)
    plt.close("all")


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
    def __init__(self, embed_dim: int, ch_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # we will use a simple TCN to estimate a single conditioning parameter
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(TCNBlock(1, ch_dim, 7, dilation=1))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=2))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=4))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=8))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=16))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=1))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=2))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=4))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=8))
        self.blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=16))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(ch_dim, 256),
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
        self.gain = Gain(sample_rate)

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
        self.gain_projector = ParameterProjector(
            self.encoder.embed_dim * 2, self.gain.num_params
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
        gain_params = self.gain_projector(z)

        # process audio with estimated parameters
        y = input.clone()
        y = self.equalizer.process_normalized(y, equalizer_params)
        y = self.compressor.process_normalized(y, compressor_params)
        y = self.reverb.process_normalized(y, reverb_params)
        y = self.gain.process_normalized(y, gain_params)

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


def validate(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    num_examples: int = 1,
    use_gpu: bool = False,
    epoch: int = 0,
    log_dir: str = "outputs/style_transfer",
):
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= num_examples:
            break

        input = batch

        if use_gpu:
            input = input.cuda()

        with torch.no_grad():
            input_a, input_b, ref_a, ref_b, output_a = step(input, model)

        # save audio examples
        input_a_filepath = os.path.join(
            log_dir, "audio", f"epoch={epoch}_input_a_{batch_idx}.wav"
        )
        input_b_filepath = os.path.join(
            log_dir, "audio", f"epoch={epoch}_input_b_{batch_idx}.wav"
        )
        ref_a_filepath = os.path.join(
            log_dir, "audio", f"epoch={epoch}_ref_a_{batch_idx}.wav"
        )
        ref_b_filepath = os.path.join(
            log_dir, "audio", f"epoch={epoch}_ref_b_{batch_idx}.wav"
        )
        output_a_filepath = os.path.join(
            log_dir, "audio", f"epoch={epoch}_output_a_{batch_idx}.wav"
        )
        torchaudio.save(
            input_a_filepath, input_a.cpu().squeeze(0), 44100, backend="soundfile"
        )
        torchaudio.save(
            input_b_filepath, input_b.cpu().squeeze(0), 44100, backend="soundfile"
        )
        torchaudio.save(
            ref_a_filepath, ref_a.cpu().squeeze(0), 44100, backend="soundfile"
        )
        torchaudio.save(
            ref_b_filepath, ref_b.cpu().squeeze(0), 44100, backend="soundfile"
        )
        torchaudio.save(
            output_a_filepath, output_a.cpu().squeeze(0), 44100, backend="soundfile"
        )


def step(input: torch.Tensor, model: torch.nn.Module):
    # generate reference by randomly processing input
    # torch.manual_seed(1)
    rand_equalizer_params = torch.rand(
        input.shape[0],
        model.equalizer.num_params,
    ).type_as(input)
    rand_compressor_params = torch.rand(
        input.shape[0],
        model.compressor.num_params,
    ).type_as(input)
    rand_reverb_params = torch.rand(
        input.shape[0],
        model.reverb.num_params,
    ).type_as(input)
    rand_gain_params = torch.rand(
        input.shape[0],
        model.gain.num_params,
    ).type_as(input)

    # process input with random parameters
    # randomly disable the effects
    ref = input.clone()
    # if torch.rand(1) < 0.5:
    ref = model.equalizer.process_normalized(ref, rand_equalizer_params)
    # if torch.rand(1) < 0.5:
    ref = model.compressor.process_normalized(ref, rand_compressor_params)
    # if torch.rand(1) < 0.5:
    ref = model.reverb.process_normalized(ref, rand_reverb_params)

    # ref = model.gain.process_normalized(ref, rand_gain_params)

    # if not stereo already, convert to stereo
    # if ref.shape[1] == 1:
    #    ref = ref.repeat(1, 2, 1)

    # peak normalize reference recordings
    peak, _ = torch.max(torch.abs(ref), dim=-1, keepdim=True)
    ref = ref / peak

    # apply random gain from -24 dB to 0 dB
    gain_db = torch.rand(input.shape[0], 1, 1).type_as(input) * 24
    gain_lin = torch.pow(10, -gain_db / 20)
    ref = ref * gain_lin

    # apply random gain to input
    gain_db = torch.rand(input.shape[0], 1, 1).type_as(input) * 24
    gain_lin = torch.pow(10, -gain_db / 20)
    input = input * gain_lin

    # split into A and B sections
    input_a, input_b = torch.chunk(input, 2, dim=-1)
    ref_a, ref_b = torch.chunk(ref, 2, dim=-1)

    # forward pass
    output_a = model(input_a, torch.mean(ref_b, dim=1, keepdim=True))

    return input_a, input_b, ref_a, ref_b, output_a


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    lr: float = 1e-4,
    epochs: int = 250,
    use_gpu: bool = False,
    log_dir: str = "outputs/style_transfer",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    epoch_loss_history = []
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        loss_history = []
        model.train()
        for batch in pbar:
            input = batch

            if use_gpu:
                input = input.cuda()

            # forward pass
            input_a, input_b, ref_a, ref_b, output_a = step(input, model)

            # compute loss on A section
            loss = loss_fn(output_a, ref_a)
            loss_history.append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss: {np.mean(loss_history):.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss_history.append(np.mean(loss_history))
        plot_loss(log_dir, epoch_loss_history)
        validate(
            model,
            val_dataloader,
            epoch=epoch + 1,
            log_dir=log_dir,
            use_gpu=use_gpu,
        )


if __name__ == "__main__":
    sample_rate = 44100
    log_dir = "outputs/style_transfer"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "audio"), exist_ok=True)
    model = StyleTransferModel(sample_rate)

    filepaths = glob.glob(
        "/import/c4dm-datasets/VocalSet1-2/data_by_singer/**/*.wav",
        recursive=True,
    )
    train_filepaths = filepaths[: int(len(filepaths) * 0.8)]
    val_filepaths = filepaths[int(len(filepaths) * 0.8) :]

    # train_filepaths = train_filepaths[:1]
    # val_filepaths = train_filepaths[:1]

    train_dataset = AudioFileDataset(train_filepaths, length=262144)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )

    val_dataset = AudioFileDataset(val_filepaths, length=262144)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    train(
        model,
        train_dataloader,
        val_dataloader=val_dataloader,
        log_dir=log_dir,
        use_gpu=True,
    )
