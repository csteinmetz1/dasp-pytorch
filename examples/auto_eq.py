import os
import glob
import torch
import auraloss
import torchaudio
import numpy as np
import dasp_pytorch
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List

# In this example we will train a neural network to perform blind estimation of
# the audio effect parameters for a dynamic range compressor.


def plot_response(
    y: torch.Tensor,
    x_hat: torch.Tensor,
    x: torch.Tensor,
    sample_rate: int = 44100,
):
    fig, ax = plt.subplots(figsize=(6, 4))

    # compute frequency response of y
    Y = torch.fft.rfft(y)
    Y = torch.abs(Y)
    Y_db = 20 * torch.log10(Y + 1e-8)

    # compute frequency response of x_hat
    X_hat = torch.fft.rfft(x_hat)
    X_hat = torch.abs(X_hat)
    X_hat_db = 20 * torch.log10(X_hat + 1e-8)

    # compute frequency response of x
    X = torch.fft.rfft(x)
    X = torch.abs(X)
    X_db = 20 * torch.log10(X + 1e-8)

    # compute frequency axis
    freqs = torch.fft.fftfreq(x.shape[-1], d=1 / sample_rate)
    freqs = freqs[: X.shape[-1] - 1]  # take only positive frequencies
    X_db = X_db[:, : X.shape[-1] - 1]
    X_hat_db = X_hat_db[:, : X_hat.shape[-1] - 1]
    Y_db = Y_db[:, : Y.shape[-1] - 1]

    # smooth frequency response
    kernel_size = 255
    X_db = torch.nn.functional.avg_pool1d(
        X_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    X_hat_db = torch.nn.functional.avg_pool1d(
        X_hat_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    Y_db = torch.nn.functional.avg_pool1d(
        Y_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    # plot frequency response
    ax.plot(freqs, Y_db[0].squeeze().cpu().numpy(), label="input", alpha=0.7)
    ax.plot(freqs, X_hat_db[0].cpu().squeeze().numpy(), label="pred", alpha=0.7)
    ax.plot(
        freqs,
        X_db[0].squeeze().cpu().numpy(),
        label="target",
        alpha=0.7,
        c="gray",
        linestyle="--",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(100, 20000)
    ax.set_xscale("log")
    plt.legend()
    plt.grid(c="lightgray")
    plt.tight_layout()
    plt.savefig("outputs/auto_eq/response.png", dpi=300)


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
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=1,
        )
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        return x


class ParameterNetwork(torch.nn.Module):
    def __init__(self, num_control_params: int):
        super().__init__()
        self.num_control_params = num_control_params

        # we will use a simple TCN to estimate the parameters
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
            torch.nn.Linear(256, num_control_params),
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)  # aggregate over time
        p = torch.sigmoid(self.mlp(x))  # map to parmeters
        return p


class AudioEffectDataset(torch.nn.Module):
    def __init__(
        self,
        filepaths: List[str],
        processor: dasp_pytorch.Processor,
        length: int = 131072,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.length = length

        assert len(filepaths) > 0, "No files found."

        self.examples = []
        # create example of length `length` from each file
        for filepath in filepaths:
            md = torchaudio.info(filepath)
            if md.num_frames < length:
                continue
            num_examples = md.num_frames // length
            for n in range(num_examples):
                frame_offset = n * length
                self.examples.append((filepath, frame_offset))

        self.examples = self.examples * 100

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


def plot_loss(log_dir, loss_history: List[float]):
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.grid(c="lightgray")
    outfilepath = os.path.join(log_dir, "loss.png")
    plt.savefig(outfilepath, dpi=300)
    plt.close("all")


def train(
    root_dir: str,
    lr: float = 2e-3,
    batch_size: int = 16,
    num_epochs: int = 400,
    use_gpu: bool = False,
    log_dir: str = "outputs/auto_eq",
):
    os.makedirs(log_dir, exist_ok=True)  # create log directory
    equalizer = dasp_pytorch.ParametricEQ(44100)  # create instance of equalizer
    net = ParameterNetwork(equalizer.num_params)  # create parameter estimation network
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # create optimizer

    # create dataset
    filepaths = glob.glob(os.path.join(root_dir, "*.wav"))
    # split into train (80%) and validation sets (20%)
    train_filepaths = filepaths[: int(len(filepaths) * 0.8)]
    val_filepaths = filepaths[int(len(filepaths) * 0.8) :]

    train_filepaths = train_filepaths[0:1]
    val_filepaths = train_filepaths

    dataset = AudioEffectDataset(train_filepaths, equalizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # use a frequency domain loss function
    criterion = auraloss.freq.STFTLoss()
    # criterion = auraloss.time.SISDRLoss()

    # move to GPU if available
    if use_gpu:
        net.cuda()
        criterion.cuda()

    # main training loop
    epoch_loss_history = []
    for epoch in range(num_epochs):
        net.train()  # make sure network is in train mode
        # iterate over the dataset
        print("Epoch:", epoch + 1)
        batch_loss_history = []
        pbar = tqdm(dataloader)
        for batch, data in enumerate(pbar):
            x = data  # input

            # move to GPU if available
            if use_gpu:
                x = x.cuda()

            # 0. create corrupted example with random parameters
            with torch.no_grad():
                torch.manual_seed(42)
                param_tensor = torch.rand(x.shape[0], equalizer.num_params).type_as(x)
                y = equalizer.process_normalized(x, param_tensor)
                peaks, _ = torch.max(torch.abs(y), dim=-1)  # normalize to [-1,1]
                peaks = peaks.unsqueeze(-1)
                y /= peaks
                # set random gain between -24 dB and 0 dB
                gain_db = torch.rand(x.shape[0], 1, 1) * -24
                gain_db = gain_db.type_as(x)
                y *= 10 ** (gain_db / 20)

            # 1. estimate parameters with network
            # we show the network the "corrupted" signal
            p_hat = net(y)

            # 2. apply effect with estimated normalized parameters
            # we apply the effect to the "corrupted" signal to recover the original
            x_hat = equalizer.process_normalized(y, p_hat)

            # 3. compute loss between the original and recovered signal
            loss = criterion(x_hat, x)

            # 4. optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_history.append(loss.item())
            pbar.set_description(f"loss: {np.mean(batch_loss_history):.4e}")

        # plot loss and validate at the end of each epoch
        epoch_loss_history.append(np.mean(batch_loss_history))
        plot_loss(log_dir, epoch_loss_history)
        validate(
            epoch + 1,
            val_filepaths,
            net,
            equalizer,
            log_dir=log_dir,
            use_gpu=use_gpu,
        )


def validate(
    epoch: int,
    filepaths: List[str],
    net: torch.nn.Module,
    equalizer: dasp_pytorch.Processor,
    log_dir: str = "logs",
    use_gpu: bool = False,
):
    audio_log_dir = os.path.join(log_dir, "audio")
    os.makedirs(audio_log_dir, exist_ok=True)

    # evaluate the network
    net.eval()

    # use one of the validation files
    filepath = np.random.choice(filepaths)

    # load audio
    x, sr = torchaudio.load(filepath, backend="soundfile")
    x = x[:, :131072]

    if use_gpu:
        x = x.cuda()

    # apply random equalization (corrupt)
    with torch.no_grad():
        # random parameters on (0,1)
        param_tensor = torch.rand(1, equalizer.num_params).type_as(x)

        # apply effect with random parameters
        y = equalizer.process_normalized(x.unsqueeze(0), param_tensor)

        # normalize to [-1,1]
        peaks, _ = torch.max(torch.abs(y), dim=-1)
        peaks = peaks.unsqueeze(-1)
        y /= peaks

        # predict parameters with network to recover original
        p_hat = net(y)

        # apply effect with estimated normalized parameters
        x_hat = equalizer.process_normalized(y, p_hat).squeeze(0)

    # plot the responses
    plot_response(y.squeeze(0), x_hat, x)

    # save the results
    target_filename = f"epoch={epoch:03d}_target.wav"
    corrupt_filename = f"epoch={epoch:03d}_corrupt.wav"
    pred_filename = f"epoch={epoch:03d}_pred.wav"
    target_filepath = os.path.join(audio_log_dir, target_filename)
    corrupt_filepath = os.path.join(audio_log_dir, corrupt_filename)
    pred_filepath = os.path.join(audio_log_dir, pred_filename)
    torchaudio.save(target_filepath, x.cpu(), sr, backend="soundfile")
    torchaudio.save(corrupt_filepath, y.squeeze(0).cpu(), sr, backend="soundfile")
    torchaudio.save(pred_filepath, x_hat.cpu(), sr, backend="soundfile")


if __name__ == "__main__":
    train("data/", use_gpu=True)
