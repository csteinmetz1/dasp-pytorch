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
            torch.nn.ReLu(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLu(),
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

        # predict parameters with network to recover original
        p_hat = net(y)

        # apply effect with estimated normalized parameters
        x_hat = equalizer.process_normalized(y, p_hat).squeeze(0)

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


def train(
    root_dir: str,
    lr: float = 1e-3,
    batch_size: int = 16,
    num_epochs: int = 400,
    use_gpu: bool = False,
    log_dir: str = "logs/auto-eq",
):
    # create log directory
    os.makedirs(log_dir, exist_ok=True)

    # create instance of equalizer
    equalizer = dasp_pytorch.ParametricEQ(44100)

    # create parameter estimation network
    net = ParameterNetwork(equalizer.num_params)

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # create dataset
    filepaths = glob.glob(os.path.join(root_dir, "*.wav"))
    # split into train (80%) and validation sets (20%)
    train_filepaths = filepaths[: int(len(filepaths) * 0.8)]
    val_filepaths = filepaths[int(len(filepaths) * 0.8) :]
    dataset = AudioEffectDataset(train_filepaths, equalizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # use a frequency domain loss function
    criterion = auraloss.freq.STFTLoss()

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
                param_tensor = torch.rand(x.shape[0], equalizer.num_params).type_as(x)
                y = equalizer.process_normalized(x, param_tensor)

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


if __name__ == "__main__":
    train("data/", use_gpu=True)
