import os
import glob
import torch
import auraloss
import torchaudio
import numpy as np
import dasp_pytorch
from tqdm import tqdm

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
        self.blocks.append(TCNBlock(1, 16, 3, dilation=1))
        self.blocks.append(TCNBlock(16, 32, 3, dilation=2))
        self.blocks.append(TCNBlock(32, 64, 3, dilation=4))
        self.blocks.append(TCNBlock(64, 128, 3, dilation=8))
        self.blocks.append(TCNBlock(128, 128, 3, dilation=16))

        self.linear = torch.nn.Linear(128, num_control_params)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)  # aggregate over time
        p = torch.sigmoid(self.linear(x))  # map to parmeters
        return p


class AudioEffectDataset(torch.nn.Module):
    def __init__(
        self, root_dir: str, processor: dasp_pytorch.Processor, length: int = 131072
    ) -> None:
        super().__init__()
        self.processor = processor
        self.length = length
        # find all audio files in the root directory
        filepaths = glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True)

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
        x = x.unsqueeze(0)  # add batch dim

        # generate random parameters on (0,1)
        param_tensor = torch.rand(1, self.processor.num_params)

        # apply effect with random parameters
        with torch.no_grad():
            y = self.processor.process_normalized(x, param_tensor)

        # remove batch dim
        x = x.squeeze(0)
        y = y.squeeze(0)

        return x, y


def train(
    root_dir: str,
    lr: float = 1e-4,
    batch_size: int = 8,
    num_epochs: int = 100,
    use_gpu: bool = False,
):
    # create instance of compressor
    compressor = dasp_pytorch.Compressor(44100)

    # create parameter estimation network
    net = ParameterNetwork(compressor.num_params)

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # create dataset
    dataset = AudioEffectDataset(root_dir, compressor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # use a frequency domain loss function
    criterion = auraloss.freq.STFTLoss()

    # move to GPU if available
    if use_gpu:
        net.cuda()
        criterion.cuda()

    # main training loop
    for epoch in range(num_epochs):
        # iterate over the dataset
        print("Epoch:", epoch + 1)
        loss_history = []
        pbar = tqdm(dataloader)
        for batch, data in enumerate(pbar):
            x, y = data  # input, output

            # move to GPU if available
            if use_gpu:
                x = x.cuda()
                y = y.cuda()

            # 1. estimate parameters with network
            p_hat = net(y)

            # 2. apply effect with estimated normalized parameters
            y_hat = compressor.process_normalized(x, p_hat)

            # 3. compute loss
            loss = criterion(y_hat, y)

            # 4. optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            pbar.set_description(f"loss: {np.mean(loss_history):.4e}")


if __name__ == "__main__":
    train("data/", use_gpu=True)
