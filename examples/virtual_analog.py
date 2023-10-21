import torch
import auraloss
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from dasp_pytorch import ParametricEQ

# This is a good example of how to chain processors together to create a more complex effect.
# In this case, we create our own torch.nn.Module using the processor functions.


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, src_filepath: str, target_filepath: str, length: int = 65536):
        self.length = length

        # load files into RAM
        src, sr = torchaudio.load(src_filepath, backend="soundfile")
        target, sr = torchaudio.load(target_filepath, backend="soundfile")

        self.examples = []
        # create examples by slicing the audio into segments of length `length`
        num_segments = src.shape[-1] // length
        for n in range(num_segments):
            start_idx = n * length
            stop_idx = start_idx + length
            src_segment = src[0:1, start_idx:stop_idx]
            target_segment = target[0:1, start_idx:stop_idx]
            self.examples.append((src_segment, target_segment))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        src_segment, target_segment = self.examples[idx]
        return src_segment, target_segment


class DistortionModel(torch.nn.Module):
    def __init__(self, sample_rate: int):
        """Create a distortion model using a pre and post filter.

        This is based on the use of filtering and non-linearity to
        model distortion based effects. We will optimize the parameters of the
        pre and post filters to create a distortion effect.

        """
        super().__init__()
        self.sample_rate = sample_rate
        # the equalizer object does not contain internal state
        # so we can use the same equalizer for the pre and post filters
        self.equalizer = ParametricEQ(sample_rate)

        # create parameters for the pre and post filters (these will be optimized)
        self.pre_filter_params = torch.nn.Parameter(
            torch.rand(1, self.equalizer.num_params)
        )
        self.dc_offset = torch.nn.Parameter(torch.tensor(0.0))
        self.mlp_nonlinearity = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.post_filter_params = torch.nn.Parameter(
            torch.rand(1, self.equalizer.num_params)
        )

    def forward(self, x: torch.Tensor):
        """Apply the distortion effect to the input signal using the internal parameters.

        Args:
            x (torch.Tensor): Input signal with shape (batch_size, n_channels, n_samples)

        """
        batch_size, n_channels, n_samples = x.shape

        # apply pre-filter
        y = self.equalizer.process_normalized(
            x,
            torch.sigmoid(self.pre_filter_params),
        )
        # apply dc offset
        y = y + self.dc_offset
        # y = torch.tanh(y)  # apply non-linearity
        y = self.mlp_nonlinearity(y.permute(0, 2, 1)).permute(0, 2, 1)
        # apply post-filter
        y = self.equalizer.process_normalized(
            y,
            torch.sigmoid(self.post_filter_params),
        )

        return y


def plot_waveforms(
    src: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    num_segments: int = 6,
    segment_length: int = 4096,
):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8, 6))
    axs = np.reshape(axs, -1)

    for n in range(num_segments):
        start_idx = n * segment_length
        stop_idx = start_idx + segment_length
        # axs[n].plot(
        #    src[0, start_idx:stop_idx].numpy(),
        #    label="src",
        #    alpha=0.7,
        # )
        axs[n].plot(
            target[0, start_idx:stop_idx].numpy(),
            label="target",
            alpha=0.7,
        )
        axs[n].plot(
            pred[0, start_idx:stop_idx].numpy(),
            label="pred",
            alpha=0.7,
        )
        axs[n].grid(c="lightgray")
        if n == 0:
            axs[n].legend()

    plt.savefig("virtual_analog.png", dpi=300)
    plt.close("all")


def plot_system(model: torch.nn.Module):
    # plot response of pre and post filters
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 1e-2,
    epochs: int = 500,
    use_gpu: bool = False,
):
    # loss function in frequency domain
    freq_criterion = auraloss.freq.STFTLoss()
    time_critertion = torch.nn.MSELoss()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if use_gpu:
        model.cuda()
        freq_criterion.cuda()

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            src, target = batch

            if use_gpu:
                src = src.cuda()
                target = target.cuda()

            # apply distortion model with current parameters
            y_hat = model(src)

            # compute loss between estimate and target
            freq_loss = freq_criterion(y_hat, target)
            time_loss = time_critertion(y_hat, target)
            loss = freq_loss + (100 * time_loss)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Epoch: {epoch+1}/{epochs}  Loss: {loss.item():.3f}")

        # plot the results
        # plot_system(model)


if __name__ == "__main__":
    # construct dataset
    scr_filepath = "audio/va/clean.wav"
    target_filepath = "audio/va/sparkle-combo-crunch.wav"
    dataset = FileDataset(scr_filepath, target_filepath)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # construct grey-box distortion model
    model = DistortionModel(sample_rate=44100)

    # train the model!
    train(model, dataloader, use_gpu=True)

    # run the final model and plot the results
    model.cpu()
    model.eval()

    src, sr = torchaudio.load(scr_filepath, backend="soundfile")
    target, sr = torchaudio.load(target_filepath, backend="soundfile")
    src = src[0:1, :]
    target = target[0:1, :]

    with torch.no_grad():
        y_hat = model(src.unsqueeze(0)).squeeze(0)

    plot_waveforms(src, target, y_hat)

    # save audio
    torchaudio.save(
        "audio/va/sparkle-combo-crunch-pred.wav",
        y_hat,
        sr,
        backend="soundfile",
    )
