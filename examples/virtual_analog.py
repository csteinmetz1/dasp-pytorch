import os
import torch
import auraloss
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from dasp_pytorch import ParametricEQ

# This is a good example of how to chain processors together to create a more complex effect.
# In this case, we create our own torch.nn.Module using the processor functions.


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
        start_idx = np.random.randint(0, src.shape[-1] - segment_length)
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

    plt.savefig("outputs/virtual_analog/audio.png", dpi=300)
    plt.close("all")


def plot_system(model: torch.nn.Module):
    # plot response of pre and post filters
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=False)

    # plot pre-filter
    axs[0].set_aspect("equal", "box")

    # plot nonlinearity
    x = torch.linspace(-3, 3, 1000)
    with torch.no_grad():
        y = model.mlp_nonlinearity(x.unsqueeze(1)).squeeze(1)

    axs[1].plot(x, y)
    axs[1].grid(c="lightgray")
    axs[1].set_title("Nonlinearity")
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-3, 3)
    axs[1].set_aspect("equal", "box")

    # plot post-filter
    axs[2].set_aspect("equal", "box")

    plt.savefig("outputs/virtual_analog/system.png", dpi=300)


def plot_loss(loss_history: list):
    plt.figure()
    plt.plot(loss_history)
    plt.grid(c="lightgray")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("outputs/virtual_analog/loss.png", dpi=300)


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

        self.input_gain = torch.nn.Parameter(torch.tensor(0.0))
        # create parameters for the pre and post filters (these will be optimized)
        self.pre_filter_params = torch.nn.Parameter(
            torch.rand(1, self.equalizer.num_params)
        )
        self.dc_offset = torch.nn.Parameter(torch.tensor(0.0))
        self.mlp_nonlinearity = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 1),
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

        # apply input gain
        x = x * torch.sigmoid(self.input_gain) * 24.0

        # apply pre-filter
        y = self.equalizer.process_normalized(
            x,
            torch.sigmoid(self.pre_filter_params),
        )
        # apply dc offset
        # y = y + self.dc_offset
        # y = torch.tanh(y)  # apply non-linearity
        y = self.mlp_nonlinearity(y.permute(0, 2, 1)).permute(0, 2, 1)
        # apply post-filter
        y = self.equalizer.process_normalized(
            y,
            torch.sigmoid(self.post_filter_params),
        )

        return y


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sample_rate: int,
    lr: float = 1e-3,
    epochs: int = 1000,
    use_gpu: bool = False,
):
    # loss function in frequency domain
    fd_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[128, 256, 512, 1024, 2048, 4096, 8192],
        hop_sizes=[64, 128, 256, 512, 1024, 2048, 4096],
        win_lengths=[128, 256, 512, 1024, 2048, 4096, 8192],
        w_sc=0.0,
        w_phs=0.0,
        w_lin_mag=1.0,
        w_log_mag=1.0,
        perceptual_weighting=True,
        sample_rate=sample_rate,
    )
    td_loss_fn = torch.nn.MSELoss()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if use_gpu:
        model.cuda()
        fd_loss_fn.cuda()

    epoch_loss_history = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        loss_history = []
        for batch in pbar:
            src, target = batch

            if use_gpu:
                src = src.cuda()
                target = target.cuda()

            # apply distortion model with current parameters
            y_hat = model(src)

            # compute loss between estimate and target
            freq_loss = fd_loss_fn(y_hat, target)
            time_loss = td_loss_fn(y_hat, target)
            loss = freq_loss + (time_loss * 100.0)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            pbar.set_description(
                f"Epoch: {epoch+1}/{epochs}  Loss: {np.mean(loss_history):.3e}"
            )
        # average loss over epoch
        epoch_loss_history.append(np.mean(loss_history))

    return epoch_loss_history


if __name__ == "__main__":
    # construct dataset
    scr_filepath = "audio/va/6amps/idmt-rock-input-varying-gain.wav"
    target_filepath = "audio/va/6amps/idmt-rock-high-gain1-brit-8000.wav"
    dataset = FileDataset(scr_filepath, target_filepath, length=32768)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # directory for outputs
    os.makedirs("outputs/virtual_analog", exist_ok=True)

    # construct grey-box distortion model
    model = DistortionModel(sample_rate=44100)

    # train the model!
    loss_history = train(model, dataloader, sample_rate=44100, use_gpu=True)

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
    plot_system(model)
    plot_loss(loss_history)

    # save audio
    filename = os.path.basename(target_filepath).replace(".wav", "")
    torchaudio.save(
        f"audio/va/6amps/{filename}-pred.wav",
        y_hat,
        sr,
        backend="soundfile",
    )
