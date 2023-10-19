import torch
import torchaudio

from dasp_pytorch.functional import parametric_eq

# This is a good example of how to chain processors together to create a more complex effect.


class DistortionModel(Processor):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def process(self, x: torch.Tensor):
        """Apply the distortion effect to the input signal using the internal parameters.

        Args:
            x (torch.Tensor): Input signal with shape (batch_size, n_channels, n_samples)

        """
        batch_size, n_channels, n_samples = x.shape

        # apply the parametric equalizer
        y = parametric_eq(x, self.sample_rate, self.pre_filter_params)
        y = torch.tanh(y)  # apply non-linearity
        y = parametric_eq(y, self.sample_rate, self.post_filter_params)

        return y


if __name__ == "__main__":
    # load the input and output recordings
    x, sr = torchaudio.load()
    y, sr = torchaudio.load()
