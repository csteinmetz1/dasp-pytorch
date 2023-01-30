import torch
import scipy.signal

class ParametricEQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward():
        return
    
class Compressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward():
        return
    
class Expander(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward():
        return

class Overdrive(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward():
        return

class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        num_taps: int = 4097,
        num_samples: int = 65537,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_taps = num_taps
        self.num_samples = num_samples

        self.register_buffer("filts", filts)


    def forward(self, g:torch.Tensor, d: torch.Tensor):
        return x