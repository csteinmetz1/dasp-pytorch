import torch
from dasp_pytorch.functional import (
    gain,
    distortion,
    compressor,
    parametric_eq,
    noise_shaped_reverberation,
)

from typing import Dict, List


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


class Processor:
    def __init__(self):
        pass

    def process_normalized(self, x: torch.Tensor, param_tensor: torch.Tensor):
        """Run the processor using normalized parameters on (0,1) in a tensor.

        This function assumes that the parameters in the tensor are in the same
        order as the parameter defined in the processor.

        Args:
            x (torch.Tensor): Input audio tensor (batch, channels, samples)
            param_tensor (torch.Tensor): Tensor of parameters on (0,1) (batch, num_params)

        Returns:
            torch.Tensor: Output audio tensor
        """
        # extract parameters from tensor
        param_dict = self.extract_param_dict(param_tensor)

        # denormalize parameters to full range
        denorm_param_dict = self.denormalize_param_dict(param_dict)

        # now process audio with denormalized parameters
        y = self.process_fn(
            x,
            self.sample_rate,
            **denorm_param_dict,
        )

        return y

    def process(self, x: torch.Tensor, *args):
        return self.process_fn(x, *args)

    def extract_param_dict(self, param_tensor: torch.Tensor):
        # check the number of parameters in tensor matches the number of processor parameters
        if param_tensor.shape[1] != len(self.param_ranges):
            raise ValueError(
                f"Parameter tensor has {param_tensor.shape[1]} parameters, but processor has {len(self.param_ranges)} parameters."
            )

        # extract parameters from tensor
        param_dict = {}
        for param_idx, param_name in enumerate(self.param_ranges.keys()):
            param_dict[param_name] = param_tensor[:, param_idx]

        return param_dict

    def denormalize_param_dict(self, param_dict: dict):
        """Given parameters on (0,1) restore them to the ranges expected by the processor.

        Args:
            param_dict (dict): Dictionary of parameter tensors on (0,1).

        Returns:
            dict: Dictionary of parameter tensors on their full range.

        """
        denorm_param_dict = {}
        for param_name, param_tensor in param_dict.items():
            # check for out of range parameters
            if param_tensor.min() < 0 or param_tensor.max() > 1:
                raise ValueError(f"Parameter {param_name} of is out of range.")
            param_val_denorm = denormalize(
                param_tensor,
                self.param_ranges[param_name][1],
                self.param_ranges[param_name][0],
            )
            denorm_param_dict[param_name] = param_val_denorm
        return denorm_param_dict


class Gain(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_gain_db: float = -24.0,
        max_gain_db: float = 24.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = gain
        self.param_ranges = {
            "gain_db": (min_gain_db, max_gain_db),
        }
        self.num_params = len(self.param_ranges)


class Distortion(Processor):
    def __init__(
        self,
        min_gain_db: float = 0.0,
        max_gain_db: float = 24.0,
    ):
        super().__init__()
        self.process_fn = distortion
        self.param_ranges = {
            "gain_db": (min_gain_db, max_gain_db),
        }
        self.num_params = len(self.param_ranges)


class ParametricEQ(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_gain_db: float = -20.0,
        max_gain_db: float = 20.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 6.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = parametric_eq
        self.param_ranges = {
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20, 2000),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (80, 2000),
            "band0_q_factor": (min_q_factor, max_q_factor),
            "band1_gain_db": (min_gain_db, max_gain_db),
            "band1_cutoff_freq": (2000, 8000),
            "band1_q_factor": (min_q_factor, max_q_factor),
            "band2_gain_db": (min_gain_db, max_gain_db),
            "band2_cutoff_freq": (8000, 12000),
            "band2_q_factor": (min_q_factor, max_q_factor),
            "band3_gain_db": (min_gain_db, max_gain_db),
            "band3_cutoff_freq": (12000, (sample_rate // 2) - 1000),
            "band3_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000, (sample_rate // 2) - 1000),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
        }
        self.num_params = len(self.param_ranges)


class Compressor(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_threshold_db: float = -60.0,
        max_threshold_db: float = 0.0,
        min_ratio: float = 1.0,
        max_ratio: float = 20.0,
        min_attack_ms: float = 5.0,
        max_attack_ms: float = 100.0,
        min_release_ms: float = 5.0,
        max_release_ms: float = 100.0,
        min_knee_db: float = 0.0,
        max_knee_db: float = 12.0,
        min_makeup_gain_db: float = 0.0,
        max_makeup_gain_db: float = 12.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = compressor
        self.param_ranges = {
            "threshold_db": (min_threshold_db, max_threshold_db),
            "ratio": (min_ratio, max_ratio),
            "attack_ms": (min_attack_ms, max_attack_ms),
            "release_ms": (min_release_ms, max_release_ms),
            "knee_db": (min_knee_db, max_knee_db),
            "makeup_gain_db": (min_makeup_gain_db, max_makeup_gain_db),
        }
        self.num_params = len(self.param_ranges)


class NoiseShapedReverb(Processor):
    def __init__(
        self,
        sample_rate,
        min_band_gain: float = 0.0,
        max_band_gain: float = 1.0,
        min_band_decay: float = 0.0,
        max_band_decay: float = 1.0,
        min_mix: float = 0.0,
        max_mix: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = noise_shaped_reverberation
        self.param_ranges = {
            "band0_gain": (min_band_gain, max_band_gain),
            "band1_gain": (min_band_gain, max_band_gain),
            "band2_gain": (min_band_gain, max_band_gain),
            "band3_gain": (min_band_gain, max_band_gain),
            "band4_gain": (min_band_gain, max_band_gain),
            "band5_gain": (min_band_gain, max_band_gain),
            "band6_gain": (min_band_gain, max_band_gain),
            "band7_gain": (min_band_gain, max_band_gain),
            "band8_gain": (min_band_gain, max_band_gain),
            "band9_gain": (min_band_gain, max_band_gain),
            "band10_gain": (min_band_gain, max_band_gain),
            "band11_gain": (min_band_gain, max_band_gain),
            "band0_decay": (min_band_decay, max_band_decay),
            "band1_decay": (min_band_decay, max_band_decay),
            "band2_decay": (min_band_decay, max_band_decay),
            "band3_decay": (min_band_decay, max_band_decay),
            "band4_decay": (min_band_decay, max_band_decay),
            "band5_decay": (min_band_decay, max_band_decay),
            "band6_decay": (min_band_decay, max_band_decay),
            "band7_decay": (min_band_decay, max_band_decay),
            "band8_decay": (min_band_decay, max_band_decay),
            "band9_decay": (min_band_decay, max_band_decay),
            "band10_decay": (min_band_decay, max_band_decay),
            "band11_decay": (min_band_decay, max_band_decay),
            "mix": (min_mix, max_mix),
        }
        self.num_params = len(self.param_ranges)
