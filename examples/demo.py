import os
import torch
import torchaudio
from dasp_pytorch import (
    distortion,
    compressor,
    parametric_eq,
    noise_shaped_reverberation,
)

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        torch.set_default_device("cuda")

    os.makedirs("outputs/demo", exist_ok=True)

    if not os.path.exists("outputs/demo/idmt-rock-input-varying-gain.wav"):
        os.system(
            "wget csteinmetz1.github.io/sounds/assets/amps/idmt-rock-input-varying-gain.wav -O outputs/demo/idmt-rock-input-varying-gain.wav"
        )

    # load DI guitar sample
    x, sr = torchaudio.load(
        "outputs/demo/idmt-rock-input-varying-gain.wav", backend="soundfile"
    )

    start_idx = int(49.0 * sr)
    end_idx = start_idx + 441000
    x = x[0:1, start_idx:end_idx]

    # add batch dim
    x = x.unsqueeze(0)

    if use_gpu:
        x = x.cuda()

    # process with chain of effects
    y = parametric_eq(
        x,
        sr,
        low_shelf_gain_db=torch.tensor([-12.0]),
        low_shelf_cutoff_freq=torch.tensor([1000]),
        low_shelf_q_factor=torch.tensor([0.5]),
        band0_gain_db=torch.tensor([0.0]),
        band0_cutoff_freq=torch.tensor([1000]),
        band0_q_factor=torch.tensor([0.5]),
        band1_gain_db=torch.tensor([0.0]),
        band1_cutoff_freq=torch.tensor([1000]),
        band1_q_factor=torch.tensor([0.5]),
        band2_gain_db=torch.tensor([0.0]),
        band2_cutoff_freq=torch.tensor([1000]),
        band2_q_factor=torch.tensor([0.5]),
        band3_gain_db=torch.tensor([0.0]),
        band3_cutoff_freq=torch.tensor([1000]),
        band3_q_factor=torch.tensor([0.5]),
        high_shelf_gain_db=torch.tensor([12.0]),
        high_shelf_cutoff_freq=torch.tensor([4000]),
        high_shelf_q_factor=torch.tensor([0.5]),
    )
    y = compressor(
        y,
        sr,
        threshold_db=torch.tensor([-12.0]),
        ratio=torch.tensor([4.0]),
        attack_ms=torch.tensor([10.0]),
        release_ms=torch.tensor([100.0]),
        knee_db=torch.tensor([12.0]),
        makeup_gain_db=torch.tensor([0.0]),
    )

    y = distortion(y, sr, drive_db=torch.tensor([42.0]))

    y = parametric_eq(
        y,
        sr,
        low_shelf_gain_db=torch.tensor([0.0]),
        low_shelf_cutoff_freq=torch.tensor([1000]),
        low_shelf_q_factor=torch.tensor([0.5]),
        band0_gain_db=torch.tensor([0.0]),
        band0_cutoff_freq=torch.tensor([1000]),
        band0_q_factor=torch.tensor([0.5]),
        band1_gain_db=torch.tensor([0.0]),
        band1_cutoff_freq=torch.tensor([1000]),
        band1_q_factor=torch.tensor([0.5]),
        band2_gain_db=torch.tensor([0.0]),
        band2_cutoff_freq=torch.tensor([1000]),
        band2_q_factor=torch.tensor([0.5]),
        band3_gain_db=torch.tensor([0.0]),
        band3_cutoff_freq=torch.tensor([1000]),
        band3_q_factor=torch.tensor([0.5]),
        high_shelf_gain_db=torch.tensor([-18.0]),
        high_shelf_cutoff_freq=torch.tensor([4000]),
        high_shelf_q_factor=torch.tensor([0.5]),
    )

    y = noise_shaped_reverberation(
        y,
        sr,
        band0_gain=torch.tensor([0.8]),
        band1_gain=torch.tensor([0.4]),
        band2_gain=torch.tensor([0.2]),
        band3_gain=torch.tensor([0.5]),
        band4_gain=torch.tensor([0.5]),
        band5_gain=torch.tensor([0.5]),
        band6_gain=torch.tensor([0.5]),
        band7_gain=torch.tensor([0.6]),
        band8_gain=torch.tensor([0.7]),
        band9_gain=torch.tensor([0.8]),
        band10_gain=torch.tensor([0.9]),
        band11_gain=torch.tensor([1.0]),
        band0_decay=torch.tensor([0.5]),
        band1_decay=torch.tensor([0.5]),
        band2_decay=torch.tensor([0.5]),
        band3_decay=torch.tensor([0.5]),
        band4_decay=torch.tensor([0.5]),
        band5_decay=torch.tensor([0.5]),
        band6_decay=torch.tensor([0.5]),
        band7_decay=torch.tensor([0.5]),
        band8_decay=torch.tensor([0.5]),
        band9_decay=torch.tensor([0.5]),
        band10_decay=torch.tensor([0.5]),
        band11_decay=torch.tensor([0.5]),
        mix=torch.tensor([0.15]),
    )

    y = y.squeeze(0)
    x = x.squeeze(0)
    print(y.shape)

    y = y / torch.max(torch.abs(y))
    x = x / torch.max(torch.abs(x))

    torchaudio.save(
        "outputs/demo/idmt-rock-input-varying-input.wav",
        x.cpu(),
        sr,
        backend="soundfile",
    )

    torchaudio.save(
        "outputs/demo/idmt-rock-input-varying-output.wav",
        y.cpu(),
        sr,
        backend="soundfile",
    )
