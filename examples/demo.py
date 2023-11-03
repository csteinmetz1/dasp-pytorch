import os
import torch
import torchaudio
from dasp_pytorch import (
    distortion,
    compressor,
    parametric_eq,
    noise_shaped_reverberation,
)

os.makedirs("outputs/demo", exist_ok=True)

if not os.path.exists("examples/demo/di_guitar.wav"):
    os.system(
        "wget csteinmetz1.github.io/sounds/assets/short_riff.wav -O outputs/demo/short_riff.wav"
    )

# load DI guitar sample
x, sr = torchaudio.load("outputs/demo/short_riff.wav", backend="soundfile")

# add batch dim
x = x.unsqueeze(0)

# process with chain of effects
y = parametric_eq(
    x,
    sr,
    low_shelf_gain_db=-12.0,
    low_shelf_cutoff_freq=1000,
    low_shelf_q_factor=0.5,
    band0_gain_db=0.0,
    band0_cutoff_freq=1000,
    band0_q_factor=0.5,
    band1_gain_db=0.0,
    band1_cutoff_freq=1000,
    band1_q_factor=0.5,
    band2_gain_db=0.0,
    band2_cutoff_freq=1000,
    band2_q_factor=0.5,
    band3_gain_db=0.0,
    band3_cutoff_freq=1000,
    band3_q_factor=0.5,
    high_shelf_gain_db=12.0,
    high_shelf_cutoff_freq=4000,
    high_shelf_q_factor=0.5,
)
print(y.shape)
y = compressor(
    y,
    sr,
    threshold_db=-12.0,
    ratio=4.0,
    attack_ms=10.0,
    release_ms=100.0,
    knee_db=12.0,
    makeup_gain_db=0.0,
)
print(y.shape)
y = distortion(y, sr, drive_db=16.0)
print(y.shape)
y = noise_shaped_reverberation(
    y,
    sr,
    band0_gain=0.8,
    band1_gain=0.4,
    band2_gain=0.2,
    band3_gain=0.5,
    band4_gain=0.5,
    band5_gain=0.5,
    band6_gain=0.5,
    band7_gain=0.6,
    band8_gain=0.7,
    band9_gain=0.8,
    band10_gain=0.9,
    band11_gain=1.0,
    band0_decay=0.5,
    band1_decay=0.5,
    band2_decay=0.5,
    band3_decay=0.5,
    band4_decay=0.5,
    band5_decay=0.5,
    band6_decay=0.5,
    band7_decay=0.5,
    band8_decay=0.5,
    band9_decay=0.5,
    band10_decay=0.5,
    band11_decay=0.5,
    mix=0.2,
)
print(y.shape)
y = y.squeeze(0)
print(y.shape)

torchaudio.save("outputs/demo/short_riff_output.wav", y, sr, backend="soundfile")
