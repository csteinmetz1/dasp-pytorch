import torch
import torchaudio
import dasp_pytorch

# In this example we will demonstrate the use of dasp_pytorch to reverse engineer
# a distortion effect. We will use a simple distortion effect with a single parameter.
# We will then use dasp_pytorch to estimate the parameter value from a target signal using gradient decent.
# This is not a realisitic example, since we know the true drive parameter value, but it demonstrates
# how to use gradient descent to estimate parameters.

# Load audio
x, sr = torchaudio.load("audio/short_riff.wav")

# create batch dim
# dasp expects (batch_size, n_channels, n_samples)
x = x.unsqueeze(0)

# apply some distortion with 16 dB drive
drive = torch.tensor([16.0])
y = dasp_pytorch.functional.distortion(x, drive)

# create a parameter to optimizer
drive_hat = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([drive_hat], lr=0.01)

# optimize the parameter
n_iters = 2500
for n in range(n_iters):
    # apply distortion with the estimated parameter
    y_hat = dasp_pytorch.functional.distortion(x, drive_hat)

    # compute distance between estimate and target
    loss = torch.nn.functional.mse_loss(y_hat, y)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}, drive: {drive_hat.item():.3f}"
    )
