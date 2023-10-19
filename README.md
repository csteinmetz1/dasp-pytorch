<div align="center">

<img src="docs/dasp-no-bg.png" width="200px">

# dasp
`dasp-pytorch` is a Python library for constructing differentiable audio processors using PyTorch. These differentiable processors can be used standalone or within the computation graph of neural networks. We provide purely functional interfaces for all processors that enables ease-of-use and portability across projects. 

</div>


## Installation 

```
pip install dasp-pytorch
```

## Examples

Unless oterhwise stated, all effect functions expect 3-dim tensors with shape `(batch_size, num_channels, num_samples)` as input and output. Using an effect in your computation graph is as simple as calling the function with the input tensor as argument. 

### Quickstart

Here is a minimal example to demonstrate reverse engineering the drive value of a simple distortion effect using gradient descent. 

```python
import torch
import torchaudio
import dasp_pytorch

# Load audio
x, sr = torchaudio.load("audio/short_riff.wav")

# create batch dim
# (batch_size, n_channels, n_samples)
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
```

For the remaining examples we will use the [GuitarSet](https://guitarset.weebly.com/) dataset. 
You can download the data using the following commands:
```bash
mkdir data
wget https://zenodo.org/records/3371780/files/audio_mono-mic.zip
unzip audio_mono-mic.zip
rm audio_mono-mic.zip
```

## Audio Processors

<table>
    <tr>
        <th>Audio Processor</th>
        <th>Interface</th>
        <th>Reference</th>
    </tr>
    <tr>
        <td>Simple Distortion</td>
        <td><code>simple_distortion()</code></td>
        <td></a></td>
    </tr>
    <tr>
        <td>Advanced Distortion</td>
        <td><code>advanced_distortion()</code></td>
        <td></a></td>
    </tr>    
    <tr>
        <td>Parametric Equalizer</td>
        <td><code>parametric_eq()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Graphic Equalizer</td>
        <td><code>graphic_eq()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Dynamic range compressor</td>
        <td><code>compressor()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Dynamic range expander</td>
        <td><code>expander()</code></td>
        <td></td>
    </tr>    
    <tr>
        <td>Reverberation</td>
        <td><code>reverberation()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Stereo Widener</td>
        <td><code>stereo_widener()</code></td>
        <td></td>
    </tr>
    <tr>
        <td>Stereo Panner</td>
        <td><code>stereo_panner()</code></td>
        <td></td>
    </tr>
</table>

## Examples

## Citations

If you use this library consider citing these papers:

Differnetiable parametric EQ and dynamic range compressor
```bibtex
@article{steinmetz2022style,
  title={Style transfer of audio effects with differentiable signal processing},
  author={Steinmetz, Christian J and Bryan, Nicholas J and Reiss, Joshua D},
  journal={arXiv preprint arXiv:2207.08759},
  year={2022}
}
```

Differentiable artificial reveberation with frequency-band noise shaping
```bibtex
@inproceedings{steinmetz2021filtered,
  title={Filtered noise shaping for time domain room impulse 
         response estimation from reverberant speech},
  author={Steinmetz, Christian J and Ithapu, Vamsi Krishna and Calamia, Paul},
  booktitle={WASPAA},
  year={2021},
  organization={IEEE}
}
```

Differnetiable IIR filters
```bibtex
@inproceedings{nercessian2020neural,
  title={Neural parametric equalizer matching using differentiable biquads},
  author={Nercessian, Shahan},
  booktitle={DAFx},
  year={2020}
}
```

```bibtex
@inproceedings{colonel2022direct,
  title={Direct design of biquad filter cascades with deep learning 
          by sampling random polynomials},
  author={Colonel, Joseph T and Steinmetz, Christian J and 
          Michelen, Marcus and Reiss, Joshua D},
  booktitle={ICASSP},
  year={2022},
  organization={IEEE}
```

## Acknowledgements
