<div align="center">

<img src="docs/dasp-no-bg.png" width="250px">

# dasp
Differentiable audio signal processors in PyTorch

</div>


## Usage 

```
pip install dasp-pytorch
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

Dynamic range processors ballistics
```bibtex
@misc{2006.11239,
Author = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
Title = {Denoising Diffusion Probabilistic Models},
Year = {2020},
Eprint = {arXiv:2006.11239},
}
```

## Acknowledgements
