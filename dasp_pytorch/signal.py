import torch
import numpy as np
import scipy.signal



def fft_freqz(b, a, n_fft: int = 512):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y


def octave_band_filterbank(num_taps: int, sample_rate: float):
    # create octave-spaced bandpass filters
    bands = [
        16,
        31.5,
        63,
        125,
        250,
        500,
        1000,
        2000,
        4000,
        5000,
        6000,
        7000,
        8000,
        10000,
        12000,
        14000,
        16000,
        18000,
    ]
    num_bands = len(bands) + 2
    filts = []  # storage for computed filter coefficients

    # lowest band is a lowpass
    filt = scipy.signal.firwin(
        num_taps,
        12,
        fs=sample_rate,
    )
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)

    for fc in bands:
        f_min = fc / np.sqrt(2)
        f_max = fc * np.sqrt(2)
        f_max = np.clip(f_max, a_min=0, a_max=(sample_rate / 2) * 0.999)
        filt = scipy.signal.firwin(
            num_taps,
            [f_min, f_max],
            fs=sample_rate,
            pass_zero=False,
        )
        filt = torch.from_numpy(filt.astype("float32"))
        filt = torch.flip(filt, dims=[0])
        filts.append(filt)

    # highest is a highpass
    filt = scipy.signal.firwin(num_taps, 18000, fs=sample_rate, pass_zero=False)
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)

    filts = torch.stack(filts, dim=0)  # stack coefficients into single filter
    filts = filts.unsqueeze(1)  # shape: num_bands x 1 x num_taps

    return filts


def lfilter_via_fsm(b, a, x):
    """Use the frequency sampling method to approximate an IIR filter.

    The filter will be applied along the final dimension of x.

    Args:
        b (torch.Tensor): Numerator coefficients with shape (bs, 2).
        a (torch.Tensor): Denominator coefficients with shape (bs, 2).
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # move coefficients to same device as x
    b = b.type_as(x)
    a = a.type_as(x)

    # compute complex response as ratio of polynomials
    H = fft_freqz(b, a, n_fft=n_fft)

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y


def one_pole_butter_lowpass(
    f_c: torch.Tensor,
    sample_rate: float,
):
    """

    This function is correct.

    """
    f_c = f_c.view(-1, 1)
    w_d = 2 * np.pi * (f_c / sample_rate)  # digital frequency in radians
    w_c = torch.tan(w_d / 2)  # apply pre-warping for analog frequency

    a0 = 1 + w_c
    a1 = w_c - 1
    b0 = w_c
    b1 = w_c

    b = torch.cat([b0, b1], dim=-1)
    a = torch.cat([a0, a1], dim=-1)

    # normalize
    b = b / a0
    a = a / a0

    print("ba", b.shape, a.shape)

    return b, a


def one_pole_filter(
    cutoff_hz: torch.Tensor,
    filter_type: str,
    sample_rate: float = 2.0,
):
    """Design a simple 1-pole highpass or lowpass IIR filter.

    Filtering is performed along the final dimension of the input tensor.

    Args:
        cutoff_hz (torch.Tensor): Lowpass filter centre frequency (normalized 0,1) with shape (bs). (Higher = more smoothing)
        filter_type (str): Specify either "highpass" or "lowpass". (Note this parameter cannot be optimize with gradient descent.)
        sample_rate (float): Sample rate of the input signal.

    Returns:
        b, a (torch.Tensor). Coefficient tensors with shape (bs, 2).
    """
    # setup shape for broadcasting
    bs = cutoff_hz.shape[0]
    cutoff_hz = cutoff_hz.view(bs, 1)
    nyquist = sample_rate // 2

    # compute coefficients
    if filter_type == "highpass":
        a1 = cutoff_hz / nyquist
    elif filter_type == "lowpass":
        a1 = -1 + (cutoff_hz / nyquist)
    else:
        raise ValueError(f"Invalid filter_type = {filter_type}.")

    print(a1)

    a0 = torch.ones(bs, 1).type_as(a1)

    b0 = 1 - torch.abs(a1)
    b1 = torch.zeros(bs, 1).type_as(a1)

    b = torch.cat([b0, b1], dim=1)
    a = torch.cat([a0, a1], dim=1)

    return b, a

