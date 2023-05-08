import math
import torch
import numpy as np
import scipy.signal


def fft_freqz(b, a, n_fft: int = 512):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def fft_sosfreqz(sos: torch.Tensor, n_fft: int = 512):
    """Compute the complex frequency response via FFT of cascade of biquads

    Args:
        sos (torch.Tensor): Second order filter sections with shape (bs, n_sections, 6)
        n_fft (int): FFT size. Default: 512
    Returns:
        H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    bs, n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order
    for section_idx in range(n_sections):
        b = sos[:, section_idx, :3]
        a = sos[:, section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y


def octave_band_filterbank(num_taps: int, sample_rate: float):
    # create octave-spaced bandpass filters
    bands = [
        31.5,
        63,
        125,
        250,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
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


def lfilter_via_fsm(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor = None):
    """Use the frequency sampling method to approximate an IIR filter.
    The filter will be applied along the final dimension of x.
    Args:
        x (torch.Tensor): Time domain signal with shape (bs, 1, timesteps)
        b (torch.Tensor): Numerator coefficients with shape (bs, N).
        a (torch.Tensor): Denominator coefficients with shape (bs, N).
    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, 1, timesteps)
    """
    bs, chs, seq_len = x.size()  # enforce shape
    assert chs == 1

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # move coefficients to same device as x
    b = b.type_as(x)

    if a is None:
        # directly compute FFT of numerator coefficients
        H = torch.fft.rfft(b, n_fft)
    else:
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


def sosfilt_via_fsm(sos: torch.Tensor, x: torch.Tensor):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    The filter will be applied along the final dimension of x.
    Args:
        sos (torch.Tensor): Tensor of coefficients with shape (bs, n_sections, 6).
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # compute complex response as ratio of polynomials
    H = fft_sosfreqz(sos, n_fft=n_fft)

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


def biquad(
    gain_db: torch.Tensor,
    cutoff_freq: torch.Tensor,
    q_factor: torch.Tensor,
    sample_rate: float,
    filter_type: str = "peaking",
):
    bs = gain_db.size(0)
    # reshape params
    gain_db = gain_db.view(bs, -1)
    cutoff_freq = cutoff_freq.view(bs, -1)
    q_factor = q_factor.view(bs, -1)

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)

    # normalize
    b = b.type_as(gain_db) / a0
    a = a.type_as(gain_db) / a0

    return b, a
