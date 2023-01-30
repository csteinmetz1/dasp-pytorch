import torch

import dasp_pytorch.signal as signal


def gain(x: torch.Tensor, gain_db: torch.Tensor):
    bs, chs, seq_len = x.size()

    # convert gain from dB to linear
    gain_lin = 10 ** (gain_db / 20.0)
    return x * gain_lin


def graphic_eq():
    # 20 bands (octave)
    return


def distortion():
    """

    [1] Colonel, Joseph T., Marco Comunità, and Joshua Reiss.
        "Reverse Engineering Memoryless Distortion Effects with Differentiable Waveshapers."
        153rd Convention of the Audio Engineering Society. Audio Engineering Society, 2022.

    [2]
    """
    return


def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    low_shelf_gain_dB: torch.Tensor,
    low_shelf_cutoff_freq: torch.Tensor,
    low_shelf_q_factor: torch.Tensor,
    first_band_gain_dB: torch.Tensor,
    first_band_cutoff_freq: torch.Tensor,
    first_band_q_factor: torch.Tensor,
    second_band_gain_dB: torch.Tensor,
    second_band_cutoff_freq: torch.Tensor,
    second_band_q_factor: torch.Tensor,
    third_band_gain_dB: torch.Tensor,
    third_band_cutoff_freq: torch.Tensor,
    third_band_q_factor: torch.Tensor,
    fourth_band_gain_dB: torch.Tensor,
    fourth_band_cutoff_freq: torch.Tensor,
    fourth_band_q_factor: torch.Tensor,
    high_shelf_gain_dB: torch.Tensor,
    high_shelf_cutoff_freq: torch.Tensor,
    high_shelf_q_factor: torch.Tensor,
):
    """Six-band Parametric Equalizer.



    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    [1] Välimäki, Vesa, and Joshua D. Reiss.
        "All about audio equalization: Solutions and frontiers."
        Applied Sciences 6.5 (2016): 129.

    [2] Nercessian, Shahan.
        "Neural parametric equalizer matching using differentiable biquads."
        Proc. Int. Conf. Digital Audio Effects (eDAFx-20). 2020.

    [3] Steinmetz, Christian J., Nicholas J. Bryan, and Joshua D. Reiss.
        "Style Transfer of Audio Effects with Differentiable Signal Processing."
        Journal of the Audio Engineering Society. Vol. 70, Issue 9, 2022, pp. 708-721.

    Args:
        x (torch.Tensor): 1d signal.
        sample_rate (float):
    """
    a_s, b_s = [], []

    # -------- design low-shelf filter --------
    b, a = signal.biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- design first-band peaking filter --------
    b, a = signal.biqaud(
        first_band_gain_dB,
        first_band_cutoff_freq,
        first_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- design second-band peaking filter --------
    b, a = signal.biqaud(
        second_band_gain_dB,
        second_band_cutoff_freq,
        second_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- design third-band peaking filter --------
    b, a = signal.biqaud(
        third_band_gain_dB,
        third_band_cutoff_freq,
        third_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- design fourth-band peaking filter --------
    b, a = signal.biqaud(
        fourth_band_gain_dB,
        fourth_band_cutoff_freq,
        fourth_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- design high-shelf filter --------
    b, a = signal.biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    b_s.append(b)
    a_s.append(a)

    x_filtered = signal.approx_iir_filter_cascade(b_s, a_s, x)

    return x_filtered


def compressor(
    x: torch.Tensor,
    sample_rate: float,
    threshold_db: torch.Tensor,
    ratio: torch.Tensor,
    attack_ms: torch.Tensor,
    release_ms: torch.Tensor,
    knee_db: torch.Tensor,
    makeup_gain_db: torch.Tensor,
    eps: float = 1e-8,
):
    """Dynamic range compressor.

    This compressor is based on the standard feedforward digital compressor design [1].
    To make the implementation differentiable the the simple compressor proposed in [2] is used.
    However, the original design utilized a single time constant for the attack and release
    ballisitics in order to parallelize the branching recursive smoothing filter.
    This implementation builds on this using approximate ballisitics similar to those
    introduced in [3]. This involves applying two 1-pole low pass filters to the gain reduction curve,
    each with different time constants. The final smoothed gain reduction curve is then computed
    by combining these two smoothed curves, setting only one to be active at a time based on
    the level of the gain reduction curve in relation to the threshold.

    [1] Giannoulis, Dimitrios, Michael Massberg, and Joshua D. Reiss.
        "Digital Dynamic Range Compressor Design - A Tutorial and Analysis."
        Journal of Audio Engineering Society. Vol. 60, Issue 6, 2012, pp. 399-408.

    [2] Steinmetz, Christian J., Nicholas J. Bryan, and Joshua D. Reiss.
        "Style Transfer of Audio Effects with Differentiable Signal Processing."
        Journal of the Audio Engineering Society. Vol. 70, Issue 9, 2022, pp. 708-721.

    [3] Colonel, Joseph, and Joshua D. Reiss.
        "Approximating Ballistics in a Differentiable Dynamic Range Compressor."
        153rd Convention of the Audio Engineering Society. Audio Engineering Society, 2022.

    Args:
        x (torch.Tensor): Audio tensor with shape (bs, chs, seq_len).
        sample_rate (float): Audio sampling rate.
        threshold_db (torch.Tensor): Threshold at which to begin gain reduction.
        ratio (torch>Tensor): Amount to reduce gain as a function of the distance above threshold.
        attack_ms (torch.Tensor): Attack time in milliseconds.
        release_ms (torch.Tensor): Release time in milliseconds.
        knee_db (torch.Tensor): Softness of the knee. Higher = softer. (Must be positive)
        makeup_gain_db (torch.Tensor): Apply gain after compression to restore signal level.
        eps (float): Epsilon value for numerical stability. Default: 1e-8

    Returns:
        y (torch.Tensor): Compressed signal.
    """
    bs, chs, seq_len = x.size()  # check shape

    threshold = threshold.squeeze()
    ratio = ratio.squeeze()
    attack_time = attack_time.squeeze()
    makeup_gain_dB = makeup_gain_dB.squeeze()

    x_db = 20 * torch.log10(torch.abs(x).clamp(eps))

    # compute time constants
    normalized_attack_time = sample_rate * (attack_ms / 1e3)
    normalized_release_time = sample_rate * (release_ms / 1e3)
    constant = torch.tensor([9.0]).type_as(attack_ms)
    alpha_A = torch.exp(-torch.log(constant) / normalized_attack_time)
    alpha_R = torch.exp(-torch.log(constant) / normalized_release_time)

    # static characteristic with soft knee
    x_sc = x_db.clone()

    # when signal is less than (T - W/2) leave as x_db

    # when signal is at the threshold engage knee
    idx1 = x_db >= (threshold_db - (knee_db / 2))
    idx2 = x_db <= (threshold_db + (knee_db / 2))
    idx = torch.logical_and(idx1, idx2)
    x_sc[idx] = x_db + ((1 / ratio) - 1)

    # when signal is above threshold linear response
    idx = x_db > (threshold_db + (knee_db / 2))
    x_sc[idx] = threshold_db + ((x_db[idx] - threshold_db) / ratio)

    # output of gain computer
    g_c = x_sc - x_db

    # attack smoothing
    g_s_attack = signal.one_pole_lowpass(g_c, alpha_A)
    attack_mask = (x_db > threshold_db).float()
    g_s_attack_masked = g_s_attack * attack_mask

    # release smoothing
    g_s_release = signal.one_pole_lowpass(g_c, alpha_R)
    release_mask = (x_db <= threshold_db).float()
    g_s_release_masked = g_s_release * release_mask

    # combine both signals
    g_s = g_s_attack_masked + g_s_release_masked

    # add makeup gain in db
    g_s = g_s + makeup_gain_db

    # convert dB gains back to linear
    g_lin = 10 ** (g_s / 20.0)

    # apply time-varying gain and makeup gain
    y = x * g_lin

    return y


def expander():

    # static characteristics with soft-knee
    x_sc = x_db.clone()

    # below knee
    idx = x_db < (threshold_db - knee_db / 2)
    x_sc_below = threshold_db + (x_db - threshold_db) * ratio
    x_sc[idx] = x_sc_below[idx]

    # at knee
    idx_1 = x_db <= (threshold_db + (knee_db / 2))
    idx_2 = x_db >= (threshold_db - (knee_db / 2))
    idx = torch.logical_and(idx_1, idx_2)
    x_sc_at = x_db + ((1 - ratio) * ((x_db - threshold_db - (knee_db / 2)) ** 2)) / (
        2 * knee_db
    )
    x_sc[idx] = x_sc_at[idx]

    # above knee signal remains the same

    # gain
    g_c = x_sc - x_db

    return x


def reverb():
    """Artificial reveberation.

    This differentiable artifical reveberation model is based on the idea of
    filtered nosie shaping, similar to that proposed in [1]. This approach leverages
    the well known idea that a room impulse response (RIR) can be modeled as the direct sound,
    a set of early reflections, and a decaying noise-like tail [2].

    [1] Steinmetz, Christian J., Vamsi Krishna Ithapu, and Paul Calamia.
        "Filtered noise shaping for time domain room impulse response estimation from reverberant speech."
        2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2021.

    [2] Moorer, James A. 
        "About this reverberation business." 
        Computer Music Journal (1979): 13-28.

    Args:
        x (torch.Tensor):
        sample_rate (float):

    Returns:
        y (torch.Tensor):

    """
    # d are the decay parameters for each band
    # g are the gains for each band

    bs, chs, samp = x.size()

    # generate white noise for IR generation
    pad_size = self.num_taps - 1
    wn = torch.randn(1, self.num_bands, self.num_samples + pad_size).type_as(d)

    # filter white noise signals with each bandpass filter
    wn_filt = torch.nn.functional.conv1d(
        wn,
        self.filts,
        groups=self.num_bands,
        # padding=self.num_taps -1,
    )
    # shape: bs x num_bands x num_samples

    # apply bandwise decay parameters (envelope)
    t = torch.linspace(0, 1, steps=self.num_samples).type_as(d)  # timesteps
    # d = (torch.rand(bs, self.num_bands, 1) * 15) + 5
    d = (d * 1000) + 1.0
    env = torch.exp(-d * t.view(1, 1, -1))

    # for e_idx in np.arange(env.shape[1]):
    #    plt.plot(env[0, e_idx, :].squeeze().numpy())
    #    plt.savefig("env.png", dpi=300)

    wn_filt *= env * g

    # sum signals to create impulse shape: bs x 1 x num_samp
    w_filt_sum = wn_filt.mean(1, keepdim=True)
    w_filt_sum = w_filt_sum.unsqueeze(1)  # hape: bs x 1 x 1 x num_samp

    # plt.plot(w_filt_sum[0, ...].squeeze().numpy())
    # plt.savefig("impulse.png", dpi=300)

    # apply impulse response for each batch item (vectorized)
    # x_pad = torch.nn.functional.pad(x, (self.num_samples - 1, 0))
    # wet = self.vconv1d(
    #    x_pad,
    #    torch.flip(w_filt_sum, dims=[-1]),
    # )

    # this is the RIR
    # y = torch.flip(w_filt_sum, dims=[-1])
    y = w_filt_sum.view(1, -1)

    # create a wet/dry mix
    # g = 0.1
    # y = (x * (1 - g)) + (wet * g)

    return y
