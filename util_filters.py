import warnings

import numpy as np
import scipy.signal


def freq2erb(freq):
    """
    Convert frequency in Hz to ERB-number.
    Same as `freqtoerb.m` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """
    Convert ERB-number to frequency in Hz.
    Same as `erbtofreq.m` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(freq_min, freq_max, num):
    """
    Create an array of frequencies in Hz evenly spaced on a ERB-number scale.
    Same as `erbspace.m` in the AMT.

    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)

    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    return erb2freq(np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num))


def erbspace_bw(freq_min, freq_max, erb):
    """
    Create an array of frequencies in Hz with given ERB-number spacing.
    Same as `erbspacebw.m` in the AMT.
    """
    erb_min = freq2erb(freq_min)
    erb_max = freq2erb(freq_max)
    n = (erb_max - erb_min) // erb
    remainder = (erb_max - erb_min) - n * erb
    return erb2freq(erb_min + np.arange(n + 1) * erb + remainder / 2)


def aud_filt_bw(cf):
    """
    Critical bandwidth of auditory filter at given center frequency.
    Same as `audfiltbw.m` in the AMT.
    """
    return 24.7 + cf / 9.265


def scipy_apply_filterbank(x, filter_coeffs):
    """
    Convert signal waveform `x` to a set of subbands `x_filtered`
    using scipy.signal.lfilter and specified `filter_coeffs`

    Args
    ----
    x (np.ndarray): input signal to be filtered with shape [..., time]
    filter_coeffs (list): list of filter coefficient dictionaries
        [{'b': [n_filters, order], 'a': [n_filters, order]}, ...]

    Returns
    -------
    x_filtered (np.ndarray): subbands with shape [..., n_filters, time]
    """
    if isinstance(filter_coeffs, dict):
        filter_coeffs = [filter_coeffs]
    assert isinstance(filter_coeffs, list)
    assert isinstance(filter_coeffs[0], dict)
    n_filters = filter_coeffs[0]["b"].shape[0]
    x_filtered = np.expand_dims(x, -2)
    reps = np.ones(x_filtered.ndim, dtype=int)
    reps[-2] = n_filters
    x_filtered = np.tile(x_filtered, reps)
    for ba in filter_coeffs:
        for itr in range(n_filters):
            x_filtered[..., itr, :] = scipy.signal.lfilter(
                ba["b"][itr],
                ba["a"][itr],
                x_filtered[..., itr, :],
                axis=-1,
            ).real
    return x_filtered


def butter(order, cutoff, **kwargs):
    """
    A wrapper around `scipy.signal.butter` with support for multiple cutoff frequencies.

    Args
    ----
    order (int): Filter order
    cutoff (float or list or np.ndarray): Cutoff frequencies with shape `(n_filters,)`
    kwargs (dict): Optional keyword arguments passed to `scipy.signal.butter`

    Returns
    -------
    b (np.ndarray): Numerator coefficients. Shape `(n_taps,)` or `(n_filters, n_taps)`
    a (np.ndarray): Denominator coefficients. Same shape as `b`
    """
    if "output" in kwargs and kwargs["output"] != "ba":
        raise ValueError(f"only 'ba' output is supported, got {kwargs['output']}")
    cutoff, scalar_input = _check_0d_or_1d(cutoff, "cutoff")
    b = np.empty((len(cutoff), order + 1))
    a = np.empty((len(cutoff), order + 1))
    for itr, f in enumerate(cutoff):
        b[itr, :], a[itr, :] = scipy.signal.butter(order, f, **kwargs)
    if scalar_input:
        b, a = b[0, :], a[0, :]
    return b, a


def fir2(order, freq, amp):
    """
    FIR filter design.

    Same as `fir2.m` in the Signal Processing Toolbox. `fir2.m` uses an interpolation
    method that is not equivalent to `np.interp` used inside `scipy.signal.firwin2`.

    Args
    ----
    order (int): Filter order
    freq (list or np.ndarray): Frequencies at which amplitude response is sampled
    amp (list or np.ndarray): Amplitude response values with same shape as `freq`

    Returns
    -------
    fir (np.ndarray): Filter coefficients with shape `(order + 1,)`
    """
    if isinstance(freq, list):
        freq = np.array(freq)
    if isinstance(amp, list):
        amp = np.array(amp)
    if freq.ndim != 1 or amp.ndim != 1:
        raise ValueError("freq and amp must be one-dimensional")
    if len(freq) != len(amp):
        raise ValueError("freq and amp must have the same length")
    if freq[0] != 0.0 or freq[-1] != 1.0:
        raise ValueError("freq must start with 0.0 and end with 1.0")
    d = np.diff(freq)
    if (d <= 0).any():
        raise ValueError("freq must be strictly increasing")
    if order % 2 == 1 and amp[-1] == 0:
        raise ValueError("odd order filter requires zero gain at Nyquist frequency")
    # fir2.m forces the minimum number of interpolation points to 513
    n_taps = order + 1
    if n_taps < 1024:
        n_interp = 513
    else:
        n_interp = 2 * int(np.ceil(np.log2(n_taps))) + 1
    # Interpolation
    H = np.zeros(n_interp)
    i_start = 0
    for i in range(len(freq) - 1):
        i_end = int(np.floor(freq[i + 1] * n_interp))
        inc = np.arange(i_end - i_start) / (i_end - i_start - 1)
        H[i_start:i_end] = inc * amp[i + 1] + (1 - inc) * amp[i]
        i_start = i_end
    # Phase shift
    shift = np.exp(-0.5 * (n_taps - 1) * 1j * np.pi * np.linspace(0, 1, n_interp))
    fir = np.fft.irfft(H * shift)
    # Keep first n_taps coefficients and multiply by window
    return fir[:n_taps] * np.hamming(n_taps)


def matched_z_transform(poles, zeros=None, sr=1.0, gain=None, f0db=None, _z_zeros=None):
    """
    Analog to digital filter using the matched Z-transform method.
    See https://en.wikipedia.org/wiki/Matched_Z-transform_method.

    Args
    ----
    poles (list or np.ndarray):
        Poles in the s-plane with shape (n_poles,) or (n_filters, n_poles).
    zeros (list or np.ndarray or None):
        Zeros in the s-plane with shape (n_zeros,) or (n_filters, n_zeros).
        If `None` the filter is all-pole.
    sr (float): Sampling rate in Hz
    gain (float or list or np.ndarray or None):
        Continuous filter gain. If np.ndarray, must have shape (n_filters,).
        If `None` and `f0db` is also `None`, no gain is applied and the first
        coefficient in `b` is `1.0`.
    f0db: (float or None):
        Frequency at which the filter should have unit gain. If `None` and
        `gain` is also `None`, no gain is applied and the first coefficient
        in `b` is `1.0`.

    Returns
    -------
    b (np.ndarray): Numerator coefficients with shape (n_filters, n_zeros + 1) or
        (n_zeros + 1,)
    a (np.ndarray): Denominator coefficients with shape (n_filters, n_poles + 1) or
        (n_poles + 1,)
    """
    poles, poles_was_1d = _check_1d_or_2d(poles, "poles")
    if zeros is None:
        zeros, zeros_was_1d = np.empty((poles.shape[0], 0)), poles_was_1d
    else:
        zeros, zeros_was_1d = _check_1d_or_2d(zeros, "zeros")
    both_1d = poles_was_1d and zeros_was_1d
    both_2d = not poles_was_1d and not zeros_was_1d
    if not both_1d and not both_2d:
        raise ValueError("poles and zeros must have the same number of dimensions")
    if both_2d and poles.shape[0] != zeros.shape[0]:
        raise ValueError("poles and zeros must have the same shape along first axis")
    z_poles = np.exp(poles / sr)
    # _z_zeros is used for hard-setting the Z-domain zeros which is useful for the
    # amt_classic and amt_allpole gammatone filters. It should not be used in general!
    z_zeros = np.exp(zeros / sr) if _z_zeros is None else _z_zeros
    # TODO: find a way to vectorize polyfromroots instead of lopping over b and a
    b = np.empty((z_zeros.shape[0], z_zeros.shape[1] + 1))
    a = np.empty((z_poles.shape[0], z_poles.shape[1] + 1))
    for i in range(z_poles.shape[0]):
        b[i, ::-1] = np.polynomial.polynomial.polyfromroots(z_zeros[i, :]).real
        a[i, ::-1] = np.polynomial.polynomial.polyfromroots(z_poles[i, :]).real
    if gain is not None and f0db is not None:
        raise ValueError("cannot specify both gain and f0db")
    elif gain is not None:
        # _z_zeros cannot be used together with s-domain gain since calculating the
        # corresponding Z-domain gain requires the initial s-domain zeros.
        if _z_zeros is not None:
            raise ValueError("cannot specify both gain and _z_zeros")
        gain, _ = _check_0d_or_1d(gain, "gain")
        z_gain = np.abs(
            gain
            * np.prod(-zeros, axis=1)
            / np.prod(-poles, axis=1)
            * np.prod(1 - z_poles, axis=1)
            / np.prod(1 - z_zeros, axis=1)
        )
        b = z_gain[:, None] * b
    elif f0db is not None:
        f0db, _ = _check_0d_or_1d(f0db)
        z_f0db = np.exp(-1j * 2 * np.pi * f0db / sr)
        z_gain = np.abs(
            np.prod(1 - z_poles * z_f0db[:, None], axis=1)
            / np.prod(1 - z_zeros * z_f0db[:, None], axis=1)
        )
        b = z_gain[:, None] * b
    if both_1d:
        b, a = b[0, :], a[0, :]
    return b, a


def goode1994():
    """
    Get stapes footplate diplacement data from Goode et al. (1994).
    Same as `data_goode1994.m` in the AMT.
    """
    return np.array(
        [
            [400, 0.19953],
            [600, 0.22909],
            [800, 0.21878],
            [1000, 0.15136],
            [1200, 0.10000],
            [1400, 0.07943],
            [1600, 0.05754],
            [1800, 0.04365],
            [2000, 0.03311],
            [2200, 0.02754],
            [2400, 0.02188],
            [2600, 0.01820],
            [2800, 0.01445],
            [3000, 0.01259],
            [3500, 0.00900],
            [4000, 0.00700],
            [4500, 0.00457],
            [5000, 0.00500],
            [5500, 0.00400],
            [6000, 0.00300],
            [6500, 0.00275],
        ]
    )


def lopezpoveda2001():
    """
    Get outer and middle ear filter data from Lopez-Poveda & Meddis (2001).
    Same as `data_lopezpoveda2001.m` with argument `fig2b` in the AMT.
    """
    data = goode1994()
    data[:, 1] *= 1e-6 * 2 * np.pi * data[:, 0]
    data[:, 1] *= 10 ** (-104 / 20)
    data[:, 1] *= 2**0.5
    extrp = np.array(
        [
            [100, 1.181e-9],
            [200, 2.363e-9],
            [7000, 8.705e-10],
            [7500, 8.000e-10],
            [8000, 7.577e-10],
            [8500, 7.168e-10],
            [9000, 6.781e-10],
            [9500, 6.240e-10],
            [10000, 6.000e-10],
        ]
    )
    return np.vstack(
        [
            extrp[extrp[:, 0] < data[0, 0]],
            data,
            extrp[extrp[:, 0] > data[-1, 0]],
        ]
    )


def middle_ear_filter_fir(sr, order=512, min_phase=True):
    """
    Create a middle ear FIR filter.
    Same as `middleearfilter.m` with argument `lopezpoveda2001` in the AMT.
    """
    data = lopezpoveda2001()
    if sr <= 20e3:
        data = data[data[:, 0] < sr / 2, :]
    else:
        i = np.arange(1, 1 + (sr / 2 - data[-1, 0]) // 1000).reshape(-1, 1)
        data = np.vstack(
            [
                data,
                np.hstack(
                    [
                        data[-1, 0] + i * 1000,
                        data[-1, 1] / 1.1**i,
                    ]
                ),
            ]
        )
    if data[-1, 0] != sr / 2:
        data = np.vstack(
            [
                data,
                np.array([[sr / 2, data[-1, 1] / (1 + (sr / 2 - data[-1, 0]) * 1e-4)]]),
            ]
        )
    data = np.vstack([np.array([0, 0]), data * np.array([[2 / sr, 1]])])
    b = fir2(order, data[:, 0], data[:, 1])
    b = b / 20e-6
    if min_phase:
        b = np.fft.fft(b)
        b = np.abs(b) * np.exp(-1j * scipy.signal.hilbert(np.log(np.abs(b))).imag)
        b = np.fft.ifft(b).real
    return b


def gammatone_filter_coeffs(
    sr,
    cfs,
    order=4,
    bw_mult=None,
    filter_type="gtf",
    iir_output="sos",
):
    """
    Gammatone filter coefficients.

    Parameters
    ----------
    sr (float):
        Sampling rate in Hz
    cfs (float or list or np.ndarray):
        Center frequencies with shape (n_filters,)
    order (int):
        Filter order
    bw_mult (float or np.ndarray or None):
        Bandwidth scaling factor  with shape (n_filters,) or None,
        in which case formula from [1]_ is used
    filter_type (str): {"gtf", "apgf", "ozgf", "amt_classic", "amt_allpole"}
        - "gtf": Accurate IIR equivalent by numerically calculating the s-place zeros.
        - "apgf": All-pole approximation from [2]_.
        - "ozgf": One-zero approximation from [2]_. The zero is set to 0 which matches
          the DAPGF denomination in more recent papers by Lyon.
        - "amt_classic": Mixed pole-zero approximation from [?]_.
          Matches the "classic" option in the AMT.
        - "amt_allpole": Same as "apgf" but uses a different scaling.
          Matches the "allpole" option in the AMT.
    iir_output (str): {"ba", "sos"}
        Determines whether to return IIR filter coefficients as a single set of
        `b` and `a` coefficients ("ba") or as a sequence of `b` and `a` coefficients
        corresponding to second-order sections ("sos") to be applied successively.
        For stability, "sos" is recommended, but is computationally more expensive.

    Returns
    -------
    filter_coeffs (list of {"b": np.ndarray, "a" np.ndarray} dicts):
        List of dicts containing numerator ("b") and denominator ("a") coefficients.
        If iir_output == "ba", then the list will have a length of one
        If iir_output == "sos", then the list will have a length equal to `order`
        Each coefficient array will have a shape (n_filters, n_taps)

    References
    ----------
    .. [1] J. Holdsworth, I. Nimmo-Smith, R. D. Patterson and P. Rice, "Annex C of the
       SVOS final report: Implementing a gammatone filter bank", Annex C of APU report
       2341, 1988.
    .. [2] R. F. Lyon, "The all-pole gammatone filter and auditory models", in Proc.
       Forum Acusticum, 1996.
    """
    cfs, scalar_input = _check_0d_or_1d(cfs, "cfs")
    if bw_mult is None:
        bw_mult = np.math.factorial(order - 1) ** 2 / (
            np.pi * np.math.factorial(2 * order - 2) * 2 ** (-2 * order + 2)
        )
    bw = 2 * np.pi * bw_mult * aud_filt_bw(cfs)
    wc = 2 * np.pi * cfs
    pole = -bw + 1j * wc
    poles = np.stack([pole, pole.conj()], axis=1)
    zeros = None
    _z_zeros = None
    if filter_type == "gtf":
        zeros = np.zeros((len(cfs), order))
        for i in range(len(cfs)):
            zeros[i, :] = np.polynomial.polynomial.polyroots(
                np.polynomial.polynomial.polyadd(
                    np.polynomial.polynomial.polypow([-pole[i], 1], order),
                    np.polynomial.polynomial.polypow([-pole[i].conj(), 1], order),
                )
            ).real
    elif filter_type == "apgf":
        pass
    elif filter_type == "ozgf":
        zeros = np.zeros((len(cfs), 1))
    elif filter_type in ["amt_classic", "amt_allpole"]:
        # The MATLAB code sets the Z-domain zeros to the real part of the Z-domain
        # poles, which seems wrong! Moreover, those zeros are used for calculating
        # the gain for both classic and allpole, which is probably why a warning is
        # raised about the scaling being wrong for allpole!
        _z_zeros = np.stack(order * [np.exp((pole) / sr).real], axis=1)
    else:
        raise ValueError(f"invalid filter_type, got {filter_type}")
    if iir_output == "ba":
        warnings.warn(
            "Using iir_output='ba' can lead to numerically unstable "
            "gammatone filters. Consider using iir_output='sos' instead."
        )
        poles = np.tile(poles, (1, order))
        b, a = matched_z_transform(poles, zeros, sr=sr, f0db=cfs, _z_zeros=_z_zeros)
        if filter_type == "amt_allpole":
            b = b[:, :1]
        filter_coeffs = [{"b": b, "a": a}]
    elif iir_output == "sos":
        filter_coeffs = []
        for i in range(order):
            zeros_i = None if zeros is None else zeros[:, i : i + 1]
            _z_zeros_i = None if _z_zeros is None else _z_zeros[:, i : i + 1]
            b, a = matched_z_transform(
                poles, zeros_i, sr=sr, f0db=cfs, _z_zeros=_z_zeros_i
            )
            if filter_type == "amt_allpole":
                b = b[:, :1]
            b = np.hstack([b, np.zeros((len(cfs), 3 - b.shape[-1]))])
            filter_coeffs.append({"b": b, "a": a})
    else:
        raise ValueError(f"iir_output must be `ba` or `sos`, got `{iir_output}`")
    if scalar_input:
        filter_coeffs = [{"b": _["b"][0], "a": _["a"][0]} for _ in filter_coeffs]
    return filter_coeffs


def gammatone_filter_fir(
    sr,
    cfs,
    fir_dur=0.05,
    order=4,
    bw_mult=None,
):
    """
    Finite impulse responses of Gammatone filter(s).
    See `gammatone_filter_coeffs` for detailed parameters.

    Args
    ----
    sr (float): Sampling rate in Hz
    fir_dur (float): Duration of FIR in seconds
    cfs (float or list or np.ndarray): Center frequencies with shape (n_filters,)
    order (int):  Filter order
    bw_mult (float or np.ndarray or None): Bandwidth scaling factor

    Returns
    -------
    fir (np.ndarray): impulse responses with shape (n_filters, int(sr * fir_dur))
    """
    cfs, scalar_input = _check_0d_or_1d(cfs, "cfs")
    if bw_mult is None:
        bw_mult = np.math.factorial(order - 1) ** 2 / (
            np.pi * np.math.factorial(2 * order - 2) * 2 ** (-2 * order + 2)
        )
    else:
        bw_mult = np.array(bw_mult)
    bw = 2 * np.pi * bw_mult * aud_filt_bw(cfs)
    wc = 2 * np.pi * cfs

    fir_ntaps = int(fir_dur * sr)
    t = np.arange(fir_ntaps) / sr
    a = (
        2
        / np.math.factorial(order - 1)
        / np.abs(1 / bw**order + 1 / (bw + 2j * wc) ** order)
        / sr
    )
    fir = (
        a[:, None]
        * t ** (order - 1)
        * np.exp(-bw[:, None] * t[None, :])
        * np.cos(wc[:, None] * t[None, :])
    )
    if scalar_input:
        fir = fir[0]
    return fir


def ihc_lowpass_filter_fir(sr, fir_dur, cutoff=3e3, order=7):
    """
    Returns finite response of IHC lowpass filter from
    bez2018model/model_IHC_BEZ2018.c
    """
    n_taps = int(sr * fir_dur)
    if n_taps % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr - 2 * np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * scipy.signal.windows.hann(n_taps)
    fir = fir / fir.sum()
    return fir


def mfbtd_filter_coeffs(sr=10e3, lmf=0, umf=1500, Q=2, bw=2, complex=False):
    """
    Modualtion filterbank (Dau et al. 1997) ported from mfbtd.m:
    (c) 1999 Stephan Ewert and Torsten Dau, Universitaet Oldenburg
    """
    if lmf == 0:
        startmf = 5
        lpcut = 2.5
    else:
        startmf = lmf
        lpcut = None

    def efilt(w0, bw):
        """Complex frequency shifted first order lowpass filter"""
        e0 = np.exp(-bw / 2)
        b = np.array([1 - e0])
        a = np.array([1, -e0 * np.exp(1j * w0)])
        return b, a

    def folp(w0):
        """First order Butterworth lowpass filter"""
        W0 = np.tan(w0 / 2)
        b = np.array([W0 / (1 + W0), W0 / (1 + W0)])
        a = np.array([1, (W0 - 1) / (1 + W0)])
        return b, a

    def solp(w0, Q):
        """Second order Butterworth lowpass filter"""
        W0 = np.tan(w0 / 2)
        b = np.array([1, 2, 1])
        a = np.array(
            [
                1 + 1 / (Q * W0) + 1 / (W0**2),
                -2 / (W0**2) + 2,
                1 - 1 / (Q * W0) + 1 / (W0**2),
            ]
        )
        b = b / a[0]
        a = a / a[0]
        return b, a

    ex = (1 + 1 / (2 * Q)) / (1 - 1 / (2 * Q))
    mf = startmf + 5 * np.arange(0, int((min(umf, 10) - startmf) / bw) + 1)
    tmp2 = (mf[-1] + bw / 2) / (1 - 1 / (2 * Q))
    tmp1 = np.power(ex, np.arange(0, int(np.log(umf / tmp2) / np.log(ex)) + 1))
    if lmf == 0:
        cfs = np.concatenate([[lmf], mf, tmp2 * tmp1])
    else:
        cfs = np.concatenate([mf, tmp2 * tmp1])
    modulation_filter_coeffs = {
        "b": np.zeros([len(cfs), 3], dtype=np.complex128 if complex else np.float64),
        "a": np.zeros([len(cfs), 3], dtype=np.complex128 if complex else np.float64),
    }
    for itr, cf in enumerate(cfs):
        if cf == 0:
            # Lowpass modulation filter
            wb2lp = 2 * np.pi * lpcut / sr
            b, a = solp(wb2lp, 1 / np.sqrt(2))
        else:
            # Bandpass modulation filter (note 2x scale factor)
            w0 = 2 * np.pi * cf / sr
            if cf < 10:
                b, a = efilt(w0, 2 * np.pi * bw / sr)
            else:
                b, a = efilt(w0, w0 / Q)
            b = 2 * b
            if not complex:
                b = np.convolve(b, np.conjugate(a)).real
                a = np.convolve(a, np.conjugate(a)).real
        modulation_filter_coeffs["b"][itr, : len(b)] = b
        modulation_filter_coeffs["a"][itr, : len(a)] = a
    b, a = butter(order=1, cutoff=150, btype="low", analog=False, output="ba", fs=sr)
    lowpass_filter_coeffs = {
        "b": np.tile(b, (len(cfs), 1)),
        "a": np.tile(a, (len(cfs), 1)),
    }
    filter_coeffs = [
        lowpass_filter_coeffs,
        modulation_filter_coeffs,
    ]
    return filter_coeffs


def mfbtd_filter_fir(sr, fir_dur, **kwargs):
    """
    Returns array of finite impulse responses (truncated to fir_dur)
    for a modulation filterbank with the specified properties.
    """
    impulse = np.zeros(int(fir_dur * sr))
    impulse[0] = 1
    filter_coeffs = mfbtd_filter_coeffs(sr, **kwargs)
    return scipy_apply_filterbank(impulse, filter_coeffs)


def _check_0d_or_1d(x, name="input"):
    """
    Checks if input is 0- or 1- dimensional, converts input to 1-dimensional
    array, and returns bool indicating if input was 0-dimensional.
    """
    is_0d = (
        isinstance(x, (int, float, np.integer, np.floating))
        or isinstance(x, np.ndarray)
        and x.ndim == 0
    )
    if is_0d:
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError(
            f"{name} must be scalar or np.ndarray, got {x.__class__.__name__}"
        )
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    return x, is_0d


def _check_1d_or_2d(x, name="input"):
    """
    Checks if input is 1- or 2- dimensional, converts input to 2-dimensional
    array, and returns bool indicating if input was 1-dimensional.
    """
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError(
            f"{name} must be list or np.ndarray, got {x.__class__.__name__}"
        )
    if isinstance(x, list):
        x = np.array(x)
    is_1d = x.ndim == 1
    if is_1d:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {x.shape}")
    return x, is_1d


def _batching_check(x, b):
    """
    Raises error if filterbank (parameterized with `b`) cannot be applied
    channelwise to tensor `x`. The word batching here is confusing, as it
    refers to applying a filterbank channelwise (a different filter for
    each channel). The word batching is used to match argument from
    `torchaudio.functional.lfilter`.
    """
    if x.ndim < 2:
        raise ValueError("batching requires input with at least two dimensions")
    if b.ndim != 2:
        raise ValueError("batching requires filter to be two-dimensional")
    if x.shape[-2] != b.shape[-2]:
        raise ValueError(
            "batching requires input and filter to have the same number of "
            f"channels, got {x.shape[-2]} and {b.shape[-2]}"
        )
