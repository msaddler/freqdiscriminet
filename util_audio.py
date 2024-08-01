import numpy as np
import scipy.interpolate
import scipy.signal


def rms(x):
    """
    Returns root mean square amplitude of x (raises ValueError if NaN).
    """
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError("rms calculation resulted in NaN")
    return out


def get_dbspl(x, mean_subtract=True):
    """
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    """
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dbspl(x, dbspl, mean_subtract=True):
    """
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    """
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dbspl / 20)
    return rms_out * x / rms(x)


def combine_signal_and_noise(signal, noise, snr, mean_subtract=True):
    """
    Adds noise to signal with the specified signal-to-noise ratio (snr).
    If snr is finite, the noise waveform is rescaled and added to the
    signal waveform. If snr is positive infinity, returned waveform is
    equal to the signal waveform. If snr is negative inifinity, returned
    waveform is equal to the noise waveform.

    Args
    ----
    signal (np.ndarray): signal waveform
    noise (np.ndarray): noise waveform
    snr (float): signal-to-noise ratio in dB
    mean_subtract (bool): if True, signal and noise are first de-meaned
        (mean_subtract=True is important for accurate snr computation)

    Returns
    -------
    signal_and_noise (np.ndarray) signal in noise waveform
    """
    if mean_subtract:
        signal = signal - np.mean(signal)
        noise = noise - np.mean(noise)
    if np.isinf(snr) and snr > 0:
        signal_and_noise = signal
    elif np.isinf(snr) and snr < 0:
        signal_and_noise = noise
    else:
        rms_noise_scaling = rms(signal) / (rms(noise) * np.power(10, snr / 20))
        signal_and_noise = signal + rms_noise_scaling * noise
    return signal_and_noise


def index_trim_zeros(x, trim="fb"):
    """
    Get `index_f` and `index_b` such that `x[index_f:index_b]`
    returns the output of `np.trim_zeros(x, trim)`.
    """
    index_f = 0
    index_b = x.shape[0]
    x_is_zero = x == 0
    if x.ndim > 1:
        x_is_zero = np.all(x_is_zero, axis=tuple(range(1, x.ndim)))
    if "f" in trim.lower():
        index_f = np.argmin(x_is_zero)
    if "b" in trim.lower():
        index_b = x.shape[0] - np.argmin(x_is_zero[::-1])
    return index_f, index_b


def pad_or_trim_to_len(x, n, mode="both", kwargs_pad={}):
    """
    Increases or decreases the length of a one-dimensional signal
    by either padding or triming the array. If the difference
    between `len(x)` and `n` is odd, this function will default to
    adding/removing the extra sample at the end of the signal.

    Args
    ----
    x (np.ndarray): one-dimensional input signal
    n (int): length of output signal
    mode (str): specify which end of signal to modify
        (default behavior is to symmetrically modify both ends)
    kwargs_pad (dict): keyword arguments for np.pad function

    Returns
    -------
    x_out (np.ndarray): one-dimensional signal with length `n`
    """
    assert len(np.array(x).shape) == 1, "input must be 1D array"
    assert mode.lower() in ["both", "start", "end"]
    n_diff = np.abs(len(x) - n)
    if len(x) > n:
        if mode.lower() == "end":
            x_out = x[:n]
        elif mode.lower() == "start":
            x_out = x[-n:]
        else:
            x_out = x[int(np.floor(n_diff / 2)) : -int(np.ceil(n_diff / 2))]
    elif len(x) < n:
        if mode.lower() == "end":
            pad_width = [0, n_diff]
        elif mode.lower() == "start":
            pad_width = [n_diff, 0]
        else:
            pad_width = [int(np.floor(n_diff / 2)), int(np.ceil(n_diff / 2))]
        kwargs = {"mode": "constant"}
        kwargs.update(kwargs_pad)
        x_out = np.pad(x, pad_width, **kwargs)
    else:
        x_out = x
    assert len(x_out) == n
    return x_out


def random_pad(x, padded_length, axis=0, buffer=None, **kwargs_pad):
    """
    Randomly pad array up to specified length along specified axis.
    """
    if isinstance(buffer, int):
        buffer_start = buffer
        buffer_end = buffer
    elif isinstance(buffer, tuple):
        buffer_start, buffer_end = buffer
    else:
        buffer_start = 0
        buffer_end = 0
    npad = padded_length - x.shape[axis]
    assert npad >= 0, "padded length < source length"
    pad_start = np.random.randint(
        low=buffer_start, high=npad - buffer_end + 1, dtype=int
    )
    pad_end = npad - pad_start
    pad_width = [(0, 0) for _ in range(len(x.shape))]
    pad_width[axis] = (pad_start, pad_end)
    return np.pad(x, pad_width, **kwargs_pad)


def random_slice(x, slice_length, axis=0, buffer=None):
    """
    Sample a random, consecutive slice of specified length from an
    input array along the specified axis.
    """
    if isinstance(buffer, int):
        buffer_start = buffer
        buffer_end = buffer
    elif isinstance(buffer, tuple):
        buffer_start, buffer_end = buffer
    else:
        buffer_start = 0
        buffer_end = 0
    assert x.shape[axis] >= slice_length, "source length < slice length"
    idx = np.random.randint(
        low=buffer_start, high=x.shape[axis] + 1 - buffer_end - slice_length, dtype=int
    )
    slice_index = [slice(None, None, None) for _ in range(len(x.shape))]
    slice_index[axis] = slice(idx, idx + slice_length, None)
    return x[tuple(slice_index)]


def random_pad_or_random_slice(x, target_length, axis=0, buffer=None, verbose=False):
    """
    Randomly pad or slice an array to a target length along specified axis.

    Args
    ----
    x (np.ndarray): input array
    target_length (int): desired length of input array along specified `axis`
    axis (int): axis of `x` to pad or slice to `target_length`
    buffer (int, tuple, function): limits how close to start and end that `x` can be
        padded or sliced. If `buffer` is a function, it will be be used to map
        source_length to buffer for pad/slice function: buffer = buffer(x.shape[axis])
    verbose (bool): if True, print whether and how x is randomly padded or sliced

    Returns
    -------
    (np.ndarray) padded or sliced to length `target_length` along specified `axis`
    """
    source_length = x.shape[axis]
    if callable(buffer):
        buffer = buffer(source_length)
    if source_length < target_length:
        if verbose:
            print(
                f"random_pad: {source_length} -> {target_length} with buffer {buffer}"
            )
        return random_pad(x, target_length, axis=axis, buffer=buffer)
    elif source_length > target_length:
        if verbose:
            print(
                f"random_slice: {source_length} -> {target_length} with buffer {buffer}"
            )
        return random_slice(x, target_length, axis=axis, buffer=buffer)
    return x


def spatialize_sound(y, brir):
    """
    This takes a left-aligned BRIR and convolves it with a left-padded signal
    (using "valid" padding) to produce the same output as "same" padding with
    a center-aligned BRIR. It is faster to pad the signal than it is to pad
    the BRIR.

    Args
    ----
    y (np.ndarray): monoaural waveform with shape [timesteps]
    brir (np.ndarray): binaural room impulse response with shape [timesteps, 2]

    Returns
    -------
    y_spatialized (np.ndarray): binaural waveform with shape [timesteps, 2]
    """
    y_padded = np.pad(y, (brir.shape[0] - 1, 0))
    y_spatialized_l = scipy.signal.convolve(
        y_padded, brir[:, 0], mode="valid", method="auto"
    )
    y_spatialized_r = scipy.signal.convolve(
        y_padded, brir[:, 1], mode="valid", method="auto"
    )
    return np.stack([y_spatialized_l, y_spatialized_r]).T


def power_spectrum(x, sr, rfft=True, dbspl=True):
    """
    Helper function for computing power spectrum of sound wave.

    Args
    ----
    x (np.ndarray): input waveform (Pa)
    sr (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectrum is returned
    dbspl (bool): if True, power spectrum has units dB re 20e-6 Pa

    Returns
    -------
    fxx (np.ndarray): frequency vector (Hz)
    pxx (np.ndarray): power spectrum (Pa^2 or dB SPL)
    """
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        pxx = 2 * np.square(np.abs(np.fft.rfft(x) / len(x)))
        fxx = np.fft.rfftfreq(len(x), d=1 / sr)
    else:
        pxx = np.square(np.abs(np.fft.fft(x) / len(x)))
        fxx = np.fft.fftfreq(len(x), d=1 / sr)
    if dbspl:
        pxx = 10.0 * np.log10(pxx / np.square(20e-6))
    return fxx, pxx


def impose_power_spectrum(x, pxx):
    """
    Impose power spectrum in frequency domain by multiplying FFT of a
    frame (x) with square root of the given power_spectrum and applying
    inverse FFT. power_spectrum must have same shape as the rfft of x.
    """
    x_fft = np.fft.rfft(x, norm="ortho")
    x_fft *= np.sqrt(pxx)
    return np.fft.irfft(x_fft, norm="ortho")


def tenoise(sr, dur, lco=None, hco=None, dbspl_per_erb=70.0):
    """
    Generates threshold equalizing noise (Moore et al. 1997) in the spectral
    domain with specified sampling rate, duration, cutoff frequencies, and
    level. TEnoise produces equal masked thresholds for normal hearing
    listeners for all frequencies between 125 Hz and 15 kHz. Assumption:
    power of the signal at threshold (Ps) is given by the equation,
    Ps = No*K*ERB, where No is the noise power spectral density and K is the
    signal to noise ratio at the output of the auditory filter required for
    threshold. TENoise is spectrally shaped so that No*K*ERB is constant.
    Values for K and ERB are taken from Moore et al. (1997).

    Based on MATLAB code last modified by A. Oxenham (2007-JAN-30).
    Modified Python implementation by M. Saddler (2020-APR-21).

    Args
    ----
    sr (int): sampling rate in Hz
    dur (float): duration of noise (s)
    lco (float): low cutoff frequency in Hz (defaults to 0.0)
    hco (float): high cutoff frequency in Hz (defaults to sr/2)
    dbspl_per_erb (float): level of TENoise is specified in terms of the
        level of a one-ERB-wide band around 1 kHz in units dB re 20e-6 Pa

    Returns
    -------
    noise (np.ndarray): noise waveform in units of Pa
    """
    # Set parameters for synthesizing TENoise
    nfft = int(dur * sr)  # nfft = duration in number of samples
    if lco is None:
        lco = 0.0
    if hco is None:
        hco = sr / 2.0

    # K values are from a personal correspondance between B.C.J. Moore
    # and A. Oxenham. A also figure appears in Moore et al. (1997).
    K = np.array(
        [
            [0.0500, 13.5000],
            [0.0630, 10.0000],
            [0.0800, 7.2000],
            [0.1000, 4.9000],
            [0.1250, 3.1000],
            [0.1600, 1.6000],
            [0.2000, 0.4000],
            [0.2500, -0.4000],
            [0.3150, -1.2000],
            [0.4000, -1.8500],
            [0.5000, -2.4000],
            [0.6300, -2.7000],
            [0.7500, -2.8500],
            [0.8000, -2.9000],
            [1.0000, -3.0000],
            [1.1000, -3.0000],
            [2.0000, -3.0000],
            [4.0000, -3.0000],
            [8.0000, -3.0000],
            [10.0000, -3.0000],
            [15.0000, -3.0000],
        ]
    )

    # K values are interpolated over rfft frequency vector
    f_interp_K = scipy.interpolate.interp1d(
        K[:, 0], K[:, 1], kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    freq = np.fft.rfftfreq(nfft, d=1 / sr)
    KdB = f_interp_K(freq / 1000)

    # Calculate ERB at each frequency and compute TENoise PSD
    ERB = 24.7 * ((4.37 * freq / 1000) + 1)  # Glasberg & Moore (1990) equation 3
    TEN_No = -1 * (KdB + (10 * np.log10(ERB)))  # Units: dB/Hz re 1

    # Generate random noise_rfft vector and scale to TENoise PSD between lco and hco
    freq_idx = np.logical_and(freq > lco, freq < hco)
    a = np.zeros_like(freq)
    b = np.zeros_like(freq)
    a[freq_idx] = np.random.randn(np.sum(freq_idx))
    b[freq_idx] = np.random.randn(np.sum(freq_idx))
    noise_rfft = a + 1j * b
    noise_rfft[freq_idx] = noise_rfft[freq_idx] * np.power(10, (TEN_No[freq_idx] / 20))

    # Estimate power in ERB centered at 1 kHz and compute scale factor for dbspl_per_erb
    freq_idx_1khz_erb = np.logical_and(freq > 935.0, freq < 1068.1)
    power_1khz_erb = (
        2 * np.sum(np.square(np.abs(noise_rfft[freq_idx_1khz_erb]))) / np.square(nfft)
    )
    dbspl_power_1khz_erb = 10 * np.log10(power_1khz_erb / np.square(20e-6))
    amplitude_scale_factor = np.power(10, (dbspl_per_erb - dbspl_power_1khz_erb) / 20)

    # Generate noise signal with inverse rfft, scale to desired dbspl_per_erb
    noise = np.fft.irfft(noise_rfft)
    noise = noise * amplitude_scale_factor
    return noise


def flat_spectrum_noise(sr, dur, dbhzspl=15.0):
    """
    Generates random noise with a maximally flat spectrum.

    Args
    ----
    sr (int): sampling rate in Hz
    dur (float): duration of noise (s)
    dbhzspl (float): power spectral density in units dB/Hz re 20e-6 Pa

    Returns
    -------
    (np.ndarray): noise waveform (Pa)
    """
    # Create flat-spectrum noise in the frequency domain
    fxx = np.ones(int(dur * sr), dtype=np.complex128)
    freqs = np.fft.fftfreq(len(fxx), d=1 / sr)
    pos_idx = np.argwhere(freqs > 0).reshape([-1])
    neg_idx = np.argwhere(freqs < 0).reshape([-1])
    if neg_idx.shape[0] > pos_idx.shape[0]:
        neg_idx = neg_idx[1:]
    phases = np.random.uniform(low=0.0, high=2 * np.pi, size=pos_idx.shape)
    phases = np.cos(phases) + 1j * np.sin(phases)
    fxx[pos_idx] = fxx[pos_idx] * phases
    fxx[neg_idx] = fxx[neg_idx] * np.flip(phases, axis=0)
    x = np.real(np.fft.ifft(fxx))
    # Re-scale to specified PSD (in units dB/Hz SPL)
    # dbhzspl = 10 * np.log10 ( PSD / (20e-6 Pa)^2 ), where PSD has units Pa^2 / Hz
    PSD = np.power(10, (dbhzspl / 10)) * np.square(20e-6)
    A_rms = np.sqrt(PSD * sr / 2)
    return A_rms * x / rms(x)


def spectrally_matched_noise(x, sr):
    """
    Generates random noise with the same power spectrum as a given signal

    Args
    ----
    x (np.ndarray): sound waveform providing the spectrum
    sr (int): sampling rate in Hz

    Returns
    -------
    x_noise (np.ndarray): sound waveform with matching spectrum (Pa)
    """
    x_noise = flat_spectrum_noise(sr, x.shape[0] / sr)
    _, pxx = power_spectrum(x, sr, dbspl=False)
    x_noise = impose_power_spectrum(x_noise, pxx)
    x_noise = rms(x) * x_noise / rms(x_noise)
    return x_noise


def modified_uniform_masking_noise(
    sr,
    dur,
    dbhzspl=15.0,
    atten_start=600.0,
    atten_slope=2.0,
):
    """
    Function for generating modified uniform masking noise as described by
    Bernstein & Oxenham, JASA 117-6 3818 (June 2005). Long-term spectrum
    level is flat below `atten_start` (Hz) and rolls off at `atten_slope`
    (dB/octave) above `atten_start` (Hz).

    Args
    ----
    sr (int): sampling rate of noise (Hz)
    dur (float): duration of noise (s)
    dbhzspl (float): power spectral density (dB/Hz re 20e-6 Pa) below atten_start
    atten_start (float): cutoff frequency for start of attenuation (Hz)
    atten_slope (float): slope in units of dB/octave above atten_start

    Returns
    -------
    (np.ndarray): noise waveform (Pa)
    """
    x = flat_spectrum_noise(sr, dur, dbhzspl=dbhzspl)
    fxx = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1 / sr)
    db_atten = np.zeros_like(freqs)
    nzidx = np.abs(freqs) > 0
    db_atten[nzidx] = -atten_slope * np.log2(np.abs(freqs[nzidx]) / atten_start)
    db_atten[db_atten > 0] = 0
    fxx = fxx * np.power(10, (db_atten / 20))
    return np.real(np.fft.ifft(fxx))


def get_inharmonic_jitter_pattern(jitter=0.5, n_harm=30, f0=200.0, min_diff=30.0):
    """
    Returns an inharmonic jitter pattern, as described in
    McPherson et al. (2022, Attention, Perception, & Psychophysics):

    Jittering was accomplished by sampling a jitter value from the distribution
    U(-0.5, 0.5), multiplying by the f0, then adding the resulting value to the
    frequency of the respective harmonic, constraining adjacent components to be
    separated by at least 30 Hz (via rejection sampling) in order to avoid salient
    beating. All harmonics above the fundamental were jittered in this way.
    """
    h = np.arange(1, n_harm + 1)
    acceptable = False
    while not acceptable:
        inharmonic_jitter_pattern = np.random.uniform(
            low=-jitter,
            high=jitter,
            size=(n_harm),
        )
        inharmonic_jitter_pattern[0] = 0
        f = f0 * (h + inharmonic_jitter_pattern)
        acceptable = np.diff(f).min() >= min_diff
    return inharmonic_jitter_pattern
