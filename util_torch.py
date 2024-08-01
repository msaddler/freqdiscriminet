import glob
import os
import re

import numpy as np
import torch
import torchaudio
import torchvision


def rms(x, dim=None, keepdim=False):
    """
    Compute root-mean-squared amplitude of `x` along axis `dim`
    """
    out = torch.sqrt(
        torch.mean(
            torch.square(x),
            dim=dim,
            keepdim=keepdim,
        ),
    )
    return out


def set_dbspl(x, dbspl, mean_subtract=True, dim=None, keepdim=False):
    """
    Set root-mean-squared amplitude of `x` along axis `dim`
    to the sound pressure level `dbspl` (in dB re 20e-6 Pa)
    """
    if mean_subtract:
        x = x - torch.mean(x, dim=dim, keepdim=keepdim)
    rms_output = 20e-6 * torch.pow(10, dbspl / 20)
    rms_input = rms(x, dim=dim, keepdim=keepdim)
    if torch.is_nonzero(rms_input):
        x = rms_output * (x / rms_input)
    return x


def pad_or_trim_to_len(x, n, dim=-1, kwargs_pad={}):
    """
    Symmetrically pad or trim `x` to length `n` along axis `dim`
    """
    n_orig = int(x.shape[dim])
    if n_orig < n:
        n0 = (n - n_orig) // 2
        n1 = (n - n_orig) - n0
        pad = []
        for _ in range(x.ndim):
            pad.extend([n0, n1] if _ == dim else [0, 0])
        x = torch.nn.functional.pad(x, pad, **kwargs_pad)
    if n_orig > n:
        n0 = (n_orig - n) // 2
        ind = [slice(None)] * x.ndim
        ind[dim] = slice(n0, n0 + n)
        x = x[ind]
    return x


class FIRFilterbank(torch.nn.Module):
    def __init__(self, fir, dtype=torch.float32, **kwargs_conv1d):
        """
        FIR filterbank

        Args
        ----
        fir (list or np.ndarray or torch.Tensor):
            Filter coefficients. Shape (n_taps,) or (n_filters, n_taps)
        dtype (torch.dtype):
            Data type to cast `fir` to in case it is not a `torch.Tensor`
        kwargs_conv1d (kwargs):
            Keyword arguments passed on to torch.nn.functional.conv1d
            (must not include `groups`, which is used for batching)
        """
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                "fir must be list, np.ndarray or torch.Tensor, got "
                f"{fir.__class__.__name__}"
            )
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(
                "fir must be one- or two-dimensional with shape (n_taps,) or "
                f"(n_filters, n_taps), got shape {fir.shape}"
            )
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Filter input signal

        Args
        ----
        x (torch.Tensor): Input signal
        batching (bool):
            If `True`, the input is assumed to have shape (..., n_filters, time)
            and each channel is filtered with its own filter

        Returns
        -------
        y (torch.Tensor): Filtered signal
        """
        y = x
        if batching:
            _batching_check(y, self.fir)
        else:
            y = y.unsqueeze(-2)
        unflatten_shape = y.shape[:-2]
        y = torch.flatten(y, start_dim=0, end_dim=-2 - 1)
        y = torch.nn.functional.conv1d(
            input=torch.nn.functional.pad(y, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=y.shape[-2] if batching else 1,
        )
        y = torch.unflatten(y, 0, unflatten_shape)
        if self.fir.ndim == 1:
            y = y.squeeze(-2)
        return y


class IIRFilterbank(torch.nn.Module):
    def __init__(self, b, a, dtype=torch.float32):
        """
        IIR filterbank

        Args
        ----
        b (list or np.ndarray or torch.Tensor):
            Numerator coefficients. Shape (n_taps,) or (n_filters, n_taps)
        a (list or np.ndarray or torch.Tensor)
            Denominator coefficients. Same shape as `b`
        dtype (torch.dtype):
            Data type to cast `a` and `b` to in case they are not `torch.Tensor`
        """
        super().__init__()
        if not isinstance(b, (list, np.ndarray, torch.Tensor)) or not isinstance(
            a, (list, np.ndarray, torch.Tensor)
        ):
            raise TypeError(
                "b and a must be list, np.ndarray or torch.Tensor, got "
                f"{b.__class__.__name__} and {a.__class__.__name__}"
            )
        if isinstance(b, (list, np.ndarray)):
            b = torch.tensor(b, dtype=dtype)
        if isinstance(a, (list, np.ndarray)):
            a = torch.tensor(a, dtype=dtype)
        if a.ndim == b.ndim == 1 or a.ndim == b.ndim == 2:
            if b.shape[-1] < a.shape[-1]:
                b = torch.nn.functional.pad(b, (0, a.shape[-1] - b.shape[-1]))
            elif b.shape[-1] > a.shape[-1]:
                a = torch.nn.functional.pad(a, (0, b.shape[-1] - a.shape[-1]))
        if b.shape != a.shape or b.ndim not in [1, 2]:
            raise ValueError(
                "b and a must have the same one- or two-dimensional shape (n_taps,) or "
                f"(n_filters, n_taps), got shapes {b.shape} and {a.shape}"
            )
        self.register_buffer("b", b)
        self.register_buffer("a", a)

    def forward(self, x, batching=False):
        """
        Filter input signal

        Args
        ----
        x (torch.Tensor): Input signal
        batching (bool):
            If `True`, the input is assumed to have shape (..., n_filters, time)
            and each channel is filtered with its own filter

        Returns
        -------
        y (torch.Tensor): Filtered signal
        """
        if batching:
            _batching_check(x, self.b)
        y = torchaudio.functional.lfilter(
            x,
            self.a.view(-1, self.a.shape[-1]),
            self.b.view(-1, self.b.shape[-1]),
            batching=batching,
            clamp=False,
        )
        return y


class GammatoneFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=None,
        dtype=torch.float32,
        **kwargs,
    ):
        """ """
        super().__init__()
        if fir_dur is None:
            filter_coeffs = gammatone_filter_coeffs(
                sr=sr,
                cfs=cfs,
                **kwargs,
            )
            self.fbs = [IIRFilterbank(**ba, dtype=dtype) for ba in filter_coeffs]
        else:
            fir = gammatone_filter_fir(
                sr=sr,
                fir_dur=fir_dur,
                cfs=cfs,
                **kwargs,
            )
            self.fbs = [FIRFilterbank(fir, dtype=dtype)]
        self.fbs = torch.nn.ModuleList(self.fbs)

    def forward(self, x, batching=False):
        """ """
        for itr, fb in enumerate(self.fbs):
            x = fb(x, batching=batching or itr > 0)
        return x


def aud_filt_bw(cf):
    """
    Critical bandwidth of auditory filter at given center frequency.
    Same as `audfiltbw.m` in the AMT.
    """
    return 24.7 + cf / 9.265


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
        print(
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


def calculate_same_pad(input_dim, kernel_dim, stride):
    """ """
    pad = (np.ceil(input_dim / stride) - 1) * stride + (kernel_dim - 1) + 1 - input_dim
    return int(max(pad, 0))


def custom_conv_pad(x, pad, weight=None, stride=None, **kwargs):
    """ """
    msg = f"Expected input shape [batch, channel, freq, time]: received {x.shape=}"
    assert x.ndim == 4, msg
    msg = f"Expected tuple or integers or a string: received {pad=}"
    assert isinstance(pad, (tuple, str)), msg
    if isinstance(pad, str):
        if pad.lower() == "same":
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() in ["same_freq", "valid_time"]:
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = 0
        elif pad.lower() in ["same_time", "valid_freq"]:
            pad_f = 0
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() == "valid":
            pad_f = 0
            pad_t = 0
        else:
            raise ValueError(f"Mode `{pad=}` is not recognized")
        pad = (pad_t // 2, pad_t - pad_t // 2, pad_f // 2, pad_f - pad_f // 2)
    return torch.nn.functional.pad(x, pad, **kwargs)


class ChannelwiseConv2d(torch.nn.Module):
    def __init__(self, kernel, pad=(0, 0), stride=(1, 1), dtype=torch.float32):
        """ """
        super().__init__()
        assert kernel.ndim == 2, "Expected kernel with shape [freq, time]"
        self.register_buffer(
            "weight",
            torch.tensor(kernel[None, None, :, :], dtype=dtype),
            persistent=True,
        )
        self.pad = pad
        self.stride = stride

    def forward(self, x):
        """ """
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant",
            value=0,
        )
        y = y.view(-1, 1, *y.shape[-2:])
        y = torch.nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding="valid",
            dilation=1,
            groups=1,
        )
        y = y.view(*x.shape[:-2], *y.shape[-2:])
        return y


class HanningPooling(ChannelwiseConv2d):
    def __init__(
        self,
        stride=[1, 1],
        kernel_size=[1, 1],
        padding="same",
        sqrt_window=False,
        normalize=False,
        dtype=torch.float32,
    ):
        """ """
        kernel = torch.ones(kernel_size, dtype=dtype)
        for dim, m in enumerate(kernel_size):
            shape = [-1 if _ == dim else 1 for _ in range(len(kernel_size))]
            kernel = kernel * torch.signal.windows.hann(
                m,
                sym=True,
                dtype=dtype,
            ).reshape(shape)
        if sqrt_window:
            kernel = torch.sqrt(kernel)
        if normalize:
            kernel = kernel / torch.sum(kernel)
        super().__init__(kernel.numpy(), pad=padding, stride=stride, dtype=dtype)


class CustomPaddedConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """ """
        self.pad = kwargs.get("padding", 0)
        if isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        if isinstance(self.pad, str):
            kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """ """
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
            value=0,
        )
        y = torch.nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y


class CustomFlatten(torch.nn.Module):
    def __init__(
        self,
        start_dim=0,
        end_dim=-1,
        permute_dims=None,
    ):
        """ """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.permute_dims = permute_dims

    def forward(self, x):
        """ """
        if self.permute_dims is not None:
            x = torch.permute(x, dims=self.permute_dims)
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class CustomNorm(torch.nn.Module):
    def __init__(
        self,
        input_shape=[None, None, None, None],
        dim_affine=None,
        dim_norm=None,
        correction=1,
        eps=1e-05,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.input_shape = input_shape
        self.dim_affine = dim_affine
        self.dim_norm = dim_norm
        self.correction = correction
        self.eps = eps
        self.dtype = dtype
        if self.dim_affine is not None:
            msg = "`input_shape` is required when `dim_affine` is not None"
            assert self.input_shape is not None, msg
            size = input_shape[self.dim_affine]
            self.shape = [1 for _ in self.input_shape]
            self.shape[self.dim_affine] = input_shape[self.dim_affine]
            self.weight = torch.nn.parameter.Parameter(
                data=torch.squeeze(torch.ones(size, dtype=self.dtype)),
                requires_grad=True,
            )
            self.bias = torch.nn.parameter.Parameter(
                data=torch.squeeze(torch.zeros(size, dtype=self.dtype)),
                requires_grad=True,
            )

    def forward(self, x):
        """ """
        x_var, x_mean = torch.var_mean(
            x, dim=self.dim_norm, correction=self.correction, keepdim=True
        )
        y = (x - x_mean) / torch.sqrt(x_var + self.eps)
        if self.dim_affine is not None:
            w = self.weight.view(self.shape)
            b = self.bias.view(self.shape)
            y = (y * w) + b
        return y


class Permute(torch.nn.Module):
    def __init__(self, dims=None):
        """ """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """ """
        return torch.permute(x, dims=self.dims)


class Reshape(torch.nn.Module):
    def __init__(self, shape=None):
        """ """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """ """
        return torch.reshape(x, shape=self.shape)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim=None):
        """ """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """ """
        return torch.unsqueeze(x, dim=self.dim)


class RandomSlice(torch.nn.Module):
    def __init__(self, size=[50, 20000], buffer=[0, 0], **kwargs):
        """ """
        super().__init__()
        self.size = size
        self.pre_crop_slice = []
        for b in buffer:
            if b is None:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, int) and b > 0:
                self.pre_crop_slice.append(slice(b, -b))
            elif isinstance(b, int) and b == 0:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, (tuple, list)):
                self.pre_crop_slice.append(slice(*b))
        self.crop = torchvision.transforms.RandomCrop(size=self.size, **kwargs)

    def forward(self, x):
        """ """
        return self.crop(x[..., *self.pre_crop_slice])


def save_model_checkpoint(
    model,
    dir_model=None,
    step=None,
    fn_ckpt="ckpt_BEST.pt",
    fn_ckpt_step="ckpt_{:04d}.pt",
    **kwargs,
):
    """ """
    filename = fn_ckpt
    if step is not None:
        filename = fn_ckpt_step.format(step)
    if dir_model is not None:
        filename = os.path.join(dir_model, filename)
    torch.save(model.state_dict(), filename + "~", **kwargs)
    os.rename(filename + "~", filename)
    print(f"[save_model_checkpoint] {filename}")
    return filename


def load_model_checkpoint(
    model,
    dir_model=None,
    step=None,
    fn_ckpt="ckpt_BEST.pt",
    fn_ckpt_step="ckpt_{:04d}.pt",
    **kwargs,
):
    """ """
    filename = fn_ckpt
    if step is not None:
        filename = fn_ckpt_step
    if dir_model is not None:
        filename = os.path.join(dir_model, filename)
    if (step is not None) and (step >= 0):
        # Load checkpoint specified by step
        filename = filename.format(step)
    elif step is not None:
        # Load recent checkpoint if step < 0
        unformatted = filename.replace(
            filename[filename.find("{") + 1 : filename.find("}")],
            "",
        )
        list_filename = []
        list_step = []
        for filename in glob.glob(unformatted.format("*")):
            output = re.findall(r"\d+", os.path.basename(filename))
            if len(output) == 1:
                list_filename.append(filename)
                list_step.append(int(output[0]))
        if len(list_filename) == 0:
            print("[load_model_checkpoint] No prior checkpoint found")
            return 0
        list_filename = [list_filename[_] for _ in np.argsort(list_step)]
        filename = list_filename[step]
    state_dict = torch.load(filename, **kwargs)
    model.load_state_dict(state_dict, strict=True, assign=False)
    print(f"[load_model_checkpoint] {filename}")
    if (step is not None) and (step < 0):
        return list_step[step]
    return filename


def set_trainable(
    model,
    trainable=False,
    trainable_layers=None,
    trainable_batchnorm=None,
    verbose=True,
):
    """ """
    if not trainable:
        model.train(trainable)
        trainable_params = []
    elif trainable_layers is None:
        model.train(trainable)
        trainable_params = list(model.parameters())
    else:
        trainable_layers_names = []
        trainable_params_names = []
        trainable_params = []
        model.train(False)
        if verbose:
            print(f"[set_trainable] {trainable_layers=}")
        for m_name, m in model.named_modules():
            msg = f"invalid trainable_layers ({m_name} -> multiple matches)"
            for pattern in trainable_layers:
                if pattern in m_name:
                    assert m_name not in trainable_layers_names, msg
                    trainable_layers_names.append(m_name)
                    m.train(trainable)
                    if verbose:
                        print(f"{m_name} ('{pattern}') -> {m.training}")
                    for p_basename, p in m.named_parameters():
                        p_name = f"{m_name}.{p_basename}"
                        assert p_name not in trainable_params_names, msg
                        trainable_params_names.append(p_name)
                        trainable_params.append(p)
                        if verbose:
                            print(f"|__ {p_name} {p.shape}")
            if trainable_batchnorm is not None:
                if "batchnorm" in str(type(m)).lower():
                    m.train(trainable_batchnorm)
                    if verbose:
                        print(f"{m_name} ({trainable_batchnorm=}) -> {m.training}")
        if verbose:
            print(f"[set_trainable] {len(trainable_layers_names)=}")
    if verbose:
        print(f"[set_trainable] {trainable} -> {len(trainable_params)=}")
    return trainable_params


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
