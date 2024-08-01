import glob
import os
import re

import h5py
import numpy as np
import torch
import torchaudio
import torchmetrics
import torchvision

import util_filters
import util_misc


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
            util_filters._batching_check(y, self.fir)
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
            util_filters._batching_check(x, self.b)
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
        cfs=util_filters.erbspace(8e1, 8e3, 50),
        dtype=torch.float32,
        **kwargs,
    ):
        """ """
        super().__init__()
        if fir_dur is None:
            filter_coeffs = util_filters.gammatone_filter_coeffs(
                sr=sr,
                cfs=cfs,
                **kwargs,
            )
            self.fbs = [IIRFilterbank(**ba, dtype=dtype) for ba in filter_coeffs]
        else:
            fir = util_filters.gammatone_filter_fir(
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


class MiddleEarFilter(FIRFilterbank):
    def __init__(
        self,
        sr=20e3,
        order=512,
        min_phase=True,
        dtype=torch.float32,
    ):
        """ """
        fir = util_filters.middle_ear_filter_fir(
            sr=sr,
            order=order,
            min_phase=min_phase,
        )
        super().__init__(fir, dtype=dtype)


class DRNLFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=util_filters.erbspace(8e1, 8e3, 50),
        middle_ear=True,
        lin_ngt=2,
        lin_nlp=4,
        lin_cfs=[-0.06762, 1.01679],
        lin_bw=[0.03728, 0.78563],
        lin_gain=[4.20405, -0.47909],
        lin_lp_cutoff=[-0.06762, 1.01679],
        nlin_ngt_before=3,
        nlin_ngt_after=None,
        nlin_nlp=3,
        nlin_cfs_before=[-0.05252, 1.01650],
        nlin_cfs_after=None,
        nlin_bw_before=[-0.03193, 0.77426],
        nlin_bw_after=None,
        nlin_lp_cutoff=[-0.05252, 1.01650],
        nlin_a=[1.40298, 0.81916],
        nlin_b=[1.61912, -0.81867],
        nlin_c=[np.log10(0.25), 0],
        nlin_d=1,
        filter_type="gtf",
        iir_output="sos",
        dtype=torch.float32,
    ):
        """
        Dual-resonance non-linear filterbank.
        Same as `lopezpoveda2001.m` in the AMT.
        """
        super().__init__()
        if middle_ear:
            self.middle_ear_filter = MiddleEarFilter(
                sr=sr,
                order=512,
                min_phase=True,
                dtype=dtype,
            )
        else:
            self.middle_ear_filter = None
        if nlin_ngt_after is None:
            nlin_ngt_after = nlin_ngt_before
        if nlin_cfs_after is None:
            nlin_cfs_after = nlin_cfs_before
        if nlin_bw_after is None:
            nlin_bw_after = nlin_bw_before

        def polfun(x, par):
            return 10 ** par[0] * x ** par[1]

        lin_cfs = polfun(cfs, lin_cfs)
        lin_bw = polfun(cfs, lin_bw)
        lin_lp_cutoff = polfun(cfs, lin_lp_cutoff)
        lin_gain = polfun(cfs, lin_gain)
        nlin_cfs_before = polfun(cfs, nlin_cfs_before)
        nlin_cfs_after = polfun(cfs, nlin_cfs_after)
        nlin_bw_before = polfun(cfs, nlin_bw_before)
        nlin_bw_after = polfun(cfs, nlin_bw_after)
        nlin_lp_cutoff = polfun(cfs, nlin_lp_cutoff)
        nlin_a = polfun(cfs, nlin_a)
        nlin_b = polfun(cfs, nlin_b)
        nlin_c = polfun(cfs, nlin_c)
        if fir_dur is None:
            kwargs_gammatone = {
                "filter_type": filter_type,
                "iir_output": iir_output,
                "dtype": dtype,
            }
        else:
            kwargs_gammatone = {
                "dtype": dtype,
            }
        self.gtf_lin = GammatoneFilterbank(
            sr=sr,
            cfs=lin_cfs,
            fir_dur=fir_dur,
            order=lin_ngt,
            bw_mult=lin_bw / util_filters.aud_filt_bw(lin_cfs),
            **kwargs_gammatone,
        )
        self.gtf_nlin_before = GammatoneFilterbank(
            sr=sr,
            cfs=nlin_cfs_before,
            fir_dur=fir_dur,
            order=nlin_ngt_before,
            bw_mult=nlin_bw_before / util_filters.aud_filt_bw(nlin_cfs_before),
            **kwargs_gammatone,
        )
        self.gtf_nlin_after = GammatoneFilterbank(
            sr=sr,
            cfs=nlin_cfs_after,
            fir_dur=fir_dur,
            order=nlin_ngt_after,
            bw_mult=nlin_bw_after / util_filters.aud_filt_bw(nlin_cfs_after),
            **kwargs_gammatone,
        )
        self.lpf_lin = IIRFilterbank(
            *util_filters.butter(2, lin_lp_cutoff / (sr / 2)),
            dtype=dtype,
        )
        self.lpf_nlin = IIRFilterbank(
            *util_filters.butter(2, nlin_lp_cutoff / (sr / 2)),
            dtype=dtype,
        )
        self.register_buffer("lin_gain", torch.tensor(lin_gain, dtype=dtype))
        self.register_buffer("nlin_a", torch.tensor(nlin_a, dtype=dtype))
        self.register_buffer("nlin_b", torch.tensor(nlin_b, dtype=dtype))
        self.register_buffer("nlin_c", torch.tensor(nlin_c, dtype=dtype))
        self.register_buffer("nlin_d", torch.tensor(nlin_d, dtype=dtype))
        self.register_buffer("lin_nlp", torch.tensor(lin_nlp))
        self.register_buffer("nlin_nlp", torch.tensor(nlin_nlp))

    def forward(self, x):
        """ """
        unbatched = x.ndim == 1
        if unbatched:
            x = x[None, :]
        # Middle ear filter
        if self.middle_ear_filter is not None:
            x = x * 10 ** ((93.98 - 100) / 20)
            x = self.middle_ear_filter(x)
        y_lin = torch.einsum("bi,bj->bij", (self.lin_gain[None, :], x))
        # Linear path gammatone filterbank
        y_lin = self.gtf_lin(y_lin, batching=True)
        # Linear path lowpass filter
        for _ in range(self.lin_nlp):
            y_lin = self.lpf_lin(y_lin, batching=True)
        # Nonlinear path gammatone filterbank (before nonlinearity)
        y_nlin = self.gtf_nlin_before(x, batching=False)
        # Broken stick nonlinearity
        y_nlin = y_nlin.sign() * torch.minimum(
            self.nlin_a[:, None] * y_nlin.abs() ** self.nlin_d,
            self.nlin_b[:, None] * y_nlin.abs() ** self.nlin_c[:, None],
        )
        # Nonlinear path gammatone filterbank (after nonlinearity)
        y_nlin = self.gtf_nlin_after(y_nlin, batching=True)
        # Nonlinear path lowpass filter
        for _ in range(self.nlin_nlp):
            y_nlin = self.lpf_nlin(y_nlin, batching=True)
        # Combine linear and nonlinear path
        y = y_lin + y_nlin
        if unbatched:
            y = y[0, :]
        return y


class IHCTransduction(torch.nn.Module):
    def __init__(
        self,
        compression_power=None,
        compression_dbspl_min=None,
        compression_dbspl_max=None,
        rectify=True,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        if compression_power is not None:
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
        else:
            self.compression_power = None
        if compression_dbspl_min is not None:
            self.compression_pa_min = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_min / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_min = torch.tensor(-np.inf, dtype=dtype)
        if compression_dbspl_max is not None:
            self.compression_pa_max = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_max / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_max = torch.tensor(np.inf, dtype=dtype)
        self.rectify = rectify

    def forward(self, x):
        """ """
        if self.compression_power is not None:
            # Broken-stick compression (power compression between
            # compression_dbspl_min and compression_dbspl_max)
            if self.compression_power.ndim > 0:
                if not self.compression_power.ndim == x.ndim:
                    shape = [1 for _ in range(x.ndim)]
                    shape[-2] = x.shape[-2]
                    self.compression_power = self.compression_power.view(*shape)
            abs_x = torch.abs(x)
            IDX_COMPRESSION = torch.logical_and(
                abs_x >= self.compression_pa_min,
                abs_x < self.compression_pa_max,
            )
            IDX_AMPLIFICATION = abs_x < self.compression_pa_min
            x = torch.sign(x) * torch.where(
                IDX_COMPRESSION,
                abs_x**self.compression_power,
                torch.where(
                    IDX_AMPLIFICATION,
                    abs_x * (self.compression_pa_min ** (self.compression_power - 1)),
                    abs_x,
                ),
            )
        if self.rectify:
            # Half-wave rectification
            x = torch.nn.functional.relu(x, inplace=False)
        return x


class IHCLowpassFilter(FIRFilterbank):
    def __init__(
        self,
        sr_input=20e3,
        sr_output=10e3,
        fir_dur=0.05,
        cutoff=3e3,
        order=7,
        dtype=torch.float32,
    ):
        """ """
        fir = util_filters.ihc_lowpass_filter_fir(
            sr=sr_input,
            fir_dur=fir_dur,
            cutoff=cutoff,
            order=order,
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)


class ModulationFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=10e3,
        fir_dur=0.5,
        lmf=0,
        umf=1500,
        Q=2,
        bw=2,
        flatten_channels=True,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.flatten_channels = flatten_channels
        if fir_dur is None:
            filter_coeffs = util_filters.mfbtd_filter_coeffs(
                sr=sr,
                lmf=lmf,
                umf=umf,
                Q=Q,
                bw=bw,
            )
            self.fbs = [IIRFilterbank(**ba, dtype=dtype) for ba in filter_coeffs]
        else:
            fir = util_filters.mfbtd_filter_fir(
                sr=sr,
                fir_dur=fir_dur,
                lmf=lmf,
                umf=umf,
                Q=Q,
                bw=bw,
            )
            self.fbs = [FIRFilterbank(fir, dtype=dtype)]
        self.fbs = torch.nn.ModuleList(self.fbs)

    def forward(self, x, batching=False):
        """ """
        for itr, fb in enumerate(self.fbs):
            x = fb(x, batching=batching or itr > 0)
        if self.flatten_channels:
            assert x.ndim == 5, "expected shape [batch, spont, freq, mod, time]"
            x = torch.transpose(x, -2, -3)
            x = torch.flatten(x, start_dim=-4, end_dim=-3)
        return x


class Hilbert(torch.nn.Module):
    def __init__(self, dim=-1):
        """
        Compute the analytic signal, using the Hilbert transform
        (torch implementation of `scipy.signal.hilbert`)
        """
        super().__init__()
        self.dim = dim

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        """ """
        n = x.shape[self.dim]
        X = torch.fft.fft(x, n=n, dim=self.dim, norm=None)
        h = torch.zeros(n, dtype=X.dtype).to(X.device)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        ind = [np.newaxis] * x.ndim
        ind[self.dim] = slice(None)
        return torch.fft.ifft(
            X * h[ind],
            n=n,
            dim=self.dim,
            norm=None,
        )


class HilbertEnvelope(torch.nn.Module):
    def __init__(self, **args):
        """ """
        super().__init__()
        self.hilbert = Hilbert(**args)

    def forward(self, x):
        return torch.abs(self.hilbert(x))


class SigmoidRateLevelFunction(torch.nn.Module):
    def __init__(
        self,
        rate_spont=[0.0, 0.0, 0.0],
        rate_max=[250.0, 250.0, 250.0],
        threshold=[0.0, 12.0, 28.0],
        dynamic_range=[20.0, 40.0, 80.0],
        dynamic_range_interval=0.95,
        compression_power=None,
        compression_power_default=0.3,
        envelope_mode=True,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        if compression_power is not None:
            # Explicitly incorporate power compression into the rate-level function
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
            if compression_power_default is not None:
                # Adjust threshold and dynamic range for `compression_power_default`
                shift = 20 * np.log10(20e-6 ** (compression_power_default - 1))
                threshold = np.array(threshold) * compression_power_default + shift
                dynamic_range = np.array(dynamic_range) * compression_power_default
        else:
            self.compression_power = None
        # Check arguments and register tensors with channel-specific shapes
        assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
        argument_lengths = [
            len(rate_spont),
            len(rate_max),
            len(threshold),
            len(dynamic_range),
        ]
        channel_specific_size = [1, max(argument_lengths), 1, 1]
        rate_spont = self.resize(rate_spont, channel_specific_size)
        rate_max = self.resize(rate_max, channel_specific_size)
        threshold = self.resize(threshold, channel_specific_size)
        dynamic_range = self.resize(dynamic_range, channel_specific_size)
        y_threshold = (1 - dynamic_range_interval) / 2
        k = np.log((1 / y_threshold) - 1) / (dynamic_range / 2)
        x0 = threshold - (np.log((1 / y_threshold) - 1) / (-k))
        self.register_buffer(
            "rate_spont", torch.tensor(rate_spont, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "rate_max", torch.tensor(rate_max, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "threshold", torch.tensor(threshold, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "dynamic_range", torch.tensor(dynamic_range, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "dynamic_range_interval",
            torch.tensor(dynamic_range_interval, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "y_threshold", torch.tensor(y_threshold, dtype=dtype), persistent=True
        )
        self.register_buffer("k", torch.tensor(k, dtype=dtype), persistent=True)
        self.register_buffer("x0", torch.tensor(x0, dtype=dtype), persistent=True)
        # Construct envelope extraction function if needed
        self.envelope_mode = envelope_mode
        if self.envelope_mode:
            self.envelope_function = HilbertEnvelope(dim=-1)

    def resize(self, x, shape):
        """ """
        x = np.array(x).reshape([-1])
        if len(x) == 1:
            x = np.full(shape, x[0])
        else:
            x = np.reshape(x, shape)
        return x

    def forward(self, tensor_subbands):
        """ """
        while tensor_subbands.ndim < 4:
            tensor_subbands = tensor_subbands.unsqueeze(-3)
        if self.envelope_mode:
            # Subband envelopes are passed through sigmoid and recombined with TFS
            tensor_env = self.envelope_function(tensor_subbands)
            tensor_tfs = torch.divide(tensor_subbands, tensor_env)
            tensor_tfs = torch.where(
                torch.isfinite(tensor_tfs), tensor_tfs, tensor_subbands
            )
            tensor_pa = tensor_env
        else:
            # Subbands are passed through sigmoid (alters spike timing at high levels)
            tensor_pa = tensor_subbands
        if self.compression_power is not None:
            # Apply power compression (supports frequency-specific power compression)
            tensor_pa = tensor_pa ** self.compression_power.view(1, 1, -1, 1)
        # Compute sigmoid function with tensor broadcasting
        x = 20.0 * torch.log(tensor_pa / 20e-6) / np.log(10)
        y = 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))
        if self.envelope_mode:
            y = y * tensor_tfs
        tensor_rates = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return tensor_rates


class BinomialSpikeGenerator(torch.nn.Module):
    def __init__(
        self,
        sr=10000,
        mode="approx",
        n_per_channel=[384, 160, 96],
        n_per_step=48,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.sr = sr
        self.mode = mode
        self.n_per_step = n_per_step
        self.register_buffer(
            "n_per_channel",
            torch.tensor(n_per_channel, dtype=dtype).view([-1]),
            persistent=True,
        )

    def forward(self, tensor_rates):
        """ """
        msg = "Requires input shape [batch, channel, freq, time]"
        assert tensor_rates.ndim == 4, msg
        tensor_probs = tensor_rates / self.sr
        if self.mode == "approx":
            # Sample from normal approximation of binomial distribution
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            sample = torch.distributions.normal.Normal(
                loc=n * p,
                scale=torch.sqrt(n * p * (1 - p)),
                validate_args=False,
            ).rsample()
            tensor_spike_counts = torch.round(torch.nn.functional.relu(sample))
        elif self.mode == "exact":
            # Binomial distribution implemented as sum of Bernoulli random variables
            n = self.n_per_channel
            p = tensor_probs
            assert (n.ndim == 1) and (n.shape[0] == p.shape[1])
            tensor_spike_counts = torch.zeros_like(p)
            for channel in range(p.shape[1]):
                total = int(n[channel])
                count = 0
                while count < total:
                    n_sample_per_step = min(self.n_per_step, total - count)
                    sample = (
                        torch.rand(
                            size=(n_sample_per_step, *p[:, channel, :, :].shape),
                            device=self.n_per_channel.device,
                        )
                        < p[None, :, channel, :, :]
                    )
                    tensor_spike_counts[:, channel, :, :] += sample.sum(dim=0)
                    count += n_sample_per_step
        elif self.mode == "additive":
            # Replace sampling with additive noise to enable back-propagation
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            noise = torch.randn_like(p) / n
            tensor_spike_counts = torch.nn.functional.relu((p + noise) * n)
        else:
            raise NotImplementedError(f"mode=`{self.mode}` is not implemented")
        return tensor_spike_counts


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


class MultitaskCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        task_list=[],
        task_weights=None,
        ignore_index=None,
    ):
        """ """
        super().__init__()
        self.task_list = task_list
        self.task_weights = task_weights
        self.ignore_index = ignore_index
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=None,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=0.0,
        )
        if self.task_weights is None:
            self.task_weights = {k: torch.tensor(1.0) for k in self.task_list}

    def forward(self, logits, targets, return_task_loss=True):
        """ """
        task_loss = {}
        loss = 0
        task_weights = self.task_weights
        if (isinstance(task_weights, str)) and (task_weights == "random"):
            tmp = logits[self.task_list[0]]
            w = torch.rand(
                size=(len(self.task_list), tmp.shape[0]),
                device=tmp.device,
            )
            w = len(self.task_list) * w / torch.sum(w, dim=0, keepdim=True)
            task_weights = {k: w[itr] for itr, k in enumerate(self.task_list)}
        for k in self.task_list:
            task_loss[k] = self.loss_function(logits[k], targets[k])
            IDX = torch.isfinite(task_loss[k])
            if task_weights[k].shape == IDX.shape:
                tmp = task_weights[k][IDX] * task_loss[k][IDX]
            else:
                tmp = task_weights[k] * task_loss[k][IDX]
            task_loss[k] = torch.nan_to_num(
                torch.mean(tmp),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            loss = loss + task_loss[k]
        if return_task_loss:
            return loss, task_loss
        return loss


class MultitaskMulticlassAccuracy(torchmetrics.wrappers.MultitaskWrapper):
    def __init__(self, heads={}, **kwargs):
        """ """
        task_metrics = {}
        for k, num_classes in heads.items():
            task_metrics[k] = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, **kwargs
            )
        super().__init__(task_metrics)


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


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        regex_filenames,
        keys=None,
    ):
        """ """
        super().__init__()
        self.filenames = list(glob.glob(regex_filenames))
        self.files = [h5py.File(_, "r") for _ in self.filenames]
        self.keys = keys
        if self.keys is None:
            self.keys = util_misc.get_hdf5_dataset_key_list(self.files[0])
            self.keys = [
                key
                for key in self.keys
                if np.issubdtype(self.files[0][key].dtype, np.number)
            ]
        n = [self.files[0][key].shape[0] for key in self.keys]
        assert len(np.unique(n)) == 1, "dataset keys must have same length"
        self.n_per_file = [f[self.keys[0]].shape[0] for f in self.files]
        self.index_map = []
        for index_file in range(len(self.files)):
            for index_data in range(self.n_per_file[index_file]):
                self.index_map.append((index_file, index_data))

    def __getitem__(self, index):
        """ """
        index_file, index_data = self.index_map[index]
        return {key: self.files[index_file][key][index_data] for key in self.keys}

    def __len__(self):
        """ """
        return len(self.index_map)

    def __del__(self):
        """ """
        for f in self.files:
            f.close()
