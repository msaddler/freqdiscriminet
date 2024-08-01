import itertools
import sys

import numpy as np
import torch

import util_torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sr=40e3,
        dur=0.200,
        dur_ramp=0.020,
        dur_padded=0.250,
        f_min=2e2,
        f_max=1e4,
        f_list=None,
        interval_min=1e-5,
        interval_max=1e-0,
        interval_list=None,
        dbspl_min=37.0,
        dbspl_max=43.0,
        dbspl_fixed=False,
        phase_fixed=0.0,
    ):
        """ """
        self.sr = sr
        self.dur = dur
        self.dur_ramp = dur_ramp
        self.dur_padded = dur_padded
        self.f_min = torch.tensor(f_min)
        self.f_max = torch.tensor(f_max)
        self.dbspl_min = dbspl_min
        self.dbspl_max = dbspl_max
        self.dbspl_fixed = dbspl_fixed
        self.phase_fixed = phase_fixed
        self.phase = None
        if isinstance(self.phase_fixed, float):
            self.phase = torch.tensor(self.phase_fixed, dtype=torch.float32)
        self.eval_mode = (f_list is not None) and (interval_list is not None)
        if self.eval_mode:
            tmp = np.array(list(itertools.product(f_list, interval_list)))
            self.f_list = torch.tensor(tmp[:, 0])
            self.interval_list = torch.tensor(tmp[:, 1])
        else:
            self.dist_log_f = torch.distributions.Uniform(
                low=torch.log(self.f_min),
                high=torch.log(self.f_max),
            )
            self.dist_log_interval = torch.distributions.Uniform(
                low=torch.log(torch.tensor(interval_min)),
                high=torch.log(torch.tensor(interval_max)),
            )
            self.dist_label = torch.distributions.bernoulli.Bernoulli(
                probs=0.5,
            )
        self.dist_dbspl = torch.distributions.Uniform(
            low=self.dbspl_min,
            high=self.dbspl_max,
        )
        self.dist_phase = torch.distributions.Uniform(
            low=0,
            high=2 * np.pi,
        )
        self.t = torch.arange(0, self.dur + self.dur_ramp, 1 / sr)
        self.ramp = torch.signal.windows.hann(int(2 * sr * self.dur_ramp))
        self.ramp_rise_fall = torch.concat(
            [
                self.ramp[: int(sr * self.dur_ramp)],
                torch.ones(int(np.round(sr * (self.dur - self.dur_ramp)))),
                self.ramp[-int(sr * self.dur_ramp) :],
            ]
        )
        if self.dur_padded is not None:
            self.n_pad = int(sr * self.dur_padded) - self.t.shape[0]

    def generate_stim(self, f0, f1, dbspl0, dbspl1, phase0, phase1):
        """ """
        assert (f0 < self.sr / 2) and (f1 < self.sr / 2)
        x0 = torch.cos(2 * np.pi * f0 * self.t + phase0)
        x1 = torch.cos(2 * np.pi * f1 * self.t + phase1)
        x0 = x0 * self.ramp_rise_fall
        x1 = x1 * self.ramp_rise_fall
        x0 = util_torch.set_dbspl(x0, dbspl0)
        x1 = util_torch.set_dbspl(x1, dbspl1)
        if self.dur_padded is not None:
            x0 = torch.nn.functional.pad(x0, [0, self.n_pad])
            x1 = torch.nn.functional.pad(x1, [0, self.n_pad])
        return x0, x1

    def __getitem__(self, idx):
        """ """
        if self.eval_mode:
            interval = self.interval_list[idx]
            label = (interval > 0).type(interval.dtype)
            f0 = self.f_list[idx]
        else:
            interval = torch.exp(self.dist_log_interval.sample())
            label = self.dist_label.sample()
            if label == 0:
                interval *= -1
            f0 = torch.exp(self.dist_log_f.sample())
        f1 = f0 * torch.pow(2, interval)
        [dbspl0, dbspl1] = self.dist_dbspl.sample([2])
        [phase0, phase1] = self.dist_phase.sample([2])
        if self.dbspl_fixed:
            dbspl1 = dbspl0
        if self.phase_fixed:
            phase1 = phase0
        if self.phase is not None:
            phase0 = self.phase
            phase1 = self.phase
        x0, x1 = self.generate_stim(f0, f1, dbspl0, dbspl1, phase0, phase1)
        out = {
            "sr": torch.tensor(self.sr),
            "x0": x0,
            "x1": x1,
            "label": label,
            "interval": interval,
            "f0": f0,
            "f1": f1,
            "dbspl0": dbspl0,
            "dbspl1": dbspl1,
            "phase0": phase0,
            "phase1": phase1,
        }
        return out

    def __len__(self):
        """ """
        if self.eval_mode:
            return len(self.f_list)
        else:
            return sys.maxsize - 1
