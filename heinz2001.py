import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchmetrics

import util_misc
import util_tf2torch
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
        transform=None,
    ):
        """ """
        self.transform = transform
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
        if self.transform is not None:
            out = self.transform(out)
        return out

    def __len__(self):
        """ """
        if self.eval_mode:
            return len(self.f_list)
        else:
            return sys.maxsize - 1


class Model(torch.nn.Module):
    def __init__(
        self,
        config_model={},
        architecture=[],
        input_shape=[2, 2, 10000],
        device=None,
    ):
        """ """
        super().__init__()
        self.input_shape = input_shape
        self.peripheral_model = ANModel(**config_model["kwargs_anmodel"])
        self.perceptual_model = util_tf2torch.PerceptualModelFromConfig(
            architecture=architecture,
            input_shape=self.peripheral_model(torch.zeros(self.input_shape)).shape,
            heads=config_model["n_classes_dict"],
            device=device,
        )

    def forward(self, x):
        """ """
        return self.perceptual_model(self.peripheral_model(x))


class ANModel(torch.nn.Module):
    def __init__(
        self,
        sr_input=40e3,
        sr_output=20e3,
        min_cf=1e2,
        max_cf=1e4,
        num_cf=60,
        fir_dur=None,
        scale_gammatone_filterbank=None,
        ihc_lowpass_cutoff=4.8e3,
        anf_per_channel=200,
    ):
        """ """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_output
        if self.sr_output is None:
            self.sr_output = self.sr_input
        self.cfs = self.x2f(np.linspace(self.f2x(min_cf), self.f2x(max_cf), num_cf))
        self.gammatone_filterbank = util_torch.GammatoneFilterbank(
            sr=self.sr_input,
            fir_dur=fir_dur,
            cfs=self.cfs,
            dtype=torch.float32,
        )
        self.scale_gammatone_filterbank = scale_gammatone_filterbank
        self.ihc_nonlinearity = IHCNonlinearity()
        self.ihc_lowpassfilter = IHCLowpassFilter(
            sr_input=self.sr_input,
            sr_output=self.sr_output,
            cutoff=ihc_lowpass_cutoff,
        )
        self.neural_adaptation = NeuralAdaptation(sr=self.sr_output)
        self.anf_per_channel = anf_per_channel

    def f2x(self, f):
        """ """
        msg = "frequency out of human range"
        assert np.all(np.logical_and(f >= 20, f <= 20677)), msg
        x = (1.0 / 0.06) * np.log10((f / 165.4) + 0.88)
        return x

    def x2f(self, x):
        """ """
        msg = "BM distance out of human range"
        assert np.all(np.logical_and(x >= 0, x <= 35)), msg
        f = 165.4 * (np.power(10.0, (0.06 * x)) - 0.88)
        return f

    def spike_generator(self, rate):
        """ """
        if self.anf_per_channel is None:
            return rate
        p = rate / self.sr_output
        spikes = (
            torch.rand(
                size=(self.anf_per_channel, *p.shape),
                device=rate.device,
            )
            < p[None, :]
        )
        return spikes.sum(dim=0).to(rate.dtype)

    def forward(self, x):
        """ """
        if x.shape[-1] == 2:
            x = torch.swapaxes(x, -1, -2)
        x = self.gammatone_filterbank(x)
        if self.scale_gammatone_filterbank is not None:
            x = self.scale_gammatone_filterbank * x
        x = self.ihc_nonlinearity(x)
        x = self.ihc_lowpassfilter(x)
        x = self.neural_adaptation(x)
        x = self.spike_generator(x)
        return x


class IHCNonlinearity(torch.nn.Module):
    def __init__(
        self,
        ihc_asym=3,
        ihc_k=1225.0,
    ):
        """ """
        super().__init__()
        self.ihc_beta = torch.tensor(
            np.tan(np.pi * (-0.5 + 1 / (ihc_asym + 1))),
            dtype=torch.float32,
        )
        self.ihc_k = torch.tensor(ihc_k, dtype=torch.float32)

    def forward(self, x):
        """ """
        x = torch.atan(self.ihc_k * x + self.ihc_beta) - torch.atan(self.ihc_beta)
        x = x / (np.pi / 2 - torch.atan(self.ihc_beta))
        return x


class IHCLowpassFilter(torch.nn.Module):
    def __init__(
        self,
        sr_input=40e3,
        sr_output=20e3,
        cutoff=4.8e3,
        n=7,
    ):
        """ """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_output
        self.n = n
        c = 2 * self.sr_input
        c1LPihc = (c - 2 * np.pi * cutoff) / (c + 2 * np.pi * cutoff)
        c2LPihc = 2 * np.pi * cutoff / (2 * np.pi * cutoff + c)
        b = torch.tensor([c2LPihc, c2LPihc], dtype=torch.float32)
        a = torch.tensor([1.0, -c1LPihc], dtype=torch.float32)
        self.register_buffer("b", b)
        self.register_buffer("a", a)
        self.stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(self.stride, sr_input / sr_output), msg

    def forward(self, x):
        """ """
        for _ in range(self.n):
            x = torchaudio.functional.lfilter(
                waveform=x,
                a_coeffs=self.a,
                b_coeffs=self.b,
            )
        if not self.sr_output == self.sr_input:
            x = x[..., :: self.stride]
        return x


class NeuralAdaptation(torch.nn.Module):
    def __init__(
        self,
        sr=20e3,
        VI=5e-4,
        VL=5e-3,
        PG=3e-2,
        PL=6e-2,
        PIrest=1.2e-2,
        PImax=6e-1,
        spont=5e1,
    ):
        """ """
        super().__init__()
        self.sr = torch.tensor(sr, dtype=torch.float32)
        self.VI = torch.tensor(VI, dtype=torch.float32)
        self.VL = torch.tensor(VL, dtype=torch.float32)
        self.PG = torch.tensor(PG, dtype=torch.float32)
        self.PL = torch.tensor(PL, dtype=torch.float32)
        self.PIrest = torch.tensor(PIrest, dtype=torch.float32)
        self.PImax = torch.tensor(PImax, dtype=torch.float32)
        self.spont = torch.tensor(spont, dtype=torch.float32)
        self.ln2 = torch.log(torch.tensor(2.0))

    def forward(self, ihcl):
        """ """
        SPER = 1 / self.sr
        CI = torch.ones_like(ihcl[..., 0]) * self.spont / self.PIrest
        CL = CI * (self.PIrest + self.PL) / self.PL
        CG = CL * (1 + self.PL / self.PG) - CI * self.PL / self.PG
        p1 = torch.log(torch.exp(self.ln2 * self.PImax / self.PIrest) - 1)
        p3 = p1 * self.PIrest / self.ln2
        PPI = p3 / p1 * torch.log(1 + torch.exp(p1 * ihcl))
        ifr = torch.ones_like(ihcl) * self.spont
        for k in range(1, ihcl.shape[-1]):
            CI = CI + (SPER / self.VI) * (-PPI[..., k] * CI + self.PL * (CL - CI))
            CL = CL + (SPER / self.VL) * (-self.PL * (CL - CI) + self.PG * (CG - CL))
            ifr[..., k] = CI * PPI[..., k]
        return ifr


def evaluate(
    model,
    dataset,
    dir_model=None,
    fn_ckpt="ckpt_BEST.pt",
    fn_eval_output="eval.csv",
    batch_size=32,
    num_steps_per_display=10,
    num_workers=8,
    device=None,
    overwrite=False,
):
    """ """
    print(f"[evaluate] {dir_model=}")
    print(f"[evaluate] {fn_ckpt=}")
    print(f"[evaluate] {fn_eval_output=}")
    print(f"[evaluate] {len(dataset)} examples")
    print(f"[evaluate] {int(np.ceil(len(dataset) / batch_size))} batches")
    fn_eval_output = os.path.join(dir_model, fn_eval_output)
    if os.path.exists(fn_eval_output):
        if overwrite:
            print(f"[evaluate] Overwriting pre-existing {fn_eval_output=}")
            os.remove(fn_eval_output)
        else:
            print(f"[complete] {fn_eval_output=} already exists!")
            return
    util_torch.load_model_checkpoint(
        model=model.perceptual_model,
        dir_model=dir_model,
        step=None,
        fn_ckpt=fn_ckpt,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )
    model.train(False)
    with torch.no_grad():
        for itr, example in enumerate(dataloader):
            x0 = example.pop("x0").to(device)
            x1 = example.pop("x1").to(device)
            logits = torch.squeeze(model(torch.stack([x0, x1], dim=1))["label"])
            out = {k: v.cpu().detach().numpy() for k, v in example.items()}
            out["logits"] = logits.cpu().detach().numpy()
            out["prob"] = torch.nn.functional.sigmoid(logits).cpu().detach().numpy()
            if itr == 0:
                df = {k: v.tolist() for k, v in out.items()}
            else:
                for k, v in out.items():
                    df[k].extend(v.tolist())
            if itr % num_steps_per_display == 0:
                print(f"... {len(df['prob'])} of {len(dataset)} examples")
    df = pd.DataFrame(df)
    df = df.sort_index(axis=1)
    df.to_csv(fn_eval_output + "~", index=False)
    os.rename(fn_eval_output + "~", fn_eval_output)
    print(f"[complete] {fn_eval_output=}")


def optimize(
    model,
    dataset,
    dir_model=None,
    optimizer=None,
    scheduler=None,
    loss_function=None,
    metric_acc=None,
    batch_size=32,
    num_epochs=20,
    num_steps_per_epoch=1000,
    num_steps_per_display=10,
    num_steps_valid=100,
    num_workers=8,
    device=None,
):
    """ """
    logfile = os.path.join(dir_model, "log_optimize.csv")
    if os.path.exists(logfile):
        df = pd.read_csv(logfile)
        init_epoch = df["epoch"].max()
        util_torch.load_model_checkpoint(
            model=model.perceptual_model,
            dir_model=dir_model,
            step=None,
        )
    else:
        df = pd.DataFrame({})
        init_epoch = 0
        if os.path.exists(os.path.join(dir_model, "ckpt_BEST.pt")):
            util_torch.load_model_checkpoint(
                model=model.perceptual_model,
                dir_model=dir_model,
                step=None,
            )
    metric_loss = torchmetrics.aggregation.MeanMetric().to(device)
    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    )
    if scheduler is not None:
        if init_epoch > 0:
            for epoch in range(init_epoch + 1):
                scheduler.step()
                print(f"[scheduler] {epoch=} | lr={scheduler.get_last_lr()}")
        lr0 = optimizer.param_groups[0]["lr"]
        print(f"[scheduler] initial learning rate: {lr0}")
    for epoch in range(init_epoch, num_epochs):
        # Training steps
        util_torch.set_trainable(model=model, trainable=True)
        metric_loss.reset()
        metric_acc.reset()
        t0 = time.time()
        for step in range(num_steps_per_epoch):
            example = next(dataloader)
            x0 = example.pop("x0").to(device)
            x1 = example.pop("x1").to(device)
            targets = example.pop("label").to(device)
            optimizer.zero_grad()
            logits = torch.squeeze(model(torch.stack([x0, x1], dim=-1))["label"])
            loss = loss_function(logits, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=1.0,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=None,
            )
            if torch.isfinite(grad_norm).all():
                optimizer.step()
            else:
                print("* * * NaN or Inf gradient --> skipping batch * * *")
            metric_loss.update(loss)
            metric_acc.update(torch.nn.functional.sigmoid(logits), targets)
            if step % num_steps_per_display == num_steps_per_display - 1:
                display_str = util_misc.get_model_progress_display_str(
                    epoch=epoch + 1,
                    step=step + 1,
                    num_steps=step + 1,
                    t0=t0,
                    mem=True,
                    loss=None,
                    task_loss=metric_loss.compute().item(),
                    task_acc=metric_acc.compute().item(),
                )
                display_str += "grad: {:.4f}".format(grad_norm)
                print(display_str)
        train_loss = metric_loss.compute().item()
        train_acc = metric_acc.compute().item()
        # Validation steps
        if num_steps_valid is not None:
            metric_loss.reset()
            metric_acc.reset()
            util_torch.set_trainable(model=model, trainable=False)
            with torch.no_grad():
                for step in range(num_steps_valid):
                    example = next(dataloader)
                    x0 = example.pop("x0").to(device)
                    x1 = example.pop("x1").to(device)
                    targets = example.pop("label").to(device)
                    logits = torch.squeeze(model(torch.stack([x0, x1], dim=1))["label"])
                    loss = loss_function(logits, targets)
                    metric_loss.update(loss)
                    metric_acc.update(torch.nn.functional.sigmoid(logits), targets)
            valid_loss = metric_loss.compute().item()
            valid_acc = metric_acc.compute().item()
            print(f"[validation] epoch {epoch + 1} | {valid_loss=} | {valid_acc=}")
        else:
            valid_loss = train_loss
            valid_acc = train_acc
        # Logging training progress
        df_epoch = pd.DataFrame(
            {
                "epoch": [epoch + 1],
                "valid.acc": [valid_acc],
                "valid.loss": [valid_loss],
                "train.acc": [train_acc],
                "train.loss": [train_loss],
            }
        )
        df_epoch.to_csv(
            logfile,
            mode="a",
            header=not os.path.exists(logfile),
            index=False,
        )
        df = pd.concat([df, df_epoch])
        # Save model weights if better than previous best
        if np.isclose(valid_acc, df["valid.acc"].max()):
            util_torch.save_model_checkpoint(
                model=model.perceptual_model,
                dir_model=dir_model,
                step=None,
            )
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
            lr1 = optimizer.param_groups[0]["lr"]
            if not lr1 == lr0:
                print(f"[scheduler] updated learning rate: {lr0} -> {lr1}")
                lr0 = lr1
    return df


def main(dir_model=None, eval_mode=False, overwrite=False):
    """ """
    with open(os.path.join(dir_model, "config.json"), "r") as f:
        config_model = json.load(f)
    with open(os.path.join(dir_model, "arch.json"), "r") as f:
        architecture = json.load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(
        config_model=config_model,
        architecture=architecture,
        input_shape=[2, 2, 10000],
        device=device,
    ).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    metric_acc = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
    if eval_mode:
        log2_max = -1
        log2_min = -21
        interval_list = np.logspace(
            log2_max,
            log2_min,
            log2_max - log2_min + 1,
            base=2,
        )
        if ("specific" in dir_model) and (config_model.get("kwargs_dataset", False)):
            interval_list = np.concatenate(
                [interval_list] * 250 + [-1 * interval_list] * 250
            )
            f_list = np.array(
                np.exp(
                    np.mean(
                        np.log(
                            [
                                config_model["kwargs_dataset"]["f_min"],
                                config_model["kwargs_dataset"]["f_max"],
                            ]
                        )
                    )
                )
            ).reshape([-1])
            if "interval_max" in config_model["kwargs_dataset"]:
                interval_max = config_model["kwargs_dataset"]["interval_max"]
                interval_list = interval_list[np.abs(interval_list) <= interval_max]
        else:
            interval_list = np.concatenate(
                [interval_list] * 50 + [-1 * interval_list] * 50
            )
            f_list = np.power(2, (np.arange(np.log2(250.0), np.log2(10000.0), 1 / 5)))
        dataset = Dataset(
            f_list=f_list,
            interval_list=interval_list,
        )
        evaluate(
            model=model,
            dataset=dataset,
            dir_model=dir_model,
            fn_ckpt="ckpt_BEST.pt",
            fn_eval_output="eval.csv",
            batch_size=config_model["kwargs_optimize"]["batch_size"],
            num_steps_per_display=10,
            num_workers=8,
            device=device,
            overwrite=overwrite,
        )
    else:
        dataset = Dataset(**config_model.get("kwargs_dataset", {}))
        trainable_params = util_torch.set_trainable(model, True)
        optimizer = torch.optim.Adam(
            params=trainable_params,
            lr=config_model["kwargs_optimize"]["learning_rate"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            **config_model["kwargs_optimize"]["kwargs_scheduler"],
        )
        optimize(
            model=model,
            dataset=dataset,
            dir_model=dir_model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            metric_acc=metric_acc,
            batch_size=config_model["kwargs_optimize"]["batch_size"],
            num_epochs=config_model["kwargs_optimize"]["epochs"],
            num_steps_per_epoch=config_model["kwargs_optimize"]["steps_per_epoch"],
            num_steps_per_display=10,
            num_steps_valid=config_model["kwargs_optimize"]["validation_steps"],
            num_workers=8,
            device=device,
        )


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--dir_model", type=str, default=None)
    parser.add_argument("-e", "--eval_mode", type=int, default=0)
    parser.add_argument("-o", "--overwrite", type=int, default=0)
    main(**vars(parser.parse_args()))
