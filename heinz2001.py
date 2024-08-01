import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torchmetrics

import util
import util_torch

from pure_tone_dataset import Dataset
from peripheral_model import PeripheralModel
from perceptual_model import PerceptualModel


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
        self.peripheral_model = PeripheralModel(
            **config_model["kwargs_anmodel"],
        )
        self.perceptual_model = PerceptualModel(
            architecture=architecture,
            input_shape=self.peripheral_model(torch.zeros(self.input_shape)).shape,
            heads=config_model["n_classes_dict"],
            device=device,
        )

    def forward(self, x):
        """ """
        return self.perceptual_model(self.peripheral_model(x))


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
                display_str = util.get_model_progress_display_str(
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
