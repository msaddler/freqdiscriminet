import collections
import glob
import json
import os

import numpy as np
import torch

import util_filters
import util_torch

if not os.environ.get("IGNORE_TENSORFLOW", False):
    import tensorflow as tf

    import util_tfrecords
else:
    print(f"{os.environ['IGNORE_TENSORFLOW']=}")


class AuditoryModelFromConfig(torch.nn.Module):
    def __init__(
        self,
        dir_model=None,
        fn_config="config.json",
        fn_arch="arch.json",
        config_model={},
        architecture=[],
        input_shape=[2, 110000, 2],
        config_random_slice={"size": [50, 20000], "buffer": [0, 500]},
        device=None,
    ):
        """ """
        super().__init__()
        self.input_shape = input_shape
        if not config_model:
            with open(os.path.join(dir_model, fn_config), "r") as f:
                config_model = json.load(f)
        if not architecture:
            with open(os.path.join(dir_model, fn_arch), "r") as f:
                architecture = json.load(f)
        kwargs_peripheral_model = {
            "sr_input": config_model["kwargs_cochlea"].get("sr_input", None),
            "sr_output": config_model["kwargs_cochlea"].get("sr_output", None),
            "config_cochlear_filterbank": config_model["kwargs_cochlea"].get(
                "config_filterbank", {}
            ),
            "config_ihc_transduction": config_model["kwargs_cochlea"].get(
                "config_subband_processing", {}
            ),
            "config_ihc_lowpass_filter": config_model["kwargs_cochlea"].get(
                "kwargs_fir_lowpass_filter_output", {}
            ),
            "config_anf_rate_level": config_model["kwargs_cochlea"].get(
                "kwargs_sigmoid_rate_level_function", {}
            ),
            "config_anf_spike_generator": config_model["kwargs_cochlea"].get(
                "kwargs_spike_generator_binomial", {}
            ),
            "config_modulation_filterbank": config_model["kwargs_cochlea"].get(
                "kwargs_modulation_filterbank", {}
            ),
            "config_random_slice": config_random_slice,
        }
        assert kwargs_peripheral_model["config_ihc_lowpass_filter"].pop(
            "ihc_filter", True
        )
        self.peripheral_model = PeripheralModelFromConfig(
            **kwargs_peripheral_model,
        )
        self.perceptual_model = PerceptualModelFromConfig(
            architecture=architecture,
            input_shape=self.peripheral_model(torch.zeros(self.input_shape)).shape,
            heads=config_model["n_classes_dict"],
            device=device,
        )

    def forward(self, x):
        """ """
        return self.perceptual_model(self.peripheral_model(x))


class PeripheralModelFromConfig(torch.nn.Module):
    def __init__(
        self,
        sr_input=None,
        sr_output=None,
        config_cochlear_filterbank={},
        config_ihc_transduction={},
        config_ihc_lowpass_filter={},
        config_anf_rate_level={},
        config_anf_spike_generator={},
        config_modulation_filterbank={},
        config_random_slice={},
    ):
        """ """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_input if sr_output is None else sr_output
        self.body = collections.OrderedDict()
        # Bandpass filterbank determines cochlear frequency tuning
        filterbank_mode = config_cochlear_filterbank.get("mode", "").lower()
        if filterbank_mode == "":
            self.body["cochlear_filterbank"] = util_torch.Unsqueeze(dim=1)
        elif filterbank_mode == "fir_gammatone_filterbank":
            if config_cochlear_filterbank.get("cfs", False):
                cfs = np.array(config_cochlear_filterbank["cfs"])
            else:
                cfs = util_filters.erbspace(
                    config_cochlear_filterbank["min_cf"],
                    config_cochlear_filterbank["max_cf"],
                    config_cochlear_filterbank["num_cf"],
                )
            self.body["cochlear_filterbank"] = util_torch.GammatoneFilterbank(
                sr=sr_input,
                fir_dur=config_cochlear_filterbank.get("fir_dur", 0.05),
                cfs=cfs,
                **config_cochlear_filterbank.get("kwargs_filter_coefs", {}),
            )
        else:
            raise NotImplementedError(f"{filterbank_mode=} is not implemented")
        # IHC transduction (includes compression and half-wave rectification)
        if config_ihc_transduction:
            self.body["ihc_transduction"] = util_torch.IHCTransduction(
                **config_ihc_transduction,
            )
        # IHC lowpass filter determines phase locking limit
        if config_ihc_lowpass_filter:
            self.body["ihc_lowpass_filter"] = util_torch.IHCLowpassFilter(
                sr_input=self.sr_input,
                sr_output=self.sr_output,
                **config_ihc_lowpass_filter,
            )
        # Rate-level function determines thresholds and dynamic ranges
        if config_anf_rate_level:
            self.body["anf_rate_level"] = util_torch.SigmoidRateLevelFunction(
                **config_anf_rate_level,
            )
        # ANF spike generator determines noisiness of spont rate channels
        if config_anf_spike_generator:
            self.body["anf_spike_generator"] = util_torch.BinomialSpikeGenerator(
                **config_anf_spike_generator,
            )
        # Modulation filterbank (envelope frequency decomposition)
        if config_modulation_filterbank:
            self.body["modulation_filterbank"] = util_torch.ModulationFilterbank(
                sr=self.sr_output,
                **config_modulation_filterbank,
            )
        self.body = torch.nn.Sequential(self.body)
        # Randomly slice peripheral model representation (trim boundary artifacts)
        if config_random_slice:
            self.head = util_torch.RandomSlice(**config_random_slice)
        else:
            self.head = torch.nn.Identity()

    def forward(self, x):
        """ """
        if x.ndim > 2:
            assert x.shape[-1] == 2, "Expected binaural input"
            y = torch.concat(
                [self.body(x[..., 0]), self.body(x[..., 1])],
                axis=1,
            )
        else:
            y = self.body(x)
        y = self.head(y)
        return y


class PerceptualModelFromConfig(torch.nn.Module):
    def __init__(
        self,
        architecture=[],
        input_shape=[2, 6, 50, 20000],
        heads={
            "label_env_int": 50,
            "label_loc_int": 505,
            "label_talker_int": 501,
            "label_word_int": 801,
        },
        device=None,
    ):
        """ """
        super().__init__()
        self.input_shape = input_shape
        self.body = collections.OrderedDict()
        self.heads = heads
        self.head = {k: collections.OrderedDict() for k in self.heads}
        self.construct_model(architecture, device)

    def get_layer_from_description(self, d, x):
        """ """
        layer_type = d["layer_type"].lower()
        if "conv" in layer_type:
            layer = util_torch.CustomPaddedConv2d(
                in_channels=x.shape[1],
                out_channels=d["args"]["filters"],
                kernel_size=d["args"]["kernel_size"],
                stride=d["args"]["strides"],
                padding=d["args"]["padding"],
                dilation=d["args"].get("dilation", 1),
                groups=d["args"].get("groups", 1),
                bias=d["args"].get("bias", True),
                padding_mode="zeros",
            )
        elif "dense" in layer_type:
            layer = torch.nn.Linear(
                in_features=x.shape[1],
                out_features=d["args"]["units"],
                bias=d["args"].get("use_bias", True),
            )
        elif "dropout" in layer_type:
            layer = torch.nn.Dropout(p=d["args"]["rate"], inplace=False)
        elif "flatten" in layer_type:
            layer = util_torch.CustomFlatten(
                start_dim=1,
                end_dim=-1,
                permute_dims=(0, 2, 3, 1),
            )
        elif "maxpool" in layer_type:
            layer = torch.nn.MaxPool2d(
                kernel_size=d["args"]["pool_size"],
                stride=d["args"]["strides"],
                padding=0,
            )
        elif "hpool" in layer_type:
            layer = util_torch.HanningPooling(
                stride=d["args"]["strides"],
                kernel_size=d["args"]["pool_size"],
                padding=d["args"]["padding"],
                sqrt_window=d["args"].get("sqrt_window", False),
                normalize=d["args"].get("normalize", False),
            )
        elif "batchnorm" in layer_type.replace("_", ""):
            layer = torch.nn.SyncBatchNorm(
                num_features=x.shape[1] if d["args"].get("axis", -1) == -1 else None,
                eps=d["args"].get("eps", 1e-05),
                momentum=d["args"].get("momentum", 0.1),
                affine=d["args"].get("scale", True),
            )
        elif "layernorm" in layer_type.replace("_", ""):
            layer = util_torch.CustomNorm(
                input_shape=x.shape,
                dim_affine=1 if d["args"].get("scale", True) else None,
                dim_norm=1 if d["args"]["axis"] == -1 else d["args"]["axis"],
                correction=1,
                eps=d["args"].get("eps", 1e-05),
            )
        elif "permute" in layer_type:
            layer = util_torch.Permute(dims=d["args"]["dims"])
        elif "reshape" in layer_type:
            layer = util_torch.Reshape(shape=d["args"]["shape"])
        elif "unsqueeze" in layer_type:
            layer = util_torch.Unsqueeze(dim=d["args"]["dim"])
        elif "randomslice" in layer_type.replace("_", ""):
            layer = util_torch.RandomSlice(
                size=d["args"]["size"],
                buffer=d["args"]["buffer"],
            )
        elif "ihclowpassfilter" in layer_type.replace("_", ""):
            layer = util_torch.IHCLowpassFilter(
                sr_input=d["args"]["sr_input"],
                sr_output=d["args"]["sr_output"],
                fir_dur=d["args"]["fir_dur"],
                cutoff=d["args"]["cutoff"],
                order=d["args"]["order"],
            )
        elif "relu" in layer_type:
            layer = torch.nn.ReLU(inplace=False)
        elif ("branch" in layer_type) or ("fc_top" in layer_type):
            layer = None
        else:
            raise ValueError(f"{layer_type=} not recognized")
        return layer

    def construct_model(self, architecture, device):
        """ """
        x = torch.zeros(self.input_shape).to(device)
        is_body_layer = True
        for d in architecture:
            if is_body_layer:
                layer = self.get_layer_from_description(d, x)
            else:
                layer = {
                    k: self.get_layer_from_description(d, x[k]) for k in self.heads
                }
            if (layer is None) or (
                isinstance(layer, dict) and list(layer.values())[0] is None
            ):
                is_body_layer = False
                if not isinstance(x, dict):
                    x = {k: torch.clone(x) for k in self.heads}
            else:
                if is_body_layer:
                    self.body[d["args"]["name"]] = layer
                    x = layer.to(x.device)(x)
                else:
                    for k in self.heads:
                        self.head[k][d["args"]["name"]] = layer[k]
                        x[k] = layer[k].to(x[k].device)(x[k])
        self.body = torch.nn.Sequential(self.body)
        if not isinstance(x, dict):
            x = {k: torch.clone(x) for k in self.heads}
        for k in self.heads:
            self.head[k]["fc_output"] = torch.nn.Linear(
                in_features=x[k].shape[1],
                out_features=self.heads[k],
                bias=True,
            )
            self.head[k] = torch.nn.Sequential(self.head[k])
        self.head = torch.nn.ModuleDict(self.head)

    def forward(self, x):
        """ """
        x = self.body(x)
        logits = {k: self.head[k](x) for k in self.heads}
        return logits


class ActivationAccessModelFromConfig(torch.nn.Module):
    def __init__(
        self,
        dir_model=None,
        fn_config="config.json",
        fn_arch="arch.json",
        config_model={},
        architecture=[],
        input_shape=[2, 110000, 2],
        activations="relu",
        **kwargs_auditory_model,
    ):
        """ """
        super().__init__()
        if not config_model:
            with open(os.path.join(dir_model, fn_config), "r") as f:
                config_model = json.load(f)
        if not architecture:
            with open(os.path.join(dir_model, fn_arch), "r") as f:
                architecture = json.load(f)
        self.auditory_model = AuditoryModelFromConfig(
            config_model=config_model,
            architecture=architecture,
            input_shape=input_shape,
            **kwargs_auditory_model,
        )
        self.activations = collections.OrderedDict()
        for name, module in self.auditory_model.named_modules():
            if isinstance(activations, str):
                if activations in name:
                    self.activations[name] = None
            elif isinstance(activations, tuple):
                if any([_ in name for _ in activations]):
                    self.activations[name] = None
            elif isinstance(activations, list):
                list_prefix = [
                    "peripheral_model.",
                    "perceptual_model.",
                    "body.",
                    "head.",
                ]
                k = name
                for prefix in list_prefix:
                    k = k.removeprefix(prefix)
                if (name in activations) or (k in activations):
                    self.activations[name] = None
            else:
                raise ValueError("activations must be string, tuple, or list")
            if name in self.activations:
                module.register_forward_hook(self.get_activation(name))
        assert self.activations, "no matching activations found"

    def get_activation(self, name):
        """ """

        def hook(model, input, output):
            assert name in self.activations
            self.activations[name] = output

        return hook

    def forward(self, x):
        """ """
        self.auditory_model(x)
        return self.activations


def map_paramter_tf2torch(tf_parameter, override_name_map={}):
    """ """
    # Re-name parameter
    tf_name = tf_parameter.name
    torch_name = ""
    if "label_" in tf_name:
        torch_name += "head."
        body = False
    else:
        torch_name += "body."
        body = True
    prefix, suffix = tf_name.split("/")
    if body:
        torch_name += prefix
    else:
        head = prefix[prefix.find("label_") :]
        prefix = prefix[: prefix.find("_label_")]
        torch_name += f"{head}.{prefix}"
    if ("kernel" in suffix) or ("gamma" in suffix):
        suffix = "weight"
    elif ("bias" in suffix) or ("beta" in suffix):
        suffix = "bias"
    elif "moving_mean" in suffix:
        suffix = "running_mean"
    elif "moving_var" in suffix:
        suffix = "running_var"
    else:
        raise NotImplementedError(f"Did not recognize {suffix=}")
    torch_name += f".{suffix}"
    torch_name = torch_name.replace("fc_top", "fc_output")
    # Override parameter name if specified in `override_name_map`
    if tf_name in override_name_map.keys():
        torch_name = override_name_map[tf_name]
    # Re-shape parameter (NHWC --> NCHW)
    if len(tf_parameter.shape) == 2:
        torch_parameter = tf.transpose(tf_parameter, [1, 0])
    elif len(tf_parameter.shape) == 4:
        torch_parameter = tf.transpose(tf_parameter, [3, 2, 0, 1])
    else:
        torch_parameter = tf.identity(tf_parameter)
    return torch_name, torch_parameter


class TFRecordParallelDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        regex_filenames,
        batch_size=1,
        num_workers=1,
        worker_id=0,
        kwargs_get_dataset_from_tfrecords={},
    ):
        """ """
        super().__init__()
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filenames = list(
            np.array_split(
                glob.glob(regex_filenames),
                self.num_workers,
            )[self.worker_id]
        )
        self.dataset = util_tfrecords.get_dataset_from_tfrecords(
            filenames=self.filenames,
            eval_mode=False,
            batch_size=self.batch_size,
            **kwargs_get_dataset_from_tfrecords,
        )

    def __iter__(self):
        """ """
        return iter(self.dataset)


def tf_rms(x, axis=-1, keepdims=True):
    """
    Compute root mean square amplitude of a tensor.

    Args
    ----
    x (tensor): tensor for which root mean square amplitude is computed
    axis (int or list): axis along which to compute mean
    keepdims (bool): specifies if mean should keep collapsed dimension(s)

    Returns
    -------
    out (tensor): root mean square amplitude of x
    """
    out = tf.math.sqrt(
        tf.math.reduce_mean(tf.math.square(x), axis=axis, keepdims=keepdims)
    )
    return out


def tf_set_dbspl(x, dbspl, mean_subtract=True, axis=-1, strict=True):
    """
    Set sound pressure level in dB re 20e-6 Pa (dB SPL) of a tensor.

    Args
    ----
    x (tensor): tensor for which sound pressure level is set
    dbspl (tensor): desired sound pressure level in dB re 20e-6 Pa
    mean_subtract (bool): if True, x is first de-meaned
    axis (int or list): axis along which to measure RMS amplitudes
    strict (bool): if True, an error will be raised if x is silent;
        if False, silent signals will be returned as-is

    Returns
    -------
    out (tensor): sound pressure level of x in dB re 20e-6 Pa
    """
    if mean_subtract:
        x = x - tf.math.reduce_mean(x, axis=axis, keepdims=True)
    rms_new = 20e-6 * tf.math.pow(10.0, dbspl / 20.0)
    rms_old = tf_rms(x, axis=axis, keepdims=True)
    if strict:
        tf.debugging.assert_none_equal(
            rms_old,
            tf.zeros_like(rms_old),
            message="Failed to set dB SPL of all-zero signal",
        )
        out = tf.math.multiply(rms_new / rms_old, x)
    else:
        out = tf.where(
            tf.math.equal(rms_old, tf.zeros_like(rms_old)),
            x,
            tf.math.multiply(rms_new / rms_old, x),
        )
    return out
