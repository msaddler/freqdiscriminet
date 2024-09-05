import copy
import glob
import json
import os
import resource
import time

import matplotlib
import numpy as np
import pandas as pd
import scipy


def logistic_function(x, x0, k, chance=0.5):
    """ """
    return ((1 - chance) / (1 + np.exp(-k * (x - x0)))) + chance


def logistic_function_inv(y, x0, k, chance=0.5):
    """ """
    return (np.log(((1 - chance) / (y - chance)) - 1) / -k) + x0


def fit_logistic_function(x, y, method="trf", chance=0.5, p0=None, **kwargs):
    """ """
    if p0 is None:
        p0 = (x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - chance))], 1)
    popt, pcov = scipy.optimize.curve_fit(
        lambda _, x0, k: logistic_function(_, x0, k, chance=chance),
        xdata=x,
        ydata=y,
        p0=p0,
        method=method,
        **kwargs,
    )
    return np.squeeze(popt), np.squeeze(pcov)


class PsychoacousticExperiment:
    def __init__(
        self,
        read=True,
        write=True,
        overwrite=False,
        verbose=False,
        **kwargs,
    ):
        """
        Base class for running a psychoacoustic experiment on a
        model (converts evaluation output file to a results file)
        """
        self.read = read
        self.write = write
        self.overwrite = overwrite
        self.verbose = verbose
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, regex_dir_model):
        """ """
        for key in ["basename_eval", "basename_results", "run"]:
            msg = f"experiment subclass is missing attribute: {key}"
            assert hasattr(self, key), msg
        if isinstance(regex_dir_model, list):
            list_dir_model = regex_dir_model
        elif isinstance(regex_dir_model, str):
            list_dir_model = glob.glob(regex_dir_model)
        else:
            raise ValueError(f"unrecognized format: {regex_dir_model=}")
        list_df = []
        for dir_model in list_dir_model:
            fn_eval = os.path.join(dir_model, self.basename_eval)
            fn_results = os.path.join(dir_model, self.basename_results)
            df = None
            if self.read:
                if (os.path.exists(fn_results)) and (not self.overwrite):
                    df = pd.read_csv(fn_results)
                    if self.verbose:
                        print(f"[experiment] READ {fn_results=}")
            if df is None:
                df = self.run(fn_eval)
                if self.verbose:
                    print(f"[experiment] READ {fn_eval=}")
            if self.write:
                if (not os.path.exists(fn_results)) or (self.overwrite):
                    df.to_csv(fn_results, index=False)
                    if self.verbose:
                        print(f"[experiment] WROTE {fn_results=}")
            list_df.append(df.assign(dir_model=dir_model, fn_eval=fn_eval))
        return pd.concat(list_df)

    def __repr__(self):
        """ """
        d = {
            a: getattr(self, a)
            for a in dir(self)
            if (not a.startswith("__")) and (isinstance(getattr(self, a), str))
        }
        return json.dumps(d)


class ExperimentHeinz2001FrequencyDiscrimination(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval.csv",
        basename_results="results.csv",
        **kwargs,
    ):
        """
        PsychoacousticExperiment class for pure tone frequency discrimination
        experiment from Heinz et al. (2001, Neural Computation)
        """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def octave_to_weber_fraction(self, interval):
        """ """
        return np.power(2, interval) - 1.0

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["correct"] = df["label"] == (df["logits"] > 0)
        df["interval"] = np.abs(df["interval"])
        df = df.groupby(["f0", "interval"]).agg({"correct": "mean"}).reset_index()
        df = df.sort_values(by=["f0", "interval"])
        df = df.groupby(["f0"]).agg({"interval": list, "correct": list}).reset_index()
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(np.log(_["interval"]), _["correct"], **kw)[
                0
            ],
            axis=1,
        )
        df["log_threshold"] = df["popt"].map(lambda _: logistic_function_inv(0.75, *_))
        df["threshold"] = np.exp(df["log_threshold"])
        df["weber_fraction"] = self.octave_to_weber_fraction(df["threshold"])
        df = df.drop(columns=["interval", "correct", "popt"])
        return df


def get_model_progress_display_str(
    epoch=None,
    step=None,
    num_steps=None,
    t0=None,
    mem=True,
    loss=None,
    task_loss={},
    task_acc={},
    single_line=True,
):
    """
    Returns a string to print model progress.

    Args
    ----
    epoch (int): current training epoch
    step (int): current training step
    num_steps (int): total steps taken since t0
    t0 (float): start time in seconds
    mem (bool): if True, include total memory usage
    loss (float): current loss
    task_loss (dict): current task-specific losses
    task_acc (dict): current task-specific accuracies
    single_line (bool): if True, remove linebreaks

    Returns
    -------
    display_str (str): formatted string to print
    """
    display_str = ""
    if (epoch is not None) and (step is not None):
        display_str += "step {:02d}_{:06d} | ".format(epoch, step)
    if (num_steps is not None) and (t0 is not None):
        display_str += "{:.4f} s/step | ".format((time.time() - t0) / num_steps)
    if mem:
        display_str += "mem: {:06.3f} GB | ".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        )
    if loss is not None:
        display_str += "loss: {:.4f} | ".format(loss)
    if task_loss:
        if isinstance(task_loss, dict):
            display_str += "\n|___ task loss | "
            for k, v in task_loss.items():
                display_str += "{}: {:.4f} ".format(
                    k.replace("label_", "").replace("_int", ""), v
                )
            display_str += "| "
        else:
            display_str += "task_loss: {:.4f} | ".format(task_loss)
    if task_acc:
        if isinstance(task_acc, dict):
            display_str += "\n|___ task accs | "
            for k, v in task_acc.items():
                display_str += "{}: {:.4f} ".format(
                    k.replace("label_", "").replace("_int", ""), v
                )
            display_str += "| "
        else:
            display_str += "task_acc: {:.4f} | ".format(task_acc)
    if single_line:
        display_str = display_str.replace("\n|___ ", "")
    return display_str


def format_axes(
    ax,
    str_title=None,
    str_xlabel=None,
    str_ylabel=None,
    fontsize_title=12,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_title=None,
    fontweight_labels=None,
    xscale="linear",
    yscale="linear",
    xlimits=None,
    ylimits=None,
    xticks=None,
    yticks=None,
    xticks_minor=None,
    yticks_minor=None,
    xticklabels=None,
    yticklabels=None,
    spines_to_hide=[],
    major_tick_params_kwargs_update={},
    minor_tick_params_kwargs_update={},
):
    """
    Helper function for setting axes-related formatting parameters.
    """
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    major_tick_params_kwargs = {
        "axis": "both",
        "which": "major",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 2,
        "direction": "out",
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    minor_tick_params_kwargs = {
        "axis": "both",
        "which": "minor",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 4,
        "direction": "out",
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    return ax


def get_xy_from_center(center=(0, 0), w=1.0, h=1.0):
    """
    Helper function returns vertices of rectangle with specified
    center, width, and height (4-by-2 array).
    """
    xy = np.array(
        [
            [center[0] - w / 2, center[1] - h / 2],
            [center[0] - w / 2, center[1] + h / 2],
            [center[0] + w / 2, center[1] + h / 2],
            [center[0] + w / 2, center[1] - h / 2],
        ]
    )
    return xy


def get_dim_from_raw_value(raw_value, range_dim=None, scaling="log2"):
    """
    Helper function for scaling architectural parameter values to
    plot coordinates. Output dimensions can be thresholded to fall
    within range specified by `range_dim`.
    """
    if scaling == "log2":
        dim = np.log2(raw_value)
    elif scaling == "linear":
        dim = raw_value
    else:
        raise ValueError("Unrecognized scaling: {}".format(scaling))
    if range_dim is not None:
        dim = np.max([range_dim[0], dim])
        dim = np.min([range_dim[1], dim])
    return dim


def get_affine_transform(center=(0, 0), deg_scale_x=0, deg_skew_y=0):
    """
    Helper function build matplotlib Affine2D transformation to approximate
    3D rotation in 2D plane.
    """
    transform = matplotlib.transforms.Affine2D()
    transform = transform.translate(-center[0], -center[1])
    transform = transform.scale(np.cos(np.deg2rad(deg_scale_x)), 1)
    transform = transform.skew_deg(0, deg_skew_y)
    transform = transform.translate(center[0], center[1])
    return transform


def draw_conv_kernel_on_image(
    ax,
    args_image={},
    args_kernel={},
    kwargs_polygon_kernel={},
    kwargs_polygon_kernel_line={},
    kwargs_transform={},
    kernel_x_shift=-0.75,
    kernel_y_shift=0.75,
):
    """
    Helper function draws convolutional kernels superimposed on inputs.
    """
    # Get image and kernel shape from args_image and args_kernel
    ishape = np.array(args_image["shape"])
    assert len(ishape) == 2, "image shape must be 2D"
    kshape = np.array(args_kernel["shape"])
    assert len(kshape) == 2, "kernel shape must be 2D"
    # Define kernel plot dimensions using image plot dimensions
    for key in ["x", "y", "w", "h", "zorder"]:
        assert key in args_image.keys(), "missing args_image key: {}".format(key)
    args_kernel["w"] = (kshape[1] / ishape[1]) * args_image["w"]
    args_kernel["h"] = (kshape[0] / ishape[0]) * args_image["h"]
    args_kernel["x"] = (
        args_image["x"] + kernel_x_shift * (args_image["w"] - args_kernel["w"]) / 2
    )
    args_kernel["y"] = (
        args_image["y"] + kernel_y_shift * (args_image["h"] - args_kernel["h"]) / 2
    )
    args_kernel["zorder"] = args_image["zorder"] + 0.5
    # Define transform around the image center
    center_kernel = (args_kernel["x"], args_kernel["y"])
    center_image = (args_image["x"], args_image["y"])
    xy = get_xy_from_center(
        center=center_kernel, w=args_kernel["w"], h=args_kernel["h"]
    )
    transform = get_affine_transform(center=center_image, **kwargs_transform)
    # Add a filled polygon patch for the kernel
    patch = matplotlib.patches.Polygon(
        xy, **kwargs_polygon_kernel, zorder=args_kernel["zorder"]
    )
    patch.set_transform(transform + ax.transData)
    ax.add_patch(patch)
    # Add an unfilled polygon patch for the kernel boundary
    patch = matplotlib.patches.Polygon(
        xy, **kwargs_polygon_kernel_line, fill=False, zorder=args_kernel["zorder"]
    )
    patch.set_transform(transform + ax.transData)
    ax.add_patch(patch)
    # Store kernel vertices and shift proportions in args_kernel for connecting lines
    args_kernel["vertices"] = transform.transform(xy)
    args_kernel["x_shift"] = (args_kernel["x"] - args_image["x"]) / (
        args_image["w"] / 2
    )
    args_kernel["y_shift"] = (args_kernel["y"] - args_image["y"]) / (
        args_image["h"] / 2
    )
    return ax, args_image, args_kernel


def draw_cnn_from_layer_list(
    ax,
    layer_list,
    scaling_w="log2",
    scaling_h="log2",
    scaling_n="log2",
    include_kernels=True,
    input_image=None,
    input_image_shape=None,
    gap_input_scale=2.0,
    gap_interlayer=2.0,
    gap_intralayer=0.2,
    gap_output=0,
    deg_scale_x=60,
    deg_skew_y=30,
    deg_fc=0,
    range_w=None,
    range_h=None,
    kernel_x_shift=-0.75,
    kernel_y_shift=0.75,
    limits_buffer=1e-2,
    arrow_width=0.25,
    scale_fc=1.0,
    spines_to_hide=["top", "bottom", "left", "right"],
    kwargs_imshow_update={},
    kwargs_polygon_update={},
    kwargs_polygon_kernel_update={},
    kwargs_polygon_kernel_line_update={},
    kwargs_arrow_update={},
    binaural=False,
    kwargs_imshow_binaural=({}, {}),
):
    """
    Main function for drawing CNN architecture schematic.
    """
    # Define and update default keyword arguments for matplotlib drawing
    kwargs_imshow = {
        "cmap": matplotlib.cm.gray,
        "aspect": "auto",
        "origin": "lower",
        "alpha": 1.0,
    }
    kwargs_imshow.update(kwargs_imshow_update)
    kwargs_polygon = {
        "ec": [0.0, 0.0, 0.0],
        "fc": np.array([126, 126, 126]) / 256,
        "lw": 1.25,
        "fill": True,
        "alpha": 1.0,
    }
    kwargs_polygon.update(kwargs_polygon_update)
    kwargs_polygon_kernel = copy.deepcopy(kwargs_polygon)
    kwargs_polygon_kernel["alpha"] = 0.5
    kwargs_polygon_kernel["fc"] = np.array([215, 229, 236]) / 256
    kwargs_polygon_kernel["lw"] = 0.0
    kwargs_polygon_kernel["fill"] = True
    kwargs_polygon_kernel.update(kwargs_polygon_kernel_update)
    kwargs_polygon_kernel_line = {
        "alpha": 1.0,
        "color": kwargs_polygon_kernel["fc"],
        "lw": 1.75,
    }
    kwargs_polygon_kernel_line.update(kwargs_polygon_kernel_line_update)
    kwargs_arrow = {
        "width": arrow_width,
        "length_includes_head": True,
        "head_width": arrow_width * 2.5,
        "head_length": arrow_width * 2.5,
        "overhang": 0.0,
        "head_starts_at_zero": False,
        "color": "k",
    }
    kwargs_arrow.update(kwargs_arrow_update)
    kwargs_arrow_gap = copy.deepcopy(kwargs_arrow)
    kwargs_arrow_gap["head_width"] = 0
    kwargs_arrow_gap["head_length"] = 0
    kwargs_transform = {"deg_scale_x": deg_scale_x, "deg_skew_y": deg_skew_y}

    # Define coordinate tracker variables
    (xl, yl, zl) = (0, 0, 0)

    # Display the input image
    if input_image is not None:
        input_image_shape = list(input_image.shape)
        assert len(input_image_shape) == 3, "input_image_shape must be [h, w, channel]"
        h = get_dim_from_raw_value(
            input_image_shape[0], range_dim=range_h, scaling=scaling_h
        )
        w = get_dim_from_raw_value(
            input_image_shape[1], range_dim=range_w, scaling=scaling_w
        )
        n_channel = input_image.shape[2]
        for itr_channel in range(n_channel):
            if binaural:
                if itr_channel < n_channel / 2:
                    kwargs_imshow.update({"cmap": "Blues_r"})
                    kwargs_imshow.update(kwargs_imshow_binaural[0])
                else:
                    kwargs_imshow.update({"cmap": "Reds_r"})
                    kwargs_imshow.update(kwargs_imshow_binaural[1])
            extent = np.array([xl - w / 2, xl + w / 2, yl - h / 2, yl + h / 2])
            im = ax.imshow(
                input_image[:, :, itr_channel],
                extent=extent,
                zorder=zl,
                **kwargs_imshow,
            )
            args_image = {
                "x": xl,
                "y": yl,
                "w": w,
                "h": h,
                "zorder": zl,
                "shape": [input_image_shape[0], input_image_shape[1]],
            }
            zl += 1
            transform = get_affine_transform(center=(xl, yl), **kwargs_transform)
            im.set_transform(transform + ax.transData)
            M = transform.transform(extent.reshape([2, 2]).T)
            dx_arrow = np.min([M[-1, 0] - xl, gap_interlayer * gap_input_scale])
            ax.arrow(x=xl, y=yl, dx=dx_arrow, dy=0, zorder=zl, **kwargs_arrow_gap)
            if itr_channel < n_channel - 1:
                xl += gap_intralayer * gap_input_scale * 2
            zl += 1
        xl += gap_interlayer * gap_input_scale
    else:
        args_image = None

    if isinstance(binaural, bool):
        binaural = False

    # Display the network architecture
    kernel_to_connect = False
    for itr_layer, layer in enumerate(layer_list):
        # Draw convolutional layer
        if "conv" in layer["type"]:
            # Draw convolutional kernel superimposed on previous layer
            if include_kernels:
                args_kernel = {
                    "shape": layer["shape_kernel"],
                }
                if args_image:
                    ax, args_image, args_kernel = draw_conv_kernel_on_image(
                        ax,
                        args_image=args_image,
                        args_kernel=args_kernel,
                        kwargs_polygon_kernel=kwargs_polygon_kernel,
                        kwargs_polygon_kernel_line=kwargs_polygon_kernel_line,
                        kwargs_transform=kwargs_transform,
                        kernel_x_shift=kernel_x_shift,
                        kernel_y_shift=kernel_y_shift,
                    )
                    kernel_to_connect = True
            # Draw convolutional layer activations as stacked rectangles
            [h, w] = layer["shape_activation"]
            n = int(
                get_dim_from_raw_value(
                    layer["channels"], range_dim=None, scaling=scaling_n
                )
            )
            if n % 2 == 1:
                n += 1
            w = get_dim_from_raw_value(w, range_dim=range_w, scaling=scaling_w)
            h = get_dim_from_raw_value(h, range_dim=range_h, scaling=scaling_h)
            for itr_sublayer in range(n):
                xy = get_xy_from_center(center=(xl, yl), w=w, h=h)
                if binaural:
                    kwargs_polygon_binaural = dict(kwargs_polygon)
                    if (itr_layer < binaural) and (itr_sublayer < n / 2):
                        kwargs_polygon_binaural.update(
                            {
                                "ec": [0, 0, 1],
                                "fc": [0.5, 0.5, 0.6],
                            }
                        )
                    elif (itr_layer < binaural) and (itr_sublayer >= n / 2):
                        kwargs_polygon_binaural.update(
                            {
                                "ec": [1, 0, 0],
                                "fc": [0.6, 0.5, 0.5],
                            }
                        )
                    patch = matplotlib.patches.Polygon(
                        xy, **kwargs_polygon_binaural, zorder=zl
                    )
                else:
                    patch = matplotlib.patches.Polygon(xy, **kwargs_polygon, zorder=zl)
                transform = get_affine_transform(center=(xl, yl), **kwargs_transform)
                patch.set_transform(transform + ax.transData)
                ax.add_patch(patch)
                args_image = {
                    "x": xl,
                    "y": yl,
                    "w": w,
                    "h": h,
                    "zorder": zl,
                    "shape": layer["shape_activation"],
                }
                if kernel_to_connect:
                    # If convolutional kernel was drawn, add connecting lines to layer
                    vertex_output_x = args_image["x"] + args_kernel["x_shift"] * (
                        args_image["w"] / 2
                    )
                    vertex_output_y = args_image["y"] + args_kernel["y_shift"] * (
                        args_image["h"] / 2
                    )
                    vertex_output = transform.transform(
                        np.array([vertex_output_x, vertex_output_y])
                    )
                    for vertex_input in args_kernel["vertices"]:
                        ax.plot(
                            [vertex_input[0], vertex_output[0]],
                            [vertex_input[1], vertex_output[1]],
                            **kwargs_polygon_kernel_line,
                            zorder=args_kernel["zorder"],
                        )
                    kernel_to_connect = False
                if itr_sublayer == n - 1:
                    dx_arrow = transform.transform(xy)[-1, 0] - xl
                    ax.arrow(
                        x=xl, y=yl, dx=dx_arrow, dy=0, zorder=zl, **kwargs_arrow_gap
                    )
                    zl += 1
                zl += 1
                xl += gap_intralayer
            xl += gap_interlayer
        # Draw fully-connected layer
        elif ("dense" in layer["type"]) or ("fc" in layer["type"]):
            xl += gap_output
            n = layer["shape_activation"][0]
            w = gap_intralayer
            h = get_dim_from_raw_value(n, range_dim=None, scaling=scaling_n) * scale_fc
            xy = get_xy_from_center(center=(xl, yl), w=w, h=h)
            patch = matplotlib.patches.Polygon(xy, **kwargs_polygon, zorder=zl)
            ax.add_patch(patch)
            zl += 1
            xl += gap_interlayer

    # Draw underlying arrow and format axes
    ax.arrow(x=0, y=yl, dx=xl, dy=0, **kwargs_arrow, zorder=-1)
    ax.update_datalim([[0, yl], [xl, yl]])
    [xb, yb, dxb, dyb] = ax.dataLim.bounds
    ax.set_xlim([xb - limits_buffer * dxb, xb + (1 + limits_buffer) * dxb])
    ax.set_ylim([yb - limits_buffer * dyb, yb + (1 + limits_buffer) * dyb])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    return ax
