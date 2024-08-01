import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import util_audio
import util_filters


def get_color_list(num_colors, cmap_name="Accent"):
    """
    Helper function returns list of colors for plotting.
    """
    if isinstance(cmap_name, list):
        return cmap_name
    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_colors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = [scalar_map.to_rgba(x) for x in range(num_colors)]
    return color_list


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


def make_line_plot(
    ax, x, y, legend_on=False, kwargs_plot={}, kwargs_legend={}, **kwargs_format_axes
):
    """
    Helper function for basic line plot with optional legend.
    """
    kwargs_plot_tmp = {
        "marker": "",
        "ls": "-",
        "color": [0, 0, 0],
        "lw": 1,
    }
    kwargs_plot_tmp.update(kwargs_plot)
    ax.plot(x, y, **kwargs_plot_tmp)
    ax = format_axes(ax, **kwargs_format_axes)
    if legend_on:
        kwargs_legend_tmp = {
            "loc": "lower right",
            "frameon": False,
            "handlelength": 1.0,
            "markerscale": 1.0,
            "fontsize": 12,
        }
        kwargs_legend_tmp.update(kwargs_legend)
        ax.legend(**kwargs_legend_tmp)
    return ax


def make_nervegram_plot(
    ax,
    nervegram,
    sr=20e3,
    cfs=None,
    cmap=matplotlib.cm.gray,
    cbar_on=False,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_labels=None,
    nxticks=6,
    nyticks=5,
    tmin=None,
    tmax=None,
    treset=True,
    vmin=None,
    vmax=None,
    interpolation="none",
    vticks=None,
    str_clabel=None,
    **kwargs_format_axes,
):
    """
    Helper function for visualizing auditory nervegram (or similar) representation.
    """
    # Trim nervegram if tmin and tmax are specified
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be freq-by-time array"
    t = np.arange(0, nervegram.shape[1]) / sr
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    # Setup time and frequency ticks and labels
    time_idx = np.linspace(0, t.shape[0] - 1, nxticks, dtype=int)
    time_labels = ["{:.0f}".format(1e3 * t[itr0]) for itr0 in time_idx]
    if cfs is None:
        cfs = np.arange(0, nervegram.shape[0])
    else:
        cfs = np.array(cfs)
        assert (
            cfs.shape[0] == nervegram.shape[0]
        ), "cfs.shape[0] must match nervegram.shape[0]"
    freq_idx = np.linspace(0, cfs.shape[0] - 1, nyticks, dtype=int)
    freq_labels = ["{:.0f}".format(cfs[itr0]) for itr0 in freq_idx]
    # Display nervegram image
    im_nervegram = ax.imshow(
        nervegram,
        origin="lower",
        aspect="auto",
        extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    # Add colorbar if `cbar_on == True`
    if cbar_on:
        cbar = plt.colorbar(im_nervegram, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(
            str_clabel, fontsize=fontsize_labels, fontweight=fontweight_labels
        )
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nyticks, integer=True)
            )
        cbar.ax.tick_params(
            direction="out",
            axis="both",
            which="both",
            labelsize=fontsize_ticks,
            length=fontsize_ticks / 2,
        )
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%03d"))
    # Format axes
    ax = format_axes(
        ax,
        xticks=time_idx,
        yticks=freq_idx,
        xticklabels=time_labels,
        yticklabels=freq_labels,
        fontsize_labels=fontsize_labels,
        fontsize_ticks=fontsize_ticks,
        fontweight_labels=fontweight_labels,
        **kwargs_format_axes,
    )
    return ax


def make_stimulus_summary_plot(
    ax_arr,
    ax_idx_waveform=None,
    ax_idx_spectrum=None,
    ax_idx_nervegram=None,
    ax_idx_excitation=None,
    waveform=None,
    nervegram=None,
    sr_waveform=None,
    sr_nervegram=None,
    cfs=None,
    tmin=None,
    tmax=None,
    treset=True,
    vmin=None,
    vmax=None,
    interpolation="none",
    n_anf=None,
    erb_freq_axis=True,
    spines_to_hide_waveform=[],
    spines_to_hide_spectrum=[],
    spines_to_hide_excitation=[],
    nxticks=6,
    nyticks=6,
    kwargs_plot={},
    limits_buffer=0.1,
    ax_arr_clear_leftover=True,
    **kwargs_format_axes,
):
    """
    Helper function for generating waveform, power spectrum, nervegram,
    and excitation pattern plots to summarize a stimulus.
    """
    # Axes are tracked in flattened array
    ax_arr = np.array([ax_arr]).reshape([-1])
    assert len(ax_arr.shape) == 1
    ax_idx_list = []

    # Plot stimulus waveform
    if ax_idx_waveform is not None:
        ax_idx_list.append(ax_idx_waveform)
        y_wav = np.squeeze(waveform)
        assert len(y_wav.shape) == 1, "waveform must be 1D array"
        x_wav = np.arange(0, y_wav.shape[0]) / sr_waveform
        if (tmin is not None) and (tmax is not None):
            IDX = np.logical_and(x_wav >= tmin, x_wav < tmax)
            x_wav = x_wav[IDX]
            y_wav = y_wav[IDX]
        if treset:
            x_wav = x_wav - x_wav[0]
        xlimits_wav = [x_wav[0], x_wav[-1]]
        ylimits_wav = [-np.max(np.abs(y_wav)), np.max(np.abs(y_wav))]
        ylimits_wav = np.array(ylimits_wav) * (1 + limits_buffer)
        make_line_plot(
            ax_arr[ax_idx_waveform],
            x_wav,
            y_wav,
            legend_on=False,
            kwargs_plot=kwargs_plot,
            kwargs_legend={},
            xlimits=xlimits_wav,
            ylimits=ylimits_wav,
            xticks=[],
            yticks=[],
            xticklabels=[],
            yticklabels=[],
            spines_to_hide=spines_to_hide_waveform,
            **kwargs_format_axes,
        )

    # Plot stimulus power spectrum
    if ax_idx_spectrum is not None:
        ax_idx_list.append(ax_idx_spectrum)
        fxx, pxx = util_audio.power_spectrum(waveform, sr_waveform)
        if cfs is not None:
            msg = "Frequency axes will not align when highest CF exceeds Nyquist"
            assert np.max(cfs) <= np.max(fxx), msg
            IDX = np.logical_and(fxx >= np.min(cfs), fxx <= np.max(cfs))
            pxx = pxx[IDX]
            fxx = fxx[IDX]
        xlimits_pxx = [np.max(pxx) * (1 + limits_buffer), 0]  # Reverses x-axis
        xlimits_pxx = np.ceil(np.array(xlimits_pxx) * 5) / 5
        if erb_freq_axis:
            fxx = util_filters.freq2erb(fxx)
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ["{:.0f}".format(yt) for yt in util_filters.erb2freq(yticks)]
        else:
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ["{:.0f}".format(yt) for yt in yticks]
        make_line_plot(
            ax_arr[ax_idx_spectrum],
            pxx,
            fxx,
            legend_on=False,
            kwargs_plot=kwargs_plot,
            str_xlabel="Power\n(dB SPL)",
            str_ylabel="Frequency (Hz)",
            xlimits=xlimits_pxx,
            ylimits=ylimits_fxx,
            xticks=xlimits_pxx,
            yticks=yticks,
            xticklabels=xlimits_pxx.astype(int),
            yticklabels=yticklabels,
            spines_to_hide=spines_to_hide_spectrum,
            **kwargs_format_axes,
        )

    # Plot stimulus nervegram
    if ax_idx_nervegram is not None:
        ax_idx_list.append(ax_idx_nervegram)
        if ax_idx_spectrum is not None:
            nervegram_nxticks = nxticks
            nervegram_nyticks = 0
            nervegram_str_xlabel = "Time\n(ms)"
            nervegram_str_ylabel = None
        else:
            nervegram_nxticks = nxticks
            nervegram_nyticks = nyticks
            nervegram_str_xlabel = "Time (ms)"
            nervegram_str_ylabel = "Characteristic frequency (Hz)"
        make_nervegram_plot(
            ax_arr[ax_idx_nervegram],
            nervegram,
            sr=sr_nervegram,
            cfs=cfs,
            nxticks=nervegram_nxticks,
            nyticks=nervegram_nyticks,
            tmin=tmin,
            tmax=tmax,
            treset=treset,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            str_xlabel=nervegram_str_xlabel,
            str_ylabel=nervegram_str_ylabel,
        )

    # Plot stimulus excitation pattern
    if ax_idx_excitation is not None:
        ax_idx_list.append(ax_idx_excitation)
        if np.all(np.mod(nervegram, 1) == 0):
            # Compute mean firing rate from spike counts if all values are integers
            x_exc = np.sum(nervegram, axis=1) / (nervegram.shape[1] / sr_nervegram)
            if n_anf is not None:
                # If a number of ANFs is specified, divide firing rate by n_anf
                x_exc = x_exc / n_anf
        else:
            # Otherwise, compute mean firing rates from instantaneous firing rates
            x_exc = np.mean(nervegram, axis=1)
        xlimits_exc = [0, np.max(x_exc) * (1 + limits_buffer)]
        xlimits_exc = np.ceil(np.array(xlimits_exc) / 10) * 10
        y_exc = np.arange(0, nervegram.shape[0])
        ylimits_exc = [np.min(y_exc), np.max(y_exc)]
        make_line_plot(
            ax_arr[ax_idx_excitation],
            x_exc,
            y_exc,
            legend_on=False,
            kwargs_plot=kwargs_plot,
            str_xlabel="Excitation\n(spikes/s)",
            xlimits=xlimits_exc,
            ylimits=ylimits_exc,
            xticks=xlimits_exc,
            yticks=[],
            xticklabels=xlimits_exc.astype(int),
            yticklabels=[],
            spines_to_hide=spines_to_hide_excitation,
            **kwargs_format_axes,
        )

    # Clear unused axes in ax_arr
    if ax_arr_clear_leftover:
        for ax_idx in range(ax_arr.shape[0]):
            if ax_idx not in ax_idx_list:
                ax_arr[ax_idx].axis("off")
    return ax_arr


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
                else:
                    kwargs_imshow.update({"cmap": "Reds_r"})
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
