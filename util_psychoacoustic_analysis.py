import functools
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import scipy
import util_psychoacoustic_localization

import util_misc


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


def fit_normcdf(x, y, method="trf", sigma_only=False, **kwargs):
    """ """
    if sigma_only:
        popt, pcov = scipy.optimize.curve_fit(
            lambda _, sigma: scipy.stats.norm(0.0, sigma).cdf(_),
            xdata=x,
            ydata=y,
            method=method,
            **kwargs,
        )
    else:
        mu0 = x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - 0.5))]
        popt, pcov = scipy.optimize.curve_fit(
            lambda _, mu, sigma: scipy.stats.norm(mu, sigma).cdf(_),
            xdata=x,
            ydata=y,
            p0=(mu0, 1.0),
            method=method,
            **kwargs,
        )
    return np.squeeze(popt), np.squeeze(pcov)


def psychometric_function(x, a, mu, sigma):
    """ """
    return a * scipy.stats.norm(mu, sigma).cdf(x)


def psychometric_function_inv(y, a, mu, sigma):
    """ """
    return scipy.stats.norm(mu, sigma).ppf(y / a)


def fit_psychometric_function(x, y, method="trf", p0=None, **kwargs):
    """ """
    if p0 is None:
        p0 = (1, x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - 0.5))], 1)
    try:
        popt, pcov = scipy.optimize.curve_fit(
            lambda _, a, mu, sigma: psychometric_function(_, a, mu, sigma),
            xdata=x,
            ydata=y,
            p0=p0,
            method=method,
            **kwargs,
        )
    except RuntimeError:
        popt = np.ones_like(p0) * np.nan
        pcov = np.ones_like(p0) * np.nan
    return np.squeeze(popt), np.squeeze(pcov)


def compute_srt_from_popt(popt, threshold_value="half"):
    """ """
    if isinstance(threshold_value, str):
        if "half" in threshold_value:
            srt = popt[1]
        else:
            raise ValueError(f"unrecognized {threshold_value=}")
    else:
        srt = psychometric_function_inv(threshold_value, *popt)
    return srt


def compute_discrimination_threshold(
    independent_variable,
    dependent_variable,
    threshold_value=0.707,
):
    """ """
    x = np.array(list(itertools.product(independent_variable, independent_variable)))
    y = np.array(list(itertools.product(dependent_variable, dependent_variable)))
    index_valid_trials = x[:, 1] > x[:, 0]
    x = x[index_valid_trials]
    y = y[index_valid_trials]
    x = x[:, 1] - x[:, 0]
    y = y[:, 1] > y[:, 0] + (np.random.randn(*x.shape) * np.finfo(np.float32).eps)
    df = util_misc.flatten_columns(
        pd.DataFrame({"x": x, "y": y})
        .groupby("x")
        .agg({"y": ["mean", "count"]})
        .reset_index()
        .sort_values(by="x")
    )
    popt, pcov = fit_normcdf(
        df.x.values,
        df.y_mean.values,
        sigma_only=True,
        sigma=1 / np.sqrt(df.y_count.values),  # Weight by 1 / sqrt(num_trials)
    )
    threshold = scipy.stats.norm(0, popt).ppf(threshold_value)
    return threshold


def interpolate_dataframe(
    df_to_interp,
    df_reference,
    key_x="fmod",
    key_y="threshold_dbm",
    key_condition=[
        "tone_carrier",
        "masker_dbm",
        "masker_fmod_center",
        "masker_fmod_octave",
    ],
    kwargs_interp={"left": np.nan, "right": np.nan},
):
    """
    Function interpolates a dataframe at x-values from the reference dataframe
    while preserving the specified conditions indicated in both dataframes.

    Args
    ----
    df_to_interp (pd.DataFrame): dataframe to interpolate
    df_reference (pd.DataFrame): dataframe providing x-values for interpolation
    key_x (str): key for x-values
    key_y (str): key for y-values
    key_condition (list): keys for all conditions to match between dataframes
    kwargs_interp (dict): keyword arguments for np.interp (i.e., prevent extrapolation)

    Returns
    -------
    df_interpolated (pd.DataFrame): interpolated dataframe
    df_reference_overlap (pd.DataFrame): reference dataframe (overlapping conditions)
    """
    if not isinstance(key_condition, list):
        key_condition = [key_condition]
    d0 = {k: v for k, v in df_to_interp.groupby(key_condition)}
    d1 = {k: v for k, v in df_reference.groupby(key_condition)}
    shared_conditions = set(d0.keys()).intersection(d1.keys())
    d0 = {k: d0[k] for k in shared_conditions}
    d1 = {k: d1[k] for k in shared_conditions}
    df_interpolated = []
    df_reference_overlap = []
    for k, sub_df_reference in d1.items():
        xp = d0[k][key_x].values
        fp = d0[k][key_y].values
        x = sub_df_reference[key_x].values
        y = np.interp(x, xp, fp, **kwargs_interp)
        sub_df_to_interp = pd.DataFrame({key_x: x, key_y: y})
        for kc in key_condition:
            sub_df_to_interp[kc] = sub_df_reference.iloc[0][kc]
        df_interpolated.append(sub_df_to_interp)
        df_reference_overlap.append(sub_df_reference)
    df_interpolated = pd.concat(df_interpolated)
    df_reference_overlap = pd.concat(df_reference_overlap)
    return df_interpolated, df_reference_overlap


class PsychoacousticExperiment:
    def __init__(
        self,
        read=True,
        write=True,
        overwrite=False,
        verbose=False,
        **kwargs,
    ):
        """ """
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


class ExperimentAudiogram(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_pure_tone_detection_eval.csv",
        basename_results="audiogram.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df = (
            df.groupby("f")
            .agg(
                {
                    "dbspl": list,
                    "prob": list,
                }
            )
            .reset_index()
        )
        df["popt"] = df.apply(
            lambda _: fit_normcdf(_["dbspl"], _["prob"])[0],
            axis=1,
        )
        df["threshold"] = df["popt"].map(lambda _: _[0])
        df = df.drop(columns=["dbspl", "popt", "prob"])
        df = df.rename(columns={"f": "frequency_hz", "threshold": "threshold_dbspl"})
        return df


class ExperimentTemporalGapDetection(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_temporal_gap_detection_eval.csv",
        basename_results="classifier_temporal_gap_detection_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df = df.groupby("dbspl").agg({"gap": list, "prob": list}).reset_index()
        df["popt"] = df.apply(
            lambda _: fit_normcdf(np.log(_["gap"]), _["prob"], method="dogbox")[0],
            axis=1,
        )
        df["threshold"] = df["popt"].map(lambda _: scipy.stats.norm(*_).ppf(0.707))
        df["threshold"] = np.exp(df["threshold"])
        df = df.drop(columns=["gap", "popt", "prob"])
        return df


class ExperimentSAMDetection(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_sam_discrimination_cf*_bw*_eval.csv",
        basename_results="classifier_sam_discrimination_cfbw_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["bandwidth"] = df["classifier_fn_ckpt"].map(
            lambda x: int([_ for _ in x.split("_") if "bw" in _][0][-4:]),
        )
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(["classifier_fn_ckpt", "bandwidth", "fmod", "dbm"])
            .agg({"correct": "mean"})
            .reset_index()
            .groupby(["classifier_fn_ckpt", "bandwidth", "fmod"])
            .agg({"dbm": list, "correct": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["dbm"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_dbm"] = df["popt"].map(lambda _: logistic_function_inv(0.707, *_))
        df.loc[df["correct"].map(max) < 0.7, "threshold_dbm"] = np.inf
        df = df.drop(columns=["classifier_fn_ckpt", "correct", "dbm", "popt"])
        return df


class ExperimentSAMDetectionCarrier(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_sam_discrimination_carrier*_eval.csv",
        basename_results="classifier_sam_discrimination_carrier_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["carrier"] = df["classifier_fn_ckpt"].map(
            lambda x: [_ for _ in x.split("_") if "carrier" in _][0],
        )
        df["carrier"] = df["carrier"].map(lambda _: _.replace("carrier", ""))
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(["classifier_fn_ckpt", "carrier", "fmod", "dbm"])
            .agg({"correct": "mean"})
            .reset_index()
            .groupby(["classifier_fn_ckpt", "carrier", "fmod"])
            .agg({"dbm": list, "correct": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["dbm"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_dbm"] = df["popt"].map(lambda _: logistic_function_inv(0.707, *_))
        df.loc[df["correct"].map(max) < 0.7, "threshold_dbm"] = np.inf
        df = df.drop(columns=["classifier_fn_ckpt", "correct", "dbm", "popt"])
        return df


class ExperimentSAMMasking(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_sam_masking_*_eval.csv",
        basename_results="classifier_sam_masking_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df = df.rename(columns={"signal_dbm": "dbm", "signal_fmod": "fmod"})
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(
                [
                    "classifier_fn_ckpt",
                    "masker_dbm",
                    "masker_fmod_center",
                    "masker_fmod_octave",
                    "fmod",
                    "dbm",
                ]
            )
            .agg({"correct": "mean"})
            .reset_index()
        )
        df = (
            df.groupby(
                [
                    "classifier_fn_ckpt",
                    "masker_dbm",
                    "masker_fmod_center",
                    "masker_fmod_octave",
                    "fmod",
                ]
            )
            .agg({"dbm": list, "correct": list})
            .reset_index()
        )
        df["tone_carrier"] = df["classifier_fn_ckpt"].str.contains("tone").astype(int)
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["dbm"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_dbm"] = df["popt"].map(lambda _: logistic_function_inv(0.707, *_))
        df.loc[df["correct"].map(max) < 0.7, "threshold_dbm"] = np.inf
        df = df.drop(columns=["classifier_fn_ckpt", "correct", "dbm", "popt"])
        return df


class ExperimentSpectralMasking(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_spectral_masking_*_eval.csv",
        basename_results="classifier_spectral_masking_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df = (
            df.groupby(
                [
                    "classifier_fn_ckpt",
                    "masker_dbspl",
                    "masker_cf",
                    "signal_cf",
                ]
            )
            .agg({"signal_dbspl": list, "prob": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda dfi: fit_normcdf(dfi["signal_dbspl"], dfi["prob"], **kw)[0],
            axis=1,
        )
        df["threshold_db"] = df["popt"].apply(lambda _: scipy.stats.norm(*_).ppf(0.707))
        df0, df1, df2 = [dfi for _, dfi in df.groupby("masker_dbspl")]
        assert np.all(df0["masker_dbspl"].values == -np.inf)
        df0 = df0.drop(
            columns=[
                "classifier_fn_ckpt",
                "masker_dbspl",
                "signal_dbspl",
                "prob",
                "popt",
            ],
        ).rename(
            columns={"threshold_db": "threshold_db_unmasked"},
        )
        df = pd.concat(
            [
                pd.merge(df1, df0),
                pd.merge(df2, df0),
            ]
        )
        df["masking_db"] = df["threshold_db"] - df["threshold_db_unmasked"]
        df = df.drop(columns=["classifier_fn_ckpt", "signal_dbspl", "prob", "popt"])
        return df


class ExperimentSpectralRippleDetection(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_spectral_ripple_detection_eval.csv",
        basename_results="classifier_spectral_ripple_detection_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(["classifier_fn_ckpt", "ripples_per_octave", "ripple_depth"])
            .agg({"correct": "mean"})
            .reset_index()
        )
        df = df.sort_values(by=["ripples_per_octave", "ripple_depth"])
        df = (
            df.groupby(["classifier_fn_ckpt", "ripples_per_octave"])
            .agg({"ripple_depth": list, "correct": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["ripple_depth"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_db"] = df["popt"].map(lambda _: logistic_function_inv(0.707, *_))
        df.loc[df["correct"].map(max) < 0.7, "threshold_db"] = np.inf
        df = df.drop(columns=["classifier_fn_ckpt", "correct", "ripple_depth", "popt"])
        return df


class ExperimentSpectralRippleDiscrimination(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_spectral_ripple_discrimination_*_eval.csv",
        basename_results="classifier_spectral_ripple_discrimination_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        map_classifier_fn_ckpt_to_mode = {
            "classifier_spectral_ripple_discrimination_standard_ckpt_BEST.pt": 0,
            "classifier_spectral_ripple_discrimination_inverted_ckpt_BEST.pt": 1,
            "classifier_spectral_ripple_discrimination_samediff_ckpt_BEST.pt": 2,
        }
        df["mode"] = df["classifier_fn_ckpt"].map(map_classifier_fn_ckpt_to_mode)
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(["mode", "ripples_per_octave"])
            .agg({"correct": "mean"})
            .reset_index()
        )
        df = df.sort_values(by="ripples_per_octave", ascending=False)
        df["nlogrpo"] = np.log(1 / df["ripples_per_octave"])
        df = (
            df.groupby(["mode"])
            .agg({"ripples_per_octave": list, "nlogrpo": list, "correct": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["nlogrpo"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_nlogrpo"] = df["popt"].map(
            lambda _: logistic_function_inv(0.707, *_)
        )
        df.loc[df["correct"].map(max) < 0.7, "threshold_nlogrpo"] = np.inf
        df["threshold_ripples_per_octave"] = 1 / np.exp(df["threshold_nlogrpo"])
        df = df.drop(columns=["popt"])
        df = df.explode(column=["ripples_per_octave", "nlogrpo", "correct"])
        return df


class ExperimentSTMDetection(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="classifier_stm_detection_eval.csv",
        basename_results="classifier_stm_detection_results.csv",
        **kwargs,
    ):
        """ """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["ripples_per_octave"] = df["ripples_per_octave"].round(decimals=3)
        df["ripples_per_second"] = df["ripples_per_second"].round(decimals=3)
        df["ripples_per_second"] = np.abs(df["ripples_per_second"])
        df["correct"] = df["label"] == (df["logits"] > 0)
        df = (
            df.groupby(["ripples_per_octave", "ripples_per_second", "dbm"])
            .agg({"correct": "mean"})
            .reset_index()
            .groupby(["ripples_per_octave", "ripples_per_second"])
            .agg({"dbm": list, "correct": list})
            .reset_index()
        )
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(_["dbm"], _["correct"], **kw)[0],
            axis=1,
        )
        df["threshold_dbm"] = df["popt"].map(lambda _: logistic_function_inv(0.707, *_))
        df.loc[df["correct"].map(max) < 0.7, "threshold_dbm"] = np.inf
        df = df.drop(columns=["correct", "dbm", "popt"])
        return df


class ExperimentWordInNoiseConditions(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval_phaselocknet_spkr_word_recognition_human_experiment_v00_foreground60dbspl.csv",
        basename_results=None,
        **kwargs,
    ):
        """ """
        if basename_results is None:
            basename_results = basename_eval.replace(
                os.path.splitext(basename_eval)[1],
                "_results" + os.path.splitext(basename_eval)[1],
            )
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.read_csv(fn_eval)
        df_word_int = pd.read_csv("data/misc/word_int_encoding.csv")
        df_word_int = df_word_int[~df_word_int.cv.isna() & ~df_word_int.wsn.isna()]
        df["label_word_int"] = df["label_word_int"].map(
            dict(zip(df_word_int.wsn.astype(int), df_word_int.cv.astype(int)))
        )
        df["correct_word"] = df["label_word_int.pred"] == df["label_word_int"]
        df = (
            df.groupby(["background_condition", "snr"])
            .agg({"correct_word": "mean"})
            .reset_index()
        )
        return df


class ExperimentWordInSyntheticTextures(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval_phaselocknet_spkr_word_recognition_speech_in_synthetic_textures.csv",
        basename_results=None,
        **kwargs,
    ):
        """ """
        if basename_results is None:
            basename_results = basename_eval.replace(
                os.path.splitext(basename_eval)[1],
                "_results" + os.path.splitext(basename_eval)[1],
            )
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.read_csv(fn_eval)
        df_word_int = pd.read_csv("data/misc/word_int_encoding.csv")
        df_word_int = df_word_int[~df_word_int.cv.isna() & ~df_word_int.wsn.isna()]
        df["label_word_int"] = df["label_word_int"].map(
            dict(zip(df_word_int.wsn.astype(int), df_word_int.cv.astype(int)))
        )
        df["correct_word"] = df["label_word_int.pred"] == df["label_word_int"]
        df = (
            df.groupby(["index_texture", "snr"])
            .agg({"correct_word": "mean"})
            .reset_index()
        )
        return df


class ExperimentWordSpatialReleaseFromMasking(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval_bmld_v03.csv",
        basename_results=None,
        **kwargs,
    ):
        """ """
        if basename_results is None:
            basename_results = basename_eval.replace(
                os.path.splitext(basename_eval)[1],
                "_results" + os.path.splitext(basename_eval)[1],
            )
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.read_csv(fn_eval)
        df["correct_word"] = df["label_word_int.pred"] == df["label_word_int"]
        df["background_azim"] = df["background_azim"].map(
            util_psychoacoustic_localization.normalize_angle
        )
        df = (
            df.groupby(["condition", "index_room", "background_azim", "snr"])
            .agg({"correct_word": "mean"})
            .reset_index()
        )
        df = (
            df.groupby(["condition", "index_room", "background_azim"])
            .agg({"snr": list, "correct_word": list})
            .reset_index()
        )
        df["popt"] = df.apply(
            lambda _: fit_normcdf(_["snr"], _["correct_word"])[0],
            axis=1,
        )
        df["srt"] = df["popt"].map(lambda _: _[0])
        df = df.drop(columns=["popt", "snr", "correct_word"])
        return df


class ExperimentITDThreshold(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval_phaselocknet_localization_itd_threshold.csv",
        basename_results=None,
        **kwargs,
    ):
        """ """
        if basename_results is None:
            basename_results = basename_eval.replace(
                os.path.splitext(basename_eval)[1],
                "_results" + os.path.splitext(basename_eval)[1],
            )
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.read_csv(fn_eval)
        df = df.join(pd.read_pickle(fn_eval.replace(".csv", "_prob.gz")))
        n_loc_classes = len(df["label_loc_int.prob"].iloc[0])
        func_label_to_azim_elev = functools.partial(
            util_psychoacoustic_localization.label_to_azim_elev,
            multitask=n_loc_classes > 504,
        )
        df = util_psychoacoustic_localization.experiment_itd_threshold(
            df_eval=df,
            func_label_to_azim_elev=func_label_to_azim_elev,
        )
        df = df.drop(columns=["itd", "azim_pred"])
        return df


class ExperimentEnvelopeITDThreshold(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval_phaselocknet_localization_envelope_itd_threshold.csv",
        basename_results=None,
        **kwargs,
    ):
        """ """
        if basename_results is None:
            basename_results = basename_eval.replace(
                os.path.splitext(basename_eval)[1],
                "_results" + os.path.splitext(basename_eval)[1],
            )
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def run(self, fn_eval):
        """ """
        df = pd.read_csv(fn_eval)
        df = df.join(pd.read_pickle(fn_eval.replace(".csv", "_prob.gz")))
        n_loc_classes = len(df["label_loc_int.prob"].iloc[0])
        func_label_to_azim_elev = functools.partial(
            util_psychoacoustic_localization.label_to_azim_elev,
            multitask=n_loc_classes > 504,
        )
        df = util_psychoacoustic_localization.experiment_envelope_itd_threshold(
            df_eval=df,
            func_label_to_azim_elev=func_label_to_azim_elev,
        )
        df = df.drop(columns=["itd", "azim_pred"])
        return df


class ExperimentHeinz2001FrequencyDiscrimination(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval.csv",
        basename_results="results.csv",
        **kwargs,
    ):
        """ """
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
