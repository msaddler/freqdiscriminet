import collections
import copy
import json
import resource
import time

import h5py
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Helper class to JSON serialize numpy arrays.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def flatten_columns(df, sep="_"):
    """
    Flatten multi-level columns in a pandas DataFrame to single-level.
    """
    df.columns = [
        col[0] if (len(col[0]) == 0) or (len(col[1]) == 0) else sep.join(col)
        for col in df.columns.to_flat_index()
    ]
    return df


def get_hdf5_dataset_key_list(f_input):
    """
    Walks hdf5 file and returns list of all dataset keys.

    Args
    ----
    f_input (str or h5py.File): hdf5 filename or file object

    Returns
    -------
    hdf5_dataset_key_list (list): list of paths to datasets in f_input
    """
    if isinstance(f_input, str):
        f = h5py.File(f_input, "r")
    else:
        f = f_input
    hdf5_dataset_key_list = []

    def get_dataset_keys(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)

    f.visititems(get_dataset_keys)
    if isinstance(f_input, str):
        f.close()
    return hdf5_dataset_key_list


def recursive_dict_merge(dict1, dict2):
    """
    Returns a new dictionary by merging two dictionaries recursively.
    This function is useful for minimally updating dict1 with dict2.
    """
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.Mapping):
            result[key] = recursive_dict_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])
    return result


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
