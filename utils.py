# @Author: CAG
# @Github: CAgAG
# @Encode: UTF-8
# @FileName: utils.py

import sys, os, warnings, numbers
import shutil

import requests
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


class PrintToFile(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.file = open(self.file_path, 'a', encoding="utf-8")

    def __enter__(self):
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        self.file.close()


def set_top_k_to_one(arr, k):
    top_k_indices = np.argpartition(arr, -k)[-k:]
    result = np.zeros_like(arr)
    result[top_k_indices] = 1
    return result


def eval_roc_auc(labels, pred):
    roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    return roc_auc


# ===================================================================
# https://github.com/pygod-team/pygod/blob/v0.3.1/pygod/utils/utility.py
MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def validate_device(gpu_id):
    """Validate the input device id (GPU id) is valid on the given
    machine. If no GPU is presented, return 'cpu'.

    Parameters
    ----------
    gpu_id : int
        GPU id to be used. The function will validate the usability
        of the GPU. If failed, return device as 'cpu'.

    Returns
    -------
    device_id : str
        Valid device id, e.g., 'cuda:0' or 'cpu'
    """
    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(), param_name='gpu id', include_left=True,
                        include_right=False)
        device_id = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device_id = 'cpu'

    return device_id


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def load_data(name, cache_dir=None):
    """
    Data loading function. See `data repository
    <https://github.com/pygod-team/data>`_ for supported datasets.
    For injected/generated datasets, the labels meanings are as follows.

    - 0: inlier
    - 1: contextual outlier only
    - 2: structural outlier only
    - 3: both contextual outlier and structural outlier

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The outlier dataset.

    Examples
    --------
    >>> from pygod.utils import load_data
    >>> data = load_data(name='weibo') # in PyG format
    >>> y = data.y.bool()    # binary labels (inlier/outlier)
    >>> yc = data.y >> 0 & 1 # contextual outliers
    >>> ys = data.y >> 1 & 1 # structural outliers
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name + '.pt')
    zip_path = os.path.join(cache_dir, name + '.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path)
    return data
# ===================================================================
