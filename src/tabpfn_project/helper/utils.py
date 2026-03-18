from copy import deepcopy
import pathlib
import pickle
import platform
import numpy as np
import torch

def dict_to_cpu(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu()
        elif isinstance(v, dict):
            result[k] = dict_to_cpu(v)
        elif hasattr(v, 'cpu') and callable(v.cpu):
            result[k] = deepcopy(v).cpu()
        else:
            result[k] = v
    return result

class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PosixPath' and 'pathlib' in module:
            return pathlib.WindowsPath
        return super().find_class(module, name)

def load_pickle(path, access_mode='rb'):
    with open(path, access_mode) as f:
        # Use our custom Unpickler on Windows, otherwise standard pickle
        if platform.system() == 'Windows':
            results_dict = WindowsPathUnpickler(f).load()
        else:
            results_dict = pickle.load(f)
    return results_dict

def subsample_flattened_data(X_train_flat, y_train_flat, context_size, seed, subsample_method):
    """
    Subsamples the flattened dataset. Currently supports 'flatten-random' which randomly samples from the flattened training data.
    
    Args:
        X_train_flat: (n_instances, n_features)
        y_train_flat: (n_instances, 1)
        context_size: Total number of (X, y) pairs to return.
        seed: Random seed for reproducibility.
        subsample_method: Strategy for sampling ('flatten-random')
        
    Returns:
        X_out: (context_size, n_features)
        y_out: (context_size, 1)
    """
    rng = np.random.default_rng(seed)
    n_samples = X_train_flat.shape[0]
    
    if subsample_method == 'flatten-random':
        selected_indices = rng.choice(n_samples, size=context_size, replace=True)
        X_out = X_train_flat[selected_indices]
        y_out = y_train_flat[selected_indices]
        return X_out, y_out

def subsample_features(X_train, *arrays, drop_rate, seed):
    """
    Randomly samples a subset of features from the input arrays based on the specified drop rate.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features) to subsample features from
        drop_rate: Fraction of features to drop (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of subsampled arrays, with the same order as input (X_train, *arrays)

    """

    rng = np.random.default_rng(seed=seed)
    n_features = X_train.shape[1]

    if drop_rate == 1:  # dropping all but one feature
        size_features = 1
    else:
        size_features = max(1, int(n_features * (1 - drop_rate)))
        
    feature_idx = rng.choice(n_features, size=size_features, replace=False)
    processed_arrays = [arr[:, feature_idx] for arr in arrays]

    return (X_train[:, feature_idx], *processed_arrays)


def subsample_targets_per_instance(y_train, num_samples_per_instance, seed_samples_per_instance):
    """
    Subsamples a specified number of samples per instance from the training data.
    
    Args:
        y_train: (n_instances, n_samples) - The training labels.
        num_samples_per_instance: The number of samples to subsample per instance.
        seed_samples_per_instance: The random seed for reproducibility.

    Returns:
        y_train: (n_instances, num_samples_per_instance) - The subsampled training labels.
    """
    rng = np.random.default_rng(seed=seed_samples_per_instance)
    subsample_idx = rng.choice(y_train.shape[1], size=num_samples_per_instance, replace=False)
    y_train = y_train[:, subsample_idx]
    return y_train