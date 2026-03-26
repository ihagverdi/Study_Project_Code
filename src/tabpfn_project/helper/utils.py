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
        elif isinstance(v, (list, tuple)):
            processed_iterable = [
                vi.detach().cpu() if isinstance(vi, torch.Tensor) 
                else dict_to_cpu(vi) if isinstance(vi, dict) 
                else vi 
                for vi in v
            ]
            result[k] = type(v)(processed_iterable) # keep it a list or tuple
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

    
def load_tabpfn_preds(
    tabpfn_preds_dir,
    scenario_name,
    fold_idx,
    context_size,
    context_seed,
    seed_features,
    seed_samples_per_instance,
    feature_drop_rate,
    target_scale,
    subsample_method,
    num_samples_per_instance,
    use_cpu,
 ):
    device_tag = "cpu" if use_cpu else "gpu"
    fname = (
        f"tabpfn_{scenario_name}_{fold_idx}_{context_seed}_{seed_features}_{seed_samples_per_instance}_{feature_drop_rate}_"
        f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{device_tag}_test_preds.pkl"
    )
    fpath = tabpfn_preds_dir / fname

    with open(fpath, "rb") as f:
        if platform.system() == "Windows":
            return WindowsPathUnpickler(f).load()
        return pickle.load(f)

def fetch_save_dict(results_dir: pathlib.Path, metadata_dir: pathlib.Path, model_name: str, scenario: str = None) -> None:
    """Build and save a nested results dictionary filtered by model and scenario."""
    experiment_results_lst = []

    for fpath in sorted(metadata_dir.glob("*.pkl")):
        with open(fpath, "rb") as f:
            if platform.system() == "Windows":
                results_dict = WindowsPathUnpickler(f).load()
            else:
                results_dict = pickle.load(f)

        if scenario is not None and results_dict.get("scenario") != scenario:
            continue
        if results_dict.get("model_name") != model_name:
            continue

        context_size = results_dict["context_size"]
        if context_size in {2**13 + 2000, 2**13 + 4000}:
            continue

        metrics_summary = None
        instance_summary = None
        best_params = None
        y_preds = None
        fit_time = None
        predict_time = None
        fit_gpu = None
        predict_gpu = None
        hpo_time = None


        if model_name == "baseline":
            metrics_summary = results_dict['test_preds'][0]
            instance_summary = results_dict['test_preds'][1]

        elif model_name == "rf_baseline":
            metrics_summary = results_dict['result_metrics']['metrics_summary']
            instance_summary = results_dict['result_metrics']['instance_summary']
            best_params = results_dict['best_params']
            y_preds = results_dict['test_preds']  # [rf_means, rf_variances]
            fit_time = results_dict['result_metrics']['fit_time']
            predict_time = results_dict['result_metrics']['predict_time']
            hpo_time = results_dict['result_metrics']['hpo_time']

        elif model_name == "tabpfn":
            fit_gpu = results_dict['result_metrics']['fit']
            predict_gpu = results_dict['result_metrics']['predict']

        temp = {
            "scenario": results_dict["scenario"],
            "model": results_dict["model_name"],
            "context_size": results_dict["context_size"],
            "fold": results_dict["fold"],
            "context_seed": results_dict["seed_context"],
            "metrics_summary": metrics_summary,
            "instance_summary": instance_summary,
            "best_params": best_params,
            "y_preds": y_preds,
            "target_scale": results_dict["target_scale"],
            "fit_time": fit_time,
            "predict_time": predict_time,
            "hpo_time": hpo_time,
            "fit_gpu": fit_gpu,
            "predict_gpu": predict_gpu,
            "use_cpu": results_dict["use_cpu"],
        }

        experiment_results_lst.append(temp)

    if scenario is None:
        scenario = "all_scenarios"
    save_file_path = results_dir / f"{model_name}_{scenario}.pkl"
    with open(save_file_path, "wb") as f:
        pickle.dump(experiment_results_lst, f)

    print(f"Saved to {save_file_path}")