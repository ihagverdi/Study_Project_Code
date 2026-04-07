from copy import deepcopy
import pathlib
import pickle
import platform
import pandas as pd
import warnings
import torch
import numpy as np
from scipy.stats import wilcoxon

def generate_experiment_id(cfg) -> str:
    """
    Generates a unique, human-readable identifier based on the experiment configuration.
    This ID is used to ensure that files are not overwritten when hyperparameters change.
    """
    # We use a list of tuples (label, value) to keep the ID organized and readable
    params = [
        (None, cfg.model_name),
        (None, cfg.scenario),
        (None, str(cfg.fold)),
        ("samp", f"{cfg.num_samples_per_instance}_s{cfg.seed_samples_per_instance}"),
        ("ctx", f"{cfg.context_size}_{cfg.subsample_method}_s{cfg.seed_context_size}"),
        ("drop", f"{cfg.feature_drop_rate}_s{cfg.seed_feature_drop_rate}"),
        ("agnostic", cfg.feature_agnostic),
        ("scale", cfg.target_scale),
        ("oracle", cfg.oracle),
        ("es", cfg.early_stopping),
        (None, 'cpu' if cfg.use_cpu else 'gpu')
    ]
    
    parts = [f"{label}{val}" if label else str(val) for label, val in params]
    
    return "_".join(parts)

class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PosixPath' and 'pathlib' in module:
            return pathlib.WindowsPath
        return super().find_class(module, name)

def find_optimal_context(results_list, scenario, target_metric, alpha=0.05, display_results=True):
    """
    Rigorously determines the optimal context size using the Principle of Parsimony 
    and a one-sided Wilcoxon signed-rank test.
    
    Parameters:
    - results_list: List of dictionaries loaded from the .pkl file
    - scenario: String, the dataset name to filter by
    - target_metric: String, e.g., "CRPS", "NLLH", "Wasserstein", "KS"
    - alpha: Float, the statistical significance threshold (default 0.05)
    
    Returns:
    - optimal_context: The selected context size (integer)
    - summary_df: A pandas DataFrame containing the thesis-ready results table
    """
    
    # ---------------------------------------------------------
    # STEP 1: Data Extraction & Instance Aggregation
    # ---------------------------------------------------------
    records = []
    for run in results_list:
        if run['scenario'] == scenario:
            # Aggregate the 200 instances by taking the mean for the target metric
            run_data = run['instance_summary'][target_metric]
            if hasattr(run_data, 'cpu'):
                run_data = run_data.detach().cpu().numpy()
            run_score = np.mean(run_data)
            records.append({
                'context_size': run['context_size'],
                'fold': run['fold'],
                'seed': run['context_seed'],
                'score': run_score
            })
            
    if not records:
        raise ValueError(f"No data found for scenario '{scenario}'. Please check the name.")
        
    df = pd.DataFrame(records)
    
    # ---------------------------------------------------------
    # STEP 2: Seed Aggregation (Handling intra-fold correlation)
    # ---------------------------------------------------------
    # Average the 5 seeds for every (context_size, fold) combination
    fold_df = df.groupby(['context_size', 'fold'])['score'].mean().reset_index()
    
    # ---------------------------------------------------------
    # STEP 3: Identify the Empirical Best (C_best)
    # ---------------------------------------------------------
    # Calculate the grand mean across the 10 independent folds
    summary_df = fold_df.groupby('context_size')['score'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.sort_values('context_size').reset_index(drop=True)
    
    # Find the context_size with the absolute lowest mean (since lower error is better)
    c_best_idx = summary_df['mean'].idxmin()
    c_best = int(summary_df.loc[c_best_idx, 'context_size'])
    
    # Extract the 10 matched fold scores for the empirical best, ensuring order by fold
    scores_best = fold_df[fold_df['context_size'] == c_best].sort_values('fold')['score'].values
    
    # ---------------------------------------------------------
    # STEP 4: Wilcoxon-Parsimony Test
    # ---------------------------------------------------------
    results_log =[]
    optimal_context = None
    
    for _, row in summary_df.iterrows():
        c_test = int(row['context_size'])
        mean_test = row['mean']
        std_test = row['std']
        
        # Extract the 10 matched fold scores for the current candidate
        scores_test = fold_df[fold_df['context_size'] == c_test].sort_values('fold')['score'].values
        
        if c_test == c_best:
            p_value = 1.0  # Comparing to itself
            is_worse = False
            status = "*** Empirical Best ***"
        else:
            # One-sided Wilcoxon test
            # H0: c_test and c_best are symmetric.
            # Ha (greater): scores_test > scores_best (i.e., c_test is strictly WORSE than c_best)
            try:
                # Suppress scipy ties warning temporarily to keep terminal clean
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = wilcoxon(scores_test, scores_best, alternative='greater')
            except ValueError:
                # Triggers if differences are exactly 0 across all 10 folds
                print(f"differences are exactly 0 across all 10 folds")
                p_value = 1.0 
                
            is_worse = p_value < alpha
            status = "Significantly Worse" if is_worse else "Statistical Tie"
            
        # Log the result for the thesis table
        results_log.append({
            'Context Size': c_test,
            f'Mean {target_metric}': mean_test,
            f'Std {target_metric}': std_test,
            'p-value (vs Best)': p_value,
            'Status': status
        })
        
        # Apply Parsimony Rule: 
        # Select the *smallest* context size that is <= c_best and NOT significantly worse.
        # Since we iterate in ascending order, the very first non-worse candidate is our optimal!
        if not is_worse and optimal_context is None and c_test <= c_best:
            optimal_context = c_test

    # ---------------------------------------------------------
    # STEP 5: Formatting and Thesis-Ready Output
    # ---------------------------------------------------------
    log_df = pd.DataFrame(results_log)
    # Update the status string of the chosen optimal context for display purposes
    idx_optimal = log_df.index[log_df['Context Size'] == optimal_context].tolist()[0]
    if optimal_context != c_best:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN (Parsimony)"
    else:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN"
        
    if display_results:        
        # Formatting for terminal beauty
        print(f"\n{'='*75}")
        print(f" Parsimony Analysis | Scenario: {scenario} | Metric: {target_metric}")
        print(f"{'='*75}")
        print(f"Empirical Best Context Size (C_best) : {c_best}")
        print(f"Optimal Context Size Chosen          : {optimal_context}")
        print("-" * 75)
        
        # Pandas display formatting
        formatters = {
            f'Mean {target_metric}': '{:.5f}'.format,
            f'Std {target_metric}': '{:.5f}'.format,
            'p-value (vs Best)': '{:.4f}'.format
        }
        print(log_df.to_string(index=False, formatters=formatters))
        print(f"{'='*75}\n")
    
    return optimal_context, log_df

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

def load_pickle(path, access_mode='rb'):
    with open(path, access_mode) as f:
        # Use our custom Unpickler on Windows, otherwise standard pickle
        if platform.system() == 'Windows':
            results_dict = WindowsPathUnpickler(f).load()
        else:
            results_dict = pickle.load(f)
    return results_dict

def subsample_flattened_data(*arrays, context_size, seed, subsample_method):
    """
    Subsamples the flattened dataset. Currently supports 'flatten-random' which randomly samples from the flattened training data.
    
    Args:
        *arrays: Any number of numpy arrays (e.g., X_train_flat, y_train_flat, instance_ids_flat). 
                 All must have the same length in the first dimension.
        context_size: Total number of items to return.
        seed: Random seed for reproducibility.
        subsample_method: Strategy for sampling ('flatten-random')
        
    Returns:
        Tuple of subsampled arrays corresponding to the inputs.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
        
    n_samples = arrays[0].shape[0]
    
    # Assert all arrays have the same number of instances
    assert all(arr.shape[0] == n_samples for arr in arrays), \
        "All arrays must have the same number of instances (shape[0])."
    
    rng = np.random.default_rng(seed)
    
    if subsample_method == 'flatten-random':
        selected_indices = rng.choice(n_samples, size=context_size, replace=True)
        # Apply the selected indices to all provided arrays
        return tuple(arr[selected_indices] for arr in arrays)
    else:
        raise ValueError(f"Unknown subsample_method: {subsample_method}")

def subsample_features(X_train, *arrays, drop_rate, seed):
    """
    Randomly samples a subset of features from the input arrays based on the specified drop rate.
    If drop_rate >= 1.0, implements a strict marginal baseline (0 features) via a dummy column.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features) to subsample features from
        drop_rate: Fraction of features to drop (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of subsampled arrays, with the same order as input (X_train, *arrays)
    """
    
    # 1. Handle the "0 Features" Marginal Baseline
    if drop_rate >= 1.0:
        dummy_X_train = X_train[:, :1] * 0.0
        processed_arrays = [arr[:, :1] * 0.0 for arr in arrays]
        
        return (dummy_X_train, *processed_arrays)

    # 2. Handle standard feature subsampling
    rng = np.random.default_rng(seed=seed)
    n_features = X_train.shape[1]
    
    # Calculate how many features to KEEP. 
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

def load_tabpfn_preds(cfg, tabpfn_preds_dir):
    exp_id = generate_experiment_id(cfg)
    fname = f"tabpfn_{exp_id}_test_preds.pkl"
    fpath = pathlib.Path(tabpfn_preds_dir) / fname

    with open(fpath, "rb") as f:
        if platform.system() == "Windows":
            from tabpfn_project.helper.utils import WindowsPathUnpickler 
            return WindowsPathUnpickler(f).load()
        return pickle.load(f)

def fetch_save_dict(
    results_dir: pathlib.Path,
    metadata_dir: pathlib.Path,
    model_save_name: str,
    model_name: str,
    search_key: str = None,
    search_value=None,
    scenario: str = None,
) -> None:
    """
    Build and save a normalized list of experiment results filtered by model/scenario
    and optional key/value filter.
    """
    experiment_results_lst = []

    for fpath in sorted(metadata_dir.glob("*.pkl")):
        with open(fpath, "rb") as f:
            if platform.system() == "Windows":
                results_dict = WindowsPathUnpickler(f).load()
            else:
                results_dict = pickle.load(f)

        saved_model = results_dict.get("model_name")
        if saved_model != model_name:
            continue

        # Optional scenario filter
        if scenario is not None and results_dict.get("scenario") != scenario:
            continue

        # Optional arbitrary key-value filter
        if search_key is not None and results_dict.get(search_key) != search_value:
            continue

        context_size = results_dict.get("context_size")
        if context_size in {2**13 + 2000, 2**13 + 4000}:
            continue

        result_metrics = results_dict.get("result_metrics", {})
        model_specific_info = results_dict.get("model_specific_info", {})

        metrics_summary = result_metrics.get("metrics_summary")
        instance_summary = result_metrics.get("instance_summary")
        y_preds = results_dict.get("y_test_preds")

        # Keep legacy field names expected by notebook code
        best_params = results_dict.get("best_params")

        # Times are model-specific in current schema
        fit_time = model_specific_info.get("fit_time")
        predict_time = model_specific_info.get("predict_time")
        gpu_metrics = model_specific_info.get("mem_time_stats")
        hpo_time = results_dict.get("hpo_time")

        temp = {
            "scenario": results_dict.get("scenario"),
            "model": model_save_name,
            "context_size": context_size,
            "fold": results_dict.get("fold"),
            "context_seed": results_dict.get("seed_context_size", results_dict.get("seed_context")),
            "feature_drop_rate": results_dict.get("feature_drop_rate"),
            "seed_feature_drop_rate": results_dict.get("seed_feature_drop_rate", results_dict.get("seed_features")),
            "metrics_summary": metrics_summary,
            "instance_summary": instance_summary,
            "best_params": best_params,
            "y_preds": y_preds,
            "target_scale": results_dict.get("target_scale"),
            "fit_time": fit_time,
            "predict_time": predict_time,
            "hpo_time": hpo_time,
            "gpu_metrics": gpu_metrics,
            "use_cpu": results_dict.get("use_cpu"),
        }

        experiment_results_lst.append(temp)

    scenario_label = scenario if scenario is not None else "all_scenarios"
    if search_key is None:
        save_name = f"{model_save_name}_{scenario_label}.pkl"
    else:
        save_name = f"{model_save_name}_{search_key}_{search_value}_{scenario_label}.pkl"

    save_file_path = results_dir / save_name
    with open(save_file_path, "wb") as f:
        pickle.dump(experiment_results_lst, f)

    print(f"Saved to {save_file_path}")