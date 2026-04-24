import contextlib
from copy import deepcopy
import gc
import pathlib
import pickle
import platform
import time
from typing import Optional, Union
import pandas as pd
import warnings
import torch
import numpy as np
from scipy.stats import wilcoxon

@contextlib.contextmanager
def track_gpu_memory_and_time(device_input):
    is_cuda = False
    try:
        device = torch.device(device_input)
        if device.type == 'cuda' and torch.cuda.is_available():
            is_cuda = True
    except Exception:
        pass

    stats = {"baseline_mb": 0.0, "peak_mb": 0.0, "spike_mb": 0.0, "time_s": 0.0}
    if is_cuda:
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        baseline_mem_bytes = torch.cuda.memory_allocated(device)
    
    start_time = time.perf_counter()
    yield stats
    
    if is_cuda:
        torch.cuda.synchronize(device)
        
    stats["time_s"] = time.perf_counter() - start_time
    if is_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        stats["baseline_mb"] = baseline_mem_bytes / (1024 ** 2)
        stats["peak_mb"] = peak_mem_bytes / (1024 ** 2)
        stats["spike_mb"] = (peak_mem_bytes - baseline_mem_bytes) / (1024 ** 2)

def sample_k_per_instance(X, y, ids, k, seed):
    """
    Samples up to k random samples for each unique instance id without replacement.
    
    Args:
        X (np.ndarray): Flattened features of shape (M, D)
        y (np.ndarray): Flattened targets of shape (M, 1)
        ids (np.ndarray): Flattened instance ids of shape (M,)
        k (int): Max number of samples to keep per unique id.
        seed (int): Random seed for reproducibility.
        
    Returns:
        X_sampled, y_sampled, ids_sampled
    """
    rng = np.random.RandomState(seed)
    
    # 1. Generate random noise for every single row
    # This determines the "random selection" when we sort
    noise = rng.rand(len(ids))
    
    # 2. Sort by instance_id (primary) and then by noise (secondary)
    # np.lexsort sorts by the last sequence provided first.
    # This groups all identical IDs together, but shuffles them internally.
    sorted_indices = np.lexsort((noise, ids))
    
    # 3. Reorder the ids to find where each group starts
    sorted_ids = ids[sorted_indices]
    
    # 4. Identify boundaries between different instance IDs
    # mask[i] is True if sorted_ids[i] is the start of a new ID group
    mask = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    
    # 5. Calculate the rank of each element within its own ID group
    # Find the index where each group starts
    start_indices = np.where(mask)[0]
    # For every row, find the index where its group started
    # np.diff calculates the size of each group; np.repeat maps the start_index to all rows in that group
    group_offsets = np.repeat(start_indices, np.diff(np.append(start_indices, len(ids))))
    
    # Rank is: (Current Index in sorted array) - (Index where this ID group started)
    # This creates sequences like: [0, 1, 2, 0, 1, 0, 1, 2, 3...]
    ranks = np.arange(len(ids)) - group_offsets
    
    # 6. Filter: Keep only those with a rank < k
    # This naturally handles cases where an instance has fewer than k samples
    keep_mask = ranks < k
    
    # 7. Map the mask back to the original data indexing
    final_indices = sorted_indices[keep_mask]
    
    return X[final_indices], y[final_indices], ids[final_indices]

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
        ("remove_dups", cfg.remove_duplicates),
        ("scale", cfg.target_scale),
        ("oracle", cfg.oracle),
        ("es", cfg.early_stopping),
        ("jitterX", f"{cfg.jitter_x}_val{cfg.jitter_val}"),
        ("randExt", f"{cfg.rand_extend_x}_n{cfg.n_rand_cols}"),
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

def subsample_data(*arrays, context_size, seed, subsample_method='flatten-random', with_replacement=True):
    """
    Subsamples the dataset. Currently supports 'flatten-random' which randomly samples from the training data.
    
    Args:
        *arrays: Any number of numpy arrays (e.g., X_train, y_train, instance_ids). 
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
        selected_indices = rng.choice(n_samples, size=context_size, replace=with_replacement)
        # Apply the selected indices to all provided arrays
        return tuple(arr[selected_indices] for arr in arrays)
    else:
        raise ValueError(f"Unknown subsample_method: {subsample_method}")

def append_random_columns(
    *arrays,
    n_random_cols: int = 1, 
    random_state: Optional[Union[int, np.random.Generator]] = None
):
    """
    Augments the feature matrices by appending entirely random continuous columns.
    
    Args:
        X (np.ndarray): The flattened, standardized feature matrix of shape (N, d).
        n_random_cols (int): The number of independent random columns to append.
        random_state (int, Generator, optional): Seed or RNG for reproducibility.
        
    Returns:
        List[np.ndarray, ...]: Augmented feature matrices, each of shape (N, d + n_random_cols).
    """
    N = arrays[0].shape[0]
    D = arrays[0].shape[1]
    assert all(arr.shape[1] == D for arr in arrays), "All input arrays must have the same number of features (shape[1])."

    if n_random_cols < 1:
        return arrays  # No augmentation needed, return original arrays
        
    rng = np.random.default_rng(random_state)
    
    augmented_arrays = tuple(np.hstack((arr, rng.standard_normal(
        size=(arr.shape[0], n_random_cols), 
        dtype=np.float64
    ))) for arr in arrays)
    
    return augmented_arrays

def add_feature_jitter(
    X: np.ndarray, 
    jitter_intensity: float, 
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Injects numerically stable, zero-mean Gaussian noise into feature matrix X.
    
    Args:
        X (np.ndarray): Input feature matrix of shape (N, d).
        jitter_intensity (float): The relative scaling factor for the noise (alpha).
                                  Default is 1e-3.
        random_state (int, Generator, optional): Seed or RNG for reproducibility.
        
    Returns:
        np.ndarray: A new jittered feature matrix of shape (N, d).
    """
    X_jittered = np.array(X, dtype=np.float64, copy=True)
    
    rng = np.random.default_rng(random_state)
    
    base_std = np.std(X_jittered, axis=0)
        
    noise_stds = jitter_intensity * base_std
    
    noise = rng.normal(loc=0.0, scale=noise_stds, size=X_jittered.shape)
    
    return X_jittered + noise

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

    assert all(arr.shape[1] == n_features for arr in arrays), (
        "All input arrays must have the same number of features (shape[1])."
    )
    
    # Calculate how many features to KEEP. 
    size_features = max(1, int(n_features * (1 - drop_rate)))
    
    feature_idx = rng.choice(n_features, size=size_features, replace=False)
    
    processed_arrays = [arr[:, feature_idx] for arr in arrays]

    return (X_train[:, feature_idx], *processed_arrays)

def subsample_targets_per_instance(y_train, num_samples_per_instance, seed_samples_per_instance):
    """
    Subsamples a specified number of samples per instance from the training data independently per row.
    
    Args:
        y_train: (n_instances, n_samples) - The training labels.
        num_samples_per_instance: The number of samples to subsample per instance.
        seed_samples_per_instance: The random seed for reproducibility.

    Returns:
        y_train_subsampled: (n_instances, num_samples_per_instance) - The subsampled training labels.
    """
    rng = np.random.default_rng(seed=seed_samples_per_instance)
    
    # Generate a matrix of random floats with the same shape as y_train
    rand_matrix = rng.random(y_train.shape)
    
    # Argsort each row to get random permutations of indices per row.
    # Take the first `num_samples_per_instance` indices for each row.
    # This mathematically guarantees independent sampling WITHOUT replacement per row.
    subsample_idx = np.argsort(rand_matrix, axis=1)[:, :num_samples_per_instance]
    
    # Create row indices for advanced indexing
    row_idx = np.arange(y_train.shape[0])[:, None]
    
    # Extract the independently sampled runtimes
    y_train_subsampled = y_train[row_idx, subsample_idx]
    
    return y_train_subsampled

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
    model_name: str,
    save_name: str,
    search_key: str | None = None,
    search_value=None,
    scenario: str | None = None,
) -> None:
    """
    Build and save a normalized list of experiment results filtered by model/scenario
    and optional key/value filter, aligned with current metadata produced by scripts/main.py
    and scripts/model_handler.py.

    Notes:
    - Keeps backward-compat aliases used by older notebooks.
    - Resolves overlap fields explicitly (HPO/memory metrics).
    - Preserves unknown future fields in dedicated pass-through buckets.
    """
    experiment_results_lst = []

    # Core top-level fields produced by scripts/main.py (+ merged model outputs).
    # Keep this list explicit so schema drift is visible and controlled.
    expected_top_level = {
        "model_name",
        "scenario",
        "fold",
        "seed_context_size",
        "seed_feature_drop_rate",
        "seed_samples_per_instance",
        "jitter_x",
        "jitter_val",
        "rand_extend_x",
        "n_rand_cols",
        "subsample_from_unflattened",
        "feature_drop_rate",
        "context_size",
        "target_scale",
        "subsample_method",
        "num_samples_per_instance",
        "use_cpu",
        "save_dir",
        "n_samples",
        "n_features",
        "instance_ids",
        "feature_agnostic",
        "remove_duplicates",
        "oracle",
        "do_hpo",
        "y_test_preds",
        "result_metrics",
        "model_specific_info",
    }

    # Legacy/optional top-level fields that may appear in older or alternate runs.
    optional_top_level = {
        "early_stopping",
        "hpo_time",
        "hpo_results",
    }

    # Explicitly normalized keys from model_specific_info across handlers.
    normalized_model_specific_keys = {
        "fit_time",
        "predict_time",
        "mem_time_stats",
        "gpu_metrics",
        "y_scale",
        "best_epoch",
        "val_batch_size",
        "n_epochs",
        "model_config",
        "best_hyperparameters",
        # RF HPO run diagnostics
        "num_unique_configs",
        "num_finished_trials",
        "num_submitted_trials",
        # Optional legacy/alt locations
        "hpo_time",
        "hpo_results",
    }

    schema_warning_count = 0

    for fpath in sorted(metadata_dir.glob("*.pkl")):
        results_dict = load_pickle(fpath)

        if results_dict.get("model_name") != model_name:
            continue

        if scenario is not None and results_dict.get("scenario") != scenario:
            continue

        if search_key is not None and results_dict.get(search_key) != search_value:
            continue

        
        result_metrics = results_dict.get("result_metrics") or {}
        model_specific_info = results_dict.get("model_specific_info") or {}

        # Explicit overlap resolution:
        # 1) hpo_results priority: top-level -> model_specific_info.hpo_results -> best_hyperparameters
        hpo_results = results_dict.get("hpo_results")
        if hpo_results is None:
            hpo_results = model_specific_info.get("hpo_results")
        if hpo_results is None:
            hpo_results = model_specific_info.get("best_hyperparameters")

        # 2) hpo_time priority: top-level -> model_specific_info
        hpo_time = results_dict.get("hpo_time")
        if hpo_time is None:
            hpo_time = model_specific_info.get("hpo_time")

        # 3) memory/time compatibility:
        #    TabPFN uses mem_time_stats; some consumers expect gpu_metrics.
        mem_time_stats = model_specific_info.get("mem_time_stats")
        gpu_metrics = model_specific_info.get("gpu_metrics")
        if gpu_metrics is None:
            gpu_metrics = mem_time_stats

        temp = {
            # Normalized label used by notebooks/plots
            "model_name": results_dict.get("model_name"),

            # Caller-side tag used in downstream aggregation scripts
            "save_name": save_name,

            # Core experiment config / identifiers from top-level metadata
            "scenario": results_dict.get("scenario"),
            "fold": results_dict.get("fold"),
            "context_size": results_dict.get("context_size"),
            "seed_context_size": results_dict.get("seed_context_size"),
            "seed_feature_drop_rate": results_dict.get("seed_feature_drop_rate"),
            "seed_samples_per_instance": results_dict.get("seed_samples_per_instance"),
            "feature_drop_rate": results_dict.get("feature_drop_rate"),
            "target_scale": results_dict.get("target_scale"),
            "subsample_method": results_dict.get("subsample_method"),
            "subsample_from_unflattened": results_dict.get("subsample_from_unflattened"),
            "num_samples_per_instance": results_dict.get("num_samples_per_instance"),
            "use_cpu": results_dict.get("use_cpu"),
            "save_dir": results_dict.get("save_dir"),
            "n_samples": results_dict.get("n_samples"),
            "n_features": results_dict.get("n_features"),
            # "instance_ids": results_dict.get("instance_ids"),
            "feature_agnostic": results_dict.get("feature_agnostic"),
            "remove_duplicates": results_dict.get("remove_duplicates"),
            "oracle": results_dict.get("oracle"),
            "do_hpo": results_dict.get("do_hpo"),
            "early_stopping": results_dict.get("early_stopping"),
            "hpo_time": hpo_time,
            "hpo_results": hpo_results,

            # Newly introduced main.py args that were previously omitted
            "jitter_x": results_dict.get("jitter_x"),
            "jitter_val": results_dict.get("jitter_val"),
            "rand_extend_x": results_dict.get("rand_extend_x"),
            "n_rand_cols": results_dict.get("n_rand_cols"),

            # Current model outputs
            "y_test_preds": results_dict.get("y_test_preds"),
            # "metrics_summary": result_metrics.get("metrics_summary"),
            "instance_summary": result_metrics.get("instance_summary"),

            # Model-specific normalized block
            "fit_time": model_specific_info.get("fit_time"),
            "predict_time": model_specific_info.get("predict_time"),
            "mem_time_stats": mem_time_stats,
            "gpu_metrics": gpu_metrics,
            "y_scale": model_specific_info.get("y_scale"),
            "best_epoch": model_specific_info.get("best_epoch"),
            "val_batch_size": model_specific_info.get("val_batch_size"),
            "n_epochs": model_specific_info.get("n_epochs"),
            "model_config": model_specific_info.get("model_config"),
            "best_hyperparameters": model_specific_info.get("best_hyperparameters"),
            "num_unique_configs": model_specific_info.get("num_unique_configs"),
            "num_finished_trials": model_specific_info.get("num_finished_trials"),
            "num_submitted_trials": model_specific_info.get("num_submitted_trials"),
            "ensemble": model_specific_info.get("ensemble"),
            "ensemble_size": model_specific_info.get("ensemble_size"),

            # Keep raw nested blocks for forward-compat consumers
            "result_metrics": result_metrics,
            "model_specific_info": model_specific_info,

            # Backward-compat aliases used by older notebook logic
            "context_seed": results_dict.get("seed_context_size"),
            "y_preds": results_dict.get("y_test_preds"),
        }

        # Extensibility: preserve future unknown keys without flattening duplicate semantics.
        known_top = expected_top_level | optional_top_level
        temp["extra_top_level"] = {
            k: v
            for k, v in results_dict.items()
            if k not in known_top
        }

        temp["extra_model_specific_info"] = {
            k: v
            for k, v in model_specific_info.items()
            if k not in normalized_model_specific_keys
        }

        experiment_results_lst.append(temp)

    scenario_label = scenario if scenario is not None else "all_scenarios"
    if search_key is None:
        output_name = f"{save_name}_{scenario_label}.pkl"
    else:
        output_name = f"{save_name}_{search_key}_{search_value}_{scenario_label}.pkl"

    save_file_path = results_dir / output_name
    with open(save_file_path, "wb") as f:
        pickle.dump(experiment_results_lst, f)

    print(f"Saved to {save_file_path}")
    if schema_warning_count:
        print(
            f"[fetch_save_dict] Completed with schema warnings in "
            f"{schema_warning_count} metadata file(s)."
        )