import argparse
import contextlib
import gc
import os
import pickle
import time
from ConfigSpace import CategoricalHyperparameter
import numpy as np
import torch
from tabpfn_project.helper.load_data import load_distnet_data
from tabpfn_project.helper.preprocess import (
    delete_constant_features,
    preprocess_features,
)
from tabpfn_project.helper.utils import (
    subsample_features, 
    subsample_flattened_data, 
    subsample_targets_per_instance
)
from tabpfn_project.globals import (
    DISTNET_SCENARIOS, MODELS, 
    N_GRID_POINTS, RANDOM_STATE, 
    SUBSAMPLE_METHOD_CHOICES, TARGET_SCALES
)
from sklearn.model_selection import KFold, train_test_split
from tabpfn_project.helper.y_scalers import max_scaling, log_scaling
from tabpfn_project.paths import RESULTS_DIR, DISTNET_DATA_DIR

@contextlib.contextmanager
def track_gpu_memory_and_time(device_input):
    is_cuda = False
    try:
        device = torch.device(device_input)
        if device.type == 'cuda' and torch.cuda.is_available():
            is_cuda = True
    except Exception:
        pass

    # Add time_s to the dictionary
    stats = {"baseline_mb": 0.0, "peak_mb": 0.0, "spike_mb": 0.0, "time_s": 0.0}

    if is_cuda:
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Sync before starting the timer to ensure a clean slate
        torch.cuda.synchronize(device)
        baseline_mem_bytes = torch.cuda.memory_allocated(device)
    
    # --- START CLOCK ---
    start_time = time.perf_counter()
    
    yield stats
    
    # --- STOP CLOCK ---
    # First, force GPU to finish the user's operation
    if is_cuda:
        torch.cuda.synchronize(device)
        
    end_time = time.perf_counter()
    stats["time_s"] = end_time - start_time
    
    # Calculate memory
    if is_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        operation_spike_bytes = peak_mem_bytes - baseline_mem_bytes
        
        stats["baseline_mb"] = baseline_mem_bytes / (1024 ** 2)
        stats["peak_mb"] = peak_mem_bytes / (1024 ** 2)
        stats["spike_mb"] = operation_spike_bytes / (1024 ** 2)
  
def train_test_model(
    model_name,
    scenario,
    fold,
    save_dir,
    num_samples_per_instance,
    context_size,
    use_cpu,
    target_scale,
    subsample_method,
    early_stopping,
    seed_context_size,
    seed_feature_drop_rate,
    feature_drop_rate,
    val_batch_size,
    seed_samples_per_instance,
    n_epochs,
    batch_size,
    wc_time_limit,
    do_hpo,
    hpo_time,
    feature_agnostic,
    oracle,
):
    assert 0 <= fold <= 9, "Fold must be between 0 and 9"
    
    # Create save directory if it doesn't exist
    clean_name = save_dir.lstrip('/\\')
    save_dir = RESULTS_DIR / clean_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get scenario configuration and data

    # Load data
    X_train, X_test, y_train, y_test = load_distnet_data(DISTNET_DATA_DIR, scenario, fold, return_all=False)

    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "X and y must have the same length."
    assert y_train.shape[1] == 100 and y_test.shape[1] == 100, "Data must have 100 runtime observations per instance."

    if num_samples_per_instance != 100:  # 100 is the full data
        print(f"Subsampling the training data to {num_samples_per_instance} samples per instance (without replacement).")
        assert 1 <= num_samples_per_instance <= 100, "num_samples_per_instance must be between 1 and 100"
        assert seed_samples_per_instance is not None, "seed_samples_per_instance must be provided"
        y_train = subsample_targets_per_instance(y_train, num_samples_per_instance, seed_samples_per_instance)
    
    # Flatten the whole training set
    X_train_flat = np.repeat(X_train, repeats=num_samples_per_instance, axis=0)
    y_train_flat = y_train.reshape(-1, 1)

    assert X_train_flat.shape[0] == y_train_flat.shape[0], "After flattening, X and y must have the same number of samples."

    if context_size is not None:
        assert seed_context_size is not None, "seed_context must be provided when context_size is specified."
        assert 1 <= context_size <= X_train_flat.shape[0], "invalid context_size value."
        if context_size < X_train_flat.shape[0]:
            print(f"Subsampling the training data to context size {context_size} using method '{subsample_method}'")
            X_train_flat, y_train_flat = subsample_flattened_data(X_train_flat, y_train_flat, context_size=context_size, seed=seed_context_size, subsample_method=subsample_method)
    
    # remove constant features
    X_train_flat, X_test = delete_constant_features(X_train_flat, X_test)

    if feature_drop_rate is not None and feature_drop_rate > 0.0:
        assert feature_agnostic is False, "can only set one at a time {feature_drop_rate or feature_agnostic}"
        assert seed_feature_drop_rate is not None, "seed_features must be provided when feature_drop_rate > 0.0"
        assert feature_drop_rate <= 1.0, "feature_drop_rate must be <=1.0"

        print(f"Sampling features with drop rate {feature_drop_rate}")

        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=feature_drop_rate, seed=seed_feature_drop_rate)

    if feature_agnostic:
        print("Feature agnostic mode on: dropping all features.")
        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=1.0, seed=-1)
    
    results_dict = None  # the dict to store after model fit&predict
    if model_name == 'distnet':
        from tabpfn_project.helper.distnet_lognormal import DistNetModel
        from tabpfn_project.helper.utils import calculate_all_distribution_metrics_distnet_logspace

        assert target_scale in ['max'], "DistNet only supports 'max' scaling currently."

        N = X_train_flat.shape[0]
        early_stopping_patience = 50
        E_final = None
        y_scale = None
        if early_stopping and N < 512:
            # 5-fold CV to find the optimal epoch, then retrain on the full training set.
            print(f"[DistNet] N={N} < 512: running 5-fold CV to determine E_final.")
            kf_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            best_epochs = []
            for cv_fold, (tr_idx, val_idx) in enumerate(kf_inner.split(X_train_flat), start=1):
                X_f_tr, X_f_val = X_train_flat[tr_idx], X_train_flat[val_idx]
                y_f_tr, y_f_val = y_train_flat[tr_idx], y_train_flat[val_idx]
                X_f_tr, X_f_val = preprocess_features(X_f_tr, X_f_val, scal="meanstd")
                y_f_tr, y_f_val, _ = max_scaling(y_f_tr, y_f_val)
                fold_model = DistNetModel(
                    n_input_features=X_f_tr.shape[1],
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    wc_time_limit=wc_time_limit,
                    save_path=None,
                    X_valid=X_f_val,
                    y_valid=y_f_val,
                    early_stopping=True,
                    early_stopping_patience=early_stopping_patience,
                    random_state=RANDOM_STATE,
                )
                fold_model.train(X_f_tr, y_f_tr)
                best_epochs.append(fold_model.best_epoch)
                print(f"  [CV {cv_fold}/5] best_epoch={fold_model.best_epoch}")
            E_final = int(round(np.mean(best_epochs)))
            print(f"[DistNet] best_epochs={best_epochs} -> E_final={E_final}")

            # Retrain on the full training set for exactly E_final epochs (no hold-out).
            X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")
            y_train_flat, y_test, y_scale = max_scaling(y_train_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=E_final,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )

        elif early_stopping and N >= 512:
            # Direct early stopping on an 80/20 hold-out split.
            print(f"[DistNet] N={N} >= 512: using direct early stopping on an 80/20 hold-out split.")
            X_train_flat, X_valid_flat, y_train_flat, y_valid_flat = train_test_split(
                X_train_flat, y_train_flat, test_size=0.2, random_state=RANDOM_STATE
            )
            X_train_flat, X_valid_flat, X_test = preprocess_features(X_train_flat, X_valid_flat, X_test, scal="meanstd")
            y_train_flat, y_valid_flat, y_test, y_scale = max_scaling(y_train_flat, y_valid_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=n_epochs,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                X_valid=X_valid_flat,
                y_valid=y_valid_flat,
                early_stopping=True,
                early_stopping_patience=early_stopping_patience,
                random_state=RANDOM_STATE,
            )

        else:
            # No early stopping: train on the full training set for n_epochs.
            X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")
            y_train_flat, y_test, y_scale = max_scaling(y_train_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=n_epochs,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )

        distnet_fit_time_start = time.perf_counter()
        model.train(X_train_flat, y_train_flat)
        distnet_fit_time = time.perf_counter() - distnet_fit_time_start

        # For direct early stopping, record the best epoch found.
        if early_stopping and N >= 512:
            E_final = model.best_epoch

        distnet_predict_time_start = time.perf_counter()
        y_pred = model.predict(X_test)
        distnet_predict_time = time.perf_counter() - distnet_predict_time_start

        assert y_scale is not None, "y_scale should not be None for DistNet when using max_scaling."

        device = torch.device('cpu')
        metrics_summary_distnet, instance_summary_distnet = calculate_all_distribution_metrics_distnet_logspace(y_test, y_pred, device=device, target_scale=target_scale, y_scaler=y_scale, N_grid_points=N_GRID_POINTS)

        results_dict = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,

            'seed_context_size': seed_context_size,
            'seed_feature_drop_rate': seed_feature_drop_rate,
            'seed_samples_per_instance': seed_samples_per_instance,

            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'n_features': X_train_flat.shape[1],
            'feature_agnostic': feature_agnostic,
            'oracle': oracle,

            'y_test_preds': y_pred,

            'do_hpo': do_hpo,
            'hpo_time': hpo_time,

            'result_metrics': {
                'metrics_summary': metrics_summary_distnet,
                'instance_summary': instance_summary_distnet,
            },

            'model_specific_info': {
                'model_config': model.model.state_dict(),
                'random_state': RANDOM_STATE,
                'early_stopping': early_stopping,
                'early_stopping_patience': early_stopping_patience,
                'E_final': E_final,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'wc_time_limit': wc_time_limit,
                'y_scale': y_scale,
                'fit_time': distnet_fit_time,
                'predict_time': distnet_predict_time,
            },

            'hpo_results': dict(),
        }

    elif model_name == 'tabpfn':
        from tabpfn_project.helper.tabpfn_helpers import batch_predict_tabpfn
        from tabpfn import TabPFNRegressor
        from tabpfn_project.helper.utils import calculate_all_distribution_metrics_tabpfn_logspace
        from tabpfn_project.helper.tabpfn_helpers import oracle_predict_tabpfn
        
        # Scale y (runtime) values
        assert target_scale in ['log', 'original'], "TabPFN currently only supports 'log' or 'original' scaling for the target variable."
        if target_scale == 'log':
            y_train_flat, y_test_scaled = log_scaling(y_train_flat, y_test)
        elif target_scale == "original":
            pass  # no scaling
    
        # initialize and train model
        device = torch.device('cuda' if (torch.cuda.is_available() and not use_cpu) else 'cpu')
        print(f"TabPFN using device: {device}")

        mem_time_stats = {
            "device": device,
            "fit": {
                "baseline_mb": None,
                "peak_mb": None,
                "spike_mb": None,
                "time_s": None,
            },
            "predict": {
                "baseline_mb": None,
                "peak_mb": None,
                "spike_mb": None,
                "time_s": None,
            },
        }

        model = TabPFNRegressor(device=device, random_state=RANDOM_STATE, ignore_pretraining_limits=True)
        
        if not oracle:
            with track_gpu_memory_and_time(device) as stats:
                model.fit(X_train_flat, y_train_flat.ravel())
            
            mem_time_stats["fit"]["baseline_mb"] = stats["baseline_mb"]
            mem_time_stats["fit"]["peak_mb"] = stats["peak_mb"]
            mem_time_stats["fit"]["spike_mb"] = stats["spike_mb"]
            mem_time_stats["fit"]["time_s"] = stats["time_s"]

            # evaluate model
            with track_gpu_memory_and_time(device) as stats:
                tabpfn_preds_full = batch_predict_tabpfn(model, X_test, validation_batch_size=val_batch_size)
            
        else:
            with track_gpu_memory_and_time(device) as stats:
                tabpfn_preds_full = oracle_predict_tabpfn(model, y_test_scaled)

        mem_time_stats["predict"]["baseline_mb"] = stats["baseline_mb"]
        mem_time_stats["predict"]["peak_mb"] = stats["peak_mb"]
        mem_time_stats["predict"]["spike_mb"] = stats["spike_mb"]
        mem_time_stats["predict"]["time_s"] = stats["time_s"]

        tabpfn_preds_full_fileName = (f"{model_name}_{scenario}_{fold}_{seed_context_size}_{seed_feature_drop_rate}_{seed_samples_per_instance}_{feature_drop_rate}_"
                         f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{oracle}_{feature_agnostic}{'cpu' if use_cpu else 'gpu'}_test_preds.pkl")

        os.makedirs(os.path.join(save_dir, "tabpfn_preds_full"), exist_ok=True)
        tabpfn_preds_full_path = os.path.join(save_dir, "tabpfn_preds_full", tabpfn_preds_full_fileName)
        with open(tabpfn_preds_full_path, 'wb') as f:
            pickle.dump(tabpfn_preds_full, f)

        metrics_summary_pfn, instance_summary_pfn = calculate_all_distribution_metrics_tabpfn_logspace(y_test, tabpfn_preds_full, device=device, target_scale=target_scale, N_grid_points=N_GRID_POINTS)

        results_dict = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,

            'seed_context_size': seed_context_size,
            'seed_feature_drop_rate': seed_feature_drop_rate,
            'seed_samples_per_instance': seed_samples_per_instance,

            'context_size': context_size,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'target_scale': target_scale,
            'n_features': X_train_flat.shape[1],
            'feature_agnostic': feature_agnostic,
            'oracle': oracle,

            'use_cpu': use_cpu,
            'save_dir': save_dir,

            'do_hpo': do_hpo,
            'hpo_time': hpo_time,
            'hpo_results': dict(),

            'y_test_preds': None,

            'result_metrics': {
                'metrics_summary': metrics_summary_pfn,
                'instance_summary': instance_summary_pfn,
            },

            'model_specific_info': {
                'random_state': RANDOM_STATE,
                'val_batch_size': val_batch_size,
                'mem_time_stats': mem_time_stats,
            },
        }
    
    elif model_name == 'random_forest':
        from tabpfn_project.helper.random_forest import RuntimePredictionRandomForest
        from tabpfn_project.helper.utils import calculate_all_distribution_metrics_randomForest_logspace

        X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")
        # log-scale the targets
        assert target_scale == 'log', "Random Forest currently only supports 'log' scaling for the target variable."
        y_train_flat_scaled = log_scaling(y_train_flat)[0]
        
        rf_model = RuntimePredictionRandomForest(random_state=RANDOM_STATE)
        start_time_rf = time.perf_counter()
        rf_model.fit(X_train_flat, y_train_flat_scaled)
        end_time_rf_fit = time.perf_counter()

        rf_means, rf_variances = rf_model.predict(X_test)
        end_time_rf_predict = time.perf_counter()

        device = torch.device("cuda" if (torch.cuda.is_available() and not use_cpu) else "cpu")

        metrics_summary_rf, instance_summary_rf = calculate_all_distribution_metrics_randomForest_logspace(
            y_test_orig=y_test,
            preds=(rf_means.ravel(), rf_variances.ravel()),
            device=device,
            N_grid_points=N_GRID_POINTS
        )

        results_dict = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,

            'seed_context_size': seed_context_size,
            'seed_feature_drop_rate': seed_feature_drop_rate,
            'seed_samples_per_instance': seed_samples_per_instance,

            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'n_features': X_train_flat.shape[1],
            'feature_agnostic': feature_agnostic,
            'oracle': oracle,

            'y_test_preds': [rf_means, rf_variances],

            'do_hpo': do_hpo,
            'hpo_time': hpo_time,

            'result_metrics': {
                'metrics_summary': metrics_summary_rf,
                'instance_summary': instance_summary_rf,
            },

            'model_specific_info': {
                'random_state': RANDOM_STATE,
                'fit_time': (end_time_rf_fit - start_time_rf),
                'predict_time': (end_time_rf_predict - end_time_rf_fit),
            },

            'hpo_results': dict(),
        }

        print(metrics_summary_rf)

    elif model_name == 'dist_lognormal':
        from tabpfn_project.helper.utils import calculate_all_distribution_metrics_logNormalDist_logspace
        # a feature-agnostic lognormal model fitted to the targets
        del X_train_flat, X_test  # no need for X

        device = torch.device('cuda' if (torch.cuda.is_available() and not use_cpu) else 'cpu')

        if oracle:
            # Oracle calculates ground-truth mu and sigma per instance
            y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
            log_y_test = torch.log(y_test)
            mu = torch.mean(log_y_test, dim=1, keepdim=True)
            sigma = torch.std(log_y_test, dim=1, keepdim=True)

        else:
            y_train_flat = torch.as_tensor(y_train_flat, dtype=torch.float32, device=device)
            log_y = torch.log(y_train_flat) 
            mu = torch.mean(log_y)
            sigma = torch.std(log_y)
            
        dist = torch.distributions.LogNormal(loc=mu, scale=sigma)
        metrics_summary_lognormal, instance_summary_lognormal = (
            calculate_all_distribution_metrics_logNormalDist_logspace(y_test, dist, device=device, N_grid_points=N_GRID_POINTS)
        )

        results_dict = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,

            'seed_context_size': seed_context_size,
            'seed_feature_drop_rate': seed_feature_drop_rate,
            'seed_samples_per_instance': seed_samples_per_instance,

            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'feature_agnostic': feature_agnostic,
            'oracle': oracle,

            'y_test_preds': [mu, sigma],

            'do_hpo': do_hpo,
            'hpo_time': hpo_time,

            'result_metrics': {
                'metrics_summary': metrics_summary_lognormal,
                'instance_summary': instance_summary_lognormal,
            },

            'model_specific_info': {
            },

            'hpo_results': dict(),
        }

        print(metrics_summary_lognormal)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    assert results_dict is not None, "results_dict is NONE"
    results_file_name = (f"{model_name}_{scenario}_{fold}_{seed_context_size}_{seed_feature_drop_rate}_{seed_samples_per_instance}_{feature_drop_rate}_"
                         f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{early_stopping}_{'cpu' if use_cpu else 'gpu'}.pkl")

    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    results_save_path = os.path.join(save_dir, "metadata", results_file_name)

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {results_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate one model on one scenario/fold."
    )
    parser.add_argument("--scenario", type=str, required=True, choices=DISTNET_SCENARIOS,
                        help="DistNet scenario.")
    parser.add_argument("--model", type=str, required=True, choices=MODELS,
                        help="Model to run.")
    parser.add_argument("--feature_agnostic", action="store_true",
                        help="Whether the ML model should ignore the instance features.")
    parser.add_argument("--oracle", action="store_true",
                        help="Whether to fit on the test data (suitable for oracle baselies).")
    parser.add_argument("--fold", type=int, required=True, choices=range(10),
                        help="CV fold index.")
    parser.add_argument("--num_samples_per_instance", type=int, default=100, choices=range(1, 101),
                        help="Targets per train instance.")
    parser.add_argument("--val_batch_size", type=int, default=1000,
                        help="TabPFN prediction batch size.")
    parser.add_argument("--target_scale", type=str, required=False, default=None, choices=TARGET_SCALES,
                        help="Target scaling.")
    parser.add_argument("--subsample_method", type=str, default='flatten-random', choices=SUBSAMPLE_METHOD_CHOICES,
                        help="Context subsampling method.")
    parser.add_argument("--context_size", type=int, default=None,
                        help="Flattened train set size.")
    parser.add_argument("--feature_drop_rate", type=float, default=None,
                        help="Fraction of features to drop.")
    parser.add_argument("--seed_context_size", type=int, default=None,
                        help="Seed for context sampling.")
    parser.add_argument("--seed_feature_drop_rate", type=int, default=None,
                        help="Seed for feature sampling.")
    parser.add_argument("--seed_samples_per_instance", type=int, default=None,
                        help="Seed for target subsampling.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output subdirectory under RESULTS_DIR.")
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help="DistNet max epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="DistNet batch size.")
    parser.add_argument("--wc_time_limit", type=int, default=60 * 59,
                        help="DistNet wall-clock limit (s).")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable DistNet early stopping.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU.")
    parser.add_argument("--do_hpo", action="store_true",
                        help="Enable HPO.")
    parser.add_argument("--hpo_time", type=int, default=None,
                        help="HPO wall-clock limit (s).")

    args = parser.parse_args()
    
    print(f"🧪 Starting the experiment with the following configuration:\n{args}")
    start = time.perf_counter()
    train_test_model(
        model_name=args.model,
        scenario=args.scenario,
        fold=args.fold,
        save_dir=args.save_dir,
        num_samples_per_instance=args.num_samples_per_instance,
        context_size=args.context_size,
        use_cpu=args.use_cpu,
        target_scale=args.target_scale,
        subsample_method=args.subsample_method,
        early_stopping=args.early_stopping,
        seed_context_size=args.seed_context_size,
        seed_feature_drop_rate=args.seed_feature_drop_rate,
        feature_drop_rate=args.feature_drop_rate,
        val_batch_size=args.val_batch_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        wc_time_limit=args.wc_time_limit,
        seed_samples_per_instance=args.seed_samples_per_instance,
        do_hpo=args.do_hpo,
        hpo_time=args.hpo_time,
        feature_agnostic=args.feature_agnostic,
        oracle=args.oracle,
    )
    end = time.perf_counter()
    print(f"✅ Experiment completed in {end - start:.2f} seconds.")