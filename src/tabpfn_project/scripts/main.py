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
from tabpfn_project.helper.scalers import max_scaling, log_scaling, z_score_scaling

from tabpfn_project.helper.preprocess import (
    delete_constant_features,
    preprocess_features,
)

# import globals
from tabpfn_project.globals import EPS, N_GRID_POINTS, RANDOM_STATE
from sklearn.model_selection import KFold, train_test_split
from tabpfn_project.helper.utils import subsample_features, subsample_targets_per_instance, subsample_flattened_data
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
    seed_context,
    seed_features,
    feature_drop_rate,
    val_batch_size,
    seed_samples_per_instance,
    n_epochs,
    batch_size,
    wc_time_limit,
    do_hpo,
    hpo_time,
):
    assert 0 <= fold <= 9, "Fold must be between 0 and 9"

    if do_hpo:
        assert hpo_time is not None and hpo_time > 0, "hpo_time must be a positive integer when do_hpo is True"
    
    # Create save directory if it doesn't exist
    clean_name = save_dir.lstrip('/\\')
    save_dir = RESULTS_DIR / clean_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get scenario configuration and data

    # Load data
    X_train, X_test, y_train, y_test = load_distnet_data(DISTNET_DATA_DIR, scenario, fold)

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
        assert seed_context is not None, "seed_context must be provided when context_size is specified."
        assert 1 <= context_size <= X_train_flat.shape[0], "invalid context_size value."
        if context_size < X_train_flat.shape[0]:
            print(f"Subsampling the training data to context size {context_size} using method '{subsample_method}'")
            X_train_flat, y_train_flat = subsample_flattened_data(X_train_flat, y_train_flat, context_size=context_size, seed=seed_context, subsample_method=subsample_method)
    
    # remove constant features
    X_train_flat, X_test = delete_constant_features(X_train_flat, X_test)

    if feature_drop_rate is not None:
        assert seed_features is not None, "seed_features must be provided when feature_drop_rate > 0.0"
        assert 0.0 < feature_drop_rate <= 1.0, "feature_drop_rate must be in (0.0, 1.0]"
        print(f"Sampling features with drop rate {feature_drop_rate}")
        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=feature_drop_rate, seed=seed_features)
    
    results_dict = None  # the ultimate dict to store after model fit&predict
    if model_name == 'distnet':
        from tabpfn_project.helper.distnet_lognormal import DistNetModel

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
        results_dict = {
            'model_name': 'distnet',
            'model_config': model.model.state_dict(),
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'E_final': E_final,
            'save_dir': save_dir,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'wc_time_limit': wc_time_limit,
            'test_preds': y_pred,
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'y_scale': y_scale,
            'result_metrics': {
                'fit_time': distnet_fit_time,
                'predict_time': distnet_predict_time,
            }
        }

    elif model_name == 'tabpfn':
        from tabpfn_project.helper.pfn_helpers import batch_predict_tabpfn
        from tabpfn import TabPFNRegressor
        
        # Scale y (runtime) values
        args = None
        if target_scale == 'max':
            y_train_flat, y_test, _ = max_scaling(y_train_flat, y_test)
        elif target_scale == 'log':
            y_train_flat, y_test = log_scaling(y_train_flat, y_test)
        elif target_scale == "z-score":
            y_train_flat, y_test, mean, std = z_score_scaling(y_train_flat, y_test)
            args = [mean, std]
        elif target_scale == "none":
            pass  # no scaling
        
        # no preprocessing of features for TabPFN
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

        with track_gpu_memory_and_time(device) as stats:
            model.fit(X_train_flat, y_train_flat.ravel())
        
        mem_time_stats["fit"]["baseline_mb"] = stats["baseline_mb"]
        mem_time_stats["fit"]["peak_mb"] = stats["peak_mb"]
        mem_time_stats["fit"]["spike_mb"] = stats["spike_mb"]
        mem_time_stats["fit"]["time_s"] = stats["time_s"]

        # evaluate model
        with track_gpu_memory_and_time(device) as stats:
            tabpfn_preds_full = batch_predict_tabpfn(model, X_test, validation_batch_size=val_batch_size)
        
        mem_time_stats["predict"]["baseline_mb"] = stats["baseline_mb"]
        mem_time_stats["predict"]["peak_mb"] = stats["peak_mb"]
        mem_time_stats["predict"]["spike_mb"] = stats["spike_mb"]
        mem_time_stats["predict"]["time_s"] = stats["time_s"]

        tabpfn_preds_full_fileName = (f"{model_name}_{scenario}_{fold}_{seed_context}_{seed_features}_{seed_samples_per_instance}_{feature_drop_rate}_"
                         f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{'cpu' if use_cpu else 'gpu'}_test_preds.pkl")

        os.makedirs(os.path.join(save_dir, "tabpfn_preds_full"), exist_ok=True)
        tabpfn_preds_full_path = os.path.join(save_dir, "tabpfn_preds_full", tabpfn_preds_full_fileName)
        with open(tabpfn_preds_full_path, 'wb') as f:
            pickle.dump(tabpfn_preds_full, f)

        results_dict = {
            'model_name': 'tabpfn',
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'val_batch_size': val_batch_size,
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'scaler_args': args,
            'result_metrics': mem_time_stats,
        }

    elif model_name == 'ngboost':
        assert target_scale == 'max', "NGBoost currently only supports 'max' scaling for the target variable."
        from tabpfn_project.helper.tree_based_models import train_evaluate_ngboost

        X_train_flat, X_test = preprocess_features(X_train_flat, X_test)
        if target_scale == 'max':
            y_train_flat_scaled, y_scale = max_scaling(y_train_flat)
        y_preds_ngboost, best_params_ngboost, fit_time_ngboost, predict_time_ngboost = train_evaluate_ngboost(X_train_flat, y_train_flat_scaled.ravel(), X_test, do_hpo, hpo_time)
        
        results_dict = {
            'model_name': 'ngboost',
            'best_params': best_params_ngboost,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'test_preds': y_preds_ngboost,
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'y_scale': y_scale,
            'result_metrics': {
                'fit_time': fit_time_ngboost,
                'predict_time': predict_time_ngboost,
            }
        }

    elif model_name == 'qrf':
        assert target_scale == 'log', "Quantile Regression Forest currently only supports 'log' scaling for the target variable."
        from tabpfn_project.helper.tree_based_models import train_evaluate_qrf

        X_train_flat, X_test = preprocess_features(X_train_flat, X_test)
        if target_scale == 'log':
            y_train_flat_scaled = log_scaling(y_train_flat)[0]

        y_preds_quantiles_qrf, best_params_qrf, fit_time_qrf, predict_time_qrf = train_evaluate_qrf(X_train_flat, y_train_flat_scaled.ravel(), X_test, do_hpo, hpo_time)

        results_dict = {
            'model_name': 'qrf',
            'best_params': best_params_qrf,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'test_preds': y_preds_quantiles_qrf,
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'result_metrics': {
                'fit_time': fit_time_qrf,
                'predict_time': predict_time_qrf,
            }
        }

    elif model_name == 'naive_baseline':
        assert target_scale in ['log'], "Baseline currently only supports 'log' scaling for the target variable."
        from tabpfn_project.helper.naive_baseline import calculate_all_distribution_metrics_baseline, get_marginal_empirical_predictor

        if target_scale == 'log':
            y_train_flat_scaled = log_scaling(y_train_flat)[0]

        start_time_baseline = time.perf_counter()
        cdf_object, pdf_object = get_marginal_empirical_predictor(y_train_flat_scaled.ravel())
        metrics_summary_baseline, instance_summary_baseline = calculate_all_distribution_metrics_baseline(y_test, cdf_object, pdf_object, N_grid_points=N_GRID_POINTS)
        end_time_baseline_ = time.perf_counter()

        results_dict = {
            'model_name': 'baseline',
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'test_preds': [metrics_summary_baseline, instance_summary_baseline],
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'N_grid_points': N_GRID_POINTS,
            'result_metrics': {
                'total_time': end_time_baseline_ - start_time_baseline,
            }
        }
    
    elif model_name == 'random_forest':
        from smac import HyperparameterOptimizationFacade, Scenario
        from ConfigSpace import Configuration, ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
        from tabpfn_project.helper.random_forest import HutterRandomForest
        from tabpfn_project.helper.random_forest import calculate_all_distribution_metrics_rf_baseline
        
        # log-scale the targets
        assert target_scale == 'log', "Random Forest baseline currently only supports 'log' scaling for the target variable."
        y_train_flat_scaled = log_scaling(y_train_flat)[0]

        if do_hpo:
            print(f"Starting SMAC3 HPO for rf_baseline with a walltime limit of {hpo_time} seconds...")
            hpo_start_time = time.perf_counter()
            
            # 1. Define the Configuration Space for HPO
            cs = ConfigurationSpace()
            cs.add([
                UniformIntegerHyperparameter("n_trees", lower=10, upper=200, default_value=10),
                CategoricalHyperparameter("min_samples_split", choices=[5, 10], default_value=5),
                UniformFloatHyperparameter("ratio_features", lower=0.1, upper=1.0, default_value=0.5),
            ])

            # 2. Define the target function to evaluate configurations using 3-Fold CV NLL
            def evaluate_rf(config: Configuration, seed: int = RANDOM_STATE) -> float:
                kf = KFold(n_splits=3, shuffle=True, random_state=seed)
                nll_scores =[]
                
                for train_idx, val_idx in kf.split(X_train_flat):
                    X_tr, X_val = X_train_flat[train_idx], X_train_flat[val_idx]
                    y_tr, y_val = y_train_flat_scaled[train_idx], y_train_flat_scaled[val_idx]

                    # Instantiate model with the suggested configuration
                    model = HutterRandomForest(
                        n_trees=int(config["n_trees"]),
                        min_samples_split=int(config["min_samples_split"]),
                        ratio_features=float(config["ratio_features"]),
                    )
                    
                    # Fit and predict
                    model.fit(X_tr, y_tr.ravel())
                    means, variances = model.predict(X_val)

                    # Calculate Negative Log-Likelihood (NLL)
                    # For a normal distribution N(mu, sigma^2): 
                    # NLL = 0.5 * log(2 * pi * sigma^2) + (y - mu)^2 / (2 * sigma^2)
                    nll = 0.5 * np.log(2 * np.pi * variances) + ((y_val.ravel() - means) ** 2) / (2 * variances)
                    nll_scores.append(np.mean(nll))
                
                return float(np.mean(nll_scores))

            # 3. Set up the SMAC3 Scenario
            smac_scenario = Scenario(
                configspace=cs,
                deterministic=True,
                n_trials=100000, # Large bound, mostly constrained by walltime_limit
                walltime_limit=hpo_time,
                seed=RANDOM_STATE,
                name=f"rf_hpo_{scenario}_f{fold}_{int(time.time())}"
            )

            # 4. Run optimization
            smac = HyperparameterOptimizationFacade(scenario=smac_scenario, target_function=evaluate_rf)
            incumbent = smac.optimize()
            
            hpo_end_time = time.perf_counter()
            hpo_duration = hpo_end_time - hpo_start_time
            print(f"SMAC3 HPO completed in {hpo_duration:.2f} seconds.")
            print(f"Best configuration found: {incumbent}")
            
            # Use the best hyperparameters found
            best_params_rf = dict(incumbent)
            rf_model = HutterRandomForest(
                n_trees=int(best_params_rf["n_trees"]),
                min_samples_split=int(best_params_rf["min_samples_split"]),
                ratio_features=float(best_params_rf["ratio_features"]),
            )
        else:
            hpo_duration = 0.0
            best_params_rf = {
                "n_trees": 10,
                "min_samples_split": 5,
                "ratio_features": 0.5,
            }

            rf_model = HutterRandomForest(n_trees=10, min_samples_split=5, ratio_features=0.5)

        start_time_rf = time.perf_counter()
        rf_model.fit(X_train_flat, y_train_flat_scaled.ravel())
        end_time_rf_fit = time.perf_counter()

        rf_means, rf_variances = rf_model.predict(X_test)
        end_time_rf_predict = time.perf_counter()

        device = torch.device("cuda" if (torch.cuda.is_available() and not use_cpu) else "cpu")

        metrics_summary, instance_summary = calculate_all_distribution_metrics_rf_baseline(
            y_test_orig=y_test,
            preds=(rf_means, rf_variances),
            device=device,
            N_grid_points=N_GRID_POINTS
        )

        results_dict = {
            'model_name': 'rf_baseline',
            'best_params': best_params_rf,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'test_preds': [rf_means, rf_variances],
            'random_state': RANDOM_STATE,
            'n_features': X_train_flat.shape[1],
            'N_grid_points': N_GRID_POINTS,
            'result_metrics': {
                'fit_time': end_time_rf_fit - start_time_rf,
                'predict_time': end_time_rf_predict - end_time_rf_fit,
                'hpo_time': hpo_duration,
                'metrics_summary': metrics_summary,
                'instance_summary': instance_summary,
            }
        }

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    assert results_dict is not None, "results_dict is NONE"
    results_file_name = (f"{model_name}_{scenario}_{fold}_{seed_context}_{seed_features}_{seed_samples_per_instance}_{feature_drop_rate}_"
                         f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{early_stopping}_{'cpu' if use_cpu else 'gpu'}.pkl")

    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    results_save_path = os.path.join(save_dir, "metadata", results_file_name)

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {results_save_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model for Given Scenario (DistNet or TabPFN)')

    parser.add_argument('--scenario', type=str, required=True, help='Scenario name (e.g., lpg-zeno, clasp_factoring)')
    parser.add_argument('--model', type=str, required=True, help='model type to train (distnet, tabpfn, ngboost, qrf, naive_baseline, random_forest)')
    parser.add_argument('--fold', type=int, required=True, help='Cross-validation fold index (0-9)')
    parser.add_argument('--num_samples_per_instance', type=int, default=100, help='Number of training samples per instance (1-100)')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='Validation batch size for TabPFN (default: 1000)')
    parser.add_argument('--target_scale', type=str, required=True, help='Target scaling method (log, z-score, max, none)')
    parser.add_argument('--subsample_method', type=str, default=None, help='Sampling method (flatten-random)')
    parser.add_argument('--context_size', type=int, default=None, help='Number of flattened training samples (default: None, use all)')
    parser.add_argument('--feature_drop_rate', type=float, default=None, help='Feature drop rate (default: None, use all features)')
    parser.add_argument('--seed_context', type=int, default=None, help='Random seed for sampling context')
    parser.add_argument('--seed_features', type=int, default=None, help='Random seed for sampling features')
    parser.add_argument('--seed_samples_per_instance', type=int, default=None, help='Random seed for sampling samples per instance')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Max training epochs for DistNet (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size for DistNet training (default: 16)')
    parser.add_argument('--wc_time_limit', type=int, default=60*59, help='Wall-clock time limit in seconds per DistNet training run (default: 3540)')
    parser.add_argument('--early_stopping', action='store_true', help='Enable adaptive epoch search (find_optimal_epochs) for DistNet')
    parser.add_argument('--use_cpu', action='store_true', help='add this flag to use CPU instead of GPU (if applicable)')
    parser.add_argument('--do_hpo', action='store_true', help='add this flag to perform hyperparameter optimization')
    parser.add_argument('--hpo_time', type=int, default=None, help='Wall-clock time limit in seconds for hyperparameter optimization')



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
        seed_context=args.seed_context,
        seed_features=args.seed_features,
        feature_drop_rate=args.feature_drop_rate,
        val_batch_size=args.val_batch_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        wc_time_limit=args.wc_time_limit,
        seed_samples_per_instance=args.seed_samples_per_instance,
        do_hpo=args.do_hpo,
        hpo_time=args.hpo_time,
    )
    end = time.perf_counter()
    print(f"✅ Experiment completed in {end - start:.2f} seconds.")